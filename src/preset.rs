//! 预设检索：库预先配好的几种搜索方法。
//!
//! 每次返回固定的三个字段——记忆、图谱、笔记——各自独立排序，不混在一起。
//! 图谱字段内部按流程的阶段分块：种子实体一块、命中的关系一块、铺开的关系与事件一块。
//! 预设里的阈值（字符数、种子实体个数）都带默认值，调用方实例化时可以覆盖。

use rusqlite::{params_from_iter, types::Value as SqlValue, Connection};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

use crate::graph::{Entity, Event, Relation};
use crate::search::{SearchHit, SearchRequest, SearchResult};
use crate::storage::{self, KnowledgeBase};
use crate::types::{ReadFilter, RecordKind, SearchDiagnostics};
use crate::{text, Error, Result};

/// 库预先配好的搜索方法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchPreset { Memory, Graph, Notes, Rag, Broad }

impl SearchPreset {
    pub const ALL: [Self; 5] = [Self::Memory, Self::Graph, Self::Notes, Self::Rag, Self::Broad];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory", Self::Graph => "graph", Self::Notes => "notes",
            Self::Rag => "rag", Self::Broad => "broad",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        Self::ALL.into_iter().find(|preset| preset.as_str() == value)
            .ok_or_else(|| Error::Validation(format!("preset must be memory, graph, notes, rag or broad, got {value}")))
    }

    /// 这次要不要出记忆那一路。
    pub fn uses_memory(self) -> bool { matches!(self, Self::Memory | Self::Rag | Self::Broad) }
    /// 这次要不要出图谱那一路。
    pub fn uses_graph(self) -> bool { matches!(self, Self::Graph | Self::Rag | Self::Broad) }
    /// 这次要不要出笔记那一路。
    pub fn uses_notes(self) -> bool { matches!(self, Self::Notes | Self::Broad) }
}

/// 预设的阈值。输出量按字符数封顶（重排模型按字符数算，不看条数），
/// 种子实体是中间量，仍按个数。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PresetBudget {
    /// 记忆那一路的字符数上限。
    pub memory_chars: usize,
    /// 笔记那一路的字符数上限。
    pub notes_chars: usize,
    /// 图谱那一路取前几个实体当种子。
    pub seed_entities: usize,
    /// 图谱第二步命中的关系的字符数上限。
    pub graph_relations_chars: usize,
    /// 图谱第三步铺开的关系与事件的字符数上限（关系与事件共用这一份）。
    pub graph_context_chars: usize,
}

impl Default for PresetBudget {
    fn default() -> Self {
        Self { memory_chars: 2000, notes_chars: 3000, seed_entities: 4,
            graph_relations_chars: 1000, graph_context_chars: 2000 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PresetRequest {
    pub preset: SearchPreset,
    pub query: String,
    pub filter: ReadFilter,
    /// 走向量路时用哪条向量空间；不给就只走全文。
    pub embed_space: Option<String>,
    /// 本次是否走全文路。
    pub text: bool,
    /// 本次是否走向量路（需要 `embed_space`）。
    pub vector: bool,
    /// 本次是否重排；未注册重排回调时被忽略。
    pub rerank: bool,
    /// 覆盖默认阈值。
    pub budget: PresetBudget,
    /// 每条路按字符数封顶之前，先取多少条候选。
    pub candidate_limit: usize,
}

impl Default for PresetRequest {
    fn default() -> Self {
        Self { preset: SearchPreset::Rag, query: String::new(), filter: ReadFilter::default(),
            embed_space: None, text: true, vector: true, rerank: true,
            budget: PresetBudget::default(), candidate_limit: 64 }
    }
}

/// 图谱那一路的产物，按流程阶段分块。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphSection {
    /// 第一步搜出来的种子实体（按分排，带别名）。
    pub entities: Vec<Entity>,
    /// 第二步：种子实体各自到自己的关系里敲查询词，命中的关系（按相关度排）。
    pub relations: Vec<Relation>,
    /// 第三步：这批实体两两之间的关系，不筛。
    pub context_relations: Vec<Relation>,
    /// 第三步：这批实体两两之间的事件（参与者至少两个落在集合内），不筛。
    pub context_events: Vec<Event>,
}

/// 一次预设检索的结果。三个字段各自独立排序，没走的那一路是空的。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetResult {
    pub preset: SearchPreset,
    pub memories: Vec<SearchHit>,
    pub graph: GraphSection,
    pub notes: Vec<SearchHit>,
    pub revision: i64,
    pub indexed_revision: i64,
    pub diagnostics: SearchDiagnostics,
}

impl KnowledgeBase {
    /// 按预设检索。三个字段各自独立排序，互不挤占。
    pub fn search_preset(&self, request: &PresetRequest) -> Result<PresetResult> {
        let query = request.query.trim();
        if query.is_empty() { return Err(Error::Validation("a text query is required".into())); }
        storage::validate_filter(&request.filter)?;
        if request.budget.seed_entities == 0 { return Err(Error::Validation("seed_entities must be at least 1".into())); }
        if request.candidate_limit == 0 { return Err(Error::Validation("candidate_limit must be at least 1".into())); }

        let mut diagnostics = SearchDiagnostics::default();
        let memories = if request.preset.uses_memory() {
            let result = self.preset_hits(query, request, &[RecordKind::Memory])?;
            merge_diagnostics(&mut diagnostics, &result.diagnostics);
            self.truncate_hits_by_chars(result.hits, request.budget.memory_chars)?
        } else { Vec::new() };
        let notes = if request.preset.uses_notes() {
            let result = self.preset_hits(query, request, &[RecordKind::Chunk])?;
            merge_diagnostics(&mut diagnostics, &result.diagnostics);
            self.truncate_hits_by_chars(result.hits, request.budget.notes_chars)?
        } else { Vec::new() };
        let graph = if request.preset.uses_graph() {
            self.preset_graph(query, request, &mut diagnostics)?
        } else { GraphSection::default() };

        let (revision, indexed_revision) = {
            let state = self.read()?;
            (storage::current_revision(state.conn())?, storage::meta(state.conn(), "indexed_revision")?)
        };
        Ok(PresetResult { preset: request.preset, memories, graph, notes, revision, indexed_revision, diagnostics })
    }

    /// 单一路的候选：复用现有的全文 + 向量融合，再按字符数截断。
    fn preset_hits(&self, query: &str, request: &PresetRequest, kinds: &[RecordKind]) -> Result<SearchResult> {
        let inner = SearchRequest {
            query: query.to_string(), filter: request.filter.clone(), kinds: kinds.to_vec(),
            limit: request.candidate_limit, embed_space: request.embed_space.clone(),
            text: request.text, vector: request.vector, rerank: request.rerank,
            ..Default::default()
        };
        self.search(&inner)
    }

    /// 按字符数封顶。长度取索引里的纯正文——切片正文只存在索引里；
    /// 取不到正文的条目按 0 计。第一条无论多长都留下，避免「预算略小就整条路空掉」。
    fn truncate_hits_by_chars(&self, hits: Vec<SearchHit>, chars: usize) -> Result<Vec<SearchHit>> {
        if chars == 0 || hits.is_empty() { return Ok(Vec::new()); }
        let ids: Vec<i64> = hits.iter().map(|hit| hit.key.id).collect();
        // 索引取不到正文不该让整次检索失败：这一路退化成只按条数，不打断别的路。
        let bodies = match self.index() { Ok(index) => index.bodies(&ids).unwrap_or_default(), Err(_) => BTreeMap::new() };
        let mut used = 0usize;
        let mut out = Vec::new();
        for hit in hits {
            let len = bodies.get(&hit.key.id).map(|body| body.chars().count()).unwrap_or(0);
            if !out.is_empty() && used + len > chars { break; }
            used += len;
            out.push(hit);
        }
        Ok(out)
    }

    /// 图谱那一路：搜实体 → 种子各自敲自己的关系 → 两两之间铺开关系与事件。
    fn preset_graph(&self, query: &str, request: &PresetRequest, diagnostics: &mut SearchDiagnostics) -> Result<GraphSection> {
        // 第一步：查询词搜实体，取前几个当种子。
        let entity_result = self.preset_hits(query, request, &[RecordKind::Entity])?;
        merge_diagnostics(diagnostics, &entity_result.diagnostics);
        let seeds: Vec<i64> = entity_result.hits.iter().take(request.budget.seed_entities).map(|hit| hit.key.id).collect();
        if seeds.is_empty() { return Ok(GraphSection::default()); }

        let state = self.read()?;
        let conn = state.conn();
        // 种子实体连别名一起给出：实体名与别名都在它的正文里，取回的是记录本体。
        let entity_filter = ReadFilter { tags: vec![], ..request.filter.clone() };
        let loaded: BTreeMap<i64, Entity> = storage::load_many(conn, &seeds, &entity_filter)?;
        let entities: Vec<Entity> = seeds.iter().filter_map(|id| loaded.get(id).cloned()).collect();

        // 第二步：种子实体各自到「以它为端点」的关系里敲查询词，命中的留下并按相关度排。
        let candidate_ids = incident_relations(conn, &seeds, &request.filter)?;
        let candidates = storage::record_values(conn, &candidate_ids)?;
        // 与全文路同一套扩散：关系正文里写「丈夫」时，查询「艾莉儿的老公」也能敲中。
        let graph_query = crate::graph::match_predicate_synonyms(conn, &request.filter.namespace, query)?;
        let ranked = rank_relations_by_query(&candidates, &graph_query);
        let hit_ids = truncate_values_by_chars(&candidates, &ranked, RecordKind::Relation, request.budget.graph_relations_chars);
        let relations: Vec<Relation> = decode_all(&candidates, &hit_ids)?;

        // 第三步：实体集合 = 种子 + 命中关系的另一端；两两之间的关系与事件全部保留，不筛。
        let mut members: BTreeSet<i64> = seeds.iter().copied().collect();
        for relation in &relations { members.insert(relation.subject_id); members.insert(relation.object_id); }
        let member_ids: Vec<i64> = members.into_iter().collect();
        let hit_set: BTreeSet<i64> = hit_ids.iter().copied().collect();

        let between_ids: Vec<i64> = relations_between(conn, &member_ids, &request.filter)?
            .into_iter().filter(|id| !hit_set.contains(id)).collect();
        let between_values = storage::record_values(conn, &between_ids)?;
        let between_order: Vec<i64> = between_ids.iter().copied().filter(|id| between_values.contains_key(id)).collect();
        let kept_relations = truncate_values_by_chars(&between_values, &between_order, RecordKind::Relation, request.budget.graph_context_chars);
        let used = chars_of(&between_values, &kept_relations, RecordKind::Relation);
        let context_relations: Vec<Relation> = decode_all(&between_values, &kept_relations)?;

        let event_ids = events_between(conn, &member_ids, &request.filter)?;
        let event_values = storage::record_values(conn, &event_ids)?;
        let event_order: Vec<i64> = event_ids.iter().copied().filter(|id| event_values.contains_key(id)).collect();
        let remaining = request.budget.graph_context_chars.saturating_sub(used);
        let kept_events = truncate_values_by_chars(&event_values, &event_order, RecordKind::Event, remaining);
        let context_events: Vec<Event> = decode_all(&event_values, &kept_events)?;

        Ok(GraphSection { entities, relations, context_relations, context_events })
    }
}

/// 把子路的诊断并进整次检索的诊断：开关取或，候选数累加，降级去重。
fn merge_diagnostics(target: &mut SearchDiagnostics, source: &SearchDiagnostics) {
    target.text_used |= source.text_used;
    target.vector_used |= source.vector_used;
    target.reranked |= source.reranked;
    target.rerank_candidates += source.rerank_candidates;
    target.rerank_truncated += source.rerank_truncated;
    for degrade in &source.degraded {
        if !target.degraded.contains(degrade) { target.degraded.push(*degrade); }
    }
}

/// 查询词元在这段文本里的权重：命中的二字及以上词元算两份，单字算一份。
fn token_weight(body: &str, tokens: &[String]) -> usize {
    let present: BTreeSet<String> = text::tokenize(body).into_iter().collect();
    tokens.iter().map(|token| {
        if !present.contains(token) { 0 } else if token.chars().count() > 1 { 2 } else { 1 }
    }).sum()
}

/// 第二步的排序：命中权重高的在前，同权重按记录 id 升序，保证结果稳定。
fn rank_relations_by_query(values: &BTreeMap<i64, Value>, query: &str) -> Vec<i64> {
    let tokens = text::query_terms(query, true);
    if tokens.is_empty() { return values.keys().copied().collect(); }
    let mut scored: Vec<(usize, i64)> = values.iter()
        .map(|(id, payload)| (token_weight(&storage::record_text(RecordKind::Relation, payload), &tokens), *id))
        .filter(|(weight, _)| *weight > 0)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, id)| id).collect()
}

/// 按字符数封顶，顺序不变；第一条无论多长都留下。
fn truncate_values_by_chars(values: &BTreeMap<i64, Value>, order: &[i64], kind: RecordKind, chars: usize) -> Vec<i64> {
    if chars == 0 { return Vec::new(); }
    let mut used = 0usize;
    let mut kept = Vec::new();
    for id in order {
        let Some(payload) = values.get(id) else { continue };
        let len = storage::record_text(kind, payload).chars().count();
        if !kept.is_empty() && used + len > chars { break; }
        used += len;
        kept.push(*id);
    }
    kept
}

fn chars_of(values: &BTreeMap<i64, Value>, ids: &[i64], kind: RecordKind) -> usize {
    ids.iter().filter_map(|id| values.get(id)).map(|payload| storage::record_text(kind, payload).chars().count()).sum()
}

fn decode_all<T: DeserializeOwned>(values: &BTreeMap<i64, Value>, ids: &[i64]) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for id in ids {
        if let Some(payload) = values.get(id) { out.push(serde_json::from_value(payload.clone())?); }
    }
    Ok(out)
}

/// 以这批实体为端点的关系 id（它是主语或宾语都算）。
fn incident_relations(conn: &Connection, entity_ids: &[i64], filter: &ReadFilter) -> Result<Vec<i64>> {
    let (condition, values) = storage::filter_sql(filter, &[RecordKind::Relation], false)?;
    let placeholders = vec!["?"; entity_ids.len()].join(",");
    let sql = format!("SELECT rl.record_id FROM relations rl JOIN records r ON r.id=rl.record_id \
        WHERE (rl.subject_id IN ({placeholders}) OR rl.object_id IN ({placeholders})) AND {condition} ORDER BY rl.record_id");
    let params: Vec<SqlValue> = entity_ids.iter().map(|id| SqlValue::Integer(*id))
        .chain(entity_ids.iter().map(|id| SqlValue::Integer(*id))).chain(values).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(params), |row| row.get::<_, i64>(0))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

/// 两端都落在这批实体里的关系 id。
fn relations_between(conn: &Connection, entity_ids: &[i64], filter: &ReadFilter) -> Result<Vec<i64>> {
    let (condition, values) = storage::filter_sql(filter, &[RecordKind::Relation], false)?;
    let placeholders = vec!["?"; entity_ids.len()].join(",");
    let sql = format!("SELECT rl.record_id FROM relations rl JOIN records r ON r.id=rl.record_id \
        WHERE rl.subject_id IN ({placeholders}) AND rl.object_id IN ({placeholders}) AND {condition} ORDER BY rl.record_id");
    let params: Vec<SqlValue> = entity_ids.iter().map(|id| SqlValue::Integer(*id))
        .chain(entity_ids.iter().map(|id| SqlValue::Integer(*id))).chain(values).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(params), |row| row.get::<_, i64>(0))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

/// 参与者里至少有两个落在这批实体里的事件 id。
fn events_between(conn: &Connection, entity_ids: &[i64], filter: &ReadFilter) -> Result<Vec<i64>> {
    let (condition, values) = storage::filter_sql(filter, &[RecordKind::Event], false)?;
    let placeholders = vec!["?"; entity_ids.len()].join(",");
    let sql = format!("SELECT ep.event_id FROM event_participants ep JOIN records r ON r.id=ep.event_id \
        WHERE ep.entity_id IN ({placeholders}) AND {condition} \
        GROUP BY ep.event_id HAVING COUNT(DISTINCT ep.entity_id) >= 2 ORDER BY ep.event_id");
    let params: Vec<SqlValue> = entity_ids.iter().map(|id| SqlValue::Integer(*id)).chain(values).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(params), |row| row.get::<_, i64>(0))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}
