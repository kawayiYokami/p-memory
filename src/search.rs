use crate::{embeddings, graph::{Entity, Neighborhood, Relation}, storage::{self, KnowledgeBase}, types::*, Error, Result};
use rusqlite::{params_from_iter, types::Value as SqlValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};

fn default_limit() -> usize { 10 }
fn weight() -> f64 { 1.0 }
pub fn default_kinds() -> Vec<RecordKind> { vec![RecordKind::Memory, RecordKind::Entity, RecordKind::Relation, RecordKind::Event, RecordKind::Chunk] }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryVector {
    pub space_id: String, pub values: Vec<f32>,
    #[serde(default = "weight")] pub weight: f64,
    #[serde(default)] pub min_score: Option<f64>,
}
/// Graph-First 剪枝：先在图邻域里取到候选实体 id，向量检索只在这些 id 内打分。
/// `SearchRequest::prune` 为 `None` 时保持全量检索，不改默认行为。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPrune {
    /// 图展开起点（实体 record_id）。
    pub root: i64,
    /// 展开跳数，至少 1。
    pub depth: usize,
    /// 邻域节点上限，至少 1。
    pub limit: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchRequest {
    pub query: String, pub filter: ReadFilter, pub kinds: Vec<RecordKind>,
    pub limit: usize, pub candidate_limit: Option<usize>,
    pub vectors: Vec<QueryVector>, pub text_weight: f64,
    pub prune: Option<GraphPrune>,
}
impl Default for SearchRequest {
    fn default() -> Self { Self { query: String::new(), filter: ReadFilter::default(), kinds: default_kinds(), limit: default_limit(), candidate_limit: None, vectors: vec![], text_weight: 1.0, prune: None } }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub key: RecordKey,
    /// Reciprocal rank fusion score, k=60. This is not a probability.
    pub score: f64, pub text_score: Option<f64>, pub vector_scores: BTreeMap<String, f64>,
    pub record: Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult { pub hits: Vec<SearchHit>, pub revision: i64, pub indexed_revision: i64 }

/// 一条命中，附上它在图上挂载的实体邻域。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualHit { pub hit: SearchHit, pub context: Neighborhood }

impl KnowledgeBase {
    pub fn search(&self, request: &SearchRequest) -> Result<SearchResult> {
        storage::validate_filter(&request.filter)?;
        storage::validate_limit(request.limit)?;
        let limit = request.candidate_limit.unwrap_or((request.limit * 5).max(100).min(10_000));
        storage::validate_limit(limit)?;
        if limit < request.limit { return Err(Error::Validation("candidate_limit must be at least limit".into())); }
        if request.query.trim().is_empty() && request.vectors.is_empty() { return Err(Error::Validation("provide a text query or query vectors".into())); }
        if !request.text_weight.is_finite() || request.text_weight <= 0.0 { return Err(Error::Validation("text_weight must be finite and positive".into())); }
        let mut seen = BTreeSet::new();
        for vector in &request.vectors {
            if !seen.insert(&vector.space_id) { return Err(Error::Validation("duplicate query embedding space".into())); }
            if !vector.weight.is_finite() || vector.weight <= 0.0 || vector.min_score.is_some_and(|s| !s.is_finite() || !(-1.0..=1.0).contains(&s)) {
                return Err(Error::Validation("invalid vector weight or min_score".into()));
            }
        }
        // Graph-First 剪枝：图邻域的候选 id 在取库锁之前算好（build_graph 自己会先取一次锁）。
        let allowed = match &request.prune {
            Some(prune) => {
                if prune.depth == 0 { return Err(Error::Validation("graph prune depth must be at least 1".into())); }
                storage::validate_limit(prune.limit)?;
                let mut ids: HashSet<i64> = self.graph().build_graph(&request.filter)?.ego_ids(prune.root, prune.depth, prune.limit).into_iter().collect();
                ids.insert(prune.root);
                Some(ids)
            }
            None => None,
        };
        let state = self.read()?;
        let conn = state.conn();
        let index = self.index()?;
        // Never silently search an old text index after a committed write failure.
        if !request.query.trim().is_empty() { self.sync_index_if_behind(conn)?; }
        let mut scores: BTreeMap<RecordKey, (f64, Option<f64>, BTreeMap<String,f64>)> = BTreeMap::new();
        if !request.query.trim().is_empty() {
            for (rank, (key, score)) in index.search(&request.query, &request.filter, &request.kinds, limit)?.into_iter().enumerate() {
                let hit = scores.entry(key).or_default();
                hit.0 += request.text_weight / (60.0 + (rank + 1) as f64); hit.1 = Some(score);
            }
        }
        for vector in &request.vectors {
            let space = embeddings::get_space(conn, &vector.space_id)?;
            // 分区键与载入查询同源：都用归一化后的 namespace / scope，避免大小写差异导致重复分区。
            let namespace = crate::text::normalized_tag(&request.filter.namespace);
            let scopes: Vec<String> = request.filter.scopes.iter().map(|s| crate::text::normalized_tag(s)).collect();
            let tags: Vec<String> = request.filter.tags.iter().map(|t| crate::text::normalized_tag(t)).collect();
            let mut scored: Vec<(RecordKey, f64)> = Vec::new();
            for scope in scopes {
                let Some(partition) = self.partition(conn, &space, &namespace, &scope)? else { continue };
                scored.extend(partition.search(&vector.values, &request.kinds, &tags, limit, vector.min_score, allowed.as_ref())?);
            }
            // 跨分区汇总后再统一排名：分区各自从 0 计 rank 会破坏 RRF 融合语义。
            scored.sort_by(|a,b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scored.truncate(limit);
            for (rank, (key, score)) in scored.into_iter().enumerate() {
                let hit = scores.entry(key).or_default();
                hit.0 += vector.weight / (60.0 + (rank + 1) as f64); hit.2.insert(space.id.clone(), score);
            }
        }
        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a,b| b.1.0.total_cmp(&a.1.0).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(request.limit);
        // 一次批量取回全部命中本体：`load_many` 的语义等同于逐条 `get`（同样的过滤、同样的装配），
        // 但把每条命中的两轮 SQL（matches_filter + record_value）压成固定三条。
        // 用 id 做键再按 `ranked` 顺序装配，批量取回的顺序不会影响名次。
        let ids: Vec<i64> = ranked.iter().map(|(key, _)| key.id).collect();
        let mut records: BTreeMap<i64, Value> = storage::load_many(conn, &ids, &request.filter)?;
        let mut hits = Vec::new();
        for (key, (score, text_score, vector_scores)) in ranked {
            // 索引里还有文档、库里记录已不在（并发删除、或索引尚未追上）时跳过这一条：
            // 为一条已消失的记录让别人的整次检索失败，代价不对等。
            let Some(record) = records.remove(&key.id) else { continue };
            hits.push(SearchHit { record, key, score, text_score, vector_scores });
        }
        Ok(SearchResult { hits, revision: storage::current_revision(conn)?, indexed_revision: storage::meta(conn, "indexed_revision")? })
    }

    /// 先按 `request` 检索，再给每条命中挂上它在图上所在实体的邻域（实体 + 关系）。
    ///
    /// 「挂载」靠标签文本：命中的记录所带的 tag，与同一 namespace / scope 内某实体的名字或别名
    /// 文本相同，就认为该记录挂在这个实体上（两者都落在 `strings` 表，同一套归一化）。
    /// `limit` 是每个实体邻域的规模上限。命中记录若没有命中任何实体，返回空邻域。
    pub fn search_with_context(&self, request: &SearchRequest, limit: usize) -> Result<Vec<ContextualHit>> {
        storage::validate_limit(limit)?;
        // 只用 namespace / scope 定领域边界；tag 是「挂在哪个实体」的线索，不能反过来筛实体。
        let scope = ReadFilter { tags: vec![], ..request.filter.clone() };
        let hits = self.search(request)?.hits;
        let keys: Vec<i64> = hits.iter().map(|hit| hit.key.id).collect();
        let mut contexts = self.entity_contexts(&keys, &scope, limit)?;
        Ok(hits.into_iter().map(|hit| ContextualHit {
            context: contexts.remove(&hit.key.id).unwrap_or_else(|| Neighborhood { entities: vec![], relations: vec![] }),
            hit,
        }).collect())
    }

    /// 批量解析命中记录的实体邻域，一次取锁、把原来每条命中各自的 N+1 查询压成固定 4 条批量查询。
    /// 结果与逐条展开一致：记录 → tags 文本匹配同领域实体 → 每实体 1 跳关系（各自独立按 `limit` 截断）→ 汇总去重后取 `limit`。
    fn entity_contexts(&self, record_ids: &[i64], filter: &ReadFilter, limit: usize) -> Result<BTreeMap<i64, Neighborhood>> {
        let empty = || Neighborhood { entities: vec![], relations: vec![] };
        let mut out: BTreeMap<i64, Neighborhood> = record_ids.iter().map(|id| (*id, empty())).collect();
        if record_ids.is_empty() { return Ok(out); }
        let state = self.read()?;
        let conn = state.conn();
        // 1. 记录 → 实体：所有命中记录一次查完。
        let (entity_condition, entity_values) = storage::filter_sql(filter, &[RecordKind::Entity], false)?;
        let record_placeholders = vec!["?"; record_ids.len()].join(",");
        let mut stmt = conn.prepare(&format!(
            "SELECT DISTINCT rt.record_id, ea.entity_id FROM record_tags rt \
             JOIN entity_aliases ea ON ea.alias_id=rt.tag_id \
             JOIN records r ON r.id=ea.entity_id \
             WHERE rt.record_id IN ({record_placeholders}) AND {entity_condition} ORDER BY rt.record_id, ea.entity_id"
        ))?;
        let params = record_ids.iter().map(|id| SqlValue::Integer(*id)).chain(entity_values).collect::<Vec<_>>();
        let mut seeds: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        for row in stmt.query_map(params_from_iter(params), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))? {
            let (record_id, entity_id) = row?;
            seeds.entry(record_id).or_default().insert(entity_id);
        }
        let roots: Vec<i64> = seeds.values().flatten().copied().collect::<BTreeSet<_>>().into_iter().collect();
        if roots.is_empty() { return Ok(out); }
        // 2. 一次查出这批实体上的全部 1 跳关系记录。关系记录按 `filter` 过滤，端点按去掉 tags 的
        // `entity_filter` 过滤——与逐条版保持一致（tags 约束关系，scope 约束端点）。
        let (relation_condition, relation_values) = storage::filter_sql(filter, &[RecordKind::Relation], false)?;
        let root_placeholders = vec!["?"; roots.len()].join(",");
        let mut stmt = conn.prepare(&format!(
            "SELECT rl.record_id, rl.subject_id, rl.object_id FROM relations rl JOIN records r ON r.id=rl.record_id \
             WHERE (rl.subject_id IN ({root_placeholders}) OR rl.object_id IN ({root_placeholders})) AND {relation_condition} \
             ORDER BY rl.record_id"
        ))?;
        let params = roots.iter().map(|id| SqlValue::Integer(*id)).chain(roots.iter().map(|id| SqlValue::Integer(*id))).chain(relation_values).collect::<Vec<_>>();
        let mut edges: Vec<(i64, i64, i64)> = Vec::new();
        for row in stmt.query_map(params_from_iter(params), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))? {
            edges.push(row?);
        }
        // 3. 批量取种子实体、端点实体和关系记录。
        let entity_filter = ReadFilter { tags: vec![], ..filter.clone() };
        let root_entities: BTreeMap<i64, Entity> = storage::load_many(conn, &roots, filter)?;
        let endpoint_ids: Vec<i64> = edges.iter().flat_map(|(_, subject, object)| [*subject, *object]).collect::<BTreeSet<_>>().into_iter().collect();
        let endpoint_entities: BTreeMap<i64, Entity> = storage::load_many(conn, &endpoint_ids, &entity_filter)?;
        let relation_ids: Vec<i64> = edges.iter().map(|(id, _, _)| *id).collect();
        let relation_records: BTreeMap<i64, Relation> = storage::load_many(conn, &relation_ids, filter)?;
        // 4. 在内存里按记录分发，并复现「每个实体各自按 limit 截断」的语义。
        let mut incident: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (i, &(_, subject, object)) in edges.iter().enumerate() {
            incident.entry(subject).or_default().push(i);
            if object != subject { incident.entry(object).or_default().push(i); }
        }
        for (record_id, root_ids) in &seeds {
            let mut entities: BTreeMap<i64, Entity> = BTreeMap::new();
            for id in root_ids { if let Some(entity) = root_entities.get(id) { entities.insert(*id, entity.clone()); } }
            let mut relations: BTreeMap<i64, Relation> = BTreeMap::new();
            for root in root_ids {
                let mut count = 0usize;
                for &i in incident.get(root).map(Vec::as_slice).unwrap_or(&[]) {
                    let (relation_id, subject, object) = edges[i];
                    let Some(relation) = relation_records.get(&relation_id) else { continue };
                    let endpoint = if subject == *root { object } else { subject };
                    let Some(entity) = endpoint_entities.get(&endpoint) else { continue };
                    entities.entry(endpoint).or_insert_with(|| entity.clone());
                    relations.entry(relation_id).or_insert_with(|| relation.clone());
                    count += 1;
                    if count == limit { break; }
                }
            }
            out.insert(*record_id, Neighborhood { entities: entities.into_values().collect(), relations: relations.into_values().take(limit).collect() });
        }
        Ok(out)
    }
}
