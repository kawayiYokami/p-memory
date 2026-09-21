use crate::{embeddings, graph::{Entity, Neighborhood, Relation}, storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use parking_lot::Mutex;
use rusqlite::{params_from_iter, types::Value as SqlValue, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

fn default_limit() -> usize { 10 }
/// 阶段打点：只在注册了事件 sink 时才计时，没注册就是一次空判断。
fn mark(stages: &mut Option<crate::events::StageTimer>, name: &str) {
    if let Some(stages) = stages.as_mut() { stages.mark(name); }
}
fn yes() -> bool { true }
pub fn default_kinds() -> Vec<RecordKind> { vec![RecordKind::Memory, RecordKind::Entity, RecordKind::Relation, RecordKind::Event, RecordKind::Chunk] }

/// 一个标记文本在 strings 表里的 id；库里没有这个标记就是 `None`。
fn string_id(conn: &rusqlite::Connection, value: &str) -> Result<Option<i64>> {
    Ok(conn.query_row("SELECT id FROM strings WHERE text=?1", [text::normalized_tag(value)], |r| r.get(0)).optional()?)
}

/// 把 `filter` 里的标记文本折算成 id：索引里只存 id，折算在进索引之前做完。
/// 换不到 id 说明库里没有这个标记，本次不可能有命中，直接给空结果，不必进索引碰。
fn index_filter(conn: &rusqlite::Connection, filter: &ReadFilter, kinds: &[RecordKind]) -> Result<Option<crate::index::IndexFilter>> {
    let Some(namespace) = string_id(conn, &filter.namespace)? else { return Ok(None) };
    let mut scopes = Vec::with_capacity(filter.scopes.len());
    for scope in &filter.scopes {
        match string_id(conn, scope)? { Some(id) => scopes.push(id), None => return Ok(None) }
    }
    let mut tags = Vec::with_capacity(filter.tags.len());
    for tag in &filter.tags {
        match string_id(conn, tag)? { Some(id) => tags.push(id), None => return Ok(None) }
    }
    Ok(Some(crate::index::IndexFilter { namespace, scopes, kinds: kinds.iter().map(|kind| kind.code()).collect(), tags, note_ids: filter.note_ids.clone() }))
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
    /// 走向量路时用哪条向量空间；库用该空间注册的回调嵌入查询词，宿主不接触向量。
    pub embed_space: Option<String>,
    pub text_weight: f64, pub prune: Option<GraphPrune>,
    /// 本次是否走全文路。
    #[serde(default = "yes")] pub text: bool,
    /// 本次是否走向量路（需要 `embed_space`）。
    #[serde(default = "yes")] pub vector: bool,
    /// 本次是否重排；未注册重排回调时被忽略。
    #[serde(default = "yes")] pub rerank: bool,
    /// 是否额外返回过滤后的匹配总量。
    #[serde(default)] pub with_total: bool,
    /// 本次全文路限定在哪一列命中：`all`（默认，正文+名字）、`text`、`name`、`path`。
    #[serde(default)] pub match_field: MatchField,
    /// 切片折叠后，每条命中最多聚合这一篇里排名最高的几片（含本条）到 `top_chunks`；
    /// 0 表示不聚合，只留排名最高的那一片。只看名字/目录列时不做聚合（那两列是笔记级的）。
    #[serde(default = "default_top_chunks_per_note")] pub top_chunks_per_note: usize,
}
fn default_top_chunks_per_note() -> usize { 3 }
impl Default for SearchRequest {
    fn default() -> Self {
        Self { query: String::new(), filter: ReadFilter::default(), kinds: default_kinds(), limit: default_limit(),
            candidate_limit: None, embed_space: None, text_weight: 1.0, prune: None,
            text: true, vector: true, rerank: true, with_total: false, match_field: MatchField::All,
            top_chunks_per_note: default_top_chunks_per_note() }
    }
}
/// 折叠后挂在代表命中上的一个片段：同一篇笔记里也命中本次查询的那一片。
/// 只带定位，不带正文——正文按 `id` 从索引取回（`notes.get_chunk`）。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChunkRef {
    /// 切片记录 id。
    pub id: i64,
    /// 该片在原文里的起始行（1 起）。
    pub offset: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub key: RecordKey,
    /// Reciprocal rank fusion score, k=60. This is not a probability.
    pub score: f64, pub text_score: Option<f64>, pub vector_scores: BTreeMap<String, f64>,
    /// 本条目在重排回调那里的分数；未重排时为 `None`。
    #[serde(default)] pub rerank_score: Option<f64>,
    /// 本条目是切片时：其所属笔记里命中本次查询的切片总数（按请求的匹配列计，含本条）。
    /// 非切片命中、或请求只看名字/目录列时为 `None`。
    /// 它只描述「这篇文档有多少相关片段」，与结果窗口、翻页无关。
    #[serde(default)] pub note_chunks: Option<usize>,
    /// 本条代表的那一篇里，命中本次查询、排名最高的若干片段（第 0 条就是本条自身），
    /// 按名次排列、至多 `top_chunks_per_note` 条。同一篇内的次序取索引里的相关度
    /// （严格词元命中排在宽松命中之前），与命中本身的 `score` 不同源。
    /// 被折叠掉的片段在这里找回来，调用方不必为「同一篇还有别的相关片段」再搜一次。
    /// 非切片命中、或请求只看名字/目录列时为空。
    #[serde(default)] pub top_chunks: Vec<ChunkRef>,
    pub record: Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>, pub revision: i64, pub indexed_revision: i64,
    /// 过滤之后、截断之前的匹配数；`with_total=false` 时为 `None`。
    #[serde(default)] pub total: Option<usize>,
    #[serde(default)] pub diagnostics: SearchDiagnostics,
}

/// 一条命中，附上它在图上挂载的实体邻域。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualHit { pub hit: SearchHit, pub context: Neighborhood }

// ── 宿主重排回调 ─────────────────────────────────────────────────────

/// 重排回调：`(查询词, 文档) -> 分数`，与输入文档等长、按序给出。
/// 重排不绑定向量空间，一个进程内一个回调；多宿主各在自己的进程里注册。
pub trait Reranker: Send {
    fn rerank(&mut self, query: &str, documents: &[String]) -> std::result::Result<Vec<f32>, String>;
}

impl<F> Reranker for F
where F: FnMut(&str, &[String]) -> std::result::Result<Vec<f32>, String> + Send {
    fn rerank(&mut self, query: &str, documents: &[String]) -> std::result::Result<Vec<f32>, String> { self(query, documents) }
}

fn default_max_tokens_total() -> usize { 8192 }
fn default_max_tokens_per_doc() -> usize { 1024 }
fn default_max_candidates() -> usize { 50 }

/// 两路候选按名次交替合并、去重：全文第 0 名、向量第 0 名、全文第 1 名、向量第 1 名……
/// 用于把两路结果直接交给重排——两路各占约一半名额，不偏向任何一路。
fn merge_candidates(text: &[(RecordKey, f64)], vector: &[(RecordKey, f64)]) -> Vec<(RecordKey, f64)> {
    let mut seen = HashSet::new();
    let mut merged = Vec::with_capacity(text.len() + vector.len());
    let mut index = 0;
    while index < text.len() || index < vector.len() {
        if let Some(item) = text.get(index) { if seen.insert(item.0) { merged.push(*item); } }
        if let Some(item) = vector.get(index) { if seen.insert(item.0) { merged.push(*item); } }
        index += 1;
    }
    merged
}

/// 宿主注册重排回调时一并声明的预算。条数上限与 token 上限是**与门**，任一到顶就停：
/// 几十字符的短候选，token 预算能装几百条，而重排的成本由批次条数主导，所以条数必须单独封顶；
/// 长正文则由 token 封顶（模型吃不下更多）。两者超出的候选不是变慢就是直接报错。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerankerOptions {
    /// 送进重排的总预算：查询词 + 所有候选文档，从前往后累加到超额为止。
    #[serde(default = "default_max_tokens_total")] pub max_tokens_total: usize,
    /// 送进重排的候选条数上限。
    #[serde(default = "default_max_candidates")] pub max_candidates: usize,
    #[serde(default = "default_max_tokens_per_doc")] pub max_tokens_per_doc: usize,
    /// 查询词的 token 预算，`None` 表示不截断。
    #[serde(default)] pub max_tokens_query: Option<usize>,
}
impl Default for RerankerOptions {
    fn default() -> Self { Self { max_tokens_total: default_max_tokens_total(), max_candidates: default_max_candidates(), max_tokens_per_doc: default_max_tokens_per_doc(), max_tokens_query: None } }
}

pub(crate) struct RerankerEntry { pub options: RerankerOptions, pub reranker: Box<dyn Reranker> }

/// 进程内的重排回调位。`Arc` 让检索线程拿出去后立刻放掉注册表锁。
#[derive(Default)]
pub(crate) struct RerankerRegistry { entry: Mutex<Option<Arc<Mutex<RerankerEntry>>>> }

impl RerankerRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn is_registered(&self) -> bool { self.entry.lock().is_some() }
    pub fn get(&self) -> Option<Arc<Mutex<RerankerEntry>>> { self.entry.lock().clone() }
    pub fn register(&self, entry: RerankerEntry) { *self.entry.lock() = Some(Arc::new(Mutex::new(entry))); }
    pub fn remove(&self) -> bool { self.entry.lock().take().is_some() }
}

/// 注册校验用的样本：真实调用一次，条数与有限性不符即拒绝绑定。
const RERANK_SAMPLE_DOCS: [&str; 2] = ["重排校验样本一", "rerank probe two"];

impl KnowledgeBase {
    /// 注册重排回调。调用前按声明的定长约束强制截断，宿主不自己重排。
    pub fn register_reranker<F: Reranker + 'static>(&self, reranker: F) -> Result<()> {
        self.register_reranker_with(reranker, RerankerOptions::default())
    }

    pub fn register_reranker_with<F: Reranker + 'static>(&self, reranker: F, options: RerankerOptions) -> Result<()> {
        if options.max_tokens_total == 0 { return Err(Error::Validation("max_tokens_total must be at least 1".into())); }
        if options.max_candidates == 0 { return Err(Error::Validation("max_candidates must be at least 1".into())); }
        if options.max_tokens_per_doc == 0 { return Err(Error::Validation("max_tokens_per_doc must be at least 1".into())); }
        if options.max_tokens_query == Some(0) { return Err(Error::Validation("max_tokens_query must be positive".into())); }
        let mut entry = RerankerEntry { options, reranker: Box::new(reranker) };
        let documents: Vec<String> = RERANK_SAMPLE_DOCS.iter().map(|sample| (*sample).to_string()).collect();
        // 样本返回 Ok 就校验条数与有限性；回调当场不可用时无从校验形状，允许绑定，
        // 可用性留到检索时降级并写进诊断——这与「重排挂了不报错」的降级口径一致。
        if let Ok(produced) = entry.reranker.rerank("校验样本", &documents) {
            if produced.len() != documents.len() {
                return Err(Error::Validation(format!("reranker returned {} scores for {} documents", produced.len(), documents.len())));
            }
            if produced.iter().any(|score| !score.is_finite()) { return Err(Error::Validation("reranker scores must be finite".into())); }
        }
        self.engine.rerankers.register(entry);
        Ok(())
    }

    pub fn unregister_reranker(&self) -> bool { self.engine.rerankers.remove() }

    pub fn reranker_registered(&self) -> bool { self.engine.rerankers.is_registered() }

    /// 全文路。派生索引查询失败时返回 `Index`，由调用方触发重建后重试。
    fn search_text(&self, conn: &rusqlite::Connection, query: &str, filter: &ReadFilter, kinds: &[RecordKind], limit: usize, field: MatchField) -> Result<Vec<(RecordKey, f64)>> {
        self.sync_index_if_behind(conn)?;
        let Some(index_filter) = index_filter(conn, filter, kinds)? else { return Ok(Vec::new()) };
        // 领域登记了谓词等价词时先扩散：把同义写法一并纳入召回（如「beta」补「alpha」）。
        let expanded = crate::graph::match_predicate_synonyms(conn, &filter.namespace, query)?;
        self.index()?.search_in(&expanded, &index_filter, limit, field)
    }

    pub fn search(&self, request: &SearchRequest) -> Result<SearchResult> {
        storage::validate_filter(&request.filter)?;
        storage::validate_limit(request.limit)?;
        let query = request.query.trim();
        if query.is_empty() { return Err(Error::Validation("a text query is required".into())); }
        // 向量路要同时满足「开关打开」与「给出目标空间」；缺任一条件就不走向量路，不是错误。
        let vector_path = request.vector && request.embed_space.is_some();
        if !request.text && !vector_path { return Err(Error::Validation("enable at least one of text or vector".into())); }
        if !request.text_weight.is_finite() || request.text_weight <= 0.0 { return Err(Error::Validation("text_weight must be finite and positive".into())); }
        let limit = request.candidate_limit.unwrap_or((request.limit * 5).max(1_000).min(10_000));
        storage::validate_limit(limit)?;
        if limit < request.limit { return Err(Error::Validation("candidate_limit must be at least limit".into())); }
        // 事件只在宿主注册了 sink 时才产出：没注册就全程不构造、不格式化。
        let sink = self.engine.events.get();
        let started = sink.is_some().then(|| std::time::Instant::now());
        let mut stages = sink.is_some().then(crate::events::StageTimer::start);
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
        mark(&mut stages, "prepare");
        let mut diagnostics = SearchDiagnostics::default();
        // 向量路的第一步是「库自己把查询词嵌入」——宿主只给词，不给向量。
        // 这一步是模型往返，必须在取库锁之前做完。
        let mut embedded_query: Option<(embeddings::EmbeddingSpace, Vec<f32>)> = None;
        // 向量路的候选限在「该领域实际启用了向量化的类型」上：请求的 kinds 与它取交集。
        // 交集为空就整条向量路不走——空的 kinds 在打分侧表示「不过滤」，
        // 直接传下去会把已关闭类型的存量向量也捞回来。
        let mut vector_kinds: Vec<RecordKind> = Vec::new();
        if let Some(space_id) = request.embed_space.as_deref().filter(|_| request.vector) {
            let gated = { let state = self.read()?; embeddings::namespace_vectorization(state.conn(), &request.filter.namespace)? };
            if !gated {
                diagnostics.degraded.push(Degrade::NamespaceDisabled);
            } else {
                // 只让「已启用、且这一档自己已经补齐」的类型走向量：
                // 某一档没补完，就把它从向量路里剔除，它的存量向量先不参与打分。
                // 半个领域的向量参会比只用全文更糟——排名会偏向「先补完的那部分」。
                let (enabled, ready) = {
                    let state = self.read()?;
                    let conn = state.conn();
                    let namespace = &request.filter.namespace;
                    (embeddings::enabled_kinds(conn, namespace)?, embeddings::ready_kinds(conn, namespace, space_id)?)
                };
                let requested: Vec<RecordKind> = request.kinds.iter().copied().filter(|kind| enabled.contains(kind)).collect();
                vector_kinds = requested.iter().copied().filter(|kind| ready.contains(kind)).collect();
                if !requested.is_empty() {
                    // 空间没登记过是配置错误，直接报；登记过但没绑回调才是可降级的情形。
                    let space = { let state = self.read()?; embeddings::get_space(state.conn(), space_id)? };
                    if vector_kinds.is_empty() {
                        // 请求要的类型都启用了，但没有一档补齐：这一轮只给全文。
                        diagnostics.degraded.push(Degrade::VectorNotReady);
                    } else {
                        match self.engine.embedders.get(space_id) {
                            None => diagnostics.degraded.push(Degrade::NoEmbedder),
                            Some(entry) => {
                                let produced = { let mut guard = entry.lock(); guard.embed(&[query.to_string()]) };
                                match produced {
                                    Ok(mut values) if values.len() == 1 => embedded_query = Some((space, values.remove(0))),
                                    _ => diagnostics.degraded.push(Degrade::EmbedFailed),
                                }
                            }
                        }
                    }
                }
            }
        }
        if vector_path { mark(&mut stages, "embed"); }
        let state = self.read()?;
        let conn = state.conn();
        let mut text_rank: Vec<(RecordKey, f64)> = Vec::new();
        if request.text {
            let text_hits = match self.search_text(conn, query, &request.filter, &request.kinds, limit, request.match_field) {
                Ok(hits) => Some(hits),
                // 派生索引查询失败：当场从权威数据重建一次再重试。恢复得了就照常给结果；
                // 连重建都失败才隔离文本路。不静默退回空文本——那等于把 BM25 地板也丢掉。
                Err(Error::Index(_)) => match self.rebuild_indexes() {
                    Ok(_) => match self.search_text(conn, query, &request.filter, &request.kinds, limit, request.match_field) {
                        Ok(hits) => Some(hits),
                        Err(_) => { diagnostics.degraded.push(Degrade::TextIndexUnavailable); Some(Vec::new()) }
                    },
                    Err(_) => { diagnostics.degraded.push(Degrade::TextIndexUnavailable); Some(Vec::new()) }
                },
                // 库已关闭是调用错误，不该被降级吞掉。
                Err(error) => return Err(error),
            };
            text_rank = text_hits.unwrap_or_default();
            diagnostics.text_used = true;
        }
        mark(&mut stages, "text");
        let mut vector_rank: Vec<(RecordKey, f64)> = Vec::new();
        let mut vector_space_id: Option<String> = None;
        if let Some((space, vector)) = &embedded_query {
            // 分区键与载入查询同源：都用归一化后的 namespace / scope，避免大小写差异导致重复分区。
            let namespace = text::normalized_tag(&request.filter.namespace);
            let scopes: Vec<String> = request.filter.scopes.iter().map(|s| text::normalized_tag(s)).collect();
            let tags: Vec<String> = request.filter.tags.iter().map(|t| text::normalized_tag(t)).collect();
            let mut scored: Vec<(RecordKey, f64)> = Vec::new();
            for scope in scopes {
                let Some(partition) = self.partition(conn, space, &namespace, &scope)? else { continue };
                scored.extend(partition.search(vector, &vector_kinds, &tags, &request.filter.note_ids, limit, allowed.as_ref())?);
            }
            // 跨分区汇总后再统一排名：分区各自从 0 计 rank 会破坏 RRF 融合语义。
            scored.sort_by(|a,b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scored.truncate(limit);
            diagnostics.vector_used = true;
            vector_space_id = Some(space.id.clone());
            vector_rank = scored;
        }
        if diagnostics.vector_used { mark(&mut stages, "vector"); }
        // 两路各自的分数与 RRF 融合分：融合分只用来充当 `score` 字段与「没法重排」时的兜底排序，
        // 重排可用时它不参与候选取舍，也不参与名次决定。
        let mut scores: BTreeMap<RecordKey, (f64, Option<f64>, BTreeMap<String,f64>)> = BTreeMap::new();
        for (rank, (key, score)) in text_rank.iter().enumerate() {
            let hit = scores.entry(*key).or_default();
            hit.0 += request.text_weight / (60.0 + (rank + 1) as f64); hit.1 = Some(*score);
        }
        if let Some(space_id) = &vector_space_id {
            for (rank, (key, score)) in vector_rank.iter().enumerate() {
                let hit = scores.entry(*key).or_default();
                hit.0 += 1.0 / (60.0 + (rank + 1) as f64); hit.2.insert(space_id.clone(), *score);
            }
        }
        let total = if request.with_total { Some(storage::count_matches(conn, &request.filter, &request.kinds)?) } else { None };
        let mut rerank_scores: BTreeMap<RecordKey, f64> = BTreeMap::new();
        // 候选顺序：重排可用时按两路交替合并（融合分不参与这一步），否则按 RRF 融合分。
        // 折叠排在这之后、送重排之前——同一篇只留一个代表片，重排预算不重复花在同一篇上。
        let rerank_entry = if request.rerank { self.engine.rerankers.get() } else { None };
        let use_rerank = rerank_entry.is_some();
        let ordered: Vec<RecordKey> = if use_rerank {
            merge_candidates(&text_rank, &vector_rank).into_iter().map(|(key, _)| key).collect()
        } else {
            let mut by_score: Vec<(RecordKey, f64)> = scores.iter().map(|(key, hit)| (*key, hit.0)).collect();
            by_score.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            by_score.into_iter().map(|(key, _)| key).collect()
        };
        let candidate_count = ordered.len();
        mark(&mut stages, "fuse");
        // 同一篇笔记的多个命中切片折叠成一条：只保留最靠前的那一片，
        // 否则一篇对话体文档会用自己几十个片段占满整个结果列表，把别的文档全挤出去。
        // 折叠时顺手把同篇在窗口内的其余命中片段收进 `top_chunks`（第 0 条是代表片自身）。
        // 聚合与计数只在按正文列（或默认的正文+名字列）检索时做：名字与目录列是笔记级的，
        // 一篇至多一条命中，聚无可聚、也数无可数。
        let aggregate = request.text && matches!(request.match_field, MatchField::All | MatchField::Text);
        let per_note = if aggregate { request.top_chunks_per_note } else { 0 };
        let ordered_ids: Vec<i64> = ordered.iter().map(|key| key.id).collect();
        let chunk_notes = storage::chunk_notes(conn, &ordered_ids)?;
        let mut folded: Vec<RecordKey> = Vec::new();
        let mut top_chunks: HashMap<i64, Vec<ChunkRef>> = HashMap::new();
        if chunk_notes.is_empty() {
            folded = ordered;
        } else {
            let mut seen_notes: HashSet<i64> = HashSet::new();
            for key in ordered {
                let Some(&(note_id, offset)) = chunk_notes.get(&key.id) else { folded.push(key); continue };
                if seen_notes.insert(note_id) {
                    if per_note > 0 { top_chunks.insert(note_id, vec![ChunkRef { id: key.id, offset }]); }
                    folded.push(key);
                } else if per_note > 0 {
                    let list = top_chunks.entry(note_id).or_default();
                    if list.len() < per_note { list.push(ChunkRef { id: key.id, offset }); }
                }
            }
        }
        let folded_count = folded.len();
        mark(&mut stages, "fold");
        // 候选取舍：重排可用时从前往后取，条数上限与 token 总预算任一先到顶就停——短候选靠
        // 条数封顶（批次条数才是重排的成本），长正文靠 token 封顶。没有重排时直接取前 limit 条。
        let mut selected: Vec<RecordKey> = Vec::new();
        let mut reranked = false;
        let mut rerank_docs = 0usize;
        let mut rerank_tokens = 0usize;
        if let Some(entry) = rerank_entry {
            let options = entry.lock().options;
            let budgeted_query = options.max_tokens_query.map(|budget| text::truncate_to_tokens(query, budget)).unwrap_or_else(|| query.to_string());
            let ids: Vec<i64> = folded.iter().map(|key| key.id).collect();
            let bodies = match self.index() { Ok(index) => index.bodies(&ids)?, Err(_) => BTreeMap::new() };
            let names = storage::entity_names(conn, &ids).unwrap_or_default();
            // 候选正文取自索引的 stored 字段（切片正文在这里）。实体的正文列不含规范名，
            // 这里给实体把规范名拼在正文前——否则纯名实体送到重排的文档是空的，等于没内容可判。
            let mut used = text::count_tokens(&budgeted_query);
            let mut candidates: Vec<RecordKey> = Vec::new();
            let mut documents: Vec<String> = Vec::new();
            for key in &folded {
                // 与门：条数先到顶就停，不必再为这一条算 token。
                if candidates.len() >= options.max_candidates { break; }
                let body = bodies.get(&key.id).map(String::as_str).unwrap_or("");
                let full = match names.get(&key.id).filter(|name| !name.is_empty()) {
                    Some(name) => format!("{name} {body}"),
                    None => body.to_string(),
                };
                let document = text::truncate_to_tokens(&full, options.max_tokens_per_doc);
                let cost = text::count_tokens(&document);
                if used + cost > options.max_tokens_total { break; }
                used += cost;
                candidates.push(*key);
                documents.push(document);
            }
            diagnostics.rerank_candidates = candidates.len();
            diagnostics.rerank_truncated = folded.len().saturating_sub(candidates.len());
            rerank_docs = candidates.len();
            rerank_tokens = used;
            // 没有候选就不必调模型：多数重排服务把空文档列表当无效请求，会误标降级。
            let produced = if candidates.is_empty() { Ok(Vec::new()) }
            else { let mut guard = entry.lock(); guard.reranker.rerank(&budgeted_query, &documents) };
            match produced {
                Ok(values) if values.len() == candidates.len() && values.iter().all(|value| value.is_finite()) => {
                    let mut pairs: Vec<(RecordKey, f32)> = candidates.into_iter().zip(values).collect();
                    pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    for (key, score) in pairs { rerank_scores.insert(key, f64::from(score)); selected.push(key); }
                    diagnostics.reranked = true;
                    reranked = true;
                }
                // 重排产出不符：退回候选顺序，不把整次检索判死。
                _ => diagnostics.degraded.push(Degrade::RerankFailed),
            }
        }
        if !reranked { selected = folded; }
        if use_rerank { mark(&mut stages, "rerank"); }
        selected.truncate(request.limit);
        // 「这一篇命中多少片」只对最终返回的条目算：它是结果上的字段，不是候选窗口的属性。
        // 折叠后的候选常有几百篇，而返回只有 limit 条——按候选窗口算等于白算几十倍。
        // 计数走一次遍历、按笔记分桶，不为每篇各发一次查询。
        let note_counts: BTreeMap<i64, usize> = if aggregate {
            match index_filter(conn, &request.filter, &request.kinds)? {
                Some(ifilter) => {
                    let mut targets: Vec<i64> = selected.iter().filter_map(|key| chunk_notes.get(&key.id).map(|(note_id, _)| *note_id)).collect();
                    targets.sort_unstable();
                    targets.dedup();
                    self.index()?.count_in_many(query, &ifilter, request.match_field, &targets)?.into_iter().collect()
                }
                None => BTreeMap::new(),
            }
        } else { BTreeMap::new() };
        mark(&mut stages, "count");
        // 一次批量取回全部命中本体：`load_many` 的语义等同于逐条 `get`（同样的过滤、同样的装配），
        // 但把每条命中的两轮 SQL（matches_filter + record_value）压成固定三条。
        // 用 id 做键再按 `selected` 顺序装配，批量取回的顺序不会影响名次。
        let ids: Vec<i64> = selected.iter().map(|key| key.id).collect();
        let mut records: BTreeMap<i64, Value> = storage::load_many(conn, &ids, &request.filter)?;
        let mut hits = Vec::new();
        for key in selected {
            // 索引里还有文档、库里记录已不在（并发删除、或索引尚未追上）时跳过这一条：
            // 为一条已消失的记录让别人的整次检索失败，代价不对等。
            let Some(record) = records.remove(&key.id) else { continue };
            let note_id = chunk_notes.get(&key.id).map(|(note_id, _)| *note_id);
            let note_chunks = note_id.and_then(|id| note_counts.get(&id).copied());
            let top_chunks = note_id.and_then(|id| top_chunks.remove(&id)).unwrap_or_default();
            let (score, text_score, vector_scores) = scores.remove(&key).unwrap_or((0.0, None, BTreeMap::new()));
            hits.push(SearchHit { record, key, score, text_score, vector_scores, rerank_score: rerank_scores.get(&key).copied(), note_chunks, top_chunks });
        }
        for degrade in &diagnostics.degraded { self.note_degrade(*degrade); }
        if let (Some(sink), Some(stages)) = (sink, stages) {
            let mut event = crate::events::LogEvent::new("search");
            event.ms = started.map(|started| started.elapsed().as_millis() as u64).unwrap_or(0);
            event.stages = stages.finish("load");
            event.candidates = Some(candidate_count);
            event.folded = Some(folded_count);
            event.rerank_docs = Some(rerank_docs);
            event.rerank_tokens = Some(rerank_tokens);
            event.hits = Some(hits.len());
            event.degraded = diagnostics.degraded.clone();
            // 日志回调不该有能力打断检索：宿主 sink 里 panic 只丢这一条事件。
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink(&event)));
        }
        Ok(SearchResult { hits, revision: storage::current_revision(conn)?, indexed_revision: storage::meta(conn, "indexed_revision")?, total, diagnostics })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryInput;
    use std::sync::atomic::Ordering;

    /// 索引查询失败时，检索应触发一次重建并重试：恢复得了就照常出结果，
    /// 连重建都失败才隔离文本路——不再静默退回空文本。
    /// token 计数按字符密度估算：ASCII ≈ 4 字符/token，非 ASCII ≈ 2 字符/token，向上取整。
    #[test]
    fn token_count_follows_character_density() {
        assert_eq!(text::count_tokens(""), 0);
        assert_eq!(text::count_tokens("abcd"), 1);
        assert_eq!(text::count_tokens("abcde"), 2);
        assert_eq!(text::count_tokens("中"), 1);
        assert_eq!(text::count_tokens("中国"), 1);
        assert_eq!(text::count_tokens("中国人"), 2);
        // 截断与计数同源：截断后的文本不超预算。
        assert_eq!(text::truncate_to_tokens("abcd", 1), "abcd");
        assert_eq!(text::truncate_to_tokens("abcde", 1), "abcd");
        assert_eq!(text::truncate_to_tokens("中国人", 1), "中国");
        assert!(text::count_tokens(&text::truncate_to_tokens("中国人", 1)) <= 1);
    }

    #[test]
    fn index_query_failure_rebuilds_then_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        kb.memories().upsert(MemoryInput::new("索引故障恢复的独有措辞")).unwrap();
        let index = kb.index().unwrap();
        let request = SearchRequest {
            query: "索引故障恢复的独有措辞".into(), kinds: vec![RecordKind::Memory],
            vector: false, rerank: false, ..Default::default()
        };

        index.fail_search.store(true, Ordering::SeqCst);
        let degraded = kb.search(&request).unwrap();
        assert!(index.rebuilds.load(Ordering::SeqCst) >= 1, "索引查询失败必须触发重建");
        assert!(degraded.diagnostics.degraded.contains(&Degrade::TextIndexUnavailable),
            "重建之后仍失败，才隔离文本路");

        index.fail_search.store(false, Ordering::SeqCst);
        let recovered = kb.search(&request).unwrap();
        assert_eq!(recovered.hits.len(), 1, "故障排除后索引可用，照常命中");
        assert!(!recovered.diagnostics.degraded.contains(&Degrade::TextIndexUnavailable));
    }
}
