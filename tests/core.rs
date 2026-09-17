use p_memory::*;
use p_memory::graph::{EntityInput, EventInput, RelationInput};
use p_memory::notes::chunk_text;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn memory(text: &str, scope: &str) -> MemoryInput {
    let mut value = MemoryInput::new(text);
    value.record.scope = scope.into(); value
}
fn entity(name: &str) -> EntityInput {
    EntityInput { record: RecordInput::default(), name: name.into(), entity_type: "person".into(),
        aliases: vec![], attributes: BTreeMap::new(), summary: String::new() }
}

// ── 假嵌入回调 ────────────────────────────────────────────────────────

/// 确定性兜底向量：同一文本永远得到同一向量，且不为零。
fn fallback_vector(text: &str, dimension: usize) -> Vec<f32> {
    let mut values = vec![0f32; dimension];
    for (index, byte) in text.bytes().enumerate() {
        values[(index + usize::from(byte)) % dimension] += 1.0 + f32::from(byte % 7) * 0.1;
    }
    if values.iter().all(|value| *value == 0.0) { values[0] = 1.0; }
    values
}

fn attempt_error() -> EmbedCallbackError { EmbedCallbackError::other("boom") }

/// 假嵌入：可以指定「文本 → 向量」表，未列出的走确定性兜底；并记录每批实收条数。
struct FakeEmbedder {
    dimension: usize,
    table: Arc<Mutex<BTreeMap<String, Vec<f32>>>>,
    calls: Arc<Mutex<Vec<usize>>>,
}

impl FakeEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dimension, table: Arc::new(Mutex::new(BTreeMap::new())), calls: Arc::new(Mutex::new(Vec::new())) }
    }
    fn with_table(dimension: usize, entries: &[(&str, Vec<f32>)]) -> Self {
        let embedder = Self::new(dimension);
        {
            let mut table = embedder.table.lock().unwrap();
            for (text, vector) in entries { table.insert((*text).to_string(), vector.clone()); }
        }
        embedder
    }
    fn lengths(&self) -> Arc<Mutex<Vec<usize>>> { self.calls.clone() }
}

impl Embedder for FakeEmbedder {
    fn embed(&mut self, texts: &[String]) -> std::result::Result<Vec<Vec<f32>>, EmbedCallbackError> {
        self.calls.lock().unwrap().push(texts.len());
        if texts.iter().any(|text| text.contains("触发降级")) { return Err(attempt_error()); }
        let table = self.table.lock().unwrap();
        Ok(texts.iter().map(|text| table.get(text).cloned().unwrap_or_else(|| fallback_vector(text, self.dimension))).collect())
    }
}

/// 注册空间与它的假嵌入回调。
fn space(kb: &KnowledgeBase, id: &str, dimension: usize) {
    kb.embeddings().register_space(EmbeddingSpace { id: id.into(), model: "fixture/v1".into(), dimension, text_version: 1, encoding: "f32".into() }).unwrap();
    kb.embeddings().register_embedder(id, FakeEmbedder::new(dimension)).unwrap();
}
fn vector_query(space_id: &str, query: &str, kinds: Vec<RecordKind>) -> SearchRequest {
    SearchRequest { query: query.into(), kinds, embed_space: Some(space_id.into()), text: false, ..Default::default() }
}

#[test]
fn transactions_scopes_pagination_and_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    assert!(matches!(KnowledgeBase::open(dir.path()), Err(Error::Locked(_))));
    let mut first = memory("上海 茶 Rust ＡＰＩ", "public");
    first.record.tags = vec![" RUST ".into(), "rust".into()];
    let receipt = kb.memories().upsert(first.clone()).unwrap();
    let a = receipt.value.header.id;
    let b = kb.memories().upsert(memory("其他内容", "public")).unwrap().value.header.id;
    let c = kb.memories().upsert(memory("上海 茶", "private")).unwrap().value.header.id;
    let mut page = PageRequest { limit: 1, ..Default::default() };
    let result = kb.memories().list(&page).unwrap();
    assert_eq!(result.items[0].header.id, a); page.after = result.next_cursor;
    assert_eq!(kb.memories().list(&page).unwrap().items[0].header.id, b);
    assert!(matches!(kb.memories().get(c, &ReadFilter::default()), Err(Error::NotFound(_))));
    let mut moved = memory("moved", "private"); moved.record.id = Some(a);
    assert!(matches!(kb.memories().upsert(moved), Err(Error::Conflict(_))));
    let revision = kb.health().unwrap().revision;
    assert!(kb.memories().upsert_many(&[memory("valid", "public"), MemoryInput::new(" ")]).is_err());
    assert_eq!(kb.health().unwrap().revision, revision);
    first.record.expected_revision = Some(0); first.record.id = Some(a);
    assert!(matches!(kb.memories().upsert(first), Err(Error::StaleRevision(_))));
    let req = SearchRequest { query: "api".into(), filter: ReadFilter { tags: vec!["rust".into()], ..Default::default() }, ..Default::default() };
    assert_eq!(kb.search(&req).unwrap().hits[0].key.id, a);
    let clone = kb.clone(); kb.close().unwrap();
    assert!(matches!(clone.health(), Err(Error::Closed)));
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    assert_eq!(kb.search(&req).unwrap().hits[0].key.id, a);
    let health = kb.health().unwrap();
    assert_eq!(health.sqlite_integrity, "ok"); assert_eq!(health.foreign_key_errors, 0);
    assert_eq!(health.record_count, health.index_document_count);
}

#[test]
fn registration_binds_validates_and_unbinds() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    let embeddings = kb.embeddings();
    // 空间没登记过就绑定回调：配置错误，直接报 not_found。
    assert!(matches!(embeddings.register_embedder("ghost", FakeEmbedder::new(2)), Err(Error::NotFound(_))));
    embeddings.register_space(EmbeddingSpace { id: "v".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "f32".into() }).unwrap();
    // 回调绑错空间：注册即校验，维度对不上直接拒绝，并报出实际维度。
    let wrong = embeddings.register_embedder("v", FakeEmbedder::new(3)).unwrap_err();
    assert!(matches!(wrong, Error::InvalidVector(_)), "{wrong}");
    assert!(wrong.to_string().contains("expected dimension 4"), "{wrong}");
    assert!(!kb.health().unwrap().embedder_spaces.contains(&"v".to_string()));
    // 条数不符、非有限值、零范数同样拒绝。
    assert!(embeddings.register_embedder("v", |texts: &[String]| Ok(vec![vec![1.0f32, 0.0, 0.0, 0.0]; texts.len() - 1])).is_err());
    assert!(embeddings.register_embedder("v", |texts: &[String]| Ok(vec![vec![f32::NAN; 4]; texts.len()])).is_err());
    assert!(embeddings.register_embedder("v", |texts: &[String]| Ok(vec![vec![0.0f32; 4]; texts.len()])).is_err());
    // 合格的回调绑定成功，且可被读出与注销。
    embeddings.register_embedder("v", FakeEmbedder::new(4)).unwrap();
    assert_eq!(embeddings.embedder_space("v").unwrap().unwrap().dimension, 4);
    assert_eq!(kb.health().unwrap().embedder_spaces, vec!["v".to_string()]);
    assert!(embeddings.unregister_embedder("v").unwrap());
    assert!(!embeddings.unregister_embedder("v").unwrap());
    assert!(kb.health().unwrap().embedder_spaces.is_empty());
    assert!(embeddings.embedder_space("ghost").unwrap().is_none());
}

#[test]
fn sync_fills_missing_vectors_incrementally_and_aborts_on_failure() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for i in 0..5 { kb.memories().upsert(memory(&format!("待向量化 {i}"), "public")).unwrap(); }
    kb.embeddings().register_space(EmbeddingSpace { id: "v".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "f32".into() }).unwrap();
    // 没有注册回调时 sync 是配置错误。
    assert!(matches!(kb.embeddings().sync("v", 4), Err(Error::Validation(_))));
    let embedder = FakeEmbedder::new(4);
    let lengths = embedder.lengths();
    kb.embeddings().register_embedder_with("v", embedder, EmbedderOptions { max_batch: 4, max_tokens_per_text: None }).unwrap();
    let report = kb.embeddings().sync("v", 32).unwrap().value;
    assert_eq!((report.scanned, report.written, report.batches), (5, 5, 2));
    assert!(report.interrupted.is_none());
    assert!(lengths.lock().unwrap().iter().all(|length| *length <= 4), "每批不得超过声明的 max_batch");
    // 已补齐再 sync：增量为零，回调不再被调用。
    let calls_after_fill = lengths.lock().unwrap().len();
    assert_eq!(kb.embeddings().sync("v", 32).unwrap().value.written, 0);
    assert_eq!(lengths.lock().unwrap().len(), calls_after_fill);
    // 检索只给搜索词：库自己嵌入查询词。
    let hits = kb.search(&vector_query("v", "待向量化 1", vec![RecordKind::Memory])).unwrap();
    assert!(hits.diagnostics.vector_used && !hits.diagnostics.text_used, "向量路自行嵌入查询词");
    assert!(hits.hits.iter().any(|hit| hit.record["judgment"] == json!("待向量化 1")), "查询词命中它对应的记录");
}

#[test]
fn sync_stops_after_the_first_failing_batch() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.embeddings().register_space(EmbeddingSpace { id: "v".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "f32".into() }).unwrap();
    // 先把记录留下待补（此刻还没回调），再注册；这样 sync 才能逐批推进并卡在第二批。
    kb.memories().upsert(memory("第一批 甲", "public")).unwrap();
    kb.memories().upsert(memory("第一批 乙", "public")).unwrap();
    kb.memories().upsert(memory("第二批 丙", "public")).unwrap();
    // 样本能过、真实语料会挂：注册成功，但 sync 到第二批中断。
    kb.embeddings().register_embedder_with("v", |texts: &[String]| {
        if texts.iter().any(|text| text.contains("第二批")) { return Err(attempt_error()); }
        Ok(texts.iter().map(|text| fallback_vector(text, 4)).collect())
    }, EmbedderOptions { max_batch: 2, max_tokens_per_text: None }).unwrap();
    let report = kb.embeddings().sync("v", 2).unwrap().value;
    // 第一批已入库，第二批未入库：中断不影响已完成的部分。
    assert_eq!(report.written, 2);
    assert!(report.interrupted.is_some());
    let mut requests = vector_query("v", "甲", vec![RecordKind::Memory]);
    requests.limit = 10;
    assert_eq!(kb.search(&requests).unwrap().hits.len(), 2, "只应有第一批的向量");
}

#[test]
fn writes_vectorize_in_place_and_notes_stay_out_by_default() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb, "v", 4);
    let mut tagged = memory("记住她喜欢苹果", "public");
    tagged.record.tags = vec!["偏好".into()];
    let id = kb.memories().upsert(tagged).unwrap().value.header.id;
    // 写入即向量化：不需要宿主调 sync，直接就能被向量路命中。
    let hits = kb.search(&vector_query("v", "记住她喜欢苹果", vec![RecordKind::Memory])).unwrap().hits;
    assert_eq!(hits[0].key.id, id);
    // tags 进该记录的向量文本：换一个只出现在 tags 里的词也能召回。
    let by_tag = kb.search(&SearchRequest { text: false, embed_space: Some("v".into()),
        kinds: vec![RecordKind::Memory], ..SearchRequest { query: "偏好".into(), ..Default::default() } }).unwrap().hits;
    assert_eq!(by_tag[0].key.id, id);
    // 笔记与切片默认不进向量：向量路看不到，全文路仍能看到。
    let note_path = dir.path().join("a.md");
    std::fs::write(&note_path, "笔记正文里的独有措辞").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&note_path)).unwrap().value;
    let vector_hits = kb.search(&vector_query("v", "笔记正文里的独有措辞", vec![RecordKind::Note, RecordKind::Chunk])).unwrap().hits;
    assert!(vector_hits.is_empty(), "笔记默认不应生成向量");
    let text_hits = kb.search(&SearchRequest { query: "独有措辞".into(), kinds: vec![RecordKind::Note, RecordKind::Chunk], ..Default::default() }).unwrap().hits;
    // 笔记正文不进索引，检索面交给切片：命中落在切片上，正文由笔记原文派生。
    let chunk_id = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap()[0].header.id;
    assert!(text_hits.iter().any(|hit| hit.key.id == chunk_id), "正文命中应落在切片上");
    assert_eq!(kb.notes().get_chunk(chunk_id, &ReadFilter::default()).unwrap().content, "笔记正文里的独有措辞");
    assert_eq!(kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap().len(), 1);
}

#[test]
fn namespace_switch_disables_vectorization_and_degrades() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb, "v", 4);
    let mut other = memory("另一个知识域的内容", "public"); other.record.namespace = "other".into();
    let other_id = kb.memories().upsert(other).unwrap().value.header.id;
    // 另一个 namespace 先关掉向量化，再写入：不生成向量，但全文照常命中。
    kb.embeddings().set_namespace_vectorization("other", false).unwrap();
    let mut muted = memory("被关闭向量化的内容", "public"); muted.record.namespace = "other".into();
    kb.memories().upsert(muted).unwrap();
    let request = SearchRequest { query: "被关闭".into(), filter: ReadFilter { namespace: "other".into(), ..Default::default() },
        embed_space: Some("v".into()), ..Default::default() };
    let result = kb.search(&request).unwrap();
    assert!(result.diagnostics.degraded.contains(&Degrade::NamespaceDisabled));
    assert_eq!(result.hits.len(), 1, "关掉向量化之后全文路仍然给结果");
    // 同库另一 namespace 不受影响。
    let default_scope = SearchRequest { query: "被关闭".into(),
        filter: ReadFilter { namespace: "default".into(), ..Default::default() }, embed_space: Some("v".into()), ..Default::default() };
    assert!(kb.search(&default_scope).unwrap().hits.is_empty());
    let mut restored = SearchRequest { ..default_scope };
    restored.filter = ReadFilter { namespace: "other".into(), ..Default::default() };
    kb.embeddings().set_namespace_vectorization("other", true).unwrap();
    assert!(kb.embeddings().namespace_vectorization("other").unwrap());
    assert_eq!(kb.embeddings().namespace_vectorization("default").unwrap(), true);
    assert_eq!(kb.memories().get(other_id, &ReadFilter { namespace: "other".into(), ..Default::default() }).unwrap().judgment, "另一个知识域的内容");
}

#[test]
fn callback_failures_still_commit_and_degrade_to_text() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.embeddings().register_space(EmbeddingSpace { id: "v".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "f32".into() }).unwrap();
    kb.embeddings().register_embedder("v", FakeEmbedder::new(4)).unwrap();
    // 写入侧：回调挂掉，记录照常写入，向量留待补齐，不抛错。
    let id = kb.memories().upsert(memory("触发降级的记忆", "public")).unwrap().value.header.id;
    assert!(kb.health().unwrap().last_degraded.contains(&Degrade::EmbedFailed));
    // 检索侧：查询词嵌入失败就退纯全文，不报错、不返回空。
    let result = kb.search(&SearchRequest { query: "触发降级".into(), embed_space: Some("v".into()), ..Default::default() }).unwrap();
    assert!(result.diagnostics.degraded.contains(&Degrade::EmbedFailed));
    assert!(!result.diagnostics.vector_used);
    assert_eq!(result.hits[0].key.id, id);
    // 目标空间没有登记过是配置错误；登记过但没绑回调才降级。
    assert!(matches!(kb.search(&SearchRequest { query: "任意".into(), embed_space: Some("ghost".into()), ..Default::default() }), Err(Error::NotFound(_))));
    kb.embeddings().unregister_embedder("v").unwrap();
    let degraded = kb.search(&SearchRequest { query: "触发降级".into(), embed_space: Some("v".into()), ..Default::default() }).unwrap();
    assert!(degraded.diagnostics.degraded.contains(&Degrade::NoEmbedder));
    assert_eq!(degraded.hits.len(), 1);
}

#[test]
fn search_parameters_control_paths_and_totals() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb, "v", 4);
    for i in 0..3 { kb.memories().upsert(memory(&format!("参数 目标 {i}"), "public")).unwrap(); }
    kb.memories().upsert(memory("无关内容", "public")).unwrap();
    let kinds = vec![RecordKind::Memory];

    let both = kb.search(&SearchRequest { query: "参数 目标".into(), kinds: kinds.clone(), limit: 3, embed_space: Some("v".into()), ..Default::default() }).unwrap();
    assert!(both.diagnostics.text_used && both.diagnostics.vector_used);
    assert_eq!(both.hits.len(), 3);
    assert_eq!(both.total, None);

    // 纯全文、不重排。
    let text_only = kb.search(&SearchRequest { query: "参数 目标".into(), kinds: kinds.clone(), limit: 3, vector: false, ..Default::default() }).unwrap();
    assert!(text_only.diagnostics.text_used && !text_only.diagnostics.vector_used);
    assert_eq!(text_only.hits.len(), 3);
    assert!(text_only.hits.iter().all(|hit| hit.rerank_score.is_none()));

    // 纯向量。
    let mut vector_only_request = vector_query("v", "参数 目标", kinds.clone());
    vector_only_request.limit = 3;
    let vector_only = kb.search(&vector_only_request).unwrap();
    assert!(!vector_only.diagnostics.text_used && vector_only.diagnostics.vector_used);
    assert_eq!(vector_only.hits.len(), 3);

    // 两条路都关掉：无路可走。
    assert!(kb.search(&SearchRequest { query: "参数".into(), text: false, vector: false, ..Default::default() }).is_err());
    // 走向量路却没给目标空间：向量路自动让位，退纯全文，不算错误。
    let no_space = kb.search(&SearchRequest { query: "参数 目标".into(), kinds: kinds.clone(), ..Default::default() }).unwrap();
    assert!(no_space.diagnostics.text_used && !no_space.diagnostics.vector_used);
    // 没有搜索词一律拒绝：宿主给的是词，不是向量。
    assert!(kb.search(&SearchRequest { query: "  ".into(), ..Default::default() }).is_err());

    // 总量是过滤之后、截断之前的匹配数：库里 4 条记忆，取 1 条命中但总量仍为 4。
    let counted = kb.search(&SearchRequest { query: "参数 目标".into(), kinds: kinds.clone(), limit: 1, with_total: true, ..Default::default() }).unwrap();
    assert_eq!(counted.hits.len(), 1);
    assert_eq!(counted.total, Some(4));
    let page = kb.memories().list(&PageRequest { filter: ReadFilter { tags: vec![], ..Default::default() }, limit: 100, ..Default::default() }).unwrap();
    assert_eq!(page.items.len(), 4, "总量与同条件分页计数一致");
}

#[test]
fn reranker_reorders_candidates_and_enforces_length_limits() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for tag in ["甲", "乙", "丙", "丁", "戊"] { kb.memories().upsert(memory(&format!("重排目标 {tag}"), "public")).unwrap(); }
    let request = SearchRequest { query: "重排目标".into(), limit: 5, kinds: vec![RecordKind::Memory], vector: false, ..Default::default() };
    let baseline: Vec<i64> = kb.search(&request).unwrap().hits.iter().map(|hit| hit.key.id).collect();
    assert_eq!(baseline.len(), 5);

    // 定长约束：候选按 RRF 顺序截到 max_docs，文档按 token 预算截断。
    let seen: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    kb.register_reranker_with(move |_: &str, documents: &[String]| {
        recorder.lock().unwrap().push((documents.len(), documents.iter().map(|d| d.chars().count()).max().unwrap_or(0)));
        // 故意倒序给分：最后一个候选拿最高分。
        Ok((0..documents.len()).map(|i| i as f32).collect())
    }, RerankerOptions { max_docs: 2, max_tokens_per_doc: 6, max_tokens_query: None }).unwrap();
    // 注册校验已经调用过回调一次，清掉只留检索那次。
    seen.lock().unwrap().clear();
    let reranked = kb.search(&SearchRequest { rerank: true, ..request.clone() }).unwrap();
    assert!(reranked.diagnostics.reranked);
    assert_eq!(reranked.diagnostics.rerank_candidates, 2);
    assert_eq!(reranked.diagnostics.rerank_truncated, 3);
    assert_eq!(reranked.hits[0].key.id, baseline[1], "倒序给分后原来的第二名排到最前");
    assert!(reranked.hits.iter().all(|hit| hit.rerank_score.is_some()));
    assert_eq!(*seen.lock().unwrap(), vec![(2, 3)], "回调实收 2 条候选、每条按 6 token 预算截断");

    // 注销之后 rerank 开关被忽略，结果与不重排一致。
    assert!(kb.unregister_reranker());
    let ignored = kb.search(&SearchRequest { rerank: true, ..request.clone() }).unwrap();
    assert!(!ignored.diagnostics.reranked);
    assert_eq!(ignored.hits.iter().map(|hit| hit.key.id).collect::<Vec<_>>(), baseline);
}

#[test]
fn reranker_registration_validates_and_failures_degrade() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for i in 0..3 { kb.memories().upsert(memory(&format!("降级目标 {i}"), "public")).unwrap(); }
    // 返回条数与输入不符、分数非有限值都拒绝绑定。
    assert!(kb.register_reranker(|_: &str, _: &[String]| Ok(vec![1.0f32])).is_err());
    assert!(kb.register_reranker(|_: &str, documents: &[String]| Ok(vec![f32::NAN; documents.len()])).is_err());
    assert!(!kb.reranker_registered());
    kb.register_reranker(|_: &str, _: &[String]| Err("重排服务不可用".to_string())).unwrap();
    assert!(kb.reranker_registered());
    let request = SearchRequest { query: "降级目标".into(), kinds: vec![RecordKind::Memory], vector: false, ..Default::default() };
    let result = kb.search(&request).unwrap();
    assert!(!result.diagnostics.reranked);
    assert!(result.diagnostics.degraded.contains(&Degrade::RerankFailed));
    assert_eq!(result.hits.len(), 3, "重排挂了也要给结果，只是按融合分排序");
}

#[test]
fn oversized_batches_shrink_to_fit() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.embeddings().register_space(EmbeddingSpace { id: "v".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "f32".into() }).unwrap();
    for i in 0..8 { kb.memories().upsert(memory(&format!("批次 {i}"), "public")).unwrap(); }
    let lengths: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = lengths.clone();
    // 声明 8 条，但真实模型只吃 4 条：库应当自己减半并沿用。
    kb.embeddings().register_embedder_with("v", move |texts: &[String]| {
        recorder.lock().unwrap().push(texts.len());
        if texts.len() > 4 { return Err(EmbedCallbackError::too_large("at most 4")); }
        Ok(texts.iter().map(|text| fallback_vector(text, 4)).collect())
    }, EmbedderOptions { max_batch: 8, max_tokens_per_text: None }).unwrap();
    let report = kb.embeddings().sync("v", 32).unwrap().value;
    assert_eq!(report.written, 8);
    let calls = lengths.lock().unwrap().clone();
    assert_eq!(&calls[calls.len() - 2..], &[4, 4], "减半后沿用 4，不再从 8 重试");
}

#[test]
fn swapping_models_keeps_the_old_space_usable() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for i in 0..3 { kb.memories().upsert(memory(&format!("换模型 {i}"), "public")).unwrap(); }
    space(&kb, "old", 4);
    kb.embeddings().sync("old", 32).unwrap();
    // 换模型：新建空间 + 新回调 + sync；旧空间不动，退回只是检索改回旧空间。
    kb.embeddings().register_space(EmbeddingSpace { id: "new".into(), model: "fixture/v2".into(), dimension: 8, text_version: 1, encoding: "sq8".into() }).unwrap();
    kb.embeddings().register_embedder("new", FakeEmbedder::new(8)).unwrap();
    assert_eq!(kb.embeddings().sync("new", 32).unwrap().value.written, 3);
    let kinds = vec![RecordKind::Memory];
    let old_hits = kb.search(&vector_query("old", "换模型 1", kinds.clone())).unwrap().hits;
    let new_hits = kb.search(&vector_query("new", "换模型 1", kinds.clone())).unwrap().hits;
    assert_eq!(old_hits.len(), 3); assert_eq!(new_hits.len(), 3);
    assert_eq!(old_hits[0].key.id, new_hits[0].key.id);
    let mut request = vector_query("new", "换模型 1", kinds);
    request.limit = 10;
    let mut ids: Vec<i64> = kb.search(&request).unwrap().hits.iter().map(|hit| hit.key.id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3]);
    // 空间不可变：同 id 改模型直接冲突，旧向量永远对得上旧模型。
    assert!(kb.embeddings().register_space(EmbeddingSpace { id: "old".into(), model: "fixture/v9".into(), dimension: 4, text_version: 1, encoding: "f32".into() }).is_err());
}

#[test]
fn vector_partitions_are_isolated_by_scope() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    // 两个 scope 各写一条方向不同的向量，用分数判断命中的是哪个分区。
    // 记忆的向量文本是「论断\n标签」，所以表按这个形状给。
    kb.embeddings().register_space(EmbeddingSpace { id: "v".into(), model: "fixture/v1".into(), dimension: 2, text_version: 1, encoding: "f32".into() }).unwrap();
    kb.embeddings().register_embedder("v", FakeEmbedder::with_table(2, &[
        ("public vector\n", vec![1.0, 0.0]),
        ("private vector\n", vec![0.0, 1.0]),
        ("probe", vec![1.0, 0.0]),
    ])).unwrap();
    let public_id = kb.memories().upsert(memory("public vector", "public")).unwrap().value.header.id;
    let private_id = kb.memories().upsert(memory("private vector", "private")).unwrap().value.header.id;
    let query = |scope: &str| SearchRequest {
        query: "probe".into(), text: false, embed_space: Some("v".into()),
        filter: ReadFilter { scopes: vec![scope.into()], ..Default::default() },
        ..Default::default()
    };
    let public = kb.search(&query("public")).unwrap().hits;
    assert_eq!(public.len(), 1); assert_eq!(public[0].key.id, public_id);
    // private 查询只能看到 private 分区，public 的向量不能串进来。
    let private = kb.search(&query("private")).unwrap().hits;
    assert!(private.iter().all(|hit| hit.key.id == private_id), "public 分区的向量混进了 private 查询");
    // 空分区（无任何向量的 scope）按空结果缓存，不应报错，也不该回退到别的分区。
    assert!(kb.search(&query("empty")).unwrap().hits.is_empty());
}

#[test]
fn packed_and_precise_spaces_rank_the_same_vectors_together() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for i in 0..8 { kb.memories().upsert(memory(&format!("记录{i}"), "public")).unwrap(); }
    for (id, encoding) in [("precise", "f32"), ("packed", "sq8")] {
        kb.embeddings().register_space(EmbeddingSpace { id: id.into(), model: "fixture/v1".into(), dimension: 8, text_version: 1, encoding: encoding.into() }).unwrap();
        kb.embeddings().register_embedder(id, FakeEmbedder::new(8)).unwrap();
        assert_eq!(kb.embeddings().sync(id, 50).unwrap().value.written, 8);
    }
    let kinds = vec![RecordKind::Memory];
    let precise = kb.search(&vector_query("precise", "记录3", kinds.clone())).unwrap().hits;
    let packed = kb.search(&vector_query("packed", "记录3", kinds)).unwrap().hits;
    let precise_ids: Vec<i64> = precise.iter().map(|h| h.key.id).collect();
    let packed_ids: Vec<i64> = packed.iter().map(|h| h.key.id).collect();
    assert_eq!(precise_ids.len(), 8);
    assert_eq!(precise_ids, packed_ids, "sq8 空间的名次应与 f32 空间一致");
    // 分数只允许有量化误差量级的偏差：库侧与查询侧各量化一次，8 维下步长约百分之一。
    for (a, b) in precise.iter().zip(packed.iter()) {
        let (x, y) = (a.vector_scores["precise"], b.vector_scores["packed"]);
        assert!((x - y).abs() < 5e-3, "{x} vs {y}");
    }
}

#[test]
fn concurrent_reads_are_isolated_and_agree_with_serial_results() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    // 各 scope 写入互不相同的记录，读线程只能看到自己 scope 的数据。
    for (i, scope) in (0..8).map(|i| (i, format!("scope{i}"))) {
        let mut value = memory(&format!("内容 {i} 独有"), &scope);
        value.record.tags = vec![format!("tag{i}")];
        kb.memories().upsert(value).unwrap();
    }
    let expected: Vec<usize> = (0..8).map(|i| {
        kb.search(&SearchRequest { query: format!("内容 {i}"), filter: ReadFilter { scopes: vec![format!("scope{i}")], ..Default::default() },
            limit: 10, ..Default::default() }).unwrap().hits.len()
    }).collect();
    assert!(expected.iter().all(|n| *n == 1));
    let mut handles = Vec::new();
    for round in 0..24 {
        let kb = kb.clone();
        let expected = expected.clone();
        handles.push(std::thread::spawn(move || {
            let i = round % 8;
            let request = SearchRequest { query: format!("内容 {i}"),
                filter: ReadFilter { namespace: default_namespace(), scopes: vec![format!("scope{i}")], tags: vec![format!("tag{i}")] },
                limit: 10, ..Default::default() };
            for _ in 0..40 {
                let hits = kb.search(&request).unwrap().hits;
                // 串行下每个 scope 恰好一条，并发下必须一致：既不能丢，也不能串进别的域。
                assert_eq!(hits.len(), expected[i], "并发读结果与串行不一致");
                assert_eq!(hits[0].record["scope"], format!("scope{i}"));
            }
        }));
    }
    for handle in handles { handle.join().unwrap(); }
    // 读线程全部结束后仍然可以正常写入。
    kb.memories().upsert(memory("收尾", "scope0")).unwrap();
    assert_eq!(kb.health().unwrap().record_count, 9);
}

/// 在固定轮数下用 `threads` 个线程检索，返回耗时。
fn read_rounds(kb: &KnowledgeBase, request: &SearchRequest, threads: usize, rounds: usize) -> std::time::Duration {
    let start = std::time::Instant::now();
    let mut readers = Vec::new();
    for _ in 0..threads {
        let kb = kb.clone();
        let request = request.clone();
        readers.push(std::thread::spawn(move || {
            for _ in 0..rounds { assert!(!kb.search(&request).unwrap().hits.is_empty()); }
        }));
    }
    for reader in readers { reader.join().unwrap(); }
    start.elapsed()
}

#[test]
fn concurrent_readers_are_not_blocked_by_a_writer() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.memories().upsert(memory("初始内容", "public")).unwrap();
    // 写入不再就地索引：先把基线追平，之后写者产生的待办由读者自愈或被忽略都不影响这条。
    kb.update_index().unwrap();
    let request = SearchRequest { query: "初始内容".into(), limit: 5, ..Default::default() };
    // 读路径曾经这样退化成串行：待索引队列非空时读者去抢写锁，而写者每次索引提交要二十毫秒量级，
    // 于是读者全排到写者后面，吞吐掉到基线的百分之七。这里用同一个读者组在写者存在前后各跑一遍，
    // 用耗时比钉住「读者不等写者」，只断言数量级，不追求具体数值。
    let quiet = read_rounds(&kb, &request, 4, 60);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = stop.clone();
    let writer_kb = kb.clone();
    let writer = std::thread::spawn(move || {
        let mut n = 0usize;
        while !flag.load(std::sync::atomic::Ordering::Relaxed) {
            writer_kb.memories().upsert(memory(&format!("并发写入 {n}"), "public")).unwrap();
            n += 1;
        }
        n
    });
    let busy = read_rounds(&kb, &request, 4, 60);
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let written = writer.join().unwrap();
    assert!(written > 0, "写者在读者跑完前一条都没写成，这次比较没有意义");
    assert!(busy.as_secs_f64() < quiet.as_secs_f64() * 4.0,
        "写者进行中读者不应排队等锁：无写入 {quiet:?}，有写入 {busy:?}（写者写入 {written} 条）");
}

#[test]
fn graph_integrity_aliases_and_rename_propagation() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
    let mut alice = entity("Alice"); alice.aliases = vec!["小艾".into()];
    let mut bob = entity("Bob"); bob.aliases = vec!["小艾".into()];
    let entities = kb.graph().apply_batch(&GraphBatch { entities: vec![alice, bob], ..Default::default() }).unwrap().value.entities;
    let a = entities[0].header.id; let b = entities[1].header.id;
    let relation = RelationInput { record: RecordInput::default(), subject_id: a, predicate: "knows".into(), object_id: b, confidence: 0.8, reason: String::new() };
    let event = EventInput { record: RecordInput::default(), name: "meeting".into(), summary: String::new(), participants: vec![a, b], confidence: 0.9, reason: String::new() };
    kb.graph().apply_batch(&GraphBatch { entities: vec![], relations: vec![relation.clone()], events: vec![event] }).unwrap();
    assert_eq!(kb.graph().resolve("小艾", &ReadFilter::default(), 10).unwrap().len(), 2);
    assert_eq!(kb.graph().neighbors(a, &ReadFilter::default(), 10).unwrap().entities[0].name, "Bob");
    assert_eq!(kb.graph().events_for_entity(a, &ReadFilter::default(), 10).unwrap().len(), 1);
    assert!(matches!(kb.graph().delete(RecordKind::Entity, a, &ReadFilter::default()), Err(Error::Conflict(_))));
    // 改名会改写引用它的关系与事件正文，这些记录必须重新生成向量（旧向量已随指纹作废）。
    let mut renamed = entity("Carol"); renamed.record.id = Some(a);
    kb.graph().apply_batch(&GraphBatch { entities: vec![renamed], ..Default::default() }).unwrap();
    let mut request = vector_query("v", "Carol", vec![RecordKind::Relation, RecordKind::Event]);
    request.limit = 10;
    let hits = kb.search(&request).unwrap().hits;
    assert_eq!(hits.len(), 2, "关系与事件在改名后应当带着新向量回到向量路");
    let search = SearchRequest {query:"Carol".into(),kinds:vec![RecordKind::Relation,RecordKind::Event],..Default::default()};
    assert_eq!(kb.search(&search).unwrap().hits.len(), 2);
    let bad = RelationInput { object_id: 9_999_999, ..relation };
    assert!(kb.graph().apply_batch(&GraphBatch { entities: vec![], relations: vec![bad], events: vec![] }).is_err());
}

#[test]
fn note_replacement_keeps_evidence_and_removes_stale_chunks() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    let tea_path = dir.path().join("tea.md");
    std::fs::write(&tea_path, "# 茶\n\n上海喝茶\n\n```rust\nlet x = 1;\n```\n\n最后一段").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&tea_path)).unwrap().value;
    let note_id = note.header.id;
    let chunks = kb.notes().chunks(note_id, &ReadFilter::default()).unwrap();
    let tea = chunks.iter().find(|c| c.content == "上海喝茶").unwrap();
    assert_eq!((tea.offset, tea.limit), (3, 1));
    assert!(chunks.iter().any(|c| c.content.contains("```rust") && c.offset == 5 && c.limit == 3));
    let evidence = Evidence { source: "docs/tea.md".into(), offset: Some(3), limit: Some(1), quote: "上海喝茶".into(), ..Default::default() };
    let mut m = memory("source fact", "public"); m.record.evidence = vec![evidence];
    let memory_id = kb.memories().upsert(m).unwrap().value.header.id;
    std::fs::write(&tea_path, "replacement").unwrap();
    let new = kb.notes().upsert_file(NoteFileInput::new(&tea_path)).unwrap().value;
    assert_eq!(new.header.id, note_id);
    assert!(kb.notes().get_chunk(tea.header.id, &ReadFilter::default()).is_err());
    assert_eq!(kb.memories().get(memory_id, &ReadFilter::default()).unwrap().header.evidence[0].quote, "上海喝茶");
    kb.notes().delete(note_id, &ReadFilter::default()).unwrap();
    assert_eq!(kb.health().unwrap().record_count, 1);
    assert_eq!(kb.health().unwrap().foreign_key_errors, 0);
    let long = chunk_text(&"一".repeat(500), 220).unwrap();
    assert_eq!(long.len(), 3); assert!(long.iter().all(|c| c.offset == 1 && c.limit == 1));
}

#[test]
fn explicit_lifecycle_has_no_hidden_deletion() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    let mut weak = memory("weak", "public"); weak.record.created_at_us = Some(0); weak.record.updated_at_us = Some(0);
    let weak_id = kb.memories().upsert(weak).unwrap().value.header.id;
    let mut pinned = memory("pinned", "public"); pinned.state = Some(MemoryState { pinned: true, ..Default::default() });
    kb.memories().upsert(pinned).unwrap();
    let result = kb.memories().decay(&ReadFilter::default(), &DecayPolicy::default(), Some(4*86_400_000_000)).unwrap().value;
    assert_eq!(result.decayed,1); assert_eq!(result.retirement_candidates[0].id, weak_id); assert_eq!(kb.health().unwrap().record_count,2);
    let feedback = FeedbackRequest {recalled_ids:vec![weak_id],useful_ids:vec![weak_id],now_us:Some(5*86_400_000_000),..Default::default()};
    kb.memories().feedback(&feedback).unwrap(); assert_eq!(kb.memories().get(weak_id, &ReadFilter::default()).unwrap().state.strength,1);
    let bad = FeedbackRequest {useful_ids:vec![9_999_999],..Default::default()}; assert!(kb.memories().feedback(&bad).is_err());
    let second = kb.memories().decay(&ReadFilter::default(), &DecayPolicy::default(), Some(5*86_400_000_000)).unwrap(); assert_eq!(second.value.decayed,0);
}

#[test]
fn backup_restore_and_derived_index_recovery() {
    let root=tempfile::tempdir().unwrap();let data=root.path().join("data");let kb=KnowledgeBase::open(&data).unwrap();
    kb.memories().upsert(memory("recoverable","public")).unwrap();space(&kb,"v",2);
    let vector_id = kb.memories().upsert(memory("带向量的记录","public")).unwrap().value.header.id;
    assert_eq!(kb.search(&vector_query("v", "带向量的记录", vec![RecordKind::Memory])).unwrap().hits[0].key.id, vector_id);
    let backup=root.path().join("backup.sqlite3");kb.backup(&backup).unwrap();assert!(kb.backup(&backup).is_err());kb.close().unwrap();
    std::fs::write(data.join("text-v2/meta.json"),b"broken index metadata").unwrap();
    let reopened=KnowledgeBase::open(&data).unwrap();assert_eq!(reopened.search(&SearchRequest {query:"recoverable".into(),..Default::default()}).unwrap().hits.len(),1);
    let restored=KnowledgeBase::restore(&backup,root.path().join("restored")).unwrap();
    assert_eq!(restored.health().unwrap().record_count,2);
    // 快照里带着 embeddings：恢复后重新注册同一回调，向量路立刻可用。
    space(&restored,"v",2);
    assert_eq!(restored.search(&vector_query("v", "带向量的记录", vec![RecordKind::Memory])).unwrap().hits[0].key.id, vector_id);
    assert!(KnowledgeBase::restore(&backup,&data).is_err());
    assert_eq!(json!(restored.health().unwrap())["sqlite_integrity"],"ok");
}

#[test]
fn ties_and_chinese_queries_are_deterministic_across_rebuilds() {
    let dir=tempfile::tempdir().unwrap();let kb=KnowledgeBase::open(dir.path()).unwrap();
    let values:Vec<_>=(0..140).map(|_| memory("上海 茶 相同", "public")).collect();kb.memories().upsert_many(&values).unwrap();
    let q=SearchRequest {query:"上海 茶".into(),limit:3,candidate_limit:Some(3),..Default::default()};
    let ids=|kb:&KnowledgeBase|kb.search(&q).unwrap().hits.into_iter().map(|h|h.key.id).collect::<Vec<_>>();
    let first=ids(&kb);assert_eq!(first.len(),3);assert!(first.windows(2).all(|w|w[0]<w[1]));
    kb.rebuild_indexes().unwrap();assert_eq!(ids(&kb),first);
}

#[test]
fn exact_keywords_recall_tags_and_parent_dirs() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    // 正文不含"星见雅"，只有 tag 带它：靠精确整词关键字召回。
    let mut m = memory("她喜欢苹果", "public");
    m.record.tags = vec!["星见雅".into()];
    let id = kb.memories().upsert(m).unwrap().value.header.id;
    let hits = kb.search(&SearchRequest { query: "星见雅".into(), ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key.id, id);
    // 正文不含"角色"，只有 source 的父目录带它：靠父目录整词关键字召回切片。
    let note_dir = dir.path().join("绝区零").join("角色");
    std::fs::create_dir_all(&note_dir).unwrap();
    let note_path = note_dir.join("雅.md");
    std::fs::write(&note_path, "这是一段无关内容").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&note_path)).unwrap().value;
    let hits = kb.search(&SearchRequest { query: "角色".into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
    assert!(hits.iter().any(|h| h.record["note_id"] == json!(note.header.id)));
}

#[test]
fn sq8_encoding_roundtrips_through_storage() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.embeddings().register_space(EmbeddingSpace { id: "q".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "sq8".into() }).unwrap();
    assert!(kb.embeddings().register_space(EmbeddingSpace { id: "bad".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "q4".into() }).is_err());
    // 归一化后 [0.6, 0.8, 0, 0]；sq8 往返后仍应被同一查询命中。
    kb.embeddings().register_embedder("q", FakeEmbedder::with_table(4, &[
        ("quantized target\n", vec![3.0, 4.0, 0.0, 0.0]),
        ("probe", vec![0.6, 0.8, 0.0, 0.0]),
    ])).unwrap();
    let id = kb.memories().upsert(memory("quantized target", "public")).unwrap().value.header.id;
    let hits = kb.search(&vector_query("q", "probe", vec![RecordKind::Memory])).unwrap().hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key.id, id);
    assert!((hits[0].score - 1.0 / 61.0).abs() < 1e-9);
}

#[test]
fn graph_prune_limits_vector_scoring_to_neighborhood() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb, "v", 2);
    let ents = kb
        .graph()
        .apply_batch(&GraphBatch { entities: vec![entity("A"), entity("B"), entity("C")], ..Default::default() })
        .unwrap()
        .value
        .entities;
    let (a, b) = (ents[0].header.id, ents[1].header.id);
    // 只连 A--knows-->B，C 孤立
    kb.graph()
        .apply_batch(&GraphBatch { relations: vec![RelationInput { record: RecordInput::default(), subject_id: a, predicate: "knows".into(), object_id: b, confidence: 1.0, reason: String::new() }], ..Default::default() })
        .unwrap();
    // 三个实体写同一向量：全量检索会命中 3 条
    let base = vector_query("v", "probe", vec![RecordKind::Entity]);
    let mut wide = base.clone(); wide.limit = 10;
    assert_eq!(kb.search(&wide).unwrap().hits.len(), 3);

    // 剪枝到 A 的 1 跳邻域：只剩 A（起点）与 B，孤立的 C 被排除
    let pruned = SearchRequest { prune: Some(GraphPrune { root: a, depth: 1, limit: 64 }), limit: 10, ..base.clone() };
    let hits = kb.search(&pruned).unwrap().hits;
    let ids: Vec<i64> = hits.iter().map(|h| h.key.id).collect();
    assert_eq!(hits.len(), 2);
    assert!(ids.contains(&a) && ids.contains(&b));

    // depth 必须至少为 1
    assert!(kb.search(&SearchRequest { prune: Some(GraphPrune { root: a, depth: 0, limit: 8 }), ..base.clone() }).is_err());
}

#[test]
fn search_with_context_attaches_entity_neighborhood() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let mut suspect = entity("张三");
    suspect.aliases = vec!["嫌疑人".into()];
    let ents = kb
        .graph()
        .apply_batch(&GraphBatch { entities: vec![suspect, entity("李四")], ..Default::default() })
        .unwrap()
        .value
        .entities;
    let (zhang, li) = (ents[0].header.id, ents[1].header.id);
    kb.graph()
        .apply_batch(&GraphBatch { relations: vec![RelationInput { record: RecordInput::default(), subject_id: zhang, predicate: "knows".into(), object_id: li, confidence: 1.0, reason: String::new() }], ..Default::default() })
        .unwrap();

    // 这条记忆的 tag 命中实体的别名「嫌疑人」，因此挂载到张三
    let mut note = memory("深夜的会面记录", "public");
    note.record.tags = vec!["嫌疑人".into()];
    let linked_id = kb.memories().upsert(note).unwrap().value.header.id;
    // 另一条没有挂载任何实体
    let plain_id = kb.memories().upsert(memory("深夜的另一次会面", "public")).unwrap().value.header.id;

    let request = SearchRequest { query: "深夜 会面".into(), kinds: vec![RecordKind::Memory], ..Default::default() };
    let hits = kb.search_with_context(&request, 10).unwrap();
    assert_eq!(hits.len(), 2);
    let linked = hits.iter().find(|h| h.hit.key.id == linked_id).unwrap();
    let names: Vec<&str> = linked.context.entities.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"张三") && names.contains(&"李四"));
    assert_eq!(linked.context.relations.len(), 1);
    let plain = hits.iter().find(|h| h.hit.key.id == plain_id).unwrap();
    assert!(plain.context.entities.is_empty() && plain.context.relations.is_empty());
}

#[test]
fn batched_context_matches_per_hit_expansion() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let mut a = entity("甲真人");
    a.aliases = vec!["甲".into()];
    let mut b = entity("乙真人");
    b.aliases = vec!["乙".into()];
    let c = entity("丙真人");
    // 别名与甲相同、但落在另一个 scope：不该被挂到 public 的记忆上（同名不等于同一实体）。
    let mut shadow = entity("甲分身");
    shadow.aliases = vec!["甲".into()];
    shadow.record.scope = "private".into();
    let ents = kb.graph().apply_batch(&GraphBatch { entities: vec![a, b, c, shadow], ..Default::default() }).unwrap().value.entities;
    let (a_id, b_id, c_id, shadow_id) = (ents[0].header.id, ents[1].header.id, ents[2].header.id, ents[3].header.id);
    let relation = |subject_id: i64, object_id: i64, predicate: &str| RelationInput {
        record: RecordInput::default(), subject_id, object_id, predicate: predicate.into(), confidence: 1.0, reason: String::new(),
    };
    kb.graph().apply_batch(&GraphBatch { relations: vec![relation(a_id, b_id, "认识"), relation(a_id, c_id, "同伙")], ..Default::default() }).unwrap();

    let mut first = memory("共同出现的甲", "public"); first.record.tags = vec!["甲".into()];
    let first_id = kb.memories().upsert(first).unwrap().value.header.id;
    let mut second = memory("共同出现的乙", "public"); second.record.tags = vec!["乙".into()];
    let second_id = kb.memories().upsert(second).unwrap().value.header.id;

    let request = SearchRequest { query: "共同出现".into(), kinds: vec![RecordKind::Memory], ..Default::default() };
    let hits = kb.search_with_context(&request, 1).unwrap();
    assert_eq!(hits.len(), 2);
    let first_ctx = &hits.iter().find(|h| h.hit.key.id == first_id).unwrap().context;
    // limit=1：甲的两条关系各自按 limit 截断后只留一条。
    assert_eq!(first_ctx.relations.len(), 1);
    let names: Vec<&str> = first_ctx.entities.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"甲真人") && (names.contains(&"乙真人") || names.contains(&"丙真人")), "邻域应带回种子实体与端点实体");
    assert!(!names.contains(&"甲分身"), "跨 scope 的同名实体不该被挂上");
    // 放宽 limit 后两条关系都回来了，跨 scope 实体依旧不出现。
    let wide = kb.search_with_context(&request, 10).unwrap();
    let wide_first = &wide.iter().find(|h| h.hit.key.id == first_id).unwrap().context;
    assert_eq!(wide_first.relations.len(), 2);
    let names: Vec<&str> = wide_first.entities.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"甲真人") && names.contains(&"乙真人") && names.contains(&"丙真人"));
    assert!(!names.contains(&"甲分身"));
    // 命中的第二条记忆只跟着乙这一跳。
    let second_ctx = &wide.iter().find(|h| h.hit.key.id == second_id).unwrap().context;
    assert_eq!(second_ctx.relations.len(), 1);
    assert_eq!(second_ctx.relations[0].predicate, "认识");
    assert!(kb.graph().get(RecordKind::Entity, shadow_id, &ReadFilter { scopes: vec!["private".into()], ..Default::default() }).is_ok());
}

#[test]
fn text_gate_excludes_records_that_do_not_match_the_query() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    for i in 0..20 { kb.memories().upsert(memory(&format!("记忆 {i} 独有措辞"), "public")).unwrap(); }
    // 过滤条件命中全部 20 条。正文条件必须独立生效：正文里根本不存在的词一律零命中。
    // 曾经正文子句与过滤子句同层，同层存在 Must 时 Should 降级为可选，于是一次「查不到的词」
    // 会返回该过滤域下的任意记录，检索退化成「只按过滤条件取记录」。
    let missing = kb.search(&SearchRequest { query: "量子纠缠zzz".into(), limit: 10, ..Default::default() }).unwrap().hits;
    assert!(missing.is_empty(), "正文条件失效：过滤域内的 {} 条记录被当成了命中", missing.len());
    // 只存在于一条记录里的词，必须只把那条取回来。
    let target = kb.memories().upsert(memory("全息投影仪维修记录", "public")).unwrap().value.header.id;
    let hits = kb.search(&SearchRequest { query: "全息投影仪".into(), limit: 10, ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key.id, target);
}

#[test]
fn note_upsert_file_reads_path_uses_stem_and_keeps_raw_text() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    let path = dir.path().join("世界观.md");
    let raw = "# 标题\n\n这里有 **独有措辞** 正文。";
    std::fs::write(&path, raw).unwrap();

    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    assert_eq!(note.title, "世界观", "标题取文件名");
    assert_eq!(note.source, *path.to_string_lossy(), "路径即身份");
    // 正文不落库：切片正文读时由文件原文派生。
    let chunks = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap();
    assert!(chunks.iter().any(|c| c.content.contains("独有措辞")), "切片正文由文件原文派生");

    // 索引在分词前清洗：标记符不干扰，正文词照常命中切片。
    let hits = kb.search(&SearchRequest { query: "独有措辞".into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
    assert!(!hits.is_empty(), "清洗后的文本仍可检索");

    // 同一路径重新同步：定位到同一笔记、更新正文、重建切片。
    std::fs::write(&path, "改过的正文 **新词** 在这里。").unwrap();
    let updated = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    assert_eq!(updated.header.id, note.header.id, "同一路径复用同一笔记");
    let hits = kb.search(&SearchRequest { query: "新词".into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
    assert!(!hits.is_empty(), "更新后的正文进入索引");

    // 文件不存在与非 UTF-8 都直接报错，不落库。
    assert!(kb.notes().upsert_file(NoteFileInput::new(dir.path().join("nope.md"))).is_err());
    let bad = dir.path().join("bad.md");
    std::fs::write(&bad, [0xffu8, 0xfe, 0xfd]).unwrap();
    assert!(kb.notes().upsert_file(NoteFileInput::new(&bad)).is_err());
}

/// 笔记正文只在宿主文件里：库内任何文本列都不留副本，切片正文读时从文件取回。
#[test]
fn note_body_stays_in_the_file_not_in_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let path = dir.path().join("note.md");
    std::fs::write(&path, "ZZBODYMARK 独有正文标记").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;

    // 显式读切片正文：由文件原文派生。
    let chunks = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap();
    assert!(chunks.iter().any(|chunk| chunk.content.contains("ZZBODYMARK")));
    drop(kb);

    // 库内所有文本列都不含正文标记。
    let conn = rusqlite::Connection::open(dir.path().join("store.sqlite3")).unwrap();
    let mut stmt = conn.prepare("SELECT payload_json||metadata_json||evidence_json FROM records").unwrap();
    let hits = stmt.query_map([], |row| row.get::<_, String>(0)).unwrap()
        .filter(|row| row.as_ref().unwrap().contains("ZZBODYMARK")).count();
    assert_eq!(hits, 0, "笔记正文不落 SQLite");
}

/// 文件缺失时：读正文（显式读取与全量重建索引）直接报错，不静默兜底——
/// 按分工这是一致性被破坏，由上游主动删除记录来消除。
#[test]
fn missing_source_file_surfaces_as_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let path = dir.path().join("gone.md");
    std::fs::write(&path, "会被删掉的正文").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    let chunk_id = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap()[0].header.id;
    std::fs::remove_file(&path).unwrap();

    assert!(kb.notes().get_chunk(chunk_id, &ReadFilter::default()).is_err(), "文件缺失时显式读正文报错");
    assert!(kb.rebuild_indexes().is_err(), "全量重建索引遇到缺失文件报错");
}
