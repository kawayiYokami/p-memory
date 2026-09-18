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

/// 假嵌入：可以指定「文本 → 向量」表，未列出的走确定性兜底。
/// 每批记下「调用发生在哪条线程、实收几条」——库内补齐线程与调用方线程要分得开。
struct FakeEmbedder {
    dimension: usize,
    table: Arc<Mutex<BTreeMap<String, Vec<f32>>>>,
    calls: Arc<Mutex<Vec<(String, usize)>>>,
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
    fn lengths(&self) -> Arc<Mutex<Vec<(String, usize)>>> { self.calls.clone() }
}

impl Embedder for FakeEmbedder {
    fn embed(&mut self, texts: &[String]) -> std::result::Result<Vec<Vec<f32>>, EmbedCallbackError> {
        let caller = std::thread::current().name().unwrap_or_default().to_string();
        self.calls.lock().unwrap().push((caller, texts.len()));
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
/// 批次结束之后调用方触发的那次补齐。补完缺口，该档才会被标成就绪、向量路才放行。
fn fill(kb: &KnowledgeBase, space_id: &str) -> SyncReport { kb.embeddings().sync(space_id, 32).unwrap().value }
/// 等某一档补齐。补齐由库内线程主动触发，所以这里只等结果，不假定是谁补的。
fn wait_until_ready(kb: &KnowledgeBase, namespace: &str, space_id: &str, target: &str, budget_ms: u64) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(budget_ms);
    while std::time::Instant::now() < deadline {
        if kb.embeddings().vector_ready(namespace, space_id, target).unwrap() { return true; }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    false
}
/// 假回调在某条线程上的调用次数。补齐线程与自己这条线程要分开算：
/// 「写入不碰模型」这件事只能从调用方线程上看。
fn calls_from(calls: &Arc<Mutex<Vec<(String, usize)>>>, thread: &str) -> usize {
    calls.lock().unwrap().iter().filter(|(caller, _)| caller == thread).count()
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
    // 注册回调这个动作自己就触发一次补齐；这里不假定是谁补的，只等它补完。
    assert!(wait_until_ready(&kb, "default", "v", "memory", 15_000));
    assert!(lengths.lock().unwrap().iter().all(|(_, length)| *length <= 4), "每批不得超过声明的 max_batch");
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
    // 样本能过、真实语料会挂：注册成功，但「第二批 丙」永远补不上。
    kb.embeddings().register_embedder_with("v", |texts: &[String]| {
        if texts.iter().any(|text| text.contains("第二批")) { return Err(attempt_error()); }
        Ok(texts.iter().map(|text| fallback_vector(text, 4)).collect())
    }, EmbedderOptions { max_batch: 2, max_tokens_per_text: None }).unwrap();
    // 有补不上的记录：记忆档一直停在未就绪，它的向量不许参与打分。
    assert!(!wait_until_ready(&kb, "default", "v", "memory", 1_000), "补不上的记录让记忆档停在未就绪");
    assert!(!kb.embeddings().vector_ready("default", "v", "memory").unwrap());
    let gated = kb.search(&vector_query("v", "甲", vec![RecordKind::Memory])).unwrap();
    assert!(gated.hits.is_empty() && gated.diagnostics.degraded.contains(&Degrade::VectorNotReady));
    // 中断不影响已完成的批次：稳定之后再 sync，能补的都已补过，没有可写的增量。
    let settled = kb.embeddings().sync("v", 2).unwrap().value;
    assert_eq!(settled.written, 0);
    assert!(settled.interrupted.is_some());
    // 换一条能用的回调：上次补好的向量留着不重算，补完才放行。
    kb.embeddings().register_embedder("v", FakeEmbedder::new(4)).unwrap();
    assert!(wait_until_ready(&kb, "default", "v", "memory", 15_000), "换成能用的回调后补完即就绪");
    let mut requests = vector_query("v", "甲", vec![RecordKind::Memory]);
    requests.limit = 10;
    assert_eq!(kb.search(&requests).unwrap().hits.len(), 3, "三条都补上了");
}

#[test]
fn writes_stay_clean_and_the_batch_end_sync_fills_vectors() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    // 空间先登记好（登记不算有模型）；回调稍后再接，用来证明写入那次一次模型都没调。
    kb.embeddings().register_space(EmbeddingSpace { id: "v".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "f32".into() }).unwrap();
    let mut tagged = memory("记住她喜欢苹果", "public");
    tagged.record.tags = vec!["偏好".into()];
    let id = kb.memories().upsert(tagged).unwrap().value.header.id;
    assert!(!kb.embeddings().vector_ready("default", "v", "memory").unwrap(), "还没补过，谈不上就绪");
    let gated = kb.search(&vector_query("v", "记住她喜欢苹果", vec![RecordKind::Memory])).unwrap();
    assert!(gated.hits.is_empty() && gated.diagnostics.degraded.contains(&Degrade::VectorNotReady),
        "记忆档还没补齐时它的向量不参与打分");
    // 把模型接上：注册这个动作自己就触发一次补齐，不需要谁再喊一声。
    let embedder = FakeEmbedder::new(4);
    let calls = embedder.lengths();
    let me = std::thread::current().name().unwrap_or_default().to_string();
    kb.embeddings().register_embedder("v", embedder).unwrap();
    // 模型已就位的情况下再写一条：调用方这条线程上依然一次模型都不调。
    let after_registration = calls_from(&calls, &me);
    kb.memories().upsert(memory("第二条 也喜欢梨", "public")).unwrap();
    assert_eq!(calls_from(&calls, &me), after_registration, "写入不许碰模型");
    assert!(wait_until_ready(&kb, "default", "v", "memory", 15_000), "补完记忆档才放行向量路");
    let hits = kb.search(&vector_query("v", "记住她喜欢苹果", vec![RecordKind::Memory])).unwrap().hits;
    assert_eq!(hits[0].key.id, id);
    // tags 进该记录的向量文本：换一个只出现在 tags 里的词也能召回。
    let by_tag = kb.search(&SearchRequest { text: false, embed_space: Some("v".into()),
        kinds: vec![RecordKind::Memory], ..SearchRequest { query: "偏好".into(), ..Default::default() } }).unwrap().hits;
    assert!(by_tag.iter().any(|hit| hit.key.id == id), "只出现在标签里的词也能召回");
    // 笔记与切片默认不进向量：向量路看不到，全文路仍能看到。
    let note_path = dir.path().join("a.md");
    std::fs::write(&note_path, "笔记正文里的独有措辞").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&note_path)).unwrap().value;
    kb.update_index().unwrap();
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
fn vectorization_targets_are_independent_per_namespace() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
    // 内置默认：记忆与图谱开、笔记关。
    assert!(kb.embeddings().vectorization("default", "memory").unwrap());
    assert!(kb.embeddings().vectorization("default", "graph").unwrap());
    assert!(!kb.embeddings().vectorization("default", "notes").unwrap());
    // 档位名只认三档。
    assert!(matches!(kb.embeddings().set_vectorization("default", "knowledge", true), Err(Error::Validation(_))));
    assert!(matches!(kb.embeddings().vectorization("default", "文本"), Err(Error::Validation(_))));
    // 关掉记忆档，图谱档与笔记档的取值不动。
    kb.embeddings().set_vectorization("default", "memory", false).unwrap();
    assert!(!kb.embeddings().vectorization("default", "memory").unwrap());
    assert!(kb.embeddings().vectorization("default", "graph").unwrap());
    assert!(!kb.embeddings().vectorization("default", "notes").unwrap());

    let muted = kb.memories().upsert(memory("记忆档关闭时的措辞", "public")).unwrap().value.header.id;
    let entity_id = kb.graph().apply_batch(&GraphBatch { entities: vec![entity("记忆档关闭时写入的实体")], ..Default::default() })
        .unwrap().value.entities[0].header.id;
    fill(&kb, "v");
    // 记忆不生成向量；同一时刻写的实体照旧生成，两档互不牵连。
    assert!(kb.search(&vector_query("v", "记忆档关闭时的措辞", vec![RecordKind::Memory])).unwrap().hits.is_empty());
    assert_eq!(kb.search(&vector_query("v", "记忆档关闭时写入的实体", vec![RecordKind::Entity])).unwrap().hits[0].key.id, entity_id);
    assert_eq!(kb.embeddings().sync("v", 32).unwrap().value.written, 0, "关闭的档位不该被 sync 补出向量");
    // 开关只管「是否生成」：重新打开后这一条才补上向量。
    kb.embeddings().set_vectorization("default", "memory", true).unwrap();
    assert!(wait_until_ready(&kb, "default", "v", "memory", 15_000), "重新打开后补齐缺的那一条");
    assert_eq!(kb.search(&vector_query("v", "记忆档关闭时的措辞", vec![RecordKind::Memory])).unwrap().hits[0].key.id, muted);
}

#[test]
fn notes_switch_gates_chunk_vectors_and_keeps_existing_ones() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
    let path = dir.path().join("n.md");
    std::fs::write(&path, "切片正文里的独有措辞").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    kb.update_index().unwrap();
    let chunk = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap()[0].header.id;
    // 笔记档默认关：切片不进向量路，全文路照常命中在切片上。
    assert!(kb.search(&vector_query("v", "切片正文里的独有措辞", vec![RecordKind::Chunk])).unwrap().hits.is_empty());
    assert_eq!(kb.search(&SearchRequest { query: "独有措辞".into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits[0].key.id, chunk);
    // 打开笔记档：补齐切片向量，向量路命中它，且向量取自切片正文本身。
    kb.embeddings().set_vectorization("default", "notes", true).unwrap();
    assert!(wait_until_ready(&kb, "default", "v", "notes", 15_000));
    let hits = kb.search(&vector_query("v", "切片正文里的独有措辞", vec![RecordKind::Chunk])).unwrap().hits;
    assert_eq!(hits[0].key.id, chunk);
    assert!((hits[0].vector_scores["v"] - 1.0).abs() < 1e-6, "查询词与切片正文逐字相同，余弦应为 1");
    // 关掉笔记档：向量留在库里（sync 没有需要补的），只是不再进向量路。
    kb.embeddings().set_vectorization("default", "notes", false).unwrap();
    assert!(kb.search(&vector_query("v", "切片正文里的独有措辞", vec![RecordKind::Chunk])).unwrap().hits.is_empty());
    assert_eq!(kb.embeddings().sync("v", 32).unwrap().value.written, 0, "已有向量仍在，无需重算");
    kb.embeddings().set_vectorization("default", "notes", true).unwrap();
    fill(&kb, "v");
    assert_eq!(kb.search(&vector_query("v", "切片正文里的独有措辞", vec![RecordKind::Chunk])).unwrap().hits[0].key.id, chunk);
}

#[test]
fn namespace_master_switch_overrides_every_target() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
    kb.embeddings().set_namespace_vectorization("other", false).unwrap();
    // 域级键不渗进档位读取：三档仍按各自的内置默认。
    assert!(kb.embeddings().vectorization("other", "memory").unwrap());
    assert!(!kb.embeddings().vectorization("other", "notes").unwrap());
    // 三档全开也压不过总闸。
    for target in ["memory", "graph", "notes"] { kb.embeddings().set_vectorization("other", target, true).unwrap(); }
    let mut muted = memory("总闸关闭时的措辞", "public"); muted.record.namespace = "other".into();
    kb.memories().upsert(muted).unwrap();
    assert_eq!(kb.embeddings().sync("v", 32).unwrap().value.written, 0);
    let request = SearchRequest { query: "总闸关闭".into(), filter: ReadFilter { namespace: "other".into(), ..Default::default() },
        embed_space: Some("v".into()), ..Default::default() };
    let result = kb.search(&request).unwrap();
    assert!(result.diagnostics.degraded.contains(&Degrade::NamespaceDisabled));
    assert!(!result.diagnostics.vector_used, "总闸关闭时不走向量路");
    assert!(!result.hits.is_empty(), "总闸只关向量路，全文路照常给结果");
}

#[test]
fn vectorization_switches_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
        kb.embeddings().set_vectorization("default", "memory", false).unwrap();
        kb.embeddings().set_vectorization("default", "notes", true).unwrap();
        kb.close().unwrap();
    }
    let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
    assert!(!kb.embeddings().vectorization("default", "memory").unwrap());
    assert!(kb.embeddings().vectorization("default", "notes").unwrap());
    assert!(kb.embeddings().vectorization("default", "graph").unwrap(), "没设过的档位仍取内置默认");
    kb.memories().upsert(memory("重开之后写入的记忆", "public")).unwrap();
    std::fs::write(dir.path().join("n.md"), "重开之后写入的切片").unwrap();
    kb.notes().upsert_file(NoteFileInput::new(dir.path().join("n.md"))).unwrap();
    kb.update_index().unwrap();
    assert!(wait_until_ready(&kb, "default", "v", "notes", 15_000), "只有笔记档那一条会被补上");
    assert!(kb.search(&vector_query("v", "重开之后写入的记忆", vec![RecordKind::Memory])).unwrap().hits.is_empty());
    assert!(!kb.search(&vector_query("v", "重开之后写入的切片", vec![RecordKind::Chunk])).unwrap().hits.is_empty());
}

#[test]
fn vector_cleanup_follows_record_lifecycle_not_the_switch() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
    let id = kb.memories().upsert(memory("待删除的记忆措辞", "public")).unwrap().value.header.id;
    fill(&kb, "v");
    assert_eq!(kb.search(&vector_query("v", "待删除的记忆措辞", vec![RecordKind::Memory])).unwrap().hits[0].key.id, id);
    // 关闭状态下删除：向量随记录一起消失。
    kb.embeddings().set_vectorization("default", "memory", false).unwrap();
    kb.memories().delete(id, &ReadFilter::default()).unwrap();
    assert!(kb.search(&vector_query("v", "待删除的记忆措辞", vec![RecordKind::Memory])).unwrap().hits.is_empty());
    // 关闭状态下改笔记正文：旧切片的向量照旧随记录作废，也不生成新的。
    kb.embeddings().set_vectorization("default", "notes", true).unwrap();
    let path = dir.path().join("n.md");
    std::fs::write(&path, "第一版切片措辞").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    kb.update_index().unwrap();
    kb.embeddings().sync("v", 32).unwrap();
    let first = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap()[0].header.id;
    assert_eq!(kb.search(&vector_query("v", "第一版切片措辞", vec![RecordKind::Chunk])).unwrap().hits[0].key.id, first);
    // 关闭状态下改正文：旧切片的向量照旧随记录作废，也不生成新的。
    kb.embeddings().set_vectorization("default", "notes", false).unwrap();
    std::fs::write(&path, "第二版切片措辞").unwrap();
    kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap();
    kb.update_index().unwrap();
    let second = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap()[0].header.id;
    assert_ne!(second, first);
    assert_eq!(kb.embeddings().sync("v", 32).unwrap().value.written, 0, "关闭的笔记档不补新向量");
    kb.embeddings().set_vectorization("default", "notes", true).unwrap();
    assert!(wait_until_ready(&kb, "default", "v", "notes", 15_000), "只有新切片需要补向量");
    let mut wide = vector_query("v", "第二版切片措辞", vec![RecordKind::Chunk]); wide.limit = 10;
    let hits = kb.search(&wide).unwrap().hits;
    assert_eq!(hits[0].key.id, second);
    assert!(hits.iter().all(|hit| hit.key.id != first), "旧切片的向量没有留下");
}

#[test]
fn background_thread_backfills_after_the_batch_ends() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    // 写入的时候还没有模型可用：缺口留着，谁都不许拿半个领域去走向量路。
    kb.memories().upsert(memory("等模型上线的记忆", "public")).unwrap();
    assert!(!kb.embeddings().vector_ready("default", "v", "memory").unwrap());
    // 注册模型这个动作自己就会触发补齐，不需要谁再喊一次。
    space(&kb, "v", 4);
    assert!(wait_until_ready(&kb, "default", "v", "memory", 15_000), "库内线程应当把缺口补上");
    let hits = kb.search(&vector_query("v", "等模型上线的记忆", vec![RecordKind::Memory])).unwrap().hits;
    assert_eq!(hits.len(), 1);
    // 关闭库时线程收尾：关掉之后不再有任何后台写，库也拒绝再被使用。
    kb.close().unwrap();
    assert!(matches!(kb.embeddings().sync("v", 32), Err(Error::Closed)));
    // 重开同一目录：上次补好的向量还在，只补这之后新出现的缺口。
    let reopened = KnowledgeBase::open(dir.path()).unwrap();
    space(&reopened, "v", 4);
    let fresh = reopened.memories().upsert(memory("重开之后写入的记忆", "public")).unwrap().value.header.id;
    // 写入本身就会叫线程：不用谁再调 sync。
    assert!(wait_until_ready(&reopened, "default", "v", "memory", 15_000), "写入之后线程自己会补");
    let hits = reopened.search(&vector_query("v", "重开之后写入的记忆", vec![RecordKind::Memory])).unwrap().hits;
    assert!(hits.iter().any(|hit| hit.key.id == fresh), "写入那条进了向量路");
    reopened.close().unwrap();
}

#[test]
fn chunk_vectors_come_from_the_real_body_not_an_empty_one() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
    kb.embeddings().set_vectorization("default", "notes", true).unwrap();
    let path = dir.path().join("n.md");
    std::fs::write(&path, "切片正文独有措辞").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    let chunk = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap()[0].header.id;
    // 故意不调 update_index：这一刻切片正文还压在索引 writer 里，没提交。
    // 补齐自己会先把索引追平再取正文，绝不用空文本凑一个向量。
    assert!(wait_until_ready(&kb, "default", "v", "notes", 15_000));
    let hits = kb.search(&vector_query("v", "切片正文独有措辞", vec![RecordKind::Chunk])).unwrap().hits;
    assert_eq!(hits[0].key.id, chunk);
    assert!((hits[0].vector_scores["v"] - 1.0).abs() < 1e-6, "向量取自切片正文本身，不是空文本");
}

#[test]
fn the_gap_counts_only_enabled_targets() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
    // 笔记档默认关：写入的笔记不生成向量，笔记档也就谈不上就绪。
    let path = dir.path().join("n.md");
    std::fs::write(&path, "笔记正文里的措辞").unwrap();
    kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap();
    kb.update_index().unwrap();
    assert!(!kb.embeddings().vector_ready("default", "v", "notes").unwrap(), "关掉的档位谈不上就绪");
    let vector_hits = kb.search(&vector_query("v", "笔记正文里的措辞", vec![RecordKind::Chunk])).unwrap().hits;
    assert!(vector_hits.is_empty(), "关掉的档位一条向量都不生成");
    // 打开笔记档：切片这才需要补，补齐之后该档才放行。
    kb.embeddings().set_vectorization("default", "notes", true).unwrap();
    assert!(wait_until_ready(&kb, "default", "v", "notes", 15_000));
    let hits = kb.search(&vector_query("v", "笔记正文里的措辞", vec![RecordKind::Chunk])).unwrap().hits;
    assert_eq!(hits.len(), 1, "开档后才补出切片向量");
}

#[test]
fn callback_failures_still_commit_and_degrade_to_text() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.embeddings().register_space(EmbeddingSpace { id: "v".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "f32".into() }).unwrap();
    kb.embeddings().register_embedder("v", FakeEmbedder::new(4)).unwrap();
    // 记录本身能嵌入：这一批补完，记忆档就绪，向量路放行——下面才测得到「查询词嵌入失败」这一档。
    let id = kb.memories().upsert(memory("平铺直叙的记忆", "public")).unwrap().value.header.id;
    assert!(wait_until_ready(&kb, "default", "v", "memory", 15_000));
    assert!(kb.embeddings().vector_ready("default", "v", "memory").unwrap());
    // 检索侧：查询词嵌入失败就退纯全文，不报错、不返回空。
    let asking = SearchRequest { query: "触发降级 平铺直叙".into(), embed_space: Some("v".into()), ..Default::default() };
    let result = kb.search(&asking).unwrap();
    assert!(result.diagnostics.degraded.contains(&Degrade::EmbedFailed));
    assert!(!result.diagnostics.vector_used && result.diagnostics.text_used);
    assert_eq!(result.hits.len(), 1);
    // 目标空间没有登记过是配置错误；登记过但没绑回调才降级。
    assert!(matches!(kb.search(&SearchRequest { query: "任意".into(), embed_space: Some("ghost".into()), ..Default::default() }), Err(Error::NotFound(_))));
    kb.embeddings().unregister_embedder("v").unwrap();
    let degraded = kb.search(&SearchRequest { query: "平铺直叙".into(), embed_space: Some("v".into()), ..Default::default() }).unwrap();
    assert!(degraded.diagnostics.degraded.contains(&Degrade::NoEmbedder));
    assert_eq!(degraded.hits[0].key.id, id);
}

#[test]
fn an_interrupted_fill_leaves_the_domain_unready() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb, "v", 4);
    // 有一条记录的回调会挂：补齐反复中断，记录照常在库里，但记忆档不许标成就绪。
    let id = kb.memories().upsert(memory("触发降级的记忆", "public")).unwrap().value.header.id;
    assert!(!wait_until_ready(&kb, "default", "v", "memory", 1_000), "补不上的记录让记忆档停在未就绪");
    assert!(kb.health().unwrap().last_degraded.contains(&Degrade::EmbedFailed), "补齐失败要记一档降级");
    assert!(!kb.embeddings().vector_ready("default", "v", "memory").unwrap());
    let gated = kb.search(&SearchRequest { query: "触发降级".into(), kinds: vec![RecordKind::Memory], embed_space: Some("v".into()), ..Default::default() }).unwrap();
    assert!(gated.diagnostics.degraded.contains(&Degrade::VectorNotReady) && !gated.diagnostics.vector_used);
    assert_eq!(gated.hits[0].key.id, id, "向量路关了，全文路照常给结果");
}

#[test]
fn search_parameters_control_paths_and_totals() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb, "v", 4);
    for i in 0..3 { kb.memories().upsert(memory(&format!("参数 目标 {i}"), "public")).unwrap(); }
    kb.memories().upsert(memory("无关内容", "public")).unwrap();
    let kinds = vec![RecordKind::Memory];
    fill(&kb, "v");

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
fn rerank_consumes_the_merged_paths_before_fusion() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb, "v", 8);
    let texts = ["重排合并 甲", "重排合并 乙", "重排合并 丙", "重排合并 丁", "重排合并 戊", "重排合并 己"];
    let mut id_to_text: BTreeMap<i64, String> = BTreeMap::new();
    for text in texts {
        let id = kb.memories().upsert(memory(text, "public")).unwrap().value.header.id;
        id_to_text.insert(id, text.to_string());
    }
    fill(&kb, "v");
    let kinds = vec![RecordKind::Memory];

    // 两路各自的名次：只走一路、不重排。
    let text_only = SearchRequest { query: "重排合并".into(), kinds: kinds.clone(), vector: false, rerank: false, limit: 6, ..Default::default() };
    let text_order: Vec<i64> = kb.search(&text_only).unwrap().hits.iter().map(|hit| hit.key.id).collect();
    let vector_only = SearchRequest { query: "重排合并".into(), kinds: kinds.clone(), text: false, rerank: false, limit: 6, embed_space: Some("v".into()), ..Default::default() };
    let vector_order: Vec<i64> = kb.search(&vector_only).unwrap().hits.iter().map(|hit| hit.key.id).collect();

    // 期望：两路名次交替、去重。
    let mut seen = std::collections::HashSet::new();
    let mut expected: Vec<String> = Vec::new();
    let mut index = 0;
    while index < text_order.len() || index < vector_order.len() {
        if let Some(key) = text_order.get(index) { if seen.insert(*key) { expected.push(id_to_text[key].clone()); } }
        if let Some(key) = vector_order.get(index) { if seen.insert(*key) { expected.push(id_to_text[key].clone()); } }
        index += 1;
    }

    let seen_docs: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen_docs.clone();
    kb.register_reranker_with(move |_: &str, documents: &[String]| {
        recorder.lock().unwrap().push(documents.to_vec());
        Ok(vec![0.0f32; documents.len()])
    }, RerankerOptions { max_docs: 3, max_tokens_per_doc: 1024, max_tokens_query: None }).unwrap();
    seen_docs.lock().unwrap().clear(); // 注册校验会先调一次，清掉只留检索那次。

    let result = kb.search(&SearchRequest { query: "重排合并".into(), kinds, limit: 6, embed_space: Some("v".into()), ..Default::default() }).unwrap();
    assert!(result.diagnostics.reranked);
    assert_eq!(result.diagnostics.rerank_candidates, 3);

    let seen = seen_docs.lock().unwrap();
    assert_eq!(seen.len(), 1);
    let expected_docs: Vec<String> = expected.into_iter().take(3).collect();
    assert_eq!(seen[0], expected_docs, "送重排的候选是两路名次交替合并，不是 RRF 截断出来的");
}

#[test]
fn rerank_failure_falls_back_to_the_full_candidate_set() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for tag in ["甲", "乙", "丙", "丁", "戊"] { kb.memories().upsert(memory(&format!("兜底目标 {tag}"), "public")).unwrap(); }
    let request = SearchRequest { query: "兜底目标".into(), kinds: vec![RecordKind::Memory], vector: false, limit: 5, ..Default::default() };
    let baseline: Vec<i64> = kb.search(&SearchRequest { rerank: false, ..request.clone() }).unwrap().hits.iter().map(|hit| hit.key.id).collect();
    assert_eq!(baseline.len(), 5);

    // 重排回调报错、且 max_docs 小于候选数：兜底必须对完整候选集按融合分排序，
    // 而不是只返回被 max_docs 截断过的那几条。
    kb.register_reranker_with(|_: &str, _: &[String]| Err("重排服务不可用".to_string()),
        RerankerOptions { max_docs: 2, max_tokens_per_doc: 1024, max_tokens_query: None }).unwrap();
    let result = kb.search(&request).unwrap();
    assert!(!result.diagnostics.reranked);
    assert!(result.diagnostics.degraded.contains(&Degrade::RerankFailed));
    assert_eq!(result.diagnostics.rerank_truncated, 3);
    assert_eq!(result.hits.iter().map(|hit| hit.key.id).collect::<Vec<_>>(), baseline, "重排挂了也要给出完整的融合排序结果");
}

#[test]
fn rerank_is_not_called_when_there_are_no_candidates() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.memories().upsert(memory("无关内容", "public")).unwrap();
    // 候选为空时不该调模型：空文档列表对多数重排服务是无效请求，会误标降级。
    let calls = Arc::new(Mutex::new(0usize));
    let recorder = calls.clone();
    kb.register_reranker(move |_: &str, documents: &[String]| {
        *recorder.lock().unwrap() += 1;
        Ok(vec![0.0f32; documents.len()])
    }).unwrap();
    *calls.lock().unwrap() = 0; // 注册校验会先调一次，清掉。
    let result = kb.search(&SearchRequest { query: "查不到的词".into(), kinds: vec![RecordKind::Memory], vector: false, ..Default::default() }).unwrap();
    assert!(result.hits.is_empty());
    assert_eq!(*calls.lock().unwrap(), 0, "没有候选就不该调重排回调");
    assert!(!result.diagnostics.degraded.contains(&Degrade::RerankFailed));
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
    kb.embeddings().sync("v", 32).unwrap();
    assert!(wait_until_ready(&kb, "default", "v", "memory", 15_000), "八条都补上了");
    let calls = lengths.lock().unwrap().clone();
    let shrink_at = calls.iter().position(|len| *len == 8).expect("先按声明的 8 条试一次");
    assert_eq!(calls.iter().filter(|len| **len == 8).count(), 1, "减半后不再从 8 重试");
    assert!(calls[shrink_at + 1..].iter().all(|len| *len <= 4), "此后一律沿用减半后的 4");
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
    kb.embeddings().sync("new", 32).unwrap();
    // 后台线程可能在同一时刻已经把这三条补掉，所以只断言终态，不数这一次写了多少。
    assert!(wait_until_ready(&kb, "default", "new", "memory", 15_000), "换上新模型后补齐并标记就绪");
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
    fill(&kb, "v");
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
        // 注册回调自己就触发一次补齐，谁补的不重要，只等终态。
        kb.embeddings().sync(id, 50).unwrap();
        assert!(wait_until_ready(&kb, "default", id, "memory", 15_000), "空间 {id} 补齐并标记就绪");
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
            // 写入只登记待办；使用方按自己的节奏追平索引，这也是读者不必替写者收尾的常态。
            if n % 16 == 15 { writer_kb.update_index().unwrap(); }
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
    fill(&kb, "v");
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
    fill(&kb,"v");
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

/// 标签集拼在承载它的那一条正文前面：记忆是它自己，笔记是它的第一片。
/// 正文里根本没有的词，靠这份标签也能搜到。
#[test]
fn the_tag_set_rides_on_the_owning_record_and_carries_queries_the_body_cannot() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    // 正文不含"星见雅"，只有拼在它前面的标签带它。
    let mut m = memory("她喜欢苹果", "public");
    m.record.tags = vec!["星见雅".into()];
    let id = kb.memories().upsert(m).unwrap().value.header.id;
    let hits = kb.search(&SearchRequest { query: "星见雅".into(), ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key.id, id);
    // 笔记：登记根目录之后，相对路径的目录段拆成标签，拼在这一篇第一片的正文前面。
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
    let note_dir = dir.path().join("绝区零").join("角色");
    std::fs::create_dir_all(&note_dir).unwrap();
    let note_path = note_dir.join("雅.md");
    std::fs::write(&note_path, "这是一段无关内容").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&note_path)).unwrap().value;
    let chunks = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap();
    assert_eq!(chunks.len(), 1);
    for word in ["绝区零", "角色", "雅"] {
        let hits = kb.search(&SearchRequest { query: word.into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
        assert_eq!(hits.len(), 1, "路径段的标签拼在第一片的正文前面：{word}");
        assert_eq!(hits[0].key.id, chunks[0].header.id);
    }
    // 取回的切片正文只有它自己那一段，标签不在里面。
    assert_eq!(chunks[0].content, "这是一段无关内容", "取回的是纯正文");
    // 笔记不占索引文档：要文件列表按库里的标签翻。
    let hits = kb.search(&SearchRequest { query: "雅".into(), kinds: vec![RecordKind::Note], ..Default::default() }).unwrap().hits;
    assert!(hits.is_empty(), "笔记不进索引");
    let page = kb.notes().list(&PageRequest { filter: ReadFilter { tags: vec!["雅".into()], ..Default::default() }, ..Default::default() }).unwrap();
    assert_eq!(page.items.len(), 1, "按标签翻笔记仍能筛出这一篇");
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
    fill(&kb, "q");
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
    fill(&kb, "v");
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

/// 正文唯一副本在索引里：库内任何文本列都不留，切片正文按 ID 从索引取回（必要时先提交一次）。
#[test]
fn note_body_lives_only_in_the_index_not_in_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let path = dir.path().join("note.md");
    std::fs::write(&path, "ZZBODYMARK 独有正文标记").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;

    // 显式读切片正文：写入时切好、随文档进了索引，这里按 ID 取回。
    let chunks = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap();
    assert!(chunks.iter().any(|chunk| chunk.content.contains("ZZBODYMARK")));
    drop(kb);

    // 库内所有文本列都不含正文标记。
    // 库里任何字节都不含正文标记：整个库文件连同 WAL 一起扫。
    for name in ["store.sqlite3", "store.sqlite3-wal"] {
        if let Ok(bytes) = std::fs::read(dir.path().join(name)) {
            assert!(!String::from_utf8_lossy(&bytes).contains("ZZBODYMARK"), "{name} 里出现正文标记");
        }
    }
    let conn = rusqlite::Connection::open(dir.path().join("store.sqlite3")).unwrap();
    let mut stmt = conn.prepare("SELECT payload_json||metadata_json||evidence_json FROM records").unwrap();
    let hits = stmt.query_map([], |row| row.get::<_, String>(0)).unwrap()
        .filter(|row| row.as_ref().unwrap().contains("ZZBODYMARK")).count();
    assert_eq!(hits, 0, "笔记正文不落 SQLite");
}

/// 切片只装它自己那一段正文；标签集只拼在这一篇的第一片前面，其余片一份都不带。
/// 没登记领域根目录就不拆路径标签，路径词谁都搜不到。
#[test]
fn chunks_carry_their_own_text_and_only_the_first_one_carries_the_path_tags() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let root = dir.path().to_string_lossy().into_owned();
    let note_dir = dir.path().join("绝区零").join("角色");
    std::fs::create_dir_all(&note_dir).unwrap();
    let short_path = note_dir.join("雅.md");
    std::fs::write(&short_path, "苹果 香蕉 橘子").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&short_path)).unwrap().value;
    let chunks = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "苹果 香蕉 橘子", "切片正文逐字符等于切片原文");
    for word in ["雅", "角色", "绝区零"] {
        assert!(kb.search(&SearchRequest { query: word.into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits.is_empty(),
            "没登记根目录就不拆路径标签：{word}");
    }
    // 登记根目录后重新写入：相对路径按段拆成标签，拼到这一篇第一片的正文前面。
    kb.notes().set_root("default", &root).unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&short_path)).unwrap().value;
    let conn = rusqlite::Connection::open(dir.path().join("store.sqlite3")).unwrap();
    let stored: String = conn.query_row("SELECT path FROM notes WHERE record_id=?1", [note.header.id], |r| r.get(0)).unwrap();
    assert_eq!(stored, "绝区零/角色/雅.md", "库里存的是减掉根目录的相对路径");
    drop(conn);
    let chunks = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap();
    for word in ["绝区零", "角色", "雅"] {
        let hits = kb.search(&SearchRequest { query: word.into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
        assert_eq!(hits.len(), 1, "路径段的标签拼在第一片上：{word}");
        assert_eq!(hits[0].key.id, chunks[0].header.id);
    }
    // 长正文切成多片：标签只在第一片上，其余片连文件名都搜不到。
    let long_path = note_dir.join("长文.md");
    std::fs::write(&long_path, "青提".repeat(400)).unwrap();
    let long_note = kb.notes().upsert_file(NoteFileInput::new(&long_path)).unwrap().value;
    let long_chunks = kb.notes().chunks(long_note.header.id, &ReadFilter::default()).unwrap();
    assert!(long_chunks.len() > 1, "长正文应当切成多片");
    for chunk in &long_chunks { assert!(!chunk.content.contains("长文"), "切片正文不携带文件名"); }
    // 文件名只在它自己那一篇的第一片上：搜「长文」只有一条。
    let hits = kb.search(&SearchRequest { query: "长文".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 1, "文件名标签只拼在第一片上");
    assert_eq!(hits[0].key.id, long_chunks[0].header.id);
    // 目录段两篇都带：每篇各出一条，多片的那篇不会一片一条。
    let hits = kb.search(&SearchRequest { query: "角色".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    let mut hit_ids: Vec<i64> = hits.iter().map(|hit| hit.key.id).collect();
    let mut expected = vec![chunks[0].header.id, long_chunks[0].header.id];
    hit_ids.sort(); expected.sort();
    assert_eq!(hit_ids, expected, "目录段两篇各出一条，都落在各自的第一片");
}

/// 正文只在索引里存一份，源文件在写入之后就可以消失：读切片正文不再回文件。
/// 代价是索引全量重建必须回源——源不在时那批切片正文只能退化为空。
#[test]
fn indexed_text_survives_the_source_file_but_a_rebuild_needs_it() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let path = dir.path().join("gone.md");
    std::fs::write(&path, "会被删掉的正文").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    kb.update_index().unwrap();
    let chunk_id = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap()[0].header.id;
    std::fs::remove_file(&path).unwrap();

    assert_eq!(kb.notes().get_chunk(chunk_id, &ReadFilter::default()).unwrap().content, "会被删掉的正文",
        "正文随写入一起进了索引，源文件没了也读得到");
    assert!(kb.rebuild_indexes().is_ok(), "重建遇到缺失文件不应整体失败");
    assert_eq!(kb.notes().get_chunk(chunk_id, &ReadFilter::default()).unwrap().content, "",
        "重建要回源，源不在的那批切片正文只能为空");
}

/// 重排取候选正文也走索引：源文件删掉之后，回调拿到的仍是切片原文。
#[test]
fn rerank_reads_chunk_bodies_from_the_index_not_from_the_source_file() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let path = dir.path().join("唯一笔记.md");
    std::fs::write(&path, "青提苹果 ZZTEXTMARK").unwrap();
    kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap();
    kb.update_index().unwrap();
    std::fs::remove_file(&path).unwrap();

    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    kb.register_reranker_with(move |_: &str, documents: &[String]| {
        recorder.lock().unwrap().extend(documents.iter().cloned());
        Ok(vec![0.5; documents.len()])
    }, RerankerOptions::default()).unwrap();
    // 注册本身会校验性地调一次回调，清掉只留检索那次。
    seen.lock().unwrap().clear();
    let result = kb.search(&SearchRequest { query: "青提苹果".into(), kinds: vec![RecordKind::Chunk],
        vector: false, ..Default::default() }).unwrap();
    assert!(result.diagnostics.reranked && !result.hits.is_empty());
    assert!(seen.lock().unwrap().iter().any(|body| body.contains("ZZTEXTMARK")), "重排候选正文来自索引");
}

/// 记忆与切片是同一个结构：同一段文本落在谁的文本列上，得分就该一样。
#[test]
fn memory_and_chunk_score_the_same_text_identically() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let text = "青提苹果";
    kb.memories().upsert(memory(text, "public")).unwrap();
    let path = dir.path().join("同文本.md");
    std::fs::write(&path, text).unwrap();
    kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap();

    let score = |kinds: Vec<RecordKind>| {
        let request = SearchRequest { query: text.into(), kinds, vector: false, rerank: false, ..Default::default() };
        let result = kb.search(&request).unwrap();
        assert_eq!(result.hits.len(), 1, "该类型下只有一条记录");
        result.hits[0].text_score.unwrap()
    };
    assert_eq!(score(vec![RecordKind::Memory]), score(vec![RecordKind::Chunk]), "同文本的文本列分数一致");
}

/// 领域根目录：写入路径必须在根目录之内，库里存相对路径，标签集拼在这一篇的第一片前面。
#[test]
fn domain_root_relative_paths_and_the_tag_set_ride_on_the_first_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let root = dir.path().join("data").join("domain").join("gi");
    std::fs::create_dir_all(root.join("bwiki").join("沧州")).unwrap();
    let note_path = root.join("bwiki").join("沧州").join("澜川.md");
    std::fs::write(&note_path, "澜川".repeat(300)).unwrap();
    kb.notes().set_root("default", &root.to_string_lossy()).unwrap();
    assert_eq!(kb.notes().root("default").unwrap(), Some(root.to_string_lossy().replace('\\', "/")));
    // 根目录之外的路径报校验错：不落库、不进索引。
    let outside = dir.path().join("outside.md");
    std::fs::write(&outside, "根目录之外的正文").unwrap();
    assert!(matches!(kb.notes().upsert_file(NoteFileInput::new(&outside)), Err(Error::Validation(_))));
    assert_eq!(kb.health().unwrap().record_count, 0, "越界路径一条记录都不许落库");
    // 正文里不出现路径词：它们只以拼在第一片前面的标签形态存在。
    let mut input = NoteFileInput::new(&note_path);
    input.record.tags = vec!["人物".into()];
    let note = kb.notes().upsert_file(input).unwrap().value;
    let chunks = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap();
    assert!(chunks.len() > 1, "长正文应当切成多片");
    for chunk in &chunks {
        assert!(!chunk.content.contains("bwiki") && !chunk.content.contains("沧州"), "切片正文不携带路径词");
        let mut tags = chunk.header.tags.clone();
        tags.sort();
        assert_eq!(tags, vec!["bwiki".to_string(), "人物".to_string(), "沧州".to_string(), "澜川".to_string()],
            "调用方标签与路径段标签合并去重后挂在每一片上");
    }
    // 查「澜川」：正文里全是它，每一片都由自己的正文命中。
    let hits = kb.search(&SearchRequest { query: "澜川".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), chunks.len(), "每片都能被搜到");
    assert!(hits.iter().all(|hit| hit.text_score.is_some_and(|score| score > 0.0)), "每片都拿到分");
    // 查路径段与调用方标签：这份标签只拼在第一片上，所以只有第一片命中。
    for word in ["bwiki", "沧州", "人物"] {
        let hits = kb.search(&SearchRequest { query: word.into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
        assert_eq!(hits.len(), 1, "标签集只拼在第一片上：{word}");
        assert_eq!(hits[0].key.id, chunks[0].header.id, "命中的是第一片：{word}");
    }
    // 按标签筛切片：`沧州` 筛得到，库里没有的标签换不到 id 就是空结果。
    let by_tag = |tag: &str| kb.search(&SearchRequest { query: "澜川".into(),
        filter: ReadFilter { tags: vec![tag.into()], ..Default::default() }, kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() })
        .unwrap().hits.len();
    assert_eq!(by_tag("沧州"), chunks.len());
    assert_eq!(by_tag("库里没有的标签"), 0);
    // 笔记自己不占索引文档：文档数 = 记录数 − 笔记数。
    let before = kb.health().unwrap();
    assert_eq!(before.index_document_count, before.record_count - 1);
    // 整库重建之后同一批查询结果一致。
    kb.rebuild_indexes().unwrap();
    let again = kb.search(&SearchRequest { query: "澜川".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert_eq!(again.len(), hits.len(), "重建后命中条数一致");
    assert_eq!(again[0].key.id, hits[0].key.id, "重建后名次一致");
    assert_eq!(kb.health().unwrap().index_document_count, before.record_count - 1);
}

// ── 预设检索 ──────────────────────────────────────────────────────────

fn relation_of(subject_id: i64, predicate: &str, object_id: i64) -> RelationInput {
    RelationInput { record: RecordInput::default(), subject_id, predicate: predicate.into(), object_id, confidence: 1.0, reason: String::new() }
}

fn event_of(name: &str, participants: Vec<i64>) -> EventInput {
    EventInput { record: RecordInput::default(), name: name.into(), summary: String::new(), participants, confidence: 1.0, reason: String::new() }
}

/// 「朱樱和白露的同学是谁」：搜实体得种子 → 种子各自敲自己的关系 → 两两之间铺开关系与事件。
#[test]
fn preset_graph_walks_entities_then_relations_then_events() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let mut hua = entity("朱樱");
    hua.aliases = vec!["堂主".into()];
    let created = kb.graph().apply_batch(&GraphBatch {
        entities: vec![hua, entity("白露"), entity("青萍"), entity("玄霜"), entity("长夜堂")], ..Default::default()
    }).unwrap().value;
    let id = |name: &str| created.entities.iter().find(|entity| entity.name == name).unwrap().header.id;
    kb.graph().apply_batch(&GraphBatch {
        relations: vec![
            relation_of(id("朱樱"), "同学", id("青萍")),
            relation_of(id("白露"), "同学", id("玄霜")),
            relation_of(id("青萍"), "同门", id("玄霜")),
            relation_of(id("朱樱"), "客卿于", id("长夜堂")),
        ],
        events: vec![
            event_of("别鹤典仪", vec![id("朱樱"), id("青萍")]),
            event_of("堂中自语", vec![id("青萍")]),
            event_of("开张", vec![id("长夜堂")]),
        ],
        ..Default::default()
    }).unwrap();
    kb.memories().upsert(memory("朱樱的同学是青萍", "public")).unwrap();

    let result = kb.search_preset(&PresetRequest {
        preset: SearchPreset::Rag, query: "朱樱和白露的同学是谁".into(), ..Default::default()
    }).unwrap();

    // 种子实体：查询词命中的实体，按分排，别名一起给出。
    let mut names: Vec<&str> = result.graph.entities.iter().map(|entity| entity.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["朱樱", "白露"], "查询词命中的实体成为种子");
    let hua = result.graph.entities.iter().find(|entity| entity.name == "朱樱").unwrap();
    assert_eq!(hua.aliases, vec!["堂主".to_string()], "实体连别名一起给出");

    // 第二步：种子实体各自到自己的关系里敲查询词，命中的留下。
    let hits: Vec<(i64, String, i64)> = result.graph.relations.iter()
        .map(|relation| (relation.subject_id, relation.predicate.clone(), relation.object_id)).collect();
    assert!(hits.contains(&(id("朱樱"), "同学".into(), id("青萍"))), "朱樱那边的同学关系是结果");
    assert!(hits.contains(&(id("白露"), "同学".into(), id("玄霜"))), "白露那边的同学关系是结果");
    // 第二步敲的是「去掉实体名后的剩余词」：查询去掉朱樱、白露后只剩「的同学是谁」，
    // 「客卿于」这条与剩余词无关，不该仅因正文里写着「朱樱」就被留下。
    assert!(!hits.iter().any(|(_, predicate, _)| predicate == "客卿于"),
        "与剩余词无关的关系不该进第二步结果");

    // 第三步：实体集合两两之间的关系与事件，不筛。
    let context: Vec<String> = result.graph.context_relations.iter().map(|relation| relation.predicate.clone()).collect();
    assert!(context.contains(&"同门".to_string()), "两端都落在集合内的关系进第三段");
    assert!(!context.contains(&"客卿于".to_string()), "只有一端落在集合内的关系不进第三段");
    let events: Vec<&str> = result.graph.context_events.iter().map(|event| event.name.as_str()).collect();
    assert_eq!(events, vec!["别鹤典仪"], "参与者至少两个落在集合内才算第三段的事件");

    // 记忆那一路独立出结果，字段不混。
    assert!(result.memories.iter().any(|hit| hit.record["judgment"].as_str().unwrap_or_default().contains("朱樱的同学")));
    assert!(result.notes.is_empty(), "RAG 不出笔记那一路");
}

/// 查询词本身就是一个实体名时，「去掉实体名后的剩余词」为空：不筛，保留它的全部端点关系。
#[test]
fn preset_graph_keeps_all_incident_relations_for_a_pure_name_query() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let created = kb.graph().apply_batch(&GraphBatch {
        entities: vec![entity("墨团"), entity("派罗"), entity("月裔")], ..Default::default()
    }).unwrap().value;
    let id = |name: &str| created.entities.iter().find(|entity| entity.name == name).unwrap().header.id;
    kb.graph().apply_batch(&GraphBatch {
        relations: vec![
            relation_of(id("墨团"), "朋友", id("派罗")),
            relation_of(id("墨团"), "隶属于", id("月裔")),
        ], ..Default::default()
    }).unwrap();

    let result = kb.search_preset(&PresetRequest {
        preset: SearchPreset::Graph, query: "墨团".into(), ..Default::default()
    }).unwrap();
    let predicates: Vec<&str> = result.graph.relations.iter().map(|r| r.predicate.as_str()).collect();
    assert!(predicates.contains(&"朋友"), "剩余词为空时墨团的朋友关系不该被筛掉");
    assert!(predicates.contains(&"隶属于"), "剩余词为空时墨团的隶属关系不该被筛掉");
}

/// 第二步的扩散：剩余词里的谓词是等价词时，也能召回关系正文里的另一种写法。
#[test]
fn preset_graph_step2_expands_the_remainder_with_predicate_synonyms() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let created = kb.graph().apply_batch(&GraphBatch {
        entities: vec![entity("艾莉儿"), entity("克劳斯")], ..Default::default()
    }).unwrap().value;
    let id = |name: &str| created.entities.iter().find(|entity| entity.name == name).unwrap().header.id;
    kb.graph().apply_batch(&GraphBatch {
        relations: vec![relation_of(id("克劳斯"), "丈夫", id("艾莉儿"))], ..Default::default()
    }).unwrap();

    // 没登记等价词：剩余词「老公」敲不到正文里的「丈夫」。
    let before = kb.search_preset(&PresetRequest {
        preset: SearchPreset::Graph, query: "艾莉儿的老公".into(), ..Default::default()
    }).unwrap();
    assert!(!before.graph.relations.iter().any(|r| r.predicate == "丈夫"), "没登记等价词时敲不到另一写法");

    // 登记「丈夫=老公」后，剩余词扩散带上「丈夫」，关系被第二步召回。
    kb.graph().set_predicate_equivalents("default", &[vec!["丈夫".into(), "老公".into()]]).unwrap();
    let after = kb.search_preset(&PresetRequest {
        preset: SearchPreset::Graph, query: "艾莉儿的老公".into(), ..Default::default()
    }).unwrap();
    assert!(after.graph.relations.iter().any(|r| r.predicate == "丈夫"),
        "登记等价词后「老公」扩散出「丈夫」，关系应被第二步召回");
}

/// 单路预设只填自己那一个字段；广撒网三个字段都填。
#[test]
fn preset_fields_stay_separate() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.memories().upsert(memory("预设字段分离用的记忆", "public")).unwrap();
    kb.graph().apply_batch(&GraphBatch { entities: vec![entity("预设字段分离用的实体")], ..Default::default() }).unwrap();
    let path = dir.path().join("预设字段分离用的笔记.md");
    std::fs::write(&path, "预设字段分离用的正文".repeat(20)).unwrap();
    kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap();

    let query = "预设字段分离用";
    let of = |preset| kb.search_preset(&PresetRequest { preset, query: query.into(), ..Default::default() }).unwrap();

    let memory_only = of(SearchPreset::Memory);
    assert!(!memory_only.memories.is_empty());
    assert!(memory_only.graph.entities.is_empty() && memory_only.notes.is_empty(), "记忆预设只填记忆那一个字段");

    let graph_only = of(SearchPreset::Graph);
    assert!(graph_only.memories.is_empty() && graph_only.notes.is_empty(), "图谱预设只填图谱那一个字段");
    assert!(graph_only.graph.entities.iter().any(|entity| entity.name == "预设字段分离用的实体"));

    let notes_only = of(SearchPreset::Notes);
    assert!(!notes_only.notes.is_empty());
    assert!(notes_only.memories.is_empty() && notes_only.graph.entities.is_empty(), "笔记预设只填笔记那一个字段");

    let broad = of(SearchPreset::Broad);
    assert!(!broad.memories.is_empty() && !broad.notes.is_empty() && !broad.graph.entities.is_empty(), "广撒网三路都出");
}

/// 阈值按字符数而不是条数：预算装不下下一条就停，第一条无论多长都留下。
#[test]
fn preset_budgets_cap_by_characters_not_counts() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    for index in 0..5 {
        kb.memories().upsert(memory(&format!("字符封顶样例 {index} {}", "长".repeat(300)), "public")).unwrap();
    }
    let request = |chars| PresetRequest { preset: SearchPreset::Memory, query: "字符封顶样例".into(),
        budget: PresetBudget { memory_chars: chars, ..Default::default() }, ..Default::default() };

    assert_eq!(kb.search_preset(&request(10_000)).unwrap().memories.len(), 5, "预算够时全部返回");
    assert_eq!(kb.search_preset(&request(600)).unwrap().memories.len(), 1, "三百多字的正文只装得下一条");
    assert!(kb.search_preset(&request(0)).unwrap().memories.is_empty(), "预算为 0 就是空");
}

/// 没有种子实体时图谱那一路空着，记忆那一路照常出结果。
#[test]
fn preset_without_seed_entities_keeps_other_routes() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.memories().upsert(memory("只写在记忆里的独有措辞", "public")).unwrap();

    let result = kb.search_preset(&PresetRequest {
        preset: SearchPreset::Rag, query: "只写在记忆里的独有措辞".into(), ..Default::default()
    }).unwrap();
    assert!(result.graph.entities.is_empty() && result.graph.relations.is_empty() && result.graph.context_events.is_empty(),
        "没有实体命中，图谱那一路整体空着");
    assert!(!result.memories.is_empty(), "记忆那一路不受影响");
}

/// 实体规范名单独成列并按固定倍数加权：名字命中的实体要压过「正文里堆词频」的干扰实体。
#[test]
fn entity_name_column_outweighs_long_body() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    // 正主：规范名就是查询词，正文只有名字本身。
    let target = kb.graph().apply_batch(&GraphBatch { entities: vec![entity("苹果")], ..Default::default() })
        .unwrap().value.entities[0].header.id;
    // 干扰：规范名与查询无关，但别名与属性里反复出现查询词——纯 BM25 下它靠词频与短正文领先。
    let mut noisy = entity("香蕉");
    noisy.aliases = vec!["苹果".into()];
    for key in ["别称", "俗称", "外号"] {
        noisy.attributes.insert(key.into(), vec!["苹果".into(), "苹果".into(), "苹果".into()]);
    }
    let noise = kb.graph().apply_batch(&GraphBatch { entities: vec![noisy], ..Default::default() })
        .unwrap().value.entities[0].header.id;

    let req = SearchRequest { query: "苹果".into(), kinds: vec![RecordKind::Entity],
        text: true, vector: false, rerank: false, ..Default::default() };
    let hits = kb.search(&req).unwrap().hits;
    assert!(hits.len() >= 2, "两条都该被召回");
    assert_eq!(hits[0].key.id, target, "规范名命中应压过长正文里堆起来的词频");
    assert!(hits.iter().any(|hit| hit.key.id == noise));
}

/// 实体的规范名要进重排文档：正文列不含规范名，纯名实体的正文是空串；
/// 重排若只拿到正文，就等于收到空文档、无从判断。
#[test]
fn entity_name_reaches_the_rerank_document() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.graph().apply_batch(&GraphBatch { entities: vec![entity("孤名实体")], ..Default::default() }).unwrap();

    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    kb.register_reranker_with(move |_: &str, documents: &[String]| {
        recorder.lock().unwrap().extend(documents.iter().cloned());
        Ok(vec![1.0f32; documents.len()])
    }, RerankerOptions::default()).unwrap();
    // 注册校验已调用过回调一次，清掉只留检索那次。
    seen.lock().unwrap().clear();

    let request = SearchRequest { query: "孤名实体".into(), kinds: vec![RecordKind::Entity],
        text: true, vector: false, rerank: true, ..Default::default() };
    let hits = kb.search(&request).unwrap().hits;
    assert!(!hits.is_empty(), "纯名实体靠名字列也该被召回");
    let documents = seen.lock().unwrap().clone();
    assert!(documents.iter().any(|doc| doc.contains("孤名实体")),
        "实体的规范名必须出现在重排文档里，实收 {documents:?}");
}

/// 谓词等价词：登记后可列出、可扩散，按领域隔离；没登记就是空。
#[test]
fn predicate_equivalents_register_list_and_expand_per_domain() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    assert!(kb.graph().predicate_equivalents("gi").unwrap().is_empty(), "没登记就是空");

    kb.graph().set_predicate_equivalents("gi", &[vec!["丈夫".into(), "老公".into(), "夫君".into()]]).unwrap();
    let groups = kb.graph().predicate_equivalents("gi").unwrap();
    assert_eq!(groups, vec![vec!["丈夫".to_string(), "夫君".to_string(), "老公".to_string()]], "组内按文本排序");

    // 扩散：查询里出现登记词（老公），返回整组同义词。
    let expanded = kb.graph().expand_query("gi", "艾莉儿的老公").unwrap();
    assert!(expanded.contains(&"丈夫".to_string()) && expanded.contains(&"夫君".to_string()), "命中「老公」应展开出「丈夫」「夫君」: {expanded:?}");
    assert!(kb.graph().expand_query("gi", "艾莉儿").unwrap().is_empty(), "查询里没有登记词就是空");

    // 领域隔离：hsr 没登记，同一查询扩散不出东西。
    assert!(kb.graph().expand_query("hsr", "艾莉儿的老公").unwrap().is_empty(), "别的领域不共享等价词");
}

/// 谓词等价词接入检索：同义查询在登记后能召回关系，且关系里存着的谓词原文不变。
#[test]
fn predicate_equivalents_expand_search_without_rewriting_storage() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let made = kb.graph().apply_batch(&GraphBatch { entities: vec![entity("克劳斯"), entity("艾莉儿")], ..Default::default() }).unwrap();
    let subject = made.value.entities[0].header.id;
    let object = made.value.entities[1].header.id;
    let relation = kb.graph().apply_batch(&GraphBatch { relations: vec![RelationInput {
        record: RecordInput::default(), subject_id: subject, predicate: "配偶".into(),
        object_id: object, confidence: 0.9, reason: String::new(),
    }], ..Default::default() }).unwrap().value.relations[0].header.id;

    let request = SearchRequest { query: "伴侣".into(), kinds: vec![RecordKind::Relation],
        text: true, vector: false, rerank: false, ..Default::default() };
    // 未登记：查询词与关系正文没有字面交集，召回不到。
    assert!(kb.search(&request).unwrap().hits.is_empty(), "没登记时同义查询召回不到");
    // 登记一组等价词后，同一查询被扩散到「配偶」，命中。
    kb.graph().set_predicate_equivalents("default", &[vec!["配偶".into(), "伴侣".into()]]).unwrap();
    let hits = kb.search(&request).unwrap().hits;
    assert_eq!(hits.len(), 1, "登记后同义查询能召回");
    assert_eq!(hits[0].key.id, relation);

    // 落盘不变：关系里存的谓词仍是登记时写的「配偶」，扩散没有改写记录。
    let stored: serde_json::Value = kb.graph().get(RecordKind::Relation, relation, &ReadFilter::default()).unwrap();
    assert_eq!(stored["predicate"], serde_json::json!("配偶"));
}
