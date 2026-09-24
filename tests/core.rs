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
    kb.memories().upsert(moved).unwrap();
    let private = ReadFilter { namespace: "default".into(), scopes: vec!["private".into()], ..Default::default() };
    assert_eq!(kb.memories().get(a, &private).unwrap().header.scope, "private", "作用域可以改");
    let mut restored = first.clone(); restored.record.id = Some(a);
    kb.memories().upsert(restored).unwrap();
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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
    // 索引提交由使用方择时做：这里显式追平，不依赖读者自愈撞上写锁空窗的时机。
    kb.update_index().unwrap();
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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

/// 删空间要连向量库一起清：定义在主库，向量与就绪标记在外挂库，跨库没有外键级联。
#[test]
fn deleting_a_space_removes_its_vectors_and_readiness() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb, "v", 4);
    kb.memories().upsert(memory("会被忘掉的一条", "public")).unwrap();
    fill(&kb, "v");
    assert!(kb.embeddings().vector_ready("default", "v", "memory").unwrap());
    // 向量真的写进了外挂库。
    let vconn = rusqlite::Connection::open(dir.path().join("vectors.sqlite3")).unwrap();
    let before: i64 = vconn.query_row("SELECT COUNT(*) FROM embeddings WHERE space_id='v'", [], |r| r.get(0)).unwrap();
    assert!(before > 0, "补齐后向量库里该空间该有向量");

    kb.embeddings().delete_space("v").unwrap();

    let after: i64 = vconn.query_row("SELECT COUNT(*) FROM embeddings WHERE space_id='v'", [], |r| r.get(0)).unwrap();
    assert_eq!(after, 0, "删空间后向量库里不该留孤儿向量");
    let marks: i64 = vconn.query_row("SELECT COUNT(*) FROM vector_meta WHERE key GLOB 'vector_ready:*:v:*'", [], |r| r.get(0)).unwrap();
    assert_eq!(marks, 0, "删空间后就绪标记也该清掉");
    // 主库定义也没了。
    assert!(kb.embeddings().spaces().unwrap().iter().all(|s| s.id != "v"), "主库定义已删");
    // 同名空间重新注册后从头就绪。
    space(&kb, "v", 4);
    assert!(!kb.embeddings().vector_ready("default", "v", "memory").unwrap(), "重注册后要重新补齐才算就绪");
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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

    // token 预算：候选从前往后累加到超额为止，文档按单篇预算截断。
    let seen: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    kb.register_reranker_with(move |_: &str, documents: &[String]| {
        recorder.lock().unwrap().push((documents.len(), documents.iter().map(|d| d.chars().count()).max().unwrap_or(0)));
        // 故意倒序给分：最后一个候选拿最高分。
        Ok((0..documents.len()).map(|i| i as f32).collect())
    }, RerankerOptions { max_tokens_total: 10, max_tokens_per_doc: 6, ..Default::default() }).unwrap();
    // 注册校验已经调用过回调一次，清掉只留检索那次。
    seen.lock().unwrap().clear();
    let reranked = kb.search(&SearchRequest { rerank: true, ..request.clone() }).unwrap();
    assert!(reranked.diagnostics.reranked);
    assert_eq!(reranked.diagnostics.rerank_candidates, 2);
    assert_eq!(reranked.diagnostics.rerank_truncated, 3);
    assert_eq!(reranked.hits[0].key.id, baseline[1], "倒序给分后原来的第二名排到最前");
    assert!(reranked.hits.iter().all(|hit| hit.rerank_score.is_some()));
    assert_eq!(*seen.lock().unwrap(), vec![(2, 6)], "回调实收 2 条候选，累加到总预算超额为止");

    // 注销之后 rerank 开关被忽略，结果与不重排一致。
    assert!(kb.unregister_reranker());
    let ignored = kb.search(&SearchRequest { rerank: true, ..request.clone() }).unwrap();
    assert!(!ignored.diagnostics.reranked);
    assert_eq!(ignored.hits.iter().map(|hit| hit.key.id).collect::<Vec<_>>(), baseline);
}

#[test]
fn reranker_stops_at_the_candidate_cap() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for tag in ["甲", "乙", "丙", "丁", "戊"] { kb.memories().upsert(memory(&format!("条数上限 {tag}"), "public")).unwrap(); }
    let request = SearchRequest { query: "条数上限".into(), limit: 5, kinds: vec![RecordKind::Memory], vector: false, ..Default::default() };

    // token 预算装得下全部 5 条短候选，能截到 2 条的只有条数上限。
    let seen: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    kb.register_reranker_with(move |_: &str, documents: &[String]| {
        recorder.lock().unwrap().push(documents.len());
        Ok(vec![0.0f32; documents.len()])
    }, RerankerOptions { max_candidates: 2, ..Default::default() }).unwrap();
    seen.lock().unwrap().clear(); // 注册校验会先调一次，清掉只留检索那次。

    let capped = kb.search(&SearchRequest { rerank: true, ..request.clone() }).unwrap();
    assert!(capped.diagnostics.reranked);
    assert_eq!(capped.diagnostics.rerank_candidates, 2, "条数上限把候选截到 2 条");
    assert_eq!(capped.diagnostics.rerank_truncated, 3);
    assert_eq!(*seen.lock().unwrap(), vec![2], "回调实收 2 条候选");

    // 0 不是「不限制」，而是无效配置：静默把重排关掉比报错更难查。
    assert!(kb.register_reranker_with(|_: &str, documents: &[String]| Ok(vec![0.0f32; documents.len()]),
        RerankerOptions { max_candidates: 0, ..Default::default() }).is_err());
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
    }, RerankerOptions { max_tokens_total: 12, max_tokens_per_doc: 1024, ..Default::default() }).unwrap();
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

    // 重排回调报错、且 token 预算装不下全部候选：兜底必须对完整候选集按融合分排序，
    // 而不是只返回被预算截断过的那几条。
    kb.register_reranker_with(|_: &str, _: &[String]| Err("重排服务不可用".to_string()),
        RerankerOptions { max_tokens_total: 10, max_tokens_per_doc: 1024, ..Default::default() }).unwrap();
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
                filter: ReadFilter { namespace: default_namespace(), scopes: vec![format!("scope{i}")], tags: vec![format!("tag{i}")], note_ids: vec![] },
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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
/// 正文里根本没有的词，靠这份标签也能搜到。目录段已升格成独立列，不再进这份标签。
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
    // 笔记：登记根目录之后，文件名走名字列（可搜），目录段只当标签（可筛不可搜）。
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
    let note_dir = dir.path().join("绝区零").join("角色");
    std::fs::create_dir_all(&note_dir).unwrap();
    let note_path = note_dir.join("雅.md");
    std::fs::write(&note_path, "这是一段无关内容").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&note_path)).unwrap().value;
    let chunks = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap();
    assert_eq!(chunks.len(), 1);
    // 目录段降为纯过滤器：常规检索搜不到它们。
    for word in ["绝区零", "角色"] {
        let hits = kb.search(&SearchRequest { query: word.into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
        assert!(hits.is_empty(), "目录段不参与常规匹配：{word}");
    }
    // 文件名走名字列：常规检索搜得到，并且是这一片。
    let hits = kb.search(&SearchRequest { query: "雅".into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 1, "文件名是名字，能搜到");
    assert_eq!(hits[0].key.id, chunks[0].header.id);
    // 取回的切片正文只有它自己那一段，标签不在里面。
    assert_eq!(chunks[0].content, "这是一段无关内容", "取回的是纯正文");
    // 笔记不占索引文档：要文件列表按库里的标签翻。
    let hits = kb.search(&SearchRequest { query: "雅".into(), kinds: vec![RecordKind::Note], ..Default::default() }).unwrap().hits;
    assert!(hits.is_empty(), "笔记不进索引");
    // 目录段仍留在标签表里：按文件夹筛笔记照样筛得出来（只是不再参与匹配）。
    let page = kb.notes().list(&PageRequest { filter: ReadFilter { tags: vec!["角色".into()], ..Default::default() }, ..Default::default() }).unwrap();
    assert_eq!(page.items.len(), 1, "按目录段筛笔记仍能筛出这一篇");
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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

/// 切片只装它自己那一段正文；文件名与目录只挂在这一篇的第一片上。
/// 根目录是写入前提：没登记就拒绝写入；登记后文件名进名字列、目录段进目录列且不参与常规匹配。
#[test]
fn chunks_carry_their_own_text_and_only_the_first_one_carries_the_path_tags() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let root = dir.path().to_string_lossy().into_owned();
    let note_dir = dir.path().join("绝区零").join("角色");
    std::fs::create_dir_all(&note_dir).unwrap();
    // 没登记根目录就写笔记：直接拒绝——根目录是写入的前提，库从头到尾不接触上游绝对路径。
    let bare_path = note_dir.join("无根.md");
    std::fs::write(&bare_path, "苹果 香蕉 橘子").unwrap();
    assert!(matches!(kb.notes().upsert_file(NoteFileInput::new(&bare_path)), Err(Error::Validation(_))),
        "没登记根目录就拒绝写入");
    assert_eq!(kb.health().unwrap().record_count, 0, "被拒的笔记一条都不落库");
    // 登记根目录后写入这一篇：文件名进名字列、目录段进目录列，都只挂第一片。
    kb.notes().set_root("default", &root).unwrap();
    let short_path = note_dir.join("雅.md");
    std::fs::write(&short_path, "苹果 香蕉 橘子").unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&short_path)).unwrap().value;
    let conn = rusqlite::Connection::open(dir.path().join("store.sqlite3")).unwrap();
    let stored: String = conn.query_row("SELECT path FROM notes WHERE record_id=?1", [note.header.id], |r| r.get(0)).unwrap();
    assert_eq!(stored, "绝区零/角色/雅.md", "库里存的是减掉根目录的相对路径");
    drop(conn);
    let chunks = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap();
    // 目录段不参与常规匹配。
    for word in ["绝区零", "角色"] {
        let hits = kb.search(&SearchRequest { query: word.into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
        assert!(hits.is_empty(), "目录段不参与常规匹配：{word}");
    }
    // 文件名进名字列，能搜到，命中的是第一片。
    let hits = kb.search(&SearchRequest { query: "雅".into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 1, "文件名是名字，能搜到");
    assert_eq!(hits[0].key.id, chunks[0].header.id);
    // 长正文切成多片：名字与目录只挂第一片，其余片连文件名都搜不到。
    let long_path = note_dir.join("长文.md");
    std::fs::write(&long_path, "青提".repeat(400)).unwrap();
    let long_note = kb.notes().upsert_file(NoteFileInput::new(&long_path)).unwrap().value;
    let long_chunks = kb.notes().chunks(long_note.header.id, &ReadFilter::default()).unwrap();
    assert!(long_chunks.len() > 1, "长正文应当切成多片");
    for chunk in &long_chunks { assert!(!chunk.content.contains("长文"), "切片正文不携带文件名"); }
    // 文件名只在它自己那一篇的第一片上：搜「长文」只有一条。
    let hits = kb.search(&SearchRequest { query: "长文".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 1, "文件名只挂第一片");
    assert_eq!(hits[0].key.id, long_chunks[0].header.id);
    // 目录段整篇都不参与匹配：两篇都搜不到。
    let hits = kb.search(&SearchRequest { query: "角色".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert!(hits.is_empty(), "目录段不参与常规匹配，两篇都搜不到");
}

/// 正文只在索引里存一份，源文件在写入之后就可以消失：读切片正文不再回文件。
/// 代价是索引全量重建必须回源——源不在时那批切片正文只能退化为空。
#[test]
fn indexed_text_survives_the_source_file_but_a_rebuild_needs_it() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
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

/// 领域根目录：写入路径必须在根目录之内，库里存相对路径，文件名与目录段各成独立列、只挂第一片。
#[test]
fn domain_root_relative_paths_and_the_tag_set_ride_on_the_first_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let root = dir.path().join("data").join("domain").join("demo");
    std::fs::create_dir_all(root.join("notes").join("characters")).unwrap();
    let note_path = root.join("notes").join("characters").join("overview.md");
    std::fs::write(&note_path, "overview ".repeat(300)).unwrap();
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
        assert!(!chunk.content.contains("notes") && !chunk.content.contains("characters"), "切片正文不携带路径词");
        let mut tags = chunk.header.tags.clone();
        tags.sort();
        assert_eq!(tags, vec!["characters".to_string(), "notes".to_string(), "overview".to_string(), "人物".to_string()],
            "调用方标签与路径段标签合并去重后挂在每一片上");
    }
    // 查「overview」：正文里全是它，每一片都命中——但同一篇只保留排名最高的一片，
    // 并带上「这一篇共有多少片段命中」，否则这一篇会用自己的几十片占满整个结果列表。
    let hits = kb.search(&SearchRequest { query: "overview".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 1, "同一篇笔记折叠成一条");
    assert_eq!(hits[0].note_chunks, Some(chunks.len()), "并告诉调用方这一篇共有多少片段命中");
    assert!(hits[0].text_score.is_some_and(|score| score > 0.0), "命中的那一片拿到分");
    // 调用方标签拼在第一片上：只有第一片命中。
    let caller_tag_hits = kb.search(&SearchRequest { query: "人物".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert_eq!(caller_tag_hits.len(), 1, "调用方标签只拼在第一片上");
    assert_eq!(caller_tag_hits[0].key.id, chunks[0].header.id, "命中的是第一片");
    // 目录段不参与常规匹配：搜 notes / characters 一片都搜不到。
    for word in ["notes", "characters"] {
        let hits = kb.search(&SearchRequest { query: word.into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
        assert!(hits.is_empty(), "目录段不参与常规匹配：{word}");
    }
    // 按标签筛切片：`characters` 筛得到，库里没有的标签换不到 id 就是空结果。
    let by_tag = |tag: &str| kb.search(&SearchRequest { query: "overview".into(),
        filter: ReadFilter { tags: vec![tag.into()], ..Default::default() }, kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() })
        .unwrap().hits.len();
    assert_eq!(by_tag("characters"), 1, "同一篇折叠成一条，但按目录段筛选仍能定位到它");
    assert_eq!(by_tag("库里没有的标签"), 0);
    // 笔记自己不占索引文档：文档数 = 记录数 − 笔记数。
    let before = kb.health().unwrap();
    assert_eq!(before.index_document_count, before.record_count - 1);
    // 整库重建之后同一批查询结果一致。
    kb.rebuild_indexes().unwrap();
    let again = kb.search(&SearchRequest { query: "overview".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert_eq!(again.len(), hits.len(), "重建后命中条数一致");
    assert_eq!(again[0].key.id, hits[0].key.id, "重建后名次一致");
    assert_eq!(kb.health().unwrap().index_document_count, before.record_count - 1);
}

/// 同一篇笔记的多个命中片段折叠成一条：只留排名最高的一片，
/// 并告诉调用方这一篇共有多少片段命中——这个数与结果窗口、翻页无关。
#[test]
fn chunks_from_one_note_collapse_to_one_hit_with_the_note_total() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
    // 一篇：正文反复出现同一个词，切成多片，每片都命中。
    let many_path = dir.path().join("对话.md");
    std::fs::write(&many_path, "派蒙".repeat(800)).unwrap();
    let many = kb.notes().upsert_file(NoteFileInput::new(&many_path)).unwrap().value;
    let many_chunks = kb.notes().chunks(many.header.id, &ReadFilter::default()).unwrap();
    assert!(many_chunks.len() > 1, "这篇应当切成多片");
    // 另一篇：只命中一片。
    let once_path = dir.path().join("独白.md");
    std::fs::write(&once_path, "派蒙").unwrap();
    let once = kb.notes().upsert_file(NoteFileInput::new(&once_path)).unwrap().value;

    let hits = kb.search(&SearchRequest { query: "派蒙".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 2, "两篇各出且只出一条，多片段那篇不许刷屏");
    let many_hit = hits.iter().find(|hit| hit.record["note_id"] == json!(many.header.id)).unwrap();
    assert_eq!(many_hit.note_chunks, Some(many_chunks.len()), "多片段那篇报出它的片段总数");
    let once_hit = hits.iter().find(|hit| hit.record["note_id"] == json!(once.header.id)).unwrap();
    assert_eq!(once_hit.note_chunks, Some(1), "单片段那篇报 1");

    // 折叠掉的那几片不丢：代表命中带这一篇排名最高的若干片，第 0 条就是本条自身。
    assert_eq!(many_hit.top_chunks.len(), 3, "默认把这一篇排名最高的三片聚合在一条里");
    assert_eq!(many_hit.top_chunks[0].id, many_hit.key.id, "第 0 条就是本条自身");
    assert_eq!(many_hit.top_chunks[0].offset, many_hit.record["offset"].as_u64().unwrap() as usize);
    for chunk in &many_hit.top_chunks {
        let source = many_chunks.iter().find(|candidate| candidate.header.id == chunk.id)
            .expect("聚合进来的都是这一篇的片段");
        assert_eq!(chunk.offset, source.offset, "行号取自 chunks 表，与正文定位一致");
    }
    assert_eq!(once_hit.top_chunks.len(), 1, "只命中一片的篇，聚合里就它自己");
    assert_eq!(once_hit.top_chunks[0].id, once_hit.key.id);

    // 计数是查询本身的属性：把 limit 收到 1，这一篇的计数不因此变小。
    let narrow = kb.search(&SearchRequest { query: "派蒙".into(), kinds: vec![RecordKind::Chunk], limit: 1, ..Default::default() }).unwrap().hits;
    assert_eq!(narrow.len(), 1);
    assert_eq!(narrow[0].note_chunks, Some(many_chunks.len()), "窗口变小，片段计数不变");

    // 聚合条数可控：0 表示只留排名最高的一片，连它自己也不放进聚合里。
    let plain = kb.search(&SearchRequest { query: "派蒙".into(), kinds: vec![RecordKind::Chunk], limit: 50,
        top_chunks_per_note: 0, ..Default::default() }).unwrap().hits;
    let plain_many = plain.iter().find(|hit| hit.record["note_id"] == json!(many.header.id)).unwrap();
    assert!(plain_many.top_chunks.is_empty(), "关掉聚合就不带片段");
    assert_eq!(plain_many.note_chunks, Some(many_chunks.len()), "计数与聚合开关无关");

    // 聚合的输入是本次检索窗口：窗口里只有这一片时，聚合就一片。
    let single = kb.search(&SearchRequest { query: "派蒙".into(), kinds: vec![RecordKind::Chunk],
        limit: 1, candidate_limit: Some(1), ..Default::default() }).unwrap().hits;
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].top_chunks.len(), 1, "窗口里只有这一片，聚合就一片");
    assert_eq!(single[0].top_chunks[0].id, single[0].key.id);

    // 名字与目录列是笔记级的，一篇至多一条，不参与片段计数与聚合。
    let by_name = kb.search(&SearchRequest { query: "对话".into(), kinds: vec![RecordKind::Chunk],
        match_field: MatchField::Name, ..Default::default() }).unwrap().hits;
    assert!(!by_name.is_empty());
    assert!(by_name.iter().all(|hit| hit.note_chunks.is_none()), "只看名字列时不报片段数");
    assert!(by_name.iter().all(|hit| hit.top_chunks.is_empty()), "只看名字列时不聚合片段");
}

/// 送重排的候选在折叠之后产生：同一篇只出现一次，重排预算不重复花在同一篇上。
#[test]
fn rerank_receives_one_candidate_per_note() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
    let path = dir.path().join("对话.md");
    std::fs::write(&path, "派蒙".repeat(800)).unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    assert!(kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap().len() > 1, "这篇应当切成多片");

    let seen: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    kb.register_reranker_with(move |_: &str, documents: &[String]| {
        recorder.lock().unwrap().push(documents.to_vec());
        Ok(vec![0.0f32; documents.len()])
    }, RerankerOptions::default()).unwrap();
    seen.lock().unwrap().clear(); // 注册校验会先调一次，清掉只留检索那次。

    let result = kb.search(&SearchRequest { query: "派蒙".into(), kinds: vec![RecordKind::Chunk],
        limit: 10, ..Default::default() }).unwrap();
    assert!(result.diagnostics.reranked);
    assert_eq!(result.diagnostics.rerank_candidates, 1, "同一篇只送一条进重排");
    assert_eq!(seen.lock().unwrap()[0].len(), 1);
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
    hua.aliases = vec!["alias-one".into()];
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
    assert_eq!(hua.aliases, vec!["alias-one".to_string()], "实体连别名一起给出");

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
    assert!(notes_empty(&result.notes), "RAG 不出笔记那一路");
}

/// 第三步的事件那一路按字符预算截断：预算只装得下前两条时，返回的就是按 id 升序的前两条。
/// 顺带锁住换驱动表后的领域过滤——别处领域里、参与者同样落在这批实体的事件不该被带进来。
#[test]
fn preset_graph_event_route_stops_at_the_character_budget() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let created = kb.graph().apply_batch(&GraphBatch {
        entities: vec![entity("甲"), entity("乙"), entity("丙"), entity("丁")], ..Default::default()
    }).unwrap().value;
    let id = |name: &str| created.entities.iter().find(|entity| entity.name == name).unwrap().header.id;
    kb.graph().apply_batch(&GraphBatch {
        relations: vec![
            relation_of(id("甲"), "同伴", id("乙")),
            relation_of(id("甲"), "同伴", id("丙")),
            relation_of(id("甲"), "同伴", id("丁")),
        ],
        events: vec![
            event_of("事件一", vec![id("甲"), id("乙")]),
            event_of("很长很长很长很长的事件名", vec![id("甲"), id("丙")]),
            event_of("事件三", vec![id("甲"), id("丁")]),
        ], ..Default::default()
    }).unwrap();
    // 先造一条「别处事件」（参与者一样），随后把它挪到另一个可见范围：写入 API 不允许跨范围
    // 参与者，只能建完再改库。它若没被领域过滤掉，会因为 id 最小而排在结果最前面。
    let elsewhere = kb.graph().apply_batch(&GraphBatch {
        events: vec![event_of("别处事件", vec![id("甲"), id("乙")])], ..Default::default()
    }).unwrap().value;
    {
        let conn = rusqlite::Connection::open(dir.path().join("store.sqlite3")).unwrap();
        conn.execute("INSERT OR IGNORE INTO strings(text) VALUES('private')", []).unwrap();
        let scope_id: i64 = conn.query_row("SELECT id FROM strings WHERE text='private'", [], |row| row.get(0)).unwrap();
        conn.execute("UPDATE records SET scope_id=?1 WHERE id=?2",
            rusqlite::params![scope_id, elsewhere.events[0].header.id]).unwrap();
    }

    // 事件正文 = 名字 + 空格 + 摘要 + 空格 + 参与者名 + 空格 + 理由：第一条 9 字符、第二条 18 字符、
    // 第三条 9 字符。预算 20 装得下第一条，第二条一加就超——按「顺序累加、超了就停」的规则，
    // 第三条即使短也不该被跳过补进来。
    let result = kb.search_preset(&PresetRequest {
        preset: SearchPreset::Graph, query: "甲".into(),
        budget: PresetBudget { graph_context_chars: 20, ..Default::default() }, ..Default::default()
    }).unwrap();

    let names: Vec<&str> = result.graph.context_events.iter().map(|event| event.name.as_str()).collect();
    assert_eq!(names, vec!["事件一"], "预算装不下第二条时就停在第一条，不跳过去捡后面的短事件");
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
        entities: vec![entity("alice"), entity("bob")], ..Default::default()
    }).unwrap().value;
    let id = |name: &str| created.entities.iter().find(|entity| entity.name == name).unwrap().header.id;
    kb.graph().apply_batch(&GraphBatch {
        relations: vec![relation_of(id("bob"), "alpha", id("alice"))], ..Default::default()
    }).unwrap();

    // 没登记等价词：剩余词「beta」敲不到正文里的「alpha」。
    let before = kb.search_preset(&PresetRequest {
        preset: SearchPreset::Graph, query: "alice的beta".into(), ..Default::default()
    }).unwrap();
    assert!(!before.graph.relations.iter().any(|r| r.predicate == "alpha"), "没登记等价词时敲不到另一写法");

    // 登记「alpha=beta」后，剩余词扩散带上「alpha」，关系被第二步召回。
    kb.graph().set_predicate_equivalents("default", &[vec!["alpha".into(), "beta".into()]]).unwrap();
    let after = kb.search_preset(&PresetRequest {
        preset: SearchPreset::Graph, query: "alice的beta".into(), ..Default::default()
    }).unwrap();
    assert!(after.graph.relations.iter().any(|r| r.predicate == "alpha"),
        "登记等价词后「beta」扩散出「alpha」，关系应被第二步召回");
}

/// 笔记那一路是否空着：三个块都空才算空。
fn notes_empty(section: &NoteSection) -> bool {
    section.titles.is_empty() && section.contents.is_empty() && section.paths.is_empty()
}

/// 取笔记预设那一路的结果。
fn notes_section(kb: &KnowledgeBase, query: &str) -> NoteSection {
    kb.search_preset(&PresetRequest { preset: SearchPreset::Notes, query: query.into(), ..Default::default() }).unwrap().notes
}

/// 笔记预设：书名块（文件名命中）在上、内容块（正文命中）在下，两者同时出；
/// 只有书名块条数不足时，才用目录段兜底补足、单独成块。
#[test]
fn preset_notes_split_titles_contents_and_path_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let root = dir.path().to_string_lossy().into_owned();
    kb.notes().set_root("default", &root).unwrap();

    // A：文件名就叫「city」，正文与「city」无关 -> 靠书名命中。
    let a_dir = dir.path().join("city");
    std::fs::create_dir_all(&a_dir).unwrap();
    let a_path = a_dir.join("city.md");
    std::fs::write(&a_path, "岩王帝君坐镇此地").unwrap();
    kb.notes().upsert_file(NoteFileInput::new(&a_path)).unwrap();

    // B：正文里出现「city」，文件名无关 -> 靠内容命中。
    let b_dir = dir.path().join("杂记");
    std::fs::create_dir_all(&b_dir).unwrap();
    let b_path = b_dir.join("港口见闻.md");
    std::fs::write(&b_path, "city 港口 今日格外热闹").unwrap();
    kb.notes().upsert_file(NoteFileInput::new(&b_path)).unwrap();

    // C：只有目录段带「沧浪」，文件名与正文都没有 -> 只能靠目录兜底。
    let c_dir = dir.path().join("沧浪");
    std::fs::create_dir_all(&c_dir).unwrap();
    let c_path = c_dir.join("寒天之钉.md");
    std::fs::write(&c_path, "封冻的极北之地一览无余").unwrap();
    kb.notes().upsert_file(NoteFileInput::new(&c_path)).unwrap();

    // 搜「city」：A 进书名块，B 进内容块，两块同时出；B 不再被目录兜底重复一次。
    let by_title = notes_section(&kb, "city");
    assert!(!by_title.titles.is_empty(), "文件名命中的 A 进书名块");
    assert!(!by_title.contents.is_empty(), "正文命中的 B 进内容块");
    let title_ids: Vec<i64> = by_title.titles.iter().map(|hit| hit.key.id).collect();
    let content_ids: Vec<i64> = by_title.contents.iter().map(|hit| hit.key.id).collect();
    assert!(title_ids.iter().all(|id| !content_ids.contains(id)), "同一片不重复出现在两块");
    assert!(by_title.paths.is_empty(), "已被书名或内容覆盖的目录命中不再进兜底块");

    // 常规检索搜「沧浪」搜不到 C，但笔记预设的兜底能把它捞回。
    assert!(kb.search(&SearchRequest { query: "沧浪".into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits.is_empty(),
        "目录段不参与常规匹配");
    let by_path = notes_section(&kb, "沧浪");
    assert!(by_path.titles.is_empty() && by_path.contents.is_empty(), "沧浪只在目录里，书名与内容都不命中");
    assert!(!by_path.paths.is_empty(), "书名块不足时由目录兜底补足");

    // 重建之后同一批查询结果一致。
    kb.rebuild_indexes().unwrap();
    let after = notes_section(&kb, "city");
    assert_eq!(after.titles.iter().map(|hit| hit.key.id).collect::<Vec<_>>(), title_ids, "重建后书名块一致");
    assert_eq!(after.contents.iter().map(|hit| hit.key.id).collect::<Vec<_>>(), content_ids, "重建后内容块一致");
    assert!(!notes_section(&kb, "沧浪").paths.is_empty(), "重建后目录兜底仍生效");
}

/// 单路预设只填自己那一个字段；广撒网三个字段都填。
#[test]
fn preset_fields_stay_separate() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.memories().upsert(memory("预设字段分离用的记忆", "public")).unwrap();
    kb.graph().apply_batch(&GraphBatch { entities: vec![entity("预设字段分离用的实体")], ..Default::default() }).unwrap();
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
    let path = dir.path().join("预设字段分离用的笔记.md");
    std::fs::write(&path, "预设字段分离用的正文".repeat(20)).unwrap();
    kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap();

    let query = "预设字段分离用";
    let of = |preset| kb.search_preset(&PresetRequest { preset, query: query.into(), ..Default::default() }).unwrap();

    let memory_only = of(SearchPreset::Memory);
    assert!(!memory_only.memories.is_empty());
    assert!(memory_only.graph.entities.is_empty() && notes_empty(&memory_only.notes), "记忆预设只填记忆那一个字段");

    let graph_only = of(SearchPreset::Graph);
    assert!(graph_only.memories.is_empty() && notes_empty(&graph_only.notes), "图谱预设只填图谱那一个字段");
    assert!(graph_only.graph.entities.iter().any(|entity| entity.name == "预设字段分离用的实体"));

    let notes_only = of(SearchPreset::Notes);
    assert!(!notes_only.notes.titles.is_empty() || !notes_only.notes.contents.is_empty(), "笔记预设出笔记结果");
    assert!(notes_only.memories.is_empty() && notes_only.graph.entities.is_empty(), "笔记预设只填笔记那一个字段");

    let broad = of(SearchPreset::Broad);
    assert!(!broad.memories.is_empty() && !notes_empty(&broad.notes) && !broad.graph.entities.is_empty(), "广撒网三路都出");
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
    assert!(kb.graph().predicate_equivalents("demo").unwrap().is_empty(), "没登记就是空");

    kb.graph().set_predicate_equivalents("demo", &[vec!["alpha".into(), "beta".into(), "gamma".into()]]).unwrap();
    let groups = kb.graph().predicate_equivalents("demo").unwrap();
    assert_eq!(groups, vec![vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]], "组内按文本排序");

    // 扩散：查询里出现登记词（beta），返回整组同义词。
    let expanded = kb.graph().expand_query("demo", "alice的beta").unwrap();
    assert!(expanded.contains(&"alpha".to_string()) && expanded.contains(&"gamma".to_string()), "命中「beta」应展开出「alpha」「gamma」: {expanded:?}");
    assert!(kb.graph().expand_query("demo", "alice").unwrap().is_empty(), "查询里没有登记词就是空");

    // 领域隔离：别的领域没登记，同一查询扩散不出东西。
    assert!(kb.graph().expand_query("other", "alice的beta").unwrap().is_empty(), "别的领域不共享等价词");
}

/// 谓词等价词接入检索：同义查询在登记后能召回关系，且关系里存着的谓词原文不变。
#[test]
fn predicate_equivalents_expand_search_without_rewriting_storage() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let made = kb.graph().apply_batch(&GraphBatch { entities: vec![entity("bob"), entity("alice")], ..Default::default() }).unwrap();
    let subject = made.value.entities[0].header.id;
    let object = made.value.entities[1].header.id;
    let relation = kb.graph().apply_batch(&GraphBatch { relations: vec![RelationInput {
        record: RecordInput::default(), subject_id: subject, predicate: "delta".into(),
        object_id: object, confidence: 0.9, reason: String::new(),
    }], ..Default::default() }).unwrap().value.relations[0].header.id;

    let request = SearchRequest { query: "epsilon".into(), kinds: vec![RecordKind::Relation],
        text: true, vector: false, rerank: false, ..Default::default() };
    // 未登记：查询词与关系正文没有字面交集，召回不到。
    assert!(kb.search(&request).unwrap().hits.is_empty(), "没登记时同义查询召回不到");
    // 登记一组等价词后，同一查询被扩散到「delta」，命中。
    kb.graph().set_predicate_equivalents("default", &[vec!["delta".into(), "epsilon".into()]]).unwrap();
    let hits = kb.search(&request).unwrap().hits;
    assert_eq!(hits.len(), 1, "登记后同义查询能召回");
    assert_eq!(hits[0].key.id, relation);

    // 落盘不变：关系里存的谓词仍是登记时写的「delta」，扩散没有改写记录。
    let stored: serde_json::Value = kb.graph().get(RecordKind::Relation, relation, &ReadFilter::default()).unwrap();
    assert_eq!(stored["predicate"], serde_json::json!("delta"));
}

// ── 事件流 ────────────────────────────────────────────────────────────

/// 注册 sink 后，一次检索产出一条 `search` 事件，阶段与计数齐全；没走的阶段不出现。
#[test]
fn event_sink_receives_one_search_event_with_stages() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for text in ["事件流 甲", "事件流 乙", "事件流 丙"] { kb.memories().upsert(memory(text, "public")).unwrap(); }
    let events: Arc<Mutex<Vec<LogEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = events.clone();
    kb.register_event_sink(move |event: &LogEvent| recorder.lock().unwrap().push(event.clone()));
    assert!(kb.event_sink_registered());

    let request = SearchRequest { query: "事件流".into(), limit: 3, kinds: vec![RecordKind::Memory], vector: false, ..Default::default() };
    let result = kb.search(&request).unwrap();
    let seen = events.lock().unwrap();
    assert_eq!(seen.len(), 1, "一次检索一条事件");
    let event = &seen[0];
    assert_eq!(event.kind, "search");
    assert!(!event.ts.is_empty());
    assert_eq!(event.hits, Some(result.hits.len()));
    assert_eq!(event.candidates, Some(3));
    assert_eq!(event.folded, Some(3));
    assert_eq!(event.rerank_docs, Some(0), "没启用重排就不送候选");
    assert!(event.degraded.is_empty());
    for stage in ["prepare", "text", "fuse", "fold", "load"] {
        assert!(event.stages.contains_key(stage), "缺阶段 {stage}");
    }
    assert!(!event.stages.contains_key("vector"), "没走向量路就不该有向量格");
    assert!(!event.stages.contains_key("embed"), "没走向量路就不该有嵌入格");
    assert!(!event.stages.contains_key("rerank"), "没启用重排就不该有重排格");
    assert!(event.ms >= event.stages.values().copied().max().unwrap_or(0), "总耗时覆盖各段");
}

/// 启用重排时事件里带重排格，以及实际送进回调的文档数与 token 数。
#[test]
fn event_sink_reports_the_rerank_stage() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for text in ["事件重排 甲", "事件重排 乙", "事件重排 丙"] { kb.memories().upsert(memory(text, "public")).unwrap(); }
    let received: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let recorder = received.clone();
    kb.register_reranker_with(move |_: &str, documents: &[String]| {
        *recorder.lock().unwrap() = documents.len();
        Ok(vec![0.0f32; documents.len()])
    }, RerankerOptions { max_tokens_total: 10, max_tokens_per_doc: 6, ..Default::default() }).unwrap();
    // 注册校验已经调用过回调一次，清掉只留检索那次。
    *received.lock().unwrap() = 0;
    let events: Arc<Mutex<Vec<LogEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = events.clone();
    kb.register_event_sink(move |event: &LogEvent| sink.lock().unwrap().push(event.clone()));

    let request = SearchRequest { query: "事件重排".into(), limit: 3, kinds: vec![RecordKind::Memory], rerank: true, vector: false, ..Default::default() };
    let result = kb.search(&request).unwrap();
    let seen = events.lock().unwrap();
    assert_eq!(seen.len(), 1);
    let event = &seen[0];
    assert!(event.stages.contains_key("rerank"), "启用重排就该有重排格");
    assert_eq!(event.rerank_docs, Some(result.diagnostics.rerank_candidates));
    assert_eq!(event.rerank_docs, Some(*received.lock().unwrap()), "事件里的文档数就是回调实收数");
    assert!(event.rerank_tokens.unwrap_or(0) > 0, "token 预算里有查询词");
}

/// sink 自己炸掉只丢这一条事件，检索照常返回同样的结果。
#[test]
fn event_sink_panic_does_not_break_search() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for text in ["事件抛错 甲", "事件抛错 乙"] { kb.memories().upsert(memory(text, "public")).unwrap(); }
    let request = SearchRequest { query: "事件抛错".into(), limit: 2, kinds: vec![RecordKind::Memory], vector: false, ..Default::default() };
    let baseline: Vec<i64> = kb.search(&request).unwrap().hits.iter().map(|hit| hit.key.id).collect();
    kb.register_event_sink(|_: &LogEvent| panic!("sink 自己炸了"));
    let after = kb.search(&request).unwrap();
    assert_eq!(after.hits.iter().map(|hit| hit.key.id).collect::<Vec<_>>(), baseline);
}

/// 事件不改变任何行为：注册前后同一请求的命中与分数逐项一致；注销后不再产出。
#[test]
fn event_sink_does_not_change_results() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for text in ["事件中立 甲", "事件中立 乙"] { kb.memories().upsert(memory(text, "public")).unwrap(); }
    let request = SearchRequest { query: "事件中立".into(), limit: 2, kinds: vec![RecordKind::Memory], vector: false, ..Default::default() };
    let before = kb.search(&request).unwrap();
    let events: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let counter = events.clone();
    kb.register_event_sink(move |_: &LogEvent| *counter.lock().unwrap() += 1);
    let after = kb.search(&request).unwrap();
    assert_eq!(*events.lock().unwrap(), 1);
    assert_eq!(before.hits.iter().map(|hit| (hit.key.id, hit.score)).collect::<Vec<_>>(),
               after.hits.iter().map(|hit| (hit.key.id, hit.score)).collect::<Vec<_>>());
    assert!(kb.unregister_event_sink());
    assert!(!kb.event_sink_registered());
    kb.search(&request).unwrap();
    assert_eq!(*events.lock().unwrap(), 1, "注销后不再产出");
}

/// 索引重建产出一条 `index_rebuild` 事件，带文档数与格式串。
#[test]
fn event_sink_receives_index_rebuild() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for text in ["事件重建 甲", "事件重建 乙"] { kb.memories().upsert(memory(text, "public")).unwrap(); }
    kb.update_index().unwrap();
    let events: Arc<Mutex<Vec<LogEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = events.clone();
    kb.register_event_sink(move |event: &LogEvent| recorder.lock().unwrap().push(event.clone()));
    let report = kb.rebuild_indexes().unwrap();
    let seen = events.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].kind, "index_rebuild");
    assert_eq!(seen[0].documents, Some(report.index_document_count));
    assert!(seen[0].format.as_deref().unwrap_or_default().starts_with("p-memory-text-"));
    assert!(seen[0].stages.is_empty(), "重建事件不分阶段");
}

/// 删掉的记录必须离开索引：写入路径只按 id 覆盖，删除没有别的地方会把文档摘掉，
/// 留着就会继续参与打分、占住候选配额。
#[test]
fn deleted_records_leave_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let doomed = kb.memories().upsert(memory("删档独有词", "public")).unwrap().value.header.id;
    kb.memories().upsert(memory("保留记录", "public")).unwrap();
    kb.update_index().unwrap();
    assert_eq!(kb.health().unwrap().index_document_count, 2, "两条都进了索引");
    let hits = |query: &str| kb.search(&SearchRequest { query: query.into(), limit: 10, ..Default::default() }).unwrap().hits.len();
    assert_eq!(hits("删档独有词"), 1);

    kb.memories().delete(doomed, &ReadFilter::default()).unwrap();
    kb.update_index().unwrap();
    assert_eq!(kb.health().unwrap().index_document_count, 1, "删掉的记录要从索引里摘掉");
    assert_eq!(hits("删档独有词"), 0, "删掉的记录不能再被搜到");
    assert_eq!(hits("保留记录"), 1, "同批的其他记录不受影响");
}

/// 笔记改写变短之后，废弃的切片记录必须离开索引：它们只被 SQLite 删掉，
/// 留着会继续占候选，还会把这一篇的片段数算大。
#[test]
fn rewritten_note_drops_its_obsolete_chunks_from_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
    let path = dir.path().join("长文.md");
    std::fs::write(&path, "overview".repeat(800)).unwrap();
    let wide = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    let chunks = kb.notes().chunks(wide.header.id, &ReadFilter::default()).unwrap().len();
    assert!(chunks > 1, "这篇应当切成多片");
    kb.update_index().unwrap();
    assert_eq!(kb.health().unwrap().index_document_count, chunks, "多片都在索引里");

    std::fs::write(&path, "overview").unwrap();
    let narrow = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    kb.update_index().unwrap();
    assert_eq!(kb.notes().chunks(narrow.header.id, &ReadFilter::default()).unwrap().len(), 1);
    assert_eq!(kb.health().unwrap().index_document_count, 1, "废弃的切片要从索引里摘掉");

    let hits = kb.search(&SearchRequest { query: "overview".into(), kinds: vec![RecordKind::Chunk], limit: 50, ..Default::default() }).unwrap().hits;
    assert_eq!(hits.len(), 1, "同一篇折叠成一条");
    assert_eq!(hits[0].note_chunks, Some(1), "旧切片不再计入这一篇的片段数");
}

// ── 按笔记限定检索范围 ────────────────────────────────────────────────

/// 建两篇各一片的笔记：正文各含独有词，标签各不相同。返回 (A, B) 的记录 id。
fn two_notes(kb: &KnowledgeBase, dir: &std::path::Path) -> (i64, i64) {
    kb.notes().set_root("default", &dir.to_string_lossy()).unwrap();
    let mut first = NoteFileInput::new(dir.join("a.md"));
    first.record.tags = vec!["地理".into()];
    let mut second = NoteFileInput::new(dir.join("b.md"));
    second.record.tags = vec!["商贸".into()];
    std::fs::write(dir.join("a.md"), "harbor的契约与地契").unwrap();
    std::fs::write(dir.join("b.md"), "harbor的商船与货单").unwrap();
    let a = kb.notes().upsert_file(first).unwrap().value;
    let b = kb.notes().upsert_file(second).unwrap().value;
    kb.update_index().unwrap();
    (a.header.id, b.header.id)
}

/// 一篇笔记的切片记录 id。
fn note_chunks(kb: &KnowledgeBase, note_id: i64) -> Vec<i64> {
    kb.notes().chunks(note_id, &ReadFilter::default()).unwrap().into_iter().map(|chunk| chunk.header.id).collect()
}

/// 只在切片上做全文检索，返回命中的记录 id。
fn chunk_hits(kb: &KnowledgeBase, query: &str, filter: ReadFilter) -> Vec<i64> {
    kb.search(&SearchRequest { query: query.into(), kinds: vec![RecordKind::Chunk], limit: 50, filter, ..Default::default() })
        .unwrap().hits.into_iter().map(|hit| hit.key.id).collect()
}

/// 限定到给定笔记后，命中只可能来自这批笔记。
#[test]
fn search_can_be_limited_to_the_given_notes() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let (a, b) = two_notes(&kb, dir.path());
    let (in_a, in_b) = (note_chunks(&kb, a), note_chunks(&kb, b));
    let all = chunk_hits(&kb, "harbor", ReadFilter::default());
    assert!(in_a.iter().all(|id| all.contains(id)) && in_b.iter().all(|id| all.contains(id)), "不限定范围时两篇都命中");
    let scoped = chunk_hits(&kb, "harbor", ReadFilter { note_ids: vec![a], ..Default::default() });
    assert!(!scoped.is_empty(), "限定到目标笔记后仍应有命中");
    assert!(scoped.iter().all(|id| in_a.contains(id)), "命中必须全部落在被限定的笔记里");
    let both = chunk_hits(&kb, "harbor", ReadFilter { note_ids: vec![a, b], ..Default::default() });
    assert!(both.len() >= scoped.len(), "给多篇时覆盖不少于单篇");
    assert!(both.iter().all(|id| in_a.contains(id) || in_b.contains(id)), "多篇之间是并集");
}

/// 空集与不传等价：笔记限定是叠加维度，不是模式开关。
#[test]
fn omitting_the_note_limit_keeps_the_old_behaviour() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    two_notes(&kb, dir.path());
    let omitted = chunk_hits(&kb, "harbor", ReadFilter::default());
    let empty = chunk_hits(&kb, "harbor", ReadFilter { note_ids: vec![], ..Default::default() });
    assert!(!omitted.is_empty(), "全域检索本来就该有命中");
    assert_eq!(empty, omitted, "空集与不传必须逐条一致");
}

/// 笔记限定与标签取交集：另一篇上的标签不该把命中放回来。
#[test]
fn the_note_limit_intersects_with_tags() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let (a, _) = two_notes(&kb, dir.path());
    let same_note = chunk_hits(&kb, "harbor", ReadFilter { note_ids: vec![a], tags: vec!["地理".into()], ..Default::default() });
    assert!(!same_note.is_empty(), "目标笔记自己的标签不挡命中");
    let other = chunk_hits(&kb, "harbor", ReadFilter { note_ids: vec![a], tags: vec!["商贸".into()], ..Default::default() });
    assert!(other.is_empty(), "标签取自另一篇时交集为空");
}

/// 目标笔记里没有命中、或给了一个不存在的笔记 id，都给空结果而不是报错。
#[test]
fn a_note_without_hits_returns_empty_instead_of_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let (a, _) = two_notes(&kb, dir.path());
    assert!(!chunk_hits(&kb, "商船", ReadFilter::default()).is_empty(), "这个词全域搜得到");
    assert!(chunk_hits(&kb, "商船", ReadFilter { note_ids: vec![a], ..Default::default() }).is_empty(), "目标笔记里没有这个词");
    assert!(chunk_hits(&kb, "harbor", ReadFilter { note_ids: vec![i64::MAX], ..Default::default() }).is_empty(), "不存在的笔记 id 给空结果");
}

/// 向量路同样受笔记限定：候选在打分阶段就被收窄，而不是取回来再筛。
#[test]
fn the_vector_path_honours_the_note_limit() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb, "v", 4);
    kb.embeddings().set_vectorization("default", "notes", true).unwrap();
    let (a, b) = two_notes(&kb, dir.path());
    fill(&kb, "v");
    assert!(kb.embeddings().vector_ready("default", "v", "notes").unwrap(), "切片向量补齐后才走向量路");
    let (in_a, in_b) = (note_chunks(&kb, a), note_chunks(&kb, b));
    // 先确认不限定范围时两篇都进得来，否则「只剩 A」说明不了限定起了作用。
    let all = kb.search(&vector_query("v", "harbor", vec![RecordKind::Chunk])).unwrap().hits;
    assert!(all.iter().any(|hit| in_a.contains(&hit.key.id)) && all.iter().any(|hit| in_b.contains(&hit.key.id)),
        "不限定范围时两篇的切片都参与打分");
    let mut request = vector_query("v", "harbor", vec![RecordKind::Chunk]);
    request.filter.note_ids = vec![a];
    let scoped = kb.search(&request).unwrap().hits;
    assert!(!scoped.is_empty(), "限定后向量路仍应有命中");
    assert!(scoped.iter().all(|hit| in_a.contains(&hit.key.id)), "向量候选也必须来自被限定的笔记");
}

// ── 按过滤批量删除 ────────────────────────────────────────────────────

fn tagged_entity(name: &str, tags: &[&str]) -> EntityInput {
    let mut value = entity(name);
    value.record.tags = tags.iter().map(|tag| (*tag).to_string()).collect();
    value
}

/// 关系必须引用已存在的实体 id，所以先写实体、拿到 id 再写边。
fn write_entity(kb: &KnowledgeBase, input: EntityInput) -> i64 {
    kb.graph().apply_batch(&GraphBatch { entities: vec![input], ..Default::default() }).unwrap().value.entities[0].header.id
}

fn write_relation(kb: &KnowledgeBase, subject: i64, predicate: &str, object: i64) {
    let relation = RelationInput { record: RecordInput::default(), subject_id: subject,
        predicate: predicate.into(), object_id: object, confidence: 0.8, reason: String::new() };
    kb.graph().apply_batch(&GraphBatch { relations: vec![relation], ..Default::default() }).unwrap();
}

/// 记忆域：按域一次清空，返回条数，列表随之变空。
#[test]
fn a_batch_delete_clears_the_whole_memory_domain() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let inputs: Vec<MemoryInput> = (1..=3).map(|i| memory(&format!("待清的记忆 {i}"), "public")).collect();
    kb.memories().upsert_many(&inputs).unwrap();
    assert_eq!(kb.memories().delete_by_filter(&ReadFilter::default()).unwrap().value, 3);
    assert!(kb.memories().list(&PageRequest::default()).unwrap().items.is_empty(), "清空之后一条都不剩");
}

/// 标签收窄时只删命中的那些，没带标签的记录原样留下。
#[test]
fn a_tag_narrows_the_batch_delete() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let mut tagged = memory("带标签的记忆", "public");
    tagged.record.tags = vec!["draft".into()];
    kb.memories().upsert_many(&[tagged, memory("留下的记忆", "public")]).unwrap();
    let filter = ReadFilter { tags: vec!["draft".into()], ..Default::default() };
    assert_eq!(kb.memories().delete_by_filter(&filter).unwrap().value, 1);
    let left = kb.memories().list(&PageRequest::default()).unwrap().items;
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].judgment, "留下的记忆");
}

/// 空命中不是错误：没有匹配就返回 0。
#[test]
fn an_empty_match_deletes_nothing_and_returns_zero() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    assert_eq!(kb.memories().delete_by_filter(&ReadFilter::default()).unwrap().value, 0);
    assert_eq!(kb.graph().delete_by_filter(&ReadFilter::default()).unwrap().value, 0);
    assert_eq!(kb.notes().delete_by_filter(&ReadFilter::default()).unwrap().value, 0);
}

/// 图谱域：清一个域时实体与它的边一起走，不留下半截状态。
#[test]
fn clearing_a_graph_domain_takes_its_edges_along() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let subject = write_entity(&kb, entity("甲"));
    let object = write_entity(&kb, entity("乙"));
    write_relation(&kb, subject, "创造", object);
    // 计数是实体加关系，参与行与边本身是级联产物。
    assert_eq!(kb.graph().delete_by_filter(&ReadFilter::default()).unwrap().value, 3);
    for kind in [RecordKind::Entity, RecordKind::Relation, RecordKind::Event] {
        assert!(kb.graph().list(kind, &PageRequest::default()).unwrap().items.is_empty(), "域清空后 {kind:?} 一条不剩");
    }
}

/// 过滤条件之外的边还指着待删实体时，删除被外键拦下，整个事务回滚。
#[test]
fn a_batch_delete_is_refused_when_an_out_of_scope_edge_still_points_at_the_entity() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let subject = write_entity(&kb, tagged_entity("甲", &["keep"]));
    let object = write_entity(&kb, entity("乙"));
    write_relation(&kb, subject, "创造", object);
    // 只有实体带标签，过滤条件命中的实体、不命中它那条边。
    let filter = ReadFilter { tags: vec!["keep".into()], ..Default::default() };
    assert!(matches!(kb.graph().delete_by_filter(&filter), Err(Error::Conflict(_))), "边还在，删实体必须被拦下");
    assert!(kb.graph().get(RecordKind::Entity, subject, &ReadFilter::default()).is_ok(), "事务回滚，实体仍在");
}

/// 笔记域：删笔记连同它的切片，切片不会变成孤儿。
#[test]
fn clearing_a_note_domain_takes_its_chunks_along() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let path = dir.path().join("n.md");
    std::fs::write(&path, "笔记正文里的独有措辞").unwrap();
    kb.notes().set_root("default", &dir.path().to_string_lossy()).unwrap();
    let note = kb.notes().upsert_file(NoteFileInput::new(&path)).unwrap().value;
    let chunk = kb.notes().chunks(note.header.id, &ReadFilter::default()).unwrap()[0].header.id;
    // 计数只算笔记本身，切片是级联产物。
    assert_eq!(kb.notes().delete_by_filter(&ReadFilter::default()).unwrap().value, 1);
    assert!(kb.notes().list(&PageRequest::default()).unwrap().items.is_empty());
    assert!(kb.notes().get_chunk(chunk, &ReadFilter::default()).is_err(), "切片随笔记一起删掉");
}

/// 批量删除同样要进索引追平：删掉的记录不能再被检索到。
#[test]
fn the_index_forgets_batch_deleted_records() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.memories().upsert(memory("批量删除后不该被检索到的措辞", "public")).unwrap();
    kb.update_index().unwrap();
    let query = SearchRequest { query: "批量删除后".into(), ..Default::default() };
    assert!(!kb.search(&query).unwrap().hits.is_empty(), "删之前检索得到");
    kb.memories().delete_by_filter(&ReadFilter::default()).unwrap();
    kb.update_index().unwrap();
    assert!(kb.search(&query).unwrap().hits.is_empty(), "删之后索引里也没有了");
}

/// 谓词等价登记要能撤：按词删只动它自己，清整域一次清空。
#[test]
fn predicate_equivalents_can_be_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.graph().set_predicate_equivalents("demo", &[
        vec!["alpha".into(), "beta".into(), "gamma".into()],
        vec!["delta".into(), "epsilon".into()],
    ]).unwrap();
    assert_eq!(kb.graph().predicate_equivalents("demo").unwrap().len(), 2, "两组");
    // 删一个词，它所属的组缩小，别组不动。
    let only = vec!["gamma".to_string()];
    assert_eq!(kb.graph().delete_predicate_equivalents("demo", Some(&only)).unwrap().value, 1);
    let groups = kb.graph().predicate_equivalents("demo").unwrap();
    assert_eq!(groups.len(), 2, "同组其余词还在");
    assert!(!groups.iter().flatten().any(|term| term == "gamma"), "gamma 已经不在表里");
    // 没登记过的词删不动，也不报错。
    let unknown = vec!["zeta".to_string()];
    assert_eq!(kb.graph().delete_predicate_equivalents("demo", Some(&unknown)).unwrap().value, 0);
    // 不给词就是清整域。
    assert_eq!(kb.graph().delete_predicate_equivalents("demo", None).unwrap().value, 4);
    assert!(kb.graph().predicate_equivalents("demo").unwrap().is_empty(), "整域清空");
}

/// 谓词元规则要能撤，内置的 sys:same_as 同样能撤。
#[test]
fn predicate_rules_can_be_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.graph().set_predicate_rule("father", Some("child"), false).unwrap();
    assert!(kb.graph().delete_predicate_rule("father").unwrap().value);
    assert!(!kb.graph().delete_predicate_rule("father").unwrap().value, "再删就是没命中");
    assert!(!kb.graph().delete_predicate_rule("never-registered").unwrap().value);
    assert!(kb.graph().delete_predicate_rule("sys:same_as").unwrap().value, "内置规则也要能撤");
}

/// 笔记根目录登记要能注销，注销后这个领域回到「没登记」的状态。
#[test]
fn a_notes_root_can_be_deregistered() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let root = dir.path().join("docs");
    std::fs::create_dir_all(&root).unwrap();
    kb.notes().set_root("demo", &root.to_string_lossy()).unwrap();
    assert!(kb.notes().root("demo").unwrap().is_some());
    assert!(kb.notes().unset_root("demo").unwrap());
    assert!(kb.notes().root("demo").unwrap().is_none());
    assert!(!kb.notes().unset_root("demo").unwrap(), "再注销就是没命中");
}

/// 记录的作用域可以改；被关系引用的实体不能直接换域，得先解除引用。
#[test]
fn a_record_can_change_scope() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    let id = kb.memories().upsert(memory("换作用域", "public")).unwrap().value.header.id;
    let mut moved = memory("换作用域", "private");
    moved.record.id = Some(id);
    kb.memories().upsert(moved).unwrap();
    let private = ReadFilter { namespace: "default".into(), scopes: vec!["private".into()], ..Default::default() };
    assert_eq!(kb.memories().get(id, &private).unwrap().header.scope, "private");

    let entities = kb.graph().apply_batch(&GraphBatch { entities: vec![entity("alice"), entity("bob")], ..Default::default() })
        .unwrap().value.entities;
    kb.graph().apply_batch(&GraphBatch { relations: vec![RelationInput { record: RecordInput::default(),
        subject_id: entities[0].header.id, predicate: "knows".into(), object_id: entities[1].header.id,
        confidence: 1.0, reason: String::new() }], ..Default::default() }).unwrap();
    let mut linked = entity("alice");
    linked.record.id = Some(entities[0].header.id);
    linked.record.scope = "private".into();
    assert!(kb.graph().apply_batch(&GraphBatch { entities: vec![linked], ..Default::default() }).is_err(),
        "被关系引用的实体不能直接换域");
}
