use p_memory::*;
use p_memory::graph::{EntityInput, EventInput, RelationInput};
use p_memory::notes::chunk_text;
use serde_json::json;
use std::collections::BTreeMap;

fn memory(text: &str, scope: &str) -> MemoryInput {
    let mut value = MemoryInput::new(text);
    value.record.scope = scope.into(); value
}
fn entity(name: &str) -> EntityInput {
    EntityInput { record: RecordInput::default(), name: name.into(), entity_type: "person".into(),
        aliases: vec![], attributes: BTreeMap::new(), summary: String::new() }
}
fn space(kb: &KnowledgeBase) {
    kb.embeddings().register_space(EmbeddingSpace { id: "test".into(), model: "fixture/v1".into(), dimension: 2, text_version: 1, encoding: "f32".into() }).unwrap();
}
fn pending(kb: &KnowledgeBase, filter: ReadFilter) -> Vec<EmbeddingInput> {
    kb.embeddings().pending("test", &PageRequest { filter, ..Default::default() }, &[]).unwrap().items
}

#[test]
fn transactions_scopes_pagination_and_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    assert!(matches!(KnowledgeBase::open(dir.path()), Err(Error::Locked(_))));
    let mut first = memory("上海 茶 Rust ＡＰＩ", "public");
    first.record.tags = vec![" RUST ".into(), "rust".into()];
    let receipt = kb.memories().upsert(first.clone()).unwrap();
    assert!(receipt.index_ready);
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
fn vectors_are_atomic_stale_safe_and_filtered_before_top_k() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb);
    for _ in 0..110 { kb.memories().upsert(memory("private target", "private")).unwrap(); }
    let visible_id = kb.memories().upsert(memory("public target", "public")).unwrap().value.header.id;
    let inputs = pending(&kb, ReadFilter::default()); let visible = &inputs[0];
    let write = EmbeddingWrite { key: visible.key, fingerprint: visible.fingerprint.clone(), values: vec![1.0, 1.0] };
    assert!(matches!(kb.embeddings().put("test", &[EmbeddingWrite { values: vec![0.0,0.0], ..write.clone() }]), Err(Error::InvalidVector(_))));
    assert!(kb.embeddings().put("test", &[write.clone(), EmbeddingWrite { values: vec![f32::NAN,1.0], ..write.clone() }]).is_err());
    assert_eq!(pending(&kb, ReadFilter::default()).len(), 1);
    let mut writes = vec![write.clone()];
    let private = kb.embeddings().pending("test", &PageRequest { filter: ReadFilter { scopes: vec!["private".into()], ..Default::default() }, limit: 200, ..Default::default() }, &[]).unwrap();
    writes.extend(private.items.into_iter().map(|i| EmbeddingWrite { key:i.key, fingerprint:i.fingerprint, values:vec![1.0,0.0] }));
    kb.embeddings().put("test", &writes).unwrap();
    let req = SearchRequest { query: "target".into(), limit:1, candidate_limit:Some(1), vectors: vec![QueryVector { space_id:"test".into(), values:vec![1.0,0.0], weight:1.0,min_score:None }], ..Default::default() };
    let hits = kb.search(&req).unwrap().hits;
    assert_eq!(hits.len(),1); assert_eq!(hits[0].key.id, visible_id);
    assert!(hits[0].text_score.is_some()); assert!((hits[0].score-2.0/61.0).abs()<1e-9);
    let mut changed = memory("changed", "public"); changed.record.id = Some(visible_id);
    kb.memories().upsert(changed).unwrap();
    assert!(matches!(kb.embeddings().put("test", &[write]), Err(Error::StaleRevision(_))));
    let vector_only = SearchRequest {query:String::new(), ..req};
    assert!(kb.search(&vector_only).unwrap().hits.is_empty());
    assert_eq!(pending(&kb, ReadFilter::default()).len(),1);
    assert!(matches!(kb.embeddings().register_space(EmbeddingSpace { id:"test".into(),model:"different".into(),dimension:2,text_version:1,encoding:"f32".into() }), Err(Error::Conflict(_))));
}

#[test]
fn vector_partitions_are_isolated_by_scope_and_cached_per_scope() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb);
    let public_id = kb.memories().upsert(memory("public vector", "public")).unwrap().value.header.id;
    let private_id = kb.memories().upsert(memory("private vector", "private")).unwrap().value.header.id;
    // 两个 scope 各写一条方向不同的向量，用分数就能判断命中的是哪个分区。
    for (scope, values) in [("public", vec![1.0, 0.0]), ("private", vec![0.0, 1.0])] {
        let filter = ReadFilter { scopes: vec![scope.into()], ..Default::default() };
        let inputs = pending(&kb, filter);
        assert_eq!(inputs.len(), 1);
        kb.embeddings().put("test", &inputs.iter().map(|v| EmbeddingWrite { key: v.key, fingerprint: v.fingerprint.clone(), values: values.clone() }).collect::<Vec<_>>()).unwrap();
    }
    let query = |scope: &str| SearchRequest {
        query: String::new(),
        filter: ReadFilter { scopes: vec![scope.into()], ..Default::default() },
        vectors: vec![QueryVector { space_id: "test".into(), values: vec![1.0, 0.0], weight: 1.0, min_score: Some(0.5) }],
        ..Default::default()
    };
    let public = kb.search(&query("public")).unwrap().hits;
    assert_eq!(public.len(), 1); assert_eq!(public[0].key.id, public_id);
    // 不限分数时 private 查询只能看到 private 分区的向量，public 的不能串进来。
    let mut all_scores = query("private");
    all_scores.vectors[0].min_score = None;
    let private = kb.search(&all_scores).unwrap().hits;
    assert!(private.iter().all(|h| h.key.id == private_id), "public 分区的向量混进了 private 查询");
    let mut flipped = query("private");
    flipped.vectors[0].values = vec![0.0, 1.0];
    let private = kb.search(&flipped).unwrap().hits;
    assert_eq!(private.len(), 1); assert_eq!(private[0].key.id, private_id);
    // 空分区（无任何向量的 scope）按空结果缓存，不应报错，也不该回退到别的分区。
    assert!(kb.search(&query("empty")).unwrap().hits.is_empty());
}

#[test]
fn packed_and_precise_spaces_rank_the_same_vectors_together() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap();
    for (id, encoding) in [("precise", "f32"), ("packed", "sq8")] {
        kb.embeddings().register_space(EmbeddingSpace { id: id.into(), model: "fixture/v1".into(), dimension: 8, text_version: 1, encoding: encoding.into() }).unwrap();
    }
    for i in 0..8 { kb.memories().upsert(memory(&format!("记录{i}"), "public")).unwrap(); }
    let vector = |i: usize| (0..8).map(|d| ((i as f32) * 0.7 + (d as f32) * 0.3).sin() + i as f32 * 0.2).collect::<Vec<f32>>();
    for space in ["precise", "packed"] {
        let inputs = kb.embeddings().pending(space, &PageRequest { limit: 50, ..Default::default() }, &[]).unwrap().items;
        assert_eq!(inputs.len(), 8);
        let writes: Vec<_> = inputs.iter().enumerate()
            .map(|(i, v)| EmbeddingWrite { key: v.key, fingerprint: v.fingerprint.clone(), values: vector(i) })
            .collect();
        kb.embeddings().put(space, &writes).unwrap();
    }
    // 查询取第 3 条的向量，两个空间用的向量完全相同，只有打分路径不同。
    let request = |space: &str| SearchRequest {
        query: String::new(),
        vectors: vec![QueryVector { space_id: space.into(), values: vector(3), weight: 1.0, min_score: None }],
        limit: 8,
        ..Default::default()
    };
    let precise = kb.search(&request("precise")).unwrap().hits;
    let packed = kb.search(&request("packed")).unwrap().hits;
    let precise_ids: Vec<i64> = precise.iter().map(|h| h.key.id).collect();
    let packed_ids: Vec<i64> = packed.iter().map(|h| h.key.id).collect();
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
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb);
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
    let inputs = pending(&kb, ReadFilter::default());
    kb.embeddings().put("test", &inputs.iter().map(|v| EmbeddingWrite { key:v.key, fingerprint:v.fingerprint.clone(), values:vec![1.0,0.0] }).collect::<Vec<_>>()).unwrap();
    let mut renamed = entity("Carol"); renamed.record.id = Some(a);
    kb.graph().apply_batch(&GraphBatch { entities: vec![renamed], ..Default::default() }).unwrap();
    let kinds: Vec<_> = pending(&kb, ReadFilter::default()).into_iter().map(|v|v.key).collect();
    assert_eq!(kinds.len(), 3);
    let search = SearchRequest {query:"Carol".into(),kinds:vec![RecordKind::Relation,RecordKind::Event],..Default::default()};
    assert_eq!(kb.search(&search).unwrap().hits.len(), 2);
    let bad = RelationInput { object_id: 9_999_999, ..relation };
    assert!(kb.graph().apply_batch(&GraphBatch { entities: vec![], relations: vec![bad], events: vec![] }).is_err());
}

#[test]
fn note_replacement_keeps_evidence_and_removes_stale_chunks() {
    let dir = tempfile::tempdir().unwrap(); let kb = KnowledgeBase::open(dir.path()).unwrap(); space(&kb);
    let input = NoteInput::new("docs/tea.md", "# 茶\n\n上海喝茶\n\n```rust\nlet x = 1;\n```\n\n最后一段");
    let note = kb.notes().upsert(input).unwrap().value;
    let note_id = note.header.id;
    let chunks = kb.notes().chunks(note_id, &ReadFilter::default()).unwrap();
    let tea = chunks.iter().find(|c| c.content == "上海喝茶").unwrap();
    assert_eq!((tea.offset, tea.limit), (3, 1));
    assert!(chunks.iter().any(|c| c.content.contains("```rust") && c.offset == 5 && c.limit == 3));
    let evidence = Evidence { source: "docs/tea.md".into(), offset: Some(3), limit: Some(1), quote: "上海喝茶".into(), ..Default::default() };
    let mut m = memory("source fact", "public"); m.record.evidence = vec![evidence];
    let memory_id = kb.memories().upsert(m).unwrap().value.header.id;
    let new = kb.notes().upsert(NoteInput::new("docs/tea.md", "replacement")).unwrap().value;
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
    kb.memories().upsert(memory("recoverable","public")).unwrap();space(&kb);
    let i=pending(&kb,ReadFilter::default()).remove(0);kb.embeddings().put("test",&[EmbeddingWrite {key:i.key,fingerprint:i.fingerprint,values:vec![1.0,0.0]}]).unwrap();
    let backup=root.path().join("backup.sqlite3");kb.backup(&backup).unwrap();assert!(kb.backup(&backup).is_err());kb.close().unwrap();
    std::fs::write(data.join("text-v2/meta.json"),b"broken index metadata").unwrap();
    let reopened=KnowledgeBase::open(&data).unwrap();assert_eq!(reopened.search(&SearchRequest {query:"recoverable".into(),..Default::default()}).unwrap().hits.len(),1);
    let restored=KnowledgeBase::restore(&backup,root.path().join("restored")).unwrap();
    assert!(pending(&restored,ReadFilter::default()).is_empty());assert_eq!(restored.health().unwrap().record_count,1);
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
    let note = kb.notes().upsert(NoteInput::new("绝区零/角色/雅.md", "这是一段无关内容")).unwrap().value;
    let hits = kb.search(&SearchRequest { query: "角色".into(), kinds: vec![RecordKind::Chunk], ..Default::default() }).unwrap().hits;
    assert!(hits.iter().any(|h| h.record["note_id"] == json!(note.header.id)));
}

#[test]
fn sq8_encoding_roundtrips_through_storage() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    kb.embeddings().register_space(EmbeddingSpace { id: "q".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "sq8".into() }).unwrap();
    assert!(kb.embeddings().register_space(EmbeddingSpace { id: "bad".into(), model: "fixture/v1".into(), dimension: 4, text_version: 1, encoding: "q4".into() }).is_err());
    let id = kb.memories().upsert(memory("quantized target", "public")).unwrap().value.header.id;
    let inputs = kb.embeddings().pending("q", &PageRequest::default(), &[]).unwrap().items;
    assert_eq!(inputs.len(), 1);
    // 归一化后 [0.6, 0.8, 0, 0]；sq8 往返后仍应被同一查询命中。
    kb.embeddings().put("q", &[EmbeddingWrite { key: inputs[0].key, fingerprint: inputs[0].fingerprint.clone(), values: vec![3.0, 4.0, 0.0, 0.0] }]).unwrap();
    let req = SearchRequest { query: String::new(), vectors: vec![QueryVector { space_id: "q".into(), values: vec![0.6, 0.8, 0.0, 0.0], weight: 1.0, min_score: None }], ..Default::default() };
    let hits = kb.search(&req).unwrap().hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key.id, id);
    assert!((hits[0].score - 1.0 / 61.0).abs() < 1e-9);
}

#[test]
fn graph_prune_limits_vector_scoring_to_neighborhood() {
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open(dir.path()).unwrap();
    space(&kb);
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
    let inputs = pending(&kb, ReadFilter::default());
    kb.embeddings().put("test", &inputs.iter().map(|i| EmbeddingWrite { key: i.key, fingerprint: i.fingerprint.clone(), values: vec![1.0, 0.0] }).collect::<Vec<_>>()).unwrap();

    let base = SearchRequest {
        query: String::new(), kinds: vec![RecordKind::Entity], limit: 10,
        vectors: vec![QueryVector { space_id: "test".into(), values: vec![1.0, 0.0], weight: 1.0, min_score: None }],
        ..Default::default()
    };
    assert_eq!(kb.search(&base).unwrap().hits.len(), 3);

    // 剪枝到 A 的 1 跳邻域：只剩 A（起点）与 B，孤立的 C 被排除
    let pruned = SearchRequest { prune: Some(GraphPrune { root: a, depth: 1, limit: 64 }), ..base.clone() };
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
