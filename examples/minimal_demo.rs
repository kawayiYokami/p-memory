//! 最小端到端 Demo。
//!
//! 运行：`cargo run --example minimal_demo`
//!
//! 它只做一件事：把一小份记忆与档案写进库，注册一个向量模型回调和一个重排模型回调，
//! 然后直接拿搜索结果。向量化与重排都在库内部完成——宿主只提供内容与模型接口，
//! 不自己算向量，也不自己重排。它跑通，就说明整条路径没有漏。

use p_memory::{EmbedCallbackError, EmbeddingSpace, KnowledgeBase, MemoryInput, NoteFileInput,
               ReadFilter, RecordKind, SearchRequest};

/// 确定性假向量：同一文本永远得到同一向量，也不为零。Demo 零外部依赖、结果可复现，
/// 但真实走完回调链路——注册校验、分批、截断、降级都在里面跑。
fn fake_embedding(text: &str, dimension: usize) -> Vec<f32> {
    let mut values = vec![0f32; dimension];
    for (index, byte) in text.bytes().enumerate() {
        values[(index + usize::from(byte)) % dimension] += 1.0 + f32::from(byte % 7) * 0.1;
    }
    if values.iter().all(|value| *value == 0.0) { values[0] = 1.0; }
    values
}

/// 确定性假重排：按查询词里出现在文档中的字符数给分。
fn fake_relevance(query: &str, document: &str) -> f32 {
    query.chars().filter(|ch| !ch.is_whitespace() && document.contains(*ch)).count() as f32
}

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let kb = KnowledgeBase::open(dir.path()).expect("open knowledge base");

    // 一个向量模型 = 一个向量空间。注册即用样本真跑一遍校验，产出不符该空间契约会被拒绝绑定。
    let dimension = 32;
    kb.embeddings()
        .register_space(EmbeddingSpace {
            id: "demo-v1".into(),
            model: "demo-embed".into(),
            dimension,
            text_version: 1,
            encoding: "f32".into(),
        })
        .expect("register space");
    kb.embeddings()
        .register_embedder("demo-v1", move |texts: &[String]| -> std::result::Result<Vec<Vec<f32>>, EmbedCallbackError> {
            Ok(texts.iter().map(|text| fake_embedding(text, dimension)).collect())
        })
        .expect("register embedder");

    // 重排：库按声明的定长约束截断候选与文档后再调用，宿主不自己重排。
    kb.register_reranker(|query: &str, documents: &[String]| -> std::result::Result<Vec<f32>, String> {
            Ok(documents.iter().map(|document| fake_relevance(query, document)).collect())
        })
        .expect("register reranker");

    // 一小份记忆与档案：只写内容，向量化由库在写入路径里完成。
    for (judgment, tag) in [
        ("红豆喜欢简短直接的回答", "偏好"),
        ("p-memory 是嵌入式记忆库", "项目"),
        ("搜索优先返回精确结果", "原则"),
    ] {
        let mut memory = MemoryInput::new(judgment);
        memory.record.tags = vec![tag.to_string()];
        kb.memories().upsert(memory).expect("upsert memory");
    }
    let note_path = dir.path().join("demo.md");
    std::fs::write(&note_path, "档案：最小 Demo 只用一小份样本。\n\n它直接完成搜索，不做多余处理。").expect("write note file");
    kb.notes()
        .upsert_file(NoteFileInput::new(&note_path))
        .expect("upsert note");

    // 直接检索：宿主只给搜索词与目标空间，库用它注册的回调嵌入查询词。
    let request = SearchRequest {
        query: "p-memory 搜索".into(),
        filter: ReadFilter { namespace: "default".into(), scopes: vec!["public".into()], tags: vec![], note_ids: vec![] },
        kinds: vec![RecordKind::Memory, RecordKind::Note, RecordKind::Chunk],
        embed_space: Some("demo-v1".into()),
        limit: 5,
        with_total: true,
        ..Default::default()
    };
    let result = kb.search(&request).expect("search");
    let diagnostics = &result.diagnostics;

    println!("命中 {} 条（过滤后总量 {:?}）", result.hits.len(), result.total);
    println!("走了哪几条路：全文={} 向量={} 重排={}", diagnostics.text_used, diagnostics.vector_used, diagnostics.reranked);
    if diagnostics.rerank_truncated > 0 {
        println!("重排截断：{} 条候选未送入重排", diagnostics.rerank_truncated);
    }
    if !diagnostics.degraded.is_empty() {
        println!("降级档位：{:?}", diagnostics.degraded);
    }
    for (rank, hit) in result.hits.iter().enumerate() {
        let label = hit.record.get("judgment")
            .or_else(|| hit.record.get("source"))
            .and_then(|value| value.as_str())
            .unwrap_or("<记录>");
        println!("{}. #{} {}  score={:.4} rerank={:?}", rank + 1, hit.key.id, label, hit.score, hit.rerank_score);
    }

    // 多档降级：注销嵌入回调后再搜同一个词。空间还注册着、向量开关也开着，但回调不在了，
    // 于是落到「无回调」档走纯全文——结果照常返回，降级档位可观测。
    kb.embeddings().unregister_embedder("demo-v1").expect("unregister embedder");
    let degraded = kb.search(&request).expect("search without embedder");
    let flags = &degraded.diagnostics;
    println!("--- 注销嵌入回调后 ---");
    println!("命中 {} 条，走了哪几条路：全文={} 向量={} 降级={:?}",
        degraded.hits.len(), flags.text_used, flags.vector_used, flags.degraded);
}
