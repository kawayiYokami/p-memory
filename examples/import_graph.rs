//! 把一个 name 引用的 GraphBatch JSON 导入 p-memory 开发库，并跑图搜索验收。
//!
//! 用法：cargo run --example import_graph -- <json路径> <库目录>

use p_memory::graph::{EntityInput, GraphBatch, RelationInput};
use p_memory::types::{ReadFilter, RecordInput};
use p_memory::KnowledgeBase;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

fn record() -> RecordInput {
    let mut r = RecordInput::default();
    r.namespace = "default".into();
    r.scope = "public".into();
    r
}

fn filter() -> ReadFilter {
    ReadFilter { namespace: "default".into(), scopes: vec!["public".into()], tags: vec![], note_ids: vec![] }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let json_path = args.next().unwrap_or_else(|| ".pai/temp/graph.json".into());
    let dir = args.next().unwrap_or_else(|| ".pai/temp/graph_kb".into());
    let raw = std::fs::read_to_string(&json_path).expect("read json");
    let data: Value = serde_json::from_str(&raw).expect("parse json");
    let kb = KnowledgeBase::open(&dir).expect("open kb");

    // ---- 实体 ----
    let entities = data["entities"].as_array().cloned().unwrap_or_default();
    let mut name_to_id: HashMap<String, i64> = HashMap::new();
    let mut dup_names = 0usize;
    let mut batch: Vec<EntityInput> = Vec::new();
    for e in &entities {
        let name = e["name"].as_str().unwrap_or("").to_string();
        let etype = e["type"].as_str().unwrap_or("concept").to_string();
        let aliases: Vec<String> = e["aliases"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let mut attrs: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if let Some(list) = e["attributes"].as_array() {
            for a in list {
                if let (Some(k), Some(v)) = (a["key"].as_str(), a["value"].as_str()) {
                    attrs.entry(k.to_string()).or_default().push(v.to_string());
                }
            }
        }
        batch.push(EntityInput {
            record: record(),
            name,
            entity_type: etype,
            aliases,
            attributes: attrs,
            summary: String::new(),
        });
    }
    for chunk in batch.chunks(1000) {
        let res = kb
            .graph()
            .apply_batch(&GraphBatch { entities: chunk.to_vec(), ..Default::default() })
            .expect("apply entities");
        for ent in res.value.entities {
            if name_to_id.insert(ent.name.clone(), ent.header.id).is_some() {
                dup_names += 1;
            }
        }
    }
    println!("实体导入 {} 个（同名覆盖 {dup_names}）", batch.len());

    // ---- 关系 ----
    let relations = data["relations"].as_array().cloned().unwrap_or_default();
    let mut rel_batch: Vec<RelationInput> = Vec::new();
    let mut missing = 0usize;
    let mut missing_examples: Vec<String> = Vec::new();
    for r in &relations {
        let s = r["subject"].as_str().unwrap_or("");
        let p = r["predicate"].as_str().unwrap_or("");
        let o = r["object"].as_str().unwrap_or("");
        match (name_to_id.get(s), name_to_id.get(o)) {
            (Some(&si), Some(&oi)) if !p.is_empty() => rel_batch.push(RelationInput {
                record: record(),
                subject_id: si,
                predicate: p.into(),
                object_id: oi,
                confidence: 0.8,
                reason: String::new(),
            }),
            _ => {
                missing += 1;
                if missing_examples.len() < 5 {
                    missing_examples.push(format!("{s} -{p}-> {o}"));
                }
            }
        }
    }
    for chunk in rel_batch.chunks(1000) {
        kb.graph()
            .apply_batch(&GraphBatch { relations: chunk.to_vec(), ..Default::default() })
            .expect("apply relations");
    }
    println!("关系导入 {} 条（端点缺失跳过 {missing}）", rel_batch.len());
    for ex in &missing_examples {
        println!("  缺: {ex}");
    }

    // ---- 图搜索验收 ----
    {
        let view = kb.graph().build_graph(&filter()).expect("build graph");
        println!(
            "\n图：{} 节点 / {} 边 / {} 连通分量",
            view.node_count(),
            view.edge_count(),
            view.component_count()
        );
        if let Some(z) = kb.graph().resolve("实体A", &filter(), 1).expect("resolve").first() {
            for d in 1..=3 {
                println!("实体A {d} 跳可达 {}", view.ego_ids(z.header.id, d, 100000).len());
            }
        } else {
            println!("没解析到「实体A」");
        }
    }
}
