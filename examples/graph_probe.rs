//! 图搜索探针：打开已导入的开发库，跑邻域 / 桥接 / 连通诊断。
//!
//! 用法：cargo run --example graph_probe -- <库目录> <实体A> <实体B>

use p_memory::types::ReadFilter;
use p_memory::KnowledgeBase;

fn filter() -> ReadFilter {
    ReadFilter { namespace: "default".into(), scopes: vec!["public".into()], tags: vec![], note_ids: vec![] }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| ".pai/temp/graph_kb".into());
    let a = args.next().unwrap_or_else(|| "实体A".into());
    let b = args.next().unwrap_or_else(|| "实体B".into());

    let kb = KnowledgeBase::open(&dir).expect("open kb");
    let view = kb.graph().build_graph(&filter()).expect("build graph");
    println!(
        "图：{} 节点 / {} 边 / {} 连通分量",
        view.node_count(),
        view.edge_count(),
        view.component_count()
    );

    let id_of = |name: &str| kb.graph().resolve(name, &filter(), 1).expect("resolve").first().map(|e| e.header.id);
    let ida = id_of(&a);
    let idb = id_of(&b);
    if let Some(ida) = ida {
        println!(
            "{a} 1/2/3 跳可达：{} / {} / {}",
            view.ego_ids(ida, 1, 100_000).len(),
            view.ego_ids(ida, 2, 100_000).len(),
            view.ego_ids(ida, 3, 100_000).len()
        );
        let sample: Vec<String> = kb
            .graph()
            .ego(ida, 1, &filter(), 8)
            .unwrap()
            .into_iter()
            .map(|e| format!("{}[{}]", e.name, e.entity_type))
            .collect();
        println!("{a} 一跳样本（Entity 级）：{}", sample.join("、"));
    } else {
        println!("没解析到「{a}」");
    }
    let scc = kb.graph().strongly_connected(&filter()).expect("scc");
    let max = scc.iter().map(|c| c.len()).max().unwrap_or(0);
    println!("强连通分量（>1）：{} 个，最大 {} 个实体", scc.len(), max);
    match (ida, idb) {
        (Some(x), Some(y)) => match view.path_ids(x, y) {
            Some(path) => println!("{a} → {b} 桥接：经过 {} 个节点（{} 步）", path.len(), path.len() - 1),
            None => println!("{a} 与 {b} 不连通"),
        },
        _ => println!("端点缺失，跳过桥接"),
    }
}
