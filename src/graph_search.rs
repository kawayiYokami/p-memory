//! 图谱搜索：基于 petgraph 的**内存态图**。
//!
//! 存储仍是 SQLite（权威）；本模块按需把 `relations` 投影表读进内存建图，
//! 调 petgraph 现成算法，用完即弃。算法一律用 petgraph 实现，不自己造。

use crate::graph::{Entity, GraphStore};
use crate::storage::{self, filter_sql};
use crate::types::{ReadFilter, RecordKind, WriteReceipt};
use crate::{Error, Result};
use petgraph::algo::{astar, connected_components, tarjan_scc};
use petgraph::graph::{DiGraph, Graph, NodeIndex};
use rusqlite::{params, params_from_iter, Connection};
use std::collections::{BTreeMap, HashMap, HashSet};

/// 别名等价关系的内置谓词：建图时并查集缩点，把互为别名的实体折成同一超节点。
const SAME_AS: &str = "sys:same_as";

/// 查询期并查集：只服务 `sys:same_as` 的缩点，用完即弃，不落盘。
#[derive(Default)]
struct UnionFind { parent: HashMap<i64, i64> }

impl UnionFind {
    fn find(&mut self, x: i64) -> i64 {
        if !self.parent.contains_key(&x) { self.parent.insert(x, x); return x; }
        let mut root = x;
        while self.parent[&root] != root { root = self.parent[&root]; }
        let mut cursor = x;
        while self.parent[&cursor] != root { let next = self.parent[&cursor]; self.parent.insert(cursor, root); cursor = next; }
        root
    }
    fn union(&mut self, a: i64, b: i64) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb { self.parent.insert(ra, rb); }
    }
}

/// 从 `relations` 构建的内存态图。节点权重 = 实体 record_id，边权重 = 谓词文本。
///
/// 图是**有向**的（petgraph `Graph` 默认 `Directed`，`neighbors` 只返回出边）：
/// 反向查询（如由「父亲」反查「子女」）不会自动连通，靠 `predicate_rules` 在 `snapshot`
/// 里补出对称/逆关系的虚拟边来打通，物理表不落双向边。
///
/// 节点 id 是**缩点后的代表 id**：互为 `sys:same_as` 的实体被并查集折叠成同一节点，
/// `canonical` 保存「原始 id -> 代表 id」的映射，查询入口先做一次换算。
pub struct GraphView {
    graph: Graph<i64, String>,
    index: HashMap<i64, NodeIndex>,
    canonical: HashMap<i64, i64>,
}

/// 读一次快照：`filter` 范围内的实体 id，以及有向边 (subject, object, predicate)。
///
/// 边里已按 `predicate_rules` 补入内存虚拟边——对称谓词补反向同谓词边、有逆谓词的补对偶边。
fn snapshot(conn: &Connection, filter: &ReadFilter) -> Result<(Vec<i64>, Vec<(i64, i64, String)>)> {
    let (cond, values) = filter_sql(filter, &[RecordKind::Entity], false)?;
    let mut stmt = conn.prepare(&format!(
        "SELECT e.record_id FROM entities e JOIN records r ON r.id=e.record_id WHERE {cond}"
    ))?;
    let nodes = stmt
        .query_map(params_from_iter(values), |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let (cond, values) = filter_sql(filter, &[RecordKind::Relation], false)?;
    let mut stmt = conn.prepare(&format!(
        "SELECT rel.subject_id, rel.object_id, s.text, pr.is_symmetric, inv.text \
         FROM relations rel JOIN records r ON r.id=rel.record_id \
         JOIN strings s ON s.id=rel.predicate_id \
         LEFT JOIN predicate_rules pr ON pr.predicate_id=rel.predicate_id \
         LEFT JOIN strings inv ON inv.id=pr.inverse_predicate_id WHERE {cond}"
    ))?;
    let mut edges: Vec<(i64, i64, String)> = Vec::new();
    let mut rows = stmt.query(params_from_iter(values))?;
    while let Some(row) = rows.next()? {
        let subject: i64 = row.get(0)?;
        let object: i64 = row.get(1)?;
        let predicate: String = row.get(2)?;
        let symmetric: Option<i64> = row.get(3)?;
        let inverse: Option<String> = row.get(4)?;
        edges.push((subject, object, predicate.clone()));
        if symmetric == Some(1) {
            edges.push((object, subject, predicate));
        } else if let Some(inverse) = inverse {
            edges.push((object, subject, inverse));
        }
    }
    Ok((nodes, edges))
}

/// 按 `sys:same_as` 求每个 id 的代表 id（含 nodes 与所有边端点）。
fn canonical_ids(nodes: &[i64], edges: &[(i64, i64, String)]) -> HashMap<i64, i64> {
    let mut uf = UnionFind::default();
    for (s, o, pred) in edges {
        if pred == SAME_AS { uf.union(*s, *o); }
    }
    let mut canonical: HashMap<i64, i64> = HashMap::new();
    for id in nodes.iter().copied().chain(edges.iter().flat_map(|(s, o, _)| [*s, *o])) {
        let representative = uf.find(id);
        canonical.insert(id, representative);
    }
    canonical
}

impl GraphStore {
    /// 把 `filter` 范围（namespace/scope/tag）内的实体铺成节点、关系连成边，建一张有向图。
    ///
    /// 节点来自 entity 记录（含没有关系的孤立实体，「谁没连进主图」才看得出来），
    /// 边来自 relation 记录。范围外的记录不进图——多命名空间不会串。
    pub fn build_graph(&self, filter: &ReadFilter) -> Result<GraphView> {
        let state = self.0.read()?;
        let (nodes, edges) = snapshot(state.conn(), filter)?;
        let canonical = canonical_ids(&nodes, &edges);
        let mut graph = Graph::<i64, String>::new();
        let mut index: HashMap<i64, NodeIndex> = HashMap::new();
        for raw in nodes.iter().copied().chain(edges.iter().flat_map(|(s, o, _)| [*s, *o])) {
            let representative = canonical[&raw];
            if !index.contains_key(&representative) {
                let i = graph.add_node(representative);
                index.insert(representative, i);
            }
        }
        for (s, o, predicate) in edges {
            if predicate == SAME_AS { continue; }
            graph.add_edge(index[&canonical[&s]], index[&canonical[&o]], predicate);
        }
        Ok(GraphView { graph, index, canonical })
    }

    /// 强连通分量（有向，`tarjan_scc`）：互相可达的实体环，如「互为对手/同伙」的闭环。
    /// 只返回大小 > 1 的分量，孤点自环被滤掉。
    pub fn strongly_connected(&self, filter: &ReadFilter) -> Result<Vec<Vec<i64>>> {
        let (nodes, edges) = {
            let state = self.0.read()?;
            snapshot(state.conn(), filter)?
        };
        let canonical = canonical_ids(&nodes, &edges);
        let mut graph = DiGraph::<i64, ()>::new();
        let mut index: HashMap<i64, NodeIndex> = HashMap::new();
        for raw in nodes.iter().copied().chain(edges.iter().flat_map(|(s, o, _)| [*s, *o])) {
            let representative = canonical[&raw];
            if !index.contains_key(&representative) {
                let i = graph.add_node(representative);
                index.insert(representative, i);
            }
        }
        for (s, o, _pred) in edges {
            let (si, oi) = (index[&canonical[&s]], index[&canonical[&o]]);
            if si != oi { graph.add_edge(si, oi, ()); }
        }
        Ok(tarjan_scc(&graph)
            .into_iter()
            .filter(|c| c.len() > 1)
            .map(|c| c.into_iter().map(|n| graph[n]).collect())
            .collect())
    }

    /// 从 `root` 出发 `depth` 跳内的实体（带 name / aliases / attributes）。
    pub fn ego(&self, root: i64, depth: usize, filter: &ReadFilter, limit: usize) -> Result<Vec<Entity>> {
        let ids = self.build_graph(filter)?.ego_ids(root, depth, limit);
        let state = self.0.read()?;
        // 一次批量取回：逐条 `get` 会为每个实体各跑一遍过滤与装配。
        let mut loaded: BTreeMap<i64, Entity> = storage::load_many(state.conn(), &ids, filter)?;
        Ok(ids.iter().filter_map(|id| loaded.remove(id)).collect())
    }

    /// `from` → `to` 桥接路径上的实体（含两端）。不连通返回 None。
    pub fn path(&self, from: i64, to: i64, filter: &ReadFilter) -> Result<Option<Vec<Entity>>> {
        let Some(ids) = self.build_graph(filter)?.path_ids(from, to) else {
            return Ok(None);
        };
        let state = self.0.read()?;
        let mut loaded: BTreeMap<i64, Entity> = storage::load_many(state.conn(), &ids, filter)?;
        Ok(Some(ids.iter().filter_map(|id| loaded.remove(id)).collect()))
    }

    /// 登记谓词元规则：`symmetric` 声明对称谓词（反向即自身），`inverse` 声明逆谓词
    /// （反向补一条对偶边，如 `父亲` 的逆是 `子女`）。二者互斥，规则只影响建图时的内存补边。
    pub fn set_predicate_rule(&self, predicate: &str, inverse: Option<&str>, symmetric: bool) -> Result<WriteReceipt<()>> {
        storage::validate_identity("predicate", predicate)?;
        if symmetric && inverse.is_some() {
            return Err(Error::Validation("a symmetric predicate must not declare an inverse".into()));
        }
        if let Some(inverse) = inverse { storage::validate_identity("inverse predicate", inverse)?; }
        self.0.mutate_meta(|tx| {
            let predicate_id = storage::term_id(tx, predicate)?;
            let inverse_id = match inverse { Some(text) => Some(storage::term_id(tx, text)?), None => None };
            tx.execute("INSERT INTO predicate_rules(predicate_id,inverse_predicate_id,is_symmetric) VALUES (?1,?2,?3)
                ON CONFLICT(predicate_id) DO UPDATE SET inverse_predicate_id=excluded.inverse_predicate_id,is_symmetric=excluded.is_symmetric",
                params![predicate_id, inverse_id, i64::from(symmetric)])?;
            Ok(())
        })
    }
}

impl GraphView {
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// 从 `root` 出发 `depth` 跳内的实体 id（不含 root）。`root` 会先折算成缩点后的代表 id。
    pub fn ego_ids(&self, root: i64, depth: usize, limit: usize) -> Vec<i64> {
        let root = self.canonical.get(&root).copied().unwrap_or(root);
        let start = match self.index.get(&root) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        seen.insert(start);
        let mut frontier = vec![start];
        let mut out: Vec<i64> = Vec::new();
        for _ in 0..depth {
            let mut next = Vec::new();
            for n in &frontier {
                for nb in self.graph.neighbors(*n) {
                    if seen.insert(nb) {
                        out.push(self.graph[nb]);
                        next.push(nb);
                        if out.len() >= limit {
                            return out;
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        out
    }

    /// `from` 到 `to` 的最短路径（节点 id 列表，含两端）。用 petgraph 的 `astar`。
    /// 两端会先折算成缩点后的代表 id。
    pub fn path_ids(&self, from: i64, to: i64) -> Option<Vec<i64>> {
        let from = self.canonical.get(&from).copied().unwrap_or(from);
        let to = self.canonical.get(&to).copied().unwrap_or(to);
        let start = self.index.get(&from)?;
        let goal = self.index.get(&to)?;
        let (_cost, path) = astar(&self.graph, *start, |n| n == *goal, |_e| 1i32, |_n| 0i32)?;
        Some(path.into_iter().map(|n| self.graph[n]).collect())
    }

    /// 连通分量数量（无向）。用 petgraph 的 `connected_components`。
    pub fn component_count(&self) -> usize {
        connected_components(&self.graph)
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{EntityInput, GraphBatch, RelationInput};
    use crate::types::{ReadFilter, RecordInput};
    use crate::KnowledgeBase;
    use std::collections::BTreeMap;

    fn ent(name: &str) -> EntityInput {
        EntityInput {
            record: RecordInput::default(),
            name: name.into(),
            entity_type: "person".into(),
            aliases: Vec::new(),
            attributes: BTreeMap::new(),
            summary: String::new(),
        }
    }

    fn rel(s: i64, p: &str, o: i64) -> RelationInput {
        RelationInput {
            record: RecordInput::default(),
            subject_id: s,
            predicate: p.into(),
            object_id: o,
            confidence: 1.0,
            reason: String::new(),
        }
    }

    #[test]
    fn ego_path_components() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let ents = kb
            .graph()
            .apply_batch(&GraphBatch { entities: vec![ent("A"), ent("B"), ent("C")], ..Default::default() })
            .unwrap()
            .value
            .entities;
        let (a, b, c) = (ents[0].header.id, ents[1].header.id, ents[2].header.id);
        kb.graph()
            .apply_batch(&GraphBatch { relations: vec![rel(a, "knows", b), rel(b, "knows", c)], ..Default::default() })
            .unwrap();

        let view = kb.graph().build_graph(&ReadFilter::default()).unwrap();
        assert_eq!(view.node_count(), 3);
        assert_eq!(view.edge_count(), 2);
        assert_eq!(view.component_count(), 1);

        // 邻域：A 的一跳只有 B，两跳兜住 B、C
        assert_eq!(view.ego_ids(a, 1, 100), vec![b]);
        let mut ego = view.ego_ids(a, 2, 100);
        ego.sort();
        let mut expect = vec![b, c];
        expect.sort();
        assert_eq!(ego, expect);

        // 桥接：A→C 的最短路径
        assert_eq!(view.path_ids(a, c), Some(vec![a, b, c]));

        // Entity 级查询：返回的实体对象自带 name 字段
        let names: Vec<String> = kb
            .graph()
            .ego(a, 2, &ReadFilter::default(), 100)
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(names.contains(&"B".to_string()) && names.contains(&"C".to_string()));
        let hop: Vec<String> = kb
            .graph()
            .path(a, c, &ReadFilter::default())
            .unwrap()
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(hop, vec!["A", "B", "C"]);

        // 强连通：补 B→A 形成环，SCC 找到互相可达的 {A,B}
        kb.graph()
            .apply_batch(&GraphBatch { relations: vec![rel(b, "knows", a)], ..Default::default() })
            .unwrap();
        let scc = kb.graph().strongly_connected(&ReadFilter::default()).unwrap();
        assert_eq!(scc.len(), 1);
        let mut ring = scc[0].clone();
        ring.sort();
        let mut exp = vec![a, b];
        exp.sort();
        assert_eq!(ring, exp);

        // 孤立实体独立成块
        kb.graph()
            .apply_batch(&GraphBatch { entities: vec![ent("D")], ..Default::default() })
            .unwrap();
        let view = kb.graph().build_graph(&ReadFilter::default()).unwrap();
        assert_eq!(view.node_count(), 4);
        assert_eq!(view.component_count(), 2);
    }

    #[test]
    fn filter_isolates_namespaces() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();

        let mut ea = ent("A");
        ea.record.namespace = "a".into();
        let mut eb = ent("B");
        eb.record.namespace = "a".into();
        let mut ec = ent("C");
        ec.record.namespace = "b".into();
        let got = kb
            .graph()
            .apply_batch(&GraphBatch { entities: vec![ea, eb, ec], ..Default::default() })
            .unwrap()
            .value
            .entities;
        let (a, b) = (got[0].header.id, got[1].header.id);
        let mut r = rel(a, "knows", b);
        r.record.namespace = "a".into();
        kb.graph()
            .apply_batch(&GraphBatch { relations: vec![r], ..Default::default() })
            .unwrap();

        let fa = ReadFilter { namespace: "a".into(), scopes: vec!["public".into()], tags: vec![], note_ids: vec![] };
        let fb = ReadFilter { namespace: "b".into(), scopes: vec!["public".into()], tags: vec![], note_ids: vec![] };
        let va = kb.graph().build_graph(&fa).unwrap();
        assert_eq!((va.node_count(), va.edge_count()), (2, 1));
        let vb = kb.graph().build_graph(&fb).unwrap();
        assert_eq!((vb.node_count(), vb.edge_count()), (1, 0));
    }

    #[test]
    fn virtual_inverse_and_symmetric_edges_enable_reverse_queries() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let filter = ReadFilter::default();
        let ents = kb
            .graph()
            .apply_batch(&GraphBatch { entities: vec![ent("张伟"), ent("张父"), ent("甲"), ent("乙")], ..Default::default() })
            .unwrap()
            .value
            .entities;
        let (son, father, jia, yi) = (ents[0].header.id, ents[1].header.id, ents[2].header.id, ents[3].header.id);
        // 物理只写单向边：张伟 --父亲--> 张父；甲 --同事--> 乙
        kb.graph()
            .apply_batch(&GraphBatch { relations: vec![rel(son, "父亲", father), rel(jia, "同事", yi)], ..Default::default() })
            .unwrap();

        // 未登记规则：反向查询天然不通（有向图）
        let view = kb.graph().build_graph(&filter).unwrap();
        assert!(view.ego_ids(father, 1, 100).is_empty());
        assert_eq!(view.path_ids(father, son), None);

        // 登记逆谓词：反向补出「子女」边，反向 ego/path 打通
        kb.graph().set_predicate_rule("父亲", Some("子女"), false).unwrap();
        let view = kb.graph().build_graph(&filter).unwrap();
        assert_eq!(view.ego_ids(father, 1, 100), vec![son]);
        assert_eq!(view.path_ids(father, son), Some(vec![father, son]));

        // 登记对称谓词：反向即自身
        kb.graph().set_predicate_rule("同事", None, true).unwrap();
        let view = kb.graph().build_graph(&filter).unwrap();
        assert_eq!(view.ego_ids(yi, 1, 100), vec![jia]);

        // 对称谓词声明逆谓词属于非法组合
        assert!(kb.graph().set_predicate_rule("同事", Some("同事"), true).is_err());
    }

    #[test]
    fn same_as_contracts_alias_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let filter = ReadFilter::default();
        let ents = kb
            .graph()
            .apply_batch(&GraphBatch { entities: vec![ent("乙"), ent("乙先生"), ent("导师")], ..Default::default() })
            .unwrap()
            .value
            .entities;
        let (lin, alias, mentor) = (ents[0].header.id, ents[1].header.id, ents[2].header.id);
        // 别名等价用 sys:same_as 显式关系表达，绝不物理合并实体
        kb.graph()
            .apply_batch(&GraphBatch { relations: vec![rel(lin, "sys:same_as", alias), rel(alias, "导师", mentor)], ..Default::default() })
            .unwrap();

        let view = kb.graph().build_graph(&filter).unwrap();
        // 三个实体折叠成两个节点：别名与主实体是同一超节点
        assert_eq!(view.node_count(), 2);
        assert_eq!(view.edge_count(), 1);
        // 别名不再打断邻域：乙与乙先生的一跳都直接到导师
        assert_eq!(view.ego_ids(lin, 1, 100), vec![mentor]);
        assert_eq!(view.ego_ids(alias, 1, 100), vec![mentor]);
        // 路径里不再夹着别名中间节点：只剩两个节点
        let path = view.path_ids(lin, mentor).unwrap();
        assert_eq!(path.len(), 2);
        assert!(path.contains(&mentor));
    }
}
