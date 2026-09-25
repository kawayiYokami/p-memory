use crate::{storage, text, types::*, Error, Result};
use parking_lot::Mutex;
use rusqlite::Connection;
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use tantivy::{collector::{Collector, DocSetCollector, SegmentCollector, TopDocs, sort_key::{SortBySimilarityScore, SortByStaticFastValue}}, columnar::Column, directory::MmapDirectory, doc, DocId, Order, Score, SegmentOrdinal, SegmentReader,
    query::{AllQuery, BoostQuery, BooleanQuery, ConstScoreQuery, Occur, Query, TermQuery},
    schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value as TantivyValue, FAST, INDEXED, STORED},
    tokenizer::WhitespaceTokenizer, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

pub(crate) const FORMAT: &str = "p-memory-text-v13";

struct Fields { key: Field, namespace: Field, scope: Field, kind: Field, tags: Field, text: Field, name: Field, path: Field, note: Field, body: Field }

/// 名字字段命中时的固定加权：规范名单独成列后，名字整段命中是最强的相关信号，给它固定倍数抬高。
const NAME_FIELD_BOOST: f32 = 3.0;

/// 索引 writer 的内存预算，只作单线程写器的内存天花板。
/// 每次提交即刷盘，段不驻留，预留一份固定余量即可，不随语料规模变化。
const WRITER_MEMORY_BUDGET: usize = 1_000_000_000;

/// 一条要写进索引的记录：正文、标签全部由写入流程就地提供。
/// 写入时读一次源文件、切一次，切片正文一路带到这里，索引阶段不再回头读文件。
pub(crate) struct IndexDocument {
    pub id: i64,
    pub namespace_id: i64,
    pub scope_id: i64,
    pub kind: RecordKind,
    /// 正文列：这条记录自己的文本。记忆是 judgment，切片是它那一段。笔记不进索引。
    pub text: String,
    /// 实体规范名：单独一列，检索时按 `NAME_FIELD_BOOST` 加权。其它记录为空串、不写这一列。
    pub name: String,
    /// 笔记所在目录：相对路径里除文件名之外的各段，空格连接。只有切片记录写这一列，
    /// 常规检索不查它——它是「书名块不够时」才动用的兜底。
    pub path: String,
    /// 切片所属笔记的记录 id（只挂切片，其它记录为 0）：用来按笔记精确统计「还有多少片段命中」。
    pub note_id: i64,
    /// 拼在可搜正文前面的标签集（空格连接，空串表示不带）。索引里没有单独的标签文本列：
    /// 标签只有承载它的那一条带——切片是第一片，其余记录是它自己；取值规则见 `storage::tags_prefix`。
    pub tags_prefix: String,
    /// 标签 id 列：只用来按标签过滤，不参与打分。
    pub tag_ids: Vec<i64>,
}

/// 已在库侧折算成 id 的检索范围。索引里的 namespace / scope / kind / tags 一律是整数，
/// 标记文本换成 id 这件事在进索引之前做完。
pub(crate) struct IndexFilter { pub namespace: i64, pub scopes: Vec<i64>, pub kinds: Vec<i64>, pub tags: Vec<i64>, pub note_ids: Vec<i64> }

/// 文本索引。`IndexReader` 可并发检索，`IndexWriter` 收进内部互斥锁：
/// 整个结构可以直接共享给多个读线程，提交只在写者之间串行。
pub(crate) struct TextIndex {
    reader: IndexReader, writer: Mutex<IndexWriter>, fields: Fields,
    /// writer 里是否攒有尚未提交的增删。无变更时 `sync` 连提交都不做。
    /// 与 writer 同一把锁下读写，避免与并发的 `stage` 竞态。
    dirty: AtomicBool,
    /// 测试专用：注入查询故障，验证「查询失败即降级」的路径。
    #[cfg(test)] pub(crate) fail_search: AtomicBool,
}

impl TextIndex {
    pub fn open(root: &Path) -> Result<Self> {
        let directory = root.join("text-v2");
        std::fs::create_dir_all(&directory)?;
        let mut builder = Schema::builder();
        // 分词字段共用一套预分词 + 空格分词器：写入侧切成词元后按空格连接，
        // 查询侧用同一套规则产出词元，两边严格同源。
        let tokenized = || TextOptions::default().set_indexing_options(TextFieldIndexing::default()
            .set_tokenizer("pretokenized").set_index_option(IndexRecordOption::WithFreqsAndPositions));
        let fields = Fields {
            // 记录 id 的整数快字段：既做精确匹配（删除、取正文），又做同分时的确定性排序键。
            // 存字符串时同分比较要逐条查字典比字符串，候选上千条时它是全流程最贵的一步。
            key: builder.add_u64_field("key", INDEXED | STORED | FAST),
            // 标记列一律存 strings 表的整数 id：索引里不留第二份标记文本。
            namespace: builder.add_u64_field("namespace", INDEXED),
            scope: builder.add_u64_field("scope", INDEXED),
            kind: builder.add_u64_field("kind", INDEXED),
            // 标签 id 列（多值）：按标签过滤只在这一列上做精确匹配，不参与打分。
            tags: builder.add_u64_field("tags", INDEXED),
            // 正文列：唯一承载「这条记录讲了什么」的列，也是相关性打分的主力。
            // 标签集拼在承载它的那一条的正文前面，与正文同列、同长度、共同参与打分。
            // 实体规范名不进这一列。
            text: builder.add_text_field("text", tokenized()),
            // 实体规范名单独一列：正文列不含规范名，名字只活在这一列，检索时按固定倍数加权，
            // 并作为独立的命中条件——名字命中也算这条记录被搜到。笔记的文件名也落在这一列。
            name: builder.add_text_field("name", tokenized()),
            // 笔记所在目录单独一列：目录段不进正文列（否则搜「city」会命中该目录下每一篇），
            // 只在「书名块不够」的兜底查询里被查。
            path: builder.add_text_field("path", tokenized()),
            // 切片所属笔记的记录 id：只挂在切片上，用来数「这一篇里有多少切片命中」。
            // 这条计数与结果窗口无关，所以只能在索引里按笔记精确统计，不能靠截断后的结果集去数。
            // 存成快字段是为了按笔记分桶计数：一次遍历匹配集就能读出每篇的片数，
            // 不必为每篇各发一次查询——逐篇查询的成本几乎全是每次 `search()` 的固定开销。
            note: builder.add_u64_field("note", INDEXED | FAST),
            // 正文原值随命中取回：库里不留正文副本，取正文只走这一列。
            body: builder.add_text_field("body", STORED),
        };
        let schema = builder.build();
        let open = || -> Result<Index> {
            let dir = MmapDirectory::open(&directory).map_err(|e| Error::Index(e.to_string()))?;
            Ok(Index::open_or_create(dir, schema.clone())?)
        };
        let index = match open() {
            Ok(index) => index,
            Err(_) => {
                // Quarantine only this derived index, never the authoritative database.
                std::fs::rename(&directory, root.join(format!("text-v2.corrupt-{}", uuid::Uuid::new_v4())))?;
                std::fs::create_dir(&directory)?;
                open()?
            }
        };
        index.tokenizers().register("pretokenized", WhitespaceTokenizer::default());
        // 索引 writer 固定单线程：本机实测（8 核 16 线程，20 万条）1 线程最快 9.3s，
        // 2/4/8/16 线程依次退化到 11.0/14.1/19.1/32.8s——多 worker 只增加协调与合并开销，
        // 所以不交给 tantivy 按核数拆线程。
        let writer = index.writer_with_num_threads(1, WRITER_MEMORY_BUDGET)?;
        let reader = index.reader_builder().reload_policy(ReloadPolicy::Manual).try_into()?;
        // writer / reader 各自持有索引句柄，本结构无需再留一份 `Index`。
        Ok(Self { reader, writer: Mutex::new(writer), fields,
            dirty: AtomicBool::new(false),
            #[cfg(test)] fail_search: AtomicBool::new(false) })
    }

    pub fn document_count(&self) -> usize { self.reader.searcher().num_docs() as usize }

    /// 索引里实际存在的记录 id 集合：扫 `key` 快字段，不解码任何文档。
    fn indexed_ids(&self) -> Result<HashSet<i64>> {
        let searcher = self.reader.searcher();
        let mut columns: HashMap<u32, Column<u64>> = HashMap::new();
        let mut out = HashSet::new();
        for address in searcher.search(&AllQuery, &DocSetCollector)? {
            if !columns.contains_key(&address.segment_ord) {
                let column = searcher.segment_reader(address.segment_ord).fast_fields().u64("key")?;
                columns.insert(address.segment_ord, column);
            }
            if let Some(value) = columns[&address.segment_ord].first(address.doc_id) { out.insert(value as i64); }
        }
        Ok(out)
    }

    /// 让索引向主库收敛：把索引里实际存在的记录 id 与主库现存的记录 id 求差集——
    /// 索引里多出来的（记录已删）按 `key` 摘掉，缺的（上次提交没跟上）就地补上。
    /// 这是索引与主库之间唯一的同步方式：不记待办、不做「推倒重来」。
    /// 稳态下差集为空，连一次提交都不发生；删除一批记录时也只摘对应的那几个 doc。
    pub fn reconcile(&self, conn: &Connection) -> Result<()> {
        let indexed = self.indexed_ids()?;
        let live: HashMap<i64, (i64, i64, i64, String)> = {
            let mut stmt = conn.prepare("SELECT id,namespace_id,kind,scope_id,payload_json FROM records WHERE kind<>?1")?;
            let rows = stmt.query_map([RecordKind::Note.code()], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?, r.get::<_, i64>(3)?, r.get::<_, String>(4)?)))?;
            let mut map = HashMap::new();
            for row in rows { let (id, ns, kind, scope, payload) = row?; map.insert(id, (ns, kind, scope, payload)); }
            map
        };
        let removals: Vec<i64> = indexed.iter().filter(|id| !live.contains_key(id)).copied().collect();
        let additions: Vec<i64> = live.keys().filter(|id| !indexed.contains(id)).copied().collect();
        if removals.is_empty() && additions.is_empty() { return Ok(()); }
        let mut writer = self.writer.lock();
        for id in removals { writer.delete_term(Term::from_field_u64(self.fields.key, id as u64)); }
        let mut files: HashMap<i64, Vec<String>> = HashMap::new();
        let mut paths: HashMap<i64, (Vec<String>, String)> = HashMap::new();
        for id in additions {
            let Some((namespace_id, kind_code, scope_id, payload_json)) = live.get(&id).cloned() else { continue };
            let Some(kind) = RecordKind::from_code(kind_code) else { continue };
            if let Some(item) = self.document_for(&mut files, &mut paths, id, namespace_id, scope_id, kind, &payload_json, conn) {
                self.add(&mut writer, &item)?;
            }
        }
        self.finish(&mut writer, conn, storage::current_revision(conn)?)
    }

    /// 把写入流程就地交过来的文档写进索引：按记录 ID 覆盖，尚未提交所以对搜索不可见。
    /// commit 由 `update_index`（或关闭时的收尾）一次做完。
    pub fn stage(&self, docs: &[IndexDocument]) -> Result<()> {
        if docs.is_empty() { return Ok(()); }
        let mut writer = self.writer.lock();
        for item in docs { self.add(&mut writer, item)?; }
        self.dirty.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn add(&self, writer: &mut IndexWriter, item: &IndexDocument) -> Result<()> {
        writer.delete_term(Term::from_field_u64(self.fields.key, item.id as u64));
        let mut document = doc!(self.fields.body => item.text.clone());
        document.add_u64(self.fields.key, item.id as u64);
        document.add_u64(self.fields.namespace, item.namespace_id as u64);
        document.add_u64(self.fields.scope, item.scope_id as u64);
        document.add_u64(self.fields.kind, item.kind.code() as u64);
        for tag_id in &item.tag_ids { document.add_u64(self.fields.tags, *tag_id as u64); }
        // 标签集拼在可搜正文前面，进的是同一条正文列：它跟着这一条的文档长度一起被归一化，
        // 命中标签的分会被这一条的正文稀释。`body` 仍是纯正文——取回给人看的、送进模型的都不带标签。
        let searchable = if item.tags_prefix.is_empty() { item.text.clone() }
            else { format!("{}\n{}", item.tags_prefix, item.text) };
        let cleaned = text::clean_markdown(&searchable);
        if !cleaned.is_empty() { document.add_text(self.fields.text, text::tokenize(&cleaned).join(" ")); }
        if !item.name.is_empty() { document.add_text(self.fields.name, text::tokenize(&item.name).join(" ")); }
        if !item.path.is_empty() { document.add_text(self.fields.path, text::tokenize(&item.path).join(" ")); }
        if item.note_id != 0 { document.add_u64(self.fields.note, item.note_id as u64); }
        writer.add_document(document)?;
        Ok(())
    }

    /// 提交落盘。`revision` 只作进度标记写进 payload，不参与任何对账。
    fn finish(&self, writer: &mut IndexWriter, conn: &Connection, revision: i64) -> Result<()> {
        let mut prepared = writer.prepare_commit()?;
        prepared.set_payload(&format!("{FORMAT}:{revision}"));
        prepared.commit()?;
        self.reader.reload()?;
        // 只增不减：并发下后到的旧提交不允许把进度回退。
        conn.execute("UPDATE meta SET value=?1 WHERE key='indexed_revision' AND value<?1", [revision])?;
        Ok(())
    }

    /// 索引是否可能与主库不一致（有攒下未提交的增删，或删过记录尚未对账）。
    pub fn is_dirty(&self) -> bool { self.dirty.load(Ordering::SeqCst) }

    /// 标脏：记录被删时调用。删除不动索引，只把索引标成「需要重新对账」。
    pub fn mark_dirty(&self) { self.dirty.store(true, Ordering::SeqCst); }

    /// 把写路径攒下的增删提交落盘。无变更时连提交都不做。
    pub fn commit_staged(&self, conn: &Connection) -> Result<()> {
        let mut writer = self.writer.lock();
        if !self.dirty.load(Ordering::SeqCst) { return Ok(()); }
        self.finish(&mut writer, conn, storage::current_revision(conn)?)?;
        self.dirty.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// 同步：先把写路径攒下的增删提交（让读者看见），再对账收敛。
    /// 没有待办账本，也没有第二条路——索引和主库一致与否，只由 `reconcile` 的差集判定。
    pub fn sync(&self, conn: &Connection) -> Result<()> {
        self.commit_staged(conn)?;
        self.reconcile(conn)?;
        self.dirty.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// 从一条记录的原始行折成一个索引文档。切片正文只存在索引里，索引侧补它时回源文件重切；
    /// 其余记录的正文由 payload 直接给出。`files`/`paths` 是本次对账内的笔记缓存，避免同一篇
    /// 笔记的多个切片重复读文件、重复解析路径。
    fn document_for(&self, files: &mut HashMap<i64, Vec<String>>, paths: &mut HashMap<i64, (Vec<String>, String)>,
        id: i64, namespace_id: i64, scope_id: i64, kind: RecordKind, payload_json: &str, conn: &Connection) -> Option<IndexDocument> {
        let payload = serde_json::from_str::<serde_json::Value>(payload_json).ok()?;
        let ordinal = payload.get("ordinal").and_then(|v| v.as_u64()).unwrap_or(0);
        let text = match kind {
            RecordKind::Chunk => {
                let note_id = payload.get("note_id").and_then(|v| v.as_i64()).unwrap_or(0);
                if !files.contains_key(&note_id) { files.insert(note_id, self.note_file_chunks(conn, note_id)); }
                files.get(&note_id).and_then(|chunks| chunks.get(ordinal as usize)).cloned().unwrap_or_default()
            }
            _ => storage::record_text(kind, &payload),
        };
        let pairs = storage::record_tag_pairs(conn, id).unwrap_or_default();
        let tags: Vec<String> = pairs.iter().map(|(_, tag)| tag.clone()).collect();
        // 名字与目录只挂在这一篇的第一片上——否则每一片都会命中同一查询，把结果刷屏。
        let (name, path, exclude) = match kind {
            RecordKind::Chunk if ordinal == 0 => {
                let note_id = payload.get("note_id").and_then(|v| v.as_i64()).unwrap_or(0);
                let (dirs, stem) = paths.entry(note_id).or_insert_with(|| storage::note_path_parts(conn, note_id)).clone();
                let mut exclude = dirs.clone();
                if !stem.is_empty() { exclude.push(stem.clone()); }
                (stem, dirs.join(" "), exclude)
            }
            RecordKind::Chunk => (String::new(), String::new(), Vec::new()),
            _ => (storage::record_name(kind, &payload), String::new(), Vec::new()),
        };
        Some(IndexDocument { id, namespace_id, scope_id, kind, text, name, path,
            note_id: if kind == RecordKind::Chunk { payload.get("note_id").and_then(|v| v.as_i64()).unwrap_or(0) } else { 0 },
            tags_prefix: storage::tags_prefix(kind, &tags, &exclude, &payload),
            tag_ids: pairs.into_iter().map(|(tag_id, _)| tag_id).collect() })
    }

    /// 读一篇笔记的源文件并重跑切分，返回每片正文（按 ordinal 顺序）。
    fn note_file_chunks(&self, conn: &Connection, note_id: i64) -> Vec<String> {
        let Ok((path, namespace_id)) = conn.query_row("SELECT n.path,n.namespace_id FROM notes n WHERE n.record_id=?1",
            [note_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))) else { return Vec::new() };
        let Ok(content) = std::fs::read_to_string(storage::absolute_note_path(conn, namespace_id, &path)) else { return Vec::new() };
        let chunk_chars = conn.query_row("SELECT payload_json FROM records WHERE id=?1", [note_id], |r| r.get::<_, String>(0))
            .ok().and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|value| value.get("chunk_chars").and_then(|v| v.as_u64()))
            .unwrap_or(220) as usize;
        crate::notes::chunk_text(&content, chunk_chars).map(|chunks| chunks.into_iter().map(|chunk| chunk.content).collect()).unwrap_or_default()
    }

    fn exact_u64(field: Field, value: i64) -> Box<dyn Query> {
        Box::new(TermQuery::new(Term::from_field_u64(field, value as u64), IndexRecordOption::Basic))
    }

    fn term(field: Field, value: &str, occurrence: Occur) -> (Occur, Box<dyn Query>) {
        (occurrence, Box::new(TermQuery::new(Term::from_field_text(field, value), IndexRecordOption::WithFreqs)) as Box<dyn Query>)
    }

    fn filtered_query(&self, tokens: &[String], strict: bool, filter: &IndexFilter, field: MatchField) -> Box<dyn Query> {
        let occurrence = if strict { Occur::Must } else { Occur::Should };
        // 命中层：每个词元在指定列里命中即算这个词元命中。
        // `All` 看「正文列 + 名字列」——规范名已从正文列移出，所以名字命中也必须能独立把这条
        // 记录带进结果，不再只是加分项。目录列只在 `Path` 兜底查询里被查。
        let hit = BooleanQuery::new(tokens.iter().map(|token| {
            let column: Box<dyn Query> = match field {
                MatchField::All => Box::new(BooleanQuery::new(vec![
                    Self::term(self.fields.text, token, Occur::Should),
                    Self::term(self.fields.name, token, Occur::Should),
                ])),
                MatchField::Text => Box::new(TermQuery::new(Term::from_field_text(self.fields.text, token), IndexRecordOption::WithFreqs)),
                MatchField::Name => Box::new(TermQuery::new(Term::from_field_text(self.fields.name, token), IndexRecordOption::WithFreqs)),
                MatchField::Path => Box::new(TermQuery::new(Term::from_field_text(self.fields.path, token), IndexRecordOption::WithFreqs)),
            };
            (occurrence, column)
        }).collect());
        // 名字层只负责抬分：名字整段等于查询词的记录，不该被它那几百字的别名与属性摊薄；
        // 其它记录的 name 列为空，天然不参与。单列查询已经只认那一列，不再叠加权。
        let relevance: Box<dyn Query> = if field == MatchField::All {
            let name_relevance = BooleanQuery::new(tokens.iter().map(|token| Self::term(self.fields.name, token, Occur::Should)).collect());
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, Box::new(hit) as Box<dyn Query>),
                (Occur::Should, Box::new(BoostQuery::new(Box::new(name_relevance), NAME_FIELD_BOOST)) as Box<dyn Query>),
            ]))
        } else {
            Box::new(hit)
        };
        // 相关性这层只放 Should，所以至少要命中一个词元才算相关，过滤维度另起一层放 Must。
        // 两层不能合并：同一层里一旦存在 Must，全部 Should 都会降级为「有则加分、无也无妨」，
        // 正文条件就形同虚设，检索退化成「只按过滤条件取记录」——
        // 查一个正文里根本不存在的词，也会返回该过滤域下的任意记录。
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = vec![(Occur::Must, Box::new(relevance))];
        let scopes = BooleanQuery::new(filter.scopes.iter().map(|scope| (Occur::Should, Self::exact_u64(self.fields.scope, *scope))).collect());
        clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Self::exact_u64(self.fields.namespace, filter.namespace), 0.0))));
        clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Box::new(scopes), 0.0))));
        if !filter.kinds.is_empty() {
            let kinds = BooleanQuery::new(filter.kinds.iter().map(|kind| (Occur::Should, Self::exact_u64(self.fields.kind, *kind))).collect());
            clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Box::new(kinds), 0.0))));
        }
        for tag_id in &filter.tags {
            clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Self::exact_u64(self.fields.tags, *tag_id), 0.0))));
        }
        // 笔记限定与 kinds 同构：给的是一批 id，命中其中任一即可。
        // 必须和 kinds 一样留在过滤层：挪进相关性层会让它降级成加分项，限定就形同虚设。
        // 非 chunk 文档的 `note` 列为空，给出了 note_ids 时它们自然不参与命中。
        if !filter.note_ids.is_empty() {
            let notes = BooleanQuery::new(filter.note_ids.iter().map(|id| (Occur::Should, Self::exact_u64(self.fields.note, *id))).collect());
            clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Box::new(notes), 0.0))));
        }
        Box::new(BooleanQuery::new(clauses))
    }

    /// 在指定列上检索。`field` 决定命中限定在哪一列：`All` 是正文+名字（默认），
    /// `Text` / `Name` / `Path` 各自单独成路，供预设把「书名」「内容」「目录兜底」分开取。
    pub fn search_in(&self, query: &str, filter: &IndexFilter, limit: usize, field: MatchField) -> Result<Vec<(RecordKey, f64)>> {
        #[cfg(test)]
        if self.fail_search.load(Ordering::SeqCst) { return Err(Error::Index("injected index failure".into())); }
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        let searcher = self.reader.searcher();
        let mut strict_rank = HashMap::new();
        // `key` 是整数快字段：命中只要 id，就从列上取，不解码文档。文档里有 stored 的正文，
        // 解码一条等于把整篇切片正文读出来解一遍。
        let mut key_columns: HashMap<u32, tantivy::columnar::Column<u64>> = HashMap::new();
        for strict in [true, false] {
            if !strict && result.len() >= limit { break; }
            let tokens = text::query_terms(query, strict);
            if tokens.is_empty() { continue; }
            let query = self.filtered_query(&tokens, strict, filter, field);
            let collector = TopDocs::with_limit(limit).order_by(((SortBySimilarityScore, Order::Desc), (SortByStaticFastValue::<u64>::for_field("key"), Order::Asc)));
            let hits = searcher.search(&*query, &collector)?;
            for ((score, _), address) in hits {
                if !key_columns.contains_key(&address.segment_ord) {
                    let column = searcher.segment_reader(address.segment_ord).fast_fields().u64("key")?;
                    key_columns.insert(address.segment_ord, column);
                }
                let Some(value) = key_columns[&address.segment_ord].first(address.doc_id) else { continue };
                let key = RecordKey { id: value as i64 };
                if !seen.insert(key) { continue; }
                strict_rank.insert(key, strict);
                result.push((key, score as f64));
            }
        }
        result.sort_by(|a, b| strict_rank[&b.0].cmp(&strict_rank[&a.0]).then_with(|| b.1.total_cmp(&a.1)).then_with(|| a.0.cmp(&b.0)));
        result.truncate(limit);
        Ok(result)
    }

    /// 数若干篇笔记各自命中这个查询的切片数。
    ///
    /// 与 `search_in` 同源：同一套词元、同一套过滤条件。区别只在「属于某篇笔记」这条约束
    /// 怎么加：原先每篇各发一次查询，现在一次遍历匹配集、按 `note` 快字段分桶。
    /// 省掉的是每次 `search()` 的固定开销（每段建 weight、逐段枚举），匹配集本身只走一遍。
    /// 只跑宽松词元集那一遍：严格模式多带二元组、词元集是宽松的超集，但要求全部词元命中，
    /// 命中集反而是宽松命中集的子集，所以宽松那一遍数出来的就是并集。
    /// 口径提醒：宽松模式对中文只留单字，所以中文短查询的计数是「含任一单字」的切片数，
    /// 会比「含完整词」的切片数大（拉丁词是整词，两者一致）。
    pub fn count_in_many(&self, query: &str, filter: &IndexFilter, field: MatchField, note_ids: &[i64]) -> Result<HashMap<i64, usize>> {
        let mut out: HashMap<i64, usize> = note_ids.iter().map(|id| (*id, 0)).collect();
        if out.is_empty() { return Ok(out); }
        let tokens = text::query_terms(query, false);
        if tokens.is_empty() { return Ok(out); }
        let base = self.filtered_query(&tokens, false, filter, field);
        // 先把匹配集收在目标笔记上，再在这一次遍历里分桶。
        // 不收的话要遍历整个匹配集（中文单字 OR 下常有几十万片），
        // 而目标笔记的切片总数通常比它小一到两个数量级。
        let notes = BooleanQuery::new(note_ids.iter().map(|id| (Occur::Should, Self::exact_u64(self.fields.note, *id))).collect());
        let scoped = BooleanQuery::new(vec![(Occur::Must, base), (Occur::Must, Box::new(notes))]);
        let collector = NoteCountCollector { targets: Arc::new(note_ids.iter().copied().collect()) };
        for (note, count) in self.reader.searcher().search(&scoped, &collector)? {
            if let Some(slot) = out.get_mut(&note) { *slot = count; }
        }
        Ok(out)
    }

    /// 按 id 取回索引里存储的正文：重排候选取正文、以及「切片正文从索引读」都走这里，不再回源文件。
    pub fn bodies(&self, ids: &[i64]) -> Result<BTreeMap<i64, String>> {
        let mut out = BTreeMap::new();
        if ids.is_empty() { return Ok(out); }
        let searcher = self.reader.searcher();
        // 一次查询把这一批 id 全收进来，再逐条解码 stored 文档。
        // 逐条查询的成本几乎全是每次 `search()` 的固定开销（建 weight、逐段枚举），
        // 与「这一条要取多少正文」无关——一批几百个 id 时那就是几百倍。
        let query = BooleanQuery::new(ids.iter().map(|id| (Occur::Should, Self::exact_u64(self.fields.key, *id))).collect());
        for address in searcher.search(&query, &DocSetCollector)? {
            let document: tantivy::TantivyDocument = searcher.doc(address)?;
            let key = document.get_first(self.fields.key).and_then(|value| value.as_u64());
            let body = document.get_first(self.fields.body).and_then(|value| value.as_str());
            if let (Some(key), Some(body)) = (key, body) { out.insert(key as i64, body.to_string()); }
        }
        Ok(out)
    }
}

/// 一次遍历匹配集、按笔记分桶计数。
///
/// 目标笔记最多只有 limit 篇，所以只对目标集合累加，其余笔记在遍历里直接跳过。
/// `note` 是快字段，读一篇切片归属哪篇笔记不需要解码文档。
struct NoteCountCollector {
    targets: Arc<HashSet<i64>>,
}

struct NoteCountChild {
    column: Column<u64>,
    targets: Arc<HashSet<i64>>,
    counts: HashMap<i64, usize>,
}

impl Collector for NoteCountCollector {
    type Fruit = HashMap<i64, usize>;
    type Child = NoteCountChild;

    fn for_segment(&self, _segment_ord: SegmentOrdinal, segment: &SegmentReader) -> tantivy::Result<Self::Child> {
        Ok(NoteCountChild {
            column: segment.fast_fields().u64("note")?,
            targets: self.targets.clone(),
            counts: HashMap::new(),
        })
    }

    fn requires_scoring(&self) -> bool { false }

    fn merge_fruits(&self, segment_fruits: Vec<HashMap<i64, usize>>) -> tantivy::Result<HashMap<i64, usize>> {
        let mut out = HashMap::new();
        for fruit in segment_fruits {
            for (note, count) in fruit { *out.entry(note).or_insert(0) += count; }
        }
        Ok(out)
    }
}

impl SegmentCollector for NoteCountChild {
    type Fruit = HashMap<i64, usize>;

    fn collect(&mut self, doc: DocId, _score: Score) {
        if let Some(value) = self.column.first(doc) {
            let note = value as i64;
            if self.targets.contains(&note) { *self.counts.entry(note).or_insert(0) += 1; }
        }
    }

    fn harvest(self) -> HashMap<i64, usize> { self.counts }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KnowledgeBase, MemoryInput, SearchRequest};

    fn seed(kb: &KnowledgeBase, n: usize) {
        for i in 0..n { kb.memories().upsert(MemoryInput::new(&format!("对账测试条目{i}"))).unwrap(); }
        kb.update_index().unwrap();
    }

    fn all_ids(kb: &KnowledgeBase) -> Vec<i64> {
        kb.memories().list(&crate::PageRequest { limit: 1000, ..Default::default() }).unwrap().items.iter().map(|m| m.header.id).collect()
    }

    /// 删除只摘对应的 doc，不碰别的文档、也不做任何「推倒重来」。对齐下面两条一起看：
    /// 删两条后同步，索引里恰好少这两个 id，其余仍在。
    #[test]
    fn delete_removes_exactly_the_matching_documents() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        seed(&kb, 5);
        let index = kb.index().unwrap();
        assert_eq!(index.indexed_ids().unwrap().len(), 5);

        // 删两条，此刻不追索引：主库已删、索引还留着，是允许的短暂不一致。
        let ids = all_ids(&kb);
        kb.memories().delete(ids[0], &crate::ReadFilter::default()).unwrap();
        kb.memories().delete(ids[2], &crate::ReadFilter::default()).unwrap();

        kb.update_index().unwrap();
        let remaining = index.indexed_ids().unwrap();
        assert_eq!(remaining.len(), 3, "对账后索引里恰好多出的那两条被摘掉");
        assert!(!remaining.contains(&ids[0]) && !remaining.contains(&ids[2]));
        assert!(remaining.contains(&ids[1]) && remaining.contains(&ids[3]) && remaining.contains(&ids[4]));
    }

    /// 删一条、不调 update_index 就丢弃句柄（进程未收尾），重开库也不允许出现任何重建：
    /// 开库的对账只摘掉那条孤儿 doc，其余文档一个不动。
    #[test]
    fn reopen_after_delete_does_not_rebuild_the_whole_index() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        seed(&kb, 4);
        let ids = all_ids(&kb);
        kb.memories().delete(ids[1], &crate::ReadFilter::default()).unwrap();
        // 不调 update_index，直接丢弃句柄：删除只落在主库，索引里还留着那条 doc。
        drop(kb);

        let kb = KnowledgeBase::open(dir.path()).unwrap();
        let remaining = kb.index().unwrap().indexed_ids().unwrap();
        assert_eq!(remaining.len(), 3, "重开时的对账只摘掉已删那一条");
        assert!(remaining.contains(&ids[0]) && remaining.contains(&ids[2]) && remaining.contains(&ids[3]));
    }

    /// 索引文档缺失（模拟上次提交没跟上）时，对账把缺的补回来，其余不动。
    /// 这里用「建好索引后删掉索引目录再重开」来制造「索引里什么都缺」的极端情形。
    #[test]
    fn reconcile_restores_missing_documents() {
        let dir = tempfile::tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path()).unwrap();
        seed(&kb, 6);
        drop(kb);
        std::fs::remove_dir_all(dir.path().join("text-v2")).unwrap();

        let kb = KnowledgeBase::open(dir.path()).unwrap();
        assert_eq!(kb.index().unwrap().indexed_ids().unwrap().len(), 6, "缺的文档应被对账补回");
        let request = SearchRequest { query: "对账测试条目3".into(), kinds: vec![RecordKind::Memory],
            vector: false, rerank: false, ..Default::default() };
        assert!(!kb.search(&request).unwrap().hits.is_empty(), "补回的文档应可检索");
    }
}
