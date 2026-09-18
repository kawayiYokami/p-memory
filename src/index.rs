use crate::{storage, text, types::*, Error, Result};
use parking_lot::Mutex;
use rusqlite::Connection;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use tantivy::{collector::{DocSetCollector, TopDocs, sort_key::{SortBySimilarityScore, SortByString}}, directory::MmapDirectory, doc, Order,
    query::{BoostQuery, BooleanQuery, ConstScoreQuery, Occur, Query, TermQuery},
    schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value as TantivyValue, INDEXED, STRING, STORED},
    tokenizer::WhitespaceTokenizer, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

const FORMAT: &str = "p-memory-text-v7";

struct Fields { key: Field, namespace: Field, scope: Field, kind: Field, tags: Field, text: Field, name: Field, body: Field }

/// 名字字段命中时的固定加权：名字是实体的规范名，比正文里的同名提及更该决定这条记录的相关度。
const NAME_FIELD_BOOST: f32 = 3.0;

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
    /// 拼在可搜正文前面的标签集（空格连接，空串表示不带）。索引里没有单独的标签文本列：
    /// 标签只有承载它的那一条带——切片是第一片，其余记录是它自己；取值规则见 `storage::tags_prefix`。
    pub tags_prefix: String,
    /// 标签 id 列：只用来按标签过滤，不参与打分。
    pub tag_ids: Vec<i64>,
}

/// 已在库侧折算成 id 的检索范围。索引里的 namespace / scope / kind / tags 一律是整数，
/// 标记文本换成 id 这件事在进索引之前做完。
pub(crate) struct IndexFilter { pub namespace: i64, pub scopes: Vec<i64>, pub kinds: Vec<i64>, pub tags: Vec<i64> }

/// 文本索引。`IndexReader` 可并发检索，`IndexWriter` 收进内部互斥锁：
/// 整个结构可以直接共享给多个读线程，提交只在写者之间串行。
pub(crate) struct TextIndex {
    index: Index, reader: IndexReader, writer: Mutex<IndexWriter>, fields: Fields,
    /// 测试专用：注入查询故障，验证「索引查询失败即重建并重试」的恢复路径。
    #[cfg(test)] pub(crate) fail_search: AtomicBool,
    /// 测试专用：统计重建次数。
    #[cfg(test)] pub(crate) rebuilds: AtomicUsize,
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
            key: builder.add_text_field("key", (STRING | STORED).set_fast(None)),
            // 标记列一律存 strings 表的整数 id：索引里不留第二份标记文本。
            namespace: builder.add_u64_field("namespace", INDEXED),
            scope: builder.add_u64_field("scope", INDEXED),
            kind: builder.add_u64_field("kind", INDEXED),
            // 标签 id 列（多值）：按标签过滤只在这一列上做精确匹配，不参与打分。
            tags: builder.add_u64_field("tags", INDEXED),
            // 正文列：唯一承载「这条记录讲了什么」的列，也是相关性打分的主力。
            // 标签集拼在承载它的那一条的正文前面，与正文同列、同长度、共同参与打分。
            text: builder.add_text_field("text", tokenized()),
            // 实体规范名单独一列：与正文分开，检索时按固定倍数加权，抵消长正文对名字的分摊。
            name: builder.add_text_field("name", tokenized()),
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
        let writer = index.writer(20_000_000)?;
        let reader = index.reader_builder().reload_policy(ReloadPolicy::Manual).try_into()?;
        Ok(Self { index, reader, writer: Mutex::new(writer), fields,
            #[cfg(test)] fail_search: AtomicBool::new(false),
            #[cfg(test)] rebuilds: AtomicUsize::new(0) })
    }

    pub fn document_count(&self) -> usize { self.reader.searcher().num_docs() as usize }

    /// 索引自检：格式过期、或上一次写入没走完提交（待办队列还压着东西）时重建。
    /// 重建是唯一允许重新读源文件的路径——正文没有第二份副本，异常恢复只能回源。
    pub fn recover(&self, conn: &Connection) -> Result<()> {
        let expected = format!("{FORMAT}:{}", storage::meta(conn, "indexed_revision")?);
        let pending: i64 = conn.query_row("SELECT COUNT(*) FROM index_updates", [], |r| r.get(0))?;
        if pending > 0 || self.index.load_metas()?.payload.as_deref() != Some(&expected) { self.rebuild(conn) }
        else { Ok(()) }
    }

    /// 把写入流程就地交过来的文档写进索引：按记录 ID 覆盖，尚未提交所以对搜索不可见。
    /// commit 由 `update_index`（或关闭时的收尾）一次做完。
    pub fn stage(&self, docs: &[IndexDocument]) -> Result<()> {
        if docs.is_empty() { return Ok(()); }
        let mut writer = self.writer.lock();
        for item in docs { self.add(&mut writer, item)?; }
        Ok(())
    }

    fn add(&self, writer: &mut IndexWriter, item: &IndexDocument) -> Result<()> {
        let encoded = item.id.to_string();
        writer.delete_term(Term::from_field_text(self.fields.key, &encoded));
        let mut document = doc!(self.fields.key => encoded, self.fields.body => item.text.clone());
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
        writer.add_document(document)?;
        Ok(())
    }

    /// 提交并记账。`revision` 是本次**实际覆盖到**的最大 revision：
    /// 并发写入时可能有更新的 revision 在读取之后才提交，记账必须只认自己真索引过的那一段。
    fn finish(&self, writer: &mut IndexWriter, conn: &Connection, revision: i64) -> Result<()> {
        let mut prepared = writer.prepare_commit()?;
        prepared.set_payload(&format!("{FORMAT}:{revision}"));
        prepared.commit()?;
        self.reader.reload()?;
        // A crash before this acknowledgement simply replays idempotent replacements.
        // 记账只增不减：并发下后到的旧提交不允许把进度回退。
        conn.execute("UPDATE meta SET value=?1 WHERE key='indexed_revision' AND value<?1", [revision])?;
        conn.execute("DELETE FROM index_updates WHERE revision<=?1", [revision])?;
        Ok(())
    }

    /// 提交本进程已经写进 writer 的文档。写入路径不再回库重建文档，所以这里只提交与记账；
    /// 队列为空是稳态（无待办时连写锁都不碰），非空说明有尚未提交的写入。
    pub fn sync(&self, conn: &Connection) -> Result<()> {
        let pending: i64 = conn.query_row("SELECT COUNT(*) FROM index_updates", [], |r| r.get(0))?;
        if pending == 0 { return Ok(()); }
        let covered: i64 = conn.query_row("SELECT COALESCE(MAX(revision),0) FROM index_updates", [], |r| r.get(0))?;
        let mut writer = self.writer.lock();
        self.finish(&mut writer, conn, covered)
    }

    /// 全量重建：清空索引，按库里现有记录重新读源文件、重跑同一套切分。
    /// 只在索引格式过期、索引损坏、或上次写入未收尾时走这里，属于异常恢复而非日常路径。
    pub fn rebuild(&self, conn: &Connection) -> Result<()> {
        #[cfg(test)]
        self.rebuilds.fetch_add(1, Ordering::SeqCst);
        let mut writer = self.writer.lock();
        writer.delete_all_documents()?;
        for item in self.rebuild_documents(conn) { self.add(&mut writer, &item)?; }
        self.finish(&mut writer, conn, storage::current_revision(conn)?)
    }

    /// 从库里现有的记录重建索引文档。切片正文没有第二份副本，这里按同一套切分规则
    /// 重新读文件切一遍；文件读不到时任该切片正文为空，不让一次恢复失败于单个缺文件。
    fn rebuild_documents(&self, conn: &Connection) -> Vec<IndexDocument> {
        let mut documents = Vec::new();
        let Ok(mut stmt) = conn.prepare("SELECT r.id,r.namespace_id,r.kind,r.scope_id,r.payload_json FROM records r ORDER BY r.id") else { return Vec::new() };
        let Ok(records) = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?,
            r.get::<_, i64>(3)?, r.get::<_, String>(4)?))) else { return Vec::new() };
        let mut rows = Vec::new();
        for row in records {
            let Ok(row) = row else { continue };
            rows.push(row);
        }
        // 同一篇笔记的多片共用一次文件读取与一次切分。
        let mut files: HashMap<i64, Vec<String>> = HashMap::new();
        for (id, namespace_id, kind_code, scope_id, payload_json) in rows {
            let Some(kind) = RecordKind::from_code(kind_code) else { continue };
            // 笔记不占索引文档：路径信息以标签形态挂在它的每个切片上，文件列表按库里的标签翻。
            if kind == RecordKind::Note { continue; }
            let Ok(payload) = serde_json::from_str::<serde_json::Value>(&payload_json) else { continue };
            let text = match kind {
                RecordKind::Chunk => {
                    let note_id = payload.get("note_id").and_then(|v| v.as_i64()).unwrap_or(0);
                    let ordinal = payload.get("ordinal").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    if !files.contains_key(&note_id) { files.insert(note_id, self.note_file_chunks(conn, note_id)); }
                    files.get(&note_id).and_then(|chunks| chunks.get(ordinal)).cloned().unwrap_or_default()
                }
                _ => storage::record_text(kind, &payload),
            };
            let pairs = storage::record_tag_pairs(conn, id).unwrap_or_default();
            let tags: Vec<String> = pairs.iter().map(|(_, tag)| tag.clone()).collect();
            documents.push(IndexDocument { id, namespace_id, scope_id, kind, text,
                name: storage::record_name(kind, &payload),
                tags_prefix: storage::tags_prefix(kind, &tags, &payload),
                tag_ids: pairs.into_iter().map(|(tag_id, _)| tag_id).collect() });
        }
        documents
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

    fn exact_text(field: Field, value: &str) -> Box<dyn Query> {
        Box::new(TermQuery::new(Term::from_field_text(field, value), IndexRecordOption::Basic))
    }

    fn exact_u64(field: Field, value: i64) -> Box<dyn Query> {
        Box::new(TermQuery::new(Term::from_field_u64(field, value as u64), IndexRecordOption::Basic))
    }

    fn term(field: Field, value: &str, occurrence: Occur) -> (Occur, Box<dyn Query>) {
        (occurrence, Box::new(TermQuery::new(Term::from_field_text(field, value), IndexRecordOption::WithFreqs)) as Box<dyn Query>)
    }

    fn filtered_query(&self, tokens: &[String], strict: bool, filter: &IndexFilter) -> Box<dyn Query> {
        let occurrence = if strict { Occur::Must } else { Occur::Should };
        // 正文层决定「命中」：strict 轮要求全部词元命中，放宽轮命中一个即可。
        let body_relevance = BooleanQuery::new(tokens.iter().map(|token| Self::term(self.fields.text, token, occurrence)).collect());
        // 名字层只负责抬分：与正文层同处一个 BooleanQuery，正文层是 Must、名字层是 Should，
        // 因此名字命中只加分、不改变「必须命中正文」这个必要条件。实体名单独成列、长度均匀，
        // 加权后不会被它的别名与属性摊薄；其它记录的 name 列为空，天然不参与。
        let name_relevance = BooleanQuery::new(tokens.iter().map(|token| Self::term(self.fields.name, token, Occur::Should)).collect());
        let relevance = BooleanQuery::new(vec![
            (Occur::Must, Box::new(body_relevance) as Box<dyn Query>),
            (Occur::Should, Box::new(BoostQuery::new(Box::new(name_relevance), NAME_FIELD_BOOST)) as Box<dyn Query>),
        ]);
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
        Box::new(BooleanQuery::new(clauses))
    }

    pub fn search(&self, query: &str, filter: &IndexFilter, limit: usize) -> Result<Vec<(RecordKey, f64)>> {
        #[cfg(test)]
        if self.fail_search.load(Ordering::SeqCst) { return Err(Error::Index("injected index failure".into())); }
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        let searcher = self.reader.searcher();
        let mut strict_rank = HashMap::new();
        for strict in [true, false] {
            if !strict && result.len() >= limit { break; }
            let tokens = text::query_terms(query, strict);
            if tokens.is_empty() { continue; }
            let query = self.filtered_query(&tokens, strict, filter);
            let collector = TopDocs::with_limit(limit).order_by(((SortBySimilarityScore, Order::Desc), (SortByString::for_field("key"), Order::Asc)));
            let hits = searcher.search(&*query, &collector)?;
            for ((score, _), address) in hits {
                let document: tantivy::TantivyDocument = searcher.doc(address)?;
                let encoded = document.get_first(self.fields.key).and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Index("missing document key".into()))?;
                if !seen.insert(encoded.to_string()) { continue; }
                let id: i64 = encoded.parse().map_err(|_| Error::Index("invalid document key".into()))?;
                let key = RecordKey { id };
                strict_rank.insert(key, strict);
                result.push((key, score as f64));
            }
        }
        result.sort_by(|a, b| strict_rank[&b.0].cmp(&strict_rank[&a.0]).then_with(|| b.1.total_cmp(&a.1)).then_with(|| a.0.cmp(&b.0)));
        result.truncate(limit);
        Ok(result)
    }

    /// 按 id 取回索引里存储的正文：重排候选取正文、以及「切片正文从索引读」都走这里，不再回源文件。
    pub fn bodies(&self, ids: &[i64]) -> Result<BTreeMap<i64, String>> {
        let mut out = BTreeMap::new();
        if ids.is_empty() { return Ok(out); }
        let searcher = self.reader.searcher();
        for id in ids {
            let query = Self::exact_text(self.fields.key, &id.to_string());
            if let Some(address) = searcher.search(&query, &DocSetCollector)?.into_iter().next() {
                let document: tantivy::TantivyDocument = searcher.doc(address)?;
                if let Some(body) = document.get_first(self.fields.body).and_then(|v| v.as_str()) {
                    out.insert(*id, body.to_string());
                }
            }
        }
        Ok(out)
    }
}
