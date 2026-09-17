use crate::{storage, text, types::*, Error, Result};
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use tantivy::{collector::{DocSetCollector, TopDocs, sort_key::{SortBySimilarityScore, SortByString}}, directory::MmapDirectory, doc, Order,
    query::{BooleanQuery, ConstScoreQuery, Occur, Query, TermQuery},
    schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value as TantivyValue, STRING, STORED},
    tokenizer::WhitespaceTokenizer, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

const FORMAT: &str = "p-memory-text-v4";

struct Fields { key: Field, namespace: Field, scope: Field, kind: Field, tags: Field, path: Field, text: Field, body: Field }

/// 一条要写进索引的记录：正文、标签、路径全部由写入流程就地提供。
/// 写入时读一次源文件、切一次，切片正文一路带到这里，索引阶段不再回头读文件。
pub(crate) struct IndexDocument {
    pub id: i64,
    pub namespace: String,
    pub scope: String,
    pub kind: RecordKind,
    /// 正文列：这条记录自己的文本。记忆是 judgment，切片是它那一段，笔记没有（检索面交给切片）。
    pub text: String,
    /// 路径列：笔记自己的路径，切片按从属关系取回同一串；其余记录为空。
    pub path: String,
    pub tags: Vec<String>,
}

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
            namespace: builder.add_text_field("namespace", STRING),
            scope: builder.add_text_field("scope", STRING),
            kind: builder.add_text_field("kind", STRING),
            // 标签列：整串一个词项。既能被搜索（查一个标签词就给这条记录加分），
            // 也能被精确匹配（按标签筛记录）。标签文本仍只存在库里一份，索引不存值。
            tags: builder.add_text_field("tags", STRING),
            // 路径列：笔记自己的路径。文档名不一定出现在正文里，它要有自己的路才搜得到。
            path: builder.add_text_field("path", tokenized()),
            // 正文列：唯一承载「这条记录讲了什么」的列，也是相关性打分的主力。
            text: builder.add_text_field("text", tokenized()),
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
        let mut document = doc!(self.fields.key => encoded,
            self.fields.namespace => text::normalized_tag(&item.namespace),
            self.fields.scope => text::normalized_tag(&item.scope),
            self.fields.kind => item.kind.as_str().to_string(),
            self.fields.body => item.text.clone());
        let cleaned = text::clean_markdown(&item.text);
        if !cleaned.is_empty() { document.add_text(self.fields.text, text::tokenize(&cleaned).join(" ")); }
        if !item.path.is_empty() { document.add_text(self.fields.path, text::tokenize(&item.path).join(" ")); }
        for tag in &item.tags { document.add_text(self.fields.tags, text::normalized_tag(tag)); }
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
        let mut rows = Vec::new();
        let mut failed: Vec<IndexDocument> = Vec::new();
        let Ok(mut stmt) = conn.prepare("SELECT r.id,n.text,r.kind,s.text,r.payload_json FROM records r
            JOIN strings n ON n.id=r.namespace_id JOIN strings s ON s.id=r.scope_id ORDER BY r.id") else { return Vec::new() };
        let Ok(records) = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?,
            r.get::<_, String>(3)?, r.get::<_, String>(4)?))) else { return Vec::new() };
        for row in records {
            let Ok(row) = row else { continue };
            rows.push(row);
        }
        // 同一篇笔记的多片共用一次文件读取与一次切分。
        let mut files: HashMap<i64, Vec<String>> = HashMap::new();
        for (id, namespace, kind_code, scope, payload_json) in rows {
            let Some(kind) = RecordKind::from_code(kind_code) else { continue };
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
            // 路径列只挂在笔记自己那条记录上：切片复制它就是「一篇 N 片各带同一串字」，
            // 这串字在一篇里被计 N 次，排序会变成片多者占位多。切片靠正文列与标签列被搜到。
            let path = if matches!(kind, RecordKind::Note) { self.note_path(conn, id).unwrap_or_default() } else { String::new() };
            failed.push(IndexDocument { id, namespace, scope, kind, text, path, tags: self.record_tags(conn, id) });
        }
        failed
    }

    /// 读一篇笔记的源文件并重跑切分，返回每片正文（按 ordinal 顺序）。
    fn note_file_chunks(&self, conn: &Connection, note_id: i64) -> Vec<String> {
        let Ok(path) = conn.query_row("SELECT path FROM notes WHERE record_id=?1", [note_id], |r| r.get::<_, String>(0)) else { return Vec::new() };
        let Ok(content) = std::fs::read_to_string(&path) else { return Vec::new() };
        let chunk_chars = conn.query_row("SELECT payload_json FROM records WHERE id=?1", [note_id], |r| r.get::<_, String>(0))
            .ok().and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|value| value.get("chunk_chars").and_then(|v| v.as_u64()))
            .unwrap_or(220) as usize;
        crate::notes::chunk_text(&content, chunk_chars).map(|chunks| chunks.into_iter().map(|chunk| chunk.content).collect()).unwrap_or_default()
    }

    fn record_tags(&self, conn: &Connection, id: i64) -> Vec<String> {
        let mut tags = Vec::new();
        let Ok(mut stmt) = conn.prepare("SELECT t.text FROM record_tags rt JOIN strings t ON t.id=rt.tag_id WHERE rt.record_id=?1 ORDER BY t.text") else { return tags };
        let Ok(rows) = stmt.query_map([id], |r| r.get::<_, String>(0)) else { return tags };
        for row in rows { if let Ok(tag) = row { tags.push(tag); } }
        tags
    }

    /// 笔记沿自己的 record_id 取回路径；路径属于笔记，切片不取。
    fn note_path(&self, conn: &Connection, id: i64) -> Result<String> {
        Ok(conn.query_row("SELECT path FROM notes WHERE record_id=?1", [id], |r| r.get::<_, String>(0)).optional()?
            .unwrap_or_default())
    }

    fn exact(field: Field, value: &str) -> Box<dyn Query> {
        Box::new(TermQuery::new(Term::from_field_text(field, value), IndexRecordOption::Basic))
    }

    fn term(field: Field, value: &str, occurrence: Occur) -> (Occur, Box<dyn Query>) {
        (occurrence, Box::new(TermQuery::new(Term::from_field_text(field, value), IndexRecordOption::WithFreqs)) as Box<dyn Query>)
    }

    fn filtered_query(&self, tokens: &[String], tags: &[String], strict: bool, filter: &ReadFilter, kinds: &[RecordKind]) -> Box<dyn Query> {
        let occurrence = if strict { Occur::Must } else { Occur::Should };
        // 三列各算一次分、相加：正文列与路径列吃同一套词元（「史记」这种只出现在文档名里的
        // 查询词落在路径列，正文里的词落在正文列），标签列按整词命中。
        let text_query = BooleanQuery::new(tokens.iter().map(|token| Self::term(self.fields.text, token, occurrence)).collect());
        let path_query = BooleanQuery::new(tokens.iter().map(|token| Self::term(self.fields.path, token, occurrence)).collect());
        // 这一层只放 Should，所以至少要命中一个才算相关，过滤维度另起一层放 Must。
        // 两层不能合并：同一层里一旦存在 Must，全部 Should 都会降级为「有则加分、无也无妨」，
        // 正文条件就形同虚设，检索退化成「只按过滤条件取记录」——
        // 查一个正文里根本不存在的词，也会返回该过滤域下的任意记录。
        let mut relevance: Vec<(Occur, Box<dyn Query>)> = vec![
            (Occur::Should, Box::new(text_query)),
            (Occur::Should, Box::new(path_query)),
        ];
        if !tags.is_empty() {
            let tag_query = BooleanQuery::new(tags.iter().map(|word| (Occur::Should,
                Box::new(TermQuery::new(Term::from_field_text(self.fields.tags, word), IndexRecordOption::Basic)) as Box<dyn Query>)).collect());
            relevance.push((Occur::Should, Box::new(tag_query)));
        }
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = vec![(Occur::Must, Box::new(BooleanQuery::new(relevance)))];
        let scopes = BooleanQuery::new(filter.scopes.iter().map(|scope| (Occur::Should, Self::exact(self.fields.scope, &text::normalized_tag(scope)))).collect());
        clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Self::exact(self.fields.namespace, &text::normalized_tag(&filter.namespace)), 0.0))));
        clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Box::new(scopes), 0.0))));
        if !kinds.is_empty() {
            let kinds = BooleanQuery::new(kinds.iter().map(|kind| (Occur::Should, Self::exact(self.fields.kind, kind.as_str()))).collect());
            clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Box::new(kinds), 0.0))));
        }
        for tag in &filter.tags {
            clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(Self::exact(self.fields.tags, &text::normalized_tag(tag)), 0.0))));
        }
        Box::new(BooleanQuery::new(clauses))
    }

    pub fn search(&self, query: &str, filter: &ReadFilter, kinds: &[RecordKind], limit: usize) -> Result<Vec<(RecordKey, f64)>> {
        #[cfg(test)]
        if self.fail_search.load(Ordering::SeqCst) { return Err(Error::Index("injected index failure".into())); }
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        let searcher = self.reader.searcher();
        let mut strict_rank = HashMap::new();
        let tags = text::keyword_terms(query);
        for strict in [true, false] {
            if !strict && result.len() >= limit { break; }
            let tokens = text::query_terms(query, strict);
            if tokens.is_empty() { continue; }
            let query = self.filtered_query(&tokens, &tags, strict, filter, kinds);
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
            let query = Self::exact(self.fields.key, &id.to_string());
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
