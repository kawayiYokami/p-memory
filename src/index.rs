use crate::{storage, text, types::*, Error, Result};
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::{collections::{BTreeMap, BTreeSet, HashMap, HashSet}, path::Path};
use tantivy::{collector::{DocSetCollector, TopDocs, sort_key::{SortBySimilarityScore, SortByString}}, directory::MmapDirectory, doc, Order,
    query::{BooleanQuery, ConstScoreQuery, Occur, Query, TermQuery},
    schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value as TantivyValue, STRING, STORED},
    tokenizer::WhitespaceTokenizer, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

const FORMAT: &str = "p-memory-text-v3";

struct Fields { key: Field, namespace: Field, scope: Field, kind: Field, tags: Field, keywords: Field, text: Field, body: Field }
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
        let fields = Fields {
            key: builder.add_text_field("key", (STRING | STORED).set_fast(None)),
            namespace: builder.add_text_field("namespace", STRING),
            scope: builder.add_text_field("scope", STRING),
            kind: builder.add_text_field("kind", STRING),
            tags: builder.add_text_field("tags", STRING),
            keywords: builder.add_text_field("keywords", STRING),
            text: builder.add_text_field("text", TextOptions::default().set_indexing_options(
                TextFieldIndexing::default().set_tokenizer("pretokenized").set_index_option(IndexRecordOption::WithFreqsAndPositions))),
            // 可检索正文的存储字段：SQLite 不再保留派生文本，重排候选与切片正文从这里取回。
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

    pub fn recover(&self, conn: &Connection) -> Result<()> {
        let expected = format!("{FORMAT}:{}", storage::meta(conn, "indexed_revision")?);
        if self.index.load_metas()?.payload.as_deref() != Some(&expected) { self.rebuild(conn) }
        else { self.sync(conn) }
    }

    fn add_current(&self, writer: &mut IndexWriter, conn: &Connection, id: i64) -> Result<()> {
        let encoded = id.to_string();
        writer.delete_term(Term::from_field_text(self.fields.key, &encoded));
        let row = conn.query_row("SELECT n.text,r.kind,s.text,r.payload_json FROM records r
            JOIN strings n ON n.id=r.namespace_id JOIN strings s ON s.id=r.scope_id WHERE r.id=?1",
            [id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?))).optional()?;
        if let Some((namespace, kind_code, scope, payload_json)) = row {
            let kind = RecordKind::from_code(kind_code).ok_or_else(|| Error::Index("invalid stored record kind".into()))?;
            let payload: serde_json::Value = serde_json::from_str(&payload_json).map_err(|e| Error::Index(e.to_string()))?;
            // 可检索正文不落 SQLite，写入索引时按 kind 从 payload 现算。
            let body = storage::search_text(conn, id, kind, &payload)?;
            let mut document = doc!(self.fields.key => encoded, self.fields.namespace => namespace,
                self.fields.scope => scope, self.fields.kind => kind.as_str(),
                self.fields.body => body,
                self.fields.text => text::tokenize(&text::clean_markdown(&body)).join(" "));
            let mut tags = conn.prepare("SELECT t.text FROM record_tags rt JOIN strings t ON t.id=rt.tag_id WHERE rt.record_id=?1")?;
            for tag in tags.query_map([id], |r| r.get::<_, String>(0))? {
                let tag = tag?;
                document.add_text(self.fields.tags, tag.clone());
                document.add_text(self.fields.keywords, tag);
            }
            if let Some(source) = self.note_source(conn, id, kind)? {
                for dir in text::ancestor_dirs(&source) { document.add_text(self.fields.keywords, dir); }
            }
            writer.add_document(document)?;
        }
        Ok(())
    }

    /// 笔记/切片沿 note_id 取回宿主路径；其余记录没有父目录。
    fn note_source(&self, conn: &Connection, id: i64, kind: RecordKind) -> Result<Option<String>> {
        let source = match kind {
            RecordKind::Note => conn.query_row(
                "SELECT s.text FROM notes n JOIN strings s ON s.id=n.source_id WHERE n.record_id=?1", [id], |r| r.get::<_, String>(0)).optional()?,
            RecordKind::Chunk => conn.query_row(
                "SELECT s.text FROM chunks c JOIN notes n ON n.record_id=c.note_id JOIN strings s ON s.id=n.source_id WHERE c.record_id=?1", [id], |r| r.get::<_, String>(0)).optional()?,
            _ => None,
        };
        Ok(source)
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

    /// 追平待索引队列。整个「读队列 → 写文档 → 提交 → 记账」过程持内部写锁，
    /// 保证同时只有一个同步在跑，谁也不会把别人还没落盘的记录标记成已索引。
    pub fn sync(&self, conn: &Connection) -> Result<()> {
        // 先无锁探一次待办：队列为空就直接返回。读路径每次检索都会走到这里，
        // 而队列在稳态下总是空的（写路径提交后已就地清空），这一探让读完全不碰写锁。
        let pending: i64 = conn.query_row("SELECT COUNT(*) FROM index_updates", [], |r| r.get(0))?;
        if pending == 0 { return Ok(()); }
        let mut writer = self.writer.lock();
        // 拿锁前的快照可能已过期，队列必须重新读一次。
        let mut stmt = conn.prepare("SELECT record_id,revision FROM index_updates ORDER BY revision")?;
        let mut ids = BTreeSet::new();
        let mut covered = 0i64;
        for row in stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))? {
            let (id, revision) = row?;
            ids.insert(id);
            covered = covered.max(revision);
        }
        if ids.is_empty() { return Ok(()); }
        for id in ids { self.add_current(&mut writer, conn, id)?; }
        self.finish(&mut writer, conn, covered)
    }

    pub fn rebuild(&self, conn: &Connection) -> Result<()> {
        #[cfg(test)]
        self.rebuilds.fetch_add(1, Ordering::SeqCst);
        let mut writer = self.writer.lock();
        writer.delete_all_documents()?;
        let mut stmt = conn.prepare("SELECT id FROM records ORDER BY id")?;
        for row in stmt.query_map([], |r| r.get::<_, i64>(0))? { self.add_current(&mut writer, conn, row?)?; }
        self.finish(&mut writer, conn, storage::current_revision(conn)?)
    }

    fn exact(field: Field, value: &str) -> Box<dyn Query> {
        Box::new(TermQuery::new(Term::from_field_text(field, value), IndexRecordOption::Basic))
    }

    fn filtered_query(&self, tokens: &[String], keywords: &[String], strict: bool, filter: &ReadFilter, kinds: &[RecordKind]) -> Box<dyn Query> {
        let occurrence = if strict { Occur::Must } else { Occur::Should };
        let text_query = BooleanQuery::new(tokens.iter().map(|token| (occurrence,
            Box::new(TermQuery::new(Term::from_field_text(self.fields.text, token), IndexRecordOption::WithFreqs)) as Box<dyn Query>)).collect());
        // 正文 1+2 分词与「整词关键字」并行命中。这一层只放 Should，所以至少要命中一个才算相关，
        // 过滤维度另起一层放 Must。两层不能合并：同一层里一旦存在 Must，全部 Should 都会降级为
        // 「有则加分、无也无妨」，正文条件就形同虚设，检索退化成「只按过滤条件取记录」——
        // 查一个正文里根本不存在的词，也会返回该过滤域下的任意记录。
        let mut relevance: Vec<(Occur, Box<dyn Query>)> = vec![(Occur::Should, Box::new(text_query))];
        if !keywords.is_empty() {
            let keyword_query = BooleanQuery::new(keywords.iter().map(|word| (Occur::Should,
                Box::new(TermQuery::new(Term::from_field_text(self.fields.keywords, word), IndexRecordOption::Basic)) as Box<dyn Query>)).collect());
            relevance.push((Occur::Should, Box::new(keyword_query)));
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
        let keywords = text::keyword_terms(query);
        for strict in [true, false] {
            if !strict && result.len() >= limit { break; }
            let tokens = text::query_terms(query, strict);
            if tokens.is_empty() { continue; }
            let query = self.filtered_query(&tokens, &keywords, strict, filter, kinds);
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

    /// 按 id 取回索引里存储的可检索正文：重排候选与「从命中直接取正文」都走这里，不再回 SQLite。
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
