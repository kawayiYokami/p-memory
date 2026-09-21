//! 对照实验：自建 1+2 元切分（喂 pretokenized）vs 直接把 ngram 交给 Tantivy。
//!
//! A 路 = p-memory 现在的做法：text::tokenize 切好、空格连接、WhitespaceTokenizer 索引；
//!         查询用「严格（Must）→ 宽松（Should）」两轮。
//! B 路 = 引擎自带 NgramTokenizer(1,2) + 小写过滤器直接吃原文，查询走 QueryParser。
//!
//! 跑法：cargo run --release --example search_compare -- [数据目录] [复制倍数]

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use p_memory::text;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, STORED, STRING};
use tantivy::tokenizer::{LowerCaser, NgramTokenizer, TextAnalyzer, WhitespaceTokenizer};
use tantivy::{doc, Index, ReloadPolicy, Searcher, TantivyDocument, Term};

const ITER: usize = 300;

fn collect(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<(String, String)>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .expect("read_dir")
            .map(|e| e.expect("entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();
                    out.push((name, content));
                }
            }
        }
    }
    walk(root, &mut out);
    out
}

/// 复刻 text.rs 里 pub(crate) 的 query_terms：严格轮保留全部词条，宽松轮丢掉纯 CJK 双字。
fn query_terms(value: &str, strict: bool) -> Vec<String> {
    let mut seen = HashSet::new();
    text::tokenize(value)
        .into_iter()
        .filter(|t| {
            (strict || !(t.chars().count() == 2 && t.chars().all(text::is_cjk))) && seen.insert(t.clone())
        })
        .collect()
}

fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let p = e.path();
                    if p.is_dir() {
                        dir_size(&p)
                    } else {
                        e.metadata().map(|m| m.len()).unwrap_or(0)
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

fn vocab(searcher: &Searcher, field: Field) -> usize {
    searcher
        .segment_readers()
        .iter()
        .map(|segment| segment.inverted_index(field).map(|idx| idx.terms().num_terms()).unwrap_or(0))
        .sum()
}

fn key_of(searcher: &Searcher, field: Field, addr: tantivy::DocAddress) -> String {
    let document: TantivyDocument = searcher.doc(addr).unwrap();
    document.get_first(field).and_then(|v| v.as_str()).unwrap_or("").to_string()
}

/// A 路查询：严格轮 + 宽松轮，返回命中数。
fn search_a(searcher: &Searcher, field: Field, query: &str) -> usize {
    let mut seen = HashSet::new();
    let mut count = 0usize;
    for strict in [true, false] {
        let tokens = query_terms(query, strict);
        if tokens.is_empty() {
            continue;
        }
        let occurrence = if strict { Occur::Must } else { Occur::Should };
        let boolean = BooleanQuery::new(
            tokens
                .iter()
                .map(|t| {
                    (
                        occurrence,
                        Box::new(TermQuery::new(
                            Term::from_field_text(field, t),
                            IndexRecordOption::WithFreqs,
                        )) as Box<dyn Query>,
                    )
                })
                .collect(),
        );
        for hit in searcher.search(&boolean, &TopDocs::with_limit(200).order_by_score()).unwrap() {
            if seen.insert(hit.1) {
                count += 1;
            }
        }
    }
    count
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args()
        .nth(1)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| ".".to_string());
    let repeat: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut docs = collect(Path::new(&root));
    if repeat > 1 {
        let base = docs.clone();
        docs.clear();
        for r in 0..repeat {
            for (name, content) in &base {
                docs.push((format!("{name}#{r}"), content.clone()));
            }
        }
    }
    println!("文档数: {}（原始 {} × {}）", docs.len(), docs.len() / repeat, repeat);

    let tmp = tempfile::tempdir()?;

    // ---- A 路：自建切分 + pretokenized ----
    let dir_a = tmp.path().join("a");
    std::fs::create_dir_all(&dir_a)?;
    let mut builder = Schema::builder();
    let key_a = builder.add_text_field("key", (STRING | STORED).set_fast(None));
    let text_a = builder.add_text_field(
        "text",
        TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("pretokenized")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        ),
    );
    let index_a = Index::create_in_dir(&dir_a, builder.build())?;
    index_a.tokenizers().register("pretokenized", WhitespaceTokenizer::default());
    let build_a = Instant::now();
    {
        let mut writer = index_a.writer(50_000_000)?;
        for (name, content) in &docs {
            writer.add_document(doc!(key_a => name.clone(), text_a => text::tokenize(content).join(" ")))?;
        }
        writer.commit()?;
    }
    let build_a = build_a.elapsed();
    let reader_a = index_a.reader_builder().reload_policy(ReloadPolicy::Manual).try_into()?;
    reader_a.reload()?;

    // ---- B 路：引擎自带 ngram(1,2) + 小写过滤器 ----
    let dir_b = tmp.path().join("b");
    std::fs::create_dir_all(&dir_b)?;
    let mut builder_b = Schema::builder();
    let key_b = builder_b.add_text_field("key", (STRING | STORED).set_fast(None));
    let text_b = builder_b.add_text_field(
        "text",
        TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("ngram12")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        ),
    );
    let index_b = Index::create_in_dir(&dir_b, builder_b.build())?;
    let analyzer = TextAnalyzer::builder(NgramTokenizer::new(1, 2, false)?)
        .filter(LowerCaser)
        .build();
    index_b.tokenizers().register("ngram12", analyzer);
    let build_b = Instant::now();
    {
        let mut writer = index_b.writer(50_000_000)?;
        for (name, content) in &docs {
            writer.add_document(doc!(key_b => name.clone(), text_b => content.clone()))?;
        }
        writer.commit()?;
    }
    let build_b = build_b.elapsed();
    let reader_b = index_b.reader_builder().reload_policy(ReloadPolicy::Manual).try_into()?;
    reader_b.reload()?;

    let searcher_a = reader_a.searcher();
    let searcher_b = reader_b.searcher();

    println!("\n== 构建与体积 ==");
    println!(
        "构建耗时: A 自建切分 = {} ms, B 引擎 ngram = {} ms",
        build_a.as_millis(),
        build_b.as_millis()
    );
    println!(
        "索引体积: A 自建切分 = {} KB, B 引擎 ngram = {} KB",
        dir_size(&dir_a) / 1024,
        dir_size(&dir_b) / 1024
    );
    println!(
        "词表规模: A = {} 个词条, B = {} 个词条",
        vocab(&searcher_a, text_a),
        vocab(&searcher_b, text_b)
    );

    println!("\n== 查询延迟（每个查询跑 {ITER} 次，取均值）==");
    println!("{:<12} {:>8} {:>8} {:>12} {:>12}", "查询", "A命中", "B命中", "A µs", "B µs");
    let queries = ["雅", "星见雅", "佩刀 保养", "珍贵的影像", "虚狩"];
    let mut sink = 0usize;
    for query in queries {
        // 热身
        let _ = search_a(&searcher_a, text_a, query);
        let parser = QueryParser::for_index(&index_b, vec![text_b]);

        let start = Instant::now();
        let mut hit_a = 0usize;
        for _ in 0..ITER {
            hit_a = search_a(&searcher_a, text_a, query);
        }
        let time_a = start.elapsed().as_micros() as f64 / ITER as f64;

        let start = Instant::now();
        let mut hit_b = 0usize;
        for _ in 0..ITER {
            let parsed = parser.parse_query(query)?;
            hit_b = searcher_b.search(&*parsed, &TopDocs::with_limit(200).order_by_score())?.len();
        }
        let time_b = start.elapsed().as_micros() as f64 / ITER as f64;

        sink += hit_a + hit_b;
        println!("{query:<12} {hit_a:>8} {hit_b:>8} {time_a:>12.1} {time_b:>12.1}");
    }
    if sink == usize::MAX {
        println!("sink={sink}");
    }
    let _ = key_of(&searcher_a, key_a, tantivy::DocAddress::new(0, 0));
    Ok(())
}
