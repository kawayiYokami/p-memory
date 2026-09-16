use crate::{storage::{self, KnowledgeBase}, text, types::*, Error, Result};
use parking_lot::Mutex;
use rusqlite::{params, params_from_iter, types::Value as SqlValue, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::{BinaryHeap, HashMap, HashSet}, sync::Arc};

fn text_version() -> u32 { 1 }
fn default_encoding() -> String { "sq8".into() }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    pub id: String,
    /// Include provider/model revision in this identity when it changes.
    pub model: String,
    pub dimension: usize,
    #[serde(default = "text_version")] pub text_version: u32,
    /// On-disk vector encoding: "f32" (4 bytes/dim) or "sq8" (1 byte/dim, lossy).
    #[serde(default = "default_encoding")] pub encoding: String,
}
/// 待嵌入文本，只在库内部流转（`sync` 三段式循环的中间产物）。
#[derive(Debug, Clone)]
pub(crate) struct EmbeddingInput { pub key: RecordKey, pub text: String, pub fingerprint: String }
/// 一批算好的向量，只由库自己产出并写回。
#[derive(Debug, Clone)]
pub(crate) struct EmbeddingWrite { pub key: RecordKey, pub fingerprint: String, pub values: Vec<f32> }

#[derive(Clone)]
pub struct EmbeddingStore(pub(crate) KnowledgeBase);

// ── 宿主回调：错误分类、约束声明与注册表 ─────────────────────────────

/// 回调失败的类别。类别必须由宿主显式给出，库不解析错误文案去猜。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedErrorKind {
    /// 单次请求条数超上限：库把 batch 减半后重试，减半值进程内持久。
    TooLarge,
    /// 限流 / 配额：退避后重试有限次。
    RateLimited,
    /// 其它（网络、鉴权、模型不存在）：不重试，直接降级。
    Other,
}

impl EmbedErrorKind {
    pub fn code(self) -> &'static str {
        match self { Self::TooLarge => "too_large", Self::RateLimited => "rate_limited", Self::Other => "other" }
    }
    pub fn from_code(code: &str) -> Option<Self> {
        match code { "too_large" => Some(Self::TooLarge), "rate_limited" => Some(Self::RateLimited), "other" => Some(Self::Other), _ => None }
    }
}

/// 嵌入回调返回的错误，带类别。
#[derive(Debug, Clone)]
pub struct EmbedCallbackError { pub kind: EmbedErrorKind, pub message: String }

impl EmbedCallbackError {
    pub fn new(kind: EmbedErrorKind, message: impl Into<String>) -> Self { Self { kind, message: message.into() } }
    pub fn too_large(message: impl Into<String>) -> Self { Self::new(EmbedErrorKind::TooLarge, message) }
    pub fn rate_limited(message: impl Into<String>) -> Self { Self::new(EmbedErrorKind::RateLimited, message) }
    pub fn other(message: impl Into<String>) -> Self { Self::new(EmbedErrorKind::Other, message) }
}

impl std::fmt::Display for EmbedCallbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}: {}", self.kind.code(), self.message) }
}
impl std::error::Error for EmbedCallbackError {}

/// 嵌入回调：`[文本] -> [向量]`，条数必须与输入一致。Rust 侧直接收闭包。
pub trait Embedder: Send {
    fn embed(&mut self, texts: &[String]) -> std::result::Result<Vec<Vec<f32>>, EmbedCallbackError>;
}

impl<F> Embedder for F
where F: FnMut(&[String]) -> std::result::Result<Vec<Vec<f32>>, EmbedCallbackError> + Send {
    fn embed(&mut self, texts: &[String]) -> std::result::Result<Vec<Vec<f32>>, EmbedCallbackError> { self(texts) }
}

fn default_max_batch() -> usize { 32 }
/// 宿主注册嵌入回调时一并声明的批次与截断约束；库负责不越界。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderOptions {
    #[serde(default = "default_max_batch")] pub max_batch: usize,
    /// 单条文本的 token 预算，`None` 表示不截断。
    #[serde(default)] pub max_tokens_per_text: Option<usize>,
}
impl Default for EmbedderOptions {
    fn default() -> Self { Self { max_batch: default_max_batch(), max_tokens_per_text: None } }
}

pub(crate) struct EmbedderEntry {
    pub options: EmbedderOptions,
    /// 当前生效的批次上限。命中「批次过大」后减半，并在进程内持久。
    pub effective_batch: usize,
    pub embedder: Box<dyn Embedder>,
}

impl EmbedderEntry {
    /// 按声明强制截断后调用回调：库永远不把超出声明的文本送出去。
    pub(crate) fn embed(&mut self, texts: &[String]) -> std::result::Result<Vec<Vec<f32>>, EmbedCallbackError> {
        match self.options.max_tokens_per_text {
            Some(budget) => {
                let budgeted: Vec<String> = texts.iter().map(|value| text::truncate_to_tokens(value, budget)).collect();
                self.embedder.embed(&budgeted)
            }
            None => self.embedder.embed(texts),
        }
    }
}

/// 进程内的回调表：一个向量空间最多绑一个回调。`Arc` 让调用方拿出去后立刻放掉注册表锁。
#[derive(Default)]
pub(crate) struct EmbedderRegistry { entries: Mutex<HashMap<String, Arc<Mutex<EmbedderEntry>>>> }

impl EmbedderRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn is_empty(&self) -> bool { self.entries.lock().is_empty() }
    pub fn space_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.entries.lock().keys().cloned().collect();
        ids.sort();
        ids
    }
    pub fn get(&self, space_id: &str) -> Option<Arc<Mutex<EmbedderEntry>>> { self.entries.lock().get(space_id).cloned() }
    pub fn register(&self, space_id: String, entry: EmbedderEntry) { self.entries.lock().insert(space_id, Arc::new(Mutex::new(entry))); }
    pub fn remove(&self, space_id: &str) -> bool { self.entries.lock().remove(space_id).is_some() }
}

/// 注册校验用的短样本：真实走一遍回调，按空间契约逐项检查产出。
const SAMPLE_TEXTS: [&str; 3] = ["样本一 sample", "样本二 sample", "样本三 sample"];

const RATE_LIMIT_ATTEMPTS: u32 = 3;
const RATE_LIMIT_BACKOFF_MS: u64 = 20;

pub(crate) fn get_space(conn: &Connection, id: &str) -> Result<EmbeddingSpace> {
    conn.query_row("SELECT id,model,dimension,text_version,encoding FROM embedding_spaces WHERE id=?1", [id], |r|
        Ok(EmbeddingSpace { id: r.get(0)?, model: r.get(1)?, dimension: r.get::<_, u32>(2)? as usize, text_version: r.get(3)?, encoding: r.get(4)? })).optional()?
        .ok_or_else(|| Error::NotFound(format!("embedding space {id}")))
}

pub(crate) fn normalize(values: &[f32], dimension: usize) -> Result<Vec<f32>> {
    if values.len() != dimension { return Err(Error::InvalidVector(format!("expected dimension {dimension}, received {}", values.len()))); }
    if values.iter().any(|v| !v.is_finite()) { return Err(Error::InvalidVector("values must be finite".into())); }
    let norm = values.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() { return Err(Error::InvalidVector("zero or invalid vector norm".into())); }
    Ok(values.iter().map(|v| (f64::from(*v) / norm) as f32).collect())
}

/// Dot product of two equal-length f32 slices, unrolled by 8 so the compiler
/// can vectorize it. Inputs are normalized, so the result is within [-1, 1].
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let o = c * 8;
        for k in 0..8 { acc[k] += a[o + k] * b[o + k]; }
    }
    let mut sum = acc.iter().sum::<f32>();
    for i in chunks * 8..a.len() { sum += a[i] * b[i]; }
    sum
}

/// Quantize a normalized query into i8 codes plus its scale, using the same
/// symmetric rule as the stored sq8 encoding. Quantizing the query lets both
/// sides of the dot product stay as i8, so the hot loop never expands to f32.
fn encode_query_sq8(query: &[f32]) -> (Vec<i8>, f32) {
    let max = query.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
    let codes = query.iter().map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8).collect();
    (codes, scale)
}

/// Integer dot product of two i8 rows. The hot loop runs on AVX2 when the
/// running CPU has it, otherwise it falls back to the scalar path, so a build
/// compiled for the generic target still gets the vector path on new machines.
#[inline]
fn dot_codes(left: &[i8], right: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        // Safety: the branch only runs after the CPU is confirmed to have avx2.
        if std::is_x86_feature_detected!("avx2") { return unsafe { dot_codes_avx2(left, right) }; }
    }
    dot_codes_scalar(left, right)
}

fn dot_codes_scalar(left: &[i8], right: &[i8]) -> i32 {
    left.iter().zip(right).map(|(a, b)| i32::from(*a) * i32::from(*b)).sum()
}

/// 16 codes per iteration: widen to i16, multiply-add adjacent pairs into i32.
/// Each accumulator lane stays well inside i32 even at 65536 dimensions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_codes_avx2(left: &[i8], right: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let chunks = left.len() / 16;
    for c in 0..chunks {
        let o = c * 16;
        let a = _mm_loadu_si128(left.as_ptr().add(o) as *const __m128i);
        let b = _mm_loadu_si128(right.as_ptr().add(o) as *const __m128i);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b)));
    }
    let mut total = {
        let sum = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10));
        let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b10_11_00_01));
        _mm_cvtsi128_si32(sum)
    };
    for i in chunks * 16..left.len() { total += i32::from(left[i]) * i32::from(right[i]); }
    total
}

/// Encode a normalized vector for storage. `f32` is lossless; `sq8` is a
/// per-vector symmetric quantization (one f32 scale, then one i8 per dimension).
fn encode_values(normalized: &[f32], encoding: &str) -> Result<Vec<u8>> {
    match encoding {
        "f32" => Ok(normalized.iter().flat_map(|v| v.to_le_bytes()).collect()),
        "sq8" => {
            let max = normalized.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            let mut out = Vec::with_capacity(4 + normalized.len());
            out.extend_from_slice(&scale.to_le_bytes());
            for v in normalized {
                out.push((v / scale).round().clamp(-127.0, 127.0) as i8 as u8);
            }
            Ok(out)
        }
        other => Err(Error::Validation(format!("unknown encoding {other}"))),
    }
}

/// Decode a stored vector; the payload length is checked against the encoding.
fn decode_values(bytes: &[u8], dimension: usize, encoding: &str) -> Result<Vec<f32>> {
    match encoding {
        "f32" => {
            if bytes.len() != dimension * 4 { return Err(Error::InvalidVector("stored vector dimension mismatch".into())); }
            Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
        }
        "sq8" => {
            let (scale, codes) = decode_sq8(bytes, dimension)?;
            Ok(codes.into_iter().map(|c| f32::from(c) * scale).collect())
        }
        other => Err(Error::Validation(format!("unknown encoding {other}"))),
    }
}

/// Split an sq8 payload into its per-vector scale and raw i8 codes, without
/// expanding them back to f32.
fn decode_sq8(bytes: &[u8], dimension: usize) -> Result<(f32, Vec<i8>)> {
    if bytes.len() != dimension + 4 { return Err(Error::InvalidVector("stored vector dimension mismatch".into())); }
    let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    Ok((scale, bytes[4..].iter().map(|b| *b as i8).collect()))
}

// ── 向量化开关与默认值 ───────────────────────────────────────────────

fn vectorize_key(namespace: &str) -> String { format!("vectorize:{}", text::normalized_tag(namespace)) }

/// 记录类型是否默认生成向量。笔记与切片默认关闭：它们体积大、切分多，
/// 走全文进重排候选即可；记忆与图谱记录默认开启。
pub(crate) fn vectorized_by_default(kind: RecordKind) -> bool {
    !matches!(kind, RecordKind::Note | RecordKind::Chunk)
}

/// 该 namespace 是否启用向量化。缺省启用；开关由库落盘，宿主配置一次即生效。
pub(crate) fn namespace_vectorization(conn: &Connection, namespace: &str) -> Result<bool> {
    let value: Option<i64> = conn.query_row("SELECT value FROM meta WHERE key=?1", [vectorize_key(namespace)], |r| r.get(0)).optional()?;
    Ok(value != Some(0))
}

// ── 待嵌入批次（库内部） ─────────────────────────────────────────────

/// 取一批「已启用、已声明、且缺当前指纹向量」的文本。
///
/// 合格判定全部落在 SQL 里：记录类型默认开关、namespace 开关、指纹是否已补齐。
/// 这样游标推进永远不会跳过仍需处理的记录，也不会把不合格记录反复取回来。
pub(crate) fn pending_batch(conn: &Connection, space_id: &str, limit: usize, after: Option<i64>, ids: Option<&[i64]>) -> Result<Vec<EmbeddingInput>> {
    let mut sql = String::from("SELECT r.id,r.kind,r.payload_json,r.fingerprint FROM records r JOIN strings n ON n.id=r.namespace_id WHERE 1=1");
    let mut values: Vec<SqlValue> = Vec::new();
    let disabled: Vec<RecordKind> = [RecordKind::Memory, RecordKind::Entity, RecordKind::Relation, RecordKind::Event, RecordKind::Note, RecordKind::Chunk]
        .into_iter().filter(|kind| !vectorized_by_default(*kind)).collect();
    if !disabled.is_empty() {
        sql.push_str(&format!(" AND r.kind NOT IN ({})", vec!["?"; disabled.len()].join(",")));
        values.extend(disabled.iter().map(|kind| SqlValue::Integer(kind.code())));
    }
    sql.push_str(" AND COALESCE((SELECT value FROM meta WHERE key='vectorize:'||n.text),1)=1");
    sql.push_str(" AND NOT EXISTS(SELECT 1 FROM embeddings e WHERE e.space_id=? AND e.record_id=r.id AND e.fingerprint=r.fingerprint)");
    values.push(SqlValue::Text(space_id.into()));
    if let Some(ids) = ids {
        if ids.is_empty() { return Ok(Vec::new()); }
        sql.push_str(&format!(" AND r.id IN ({})", vec!["?"; ids.len()].join(",")));
        values.extend(ids.iter().map(|id| SqlValue::Integer(*id)));
    }
    if let Some(cursor) = after {
        sql.push_str(" AND r.id>?");
        values.push(SqlValue::Integer(cursor));
    }
    sql.push_str(" ORDER BY r.id LIMIT ?");
    values.push(SqlValue::Integer(limit as i64));
    let mut stmt = conn.prepare(&sql)?;
    let mut items = Vec::new();
    for row in stmt.query_map(params_from_iter(values), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?)))? {
        let (id, kind_code, payload_json, fingerprint) = row?;
        // 向量输入文本不落盘：按 kind 从 payload 现算（记忆额外拼标签，切片由其笔记正文取出）。
        let kind = RecordKind::from_code(kind_code).ok_or_else(|| Error::Validation("invalid stored record kind".into()))?;
        let payload: serde_json::Value = serde_json::from_str(&payload_json)?;
        let tags = record_tags(conn, id)?;
        let body = storage::embedding_text(conn, id, kind, &payload, &tags)?;
        items.push(EmbeddingInput { key: RecordKey { id }, text: body, fingerprint });
    }
    Ok(items)
}

/// 一条记录的标签文本（按字典序，与写入时 `normalize_tags` 的顺序一致）。
fn record_tags(conn: &Connection, id: i64) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT t.text FROM record_tags rt JOIN strings t ON t.id=rt.tag_id WHERE rt.record_id=?1 ORDER BY t.text")?;
    let mut tags = Vec::new();
    for row in stmt.query_map([id], |r| r.get::<_, String>(0))? { tags.push(row?); }
    Ok(tags)
}

/// 单批回调的结果。
enum EmbedOutcome { Vectors(Vec<Vec<f32>>), Shrunk, Failed(String) }

/// 调回调并按类别处理：批次过大减半、限流退避重试、其它直接失败。
/// 减半值写在 `entry.effective_batch` 上，因此在本进程内持久生效。
fn embed_with_retry(entry: &mut EmbedderEntry, texts: &[String]) -> EmbedOutcome {
    let mut attempts = 0u32;
    loop {
        match entry.embed(texts) {
            Ok(values) => return EmbedOutcome::Vectors(values),
            Err(error) if error.kind == EmbedErrorKind::TooLarge => {
                if entry.effective_batch <= 1 { return EmbedOutcome::Failed(format!("batch size 1 was still rejected: {}", error.message)); }
                entry.effective_batch /= 2;
                return EmbedOutcome::Shrunk;
            }
            Err(error) if error.kind == EmbedErrorKind::RateLimited && attempts < RATE_LIMIT_ATTEMPTS => {
                attempts += 1;
                std::thread::sleep(std::time::Duration::from_millis(RATE_LIMIT_BACKOFF_MS * u64::from(attempts)));
            }
            Err(error) => return EmbedOutcome::Failed(error.message),
        }
    }
}

/// 一次内部同步的实际完成量。中断时已写回的批次保留，未跑的批次不写。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncReport { pub scanned: usize, pub written: usize, pub batches: usize, pub interrupted: Option<String> }

impl EmbeddingStore {
    pub fn register_space(&self, space: EmbeddingSpace) -> Result<WriteReceipt<EmbeddingSpace>> {
        storage::validate_identity("space id", &space.id)?;
        storage::validate_identity("model", &space.model)?;
        if !(1..=65_536).contains(&space.dimension) || space.text_version != 1 { return Err(Error::Validation("dimension must be 1..65536 and text_version must be 1".into())); }
        if space.encoding != "f32" && space.encoding != "sq8" { return Err(Error::Validation("encoding must be \"f32\" or \"sq8\"".into())); }
        self.0.mutate(|tx| {
            match get_space(tx, &space.id) {
                Ok(old) if old == space => return Ok(old),
                Ok(_) => return Err(Error::Conflict("embedding space is immutable; register a new ID for a new model or dimension".into())),
                Err(Error::NotFound(_)) => (),
                Err(err) => return Err(err),
            }
            tx.execute("INSERT INTO embedding_spaces(id,model,dimension,text_version,encoding) VALUES (?1,?2,?3,?4,?5)", params![space.id, space.model, space.dimension as i64, space.text_version, space.encoding])?;
            Ok(space)
        })
    }
    pub fn spaces(&self) -> Result<Vec<EmbeddingSpace>> {
        let state = self.0.read()?;
        let mut stmt = state.conn().prepare("SELECT id,model,dimension,text_version,encoding FROM embedding_spaces ORDER BY id")?;
        let rows = stmt.query_map([], |r| Ok(EmbeddingSpace { id: r.get(0)?, model: r.get(1)?, dimension: r.get::<_, u32>(2)? as usize, text_version: r.get(3)?, encoding: r.get(4)? }))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// 把一个嵌入回调绑到一个向量空间。一个向量模型 = 一个向量空间。
    pub fn register_embedder<F: Embedder + 'static>(&self, space_id: &str, embedder: F) -> Result<()> {
        self.register_embedder_with(space_id, embedder, EmbedderOptions::default())
    }

    /// 注册即校验：用样本真跑一遍完整链路，产出不符该空间契约就拒绝绑定。
    /// 这样能挡住「回调绑错空间」「换模型后忘改 dimension」——它们若不在这里拒掉，
    /// 要等第一次写回时才报 `invalid_vector`，那时空间已经建好、离原因很远。
    pub fn register_embedder_with<F: Embedder + 'static>(&self, space_id: &str, embedder: F, options: EmbedderOptions) -> Result<()> {
        storage::validate_identity("space id", space_id)?;
        if !(1..=10_000).contains(&options.max_batch) { return Err(Error::Validation("max_batch must be between 1 and 10000".into())); }
        if options.max_tokens_per_text == Some(0) { return Err(Error::Validation("max_tokens_per_text must be positive".into())); }
        let space = { let state = self.0.read()?; get_space(state.conn(), space_id)? };
        let mut entry = EmbedderEntry { options, effective_batch: options.max_batch, embedder: Box::new(embedder) };
        let samples: Vec<String> = SAMPLE_TEXTS.iter().take(3.min(entry.effective_batch)).map(|sample| (*sample).to_string()).collect();
        // 校验调用发生在注册表之外：此刻还没有任何锁被持有。
        let produced = entry.embed(&samples)
            .map_err(|error| Error::Validation(format!("embedder failed during registration ({}): {}", error.kind.code(), error.message)))?;
        validate_vectors(&produced, samples.len(), &space)?;
        self.0.engine.embedders.register(space_id.to_string(), entry);
        Ok(())
    }

    /// 该空间的定义；从未注册过该空间时返回 `None`。
    pub fn embedder_space(&self, space_id: &str) -> Result<Option<EmbeddingSpace>> {
        let state = self.0.read()?;
        match get_space(state.conn(), space_id) { Ok(space) => Ok(Some(space)), Err(Error::NotFound(_)) => Ok(None), Err(error) => Err(error) }
    }

    pub fn unregister_embedder(&self, space_id: &str) -> Result<bool> { Ok(self.0.engine.embedders.remove(space_id)) }

    /// 该 namespace 是否启用向量化。开关落盘，配置一次即生效。
    pub fn namespace_vectorization(&self, namespace: &str) -> Result<bool> {
        storage::validate_identity("namespace", namespace)?;
        let state = self.0.read()?;
        namespace_vectorization(state.conn(), namespace)
    }

    pub fn set_namespace_vectorization(&self, namespace: &str, enabled: bool) -> Result<WriteReceipt<bool>> {
        storage::validate_identity("namespace", namespace)?;
        self.0.mutate(|tx| {
            tx.execute("INSERT INTO meta(key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![vectorize_key(namespace), i64::from(enabled)])?;
            Ok(enabled)
        })
    }

    /// 内部同步：库拿该空间注册的回调，把缺失向量的记录分批补齐。宿主不参与。
    ///
    /// 模型调用是网络往返，绝不能持有库锁，所以循环严格三段式：
    /// 读快照取一批文本（随即放锁）→ 调回调（不持任何库锁）→ 短事务写回这一批。
    /// 失败即中断：已写回的批次保留，未跑的批次不写。
    pub fn sync(&self, space_id: &str, batch: usize) -> Result<WriteReceipt<SyncReport>> {
        storage::validate_limit(batch)?;
        let entry = self.0.engine.embedders.get(space_id)
            .ok_or_else(|| Error::Validation(format!("no embedder registered for space {space_id}")))?;
        let report = self.drain(space_id, &entry, batch, None, true)?;
        let state = self.0.read()?;
        let conn = state.conn();
        let pending: i64 = conn.query_row("SELECT COUNT(*) FROM index_updates", [], |r| r.get(0))?;
        Ok(WriteReceipt { value: report, revision: storage::current_revision(conn)?, index_ready: pending == 0, index_error: None })
    }

    /// 分段补齐的实际循环。`ids` 为 `Some` 时只处理这批记录（写入路径用）。
    fn drain(&self, space_id: &str, entry: &Arc<Mutex<EmbedderEntry>>, batch: usize, ids: Option<&[i64]>, blocking: bool) -> Result<SyncReport> {
        let mut report = SyncReport::default();
        let acquired = if blocking { Some(entry.lock()) } else { entry.try_lock() };
        // 写入路径绝不为了等模型而阻塞写入：回调正忙就跳过，记录留在待补状态。
        let Some(mut guard) = acquired else { return Ok(report) };
        let mut cursor: Option<i64> = None;
        loop {
            let limit = batch.min(guard.effective_batch).max(1);
            let pending = {
                let state = self.0.read()?;
                pending_batch(state.conn(), space_id, limit, cursor, ids)?
            };
            if pending.is_empty() { break; }
            report.scanned += pending.len();
            let texts: Vec<String> = pending.iter().map(|input| input.text.clone()).collect();
            let values = match embed_with_retry(&mut guard, &texts) {
                // 批次过大：减半后重跑同一段，游标不动，因此不会漏掉记录。
                EmbedOutcome::Shrunk => continue,
                EmbedOutcome::Failed(message) => { report.interrupted = Some(message); break; }
                EmbedOutcome::Vectors(values) => values,
            };
            if values.len() != pending.len() {
                report.interrupted = Some(format!("embedder returned {} vectors for {} inputs", values.len(), pending.len()));
                break;
            }
            let writes: Vec<EmbeddingWrite> = pending.iter().zip(values).map(|(input, vector)|
                EmbeddingWrite { key: input.key, fingerprint: input.fingerprint.clone(), values: vector }).collect();
            match self.put(space_id, &writes) {
                Ok(receipt) => { report.written += receipt.value; report.batches += 1; }
                // 回调期间记录被改写：本批作废，它会以新指纹在下一轮重新出现。
                Err(Error::StaleRevision(_)) => {}
                Err(error) => return Err(error),
            }
            cursor = pending.last().map(|input| input.key.id);
        }
        Ok(report)
    }

    /// Atomic batch: no vectors are written if any input is invalid or stale.
    pub(crate) fn put(&self, space_id: &str, writes: &[EmbeddingWrite]) -> Result<WriteReceipt<usize>> {
        self.0.mutate(|tx| {
            let space = get_space(tx, space_id)?;
            for write in writes {
                let actual: Option<String> = tx.query_row("SELECT fingerprint FROM records WHERE id=?1", [write.key.id], |r| r.get(0)).optional()?;
                let actual = actual.ok_or_else(|| Error::NotFound(write.key.id.to_string()))?;
                if actual != write.fingerprint { return Err(Error::StaleRevision(write.key.id.to_string())); }
                let normalized = normalize(&write.values, space.dimension)?;
                let bytes = encode_values(&normalized, &space.encoding)?;
                tx.execute("INSERT INTO embeddings(space_id,record_id,fingerprint,vector) VALUES (?1,?2,?3,?4)
                    ON CONFLICT(space_id,record_id) DO UPDATE SET fingerprint=excluded.fingerprint,vector=excluded.vector",
                    params![space_id, write.key.id, write.fingerprint, bytes])?;
            }
            Ok(writes.len())
        })
    }

    pub fn delete_space(&self, id: &str) -> Result<WriteReceipt<bool>> {
        self.0.mutate(|tx| Ok(tx.execute("DELETE FROM embedding_spaces WHERE id=?1", [id])? > 0))
    }
}

/// 注册校验：条数一致，且逐条满足维度、有限性、非零范数。
fn validate_vectors(produced: &[Vec<f32>], expected: usize, space: &EmbeddingSpace) -> Result<()> {
    if produced.len() != expected {
        return Err(Error::InvalidVector(format!("embedder returned {} vectors for {expected} inputs", produced.len())));
    }
    for values in produced {
        normalize(values, space.dimension)
            .map_err(|error| Error::InvalidVector(format!("embedder output does not satisfy space {}: {error}", space.id)))?;
    }
    Ok(())
}

/// 写入路径的内部向量化：对每个已注册空间补齐这批记录的向量。
/// 拿不到回调（另一个线程正在同步）就跳过，不阻塞写入；失败返回降级档位。
pub(crate) fn vectorize_records(kb: &KnowledgeBase, ids: &[i64]) -> Vec<Degrade> {
    let mut degraded = Vec::new();
    for space_id in kb.engine.embedders.space_ids() {
        let Some(entry) = kb.engine.embedders.get(&space_id) else { continue };
        match EmbeddingStore(kb.clone()).drain(&space_id, &entry, ids.len().max(1), Some(ids), false) {
            Ok(report) if report.interrupted.is_some() => degraded.push(Degrade::EmbedFailed),
            Ok(_) => {}
            Err(_) => degraded.push(Degrade::EmbedFailed),
        }
    }
    degraded
}

struct VectorRow { key: RecordKey, kind: RecordKind, tags: Vec<String> }
/// 分区内的向量以**存储形态**常驻：sq8 空间保留原始 i8 码与逐条 scale，
/// 不再展开成 f32，因此常驻内存与磁盘体积同量级（1024 维约 1KB/条，而非 4KB/条）。
enum PartitionData {
    F32(Vec<f32>),
    Sq8 { codes: Vec<i8>, scales: Vec<f32> },
}
/// 一个 `(space, namespace, scope)` 分区的向量：按需载入，一次查询只碰自己这块。
pub(crate) struct Partition { dimension: usize, rows: Vec<VectorRow>, data: PartitionData }

// Reverse score ordering makes the heap root the worst retained candidate.
struct Candidate { score: f64, key: RecordKey }
impl PartialEq for Candidate { fn eq(&self, other: &Self) -> bool { self.score == other.score && self.key == other.key } }
impl Eq for Candidate {}
impl PartialOrd for Candidate { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering { other.score.total_cmp(&self.score).then_with(|| self.key.cmp(&other.key)) }
}

impl Partition {
    /// 载入指定 `(space, namespace, scope)` 的向量。该范围内没有向量时返回 `None`，
    /// 空结果也会被缓存，避免每次查询都回库。
    pub fn load(conn: &Connection, space: &EmbeddingSpace, namespace: &str, scope: &str) -> Result<Option<Self>> {
        let namespace = text::normalized_tag(namespace);
        let scope = text::normalized_tag(scope);
        let mut stmt = conn.prepare("SELECT r.id,r.kind,e.vector,
            (SELECT json_group_array(t.text) FROM record_tags rt JOIN strings t ON t.id=rt.tag_id WHERE rt.record_id=r.id)
            FROM embeddings e JOIN records r ON r.id=e.record_id AND r.fingerprint=e.fingerprint
            WHERE e.space_id=?1 AND r.namespace_id=(SELECT id FROM strings WHERE text=?2)
              AND r.scope_id=(SELECT id FROM strings WHERE text=?3) ORDER BY r.id")?;
        let sq8 = space.encoding == "sq8";
        let mut partition = Self {
            dimension: space.dimension,
            rows: vec![],
            data: if sq8 { PartitionData::Sq8 { codes: vec![], scales: vec![] } } else { PartitionData::F32(vec![]) },
        };
        let mut rows = stmt.query(params![space.id, namespace, scope])?;
        while let Some(row) = rows.next()? {
            let key = RecordKey { id: row.get(0)? };
            let kind = RecordKind::from_code(row.get::<_, i64>(1)?).ok_or_else(|| Error::InvalidVector("invalid stored record kind".into()))?;
            let bytes: Vec<u8> = row.get(2)?;
            let tags: Vec<String> = serde_json::from_str(&row.get::<_, String>(3)?)?;
            match &mut partition.data {
                PartitionData::F32(values) => {
                    let decoded = decode_values(&bytes, space.dimension, "f32")?;
                    if decoded.iter().any(|v| !v.is_finite()) { return Err(Error::InvalidVector("stored vector contains nonfinite values".into())); }
                    values.extend(decoded);
                }
                PartitionData::Sq8 { codes, scales } => {
                    let (scale, decoded) = decode_sq8(&bytes, space.dimension)?;
                    if !scale.is_finite() { return Err(Error::InvalidVector("stored vector contains nonfinite values".into())); }
                    scales.push(scale);
                    codes.extend(decoded);
                }
            }
            partition.rows.push(VectorRow { key, kind, tags });
        }
        Ok(if partition.rows.is_empty() { None } else { Some(partition) })
    }

    /// 行是否参与本次打分：白名单、类型、标签三重过滤。
    fn matches(&self, row: &VectorRow, kinds: &[RecordKind], tags: &[String], allowed: Option<&HashSet<i64>>) -> bool {
        if allowed.is_some_and(|set| !set.contains(&row.key.id)) { return false; }
        (kinds.is_empty() || kinds.contains(&row.kind)) && tags.iter().all(|t| row.tags.contains(t))
    }

    /// 维护 top-`limit` 的最小堆：满员后只在分数更好或分数相同但 key 更小时替换。
    fn retain(heap: &mut BinaryHeap<Candidate>, key: RecordKey, scored: f64, limit: usize) {
        let score = scored.clamp(-1.0, 1.0);
        if heap.len() < limit { heap.push(Candidate { score, key }); }
        else if let Some(worst) = heap.peek() {
            if score > worst.score || (score == worst.score && key < worst.key) {
                heap.pop(); heap.push(Candidate { score, key });
            }
        }
    }

    /// 分区内精确打分，返回本分区 top-`limit`（按分数、key 排序）。
    /// `kinds` / `tags` / `allowed` 均由调用方归一化并传入；`allowed` 为 `Some` 时只给这批 record_id 打分。
    pub fn search(&self, query: &[f32], kinds: &[RecordKind], tags: &[String], limit: usize, allowed: Option<&HashSet<i64>>) -> Result<Vec<(RecordKey, f64)>> {
        let query = normalize(query, self.dimension)?;
        let mut heap = BinaryHeap::<Candidate>::new();
        match &self.data {
            PartitionData::F32(values) => {
                for (i, row) in self.rows.iter().enumerate() {
                    if !self.matches(row, kinds, tags, allowed) { continue; }
                    let offset = i * self.dimension;
                    Self::retain(&mut heap, row.key, f64::from(dot(&query, &values[offset..offset + self.dimension])), limit);
                }
            }
            PartitionData::Sq8 { codes, scales } => {
                // 查询也量化一次，整条打分链路保持 i8；查询的 scale 对本分区所有行一致，不影响排序。
                let (query_codes, query_scale) = encode_query_sq8(&query);
                for (i, row) in self.rows.iter().enumerate() {
                    if !self.matches(row, kinds, tags, allowed) { continue; }
                    let offset = i * self.dimension;
                    let raw = query_scale * scales[i] * dot_codes(&query_codes, &codes[offset..offset + self.dimension]) as f32;
                    Self::retain(&mut heap, row.key, f64::from(raw), limit);
                }
            }
        }
        let mut result: Vec<_> = heap.into_iter().map(|c| (c.key, c.score)).collect();
        result.sort_by(|a,b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    /// AVX2 内核与标量内核必须逐位一致，包括长度不是 16 倍数的尾部。
    #[test]
    fn integer_kernel_matches_scalar() {
        for dimension in [1usize, 15, 16, 17, 250, 1024] {
            let left: Vec<i8> = (0..dimension).map(|i| ((i * 37 % 255) as i32 - 127) as i8).collect();
            let right: Vec<i8> = (0..dimension).map(|i| ((i * 91 % 255) as i32 - 127) as i8).collect();
            assert_eq!(dot_codes(&left, &right), dot_codes_scalar(&left, &right), "dimension {dimension}");
        }
    }

    /// sq8 常驻 i8、查询也量化后，分数必须贴合「解码成 f32 再点积」的参考值。
    #[test]
    fn quantized_query_tracks_decoded_f32_kernel() {
        let dimension = 256usize;
        let stored_raw: Vec<f32> = (0..dimension).map(|i| (i as f32 * 0.37).sin() + 0.25).collect();
        let query_raw: Vec<f32> = (0..dimension).map(|i| (i as f32 * 0.11).cos() - 0.1).collect();
        let stored = normalize(&stored_raw, dimension).unwrap();
        let query = normalize(&query_raw, dimension).unwrap();
        let encoded = encode_values(&stored, "sq8").unwrap();
        let (scale, codes) = decode_sq8(&encoded, dimension).unwrap();
        let decoded = decode_values(&encoded, dimension, "sq8").unwrap();
        let reference = dot(&query, &decoded);
        let (query_codes, query_scale) = encode_query_sq8(&query);
        let actual = query_scale * scale * dot_codes(&query_codes, &codes) as f32;
        // 差异只来自查询那一次的 sq8 量化，量级应在千分之一以内。
        assert!((reference - actual).abs() < 1e-3, "reference {reference} vs actual {actual}");
    }

    /// 批次过大时减半重试，减半值落在 entry 上（进程内持久），且下一批按新上限切分。
    #[test]
    fn oversized_batches_shrink_and_persist() {
        let lengths = std::sync::Arc::new(Mutex::new(Vec::<usize>::new()));
        let observed = lengths.clone();
        let mut entry = EmbedderEntry {
            options: EmbedderOptions { max_batch: 8, max_tokens_per_text: None },
            effective_batch: 8,
            embedder: Box::new(move |texts: &[String]| {
                observed.lock().push(texts.len());
                if texts.len() > 4 { return Err(EmbedCallbackError::too_large("too many texts")); }
                Ok(texts.iter().map(|_| vec![1.0f32, 0.0]).collect())
            }),
        };
        let texts: Vec<String> = (0..8).map(|i| format!("文本 {i}")).collect();
        assert!(matches!(embed_with_retry(&mut entry, &texts), EmbedOutcome::Shrunk));
        assert_eq!(entry.effective_batch, 4, "减半值应当写在 entry 上并持久");
        assert!(matches!(embed_with_retry(&mut entry, &texts[..4]), EmbedOutcome::Vectors(_)));
        assert_eq!(*lengths.lock(), vec![8, 4]);
    }

    /// 减半到 1 仍被拒绝即视为模型不可用；其它类别不重试、也不减半。
    #[test]
    fn callback_errors_are_classified() {
        let mut broken = EmbedderEntry {
            options: EmbedderOptions::default(), effective_batch: 1,
            embedder: Box::new(|_: &[String]| Err(EmbedCallbackError::too_large("still too large"))),
        };
        assert!(matches!(embed_with_retry(&mut broken, &["a".to_string()]), EmbedOutcome::Failed(_)));
        assert_eq!(broken.effective_batch, 1);

        let attempts = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = attempts.clone();
        let mut throttled = EmbedderEntry {
            options: EmbedderOptions::default(), effective_batch: 4,
            embedder: Box::new(move |_: &[String]| {
                let attempt = counter.fetch_add(1, AtomicOrdering::SeqCst);
                if attempt < 2 { Err(EmbedCallbackError::rate_limited("slow down")) } else { Ok(vec![vec![1.0f32, 0.0]]) }
            }),
        };
        assert!(matches!(embed_with_retry(&mut throttled, &["a".to_string()]), EmbedOutcome::Vectors(_)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 3, "限流应当退避重试后成功");
        assert_eq!(throttled.effective_batch, 4, "限流不触发减半");
    }
}
