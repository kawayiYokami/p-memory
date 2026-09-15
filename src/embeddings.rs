use crate::{storage::{self, KnowledgeBase}, types::*, Error, Result};
use rusqlite::{params, params_from_iter, types::Value as SqlValue, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::{BinaryHeap, HashSet}};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingInput { pub key: RecordKey, pub text: String, pub fingerprint: String, pub revision: i64 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingWrite { pub key: RecordKey, pub fingerprint: String, pub values: Vec<f32> }
#[derive(Clone)]
pub struct EmbeddingStore(pub(crate) KnowledgeBase);

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
    /// Fetch text for host-side embedding. Cursor pagination covers only missing
    /// embeddings; start a fresh pass to catch edits behind an earlier cursor.
    pub fn pending(&self, space_id: &str, page: &PageRequest, kinds: &[RecordKind]) -> Result<Page<EmbeddingInput>> {
        storage::validate_limit(page.limit)?;
        let state = self.0.read()?;
        let conn = state.conn();
        get_space(conn, space_id)?;
        let (mut condition, mut values) = storage::filter_sql(&page.filter, kinds, false)?;
        condition.push_str(" AND NOT EXISTS(SELECT 1 FROM embeddings e WHERE e.space_id=? AND e.record_id=r.id AND e.fingerprint=r.fingerprint)");
        values.push(SqlValue::Text(space_id.into()));
        if let Some(after) = &page.after {
            let id: i64 = after.parse().map_err(|_| Error::Validation("invalid page cursor".into()))?;
            condition.push_str(" AND r.id>?");
            values.push(SqlValue::Integer(id));
        }
        values.push(SqlValue::Integer((page.limit + 1) as i64));
        let mut stmt = conn.prepare(&format!("SELECT r.id,r.kind,r.embedding_text,r.fingerprint,r.revision FROM records r WHERE {condition} ORDER BY r.id LIMIT ?"))?;
        let rows = stmt.query_map(params_from_iter(values), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?, r.get::<_, i64>(4)?)))?;
        let mut items = Vec::new();
        for row in rows {
            let (id, kind_code, text, fingerprint, revision) = row?;
            let _kind = RecordKind::from_code(kind_code).ok_or_else(|| Error::Validation("invalid stored record kind".into()))?;
            items.push(EmbeddingInput { key: RecordKey { id }, text, fingerprint, revision });
        }
        let has_more = items.len() > page.limit;
        items.truncate(page.limit);
        let next_cursor = if has_more { items.last().map(|v| v.key.index_key()) } else { None };
        Ok(Page { items, next_cursor })
    }
    /// Atomic batch: no vectors are written if any input is invalid or stale.
    pub fn put(&self, space_id: &str, writes: &[EmbeddingWrite]) -> Result<WriteReceipt<usize>> {
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
        let namespace = crate::text::normalized_tag(namespace);
        let scope = crate::text::normalized_tag(scope);
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

    /// 维护 top-`limit` 的最小堆，`min_score` 与 clamp 的先后顺序与旧实现一致。
    fn retain(heap: &mut BinaryHeap<Candidate>, key: RecordKey, scored: f64, limit: usize, min_score: Option<f64>) {
        let score = scored.clamp(-1.0, 1.0);
        if min_score.is_some_and(|min| score < min) { return; }
        if heap.len() < limit { heap.push(Candidate { score, key }); }
        else if let Some(worst) = heap.peek() {
            if score > worst.score || (score == worst.score && key < worst.key) {
                heap.pop(); heap.push(Candidate { score, key });
            }
        }
    }

    /// 分区内精确打分，返回本分区 top-`limit`（按分数、key 排序）。
    /// `kinds` / `tags` / `allowed` 均由调用方归一化并传入；`allowed` 为 `Some` 时只给这批 record_id 打分。
    pub fn search(&self, query: &[f32], kinds: &[RecordKind], tags: &[String], limit: usize, min_score: Option<f64>, allowed: Option<&HashSet<i64>>) -> Result<Vec<(RecordKey, f64)>> {
        let query = normalize(query, self.dimension)?;
        let mut heap = BinaryHeap::<Candidate>::new();
        match &self.data {
            PartitionData::F32(values) => {
                for (i, row) in self.rows.iter().enumerate() {
                    if !self.matches(row, kinds, tags, allowed) { continue; }
                    let offset = i * self.dimension;
                    Self::retain(&mut heap, row.key, f64::from(dot(&query, &values[offset..offset + self.dimension])), limit, min_score);
                }
            }
            PartitionData::Sq8 { codes, scales } => {
                // 查询也量化一次，整条打分链路保持 i8；查询的 scale 对本分区所有行一致，不影响排序。
                let (query_codes, query_scale) = encode_query_sq8(&query);
                for (i, row) in self.rows.iter().enumerate() {
                    if !self.matches(row, kinds, tags, allowed) { continue; }
                    let offset = i * self.dimension;
                    let raw = query_scale * scales[i] * dot_codes(&query_codes, &codes[offset..offset + self.dimension]) as f32;
                    Self::retain(&mut heap, row.key, f64::from(raw), limit, min_score);
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
}
