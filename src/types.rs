use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;

pub type Metadata = Map<String, Value>;
pub fn default_namespace() -> String { "default".into() }
pub fn public_scope() -> String { "public".into() }
pub fn default_limit() -> usize { 50 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind { Memory, Entity, Relation, Event, Note, Chunk }

impl RecordKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory", Self::Entity => "entity", Self::Relation => "relation",
            Self::Event => "event", Self::Note => "note", Self::Chunk => "chunk",
        }
    }
    /// 固定枚举的整数编码，直接落库，不建字典表。
    pub fn code(self) -> i64 {
        match self {
            Self::Memory => 0, Self::Entity => 1, Self::Relation => 2,
            Self::Event => 3, Self::Note => 4, Self::Chunk => 5,
        }
    }
    pub fn from_code(code: i64) -> Option<Self> {
        Some(match code {
            0 => Self::Memory, 1 => Self::Entity, 2 => Self::Relation,
            3 => Self::Event, 4 => Self::Note, 5 => Self::Chunk, _ => return None,
        })
    }
}

impl fmt::Display for RecordKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
}

/// 内部记录键：单一自增整数，与任何外部 ID 无关。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RecordKey {
    pub id: i64,
}

impl RecordKey {
    pub fn new(id: i64) -> Self { Self { id } }
    pub(crate) fn index_key(&self) -> String { self.id.to_string() }
}

/// Source locations are snapshots; they do not change when note chunks are replaced.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub source: String,
    #[serde(default)]
    pub source_revision: Option<String>,
    #[serde(default)]
    pub chunk_id: Option<i64>,
    /// `offset` 为 1 起始的起始行，`limit` 为行数。结束行 = offset + limit - 1。
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub quote: String,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordInput {
    /// None 表示新建；Some(id) 表示更新既有记录。
    #[serde(default)]
    pub id: Option<i64>,
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(default = "public_scope")]
    pub scope: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub evidence: Vec<Evidence>,
    #[serde(default)]
    pub metadata: Metadata,
    #[serde(default)]
    pub created_at_us: Option<i64>,
    #[serde(default)]
    pub updated_at_us: Option<i64>,
    #[serde(default)]
    pub expected_revision: Option<i64>,
}

impl Default for RecordInput {
    fn default() -> Self {
        Self { id: None, namespace: default_namespace(), scope: public_scope(), tags: vec![],
            evidence: vec![], metadata: Metadata::new(), created_at_us: None,
            updated_at_us: None, expected_revision: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordHeader {
    pub id: i64,
    pub namespace: String,
    pub kind: RecordKind,
    pub scope: String,
    pub tags: Vec<String>,
    pub evidence: Vec<Evidence>,
    pub metadata: Metadata,
    pub created_at_us: i64,
    pub updated_at_us: i64,
    pub revision: i64,
}

impl RecordHeader {
    pub fn key(&self) -> RecordKey { RecordKey { id: self.id } }
    pub fn as_input(&self) -> RecordInput {
        RecordInput { id: Some(self.id), namespace: self.namespace.clone(), scope: self.scope.clone(), tags: self.tags.clone(),
            evidence: self.evidence.clone(), metadata: self.metadata.clone(), created_at_us: Some(self.created_at_us),
            updated_at_us: None, expected_revision: Some(self.revision) }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadFilter {
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(default = "default_scopes")]
    pub scopes: Vec<String>,
    /// All listed tags must match. Tag matching is case-insensitive.
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_scopes() -> Vec<String> { vec![public_scope()] }
impl Default for ReadFilter {
    fn default() -> Self { Self { namespace: default_namespace(), scopes: default_scopes(), tags: vec![] } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRequest {
    #[serde(default)]
    pub filter: ReadFilter,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub after: Option<String>,
}

impl Default for PageRequest {
    fn default() -> Self { Self { filter: ReadFilter::default(), limit: default_limit(), after: None } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page<T> { pub items: Vec<T>, pub next_cursor: Option<String> }

/// A committed write remains committed if the derived index cannot be updated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteReceipt<T> {
    pub value: T,
    pub revision: i64,
    pub index_ready: bool,
    pub index_error: Option<String>,
}

/// 一次检索实际落在了哪一档。多档降级要求「结果为什么变差」可被宿主读到。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Degrade {
    /// 该 namespace 关闭了向量化：只走全文。
    NamespaceDisabled,
    /// 目标空间没有注册嵌入回调：只走全文。
    NoEmbedder,
    /// 嵌入回调调用失败：只走全文。
    EmbedFailed,
    /// 重排回调调用失败或返回不符：按融合分排序。
    RerankFailed,
    /// 全文派生索引不可用：退到 SQLite 直查。
    TextIndexUnavailable,
}

/// 一次检索的可观测信息：走了哪几条路、是否重排、重排截断了几条、落在了哪一档。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchDiagnostics {
    pub text_used: bool,
    pub vector_used: bool,
    pub reranked: bool,
    /// 实际送入重排回调的候选数。
    pub rerank_candidates: usize,
    /// 因重排回调声明的 `max_docs` 而未送入重排的候选数。
    pub rerank_truncated: usize,
    #[serde(default)]
    pub degraded: Vec<Degrade>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub schema_version: i64,
    pub revision: i64,
    pub indexed_revision: i64,
    pub record_count: usize,
    pub index_document_count: usize,
    pub pending_index_updates: usize,
    pub sqlite_integrity: String,
    pub foreign_key_errors: usize,
    pub counts: std::collections::BTreeMap<String, usize>,
    /// 已注册嵌入回调的向量空间。
    pub embedder_spaces: Vec<String>,
    pub reranker_registered: bool,
    /// 最近观察到的降级档位，去重后保留少量。
    pub last_degraded: Vec<Degrade>,
}
