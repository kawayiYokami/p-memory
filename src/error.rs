use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    Validation(String),
    #[error("record not found: {0}")]
    NotFound(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("data directory is already open: {0}")]
    Locked(String),
    #[error("knowledge base is closed")]
    Closed,
    #[error("unsupported database schema {found}; maximum supported version is {supported}")]
    SchemaVersion { found: i64, supported: i64 },
    #[error("invalid vector: {0}")]
    InvalidVector(String),
    #[error("stale content: {0}")]
    StaleRevision(String),
    #[error("index unavailable: {0}")]
    Index(String),
    #[error("SQLite: {0}")]
    Storage(#[from] rusqlite::Error),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<tantivy::TantivyError> for Error {
    fn from(value: tantivy::TantivyError) -> Self {
        Self::Index(value.to_string())
    }
}

impl Error {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Validation(_) | Self::Json(_) => "validation",
            Self::NotFound(_) => "not_found",
            Self::Conflict(_) => "conflict",
            Self::Locked(_) => "locked",
            Self::Closed => "closed",
            Self::SchemaVersion { .. } => "schema_version",
            Self::InvalidVector(_) => "invalid_vector",
            Self::StaleRevision(_) => "stale_revision",
            Self::Index(_) => "index",
            Self::Storage(_) => "storage",
            Self::Io(_) => "io",
        }
    }
}

