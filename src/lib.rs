//! Shared, offline storage and retrieval for memories, graphs and notes.
//!
//! SQLite is authoritative. Full-text indexes are replayable projections.
//! Model requests and filesystem watching belong to callers.

mod error;
mod schema;
mod storage;
mod index;
pub mod text;
pub mod types;
pub mod memory;
pub mod graph;
pub mod graph_search;
pub mod notes;
pub mod embeddings;
pub mod search;
pub mod legacy;
pub mod protocol;

pub use error::{Error, Result};
pub use storage::KnowledgeBase;
pub use types::*;
pub use memory::{Memory, MemoryInput, MemoryState, DecayPolicy, FeedbackRequest};
pub use graph::{Entity, EntityInput, Relation, RelationInput, Event, EventInput, GraphBatch};
pub use notes::{Note, NoteFileInput, NoteInput, Chunk};
pub use embeddings::{EmbeddingSpace, Embedder, EmbedderOptions, EmbedCallbackError, EmbedErrorKind, SyncReport};
pub use search::{SearchRequest, SearchResult, SearchHit, GraphPrune, ContextualHit, Reranker, RerankerOptions};

