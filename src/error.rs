use thiserror::Error;

#[derive(Debug, Error)]
pub enum McpMemoryError {
    #[error("LatticeDB error: {0}")]
    Lattice(#[from] lattice_core::LatticeError),

    #[error("Embedding error: {message}")]
    Embedding { message: String },

    #[error("Chat provider error: {message}")]
    Chat { message: String },

    #[error("Reranker error: {message}")]
    Reranker { message: String },

    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Store error: {message}")]
    Store { message: String },

    #[error("Memory not found: {id}")]
    MemoryNotFound { id: u64 },

    #[error("Collection not initialized: {name}")]
    CollectionNotInitialized { name: String },

    #[error("Pin limit exceeded: {message}")]
    PinLimit { message: String },

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, McpMemoryError>;
