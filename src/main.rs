use anyhow::Result;
use clap::Parser;
use rmcp::ServiceExt;

use lattice_mcp::config::{expand_path, Config, ConsolidationMode, DatabaseMode};
use lattice_mcp::embedding::chat::OpenAiChat;
use lattice_mcp::embedding::openai::OpenAiEmbedder;
use lattice_mcp::memory::MemorySystem;
use lattice_mcp::reranker::http::HttpReranker;
use lattice_mcp::store::embedded::EmbeddedStore;
use lattice_mcp::store::remote::RemoteStore;
use lattice_mcp::tools::MemoryServer;

#[derive(Parser)]
#[command(
    name = "lattice-mcp",
    about = "Agent supervisor memory system powered by LatticeDB"
)]
struct Args {
    /// Path to configuration file
    #[arg(long, default_value = "~/.lattice-mcp/config.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to stderr (stdout reserved for MCP stdio)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "lattice_mcp=info".into()),
        )
        .init();

    let args = Args::parse();
    let config_path = expand_path(&args.config);
    let config = Config::from_file(&config_path)?;

    tracing::info!("Starting lattice-mcp");

    // Build embedding provider
    let embedder = Box::new(OpenAiEmbedder::new(&config.embedding)?);

    // Build optional chat provider (for LLM consolidation)
    let chat: Option<Box<dyn lattice_mcp::embedding::chat::ChatProvider>> =
        match config.consolidation.mode {
            ConsolidationMode::Llm | ConsolidationMode::Auto => {
                match OpenAiChat::new(&config.consolidation, config.embedding.api_key_env.as_deref())? {
                    Some(c) => Some(Box::new(c)),
                    None => None,
                }
            }
            ConsolidationMode::Algorithmic => None,
        };

    // Build memory store
    let store: Box<dyn lattice_mcp::store::MemoryStore> = match config.database.mode {
        DatabaseMode::Embedded => {
            let data_dir = expand_path(&config.database.data_dir.to_string_lossy());
            Box::new(EmbeddedStore::new(data_dir, config.embedding.dimensions))
        }
        DatabaseMode::Remote => {
            let url = config
                .database
                .remote_url
                .as_ref()
                .expect("remote_url required for remote mode")
                .clone();
            Box::new(RemoteStore::new(url, config.embedding.dimensions))
        }
    };

    // Build optional reranker provider
    let reranker: Option<Box<dyn lattice_mcp::reranker::RerankerProvider>> =
        match &config.reranker {
            Some(rr_cfg) => Some(Box::new(HttpReranker::new(rr_cfg)?)),
            None => None,
        };

    // Build memory system
    let mut memory_system = MemorySystem::new(store, embedder, chat, reranker, config);
    memory_system.initialize().await?;

    tracing::info!("Memory system initialized, starting MCP server");

    // Create and serve MCP server via stdio
    let server = MemoryServer::new(memory_system);

    let service = server.serve(rmcp::transport::io::stdio()).await?;
    service.waiting().await?;

    Ok(())
}
