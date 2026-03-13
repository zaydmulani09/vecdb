mod client;
mod commands;
mod output;
mod types;

#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::process;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "vecdb",
    version = "0.1.0",
    about = "vecdb command-line client — talk to a running vecdb-api server"
)]
struct Cli {
    /// Server base URL
    #[arg(
        long,
        env = "VECDB_SERVER",
        default_value = "http://localhost:8080",
        global = true
    )]
    server: String,

    /// API key (matches X-Api-Key header on the server)
    #[arg(long, env = "VECDB_API_KEY", global = true)]
    api_key: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Check if the server is reachable
    Ping,

    /// Manage collections
    Collection {
        #[command(subcommand)]
        cmd: commands::collection::CollectionCmd,
    },

    /// Ingest records from a JSONL or CSV file (or stdin)
    Ingest {
        /// Target collection name
        #[arg(long)]
        collection: String,
        /// Path to .jsonl or .csv file; omit to read stdin
        #[arg(long)]
        file: Option<PathBuf>,
        /// Records per HTTP batch
        #[arg(long, default_value_t = 100)]
        batch_size: usize,
        /// Parse without sending to the server
        #[arg(long)]
        dry_run: bool,
    },

    /// Show details about a collection
    Inspect {
        /// Collection name
        collection: String,
    },

    /// Search a collection
    Search {
        #[command(subcommand)]
        cmd: commands::search::SearchCmd,
    },
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {:#}", e);
        process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let client = client::VecDbClient::new(cli.server, cli.api_key)?;

    match cli.command {
        Commands::Ping => commands::ping::run(&client).await,
        Commands::Collection { cmd } => commands::collection::run(&client, cmd).await,
        Commands::Ingest {
            collection,
            file,
            batch_size,
            dry_run,
        } => {
            commands::ingest::run(&client, &collection, file.as_deref(), batch_size, dry_run).await
        }
        Commands::Inspect { collection } => commands::inspect::run(&client, &collection).await,
        Commands::Search { cmd } => commands::search::run(&client, cmd).await,
    }
}
