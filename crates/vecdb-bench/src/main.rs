mod client;
mod datasets;
mod runner;
mod types;

#[cfg(test)]
mod tests;

use anyhow::Result;
use clap::Parser;
use runner::{BenchmarkConfig, BenchmarkResult, BenchmarkRunner, DatasetKind, SearchType};

#[derive(Parser, Debug)]
#[command(
    name = "vecdb-bench",
    version = "0.1.0",
    about = "vecdb benchmark harness — measures recall, latency, and throughput of a running vecdb server"
)]
struct Cli {
    #[arg(long, default_value = "http://localhost:8080", env = "VECDB_SERVER")]
    server: String,

    #[arg(long, default_value = "", env = "VECDB_API_KEY")]
    api_key: String,

    /// Dataset source: "synthetic" or "jsonl"
    #[arg(long, default_value = "synthetic")]
    dataset: String,

    /// Path to .jsonl file (required when --dataset jsonl)
    #[arg(long)]
    dataset_file: Option<std::path::PathBuf>,

    /// Corpus size for synthetic dataset
    #[arg(long, default_value_t = 10_000)]
    n: usize,

    /// Vector dimension for synthetic dataset
    #[arg(long, default_value_t = 128)]
    dim: usize,

    /// RNG seed for synthetic dataset
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of query vectors to run
    #[arg(long, default_value_t = 100)]
    queries: usize,

    /// Top-k results per query
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Upsert batch size
    #[arg(long, default_value_t = 500)]
    batch_size: usize,

    /// Collection name to create on the server
    #[arg(long, default_value = "bench")]
    collection: String,

    /// Output format: "table" or "json"
    #[arg(long, default_value = "table")]
    output: String,
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {:#}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .try_init();

    let dataset = match cli.dataset.as_str() {
        "jsonl" => {
            let path = cli.dataset_file.unwrap_or_else(|| {
                eprintln!("error: --dataset jsonl requires --dataset-file <PATH>");
                std::process::exit(1);
            });
            DatasetKind::Jsonl { path }
        }
        _ => DatasetKind::Synthetic {
            n: cli.n,
            dim: cli.dim,
            seed: cli.seed,
        },
    };

    let config = BenchmarkConfig {
        dataset,
        index_type: "hnsw".to_string(),
        search_type: SearchType::Dense,
        query_count: cli.queries,
        k: cli.k,
        alpha: 0.7,
        collection: cli.collection,
        batch_size: cli.batch_size,
        server_url: cli.server,
        api_key: cli.api_key,
    };

    let result = BenchmarkRunner::run(config).await?;

    match cli.output.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&result)?),
        _ => print_table(&result),
    }

    Ok(())
}

fn print_table(r: &BenchmarkResult) {
    println!("| Metric            | Value     |");
    println!("|-------------------|-----------|");
    println!("| Dataset size      | {}        |", r.dataset_size);
    println!("| Query count       | {}        |", r.query_count);
    println!("| k                 | {}         |", r.k);
    println!("| Recall@k          | {:.4}     |", r.recall_at_k);
    println!("| p50 latency (ms)  | {:.2}      |", r.p50_ms);
    println!("| p95 latency (ms)  | {:.2}      |", r.p95_ms);
    println!("| p99 latency (ms)  | {:.2}      |", r.p99_ms);
    println!("| Mean latency (ms) | {:.2}      |", r.mean_ms);
    println!("| QPS               | {:.1}      |", r.qps);
}
