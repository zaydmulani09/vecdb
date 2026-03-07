use clap::Parser;
use vecdb_core::ServerConfig;

#[derive(Parser, Debug)]
#[command(name = "vecdb", about = "Open source vector database")]
struct Args {
    #[arg(long)]
    config: Option<String>,

    #[arg(long)]
    port: Option<u16>,

    #[arg(long)]
    data_dir: Option<String>,

    #[arg(long)]
    log_level: Option<String>,

    #[arg(long)]
    api_key: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut config = ServerConfig::from_env_and_file(args.config.as_deref()).unwrap_or_default();

    if let Some(port) = args.port {
        config.port = port;
    }
    if let Some(dir) = args.data_dir {
        config.data_dir = dir.into();
    }
    if let Some(level) = args.log_level {
        config.log_level = level;
    }
    if let Some(key) = args.api_key {
        config.api_key = Some(key);
    }

    vecdb_api::server::run(config).await
}
