use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub log_level: String,
    #[serde(default)]
    pub api_key: Option<String>,
    pub max_connections: usize,
    pub query_timeout_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 6333,
            data_dir: PathBuf::from("./data"),
            log_level: "info".to_string(),
            api_key: None,
            max_connections: 100,
            query_timeout_ms: 5000,
        }
    }
}

impl ServerConfig {
    pub fn from_env_and_file(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = ::config::Config::builder()
            .set_default("host", "127.0.0.1")?
            .set_default("port", 6333_i64)?
            .set_default("data_dir", "./data")?
            .set_default("log_level", "info")?
            .set_default("max_connections", 100_i64)?
            .set_default("query_timeout_ms", 5000_i64)?;

        if let Some(p) = path {
            builder = builder.add_source(::config::File::with_name(p).required(false));
        }

        builder = builder.add_source(::config::Environment::with_prefix("VECDB").separator("__"));

        Ok(builder.build()?.try_deserialize()?)
    }
}
