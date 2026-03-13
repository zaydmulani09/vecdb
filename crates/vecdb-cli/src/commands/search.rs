use anyhow::{Context, Result};
use clap::Subcommand;

use crate::{
    client::VecDbClient,
    output,
    types::{
        DenseSearchRequest, HybridSearchRequest, SearchResponse, SparseSearchRequest,
        SqlQueryRequest,
    },
};

#[derive(Subcommand, Debug)]
pub enum SearchCmd {
    /// Dense vector search
    Dense {
        #[arg(long)]
        collection: String,
        /// JSON array of floats, e.g. "[0.1,0.2,0.3]"
        #[arg(long)]
        vector: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// BM25 sparse / full-text search
    Sparse {
        #[arg(long)]
        collection: String,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// Hybrid dense + sparse search
    Hybrid {
        #[arg(long)]
        collection: String,
        /// JSON float array (optional)
        #[arg(long)]
        vector: Option<String>,
        #[arg(long)]
        query: Option<String>,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 0.7)]
        alpha: f32,
    },
    /// Execute a SQL query against the server
    Sql {
        #[arg(long)]
        sql: String,
    },
}

fn parse_vector(s: &str) -> Result<Vec<f32>> {
    serde_json::from_str(s)
        .with_context(|| format!("invalid vector — expected JSON float array, got: {}", s))
}

pub async fn run(client: &VecDbClient, cmd: SearchCmd) -> Result<()> {
    match cmd {
        SearchCmd::Dense {
            collection,
            vector,
            k,
        } => {
            let v = parse_vector(&vector)?;
            let body = DenseSearchRequest {
                vector: v,
                k: Some(k),
            };
            let resp: SearchResponse = client
                .post(&format!("/collections/{}/search/dense", collection), &body)
                .await?;
            output::print_search_results(&resp);
        }
        SearchCmd::Sparse {
            collection,
            query,
            k,
        } => {
            let body = SparseSearchRequest { query, k: Some(k) };
            let resp: SearchResponse = client
                .post(&format!("/collections/{}/search/sparse", collection), &body)
                .await?;
            output::print_search_results(&resp);
        }
        SearchCmd::Hybrid {
            collection,
            vector,
            query,
            k,
            alpha,
        } => {
            let parsed_vector = vector.as_deref().map(parse_vector).transpose()?;
            let body = HybridSearchRequest {
                vector: parsed_vector,
                query,
                k: Some(k),
                alpha: Some(alpha),
            };
            let resp: SearchResponse = client
                .post(&format!("/collections/{}/search/hybrid", collection), &body)
                .await?;
            output::print_search_results(&resp);
        }
        SearchCmd::Sql { sql } => {
            let body = SqlQueryRequest { sql };
            let resp: SearchResponse = client.post("/query", &body).await?;
            output::print_search_results(&resp);
        }
    }
    Ok(())
}
