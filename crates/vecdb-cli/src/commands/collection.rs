use anyhow::Result;
use clap::Subcommand;

use crate::{
    client::VecDbClient,
    output,
    types::{
        CollectionResponse, CreateCollectionRequest, DeleteCollectionResponse,
        ListCollectionsResponse,
    },
};

#[derive(Subcommand, Debug)]
pub enum CollectionCmd {
    /// List all collections
    List,
    /// Get details about a collection
    Get {
        /// Collection name
        name: String,
    },
    /// Create a new collection
    Create {
        /// Collection name
        name: String,
        /// Vector dimension
        #[arg(long)]
        dimension: usize,
        /// Distance metric: cosine, euclidean, dot
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
    /// Delete a collection
    Delete {
        /// Collection name
        name: String,
    },
}

pub async fn run(client: &VecDbClient, cmd: CollectionCmd) -> Result<()> {
    match cmd {
        CollectionCmd::List => {
            let resp: ListCollectionsResponse = client.get("/collections").await?;
            println!("Collections: {}", resp.count);
            println!();
            for c in &resp.collections {
                output::print_collection(c);
                println!();
            }
        }
        CollectionCmd::Get { name } => {
            let resp: CollectionResponse = client.get(&format!("/collections/{}", name)).await?;
            output::print_collection(&resp);
        }
        CollectionCmd::Create {
            name,
            dimension,
            metric,
        } => {
            let body = CreateCollectionRequest {
                name: name.clone(),
                dimension,
                metric: Some(metric),
            };
            let resp: CollectionResponse = client.post("/collections", &body).await?;
            println!("Collection '{}' created.", resp.name);
            output::print_collection(&resp);
        }
        CollectionCmd::Delete { name } => {
            let resp: DeleteCollectionResponse = client
                .delete_no_body(&format!("/collections/{}", name))
                .await?;
            println!("Collection '{}' deleted: {}", resp.name, resp.deleted);
        }
    }
    Ok(())
}
