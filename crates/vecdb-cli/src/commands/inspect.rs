use anyhow::Result;

use crate::{client::VecDbClient, output, types::CollectionResponse};

pub async fn run(client: &VecDbClient, collection: &str) -> Result<()> {
    let resp: CollectionResponse = client.get(&format!("/collections/{}", collection)).await?;
    output::print_collection(&resp);
    Ok(())
}
