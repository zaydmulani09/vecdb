use anyhow::Result;

use crate::{client::VecDbClient, output, types::HealthResponse};

pub async fn run(client: &VecDbClient) -> Result<()> {
    let health: HealthResponse = client.get("/health").await?;
    output::print_health(&health);
    Ok(())
}
