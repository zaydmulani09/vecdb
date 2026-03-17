use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{de::DeserializeOwned, Serialize};
use std::time::Duration;

pub struct BenchClient {
    client: reqwest::Client,
    base_url: String,
}

impl BenchClient {
    pub fn new(base_url: &str, api_key: &str) -> Result<Self> {
        let mut headers = HeaderMap::new();
        if !api_key.is_empty() {
            let val = HeaderValue::from_str(api_key)
                .context("api-key contains invalid header characters")?;
            headers.insert("X-Api-Key", val);
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    async fn check_status(resp: reqwest::Response) -> Result<reqwest::Response> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(anyhow!(
                "server returned 401 Unauthorized — set --api-key or VECDB_API_KEY"
            ));
        }
        let body = resp
            .text()
            .await
            .unwrap_or_else(|_| "<unreadable body>".to_string());
        Err(anyhow!("server returned {}: {}", status, body))
    }

    pub async fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let resp = self
            .client
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {} failed", path))?;
        let resp = Self::check_status(resp).await?;
        resp.json::<T>()
            .await
            .with_context(|| format!("failed to decode response from POST {}", path))
    }

    pub async fn delete_no_body<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let resp = self
            .client
            .delete(self.url(path))
            .send()
            .await
            .with_context(|| format!("DELETE {} failed", path))?;
        let resp = Self::check_status(resp).await?;
        resp.json::<T>()
            .await
            .with_context(|| format!("failed to decode response from DELETE {}", path))
    }
}
