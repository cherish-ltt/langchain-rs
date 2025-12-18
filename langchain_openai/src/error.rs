use thiserror::Error;

#[derive(Error, Debug)]
pub enum OpenAiError {
    #[error("Invalid API Key")]
    InvalidApiKey,
    #[error("Request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("Serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Invalid header value: {0}")]
    InvalidHeader(#[from] reqwest::header::InvalidHeaderValue),
    #[error("API Error: {0}")]
    ApiError(String),
}
