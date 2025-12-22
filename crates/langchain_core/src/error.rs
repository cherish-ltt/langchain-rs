use std::error::Error;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("tool call error: {0}")]
    ToolCall(#[source] Box<dyn Error + Send + Sync>),
}

impl ToolError {
    pub fn tool_call<E>(e: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::ToolCall(Box::new(e))
    }
}
