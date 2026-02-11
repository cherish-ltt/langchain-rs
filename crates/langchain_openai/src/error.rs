use langchain_core::error::ModelError;
use thiserror::Error;

/// OpenAI API 错误
#[derive(Debug, Error)]
pub enum OpenAIError {
    /// 无效的 API 密钥
    #[error("无效的 API 密钥")]
    InvalidApiKey,
    /// 无效的请求头
    #[error("无效的请求头值: {0}")]
    InvalidHeaderValue(String),
    /// 模型不存在
    #[error("模型不存在")]
    ModelNotFound,
    /// 超时错误    
    #[error("超时错误")]
    Timeout,
    /// 网络错误
    #[error("网络错误")]
    Http(reqwest::Error),
    /// 响应体解析错误
    #[error("响应体解析错误")]
    ResponseBodyParse(reqwest::Error),
    /// 其他错误
    #[error("其他错误: {0}")]
    Other(String),
}

impl From<OpenAIError> for ModelError {
    fn from(e: OpenAIError) -> Self {
        match e {
            OpenAIError::InvalidApiKey => ModelError::InvalidApiKey,
            OpenAIError::ModelNotFound => ModelError::ModelNotFound("OpenAI model".to_owned()),
            OpenAIError::Timeout => ModelError::Timeout(0),
            OpenAIError::Http(e) => ModelError::RequestFailed(e),
            OpenAIError::ResponseBodyParse(e) => ModelError::RequestFailed(e),
            OpenAIError::InvalidHeaderValue(s) => {
                ModelError::Other(Box::new(OpenAIError::InvalidHeaderValue(s)))
            }
            OpenAIError::Other(s) => ModelError::ResponseError(s),
        }
    }
}
