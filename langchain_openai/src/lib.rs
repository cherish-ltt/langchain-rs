use langchain::{LlmModel, NodeRunError};
use langchain_core::{
    request::{RequestBody, ToolSpec},
    response::ResponseBody,
    state::{MessageDiff, MessageState},
};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};

pub mod error;

pub use error::OpenAiError;

pub const CHAT_COMPLETIONS: &str = "/chat/completions";

#[derive(Debug, Clone)]
pub struct ChatOpenAi {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl ChatOpenAi {
    pub fn new<T: Into<String>>(base_url: T, api_key: T, model: T) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl LlmModel for ChatOpenAi {
    async fn invoke(
        &self,
        state: &MessageState,
        tools: &[ToolSpec],
    ) -> Result<MessageDiff, NodeRunError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(OpenAiError::InvalidHeader)
                .map_err(|e| NodeRunError::LlmRunError(Box::new(e)))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let body = RequestBody::from_model(&self.model)
            .with_messages(state.messages.clone())
            .with_tools(tools.to_vec());
        tracing::debug!("请求Body:\n{:?}", &body);

        let response = self
            .client
            .post(format!("{}{CHAT_COMPLETIONS}", self.base_url))
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(OpenAiError::Request)
            .map_err(|e| NodeRunError::LlmRunError(Box::new(e)))?
            .json::<ResponseBody>()
            .await
            .map_err(OpenAiError::Request)
            .map_err(|e| NodeRunError::LlmRunError(Box::new(e)))?;
        tracing::debug!("回复Body:\n{:?}", response);

        let message = response
            .choices
            .into_iter()
            .map(|choice| choice.message)
            .collect::<Vec<_>>();

        Ok(MessageDiff {
            new_messages: message,
            llm_calls_delta: 1,
        })
    }
}

pub struct ChatOpenAiBuilder {
    base_url: String,
    api_key: String,
    model: String,
}

impl ChatOpenAiBuilder {
    pub fn new<T: Into<String>>(base_url: T, api_key: T, model: T) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    pub fn build(self) -> ChatOpenAi {
        ChatOpenAi {
            base_url: self.base_url,
            api_key: self.api_key,
            model: self.model,
            client: reqwest::Client::new(),
        }
    }
}
