use async_stream::try_stream;
use futures::{StreamExt, stream::BoxStream};
use langchain::{LlmModel, NodeRunError};
use langchain_core::{
    message::{FunctionCall, Message, ToolCall},
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

#[derive(Default)]
struct StreamingToolCall {
    id: Option<String>,
    type_name: Option<String>,
    function_name: Option<String>,
    arguments: String,
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

        let resp = self
            .client
            .post(format!("{}{CHAT_COMPLETIONS}", self.base_url))
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(OpenAiError::Request)
            .map_err(|e| NodeRunError::LlmRunError(Box::new(e)))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(OpenAiError::Request)
            .map_err(|e| NodeRunError::LlmRunError(Box::new(e)))?;

        if !status.is_success() {
            tracing::error!("OpenAI API error: status = {}, body = {}", status, text);
            return Err(NodeRunError::LlmRunError(Box::new(OpenAiError::ApiError(
                format!("HTTP {}: {}", status, text),
            ))));
        }

        let response: ResponseBody = serde_json::from_str(&text)
            .map_err(OpenAiError::Serialization)
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

    fn stream(
        &self,
        state: MessageState,
        tools: Vec<ToolSpec>,
    ) -> BoxStream<'static, Result<MessageDiff, NodeRunError>> {
        let this = self.clone();
        let stream = try_stream! {
            let mut headers = HeaderMap::new();
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", this.api_key))
                    .map_err(OpenAiError::InvalidHeader)
                    .map_err(|e| NodeRunError::LlmRunError(Box::new(e)))?,
            );
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            let mut body = RequestBody::from_model(&this.model)
                .with_messages(state.messages.clone())
                .with_tools(tools);
            body.stream = true;

            tracing::debug!("请求Body:\n{:?}", &body);

            let resp = this
                .client
                .post(format!("{}{CHAT_COMPLETIONS}", this.base_url))
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(OpenAiError::Request)
                .map_err(|e| NodeRunError::LlmRunError(Box::new(e)))?;

            let status = resp.status();

            if !status.is_success() {
                Err(NodeRunError::LlmRunError(Box::new(OpenAiError::ApiError(
                    format!("HTTP {}", status),
                ))))?;
            }

            let mut byte_stream = resp.bytes_stream();
            let mut full_content = String::new();
            let mut pending_tool_calls: Vec<StreamingToolCall> = Vec::new();

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk
                    .map_err(OpenAiError::Request)
                    .map_err(|e| NodeRunError::LlmRunError(Box::new(e)))?;
                let text = String::from_utf8_lossy(&chunk);
                for line in text.lines() {
                    if !line.starts_with("data: ") {
                        continue;
                    }
                    let data = line.trim_start_matches("data: ").trim();
                    if data == "[DONE]" {
                        break;
                    }
                    let json: serde_json::Value = serde_json::from_str(data)
                        .map_err(OpenAiError::Serialization)
                        .map_err(|e| NodeRunError::LlmRunError(Box::new(e)))?;
                    if let Some(delta) = json
                        .get("choices")
                        .and_then(|choices| choices.get(0))
                        .and_then(|choice| choice.get("delta")) {
                        if let Some(content) = delta
                            .get("content")
                            .and_then(|c| c.as_str())
                        {
                            full_content.push_str(content);
                            yield MessageDiff {
                                new_messages: vec![Message::assistant(content.to_string())],
                                llm_calls_delta: 0,
                            };
                        }
                        if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                            for tc in tool_calls {
                                let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                if pending_tool_calls.len() <= index {
                                    pending_tool_calls.resize_with(index + 1, StreamingToolCall::default);
                                }
                                let pending = &mut pending_tool_calls[index];
                                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                    if pending.id.is_none() {
                                        pending.id = Some(id.to_string());
                                    }
                                }
                                if let Some(type_name) = tc.get("type").and_then(|v| v.as_str()) {
                                    if pending.type_name.is_none() {
                                        pending.type_name = Some(type_name.to_string());
                                    }
                                }
                                if let Some(function) = tc.get("function") {
                                    if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
                                        if pending.function_name.is_none() {
                                            pending.function_name = Some(name.to_string());
                                        }
                                    }
                                    if let Some(args) = function.get("arguments").and_then(|v| v.as_str()) {
                                        pending.arguments.push_str(args);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let mut tool_calls = Vec::new();
            for pending in pending_tool_calls.into_iter() {
                if let (Some(id), Some(type_name), Some(function_name)) =
                    (pending.id, pending.type_name, pending.function_name)
                {
                    let function = FunctionCall {
                        name: function_name,
                        arguments: serde_json::Value::String(pending.arguments),
                    };
                    let tc = ToolCall {
                        id,
                        type_name,
                        function,
                    };
                    tool_calls.push(tc);
                }
            }

            let _ = !full_content.is_empty() || !tool_calls.is_empty();
            tracing::warn!("tool_calls: {:?}", tool_calls);
            if !tool_calls.is_empty() {
                // TODO: 处理tool_calls
                let _ = Message::Assistant {
                    content: full_content.clone(),
                    tool_calls: Some(tool_calls),
                    name: None,
                };
            }


        };
        Box::pin(stream)
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
