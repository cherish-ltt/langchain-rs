use async_trait::async_trait;
use futures_core::Stream;
use im::Vector;
use std::pin::Pin;

use crate::{
    message::{Message, ToolCall},
    request::ToolSpec,
    response::Usage,
};

#[derive(Debug, Clone, Default)]
pub struct MessagesState {
    pub messages: Vector<Message>,
    pub llm_calls: u32,
}

impl MessagesState {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            llm_calls: 0,
        }
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push_back(message);
    }

    pub fn extend_messages(&mut self, messages: Vec<Message>) {
        self.messages.extend(messages);
    }

    pub fn increment_llm_calls(&mut self) {
        self.llm_calls = self.llm_calls.saturating_add(1);
    }

    pub fn last_message(&self) -> Option<&Message> {
        let len = self.messages.len();
        if len == 0 {
            None
        } else {
            self.messages.get(len - 1)
        }
    }

    pub fn last_assistant(&self) -> Option<&Message> {
        self.messages
            .iter()
            .rev()
            .find(|m| matches!(m, Message::Assistant { .. }))
    }

    pub fn last_tool_calls(&self) -> Option<&[ToolCall]> {
        match self.last_assistant() {
            Some(Message::Assistant {
                tool_calls: Some(calls),
                ..
            }) => Some(calls.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatCompletion {
    pub messages: Vec<Message>,
    pub usage: Usage,
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Content(String),
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        type_name: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
    Done {
        finish_reason: Option<String>,
        usage: Option<Usage>,
    },
}

pub type ChatStream<E> = Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, E>> + Send>>;

#[async_trait]
pub trait ChatModel: Send + Sync {
    type Error: Send + Sync + 'static;

    async fn invoke(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolSpec>,
    ) -> Result<ChatCompletion, Self::Error>;

    async fn stream(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolSpec>,
    ) -> Result<ChatStream<Self::Error>, Self::Error>;
}
