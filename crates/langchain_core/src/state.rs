use async_trait::async_trait;
use futures_core::Stream;
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    message::{Message, ToolCall},
    request::{ToolFunction, ToolSpec},
};

#[derive(Debug, Clone, Default)]
pub struct MessagesState {
    pub messages: Vec<Message>,
    pub llm_calls: u32,
}

impl MessagesState {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            llm_calls: 0,
        }
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn increment_llm_calls(&mut self) {
        self.llm_calls = self.llm_calls.saturating_add(1);
    }

    pub fn last_message(&self) -> Option<&Message> {
        self.messages.last()
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

pub type ChatStream<E> = Pin<Box<dyn Stream<Item = Result<Message, E>> + Send>>;

#[async_trait]
pub trait ChatModel: Send + Sync {
    type Error: Send + Sync + 'static;

    async fn invoke(&self, messages: &[Message]) -> Result<Message, Self::Error>;

    async fn stream(&self, messages: &[Message]) -> Result<ChatStream<Self::Error>, Self::Error>;
}

/// 工具函数统一使用的异步返回类型。
/// 将任意 `Future<Output = Result<Value, E>>` 封装为可在运行时保存和调度的 trait object。
pub type ToolFuture<E> = Pin<Box<dyn Future<Output = Result<Value, E>> + Send>>;

/// 工具函数的标准签名：接收 JSON 参数并返回异步结果。
/// 适用于通过 LLM 调用的任意工具，实现方通常使用 `async fn` 或 `async move` 闭包适配到该类型。
pub type ToolFn<E> = dyn Fn(Value) -> ToolFuture<E> + Send + Sync;

pub struct RegisteredTool<E> {
    pub function: ToolFunction,
    pub handler: Box<ToolFn<E>>,
}

impl<E> RegisteredTool<E> {
    pub fn new(
        name: String,
        description: String,
        parameters: Value,
        handler: Box<ToolFn<E>>,
    ) -> Self {
        let function = ToolFunction {
            name,
            description,
            parameters,
        };
        Self { function, handler }
    }

    pub fn spec(&self) -> ToolSpec {
        ToolSpec::Function {
            function: self.function.clone(),
        }
    }
}

#[macro_export]
macro_rules! tool {
    // 无参数描述版本
    ($name:expr, $description:expr, |$($arg:ident : $ty:ty),*| $body:expr) => {{

        #[derive(::serde::Deserialize, ::schemars::JsonSchema)]
        struct __ToolArgs {
            $( $arg: $ty ),*
        }

        $crate::state::RegisteredTool::<serde_json::Error>::from_typed(
            $name.to_string(),
            $description.to_string(),
            |args: __ToolArgs| async move {
                let __ToolArgs { $( $arg ),* } = args;
                $body
            }
        )
    }};

    // 带参数描述版本，顺序与参数列表相同
    ($name:expr, $description:expr, |$($arg:ident : $ty:ty),*| $body:expr, $($desc:expr),* $(,)?) => {{
        #[derive(::serde::Deserialize, ::schemars::JsonSchema)]
        struct __ToolArgs {
            $(
                #[doc = $desc]
                $arg: $ty
            ),*
        }

        $crate::state::RegisteredTool::<serde_json::Error>::from_typed(
            $name.to_string(),
            $description.to_string(),
            |args: __ToolArgs| async move {
                let __ToolArgs { $( $arg ),* } = args;
                $body
            }
        )
    }};
}

impl<E> RegisteredTool<E>
where
    E: From<serde_json::Error> + Send + Sync + 'static,
{
    pub fn from_typed<F, Fut, Args, Output>(name: String, description: String, f: F) -> Self
    where
        F: Fn(Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Output, E>> + Send + 'static,
        Args: DeserializeOwned + JsonSchema + Send + 'static,
        Output: Serialize + Send + 'static,
    {
        let schema = schemars::schema_for!(Args);
        let mut parameters = serde_json::to_value(schema.schema).unwrap();
        if let Value::Object(map) = &mut parameters {
            map.remove("title");
        }
        let f = Arc::new(f);
        let handler: Box<ToolFn<E>> = Box::new(move |value: Value| {
            let f = f.clone();
            Box::pin(async move {
                let args: Args = serde_json::from_value(value).map_err(E::from)?;
                let output = (f.as_ref())(args).await?;
                let value = serde_json::to_value(output).map_err(E::from)?;
                Ok(value)
            })
        });
        RegisteredTool::new(name, description, parameters, handler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug)]
    enum TestError {
        Json(serde_json::Error),
    }

    impl From<serde_json::Error> for TestError {
        fn from(e: serde_json::Error) -> Self {
            TestError::Json(e)
        }
    }

    #[derive(Deserialize, JsonSchema)]
    struct AddArgs {
        a: i32,
        b: i32,
    }

    async fn add(args: AddArgs) -> Result<i32, TestError> {
        Ok(args.a + args.b)
    }

    #[test]
    fn registered_tool_from_typed_builds_schema() {
        let tool: RegisteredTool<TestError> = RegisteredTool::from_typed(
            "add".to_string(),
            "add numbers".to_string(),
            |args: AddArgs| add(args),
        );
        assert_eq!(tool.function.name, "add");
        if let Value::Object(map) = &tool.function.parameters {
            assert!(map.get("type").is_some());
            assert!(map.get("properties").is_some());
            assert!(map.get("required").is_some());
            assert!(map.get("title").is_none());
        } else {
            panic!("parameters must be object");
        }
    }

    #[test]
    fn tool_macro_builds_registered_tool_basic() {
        let tool: RegisteredTool<serde_json::Error> = tool!(
            "calc_add",
            "add numbers via llm",
            |a: i32, b: i32| Ok(a + b)
        );

        assert_eq!(tool.function.name, "calc_add");
        if let Value::Object(map) = &tool.function.parameters {
            assert!(map.get("properties").is_some());
        } else {
            panic!("parameters must be object");
        }
    }

    #[test]
    fn tool_macro_builds_registered_tool_with_descriptions() {
        let tool: RegisteredTool<serde_json::Error> = tool!(
            "calc_add2",
            "add numbers via llm 2",
            |a: i32, b: i32| Ok(a + b),
            "first number",
            "second number",
        );

        assert_eq!(tool.function.name, "calc_add2");
        println!(
            "{}",
            serde_json::to_string_pretty(&tool.function.parameters).unwrap()
        );
        if let Value::Object(map) = &tool.function.parameters {
            assert!(map.get("properties").is_some());
        } else {
            panic!("parameters must be object");
        }
    }
}
