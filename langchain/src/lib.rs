use futures::{StreamExt, future::join_all, stream, stream::BoxStream};
use langchain_core::{
    message::Message,
    request::ToolSpec,
    state::{MessageDiff, MessageState},
};
use langgraph::{
    graph::StateGraph,
    graph_runner::{DEFAULT_MAX_STEPS, GraphRunnerError, GraphStepper, StepEvent},
    node::{BaseAgentLabel, GraphLabel, InternedGraphLabel, Node},
};
use std::{collections::HashMap, sync::Arc};

pub use langchain_macros::{tool, tools_from_fns};
pub use langgraph::node::NodeRunError;

#[async_trait::async_trait]
pub trait LlmModel: Clone + Send + Sync + 'static {
    async fn invoke(
        &self,
        state: &MessageState,
        tools: &[ToolSpec],
    ) -> Result<MessageDiff, NodeRunError>;
    fn stream(
        &self,
        state: MessageState,
        tools: Vec<ToolSpec>,
    ) -> BoxStream<'static, Result<MessageDiff, NodeRunError>> {
        let this = self.clone();
        stream::once(async move { this.invoke(&state, &tools).await }).boxed()
    }
}

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    type Output: serde::Serialize;
    fn spec(&self) -> ToolSpec;
    async fn invoke(
        &self,
        state: &MessageState,
        args: serde_json::Value,
    ) -> Result<Self::Output, NodeRunError>;
}

pub type DynTool = Arc<dyn Tool<Output = serde_json::Value> + Send + Sync>;

pub trait IntoDynTool {
    fn into_dyn_tool(self) -> DynTool;
}

impl<T> IntoDynTool for T
where
    T: Tool<Output = serde_json::Value> + Send + Sync + 'static,
{
    fn into_dyn_tool(self) -> DynTool {
        Arc::new(self)
    }
}

impl IntoDynTool for DynTool {
    fn into_dyn_tool(self) -> DynTool {
        self
    }
}

pub trait AgentMiddleware: Send + Sync {
    fn before_run(&self, _state: &MessageState) {}
    fn after_run(&self, _state: &MessageState) {}
    fn on_run_error(&self, _state: &MessageState, _error: &GraphRunnerError) {}
    fn before_model(&self, _state: &MessageState, _tools: &[ToolSpec]) {}
    fn after_model(&self, _state: &MessageState) {}
    fn on_model_error(&self, _state: &MessageState, _error: &NodeRunError) {}
    fn before_tool(&self, _state: &MessageState, _tool_name: &str) {}
    fn after_tool(&self, _state: &MessageState, _tool_name: &str) {}
    fn on_tool_error(&self, _state: &MessageState, _tool_name: &str, _error: &NodeRunError) {}
}

pub type DynAgentMiddleware = Arc<dyn AgentMiddleware + Send + Sync>;

#[derive(Debug, Hash, PartialEq, Eq, Clone, GraphLabel)]
pub enum AgentLabel {
    CallModel,
    ToolExecutor,
}

pub struct LlmNode<M> {
    pub model: M,
    pub tool_specs: Vec<ToolSpec>,
    pub middlewares: Vec<DynAgentMiddleware>,
}

#[async_trait::async_trait]
impl<M> Node<MessageState> for LlmNode<M>
where
    M: LlmModel + Send + Sync + 'static,
{
    async fn run(&self, state: &MessageState) -> Result<MessageDiff, NodeRunError> {
        for middleware in &self.middlewares {
            middleware.before_model(state, &self.tool_specs);
        }
        let result = self.model.invoke(state, &self.tool_specs).await;
        match &result {
            Ok(_) => {
                for middleware in &self.middlewares {
                    middleware.after_model(state);
                }
            }
            Err(error) => {
                for middleware in &self.middlewares {
                    middleware.on_model_error(state, error);
                }
            }
        }
        result
    }
}

pub struct ToolNode {
    tools: HashMap<String, DynTool>,
    middlewares: Vec<DynAgentMiddleware>,
}

#[async_trait::async_trait]
impl Node<MessageState> for ToolNode {
    async fn run(&self, state: &MessageState) -> Result<MessageDiff, NodeRunError> {
        if let Some(last_message) = state.messages.last() {
            if let Message::Assistant { tool_calls, .. } = last_message
                && tool_calls.is_some()
            {
                let tool_calls = tool_calls.as_ref().unwrap();

                let mut futures = Vec::new();

                tracing::info!("工具调用个数: {}", tool_calls.len());

                for call in tool_calls {
                    let tool_name = call.function_name().to_string();
                    let args = call.arguments();
                    let id = call.id().to_string();

                    let tool = match self.tools.get(&tool_name).cloned() {
                        Some(tool) => tool,
                        None => {
                            let available: Vec<String> = self.tools.keys().cloned().collect();
                            let error = NodeRunError::ToolRunError(format!(
                                "tool '{}' not found. available tools: {}",
                                tool_name,
                                available.join(", ")
                            ));
                            for middleware in &self.middlewares {
                                middleware.on_tool_error(state, &tool_name, &error);
                            }
                            return Err(error);
                        }
                    };

                    for middleware in &self.middlewares {
                        middleware.before_tool(state, &tool_name);
                    }

                    let state_ref = state;
                    let middlewares = self.middlewares.clone();
                    let tool_name_clone = tool_name.clone();
                    let fut = async move {
                        let result = tool.invoke(state_ref, args).await;
                        match result {
                            Ok(content) => {
                                for middleware in &middlewares {
                                    middleware.after_tool(state_ref, &tool_name_clone);
                                }
                                Ok::<Message, NodeRunError>(Message::tool(content.to_string(), id))
                            }
                            Err(error) => {
                                for middleware in &middlewares {
                                    middleware.on_tool_error(state_ref, &tool_name_clone, &error);
                                }
                                Err(error)
                            }
                        }
                    };

                    futures.push(fut);
                }

                let results = join_all(futures).await;
                let mut new_messages = Vec::new();

                for res in results {
                    let msg = res?;
                    new_messages.push(msg);
                }

                return Ok(MessageDiff {
                    new_messages,
                    llm_calls_delta: 0,
                });
            }
        }

        Err(NodeRunError::ToolRunError("no tool call".to_string()))
    }
}

// TODO: 实现反思节点
pub struct ReflectNode;

#[async_trait::async_trait]
impl Node<MessageState> for ReflectNode {
    async fn run(&self, _: &MessageState) -> Result<MessageDiff, NodeRunError> {
        Ok(MessageDiff {
            new_messages: vec![],
            llm_calls_delta: 0,
        })
    }
}

pub struct EndNode;

#[async_trait::async_trait]
impl Node<MessageState> for EndNode {
    async fn run(&self, _: &MessageState) -> Result<MessageDiff, NodeRunError> {
        Ok(MessageDiff {
            new_messages: Vec::new(),
            llm_calls_delta: 0,
        })
    }
}

fn route(state: &MessageState) -> InternedGraphLabel {
    if let Some(last_message) = state.messages.last() {
        if let Message::Assistant { tool_calls, .. } = last_message
            && tool_calls.is_some()
        {
            return AgentLabel::ToolExecutor.intern();
        } else {
            return BaseAgentLabel::End.intern();
        };
    }
    BaseAgentLabel::End.intern()
}

fn build_message_agent_graph<M>(
    model: &M,
    tools: &[DynTool],
    middlewares: &[DynAgentMiddleware],
) -> StateGraph<MessageState>
where
    M: LlmModel + Send + Sync + 'static,
{
    let mut tool_specs = Vec::new();
    let mut tool_map = HashMap::new();

    for tool in tools {
        let spec = tool.spec();
        let name = spec.function_name().to_string();
        tool_specs.push(spec);
        tool_map.insert(name, tool.clone());
    }

    let mut graph = StateGraph::<MessageState>::default();

    graph.add_node(
        AgentLabel::CallModel,
        LlmNode {
            model: model.clone(),
            tool_specs,
            middlewares: middlewares.iter().cloned().collect(),
        },
    );
    graph.add_node(
        AgentLabel::ToolExecutor,
        ToolNode {
            tools: tool_map,
            middlewares: middlewares.iter().cloned().collect(),
        },
    );
    graph.add_node(BaseAgentLabel::End, EndNode);

    graph.set_start(BaseAgentLabel::Start);
    graph.set_end(BaseAgentLabel::End);

    graph.add_node_edge(BaseAgentLabel::Start, AgentLabel::CallModel);
    graph.add_condition_edge(AgentLabel::CallModel, route);
    graph.add_node_edge(AgentLabel::ToolExecutor, AgentLabel::CallModel);

    graph
}

pub struct MessageGraphBuilder;

impl MessageGraphBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn build_react<M>(
        &self,
        model: &M,
        tools: &[DynTool],
        middlewares: &[DynAgentMiddleware],
    ) -> StateGraph<MessageState>
    where
        M: LlmModel + Send + Sync + 'static,
    {
        build_message_agent_graph(model, tools, middlewares)
    }
}

pub struct AgentConfig {
    pub max_steps: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
        }
    }
}

pub struct ReactAgent {
    tools: Vec<DynTool>,
    system_prompt: Option<String>,
    config: AgentConfig,
    graph: StateGraph<MessageState>,
    middlewares: Vec<DynAgentMiddleware>,
}

impl ReactAgent {
    pub fn create_agent<M>(model: M) -> Self
    where
        M: LlmModel + Send + Sync + 'static,
    {
        let tools = Vec::new();
        let graph = build_message_agent_graph(&model, &tools, &[]);

        Self {
            tools,
            system_prompt: None,
            config: AgentConfig::default(),
            graph,
            middlewares: Vec::new(),
        }
    }

    pub fn create_agent_with_tools<T, I, M>(model: M, tools: I) -> Self
    where
        M: LlmModel + Send + Sync + 'static,
        T: IntoDynTool,
        I: IntoIterator<Item = T>,
    {
        let tools_vec: Vec<DynTool> = tools.into_iter().map(|t| t.into_dyn_tool()).collect();
        let graph = build_message_agent_graph(&model, &tools_vec, &[]);

        Self {
            tools: tools_vec,
            system_prompt: None,
            config: AgentConfig::default(),
            graph,
            middlewares: Vec::new(),
        }
    }

    pub fn with_tools<T, I>(mut self, tools: I) -> Self
    where
        T: IntoDynTool,
        I: IntoIterator<Item = T>,
    {
        self.tools = tools.into_iter().map(|t| t.into_dyn_tool()).collect();
        self
    }

    pub fn with_tool<T>(mut self, tool: T) -> Self
    where
        T: IntoDynTool,
    {
        self.tools.push(tool.into_dyn_tool());
        self
    }

    pub fn with_system_prompt(mut self, system_prompt: String) -> Self {
        self.system_prompt = Some(system_prompt);
        self
    }

    pub async fn invoke(&self, message: Message) -> Result<MessageState, GraphRunnerError> {
        let mut messages = Vec::new();
        if let Some(system_prompt) = &self.system_prompt {
            messages.push(Message::system(system_prompt.clone()));
        }
        messages.push(message);
        let initial = MessageState::new(messages);

        for middleware in &self.middlewares {
            middleware.before_run(&initial);
        }
        let mut stepper = GraphStepper::new(&self.graph, initial.clone(), self.config.max_steps);
        let result = loop {
            match stepper.step().await {
                Ok(StepEvent::Finished { .. }) => break Ok(stepper.state().clone()),
                Ok(StepEvent::NodeEnd { .. }) => {}
                Err(error) => break Err(error),
            }
        };
        match &result {
            Ok(state) => {
                for middleware in &self.middlewares {
                    middleware.after_run(state);
                }
            }
            Err(error) => {
                for middleware in &self.middlewares {
                    middleware.on_run_error(&initial, error);
                }
            }
        }
        result
    }
}

pub struct ReactAgentBuilder<M> {
    model: M,
    tools: Vec<DynTool>,
    system_prompt: Option<String>,
    config: AgentConfig,
    middlewares: Vec<DynAgentMiddleware>,
}

impl<M> ReactAgentBuilder<M>
where
    M: LlmModel + Clone + Send + Sync + 'static,
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            tools: Vec::new(),
            system_prompt: None,
            config: AgentConfig::default(),
            middlewares: Vec::new(),
        }
    }

    pub fn with_tools<T, I>(mut self, tools: I) -> Self
    where
        T: IntoDynTool,
        I: IntoIterator<Item = T>,
    {
        self.tools = tools.into_iter().map(|t| t.into_dyn_tool()).collect();
        self
    }

    pub fn with_tool<T>(mut self, tool: T) -> Self
    where
        T: IntoDynTool,
    {
        self.tools.push(tool.into_dyn_tool());
        self
    }

    pub fn with_middleware(mut self, middleware: DynAgentMiddleware) -> Self {
        self.middlewares.push(middleware);
        self
    }

    pub fn with_middlewares<I>(mut self, middlewares: I) -> Self
    where
        I: IntoIterator<Item = DynAgentMiddleware>,
    {
        self.middlewares.extend(middlewares);
        self
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.config.max_steps = max_steps;
        self
    }

    pub fn build(self) -> ReactAgent {
        let builder = MessageGraphBuilder::new();
        let graph = builder.build_react(&self.model, &self.tools, &self.middlewares);

        ReactAgent {
            tools: self.tools,
            system_prompt: self.system_prompt,
            config: self.config,
            graph,
            middlewares: self.middlewares,
        }
    }
}
