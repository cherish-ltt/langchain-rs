// Human-in-the-Loop Agent Example
//
// 这个示例展示了如何在 Agent 工作流中使用 Human-in-the-Loop (HITL) 功能。
//
// 功能演示：
// 1. 创建带 HITL 节点的 Agent 工作流
// 2. 在执行过程中等待人类确认
// 3. 处理不同类型的中断（确认、输入、审核、选择）

use async_trait::async_trait;
use langchain_core::{
    message::{Content, Message},
    state::MessagesState,
};
use langgraph::{
    GraphLabel,
    checkpoint::{MemorySaver, RunnableConfig},
    hitl_node::HumanInTheLoopNode,
    interrupt::{InMemoryInterruptManager, InterruptError, InterruptManager, InterruptResponse},
    node::{Node, NodeContext},
    state_graph::{RunStrategy, StateGraph},
};
use std::sync::Arc;

// 自定义节点：模拟 LLM 节点
struct MockLlmNode {
    response: String,
}

#[async_trait]
impl Node<MessagesState, MessagesState, InterruptError, ()> for MockLlmNode {
    async fn run_sync(
        &self,
        _input: &MessagesState,
        _context: NodeContext<'_>,
    ) -> Result<MessagesState, InterruptError> {
        let mut delta = MessagesState::default();
        delta.push_message(Arc::new(Message::Assistant {
            content: self.response.clone(),
            tool_calls: None,
            name: None,
        }));
        Ok(delta)
    }

    async fn run_stream(
        &self,
        input: &MessagesState,
        _sink: &mut dyn langgraph::node::EventSink<()>,
        context: NodeContext<'_>,
    ) -> Result<MessagesState, InterruptError> {
        self.run_sync(input, context).await
    }
}

// 自定义节点：模拟工具执行
struct ToolExecutorNode {
    tool_name: String,
}

#[async_trait]
impl Node<MessagesState, MessagesState, InterruptError, ()> for ToolExecutorNode {
    async fn run_sync(
        &self,
        _input: &MessagesState,
        _context: NodeContext<'_>,
    ) -> Result<MessagesState, InterruptError> {
        let mut delta = MessagesState::default();
        delta.push_message(Arc::new(Message::Tool {
            content: format!("工具 {} 执行结果", self.tool_name),
            tool_call_id: "test_id".to_owned(),
        }));
        Ok(delta)
    }

    async fn run_stream(
        &self,
        input: &MessagesState,
        _sink: &mut dyn langgraph::node::EventSink<()>,
        context: NodeContext<'_>,
    ) -> Result<MessagesState, InterruptError> {
        self.run_sync(input, context).await
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 初始化日志
    tracing_subscriber::fmt::init();

    // 创建中断管理器
    let interrupt_manager = Arc::new(InMemoryInterruptManager::new());

    // 创建检查点保存器
    let checkpointer = MemorySaver::new();
    let config = RunnableConfig {
        thread_id: "hitl-demo-1".to_owned(),
    };

    // 创建自定义标签类型
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, langgraph::GraphLabel)]
    enum HitlLabel {
        Start,
        Confirm,
        Tool,
    }

    // 创建 StateGraph
    let mut graph: StateGraph<MessagesState, MessagesState, InterruptError, ()> = StateGraph::new(
        HitlLabel::Start,
        |state: MessagesState, update: MessagesState| {
            // Reducer: 合并消息列表
            let mut merged = state;
            for msg in update.messages {
                merged.push_message(msg);
            }
            merged
        },
    );

    // 添加节点
    graph.add_node(
        HitlLabel::Start,
        MockLlmNode {
            response: "我需要执行敏感操作，请确认".to_owned(),
        },
    );

    // 添加 HITL 确认节点
    graph.add_node(
        HitlLabel::Confirm,
        HumanInTheLoopNode::confirm(interrupt_manager.clone(), "是否允许执行敏感操作？"),
    );

    // 添加工具执行节点
    graph.add_node(
        HitlLabel::Tool,
        ToolExecutorNode {
            tool_name: "sensitive_operation".to_owned(),
        },
    );

    // 添加边
    graph.add_edge(HitlLabel::Start, HitlLabel::Confirm);
    graph.add_edge(HitlLabel::Confirm, HitlLabel::Tool);

    // 设置检查点
    let graph = graph.with_checkpointer(checkpointer);

    println!("=== Human-in-the-Loop Agent 示例 ===\n");

    // 在后台任务中处理中断
    let manager_respond = interrupt_manager.clone();
    let handle = tokio::spawn(async move {
        // 等待中断出现
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let pending = manager_respond
            .get_pending(&InMemoryInterruptManager::default_thread_id())
            .await?;

        if let Some(interrupt) = pending.first() {
            println!("🔔 检测到中断:");
            println!("   ID: {}", interrupt.id);
            println!("   提示: {}\n", interrupt.as_prompt());

            // 提交确认响应
            println!("✓ 用户确认操作\n");
            manager_respond
                .respond(&interrupt.id, InterruptResponse::Confirm)
                .await?;
        }

        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    });

    // 执行图
    println!("▶ 开始执行 Agent...\n");
    let initial_state = MessagesState::default();

    match graph
        .run(initial_state, Some(&config), 10, RunStrategy::PickFirst)
        .await
    {
        Ok((final_state, _)) => {
            println!("✅ Agent 执行完成!\n");
            println!("=== 最终消息 ===");
            for (i, msg) in final_state.messages.iter().enumerate() {
                match msg.as_ref() {
                    Message::User { content, .. } => {
                        if let Content::Text(text) = content {
                            println!("{}. User: {}", i + 1, text)
                        }
                    }
                    Message::Assistant { content, .. } => {
                        println!("{}. Assistant: {}", i + 1, content)
                    }
                    Message::Tool { content, .. } => {
                        println!("{}. Tool: {}", i + 1, content)
                    }
                    _ => {}
                }
            }
        }
        Err(e) => {
            println!("❌ 执行失败: {:?}", e);
        }
    }

    // 等待后台任务完成
    handle.await??;

    println!("\n=== 演示完成 ===");

    Ok(())
}
