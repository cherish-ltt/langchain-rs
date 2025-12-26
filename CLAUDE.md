# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**langchain-rs** 是 LangChain 和 LangGraph 概念的 Rust 实现 - 一个高性能、类型安全的框架，用于构建基于 LLM 的有状态智能体和工作流。

这是一个使用 Rust 2024 版本的 Cargo workspace 项目，采用异步优先设计（tokio）。

## 工作空间结构

```
crates/
├── langchain_core/       # 核心类型、trait 和抽象
├── langgraph/            # 图执行引擎
├── langchain/            # 高层 Agent 框架（ReAct Agent、节点）
├── langchain_openai/     # OpenAI 兼容的 HTTP 客户端
└── [proc-macro crates]   # langgraph_macro, langchain_core_macro
```

## 常用命令

```bash
# 构建工作空间
cargo build --workspace

# 格式检查（CI 中强制执行）
cargo fmt --check --all

# 代码检查（警告视为错误，CI 中强制执行）
cargo clippy --workspace --tests -- -D warnings

# 运行所有测试
cargo test --workspace --verbose

# 运行特定测试
cargo test --package langchain test_name

# 运行示例
cargo run --example agent_openai
cargo run --example agent_openai_stream

# 检查文档
cargo doc --workspace --no-deps --open
```

## 架构设计

### 分层设计

1. **langchain_core** - 基础层，包含：
   - `Message` 类型（User、Assistant、System、Tool、Developer）
   - `ChatModel` trait 用于 LLM 实现
   - `MessagesState` - 仅追加的消息列表，带 LLM 调用跟踪
   - `RegisteredTool<E>` - 带有 JSON schema 生成的工具包装器
   - OpenAI 兼容的请求/响应结构

2. **langgraph** - 图执行运行时：
   - **核心哲学**：最小化图引擎，只计算"有哪些后继节点"但**不决定"如何执行它们"**
   - `Graph<I, O, E, B, Ev>` - 通用图，带节点/边管理
   - `StateGraph<S, E, B, Ev>` - 简化的 API，其中 input=output=state
   - `RunStrategy` 枚举用于处理多个后继（StopAtNonLinear、PickFirst、PickLast、Parallel）
   - `GraphLabel` trait，使用动态标签系统和内化（interning）
   - `Node<I, O, E, Ev>` trait 用于节点操作（同步 + 流式）

3. **langchain** - 面向用户的框架：
   - `ReactAgent` - ReAct（推理 + 行动）Agent 实现
   - `LlmNode<M, E>` - 包装 ChatModel 用于图执行
   - `ToolNode<E>` - 执行工具调用（通过 try_join_all 并行执行）
   - Agent 图模式：Start → Llm → [Tool → Llm]* → End

4. **langchain_openai** - OpenAI 集成：
   - `ChatOpenAI` 实现 `ChatModel` trait
   - Builder 模式用于配置（base_url、model、api_key、temperature 等）
   - SSE（Server-Sent Events）流式支持

### 核心架构模式

**标签系统**：使用 `GraphLabel` trait 的动态类型擦除节点标签，`DynEq`/`DynHash` 用于运行时相等性比较，全局内化用于去重。使用 `#[derive(GraphLabel)]` 过程宏。

**执行策略**：图层（`Graph::run_once`）只计算可达的后继节点。如何执行多个后继节点委托给上层（`StateGraph`）通过 `RunStrategy` 决策。这允许支持并行执行、路由、竞态条件或自定义语义。

**类型安全**：整个代码库对错误类型泛型化。用户可以自定义错误类型。Trait 使用关联类型和约束以保持灵活性。

**流式处理**：所有操作同时支持同步和流式模式。使用 `Pin<Box<dyn Stream>>` 配合 `EventStream<'a, Ev>` 包装器。HTTP 流式使用 SSE 解析。

**状态管理**：使用 `im::Vector`（持久化数据结构）的不可变状态。消息是仅追加的。状态通过节点传递：每个节点接收状态，返回新状态。

**工具定义**：`#[tool]` 属性宏生成：
- 带有 serde/schemars derives 的 Args 结构体
- 返回 `RegisteredTool<E>` 的 `*_tool()` 函数
- 自动 JSON schema 生成
- 支持：自定义错误类型、参数描述、Result<T, E> 或无错误的 T 返回

### 图执行流程

1. `StateGraph::run()` 或 `stream()` 启动执行
2. `run_once()` 执行当前节点，计算后继节点
3. 如果 `next.len() == 1`：自动推进到线性后继节点
4. 如果 `next.len() != 1`：根据 `RunStrategy` 决策
   - `StopAtNonLinear`：分叉时停止
   - `PickFirst`/`PickLast`：选择单个分支
   - `Parallel`：执行所有（当前简化为 PickFirst）
5. 重复直到无后继节点或达到 max_steps

## 编码规范

**命名约定**：始终使用 `to_owned()` 而非 `to_string()`（由 clippy lint `str_to_string` 强制执行）。工具生成函数使用 `_tool()` 后缀。工具参数结构体使用 `Args` 后缀。

**错误处理**：使用 `thiserror` derive 宏。为错误转换实现 `From`。使用 `#[error(transparent)]` 处理包装错误。

**异步模式**：所有 I/O 操作都是异步的。使用 `Pin<Box<dyn Stream>>` 返回流，配合生命周期标注 `EventStream<'a, Ev>`。

**过程宏**：`#[derive(GraphLabel)]` 和 `#[tool]` 都在 `crates/*/macro/` 中实现为单独的 crate。使用 `darling` 进行属性解析。

**文档约定**：实现代码使用中文注释，公共 API 和文档注释使用英文。文档测试中使用示例。

**测试规范**：单元测试在 `#[cfg(test)]` 模块中。集成测试使用 `#[tokio::test]`。使用 `#[ignore]` 标记需要外部服务（API 密钥）的测试。

## 环境变量

- `OPENAI_API_KEY` - 运行示例和 OpenAI 集成测试所需

## 依赖概览

- **异步**：tokio 1.48.0、async-trait、async-stream、futures-util
- **HTTP**：reqwest 0.12.24
- **序列化**：serde、serde_json、schemars
- **类型系统**：downcast-rs、thiserror、anyhow
- **数据结构**：im（不可变集合）
- **过程宏**：syn、proc-macro2、quote、darling、variadics_please
- **日志**：tracing、tracing-subscriber

## 关键设计决策

1. **错误类型泛型化**：允许用户自定义错误类型，而非强制使用特定的错误层次结构。
2. **标签内化**：全局内化减少内存使用并启用节点标签的快速比较。
3. **分离同步/流式路径**：针对两种用例优化，而非强制所有操作都经过流式处理。
4. **OpenAI 标准**：遵循事实标准，实现最大兼容性的消息格式和 API 结构。
5. **最小化核心**：图层只提供后继节点计算；执行语义通过 StateGraph 和 RunStrategy 分层实现。
