# AGENTS.md

## Build & Verify

```bash
cargo build --workspace
cargo fmt --check --all          # CI runs this first
cargo clippy --workspace --tests -- -D warnings  # CI runs this second
cargo test --workspace --verbose  # CI runs this last
```

**CI order is fixed**: fmt → clippy → test. Run them in this order.

## Workspace Structure

```
crates/
├── langchain_core/           # Message, ChatModel trait, RegisteredTool, #[tool] macro
├── langchain/                # ReactAgent, LlmNode, ToolNode, examples/
├── langgraph/                # Graph, StateGraph, RunStrategy, #[derive(GraphLabel)]
├── langgraph/macro/          # proc-macro for GraphLabel
├── langchain_core/macro/     # proc-macro for tool
├── langchain_openai/          # ChatOpenAI implementation

└── langchain_tools/
```

## Required Conventions

- Use `to_owned()` not `to_string()` (enforced by `str_to_string` clippy lint)
- Tool functions: suffix with `_tool()`, params struct suffix with `Args`
- Tests: unit tests in `#[cfg(test)]`, integration tests with `#[tokio::test]`
- Mark API-key-requiring tests with `#[ignore]`

## Environment

- `OPENAI_API_KEY` required for examples and integration tests

## Architecture Notes

- `Graph::run_once` only computes successors; `RunStrategy` decides how to execute multiple branches
- State is immutable via `im::Vector`; `MessagesState` is append-only
- All I/O is async; streaming uses `Pin<Box<dyn Stream>>`
- Error types are generic throughout; users define their own error types
- `#[derive(GraphLabel)]` and `#[tool]` are proc-macros in separate crates

## Testing a Single Package

```bash
cargo test --package langchain test_name
cargo run --example agent_openai
```
