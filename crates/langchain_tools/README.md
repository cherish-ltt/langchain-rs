# langchain-tools

LangChain 内置工具集合，提供常用的工具函数，可直接用于 Agent 执行任务。

## 特性

✅ **Web 搜索** - 使用 DuckDuckGo 进行网络搜索
✅ **文件操作** - 读取、写入、列表目录
✅ **实用工具** - 日期时间、计算器等
✅ **类型安全** - 基于 Rust 的强类型检查
✅ **自动 Schema** - 自动生成 JSON Schema
✅ **异步 API** - 完全基于 Tokio 异步运行时

## 安装

在 `Cargo.toml` 中添加：

```toml
[dependencies]
langchain-tools = "0.1"
```

## 可用工具

### Web 搜索

使用 DuckDuckGo API 进行网络搜索。

```rust
use langchain_tools::{search_web, SearchResult};

let results = search_web(
    "Rust programming language".to_string(),
    Some(5)  // 最多 5 个结果
).await?;

for result in results {
    println!("Title: {}", result.title);
    println!("URL: {}", result.url);
    println!("Snippet: {}", result.snippet);
}
```

### 文件操作

#### 读取文件

```rust
use langchain_tools::read_file;

let content = read_file("/path/to/file.txt".to_string()).await?;
```

#### 写入文件

```rust
use langchain_tools::write_file;

let result = write_file(
    "/path/to/file.txt".to_string(),
    "Hello, World!".to_string()
).await?;
```

#### 列出目录

```rust
use langchain_tools::{list_directory, FileInfo};

let files = list_directory("/path/to/dir".to_string()).await?;

for file in files {
    println!("{} ({})", file.name, if file.is_dir { "DIR" } else { "FILE" });
}
```

### 实用工具

#### 获取当前时间

```rust
use langchain_tools::get_current_time;

// ISO 8601 格式
let time = get_current_time(None).await?;

// 自定义格式
let time = get_current_time(Some("%Y-%m-%d %H:%M:%S".to_string())).await?;
```

#### 计算器

```rust
use langchain_tools::calculate;

let result = calculate(10.0, "+".to_string(), 5.0).await?;
assert_eq!(result, 15.0);

// 支持的运算符: +, -, *, /
```

## 在 Agent 中使用

直接使用工具函数（由 `#[tool]` 宏自动生成工具包装器）：

```rust
use langchain_tools::{search_web, calculate, read_file};
use langchain_core::tool;

// 工具函数可以直接使用
let results = search_web("Rust".to_string(), Some(5)).await?;
let result = calculate(10.0, "+".to_string(), 5.0).await?;

// 在 Agent 中，工具会通过 RegisteredTool<E> 包装使用
// 工具宏会自动生成相应的工具定义和 JSON Schema
```

## 运行示例

```bash
# 工具演示
cargo run --package langchain-tools --example tools_demo
```

## 工具列表

| 工具 | 描述 | 参数 |
|------|------|------|
| `search_web` | Web 搜索 | query (字符串), max_results (可选数字) |
| `read_file` | 读取文件 | path (文件路径字符串) |
| `write_file` | 写入文件 | path (文件路径字符串), content (内容字符串) |
| `list_directory` | 列出目录 | path (目录路径字符串) |
| `delete_file` | 删除文件 | path (文件路径字符串) |
| `create_directory` | 创建目录 | path (目录路径字符串), recursive (可选布尔) |
| `get_current_time` | 获取当前时间 | format (可选格式字符串) |
| `calculate` | 计算器 | a (数字), op (运算符), b (数字) |
| `eval_expression` | 表达式计算 | expression (数学表达式字符串) |

## 自定义工具

创建自定义工具很简单：

```rust
use langchain_core::tool;

#[tool(description = "My custom tool")]
async fn my_tool(
    #[arg(description = "Input parameter")] input: String,
) -> Result<String, MyError> {
    // 实现你的工具逻辑
    Ok(format!("Processed: {}", input))
}

#[derive(Debug, thiserror::Error)]
pub enum MyError {
    #[error("Tool error: {0}")]
    ToolError(String),
}

// 工具会自动生成：
// - my_tool() 函数返回 RegisteredTool<MyError>
// - JSON Schema
// - 参数验证
```

## 错误处理

每个工具模块都有自己的错误类型：

- `WebSearchError` - Web 搜索错误
- `FileToolError` - 文件操作错误
- `UtilError` - 实用工具错误

所有错误都实现了 `std::error::Error` 和 `Into<ToolError>`，可以轻松集成到 Agent 中。

## 运行测试

```bash
# 所有测试
cargo test --package langchain-tools

# 跳过需要网络的测试
cargo test --package langchain-tools -- --ignored
```

## 依赖项

- `tokio` - 异步运行时
- `reqwest` - HTTP 客户端（Web 搜索）
- `serde`/`serde_json` - 序列化
- `chrono` - 日期时间处理
- `thiserror` - 错误处理
- `urlencoding` - URL 编码

## 性能考虑

- 所有文件操作都是异步的，不会阻塞执行
- HTTP 请求使用连接池复用
- 表达式计算使用安全的解析器

## 安全注意事项

⚠️ **重要**: 文件操作工具可以访问文件系统中的任意路径。在生产环境中使用时，请考虑：

1. **路径验证** - 添加路径白名单或黑名单
2. **沙箱** - 使用 chroot 或容器隔离
3. **权限限制** - 以低权限用户运行
4. **审计日志** - 记录所有文件操作

## 许可证

MIT OR Apache-2.0
