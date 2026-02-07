//! LangChain 内置工具集合
//!
//! 此 crate 提供了一系列常用的内置工具，可直接用于 Agent 执行任务。
//!
//! # 特性
//!
//! - ✅ Web 搜索（DuckDuckGo）
//! - ✅ 文件操作（读、写、列目录）
//! - ✅ 实用工具（日期、计算等）
//! - ✅ 类型安全的工具定义
//! - ✅ 自动 JSON Schema 生成
//! - ✅ 异步 API
//!
//! # 使用示例
//!
//! ```no_run
//! use langchain_tools::{search_web, read_file, write_file};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Web 搜索
//! let results = search_web("Rust programming".to_string(), Some(5)).await?;
//!
//! // 文件操作
//! let content = read_file("example.txt".to_string()).await?;
//! write_file("output.txt".to_string(), content).await?;
//! # Ok(())
//! # }
//! ```

pub mod web;
pub mod file;
pub mod util;

// 重新导出常用工具和类型
pub use web::{search_web, SearchResult, WebSearchError};
pub use file::{read_file, write_file, list_directory, delete_file, create_directory, FileInfo, FileToolError};
pub use util::{get_current_time, calculate, eval_expression, UtilError};
