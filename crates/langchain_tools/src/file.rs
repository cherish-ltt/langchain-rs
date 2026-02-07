//! 文件操作工具
//!
//! 提供文件读取、写入和目录列表功能。

use langchain_core::tool;
use thiserror::Error;

/// 文件操作错误
#[derive(Debug, Error)]
pub enum FileToolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Path not found: {0}")]
    PathNotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),
}

/// 文件信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct FileInfo {
    pub name: String,
    pub is_dir: bool,
    pub size: Option<u64>,
}

/// 读取文件内容
#[tool(
    description = "Read the contents of a text file",
    args(path = "File path to read")
)]
pub async fn read_file(path: String) -> Result<String, FileToolError> {
    tracing::debug!("Reading file: {}", path);

    let content = tokio::fs::read_to_string(&path).await?;

    Ok(content)
}

/// 写入内容到文件
#[tool(
    description = "Write content to a file",
    args(path = "File path to write", content = "Content to write")
)]
pub async fn write_file(path: String, content: String) -> Result<String, FileToolError> {
    tracing::debug!("Writing to file: {} ({} bytes)", path, content.len());

    tokio::fs::write(&path, &content).await?;

    Ok(format!(
        "Successfully wrote {} bytes to {}",
        content.len(),
        path
    ))
}

/// 列出目录内容
#[tool(
    description = "List files in a directory",
    args(path = "Directory path")
)]
pub async fn list_directory(path: String) -> Result<Vec<FileInfo>, FileToolError> {
    tracing::debug!("Listing directory: {}", path);

    let mut entries = tokio::fs::read_dir(&path).await?;
    let mut result = Vec::new();

    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name().to_string_lossy().to_string();
        let file_type = entry.file_type().await?;
        let metadata = entry.metadata().await.ok();

        result.push(FileInfo {
            name,
            is_dir: file_type.is_dir(),
            size: metadata.map(|m| m.len()),
        });
    }

    result.sort_by(|a, b| {
        // 目录排在前面
        match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        }
    });

    Ok(result)
}

/// 删除文件
#[tool(description = "Delete a file", args(path = "File path to delete"))]
pub async fn delete_file(path: String) -> Result<String, FileToolError> {
    tracing::debug!("Deleting file: {}", path);

    tokio::fs::remove_file(&path).await?;

    Ok(format!("Successfully deleted: {}", path))
}

/// 创建目录
#[tool(
    description = "Create a directory",
    args(
        path = "Directory path to create",
        recursive = "Create parent directories if needed"
    )
)]
pub async fn create_directory(
    path: String,
    recursive: Option<bool>,
) -> Result<String, FileToolError> {
    tracing::debug!("Creating directory: {}", path);

    if recursive.unwrap_or(false) {
        tokio::fs::create_dir_all(&path).await?;
    } else {
        tokio::fs::create_dir(&path).await?;
    }

    Ok(format!("Successfully created directory: {}", path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_file_operations() -> anyhow::Result<()> {
        // 创建临时目录
        let temp_dir = std::env::temp_dir().join("langchain_tools_test");
        tokio::fs::create_dir_all(&temp_dir).await?;

        // 写入文件
        let test_file = temp_dir.join("test.txt");
        let content = "Hello, World!";
        let result =
            write_file(test_file.to_string_lossy().to_string(), content.to_string()).await?;
        assert!(result.contains("Successfully wrote"));

        // 读取文件
        let read_content = read_file(test_file.to_string_lossy().to_string()).await?;
        assert_eq!(read_content, content);

        // 列出目录
        let entries = list_directory(temp_dir.to_string_lossy().to_string()).await?;
        assert!(entries.iter().any(|e| e.name == "test.txt"));

        // 删除文件
        let result = delete_file(test_file.to_string_lossy().to_string()).await?;
        assert!(result.contains("Successfully deleted"));

        // 清理
        tokio::fs::remove_dir_all(temp_dir).await?;

        Ok(())
    }
}
