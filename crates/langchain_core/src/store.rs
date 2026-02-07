// Store 模块 - 跨线程数据存储抽象
//
// 这个模块提供了 BaseStore trait，用于在不同的节点和组件之间共享数据。
// 支持命名空间隔离、类型安全的数据存储和检索。

use std::{fmt::Display, str::FromStr};

pub mod memory;

pub use memory::InMemoryStore;

use async_trait::async_trait;

/// Store 操作错误
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("serialization error: {0}")]
    Serialization(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("deserialization error: {0}")]
    Deserialization(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("key not found: namespace={0}, key={1}")]
    NotFound(String, String),

    #[error("store backend error: {0}")]
    Backend(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// Store 命名空间配置
///
/// 命名空间用于隔离不同上下文的数据。例如：
/// - `["user", "123", "profile"]` - 用户123的配置文件数据
/// - `["thread", "456", "documents"]` - 线程456的文档数据
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Namespace {
    /// 命名空间路径，支持层级结构
    pub parts: Vec<String>,
}

impl Namespace {
    /// 创建新的命名空间
    pub fn new(parts: Vec<String>) -> Self {
        Self { parts }
    }

    /// 创建根命名空间
    pub fn root() -> Self {
        Self { parts: vec![] }
    }

    /// 创建子命名空间
    pub fn child(&self, part: impl Into<String>) -> Self {
        let mut parts = self.parts.clone();
        parts.push(part.into());
        Self { parts }
    }
}

impl FromStr for Namespace {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            parts: s.split(':').map(|s| s.to_owned()).collect(),
        })
    }
}

impl Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.parts.join(":"))
    }
}

/// Store 过滤条件
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreFilter {
    /// 键前缀匹配
    Prefix(String),
    /// 精确键匹配
    Exact(String),
    /// 范围查询 [start, end)
    Range { start: String, end: String },
}

/// Base Store trait - 跨线程数据存储抽象
///
/// 这个 trait 定义了跨线程、跨节点共享数据的接口。实现可以是内存存储、
/// 数据库存储或任何其他持久化存储。
///
/// # 示例
/// ```ignore
/// use langchain_core::store::{BaseStore, Namespace, InMemoryStore};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let store = InMemoryStore::new();
///
///     // 存储数据
///     let namespace = Namespace::from_str("user:123");
///     store.put(&namespace, "name", b"Alice").await?;
///
///     // 获取数据
///     let name = store.get(&namespace, "name").await?;
///     assert_eq!(name, Some(b"Alice".to_vec()));
///
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait BaseStore: Send + Sync {
    /// 存储数据到指定命名空间和键
    ///
    /// # 参数
    /// - `namespace`: 命名空间，用于隔离数据
    /// - `key`: 键名
    /// - `value`: 要存储的值（字节数组）
    ///
    /// # 错误
    /// - 如果序列化失败，返回 `StoreError::Serialization`
    /// - 如果后端存储失败，返回 `StoreError::Backend`
    async fn put(&self, namespace: &Namespace, key: &str, value: Vec<u8>)
    -> Result<(), StoreError>;

    /// 从指定命名空间和键获取数据
    ///
    /// # 参数
    /// - `namespace`: 命名空间
    /// - `key`: 键名
    ///
    /// # 返回
    /// - `Ok(Some(Vec<u8>))`: 如果键存在
    /// - `Ok(None)`: 如果键不存在
    /// - `Err(...)`: 如果反序列化失败或后端错误
    ///
    /// # 错误
    /// - 如果反序列化失败，返回 `StoreError::Deserialization`
    /// - 如果后端读取失败，返回 `StoreError::Backend`
    async fn get(&self, namespace: &Namespace, key: &str) -> Result<Option<Vec<u8>>, StoreError>;

    /// 删除指定命名空间和键的数据
    ///
    /// # 参数
    /// - `namespace`: 命名空间
    /// - `key`: 键名
    ///
    /// # 返回
    /// - `Ok(true)`: 键存在并被删除
    /// - `Ok(false)`: 键不存在
    /// - `Err(...)`: 如果后端删除失败
    async fn delete(&self, namespace: &Namespace, key: &str) -> Result<bool, StoreError>;

    /// 批量获取符合过滤条件的数据
    ///
    /// # 参数
    /// - `namespace`: 命名空间
    /// - `filter`: 过滤条件
    /// - `limit`: 可选的结果数量限制
    ///
    /// # 返回
    /// 符合条件的 (键, 值) 对列表
    ///
    /// # 示例
    /// ```ignore
    /// use langchain_core::store::{BaseStore, Namespace, StoreFilter, InMemoryStore};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let store = InMemoryStore::new();
    /// let namespace = Namespace::from_str("documents");
    ///
    /// // 列出所有以 "doc_" 开头的文档
    /// let docs = store
    ///     .list(&namespace, &StoreFilter::Prefix("doc_".to_string()), Some(10))
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn list(
        &self,
        namespace: &Namespace,
        filter: &StoreFilter,
        limit: Option<usize>,
    ) -> Result<Vec<(String, Vec<u8>)>, StoreError>;

    /// 检查键是否存在
    ///
    /// # 参数
    /// - `namespace`: 命名空间
    /// - `key`: 键名
    ///
    /// # 返回
    /// - `Ok(true)`: 键存在
    /// - `Ok(false)`: 键不存在
    /// - `Err(...)`: 如果后端检查失败
    async fn exists(&self, namespace: &Namespace, key: &str) -> Result<bool, StoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_from_str() {
        let ns = Namespace::from_str("user:123:profile").unwrap();
        assert_eq!(ns.parts, vec!["user", "123", "profile"]);
    }

    #[test]
    fn test_namespace_root() {
        let ns = Namespace::root();
        assert!(ns.parts.is_empty());
    }

    #[test]
    fn test_namespace_child() {
        let root = Namespace::root();
        let user = root.child("user");
        assert_eq!(user.parts, vec!["user"]);

        let profile = user.child("123");
        assert_eq!(profile.parts, vec!["user", "123"]);
    }

    #[test]
    fn test_namespace_to_string() {
        let ns = Namespace::from_str("user:123:profile").unwrap();
        assert_eq!(ns.to_string(), "user:123:profile");
    }

    #[test]
    fn test_namespace_equality() {
        let ns1 = Namespace::from_str("user:123");
        let ns2 = Namespace::from_str("user:123");
        assert_eq!(ns1, ns2);

        let ns3 = Namespace::from_str("user:456");
        assert_ne!(ns1, ns3);
    }
}
