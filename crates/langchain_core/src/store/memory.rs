// 内存存储实现
//
// 提供基于内存的 BaseStore 实现，适用于开发、测试和简单的生产场景。

use crate::store::{BaseStore, Namespace, StoreError, StoreFilter};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 存储条目
#[derive(Debug, Clone)]
struct StoreEntry {
    /// 字节数组值
    value: Vec<u8>,
    /// 创建时间戳（Unix 时间戳，秒）
    created_at: u64,
    /// 更新时间戳（Unix 时间戳，秒）
    _updated_at: u64,
}

/// 内存存储实现
///
/// 这是一个使用 HashMap 和 RwLock 实现的简单内存存储。
/// 适用于：
/// - 开发和测试环境
/// - 单进程应用
/// - 缓存场景
///
/// 对于生产环境，建议使用持久化存储（如数据库）。
///
/// # 示例
/// ```ignore
/// use langchain_core::store::{BaseStore, Namespace, InMemoryStore};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let store = InMemoryStore::new();
///
///     // 存储数据（需要手动序列化为字节）
///     let data = b"Alice".to_vec();
///     let namespace = Namespace::from_str("user:123");
///     store.put(&namespace, "name", data).await?;
///
///     // 获取数据
///     let bytes = store.get(&namespace, "name").await?;
///     assert_eq!(bytes, Some(b"Alice".to_vec()));
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Default, Clone)]
pub struct InMemoryStore {
    /// 存储结构: (namespace_string, key) -> StoreEntry
    storage: Arc<RwLock<HashMap<(String, String), StoreEntry>>>,
}

impl InMemoryStore {
    /// 创建新的内存存储实例
    pub fn new() -> Self {
        Self::default()
    }

    /// 将命名空间转换为字符串
    fn namespace_to_string(ns: &Namespace) -> String {
        ns.to_string()
    }
}

#[async_trait]
impl BaseStore for InMemoryStore {
    async fn put(
        &self,
        namespace: &Namespace,
        key: &str,
        value: Vec<u8>,
    ) -> Result<(), StoreError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let ns_key = Self::namespace_to_string(namespace);
        let mut storage = self.storage.write().await;

        // 检查是否已存在，如果存在则保留 created_at
        let created_at = if let Some(existing) = storage.get(&(ns_key.clone(), key.to_owned())) {
            existing.created_at
        } else {
            now
        };

        storage.insert(
            (ns_key, key.to_owned()),
            StoreEntry {
                value,
                created_at,
                _updated_at: now,
            },
        );

        Ok(())
    }

    async fn get(&self, namespace: &Namespace, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let ns_key = Self::namespace_to_string(namespace);
        let storage = self.storage.read().await;

        match storage.get(&(ns_key, key.to_owned())) {
            Some(entry) => Ok(Some(entry.value.clone())),
            None => Ok(None),
        }
    }

    async fn delete(&self, namespace: &Namespace, key: &str) -> Result<bool, StoreError> {
        let ns_key = Self::namespace_to_string(namespace);
        let mut storage = self.storage.write().await;

        Ok(storage.remove(&(ns_key, key.to_owned())).is_some())
    }

    async fn list(
        &self,
        namespace: &Namespace,
        filter: &StoreFilter,
        limit: Option<usize>,
    ) -> Result<Vec<(String, Vec<u8>)>, StoreError> {
        let ns_key = Self::namespace_to_string(namespace);
        let storage = self.storage.read().await;

        let mut results = Vec::new();

        for ((ns, key), entry) in storage.iter() {
            // 过滤命名空间
            if ns != &ns_key {
                continue;
            }

            // 应用过滤条件
            match filter {
                StoreFilter::Prefix(prefix) if !key.starts_with(prefix) => continue,
                StoreFilter::Exact(exact) if key != exact => continue,
                StoreFilter::Range { start, end } if !(key >= start && key < end) => continue,
                _ => {}
            }

            results.push((key.clone(), entry.value.clone()));

            // 检查限制
            if let Some(limit) = limit
                && results.len() >= limit
            {
                break;
            }
        }

        Ok(results)
    }

    async fn exists(&self, namespace: &Namespace, key: &str) -> Result<bool, StoreError> {
        let ns_key = Self::namespace_to_string(namespace);
        let storage = self.storage.read().await;
        Ok(storage.contains_key(&(ns_key, key.to_owned())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[tokio::test]
    async fn test_store_put_get() {
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        // 存储数据
        store
            .put(&namespace, "key1", b"value1".to_vec())
            .await
            .unwrap();

        // 获取数据
        let value = store.get(&namespace, "key1").await.unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));
    }

    #[tokio::test]
    async fn test_store_get_nonexistent() {
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        let value = store.get(&namespace, "nonexistent").await.unwrap();
        assert!(value.is_none());
    }

    #[tokio::test]
    async fn test_store_delete() {
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        store
            .put(&namespace, "key1", b"value1".to_vec())
            .await
            .unwrap();

        // 删除存在的键
        let deleted = store.delete(&namespace, "key1").await.unwrap();
        assert!(deleted);

        // 验证已删除
        let value = store.get(&namespace, "key1").await.unwrap();
        assert!(value.is_none());

        // 删除不存在的键
        let deleted = store.delete(&namespace, "key1").await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_store_exists() {
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        store
            .put(&namespace, "key1", b"value1".to_vec())
            .await
            .unwrap();

        assert!(store.exists(&namespace, "key1").await.unwrap());
        assert!(!store.exists(&namespace, "nonexistent").await.unwrap());
    }

    #[tokio::test]
    async fn test_store_list_prefix() {
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        // 存储多个键
        for i in 1..=5 {
            store
                .put(
                    &namespace,
                    &format!("doc_{}", i),
                    format!("value_{}", i).into_bytes(),
                )
                .await
                .unwrap();
        }

        // 列出所有以 "doc_" 开头的键
        let results = store
            .list(&namespace, &StoreFilter::Prefix("doc_".to_owned()), None)
            .await
            .unwrap();

        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|(k, _)| k.starts_with("doc_")));
    }

    #[tokio::test]
    async fn test_store_list_with_limit() {
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        // 存储10个键
        for i in 1i32..=10 {
            store
                .put(&namespace, &format!("key_{}", i), i.to_be_bytes().to_vec())
                .await
                .unwrap();
        }

        // 只获取前5个
        let results = store
            .list(&namespace, &StoreFilter::Prefix("key_".to_owned()), Some(5))
            .await
            .unwrap();

        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn test_store_list_exact() {
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        store
            .put(&namespace, "key1", b"value1".to_vec())
            .await
            .unwrap();
        store
            .put(&namespace, "key2", b"value2".to_vec())
            .await
            .unwrap();

        let results = store
            .list(&namespace, &StoreFilter::Exact("key1".to_owned()), None)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "key1");
        assert_eq!(results[0].1, b"value1");
    }

    #[tokio::test]
    async fn test_store_list_range() {
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        store
            .put(&namespace, "a", b"value_a".to_vec())
            .await
            .unwrap();
        store
            .put(&namespace, "b", b"value_b".to_vec())
            .await
            .unwrap();
        store
            .put(&namespace, "c", b"value_c".to_vec())
            .await
            .unwrap();
        store
            .put(&namespace, "d", b"value_d".to_vec())
            .await
            .unwrap();

        let mut results = store
            .list(
                &namespace,
                &StoreFilter::Range {
                    start: "b".to_owned(),
                    end: "d".to_owned(),
                },
                None,
            )
            .await
            .unwrap();

        // HashMap 迭代顺序不确定，需要排序
        results.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "b");
        assert_eq!(results[1].0, "c");
    }

    #[tokio::test]
    async fn test_store_namespace_isolation() {
        let store = InMemoryStore::new();

        let ns1 = Namespace::from_str("user:123").unwrap();
        let ns2 = Namespace::from_str("user:456").unwrap();

        store.put(&ns1, "name", b"Alice".to_vec()).await.unwrap();
        store.put(&ns2, "name", b"Bob".to_vec()).await.unwrap();

        let name1 = store.get(&ns1, "name").await.unwrap();
        let name2 = store.get(&ns2, "name").await.unwrap();

        assert_eq!(name1, Some(b"Alice".to_vec()));
        assert_eq!(name2, Some(b"Bob".to_vec()));
    }

    #[tokio::test]
    async fn test_store_update_preserves_created_at() {
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        // 存储初始值
        store
            .put(&namespace, "key1", b"value1".to_vec())
            .await
            .unwrap();

        // 获取创建时间
        let storage = store.storage.read().await;
        let created_at = storage
            .get(&(namespace.to_string(), "key1".to_owned()))
            .unwrap()
            .created_at;
        drop(storage);

        // 等待确保时间戳变化
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // 更新值
        store
            .put(&namespace, "key1", b"value2".to_vec())
            .await
            .unwrap();

        // 验证创建时间不变
        let storage = store.storage.read().await;
        let entry = storage
            .get(&(namespace.to_string(), "key1".to_owned()))
            .unwrap();
        assert_eq!(entry.created_at, created_at);
        assert!(entry._updated_at >= created_at);
    }

    #[tokio::test]
    async fn test_store_with_serialized_data() {
        // 演示如何使用 serde 序列化复杂数据
        let store = InMemoryStore::new();
        let namespace = Namespace::from_str("test").unwrap();

        // 序列化复杂数据
        let data = serde_json::to_vec(&vec!["item1", "item2", "item3"]).unwrap();

        // 存储序列化后的字节
        store.put(&namespace, "items", data.clone()).await.unwrap();

        // 获取并反序列化
        let retrieved = store.get(&namespace, "items").await.unwrap();
        assert_eq!(retrieved, Some(data));

        // 反序列化
        let items: Vec<String> = serde_json::from_slice(&retrieved.unwrap()).unwrap();
        assert_eq!(items, vec!["item1", "item2", "item3"]);
    }
}
