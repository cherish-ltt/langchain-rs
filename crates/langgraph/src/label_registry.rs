//! 全局标签注册表
//!
//! 提供 String <-> InternedGraphLabel 的双向快速查找，将 O(n) 的标签恢复操作优化为 O(1)。

use crate::label::InternedGraphLabel;
use std::collections::HashMap;
use std::sync::RwLock;

/// 标签注册表，提供双向映射
pub struct LabelRegistry {
    /// String -> InternedGraphLabel
    forward: HashMap<String, InternedGraphLabel>,
    /// InternedGraphLabel -> String
    reverse: HashMap<InternedGraphLabel, String>,
}

impl LabelRegistry {
    /// 创建新的标签注册表
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// 注册一个标签
    pub fn register(&mut self, label: InternedGraphLabel) {
        let s = label.as_str().to_owned();
        self.reverse.insert(label, s.clone());
        self.forward.insert(s, label);
    }

    /// 通过字符串获取 InternedGraphLabel（O(1)）
    pub fn get_interned(&self, s: &str) -> Option<InternedGraphLabel> {
        self.forward.get(s).copied()
    }

    /// 通过 InternedGraphLabel 获取字符串（O(1)）
    pub fn get_string(&self, label: InternedGraphLabel) -> Option<&str> {
        self.reverse.get(&label).map(|s| s.as_str())
    }

    /// 获取已注册的标签数量
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }
}

impl Default for LabelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    static GLOBAL_REGISTRY: RwLock<LabelRegistry> = RwLock::new(LabelRegistry::new());
}

/// 注册标签到全局注册表
///
/// 应在图构建期间为每个节点标签调用此函数
pub fn register_label(label: InternedGraphLabel) {
    GLOBAL_REGISTRY.with(|registry| {
        let mut reg = registry.write().unwrap();
        reg.register(label);
    });
}

/// 通过字符串快速查找 InternedGraphLabel
///
/// 用于 checkpoint 恢复时的标签查找，时间复杂度 O(1)
pub fn str_to_label(s: &str) -> Option<InternedGraphLabel> {
    GLOBAL_REGISTRY.with(|registry| {
        let reg = registry.read().unwrap();
        reg.get_interned(s)
    })
}

/// 通过 InternedGraphLabel 快速查找字符串
pub fn label_to_str(label: InternedGraphLabel) -> Option<String> {
    GLOBAL_REGISTRY.with(|registry| {
        let reg = registry.read().unwrap();
        reg.get_string(label).map(|s| s.to_owned())
    })
}

/// 批量注册标签
pub fn register_labels(labels: impl IntoIterator<Item = InternedGraphLabel>) {
    GLOBAL_REGISTRY.with(|registry| {
        let mut reg = registry.write().unwrap();
        for label in labels {
            reg.register(label);
        }
    });
}

/// 检查全局注册表中是否包含某个字符串标签
pub fn contains_label(s: &str) -> bool {
    GLOBAL_REGISTRY.with(|registry| {
        let reg = registry.read().unwrap();
        reg.forward.contains_key(s)
    })
}

/// 获取全局注册表中的标签数量
pub fn registered_count() -> usize {
    GLOBAL_REGISTRY.with(|registry| {
        let reg = registry.read().unwrap();
        reg.len()
    })
}

/// 清空全局注册表
///
/// ⚠️ 警告：这会清空所有已注册的标签，仅在测试中使用
pub fn clear_registry() {
    GLOBAL_REGISTRY.with(|registry| {
        let mut reg = registry.write().unwrap();
        reg.forward.clear();
        reg.reverse.clear();
    });
}

#[cfg(test)]
mod tests {
    use crate::label::GraphLabel;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, GraphLabel)]
    struct NodeA;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, GraphLabel)]
    struct NodeB;

    #[test]
    fn test_registry_basic() {
        let mut registry = LabelRegistry::new();

        let label1 = NodeA.intern();
        let label2 = NodeB.intern();

        registry.register(label1);
        registry.register(label2);

        assert_eq!(registry.get_interned("NodeA"), Some(label1));
        assert_eq!(registry.get_interned("NodeB"), Some(label2));
        assert_eq!(registry.get_interned("NodeC"), None);

        assert_eq!(registry.get_string(label1), Some("NodeA"));
        assert_eq!(registry.get_string(label2), Some("NodeB"));
    }

    #[test]
    fn test_global_registry() {
        clear_registry();

        let label1 = NodeA.intern();
        let label2 = NodeB.intern();

        register_label(label1);
        register_label(label2);

        assert_eq!(registered_count(), 2);
        assert!(contains_label("NodeA"));
        assert!(contains_label("NodeB"));

        assert_eq!(str_to_label("NodeA"), Some(label1));
        assert_eq!(str_to_label("NodeB"), Some(label2));

        clear_registry();
        assert_eq!(registered_count(), 0);
    }

    #[test]
    fn test_batch_register() {
        clear_registry();

        let labels = vec![NodeA.intern(), NodeB.intern()];

        register_labels(labels);

        assert_eq!(registered_count(), 2);
        assert!(contains_label("NodeA"));
        assert!(contains_label("NodeB"));

        clear_registry();
    }

    #[test]
    fn test_round_trip() {
        clear_registry();

        let original = NodeA.intern();
        register_label(original);

        // String -> Label -> String
        let label = str_to_label("NodeA").unwrap();
        let str_back = label_to_str(label).unwrap();

        assert_eq!(str_back, "NodeA");

        clear_registry();
    }
}
