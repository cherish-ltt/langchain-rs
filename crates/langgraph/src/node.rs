use std::fmt::Debug;

use async_trait::async_trait;
use downcast_rs::{Downcast, impl_downcast};
use thiserror::Error;

use crate::{
    edge::{BranchKind, Edge},
    label::InternedGraphLabel,
};

#[derive(Debug, Error)]
pub enum NodeError {}

/// 节点 trait，定义了节点的运行行为
#[async_trait]
pub trait Node<I, O, E>: Downcast + Send + Sync + 'static {
    async fn run(&self, input: &I) -> Result<O, E>;
}

impl_downcast!(Node<I, O, E>);

/// 节点状态结构体，包含节点的标签、类型名称等元数据和节点实例
///
/// # 类型参数
///
/// * `I` - 节点输入类型
/// * `O` - 节点输出类型
/// * `E` - 节点错误类型
/// * `B` - 分支类型，必须实现 `BranchKind` trait
pub struct NodeState<I, O, E, B: BranchKind> {
    pub label: InternedGraphLabel,
    /// 节点类型名称，用于调试和日志记录
    pub type_name: &'static str,
    pub node: Box<dyn Node<I, O, E>>,
    pub edges: Vec<Edge<O, B>>,
}

impl<I, O, E, B: BranchKind> NodeState<I, O, E, B> {
    pub fn new<T>(label: InternedGraphLabel, node: T) -> Self
    where
        T: Node<I, O, E>,
    {
        Self {
            label,
            type_name: core::any::type_name::<T>(),
            node: Box::new(node),
            edges: Vec::new(),
        }
    }
}

impl<I, O, E, B: BranchKind> Debug for NodeState<I, O, E, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:?} ({})", self.label, self.type_name)
    }
}
