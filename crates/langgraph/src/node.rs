use std::fmt::Debug;
use std::pin::Pin;

use async_trait::async_trait;
use downcast_rs::{Downcast, impl_downcast};
use futures::Stream;
use thiserror::Error;

use crate::{
    edge::{BranchKind, Edge},
    label::InternedGraphLabel,
};

#[derive(Debug, Error)]
pub enum NodeError {}

#[async_trait]
pub trait EventSink<Ev>: Send {
    async fn emit(&mut self, event: Ev);
}

/// 节点执行结果
///
/// # 类型参数
///
/// * `O` - 节点输出类型
/// * `Ev` - 事件类型
pub enum NodeResult<'a, O, Ev> {
    /// 同步调用时的输出。同步调用下和 非流式调用节点类型。
    Sync { output: O },
    /// 事件流Stream 模式下有类型
    Stream { events: EventStream<'a, Ev> },
}

impl<'a, O, Ev> NodeResult<'a, O, Ev> {
    /// 创建同步结果（无事件流）
    pub fn sync(output: O) -> Self {
        Self::Sync { output }
    }

    /// 创建流式结果
    pub fn stream(events: impl Stream<Item = Ev> + Send + 'a) -> Self {
        Self::Stream {
            events: EventStream::new(events),
        }
    }
}

/// 事件流类型
pub struct EventStream<'a, Ev> {
    inner: Pin<Box<dyn Stream<Item = Ev> + Send + 'a>>,
}

impl<'a, Ev> EventStream<'a, Ev> {
    pub fn new(stream: impl Stream<Item = Ev> + Send + 'a) -> Self {
        Self {
            inner: Box::pin(stream),
        }
    }
}

impl<'a, Ev> Stream for EventStream<'a, Ev> {
    type Item = Ev;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// 节点 trait，定义了节点的运行行为
///
/// # 类型参数
///
/// * `I` - 节点输入类型
/// * `O` - 节点输出类型
/// * `E` - 节点错误类型
/// * `Ev` - 事件类型
#[async_trait]
pub trait Node<I, O, E, Ev>: Downcast + Send + Sync + 'static {
    async fn run_sync(&self, input: &I) -> Result<O, E>;

    async fn run_stream(&self, input: &I, sink: &mut dyn EventSink<Ev>) -> Result<O, E>;
}

impl_downcast!(Node<I, O, E, Ev>);

/// 节点状态结构体，包含节点的标签、类型名称等元数据和节点实例
///
/// # 类型参数
///
/// * `I` - 节点输入类型
/// * `O` - 节点输出类型
/// * `E` - 节点错误类型
/// * `B` - 分支类型，必须实现 `BranchKind` trait
/// * `Ev` - 事件类型
pub struct NodeState<I, O, E, B: BranchKind, Ev: Debug> {
    pub label: InternedGraphLabel,
    /// 节点类型名称，用于调试和日志记录
    pub type_name: &'static str,
    pub node: Box<dyn Node<I, O, E, Ev>>,
    pub edges: Vec<Edge<O, B>>,
}

impl<I, O, E, B: BranchKind, Ev: Debug> NodeState<I, O, E, B, Ev> {
    pub fn new<T>(label: InternedGraphLabel, node: T) -> Self
    where
        T: Node<I, O, E, Ev>,
    {
        Self {
            label,
            type_name: core::any::type_name::<T>(),
            node: Box::new(node),
            edges: Vec::new(),
        }
    }
}

impl<I, O, E, B: BranchKind, Ev: Debug> Debug for NodeState<I, O, E, B, Ev> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:?} ({})", self.label, self.type_name)
    }
}
