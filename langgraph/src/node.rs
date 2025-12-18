use crate::edge::{Edge, Route};
use async_trait::async_trait;
use bevy_ecs::{define_label, intern::Interned};
use downcast_rs::{Downcast, impl_downcast};
use langchain_core::state::State;
use std::fmt::Debug;
use thiserror::Error;
use variadics_please::all_tuples_with_size;

pub use graph_marcos::GraphLabel;

define_label!(
    #[diagnostic::on_unimplemented(
        note: "consider annotating `{Self}` with `#[derive(GraphLabel)]`"
    )]
    GraphLabel,
    GRAPH_LABEL
);

pub type InternedGraphLabel = Interned<dyn GraphLabel>;

pub trait IntoGraphNodeArray<const N: usize> {
    fn into_array(self) -> [InternedGraphLabel; N];
}

macro_rules! impl_graph_label_tuples {
    ($N: expr, $(#[$meta:meta])* $(($T: ident, $I: ident)),*) => {
        $(#[$meta])*
        impl<$($T: GraphLabel),*> IntoGraphNodeArray<$N> for ($($T,)*) {
            #[inline]
            fn into_array(self) -> [InternedGraphLabel; $N] {
                let ($($I,)*) = self;
                [$($I.intern(), )*]
            }
        }
    }
}

// 为长度 1~32 的 GraphLabel 元组生成 IntoGraphNodeArray<N> 实现
all_tuples_with_size!(impl_graph_label_tuples, 1, 32, T, l);

#[derive(Error, Debug)]
pub enum NodeRunError {
    #[error("node run error")]
    NodeRunError,
    #[error("tool run error: {0}")]
    ToolRunError(String),
    #[error("llm run error: {0}")]
    LlmRunError(#[source] Box<dyn std::error::Error + Send + Sync>),
}

#[async_trait]
pub trait Node<S: State>: Downcast + Send + Sync + 'static {
    async fn run(&self, state: &S) -> Result<S::Diff, NodeRunError>;
}

impl_downcast!(Node<S> where S: State);

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash, GraphLabel)]
pub enum BaseAgentLabel {
    #[default]
    Start,
    End,
}

pub struct NodeState<S: State> {
    pub label: InternedGraphLabel,
    /// The name of the type that implements [`Node`].
    pub type_name: &'static str,
    pub node: Box<dyn Node<S>>,
}

impl<S: State> Debug for NodeState<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "{:?} ({})", self.label, self.type_name)
    }
}

impl<S: State> NodeState<S> {
    pub fn new<T>(label: InternedGraphLabel, node: T) -> Self
    where
        T: Node<S>,
    {
        NodeState {
            label,
            node: Box::new(node),
            type_name: core::any::type_name::<T>(),
        }
    }
}

#[derive(Debug)]
pub enum EdgesA<S: State> {
    NodeEdge(Edge),
    ConditionEdge(Route<S>),
}
