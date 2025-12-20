use crate::label::InternedGraphLabel;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

pub trait BranchKind: Copy + Eq + Hash + Debug + Send + Sync + 'static {}

impl<T> BranchKind for T where T: Copy + Eq + Hash + Debug + Send + Sync + 'static {}

/// 条件边的条件函数，输入为边的输出，输出为后继节点标签列表
pub type EdgeCondition<O, B> = Box<dyn Fn(&O) -> Vec<B> + Send + Sync>;

pub enum Edge<O, B: BranchKind> {
    /// 普通边，直接连接两个节点
    NodeEdge(InternedGraphLabel),
    /// 条件边，根据条件判断是否执行下一个节点
    ConditionalEdge {
        /// 条件边的可能的输出节点列表，根据条件判断执行哪个节点
        next_nodes: HashMap<B, InternedGraphLabel>,
        condition: EdgeCondition<O, B>,
    },
}
