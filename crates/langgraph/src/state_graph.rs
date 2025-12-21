use std::collections::HashMap;

use crate::{
    edge::BranchKind,
    graph::{Graph, GraphStepError},
    label::{GraphLabel, InternedGraphLabel},
    node::Node,
};

pub struct StateGraph<S, E, B: BranchKind> {
    pub graph: Graph<S, S, E, B>,
    pub entry: InternedGraphLabel,
}

impl<S, E, B: BranchKind> StateGraph<S, E, B> {
    pub fn new(entry: InternedGraphLabel, graph: Graph<S, S, E, B>) -> Self {
        Self { graph, entry }
    }

    pub fn set_entry(&mut self, entry: impl GraphLabel) {
        self.entry = entry.intern();
    }

    pub fn add_node<T>(&mut self, label: impl GraphLabel, node: T)
    where
        T: Node<S, S, E>,
    {
        self.graph.add_node(label, node);
    }

    pub fn add_edge(&mut self, pred: impl GraphLabel, next: impl GraphLabel) {
        self.graph.add_node_edge(pred, next);
    }

    pub fn add_condition_edge<F>(
        &mut self,
        pred: impl GraphLabel,
        branches: HashMap<B, InternedGraphLabel>,
        condition: F,
    ) where
        F: Fn(&S) -> Vec<B> + Send + Sync + 'static,
    {
        self.graph
            .add_node_condition_edge(pred, branches, condition);
    }

    pub async fn run_until_stuck(
        &self,
        mut state: S,
        max_steps: usize,
    ) -> Result<(S, InternedGraphLabel), GraphStepError<E>>
    where
        S: Send + Sync + 'static,
        E: Send + Sync + 'static,
    {
        let mut current = self.entry;

        for _ in 0..max_steps {
            let (new_state, next) = self.graph.run_once(current, &state).await?;
            state = new_state;

            if next.len() != 1 {
                return Ok((state, current));
            }

            current = next[0];
        }

        Ok((state, current))
    }
}

pub struct StateGraphRunner<'g, S, E, B: BranchKind> {
    pub state_graph: &'g StateGraph<S, E, B>,
    pub current: InternedGraphLabel,
    pub state: S,
}

impl<'g, S, E, B: BranchKind> StateGraphRunner<'g, S, E, B> {
    pub fn new(state_graph: &'g StateGraph<S, E, B>, initial_state: S) -> Self {
        Self {
            state_graph,
            current: state_graph.entry,
            state: initial_state,
        }
    }

    pub async fn step(&mut self) -> Result<Vec<InternedGraphLabel>, GraphStepError<E>>
    where
        S: Send + Sync + 'static,
        E: Send + Sync + 'static,
    {
        let (new_state, next) = self
            .state_graph
            .graph
            .run_once(self.current, &self.state)
            .await?;
        self.state = new_state;
        if next.len() == 1 {
            self.current = next[0];
        }
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::GraphLabel;
    use crate::node::NodeError;
    use async_trait::async_trait;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, GraphLabel)]
    enum TestLabel {
        A,
        B,
        C,
    }

    #[derive(Debug)]
    struct AddOne;

    #[async_trait]
    impl Node<i32, i32, NodeError> for AddOne {
        async fn run(&self, input: &i32) -> Result<i32, NodeError> {
            Ok(*input + 1)
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    enum TestBranch {
        Default,
    }

    #[tokio::test]
    async fn state_graph_runs_linear_chain() {
        let mut graph: Graph<i32, i32, NodeError, TestBranch> = Graph {
            nodes: HashMap::new(),
        };

        graph.add_node(TestLabel::A, AddOne);
        graph.add_node(TestLabel::B, AddOne);
        graph.add_node(TestLabel::C, AddOne);

        graph.add_node_edge(TestLabel::A, TestLabel::B);
        graph.add_node_edge(TestLabel::B, TestLabel::C);

        let entry = TestLabel::A.intern();
        let sg: StateGraph<i32, NodeError, TestBranch> = StateGraph::new(entry, graph);

        let (final_state, final_label) = sg.run_until_stuck(0, 10).await.unwrap();

        assert_eq!(final_state, 3);
        assert_eq!(final_label, TestLabel::C.intern());
    }

    #[tokio::test]
    async fn state_graph_runs_conditional_branch_taken() {
        let mut graph: Graph<i32, i32, NodeError, TestBranch> = Graph {
            nodes: HashMap::new(),
        };

        graph.add_node(TestLabel::A, AddOne);
        graph.add_node(TestLabel::B, AddOne);

        let mut branches = HashMap::new();
        branches.insert(TestBranch::Default, TestLabel::B.intern());

        graph.add_node_condition_edge(TestLabel::A, branches, |state: &i32| {
            if *state >= 0 {
                vec![TestBranch::Default]
            } else {
                Vec::new()
            }
        });

        let entry = TestLabel::A.intern();
        let sg: StateGraph<i32, NodeError, TestBranch> = StateGraph::new(entry, graph);

        let (final_state, final_label) = sg.run_until_stuck(0, 10).await.unwrap();

        assert_eq!(final_state, 2);
        assert_eq!(final_label, TestLabel::B.intern());
    }

    #[tokio::test]
    async fn state_graph_runs_conditional_branch_not_taken_stops_at_predicate() {
        let mut graph: Graph<i32, i32, NodeError, TestBranch> = Graph {
            nodes: HashMap::new(),
        };

        graph.add_node(TestLabel::A, AddOne);

        let mut branches = HashMap::new();
        branches.insert(TestBranch::Default, TestLabel::B.intern());

        graph.add_node_condition_edge(TestLabel::A, branches, |state: &i32| {
            if *state > 0 {
                vec![TestBranch::Default]
            } else {
                Vec::new()
            }
        });

        let entry = TestLabel::A.intern();
        let sg: StateGraph<i32, NodeError, TestBranch> = StateGraph::new(entry, graph);

        let (final_state, final_label) = sg.run_until_stuck(-1, 10).await.unwrap();

        assert_eq!(final_state, 0);
        assert_eq!(final_label, TestLabel::A.intern());
    }

    #[tokio::test]
    async fn state_graph_runner_steps_through_linear_chain() {
        let mut graph: Graph<i32, i32, NodeError, TestBranch> = Graph {
            nodes: HashMap::new(),
        };

        graph.add_node(TestLabel::A, AddOne);
        graph.add_node(TestLabel::B, AddOne);
        graph.add_node(TestLabel::C, AddOne);

        graph.add_node_edge(TestLabel::A, TestLabel::B);
        graph.add_node_edge(TestLabel::B, TestLabel::C);

        let entry = TestLabel::A.intern();
        let sg: StateGraph<i32, NodeError, TestBranch> = StateGraph::new(entry, graph);

        let mut runner = StateGraphRunner::new(&sg, 0);

        let next1 = runner.step().await.unwrap();
        assert_eq!(runner.state, 1);
        assert_eq!(runner.current, TestLabel::B.intern());
        assert_eq!(next1, vec![TestLabel::B.intern()]);

        let next2 = runner.step().await.unwrap();
        assert_eq!(runner.state, 2);
        assert_eq!(runner.current, TestLabel::C.intern());
        assert_eq!(next2, vec![TestLabel::C.intern()]);

        let next3 = runner.step().await.unwrap();
        assert_eq!(runner.state, 3);
        assert_eq!(runner.current, TestLabel::C.intern());
        assert!(next3.is_empty());
    }
}
