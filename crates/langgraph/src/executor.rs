use crate::{
    edge::{BranchKind, Edge},
    graph::{Graph, GraphError, GraphStepError},
    label::InternedGraphLabel,
};

pub struct Executor<'g, I, O, E, B: BranchKind> {
    pub graph: &'g Graph<I, O, E, B>,
    pub current: InternedGraphLabel,
}

impl<'g, I, O, E, B: BranchKind> Executor<'g, I, O, E, B> {
    pub fn new(graph: &'g Graph<I, O, E, B>, start: InternedGraphLabel) -> Self {
        Self {
            graph,
            current: start,
        }
    }

    pub async fn step(
        &mut self,
        input: &I,
    ) -> Result<(O, Vec<InternedGraphLabel>), GraphStepError<E>>
    where
        I: Send + Sync + 'static,
        O: Send + Sync + 'static,
        E: Send + Sync + 'static,
    {
        let (output, next) = self.graph.run_once(self.current, input).await?;
        if next.len() == 1 {
            self.current = next[0];
        }
        Ok((output, next))
    }

    pub async fn run_until_stuck(
        &mut self,
        input: &I,
        max_steps: usize,
    ) -> Result<Vec<(O, Vec<InternedGraphLabel>)>, GraphStepError<E>>
    where
        I: Send + Sync + 'static,
        O: Send + Sync + 'static,
        E: Send + Sync + 'static,
    {
        let mut steps = Vec::new();

        for _ in 0..max_steps {
            let (output, next) = self.step(input).await?;
            let should_stop = next.len() != 1;
            steps.push((output, next));
            if should_stop {
                break;
            }
        }

        Ok(steps)
    }
}

impl<I, O, E, B: BranchKind> Graph<I, O, E, B> {
    pub async fn run_once(
        &self,
        current: InternedGraphLabel,
        input: &I,
    ) -> Result<(O, Vec<InternedGraphLabel>), GraphStepError<E>>
    where
        I: Send + Sync + 'static,
        O: Send + Sync + 'static,
        E: Send + Sync + 'static,
    {
        let state = self
            .nodes
            .get(&current)
            .ok_or_else(|| GraphStepError::Graph(GraphError::InvalidNode(current)))?;

        let output = state.node.run(input).await.map_err(GraphStepError::Node)?;

        let mut next_nodes = Vec::new();

        for edge in &state.edges {
            match edge {
                Edge::NodeEdge(label) => next_nodes.push(*label),
                Edge::ConditionalEdge {
                    next_nodes: branches,
                    condition,
                } => {
                    let branches_to_take = (condition)(&output);
                    for branch in branches_to_take {
                        if let Some(label) = branches.get(&branch) {
                            next_nodes.push(*label);
                        }
                    }
                }
            }
        }

        Ok((output, next_nodes))
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use async_trait::async_trait;

    use crate::{
        executor::Executor,
        graph::Graph,
        label::GraphLabel,
        node::{Node, NodeError},
    };

    #[derive(Debug, Clone, PartialEq, Eq, Hash, GraphLabel)]
    enum TestLabel {
        A,
        B,
    }

    #[derive(Debug)]
    struct IncNode;

    #[async_trait]
    impl Node<i32, i32, NodeError> for IncNode {
        async fn run(&self, input: &i32) -> Result<i32, NodeError> {
            Ok(*input + 1)
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    enum TestBranch {
        Default,
        Alt,
    }

    #[tokio::test]
    async fn executor_drives_linear_flow() {
        let mut graph: Graph<i32, i32, NodeError, TestBranch> = Graph {
            nodes: HashMap::new(),
        };

        graph.add_node(TestLabel::A, IncNode);
        graph.add_node(TestLabel::B, IncNode);

        graph.add_node_edge(TestLabel::A, TestLabel::B);

        let a_label = TestLabel::A.intern();
        let mut executor = Executor::new(&graph, a_label);

        let (out1, next1) = executor.step(&0).await.unwrap();
        assert_eq!(out1, 1);
        assert_eq!(next1, vec![TestLabel::B.intern()]);
        assert_eq!(executor.current, TestLabel::B.intern());

        let (out2, next2) = executor.step(&1).await.unwrap();
        assert_eq!(out2, 2);
        assert!(next2.is_empty());
        assert_eq!(executor.current, TestLabel::B.intern());
    }

    #[tokio::test]
    async fn run_until_stuck_executes_multiple_steps() {
        let mut graph: Graph<i32, i32, NodeError, TestBranch> = Graph {
            nodes: HashMap::new(),
        };

        graph.add_node(TestLabel::A, IncNode);
        graph.add_node(TestLabel::B, IncNode);

        graph.add_node_edge(TestLabel::A, TestLabel::B);

        let a_label = TestLabel::A.intern();
        let mut executor = Executor::new(&graph, a_label);

        let steps = executor.run_until_stuck(&0, 10).await.unwrap();

        assert_eq!(steps.len(), 2);
        assert_eq!(executor.current, TestLabel::B.intern());

        assert_eq!(steps[0].1, vec![TestLabel::B.intern()]);
        assert!(steps[1].1.is_empty());
    }
}
