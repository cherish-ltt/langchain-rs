use langchain_core::state::State;
use thiserror::Error;

use crate::{
    GraphError,
    graph::StateGraph,
    node::{EdgesA, InternedGraphLabel, NodeRunError},
};

#[derive(Error, Debug)]
pub enum GraphRunnerError {
    #[error(transparent)]
    NodeRunFailed(#[from] NodeRunError),
    #[error("node {0:?} not found")]
    NodeNotFound(InternedGraphLabel),
    #[error("max steps {0} exceeded")]
    MaxStepsExceeded(usize),
    #[error(transparent)]
    InvalidGraph(#[from] GraphError),
}

pub struct AgentGraphRunner;

impl AgentGraphRunner {
    const MAX_STEPS: usize = 25;

    pub async fn run<S: State>(graph: &StateGraph<S>, initial: S) -> Result<S, GraphRunnerError> {
        Self::run_graph(graph, initial, Self::MAX_STEPS).await
    }

    pub async fn run_graph<S: State>(
        graph: &StateGraph<S>,
        mut state: S,
        max_steps: usize,
    ) -> Result<S, GraphRunnerError> {
        let end = graph.get_end();
        let mut current_node = graph.get_start();
        let mut steps = 0;

        loop {
            if steps >= max_steps {
                return Err(GraphRunnerError::MaxStepsExceeded(max_steps));
            }

            let Some(edges) = graph.get_node_edges(current_node) else {
                break;
            };

            let next_node = match edges {
                EdgesA::NodeEdge(edge) => edge.get_output_node(),
                EdgesA::ConditionEdge(route) => route(&state),
            };

            let node_state = graph
                .get_node(next_node)
                .ok_or(GraphRunnerError::NodeNotFound(next_node))?;
            let diff = node_state.node.run(&state).await?;

            state.apply_diff(diff);

            current_node = next_node;

            if current_node == end {
                break;
            }
            steps += 1;
        }

        Ok(state)
    }
}

#[cfg(test)]
mod test {
    use langchain_core::state::State;

    use super::*;
    use crate::graph::StateGraph;
    use crate::node::{BaseAgentLabel, GraphLabel, InternedGraphLabel, Node, NodeRunError};

    #[derive(Debug, Clone, Default)]
    struct AgentTestState {
        steps: usize,
        trace: Vec<&'static str>,
    }

    impl State for AgentTestState {
        type Diff = (&'static str, usize);

        fn apply_diff(&mut self, diff: Self::Diff) {
            let (msg, inc) = diff;
            self.trace.push(msg);
            self.steps += inc;
        }
    }

    #[derive(Debug, Hash, PartialEq, Eq, Clone, GraphLabel)]
    enum CAgentLabel {
        Observe,
        Act,
        Reflect,
    }

    #[derive(Debug)]
    struct AgentNodeImpl {
        name: &'static str,
        inc: usize,
    }

    #[async_trait::async_trait]
    impl Node<AgentTestState> for AgentNodeImpl {
        async fn run(&self, _: &AgentTestState) -> Result<(&'static str, usize), NodeRunError> {
            Ok((self.name, self.inc))
        }
    }

    fn route(state: &AgentTestState) -> InternedGraphLabel {
        if state.steps < 2 {
            CAgentLabel::Act.intern()
        } else {
            BaseAgentLabel::End.intern()
        }
    }

    #[tokio::test]
    async fn test_agent_graph_runner_loop() {
        let mut graph = StateGraph::<AgentTestState>::default();

        graph.add_node(
            CAgentLabel::Observe,
            AgentNodeImpl {
                name: "observe",
                inc: 1,
            },
        );
        graph.add_node(
            CAgentLabel::Act,
            AgentNodeImpl {
                name: "act",
                inc: 0,
            },
        );
        graph.add_node(
            CAgentLabel::Reflect,
            AgentNodeImpl {
                name: "reflect",
                inc: 0,
            },
        );
        graph.add_node(
            BaseAgentLabel::End,
            AgentNodeImpl {
                name: "end",
                inc: 0,
            },
        );

        graph.set_start(BaseAgentLabel::Start);
        graph.set_end(BaseAgentLabel::End);

        graph.add_node_edge(BaseAgentLabel::Start, CAgentLabel::Observe);
        graph.add_condition_edge(CAgentLabel::Observe, route);
        graph.add_node_edge(CAgentLabel::Act, CAgentLabel::Reflect);
        graph.add_node_edge(CAgentLabel::Reflect, CAgentLabel::Observe);

        let state = AgentGraphRunner::run(&graph, AgentTestState::default())
            .await
            .unwrap();

        assert_eq!(state.steps, 2);
        assert_eq!(
            state.trace,
            vec!["observe", "act", "reflect", "observe", "end"]
        );
    }
}
