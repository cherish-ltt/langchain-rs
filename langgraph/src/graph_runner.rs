use futures::stream::unfold;
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

pub const DEFAULT_MAX_STEPS: usize = 25;

pub struct AgentGraphRunner;

#[derive(Debug)]
pub enum StepEvent {
    NodeEnd { label: InternedGraphLabel },
    Finished { label: InternedGraphLabel },
}

pub struct GraphStepper<'a, S: State> {
    graph: &'a StateGraph<S>,
    state: S,
    current: InternedGraphLabel,
    end: InternedGraphLabel,
    steps: usize,
    max_steps: usize,
}

impl<'a, S: State> GraphStepper<'a, S> {
    pub fn new(graph: &'a StateGraph<S>, initial: S, max_steps: usize) -> Self {
        Self {
            graph,
            state: initial,
            current: graph.get_start(),
            end: graph.get_end(),
            steps: 0,
            max_steps,
        }
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    pub async fn step(&mut self) -> Result<StepEvent, GraphRunnerError> {
        if self.steps >= self.max_steps {
            return Err(GraphRunnerError::MaxStepsExceeded(self.max_steps));
        }

        let Some(edges) = self.graph.get_node_edges(self.current) else {
            return Ok(StepEvent::Finished {
                label: self.current,
            });
        };

        let next = match edges {
            EdgesA::NodeEdge(edge) => edge.get_output_node(),
            EdgesA::ConditionEdge(route) => route(&self.state),
        };

        let node_state = self
            .graph
            .get_node(next)
            .ok_or(GraphRunnerError::NodeNotFound(next))?;
        let diff = node_state.node.run(&self.state).await?;

        self.state.apply_diff(diff);
        self.current = next;

        if self.current == self.end {
            return Ok(StepEvent::Finished {
                label: self.current,
            });
        }

        self.steps += 1;
        Ok(StepEvent::NodeEnd { label: next })
    }
}

impl AgentGraphRunner {
    pub async fn run<S: State>(graph: &StateGraph<S>, initial: S) -> Result<S, GraphRunnerError> {
        Self::run_graph(graph, initial, DEFAULT_MAX_STEPS).await
    }

    pub async fn run_graph<S: State>(
        graph: &StateGraph<S>,
        state: S,
        max_steps: usize,
    ) -> Result<S, GraphRunnerError> {
        let mut stepper = GraphStepper::new(graph, state, max_steps);
        loop {
            match stepper.step().await {
                Ok(StepEvent::Finished { .. }) => break,
                Ok(StepEvent::NodeEnd { .. }) => {}
                Err(err) => return Err(err),
            }
        }
        Ok(stepper.state().clone())
    }
}

/// 节点事件流，用于调试、可视化、交互式控制、HITL（Human-In-The-Loop）
pub fn stream_graph<'a, S: State>(
    graph: &'a StateGraph<S>,
    initial: S,
    max_steps: usize,
) -> impl futures::Stream<Item = Result<StepEvent, GraphRunnerError>> + 'a {
    let stepper = GraphStepper::new(graph, initial, max_steps);
    unfold(Some(stepper), |state| async {
        let mut stepper = match state {
            Some(stepper) => stepper,
            None => return None,
        };
        match stepper.step().await {
            Ok(event) => {
                let done = matches!(event, StepEvent::Finished { .. });
                let next_state = if done { None } else { Some(stepper) };
                Some((Ok(event), next_state))
            }
            Err(err) => Some((Err(err), None)),
        }
    })
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
