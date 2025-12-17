use std::collections::HashMap;

use langchain_core::state::State;

use crate::{
    GraphError,
    edge::{Edge, Route},
    node::{BaseAgentLabel, EdgesA, GraphLabel, InternedGraphLabel, Node, NodeState},
};

#[derive(Debug)]
pub struct StateGraph<S: State> {
    nodes: HashMap<InternedGraphLabel, NodeState<S>>,
    edges: HashMap<InternedGraphLabel, EdgesA<S>>,
    start: InternedGraphLabel,
    end: InternedGraphLabel,
}

impl<S: State> Default for StateGraph<S> {
    fn default() -> Self {
        Self {
            start: BaseAgentLabel::Start.intern(),
            end: BaseAgentLabel::End.intern(),
            nodes: HashMap::default(),
            edges: HashMap::default(),
        }
    }
}

impl<S: State> StateGraph<S> {
    pub fn add_node<T>(&mut self, label: impl GraphLabel, node: T)
    where
        T: Node<S>,
    {
        let label = label.intern();
        let node_state = NodeState::new(label, node);
        self.nodes.insert(label, node_state);
    }

    pub fn try_set_start(&mut self, label: impl GraphLabel) -> Result<(), GraphError> {
        let label = label.intern();
        self.start = label;
        Ok(())
    }

    pub fn set_start(&mut self, label: impl GraphLabel) {
        self.try_set_start(label).unwrap();
    }

    pub fn try_set_end(&mut self, label: impl GraphLabel) -> Result<(), GraphError> {
        let label = label.intern();
        if !self.nodes.contains_key(&label) {
            return Err(GraphError::InvalidNode(label));
        }
        self.end = label;
        Ok(())
    }

    pub fn set_end(&mut self, label: impl GraphLabel) {
        self.try_set_end(label).unwrap();
    }

    pub fn try_add_node_edge(
        &mut self,
        from: impl GraphLabel,
        to: impl GraphLabel,
    ) -> Result<(), GraphError> {
        let from = from.intern();
        let to = to.intern();
        if self.edges.contains_key(&from) {
            return Err(GraphError::StateEdgeAlreadyExists(from));
        }
        let edge = Edge::new(from, to);
        self.edges.insert(from, EdgesA::NodeEdge(edge));
        Ok(())
    }

    pub fn add_node_edge(&mut self, from: impl GraphLabel, to: impl GraphLabel) {
        self.try_add_node_edge(from, to).unwrap();
    }

    pub fn try_add_condition_edge(
        &mut self,
        from: impl GraphLabel,
        route: Route<S>,
    ) -> Result<(), GraphError> {
        let from = from.intern();
        if self.edges.contains_key(&from) {
            return Err(GraphError::StateEdgeAlreadyExists(from));
        }
        self.edges.insert(from, EdgesA::ConditionEdge(route));
        Ok(())
    }

    pub fn add_condition_edge(&mut self, from: impl GraphLabel, route: Route<S>) {
        self.try_add_condition_edge(from, route).unwrap();
    }

    pub fn get_node_edges(&self, label: impl GraphLabel) -> Option<&EdgesA<S>> {
        self.edges.get(&label.intern())
    }

    pub fn get_node_state(&self, label: impl GraphLabel) -> Option<&NodeState<S>> {
        self.nodes.get(&label.intern())
    }

    pub fn get_start(&self) -> InternedGraphLabel {
        self.start
    }

    pub fn get_end(&self) -> InternedGraphLabel {
        self.end
    }

    pub(crate) fn get_node(&self, current_node: impl GraphLabel) -> Option<&NodeState<S>> {
        self.nodes.get(&current_node.intern())
    }
}
