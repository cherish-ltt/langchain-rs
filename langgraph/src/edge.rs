use std::fmt::Debug;

use crate::node::InternedGraphLabel;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Edge {
    input_node: InternedGraphLabel,
    output_node: InternedGraphLabel,
}

impl Edge {
    pub fn new(input_node: InternedGraphLabel, output_node: InternedGraphLabel) -> Self {
        Self {
            input_node,
            output_node,
        }
    }

    /// Returns the id of the `input_node`.
    pub fn get_input_node(&self) -> InternedGraphLabel {
        self.input_node
    }

    /// Returns the id of the `output_node`.
    pub fn get_output_node(&self) -> InternedGraphLabel {
        self.output_node
    }
}

#[derive(PartialEq, Eq)]
pub enum EdgeExistence {
    Exists,
    DoesNotExist,
}

pub type Route<S> = fn(&S) -> InternedGraphLabel;
