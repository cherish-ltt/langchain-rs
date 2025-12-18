use crate::message::Message;

pub trait State: Clone + Default + Send + Sync + 'static {
    type Diff: Send + Sync;
    fn apply_diff(&mut self, diff: Self::Diff);
}

#[derive(Clone, Default)]
pub struct MessageState {
    pub messages: Vec<Message>,
    pub llm_calls: usize,
}

impl MessageState {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            llm_calls: 0,
        }
    }
}

#[derive(Clone)]
pub struct MessageDiff {
    pub new_messages: Vec<Message>,
    pub llm_calls_delta: usize,
}

impl State for MessageState {
    type Diff = MessageDiff;

    fn apply_diff(&mut self, diff: Self::Diff) {
        self.messages.extend(diff.new_messages);
        self.llm_calls += diff.llm_calls_delta;
    }
}
