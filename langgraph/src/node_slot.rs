use derive_more::derive::From;
use std::borrow::Cow;

/// A [`SlotLabel`] is used to reference a slot by either its name or index
/// inside the [`RenderGraph`](super::RenderGraph).
#[derive(Debug, Clone, Eq, PartialEq, From)]
pub enum SlotLabel {
    Index(usize),
    Name(Cow<'static, str>),
}

impl From<&SlotLabel> for SlotLabel {
    fn from(value: &SlotLabel) -> Self {
        value.clone()
    }
}

impl From<String> for SlotLabel {
    fn from(value: String) -> Self {
        SlotLabel::Name(value.into())
    }
}

impl From<&'static str> for SlotLabel {
    fn from(value: &'static str) -> Self {
        SlotLabel::Name(value.into())
    }
}

#[derive(Debug, Default)]
pub struct SlotInfo {
    pub name: Cow<'static, str>,
}

impl SlotInfo {
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Self { name: name.into() }
    }
}

#[derive(Debug, Default)]
pub struct SlotInfos {
    pub slots: Vec<SlotInfo>,
}

impl<T: IntoIterator<Item = SlotInfo>> From<T> for SlotInfos {
    fn from(slots: T) -> Self {
        SlotInfos {
            slots: slots.into_iter().collect(),
        }
    }
}

impl SlotInfos {
    /// Returns the count of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Returns true if there are no slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Retrieves the [`SlotInfo`] for the provided label.
    pub fn get_slot(&self, label: impl Into<SlotLabel>) -> Option<&SlotInfo> {
        let label = label.into();
        let index = self.get_slot_index(label)?;
        self.slots.get(index)
    }

    /// Retrieves the index (inside input or output slots) of the slot for the provided label.
    pub fn get_slot_index(&self, label: impl Into<SlotLabel>) -> Option<usize> {
        let label = label.into();
        match label {
            SlotLabel::Index(index) => Some(index),
            SlotLabel::Name(ref name) => self.slots.iter().position(|s| s.name == *name),
        }
    }
}
