/// 执行模式枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    /// 同步执行模式，不返回流式事件
    Sync,
    /// 流式执行模式，返回流式事件
    Stream,
}
