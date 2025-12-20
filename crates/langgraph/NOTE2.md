这里其实牵扯的是“**Graph 这一层到底要不要绑定某一种执行语义**”，我现在的选择是：  
底层只告诉你“有哪些后继”，**但不替你决定“怎么走”**。原因主要有三类。

---

**1. 多后继在不同语义下含义完全不同**

一旦允许一个节点有多个后继（不论是多条普通边，还是条件边返回多个分支），这里至少有几种常见的执行含义：

- **并行执行**  
  - 当前节点跑完，所有后继都要跑（像 DAG 任务调度 / 工作流引擎）。  
  - 对应语义：fan-out / broadcast。
- **多路分支（二选一、多选一）**  
  - 条件边返回多个候选下游，但你只想选其中一个（比如按优先级、按某个评分）。  
  - 对应语义：route / switch / router。
- **竞速（race）**  
  - 同时启动多个后继，哪个先完成就采用哪个结果，其他取消。  
  - 对应语义：any-of / race。
- **状态机“同时存在多个活动节点”**  
  - 比如 LangGraph 里的 agent graph，可以有多个并发分支，最后再 merge 成一个 state。  

这些模式**都合法**，但互相矛盾：  
底层 `Graph` / `Executor` 没有足够的信息判断你想的是哪一种。

如果在这一层就强行做决定（比如：

- 默认并行跑所有后继；或者
- 默认只取第一个后继；或者
- 默认 panic/报错），

都会把某一种执行模型**硬编码**到库的核心里，未来要支持其他语义就会反过来受限。

---

**2. 当前 Executor 的定位是“最小骨架”，不是“完成体调度器”**

你现在的 `Executor` 和 `run_once` 的职责非常窄（`graph.rs:121-166, 147-159`）：

- `run_once`：从当前节点出发，执行一次节点逻辑，算出所有**可达后继 label 列表**：
  ```rust
  let mut next_nodes = Vec::new();
  for edge in &state.edges {
      match edge {
          Edge::NodeEdge(label) => next_nodes.push(*label),
          Edge::ConditionalEdge { next_nodes: branches, condition } => {
              let branches_to_take = (condition)(&output);
              for branch in branches_to_take {
                  if let Some(label) = branches.get(&branch) {
                      next_nodes.push(*label);
                  }
              }
          }
      }
  }
  ```
- `Executor::step` 目前只做了一个很保守的小优化：  
  当 `next.len() == 1` 时，自动把 `current` 更新为那一个（线性流场景）：

  ```rust
  if next.len() == 1 {
      self.current = next[0];
  }
  ```

而当 `next.len() != 1`（0 个或 >1 个）时，执行器只是原样把结果交给调用方。  
这是故意留的“扩展点”：

- 对“**单链路**”图来说，你直接用这个 executor 就足够：
  - 每个节点要么没有后继，要么只有一个后继；
  - 这时 `current` 会自动推进，你就能像跑 FSM 一样一跳一跳往后跑。
- 对“**复杂路由/并发**”图来说，底层只负责告诉你“有哪些候选后继”，  
  至于：
  - 要不要全部 fork；
  - 要不要选一个优先级最高的；
  - 要不要 race；
  - 甚至要不要根据外部状态再过滤一遍；  
  这些策略都应该在 **更高层的状态机 / runtime / orchestration 层**来实现。

这样一来：

- 核心 `Graph` 和 `Executor` 仍然是**通用图运行时**；
- 上层可以自由组合不同调度策略，而不用去 hack 底层代码。

---

**3. 你有两个自然方向可以走**

如果你不想让上层决定，而是希望“框架给一个强约束的执行语义”，那有两个选项：

1. **限制图结构，保证每个节点最多一个后继**  
   - 做法：
     - 在 `try_add_node_edge` / `try_add_node_condition_edge` 里再加约束：  
       一个节点的 `edges` 中只允许：
       - 一条普通边，或
       - 一条条件边且 condition 最终只返回 0/1 个后继；
     - 或者在 `run_once` 中 assert `next.len() <= 1`。
   - 效果：
     - Executor 可以放心地自动推进 `current`，根本不会出现“多后继”的问题；
     - 但你就失去了很多 LangGraph v1 那种“多路分支/并发的表达力”。

2. **在上层封装一个“带策略的执行器”**  
   - 比如 `LinearExecutor` / `ParallelExecutor` / `RouterExecutor`：
     - `LinearExecutor`：要求 `next.len() <= 1`，否则 panic；
     - `ParallelExecutor`：对 `next` 中的所有 label 并发调用 `run_once`；
     - `RouterExecutor`：给每个后继附加权重/优先级，从中选一个；
   - 这些执行器内部分别用现在的 `Graph + Executor` 作为底层，然后组合出不同语义。

结合你的目标（对标 LangGraph v1），**第二种更灵活也更贴近原版设计**：  
- `Graph + Executor` 做底层通用 runtime；  
- 上层 `StateGraph` / `Workflow` / `AgentGraph` 再给出具体的“如何消费多个后继”的策略。

---

**小结**

- 允许一个节点存在多个后继时，“应该怎么走”在不同业务/模型里含义完全不同：
  - 并发、择一、race、merge… 没有单一正确答案。
- 为了让 `langgraph` 这个 crate 保持通用性、可扩展性，底层 `Graph::run_once` 和 `Executor` 只负责：
  - 正确执行当前节点；
  - 准确产出“所有可达后继”的集合；
  - 在线性场景下帮你自动推进 `current`（`len == 1`）。
- 一旦 `next.len() > 1`，就进入“策略层”的问题，最好交给上层的执行模型（StateGraph/Workflow/Agent orchestrator）决定，而不是在这个基础库里硬编码。

如果你倾向于某一种固定语义（比如“我们这个库只做线性/树状流程，不允许并发分支”），我也可以帮你把这些约束固化到 `Graph`/`Executor`，在类型和运行时层面都强行禁止“多后继”的图。