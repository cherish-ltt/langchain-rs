**GraphLabel 与标签系统设计说明**

-对应文件：`crates/langgraph/src/label.rs`、`crates/langgraph/src/intern.rs`、`crates/langgraph/macro/src/lib.rs`-

---

**整体目标**

`langgraph` 中的标签系统需要满足：

- 标签可以是任意用户自定义类型，而不是固定的枚举；
- 标签要能作为 trait 对象使用（`dyn GraphLabel`），仍然支持：
  - 相等比较（`==`）；
  - 哈希（可作为 `HashMap`/`HashSet` 的 key）；
  - 克隆（`dyn_clone`）；
  - 转字符串（`as_str`）；
- 同一个“逻辑相同”的标签在系统中最好只保留一份共享实例（驻留 / interning），减少内存、加快比较。

为此，代码构建了三层抽象：

1. 动态相等 / 哈希：`DynEq`、`DynHash`；
2. 标签 trait：`GraphLabel` 及其 derive 宏；
3. 驻留机制：`Internable`、`Interner`、`Interned<T>`。

---

**1. 动态相等与哈希：`DynEq` / `DynHash`**

文件：`crates/langgraph/src/label.rs:21-60`

- `DynEq` 解决的问题：  
  直接比较 `&dyn Trait` 时，Rust 不知道怎么做“值相等”，只能做指针比较。`DynEq` 提供一个统一接口，让“任何实现 `Eq + Any` 的类型”在作为 trait 对象时仍然能做值相等比较。

  ```rust
  pub trait DynEq: Any {
      fn dyn_eq(&self, other: &dyn DynEq) -> bool;
  }

  impl<T> DynEq for T
  where
      T: Any + Eq,
  {
      fn dyn_eq(&self, other: &dyn DynEq) -> bool {
          (other as &dyn Any)
              .downcast_ref::<T>()
              .map_or(false, |other| self == other)
      }
  }
  ```

  设计要点：

  - 用 `Any` 支持运行时类型检查；
  - 先用 `downcast_ref::<T>()` 检查 `other` 的实际类型是否为 `T`：
    - 类型相同：用 `Eq` 做值比较；
    - 类型不同：直接 `false`；
  - 保证“跨类型永不相等”，避免意外把不同类型的值当成一样的标签。

- `DynHash` 解决的问题：  
  对 `dyn Trait` 做哈希时，同样需要一个统一接口，同时需要确保“不同类型”不会因为底层哈希实现巧合而产生同样的 hash。

  ```rust
  pub trait DynHash: DynEq {
      fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
  }

  impl<T> DynHash for T
  where
      T: DynEq + Hash,
  {
      fn dyn_hash(&self, mut state: &mut dyn Hasher) {
          T::hash(self, &mut state);
          self.type_id().hash(&mut state);
      }
  }
  ```

  设计要点：

  - 基于具体类型 `T` 的 `Hash` 实现写入哈希；
  - 再额外把 `type_id()` 写入：
    - 同一类型同值 → hash 一致；
    - 不同类型即使底层 `Hash` 完全相同 → hash 也不同；
  - 保证 “相等 ⇒ hash 相等” 且 “类型不同 ⇒ hash 通常不同”。

- 在 `dyn GraphLabel` 上，直接使用这两个 trait：

  ```rust
  impl PartialEq for dyn GraphLabel {
      fn eq(&self, other: &Self) -> bool {
          self.dyn_eq(other)
      }
  }

  impl Eq for dyn GraphLabel {}

  impl Hash for dyn GraphLabel {
      fn hash<H: Hasher>(&self, state: &mut H) {
          self.dyn_hash(state);
      }
  }
  ```

  让 `dyn GraphLabel` 具备“值相等、哈希”语义，而不仅是指针相等。

---

**2. 标签 trait：`GraphLabel` 与 derive 宏**

文件：`crates/langgraph/src/label.rs:62-85`，`crates/langgraph/macro/src/lib.rs:40-81`

- `GraphLabel` 是在图中使用的“标签抽象”，要求：

  ```rust
  pub trait GraphLabel: Debug + Send + Sync + DynHash + DynEq {
      fn dyn_clone(&self) -> Box<dyn GraphLabel>;

      /// 返回当前标签的字符串表示
      /// Return a string representation of this label
      fn as_str(&self) -> &'static str;

      fn intern(&self) -> Interned<dyn GraphLabel>
      where
          Self: Sized,
      {
          GRAPH_LABEL_INTERNER.intern(self)
      }
  }
  ```

  设计要点：

  - 必须是线程安全（`Send + Sync`）、可调试（`Debug`）；
  - 要支持前面定义的动态相等 / 哈希（`DynEq + DynHash`）；
  - `dyn_clone` 用于从 trait 对象上获得一个克隆的 `Box<dyn GraphLabel>`；
  - `as_str` 给标签一个稳定的字符串表示（用于日志、调试等）；
  - `intern` 封装了驻留逻辑，使用全局 interner 存储共享实例。

- derive 宏 `#[derive(GraphLabel)]`（`langgraph_macro`）会自动为你的类型生成：

  - 对 `GraphLabel` 的实现：
    - `dyn_clone`：调用 `Clone`，然后装进 `Box<dyn GraphLabel>`；
    - `as_str`：  
      - 对 struct：返回类型名 `"MyLabel"`；
      - 对 enum：返回变体名 `"A"`, `"B"`。
  - 宏展开大致形如（简化后）：

    ```rust
    const _: () = {
        extern crate alloc;

        impl GraphLabel for MyType {
            fn dyn_clone(&self) -> alloc::boxed::Box<dyn GraphLabel> {
                alloc::boxed::Box::new(self.clone())
            }

            fn as_str(&self) -> &'static str {
                "MyType"        // 或 enum 变体名
            }
        }
    };
    ```

  这里用匿名 `const` + `extern crate alloc` 的技巧，是为了：

  - 确保宏代码里 `alloc` 可用；
  - 不污染用户模块的命名空间（不会暴露一个顶层 `extern crate alloc;`）。

---

**3. 驻留机制：`Internable`、`Interner`、`Interned<T>`**

文件：`crates/langgraph/src/intern.rs`

- `Internable` 定义“如何把一个值变成 `'static` 引用，并基于指针做相等与哈希”：

  ```rust
  pub trait Internable: Hash + Eq {
      /// 创建指向 `self` 的静态引用，可能会导致内存泄漏。
      fn leak(&self) -> &'static Self;

      /// 如果两个引用指向同一个值，则返回 `true`。
      fn ref_eq(&self, other: &Self) -> bool;

      /// 将该引用的指针信息写入哈希器。
      fn ref_hash<H: core::hash::Hasher>(&self, state: &mut H);
  }
  ```

  对 `dyn GraphLabel` 的实现（`label.rs:107-129`）：

  - `leak`：通过 `dyn_clone` 克隆到堆上，然后 `Box::leak` 得到 `'static` 引用；
  - `ref_eq`：
    - 比较 `type_id`；
    - 再用 `ptr::addr_eq` 比较指针地址（指针身份）；
  - `ref_hash`：
    - 将 `type_id` 和指针地址写入哈希器。

  这个“指针身份”语义是为 interner 服务的：  
  同一个驻留实例 → 指针相同，ref_eq 为 true，ref_hash 一致。

- `Interner<T>` 是一个线程安全的驻留器：

  ```rust
  pub struct Interner<T: ?Sized + 'static>(RwLock<HashSet<&'static T>>);
  ```

  - 内部是 `RwLock<HashSet<&'static T>>`：
    - 多读单写；
    - 集合元素是 `'static` 引用。

  核心方法：

  ```rust
  impl<T: Internable + ?Sized> Interner<T> {
      pub fn intern(&self, value: &T) -> Interned<T> {
          {
              // 先读锁查找已有条目
              let set = self.0.read().unwrap_or_else(PoisonError::into_inner);

              if let Some(value) = set.get(value) {
                  return Interned(*value);
              }
          }

          {
              // 未命中则写锁 + 可能的泄漏
              let mut set = self.0.write().unwrap_or_else(PoisonError::into_inner);

              if let Some(value) = set.get(value) {
                  Interned(*value)
              } else {
                  let leaked = value.leak();
                  set.insert(leaked);
                  Interned(leaked)
              }
          }
      }
  }
  ```

  行为：

  - 第一次见到某个值：
    - 调用 `value.leak()`，生成 `'static` 引用；
    - 将其插入 `HashSet`；
    - 返回指向该引用的 `Interned<T>`。
  - 之后再次 intern 同样的值：
    - 通过 `Hash + Eq` 和 `Internable::ref_eq/ref_hash` 找到同一实例；
    - 返回同一个 `'static` 引用的 `Interned<T>`。

- `Interned<T>` 是对 `'static` 引用的轻量包装：

  ```rust
  pub struct Interned<T: ?Sized + 'static>(pub &'static T);
  ```

  - 实现了 `Deref`、`Clone`、`Copy`；
  - `PartialEq/Eq/Hash` 基于 `Internable` 的 ref_eq/ref_hash（即“指针身份”）；
  - 对 `Interned<dyn GraphLabel>` 还专门实现了 `GraphLabel`（`label.rs:132-151`）：
    - `dyn_clone` / `as_str` 委托给内部；
    - `intern` 对已经驻留的值是幂等的（直接返回自身）。

---

**4. 全局标签 interner**

文件：`crates/langgraph/src/label.rs:62-65`

```rust
static GRAPH_LABEL_INTERNER: LazyLock<Interner<dyn GraphLabel>> =
    LazyLock::new(|| Interner::new());
```

- 在整个 crate 内提供一个全局的 `Interner<dyn GraphLabel>`；
- `GraphLabel::intern` 直接调用这个全局实例；
- 对使用方来说，只要实现（或 derive）了 `GraphLabel`，就可以：

  ```rust
  #[derive(Debug, Clone, PartialEq, Eq, Hash, GraphLabel)]
  struct MyLabel;

  let a = MyLabel.intern();
  let b = MyLabel.intern();
  assert_eq!(a, b);              // 驻留后的相等
  assert_eq!(a.as_str(), "MyLabel");
  ```

---

**5. 测试保障的行为**

文件：`crates/langgraph/src/label.rs:154-290`

当前单元测试覆盖了：

- `GraphLabel::intern` 和 `as_str`：
  - struct / enum 的标签字符串；
  - 同值多次 intern 得到等价标签。
- `DynEq`：
  - 同类型、值相等 → `dyn_eq` 为 true；
  - 不同类型（`i32` vs `i64`）→ 即使数值相等也为 false。
- `DynHash`：
  - 同类型同值 → `dyn_hash` 一致；
  - 不同类型、底层 `Hash` 完全相同 → `dyn_hash` 不一致（证明 `type_id` 被加入哈希）。
- `GraphLabel for Interned<dyn GraphLabel>`：
  - `dyn_clone` / `as_str` 能正常工作；
  - 对已驻留的标签再次 `intern` 是幂等的。
- `Internable for dyn GraphLabel`：
  - `ref_eq` 使用指针身份（不同分配的实例逻辑上相等但 ref_eq 为 false）；
  - `ref_hash` 使用指针地址产生不同哈希；  
  - 同时 `GraphLabel` 的逻辑相等仍然为 true。

这些测试保证了：

- trait 对象层面的相等 / 哈希行为与预期一致；
- 驻留器正确重用同一实例；
- derive 宏自动生成的 `GraphLabel` 实现符合设计。

---

如果你后续想把这段设计说明放到某个正式文档里（比如 crate-level 文档注释或单独的设计文档），我可以按你指定的格式再帮你重组一版。