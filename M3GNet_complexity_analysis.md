# M3GNet 在 MatGL 中的推理时间复杂度分析

## 1. 目的

本文档整理 `matgl` 仓库中 `M3GNet` 模型的代码路径，并从实现角度分析：

- 单次 **能量推理** 的主要时间开销；
- 单次 **力推理** 的主要时间开销；
- 复杂度如何用原子数、边数、邻居数和 three-body triples 数来表示。

本文分析基于 MatGL 仓库中的 DGL 版本 M3GNet 实现。

---

## 2. M3GNet 的核心实现位置

MatGL 中与 M3GNet 推理最相关的实现主要在以下文件：

- `src/matgl/models/_m3gnet.py`
  - 定义 `M3GNet` 模型本体；
  - 串联 bond expansion、three-body basis、three-body interaction、graph convolution 和 readout。
- `src/matgl/layers/_graph_convolution_dgl.py`
  - 定义 `M3GNetGraphConv` 与 `M3GNetBlock`；
  - 实现 edge / node / state 的更新。
- `src/matgl/layers/_three_body.py`
  - 定义 `ThreeBodyInteractions`；
  - 实现每个 block 中的 three-body 更新。
- `src/matgl/graph/_compute_dgl.py`
  - 构造 line graph；
  - 枚举 three-body triples；
  - 计算角度所需的几何量。
- `src/matgl/apps/_pes_dgl.py`
  - 定义 `Potential`；
  - 对总能量进行自动求导得到 forces 和 stresses。

---

## 3. 变量记号

为了表述复杂度，定义如下符号：

- \(N\)：原子数；
- \(E\)：原子图（pair graph）中的边数；
- \(d_i\)：第 \(i\) 个原子的邻居数；
- \(k\)：平均邻居数；
- \(T\)：three-body triples 数量；
- \(B\)：M3GNet block 的数量，即 `nblocks`。

在 M3GNet 中，three-body triples 数量由 line graph 的边数决定，其规模大致为：

\[
T = \sum_i d_i(d_i - 1)
\]

如果平均邻居数为 \(k\)，则可以粗略写成：

\[
E \sim Nk, \qquad T \sim Nk^2
\]

因此，M3GNet 比普通只做 pair message passing 的图网络更重的根本原因，是存在显著的 three-body 开销。

---

## 4. M3GNet 单次能量推理的代码路径

`M3GNet.forward()` 的核心流程可以概括为：

1. 计算 pair graph 上每条边的键向量和距离；
2. 对距离做 radial / bond expansion；
3. 构造 three-body 用的 line graph；
4. 在 line graph 上计算角度并做 spherical Bessel + spherical Harmonics 展开；
5. 做 node / edge / state embedding；
6. 循环执行 `nblocks` 个 block：
   - 先做 `ThreeBodyInteractions`；
   - 再做 `M3GNetBlock` 图卷积更新；
7. 最后 readout 得到图级能量或原子能量并求和。

从实现上说，`_m3gnet.py` 中的主循环是：

- 前半段准备 `bond_dist`、`rbf`、`l_g`、`three_body_basis`；
- 主体循环中每个 block 先 `three_body_interactions[i](...)`，再 `graph_layers[i](...)`；
- 最后调用 readout 和 final layer 输出结果。

---

## 5. 各阶段复杂度分析

### 5.1 计算 bond vectors 和 bond distances

模型先在 pair graph 上计算每条边的：

- `bond_vec`
- `bond_dist`

这一步只对边做一次几何计算，因此时间复杂度是：

\[
O(E)
\]

---

### 5.2 bond expansion / radial basis expansion

之后模型会对每条边的距离做 basis expansion，生成每条边的 radial 特征。

如果 radial basis 维度记为 \(d_{rbf}\)，则复杂度为：

\[
O(E \cdot d_{rbf})
\]

在 `max_n`、`max_l` 固定时，\(d_{rbf}\) 可看成常数，因此这部分可近似视为：

\[
O(E)
\]

---

### 5.3 构造 line graph / three-body triples

这是 M3GNet 的关键额外成本。

代码中 `create_line_graph(g, threebody_cutoff)` 会先按 three-body cutoff 保留可参与 three-body 的边，然后在 `_compute_3body()` 中：

- 统计每个原子的出边数 `n_bond_per_atom`；
- 为每个原子枚举所有有序邻边对；
- 因而对每个原子产生 \(d_i(d_i-1)\) 个 three-body 组合。

所以这一步复杂度是：

\[
O\left(\sum_i d_i^2\right)
\]

或者直接写成：

\[
O(T)
\]

其中

\[
T = \sum_i d_i(d_i-1)
\]

如果平均邻居数为 \(k\)，则这部分约为：

\[
O(Nk^2)
\]

---

### 5.4 three-body basis expansion

line graph 构造好后，模型会在 line graph 的每条边上：

- 计算 `cos_theta` 和 `phi`；
- 对 triple bond lengths 做 spherical Bessel expansion；
- 对角度做 spherical harmonics expansion；
- 最后合并成 three-body basis。

若 three-body basis 维度记为 \(d_{3b}\)，则这一部分复杂度可写成：

\[
O(T \cdot d_{3b})
\]

在默认超参数固定时，\(d_{3b}\) 是常数，因此可简化为：

\[
O(T)
\]

---

### 5.5 embedding

embedding 阶段分别对：

- node 类型；
- edge 的 radial 特征；
- 可选 state 特征；

做 embedding / MLP 变换。

其复杂度可写成：

\[
O(N \cdot d_n + E \cdot d_e)
\]

若隐藏维度固定，则可近似为：

\[
O(N + E)
\]

---

### 5.6 每个 block 的 ThreeBodyInteractions

在每个 block 中，`ThreeBodyInteractions` 会：

1. 用 atom update network 更新节点特征；
2. gather 每个 triple 对应的终点原子特征；
3. 与 `three_body_basis` 做逐 triple 乘法；
4. 根据 cutoff 权重修正；
5. 用 `scatter_sum` 聚合回 bond；
6. 再更新 edge 特征。

因此，单个 block 的 three-body 更新主导项是按 triple 的计算和聚合，复杂度可近似写为：

\[
O(N + T + E)
\]

通常由于 \(T\) 远大于 \(N\)，该部分工程上常视为：

\[
O(T + E)
\]

甚至常常可以认为 **three-body 是每个 block 的主导成本**。

---

### 5.7 每个 block 的图卷积 M3GNetGraphConv

`M3GNetGraphConv` 主要包括：

- `edge_update_()`：按边计算 edge update；
- `node_update_()`：按边生成消息，再按节点聚合；
- `state_update_()`：可选的图级 state 更新。

所以单个 block 的图卷积复杂度近似为：

\[
O(E \cdot C_{edge/node} + N)
\]

如果把隐藏层维度和 MLP 规模看作常数，则可以简化成：

\[
O(E + N) \approx O(E)
\]

---

### 5.8 readout

M3GNet 最后会做 readout：

- intensive 情况下，对 node 或 edge 做图级 readout；
- extensive 情况下，先预测原子能，再对节点求和。

因此 readout 的复杂度一般为：

\[
O(N)
\]

若使用 edge readout，则可以看成与 \(E\) 同阶；但在常见的 atom-based readout 下通常近似为 \(O(N)\)。

---

## 6. 单次能量推理的总复杂度

把各部分加起来，可以把 M3GNet 的 energy inference 写成：

\[
T_{\text{energy}} = O\Big(
E
+ T
+ T \cdot d_{3b}
+ B(E + T + N)
+ N
\Big)
\]

更简洁地写：

\[
T_{\text{energy}} = O\Big(B(E + T) + T \cdot d_{3b}\Big)
\]

若 `max_n`、`max_l`、hidden dim、`nblocks` 都看成固定超参数，则可进一步简化为：

\[
T_{\text{energy}} = O(E + T)
\]

而由于

\[
T = \sum_i d_i(d_i-1)
\]

所以也可以写成：

\[
T_{\text{energy}} = O\left(E + \sum_i d_i(d_i-1)\right)
\]

或者强调 three-body 主导时写成：

\[
T_{\text{energy}} = O\left(\sum_i d_i^2\right)
\]

---

## 7. 用平均邻居数 \(k\) 表示的复杂度

如果用平均邻居数 \(k\) 来表示：

- \(E \sim Nk\)
- \(T \sim Nk^2\)

因此：

\[
T_{\text{energy}} = O(Nk + Nk^2)
\]

在 three-body 主导时，通常记作：

\[
T_{\text{energy}} \approx O(Nk^2)
\]

### 特别说明

- 如果 cutoff 固定、材料密度和局域环境分布相近，则 \(k\) 近似为常数；此时 M3GNet 对原子数表现为近似线性扩展：

\[
T_{\text{energy}} \approx O(N)
\]

- 但更一般地说，M3GNet 的真实复杂度并不只是 \(O(N)\)，而是 strongly dependent on neighbor count，更准确的表达仍然是：

\[
O(E + T) \quad \text{或} \quad O\left(\sum_i d_i^2\right)
\]

---

## 8. 力推理的复杂度

MatGL 中对力的推理不是模型直接输出独立 head，而是：

1. 先通过模型前向得到总能量 `total_energies`；
2. 然后调用 `torch.autograd.grad(total_energies, positions)` 对坐标求导；
3. 最终得到 `forces = -grads[0]`。

因此，力推理的时间复杂度可理解为：

\[
T_{\text{force}} = O(T_{\text{energy}} + T_{\text{backward}})
\]

对神经网络来说，backward 一般与 forward 同阶，但常数因子更大，所以工程上通常可认为：

- **force inference 明显慢于 energy inference**；
- 其阶数仍与 \(E\) 和 \(T\) 相关；
- 只是常数项大得多。

因此可写为：

\[
T_{\text{force}} = \Theta(E + T)
\]

但需要强调：

> 这里的 `\Theta(E + T)` 是指与 energy inference 同阶；实际运行时间通常会因为 autograd 而显著增大。

如果同时计算 stress，则还需要对 strain 求导，会进一步增加常数开销。

---

## 9. 哪些部分最可能成为性能瓶颈

从代码结构看，M3GNet 推理最可能的性能瓶颈有三类：

### 9.1 line graph / three-body 枚举

因为 three-body triples 的数量按 \(\sum_i d_i(d_i-1)\) 增长，邻居数上升时会迅速放大。

### 9.2 three-body basis expansion

在每个 triple 上计算 spherical Bessel 和 spherical harmonics，会对 line graph 的边数线性放大。

### 9.3 每个 block 的 ThreeBodyInteractions

每个 block 都要在 triples 上做乘法、加权和 `scatter_sum` 聚合，因此 block 数 \(B\) 增加时，three-body 部分成本几乎线性增加。

---

## 10. 一个适合报告/benchmark 的最终表述

如果需要在报告、论文笔记或 benchmark 文档中简洁表述，可使用下面几种写法。

### 写法 A：最准确、最贴近实现

\[
T_{\text{energy}} = O\Big(B(E + T)\Big), \qquad T = \sum_i d_i(d_i-1)
\]

其中 \(E\) 为 pair graph 边数，\(T\) 为 three-body triples 数，\(B\) 为 M3GNet blocks 数。

### 写法 B：用邻居数表示

\[
T_{\text{energy}} = O(BNk^2)
\]

其中 \(k\) 是平均邻居数。

### 写法 C：固定 cutoff / 固定邻居分布的工程近似

\[
T_{\text{energy}} \approx O(N)
\]

### 写法 D：力推理

\[
T_{\text{force}} = O(T_{\text{energy}} + T_{\text{backward}})
\]

工程上通常理解为：

- 力推理与能量推理同阶；
- 但因为需要 autograd，常数因子更大；
- 实际 wall-clock time 往往显著高于只算能量。

---

## 11. 实际 benchmark 时的建议

如果后续要测 M3GNet 的实际时间性能，建议把时间拆成以下几段分别统计：

1. **graph construction 时间**
   - 结构转 DGL graph；
   - 包括 `g` 和可能的 `l_g` 构造。
2. **energy forward 时间**
   - 只调用模型前向。
3. **force 时间**
   - 调用 `Potential.forward()`，让 autograd 产生力。
4. **是否缓存 line graph**
   - 如果 `l_g` 可复用，应与“每次重新构造 line graph”的情况区分统计。

否则测得的时间会混合：

- 图构造成本；
- three-body 预处理成本；
- 神经网络前向成本；
- 自动求导成本。

---

## 12. 总结

M3GNet 在 MatGL 中的推理复杂度，本质上由两部分组成：

- pair-wise 图上的边更新，规模约为 \(O(E)\)；
- three-body 关系上的 triple 更新，规模约为 \(O(T)\)，其中 \(T = \sum_i d_i(d_i-1)\)。

因此：

- **单次能量推理** 更合适的表达是
  \[
  O(E + T)
  \]
  或写成
  \[
  O\left(\sum_i d_i^2\right)
  \]
- **单次力推理** 则是在能量推理基础上再加一次 automatic differentiation，因此与能量推理同阶，但常数明显更大。
- 如果 cutoff 固定、邻居数近似常数，则 M3GNet 对系统大小表现为近似线性扩展；但更一般地，three-body 项决定了它对邻居数更敏感。

