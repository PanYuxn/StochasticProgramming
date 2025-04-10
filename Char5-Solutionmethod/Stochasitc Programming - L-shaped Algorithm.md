---
title: Stochasitc Programming - L-shaped Algorithm
author: Pan Youxuan
link: 
sources: 
tags:
  - "#随机规划"
  - "#算法"
datetime: 2025-04-10
---
----
**Link File Info**： <font color="#c00000">Please think what will u do if u dont know this, Link the unknown to your known.</font>
* 

---
# **L-shaped 算法**
这里先看一下随机规划的标准型的样子如下：
$$ \begin{array} \min z = c^T x + \mathcal{Q}(x) \\ \text{s. t. } Ax = b, \\ x \geq 0, \end{array}$$
其中：
$\mathcal{Q}(x) = \mathrm{E}_\xi Q(x, \xi(\omega))$
 $\mathcal{Q}(x, \xi(\omega)) = \min_{y} \{q(\omega)^T y \mid Wy = h(\omega) - T(\omega)x, y \geq 0\}.$

如果想把上面这个式子写成确定的形式，我们可以把 $\mathcal{Q}(x)$ 部分进行展开，也就是针对每一个情况下 $y$ 决策的情况，具体式子如下：
$$\begin{array}{llll} \min c^T x + \sum_{k=1}^{K} p_k q_k^T y_k \\ \text{s. t.} \quad Ax = b, \\ T_k x + W y_k = h_k, \quad k=1,\ldots,K; \\ x \geq 0, y_k \geq 0, \quad k=1,\ldots,K. \end{array}$$
上面这个式子有一个比较特殊的形式，也就是 `L` 形，具体来说就是左侧的 RHS 矩阵的形状比较特殊，而这种特殊性刚好是针对问题性质的，也就是 $k\in K$ 每一个场景下的情况。

`L-shaped算法` 的核心思想是对上面这个式子进行重构，也就是我们忽略后面 `k \in K` 部分的情况，我们尝试使用关于 `x` 的表达式来对其进行表达，但是这个 `x` 肯定是和 `y` 相关的，因为第二条约束是 `x和y` 的约束。（<font color="#c00000">这里我们要做的就是，在确定一个 `x` 的情况下，我们生成一个关于 `x` 的 cut，这个 cut 包含了这两个信息，而它之所以能包含的原因是我们可以构建一个子问题来得到对应的 Bound</font>）

这里有几个比较关键的点，`L-shaped的影响`：
1. 矩阵 $W$ 的不变性，所有场景共享了矩阵 W.
2. 矩阵 $T_{k}$  表示特定场景的技术系数，它们定义了第一阶段和第二阶段的交互。

$$ \begin{align} \min \quad & z = c^T x + \theta \\ \text{s.t.} \quad & Ax = b, \\ & D_\ell x \geq d_\ell, \quad \ell = 1,\ldots,r, \\ & E_\ell x + \theta \geq e_\ell, \quad \ell = 1,\ldots,s, \\ & x \geq 0, \\ & \theta \in \mathfrak{R}. \end{align} $$

上面这个表达式中，我们发现存在两个类别的 cut，这两个类别的 cut 就是我们后续需要生成的，一个是 `optimal cut`，另一个是 `feasiblity cut`。下面分别讲解对应 cut 的生成方法。

## **1. Optimal Cut**
最优割的意思是啥？从上面的 Formulation 中可以看到 $E_\ell x + \theta \geq e_\ell$ 这个式子中存在三个字母，其中 $\theta$ 表示随机环境对应的 `feature value`，也就是可能存在的值，这里想一下，什么是 `最优`？ -> 我们可以这么考虑，如果在一个固定的 `x` 的情况下，接下来所有的 `随机场景` 均可行，那么他们的值就是 `E(所有场景的obj)`，所以 $\theta$ 就应该大于这个 `E（所有场景的obj）`，这就是这个 Cut 的含义，而 $E_\ell x$ 和 $e_\ell$ 可以理解为对应 `x` 情况下子问题的 `零阶项` 和 `一阶项`

这个最优割其实可以从之前的表达式推导出来，下面写成推导过程。
 $\mathcal{Q}(x, \xi(\omega)) = \min_{y} \{q(\omega)^T y \mid Wy = h(\omega) - T(\omega)x, y \geq 0\}.$
 既然我们已知该式是一个可行的，并且存在最优解，因此我们可以得到最优基 $(y_B,y_N)$，同理，约束中则是 $y_B=B^{-1}(h-Tx)$，将 $y_{B}$ 带入到目标函数中，我们可以得到 $obj=q_B \cdot B^{-1}(h-Tx)$，其中存在一项和 `x` 相关，另一项无关，分别对应 $E_\ell x$ 和 $e_\ell$ 。而我们对其进行期望，也就是我们最终的 $\theta$ 必须要大于等于 $E(q_B \cdot B^{-1}(h-Tx))$.

## **2. Feasibility Cut**
上面考虑的情况是才能在可行解的情况，也就是针对 `x` 所有的随机场景都是可行的情况，但是可能还有一个情况，那就是存在某一个随即场景不可行，这种不可行就对应了 `feasiblity cut`。那么这里就存在一个问题，不可行的情况我们如何得到这个 cut 呢？这里的方法是构建一个子问题，这个子问题 `等价于` 原问题。
$$ \begin{array}{lllll} \min w^{\prime} = e^T v^+ + e^T v^- \\ \text{s.t.} \quad Wy + Iv^+ - Iv^- = h_k - T_k x^{\nu}, \\ y \geq 0, \quad v^+ \geq 0, \quad v^- \geq 0, \end{array} $$
上面这个式子的 Dual 模型是：

$$ \begin{align} \max \quad & \pi^T(h_k - T_k x^{\nu}) \\ \text{s.t.} \quad & \pi^T W \leq 0, \\ & -e \leq \pi \leq e, \end{align} $$
当原问题不可行的时候，说明下面这个式子 `Unbounded`，那么在此时这个 Dual 模型是什么样的呢？
1. Objetive 的优化方向肯定是无穷的，也就是 $\pi^T(h_k - T_k x^{\nu})\ge 0$
2. 满足原约束条件 $\pi^T W \leq 0$

为了让这个模型存在 Bound，我们需要做的就是：$\pi^T(h_k - T_k x^{\nu})\le 0$，这里只需要重新定义一下符号就是上面那个情况了。

## **伪代码**

Algorithm: L-shaped Method for Two-Stage Stochastic Linear Programming

Step 0. Set r = s = v = 0.

Step 1. Set v = v + 1. Solve the linear program
   min $z = c^T x + θ$
   s.t. Ax = b,
	$D_ℓ x ≥ d_ℓ$,         ℓ = 1,...,r,
	$E_ℓ x + θ ≥ e_ℓ$,      ℓ = 1,...,s,
	x ≥ 0,
	θ ∈ ℝ.
   Let ($x^v$, $θ^v$) be an optimal solution. If no constraint (1.4) is present, $θ^v$ is set equal
   to -∞ and is not considered in the computation of $x^v$.

Step 2. Check if $x ∈ K₂$. If not, add at least one cut (1.3) and return to Step 1.
   Otherwise, go to Step 3. (这里先做可行割 1.3，如果找到一个可行割，就重新解原问题)

Step 3. For k = 1,...,K solve the linear program
   min w = $q_k^T$ y
   s.t. Wy = $h_k$ - $T_k$ $x^v$,
        y ≥ 0.
   Let $π_k^v$ be the simplex multipliers associated with the optimal solution of Problem
   k of type (1.5 最优割). Define
   
   $E_{s+1} = ∑_{k=1}^K p_k · (π_k^v)^T T_k$
   $e_{s+1} = ∑_{k=1}^K p_k · (π_k^v)^T h_k.$
   
   Let $w^v = e_{s+1} - E_{s+1}x^v. If θ^v ≥ w^v$, stop; $x^v$ is an optimal solution. Otherwise,
   set s = s + 1, add to the constraint set (1.4), and return to Step 1.

# **一些变体**
## **1. 多切割L-shaped方法 (Multicut L-shaped)**

**动机**：原始L-shaped算法在每次迭代中只生成一个切割平面（聚合了所有场景的信息），这可能导致对问题空间的近似不够精确，尤其是在场景数量大或场景间差异显著时。

**区别**：
- 原始L-shaped：在每次迭代中合并所有场景的信息，生成一个单一的优化性切割
- 多切割L-shaped：为每个场景（或场景组）单独生成切割平面，添加多个切割到主问题

## **2. 正则化L-shaped方法 (Regularized L-shaped)**

**动机**：原始L-shaped算法中，主问题的解可能在迭代过程中剧烈波动，导致收敛缓慢；同时对于退化问题（有多个最优解的情况）表现不佳。

**区别**：
- 原始L-shaped：仅通过切割平面引导搜索方向
- 正则化L-shaped：在主问题目标函数中添加二次正则化项，稳定迭代过程

**计算形式**： 通常在主问题中添加形如`||x - x^k||²`的项，其中x^k是当前迭代的解，这使得新解不会偏离当前解太远。

## **3. 信任域L-shaped方法 (Trust-region L-shaped)**

**动机**：需要更好地控制迭代过程中解的变化范围，同时更有效地利用问题结构信息。

**区别**：
- 原始L-shaped：对解的搜索范围没有显式限制
- 信任域L-shaped：引入信任域约束，限制每次迭代中变量变化的幅度

**实现方式**： 通常通过添加约束`||x - x^k|| ≤ Δ^k`实现，其中Δ^k是当前迭代的信任域半径，会根据模型预测质量动态调整。
