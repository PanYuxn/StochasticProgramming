---
title: Generalized Benders Decomposition
author: Pan Youxuan
link: 
sources: 
tags: 
datetime: 2025-03-27
---
----
**Link File Info**： <font color="#c00000">Please think what will u do if u dont know this, Link the unknown to your known.</font>
* 

---
# **Generalized Benders Decompostion**
GBD是由Geoffrion在1972年提出的，它扩展了Benders分解方法到更一般的凸优化问题。这个算法主要用于处理如下形式的问题：

```
min f(x,y)  # 非线性的，也可以是线性的
s.t. G(x,y) ≥ 0  # 和上面同理	
     x ∈ X
     y ∈ Y
```
**假设：**
- X和Y都是非空凸集
- 对于固定的x∈X，f(x,·)和G(x,·)在Y上是凸的
- 对于固定的y∈Y，f(·,y)和G(·,y)在X上是凸的

GBD的核心思想是<font color="#c00000">通过对偶理论将原问题表示在x空间中</font>（这个想法是由于变量 y 那部分比较难，那我就让他变成和 x 相关的表达式，也就是后面一直生成的cut）：
```
# 这个是我们最终想要的 -> 我们希望它的格式就是这样，其中x ∈ X可以看作多条约束形成的一个凸包
min v(x)
s.t. 

其中，如果我们给定一个x的情况下：
v(x) = inf_y f(x,y)
s.t. G(x,y) ≥ 0
     y ∈ Y
上面这个式子我们发现变量是在约束里面的，这个其实我们并不喜欢，因为we want the feasible region is fixed. 所以我们可以做一件事情，把Dual var加进来，然后松弛上去.（得到一个LB）

v(x) = sup_Π inf_y f(x,y)+ΠG(x,y)

那么上面就是一个固定的X解集可以得到一个最优值，同时这个最优值需要不断的上升直到可以和UB接触.
所以，我们需要构建一个受限的主问题，这个主问题的目的就是用θ来替代y部分的事情.
min θ
s.t. θ ≥ inf_y f(x,y)+ΠG(x,y)
	x ∈ X

`这里有一个小细节，约束当中的RHS部分是没有y的！因为我们在生成这个cut的时候，y是固定的，所以完美实现了what we want -> only x in the formulation`

```
# **实际案例**
## **问题**
$$ \begin{align} \min & \quad x_1(-y_1-1) + x_2(-0.8y_2-1.2) \\\text{s.t.} & \quad x_1y_1 + x_2y_2 \leq 0.75 \ \\& \quad x_1 + x_2 = 1 \\ & \quad x_1, x_2 \geq 0 \\ & \quad 0 \leq y_1, y_2 \leq 10 \end{align} $$

---

## **第一次迭代**
### **步骤1：求解子问题**

给定初始解 $x = [1.0, 0.0]$，我们求解子问题：

$$ \begin{align} \min & \quad 1.0 \cdot (-y_1-1) + 0.0 \cdot (-0.8y_2-1.2) \\ \text{s.t.} & \quad 1.0 \cdot y_1 + 0.0 \cdot y_2 \leq 0.75 \\ & \quad 0 \leq y_1, y_2 \leq 10 \end{align} $$

**信息如下：**
 $y_1 = 0.75$，$y_2 = 0$ , $\pi = -1$
存在一个**可行解**为 $x_{1}=1,x_{2}=0,y_{1}=0.75,y_{2}=0$, $obj=-1.75$ 

### **步骤 2：生成割平面**

为了生成割平面，我们需要计算拉格朗日函数对 $x$ 的梯度。拉格朗日函数为：
$$L(x, \pi) = x_1(-y_1-1) + x_2(-0.8y_2-1.2) + \pi(0.75 - x_1y_1 - x_2y_2)$$
在 $x = [1.0, 0.0], y = [0.75, 0], \pi = -1$ 处：
$$L(x, \pi) = 1.0 \cdot (-0.75-1) + 0.0 \cdot (-0.8 \cdot 0-1.2) + (-1) \cdot (0.75 - 1.0 \cdot 0.75 - 0.0 \cdot 0) = -1.75 + 0 + 0 = -1.75$$

对于梯度计算，我们需要确定 $y$ 取什么值才能使 $L(x, \pi)$ 最小。

对于 $x_1$ 的梯度，我们考虑 $y_1$ 的系数：$c_2[0] - \pi = -1 - (-1) = 0$ 由于系数为0，$y_1$ 可以取任意值，梯度为 $c_1[0] = -1$
对于 $x_2$ 的梯度，我们考虑 $y_2$ 的系数：$c_2[1] - \pi = -0.8 - (-1) = 0.2$ 由于系数为正，$y_2$ 取下界0，梯度为 $c_1[1] = -1.2$

因此，割平面的系数为 $[-1, -1.2]$，右侧常数项为 $-\pi \cdot 0.75 = -(-1) \cdot 0.75 = -0.75$
割平面：$\theta \geq -0.75 - 1 \cdot x_1 - 1.2 \cdot x_2$

### **步骤 3：求解主问题**

主问题为：

$$ \begin{align} \min & \quad \theta \\ \text{s.t.} & \quad x_1 + x_2 = 1 \\& \quad \theta \geq -0.75 - x_1 - 1.2x_2  \\& \quad x_1, x_2 \geq 0 \end{align} $$

求解得：$x = [0, 1], \theta = -1.95$ ，更新下界为 -1.95
```
UB: -1.75
LB: -1.95
```

---

## **第二次迭代**

### **步骤1：求解子问题**

给定新的 $x = [0, 1]$，我们求解子问题：

$$ \begin{align} \min & \quad 0 \cdot (-y_1-1) + 1.0 \cdot (-0.8y_2-1.2) \\ \text{s.t.} & \quad 0 \cdot y_1 + 1.0 \cdot y_2 \leq 0.75 \\ & \quad 0 \leq y_1, y_2 \leq 10 \end{align} $$

**信息如下：**
 $y_1 = 0$，$y_2 = 0.75$ , $\pi = -0.8$
存在一个**可行解**为 $x_{1}=0,x_{2}=1,y_{1}=0,y_{2}=0.75$, $obj=-1.8$ 

### **步骤 2：生成割平面**

对于 $x_1$ 的梯度，我们考虑 $y_1$ 的系数：$c_2[0] - \pi = -1 - (-0.8) = -0.2$ 由于系数为负，$y_1$ 取上界10，梯度为 $-0.2 \cdot 10 + (-1) = -3$
对于 $x_2$ 的梯度，我们考虑 $y_2$ 的系数：$c_2[1] - \pi = -0.8 - (-0.8) = 0$ 由于系数为0，$y_2$ 可以取任意值，梯度为 $c_1[1] = -1.2$
因此，割平面的系数为 $[-3, -1.2]$，右侧常数项为 $-\pi \cdot 0.75 = -(-0.8) \cdot 0.75 = -0.6$

割平面：$\theta \geq -0.6 - 3 \cdot x_1 - 1.2 \cdot x_2$

### **步骤 3：求解主问题**

主问题现在包含两个割平面：

$$ \begin{align} \min & \quad \theta \\ \text{s.t.} & \quad x_1 + x_2 = 1 \\ & \quad \theta \geq -0.75 - x_1 - 1.2x_2 \\ & \quad \theta \geq -0.6 - 3x_1 - 1.2x_2 \\ & \quad x_1, x_2 \geq 0 \end{align} $$

求解得：$x = [0.075, 0.925], \theta = -1.935$ ，更新下界为 -1.935.

```
UB: -1.8
LB: -1.935
```

---

## **第三次迭代**

### **步骤1：求解子问题**
给定新的 $x = [0.075, 0.925]$，我们求解子问题：
$$ \begin{align} \min & \quad 0.075 \cdot (-y_1-1) + 0.925 \cdot (-0.8y_2-1.2) \\ \text{s.t.} & \quad 0.075 \cdot y_1 + 0.925 \cdot y_2 \leq 0.75 \\ & \quad 0 \leq y_1, y_2 \leq 10 \end{align} $$
 **信息如下：**
 $y_1 = 10$，$y_2 = 0$
存在一个**可行解**为 $x_{1}=0.075,x_{2}=0.925,y_{1}=10,y_{2}=0$, $obj=-1.935$ 

LB = UB
Finish it!！

# **CODE**
[PanYuxn/StochasticProgramming: 《Introduction to Stochastic Programming》代码复现，随机优化学习中。](https://github.com/PanYuxn/StochasticProgramming/tree/main)
Char 5-Solutionmethod\GeneralizedBendersDecompostion.Py