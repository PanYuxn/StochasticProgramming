本仓库用来实现书本《Introduction to Stochastic Programming》当中所有模型和代码细节,如果有兴趣欢迎交流。

记录一下序论里的一句话: The art of Managing is in foreseeing.

章节进度:4/10

# CHAR1

Char1.1 - Farming Problem
* 难点1：Page13和Page14的推导，这部分推导当时看的云里雾里，最后总算全部推完了，具体推导见`FarmingExample.ipynb`最后一个单元格笔记.
* 难点2：Page9出现的`VSS`和`EVPI`概念，这里的`VSS`种出现的数值107240有点懵，后来看了资料才理解具体的含义，资料链接参考Reference文件夹中的资料2.

Char1.2 - Financial Problem
* 难点1：多阶段决策情况下决策变量的设置技巧，这里主要要关注每次决策都可以用1或者2表示，来说明走向不同的分支。

Char1.3 - Capacity Expansion Problem
* 难点1：对于不同mode情况下的理解，这里需要注意，不同mode表示的是不同种类的发电情况的不同。（比如核发电的`开发成本`高，但是`运营成本`低；火力发电的`开发成本`低，但是`运营成本`高）
* 难点2：书本Page33页没有按照直观的分配场景`s`，导致在模型复现的过程中有点摸不着头脑，最后自己一个个理解了约束并且增加了场景集合`s`后，代码结果运行一样了。

Char1.4 - ManufacturingMQuality Problem
* 难点1：问题理解，刚开始其实没怎么看懂这个问题要解决什么，后面针对具体变量进行查看后才了解基本含义。本来想尝试用gurobi将模型复现，但是发现这中间存在非线性部分，暂列到todolist，后续再看。

Char1.5 - Routing Problem
* 难点1：对于三种不同情况下的Routing Problem解法的理解。

# CHAR2 

Char2.4 - Two Stage Program with fixed recourse
* 难点1： 对于原始的UFLP问题，在不确定性情况下，引申出了很多变种，想全部理解其实有点困难。（其中有一个很有意思的事情，如果随机性不对决策产生干扰，那么我们可以用Exception帮他当成确定性求解）

Char2.6 - Implicit Representation of the Second Stage
* 难点1：对于急救车调度问题，其实到现在还没懂$w(x)$的含义，估计是排队论相关的知识，到时候补一点相关知识。

Char2.8 - Modeling Exercise
* 难点1：这个章节比较有趣，它将一个原问题通过多种不确定性呈现了出来，让大家从不同方面来观察这个问题。（同时用两种Formualtion来表示，具体来说是one-stage和second-stage）

# Char3

看懂了其中几个定理，其他定理的推导，ennn，完全很难能够跟上，但是有那么一点感觉。

# Char4 

* 这里主要描述了随机性的价值，也就是我们为社么要研究这个问题？通过三种不同的formulation之间的比较，然后给出了一些定义，说明了其有效性。

# Char5 

这一章节写了`Generalized Benders Decomposition`和对应的具体代码实现，很简单的一个小case，同时记录了一下L-shaped算法。

# Char6

实现了`Nested L-shaped`算法，其中case2对应的是书本中的例子，AsimpleExample.jpg对应的是其迭代过程。


邮箱:shhspyx@163.com
