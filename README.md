本仓库用来实现书本《Introduction to Stochastic Programming》当中所有模型和代码细节,如果有兴趣欢迎交流。

记录一下序论里的一句话: The art of Managing is in foreseeing.

章节进度:1/10

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

邮箱:shhspyx@163.com
