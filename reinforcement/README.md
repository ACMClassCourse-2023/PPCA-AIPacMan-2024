#### 介绍

在这个项目中，你将实现值迭代和 Q 学习。你将首先在 Gridworld（网格世界） 上测试你的 agent，然后将它们应用于模拟机器人控制器（Crawler）和吃豆人（Pacman）。

像之前的项目一样，这个项目包含一个自动评分器，你可以在自己的机器上对你的解决方案进行评分。可以通过以下命令运行所有问题的评分：

```
python autograder.py
```

也可以只针对某个特定问题运行评分，例如q2：

```
python autograder.py -q q2
```

还可以通过以下形式的命令针对某个特定测试运行评分：

```
python autograder.py -t test_cases/q2/1-bridge-grid
```

你需要补全的代码文件有：

- `valueIterationAgents.py`：用于解决已知MDP的值迭代 agent。
- `qlearningAgents.py`：用于Gridworld、Crawler和Pacman的Q学习 agent。
- `analysis.py`：用于填写项目中提出的问题的答案。


你可以阅读并参考来帮助你实现代码的文件有：

- `mdp.py`：定义一般MDP的方法。
- `learningAgents.py`：定义基类ValueEstimationAgent和QLearningAgent，你的 agent 将扩展这些类。
- `util.py`：实用工具，包括util.Counter，对Q学习者特别有用。
- `gridworld.py`：Gridworld的实现。
- `featureExtractors.py`：用于提取（状态，动作）对的特征的类。用于近似Q学习 agent（在 `qlearningAgents.py` 中）。

你可以忽略其他支持文件。

#### MDPs

要开始，请以手动控制模式运行 Gridworld，使用箭头键控制：

```
python gridworld.py -m
```

你将看到双出口布局。蓝点是 agent。注意，当你按向上键时， agent 实际上只有 80% 的情况会向北移动。这就是 Gridworld agent 的生活！

你可以控制模拟的许多方面。运行以下命令可以查看完整的选项列表：

```
python gridworld.py -h
```

默认 agent 随机移动

```
python gridworld.py -g MazeGrid
```

你应该会看到随机 agent 在网格中四处碰撞，直到偶然找到出口。这并不是 AI agent 最光辉的时刻。

注意：Gridworld MDP 的特点是你必须先进入一个预终止状态（GUI中显示的双框）然后执行特殊的“出口”动作，才能真正结束这一轮（真正的终止状态称为TERMINAL_STATE，不在GUI中显示）。如果你手动运行一轮，最终回报可能会比预期少，因为有折扣率（用 `-d` 来改变；默认是0.9）。

查看图形输出伴随的控制台输出（或使用 `-t` 显示所有文本）。你会被告知 agent经历的每次转变（要关闭这个提示，使用 `-q`）。

与吃豆人类似，位置用 `(x, y)` 笛卡尔坐标表示，任何数组按 `[x][y]` 索引，`north` 方向是 `y` 增加的方向，等等。默认情况下，大多数转变的奖励为零，尽管你可以用生存奖励选项（`-r`）来改变这个设置。

#### Q1: Value Iteration

回顾价值迭代的状态更新方程：

\[ V_{k+1}(s) \leftarrow \max_{a} \sum_{s'} T(s, a, s') \left[ R(s, a, s') + \gamma V_k(s') \right] \]

请在 `valueIterationAgents.py` 中编写一个值迭代代理（ValueIterationAgent），该文件已经部分为你指定。你的值迭代代理是一个离线规划器，而不是一个强化学习代理，因此相关的训练选项是在初始规划阶段运行的价值迭代次数（选项 `-i`）。ValueIterationAgent 在构造时接受一个 MDP，并在构造函数返回之前运行指定次数的价值迭代。

价值迭代计算最优值的 $ k $ 步估计 $ V_k $。除了运行 `runValueIteration` 外，还要实现以下方法来为 `ValueIterationAgent` 使用 $ V_k $：

- `computeActionFromValues(state)` 根据由 `self.values` 给出的值函数计算最佳动作。
- `computeQValueFromValues(state, action)` 根据由 `self.values` 给出的值函数返回状态-动作对的 Q 值。

这些量都会显示在 GUI 中：值显示为方块中的数字，Q 值显示为方块四分之一中的数字，策略显示为从每个方块延伸出来的箭头。

重要提示：使用价值迭代的“批处理”版本，其中每个向量 $ V_k $ 是从固定向量 $ V_{k-1} $ 计算的，而不是“在线”版本，其中单个权重向量是就地更新的。这意味着当在迭代 $ k $ 中基于其后续状态的值更新状态的值时，所使用的后续状态值应来自迭代 $ k-1 $（即使某些后续状态已经在迭代 $ k $ 中更新）。[Sutton & Barto](https://web.archive.org/web/20230417150626/https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) 在第 4.1 章第 91 页讨论了两者的区别。

注意：从深度为 $ k $ 的值合成的策略（反映接下来的 $ k $ 次奖励）实际上会反映接下来的 $ k+1 $ 次奖励（即你返回 $ \pi_{k+1} $）。同样，Q 值也将比值多反映一次奖励（即你返回 $ Q_{k+1} $）。

你应该返回合成的策略 $ \pi_{k+1} $。

提示：你可以选择使用 `util.py` 中的 `util.Counter` 类，它是一个默认值为零的字典。但是，请注意 `argMax`：你想要的实际 argmax 可能是计数器中没有的键！

注意：确保处理在 MDP 中状态没有可用动作的情况（想一想这对未来奖励意味着什么）。

要测试你的实现，请运行自动评分程序：

```
python autograder.py -q q1
```

以下命令加载你的 ValueIterationAgent，它将计算一个策略并执行 10 次。按一个键可以循环显示值、Q 值和模拟。你应该发现起始状态的值（V(start)，可以从 GUI 读取）和执行 10 轮后打印的经验平均奖励非常接近。

```
python gridworld.py -a value -i 100 -k 10
```

提示：在默认的 BookGrid 上运行 5 次价值迭代应该会给你以下输出：

```
python gridworld.py -a value -i 5
```

![](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/value_iter_diagram.png)