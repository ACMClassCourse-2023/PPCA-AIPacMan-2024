#### 介绍

在这个项目中，你将为经典版 Pacman 设计 agent，包括幽灵。

代码库与之前的项目相比没有太大变化，但请从全新开始，而不是混合项目 1 中的文件。

你需要补全的代码文件有：

- `multiAgents.py` 

你可以阅读并参考来帮助你实现代码的文件有：

- `pacman.py`
- `game.py`
- `util.py`

你可以忽略其他支持文件。

#### Q1: Reflex Agent

你需要改进 `multiAgents.py` 中的 `ReflexAgent` 以使其表现出色。提供的 `ReflexAgent` 代码提供了一些查询 `GameState` 信息的方法的示例。与之前不同的是，你每次只需返回一个动作，而不是一个动作列表。详情请见代码。

一个好的 `ReflexAgent` 需要同时考虑食物位置和幽灵位置以取得好的表现。你的 agent 应该能够轻松且可靠地通过 `testClassic` 布局测试。
```
python pacman.py -p ReflexAgent -l testClassic
```

在默认布局 `mediumClassic` 上尝试使用一个或两个幽灵来执行反射 agent （并关闭动画以加快显示速度）：
```
python pacman.py --frameTime 0 -p ReflexAgent -k 1
python pacman.py --frameTime 0 -p ReflexAgent -k 2
```

注意：`newFood` 具有 `asList()` 功能。

注意：作为特征，尝试重要值的倒数（例如到食物的距离），而不仅仅是值本身。

注意：你可能会发现查看各种对象的内部内容对于调试很有用。你可以通过打印对象的字符串表示来实现这一点。例如，你可以对 `newGhostStates` 使用打印 `print(newGhostStates)`。

选项：默认幽灵是随机的；你也可以使用 `-g DirectionalGhost` 来玩一些更智能的定向幽灵。如果随机性阻止你判断 agent 是否正在改进，你可以使用 `-f` 以固定的随机种子运行（每场游戏都使用相同的随机选择）。你还可以使用 `-n` 连续玩多场游戏。使用 `-q` 关闭图形以快速运行大量游戏。

评分：我们将在 `openClassic` 布局上运行你的 agent 10 次。如果你的 agent 超时或从未获胜，你将获得 0 分。如果你的 agent 至少获胜 5 次，你将获得 1 分，如果你的 agent 赢得所有 10 场比赛，你将获得 2 分。如果你的 agent 的平均得分大于 500，你将获得额外的 1 分，如果大于 1000，你将获得 2 分。

运行以下命令来查看你的实现是否通过了所有自动评分测试用例。

```
python autograder.py -q q1
```

要运行不带图形的程序，请使用：

```
python autograder.py -q q1 --no-graphics
```

不过，不要在这个问题上花费太多时间，因为项目的核心还在后面。

#### Q2: Minimax

[Q2 和 Q3 前置知识介绍](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/blob/main/docs/minimax.md)

现在，你需要在提供的 `multiAgents.py` 文件中的 `MinimaxAgent` 类中编写一个对抗搜索 agent。你的 minimax agent 应该适用于任意数量的幽灵，因此你需要编写一个比介绍中所见的稍微更通用的算法。特别地，对于每一层最大化节点（max layer），你的 Minimax 树将有多个最小化节点（min layer）（每个幽灵一个）。

你的代码还应该将游戏树扩展到任意深度。使用提供的 `self.evaluationFunction` 来评估 minimax 树的叶节点，该函数默认为 `scoreEvaluationFunction`。`MinimaxAgent` 继承了 `MultiAgentSearchAgent`，这使得你可以访问 `self.depth` 和 `self.evaluationFunction`。确保你的 minimax 代码在适当的地方引用了这两个变量，因为这些变量是根据命令行选项填充的。

**重要提示**：一次搜索轮次被认为是一次吃豆人（Pacman）移动和所有幽灵的回应，所以深度为 2 的搜索将涉及吃豆人和每个幽灵移动两次（参见下图）。

![](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/minimax_depth.png)

评分：我们将检查你的代码，以确定它是否探索了正确数量的游戏状态。这是检测 minimax 实现中某些非常微妙的错误的唯一可靠方法。因此，自动评分器对你调用 `GameState.generateSuccessor` 的次数会非常挑剔。如果你调用的次数多于或少于必要的次数，自动评分器将会报错。要测试和调试你的代码，请运行：

```
python autograder.py -q q2
```

要在没有图形的情况下运行它，请使用：

```
python autograder.py -q q2 --no-graphics
```

提示和观察：

- 使用辅助函数递归地实现算法。
- minimax 算法的正确实现会导致吃豆人在某些测试中输掉游戏。这不是问题，因为这是正确的行为，它会通过测试。
- 这一部分的吃豆人测试的评估函数已经写好（`self.evaluationFunction`）。你不应该更改这个函数。
- **吃豆人总是 agent 0**，agent 按 agent 索引递增的顺序移动。
- minimax 中的所有状态都应该是 `GameStates`，要么传递给 `getAction`，要么通过 `GameState.generateSuccessor` 生成。在这个项目中，你不会抽象到简化的状态。
- 在更大的场景下，如 `openClassic` 和 `mediumClassic`（默认），你会发现吃豆人很擅长不死，但很难赢。他常常会毫无进展地四处游荡。有时，他甚至会在一个豆子旁边游荡而不吃掉它，因为他不知道吃掉那个豆子后要去哪儿。如果你看到这种行为，不要担心，第五个问题会解决这些问题。
- 当吃豆人认为他的死亡是不可避免的时，他会试图尽快结束游戏，因为存在持续的生存惩罚。有时，对于随机幽灵来说，这是错误的做法，但 minimax agent 总是假设最坏情况：
  ```
  python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
  ```
    确保你理解为什么在这种情况下吃豆人会冲向最近的幽灵。

#### Q3 : Alpha-Beta Pruning

在 `AlphaBetaAgent` 中编写一个使用 alpha-beta 剪枝以更高效地探索 minimax 树的新 agent 。同样，你的算法将比介绍中的伪代码稍微更通用，因此，部分挑战在于适当地将 alpha-beta 剪枝逻辑扩展到多个最小化 agent 。

你应该能看到速度提升（也许深度为 3 的 alpha-beta 剪枝运行速度能与深度为 2 的 minimax 相当）。理想情况下，在 smallClassic 布局中，深度 3 的每次移动应该只需几秒钟或更快。

```
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
```

`AlphaBetaAgent` 的 minimax 值应该与 `MinimaxAgent` 的 minimax 值相同，尽管它选择的动作可能会有所不同，因为它们的平局决策行为不同。

评分：因为我们检查你的代码是否探索了正确数量的状态，所以重要的是你在不重新排序子节点的情况下执行 alpha-beta 剪枝。换句话说，后继状态应该始终按 `GameState.getLegalActions` 返回的顺序处理。同样，不要多于必要地调用 `GameState.generateSuccessor`。

为了匹配我们的自动评分器探索的状态集，**你不能在相等时进行剪枝**。（确实，另一种方法是允许在相等时进行剪枝，并在根节点的每个子节点上调用一次 alpha-beta，但这将不符合我们的自动评分器。）

下面的伪代码表示你应该为这个问题实现的算法。

![](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/alpha_beta_impl.png)

要测试和调试代码，请运行

```
python autograder.py -q q3
```

要在没有图形的情况下运行它，请使用：

```
python autograder.py -q q3 --no-graphics
```

正确实施 alpha-beta 剪枝将导致 Pacman 输掉部分测试。这不是问题：因为它是正确的行为，所以它将通过测试。

#### Q4: Expectimax

minimax 和 alpha-beta 算法都很棒，但它们都假设你在与一个做出最优决策的对手对战。任何赢过井字游戏的人都会告诉你，这并不总是如此。在这个问题中，你将实现 `ExpectimaxAgent`，它对于模拟可能做出次优选择的 agent 的概率行为非常有用。

随机幽灵当然不是最优的 minimax agent ，因此用 minimax 搜索来模拟它们可能不合适。`ExpectimaxAgent` 不再对所有幽灵动作取最小值，而是根据你对幽灵行为模型的期望值。为了简化你的代码，假设你只会对一个在其 `getLegalActions` 中均匀随机选择的对手运行。

你可以使用以下命令在小型游戏树上调试你的实现：

```
python autograder.py -q q4
```

建议在这些小而易管理的测试用例上进行调试，这将帮助你快速找到错误。

一旦你的算法在小型树上运行正常，你可以在吃豆人中观察其成功情况。要观察 `ExpectimaxAgent` 在吃豆人中的表现，请运行：

```
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

你现在应该会观察到在幽灵近距离接触时，`ExpectimaxAgent` 会采取更大胆的策略。特别是，如果吃豆人认为自己可能被困住但可以逃脱以抓住更多的食物，他至少会尝试。研究以下两种情境的结果：

```
python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10
python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10
```

你应该会发现你的 `ExpectimaxAgent` 大约能赢一半，而你的 `AlphaBetaAgent` 总是输。确保你理解为什么这里的行为与 minimax 情况不同。

正确实现的 expectimax 将导致吃豆人在一些测试中输掉比赛。这不是问题：因为这是正确的行为，它会通过测试。

#### Q5: Evaluation Function

在提供的 `betterEvaluationFunction` 函数中为吃豆人编写一个更好的评估函数。使用深度为 2 的搜索时，你的评估函数应该能在 `smallClassic` 布局中面对一个随机幽灵时胜率超过一半，并且运行速度合理（要获得满分，吃豆人在获胜时的平均得分应约为 1000 分）。

评分：自动评分器将在 `smallClassic` 布局上运行你的 agent 10 次。我们将以以下方式给你的评估函数打分：

- 如果你至少赢一次且没有超时，你将获得1分。任何不满足这些标准的 agent 将获得0分。
- 赢至少5次加1分，赢所有10次加2分。
- 平均得分至少500加1分，平均得分至少1000加2分（包括输掉比赛的得分）。
- 如果在自动评分器机器上运行时，游戏平均用时少于30秒，加1分（使用 --no-graphics 运行）。
- 平均得分和计算时间的额外分数仅在你至少赢5次的情况下才会被授予。
  
请不要复制项目1中的任何文件，因为它们不会通过。你可以在以下条件下尝试你的 agent：

```
python autograder.py -q q5
```

要无图形界面运行，请使用：

```
python autograder.py -q q5 --no-graphics
```