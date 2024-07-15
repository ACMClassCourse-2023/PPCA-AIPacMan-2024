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
- `model.py`：用于实现深度 Q 学习的神经网络。


你可以阅读并参考来帮助你实现代码的文件有：

- `mdp.py`：定义一般MDP的方法。
- `learningAgents.py`：定义基类ValueEstimationAgent和QLearningAgent，你的 agent 将扩展这些类。
- `util.py`：实用工具，包括util.Counter，对Q学习者特别有用。
- `gridworld.py`：Gridworld的实现。
- `featureExtractors.py`：用于提取（状态，动作）对的特征的类。用于近似Q学习 agent（在 `qlearningAgents.py` 中）。

你可以忽略其他支持文件。

#### MDPs

[介绍](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/blob/main/docs/MDP.md)

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

$V_{k+1}(s) \leftarrow \max_{a} \sum_{s'} T(s, a, s') \left[ R(s, a, s') + \gamma V_k(s') \right]$

请在 `valueIterationAgents.py` 中编写一个值迭代代理（`ValueIterationAgent`），该文件已经部分为你指定。你的值迭代代理是一个离线规划器，而不是一个强化学习代理，因此相关的训练选项是在初始规划阶段运行的价值迭代次数（选项 `-i`）。`ValueIterationAgent` 在构造时接受一个 MDP，并在构造函数返回之前运行指定次数的价值迭代。

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

以下命令加载你的 `ValueIterationAgent`，它将计算一个策略并执行 10 次。按一个键可以循环显示值、Q 值和模拟。你应该发现起始状态的值（V(start)，可以从 GUI 读取）和执行 10 轮后打印的经验平均奖励非常接近。

```
python gridworld.py -a value -i 100 -k 10
```

提示：在默认的 BookGrid 上运行 5 次价值迭代应该会给你以下输出：

```
python gridworld.py -a value -i 5
```

![](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/value_iter_diagram.png)

评分：你的价值迭代代理将在一个新的网格上进行评分。我们将在固定次数迭代后以及收敛时（例如，经过 100 次迭代后）检查你的值、Q 值和策略。

#### Q2: Policies

考虑 `DiscountGrid` 布局，如下图所示。这个网格有两个带正收益的终端状态（在中间一行），一个接近的出口收益为 +1，另一个较远的出口收益为 +10。网格的底行由带负收益的终端状态组成（红色显示）；在这个“悬崖”区域中的每个状态收益为 -10。起始状态是黄色方块。我们区分两种路径类型：(1) “冒险靠近悬崖”的路径，沿网格底行附近行进；这些路径较短但冒着获得大额负收益的风险，如下图中的红色箭头所示。(2) “避免悬崖”的路径，沿网格顶部边缘行进。这些路径较长，但较少可能遭受巨大负收益，如下图中的绿色箭头所示。

![Gridworld 中的路径](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/value_2_paths.png)

在这个问题中，你需要选择此 MDP 的折扣因子、噪声和生活奖励参数设置，以生成几种不同类型的最优策略。你需要在 `analysis.py` 中的 `question2a()` 到 `question2e()` 应分别返回一个包含（折扣因子`discount`、噪声`noise`、生活奖励`living reward`）的三项元组。**你为每部分选择的参数值应该具有这样的属性：如果你的代理遵循其最优策略且不受任何噪声影响，它将表现出给定的行为。** 如果某个特定行为无法通过任何参数设置实现，请通过返回字符串 `NOT POSSIBLE` 来断言该策略是不可能的。

你应该尝试生成以下最优策略类型：

1. 偏好接近的出口（+1），冒险靠近悬崖（-10）
2. 偏好接近的出口（+1），但避免悬崖（-10）
3. 偏好较远的出口（+10），冒险靠近悬崖（-10）
4. 偏好较远的出口（+10），避免悬崖（-10）
5. 避免所有出口和悬崖（因此情节永不终止）

要查看一组参数产生的行为，请运行以下命令以查看 GUI：

```
python gridworld.py -g DiscountGrid -a value --discount [YOUR_DISCOUNT] --noise [YOUR_NOISE] --livingReward [YOUR_LIVING_REWARD]
```

要检查你的答案，请运行自动评分程序：

```
python autograder.py -q q2
```

注意：你可以在 GUI 中检查你的策略。例如，使用 3(a) 的正确答案，（0,1）中的箭头应指向东，（1,1）中的箭头也应指向东，（2,1）中的箭头应指向北。

注意：在某些机器上，你可能看不到箭头。在这种情况下，请按键盘上的任意键切换到 qValue 显示，并通过对每个状态的可用 qValue 取最大值来算出策略。

评分：我们将检查在每种情况下是否返回了期望的策略。

#### Q3: Q-Learning

注意，你的价值迭代代理实际上并不会从经验中学习。相反，它在实际与环境交互之前就思考其 MDP 模型制定出完整的策略。当它与环境交互时，它只需遵循预先计算好的策略（例如，它变成了一个反射代理）。在像 Gridworld 这样的模拟环境中，这种区别可能很微妙，但在真实世界中，这一点非常重要，因为真实的 MDP 是不可用的。

你现在将编写一个 Q-learning 代理，该代理在构造时几乎不做什么，而是通过其 `update(state, action, nextState, reward)` 方法从与环境的交互中通过试错学习。在 `qlearningAgents.py` 中的 `QLearningAgent` 指定了一个 Q-learner 的框架，你可以通过选项 `-a q` 选择它。对于这个问题，你必须实现 `update`、`computeValueFromQValues`、`getQValue` 和 `computeActionFromQValues` 方法。

注意：对于 `computeActionFromQValues`，为了获得更好的行为，你应该随机打破平局。`random.choice()` 函数会有帮助。在特定状态下，代理尚未见过的动作仍然具有 Q 值，具体来说是零的 Q 值，如果代理见过的所有动作都具有负的 Q 值，则未见过的动作可能是最优的。

重要：确保在 `computeValueFromQValues` 和 `computeActionFromQValues` 函数中，你只通过调用 `getQValue` 来访问 Q 值。这种抽象在当你在问题 10 中 override `getQValue` 函数以使用状态-动作对的特征，而不是直接使用状态-动作对时会很有用。

在实现 Q-learning 更新后，你可以使用键盘手动控制观看你的 Q-learner 学习：

```
python gridworld.py -a q -k 5 -m
```

回想一下，`-k` 将控制代理学习的剧集数量。观看代理如何学习它刚刚处于的状态，而不是它移动到的状态，并“在其后留下学习的痕迹”。提示：为帮助调试，你可以通过使用 `--noise 0.0` 参数来关闭噪声（尽管这显然使 Q-learning 不那么有趣）。如果你手动将 Pacman 向北然后沿着最佳路径向东四个剧集，你应该会看到以下 Q 值：

![Q 值图](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/q_learning.png)

评分：我们将运行你的 Q-learning 代理，并检查在每个呈现相同示例集的情况下，它是否学习了与我们的参考实现相同的 Q 值和策略。要评分你的实现，请运行自动评分程序：

```
python autograder.py -q q3
```

#### Q4: Epsilon Greedy

通过在 `getAction` 中实现 epsilon-贪婪策略来完成你的 Q-learning 代理，这意味着它会在 epsilon 的时间内选择随机动作，而在其他时间内则遵循当前最好的 Q 值。注意，选择随机动作可能会导致选择最佳动作——即，你不应该选择一个随机的次优动作，而应该选择任何一个随机的合法动作。

你可以通过调用 `random.choice` 函数从列表中均匀随机地选择一个元素。你可以使用 `util.flipCoin(p)` 来模拟一个成功概率为 `p` 的二元变量，它会以概率 `p` 返回 `True`，以概率 `1-p` 返回 `False`。

在实现 `getAction` 方法后，观察代理在 `GridWorld` 中的以下行为（epsilon = 0.3）：

```
python gridworld.py -a q -k 100
```

你的最终 Q 值应该类似于你的价值迭代代理的 Q 值，尤其是在经常走的路径上。然而，由于随机动作和初始学习阶段，你的平均回报会比 Q 值预测的要低。

你还可以观察以下不同 epsilon 值的模拟。代理的行为是否与预期一致？

```
python gridworld.py -a q -k 100 --noise 0.0 -e 0.1

python gridworld.py -a q -k 100 --noise 0.0 -e 0.9
```

要测试你的实现，请运行自动评分程序：

```
python autograder.py -q q4
```

无需额外代码，你现在应该能够运行一个 Q-learning 的爬行机器人：

```
python crawler.py
```

如果这不起作用，你可能写了一些太具体于 GridWorld 问题的代码，你应该使其更加通用于所有 MDPs。

这将使用你的 Q-learner 调用课堂上的爬行机器人。玩一玩各种学习参数，看看它们如何影响代理的策略和动作。注意，步长延迟是模拟的一个参数，而学习率和 epsilon 是你的学习算法的参数，折扣因子是环境的属性。

#### Q5: Q-Learning and Pacman

现在是时候玩一些吃豆人游戏了！吃豆人将在两个阶段进行游戏。在第一个阶段，即训练阶段，吃豆人将开始学习位置和动作的值。由于即使是微小的网格也需要很长时间才能学习准确的 Q 值，吃豆人的训练游戏默认在静默模式下运行，不显示 GUI（或控制台）显示。训练完成后，吃豆人将进入测试模式。在测试时，吃豆人的 `self.epsilon` 和 `self.alpha` 将设置为 0.0，有效地停止 Q-learning 并禁用探索，以便吃豆人能够利用他学到的策略。默认情况下，测试游戏会在 GUI 中显示。无需任何代码更改，你应该能够运行 Q-learning 吃豆人来进行非常小的网格游戏，如下所示：

```
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

请注意，`PacmanQAgent` 已经基于你编写的 `QLearningAgent` 定义好了。`PacmanQAgent` 只是在默认学习参数上有所不同，这些参数对于吃豆人问题更有效（`epsilon=0.05，alpha=0.2，gamma=0.8`）。如果上述命令能够无异常运行并且你的代理能至少 80%  的时间获胜，你将获得本题的全部分数。自动评分器将在 2000 个训练游戏后运行 100 个测试游戏。

提示：如果你的 `QLearningAgent` 适用于 `gridworld.py` 和 `crawler.py`，但似乎没有为吃豆人在 smallGrid 上学习到一个好的策略，可能是因为你的 `getAction` 和/或 `computeActionFromQValues` 方法在某些情况下没有正确考虑未见过的动作。特别是，因为未见过的动作定义上有一个 Q 值为零，如果所有见过的动作都有负的 Q 值，未见过的动作可能是最佳的。注意 `util.Counter` 中的 `argMax` 函数！

要评分你的答案，运行：

```
python autograder.py -q q5
```

注意：如果你想试验学习参数，可以使用 `-a` 选项，例如 `-a epsilon=0.1,alpha=0.3,gamma=0.7`。这些值将在代理内部作为 `self.epsilon`、`self.gamma` 和 `self.alpha`。

注意：虽然总共将进行 2010 个游戏，但由于 -x 2000 选项，第前 2000 个游戏不会显示输出，因为该选项指定前 2000 个游戏用于训练（无输出）。因此，你只会看到吃豆人玩最后 10 个游戏。训练游戏的数量也会作为 numTraining 选项传递给你的代理。

注意：如果你想观看 10 个训练游戏以查看发生了什么，请使用以下命令：

```
python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10
```

在训练过程中，你将每 100 个游戏看到一次输出，包含有关吃豆人表现的统计数据。在训练期间，epsilon 是正数，因此即使吃豆人已经学习到一个好的策略，他仍然会表现得很差：这是因为他偶尔会随机探索走到幽灵身边。作为基准，大约需要 1000 到 1400 个游戏才能使吃豆人在 100 集段中的奖励变为正数，反映他开始胜过输的次数。到训练结束时，它应该保持为正数且相当高（在 100 到 350 之间）。

确保你理解这里发生的事情：MDP 状态是吃豆人面对的确切棋盘配置，现在的复杂转变描述了该状态的整个变化步骤。吃豆人移动但幽灵未回复的中间游戏配置不是 MDP 状态，而是捆绑在转变中的。

一旦吃豆人完成训练，他在测试游戏中应该非常可靠地获胜（至少 90% 的时间），因为现在他在利用他学到的策略。

然而，你会发现相同的代理在看似简单的 `mediumGrid` 上的训练效果不好。在我们的实现中，吃豆人的平均训练奖励在整个训练期间保持为负数。在测试时，他表现得很差，可能会输掉所有测试游戏。尽管训练效率低下，训练也会花很长时间。

吃豆人在较大布局上失败的原因是每个棋盘配置都是一个独立的状态，具有独立的 Q 值。他无法推广“撞到幽灵在所有位置都很糟糕”这一点。显然，这种方法无法扩展。

#### Q6: Approximate Q-Learning

[介绍](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/blob/main/docs/ApproximateQLearning.md)

实现一个近似 Q 学习代理，该代理学习状态特征的权重，其中许多状态可能共享相同的特征。请在 `qlearningAgents.py` 中的 `ApproximateQAgent` 类中编写你的实现，该类是 `PacmanQAgent` 的子类。

注意：近似 Q 学习假定存在一个特征函数 $f(s, a)$ 作用于状态和动作对上，生成一个特征值向量 $[ f_1(s, a), …, f_i(s, a), …, f_n(s, a) ]$。我们在 `featureExtractors.py` 中提供了特征函数。特征向量是 `util.Counter`（类似于字典）的对象，包含非零特征和值的对；所有省略的特征值为零。因此，不是索引定义向量中的哪个特征，而是字典中的键定义特征的身份。

近似 Q 函数具有以下形式：

$Q(s, a) = \sum_{i=1}^n f_i(s, a) w_i$

其中每个权重 $w_i$ 关联到特定特征 $f_i(s, a)$。在你的代码中，你应该将权重向量实现为将特征（特征提取器返回的特征）映射到权重值的字典。你将类似于更新 Q 值的方式更新权重向量：

$w_i \leftarrow w_i + \alpha \cdot \text{difference} \cdot f_i(s, a)$

$\text{difference} = \left( r + \gamma \max_{a'} Q(s', a') \right) - Q(s, a)$

注意 $\text{difference}$ 项与普通 Q 学习中的相同，$r$ 是经历的奖励。

默认情况下，`ApproximateQAgent` 使用 `IdentityExtractor`，它为每个（状态，动作）对分配单个特征。使用此特征提取器时，你的近似 Q 学习代理应与 `PacmanQAgent` 完全相同。你可以使用以下命令测试这一点：

```
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid
```

重要：`ApproximateQAgent` 是 `QLearningAgent` 的子类，因此它共享多个方法，例如 `getAction`。确保 `QLearningAgent` 中的方法调用 `getQValue` 而不是直接访问 Q 值，这样当你在近似代理中重写 `getQValue` 时，将使用新的近似 Q 值来计算动作。

一旦你确信你的近似学习器在身份特征下正确工作，请使用我们的自定义特征提取器运行你的近似 Q 学习代理，它可以轻松地学会获胜：

```
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

即使是更大的布局，你的 `ApproximateQAgent` 也不应有问题（警告：这可能需要几分钟的训练时间）：

```
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic
```

如果没有错误，你的近似 Q 学习代理应该在这些简单特征下几乎每次都能获胜，即使只有 50 个训练游戏。

评分：我们将运行你的近似 Q 学习代理，并检查它在每种情况下学习到的 Q 值和特征权重是否与我们的参考实现相同。要评分你的实现，运行：

```
python autograder.py -q q6
```

#### Q7: Deep Q-Learning

你将结合本项目之前的 Q 学习概念和前一个项目中的机器学习概念, 在 `model.py` 中，你将实现 `DeepQNetwork`，它是一个神经网络，能够在给定状态下预测所有可能动作的 Q 值。

你需要实现以下函数：

1. `__init__()`：就像在项目 5 中一样，你将在这里初始化神经网络的所有参数。你还必须初始化以下变量：
    - `self.parameters`：一个包含你前向传播过程中所有参数的列表（仅当你使用此项目的原始版本时）。
    - `self.learning_rate`：你将在 `gradient_update()` 中使用它。
    - `self.numTrainingGames`：Pacman 将要进行的游戏数量，以收集转换并学习其 Q 值；注意，这个数量应该大于 1000，因为大约前 1000 场游戏用于探索，不用于更新 Q 网络。
    - `self.batch_size`：模型每次梯度更新应使用的转换数量。自动评分器将使用此变量；在设置后你不需要访问此变量。
2. `get_loss()`：返回预测的 Q 值（由你的网络输出）与 `Q_targets`（你将其视为真实值）之间的平方误差。
3. `forward()`：类似于项目 5 中的同名方法，在这里你将返回通过你的 Q 网络进行前向传播的结果。（输出应为大小为 `(batch_size, num_actions)` 的向量，因为我们想返回在给定状态下所有可能动作的 Q 值。）
4. `gradient_update()`：遍历你的 `self.parameters` 并根据计算出的梯度更新每一个参数。然而，与项目 5 不同的是，你不会在这个函数中遍历整个数据集，也不会反复更新参数直到收敛。这个函数应该只为每个参数执行一次梯度更新。自动评分器将反复调用此函数来更新你的网络。

对于 `gradient_update()`，我们建议使用 SGD 优化器，而不是像在机器学习项目中使用的 Adam。两者的用法完全相同，但在这种情况下，SGD 往往表现得更好。你也可以尝试使用不同的优化器，但 SGD 是工作人员的解决方案中使用的，并且表现相对较好。

评分标准：我们将在你的 Deep Q 学习 Pacman 代理在 `self.numTrainingGames` 场游戏后训练后运行 10 场游戏。如果你的代理至少赢得 6/10 场游戏，你将获得满分。如果你的代理至少赢得 8/10 场游戏，你将获得 1 个额外分数（5/4）。请注意，尽管我们在后台训练循环中实施了一些技巧，深度 Q 学习并不以稳定性著称。你的代理每次运行的获胜场次可能会有所不同。要获得额外分数，你的实现应该始终超过 80% 的门槛。

```
python autograder.py -q q7
```