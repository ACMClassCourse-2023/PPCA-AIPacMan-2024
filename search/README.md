#### 介绍

接下来这段时间，你将用搜索算法实现吃豆人的行为。

你需要补全的代码文件有：

- `search.py` 
- `searchAgents.py` 

你可以阅读并参考来帮助你实现代码的文件有：

- `pacman.py`
- `game.py`
- `util.py`

你可以忽略其他支持文件。

运行 `python pacman.py` 即可启动吃豆人游戏，你可以用你的键盘操作吃豆人并探索。

在 `searchAgents.py` 中，将会存在不同的 `agent` 类来决定吃豆人的行为模式。它规划出一条吃豆人穿越迷宫的路径，然后逐步执行该路径。

比如 `GoWestAgent` 会让吃豆人一路向西。你可以运行 `python pacman.py --layout testMaze --pacman GoWestAgent` 查看。

显然他会遇到糟糕的情况：`python pacman.py --layout tinyMaze --pacman GoWestAgent`。

另一个 `SearchAgent` 的例子：`python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch`。这次是 `tinyMazeSearch`。

但是，寻找路径的算法其实还未实现，这是你的任务。不久后，你的吃豆人 agent 或许将所向披靡！

运行 `python pacman.py -h` 可以查看相关参数列表。

#### Q1：DFS

是时候编写成熟的通用搜索函数来帮助吃豆人规划路线了。在这个场景里，地图中只有一个目标豆子。

你需要实现 `search.py` 中的 `depthFirstSearch` 函数，使用深度优先搜索来规划一条吃到豆子的路线。

**重要提示**：所有搜索函数都需要返回一个操作列表，这些操作将引导吃豆人从起点到达目标。这些操作都必须是合法的移动（有效方向，不穿过墙壁）。

**重要提示**：请务必使用 `util.py` 中提供的 `Stack` , `Queue` 和 `PriorityQueue` 数据结构！这些数据结构实现具有与自动评分器兼容所需的特定属性。

提示：在实现前，可以仔细阅读函数内的注释，和 `SearchProblem` 类定义的接口。

如果你成功实现，应该可以快速解决：

```
python pacman.py -l tinyMaze -p SearchAgent
python pacman.py -l mediumMaze -p SearchAgent
python pacman.py -l bigMaze -z .5 -p SearchAgent
```

迷宫将显示已探索状态的叠加图，以及探索顺序（红色越亮表示探索越早）。探索顺序是否符合你的预期？吃豆人在到达目标的途中是否真的会经过所有已探索的方格？

提示：如果你使用 `Stack` 作为数据结构，则 DFS 算法找到的 `mediumMaze` 的解决方案的长度应为 130（前提是你按照 `getSuccessors` 提供的顺序探索后继者；如果按相反顺序推入它们，则可能会得到 246）。这是最低成本解决方案吗？如果不是，请考虑深度优先搜索做错了什么。

运行以下命令来查看你的实现是否通过了所有自动评分测试用例。

```
python autograder.py -q q1
```

#### Q2：BFS

同 DFS，用深度优先搜索实现 `search.py` 中的 `breadthFirstSearch` 函数。

运行：

```
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
```

注意：如果你已经通用地编写了搜索代码，那么你的代码无需进行任何更改就应该同样适用于八谜题搜索问题。

```
python eightpuzzle.py
```

运行以下命令来查看你的实现是否通过了所有自动评分测试用例。

```
python autograder.py -q q2
```

#### Q3：改变成本函数

虽然 BFS 会找到一条到达目标的最少操作路径，但我们可能希望找到其他意义上“最佳”的路径。考虑 `mediumDottedMaze` 和 `mediumScaryMaze` 这两个地图。通过改变成本函数，我们可以鼓励吃豆人寻找不同的路径。例如，我们可以对鬼魂出没地区的危险步骤收取更多费用，或对食物丰富地区的步骤收取更少费用，而理性的吃豆人 agent 应该会据此调整其行为。

你需要在 `search.py` 中的 `uniformCostSearch` 函数中实现均匀成本图搜索算法。我们鼓励你仔细查看 `util.py` 一些可能对你的实现有用的数据结构。现在你应该在以下所有三种地图中观察到成功的行为，其中下面的 agent 都是 UCS agent，它们仅在使用的成本函数上有所不同（agent 和成本函数已经为你编写好）：

```
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
python pacman.py -l mediumScaryMaze -p StayWestSearchAgent
```

提示：对于 `StayEastSearchAgent` 和 `StayWestSearchAgent`， 你应该分别获得非常低和非常高的路径成本（有关详细信息，请参阅`searchAgents.py`）。

运行以下命令来查看你的实现是否通过了所有自动评分测试用例。

```
python autograder.py -q q3
```

#### Q4：A* 搜索

[介绍](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/blob/main/docs/A*.md)

简单介绍启发式搜索：有一个启发式函数 $h$，在搜索时优先搜索值最小的方向。

在 `search.py` 中的函数 `aStarSearch` 中实现 A\* 搜索。A\* 以启发式函数为参数。启发式函数有两个参数：搜索问题中的状态（主要参数）和问题本身（用于参考）。`search.py` 中的 `nullHeuristic` 启发式函数就是一个简单的例子。

你可以使用曼哈顿距离启发式算法（已在 `searchAgents.py` 中实现 `manhattanHeuristic`）测试你的 A* 实现。该算法针对的是原来的问题（即寻找穿过迷宫到达固定位置的路径）。

```
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=astar,heuristic=m
anhattanHeuristic
```

你应该看到 A* 找到最佳解决方案的速度比统一成本搜索略快。

运行以下命令来查看你的实现是否通过了所有自动评分测试用例。

```
python autograder.py -q q4
```

#### Q5：找到所有角落

注意：在回答问题 5 之前，请确保先完成问题 2，因为问题 5 是以问题 2 的回答为基础的。

A* 的真正威力只有在更具挑战性的搜索问题中才会显现出来。

在角落迷宫中，有四个点，每个角落一个。我们的新搜索问题是找到穿过迷宫的最短路径，该路径接触所有四个角落（无论迷宫中是否有食物）。

在 `searchAgents.py` 中实现搜索问题 `CornersProblem`。你需要选择一个状态（ `state`）的表示，用于编码 判断是否已到达所有四个角 所需的所有信息。现在，你的搜索 agent 应该解决：

```
python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
```

要获得满分，你需要定义一个抽象的状态表示，该表示不会编码无关信息（例如鬼魂的位置、额外食物的位置等）。特别是，不要使用吃豆人 `GameState` 作为搜索状态。如果你这样做（并且出错），你的代码将非常非常慢。

类的一个实例 `CornersProblem` 代表整个搜索问题，而不是特定状态（`state`）。特定状态由你编写的函数返回你选择用于表示状态的数据结构（例如元组、集合等）。

提示 1：你在实现中仅需要参考的游戏状态：吃豆人的起始位置和四个角的位置。

提示 2：编写 `getSuccessors` 代码时，请确保将子代添加到后继列表中，成本为 1。

在 `mediumCorners` 中，我们的 `breadthFirstSearch` 实现将搜索节点扩展到近 2000 个。但是，启发式方法（与 A* 搜索一起使用）可以减少所需的搜索量。

运行以下命令来查看你的实现是否通过了所有自动评分测试用例。

```
python autograder.py -q q5
```

#### Q6：角落问题：启发式

注意：在回答问题 6 之前，请确保先完成问题 4，因为问题 6 是以问题 4 的回答为基础的。

在 `CornersProblem` 中的 `cornersHeuristic` 中实现一个非平凡的、一致的启发式方法

```
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```

注意：`AStarCornersAgent` 是一个快捷方式：

```
-p SearchAgent -a fn=aStarSearch,prob=CornersProblem,heuristic=cornersHeuristic
```

**可接受性 vs 一致性**：请记住，启发式函数只是将搜索状态作为输入并返回估计到最近目标的代价的数值的函数。更有效的启发式函数将返回更接近实际目标代价的值。为了是可接受的，启发式函数的值必须是到最近目标的实际最短路径代价的下界（且为非负值）。为了是一致的，还必须满足这样一个条件：如果一个动作的代价是 c，那么采取该动作只能导致启发式函数值下降最多 c。

在图搜索中，可接受性不足以保证正确性——你需要更强的一致性条件。然而，可接受的启发式函数通常也是一致的。因此，通常最简单的方法是先构思出可接受的启发式函数。一旦你有了一个效果良好的可接受启发式函数，你可以检查它是否也确实一致。唯一能保证一致性的方法是通过证明。然而，不一致性通常可以通过验证每个被扩展节点的后继节点的 f 值是否相等或更高来检测。此外，如果 UCS 和 A* 返回的路径长度不同，那么你的启发式函数就是不一致的。

```
python pacman.py -l mediumCorners -p SearchAgent -a fn=aStarSearch,prob=CornersProblem,heuristic=cornersHeuristic -z 0.5
python pacman.py -l mediumCorners -p SearchAgent -a fn=aStarSearch,prob=CornersProblem,heuristic=nullHeuristic -z 0.5
```

**非平凡启发式**：平凡启发式是到处都返回零（UCS）的启发式和计算真实完成成本的启发式。前者不会为你节省任何搜索，而后者会使程序超时。你需要一种可以减少总计算时间的启发式。

**评分**：你的启发式方法必须是非平凡的非负一致性启发式方法。确保你的启发式方法在每个目标状态都返回 0，并且永远不会返回负值。根据你的启发式方法扩展的节点数，你将获得以下评分：

| **搜索的节点数** | 得分 |
| ---------------- | ---- |
| >2000            | 0/3  |
| <=2000           | 1/3  |
| <=1600           | 2/3  |
| <=1200           | 3/3  |

请记住：如果你的启发式方法不一致，你将不会获得任何分数，所以要小心！

运行以下命令来查看你的实现是否通过了所有自动评分测试用例。

```
python autograder.py -q q6
```

#### Q7：吃掉所有点

注意：在回答问题 7 之前，请确保先完成问题 4，因为问题 7 是以问题 4 的回答为基础的。

现在我们将解决一个困难的搜索问题：以尽可能少的步骤吃掉所有的吃豆人食物。为此，我们需要一个新的搜索问题定义，该定义将食物清除问题形式化：在 `searchAgents.py` 中的 `FoodSearchProblem`（已经为你实现）。一个解决方案被定义为在吃豆人世界中收集所有食物的路径。在当前的项目中，解决方案不考虑任何幽灵或能量豆；解决方案仅依赖于墙壁、普通食物和吃豆人的位置。（当然，幽灵可能会破坏解决方案的执行！我们将在下一个项目中讨论这个问题。）如果你正确编写了通用搜索方法，使用一个空启发式函数的A*算法（相当于统一代价搜索）应该能够快速找到testSearch的最优解决方案，而无需修改代码（总成本为7）。

```
python pacman.py -l testSearch -p AStarFoodSearchAgent
```

注意：`AStarFoodSearchAgent` 是一个快捷方式：

```
-p SearchAgent -a fn=astar,prob=FoodSearchProblem,heuristic=foodHeuristic
```

你会发现，即使是看似简单的 `tinySearch`，统一代价搜索（UCS）也会开始变慢。作为参考，我们的实现花了2.5秒钟找到了长度为27的路径，扩展了2372个搜索节点。

```
python pacman.py -l tinySearch -p AStarFoodSearchAgent
```

在 `searchAgents.py` 中完成 `foodHeuristic` 函数，为 FoodSearchProblem 提供一个一致的启发式函数。然后在 `trickySearch` 上测试：

```
python pacman.py -l trickySearch -p AStarFoodSearchAgent
```

我们的 UCS agent 在大约 13 秒内找到了最佳解决方案，探索了超过 16,000 个节点。

```
python pacman.py -l trickySearch -p SearchAgent -a fn=ucs,prob=FoodSearchProblem
```

确保你的启发式在每个目标状态都返回 0，并且永远不会返回负值。

| **搜索的节点数** | 得分 |
| ---------------- | ---- |
| >15000           | 1/4  |
| <=15000          | 2/4  |
| <=12000          | 3/4  |
| <=9000           | 4/4  |
| <=7000           | 5/4  |

请记住：如果你的启发式方法不一致，你将不会获得任何分数，所以要小心！

运行以下命令来查看你的实现是否通过了所有自动评分测试用例。

```
python autograder.py -q q7
```

#### Q8：次优搜索

有时，即使使用 A* 和一个好的启发式函数，找到所有点的最优路径也是困难的。在这种情况下，我们仍希望能够快速找到一条相对较好的路径。在这一部分，你将编写一个 agent，它总是贪婪地吃掉最近的点。`ClosestDotSearchAgent` 已在 `searchAgents.py` 中实现，但缺少一个找到最近点路径的关键函数。

在 `searchAgents.py` 中实现函数 `findPathToClosestDot`。我们的 agent 能够在不到一秒钟的时间内以 350 的路径代价，次优地解决了这个迷宫：

```
python pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5
```

提示：完成 `findPathToClosestDot` 的最快方法是填写 `AnyFoodSearchProblem` 中缺少的目标测试 `isGoalState`。然后，用适当的搜索函数解决这个问题。解决方案应该非常简短！

你的 `ClosestDotSearchAgent` 并不总是能找到穿过迷宫的最短路径。请确保你理解原因，并尝试提出一个小例子，在这个例子中，反复寻找最近的点并不能找到吃掉所有点的最短路径。（不计分）

运行以下命令来查看你的实现是否通过了所有自动评分测试用例。

```
python autograder.py -q q7
```
