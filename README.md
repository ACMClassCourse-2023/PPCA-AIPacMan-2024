# PPCA-AIPacMan-2024

PPCA AI 吃豆人项目

本项目基于[加州大学伯克利分校的CS 188《Introduction to Artificial Intelligence》课程项目](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/)

## Week 1

### 环境配置

推荐使用 [miniconda](https://docs.anaconda.com/miniconda/)

**创建虚拟环境**（注意在创建时要指定python版本并且自己给环境命名）

  ```cmd
conda create --name pacman python=3.9
conda activate pacman
  ```

**安装相关的工具包**

```
conda install pip
conda install numpy
conda install matplotlib
```

如果下载时间过长，可以使用

  ```cmd
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free  
  ```

指令加载清华源或其他镜像源作为下载路径。

### Python 基础

[推荐阅读](https://www.liaoxuefeng.com/wiki/1016959663602400)

相关文件在 [tutorial]() 文件夹下。

我们提供了一些小测试以供大家熟悉 Python：

- 加法：阅读 `addition.py` 并补全代码。
- 给定水果价格和订单列表，计算总价：阅读 `buyLotsOfFruit.py` 并实现`buyLotsOfFruit(orderList)`函数。
- 计算最低总价：阅读 `shopSmart.py` 并补全 `shopSmart(orderList,fruitShops)` 函数。在 `shop.py` 中查看 `FruitShop` 类的实现。

运行 `python autograder.py` 会评估你对这三个问题的解决方案，运行 `python autograder.py -q q1` 将仅测试第一个问题。

### Pac-Man Agents

>你说的对，但是《吃豆人》是由南梦宫自主研发的一款全新街机游戏。游戏发生在一个被称作「迷宫」的幻想世界，在这里，被神选中的人将被授予「能量豆」，导引元素之力。你将扮演一位名为「吃豆人」的神秘角色，在自由的旅行中邂逅性格相同、能力一致的豆子们，和他们一起击败强敌，找回失散的豆子——同时，逐步发掘「吃豆人」的真相。

接下来，你将进入吃豆人的世界~

### Search

相关文件在 [search]() 文件夹下。

#### 介绍

接下来这段时间，你将用搜索算法实现吃豆人的行为。

你需要补全的代码文件有：

- `search.py` 
- `searchAgents.py` 

你可以阅读并参考来帮助你实现代码的文件有：

- `pacman.py`
- `game.py`
- `util.py`

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

虽然 BFS 会找到一条到达目标的最少操作路径，但我们可能希望找到其他意义上“最佳”的路径。考虑 `mediumDottedMaze` 和 `mediumScaryMaze` 这两个地图。通过改变成本函数，我们可以鼓励吃豆人寻找不同的路径。例如，我们可以对鬼魂出没地区的危险步骤收取更多费用，或对食物丰富地区的步骤收取更少费用，而理性的吃豆人代理应该会据此调整其行为。

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

在 `search.py` 中的函数 `aStarSearch` 中实现 A\* 搜索。A\* 以启发式函数为参数。启发式函数有两个参数：搜索问题中的状态（主要参数）和问题本身（用于参考）。`search.py` 中的 `nullHeuristic` 启发式函数就是一个简单的例子。

你可以使用曼哈顿距离启发式算法（已在 `searchAgents.py` 中实现 `manhattanHeuristic`）测试你的 A* 实现，该算法针对原来的问题，即寻找穿过迷宫到达固定位置的路径。

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

#### 
