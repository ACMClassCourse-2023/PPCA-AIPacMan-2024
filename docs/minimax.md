# Minimax 搜索算法

## 介绍

Minimax 是一种用于两人对弈游戏的决策算法，如国际象棋、井字棋和跳棋。它帮助确定玩家的最佳行动，假设对手也在最优地进行游戏。Minimax 算法的目标是在最小化可能损失的情况下最大化玩家的得分（因此得名 "minimax"）。

## 基本概念

### 游戏树

游戏树表示游戏中所有可能的移动。从当前状态开始，每个节点代表一个游戏状态，每条边代表一个移动。

### 最大化和最小化玩家

- **最大化玩家（Max）**：这个玩家试图获得尽可能高的分数。
- **最小化玩家（Min）**：这个玩家试图最小化最大化玩家的得分。

## Minimax 如何工作

1. **生成游戏树**：从当前游戏状态开始，生成所有可能的未来状态，直到某个预定深度。
2. **评估叶节点**：使用评估函数为每个叶节点（终止状态或最大深度的状态）分配一个分数。
3. **回溯分数**：
   - 对于 Max 节点，选择子节点中得分最高的。
   - 对于 Min 节点，选择子节点中得分最低的。
4. **选择最佳移动**：在根节点（当前游戏状态），选择导致 Max 玩家得分最高的移动。

## 示例

让我们以井字棋为例。假设现在是 Max 的回合，当前棋盘如下：

```
 X | O | X
-----------
 O | X |  
-----------
   |   | O
```

### 步骤 1：生成游戏树

生成 Max 的所有可能移动。由于是 Max 的回合，在每个空位上放置一个 "X"：

```
 X | O | X      X | O | X      X | O | X
-----------    -----------    -----------
 O | X | X      O | X |        O | X |  
-----------    -----------    -----------
   |   | O      X |   | O        | X | O
```

### 步骤 2：评估叶节点

为叶节点分配分数。假设：
- Max 获胜得 +10 分
- Min 获胜得 -10 分
- 平局得 0 分

```
 X | O | X      X | O | X      X | O | X
-----------    -----------    -----------
 O | X | X      O | X |        O | X |  
-----------    -----------    -----------
   |   | O      X |   | O        | X | O

得分: 0         得分: 10       得分: 0
```

### 步骤 3：回溯分数

由于根节点是 Max 的回合，选择子节点中得分最高的：

```
 Max
  |
  V
 10
```

### 步骤 4：选择最佳移动

Max 将选择导致得分为 10 的移动。

## 伪代码

以下是 Minimax 算法的简单伪代码：

```pseudo
function minimax(node, depth, maximizingPlayer)
    if depth == 0 or node is a terminal node
        return evaluate(node)
    
    if maximizingPlayer
        maxEval = -∞
        for each child of node
            eval = minimax(child, depth - 1, false)
            maxEval = max(maxEval, eval)
        return maxEval
    else
        minEval = +∞
        for each child of node
            eval = minimax(child, depth - 1, true)
            minEval = min(minEval, eval)
        return minEval
```

## Alpha-Beta 剪枝

Alpha-Beta 剪枝是 Minimax 算法的一种优化。它通过剪枝那些不会影响最终决策的分支，减少需要评估的节点数量。

### Alpha-Beta 剪枝的基本思想

Alpha-Beta 剪枝的主要思想是：在某些情况下，可以提前停止对某些节点的评估，因为这些节点不会影响最终的决策。

- **Alpha 值**：当前节点在最大化玩家层面上可以得到的最高分数。
- **Beta 值**：当前节点在最小化玩家层面上可以得到的最低分数。

如果在搜索过程中发现一个节点的评估值无法改进当前的 Alpha 或 Beta 值，就可以停止对该节点的进一步搜索。

要评估的节点数量，从而提高了效率。

### Alpha-Beta 剪枝的步骤

1. 初始条件：

    - Alpha 的初始值为负无穷（表示当前最大化玩家的已知最优值）。
    - Beta 的初始值为正无穷（表示当前最小化玩家的已知最优值）。
2. 剪枝条件：

    - 在最大化层（Max 层），如果当前节点的评估值大于或等于 Beta 值，则不再评估该节点的后续分支。因为这一分支返回的评估值不会小于当前评估值，即已经比 Beta 更大，因而上一层的最小化玩家不会选择这一分支，而是选择之前的 Beta（剪枝）。
    - 在最小化层（Min 层），如果当前节点的评估值小于或等于 Alpha 值，则不再评估该节点的后续分支，因为这一分支返回的评估值不会大于当前评估值，即已经比 Alpha 更小，因而上一层最大化玩家不会选择这一分支，而是选择之前的 Alpha（剪枝）。
3. 递归过程：

    - 在递归过程中，算法不断更新 Alpha 和 Beta 值。Alpha 表示当前最大化玩家可以保证的最高值，Beta 表示当前最小化玩家可以保证的最低值。
    - 每次评估一个节点后，根据评估结果更新 Alpha 或 Beta 值，并判断是否需要进行剪枝。

### 带 Alpha-Beta 剪枝的伪代码

```pseudo
function alphabeta(node, depth, α, β, maximizingPlayer)
    if depth == 0 or node is a terminal node
        return evaluate(node)
    
    if maximizingPlayer
        maxEval = -∞
        for each child of node
            eval = alphabeta(child, depth - 1, α, β, false)
            maxEval = max(maxEval, eval)
            α = max(α, eval)
            if β < α
                break
        return maxEval
    else
        minEval = +∞
        for each child of node
            eval = alphabeta(child, depth - 1, α, β, true)
            minEval = min(minEval, eval)
            β = min(β, eval)
            if β < α
                break
        return minEval
```

## 结论

Minimax 是一种在两人对弈游戏中做出最佳决策的强大算法。通过考虑所有可能的移动及其结果，它确保玩家在假设对手也在最优地进行游戏的情况下做出最佳移动。Alpha-Beta 剪枝进一步提高了 Minimax 的效率，减少了需要评估的节点数量。

