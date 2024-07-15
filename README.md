# PPCA-AIPacMan-2024

PPCA AI 吃豆人项目

本项目基于[加州大学伯克利分校的CS 188《Introduction to Artificial Intelligence》课程项目](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/)

[http://ai.berkeley.edu](http://ai.berkeley.edu)

阅读材料：[AIMA，第 4 版](https://aima.cs.berkeley.edu/)。

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

相关文件在 [tutorial](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/tree/main/tutorial) 文件夹下。

我们提供了一些小测试以供大家熟悉 Python：

- 加法：阅读 `addition.py` 并补全代码。
- 给定水果价格和订单列表，计算总价：阅读 `buyLotsOfFruit.py` 并实现`buyLotsOfFruit(orderList)`函数。
- 计算最低总价：阅读 `shopSmart.py` 并补全 `shopSmart(orderList,fruitShops)` 函数。在 `shop.py` 中查看 `FruitShop` 类的实现。

运行 `python autograder.py` 会评估你对这三个问题的解决方案，运行 `python autograder.py -q q1` 将仅测试第一个问题。

### Pac-Man Agents

>你说的对，但是《吃豆人》是由南梦宫自主研发的一款全新街机游戏。游戏发生在一个被称作「迷宫」的幻想世界，在这里，被神选中的人将被授予「能量豆」，导引元素之力。你将扮演一位名为「吃豆人」的神秘角色，在自由的旅行中邂逅性格相同、能力一致的豆子们，和他们一起击败强敌，找回失散的豆子——同时，逐步发掘「吃豆人」的真相。

接下来，你将进入吃豆人的世界~

简单介绍规则：

- 吃豆人吃完所有豆子后胜利，被幽灵碰到则失败。
- 吃豆人可以吃能量豆（大豆子），吃到后幽灵会在一段时间内进入惊恐状态（白色），此时可以吃幽灵。
- 吃豆人在迷宫中停留时会不断扣分。

### Search

相关文件和介绍在 [search](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/tree/main/search) 文件夹下。

## Week2

### MultiAgent

相关文件和介绍在 [multiagent](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/tree/main/multiagent) 文件夹下。

### Logic

相关文件和介绍在 [logic](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/tree/main/logic) 文件夹下。

### Tracking

相关文件和介绍在 [tracking](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/tree/main/tracking) 文件夹下。

## Week3

### Machine Learning

相关文件和介绍在 [machinelearning](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/tree/main/machinelearning) 文件夹下。

## Week4

### Reinforcement Learning

相关文件和介绍在 [reinforcement](https://github.com/ACMClassCourse-2023/PPCA-AIPacMan-2024/tree/main/reinforcement) 文件夹下。