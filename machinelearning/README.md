![](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/ml_project_teaser.png)

#### 介绍

该项目将介绍机器学习；你将建立一个神经网络来对数字进行分类，以及更多！

你需要补全的代码文件有：
- `models.py`

你可以忽略其他支持文件。

本项目推荐使用 PyTorch，安装如下：

```
conda install torch
```

#### PyTorch 项目提供的代码

这里列出了你应该使用 PyTorch 的主要函数。此列表并不详尽，我们在 `models.py` 中导入了所有可能使用的函数，并鼓励你查看 PyTorch 文档，以获取有关如何使用这些函数的更多指南。

1. `tensor()`: 张量是 PyTorch 中的主要数据结构。它们的工作方式与 Numpy 数组非常相似，你可以对它们进行加法和乘法运算。每当你使用 PyTorch 函数或将输入传递到神经网络时，应确保输入是张量形式。你可以将 Python 列表转换为张量，例如：`tensor(data)`，其中 `data` 是 n 维列表。

2. `relu(input)`: PyTorch 的 ReLU 激活函数的调用方式为：`relu(input)`。它接收一个输入，并返回 `max(input, 0)`。

3. `Linear`: 使用此类实现线性层。线性层对包含权重的向量和输入进行点积。你必须在 `__init__` 函数中初始化它，如下所示：`self.layer = Linear(length of input vector, length of output vector)`，并在运行模型时调用它：`self.layer(input)`。这样定义线性层时，PyTorch 会自动创建权重并在训练过程中更新它们。

4. `movedim(input_vector, initial_dimension_position, final_dimension_position)`: 此函数接收一个矩阵，并交换 `initial_dimension_position`（以整数形式传递）与 `final_dimension_position`。这在第 Q3 中会有帮助。

5. `cross_entropy(prediction, target)`: 此函数应作为任何分类任务（ Q3-5 ）的损失函数。预测结果与目标值相差越远，该函数返回的值越大。

6. `mse_loss(prediction, target)`: 此函数应作为任何回归任务（ Q2 ）的损失函数。其使用方式与 `cross_entropy` 相同。

所有 PyTorch 版本中的数据将以 PyTorch 数据集对象的形式提供，你需要将其转换为 PyTorch 数据加载器，以便轻松创建批量大小。

```python
>>> data = DataLoader(training_dataset, batch_size=64)
>>> for batch in data:
>>>   # 在这里编写训练代码
```

对于所有这些问题，DataLoader 返回的每个批次将是一个字典，形式为：`{‘x’: features, ‘label’: label}`，其中 `label` 是我们基于特征要预测的值。

#### Q1: Perceptron

在开始这一部分之前，请确保你已经安装了 numpy 和 matplotlib！

在这一部分，你将实现一个二元感知机。你的任务是完成 `models.py` 中 `PerceptronModel` 类的实现。

对于感知机，输出标签将是 `1` 或 `-1`，这意味着数据集中的数据点 `(x, y)` 的 `y` 将是一个包含 `1` 或 `-1` 的 `torch.Tensor`。

你的任务是：

1. 完成 `__init__(self, dimensions)` 函数。这应该初始化 PerceptronModel 中的权重参数。请注意，这里你应该确保权重变量被保存为维度为 `1` $\times$ `dimensions` 的 `Parameter()` 对象。这样我们的自动评分器和 pytorch 才能将你的权重识别为模型的参数。
2. 实现 `run(self, x)` 方法。这应该计算存储的权重向量与给定输入的点积，并返回一个 Tensor 对象。
3. 实现 `get_prediction(self, x)` 方法。如果点积非负，则应返回 1，否则返回 -1。
4. 编写 `train(self)` 方法。这应该反复遍历数据集，并对分类错误的示例进行更新。当整个数据集的遍历完成且没有错误时，已达到 100% 的训练准确率，训练可以终止。

幸运的是，Pytorch 使得在张量上运行操作变得容易。如果你想通过某个张量方向和一个常量幅度来更新权重，你可以这样做：`self.w += direction * magnitude`

对于这个问题以及所有剩下的问题，DataLoader 返回的每个批次将是一个字典，形式为：`{‘x’: features, ‘label’: label}`，其中 `label` 是我们基于特征要预测的值。

要测试你的实现，请运行自动评分器：

```bash
python autograder.py -q q1
```

注意：对于正确的实现，自动评分器最多需要运行 20 秒左右。如果自动评分器运行时间过长，可能是你的代码有问题。

#### 神经网络技巧

在项目的剩余部分中，你将实现以下模型：

- Q2：非线性回归
- Q3：手写数字分类
- Q4：语言识别

###### 构建神经网络

在项目的应用部分中，你将使用 Pytorch 提供的框架创建神经网络来解决各种机器学习问题。一个简单的神经网络具有线性层，每个线性层执行线性操作（就像感知机一样）。线性层之间用非线性分隔，这允许网络逼近通用函数。我们将使用 ReLU 作为我们的非线性操作，定义为 $\text{relu}(x)=\max(x,0)$。例如，一个简单的单隐藏层/双线性层神经网络，将输入行向量 $\mathbf x$ 映射到输出向量 $\mathbf f(\mathbf x)$ 的函数如下：

$\mathbf f(\mathbf x)=\text{relu}(\mathbf x \cdot \mathbf{W_1}+\mathbf{b_1}) \cdot \mathbf{W_2} + \mathbf{b_2}$

其中参数矩阵 $\mathbf{W_1}$ 和 $\mathbf{W_2}$ 以及参数向量 $\mathbf{b_1}$ 和 $\mathbf{b_2}$ 是在梯度下降过程中需要学习的。 $\mathbf{W_1}$ 将是一个 $i \times h$ 矩阵，其中 $i$ 是输入向量 $\mathbf{x}$ 的维度，$h$ 是隐藏层大小。 $\mathbf{b_1}$ 将是一个大小为 $h$ 的向量。我们可以自由选择任何隐藏层大小（只需确保其他矩阵和向量的维度一致，以便我们可以执行操作）。使用较大的隐藏层大小通常会使网络更强大（能够拟合更多的训练数据），但可能会使网络更难训练（因为它增加了我们需要学习的所有矩阵和向量的参数数量），或者可能导致过拟合训练数据。

我们还可以通过添加更多层来创建更深的网络，例如一个三线性层的网络：

$\mathbf{\hat y} = \mathbf{f}(\mathbf{x}) = \mathbf{\text{relu}(\mathbf{\text{relu}(\mathbf{x} \cdot \mathbf{W_1} + \mathbf{b_1})} \cdot \mathbf{W_2} + \mathbf{b_2})} \cdot \mathbf{W_3} + \mathbf{b_3}$

或者，我们可以将上述公式分解，并显式地表示出两个隐藏层：

$\mathbf{h_1} = \mathbf{f_1}(\mathbf{x}) = \text{relu}(\mathbf{x} \cdot \mathbf{W_1} + \mathbf{b_1})$
$\mathbf{h_2} = \mathbf{f_2}(\mathbf{h_1}) = \text{relu}(\mathbf{h_1} \cdot \mathbf{W_2} + \mathbf{b_2})$
$\mathbf{\hat y} = \mathbf{f_3}(\mathbf{h_2}) = \mathbf{h_2} \cdot \mathbf{W_3} + \mathbf{b_3}$

请注意，最后我们没有使用 $\text{relu}$，因为我们希望能够输出负数，并且最初使用 $\text{relu}$ 的目的是为了进行非线性变换，而让输出成为一些非线性中间结果的仿射线性变换是非常合理的。

##### 批处理

为了提高效率，你将需要一次处理整个数据批次，而不是一次处理单个示例。这意味着你将面对一个大小为 $b \times i$ 的矩阵 $\mathbf{X}$，其中 $b$ 是批次大小，$i$ 是输入向量的维度。我们提供了一个线性回归的例子来展示如何在批处理设置中实现线性层。

##### 随机性

你的神经网络参数将随机初始化，并且某些任务中的数据将以随机顺序呈现。由于这种随机性，即使具有强大的架构，你也可能偶尔会在某些任务中失败——这是局部最优的问题！但这种情况应该非常少见——如果你在测试代码时连续两次在同一问题上失败，你应该探索其他架构。

##### 设计架构

设计神经网络可能需要一些反复试验。以下是一些帮助你的提示：

1. **系统化**：记录你尝试过的每个架构、超参数（层大小、学习率等）以及结果性能。当你尝试更多内容时，你会开始看到哪些参数重要的模式。如果你在代码中发现了错误，请确保将由于错误而无效的过去结果划掉。

2. **从浅层网络开始**：开始时使用一个浅层网络（只有一个隐藏层，即一个非线性层）。更深的网络具有指数级的超参数组合，即使一个错误也可能会破坏你的性能。使用小型网络找到合适的学习率和层大小；之后你可以考虑添加更多类似大小的层。

3. **学习率的选择**：如果你的学习率不正确，其他超参数选择都无关紧要。你可以从一篇研究论文中获取一个最先进的模型，并改变学习率，使其表现不比随机好。学习率太低会导致模型学习速度太慢，而学习率太高可能会导致损失发散到无穷大。开始时尝试不同的学习率，同时观察损失随时间的减少情况。

4. **批次大小与学习率**：较小的批次需要较低的学习率。当尝试不同的批次大小时，请注意最佳学习率可能会根据批次大小而不同。

5. **避免网络太宽**：如果你不断增加网络的宽度（隐藏层大小太大），精度将逐渐下降，计算时间将随着层大小呈二次方增长——你可能会因为过于缓慢而放弃，远在精度下降太多之前。整个项目的自动评分器使用工作人员解决方案运行约 12 分钟；如果你的代码花费时间更长，你应该检查其效率。

6. **避免 Infinity 或 NaN**：如果你的模型返回 `Infinity` 或 `NaN`，可能是当前架构的学习率太高。

7. **推荐的超参数值**：
    - 隐藏层大小：100 到 500 之间。
    - 批次大小：1 到 128 之间。对于第2题和第3题，我们要求数据集的总大小能够被批次大小整除。
    - 学习率：0.0001 到 0.01 之间。
    - 隐藏层数量：1 到 3 之间（这里尤其重要的是从小处开始）。

#### 示例：线性回归（Linear Regression）

作为神经网络框架工作原理的示例，让我们拟合一条直线到一组数据点。我们将从使用函数 $ y = 7x_0 + 8x_1 + 3 $ 构造的四个训练数据点开始。以批量形式表示，我们的数据为：

$$
X = \begin{bmatrix}
0 & 0 \\
0 & 1 \\
1 & 0 \\
1 & 1
\end{bmatrix}
\quad
Y = \begin{bmatrix}
3 \\
11 \\
10 \\
18
\end{bmatrix}
$$

假设数据以 Tensor 的形式提供给我们。

```python
>>> x
torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
>>> y
torch.Tensor([[3], [11], [10], [18]])
```

让我们构建并训练一个形式为 $ f(x) = x_0 \cdot m_0 + x_1 \cdot m_1 + b $ 的模型。如果正确完成，我们应该能够学到 $ m_0 = 7 $, $ m_1 = 8 $ 和 $ b = 3 $。

首先，我们创建可训练的参数。在矩阵形式中，这些参数是：

$$
M = \begin{bmatrix}
m_0 \\
m_1
\end{bmatrix}
\quad
B = \begin{bmatrix}
b
\end{bmatrix}
$$

对应的代码如下：

```python
m = torch.Tensor(2, 1)
b = torch.Tensor(1, 1)
```

需要记住的一个小细节是，除非你用数据初始化张量，否则张量会被初始化为全零值。因此，打印它们会得到：

```python
>>> m
torch.Tensor([[0], [0]])
>>> b
torch.Tensor([[0]])
```

接下来，我们计算模型对 $ y $ 的预测值。如果你正在处理 Pytorch 版本，你必须在 `__init__()` 函数中定义一个线性层，如上面的定义所述：

```python
predicted_y = self.Linear_Layer(x)
```

我们的目标是使预测的 $ y $ 值与提供的数据匹配。在线性回归中，我们通过最小化平方损失来实现这一点：

$$
L = \frac{1}{2N} \sum_{(x,y)} (y - f(x))^2
$$

我们计算损失值：

```python
loss = mse_loss(predicted_y, y)
```

最后，在定义你的神经网络之后，为了训练你的网络，你首先需要初始化一个优化器。Pytorch 内置了几种优化器，但对于这个项目，请使用：

```python
optim.Adam(self.parameters(), lr=lr)
```

其中 `lr` 是你的学习率。一旦定义了优化器，你必须在每次迭代中执行以下操作来更新你的权重：

1. 使用 `optimizer.zero_grad()` 重置 Pytorch 计算的梯度。
2. 调用你的 `get_loss()` 函数计算损失张量。
3. 使用 `loss.backward()` 计算你的梯度，其中 `loss` 是 `get_loss` 返回的损失张量。
4. 调用 `optimizer.step()` 更新你的权重。

你可以查看 [Pytorch 官方文档](https://pytorch.org/docs/stable/optim.html)，了解如何使用 Pytorch 优化器的示例。

#### Q2: Nonlinear Regression

对于这个问题，你将训练一个神经网络来逼近 $\sin(x)$ 在 $[-2\pi, 2\pi]$ 范围内的值。

你需要完成 `models.py` 中 `RegressionModel` 类的实现。对于这个问题，一个相对简单的架构应该就足够了（请参阅神经网络提示以获取架构提示）。使用 `mse_loss`作为你的损失函数。

你的任务是：

1. 实现 `RegressionModel.__init__`，进行任何必要的初始化。
2. 实现 `RegressionModel.forward`，返回一个 `batch_size` $\times$ `1` 的节点，表示你模型的预测。
3. 实现 `RegressionModel.get_loss`，返回给定输入和目标输出的损失。
4. 实现 `RegressionModel.train`，使用基于梯度的更新方法训练你的模型。

对于这个任务，只有一个数据集划分（即，只有训练数据，没有验证数据或测试集）。如果你的实现平均在数据集中的所有示例上达到 0.02 或更好的损失，你将获得满分。你可以使用训练损失来决定何时停止训练。请注意，模型训练需要几分钟时间。

```
python autograder.py -q q2
```

#### Q3: Digit Classification

对于这个问题，你将训练一个网络来对 MNIST 数据集中的手写数字进行分类。

每个数字的大小为 28 x 28 像素，其值存储在一个 784 维的浮点数向量中。我们提供的每个输出是一个 10 维向量，在所有位置都是零，除了对应于数字正确类别的位置为一。

完成 `models.py` 中 `DigitClassificationModel` 类的实现。`DigitClassificationModel.run()` 的返回值应该是一个 `batch_size` $\times$ `10` 的节点，包含得分，其中更高的得分表示数字属于特定类别（0-9）的概率更高。你应该使用 `cross_entropy` 作为你的损失函数。在网络的最后一个线性层不要使用 ReLU 激活函数。

对于这个问题和第4题，除了训练数据外，还有验证数据和测试集。你可以使用 `dataset.get_validation_accuracy()` 计算模型的验证准确率，这在决定是否停止训练时非常有用。测试集将由自动评分器使用。

要获得这个问题的分数，你的模型在测试集上的准确率应该至少达到 97%。作为参考，我们的工作人员实现经过大约 5 个周期（epochs）的训练后，在验证数据上持续达到 98% 的准确率。请注意，测试集成绩根据测试准确率评分，而你只能访问验证准确率——因此，如果你的验证准确率达到了 97% 的门槛，但测试准确率没有达到，你仍然可能会失败。因此，可能有助于在验证准确率上设置一个稍高的停止门槛，例如 97.5% 或 98%。

要测试你的实现，请运行自动评分器：

```
python autograder.py -q q3
```