## CUDA-CNN

### 项目简介

这是一个用CUDA实现的 CNN (Convolutional Neural Network)，使用MINIST数据集进行训练，epoch=10，耗时35.8s，在测试集上的分类正确率达到96.54%。

### 运行项目

```shell
# clone项目到本地（需要CUDA环境）
git clone git@github.com:whut-zhangwx/CUDA-CNN.git
cd ./cuda-cnn
# 编译项目
make all
# 运行项目
./CNN
```

### 网络结构

$$
\begin{array}{rcl}
\text{Input Layer} & \fbox{input}\\
&\downarrow  & (1,28,28)\\
\text{Conv2d Layer1} & \fbox{$C_{in}=1,C_{out}=6,kernel=6\times5\times5,stride=1$}\\
&\downarrow & (6,24,24)\\
\text{Activation} & \fbox{Sigmoid Layer}\\
&\downarrow & (6,24,24)\\
\text{Conv2d Layer2} & \fbox{$C_{in}=6,C_{out}=6,kernel=1\times4\times4,stride=4$}\\
&\downarrow & (6,6,6)\\
\text{Activation} & \fbox{Sigmoid Layer}\\
&\downarrow & (6,6,6)\\
\text{FC Layer} & \fbox{$f_{in}=216,f_{out}=10$}\\
&\downarrow & (10)\\
\text{Activation} & \fbox{Sigmoid Layer}\\
&\downarrow & (10)\\
\text{Output Layer} & \fbox{Output}
\end{array}
$$

> $$Out = (In - Kernal + 2\times Padding)/Stride + 1$$

### Forward propagation

#### Input layer

$$
Out_{in}[i][j] = img[i][j],\qquad i,j\in\{0,1,\cdots,27\}
$$

#### Convolution layer1

Kernal size: 6×5×5, Stride: 1
Input size: 1×28×28, Output size: 6×24×24

$$
\begin{gather*}
PreA_{c1}[i_2][i_3][i_4] = \sum_{i_7=0}^4\sum_{i_8=0}^{4} Weight_{c1}[i_2][i_7][i_8] \cdot Out_{in}[i_3+i_7][i_4+i_8] + Bias_{c1}[i_2]\\
i_2\in{0,1,\cdots,5};\qquad i_3,i_4\in{0,1,\cdots,23}
\end{gather*}
$$

#### Activation

$$
\begin{gather*}
Out_{c1}[i_2][i_3][i_4] = \frac{1}{1 + \exp(-PreA_{c1}[i_2][i_3][i_4])}\\
i_2\in{0,1,\cdots,5};\qquad i_3,i_4\in{0,1,\cdots,23}
\end{gather*}
$$

#### Convolution layer2

Kernal size: 1×4×4, Stride: 4
Input size: 6×24×24, Output size: 6×6×6

$$
\begin{gather*}
PreA_{c2}[i_2][i_3][i_4] = \sum_{i_5=0}^3\sum_{i_6=0}^{3} Weight_{c2}[i_5][i_6] \cdot Out_{c1}[i_2][4i_3+i_5][4i_4+i_6] + Bias_{c2}\\
i_2\in{0,1,\cdots,5};\qquad i_3,i_4\in{0,1,\cdots,5}
\end{gather*}
$$

#### Activation

$$
\begin{gather*}
Out_{c2}[i_2][i_3][i_4] = \frac{1}{1 + \exp(-PreA_{c2}[i_2][i_3][i_4])}\\
i_2\in{0,1,\cdots,5};\qquad i_3,i_4\in{0,1,\cdots,5}
\end{gather*}
$$

#### Fully Connected Layer

Input size: 6×6×6, Output size: 10

$$
\begin{gather*}
PreA_{fc}[i_1] = \sum_{i_2=0}^{5}\sum_{i_3=0}^{5}\sum_{i_4=0}^{5} Weight_{fc}[i_1][i_2][i_3][i_4] \cdot Out_{c2}[i_2][i_3][i_4] + Bias_{fc}[i_1]\\
i_1\in{0,1,\cdots,9}
\end{gather*}
$$

#### Activation

$$
\begin{gather*}
Out_{fc}[i_1] = \frac{1}{1 + \exp(-PreA_{fc}[i_1])}\\
i_1\in{0,1,\cdots,9}
\end{gather*}
$$

### Loss Function

对于一个样本 $(img,label)$ 的输出 $Out_{fc}$, 令 $err[i], i\in\{0,1,\cdots,9\}$ 表示每个类别的预估错误

$$
err[i] =
\begin{cases}
Out_{fc}[i], & i\neq label\\
-(1-Out_{fc}[i]), & i = label
\end{cases}
$$

其中 $err[label]$ 本应为 $1-Out_{fc}[label]$, 但是为了方便后面求梯度时 $\frac{\partial Loss}{\partial err[i]} \cdot\frac{\partial err[i]}{\partial Out_{fc}[i]} = err[i]$ 的表示, 我们为它加了个负号. 这对计算Loss没有影响, 因为都要平方.

采用预估错误的平方和来计算损失

$$
Loss = \frac{1}{2}\sum_{i=0}^9 err[i]^2 = \frac{1}{2}(out_{fc}[label]-1)^2 + \frac{1}{2}\sum_{i=0,i\neq label}^{9}out_{fc}[i]^2
$$

### Back propagation

#### Fully Connected Layer

```math
\begin{split}
\frac{\partial Loss}{\partial Weight_{fc}[i_1][i_2][i_3][i_4]} &=
\frac{\partial Loss}{\partial err[i_1]} \cdot
\frac{\partial err[i_1]}{\partial Out_{fc}[i_1]} \cdot
\frac{\partial Out_{fc}[i_1]}{\partial PreA_{fc}[i_1]} \cdot
\frac{\partial PreA_{fc}[i_1]}{\partial Weight_{fc}[i_1][i_2][i_3][i_4]} \\
&= err[i_1] \cdot 1 \cdot Out_{fc}[i_1](1-Out_{fc}[i_1]) \cdot Out_{c2}[i_2][i_3][i_4]
\end{split}
```

$$
\begin{split}
\frac{\partial Loss}{\partial Bias_{fc}[i_1]} &=
\frac{\partial Loss}{\partial err[i_1]} \cdot
\frac{\partial err[i_1]}{\partial Out_{fc}[i_1]} \cdot
\frac{\partial Out_{fc}[i_1]}{\partial PreA_{fc}[i_1]} \cdot
\frac{\partial PreA_{fc}[i_1]}{\partial Bias_{fc}[i_1]} \\
&= err[i_1] \cdot 1 \cdot Out_{fc}[i_1](1-Out_{fc}[i_1]) \cdot 1
\end{split}
$$

#### Convolution layer2

$$
\begin{split}
\frac{\partial Loss}{\partial Weight_{c2}[i_5][i_6]} &=
\sum_{i_1=0}^{9}\sum_{i_2=0}^{5}\sum_{i_3=0}^{5}\sum_{i_4=0}^{5}
\frac{\partial Loss}{\partial err[i_1]} \cdot
\frac{\partial err[i_1]}{\partial Out_{fc}[i_1]} \cdot
\frac{\partial Out_{fc}[i_1]}{\partial PreA_{fc}[i_1]} \\ &\cdot
\frac{\partial PreA_{fc}[i_1]}{\partial Out_{c2}[i_2][i_3][i_4]}
\frac{\partial Out_{c2}[i_2][i_3][i_4]}{\partial PreA_{c2}[i_2][i_3][i_4]} \cdot
\frac{\partial PreA_{c2}[i_2][i_3][i_4]}{\partial Weight_{c2}[i_5][i_6]}
\\
&=
\sum_{i_1=0}^{9}\sum_{i_2=0}^{5}\sum_{i_3=0}^{5}\sum_{i_4=0}^{5}
err[i_1] \cdot 1 \cdot Out_{fc}[i_1](1-Out_{fc}[i_1])\\
&\cdot Weight_{fc}[i_1][i_2][i_3][i_4] \cdot
Out_{c2}[i_2][i_3][i_4](1-Out_{c2}[i_2][i_3][i_4]) \cdot
Out_{c1}[i_2][4i_3+i_5][4i_4+i_6]
\end{split}
$$

$$
\begin{split}
\frac{\partial Loss}{\partial Bias_{c2}}
&= \sum_{i_1=0}^{9}\sum_{i_2=0}^{5}\sum_{i_3=0}^{5}\sum_{i_4=0}^{5}
\frac{\partial Loss}{\partial err[i_1]}
\cdot \frac{\partial err[i_1]}{\partial Out_{fc}[i_1]}
\cdot \frac{\partial Out_{fc}[i_1]}{\partial PreA_{fc}[i_1]}\\
&\cdot \frac{\partial PreA_{fc}[i_1]}{\partial Out_{c2}[i_2][i_3][i_4]}
\cdot \frac{\partial Out_{c2}[i_2][i_3][i_4]}{\partial PreA_{c2}[i_2][i_3][i_4]}
\cdot \frac{\partial PreA_{c2}[i_2][i_3][i_4]}{\partial Bias_{c2}}
\\
&= \sum_{i_1=0}^{9}\sum_{i_2=0}^{5}\sum_{i_3=0}^{5}\sum_{i_4=0}^{5}
err[i_1] \cdot 1 \cdot Out_{fc}[i_1](1-Out_{fc}[i_1])
\\
&\cdot Weight_{fc}[i_1][i_2][i_3][i_4]
\cdot Out_{c2}[i_2][i_3][i_4](1-Out_{c2}[i_2][i_3][i_4]) \cdot 1
\end{split}
$$

#### Convolution layer1

$$
\begin{split}
\frac{\partial Loss}{\partial Weight_{c1}[i_2][i_7][i_8]} &=
\sum_{i_1=0}^{9}\sum_{i_3=0}^{5}\sum_{i_4=0}^{5}\sum_{i_5=0}^{3}\sum_{i_6=0}^{3}
\frac{\partial Loss}{\partial err[i_1]} \cdot
\frac{\partial err[i_1]}{\partial Out_{fc}[i_1]} \cdot
\frac{\partial Out_{fc}[i_1]}{\partial PreA_{fc}[i_1]} \\ &\cdot
\frac{\partial PreA_{fc}[i_1]}{\partial Out_{c2}[i_2][i_3][i_4]}
\frac{\partial Out_{c2}[i_2][i_3][i_4]}{\partial PreA_{c2}[i_2][i_3][i_4]} \cdot
\frac{\partial PreA_{c2}[i_2][i_3][i_4]}{\partial Out_{c1}[i_2][4i_3+i_5][4i_4+i_6]}\\
&\cdot
\frac{\partial Out_{c1}[i_2][4i_3+i_5][4i_4+i_6]}{\partial PreA_{c1}[i_2][4i_3+i_5][4i_4+i_6]} \cdot
\frac{\partial PreA_{c1}[i_2][4i_3+i_5][4i_4+i_6]}{\partial Weight_{c1}[i_2][i_7][i_8]}
\\
&= \sum_{i_1=0}^{9}\sum_{i_3=0}^{5}\sum_{i_4=0}^{5}\sum_{i_5=0}^{3}\sum_{i_6=0}^{3}
err[i_1] \cdot 1 \cdot Out_{fc}[i_1](1-Out_{fc}[i_1])\\
&\cdot Weight_{fc}[i_1][i_2][i_3][i_4] \cdot
Out_{c2}[i_2][i_3][i_4](1-Out_{c2}[i_2][i_3][i_4]) \cdot
Weight_{c2}[i_5][i_6]\\
&\cdot Out_{c1}[i_2][4i_3+i_5][4i_4+i_6](1-Out_{c1}[i_2][4i_3+i_5][4i_4+i_6])
\cdot Out_{in}[4i_3+i_5+i_7][4i_4+i_6+i_8]
\end{split}
$$

$$
\begin{split}
\frac{\partial Loss}{\partial Bias_{c1}[i_2]} &=
\sum_{i_1=0}^{9}\sum_{i_3=0}^{5}\sum_{i_4=0}^{5}\sum_{i_5=0}^{3}\sum_{i_6=0}^{3}
\frac{\partial Loss}{\partial err[i_1]} \cdot
\frac{\partial err[i_1]}{\partial Out_{fc}[i_1]} \cdot
\frac{\partial Out_{fc}[i_1]}{\partial PreA_{fc}[i_1]} \\ &\cdot
\frac{\partial PreA_{fc}[i_1]}{\partial Out_{c2}[i_2][i_3][i_4]}
\frac{\partial Out_{c2}[i_2][i_3][i_4]}{\partial PreA_{c2}[i_2][i_3][i_4]} \cdot
\frac{\partial PreA_{c2}[i_2][i_3][i_4]}{\partial Out_{c1}[i_2][4i_3+i_5][4i_4+i_6]}\\
&\cdot
\frac{\partial Out_{c1}[i_2][4i_3+i_5][4i_4+i_6]}{\partial PreA_{c1}[i_2][4i_3+i_5][4i_4+i_6]} \cdot
\frac{\partial PreA_{c1}[i_2][4i_3+i_5][4i_4+i_6]}{\partial Bias_{c1}[i_2]}
\\
&= \sum_{i_1=0}^{9}\sum_{i_3=0}^{5}\sum_{i_4=0}^{5}\sum_{i_5=0}^{3}\sum_{i_6=0}^{3}
err[i_1] \cdot 1 \cdot Out_{fc}[i_1](1-Out_{fc}[i_1])\\
&\cdot Weight_{fc}[i_1][i_2][i_3][i_4] \cdot
Out_{c2}[i_2][i_3][i_4](1-Out_{c2}[i_2][i_3][i_4]) \cdot
Weight_{c2}[i_5][i_6]\\
&\cdot Out_{c1}[i_2][4i_3+i_5][4i_4+i_6](1-Out_{c1}[i_2][4i_3+i_5][4i_4+i_6])
\cdot 1
\end{split}
$$
