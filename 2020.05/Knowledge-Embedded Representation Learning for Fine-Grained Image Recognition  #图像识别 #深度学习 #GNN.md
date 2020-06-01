# Knowledge-Embedded Representation Learning for Fine-Grained Image Recognition

[TOC]

## 1. 摘要

**讲故事** 人类通过自然累积的知识和专家经验辅助理解图像。

**方法** 统一深度学习框架专家知识 + 提出框架KERL。

知识图形式的可视化概念 + 利用门限图神经网络， 通过图传播节点信息，生成知识表示

框架优点

1. 提高区分粒度，识别小类别判定(同类别下的细致区别)
2. 根据知识图重点关注部分区域，有意义学习，可解释性强

## 2. 介绍

知识：包含**类别标签**和**属性**的复杂可视化概念组织，有利于细粒度图像识别。
根据相关**知识**，判别物体对应部位的属性是否吻合，再形成推理。**推理过程**如下图：
![图1](https://raw.githubusercontent.com/bysen32/PicGo/master/Snipaste_2020-05-25_10-05-42.jpg)

传统图像识别方法：忽视知识，仅关注低级图像线索。(基于部件模型或借助视觉注意力网络)识别细微区别。
缺点：

1. 基于部件的模型：重度依赖部件**标记**
2. 可视化注意力网络：缺乏监督或指导，只能**大致**定位零件/区域

KERL：将 类别知识 和 部件属性 用知识图谱的形式组织
文中的两大组件：

1. Gated Graph Neural Network：通过图传播节点消息以生成==知识表示==
2. 奇特的**门限机制** 将==习得表示==与**图特征学习**结合，学习属性感知特征

具体步骤：

1. 构建大规模知识图（部件级属性的相关类别标签）
2. 通过用给定图的信息初始化图节点，框架可隐式推理图的类别属性，关联属性与特征图（用有意义的配置学习特征图）
![图2](https://raw.githubusercontent.com/bysen32/PicGo/master/Snipaste_2020-05-25_13-48-51.jpg)
从对应类别样例中学会的特征映射，提高图像相关部位的关注度。

## 3. 相关工作

### 3.1 Fine-Grained Image Classification

传统CNN方法：

1. 双线性模型，计算高维表述，通过两个独立子网，更好地模拟局部成对特征交互。
2. 区分子类别间微小的视觉差异，定位差异区域，并在其中学习外貌模型

缺点：重度依赖标记，手工标记非最优
改进：调节突出局部技术，自动生成区分区域的bounding box注释。 (自动化搜索信息区域)

相关方法：

1. Fully convolutional attention localization networks: Efﬁcient attention localization for ﬁne-grained recognition：增强学习框架自适应扫描局部判别区域，提出贪婪奖赏策略训练图片级注释的框架。
2. Look closer to see better: recurrent attention convolutional neural network for ﬁne-grained image recognition：介绍了一种重复注意力卷积网络在多尺度、区域特征表示上递归学习关注区域。
3. Localizing by describing: Attribute-guided attentionlocalizationforﬁne-grainedrecognition：利用部件级属性指导注意力区域定位。

**本文**使用 知识图 + ==隐式推理判别属性==，而非直接使用对象属性对。

### 3.2 Knowledge Representation

GGNN：一种针对图结构数据的RNN架构，递归地将节点消息传播到其邻居以学习节点级特征或图级表示

相关工作中GSNN最为相关：简单连接 图和知识特征进行图像分类任务。(The more you know: Using knowledge graphs for image classiﬁcation)

本文：开发新颖的门限机制知识表示嵌入图像特征学习，增强特征表述。
学习的特征图展示深刻的配置：语义高亮的可解释性。

## 4. KERL框架

![20200526095947](https://raw.githubusercontent.com/bysen32/PicGo/master/20200526095947.png)

### 4.1 GGNN简介

GGNN：RNN结构，通过迭代更新节点特征，学习任意图结构数据的特征。
![20200526095910](https://raw.githubusercontent.com/bysen32/PicGo/master/20200526095910.png)

### 4.2 知识图构成

知识图：可视化概念（如类别标签与部位属性）的仓库。基于训练样本的属性注释构造。
**可视化概念**：类别标签或属性。
**相关性**：类别标签与属性的相关性。实际应用中，关系并不健全。使用一个评分机制估计 属性实例对 关联性。
![20200526105125](https://raw.githubusercontent.com/bysen32/PicGo/master/20200526105125.png)
$S$维度$C \times A$，$\bold 0$矩阵表示两类别或属性之间**无连接**。
构造知识图 $G = \{V, A_c\}$

### 4.3 知识表示学习

GGNN通过图传播节点信息，并计算每个节点的特征向量。所有的特征向量连接起来形成知识图最终表示。各节点的**输入特征**表示如下：
![20200526111338](https://raw.githubusercontent.com/bysen32/PicGo/master/20200526111338.png)
$s_i$：表示给定图像关于类别标签 $i$ 的置信度。

对于每个节点$v$，其隐状态的计算过程：
![20200526133715](https://raw.githubusercontent.com/bysen32/PicGo/master/20200526133715.png)
在迭代过程中，节点的隐状态由历史状态与邻居节点信息决定。（聚合与模拟转换信息）
$T$轮迭代后节点信息传遍全图，得到图中所有节点的最终隐变量。
节点级特征计算过程：
$$\bold o_v = o(h^T_v,x_v),v=1,2,\dots,|V|, \tag{5}$$

### 4.4 联合表示学习

门限机制：知识表示嵌入**图表示学习**。
**Image特征提取**：使用一个紧致双线性模型进行细粒度图像分类。
重点关注判别区域捕捉细致差异，从而进行区分。
知识表示编码类别属性关系，捕捉属性差异。
通过知识指导，**使用门限机制过滤非信息特性。**，形式如下：$$\bold{f} = \sum_{i,j}\delta(g(\bold f^I_{i,j},\bold f^g)) \bigodot \bold f^I_{i,j} \tag{6}$$

$f^I_{i,j}$：位置$(i,j)$的特征向量。
$\sigma(g(f^I_{i,j},f^g))$：门限机制，过滤非信息特征。
$g$：神经网络，串联$f^I_{i,j}$与$f^g$为输入，输出$c$维实值向量
输出值$f$后接全连接层，用以计算$score$

## 5. 实验

### 5.3 知识嵌入的贡献

基于**CB-CNN**，展示知识嵌入的重要性。实现了两个基准方法：

**Comparison with self-guided feature learning.**
验证**知识嵌入**对特征学习的作用，设计实验移除GGNN,仅仅将将图片特征反馈给门限神经网络
**Comparison with feature concatenation**
验证知识嵌入方法的效果。

实验结果对比如下图：
![20200528200940](https://raw.githubusercontent.com/bysen32/PicGo/master/20200528200940.png)
可见CB-CNN+KnowledgeEmbeding有提升。

### 5.4 表示可视化

属性对应区域高亮

## 6. 存在的疑问

- [ ] 图表示学习 Image Represent Learning
- [ ] GGNN
- [ ] Markov Logic Network
- [ ] VGG16-Net
- [x] 双线性模型
- [x] self-guided feature learning
- [ ] feature concatenation
- [ ] 知识表示结构从何而来
- [ ] 图像特征提取的方法
- [ ] 知识如何嵌入网络

## 7. 简短总结

## 拓展阅读

- Gated graph sequence neural networks.
  - 门限神经网络，知识表述嵌入图表示学习
- Compact bilinear pooling. (CB-CNN)
  - 双线性模型，图片特征提取
  - 细粒度图像分类
