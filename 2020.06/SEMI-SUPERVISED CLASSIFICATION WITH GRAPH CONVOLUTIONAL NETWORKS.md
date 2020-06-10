<!--
 * @Author: your name
 * @Date: 2020-06-05 08:35:10
 * @LastEditTime: 2020-06-05 11:38:37
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \Markdown NoteBook\200 研一 春\300 论文阅读笔记\2020.06\SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS.md
--> 
# 图卷积-半监督分类应用

在图结构中进行半监督分类。模型提出一个逐层传播规则，该规则基于==图上谱卷积的一阶近似==。

$$\mathcal{L} = \mathcal{L}_0 + \lambda\mathcal{L}_{reg}, \text{with} \mathcal{L}_{reg}=\sum_{i,j}A_{ij}||f(X_i)-f(X_j)||^2=f(X)^T\Delta f(X). \tag{1}$$

