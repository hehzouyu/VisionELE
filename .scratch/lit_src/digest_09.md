# Dimensionality reduction in surrogate modeling： A review of combined methods.txt

Data Science and Engineering (2022) 7:402-427

https://doi.org/10.1007/s41019-022-00193-5



 REVIEW/SURVEY PAPERS



Dimensionality Reduction in Surrogate Modeling: A Review

of Combined Methods



Chun Kit Jeffery Hou1 � Kamran Behdinan1



Received: 2 March 2022 / Revised: 21 July 2022 / Accepted: 9 August 2022 / Published online: 21 August 2022

� The Author(s) 2022



Abstract

Surrogate modeling has been popularized as an alternative to full-scale models in complex engineering processes such as

manufacturing and computer-assisted engineering. The modeling demand exponentially increases with complexity and

number of system parameters, which consequently requires higher-dimensional engineering solving techniques. This is

known as the curse of dimensionality. Surrogate models are commonly used to replace costly computational simulations and

modeling of complex geometries. However, an ongoing challenge is to reduce execution and memory consumption of high-

complexity processes, which often exhibit nonlinear phenomena. Dimensionality reduction algorithms have been employed

for feature extraction, selection, and elimination for simplifying surrogate models of high-dimensional problems. By applying

dimensionality reduction to surrogate models, less computation is required to generate surrogate model parts while retain-

ing sufficient representation accuracy of the full process. This paper aims to review the current literature on dimensionality

reduction integrated with surrogate modeling methods. A review of the current state-of-the-art dimensionality reduction and

surrogate modeling methods is introduced with a discussion of their mathematical implications, applications, and limitations.

Finally, current studies that combine the two topics are discussed and avenues of further research are presented.



Keywords Dimensionality reduction � Surrogate modeling � Machine learning � High dimensionality � Neural networks



1Introduction                                                  structure while sacrificing minimal accuracy and informa-

                                                               tion. DR can be achieved through feature elimination, feature

Data mining has become a rapidly growing field in recent       selection, or feature extraction. Feature elimination involves

years. At the same time, data generation has seen a surge in   reducing the input dimension space by eliminating features

volume, leading to a growth in size, complexity, and data      of the dataset that are deemed unimportant. Although this

dimensionality. High-dimensional data exists where the         simplifies the computations afterwards, no information is

number of data features is on the order of the number of       gained by dropping those features. Feature selection involves

samples or observations [1]. These datasets can be computa-    using statistics to determine and rank features based on their

tionally expensive to learn and generating mapping functions   information contribution to the overall dataset. These meth-

between input and output can be a cumbersome task. Thus,       ods can be categorized as filter and wrapper methods and

reducing the number of features, or the problem dimen-         have been explored in detail in [2]. It is important to note

sionality, can greatly simplify the learning and training of   that there is no universal method for ranking data features as

regression and classification models for extracting patterns   different tests will yield different contribution scores. Global

in the data. Dimensionality reduction (DR) techniques seek     sensitivity analysis methods [3], which identify the `most

to reduce the data dimensionality and identify intrinsic data  important' inputs in unstructured datasets and ignore the

                                                               others, have emerged as a novel feature selection method for

