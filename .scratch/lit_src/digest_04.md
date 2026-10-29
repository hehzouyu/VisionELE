# A cluster-based competitive particle swarm optimizer with a sparse truncation operator for multi-objective optimization.txt

                                                Swarm and Evolutionary Computation 71 (2022) 101083

                                                        Contents lists available at ScienceDirect



                                      Swarm and Evolutionary Computation



                                             journal homepage: www.elsevier.com/locate/swevo



A Cluster-Based Competitive Particle Swarm Optimizer with a Sparse

Truncation Operator for Multi-Objective Optimization



Xiangyu Wang a, Bingran Zhang b, Jian Wang a,, Kai Zhang c, Yaochu Jin d,e,



a College of Science, China University of Petroleum (East China), Qingdao, 266580, China

b Department of Mathematics, University College London, London, WC1E 6BT, U.K

c School of Petroleum Engineering, China University of Petroleum (East China), Qingdao, 266580, China

d Faculty of Technology, Bielefeld University, Bielefeld 33619, Germany

e Department of Computer Science, University of Surrey, Guildford GU2 7XH, U.K



article info                          a b s t r a c t



Keywords:                             Many different types of multi-objective optimization problems, e.g. multi-modal problems and large-scale prob-

Sparse Pareto optimal solutions       lems, have been solved with high performance by numbers of tailored multi-objective evolutionary algorithms.

Particle swarm optimization (PSO)     Little attention has been paid on sparse optimization problems, whose most decision variables are zero in the

Competitive swarm optimization (CSO)  Pareto optimal solution set. Most recently, algorithms for solving sparse problems have been developed rapidly,

Accumulative gradient                 and many sparse optimization problems in machine learning, such as the search for lightweight neural networks,

Neural network                        can be solved with the help of multi-objective evolutionary algorithms. In this paper, we introduce a sparse trun-

                                      cation operator which uses the accumulative gradient value as a criterion for setting a decision variable to zero. In

                                      addition, to balance the exploration and exploitation, a cluster-based competitive particle swarm optimizer is pro-

                                      posed, which takes advantage of both particle swarm optimization and competitive swarm optimizer to search

                                      efficiently and escape from local optima. Consequently, aiming at solving sparse multi-objective optimization

                                      problems, a novel cluster-based competitive particle swarm optimizer with a sparse truncation operator is pro-

                                      posed, and experimental results show that the proposed algorithm outperforms its peers on sparse test instances

                                      and neural network training tasks.



1. Introduction                                                           ulated binary crossover (SBX) to determine the search direction [8], and

                                                                          this strategy can search efficiently in training deep neural networks.

    Convex optimization is an important field in the optimization com-

munity, which has been studied extensively [1]. In fact, there are many       Since the first evolutionary algorithm (EA) was proposed by Schaffer

traditional mathematical methods to solve single objective convex op-     for solving multi-objective optimization problems (MOPs) [9], a great

timization problems, e.g. gradient descent method [2], Newton method      number of population-based meta-heuristic multi-objective evolution-

[3], and Quasi-Newton method [4]. It is relatively easy for these afore-  ary algorithms (MOEAs) have been developed [10-13] to solve MOPs,

mentioned methods to get the minimum of a convex function at a convex     due to their global search ability and insensitivity to the properties of

set by using the first-order or second-order gradient information. For    functions. These MOEAs depend on various principles of biology mech-

non-differentiable or multi-modal single-objective optimization prob-     anisms [14,15]. Among these EAs, a group of swarm intelligence algo-

lems, or for multi-objective optimization problems, evolutionary algo-    rithms, i.e., particle swarm optimization (PSO) algorithms [16-18] show

