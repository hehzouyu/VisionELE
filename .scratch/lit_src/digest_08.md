# Deepreinforcementlearningassisted surrogate modelmanagementfor.txt

                                                 Swarm and Evolutionary Computation 92 (2025) 101817

                                                        Contents lists available at ScienceDirect



                                       Swarm and Evolutionary Computation



                                              journal homepage: www.elsevier.com/locate/swevo



Deep reinforcement learning assisted surrogate model management for



expensive constrained multi-objective optimization



Shuai  Shao,  Ye  Tian       Yajie  Zhang



                          ,



School of Computer Science and Technology, Anhui University, Hefei, 230601, China



ARTICLE INFO                               ABSTRACT



Keywords:                                  Expensive constrained multi-objective optimization problems (ECMOPs) exist in a wide variety of applications

Surrogate model                            from industrial processes to engineering systems. When solving ECMOPs, with only a limited number of

Expensive optimization                     function evaluations available, a common approach is to substitute the real function evaluations with more

Evolutionary computation                   affordable evaluations provided by computationally efficient surrogate models. However, existing surrogate

Neural networks                            assisted evolutionary algorithms (SAEAs) exhibit poor versatility in handling various ECMOPs, as they only

Multiple search modes                      use a constant surrogate modeling scheme or switch the modeling schemes with expert knowledge. To address

                                           the dilemma in surrogate modeling, this paper proposes a deep reinforcement learning assisted evolutionary

                                           algorithm, which operates on two key issues. First, multiple surrogate models are employed to learn the

                                           approximate function of an ECMOP using previously evaluated solutions during the evolutionary process.

                                           Second, a deep reinforcement learning method is employed to learn the optimal surrogate model management

                                           strategy based on evolutionary experience, selecting the most suitable surrogate modeling scheme for the

                                           current generation. Experimental evaluations on a large number of expensive problems demonstrate that the

                                           proposed algorithm has a significant effect compared with state-of-the-art competitors.



1. Introduction                                                                         constraints can be transformed into inequality constraints by relaxation



                                                                                        techniques [6,7], so this paper focuses exclusively on inequality con-



In many industrial processes and engineering systems, there are a                       straints. For a CMOP with inequality constraints, the constraint function



large number of optimization problems involving multiple conflicting                    values  of a candidate solution can be calculated as follows:



objectives simultaneously subject to constraints [1-3]. For example,                                                                                   (2)

                                                                                         () = max(0, ()) .

the software product configuration problem [4,5] aims to minimize

                                                                                        =1

