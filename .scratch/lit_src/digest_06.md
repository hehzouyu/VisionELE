# A survey on evolutionary constrained multiobjective optimization.txt

IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 27, NO. 2, APRIL 2023                                                                       201



   A Survey on Evolutionary Constrained

         Multiobjective Optimization



     Jing Liang , Senior Member, IEEE, Xuanxuan Ban, Graduate Student Member, IEEE,

 Kunjie Yu , Member, IEEE, Boyang Qu , Member, IEEE, Kangjia Qiao , Member, IEEE,

Caitong Yue , Member, IEEE, Ke Chen , Member, IEEE, and Kay Chen Tan , Fellow, IEEE



   Abstract--Handling constrained multiobjective optimization               dispatch (CEED) problem [2], the urban bus scheduling

problems (CMOPs) is extremely challenging, since multiple con-              problem [3], and energy saving optimization problem [4]. This

flicting objectives subject to various constraints require to be            kind of problems is denoted as constrained multiobjective

simultaneously optimized. To deal with CMOPs, numerous con-                 optimization problems (CMOPs). CMOPs bring more chal-

strained multiobjective evolutionary algorithms (CMOEAs) have               lenges than their unconstrained counterparts, due to the co-

been proposed in recent years, and they have achieved promis-               existence of multiple objectives and constraints. For CMOPs,

ing performance. However, there has been few literature on the              because of the conflict among multiple objectives, there exist

systematic review of the related studies currently. This article            no single solution but a set of solutions that indicates the best

provides a comprehensive survey for evolutionary constrained                possible tradeoffs among the objectives. The set of solutions

multiobjective optimization. We first review a large number of              forms the so-called Pareto-optimal set (PS), and its image in

CMOEAs through categorization and analyze their advantages                  objective space is called the Pareto front (PF). Meanwhile,

and drawbacks in each category. Then, we summarize the bench-               because the presence of the constraints can change the land-

mark test problems and investigate the performance of different             scape of the search region and cause infeasible regions, the

constraint handling techniques (CHTs) and different algorithms,             feasibility of the solutions cannot be neglected. Consequently,

followed by some emerging and representative applications of                the final goal of solving CMOPs is to obtain a set of feasi-

CMOEAs. Finally, we discuss some new challenges and point out               ble solutions to approach the PF with good convergence and

some directions of the future research in the field of evolutionary         diversity.

constrained multiobjective optimization.

                                                                               In the past two decades, multiobjective evolutionary algo-

   Index Terms--Benchmark test problems, constrained                        rithms (MOEAs) have become a very popular alternative for

multiobjective optimization, constraint handling, evolutionary              addressing the multiobjective optimization problems [5], [6].

algorithms.                                                                 The most important reason is that MOEAs belong to

                                                                            population-based stochastic metaheuristics that utilize the prin-

                          I. INTRODUCTION                                   ciples of natural selection to guide a set of solutions toward the

                                                                            PF. As a result, MOEAs are capable of providing a set of solu-

A VARIETY of optimization problems in real-world                            tions for a multiobjective optimization problem in a single run.

      applications contains several conflicting objectives and              Up to now, a large number of MOEAs have been proposed, and

multiple complicate constraints, such as the robot gripper                  they can be roughly divided into the methods based on domi-

optimization problem [1], the combined economic emission                    nance [7]-[9], the methods based on decomposition [10]-[12],

                                                                            and the methods based on indicator [13]-[15]. These MOEAs

   Manuscript received 1 August 2021; revised 26 November 2021; accepted    have shown competitive performance in solving unconstrained

22 February 2022. Date of publication 1 March 2022; date of cur-            multiobjective optimization problems. However, they can not

rent version 31 March 2023. This work was supported in part by the          be directly applied to CMOPs due to the absence of a

National Natural Science Foundation of China under Grant 62176238,          constraint handling technique (CHT). In fact, CHT can be

Grant 61806179, Grant 61922072, Grant 61876169, Grant 61976237, and         seen as a selection mechanism to cope with constraints.

Grant 62106230; in part by the China Postdoctoral Science Foundation        In various practical application scenarios, CMOPs are more

