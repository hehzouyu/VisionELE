# A survey of recommender systems with multi-objective optimization.txt

                                                                Neurocomputing 474 (2022) 141-153



                                                    Contents lists available at ScienceDirect



                                                Neurocomputing



                                  journal homepage: www.elsevier.com/locate/neucom



Survey paper



A survey of recommender systems with multi-objective optimization



Yong Zheng a,, David (Xuejun) Wang b



a College of Computing, Illinois Institute of Technology, Chicago, IL 60616, USA

b Morningstar, Inc., Chicago, IL 60602, USA



article info                      abstract



Article history:                  Recommender systems have been widely applied to several domains and applications to assist decision

Received 25 September 2021        making by recommending items tailored to user preferences. One of the popular recommendation algo-

Revised 1 November 2021           rithms is the model-based approach which optimizes a specific objective to improve the recommenda-

Accepted 14 November 2021         tion performance. These traditional recommendation models usually deal with a single objective, such

Available online 1 December 2021  as minimizing the prediction errors or maximizing the ranking quality of the recommendations. In

                                  recent years, there is an emerging demand for multi-objective recommender systems in which multiple

Keywords:                         objectives are considered and the recommendations can be optimized by the multi-objective optimiza-

Recommender systems               tion. For example, a recommendation model may be built by optimizing multiple metrics, such as accu-

Multi-objective                   racy, novelty and diversity of the recommendations. The multi-objective optimization methodologies

Optimization                      have been well developed and applied to the area of recommender systems. In this article, we provide

Pareto optimal                    a comprehensive literature review of the multi-objective recommender systems. Particularly, we iden-

Multi-objective recommendation    tify the circumstances in which a multi-objective recommender system could be useful, summarize the

                                  methodologies and evaluation approaches in these systems, point out existing challenges or weak-

                                  nesses, finally provide the guidelines and suggestions for the development of multi-objective recom-

                                  mender systems.



                                                                                                                              � 2021 Elsevier B.V. All rights reserved.



1. Introduction                                                      [13-15]. Group fairness may be considered to alleviate the conflict

                                                                     between individuals and the whole group in the context of group

    The problem of information overload results in difficulties of   recommendations [16,17]. The objectives associated with multiple

getting access to desired information or consuming preferred items   tasks may need to be jointly optimized in the multi-task RSs [18-

in various applications. Recommender systems (RSs) were pro-         20]. We use the term multi-objective recommender systems

posed and developed to alleviate this issue by delivering a list of  (MORS) to refer to the RSs which produce recommendations by

recommendations tailored to user preferences. RSs have been          optimizing multiple objectives.

applied in several domains to assist users' decision making, such

