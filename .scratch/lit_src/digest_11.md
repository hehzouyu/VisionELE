# Surrogate models in evolutionary single-objective optimization：A new taxonomy and experimental study.txt

                                                             Information Sciences 562 (2021) 414-437



                                                     Contents lists available at ScienceDirect



                                                Information Sciences



                                       journal homepage: www.elsevier.com/locate/ins



Surrogate models in evolutionary single-objective optimization:

A new taxonomy and experimental study



Hao Tong b, Changwu Huang b, Leandro L. Minku a, Xin Yao b,



a School of Computer Science, University of Birmingham, Edgbaston, Birmingham B15 2TT, UK

b Guangdong Provincial Key Laboratory of Brain-inspired Intelligent Computation, Department of Computer Science and Engineering, Southern University of

Science and Technology, Shenzhen 518055, China



article info                           abstract



Article history:                       Surrogate-assisted evolutionary algorithms (SAEAs), which use efficient surrogate models

Received 14 May 2020                   or meta-models to approximate the fitness function in evolutionary algorithms (EAs), are

Received in revised form 1 March 2021  effective and popular methods for solving computationally expensive optimization prob-

Accepted 2 March 2021                  lems. During the past decades, a number of SAEAs have been proposed by combining dif-

Available online 08 March 2021         ferent surrogate models and EAs. This paper dedicates to providing a more systematical

                                       review and comprehensive empirical study of surrogate models used in single-objective

Keywords:                              SAEAs. A new taxonomy of surrogate models in SAEAs for single-objective optimization

Evolutionary algorithms                is introduced in this paper. Surrogate models are classified into two major categories: abso-

Surrogate models                       lute fitness models, which directly approximate the fitness function values of candidate

Absolute fitness models                solutions, and relative fitness models, which estimates the relative rank or preference of

Relative fitness models                candidates rather than their fitness values. Then, the characteristics of different models

Expensive optimization problems        are analyzed and compared by conducting a series of experiments in terms of time com-

                                       plexity (execution time), model accuracy, parameter influence, and the overall perfor-

                                       mance when used in EAs. The empirical results are helpful for researchers to select

                                       suitable surrogate models when designing SAEAs. Open research questions and future

                                       work are discussed at the end of the paper.

                                       � 2021 The Author(s). Published by Elsevier Inc. This is an open access article under the CC



                                                                   BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/).



1. Introduction



    Evolutionary Algorithms (EAs) require a so-called fitness function to evaluate the quality of candidate solutions when

solving a problem. A large number of fitness evaluations is usually required to obtain a satisfying solution. However, in many

real-world problems, the fitness function is very complex and expensive to compute. For example, a single Navier-Stokes

fitness evaluation will take several hours [1], and the fitness evaluation of vehicle shape design optimisation [2] involves

