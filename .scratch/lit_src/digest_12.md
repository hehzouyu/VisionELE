# annurev-physchem-083122-125941.txt

Downloaded from www.annualreviews.org. Guest (guest) IP: 185.143.220.236 On: Wed, 30 Jul 2025 10:25:36                                                         Annual Review of Physical Chemistry



                                                                                                                                                               Enhanced Sampling with

                                                                                                                                                               Machine Learning



                                                                                                                                                               Shams Mehdi,1,2 Zachary Smith,1,2 Lukas Herron,1,2

                                                                                                                                                               Ziyue Zou,3 and Pratyush Tiwary1,3



                                                                                                                                                               1Institute for Physical Science and Technology, University of Maryland, College Park,

                                                                                                                                                               Maryland, USA; email: ptiwary@umd.edu

                                                                                                                                                               2Biophysics Program, University of Maryland, College Park, Maryland, USA

                                                                                                                                                               3Department of Chemistry and Biochemistry, University of Maryland, College Park,

                                                                                                                                                               Maryland, USA



                                                                                                        Annu. Rev. Phys. Chem. 2024. 75:347-70                 Keywords



                                                                                                        First published as a Review in Advance on              molecular dynamics, enhanced sampling, machine learning, artificial neural

                                                                                                        February 21, 2024                                      networks



                                                                                                        The Annual Review of Physical Chemistry is online at   Abstract

                                                                                                        physchem.annualreviews.org

                                                                                                                                                               Molecular dynamics (MD) enables the study of physical systems with excel-

                                                                                                        https://doi.org/10.1146/annurev-physchem-083122-       lent spatiotemporal resolution but suffers from severe timescale limitations.

                                                                                                        125941                                                 To address this, enhanced sampling methods have been developed to im-

                                                                                                                                                               prove the exploration of configurational space. However, implementing

                                                                                                        Copyright � 2024 by the author(s). This work is        these methods is challenging and requires domain expertise. In recent years,

                                                                                                        licensed under a Creative Commons Attribution 4.0      integration of machine learning (ML) techniques into different domains has

                                                                                                        International License, which permits unrestricted      shown promise, prompting their adoption in enhanced sampling as well.

                                                                                                        use, distribution, and reproduction in any medium,     Although ML is often employed in various fields primarily due to its data-

                                                                                                        provided the original author and source are credited.  driven nature, its integration with enhanced sampling is more natural with

                                                                                                        See credit lines of images or other third-party        many common underlying synergies. This review explores the merging of

                                                                                                        material in this article for license information.      ML and enhanced MD by presenting different shared viewpoints. It offers a

                                                                                                                                                               comprehensive overview of this rapidly evolving field, which can be difficult

                                                                                                                                                               to stay updated on. We highlight successful strategies such as dimensional-

                                                                                                                                                               ity reduction, reinforcement learning, and flow-based methods. Finally, we

                                                                                                                                                               discuss open problems at the exciting ML-enhanced MD interface.



                                                                                                                                                               347

Downloaded from www.annualreviews.org. Guest (guest) IP: 185.143.220.236 On: Wed, 30 Jul 2025 10:25:36         1. INTRODUCTION



                                                                                                               Molecular dynamics (MD) simulations play a crucial role in the field of physical chemistry and

                                                                                                               allied sciences, offering a powerful tool to investigate the intricate motions and behaviors of atoms

                                                                                                               and molecules. Acting as a virtual microscope, MD allows scientists to explore the dynamic aspects

                                                                                                               underlying complicated processes. In addition to the deterministic forces described by Newton's

                                                                                                               laws, MD simulations incorporate thermostats to sample the canonical, constant number, volume,

