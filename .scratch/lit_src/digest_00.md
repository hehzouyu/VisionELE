# 1-s2.0-S0278612522001352-main.txt

                                    Journal of Manufacturing Systems 64 (2022) 561-577


                                          Contents lists available at ScienceDirect





                             Journal of Manufacturing Systems





                               journal homepage: www.elsevier.com/locate/jmansys





Semi-supervised multi-scale attention-aware graph convolution network for


intelligent fault diagnosis of machine under extremely-limited


labeled samples





Zongliang Xie a, Jinglong Chen a,*, Yong Feng a, Shuilong He b,*





a State Key Laboratory for Manufacturing and Systems Engineering, Xi'an Jiaotong University, Xi'an 710049, PR China


b School of Mechanical and Electrical Engineering, Guilin University of Electronic Technology, Guilin 541004, PR China





ARTICLE INFO                 ABSTRACT





Keywords:                    Labeled data are generally scarce in engineering practice, while data-driven methods fail to mine the correlations


Intelligent fault diagnosis  between samples to utilize the rich unlabeled data, so they cannot achieve satisfactory performance under


Graph convolution network    limited labeled data. To address this problem, a semi-supervised multi-scale attention-aware graph convolution


Multi-scale                  network (MSA-GCN) is proposed for fault diagnosis under extremely-limited labeled samples. First, available


Attention mechanism          labeled data are transformed with unlabeled data into a graph via determining the k-nearest neighbors in fre�


Semi-supervised learning     quency domain to construct the neighbor relations. To obtain the useful structural and feature information of


                             unlabeled samples from different neighborhoods, multi-scale graph convolution is implemented to aggregate


                             multi-scale information for labeled samples. Besides, attention mechanism is utilized and a novel adaptive


                             feature fusing layer is designed to achieve cross-scale information fusion of different neighborhoods. With semi-


                             supervised graph learning, the proposed method can fully utilize topological and feature information from un�


                             labeled samples, resulting in a powerful classifier using only few labeled samples. The proposed method is fully


                             verified on three bearing datasets, experimental results show that MSA-GCN can reach an identification accuracy


