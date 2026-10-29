# Domain_Adversarial_Graph_Convolutional_Network_for_Fault_Diagnosis_Under_Variable_Working_Conditions.txt

IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. 70, 2021                                           3515010



Domain Adversarial Graph Convolutional

    Network for Fault Diagnosis Under

        Variable Working Conditions



Tianfu Li , Graduate Student Member, IEEE, Zhibin Zhao , Chuang Sun , Member, IEEE,

      Ruqiang Yan , Senior Member, IEEE, and Xuefeng Chen , Senior Member, IEEE



   Abstract-- Unsupervised domain adaptation (UDA)-based                        data era, and it has been widely used in modern industries

methods have made great progress in mechanical fault diagnosis                  with the goal to monitor the state of equipment and reduce

under variable working conditions. In UDA, three types of infor-                the downtime [4].

mation, including class label, domain label, and data structure,

are essential to bridging the labeled source domain and unlabeled                  Under the assist of computer and deep learning techniques,

target domain. However, most existing UDA-based methods use                     a great improvement has been achieved for the performance

only the former two information and ignore the modeling of data                 of fault diagnosis methods [5]-[7]. These fault diagnosis

structure, which make the information contained in the features                 methods based on deep learning (DL) can learn meaningful

extracted by the deep network incomplete. To tackle this issue,                 features from data via the constructed deep model. Compared

a domain adversarial graph convolutional network (DAGCN) is                     with the traditional fault diagnosis methods [8]-[10], which

proposed to model the three types of information in a unified deep              need feature engineering and expert knowledge, DL-based

network and achieving UDA. The first two types of information                   methods can obtain better diagnostic accuracy. For example,

are modeled by the classifier and the domain discriminator,                     Li et al. [11] proposed a transferable sparse autoencoder that

respectively. In data structure modeling, a convolutional neural                was regularized by nonnegativity-constraint, and achieved fault

network (CNN) is first employed to exact features from input                    diagnosis when only a few labeled data can be obtained. Ding

signals. After that, the CNN features are input to the proposed                 and He [12] proposed to use a convolutional neural network

graph generation layer to construct instance graphs by mining                   (CNN) to mine the energy-fluctuated multiscale features from

the relationship of structural characteristics of samples. Then,                wavelet packet energy images for fault diagnosis of spindle

the instance graphs are modeled by a graph convolutional                        bearing.

network, and the maximum mean discrepancy metric is leveraged

to estimate the structure discrepancy of instance graphs from                      Though DL-based methods have made a great progress

different domains. Experimental results conducted on two case                   in fault diagnosis of machinery, some limitations are still

studies demonstrate that the proposed DAGCN can not only                        associated with these methods. On the one hand, most current

obtain the best performance among the comparison methods, but                   DL-based methods hold the assumption that the samples from

also can extract transferable features for domain adaptation. The               source domain and the samples from target domain share

code library is available at: https://github.com/HazeDT/DAGCN.                  the identical probability distribution. However, in an actual

                                                                                industrial scenario, this assumption does not hold. Equipment

   Index Terms-- Fault diagnosis, graph convolutional net-                      operating conditions change in real time, due to equipment

work (GCN), unsupervised domain adaptation (UDA), variable                      wear and environmental noise during operation [13]. This

working conditions.                                                             leads to distribution discrepancies between the collected data

                                                                                and causes performance degradation while applying the model

                          I. INTRODUCTION                                       trained with the previously collected data to a new task. On the

                                                                                other hand, obtaining sufficient labeled training samples are

I NTELLIGENT fault diagnosis acts a vital role in the cur-                      labor-intensive and expensive, which limit the application of

   rent Prognostics and Health Management system [1]-[3].                       intelligent fault diagnosis methods in the industry.

In the last decade, intelligent fault diagnosis techniques have

