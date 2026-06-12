# Gene expression

# Inferring gene regulatory networks from single-cell transcriptomics based on graph embedding

Yanglan Gan 1 , Jiacheng Yu1 , Guangwei Xu1 , Cairong Yan1 , Guobing Zou2,�

1 School of Computer Science and Technology, Donghua University, Shanghai 201620, China 2 School of Computer Engineering and Science, Shanghai University, Shanghai 200444, China

�Corresponding author. School of Computer Engineering and Science, Shanghai University, Shanghai 200444, China. E-mail: gbzou@shu.edu.cn Associate Editor: Peter Robinson

# Abstract

Motivation: Gene regulatory networks (GRNs) encode gene regulation in living organisms, and have become a critical tool to understand complex biological processes. However, due to the dynamic and complex nature of gene regulation, inferring GRNs from scRNA-seq data is still a challenging task. Existing computational methods usually focus on the close connections between genes, and ignore the global structure and distal regulatory relationships.

Results: In this study, we develop a supervised deep learning framework, IGEGRNS, to infer GRNs from scRNA-seq data based on graph embedding. In the framework, contextual information of genes is captured by GraphSAGE, which aggregates gene features and neighborhood structures to generate low-dimensional embedding for genes. Then, the k most influential nodes in the whole graph are filtered through Top-k pooling. Finally, potential regulatory relationships between genes are predicted by stacking CNNs. Compared with nine competing supervised and unsupervised methods, our method achieves better performance on six time-series scRNA-seq datasets.

Availability and implementation: Our method IGEGRNS is implemented in Python using the Pytorch machine learning library, and it is freely available at https://github.com/DHUDBlab/IGEGRNS.

# 1 Introduction

The identity and behavior dynamics of cells are governed by complex gene interactions, which in turn define cellular morphology and functions (Daskalaki et al. 2018). Gene regulatory networks (GRNs) can model the causal regulatory relationships between transcription factors (TFs) and their target genes. In GRNs, the regulatory relations among genes are represented as graphs, where nodes are regulators and their target genes, and edges represent that there exist regulatory relationships between genes. They have become essential tools for interpreting biological processes and identifying molecular regulators and biomarkers in complex diseases (Sec¸ilmis¸ et al. 2022). Therefore, inferring GRNs based on gene expression profiles is a long-standing computational challenge in system biology research field (Iacono et al. 2019).

A plethora of computational approaches have been developed for inferring GRNs from bulk expression data and single-cell RNA-seq data (Pratapa et al. 2020, Zhao et al. 2021). Generally, the existing methods can be classified into two main categories, unsupervised methods and supervised methods (Bravo Gonz�alez-Blas et al. 2023). Unsupervised methods explore underlying patterns and structures from the gene expression data, and then infer regulatory interactions without a known network. GENIE3 is based on decision trees, which infer the regulatory interactions of each gene independently by an integrated tree-based approach (Huynh-Thu et al. 2010). PPCOR infers GRNs by calculating partial and semipartial correlation coefficients between genes (Kim 2015). PIDC utilizes the multivariate information theory to reconstruct undirected regulatory networks among genes (Chan et al. 2017). SCODE infers regulatory networks based on ordinary differential equations and linear regression (Matsumoto et al. 2017). SINCERITIES quantifies the distance between two cumulative distribution functions of gene expressions from subsequent time points, and employs regularized linear regression to infer directed regulatory relationships among genes (Papili Gao et al. 2018). BiXGBoost respectively infers the regulatory and regulated relationships of genes through gradient boosting decision trees, and integrates the forward and reverse relationships to generate consistent gene score ranking relationships.(Zheng et al. 2019). DeepSEM is also an unsupervised method based on betavariant self-encoder, whose encoder takes the expression profile of one gene at a time as the input feature of the neural network, and later learns gene interaction relationships through a multilayer perceptron (Shu et al. 2021). Although much progress has been made, inferring GRNs from scRNAseq data is still challenging, due to its high sparsity, noise, and dropout events.

Different from unsupervised methods, supervised methods exploit on not only gene expression profiles, but also prior information to infer GRNs, such as known gene interactions, organism, or tissue information (Razaghi-Moghadam and Nikoloski 2020). Recently, as deep learning models can better handle large-scale and high-dimensional data, researchers gradually rely on the powerful representation learning capability of deep neural networks to capture complex nonlinear relationships and infer GRNs from gene expression profiles (Muzio et al. 2021, Greener et al. 2022). As an early supervised algorithm, CNNC transforms the coexpression data of gene pairs into histograms, and then deep convolutional neural networks (CNNs) are utilized to learn the relationships between genes. However, it is time-consuming to transform numerous gene pairs into matrices (Yuan and Bar-Joseph 2019). GNE uses gene expression profiles and network topology to predict gene interactions through ANN (Kc et al. 2019). TDL devises a supervised framework which represents the data as 3D tensors and trains convolutional and recurrent neural networks for predicting interactions (Yuan 2021). DGRNs are a hybrid deep learning model that effectively extracts temporal information, which constructs an input expression matrix by extracting special correlation vectors representing gene expression features, and inferring temporal and spatial features through GRU and CNN, respectively (Zhao et al. 2022). DeepRIG first constructs a prior regulatory graph by transforming the gene expression profiles into the coexpression mode, then adopts a graph autoencoder model to learn gene latent embeddings and to infer the GRN (Wang et al. 2023). STGRNs is a transformer-based method for inferring GRNs from scRNA-seq data. It converts gene pairs into contiguous subvectors, which can be used as input for the transformer encoder (Xu et al. 2023). In general, these deep learning based methods infer GRNs through two primary steps, first converting gene expression data into a suitable data format, and then employing deep neural network models to predict the regulatory relationships. Although these deep neural networks have achieved notable success in various biological tasks, these CNN model-based methods still encounter with some limitations in GRN inference. On one hand, the generation of image data not only gives rise to unanticipated noise but also conceals certain original data features. On the other hand, since this procedure alters the format of the data, the results predicted by these approaches is lack of biological explainability.

To address these limitations, we propose a supervised deep learning framework, IGEGRNS, to infer GRNs from scRNAseq data through graph embedding. We convert GRN inference task to linkage prediction problem, predicting the existence of a directed edge between TFs and target genes. IGEGRNS formulates gene–gene relationships with graph neural networks, and learns low-dimensional embeddings of gene pairs using GraphSAGE. Contextual information of genes is captured by GraphSAGE, which aggregates gene features and neighborhood structures to generate low-dimensional embedding for genes. Meanwhile, the k most influential nodes in the whole graph are filtered through Top-k pooling. Then, the regulatory relationships between TFs and target genes are further learned by stacking CNNs, which enhance the network representation and better adapt to complex input data by extracting features layer by layer. Compared with nine competing supervised and unsupervised methods, our approach achieves better performance on six time-series scRNA-seq datasets.

# 2 Materials and methods

# 2.1 The IGEGRNS framework

As predicting the regulatory relationships among genes is essential for inferring GRNs from observed gene expression data, linkage prediction is a fundamental problem in the study of GRNs. IGEGRNS converts the GRNs inference into a linkage prediction problem, determining whether there are regulatory edges between TFs and target genes. As illustrated in Fig. 1, IGEGRNS is a supervised deep learning framework, inferring GRNs from scRNA-seq data through graph embedding. Overall, the linkage prediction process can be divided into two main steps. First, the embedding of gene pairs is learned through GraphSAGE (Hamilton et al. 2017). Based on the gene expression data and prior knowledge, GraphSAGE generates low-dimensional embedding for genes, iteratively aggregating the information of gene nodes and their neighboring nodes. Meanwhile, Top-k pooling filters the top k nodes with the highest influence on the whole graph (Gao and Ji 2019). Then we concatenate the corresponding embedded vectors of the gene pairs and the feature vectors of the selected top k gene nodes. Second, based on the concatenated feature matrix, we predict whether there is a regulatory edge between each gene pair. The prediction module consists of a stacked 3-layer CNN and a fully connected layer. For the stacked 3-layer CNN, each feature extraction layer includes a regularization layer, a CNN layer, and a maximum pooling layer. Further, the high-level features learned from these three feature extraction layers are concatenated and fed into the fully connected layer, and scored by the Sigmoid function.

# 2.1.1 Learning gene node embedding

The original gene expression data is a matrix of n×m, where n denotes the number of genes and m refers to the number of cells. Our task is to reconstruct GRNs based on the given gene expression profiles. To learn the low-dimensional embedding of gene nodes, we adopt GraphSAGE to perform graph embedding. Due to its versatility in neighbor sampling and aggregation techniques, GraphSAGE enables effective learning of both local and global features within the network topology (Hamilton et al. 2017). By leveraging GraphSAGE, the proposed model can better capture the intricate interactions between genes, resulting in high-quality embedding vectors derived from gene expression data and network interactions.

For each node $\nu ( \nu \in \{ 1 , 2 , 3 , \ldots , n \} ) , \mathcal { N } ( \nu )$ is referred to the neighbor set of node v. Here, the mean function is chosen as the aggregation function, and then the d-dimensional neighbor node aggregation of node v is represented as:

$$
b _ {\mathcal {N} (\nu)} ^ {l} = \frac {1}{| \mathcal {N} (\nu) |} \sum_ {u \in \mathcal {N} (\nu)} b _ {u} ^ {l}, \tag {1}
$$

where l represents the number of hops of neighboring nodes that each vertex can aggregate.

The embedding of node v is learned based on its l-hop neighboring nodes. Therefore, a new vector $b _ { \nu } ^ { l }$ can be used to represent the embedding of node v, which captures the l-hop neighborhood information of node v:

$$
h _ {\nu} ^ {l} \leftarrow \sigma \Big (W \cdot C O N C A T \Big (h _ {\nu} ^ {l - 1}, h _ {\mathcal {N} (\nu)} ^ {l} \Big) \Big), \tag {2}
$$

where σ denotes a Relu activation function, W denotes the parameter matrix to be learned, and CONCAT represents the concatenating operation.

![](images/e6ead0a63a76fc9cbd37a0a782500d789fb58deaef8b42f260a920fde009c56a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Cells"] --> B["Top-k Pooling"]
    B --> C["Top 1 Top 2 Top k"]
    C --> D["Edge Matrix"]
    E["Linkage Information"] --> F["GraphSAGE"]
    F --> G["Generate Gene Embedding"]
    G --> H["Embedding Matrix"]
    H --> I["Each Gene Pair"]
    I --> J["Concatenate"]
    J --> K["Gene i"]
    J --> L["Gene j"]
```
</details>

![](images/d1db1cd09a9f043c60f529e3bca37f9aef4fe7aed6a1b5ec089e4bc5f7994b03.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Edge Matrix"] --> B["Feature Extraction Module 1"]
    B --> C["Feature Extraction Module 2"]
    C --> D["Feature Extraction Module 3"]
    D --> E["Linear Layer"]
    E --> F["Edge Scores"]
    F --> G["Ranking Scores"]
    G --> H["GRN Inference"]
    B --> I["Every Feature Extraction Module"]
    I --> J["Convolution Layer"]
    I --> K["Maximum Pooling Layer"]
    J --> L["X1"]
    K --> M["X2"]
    K --> N["X3"]
    L --> O["concatenate"]
    M --> O
    N --> O
    style A fill:#f9f,stroke:#333
    style F fill:#ccf,stroke:#333
    style G fill:#cfc,stroke:#333
```
</details>

Figure 1. The overview of IGEGRNS. IGEGRNS is composed of two main modules: (A) Learning gene node embedding module. To learn the lowdimensional embedding of genes, GraphSAGE is adopted to iteratively aggregate the node feature and its l-hop neighboring nodes. Meanwhile, Top-k pooling filters the top k nodes with the highest influence on the whole graph. (B) Predicting gene interactions module. This module consists of stacked CNNs and a fully connected layer. Stacked CNN further learns the high-level representation of the gene pairs, which is used to predict the regulatory relationship between TFs and target genes

Based on the given gene expression matrix and the prior link knowledge, GraphSAGE conducts graph embedding and learns a low-dimensional embedded representation for each gene node. By leveraging the known regulatory relationships in the training set, the model can effectively learn accurate embedding vectors for gene nodes, consequently facilitating the inference of GRNs. Then, the embedding matrix is composed of all gene node embeddings is represented as $\mathbf { \bar { \boldsymbol { X } } } = \{ b _ { 1 } ^ { l } , b _ { 2 } ^ { l } , \dots , b _ { n } ^ { l } \} , \boldsymbol { X } \in \mathbb { R } ^ { n \times d }$ , where d denotes the feature dimension of the gene node.

Meanwhile, We adopt Top-k pooling strategy to select the top k nodes with the highest influence on the whole graph (Gao and Ji 2019). As illustrated in Fig. 2, the detailed process is formulated as below:

$$
\overrightarrow {v e c} = \sigma \left(\frac {X \overrightarrow {p}}{\| \overrightarrow {p} \|}\right), \tag {3}
$$

$$
\overrightarrow {i d x} = \operatorname{topk} (\overrightarrow {\nu e c}), \tag {4}
$$

$$
X ^ {\prime} = \left(X \odot \tanh (\overrightarrow {v e c})\right) _ {\overrightarrow {i d x}}, \tag {5}
$$

$$
A ^ {\prime} = A \underset {i d x, i d x} {\longrightarrow}, \tag {6}
$$

![](images/9d2211e9c2cf46ee5d2a2b1a6655f509d32cacf7c1d3c8f72bd1ec066a013c4f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Embedding Matrix"] --> B["×"]
    B --> C["p̅"]
    C --> D["1/||p̅||"]
    D --> E["vec"]
    E --> F["×"]
    F --> G["Adjacency Matrix"]
    G --> H["idx"]
    H --> I["Adjacency Matrix"]
    I --> J["A'"]
    K["Top 1"] --> L["X'"]
    M["Top 2"] --> L
    N["Top k"] --> O["X'"]
    P["+"] --> Q["×"]
    Q --> R["IDX"]
    R --> S["IDX"]
    S --> T["Adjacency Matrix"]
```
</details>

Figure 2. Illustration of Top-k pooling process. Top-k pooling filters the k most influential nodes based on the embedding matrix generated by GraphSAGE

where σ denotes a Relu activation function, where $\overrightarrow { p }$ is the learnable vector, and the projection fraction of $\boldsymbol { p }$ is used to determine which node to be discarded. k � k denotes the L2- norm operator, topk function selects the top-k index from the input vector, $\overrightarrow { i d x }$ is an index operation to obtain slices according to the specified index, and � is the element multiplication.

Next, we construct the input matrix for the subproblem of determining the existence of a directed edge from gene i to gene j. The input matrix $E _ { i , j }$ consists of $k + 2$ vectors, including the embedding vector $b _ { i } ^ { l }$ of gene $i , b _ { i } ^ { l }$ of gene j, and the vectors $b _ { t _ { 1 } } , b _ { t _ { 2 } } , \ldots , b _ { t _ { k } }$ obtained by Top-k pooling, which is represented as:

$$
E _ {i, j} = \left[ \begin{array}{l} b _ {i} ^ {l} \\ b _ {j} ^ {l} \\ b _ {t _ {1}} \\ \dots \\ b _ {t _ {k}} \end{array} \right], \tag {7}
$$

where $i , j \in \nu , E _ { i j }$ is a ðk þ 2Þ × d matrix.

# 2.1.2 Predicting gene interactions

The core of the gene linkage prediction module consists of a stacked three-layer CNN and a fully connected layer. Specifically, the stacked CNN is introduced to conduct further feature extraction, which is a three-layer CNN stacking model. Each feature extraction layer includes a regularized BatchNorm layer, a CNN convolutional layer and a maximum pooling layer. The outputs of these three feature extraction layers are concatenated before feeding into the fully connected layer. The purpose of this step is to further learn the high-level feature representation, increase the nonlinear capability of the network, and better capture the complex relationships hidden in the data. The concatenated vectors are fed into the lazy linear layer and the ordinary linear layer, and finally the edges $e _ { i j }$ are scored by the Sigmoid function. The predicted scores, denoted as ${ \hat { y _ { i } } } { \hat { j } } ,$ are normalized in the interval [0,1], which represents the probability that TF i regulates target gene j. We train the model using a binary cross-entropy loss function as below:

$$
\begin{array}{l} \mathrm{BCELoss} = - \frac {1}{N} \sum_ {s = 1} ^ {N} y _ {s} \cdot \log \left(\hat {y} _ {s}\right) \tag {8} \\ + \left(1 - y _ {s}\right) \cdot \log \left(1 - \hat {y} _ {s}\right). \\ \end{array}
$$

where N denotes the number of samples involved in training, $y _ { s }$ denotes the true label of the sth sample, and $\hat { y _ { s } }$ denotes the model prediction.

# 2.2 Datasets

The proposed method IGEGARNS is evaluated on six timeseries scRNA-seq datasets, including human embryonic stem cells (hESCs), human mature hepatocytes (hHEP), mouse embryonic stem cells (mESC), mouse hematopoietic stem cells with an erythroid-lineage profile (mHSC-E), mouse hematopoietic stem cells with a granulocyte-monocyte profile (mHSC-GM) and mouse hematopoietic stem cells with a lymphoid lineage (mHSC-L). For these datasets, the corresponding cell-type-specific networks provided in previous studies are regarded as the reference network for the evaluation (Xu et al. 2013, Oki et al. 2018, Moore et al. 2020). We preprocess these six scRNA-seq datasets and infer the gene interactions as BEELINE (Pratapa et al. 2020). We respectively select 500 and 1000 the most differential genes for the inference of GRNs.

For the proposed method, the inference of GRNs is transformed into a linkage prediction problem, predicting whether there are directed regulatory edges for gene pairs. Therefore, we evaluate the supervised method by 5-fold crossvalidation. We define the TFs appearing in the reference network as TFs and the target genes as targets. We consider the edges existing in the reference network as positive. Otherwise, the edges are regarded as negative. Generating negative edges can help the algorithm to better distinguish the real regulatory relationships, and reduce the sensitivity to random noise, and thus optimize the algorithm performance. We mix positive and negative edges and randomly divide them into five equal parts. We choose four parts as the training set and the remaining part as the test set. We average the results of 5-fold cross-validation to obtain the final AUROC and AUPRC scores. For the compared unsupervised model, we use the results of the framework BEELINE (Pratapa et al. 2020).

# 2.3 Performance metrics

To compare the performance of different methods in inferring GRNs, we adopt two commonly used metrics AUROC and AUPRC. For GRN inference, changing the thresholds leads to different GRNs, and AUROC and AUPRC allow comparing the performance of different GRN inference algorithms under different thresholds. Specifically, ROC is a curve with false positive rate as the horizontal axis and true positive rate as the vertical axis. AUROC is the area under the ROC curve, and AUPRC is the area under the Precision-Recall curve. Higher AUROC and AUPRC scores indicate better performance.

$$
\mathrm{FPR} = \frac {\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}}, \tag {9}
$$

$$
\mathrm{TPR} = \frac {\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}, \tag {10}
$$

$$
\text { Precision } = \frac {\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}, \tag {11}
$$

$$
\text { Recall } = \frac {\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}. \tag {12}
$$

where TP denotes the number of true regulatory edges that are predicted to be positive, FN denotes the number of true regulatory edges that are predicted to be negative, FP denotes the number of false regulatory edges that are predicted to be positive, and TN denotes the number of false regulatory edges that are predicted to be negative.

In our experiments, we average the results of 5-fold crossvalidation to obtain the final AUROC and AUPRC scores.

# 3 Results

# 3.1 Implementation details

The proposed method adopts GraphSAGE to aggregate gene features and neighboring nodes, and to generate lowdimensional embedding. Specifically, the output dimension is set to 256, the aggregator function is MEAN aggregator function. We aggregate 1-hop neighbor node information. To evaluate the efficacy of different neighbor node configurations, we conduct experiments utilizing 1-hop, 2-hop, and 3- hop neighbor nodes. The results demonstrate that aggregating 1-hop neighbors achieves better performance, indicating that the representation of nodes can be well represented by the information of their direct neighbors. Aggregating neighbors at a further distance may increase computational costs and introduce sparsity during information propagation. For the Top-k pooling strategy, the parameter k is set to 1. The size of the CNN convolution kernel is set to 2. In addition, our learning rate is initially set to 0.01, and with every 10 epochs of training, the learning rate decreasing to 80% of the original.

In the experiments, all models are trained on the computer with configurations of Intel Xeon Silver 4208 Processor @ 2.10 GHz, 8 cores and 32GB RAM, 24GB NVIDIA GeForce RTX 3090.

# 3.2 Performance comparsion with other methods on benchmark datasets

To evaluate the performance of IGEGRNS, we apply the proposed GRN inference method to six time-series scRNA-seq datasets, including hESC, hHEP, mESC, mHSC-E, mHSC-GM, and mHSC-L. We compared it with nine competing algorithms, which have been proven to achieve good performance. According to the previous comparative analysis, we select GENIE3, GRNBoost2, PIDC, SCODE, and DeepSEM from the existing unsupervised methods. Specifically, GENIE3 and GRNBoost2 both adopt tree-based regressions to determine the gene sets that are coexpressed with TFs. PIDC method is based on multivariate information theory. SCODE applies ordinary differential equations to infer GRN. DeepSEM jointly models the GRN and the transcriptome by generating the SEM with a beta-VAE framework. For the supervised methods, CNNC, GNE, DeepRIG, and DGRNs are included in the comparison. CNNC is a supervised GRN inference method based on deep CNNs. GNE applies multilayer perception to encode gene expression profiles to predict gene interactions. DeepRIG infers GRNs through prior knowledge generated by WGCN and graph autoencoder GAE. DGRNS is a hybrid deep learning models based on CNNs and recurrent neural networks. Following the BEELINE framework, we consider only highly variable TFs and the top 500 and 1000 most differential genes for each dataset. We take the cell-type-specific network as the ground truth to evaluate the inferred GRNs. The widely used AUROC and AUPRC are adopted as the evaluation metrics.

Figure 3 shows the performance of these compared methods on the six scRNA-seq datasets. Overall, the proposed method IGEGRNS achieves the highest AUROC on all these datasets. AUPRC outperforms the compared methods on five datasets except the mESC dataset. In scenarios involving 500 TFs, IGEGRNS demonstrates an average AUROC improvement of 5.6% and an AUPRC improvement of 6.7% compared to the second-ranked algorithm. The highest AUROC improvement 8.2% is achieved on the mHSC-GM dataset, while the highest AUPRC improvement 13.8% is observed on the mHSC-L dataset. In situations with 1000 TFs, our method improves AUROC by 5.0% and AUPRC by 6.6% compared to the suboptimal algorithm. The highest AUROC improvement 7.2% is observed on the hHEP dataset and the highest AUPRC improvement 14.1% is occurred on the mHSC-L dataset. In addition, the proposed method exhibits substantial improvements in both AUROC and AUPRC compared to the unsupervised method, demonstrating the advantage of supervised algorithms over unsupervised ones.

# 3.3 Performance comparsion among different model structures

The results show that IGEGRNS outperforms the compared unsupervised and supervised methods. To verify the reasonability and feasibility of each module of IGEGRNS, we further conduct a comparative analysis to evaluate the effectiveness of different model variants. As shown in Table 1 and Table 2, we compare the performance of IGEGRNS with different aggregation functions, different k values for Top-k pooling in the Learning gene node embedding module, and reduce the number of network layers to one simplified CNN in the Predicting gene interactions module. The results demonstrate that the simple MEAN aggregation function is most effective for aggregating the neighboring nodes. Although LSTM is more complex, it might encounter difficulties in extracting effective information of nodes and their neighbors for this task. The max function might miss part information of neighboring nodes. For the stacked CNN, replacing the 3- layer stacked CNN with a simple 1-layer CNN results in an obvious decrease in performance. The results indicate that the simple 1-layer CNN cannot replace the role of the 3-layer stacked CNN, which can extract higher-level features from the complex spatial structure with local features. We also remove the nodes selected by Top-k pooling, and the performance of IGEGRNS without Top-k pooling significantly decreased. This is probably because Top-k pooling not only selects the nodes with the greatest global impact but also serves as a constraint for the model parameters, reducing the possibility of overfitting.

![](images/68e79cc4de5e11ba69a5f382399d56ba1cf32fc4dd241474609c873b63865019.jpg)

<details>
<summary>heatmap</summary>

| AUROC | IGEGRNS | DGRNS | GNE | CNNC | DeepRIG | DeepSEM | SCODE | PIDC | GRNBOOST2 | GENIE3 |
|---|---|---|---|---|---|---|---|---|---|---|
| hHEP | 0.857 | 0.784 | 0.786 | 0.638 | 0.619 | 0.564 | 0.495 | 0.502 | 0.498 | 0.498 |
| mHSC-L | 0.847 | 0.7 | 0.777 | 0.659 | 0.688 | 0.549 | 0.51 | 0.498 | 0.5 | 0.499 |
| mESC | 0.846 | 0.832 | 0.794 | 0.719 | 0.746 | 0.513 | 0.511 | 0.513 | 0.516 | 0.51 |
| mHSC-E | 0.914 | 0.837 | 0.822 | 0.673 | 0.658 | 0.53 | 0.51 | 0.497 | 0.492 | 0.49 |
| mHSC-GM | 0.909 | 0.824 | 0.827 | 0.686 | 0.669 | 0.532 | 0.517 | 0.498 | 0.498 | 0.495 |
| hESC | 0.798 | 0.776 | 0.68 | 0.67 | 0.512 | 0.576 | 0.502 | 0.499 | 0.501 | 0.501 |
</details>

![](images/915f78ea93838a912323bb3342b0362e7a4485f81f050afaaadf03722e61953e.jpg)

<details>
<summary>heatmap</summary>

| AUPRC | IGEGRNS | DGRNS | GNE | CNNC | DeepRIG | DeepSEM | SCODE | PIDC | GRNBOOST2 | GENIE3 |
|---|---|---|---|---|---|---|---|---|---|---|
| hHEP | 0.766 | 0.674 | 0.642 | 0.476 | 0.468 | 0.404 | 0.344 | 0.399 | 0.39 | 0.386 |
| mHSC-L | 0.848 | 0.71 | 0.704 | 0.645 | 0.698 | 0.535 | 0.486 | 0.498 | 0.507 | 0.501 |
| mESC | 0.74 | 0.744 | 0.653 | 0.486 | 0.557 | 0.32 | 0.314 | 0.317 | 0.321 | 0.324 |
| mHSC-E | 0.933 | 0.869 | 0.803 | 0.736 | 0.723 | 0.589 | 0.589 | 0.575 | 0.571 | 0.566 |
| mHSC-GM | 0.928 | 0.838 | 0.79 | 0.684 | 0.703 | 0.562 | 0.552 | 0.526 | 0.524 | 0.532 |
| hESC | 0.5 | 0.48 | 0.345 | 0.257 | 0.166 | 0.193 | 0.154 | 0.152 | 0.157 | 0.154 |
</details>

![](images/b0b893e7e5b3e681c8d365b8e35065e12cfbb2884c87cef7dc2e3f8e20726831.jpg)

<details>
<summary>heatmap</summary>

| AUROC | IGGRNS | DGRNS | GNE | CNNC | DeepRIG | DeepSEM | SCODE | PIDC | GRMBOOST2 | GENIE3 |
|---|---|---|---|---|---|---|---|---|---|---|
| hHEP | 0.863 | 0.791 | 0.79 | 0.654 | 0.628 | 0.563 | 0.496 | 0.498 | 0.516 | 0.51 |
| mHSC-L | 0.842 | 0.707 | 0.786 | 0.637 | 0.696 | 0.553 | 0.496 | 0.502 | 0.518 | 0.511 |
| mESC | 0.841 | 0.827 | 0.807 | 0.722 | 0.752 | 0.513 | 0.514 | 0.502 | 0.519 | 0.507 |
| mHSC-E | 0.915 | 0.839 | 0.844 | 0.703 | 0.681 | 0.536 | 0.514 | 0.502 | 0.509 | 0.501 |
| mHSC-GM | 0.915 | 0.853 | 0.837 | 0.702 | 0.677 | 0.537 | 0.525 | 0.498 | 0.51 | 0.501 |
| hESC | 0.808 | 0.784 | 0.682 | 0.715 | 0.535 | 0.593 | 0.509 | 0.515 | 0.5 | 0.494 |
</details>

![](images/c03fc10e8159ad5ea1643388971e4fa84a571183e04425bc3a21328d18ece2b4.jpg)

<details>
<summary>heatmap</summary>

| AUPRC | IGGRNS | DGRNS | GNE | CNNC | DeepRIG | DeepSEM | SCODE | PIDC | GRNBOOST2 | GENIE3 |
|---|---|---|---|---|---|---|---|---|---|---|
| hHEP | 0.781 | 0.679 | 0.653 | 0.486 | 0.473 | 0.41 | 0.343 | 0.387 | 0.375 | 0.379 |
| mHSC-L | 0.84 | 0.693 | 0.699 | 0.604 | 0.708 | 0.527 | 0.485 | 0.49 | 0.49 | 0.493 |
| mESC | 0.733 | 0.737 | 0.649 | 0.509 | 0.563 | 0.318 | 0.313 | 0.321 | 0.325 | 0.326 |
| mHSC-E | 0.936 | 0.873 | 0.804 | 0.756 | 0.74 | 0.562 | 0.563 | 0.564 | 0.552 | 0.541 |
| mHSC-GM | 0.937 | 0.876 | 0.796 | 0.704 | 0.708 | 0.563 | 0.56 | 0.531 | 0.532 | 0.532 |
| hESC | 0.521 | 0.491 | 0.339 | 0.269 | 0.172 | 0.199 | 0.151 | 0.155 | 0.155 | 0.151 |
</details>

Figure 3. AUROC and AUPRC scores of different GRN inference algorithms across the six scRNA-seq datasets. The left panels display AUROC scores for 500 and 1000 TFs datasets, while the right panels display AUPRC scores for the same datasets. The vertical axis represents the six scRNA-seq datasets, and the horizontal axis represents different GRN inference algorithms

Table 1. The AUROC of different model variants on the six scRNA-seq datasets with 1000 TFs.a 

<table><tr><td>AUROC of different model structures</td><td>hHEP</td><td>mHSC-L</td><td>mESC</td><td>mHSC-E</td><td>mHSC-GM</td><td>hESC</td></tr><tr><td>Aggregator selection=MEAN(IGEGRNS)</td><td>0.863</td><td>0.842</td><td>0.841</td><td>0.915</td><td>0.915</td><td>0.808</td></tr><tr><td>Aggregator selection=LSTM</td><td>0.853</td><td>0.829</td><td>0.826</td><td>0.904</td><td>0.908</td><td>0.774</td></tr><tr><td>Aggregator selection=MAX</td><td>0.854</td><td>0.829</td><td>0.836</td><td>0.903</td><td>0.908</td><td>0.774</td></tr><tr><td>3-layer stacked CNN(IGEGRNS)</td><td>0.863</td><td>0.842</td><td>0.841</td><td>0.915</td><td>0.915</td><td>0.808</td></tr><tr><td>1-layer CNN</td><td>0.851</td><td>0.825</td><td>0.829</td><td>0.901</td><td>0.895</td><td>0.801</td></tr><tr><td>Top-k pooling, k = 1(IGEGRNS)</td><td>0.863</td><td>0.842</td><td>0.841</td><td>0.915</td><td>0.915</td><td>0.808</td></tr><tr><td>Without Top-k pooling</td><td>0.845</td><td>0.837</td><td>0.828</td><td>0.901</td><td>0.909</td><td>0.802</td></tr></table>

a By default, IGEGRNS uses the MEAN aggregator, 3-layer stacked CNN, Top-k pooling with k ¼ 1.

To explore how embedding vector dimensionality affects gene interaction prediction, we further conduct a comparative analysis of IGEGRNS performance with four different embedding vector sizes, including 64, 128, 256, and 512. The results are presented in Fig. 4. As the dimensionality increases, the performance of IGEGRNS improves initially, reaches its peak at size 256, and then declines. This trend suggests that smaller embedding vectors may lead to information loss, while larger ones could introduce redundant information and noise, posing challenges for CNNs in extracting key features. Consequently, for optimal performance, we set the embedding vector dimensionality to 256 as the default experimental setting. Furthermore, we compare the performance of aggregating 1-hop, 2-hop, and 3-hop neighboring nodes on these six datasets. The results indicate that aggregating 1- hop neighbors achieves better performance with less computation cost. For example, on the dataset hHEP, the AUROC with 1-hop, 2-hop, 3-hop neighboring nodes are respectively 0.84, 0.77, 0.75. The AUPRC with 1-hop, 2-hop, and 3-hop neighoring nodes are respectively 0.74, 0.67, 0.66.

# 3.4 The analysis of inferred GRNs on real datasets

To further validate our proposed method, we train the model on the training set and infer the GRNs on two real datasets. The first dataset was derived from the direct reprogramming process of mouse embryonic fibroblasts (MEFs) to myofibroblasts. This dataset contains 405 cells, which were measured at 0, 2, 5, and 22 days, respectively (Treutlein et al. 2016). The second dataset was derived from the differentiation process of hESCs from qualitative endodermal cells. This dataset was measured at 0, 12, 24, 36, 72, and 96 h, respectively (Chu et al. 2016), and it contains 758 cells. Similar to SCODE (Matsumoto et al. 2017), we use the TFs in Riken TFDB for mouse (Kanamori et al. 2004) and animalTFDB for human (Zhang et al. 2015), and select the top 100 genes with large expression differences for each dataset. To validate the accuracy of the inferred GRNs on these datasets, we compare them with GRNs provided in the regulatory database (http://www.regulatorynetworks.org) (Neph et al. 2012, Stergachis et al. 2014), constructed from DNase footprints and motifs. A total of 666 directed regulatory edges exist in the first reference network and 376 directed regulatory edges exist in the second reference network. During the training and prediction process, we randomly select 20% of real directed regulatory edges as positive samples and equal-sized negative samples as training data. The AUROC of our proposed model on the two datasets are 0.672 and 0.741, while the AUPRC values are 0.447 and 0.452, respectively.

The visualization results are shown in Fig. 5, where the gene nodes are shaded from light to dark based on degree, and the edges are colored in three colors. Red represents edges that are present in the ground-truth networks and are predicted to be positive by the model, green indicates edges that are present in the ground-truth networks but are predicted to be negative, and blue indicates edges that are not present in the ground-truth networks but are predicted to be negative by the model. For the GRN during direct reprogramming of MEF to myoblasts, there are 244 red edges, 256 green edges, 256 blue edges, and the sparsity of the whole network is 0.31. According to the inferred GRN, we observe that genes MYC and EBF1 played significant regulatory roles, with MYC positively regulating 36 genes and EBF1 regulating 35 genes. MYC has been identified as a regulator that directly or indirectly activates genes associated with muscle cell specificity. It is implicated in regulating cell proliferation rates, thereby facilitating the transformation of embryonic fibroblasts into myoblast and promoting muscle differentiation (Luo et al. 2019). Similarly, EBF1 plays a direct role in muscle cell differentiation by promoting the expression of muscle cell-specific genes and modulating signaling pathways pertinent to muscle cell differentiation (Jin et al. 2014). For the GRN during the differentiation process of hESCs in qualitative endoderm cells, there are 149 red edges, 226 green edges, 226 blue edges, and the sparsity of the whole network is 0.17. Based on the inferred GRN, we observe that HAND1 and TCF7 are important to the differentiation process of hESCs in qualitative endoderm cells. Specifically, HAND1 functions as a TF governing the directional differentiation of endodermal cells into specific cell lines (Riley et al. 1998, Firulli et al. 2020). It contributes to maintaining the specificity and function of qualitative endodermal cells through the regulation of specific gene expressions. TCF7 plays a crucial role in embryonic development and cell fate determination. Additionally, TCF7 influences the state and function of embryonic stem cells by regulating the expression of genes associated with stem cell properties (Liang and Liu 2018, Sierra et al. 2018).

Table 2. The AUPRC of different model variants on the six scRNA-seq datasets with 1000 TFs.a 

<table><tr><td>AUPRC of different model structures</td><td>hHEP</td><td>mHSC-L</td><td>mESC</td><td>mHSC-E</td><td>mHSC-GM</td><td>hESC</td></tr><tr><td>Aggregator selection=MEAN(IGEGRNS)</td><td>0.781</td><td>0.840</td><td>0.733</td><td>0.936</td><td>0.937</td><td>0.521</td></tr><tr><td>Aggregator selection=LSTM</td><td>0.776</td><td>0.820</td><td>0.693</td><td>0.923</td><td>0.929</td><td>0.469</td></tr><tr><td>Aggregator selection=MAX</td><td>0.776</td><td>0.819</td><td>0.711</td><td>0.923</td><td>0.926</td><td>0.469</td></tr><tr><td>3-layer stacked CNN(IGEGRNS)</td><td>0.781</td><td>0.840</td><td>0.733</td><td>0.936</td><td>0.937</td><td>0.521</td></tr><tr><td>1-layer CNN</td><td>0.763</td><td>0.813</td><td>0.710</td><td>0.922</td><td>0.909</td><td>0.515</td></tr><tr><td>Top-k pooling, k = 1(IGEGRNS)</td><td>0.781</td><td>0.840</td><td>0.733</td><td>0.936</td><td>0.937</td><td>0.521</td></tr><tr><td>Without Top-k pooling</td><td>0.748</td><td>0.828</td><td>0.722</td><td>0.921</td><td>0.928</td><td>0.518</td></tr></table>

a By default, IGEGRNS uses the MEAN aggregator, 3-layer stacked CNN, Top-k pooling with k ¼ 1.

![](images/373d50fc9d7ccd0e39a93a58a4d0ee664fa38fbdf26f494bb1235e6b3d167842.jpg)

<details>
<summary>line</summary>

| Embedding vector dimensions | hHEP  | mHSC-L | mESC  | mHSC-E | mHSC-GM | hESC  |
| --------------------------- | ----- | ------ | ----- | ------ | ------- | ----- |
| 64                          | 0.845 | 0.838  | 0.815 | 0.895  | 0.870   | 0.772 |
| 128                         | 0.852 | 0.840  | 0.832 | 0.905  | 0.880   | 0.785 |
| 256                         | 0.858 | 0.848  | 0.845 | 0.915  | 0.910   | 0.798 |
| 512                         | 0.855 | 0.833  | 0.822 | 0.895  | 0.892   | 0.795 |
</details>

![](images/1e7dd4100e0d7c864cfd63846b55b3465b339310bfe327c2001b9edd939d35ef.jpg)

<details>
<summary>line</summary>

| Embedding vector dimensions | hHEP  | mHSC-L | mESC  | mHSC-E | mHSC-GM | hESC  |
| --------------------------- | ----- | ------ | ----- | ------ | ------- | ----- |
| 64                          | 0.76  | 0.83   | 0.70  | 0.92   | 0.87    | 0.48  |
| 128                         | 0.77  | 0.83   | 0.72  | 0.92   | 0.89    | 0.49  |
| 256                         | 0.76  | 0.85   | 0.74  | 0.93   | 0.93    | 0.50  |
| 512                         | 0.75  | 0.83   | 0.71  | 0.92   | 0.90    | 0.49  |
</details>

Figure 4. Comparison of AUROC and AUPRC across six scRNA-seq datasets with 500 TFs, varying embedding vector dimensions. The embedding dimensions are respectively set to 64, 128, 256, and 512

![](images/1ffec5b9d01c5830324659c4094fa5185e4d5bfd8b9b007e2170505477b0c7fc.jpg)  
Figure 5. The inferred gene regulatory networks during the process of mouse embryonic fibroblasts (MEF) to myoblasts (the left subfigure) and the process of differentiation of human embryonic stem cells derived from qualitative endodermal cells (the right subfigure). Gene nodes are colored from light to dark in descending order of magnitude. Edges colored in three colors, where red indicates edges that are present in the ground-truth networks and are predicted to be positive by the model, green indicates edges that are present in the ground-truth networks but are predicted to be negative by the model, and blue indicates edges that are not present in the ground-truth networks but are predicted to be negative by this model. According to the inferred GRNs, MYC genes and EBF1 genes play important roles during direct reprogramming of mouse embryonic fibroblasts (MEF) to myofibroblasts. HAND1 and TCF7 genes play important regulatory roles in the gene regulatory network during the differentiation process of human embryonic stem cells from endodermal cells

# 4 Discussion

Due to the complexity and uncertainty of regulatory relationships, it is still a challenge task to infer the regulatory relationships across multiple time nodes. In this paper, we develop a supervised deep learning framework, IGEGRNS, to infer GRNs from scRNA-seq data based on graph embedding. In the framework, contextual information of genes is captured by GraphSAGE, which aggregates gene features and neighborhood structures to generate low-dimensional embedding for genes. Then, the k most influential nodes in the whole graph are filtered through Top-k pooling. Finally, potential regulatory relationships between genes are predicted by stacking CNNs. The experimental results demonstrate that IGEGRNS outperformed nine competing methods on six cell-specific scRNA-seq datasets. The proposed IGEGRNS method shows promising results in terms of inferring GRNs and biological interpretability, implying the high-level relation information among genes and the information of the node neighbors are effective in GRN inference. In the future, with the continued advancement of deep learning models, we will further introduce more effective models to learn feature representation from complex data, and accurately predict the regulatory relationships among genes.

# Conflict of interest

None declared.

# Funding

This work was sponsored in part by the National Natural Science Foundation of China [grant number 62172088] and the Shanghai Natural Science Foundation [grant number 21ZR1400400].

# Data availability

All data used in this study are available in the public database. Gene expression raw data are available in NCBI GEO database under the following accession numbers: human embryonic stem cells (hESC): GSE75748, human mature hepatocytes (hHEP): GSE81252, mouse embryonic stem cells (mESC): GSE98664, mouse blood stem/progenitor cell (mHSC): GSE81682. Reference networks and processed gene expression data are available at https://doi.org/10.5281/zen odo.3378975.

# References

Bravo Gonz�alez-Blas C, De Winter S, Hulselmans G et al. SCENIC þ : single-cell multiomic inference of enhancers and gene regulatory networks. Nat Methods 2023;20:1355–67.   
Chan TE, Stumpf MP, Babtie AC. Gene regulatory network inference from single-cell data using multivariate information measures. Cell Syst 2017;5:251–67.e3.   
Chu L-F, Leng N, Zhang J et al. Single-cell RNA-seq reveals novel regulators of human embryonic stem cell differentiation to definitive endoderm. Genome Biol 2016;17:173.   
Daskalaki I, Gkikas I, Tavernarakis N. Hypoxia and selective autophagy in cancer development and therapy. Front Cell Dev Biol 2018; 6:104.   
Firulli BA, George RM, Harkin J et al. Hand1 loss-of-function within the embryonic myocardium reveals survivable congenital cardiac defects and adult heart failure. Cardiovasc Res 2020;116:605–18.   
Gao H, Ji S. Graph U-nets. IEEE Transactions on Pattern Analysis and Machine Intelligence 2022;44(09):4948–60.   
Greener JG, Kandathil SM, Moffat L et al. A guide to machine learning for biologists. Nat Rev Mol Cell Biol 2022;23:40–55.   
Hamilton W, Ying Z, Leskovec J. Inductive representation learning on large graphs. Adv Neural Inform Process Syst 2017;30:1024–34.   
Huynh-Thu VA, Irrthum A, Wehenkel L et al. Inferring regulatory networks from expression data using tree-based methods. PLoS One 2010;5:e12776.   
Iacono G, Massoni-Badosa R, Heyn H. Single-cell transcriptomics unveils gene regulatory network plasticity. Genome Biol 2019; 20:110–20.   
Jin S, Kim J, Willert T et al. Ebf factors and MyoD cooperate to regulate muscle relaxation via Atp2a1. Nat Commun 2014;5:3793.   
Kanamori M, Konno H, Osato N et al. A genome-wide and nonredundant mouse transcription factor database. Biochem Biophys Res Commun 2004;322:787–93.   
Kc K, Li R, Cui F et al. GNE: a deep learning framework for gene network inference by aggregating biological information. BMC Syst Biol 2019;13:1–14.   
Kim S. ppcor: an R package for a fast calculation to semi-partial correlation coefficients. Commun Stat Appl Methods 2015;22:665–74.   
Liang R, Liu Y. Tcf7l1 directly regulates cardiomyocyte differentiation in embryonic stem cells. Stem Cell Res Ther 2018;9:267.   
Luo W, Chen J, Li L et al. c-Myc inhibits myoblast differentiation and promotes myoblast proliferation and muscle fibre hypertrophy by regulating the expression of its target genes, miRNAs and lincRNAs. Cell Death Differ 2019;26:426–42.   
Matsumoto H, Kiryu H, Furusawa C et al. SCODE: an efficient regulatory network inference algorithm from single-cell RNA-seq during differentiation. Bioinformatics 2017;33:2314–21.   
Moore JE, Purcaro MJ, Pratt HE et al.; ENCODE Project Consortium. Expanded encyclopaedias of DNA elements in the human and mouse genomes. Nature 2020;583:699–710.   
Muzio G, O’Bray L, Borgwardt K. Biological network analysis with deep learning. Brief Bioinform 2021;22:1515–30.

Neph S, Stergachis AB, Reynolds A et al. Circuitry and dynamics of human transcription factor regulatory networks. Cell 2012; 150:1274–86.   
Oki S, Ohta T, Shioi G et al. ChIP-Atlas: a data-mining suite powered by full integration of public ChIP-seq data. EMBO Rep 2018; 19:e46255.   
Papili Gao N, Ud-Dean SM, Gandrillon O et al. SINCERITIES: inferring gene regulatory networks from time-stamped single cell transcriptional expression profiles. Bioinformatics 2018; 34:258–66.   
Pratapa A, Jalihal AP, Law JN et al. Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nat Methods 2020;17:147–54.   
Razaghi-Moghadam Z, Nikoloski Z. Supervised learning of gene regulatory networks. Curr Protoc Plant Biol 2020;5:e20106.   
Riley P, Anson-Cartwright L, Cross JC. The Hand1 bHLH transcription factor is essential for placentation and cardiac morphogenesis. Nat Genet 1998;18:271–5.   
Sec¸ilmis¸ D, Hillerton T, Sonnhammer EL. GRNbenchmark – a web server for benchmarking directed gene regulatory network inference methods. Nucleic Acids Res 2022;50:W398–404.   
Shu H, Zhou J, Lian Q et al. Modeling gene regulatory networks using neural network architectures. Nat Comput Sci 2021; 1:491–501.   
Sierra RA, Hoverter NP, Ramirez RN et al. TCF7L1 suppresses primitive streak gene expression to support human embryonic stem cell pluripotency. Development 2018;145:dev161075.   
Stergachis AB, Neph S, Sandstrom R et al. Conservation of trans-acting circuitry during mammalian regulatory evolution. Nature 2014; 515:365–70.   
Treutlein B, Lee QY, Camp JG et al. Dissecting direct reprogramming from fibroblast to neuron using single-cell RNA-seq. Nature 2016; 534:391–5.   
Wang J, Chen Y, Zou Q. Inferring gene regulatory network from single-cell transcriptomes with graph autoencoder model. PLoS Genet 2023;19:e1010942.   
Xu H, Baroukh C, Dannenfelser R et al. ESCAPE: database for integrating high-content published data collected from human and mouse embryonic stem cells. Database (Oxford) 2013;2013:bat045.   
Xu J, Zhang A, Liu F et al. STGRNS: an interpretable transformerbased method for inferring gene regulatory networks from singlecell transcriptomic data. Bioinformatics 2023;39:btad165.   
Yuan D. Deep learning network in relation inference from financial report. In the 2021 3rd International Conference on Big Data Engineering, Shanghai, China. 2021. 82–9.   
Yuan Y, Bar-Joseph Z. Deep learning for inferring gene relationships from single-cell expression data. Proc Natl Acad Sci USA 2019; 116:27151–8.   
Zhang H-M, Liu T, Liu C-J et al. AnimalTFDB 2.0: a resource for expression, prediction and functional study of animal transcription factors. Nucleic Acids Res 2015;43:D76–81.   
Zhao M, He W, Tang J et al. A comprehensive overview and critical evaluation of gene regulatory network inference technologies. Brief Bioinform 2021;22:bbab009.   
Zhao M, He W, Tang J et al. A hybrid deep learning framework for gene regulatory network inference from single-cell transcriptomic data. Brief Bioinform 2022;23:bbab568.   
Zheng R, Li M, Chen X et al. BiXGBoost: a scalable, flexible boostingbased method for reconstructing gene regulatory networks. Bioinformatics 2019;35:1893–900.