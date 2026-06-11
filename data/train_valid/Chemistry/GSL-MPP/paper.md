# Molecular Property Prediction Based on Graph Structure Learning

Bangyi Zhao

Fudan University

Shanghai

byzhao21@m.fudan.edu.cn

Jihong Guan

Tongji University

Shanghai

jhguan@tongji.edu.cn

Weixia Xu

Fudan University

Shanghai

xuweixia@fudan.edu.cn

Shuigeng Zhou∗

Fudan University

Shanghai

sgzhou@fudan.edu.cn

# Abstract

Molecular property prediction (MPP) is a fundamental but challenging task in the computer-aided drug discovery process. More and more recent works employ different graph-based models for MPP, which have made considerable progress in improving prediction performance. However, current models often ignore relationships between molecules, which could be also helpful for MPP. For this sake, in this paper we propose a graph structure learning (GSL) based MPP approach, called GSL-MPP. Specifically, we first apply graph neural network (GNN) over molecular graphs to extract molecular representations. Then, with molecular fingerprints, we construct a molecular similarity graph (MSG). Following that, we conduct graph structure learning on the MSG (i.e., molecule-level graph structure learning) to get the final molecular embeddings, which are the results of fusing both GNN encoded molecular representations and the relationships among molecules, i.e., combining both intra-molecule and inter-molecule information. Finally, we use these molecular embeddings to perform MPP. Extensive experiments on seven various benchmark datasets show that our method could achieve state-of-the-art performance in most cases, especially on classification tasks. Further visualization studies also demonstrate the good molecular representations of our method.

# 1 Introduction

The accurate prediction of molecular properties is a critical task in the field of drug discovery. By utilizing computational methods, this task can be accomplished with great efficiency, reducing both time and expense associated with identifying drug candidates. This is particularly important considering that the average cost of developing a new drug is currently estimated to be approximately \$2.8 billion [4, 30] and the development period lasts a dozen of years, let alone the high risk of clinical failure [20]. Naturally, a molecule can be abstracted as a topological graph, where atoms are treated as nodes and bonds are viewed as edges. In the past few years, deep graph learning methods, especially various graph neural networks (GNNs) have been applied in this field, offering effective molecular graph representations for accurate molecular property prediction [3, 22, 26]. In GNNs, nodes iteratively update their representations after aggregating information from their neighbours and a final graph-pooling layer will generate a graph representation for the molecule. Up to now, various message passing layers have been proposed and applied, including GAT [27], MPNN [5] and GIN [33]. And later studies further considered to integrate edge features into the passing messages in order to improve the expressive power of their models, like DMPNN [34] and CMPNN [22].

Despite the considerable progress, most of recent studies focus only on the message passing within individual molecules. The relationships among molecules are often ignored, which could also play an important role in property prediction [29]. One relatively easy and effective way is to construct a relationship graph among molecules using the structural similarity, because a critical assumption of medicinal chemistry is that structurally similar molecules tend to have similar biological activities [8]. For example, fingerprint (carrying the structural information of the molecules) similarity search is often used in virtual screening [15]. However, this assumption is not always true since a phenomenon called activity cliff (AC) exits. An AC is defined as a pair of structurally similar compounds with a large potency difference against a given target [12, 23, 25, 24]. Thus, the relationship graph constructed by structural similarity may be not “perfect” for the downstream tasks. We need to take certain measures to enhance this relationship graph if we want to make full and proper use of it.

To address these problems above, we propose a novel two-level graph representation learning method for molecular property prediction, called GSL-MPP. Our method operates in a two-level molecular graph representation framework: (i) Atom-level molecular graph representation where molecular graphs composed of atoms and bonds represent the intra-structures of molecules; and (ii) molecule-level graph representation where inter-molecule similarity graph (MSG in short) is constructed by fingerprint similarity to encode similarities between molecules that allows effective label propagation among connected similar molecules. Intra-molecular representation is done by GNNs, and inter-molecular representation is finished by graph structure learning (GSL). This twolevel graph representation enables us to comprehensively exploit both intra-molecule and intermolecule information to get better molecular representations and overcome (to some degree) the AC problem, consequently boosting MPP performance.

Specifically, we applies metric-based iterative graph structure learning in our method. The MSG structure and molecular embeddings are updated for T times. During each iteration, GSL-MPP learns a better MSG structure based on better molecular embeddings, and in turn, learns better molecular embeddings with a better MSG structure. Besides, during the training process, we also add a GSL-specific loss to the common supervised loss for better MSG structure learning on both classification tasks and regression tasks. Our method is evaluated on 7 benchmark datasets including 4 classification tasks and 3 regression tasks. Experimental results show that our model can achieve state-of-the-art performance in most cases. Ablation studies show that the combination of fingerprint similarity and GSL is of particular effectiveness.

# 2 Related Work

# 2.1 Molecular Property Prediction

Most methods for predicting molecular properties can be summarized using a general framework. In this framework, we first transform the input molecule m into a specific-length vector h using a representation function, $h = g ( m )$ ). Then another prediction function is used to predict a specific property y based on $h , y = f ( { \dot { h } } )$ . During this period, a good molecular representation is of vital importance to address molecular property prediction problems.

At early stages, traditional chemical fingerprints such as Extended Connectivity Fingerprints (ECFP) [14, 6] are used to encode a molecule to a vector. These fingerprints could carry the structural information of the molecules [16].

In order to improve the expressive power, recent works started to use the graph neural networks (GNNs) to acquire graph-level representation as molecular embedding. Examples include graph convolutional network (GCN) [3], graph attention network (GAT) [27], message passing neural network (MPNN) [5] and graph isomorphism network (GIN) [33].Later works extend the MPNN framework to consider bond information during message passing procedure, like DMPNN [34] and CMPNN [22]. Besides, CD-MVGNN [11] also considers both atom-level and bond-level message passing, and a cross-dependency mechanism is designed to ensure these two views rely on information from each other during feature updates, thereby enhancing expressive capabilities.

Recently, many efforts have also been made to integrate transformer to graph neural network. Molecule Attention Transformer(MAT) [13] attempts to incorporate node distance and graph structural information when calculating attention scores. Another work Grover [19] combines message-passing networks with the Transformer architecture to create a more expressive molecular encoder that captures information at two hierarchical levels. CoMPT [1] is also built upon the Transformer architecture. Unlike previous graph Transformer models that treated molecules as fully connected graphs, this approach employs a message diffusion mechanism inspired by heat diffusion phenomena to integrate information from the adjacency matrix, alleviating the over-smoothing issue.

However, these methods only focus on the structure of a single molecule, while ignoring the important role of inter-molecular relationships for property prediction.

# 2.2 Graph Structure Learning

The expressive power of GNNs often depends on the input graph structure. However, the initial graph structure is not always optimal for downstream tasks. On the one hand, the original graph is constructed from the original feature space, which may not reflect the "true" graph topology after feature extraction and transformation. On the other hand, errors can also occur when data is measured or collected, making the graph noisy or even incomplete. Graph structure learning (GSL) is one of the methods that can effectively solve this problem, through learning and optimizing the graph structure [35]. Recently, [2] proposed the method of iterative deep graph learning (IDGL) for jointly and iteratively learning graph structure and node embeddings in the field of natural language processing (NLP). It was later used by [28] for few-shot molecular property prediction. Compared to [28], our method is not based on few-shot situation and the datasets and baselines we choose are not for few-shot either. Besides, The specific implementation of GSL is different. More importantly, we try to construct an initial graph between molecules before we apply GSL, which is confirmed to be necessary in ablation study.

# 3 Method

# 3.1 Overview

The structure of our method GSL-MPP is illustrated in Fig. 1, which is operated on a two-level graph learning framework. Specifically, the two-level graph learning framework consists of (i) the lower level: atom-level molecular graphs encoded by GNN to extract the initial molecular representations, and (ii) the upper level: a molecule-level similarity graph, on which graph structure learning (GSL) is performed to iteratively learn the final molecular embeddings, where inter-molecular relationships are exploited.

The workflow of GSL-MPP is as follows: (1) molecule graphs are first encoded by a GNN to obtain initial molecular embeddings. Meanwhile, molecules are represented as feature vectors using fingerprints. (2) With the molecular feature vectors, the initial molecular similarity graph (MSG) is constructed, where each node is a molecule initially represented by the above GNN embeddings, and each edge attached with a weight — the similarity between the two corresponding molecules. (3) GSL is performed on the MSG, which iteratively updates the molecular embeddings and the graph structure. (4) The final molecular embeddings are used for property prediction.

# 3.2 Molecular Graph Embedding

Here, we describe how to represent a molecular graph as an initial vector by GNN. A molecule m can be abstracted as an attributed graph where $\bar { G } _ { m } \bar { = } ( \nu , \mathcal { E } )$ , in which $| \nu | = n _ { v }$ refers to a set of $n _ { v }$ atoms (nodes) and $| { \mathcal { E } } | = n _ { e }$ refers to a set of $n _ { \epsilon }$ e bonds (edges) in the molecule. $x _ { v }$ are used to represent the initial feature of node v and $\mathcal { N } _ { v }$ denotes the set of neighbors of node v.

# 3.2.1 Node embedding

We use Graph Isomorphism Network (GIN) [33] as intra-molecule GNN to extract each node’s embedding:

$$
h _ {v} ^ {(k)} = M L P ^ {(k)} \left(\left(1 + \epsilon^ {(k)}\right) \cdot h _ {v} ^ {(k - 1)} + \sum_ {u \in \mathcal {N} (v)} h _ {u} ^ {(k - 1)}\right), \tag {1}
$$

![](images/4d44616f75576b59ea2ee8aa6266b55da346e7e50250d6ad24798299135b63b5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Initial molecular vectors"] --> B["GNN"]
    C["FP Similarity calculation"] --> D["Initial molecular similarity matrix"]
    E["Initial molecular similarity graph"] --> F["Initial molecular similarity matrix"]
    G["Graph structure learning"] --> H["t-th Refined matrix Ã^(t)"]
    H --> I["Graph update"]
    I --> J["t-th iteration"]
    J --> K["Molecular embeddings H_r^(t)"]
    K --> L["Prediction task"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style E fill:#ccf,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#fcc,stroke:#333
    style J fill:#cff,stroke:#333
    style K fill:#ffc,stroke:#333
    style L fill:#fcc,stroke:#333
```
</details>

Figure 1: The workflow of GSL-MPP. In the initial molecular similarity graph (MSG), each node is a molecule initially represented by the GNN and each edge is attached with the FP similarity between the two corresponding molecules. GSL is then performed on the MSG.

where MLP means multi-layer perceptron, $h _ { v } ^ { ( k ) }$ is the representation vector of node v at the k-th layer. We initialize $h _ { v } ^ { ( 0 ) } = x _ { v }$ , and ϵ is a learnable parameter.

# 3.2.2 Graph pooling

After gaining each node’s embedding, a READOUT operation is applied to get the initial molecular embedding $h _ { g }$ :

$$
h _ {g} = \text { READOUT } (\{h _ {v} ^ {k} | v \in G \} | k = 0, 1,..., K). \tag {2}
$$

# 3.3 Constructing Molecular Similarity Graph (MSG)

Our inter-molecule graph reflects the relationships between molecules, where each node indicates a molecule, and each edge means the relationship between two molecules. As shown in Fig. 1, the initial feature vector of each node is the molecule’s embedding obtained by GNN (their embedding matrix is denoted as $X _ { r } ) .$ , and the initial adjacency matrix $A ^ { ( 0 ) }$ is calculated by the structural similarity between molecules. Here, we calculate the structural similarity between molecules based on their Extended Connectivity Fingerprints (ECFP) [14].

ECFPs are circular fingerprints, possessing several beneficial characteristics: 1) They can be calculated fast; 2) They are not predefined and can capture an almost limitless range of molecular characteristics including stereochemical information; 3) They indicate the presence of specific substructures, facilitating interpretation of computation results [18]. Specifically, we get each molecule’s ECFP and calculate the Tanimoto Coefficient as the similarity score. A hyperparameter $\epsilon _ { t c }$ acts as a threshold to obtain a sparse matrix. That is, we mask off those elements in the adjacency matrix that are smaller than $\epsilon _ { t c } .$ We apply molecular fingerprints to construct $A ^ { ( 0 ) }$ because it contains useful structural information [16] and could offer an informative initial inter-molecule graph.

# 3.4 Structure Learning on Molecular Similarity Graph

As we have discussed, this similarity graph constructed above may not be good enough for downstream tasks, therefore here graph structure learning is employed to enhance the graph by exploiting intermolecule relationships. Specifically, initial matrix built with fingerprint similarity only measure structural similarity between molecules and may not “perfectly” reflect true molecular property similarity, so we use GSL to refine it. The core of GSL is the structure learner that could be grouped into three types: (1) Metric-based approaches use a metric function like cosine similarity on pairwise node embeddings to calculate edge weights; (2) Neural approaches employ neural networks to infer edge weights; and (3) Direct approaches treat all elements of the adjacency matrix as learnable parameters [35].

In this paper, following IDGL [2], we adopt the metric-based approach and employ m-perspective weighted cosine similarity as the metric function:

$$
s _ {i j} ^ {p} = \cos (w _ {p} \odot v _ {i}, w _ {p} \odot v _ {j}), \quad s _ {i j} = \frac {1}{m} \sum_ {p = 1} ^ {m} s _ {i j} ^ {p}, \tag {3}
$$

where $s _ { i j } ^ { p }$ estimates the cosine similarity between nodes $v _ { i }$ and $v _ { j } .$ , each perspective $p$ considers one part of the semantics contained in the vectors and corresponds to a learnable weight vector $w _ { p }$ . The obtained $s _ { i j }$ is the entry in row i and column $j$ of the newly learned adjacency matrix A. Also the ϵ-neighborhood sparsification technique is applied to obtaining a sparse and non-negative adjacency matrix.

The node embeddings $H _ { r }$ and the adjacency matrix A will be alternately refined for $T$ times. At the t-th iteration, $A ^ { ( t ) }$ is calculated from the previously updated node embeddings $H _ { r } ^ { ( t - 1 ) }$ by Eq. (3). Then we use the learned graph structure $\hat { A ^ { ( t ) } }$ as supplementary to optimize the initial graph $A ^ { ( 0 ) }$ :

$$
\widetilde {A} ^ {(t)} = \lambda A ^ {(0)} + (1 - \lambda) \left\{\eta A ^ {(t)} + (1 - \eta) A ^ {(1)} \right\}, \tag {4}
$$

where $A ^ { ( 1 ) }$ is the adjacency matrix learned from $X _ { r }$ at the 1-st iteration in order to maintain the initial node information. λ and η are hyperparameters.

After learning the adjacency matrix, we employ an L-layer inter-molecule GNN to learn node embeddings, and in the l-th layer, $H _ { r } ^ { ( t , l ) }$ is updated by

$$
H _ {r} ^ {(t, l)} = \operatorname{ReLU} \left(\widetilde {A} ^ {(t)} H _ {r} ^ {(t, l - 1)} W _ {r} ^ {(l)}\right), \tag {5}
$$

$H _ { r } ^ { ( t ) } = H _ { r } ^ { ( t , L ) }$ is the final node embeddings in this iteration and $H _ { r } ^ { ( t , 0 ) } = X _ { r }$

# 3.5 Loss Function

After $T$ rounds of iteration, the node (molecule) embeddings $H _ { r } ^ { ( T ) }$ represent the final molecular representations. Based on this, predictions can be made for specific property $\hat { y }$ with a fully connected layer (FC) as follows:

$$
\hat {y} = F C (H _ {r} ^ {(T)}). \tag {6}
$$

The whole loss function used in our method consists of two parts: the label prediction loss and the GSL-specific loss. The label prediction loss function $\mathcal { L } _ { p r e d }$ is obtained in a manner similar to existing methods:

$$
\mathcal {L} _ {p r e d} = \ell (\hat {y}, y), \tag {7}
$$

where $\hat { y }$ represents the predicted value, y is the ground truth, and ℓ represents the loss function used. In classification tasks, it is the Cross Entropy Loss, and in regression tasks, it is the Mean Squared Error Loss.

Since the quality of the learned inter-molecule graph structure is of great importance for our method, we further design a GSL-specific loss, hoping that the learned adjacency matrix does not contain wrong edges. We use $S _ { t r a i n }$ to represent molecules in training set and $\widetilde { A } ^ { ( T ) }$ to represent the final adjacency matrix after being refined $T$ times. In classification tasks, there exists a ground truth for the matrix, $A ^ { * } ( A _ { i j } ^ { * } = 1 \operatorname { i f } y _ { i } = y _ { j }$ else 0), i.e., molecules with the same label should be connected by edges. Thus, we define the GSL-specific loss as

$$
\mathcal {L} _ {G S L} = \sum_ {x _ {i}, x _ {j} \in S _ {\text { train }}} (\widetilde {A} _ {i j} ^ {(T)} - A _ {i j} ^ {*}) ^ {2} \tag {8}
$$

However, in regression tasks, the prediction of a molecule is a real value and no native ground truth exists. We have to define it by ourselves. For the convenience of calculation, we only consider those molecular pairs with large difference (beyond a certain threshold $\epsilon _ { y } )$ in predicted values when calculating the GSL-specific loss:

$$
\mathcal {L} _ {G S L} = \sum_ {x _ {i}, x _ {j} \in S _ {\text { train }}} (\widetilde {A} _ {i j} ^ {(T)}) ^ {2}, \quad x _ {i}, x _ {j} \quad \text { satisfy } \quad | y _ {i} - y _ {j} | > \epsilon_ {y}. \tag {9}
$$

The whole loss function combines both the task prediction loss and the GSL-specific loss, that is, $\mathcal { L } = \mathcal { L } _ { p r e d } + \mathcal { L } _ { G S L }$ .

# 3.6 Algorithm

The algorithm of our method is presented in Algorithm 1. After obtaining the initial molecular embeddings and constructing the initial inter-molecule similarity graph MSG (corresponding to the adjacency matrix), T iterations of GSL are applied. During each iteration, the adjacency matrix is refined based on the node embeddings gained in the last iteration, while the node embeddings are updated based on adjacency matrix obtained in the last iteartion.

Algorithm 1 The GSL-MPP algorithm   
1: Obtain initial molecular embedding $h_{g,i}$ for each molecule $m_{i}$ by a graph-based molecular encoder (an intra-molecule GNN);
2: $X_{r} \leftarrow$ embedding matrix of all $h_{g,i}$ ;
3: $H_{r}^{(0)} \leftarrow X_{r}$ ;
4: Construct an initial molecule similarity matrix $A^{(0)}$ using molecular fingerprint similarity;
5: for t = 1 to T do
6: Use GSL to learn a refined adjacency matrix $A^{(t)}$ by $H_{r}^{(t-1)}$ using Eq. (3);
7: Combine initial and refined adjacency matrices $A^{(0)}$ and $A^{(t)}$ , $A^{(1)}$ to obtain $\widetilde{A}^{(t)}$ by Eq. (4);
8: Initialize node embeddings by $H_{r}^{(t,0)} = X_{r}$ ;
9: for l = 1 to L do
10: Update node embedding $H_{r}^{(t,l)}$ by inter-molecule GNN using Eq. (5);
11: end for
12: $H_{r}^{(t)} \leftarrow H_{r}^{(t,L)}$ ;
13: end for
14: Obtain prediction $\hat{y}$ using $H_{r}^{(T)}$ by Eq. (6);
15: if in training phase then
16: Calculate $L_{pred}$ by Eq. (7) and $L_{GSL}$ by Eq. (8) or Eq. (9) for $S_{train}$ ;
17: $L \leftarrow L_{pred} + L_{GSL}$ ;
18: Back-propagate L to update model weights;
19: end if

# 4 Performance Evaluation

# 4.1 Experimental Setting

Datasets. We use 7 benchmark datasets from MoleculeNet [31] for experiments, among which 4 are classification tasks and 3 are regression tasks. Specifically, BACE is about the binding results of several inhibitors; BBBP is the blood–brain barrier penetration dataset; SIDER and Clintox are two multi-task datasets corresponding to side effects and toxicity respectively; ESOL, Lipophilicity and Freesolv are regression datasets about physical chemistry properties.

Scaffold splitting of [34] is adopted to split the datasets into training, validation, and test, with a 0.8/0.1/0.1 ratio, which is more empirical and challenging than random splitting. Following previous works [11, 19], we use three independent runs on three random-seeded scaffold splitting for each dataset.

Baselines. We compare our method against 12 baselines. TF\_Robust [17] is a DNN-based multitask framework that takes molecular fingerprints as input. GCN (GraphConv) [3], Weave [9] and SchNet [21] are three graph convolutional models. MPNN [5] and its variants MGCN [10], DMPNN [34] and CMPNN [22] are models considering the edge features during message passing. AttentiveFP [32] is an extension of the graph attention network. GROVER [19] and CoMPT [1] are two transformer-based models. Here, GROVER is compared without the pretrain process for a fair comparison. CoMPT is a transformer-based model utilizing both nodes and edges information in message passing process while CD-MVGNN [11] also constructs two views for atoms and bonds respectively.

Table 1: Hyper-parameter settings. 

<table><tr><td>Hyper-parameter</td><td>Description</td><td>Value range</td></tr><tr><td>max_lr</td><td>maximum learning rate of polynomial decay scheduler</td><td>0.0001~0.01</td></tr><tr><td>weight_decay</td><td>weight_decay weight decay percentage for Adam optimizer</td><td>0.00001~0.001</td></tr><tr><td>gin_layers</td><td>number of the intra-molecule GIN layers</td><td>2~5</td></tr><tr><td>gin_hidden_size</td><td>number of the hidden dimensionality in the intra-molecule GIN</td><td>32, 64, 128, 256</td></tr><tr><td>tc_epsilon</td><td>threshold of Tanimoto Coefficient for  $A^{(0)}$  ( $\epsilon_{tc}$ )</td><td>0.0, 0.1, 0.2, 0.3, 0.5, 0.7</td></tr><tr><td>gsl_iter</td><td>number of the iterations for graph structure learning (T)</td><td>1~5</td></tr><tr><td>gsl_gnn_layers</td><td>number of the inter-molecule GCN layers</td><td>2, 3</td></tr><tr><td>gsl_hidden_size</td><td>number of the hidden dimensionality in the inter-molecule GCN</td><td>32, 64, 128, 256</td></tr><tr><td>gsl_epsilon</td><td>threshold of similarity score in GSL ( $\epsilon_{gsl}$ )</td><td>0, 0.1, 0.2, 0.5</td></tr><tr><td>gsl_perspective</td><td>number of perspective used in GSL (m)</td><td>1, 2, 4, 8, 16</td></tr><tr><td>gsl_skip_conn</td><td>the ratio of initial matrix while updating the graph structure (λ)</td><td>0.1, 0.3, 0.5, 0.7, 0.9</td></tr><tr><td>gsl_update_ratio</td><td>the ratio of t-th learned matrix while updating the graph structure (η)</td><td>0.1, 0.3, 0.6, 0.8, 1.0</td></tr><tr><td>dropout</td><td>dropout rate</td><td>0, 0.1, 0.2, 0.4, 0.6</td></tr><tr><td>gsl_coff</td><td>the coefficient of the GSL related loss (μ)</td><td>0.1, 0.3, 0.5, 0.7, 0.9</td></tr></table>

Table 2: Performance comparison between our model and baselines. Mean and standard deviation of AUC or RMSE values are reported. 

<table><tr><td></td><td colspan="4">Classification (ROC-AUC)</td><td colspan="3">Regression (RMSE)</td></tr><tr><td>Dataset</td><td>BACE</td><td>BBBP</td><td>ClinTox</td><td>SIDER</td><td>FreeSolv</td><td>ESOL</td><td>Lipop</td></tr><tr><td>TFRobust</td><td>0.824±0.022</td><td>0.860±0.087</td><td>0.765±0.085</td><td>0.607±0.033</td><td>4.122±0.085</td><td>1.722±0.038</td><td>0.909±0.060</td></tr><tr><td>GraphConv</td><td>0.854±0.011</td><td>0.877±0.036</td><td>0.845±0.051</td><td>0.593±0.035</td><td>2.900±0.135</td><td>1.068±0.050</td><td>0.712±0.049</td></tr><tr><td>Weave</td><td>0.791±0.008</td><td>0.837±0.065</td><td>0.823±0.023</td><td>0.543±0.034</td><td>2.398±0.250</td><td>1.158±0.055</td><td>0.813±0.042</td></tr><tr><td>SchNet</td><td>0.750±0.033</td><td>0.847±0.024</td><td>0.717±0.042</td><td>0.545±0.038</td><td>3.215±0.755</td><td>1.045±0.064</td><td>0.909±0.098</td></tr><tr><td>MPNN</td><td>0.815±0.044</td><td>0.913±0.041</td><td>0.879±0.054</td><td>0.595±0.030</td><td>2.185±0.952</td><td>1.167±0.430</td><td>0.672±0.051</td></tr><tr><td>DMPNN</td><td>0.852±0.053</td><td>0.919±0.030</td><td>0.897±0.040</td><td>0.632±0.023</td><td>2.177±0.914</td><td>0.980±0.258</td><td>0.653±0.046</td></tr><tr><td>MGCN</td><td>0.734±0.030</td><td>0.850±0.064</td><td>0.634±0.042</td><td>0.552±0.018</td><td>3.349±0.097</td><td>1.266±0.147</td><td>1.113±0.041</td></tr><tr><td>CMPNN</td><td>0.869±0.023</td><td>0.929±0.025</td><td>0.922±0.017</td><td>0.617±0.016</td><td>2.060±0.505</td><td>0.838±0.140</td><td> $\underline{0.625±0.027}$ </td></tr><tr><td>AttentiveFP</td><td>0.863±0.015</td><td>0.908±0.050</td><td>0.933±0.020</td><td>0.605±0.060</td><td>2.030±0.420</td><td>0.853±0.060</td><td>0.650±0.030</td></tr><tr><td>CD-MVGNN</td><td> $\underline{0.892±0.011}$ </td><td> $\underline{0.933±0.006}$ </td><td> $\underline{0.945±0.037}$ </td><td> $\underline{0.639±0.012}$ </td><td> $\underline{1.552±0.123}$ </td><td> $\underline{0.779±0.026}$ </td><td> $\underline{0.553±0.013}$ </td></tr><tr><td>GROVER*</td><td>0.858</td><td>0.911</td><td>0.884</td><td>0.624</td><td>1.987</td><td>0.911</td><td>0.643</td></tr><tr><td>CoMPT</td><td>0.838±0.035</td><td>0.926±0.028</td><td>0.876±0.031</td><td>0.612±0.026</td><td>2.006±0.628</td><td>0.822±0.090</td><td>0.663±0.035</td></tr><tr><td>OurModel</td><td> $\underline{0.871±0.038}$ </td><td> $\underline{0.957±0.008}$ </td><td> $\underline{0.947±0.020}$ </td><td> $\underline{0.652±0.014}$ </td><td> $\underline{1.974±0.315}$ </td><td> $\underline{0.799±0.118}$ </td><td>0.693±0.063</td></tr></table>

\*Here, GROVER does not use pretrained model for a fair comparison, and standard deviation is not provided in its the original paper.

Evaluation metrics. Following the evaluation criteria adopted by these baseline models, we use AUC-ROC to evaluate the performance of classification tasks, and RMSE to evaluate regression tasks.

Implementation details. Our model apply a polynomial decay scheduler to the learning rate with two linear increase warm-up epochs and polynomial decay afterward. The power of polynomial decay is set to 1, indicating a linear decay. The final learning rate is 1e-9 and the max\_epoch is 300. For the proposed model, on each dataset we try different hyper-parameter combinations, and take the hyper-parameter set with the best result. While building the initial inter-molecule graph, the radius of used ECFP is 2. The threshold of GSL-specific loss for regression tasks $( \epsilon _ { y } )$ is 0.01. More details of the hyper-parameter setting in the implementation of our model are presented in Table 1.

Table 3: Ablation study on four variants of our model. 

<table><tr><td></td><td colspan="4">Classification (ROC-AUC)</td><td colspan="3">Regression (RMSE)</td></tr><tr><td></td><td>BACE</td><td>BBBP</td><td>ClinTox</td><td>SIDER</td><td>FreeSolv</td><td>ESOL</td><td>Lipop</td></tr><tr><td>Not any</td><td>0.809±0.032</td><td>0.943±0.005</td><td>0.932±0.012</td><td>0.593±0.017</td><td>2.055±0.405</td><td>0.962±0.093</td><td>0.888±0.046</td></tr><tr><td>only A0</td><td>0.856±0.049</td><td>0.951±0.002</td><td>0.939±0.033</td><td>0.639±0.030</td><td>3.433±0.552</td><td>0.945±0.076</td><td>0.724±0.040</td></tr><tr><td>only GSL</td><td>0.850±0.025</td><td>0.943±0.017</td><td>0.875±0.049</td><td>0.580±0.021</td><td>2.638±0.098</td><td>1.147±0.256</td><td>0.899±0.196</td></tr><tr><td>No GSL-Loss</td><td>0.865±0.034</td><td>0.953±0.015</td><td>0.935±0.016</td><td>0.651±0.026</td><td>2.134±0.155</td><td>0.821±0.100</td><td>0.711±0.047</td></tr><tr><td>Our Model</td><td>0.871±0.038</td><td>0.957±0.008</td><td>0.947±0.020</td><td>0.652±0.014</td><td>1.974±0.315</td><td>0.799±0.118</td><td>0.693±0.063</td></tr></table>

# 4.2 Performance Comparison

Table 2 presents the results of our model and the baselines on all datasets. boldfaced values are the best results, and underlined values are the 2nd best results. From 2 we have the following observations: (1) Compared to the SOTA model CD-MVGNN, Our model performs better on 3/4 classification datasets with a 2.4% AUC lift on BBBP. Since our model is based on a simple GIN without complicated message passing procedures used in CD-MVGNN and CoMPT, this result indicates the effectiveness of the inter-molecule graph for prediction tasks. (2) Our model performs relatively poor on regression tasks compared to SOTAs, which may be explained by the lack of real ground truth of relationship graphs in regression tasks. However, our model still achieve 2nd best results on 2/3 datasets. (3) Though our model uses GIN for intra-molecule graphs, it outperforms GIN on 7 of the 8 datasets. Especially, on the ClinTox dataset, our model gets up to 7.8% performance improvement. This also reflects the effectiveness of our molecule similarity graph construction and graph structure learning.

# 4.3 Ablation Study

To investigate the contribution of each component of our model, an ablation study is conducted. We consider four variant models for comparison as follows:

• Not any: directly use $H _ { r } ^ { ( 0 ) }$ to predict. It is almost a GIN network.   
• Only $A ^ { ( 0 ) }$ : apply GNN on the initial molecular similarity graph $A ^ { ( 0 ) }$ constructed by ECFP similarity without GSL.   
• Only GSL: use de novo GSL without an initial graph reference $A ^ { ( 0 ) }$ .   
• No GSL-Loss: use $A ^ { ( 0 ) }$ and GSL, but apply only the prediction loss.

Ablation results are given in Table 3. Here we mainly consider the contribution of the initial adjacency matrix $A ^ { ( 0 ) }$ constructed by ECFP fingerprints and GSL process in our method. The results of “Not any”, “Only $A ^ { ( 0 ) } { } ^ { , }$ and “No GSL-Loss” confirm that the use of $A ^ { ( 0 ) }$ could improve the performance of our model and the improvement will be much more significant when combined with GSL. Besides, it is interesting to notice that “Only GSL” often performs worse than “Not any", which probably means learning an inter-molecule graph from scratch might be difficult and it is necessary for us to utilize the chemical information of molecular fingerprints to build an initial graph. Finally, while comparing “No GSL-Loss" and the complete “Our Model", we can see that GSL-specific loss does make a difference for our method.

We also conduct experiments to show the results of using different values of some important hyperparameters on all the datasets. Table 4 reports the results of applying different values of λ, which is used to balance the learned graph structure and the initial graph structure. It can be seen that applying a large λ value (0.8 or 0.9) will generate a relatively good results on most datasets, which indicates the importance of the initial inter-molecule graph. Besides, Table 5 shows the impact of the number of iteration $T$ on performance. We can see that as T increases from 1 to 5, performance on most datasets does not show continuous improvement, which means that the best T is data dependent.

# 4.4 Visualization of Molecular Representations

To check the molecular representation learning ability of our model, we apply t-distributed Stochastic Neighbor Embedding (t-SNE) with default hyper-parameters to visualize the final molecular represen-

Table 4: Results for different λ values on different datasets. 

<table><tr><td>lambda</td><td>BACE</td><td>BBBP</td><td>ClinTox</td><td>SIDER</td><td>FreeSolv</td><td>ESOL</td><td>Lipop</td></tr><tr><td>0.1</td><td> $0.702 \pm 0.1$ </td><td> $0.924 \pm 0.006$ </td><td> $0.914+0.026$ </td><td> $0.564+0.035$ </td><td> $2.812+0.239$ </td><td> $0.983+0.094$ </td><td> $0.714+0.056$ </td></tr><tr><td>0.3</td><td> $0.846 \pm 0.062$ </td><td> $0.944 \pm 0.008$ </td><td> $\mathbf{0.947 \pm 0.020}$ </td><td> $0.606+0.018$ </td><td> $2.371+0.171$ </td><td> $0.986+0.112$ </td><td> $0.749+0.05$ </td></tr><tr><td>0.5</td><td> $0.839 \pm 0.043$ </td><td> $0.921 \pm 0.017$ </td><td> $0.939+0.018$ </td><td> $0.616+0.007$ </td><td> $2.224+0.231$ </td><td> $0.87+0.147$ </td><td> $0.706+0.059$ </td></tr><tr><td>0.7</td><td> $0.849 \pm 0.049$ </td><td> $0.942 \pm 0.004$ </td><td> $0.916+0.037$ </td><td> $0.634+0.006$ </td><td> $2.126+0.153$ </td><td> $0.931+0.056$ </td><td> $0.725+0.066$ </td></tr><tr><td>0.8</td><td> $0.850 \pm 0.028$ </td><td> $\mathbf{0.957 \pm 0.008}$ </td><td> $0.927+0.023$ </td><td> $\mathbf{0.652 \pm 0.014}$ </td><td> $\mathbf{1.974 \pm 0.315}$ </td><td> $0.827+0.09$ </td><td> $\mathbf{0.693 \pm 0.063}$ </td></tr><tr><td>0.9</td><td> $\mathbf{0.871 \pm 0.038}$ </td><td> $0.939 \pm 0.013$ </td><td> $0.925+0.023$ </td><td> $0.643+0.004$ </td><td> $2.279+0.089$ </td><td> $\mathbf{0.799 \pm 0.118}$ </td><td> $0.707+0.066$ </td></tr></table>

Table 5: Results for different T values on different datasets. 

<table><tr><td>T</td><td>BACE</td><td>BBBP</td><td>ClinTox</td><td>SIDER</td><td>FreeSolv</td><td>ESOL</td><td>Lipop</td></tr><tr><td>1</td><td>0.848±0.062</td><td>0.945±0.007</td><td>0.93±0.028</td><td>0.64±0.017</td><td>2.724±0.484</td><td>0.863±0.046</td><td>0.718±0.047</td></tr><tr><td>2</td><td>0.858±0.045</td><td>0.923±0.023</td><td>0.947±0.020</td><td>0.636±0.015</td><td>1.974±0.315</td><td>0.86±0.127</td><td>0.751±0.053</td></tr><tr><td>3</td><td>0.857±0.043</td><td>0.944±0.008</td><td>0.93±0.005</td><td>0.652±0.014</td><td>2.182±0.364</td><td>0.872±0.087</td><td>0.693±0.063</td></tr><tr><td>4</td><td>0.871±0.038</td><td>0.957±0.008</td><td>0.907±0.018</td><td>0.64±0.007</td><td>2.209±0.41</td><td>0.861±0.093</td><td>0.752±0.047</td></tr><tr><td>5</td><td>0.851±0.057</td><td>0.937±0.011</td><td>0.918±0.034</td><td>0.63±0.015</td><td>2.309±0.291</td><td>0.799±0.118</td><td>0.751±0.058</td></tr></table>

![](images/5a04f9ae586026ca99cf1bbcaffda432c2cf6d1497d8436094107d5e68b5de7a.jpg)

<details>
<summary>scatter</summary>

| x    | y    | group |
| ---- | ---- | ----- |
| -70  | 10   | Blue  |
| -60  | 5    | Blue  |
| -50  | 0    | Blue  |
| -40  | -5   | Blue  |
| -30  | -10  | Blue  |
| -20  | -15  | Blue  |
| -10  | -20  | Blue  |
| 0    | -25  | Blue  |
| 10   | -30  | Blue  |
| 20   | -35  | Blue  |
| 30   | -40  | Blue  |
| 40   | -45  | Blue  |
| 50   | -50  | Blue  |
| 60   | -55  | Blue  |
| -70  | 15   | Red   |
| -60  | 10   | Red   |
| -50  | 5    | Red   |
| -40  | 0    | Red   |
| -30  | -5   | Red   |
| -20  | -10  | Red   |
| -10  | -15  | Red   |
| 0    | -20  | Red   |
| 10   | -25  | Red   |
| 20   | -30  | Red   |
| 30   | -35  | Red   |
| 40   | -40  | Red   |
| 50   | -45  | Red   |
| 60   | -50  | Red   |
</details>

(a) BBBP

![](images/3bdb0848fd5e8811d8fa6352afa5eb2c4729ccd607dd711a5f9267a359748a83.jpg)

<details>
<summary>scatter</summary>

| x    | y    | group |
| ---- | ---- | ----- |
| -60  | 10   | blue  |
| -50  | 5    | blue  |
| -40  | 0    | blue  |
| -30  | -5   | blue  |
| -20  | -10  | blue  |
| -10  | -15  | blue  |
| 0    | -20  | blue  |
| 10   | -25  | blue  |
| 20   | -30  | blue  |
| 30   | -35  | blue  |
| 40   | -40  | blue  |
| 50   | -45  | blue  |
| 60   | -50  | blue  |
| -60  | 20   | red   |
| -50  | 15   | red   |
| -40  | 10   | red   |
| -30  | 5    | red   |
| -20  | 0    | red   |
| -10  | -5   | red   |
| 0    | -10  | red   |
| 10   | -15  | red   |
| 20   | -20  | red   |
| 30   | -25  | red   |
| 40   | -30  | red   |
| 50   | -35  | red   |
| 60   | -40  | red   |
</details>

(b) BACE

![](images/1087816fcb6835e16474e85caa4e75d6bb665db2778a5800f8dcddda2e367b71.jpg)

<details>
<summary>scatter</summary>

| x    | y    |
| ---- | ---- |
| -40  | 10   |
| -30  | 5    |
| -20  | 15   |
| -10  | 10   |
| 0    | 5    |
| 10   | 0    |
| 20   | -5   |
| 30   | -10  |
| 40   | -20  |
</details>

(c) FreeSolv

![](images/711d3faa03602e993176543bbfa633c3e58d5abf920a522f3f7c62a503fc0825.jpg)

<details>
<summary>scatter</summary>

| x    | y    |
| ---- | ---- |
| -15  | 5    |
| -10  | 8    |
| -5   | 3    |
| 0    | -2   |
| 5    | -4   |
| 10   | -6   |
| 15   | -8   |
</details>

(d) ESOL   
Figure 2: Visualization of molecular representations for 4 datsets: (a) BBBP, (b) BACE, (c) FreeSolv and (d) ESOL. For classification datasets BACE and BBBP, molecules of label 1 are colored in red and molecules of label 0 are colored in blue. For regression datasets FreeSolv and ESOL, the colors of the points change from red to blue as the property value increases.

Table 6: Performance comparison between original and anchor-based GSL. 

<table><tr><td></td><td>BACE</td><td>BBBP</td><td>ClinTox</td><td>SIDER</td><td>FreeSolv</td><td>ESOL</td><td>Lipop</td></tr><tr><td>Origin</td><td>0.865</td><td>0.953</td><td>0.935</td><td>0.651</td><td>2.134</td><td>0.821</td><td>0.711</td></tr><tr><td>Anchor-based</td><td>0.818</td><td>0.949</td><td>0.95</td><td>0.612</td><td>2.208</td><td>0.794</td><td>0.747</td></tr></table>

Table 7: Performance comparison between our model (using anchor-based GSL) and baselines. 

<table><tr><td></td><td>ROC-AUC%</td></tr><tr><td>Our model</td><td>81.8</td></tr><tr><td>PharmHGT</td><td>80.6</td></tr><tr><td>DMPNN</td><td>78.6</td></tr><tr><td>CD-MVGNN</td><td>78.4</td></tr><tr><td>CoMPT</td><td>78.1</td></tr><tr><td>CMPNN</td><td>77.4</td></tr><tr><td>AttentiveFP</td><td>75.7</td></tr><tr><td>MPNN</td><td>74.1</td></tr><tr><td>GROVER</td><td>62.5</td></tr></table>

tations on four datasets, including two classification datasets (BACE and BBBP) and two regression datasets (FreeSolv and ESOL). The results are shown in Fig. 2.

We can see that molecules of different labels have a clear boundary for both two classification datasets, especially for BBBP. Molecules of the same label tend to be clustered together, while molecules of different labels are located apart. Also, there seems a certain distribution pattern existing among the molecules of different property values for the two regression datasets. For the FreeSolv dataset, molecules tend to move from the outer region to the inner region as the property value decreases. As for the ESOL dataset, molecules tend to move from upper left to lower right as the property value decreases. These results indicate that our model generates reasonable representations of molecules for downstream tasks.

# 4.5 Scaling Our Model to Larger Datasets

During GSL, the similarity metric function calculate similarity scores for all pairs of graph nodes, which requires $\mathcal { O } ( n ^ { 2 } )$ complexity. So we need to address the scalability issue if the size of datasets becomes larger. Following IDGL [2], we apply an anchor-based method. During each iteration, We learn a node-anchor similarity matrix $R \in \overline { { \mathbb { R } } } ^ { n \times s }$ instead of the original complete adjacency matrix $A \in \mathbb { R } ^ { n \times n }$ . s represents the number of anchor nodes, which is a hyperparameter that can be set according to different datasets. By using $R$ instead of A, the time and space complexity can be reduced from $\scriptstyle { \mathcal { O } } ( n ^ { 2 } )$ to $\mathcal { O } ( n s )$ . Therefore, Eq. (3) in the paper can be rewritten as the following:

$$
s _ {i k} ^ {p} = \cos (w ^ {p} \odot v _ {i}, w ^ {p} \odot u _ {k}), \quad s _ {i k} = \frac {1}{m} \sum_ {p = 1} ^ {m} s _ {i k} ^ {p} \tag {10}
$$

where $s _ { i k }$ is the similarity score between node $v _ { i }$ and anchor $u _ { k }$ . The procedure of message passing should also be changed accordingly. The node-anchor similarity matrix R allows only direct connections between nodes and anchors. We call a direct travel between a node and an anchor as one-step transition described by R. Based on theories of stationary Markov random walks, we can actually recover A from R by calculating the two-step transition probabilities.

Using the above anchor-based GSL, We firstly evaluate whether introducing anchor nodes will have a great impact on the original prediction performance of our model. Results are given in Table 6. We can find that anchor-based GSL performs a little worse than the original GSL in these molecule datasets but the performance degradation is not significant. So we think it is appropriate for us to apply anchor-based GSL in larger-scale molecule datasets.

After completing the above evaluation, we test the anchor-based GSL method on the HIV dataset which includes over 40000 molecules and compare it with some existing models. Except for CD-MVGNN, the results of other models on the HIV data set are from PharmHGT [7]. PharmHGT is a recently proposed model based on the Transformer structure, which treats molecules as heterogeneous graphs. The ROC-AUC of CD-MVGNN on the HIV data set is obtained experimentally by ourselves. Results are given in Table 7. Our method is able to achieve the optimal ROC-AUC on the HIV dataset, showing that after introducing anchor nodes, our method can be well extended to larger-scale datasets and achieve satisfactory results.

# 5 Conclusion

In this paper, we propose a new model based on two-level molecular representation for molecular property prediction. Unlike previous attempts focusing exclusively on message passing between atoms or bonds within individual molecule graphs, we further take use of the inter-molecule graph. Concretely, we utilize the chemical information of molecular fingerprints to construct an initial molecular similarity graph, and employ graph structure learning to refine the graph. Molecular embeddings based on GSL on the inter-molecular similarity graph are used for MPP. Extensive experiments show that our model can achieve state-of-the-art performance in most cases, especially on the classification tasks. Ablation studies also validate the major designed components of the model.

However, there is still room to improve our model in the following directions: (1) Using more sophisticated graph-based models to encode molecular graphs rather than GIN. (2) Designing new metrics other than weighted cosine similarity for graph structure learning. (3) Exploring new and more effective GSL methods.

# References

[1] Jianwen Chen, Shuangjia Zheng, Ying Song, Jiahua Rao, and Yuedong Yang. Learning attributed graph representations with communicative message passing transformer. arXiv preprint arXiv:2107.08773, 2021.   
[2] Yu Chen, Lingfei Wu, and Mohammed Zaki. Iterative deep graph learning for graph neural networks: Better and robust node embeddings. Advances in neural information processing systems, 33:19314–19326, 2020.   
[3] David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. Advances in neural information processing systems, 28, 2015.   
[4] Nic Fleming. How artificial intelligence is changing drug discovery. Nature, 557(7706):S55– S55, 2018.   
[5] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pages 1263–1272. PMLR, 2017.   
[6] Robert C Glen, Andreas Bender, Catrin H Arnby, Lars Carlsson, Scott Boyer, and James Smith. Circular fingerprints: flexible molecular descriptors with applications from physical chemistry to adme. IDrugs, 9(3):199, 2006.   
[7] Yinghui Jiang, Shuting Jin, Xurui Jin, Xianglu Xiao, Wenfan Wu, Xiangrong Liu, Qiang Zhang, Xiangxiang Zeng, Guang Yang, and Zhangming Niu. Pharmacophoric-constrained heterogeneous graph transformer model for molecular property prediction. Communications Chemistry, 6(1):60, 2023.   
[8] Mark A Johnson and Gerald M Maggiora. Concepts and applications of molecular similarity. Wiley, 1990.   
[9] Steven Kearnes, Kevin McCloskey, Marc Berndl, Vijay Pande, and Patrick Riley. Molecular graph convolutions: moving beyond fingerprints. Journal of computer-aided molecular design, 30:595–608, 2016.   
[10] Chengqiang Lu, Qi Liu, Chao Wang, Zhenya Huang, Peize Lin, and Lixin He. Molecular property prediction: A multilevel quantum interactions modeling perspective. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 1052–1060, 2019.

[11] Hehuan Ma, Yatao Bian, Yu Rong, Wenbing Huang, Tingyang Xu, Weiyang Xie, Geyan Ye, and Junzhou Huang. Cross-dependent graph neural networks for molecular property prediction. Bioinformatics, 38(7):2003–2009, 2022.   
[12] Gerald M Maggiora. On outliers and activity cliffs why qsar often disappoints, 2006.   
[13] Łukasz Maziarka, Tomasz Danel, Sławomir Mucha, Krzysztof Rataj, Jacek Tabor, and Stanisław Jastrz˛ebski. Molecule attention transformer. arXiv preprint arXiv:2002.08264, 2020.   
[14] H. L. Morgan. The generation of a unique machine description for chemical structures-a technique developed at chemical abstracts service. Journal of Chemical Documentation, 5(2):107– 113, 1965.   
[15] Ingo Muegge and Prasenjit Mukherjee. An overview of molecular fingerprint similarity search in virtual screening. Expert opinion on drug discovery, 11(2):137–148, 2016.   
[16] Ngoc-Quang Nguyen, Gwanghoon Jang, Hajung Kim, and Jaewoo Kang. Perceiver cpi: a nested cross-attention network for compound–protein interaction prediction. Bioinformatics, 39(1):btac731, 2023.   
[17] Bharath Ramsundar, Steven Kearnes, Patrick Riley, Dale Webster, David Konerding, and Vijay Pande. Massively multitask networks for drug discovery. arXiv preprint arXiv:1502.02072, 2015.   
[18] David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of chemical information and modeling, 50(5):742–754, 2010.   
[19] Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang. Self-supervised graph transformer on large-scale molecular data. Advances in Neural Information Processing Systems, 33:12559–12571, 2020.   
[20] Chayna Sarkar, Biswadeep Das, Vikram Singh Rawat, Julie Birdie Wahlang, Arvind Nongpiur, Iadarilang Tiewsoh, Nari M Lyngdoh, Debasmita Das, Manjunath Bidarolli, and Hannah Theresa Sony. Artificial intelligence and machine learning technology driven modern drug discovery and development. International Journal of Molecular Sciences, 24(3):2026, 2023.   
[21] Kristof Schütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Robert Müller. Schnet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in neural information processing systems, 30, 2017.   
[22] Ying Song, Shuangjia Zheng, Zhangming Niu, Zhang-Hua Fu, Yutong Lu, and Yuedong Yang. Communicative representation learning on attributed molecular graphs. In IJCAI, volume 2020, pages 2831–2838, 2020.   
[23] Dagmar Stumpfe and Juürgen Bajorath. Exploring activity cliffs in medicinal chemistry: miniperspective. Journal of medicinal chemistry, 55(7):2932–2942, 2012.   
[24] Dagmar Stumpfe, Huabin Hu, and Jürgen Bajorath. Advances in exploring activity cliffs. Journal of Computer-Aided Molecular Design, 34(9):929–942, 2020.   
[25] Dagmar Stumpfe, Ye Hu, Dilyana Dimova, and Juürgen Bajorath. Recent progress in understanding activity cliffs and their utility in medicinal chemistry: miniperspective. Journal of medicinal chemistry, 57(1):18–28, 2014.   
[26] Mengying Sun, Sendong Zhao, Coryandar Gilvary, Olivier Elemento, Jiayu Zhou, and Fei Wang. Graph convolutional networks for computational drug development and discovery. Briefings in bioinformatics, 21(3):919–935, 2020.   
[27] Petar Velickovi ˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua ´ Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.   
[28] Yaqing Wang, Abulikemu Abuduweili, Quanming Yao, and Dejing Dou. Property-aware relation networks for few-shot molecular property prediction. Advances in Neural Information Processing Systems, 34:17441–17454, 2021.

[29] Yaqing Wang, Song Wang, Quanming Yao, and Dejing Dou. Hierarchical heterogeneous graph representation learning for short text classification. arXiv preprint arXiv:2111.00180, 2021.   
[30] Oliver Wieder, Stefan Kohlbacher, Mélaine Kuenemann, Arthur Garon, Pierre Ducrot, Thomas Seidel, and Thierry Langer. A compact review of molecular property prediction with graph neural networks. Drug Discovery Today: Technologies, 37:1–12, 2020.   
[31] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513–530, 2018.   
[32] Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li, Xiaomin Luo, Kaixian Chen, Hualiang Jiang, et al. Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism. Journal of medicinal chemistry, 63(16):8749–8760, 2019.   
[33] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826, 2018.   
[34] Kevin Yang, Kyle Swanson, Wengong Jin, Connor Coley, Philipp Eiden, Hua Gao, Angel Guzman-Perez, Timothy Hopper, Brian Kelley, Miriam Mathea, et al. Analyzing learned molecular representations for property prediction. Journal of chemical information and modeling, 59(8):3370–3388, 2019.   
[35] Yanqiao Zhu, Weizhi Xu, Jinghao Zhang, Yuanqi Du, Jieyu Zhang, Qiang Liu, Carl Yang, and Shu Wu. A survey on graph structure learning: Progress and opportunities. arXiv e-prints, pages arXiv–2103, 2021.