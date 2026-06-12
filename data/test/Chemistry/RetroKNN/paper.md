# Retrosynthesis Prediction with Local Template Retrieval

Shufang Xie1, Rui Yan1\*, Junliang Guo2, Yingce Xia3, Lijun Wu3, Tao Qin3

1Beijing Key Laboratory of Big Data Management and Analysis Methods, Gaoling School of Artificial Intelligence (GSAI), Renmin University of China

2Microsoft Reserarch Aisa

3Microsoft Reserarch AI4Science

{shufangxie,ruiyan}@ruc.edu.cn, {junliangguo,yingce.xia,lijuwu,taoqin}@microsoft.com

# Abstract

Retrosynthesis, which predicts the reactants of a given target molecule, is an essential task for drug discovery. In recent years, the machine learing based retrosynthesis methods have achieved promising results. In this work, we introduce RetroKNN, a local reaction template retrieval method to further boost the performance of template-based systems with non-parametric retrieval. We first build an atom-template store and a bond-template store that contain the local templates in the training data, then retrieve from these templates with a k-nearest-neighbor (KNN) search during inference. The retrieved templates are combined with neural network predictions as the final output. Furthermore, we propose a lightweight adapter to adjust the weights when combing neural network and KNN predictions conditioned on the hidden representation and the retrieved templates. We conduct comprehensive experiments on two widely used benchmarks, the USPTO-50K and USPTO-MIT. Especially for the top-1 accuracy, we improved 7.1% on the USPTO-50K dataset and 12.0% on the USPTO-MIT dataset. These results demonstrate the effectiveness of our method.

# 1 Introduction

Retrosynthesis, which predicts the reactants for a given product molecule, is a fundamental task for drug discovery. The conventional methods heavily rely on the expertise and heuristics of chemists (Corey 1991). Recently, machine learning based approaches have been proposed to assist chemists and have shown promising results (Dong et al. 2021). The typical approaches includes the template-free methods that predict the reactants directly and the templatebased methods that first predict reaction templates and then obtain reactants based on templates. For these different approaches, a shared research challenge is effectively modeling this task’s particular property.

As shown in Figure 1, a key property of a chemical reaction is that it is strongly related to modifying the local structure of the target molecule, such as replacing a functional group or breaking a bond. Therefore, much recent research focuses on better modeling the local structure of molecules (Chen and Jung 2021; Somnath et al. 2021). Despite their promising results, we notice that it is still challenging to learn all reaction patterns only with neural networks, especially for the rare templates.

![](images/4d1a5c27b7c9be9ce26667e62821c3093015c2b1cc3a20b1076addd35533c16c.jpg)

<details>
<summary>chemical</summary>

Reaction scheme showing conversion of amine to benzene via target molecule and to reactants, with reaction template annotation
</details>

Figure 1: Illustration of retrosynthesis that takes the target molecule on the left side and predicts two reactants on the right side. Inside the callout is its reaction template that breaks the carbon-nitrogen bond into two parts.

Therefore, we introduce a non-parametric retrieval-based method to provide concrete guidance in prediction. Specifically, we use a local template retrieval method, the knearest-neighbor (KNN) method, to provide additional predictions to improve the prediction accuracy. Following LocalRetro (Chen and Jung 2021), We first take a trained graph-neural network (GNN) for the retrosynthesis task and offline build an atom-template and a bond-template store that contain reaction templates (Section 2.1). During this store construction phase, we iterate all target molecules in the training data and add the templates of each atom and each bond to the corresponding store. The templates are indexed by the hidden representations extracted by the GNN. During inference, for a given new target molecule, we first use the original GNN to extract the hidden representations as well as the original GNN predicted templates. Then, we use the hidden representations to search the two stores to retrieve local templates similar to the query. The GNN predicted templates and the KNN retrieved templates are merged with different weights to build the final output.

Combining the GNN and KNN predictions is one key design factor in the above processes. The conventional way is to use fixed parameters to aggregate these predictions for all reactions, which may be sub-optimal and hurt the model’s generalization (Zheng et al. 2021). Because each prediction may have a different confidence level, it would be beneficial to assign the weights adaptively for each reaction across different instances (Section 4.1). Therefore, we employ a lightweight adapter to predict these values conditioned on the GNN representations and the retrieved results. The adapter network has a simple structure and is trained with a few samples. Although the adapter has a little extra cost, it can help improve the model performance effectively. To sum up, our contribution is two fold:

• We propose RetroKNN, a novel method to improve the retrosynthesis prediction performance with local template retrieval by the non-parametric KNN method.   
• We propose a lightweight meta-network to adaptively control the weights when combining the GNN and KNN predictions.

We conduct experiments on two widely used benchmarks: the USPTO-50K and USPTO-MIT. These datasets contain organic reactions extracted from the United States Patent and Trademark Office (USPTO) literature. On the USPTO-50K dataset, we improve the top-1 accuracy from 53.4 points to 57.2 points (7.1% relative gain) and achieved new stateof-the-art. Meanwhile, on USPTO-MIT, we improve the top-1 accuracy from 54.1 points to 60.6 points (12.0% relative gain). Moreover, our method shows promising results on the zero-shot and few-shot datasets, which are challenging settings for conventional template-based methods yet essential for this research field. These results demonstrate the effectiveness of our method.

# 2 Method

# 2.1 Preliminaries

We denote a molecule as a graph $G ( \nu , \mathcal { E } )$ where the V is the node set and the E is the bond set. Given a target molecule M as input, the retrosynthesis prediction task is to generate molecules set R that are reactants of M . Instead of directly predicting R, we follow LocalRetro (Chen and Jung 2021) that predict a local reaction template t at reaction center c and apply (t, c) to molecule M . More specifically, the t is classified into two types: atom-template $t \in \mathcal { T } _ { a }$ and bondtemplate $t \in \mathcal T _ { b }$ , depending whether c is an atom or a bond.

We also assume that there are a training set $\mathcal { D } _ { \mathrm { t r a i n } } ,$ a validation set $\mathcal { D } _ { \mathrm { v a l } } .$ , and a test set $\mathcal { D } _ { \mathrm { t e s t } }$ available. Each data split contains the target and corresponding reactants, which is formulated as $\mathcal { D } = \{ ( M _ { i } , t _ { i } , c _ { i } , \mathcal { R } _ { i } ) \} _ { i = 1 } ^ { | \bar { \mathcal { D } } | }$ where $c _ { i }$ is the reaction center of Mi to apply the template ti and |D| is the data size of D.

Meanwhile, we assume a GNN model trained on $\mathcal { D } _ { \mathrm { t r a i n } }$ exist. Without loss of generality, we split the GNN into two parts: a feature extractor f and a prediction head h. The feature extractor f takes a molecule graph G(V, E) as input and output hidden representations $h _ { \tau }$ for each node $v \in \mathcal { V }$ and $h _ { e }$ for each edge $e \in { \mathcal { E } }$ . The $h _ { v }$ and $h _ { e }$ are processed by prediction head h to predict the probability distribution over the template set $\mathcal { T } _ { a }$ and ${ \mathcal { T } } _ { b } ,$ respectively.

# 2.2 Store Construction

Our method uses two data store $\mathcal { S } _ { A }$ and $\boldsymbol { S } _ { B }$ that contain the information of atoms and bonds. Both of the store are constructed offline before inference. Inside the store are key-

Algorithm 1: store construction algorithm   
Input: Training data $D_{train}$ .
Input: Feature extractor f.
Output: Atom store $S_{A}$ and bond store $S_{B}$ .
1 Let $S_{A} := \emptyset, S_{B} := \emptyset$ ; // Initialize.
2 for $(M, t, c, \mathcal{R}) \in \mathcal{D}_{train}$ do
3    Let V denotes the node set of M;
4    Let E denotes the edge set of M;
5    for $v \in V$ ; // Loop each node.
6    do
7    Let $h_{v} := \mathbf{f}(v|M)$ ;
8    if v == c then
9    Let $S_{A} := S_{A} \cup \{(h_{v}, t)\}$ ;
10    else
11    Let $S_{A} := S_{A} \cup \{(h_{v}, 0)\}$ ;
12    end
13    end
14    for $e \in E$ ; // Loop each edge.
15    do
16    Let $h_{e} := \mathbf{f}(e|M)$ ;
17    if e == c then
18    Let $S_{B} := S_{B} \cup \{(h_{e}, t)\}$ ;
19    else
20    Let $S_{B} := S_{B} \cup \{(h_{e}, 0)\}$ ;
21    end
22    end
23 end
24 return $S_{A}, S_{B}$

value pairs that are computed from $\mathcal { D } _ { \mathrm { t r a i n } }$ and the construction procedure details are in Algorithm 1.

In this algorithm, the first step is to initialize the atom store $\mathcal { S } _ { A }$ and bond store $\boldsymbol { S } _ { B }$ as an empty set. Next, for each reaction in the training data $\mathcal { D } _ { \mathrm { t r a i n } } .$ , we iterate all nodes $v \in \mathcal V$ and all edges $e \in \mathcal { E }$ of the target molecule M in line 5 to 13 and line 14 to 22, respectively. For each node $v ,$ if it is the reaction center, we add template t that indexed by the hidden representation $h _ { v }$ to the $\mathcal { S } _ { A }$ . Otherwise, we add a special token 0 to indicate that template is not applied here. Similarly, for each edge $e ,$ we add either $( h _ { e } , t )$ or $( h _ { e } , \mathbf { 0 } )$ to the bond store $\boldsymbol { S } _ { B }$ . Finally, we get the atom store $\mathcal { S } _ { A }$ and the bond store $\boldsymbol { S } _ { B }$ used during inference.

# 2.3 Inference Method

The overview of inference procedure is available in Figure 2. At inference time, given a new target molecule M, we first compute the hidden representation $h _ { v } , h _ { \epsilon }$ and template probability $P _ { \mathrm { G N N } } ( t _ { a } | M , a ) , P _ { \mathrm { G N N } } ( t _ { b } | M , b )$ for each atom a and bond $b ,$ respectively1. Next, we retrieve the templates for each node and edge, which can be written as

![](images/2d8de0ec21a90dd6977a8472ae3144f075bcbcc55a34d1226068370897c5e07d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Legend"] --> B["Atom Store"]
    B --> C["Query"]
    C --> D["K Neighbours"]
    D --> E["Adaptor"]
    E --> F["Node Predictions"]
    F --> G["Merge"]
    G --> H["Top-50 Reactions"]
    H --> I["Edge Predictions"]
    
    subgraph Node Representation h_v
        J["Target Molecule M"]
        K["Feature Extraction GNN"]
        L["Edge Representation h_e"]
    end
    
    subgraph Node Predictions
        M["Node Predictions"]
        N["Top-50 Reactions"]
    end
    
    C --> O["Node Represlaion h_v"]
    O --> P["Node Represlaion h_v"]
    P --> Q["Node Represlaion h_v"]
    Q --> R["Node Represlaion h_v"]
    R --> S["Node Represlaion h_v"]
    
    T["Node Represlaion h_v"] --> U["Node Represlaion h_v"]
    U --> V["Node Represlaion h_v"]
    V --> W["Node Represlaion h_v"]
    
    X["GNN Predictions"] --> Y["TPL. Prob"]
    Y --> Z["OH → CH₃ 0.31"]
    Y --> AA["OH → NH₂ 0.24"]
    Y --> AB["OH → Cl 0.26"]
    
    AC["GNN Predictions"] --> AD["TPL. Prob"]
    AD --> AE["OH → CH₃ 0.31"]
    AD --> AF["OH → NH₂ 0.24"]
    AD --> AG["OH → Cl 0.26"]
    
    AH["Merge"] --> AI["TPL. Prob"]
    AI --> AJ["OH → CH₃ 0.38"]
    AI --> AK["OH → NH₂ 0.28"]
    AI --> AL["OH → Cl 0.19"]
    
    style A fill:#f9f,stroke:#333
    style H fill:#ccf,stroke:#333
```
</details>

Figure 2: The illustration of RetroKNN for a target molecule in the middle left. The top and bottom half show the examples of one atom and bond retrieval. The gray, blue, green, and brown lines denote the GNN prediction, KNN prediction, adapter input/output, and merge process, respectively. The pink table denotes the final output from all predictions.

$$
P _ {\mathrm{KNN}} (t _ {a} | M, a) \propto \sum_ {(h _ {i}, t _ {i}) \in \mathcal {N} _ {a}} \mathbb {I} _ {t _ {a} = t _ {i}} \exp \left(\frac {- d (h _ {a} , h _ {i})}{T _ {A}}\right), \tag {1}
$$

$$
P _ {\mathrm{KNN}} (t _ {b} | M, b) \propto \sum_ {(h _ {i}, t _ {i}) \in \mathcal {N} _ {b}} \mathbb {I} _ {t _ {b} = t _ {i}} \exp \left(\frac {- d (h _ {b} , h _ {i})}{T _ {B}}\right). \tag {2}
$$

In Equations (1, 2), the $\mathcal { N } _ { a } , \mathcal { N } _ { b }$ are candidates sets that retrieved from $s _ { A } , s _ { B }$ , the I is the indicator function that only outputs 1 when the condition $( { \mathrm { i . e . , ~ } } t _ { a } = t _ { i } { \mathrm { ~ o r ~ } } t _ { b } = t _ { i } )$ is satisfied, and the $T _ { A } , T _ { B }$ are the softmax temperate. Meanwhile, the $d ( \cdot , \cdot )$ is the distance function to measure the similarity between $h _ { i }$ with $h _ { v }$ or $h _ { e }$ . In another words, the $P _ { \mathrm { K N N } } ( t _ { a } | M , a )$ is proportional to the sum of the weights of the neighbours whose template is $t _ { a }$ .

Finally, we combine the GNN output and KNN output with interpolation factors λ, which is

$$
P (t _ {a} | M, a) = \lambda_ {a} P _ {\mathrm{GNN}} (t _ {a} | M, a) + (1 - \lambda_ {a}) P _ {\mathrm{KNN}} (t _ {a} | M, a), \tag {3}
$$

$$
P (t _ {b} | M, b) = \lambda_ {b} P _ {\mathrm{GNN}} (t _ {b} | M, b) + (1 - \lambda_ {b}) P _ {\mathrm{KNN}} (t _ {b} | M, b). \tag {4}
$$

In the Equation (1)-(4), the temperature $T _ { A } , T _ { B } \in \mathbb { R } ^ { + }$ and the interpolation factors $\lambda _ { a } , \lambda _ { b } \in [ 0 , 1 ]$ are predicted by the adaptor network and details are introduced in Section 2.4.

In Figure 2, we only illustrate one node and one bond retrieval as examples, but in practice, we conduct such a process for all atoms and bonds. Following LocalRetro (Chen and Jung 2021), after we get the $P ( t _ { a } | M , a )$ and $P ( t _ { b } | M , b )$ for each atom a and bond b, we will rank all non-zero predictions by their probability. The atom template and bonds templates are ranked together, and the top 50 predictions are our system’s final output.

# 2.4 Adaptor Network

To adaptively choose the $T _ { A } , T _ { B } , \lambda _ { a }$ , and $\lambda _ { b }$ for each atom and bond, we design a lightweight network to predict these values. The input to adapter are hidden representation $h _ { v } , h _ { e }$ from GNN side and distance list $d ( h _ { v } , \bar { h } _ { i } ) , d ( h _ { e } , h _ { i } )$ from the KNN side.

We use a one-layer GNN followed by a few fully connected (FC) layers for the network architecture. We use the the graph isomorphism network (GIN) with edge features (Hu et al. 2019) layer to capture both node feature $h _ { v }$ and edge feature $h _ { e } .$ , which is formulated as:

$$
h _ {v} ^ {(g)} = W _ {\mathrm{vg}} ((1 + \epsilon) h _ {v} + \sum_ {e \in \mathcal {E} (v)} \operatorname{ReLU} (h _ {v} + h _ {e})) + b _ {\mathrm{vg}}, \tag {5}
$$

where the $h _ { v } ^ { ( g ) }$ is the output, ϵ and $W$ are learnable parameters of GIN, and the $\mathcal { E } ( v )$ is the set of edges around v. Meanwhile, we use the FC layer to project the KNN distances to extract the features that can be formulated as

$$
h _ {v} ^ {(k)} = W _ {\mathrm{vk}} (\{d (h _ {v}, h _ {i}) \} _ {i = 1} ^ {K}) + b _ {\mathrm{vk}}, \tag {6}
$$

$$
h _ {e} ^ {(k)} = W _ {\mathrm{ek}} \left(\left\{d \left(h _ {e}, h _ {i}\right) \right\} _ {i = 1} ^ {K}\right) + b _ {\mathrm{ek}}, \tag {7}
$$

where the brackets $\{ \cdot \} _ { i = 1 } ^ { K }$ means building a K-dimensional vector. Finally, the feature from GNN and KNN are combined to a mixed representation, which are

$$
h _ {v} ^ {(o)} = \operatorname{ReLU} (W _ {\mathrm{vo}} \operatorname{ReLU} (h _ {v} ^ {(g)} \| h _ {v} ^ {(k)}) + b _ {\mathrm{vo}}), \tag {8}
$$

$$
h _ {e} ^ {(o)} = \operatorname{ReLU} \left(W _ {\mathrm{eo}} \operatorname{ReLU} \left(h _ {e s} ^ {(g)} \| h _ {e t} ^ {(g)} \| h _ {e} ^ {(k)}\right) + b _ {\mathrm{eo}}\right), \tag {9}
$$

where the ∥ denotes tensor concatenation and es and et are start and end node of edge e.

The $T _ { A } , \lambda _ { a }$ are predicted by $h _ { v } ^ { ( o ) }$ and the $T _ { B } , \lambda _ { b }$ are predicated by $h _ { e } ^ { ( o ) }$ by another FC layer. We also use sigmoid function σ to guarantee the $\lambda _ { a } , \lambda _ { b } \overset { \bullet } { \in } ( 0 , 1 )$ and clamp the $T _ { A } , T _ { B }$ into range [1, 100]. Formally, we have

$$
T _ {A} = \max (1, \min (1 0 0, W _ {\mathrm{ta}} h _ {v} ^ {(o)} + b _ {\mathrm{ta}})), \tag {10}
$$

$$
\lambda_ {a} = \sigma (W _ {\mathrm{la}} h _ {v} ^ {(o)} + b _ {\mathrm{la}}), \tag {11}
$$

$$
T _ {B} = \max (1, \min (1 0 0, W _ {\mathrm{tb}} h _ {e} ^ {(o)} + b _ {\mathrm{tb}}, 1, 1 0 0)), \tag {12}
$$

$$
\lambda_ {b} = \sigma (W _ {\mathrm{lb}} h _ {e} ^ {(o)} + b _ {\mathrm{lb}}). \tag {13}
$$

Because all the formulas used here are differentiable, we optimize the adapter parameters W with gradient decent to minimize the template classification loss

$$
\begin{array}{l} \mathcal {L} _ {M} = - \frac {1}{| \mathcal {V} |} \sum_ {a \in \mathcal {V}} \log P (\hat {t} _ {a} | M, a) \\ - \frac {1}{| \mathcal {E} |} \sum_ {b \in \mathcal {E}} \log P (\hat {t} _ {b} | M, b), \tag {14} \\ \end{array}
$$

for each target molecule M with node set V and edge set $\varepsilon .$ . The $P ( \hat { t } _ { a } | M ) , P ( \hat { t } _ { b } | M )$ are computed by Equation (3) and Equation (4). The $\hat { t } _ { a } , \hat { t } _ { b }$ are the ground truth template.

# 3 Experiments

# 3.1 Experimental Settings

Data. Our experiments are based on the chemical reactions extracted from the United States Patent and Trademark Office (USPTO) literature. We use two versions of the USPTO benchmark: the USPTO-50K (Coley et al. 2017) and USPTO-MIT (Jin et al. 2017). The USPTO-50K contains 50k chemical reactions, split into 40k/5k/5k reactions as training, validation, and test, respectively. Meanwhile, the USPTO-MIT consists of about 479k reactions, and the split is 409k/40k/30k. All the partitions are the same as in previous works (Coley et al. 2017; Jin et al. 2017) to make fair comparisons. We also use the preprocess scripts by Chen and Jung (2021) to extract the reaction templates from these reactions, which leads to 658 and 20,221 reaction templates in USPTO-50K and USPTO-MIT.

Implementation details. We follow the same model configuration as LocalRetro (Chen and Jung 2021) to build the backbone GNN model. The feature extractor f is a 6- layer MPNN (Gilmer et al. 2017) followed by a single GRA layer (Chen and Jung 2021) with 8 heads. We use the hidden dimension 320 and dropout 0.2. The atoms’ and bonds’ input feature is extracted by DGL-LifeSci (Li et al. 2021).The prediction head h consists two dense layers with ReLU activation. The backbone model is optimized by Adam optimizer with a learning rate of 0.001 for 50 epochs. We also early stop the training when there is no improvement in the validation loss for five epochs. The configurations for backbone are all same as Chen and Jung (2021).

The implementation of KNN is based on the faiss (Johnson, Douze, and Jegou 2019) library with ´ IndexIVFPQ index for fast embedding searching, and the K of KNN is set to 32. For the adapter network, we use the same hidden dimension as the backbone GNN. The adapter is also trained with Adam optimizer with a learning rate of 0.001. Considering the data size difference, we train the adapter for ten epochs and two epochs on the validation set of the USPTO-50K and USPTO-MIT datasets, respectively. The adapter with the best validation loss is used for test.

Evaluation and baselines Following previous works, our system will predict top-50 results for each target molecule and report the top-K accuracy where K=1,3,5,10, and 50 by the script from Chen and Jung (2021). We also use representative baseline systems in recent years, include:

• Template-based methods: retrosim (Coley et al. 2017), neuralsym (Segler and Waller 2017), GLN (Dai et al. 2020), Hopfield (Seidl et al. 2021), and LocalRetro (Chen and Jung 2021);   
• Semi-template based methods: G2Gs (Shi et al. 2021), RetroXpert (Yan et al. 2020), and GraphRtro (Somnath et al. 2021);   
• Tempate-free methods: Transformer (Lin et al. 2020), MEGAN (Sacha et al. 2021), Chemformer (Irwin et al. 2021), GTA (Seo et al. 2021), and DualTF (Sun et al. 2021).

# 3.2 Main Results

The experimental results of the USPTO-50K benchmark are shown in Table 1 when the reaction type is unknown and in Table 2 when the reaction type is given. Meanwhile, the results on the USPTO-MIT benchmark are in Table 3. In these tables, we sort all systems by their top-1 accuracy and mark their type by filling the cycle symbols. Our method (RetroKNN) is in the last row and highlighted in bold.

Comparing these accuracy numbers, we can find that our method outperforms the baseline systems with a large margin. When the reaction type is unknown, we achieved 57.2 points top-1 accuracy and improved the backbone result from LocalRetro by 3.8 points, which is a 7.1% relative gain.

<table><tr><td>Method</td><td>TPL.</td><td>K = 1</td><td>3</td><td>5</td><td>10</td><td>50</td></tr><tr><td>retrosim</td><td>●</td><td>37.3</td><td>54.7</td><td>63.3</td><td>74.1</td><td>85.3</td></tr><tr><td>neuralsym</td><td>●</td><td>44.4</td><td>65.3</td><td>72.4</td><td>78.9</td><td>83.1</td></tr><tr><td>MEGAN</td><td>○</td><td>48.1</td><td>70.7</td><td>78.4</td><td>86.1</td><td>93.2</td></tr><tr><td>G2Gs</td><td>○</td><td>48.9</td><td>67.6</td><td>72.5</td><td>75.5</td><td>-</td></tr><tr><td>RetroXpert</td><td>○</td><td>50.4</td><td>61.1</td><td>62.3</td><td>63.4</td><td>64.0</td></tr><tr><td>GTA</td><td>○</td><td>51.1</td><td>67.6</td><td>67.8</td><td>81.6</td><td>-</td></tr><tr><td>Hopfield</td><td>●</td><td>51.8</td><td>74.6</td><td>81.2</td><td>88.1</td><td>94.0</td></tr><tr><td>GLN</td><td>●</td><td>52.5</td><td>69,0</td><td>75.6</td><td>83.7</td><td>92.4</td></tr><tr><td>LocalRetro</td><td>●</td><td>53.4</td><td>77.5</td><td>85.9</td><td>92.4</td><td>97.7</td></tr><tr><td>Dual-TF</td><td>○</td><td>53.6</td><td>70.7</td><td>74.6</td><td>77.0</td><td>-</td></tr><tr><td>GraphRetro</td><td>○</td><td>53.7</td><td>68.3</td><td>72.2</td><td>75.5</td><td>-</td></tr><tr><td>Chemformer</td><td>○</td><td>54.3</td><td>-</td><td>62.3</td><td>63.0</td><td>-</td></tr><tr><td>RetroKNN</td><td>●</td><td>57.2</td><td>78.9</td><td>86.4</td><td>92.7</td><td>98.1</td></tr></table>

Table 1: Top-K exact match accuracy on the USPTO-50K dataset when the reaction type is unknown. The , , and denote template-based, semi-template, and template-free, respectively. Systems are ordered by top-1 accuracy.

<table><tr><td>Method</td><td>TPL.</td><td>K = 1</td><td>3</td><td>5</td><td>10</td><td>50</td></tr><tr><td>retrosim</td><td>●</td><td>52.9</td><td>73.8</td><td>81.2</td><td>88.1</td><td>-</td></tr><tr><td>neuralsym</td><td>●</td><td>55.3</td><td>76.0</td><td>81.4</td><td>85.1</td><td>-</td></tr><tr><td>MEGAN</td><td>○</td><td>60.7</td><td>82.0</td><td>87.5</td><td>91.6</td><td>95.3</td></tr><tr><td>G2Gs</td><td>○</td><td>61.0</td><td>81.3</td><td>86.0</td><td>88.7</td><td>-</td></tr><tr><td>RetroXpert</td><td>○</td><td>62.1</td><td>75.8</td><td>78.5</td><td>80.9</td><td>-</td></tr><tr><td>GraphRetro</td><td>○</td><td>63.9</td><td>81.5</td><td>85.2</td><td>88.1</td><td>-</td></tr><tr><td>LocalRetro</td><td>●</td><td>63.9</td><td>86.8</td><td>92.4</td><td>96.3</td><td>97.9</td></tr><tr><td>GLN</td><td>●</td><td>64.2</td><td>79.1</td><td>85.2</td><td>90.0</td><td>93.2</td></tr><tr><td>Dual-TF</td><td>○</td><td>65.7</td><td>81.9</td><td>84.7</td><td>85.9</td><td>-</td></tr><tr><td>RetroKNN</td><td>●</td><td>66.7</td><td>88.2</td><td>93.6</td><td>96.6</td><td>98.4</td></tr></table>

Table 2: Top-K exact match accuracy on the USPTO-50K dataset when the reaction type is given. The , , and denote template-based, semi-template, and template-free, respectively. Systems are ordered by top-1 accuracy.

<table><tr><td>Method</td><td>TPL.</td><td>K = 1</td><td>3</td><td>5</td><td>10</td><td>50</td></tr><tr><td>Seq2Seq</td><td>○</td><td>46.9</td><td>61.6</td><td>66.3</td><td>70.8</td><td>-</td></tr><tr><td>neuralsym</td><td>●</td><td>47.8</td><td>67.9</td><td>74.1</td><td>80.2</td><td>-</td></tr><tr><td>Transformer</td><td>○</td><td>54.1</td><td>71.8</td><td>76.9</td><td>81.8</td><td>-</td></tr><tr><td>LocalRetro</td><td>●</td><td>54.1</td><td>73.7</td><td>79.4</td><td>84.4</td><td>90.4</td></tr><tr><td>RetroKNN</td><td>●</td><td>60.6</td><td>77.1</td><td>82.3</td><td>87.3</td><td>92.9</td></tr></table>

Table 3: Top-K exact match accuracy on the USPTO-MIT dataset. The and  denote template-based and templatefree methods. Systems are ordered by top-1 accuracy.

When the reaction type is given, we also improve the top-1 accuracy by 2.8 points from 63.9 to 66.7. Meanwhile, on USPTO-MIT, our method shows 60.6 points top-1 accuracy with a 6.5 points improvement or 12% relative gain. More importantly, these top-1 accuracies are also better than other strong baselines and state-of-the-art, demonstrating the effectiveness of our method.

At the same time, we achieved 78.9 points top-3 accuracy and 86.4 points accuracy in USPTO-50K when the reaction type is unknown, which are also much higher than baselines. For the top-10 and top-50 accuracy, we get 92.7 and 98.1 points accuracy. Considering that the accuracy is already very high, the improvement is still significant.

To sum up, the local template retrial method efficiently improves the retrosynthesis prediction accuracy.

# 4 Study and Analysis

# 4.1 Case Study

Retrieval case study. To better understand if we can retrieve useful reactions by the hidden representations, we conducted case studies on the USPTO-50K datasets, and the results are shown in Figure 3. We fist select an atom-template reaction and the first bond-template reaction from the data. Next, we query the atom and bond store by the corresponding atom and bond. Finally, for each retrieved template, we show the original target molecule in the training data, where the reaction atom/bond is highlighted by green background. The bond-template and atom-template reactions are available in the figure’s first and second rows. In each row, we first show the target molecule M of the reaction and then five neighbors of M. From these cases, we can find that the neighborhoods retrieved by hidden representations can effetely capture the local structure of molecules. For example, the carbon-nitrogen bond retrieves all neighbors in the edge-template reaction. Moreover, all carbon atoms are surrounded by oxygen in a double bond (=O) and a trifluorocarbon (-CF3), and all nitrogen atoms are connected to an aromatic ring. Meanwhile, for the node-template reaction, all retrieved atoms are the oxygen atoms that are connected to a phenyl. In conclusion, retrieving molecules with hidden representations is efficient because it can capture the local structure well. Therefore, we can improve the prediction accuracy by using the retrieved templates.

Adapter case study. We show three representative cases for the effect of adapter in Table 4. In each row, we show the target molecule and ground truth template id, then the λ and T output by the adapter, and finally the GNN prediction and KNN retrieved neighbors. When the GNN prediction is accurate in the first row, the adapter will generate a high λ value (e.g., 0.96) so that the GNN output has a higher weight. However, when that is not the case (the second and third row), the λ tends to be lower (e.g., 0.14), which gives more weight to KNN prediction. Meanwhile, when only the N1 has the correct prediction (the second row), the adapter tends to output a small T (e.g., 7.89) to make the sharp distribution that gives more weight to N1’s prediction. On the contrary (the third row), the adapter tends to output a larger value (e.g., 19.36) so that more neighbors can contribute to the final output. Moreover, our statistics show that when λ < 0.5, the GNN and KNN accuracy are 46.9% and 69.2%, showing that KNN is complementary to GNN prediction.

# 4.2 Zero-shot and Few-shot Study

We modify the USPTO-50K dataset to zero-shot and fewshot versions to study the domain adaptation ability of our method. Specifically, in the USPTO-50K data, each reaction has its reaction class available in class 1 to 10. To build the zero-shot data, we filter the train and validation data by removing all reactions with reaction class 6 to 10 and only keeping those with reaction class 1 to 5. Similarly, to build the few-shot data, we only keep 10% of reactions that have class 6 to 10. Finally, we evaluate the performance of these new data with the LocalRetro baseline and our RetroKNN method. The results are summarized in Figure 4.

From these plots, we notice that zero-shot is a challenging setting for conventional template-based methods, which is a known shortcoming of this kind of methods. However, when combined with KNN, our system can generate meaningful results. For example, in reaction class 8, the RetroKNN haves 6.1 points top-5 accuracy and 9.8 points top-10 accuracy in the zero-shot data. The few-shot setting is easier than the zero-shot because a few examples are available during training. Nevertheless, the RetroKNN also outperforms baseline on all reaction types. On average, the RetroKNN improved 8.56 points top-5 accuracy and 5.64 points top-10 accuracy. These results show that our method is can also improve the performance on zero/few-shot data, which are important scenarios in this field.

![](images/8e713319dd258b29d74902b80608898911349b5672217cf3caec68724899c620.jpg)

Figure 3: Case study of retrieved molecules. The bonds and atoms used in retrieval are highlighted by green background. The first column shows the target molecules, and the rest show five neighbourhood targets from the training data. 

<table><tr><td>Target Molecule</td><td>GT.</td><td> $\lambda$ </td><td>T</td><td>GNN</td><td>N1</td><td>N2</td><td>N3</td><td>N4</td><td>N5</td></tr><tr><td>Cc1ccc(-c2cccnc2C#N)cc1</td><td>b542</td><td>0.96</td><td>21.42</td><td>b542</td><td>b519(67.51)</td><td>b519(77.35)</td><td>b519(77.35)</td><td>b519(77.35)</td><td>b0(104.00)</td></tr><tr><td>CCOc1ccc(C[C@H](NC(=O)C(F)(F)F)C(=O)O)cc1</td><td>b524</td><td>0.14</td><td>7.89</td><td>b495</td><td>b524(22.79)</td><td>b523(33.84)</td><td>b495(67.3)</td><td>b495(76.21)</td><td>b495(76.55)</td></tr><tr><td>CC1(C)CC(=O)N(Cc2cccccc2)c2ccc(C#Cc3ccc(C(=O)O)cc3)cc21</td><td>a121</td><td>0.02</td><td>19.36</td><td>a124</td><td>a121(34.41)</td><td>a121(57.3)</td><td>a121(58.4)</td><td>a0(59.91)</td><td>a0(61.17)</td></tr></table>

Table 4: Case study of parameter T and λ. The GT. denotes ground truth template id, GNN denotes the GNN prediction, and N1 to N5 denotes five neighbors. The prefix a, b of template id means it is an atom or bond template. We show each neighbor’s distance in the brackets below template id. The correct predictions are highlighted in bold.

![](images/5c5b24218a1801db6ec13a95295b92b7e26541cdc50a132cd84a1dcf9dfa2502.jpg)

<details>
<summary>bar</summary>

|        | Baseline | RetroKNN |
| ------ | -------- | -------- |
| C6     | 0        | 0        |
| C7     | 0        | 3        |
| C8     | 0        | 6        |
| C9     | 3        | 8        |
| C10    | 0        | 4        |
</details>

![](images/7f3c69da657f43b3212f1a5a38c1c54077948daed3d30b5a9edcb1a431d08731.jpg)

<details>
<summary>bar</summary>

(b) Top 5 Acc. on few-shot data.
| Model | Accuracy |
|---|---|
| C6 | 72 |
| C7 | 68 |
| C8 | 64 |
| C9 | 40 |
| C10 | 80 |
</details>

![](images/c132e03ae0a6b9b66495110c16ae991ff90c29ff16ecffc5976163c787328f33.jpg)

<details>
<summary>bar</summary>

|        | Baseline | RetroKNN |
| ------ | -------- | -------- |
| C6     | 0        | 0        |
| C7     | 0        | 5.5      |
| C8     | 0        | 10       |
| C9     | 4        | 13       |
| C10    | 0        | 4.5      |
</details>

![](images/d84b29f592adb06d4f57e1228c1ff85947c120c742f9351bad878b60966f2317.jpg)

<details>
<summary>bar</summary>

(d) Top 10 Acc. on few-shot data.
| Model | Accuracy |
|---|---|
| C6 | 85 |
| C7 | 80 |
| C8 | 80 |
| C9 | 50 |
| C10 | 82 |
</details>

Figure 4: Top-5 (a, b) and top-10 (c, d) accuracy (Acc.) on the zero-shot (a, c) and few-shot (b, d) data. The columns C6 to C10 denote different reaction classes.

# 4.3 Ablation Study

We conducted an ablation study on the USPTO-50K dataset to study the contributions of different components, and the results are shown in Table 5. We show the top-1 accuracy in the table by comparing different systems. The system ⃝1 is the LocalRetro baseline without using KNN, which achieved 53.4 points accuracy. In system $\textcircled{2}$ , we add the KNN without using the adapter. To find the optimal paramters, we conduct comprehensive grid search on by $T ^ { \star } \in \ \{ 1 , 5 , 2 5 , 5 0 \}$ and $\lambda \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ , which leads to total 20 combinations. We select the parameters by the validation loss and finally get the 56.3 points accuracy. Furthermore, in system ⃝3 , we add the adapter only for T and keep the λ same as system ⃝2 . Similarly, we only add the adapter only for λ in system ⃝4 . The system ⃝5 is the full RetroKNN model.

Comparing the system ⃝1 with others that using KNN, we can find that introducing KNN to this task can effectively improve the model performance. These numbers show that the local template retrieval is vital for the system. Meanwhile, comparing system ⃝3 ⃝4 to system ⃝2 , we notice that adding both T and λ adapter is helpful. Finally, when both parameters are adaptively predicted in system ⃝5 , the accuracy can be boosted to 57.2, showing that they can work together effectively. Therefore, all components are necessary for this system.

<table><tr><td>ID</td><td>System</td><td>Accuracy</td></tr><tr><td>1</td><td>Baseline</td><td>53.4</td></tr><tr><td>2</td><td>+ KNN</td><td>56.3</td></tr><tr><td>3</td><td>+ KNN, adaptive T</td><td>56.7</td></tr><tr><td>4</td><td>+ KNN, adaptive λ</td><td>56.8</td></tr><tr><td>5</td><td>+ KNN, adaptive T, adaptive λ</td><td>57.2</td></tr></table>

Table 5: Ablation study on the USPTO-50K dataset when the reaction type is unknown.

<table><tr><td>#Retrieved reactions</td><td>1</td><td>4</td><td>8</td><td>16</td><td>32</td></tr><tr><td>Accuracy</td><td>55.6</td><td>57.4</td><td>57.1</td><td>56.9</td><td>57.2</td></tr></table>

Table 6: Study on the number of retrieved reactions by KNN.

# 4.4 Retrieved Templates Size

In Table 6, we show how the number of retrieved reactions (i.e., K of KNN) affects the model performance. More specifically, in the KNN search, we set the K ∈ [1, 4, 8, 16, 32], then train adapters for each of them. Finally, we report the top-1 accuracy in the table.

From these results, we first observe that only adding one retrieved template (K=1) can improve the accuracy from 53.4 to 55.6. When K is ≥ than 4, the accuracy can be further improved to around 57 points. There will be no further significant improvement when more reactions are retrieved, nor will more received templates hurt the performance. We suppose it is because there is already enough information to improve the accuracy as the templates far from the query will contribute less to the prediction.

# 4.5 Inference Latency

In Table 7, we study the datastore size and the inference latency. The last two rows present the latency with or without retrieval during inference, which are measured on a machine with a single NVIDIA A100 GPU. Each latency value, which is the average run time per reaction, is measured with ten independent runs. In the USPTO-50K dataset, we observe that the average latency increased from 2.71 ms to 3.31 ms, which is about 0.6 ms for each reaction. The extra latency is a little more prominent for the USPTO-MIT dataset because it is about ten times larger than the USPTO-50K. However, considering the hours or even days that a more accurate system can save for chemists, the extra tenmillisecond cost is not a real obstacle to the practical use of this method. Finally, some work (He, Neubig, and Berg-Kirkpatrick 2021; Meng et al. 2021) show that the KNN speed can be further accelerated, and we would like to add these techniques in future work.

<table><tr><td>Dataset</td><td>USPTO-50K</td><td>USPTO-MIT</td></tr><tr><td> $|\mathcal{D}_{\text{train}}|$ </td><td>40k</td><td>409k</td></tr><tr><td> $|\mathcal{S}_A|$ </td><td>1,039k</td><td>10,012k</td></tr><tr><td> $|\mathcal{S}_B|$ </td><td>2,241k</td><td>21,495k</td></tr><tr><td>Latency w/o KNN</td><td>2.71 ± 0.02 ms</td><td>3.51 ± 0.05 ms</td></tr><tr><td>Latency w/ KNN</td><td>3.31 ± 0.09 ms</td><td>14.69 ± 0.29 ms</td></tr></table>

Table 7: Study of the datastore size and inference latency.

# 5 Related Work

# 5.1 Retrosynthesis Prediction

Retrosynthesis prediction is an essential task for scientific discovery and have achieved promising results in recent years (Segler and Waller 2017; Liu et al. 2017; Coley et al. 2017; Tetko et al. 2020; Irwin et al. 2021; Dai et al. 2020; Yan et al. 2020; Seidl et al. 2021; Chen and Jung 2021; Shi et al. 2021; Somnath et al. 2021; Wan et al. 2022). A few research also use retrieval mechanisms for this task. For example, Seidl et al. (2021) use Hopfield networks to select templates, and Lee et al. (2021) use retrieval method to fetch molecules from a database. Being differently, we are the first to combine deep learning and KNN retrieval in this task.

# 5.2 Retrieval Methods

Retrieving from data store or memory to improve the machine learning model’s performance is an important research topic. SVM-KNN (Zhang et al. 2006) first combines the SVM and KNN for recognition tasks. Furthermore, the KNN-LM (Khandelwal et al. 2020) and KNN-MT (Khandelwal et al. 2021) have shown promising results when combining KNN with Transformer networks. Meanwhile, He, Neubig, and Berg-Kirkpatrick (2021); Meng et al. (2021) study the speed of retrival methods and Zheng et al. (2021) study the adaptation problem. However, we are the first to combine the strong capability of KNN with GNN and use them on the retrosynthesis task.

# 6 Conclusion

Retrosynthesis prediction is essential for scientific discovery, especially drug discovery and healthcare. In this work, we propose a novel method to improve prediction accuracy using local template retrieval. We first build the atom and bond stores with the training data and a trained GNN and retrieve templates from these stores during inference. The retrieved templates are combined with the original GNN predictions to make the final output. We further leverage a lightweight adapter to adaptively predict the weights to integrate the GNN predictions and retrieved templates. We greatly advanced the prediction performance on two widely used benchmarks, the USPTO-50K and USPTO-MIT, reaching 57.2 and 60.6 points for top-1 accuracy. These results demonstrate the effectiveness of our methods.

# Acknowledgements

We would like to thank the anonymous reviewers for their insightful comments. This work was supported by National Natural Science Foundation of China (NSFC Grant No. 62122089 and No. 61876196), Beijing Outstanding Young Scientist Program NO. BJJWZYJH012019100020098, and Intelligent Social Governance Platform, Major Innovation & Planning Interdisciplinary Platform for the “Double-First Class” Initiative, Renmin University of China. We also wish to acknowledge the support provided and contribution made by Public Policy and Decision-making Research Lab of RUC. Rui Yan is supported by Beijing Academy of Artificial Intelligence (BAAI).

# References

Chen, S.; and Jung, Y. 2021. Deep Retrosynthetic Reaction Prediction using Local Reactivity and Global Attention. JACS Au.   
Coley, C. W.; Rogers, L.; Green, W. H.; and Jensen, K. F. 2017. Computer-Assisted Retrosynthesis Based on Molecular Similarity. ACS Central Science, 3(12): 1237–1245.   
Corey, E. J. 1991. The logic of chemical synthesis. Ripol Classic.   
Dai, H.; Li, C.; Coley, C. W.; Dai, B.; and Song, L. 2020. Retrosynthesis Prediction with Conditional Graph Logic Network. arXiv:2001.01408.   
Dong, J.; Zhao, M.; Liu, Y.; Su, Y.; and Zeng, X. 2021. Deep learning in retrosynthesis planning: datasets, models and tools. Briefings in Bioinformatics, 00(August): 1–15.   
Gilmer, J.; Schoenholz, S. S.; Riley, P. F.; Vinyals, O.; and Dahl, G. E. 2017. Neural message passing for quantum chemistry. In International conference on machine learning, 1263–1272. PMLR.   
He, J.; Neubig, G.; and Berg-Kirkpatrick, T. 2021. Efficient Nearest Neighbor Language Models. EMNLP.   
Hu, W.; Liu, B.; Gomes, J.; Zitnik, M.; Liang, P.; Pande, V.; and Leskovec, J. 2019. Strategies for Pre-training Graph Neural Networks. arXiv:1905.12265.   
Irwin, R.; Dimitriadis, S.; He, J.; and Bjerrum, E. J. 2021. Chemformer: A Pre-Trained Transformer for Computational Chemistry. Machine Learning: Science and Technology, 3.   
Jin, W.; Coley, C.; Barzilay, R.; and Jaakkola, T. 2017. Predicting organic reaction outcomes with weisfeiler-lehman network. Advances in neural information processing systems, 30.   
Johnson, J.; Douze, M.; and Jegou, H. 2019. Billion-scale ´ similarity search with GPUs. IEEE Transactions on Big Data, 7(3): 535–547.   
Khandelwal, U.; Fan, A.; Jurafsky, D.; Zettlemoyer, L.; and Lewis, M. 2021. Nearest Neighbor Machine Translation. In International Conference on Learning Representations.   
Khandelwal, U.; Levy, O.; Jurafsky, D.; Zettlemoyer, L.; and Lewis, M. 2020. Generalization through Memorization: Nearest Neighbor Language Models. In International Conference on Learning Representations (ICLR).

Lee, H.; Ahn, S.; Seo, S.-W.; Song, Y. Y.; Yang, E.; Hwang, S. J.; and Shin, J. 2021. RetCL: A Selection-based Approach for Retrosynthesis via Contrastive Learning. In IJ-CAI, 2673–2679.   
Li, M.; Zhou, J.; Hu, J.; Fan, W.; Zhang, Y.; Gu, Y.; and Karypis, G. 2021. DGL-LifeSci: An Open-Source Toolkit for Deep Learning on Graphs in Life Science. ACS Omega.   
Lin, K.; Pei, J.; Lai, L.; and Xu, Y. 2020. Automatic Retrosynthetic Pathway Planning Using Template-free Models. Chemical Science.   
Liu, B.; Ramsundar, B.; Kawthekar, P.; Shi, J.; Gomes, J.; Luu Nguyen, Q.; Ho, S.; Sloane, J.; Wender, P.; and Pande, V. 2017. Retrosynthetic Reaction Prediction Using Neural Sequence-to-Sequence Models. ACS Central Science, 3(10): 1103–1113.   
Meng, Y.; Li, X.; Zheng, X.; Wu, F.; Sun, X.; Zhang, T.; and Li, J. 2021. Fast Nearest Neighbor Machine Translation. arXiv:2105.14528.   
Sacha, M.; Błaz, M.; Byrski, P.; Dabrowski-Tumanski, P.; Chrominski, M.; Loska, R.; Włodarczyk-Pruszynski, P.; and Jastrzebski, S. 2021. Molecule edit graph attention network: modeling chemical reactions as sequences of graph edits. Journal of Chemical Information and Modeling, 61(7): 3273–3284.   
Segler, M. H. S.; and Waller, M. P. 2017. Neural-Symbolic Machine Learning for Retrosynthesis and Reaction Prediction. Chemistry - A European Journal, 23(25): 5966–5971.   
Seidl, P.; Renz, P.; Dyubankova, N.; Neves, P.; Verhoeven, J.; Segler, M.; Wegner, J. K.; Hochreiter, S.; and Klambauer, G. 2021. Modern Hopfield Networks for Few- and Zero-Shot Reaction Template Prediction. arXiv:2104.03279.   
Seo, S.-W.; Young Song, Y.; Yong Yang, J.; Bae, S.; Lee, H.; Shin, J.; Ju Hwang, S.; and Yang, E. 2021. GTA: Graph Truncated Attention for Retrosynthesis. AAAI.   
Shi, C.; Xu, M.; Guo, H.; Zhang, M.; and Tang, J. 2021. A Graph to Graphs Framework for Retrosynthesis Prediction. arXiv:2003.12725.   
Somnath, V. R.; Bunne, C.; Coley, C. W.; Krause, A.; and Barzilay, R. 2021. Learning Graph Models for Retrosynthesis Prediction. arXiv:2006.07038.   
Sun, R.; Dai, H.; Li, L.; Kearnes, S.; and Dai, B. 2021. Towards understanding retrosynthesis by energy-based models. NeurIPS, 9.   
Tetko, I. V.; Karpov, P.; Van Deursen, R.; and Godin, G. 2020. State-of-the-art augmented NLP transformer models for direct and single-step retrosynthesis. Nature Communications, 11(1): 5575.   
Wan, Y.; Liao, B.; Hsieh, C.-Y.; and Zhang, S. 2022. Retroformer: Pushing the Limits of Interpretable End-to-end Retrosynthesis Transformer. arXiv:2201.12475.   
Yan, C.; Ding, Q.; Zhao, P.; Zheng, S.; Yang, J.; Yu, Y.; and Huang, J. 2020. Retroxpert: Decompose retrosynthesis prediction like a chemist. Advances in Neural Information Processing Systems, 33: 11248–11258.   
Zhang, H.; Berg, A. C.; Maire, M.; and Malik, J. 2006. SVM-KNN: Discriminative nearest neighbor classification

for visual category recognition. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), volume 2, 2126–2136. IEEE.   
Zheng, X.; Zhang, Z.; Guo, J.; Huang, S.; Chen, B.; Luo, W.; and Chen, J. 2021. Adaptive Nearest Neighbor Machine Translation. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers).