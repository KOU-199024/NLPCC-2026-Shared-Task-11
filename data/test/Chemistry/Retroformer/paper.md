# RETROFORMER: PUSHING THE LIMITS OF END-TO-END RETROSYNTHESIS TRANSFORMER

Yue Wan, Benben Liao, Chang-Yu Hsieh1, Shengyu Zhang2

Tencent Quantum Laboratory

Shenzhen, China

kimhsieh@tencent.com1

shengyzhang@tencent.com2

# ABSTRACT

Retrosynthesis prediction is one of the fundamental challenges in organic synthesis. The task is to predict the reactants given a core product. With the advancement of machine learning, computer-aided synthesis planning has gained increasing interest. Numerous methods were proposed to solve this problem with different levels of dependency on additional chemical knowledge. In this paper, we propose Retroformer, a novel Transformer-based architecture for retrosynthesis prediction without relying on any cheminformatics tools for molecule editing. Via the proposed local attention head, the model can jointly encode the molecular sequence and graph, and efficiently exchange information between the local reactive region and the global reaction context. Retroformer reaches the new state-of-the-art accuracy for the end-to-end template-free retrosynthesis, and improves over many strong baselines on better molecule and reaction validity. In addition, its generative procedure is highly interpretable and controllable. Overall, Retroformer pushes the limits of the reaction reasoning ability of deep generative models.

# 1 Introduction

Retrosynthesis [8] is one of the major building blocks in organic synthesis, which aims to discover valid and efficient synthetic routes (i.e., reactants) given a target molecule (i.e., product). It is crucial for the pharmaceutical industry as one of the main challenges for drug discovery is to efficiently synthesize novel and complex compounds in the laboratory [2].

Recently, computer-aided synthesis planning has gained vast attention for its potential to save a tremendous amount of time and efforts from traditional retrosynthesis approaches. Various machine learning approaches were proposed with different levels of dependency on additional chemical knowledge. These methods can be categorized into three groups. First, template-based methods [7, 9, 5] view the retrosynthesis prediction as the template retrieval problem, where a template encodes the core reactive rule (Figure 1). After the templates are retrieved, these methods use cheminformatics tools like RDKit [1] to build up full reactions from the templates. Despite the state-of-the-art accuracy and guaranteed molecule validity, these methods are limited to the scope of the existing template database. In contrast, template-free methods, the second class, use deep generative models to directly generate the reactants given the product. Since molecule can be represented by both the graph and the SMILES sequence, existing approaches reframe the retrosynthesis into either sequence-to-sequence [18, 3, 37, 28, 25, 15] or graph-to-sequence problem [29]. These generative methods do not rely on any additional chemical knowledge and can perform chemical reasoning within a larger reaction space. The third class is semi-template-based methods, which combine the advantages of both the generative models and the additional chemical knowledge. Conventional frameworks [34, 26, 27, 33] in this category follow the same idea: They first identify the reactive bond and convert the product into synthons by RDKit. Then, another model completes synthons into reactants. These methods are competitive in accuracy and are interpretable by their stage-wise nature.

In this work, we are interested in the template-free generative approach for retrosynthesis prediction. Existing methods fail to fully explore the potential of deep generative model in terms of reaction reasoning, and we argue that the end-to-end Transformer-based [30] architecture can reach the same competitive benchmark accuracy as well as good validity and interpretability. We propose Retroformer, a novel end-to-end retrosynthesis Transformer that introduces a special attention head. It is able to jointly encode the sequential and graphical information of the molecule and allow efficient information exchange between the local reactive region and the global reaction context. The generative process is also sensitive to the exact reactive region. Our end-to-end model does not rely on any additional helps from the cheminformatic tools for molecule editing. Experiments show that our model can improve over the vanilla Transformer by 12.5% and 14.4% top-10 accuracy in the reaction class known and unknown settings, respectively. It reaches the new state-of-the-art accuracy for template-free methods and is competitive against both template-based and semi-template-based methods. It also enjoys better molecule and reaction validity compared to strong baseline models. The model is highly interpretable and controllable for downstream usage. Our contributions are summarized as:

![](images/3f05246c3034336720e92e3a0b866a554f1e5c3c7b33319a530ed20764c80d18.jpg)

<details>
<summary>chemical</summary>

Reaction scheme showing acylation and related processes with fluorinated amine and carbonyl groups, alongside a reaction template for C5-N4 to form C2-C1-N4-C5.
</details>

Figure 1: Sample reaction (top) and its corresponding reaction template (bottom).

• We propose Retroformer, a novel Transformer-based architecture that introduces the local attention head, to push the limits of the reaction reasoning ability of deep generative models in retrosynthesis prediction.   
• The proposed method reaches 64% and 53.2% top-1 accuracy for reaction class known and unknown settings, respectively, which is the new state-of-the-art performance for template-free retrosynthesis.   
• The proposed method further improves the top-10 molecule and reaction validity by 23.6% and 22.0%, respectively, compared to the vanilla retrosynthesis Transformer.

# 2 Related Work

# 2.1 Retrosynthesis Prediction

Existing methods in retrosynthesis prediction can be grouped into three categories: template-based, template-free, and semi-template-based. The reaction template encodes the core reactive rules. As shown in Figure 1, a conventional template tells the potential reactive region within the molecule, as well as its potential chemical transformation. These templates are either expert-defined or automatically extracted by algorithms. In this work, we strictly differentiate the three categories by the levels of dependency on additional chemical knowledge during inference.

Template-based methods rely on an external template database. Since the template is a more efficient and interpretable representation for reactions [13, 32], a large body of works [7, 9, 5] focus on capturing the reactive scores between the molecules and templates. Retrosim [7] uses molecule fingerprint similarity to rank the candidate templates. GLN [9] and LocalRetro [5] use graph neural network (GNN) to capture the molecule-template and atom/bond-template relationship, respectively. Despite their state-of-the-art top-k accuracy, all template-based methods suffer from the incomplete coverage issue and do not scale well.

Template-free methods, in contrast, adopt deep generative models to directly generate the reactants molecules. Besides graph, molecules can be represented using SMILES sequence. Existing works [18, 3, 37, 28, 25, 15] take advantage of the Transformer [30] architecture and reframe the problem as the sequence-to-sequence translation from product to reactants. Graph2SMILES [29] replaces the original sequence encoder with a graph encoder to ensure the permutation invariance of SMILES. These methods rely on little additional chemical knowledge for inference. However, chemical validity can be a huge concern because validity is often not part of the training objective. Another factor is the ignorance of graphical structure during the sequence generation. Also, generated outcomes from beam search often suffer from the diversity issue [31], which is another practical concern for retrosynthesis.

Semi-template-based methods combine the advantage of both the generative models and additional chemical knowledge. In this work, we strictly categorize generative models which require additional help from RDKit [1] for molecule editing into this method group. Most existing works [34, 26, 27, 33] approach the task by a two-stage procedure.

![](images/1a290992702bc977d0b2ffe0026f383dc09f61d787f52937f36010b48f74fb23.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Local-Global Encoder"] -->|h| B["Atom RC Identifier"]
    A -->|A| C["Bond RC Identifier"]
    B --> D["Local-Global Decoder"]
    C --> D
    D --> E["CC (=O)OC(C)=O.Nc1cccccc1F"]
    D --> F["CC (=O)OC(C)=O.Nc1cccccc1"]
    G["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    H["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    I["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    J["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    K["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    L["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    M["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    N["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    O["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    P["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    Q["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    R["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    S["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    T["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    U["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    V["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    W["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    X["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    Y["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    Z["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AA["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AB["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AC["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AD["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AE["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AF["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AG["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AH["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AI["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AJ["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AK["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AL["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AM["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AN["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AO["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AP["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AQ["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AR["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AS["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AT["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AU["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AV["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AW["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AX["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AY["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    AZ["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BA["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BB["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BC["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BD["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BE["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BF["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BG["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BH["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BI["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BJ["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BK["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BL["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BM["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BN["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BO["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BP["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
    BQ["CC (=O)OC(C)=O.Nc1cccccc1F"] --> D
```
</details>

Figure 2: Architecture overview. The model takes molecular SMILES S and a) bond feature matrix A as inputs. Besides the encoder outputs h, the b) predicted reaction center $S _ { r c }$ is c) converted to attention masking and passed to the decoder.

![](images/3b57b6683c2633323ccb048a851e417f3a148748318197e44f3e83c0ecf0046e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["MatMul"] --> B["SoftMax"]
    A --> C["Scale"]
    A --> D["MatMul"]
    B --> E["Q"]
    C --> F["K"]
    D --> G["V"]
    H["Scaled Dot-Product Attention"] --> I["Linear"]
    H --> J["Linear"]
    H --> K["Linear"]
    L["Concat"] --> M["Concat_ij"]
    N["Local head (en)"] --> O["MatMul"]
    P["h^{t+1}"] --> Q["Linear"]
    Q --> R["Concat"]
    R --> S["Linear"]
    T["A_{ij}^{t+1}"] --> U["+"]
    V["V_j"] --> W["SoftMax"]
    W --> X["Scale"]
    X --> Y["MatMul"]
    Y --> Z["K_j"]
    Z --> AA["A_{ij}^t"]
    AB["global head"] --> AC["MatMul"]
    AD["Local head"] --> AE["MatMul"]
    AF["Local head"] --> AG["MatMul"]
```
</details>

Figure 3: Local-global self attention head in encoder.

Despite their architecture differences between GNN and Transformer, they follow the same idea: They first convert the product into synthons by breaking the reactive bond via RDKit, then complete the synthons into reactants by either leaving groups selection [27], graph generation [26], or SMILES generation [34, 33]. In contrast, MEGAN [20] reframes the generative procedure as a sequence of graph edits that are completed by RDKit.

Besides different architecture designs, self-supervised molecule pretraining is also shown to be effective in retrosynthesis prediction. DMP-fusion [38] pretrains the molecule with a dual view of SMILES and graph. Chemformer [14] applies masked SMILES modeling to learn the molecule representation.

# 2.2 Graph Transformer

The introduction of Transformer into the graph domain has gained increasing interest. The global receptive fields of the self-attention and the local message passing of the graph neural network are inherently complementary and compatible. Attempts have been made in ways of incorporating graph information into the self-attention computation [35, 39, 40] and integrating the conventional graph neural networks [12] with Transformer architecture [11, 4].

# 3 Preliminary

Let ${ \cal S } = [ s _ { 1 } , s _ { 2 } , . . . s _ { n } ]$ be the molecular SMILES sequence with n number of tokens. Let $G _ { m o l } = ( V _ { m } , E _ { m } )$ be the molecular graph. It is formed by $| V _ { m } |$ number of atoms with $| E _ { m } |$ number of bonds. For computation convenience, we further introduce the SMILES graph $G _ { s m i } = ( V _ { s } , E _ { s } )$ . Vs is made up of all the SMILES tokens, including the atom tokens $( \mathrm { e . g . , \mathrm { \vec { ~ } C ^ { 9 } , \mathrm { \vec { ~ } C ^ { 9 } } } } )$ as well as the other special tokens $( \mathrm { e . g . , } \mathrm { e } ^ { \mathrm { < } } \mathrm { = } ^ { \mathrm { > } } , \mathrm { } ^ { \mathrm { < } } \mathrm { 1 } ^ { \mathrm { > } } ) \colon V _ { m } \subseteq V _ { s } = S$ . In $G _ { s m i }$ , the special tokens are treated as trivial nodes with no neighbors. Its edge $E _ { s }$ represents the graphical connections between atom tokens, which is essentially the same as bond connections $E _ { m }$ in $G _ { m o l }$ . In general, $G _ { s m i }$ is a larger but sparser graph compared to $G _ { m o l }$ . The introduction of $G _ { s m i }$ is merely to ensure the alignment relationship between the atoms in graph and the tokens in SMILES.

# 4 Retroformer

We propose Retroformer, a novel Transformer-based model that is able to perform interpretable retrosynthesis prediction in an end-to-end manner. We propose a special type of local attention head that can support efficient information exchange between the local region of reactive importance and the global reaction context. Its generative procedure is also sensitive to the exact local region. The overall training and inference can be done in an end-to-end manner. It is a fully template-free method without any additional dependency on RDKit for molecule editing. The overall architecture contains an encoder, a decoder, and two reaction center identifiers. We also propose to use SMILES alignment and on-the-fly data augmentation as two additional training strategies.

# 4.1 Local-Global Encoder with Edge Update

Since molecular graph can provide additional information on top of the SMILES sequence, our encoder takes both S sequence and $G _ { s m i } ( \mathrm { i } . \mathrm { e }$ ., adjacency matrix and bond feature) as inputs. The bond features we considered are listed in Appendix 10. Different from the existing graph Transformers [35, 39, 40] that compute graph self-attention within the entire module, our model encodes the graph information at the head level. We specify two types of attention heads: global head and local head. The global head is the same as the vanilla self-attention head, where its receptive field is the entire SMILES sequence. The local head, on the other hand, considers the topological structure of the molecule. The receptive field of the individual token is restricted to its one-hop neighborhood, which is similar to [36]. In addition, we perform element-wise multiplication between the key vector and the edge feature to incorporate the bond information into the calculation. The roll-out form of the local head self-attention at layer l for the $i ^ { t h }$ token is formulated as:

$$
x _ {i} ^ {l + 1} \text {   local   } = \sum_ {j \in N (i)} \sigma (\frac {q _ {i} (A _ {i j} ^ {l} \odot k _ {j}) ^ {T}}{\sqrt {d}}) v _ {j} \tag {1}
$$

$$
[ q _ {i}, k _ {j}, v _ {j} ] = [ h _ {i} ^ {l} W ^ {Q}, h _ {j} ^ {l} W ^ {K}, h _ {j} ^ {l} W ^ {V} ]
$$

where A is the bond feature matrix, $W ^ { Q } , W ^ { K } , W ^ { V }$ are the projection matrix for query q, key $k ,$ and value $v ,$ and σ is the softmax operation. The computed representations from the global and local heads are then concatenated along the hidden dimension and passed to a linear layer, which represents the updated token features $h ^ { l + 1 }$ . Meanwhile, the edge update module is a fully connected layer (FFN) that takes the concatenation of the updated features of the receiving and sending tokens as inputs:

$$
h ^ {l + 1} = \text { Linear } ([ x _ {\text { global }} ^ {l + 1}; x _ {\text { local }} ^ {l + 1} ]) \tag {2}
$$

$$
A _ {i j} ^ {l + 1} = A _ {i j} ^ {l} + \operatorname{FFN} ([ h _ {i} ^ {l + 1}; h _ {j} ^ {l + 1} ]) \tag {3}
$$

The integration of the local, global attention heads, and the edge update module allows the model to efficiently exchange information between the local region and global molecular context. Same as the vanilla Transformer [30], layer normalization and residual connection are enforced between encoder layers. The final encoder outputs are the updated token representation h and the bond representation A.

# 4.2 Reaction Center Detection

A reaction center represents the group of atoms and bonds that are contributing factors to the chemical transformation. However, existing semi-template-based methods [34, 27, 26, 33] simplify this concept as the reactive bond. We argue that this simplification will lead to information loss of the reaction context. These methods also cannot perform retrosynthesis in an end-to-end manner, since they rely on RDKit to convert the product into synthons. Instead, Retroformer predicts the reactive probability $P _ { r c } ( . )$ of each atom and bond and infers the reactive region of S as the attention receptive field for the decoder. In other words, the detected reaction center $S _ { r c }$ is a subset of S.

![](images/ebdc1f8c9a1cf32be51a6bd144ac742e0edad26008eae1da030938c9655acc51.jpg)

<details>
<summary>chemical</summary>

Chemical structures of Graph and SMILES for Product and Reactants, showing carbon chain and functional groups
</details>

Figure 4: Token and atom alignment between product and reactants in SMILES and graph representations, respectively. The right most figure is the ground truth alignment matrix.

Figure 2b shows a heat map visualization of the predicted reactive probability. It is done by two fully connected layers named Atom RC Identifier and Bond RC Identifier:

$$
P _ {r c} \left(s _ {i}\right) = \sigma \left(\mathrm{FFN} _ {\text { atom }} \left(h _ {i}\right)\right), s _ {i} \in V _ {m} \tag {4}
$$

$$
P _ {r c} \left(e _ {i j}\right) = \sigma \left(\mathrm{FFN} _ {\text { bond }} \left(A _ {i j}\right)\right), e _ {i j} \in E _ {m} \tag {5}
$$

We will show in Section 5.3 that the learned reaction center can be easily visualized and matched with chemical heuristics. We then convert the atom and the bond reactive probability into the reactive indicator of tokens in $S _ { r c }$ by either one of the following two strategies:

• naive: we naively set a token as reactive if it exists in a reactive edge $( \mathrm { i . e . , } P _ { r c } ( e ) > 0 . 5 )$ and is reactive itself $( \mathrm { i } . \mathrm { e } . , P _ { r c } ( s ) > 0 . 5 )$ . Note that the special tokens are guaranteed to be non-reactive. This strategy is used at both training and inference stages.   
• search: we conduct a subgraph search on the molecular graph and rank the subgraphs by their reaction center score: Psi∈Src l $\sum { _ { s _ { i } \in S _ { r } } }$ og Prc(si) + Psi,sj∈Src l $\begin{array} { r } { P _ { r c } ( s _ { i } ) + \sum _ { s _ { i } , s _ { j } \in S _ { r c } } } \end{array}$ og $P _ { r c } ( e _ { i j } )$ . Only atoms with $P _ { r c } ( s ) > \alpha _ { a t o m }$ and bonds with $P _ { r c } ( e ) > \alpha _ { b o n d }$ are considered in the search to reduce the computational time. Detailed algorithm is described in Appendix 9. Then, top-n subgraphs are selected as reaction center candidates. The model then generates $k / n$ reactants for each reaction center, where k is the total number of predicted reactants. The final results are ranked by the sum of the reaction center score and the generative score. This strategy is only used at inference stage.

# 4.3 Local-Global Decoder

The decoder takes its generative outcomes from the previous step, the encoder outputs h, and the reaction center $S _ { r c }$ as inputs. Similar to the encoder, we also introduce two different heads in its cross-attention module. The global head is the same as the vanilla head. The local head, on the contrary, is only visible to the detected reaction center $S _ { r c } .$ . It computes the sparse cross-attention instead of the full cross-attention.

$$
y _ {i} ^ {l + 1} _ {\text { local }} = \sum_ {s _ {j} \in S _ {r c}} \sigma (\frac {q _ {i} k _ {j} {} ^ {T}}{\sqrt {d}}) v _ {j} \tag {6}
$$

$$
[ q _ {i}, k _ {j}, v _ {j} ] = [ g _ {i} ^ {l} W ^ {Q}, h _ {j} W ^ {K}, h _ {j} W ^ {V} ]
$$

Same as the encoder, the computed representations from the global and local heads are then concatenated along the hidden dimension and passed to a linear layer, which represents the representation $g ^ { l + 1 }$ 1. It essentially converts the decoder into a conditional generative module.

$$
g ^ {l + 1} = \operatorname{Linear} ([ y _ {\text {   global }} ^ {l + 1}; y _ {\text {   local }} ^ {l + 1} ]) \tag {7}
$$

# 4.4 SMILES Alignment

SMILES alignment is an additional learning task of Retroformer. Similar to machine translation, the SMILES sequences of the source and the target molecules are often partially aligned. A large portion of the molecules remains unchanged during the reaction. Figure 4 shows this alignment relationship in both graph and SMILES representations. The node alignment between graphs (i.e., atom mapping) can be easily converted into token alignment between SMILES. Detailed substring matching algorithm with atom-mapping is described in Appendix 8.

Table 1: Top-k accuracy for retrosynthesis prediction on USPTO-50K. \* indicates the model with SMILES augmentation. For comparison purpose, the Aug. Transformer is evaluated without the test augmentation. Best performance is in bold. 

<table><tr><td rowspan="3">Model</td><td colspan="8">Top-k accuracy (%)</td></tr><tr><td colspan="4">Reaction class known</td><td colspan="4">Reaction class unknown</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td colspan="9">Template-Based</td></tr><tr><td>GLN [9]</td><td>64.2</td><td>79.1</td><td>85.2</td><td>90.0</td><td>52.5</td><td>69.0</td><td>75.6</td><td>83.7</td></tr><tr><td>LocalRetro [5]</td><td>63.9</td><td>86.8</td><td>92.4</td><td>96.3</td><td>53.4</td><td>77.5</td><td>85.9</td><td>92.4</td></tr><tr><td colspan="9">Template-Free</td></tr><tr><td>Transformer</td><td>57.1</td><td>71.5</td><td>75.0</td><td>77.7</td><td>42.4</td><td>58.6</td><td>63.8</td><td>67.7</td></tr><tr><td>SCROP [37]</td><td>59.0</td><td>74.8</td><td>78.1</td><td>81.1</td><td>43.7</td><td>60.0</td><td>65.2</td><td>68.7</td></tr><tr><td>Tied Transformer [15]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>47.1</td><td>67.1</td><td>73.1</td><td>76.3</td></tr><tr><td>Aug. Transformer* [28]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>48.3</td><td>-</td><td>73.4</td><td>77.4</td></tr><tr><td>GTA* [25]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>51.1</td><td>67.6</td><td>74.8</td><td>81.6</td></tr><tr><td>Graph2SMILES [29]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>52.9</td><td>66.5</td><td>70.0</td><td>72.9</td></tr><tr><td>Retroformerbase (Ours)</td><td>61.5</td><td>78.3</td><td>82.0</td><td>84.9</td><td>47.9</td><td>62.9</td><td>66.6</td><td>70.7</td></tr><tr><td>Retroformeraug* (Ours)</td><td>64.0</td><td>81.8</td><td>85.4</td><td>88.3</td><td>52.9</td><td>68.2</td><td>72.5</td><td>76.4</td></tr><tr><td>Retroformeraug+* (Ours)</td><td>64.0</td><td>82.5</td><td>86.7</td><td>90.2</td><td>53.2</td><td>71.1</td><td>76.6</td><td>82.1</td></tr><tr><td colspan="9">Semi-Template-Based</td></tr><tr><td>RetroXpert* [34]</td><td>62.1</td><td>75.8</td><td>78.5</td><td>80.9</td><td>50.4</td><td>61.1</td><td>62.3</td><td>63.4</td></tr><tr><td>G2G [26]</td><td>61.0</td><td>81.3</td><td>86.0</td><td>88.7</td><td>48.9</td><td>67.6</td><td>72.5</td><td>75.5</td></tr><tr><td>GraphRetro [27]</td><td>63.9</td><td>81.5</td><td>85.2</td><td>88.1</td><td>53.7</td><td>68.3</td><td>72.2</td><td>75.5</td></tr><tr><td>RetroPrime* [33]</td><td>64.8</td><td>81.6</td><td>85.0</td><td>86.9</td><td>51.4</td><td>70.8</td><td>74.0</td><td>76.1</td></tr><tr><td>MEGAN [20]</td><td>60.7</td><td>82.0</td><td>87.5</td><td>91.6</td><td>48.1</td><td>70.7</td><td>78.4</td><td>86.1</td></tr></table>

Inspired by the effectiveness of the guided attention in [10], we introduce the attention guidance loss between the ground truth alignment and the attention weights from the decoder’s global cross-attention heads. We treat the computed cross-attention at each decoder step as a probability distribution and impose a label smoothing loss [19]. It is a soft cross entropy loss with the label smoothing technique and is shown to be effective in classification performance. Hypothetically speaking, this guided attention can encourage the model to understand chemical reactions more efficiently.

# 4.5 Data Augmentation

We follow the same data augmentation tricks used in [25, 28] for the SMILES generative models, which are the SMILES permutation of the product and the order permutation of reactants. However, instead of expanding the training dataset off-the-shelf, we choose to perform the augmentation on-the-fly. At each iteration, there is a probability of 50% to permute the product SMILES and another probability of 50% to permute the reactants order. This dynamic permutation allows the model to focus more on the canonical SMILES and use the permuted SMILES for regularization.

# 4.6 Loss

The training schema of Retroformer can be viewed as an end-to-end multi-task learning. The overall loss is made up of four parts: $\mathcal { L } = \mathcal { L } _ { L M } + \mathcal { L } _ { R C _ { b o n d } } + \mathcal { L } _ { R C _ { a t o m } } + \mathcal { L } _ { A G }$ , where $\mathcal { L } _ { L M }$ is the language modeling objective, $\mathcal { L } _ { R C , }$ ∗ is the reactive probability loss, and $\mathcal { L } _ { A G }$ is the SMILES attention guidance loss.

# 5 Experiments

Data We use the conventional retrosynthesis benchmark dataset USPTO-50K [21] to evaluate our method. It contains 50016 atom-mapped reactions that are grouped into 10 reaction classes. We use the same data split as [7]. We canonicalize the molecule SMILES with atom mapping following the same protocol given in [27]. We then use RDChiral [6] to extract the ground truth reaction center and use the algorithm in Appendix 8 to extract the ground truth SMILES token alignment.

Evaluation We adopt the conventional top-k accuracy of the full reactants to evaluate the retrosynthesis performance. We also evaluate the top-k validity of the generated routes. For molecule validity, we treat a candidate as valid if RDKit [1] can successfully identify the molecule SMILES. The top-k validity is calculated as: $V a l i d ( k ) =$ $\textstyle { \frac { 1 } { N \times k } } \sum _ { 1 } ^ { N } \sum _ { 1 } ^ { k }$ 1(SMILES is valid). We further evaluate our method with the round-trip accuracy [24], which is an approximation metric for reaction validity. It measures the percentage of predicted reactants that can lead back to the original product. We take the pretrained Molecule Transformer [23] as the oracle reaction prediction model because of its state-of-the-art performance. Our top-k round-trip accuracy calculation is slightly different from the definition adopted by RetroPrime [33] and LocalRetro [5]: RoundT $\begin{array} { r } { r i p ( k ) = \frac { 1 } { N \times k } \sum _ { 1 } ^ { N } \sum _ { 1 } ^ { k } ] } \end{array}$ 1(Reach Ground Truth Product).

Table 2: Top-k SMILES validity for retrosynthesis prediction on USPTO-50K with reaction class unknown. 

<table><tr><td rowspan="2">Model</td><td colspan="4">Top-k validity (%)</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td>Transformer</td><td>97.2</td><td>87.9</td><td>82.4</td><td>73.1</td></tr><tr><td>Graph2SMILES</td><td>99.4</td><td>90.9</td><td>84.9</td><td>74.9</td></tr><tr><td>RetroPrime</td><td>98.9</td><td>98.2</td><td>97.1</td><td>92.5</td></tr><tr><td> $Retroformer_{aug}$ </td><td>99.3</td><td>98.5</td><td>97.2</td><td>92.6</td></tr><tr><td> $Retroformer_{aug}+$ </td><td>99.2</td><td>98.5</td><td>97.4</td><td>96.7</td></tr></table>

Table 3: Top-k round-trip accuracy for retrosynthesis prediction on USPTO-50K with reaction class unknown. 

<table><tr><td rowspan="2">Model</td><td colspan="4">Top-k round-trip acc. (%)</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td>Transformer</td><td>71.9</td><td>54.7</td><td>46.2</td><td>35.6</td></tr><tr><td>Graph2SMILES</td><td>76.7</td><td>56.0</td><td>46.4</td><td>34.9</td></tr><tr><td>RetroPrime</td><td>79.6</td><td>59.6</td><td>50.3</td><td>40.4</td></tr><tr><td>Retroformeraug</td><td>78.6</td><td>71.8</td><td>67.1</td><td>57.6</td></tr><tr><td>Retroformeraug+</td><td>78.9</td><td>72.0</td><td>67.1</td><td>57.2</td></tr></table>

Baseline We take GLN [9] and LocalRetro [5] as two strong baseline models from the template-based group. We take SCROP [37], Tied Transformer [15], Augmented Transformer [28], GTA [25], and Graph2Smiles [29] as the baseline models from the template-free group. We also train a vanilla retrosynthesis Transformer from scratch using OpenNMT [17]. We take RetroXpert [34], G2G [26], GraphRetro [27], RetroPrime [33], and MEGAN [20] as strong semi-template-based baselines. We do not include the pretraining approach in the performance comparison. We experiment with three variants of the proposed model: Retroformer $\mathrm { \ b a s e }$ represents the model with no data augmentation and the naive reaction center detection strategy; Retroforme $\mathrm { r _ { a u g } }$ represents the model with data augmentation and the naive strategy; $\mathrm { R e t r o f o r m e r _ { a u g } + }$ represents the model with data augmentation and the search strategy.

Implementation Details Built on top of the vanilla Transformer [30], our model has 8 encoder layers and 8 decoder layers. The model is trained using the Adam optimizer [16] with a fixed learning rate of $1 e - 4$ , and a dropout rate of 0.3. The embedding dimension is set to 256, and the total amount of heads is set to 8. We split the heads by half for global and local heads. The Retroforme $\mathrm { r } _ { \mathrm { b a s e } }$ is trained on 1 NVIDIA Tesla V100 GPU for 24 hours.

# 5.1 Performance

Top-k Accuracy With the reaction class known, our augmented model can achieve a 64.0% top-1 and 88.3% top-10 accuracy. It reaches the state-of-the-art performance for template-free methods and is competitive against templatebased and semi-template-based methods. It improves over the vanilla retrosynthesis Transformer by 6.9% top-1 and 11.9% top-10, respectively. With the reaction class unknown, our augmented model can achieve a 52.9% top-1 and 76.4% top-10 accuracy. The top-1 accuracy reaches the state-of-the-art performance as Graph2SMILES. In addition, Retroforme $\mathbf { \dot { b } a s e }$ surpasses the vanilla retrosynthesis Transformer by a large margin in both settings. It demonstrates the promising potential for the deep generative model to perform end-to-end retrosynthesis prediction and reaction space exploration.

We further demonstrate the strength of the reaction center detection. With the top-n subgraphs proposed, Retroforme $\mathrm { { r } _ { a u g } + }$ can further boost the performance to the new state-of-the-art accuracy for template-free retrosynthesis in both experiment settings. We provide further interpretation of the search strategy in Section 5.3 and Appendix 9.

Table 4: Effects of different modules on retrosynthesis performance in reaction class unknown setting. 

<table><tr><td colspan="2">Settings</td><td colspan="4">Modules</td><td colspan="4">Top-k accuracy (%)</td></tr><tr><td></td><td>Guidedlast</td><td>Guidedall</td><td>Local-global Encoder</td><td>Local-global Decoder</td><td>Reaction Center Search</td><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td>(a)</td><td></td><td></td><td>√</td><td>√</td><td></td><td>45.5</td><td>60.7</td><td>65.4</td><td>69.9</td></tr><tr><td>(b)</td><td></td><td>√</td><td>√</td><td>√</td><td></td><td>47.0</td><td>63.1</td><td>66.9</td><td>71.1</td></tr><tr><td>(c)</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td>47.9</td><td>62.9</td><td>66.6</td><td>70.7</td></tr><tr><td>(d)</td><td>√</td><td></td><td></td><td>√</td><td></td><td>44.1</td><td>60.1</td><td>64.7</td><td>70.2</td></tr><tr><td>(e)</td><td>√</td><td></td><td>√</td><td></td><td></td><td>46.7</td><td>63.7</td><td>68.4</td><td>73.9</td></tr><tr><td>(f)</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>48.4</td><td>66.8</td><td>73.2</td><td>78.8</td></tr></table>

Top-k SMILES Validity We take the vanilla retrosynthesis Transformer, Graph2SMILES, and RetroPrime as strong SMILES generative baselines for validity comparison with our model. We do not include template-based methods in this evaluation since molecule SMILES built from templates are guaranteed to be valid. As we mentioned before, SMILES generative models are more likely to struggle with the validity issue. Without knowing the proper reaction center, the models may modify the molecule fragments that are distant from the core reactive region, which is chemically trivial. As shown in Table 2, both of our model variants enjoy better molecule validity than others. It improves the top-10 validity over the vanilla Transformer by 23.6%. It shows that being aware of the local reactive region can encourage the model to avoid errors that propagate via the non-reactive regions.

Top-k Round-trip Accuracy To measure the reaction validity, we take the pretrained Molecule Transformer [23] as the oracle reaction prediction model to measure the percentage of top-k proposed synthetic routes that can lead back to the ground truth product. Table 3 shows the performance comparison of the round-trip accuracy. It shows that our method improves over the existing methods by a large margin. Our model exceeds the vanilla Transformer by 22.0% top-10 round-trip accuracy, and it also improves over RetroPrime by 12.2%. It shows that our model is more likely to propose valid and efficient synthetic routes for downstream usage.

# 5.2 Ablation Study

We further conduct ablation study to evaluate the effects of each component on retrosynthesis performance. As for the guided alignment loss, we experiment with two settings: $\mathrm { G u i d e d _ { \mathrm { a l l } } \mathrm { : } }$ the alignment loss is enforced at the first global heads of all decoder layers; $\mathrm { G u i d e d _ { l a s t } } \mathrm { : }$ the loss is enforced at the first global head of the last decoder layer.

Table 4 shows that all proposed components are necessary for Retroforme $\mathrm { \Delta t _ { b a s e } }$ to reach the best retrosynthesis performance. The improvement from (a) to (b) and (a) to (c) shows that the model can better capture the reaction knowledge from learning the SMILES alignment. We choose (c) over (b) as our final alignment loss because of its comparable performance and its lighter training duty. The 2.9% top-1 improvement from (d) to (c) indicates the effectiveness of our local-global graph Transformer encoder. Comparing (c) and (e), we could see that the local-global decoder achieves higher top-1 accuracy than the full global decoder, whereas the latter version has better top-k accuracy for $k > 1$ This is also reasonable. Focusing on a specific reaction center makes the generative process more constrained. The performance drop for $k > 1$ indicates the loss of outcome diversity. However, the local-global decoder is compatible with the reaction center search, whereas the full global decoder is not. With the search strategy, the model can boost the top-k accuracy and improves the top-10 accuracy by 4.9% from (e) to (f).

# 5.3 Qualitative Analysis

In addition to its competitive benchmark performance, Retroformer is fully interpretable and controllable with external chemical guidance.

Reaction Center Detection To evaluate the interpretability and the quality of the detected reaction center, we randomly select a product molecule from the test set of USPTO-50K and predict the reactants with the search strategy. We also evaluate the setting where we explicitly specify a reaction center and give it to the decoder. We term this setting as the reaction center retrieval. As shown in Figure 5, the search algorithm proposes three different reaction centers (highlighted in green) given the raw reactive probabilities. The numbers represent the reactive scores of each subgraph candidate. In this example, the top candidate corresponds to the ground truth reaction center in the data. The third column indicates the top-2 verified predicted reactants given both the reaction center and the encoded molecule. The numbers represent the generative scores of each reactants. It shows that the model can understand the concept of reaction center and propose chemical transformations compatible with it. The outcomes from the retrieved reaction center (highlighted in blue) also demonstrate that the generation is fully controllable by external guidance.

![](images/0d884867d54391176a884eebdd72f29b913e58e07eb19f391621b3b52348ecc7.jpg)

<details>
<summary>chemical</summary>

Reaction center detection and retrieval steps for four candidates, showing molecular structures and their corresponding ground truth values.
</details>

Figure 5: Generative procedure for a randomly selected molecule.   
![](images/31e69da5af0b253182c555d8300acef00c8e09a905ddffb7c8deeafc145b5939.jpg)

<details>
<summary>chemical</summary>

Two chemical reaction pathways showing structural transformations under Success Case and Failure Case, with labeled atoms and bonds
</details>

Figure 6: Sample Atom Mapping.

Atom Mapping Since our model is trained to learn the token alignment between the source and the target SMILES, the predicted attention can be easily converted to atom mapping. Different from the RXNMapper [22] that uses an additional neighbor attention multiplier to calculate the atom mapping from the attention weights, we directly use the attention weights to do the assignment while ignoring the molecular graphical structure. Note that it does not guarantee either the one-to-one mapping from product atom to reactants atom or the equivalence of the mapped element. Figure 6(a) shows a success case of the inferred atom mapping. Figure 6(b) shows a typical failure case. This assignment mistakenly aligns [O:11] with [N:11]. The mistake is explainable since [HN:8] and [O:11] within the reactants are the exact position where chemical transformation happens. Also, this naive atom mapping fails to assign the one-to-one mapping, which is also reasonable because of the symmetry present in the second reactant.

# 6 Conclusion

We propose Retroformer, a novel Transformer-based architecture that reaches the new state-of-the-art performance for template-free retrosynthesis. With the proposed local attention heads and the incorporation of the graph information, the model is able to identify local reactive regions and generate reactants conditionally on the detected reaction center. Being aware of the reaction center also encourages the model to generate reactants with improved molecule validity, reaction validity, and interpretability. We plan to further research the multi-step template-free retrosynthesis planning problem using Retroformer as the single-step retrosynthesis prediction backbone.

# References

[1] Rdkit: Open-source cheminformatics.   
[2] BLAKEMORE, D. C., CASTRO, L., CHURCHER, I., REES, D. C., THOMAS, A. W., WILSON, D. M., AND WOOD, A. Organic synthesis provides opportunities to transform drug discovery. Nature Chemistry 10, 4 (apr 2018), 383–394.   
[3] CHEN, B., SHEN, T., JAAKKOLA, T. S., AND BARZILAY, R. Learning to make generalizable and diverse predictions for retrosynthesis, 2019.   
[4] CHEN, J., ZHENG, S., SONG, Y., RAO, J., AND YANG, Y. Learning attributed graph representation with communicative message passing transformer. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21 (8 2021), Z.-H. Zhou, Ed., International Joint Conferences on Artificial Intelligence Organization, pp. 2242–2248. Main Track.   
[5] CHEN, S., AND JUNG, Y. Deep retrosynthetic reaction prediction using local reactivity and global attention. JACS Au (2021).   
[6] COLEY, C. W., GREEN, W. H., AND JENSEN, K. F. Rdchiral: An rdkit wrapper for handling stereochemistry in retrosynthetic template extraction and application. Journal of Chemical Information and Modeling 59, 6 (2019), 2529–2537.   
[7] COLEY, C. W., ROGERS, L., GREEN, W. H., AND JENSEN, K. F. Computer-assisted retrosynthesis based on molecular similarity. ACS Central Science 3, 12 (nov 2017), 1237–1245.   
[8] COREY, E., AND CHENG, X. The Logic of Chemical Synthesis. Wiley, 1989.   
[9] DAI, H., LI, C., COLEY, C., DAI, B., AND SONG, L. Retrosynthesis prediction with conditional graph logic network. In Advances in Neural Information Processing Systems (2019), pp. 8870–8880.   
[10] DESHPANDE, A., AND NARASIMHAN, K. Guiding attention for self-supervised learning with transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020 (Online, Nov. 2020), Association for Computational Linguistics, pp. 4676–4686.   
[11] DWIVEDI, V. P., AND BRESSON, X. A generalization of transformer networks to graphs. AAAI Workshop on Deep Learning on Graphs: Methods and Applications (2021).   
[12] GILMER, J., SCHOENHOLZ, S. S., RILEY, P. F., VINYALS, O., AND DAHL, G. E. Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning - Volume 70 (2017), ICML’17, JMLR.org, p. 1263–1272.   
[13] HEID, E., LIU, J., AUDE, A., AND GREEN, W. H. On the influence of template size, canonicalization and exclusivity for retrosynthesis and reaction prediction applications. ChemRxiv (2021).   
[14] IRWIN, R., DIMITRIADIS, S., HE, J., AND BJERRUM, E. J. Chemformer: A pre-trained transformer for computational chemistry. Machine Learning: Science and Technology (2021).   
[15] KIM, E., LEE, D., KWON, Y., PARK, M. S., AND CHOI, Y.-S. Valid, plausible, and diverse retrosynthesis using tied two-way transformers with latent variables. Journal of Chemical Information and Modeling 61, 1 (2021), 123–133. PMID: 33410697.   
[16] KINGMA, D. P., AND BA, J. Adam: A method for stochastic optimization, 2017.

[17] KLEIN, G., KIM, Y., DENG, Y., SENELLART, J., AND RUSH, A. OpenNMT: Open-source toolkit for neural machine translation. In Proceedings of ACL 2017, System Demonstrations (Vancouver, Canada, July 2017), Association for Computational Linguistics, pp. 67–72.   
[18] LIN, K., XU, Y., PEI, J., AND LAI, L. Automatic retrosynthetic route planning using template-free models. Chem. Sci. 11 (2020), 3355–3364.   
[19] PEREYRA, G., TUCKER, G., CHOROWSKI, J., KAISER, L., AND HINTON, G. E. Regularizing neural networks by penalizing confident output distributions. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Workshop Track Proceedings (2017), OpenReview.net.   
[20] SACHA, M., BŁAZ˙ , M., BYRSKI, P., D ˛ABROWSKI-TUMANSKI ´ , P., CHROMINSKI ´ , M., LOSKA, R., WŁODARCZYK-PRUSZYNSKI ´ , P., AND JASTRZ ˛EBSKI, S. Molecule edit graph attention network: Modeling chemical reactions as sequences of graph edits. Journal of Chemical Information and Modeling 61, 7 (2021), 3273–3284. PMID: 34251814.   
[21] SCHNEIDER, N., STIEFL, N., AND LANDRUM, G. A. What’s what: The (nearly) definitive guide to reaction role assignment. Journal of Chemical Information and Modeling 56, 12 (2016), 2336–2346. PMID: 28024398.   
[22] SCHWALLER, P., HOOVER, B., REYMOND, J.-L., STROBELT, H., AND LAINO, T. Extraction of organic chemistry grammar from unsupervised learning of chemical reactions. Science Advances 7, 15 (2021), eabe4166.   
[23] SCHWALLER, P., LAINO, T., GAUDIN, T., BOLGAR, P., HUNTER, C. A., BEKAS, C., AND LEE, A. A. Molecular transformer: A model for uncertainty-calibrated chemical reaction prediction. ACS Central Science 5, 9 (2019), 1572–1583. PMID: 31572784.   
[24] SCHWALLER, P., PETRAGLIA, R., ZULLO, V., NAIR, V. H., HAEUSELMANN, R. A., PISONI, R., BEKAS, C., IULIANO, A., AND LAINO, T. Predicting retrosynthetic pathways using transformer-based models and a hyper-graph exploration strategy. Chem. Sci. 11 (2020), 3316–3325.   
[25] SEO, S.-W., SONG, Y. Y., YANG, J. Y., BAE, S., LEE, H., SHIN, J., HWANG, S. J., AND YANG, E. Gta: Graph truncated attention for retrosynthesis. Proceedings of the AAAI Conference on Artificial Intelligence 35, 1 (May 2021), 531–539.   
[26] SHI, C., XU, M., GUO, H., ZHANG, M., AND TANG, J. A graph to graphs framework for retrosynthesis prediction. In Proceedings of the 37th International Conference on Machine Learning (ICML) (2020), pp. 8818– 8827.   
[27] SOMNATH, V. R., BUNNE, C., COLEY, C. W., KRAUSE, A., AND BARZILAY, R. Learning graph models for template-free retrosynthesis, 2020.   
[28] TETKO, I. V., KARPOV, P., VAN DEURSEN, R., AND GODIN, G. State-of-the-art augmented nlp transformer models for direct and single-step retrosynthesis. Nature Communications (11 2020).   
[29] TU, Z., AND COLEY, C. W. Permutation invariant graph-to-sequence model for template-free retrosynthesis and reaction prediction, 2021.   
[30] VASWANI, A., SHAZEER, N., PARMAR, N., USZKOREIT, J., JONES, L., GOMEZ, A. N., KAISER, L. U., AND POLOSUKHIN, I. Attention is all you need. In Advances in Neural Information Processing Systems (2017), I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30, Curran Associates, Inc.   
[31] VIJAYAKUMAR, A. K., COGSWELL, M., SELVARAJU, R. R., SUN, Q., LEE, S., CRANDALL, D., AND BATRA, D. Diverse beam search: Decoding diverse solutions from neural sequence models, 2018.   
[32] WAN, Y., LI, X., WANG, X., YAO, X., LIAO, B., HSIEH, C.-Y., AND ZHANG, S. Neuraltpl: a deep learning approach for efficient reaction space exploration. ChemRxiv (2021).   
[33] WANG, X., LI, Y., QIU, J., CHEN, G., LIU, H., LIAO, B., HSIEH, C.-Y., AND YAO, X. Retroprime: A diverse, plausible and transformer-based method for single-step retrosynthesis predictions. Chemical Engineering Journal 420 (2021), 129845.   
[34] YAN, C., DING, Q., ZHAO, P., ZHENG, S., YANG, J., YU, Y., AND HUANG, J. Retroxpert: Decompose retrosynthesis prediction like a chemist. In Advances in Neural Information Processing Systems (NeurIPS) 33. Curran Associates, Inc., 2020, pp. 11248–11258.   
[35] YING, C., CAI, T., LUO, S., ZHENG, S., KE, G., HE, D., SHEN, Y., AND LIU, T.-Y. Do transformers really perform bad for graph representation?, 2021.   
[36] ZHANG, X.-C., WU, C.-K., YANG, Z.-J., WU, Z.-X., YI, J.-C., HSIEH, C.-Y., HOU, T.-J., AND CAO, D.-S. MG-BERT: leveraging unsupervised atomic representation learning for molecular property prediction. Briefings in Bioinformatics 22, 6 (05 2021). bbab152.

[37] ZHENG, S., RAO, J., ZHANG, Z., XU, J., AND YANG, Y. Predicting retrosynthetic reactions using self-corrected transformer neural networks. Journal of Chemical Information and Modeling 60, 1 (2020), 47–55. PMID: 31825611.   
[38] ZHU, J., XIA, Y., QIN, T., ZHOU, W., LI, H., AND LIU, T.-Y. Dual-view molecule pre-training, 2021.   
[39] ŁUKASZ MAZIARKA, DANEL, T., MUCHA, S., RATAJ, K., TABOR, J., AND JASTRZ ˛EBSKI, S. Molecule attention transformer, 2020.   
[40] ŁUKASZ MAZIARKA, MAJCHROWSKI, D., DANEL, T., GAINSKI ´ , P., TABOR, J., PODOLAK, I., MORKISZ, P., AND JASTRZ ˛EBSKI, S. Relative molecule self-attention transformer, 2021.

# 7 Appendix: SMILES Graph Construction

To ensure the alignment between the SMILES token and the atoms in graph, we expand the original molecular graph $G _ { m o l }$ into the SMILES graph $G _ { s m i }$ by Algorithm 1. readSmiles(), getAtoms(), getNeighbors(), and writeSmiles() are functions supported in RDKit [1]. The tagging procedure is to inform the connection relationship between SMILES tokens. Rewriting the tagged SMILES without canonicalization is to ensure that the SMILES syntax does not change after the tagging.

Algorithm 1 SMILES Graph Construction   
Input: molecule canonical SMILES S
Initialize V as a token list of S.
Initialize E as an empty set.
Initialize A as an empty list.
M = readSmiles(S)
for $a_{i} \in getAtoms(M)$ do
    Assign tag #1 to the SMILES symbol of $a_{i}$ .
    for $a_{j} \in getNeighbors(M, a_{i})$ do
    Assign tag #2 to the SMILES symbol of $a_{j}$ .
    end for
    Get the tagged SMILES $S' = writeSmiles(M, canonical=False)$ Retrieve the token connections $e_{(\#1,\#2)}$ by the tagged $S'$ and add them to E.
    Retrieve the bond feature of $e_{(\#1,\#2)}$ and add them to A.
end for
Output: V, E, A

# 8 Appendix: SMILES Token Alignment Computation

The ground truth token alignment between the product SMILES and the reactants SMILES is computed as Algorithm 2. The algorithm takes the atom-mapped product and reactants as inputs. The computation works with both the canonical SMILES and the permuted SMILES.

Algorithm 2 SMILES Token Alignment Computation   
Input: atom-mapped product SMILES $S_{p}$ and atom-mapped reactants SMILES $S_{r}$ . Initialize the token mapping dictionary $r2s$ .

for $s_{r_i} \in S_r$ do
    if $s_{r_i}$ is not visited and $s_r$ is an atom token then
    Locate the token $s_{p_j}$ in $S_p$ with the same atom mapping number as $s_{r_i}: am(s_{p_j}) == am(s_{r_i})$ .
    while $s_{r_i} == s_{p_j}$ or $am(s_{p_j}) == am(s_{r_i})$ do
    Add alignment relationship $\{i:j\}$ into $r2s$ .
    Increment $i$ and $j$ .
    end while
    end if
end for
for $\{i:j\} \in r2s$ do
    Decrement $i$ and $j$ .
    while $s_{r_i} == s_{p_j}$ and $s_{r_i}$ is not an atom symbol do
    Add alignment relationship $\{i:j\}$ into $r2s$ .
    Decrement $i$ and $j$ .
    end while
end for
Output: $r2s$ .

![](images/a2aab3b76e33470ae8f713f398a2f79ff08226c18eecaa2555138f9f957b1d1b.jpg)

<details>
<summary>chemical</summary>

Five-step organic synthesis pathway showing brominated and fluorinated intermediates with corresponding molecular structures and reaction pathways
</details>

Figure 7: Visualization of the reaction center subgraph search algorithm: (1) Predict raw atom and bond reactive probability (i.e., reaction center detection); (2) Retrieve connected components based on the reactive probability; (3) Iterative pruning with maxBranch = 3; (4) Retrieve and rank all the candidate subgraphs (i.e. reaction centers); (5) Select the top-n diverse candidates from all subgraphs.

# 9 Appendix: Reaction Center Subgraph Search

Algorithm 3 shows the detailed search reaction center subgraph search algorithm, and Figure 7 shows a visualization of its procedure. In general, it searches for the candidate subgraphs (i.e., reaction centers) within the molecular graph via recursive pruning. It adopts a set of hyperparameters to avoid searching over the entire subgraph space.

The reaction center subgraph search is done in multiple stages: First, the algorithm removes all nodes whose $P _ { r c } ( s _ { i } ) <$ < $\alpha _ { a t o m }$ and edges whose $\bar { P } _ { r c } ( e _ { i j } ) < \alpha _ { b o n d } ,$ and retrieves all the connected components C from the edited graph. Second, for each connected component $c = ( V _ { c } , E _ { c } )$ , the algorithm retrieves all its subgraphs and the corresponding reactive scores via recursive pruning. The recursive pruning takes maxRootSize, minLeafSize, maxBranch as three arguments to control its search space. If the number of nodes $| V _ { c } |$ is larger than maxRootSize, then the algorithm directly removes $| V _ { c } | -$ maxRootSize number of nodes with the lowest reactive scores. At each iteration, the algorithm considers nodes that lie along the border of the current graph as pruning candidates. maxBranch is the maximum branching factor of the recursive pruning. The algorithm first ranks the pruning candidates by their atom reactive probability, and keeps only the top-maxBranch candidates for pruning. The recursion stops when $| V _ { c } | = m i n L e a f S i z e .$ . After all subgraphs are retrieved from a root connected component c, we rank them by their reactive scores. Prior to the overall subgraph ranking, we remove all subgraphs (excluding the top-1 subgraph) that share at least two common nodes with the top-1 subgraph to ensure the candidates’ diversity. At last, we gather the remaining subgraphs from all the root connected components C and retrieve the top-n reaction center candidates by their reactive scores.

In our experiments, we set $n = 3$ . Note that it only guarantees the maximum amount of reaction center candidates. We set the temperature $T = 1 0$ to flatten the reactive probabilities $P _ { r c } ( s )$ and $P _ { r c } ( e )$ . As for $\alpha _ { a t o m }$ and $\alpha _ { b o n d }$ , instead of having a fixed value, we dynamically set the two parameters as the $k _ { \mathrm { s } } ^ { t h }$ and $k _ { e } ^ { t h }$ percentile of $P _ { r c } ( s )$ and $P _ { r c } ( e )$ respectively. Based on the best validation performance, we set $k _ { s } = 4 0 , k _ { e } = 4 0$ for reaction class unknown setting, and $k _ { s } = 4 0 , k _ { e } = 5 5$ for reaction class known setting; $\beta$ is the parameter that controls the minLeaf Size. For simplicity reason, we set $\beta = 0 . 5$ , maxRootSize = 25, and $m a x B r a n c h = 5$ .

Algorithm 3 Reaction Center Subgraph Search   
Input: $P_{rc}(s)$ , $P_{rc}(e)$ , $G_{smi}$ , $\alpha_{atom}$ , $\alpha_{bond}$ , $\beta$ .

Remove all nodes whose $P_{rc}(s_i) < \alpha_{atom}$ from $G_{smi}$ .

Remove all edges whose $P_{rc}(e_{ij}) < \alpha_{bond}$ from $G_{smi}$ .

Retrieve all the connected components C from the edited SMILES graph.

for $c = (V_c, E_c) \in C$ do

    Set maxRootSize = 25.

    Set maxBranch = 5.

    Set minLeafSize = $\sum_{s_i \in c} \mathbb{1}(P_{rc}(s_i) > \beta)$ , where $s_i \in c$ .

    if $|V_c| > maxRootSize$ then

    Remove $|V_c| - maxRootSize$ nodes with the lowest $P_{rc}(s_i)$ .

    end if

    Retrieve all subgraphs of c (with scores) via recursive pruning with maxBranch and minLeafSize.

    Remove all subgraphs who share more than two common nodes with the top-1 subgraph.

end for

Output: Subgraphs with reactive scores.

The exact reactive score for a candidate subgraph $G = ( V , E )$ is computed as follow:

$$
\frac {1 + \varphi (| V | , \mu , \sigma^ {2})}{M} \left(\sum_ {s _ {i} \in V} \log P _ {r c} \left(s _ {i}\right) + \sum_ {e _ {i j} \in E} \log P _ {r c} \left(e _ {i j}\right)\right) \tag {8}
$$

where $M = \left| V \right| + \left| E \right|$ is the normalization factor, and $\varphi ( . )$ is the density function of a normal distribution of the size of reaction centers. We set $\mu = 5 . 5 5$ and $\sigma = 1 . 2$ , which are computed from the training dataset. This is a heuristic factor taken from the observation that the size of the reaction centers has little relationship with the size of the molecule, but is rather normally distributed (Figure 8).

![](images/770501f1a65f7b2c4804e45870b30ea29d360e1a2a81eda417b66fa3666c6ab4.jpg)

<details>
<summary>histogram</summary>

| Number of atoms in reaction center | Count |
| ---------------------------------- | ----- |
| 0-1                                | 0     |
| 1-2                                | 6600  |
| 2-3                                | 3800  |
| 3-4                                | 5600  |
| 4-5                                | 13600 |
| 5-6                                | 9000  |
| 6-7                                | 1000  |
| 7-8                                | 200   |
| 8-9                                | 100   |
| 9-10                               | 50    |
| 10-11                              | 20    |
| 11-12                              | 10    |
| 12-13                              | 5     |
| 13-14                              | 2     |
| 14-15                              | 1     |
| 15-16                              | 0     |
| 16-17                              | 0     |
| 17-18                              | 0     |
| 18-19                              | 0     |
| 19-20                              | 0     |
| 20-21                              | 0     |
| 21-22                              | 0     |
| 22-23                              | 0     |
| 23-24                              | 0     |
| 24-25                              | 0     |
</details>

Figure 8: Histogram of the size of reaction centers in the training set of USPTO-50K.

# 10 Appendix: Bond Features

Table 5 shows the bond features considered in the proposed Retroformer.

Table 5: Bond features. 

<table><tr><td>Bond Feature</td><td>Possible Values</td><td>Size</td></tr><tr><td>Bond Type</td><td>Single, aromatic, double, triple</td><td>4</td></tr><tr><td>Aromatic</td><td>True, false</td><td>1</td></tr><tr><td>Conjugated</td><td>True, false</td><td>1</td></tr><tr><td>Part of Ring</td><td>True, false</td><td>1</td></tr></table>