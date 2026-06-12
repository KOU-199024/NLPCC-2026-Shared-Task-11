# UNDERSTANDING TASK VECTORS IN IN-CONTEXT LEARNING: EMERGENCE, FUNCTIONALITY, AND LIM-ITATIONS

Yuxin Dong, Jiachen Jiang, Zhihui Zhu & Xia Ning

Department of Computer Science and Engineering, The Ohio State University

{dong.1357, jiang.2880, zhu.3440, ning.104}@osu.edu

# ABSTRACT

Task vector is a compelling mechanism for accelerating inference in in-context learning (ICL) by distilling task-specific information into a single, reusable representation. Despite their empirical success, the underlying principles governing their emergence and functionality remain unclear. This work proposes the Task Vectors as Representative Demonstrations conjecture, positing that task vectors encode single in-context demonstrations distilled from the original ones. We provide both theoretical and empirical support for this conjecture. First, we show that task vectors naturally emerge in linear transformers trained on triplet-formatted prompts through loss landscape analysis. Next, we predict the failure of task vectors in representing high-rank mappings and confirm this on practical LLMs. Our findings are further validated through saliency analyses and parameter visualization, suggesting an enhancement of task vectors by injecting multiple ones into few-shot prompts. Together, our results advance the understanding of task vectors and shed light on the mechanisms underlying ICL in transformer-based models.

# 1 INTRODUCTION

In-context learning (ICL) is a core capability of large language models (LLMs), allowing them to perform new tasks without parameter updates by conditioning on a few input-output examples in the prompt (Brown et al., 2020). Unlike traditional training, ICL relies on attention-based mechanisms to infer task structure directly from context. This surprising generalization ability has led to growing interest in uncovering the principles of learning purely from contextual examples (Xie et al., 2022; Chan et al., 2022; Dai et al., 2023; Shen et al., 2024; Deutch et al., 2024).

A recent work investigates the task vector method (Hendel et al., 2023) (concurrent works include function vectors (Todd et al., 2024) and in-context vectors (Liu et al., 2024)), a technique that distills underlying task information from ICL demonstrations into a single vector. Typically, ICL prompts are structured as sequences of triplets, each encoding a semantic mapping, in addition to a query at the end (e.g., “hot → cold, up → down, dark →”). Task vectors are then extracted from the hidden states of the last (→) token. Once obtained, these vectors can be injected into new zero-shot prompts $( \mathrm { e . g . , ~ } ^ { \left. \right.} b i g  ^ { \prime \prime } )$ , enabling the model to generalize to unseen inputs in a zero-shot fashion.

Task vectors naturally emerge even in small transformer models trained from scratch (Yang et al., 2025), suggesting that their formation is a general property of attention-based architectures. Recent studies further demonstrate that task vectors can be enhanced by aggregating hidden states across multiple layers and arrow tokens (Li et al., 2024). Beyond language models, task vectors are also effective in large-scale visual (Hojel et al., 2024) and multi-modal (Huang et al., 2024) models.

Despite their empirical effectiveness, the underlying mechanism of task vectors, especially how they emerge, function, and encode task information, remains poorly understood. This paper takes a step toward unveiling the principles behind it by introducing the following conjecture:

![](images/aa449f122d09278de8c472d211eb625ccf7aa674fd29576bdf8cc04f970afc85.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a)
        A1["Predict & Output"] --> B1["Task Vector Formation"]
        B1 --> C1["Preprocessing Stage"]
    end
    subgraph (b)
        D1["small"] --> E1["Weighted Summation"]
        E1 --> F1["big"]
    end
    B1 --> E1
    style A1 fill:#f9f,stroke:#333
    style B1 fill:#bbf,stroke:#333
    style C1 fill:#f9f,stroke:#333
    style E1 fill:#bbf,stroke:#333
    style F1 fill:#f9f,stroke:#333
```
</details>

Figure 1: Overview of task vector and our main conjecture. (a) Task vector emerges during ICL by distilling from the preceding in-context demonstrations. (b) It can then be injected into zero-shot prompts and functions as a single, representative demonstration, facilitating efficient prediction.

# Conjecture (Task Vectors as Representative Demonstrations)

The injected task vector facilitates zero-shot inference by encoding a single representative demonstration, distilled from the original in-context examples.

An intuitive illustration is provided in Figure 1. In the following sections, we validate this conjecture through various empirical and theoretical perspectives. These analyses comprehensively explain how task vectors naturally emerge within attention-based model architectures, effectively encode task-related information, and facilitate inference in zero-shot prompts. Our work advances the understanding of the underlying mechanisms behind ICL, clarifying both the efficacy and limitations of task vectors in transformer-based LLMs. The highlights of this paper are as follows:

• Theoretical Justification in Linear-Attention Models: We theoretically characterize the critical points of linear-attention models and demonstrate how they solve random linear regression tasks through embedding concatenation and gradient descent. With a triplet-formatted input prompt structure, task vectors naturally emerge at arrow tokens as weighted summations of the in-context demonstrations, potentially enhancing robustness under representational perturbations by redundantly encoding task information. Empirically, the learned linear model parameters closely align with the predicted structure and successfully replicate the task vector mechanism.   
• Empirical Verification in Practical LLMs: We visualize the information flow in LLMs with saliency analysis and observe patterns consistent with linear models, suggesting they share similar underlying mechanisms. According to our conjecture, inference with task vectors is analogous to 1-shot ICL, which is inherently limited to rank-one meta-predictors under the gradient descent perspective. To validate this, we introduce a series of bijection tasks that are provably unsolvable by rank-one predictors, and empirically confirm this failure in real-world transformers. Building on these insights, we enhance the standard task vector method by injecting multiple vectors into few-shot prompts, resulting in consistent performance gains across a range of ICL tasks.

# 1.1 RELATED WORKS

Theory of ICL. Recent analyses have shown that attention layers can simulate gradient-descent algorithms for regression tasks (Garg et al., 2022; Von Oswald et al., 2023a; Ahn et al., 2023; Wu et al., 2024). Other works study generalization and sample complexity (Xie et al., 2022; Chan et al., 2022; Shen et al., 2024; Von Oswald et al., 2023b; Deutch et al., 2024). These works reveal the inductive bias of attention but leave open how abstract task representations are formed or encoded.

Task Vector Mechanism. Multiple recent works identified the mechanism of task vectors during ICL inference (Hendel et al., 2023; Todd et al., 2024; Liu et al., 2024). These vectors emerge in the pretraining stage of LLMs (Yang et al., 2025) and extend beyond text to vision (Hojel et al., 2024) and multimodal (Huang et al., 2024) models. Despite the effectiveness, their underlying mechanism remains poorly understood. A concurrent work (Bu et al., 2025) interprets them via a word2vec-like additive scheme, but is limited to simple additive tasks, single-token prompts, and 1-layer models. In contrast, our analysis extends to pairwise or triplet prompts and multi-layer attention.

A more comprehensive discussion of the related works can be found in Appendix A.2.

# 2 SETTING: LINEAR REGRESSION WITH LINEAR-ATTENTION MODELS

Notations: We write $[ n ] = \{ 1 , \cdots , n \}$ . The Hadamard product is denoted by ◦, and the Kronecker product by ⊗. The identity matrix of dimension n is denoted by $I _ { n } ,$ , while $0 _ { n }$ and $0 _ { m \times n }$ represent zero vectors or matrices of the corresponding dimensions. Subscripts are omitted when the dimensions are clear from context. We define $\mathcal { M } ( \bar { M } ) = \big \{ \Lambda \in \mathbb { R } ^ { \dim ( M ) } \big \vert ^ { \cdot } \Lambda = M \circ A , A \in \mathbb { R } ^ { \dim ( M ) } \big \}$ as the set of masked matrices induced by mask M. For a general matrix $A ,$ the element at the i-th row and j-th column is denoted by $A _ { i , j }$ , and the sub-block from rows i to k and columns j to l is denoted by $A _ { i : k , j : l } . \deg ( A _ { 1 } , \cdot \cdot \cdot , A _ { n } )$ represents the block-diagonal matrix constructed by $\{ A _ { i } \} _ { i = 1 } ^ { n }$ .

Random Linear Regression: Following works (Garg et al., 2022; Von Oswald et al., 2023a; Ahn et al., 2023; Wu et al., 2024), we consider training linear transformers on random instances of linear regression. Let $\{ x _ { i } \} _ { i = 1 } ^ { n + 1 }$ , where $x _ { i } \in \mathbb { R } ^ { d }$ , denote covariates drawn i.i.d. from distribution $P _ { x }$ , and let $\{ w _ { i } \} _ { i = 1 } ^ { d }$ , where $\bar { w _ { i } } \in \mathbb { R } ^ { d }$ , denote coefficients drawn i.i.d. from distribution $P _ { w }$ . Define the coefficient matrix $W = [ w _ { 1 } \quad \cdot \cdot \cdot \quad w _ { d } ] ^ { \intercal } \in \mathbb { R } ^ { d \times d }$ . The responses are then generated as $y _ { i } = W x _ { i }$ for $i \in [ n + 1 ]$ . We denote by X, $Y \in \mathbb { R } ^ { d \times n }$ the matrices whose columns are $x _ { i }$ and $y _ { i }$ , respectively. The query covariate and response are denoted by $x _ { \mathrm { t e s t } } = x _ { n + 1 }$ and $y _ { \mathrm { t e s t } } = y _ { n + 1 }$ respectively.

Linear Self-Attention Model: Following prior works (Von Oswald et al., 2023a; Ahn et al., 2023; Wu et al., 2024), we consider transformers composed of linear self-attention layers. Let $Z _ { 0 } ~ \in$ $\mathbb { R } ^ { 2 d \times d _ { p } }$ denote the input matrix constructed from X, Y and $x _ { \mathrm { t e s t } }$ but excluding $y _ { \mathrm { t e s t } }$ , where $d _ { p }$ denotes the number of tokens and varies across prompt structures. The model is defined by stacking L attention blocks with skip connections, where the l-th layer is expressed as:

$$
Z _ {l} = Z _ {l - 1} + \frac {1}{n} \operatorname{Attn} _ {V _ {l}, Q _ {l}} (Z _ {l - 1}), \quad \operatorname{Attn} _ {V, Q} (Z) = V Z M \left(Z ^ {\top} Q Z\right). \tag {1}
$$

Here, the trainable parameters are $\{ V _ { l } , Q _ { l } \} _ { l = 1 } ^ { L }$ , where $V _ { l } \in \mathbb { R } ^ { 2 d \times 2 d }$ denotes the projection and value matrices, and $Q _ { l } \in \mathbb { R } ^ { 2 d \times 2 d }$ denotes the query and key matrices. Following the work (Ahn et al., 2023), we adopt a masking matrix $M = \bar { \mathrm { d i a g } ( } I _ { d _ { p } - 1 } , \bar { 0 } )$ to prevent attention from earlier tokens to the final one. The output of the model is defined as ${ \sf T F } \left( Z _ { 0 } ; \{ V _ { l } , Q _ { l } \} _ { l = 1 } ^ { L } \right) = \left( Z _ { L } \right) _ { ( d + 1 : 2 d ) , d _ { p } } ( { \bf i . e . } _ { - }$ the latter half of the last column). This definition aligns with the structure of the input $Z _ { 0 } ,$ , which will be further discussed in subsequent sections. During training, the parameters are optimized to minimize the expected ICL risk over random linear regression instances:

$$
\mathcal {L} \left(\left\{V _ {l}, Q _ {l} \right\} _ {l = 1} ^ {L}\right) = \mathbb {E} _ {Z _ {0}, W} \left\| \mathrm{TF} \left(Z _ {0}; \left\{V _ {l}, Q _ {l} \right\} _ {l = 1} ^ {L}\right) + W x _ {\text { test }} \right\| _ {2} ^ {2}. \tag {2}
$$

# 3 EMERGENCE OF TASK VECTORS IN LINEAR-ATTENTION MODELS

Firstly, we present theoretical evidence that task vectors naturally arise in simple linear transformers. Specifically, we analyze the loss landscape of the in-context risk, focusing on the properties of its critical points. As a startup, recall the standard linear regression setup (Ahn et al., 2023; Wu et al., 2024), where the $( x _ { i } , y _ { i } )$ pairs for each demonstration are concatenated to form the input prompt:

$$
Z _ {0} = \left[ \begin{array}{c c} X & x _ {\text {test}} \\ Y & 0 \end{array} \right] = \left[ \begin{array}{c c c c c} x _ {1} & x _ {2} & \dots & x _ {n} & x _ {\text {test}} \\ y _ {1} & y _ {2} & \dots & y _ {n} & 0 \end{array} \right] \in \mathbb {R} ^ {2 d \times d _ {p}}, \quad d _ {p} = n + 1. \tag {3}
$$

According to existing analyses (Ahn et al., 2023; Zhang et al., 2024; Mahankali et al., 2024), each attention layer in this setting performs one step of gradient descent on the learned coefficient matrix. Specifically, the theoretically optimal single-layer (possibly nonlinear) attention (Katharopoulos et al., 2020) implements the following predictive function (Ahn et al., 2023) when the covariates are drawn from $\bar { P _ { x } } = \mathcal { N } ( 0 , I _ { d } )$ , by selecting $V _ { 1 } \propto \mathrm { d i a g } ( 0 _ { d \times d } , I _ { d } )$ and $Q _ { 1 } \propto \mathrm { d i a g } ( I _ { d } , 0 _ { d \times d } )$ :

$$
\mathsf {T F} (Z _ {0}; (V _ {1}, Q _ {1})) = - \frac {1}{n} Y \sigma (X) ^ {\top} \sigma (x _ {\text { test }}), \quad \text { where   } \sigma : \mathbb {R} ^ {d} \mapsto \mathbb {R} ^ {r} \text {   is   a   kernel   function. } \tag {4}
$$

Here, we abbreviate $\left[ \sigma ( x _ { 1 } ) \quad \cdots \quad \sigma ( x _ { n } ) \right]$ as σ(X). This model employs $W ^ { \prime } \propto Y \sigma ( X ) ^ { \top }$ as an estimate of W , yielding prediction $\hat { y } _ { \mathrm { t e s t } } = W ^ { \prime } \sigma ( x _ { \mathrm { t e s t } } )$ . This paper considers alternative settings more reflective of practical scenarios, where $x _ { i }$ and $y _ { i }$ are separated as distinct tokens. As noted (Zuo et al., 2025), such separation necessitates the usage of position encodings for bi-directional attention. Following prior analysis (Kazemnejad et al., 2023), we assume that position encodings are appended to the input tokens, and reformulate the layer-wise update rule of self-attention as:

$$
\operatorname{Attn} _ {V, Q} (Z) = V Z M \left[ \begin{array}{l l} Z ^ {\top} & P ^ {\top} \end{array} \right] Q \left[ \begin{array}{l} Z \\ P \end{array} \right], \quad \text { where } P \in \mathbb {R} ^ {d _ {p} \times d _ {p}}. \tag {5}
$$

For analytical tractability, we take $P = I _ { d _ { v } }$ as one-hot position encodings. Following previous work (Ahn et al., 2023) (see Appendix A.3 for more explanation), we further impose that:

$$
V _ {l} = \operatorname{diag} \left(A _ {l}, B _ {l}\right), \quad Q _ {l} = \operatorname{diag} \left(C _ {l}, 0 _ {d \times d}, D _ {l}\right), \quad \text { where } A _ {l}, B _ {l}, C _ {l} \in \mathbb {R} ^ {d \times d}, D _ {l} \in \mathbb {R} ^ {d _ {p} \times d _ {p}}. \tag {6}
$$

These parameterizations ensure that the projection and attention operations act independently on the covariate, response, and positional components of the input. This structural decoupling is essential for understanding how the transformer identifies the dependency between each $( x _ { i } , y _ { i } )$ pair and revealing the actual optimization algorithm being executed by the model. The proofs for the main theoretical results in this paper are available in Appendix D.

# 3.1 WARM-UP: LEARNING WITH PAIRWISE DEMONSTRATIONS

We begin by analyzing the optimization of linear transformers on pairwise demonstrations. Following previous approach (Garg et al., 2022; Wibisono & Wang, 2023; Xing et al., 2024), we decompose each demonstration in eq. (3) into a pair of tokens $Z _ { 0 } ^ { i } = \bar { \left[ \begin{array} { l l } { { x _ { i } } } & { { 0 } } \\ { { 0 } } & { { y _ { i } } } \end{array} \right] } \in \mathbb { R } ^ { 2 \breve { d } \times 2 }$ to better reflect the practical ICL prompt structure:

$$
Z _ {0} = \left[ \begin{array}{c c c c c c c} Z _ {0} ^ {1} & \dots & Z _ {0} ^ {n} & Z _ {0} ^ {\text {test}} \end{array} \right] = \left[ \begin{array}{c c c c c c c} x _ {1} & 0 & \dots & x _ {n} & 0 & x _ {\text {test}} & 0 \\ 0 & y _ {1} & \dots & 0 & y _ {n} & 0 & 0 \end{array} \right], \quad d _ {p} = 2 n + 2. \tag {7}
$$

The following theorem suggests that certain critical points of the in-context risk effectively solve the regression problem by first concatenating each pair of $( x _ { i } , y _ { i } )$ into the same tokens, and then executing a variant of the gradient descent algorithm to compute the prediction. To simplify notation, we denote $A = \{ A _ { l } \} _ { l = 1 } ^ { L }$ (similarly for $B , { \bar { C } } ,$ and D) and present:

Theorem 1 (Critical Points; Pairwise Demonstrations). Assume that $P _ { x } = \mathcal { N } ( 0 , \Sigma )$ and $P _ { w } =$ $\mathcal { N } ( 0 , \Sigma ^ { - 1 } )$ with $\Sigma \in \mathbb { R } ^ { d \times d }$ satisfying $\Sigma \succ 0$ . Define $S _ { I } , S _ { \Sigma } \subset \mathbb { R } ^ { d \times d }$ and $S _ { P } \subset \mathbb { R } ^ { d _ { p } \times d _ { p } }$ as

$$
\mathcal {S} _ {I} = \left\{\lambda I _ {d} \mid \lambda \in \mathbb {R} \right\}, \quad \mathcal {S} _ {\Sigma} = \left\{\lambda \Sigma^ {- 1} \mid \lambda \in \mathbb {R} \right\}, \quad \mathcal {S} _ {P} = \left\{\operatorname{diag} (I _ {n} \otimes \Lambda_ {1}, \Lambda_ {2}) \mid \Lambda_ {1}, \Lambda_ {2} \in \mathbb {R} ^ {2 \times 2} \right\}.
$$

Consider optimizing an L-layer transformer under parameter configuration in eq. (6), we have

$$
\inf _ {A, B \in \mathcal {S} _ {I} ^ {L}, C \in \mathcal {S} _ {\Sigma} ^ {L}, D \in \mathcal {S} _ {P} ^ {L}} \sum_ {H \in A \cup B \cup C \cup D} \left\| \nabla_ {H} \mathcal {L} \big (\{V _ {l}, Q _ {l} \} _ {l = 1} ^ {L} \big) \right\| _ {F} ^ {2} = 0.
$$

To understand the behavior of these critical points within a self-attention layer, we fix $\Sigma = I _ { d }$ and take $A _ { l } , B _ { l } = I _ { d } , C _ { l } = - \lambda I _ { d }$ , and $D _ { l } = \mathrm { d i a g } ( I _ { n } \otimes \Lambda _ { 1 } , \Lambda _ { 2 } )$ . Let the first and last d rows of $Z _ { l }$ be denoted by $X _ { l }$ and $Y _ { l } ,$ , respectively. Under these settings, the update rule of each layer becomes:

$$
Z _ {l} = Z _ {l - 1} - \lambda Z _ {l - 1} M X _ {l - 1} ^ {\top} X _ {l - 1} + \left[ Z _ {l - 1} ^ {1} \Lambda_ {1} \quad \dots \quad Z _ {l - 1} ^ {n} \Lambda_ {1} \quad Z _ {l - 1} ^ {\text { test }} \operatorname{diag} (1, 0) \Lambda_ {2} \right]. \tag {8}
$$

The above update can be decomposed into the following two distinct components:

• Gradient Descent: The first component, $Z _ { l } \gets Z _ { l - 1 } - \lambda Z _ { l - 1 } M X _ { l - 1 } ^ { \top } X _ { l - 1 }$ , implements the GD++ algorithm (Von Oswald et al., 2023a). This variant enhances convergence speed over standard gradient descent by improving the condition number of $X _ { l - 1 } ^ { \top } X _ { l - 1 }$ . Notably, this operation modifies only $X _ { l }$ but not $Y _ { l }$ for the first layer, as implied by the structure of $Q _ { l } \ ( \mathsf { e q . } \left( 6 \right) )$ .   
• Embedding Concatenation: The second component, $Z _ { l } ^ { i } \gets Z _ { l - 1 } ^ { i } + Z _ { l - 1 } ^ { i } \Lambda _ { 1 }$ for $i \in [ n ]$ , mixes each pair of $( x _ { i } , y _ { i } )$ tokens. Given that $x _ { i }$ and $y _ { i }$ tokens are initially linearly separable as in our formulation, this operation concatenates each $( x _ { i } , y _ { i } )$ pair, thereby transforming pairwise demonstrations into the original single-token format. For the query token $Z _ { l } ^ { \mathrm { t e s t } }$ , this operation copies $x _ { \mathrm { t e s t } }$ into the final token, reconstructing the structure in eq. (3), where each non-final token directly concatenates $( x _ { i } , y _ { i } )$ of a demonstration, and the final token contains only $x _ { \mathrm { t e s t } }$ .

In summary, our analysis reveals that for pairwise demonstrations, the first attention layer leverages position encodings to distinguish between covariate and response tokens, subsequently concatenating them to form a single-token prompt structure. The remaining layers then apply the GD++ algorithm, mirroring the learning dynamics on single-token demonstrations. As a result, an L-layer linear transformer allocates one layer for embedding concatenation and utilizes the remaining $L - 1$ layers to perform gradient descent. In Figure 2a, we visualize the learned $D _ { l }$ weights under the setting of Theorem 1, and observe that they closely match the critical point structure of $S _ { P }$ .

![](images/48725f7de5db83fb510e2fd543cbc8384eb891849e604f332d8d3d7c1c0c458c.jpg)  
(a) Dl (Pairwise)

![](images/485024149a1285d4a2cfa7156ef88a11662a1d5072ba31b0394c3d50389359f5.jpg)

<details>
<summary>heatmap</summary>

| | -1 | 0 | 1 | 2 |
|---|---|---|---|---|
| Row 1 | -2 | -1 | 0 | 1 |
| Row 2 | -1 | 0 | 1 | 2 |
| Row 3 | 0 | 1 | 2 | 1 |
| Row 4 | 1 | 2 | 1 | 0 |
| Row 5 | 2 | 1 | 0 | -1 |
| Row 6 | 1 | 0 | -1 | -2 |
| Row 7 | 0 | -1 | -2 | -1 |
| Row 8 | -1 | -2 | -1 | -2 |
| Row 9 | -2 | -1 | 0 | -2 |
| Row 10 | -1 | 0 | 1 | -2 |
| Row 11 | 0 | 1 | 2 | -1 |
| Row 12 | 1 | 2 | 1 | -0 |
| Row 13 | 2 | 1 | 0 | -1 |
| Row 14 | 1 | 0 | -1 | -2 |
| Row 15 | 0 | -1 | -2 | -1 |
| Row 16 | -1 | -2 | -1 | -2 |
| Row 17 | -2 | -1 | 0 | -2 |
| Row 18 | -1 | 0 | -1 | -2 |
| Row 19 | 0 | 1 | 0 | -2 |
| Row 20 | 1 | 2 | 1 | -1 |
| Row 21 | 2 | 1 | 0 | -0 |
| Row 22 | 1 | 0 | -1 | -1 |
| Row 23 | 0 | -1 | -2 | -2 |
| Row 24 | -1 | -2 | -1 | -2 |
| Row 25 | -2 | -1 | 0 | -2 |
| Row 26 | -1 | 0 | -1 | -2 |
| Row 27 | 0 | 1 | 0 | -2 |
| Row 28 | 1 | 2 | 1 | -1 |
| Row 29 | 2 | 1 | 0 | -0 |
| Row 30 | 1 | 0 | -1 | -1 |
| Row 31 | 0 | -1 | -2 | -2 |
| Row 32 | -1 | -2 | -1 | -2 |
| Row 33 | -2 | -1 | 0 | -2 |
| Row 34 | -1 | 0 | -1 | -2 |
| Row 35 | 0 | 1 | 0 | -2 |
| Row 36 | 1 | 2 | 1 | -1 |
| Row 37 | 2 | 1 | 0 | -0 |
| Row 38 | 1 | 0 | -1 | -1 |
| Row 39 | 0 | -1 | -2 | -2 |
| Row 40 | -1 | -2 | -1 | -2 |
| Row 41 | -2 | -1 | 0 | -2 |
| Row 42 | -1 | 0 | -1 | -2 |
| Row 43 | 0 | 1 | 0 | -2 |
| Row 44 | 1 | 2 | 1 | -1 |
| Row 45 | 2 | 1 | 0 | -0 |
| Row 46 | 1 | 0 | -1 | -1 |
| Row 47 | 0 | -1 | -2 | -2 |
| Row 48 | -1 | -2 | -1 | -2 |
| Row 49 | -2 | -1 | 0 | -2 |
| Row 50+<fcel>-1<fcel>-2<fcel>-1<fcel>-2<fcel>-2<nl>
</details>

(b) Dl (Triplet)

![](images/934f2d49e7682276bcc3ec12d848246606ad8009c8744f71763ee43f8af9d7b5.jpg)

<details>
<summary>heatmap</summary>

| X  | Y  | Value |
|----|----|-------|
| 1  | 1  | -0.2  |
| 1  | 2  | -0.1  |
| 1  | 3  | 0.0   |
| 1  | 4  | 0.1   |
| 1  | 5  | 0.2   |
| 1  | 6  | 0.3   |
| 2  | 1  | -0.2  |
| 2  | 2  | -0.1  |
| 2  | 3  | 0.0   |
| 2  | 4  | 0.1   |
| 2  | 5  | 0.2   |
| 2  | 6  | 0.3   |
| 2  | 7  | -0.2  |
| 2  | 8  | -0.1  |
| 2  | 9  | 0.0   |
| 2  | 10 | 0.1   |
| 2  | 11 | 0.2   |
| 2  | 12 | 0.3   |
| 3  | 1  | -0.2  |
| 3  | 2  | -0.1  |
| 3  | 3  | 0.0   |
| 3  | 4  | 0.1   |
| 3  | 5  | 0.2   |
| 3  | 6  | 0.3   |
| 3  | 7  | -0.2  |
| 3  | 8  | -0.1  |
| 3  | 9  | 0.0   |
| 3  | 10 | 0.1   |
| 3  | 11 | 0.2   |
| 3  | 12 | 0.3   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| ...| ...| ...   |
| Note: The actual values in the 'Value' column are not provided in the code, so they are represented as placeholders (e.g., '0.3' or '0.2').
</details>

(c) $\Lambda _ { 4 } \Lambda _ { 4 } ^ { \top }$   
Figure 2: Visualization of learned $D _ { l }$ weights. (a) Pairwise demonstrations yield a block-diagonal structure aligned with Theorem 1. (b) Triplet demonstrations yield a richer structure aligned with Theorem 2. (c) The learned matrix $\Lambda _ { 4 }$ has nearly orthonormal rows as suggested by Proposition 3.

# 3.2 EMERGENCE OF TASK VECTORS WITH TRIPLET DEMONSTRATIONS

Next, to better reflect the prompt structure of practical ICL, we insert additional zero tokens between each pair of $( x _ { i } , y _ { i } )$ to simulate the arrow (→) tokens. This reformulates each demonstration as a triplet $( x _ { i } ,  , y _ { i } )$ ), enabling us to analyze the critical points with these triplet demonstrations:

$$
Z _ {0} = \left[ \begin{array}{c c c c c c c c c c} x _ {1} & 0 & 0 & \dots & x _ {n} & 0 & 0 & x _ {\text {test}} & 0 & 0 \\ 0 & 0 & y _ {1} & \dots & 0 & 0 & y _ {n} & 0 & 0 & 0 \end{array} \right], \quad d _ {p} = 3 n + 3. \tag {9}
$$

Theorem 2 (Critical Points; Triplet Demonstrations). Assume that $P _ { x } = \mathcal { N } ( 0 , \Sigma )$ and $P _ { w } =$ $\mathcal { N } ( 0 , \Sigma ^ { - 1 } )$ with $\Sigma \in \mathbb { R } ^ { d \times d }$ satisfying Σ ≻ 0. Define $S _ { I } , S _ { \Sigma } \subset \mathbb { R } ^ { d \times d }$ and $S _ { P } \subset \mathbb { R } ^ { \dot { d } _ { p } \times d _ { p } ^ { \prime } }$ as

$$
\mathcal {S} _ {I} = \bigl \{\lambda I _ {d} \mid \lambda \in \mathbb {R} \bigr \}, \quad \mathcal {S} _ {\Sigma} = \bigl \{\lambda \Sigma^ {- 1} \mid \lambda \in \mathbb {R} \bigr \},
$$

$$
\mathcal {S} _ {P} = \left\{\operatorname{diag} (I _ {n} \otimes \Lambda_ {1}, \Lambda_ {2}) + I _ {n + 1} \otimes \Lambda_ {3} + \Lambda_ {4} \otimes \Lambda_ {5} \right|
$$

$$
\Lambda_ {1}, \Lambda_ {2} \in \mathcal {M} \Big ( \begin{array}{c c c} 1 & 0 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 1 \end{array} \Big), \Lambda_ {3} \in \mathcal {M} \Big ( \begin{array}{c c c} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{array} \Big), \Lambda_ {4} \in \mathbb {R} ^ {(n + 1) \times (n + 1)}, \Lambda_ {5} \in \mathcal {M} \Big ( \begin{array}{c c c} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 1 & 0 \end{array} \Big) \Big \}.
$$

Consider optimizing an L-layer transformer under parameter configuration in eq. (6), we have

$$
\inf _ {A, B \in \mathcal {S} _ {I} ^ {L}, C \in \mathcal {S} _ {\Sigma} ^ {L}, D \in \mathcal {S} _ {P} ^ {L}} \sum_ {H \in A \cup B \cup C \cup D} \left\| \nabla_ {H} \mathcal {L} \big (\{V _ {l}, Q _ {l} \} _ {l = 1} ^ {L} \big) \right\| _ {F} ^ {2} = 0.
$$

To analyze the behavior of each attention layer, we note that the critical points for the matrices $A _ { l } ,$ $B _ { l }$ , and $C _ { l }$ remain consistent with Theorem 1, thereby implementing the GD++ algorithm. For the matrix $D _ { l } ,$ , we decompose its structure into three distinct components:

• Embedding Concatenation: The first component, diag ${ { I } _ { n } } \otimes { { \Lambda } _ { 1 } } , { { \Lambda } _ { 2 } } )$ , mixes each pair of $( x _ { i } , y _ { i } )$ tokens, effectively concatenating them — analogous to the operation analyzed in the previous section. This converts all non-arrow tokens into single-token demonstrations.   
• Self Magnification: The second component, ${ \cal I } _ { n + 1 } \otimes \Lambda _ { 3 }$ , scales the embeddings corresponding to each arrow (→) token by a fixed constant and adds them back to themselves.   
• Task Vector Formation: The third component, $\Lambda _ { 4 } \otimes \Lambda _ { 5 }$ , performs a weighted summation across all demonstrations in the prompt. This operation is central to the emergence of task vectors. Let $[ \beta _ { 1 } \mathrm { ~  ~ \beta ~ } \cdot \cdot \cdot \mathrm { ~  ~ \beta ~ } \beta _ { n + 1 } ] \in \mathbb { R } ^ { n \times ( n + 1 ) }$ denote the first n rows of $\Lambda _ { 4 }$ (we will soon show that the last row of $\Lambda _ { 4 }$ converges to zero), the first self-attention layer then outputs $n + 1$ linear combinations of the demonstrations as the hidden states for the arrow tokens, expressed as $\begin{array} { r } { z _ { \mathrm { t v } } ^ { i } = \big [ { \alpha _ { 1 } } X { \beta _ { i } } } \\ { \alpha _ { 2 } Y { \beta _ { i } } } \end{array}$ for $i \in [ n + 1 ]$ , where $\alpha _ { 1 } , \alpha _ { 2 } \in \mathbb { R }$ are the two non-zero entries of $\Lambda _ { 5 }$ . These vectors can then be injected into zero-shot prompts and function as single-token demonstrations.

This mechanism provides strong theoretical evidence for our main conjecture, demonstrating that task vectors naturally emerge from the pretraining stage of linear-attention transformers on triplet-formatted prompts. Notably, the structure of $ { \boldsymbol { S } } _ { P }$ closely aligns with our visualization of $D _ { l }$ in Figure 2b, confirming our theoretical analysis. We now further investigate the structure of the weight matrix $\Lambda _ { 4 }$ , and present the following result:

![](images/49efb2874e409daf917d082d46ccb5e220a656bc45a83cf6a611407fe6aa4ccc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    x1["x₁"] --> y1["y₁"]
    y1 --> x2["x₂"]
    x2 --> y2["y₂"]
    y2 --> x3["x₃"]
    x3 --> y3["y₃"]
    y3 --> xtest["x_test"]
    x1 --> y1
    y1 --> x2
    x2 --> y2
    y2 --> x3
    x3 --> y3
    y3 --> xtest
    xtest --> y3
    style x1 fill:#000,stroke:#000,color:#fff
    style y1 fill:#000,stroke:#000,color:#fff
    style x2 fill:#000,stroke:#000,color:#fff
    style y2 fill:#000,stroke:#000,color:#fff
    style x3 fill:#000,stroke:#000,color:#fff
    style y3 fill:#000,stroke:#000,color:#fff
    style xtest fill:#000,stroke:#000,color:#fff
```
</details>

(a) Saliency Map (l = 10)

![](images/bf4042a6bc256a320d6a9edca794698ca0ac379c6bc9cc61b4065cb01d58867d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["x1"] --> B["y1"]
    A --> C["x2"]
    A --> D["y2"]
    A --> E["x3"]
    A --> F["y3"]
    A --> G["xtest"]
    H["x1"] --> I["y1"]
    H --> J["x2"]
    H --> K["y2"]
    H --> L["x3"]
    H --> M["y3"]
    H --> N["xtest"]
    O["x1"] --> P["y1"]
    O --> Q["x2"]
    O --> R["y2"]
    O --> S["x3"]
    O --> T["y3"]
    O --> U["xtest"]
    V["Output Node"] --> W["Output Node"]
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
    style V fill:#ccf,stroke:#333
    style W fill:#ccf,stroke:#333
    style X fill:#ccf,stroke:#333
```
</details>

(b) Saliency Map (l = 12)

![](images/e96d239db5f8f892d538b115d3bd5902baa9b1827234a353ae7de8be447a3d01.jpg)  
(c) Task Vector Weights   
Figure 3: Visualizations on Llama-7B: (1) saliency matrices as bipartite graphs between layer l ( ) and $l + 1 \left( \bullet \right)$ , edge widths indicate saliency magnitude; (2) variations in the extracted task vector after perturbing the i-th demonstration ( ) and the optimal task vector weights ( ) obtained by optimizing Proposition 5. (a) Each $y _ { i }$ token attends to its corresponding $( x _ { i } , y _ { i } )$ pair, reflecting embedding concatenation. (b) The final ( ) token attends broadly to all $y _ { i }$ tokens, indicating task vector formation. This occurs just before the optimal injection layer $( l = 1 3 )$ . (c) The predicted task vector weights closely match the trend of empirical results, validating our theoretical model.

Proposition 3 (Optimal Task Vector Weights). Assume $P _ { x } , P _ { w } = \mathcal { N } ( 0 , I _ { d } )$ . Consider optimizing a 2-layer linear-attention transformer with triplet demonstrations and parameter configuration given in eq. (6), and assume $C _ { 1 } = 0 _ { d \times d }$ . Let

$$
D _ {1} = \operatorname{diag} (I _ {n} \otimes \Lambda_ {1}, \Lambda_ {2}) + I _ {n + 1} \otimes \Lambda_ {3} + \Lambda_ {4} \otimes \Lambda_ {5} \in \mathcal {S} _ {P}
$$

be any minimizer of the in-context risk $\mathcal { L } \big ( \{ V _ { l } , Q _ { l } \} _ { l = 1 } ^ { L } \big )$ , we then have $\Lambda _ { 4 } \in S _ { U }$ , where

$$
\mathcal {S} _ {U} = \left\{\Lambda \mid \Lambda \Lambda^ {\top} = \lambda \operatorname{diag} (I _ {n}, 0), \lambda \in \mathbb {R} \right\}.
$$

This result suggests that the optimal $\Lambda _ { 4 }$ weight matrix satisfies two key properties: (1) the last row is zero, and (2) the first n rows are mutually orthonormal. These conditions imply that the learned weight vectors $\beta _ { 1 } , \cdots , \beta _ { n + 1 }$ 1 are likely to be distinct. Therefore, the $n + 1$ task vectors produce diverse linear combinations of the demonstrations, thereby enriching the representation within the input prompt. This implication is verified in Figure 2c. While task vectors are typically extracted from the final arrow (→) token in standard usage, here we consider all arrow tokens as task vectors as bi-directional attention allows each to aggregate information from the full prompt.

# 4 PREDICTED FAILURE OF TASK VECTORS ON BIJECTION TASKS

We then present an empirical observation that supports our conjecture. Consider the setting where task vectors are injected into zero-shot prompts. Based on our prior analysis, the injected task vector $z _ { \mathrm { t v } }$ is formed as a weighted summation of the original demonstrations. As a result, we show that the injected prompt reconstructs the single-token structure in eq. (3) with only 1 demonstration:

$$
Z _ {0} = \left[ \begin{array}{c c c} z _ {\text {test}} & z _ {\mathrm{tv}} & 0 \end{array} \right] = \left[ \begin{array}{c c c} x _ {\text {test}} & x _ {\mathrm{tv}} & 0 \\ 0 & y _ {\mathrm{tv}} & 0 \end{array} \right] = \left[ \begin{array}{c c c} x _ {\text {test}} & \alpha_ {1} X \beta & 0 \\ 0 & \alpha_ {2} Y \beta & 0 \end{array} \right] \in \mathbb {R} ^ {2 d \times 3}, \tag {10}
$$

where the weight vector $\beta \in \mathbb { R } ^ { n }$ comes from the last column of $\Lambda _ { 4 } ,$ and the weights $\alpha _ { 1 } , \alpha _ { 2 }$ come from $\Lambda _ { 5 }$ (see our discussion after Theorem 2). After the first layer, the $\Lambda _ { 2 }$ matrix of $S _ { P }$ moves $x _ { \mathrm { t e s t } }$ to the last token, reducing the prompt to a single-shot, single-token demonstration. According to the optimal single-layer transformer (eq. (4)), the estimated coefficient matrix is now $W ^ { \prime } = \bar { \alpha } _ { 1 } \alpha _ { 2 } Y \beta \bar { ( } X \beta ) ^ { \top }$ , which is rank-one. Therefore, task vectors are inherently limited in their expressiveness: they can only replicate 1-shot ICL, which is restricted to rank-one coefficient matrices. This implication also naturally extends to multi-layer transformers.

While our analysis is conducted on linear-attention transformers, we demonstrate that similar learning patterns also emerge within practical LLMs. Specifically, we visualize the layer-wise information flow between tokens using saliency maps (Wang et al., 2023), where the saliency score for each attention matrix is computed as $\begin{array} { r } { S ( A _ { l } ) ^ { * } = \dot { \sum } _ { h } | A _ { l , h } \cdot \partial \mathcal { L } / \partial A _ { l , h } | , A _ { l , h } } \end{array}$ denotes the attention matrix of the h-th head at layer $l ,$ and L is the ICL loss (i.e., the cross-entropy loss for predicting $y _ { \mathrm { t e s t } } )$ .

Table 1: Comparison of the accuracies of many-shot ICL and task vector on bijection tasks (Llama-7B, n = 10). We use gray text to indicate accuracies lower than 60%. 

<table><tr><td rowspan="2">Task</td><td rowspan="2">Domain X</td><td rowspan="2">Domain Y</td><td rowspan="2">Example</td><td colspan="2">X→Y</td><td colspan="2">Y→X</td><td colspan="2">X↔Y</td></tr><tr><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td></tr><tr><td>To Upper</td><td>{a,···,z}</td><td>{A,···,Z}</td><td>a → A</td><td>1.00</td><td>0.91</td><td>1.00</td><td>0.99</td><td>1.00</td><td>0.55</td></tr><tr><td rowspan="3">Translation</td><td>English</td><td>French</td><td>hello → bonjour</td><td>0.83</td><td>0.84</td><td>0.82</td><td>0.70</td><td>0.54</td><td>0.35</td></tr><tr><td>English</td><td>Italian</td><td>hello → ciao</td><td>0.84</td><td>0.78</td><td>0.82</td><td>0.74</td><td>0.70</td><td>0.47</td></tr><tr><td>English</td><td>Spanish</td><td>hello → hola</td><td>0.92</td><td>0.88</td><td>0.89</td><td>0.75</td><td>0.64</td><td>0.43</td></tr><tr><td rowspan="4">Linguistic</td><td>Present</td><td>Gerund</td><td>go → going</td><td>0.99</td><td>0.95</td><td>1.00</td><td>0.97</td><td>0.80</td><td>0.41</td></tr><tr><td>Present</td><td>Past</td><td>go → went</td><td>0.98</td><td>0.91</td><td>0.99</td><td>0.96</td><td>0.52</td><td>0.33</td></tr><tr><td>Present</td><td>Past Perfect</td><td>go → gone</td><td>0.82</td><td>0.82</td><td>0.94</td><td>0.65</td><td>0.55</td><td>0.33</td></tr><tr><td>Singular</td><td>Plural</td><td>dog → dogs</td><td>0.88</td><td>0.78</td><td>0.94</td><td>0.89</td><td>0.76</td><td>0.51</td></tr><tr><td>Copy</td><td colspan="2">{a,···,z,A,···,Z}</td><td>A → A</td><td colspan="2">-</td><td colspan="2">-</td><td>1.00</td><td>0.98</td></tr><tr><td>Antonym</td><td colspan="2">Adjectives</td><td>happy → sad</td><td>0.89</td><td>0.83</td><td colspan="2">-</td><td>0.83</td><td>0.73</td></tr></table>

As demonstrated in Figures 3a and 3b, the saliency maps reveal certain patterns matching the ones of embedding concatenation and weighted summation. This suggests that real-world transformers implement a similar algorithm to solve ICL tasks and, consequently, inherit the same expressiveness limitation. The full saliency score maps are given in Appendix B.5.

To verify this, we construct a specialized class of ICL tasks, named bijection tasks. Specifically, given a bijective mapping from domain X to codomain Y, one can combine it with its inverse mapping to form a new task that maps $\mathcal { X } \cup \mathcal { V }$ onto itself. For instance, combining the “to uppercase” task with its inverse ”to lowercase” yields a bijection task that maps each letter to its opposite case, and a valid ICL prompt takes the form: ${ } ^ { \cdots } a  A , B  b , c  C , D  { } ^ { \cdots }$ . Note that this differs from task superposition (Xiong et al., 2024), as each input corresponds to a unique, well-defined output. We then establish a key limitation of rank-one coefficient matrices in addressing such tasks:

Proposition 4. Let $x , y \in \mathbb { R } ^ { d }$ be non-zero vectors. Then the following are equivalent: (1) There exists a rank-one matrix $W \in \mathbb { R } ^ { d \times d }$ such that $y = W x$ and x = W y; (2) x = y or x = −y.

This result highlights that rank-one coefficient matrices cannot solve general bijection tasks, and are restricted to two special cases: the identity mapping $( x = y )$ , or the negation mapping $( x =$ −y). We further verify this implication in real-world LLMs: in Table 1, both ICL and task vectors perform well on the original tasks and their inverses. But for bijection tasks, while ICL preserves performance in many cases, the task vector method consistently fails, confusing examples from the two domains and yielding near-random predictions (50%) (e.g., in “To Upper”, task vectors predict the correct letter but fail to distinguish between uppercase and lowercase. See Appendix B.4 for further results). The only exceptions are Copy and Antonym, the special cases in Proposition 4.

Together, these findings empirically validate our main conjecture: the task vector approach, which is restricted to one-shot ICL, is limited to rank-one mappings and cannot solve general ICL tasks (e.g., bijection tasks). While a variety of ICL tasks have been explored to assess the capabilities of task vectors (Hendel et al., 2023; Todd et al., 2024; Li et al., 2024), the fundamental limitation of task vectors in addressing these bijection tasks has not been previously identified.

# 5 FURTHER DISCUSSIONS

Effect of Causal Attention and Dropout. While task vectors naturally emerge in linear attention, their embeddings do not directly help minimize the ICL risk, as evidenced by the identical performance between pairwise and triplet formatted prompts (Figures 4a and 4b). Instead, we show that task vectors do contribute to optimization under token-wise dropout, acting as redundancies for incontext demonstrations that may be randomly dropped during training. This redundancy ensures that essential task information is preserved to facilitate inference despite partial context loss.

Proposition 5. Under the same settings as Proposition 3, consider adding token-wise dropouts $O _ { l } .$

$$
Z _ {l} = Z _ {l - 1} O _ {l} + \frac {1}{n} \operatorname{Attn} _ {V _ {l}, Q _ {l}} (Z _ {l - 1}) O _ {l}, \quad w h e r e O _ {l} = \mathrm{diag} (o _ {l} ^ {1}, \dots , o _ {l} ^ {d _ {p}}), o _ {l} ^ {i} \stackrel {i. i. d.} {\sim} \mathrm{Bern} (p).
$$

![](images/3812eb0a916bf5e0ceddad8d121123ee7152c91e2c3b51b5b96b732c9300f11b.jpg)

<details>
<summary>bar</summary>

| n   | S (L = 1) | P (L = 2) | T (L = 2) | S (L = 2) |
| --- | --------- | --------- | --------- | --------- |
| 5   | 2.0       | 1.5       | 1.8       | 1.0       |
| 10  | 1.5       | 1.0       | 0.8       | 0.3       |
| 15  | 1.2       | 0.6       | 0.5       | 0.1       |
| 20  | 1.0       | 0.4       | 0.3       | 0.05      |
| 25  | 0.8       | 0.3       | 0.2       | 0.03      |
| 30  | 0.6       | 0.2       | 0.15      | 0.01      |
</details>

(a) Single- vs. Multi-Token (L = 2)

![](images/04e63e14a158d4e2f988f9fb8c50ce8bf6aaab7e59ca51ef938451da6e28b9e7.jpg)

<details>
<summary>bar</summary>

| n   | S (L = 2) | P (L = 3) | T (L = 3) | S (L = 3) |
| --- | --------- | --------- | --------- | --------- |
| 5   | 0.1       | 0.1       | 0.1       | 0.1       |
| 10  | 0.05      | 0.05      | 0.05      | 0.03      |
| 15  | 0.02      | 0.02      | 0.02      | 0.01      |
| 20  | 0.01      | 0.01      | 0.01      | 0.005     |
| 25  | 0.005     | 0.005     | 0.005     | 0.001     |
| 30  | 0.002     | 0.002     | 0.002     | 0.0005    |
</details>

(b) Single- vs. Multi-Token (L = 3)

![](images/964739817da9ff299b6616e92e985accb69b2218585b5c0fc8fbf0d43a5d5f99.jpg)

<details>
<summary>bar</summary>

| L | ICL Risk (n = 1) | Task Vector |
| --- | --- | --- |
| 2 | 0.20 | 0.29 |
| 3 | 0.05 | 0.12 |
</details>

(c) ICL vs. TV   
Figure 4: (a, b) Comparison of the best ICL risk achieved using single (S), pairwise (P), and triplet (T) formatted prompts. (c) Performance comparison between 1-shot ICL and task vector.

Then any minimizer $\Lambda _ { 4 }$ of the in-context risk $\mathcal { L } \big ( \{ V _ { l } , Q _ { l } \} _ { l = 1 } ^ { L } \big )$ satisfies $( \Lambda _ { 4 } ) _ { n + 1 , : } = 0$ and:

$$
(\Lambda_ {4}) _ {1: n,:} \propto \underset {\Lambda} {\arg \min} c _ {1} \| \Lambda \| _ {4} ^ {4} + c _ {2} \sum_ {i = 1} ^ {n} \| \Lambda_ {i,:} \| _ {2} ^ {4} + c _ {3} \sum_ {j = 1} ^ {n + 1} \| \Lambda_ {:, j} \| _ {2} ^ {4} + c _ {4} \big \| \Lambda \Lambda^ {\top} \big \| _ {F} ^ {2}, s. t. \| \Lambda \| _ {F} ^ {2} = 1.
$$

where $c _ { 1 } , \cdots , c _ { 4 }$ are non-negative constants depending on $V _ { l } , Q _ { l } ,$ , and $p .$

This result suggests that dropout introduces additional higher-order regularization on the task vector weights, encouraging them to distribute more uniformly across demonstrations. Furthermore, when considering causal attention (i.e., enforcing $\Lambda _ { 4 }$ to be upper-triangular), it induces a decaying weight pattern from later to earlier demonstrations, which exactly matches the practical behavior observed in practical transformer models (as evidenced in Figure 3c).

Decoding the Vocabulary of Task Vectors. Multiple prior works (Hendel et al., 2023; Todd et al., 2024) have observed an interesting phenomenon that, when task vectors are directly decoded through the final classification layer, the top tokens often belong to the output space of the current task (see Table 4 in the Appendix). Our theoretical analysis provides a natural explanation for this: assuming a 2d-dimensional hidden state space partitioned into input $( x _ { i } )$ and output $( y _ { i } )$ halves, the output half of task vectors then encodes weighted summations of $y _ { i } .$ . Since the final prediction relies on the output half, decoding a task vector yields a combination of $y _ { i } ,$ which is likely lying in the output space. This observation suggests that practical LLMs adopt a similar hidden-state partition.

Extra EOS Tokens. In our previous analysis, we consistently imposed an additional zero token at the end of the input prompt. While this token can be interpreted as an EOS token in practical models, such a design choice is uncommon in standard ICL tasks. We justify this modeling decision with:

Proposition 6 (Informal). Given any L-layer, 1-head, d-dimensional linear-attention model with EOS, there exists an equivalent L-layer, 2-head, 2d-dimensional model operating without EOS.

This equivalence suggests that the same learning dynamics can be realized through multi-head architectures without relying on explicit EOS tokens. Specifically, the first head is dedicated to task vector formation, while the other handles ICL prediction. This separation allows the model to retain the functional role of the EOS token implicitly within its hidden states.

# 6 EXPERIMENTAL STUDIES

# 6.1 SYNTHETIC RESULTS WITH RANDOM LINEAR REGRESSION

In this section, we validate our critical points analysis with synthetic linear regression tasks. Specifically, we examine the achievable ICL risk of linear-attention models with single-token (eq. (3)), pairwise $\left( \mathrm { e q . ~ } ( 7 ) \right)$ , and triplet (eq. (9)) demonstrations. We set the input dimension to $d = 4$ and $\mathbf { \bar { \rho } } _ { P _ { x } } = P _ { w } = \mathcal { N } ( 0 , I _ { d } )$ . For each setting, we train multiple models with different random seeds and report the minimum ICL risk achieved as a proxy for the global optimum. The comparative results across different numbers of layers L and demonstration formats are shown in Figures 4a and 4b.

These results support our theoretical analysis: when trained with pairwise or triplet demonstrations, the model recovers the GD++ algorithm similar to the single-token case. Notably, the performance of L-layer models with pairwise (P) and triplet (T) demonstrations closely aligns, indicating a shared underlying learning pattern. Moreover, their performance consistently lies between that of singletoken (S) case L-layer and (L − 1)-layer models. The observed improvement over the (L − 1)-layer single-token baselines comes from the additional GD++ performed solely on $x _ { i }$ tokens in the first layer, effectively acting as a “half-step” of gradient descent.

Table 2: Accuracy comparison between few-shot ICL (Baseline), the task vector method (TaskV), and our strategy (TaskV-M). The experiment is conducted on Llama-13B with n = 10. 

<table><tr><td colspan="2">Method</td><td>Knowledge</td><td>Algorithmic</td><td>Translation</td><td>Linguistic</td><td>Bijection</td><td>Average</td></tr><tr><td rowspan="2">0-shot</td><td>Baseline</td><td>6.90 ± 2.08</td><td>15.60 ± 1.72</td><td>7.00 ± 1.65</td><td>12.44 ± 1.74</td><td>8.27 ± 1.33</td><td>10.28 ± 0.98</td></tr><tr><td>TaskV</td><td>68.80 ± 2.66</td><td>86.20 ± 1.61</td><td>73.53 ± 0.91</td><td>85.24 ± 1.80</td><td>50.67 ± 2.32</td><td>72.26 ± 1.01</td></tr><tr><td rowspan="3">1-shot</td><td>Baseline</td><td>69.50 ± 3.86</td><td>73.67 ± 1.56</td><td>57.80 ± 2.01</td><td>56.22 ± 1.57</td><td>44.76 ± 2.44</td><td>58.11 ± 0.63</td></tr><tr><td>TaskV</td><td>79.50 ± 2.35</td><td>88.47 ± 0.75</td><td>80.67 ± 2.56</td><td>89.11 ± 0.84</td><td>60.44 ± 2.07</td><td>78.79 ± 0.77</td></tr><tr><td>TaskV-M</td><td>81.30 ± 2.80</td><td>89.53 ± 0.65</td><td>80.13 ± 2.14</td><td>88.71 ± 0.62</td><td>61.78 ± 0.96</td><td>79.34 ± 0.37</td></tr><tr><td rowspan="3">2-shot</td><td>Baseline</td><td>78.80 ± 3.30</td><td>85.07 ± 1.37</td><td>75.67 ± 2.64</td><td>76.80 ± 1.18</td><td>56.49 ± 2.87</td><td>72.92 ± 0.59</td></tr><tr><td>TaskV</td><td>84.60 ± 2.11</td><td>88.40 ± 0.68</td><td>84.33 ± 0.92</td><td>90.13 ± 0.92</td><td>62.44 ± 2.16</td><td>80.82 ± 0.42</td></tr><tr><td>TaskV-M</td><td>85.70 ± 1.63</td><td>89.27 ± 1.10</td><td>84.13 ± 1.15</td><td>89.64 ± 0.86</td><td>64.49 ± 2.02</td><td>81.48 ± 0.37</td></tr><tr><td rowspan="3">3-shot</td><td>Baseline</td><td>86.20 ± 2.69</td><td>88.07 ± 1.06</td><td>80.00 ± 1.67</td><td>84.04 ± 1.19</td><td>62.18 ± 1.52</td><td>78.51 ± 0.42</td></tr><tr><td>TaskV</td><td>90.20 ± 2.23</td><td>88.67 ± 0.89</td><td>86.27 ± 2.31</td><td>92.31 ± 0.48</td><td>66.53 ± 0.94</td><td>83.53 ± 0.41</td></tr><tr><td>TaskV-M</td><td>90.30 ± 1.50</td><td>89.87 ± 0.83</td><td>86.07 ± 2.17</td><td>92.36 ± 0.72</td><td>68.13 ± 0.76</td><td>84.15 ± 0.52</td></tr><tr><td rowspan="3">4-shot</td><td>Baseline</td><td>84.80 ± 2.06</td><td>88.07 ± 0.61</td><td>83.27 ± 1.82</td><td>88.89 ± 1.91</td><td>67.16 ± 1.47</td><td>81.52 ± 0.66</td></tr><tr><td>TaskV</td><td>88.70 ± 1.69</td><td>89.53 ± 1.34</td><td>86.27 ± 1.08</td><td>92.76 ± 0.54</td><td>70.44 ± 1.35</td><td>84.66 ± 0.39</td></tr><tr><td>TaskV-M</td><td>89.60 ± 1.43</td><td>91.00 ± 1.01</td><td>87.20 ± 0.62</td><td>92.36 ± 1.44</td><td>72.53 ± 0.94</td><td>85.64 ± 0.29</td></tr></table>

We then reproduce the task vector method in linear models. Specifically, we extract the hidden state of the final (→) token from triplet demonstrations after the first layer, and inject this vector into zero-shot prompts consisting of $x _ { \mathrm { t e s t } }$ only. To simulate the effect of layer normalization, we normalize the task vectors before inference and the output vectors before ICL risk evaluation. As shown in Figure 4c, the performance of task vectors is highly related to that of standard 1-shot ICL. This validates our conjecture that the injected task vector effectively acts as a single demonstration.

# 6.2 ENHANCING THE TASK VECTOR METHOD

We further explore an enhancement to the original task vector method. According to our previous analysis, a single injected task vector may not provide sufficient information for inference on complex tasks (e.g., bijection tasks). Moreover, in linear-attention models, each (→) token functions as an individual in-context demonstration during the gradient descent phase and thus contributes equally to the ICL risk. Motivated by this, we extend the standard task vector method, which modifies only the final arrow token, and propose a multi-vector variant that injects into every single arrow token in few-shot prompts. This enriched injection scheme enables the model to leverage multiple new demonstrations, thereby providing a more informative and distributed context for prediction.

We compare our multi-vector injection strategy (TaskV-M) against standard N-shot ICL (Baseline) and the original task vector method (TaskV). Note that Baseline uses few-shot ICL and TaskV is injecting into few-shot prompts, which are different from the settings in Table 1 which uses many-shot prompts for ICL and zero-shot prompts for task vectors. For each N-shot prompt, we generate N +1 distinct ICL prompts to produce N + 1 task vectors, which are then used to replace the embeddings of all arrow tokens in the input. For each task, performance is evaluated over 50 randomly sampled prompts, with mean accuracy and standard deviation reported across 5 independent trials. The final results, summarized in Table 2, span a diverse set of ICL task types, showing that TaskV-M consistently outperforms TaskV, especially the challenging bijection tasks. While the improvement is not dramatic, we believe that the current results sufficiently demonstrate the potential of multi-vector injection, thereby providing insights for the design of future ICL or task vector methods.

# 7 CONCLUSION, LIMITATIONS, AND FUTURE WORKS

This paper proposes a plausible explanation for the emergence and functionality of task vectors in ICL. We support this conjecture with both empirical observations and theoretical analysis, demonstrating how task vectors naturally arise under ICL-style training prompts, and why this method inherently fails on general ICL tasks beyond rank-one mappings. Our work provides a new perspective on the underlying mechanisms and offers a promising direction for interpreting intermediate hidden states in modern transformer-based language models.

While our analysis provides new insights into the emergence and functionality of task vectors, it is primarily conducted on simplified linear-attention transformers and synthetic tasks, which may not fully capture the complexity of real-world LLMs. Moreover, our theoretical framework focuses solely on critical point analysis, and there is still a lack of convergence guarantee or sample complexity analysis to fully understand the learning dynamics during model pretraining.

Future directions of this work may include: (1) extending the current theoretical framework to causal and multimodal settings; (2) exploring how richer architectures (e.g., non-linear attention) or training objectives (e.g., auto-regressive loss) influence the behavior of task vectors; (3) synthesizing orthogonal enhancements of the task vector method (e.g., function vectors (Todd et al., 2024) and in-context vectors (Liu et al., 2024)), and extending to more complex reasoning tasks.

# ACKNOWLEDGMENTS

This work was supported by the Natural Science Foundation under grants IIS-2312840 and IIS-2402952. We would like to thank the anonymous reviewers for their valuable suggestions.

# ETHICS STATEMENT

This work advances the theoretical understanding of in-context learning and task vector mechanisms, which can lead to more efficient and interpretable language models. By enabling faster inference through task vectors, it may reduce the computational cost and energy consumption of large-scale deployment, thereby making AI systems more accessible and environmentally sustainable. Improved interpretability could also enhance trust and transparency in AI applications across education, healthcare, and other socially beneficial domains.

As task vector methods improve efficiency and transferability, they may also be misused to replicate or extract functionality from proprietary models without authorization, raising concerns around model intellectual property. Additionally, while interpretability is often framed as a benefit, deeper insights into model internals could be exploited to engineer adversarial inputs or extract sensitive training data. Careful consideration and mitigation strategies are essential to ensure that such work aligns with the broader goals of safe and beneficial AI.

# REPRODUCIBILITY STATEMENT

We provide complete proofs for our main theoretical results in Appendices C and D, experimental details about the dataset and implementation in Appendix B, and full source codes to reproduce our experimental results at https://github.com/Yuxin-Dong/ICL-TaskVector.

# USAGE OF LLMS

We used LLMs only to improve grammar and polish academic writing. All technical ideas, proofs, experiments, and conclusions were entirely conceived and verified by the authors.

# REFERENCES

Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. Advances in Neural Information

Processing Systems, 36:45614–45650, 2023.   
Ekin Akyurek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning ¨ algorithm is in-context learning? investigations with linear models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=0g0X4H8yN4I.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
Dake Bu, Wei Huang, Andi Han, Atsushi Nitanda, Qingfu Zhang, Hau-San Wong, and Taiji Suzuki. Provable in-context vector arithmetic via retrieving task concepts. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id= DbUmeNnNpt.   
Stephanie Chan, Adam Santoro, Andrew Lampinen, Jane Wang, Aaditya Singh, Pierre Richemond, James McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. Advances in neural information processing systems, 35:18878–18891, 2022.   
Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can gpt learn in-context? language models secretly perform gradient descent as meta-optimizers. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 4005–4019, 2023.   
Gilad Deutch, Nadav Magar, Tomer Natan, and Guy Dar. In-context learning and gradient descent revisited. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 1017–1028, 2024.   
Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583–30598, 2022.   
Seungwook Han, Jinyeop Song, Jeff Gore, and Pulkit Agrawal. Emergence and effectiveness of task vectors in in-context learning: An encoder decoder perspective. In Forty-second International Conference on Machine Learning, 2025.   
Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 9318–9333, 2023.   
Alberto Hojel, Yutong Bai, Trevor Darrell, Amir Globerson, and Amir Bar. Finding visual task vectors. In European Conference on Computer Vision, pp. 257–273. Springer, 2024.   
Brandon Huang, Chancharik Mitra, Leonid Karlinsky, Assaf Arbelle, Trevor Darrell, and Roei Herzig. Multimodal task vectors enable many-shot multimodal in-context learning. Advances in Neural Information Processing Systems, 37:22124–22153, 2024.   
Joonseong Kang, Soojeong Lee, Subeen Park, Sumin Park, Taero Kim, Jihee Kim, Ryunyi Lee, and Kyungwoo Song. Adaptive task vectors for large language models. arXiv preprint arXiv:2506.03426, 2025.   
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.   
Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. Advances in Neural Information Processing Systems, 36:24892–24928, 2023.   
Dongfang Li, Xinshuo Hu, Zetian Sun, Baotian Hu, Min Zhang, et al. In-context learning state vector with inner and momentum optimization. Advances in Neural Information Processing Systems, 37: 7797–7820, 2024.

Sheng Liu, Haotian Ye, Lei Xing, and James Y Zou. In-context vectors: Making in context learning more effective and controllable through latent space steering. In International Conference on Machine Learning, pp. 32287–32307. PMLR, 2024.   
Grace Luo, Trevor Darrell, and Amir Bar. Vision-language models create cross-modal task representations. In Forty-second International Conference on Machine Learning, 2025.   
Arvind V. Mahankali, Tatsunori Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=8p3fu56lKc.   
Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. Language models implement simple word2vecstyle vector arithmetic. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 5030–5047, 2024.   
Yingzhe Peng, Xinting Hu, Jiawei Peng, Xin Geng, Xu Yang, et al. Live: Learnable in-context vector for visual question answering. Advances in Neural Information Processing Systems, 37: 9773–9800, 2024.   
Lingfeng Shen, Aayush Mishra, and Daniel Khashabi. Position: Do pretrained transformers learn in-context by gradient descent? In Proceedings of the 41st International Conference on Machine Learning, pp. 44712–44740. PMLR, 2024.   
Pavel Tikhonov, Ivan Oseledets, and Elena Tutubalina. One task vector is not enough: A large-scale study for in-context learning. arXiv preprint arXiv:2505.23911, 2025.   
Eric Todd, Millicent Li, Arnab Sen Sharma, Aaron Mueller, Byron C Wallace, and David Bau. Function vectors in large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=AwyxtyMwaG.   
Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, Joao Sacramento, Alexander Mordv-˜ intsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pp. 35151–35174. PMLR, 2023a.   
Johannes Von Oswald, Maximilian Schlegel, Alexander Meulemans, Seijin Kobayashi, Eyvind Niklasson, Nicolas Zucchet, Nino Scherrer, Nolan Miller, Mark Sandler, Max Vladymyrov, et al. Uncovering mesa-optimization algorithms in transformers. arXiv preprint arXiv:2309.05858, 2023b.   
Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Label words are anchors: An information flow perspective for understanding in-context learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 9840–9855, 2023.   
Kevin Christian Wibisono and Yixin Wang. On the role of unstructured training data in transformers’ in-context learning capabilities. In NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning, 2023.   
Jingfeng Wu, Difan Zou, Zixiang Chen, Vladimir Braverman, Quanquan Gu, and Peter Bartlett. How many pretraining tasks are needed for in-context learning of linear regression? In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=vSh5ePa0ph.   
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=RdJVFCHjUMI.   
Yue Xing, Xiaofeng Lin, Chenheng Xu, Namjoon Suh, Qifan Song, and Guang Cheng. Theoretical understanding of in-context learning in shallow transformers with unstructured data. arXiv preprint arXiv:2402.00743, 2024.

Zheyang Xiong, Ziyang Cai, John Cooper, Albert Ge, Vasilis Papageorgiou, Zack Sifakis, Angeliki Giannou, Ziqian Lin, Liu Yang, Saurabh Agarwal, et al. Everything everywhere all at once: Llms can in-context learn multiple tasks in superposition. arXiv preprint arXiv:2410.05603, 2024.   
Liu Yang, Ziqian Lin, Kangwook Lee, Dimitris Papailiopoulos, and Robert Nowak. Task vectors in in-context learning: Emergence, formation, and benefit. arXiv preprint arXiv:2501.09240, 2025.   
Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. Journal of Machine Learning Research, 25(49):1–55, 2024.   
Chunsheng Zuo, Pavel Guerzhoy, and Michael Guerzhoy. Position information emerges in causal transformers without positional encodings via similarity of nearby embeddings. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 9418–9430, 2025.

# A ADDITIONAL DISCUSSIONS

# A.1 SUMMARY OF MATHEMATICAL NOTATIONS

Table 3: Summary of key mathematical notations used throughout the paper. 

<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $n$ </td><td>Number of demonstrations in the input prompt</td></tr><tr><td> $L$ </td><td>Number of transformer layers</td></tr><tr><td> $d$ </td><td>Dimension of covariate and response embeddings</td></tr><tr><td> $d_p$ </td><td>Prompt length (depends on demonstration structure)</td></tr><tr><td> $\text{Attn}_{V,Q}$ </td><td>Linear-attention layer with parameter  $V, Q$ </td></tr><tr><td> $\text{TF}$ </td><td>Linear-attention model by stacking linear-attention layers</td></tr><tr><td> $x_i \in \mathbb{R}^d$ </td><td>Covariate (input) of the  $i$ -th demonstration</td></tr><tr><td> $y_i \in \mathbb{R}^d$ </td><td>Response (output) of the  $i$ -th demonstration</td></tr><tr><td> $X, Y \in \mathbb{R}^{d \times n}$ </td><td>Matrices of covariates and responses for  $n$  demonstrations</td></tr><tr><td> $x_{\text{test}}, y_{\text{test}}$ </td><td>Query covariate and ground-truth response</td></tr><tr><td> $w_j \in \mathbb{R}^d$ </td><td> $j$ -th regression coefficient vector</td></tr><tr><td> $W \in \mathbb{R}^{d \times d}$ </td><td>Coefficient matrix,  $W = [w_1, \cdots, w_d]^{\top}$ </td></tr><tr><td> $Z_0 \in \mathbb{R}^{2d \times d_p}$ </td><td>Input prompt embeddings before the transformer</td></tr><tr><td> $Z_l \in \mathbb{R}^{2d \times d_p}$ </td><td>Hidden states after the  $l$ -th layer</td></tr><tr><td> $P \in \mathbb{R}^{d_p \times d_p}$ </td><td>Positional encoding matrix</td></tr><tr><td> $V_l, Q_l$ </td><td>Value and key-query matrices of the  $l$ -th attention layer</td></tr><tr><td> $A_l, B_l, C_l, D_l$ </td><td>Block components of  $V_l, Q_l$  in layer  $l$ </td></tr><tr><td> $\Lambda_k$ </td><td>Sub-block matrices of  $D_l$  used in critical point analysis</td></tr><tr><td> $\mathcal{L}$ </td><td>In-context learning loss (ICL risk)</td></tr><tr><td> $\mathcal{M}(M)$ </td><td>Set of masked matrices with binary mask  $M$ </td></tr><tr><td> $\mathcal{S}_I, \mathcal{S}_\Sigma, \mathcal{S}_P$ </td><td>Structured sets of matrices defining critical points</td></tr><tr><td> $z_{\text{tv}}$ </td><td>Task vector extracted from an arrow ( $\rightarrow$ ) token</td></tr><tr><td> $\beta \in \mathbb{R}^n$ </td><td>Weight vector for task vector formation</td></tr></table>

# A.2 ADDITIONAL RELATED WORKS

In-Context Learning in Attention-based LLMs. The ability of LLMs to learn from examples provided in the input prompt, without updating parameters, has attracted wide attention since the discovery of ICL in GPT-3 (Brown et al., 2020). A growing body of theoretical work has sought to explain this phenomenon. Early analyses show that transformer attention layers can implement gradient descent–like algorithms over linear regression objectives (Garg et al., 2022; Akyurek et al., ¨ 2023; Von Oswald et al., 2023a; Ahn et al., 2023; Wu et al., 2024), while others investigate sample complexity and generalization behavior (Xie et al., 2022; Chan et al., 2022; Shen et al., 2024; Von Oswald et al., 2023b; Deutch et al., 2024). These works collectively suggest that ICL is closely tied to the inductive biases of the attention mechanism, but do not fully explain how higher-level abstractions of tasks are formed or encoded in LLMs.

The Task Vector Method in ICL. Task vectors have recently been proposed as an abstraction of ICL demonstrations into compact hidden-state representations. Hendel et al. (2023) introduced task vectors as hidden states extracted from the last arrow token in triplet prompts, enabling zero-shot transfer by injecting them into new contexts. Concurrent works developed similar notions, such as function vectors (Todd et al., 2024) and in-context vectors (Liu et al., 2024). These studies show that task vectors accelerate inference and sometimes match the effectiveness of ICL with fewer demonstrations. However, they remain largely empirical, without a clear theoretical explanation of how or why such vectors encode task information.

Subsequent research has expanded the scope and utility of task vectors. Yang et al. (2025) demonstrates that task vectors naturally emerge even in small transformers trained from scratch with synthetic data, suggesting that their formation is an inherent property of attention-based architectures.

Table 4: Top 20 tokens with the highest output probability by decoding the task vector, results from (Hendel et al., 2023). We underline the tokens in the output space of the current task. 

<table><tr><td>Model</td><td>Task</td><td>Tokens</td></tr><tr><td rowspan="4">GPT-J 6B</td><td>Prev Letter</td><td>b, c, v, g, s, name, i, ro, n, j, d, t, A, ai, com, m, ust, test, active, k</td></tr><tr><td>French to English</td><td>other, name, the, true, is, social, s, active, time, car, type, money, F, force, a, public, heart, one, ms, life</td></tr><tr><td>Present to Gerund</td><td>getting, storing, working, moving, playing, doing, making, driving, shooting, picking, being, sending, putting, selling, watching, changing, taking, collecting, feeding, reading</td></tr><tr><td>Country to Capital</td><td>London, Paris, New, West, Berlin, South, Tokyo, San, Chicago, City, Moscow, Jerusalem, Amsterdam, Philadelphia, East, Madrid, Vienna, Beijing, Mexico, Germany</td></tr></table>

Li et al. (2024) shows that aggregating hidden states across layers and multiple arrow tokens leads to stronger task representations. Kang et al. (2025) proposes to generate task vectors conditioned on each input query. Beyond text, task vectors have also been applied in vision (Hojel et al., 2024; Peng et al., 2024) and multimodal models (Huang et al., 2024; Luo et al., 2025), where they enable flexible transfer across modalities. Han et al. (2025) connects the performance of task vectors by task decodability, defined by the similarity between task vectors from different ICL tasks. These works highlight the empirical utility of task vectors but stop short of explaining their inner mechanisms.

Explaining the Task Vector Method. Task vectors were initially conjectured to encapsulate the complete knowledge of the current task (Hendel et al., 2023). However, this view fails to account for their inconsistent performance across tasks of varying complexity. Empirical observations further suggest that directly decoding task vectors typically yields tokens from the task output space (Todd et al., 2024), rather than explicit task descriptions (Merullo et al., 2024). Concurrent work by Bu et al. (2025) analyzes the learning dynamics of 1-layer transformers with ICL-style prompts, explaining the utility of task vectors through a word2vec-like scheme (i.e., the existence of a vector $z _ { t }$ for task t such that $y \approx z _ { t } + x$ for all input-output pairs $( x , y ) )$ . While insightful, this characterization is restricted to additive translation tasks, single-token prompts, and single-layer architectures, limiting its generality. By contrast, our analysis encompasses richer prompt structures, including pairwise and triplet formats that better reflect practical ICL settings. Moreover, our critical point characterization extends beyond 1-layer models, and our linear regression formulation captures a broader spectrum of ICL tasks. Complementing our findings, Tikhonov et al. (2025) independently shows that standard task vectors lack sufficient expressiveness for complex ICL tasks, reinforcing our conclusion that task vectors are fundamentally constrained by rank-one mappings.

# A.3 JUSTIFICATION OF THE BLOCK-DIAGONAL ASSUMPTION

In our main analysis, we impose an assumption on the trainable parameters of linear-attention layers, such that the $V _ { l }$ and $Q _ { l }$ matrices are block-diagonal in eq. (6). This block-diagonal formulation is a widely adopted assumption in theoretical studies of ICL for transformer models, as it facilitates tractable analysis (Ahn et al., 2023; Mahankali et al., 2024; Wu et al., 2024; Zhang et al., 2024). Prior work by Ahn et al. (2023) demonstrates that the global minimizer of single-layer linear-attention transformers indeed exhibits such a block-diagonal structure. Although finding exact solutions for multi-layer transformers is more involved, it is reasonable to conjecture that similar structural patterns hold. Empirically, we observe that when optimizing the full matrices, gradient-based training also tends to converge to block-diagonal solutions.

Intuitively, given the high dimensionality of hidden states in modern LLMs, it is plausible to assume that the $x _ { i }$ and $y _ { i }$ components can be projected into orthogonal or nearly orthogonal subspaces when mixed in the hidden state space. This motivates a decomposition of the projection matrices Vl and $Q _ { l }$ into two separate parts that operate independently on $x _ { i }$ and $y _ { i }$ , which can be equivalently formulated as the block-diagonal structures.

# A.4 INSEPARABLE COVARIATES AND RESPONSES

In our main analysis, we assume that $x _ { i }$ and $y _ { i }$ embeddings are linearly separable, allowing the addition $x _ { i } + y _ { i }$ to act a concatenation operation. However, recognizing that this assumption does not generally hold for real-world transformers, we extend our analysis to the following setting, where $x _ { i }$ and $y _ { i }$ are no longer linearly separable. While this still imposes a 2d-dimensional requirement on the hidden space, such a constraint is easily satisfied in practical transformers, given the high dimensionality of their internal representations.

$$
Z _ {0} = \left[ \begin{array}{c c c c c c c} 0 & 0 & \dots & 0 & 0 & 0 & 0 \\ x _ {1} & y _ {1} & \dots & x _ {n} & y _ {n} & x _ {\text { test }} & 0 \end{array} \right] \in \mathbb {R} ^ {(2 d) \times (2 n + 2)}. \tag {11}
$$

We slightly modify the sparsity constraints for the first layer, and require $( D _ { 0 } ) _ { 2 i , : } = 0$ for $i \in [ n { + 1 } ]$ :

$$
V _ {0} = \left[ \begin{array}{c c} 0 & A _ {0} \\ 0 _ {d \times d} & 0 \end{array} \right], \quad Q _ {0} = \left[ \begin{array}{c c} 0 _ {2 d \times 2 d} & 0 \\ 0 & D _ {0} \end{array} \right], \quad \text { where } A _ {0} \in \mathbb {R} ^ {d \times d}, D _ {0} \in \mathbb {R} ^ {d _ {p} \times d _ {p}}. \tag {12}
$$

With these conditions, we are ready to establish the critical points for inseparable demonstrations. Note that $V _ { 0 }$ and $Q _ { 0 }$ do not involve $B _ { 0 }$ and $C _ { 0 }$ , so the sequences B and C have size $L - 1$ .

Theorem 7. Under the same settings as Theorem 1, define $S _ { I } , S _ { \Sigma } \subset \mathbb { R } ^ { d \times d }$ and $S _ { P } \subset \mathbb { R } ^ { d _ { p } \times d _ { p } }$ as

$$
\mathcal {S} _ {I} = \left\{\lambda I _ {d} \mid \lambda \in \mathbb {R} \right\}, \quad \mathcal {S} _ {\Sigma} = \left\{\lambda \Sigma^ {- 1} \mid \lambda \in \mathbb {R} \right\}, \quad \mathcal {S} _ {P} = \left\{\operatorname{diag} (I _ {n} \otimes \Lambda_ {1}, \Lambda_ {2}) \mid \Lambda_ {1}, \Lambda_ {2} \in \mathbb {R} ^ {2 \times 2} \right\}.
$$

Consider optimizing an L-layer linear transformer with inseparable pairwise demonstrations and parameter configuration given in eq. (12) for the first layer and eq. (6) for the remaining layers, then

$$
\inf _ {A \in \mathcal {S} _ {I} ^ {L}, B \in \mathcal {S} _ {I} ^ {L - 1}, C \in \mathcal {S} _ {\Sigma} ^ {L - 1}, D \in \mathcal {S} _ {P} ^ {L}} \sum_ {H \in A \cup B \cup C \cup D} \left\| \nabla_ {H} \mathcal {L} \big (\{V _ {l}, Q _ {l} \} _ {l = 1} ^ {L} \big) \right\| _ {F} ^ {2} = 0.
$$

This result suggests that for inseparable demonstrations, the first layer performs a functionally similar concatenation operation by “moving” the embedding of each $x _ { i }$ to the corresponding $y _ { i }$ position. This enables the model to reconstruct the single-token structure without linear separability.

# A.5 LAST TASK VECTOR WEIGHTS THE MOST

While our analysis of linear-attention models suggests that each formed task vector (i.e., the hidden state at each arrow token) contributes equally to the final prediction, this assumption does not fully hold in practical LLMs. As demonstrated by the conflicting tasks experiment in (Hendel et al., 2023), injecting a task vector from task B into an ICL prompt designed for task A causes the model to predominantly perform task B. This behavior indicates that LLMs largely rely on the last arrow token to determine the task identity. We attribute this to the causal attention mechanism used in practical LLMs, which is not captured by our current theoretical analysis. In causal attention, only the final arrow token can aggregate information from the entire preceding context, making it the most informative and influential for prediction. This explains why our multi-vector strategy offers modest, though consistent, performance gains. The improvement suggests that intermediate arrow tokens do participate in the inference process, albeit less effectively. Enhancing how LLMs utilize information from all arrow tokens remains a promising direction for improving task vector accuracy and robustness.

# B EXPERIMENT DETAILS AND ADDITIONAL RESULTS

In this section, we present experiment details and additional results not included in the main text due to space limitations. Our experiments are conducted on an A100 40G GPU. It takes around 30 GPU hours to fully reproduce our results.

# B.1 SYNTHETIC EXPERIMENTS ON LINEAR-ATTENTION MODELS

We consider training linear-attention models on random linear regression instances. We take embedding dimension $d = 4$ , and the distributions for generating $x _ { i }$ and wi are both $P _ { x } = P _ { w } = \mathcal { N } ( 0 , I _ { d } )$ . We optimize the ICL risk for L-layer linear-attention models with n in-context demonstrations using AdamW, where $L \in [ 3 ]$ and $n \in [ 5 , 3 0 ]$ . Each gradient step is computed from a batch size of 1000. We additionally apply ℓ1 regularization to simplify the found solutions. For training efficiency and stability, we restrict the $A _ { l } , B _ { l }$ , and $C _ { l }$ matrices to $ { \boldsymbol { S } } _ { I }$ during training, and initialize $D _ { l } \in \mathbb { R } ^ { \check { d } _ { p } \times d _ { p } }$ with i.i.d. Gaussian matrices. For each case, we train 40 models with different random seeds, and report the minimum achieved ICL risk to approximate the global minimum.

To reproduce the task vector mechanism, we focus on models trained with triplet-formatted prompts. The training procedure is identical to the above. For inference, we restrict $P _ { w }$ to rank-one coefficient matrices, by letting $W = w _ { 1 } w _ { 2 } ^ { \top }$ , where $w _ { 1 } , w _ { 2 } \sim \mathcal { N } ( 0 , I _ { d } )$ . We first generate normal ICL prompts to generate task vectors as the hidden states of the last arrow token after the first attention layer, and then inject them into zero-shot prompts after normalization. The final outputs $\hat { y } _ { \mathrm { t e s t } }$ are taken as the output of these injected zero-shot prompts after being processed with the same transformer model. We compute the final risk as $\begin{array} { r } { \bar { \mathsf { L } } \left\| \frac { \hat { y } _ { \mathrm { t e s t } } } { \| \hat { y } _ { \mathrm { t e s t } } \| } + \frac { y _ { \mathrm { t e s t } } } { \| y _ { \mathrm { t e s t } } \| } \right\| } \end{array}$ to simulate the layer normalization blocks in practical LLMs. The reported scores are averaged for $n \in [ 5 , 3 0 ]$ ].

# B.2 EXPERIMENTS ON PRACTICAL LLMS

Datasets. Following the settings of the original task vector method (Hendel et al., 2023), our study covers 33 tasks in 5 categories. The detailed description for each task is provided in Table 5.

Prompt Template. The template used to construct ICL demonstrations is “Example: $\{ x _ { i } \}  \{ y _ { i } \}$ , where $x _ { i }$ and $y _ { i }$ are subsequently replaced by the input and output of the semantic mapping. For the query part, $y _ { i }$ is omitted from the prompt. After concatenating each demonstration with $^ { \ 6 6 } \backslash \ n ^ { \prime \ }$ , an example of the full input prompt is:

$$
\text { Example: } \{x _ {1} \} \rightarrow \{y _ {1} \} \backslash n \dots \text { Example: } \{x _ {n} \} \rightarrow \{y _ {n} \} \backslash n \text { Example: } \{x _ {\text { test }} \} \rightarrow \tag {13}
$$

Evaluation. To evaluate the N-shot performance, we generate $5 0 \times ( N + 1 )$ i.i.d. prompts for each task with number of demonstrations $n = 1 0$ for task vector extraction. The hidden states of the last  token, which is also literally the last token in the prompt, are recorded for every layer in the transformer. Thereafter, we generate another 50 i.i.d. prompts with N demonstrations, where $x _ { \mathrm { t e s t } }$ is selected to be distinct from the previous chosen ones. The final accuracy is measured by whether the next word predicted matches the expected answer. The performance of the standard ICL method (Baseline) is acquired by inferring without interference. For the task vector method (TaskV) and our multi-vector variant (TaskV-M), the extracted task vectors are injected to replace the hidden states of the arrow  tokens at a specified layer l. For TaskV, only the last arrow token is injected, while for TaskV-M, each of the $N + 1$ arrow tokens is injected with the $N + 1$ extracted task vectors for the same task. The performance is reported for the layer $l \in L$ achieving the highest accuracy. For each case, the mean and standard deviation are evaluated through 5 independent trials.

Additional Results. Besides Llama-13B, we also observe consistent accuracy improvement of our TaskV-M method on the Pythia-12B model, as reported in Table 6.

While the performance gains of TaskV-M over TaskV are not dramatic across all ICL tasks, the goal of TaskV-M is not to surpass state-of-the-art ICL techniques but to demonstrate that the task vector framework can be systematically extended by injecting multiple vectors simultaneously. This is especially valuable for complex tasks that inherently require higher-rank representations. Our results on bijection tasks clearly validate this motivation: TaskV-M yields notable improvements over the standard TaskV method. For other simpler tasks, the marginal gains from TaskV-M suggest that the expressiveness of W may not be the primary performance bottleneck. We believe these insights facilitate the design of future ICL and task vector methods.

# B.3 ANOTHER MULTI-VECTOR INJECTION VARIANT

In our main experiments, we implement TaskV-M by extracting $N + 1$ task vectors from the same number of different prompts. Another possible implementation for TaskV-M is to extract multiple

Table 5: Descriptions of the tasks used in our empirical studies. 

<table><tr><td>Category</td><td>Task</td><td>Example</td><td>Description</td></tr><tr><td rowspan="4">Knowledge</td><td>Contry to Capital</td><td>France → Paris</td><td>Output the capital city of the given country.</td></tr><tr><td>Person to Language</td><td>Macron → French</td><td>Output the native language of the given person.</td></tr><tr><td>Location to Continent</td><td>Paris → Europe</td><td>Output the corresponding continent of the given location.</td></tr><tr><td>Religion</td><td>Saladin → Muslim</td><td>Output the associated religion of the given location or person.</td></tr><tr><td rowspan="6">Algorithmic</td><td>List First</td><td>[a,b,c] → a</td><td>Output the first item in the given list.</td></tr><tr><td>List Last</td><td>[a,b,c] → c</td><td>Output the last item in the given list.</td></tr><tr><td>Next Letter</td><td>a → b</td><td>Output the next letter of the given letter in the alphabet.</td></tr><tr><td>Prev Letter</td><td>b → a</td><td>Output the previous letter of the given letter in the alphabet.</td></tr><tr><td>To Upper</td><td>a → A</td><td>Output the corresponding uppercase letter of the given lowercase letter.</td></tr><tr><td>To Lower</td><td>A → a</td><td>Output the corresponding lowercase letter of the given uppercase letter.</td></tr><tr><td rowspan="6">Translation</td><td>English to French</td><td>hello → bonjour</td><td>Translate the given word in English to French.</td></tr><tr><td>English to Italian</td><td>hello → ciao</td><td>Translate the given word in English to Italian.</td></tr><tr><td>English to Spanish</td><td>hello → hola</td><td>Translate the given word in English to Spanish.</td></tr><tr><td>French to English</td><td>bonjour → hello</td><td>Translate the given word in French to English.</td></tr><tr><td>Italian to English</td><td>ciao → hello</td><td>Translate the given word in Italian to English.</td></tr><tr><td>Spanish to English</td><td>hola → hello</td><td>Translate the given word in Spanish to English.</td></tr><tr><td rowspan="9">Linguistic</td><td>Present to Gerund</td><td>go → going</td><td>Output the corresponding gerund form of the given verb in present simple tense.</td></tr><tr><td>Present to Past</td><td>go → went</td><td>Output the corresponding past simple form of the given verb in present simple tense.</td></tr><tr><td>Present to Past Perfect</td><td>go → gone</td><td>Output the corresponding past perfect form of the given verb in present simple tense.</td></tr><tr><td>Gerund to Present</td><td>going → go</td><td>Output the corresponding present simple form of the given verb in gerund form.</td></tr><tr><td>Past to Present</td><td>went → go</td><td>Output the corresponding present simple form of the given verb in past simple tense.</td></tr><tr><td>Past Perfect to Present</td><td>gone → go</td><td>Output the corresponding present simple form of the given verb in past perfect tense.</td></tr><tr><td>Singular to Plural</td><td>dog → dogs</td><td>Output the corresponding plural form of the given noun in singular form.</td></tr><tr><td>Plural to Singular</td><td>dogs → dog</td><td>Output the corresponding singular form of the given noun in plural form.</td></tr><tr><td>Antonym</td><td>happy → sad</td><td>Output the antonym of the given adjective.</td></tr><tr><td rowspan="8">Bijection</td><td>To Upper &amp; Lower</td><td>a ↔ A</td><td>Output the given letter in uppercase if it is in lowercase, and vice versa.</td></tr><tr><td>English &amp; French</td><td>hello ↔ bonjour</td><td>Translate the given word to French if it is in English, and vice versa.</td></tr><tr><td>English &amp; Italian</td><td>hello ↔ ciao</td><td>Translate the given word to Italian if it is in English, and vice versa.</td></tr><tr><td>English &amp; Spanish</td><td>hello ↔ hola</td><td>Translate the given word to Spanish if it is in English, and vice versa.</td></tr><tr><td>Present &amp; Gerund</td><td>go ↔ going</td><td>Output the given verb in gerund form if it is in present simple tense, and vice versa.</td></tr><tr><td>Present &amp; Past</td><td>go ↔ went</td><td>Output the given verb in past simple form if it is in present simple tense, and vice versa.</td></tr><tr><td>Present &amp; Past Perfect</td><td>go ↔ gone</td><td>Output the given verb in past perfect form if it is in present simple tense, and vice versa.</td></tr><tr><td>Singular &amp; Plural</td><td>dog ↔ dogs</td><td>Output the given noun in plural form if it is in singular form, and vice versa.</td></tr></table>

Table 6: Accuracy comparison between standard ICL (Baseline), the task vector method (TaskV), and our strategy (TaskV-M). The experiment is conducted on Pythia-12B with n = 10. 

<table><tr><td colspan="2">Method</td><td>Knowledge</td><td>Algorithmic</td><td>Translation</td><td>Linguistic</td><td>Bijection</td><td>Average</td></tr><tr><td rowspan="2">0-shot</td><td>Baseline</td><td>6.60 ± 1.59</td><td>14.07 ± 1.45</td><td>8.60 ± 0.68</td><td>12.53 ± 1.57</td><td>10.31 ± 0.70</td><td>10.82 ± 0.48</td></tr><tr><td>TaskV</td><td>63.30 ± 2.62</td><td>84.73 ± 1.22</td><td>62.07 ± 0.98</td><td>82.58 ± 1.22</td><td>42.27 ± 0.92</td><td>66.40 ± 0.96</td></tr><tr><td rowspan="3">1-shot</td><td>Baseline</td><td>61.80 ± 5.45</td><td>72.80 ± 1.15</td><td>43.27 ± 2.92</td><td>57.07 ± 1.15</td><td>41.91 ± 2.83</td><td>53.95 ± 1.02</td></tr><tr><td>TaskV</td><td>76.40 ± 2.40</td><td>84.20 ± 1.05</td><td>71.47 ± 1.41</td><td>87.16 ± 2.04</td><td>53.11 ± 2.37</td><td>73.59 ± 0.79</td></tr><tr><td>TaskV-M</td><td>77.70 ± 2.52</td><td>83.73 ± 1.37</td><td>71.00 ± 1.48</td><td>86.80 ± 1.59</td><td>53.87 ± 2.90</td><td>73.68 ± 0.90</td></tr><tr><td rowspan="3">2-shot</td><td>Baseline</td><td>70.30 ± 3.71</td><td>82.13 ± 0.54</td><td>60.80 ± 1.81</td><td>81.16 ± 1.57</td><td>50.76 ± 2.17</td><td>68.41 ± 0.64</td></tr><tr><td>TaskV</td><td>80.30 ± 2.46</td><td>87.00 ± 1.63</td><td>76.13 ± 3.77</td><td>89.33 ± 0.70</td><td>58.67 ± 2.44</td><td>77.41 ± 0.50</td></tr><tr><td>TaskV-M</td><td>81.60 ± 1.56</td><td>86.47 ± 0.40</td><td>77.27 ± 2.53</td><td>89.51 ± 0.88</td><td>59.24 ± 2.48</td><td>77.87 ± 0.76</td></tr><tr><td rowspan="3">3-shot</td><td>Baseline</td><td>77.60 ± 2.40</td><td>81.87 ± 0.81</td><td>68.13 ± 2.02</td><td>86.31 ± 1.93</td><td>55.73 ± 1.60</td><td>73.20 ± 0.31</td></tr><tr><td>TaskV</td><td>84.00 ± 2.76</td><td>86.33 ± 1.17</td><td>79.53 ± 2.27</td><td>92.00 ± 0.67</td><td>58.76 ± 1.53</td><td>79.06 ± 0.67</td></tr><tr><td>TaskV-M</td><td>85.40 ± 2.31</td><td>87.07 ± 1.18</td><td>78.13 ± 1.86</td><td>92.84 ± 0.68</td><td>59.56 ± 1.27</td><td>79.54 ± 0.35</td></tr><tr><td rowspan="3">4-shot</td><td>Baseline</td><td>78.40 ± 1.83</td><td>82.73 ± 0.44</td><td>72.40 ± 1.24</td><td>88.89 ± 1.25</td><td>57.91 ± 1.46</td><td>75.46 ± 0.64</td></tr><tr><td>TaskV</td><td>83.80 ± 1.12</td><td>87.60 ± 1.81</td><td>80.20 ± 2.39</td><td>92.18 ± 0.96</td><td>59.38 ± 0.47</td><td>79.59 ± 0.62</td></tr><tr><td>TaskV-M</td><td>84.30 ± 1.50</td><td>88.13 ± 0.81</td><td>80.00 ± 2.67</td><td>91.87 ± 1.25</td><td>60.31 ± 0.86</td><td>79.87 ± 0.51</td></tr></table>

Table 7: Accuracy comparison between few-shot ICL (Baseline), the task vector method (TaskV), the multi-vector method (TaskV-M), and the single-prompt variant (TaskV-MS). The experiment is conducted on Llama-13B with n = 10. 

<table><tr><td colspan="2">Method</td><td>Knowledge</td><td>Algorithmic</td><td>Translation</td><td>Linguistic</td><td>Bijection</td><td>Average</td></tr><tr><td rowspan="2">0-shot</td><td>Baseline</td><td>6.90 ± 2.08</td><td>15.60 ± 1.72</td><td>7.00 ± 1.65</td><td>12.44 ± 1.74</td><td>8.27 ± 1.33</td><td>10.28 ± 0.98</td></tr><tr><td>TaskV</td><td>68.80 ± 2.66</td><td>86.20 ± 1.61</td><td>73.53 ± 0.91</td><td>85.24 ± 1.80</td><td>50.67 ± 2.32</td><td>72.26 ± 1.01</td></tr><tr><td rowspan="4">1-shot</td><td>Baseline</td><td>69.50 ± 3.86</td><td>73.67 ± 1.56</td><td>57.80 ± 2.01</td><td>56.22 ± 1.57</td><td>44.76 ± 2.44</td><td>58.11 ± 0.63</td></tr><tr><td>TaskV</td><td>79.50 ± 2.35</td><td>88.47 ± 0.75</td><td>80.67 ± 2.56</td><td>89.11 ± 0.84</td><td>60.44 ± 2.07</td><td>78.79 ± 0.77</td></tr><tr><td>TaskV-M</td><td>81.30 ± 2.80</td><td>89.53 ± 0.65</td><td>80.13 ± 2.14</td><td>88.71 ± 0.62</td><td>61.78 ± 0.96</td><td>79.34 ± 0.37</td></tr><tr><td>TaskV-MS</td><td>80.90 ± 3.10</td><td>88.40 ± 0.93</td><td>80.13 ± 2.54</td><td>88.89 ± 0.73</td><td>61.11 ± 1.31</td><td>78.96 ± 0.43</td></tr><tr><td rowspan="4">2-shot</td><td>Baseline</td><td>78.80 ± 3.30</td><td>85.07 ± 1.37</td><td>75.67 ± 2.64</td><td>76.80 ± 1.18</td><td>56.49 ± 2.87</td><td>72.92 ± 0.59</td></tr><tr><td>TaskV</td><td>84.60 ± 2.11</td><td>88.40 ± 0.68</td><td>84.33 ± 0.92</td><td>90.13 ± 0.92</td><td>62.44 ± 2.16</td><td>80.82 ± 0.42</td></tr><tr><td>TaskV-M</td><td>85.70 ± 1.63</td><td>89.27 ± 1.10</td><td>84.13 ± 1.15</td><td>89.64 ± 0.86</td><td>64.49 ± 2.02</td><td>81.48 ± 0.37</td></tr><tr><td>TaskV-MS</td><td>84.40 ± 2.13</td><td>89.53 ± 0.98</td><td>84.67 ± 1.73</td><td>90.18 ± 1.39</td><td>64.49 ± 2.30</td><td>81.61 ± 0.80</td></tr><tr><td rowspan="4">3-shot</td><td>Baseline</td><td>86.20 ± 2.69</td><td>88.07 ± 1.06</td><td>80.00 ± 1.67</td><td>84.04 ± 1.19</td><td>62.18 ± 1.52</td><td>78.51 ± 0.42</td></tr><tr><td>TaskV</td><td>90.20 ± 2.23</td><td>88.67 ± 0.89</td><td>86.27 ± 2.31</td><td>92.31 ± 0.48</td><td>66.53 ± 0.94</td><td>83.53 ± 0.41</td></tr><tr><td>TaskV-M</td><td>90.30 ± 1.50</td><td>89.87 ± 0.83</td><td>86.07 ± 2.17</td><td>92.36 ± 0.72</td><td>68.13 ± 0.76</td><td>84.15 ± 0.52</td></tr><tr><td>TaskV-MS</td><td>90.60 ± 2.20</td><td>89.47 ± 0.78</td><td>86.20 ± 1.89</td><td>91.91 ± 0.87</td><td>67.69 ± 1.40</td><td>83.91 ± 0.45</td></tr><tr><td rowspan="4">4-shot</td><td>Baseline</td><td>84.80 ± 2.06</td><td>88.07 ± 0.61</td><td>83.27 ± 1.82</td><td>88.89 ± 1.91</td><td>67.16 ± 1.47</td><td>81.52 ± 0.66</td></tr><tr><td>TaskV</td><td>88.70 ± 1.69</td><td>89.53 ± 1.34</td><td>86.27 ± 1.08</td><td>92.76 ± 0.54</td><td>70.44 ± 1.35</td><td>84.66 ± 0.39</td></tr><tr><td>TaskV-M</td><td>89.60 ± 1.43</td><td>91.00 ± 1.01</td><td>87.20 ± 0.62</td><td>92.36 ± 1.44</td><td>72.53 ± 0.94</td><td>85.64 ± 0.29</td></tr><tr><td>TaskV-MS</td><td>90.10 ± 1.39</td><td>90.67 ± 1.10</td><td>87.00 ± 1.17</td><td>92.22 ± 0.92</td><td>72.09 ± 1.46</td><td>85.45 ± 0.26</td></tr></table>

task vectors from each arrow token in a single few-shot prompt simultaneously. We name this alternative approach as TaskV-MS. As discussed in Proposition 3, the task vector weights that emerge at each arrow token are approximately orthonormal, suggesting they encode distinct information subsets and can be simultaneously injected to enhance model performance (e.g., by increasing the rank of the induced coefficient matrix W ). Table 7 shows a comparison between the current multi-vector method (TaskV-M) and this single-prompt variant (TaskV-MS).

While TaskV-MS also delivers strong performance, it slightly underperforms TaskV-M. We believe this is due to the causal attention mechanism in real LLMs, where earlier arrow tokens can only aggregate information from a subset of demonstrations. Nonetheless, TaskV-MS is a promising alternative for accelerating inference.

Table 8: Comparison of the accuracies of n-shot ICL and task vector on bijection tasks (n = 10). We use gray text to indicate accuracies lower than 60%. 

<table><tr><td rowspan="2">Task</td><td colspan="2">GPT-J</td><td colspan="2">Pythia-6.9B</td><td colspan="2">Pythia-12B</td><td colspan="2">Llama-7B</td><td colspan="2">Llama-13B</td><td colspan="2">Qwen3-8B</td><td colspan="2">Llama3-8B</td></tr><tr><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td></tr><tr><td>Lower ↔ Upper</td><td>1.00</td><td>0.08</td><td>0.90</td><td>0.28</td><td>0.96</td><td>0.24</td><td>1.00</td><td>0.55</td><td>1.00</td><td>0.58</td><td>1.00</td><td>0.56</td><td>1.00</td><td>0.38</td></tr><tr><td>English ↔ French</td><td>0.64</td><td>0.50</td><td>0.38</td><td>0.28</td><td>0.52</td><td>0.28</td><td>0.54</td><td>0.35</td><td>0.64</td><td>0.32</td><td>0.84</td><td>0.48</td><td>0.66</td><td>0.42</td></tr><tr><td>English ↔ Italian</td><td>0.68</td><td>0.56</td><td>0.62</td><td>0.48</td><td>0.60</td><td>0.56</td><td>0.70</td><td>0.47</td><td>0.72</td><td>0.44</td><td>0.68</td><td>0.36</td><td>0.70</td><td>0.36</td></tr><tr><td>English ↔ Spanish</td><td>0.70</td><td>0.52</td><td>0.62</td><td>0.56</td><td>0.66</td><td>0.56</td><td>0.64</td><td>0.43</td><td>0.84</td><td>0.56</td><td>0.70</td><td>0.32</td><td>0.72</td><td>0.32</td></tr><tr><td>Present ↔ Gerund</td><td>0.64</td><td>0.36</td><td>0.44</td><td>0.32</td><td>0.40</td><td>0.22</td><td>0.80</td><td>0.41</td><td>0.74</td><td>0.26</td><td>0.72</td><td>0.34</td><td>0.94</td><td>0.52</td></tr><tr><td>Present ↔ Past</td><td>0.60</td><td>0.38</td><td>0.48</td><td>0.36</td><td>0.54</td><td>0.16</td><td>0.52</td><td>0.33</td><td>0.68</td><td>0.44</td><td>0.78</td><td>0.42</td><td>0.90</td><td>0.58</td></tr><tr><td>Present ↔ Perfect</td><td>0.46</td><td>0.14</td><td>0.38</td><td>0.24</td><td>0.46</td><td>0.28</td><td>0.55</td><td>0.33</td><td>0.54</td><td>0.42</td><td>0.66</td><td>0.42</td><td>0.78</td><td>0.50</td></tr><tr><td>Singular ↔ Plural</td><td>0.66</td><td>0.50</td><td>0.56</td><td>0.28</td><td>0.44</td><td>0.28</td><td>0.76</td><td>0.51</td><td>0.80</td><td>0.52</td><td>0.84</td><td>0.58</td><td>0.88</td><td>0.58</td></tr><tr><td>Antonym</td><td>0.86</td><td>0.78</td><td>0.76</td><td>0.66</td><td>0.76</td><td>0.70</td><td>0.83</td><td>0.73</td><td>0.78</td><td>0.72</td><td>0.82</td><td>0.74</td><td>0.82</td><td>0.76</td></tr></table>

Table 9: Comparison of the accuracies of n-shot ICL and task vector on bijection tasks (n = 20). We use gray text to indicate accuracies lower than 60%. 

<table><tr><td rowspan="2">Task</td><td colspan="2">GPT-J</td><td colspan="2">Pythia-6.9B</td><td colspan="2">Pythia-12B</td><td colspan="2">Llama-7B</td><td colspan="2">Llama-13B</td><td colspan="2">Qwen3-8B</td><td colspan="2">Llama3-8B</td></tr><tr><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td><td>ICL</td><td>TV</td></tr><tr><td>Lower ↔ Upper</td><td>1.00</td><td>0.12</td><td>1.00</td><td>0.32</td><td>0.94</td><td>0.38</td><td>1.00</td><td>0.48</td><td>1.00</td><td>0.60</td><td>1.00</td><td>0.58</td><td>1.00</td><td>0.36</td></tr><tr><td>English ↔ French</td><td>0.74</td><td>0.54</td><td>0.44</td><td>0.40</td><td>0.52</td><td>0.40</td><td>0.52</td><td>0.34</td><td>0.58</td><td>0.34</td><td>0.58</td><td>0.30</td><td>0.74</td><td>0.28</td></tr><tr><td>English ↔ Italian</td><td>0.62</td><td>0.54</td><td>0.66</td><td>0.46</td><td>0.68</td><td>0.48</td><td>0.78</td><td>0.50</td><td>0.74</td><td>0.48</td><td>0.76</td><td>0.38</td><td>0.76</td><td>0.32</td></tr><tr><td>English ↔ Spanish</td><td>0.80</td><td>0.58</td><td>0.54</td><td>0.38</td><td>0.56</td><td>0.40</td><td>0.78</td><td>0.58</td><td>0.84</td><td>0.58</td><td>0.66</td><td>0.32</td><td>0.86</td><td>0.40</td></tr><tr><td>Present ↔ Gerund</td><td>0.54</td><td>0.26</td><td>0.54</td><td>0.22</td><td>0.46</td><td>0.14</td><td>0.84</td><td>0.44</td><td>0.94</td><td>0.38</td><td>0.88</td><td>0.28</td><td>0.98</td><td>0.52</td></tr><tr><td>Present ↔ Past</td><td>0.66</td><td>0.26</td><td>0.54</td><td>0.30</td><td>0.58</td><td>0.28</td><td>0.72</td><td>0.30</td><td>0.76</td><td>0.44</td><td>0.74</td><td>0.40</td><td>1.00</td><td>0.48</td></tr><tr><td>Present ↔ Perfect</td><td>0.42</td><td>0.18</td><td>0.44</td><td>0.20</td><td>0.46</td><td>0.24</td><td>0.48</td><td>0.30</td><td>0.52</td><td>0.48</td><td>0.80</td><td>0.44</td><td>0.90</td><td>0.48</td></tr><tr><td>Singular ↔ Plural</td><td>0.64</td><td>0.40</td><td>0.62</td><td>0.36</td><td>0.52</td><td>0.28</td><td>0.80</td><td>0.52</td><td>0.94</td><td>0.42</td><td>0.86</td><td>0.60</td><td>0.92</td><td>0.60</td></tr><tr><td>Antonym</td><td>0.84</td><td>0.76</td><td>0.84</td><td>0.70</td><td>0.90</td><td>0.82</td><td>0.90</td><td>0.84</td><td>0.90</td><td>0.84</td><td>0.84</td><td>0.74</td><td>0.84</td><td>0.76</td></tr></table>

# B.4 FURTHER RESULTS ON BIJECTION TASKS

Here, we extend the results from Table 1 that illustrate the failure of task vectors on bijection tasks across a broader range of LLMs and varying numbers of input demonstrations. We keep the same experimental settings as Table 1 while increasing the number of demonstrations to $n \in \{ 1 0 , 2 0 \}$ , and report the results for 7 distinct LLMs: GPT-J, Pythia-6.9B, Pythia-12B, Llama-7B, Llama-13B, Qwen3-8B and Llama3-8B. As shown in Tables 8 and 9, the task vector method results in a significant performance drop compared to the standard ICL on bijection tasks. These results further support our claims that:

• Task vectors systematically fail on bijection tasks, even when further increasing the number of demonstrations in the prompt.   
• The failure is consistent across multiple model architectures, validating that the issue stems from a fundamental expressiveness limitation rather than model-specific artifacts.

# B.5 FULL SALIENCY ANALYSIS RESULTS

In the main text, we reported a simplified version of the saliency map due to space limitations, focusing only on the demonstration tokens $x _ { i } ,  , y _ { i }$ . In Figure 5, we report the full saliency map covering every token in the prompt. Here, $\mathbf { \ddot { \delta } } \mathbf { \vec { B } } ^ { \prime }$ stands for the [BOS] token, and “E” stands for the word “Example”. Please refer to eq. (13) for further details about the structure of the input prompt. As can be seen, the highlighted saliency weights exhibit clear patterns of embedding concatenation and weighted summation. It can also be observed that latter demonstrations weigh more for task vector formation (i.e., saliency magnitudes for latter $y _ { i }$ tokens are larger in Figure 5b).

![](images/88a1c9c6dd80680255aea20fa52d005de97fb8b6f4edcd8edef5b7d1276f01c2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Input
        A["x1"] --> B["x2"]
        C["x3"] --> D["x4"]
        E["x5"] --> F["x6"]
        G["x7"] --> H["x8"]
        I["x9"] --> J["x10"]
    end
    subgraph Test
        K["x_test"] --> L["x_test"]
    end
    style Input fill:#f9f,stroke:#333
    style Test fill:#bbf,stroke:#333
```
</details>

(a) Full Saliency Map (l = 10)   
![](images/22d9273b1330e8d4449dadd979b9923e3346d3cb0dbfec4c568017ad294f689b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Red Dot"] --> B["Orange Dot"]
    B --> C["Gray Line"]
    C --> D["Blue Dot"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
```
</details>

(b) Full Saliency Map (l = 12)   
Figure 5: Visualization of full saliency matrices as bipartite graphs between layer l ( ) and $l + 1$ ( ), edge widths indicate saliency magnitude (Llama- $\cdot 7 \mathrm { B } , n = 1 0 )$ . (a) Each $y _ { i }$ token attends to its corresponding $( x _ { i } , y _ { i } )$ pair, reflecting embedding concatenation. (b) The final (→) token attends broadly to all $y _ { i }$ tokens, indicating task vector formation.

# C AUXILIARY LEMMAS

Lemma 8 (Proposed in (Ahn et al., 2023)). Given positive objective function $f ( A )$ taking parameters $A = \{ A _ { i } \} _ { i = 1 } ^ { n } ,$ , where $A _ { i } \in \mathbb { R } ^ { d _ { i } \times d _ { i } }$ . Let $\pmb { S } = \hat { \Pi } _ { i = 1 } ^ { n } \pmb { S } _ { i } \subset \dot { \Pi _ { i = 1 } ^ { n } } \mathbb { R } ^ { d _ { i } \times d _ { i } }$ be a predefined parameter subspace. Define $\widetilde A ( t , R _ { i } ) = \{ A _ { 1 } , \cdots , A _ { i } + t R _ { i } , \cdots , A _ { n } \}$ given $i \in [ 1 , n ] , R _ { i } \in \mathbb { R } ^ { d _ { i } \times d _ { i } }$ and $t \in \mathbb { R } .$ . If for any $A \in S$ and $R _ { i } \in \mathbb { R } ^ { d _ { i } \times d _ { i } }$ , there exists $\widetilde { R } _ { i } \in { S } _ { i }$ such that

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} f \Big (\widetilde {A} (t, \widetilde {R} _ {i}) \Big) \right| _ {t = 0} \leq \left. \frac {\mathrm{d}}{\mathrm{d} t} f \Big (\widetilde {A} (t, R _ {i}) \Big) \right| _ {t = 0},
$$

then we have

$$
\inf _ {A \in \mathcal {S}} \sum_ {i = 1} ^ {n} \| \nabla_ {A _ {i}} f (A) \| _ {F} ^ {2} = 0.
$$

Proof. This lemma is proved as part of the main theorems in (Ahn et al., 2023). We rearrange the proof here to accommodate arbitrary function of matrices. Firstly, notice that for any $R =$ $\{ R _ { i } \} _ { i = 1 } ^ { n } \in \Pi _ { i = 1 } ^ { n } \mathbb { R } ^ { d _ { i } \times d _ { i } }$ ,

$$
\sum_ {i = 1} ^ {n} \frac {\mathrm{d}}{\mathrm{d} t} f (\widetilde {A} (t, R _ {i})) \bigg | _ {t = 0} = \left. \frac {\mathrm{d}}{\mathrm{d} t} f (A + t R) \right| _ {t = 0}.
$$

Therefore, the provided precondition is equivalent to stating that for any $A \in \mathcal { S }$ and $R \in$ $\Pi _ { i = 1 } ^ { n } \mathbb { R } ^ { d _ { i } \times d _ { i } }$ , there exists ${ \widetilde { R } } \in { \mathcal { S } }$ such that:

$$
\frac {\mathrm{d}}{\mathrm{d} t} f \Big (A + t \widetilde {R} \Big) \bigg | _ {t = 0} \leq \frac {\mathrm{d}}{\mathrm{d} t} f (A + t R) \bigg | _ {t = 0}.
$$

Let $R = - \nabla _ { A } f ( A )$ , we then have

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} f (A + t R) \right| _ {t = 0} = \left. \left\langle \frac {\mathrm{d} f (A - t \nabla_ {A} f (A))}{\mathrm{d} (A - t \nabla_ {A} f (A))}, \frac {\mathrm{d} (A - t \nabla_ {A} f (A))}{t} \right\rangle \right| _ {t = 0}
$$

$$
= \left\langle \nabla_ {A} f (A), - \nabla_ {A} f (A) \right\rangle = - \left\| \nabla_ {A} f (A) \right\| _ {F} ^ {2}.
$$

If the infimum of $\big \| \nabla _ { A } f ( A ) \big \| _ { F } ^ { 2 }$ is not zero but some positive value $p ,$ then the S-constrained gradient flow induced by R will lead to unbounded descent:

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} f (A + t \widetilde {R}) \right| _ {t = 0} \leq - p.
$$

This contradicts the fact that $f ( A ) \geq 0$ and concludes the proof.

![](images/34d1e69e2b7cd184e67a2a5f8606d4a2cb4a2b566647dc6d8882c4452eadd5b4.jpg)

The following lemma is an extension of Lemma 5 in (Ahn et al., 2023) by accommodating multivariate $y$ samples as well as enabling a wider range of demonstration and transformer parameter configurations.

Lemma 9. Let $x _ { 1 } , \cdots , x _ { n + 1 }$ be i.i.d. samples from an input distribution, and let W be sampled independently $o f \{ x _ { i } \} _ { i = 1 } ^ { n + 1 }$ . Let $Z _ { 0 } \in \mathbb { R } ^ { ( 2 d ) \times N }$ , where $N \in \mathbb { Z } ,$ , be constructed of form

$$
Z _ {0} = \left[ \begin{array}{c c c c} * & \dots & * & * \\ * & \dots & * & 0 _ {d} \end{array} \right] \in \mathbb {R} ^ {(2 d) \times N},
$$

where the ∗ parts can be arbitrarily constructed from $\{ x _ { i } \} _ { i = 1 } ^ { n + 1 }$ and W . Let $\widetilde { Z } _ { 0 }$ be defined as replacing the zero part of $Z _ { 0 }$ by $y _ { n + 1 } .$ :

$$
\widetilde {Z} _ {0} = \left[ \begin{array}{c c c c} * & \dots & * & * \\ * & \dots & * & y _ {n + 1} \end{array} \right] \in \mathbb {R} ^ {(2 d) \times N}.
$$

Let $\widetilde { Z } _ { l }$ be the output of the l-th layer of the linear transformer, and let $\boldsymbol { \widetilde { X } } _ { l } , \boldsymbol { \widetilde { Y } } _ { l } \in \mathbb { R } ^ { d \times N }$ be the first and last d rows of $\cdot \widetilde { Z } _ { l }$ , respectively. Suppose that the $\{ Q _ { l } \} _ { l = } ^ { L }$ 1 matrices are of form

$$
Q _ {l} = \left[ \begin{array}{c c c} \underbrace {*} _ {d   c o l u m n s} & 0 _ {(2 d + d _ {p}) \times d} & \underbrace {*} _ {d _ {p}   c o l u m n s} \end{array} \right],
$$

Then the in-context risk of this L-layer linear transformer is equivalent to

$$
\mathcal {L} \big (\{V _ {l}, Q _ {l} \} _ {l = 1} ^ {L} \big) = \mathbb {E} _ {\widetilde {Z} _ {0}, W} \Big [ \mathrm{tr} \Big ((I _ {N} - M) \widetilde {Y} _ {L} ^ {\top} \widetilde {Y} _ {L} (I _ {N} - M) \Big) \Big ]. \tag {14}
$$

Proof. Let the $V _ { l }$ and $Q _ { l }$ matrices be represented as:

$$
V _ {l} = \left[ \begin{array}{c} V _ {l} ^ {1} \\ V _ {l} ^ {2} \end{array} \right], \quad Q _ {l} = \left[ \begin{array}{c c c} Q _ {l} ^ {1} & 0 & Q _ {l} ^ {2} \end{array} \right],
$$

where $V _ { l } ^ { 1 } , V _ { l } ^ { 2 } \in \mathbb { R } ^ { d \times 2 d } , Q _ { l } ^ { 1 } \in \mathbb { R } ^ { ( 2 d + d _ { p } ) \times d } , Q _ { l } ^ { 2 } \in \mathbb { R } ^ { ( 2 d + d _ { p } ) \times d _ { p } }$ . Then the update rule in eq. (5) can be rephrased as

$$
X _ {l} = X _ {l - 1} + \frac {1}{n} V _ {l} ^ {1} Z _ {l - 1} M \left[ Z _ {l - 1} ^ {\top}, P \right] \left(Q _ {l} ^ {1} X _ {l - 1} + Q _ {l} ^ {2} P\right),
$$

$$
Y _ {l} = Y _ {l - 1} + \frac {1}{n} V _ {l} ^ {2} Z _ {l - 1} M \left[ Z _ {l - 1} ^ {\top}, P \right] \left(Q _ {l} ^ {1} X _ {l - 1} + Q _ {l} ^ {2} P\right).
$$

Let $\Delta _ { Z } = \widetilde { Z } _ { 0 } - Z _ { 0 }$ , i.e. an all-zero matrix except that the last half of the last column is $y _ { n + 1 }$ . Let $\Delta _ { X }$ and $\Delta _ { Y }$ be its first and last d rows respectively, then $\Delta _ { X } = 0$ and $\Delta _ { Y } = [ 0 $ · · · 0 $y _ { n + 1 } ]$ . Note that $\widetilde { Z } _ { l } = Z _ { l } + \Delta _ { Z }$ holds for $l = 0$ trivially. Now suppose it holds for some $l = k - 1$ , then

$$
\begin{array}{l} \widetilde {X} _ {k} = \widetilde {X} _ {k - 1} + \frac {1}{n} V _ {k} ^ {1} \widetilde {Z} _ {k - 1} M \left[ \widetilde {Z} _ {k - 1} ^ {\top}, P \right] \left(Q _ {k} ^ {1} \widetilde {X} _ {k - 1} + Q _ {k} ^ {2} P\right) \\ = X _ {k - 1} + \frac {1}{n} V _ {k} ^ {1} Z _ {k - 1} M \left[ Z _ {k - 1} ^ {\top}, P \right] \left(Q _ {k} ^ {1} X _ {k - 1} + Q _ {k} ^ {2} P\right) \\ + \frac {1}{n} V _ {k} ^ {1} \Delta_ {Z} M \left[ Z _ {k - 1} ^ {\top}, P \right] \left(Q _ {k} ^ {1} X _ {k - 1} + Q _ {k} ^ {2} P\right) \\ + \frac {1}{n} V _ {k} ^ {1} Z _ {k - 1} M \left[ \Delta_ {Z} ^ {\top}, 0 _ {d _ {p} \times d _ {p}} \right] \left(Q _ {k} ^ {1} X _ {k - 1} + Q _ {k} ^ {2} P\right) \\ \end{array}
$$

$$
+ \frac {1}{n} V _ {k} ^ {1} \Delta_ {Z} M \left[ \Delta_ {Z} ^ {\top}, 0 _ {d _ {p} \times d _ {p}} \right] \left(Q _ {k} ^ {1} X _ {k - 1} + Q _ {k} ^ {2} P\right)
$$

$$
= X _ {k - 1} + \frac {1}{n} V _ {k} ^ {1} Z _ {k - 1} M \left[ Z _ {k - 1} ^ {\top}, P \right] \left(Q _ {k} ^ {1} X _ {k - 1} + Q _ {k} ^ {2} P\right) = X _ {k},
$$

where the last step holds by noticing that $\Delta _ { Z } M = 0$ . Similarly, one can prove that

$$
\widetilde {Y} _ {k} = Y _ {k - 1} + \Delta_ {Y} + \frac {1}{n} V _ {k} ^ {2} Z _ {k - 1} M \big [ Z _ {k - 1} ^ {\top}, P \big ] \big (Q _ {k} ^ {1} X _ {k - 1} + Q _ {k} ^ {2} P \big) = Y _ {k} + \Delta_ {Y}.
$$

Therefore, it holds that for any $l \in [ 1 , L ] , \widetilde { Z } _ { l } = Z _ { l } + \Delta _ { Z }$ . Recall the in-context risk in eq. (2):

$$
\mathcal {L} \big (\{V _ {l}, Q _ {l} \} _ {l = 1} ^ {L} \big) = \mathbb {E} _ {Z _ {0}, W} \left\| (Z _ {L}) _ {(d + 1: 2 d), N} + y _ {n + 1} \right\| _ {2} ^ {2}
$$

$$
= \mathbb {E} _ {Z _ {0}, W} \| (Y _ {L} + \Delta_ {Y}) (I _ {N} - M) \| _ {2} ^ {2}
$$

$$
= \mathbb {E} _ {\widetilde {Z} _ {0}, W} \left[ \operatorname{tr} \left((I _ {N} - M) \widetilde {Y} _ {L} ^ {\top} \widetilde {Y} _ {L} (I _ {N} - M)\right) \right].
$$

The proof is complete.

![](images/49d24efdbc501f7ff4d174898ab169476bdccd103b46a3c106bc13711337e1ef.jpg)

# D PROOF OF THEORETICAL RESULTS

# D.1 PROOF OF PROPOSITION 4

Proof. We will first prove sufficiency. Let $W = a b ^ { \top }$ be a rank-one matrix, where $a , b \in \mathbb { R } ^ { d }$ . The given conditions imply that $x = W \bar { y } = W W x = a b ^ { \top } a b ^ { \top } x$ , we then have $b ^ { \top } x = b ^ { \top } a b ^ { \top } a b ^ { \top } x =$ $\overline { { ( b ^ { \intercal } a ) ^ { 2 } b ^ { \intercal } } } _ { x }$ . Since $\bar { b ^ { \intercal } } \bar { x } \neq 0$ , we can conclude that $b ^ { \top } a = \pm 1$ . Then, $x = a b ^ { \top } a b ^ { \top } x = \pm a b ^ { \top } x =$ $\pm y .$

To prove the necessity, it suffices to show that selecting $W = x x ^ { \top } / \| x \| _ { 2 } ^ { 2 }$ when $x = y$ satisfies the given conditions (alternatively, select $W = - x x ^ { \top } / \| x \| _ { 2 } ^ { 2 }$ when $x = - y )$ . □

# D.2 PROOF OF THEOREM 1

Proof. To enhance the readability of the notations in this proof, we will drop the constant $\textstyle { \frac { 1 } { n } }$ factor in linear attention. Furthermore, we will simplify $\widetilde { Z } _ { 0 } , \widetilde { X } _ { 0 }$ and $\widetilde { Y } _ { 0 }$ in Lemma 9 as $Z _ { 0 } , X _ { 0 }$ and $Y _ { 0 }$ respectively. This results in different definitions compared to the original ones, but we will not refer to the original definitions in the remainder of this proof.

$$
Z _ {0} = \left[ \begin{array}{c} X _ {0} \\ Y _ {0} \end{array} \right] = \left[ \begin{array}{c c c c c c c} x _ {1} & 0 & \dots & x _ {n} & 0 & x _ {\text {test}} & 0 \\ 0 & y _ {1} & \dots & 0 & y _ {n} & 0 & y _ {\text {test}} \end{array} \right] \in \mathbb {R} ^ {(2 d) \times (2 n + 2)}.
$$

Let $Z _ { l }$ be the output of the l-th layer of the transformer, and let $X _ { l } , Y _ { l } \in \mathbb { R } ^ { d \times ( 2 n + 2 ) }$ denote the first and last d rows of $Z _ { l } .$ , respectively. Under the constraint in eq. (6), we can verify that

$$
X _ {l} = X _ {l - 1} + A _ {l} X _ {l - 1} M \left(X _ {l - 1} ^ {\top} C _ {l} X _ {l - 1} + D _ {l}\right), \tag {15}
$$

$$
Y _ {l} = Y _ {l - 1} + B _ {l} Y _ {l - 1} M (X _ {l - 1} ^ {\top} C _ {l} X _ {l - 1} + D _ {l}).
$$

In the following analysis, we will use $f ( A  B )$ to denote the result of the function f of A when replacing the value of A with B. Additionally, we denote $f ( A \gets B * A )$ as $f ( A \stackrel { * } {  } B )$ for any operator ∗. Therefore, $f ( A \stackrel { + } {  } B ) = f ( A  A + B )$ ). We also denote $f ( A \stackrel { \times } {  } B ) = f ( A  B A )$ and $f ( A \stackrel { \circ } {  } B ) = f ( A  A B )$ for convenience.

Our goal is proving that, for any $E \in A \cup B \cup C \cup D$ and an arbitrary matrix $R \in \mathbb { R } ^ { d \times d } ( \mathbb { R } ^ { d _ { p } \times d _ { \tau } }$ for D), there exists $\widetilde { R } \in S _ { I } \left( S _ { \Sigma } \right.$ for $C , S _ { P }$ for D) such that

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (E \stackrel {+} {\leftarrow} t \widetilde {R}) \right| _ {t = 0} \leq \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (E \stackrel {+} {\leftarrow} t R) \right| _ {t = 0}. \tag {16}
$$

Let $\overline { { X } } _ { 0 } = [ 0 , x _ { 1 } , \cdot \cdot \cdot , 0 , x _ { \mathrm { t e s t } } ]$ be a function of $X _ { 0 } .$ , we then have $Y _ { 0 } = W \overline { { { X } } } _ { 0 }$ . Let $U _ { \perp } \in \mathbb { R } ^ { d \times d }$ be a uniformly sampled random orthonormal matrix, and let $U _ { \Sigma } = \Sigma ^ { 1 / 2 } U _ { \bot } \Sigma ^ { - 1 / 2 }$ . One can verify that $\begin{array} { r } { U _ { \Sigma } ^ { - 1 } = \Sigma ^ { 1 / 2 } U _ { \iota } ^ { \top } \Sigma ^ { - 1 / 2 } } \end{array}$ . By applying Lemma 9 and the fact that $X _ { 0 } \overset { d } { = } U _ { \Sigma } X _ { 0 }$ , we have that for any given matrix $\bar { R , }$

$$
\begin{array}{l} \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (E \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathbb {E} _ {X _ {0}, W} \Big [ \mathrm{tr} \Big ((I - M) Y _ {L} ^ {\top} (E \stackrel {+} {\leftarrow} t R) Y _ {L} (E \stackrel {+} {\leftarrow} t R) (I - M) \Big) \Big ] \right| _ {t = 0} \\ = 2 \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} \left. \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} (E \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}, W, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, E \stackrel {{+}} {{\leftarrow}} t R) \Big | _ {t = 0} (I - M)\right) \right]. \\ \end{array}
$$

Next, we will show that eq. (16) holds for each one of $A _ { i } , B _ { i } , C _ { i } , D _ { i }$ for any $i \in [ 1 , L ]$ .

# 1. Equation (16) holds for $A _ { i } .$ .

We first show that for any $l \in [ 1 , L ]$ , the following equations hold:

$$
X _ {l} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) = U _ {\Sigma} X _ {l}, \tag {17}
$$

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, A _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0}. \tag {18}
$$

It is straightforward to verify that eq. (17) holds for $l = 0$ . Now suppose that eq. (17) holds for some $l = k - 1$ , we then have

$$
\begin{array}{l} X _ {k} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) \\ = X _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) + A _ {l} X _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) M \Big (X _ {k - 1} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) C _ {l} X _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) + D _ {l} \Big) \\ = U _ {\Sigma} X _ {k - 1} + A _ {l} U _ {\Sigma} X _ {k - 1} M \left(X _ {k - 1} ^ {\top} U _ {\Sigma} ^ {\top} C _ {l} U _ {\Sigma} X _ {k - 1} + D _ {l}\right) \\ = U _ {\Sigma} \left(X _ {k - 1} + A _ {l} X _ {k - 1} M \left(X _ {k - 1} ^ {\top} C _ {l} X _ {k - 1} + D _ {l}\right)\right) = U _ {\Sigma} X _ {k}, \\ \end{array}
$$

where the third equality follows by noticing that when $A _ { l } \ = \ a _ { l } I _ { d }$ and $C _ { l } \ = \ c _ { l } \Sigma ^ { - 1 }$ , we have $A _ { l } U _ { \Sigma } = U _ { \Sigma } A _ { l }$ and $U _ { \Sigma } ^ { \top } \dot { C } _ { l } U _ { \Sigma } = C _ { l }$ . This concludes the proof of eq. (17).

We now turn to the proof of eq. (18). Notice that when $l < i ,$ we naturally have

$$
\frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \bigg | _ {t = 0} = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (A _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} = 0.
$$

When $l = i ,$ it is easy to verify that

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \bigg | _ {t = 0} = R U _ {\Sigma} X _ {l - 1} M (X _ {l - 1} ^ {\top} U _ {\Sigma} ^ {\top} C _ {l} U _ {\Sigma} X _ {l - 1} + D _ {l}) \\ = U _ {\Sigma} \cdot U _ {\Sigma} ^ {- 1} R U _ {\Sigma} M (X _ {l - 1} ^ {\top} C _ {l} X _ {l - 1} + D _ {l}) \\ = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0}. \\ \end{array}
$$

Now suppose that eq. (18) holds for some $l = k - 1 \geq i$ , one can verify that:

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} X _ {k} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, A _ {i} \stackrel {+} {\leftarrow} t R) \bigg | _ {t = 0} \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, A _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} + \frac {\mathrm{d}}{\mathrm{d} t} A _ {k} X _ {k - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, A _ {i} \stackrel {+} {\leftarrow} t R) M \\ \cdot \left(X _ {k - 1} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) C _ {k} X _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) + D _ {k}\right) \bigg | _ {t = 0} \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, A _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} \\ \end{array}
$$

$$
\begin{array}{l} + A _ {k} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \right| _ {t = 0} M \Big (X _ {k - 1} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) C _ {k} X _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) + D _ {k} \Big) \\ + A _ {k} X _ {k - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) M \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} ^ {\top} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, A _ {i} \stackrel {\pm} {\leftarrow} t R) \right| _ {t = 0} C _ {k} X _ {k - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) \\ \left. + A _ {k} X _ {k - 1} \left(X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}\right) M X _ {k - 1} ^ {\top} \left(X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}\right) C _ {k} \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} \left(X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R\right) \right| _ {t = 0} \\ = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} \\ + U _ {\Sigma} A _ {k} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} M \big (X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} + D _ {k} \big) \\ + U _ {\Sigma} A _ {k} X _ {k - 1} M \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} ^ {\top} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} C _ {k} X _ {k - 1} \\ \left. + U _ {\Sigma} A _ {k} X _ {k - 1} M X _ {k - 1} ^ {\top} C _ {k} \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} (A _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} \\ = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} + U _ {\Sigma} \frac {\mathrm{d}}{\mathrm{d} t} A _ {k} X _ {k - 1} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) M \\ \cdot \left. \left(X _ {k - 1} ^ {\top} (A _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) C _ {k} X _ {k - 1} (A _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) + D _ {k}\right) \right| _ {t = 0} \\ = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k} (A _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0}. \\ \end{array}
$$

This completes the proof of eq. (18).

Under the condition that $B _ { l } = b _ { l } I _ { d }$ for some $b _ { l } \in \mathbb { R }$ , we can simplify eq. (15) as

$$
Y _ {l} = Y _ {l - 1} + b _ {l} Y _ {l - 1} M \left(X _ {l - 1} ^ {\top} C _ {l} X _ {l - 1} + D _ {l}\right)
$$

$$
= Y _ {l - 1} \left(I + b _ {l} M \left(X _ {l - 1} ^ {\top} C _ {l} X _ {l - 1} + D _ {l}\right)\right)
$$

$$
= Y _ {0} \prod_ {j = 1} ^ {l} \left(I + b _ {j} M \left(X _ {j - 1} ^ {\top} C _ {j} X _ {j - 1} + D _ {j}\right)\right).
$$

Define $\begin{array} { r } { G _ { l } = \overline { { X } } _ { 0 } \prod _ { j = 1 } ^ { l } \bigl ( I + b _ { j } M ( X _ { j - 1 } ^ { \top } C _ { j } X _ { j - 1 } + D _ { j } ) \bigr ) } \end{array}$ , then it satisfies that $Y _ { l } = W G _ { l }$ . We are ready to prove that similar results to eqs. (17) and (18) also hold for $G _ { l } , l \in [ 1 , L ]$ :

$$
G _ {l} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) = U _ {\Sigma} G _ {l}, \tag {19}
$$

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, A _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0}. \tag {20}
$$

Notice that eq. (19) holds trivially for $l = 0 \mathrm { a s } G _ { 0 } = \overline { { X } } _ { 0 }$ . Now suppose that eq. (19) holds for some $l = k - 1$ , we then have

$$
G _ {k} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) = G _ {k - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) \Big (I + b _ {k} M (X _ {k - 1} ^ {\top} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) C _ {k} X _ {k - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) + D _ {k}) \Big)
$$

$$
= U _ {\Sigma} G _ {k - 1} \big (I + b _ {k} M (X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} + D _ {k}) \big) = U _ {\Sigma} G _ {k}.
$$

This concludes eq. (19). As for eq. (20), notice that both sides equal 0 when $l \leq i .$ . Now suppose that eq. (20) holds for some $l = k - 1 \geq i$ , we then have:

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} G _ {k} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \bigg | _ {t = 0} \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \right| _ {t = 0} + \frac {\mathrm{d}}{\mathrm{d} t} b _ {k} G _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) M \\ \cdot \left(X _ {k - 1} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) C _ {k} X _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) + D _ {k}\right) \bigg | _ {t = 0} \\ \end{array}
$$

$$
\begin{array}{l} = \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {k - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, A _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} \\ + b _ {k} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \right| _ {t = 0} M \Big (X _ {k - 1} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) C _ {k} X _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) + D _ {k} \Big) \\ + b _ {k} G _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) M \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \right| _ {t = 0} C _ {k} X _ {k - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) \\ \left. + b _ {k} G _ {k - 1} \left(X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}\right) M X _ {k - 1} ^ {\top} \left(X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}\right) C _ {k} \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} \left(X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R\right) \right| _ {t = 0} \\ = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {k - 1} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} \\ + b _ {k} U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {k - 1} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} M \big (X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} + D _ {k} \big) \\ + b _ {k} U _ {\Sigma} G _ {k - 1} M \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} ^ {\top} (A _ {i} \stackrel {{\pm}} {{\leftarrow}} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} C _ {k} X _ {k - 1} \\ \left. + b _ {k} U _ {\Sigma} G _ {k - 1} M X _ {k - 1} ^ {\top} C _ {k} \frac {\mathrm{d}}{\mathrm{d} t} X _ {k - 1} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} \\ = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {k} (A _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0}. \\ \end{array}
$$

This concludes the proof of eq. (20). Consider the in-context risk:

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (A _ {i} \stackrel {+} {\leftarrow} t R) \bigg | _ {t = 0} \\ = 2 \mathbb {E} _ {X _ {0}, W, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \Big | _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}, W, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} U _ {\Sigma} ^ {\top} W ^ {\top} W U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} \mathbb {E} _ {U _ {\perp}} \left[ G _ {L} (A _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right] \Big | _ {t = 0} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (A _ {i} \stackrel {{+}} {{\leftarrow}} \mathbb {E} _ {U _ {\perp}} [ t U _ {\Sigma} ^ {- 1} R U _ {\Sigma} ]) \Big | _ {t = 0} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (A _ {i} \stackrel {+} {\leftarrow} t r I _ {d}) \Bigg | _ {t = 0} (I - M)\right) \right] \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} (A _ {i} \stackrel {{+}} {{\leftarrow}} t r I _ {d}) Y _ {L} (A _ {i} \stackrel {{+}} {{\leftarrow}} t r I _ {d}) (I - M)\right) \right] \right| _ {t = 0} \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (A _ {i} \stackrel {+} {\leftarrow} t r I _ {d}) \right| _ {t = 0}, \\ \end{array}
$$

where $r = \mathbb { E } _ { U _ { \perp } } [ U _ { \Sigma } ^ { - 1 } R U _ { \Sigma } ] = { \frac { 1 } { d } } \operatorname { t r } \left( \Sigma ^ { - 1 / 2 } R \Sigma ^ { 1 / 2 } \right)$ , and we used the fact that $U _ { \Sigma } ^ { \top } \Sigma ^ { - 1 } U _ { \Sigma } = \Sigma ^ { - 1 }$ , and $\begin{array} { r } {  \frac { \mathrm { d } } { \mathrm { d } t } G _ { L } ( A _ { i } \stackrel { + } {  } t R ) | _ { t = 0 } } \end{array}$ is affine in R. This concludes that eq. (16) holds for $A _ { i } , i \in [ 1 , L ]$ .

# 2. Equation (16) holds for $B _ { i } .$

From the recursive expressions in eq. (15), we can conclude that the values of $X _ { l }$ do not depend on $B _ { i }$ . Therefore, we naturally have

$$
X _ {l} (B _ {i} \stackrel {+} {\leftarrow} t R) = X _ {l}. \tag {21}
$$

Next, we would like to show that for any $l \in [ 1 , L ]$ ,

$$
\mathbb {E} _ {W} \left[ W ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} Y _ {l} (B _ {i} \stackrel {+} {\leftarrow} t R) \bigg | _ {t = 0} \right] = \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (b _ {i} \stackrel {+} {\leftarrow} t \operatorname{tr} (R)) \bigg | _ {t = 0}. \tag {22}
$$

When $l < i ,$ we can easily verify eq. (22) since both sides equal 0. When $l = i ,$ , we can get

$$
\begin{array}{l} \mathbb {E} _ {W} \left[ W ^ {\top} \left. \frac {\mathrm{d}}{\mathrm{d} t} Y _ {l} (B _ {i} \stackrel {\pm} {\leftarrow} t R) \right| _ {t = 0} \right] = \mathbb {E} _ {W} \left[ W ^ {\top} R Y _ {l - 1} M \big (X _ {l - 1} ^ {\top} C _ {l} X _ {l - 1} + D _ {l} \big) \right] \\ = \mathbb {E} _ {W} \left[ W ^ {\top} R W \right] G _ {l - 1} M \left(X _ {l - 1} ^ {\top} C _ {l} X _ {l - 1} + D _ {l}\right) \\ = \mathrm{tr} (R) \Sigma^ {- 1} G _ {l - 1} M \big (X _ {l - 1} ^ {\top} C _ {l} X _ {l - 1} + D _ {l} \big) \\ = \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (b _ {i} \stackrel {+} {\leftarrow} t \operatorname{tr} (R)) \right| _ {t = 0}. \\ \end{array}
$$

Suppose that eq. (22) holds for some $l = k - 1 \geq i .$ . One can then verify

$$
\begin{array}{l} \mathbb {E} _ {W} \left[ W ^ {\top} \left. \frac {\mathrm{d}}{\mathrm{d} t} Y _ {k} (B _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} \right] \\ = \mathbb {E} _ {W} \left[ W ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} Y _ {k - 1} \left(B _ {i} \stackrel {{+}} {{\leftarrow}} t R\right) \left(I + b _ {k} M \left(X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} + D _ {k}\right)\right) \Bigg | _ {t = 0} \right] \\ = \mathbb {E} _ {W} \left[ W ^ {\top} \left. \frac {\mathrm{d}}{\mathrm{d} t} Y _ {k - 1} \left(B _ {i} \stackrel {{+}} {{\leftarrow}} t R\right) \right| _ {t = 0} \right] \left(I + b _ {k} M \left(X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} + D _ {k}\right)\right) \\ = \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {k - 1} (b _ {i} \stackrel {+} {\leftarrow} t \operatorname{tr} (R)) \right| _ {t = 0} \left(I + b _ {k} M (X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} + D _ {k})\right) \\ = \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {k} (b _ {i} \stackrel {+} {\leftarrow} t \operatorname{tr} (R)) \right| _ {t = 0}. \\ \end{array}
$$

The proof of eq. (22) is complete. Now, look at the in-context risk, we have

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (B _ {i} \stackrel {{+}} {{\leftarrow}} t R) \bigg | _ {t = 0} = 2 \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} (B _ {i} \stackrel {{+}} {{\leftarrow}} t R) \bigg | _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \mathbb {E} _ {W} \left[ W ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} (B _ {i} \stackrel {+} {\leftarrow} t R) \Big | _ {t = 0} \right] (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (b _ {i} \stackrel {+} {\leftarrow} t \operatorname{tr} (R)) \right| _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} (B _ {i} \stackrel {{+}} {{\leftarrow}} t \operatorname{tr} (R) I _ {d}) \Bigg | _ {t = 0} (I - M)\right) \right] \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (B _ {i} \stackrel {+} {\leftarrow} t \operatorname{tr} (R) I _ {d}) \right| _ {t = 0}. \\ \end{array}
$$

This concludes that eq. (16) holds for $B _ { i } , i \in [ 1 , L ]$ .

# 3. Equation (16) holds for $C _ { i } .$ .

Similar to the $A _ { i }$ case, we will first prove that for any $l \in [ 1 , L ]$ ,

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, C _ {i} \stackrel {\pm} {\leftarrow} t R) \right| _ {t = 0} = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (C _ {i} \stackrel {\pm} {\leftarrow} t U _ {\Sigma} ^ {\top} R U _ {\Sigma}) \right| _ {t = 0}. \tag {23}
$$

The equation above holds trivially for $l < i .$ For the case $l = i ,$ we have

$$
\begin{array}{l} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, C _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} \\ = A _ {j} X _ {l - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) M X _ {l - 1} ^ {\top} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) R X _ {l - 1} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}) \\ = U _ {\Sigma} A _ {j} X _ {l - 1} M X _ {l - 1} ^ {\top} U _ {\Sigma} ^ {\top} R U _ {\Sigma} X _ {l - 1} = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (C _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\Sigma} ^ {\top} R U _ {\Sigma}) \right| _ {t = 0}. \\ \end{array}
$$

One can conclude the proof of eq. (23) through a similar reduction as eq. (18) for $l > i$ layers. Next, we establish the corresponding result for $G _ { l }$ :

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, C _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (C _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {\top} R U _ {\Sigma}) \right| _ {t = 0}. \tag {24}
$$

This equation holds trivially for $l < i .$ . When taking $l = i ,$ , we can verify that

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, C _ {i} \stackrel {{+}} {{\leftarrow}} t R) \bigg | _ {t = 0} = b _ {l} G _ {l - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) M X _ {l - 1} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) R X _ {l - 1} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) \\ = b _ {l} U _ {\Sigma} G _ {l - 1} \left(X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}\right) M X _ {l - 1} ^ {\top} U _ {\Sigma} ^ {\top} R U _ {\Sigma} X _ {l - 1} \\ = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (C _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {\top} R U _ {\Sigma}) \right| _ {t = 0}. \\ \end{array}
$$

For $l > i$ layers, one can follow similar reductions as eq. (20) to finish the proof. We then consider the in-context risk:

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (C _ {i} \stackrel {+} {\leftarrow} t R) \bigg | _ {t = 0} \\ = 2 \mathbb {E} _ {X _ {0}, W, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} (X _ {0} \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, C _ {i} \stackrel {{+}} {{\leftarrow}} t R) \Big | _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}, W, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} U _ {\Sigma} ^ {\top} W ^ {\top} W U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (C _ {i} \stackrel {{+}} {{\leftarrow}} t R) \right| _ {t = 0} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} \mathbb {E} _ {U _ {\perp}} \left[ G _ {L} (C _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {\top} R U _ {\Sigma}) \right] \Bigg | _ {t = 0} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (C _ {i} \stackrel {{+}} {{\leftarrow}} t r \Sigma^ {- 1}) \Big | _ {t = 0} (I - M)\right) \right] \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} (C _ {i} \stackrel {+} {\leftarrow} t r \Sigma^ {- 1}) Y _ {L} (C _ {i} \stackrel {+} {\leftarrow} t r \Sigma^ {- 1}) (I - M)\right) \right] \right| _ {t = 0} \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (C _ {i} \stackrel {+} {\leftarrow} t r \Sigma^ {- 1}) \right| _ {t = 0}, \\ \end{array}
$$

where $\begin{array} { r } { r = \mathbb { E } _ { U _ { \Sigma } } [ U _ { \Sigma } ^ { \top } R U _ { \Sigma } ] = \frac { 1 } { d } \operatorname { t r } \left( \Sigma ^ { 1 / 2 } R \Sigma ^ { 1 / 2 } \right) } \end{array}$ . This concludes that eq. (16) holds for $C _ { i }$

# 4. Equation (16) holds for $D _ { i }$ .

Let $U _ { p } \in \mathbb { R } ^ { n \times n }$ be a uniformly sampled permutation matrix, i.e., a binary matrix that has exactly one 1 entry in each row and column with all other entries 0. Let $U _ { \circ } ~ \stackrel { \cdot } { = } ~ \mathrm { d i a g } ( U _ { p } \otimes I _ { 2 } , I _ { 2 } ) ~ \stackrel { \cdot } { \in }$ $\mathbb { R } ^ { ( 2 n + 2 ) \times ( 2 n + 2 ) }$ . One can verify that by multiplying $X _ { 0 } U _ { \circ }$ , it is equal to shuffling the first $n \ : 2 \cdot$ column sub-blocks of $X _ { 0 }$ and keeping the last 2 columns unchanged.

Then, consider a matrix $U _ { \xi } = \mathrm { d i a g } ( \xi _ { 1 } , \dots , \xi _ { n + 1 } ) \in \mathbb { R } ^ { ( n + 1 ) \times ( n + 1 ) }$ where $\xi _ { i } \stackrel { \mathrm { i . i . d . } } { \sim }$ Unif {±1}, i.e., a diagonal matrix with random ±1 entries. Let $U _ { \pm } = U _ { \xi } \otimes I _ { 2 } \in \mathbb { R } ^ { ( 2 n + 2 ) \times ( 2 n + 2 ) }$ . Thus, $U _ { \pm } = U _ { \pm } ^ { \top }$ and $X _ { 0 } U _ { \pm }$ is randomly flipping the sign of each 2-column sub-block in $X _ { 0 }$ .

We are going to prove that for any $l \in [ 1 , L ]$ , recalling that $f ( A \stackrel { \circ } {  } B ) = f ( A  A B )$ ,

$$
X _ {l} (X _ {0} \stackrel {\diamond} {\leftarrow} U _ {\pm} U _ {\circ}) = X _ {l} U _ {\pm} U _ {\circ}, \tag {25}
$$

$$
G _ {l} (X _ {0} \stackrel {\diamond} {\leftarrow} U _ {\pm} U _ {\circ}) = G _ {l} U _ {\pm} U _ {\circ}. \tag {26}
$$

Equation (25) holds trivially for $l = 0$ . When eq. (25) holds for some $l = k - 1$ , we can verify that

$$
\begin{array}{l} X _ {k} (X _ {0} \stackrel {\diamond} {\leftarrow} U _ {\pm} U _ {\circ}) \\ = X _ {k - 1} U _ {\pm} U _ {\circ} + A _ {k} X _ {k - 1} U _ {\pm} U _ {\circ} M \left(U _ {\circ} ^ {\top} U _ {\pm} ^ {\top} X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} U _ {\pm} U _ {\circ} + D _ {k}\right) \\ = X _ {k - 1} U _ {\pm} U _ {\circ} + A _ {k} X _ {k - 1} U _ {\pm} U _ {\circ} M U _ {\circ} ^ {\top} U _ {\pm} ^ {\top} \left(X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} + U _ {\pm} U _ {\circ} D _ {k} U _ {\circ} ^ {\top} U _ {\pm} ^ {\top}\right) U _ {\pm} U _ {\circ} \\ = X _ {k - 1} U _ {\pm} U _ {\circ} + A _ {k} X _ {k - 1} M \left(X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} + D _ {k}\right) U _ {\pm} U _ {\circ} \\ = \left(X _ {k - 1} + A _ {k} X _ {k - 1} M \left(X _ {k - 1} ^ {\top} C _ {k} X _ {k - 1} + D _ {k}\right)\right) U _ {\pm} U _ {\circ} = X _ {k} U _ {\pm} U _ {\circ}. \\ \end{array}
$$

It uses the fact that there exists some $D _ { i } ^ { 1 } , D _ { i } ^ { 2 } \in \mathbb R ^ { 2 \times 2 }$ such that $D _ { i } = \mathrm { d i a g } ( I _ { n } \otimes D _ { i } ^ { 1 } , D _ { i } ^ { 2 } )$ , so shuffling the first $\textit { n 2 } \times \ : 2$ diagonal sub-blocks of $D _ { i }$ does not change the matrix, and we have $U _ { \circ } D _ { i } U _ { \circ } ^ { \dagger } = D _ { i }$ . Similarly, we have $U _ { \pm } D _ { k } U _ { \pm } ^ { \top } = D _ { k }$ . This concludes eq. (25), and eq. (26) could be acquired similarly.

Next, we will establish the following equalities for $X _ { l }$ and $G _ { l } \colon$

$$
\frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (X _ {0} \stackrel {\diamond} {\leftarrow} U _ {\pm} U _ {\circ}, D _ {i} \stackrel {\leftarrow} {\leftarrow} t R) \bigg | _ {t = 0} = \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (D _ {i} \stackrel {\leftarrow} {\leftarrow} t U _ {\pm} U _ {\circ} R U _ {\circ} ^ {\top} U _ {\pm} ^ {\top}) \bigg | _ {t = 0} U _ {\pm} U _ {\circ}, \tag {27}
$$

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (X _ {0} \stackrel {\diamond} {\leftarrow} U _ {\pm} U _ {\circ}, D _ {i} \stackrel {\pm} {\leftarrow} t R) \right| _ {t = 0} = \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (D _ {i} \stackrel {\pm} {\leftarrow} t U _ {\pm} U _ {\circ} R U _ {\circ} ^ {\top} U _ {\pm} ^ {\top}) \right| _ {t = 0} U _ {\pm} U _ {\circ}. \tag {28}
$$

The proof follows by similar reductions as proving eqs. (18) and (20).

Finally, we consider the in-context risk under the permutation of $U _ { p }$ and $U _ { \xi }$ . Since each pair of $( x _ { i } , y _ { i } )$ is equivalently sampled from Gaussian distributions, we have $X _ { 0 } \overset { d } { = } X _ { 0 } U _ { \pm } U _ { \circ }$ . Therefore,

$$
\begin{array}{l} \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (D _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} \\ = 2 \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} (D _ {i} \stackrel {{+}} {{\leftarrow}} t R) \Big | _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}, W, U _ {p}, U _ {\xi}} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} \left(X _ {0} \stackrel {{\diamond}} {{\leftarrow}} U _ {\pm} U _ {\circ}\right) \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} \left(X _ {0} \stackrel {{\diamond}} {{\leftarrow}} U _ {\pm} U _ {\circ}, D _ {i} \stackrel {{+}} {{\leftarrow}} t R\right) \Bigg | _ {t = 0} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}, U _ {p}, U _ {\xi}} \left[ \operatorname{tr} \left((I - M) U _ {\circ} ^ {\top} U _ {\pm} ^ {\top} G _ {L} ^ {\top} \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (D _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\pm} U _ {\circ} R U _ {\circ} ^ {\top} U _ {\pm} ^ {\top}) \right| _ {t = 0} U _ {\pm} U _ {\circ} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} \mathbb {E} _ {U _ {p}, U _ {\xi}} \left[ G _ {L} (D _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\pm} U _ {\circ} ^ {\top} R U _ {\circ} U _ {\pm}) \right] \Bigg | _ {t = 0} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (D _ {i} \stackrel {+} {\leftarrow} t \widetilde {R}) \right| _ {t = 0} (I - M)\right) \right] = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (D _ {i} \stackrel {+} {\leftarrow} t \widetilde {R}) \right| _ {t = 0}, \\ \end{array}
$$

where $\begin{array} { r } { \widetilde { R } = \mathbb { E } _ { U _ { p } , U _ { \xi } } [ U _ { \pm } U _ { \circ } ^ { \top } R U _ { \circ } U _ { \pm } ] = \mathrm { d i a g } ( I _ { n } \otimes R ^ { 1 } , R ^ { 2 } ) , R ^ { 1 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } R _ { j } , R ^ { 2 } = R _ { n + 1 } } \end{array}$ , and $R _ { j }$ is the j-th 2×2 diagonal block of R. The 4th equality uses the fact that $\mathsf { \bar { t r } } [ ( I - M ) A ( I - M ) ]$ is extracting the right-bottom element of A, so it should be equal to tr $\left[ ( I - \dot { M } ) \dot { U _ { \circ } } ^ { \top } U _ { \pm } ^ { \top } A \dot { U _ { \pm } } U _ { \circ } ( \dot { I } - M ) \right]$ for any matrix A. This concludes that eq. (16) holds for $D _ { i }$ .

Till now, we have proved that eq. (16) holds for each one of $A _ { i } , B _ { i } , C _ { i } , D _ { i }$ . The proof of the whole theorem is then completed by applying Lemma 8. □

# D.3 PROOF OF THEOREM 2

Proof. In this proof, we follow the same notations as the proof of Theorem 1, where the constant $\textstyle { \frac { 1 } { n } }$ factor is dropped and $\widetilde { Z } _ { 0 } , \widetilde { X } _ { 0 } , \widetilde { Y } _ { 0 }$ are simplified as $Z _ { 0 } , X _ { 0 } , Y _ { 0 }$ respectively.

$$
Z _ {0} = \left[ \begin{array}{c c c c c c c c c c} x _ {1} & 0 & 0 & \dots & x _ {n} & 0 & 0 & x _ {\text {test}} & 0 & 0 \\ 0 & 0 & y _ {1} & \dots & 0 & 0 & y _ {n} & 0 & 0 & y _ {\text {test}} \end{array} \right] \in \mathbb {R} ^ {(2 d) \times (3 n + 3)}. \tag {29}
$$

Let $Z _ { l } \in \mathbb { R } ^ { 2 d \times ( 3 n + 3 ) }$ be the l-th layer’s output and let $X _ { l } , Y _ { l } \in \mathbb { R } ^ { d \times ( 3 n + 3 ) }$ be its first and last d rows. Our goal is to prove that, for any $E \in A \cup B \cup C \cup D$ and an arbitrary matrix $R \in \mathbb { R } ^ { d \times d }$ $( \mathbb { R } ^ { d _ { p } \times d _ { p } } f o r D )$ , there exists $\widetilde { R } \in S _ { I } \left( S _ { \Sigma } \right.$ for $\complement , S _ { P }$ for D) such that

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (E \stackrel {+} {\leftarrow} t \widetilde {R}) \right| _ {t = 0} \leq \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (E \stackrel {+} {\leftarrow} t R) \Bigg | _ {t = 0}. \tag {30}
$$

The proofs of eq. (30) for $A _ { i } , B _ { i }$ and $C _ { i }$ are identical with the proof of Theorem 1 so we omit them. We will be focusing on $D _ { i }$ for the rest of the proof.

Let $U _ { p } ^ { s } ~ \in ~ \mathbb { R } ^ { n \times n }$ and $U _ { p } ^ { t } ~ \in ~ \mathbb { R } ^ { ( n + 1 ) \times ( n + 1 ) }$ be uniformly sampled permutation matrices. Let $U _ { \circ } ^ { s } \ = \ \mathrm { d i a g } ( U _ { o } ^ { s } , 1 ) \otimes \mathrm { d i a g } ( 1 , 0 , 1 )$ and $U _ { \circ } ^ { t } ~ = ~ U _ { \upsilon } ^ { t } \otimes \mathrm { d i a g } ( 0 , 1 , 0 )$ . Therefore, $X _ { 0 } U _ { \circ } ^ { s }$ is shuffling the 1-st and 3-rd columns among each 3-column sub-block of $X _ { 0 }$ (except for the last $3 \cdot$ column sub-block), and $X _ { 0 } U _ { \circ } ^ { s }$ is shuffling the 2-nd column among each 3-column sub-block. Next, let $U _ { \xi } ^ { s } , U _ { \xi } ^ { t } ~ \in ~ \mathbb { R } ^ { ( n + 1 ) \times ( n + 1 ) }$ be diagonal matrices with uniformly sampled ±1 entries. Define $U _ { \pm } ^ { s } = U _ { \xi } ^ { s } \otimes \mathrm { d i a g } ( 1 , 0 , 1 )$ and $U _ { \pm } ^ { t } = U _ { \xi } ^ { t } \otimes \mathrm { d i a g } ( 0 , 1 , 0 )$ ). It can then be verified that $X _ { 0 } U _ { \pm } ^ { s } U _ { \pm } ^ { t } \overset { d } { = } X _ { 0 }$ .

To simplify the notations, let $U _ { \equiv }$ denote $U _ { \pm } ^ { s } U _ { \pm } ^ { t } U _ { \circ } ^ { s } U _ { \circ } ^ { t }$ . We will focus on a subset of $\begin{array} { r } { { \cal { S } } _ { P } \colon } \end{array}$ :

$$
\mathcal {S} _ {P} ^ {\prime} = \Big \{\mathrm{diag} (I _ {n} \otimes \Lambda_ {1}, \Lambda_ {2}) + I _ {n + 1} \otimes \Lambda_ {3}   \Big |   \Lambda_ {1}, \Lambda_ {2} \in \mathcal {M} \bigg ( \begin{array}{c c c} 1 & 0 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 1 \end{array} \bigg), \Lambda_ {3} \in \mathcal {M} \bigg ( \begin{array}{c c c} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{array} \bigg) \Big \}.
$$

Assume $D _ { k } = \mathrm { d i a g } ( I _ { n } \otimes \Lambda _ { 1 } , \Lambda _ { 2 } ) + I _ { n + 1 } \otimes \Lambda _ { 3 } \in { \mathcal S } _ { P } ^ { \prime }$ as defined above, one can verify that it is a block-diagonal matrix constructed from the same $3 \times 3$ sub-blocks, and thus is invariant under $U _ { \equiv } D _ { k } U _ { = } ^ { \top }$ . We will then prove that for any $l \in [ 1 , L ]$ ,

$$
X _ {l} (X _ {0} \stackrel {\diamond} {\leftarrow} U _ {\equiv}) = X _ {l} U _ {\equiv}, \tag {31}
$$

$$
G _ {l} (X _ {0} \stackrel {\diamond} {\leftarrow} U _ {\equiv}) = G _ {l} U _ {\equiv}, \tag {32}
$$

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (X _ {0} \stackrel {\diamond} {\leftarrow} U _ {\equiv}, D _ {i} \stackrel {\pm} {\leftarrow} t R) \right| _ {t = 0} = \left. \frac {\mathrm{d}}{\mathrm{d} t} X _ {l} (D _ {i} \stackrel {\pm} {\leftarrow} t U _ {\equiv} R U _ {\equiv} ^ {\top}) \right| _ {t = 0} U _ {\equiv}, \tag {33}
$$

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (X _ {0} \stackrel {\diamond} {\leftarrow} U _ {\equiv}, D _ {i} \stackrel {\pm} {\leftarrow} t R) \right| _ {t = 0} = \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (D _ {i} \stackrel {\pm} {\leftarrow} t U _ {\equiv} R U _ {\equiv} ^ {\top}) \right| _ {t = 0} U _ {\equiv}. \tag {34}
$$

These results can be acquired by similar proofs as eqs. (25) to (28). We then consider the in-context risk under the permutations of $U _ { \equiv }$ . Similarly, we have $X _ { 0 } \overset { d } { = } X _ { 0 } U _ { \equiv }$ and

$$
\begin{array}{l} \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (D _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} \\ = 2 \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} \left. \frac {\mathrm{d}}{\mathrm{d} t} Y _ {L} (D _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}, U _ {\equiv}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} (X _ {0} \stackrel {{\diamond}} {{\leftarrow}} U _ {\equiv}) \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (X _ {0} \stackrel {{\diamond}} {{\leftarrow}} U _ {\equiv}, D _ {i} \stackrel {{+}} {{\leftarrow}} t R) \Big | _ {t = 0} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}, U _ {\equiv}} \left[ \operatorname{tr} \left((I - M) U _ {\equiv} ^ {\top} G _ {L} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (D _ {i} \stackrel {+} {\leftarrow} t U _ {\equiv} R U _ {\equiv} ^ {\top}) \Big | _ {t = 0} U _ {\equiv} (I - M)\right) \right] \\ = 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (D _ {i} \stackrel {{+}} {{\leftarrow}} t \mathbb {E} _ {U _ {\equiv}} [ U _ {\equiv} R U _ {\equiv} ^ {\top} ]) \Bigg | _ {t = 0} (I - M)\right) \right] \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (D _ {i} \stackrel {+} {\leftarrow} t \widetilde {R}) \right| _ {t = 0}. \\ \end{array}
$$

Let $R _ { j }$ be the j-th $3 \times 3$ diagonal block of $R ,$ then $\begin{array} { r } { R ^ { 1 } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } R _ { j } \circ \binom { 1 } { 1 0 0 } , R ^ { 2 } = R _ { n + 1 } \circ } \end{array}$ ${ \begin{array} { r } { { \left( \begin{array} { l l } { 1 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { 0 } \\ { 1 } & { 0 } & { 1 } \end{array} \right) } , R ^ { 3 } = { \frac { 1 } { n + 1 } } \sum _ { j = 1 } ^ { n + 1 } R _ { j } \circ { \left( \begin{array} { l l } { 0 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 0 } \end{array} \right) } } \end{array} }$ R j ◦ and $\widetilde { R } = \mathbb { E } _ { U _ { \equiv } } \bigl [ U _ { \equiv } R U _ { \equiv } ^ { \top } \bigr ] = \mathrm { d i a g } ( I _ { n } \otimes R ^ { 1 } , R ^ { 2 } ) + I _ { n + 1 } \otimes R ^ { 3 }$ . This indicates that eq. (30) holds for each ${ \cal D } _ { i } \in { \cal S } _ { P } ^ { \prime }$ , and thus the proof of the whole theorem completes by applying Lemma 8 and noticing that ${ \cal S } _ { P } ^ { \prime } \subset { \cal S } _ { P }$ .

# D.4 PROOF OF THEOREM 7

Proof. We keep the same notations as the proof of Theorem 1, dropping the $\textstyle { \frac { 1 } { n } }$ factor and simplifying $\widetilde { X } _ { 0 } , \widetilde { Y } _ { 0 } , \widetilde { Z } _ { 0 }$ as $X _ { 0 } , Y _ { 0 } , Z _ { 0 }$ , as follows:

$$
Z _ {0} = \left[ \begin{array}{c c c c c c c} 0 & 0 & \dots & 0 & 0 & 0 & 0 \\ x _ {1} & y _ {1} & \dots & x _ {n} & y _ {n} & x _ {\text {test}} & y _ {\text {test}} \end{array} \right] \in \mathbb {R} ^ {(2 d) \times (2 n + 2)}. \tag {35}
$$

Note that we now have $X _ { 0 }$ and $Y _ { 0 }$ containing both $x _ { i }$ and $y _ { i }$ . Define

$$
X = [ x _ {1} \quad 0 \quad \dots \quad x _ {n} \quad 0 \quad x _ {\text {test}} \quad 0 ],
$$

$$
\overline {{X}} = \left[ \begin{array}{c c c c c c c} 0 & x _ {1} & \dots & 0 & x _ {n} & 0 & x _ {\text {test}} \end{array} \right],
$$

$$
Y = \left[ \begin{array}{c c c c c c c} 0 & y _ {1} & \dots & 0 & y _ {n} & 0 & y _ {\text {test}} \end{array} \right].
$$

we then have $Y _ { 0 } = X + Y = X + W \overline { { { X } } }$ . From the parameter configuration in eq. (12), the update rule of the first attention layer is

$$
X _ {1} = A _ {1} Y _ {0} M D _ {1} = A _ {1} X M D _ {1}, \quad Y _ {1} = Y _ {0} = X + W \overline {{{X}}}. \tag {36}
$$

The update rule for the following layers is the same as eq. (15). We are going to prove that, for any $E \in A \cup B \cup C \cup D$ and an arbitrary matrix $R \in \mathbb { R } ^ { d \times d } ( \mathbb { R } ^ { d _ { p } \times d _ { p } }$ for $D )$ , there exists $\widetilde { R } \in { S } _ { I } \left( { S } _ { \Sigma } \right.$ for $C , S _ { P }$ for D) such that

$$
\left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (E \stackrel {+} {\leftarrow} t \widetilde {R}) \right| _ {t = 0} \leq \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (E \stackrel {+} {\leftarrow} t R) \right| _ {t = 0}. \tag {37}
$$

Similarly to Theorem 1, we uniformly sample $U _ { \perp } \in \mathbb { R } ^ { d \times d }$ as an orthonormal random matrix, and let $U _ { \Sigma } = \bar { \Sigma } ^ { 1 / 2 } U _ { \bot } \Sigma ^ { - 1 / 2 }$ . Under the condition that $B _ { l } = b _ { l } I _ { d }$ for some $b _ { l } \in \mathbb { R }$ , we have

$$
Y _ {l} = Y _ {1} \prod_ {j = 2} ^ {l} \big (I + b _ {j} M \big (X _ {j - 1} ^ {\top} C _ {j} X _ {j - 1} + D _ {j} \big) \big).
$$

$\begin{array} { r l r l r l r l r l r l r l } { \mathrm { L e t } } & { } & & { { } \mathit { F } _ { l } } & { } & { } & { { } = } & { } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } & { { } } \end{array}$ $\begin{array} { r } { \overline { { X } } \prod _ { j = 2 } ^ { l } \bigl ( I + b _ { j } M \bigl ( X _ { j - 1 } ^ { \top } C _ { j } X _ { j - 1 } + D _ { j } \bigr ) \bigr ) } \end{array}$ , we then have $Y _ { l } = F _ { l } + W G _ { l } .$ . According to Lemma 9,

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (E \stackrel {+} {\leftarrow} t R) \bigg | _ {t = 0} \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) Y _ {L} ^ {\top} (E \stackrel {+} {\leftarrow} t R) Y _ {L} (E \stackrel {+} {\leftarrow} t R) (I - M)\right) \right] \right| _ {t = 0} \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) F _ {L} ^ {\top} (E \stackrel {+} {\leftarrow} t R) F _ {L} (E \stackrel {+} {\leftarrow} t R) (I - M)\right) \right] \right| _ {t = 0} \\ \left. + \frac {\mathrm{d}}{\mathrm{d} t} \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} (E \stackrel {{+}} {{\leftarrow}} t R) W ^ {\top} W G _ {L} (E \stackrel {{+}} {{\leftarrow}} t R) (I - M)\right) \right] \right| _ {t = 0} \\ = 2 \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) F _ {L} ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} F _ {L} (E \stackrel {{+}} {{\leftarrow}} t R) \Bigg | _ {t = 0} (I - M)\right) \right] \\ + 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (E \stackrel {{+}} {{\leftarrow}} t R) \right| _ {t = 0} (I - M)\right) \right]. \\ \end{array}
$$

Next, we will show that eq. (37) holds for each one of $A _ { i } , B _ { i } , C _ { i } , D _ { i }$ for any $i \in [ 1 , L ]$ .

# 1. Equation (37) holds for Ai.

One can easily verify that eqs. (17) and (18) still hold. Furthermore, eqs. (19) and (20) hold for both $F _ { l }$ and $G _ { l }$ . With these observations, we can then verify

$$
\begin{array}{l} \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (A _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} \\ = 2 \mathbb {E} _ {X _ {0}, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) F _ {L} ^ {\top} (X \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) \frac {\mathrm{d}}{\mathrm{d} t} F _ {L} (X \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \Bigg | _ {t = 0} (I - M)\right) \right] \\ + 2 d \mathbb {E} _ {X _ {0}, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} (X \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (X \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, A _ {i} \stackrel {{+}} {{\leftarrow}} t R) \Big | _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) F _ {L} ^ {\top} U _ {\Sigma} ^ {\top} U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} F _ {L} (A _ {i} \stackrel {{+}} {{\leftarrow}} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} (I - M)\right) \right] \\ + 2 d \mathbb {E} _ {X _ {0}, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} U _ {\Sigma} ^ {\top} \Sigma^ {- 1} U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (A _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) F _ {L} ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} F _ {L} (A _ {i} \stackrel {+} {\leftarrow} t r I _ {d}) \Big | _ {t = 0} (I - M)\right) \right] \\ + 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {L} ^ {\top} \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {L} (A _ {i} \stackrel {{+}} {{\leftarrow}} t r I _ {d}) \right| _ {t = 0} (I - M)\right) \right] \\ = \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (A _ {i} \stackrel {+} {\leftarrow} t r I _ {d}) \right| _ {t = 0}, \\ \end{array}
$$

$\begin{array} { r } { \mathrm { w h e r e } r = \mathbb { E } _ { U _ { \perp } } [ U _ { \Sigma } ^ { - 1 } R U _ { \Sigma } ] = \frac { 1 } { d } \mathrm { t r } \big ( \Sigma ^ { - 1 / 2 } R \Sigma ^ { 1 / 2 } \big ) . } \end{array}$

# 2. Equation (37) holds for $B _ { i }$ .

From the definition of $F _ { l }$ and $G _ { l } .$ , we can verify that

$$
\frac {\mathrm{d}}{\mathrm{d} t} Y _ {l} (B _ {i} \stackrel {+} {\leftarrow} t R) \bigg | _ {t = 0}
$$

$$
= R (F _ {i - 1} + W G _ {i - 1}) M (X _ {i - 1} ^ {\top} C _ {i} X _ {i - 1} + D _ {i}) \prod_ {j = i + 1} ^ {l} \big (I + b _ {j} M (X _ {j - 1} ^ {\top} C _ {j} X _ {j - 1} + D _ {j}) \big).
$$

Define

$$
\overline {{F}} _ {l} ^ {i} = \left(F _ {i - 1} + B _ {i} F _ {i - 1} M (X _ {i - 1} ^ {\top} C _ {i} X _ {i - 1} + D _ {i})\right) \prod_ {j = i + 1} ^ {l} \left(I + b _ {j} M (X _ {j - 1} ^ {\top} C _ {j} X _ {j - 1} + D _ {j})\right),
$$

$$
\overline {{G}} _ {l} ^ {i} = \big (W G _ {i - 1} + B _ {i} W G _ {i - 1} M (X _ {i - 1} ^ {\top} C _ {i} X _ {i - 1} + D _ {i}) \big) \prod_ {j = i + 1} ^ {l} \big (I + b _ {j} M (X _ {j - 1} ^ {\top} C _ {j} X _ {j - 1} + D _ {j}) \big),
$$

We then have

$$
\frac {\mathrm{d}}{\mathrm{d} t} Y _ {l} (B _ {i} \stackrel {+} {\leftarrow} t R) \bigg | _ {t = 0} = \left. \frac {\mathrm{d}}{\mathrm{d} t} \overline {{F}} _ {l} ^ {i} (B _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} + \left. \frac {\mathrm{d}}{\mathrm{d} t} \overline {{G}} _ {l} ^ {i} (B _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0}.
$$

Similar to eqs. (20) and (22), we can prove that

$$
\frac {\mathrm{d}}{\mathrm{d} t} \overline {{F}} _ {l} ^ {i} (X _ {0} \stackrel {\times} {\leftarrow} U _ {\Sigma}, B _ {i} \stackrel {+} {\leftarrow} t R) \bigg | _ {t = 0} = U _ {\Sigma} \left. \frac {\mathrm{d}}{\mathrm{d} t} \overline {{F}} _ {l} ^ {i} (B _ {i} \stackrel {+} {\leftarrow} t U _ {\Sigma} ^ {- 1} R U _ {\Sigma}) \right| _ {t = 0},
$$

$$
\mathbb {E} _ {W} \left[ W ^ {\top} \left. \frac {\mathrm{d}}{\mathrm{d} t} \overline {{G}} _ {l} ^ {i} (B _ {i} \stackrel {+} {\leftarrow} t R) \right| _ {t = 0} \right] = \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} \overline {{G}} _ {l} ^ {i} (B _ {i} \stackrel {+} {\leftarrow} t \operatorname{tr} (R) I _ {d}) \right| _ {t = 0}.
$$

Without loss of generality, we assume that $r \ = \ { \textstyle \frac { 1 } { d } } \operatorname { t r } \bigl ( \Sigma ^ { - 1 / 2 } R \Sigma ^ { 1 / 2 } \bigr ) \ \leq \ { \textstyle \frac { 1 } { d } } \operatorname { t r } ( R )$ , and let $\gamma =$ $r d / \operatorname { t r } ( R ) \leq 1$ . Then, one can verify that

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (B _ {i} \stackrel {+} {\leftarrow} t R) \bigg | _ {t = 0} \\ = 2 \mathbb {E} _ {X _ {0}, U _ {\perp}} \left[ \operatorname{tr} \left((I - M) F _ {l} ^ {\top} (X \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}) \frac {\mathrm{d}}{\mathrm{d} t} \overline {{F}} _ {l} ^ {i} (X \stackrel {{\times}} {{\leftarrow}} U _ {\Sigma}, B _ {i} \stackrel {{+}} {{\leftarrow}} t R) \Big | _ {t = 0} (I - M)\right) \right] \\ + 2 \mathbb {E} _ {X _ {0}, W} \left[ \operatorname{tr} \left((I - M) G _ {l} ^ {\top} W ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} \overline {{G}} _ {l} ^ {i} (B _ {i} \stackrel {+} {\leftarrow} t R) \Big | _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) F _ {l} ^ {\top} \left. \frac {\mathrm{d}}{\mathrm{d} t} \overline {{F}} _ {l} ^ {i} \left(B _ {i} \stackrel {{+}} {{\leftarrow}} t r I _ {d}\right) \right| _ {t = 0} (I - M)\right) \right] \\ + 2 \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {l} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} \bar {G} _ {l} ^ {i} (B _ {i} \stackrel {+} {\leftarrow} t \operatorname{tr} (R) I _ {d}) \Big | _ {t = 0} (I - M)\right) \right] \\ = 2 \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) F _ {l} ^ {\top} \frac {\mathrm{d}}{\mathrm{d} t} F _ {l} \left(B _ {i} \stackrel {{+}} {{\leftarrow}} t r I _ {d}\right) \Big | _ {t = 0} (I - M)\right) \right] \\ \left. + \frac {1}{\gamma} 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {l} ^ {\top} \Sigma^ {- 1} \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} \left(B _ {i} \stackrel {{+}} {{\leftarrow}} t r I _ {d}\right) \Big | _ {t = 0} (I - M)\right) \right] \right. \\ = \left(\frac {1}{\gamma} - 1\right) 2 d \mathbb {E} _ {X _ {0}} \left[ \operatorname{tr} \left((I - M) G _ {l} ^ {\top} \Sigma^ {- 1} \left. \frac {\mathrm{d}}{\mathrm{d} t} G _ {l} (B _ {i} \stackrel {+} {\leftarrow} t r I _ {d}) \right| _ {t = 0} (I - M)\right) \right] \\ + \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (B _ {i} \stackrel {+} {\leftarrow} t r I _ {d}) \right| _ {t = 0} \geq \left. \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (B _ {i} \stackrel {+} {\leftarrow} t r I _ {d}) \right| _ {t = 0}. \\ \end{array}
$$

The last inequality assumes the positivity of the term involving $G _ { l }$ . Otherwise, one can simply flip the numerator and denominator of $\gamma$ and scale the derivative of $F _ { l }$ instead of $G _ { l }$ to yield an additional positive term besides the risk term to finish the proof.

# 3. Equation (37) holds for $C _ { i } , D _ { i } .$ .

Similarly, one can verify that eqs. (23) and (24) still hold (also eqs. (25) to (28)), and finish the proof by following the same reductions as Theorem 1 with $F _ { l }$ and $G _ { l } .$ □

# D.5 PROOF OF PROPOSITION 3

Proof. Let $A _ { l } = a _ { l } I _ { d } , B _ { l } = b _ { l } I _ { d } , C _ { l } = c _ { l } I _ { d } \mathrm { a n d } D _ { l } = \mathrm { d i a g } \big ( I _ { n } \otimes D _ { l } ^ { 1 } , D _ { l } ^ { 2 } \big ) + I _ { n + 1 } \otimes D _ { l } ^ { 3 } + D _ { l } ^ { 4 } \otimes D _ { l } ^ { 5 }$ for $l \in [ 1 , 2 ]$ . Let $Z _ { l } \in \mathbb { R } ^ { 2 d \times ( 3 n + 3 ) }$ be the output of the l-th attention layer, and let $X _ { l } , Y _ { l } \in \mathbb { R } ^ { d \times ( 3 n + 3 ) }$ be its first and last d rows respectively. Note that $Y _ { l }$ in this proof does not contain $y _ { \mathrm { t e s t } }$ .

Let $\begin{array} { r } { D _ { 1 } ^ { 1 } = { \binom { d _ { x } ^ { x } \ 0 \ d _ { x } ^ { y } } { \ d _ { y } ^ { x } \ 0 \ d _ { y } ^ { y } } } , D _ { 1 } ^ { 2 } = { \left( \begin{array} { l l l } { s _ { x } \ 0 \ s _ { y } } \\ { 0 \ 0 \ 0 } \\ { 0 \ 0 \ 0 } \end{array} \right) } } \end{array}$ (note that the last row of $D _ { 1 } ^ { 2 }$ is masked out by M , so we simply set it to 0), and $D _ { 1 } ^ { 5 } = { \left( \begin{array} { l l l } { 0 } & { t _ { x } } & { 0 } \\ { 0 } & { 0 } & { 0 } \\ { 0 } & { t _ { y } } & { 0 } \end{array} \right) }$ 0 ty We use $D$ as an abbreviation for $D _ { 1 } ^ { 4 }$ , and use $d _ { i , j }$ to denote the elements in D. One can verify that

$$
\begin{array}{l} X _ {1} = X _ {0} + a _ {1} X _ {0} M \left(\operatorname{diag} \left(I _ {n} \otimes D _ {1} ^ {1}, D _ {1} ^ {2}\right) + I _ {n + 1} \otimes D _ {1} ^ {3} + D _ {1} ^ {4} \otimes D _ {1} ^ {5}\right) \\ [ \quad (1 + a _ {1} d _ {x} ^ {x}) x _ {1} \qquad a _ {1} t _ {x} \sum_ {i = 1} ^ {n + 1} d _ {i, 1} x _ {i} \qquad a _ {1} d _ {x} ^ {y} x _ {1} \\ \begin{array}{r l r} & = & (1 + a _ {1} d _ {x} ^ {x}) x _ {n} \quad a _ {1} t _ {x} \sum_ {i = 1} ^ {n + 1} d _ {i, n} x _ {i} \quad a _ {1} d _ {x} ^ {y} x _ {n} \\ & & (1 + a _ {1} d _ {x} ^ {x}) x _ {\text {test}} \quad a _ {1} t _ {x} \sum_ {i = 1} ^ {n + 1} d _ {i, n + 1} x _ {i} \quad a _ {1} d _ {x} ^ {y} x _ {\text {test}} \end{array} . \\ \end{array}
$$

Similarly, we have

$$
\begin{array}{l} Y _ {1} = Y _ {0} + b _ {1} Y _ {0} M \left(\operatorname{diag} \left(I _ {n} \otimes D _ {1} ^ {1}, D _ {1} ^ {2}\right) + I _ {n + 1} \otimes D _ {1} ^ {3} + D _ {1} ^ {4} \otimes D _ {1} ^ {5}\right) \\ [ b _ {1} d _ {y} ^ {x} y _ {1} \quad b _ {1} t _ {y} \sum_ {i = 1} ^ {n} d _ {i, 1} y _ {i} \quad (1 + b _ {1} d _ {y} ^ {y}) y _ {1} \\ = \begin{array}{c c c c} & \dots \\ b _ {1} d _ {y} ^ {x} y _ {n} & b _ {1} t _ {y} \sum_ {i = 1} ^ {n} d _ {i, n} y _ {i} & (1 + b _ {1} d _ {y} ^ {y}) y _ {n} \\ 0 & b _ {1} t _ {y} \sum_ {i = 1} ^ {n} d _ {i, n + 1} y _ {i} & 0 \end{array} . \\ \end{array}
$$

By the definition of linear attention, we can show that

$$
\begin{array}{l} \mathsf {T F} (Z _ {0}; \{V _ {l}, Q _ {l} \} _ {l = 1} ^ {2}) = (Y _ {2}) _ {3 n + 3} = b _ {2} Y _ {1} M \left(c _ {2} X _ {1} ^ {\top} (X _ {1}) _ {3 n + 3} + (D _ {2}) _ {3 n + 3}\right) \\ = b _ {2} c _ {2} a _ {1} d _ {x} ^ {y} \left(\sum_ {i = 1} ^ {3 n + 2} (Y _ {1}) _ {i} (X _ {1}) _ {i} ^ {\top}\right) x _ {\text { test }}. \\ \end{array}
$$

Define $\Delta X _ { 1 } = [ 0 \quad a _ { 1 } t _ { x } d _ { n + 1 , 1 } x _ { \mathrm { t e s t } } \quad 0 \quad \cdots \quad 0 \quad a _ { 1 } t _ { x } d _ { n + 1 , n + 1 } x _ { \mathrm { t e s t } } \quad 0 ]$ , and let ${ \overline { { X } } } _ { 1 } = X _ { 1 } -$ $\Delta X _ { 1 }$ , then $\mathsf { T F } ( Z _ { 0 } ; \{ V _ { l } , Q _ { l } \} _ { l = 1 } ^ { 2 } ) = \mathsf { T F } ( Z _ { 0 } ; \{ V _ { l } , Q _ { l } \} _ { l = 1 } ^ { 2 } , X _ { 1 } \gets \overline { { X } } _ { 1 } ) + \mathsf { T F } ( Z _ { 0 } ; \{ V _ { l } , Q _ { l } \} _ { l = 1 } ^ { 2 } , X _ { 1 } \gets \overline { { X } } _ { 1 } ^ { 2 } )$ $\Delta X _ { 1 } )$ . Let $b _ { 1 } d _ { y } ^ { x } ( 1 + a _ { 1 } d _ { x } ^ { x } ) + \bar { ( 1 + b _ { 1 } d _ { y } ^ { x } ) } a _ { 1 } d _ { x } ^ { x } = a , \bar { b } _ { 1 } \bar { t } _ { y } a _ { 1 } t _ { x } = b , b _ { 2 } c _ { 2 } a _ { 1 } d _ { x } ^ { y } = c _ { 3 } \bar { t } _ { y } a _ { 2 } d _ { 1 } ^ { x } ,$ we then have

$$
\begin{array}{l} \mathsf {T F} (Z _ {0}; \{V _ {l}, Q _ {l} \} _ {l = 1} ^ {2}, X _ {1} \leftarrow \overline {{X}} _ {1}) = c \left(a \sum_ {i = 1} ^ {n} y _ {i} x _ {i} ^ {\top} + b \sum_ {i = 1} ^ {n + 1} \left(\sum_ {j = 1} ^ {n} d _ {j, i} y _ {j}\right) \left(\sum_ {j = 1} ^ {n} d _ {j, i} x _ {j} ^ {\top}\right)\right) x _ {\text {test}} \\ = c \left(a \sum_ {i = 1} ^ {n} y _ {i} x _ {i} ^ {\top} + b \sum_ {j = 1} ^ {n} \sum_ {k = 1} ^ {n} \left(\sum_ {i = 1} ^ {n + 1} d _ {j, i} d _ {k, i}\right) y _ {j} x _ {k} ^ {\top}\right) x _ {\text { test }}, (38) \\ \mathsf {T F} (Z _ {0}; \{V _ {l}, Q _ {l} \} _ {l = 1} ^ {2}, X _ {1} \leftarrow \Delta X _ {1}) = b c \sum_ {i = 1} ^ {n + 1} \sum_ {j = 1} ^ {n} d _ {j, i} y _ {j} d _ {n + 1, i} x _ {\mathrm{test}} ^ {\top} x _ {\mathrm{test}} \\ = b c \sum_ {j = 1} ^ {n} \left(\sum_ {i = 1} ^ {n + 1} d _ {j, i} d _ {n + 1, i}\right) y _ {j} x _ {\text { test }} ^ {\top} x _ {\text { test }}. (39) \\ \end{array}
$$

Now consider the in-context risk,

$$
\begin{array}{l} \mathcal {L} (V, Q) = \mathbb {E} _ {Z _ {0}, W} \| \mathsf {T F} (Z _ {0}; \{V, Q \}) + W x _ {\text { test }} \| _ {2} ^ {2} \\ = \mathbb {E} _ {Z _ {0}, W} \Big [ (\mathsf {T F} (Z _ {0}; \{V, Q \}) + W x _ {\mathrm{test}}) ^ {\top} (\mathsf {T F} (Z _ {0}; \{V, Q \}) + W x _ {\mathrm{test}}) \Big ] \\ = \mathbb {E} _ {Z _ {0}, W} \left[ \left(\mathsf {T F} (Z _ {0}; \{V, Q \}, X _ {1} \leftarrow \overline {{X}} _ {1}) + W x _ {\text {test}}\right) ^ {\top} \left(\mathsf {T F} (Z _ {0}; \{V, Q \}, X _ {1} \leftarrow \overline {{X}} _ {1}) + W x _ {\text {test}}\right) \right] \\ + 2 \mathbb {E} _ {Z _ {0}, W} \left[ \mathsf {T F} (Z _ {0}; \{V, Q \}, X _ {1} \leftarrow \Delta X _ {1}) ^ {\top} \left(\mathsf {T F} (Z _ {0}; \{V, Q \}, X _ {1} \leftarrow \overline {{X}} _ {1}) + W x _ {\text { test }}\right) \right] \\ + \mathbb {E} _ {Z _ {0}, W} \left[ \mathsf {T F} (Z _ {0}; \{V, Q \}, X _ {1} \leftarrow \Delta X _ {1}) ^ {\top} \mathsf {T F} (Z _ {0}; \{V, Q \}, X _ {1} \leftarrow \Delta X _ {1}) \right]. \\ \end{array}
$$

In the equation above, the 3-rd part is always positive. We then examine the second part:

$$
\begin{array}{l} \mathbb {E} _ {Z _ {0}, W} \left[ \mathsf {T F} (Z _ {0}; \{V, Q \}, X _ {1} \leftarrow \Delta X _ {1}) ^ {\top} \left(\mathsf {T F} (Z _ {0}; \{V, Q \}, X _ {1} \leftarrow \overline {{X}} _ {1}) + W x _ {\text { test }}\right) \right] \\ = \mathbb {E} _ {Z _ {0}, W} \big [ x _ {\mathrm{test}} ^ {\top} x _ {\mathrm{test}} v _ {1} x _ {\mathrm{test}} + x _ {\mathrm{test}} ^ {\top} x _ {\mathrm{test}} v _ {2} x _ {\mathrm{test}} \big ] = 0, \\ \end{array}
$$

where $\begin{array} { r } { v _ { 1 } = b c \sum _ { j = 1 } ^ { n } \biggl ( \sum _ { i = 1 } ^ { n + 1 } d _ { j , i } d _ { n + 1 , i } \biggr ) y _ { j } ^ { \top } c \biggl ( a \sum _ { i = 1 } ^ { n } y _ { i } x _ { i } ^ { \top } + b \sum _ { j = 1 } ^ { n } \sum _ { k = 1 } ^ { n } \biggl ( \sum _ { i = 1 } ^ { n + 1 } d _ { j , i } d _ { k , i } \biggr ) y _ { j } x _ { k } ^ { \top } \biggr ) } \end{array}$ and $\begin{array} { r } { v _ { 2 } = b c \sum _ { j = 1 } ^ { n } \left( \sum _ { i = 1 } ^ { n + 1 } d _ { j , i } d _ { n + 1 , i } \right) y _ { j } ^ { \top } W } \end{array}$ are independent of $x _ { \mathrm { t e s t } }$ . Therefore, $\mathcal { L } ( V , Q )$ attains its minimum only $\mathrm { i f } \mathsf { \bar { T } F } ( Z _ { 0 } ; \{ V , Q \} , \dot { X } _ { 1 } \gets \Delta X _ { 1 } ) = 0 ,$ , implying $d _ { n + 1 , i } = 0 { \mathrm { ~ f o r ~ } } i \in [ 1 , n + 1 ]$ .

In the following analysis, we will assume that the last row of D is 0, and let $M \in \mathbb { R } ^ { n \times ( n + 1 ) }$ be the first n rows of D. Additionally, we will drop the c factor in eq. (38), since its position could be substituted by a and b. We then define $\begin{array} { r } { \widetilde { W } = a \sum _ { i = 1 } ^ { n } y _ { i } x _ { i } ^ { \top } + b \sum _ { j = 1 } ^ { n } \sum _ { k = 1 } ^ { n } \Bigl ( \sum _ { i = 1 } ^ { n + 1 } d _ { j , i } d _ { k , i } \Bigr ) y _ { j } x _ { k } ^ { \top } } \end{array}$ , $X = [ x _ { 1 } \quad \cdots \quad x _ { n } ]$ and $Y = [ y _ { 1 } \quad \cdot \cdot \cdot \quad y _ { n } ]$ . One can verify that

$$
\widetilde {W} = a Y X ^ {\top} + b Y M M ^ {\top} X ^ {\top} = a W X X ^ {\top} + b W X M M ^ {\top} X ^ {\top}. \tag {40}
$$

Furthermore, the in-context risk could be expanded as

$$
\begin{array}{l} \mathcal {L} (V, Q) = \mathbb {E} _ {Z _ {0}, W} \left\| \widetilde {W} x _ {\mathrm{test}} + W x _ {\mathrm{test}} \right\| _ {2} ^ {2} = \mathbb {E} _ {Z _ {0}, W} \Big [ x _ {\mathrm{test}} ^ {\top} (\widetilde {W} + W) ^ {\top} (\widetilde {W} + W) x _ {\mathrm{test}} \Big ] \\ = \mathbb {E} _ {Z _ {0}, W} \left[ \operatorname{tr} \left(\left(\widetilde {W} + W\right) ^ {\top} (\widetilde {W} + W)\right) \right] \\ = \mathbb {E} _ {Z _ {0}, W} \Big [ \mathrm{tr} \Big (\widetilde {W} ^ {\top} \widetilde {W} \Big) + 2   \mathrm{tr} \Big (W ^ {\top} \widetilde {W} \Big) + \mathrm{tr} \big (W ^ {\top} W \big) \Big ]. \\ \end{array}
$$

We will use the identity $\mathbb { E } _ { X } [ X A X ^ { \top } X B X ^ { \top } ] = \left( \operatorname { t r } ( A ) \operatorname { t r } ( B ) + \operatorname { t r } \big ( A B ^ { \top } \big ) + d \operatorname { t r } ( A B ) \right) I _ { d }$ for any $A , B \in \mathbb { R } ^ { n \times n }$ , which can be acquired by expanding each element and applying Isserlis’ theorem. Let $T _ { 1 } = \mathrm { t r } \big ( M M ^ { \top } \big )$ and $T _ { 2 } = \dot { \mathrm { t r } } \big ( M M ^ { \dagger } M \dot { M } ^ { \top } \big )$ , then

$$
\begin{array}{l} \mathbb {E} _ {Z _ {0}, W} \left[ \operatorname{tr} \left(\left(a W X X ^ {\top} + b W X M M ^ {\top} X ^ {\top}\right) ^ {\top} \left(a W X X ^ {\top} + b W X M M ^ {\top} X ^ {\top}\right)\right) \right] \\ = \mathbb {E} _ {Z _ {0}, W} \left[ a ^ {2} \operatorname{tr} \left(X X ^ {\top} W ^ {\top} W X X ^ {\top}\right) + 2 a b \operatorname{tr} \left(X X ^ {\top} W ^ {\top} W X M M ^ {\top} X ^ {\top}\right) \right] \\ + \mathbb {E} _ {Z _ {0}, W} \left[ b ^ {2} \operatorname{tr} \left(X M M ^ {\top} X ^ {\top} W ^ {\top} W X M M ^ {\top} X ^ {\top}\right) \right] \\ = d \mathbb {E} _ {Z _ {0}} \left[ a ^ {2} \operatorname{tr} \left(X X ^ {\top} X X ^ {\top}\right) + 2 a b \operatorname{tr} \left(X X ^ {\top} X M M ^ {\top} X ^ {\top}\right) + b ^ {2} \operatorname{tr} \left(X M M ^ {\top} X ^ {\top} X M M ^ {\top} X ^ {\top}\right) \right] \\ = a ^ {2} d ^ {2} n (n + 1 + d) + 2 a b d ^ {2} (n + 1 + d) T _ {1} + b ^ {2} d ^ {2} (T _ {1} ^ {2} + (1 + d) T _ {2}). \\ \end{array}
$$

Simultaneously, we can verify that $\mathbb { E } _ { Z _ { 0 } , W } [ \mathrm { t r } \big ( W ^ { \top } W \big ) ] = d ^ { 2 }$ and

$$
\mathbb {E} _ {Z _ {0}, W} \Big [ \mathrm{tr} \Big (W ^ {\top} \widetilde {W} \Big) \Big ] = \mathbb {E} _ {Z _ {0}, W} \big [ a W ^ {\top} W X X ^ {\top} + b W ^ {\top} W X M M ^ {\top} X ^ {\top} \big ] = a d ^ {2} n + b d ^ {2} T _ {1}.
$$

Combining the results above, we aim to find the optimal $a , b ,$ M that minimize

$$
\frac {1}{d ^ {2}} \mathcal {L} (V, Q) = c _ {0} + c _ {1} T _ {1} + c _ {2} T _ {1} ^ {2} + c _ {3} T _ {2},
$$

where

$$
c _ {0} = a ^ {2} n (n + 1 + d) + 1 + 2 a n, \quad c _ {1} = 2 a b (n + 1 + d) + 2 b,
$$

$$
c _ {2} = b ^ {2}, \quad c _ {3} = b ^ {2} (1 + d).
$$

Since $c _ { 3 } \geq 0 ,$ , to minimize $\mathcal { L } ( V , Q )$ we need to minimize $T _ { 2 } .$ . Given that $M M ^ { \top }$ is symmetric, we denote its n eigenvalues as $\lambda _ { i } , i \in [ 1 , n ]$ . Then by Cauchy–Schwarz inequality,

$$
\operatorname{tr} \left(M M ^ {\top} M M ^ {\top}\right) = \sum_ {i = 1} ^ {n} \lambda_ {i} ^ {2} \geq \frac {1}{n} \left(\sum_ {i = 1} ^ {n} \lambda_ {i}\right) ^ {2} = \frac {1}{n} \operatorname{tr} ^ {2} (M M ^ {\top}).
$$

Therefore, $\mathcal { L } ( V , Q )$ is minimized only if the inequality above holds with equality, which implies that $\lambda _ { i } = \lambda _ { j }$ for any $i \neq j$ . This concludes the proof by showing that there exists $\lambda \in \mathbb { R }$ such that $M M ^ { \top } = \dot { \lambda \boldsymbol { I } _ { d } } .$ , and thus $D D ^ { \top } = \mathrm { d i a g } ( \lambda I _ { d } , 0 )$ . □

# D.6 PROOF OF PROPOSITION 5

Proof. We will continue from eqs. (38) and (39). After applying token-wise dropout, we have

$$
\begin{array}{l} \mathsf {T F} (Z _ {0}; \{V _ {l}, Q _ {l} \} _ {l = 1} ^ {2}, X _ {1} \leftarrow \overline {{X}} _ {1}) = \sum_ {i = 1} ^ {n} (a o _ {2} ^ {3 i - 2} + b o _ {2} ^ {3 i}) o _ {1} ^ {3 i - 2} o _ {1} ^ {3 i} y _ {i} x _ {i} ^ {\top} o _ {1} ^ {3 n + 1} o _ {2} ^ {3 n + 3} x _ {\mathrm{test}} \\ + c \sum_ {j = 1} ^ {n} \sum_ {k = 1} ^ {n} \left(\sum_ {i = 1} ^ {n + 1} o _ {2} ^ {3 i - 1} d _ {j, i} d _ {k, i}\right) o _ {1} ^ {3 j} o _ {1} ^ {3 k - 2} y _ {j} x _ {k} ^ {\top} o _ {1} ^ {3 n + 1} o _ {2} ^ {3 n + 3} x _ {\text { test }}, \tag {41} \\ \mathsf {T F} (Z _ {0}; \{V _ {l}, Q _ {l} \} _ {l = 1} ^ {2}, X _ {1} \leftarrow \Delta X _ {1}) = c o _ {2} ^ {3 n + 3} \sum_ {j = 1} ^ {n} \left(\sum_ {i = 1} ^ {n + 1} d _ {j, i} d _ {n + 1, i}\right) o _ {1} ^ {3 j} o _ {1} ^ {3 n + 1} y _ {j} x _ {\mathrm{test}} ^ {\top} x _ {\mathrm{test}}, \\ \end{array}
$$

where $a = b _ { 2 } c _ { 2 } a _ { 1 } d _ { x } ^ { y } b _ { 1 } d _ { y } ^ { x } ( 1 + a _ { 1 } d _ { x } ^ { x } ) , b = b _ { 2 } c _ { 2 } a _ { 1 } d _ { x } ^ { y } ( 1 + b _ { 1 } d _ { y } ^ { x } ) a _ { 1 } d _ { x } ^ { x } \mathrm { a n d } c = b _ { 2 } c _ { 2 } a _ { 1 } d _ { x } ^ { y } b _ { 1 } t _ { y } a _ { 1 } t _ { x } .$ One can verify that our previous analysis about TF $( Z _ { 0 } ; \{ V _ { l } , Q _ { l } \} _ { l = 1 } ^ { 2 } , X _ { 1 }  \Delta X _ { 1 } )$ still holds and we thus have $d _ { n + 1 , : } = 0$ . We then define:

$$
O _ {l} ^ {1} = \mathrm{diag} (o _ {l} ^ {1}, \dots , o _ {l} ^ {3 n - 2}) \in \mathbb {R} ^ {n \times n}, \quad O _ {l} ^ {2} = \mathrm{diag} (o _ {l} ^ {3}, \dots , o _ {l} ^ {3 n}) \in \mathbb {R} ^ {n \times n}, \quad \text { for } l \in [ 2 ],
$$

$$
O _ {2} ^ {3} = \operatorname{diag} (o _ {2} ^ {2}, \dots , o _ {2} ^ {3 n + 2}) \in \mathbb {R} ^ {(n + 1) \times (n + 1)}.
$$

By defining

$$
\widetilde {W} = \sum_ {i = 1} ^ {n} (a o _ {2} ^ {3 i - 2} + b o _ {2} ^ {3 i}) o _ {1} ^ {3 i - 2} o _ {1} ^ {3 i} y _ {i} x _ {i} ^ {\top} + c \sum_ {j = 1} ^ {n} \sum_ {k = 1} ^ {n} \left(\sum_ {i = 1} ^ {n + 1} o _ {2} ^ {3 i - 1} d _ {j, i} d _ {k, i}\right) o _ {1} ^ {3 j} o _ {1} ^ {3 k - 2} y _ {j} x _ {k} ^ {\top},
$$

One can verify that

$$
\widetilde {W} = A + B + C \triangleq a Y O _ {1} ^ {2} O _ {2} ^ {1} O _ {1} ^ {1} X ^ {\top} + b Y O _ {1} ^ {2} O _ {2} ^ {2} O _ {1} ^ {1} X ^ {\top} + c Y O _ {1} ^ {2} M O _ {2} ^ {3} M ^ {\top} O _ {1} ^ {1} X ^ {\top}.
$$

Then, we will compute the expectation of each term in the following decomposition:

$$
\mathcal {L} (V, Q) = \mathbb {E} _ {Z _ {0}, W} \Big [ \mathrm{tr} \Big (\widetilde {W} ^ {\top} \widetilde {W} \Big) + 2   \mathrm{tr} \Big (W ^ {\top} \widetilde {W} \Big) + \mathrm{tr} \big (W ^ {\top} W \big) \Big ],
$$

Specifically, let $\begin{array} { r } { T _ { 1 } = \mathrm { t r } \big ( \boldsymbol { M } \boldsymbol { M } ^ { \top } \big ) , T _ { 2 } = \mathrm { t r } \big ( \boldsymbol { M } \boldsymbol { M } ^ { \top } \boldsymbol { M } \boldsymbol { M } ^ { \top } \big ) , T _ { 3 } = \| \boldsymbol { M } \| _ { 4 } ^ { 4 } , T _ { 4 } = \sum _ { i = 1 } ^ { n } \| \boldsymbol { M } _ { i , : } \| _ { 2 } ^ { 4 } , } \end{array}$ $\begin{array} { r } { T _ { 5 } = \sum _ { j = 1 } ^ { n + 1 } \left. \boldsymbol { M } _ { : , j } \right. _ { 2 } ^ { 4 } } \end{array}$ , we then have

$$
\begin{array}{l} \mathbb {E} \left[ \operatorname{tr} \left(A ^ {\top} A\right) \right] = a ^ {2} d ^ {2} \left(n p ^ {3} + n (n - 1) p ^ {6} + (1 + d) n p ^ {3}\right), \\ \mathbb {E} \left[ \operatorname{tr} \left(B ^ {\top} B\right) \right] = b ^ {2} d ^ {2} \left(n p ^ {3} + n (n - 1) p ^ {6} + (1 + d) n p ^ {3}\right), \\ + (1 + d) (p ^ {3} - p ^ {4} - p ^ {5} + p ^ {6}) T _ {3} + (p ^ {3} - p ^ {4}) T _ {4} + p ^ {4} T _ {2} + d p ^ {6} T _ {2}), \\ \end{array}
$$

$$
\mathbb {E} [ \mathrm{tr} (C ^ {\top} C) ] = c ^ {2} d ^ {2} (p ^ {6} T _ {1} ^ {2} + (1 + d) (p ^ {4} - p ^ {6}) T _ {4} + (1 + d) (p ^ {5} - p ^ {6}) T _ {5}
$$

$$
\mathbb {E} \left[ \operatorname{tr} \left(A ^ {\top} B\right) \right] = a b d ^ {2} \left(n p ^ {4} + n (n - 1) p ^ {6} + (1 + d) n p ^ {4}\right),
$$

$$
\mathbb {E} [ \mathrm{tr} (A ^ {\top} C) ] = a c d ^ {2} ((p ^ {4} + (n - 1) p ^ {6}) T _ {1} + (1 + d) p ^ {4} T _ {1}),
$$

$$
\mathbb {E} [ \mathrm{tr} (B ^ {\top} C) ] = b c d ^ {2} ((p ^ {4} + (n - 1) p ^ {6}) T _ {1} + (1 + d) p ^ {4} T _ {1}),
$$

$$
\mathbb {E} [ \mathrm{tr} (W ^ {\top} A) ] = a d ^ {2} n p ^ {3}, \quad \mathbb {E} [ \mathrm{tr} (W ^ {\top} B) ] = b d ^ {2} n p ^ {3}, \quad \mathbb {E} [ \mathrm{tr} (W ^ {\top} C) ] = c d ^ {2} p ^ {3} T _ {1}.
$$

Summarizing our analysis above, minM $\mathcal { L } ( V , Q )$ is equivalent to:

$$
\min _ {M} \bigl \{c _ {0} + c _ {1} T _ {1} + c _ {2} T _ {2} + c _ {3} T _ {3} + c _ {4} T _ {4} + c _ {5} T _ {5} + c _ {6} T _ {1} ^ {2} \bigr \},
$$

where

$$
c _ {0} = 1 + n (2 + d) p ^ {3} (a ^ {2} + b ^ {2}) + 2 n p ^ {3} (a + b) + 2 n (2 + d) p ^ {4} a b + n (n - 1) p ^ {6} (a + b) ^ {2},
$$

$$
c _ {1} = 2 (a + b) c (p ^ {4} + (n - 1) p ^ {6} + (1 + d) p ^ {4}) + 2 c p ^ {3},
$$

$$
c _ {2} = c ^ {2} (p ^ {4} + d p ^ {6}),
$$

$$
c _ {3} = c ^ {2} (1 + d) (p ^ {3} - p ^ {4} - p ^ {5} + p ^ {6}),
$$

$$
c _ {4} = c ^ {2} ((1 + d) (p ^ {4} - p ^ {6}) + (p ^ {3} - p ^ {4})),
$$

$$
c _ {5} = c ^ {2} (1 + d) (p ^ {5} - p ^ {6}),
$$

$$
c _ {6} = c ^ {2} p ^ {6}.
$$

It is easy to verify that $c _ { 2 } , c _ { 3 } , c _ { 4 } , c _ { 5 } , c _ { 6 } \geq 0$ .

![](images/ee8c6fa1d65c196473eb691e6670b3a62ad71563cd610120ecfd1e4fe0c5b63c.jpg)

# D.7 PROOF OF PROPOSITION 6

Proposition 6 (Restate). Let $d _ { p }$ denote the number of non-EOS tokens. Given any L-layer, singlehead, d-dimensional linear-attention transformer with EOS tokens:

$$
\mathsf {T F} \big (Z _ {0}; \{V _ {l}, Q _ {l}, P _ {l} \} _ {l \in [ L ]} \big) = (Z _ {L}) _ {:, d _ {p} + 1}, \quad (Z _ {0}) _ {:, d _ {p} + 1} = 0,
$$

where

$$
Z _ {l} \in \mathbb {R} ^ {d \times (d _ {p} + 1)}, V _ {l}, Q _ {l} \in \mathbb {R} ^ {d \times d}, P _ {l} \in \mathbb {R} ^ {(d _ {p} + 1) \times (d _ {p} + 1)},
$$

$$
Z _ {l} = Z _ {l - 1} + V _ {l} Z _ {l - 1} M (Z _ {l - 1} ^ {\top} Q _ {l} Z _ {l - 1} ^ {\top} + P _ {l}), \quad M = \mathrm{diag} (I _ {d _ {p}}, 0).
$$

There exists an L-layer, two-head, 2d-dimensional linear-attention transformer operating without EOS tokens:

$$
\mathsf {T F} \big (\overline {{Z}} _ {0}; \{\overline {{V}} _ {l} ^ {h}, \overline {{Q}} _ {l} ^ {h}, \overline {{P}} _ {l} ^ {h} \} _ {l \in [ L ], h \in [ 2 ]} \big) = (\overline {{Z}} _ {L}) _ {d: 2 d, d _ {p}},
$$

where

$$
\overline {{Z}} _ {l} \in \mathbb {R} ^ {2 d \times d _ {p}},   \overline {{V}} _ {l} ^ {h}, \overline {{Q}} _ {l} ^ {h} \in \mathbb {R} ^ {2 d \times 2 d},   \overline {{P}} _ {l} ^ {h} \in \mathbb {R} ^ {d _ {p} \times d _ {p}},
$$

$$
\overline {{Z}} _ {l} = \overline {{Z}} _ {l - 1} + \sum_ {h = 1} ^ {2} \overline {{V}} _ {l} ^ {h} \overline {{Z}} _ {l - 1} (\overline {{Z}} _ {l - 1} ^ {\top} \overline {{Q}} _ {l} ^ {h} \overline {{Z}} _ {l - 1} ^ {\top} + \overline {{P}} _ {l} ^ {h}).
$$

Such that for any $Z \in \mathbb { R } ^ { d \times d _ { p } }$ , by letting $Z _ { 0 } = [ Z \quad 0 ]$ and $\overline { { Z } } _ { 0 } = \bigg [ \frac { Z } { 0 } \bigg ]$ , we have

$$
\mathsf {T F} \big (Z _ {0}; \{V _ {l}, Q _ {l}, P _ {l} \} _ {l \in [ L ]} \big) = \mathsf {T F} \big (\overline {{Z}} _ {0}; \{\overline {{V}} _ {l} ^ {h}, \overline {{Q}} _ {l} ^ {h}, \overline {{P}} _ {l} ^ {h} \} _ {l \in [ L ], h \in [ 2 ]} \big).
$$

Proof. We construct $\overline { { V } } _ { l } ^ { h } , \overline { { Q } } _ { l } ^ { h }$ , and $\overline { { P } } _ { l } ^ { h }$ as follows:

$$
\overline {{V}} _ {l} ^ {1} = \left[ \begin{array}{c c} V _ {l} & 0 \\ 0 & 0 \end{array} \right], \quad \overline {{Q}} _ {l} ^ {1} = \left[ \begin{array}{c c} Q _ {l} & 0 \\ 0 & 0 \end{array} \right], \quad \overline {{P}} _ {l} ^ {1} = (P _ {l}) _ {1: d _ {p}, 1: d _ {p}},
$$

$$
\overline {{V}} _ {l} ^ {2} = \left[ \begin{array}{c c} 0 & 0 \\ V _ {l} & 0 \end{array} \right], \quad \overline {{Q}} _ {l} ^ {2} = \left[ \begin{array}{c c} 0 & Q _ {l} \\ 0 & 0 \end{array} \right], \quad \overline {{P}} _ {l} ^ {2} = \left[ \begin{array}{c c} 0 & (P _ {l}) _ {:, d _ {p} + 1} \end{array} \right].
$$

We will show that for any l ∈ [L], it satisfies Zl = $l \in [ L ]$ $\overline { { { Z } } } _ { l } = \left\lceil \begin{array} { c c } { { \left( Z _ { l } \right) _ { : , ( 1 : d _ { p } - 1 ) } } } & { { \left( Z _ { l } \right) _ { : , d _ { p } } } } \\ { { 0 } } & { { \left( Z _ { l } \right) _ { : , d _ { p } + 1 } } } \end{array} \right\rceil$ (Zl):,dp+1 One can verify that it holds trivially for $l = 0$ . Then, suppose it holds for some $l = k - 1$ , we have

$$
\overline {{Z}} _ {k} = \overline {{Z}} _ {k - 1} + \overline {{V}} _ {k} ^ {1} \overline {{Z}} _ {k - 1} (\overline {{Z}} _ {k - 1} ^ {\top} \overline {{Q}} _ {k} ^ {1} \overline {{Z}} _ {k - 1} ^ {\top} + \overline {{P}} _ {k} ^ {1}) + \overline {{V}} _ {k} ^ {2} \overline {{Z}} _ {k - 1} (\overline {{Z}} _ {k - 1} ^ {\top} \overline {{Q}} _ {k} ^ {2} \overline {{Z}} _ {k - 1} ^ {\top} + \overline {{P}} _ {k} ^ {2})
$$

$$
= \overline {{Z}} _ {k - 1} + \left[ \begin{array}{c} V _ {k} (Z _ {k - 1}) _ {:, 1: d _ {p}} \Big ((Z _ {k - 1}) _ {:, 1: d _ {p}} ^ {\top} Q _ {k} (Z _ {k - 1}) _ {:, 1: d _ {p}} + (P _ {k}) _ {1: d _ {p}, 1: d _ {p}} \Big) \\ 0 \end{array} \right]
$$

$$
+ \left[ \begin{array}{c} 0 \\ V _ {k} (Z _ {k - 1}) _ {:, 1: d _ {p}} \end{array} \right] \left(\left[ \begin{array}{c c} 0 & (Z _ {k - 1}) _ {:, 1: d _ {p}} ^ {\top} Q _ {k} (Z _ {k - 1}) _ {:, d _ {p} + 1} \end{array} \right] + \left[ \begin{array}{c c} 0 & (P _ {k}) _ {:, d _ {p} + 1} \end{array} \right]\right)
$$

$$
= \overline {{Z}} _ {k - 1} + \left[ \begin{array}{c} V _ {k} Z _ {k - 1} M \big (Z _ {k - 1} ^ {\top} Q _ {k} (Z _ {k - 1}) _ {:, 1: d _ {p}} + (P _ {k}) _ {:, 1: d _ {p}} \big) \\ 0 \end{array} \right]
$$

$$
+ \left[ \begin{array}{c c} 0 & 0 \\ 0 & V _ {k} Z _ {k - 1} M \big (Z _ {k - 1} ^ {\top} Q _ {k} (Z _ {k - 1}) _ {:, d _ {p} + 1} + (P _ {k}) _ {:, d _ {p} + 1} \big) \end{array} \right]
$$

$$
= \left[ \begin{array}{c} (Z _ {k}) _ {:, 1: d _ {p}} \\ 0 \end{array} \right] + \left[ \begin{array}{c c} 0 & 0 \\ 0 & (Z _ {k}) _ {:, d _ {p} + 1} \end{array} \right].
$$

The proof is complete.

![](images/de1652aede0d4f9048b662b3e38a93971c44935a7b39b632e131e70c6595779b.jpg)