# CACTI: Leveraging Copy Masking and Contextual Information to Improve Tabular Data Imputation

Aditya Gorla 1 2 Ryan Wang 3 Zhengtong Liu 3 Ulzee An 1 3 Sriram Sankararaman 1 3 4

# Abstract

We present CACTI, a masked autoencoding approach for imputing tabular data that leverages the structure in missingness patterns and contextual information. Our approach employs a novel median truncated copy masking training strategy that encourages the model to learn from empirical patterns of missingness while incorporating semantic relationships between features – captured by column names and text descriptions – to better represent feature dependence. These dual sources of inductive bias enable CACTI to outperform state-of-the-art methods – an average R2 gain of 7.8% over the next best method (13.4%, 6.1%, and 5.3% under missing not at random, at random and completely at random, respectively) – across a diverse range of datasets and missingness conditions. Our results highlight the value of leveraging dataset-specific contextual information and missingness patterns to enhance imputation performance. Code is publicly available at github.com/sriramlab/CACTI

# 1. Introduction

Missingness is a pervasive problem in real-world tabular datasets with the potential to adversely affect downstream inferential tasks (Rubin, 1987; Schafer & Graham, 2002). While many techniques to estimate or impute missing entries have been proposed (see Section 2.1), missing data imputation remains a challenging problem.

A primary reason underlying this challenge is that missing-

1Department of Computational Medicine, David Geffen School of Medicine, UCLA, Los Angeles, CA, USA 2Bioinformatics Interdepartmental Program, UCLA, Los Angeles, CA, USA 3Department of Computer Science, UCLA, Los Angeles, CA, USA 4Department of Human Genetics, UCLA, Los Angeles, CA, USA. Correspondence to: Aditya Gorla <adityagorla@ucla.edu>, Sriram Sankararaman <sriram@cs.ucla.edu>.

Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

ness can arise due to a variety of mechanisms. Existing methods either explicitly or implicitly make simplifying assumptions about these mechanisms motivated by inferential tractability (see Section 2.1 and Jarrett et al. (2022)). These assumptions rarely hold in real-world settings and practitioners often lack prior knowledge on the underlying missingness mechanism. Consider a medical survey where questions are hierarchically structured, with more specific inquiries contingent upon affirmative responses to general ones so that a patient is only asked about specific symptoms if they report a broader health issue. In this example, entries pertaining to more specific health issues will be missing depending on the observed values or missingness status of entries relevant to broader health status. We hypothesize that the missingness patterns in the data could potentially be leveraged to improve imputation accuracy.

Additionally, existing methods underutilize the rich contextual information in the data. While they allow for the inclusion of fully observed covariates, they lack a straightforward mechanism to effectively incorporate unstructured knowledge about the relatedness between or the context of the features being imputed. In the medical surveys example, the answer to the broader question can constrain the answers to more specific questions. We hypothesize that imputation models can use this prior information to inform their imputation.

Contributions In this work, we present Context Aware Copy masked Tabular Imputation (CACTI), a transformerbased architecture that leverages inductive biases from observed missingness patterns and textual information about features to address existing gaps in tabular data imputation. CACTI makes several novel contributions to tabular imputation. First, we introduce median truncated copy masking (MT-CM), a novel training strategy that enables the effective application of copy masking (An et al., 2023) to transformer-based Masked Autoencoders (MAE) (He et al., 2021). Unlike existing approaches which use complete data or random masks (Du et al., 2024), MT-CM uses empirical missingness patterns to guide the learning process. Our results demonstrate that a naive application of copy masking to transformer-based MAE architectures leads to suboptimal performance while MT-CM addresses this gap. Second, we provide theoretical motivation for MAE training without fully observed data which motivates the need for copy masking. Third, we leverage contextual information from feature names and descriptions as a source of inductive bias. This context-aware approach enhances learning efficiency by minimizing reliance on learning features’ relationships solely from limited observed data and provides a direct way to incorporate unstructured information or prior knowledge. Fourth, our comprehensive evaluation establishes CACTI as a state-of-the-art tabular imputation approach across various missingness settings. Finally, both MT-CM and context awareness frameworks are simple and modular, allowing them to be used in conjunction with any deep learning framework beyond the tabular imputation domain.

# 2. Background

We begin by introducing the tabular imputation task adopting notation similar to previous works (Jarrett et al., 2022; Ipsen et al., 2021; Yoon et al., 2018) to ensure consistency. The complete data for a single sample with K features, $\mathbf { X } _ { n } \ : = \ ( x _ { n 1 } , \cdot \cdot \cdot , x _ { n K } ) \in \ { \mathcal { X } } \ = \ { \mathcal { X } } _ { 1 } \times \cdot \cdot \cdot \times \ { \mathcal { X } } _ { K }$ with $k \in [ K ]$ features and $n \in [ N ]$ observations, is drawn i.i.d from an arbitrary data generating process $\mathbf { X } _ { n } \sim { \mathcal { D } } _ { K }$ .

We do not have access to the complete data but only the incomplete, observed data: $\tilde { \mathbf { X } } _ { n } : = ( \tilde { x } _ { n 1 } , \dots , \tilde { x } _ { n K } )$ , Equation (1). The incomplete data can be viewed as a corrupted version of the complete data mediated by the missingness mask $\mathbf { M } _ { n } = ( m _ { n 1 } , \ldots , m _ { n K } ) \in \{ 0 , 1 \} ^ { K }$ , where $x _ { n k }$ is observed if $m _ { n k } = 1$ and $x _ { n k }$ is missing (denoted as ∗) if $m _ { n k } = 0 $ :

$$
\tilde {x} _ {n k} = \left\{ \begin{array}{l l} x _ {n k}, & \text { if   } m _ {n k} = 1 \\ * \quad , & \text { if   } m _ {n k} = 0 \end{array} \right. \in \tilde {\mathcal {X}} _ {k} := \mathcal {X} _ {k} \cup \{* \} \tag {1}
$$

Across N observations, this process results in an observed data matrix $\tilde { \mathbf { X } } = ( \tilde { \mathbf { X } } _ { 1 } ; \ldots ; \tilde { \mathbf { X } } _ { N } )$ and the associated mask $\mathbf { M } = ( \mathbf { M } _ { 1 } ; \ldots ; \mathbf { M } _ { N } )$ .

The imputation task can be formalized as a learning a function $f : \tilde { \mathcal { X } } \to \mathcal { X }$ resulting in an uncorrupted version of the incomplete data ${ \bar { \bf X } } _ { n } : = ( { \bar { x } } _ { n 1 } , \dots , { \bar { x } } _ { n K } )$ resulting in the final imputed dataset $( \hat { \mathbf { X } } _ { n } = ( \hat { x } _ { n 1 } , \dots , \hat { x } _ { n K } )$ , Equation (2)):

$$
\hat {x} _ {n k} = \left\{ \begin{array}{l l} x _ {n k}, & \text { if   } m _ {n k} = 1 \\ \bar {x} _ {n k}, & \text { if   } m _ {n k} = 0 \end{array} \right. \tag {2}
$$

Additionally, we might have access to additional information that can be leveraged to aid imputation. Specifically, we assume we have access to external information (shared across all N samples) such as the semantic context and relatedness between features which can be represented as $\mathbf { C } : = ( \mathbf { C } _ { 1 } , \hdots , \mathbf { C } _ { K } ) \in \mathbb { R } ^ { C \times K }$ , a C-dimensional embedding representation of context information for each feature. Missingness Mechanisms Let us define a selector function $s _ { \mathbf { M } _ { n } } : \mathcal { X } \to \prod _ { k \in \{ k : m _ { n k } = 1 \} } \mathcal { X } _ { k }$ that selects all the observed features in the complete data. $\mathbf { X } _ { n } ^ { o } : = s _ { \mathbf { M } _ { n } } ( \mathbf { X } _ { n } )$ defines the observed part and $\mathbf { X } _ { n } ^ { m } \ : = \ s _ { 1 - \mathbf { M } _ { n } } ( \mathbf { X } _ { n } )$ defines the missing part. The framework laid out by Rubin (1976) (also (Little & Rubin, 1987)) prescribes the following underlying missingness mechanisms, from the most to the least restrictive assumption: MCAR $\begin{array} { r l } { ( p ( \mathbf { M } _ { n } | \mathbf { X } _ { n } ) } & { { } = } \end{array}$ $p ( \mathbf { M } _ { n } ) , \mathbf { i . e . \mathbf { M } } _ { n } \perp \mathbf { X } _ { n } ;$ missingness is independent of the data), MAR $( p ( \mathbf { M } _ { n } | \mathbf { X } _ { n } ) = p ( \mathbf { M } _ { n } | \mathbf { X _ { n } } ^ { o } )$ ; missingness only depends on the fully observed data), and MNAR when the mechanism is neither MCAR nor MAR.

# 2.1. Related work

There are two main classes of tabular imputation methods: iterative and generative. Iterative methods iteratively impute the missing values in each feature by estimating the conditional distribution given all other features’ observed data (van Buuren & Groothuis-Oudshoorn, 2011; Stekhoven & Buhlmann ¨ , 2011; Jarrett et al., 2022). While estimating the conditional distribution is a simpler problem, these approaches are limited by challenges in selecting optimal conditional distributions and sometimes requiring complete observations for model fitting. In contrast, generative approaches attempt to estimate a joint distribution of all the features which is a considerably harder statistical task than estimating univariate conditional probabilities (Yoon et al., 2018; Dai et al., 2021; Yoon & Sull, 2020; Mattei & Frellsen, 2019; Ipsen et al., 2021; Nazabal et al., 2020; Zhang et al., 2024a; Zheng & Charoenphakdee, 2023; Muzellec et al., 2020). Many of these approaches require either complete data or restrictive assumptions on the missingness mechanisms (Nazabal et al., 2020; Richardson et al., 2020; Mattei & Frellsen, 2019). Other classical imputation approaches include: K-nearest neighbors, matrix completion and unconditional mean substitution (Hastie et al., 2014; Hawthorne & Elliott, 2005).

Transformers Several recent works have proposed transformer (or self-attention) based architectures to model tabular data (Huang et al., 2020; Arik & Pfister, 2020; Majmundar et al., 2022; Yoon et al., 2020; Hollmann et al., 2025; Gardner et al., 2024). However, these approaches primarily focus on self-supervised learning tasks by employing a masked reconstruction task or target direct downstream prediction and do not explicitly address the imputation problem. Recent works (Yin et al., 2020; Yang et al., 2024; Lin et al., 2024; An et al., 2025) have also leveraged unstructured (natural language) contextual awareness to improve representation learning, pre-training efficiency and the performance of generative tabular models; however, these approaches have not yet been effectively leveraged in tabular imputation.

![](images/6710a594c6d3f3d6ead60341497bcff5a443ffc829eb92f0a537d77cc66938e9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Observed Data"] --> B["Masked Data"]
    B --> C["Random Masking"]
    B --> D["Naive Copy Masking"]
    B --> E["Median Truncated Copy Masking"]
    F["* : Missing Cell Values"] --> G["x11 x12 x13 x14 x15"]
    H["X : Masked Cell Values"] --> I["x21 * x23 * *"]
    J["X : Observed Cell Values"] --> K["x32 * x34 * x35"]
    G --> L["x11 x12 x13 x14 x15"]
    H --> M["x21 * x23 * *"]
    I --> N["x32 * x34 * x35"]
    L --> O["x11 x12 x13 x14 x15"]
    M --> P["x21 * x23 * *"]
    N --> Q["x32 * x34 * x35"]
    O --> R["x11 x13"]
    P --> S["x21 x23"]
    Q --> T["x32 x34"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#cfc,stroke:#333
    style L fill:#cfc,stroke:#333
    style M fill:#cfc,stroke:#333
    style N fill:#cfc,stroke:#333
    style O fill:#cfc,stroke:#333
    style P fill:#cfc,stroke:#333
    style Q fill:#cfc,stroke:#333
    style R fill:#cfc,stroke:#333
```
</details>

Figure 1. Median Truncated Copy Masking overview. In contrast to random masking, where some subset of features are masked uniformly at random, copy masking recycles missing value patterns actually present in the dataset. This approach simulates realistic missingness patterns that provide a source of useful inductive bias during training. Median Truncated Copy Masking extends this strategy for MAE training by truncating the number of features available to the encoder, ensuring it has access to at most the median number of fully observed features in each batch.

ReMasker (Du et al., 2024), a transformer-based approach for tabular imputation that builds on the MAE approach (He et al., 2021), learns to reconstruct randomly masked values based on the unmasked observed values (Figure 1). The model is highly expressive but is trained under a (completely) random masking strategy during training.

Copy masking Recent work by An et al. (2023) proposed using the missingness patterns in the observed data to create masks to train an imputation model under a reconstruction loss function. Given an observed missingness mask M, copy maksing involves shuffling the matrix row-wise to create a mask $\mathbf { \bar { M } } ^ { p e r m } \in \{ 0 , 1 \} ^ { N \bar { \times } K }$ where with probability $p _ { c m }$ (masking ratio) we either apply the ${ \bf M } ^ { p e r m }$ mask for a sample or leave it unchanged. We term the resulting mask matrix as the naive copy mask ${ \bf M } ^ { c m }$ (Figure 1; See Algorithm 1 for details1). While Autocomplete (An et al., 2023) implements naive copy masking in conjunction with a shallow MLP to show strong downstream performance, this approach is limited in its expressivity to learn complex relational patterns between the features.

# 3. CACTI

CACTI employs an encoder-decoder Transformer architecture for tabular data imputation. This architecture needs to be trained on a reconstruction task. However, since the observed data is incomplete, a masking strategy that introduces additional missingness on which the quality of reconstruction can be assessed must be devised.

# 3.1. Median truncated copy masking

Previous works for tabular data imputation (Du et al., 2024) adopt the same approach used in MAEs: applying a random mask on the observed portions of the incomplete data during training. Our first contribution is in replacing random masking. We extend naive copy masking (An et al., 2023) to develop median truncated copy masking (MT-CM) which leverages the missingness structure in the observed data to create masks that better reflect true missingness patterns. We hypothesize that this approach provides a useful inductive bias for the model that can be particularly effective in cases where missingness is structured (Jackson et al., 2023), e.g., consider the missingness pattern $p ( m _ { n i } = 0 | m _ { n j } = 0 ) = 1$ where feature i is missing any time $j$ is missing. While it is challenging to define a unified or well-defined generative model for the mask, the empirical patterns of missingness provide useful information to design such a mask.

Under MT-CM (and naive copy masking), we can segregate features in each sample into three sets: the (observed but) masked values $\mathbb { M } _ { n } ^ { c m } = \left\{ k : \left( m _ { n k } = 1 \right) \cap \left( m _ { n k } ^ { c m } = 0 \right) \right\}$ , the unmasked values $\mathbb { O } _ { n } ^ { c m } = \{ k : ( m _ { n k } = 1 ) \cap ( m _ { n k } ^ { c m } = 1 ) \}$ and the true missing values $\mathbb { V } _ { n } = \left. k : m _ { n k } = 0 \right.$ . Consequently, we can define a training strategy by minimizing a reconstruction loss over the value sets $\mathbb { M } _ { n } ^ { c m }$ and $\mathbb { O } _ { n } ^ { c m }$ .

A naive application of copy masking (Figure 1) to transformer-based MAE architectures, however, leads to inefficient learning due to the large variance in missingness proportions across samples (Mitra et al., 2023; Jackson et al., 2023; An et al., 2023) while uniform feature sizes (or sequence lengths) within a batch are critical for efficient learning with a transformer-based encoder (Krell et al., 2022). A possible approach to enforce uniformity when using naive copy masking is to replace all missing or masked features with a null token. This strategy, even at low copy masking rates, results in a significant proportion of null tokens in each batch, which provides no meaningful information for learning a robust latent representation. Furthermore, increasing the copy masking rate proportionally increases the fraction of null tokens that can further reduce learning efficiency and overall model performance. Empirical results confirm this trend, with higher rates of naive copy masking leading to reduced model performance (see Appendix A).

To tackle this issue, we propose the Median Truncation Copy

![](images/fc10cdc35ae4d2bdb078efd3956d368d828963d6f0dabbf462e42be17b405dcc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Observed missingness pattern"] --> B["Copy observed missingness patterns"]
    B --> C["Median Truncated Copy Mask (MT-CM)"]
    C --> D["Input"]
    D --> E["MT-CM Context Aware Input Embeddings"]
    E --> F["Encoder"]
    F --> G["Latent Representation"]
    G --> H["Context Aware Latent Embeddings"]
    H --> I["Decoder"]
    I --> J["Imputed Values"]

    K["Context Information"] --> L["Language model embeddings"]
    L --> M["Feature 1 description"]
    L --> N["Feature 2 description"]
    L --> O["Feature 3 description"]
    L --> P["Feature 4 description"]
    L --> Q["Feature 5 description"]
    L --> R["Feature 6 description"]

    S["Input"] --> T["V21 V22 V23 C2"]
    T --> U["V31 V32 V33 C3"]
    U --> V["V61 V62 V63 C6"]

    W["Encoder"] --> X["L2 L3 L6"]

    Y["Latent Representation"] --> Z["[MASK"] [MASK] [MASK] C1]
    Z --> AA["Z21 Z22 Z33 C2"]
    AA --> AB["Z31 Z32 Z33 C3"]
    AB --> AC["[MASK"] [MASK] [MASK] C4]
    AC --> AD["[MASK"] [MASK] [MASK] C5]
    AD --> AE["Z61 Z62 Z63 C6"]

    AF["Context Aware Latent Embeddings"] --> AG["[MASK"] [MASK] [MASK] C1]
    AG --> AH["Z21 Z22 Z33 C2"]
    AH --> AI["Z31 Z32 Z33 C3"]
    AI --> AJ["[MASK"] [MASK] [MASK] C4]
    AJ --> AK["[MASK"] [MASK] [MASK] C5]
    AK --> AL["Z61 Z62 Z63 C6"]

    AM["Decoder"] --> AN["[MASK"] [MASK] [MASK] C1]
    AN --> AO["Z21 Z22 Z33 C2"]
    AO --> AP["Z31 Z32 Z33 C3"]
    AP --> AQ["[MASK"] [MASK] [MASK] C4]
    AQ --> AR["[MASK"] [MASK] [MASK] C5]
    AR --> AS["Z61 Z62 Z63 C6"]

    AT["Imputed Values"] --> AU["[Xn1 Xn2 Xn3 Xn4 Xn5 Xn6"]]
    AU --> AV["LOn = ∑k=1{|O_n| / |O_n|"]
    AV --> AW["L_Mn = ∑k=1{|M_n| / |M_n|}"]
    AW --> AX["L_CACTI(Ñ,Ñ) = L_On + L_Mn"]
    AX --> AY["O_n: observed values, Mn: masked values"]
```
</details>

Figure 2. CACTI model overview. CACTI samples observed missingness patterns to generate masks via Median Truncated Copy Masking (MT-CM) to guide the learning. Features’ context are also embedded with a language model. The MT-CM strategy masks out some portion of the observed features from sample n using observed missingness patterns from other samples (j) in the same dataset. This is followed by concatenating context information to the remaining (unmasked) features. A transformer encoder processes this data. Then the model adds context information and [MASK] tokens for the missing/masked features before being processed by the decoding transformer which reconstructs the values. CACTI optimizes reconstruction loss $( \mathcal { L } _ { C A C T I } )$ ) over observed and masked features to produce the final imputation estimates.

Masking (MT-CM) training strategy (Figure 1). Let $N _ { B }$ be the number of samples in the B-th batch, and $o _ { n } = | \mathbb { O } _ { n } ^ { c m } |$ be the number of observed features in the n-th sample after the application of naive copy masking in this batch. The median number of observed values within the batch is defined as $o _ { B } ^ { m e d i a n } = \mathrm { m e d i a n } ( o _ { 1 } , \dots , o _ { N _ { B } } )$ .

The MT-CM strategy truncates the sequence length of observed values for each sample to ensure it contains no more than $o _ { B } ^ { m e d i a n }$ observed values. Formally, for each sample n in the batch, the truncated sequence length $o _ { n } ^ { t r u n c }$ is computed as $o _ { n } ^ { t r u n c } = \operatorname* { m i n } ( o _ { n } , o _ { B } ^ { m e d i a n } ) ^ { 2 }$ . This ensures that the proportion of null tokens in any batch is upper bounded by 50% regardless of the copy masking rate. Overall, MT-CM results in the final set of observed and masked features: $\mathbb { O } _ { n } = \{ k : ( m _ { n k } = 1 ) \cap ( m _ { n k } ^ { c m } = 1 ) \cap ( k \leq o _ { n } ^ { t r u n c } ) \}$ } and $\mathbb { M } _ { n } = \mathbb { M } _ { n } ^ { c m } \cup ( \mathbb { O } _ { n } ^ { c m } \backslash \mathbb { O } _ { n } )$ , respectively. During training, the feature order of each sample in every batch is permuted to ensure that the first $o _ { n } ^ { t r u n c }$ features are retained as observed features and is different every iteration. See Algorithm 2 for extended MT-CM details. We also empirically show that, unlike naive copy masking, our MT-CM strategy results in overall performance increasing as the copy masking rate increases (see Appendix A).

# 3.1.1. THEORETICAL MOTIVATION FOR COPY MASKING

In this section, we provide a brief theoretical motivation of the need for copy masking.

Assume the complete data for a single sample is drawn from an arbitrary data generating distribution $\mathbf { X } \overset { i . i . d } { \sim } P _ { X } ( \mathbf { x } )$ . This complete data vector undergoes a corruption process mediated by a missingness mask which results in the partially observed data: $\tilde { \mathbf { X } } = \mathbf { X } \odot$ M where the missingness mask process $\mathbf { M } | \mathbf { X } \sim P _ { M | X }$ .

Under a masked autoecoding model, we aim to learn an encoder-decoder $( f _ { \psi }$ and $d _ { \theta }$ respectively) that minimizes the risk:

$$
R (\psi , \theta) =
$$

$$
\mathbb {E} _ {\mathbf {X}, \mathbf {M}} \left[ | | \mathbf {X} \odot (1 - \mathbf {M}) - d _ {\theta} \left(f _ {\psi} (\mathbf {X} \odot \mathbf {M})\right) \odot (1 - \mathbf {M}) | | _ {2} ^ {2} \right] \tag {3}
$$

Here ⊙ denotes entrywise product.

The risk (or its finite-sample approximation) defined in Equation 3 cannot be computed since we only observe X˜ . Instead, given the missing data X˜ , we generate a mask $\mathbf { M } ^ { \prime } | \tilde { \mathbf { X } } , \mathbf { M } \sim Q _ { \mathbf { M } ^ { \prime } | \tilde { \mathbf { X } } , \mathbf { M } }$ and aim to minimize the alternate risk:

$$
\begin{array}{l} R _ {Q} (\psi , \theta) = \mathbb {E} _ {\mathbf {X}, \mathbf {M}} \left[ \mathbb {E} _ {\mathbf {M} ^ {\prime} | \tilde {\mathbf {X}}, \mathbf {M}} [ | | \mathbf {X} \odot \mathbf {M} \odot (1 - \mathbf {M} ^ {\prime}) \right. \\ \left. - d _ {\theta} \left(f _ {\psi} \left(\mathbf {X} \odot \mathbf {M} \odot \mathbf {M} ^ {\prime}\right)\right) \odot \mathbf {M} \odot \left(1 - \mathbf {M} ^ {\prime}\right) \mid | _ {2} ^ {2} \right] \bigg ] \tag {4} \\ \end{array}
$$

Consider a sample that is completely observed so that $\mathbf { M } =$ 1 so that $\tilde { \mathbf { X } } = \bar { \mathbf { X } }$ . On this sample, $R _ { Q }$ becomes:

$$
\begin{array}{l} R _ {Q} (\psi , \theta) = \mathbb {E} _ {\mathbf {X}, \mathbf {M} = \mathbf {1}} \left[ \mathbb {E} _ {\mathbf {M} ^ {\prime} | \mathbf {X}, \mathbf {M} = \mathbf {1}} [ | | \mathbf {X} \odot (1 - \mathbf {M} ^ {\prime}) \right. \\ \left. - d _ {\theta} (f _ {\psi} (\mathbf {X} \odot \mathbf {M} ^ {\prime})) \odot (1 - \mathbf {M} ^ {\prime}) | | _ {2} ^ {2} ] \right] \tag {5} \\ \end{array}
$$

Equation 5 motivates choosing Q to be the same distribution as M|X so that $R _ { Q } \approx R$ . More broadly, this motivates choosing a masking distribution Q that approximates the true distribution of missing entries M|X. For example, if the true missingness mechanism is MCAR where the probability of each feature being missing is independent and identically distributed, a random masking strategy where each entry masked independent of other features with a constant probability is expected to provide an appropriate inductive bias for the imputation model.

These observations drive the core rationale for copy masking. Copy masking tries to approximate the true masking distribution and missingness structure by sampling from the observed missingness mask. For example, in the MCAR setting where each feature has a constant probability of being missing, copy masking will reduce to random masking. On the other hand, when the missingness probability varies across features, copy masking will lead to features being masked with differential probabilities based on their empirical frequencies. The use of empirical masks can also capture correlations among features in the missingness mechanism. While copy masking is still a simplification and does not fully model the missingness mechanism, our empirical results suggest that it enables the imputation model to attend to realistic patterns of missingness.

# 3.2. Context Awareness

Our second key contribution is making the imputation backbone context aware by incorporating prior information about the semantic information associated with each feature by using language model embedding of feature description into the value embedding vector in both the encoder and decoder stage. We use the semantic similarity between column name and description information to make our imputation backbone context-aware, providing useful inductive bias to improve imputation performance.

Let $\tilde { \mathbf { X } } _ { n } : = ( \tilde { \mathbf { X } } _ { n 1 } , \ldots , \tilde { \mathbf { X } } _ { n K } )$ be the observed data for a single sample in X˜. For the final model embedding dimension $E ,$ we would like to achieve a context-aware embedding of the data sample $\mathbf { E } _ { n } = ( \mathbf { E } _ { n 1 } , \ldots , \mathbf { E } _ { n K } ) \in \mathbb { R } ^ { E \times K }$ (where $E = \dim ( \mathbf { E } _ { n k } ) )$ . To achieve this, we can create a partitioned embedding for each feature, which has a value component ${ \bf U } _ { n k }$ and context component ${ \bf C } _ { n k }$ such that ${ \bf { E } } _ { n k } = ( { \bf { U } } _ { n k } ; { \bf { C } } _ { k } ) ^ { 3 }$ , where $\mathbf { U } _ { n k } \ \in \ \mathbb { R } ^ { U } , \mathbf { C } _ { k } \ \in \ \mathbb { R } ^ { C }$ and $E = U + C$ . As a design choice, we set $U = 0 . 7 5 E$ and $C = 0 . 2 5 E$ , prioritizing value information as the primary object of relevance which warrants its overrepresentation relative to context information. We define a linear projection ${ } ^ { 4 } \ l : \tilde { \mathcal { X } } \to \mathcal { U }$ that maps each scalar feature value to a U-dimensional embedding vector representation, resulting in $\mathbf { U } _ { n } = ( \mathbf { U } _ { n 1 } , \ldots , \mathbf { U } _ { n K } ) \in \mathbb { R } ^ { U \times K }$ .

We propose using of language models to obtain representa-

tions (embeddings) of each column’s semantic information. For each of the K columns in the data X˜ , we process the column name and description (when available) through a language model (using default tokenizer) to obtain embeddings $\mathbf { C } _ { k } ^ { c i }$ . Given a set of tokenized descriptions $\mathcal { T } _ { k } ,$ we obtain the last layer hidden state for each token and aggregate the information to obtain the column’s semantic context $\begin{array} { r } { \mathbf { C } _ { k } ^ { c i } = \frac { 1 } { | T _ { k } | } \sum _ { i = 1 } ^ { | T _ { k } | } } \end{array}$ |Tk| Embd $\left( t _ { k i } \right)$ . Since language mod-|Tk| els (Devlin et al., 2019; Lee et al., 2025) typically have hidden state dimensions in the range [768, 4096], we perform a linear projection $r _ { e } : \mathcal { C } ^ { c i }  \mathcal { C }$ that maps each column information embedding to an C-dimensional context embedding, resulting in $\mathbf { C } ^ { \mathsf { ^ { - } } } = ( \mathbf { C } _ { 1 } , \ldots , \mathbf { C } _ { K } ) \in \mathbb { R } ^ { C \times K }$ . Transformers also require fixed sin-cosine embeddings $\mathbf { P } = ( \mathbf { P } _ { 1 } , \dots , \mathbf { P } _ { K } ) \in \mathbb { R } ^ { E \times K }$ to preserve positional information (Dufter et al., 2021). Thus final context-aware embeddings are achieved by concatenation of the value and context ${ \bf E } _ { n } = [ { \bf U } _ { n } | | { \bf C } ] + { \bf P }$ , with positional information added.

Different base models can be used for generating context embeddings. In this study, we use the GTE-en-MLM-large, a new state-of-the-art text embedding model (Zhang et al., 2024b) as the default based on our empirical results comparing the effectiveness of these models. We note that the generation of column context embeddings has a one-time, fixed cost. These embeddings can be pre-computed and reused across multiple runs for the same dataset.

# 3.3. Transformer Backbone

Figure 2 provides a pictorial description of the CACTI autoencoder architecture backbone with a detailed description deferred to Appendix B. Briefly, the CACTI backbone consists of an encoder and decoder, both utilizing transformer architectures with (residual) self-attention blocks. The encoder processes context-aware embeddings $( \mathbf { E } _ { n } )$ of the observed data, dropping missing or masked features after applying the MT-CM strategy. The decoder combines context information embeddings and a latent representation of the MT-CM input features to estimate the uncorrupted version $( { \bar { \mathbf { X } } } _ { n } )$ of the incomplete data. The model is trained to minimize the reconstruction loss between the imputed and observed data, using a unified MSE loss over the masked $( \mathbb { M } _ { n } )$ and fully observed values $( \mathbb { O } _ { n } )$ . See Algorithm 3 and Algorithm 4 for a sketch of the CACTI implementation.

# 4. Evaluation Results

We empirically evaluate CACTI’s performance against stateof-the-art methods using 10 benchmarking datasets across all three missingness scenarios. Next, we conduct a thorough ablation analysis to quantify the contributions of our proposed MT-CM and context awareness strategies. Finally, we conduct a comprehensive sensitivity analysis to identify key aspects and hyperparameter configurations that significantly impact the performance and usability of our method.

Table 1. Overall benchmark results. Average performance comparison of CACTI and CMAE (CACTI without context) against existing imputation methods on the train/test splits (separated by |) over 10 datasets. Metrics (arrows indicate direction of better performance) are evaluated under MAR, MCAR, and MNAR at 30% missingness. – indicates method which cannot perform out-of-samples (test split) imputation. Best metric in bold and second best underlined. Extended table with standard errors in Appendix. 

<table><tr><td rowspan="2">METHOD</td><td colspan="3"> $R^2$ (↑)</td><td colspan="3">RMSE (↓)</td><td colspan="3">WD (↑)</td></tr><tr><td>MCAR</td><td>MAR</td><td>MNAR</td><td>MCAR</td><td>MAR</td><td>MNAR</td><td>MCAR</td><td>MAR</td><td>MNAR</td></tr><tr><td>CACTI (OURS)</td><td>0.46|0.46</td><td>0.47|0.47</td><td>0.46|0.46</td><td>0.66|0.64</td><td>0.67|0.69</td><td>0.68|0.67</td><td>4.35|4.46</td><td>1.87|1.94</td><td>4.45|4.57</td></tr><tr><td>CMAE (OURS)</td><td>0.44|0.45</td><td>0.46|0.46</td><td>0.44|0.44</td><td>0.67|0.65</td><td>0.69|0.70</td><td>0.70|0.69</td><td>4.40|4.50</td><td>1.94|2.02</td><td>4.57|4.69</td></tr><tr><td>REMASKER</td><td>0.44|0.44</td><td>0.44|0.44</td><td>0.40|0.40</td><td>0.68|0.67</td><td>0.69|0.71</td><td>0.73|0.71</td><td>4.62|4.72</td><td>2.46|2.52</td><td>4.79|4.93</td></tr><tr><td>DIFFPUTER</td><td>0.40|0.42</td><td>0.39|0.43</td><td>0.36|0.37</td><td>0.73|0.70</td><td>0.77|0.75</td><td>0.79|0.77</td><td>4.53|4.56</td><td>2.55|2.38</td><td>4.79|4.90</td></tr><tr><td>HYPERIMPUTE</td><td>0.41|-</td><td>0.44|-</td><td>0.39|-</td><td>0.72|-</td><td>0.73|-</td><td>0.76|-</td><td>4.26|-</td><td>2.46|-</td><td>4.30|-</td></tr><tr><td>MISSFOREST</td><td>0.35|0.34</td><td>0.38|0.36</td><td>0.34|0.32</td><td>0.77|0.75</td><td>0.79|0.82</td><td>0.79|0.78</td><td>6.78|6.80</td><td>3.80|3.83</td><td>7.01|7.06</td></tr><tr><td>NOTMIWAE</td><td>0.35|0.35</td><td>0.35|0.35</td><td>0.29|0.30</td><td>0.75|0.74</td><td>0.80|0.82</td><td>0.82|0.80</td><td>5.56|5.60</td><td>2.38|2.39</td><td>6.26|6.20</td></tr><tr><td>SINKHORN</td><td>0.28|-</td><td>0.29|-</td><td>0.26|-</td><td>0.84|-</td><td>0.89|-</td><td>0.88|-</td><td>7.02|-</td><td>3.96|-</td><td>7.51|-</td></tr><tr><td>ICE</td><td>0.28|0.27</td><td>0.34|0.33</td><td>0.26|0.25</td><td>0.86|0.87</td><td>0.78|0.83</td><td>0.93|0.93</td><td>4.82|5.18</td><td>2.74|2.81</td><td>5.32|5.67</td></tr><tr><td>AUTOCOMPLETE</td><td>0.24|0.24</td><td>0.29|0.29</td><td>0.21|0.21</td><td>0.88|0.86</td><td>0.88|0.89</td><td>0.94|0.92</td><td>10.14|10.18</td><td>5.04|5.07</td><td>10.44|10.42</td></tr><tr><td>MICE</td><td>0.19|0.19</td><td>0.23|0.23</td><td>0.18|0.18</td><td>1.06|1.04</td><td>1.04|1.05</td><td>1.08|1.07</td><td>8.25|8.33</td><td>4.16|4.23</td><td>8.34|8.49</td></tr><tr><td>GAIN</td><td>0.19|0.21</td><td>0.18|0.22</td><td>0.17|0.18</td><td>0.91|0.86</td><td>0.95|0.93</td><td>1.01|0.96</td><td>7.73|7.34</td><td>4.44|4.10</td><td>9.53|9.14</td></tr><tr><td>SOFTIMPUTE</td><td>0.09|0.10</td><td>0.10|0.11</td><td>0.09|0.09</td><td>1.02|0.96</td><td>1.06|1.02</td><td>1.05|0.99</td><td>8.35|7.86</td><td>4.84|4.46</td><td>8.73|8.23</td></tr><tr><td>MIWAE</td><td>0.00|0.00</td><td>0.00|0.00</td><td>0.00|0.00</td><td>1.00|0.98</td><td>1.05|1.07</td><td>1.03|1.00</td><td>7.83|7.90</td><td>4.53|4.57</td><td>8.36|8.37</td></tr><tr><td>MEAN</td><td>0.00|0.00</td><td>0.00|0.00</td><td>0.00|0.00</td><td>0.95|0.93</td><td>1.00|1.02</td><td>0.98|0.95</td><td>11.96|12.00</td><td>6.35|6.38</td><td>12.25|12.26</td></tr></table>

![](images/ca6b595804d60513b88d3ba522b1a42abb37ee0a88c02e3ba17e3f7b7a0dc784.jpg)

<details>
<summary>violin</summary>

| model      | Method 1 | Method 2 | Method 3 | Method 4 | Method 5 |
|------------|----------|----------|----------|----------|----------|
| MissForest | 0.60     | 0.70     | 0.80     | 0.90     | 0.95     |
| HyperImpute| 0.75     | 0.85     | 0.95     | 0.98     | 0.99     |
| DiffPuter  | 0.65     | 0.75     | 0.85     | 0.92     | 0.97     |
| ReMasker   | 0.55     | 0.65     | 0.75     | 0.85     | 0.92     |
| CACTI      | 0.45     | 0.55     | 0.65     | 0.75     | 0.82     |
</details>

Figure 3. Top-5 methods benchmark. Violin plots display the distribution of $R ^ { 2 }$ metrics for each of the top five methods across all missing percentages and conditions over all datasets.

Baseline methods We benchmark CACTI against 13 top methods from the field. Detailed descriptions of the methods are defered to Appendix C.1. Briefly, we compare against ReMasker (Du et al., 2024) as the primary masked transformer-based autoencoing method. Diff-Puter (Zhang et al., 2024a) represents the recent state-of-theart in diffusion-based imputation. AutoComplete (An et al., 2023) is a naive copy masking autoencoder model developed for biomedical data. Hyperimpute (Jarrett et al., 2022) is the current best iterative hybrid machine learning approach. We also compared to leading iterative methods: Missforest (Stekhoven & Buhlmann ¨ , 2011), ICE (Royston & White, 2011) and MICE (van Buuren & Groothuis-Oudshoorn, 2011). and generative approaches: Sinkhorn (Muzellec et al., 2020), GAIN (Yoon et al., 2018), MIWAE (Mattei & Frellsen, 2019) and notMIWAE (Ipsen et al., 2021) (an extention of MIWAE for MNAR). Lastly, we also include widely-used approaches such as Softimpute (Hastie et al., 2014) and unconditional Mean (Hawthorne & Elliott, 2005). For all baselines, we use default (or recommended if available) settings for all models and CACTI default settings are outlined in Appendix C.2.

Datasets To allow for comparison with previous works (Section 2.1), we use ten real-world datasets (Kelly et al.), with details included in Appendix C.3. For each dataset, we create an 80-20 train-test split to test both in-sample and outof-sample imputation. The data is fully observed, so we can simulate missingness under each of three missingness conditions: MCAR, MAR, and MNAR. For MCAR, each value is masked according to a Bernoulli random variable with fixed mean. In MAR, a random subset of features are fixed as fully observed while entries in the remaining features are masked based on a logistic model. For MNAR, we take the input features of the MAR mechanism and further mask them according to a Bernoulli random variable with fixed mean. In accordance with prior work, the primary benchmarking is performed under 30% simulated missingness proportion while extended results are included for 10%, 50% and 70% simulated missingness proportions. Simulations were performed using the HyperImpute package (Jarrett et al., 2022).

Evaluations metrics We evaluate imputation performance along three metrics: Pearson’s $R ^ { 2 }$ , root mean square error (RMSE) and Wasserstein distance (WD). We use $R ^ { 2 }$ as a measure of imputation concordance due to its invariance to mean or scale shifts and its direct applicability across continuous, binary, and ordinal features. For consistency with previous works (Zhang et al., 2024a; Jarrett et al., 2022), we report RMSE as an absolute measure of imputation accuracy and WD as a measure of alignment between the imputed and true values. Note that we perform evaluation on the original scale of each feature as opposed to the min-max transformed scale to reflect utility in real world applications. The main tables and figures present the mean of the metrics aggregated across the test split of the relevant datasets except for Table 1, which reports metrics on both the train and test splits. The Appendix contains figures reporting all the per-dataset metrics with 95% confidence intervals on both train and test splits.

Table 1 summarizes the average performance of each model across the 10 datasets under 30% simulated missingness. CACTI outperforms all existing baselines across all metrics and missingness conditions. We observe an average relative improvement over the next best method (with respect to $R ^ { 2 } )$ of 13.4%, 6.1%, and 5.3% under MNAR, MAR, and MCAR, respectively. Notably in Appendix D we also observe that across the ten datasets under all three missingness mechanisms, CACTI dominates all methods in at least one of the three metrics and in a majority of the datasets outperform all other methods on all metrics. In our experiments, we also include median truncated Copy Masked Auto Encoder (CMAE; CACTI without context) as an additional baseline to demonstrate that CMAE alone consistently dominates ReMasker across R2 and RMSE metrics.

These results underscore the versatility of our approach in achieving effective imputation across diverse missingness scenarios without strong assumptions about the source of the missingness. The robust performance, particularly in the most challenging MNAR settings, highlights the advantage of leveraging inductive biases from observed data through the use of the MT-CM training strategy. Additionally, the improved accuracy of CACTI over ReMasker under MCAR, where MT-CM and random masking should be approximately equivalent, highlights benefits of context awareness.

In Appendix D, we extend our benchmarking to 10%, 50% and 70% simulated missingness proportions for all three mechanisms. Figure 3 displays the results of the top five methods for each dataset under each of the four missingness percentages and three mechanisms. The results show that, on average, CACTI is the most effective imputation approach across all settings. Lastly, we verify in Table A14 that the resource requirements while training CACTI are reasonable (< 5.8 seconds per epoch on the largest dataset and requiring < 300MB of GPU memory both of which are comparable to that of ReMasker).

Table 2. Ablation analysis. Comparison of models with RM, MT-CM, and/or CTX. ✓ indicates model has the feature and × if not. Metrics presented represent the average model performance at 30% missingness. 

<table><tr><td rowspan="2">MODEL</td><td rowspan="2">RM</td><td rowspan="2">CTX</td><td rowspan="2">MT-CM</td><td colspan="2"> $R^{2}$  (↑)</td><td colspan="2">RMSE (↓)</td></tr><tr><td>MAR</td><td>MNAR</td><td>MAR</td><td>MNAR</td></tr><tr><td>RMAE</td><td>√</td><td>×</td><td>×</td><td>0.21</td><td>0.20</td><td>1.00</td><td>1.03</td></tr><tr><td>RMAE+CTX</td><td>√</td><td>√</td><td>×</td><td>0.26</td><td>0.26</td><td>0.96</td><td>0.86</td></tr><tr><td>CMAE</td><td>×</td><td>×</td><td>√</td><td>0.46</td><td>0.43</td><td>0.68</td><td>0.70</td></tr><tr><td>CACTI</td><td>×</td><td>√</td><td>√</td><td>0.46</td><td>0.45</td><td>0.67</td><td>0.68</td></tr></table>

# 4.1. Ablation Analysis

We aim to investigate the relative contributions of key aspects of CACTI via a series of ablation analyses.First, we assess the relative contribution of MT-CM compared to random masking (RM). Second, we evaluate the impact of context awareness (CTX) when used in conjunction with random masking alone. Third, we analyze the additional gains achieved by incorporating context awareness on top of our MT-CM training strategy. Finally, we explore the value of each of the sources of inductive bias: the observed missingness patterns or the features’ context information. To do this, we construct three additional models: 1) Random Masking Auto Encoder (RMAE), 2) Random Masking Auto Encoder with ConTeXt awareness (RMAE+CTX) and 3) CMAE. The RMAE model uses the same transformer backbone as CACTI while using the same random masking strategy as ReMasker, RMAE+CTX extends the RMAE model with the same context aware (CTX) embeddings used in CACTI, and CMAE is the CACTI model without the CTX embeddings. We conduct the ablation analysis over four different datasets (see Appendix C.3) under MAR and MNAR with masking ratio fixed at 90%.

The ablation results in Table 2 first indicate that both MT-CM and context awareness are essential for achieving good performance. Next, we observe that, under MNAR, MT-CM provides a 115% gain in $R ^ { 2 }$ over random masking while context awareness provides a 30% gain when used with random masking. We note since this is an internal ablation, all hyperparameters were held constant. This resulted in the performance of RMAE being lower than ReMasker due to differences in their masking rates. In Table A15, we conducted additional direct comparisons between CMAE and ReMasker on all ten datasets that demonstrate that CMAE (by replacing random masking with MT-CM) alone provides a statically significant improvement in performance compared to ReMasker (t-test p<0.05).

Table 3. Context Contributions. One sided paired T-test between CACTI and CMAE imputation $R ^ { 2 }$ to evaluate the statistical significance of contexts’ contribution and out performance (win rate). 

<table><tr><td rowspan="2">Miss %</td><td colspan="3">AVG.  $R^{2}$  GAIN(P-VALUE $\times 10^{-2}$ )</td><td colspan="3">WIN RATE%</td></tr><tr><td>MCAR</td><td>MAR</td><td>MNAR</td><td>MCAR</td><td>MAR</td><td>MNAR</td></tr><tr><td>10</td><td>0.014(1.96)</td><td>0.023(0.79)</td><td>0.017(0.23)</td><td>80</td><td>89</td><td>80</td></tr><tr><td>30</td><td>0.014(0.35)</td><td>0.007(9.24)</td><td>0.017(0.09)</td><td>80</td><td>70</td><td>100</td></tr><tr><td>50</td><td>0.011(0.65)</td><td>0.010(3.66)</td><td>0.016(0.13)</td><td>90</td><td>70</td><td>100</td></tr><tr><td>70</td><td>0.010(2.01)</td><td>0.012(0.72)</td><td>0.015(0.05)</td><td>89</td><td>90</td><td>100</td></tr></table>

Table 2 also shows that using context awareness in conjunction with MT-CM leads to a nearly 5% improvement. To quantify whether context provides a statistically significant contribution, we extended our analysis to directly contrast CACTI and CMAE (CACTI without context) under all missingness percentages, datasets and missingness settings (Table 3 and Figure A9). CACTI outperforms CMAE (win rate) in a majority of the datasets across all settings, with respect to $R ^ { 2 }$ . We then performed one-sided paired t-tests to demonstrate that context provides a statistically significant improvement $_ { ( \mathrm { p < 0 . 0 5 } ) }$ in all settings except for MAR at 30%. These results confirm that context can improve imputation accuracy though its contribution can vary depending on the dataset and the missingness setting. Overall, while either one of our strategies could provide meaningful improvements in imputation accuracy, the use of empirical missingness patterns through copy masking tends to be more useful than the contextual information.

Lastly, we explored design choices involving the loss function (Table $\mathbf { A l } 6 )$ . Training with the loss over observed values alone $( \mathcal { L } _ { \mathbb { O } } )$ yields poor imputation performance. In contrast, training on reconstruction of masked values $( \mathcal { L } _ { \mathbb { M } } )$ , by forcing the model to learn relationships between the observed and masked features, leads to significantly better performance. As expected, the combined (observed and masked value) reconstruction loss $( \mathcal { L } _ { \mathbb { O } } + \mathcal { L } _ { \mathbb { M } } )$ consistently achieves the best performance, due to the constraint of maintaining a latent space that both preserves the relationship between observed while inferring missing features.

# 4.2. Model Sensitivity Analysis

# 4.2.1. MODEL ARCHITECTURE

First, we investigate CACTI’s sensitivity to three core architectural configuration choices: encoder depth $( N _ { e } )$ , decoder depth $( N _ { d } )$ and overall embedding dimension size (E). The aggregated results of this analysis over four different datasets (see Appendix C.3) under MAR and MNAR, with the masking ratio fixed at 90%, are summarized in Table 4. These results indicate that the encoder and decoder depths have a relatively minor impact (especially in the MNAR setting) although our results tend to slightly favor a deeper encoder $( N _ { e } = 1 0 )$ and a shallower decoder $( N _ { d } = 4 )$ . In contrast, we observe higher sensitivity of our model with respect to the choice of embedding dimension size with highest accuracy attained at $E = 6 4$ . Notably, we see a significant drop-off in performance at very large embedding sizes (near 512) likely due to over-fitting.

# 4.2.2. MT-CM MASKING RATE

We next investigate the impact of the choice of MT-CM masking ratio $( p _ { c m } )$ . This parameter can be interpreted as controlling the strength of the inductive bias during the learning process. A higher $p _ { c m }$ encourages the model to place greater emphasis on the observed missingness patterns in the data, allowing the model to capture and extract additional information. Figure A12 summarizes the results of this analysis over 4 different datasets (see Appendix C.3) under MAR and MNAR. Our experiments indicate that, on average, $p _ { c m } \ge 0 . 9 0$ results in the most accurate results, with slight differences based on the missingness mechanism $( p _ { c m } = 0 . 9 9$ for MAR and $p _ { c m } = 0 . 9 5$ for MNAR). We remark that this is a notable departure from existing random masking approaches (Du et al., 2024) which report the optimal choice of masking rate can differ significantly (> 10%) based on the dataset.

# 4.2.3. CONTEXT EMBEDDING MODEL

Since our ablation analysis indicates that context awareness does provide a meaningful improvement to performance, we would like to understand the sensitivity with respect to the choice of language model used to derive the contextual embeddings. To this end, we assess six open-source base models: BERT-base, BERT-large (Devlin et al., 2019), DeBERTa-v3-base, DeBERTa-v3-large (He et al., 2023), GTE-en-MLM-large (Zhang et al., 2024b) and NV-Embedv2 (Lee et al., 2025). These base models usually have a dimension of 768 for their last layer while the large models have a dimension of 1024 and NV-Embed-v2 has a dimension of 4096. Table 5 summarizes the results of this analysis over four different datasets (see Appendix C.3) under MAR and MNAR. These results indicate that there is marginal sensitivity to the choice of embedding model with GTE-en-MLM-large leading to the highest accuracy while DeBERTav3-large obtains the lowest accuracy. There does not appear to be a clear relation between overall performance and embedding size. This indicates that the semantic context learned by each model is more important that the size of the model. This is also supported by the fact that NV-Embed-v2 (7B parameters) consistently under performs BERT-base (110M parameters). Overall GTE-en-MLM-large or BERTlarge seems to be good default choices for generic English language tabular data. Additionally, the ratio of context dimension to total embedding dimension (CTX proportion $\textstyle { \frac { C } { E } } )$ directly influences the contirbution of context awareness. Table A17 shows that CACTI is fairly insensitive to the choice of CTX proportion with 50% or 25% of the embeddings (E) containing context information as optimal. Finally, context embeddings from domain-specific models like BioClinicalBERT (Alsentzer et al., 2019) may help improve imputation for specialized fields like biomedicine, where features have unique contextual relations (e.g., disease classifications). Prior work by Lehman et al. (2023) shows these models outperform general-purpose models on domain-specific tasks. But we leave this line of inquiry for future work.

Table 4. Model architecture sensitivity. Average performance effect of (a) encoder depth, (b) decoder depth, and (c) embedding size. Metrics represent the average across four datasets at 30% missingness proportion.   
(a) Encoder Depth 

<table><tr><td rowspan="2">DEPTH</td><td colspan="2"> $R^{2}$  (↑)</td><td colspan="2">RMSE (↓)</td></tr><tr><td>MAR</td><td>MNAR</td><td>MAR</td><td>MNAR</td></tr><tr><td>4</td><td>0.46</td><td>0.44</td><td>0.68</td><td>0.69</td></tr><tr><td>6</td><td>0.46</td><td>0.44</td><td>0.68</td><td>0.69</td></tr><tr><td>8</td><td>0.45</td><td>0.44</td><td>0.68</td><td>0.69</td></tr><tr><td>10</td><td>0.47</td><td>0.44</td><td>0.67</td><td>0.69</td></tr><tr><td>12</td><td>0.46</td><td>0.44</td><td>0.68</td><td>0.69</td></tr></table>

(b) Decoder Depth 

<table><tr><td rowspan="2">DEPTH</td><td colspan="2"> $R^{2}$  (↑)</td><td colspan="2">RMSE (↓)</td></tr><tr><td>MAR</td><td>MNAR</td><td>MAR</td><td>MNAR</td></tr><tr><td>4</td><td>0.47</td><td>0.44</td><td>0.67</td><td>0.69</td></tr><tr><td>6</td><td>0.46</td><td>0.44</td><td>0.67</td><td>0.69</td></tr><tr><td>8</td><td>0.47</td><td>0.44</td><td>0.67</td><td>0.69</td></tr><tr><td>10</td><td>0.44</td><td>0.44</td><td>0.69</td><td>0.69</td></tr><tr><td>12</td><td>0.43</td><td>0.43</td><td>0.70</td><td>0.71</td></tr></table>

(c) Embedding Size 

<table><tr><td rowspan="2">SIZE</td><td colspan="2"> $R^{2}$  (↑)</td><td colspan="2">RMSE (↓)</td></tr><tr><td>MAR</td><td>MNAR</td><td>MAR</td><td>MNAR</td></tr><tr><td>32</td><td>0.46</td><td>0.42</td><td>0.67</td><td>0.70</td></tr><tr><td>64</td><td>0.46</td><td>0.44</td><td>0.67</td><td>0.69</td></tr><tr><td>128</td><td>0.40</td><td>0.43</td><td>0.73</td><td>0.70</td></tr><tr><td>256</td><td>0.41</td><td>0.39</td><td>0.72</td><td>0.73</td></tr><tr><td>512</td><td>0.38</td><td>0.35</td><td>0.85</td><td>0.97</td></tr></table>

Table 5. Embedding model sensitivity. Average performance effect of embedding models (30% missingness proportion). 

<table><tr><td rowspan="2">EMBEDDING MODEL</td><td colspan="2"> $R^{2}$  (↑)</td><td colspan="2">RMSE (↓)</td></tr><tr><td>MAR</td><td>MNAR</td><td>MAR</td><td>MNAR</td></tr><tr><td>BERT-BASE</td><td>0.47</td><td>0.45</td><td>0.67</td><td>0.69</td></tr><tr><td>BERT-LARGE</td><td>0.46</td><td>0.45</td><td>0.67</td><td>0.68</td></tr><tr><td>DEBERTA-V3-BASE</td><td>0.47</td><td>0.45</td><td>0.67</td><td>0.69</td></tr><tr><td>DEBERTA-V3-LARGE</td><td>0.45</td><td>0.43</td><td>0.68</td><td>0.70</td></tr><tr><td>GTE-EN-MLM-LARGE</td><td>0.47</td><td>0.45</td><td>0.67</td><td>0.68</td></tr><tr><td>NVEMBED-V2</td><td>0.46</td><td>0.44</td><td>0.68</td><td>0.69</td></tr></table>

# 4.2.4. TRAINING CONVERGENCE

We finally evaluate the training convergence behavior of our model in the letter dataset. The results in Figure 4 indicate that the convergence behavior differ based on the missingness setting. Under the more difficult MNAR imputation setting, increased training epochs results in a consistent increase in imputation accuracy that does not fully saturate even at 1500 epochs. In contrast, under the simpler MAR setting, the model quickly converges to its optimal performance around 300 epochs, with increased training causing overfitting as indicated by a reduction in test set accuracy. Given these results and assuming that we do not know the missingness regime a priori, we recommend users start with

![](images/b3ffdbbfe5409f400e34b434191a9afa5ec9c111c1a1e81abbfb5de72352007b.jpg)

<details>
<summary>line</summary>

| Dataset | Split | test | train |
|---|---|---|---|
| MAR | 100 | 0.75 | 0.80 |
| MAR | 300 | 0.82 | 0.825 |
| MAR | 600 | 0.81 | 0.825 |
| MAR | 900 | 0.805 | 0.825 |
| MAR | 1200 | 0.80 | 0.82 |
| MAR | 1500 | 0.805 | 0.825 |
| MNAR | 100 | 0.75 | 0.80 |
| MNAR | 300 | 0.775 | 0.825 |
| MNAR | 600 | 0.80 | 0.825 |
| MNAR | 900 | 0.805 | 0.825 |
| MNAR | 1200 | 0.81 | 0.825 |
| MNAR | 1500 | 0.81 | 0.825 |
| WD | 100 | 1.05 | 1.05 |
| WD | 300 | 0.95 | 0.95 |
| WD | 600 | 0.94 | 0.94 |
| WD | 900 | 0.93 | 0.93 |
| WD | 1200 | 0.92 | 0.92 |
| WD | 1500 | 0.91 | 0.91 |
| WD | 100 | 2.75 | 2.75 |
| WD | 300 | 2.4 | 2.4 |
| WD | 600 | 2.25 | 2.25 |
| WD | 900 | 2.15 | 2.15 |
| WD | 1200 | 2.1 | 2.1 |
| WD | 1500 | 2.05 | 2.05 |
The chart displays R² and WD values for two datasets (MAR and MNAR) across four epochs (100 to 150). Error bars are present on the data points.
</details>

Figure 4. CACTI Training profile. Evaluated across training epochs under MAR and MNAR on the letter dataset with 30% missingness. Points are mean ± 95% CI.

300-600 epochs and monitor overfitting on validation data.

# 5. Conclusion

This work introduces a conceptual framework for leveraging information about context and the missingness patterns in the data to improve tabular data imputation. We posit that the observed missingness patterns and semantic information associated with the features serve as both crucial and valuable sources of inductive bias. These hypotheses led us to develop CACTI which integrates these dual sources of bias into a transformer-based imputation model. Our extensive benchmarking and ablation analysis demonstrate that information from each dataset’s unique missingness patterns and column context significantly improves imputation accuracy, allowing CACTI to reach state-of-the-art performance. Our MT-CM masking strategy can be used with any masked learning model, while context awareness can be integrated into any deep learning-based imputation framework, demonstrating the broad applicability of our results. These results suggest that identifying additional sources and structures of useful bias is a worthwhile avenue for future tabular imputation research, particularly in fields with smaller datasets with high MNAR missingness such a biomedical data.

# Impact Statement

This paper presents CACTI whose goal is to advance the classical machine learning field of imputation for tabular data. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. This is general approach that is compatible with any generic or field specific tabular dataset. CACTI allows users to more effectively learn an imputation function by leveraging the structure of missingness unique to each dataset and allows for straightforward integration of the unstructured textual information about the features being imputed.

# Acknowledgments

We thank Jonathan Flint for their valuable feedback and discussions throughout this project. S.S. was supported, in part, by NIH grant R35GM153406 and NSF grant CAREER-1943497. A.G. was supported, in part, by NIH grant R01MH130581 and R01MH122569.

# References

Alsentzer, E., Murphy, J. R., Boag, W., Weng, W.-H., Jin, D., Naumann, T., and McDermott, M. B. A. Publicly available clinical bert embeddings, 2019. URL https: //arxiv.org/abs/1904.03323.   
An, U., Pazokitoroudi, A., Alvarez, M., Huang, L., Bacanu, S., Schork, A. J., Kendler, K., Pajukanta, P., Flint, J., Zaitlen, N., and et al. Deep learning-based phenotype imputation on population-scale biobank data increases genetic discoveries. Nature Genetics, 55(12):2269–2276, Nov 2023. doi: 10.1038/s41588-023-01558-w.   
An, U., Lee, S. A., Jeong, M., Gorla, A., Chiang, J. N., and Sankararaman, S. Dk-behrt: Teaching language models international classification of disease (icd) codes using known disease descriptions. In Proceedings of The First AAAI Bridge Program on AI for Medicine and Healthcare, volume 281 of Proceedings of Machine Learning Research, pp. 133–143. PMLR, 25 Feb 2025.   
Arik, S. O. and Pfister, T. Tabnet: Attentive interpretable tabular learning, 2020. URL https://arxiv.org/ abs/1908.07442.   
Dai, Z., Bu, Z., and Long, Q. Multiple imputation via generative adversarial network for high-dimensional blockwise missing value problems, 2021. URL https: //arxiv.org/abs/2112.11507.   
Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. URL https://arxiv. org/abs/1810.04805.

Du, T., Melis, L., and Wang, T. Remasker: Imputing tabular data with masked autoencoding. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=KI9NqjLVDT.

Dufter, P., Schmitt, M., and Schutze, H. Position in-¨ formation in transformers: An overview, 2021. URL https://arxiv.org/abs/2102.11090.

Gardner, J., Perdomo, J. C., and Schmidt, L. Large scale transfer learning for tabular data via language modeling, 2024. URL https://arxiv.org/abs/2406. 12031.

Hastie, T., Mazumder, R., Lee, J., and Zadeh, R. Matrix completion and low-rank svd via fast alternating least squares, 2014. URL https://arxiv.org/abs/ 1410.2596.

Hawthorne, G. and Elliott, P. Imputing cross-sectional missing data: Comparison of common techniques. Australian and New Zealand Journal of Psychiatry, 39(7):583–590, Jul 2005. doi: 10.1080/j.1440-1614.2005.01630.x.

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, R.´ Masked autoencoders are scalable vision learners, 2021. URL https://arxiv.org/abs/2111.06377.

He, P., Gao, J., and Chen, W. Debertav3: Improving deberta using electra-style pre-training with gradientdisentangled embedding sharing, 2023. URL https: //arxiv.org/abs/2111.09543.

Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A., ¨ Korfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter, ¨ F. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, Jan 2025. doi: 10.1038/s41586-024-08328-6.

Huang, X., Khetan, A., Cvitkovic, M., and Karnin, Z. Tabtransformer: Tabular data modeling using contextual embeddings, 2020. URL https://arxiv.org/abs/ 2012.06678.

Ipsen, N. B., Mattei, P.-A., and Frellsen, J. not-miwae: Deep generative modelling with missing not at random data, 2021. URL https://arxiv.org/abs/2006. 12871.

Jackson, J., Mitra, R., Hagenbuch, N., McGough, S., and Harbron, C. A complete characterisation of structured missingness, 2023. URL https://arxiv.org/ abs/2307.02650.

Jarrett, D., Cebere, B. C., Liu, T., Curth, A., and van der Schaar, M. Hyperimpute: Generalized iterative imputation with automatic model selection. In International Conference on Machine Learning, pp. 9916–9937. PMLR, 2022.   
Kelly, M., Longjohn, R., and Nottingham, K. The uci machine learning repository. URL https://archive. ics.uci.edu.   
Krell, M. M., Kosec, M., Perez, S. P., and Fitzgibbon, A. Efficient sequence packing without cross-contamination: Accelerating large language models without impacting performance, 2022. URL https://arxiv.org/ abs/2107.02027.   
Lee, C., Roy, R., Xu, M., Raiman, J., Shoeybi, M., Catanzaro, B., and Ping, W. Nv-embed: Improved techniques for training llms as generalist embedding models, 2025. URL https://arxiv.org/abs/2405.17428.   
Lehman, E., Hernandez, E., Mahajan, D., Wulff, J., Smith, M. J., Ziegler, Z., Nadler, D., Szolovits, P., Johnson, A., and Alsentzer, E. Do we still need clinical language models?, 2023. URL https://arxiv.org/abs/ 2302.08091.   
Lin, X., Xu, C., Yang, M., and Cheng, G. Ctsyn: A foundational model for cross tabular data generation, 2024. URL https://arxiv.org/abs/2406.04619.   
Little, R. J. A. and Rubin, D. B. Statistical analysis with missing data. Wiley, 1987.   
Majmundar, K., Goyal, S., Netrapalli, P., and Jain, P. Met: Masked encoding for tabular data, 2022. URL https: //arxiv.org/abs/2206.08564.   
Mattei, P.-A. and Frellsen, J. Miwae: Deep generative modelling and imputation of incomplete data, 2019. URL https://arxiv.org/abs/1812.02633.   
Mitra, R., McGough, S. F., Chakraborti, T., Holmes, C., Copping, R., Hagenbuch, N., Biedermann, S., Noonan, J., Lehmann, B., Shenvi, A., and et al. Learning from data with structured missingness. Nature Machine Intelligence, 5(1):13–23, Jan 2023. doi: 10.1038/ s42256-022-00596-z.   
Muzellec, B., Josse, J., Boyer, C., and Cuturi, M. Missing data imputation using optimal transport, 2020. URL https://arxiv.org/abs/2002.03860.   
Nazabal, A., Olmos, P. M., Ghahramani, Z., and Valera, I. Handling incomplete heterogeneous data using vaes, 2020. URL https://arxiv.org/abs/1807. 03653.

Richardson, T. W., Wu, W., Lin, L., Xu, B., and Bernal, E. A. Mcflow: Monte carlo flow models for data imputation, 2020. URL https://arxiv.org/abs/2003. 12628.   
Royston, P. and White, I. R. Multiple imputation by chained equations (mice): Implementation in stata. Journal of Statistical Software, 45(4):1–20, 2011. doi: 10.18637/jss. v045.i04. URL https://www.jstatsoft.org/ index.php/jss/article/view/v045i04.   
Rubin, D. B. Inference and missing data. Biometrika, 63(3): 581, Dec 1976. doi: 10.2307/2335739.   
Rubin, D. B. Multiple imputation for nonresponse in surveys. Wiley Series in Probability and Statistics, Jun 1987. doi: 10.1002/9780470316696.   
Schafer, J. L. and Graham, J. W. Missing data: our view of the state of the art. Psychological methods, 7 2:147– 77, 2002. URL https://api.semanticscholar. org/CorpusID:7745507.   
Stekhoven, D. J. and Buhlmann, P. Missforest—non-¨ parametric missing value imputation for mixed-type data. Bioinformatics, 28(1):112–118, Oct 2011. doi: 10.1093/bioinformatics/btr597.   
van Buuren, S. and Groothuis-Oudshoorn, K. mice: Multivariate imputation by chained equations in r. Journal of Statistical Software, 45(3):1–67, 2011. doi: 10.18637/jss. v045.i03. URL https://www.jstatsoft.org/ index.php/jss/article/view/v045i03.   
Yang, Y., Wang, Y., Liu, G., Wu, L., and Liu, Q. Unitabe: A universal pretraining protocol for tabular foundation model in data science, 2024. URL https://arxiv. org/abs/2307.09249.   
Yin, P., Neubig, G., tau Yih, W., and Riedel, S. Tabert: Pretraining for joint understanding of textual and tabular data, 2020. URL https://arxiv.org/abs/2005. 08314.   
Yoon, J., Jordon, J., and van der Schaar, M. GAIN: Missing data imputation using generative adversarial nets. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 5689–5698. PMLR, 10–15 Jul 2018. URL https://proceedings.mlr.press/v80/ yoon18a.html.   
Yoon, J., Zhang, Y., Jordon, J., and van der Schaar, M. Vime: Extending the success of self- and semi-supervised learning to tabular domain. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems,

volume 33, pp. 11033–11043. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper\_files/paper/2020/file/ 7d97667a3e056acab9aaf653807b4a03-Paper. pdf.   
Yoon, S. and Sull, S. Gamin: Generative adversarial multiple imputation network for highly missing data. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8453–8461, 2020. doi: 10.1109/CVPR42600.2020.00848.   
Zhang, H., Fang, L., and Yu, P. S. Unleashing the potential of diffusion models for incomplete data imputation, 2024a. URL https://arxiv.org/abs/ 2405.20690.   
Zhang, X., Zhang, Y., Long, D., Xie, W., Dai, Z., Tang, J., Lin, H., Yang, B., Xie, P., Huang, F., Zhang, M., Li, W., and Zhang, M. mgte: Generalized long-context text representation and reranking models for multilingual text retrieval, 2024b. URL https://arxiv.org/abs/ 2407.19669.   
Zheng, S. and Charoenphakdee, N. Diffusion models for missing value imputation in tabular data, 2023. URL https://arxiv.org/abs/2210.17128.

A. Copy Masking 

<table><tr><td colspan="2">Algorithm 1 Naive copy masking.</td></tr><tr><td colspan="2">1: Input: Observed mask  $\mathbf{M} \in \{0,1\}^{N \times K}$ , masking ratio  $p_{cm} \in (0,1)$ </td></tr><tr><td>2:  $\pi : \{1,\dots,N\} \to \{1,\dots,N\}$ </td><td>{permute indices}</td></tr><tr><td>3:  $\mathbf{P}_{ij} = \begin{cases} 1, & \text{if } i = \pi(j) \\ 0, & \text{otherwise} \end{cases} \in \{0,1\}^{N \times N}$ </td><td></td></tr><tr><td>4:  $\mathbf{M}^{perm} = \mathbf{P}\mathbf{M}$ </td><td>{Apply row-wise permutation}</td></tr><tr><td>5:  $\mathbf{M}^{cm} = \mathbf{M}$ </td><td>{init copy mask}</td></tr><tr><td colspan="2">6: for  $i \leftarrow 1,\dots,N$  do</td></tr><tr><td>7:  $u \sim \mathcal{U}(0,1)$ </td><td>{sample from uniform}</td></tr><tr><td>8:  $ct_{feat} = \mathbf{M}_{i,:}^{cm} \bullet \mathbf{M}_{i,:}^{perm}$ </td><td>{count features left via dot product}</td></tr><tr><td colspan="2">9: if  $u < p_{cm}$  and  $ct_{feat} \geq 1$  then</td></tr><tr><td>10:  $\mathbf{M}_{i,:}^{cm} \leftarrow \mathbf{M}_{i,:}^{perm}$ </td><td>{use copy mask}</td></tr><tr><td colspan="2">11: end if</td></tr><tr><td colspan="2">12: end for</td></tr><tr><td colspan="2">13: Output:  $\mathbf{M}^{cm}$ </td></tr></table>

Copy Masking ablation analysis We compare the performance characteristics of naive copy masking (CM) and MT-CM training strategy with respect to the masking rate $( p _ { c m } )$ . To perform this analysis, we construct a copy masking auto encoder architecture (CMAE) which is our CACTI model without the context aware embeddings and use either the naive CM or MT-CM training.This analysis is performed on bike and obesity datasets and the average results across these two datasets under all three missingness scenarios are reported in Table A6. The first three rows of Table A6 demonstrate a consistent decrease in performance across all three missingness settings as $p _ { c m }$ is increased. The higher mask probability $( p _ { c m } )$ leads to more null tokens in each training batch which reduces training performance because the model must create meaningful latent representation (for the decoder) from positions that contain no information for the encoding layers to work with. As a result, a low $p _ { c m }$ of about 10% producing the best performance. Strikingly, this trend is reversed by the MT-CM strategy (as seen in the last 5 rows of Table A6) where increasing the $p _ { c m }$ results in increased performance with best performance attained at 90% masking. We also see that best performance (w.r.t $R ^ { 2 } )$ under MT-CM is 6.7%, 5.8% and 2.1% higher than naive copy masking under MNAR, MAR and MCAR, respectively.

Table A6. Performance comparison of MT-CM vs naive CM and varying masking rate for CMAE. Metrics represent the average across four datasets. Experiments were performed under three missingness scenarios at 30% missingness. Best in bold and second best underlined. 

<table><tr><td rowspan="2">MASKING TYPE</td><td rowspan="2">MASKING RATE</td><td colspan="3"> $R^{2}$ (↑)</td><td colspan="3">RMSE (↓)</td></tr><tr><td>MCAR</td><td>MAR</td><td>MNAR</td><td>MCAR</td><td>MAR</td><td>MNAR</td></tr><tr><td rowspan="3">NAIVE COPY MASKING</td><td>10</td><td>0.378</td><td>0.396</td><td>0.358</td><td>0.644</td><td>0.620</td><td>0.643</td></tr><tr><td>30</td><td>0.355</td><td>0.397</td><td>0.339</td><td>0.658</td><td>0.626</td><td>0.663</td></tr><tr><td>50</td><td>0.348</td><td>0.384</td><td>0.324</td><td>0.667</td><td>0.638</td><td>0.673</td></tr><tr><td rowspan="5">MEDIAN TRUNCATED COPY MASKING (MT-CM)</td><td>10</td><td>0.366</td><td>0.346</td><td>0.341</td><td>0.651</td><td>0.654</td><td>0.656</td></tr><tr><td>30</td><td>0.378</td><td>0.399</td><td>0.362</td><td>0.643</td><td>0.623</td><td>0.645</td></tr><tr><td>50</td><td>0.375</td><td>0.408</td><td>0.371</td><td>0.647</td><td>0.617</td><td>0.637</td></tr><tr><td>90</td><td>0.386</td><td>0.416</td><td>0.382</td><td>0.638</td><td>0.615</td><td>0.635</td></tr><tr><td>95</td><td>0.386</td><td>0.420</td><td>0.376</td><td>0.641</td><td>0.622</td><td>0.640</td></tr></table>

Algorithm 2 Median Truncated Copy Masking (MT-CM)   
1: Input: Batch of embeddings $E \in R^{N_{B} \times K \times D}$ , observed masks $M \in \{0,1\}^{N_{B} \times K}$ , naive copy masks $M^{cm} \in \{0,1\}^{N_{B} \times K}$ (from Algorithm 1)
{Step 1: Calculate observed feature counts after copy masking}
2: for $n \leftarrow 1, ..., N_{B}$ do
3: $o_{n} \leftarrow \sum_{k=1}^{K} M_{n,k}^{cm}$ 4: end for
5: $o_{B}^{median} \leftarrow median(\{o_{1}, ..., o_{N_{B}}\})$ {Step 2: Find median observed count}
{Step 3: Apply median truncation with permutation}
6: for $n \leftarrow 1, ..., N_{B}$ do
7: $o_{n}^{trunc} \leftarrow \min(o_{n}, o_{B}^{median})$ {Truncate to median at most}
8: $\pi_{n}: \{1, ..., K\} \rightarrow \{1, ..., K\}$ {Random permutation of features}
9: $O_{n} \leftarrow \emptyset, M_{n} \leftarrow \emptyset$ {Initialize observed and masked sets}
10: $E_{n}' \leftarrow []$ {Initialize truncated embeddings}
11: count $\leftarrow 0$ 12: for $k \leftarrow 1, ..., K$ do
13: if $M_{n,\pi_{n}(k)}^{cm} = 1$ and count $< o_{n}^{trunc}$ then
14: $O_{n} \leftarrow O_{n} \cup \{\pi_{n}(k)\}$ {Add to observed set}
15: $E_{n}' \leftarrow [E_{n}' || E_{n,\pi_{n}(k)}]$ {Append embedding}
16: count $\leftarrow count + 1$ 17: else if $M_{n,\pi_{n}(k)} = 1$ then
18: $M_{n} \leftarrow M_{n} \cup \{\pi_{n}(k)\}$ {Add to masked set}
19: end if
20: end for
21: if $o_{n}^{trunc} < o_{B}^{median}$ then
22: $pz = o_{B}^{median} - o_{n}^{trunc}$ {Null token padding size}
23: $NT_{n} \leftarrow 0^{pz \times D}$ 24: $E_{n}' \leftarrow [E_{n}' || NT_{n}]$ {Concat null token padding when needed}
25: end if
26: end for
27: Output: $[E_{1}', ...; E_{N_{B}}']$ , $\{O_{1}, ..., O_{N_{B}}\}$ , $\{M_{1}, ..., M_{N_{B}}\}$

# B. CACTI method extended details

We now provide a detailed description on the transformer-based autonencoding backbone architecture of CACTI.

Encoder : The context-aware embeddings $\mathbf { E } _ { n }$ are used for the subsequent steps in the transformer backbone. For the embedding $\mathbf { E } _ { n }$ of the incomplete data $\tilde { \mathbf { X } } _ { n } ,$ we define a selector function $s e _ { \mathbf { C P - M T } }$ that drops all missing or masked features based on the missingness mask $\mathbf { M } _ { n } ^ { c m }$ , median truncates each batch as previously described and any remaining missing/masked cells are replaced with fixed null padding. This process results in $\mathbf { E } _ { n } ^ { \prime } = s e _ { \mathbf { C P - M T } } ( \mathbf { E } _ { n } )$ where ${ \bf E } _ { n } ^ { \prime } \in  { \cal B }$ $\mathbb { R } ^ { E \times K ^ { \vee } }$ and $K ^ { \prime } ( = o _ { B } ^ { m e d i a n } )$ is the number of features after median truncation. This matrix ${ \bf E } _ { n } ^ { \prime }$ is processed by the encoder that consists of a series of $N _ { e }$ self-attention blocks with residual connections, where the output is ${ \mathbf L } _ { n }$ .

Decoder : We transform the context $\mathbf { C } ^ { c i }$ information embedding into decoder context embeddings using a linear projection $r _ { d } : { \mathcal { C } } ^ { c i } \to { \mathcal { C } } ^ { \prime }$ where $\mathcal { C } ^ { \prime } \subseteq \mathbb { R } ^ { C \times K }$ . The underlying rationale is that the encoder and decoder benefit from different kinds of context information. Next, define selector function $s d _ { \bf M }$ that maps the latent representation L to match the shape and order of the original input features. The missing/masked features are filled with a fixed mask vector which is then passed through a linear projection to become decoder value information, resulting in $\mathbf { V } _ { n } \in \mathbb { R } ^ { U \times K }$ . The decoder context and value information is concated (denoted by ||), with positional encoding, to get the context-aware decoder latent representational ${ \bf Z } _ { n } = [ { \bf V } _ { n } | | { \bf C } ^ { \prime } ] + { \bf P }$ . This latent representation is processed through $N _ { d }$ layers of self-attention with residual connections. The final output is passed through a 2-layer ML $\mathbf { \nabla } \cdot \mathbf { \mathcal { Z } } : \mathcal { Z } : \bar { \mathcal { X } }$ to estimate the uncorrupted version of the incomplete data $\bar { \mathbf { X } } _ { n }$ .

Optimization : The model is trained to minimize the reconstruction loss $\mathcal { L } ( \tilde { \mathbf { X } } _ { n } , \bar { \mathbf { X } } _ { n } )$ between the imputed and the observed data. We optimize our model against a loss function which is a sum of the loss over the observed value $( \mathbb { O } _ { n } )$ and masked (i.e., observed but hidden) values $( \mathbb { M } _ { n } )$ . Note that we perform a min-max scaling of the input data before passing the cell values to the model. This allows us to use a unified MSE loss for all features and constrains the models search space. The model’s internal output $\bar { \mathbf { X } } _ { n }$ is therefor logits. The final loss formulation is:

$$
\mathcal {L} _ {\mathbb {O} _ {n}} = \frac {\sum_ {k = 1} ^ {| \mathbb {O} _ {n} |} (\tilde {x} _ {n k} - \bar {x} _ {n k}) ^ {2}}{| \mathbb {O} _ {n} |}
$$

$$
\mathcal {L} _ {\mathbb {M} _ {n}} = \frac {\sum_ {k = 1} ^ {| \mathbb {M} _ {n} |} (\tilde {x} _ {n k} - \bar {x} _ {n k}) ^ {2}}{| \mathbb {M} _ {n} |} \tag {6}
$$

$$
\mathcal {L} (\tilde {\mathbf {X}} _ {n}, \bar {\mathbf {X}} _ {n}) = \mathcal {L} _ {\mathbb {O} _ {n}} + \mathcal {L} _ {\mathbb {M} _ {n}}
$$

We train our model through stochastic gradient descent using the AdamW optimizer with learning rate (lr) 0.001, default decay settings (0.90, 0.95) and Cosine Annealing with warmup lr scheduler. During inference, the model’s internal output is transformed back to the original space for continuous features by inverting in the min-max scaling.

Algorithm 3 CACTI Training Algorithm   
1: Input: Training dataset $\mathcal{D} = \{(\tilde{\mathbf{X}}_i, \mathbf{M}_i)\}_{i=1}^N$ , Context information $\mathbf{C}^{ci}$ , Masking ratio $p_{cm}$ , Training epochs $Epoch_{max}$ , Batch size $B$ 2: Parameters: Encoder weights $\psi$ , Decoder weights $\theta$ , (encoder, decoder) Value embedding weights $\phi_{e,d}$ , (encoder, decoder) Context embedding weights $\omega_{e,d}$ , Fixed sin-cos positional embeddings $\mathbf{P}$ 3: for epoch $\leftarrow 1, \ldots, Epoch_{max}$ do
4: $\mathbf{M}^{cm} \leftarrow \text{NaiveCopyMask}([M_1; \ldots; M_N], p_{cm})$ {Apply naive copy masking, Algorithm 1}
5: for each batch $\mathcal{B} = \{(\tilde{\mathbf{X}}_n, \mathbf{M}_n, \mathbf{M}_n^{cm})\}_{n=1}^B$ sampled from $\mathcal{D}$ and $\mathbf{M}^{cm}$ do
6: {Process batch of samples}
7: for $n \leftarrow 1, \ldots, B$ do
8: $\mathbf{U}_n \leftarrow l_{\phi_e}(\tilde{\mathbf{X}}_n)$ {Project observed data to value embeddings}
9: $\mathbf{C}_n \leftarrow r_{\omega_e}(\mathbf{C}^{ci})$ {Project context info. to context embeddings}
10: $\mathbf{E}_n \leftarrow [\mathbf{U}_n||\mathbf{C}_n] + \mathbf{P}$ {Concat. value and context embeddings + add positional embeddings}
11: end for
12: $\mathbf{E}', \mathbb{O}, \mathbb{M} \leftarrow \text{MT-CM}([\mathbf{E}_1; \ldots; \mathbf{E}_B], [\mathbf{M}_1; \ldots; \mathbf{M}_B], [\mathbf{M}_1^{cm}; \ldots; \mathbf{M}_B^{cm}])$ {Median truncated copy masking; Algorithm 2}
13: $\mathcal{L}_{batch} \leftarrow 0$ 14: for $n \leftarrow 1, \ldots, B$ do
15: $\mathbf{L}_n \leftarrow \text{Encode}_\psi(\mathbf{E}_n')$ {Apply encoder on remaining context aware embeddings}
16: $\mathbf{V}_n \leftarrow \text{MaskReorder}_{\phi_d}(\mathbf{L}_n, \mathbb{O}_n, \mathbb{M}_n)$ {match original feat. order, set missing feats. to [MASK] and project}
17: $\mathbf{C}' \leftarrow r_{\omega_d}(\mathbf{C}^{ci})$ {Project context info. to decoder context embeddings}
18: $\mathbf{Z}_n \leftarrow [\mathbf{V}_n||\mathbf{C}'] + \mathbf{P}$ {Concat. value and context embeddings and add positional embeddings}
19: $\bar{\mathbf{X}}_n \leftarrow \text{Decode}_\theta(\mathbf{Z}_n)$ {Decode a.k.a do imputation}
20: $\mathcal{L}_n \leftarrow \mathcal{L}_{\mathbb{O}_n}(\tilde{\mathbf{X}}_n, \bar{\mathbf{X}}_n) + \mathcal{L}_{\mathbb{M}_n}(\tilde{\mathbf{X}}_n, \bar{\mathbf{X}}_n)$ {Calc. MSE loss over observed and masked values}
21: $\mathcal{L}_{batch} \leftarrow \mathcal{L}_{batch} + \mathcal{L}_n$ 22: end for
23: $\mathcal{L}_{batch} \leftarrow \mathcal{L}_{batch}/B$ {Average loss over batch}
24: $\psi, \theta, \phi_{e,d}, \omega_{e,d} \leftarrow \text{UpdateWeights}(\nabla\mathcal{L}_{batch})$ {Gradient update}
25: end for
26: end for
27: Output: Trained parameters $\psi, \theta, \phi_{e,d}, \omega_{e,d}$

Algorithm 4 CACTI Inference Algorithm   
1: Input: Observed data $\mathcal{D} = \{(\tilde{\mathbf{X}}_i, \mathbf{M}_i)\}_{i=1}^N$ , Context information $\mathbf{C}^{ci}$ 2: Parameters: (From Algorithm 3) Trained encoder weights $\psi$ , decoder weights $\theta$ , value embedding weights $\phi_{e,d}$ , context embedding weights $\omega_{e,d}$ , positional embeddings $\mathbf{P}$ 3: $\mathbf{C} \leftarrow r_{\omega_e}(\mathbf{C}^{ci})$ 4: $\mathbf{C}' \leftarrow r_{\omega_d}(\mathbf{C}^{ci})$ 5: for $n \leftarrow 1, \ldots, N$ do
6: $\mathbf{U}_n \leftarrow l_{\phi_e}(\tilde{\mathbf{X}}_n)$ 7: $\mathbf{E}_n \leftarrow [\mathbf{U}_n||\mathbf{C}] + \mathbf{P}$ 8: $\mathbf{E}_n' \leftarrow \text{TRUNC}(\mathbf{E}_n, \mathbf{M}_n)$ {Drop all true missing feats.}
9: $\mathbf{L}_n \leftarrow \text{Encode}_\psi(\mathbf{E}_n')$ 10: $\mathbf{V}_n \leftarrow \text{Mask}_{\phi_d}(\mathbf{L}_n)$ {set missing feats. to [MASK] and project}
11: $\mathbf{Z}_n \leftarrow [\mathbf{V}_n||\mathbf{C}'] + \mathbf{P}$ 12: $\bar{\mathbf{X}}_n \leftarrow \text{Decode}_\theta(\mathbf{Z}_n)$ 13: $\hat{\mathbf{X}}_n \leftarrow \tilde{\mathbf{X}}_n \odot \mathbf{M}_n + \bar{\mathbf{X}}_n \odot (1 - \mathbf{M}_n)$ {set missing feats. to imputed values}
14: end for
15: Output: $\hat{\mathbf{X}}_n$

# C. Experimental Details

# C.1. Baseline methods overview

ReMasker (Du et al., 2024) applies the random masking (transformer) autoencoder framework to impute missing values. DiffPuter (Zhang et al., 2024a) is a method that leverages the Expectation-Maximization algorithm along with a diffusion model to iteratively learn the conditional probability of missing data. AutoComplete (An et al., 2023) is a naive copy masking autoencoder based model which learns to reconstruct missing values. Hyperimpute (Jarrett et al., 2022) is an iterative imputation framework that automatically selects and configures (classical machine learning) models for column-wise imputation. Missforest (Stekhoven & Buhlmann ¨ , 2011) iteratively trains random forest models on observed data and applies them to impute missing data. ICE (Royston & White, 2011) is a method which conditionally models and imputes missing data iteratively until convergence. MICE (van Buuren & Groothuis-Oudshoorn, 2011) is variation of ICE which utilizes Bayesian ridge regression. Softimpute (Hastie et al., 2014) uses iterative rank-restricted soft singular value decomposition to complete a matrix with missing values. Sinkhorn (Muzellec et al., 2020) is a generative method utilizing optimal transport distances as a loss criterion to impute missing values. GAIN (Yoon et al., 2018) is a generative-adversarial network with the generator trained to impute missing values conditioned on observed values, and the discriminator trained to identify which values were imputed. MIWAE (Mattei & Frellsen, 2019) is an (generative) approach which applies the importance weighted autoencoder framework and imputes missing data by optimizing the variational lower bound on log likelihood on observed data. notMIWAE (Ipsen et al., 2021) is an extension of MIWAE which incorporates prior information about the type of missingness, allowing modeling of the conditional distribution of the missingness pattern given the data, to try to effectively tackle the MNAR setting (particaully self-masking MNAR). Finally, Mean (Hawthorne & Elliott, 2005) imputes missing values with the column-wise unconditional mean.

Table A7. Default hyperparameter settings used for baseline methods. 

<table><tr><td>MODEL</td><td>HYPERPARAMETERS</td></tr><tr><td>HYPERIMPUTE</td><td>CLASS_THRESHOLD = 2, BASELINE_IMPUTER = 0, OPTIMIZER = &quot;SIMPLE&quot;</td></tr><tr><td>GAIN</td><td>BATCH_SIZE = 256, N_EPOCHS = 1000, HINT_RATE = 0.9, LOSS_ALPHA = 10</td></tr><tr><td>ICE</td><td>MAX_ITER = 500</td></tr><tr><td>MEAN</td><td>NONE</td></tr><tr><td>MICE</td><td>N_IMPUTATIONS = 1, MAX_ITER = 100, TOL = 0.001</td></tr><tr><td>MISSFOREST</td><td>N_ESTIMATORS = 10, MAX_ITER = 500</td></tr><tr><td>MIWAE</td><td>N_EPOCHS = 500, BATCH_SIZE = 256, LATENT_SIZE = 1, N_HIDDEN = 1, K = 20</td></tr><tr><td>SINKHORN</td><td>EPS = 0.01, LR = 1E-3, OPT = TORCH.OPTIM.ADAM, N_EPOCHS = 500, BATCH_SIZE = 256, N_PAIRS = 1, NOISE = 1E-2, SCALING = 0.9</td></tr><tr><td>SOFTIMPUTE</td><td>MAXIT = 1000, CONVERGENCE_THRESHOLD = 1E-5, MAX_RANK = 2, SHRINK_LAMBDA = 0, CV_LEN = 3, RANDOM_STATE = 0</td></tr><tr><td>REMASKER</td><td>MAX_EPOCHS = 300, BATCH_SIZE = 64, MASK_RATIO = 0.5, EMBED_DIM = 32, DEPTH = 6, DECODER_DEPTH = 4, NUM_HEADS = 4, MLP_RATIO = 4, ENCODER_FUNC = &#x27;LINEAR&#x27;, WEIGHT_DECAY = 0.05, BASE_LR = 1E-3, MIN_LR = 1E-5, WARMUP_EPOCHS = 40</td></tr><tr><td>DIFFPUTER</td><td>MAX_ITER = 10, RATIO = 30, HID_DIM = 1024, NUM_TRIALS = 10, NUM_STEPS = 50</td></tr><tr><td>AUTOCOMPLETE</td><td>LR = 0.001, BATCH_SIZE = 1024, EPOCHS = 300, MOMENTUM = 0.9, ENCODING_RATIO = 1, DEPTH = 1, COPYMASK_AMOUNT = 0.5, NUM_TORCH_THREADS = 8, SIMULATE_MISSING = 0.01</td></tr><tr><td>NOTMIWAE</td><td>N_HIDDEN=128, N_SAMPLES=20, BATCH_SIZE=16,EMBEDDING_SIZE=20, MISSING_PROCESS=SELFMASKING_Known</td></tr></table>

# C.2. CACTI hyperparameter configuration

Table A8. Default Parameters for CACTI 

<table><tr><td>MODEL</td><td>PARAMETER</td><td>SETTING</td></tr><tr><td rowspan="9">GLOBAL</td><td>OPTIMIZER</td><td>ADAMW</td></tr><tr><td>INITIAL LEARNING RATE</td><td>1E-3</td></tr><tr><td>LR SCHEDULER</td><td rowspan="2">STEP WISE WARMUP COSINE ANNEALING (0.90, 0.95)</td></tr><tr><td>BETAS (GRAD MOMENTS DECAY)</td></tr><tr><td>WARMUP EPOCHS</td><td>50</td></tr><tr><td>GRADIENT CLIPPING THRESHOLD</td><td>5.0</td></tr><tr><td>TRAINING EPOCHS</td><td>300</td></tr><tr><td>BATCH SIZE</td><td>128</td></tr><tr><td>MASKING RATIO ( $p_{cm}$ )</td><td>0.90</td></tr><tr><td rowspan="3">ENCODER</td><td>DEPTH ( $N_e$ )</td><td>10</td></tr><tr><td>EMBEDDING WIDTH (E)</td><td>64</td></tr><tr><td>NUMBER OF HEADS</td><td>8</td></tr><tr><td rowspan="3">DECODER</td><td>DEPTH ( $N_d$ )</td><td>4</td></tr><tr><td>EMBEDDING WIDTH (E)</td><td>64</td></tr><tr><td>NUMBER OF HEADS</td><td>8</td></tr><tr><td rowspan="3">CONTEXT EMBEDDINGS</td><td>MODEL</td><td>GTEv1.5-EN-MLM-LARGE-8192</td></tr><tr><td>EMBEDDING SIZE ( $dim(\mathbf{C}_k^{ci})$ )</td><td>1024</td></tr><tr><td>CONTEXT EMBEDDING RATIO ( $\frac{C}{E}$ )</td><td>0.25</td></tr></table>

# C.3. Datasets details

To evaluate the performance of CACTI across a diverse set of data types, we chose datasets that contain only continuous features, as well as datasets that contain some combination of categorical, binary, and integer features, labeled as mixed. Also, to demonstrate robustness across datasets of different feature counts and dataset sizes, we benchmark across datasets ranging from 8 features to 57 features, as well as datasets ranging from 2,111 samples to 47,621 samples.

Baseline benchmarking studies are conducted on all 10 datasets which are fully observed. The ablation and sensitivity analysis are conducted on the four following datasets: bike, default, spam and students.

Table A9. Dataset summary 

<table><tr><td>NAME</td><td>FEATURE COUNT</td><td>TRAIN SPLIT SIZE</td><td>TEST SPLIT SIZE</td><td>TOTAL SIZE</td><td>FEATURE TYPE</td><td>FEAT. DESC.</td></tr><tr><td>California HOUSING</td><td>8</td><td>16,512</td><td>4,128</td><td>20,640</td><td>CONTINUOUS ONLY</td><td>YES</td></tr><tr><td>Magic GAMMA TELESCOPE</td><td>10</td><td>15,216</td><td>3,804</td><td>19,020</td><td>CONTINUOUS ONLY</td><td>YES</td></tr><tr><td>Spam BASE</td><td>57</td><td>3,680</td><td>921</td><td>4,601</td><td>CONTINUOUS ONLY</td><td>NO</td></tr><tr><td>Letter RECOGNITION</td><td>16</td><td>16,000</td><td>4,000</td><td>20,000</td><td>CONTINUOUS ONLY</td><td>YES</td></tr><tr><td>ESTIMATION OF Obesity LEVELS</td><td>16</td><td>1,688</td><td>423</td><td>2,111</td><td>MIXED</td><td>YES</td></tr><tr><td>SEOUL Bike SHARING DEMAND</td><td>12</td><td>7,008</td><td>1,752</td><td>8,760</td><td>MIXED</td><td>NO</td></tr><tr><td>Default OF CREDIT CARD CLIENTS</td><td>23</td><td>24,000</td><td>6,000</td><td>30,000</td><td>MIXED</td><td rowspan="2">NO YES (SOMETIMES)</td></tr><tr><td>ADULT Income ONLINE Shoppers</td><td>14</td><td>38,096</td><td>9,525</td><td>47,621</td><td>MIXED</td></tr><tr><td>PURCHASING INTENTION PREDICT Students&#x27;</td><td>17</td><td>9,864</td><td>2,466</td><td>12,330</td><td>MIXED</td><td>NO</td></tr><tr><td>DROPOUT AND ACADEMIC SUCCESS</td><td>36</td><td>3,539</td><td>885</td><td>4,424</td><td>MIXED</td><td>YES</td></tr></table>

# D. Extended Benchmarking Results

This section provides additional results to demonstrate CACTI’s performance against the 13 baseline methods as measured by $R ^ { 2 }$ , RMSE, and WD across 10 datasets, 3 missingness scenarios (MCAR, MAR, and MNAR), and 4 missingness ratios (0.1,0.3,0.5,0.7). Furthermore, we show performance on both the train split and the test split, demonstrating performance in both in sample and out of sample scenarios.

Here we point out that a method that is missing from the $R ^ { 2 }$ plots (not to be confused with $R ^ { 2 } \approx 0 )$ implies lack of convergence due to the loss function taking on NaN values. Notably, at missingness rates ≥ 30% ReMasker, DiffPuter, notMIWAE and AutoComplete show convergence difficulties for one or more datasets using their recommend parameter settings. In particular, ReMasker failed in all MCAR and MNAR datasets at 70% simulated missingness and in shoppers in MCAR and MNAR at 30%. DiffPuter also fails in shoppers at 30% simulated missingness under all 3 missingness settings. notMIWAE fails on income at MCAR 30% and spam in almost all settings except for MCAR 70%. Finally, AutoComplete fails on the income dataset under MNAR and MCAR at 70% simulated missingness. We made sure to re-run these failed runs at least twice to rule out random chance or a hardware issue. We report these failed runs to ensure transparency and did not adjust the recommend/default parameters to try to force these methods to not converge to NaN to ensure a fair comparison with all other methods which successfully ran and converged on all datasets.

Table A10. Average performance comparison with standard errors (in parenthesis) of 15 imputation methods on the train/test splits (separated by |) over 10 datasets at 10% missingness. Metrics (arrows indicate direction of better performance) evaluated under the MAR, MCAR, and MNAR conditions. – indicates method cannot perform out-of-samples imputation. Best in bold and second best underlined. 

<table><tr><td></td><td>MCAR</td><td>MAR</td><td>MINAR</td><td>RMSE(↑)</td><td>MCAR</td><td>MAR</td><td>MINAR</td><td>WID(↑)</td></tr><tr><td rowspan="2">METHOD</td><td colspan="8">R2(↓)</td></tr><tr><td>0.528</td><td>0.538</td><td>0.520</td><td>0.528</td><td>0.533</td><td>10.543</td><td>10.590</td><td>0.548</td></tr><tr><td>CACTI</td><td>0.513</td><td>0.524</td><td>0.495</td><td>0.516</td><td>0.513</td><td>0.525</td><td>0.604</td><td>0.561</td></tr><tr><td>CMAE</td><td>(0.002)</td><td>(0.004)</td><td>(0.011)</td><td>(0.012)</td><td>(0.004)</td><td>(0.005)</td><td>(0.005)</td><td>(0.006)</td></tr><tr><td>SMAKE</td><td>(0.002)</td><td>(0.003)</td><td>(0.011)</td><td>(0.017)</td><td>(0.004)</td><td>(0.004)</td><td>(0.003)</td><td>(0.006)</td></tr><tr><td>SHYPERMPUTE</td><td>(0.004)</td><td>0.475</td><td>0.409</td><td>0.417</td><td>0.428</td><td>0.449</td><td>2.912</td><td>0.637</td></tr><tr><td>DIFPUTER</td><td>(0.004)</td><td>(0.004)</td><td>(0.012)</td><td>(0.016)</td><td>(0.003)</td><td>(0.004)</td><td>(1.814)</td><td>(0.006)</td></tr><tr><td>MISSFORST</td><td>(0.003)</td><td>(0.005)</td><td>(0.010)</td><td>(0.015)</td><td>(0.004)</td><td>(0.004)</td><td>(0.009)</td><td>(0.017)</td></tr><tr><td>NORTMIWAE</td><td>(0.004)</td><td>(0.004)</td><td>(0.013)</td><td>(0.013)</td><td>(0.003)</td><td>(0.003)</td><td>(0.006)</td><td>(0.005)</td></tr><tr><td>ANTOMWAIE</td><td>(0.004)</td><td>(0.004)</td><td>(0.013)</td><td>(0.013)</td><td>(0.003)</td><td>(0.003)</td><td>(0.006)</td><td>(0.011)</td></tr><tr><td>ASTOCOMPLETE</td><td>(0.002)</td><td>(0.003)</td><td>(0.012)</td><td>(0.012)</td><td>(0.002)</td><td>(0.002)</td><td>(0.009)</td><td>(0.013)</td></tr><tr><td>MICICE</td><td>(0.004)</td><td>(0.007)</td><td>(0.013)</td><td>(0.016)</td><td>(0.005)</td><td>(0.006)</td><td>(0.004)</td><td>(0.067)</td></tr><tr><td>AUTOCOMPLETED</td><td>(0.002)</td><td>(0.003)</td><td>(0.012)</td><td>(0.012)</td><td>(0.002)</td><td>(0.002)</td><td>(0.009)</td><td>(0.013)</td></tr><tr><td>GAIN</td><td>(0.003)</td><td>(0.004)</td><td>(0.010)</td><td>(0.014)</td><td>(0.003)</td><td>(0.002)</td><td>(0.011)</td><td>(0.008)</td></tr><tr><td>SOFIMPUTE</td><td>(0.001)</td><td>(0.003)</td><td>(0.007)</td><td>(0.007)</td><td>(0.002)</td><td>(0.002)</td><td>(0.009)</td><td>(0.006)</td></tr><tr><td>MIWAE</td><td>(0.003)</td><td>(0.002)</td><td>(0.010)</td><td>(0.010)</td><td>(0.004)</td><td>(0.002)</td><td>(0.009)</td><td>(0.022)</td></tr><tr><td>MEAN</td><td>(0.000)</td><td>(0.001)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.001)</td><td>(0.010)</td><td>(0.016)</td></tr></table>

Table A11. Average performance comparison with standard errors (in parenthesis) of 15 imputation methods on the train/test splits (separated by |) over 10 datasets at 30% missingness. Metrics (arrows indicate direction of better performance) evaluated under the MAR, MCAR, and MNAR conditions. – indicates method cannot perform out-of-samples imputation. Best in bold and second best underlined. 

<table><tr><td rowspan="3"></td><td colspan="3"> $R^2$ (↑)</td><td colspan="101"> $RMSE$ (↑)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">MCAR</td><td rowspan="2">MAR</td><td rowspan="2" colspan="100">MINAR</td><td>METHOD</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">CACTI</td><td>0.456</td><td>0.461</td><td>0.468</td><td>0.470</td><td>0.461</td><td>0.456</td><td>0.662</td><td>0.641</td><td>0.670</td><td>0.686</td><td>0.680</td><td>0.666</td><td>4.346</td><td>4.459</td><td>1.870</td><td>1.942</td><td>4.449</td><td>4.568</td><td colspan="71">WID (↑)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(0.002)</td><td>(0.003)</td><td>(0.010)</td><td>(0.011)</td><td>(0.003)</td><td>(0.004)</td><td>(0.008)</td><td>(0.005)</td><td>(0.018)</td><td>(0.016)</td><td>(0.004)</td><td>(0.006)</td><td>(0.008)</td><td>(0.007)</td><td>(0.017)</td><td>(0.014)</td><td>(0.005)</td><td>(0.008)</td><td>(0.007)</td><td>(0.017)</td><td>(0.014)</td><td>(0.005)</td><td>(0.006)</td><td>(0.007)</td><td>(0.018)</td><td>(0.027)</td><td>(0.033)</td><td>(0.030)</td><td>(0.033)</td><td>(0.012)</td><td>(0.023)</td><td>(0.032)</td><td>(0.043)</td><td>(0.042)</td><td>(0.030)</td><td>(0.031)</td><td>(0.031)</td><td>(0.012)</td><td>(0.026)</td><td>(0.048)</td><td>(0.035)</td><td>(0.031)</td><td>(0.031)</td><td colspan="55">WID (↑)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(0.002)</td><td>(0.002)</td><td>(0.010)</td><td>(0.010)</td><td>(0.003)</td><td>(0.004)</td><td>(0.008)</td><td>(0.006)</td><td>(0.017)</td><td>(0.014)</td><td>(0.005)</td><td>(0.008)</td><td>(0.007)</td><td>(0.017)</td><td>(0.012)</td><td>(0.023)</td><td>(0.032)</td><td>(0.043)</td><td>(0.042)</td><td>(0.032)</td><td>(0.043)</td><td>(0.042)</td><td>(0.032)</td><td>(0.043)</td><td>(0.042)</td><td>(0.032)</td><td>(0.043)</td><td>(0.042)</td><td>(0.032)</td><td>(0.043)</td><td>(0.042)</td><td>(1.2248)</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>12.257</td><td>4.688</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4,298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td>4.298</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(0.002)</td><td>(0.002)</td><td>(0.009)</td><td>(0.010)</td><td>(0.002)</td><td>(0.003)</td><td>(0.004)</td><td>(0.006)</td><td>(0.007)</td><td>(0.002)</td><td>(0.004)</td><td>(0.006)</td><td>(0.007)</td><td>(0.006)</td><td>(0.009)</td><td>(0.006)</td><td>(0.014)</td><td>(0.010)</td><td>(0.005)</td><td>(0.008)</td><td>(0.008)</td><td>(0.006)</td><td>(0.007)</td><td>(0.006)</td><td>(0.009)</td><td>(0.006)</td><td>(0.013)</td><td>(0.013)</td><td>(0.008)</td><td>(0.008)</td><td>(0.007)</td><td>(0.150)</td><td>(0.026)</td><td>(0.100)</td><td>(0.038)</td><td>(0.078)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(1.150)</td><td>(0.026)</td><td>(0.100)</td><td>(0.038)</td><td>(0.078)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.056)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(0.057)</td><td>(8.374)</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.374</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.137</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.235</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8 235</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345 8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.345</td><td>8.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table A12. Average performance comparison with standard errors (in parenthesis) of 15 imputation methods on the train/test splits (separated by |) over 10 datasets at 50% missingness. Metrics (arrows indicate direction of better performance) evaluated under the MAR, MCAR, and MNAR conditions. – indicates method cannot perform out-of-samples imputation. Best in bold and second best underlined. 

<table><tr><td rowspan="2"></td><td colspan="3"> $R^2$ (↑)</td><td colspan="101">RMSE(↑)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MCAR</td><td>MAR</td><td>MNAR</td><td>MCAR</td><td>MAR</td><td>MNAR</td><td>MNR</td><td>WD(↑)</td><td>Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0.354</td><td>0.358</td><td>0.431</td><td>0.435</td><td>0.354</td><td>0.355</td><td>0.730</td><td>0.723</td><td>0.683</td><td>0.694</td><td>0.771</td><td>0.755</td><td>5.128</td><td>5.152</td><td>1.835</td><td>1.905</td><td>5.209</td><td>5.289</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0.341</td><td>0.344</td><td>0.420</td><td>0.423</td><td>0.338</td><td>0.339</td><td>0.747</td><td>0.743</td><td>0.692</td><td>0.702</td><td>0.782</td><td>0.766</td><td>5.331</td><td>5.358</td><td>1.899</td><td>1.980</td><td>5.359</td><td>5.436</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.341</td><td>0.344</td><td>0.420</td><td>0.423</td><td>0.338</td><td>0.339</td><td>0.747</td><td>0.743</td><td>0.692</td><td>0.702</td><td>0.782</td><td>0.766</td><td>5.331</td><td>5.358</td><td>1.899</td><td>1.980</td><td>5.539</td><td>5.436</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.341</td><td>(0.002)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>( 0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.008)</td><td>(0.011)</td><td>(0.011)</td><td>(0.023)</td><td>(0.007)</td><td>(0.142)</td><td>(0.183)</td><td>(0.111)</td><td>(0.110)</td><td>(0.153)</td><td>(0.136)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>( 0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.000)</td><td>(0.011)</td><td>(0.011)</td><td>(0.123)</td><td>(0.107)</td><td>(1.264)</td><td>(1.2638)</td><td>6.340</td><td>6.393</td><td>12.573</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td><td>12.583</td></tr><tr><td>0.354</td><td>0.358</td><td>0.431</td><td>0.435</td><td>0.354</td><td>0.355</td><td>0.730</td><td>0.723</td><td>0.683</td><td>0.694</td><td>0.771</td><td>0.755</td><td>5.128</td><td>5.152</td><td>1.835</td><td>1.905</td><td>S.209</td><td>S.289</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table A13. Average performance comparison with standard errors (in parenthesis) of 15 imputation methods on the train/test splits (separated by |) over 10 datasets at 70% missingness. Metrics (arrows indicate direction of better performance) evaluated under the MAR, MCAR, and MNAR conditions. – indicates method cannot perform out-of-samples imputation and NA indicates method failed. Best in bold and second best underlined. 

<table><tr><td rowspan="2">METHOD</td><td colspan="3"> $R^2$ (↓)</td><td colspan="3">RMSE (↑)</td><td rowspan="2">WD (↑)</td><td></td></tr><tr><td>MCAR</td><td>MAR</td><td>MNAR</td><td>MCAR</td><td>MAR</td><td>MNAR</td><td></td></tr><tr><td rowspan="2">CACTI</td><td>0.227|0.228</td><td>0.386|0.383</td><td>0.227|0.231</td><td>0.827|0.826</td><td>0.746|0.721</td><td>0.839|0.825</td><td>6.215|6.262</td><td>1.740|1.765</td></tr><tr><td>(0.001)|(0.001)</td><td>(0.008)|(0.009)</td><td>(0.002)|(0.002)</td><td>(0.002)|(0.009)</td><td>(0.008)|(0.010)</td><td>(0.004)|(0.008)</td><td>(0.002)|(0.002)</td><td></td></tr><tr><td rowspan="6">CMAE</td><td>(0.001)|(0.002)</td><td>(0.008)|(0.009)</td><td>(0.002)|(0.002)</td><td>(0.005)|(0.008)</td><td>(0.008)|(0.010)</td><td>(0.004)|(0.008)</td><td>(0.002)|(0.002)</td><td></td></tr><tr><td>(0.222|0.224)</td><td>0.374|0.374</td><td>0.214|0.217</td><td>0.836|0.836</td><td>0.755|0.724</td><td>0.853|0.841</td><td>6.440|6.519</td><td></td></tr><tr><td>(0.001)|(0.002)</td><td>(0.008)|(0.009)</td><td>(0.002)|(0.002)</td><td>(0.002)|(0.009)</td><td>(0.008)|(0.010)</td><td>(0.004)|(0.008)</td><td>(0.002)|(0.002)</td><td></td></tr><tr><td>(0.222|0.224)</td><td>0.374|0.374</td><td>0.214|0.217</td><td>0.836|0.836</td><td></td><td></td><td></td><td></td></tr><tr><td>(0.001)|(0.002)</td><td>(0.008)|(0.009)</td><td>(0.002)|(0.002)</td><td>(0.002)|(0.009)</td><td>(0.008)|(0.010)</td><td>(0.004)|(0.008)</td><td>(0.003)|(0.002)</td><td></td></tr><tr><td>(0.222|0.224)</td><td>0.374|0.374</td><td>0.214|0.217</td><td>0.836|0.836</td><td>0.755|0.724</td><td>0.853|0.841</td><td>6.440|6.519</td><td></td></tr></table>

![](images/ea31d36622e3acb4902e80a35de106913432567afe1259f1bd6f1a2106a86412.jpg)  
(a) MCAR at 10% missingness

![](images/62082d682e9711fc692fb6e4e3d8bd653e9ebe6259928c828c96390c259f6012.jpg)  
(b) MAR at 10% missingness

![](images/784c3fb233ace7c8e1067c3340b9f34af145a2ceef1964f834f936af7a5d9eef.jpg)

<details>
<summary>bar</summary>

| Category | Test | Train | R2 | RMSE | WD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| bike | 0.75 | 0.65 | 0.35 | 0.55 | 2 |
| california | 0.65 | 0.55 | 0.25 | 0.45 | 3 |
| default | 0.60 | 0.50 | 0.30 | 0.40 | 4 |
| income | 0.40 | 0.35 | 0.20 | 0.30 | 5 |
| letter | 0.80 | 0.75 | 0.40 | 0.60 | 10 |
| magic | 0.65 | 0.60 | 0.35 | 0.55 | 12 |
| obesity | 0.35 | 0.30 | 0.25 | 0.35 | 1 |
| shoppers | 0.45 | 0.40 | 0.30 | 0.40 | 11 |
| spam | 0.30 | 0.25 | 0.20 | 0.25 | 12 |
| students | 0.60 | 0.55 | 0.35 | 0.45 | 13 |
| test | 0.70 | 0.65 | 0.40 | 0.50 | 14 |
| train | 0.65 | 0.60 | 0.35 | 0.45 | 15 |
| test | 0.75 | 0.70 | 0.45 | 0.55 | 16 |
| train | 0.70 | 0.65 | 0.40 | 0.50 | 17 |
| test | 0.85 | 0.80 | 0.55 | 0.65 | 18 |
| train | 0.80 | 0.75 | 0.50 | 0.60 | 19 |
| test | 0.90 | 0.85 | 0.60 | 0.70 | 20 |
| train | 0.85 | 0.80 | 0.65 | 0.75 | 21 |
| test | 1.25 | 1.20 | 1.15 | 1.35 | 22 |
| train | 1.20 | 1.15 | 1.10 | 1.30 | 23 |
| test | 1.45 | 1.40 | 1.35 | 1.55 | 24 |
| train | 1.40 | 1.35 | 1.30 | 1.50 | 25 |
| test | 1.65 | 1.60 | 1.55 | 1.75 | 26 |
| train | 1.60 | 1.55 | 1.50 | 1.70 | 27 |
| test | 1.85 | 1.80 | 1.75 | 1.95 | 28 |
| train | 1.80 | 1.75 | 1.70 | 2.15 | 29 |
| test | 2.25 | 2.20 | 1.85 | 2.35 | 30 |
| train | 2.20 | 2.15 | 1.90 | 2.55 | 31 |
| test | 2.65 | 2.60 | 2.15 | 2.85 | 32 |
| train | 2.60 | 2.55 | 2.20 | 3.15 | 33 |
| test | 3.15 | 3.10 | 2.45 | 3.45 | 34 |
| train | 3.10 | 3.05 | 2.40 | 3.75 | 35 |
| test | 3.65 | 3.60 | 2.75 | 4.15 | 36 |
| train | 3.60 | 3.55 | 2.70 | 4.45 | 37 |
| test | 4.25 | 4.20 | 3.15 | 4.85 | 38 |
| train | 4.20 | 4.15 | 3.10 | 5.25 | 39 |
| test | 4.85 | 4.80 | 3.65 | 5.75 | 40 |
| train | 4.80 | 4.75 | 3.60 | 6.15 | 41 |
| test | 5.45 | 5.40 | 3.95 | 6.65 | 42 |
| train | 5.40 | 5.35 | 4.25 | 7.15 | 43 |
| test | 6.15 | 6.10 | 4.65 | 7.65 | 44 |
| train | 6.10 | 6.05 | 4.60 | 8.15 | 45 |
| test | -     | -   | -   | -   | -   |
| train - weight loss (R²) - CACTI (test)    + CMAE (test)        A: -          A: -          B: -          C: -            D: -            E: -            F: -            G: -            H: -             I: -              J: -              K: -              L: -              M: -              N: -              O: -              P: -              Q: -              R: -              S: -              T: -              U: -              V: -              W: -              X: -              Y: -              Z: -              AA: -              AB: -              AC: -              AD: -              AE: -              AF: -              AG: -              AH: -              AI: -              AJ: -              AK: -              AL: -              AM: -              AN: -              AO: -              AP: -              AQ: -              AR: -              AS: -              AT: -              AU: -              AV: -              AW: -             AXA: -             AXB: -             AXC: -             AXD: -             AXE: -             AXF: -             AXG: -             AXH: -             AXI: -             AXJ: -             AXK: -             AXL: -             AXM: -             AXN: -             AXO: -             AXP: -             AXQ: -             AXQI: -             AXQII: -            AXQIII: -            AXQIV:             AXQV         /                 X-axis label       /      Y-axis label       /        Legend:
- Method
- RMSE (RMSE)        Method Color      CACITI (RMSE)      CMAE (RMSE)      reMasker (RMSE)      HyperImpute (RMSE)      MissForest (RMSE)      notMIWAE (RMSE)      Sinkhorn (RMSE)      ICE (RMSE)           AutoComplete (RMSE)      MICE (RMSE)         GAIN (RMSE)          GAIN (RMSE)          SoftImpute (RMSE)      MIWAE (RMSE)            GAIN (RMSE)           MIWAE (RMSE)            GAIN (RMSE)           MIWAE (RMSE)            GAIN (RMSE)           MIWAE (RMSE)            GAIN (RMSE)           MIWAE (RMSE)            GAIN (RMSE)           MIWAE (RMSE)            GAIN (RMSE)           MIWAE (RMSE)            GAIN (RMSE)           MIWAE (RMSE)            GAIY(GAIN)(GAIN)(SoftImpute)(SoftImpute)(MIWAE)(MIWAE)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)(GAIN)            GAIN(GAIN)(GAIN)(SoftImpute)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)            GAIN(GAIN)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)(GAIN)(SoftImpute)(MIWAE)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIN(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(GAIN)            GAIY(Gain)|
</details>

(c) MNAR at 10% missingness   
Figure A5. Performance comparison of CACTI against 13 baseline methods. Experiments were performed on 10 datasets, under MCAR, MAR, and MNAR, at 10% missingness. Results shown as mean ± 95% CI.

![](images/263e92142d7e7250c7264774e32494f082abe7d7bc25f19e4e402d10b1dbbe06.jpg)  
(a) MCAR at 30% missingness

![](images/9e967114def2fb7d0dd8ff8d5f8037c0535da1ab373a95f34f1c7b55e61f2723.jpg)  
(b) MAR at 30% missingness

![](images/2737bcf859fa1bd74cafe0892f323dcb80f40e4a8287711f6c5a892c6fdeb1fc.jpg)  
(c) MNAR at 30% missingness

Figure A6. Performance comparison of CACTI against 13 baseline methods. Experiments performed on 10 datasets, under MCAR, MAR, and MNAR, at 30% missingness. Results shown as mean ± 95% CI.   
![](images/b330b9d0636ce3786619f75946fbc004dae142f57a4a4ac500eafb00d8b3c3bf.jpg)

<details>
<summary>bar</summary>

| Dataset    | Test | Train | R2   | RMSE | WD  |
| ---------- | ---- | ----- | ---- | ---- | --- |
| bike       | 0.4  | 0.4   | 0.3  | 0.7  | 3   |
| california | 0.3  | 0.3   | 0.2  | 0.8  | 5   |
| default    | 0.5  | 0.5   | 0.4  | 1.6  | 9   |
| income     | 0.2  | 0.2   | 0.1  | 0.9  | 6   |
| letter     | 0.6  | 0.6   | 0.5  | 1.2  | 12  |
| magic      | 0.5  | 0.5   | 0.4  | 1.4  | 10  |
| obesity    | 0.2  | 0.2   | 0.1  | 0.6  | 5   |
| shoppers   | 0.3  | 0.3   | 0.2  | 1.0  | 8   |
| spam       | 0.2  | 0.2   | 0.1  | 0.5  | 6   |
| students   | 0.4  | 0.4   | 0.3  | 0.9  | 16  |
</details>

(a) MCAR at 50% missingness

![](images/74f6f4b222880f8eec2d5843bbc80cbc06b16a32b04131fedab0e2bf7ccf5234.jpg)  
(b) MAR at 50% missingness

![](images/1aa892647c9063000e9188ce8ca94003f66a9b38fa163a508b1ed91b7a2d754f.jpg)  
(c) MNAR at 50% missingness   
Figure A7. Performance comparison of CACTI against 13 baseline methods. Experiments were performed on 10 datasets under MCAR, MAR, and MNAR at 50% missingness. Results shown as mean ± 95% CI.

![](images/2aa9bb35b1d333a4786889a738f3aa4b50c80e645a00d5c58d07b9f0d2cb9b1b.jpg)  
(a) MCAR at 70% missingness

![](images/176b468ed3254fcd254573bd8d6ecdfaea993deef164b653dfede1f90cf494d3.jpg)

<details>
<summary>bar</summary>

| Category    | Test | Train |
| ----------- | ---- | ----- |
| bike        | 0.4  | 0.3   |
| california  | 0.35 | 0.25  |
| default     | 0.5  | 0.45  |
| income      | 0.1  | 0.05  |
| letter      | 0.7  | 0.6   |
| magic       | 0.65 | 0.55  |
| obesity     | 0.25 | 0.15  |
| shoppers    | 0.3  | 0.2   |
| spam        | 0.15 | 0.1   |
| students    | 0.45 | 0.35  |
| test        | 0.6  | 0.5   |
| train       | 0.7  | 0.6   |
| test        | 0.8  | 0.7   |
| train       | 0.9  | 0.8   |
| test        | 1.0  | 0.9   |
| train       | 1.1  | 1.0   |
| test        | 1.2  | 1.1   |
| train       | 1.3  | 1.2   |
| test        | 1.4  | 1.3   |
| train       | 1.5  | 1.4   |
| test        | 1.6  | 1.5   |
| train       | 1.7  | 1.6   |
| test        | 1.8  | 1.7   |
| train       | 1.9  | 1.8   |
| test        | 2.0  | 1.9   |
| train       | 2.1  | 2.0   |
| test        | 2.2  | 2.1   |
| train       | 2.3  | 2.2   |
| test        | 2.4  | 2.3   |
| train       | 2.5  | 2.4   |
| test        | 2.6  | 2.5   |
| train       | 2.7  | 2.6   |
| test        | 2.8  | 2.7   |
| train       | 2.9  | 2.8   |
| test        | 3.0  | 2.9   |
| train       | 3.1  | 3.0   |
| test        | 3.2  | 3.1   |
| train       | 3.3  | 3.2   |
| test        | 3.4  | 3.3   |
| train       | 3.5  | 3.4   |
| test        | 3.6  | 3.5   |
| train       | 3.7  | 3.6   |
| test        | 3.8  | 3.7   |
| train       | 3.9  | 3.8   |
| test        | 4.0  | 3.9   |
| train       | 4.1  | 4.0   |
| test        | 4.2  | 4.1   |
| train       | 4.3  | 4.2   |
| test        | 4.4  | 4.3   |
| train       | 4.5  | 4.4   |
| test        | 4.6  | 4.5   |
| train       | 4.7  | 4.6   |
| test        | 4.8  | 4.7   |
| train       | 4.9  | 4.8   |
| test        | 5.0  | 4.9   |
| train       | 5.1  | 5.0   |
| test        | 5.2  | 5.1   |
| train       | 5.3  | 5.2   |
| test        | 5.4  | 5.3   |
| train       | 5.5  | 5.4   |
| test        | 5.6  | 5.5   |
| train       | 5.7  | 5.6   |
| test        | 5.8  | 5.7   |
| train       | 5.9  | 5.8   |
| test        | 6.0  | 5.9   |
| train       | 6.1  | 6.0   |
| test        | 6.2  | 6.1   |
| train       | 6.3  | 6.2   |
| test        | 6.4  | 6.3   |
| train       | 6.5  | 6.4   |
| test        | 6.6  | 6.5   |
| train       | 6.7  | 6.6   |
| test        | 6.8  | 6.7   |
| train       | 6.9  | 6.8   |
| test        | 7.0  | 6.9   |
| train       | 7.1  | 7.0   |
| test        | 7.2  | 7.1   |
| train       | 7.3  | 7.2   |
| test        | 7.4  | 7.3   |
| train       | 7.5  | 7.4   |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -      |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -:     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -     |
| test        | -    | -     |
| train       | -    | -
     |

!
</details>

(b) MAR at 70% missingness

![](images/495f473b8dde24817a9f211e42175583883e2035780b595fc292c8f716e47c9b.jpg)  
(c) MNAR at 70% missingness   
Figure A8. Performance comparison of CACTI against 13 baseline methods. Experiments were performed on 10 datasets under MCAR, MAR, and MNAR at 70% missingness. Results shown as mean ± 95% CI.

Table A14. Comparison of runtime and memory statistics of CACTI with ReMasker on 10 datasets. Experiments were performed under MAR scenario at 30% missingness. The runtime is measured for the training and inference (on train/test split) stages in seconds (s) and peak GPU memory consumed in gigabytes (GB). 

<table><tr><td>METHOD</td><td>DATA</td><td>PER EPOCH (s)</td><td>INFER TRAIN SPLIT (s)</td><td>INFER TEST SPLIT (s)</td><td>PEAK GPU MEM (GB)</td></tr><tr><td rowspan="10">CACTI</td><td>OBESITY</td><td>0.42</td><td>6.15</td><td>1.52</td><td>0.16</td></tr><tr><td>STUDENTS</td><td>0.64</td><td>15.53</td><td>3.84</td><td>0.18</td></tr><tr><td>SPAM</td><td>0.71</td><td>16.36</td><td>4.11</td><td>0.26</td></tr><tr><td>BIKE</td><td>1.24</td><td>25.82</td><td>6.38</td><td>0.16</td></tr><tr><td>SHOPPERS</td><td>1.68</td><td>34.85</td><td>8.67</td><td>0.16</td></tr><tr><td>MAGIC</td><td>2.39</td><td>55.37</td><td>13.70</td><td>0.16</td></tr><tr><td>LETTER</td><td>2.91</td><td>82.06</td><td>21.01</td><td>0.16</td></tr><tr><td>CALIFORNIA</td><td>2.95</td><td>59.93</td><td>14.92</td><td>0.08</td></tr><tr><td>DEFAULT</td><td>4.61</td><td>101.91</td><td>24.84</td><td>0.16</td></tr><tr><td>INCOME</td><td>5.76</td><td>165.59</td><td>42.17</td><td>0.16</td></tr><tr><td rowspan="10">REMASKER</td><td>OBESITY</td><td>0.28</td><td>4.28</td><td>1.07</td><td>0.14</td></tr><tr><td>STUDENTS</td><td>0.49</td><td>9.01</td><td>2.24</td><td>0.14</td></tr><tr><td>SPAM</td><td>0.58</td><td>9.30</td><td>2.34</td><td>0.15</td></tr><tr><td>BIKE</td><td>1.06</td><td>17.74</td><td>4.41</td><td>0.14</td></tr><tr><td>SHOPPERS</td><td>1.55</td><td>25.46</td><td>6.30</td><td>0.14</td></tr><tr><td>MAGIC</td><td>2.09</td><td>38.23</td><td>9.53</td><td>0.14</td></tr><tr><td>LETTER</td><td>2.10</td><td>40.67</td><td>10.09</td><td>0.15</td></tr><tr><td>CALIFORNIA</td><td>2.37</td><td>41.89</td><td>10.44</td><td>0.04</td></tr><tr><td>DEFAULT</td><td>3.18</td><td>61.21</td><td>15.18</td><td>0.15</td></tr><tr><td>INCOME</td><td>4.80</td><td>95.45</td><td>23.94</td><td>0.15</td></tr></table>

# E. Extended Ablation Analysis Results

In this section, we present additional ablation analysis results. First, Table A15 summarizes the results of a 3-way comparison (over all 10 datasets) between CACTI, CMAE (CACTI without context) and ReMasker (the strongest MAE with random masking) to quantify the magnitude and the statistical significances of the improvements driven by MT-CM, context awareness and the combination of the two. A one-sided paired t-test is used to evaluate the statistical significance of improvement of the target method over the baseline. Next, Figure A9 shows the per-dataset difference in $R ^ { 2 }$ performance between CACTI and CMAE across all datasets, missingness situations and simulated missingness percentages. We also perform additional ablation comparing the effects of calculating the loss over observed only values, masked only values and a combination of the two in Table A16.

These results are followed by the complete (per-dataset) results of all our ablation analysis, demonstrating the effects of (a) MT-CM, RM, and/or CTX and (b) loss function on model performance. For this experiment we use 4 UCI datasets (bike, default, spam, and students), each under three missingness scenarios (MCAR, MAR, and MNAR), with a simulated missingness ratio of 0.3. We measure performance using $R ^ { 2 }$ , RMSE, and WD, on both the train and test split.

Table A15. Paired t-test to evaluate statistical significance of gain in performance between CACTI, CMAE (CACTI w/o context) and ReMasker. 

<table><tr><td>MISSINGNESS</td><td>TARGET METHOD</td><td>BASELINE METHOD</td><td>AVG.  $R^{2}$  GAIN</td><td>P-VALUE</td></tr><tr><td rowspan="3">ALL</td><td>CACTI</td><td>REMASKER</td><td>0.034</td><td>4.4E-7</td></tr><tr><td>CACTI</td><td>CMAE</td><td>0.013</td><td>1.1E-5</td></tr><tr><td>CMAE</td><td>REMASKER</td><td>0.021</td><td>4.2E-5</td></tr><tr><td rowspan="3">MCAR</td><td>CACTI</td><td>REMASKER</td><td>0.023</td><td>5.5E-4</td></tr><tr><td>CACTI</td><td>CMAE</td><td>0.014</td><td>3.E-3</td></tr><tr><td>CMAE</td><td>REMASKER</td><td>0.017</td><td>8.9E-3</td></tr><tr><td rowspan="3">MAR</td><td>CACTI</td><td>REMASKER</td><td>0.025</td><td>2.3E-2</td></tr><tr><td>CACTI</td><td>CMAE</td><td>0.007</td><td>9.4E-2</td></tr><tr><td>CMAE</td><td>REMASKER</td><td>0.018</td><td>4.8E-2</td></tr><tr><td rowspan="3">MNAR</td><td>CACTI</td><td>REMASKER</td><td>0.054</td><td>1.1E-4</td></tr><tr><td>CACTI</td><td>CMAE</td><td>0.017</td><td>8.7E-4</td></tr><tr><td>CMAE</td><td>REMASKER</td><td>0.037</td><td>4.6E-4</td></tr></table>

Table A16. Loss ablations. Effect of the loss function on accuracy. Metrics represent the average across four datasets (30% missingness). 

<table><tr><td rowspan="2">LOSS TYPE</td><td colspan="2"> $R^{2}$  (↑)</td><td colspan="2">RMSE (↓)</td></tr><tr><td>MAR</td><td>MNAR</td><td>MAR</td><td>MNAR</td></tr><tr><td> $\mathcal{L}_{\mathbb{O}} + \mathcal{L}_{\mathbb{M}}$ </td><td>0.46</td><td>0.46</td><td>0.68</td><td>0.67</td></tr><tr><td> $\mathcal{L}_{\mathbb{M}}$ </td><td>0.41</td><td>0.43</td><td>0.71</td><td>0.70</td></tr><tr><td> $\mathcal{L}_{\mathbb{O}}$ </td><td>0.03</td><td>0.04</td><td>2.67</td><td>2.93</td></tr></table>

![](images/f4e32e92182a2e267af8ca7c6c26c488de6cf33c0c92600ef15f051eff4e5a1e.jpg)

<details>
<summary>bar</summary>

| Missingness | Dataset | R² Change (CACTI vs CMAE) |
| :--- | :--- | :--- |
| 10% | bike | 0.01 |
| 10% | california | 0.07 |
| 10% | default | 0.014 |
| 10% | income | -0.002 |
| 10% | letter | 0.004 |
| 10% | magic | 0.017 |
| 10% | obesity | 0.028 |
| 10% | shoppers | 0.045 |
| 10% | spam | 0.023 |
| 10% | students | 0.023 |
| 30% | bike | 0.023 |
| 30% | california | -0.006 |
| 30% | default | 0.002 |
| 30% | income | 0.001 |
| 30% | letter | 0.014 |
| 30% | magic | -0.006 |
| 30% | obesity | -0.002 |
| 30% | shoppers | -0.002 |
| 30% | spam | 0.005 |
| 30% | students | 0.041 |
| 50% | bike | 0.017 |
| 50% | california | 0.015 |
| 50% | default | -0.002 |
| 50% | income | 0.016 |
| 50% | letter | 0.017 |
| 50% | magic | -0.002 |
| 50% | obesity | -0.017 |
| 50% | shoppers | -0.017 |
| 50% | spam | 0.029 |
| 50% | students | 0.025 |
| 70% | missingness: MAR (Missingness: 10%) | 0.01 |
| missingness: MAR (Missingness: 30%) | missingness: MAR (Missingness: 50%) | 0.017 |
| missingness: MAR (Missingness: 70%) | missingness: MAR (Missingness: 50%) | 0.029 |
| missingness: MAR (Missingness: 70%) | missingness: MAR (Missingness: 70%) | 0.029 |
| missingness: MCAR (Missingness: 10%) | missingness: MCAR (Missingness: 30%) | -0.01 |
| missingness: MCAR (Missingness: 50%) | missingness: MCAR (Missingness: 70%) | -0.01 |
| missingness: MCAR (Missingness: 70%) | missingness: MCAR (Missingness: 50%) | -0.01 |
| missingness: MCAR (Missingness: 70%) | missingness: MCAR (Missingness: 50%) | -0.01 |
| missingness: MNAR (Missingness: 10%) | missingness: MNAR (Missingness: 30%) | -0.01 |
| missingness: MNAR (Missingness: 50%) | missingness: MNAR (Missingness: 70%) | -0.01 |
| missingness: MNAR (Missingness: 70%) | missingness: MNAR (Missingness: 50%) | -0.01 |
| missingness: MNAR (Missingness: 70%) | missingness: MNAR (Missingness: 50%) | -0.01 |
| missingness: MIR (Missingness: 10%) | missingness: MIR (Missingness: 30%) | -0.01 |
| missingness: MIR (Missingness: 50%) | missingness: MIR (Missingness: 70%) | -0.01 |
| missingness: MIR (Missingness: 70%) | missingness: MIR (Missingness: 50%) | -0.01 |
| missingness: MIR (Missingness: 70%) | missingness: MIR (Missingness: 50%) | -0.01 |
| missingness: MIR (Missingness: 70%) | missingness: MIR (Missingness: 50%) | --0.28 |
| missingness: MIR (Missingness: MIR) | missingness: MIR (Missingness: MIR) | --0.18 |
| missingness: MIR (Missingness: MIR) | missingness: MIR (Missingness: MIR) | --1.2 |
| missingness: MIR (Missingness: MIR) | missingness: MIR (Missingness: MIR) | --1.2 |
| missingness: MIR (Missingness: MIR) | missingness: MIR (Missingness: MIR) | --1.2 |
| missingness: MIR (Missingness: MIR) | missingness: MIR (Missingness: MIR) | --1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, <p> = p > p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = p> = pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=pc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc= mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc=mc- mR (<p> = pc) <p> = pc/pc
</details>

Figure A9. Difference in $R ^ { 2 }$ between CACTI and CMAE across all 10 datasets, missingness conditions and simulated missingness percentages. $R ^ { 2 }$ change $> 0$ indicates CACTI is better than CMAE under the respective setting.

![](images/b2bf9a1022d725112ec9092fa4976f1028618202969eddbcfdce557b2ac65efb.jpg)  
Figure A10. Experiments performed across four datasets split into train/test, under MCAR, MAR, and MNAR, at 30% missingness. Metrics $( R ^ { 2 }$ , RMSE, WD) are reported as mean ± 95% CI. Ablations demonstrate how model performance is affected by MT-CM, RM, and/or CTX.

![](images/6ec6d44eba93f68d2b670fee5ca04bf857677645020013b16d6e2f1c38007aa0.jpg)  
Figure A11. Experiments performed across four datasets split into train/test, under MCAR, MAR, and MNAR, at 30% missingness. Metrics $( R ^ { 2 }$ , RMSE, WD) are reported as mean ± 95% CI. Ablations demonstrate how model performance is affected by the loss function.

# F. Extended sensitivity analysis results

In this section, we present the present the average sensitivity analysis results for context embedding proportion (Table A17) and masking rate (Figure A12). All sensitivity analysis experiments are performed on 4 UCI datasets (bike, default, spams, students), under 2 missingness scenarios (MAR and MNAR), and 0.3 missingness ratio, split into train and test splits. Performance is measured using R2, RMSE, and WD.

These results are followed by the complete (per-dataset) results of all our sensitivity analysis. Figure A13, Figure A14, Figure A15, Figure A16, Figure A17, and Figure A18 show the effects of independently varying MT-CM rate, encoder depth, decoder depth, embedding size, context embedding model, and context embedding proportion respectively.

Table A17. Context embedding sensitivity. Average performance effect of context (CTX) proportions (30% missing).   
![](images/080061594046fda02a28f2dfbd2f12a539b610bf35c7fae55ceecdd521f8464e.jpg)  
Figure A12. Masking rate sensitivity. Average performance metrics over a range of MT-CM masking rate choices. Evaluated under MAR and MNAR with at 30% missingness.

![](images/1c5cbd64614d180471832d7f13f511622adc481c59c9b2aa24552553b2ce6328.jpg)  
Figure A13. Experiments performed across four datasets split into train/test, under MAR and MNAR, at 30% missingness. Metrics $( R ^ { 2 } .$ , RMSE, WD) are reported as mean ± 95% CI and show model sensitivity to MT-CM.

![](images/f9db615a1ad7c88adb80c35565f8f7aa91da277073d608f6095521e49c052708.jpg)  
Figure A14. Experiments performed across four datasets split into train/test, under MAR and MNAR, at 30% missingness. Metrics $( R ^ { 2 } .$ , RMSE, WD) are reported as mean ± 95% CI and show model sensitivity to encoder depth.

![](images/0c34a89e47bb1b68fcd65859f0d674be179013807286f8192343ee09e0e44e98.jpg)  
Figure A15. Experiments performed across four datasets split into train/test, under MAR and MNAR, at 30% missingness. Metrics $( R ^ { 2 } .$ , RMSE, WD) are reported as mean ± 95% CI and show model sensitivity to decoder depth.

![](images/a8ad78cee89abfccc0072e29fa4d7e241b4672200a0222b67a2d81a37edf3094.jpg)

<details>
<summary>bar</summary>

| Split | Dataset | Test R2 | Train R2 | Test RMSE | Train RMSE | Test WD | Train WD |
|---|---|---|---|---|---|---|---|
| test | bike | 0.55 | 0.53 | 0.65 | 0.70 | 1.0 | 1.0 |
| test | default | 0.60 | 0.62 | 0.70 | 0.75 | 1.5 | 28 |
| test | spam | 0.20 | 0.22 | 0.90 | 0.95 | 14 | 19 |
| test | students | 0.45 | 0.43 | 0.60 | 0.65 | 8 | 9 |
| train | MAR | 0.48 | 0.45 | 0.90 | 0.95 | 10 | 11 |
| train | MNAR | 0.45 | 0.43 | 0.85 | 0.90 | 12 | 13 |
| train | MAR | 0.45 | 0.43 | 0.85 | 0.90 | 12 | 13 |
| train | MNAR | 0.45 | 0.43 | 0.85 | 0.90 | 12 | 13 |
| test | 32 | 0.48 | 0.47 | 0.65 | 0.70 | 1.0 | 1.0 |
| test | 64 | 0.48 | 0.47 | 0.65 | 0.70 | 1.0 | 1.0 |
| test | 128 | 0.48 | 0.47 | 0.65 | 0.70 | 1.0 | 1.0 |
| test | 256 | 0.48 | 0.47 | 0.65 | 0.70 | 1.5 | 28 |
| test | 512 | 0.45 | 0.43 | 0.65 | 0.70 | 1.5 | 19 |
| train | MAR | - | - | - | - | - | - |
| train | MNAR | - | - | - | - | - | - |
| train | MAR | - | - | - | - | - | - |
| train | MNAR | - | - | - | - | - | - |
| train | MAR | - | - | - | - | - | - |
| train | MNAR | - | - | - | - | - | - |
| test (R2) vs WD (RMSE) vs WD (WD) = -1: The chart displays R² values for each dataset split by split between test and train sets, with error bars indicating variability. The x-axis represents the number of data points (32 to 512), and the y-axis represents the corresponding R² values for both splits within each data point group. Error bars are shown as vertical error bars.
</details>

Figure A16. Experiments performed across four datasets split into train/test, under MAR and MNAR, at 30% missingness. Metrics $( R ^ { 2 } .$ , RMSE, WD) are reported as mean ± 95% CI and show model sensitivity to embedding size.

![](images/99bd2192b3701151c5c9d0908a207ace496bca80485b1e9cbe3ffaac1e49adbd.jpg)  
Figure A17. Experiments performed across four datasets split into train/test, under MAR and MNAR, at 30% missingness. Metrics $( R ^ { 2 } .$ , RMSE, WD) are reported as mean ± 95% CI and show model sensitivity to context embedding model.

![](images/799f5d34879015ea815202cc96143101302415f9b7df67e00df654f72b83b437.jpg)  
Split test train

Figure A18. Experiments performed across four datasets split into train/test, under MAR and MNAR, at 30% missingness. Metrics $( R ^ { 2 } .$ , RMSE, WD) are reported as mean ± 95% CI and show model sensitivity to context embedding proportion.