# MULTIMODAL ALIGNED SEMANTIC KNOWLEDGE FOR UNPAIRED IMAGE-TEXT MATCHING

Laiguo Yin1, Yixin Zhang2, Yuqing Sun1∗, Lizhen Cui1∗

1Shandong University 2Nanyang Technological University lgyin@mail.sdu.edu.cn, zhangyixin9610@gmail.com, {sun yuqing,clz}@sdu.edu.cn

# ABSTRACT

While existing approaches address unpaired image-text matching by constructing cross-modal aligned knowledge, they often fail to identify semantically corresponding visual representations for Out-of-Distribution (OOD) words. Moreover, the distributional variance of visual representations associated with different words varies significantly, which negatively impacts matching accuracy. To address these issues, we propose a novel method namely Multimodal Aligned Semantic Knowledge (MASK), which leverages word embeddings as bridges to associate words with their corresponding prototypes, thereby enabling semantic knowledge alignment between the image and text modalities. For OOD words, the representative prototypes are constructed by leveraging the semantic relationships encoded in word embeddings. Beyond that, we introduce a prototype consistency contrastive learning loss to structurally regularize the feature space, effectively mitigating the adverse effects of variance. Experimental results on the Flickr30K and MSCOCO datasets demonstrate that MASK achieves superior performance in unpaired matching.

# 1 INTRODUCTION

Image–text matching has become an essential technique for various applications, such as visual question answering (Ozdemir & Akag ¨ und ¨ uz, 2024; Lerner et al., 2024), image captioning (Fu et al., ¨ 2024; Wang et al., 2024a), cross-modal retrieval (Wang et al., 2024b; Li et al., 2024b) and so forth. Due to the heterogeneous representations and asymmetry of information between images and texts, accurately learning cross-modal semantic correspondences remains a challenging problem. Although training on large-scale paired image-text data has substantially improved matching accuracy, collecting and annotating such data at scale is often impractical in real-world scenarios.

![](images/7b67fdb66dc2aaae1d41dc92e98c520226d6583107107b0df28363af15c37350.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Object detection"] --> B["Raw region representations"]
    B --> C["Visual prototype space"]
    C --> D["Words"]
    D --> E["Embedding"]
    D --> F["Proportion"]
    D --> G["Alignment"]
    D --> H["Peak words"]
    I["Mouse"] --> J["Dog"]
    K["Mouse"] --> L["cat (OOD)"]
    M["Mouse"] --> N["women"]
    O["Mouse"] --> P["Tokenize"]
    Q["A young couple with a floating object"] --> R["A young couple with a floating object"]
    S["A young couple with a floating object with a floating object with a floating object with a floating object with a floating object with a floating object with a floating object with a floating object"]
    T["A young couple with a floating object with a floating object with a floating object with a floating object with a floating object with a floating object"]
    U["A young couple with a floating object with a floating object with a floating object with a floating object with a floating object"]
    V["A young couple with a floating object with a floating object with a floating object with a floating object with a floating object"]
    W["A young couple with a floating object with a floating object with a floating object with a floating object"]
    X["A young couple with a floating object with a floating object with a floating object with a floating object"]
    Y["A young couple with a floating object with a floating object with a floating object with a floating object"]
    Z["A young couple with a floating object with a floating object with a floating object with a floating object"]
    AA["A young couple with a floating object with a floating object with a floating object with a floating object"]
    AB["A young couple with a floating object with a floating object with a floating object with a floating object"]
    AC["A young couple with a floating object with a floating object with a floating object with a floating object"]
    AD["A young couple with a floating object with a floating object with a floating object with a floating object"]
    AE["A young couple with a floating object with a floating object with a floating object"]
    AF["A young couple with a floating object with a floating object with a floating object"]
    AG["A young couple with a floating object with a floating object with a floating object"]
    AH["A young couple with a floating object with a floating object"]
    AI["A young couple with a floating object with a floating object"]
    AJ["A young couple with a floating object with a floating object"]
    AK["A young couple with a floating object with a floating object"]
    AL["A young couple with a floating object with a floating object"]
    AM["A young couple with a floating object with a floating object"]
    AN["A young couple with a floating object"]
    AO["A young couple with a floating object"]
    AP["A young couple with a floating object"]
    AQ["A young couple with a floating object"]
    AR["A young couple with a floating object"]
    AS["A young couple with a floating object"]
    AT["A young couple with a floating object"]
    AU["A young couple with a floating object"]
    AV["A young couple with a floating object"]
    AW["A young couple with a floating object"]
    AX["A young couple with a floating object"]
    AY["A young couple with a floating object"]
```
</details>

(a) Existing unpaired image-text matching

![](images/0b26bbdf1532ea2500342996bed93413ec0f72dae6aeea2b9ecbba8761b6edd4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph_VisualPrototypeSpace["Visual prototype space"]
        A["Feature regularization"] --> B["Region representations"]
        C["Feature regularization"] --> D["..."]
        E["Feature regularization"] --> F["..."]
        B --> G["Embedding"]
        D --> G
        F --> G
        G --> H["Cross-Modal Semantic Alignment"]
        H --> I["Word embedding space"]
        I --> J["dog"]
        I --> K["cat (OOD)"]
        I --> L["women"]
        I --> M["Embedding peak"]
    end

    subgraph_WordEmbedingSpace["Word embedding space"]
        N["v = f⁻¹(w)"] --> O["Cross-Modal Semantic Alignment"]
        P["w = f(v)"] --> O
        O --> Q["Sense similarity"]
    end

    style VisualPrototypeSpace fill:#f9f,stroke:#333
    style WordEmbedingSpace fill:#bbf,stroke:#333
```
</details>

(b) Multimodal aligned semantic knowledge   
Figure 1: Comparison between existing matching paradigms and our proposed unpaired framework.

To reduce the dependence of image-text matching on paired data, the unpaired image-text matching paradigm (Huang et al., 2022) was first proposed, in which domain-specific paired images and texts are assumed to be unavailable during model training. Inspired by the fact that the human brain can correlate arbitrary images and texts well while does not need to learn from large-scale paired images and texts, the unpaired image-text matching is implemented by modeling human brain-like knowledge, which is multimodal aligned and used to associate visual and linguistic information.

Note that unimodal visual or linguistic knowledge has been widely used for vision and language understanding task (Chen & Zhao, 2023). There are also some works (Li et al., 2025; Gao et al., 2025) directly combining these two types of knowledge together, but the resulting knowledge is not multimodal aligned. Another alternative is Multimodal Aligned Conceptual Knowledge (Huang et al., 2024a), which establishes correspondences between prototypical region representations and words, as shown in Figure 1 (a). However, these knowledge-based methods still face the following issues: 1) OOD words have not been thoroughly investigated. Existing knowledge-based methods fail to leverage the underlying semantic structure to transfer the visual prototypes of known words to OOD words; 2) The influence of distributional variance has been largely overlooked. The region representations corresponding to different words exhibit substantial appearance variations. Consequently, certain instances that deviate substantially from the distributional mean may be prone to misclassification into other words; 3) The raw region representation is insufficient in effectively capturing the semantic relationships between words. The raw region representation is predominantly influenced by the co-occurrence relationships among regions. However, there is no inherent relationship between semantic relevance and co-occurrence patterns. For instance, while ‘human’ and ‘hat’ often co-occur in visual contexts, ‘human’ and ‘gentleman’ may exhibit a higher semantic similarity.

To address these issues, we propose a new method namely Multimodal Aligned Semantic Knowledge for unpaired image-text matching, which establishes semantic alignment between prototypical region representations and word embeddings, as shown in Figure 1 (b). We summarize our key contributions as follows:

• We propose a novel cross-modal semantic alignment method, MASK, which constructs representative prototypes for OOD words by exploiting the intrinsic relationships among word embeddings, thereby enhancing the model’s generalization ability in unpaired imagetext matching.   
• We introduce a prototype consistency contrastive learning loss to structurally regularize the feature space, which explicitly encourages region representations associated with the same word to align closely with their prototype, thereby mitigating the adverse impact of distributional variance.   
• We incorporate external knowledge from pre-trained word vectors as auxiliary supervision signals, which establishes a relation-preserving equivariant mapping between region representations and word embeddings, enabling the region representations to effectively capture semantic relationships among words.

# 2 RELATED WORK

# 2.1 MODEL-BASED MATCHING

Extensive model-based matching works have been made on measuring the semantic correlation between vision and language. To our knowledge, Socher et al. (Socher et al., 2013) might propose the first framework of Visual-Semantic Embedding (VSE) to correlate images and their class labels in a two-stream manner. Lee et al. (Lee et al., 2018) propose a Stacked Cross Attention Network(SCAN) to discover all latent alignments by using regions of the image and words in a sentence as context. The SCAN has been extensively studied from various aspects such as memory modeling (Huang et al., 2021), context modeling (Zhang et al., 2020) and graph structure (Liu et al., 2020). Later, by using millions or billions of paired images and texts for supervised model learning, many models (Li et al., 2020; Pan et al., 2023; Wu et al., 2024; Li et al., 2024a; Pham et al., 2024; Ge et al., 2024) based on multimodal versions of Transformer have been proposed and have achieved remarkable results. However, while these existing methods achieve relatively strong performance, they rely heavily on extensive paired image-text datasets for supervised training, which significantly restricts their applicability.

# 2.2 KNOWLEDGE-BASED MATCHING

Knowledge-based matching has been explored in a few vision and language understanding tasks. Feng et al. (Feng et al., 2019) use a visual concept detector to encourage generated captions to be semantically consistent with visual concepts. This work focuses more on the visual concepts while paying less attention to visual relations. Gu et al. (Gu et al., 2019) propose to align two scene graphs of images and texts, by mapping one to the other one, and vice versa. The performance of this work depends more on the accuracy of predicted scene graphs that contain both visual concepts and relations. Huang et al. (Huang et al., 2022) introduce Multimodal Aligned Conceptual Knowledge (MACK) for unpaired image-text matching, effectively addressing challenges associated with diverse appearances. Subsequently, Huang et al. (Huang et al., 2024b) further extend MACK to broaden its applicability. However, these approaches still face certain limitations, particularly in their inability to leverage the underlying semantic structure to transfer the prototype representations of known words to OOD words.

# 3 METHOD

This section illustrates the pipeline of obtaining the proposed MASK for unpaired image-text matching, as shown in Figure 2. For each region, the image embedding branch is utilized to extract region representations characterized by high cohesion and low coupling. For each word, the text embedding branch is employed to generate its corresponding word embedding. Consequently, we obtain a knowledge set in which each word is aligned with its corresponding prototypical region representation. This knowledge serves as a bridge in the knowledge-based image-text matching module,51119 enabling the association between domain-specific images and texts and thereby supporting unpaired matching. It is worth noting that the knowledge can be fine-tuned to better adapt to specific domain (Appendix H). However, the fine-tuning step is optional according to whether unpaired data in certain domain are given or not.

![](images/4e9a5fc78ceb9d11221ea6388608311cdbc4c5ca649df08366e83c89ce041a1a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Region"] --> B["Pretrained Faster-RCNN"]
    B --> C["Image Embedding Branch"]
    C --> D["Text Embedding Branch"]
    D --> E["Text Embedding Branch"]
    E --> F["Text Embedding Branch"]
    F --> G["Text Embedding Branch"]
    G --> H["Text Embedding Branch"]
    H --> I["Text Embedding Branch"]
    I --> J["Text Embedding Branch"]
    J --> K["Text Embedding Branch"]
    K --> L["Text Embedding Branch"]
    L --> M["Text Embedding Branch"]
    M --> N["Text Embedding Branch"]
    N --> O["Text Embedding Branch"]
    O --> P["Text Embedding Branch"]
    P --> Q["Text Embedding Branch"]
    Q --> R["Text Embedding Branch"]
    R --> S["Text Embedding Branch"]
    S --> T["Text Embedding Branch"]
    T --> U["Text Embedding Branch"]
    U --> V["Text Embedding Branch"]
    V --> W["Text Embedding Branch"]
    W --> X["Text Embedding Branch"]
    X --> Y["Text Embedding Branch"]
    Y --> Z["Text Embedding Branch"]
    Z --> AA["Text Embedding Branch"]
    AA --> AB["Text Embedding Branch"]
    AB --> AC["Text Embedding Branch"]
    AC --> AD["Text Embedding Branch"]
    AD --> AE["Text Embedding Branch"]
    AE --> AF["Text Embedding Branch"]
    AF --> AG["Text Embedding Branch"]
    AG --> AH["Text Embedding Branch"]
    AH --> AI["Text Embedding Branch"]
    AI --> AJ["Text Embedding Branch"]
    AJ --> AK["Text Embedding Branch"]
    AK --> AL["Text Embedding Branch"]
    AL --> AM["Text Embedding Branch"]
    AM --> AN["Text Embedding Branch"]
    AN --> AO["Text Embedding Branch"]
    AO --> AP["Text Embedding Branch"]
    AP --> AQ["Text Embedding Branch"]
    AQ --> AR["Text Embedding Branch"]
    AR --> AS["Text Embedding Branch"]
    AS --> AT["Text Embedding Branch"]
    AT --> AU["Text Embedding Branch"]
    AU --> AV["Text Embedding Branch"]
    AV --> AW["Text Embedding Branch"]
    AW --> AX["Text Embedding Branch"]
    AX --> AY["Text Embedding Branch"]
    AY --> AZ["Text Embedding Branch"]
    AZ --> BA["Text Embedding Branch"]
    BA --> BB["Text Embedding Branch"]
    BB --> BC["Text Embedding Branch"]
    BC --> BD["Text Embedding Branch"]
    BD --> BE["Text Embedding Branch"]
    BE --> BF["Text Embedding Branch"]
    BF --> BG["Text Embedding Branch"]
    BG --> BH["Text Embedding Branch"]
    BH --> BI["Text Embedding Branch"]
    BI --> BJ["Text Embedding Branch"]
    BJ --> BK["Text Embedding Branch"]
    BK --> BL["Text Embedding Branch"]
    BL --> BM["Text Embedding Branch"]
    BM --> BN["Text Embedding Branch"]
    BN --> BO["Text Embedding Branch"]
    BO --> BP["Text Embedding Branch"]
    BP --> BQ["Text Embedding Branch"]
    BQ --> BR["Text Embedding Branch"]
    BR --> BS["Text Embedding Branch"]
    BS --> BT["Text Embedding Branch"]
    BT --> BU["Text Embedding Branch"]
    BU --> BV["Text Embedding Branch"]
    BV --> BW["Text Embedding Branch"]
    BW --> BX["Text Embedding Branch"]
    BX --> BY["Text Embedding Branch"]
    BY --> BZ["Text Embedding Branch"]
    BZ --> CA["Text Embedding Branch"]
    CA --> CB["Text Embedding Branch"]
    CB --> CC["Text Embedding Branch"]
    CC --> CD["Text Embedding Branch"]
    CD --> CE["Text Embedding Branch"]
    CE --> CF["Text Embedding Branch"]
    CF --> CG["Text Embedding Branch"]
    CG --> CH["Text Embedding Branch"]
    CH --> CI["Text Embedding Branch"]
    CI --> CJ["Text Embedding Branch"]
    CJ --> CK["Text Embedding Branch"]
    CK --> CL["Text Embedding Branch"]
    CL --> CM["Text Embedding Branch"]
    CM --> CN["Text Embedding Branch"]
    CN --> CO["Text Embedding Branch"]
    CO --> CP["Text Embedding Branch"]
    CP --> CQ["Text Embedding Branch"]
    CQ --> CR["Text Embedding Branch"]
    CR --> CS["Text Embedding Branch"]
    CS --> CT["Text Embedding Branch"]
    CT --> CU["Text Embedding Branch"]
    CU --> CV["Text Embedding Branch"]
    CV --> CW["Text Embedding Branch"]
    CW --> CX["Text Embedding Branch"]
    CX --> CY["Text Embedding Branch"]
    CY --> CZ["Text Embedding Branch"]
    CZ --> DA["Text Embedding Branch"]
    DA --> DB["Text Embedding Branch"]
    DB --> DC["Text Embedding Branch"]
    DC --> DD["Text Embedding Branch"]
    DD --> DJ["Text Embedding Branch"]
    DJ --> DK["Text Embedding Branch"]
    DK --> DL["Text Embedding Branch"]
    DL --> DV["Text Embedding Branch"]
    DV --> DW["Text Embedding Branch"]
    DW --> DX["Text Embedding Branch"]
    DX --> DY["Text Embedding Branch"]
    DY --> DY2["Text Embedding Branch"]

    subgraph Knowledge-based Image-text Matching
        E1["The woman read a book while her dog and cat rested beside her"]
        E2["Tokenization"]
        E3[" dog cat woman peak ..."]
        E4[" Pretrained Faster-RCNN"]
        E5[" w_q cat (OOD) women peak ..."]
        E6[" Pretrained Faster-RCNN"]
        E7[" w_out (v_out) v_q ..."]
        E8[" Pretrain Faster-RCNN"]
        E9["&quot; R "]
        E10["&quot; h "]
        E11["&quot; h "]
        E12["&quot; h "]
        E13["&quot; h "]
        E14["&quot; h "]
        E15["&quot; h "]
        E16["&quot; h "]
        E17["&quot; h "]
        E18["&quot; h "]
        E19["&quot; h "]
        E20["&quot; h "]
        E21["&quot; h "]
        E22["&quot; h "]
        E23["&quot; h "]
        E24["&quot; h "]
        E25["&quot; h "]
        E26["&quot; h "]
        E27["&quot; h "]
        E28["&quot; h "]
        E29["&quot; h "]
        E30["&quot; h "]
        E31["&quot; h "]
        E32["&quot; h "]
        E33["&quot; h "]
        E34["&quot; h "]
        E35["&quot; h "]
        E36["&quot; h "]
        E37["&quot; h "]
        E38["&quot; h "]
        E39["&quot; h "]
        E40["&quot; h "]
        E41["&quot; h "]
        E42["&quot; h "]
        E43["&quot; h "]
        E44["&quot; h "]
        E45["&quot; h "]
        E46["&quot; h "]
        E47["&quot; h "]
        E48["&quot; h "]
        E49["&quot; h "]
        E50["&quot; h "]
        E51["&quot; h "]
        E52["&quot; h "]
        E53["&quot; h "]
        E54["&quot; h "]
        E55["&quot; h "]
        E56["&quot; h "]
        E57["&quot; h "]
        E58["&quot; h "]
        E59["&quot; h "]
        E60["&quot; h "]
        E61["&quot; h "]
        E62["&quot; h "]
        E63["&quot; h "]
        E64["&quot; h "]
        E65["&quot; h "]
        E66["&quot; h "]
        E67["&quot; h "]
        E68["&quot; h "]
        E69["&quot; h "]
        E70["&quot; h "]
        E71["&quot; h "]
        E72["&quot; h "]
        E73["&quot; h "]
        E74["&quot; h "]
        E75["&quot; h "]
        E76["&quot; h "]
        E77["&quot; h "]
        E78["&quot; h "]
        E79["&quot; h "]
        E80["&quot; h "]
        E81["&quot; h "]
        E82["&quot; h "]
        E83["&quot; h "]
        E84["&quot; h "]
        E85["&quot; h "]
        E86["&quot; h "]
        E87["&quot; h "]
        E88["&quot; h "]
        E89["&quot; h "]
        E90["&quot; h "]
        E91["&quot; h "]
        E92["&quot; h "]
        E93["&quot; h "]
        E94["&quot; h "]
        E95["&quot; h "]
        E96["&quot; h "]
        E97["&quot; h "]
        E98["&quot; h "]
        E99[" h |
    end

    subgraph Multimodal Aligned Semantic Knowledge
            direction TB
            direction LR
            direction CS
            direction DV
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVout
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVOUT
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVINT
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVIN
            direction DVINA
            direction DVINA
            direction DVINA
            direction DVINA
            direction DVINA
            direction DVINA
            direction DVINA
            direction DVINA
            direction DVINA
                    global similarity s

Legend:
- Region: "Information retention loss Lir"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Lm"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Ll"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Lk"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Ls"
- Region: "Information retention loss Lls"
- Region: "Information retention loss Lls"
- Region: "Information retention loss Lls"
- Region: "Information retention loss Lls"
- Region: "Information retention loss Lls"
- Region: "Information retention loss Lls"
- Region: "Information retention loss Lls"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Region: "Information retention loss Ln"
- Regions represent data from the image.
 regions represent text representation data from the image.
 regions represent spatial representation data from the image.
 regions represent region representation data from the image.
 regions represent spatial representation data from the image.
 regions represent spatial representation data from the image.
 regions represent spatial representation data from the image.
 regions represent spatial representation data from the image.
 regions represent spatial representation data from the image.
 regions represent spatial representation data from the image.
 regions represent spatial representation data from the image.
 regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represents spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent spatial representation data from the image.
regions represent geographic distribution of the image-
text<|ref_end|><|rotate_up|>
```
</details>

Figure 2: The Multimodal Aligned Semantic Knowledge (MASK) for unpaired image-text matching. The top figures illustrate how to obtain the knowledge and the bottom figures illustrate how to use the knowledge for unpaired image-text matching.

# 3.1 MULTIMODAL ALIGNED SEMANTIC KNOWLEDGE

In addition to the semantic concepts, the studied knowledge also has another important property of cross-modal one-to-one alignment. For each word, its semantically related objects in different regions often exhibit diverse visual appearances, which could easily lead to confusion in practice. Therefore, rather than align each word to multiple related regions in an one-to-many manner, the MASK aligns each word to a single prototypical region, with the goal to alleviate the issue of appearance variation. In particular, we formulate the knowledge as a set of semantic concepts having paired multimodal representations $\{ ( w _ { k } , v _ { k } ) \} _ { k = 1 , \dots , K } .$ where $w _ { k }$ and $v _ { k }$ are the word embedding and prototypical region representation of the k-th semantic concept, respectively, and K is the total number of semantic concepts.

As shown in Figure 2, for each word, we compute the word embeddings $w _ { k }$ by using pre-trained word vectors. For each region, we first extract the raw region representations $r _ { j } ( j = 1 , \ldots , J _ { k } )$ by feeding a bounding box and an image into the pre-trained object detection model Faster-RCNN. Then we extract region representations $\mu _ { j }$ by utilizing the Prototype-Aware Encoder (P AE) h, with $r _ { j }$ as the input:

$$
\mu_ {j}, \sigma_ {j} = h (r _ {j}; \Theta_ {h}), \tag {1}
$$

where $\Theta _ { h }$ is the parameters of $h$ and $\sigma _ { j }$ represents the variance of distribution. Finally, we compute the prototypical region representations $v _ { k }$ by averaging all related region representations $\{ \mu _ { j } \} _ { j = 1 , \ldots , J _ { k } } \colon$

$$
v _ {k} = \frac {1}{J _ {k}} \sum_ {j = 1} ^ {J _ {k}} \mu_ {j}, \tag {2}
$$

where the $J _ { k }$ indicates the number of regions for the k-th semantic concept.

# 3.2 IMAGE EMBEDDING BRANCH

Given a batch B of paired regions and words, we first obtain raw region representations $R \ =$ $\{ r _ { n } \} _ { n = 1 , \dots , B } ~ ( R \in { \mathrm { \bf ~ \dot { \mathcal { R } } } } ^ { B \times M } )$ , where M is the dimension of $r _ { n }$ . We then extract region representations $\mu$ using the P AE model h, which takes R as input and consists of a Fully Connected (FC) layer followed by three self-attention layers:

$$
\mu , \sigma = h (R; \Theta_ {h}), \tag {3}
$$

where the mean $\mu \in { \mathcal R } ^ { B \times Z }$ and variance $\sigma ~ \in ~ \mathcal { R } ^ { B \times Z }$ are used to preserve the information of the raw region representations R by using the Feature Restoration Module (F RM) g comprising a self-attention layer and two FC layers:

$$
R ^ {\prime} = g (\mu , \sigma , z; \Theta_ {g}), \tag {4}
$$

where the z is a random vector sampled from a standard normal distribution and $\Theta _ { g }$ is the parameters of $g .$ The $Z$ represents the feature dimension of latent space. As shown in Figure ${ \overset { \cup } { 2 } } ,$ , the $P A E$ model h and F RM model g are trained jointly using the information retention loss function $\mathcal { L } _ { i r }$ :

$$
\mathcal {L} _ {i r} = D _ {K L} \left(N (\mu , \sigma^ {2}) | | N (0, 1)\right) + \mathbb {E} _ {\left(r _ {n}, r _ {n} ^ {\prime}\right) \sim \left(R, R ^ {\prime}\right)} \left[ \| r _ {n} - r _ {n} ^ {\prime} \| _ {2} ^ {2} \right], \tag {5}
$$

where the $D _ { K L } ( { \cal N } ( \mu , \sigma ^ { 2 } ) | | { \cal N } ( 0 , 1 ) )$ implies that the data distribution in the latent space gradually approaches the standard normal distribution. The $\mathbb { E } _ { ( r _ { n } , r _ { n } ^ { \prime } ) \sim ( R , R ^ { \prime } ) } \ \left[ \left\| r _ { n } - r _ { n } ^ { \prime } \right\| _ { 2 } ^ { 2 } \right]$ measures the difference between the reconstructed raw region representations R0 and the raw region representations R. The loss $\mathcal { L } _ { i r }$ ensures that the mean $\mu$ retains a significant amount of information from the raw region representations $R .$

Inspired by clustering theory and contrastive learning, we design a prototype consistency contrastive learning loss $\mathcal { L } _ { c l }$ to reduce the influence of distributional variance between prototypes and their related region representations. The loss $\mathcal { L } _ { c l }$ employs prototypes as class centers, maximizing the similarity between region representations and their corresponding prototypes while minimizing similarity with other prototypes, thereby achieving intra-class aggregation and inter-class separation. Compared to traditional instance-to-instance contrastive learning, $\mathcal { L } _ { c l }$ introduces prototypes as global semantic representatives, explicitly aggregating instances of the same class around their corresponding prototypes. This process constructs a more structured and discriminative representation space, enabling the model to capture clearer semantic boundaries. The loss $\mathcal { L } _ { c l }$ is defined as follows (Appendix G):

$$
\mathcal {L} _ {c l} = - \frac {1}{B} \sum_ {k = 1} ^ {B} \log \frac {\exp (v _ {k} \cdot \mu_ {+} / \tau)}{\sum_ {n = 1} ^ {B} \exp (v _ {k} \cdot \mu_ {n} / \tau)}, \tag {6}
$$

where the $\mu _ { + }$ refers to the region representations associated with the prototypical region representations vk, i.e., positive examples. The hyperparameter τ regulates the capacity of the model to distinguish between negative examples. The loss $\mathcal { L } _ { c l }$ encourages all region representations corresponding to the same word to be closer to each other, while driving the region representations corresponding to different words farther apart, which effectively mitigates the impact of variance among different words on the similarity computation.

# 3.3 TEXT EMBEDDING BRANCH

Given a batch B of paired regions and words, we obtain word embeddings $V = \{ w _ { k } \} _ { k = 1 , \dots , B } \ ( V \in$ $\mathcal { R } ^ { B \times N } )$ by utilizing the pretrained word vectors, where N is the dimension of $w _ { k }$ . Pre-trained word embeddings typically exhibit well-structured semantic properties, where semantically related words are mapped to vectors that are close to each other in the embedding space. To enable region representations to effectively capture semantic correlations between words, we utilize a Modality Transfer Model (M T M ) f with three self-attention layers and three FC layers that can map the mean $\mu$ output by the $P A E$ model h into the word embedding space:

$$
V ^ {\prime} = f (\mu ; \Theta_ {f}), \tag {7}
$$

where the $\Theta _ { f }$ is the parameters of the MT M model $f$ and $V ^ { \prime } \in \mathcal { R } ^ { B \times N }$ represents the predicted word embeddings. The model f is a relation-preserving equivariant mapping that lays the foundation for constructing prototypical region representations corresponding to OOD words. Formally, for any two region representations $\mu _ { i }$ and $\mu _ { j }$ , the function $f$ should satisfy (Appendix B):

$$
d _ {s} (f (\mu_ {i}; \Theta_ {f}), f (\mu_ {j}; \Theta_ {f})) \propto d _ {s} (\mu_ {i}, \mu_ {j}), \tag {8}
$$

where the distance metric $d _ { s }$ captures the pairwise relations between representations within each modality. The $P A E$ model h and MT M model $f$ are trained jointly using the cross-modal alignment loss function $\mathcal { L } _ { c m }$ (Appendix C, E and F):

$$
\mathcal {L} _ {c m} = \mathbb {E} [ (1 - c o s (\frac {w _ {i}}{\| w _ {i} \| _ {2}}, \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}})) ] + \mathbb {E} [ ((c o s (\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}, \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}) - c o s (\frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}}, \frac {\mu_ {j}}{\| \mu_ {j} \| _ {2}}))) ^ {2} ], \tag {9}
$$

where $w _ { i } ^ { \prime } , w _ { j } ^ { \prime } \in V ^ { \prime } ( i \neq j )$ and $w _ { i } \in V$ . The loss function $\mathcal { L } _ { c m }$ enforces the predicted word embeddings $V ^ { \prime }$ to gradually converge toward the word embeddings $V ,$ , while simultaneously ensuring that the region representations effectively capture the semantic relationships between words.

# 3.4 KNOWLEDGE-BASED IMAGE-TEXT MATCHING

To decide whether a given image and a text are matched or not, we first obtain a set of raw region representations $R = \mathsf { \bar { \{ } }  r _ { i } \} _ { i = 1 , \ldots , I } \ ( R \in { \mathcal { R } } ^ { I \times M } )$ using the Faster-RCNN above and a set of parsed words through tokenization operation implemented via $\mathrm { N L T K } ^ { 1 }$ , as shown in Figure 2. Then, we use the knowledge as a cross-modal bridge to represent all the words into the corresponding prototypical region representations $U = \{ v _ { j } \} _ { j = 1 , \dots , J } ~ ( \bar { U } \in \mathcal { R } ^ { J \times Z } )$ . For the set of regions R, we extract region representations $\mu \in \mathcal { R } ^ { I \times Z }$ by utilizing the $P A E$ model h. Finally, we obtain the desired global similarity score s for the given image and text as:

$$
s = \rho (\mu \cdot U ^ {T}), \tag {10}
$$

where $\rho ( \cdot )$ denotes the max-mean pooling operation, which first performs max pooling along the column dimension and then mean pooling along the row dimension of the input matrix.

However, the scope of knowledge is inherently limited and heavily reliant on the volume of paired data available in public datasets. The vocabulary size supported by the pre-trained word vectors significantly surpasses the scale of the existing knowledge. Therefore, for OOD words relative to the knowledge, their corresponding word embeddings can typically be obtained by leveraging pre-trained word vectors. To fully utilize these OOD words, we first sample m paired multimodal representations $\{ ( w _ { q } , v _ { q } ) \} _ { q = 1 } ^ { m }$ 1 from the knowledge. Then, we calculate the similarity scores $\{ s _ { q } \} _ { q = 1 } ^ { m }$ between the m word embeddings $\{ w _ { q } \} _ { q = 1 } ^ { m }$ and the word embedding $w _ { o u t } \colon$

$$
\{s _ {q} \} _ {q = 1} ^ {m} = \text { softmax } (w _ {\text { out }} \cdot \{w _ {q} \} _ {q = 1} ^ {m}). \tag {11}
$$

By utilizing the sampled prototypical region representations $\{ v _ { q } \} _ { q = 1 } ^ { m }$ as base vectors and the similarity scores $\{ s _ { q } \} _ { q = 1 } ^ { m }$ , we can obtain the prototypical region representation $v _ { o u t }$ corresponding to word embedding $w _ { o u t } \mathrm { . }$ :

$$
v _ {o u t} = \sum_ {q = 1} ^ {m} s _ {q} \cdot v _ {q}. \tag {12}
$$

In unpaired image-text matching, constructing prototypical region representations based on semantic similarities between words enables the effective utilization of information from OOD words.

To ensure the semantic quality of the visual prototypes constructed for OOD words, we select the top-m paired multimodal representations from the knowledge whose word embeddings are most relevant to OOD words. This selection strategy is motivated by the local linearity property of word embeddings on the semantic manifold. Semantically related words lie close to each other in the embedding space and approximately reside in a locally linear subspace. Consequently, the topm neighbors provide the most informative directions for reconstructing the corresponding visual representations. Moreover, the $\mathcal { L } _ { c m }$ constrains local alignment between the word embedding space and the visual prototype space, making nearest neighbors in the embedding space more likely to preserve geometric relationships in the prototype space, thereby reducing reconstruction bias. In this way, the top-m neighbors effectively capture the most salient semantic and structural information OOD words, we first normalize all word embeddings {wq}Kq=1 in the knowledge, and denote the needed for accurate and robust prototype estimation. To obtain these top-m semantic neighbors for $\{ w _ { q } \} _ { q = 1 } ^ { K }$ normalized embeddings as $\{ \frac { w _ { q } } { \| w _ { q } \| } \} _ { q = 1 } ^ { K }$ . Similarly, the normalized embedding of OOD words as $\frac { w _ { o u t } } { \| w _ { o u t } \| }$ The similarity between $\frac { w _ { o u t } } { \| w _ { o u t } \| }$ and { kwqk }Kq=1 are computed as: $\{ \frac { w _ { q } } { \| w _ { q } \| } \} _ { q = 1 } ^ { K }$ wq

$$
\{s _ {q} \} _ {q = 1} ^ {K} = \frac {w _ {\text { out }}}{\| w _ {\text { out }} \|} \cdot \left\{\frac {w _ {q}}{\| w _ {q} \|} \right\} _ {q = 1} ^ {K}. \tag {13}
$$

The top-m nearest neighbors $\{ s _ { q } \} _ { q = 1 } ^ { m }$ are then selected based on $\{ s _ { q } \} _ { q = 1 } ^ { K }$ to support subsequent visual prototype construction.

# 3.5 MODEL TRAINING

The studied knowledge mainly contains dataset-independent semantic concepts, with the goal to be generally applicable to different scenarios. The semantic concepts are multimodal, which includes objects and attributes in images, and nouns and adjectives in texts. To obtain them, we resort to publicly available dataset Visual Genome (VG) (Krishna et al., 2017) 2 and collect corresponding words and regions. For the textual knowledge, we obtain various words from synsets in the dataset. For the visual knowledge, we detect regions from images and then associate them with the words.

After collecting a set of words and their semantically related image regions from publicly available datasets, we train our model on these paired data to construct MASK. The loss of the entire training process is expressed as $\mathcal { L } \mathrm { : }$

$$
\mathcal {L} = \mathcal {L} _ {i r} + \lambda_ {1} \mathcal {L} _ {c m} + \lambda_ {2} \mathcal {L} _ {c l}, \tag {14}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are trade-off factors for balancing different losses. By optimizing the loss $\mathcal { L } .$ the region representations exhibit properties of high cohesion and low coupling, indicating that representations corresponding to the same word become more compact and semantically consistent.

# 3.6 RE-RANKING EXTENSION

The proposed MASK is a knowledge-based approach, which differs significantly from existing datadriven models. Due to this distinction, it is expected to exhibit complementary properties when combined with existing models. We extend MASK into a re-ranking method to re-rank the initial results produced by existing multimodal models.

Taking the sub-task of image retrieval as an example, given a text query and a gallery of L images, an existing model can compute similarity scores and produce a similarity vector $\breve { s } \in \breve { \mathcal { R } } ^ { L \times 1 }$ . By sorting the values of s˜ in descending order, the model ranks the images and identifies the top-k candidates, denoted as $\tilde { s } ^ { k } \in \mathcal { R } ^ { k \times 1 }$ . For the text query and the top-k retrieved images, we then compute an additional similarity vector $s ^ { k } \in \mathcal { R } ^ { k \times 1 }$ using MASK in an unpaired image-text matching setting. Finally, the two similarity vectors are combined using a balancing factor α:

$$
\hat {s} ^ {k} = Z S (\tilde {s} ^ {k}) + \alpha \cdot Z S (s ^ {k}), \tag {15}
$$

where ZS represents the Z-Score normalization and $\hat { s } ^ { k }$ is the new similarity vector that can be used to re-rank the top-k images to improve the rank of matched images.

# 4 EXPERIMENTS

# 4.1 DATASETS AND METRICS

We test the performance of MASK on two standard datasets: Flickr30K and MSCOCO. The commonly used evaluation criterions are “R@1”, “R@5” and “R@10”, i.e., recall rates at the top-1, 5 and 10 results. Following existing works (Ge et al., 2024), we use an additional criterion $\mathrm { { o f } ^ { \mathrm { { \sc \cdots } } } }$ b y summing all the recall rates to evaluate the overall performance. Experimental details are provided in the Appendix I.2.

# 4.2 UNPAIRED IMAGE-TEXT MATCHING

We design several experiments comparing state-of-the-art model-based matching methods (e.g., CHAN (Pan et al., 2023), DSRLN (Wu et al., 2024), CORA (Pham et al., 2024), BOOM (Li et al., 2024a), and 3SHNet (Ge et al., 2024)) and knowledge-based matching methods (e.g., MACK (Huang et al., 2022) and $M A C K ^ { V G - M }$ (Huang et al., 2024b)) to verify the effectiveness for unpaired image-text matching. We compare the performance of unpaired image-text matching in Table 1. Model-based matching and knowledge-based matching exhibit comparable performance on the Flickr30K dataset, whereas their performance diverges significantly on the MSCOCO dataset. Compared to Flickr30K, MSCOCO exhibits greater sample diversity, with images typically containing multiple target objects and semantic regions, resulting in more complex visual structures. The knowledge-based matching constructs explicit multimodal-aligned knowledge as a bridge between regions and words, facilitating more accurate modeling of local visual-semantic relationships in complex visual scenes.

Table 1: Performance comparison between model-based matching and knowledge-based matching on the Flickr30K and MSCOCO datasets for the unpaired image-text matching. 

<table><tr><td rowspan="3" colspan="2">Method</td><td colspan="7">Flickr30K dataset</td><td colspan="6">MSCOCO dataset</td><td></td></tr><tr><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td rowspan="5">model-based matching</td><td> $CHAN_{2023}$ </td><td>2.2</td><td>10.4</td><td>16.9</td><td>1.5</td><td>7.8</td><td>24.5</td><td>63.3</td><td>2.1</td><td>9.1</td><td>15.7</td><td>3.5</td><td>11.9</td><td>22.5</td><td>64.8</td></tr><tr><td> $DSRLN_{2024}$ </td><td>4.3</td><td>14.1</td><td>22.1</td><td>7.0</td><td>21.9</td><td>37.8</td><td>107.2</td><td>4.4</td><td>14.6</td><td>21.7</td><td>5.5</td><td>20.0</td><td>39.0</td><td>105.2</td></tr><tr><td> $CORA_{2024}$ </td><td>3.6</td><td>12.0</td><td>22.9</td><td>8.3</td><td>22.7</td><td>33.9</td><td>103.4</td><td>5.5</td><td>18.7</td><td>32.5</td><td>12.4</td><td>29.3</td><td>45.2</td><td>143.6</td></tr><tr><td> $BOOM_{2024}$ </td><td>3.9</td><td>12.6</td><td>23.8</td><td>8.3</td><td>22.6</td><td>35.2</td><td>106.4</td><td>5.8</td><td>19.2</td><td>32.6</td><td>12.9</td><td>29.8</td><td>45.4</td><td>145.7</td></tr><tr><td> $3SHNet_{2024}$ </td><td>3.8</td><td>12.2</td><td>23.1</td><td>8.1</td><td>22.0</td><td>34.3</td><td>103.5</td><td>6.0</td><td>19.7</td><td>33.4</td><td>13.2</td><td>29.6</td><td>47.8</td><td>149.7</td></tr><tr><td rowspan="3">knowledge-based matching</td><td> $MACK_{2022}$ </td><td>3.0</td><td>9.9</td><td>15.4</td><td>10.1</td><td>24.6</td><td>32.3</td><td>95.3</td><td>7.2</td><td>25.9</td><td>40.6</td><td>21.8</td><td>46.2</td><td>60.0</td><td>201.7</td></tr><tr><td> $MACK^{VG-M}_{2024}$ </td><td>3.8</td><td>11.3</td><td>17.4</td><td>10.4</td><td>26.8</td><td>35.1</td><td>104.8</td><td>7.2</td><td>25.2</td><td>41.4</td><td>21.9</td><td>46.6</td><td>62.9</td><td>205.2</td></tr><tr><td>MASK</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td><td>7.6</td><td>26.7</td><td>41.8</td><td>22.7</td><td>48.5</td><td>62.2</td><td>209.5</td></tr><tr><td rowspan="3">-w/o region prototypes</td><td> $MACK_{2022}$ </td><td>1.6</td><td>5.3</td><td>8.8</td><td>5.3</td><td>16.4</td><td>22.7</td><td>60.1</td><td>4.6</td><td>17.0</td><td>28.2</td><td>13.3</td><td>31.7</td><td>44.2</td><td>139.0</td></tr><tr><td> $MACK^{VG-M}_{2024}$ </td><td>1.7</td><td>7.0</td><td>10.9</td><td>5.3</td><td>16.2</td><td>23.4</td><td>64.5</td><td>4.6</td><td>16.9</td><td>29.6</td><td>13.3</td><td>31.4</td><td>45.3</td><td>141.1</td></tr><tr><td>MASK</td><td>2.5</td><td>9.1</td><td>14.5</td><td>5.6</td><td>17.1</td><td>23.9</td><td>72.7</td><td>5.2</td><td>18.9</td><td>31.7</td><td>13.6</td><td>33.5</td><td>45.6</td><td>148.5</td></tr><tr><td rowspan="3">-w/o max-mean pooling</td><td> $MACK_{2022}$ </td><td>2.5</td><td>9.0</td><td>14.3</td><td>1.7</td><td>5.6</td><td>8.6</td><td>41.7</td><td>6.3</td><td>21.4</td><td>33.3</td><td>3.9</td><td>11.6</td><td>17.6</td><td>94.1</td></tr><tr><td> $MACK^{VG-M}_{2024}$ </td><td>2.7</td><td>9.4</td><td>17.2</td><td>1.7</td><td>5.9</td><td>10.3</td><td>47.2</td><td>6.7</td><td>22.9</td><td>36.1</td><td>3.8</td><td>11.6</td><td>19.2</td><td>100.3</td></tr><tr><td>MASK</td><td>4.5</td><td>14.1</td><td>22.1</td><td>2.9</td><td>7.7</td><td>12.0</td><td>63.3</td><td>8.0</td><td>26.0</td><td>40.2</td><td>4.8</td><td>12.8</td><td>20.9</td><td>112.7</td></tr></table>

The region prototypes and max-mean pooling have a significant impact on knowledge-based unpaired matching methods. However, MASK consistently outperforms existing methods. This is primarily because the MASK exhibits a strong intra-class cohesion among region representations, i.e., the variance between any region representation and the prototypical region representation is relatively small. Consequently, replacing the prototypical region representation with a randomly selected region representation has a minimal impact on overall performance. Furthermore, incorporating semantic relationships between word embeddings reduces the coupling among region representations across words. Therefore, substituting max-mean pooling with global mean has a minor effect on overall performance.

# 4.3 ZERO-SHOT IMAGE-TEXT MATCHING

To evaluate the effectiveness of the MASK in complementing with pre-trained models for zero-shot image-text matching, we compare several state-of-the-art re-ranking strategies, including MACK (Huang et al., 2022), LeaP RR (Qu et al., 2023), $M A C K ^ { V G - M }$ (Huang et al., 2024b) and F R (Wei et al., 2025). The original and re-ranked performance are compared in Table 2. We can see that although the original accuracies are already high, further improvements can be achieved by applying re-ranking strategies to these pre-trained models. Among them, MASK yields more substantial performance gains. This can be attributed to the high cohesion and low coupling of the region representations, which ensure that each region representation remains closest to its corresponding prototypical region representation while maintaining substantial spatial separation from those of non-corresponding terms. Consequently, MASK reduces the risk of region representations being misclassified in zero-shot image-text matching. These evidences demonstrate that the MASK can be well combined with existing models to further improve their performance.

Table 2: Zero-shot image-text matching by re-ranking two state-of-the-art models on the Flickr30K and MSCOCO datasets. 

<table><tr><td rowspan="3">Method</td><td colspan="7">Flickr30K dataset</td><td colspan="7">MSCOCO dataset</td></tr><tr><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>CLIP</td><td>65.4</td><td>87.2</td><td>91.7</td><td>85.4</td><td>97.1</td><td>98.7</td><td>525.5</td><td>35.3</td><td>60.0</td><td>70.1</td><td>55.2</td><td>78.7</td><td>86.7</td><td>386.0</td></tr><tr><td> $CLIP + MACK_{2022}$ </td><td>66.8</td><td>88.2</td><td>92.6</td><td>86.2</td><td>97.2</td><td>98.9</td><td>529.9</td><td>36.9</td><td>61.6</td><td>71.7</td><td>55.7</td><td>79.6</td><td>87.1</td><td>392.6</td></tr><tr><td> $CLIP + LeaPRR_{2023}$ </td><td>65.9</td><td>87.9</td><td>92.8</td><td>86.2</td><td>97.3</td><td>98.1</td><td>528.2</td><td>36.2</td><td>61.9</td><td>70.8</td><td>55.9</td><td>80.2</td><td>85.4</td><td>390.4</td></tr><tr><td> $CLIP + MACK^{VG-M}_{2024}$ </td><td>66.9</td><td>88.4</td><td>92.8</td><td>87.6</td><td>96.9</td><td>99.0</td><td>531.6</td><td>37.2</td><td>62.0</td><td>71.8</td><td>56.3</td><td>80.0</td><td>87.1</td><td>394.4</td></tr><tr><td> $CLIP + FR_{2025}$ </td><td>66.4</td><td>88.4</td><td>93.1</td><td>86.8</td><td>97.3</td><td>99.1</td><td>531.1</td><td>36.7</td><td>62.1</td><td>72.4</td><td>56.2</td><td>80.3</td><td>87.8</td><td>395.5</td></tr><tr><td> $CLIP + MASK$ </td><td>67.3</td><td>88.7</td><td>93.9</td><td>87.9</td><td>97.6</td><td>98.9</td><td>534.3</td><td>37.6</td><td>62.7</td><td>73.9</td><td>56.7</td><td>80.8</td><td>88.7</td><td>400.4</td></tr><tr><td>ALBEF</td><td>59.9</td><td>84.8</td><td>90.5</td><td>78.2</td><td>95.5</td><td>97.9</td><td>506.9</td><td>40.2</td><td>68.4</td><td>78.9</td><td>62.4</td><td>85.9</td><td>92.1</td><td>428.3</td></tr><tr><td> $ALBEF + MACK_{2022}$ </td><td>61.8</td><td>85.8</td><td>91.3</td><td>80.1</td><td>96.4</td><td>97.7</td><td>513.1</td><td>41.0</td><td>69.0</td><td>79.4</td><td>62.4</td><td>86.1</td><td>92.7</td><td>430.9</td></tr><tr><td> $ALBEF + LeaPRR_{2023}$ </td><td>61.4</td><td>85.9</td><td>91.6</td><td>79.3</td><td>96.2</td><td>97.5</td><td>511.9</td><td>40.7</td><td>69.1</td><td>79.6</td><td>62.8</td><td>86.4</td><td>91.8</td><td>430.4</td></tr><tr><td> $ALBEF + MACK^{VG-M}_{2024}$ </td><td>61.8</td><td>86.1</td><td>91.6</td><td>80.5</td><td>95.5</td><td>97.9</td><td>513.4</td><td>41.6</td><td>69.5</td><td>79.7</td><td>63.0</td><td>86.0</td><td>92.4</td><td>432.2</td></tr><tr><td> $ALBEF + FR_{2025}$ </td><td>61.7</td><td>86.4</td><td>91.8</td><td>80.8</td><td>96.2</td><td>97.6</td><td>514.5</td><td>41.9</td><td>69.8</td><td>80.1</td><td>62.8</td><td>86.3</td><td>92.1</td><td>433.0</td></tr><tr><td> $ALBEF + MASK$ </td><td>63.1</td><td>86.6</td><td>92.0</td><td>81.6</td><td>96.9</td><td>97.9</td><td>518.1</td><td>42.1</td><td>70.5</td><td>80.8</td><td>63.6</td><td>86.9</td><td>92.3</td><td>436.2</td></tr></table>

# 4.4 KNOWLEDGE VISUALIZATION

To qualitatively illustrate major differences between MACK and MASK, we visualize two lowdimensional word distributions in Figure 3. The words in the four numbered groups (marked by dashed lines in different colors) are about animals, transports, faces and humans, respectively. In the left distribution corresponding to MACK, there are still some related words that have remote distances. In contrast, the right distribution generated by MASK is semantically more compact. The underlying mechanism is that semantic relationships between word embeddings are incorporated during model training, ensuring that the corresponding prototypical region representations also exhibit semantic associations. These evidences indicate that the MASK can make their prototypical region representations more discriminative.

# 4.5 OOD WORDS ANALYSIS AND LOSS ABLATION ANALYSIS

To evaluate the impact of OOD words on image-text matching accuracy, we conduct a comparative experiment in Table 3. We observe that image-text matching accuracy significantly improves in both image retrieval and image annotation tasks when OOD words are incorporated, and this improvement is consistently validated across different datasets. Therefore, leveraging the semantic relationships between OOD words and known words to construct corresponding prototypical region representations for OOD words is an effective approach. This phenomenon can be attributed to the relation-preserving equivariant mapping, as shown in Eq (8). As a result, region representations inherit the semantic structure encoded in the word embeddings, allowing the relationships between regions to reflect semantic distances and similarities, thereby enhancing the generalization ability of the matching process.

![](images/fa81af41e97c6391ab4569c6108ae4d5e63c2e87dbd4593499073dab26ee4c63.jpg)

<details>
<summary>text_image</summary>

chalk
heart
flies
mho
tall
correction
earlet
redish
primes
brush
carney
meele
threshold
archim
artemar
chalk
crivemum
mossus
carver
armedida
france
base
carer
center
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
carve
</details>

(a) MACK

![](images/6e87f2ec55b62b8bf12a336fcc3c98e8d56af1496c2abb944f5fb268f50b650d.jpg)

<details>
<summary>text_image</summary>

census
bus
train
sctensen
aetser
tata
shelfaner
sultensen
chick
child
height
student
systen
school
threshold
real
franchise
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetser
aetzer
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systen
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systent
systest
</details>

(b) MASK   
Figure 3: Visualization of prototypical region representations from MACK and MASK. Each word indicates its corresponding prototypical region representation embedded by t-SNE. In the two word distributions, we group semantically related words by using same-colored dashed lines.

Table 3: Ablation of the overall loss and the impact of OOD words on unpaired image-text matching. 

<table><tr><td rowspan="3">Method</td><td colspan="7">Flickr30K dataset</td><td colspan="7">MSCOCO dataset</td></tr><tr><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>MASK</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td><td>7.6</td><td>26.7</td><td>41.8</td><td>22.7</td><td>48.5</td><td>62.2</td><td>209.5</td></tr><tr><td> $MASK_{w/o\ OODwords}$ </td><td>4.5</td><td>13.6</td><td>20.5</td><td>11.8</td><td>28.5</td><td>37.3</td><td>116.2</td><td>7.2</td><td>24.5</td><td>38.2</td><td>22.2</td><td>44.6</td><td>56.4</td><td>193.1</td></tr><tr><td>MASK</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td><td>7.6</td><td>26.7</td><td>41.8</td><td>22.7</td><td>48.5</td><td>62.2</td><td>209.5</td></tr><tr><td> $MASK_{w/o\ \mathcal{L}_{cm}}$ </td><td>3.1</td><td>10.5</td><td>15.2</td><td>10.2</td><td>26.6</td><td>35.4</td><td>101.0</td><td>5.2</td><td>16.5</td><td>24.9</td><td>17.2</td><td>37.6</td><td>49.4</td><td>150.8</td></tr><tr><td> $MASK_{w/o\ \mathcal{L}_{cl}}$ </td><td>2.6</td><td>8.3</td><td>13.7</td><td>9.1</td><td>23.9</td><td>34.8</td><td>92.4</td><td>4.5</td><td>13.6</td><td>20.9</td><td>13.3</td><td>31.0</td><td>40.2</td><td>123.5</td></tr></table>

To testify the contribution of each component of loss $\mathcal { L }$ to overall performance, we compare the performance of various losses in Table 3. It can be observed that the performance of the overall loss function L deteriorates when certain components are omitted. Among them, the prototype consistency contrastive learning loss $\mathcal { L } _ { c l }$ makes the largest contribution to the performance. It is reasonable since the loss $\mathcal { L } _ { c l }$ constrains all region representations related the same word to be close to each other during model training. Additionally, the semantic relationships between word embeddings in loss $\mathcal { L } _ { c m }$ serve only as a reference for determining the degree of separation between region representations. Therefore, the loss $\mathcal { L } _ { c l }$ can make the prototypical region representations more discriminative, which obviously affects the accuracy of matching.

# 4.6 HYPERPARAMETER ANALYSIS

The proposed MASK involves two trade-off parameters $\lambda _ { 1 }$ and $\lambda _ { 2 }$ in Eq.(14). To evaluate the impact of different hyperparameters on matching accuracy, we design three controlled experiments in Table 4. The results indicate that MASK achieves the best performance in unpaired image-text matching at $\lambda _ { 1 } = \lambda _ { 2 }$ . Compared to $\begin{array} { r } { \frac { \lambda _ { 1 } } { \lambda _ { 2 } } = 3 . 0 ( \mathrm { i . e . , } \lambda _ { 1 } > \lambda _ { 2 } ) } \end{array}$ , the $\frac { \lambda _ { 1 } } { \lambda _ { 2 } } = 0 . 3$ achieves better performance, with improvements of approximately 3.5% and 10.3% in Rs on Flickr30K and MSCOCO, respectively. These findings suggest that the cross-modal alignment loss and prototype consistency contrastive loss are complementary, each contributing distinct yet essential benefits to the overall performance. By jointly optimizing these two losses in a balanced manner, we mitigate the risk of overfitting to a single loss term, thereby improving the model’s generalization capability.

Table 4: Unpaired image-text matching by MASK using different $\lambda _ { 1 } / \lambda _ { 2 }$ on the Flickr30K and MSCOCO datasets. 

<table><tr><td rowspan="3"> $\frac{\lambda_1}{\lambda_2}$ </td><td colspan="7">Flickr30K dataset</td><td colspan="7">MSCOCO dataset</td></tr><tr><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>0.3</td><td>4.2</td><td>13.0</td><td>19.2</td><td>11.6</td><td>27.9</td><td>36.8</td><td>112.7</td><td>6.8</td><td>22.9</td><td>35.5</td><td>20.9</td><td>43.2</td><td>55.8</td><td>185.1</td></tr><tr><td>1.0</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td><td>7.6</td><td>26.7</td><td>41.8</td><td>22.7</td><td>48.5</td><td>62.2</td><td>209.5</td></tr><tr><td>3.0</td><td>3.9</td><td>12.2</td><td>18.0</td><td>11.2</td><td>27.5</td><td>36.4</td><td>109.2</td><td>6.3</td><td>21.0</td><td>32.3</td><td>19.8</td><td>41.5</td><td>53.9</td><td>174.8</td></tr></table>

# 4.7 ANALYSIS OF SAMPLING SIZE

The sampling size m for selecting paired multimodal representations critically affects the semantic quality of the constructed prototypes. To evaluate the impact of different sampling sizes on matching accuracy, we perform the experiment of unpaired image-text matching by MASK using different sampling sizes m on the Flickr30K and MSCOCO datasets in Table 5. Experimental results show that the matching accuracy follows a rise–then–fall trend as the sampling size m increases, achieving its optimum around $m = 5 0$ . This phenomenon can be explained as follows. When the sampling size is too small, the constructed visual prototype relies excessively on only a few nearest neighbors. Although this preserves strong local semantic characteristics, it also makes the prototype highly sensitive to noise and outliers in the word embedding space. As the sampling size increases to a moderate level, more semantically relevant neighbors contribute their visual information, thereby enhancing robustness and discriminability. However, when the sampling size becomes too large, semantically weak or marginal neighbors begin to dominate. Their less relevant visual cues dilute the contributions of the core semantic neighbors, ultimately reducing matching accuracy.

Table 5: Unpaired image-text matching by MASK using different sampling sizes m on the Flickr30K and MSCOCO datasets. 

<table><tr><td rowspan="3">m</td><td colspan="7">Flickr30K dataset</td><td colspan="7">MSCOCO dataset</td></tr><tr><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>1</td><td>4.6</td><td>13.9</td><td>20.9</td><td>11.9</td><td>28.9</td><td>37.7</td><td>117.9</td><td>7.3</td><td>25.1</td><td>39.1</td><td>22.3</td><td>45.6</td><td>57.9</td><td>197.3</td></tr><tr><td>10</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td><td>7.6</td><td>26.7</td><td>41.8</td><td>22.7</td><td>48.5</td><td>62.2</td><td>209.5</td></tr><tr><td>50</td><td>5.0</td><td>15.8</td><td>23.2</td><td>12.3</td><td>31.4</td><td>40.4</td><td>128.1</td><td>7.9</td><td>28.5</td><td>44.7</td><td>23.1</td><td>51.6</td><td>66.8</td><td>222.6</td></tr><tr><td>100</td><td>4.9</td><td>15.3</td><td>22.6</td><td>12.2</td><td>30.7</td><td>39.7</td><td>125.4</td><td>7.7</td><td>27.6</td><td>43.3</td><td>22.9</td><td>50.0</td><td>64.5</td><td>216.0</td></tr></table>

# 4.8 RE-RANKING ANALYSIS

An important balancing factor α, is introduced for re-ranking the existing models in Eq.(15). To investigate the impact of α on final performance, we conduct zero-shot image-text matching experiments using MASK (CLIP) in Table 6. The results show that the best performance on both datasets is obtained at $\alpha = 0 . 1 5$ . When α is relatively small, the re-ranking method yields limited improvements, as matching performance is largely dominated by the pre-trained CLIP model. In contrast, as α increases, the influence of the MASK on matching performance becomes more pronounced. Since the knowledge is not learned from domain-specific paired image-text data, potential distributional discrepancies may arise, which can adversely affect the overall performance. Therefore, an appropriate value of α must be carefully chosen to achieve an optimal balance between semantic alignment capability and generalization performance.

Table 6: Zero-shot image-text matching by MASK (CLIP) using different α on the Flickr30K and MSCOCO datasets. 

<table><tr><td rowspan="3">α</td><td colspan="7">Flickr30K dataset</td><td colspan="7">MSCOCO dataset</td></tr><tr><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>0</td><td>65.4</td><td>87.2</td><td>91.7</td><td>85.4</td><td>97.1</td><td>98.7</td><td>525.5</td><td>35.3</td><td>60.0</td><td>70.1</td><td>55.2</td><td>78.7</td><td>86.7</td><td>386.0</td></tr><tr><td>0.1</td><td>67.1</td><td>88.7</td><td>93.7</td><td>87.2</td><td>97.3</td><td>99.1</td><td>533.1</td><td>37.1</td><td>62.3</td><td>73.6</td><td>56.3</td><td>80.3</td><td>88.6</td><td>398.2</td></tr><tr><td>0.15</td><td>67.3</td><td>88.7</td><td>93.9</td><td>87.9</td><td>97.6</td><td>98.9</td><td>534.3</td><td>37.6</td><td>62.7</td><td>73.9</td><td>56.7</td><td>80.8</td><td>88.7</td><td>400.4</td></tr><tr><td>0.2</td><td>66.9</td><td>88.5</td><td>93.6</td><td>86.4</td><td>97.1</td><td>99.0</td><td>531.5</td><td>36.9</td><td>62.2</td><td>73.4</td><td>55.6</td><td>80.5</td><td>88.3</td><td>396.9</td></tr></table>

# 5 CONCLUSION

This work has studied a practically important but seldom investigated problem as unpaired imagetext matching. To deal with this problem, we propose multimodal aligned semantic knowledge, which leverages word embeddings as bridges to associate words with prototypes, capturing semantic relationships between words, and further utilizing information from OOD words. Additionally, the introduction of prototype consistency contrastive loss effectively mitigates the impact of variance in unpaired matching. Code is available at https://github.com/AndroidDevelopersTools/MASK.

# ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (62376138, 92367202), and in part by the Jinan-NTU Green Technology Research Institute (GreenTRI).

# REFERENCES

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pp. 213–229. Springer, 2020.   
Shi Chen and Qi Zhao. Divide and conquer: Answering questions with object factorization and compositional reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6736–6745, 2023.   
Yang Feng, Lin Ma, Wei Liu, and Jiebo Luo. Unsupervised image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4125–4134, 2019.   
Zhongtian Fu, Kefei Song, Luping Zhou, and Yang Yang. Noise-aware image captioning with progressively exploring mismatched words. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 12091–12099, 2024.   
Yuxiao Gao, Fuwei Zhang, Zhao Zhang, Xiaoshuang Min, and Fuzhen Zhuang. Mixed-curvature multi-modal knowledge graph completion. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 11699–11707, 2025.   
Xuri Ge, Songpei Xu, Fuhai Chen, Jie Wang, Guoxin Wang, Shan An, and Joemon M Jose. 3shnet: Boosting image–sentence retrieval via visual semantic–spatial self-highlighting. Information Processing & Management, 61(4):103716, 2024.   
Jiuxiang Gu, Shafiq Joty, Jianfei Cai, Handong Zhao, Xu Yang, and Gang Wang. Unpaired image captioning via scene graph alignments. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10323–10332, 2019.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.   
Yan Huang, Jingdong Wang, and Liang Wang. Few-shot image and sentence matching via aligned cross-modal memory. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(6): 2968–2983, 2021.   
Yan Huang, Yuming Wang, Yunan Zeng, and Liang Wang. Mack: Multimodal aligned conceptual knowledge for unpaired image-text matching. Advances in Neural Information Processing Systems, 35:7892–7904, 2022.   
Yan Huang, Yuming Wang, Yunan Zeng, Junshi Huang, Zhenhua Chai, and Liang Wang. Unpaired image-text matching via multimodal aligned conceptual knowledge. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–17, 2024a. doi: 10.1109/TPAMI.2024.3432552.   
Yan Huang, Yuming Wang, Yunan Zeng, Junshi Huang, Zhenhua Chai, and Liang Wang. Unpaired image-text matching via multimodal aligned conceptual knowledge. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024b.   
Jose Iovino. Stable banach spaces and banach space structures, i: Fundamentals. In ´ Models, algebras, and proofs, pp. 77–95. CRC Press, 2021.   
Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017.

Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, and Xiaodong He. Stacked cross attention for image-text matching. In Proceedings of the European conference on computer vision (ECCV), pp. 201–216, 2018.   
Paul Lerner, Olivier Ferret, and Camille Guinaudeau. Cross-modal retrieval for knowledge-based visual question answering. In European Conference on Information Retrieval, pp. 421–438. Springer, 2024.   
Gen Li, Nan Duan, Yuejian Fang, Ming Gong, and Daxin Jiang. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 11336–11344, 2020.   
Linyu Li, Zhi Jin, Yichi Zhang, Dongming Jin, Chengfeng Dou, Yuanpeng He, Xuan Zhang, and Haiyan Zhao. Towards structure-aware model for multi-modal knowledge graph completion. arXiv preprint arXiv:2505.21973, 2025.   
Zhe Li, Lei Zhang, Kun Zhang, Yongdong Zhang, and Zhendong Mao. Improving image-text matching with bidirectional consistency of cross-modal alignment. IEEE Transactions on Circuits and Systems for Video Technology, 34(7):6590–6607, 2024a.   
Zheng Li, Caili Guo, Xin Wang, Hao Zhang, and Yanjun Wang. Integrating listwise ranking into pairwise-based image-text retrieval. Knowledge-Based Systems, 287:111431, 2024b.   
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ´ Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755. Springer, 2014.   
Chunxiao Liu, Zhendong Mao, Tianzhu Zhang, Hongtao Xie, Bin Wang, and Yongdong Zhang. Graph structured network for image-text matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10921–10930, 2020.   
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.   
Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11976–11986, 2022.   
Ovg ¨ u¨ Ozdemir and Erdem Akag ¨ und ¨ uz. Enhancing visual question answering through question- ¨ driven image captions as prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1562–1571, 2024.   
Zhengxin Pan, Fangyu Wu, and Bailing Zhang. Fine-grained image-text matching by cross-modal hard aligning network. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19275–19284, 2023.   
Khoi Pham, Chuong Huynh, Ser-Nam Lim, and Abhinav Shrivastava. Composing object relations and attributes for image-text matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14354–14363, 2024.   
Leigang Qu, Meng Liu, Wenjie Wang, Zhedong Zheng, Liqiang Nie, and Tat-Seng Chua. Learnable pillar-based re-ranking for image-text retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1252–1261, 2023.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.   
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015.

Richard Socher, Milind Ganjoo, Christopher D Manning, and Andrew Ng. Zero-shot learning through cross-modal transfer. Advances in neural information processing systems, 26, 2013.   
Liya Wang, Haipeng Chen, Yu Liu, and Yingda Lyu. Regular constrained multimodal fusion for image captioning. IEEE Transactions on Circuits and Systems for Video Technology, 2024a.   
Yabing Wang, Le Wang, Qiang Zhou, Zhibin Wang, Hao Li, Gang Hua, and Wei Tang. Multimodal llm enhanced cross-lingual cross-modal retrieval. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 8296–8305, 2024b.   
Wenzhang Wei, Zhipeng Gui, Changguang Wu, Anqi Zhao, Dehua Peng, and Huayi Wu. Dynamic visual semantic sub-embeddings and fast re-ranking for image-text retrieval. IEEE Transactions on Multimedia, 2025.   
Dongqing Wu, Huihui Li, Cang Gu, Lei Guo, and Hang Liu. Dual stream relation learning network for image-text retrieval. IEEE Transactions on Multimedia, 2024.   
Yiling Wu, Shuhui Wang, Guoli Song, and Qingming Huang. Learning fragment self-attention embeddings for image-text matching. In Proceedings of the 27th ACM international conference on multimedia, pp. 2088–2096, 2019.   
Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014.   
Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel Ni, and Heung-Yeung Shum. DINO: DETR with improved denoising anchor boxes for end-to-end object detection. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=3mRwyG5one.   
Qi Zhang, Zhen Lei, Zhaoxiang Zhang, and Stan Z Li. Context-aware attention network for imagetext retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3536–3545, 2020.

# A TABLE OF NOTATION

We list the notation used in this paper in Table 7, for the convenience of reference.

Table 7: Notation used in the paper. 

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $w_{k}$ </td><td>Word embedding directly from pre-trained word vectors</td></tr><tr><td> $v_{k}$ </td><td>Learnable prototypical region representation (prototype) of  $k$ -th semantic concept</td></tr><tr><td> $r_{j}$ </td><td>Raw region representation from pre-trained object detection model Faster-RCNN</td></tr><tr><td> $\mu_{j}$ </td><td>The mean of the latent space corresponds to the region representation</td></tr><tr><td> $\sigma_{j}$ </td><td>The variance of the latent space</td></tr><tr><td> $z$ </td><td>Random vector sampled from Gaussian distribution</td></tr><tr><td> $R$ </td><td> $\{r_{n}\}_{n=1,\dots,B} (R \in \mathcal{R}^{B \times M})$ </td></tr><tr><td> $\mu_{+}$ </td><td>Region representation associated with the prototypical region representations  $v_{k}$ </td></tr><tr><td> $V$ </td><td> $\{w_{k}\}_{k=1,\dots,B} (V \in \mathcal{R}^{B \times N})$ </td></tr><tr><td> $d_{s}$ </td><td>Distance metrics, e.g., cosine similarity</td></tr><tr><td> $U$ </td><td> $\{v_{j}\}_{j=1,\dots,J} (U \in \mathcal{R}^{J \times Z})$ </td></tr><tr><td> $\rho(\cdot)$ </td><td>Max-mean pooling</td></tr><tr><td> $w_{out}$ </td><td>Word embedding  $w_{out}$  corresponding to OOD word</td></tr><tr><td> $s_{q}$ </td><td>Similarity score</td></tr><tr><td> $v_{out}$ </td><td>Prototypical region representation  $v_{out}$  corresponding to word embedding  $w_{out}$ </td></tr><tr><td> $\tilde{s}^{k}$ </td><td>Top- $k$  similarity vector from existing multimodal model</td></tr><tr><td> $s^{k}$ </td><td>Top- $k$  similarity vector from MASK</td></tr></table>

# B PROOF: EQ. 8 FACILITATES THE CONSTRUCTION OF CORRESPONDING PROTOTYPES FOR OOD WORDS.

Given a paired multimodal knowledge $\{ ( w _ { k } , v _ { k } ) \} _ { k = 1 , \dots , K }$ , where $w _ { k } \in \mathcal { R } ^ { N }$ and $\boldsymbol { v _ { k } } \in \mathcal { R } ^ { M }$ are the word embedding and prototypical region representation of the k-th semantic concept, respectively, and K is the total number of semantic concepts. We sample from the multimodal knowledge $\{ ( w _ { k } , v _ { k } ) \} _ { k = 1 , \dots , K }$ to obtain a subset $\begin{array} { r } { \left\{ \left( w _ { k } , v _ { k } \right) \right\} _ { k = 1 , \dots , \hat { K } } . } \end{array}$ Our objective is to utilize the known word embedding $w _ { o u t }$ and $\{ ( w _ { k } , v _ { k } ) \} _ { k = 1 , \dots , \hat { K } }$ to construct $v _ { o u t }$ such that the following equation is satisfied:

$$
d _ {s} (f (v _ {k}), f (v _ {o u t})) \propto d _ {s} (v _ {k}, v _ {o u t}), k \in \{1, \dots , \hat {K} \}, \tag {16}
$$

$$
a n d \quad f (v _ {o u t}) \approx w _ {o u t}. \tag {17}
$$

We first rewrite Eq. 16 into an equidistant form. Then, there exists a $\beta > 0$ such that:

$$
d _ {s} (f (v _ {k}), f (v _ {\text { out }})) = \beta d _ {s} (v _ {k}, v _ {\text { out }}). \tag {18}
$$

Define the scaled mapping $\begin{array} { r } { F : = \frac { 1 } { \beta } \ f . } \end{array}$ For any $k ,$ out:

$$
d _ {s} (F (v _ {k}), F (v _ {o u t})) = d _ {s} (v _ {k}, v _ {o u t}). \tag {19}
$$

That is, F precisely preserves the euclidean distance among these prototypes. An isometric mapping in euclidean space exhibits a well-defined structural property, which is a fundamental result in classical geometry and functional analysis. Specifically, any distance-preserving mapping defined on the euclidean space must be a rigid transformation (Iovino, 2021). In other words, there exists an orthogonal matrix A that represents a rotation or reflection, and a translation vector t, such that the mapping can be expressed as:

$$
F (v _ {k}) = A v _ {k} + t. \tag {20}
$$

Therefore, the f can be expressed as:

$$
f (v _ {k}) = \beta A v _ {k} + \beta t. \tag {21}
$$

Since $A$ is an orthogonal matrix, its inverse $A ^ { - 1 }$ must exist and is equal to its transpose $A ^ { \top }$ . Therefore, we can derive:

$$
v _ {k} = A ^ {\top} (F (v _ {k}) - t) = A ^ {\top} (\frac {1}{\beta} f (v _ {k}) - t). \tag {22}
$$

For any given $w _ { k }$ in the knowledge, according to Eq. 17, we have:

$$
v _ {k} = A ^ {\top} (\frac {1}{\beta} w _ {k} - t). \tag {23}
$$

Therefore, we only need to determine the values of A, t, and $\beta$ to obtain the visual prototype vector $v _ { o u t }$ corresponding to the OOD words $w _ { o u t }$ . Extending Eq. 20 to the general case where A is no longer a strictly orthogonal matrix, we obtain the following equation using the column orthogonality condition:

$$
\left| \left| A v _ {k} - A v _ {\text { out }} \right| \right| _ {2} = \left| \left| v _ {k} - v _ {\text { out }} \right| \right| _ {2} \Leftrightarrow A ^ {\top} A = I _ {\min \{M, N \}}, \tag {24}
$$

Here, there exists an $A ^ { \top }$ such that Eq. 23 holds. We do not need to know the values of $A , t ,$ and $\beta$ a priori, as they can be uniquely determined from the paired knowledge $\{ ( w _ { k } , v _ { k } ) \} _ { k = 1 , \dots , K }$ using standard similarity Procrustes decomposition. First, we construct matrix $W _ { 1 } = [ v _ { 1 } , \dots , v _ { K } ]$ and matrix $W _ { 2 } = [ w _ { 1 } \mathrm { , } \ldots , w _ { K } ]$ . Next, we compute the centroids of each set:

$$
\bar {v} = \frac {1}{K} \sum_ {k = 1} ^ {K} v _ {k} \in \mathcal {R} ^ {M}, \quad \bar {w} = \frac {1}{K} \sum_ {k = 1} ^ {K} w _ {k} \in \mathcal {R} ^ {N}. \tag {25}
$$

We centralize the knowledge using the computed centroids v¯ and w¯:

$$
\tilde {W} _ {1} = \left[ v _ {1} - \bar {v}, \dots , v _ {K} - \bar {v} \right] \in \mathcal {R} ^ {M \times K}, \quad \tilde {W} _ {2} = \left[ w _ {1} - \bar {w}, \dots , w _ {K} - \bar {w} \right] \in \mathcal {R} ^ {N \times K}. \tag {26}
$$

From here, we can separate out t, and subsequently only need to consider the linear transformation and scaling. Next, we construct the covariance matrix $C { : }$

$$
C = \tilde {W} _ {1} \tilde {W} _ {2} ^ {\top} \in \mathcal {R} ^ {M \times N}. \tag {27}
$$

Let $M > N$ . And then, we utilize SVD decomposition to obtain the values of A, t, and $\beta \colon$

$$
C = U \Sigma V ^ {\top}, U \in \mathcal {R} ^ {M \times M}, \Sigma \in \mathcal {R} ^ {M \times N}, V ^ {\top} \in \mathcal {R} ^ {N \times N}, \tag {28}
$$

$$
A = U \left[ \begin{array}{c} I _ {N} \\ 0 _ {(M - N) \times N} \end{array} \right] V ^ {\top}, A \in \mathcal {R} ^ {M \times N}, \tag {29}
$$

$$
\beta = \frac {\operatorname{trace} \left(A \tilde {W} _ {2} \tilde {W} _ {1} ^ {\top}\right)}{\operatorname{trace} \left(\tilde {W} _ {2} \tilde {W} _ {1} ^ {\top}\right)}, \tag {30}
$$

$$
t = \bar {w} - \beta A \bar {v}, \quad t \in \mathcal {R} ^ {M}. \tag {31}
$$

Finally, we substitute the values of $A , t ,$ and $\beta$ into Eq. 23 to obtain the prototype vector $v _ { o u t }$ corresponding to the word embedding $w _ { o u t }$ .

# C PROOF: THE COSINE SIMILARITY IN EQ. 9 IS A REASONABLE DISTANCE METRIC $d _ { s }$ .

Euclidean space possesses strict linear structure preservation properties. It is the only metric satisfying translation and rotation invariance, and is naturally compatible with linear mappings. This allows the structural alignment problem between visual and linguistic spaces to be transformed into a standard orthogonal Procrustes problem. Therefore, we can equate Eq. 19 with Eq. 20 in euclidean space.

Cosine similarity and euclidean distance are not inherently equivalent. However, after vector normalization, there exists a strict monotonic mapping between cosine similarity and euclidean distance. This implies that, cosine similarity can be regarded as a form of ”Euclidean-like distance,” thereby satisfying the prerequisites for similarity transformations. Let two vectors $w _ { k }$ and $w _ { k } ^ { \prime } ,$ their cosine similarity is defined as:

$$
\cos (w _ {k}, w _ {k} ^ {\prime}) = \frac {w _ {k} \cdot w _ {k} ^ {\prime}}{| | w _ {k} | | | | w _ {k} ^ {\prime} | |}. \tag {32}
$$

where || · || denotes the L1-norm. The euclidean distance is defined as:

$$
d _ {E} (w _ {k}, w _ {k} ^ {\prime}) = | | w _ {k} - w _ {k} ^ {\prime} | |. \tag {33}
$$

We normalize each vector such that $\lvert | w _ { k } \rvert | = \rvert | w _ { k } ^ { \prime } \rvert | = 1$ , yielding:

$$
d _ {E} (w _ {k}, w _ {k} ^ {\prime}) ^ {2} = | | w _ {k} - w _ {k} ^ {\prime} | | ^ {2} = | | w _ {k} | | ^ {2} + | | w _ {k} ^ {\prime} | | ^ {2} - 2 \cdot w _ {k} w _ {k} ^ {\prime} = 2 - 2 c o s (w _ {k}, w _ {k} ^ {\prime}). \tag {34}
$$

$$
\cos (w _ {k}, w _ {k} ^ {\prime}) = 1 - \frac {1}{2} d _ {E} (w _ {k}, w _ {k} ^ {\prime}) ^ {2}. \tag {35}
$$

Combining Eq. 34 and Eq. 35, it can be concluded that after normalization, a one-to-one monotonic functional relationship exists between cosine similarity and euclidean distance.

$$
\mathcal {L} _ {c m} = \mathbb {E} \underbrace {[ (1 - c o s (\frac {w _ {i}}{\| w _ {i} \| _ {2}} , \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}})) ]} _ {w o r d - a l i g n m e n t} + \mathbb {E} \underbrace {[ ((c o s (\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}} , \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}) - c o s (\frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}} , \frac {\mu_ {j}}{\| \mu_ {j} \| _ {2}}))) ^ {2} ]} _ {s t r u c t u r e - p r e s e r v i n g}. (3 6)
$$

The loss $\mathcal { L } _ { c m }$ comprises two components: the first item enforces alignment between predicted and pre-trained word embeddings, while the second ensures that region representations capture the structural relationships among words.

$$
\frac {\partial (1 - \cos (\frac {w _ {i}}{\| w _ {i} \| _ {2}} , \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}))}{\partial \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}} = \frac {(w _ {i} ^ {\top} w _ {i}) w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2} ^ {2} \| w _ {i} \| _ {2}} - \frac {w _ {i}}{\| w _ {i} \| _ {2}}. \tag {37}
$$

$$
\frac {\partial (1 - \cos (\frac {w _ {i}}{\| w _ {i} \| _ {2}} , \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}))}{\partial \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}} = 0 \Longrightarrow w _ {i} = w _ {i} ^ {\prime}. \tag {38}
$$

$$
\frac {\partial \left(\left[ \left(\cos \left(\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}, \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}\right) - \cos \left(\frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}} , \frac {\mu_ {j}}{\| \mu_ {j} \| _ {2}}\right)\right) ^ {2} \right]\right)}{\partial \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}} \tag {39}
$$

$$
= 2 (\cos (\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}, \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}) - \cos (\frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}}, \frac {\mu_ {j}}{\| \mu_ {j} \| _ {2}})) \frac {\partial (\cos (\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}} , \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}))}{\partial \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}}.
$$

$$
\frac {\partial \left(\cos \left(\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}} , \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}\right)\right)}{\partial \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}} = \frac {w _ {j} ^ {\prime}}{\left\| w _ {j} ^ {\prime} \right\| _ {2}} - \frac {\left(w _ {i} ^ {\prime} \cdot w _ {j} ^ {\prime}\right) w _ {i} ^ {\prime}}{\left\| w _ {i} ^ {\prime} \right\| _ {2} ^ {2} \left\| w _ {j} ^ {\prime} \right\| _ {2}}. \tag {40}
$$

$$
\frac {\partial \left(\left[ \left(\cos \left(\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}} , \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}\right) - \cos \left(\frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}} , \frac {\mu_ {j}}{\| \mu_ {j} \| _ {2}}\right)\right) ^ {2} \right]\right)}{\partial \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}} = 0 \tag {41}
$$

$$
\Longrightarrow c o s (\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}, \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}) = c o s (\frac {\mu_ {i} ^ {\prime}}{\| \mu_ {i} ^ {\prime} \| _ {2}}, \frac {\mu_ {j} ^ {\prime}}{\| \mu_ {j} ^ {\prime} \| _ {2}}).
$$

# D PROOF: THE OBJECTIVE EXISTENCE OF THE ISOMETRIC HYPOTHESIS.

Proof: For any finite set of region representations $\{ \mu _ { i } \} _ { i = 1 } ^ { n }$ , there necessarily exists a mapping f in an appropriate Euclidean space such that:

$$
d _ {s} (f (\mu_ {i}; \Theta_ {f}), f (\mu_ {j}; \Theta_ {f})) \propto d _ {s} (\mu_ {i}, \mu_ {j}). \tag {42}
$$

Given n non-zero vectors $ { \mu } _ { 1 } , \dots ,  { \mu } _ { n } \in  { \mathcal { R } } ^ { Z }$ , normalize them to obtain:

$$
\hat {\mu} _ {i} := \frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}}, i = 1, \dots , n. \tag {43}
$$

We aim to construct a set of vectors $\{ y _ { i } \} _ { i = 1 } ^ { n }$ in a Euclidean space as $f ( \mu _ { i } )$ , and demonstrate that $c o s ( y _ { i } , y _ { j } ) = c o s ( \mu _ { i } , \mu _ { j } )$ can be achieved. Next, we organize the pairwise similarities into a matrix, facilitating the use of spectral decomposition to construct vectors that satisfy the given inner product relationships. We define the $n \times n$ Gram matrix G as follows:

$$
G _ {i j} := \cos (\hat {\mu} _ {i}, \hat {\mu} _ {j}) = \hat {\mu} _ {i} ^ {\top} \hat {\mu} _ {j}. \tag {44}
$$

Here G is a real symmetric positive semi-definite (PSD) matrix, so for any vector $z \in \mathcal { R } ^ { n }$ , it satisfies:

$$
z ^ {\top} G z = \left\| \sum_ {i} z _ {i} \hat {\mu} _ {i} \right\| _ {2} ^ {2} \geq 0, \tag {45}
$$

where the $G _ { i i } ~ = ~ 1$ . According to the properties of real symmetric PSD matrix, there exist an orthogonal matrix $\boldsymbol { \mathcal { U } } \in \mathcal { R } ^ { n \times n }$ and a diagonal matrix $\Lambda = d i a \bar { g } ( \lambda _ { 1 } , \ldots \lambda _ { n } ) ( \lambda _ { k } \geq 0 )$ such that:

$$
G = \mathcal {U} \Lambda \mathcal {U} ^ {\top}, \tag {46}
$$

where $r = r a n k ( G ) \leq n$ . Take the non-zero eigenvalue part of Λ as $\Lambda _ { r } \in \mathcal { R } ^ { r \times r }$ , and take the first r columns of the corresponding eigenvectors to obtain $\bar { \mathcal { U } _ { r } } \in \mathcal { R } ^ { n \times r }$ :

$$
G = \mathcal {U} _ {r} \Lambda_ {r} \mathcal {U} _ {r} ^ {\top}, \tag {47}
$$

$$
Y := \mathcal {U} _ {r} \Lambda_ {r} ^ {1 / 2} \in \mathcal {R} ^ {n \times r}, \tag {48}
$$

where we write Y in row-vector form, and denote its i-th row as $y _ { i } ^ { \top } \left( y _ { i } \in { \mathcal { R } } ^ { r } \right)$ :

$$
(Y Y ^ {\top}) _ {i j} = y _ {i} ^ {\top} y _ {j} = (\mathcal {U} _ {r} \Lambda_ {r} ^ {1 / 2}) (\mathcal {U} _ {r} \Lambda_ {r} ^ {1 / 2}) _ {i j} ^ {\top} = \mathcal {U} _ {r} \Lambda_ {r} \mathcal {U} _ {r i j} ^ {\top} = G _ {i j}. \tag {49}
$$

Therefore, we have constructed n vectors $y _ { 1 } , \ldots , y _ { n } \in { \mathcal { R } } ^ { r }$ that satisfy the inner-product relation $y _ { i } ^ { \top } y _ { j } = G _ { i j } \colon$

$$
\left\| y _ {i} \right\| _ {2} ^ {2} = y _ {i} ^ {\top} y _ {i} = G _ {i i} = 1, \tag {50}
$$

$$
c o s (y _ {i}, y _ {j}) = \frac {y _ {i} ^ {\top} y _ {j}}{\| y _ {i} \| \| y _ {j} \|} = \frac {G _ {i j}}{1 \cdot 1} = c o s (\hat {\mu} _ {i}, \hat {\mu} _ {j}), \tag {51}
$$

$$
\cos (y _ {i}, y _ {j}) = \cos (\hat {\mu} _ {i}, \hat {\mu} _ {j}) = \cos (\mu_ {i}, \mu_ {j}). \tag {52}
$$

By setting $f ( \mu _ { i } ) : = y _ { i } ( i = 1 , \dots , n )$ , we obtain:

$$
\cos (f (\mu_ {i}), f (\mu_ {j})) = \cos (\mu_ {i}, \mu_ {j}) \forall i, j, \tag {53}
$$

$$
d _ {s} (f (\mu_ {i}; \Theta_ {f}), f (\mu_ {j}; \Theta_ {f})) \propto d _ {s} (\mu_ {i}, \mu_ {j}). \tag {54}
$$

# E PROOF: $\mathcal { L } _ { c m } \left( \mathrm { E Q . ~ } ( 9 ) \right)$ ENCOURAGES f TO APPROACH THE ISOMETRY ASSUMPTION (EQ. (8)).

Given the pre-trained word embeddings $w _ { i }$ and the predicted word embeddings $w _ { i } ^ { \prime } = f ( \mu _ { i } ; \Theta _ { f } )$ , we normalize them to obtain:

$$
\hat {w} _ {i} ^ {\prime} := \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \|}, \quad \hat {w} _ {i} := \frac {w _ {i}}{\| w _ {i} \|}. \tag {55}
$$

Similarly, we normalize the region representations to obtain $\begin{array} { r } { \hat { \mu } _ { i } : = \frac { \mu _ { i } } { \parallel \mu _ { i } \parallel } . \mathcal { L } _ { c m } } \end{array}$ consists of $\mathcal { L } _ { w o r d }$ and $\mathcal { L } _ { s t r u c t } \left( \mathbf { A } _ { } \right.$ ppendix F). For a given batch size B, there exist constants $\epsilon _ { w } > 0$ and $\epsilon _ { s } > 0$ such that, when $\mathcal { L } _ { w o r d } \leq \epsilon _ { w }$ and $\mathcal { L } _ { s t r u c t } \leq \epsilon _ { s }$ , the following inequality holds:

$$
\left| \cos (\hat {w} _ {i} ^ {\prime}, \hat {w} _ {j} ^ {\prime}) - \cos (\mu_ {i}, \mu_ {j}) \right| \leq 4 \sqrt {2 \epsilon_ {w}} + \sqrt {\frac {\epsilon_ {s}}{M}}, \tag {56}
$$

where M denotes the number of ordered pairs $( i \neq j )$ , and $i , j \in B$ . When $\epsilon _ { w }  0$ and $\epsilon _ { s }  0$ , $c o s ( \hat { w } _ { i } ^ { \prime } , \hat { w } _ { j } ^ { \prime } ) - c o s ( \mu _ { i } , \mu _ { j } )$ admits an upper bound arbitrarily close to 0, meaning that the cosine similarity between the mapped vector pairs approaches that in the original µ-space, thereby achieving $d _ { s } ( f ( \mu _ { i } ; \Theta _ { f } ) , f ( \mu _ { j } ; \bar { \Theta } _ { f } ) ) \propto d _ { s } ( \bar { \mu } _ { i } , \mu _ { j } )$ .

We first convert the point-wise cosine error into the Euclidean difference between vectors. According to Appendix C, we have:

$$
\left\| \hat {w} _ {i} ^ {\prime} - \hat {w} _ {i} \right\| _ {2} ^ {2} = 2 (1 - \cos (\hat {w} _ {i} ^ {\prime}, \hat {w} _ {i})). \tag {57}
$$

For each case where $i \in B$ , let $\delta _ { i } : = 1 - c o s ( \hat { w } _ { i } ^ { \prime } , \hat { w } _ { i } ) \geq 0$ . Then:

$$
\left\| \hat {w} _ {i} ^ {\prime} - \hat {w} _ {i} \right\| _ {2} = \sqrt {2 \delta_ {i}}, \tag {58}
$$

$$
\sum_ {i \in B} \delta_ {i} = \mathcal {L} _ {\text { word }} \leq \epsilon_ {w}, \tag {59}
$$

$$
\delta_ {i} \leq \epsilon_ {w} \Rightarrow \| \hat {w} _ {i} ^ {\prime} - \hat {w} _ {i} \| _ {2} \leq \sqrt {2 \epsilon_ {w}}. \tag {60}
$$

To quantify how point-wise alignment errors influence the pairwise cosine discrepancies, we characterize the variation of individual entries in the cosine similarity matrix using the norm difference. Then:

$$
\left| \cos (\hat {w} _ {i} ^ {\prime}, \hat {w} _ {j} ^ {\prime}) - \cos (\hat {w} _ {i}, \hat {w} _ {j}) \right| = \left| \hat {w} _ {i} ^ {\prime} \cdot \hat {w} _ {j} ^ {\prime} - \hat {w} _ {i} \cdot \hat {w} _ {j} \right|, \tag {61}
$$

$$
\begin{array}{l} \left| \hat {w} _ {i} ^ {\prime} \cdot \hat {w} _ {j} ^ {\prime} - \hat {w} _ {i} \cdot \hat {w} _ {j} \right| \\ = \left| \left(\hat {w} _ {i} ^ {\prime} - \hat {w} _ {i}\right) \cdot \hat {w} _ {j} ^ {\prime} + \hat {w} _ {i} \cdot \left(\hat {w} _ {j} ^ {\prime} - \hat {w} _ {j}\right) \right| \tag {62} \\ \leq \left\| \hat {w} _ {i} ^ {\prime} - \hat {w} _ {i} \right\| _ {2} \cdot \left\| \hat {w} _ {j} ^ {\prime} \right\| _ {2} + \left\| \hat {w} _ {i} \right\| _ {2} \cdot \left\| \hat {w} _ {j} ^ {\prime} - \hat {w} _ {j} \right\| _ {2} \\ \leq \left\| \hat {w} _ {i} ^ {\prime} - \hat {w} _ {i} \right\| _ {2} + \left\| \hat {w} _ {j} ^ {\prime} - \hat {w} _ {j} \right\| _ {2} \\ \end{array}
$$

$$
\left| \hat {w} _ {i} ^ {\prime} \cdot \hat {w} _ {j} ^ {\prime} - \hat {w} _ {i} \cdot \hat {w} _ {j} \right| \leq \sqrt {2 \epsilon_ {w}} + \sqrt {2 \epsilon_ {w}} = 2 \sqrt {2 \epsilon_ {w}}. \tag {63}
$$

To quantify the influence of the structural loss $\mathcal { L } _ { s t r u c t }$ on the reconstruction error, we convert the structural loss into an upper bound on the per-pair error. Let the per-pair error be defined as $e _ { i j } : =$ $c o s ( \hat { w } _ { i } ^ { \prime } , \hat { w } _ { j } ^ { \prime } ) - c o s ( \mu _ { i } , \hat { \mu } _ { j } ^ { \ast } )$ . Then we obtain:

$$
\sum_ {i \neq j} e _ {i j} ^ {2} = \mathcal {L} _ {\text { struct }} \leq \epsilon_ {s}, \tag {64}
$$

$$
\max _ {i \neq j} | e _ {i j} | \leq \sqrt {\sum_ {i \neq j} e _ {i j} ^ {2}} \leq \sqrt {\epsilon_ {s}}. \tag {65}
$$

If we distribute $\epsilon _ { s }$ uniformly across all pairs, then the absolute error of each pair is bounded above by $\epsilon _ { s } / M$ . Then:

$$
\begin{array}{l} \left| \cos (\hat {w} _ {i} ^ {\prime}, \hat {w} _ {j} ^ {\prime}) - \cos (\mu_ {i}, \mu_ {j}) \right| \\ = \left| c o s (\hat {w} _ {i} ^ {\prime}, \hat {w} _ {j} ^ {\prime}) - c o s (\hat {w} _ {i}, \hat {w} _ {j}) + c o s (\hat {w} _ {i}, \hat {w} _ {j}) - c o s (\mu_ {i}, \mu_ {j}) \right| \\ \leq \left| \cos \left(\hat {w} _ {i} ^ {\prime}, \hat {w} _ {j} ^ {\prime}\right) - \cos \left(\hat {w} _ {i}, \hat {w} _ {j}\right) \right| + \left| \cos \left(\hat {w} _ {i}, \hat {w} _ {j}\right) - \cos \left(\mu_ {i}, \mu_ {j}\right) \right| \tag {66} \\ \leq 2 \sqrt {2 \epsilon_ {w}} + \left| c o s (\hat {w} _ {i}, \hat {w} _ {j}) - c o s (\hat {w} _ {i} ^ {\prime}, \hat {w} _ {j} ^ {\prime}) + c o s (\hat {w} _ {i} ^ {\prime}, \hat {w} _ {j} ^ {\prime}) - c o s (\mu_ {i}, \mu_ {j}) \right| \\ \leq 4 \sqrt {2 \epsilon_ {w}} + \sqrt {\frac {\epsilon_ {s}}{M}} \\ \end{array}
$$

Therefore, by enforcing $\epsilon _ { w } \to 0$ and $\epsilon _ { s }  0$ during training, we obtain the following expression:

$$
\left| \cos (\hat {w} _ {i} ^ {\prime}, \hat {w} _ {j} ^ {\prime}) - \cos (\mu_ {i}, \mu_ {j}) \right|\rightarrow 0. \tag {67}
$$

$$
d _ {s} (f (\mu_ {i}; \Theta_ {f}), f (\mu_ {j}; \Theta_ {f})) \propto d _ {s} (\mu_ {i}, \mu_ {j}). \tag {68}
$$

# F PROOF: THE $w ^ { \prime }$ IN EQ. (9) CANNOT BE REPLACED BY w.

The $w ^ { \prime }$ in Eq. (9) cannot be replaced by w. Doing so would remove the “structure-preservation” supervision imposed on the MT M model $f ,$ as the second term in Eq. (9) would no longer contribute a meaningful loss for trainingstructure (the second term of Eq.(9)) $f .$ The loss f defined as recommended in the article for preserving: $\mathcal { L } _ { s t r u c t } ^ { A }$

$$
\mathcal {L} _ {\text { s   t   r   u   c   t }} ^ {A} = \mathbb {E} \left[ \left(\cos \left(\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}, \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}\right) - \cos \left(\frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}}, \frac {\mu_ {j}}{\| \mu_ {j} \| _ {2}}\right)\right) ^ {2} \right]. \tag {69}
$$

$w ^ { \prime }$ $\mathcal { L } _ { s t r u c t } ^ { A }$ $\mathcal { L } _ { s t r u c t } ^ { B } \mathrm { : }$

$$
\mathcal {L} _ {\text {struct}} ^ {B} = \mathbb {E} [ (\cos (\frac {w _ {i}}{\| w _ {i} \| _ {2}}, \frac {w _ {j}}{\| w _ {j} \| _ {2}}) - \cos (\frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}}, \frac {\mu_ {j}}{\| \mu_ {j} \| _ {2}})) ^ {2} ]. \tag {70}
$$

Next, we compare the constraint capabilities of $\mathcal { L } _ { s t r u c t } ^ { A }$ and $\mathcal { L } _ { s t r u c t } ^ { B }$ on the MT M model f (with parameters $\Theta _ { f } )$ .

# F.1 FROM THE PERSPECTIVE OF THE GRADIENT OF THE LOSS FUNCTION

For the loss function $\mathcal { L } _ { s t r u c t } ^ { A }$ , we apply the chain rule to $\Theta _ { f }$ as follows:

$$
\frac {\partial \mathcal {L} _ {\text { s   t   r   u   c   t }} ^ {A}}{\partial \Theta_ {f}} = 2 \mathbb {E} \left(\cos \left(\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}}, \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}\right) - \cos \left(\frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}}, \frac {\mu_ {j}}{\| \mu_ {j} \| _ {2}}\right)\right) \cdot \frac {\partial \cos \left(\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}} , \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}}\right)}{\partial \Theta_ {f}}. \tag {71}
$$

$$
\frac {\partial \cos (\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}} , \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}})}{\partial \Theta_ {f}} = \frac {\partial \cos (\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}} , \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}})}{\partial w _ {i} ^ {\prime}} \cdot \frac {\partial w _ {i} ^ {\prime}}{\partial \Theta} + \frac {\partial \cos (\frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}} , \frac {w _ {j} ^ {\prime}}{\| w _ {j} ^ {\prime} \| _ {2}})}{\partial w _ {j} ^ {\prime}} \cdot \frac {\partial w _ {j} ^ {\prime}}{\partial \Theta}. (7 2)
$$

As long as the parameters of $f$ influence $\begin{array} { r } { w _ { i } ^ { \prime } \left( \mathrm { i . e . , } \frac { \partial w _ { i } ^ { \prime } } { \partial \Theta } \neq 0 \right) } \end{array}$ , the above equation generally does not vanish. Consequently, $\mathcal { L } _ { s t r u c t } ^ { A }$ ∂Θ produces a nonzero gradient that drives the update of $\Theta _ { f }$ , forcing the mapping f to bring cos( $f$ $\big ( \frac { \boldsymbol { w } _ { i } ^ { \prime } } { \left\| \boldsymbol { w } _ { i } ^ { \prime } \right\| _ { 2 } } , \frac { \boldsymbol { w } _ { j } ^ { \prime } } { \left\| \boldsymbol { w } _ { j } ^ { \prime } \right\| _ { 2 } } \big )$ closer to $\cos \big ( \frac { \mu _ { i } } { \left. \mu _ { i } \right. _ { 2 } } , \frac { \mu _ { j } } { \left. \mu _ { j } \right. _ { 2 } } \big )$ . Therefore, $\mathcal { L } _ { s t r u c t } ^ { A }$ directly imposes constraints on the MT M model ${ \bf \widehat { f } } .$

Moreover, the cos( wikwik , kwjk ) $\cos \big ( \frac { w _ { i } } { \left. w _ { i } \right. _ { 2 } } , \frac { w _ { j } } { \left. w _ { j } \right. _ { 2 } } \big )$ wj in $\mathcal { L } _ { s t r u c t } ^ { B }$ is a constant. We apply the chain rule to $\Theta _ { f }$ as follows:

$$
\mathcal {L} _ {\text { s   t   r   u   c   t }} ^ {B} = \mathbb {E} [ (\underbrace {\cos \left(\frac {w _ {i}}{\| w _ {i} \| _ {2}} , \frac {w _ {j}}{\| w _ {j} \| _ {2}}\right)} _ {\text { c   o   n   s   t   a   n   t }} - \cos \left(\frac {\mu_ {i}}{\| \mu_ {i} \| _ {2}}, \frac {\mu_ {j}}{\| \mu_ {j} \| _ {2}}\right)) ^ {2} ]. \tag {73}
$$

$$
\frac {\partial \mathcal {L} _ {s t r u c t} ^ {B}}{\partial \Theta_ {f}} = 0. \tag {74}
$$

$\mathcal { L } _ { s t r u c t } ^ { B }$ provides no constraints or gradient information on to preserve input similarity in any manner. Altho $f ,$ h d therefore cannot compel the map-is affected, this influence does not $\mu$ propagate to $\Theta _ { f }$ , making it impossible to fulfill the objective of structural preservation.

# F.2 FROM THE PERSPECTIVE OF THE JOINT OPTIMIZATION OBJECTIVE

The loss function recommended in the article for word alignment (the first term of $\operatorname { E q . } ( 9 ) )$ is defined as $\mathcal { L } _ { w o r d } .$

$$
\mathcal {L} _ {\text { word }} = \mathbb {E} [ (1 - \cos (\frac {w _ {i}}{\| w _ {i} \| _ {2}}, \frac {w _ {i} ^ {\prime}}{\| w _ {i} ^ {\prime} \| _ {2}})) ]. \tag {75}
$$

Next, we compared the differences between the two loss functions when jointly optimized with $\mathcal { L } _ { w o r d }$ . The joint loss function for $\mathcal { L } _ { w o r d }$ and $\mathcal { L } _ { s t r u c t } ^ { A }$ is expressed as follows:

$$
\mathcal {L} _ {c m} ^ {A} = \mathcal {L} _ {\text { word }} + \mathcal {L} _ {\text { struct }} ^ {A}. \tag {76}
$$

Both loss terms involve $\Theta _ { f }$ (through $w _ { i } ^ { \prime } )$ and therefore jointly constrain the mapping. Specifically, $\mathcal { L } _ { w o r d }$ pulls each single-point mapping $w _ { i } ^ { \prime }$ toward its corresponding $w _ { i }$ , while $\mathcal { L } _ { s t r u c t } ^ { A }$ enforces that f preserve the pairwise similarity structure among samples, consistent with the semantic relationships in the µ-space (prototype). The joint effect of these two losses ensures that the learned mapping f achieves both accurate pointwise alignment and global structural preservation.

$\mathcal { L } _ { w o r d }$ $\mathcal { L } _ { s t r u c t } ^ { B }$

$$
\mathcal {L} _ {c m} ^ {B} = \mathcal {L} _ {\text { word }} + \mathcal {L} _ {\text { struct }} ^ {B}. \tag {77}
$$

It is worth noting that $\mathcal { L } _ { s t r u c t } ^ { B }$ does not impose any constraints on the model $f .$ Consequently, $\mathcal { L } _ { w o r d }$ is the only term that directly supervises $f ,$ and minimizing it merely pulls $w _ { i } ^ { \prime }$ toward its corresponding $w _ { i } .$ . In contrast, µ $\mathcal { L } _ { s t r u c t } ^ { B }$ adjusts the µ-space (governed by the parameters of the w w $P A E$ model) to make $\begin{array} { r } { c o s \big ( \frac { \mu _ { i } } { \lVert \mu _ { i } \rVert _ { 2 } } , \frac { \mu _ { j } } { \lVert \mu _ { j } \rVert _ { 2 } } \big ) } \end{array}$ approach cos( ikwik2 , kwj k2 ) ${ \dot { \big ( } } { \frac { \dot { w _ { i } } } { \lVert w _ { i } \rVert _ { 2 } } } , { \frac { \mathbf { \tilde { w } } _ { j } } { \lVert w _ { j } \rVert _ { 2 } } } { \big ) }$ j , but it provides no mechanism to enforce structural preservation in the mapping produced by $f .$ .

# G PROOF: THE PROTOTYPE CONSISTENCY CONTRASTIVE LEARNING LOSS $\mathcal { L } _ { c l }$ ENHANCES THE DISCRIMINABILITY OF REGION REPRESENTATIONS.

The prototype consistency contrastive learning loss enhances the discriminability of region representations by reducing intra-word variance and increasing inter-word separation. Intra-word variance $\sigma _ { k } ^ { 2 }$ and inter-word separation $D _ { k , k ^ { \prime } }$ can be defined as follows:

$$
\sigma_ {k} ^ {2} = \frac {1}{J _ {k}} \sum_ {j = 1} ^ {J _ {k}} | | \mu_ {j} - v _ {k} | | ^ {2}, \tag {78}
$$

$$
D _ {k, k ^ {\prime}} := \left| \left| v _ {k} - v _ {k ^ {\prime}} \right| \right| ^ {2}. \tag {79}
$$

For any µj : $\mu _ { j } \colon$

$$
\left| \left| \mu_ {j} - v _ {k} \right| \right| ^ {2} = \left| \left| \mu_ {j} - \bar {\mu} + \bar {\mu} - v _ {k} \right| \right| ^ {2} = \left| \left| \mu_ {j} - \bar {\mu} \right| \right| ^ {2} + \left| \left| \bar {\mu} - v _ {k} \right| \right| ^ {2} + 2 \left\langle \mu_ {j} - \bar {\mu}, \bar {\mu} - v _ {k} \right\rangle , \tag {80}
$$

$$
\sigma_ {k} ^ {2} \approx \frac {1}{J _ {k}} \sum_ {j = 1} ^ {J _ {k}} | | \mu_ {j} - \bar {\mu} | | ^ {2} + | | \bar {\mu} - v _ {k} | | ^ {2}, \tag {81}
$$

where $\bar { \mu }$ serves as the temporary mean for a given batch. The intra-word variance can be decomposed into the within-batch intra-word variance and the deviation between the batch-specific word centers and the prototypes. The loss $\mathcal { L } _ { c l }$ is updated along the gradient direction:

$$
\mu_ {j} \longleftarrow \mu_ {j} - \eta (\mu_ {j} - v _ {k}). \tag {82}
$$

$$
\mu_ {j} (t + 1) = (1 - \eta) \mu_ {j} (t) + \eta v _ {k}. \tag {83}
$$

$$
\sigma_ {k} ^ {2} (t + 1) = (1 - \eta) ^ {2} \sigma_ {k} ^ {2} (t). \tag {84}
$$

As iterations progress, $\sigma _ { k } ^ { 2 }$ gradually approaches zero, causing samples within the same word to cluster tightly around their prototypes and thereby enhancing the discriminability of the region representations. The loss $\mathcal { L } _ { c l }$ is updated by moving along the gradient direction while simultaneously being repelled away from other prototypes:

$$
\mu_ {j} \longleftarrow \mu_ {j} + \eta \sum_ {k \neq k ^ {\prime}} (\mu_ {j} - v _ {k ^ {\prime}}). \tag {85}
$$

$$
v _ {k} ^ {\prime} = \frac {1}{J _ {k}} \sum_ {j = 1} ^ {J _ {k}} \mu_ {j} (t + 1) = v _ {k} + \eta \sum_ {k \neq k ^ {\prime}} (v _ {k} - v _ {k ^ {\prime}}). \tag {86}
$$

$$
D _ {k, k ^ {\prime}} (t + 1) = \left\| v _ {k} ^ {\prime} - v _ {k ^ {\prime}} ^ {\prime} \right\| ^ {2} = \left\| v _ {k} - v _ {k ^ {\prime}} + \eta \left(\sum_ {k \neq \hat {k}} \left(v _ {k} - v _ {\hat {k}}\right) - \sum_ {k ^ {\prime} \neq \hat {k}} \left(v _ {k ^ {\prime}} - v _ {\hat {k}}\right) \right\| ^ {2} \right. \tag {87}
$$

At convergence, each prototype is positioned as far as possible from all others, resulting in distinct separation between different prototypes and further enhancing the discriminability of the region representations.

# H FINE-TUNED DOMAIN KNOWLEDGE

To better adapt the multimodal aligned semantic knowledge to specific datasets, we can further finetune the prototypical region representations to obtain domain-specific knowledge. This fine-tuning step is optional and depends on the availability of unpaired data in the target dataset. Notably, the multimodal aligned semantic knowledge alone can also be applied directly and achieves strong performance. Given that annotating paired image-text data is costly, while unpaired images and texts are typically more accessible, we adopt a bidirectional region-word cycle-consistent learning approach in an unpaired learning setting.

In particular, given a batch of unpaired images and texts, we first obtain a set of raw region representations $\boldsymbol { \breve { R } } = \{ r _ { n } \} _ { n = 1 , \dots , I } ( \boldsymbol { \breve { R } } \in \mathcal { R } ^ { I \times M } )$ using the pre-trained Faster-RCNN above and a set of parsed words through tokenization operation implemented via NLTK. Then, we use the knowledge as a cross-modal bridge to represent all the words into the corresponding prototypical region representations $u = \{ v _ { j } \} _ { j = 1 , \ldots , J } \ ( u \in \mathcal { R } ^ { J \times Z } )$ . For the set of regions $R ,$ we extract region representations $\mu \in \mathcal { R } ^ { I \times Z }$ by utilizing the $P A E$ model h.

We utilize a bidirectional region-word cycle-consistent loss to learn a parametric transformation matrix $W \in \mathcal { R } ^ { Z \times Z \ 3 }$ . This loss incorporates two cross-modal similarity measurement processes: region-to-word (R2W) and word-to-region (W2R). In the R2W process, similarities between each word and all regions are first computed, and these similarities are then used as weights to aggregate all regions into a reconstructed word representation. Conversely, in the W2R process, similarities between each region and all words are computed to reconstruct regions from the word representations. The corresponding formulations are

$$
S = u W (\mu W) ^ {\top}, \tag {88}
$$

$$
\hat {u} = \text { softmax } (S) \mu W, \hat {\mu} = \text { softmax } (S ^ {\top}) u W, \tag {89}
$$

where $S \in \mathcal { R } ^ { J \times I }$ is the similarity matrix between transformed word and region representations, uˆ $\in$ $\mathcal { R } ^ { J \times Z }$ contains the reconstructed word representations from region representations, and $\hat { \mu } \in \mathcal { R } ^ { I \times Z }$ contains the reconstructed region representations from word representations.

Each original word (or region) representation is then compared with its reconstructed counterpart to determine whether they correspond to the same entity. By minimizing the cross-entropy between the predicted and ground-truth labels, we obtain a self-supervised loss, $\mathcal { L } _ { s s }$ , which is used to optimize ${ \bf \ddot { \cal W } } ;$ :

$$
\hat {Y} ^ {R 2 W} = \text { softmax } (u W \hat {u} ^ {\top}) ^ {\top}, \hat {Y} ^ {W 2 R} = \text { softmax } (\mu W \hat {\mu} ^ {\top}) ^ {\top}, \tag {90}
$$

$$
\mathcal {L} ^ {R 2 W} = - \sum_ {j = 1} ^ {J} y _ {j} ^ {\top} \log (\hat {y} _ {j}), \tag {91}
$$

$$
\mathcal {L} ^ {W 2 R} = - \sum_ {n = 1} ^ {I} y _ {n} ^ {\top} \log (\hat {y} _ {n}), \tag {92}
$$

$$
\mathcal {L} ^ {s s} = \mathcal {L} ^ {R 2 W} + \mathcal {L} ^ {W 2 R}, \tag {93}
$$

where $\hat { Y } ^ { R 2 W } \in \mathcal { R } ^ { J \times J }$ and $\hat { Y } ^ { W 2 R } \in \mathcal { R } ^ { I \times I }$ are two matrices including the predicted labels in R2W and W2R directions, respectively. In $\hat { Y } ^ { R 2 W }$ and $\hat { Y } ^ { W 2 R }$ , the $j \mathrm { - t h }$ and n-th columns are denoted as $\hat { y } _ { j }$ and ${ \hat { y } } _ { n } .$ , respectively. $y _ { j }$ and $y _ { n }$ are two groundtruth label vectors, whose the j-th and n-th values are ones and the rest are zeros. After the cycle consistent learning, we can use the learnable $W$ to transform all prototypical region representations into the fine-tuned domain knowledge, denoted as $\{ ( w _ { k } , \hat { v } _ { k } ) \} _ { k = 1 , \dots , K }$ , where $\hat { v } _ { k } = v _ { k } W \in \mathcal { R } ^ { Z }$ .

# I EXPERIMENTAL SETTING

# I.1 DATASET

The details of experimental datasets and metrics are described as follows.

Flickr30K (Young et al., 2014) consists of 31,783 images collected from the Flickr website. Each image has 5 human annotated texts. We use the public training, validation and testing splits, which contain 29,783, 1,000 and 1,000 images, respectively.

MSCOCO (Lin et al., 2014) consists of 123,287 images, each of which is associated with 5 texts. We use the public training, validation and testing splits, with 113,287, 5,000 and 5,000 images, respectively.

# I.2 IMPLEMENTATION DETAILS

In the multimodal aligned semantic knowledge, we collect all words from the VG dataset and filter out some special characters and rare words, resulting in a total of K=12,385 semantic concepts. For each image, we initially employ the pre-trained object detection model Faster-RCNN 4 to extract raw region representations, setting the number of detected regions to I=36 and the dimensionality of each region representation to M=2048. For each word, we obtain its word embedding using the pre-trained word vectors glove-840B-300d 5. The batch size is 4096 for the first 200 epochs and 2048 for the next 200 epochs. The trade-off factors $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are set to 3. The sampling size m is set to 10. We use the Adam to optimize the loss with a learning rate of 1e-4.

# I.3 PRETRAINED MODELS AND WORD VECTORS

Faster-RCNN is a widely used deep learning model for object detection, tasked with both identifying and localizing objects within an image. Building on earlier approaches such as R-CNN and Fast R-CNN, it introduces a Region Proposal Network (RPN) that generates object proposals directly within the model. This integration significantly improves both speed and accuracy, enabling efficient, real-time detection of multiple objects with high precision. We use Detectron2 as the backend to support comprehensive functions, including training, testing, and feature extraction. Additionally, we migrate the pre-trained Caffe-based model from the original repository, ensuring that it extracts visual features consistent with the original model, with deviations of less than 0.01.

GloVe is an unsupervised learning algorithm designed to generate vector representations of words. It is trained on aggregated global word-word co-occurrence statistics from a corpus, producing embeddings that capture meaningful semantic relationships and exhibit interpretable linear substructures within the word vector space. We obtain word embedding using the pre-trained word vectors glove-840B-300d. It is a set of pre-trained word embeddings derived from the GloVe model developed by Stanford University, trained specifically on Common Crawl data. The model is built using approximately 840B tokens, resulting in a vocabulary of 2.2 million words and producing 300-dimensional word vectors.

CLIP 6 is a cross-modal pre-trained model proposed by OpenAI, designed to learn a shared semantic embedding space for images and text. It is trained on large-scale natural image-text pairs, mapping images and text into the same-dimensional vector space using an image encoder and a text encoder. Contrastive learning is employed to pull corresponding image-text pairs closer in the embedding space while pushing non-corresponding pairs farther apart.

ALBEF 7 is a vision–language pre-trained model proposed by Salesforce, designed to enhance cross-modal semantic representations through an “align before fuse” strategy. Its core idea is to first align image and text features in a shared space via image–text contrastive learning, and then fuse them using a multimodal encoder to capture richer cross-modal interactions. Additionally, AL-BEF incorporates a momentum distillation mechanism, where a continuously updated momentum model generates pseudo-labels to improve training robustness. The model demonstrates strong performance on tasks such as image–text retrieval, visual question answering, and natural language visual reasoning, making it a key approach in the vision–language pretraining field.

# I.4 MODEL DETAILS

In this section, we present the architectures of the models involved in the proposed MASK framework, as shown in Table 8. The [B, M, 2048] indicates that a batch contains B images, each image is divided into M regions, and each region has a feature dimension of 2048. In our experiments, we set B = 128 and $\bar { M } = 3 6$ . It is worth noting that the model architectures are not fixed. In later sections, we will discuss how model size impacts the accuracy of image-text matching.

Table 8: Overview of the model architectures integrated into the MASK framework. 

<table><tr><td>Model</td><td>Layer</td><td>Input</td><td>Output</td><td>Params</td></tr><tr><td rowspan="6">PAE</td><td>FC</td><td> $[B, M, 2048]$ </td><td> $[B, M, 512]$ </td><td>1.05M</td></tr><tr><td>ReLU</td><td> $[B, M, 512]$ </td><td> $[B, M, 512]$ </td><td>0</td></tr><tr><td>Self-Attention</td><td> $[B, M, 512]$ </td><td> $[B, M, 512]$ </td><td>1.05M</td></tr><tr><td>Self-Attention</td><td> $[B, M, 512]$ </td><td> $[B, M, 512]$ </td><td>1.05M</td></tr><tr><td>Self-Attention</td><td> $[B, M, 512]$ </td><td> $[B, M, 512]$ </td><td>1.05M</td></tr><tr><td>Normalization</td><td> $[B, M, 512]$ </td><td> $[B, M, 512]$ </td><td>1024</td></tr><tr><td rowspan="4">FRM</td><td>FC</td><td> $[B, M, 512]$ </td><td> $[B, M, 512]$ </td><td>0.26M</td></tr><tr><td>ReLU</td><td> $[B, M, 512]$ </td><td> $[B, M, 512]$ </td><td>0</td></tr><tr><td>Self-Attention</td><td> $[B, M, 512]$ </td><td> $[B, M, 512]$ </td><td>1.05M</td></tr><tr><td>FC</td><td> $[B, M, 512]$ </td><td> $[B, M, 2048]$ </td><td>1.05M</td></tr><tr><td rowspan="8">MTM</td><td>FC</td><td> $[B, M, 512]$ </td><td> $[B, M, 300]$ </td><td>0.15M</td></tr><tr><td>ReLU</td><td> $[B, M, 300]$ </td><td> $[B, M, 300]$ </td><td>0</td></tr><tr><td>Self-Attention</td><td> $[B, M, 300]$ </td><td> $[B, M, 300]$ </td><td>0.37M</td></tr><tr><td>Self-Attention</td><td> $[B, M, 300]$ </td><td> $[B, M, 300]$ </td><td>0.37M</td></tr><tr><td>FC</td><td> $[B, M, 300]$ </td><td> $[B, M, 300]$ </td><td>0.09M</td></tr><tr><td>ReLU</td><td> $[B, M, 300]$ </td><td> $[B, M, 300]$ </td><td>0</td></tr><tr><td>Self-Attention</td><td> $[B, M, 300]$ </td><td> $[B, M, 300]$ </td><td>0.37M</td></tr><tr><td>FC</td><td> $[B, M, 300]$ </td><td> $[B, M, 300]$ </td><td>0.09M</td></tr><tr><td>Total</td><td></td><td></td><td></td><td>8.1M</td></tr></table>

# J ADDITIONAL EXPERIMENTS

# J.1 CROSS-DATASET IMAGE-TEXT MATCHING

Our proposed MASK can also enhance the generalization of conventional image-text matching models when applied to unseen datasets. Specifically, we try to re-rank conventional image-text matching models for the task of cross-dataset image-text matching. The experimental setup is as follows: (1) two representative image-text matching models (i.e., VSRN (Radford et al., 2021) and SAEM (Wu et al., 2019)) are trained on a source dataset (e.g., Flickr30K or MSCOCO), (2) these models are then evaluated on a different target dataset (e.g., MSCOCO or Flickr30K), and (3) using the proposed MASK to re-rank these models on the target dataset. It is important to note that neither the re-ranking methods nor the base models are trained on the target dataset.

Table 9: Cross-dataset image-text matching by re-ranking existing models on the Flickr30K and MSCOCO Datasets. 

<table><tr><td rowspan="3">Method</td><td colspan="7">MSCOCO → Flickr30K</td><td colspan="7">Flickr30K → MSCOCO</td></tr><tr><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td> $VSRN$ </td><td>42.3</td><td>69.3</td><td>78.1</td><td>53.1</td><td>79.5</td><td>87.5</td><td>409.9</td><td>14.0</td><td>31.7</td><td>42.2</td><td>20.4</td><td>40.0</td><td>50.0</td><td>198.4</td></tr><tr><td> $VSRN + MACK_{2022}$ </td><td>42.6</td><td>69.8</td><td>78.5</td><td>53.3</td><td>79.7</td><td>87.7</td><td>411.7</td><td>14.4</td><td>32.6</td><td>43.1</td><td>20.5</td><td>40.5</td><td>50.2</td><td>201.4</td></tr><tr><td> $VSRN + LeaPRR_{2023}$ </td><td>42.9</td><td>69.6</td><td>79.8</td><td>55.2</td><td>79.3</td><td>88.1</td><td>414.9</td><td>16.2</td><td>31.9</td><td>43.8</td><td>20.9</td><td>40.2</td><td>51.4</td><td>204.4</td></tr><tr><td> $VSRN + MACK^{VG-M}_{2024}$ </td><td>44.8</td><td>71.3</td><td>79.5</td><td>56.7</td><td>81.7</td><td>89.0</td><td>422.9</td><td>16.6</td><td>35.8</td><td>45.0</td><td>23.3</td><td>43.3</td><td>52.0</td><td>215.9</td></tr><tr><td> $VSRN + FR_{2025}$ </td><td>46.4</td><td>73.4</td><td>80.1</td><td>56.8</td><td>83.3</td><td>89.1</td><td>429.1</td><td>16.7</td><td>37.1</td><td>44.4</td><td>24.2</td><td>44.5</td><td>56.8</td><td>223.7</td></tr><tr><td> $VSRN + MASK$ </td><td>47.3</td><td>73.7</td><td>83.9</td><td>57.9</td><td>83.6</td><td>89.9</td><td>436.3</td><td>17.6</td><td>38.7</td><td>45.9</td><td>26.7</td><td>44.8</td><td>58.7</td><td>232.4</td></tr><tr><td>SAEM</td><td>41.4</td><td>70.2</td><td>80.0</td><td>53.4</td><td>80.9</td><td>89.6</td><td>415.5</td><td>14.8</td><td>34.0</td><td>45.0</td><td>23.2</td><td>45.4</td><td>57.4</td><td>219.8</td></tr><tr><td> $SAEM + MACK_{2022}$ </td><td>41.8</td><td>70.7</td><td>80.0</td><td>54.2</td><td>81.2</td><td>89.9</td><td>417.9</td><td>15.4</td><td>34.9</td><td>45.9</td><td>23.6</td><td>46.0</td><td>57.7</td><td>223.4</td></tr><tr><td> $SAEM + LeaPRR_{2023}$ </td><td>41.4</td><td>70.9</td><td>81.6</td><td>54.3</td><td>81.2</td><td>90.5</td><td>419.7</td><td>15.7</td><td>34.1</td><td>46.9</td><td>23.8</td><td>46.4</td><td>58.8</td><td>225.7</td></tr><tr><td> $SAEM + MACK^{VG-M}_{2024}$ </td><td>43.4</td><td>71.7</td><td>80.7</td><td>58.8</td><td>82.1</td><td>90.3</td><td>427.0</td><td>17.4</td><td>37.8</td><td>47.4</td><td>26.6</td><td>49.1</td><td>58.7</td><td>237.0</td></tr><tr><td> $SAEM + FR_{2025}$ </td><td>43.7</td><td>72.4</td><td>81.8</td><td>59.6</td><td>84.2</td><td>90.6</td><td>432.3</td><td>17.9</td><td>39.8</td><td>47.1</td><td>26.8</td><td>51.3</td><td>60.2</td><td>243.1</td></tr><tr><td> $SAEM + MASK$ </td><td>45.1</td><td>72.6</td><td>82.0</td><td>61.6</td><td>86.9</td><td>91.9</td><td>440.1</td><td>19.1</td><td>40.5</td><td>47.8</td><td>27.6</td><td>53.9</td><td>62.3</td><td>251.2</td></tr></table>

In Table 9, the results of two kinds of cross-dataset image-text matching are both presented, which are explained as follows. MSCOCO → Flickr30K: training existing models on the MSCOCO dataset and testing them on the Flickr30K dataset. Flickr30K → MSCOCO: training existing models on the Flickr30K dataset and testing them on the MSCOCO dataset. We observe that applying MASK to re-rank the outputs of VSRN and SAEM consistently enhances their generalization performance on unseen datasets, with substantial relative improvements observed across both sub-tasks. For example, $V S R N + M A S K$ performs much better than V SRN by 5.0% and 4.8% in R@1 on the MSCOCO → Flickr30K task, and by 3.6% and 6.3% in R@1 on the Flickr30K → MSCOCO task. Compared with $V S R N + F R ,$ , the relative improvements are also large, i.e., 7.2% ∼ 8.7% and 7.8% ∼ 8.1% in Rs when re-ranking VSRN and SAEM, respectively. Comparing the results in Table 9 with those in Table 2, we observe that the relative performance gains are more substantial for VSRN and SAEM, primarily because CLIP and ALBEF exhibit much higher baseline performance. In other words, re-ranking yields larger improvements when applied to less accurate models.

# J.2 VISUALIZATION OF POSITIVE AND NEGATIVE EXAMPLES FOR IMAGE RETRIEVAL AND IMAGE ANNOTATION

To better understand the OOD words, we show some representative examples of retrieved images or texts based on text or image queries by CLIP in Table 10. These examples are selected according to the following criteria: 1) Positive examples are those in which the ground-truth matched image or text is not ranked at top-1 by CLIP but is successfully promoted to a higher rank by MASK through re-ranking, and 2) Negative examples refer to situations where the ground-truth image or text is initially ranked at top-1 by CLIP but is pushed to a lower position after re-ranking by MASK. In the positive examples, it seems that the CLIP cannot well understand the semantic concepts such as “glasses”, “pierced”, and “broken”. For instance, in the top-1 retrieved image for the first text query, there are no clear clues indicating “pierced” or “glasses”, yet its rank is still higher than that of the ground-truth one. Similarly, in the top-1 retrieved text for the second image query, the annotation contains the word “glasses”, even though the image itself does not include such information. While our MASK especially focuses on understanding these semantic concepts and can thus increase the corresponding similarity scores between the matched regions and words. However, MASK may also make incorrect decisions. For example, in the images retrieved for the fourth text query, MASK reduces the rank of the ground-truth image. This behavior can be attributed to the presence of adverbs (e.g., “very”, “quite”), adjectives (e.g., “large”, “excited”), and pronouns (e.g., “they”, “this”) in the text. Our further analysis indicates that the OOD words negatively affecting MASK are typically those that cannot correspond to specific visual regions. Attempting to construct region prototypes for such words introduces substantial semantic noise. Finally, it is important to note that negative examples contain not only non-visual adjectives, adverbs, and pronouns, but also some informative OOD words with tense or plural variations. Consequently, the final image-text matching accuracy is influenced by the combined effects of all these words.

# J.3 DIFFERENT DETECTOR COMPARISON

This work derives region representations using the Bottom-Up and Top-Down (BUTD) model, i.e., Faster-RCNN, which consists of a Region Proposal Network (RPN) (Ren et al., 2015) and a 101- layer Residual Network (ResNet101) (He et al., 2016) pretrained on the VG dataset. Since our constructed multimodal knowledge is also based on the VG dataset, this pretraining step is crucial for learning discriminative region representations and achieving strong performance. To validate this, we experiment with three alternative detectors within the proposed MASK framework. The first is DETR (Carion et al., 2020), a recently popular Transformer-based detector. The second is DINO (Zhang et al., 2023), evaluated in two versions: the original model and a variant pretrained on the VG dataset in the same manner as BUTD. The third is an enhanced version of BUTD, referred to as BUTD+, which employs ResNet152, ConvNeXt (Liu et al., 2022), and Swin Transformer (Liu et al., 2021) as the backbone networks to replace ResNet101.

We evaluate these detectors on the task of unpaired image-text matching and compare their performance on the Flickr30K dataset in Table 11. The results show that directly using either DETR or DINO leads to poor performance. This is primarily because they fail to extract region representations as accurately as BUTD when constructing multimodal knowledge. Specifically, BUTD uses Faster R-CNN as its backbone, a two-stage object detector that allows ground-truth bounding boxes to be directly provided, yielding highly accurate feature representations. In contrast, DETR and DINO require generating hundreds of candidate bounding boxes and then selecting the one with the highest Intersection over Union (IoU) for each ground-truth box. This additional step inevitably introduces noise, thereby degrading the quality of the constructed knowledge and resulting in lower performance. In addition to inaccuracies in bounding box generation, the superior performance of

Table 10: Examples of retrieved top-3 images/texts based on text/image queries by CLIP. Groundtruth matched images are marked by black bounding boxes, which can be re-ranked higher by the proposed MASK. Similarly, the groundtruth text is annotated using the symbol “(GT)”. The underlined words indicate the representative OOD words relative to the multimodal knowledge. 

<table><tr><td>Query text/image</td><td colspan="3">Returned top-3 images/texts by CLIP</td><td>Pos/Neg</td><td colspan="3">Reranked top-3 images/texts by MASK</td></tr><tr><td>The man with pierced ears is wearing glasses and an orange hat.</td><td><img src="images/a93c57454d3340c2415c77af00ef6f40015f13afa1cd656afb29adecbcf904bd.jpg"/></td><td><img src="images/bbc6ef22a1c14e66dfc434e050264fe366acb4fb9866fd534a779c66870373fc.jpg"/></td><td><img src="images/5c45cc427708ed12cf1a59edff0915725eb4b132e7826fddaabdb6355fd3f31a.jpg"/></td><td>Pos</td><td><img src="images/5fc9bda278c678d2b14fcf74f83a18744872b508cd6159250947b238b525e841.jpg"/></td><td><img src="images/910154a723987cf84b7211c8c97dd3d5970ca5e2e666577f8c91b52ff8aaa31c.jpg"/></td><td><img src="images/6d45fa20c6e3173178463067d925111b20d9b9872a23782e3c3302d0397bc1df.jpg"/></td></tr><tr><td>A woman in a green dress sitting in a broken green and yellow chair.</td><td><img src="images/cf2d49e36fa10c13d0c54644e34e173467ebf8d8fcbae0806c8dcd22d11e7da9.jpg"/></td><td><img src="images/823b4d68034f45ecc9322f1cb76dc26e8e0690bc556789656df5978750b0d5a4.jpg"/></td><td><img src="images/b9eeae2be7e005526f94e70cdca0bef3d740dac1adfada4c8e5859bb81b80256.jpg"/></td><td>Pos</td><td><img src="images/3053e945fcd3feedd0724d7e96738bf0de0030407d39b84b09b5749f8f5a5b9c.jpg"/></td><td><img src="images/11b427f387a529068a56370daad3a65e8bdf9b0a3241a855af2d12e5e9227207.jpg"/></td><td><img src="images/6419fd194e7cf9e847610048e2a5a6bdedf02f03e9e36b256b096b8a2fdbbd46.jpg"/></td></tr><tr><td>A family is standing under a very large tree.</td><td><img src="images/d96c48633f10adc79f8467d33bcf7549e45c52430e8cd98e8a285f131d8a0e4a.jpg"/></td><td><img src="images/3ac47c3cb84d0196701bb57934fd94933f4a531c042c75007fdd1d3c89583cc8.jpg"/></td><td><img src="images/942b1e72e68c7d247eb3d8ef9f3536425d7f6b0fef2c0f7b32bf823f2a3b2f11.jpg"/></td><td>Neg</td><td><img src="images/a00e20b2ba5d85f17ff355debe3d85f25fd58e444e37d4bae97b3d5e1020e080.jpg"/></td><td><img src="images/705ac7359b179c8ecc5a686ee522455e1674ca9f574827df4c527af8c1657f3e.jpg"/></td><td><img src="images/189071102c2626ed908fbe3cb69ef99a521bd6fb095e499be4dd415b3706dc96.jpg"/></td></tr><tr><td>A couple is sitting on the sand with their feet in the water, and they are shaking hands.</td><td><img src="images/e5a0631fd30c6c0f8db75bffc8efe9d3e362dc9c9c66f2a5f308ec22a5fc8bee.jpg"/></td><td><img src="images/9c20a87b6fe2dc33d03775f91562177893ef747b710437c24c9275c53d17e848.jpg"/></td><td><img src="images/7b489d75fa323cdba2c32d830cdeea1d5f5c797fcfdabc054bce4b92385b3b3b.jpg"/></td><td>Neg</td><td><img src="images/8dd43f1e58a409bcb8c6b0e25db55790faf9b454beecbc888de984e1352f52f8.jpg"/></td><td><img src="images/3e70ea32cb6750c2b050fcf73b6813356a8acd18c276b7286773aad8155f7c94.jpg"/></td><td><img src="images/4182950c27080cec40c769c5e3149129cbeec1a25b93cf8bb6944b036f4c5c9b.jpg"/></td></tr><tr><td><img src="images/6c686bac18c7f1cff33e4c6e476b48e8d7745259e7f8d8d673766e1cc28a125c.jpg"/></td><td>A martial artist wearing a white Gi and a black belt is pinning another man in a blue Gi to the ground.</td><td>A young female student performing a downward kick to break a board held by Karate instructor. (GT)</td><td>The girl with the red belt is kicking a pad that the person in black is holding.</td><td>Pos</td><td>A young female student performing a downward kick to break a board held by Karate instructor. (GT)</td><td>The girl with the red belt is kicking a pad that the person in black is holding.</td><td>A martial artist wearing a white Gi and a black belt is pinning another man in a blue Gi to the ground.</td></tr><tr><td><img src="images/5e6a792588f575e73d4995bf2df3765cf32f5b86e40f4a0e108bc08285aa7d36.jpg"/></td><td>A girl wearing glasses is in a blue harness while rock climbing.</td><td>The person has a striped shirt on and is holding on to a rope on a mountain. (GT)</td><td>A mountaineer about to descend down a mountain with a blue helmet on.</td><td>Pos</td><td>The person has a striped shirt on and is holding on to a rope on a mountain. (GT)</td><td>A girl wearing glasses is in a blue harness while rock climbing.</td><td>A mountaineer about to descend down a mountain with a blue helmet on.</td></tr><tr><td><img src="images/79ab5e2e597480be0c260b4ba800379aba0be19109bf94c8e8b90f3bfb94490f.jpg"/></td><td>A young boy is quite excited in the throes of a ballgame. (GT)</td><td>A child wearing a blue shirt is jumping in the air.</td><td>A boy wearing jeans leaps in the air and shows shadow below.</td><td>Neg</td><td>A boy wearing jeans leaps in the air and shows shadow below.</td><td>A child wearing a blue shirt is jumping in the air.</td><td>A young boy is quite excited in the throes of a ballgame. (GT)</td></tr><tr><td><img src="images/57c69b86569dedb2c7e38e375ff291b4d7abb886803b0170411d3e21e3e4327c.jpg"/></td><td>This man bravely cuts down trees on the job. (GT)</td><td>A young adult is doing a back flip on a trampoline near a lake.</td><td>A man in blue overalls and red shirt holding a chainsaw.</td><td>Neg</td><td>A young adult is doing a back flip on a trampoline near a lake.</td><td>This man bravely cuts down trees on the job. (GT)</td><td>A man in blue overalls and red shirt holding a chainsaw.</td></tr></table>

Table 11: Unpaired image-text matching using different detectors on the Flickr30K dataset. 

<table><tr><td rowspan="2">Detector</td><td rowspan="2">Architecture</td><td rowspan="2">Pretrained</td><td rowspan="2">Boxes</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>DETR</td><td>ResNet50+Transformer</td><td>No</td><td>36</td><td>0.5</td><td>2.1</td><td>3.6</td><td>1.5</td><td>4.8</td><td>6.9</td><td>19.4</td></tr><tr><td>DINO</td><td>ResNet50+Transformer</td><td>No</td><td>36</td><td>0.8</td><td>2.7</td><td>4.2</td><td>1.9</td><td>4.6</td><td>7.4</td><td>21.6</td></tr><tr><td>DINO</td><td>ResNet50+Transformer</td><td>Yes</td><td>36</td><td>3.1</td><td>8.3</td><td>12.5</td><td>6.4</td><td>19.2</td><td>28.9</td><td>78.4</td></tr><tr><td>BUTD</td><td>RPN + ResNet101</td><td>Yes</td><td>36</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td></tr><tr><td>BUTD+</td><td>RPN + ResNet152</td><td>Yes</td><td>36</td><td>5.3</td><td>16.7</td><td>24.2</td><td>11.9</td><td>34.6</td><td>43.4</td><td>136.1</td></tr><tr><td>BUTD+</td><td>RPN + ResNet152</td><td>Yes</td><td>100</td><td>5.2</td><td>15.2</td><td>24.4</td><td>16.6</td><td>39.0</td><td>47.3</td><td>147.7</td></tr><tr><td>BUTD+</td><td>RPN + ConvNeXt</td><td>Yes</td><td>36</td><td>5.7</td><td>18.0</td><td>26.1</td><td>12.9</td><td>37.4</td><td>46.9</td><td>147.0</td></tr><tr><td>BUTD+</td><td>RPN + Swin</td><td>Yes</td><td>36</td><td>5.9</td><td>18.7</td><td>27.1</td><td>13.3</td><td>38.7</td><td>48.6</td><td>152.3</td></tr></table>

BUTD may also be attributed to its more powerful feature extraction network. To investigate this, we replace the ResNet101 backbone with ResNet152, ConvNeXt, and Swin Transformer, resulting in an enhanced version called BUTD+. This modification leads to a substantial performance improvement, demonstrating the benefit of stronger feature extraction.

# J.4 MODEL ARCHITECTURE ANALYSIS

This work acquires multimodal aligned knowledge based on three models: P AE, F RM, and MT M. Since the architecture of these models can significantly influence image-text matching accuracy, we use different architectures to perform the experiment of unpaired image-text matching and compare their performance on the Flickr30K dataset in Table 12.

We can see that increasing the number of fully connected (FC) layers leads to a decline in image-text matching performance, whereas adding more self-attention layers results in performance gains. For example, in the PAE model, increasing the number of fully connected layers from 1 to 3 results in a performance drop of approximately 19.4% in Rs on the Flickr30K dataset. When the number of layers is further increased from 3 to 5, accuracy decreases by an additional 11.3%. These findings demonstrate that FC layers have a substantial impact on model performance, and excessive depth can severely degrade accuracy. This can be explained as that FC layers apply fixed nonlinear transformations, where even minor training errors accumulate as the network deepens, gradually distorting the spatial distribution of features. In contrast, self-attention layers explicitly model similarity relationships among entities, inherently preserving or even reinforcing the semantic structure of the representations.

Table 12: Unpaired image-text matching using different model architectures on the Flickr30K dataset. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Layer</td><td rowspan="2">Params</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td rowspan="5">PAE</td><td>1 FC + 1 Self-Attention layer</td><td>2.11M</td><td>4.2</td><td>13.4</td><td>19.8</td><td>11.2</td><td>26.6</td><td>35.4</td><td>110.6</td></tr><tr><td>1 FC + 3 Self-Attention layers</td><td>4.21M</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td></tr><tr><td>1 FC + 5 Self-Attention layers</td><td>6.31M</td><td>4.6</td><td>15.3</td><td>26.3</td><td>12.3</td><td>28.5</td><td>41.7</td><td>128.7</td></tr><tr><td>3 FC + 3 Self-Attention layers</td><td>4.73M</td><td>2.6</td><td>11.8</td><td>16.4</td><td>10.3</td><td>26.1</td><td>36.2</td><td>103.4</td></tr><tr><td>5 FC + 3 Self-Attention layers</td><td>5.25M</td><td>1.9</td><td>9.3</td><td>12.6</td><td>9.6</td><td>24.8</td><td>33.9</td><td>92.1</td></tr><tr><td rowspan="5">FRM</td><td>1 FC + 1 Self-Attention layer</td><td>2.1M</td><td>4.2</td><td>13.9</td><td>21.2</td><td>11.7</td><td>29.6</td><td>37.5</td><td>118.1</td></tr><tr><td>2 FC + 1 Self-Attention layer</td><td>2.36M</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td></tr><tr><td>3 FC + 1 Self-Attention layer</td><td>2.62M</td><td>3.7</td><td>12.6</td><td>20.2</td><td>10.8</td><td>28.7</td><td>36.1</td><td>112.1</td></tr><tr><td>2 FC + 3 Self-Attention layers</td><td>4.46M</td><td>4.9</td><td>12.8</td><td>24.5</td><td>13.3</td><td>32.6</td><td>38.2</td><td>126.3</td></tr><tr><td>2 FC + 5 Self-Attention layers</td><td>6.56M</td><td>4.6</td><td>15.3</td><td>24.2</td><td>13.9</td><td>33.8</td><td>38.9</td><td>130.7</td></tr><tr><td rowspan="5">MTM</td><td>1 FC + 3 Self-Attention layers</td><td>1.26M</td><td>3.9</td><td>12.4</td><td>18.8</td><td>10.0</td><td>26.1</td><td>34.5</td><td>105.7</td></tr><tr><td>3 FC + 3 Self-Attention layers</td><td>1.44M</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td></tr><tr><td>5 FC + 3 Self-Attention layers</td><td>1.62M</td><td>2.7</td><td>11.3</td><td>16.1</td><td>9.2</td><td>23.5</td><td>32.4</td><td>95.2</td></tr><tr><td>3 FC + 1 Self-Attention layer</td><td>0.7M</td><td>4.2</td><td>13.6</td><td>20.4</td><td>11.7</td><td>28.2</td><td>37.5</td><td>115.6</td></tr><tr><td>3 FC + 5 Self-Attention layers</td><td>2.18M</td><td>5.1</td><td>13.9</td><td>22.7</td><td>12.8</td><td>32.4</td><td>41.3</td><td>128.2</td></tr></table>

# J.5 DIFFERENT PRETRAINED WORD VECTORS COMPARISON

The semantic geometry of word embeddings directly influences the construction of OOD prototypes. To evaluate the impact of different pretrained word vectors on matching accuracy, we perform the experiment of unpaired image-text matching by MASK using different word vectors on the Flickr30K and MSCOCO datasets in Table 13. Experimental results show that GloVe consistently outperforms Word2Vec and FastText across both datasets, demonstrating its superior suitability for constructing OOD prototypes. Specifically, GloVe achieves 122.8% R@s on Flickr30K and 209.5% R@s on MSCOCO, surpassing Word2Vec by 2.7 ∼ 5.2% and FastText by 8.4 ∼ 14.4%. These performance differences can be attributed to the distinct characteristics of the word vectors. GloVe encodes global word–word co-occurrence statistics, enabling it to capture broader contextual relatedness. Such global semantic structure is crucial in cross-modal matching, where visual regions and textual words need to align through high-level associative semantics rather than strict synonymy. In contrast, Word2Vec, which learns from local context windows, excels at modeling fine-grained synonymy but is less capable of capturing the broader semantic relations required for cross-modal alignment. FastText places greater emphasis on morphological similarity. However, morphological similarity does not necessarily imply semantic similarity, which introduces significant noise and ultimately reduces matching accuracy.

Table 13: Unpaired image-text matching by MASK using different pretrained word vectors on the Flickr30K and MSCOCO datasets. 

<table><tr><td rowspan="3">Word vectors</td><td colspan="7">Flickr30K dataset</td><td colspan="7">MSCOCO dataset</td></tr><tr><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>GloVe</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td><td>7.6</td><td>26.7</td><td>41.8</td><td>22.7</td><td>48.5</td><td>62.2</td><td>209.5</td></tr><tr><td>Word2Vec</td><td>4.6</td><td>14.0</td><td>21.3</td><td>11.3</td><td>28.6</td><td>37.8</td><td>117.6</td><td>7.4</td><td>26.2</td><td>41.5</td><td>22.2</td><td>47.8</td><td>61.7</td><td>206.8</td></tr><tr><td>FastText</td><td>4.2</td><td>12.6</td><td>19.9</td><td>10.0</td><td>26.1</td><td>35.6</td><td>108.4</td><td>7.0</td><td>25.4</td><td>40.7</td><td>21.3</td><td>46.3</td><td>60.4</td><td>201.1</td></tr></table>

# J.6 MODEL EFFICIENCY AND SIZE COMPARISON

As last, we compare the testing time and model size of our proposed MASK with those of CLIP, AL-BEF, and MACK on the same machine (Intel(R) Xeon(R) Platinum 8468V, 512 GB RAM memory, and 1 NVIDIA L40), based on their publicly available codes in Table 14. To eliminate the influence of other factors, such as the matching framework and feature extraction, we fix the batch size to 1 for all models during testing. In the table, the testing time is measured by seconds (s), and the model size is measured by the millions of model parameters (M).

From the table, we observe that MACK and MASK have significantly smaller model sizes compared to CLIP and ALBEF. The parameters of MACK come from the BUTD module used for extracting region representations. While the knowledge-based image-text matching is lightweight and requires no additional parameters. Similarly, for MASK, the majority of the testing time is attributed to the inference processes of the BUTD and P AE models, which is considerably faster than the testing time required by CLIP and ALBEF.

Table 14: Model Efficiency and Size Comparison. 

<table><tr><td>Detector</td><td>Testing Times (s)</td><td>Model Size (M)</td></tr><tr><td>CLIP</td><td>0.08</td><td>291.0</td></tr><tr><td>ALBEF</td><td>0.29</td><td>209.5</td></tr><tr><td>MACK</td><td>0.03</td><td>42.5</td></tr><tr><td>MASK</td><td>0.04</td><td>50.6</td></tr></table>

# J.7 HYPERPARAMETER ANALYSIS EXTENSION

To further evaluate the impact of different hyperparameters $\lambda _ { 1 }$ and $\lambda _ { 2 }$ on matching accuracy, we use different values of hyperparameters to perform the experiment of unpaired image-text matching and compare their performance on the Flickr30K and MSCOCO datasets in Table 15. The results indicate that MASK achieves the best performance in unpaired image-text matching when $\lambda _ { 1 } =$ $\lambda _ { 2 } = 3$ . This further indicates that the cross-modal alignment loss and the prototype consistency contrastive loss are complementary. By jointly optimizing the entire loss function in a balanced manner, the model’s generalization capability can be substantially improved.

# K LIMITION AND FUTURE WORK

It is important to acknowledge certain limitations of the proposed MASK, which will be addressed in future work. First, the raw region representations are extracted using the pre-trained object detection model BUTD. It would be better to pretrain more advanced detectors on the VG dataset to provide more discriminative region presentations. Second, relying solely on nouns for unpaired image-text matching is suboptimal. It would be better to take all the other words into consideration for more accurate image-text matching.

Table 15: Unpaired image-text matching by MASK using different $\lambda _ { 1 }$ and $\lambda _ { 2 }$ on the Flickr30K and MSCOCO datasets. 

<table><tr><td rowspan="3" colspan="2">Hyperparameter</td><td colspan="7">Flickr30K dataset</td><td colspan="7">MSCOCO dataset</td></tr><tr><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td><td colspan="3">Image Retrieval</td><td colspan="3">Image Annotation</td><td rowspan="2">Rs</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td rowspan="3"> $\lambda_1$ </td><td>1.0</td><td>4.2</td><td>13.0</td><td>19.2</td><td>11.6</td><td>27.9</td><td>36.8</td><td>112.7</td><td>6.8</td><td>22.9</td><td>35.5</td><td>20.9</td><td>43.2</td><td>55.8</td><td>185.1</td></tr><tr><td>3.0</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td><td>7.6</td><td>26.7</td><td>41.8</td><td>22.7</td><td>48.5</td><td>62.2</td><td>209.5</td></tr><tr><td>5.0</td><td>3.1</td><td>10.2</td><td>16.6</td><td>11.2</td><td>25.5</td><td>34.3</td><td>100.9</td><td>5.2</td><td>17.8</td><td>28.3</td><td>19.8</td><td>37.1</td><td>48.3</td><td>156.5</td></tr><tr><td rowspan="3"> $\lambda_2$ </td><td>1.0</td><td>3.9</td><td>12.2</td><td>18.0</td><td>11.2</td><td>27.5</td><td>36.4</td><td>109.2</td><td>6.3</td><td>21.0</td><td>32.3</td><td>19.8</td><td>41.5</td><td>53.9</td><td>174.8</td></tr><tr><td>3.0</td><td>4.8</td><td>14.8</td><td>22.0</td><td>12.1</td><td>30.1</td><td>39.0</td><td>122.8</td><td>7.6</td><td>26.7</td><td>41.8</td><td>22.7</td><td>48.5</td><td>62.2</td><td>209.5</td></tr><tr><td>5.0</td><td>3.6</td><td>11.7</td><td>17.8</td><td>11.6</td><td>25.2</td><td>33.5</td><td>103.4</td><td>5.6</td><td>19.4</td><td>30.7</td><td>19.6</td><td>39.9</td><td>50.2</td><td>165.4</td></tr></table>

# L THE USE OF LARGE LANGUAGE MODELS

We use LLMs solely to assist in checking grammatical correctness. After the initial check by the model, we further refine and correct any remaining grammatical issues manually. Therefore, the role of LLMs in this work is limited.