# MODA: MOdular Duplex Attention for

# Multimodal Perception, Cognition, and Emotion Understanding

Zhicheng Zhang 1 2 † Wuyou Xia 1 Chenxi Zhao 1 † Yan Zhou 3 Xiaoqiang Liu 3 Yongjie Zhu 3 ‡ Wenyu Qin 3 Pengfei Wan 3 Di Zhang 3 Jufeng Yang 1 2

# Abstract

Multimodal large language models (MLLMs) recently showed strong capacity in integrating data among multiple modalities, empowered by a generalizable attention architecture. Advanced methods predominantly focus on language-centric tuning while less exploring multimodal tokens mixed through attention, posing challenges in high-level tasks that require fine-grained cognition and emotion understanding. In this work, we identify the attention deficit disorder problem in multimodal learning, caused by inconsistent cross-modal attention and layer-by-layer decayed attention activation. To address this, we propose a novel attention mechanism, termed MOdular Duplex Attention (MODA), simultaneously conducting the inner-modal refinement and inter-modal interaction. MODA employs a correct-after-align strategy to effectively decouple modality alignment from cross-layer token mixing. In the alignment phase, tokens are mapped to duplex modality spaces based on the basis vectors, enabling the interaction between visual and language modality. Further, the correctness of attention scores is ensured through adaptive masked attention, which enhances the model’s flexibility by allowing customizable masking patterns for different modalities. Extensive experiments on 21 benchmark datasets verify the effectiveness of MODA in perception, cognition, and emotion tasks. Source code and demo are available in https://zzcheng.top/MODA.

(a)   
Corleone: A man who doesn't spend time with his family can never be a real man   
![](images/708edd3f29871e516bd06f87395b3d5114f0f571af223931f019aa7daf566d86.jpg)

<details>
<summary>natural_image</summary>

Two men in formal attire engaged in a dramatic interaction, one facing forward and the other in a suit (no visible text or symbols)
</details>

SOTA (LLaVA-NeXT): In the dialogue, we see a scene from a film where two characters ( :three), are engaged in a conversation. The character on the left, who appears to be a man in a suit and tie, is speaking to the character on the right, who is wearing a tuxedo with a red flower on the lapel. The character on the right is looking down ( :looking at Johnny), possibly at a piece of paper or a small object in his hand, and his expression is one of peace or contemplation ( :gravity).   
![](images/acbef6491fe83729901a1ff6856d33310bc5b17e796475c6d01b70308b6140a9.jpg)  
Figure 1. Illustration of deficit disorder attention problem. (a) Given the detailed image and lines from The Godfather, (b) we highlight incorrect responses, corresponding hallucinated explanations, and attached answers. (c) We visualize attention score across layers, highlighting inconsistent attention across modalities.

# 1. Introduction

Benefiting from the blossom of large language models (Chiang et al., 2023; Dubey et al., 2024; Teknium et al., 2024), multimodal large language models (MLLMs) have shown strong capacity in integrating multimodal data as human (Tong et al., 2024a; Brown et al., 2020; Bai et al., 2023), which illuminate a promising pathway toward Artificial General Intelligence (AGI). Advanced effort has been devoted to constructing MLLM (Achiam et al., 2023), focusing on exploring more insightful data curation, model tuning, and evaluation benchmarks. As the controller of agent, MLLMs provide a natural solution by conducting content perception (Liu et al., 2023), understanding role cognition (Dai et al., 2025), and analyzing human emotion (Yang et al., 2024). One more step forward into AGI lies in high-level multimodal understanding like humans, including cognition and emotion. Cognition, as a higher-level capability, requires the ability to model relationships and reasoning across modalities (Dai et al., 2025; Pessoa, 2022). Beyond cognition, emotion understanding is another critical aspect of fine-grained multimodal comprehension (Yang et al., 2024; Zhang et al., 2023). These high-level multimodal tasks pose new challenges for MLLMs.

While recent MLLMs show promising results in basic perception, they still struggle to perceive fine-grained details (Tong et al., 2024b), which is essential for understanding cognition and emotion. Public benchmarks reveal that these advanced MLLMs can underperform relative to random guessing (Yang et al., 2024), with 3 SOTAs achieving approximately 50:50 accuracy in 2-class sarcasm detection on the HFM dataset. This phenomenon arises from an excessive emphasis on the dominant modality data, leading to neglect of fine-grained details in alternative modality.

We delve deep into the reason and analyze the multimodal tokens mixed by attention in MLLM. As shown in Fig. 1 (a)&(b), we observe that SOTA MLLM struggles to capture fine-grained details (e.g., eyesights of character), leading to error in emotion understanding. The reason behind this is inconsistent attention across multiple layers in MLLM (63% disparity in Fig. 1 (c)), which we call deficit disorder attention problem. On the one hand, the attention scores in MLLM exhibit a bias towards the language component. On the other hand, layer-by-layer decay of attention further accentuates this disparity. As a result, the attention score disparity across modalities can reach up to 10 times.

Our intuition is that multimodal attention mechanisms often suffer from imbalances between self-modal and cross-modal interactions, leading to suboptimal feature co-operation across modalities. By explicitly separating and modulating these two components, we can better align multimodal features while preserving the unique characteristics of each modality. To achieve this, we propose MOdular Duplex Attention (MODA), which splits attention into self-modal and cross-modal parts, each with its own modulated attention mask. The self-modal attention component focuses on capturing the intrinsic relationships within individual modalities. In contrast, the cross-modal attention component is responsible for aligning features across different modalities, facilitating effective information exchange. At the core of the MODA model is the Duplex (V/T)-Aligner, which maps the tokens into a shared dual-modality representation space defined by two gram matrices. Additionally, the Modular Masked Attention component allows the model to adaptively focus on relevant modalities by applying customized masking patterns, further enhancing its flexibility on multimodal understanding tasks.

Our contributions are two-fold as follows: (1) From a novel perspective of the attention shift mechanism, we indicate the key bottleneck of attention among SOTA MLLMs and analyze the core reason in depth. We further propose a modular and duplex attention mechanism based on our observation. (2) We investigate a new MLLM for perception, cognition, and emotion, enabling applications in fine-grained understanding and planning. Extensive experiments on 21 benchmarks verify the generalization and effectiveness of MODA.

# 2. Related Work

Multimodal large language model (MLLM) have garnered significant attention recently due to their ability to integrate pre-trained foundational models, especially powerful Large Language Models (LLMs)(Achiam et al., 2023; Touvron et al., 2023), alongside multimodal encoders(Dosovitskiy et al., 2021; Radford et al., 2021). These models enhance the processing of multimodal inputs and outputs, as demonstrated in advanced works (Alayrac et al., 2022; Bai et al., 2023). MLLMs leverage attention mechanisms to facilitate multimodal token mixing, enabling both inductive and deductive understanding across modalities. However, the vision modality’s potential remains underutilized in many of these models. MMVP (Tong et al., 2024b) identifies a critical issue, highlighting how existing MLLMs fail to fully activate the vision modality due to improper handling of low-level visual attributes. Further, Cambrian-1 (Tong et al., 2024a) confirms this limitation and introduces a spatial vision aggregator to enhance visual feature. In this work, we investigate the root cause of these limitations, identifying the bottleneck in the design of the multimodal attention mechanism. To address the issue of imbalanced attention scores, we propose a novel multimodal attention that better balances the contributions of each modality.

Understanding cognition and emotion (Fu et al., 2023; Yang et al., 2024) play an important role in the pathway toward building an intelligent agent, except for content understanding demonstrated by prior MLLMs. As two of high-level understanding, cognition (Wang et al., 2024a; Kong et al., 2024; Salemi et al., 2024) typically refers to the ability to make decisions and judgments similar to characters (Binz & Schulz, 2023; Wang et al., 2024c; Deshpande et al., 2023), such as generating website code (Zhu et al., 2024; Wang et al., 2025), or role playing (Chen et al., 2024; Zhang et al., 2018). Emotion mainly depends on the psychology assumptions (Zhao et al., 2021; Zhang et al., 2024), where the categorical one is mostly used due to it being easily understandable (Yang et al., 2018; Mai et al., 2022; Lian et al., 2022; Zhang & Yang, 2022). However, it is less explored due to its requirements for fine-grained content understanding, which MLLMs can hardly achieve.

Attention in MLLM plays a critical role in addressing the computational and memory challenges inherent in their design. Significant progress has been made in developing efficient attention mechanisms for Transformer architectures, which include fixed patterns (Child et al., 2019), combinations of patterns (Zaheer et al., 2020), learnable patterns (Kitaev et al., 2020), neural memory (Beltagy et al., 2020), low-rank methods (Wang et al., 2020), and kernelbased techniques (Choromanski et al., 2021). For example, the Set Transformer introduces inducing points to handle set-input problems (Wang et al., 2020), while the Axial

![](images/bc57563c336e8f9248d8131a62b63ebd01cf17d27a3760e4d67a16b4b9198c48.jpg)

<details>
<summary>bar</summary>

(a)
| Category | Text | Image |
| :--- | :--- | :--- |
| General | 0.0061 | 0.0009 |
| Knowledge | 0.0067 | 0.0006 |
| Vision | 0.0059 | 0.0009 |
| DC-RC-Chart | 0.0063 | 0.0006 |
| 0-1 | 0.0063 | 0.0006 |
| 1-2 | 0.0063 | 0.0006 |
| 10-2 | 0.0063 | 0.0006 |
| 10-4 | 0.0063 | 0.0006 |
| 10-8 | 0.0063 | 0.0006 |
| 10-16 | 0.0063 | 0.0006 |
| 16-32 | 0.0063 | 0.0006 |
| 32-64 | 0.0063 | 0.0006 |
| 64-128 | 0.0063 | 0.0006 |
| 128-256 | 0.0063 | 0.0006 |
| 256-512 | 0.0063 | 0.0006 |
| 512-1024 | 0.0063 | 0.0006 |
| 1024-2564 | 0.0063 | 0.0006 |
| 2564-5124 | 0.0063 | 0.0006 |
| 5124-1M | 0.0063 | 0.0006 |
| 1M+2M+4M+8M+1M+2M+4M+8M+1M+4M+8M+1M+8M+1M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8M+8B
(+) (Figure: General) vs (Figure: Knowledge) vs (Figure: Vision) vs (Figure: DC-RC-Chart).
</details>

![](images/d8db51278231628054c015e645ce22fff2f10e50342bb93b19a55b13843b1f82.jpg)

<details>
<summary>heatmap</summary>

| Layer-wise Att. map | Layer 0 | Layer 8 | Layer 16 | Layer 24 |
| ------------------- | ------- | ------- | -------- | -------- |
| 0                   | 0       | 0       | 0        | 0        |
| 256                 | 256     | 256     | 256      | 256      |
| 512                 | 512     | 512     | 512      | 512      |
| 768                 | 768     | 768     | 768      | 768      |
</details>

![](images/7b5d6df65f5b240b029cad1e0957729ffcfa135c9fa1fe3f7ced823a98432ef1.jpg)

<details>
<summary>heatmap</summary>

(c)
| Per Layer Att. score | 0 | 8 | 16 | 24 |
| :--- | :--- | :--- | :--- | :--- |
| Layer | Text | 0.0104 | 0.0079 | 0.0058 |
| Layer | Visual | 0.0069 | 0.0074 | 0.0071 |
| Avg. Diff.: 56.97% | 0.0100 | 0.0130 | 0.0086 | 0.0080 |
| Avg. Diff.: 56.97% | 0.0087 | 0.0087 | 0.0072 | 0.0080 |
| Avg. Diff.: 56.97% | 0.0015 | 0.0015 | 0.0014 | 0.0015 |
| Avg. Diff.: 56.97% | 0.0026 | 0.0026 | 0.0015 | 0.0015 |
| Avg. Diff.: 56.97% | 0.0037 | 0.0037 | 0.0026 | 0.0026 |
| Avg. Diff.: 56.97% | 0.0148 | 0.0148 | 0.0141 | 0.0141 |
| Avg. Diff.: 56.97% | 0.0315 | 0.0315 | 0.0315 | 0.0315 |
| Avg. Diff.: 56.97% | 0.0422 | 0.0422 | 0.0422 | 0.0422 |
| Avg. Diff.: 56.97% | 0.0522 | 0.0522 | 0.0522 | 0.0522 |
| Avg. Diff.: 56.97% | 0.1636 | 0.1636 | 0.1636 | 0.1636 |
| Avg. Diff.: 56.97% | 0.2873 | 0.2873 | 0.2873 | 0.2873 |
| Avg. Diff.: 56.97% | 1.4144 | 1.4144 | 1.4144 | 1.4144 |
| Avg. Diff.: 56.97% | 2.8333 | 2.8333 | 2.8333 | 2.8333 |
| Avg. Diff.: 56.97% | 4.4542 | 4.4542 | 4.4542 | 4.4542 |
| Avg. Diff.: 56.97% | 6.1751 | 6.1751 | 6.1751 | 6.1751 |
| Avg. Diff.: 56.97% | 8.8961 | 8.8961 | 8.8961 | 8.8961 |
| Avg. Diff.: 56.97% | 12.5171 | 12.5171 | 12.5171 | 12.5171 |
| Avg. Diff.: 56.97% | 21.2381 | 21.2381 | 21.2381 | 21.2381 |
| Avg. Diff.: 56.97% | 32.8591 | 32.8591 | 32.8591 | 32.8591 |
| Avg. Diff.: 56.97% | 48.4891 | 48.4891 | 48.4891 | 48.4891 |
| Avg. Diff.: 56.97% | 67.2391 | 67.2391 | 67.2391 | 67.2391 |
| Avg. Diff.: 56.97% | 88.9691 | 88.9691 | 88.9691 | 88.9691 |
| Avg. Diff.: 56.97% | 139.5891 | 139.5891 | 139.5891 | 139.5891 |
| Avg. Diff.: 56.97% | 252.2491 | 252.2491 | 252.2491 | 252.2491 |
| Avg. Diff.: 56.97% | 385.8791 | 385.8791 | 385.8791 | 385.8791 |
| Avg. Diff.: 56.97% | 539.4391 | 539.4391 | 539.4391 | 539.4391 |
| Avg. Diff.: 56.97% | 714.2691 | 714.2691 | 714.2691 | 714.2691 |
| Avg. Diff.: 56.97% | 928.9391 | 928.9391 | 928.9391 | 928.9391 |
| Avg. Diff.: 56,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,<ecel><ecel><ecel><ecel><nl>
</details>

Figure 2. Analysis of existing MLLMs on four fine-grained understanding tasks. (a) The distribution of attention activation values among visual and textual tokens. (b) The attention map for multimodal tokens among stages. (c) The self- and cross-modal attention activation scores with their disparity among the attention layers.

Transformer applies attention along individual axes of input tensors, reducing computational overhead (Beltagy et al., 2020). These innovations collectively enhance the scalability of Transformer models, enabling their application to tasks with large inputs or long sequences (Choromanski et al., 2021; Han et al., 2024). While previous approaches have focused on improving the efficiency and scalability of attention in single-modal tasks, the multimodal context introduces unique challenges, such as balancing attention scores across heterogeneous modalities (Zhao et al., 2021). Our work extends this line of research by specifically addressing the multimodal attention mechanism in MLLMs.

# 3. Methodology

# 3.1. Preliminary

• Attention Given the input multimodal tokens, $\boldsymbol { x } \in$ $\mathbb { R } ^ { N \times d }$ , N be the number of tokens and d be the dimensionality of the hidden state. Let $\pmb { A } \in \mathbb { R } ^ { N \times N }$ denote the attention score matrix computed among N tokens, we have $A = Q K ^ { \top } / \tau$ , and the output of attention layer as:

$$
\boldsymbol {O} = \text { Softmax } (\frac {\boldsymbol {Q} \boldsymbol {K} ^ {\top}}{\tau} + \boldsymbol {M}) \boldsymbol {V}. \tag {1}
$$

where Q, K, $V \in \mathbb { R } ^ { d \times d }$ represents query, key, and value matrix derived from input tokens. Attention is also practically masked $M \in \mathbb { R } ^ { \hat { N } \times N }$ to filter out special tokens (Li et al., 2023) or conduct causal sequential modeling (Wang et al., 2024b; Achiam et al., 2023).

• Multimodal Attention Formally, consider a multimodal token sequence $X _ { M }$ comprising M modalities. The total token length is $N _ { M } = N _ { 1 } + \cdot \cdot \cdot + N _ { M }$ , where $N _ { m }$ represents the length of the $m ^ { t h }$ modality token sequence $X _ { m }$ . The attention can be split into two parts for each modality token sequence, self-modal attention and cross-modal attention. We have $( \cdot ) ^ { [ m , \bar { m } ] }$ , which represents the tokens derived from the $m ^ { t h }$ modality and rest. For the self-modal and crossmodal attention, we have

$$
\boldsymbol {O} _ {\text { self }} = \operatorname{Softmax} (\frac {\boldsymbol {Q} ^ {m} \boldsymbol {K} ^ {m ^ {\top}}}{\tau} + \boldsymbol {M}) \boldsymbol {V} ^ {m}, \tag {2}
$$

$$
\boldsymbol {O} _ {\text { cross }} = \operatorname{Softmax} (\frac {\boldsymbol {Q} ^ {m} \boldsymbol {K} ^ {\bar {m}} {} ^ {\top}}{\tau} + \boldsymbol {M}) \boldsymbol {V} ^ {\bar {m}}. \tag {3}
$$

# 3.2. Deficit Disorder Attention Problem

Recently, multimodal attention has played a very important role in multimodal areas, including diffusion models that involve cross-modal generation and MLLM that involves cross-modal understanding. The attention mechanism governs token interactions by computing similarities and applying masks. To further investigate the Attention Deficit Disorder (DDA) phenomenon, we conduct a series of analyses on four categories of fine-grained understanding tasks.

As shown in Fig. 2 (a), we observe that the attention devoted to visual content is significantly weaker compared to that for the textual modality. This observation aligns with the challenges faced by MLLMs fine-tuned from autoregressive models in handling fine-grained visual perception. The inherent design of MLLM, which is primarily optimized for text-based tasks, may lead to an underrepresentation of visual features when extended to multimodal contexts. This imbalance highlights a critical limitation in the current architecture, where the model’s proficiency in textual processing does not seamlessly translate to an equivalent capability. Further, we conduct experiments on Fig. 2 (b)&(c), and we observe a distinct cross-attention bias in the lower layers of the model across its 32 layers. This bias is notably inconsistent with the distribution of attention in the higher layers, which are known for their stronger representational capabilities. Specifically, the lower layers tend to focus disproportionately on cross-modal interactions, potentially at the expense of effectively capturing intra-modal features, leading to suboptimal multimodal integration.

(a)   
![](images/6d7b2e141841e6dd85725becb55ab5c47f099eeb97ae8dc6b6753005a7a29658.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["MODA Model"] --> B["N × Attention Block"]
    B --> C["Image: 10+ frames in a room with 'Johnny' and 'cry' labels"]
    C --> D["Context (background, dialogue)"]
    D --> E["Back view of man in white shirt"]
```
</details>

![](images/c7de90f43be601bab6d84482f1938be59b52989d172776f50f2921f420e0e904.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["FFN"] --> B["Norm"]
    B --> C["MODA-attn"]
    C --> D["V-Aligner"]
    C --> E["T-Aligner"]
    D --> F["Visual tokens"]
    E --> G["Text tokens"]
    F --> H["Norm"]
    G --> H
    H --> I["Output"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#ffc,stroke:#333
    style I fill:#cfc,stroke:#333
```
</details>

(c)   
![](images/b9d0cf2ab2c9ae3c44a14484d8579fb354b2bb36ac66df1508201035ad0cace1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Modular Attention Mask
        A["QK^T (N=M)×(N+M)"] --> B["Fix Mask"]
        B --> C["Meta"]
        C --> D["Conditional Mask"]
        E["Visual Coord"] --> B
        F["Text Coord"] --> C
    end

    subgraph Duplex Attention Alignment
        G["N×D"] --> H["K"]
        H --> I["D×N"]
        I --> J["Dot"]
        J --> K["D×D"]
        K --> L["N×D"]
        M["M×D"] --> N["K"]
        N --> O["M×D"]
        O --> P["Fusion"]
        P --> Q["M×D"]
    end

    style Modular Attention Mask fill:#f9f,stroke:#333
    style Duplex Attention Alignment fill:#bbf,stroke:#333
```
</details>

Figure 3. MOdular Duplex Attention. (a) MODA takes the image and contextual prompt as input, including the background and history of the conversation. (b) With MODA, the token flows are justified in each Transformer block of MLLM. MODA modifies the deficient attention scores in a correct-after-align manner via (c) Modular masked attention and (d) Duplex (V/T)-aligner.

This leads to the formal introduction of the Deficit Disorder Attention (DDA) problem. Given the visual tokens $x _ { v } ^ { l }$ and text tokens $x _ { t } ^ { l }$ in the block l, the multimodal attention builds the link from two parts (i.e., self-modal $x _ { t . } ^ { l }  x _ { t } ^ { l + 1 } , x _ { v } ^ { l } $ $x _ { v } ^ { l + 1 }$ and cross-modal $x _ { t } ^ { l } \to x _ { v } ^ { l + 1 } , x _ { v } ^ { l } \to x _ { t } ^ { l + 1 } )$ l +1v , x lv t v , where the links are commonly implemented by the pair-wise token similarity and weighted sum. However, the modality gap between tokens decrease the magnitude of links, as we observed, the link value of $x _ { v } ^ { l } \to x _ { v } ^ { \bar { l } + 1 }$ and $x _ { t } ^ { l }  x _ { v } ^ { l + 1 }$ decays exponentially with depth $( \alpha _ { v , t  v } ^ { l } \propto \gamma ^ { l } , \gamma \neq 1 )$ . This misalignment propagates layer-wise, causing the cumulative error in cross-modal interaction to grow as

$$
\mathbb {E} _ {D D A} = \prod_ {l} \gamma^ {l} \epsilon_ {l}, \tag {4}
$$

where $\epsilon _ { l }$ denotes the layer-specific alignment error. This phenomenon aligns with the theoretical insights in (Dong et al., 2021), where pure attention mechanisms experience rank collapse, a critical factor that exacerbates the imbalance in attention distribution.

# 3.3. MOdular Duplex Attention

When the gap across modalities arises, we propose to align the tokens from multiple modalities in the attention, which we call modular duplex attention (MODA). MODA first splits multimodal attention into the modality alignment part and the token focus correction part.

• Duplex Attention Alignment To reduce the modality inconsistency, a natural idea is to align them. Inspired by the recent advance of visual-language embedding space mapping in diffusion model (Rombach et al., 2022), we propose mapping the token into the other modality space according to the embedding space bases of the gram matrix. We extract the basis vector of each modality space according to the gram matrix of tokens (Ryu et al., 2023; Peebles & Xie, 2023), thus compressing the semantics of each modality and serving as a transfer for other modalities. Thus, the duplex attention alignment consists of V-Aligner and T-Aligner responsible for visual and language modality, respectively.

Specifically, for the $m ^ { t h }$ modality, the space bases are given according to the normed gram matrix $| | G ^ { m } | | \in \bar { \mathbb { R } } ^ { d \times d }$ , where $G _ { i j } ^ { m }$ is the inner product between tokens i and j:

$$
\boldsymbol {G} _ {i j} ^ {m} = \sum_ {k = 1} ^ {N _ {m}} \boldsymbol {K} _ {i k} ^ {m} \boldsymbol {K} _ {k j} ^ {m} = \boldsymbol {K} ^ {m \top} \boldsymbol {K} ^ {m}, \tag {5}
$$

where $\pmb { K } ^ { m }$ are the key states of the $m ^ { t h }$ modality tokens and $N _ { m }$ is the number of token belong to modality m. By including the base vectors of the space defined by the Gram matrix, we can effectively capture the relationships among the tokens within the $m ^ { t h }$ modality. This allows us to construct a feature representation that is not only rich in information but also maintains the intrinsic structure of the data.

As a following product, the normed gram matrix serves as a cross-modal token transfer function, enabling an efficient transformation of tokens from other modality m¯ into the modality m as a kernelized mapping function $\dot { f } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ The aligned tokens are computed as follows:

$$
\boldsymbol {K} ^ {\bar {m} \rightarrow m} = \boldsymbol {K} ^ {\bar {m}} | | \boldsymbol {G} ^ {m} | |, \tag {6}
$$

where ${ \pmb K } ^ { \bar { m } }$ represents the value representation from other modalities m¯ . The mapped tokens are further fused with the original ones to enhance the token similarity among all the modalities. Due to the substantial computational expense associated with training a complete MLLM, we utilize token merging and LoRA-based tuning to develop the fuser. Notably, the computation in the alignment stage keeps linear complexity to the token number, since the matrix sum among tokens is only conducted in the first round.

Table 1. Ablation Study. We conduct experiments on four types of multimodal tasks, including general QA (G), knowledge QA (K), OCR&Chart QA (O), and vision-centric QA (V). The lines with blue shallow indicate the optimal setting for our method. If not otherwise specified, this setting is used for all subsequent experiments.   
(a) Module 

<table><tr><td>MDM</td><td>DAA</td><td>G</td><td>K</td><td>O</td><td>V</td></tr><tr><td>-</td><td>-</td><td>63.6</td><td>44.0</td><td>60.8</td><td>38.0</td></tr><tr><td>√</td><td>-</td><td>69.2</td><td>45.4</td><td>60.9</td><td>42.6</td></tr><tr><td>-</td><td>√</td><td>67.8</td><td>47.6</td><td>63.3</td><td>48.1</td></tr><tr><td>√</td><td>√</td><td>69.3</td><td>48.3</td><td>67.0</td><td>54.3</td></tr></table>

(b) Attention Alignment 

<table><tr><td>align</td><td>G</td><td>K</td><td>O</td><td>V</td></tr><tr><td>MLP</td><td>69.5</td><td>47.5</td><td>66.8</td><td>46.0</td></tr><tr><td>+2xMLP</td><td>66.5</td><td>48.6</td><td>67.9</td><td>49.1</td></tr><tr><td>+GeLU</td><td>69.5</td><td>49.1</td><td>64.0</td><td>54.2</td></tr><tr><td>+CoV</td><td>69.3</td><td>48.3</td><td>67.0</td><td>54.3</td></tr></table>

(c) Attention Fusion 

<table><tr><td>fusion</td><td>G</td><td>K</td><td>O</td><td>V</td></tr><tr><td> $X_p$ </td><td>69.2</td><td>45.4</td><td>60.9</td><td>42.6</td></tr><tr><td> $X_a$ </td><td>67.8</td><td>47.6</td><td>63.3</td><td>48.1</td></tr><tr><td>Con</td><td>69.3</td><td>48.3</td><td>67.0</td><td>54.3</td></tr><tr><td>Add</td><td>62.2</td><td>47.6</td><td>67.2</td><td>52.2</td></tr></table>

(d) Attention Mask 

<table><tr><td>mask</td><td>G</td><td>K</td><td>O</td><td>V</td></tr><tr><td>Inf</td><td>67.8</td><td>47.6</td><td>63.3</td><td>48.1</td></tr><tr><td>Fix</td><td>70.1</td><td>49.0</td><td>67.0</td><td>52.3</td></tr><tr><td>Attn.</td><td>69.3</td><td>48.3</td><td>67.0</td><td>54.3</td></tr><tr><td>[M]</td><td>69.5</td><td>47.5</td><td>66.8</td><td>46.0</td></tr></table>

• Modular Attention Mask Attention mask controls the flow of tokens across transformer layers and induces the positional bias for MLLM (Wu et al., 2024). To better fit the requirements of the multimodal token sequence, we assign a modulated attention mask for each modality, where the mask is split into $M ^ { m }$ and $M ^ { \bar { m } }$ responsible for self- and cross-modality, respectively.

$$
\boldsymbol {O} _ {\text { self }} = \operatorname{Softmax} (\frac {\boldsymbol {Q} ^ {m} \boldsymbol {K} ^ {m ^ {\top}}}{\tau} + \boldsymbol {M} ^ {m}) \boldsymbol {V} ^ {m}, \tag {7}
$$

$$
\boldsymbol {O} _ {\text { cross }} = \operatorname{Softmax} \left(\frac {\boldsymbol {Q} ^ {m} \boldsymbol {K} ^ {\bar {m} ^ {\top}}}{\tau} + \boldsymbol {M} ^ {\bar {m}}\right) \boldsymbol {V} ^ {\bar {m}}. \tag {8}
$$

To alleviate the collapsed attention matrix and prevent it from under-smoothed tokens. We first introduce a modular attention mask that stores unnecessary attention values as pseudo-attention scores (Yin et al., 2024). For each row, representing the attention scores for the i-th token, the sequence length that the token can attend to is fixed at n. Consequently, each row contains n − i pseudo-attention scores, which are allocated to the excess values. The attention scores are formally represented using a masking strategy with a decay rate $\beta ,$ as follows:

$$
A _ {M M} = \left( \begin{array}{c c c c} \boldsymbol {q} _ {1} \boldsymbol {k} _ {1} ^ {\top} & p _ {1 1} & \dots & p _ {1 (n - 1)} \\ \boldsymbol {q} _ {2} \boldsymbol {k} _ {1} ^ {\top} & \boldsymbol {q} _ {2} \boldsymbol {k} _ {2} ^ {\top} & \dots & p _ {1 (n - 2)} \\ \vdots & \vdots & \ddots & \vdots \\ \boldsymbol {q} _ {n} \boldsymbol {k} _ {1} ^ {\top} & \boldsymbol {q} _ {n} \boldsymbol {k} _ {2} ^ {\top} & \dots & \boldsymbol {q} _ {n} \boldsymbol {k} _ {n} ^ {\top} \end{array} \right) \tag {9}
$$

$$
p _ {b a s e} = 0, p _ {i j} = p _ {b a s e} - (j - 1) \beta \tag {10}
$$

Except for the absolute location prior information, we further introduce the modality location to enforce the model to correct the token flow. We introduce the normed gram matrix as an indicator, to find out the part should be carried with modality location priors. We introduce the normed Gram matrix to serve as a critical indicator, guiding the model in identifying which components should leverage modality location priors. This separation allows for more precise control over how tokens from the same modality interact with each other versus how they engage with tokens from other modalities. The self-modal attention, represented by $O _ { s e l f }$ , focuses on refining the relationships within the same modality, ensuring that relevant information is effectively propagated through the layers. Conversely, the cross-modal attention, denoted by $O _ { c r o s s }$ , facilitates the exchange of information between distinct modalities, enabling the model to leverage complementary features.

# 4. Experiment

# 4.1. Benchmark Datasets

Perception: Following (Tong et al., 2024a), we conduct experiments on 4 types of perception task (i.e., general, knowledge, ocr, and vision-centric) across 16 benchmarks: MME (Fu et al., 2023), MMBench (Liu et al., 2025), SEED (Li et al., 2024), GQA (Hudson & Manning, 2019), ScienceQA (Lu et al., 2022), MMMU (Yue et al., 2024), MathVista (Lu et al., 2024), AI2D (Kembhavi et al., 2016), ChartQA (Masry et al., 2022), OCRBench (Liu et al., 2024), TextVQA (Singh et al., 2019), DocVQA (Mathew et al., 2021), MMVP (Tong et al., 2024b), RealworldQA (xAI, 2024), and CV-Bench (Tong et al., 2024a). We adopt GPT4 score to evaluate response.

Cognition: Following (Dai et al., 2025), we conduct experiments on MMRole to evaluate role-playing performance from 8 aspects: instruction adherence, fluency, coherency, image-text relevance, response accuracy, personality consistency, knowledge consistency, and tone consistency.

Emotion: Following (Yang et al., 2023; Huang et al., 2024), we conduct experiments on 4 benchmark datasets. MVSA-S and MVSA-M (Niu et al., 2016) are datasets used for sentiment polarity classification (positive or negative), while TumEmo (Yang et al., 2021) is a multimodal dataset designed for classifying six basic emotions. Additionally, HFM (Liu et al., 2022) is a multimodal dataset focused on recognizing high-level implicit emotion of sarcasm.

<table><tr><td>Model</td><td colspan="5">General</td><td colspan="5">Knowledge</td><td colspan="5">OCR &amp; Chart</td><td colspan="5">Vision-Centric</td></tr><tr><td>Method</td><td>Avg</td><td> $MME^P$ </td><td>MMB</td><td> $SEED^I$ </td><td>GQA</td><td>Avg</td><td> $SQA^I$ </td><td> $MMMU^V$ </td><td> $MathVista^M$ </td><td>AI2D</td><td>Avg</td><td>ChartQA</td><td>OCRBench</td><td>TextVQA</td><td>DocVQA</td><td>Avg</td><td>MMVP</td><td>RealworldQA</td><td> $CV-Bench^{2D}$ </td><td> $CV-Bench^{3D}$ </td></tr><tr><td>GPT-4V</td><td>63.0</td><td>1409.4</td><td>75.8</td><td>69.1</td><td>36.8</td><td>65.2</td><td>75.7</td><td>56.8</td><td>49.9</td><td>78.2</td><td>77.4</td><td>78.5</td><td>64.5</td><td>78.0</td><td>88.4</td><td>62.4</td><td>50.0</td><td>61.4</td><td>64.3</td><td>73.8</td></tr><tr><td>Gemini-1.0 Pro</td><td>-</td><td>1496.6</td><td>73.6</td><td>70.7</td><td>-</td><td>-</td><td>79.5</td><td>47.9</td><td>45.2</td><td>-</td><td>-</td><td>-</td><td>65.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Gemini-1.5 Pro</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>58.5</td><td>52.1</td><td>80.3</td><td>-</td><td>81.3</td><td>-</td><td>73.5</td><td>86.5</td><td>-</td><td>-</td><td>67.5</td><td>-</td><td>-</td></tr><tr><td>Grok-1.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>53.6</td><td>52.8</td><td>88.3</td><td>-</td><td>76.1</td><td>-</td><td>78.1</td><td>85.6</td><td>-</td><td>-</td><td>68.7</td><td>-</td><td>-</td></tr><tr><td>MM-1-8B</td><td>-</td><td>1529.3</td><td>72.3</td><td>69.9</td><td>-</td><td>-</td><td>72.6</td><td>37.0</td><td>35.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MM-1-30B</td><td>-</td><td>1637.6</td><td>75.1</td><td>72.1</td><td>-</td><td>-</td><td>81.0</td><td>44.7</td><td>39.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Base LLM: Llama-3-Ins-8B</td><td colspan="5"></td><td colspan="5"></td><td colspan="5"></td><td colspan="5"></td></tr><tr><td>Mini-Gemini-HD-8B</td><td>72.7</td><td>1606.0</td><td>72.7</td><td>73.2</td><td>64.5</td><td>55.7</td><td>75.1</td><td>37.3</td><td>37.0</td><td>73.5</td><td>62.9</td><td>59.1</td><td>47.7</td><td>70.2</td><td>74.6</td><td>51.5</td><td>18.7</td><td>62.1</td><td>62.2</td><td>63.0</td></tr><tr><td>LLaVA-NeXT-8B</td><td>72.5</td><td>1603.7</td><td>72.1</td><td>72.7</td><td>65.2</td><td>55.6</td><td>72.8</td><td>41.7</td><td>36.3</td><td>71.6</td><td>63.9</td><td>69.5</td><td>49.0</td><td>64.6</td><td>72.6</td><td>56.6</td><td>38.7</td><td>60.1</td><td>62.2</td><td>65.3</td></tr><tr><td>Cambrian-1-8B</td><td>73.1</td><td>1547.1</td><td>75.9</td><td>74.7</td><td>64.6</td><td>61.3</td><td>80.4</td><td>42.7</td><td>49.0</td><td>73.0</td><td>71.3</td><td>73.3</td><td>62.4</td><td>71.7</td><td>77.8</td><td>65.0</td><td>51.3</td><td>64.2</td><td>72.3</td><td>72.0</td></tr><tr><td>MODA-8B</td><td>72.1</td><td>1535.9</td><td>73.8</td><td>74.9</td><td>63.0</td><td>61.5</td><td>80.4</td><td>43.1</td><td>48.8</td><td>73.6</td><td>72.0</td><td>74.3</td><td>65.2</td><td>70.4</td><td>78.1</td><td>66.0</td><td>52.6</td><td>64.1</td><td>73.5</td><td>73.8</td></tr><tr><td>Base LLM: Hermes2-Yi-34B</td><td colspan="5"></td><td colspan="5"></td><td colspan="5"></td><td colspan="5"></td></tr><tr><td>Mini-Gemini-HD-34B</td><td>76.2</td><td>1659.0</td><td>80.6</td><td>75.3</td><td>65.8</td><td>62.4</td><td>77.7</td><td>48.0</td><td>43.4</td><td>80.5</td><td>68.1</td><td>67.6</td><td>51.8</td><td>74.1</td><td>78.9</td><td>63.8</td><td>37.3</td><td>67.2</td><td>71.5</td><td>79.2</td></tr><tr><td>LLaVA-NeXT-34B</td><td>76.0</td><td>1633.2</td><td>79.3</td><td>75.9</td><td>67.1</td><td>62.5</td><td>81.8</td><td>46.7</td><td>46.5</td><td>74.9</td><td>67.7</td><td>68.7</td><td>54.5</td><td>69.5</td><td>78.1</td><td>64.0</td><td>47.3</td><td>61.0</td><td>73.0</td><td>74.8</td></tr><tr><td>Cambrian-1-34B</td><td>76.8</td><td>1689.3</td><td>81.4</td><td>75.3</td><td>65.8</td><td>67.0</td><td>85.6</td><td>49.7</td><td>53.2</td><td>79.7</td><td>71.9</td><td>75.6</td><td>60.0</td><td>76.7</td><td>75.5</td><td>68.5</td><td>52.7</td><td>67.8</td><td>74.0</td><td>79.7</td></tr><tr><td>MODA-34B</td><td>76.7</td><td>1639.2</td><td>82.3</td><td>75.8</td><td>66.2</td><td>69.5</td><td>88.1</td><td>52.5</td><td>54.0</td><td>83.4</td><td>74.7</td><td>79.8</td><td>62.7</td><td>78.3</td><td>78.2</td><td>69.9</td><td>53.8</td><td>68.5</td><td>75.8</td><td>81.3</td></tr></table>

Table 2. Comparison of MODA with other leading MLLM framework on twelve perception benchmarks. MODA outperforms other open-source models and achieves competitive performance on a number of benchmarks, compared to proprietary models such as GPT-4V, Gemini, and Grok-1.5. The reported numbers of leading MLLMs come from (Tong et al., 2024a).

# 4.2. Settings

We set the same experiment setting as (Tong et al., 2024a; Liu et al., 2023). We adopt CLIP (ViT-L/14) (Radford et al., 2021) as the visual encoder. For the foundational large language model, we choose models from different scales, i.e., 8B: Llama-3-Instruct-8B (Dubey et al., 2024) and 34B: Hermes2-Yi-34B (Young et al., 2024). MODA is trained for 1 epoch with a batch size of 2048, using the AdamW (Loshchilov & Hutter, 2019) optimizer with a cosine learning rate schedule. The learning rate is set to 2e-5 for LLM and 2e-6 for visual encoder, respectively. The warmup rate is 0.03.

# 4.3. Ablation Study

To investigate the effectiveness of duplex attention alignment and modular attention mask, we conduct a componentwise ablation study in Table 1. For ablation studies, we train the MLLMs at the scale of 8B, with the base LLM of Llama-3-Ins-8B. For a fair comparison, all models are trained on 700K data samples for 1 epoch. We further discuss each component by conducting in-depth analyses of their variants to answer the following research questions.

• RQ1: How does the design of duplex attention alignment impact cross-modal feature transfer?   
• RQ2: How does the modular attention mask address modality position bias and improve attention?   
• RQ3: How do the proposed duplex attention alignment and modular attention mask respectively interact to

enhance multimodal attention?

• Response to RQ1: Modality Axis Transfer we analyze the effectiveness of duplex attention alignment in facilitating cross-modal feature transfer by examining its ability to align modality-specific features along a shared latent axis. This is motivated by the need to reduce modality gaps and ensure effective information exchange between modalities. We design experiments to test different variants of duplex attention alignment, such as using covariance matrices, attention head configurations, and linear vs. non-linear transformations.

• Response to RQ2: Modality Position Bias we investigate the role of the modular attention mask in addressing modality position bias and improving attention distribution. This analysis is crucial for understanding how the mask prevents attention collapse and ensures balanced contributions from all modalities. We experiment with different masking mechanisms, such as traditional infinity masking, fix-valued masking, and learnable masking. These variants are evaluated on tasks involving long sequences and imbalanced modality contributions, such as vision-centric perception and knowledge understanding.

• Response to RQ3: Multimodal Attention Matrix we analyze the interaction between duplex attention alignment and modular attention mask by studying their combined effect on the multimodal attention matrix. This is motivated by the hypothesis that the two components work synergistically to improve multimodal representation learning by enhancing both alignment and attention distribution. We design experiments that compare the joint use of these components against their individual use, as well as against baseline models without either component. Tasks such as question answering and multimodal summarization are chosen to simultaneously evaluate alignment and distribution.

<table><tr><td rowspan="2">Model</td><td colspan="9">Cognition</td></tr><tr><td>Avg</td><td>Instruction Adherence</td><td>Fluency</td><td>Coherency</td><td>Image-Text relevance</td><td>Response Accuracy</td><td>Personality Consistency</td><td>Knowledge Consistency</td><td>Tone Consistency</td></tr><tr><td>GPT-4 Turbo</td><td>1.099</td><td>1.055</td><td>1.032</td><td>1.084</td><td>1.097</td><td>1.092</td><td>1.168</td><td>1.103</td><td>1.161</td></tr><tr><td>Gemini 1.0 Pro</td><td>1.021</td><td>0.999</td><td>1.007</td><td>1.028</td><td>1.009</td><td>1.013</td><td>1.052</td><td>1.013</td><td>1.050</td></tr><tr><td>Claude 3 Opus</td><td>1.157</td><td>1.127</td><td>1.070</td><td>1.149</td><td>1.167</td><td>1.146</td><td>1.219</td><td>1.168</td><td>1.213</td></tr><tr><td>QWen-VL-Max</td><td>1.028</td><td>1.014</td><td>1.012</td><td>1.035</td><td>1.034</td><td>1.029</td><td>1.042</td><td>1.021</td><td>1.041</td></tr><tr><td colspan="10">Base: Llama-3-Ins-8B</td></tr><tr><td>Mini-Gemini-HD-8B</td><td>0.878</td><td>0.884</td><td>0.942</td><td>0.898</td><td>0.864</td><td>0.853</td><td>0.855</td><td>0.876</td><td>0.852</td></tr><tr><td>LLaVA-NeXT-8B</td><td>0.968</td><td>0.971</td><td>0.988</td><td>0.980</td><td>0.966</td><td>0.967</td><td>0.966</td><td>0.964</td><td>0.939</td></tr><tr><td>Cambrian-1-8B</td><td>0.895</td><td>0.901</td><td>0.957</td><td>0.934</td><td>0.886</td><td>0.889</td><td>0.860</td><td>0.892</td><td>0.838</td></tr><tr><td>MODA-8B</td><td>0.972</td><td>0.976</td><td>0.992</td><td>0.985</td><td>0.970</td><td>0.972</td><td>0.970</td><td>0.969</td><td>0.945</td></tr><tr><td colspan="10">Cognition-Specialized</td></tr><tr><td>MMRole-9B</td><td>0.994</td><td>0.998</td><td>1.000</td><td>0.997</td><td>0.993</td><td>0.987</td><td>1.000</td><td>0.992</td><td>0.988</td></tr><tr><td>MODA-8B</td><td>0.995</td><td>1.000</td><td>1.001</td><td>0.999</td><td>0.993</td><td>0.989</td><td>1.001</td><td>0.991</td><td>0.988</td></tr><tr><td>MMRole-9B (In-Test)</td><td>0.999</td><td>1.000</td><td>1.000</td><td>0.999</td><td>0.997</td><td>0.989</td><td>1.012</td><td>0.997</td><td>0.997</td></tr><tr><td>MODA-8B (In-Test)</td><td>1.000</td><td>1.002</td><td>1.001</td><td>1.000</td><td>0.998</td><td>0.992</td><td>1.013</td><td>0.996</td><td>0.996</td></tr><tr><td>MMRole-9B (Out-Test)</td><td>0.981</td><td>0.992</td><td>0.999</td><td>0.993</td><td>0.979</td><td>0.981</td><td>0.963</td><td>0.977</td><td>0.962</td></tr><tr><td>MODA-8B (Out-Test)</td><td>0.984</td><td>0.995</td><td>1.002</td><td>0.996</td><td>0.981</td><td>0.983</td><td>0.970</td><td>0.980</td><td>0.965</td></tr></table>

Table 3. Comparison of MODA with other leading MLLMs and cognition task-specialized methods on MMRole benchmark. The numbers of leading MLLMs come from (Dai et al., 2025).

<table><tr><td rowspan="2">Model</td><td colspan="9">Emotion</td></tr><tr><td>Avg</td><td> $MVSA^S (ACC)$ </td><td> $MVSA^S (F1)$ </td><td> $MVSA^M (ACC)$ </td><td> $MVSA^M (F1)$ </td><td>TumEmo (ACC)</td><td>TumEmo (F1)</td><td>HFM (ACC)</td><td>HFM (F1)</td></tr><tr><td>GPT-4V</td><td>0.633</td><td>0.507</td><td>0.570</td><td>0.609</td><td>0.631</td><td>0.608</td><td>0.612</td><td>0.764</td><td>0.765</td></tr><tr><td>Gemini 1.0 Pro</td><td>0.646</td><td>0.634</td><td>0.637</td><td>0.699</td><td>0.657</td><td>0.598</td><td>0.582</td><td>0.674</td><td>0.683</td></tr><tr><td>Claude 3 Opus</td><td>0.628</td><td>0.626</td><td>0.613</td><td>0.635</td><td>0.629</td><td>0.580</td><td>0.574</td><td>0.679</td><td>0.687</td></tr><tr><td>QWen-VL-Max</td><td>0.643</td><td>0.647</td><td>0.645</td><td>0.669</td><td>0.627</td><td>0.565</td><td>0.595</td><td>0.696</td><td>0.701</td></tr><tr><td colspan="10">Base: Llama-3-Ins-8B</td></tr><tr><td>Mini-Gemini-HD-8B</td><td>0.482</td><td>0.423</td><td>0.571</td><td>0.487</td><td>0.643</td><td>0.246</td><td>0.395</td><td>0.498</td><td>0.593</td></tr><tr><td>LLaVA-NeXT-8B</td><td>0.576</td><td>0.591</td><td>0.593</td><td>0.617</td><td>0.607</td><td>0.547</td><td>0.533</td><td>0.572</td><td>0.551</td></tr><tr><td>Cambrian-1-8B</td><td>0.547</td><td>0.694</td><td>0.661</td><td>0.662</td><td>0.579</td><td>0.439</td><td>0.344</td><td>0.512</td><td>0.487</td></tr><tr><td>MODA-8B</td><td>0.588</td><td>0.702</td><td>0.705</td><td>0.628</td><td>0.619</td><td>0.559</td><td>0.548</td><td>0.585</td><td>0.563</td></tr><tr><td colspan="10">Emotion-Specialized</td></tr><tr><td> $M^2CL$ </td><td>-</td><td>0.755</td><td>0.742</td><td>0.732</td><td>0.705</td><td>0.688</td><td>0.687</td><td>-</td><td>-</td></tr><tr><td>MULSER</td><td>-</td><td>0.757</td><td>0.755</td><td>0.739</td><td>0.738</td><td>0.775</td><td>0.775</td><td>-</td><td>-</td></tr><tr><td>CMGCN</td><td>-</td><td>0.733</td><td>0.720</td><td>0.697</td><td>0.683</td><td>-</td><td>-</td><td>0.875</td><td>0.841</td></tr><tr><td>SPFVTE</td><td>-</td><td>0.806</td><td>0.801</td><td>0.799</td><td>0.788</td><td>-</td><td>-</td><td>0.883</td><td>0.879</td></tr><tr><td>MODA-8B</td><td>0.841</td><td>0.810</td><td>0.803</td><td>0.802</td><td>0.790</td><td>0.778</td><td>0.778</td><td>0.885</td><td>0.881</td></tr></table>

Table 4. Comparison of MODA with other leading MLLMs as well as emotion task-specialized methods on four emotion benchmarks. The reported numbers of emotion-specialized methods come from their official manuscripts. The missed average performance of emotion-specialized methods due to missed datasets.

# 4.4. Results

As shown in Table 2, Table 3, and Table 4, we demonstrate the main results on 21 popular benchmarks for multimodal perception, cognition, and emotion tasks, respectively.

![](images/e3ec3a933bbaefa7bf288b60c36adb788915f529f47634467e765127fc2b96ee.jpg)  
Figure 4. Analysis of multimodal attention and MODA. (a) Multimodal attention: average difference between self and cross-modal attention is 56.97% for text and 62.44% for visual. (b) MODA: average difference is 50.31% for text and 41.01% for visual.

• Perception Benchmark. To assess the effectiveness of our proposed model, we compare it against state-of-the-art Multimodal Large Language Models (MLLMs), including the Mini-Gemini-HD series, the LLaVA-NeXT series, and the Cambrian-1 series. We conduct a comparison under two settings, where we tune these MLLMs from 8B and 34B scale large foundation models. Our proposed MODA outperforms other models of similar scale, including LLaVA-NeXT and Cambrian, achieving an average improvement of 1.0 for the base Llama-3-Ins-8B model and 0.9 for the base Hermes2-Yi-34B model. In vision-centric and OCR tasks, which require fine-grained understanding, MODA consistently performs better, achieving a metric of 66.0 for the vision-centric average and 74.7 for OCR & Chart tasks. This highlights the model’s ability in tasks demanding finegrained perception, further reinforcing its superiority.

• Cognition-specific & Emotion-specific Benchmark. We evaluate a diverse set of MLLMs on both cognitioncentric and emotion-centric benchmarks, designed to assess key dimensions of cognitive and emotional understanding across various aspects. On the cognition benchmark, MODA outperforms open-ended models of Cambrian-1 (0.981) and LLaVA-NeXT (0.979), achieving an average score of 0.996, and performs comparably to close-ended SOTA models like Claude 3 Opus (0.995). With cognition task-specific tuning, MODA achieves better performance, particularly excelling in fluency (0.999) and personality consistency (0.998). On emotion benchmarks, MODA outperforms open-ended models like Cambrian-1 (0.628) and LLaVA-NeXT (0.624), with an average score of 0.657, and achieves comparable performance to task-specialized SOTA models like SPFVTE (0.738) and MULSER (0.739). Notably, MODA excels in TumEmo (0.747) and HFM (0.753) benchmarks, demonstrating strong performance in emotion recognition tasks due to its ability to capture fine-grained emotional features and fine-grained details.

![](images/f782998f798f1573cb2b15d6c74c4d4d4d18685fc4dca8555b8d32363db74c55.jpg)

<details>
<summary>text_image</summary>

ROSES ARE RED
BUG IN YO BED
</details>

Human: Why is this image funny?

Cambrian-1: The humor in this image comes from the unexpected twist on the classic poem 'Roses are red, violets are blue.' Instead of a romantic or sweet sentiment, the poem is humorously altered to a more mundane and slightly unsettling scenario, which is a bug in the bed. This juxtaposition of a traditional romantic poem with a common, albeit slightly unpleasant, household occurrence creates a comedic effect.

MODA: The image is funny because it features a cartoon character, specifically Homer Simpson from the popular television show "The Simpsons," who is depicted in a humorous and unexpected situation. He is lying in bed with a red rose in his mouth, which is an unusual and comical scenario for the character. The image also includes a caption that reads "Roses are red, bug in yo bed," adding to the humor of the scene. This combination of elements creates a lighthearted and amusing visual that is likely to elicit laughter from viewers familiar with the character and the show.

# (a) Fine-grained Perception

![](images/036bcf963ababa8a6acaea8aadc40ad14b5517db341f1b9fc36de064f6544e50.jpg)

Phoebe Buffay: Look at this photo! Isn't it just like a fairy-tale wedding, until, you know, it wasn't?

Ross Geller played by Cambrian-1: Phoebe, you're right, the wedding does look like a fairy-tale scene, but it's not quite perfect. You see, the bride, Emily, is mistakenly called Rachel by the groom, Ross. It's a bit of a mix-up, but it's all good. It's just one of those little hiccups that happens in real life, you know?

Ross Geller played by MODA: Oh, Phoebe, you're right. It's like a fairy-tale wedding, with beautiful bride and handsome groom, and the elegant setting. But, as we all know, fairy tales don't always have a happy ending. In this case, Ross made a mistake that changed the course of wedding, and it's a moment that will be remembered by everyone. It's a reminder that even in the most perfect of settings, life can throw us curveballs that we have to navigate.

# (b) Cognitive Conversation Analysis

Post: And just like that the weekends gone.   
![](images/64378b7361a8ebdb611a890b5d9d08226b20ce53313bc17b58603f9f6b5ed674.jpg)

Human: What is the sentiment of the above image-text pair?

Cambrian-1: Sad ✅

MODA: Sad ✅

Post: My photogenic little man   
![](images/f490c3bb7f4cc023decc1fa5f877076aa52b37218e2e42bcea0512affd7e6431.jpg)

Human: What is the sentiment of the above image-text pair?

Cambrian-1: Anticipation ❌

MODA: Love ✅

Post: New GIPHY!   
![](images/945e74a5813ca71249968adc6e481c7012744aab757063d09625308bbc7e5783.jpg)

Human: What is the sentiment of the above imagetext pair?

Cambrian-1: Happy ❌

MODA: Angry ✅

# (c) Emotion Understanding

Figure 5. Visualization results of state-of-the-art (SOTA) Multimodal Large Language Models (MLLMs) and our proposed MODA on fine-grained multimodal tasks, including perception, cognition, and emotion understanding.

• Analysis of Attention. We analyze the attention distribution to evaluate the effectiveness of the proposed MODA in terms of attention distribution (Fig. 4). We conduct experiments on fine-grained understanding tasks, i.e., emotion recognition based on visual-textual affective cues. Baseline attention exhibits imbalanced scores, particularly in deeper layers, highlighting its tendency to over-focus on dominant modalities while neglecting others (i.e., visual affective cues). Our MODAachieves consistently high across all layers, preventing degradation and ensuring balanced contributions from all tokens and modalities. This improvement directly correlates with superior performance on multimodal tasks, particularly those requiring fine-grained understanding, such as TumEmo and OCRBench, demonstrating MODA’s ability to avoid attention collapse and capture multimodal interactions effectively.

• Visualization To highlight the advantages of Modular Duplex Attention (MODA), we design a visualization experiment focusing on output answers, showcasing its capacity to generate fine-grained and accurate responses. The output answers (Figure 5) further demonstrate MODA’s superior capacity for fine-grained understanding. By accurately capturing the humorous twist on the classic poem, MODA has demonstrated its unparalleled ability to reason about fine-grained multimodal details. These results underscore MODA’s transformative potential in advancing the state-ofthe-art in multimodal understanding, where it seamlessly integrates information across modalities to achieve exceptional precision in complex cognitive and emotional tasks.

Johnny: A month ago he bought the rights to this book, a best seller. The main character is a guy just like me. I wouldn’t even have to act, just be myself. Oh Godfather, I don’t know what to do, I don’t know what to do [crying]

![](images/bcffdc02752e057adef9b7d82152e4077fce3bf3a3998ce9bf755389de9b9c66.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man in formal attire viewed from behind, framed within a blue bounding box (no visible text or symbols)
</details>

Human: How is the Godfather looks like?

Cambrian-1: The Godfather, as depicted in the image, is portrayed by an actor in a classic black tuxedo, which is a traditional and iconic look for the character.

MODA: The Godfather is dressed in a classic black tuxedo, which is emblematic of his role as a powerful and respected figure in the film. His expression is one of concern, as he listens to Johnny's dilemma, indicating his deep involvement in Johnny's life and the weight of his responsibilities.

Human: What does this scene reveal about the Godfather’s influence over people in his inner circle?

Cambrian-1: This scene from 'The Godfather' illustrates the profound impact the character, Don Corleone, has on those around him. It shows how his influence extends beyond his immediate family, affecting even those who are not directly related to him.

MODA: This scene reveals the Godfather's influence over people in his inner circle as a significant overwhelming force. The Godfather's presence in the room, along with his stern demeanor, further emphasizes his control over the situation and the characters around him. The scene captures the tension and emotional turmoil that can arise from being under the Godfather's influence, highlighting the power dynamics at play within the narrative.

# (a) Understanding Godfather

Human: How might the Godfather act to proceed with the next conversation? Give me a few possible options.

Cambrian-1: Given the context of the image, which suggests a scene from "The Godfather," the character portrayed by the man in the foreground might proceed with the next conversation by either continuing to listen intently to the other character, who seems to be speaking, or by interjecting with a question or comment to steer the conversation in a particular direction.

MODA: The Godfather, upon witnessing Johnny's distress, might respond with a combination of empathy and strategic thinking. He could offer a comforting word, such as, "It's a lot to take in, but we'll figure it out together.” Alternatively, he might shift the conversation to a more practical matter, like discussing the logistics of the book's adaptation or the potential benefits of the project. Another possibility is that he might offer advice on how to handle the situation, such as seeking legal counsel or consulting with other members of the family. The Godfather's response would likely be measured and thoughtful, reflecting his role as a leader.

![](images/675d59995faa997f7abb2f18428c31041174fd248118ed0ab5ce737587f1e887.jpg)

![](images/a0d80c3f3c5014a6d2f931b92a008006e81fbc56b7a15bfea6866db967116ea6.jpg)

![](images/14071d8a2e9c9a0eb941bfa5c324ff2b1a4977474288dc86306206c7e7c1b809.jpg)

# (b) Planning for Godfather

Figure 6. MODA-enabled apps in The Godfather. (a) With a deep and fine-grained understanding of conversation, MODA captures both the emotional and cognitive states of the character. (b) MODA further simulates the Godfather’s strategic thinking, planning the next steps while considering the character’s traits.

# 5. Discussion

MODA-based MLLM can boost various downstream directions. Here, we envision two potential uses.

• Godfather-centric Understanding. Leveraging the fine-grained understanding of multimodal content, MODA demonstrates enhanced comprehension capabilities that facilitate human-centric interpretation. Integrated with the strong sensory grounding capacity, MODA can effectively process and interpret high-level human cognition and emotion. Notably, we present the first demonstration of a Multimodal Large Language Model (MLLM) that captures emotions expressed by individuals, going beyond simple classification into discrete categories. As illustrated in Fig.6 (a), MODA successfully captures the micro-expressions of the Godfather, reflecting the character’s personality and the cultural context embedded within the scene.

• Conversation with Godfather. Further, the comprehensive understanding enabled by MODA enhances its capacity to plan conversations. We integrate MODA into a conversation system between the human and the agent in three stages: description, analysis, and planning. As shown in Fig.6 (b), MODA takes the historical conversation context (e.g., video keyframe, transcript, and audio) as input and generates the desired target (e.g., behavior, characteristics, and emotion). This allows the model to simulate a conversation flow, as in the Godfather scenario, where the agent responds with strategic plan based on the established emotional context.

# 6. Conclusion

This work introduces the MOdular Duplex Attention (MODA) to tackle attention deficit disorder in multimodal large language models, characterized by inconsistent crossmodal attention and layer decay. MODA enhances multimodal perception, cognition, and emotion understanding by modularly processing diverse data streams, outperforming existing MLLMs across 21 benchmark datasets. This advancement not only improves modality alignment but also supports deeper cognitive and emotional insights, with source code and demo available for further exploration.

# Acknowledgement

This work was supported by the National Natural Science Foundation of China (No. 623B2056), the Natural Science Foundation of Tianjin, China (No.24JCZXJC00040), the Fundamental Research Funds for the Central Universities, the Supercomputing Center of Nankai University (NKSC). We sincerely thank the reviewer team (cYUZ, XinM, Mf8z, and Ack2) for their invaluable feedback to improve our manuscript.

# Impact Statement

This paper introduces a novel multimodal attention mechanism designed to enhance Multimodal LLMs for finegrained content understanding. However, as with most MLLMs, the quality of MODA’s output is influenced by the fine-tuning data and the quality of the base models, which may result in the generation of low-quality or hallucinated content. Such outputs could potentially be harmful, and users are advised to interpret the results with caution, adhering to licensing restrictions, with commercial use explicitly prohibited. All the personal information is anonymized or obfuscated to ensure confidentiality.

# References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. In arXiv, 2023.   
Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022.   
Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A frontier large vision-language model with versatile abilities. In arXiv, 2023.   
Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. In arXiv, 2020.   
Binz, M. and Schulz, E. Using cognitive psychology to understand gpt-3. PNAS, 120(6):e2218523120, 2023.   
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. In NeurIPS, 2020.   
Chen, J., Wang, X., Xu, R., Yuan, S., Zhang, Y., Shi, W., Xie, J., Li, S., Yang, R., Zhu, T., et al. From persona to personalization: A survey on role-playing language agents. TMLR, 2024.   
Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. https://vicuna.lmsys. org, 2023.   
Child, R., Gray, S., Radford, A., and Sutskever, I. Generating long sequences with sparse transformers. In arXiv, 2019.   
Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin,

A., Kaiser, L., et al. Rethinking attention with performers. In ICLR, 2021.   
Dai, Y., Hu, H., Wang, L., Jin, S., Chen, X., and Lu, Z. Mmrole: A comprehensive framework for developing and evaluating multimodal role-playing agents. In ICLR, 2025.   
Deshpande, A., Murahari, V., Rajpurohit, T., Kalyan, A., and Narasimhan, K. Toxicity in chatgpt: Analyzing persona-assigned language models. In EMNLP, 2023.   
Dong, Y., Cordonnier, J.-B., and Loukas, A. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In ICML, 2021.   
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.   
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. In arXiv, 2024.   
Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In arXiv, 2023.   
Han, D., Ye, T., Han, Y., Xia, Z., Pan, S., Wan, P., Song, S., and Huang, G. Agent attention: On the integration of softmax and linear attention. In ECCV, 2024.   
Huang, S., Xu, B., Li, C., Ye, J., and Lin, X. A sentimental prompt framework with visual text encoder for multimodal sentiment analysis. In ICMR, 2024.   
Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.   
Kembhavi, A., Salvato, M., Kolve, E., Seo, M., Hajishirzi, H., and Farhadi, A. A diagram is worth a dozen images. In ECCV, 2016.   
Kitaev, N., Kaiser, Ł., and Levskaya, A. Reformer: The efficient transformer. In ICLR, 2020.   
Kong, A., Zhao, S., Chen, H., Li, Q., Qin, Y., Sun, R., Zhou, X., Wang, E., and Dong, X. Better zero-shot reasoning with role-play prompting. In NAACL, 2024.   
Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., and Shan, Y. Seed-bench: Benchmarking multimodal large language models. In CVPR, 2024.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.   
Lian, Z., Liu, B., and Tao, J. Smin: Semi-supervised multimodal interaction network for conversational emotion recognition. TAC, 2022.   
Liu, H., Wang, W., and Li, H. Towards multi-modal sarcasm detection via hierarchical congruity modeling with knowledge enhancement. In EMNLP, 2022.   
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In NeurIPS, 2023.   
Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.-C., Liu, C.-L., Jin, L., and Bai, X. Ocrbench: on the hidden mystery of ocr in large multimodal models. SCIS, 67(12):220102, 2024.   
Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2025.   
Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In ICLR, 2019.   
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022.   
Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.   
Mai, S., Zeng, Y., Zheng, S., and Hu, H. Hybrid contrastive learning of tri-modal representation for multimodal sentiment analysis. TAC, 2022.   
Masry, A., Long, D. X., Tan, J. Q., Joty, S., and Hoque, E. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, 2022.   
Mathew, M., Karatzas, D., and Jawahar, C. Docvqa: A dataset for vqa on document images. In WACV, 2021.   
Niu, T., Zhu, S., Pang, L., and El-Saddik, A. Sentiment analysis on multi-view social data. In MMM, 2016.   
Peebles, W. and Xie, S. Scalable diffusion models with transformers. In ICCV, 2023.   
Pessoa, L. The entangled brain: How perception, cognition, and emotion are woven together. MIT Press, 2022.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In ICML, 2021.   
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.   
Ryu, J., Han, D., and Lim, J. Gramian attention heads are strong yet efficient vision learners. In ICCV, 2023.   
Salemi, A., Mysore, S., Bendersky, M., and Zamani, H. LaMP: When large language models meet personalization. In ACL, 2024.   
Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In CVPR, 2019.   
Teknium, R., Quesnelle, J., and Guang, C. Hermes 3 technical report. In arXiv, 2024.   
Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S. C., Yang, J., Yang, S., Iyer, A., Pan, X., et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, 2024a.   
Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In CVPR, 2024b.   
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., \` Azhar, F., et al. Llama: Open and efficient foundation language models. In arXiv, 2023.   
Wang, C., Luo, W., Dong, S., Xuan, X., Li, Z., Ma, L., and Gao, S. Mllm-tool: A multimodal large language model for tool agent learning. In WACV, 2025.   
Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. In arXiv, 2020.   
Wang, X., Xiao, Y., Huang, J.-t., Yuan, S., Xu, R., Guo, H., Tu, Q., Fei, Y., Leng, Z., Wang, W., Chen, J., Li, C., and Xiao, Y. InCharacter: Evaluating personality fidelity in role-playing agents through psychological interviews. In ACL, 2024a.   
Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Nexttoken prediction is all you need. In arXiv, 2024b.   
Wang, Z. M., Peng, Z., Que, H., Liu, J., Zhou, W., Wu, Y., Guo, H., Gan, R., Ni, Z., Yang, J., Zhang, M., Zhang, Z., Ouyang, W., Xu, K., Huang, S. W., Fu, J., and Peng, J. Rolellm: Benchmarking, eliciting, and enhancing roleplaying abilities of large language models. In ACL, 2024c.

Wu, X., Ajorlou, A., Wang, Y., Jegelka, S., and Jadbabaie, A. On the role of attention masks and layernorm in transformers. In NeurIPS, 2024.   
xAI. grok. https://x.ai/blog/grok-1.5v, 2024.   
Yang, J., She, D., Lai, Y.-K., Rosin, P. L., and Yang, M.-H. Weakly supervised coupled networks for visual sentiment analysis. In CVPR, 2018.   
Yang, Q., Ye, M., and Du, B. Emollm: Multimodal emotional understanding meets large language models. In arXiv, 2024.   
Yang, X., Feng, S., Wang, D., and Zhang, Y. Image-text multimodal emotion classification via multi-view attentional network. TMM, 23:4014–4026, 2021.   
Yang, X., Feng, S., Wang, D., Hong, P., and Poria, S. Multiple contrastive learning for multimodal sentiment analysis. In ICASSP, 2023.   
Yin, Q., He, X., Zhuang, X., Zhao, Y., Yao, J., Shen, X., and Zhang, Q. Stablemask: refining causal masking in decoder-only transformer. In ICML, 2024.   
Young, A., Chen, B., Li, C., Huang, C., Zhang, G., Zhang, G., Wang, G., Li, H., Zhu, J., Chen, J., et al. Yi: Open foundation models by 01.ai. In arXiv, 2024.   
Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.   
Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. Big bird: Transformers for longer sequences. In NeurIPS, 2020.   
Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., and Weston, J. Personalizing dialogue agents: I have a dog, do you have pets too? In ACL, 2018.   
Zhang, Z. and Yang, J. Temporal sentiment localization: Listen and look in untrimmed videos. In ACM MM, 2022.   
Zhang, Z., Wang, L., and Yang, J. Weakly supervised video emotion detection and prediction via cross-modal temporal erasing network. In CVPR, 2023.   
Zhang, Z., Zhao, P., Park, E., and Yang, J. Mart: Masked affective representation learning via masked temporal distribution distillation. In CVPR, 2024.   
Zhao, S., Jia, G., Yang, J., Ding, G., and Keutzer, K. Emotion recognition from multiple modalities: Fundamentals and methodologies. SPM, 38(6):59–73, 2021.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In ICLR, 2024.