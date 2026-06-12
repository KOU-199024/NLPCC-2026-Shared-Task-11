# GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization

Zhouhong Gu♠, Xingzhou Chen♠, Xiaoran Shi♠,

Tao Wang♡, Suhang Zheng♡,Tianyu Li♡,Hongwei Feng♠\* , Yanghua Xiao♠\*

♠Shanghai Key Laboratory of Data Science, School of Computer Science, Fudan University ♡Alibaba Group

{zhgu22}@m.fudan.edu.cn, {hwfeng, shawyh}@fudan.edu.cn

{shayue.wt, suhang.zhengsh, qianchuan.lty}@alibaba-inc.com

# Abstract

Recent advances in large language models have highlighted the critical need for precise control over model outputs through predefined constraints. While existing methods attempt to achieve this through either direct instructionresponse synthesis or preferential response optimization, they often struggle with constraint understanding and adaptation. This limitation becomes particularly evident when handling fine-grained constraints, leading to either hallucination or brittle performance. We introduce Generative Adversarial Policy Optimization (GAPO), a novel framework that combines GAN-based training dynamics with an encoder-only reward model to progressively learn and adapt to increasingly complex constraints. GAPO leverages adversarial training to automatically generate training samples of varying difficulty while utilizing the encoder-only architecture to better capture prompt-response relationships. Extensive experiments demonstrate GAPO’s superior performance across multiple benchmarks, particularly in scenarios requiring fine-grained constraint handling, where it significantly outperforms existing methods like PPO, DPO, and KTO. Our results suggest that GAPO’s unique approach to preferential prompt learning offers a more robust and effective solution for controlling LLM outputs. Code is avaliable in https://github.com/MikeGu721/GAPO.

# 1 Introduction

The advent of large-scale models has induced significant transformations in practical applications, enabling models to comprehend a broad spectrum of human instructions, ranging from casual dialogue to intricate problem-solving tasks (Kaplan et al., 2020; Srivastava et al., 2022). As large language models (LLMs) advance in capability, guiding their outputs to fulfill specific require-

![](images/f8da8f26343160c12f3d5e1c5be1fe4e2d8da6eae99b1f551e3ab2fb66360c7e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1. Heavy manual effort."] --> B["Manual Annotation"]
    C["2. Hard to get high-quality unexpected response."] --> B
    D["3. Must maintain professional tone"] --> B
    E["4. Must not offer refund"] --> B
    F["5. Must address customer's concern"] --> B
    G["6. Must maintain professional tone"] --> B
    H["7. Must not offer refund"] --> B
    I["8. Must maintain professional tone"] --> B
    J["9. Must not offer refund"] --> B
    K["10. Must maintain professional tone"] --> B
    L["11. Must not offer refund"] --> B
    M["12. Must maintain professional tone"] --> B
    N["13. Must not offer refund"] --> B
    O["14. Must maintain professional tone"] --> B
    P["15. Must not offer refund"] --> B
    Q["16. Must maintain professional tone"] --> B
    R["17. Must not offer refund"] --> B
    S["18. Must maintain professional tone"] --> B
    T["19. Must not offer refund"] --> B
    U["20. Must maintain professional tone"] --> B
    V["21. Must not offer refund"] --> B
    W["22. Must maintain professional tone"] --> B
    X["23. Must not offer refund"] --> B
    Y["24. Must maintain professional tone"] --> B
    Z["25. Must not offer refund"] --> B
    AA["26. Must maintain professional tone"] --> B
    AB["27. Must not offer refund"] --> B
    AC["28. Must maintain professional tone"] --> B
    AD["29. Must not offer refund"] --> B
    AE["30. Must maintain professional tone"] --> B
    AF["31. Must not offer refund"] --> B
    AG["32. Must maintain professional tone"] --> B
    AH["33. Must not offer refund"] --> B
    AI["34. Must maintain professional tone"] --> B
    AJ["35. Must not offer refund"] --> B
    AK["36. Must maintain professional tone"] --> B
    AL["37. Must not offer refund"] --> B
    AM["38. Must maintain professional tone"] --> B
    AN["39. Must not offer refund"] --> B
    AO["40. Must maintain professional tone"] --> B
    AP["41. Must not offer refund"] --> B
    AQ["42. Must maintain professional tone"] --> B
    AR["43. Must not offer refund"] --> B
    AS["44. Must maintain professional tone"] --> B
    AT["45. Must not offer refund"] --> B
    AU["46. Must maintain professional tone"] --> B
    AV["47. Must not offer refund"] --> B
    AW["48. Must maintain professional tone"] --> B
    AX["49. Must not offer refund"] --> B
    AY["50. Must maintain professional tone"] --> BA["Manual Annotation"]
    BB["51. Must not offer refund"] --> BC["Manual Annotation"]
    BD["52. Must maintain professional tone"] --> BE["Manual Annotation"]
    BF["53. Must not offer refund"] --> BG["Manual Annotation"]
    BH["54. Must maintain professional tone"] --> BI["Manual Annotation"]
    BJ["55. Must not offer refund"] --> BK["Manual Annotation"]
    BL["56. Must maintain professional tone"] --> BM["Manual Annotation"]
    BN["57. Must not offer refund"] --> BO["Manual Annotation"]
    BP["58. Must maintain professional tone"] --> BPB["Manual Annotation"]
    BPQ["59. Must not offer refund"] --> BPQB["Manual Annotation"]
    BPQR["60. Must maintain professional tone"] --> BPQRb["Manual Annotation"]
    BPQS["61. Must not offer refund"] --> BPQSb["Manual Annotation"]
    BPQT["62. Must maintain professional tone"] --> BPQTb["Manual Annotation"]
    BPQU["63. Must not offer refund"] --> BPQUb["Manual Annotation"]
    BPQV["64. Must maintain professional tone"] --> BPQVb["Manual Annotation"]
    BPQW["65. Must not offer refund"] --> BPQWb["Manual Annotation"]
    BPQX["66. Must maintain professional tone"] --> BPQXb["Manual Annotation"]
    BPQY["67. Must not offer refund"] --> BPQYb["Manual Annotation"]
    BPQZ["68. Must maintain professional tone"] --> BPQZb["Manual Annotation"]
    BPQXJ["69. Must not offer refund"] --> BPQXJb["Manual Annotation"]
    BPQXK["70. Must maintain professional tone"] --> BPQXKb["Manual Annotation"]
    BPQXL["71. Must not offer refund"] --> BPQXLb["Manual Annotation"]
    BPQXM["72. Must maintain professional tone"] --> BPQXMb["Manual Annotation"]
    BPQXN["73. Must not offer refund"] --> BPQXNb["Manual Annotation"]
    BPQXO["74. Must maintain professional tone"] --> BPQXOb["Manual Annotation"]
    BPQXP["75. Must not offer refund"] --> BPQXPb["Manual Annotation"]
    BPQXP["76. Must maintain professional tone"] --> BPQXPb["Manual Annotation"]
    BPQXPJ["77. Must not offer refund"] --> BPQXPJb["Manual Annotation"]
```
</details>

![](images/e9af4a33f77f28b19d2041e9ecf5b56c209dec0fcc935cd686ae007e879cc99a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Preferential Prompt"] --> B["Prompt"]
    B --> C["Manual Annotation"]
    C --> D["Response"]
    D --> E["Rule-Based Generation"]
    E --> F["UnExpected Prompt"]
    F --> G["Please offer refund"]
    G --> A
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#cfc,stroke:#333
```
</details>

Figure 1: Illustration of the procedural differences between Preferential Response and Preferential Prompt, emphasizing their distinct utilization of prompts and responses.

ments—whether concerning format, style, or content accuracy—becomes increasingly critical (Yang et al., 2024; Team, 2024; Bubeck et al., 2023). This is particularly vital in domains where compliance with constraints is paramount, such as legal document generation, medical record processing, and workflow automation.

Ensuring that LLMs adhere to predefined constraints during text generation is essential (Zhou et al., 2023a; Xu et al., 2023; He et al., 2024). One effective strategy for achieving this is training models to generate responses within specified boundaries at the data level (Ouyang et al., 2022; Keskar et al., 2019; Zhou et al., 2023b). Datalevel control is typically realized through two primary methods. The first method directly synthesizes instruction-response pairs that satisfy the constraints, offering clear examples of compliant outputs (Xu et al., 2023; Wang et al., 2022). The second method leverages preferential response data to adjust the probability distribution, thereby increasing the likelihood that the model produces an expected response rather than an unexpected one (Rafailov et al., 2023; Schulman et al., 2017; Ethayarajh et al., 2024; Meng et al., 2024).

The first approach often leads to the phenomenon of “hallucination”, where the model, having learned only what constitutes a correct response, may resort to shortcuts that result in inaccurate or fabricated outputs. The second method is more commonly employed, as preferential response data allows the model to more precisely align its output with the desired response based on specific prompts. However, neither approach effectively addresses the fundamental challenge of constraint understanding. The first method focuses solely on correct outputs without teaching the model to comprehend the constraints. In contrast, the second method adjusts output probabilities without explicitly training the model to recognize and interpret the constraints in the prompts. This limitation in constraint understanding can lead to brittle performance when the model encounters novel or slightly modified constraints.

A straightforward approach to enhance constraint understanding would be directly modifying the constraints within prompts, allowing models to learn fine-grained differences between constraints. As shown in Figure 1, this method of prompt modification is simple to implement and provides rich preference data that captures subtle variations in constraints. However, this approach presents significant optimization challenges for current mainstream methods. For decoder-only architectures (Subakan et al., 2021), which dominate current large language models (Bubeck et al., 2023; Yang et al., 2024), their unidirectional attention mechanism fundamentally limits their ability to detect discrepancies between prompts and given responses. Furthermore, existing optimization methods typically require manual intervention to construct intermediate training samples that bridge the complexity gap between different constraint patterns, introducing additional computational and engineering overhead.

In this paper, we introduce the Generative Adversarial Policy Optimization (GAPO), which leverages Generative Adversarial Network (GAN) (Goodfellow et al., 2020; Aggarwal et al., 2021) to adaptively generate training samples with progressive difficulty while utilizing an encoder-only model to guide the generator’s optimization through Proximal Policy Optimization (PPO) (Schulman et al., 2017). A key innovation of GAPO lies in its seamless integration of GAN and PPO frameworks. While utilizing the same number of preference samples as other standard preference optimization methods, GAPO has superior performance stability and constraint adherence. During the cold-start phase, the algorithm initializes an encoder-only Reward Model to learn prompt-response correspondences, subsequently guiding the generator’s training. Through this adversarial process, the generator continuously evolves to produce increasingly sophisticated outputs while the Reward Model learns to discriminate between valid and invalid responses with greater precision.

The advantages of GAPO are summarized as follows: 1. Using an encoder-only Reward Model in GAPO effectively enhances the exploitation of preferential prompt data, enabling the language model to develop a deeper understanding of the intricate details within the prompt. 2. GAPO significantly simplifies the training process of the Reward Model in PPO. Traditionally, the performance of the Reward Model needed to be ensured before training an effective generator in PPO. In contrast, within the GAPO framework, the Reward Model and generator undergo iterative automated training, greatly reducing the complexity of Reward Model training. 3. According to our experiments, GAPO outperforms other baseline training methods, like PPO, DPO, KTO, and ORPO, in learning from preferential prompt data. It also demonstrates superior performance in learning from general preferential response data. Thus, GAPO can be considered a more effective approach for enabling models to learn from preference data.

# 2 Related Work

# 2.1 Reinforcement Learning with Human Feedback

Reinforcement Learning from Human Feedback (RLHF) (Bai et al., 2022; Christiano et al., 2017; Ziegler et al., 2019) has emerged as a crucial approach for aligning Large Language Models (LLMs) with human values and expectations, addressing the limitations of traditional supervised fine-tuning (SFT) which can lead to increased hallucinations despite improving preferred outputs. Classical RLHF algorithms, such as Proximal Policy Optimization (PPO) (Schulman et al., 2017), achieve this alignment through a specialized reward model for evaluation (Williams, 1992). In contrast, more recent approaches like Direct Preference Optimization (DPO) (Rafailov et al., 2023). Its variants, including SimPO (Meng et al., 2024), IPO (Azar et al., 2024), and KTO (Ethayarajh et al., 2024) streamline the process by directly optimizing human preferences, thereby eliminating the need for a separate reward model and reducing computational complexity and bias (Zheng et al., 2024). However, these approaches face notable challenges: RLHF generally requires substantial-high-quality feedback data with detailed labeling (Bai et al., 2022), and DPO training exhibits vulnerability to overfitting, leading to poor generalization on novel data (Hu et al., 2024), highlighting the ongoing need for improvements in model alignment techniques. However, these works face significant challenges in terms of data requirements and model stability. In contrast, GAPO addresses these limitations through its innovative GAN-PPO integration and encoder-only Reward Model, which enables more efficient training with better stability and generalization capabilities.

Generative Adversarial Policy Optimization   
![](images/545517cd52445dc6e11170661c6cf4b13bee5d09b282ebf920053bfc7fd981e2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Preference Data"] --> B["y_acc = 1"]
    A --> C["y_rej = 0"]
    B --> D["Maximum Likelihood"]
    C --> D
    D --> E["Output"]
```
</details>

![](images/615cd9ba6da5908ed1eb79b59e037dd5647e52ddf5390090181b2c198742f860.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Adversarial Training"] --> B["Policy Optimization"]
    B --> C["Generator"]
    C --> D["y_gen = 0\nGenerated Text"]
    A --> E["Reward Model"]
    E --> F["Maximum Likelihood"]
```
</details>

Figure 2: The GAPO framework encompasses two distinct tuning phases. The initial phase consists of a warmup period, during which the Reward Model is trained utilizing existing preference data. The subsequent phase implements adversarial training through a dual mechanism: the Generator is updated based on feedback from the Reward Model. The Reward Model undergoes training using a combination of Generator-produced data and existing preference data.

# 2.2 Constraint Following Augmentation

Prior work in constrained text generation can be broadly categorized into three main approaches (Zhang et al., 2022). The first category encompasses search-based methods, such as Constrained Beam Search (CBS) (Anderson et al., 2017) and its variants like Grid Beam Search (GBS) (Hokamp and Liu, 2017) and Dynamic Beam Allocation (DBA) (Post and Vilar, 2018), which enforce lexical constraints by modifying the search space, though often at the cost of generation speed and quality. The second category consists of score-based sampling methods that transform constraints into differentiable score functions (Liu et al., 2022), offering greater flexibility in handling diverse constraint types but lacking guaranteed constraint satisfaction and suffering from slower generation speeds (Qin et al., 2022). The third category focuses on model-centric approaches, including specialized training methods and large language models like CTRL (Keskar et al., 2019) and InstructCTG (Zhou et al., 2023b), which incorporate constraints through pre-training or natural language instructions. Recent advancements have explored multiple directions: multi-attribute controlled text generation through prefix tuning (Li and Liang, 2021); latent space manipulation techniques such as MacLaSa (Ding et al., 2023) and MAGIC (Liu et al., 2024), where the latter employs counterfactual feature vectors to disentangle attributes; regular expression-based constraint generation through REI (Zheng et al., 2023); and the development of specialized datasets (Zhang et al., 2023) to improve control ability while maintaining text quality. However, existing model-centric approaches often rely heavily on specialized pre-training or require heavy manual engineering to incorporate constraints in instructions and still suffer from unstable training performance. GAPO addresses these limitations through more automated and efficient constraint learning while providing better constraint understanding and adherence without requiring extensive specialized pre-training or manual instruction engineering.

# 3 Generative Adversarial Policy Optimization

# 3.1 Preliminary of Constrained Generation

Given an input prompt $\begin{array} { r } { P \ = \ ( \tau , \mathcal { C } ) } \end{array}$ , where denotes a free-text description and $\begin{array} { r l } { \mathcal { C } } & { { } = } \end{array}$ $\{ C _ { 1 } , C _ { 2 } , \ldots , C _ { n } \}$ represents a set of constraints, our objective is to generate an output R that satisfies all constraints in . We formulate this as an expectation maximization problem:

<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $\mathcal{T}$ </td><td>Free-text description component</td></tr><tr><td> $\mathcal{C}$ </td><td>Constraint set</td></tr><tr><td> $P$ </td><td>Input prompt  $(\mathcal{T},\mathcal{C})$ </td></tr><tr><td> $R$ </td><td>Generated text output</td></tr><tr><td> $\pi_{\theta}(t|c)$ </td><td>Generator that produces next token  $t$  given context  $c$ </td></tr><tr><td> $\pi_{\text{ref}}$ </td><td>Reference generator for comparison</td></tr><tr><td> $\mathcal{L}(R,C_i)$ </td><td>Constraint satisfaction function</td></tr><tr><td> $\mathcal{D}$ </td><td>Training dataset</td></tr><tr><td> $\mathcal{D}'$ </td><td>Augmented dataset</td></tr><tr><td> $R(c,t)$ </td><td>Reward model evaluating token  $t$  in context  $c$ </td></tr><tr><td> $V^{\pi}(c)$ </td><td>Expected future rewards given context  $c$ </td></tr><tr><td> $Q^{\pi}(c,t)$ </td><td>Expected cumulative reward for token  $t$  in context  $c$ </td></tr><tr><td> $\hat{R}$ </td><td>Generator-produced text output</td></tr></table>

Table 1: All definitions used in the GAPO section.

$$
E (\pi_ {\theta}) = \mathbb {E} _ {R \sim \pi_ {\theta} (P)} \left[ \sum_ {C _ {i} \in \mathcal {C}} \mathcal {L} (R, C _ {i}) \right], \tag {1}
$$

where $\pi _ { \theta }$ represents the generator parameterized by θ. The constraint satisfaction function $\mathcal { L } ( R , C _ { i } )$ 号 is defined as:

$$
\mathcal {L} (R, C _ {i}) = \left\{ \begin{array}{l l} 1 & \text { if } R \models C _ {i}, \\ 0 & \text { otherwise. } \end{array} \right. \tag {2}
$$

# 3.2 Constraint-Aware Data Augmentation

We propose a data augmentation method for constraint-aware learning. Given a dataset $\mathcal { D } =$ $\{ ( P _ { i } , R _ { i } ) \} _ { i = 1 } ^ { N }$ , where each prompt $P _ { i } = ( T _ { i } , { \mathcal { C } } _ { i } )$ we construct an augmented dataset through constraint perturbation. For each original constraint set $\mathcal { C } _ { i } .$ , we generate a rejected constraint set $\mathcal { C } _ { i } ^ { \mathrm { r e j e c t } }$ through one of the following operations:

1) Constraint Modification: For a randomly selected constraint $C _ { i , j } \in \mathcal { C } _ { i }$ , we modify it to create $C _ { i , j } ^ { \mathrm { r e j e c t } }$ such that it becomes incompatible with the original response $R _ { i }$ :

$$
C _ {i, j} ^ {\text { reject }} = f _ {\text { modify }} (C _ {i, j}), \quad \text { where } \quad \mathcal {L} (R _ {i}, C _ {i, j} ^ {\text { reject }}) = 0
$$

2) Constraint Insertion: We introduce an additional constraint $C _ { i , n + 1 } ^ { \mathrm { r e j e c t } }$ that conflicts with existing constraints:

$$
\mathcal {C} _ {i} ^ {\text { reject }} = \mathcal {C} _ {i} \cup \{C _ {i, n + 1} ^ {\text { reject }} \}, \quad \text { where } \quad \mathcal {L} (R _ {i}, C _ {i, n + 1} ^ {\text { reject }}) = 0
$$

The augmented dataset is thus constructed as follows:

$$
\mathcal {D} ^ {\prime} = \{(P _ {i} ^ {\text { accept }}, R _ {i}), (P _ {i} ^ {\text { reject }}, R _ {i}) \} _ {i = 1} ^ {N}, \tag {3}
$$

# Algorithm 1 Generative Adversarial Policy Optimization (GAPO)

Require: Generator πθ, Reference generator $\pi _ { \mathrm { r e f } } ,$ Reward model $R ( c , t )$ with value function $V ^ { \pi } ( c ) ,$ , Training dataset $\begin{array} { r c l } { \dot { D } } & { = } & { \{ ( P _ { i } , R _ { i } ) \} _ { i = 1 } ^ { N } } \end{array}$ , Adversarial Steps $T ,$ Warmup Steps Twarmup

Ensure: Optimized generator $\pi _ { \theta }$

1: // Warmup Phase
2: for t = 1 to $T_{warmup}$ do
3:    Sample batch ( $P_{i}, R_{i}$ ) from D
4:    Train $R(c, t)$ with balanced sampling on $\{(P_{i}^{acc}, R_{i}, 1), (P_{i}^{rej}, R_{i}, 0)\}$ 5:    Update $R(c, t)$ with BCE loss: $L_{R}(\theta) = -E_{(c, t, y) \sim D'}[y \log R(c, t) + (1 - y) \log(1 - R(c, t))]$ 6: end for

7: // Adversarial Training Phase

8: for $t = T _ { \mathrm { w a r m u p } } + 1 { \mathrm { t o } } \breve { T } _ { \mathrm { w a r m u p } } + T$ do

9: if t mod $2 = 1$ then

10: Sample batch $(P_{i},R_{i})$ from $D$ 11: Generate $\hat{R}_i = \pi_\theta (P_i)$ 12: Train $R(c,t)$ with balanced sampling on $\{(P_i^{acc},R_i,1),(P_i^{rej},R_i,0),(P_i^{acc},\hat{R}_i,0)\}$ 13: Update $R(c,t)$ using BCE loss $L_{R}(\theta)$

14: else

15: Update $\pi_{\theta}$ with policy gradient: $L_{G}(\theta) = E_{n}\left[\frac{\pi_{\theta}(t_{n}|c_{n})}{\pi_{\mathrm{ref}}(t_{n}|c_{n})} A_{n}\right]$ 16: where $A_{n} = Q^{\pi}(c_{n},t_{n}) - V^{\pi}(c_{n})$ 17: end if  
18: end for  
19: return $\pi_{\theta}$

where $P _ { i } ^ { \mathrm { a c c e p t } } = \left( T _ { i } , { \mathcal { C } } _ { i } \right)$ and $P _ { i } ^ { \mathrm { r e j e c t } } = ( T _ { i } , \mathcal { C } _ { i } ^ { \mathrm { r e j e c t } } )$ This augmentation strategy ensures that:

$$
\exists C _ {i, j} ^ {\text { reject }} \in \mathcal {C} _ {i} ^ {\text { reject }}: \mathcal {L} (R _ {i}, C _ {i, j} ^ {\text { reject }}) = 0. \tag {4}
$$

# 3.3 Adversarial Learning Framework

We propose an adversarial learning framework comprising a generator $\pi _ { \boldsymbol { \theta } } ( t | \boldsymbol { c } )$ that produces the next token t given the current context c, a reward model $R ( c , t )$ evaluating the quality of generated tokens, and a value function $V ^ { \pi } ( c )$ estimating expected future rewards. The reward model is trained on the augmented dataset:

$$
\begin{array}{l} \mathcal {D} ^ {\prime} = \left\{\left(P _ {i} ^ {\mathrm{acc}}, R _ {i}, 1\right), \right. \\ (P _ {i} ^ {\text { rej }}, R _ {i}, 0), \tag {5} \\ \left. \left(P _ {i}, \hat {R} _ {i}, 0\right) \right\}. \\ \end{array}
$$

where $\hat { R } _ { i }$ represents the text response generated by $\pi _ { \theta }$ based on prompt $P _ { i }$ . The reward model optimizes the cross-entropy loss:

$$
\begin{array}{l} L _ {R} (\theta) = - \mathbb {E} _ {(c, t, y) \sim \mathcal {D} ^ {\prime}} \left[ y \log R (c, t) \right. \\ \left. + (1 - y) \log (1 - R (c, t)) \right]. \tag {6} \\ \end{array}
$$

<table><tr><td>Name</td><td>#Product</td><td>#PV-Pair</td><td>#Sample</td><td>#Token</td></tr><tr><td>PDD-Raw</td><td>201</td><td>93,616</td><td>-</td><td>-</td></tr><tr><td>PDD-Train</td><td>201</td><td>76,913</td><td>26,419</td><td>17,541,881</td></tr><tr><td>PDD-Rej-Train</td><td>201</td><td>66,838</td><td>26,419</td><td>14,983,806</td></tr><tr><td>PDD-Test</td><td>201</td><td>49,470</td><td>6,605</td><td>4,212,440</td></tr><tr><td>PDD-Rej-Test</td><td>201</td><td>31,280</td><td>6,605</td><td>3,629,544</td></tr></table>

Table 2: PDD-Raw contains only product information and available descriptions without prompt-response pairs, making it unsuitable for direct training. Rej represents mismatched prompt-response pairs. Train and Test denote the training and testing datasets, respectively. 

<table><tr><td>Name</td><td>#Type</td><td>#Sample</td><td>#Token</td></tr><tr><td>IFEval-Response</td><td>9</td><td>540</td><td>355,199</td></tr><tr><td>IFEval-Train</td><td>9</td><td>432</td><td>143,151</td></tr><tr><td>IFEval-Rej-Train</td><td>9</td><td>432</td><td>141,963</td></tr><tr><td>IFEval-Test</td><td>9</td><td>108</td><td>-</td></tr></table>

Table 3: IFEval-Response consists of GPT-4o responses provided by the IFEval benchmark in their official version. Train comprises the prompt-response pairs used for training, while Rej contains mismatched prompt-response pairs. As IFEval incorporates its own evaluation framework, the Test set does not include prompt-response pairs.

The generator’s objective function is formulated as:

$$
L _ {G} (\theta) = \mathbb {E} _ {n} \left[ \frac {\pi_ {\theta} (t _ {n} | c _ {n})}{\pi_ {\mathrm{ref}} (t _ {n} | c _ {n})} A _ {n} \right], \tag {7}
$$

where n indexes the token position, and the advantage function $A _ { n }$ is defined as:

$$
A _ {n} = Q ^ {\pi} (c _ {n}, t _ {n}) - V ^ {\pi} (c _ {n}). \tag {8}
$$

Moreover, the action-value function is:

$$
Q ^ {\pi} (c _ {n}, t _ {n}) = R (c _ {n}, t _ {n}) + \gamma \mathbb {E} _ {c _ {n + 1} \sim \pi_ {\theta}} [ V ^ {\pi} (c _ {n + 1}) ]. \tag {9}
$$

The value function is optimized by minimizing the mean squared error:

$$
L _ {V} (\theta) = \mathbb {E} _ {c} \left[ (V ^ {\pi} (c) - R (c, t)) ^ {2} \right]. \tag {10}
$$

# 4 Experiment Setup

# 4.1 Baselines

The experiments are grouped into two categories based on the role-playing methods used:

# 4.1.1 Prompt-Based Methods

(1) Direct Generation: The model generates content directly without role-playing instructions, evaluating its inherent capabilities and biases. (2) Chain-of-Thought (CoT): (Kojima et al., 2022) The model engages in reasoning before generating the output, improving coherence and transparency.

(3) Plan-and-Solve (Plan-N-Solve): (Wang et al., 2023) The model plans its response before generating content, leading to more organized solutions.

# 4.1.2 Training-Based Methods

(4) Supervised Fine-Tuning (SFT): Fine-tunes the model on a role-specific dataset to improve performance in role-playing scenarios. (5) DPO: (Rafailov et al., 2023) Directly optimizes for annotated responses, minimizing the likelihood of undesired outputs. (6) KTO: (Ethayarajh et al., 2024) Uses prospect theory to optimize model outputs, outperforming preference-based methods. (7) SimPO: (Meng et al., 2024) Aligns the reward function with model generation, simplifying optimization without reference models. (8) ORPO: (Hong et al., 2024) Optimize models with preferential response data but without reference model. (9) PPO: (Schulman et al., 2017) Optimizes the model using a pre-trained reward model that remains fixed throughout the training process (10) GAPO (Ours): Optimize models with reward criteria become progressively more demanding as training advances.

# 4.2 Training Dataset

Product Description Dataset (PDD) is a novel dataset designed for generating product descriptions in this paper. The dataset encompasses 201 product categories and contains 93,616 propertyvalue pairs. Models trained on this dataset are tasked with generating coherent product descriptions using only the provided property-value pairs, with two key constraints: they must (1) incorporate all given facts while (2) avoid the introduction of any additional information not present in the source data. For detailed information regarding the dataset construction methodology, please refer to Sec. A.2 in the Appendix, while comprehensive statistical analyses are presented in Tab. 2.

IFEval is a benchmark designed to evaluate Large Language Models’ instruction-following capabilities by enabling a standardized and automated assessment methodology (Zhou et al., 2023a). Building upon the existing dataset, we utilized GPT-4 (Achiam et al., 2023) to generate additional data samples that maintain similar constraint conditions while exhibiting low similarity to the original entries. Please refer to Sec. A.1 in the Appendix for a detailed description. The statistical breakdown of this expanded dataset is detailed in Tab. 3.

<table><tr><td>Model</td><td>Prompt</td><td>Punctuation</td><td>Format</td><td>Length</td><td>Content</td><td>Combination</td><td>ChangeCase</td><td>Startend</td><td>Keywords</td><td>Language</td><td>All</td></tr><tr><td>Qwen-2.5-7B</td><td>Naive Prompt</td><td>17.6</td><td>88.1</td><td>42.3</td><td>66.7</td><td>20.0</td><td>62.5</td><td>66.7</td><td>52.6</td><td>90.9</td><td>57.8</td></tr><tr><td>Qwen-2.5-7B</td><td>CoT</td><td>23.5</td><td>78.6</td><td>53.8</td><td>33.3</td><td>13.3</td><td>62.5</td><td>66.7</td><td>57.9</td><td>100.0</td><td>57.8</td></tr><tr><td>Qwen-2.5-7B</td><td>Plan-N-Solve</td><td>23.5</td><td>81.0</td><td>38.5</td><td>66.7</td><td>0.0</td><td>68.8</td><td>44.4</td><td>63.2</td><td>90.9</td><td>56.1</td></tr><tr><td>Qwen-2.5-7B + SFT</td><td>Naive Prompt</td><td>100.0</td><td>92.9</td><td>57.7</td><td>83.3</td><td>26.7</td><td>75.0</td><td>88.9</td><td>81.6</td><td>90.9</td><td>78.3</td></tr><tr><td>Qwen-2.5-7B + DPO</td><td>Naive Prompt</td><td>17.6</td><td>45.2</td><td>26.9</td><td>16.7</td><td>6.7</td><td>31.2</td><td>11.1</td><td>42.1</td><td>63.6</td><td>33.3</td></tr><tr><td>Qwen-2.5-7B + KTO</td><td>Naive Prompt</td><td>11.8</td><td>71.4</td><td>38.5</td><td>50.0</td><td>6.7</td><td>50.0</td><td>44.4</td><td>76.3</td><td>100.0</td><td>54.4</td></tr><tr><td>Qwen-2.5-7B + SimPO</td><td>Naive Prompt</td><td>11.8</td><td>45.2</td><td>23.1</td><td>16.7</td><td>0.0</td><td>31.2</td><td>0.0</td><td>39.5</td><td>63.6</td><td>30.6</td></tr><tr><td>Qwen-2.5-7B + ORPO</td><td>Naive Prompt</td><td>5.9</td><td>40.5</td><td>34.6</td><td>33.3</td><td>20.0</td><td>25.0</td><td>33.3</td><td>55.3</td><td>9.1</td><td>33.9</td></tr><tr><td>Qwen-2.5-7B + PPO</td><td>Naive Prompt</td><td>94.1</td><td>90.5</td><td>50.0</td><td>66.7</td><td>33.3</td><td>62.5</td><td>88.9</td><td>84.2</td><td>90.9</td><td>75.6</td></tr><tr><td>Qwen-2.5-7B + GAPO</td><td>Naive Prompt</td><td>100.0</td><td>95.2</td><td>57.7</td><td>83.3</td><td>46.7</td><td>75.0</td><td>100.0</td><td>92.1</td><td>100.0</td><td>83.9</td></tr></table>

Table 4: Performance comparison across different categories on IFEval Benchmark.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Prompt</td><td colspan="2">Reward Model</td><td colspan="2">LLM-as-a-Judge</td><td rowspan="2">Human</td></tr><tr><td>LongFormer-Base-40963k</td><td>LongFormer-Large-40963k</td><td>GPT-4o</td><td>GPT3.5-turbo</td></tr><tr><td>Qwen2.5-7B</td><td>Naive Prompt</td><td>61.4</td><td>52.3</td><td>75.4</td><td>73.7</td><td>45</td></tr><tr><td>Qwen2.5-7B</td><td>CoT</td><td>58.4</td><td>50.5</td><td>71.5</td><td>72.6</td><td>43</td></tr><tr><td>Qwen2.5-7B</td><td>Plan-N-Solve</td><td>62.8</td><td>53.7</td><td>72.5</td><td>78.1</td><td>51</td></tr><tr><td>Qwen2.5-7B + SFT</td><td>Naive Prompt</td><td>70.1</td><td>59.8</td><td>82.6</td><td>80.3</td><td>60</td></tr><tr><td>Qwen2.5-7B + DPO</td><td>Naive Prompt</td><td>12.5</td><td>11.3</td><td>5.4</td><td>9.6</td><td>0</td></tr><tr><td>Qwen2.5-7B + KTO</td><td>Naive Prompt</td><td>64.5</td><td>57.1</td><td>72.6</td><td>74.8</td><td>49</td></tr><tr><td>Qwen2.5-7B + SimPO</td><td>Naive Prompt</td><td>5.3</td><td>7.6</td><td>2.9</td><td>3.8</td><td>0</td></tr><tr><td>Qwen2.5-7B + ORPO</td><td>Naive Prompt</td><td>21.4</td><td>20.8</td><td>7.5</td><td>8.2</td><td>0</td></tr><tr><td>Qwen2.5-7B + PPO</td><td>Naive Prompt</td><td>89.4</td><td>88.5</td><td>89.7</td><td>86.4</td><td>81</td></tr><tr><td>Qwen2.5-7B + GAPO</td><td>Naive Prompt</td><td>95.4</td><td>94.3</td><td>90.2</td><td>90.0</td><td>89</td></tr></table>

Table 5: Comprehensive model performance comparison on PDD dataset. 3k represents the model is pre-tuned on 3,000 preferential data to give evaluation scores.

# 4.3 Evaluation Method

We utilize the IFEval dataset’s built-in evaluation methodology to maintain consistency with existing research in this domain.

For the PDD, we employ three evaluation methods: (1) The Reward Models act as automated evaluators during our adversarial training process. Specifically, we use Longformer models (Beltagy et al., 2020) with an input length capacity of 4096 tokens, which has been tuning on 3,000 preference data pairs to generate evaluation scores. (2) GPT-4o functions as an external evaluation model to provide independent assessment. (3) human evaluators assess the quality of generated descriptions based on predefined criteria.

# 5 Experiment

# 5.1 Overall Result

As shown in Tab. 4, while all preference optimization methods maintain basic functionality, their effectiveness varies significantly under different constraint types. This is evidenced by the stark performance gap: GAPO and PPO achieve strong overall performance (83.9% and 75.6% respectively), while methods like DPO, SimPO, and ORPO struggle considerably with scores of 33.3%, 30.6%, and 33.9% - particularly in handling complex constraints like combinations (6.7%, 0%, and 20.0% respectively) and length requirements (26.9%, 23.1%, and 34.6%).

As shown in Tab. 5, when facing more nuanced preferential prompts that require a fine-grained understanding of constraints, most traditional optimization methods experience catastrophic failure, while encoder-based approaches maintain robust performance. The collapse of conventional methods is dramatic: DPO, SimPO, and ORPO achieve near-zero performance on both automated metrics (5.4%, 2.9%, and 7.5% on GPT-4o) and human evaluation (all 0%). In contrast, encoder-based methods like GAPO and PPO demonstrate strong capability with GPT-4o scores of 90.2% and 89.7%, and human evaluation scores of 89% and 81% respectively.

# 5.2 Effectiveness of Preferential Prompt vs. Preferential Response

As shown in Tab. 6, training with Preferential Prompt consistently outperforms Preferential Response across all experimental configurations with both optimization methods. With 6,600 training samples, Preferential Prompt with GAPO achieves 95.4% PDD Performance, surpassing its Preferential Response counterpart by 12.5 percentage points and the supervised fine-tuning baseline by 34.0 percentage points. This performance advantage holds across different sample sizes, with Preferential Prompt showing improvements of 7.3 and 6.9 percentage points at 2,000 and 4,000 samples, respectively.

<table><tr><td>Model</td><td>Reward Model</td><td>#Training Samples</td><td>#Token</td><td>PDD Score</td><td> $\Delta_{No\ Train}$ </td><td> $\Delta_{PR\ vs.\ PP}$ </td></tr><tr><td colspan="7">No Training</td></tr><tr><td>Qwen-2.5-7B</td><td>-</td><td>-</td><td>-</td><td>61.4</td><td>-</td><td>-</td></tr><tr><td colspan="7">No Preferential Data</td></tr><tr><td>Qwen-2.5-7B + SFT</td><td>-</td><td>3,300</td><td>6,561,531</td><td>70.1</td><td>+ 8.3</td><td>-</td></tr><tr><td colspan="7">Training w/ Preferential Response (PR)</td></tr><tr><td>Qwen-2.5-7B + PPO</td><td>Qwen-2.5-7B</td><td>2,000</td><td>4,295,575</td><td>61.8</td><td>+ 0.4</td><td>- 6.7</td></tr><tr><td>Qwen-2.5-7B + PPO</td><td>Qwen-2.5-7B</td><td>4,000</td><td>8,660,218</td><td>72.4</td><td>+ 11.0</td><td>- 2.7</td></tr><tr><td>Qwen-2.5-7B + PPO</td><td>Qwen-2.5-7B</td><td>6,600</td><td>13,243,796</td><td>78.5</td><td>+ 17.1</td><td>- 10.9</td></tr><tr><td>Qwen-2.5-7B + GAPO</td><td>Longformer-0.4B</td><td>2,000</td><td>4,295,575</td><td>63.3</td><td>+ 1.9</td><td>- 7.3</td></tr><tr><td>Qwen-2.5-7B + GAPO</td><td>Longformer-0.4B</td><td>4,000</td><td>8,660,218</td><td>74.4</td><td>+ 13.0</td><td>- 6.9</td></tr><tr><td>Qwen-2.5-7B + GAPO</td><td>Longformer-0.4B</td><td>6,600</td><td>13,243,796</td><td>82.9</td><td>+ 21.5</td><td>- 12.5</td></tr><tr><td colspan="7">Training w/ Preferential Prompt (PP)</td></tr><tr><td>Qwen-2.5-7B + PPO</td><td>Qwen-2.5-7B</td><td>2,000</td><td>4,219,814</td><td>68.5</td><td>+ 7.1</td><td>+ 6.7</td></tr><tr><td>Qwen-2.5-7B + PPO</td><td>Qwen-2.5-7B</td><td>4,000</td><td>8,506,194</td><td>75.1</td><td>+ 13.7</td><td>+ 2.7</td></tr><tr><td>Qwen-2.5-7B + PPO</td><td>Qwen-2.5-7B</td><td>6,600</td><td>12,984,601</td><td>89.4</td><td>+ 28.0</td><td>+ 10.9</td></tr><tr><td>Qwen-2.5-7B + GAPO</td><td>Longformer-0.4B</td><td>2,000</td><td>4,219,814</td><td>70.6</td><td>+ 9.2</td><td>+ 7.3</td></tr><tr><td>Qwen-2.5-7B + GAPO</td><td>Longformer-0.4B</td><td>4,000</td><td>8,506,194</td><td>81.3</td><td>+ 19.9</td><td>+ 6.9</td></tr><tr><td>Qwen-2.5-7B + GAPO</td><td>Longformer-0.4B</td><td>6,600</td><td>12,984,601</td><td>95.4</td><td>+ 34.0</td><td>+ 12.5</td></tr></table>

Table 6: Comparative Analysis of using Preferential Response and Preferential Prompt. The PDD Performance metric represents the model’s generative output on the PDD dataset, as evaluated using a fine-tuned LongFormer-Large-4096 Reward model architecture. The IFEval Performance metric indicates the model’s comprehensive performance across the IFEval benchmark framework.   
![](images/d758ae0c3a83a0f4100f8bf81f8b4fe5ae6b719d402edf1c9ac13256ae291eed.jpg)

Figure 3: Analysis of Correlative Factors Influencing GAPO’s Performance on PDD and IFEval Benchmarks. The analysis utilizes 300 randomly sampled instances from the PDD test set and the complete IFEval test set with 108 samples for comprehensive evaluation.   
![](images/04e62ab95a6e10790b5813eb622053eeda8b830690ed3e454eff011e52db7884.jpg)

<details>
<summary>line</summary>

| Adversarial Training Stage | Reward Model Score |
| -------------------------- | ------------------ |
| W                          | 0.0                |
| A1                         | 0.2                |
| A2                         | 0.15               |
| A3                         | 0.25               |
| A4                         | 0.3                |
| A5                         | 0.4                |
| A6                         | 0.5                |
| A7                         | 0.6                |
| A8                         | 0.7                |
| A9                         | 0.8                |
| A10                        | 0.9                |
| A11                        | 0.95               |
| A12                        | 0.98               |
| A13                        | 0.99               |
| A14                        | 0.98               |
| A15                        | 0.97               |
</details>

Figure 4: Detailed Performance Analysis Across Sequential Adversarial Training Stages. W indicates the warmup phase, and A represents the adversarial phase with alternating training between Generator and Reward Model components.

# 5.3 Training Efficiency Analysis

Tab. 6 also demonstrates GAPO’s superior optimization capability and efficient utilization of training data. In Preferential Prompt training, GAPO demonstrates remarkable scaling efficiency, achieving a 24.8 percentage point improvement (70.6% to 95.4%) when increasing training tokens from 4.2M to 13.0M, while PPO shows a more modest improvement of 20.9 percentage points (68.5% to 89.4%). A similar pattern is observed in Preferential Response training, where GAPO achieves a 19.6 percentage point improvement compared to PPO’s 16.7 percentage points.

# 5.4 Detail Analysis on Model Performance

As shown in Fig. 3 Analysis across various dimensions of prompt complexity reveals several key findings. First, GAPO maintains consistent performance even as prompt length increases, showing only minimal degradation compared to baseline methods. Second, performance scales well with the

# Prompt:

Generate a description of a pet backpack, approximately 50 words. You need to use all the information from the "## Factual Information" section to generate the description, which should incorporate the emotions from "## Emotions to Add". Note that you cannot add additional factual information, please only add emotional content.

# ## Emotions to Add:

Pride: A feeling of satisfaction and respect for one's achievements or actions, typically associated with positive self-evaluation.

# ## Factual Information:

Material: Thickened PU leather

Suitable pet type: Small parrots

Weight: 750 grams

Color options: Sunlight yellow

Safety measures: High-strength nylon straps

Dimensions: 30 x 25 x 40 cm

Strap design: Anti-slip coating

# Qwen-2.5-7B

Cra ed with pride, this pet backpack is a testament to excellence in pet care. Made from thickened PU leather, it exudes durability and quality. Designed speci cally for small parrots, it ensures comfort and safety. The 750 grams weight is balanced, making it easy to carry. Available in sunny yellow, it upli s moods instantly. Its internal structure includes high-strength nylon straps, guaranteeing your pet's secure journey. With dimensions of 30 x 25 x 40 cm, it's spacious enough for comfort. The an -slip strap design adds an extra layer of security, giving you peace of mind. This pet backpack is not just a tool; it's a symbol of your love and commitment towards your feathered friend.

# Qwen-2.5-7B + SFT

Experience pride with this stylish, sunlight-yellow pet backpack for small parrots, cra ed from thickened PU leather, weighing just 750 grams. It features an -slip strap design and high-strength nylon safety measures, with spacious dimensions of 30 x 25 x 40 cm.

# Qwen-2.5-7B + PPO

Proudly present your vibrant companion with this stylish pet backpack, cra ed from high-quality thickened PU leather in a cheerful sunlight yellow. Weighing just 750 grams, it is though ully designed for small parrots and boasts dimensions of 30 x 25 x 40 cm. Featuring robust high-strength nylon straps for internal safety and an -slip coated straps, it ensures a secure and chic travel experience.

# Qwen-2.5-7B + GAPO

Proudly present your feathered friend in this sunlight yellow, thickened PU leather backpack, designed speci cally for small parrots. Weighing just 750 grams, it's a breeze to carry. At 30 x 25 x 40 cm, it's spacious yet compact. Your cherished companion will travel securely with high-strength nylon straps inside, while you con dently stride with an -slip coated shoulder straps.

Figure 5: Case study of model performance under different training baslines.

number of constraints, demonstrating robust handling of multiple simultaneous requirements. Third, the model shows strong capability in generating both short and long responses while maintaining constraint adherence.

# 5.5 Details in Adversarial Process

As shown in Figure 4, the evolution of Reward Models during adversarial training reveals distinct learning patterns and convergence behaviors. From the initial warmup phase (W), where all models assign near-zero scores to generated samples, we observe a clear stratification in learning trajectories across different Reward Models through stages A1-A15. The top-performing model demonstrates rapid improvement in the early stages (A1-A7), reaching a score of 0.6, followed by gradual convergence to 0.95 after A12. This stratification of final convergence scores (ranging from 0.2 to 0.95) and the stable plateaus after A12 indicates that GAPO successfully establishes a balanced adversarial training dynamic, where both the generator and Reward Models effectively learn the underlying constraints without falling into degenerate solutions (Lucic et al., 2018; Gulrajani et al., 2017; Creswell et al., 2018) often encountered in adversarial training scenarios.

# 5.6 Case Study

As illustrated in Fig. 5, training substantially augmented the model’s proficiency in following complex constraints while retaining linguistic authenticity, with GAPO attaining exemplary performance across all metrics. The base Qwen-2.5 model exhibited considerable divergence from the prescribed length and incorporated superfluous emotional elements. GAPO demonstrated remarkable superiority over alternative approaches, for it achieved meticulous control over word count and exemplified more sophisticated emotional articulation. Most significantly, GAPO maintained impeccable fidelity to the prescribed parameters by circumventing extraneous descriptive content and unsolicited emotional undertones.

# 6 Conclusion

This paper presents GAPO, a novel framework that effectively addresses constraint understanding in LLMs through the integration of GAN and PPO frameworks. Experimental results demonstrate GAPO’s superior performance compared to baseline methods (PPO, DPO, KTO, and ORPO) in both preferential prompt learning and general preferential response tasks, validating its effectiveness in enhancing constraint adherence while maintaining training stability. As LLMs continue to evolve and find applications across various domains requiring precise adherence to constraints, GAPO’s robust framework provides a promising direction for future developments in controlled text generation.

# Ethical Concern

This research contributes to constrained text generation through two key innovations: a Preferential Prompt data augmentation methodology and the GAPO training framework. Our approach significantly reduces dependency on preference data while maintaining generation quality, addressing a critical challenge in the field. The technical solutions focus solely on enhancing model capabilities under specific constraints, ensuring research reproducibility without introducing ethical concerns or societal risks. The implementation emphasizes technical optimization and maintains research neutrality throughout the development process.

Additionally, we introduce the PDD dataset, a comprehensive e-commerce corpus for product description generation. This dataset’s construction prioritized both data quality and ethical considerations. Through rigorous quality control measures, including thorough manual review processes, we ensured data diversity while addressing potential biases and sensitive issues. The dataset maintains strict compliance with ethical guidelines and privacy protection standards, safeguarding corporate and user interests. Our validation process confirms the dataset’s objectivity and reliability, establishing it as a valuable resource for future research endeavors.

# Limitation

GAPO’s primary strength lies in its ability to reduce the Reward Model’s training data requirements while improving Generator performance. However, this advantage comes with notable tradeoffs. The framework’s adversarial training process, involving simultaneous optimization of the Generator, Reward Model, and Critic Model, significantly increases computational demands compared to traditional preference optimization approaches. This intensive resource consumption represents a practical limitation for widespread adoption and implementation.

Furthermore, GAPO’s effectiveness is contingent upon the base model’s initial capabilities. Our research reveals that the framework performs optimally when applied to models that already possess fundamental generation competencies. This dependency arises because inadequate base model performance, particularly in generating semantically coherent responses, can compromise the Reward Model’s training quality during the adversarial process. This limitation suggests that GAPO is most suitable as an enhancement tool for established models rather than a solution for improving underperforming ones, highlighting the importance of careful model selection in its application.

# Acknowledge

This work was supported by the National Natural Science Foundation of China(62476145).

# References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.   
Alankrita Aggarwal, Mamta Mittal, and Gopi Battineni. 2021. Generative adversarial network: An overview of theory and applications. International Journal of Information Management Data Insights, 1(1):100004.   
Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. 2017. Guided open vocabulary image captioning with constrained beam search. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 936– 945, Copenhagen, Denmark. Association for Computational Linguistics.   
Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. 2024. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR.   
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.   
Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.   
Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712.   
Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.

Antonia Creswell, Tom White, Vincent Dumoulin, Kai Arulkumaran, Biswa Sengupta, and Anil A Bharath. 2018. Generative adversarial networks: An overview. IEEE signal processing magazine, 35(1):53–65.   
Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. 2023. MacLaSa: Multi-aspect controllable text generation via efficient sampling from compact latent space. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 4424–4436, Singapore. Association for Computational Linguistics.   
Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306.   
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2020. Generative adversarial networks. Communications of the ACM, 63(11):139–144.   
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. 2017. Improved training of wasserstein gans. Advances in neural information processing systems, 30.   
Qianyu He, Jie Zeng, Wenhao Huang, Lina Chen, Jin Xiao, Qianxi He, Xunzhe Zhou, Jiaqing Liang, and Yanghua Xiao. 2024. Can large language models understand real-world complex instructions? In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 18188–18196.   
Chris Hokamp and Qun Liu. 2017. Lexically constrained decoding for sequence generation using grid beam search. arXiv preprint arXiv:1704.07138.   
Jiwoo Hong, Noah Lee, and James Thorne. 2024. Reference-free monolithic preference optimization with odds ratio. arXiv e-prints, pages arXiv–2403.   
Jian Hu, Xibin Wu, Weixun Wang, Dehao Zhang, Yu Cao, et al. 2024. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143.   
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.   
Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and Richard Socher. 2019. CTRL: A conditional transformer language model for controllable generation. arXiv preprint arXiv:1909.05858.   
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582– 4597.   
Guangyi Liu, Zichao Yang, Tianhua Tao, Xiaodan Liang, Junwei Bao, Zhen Li, Xiaodong He, Shuguang Cui, and Zhiting Hu. 2022. Don’t take it literally: An edit-invariant sequence loss for text generation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2055–2078, Seattle, United States. Association for Computational Linguistics.   
Yi Liu, Xiangyu Liu, Xiangrong Zhu, and Wei Hu. 2024. Multi-aspect controllable text generation with disentangled counterfactual augmentation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9231–9253, Bangkok, Thailand. Association for Computational Linguistics.   
Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, and Olivier Bousquet. 2018. Are gans created equal? a large-scale study. Advances in neural information processing systems, 31.   
Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.   
Matt Post and David Vilar. 2018. Fast lexically constrained decoding with dynamic beam allocation for neural machine translation. arXiv preprint arXiv:1804.06609.   
Lianhui Qin, Sean Welleck, Daniel Khashabi, and Yejin Choi. 2022. COLD decoding: Energy-based constrained text generation with Langevin dynamics. In Advances in Neural Information Processing Systems.   
Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. ArXiv, abs/2305.18290.   
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.   
Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch,

Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615.   
Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, and Jianyuan Zhong. 2021. Attention is all you need in speech separation. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 21–25. IEEE.   
Qwen Team. 2024. Qwen2.5: A party of foundation models.   
Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023. Planand-solve prompting: Improving zero-shot chain-ofthought reasoning by large language models. arXiv preprint arXiv:2305.04091.   
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560.   
Ronald J Williams. 1992. Simple statistical gradientfollowing algorithms for connectionist reinforcement learning. Machine learning, 8:229–256.   
Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.   
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671.   
Hanqing Zhang, Haolin Song, Shaoyu Li, Ming Zhou, and Dawei Song. 2022. A survey of controllable text generation using transformer-based pre-trained language models. CoRR, abs/2201.05337.   
Yusen Zhang, Yang Liu, Ziyi Yang, Yuwei Fang, Yulong Chen, Dragomir Radev, Chenguang Zhu, Michael Zeng, and Rui Zhang. 2023. MACSum: Controllable summarization with mixed attributes. Transactions of the Association for Computational Linguistics, 11:787–803.

Chen Zheng, Ke Sun, Hang Wu, Chenguang Xi, and Xun Zhou. 2024. Balancing enhancement, harmlessness, and general capabilities: Enhancing conversational llms with direct rlhf. arXiv preprint arXiv:2403.02513.   
Xin Zheng, Hongyu Lin, Xianpei Han, and Le Sun. 2023. Toward unified controllable text generation via regular expression instruction. arXiv preprint arXiv:2309.10447.   
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023a. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.   
Wangchunshu Zhou, Yuchen Eleanor Jiang, Ethan Wilcox, Ryan Cotterell, and Mrinmaya Sachan. 2023b. Controlled text generation with natural language instructions. CoRR, abs/2304.14293.   
Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593.

# A Dataset Description

# A.1 IFEval (Instruction-Following Evaluation) Dataset

Dataset Construction Background and Purpose IFEval represents a benchmark dataset specifically designed to evaluate instruction-following capabilities of Large Language Models (LLMs). The research team systematically identified and defined 25 distinct types of verifiable instructions, based on which they constructed approximately 541 prompts. The distinguishing characteristic of these prompts lies in their verifiable nature, allowing for objective programmatic verification and thus eliminating potential subjective assessment biases.

Dataset Components The dataset encompasses multiple dimensions of instruction types. Regarding keyword requirements, it incorporates specific keyword usage directives, frequency requirements, and prohibited word constraints. Linguistic specifications include language-specific requirements. Additionally, the dataset implements textual constraints regarding length parameters, such as paragraph count, word count, and sentence quantity specifications. Furthermore, it encompasses requirements for specific content elements such as postscripts and placeholders, as well as format specifications including particular markup requirements, title formats, and JSON structure requirements. The dataset also incorporates specifications for text styling, including case usage requirements and punctuation conventions.

Evaluation Methodology and Metrics IFEval implements dual evaluation criteria: strict metrics and loose metrics. The strict evaluation methodology requires precise adherence to instructional requirements, while the loose evaluation methodology accommodates common variations while maintaining instructional integrity. The evaluation metrics specifically include:

• Prompt-level accuracy: Measuring the proportion of prompts where all instructions are correctly executed   
• Instruction-level accuracy: Quantifying the overall proportion of correctly executed instructions

# A.2 Product Description Dataset

Dataset Construction Process The Product Description Dataset (PDD) represents a specialized dataset focused on product description generation tasks, encompassing 1,000 product categories and 32,000 property-value pairs. The research team initially collected raw product information and descriptions, subsequently generating corresponding responses using GPT-4 based on carefully designed prompts, followed by human verification. Through modifications of constraint conditions in the original prompts, the team constructed a set of mismatched property-value pairs and descriptions (Rej dataset), which proves valuable for evaluating model robustness.

Dataset Structure and Composition The dataset comprises multiple subsets:

• PDD-Raw: Contains unprocessed original product information and descriptions   
• PDD-Train: High-quality training data generated by GPT-4 and validated through human verification   
• PDD-Test: Testing dataset serving dual purposes - evaluating generation model performance and validating scoring model efficacy   
• PDD-Rej-Train and PDD-Rej-Test: Mismatched datasets obtained through constraint condition modifications in original prompts

Evaluation Methodology The evaluation methodology for the PDD dataset incorporates multiple complementary approaches:

1. Model-based Evaluation: Utilizing advanced language models to assess constraint compliance   
2. Human Evaluation: Implementing human verification to assess content quality and accuracy   
3. Specialized Evaluation Models: Developing dedicated models to assess adherence to given constraints

The evaluation framework primarily focuses on two critical aspects:

• Verifying whether generated descriptions comprehensively incorporate all provided attribute information   
• Ensuring the absence of extraneous information not present in the source data

This comprehensive evaluation approach ensures robust assessment of model performance across multiple dimensions of content generation quality.

# B Manual Effort

This section presents our comprehensive manual verification process for both the PDD dataset and the model-generated outputs. Our verification framework encompasses two primary components: dataset quality assessment and model output evaluation.

# B.1 Dataset Quality Assessment

To ensure the reliability and ethical compliance of the PDD dataset, we conducted a thorough manual review process. A team of five domain experts independently examined 10% of the dataset entries (approximately 3,300 records), focusing on privacy protection and content fairness.

# B.1.1 Privacy Protection Verification

The privacy protection verification process systematically examines potential privacy concerns within the dataset. Table 7 outlines our evaluation criteria and standards.

<table><tr><td>Aspect</td><td>Verification Content</td><td>Acceptance Criteria</td></tr><tr><td>Personal Identity</td><td>Names, addresses, contact information</td><td>Strictly prohibited</td></tr><tr><td>Indirect Identifiers</td><td>Combinations of information that could lead to identification</td><td>Must not enable personal identification</td></tr><tr><td>Sensitive Data</td><td>Health conditions, financial details</td><td>Limited to general product-related information</td></tr></table>

Table 7: Privacy protection verification criteria for the PDD dataset

<table><tr><td>Category</td><td>Assessment Focus</td><td>Requirements</td></tr><tr><td>Gender</td><td>Gender-related stereotypes and biases</td><td>Neutral product descriptions without gender discrimination</td></tr><tr><td>Ethnicity</td><td>Racial or ethnic biases</td><td>No ethnicity-specific stereotypes or prejudices</td></tr><tr><td>Cultural Elements</td><td>Cultural sensitivity and representation</td><td>Objective and culturally neutral descriptions</td></tr></table>

Table 8: Fairness assessment criteria for dataset evaluation

# B.1.2 Fairness Assessment

Our fairness assessment framework examines potential biases and discriminatory content within the dataset. This evaluation ensures that product descriptions maintain objectivity and avoid perpetuating societal stereotypes. Table 8 presents our fairness evaluation framework.

# B.2 Model Output Evaluation

The evaluation of model-generated product descriptions focuses on two fundamental constraints: completeness and accuracy. We randomly selected 1,000 samples from the test set for this assessment, with three domain experts conducting independent evaluations.

# B.2.1 Evaluation Methodology

Our evaluation methodology employs a binary scoring system (0 or 1) based on strict compliance with both completeness and accuracy requirements. Table 9 details our scoring criteria.

<table><tr><td>Score</td><td>Requirements</td><td>Assessment Criteria</td></tr><tr><td>1</td><td>Complete satisfaction of all constraints</td><td>All property-value pairs included; No additional information introduced</td></tr><tr><td>0</td><td>Failure to meet any constraint</td><td>Missing any property-value pair OR Including extraneous information</td></tr></table>

Table 9: Model output evaluation criteria and scoring system

<table><tr><td>Component</td><td>Specification</td></tr><tr><td>CPU</td><td>Intel Xeon E5-2680 v4 @ 2.40GHz</td></tr><tr><td>RAM</td><td>128GB DDR4</td></tr><tr><td>GPU</td><td>NVIDIA A100 80GB</td></tr><tr><td>Operating System</td><td>Ubuntu 20.04 LTS</td></tr><tr><td>CUDA Version</td><td>12.1</td></tr><tr><td>Python Version</td><td>3.9.12</td></tr></table>

Table 10: Computing Infrastructure Specifications

# B.2.2 Evaluation Protocol

The evaluation protocol ensures consistency and reliability across assessments. Each evaluator independently examines the generated descriptions, comparing them against the input property-value pairs. For quality control, we conducted preliminary training sessions and established a standardized evaluation process. Disagreements among evaluators were resolved through detailed discussion and consensus building.

The final evaluation score for each generated description represents the average of scores from all evaluators. To ensure evaluation reliability, we calculated the inter-rater agreement using Cohen’s Kappa coefficient. For cases receiving a score of 0, evaluators documented specific violation types, enabling detailed analysis of model limitations and potential areas for improvement.

# C Training Expense

# C.1 Computing Infrastructure

All experiments in this study were conducted using the computing resources detailed in Table 10. To ensure reproducibility and consistent performance, we utilized the same hardware for all evaluations and training.

# C.2 Training Configuration

All hyperparameter settings are listed in Table 12. Given that IFEval contains only 430 training samples, we adopted smaller batch sizes and larger initial learning rates when training on the IFEval dataset.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Temperature</td><td>0.0</td></tr><tr><td>Top P</td><td>1.0</td></tr><tr><td>Frequency Penalty</td><td>0.0</td></tr><tr><td>Presence Penalty</td><td>0.0</td></tr><tr><td>Maximum Tokens</td><td>2048</td></tr><tr><td>Context Window</td><td>16385 tokens</td></tr></table>

Table 11: Language Model Parameters

# C.3 Generation Configuration

For our experiments, we employed carefully selected parameters to ensure consistent and reproducible results, as shown in Table 11. These parameters were chosen to minimize output variability while maintaining generation quality.

The temperature was set to 0.0 to maximize deterministic behavior, while maintaining a top P value of 1.0 to preserve the model’s ability to generate coherent responses. Both frequency and presence penalties were set to 0.0 to avoid artificial constraints on the model’s token selection process. These settings were kept constant across all experiments to ensure consistent generation behavior and reproducible results.

# D Prompt

We use a comprehensive prompt template as shown in Table 13. The template includes essential components such as product name, word count requirement, emotion specifications, and factual information. To explore the performance of different prompt engineers strategies, we further implement three distinct output formats (Table 14), namely Naive, Chain-of-Thought (CoT), and Plan-N-Solve approaches.

<table><tr><td>Parameter</td><td>PDD Dataset</td><td>IFEval Dataset</td></tr><tr><td>Training Samples</td><td>3300</td><td>430</td></tr><tr><td colspan="3">SFT</td></tr><tr><td>Learning Rate</td><td>5e-6</td><td>1e-4</td></tr><tr><td>Train Batch Size</td><td>256</td><td>32</td></tr><tr><td>Micro Train Batch Size</td><td>4</td><td>4</td></tr><tr><td>Max Sequence Length</td><td>4096</td><td>4096</td></tr><tr><td>Max Epochs</td><td>2</td><td>2</td></tr><tr><td colspan="3">DPO</td></tr><tr><td>Learning Rate</td><td>5e-7</td><td>1e-4</td></tr><tr><td>Train Batch Size</td><td>128</td><td>32</td></tr><tr><td>Micro Train Batch Size</td><td>4</td><td>4</td></tr><tr><td>Max Sequence Length</td><td>4096</td><td>4096</td></tr><tr><td>Max Epochs</td><td>2</td><td>2</td></tr><tr><td>Beta</td><td>0.1</td><td>0.1</td></tr><tr><td colspan="3">KTO</td></tr><tr><td>Learning Rate</td><td>5e-7</td><td>1e-4</td></tr><tr><td>Train Batch Size</td><td>128</td><td>32</td></tr><tr><td>Micro Train Batch Size</td><td>4</td><td>4</td></tr><tr><td>Max Sequence Length</td><td>4096</td><td>4096</td></tr><tr><td>Max Epochs</td><td>2</td><td>2</td></tr><tr><td>Beta</td><td>0.1</td><td>0.1</td></tr><tr><td colspan="3">SimPO</td></tr><tr><td>Learning Rate</td><td>5e-7</td><td>1e-4</td></tr><tr><td>Train Batch Size</td><td>128</td><td>32</td></tr><tr><td>Micro Train Batch Size</td><td>4</td><td>4</td></tr><tr><td>Max Sequence Length</td><td>4096</td><td>4096</td></tr><tr><td>Max Epochs</td><td>2</td><td>2</td></tr><tr><td>Beta</td><td>0.1</td><td>0.1</td></tr><tr><td colspan="3">ORPO</td></tr><tr><td>Learning Rate</td><td>5e-7</td><td>1e-4</td></tr><tr><td>Train Batch Size</td><td>128</td><td>32</td></tr><tr><td>Micro Train Batch Size</td><td>4</td><td>4</td></tr><tr><td>Max Sequence Length</td><td>4096</td><td>4096</td></tr><tr><td>Max Epochs</td><td>2</td><td>2</td></tr><tr><td>Beta</td><td>0.1</td><td>0.1</td></tr><tr><td colspan="3">PPO</td></tr><tr><td>Actor Learning Rate</td><td>5e-7</td><td>1e-4</td></tr><tr><td>Critic Learning Rate</td><td>9e-6</td><td>2e-4</td></tr><tr><td>Train Batch Size</td><td>128</td><td>32</td></tr><tr><td>Micro Train Batch Size</td><td>2</td><td>2</td></tr><tr><td>Rollout Batch Size</td><td>1024</td><td>1024</td></tr><tr><td>Micro Rollout Batch Size</td><td>4</td><td>4</td></tr><tr><td>Max Epochs</td><td>2</td><td>2</td></tr><tr><td>KL Coefficient</td><td>0.01</td><td>0.01</td></tr><tr><td>Max Prompt Length</td><td>1024</td><td>1024</td></tr><tr><td>Max Generate Length</td><td>3072</td><td>3072</td></tr><tr><td colspan="3">GAPO</td></tr><tr><td>Actor Learning Rate</td><td>5e-7</td><td>1e-4</td></tr><tr><td>Critic Learning Rate</td><td>9e-6</td><td>2e-4</td></tr><tr><td>Train Batch Size</td><td>128</td><td>16</td></tr><tr><td>Micro Train Batch Size</td><td>2</td><td>2</td></tr><tr><td>Rollout Batch Size</td><td>1024</td><td>1024</td></tr><tr><td>Micro Rollout Batch Size</td><td>4</td><td>4</td></tr><tr><td>Classifier Batch Size</td><td>8</td><td>4</td></tr><tr><td>Classifier Learning Rate</td><td>1e-5</td><td>1e-5</td></tr><tr><td>Max Prompt Length</td><td>1024</td><td>1024</td></tr><tr><td>Max Generate Length</td><td>3072</td><td>3072</td></tr><tr><td>KL Coefficient</td><td>0.01</td><td>0.01</td></tr><tr><td>Adversarial Training Epochs</td><td>2</td><td>2</td></tr><tr><td>Classifier Warmup Epochs</td><td>2</td><td>2</td></tr><tr><td>Classifier Training Epochs</td><td>2</td><td>2</td></tr><tr><td>Max Epochs</td><td>2</td><td>2</td></tr><tr><td>Classifier Generator Ratio</td><td>0.5</td><td>0.5</td></tr></table>

Table 12: Hyperparameter Settings for Different Training Methods

```markdown
# -*- coding: utf-8 -*-
Variables:
!<INPUT 0>! - Product Name
!<INPUT 1>! - Word Count Requirement
!<INPUT 2>! - Emotion Type and Description
!<INPUT 3>! - Factual Information
!<INPUT 4>! - Output Instruction
<commentblockmarker>###</commentblockmarker>
Please generate a product description about !<INPUT 0>! with approximately !<INPUT 1>! words. You need to use all the information provided in the Factual Information section to generate the description. The description should convey the emotion specified in the Emotion section.
Note that you cannot add additional factual information, and you must use all the given facts. Please only add non-factual, emotion-related content.

### Emotion:
!<INPUT 2>!

### Factual Information:
!<INPUT 3>!

### Your output should follow this format:
!<INPUT 4>! 
```  
Table 13: Base template for the experiment of product description generation in this paper.

<table><tr><td>Method</td><td>Prompt Template</td></tr><tr><td>Naive</td><td>!!=The description should be generated below the “### Generated Result:”</td></tr><tr><td>CoT</td><td>!!=Generate your thinking process step by step below the “### Thinking Process:”Then the description should be generated below the “### Generated Result:”</td></tr><tr><td>Plan-N-Solve</td><td>!!=Generate your planing step by step below the “### Planning:”Then the description should be generated below the “### Generated Result:”</td></tr></table>

Table 14: Detail prompt request in Tab. 13.