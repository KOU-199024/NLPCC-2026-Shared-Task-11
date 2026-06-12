# Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process

Ermo Hua1, Biqing Qi2,∗, Kaiyan Zhang1, Kai Tian1, Xingtai Lv1, Ning Ding1, Bowen Zhou1,2,\*

1Department of Electronic Engineering, Tsinghua University, Beijing, China

2Shanghai AI Laboratory, Shanghai, China

hem23@mails.tsinghua.edu.cn, zhoubowen@tsinghua.edu.cn

# Abstract

Supervised Fine-Tuning (SFT) and Preference Optimization (PO) are key processes for aligning Language Models (LMs) with human preferences post pre-training. While SFT excels in efficiency and PO in effectiveness, they are often combined sequentially without integrating their optimization objectives. This approach ignores the opportunities to bridge their paradigm gap and take the strengths from both. In this paper, we interpret SFT and PO with two subprocesses — Preference Estimation and Transition Optimization — defined at token level within the Markov Decision Process (MDP). This modeling shows that SFT is only a special case of PO with inferior estimation and optimization. PO estimates the model’s preference by its entire generation, while SFT only scores model’s subsequent predicted tokens based on prior tokens from ground truth answer. These priors deviates from model’s distribution, hindering the preference estimation and transition optimization. Building on this view, we introduce Intuitive Fine-Tuning (IFT) to integrate SFT and PO into a single process. Through a temporal residual connection, IFT brings better estimation and optimization by capturing LMs’ intuitive sense of its entire answers. But it solely relies on a single policy and the same volume of non-preference-labeled data as SFT. Our experiments show that IFT performs comparably or even superiorly to SFT and some typical PO methods across several tasks, particularly those requires generation, reasoning, and fact-following abilities. An explainable Frozen Lake game further validates the effectiveness of IFT for getting competitive policy. Code is available at https://github.com/ TsinghuaC3I/Intuitive-Fine-Tuning.

# 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable powerful potential across vari-

![](images/9de68c42e7109d0e2ff478766f09c0699678974233c087db4922e95c5df25b54.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Positive Samples"] --> B["SFT"]
    C["Negative Samples"] --> D["Reward Modeling"]
    D --> E["Policy"]
    D --> F["Reference"]
    E --> G["PPO"]
    F --> G
    H["Pre-Trained Model"] --> I["SFT"]
    I --> J["Policy"]
    I --> K["Reference"]
    J --> L["DPO"]
    K --> L
    L --> M["ORPO"]
    N["Pre-Trained Model"] --> O["IFT"]
    O --> P["Positive Samples"]
```
</details>

Figure 1: Comparison of Alignment Methods. IFT conducts alignment solely relying on positive samples and a single policy, starting from a pre-trained base model. IFT shows similar efficiency as SFT and effectiveness as PO methods.

ous downstream tasks after pre-training on largescale corpora (Brown et al., 2020; Achiam et al., 2023; Zhou and Ding, 2024). However, their instruction-following skills and trustworthiness still fall short of expectations (Bender et al., 2021; Bommasani et al., 2021; Li et al., 2022). Therefore, algorithms such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) (Ziegler et al., 2019; Ouyang et al., 2022; Lee et al., 2023) are used to further enhance LLMs’ abilities and align them better with human preferences.

Considering the limited effectiveness of SFT and the high cost of data construction and training computation for RLHF, these two methods are often combined to leverage their respective strengths. Unfortunately, they are typically implemented as a sequential recipe constrained by the paradigm gap between SFT and early RLHF methods, stemming from differences in loss functions, data formats, and the requirement for auxiliary models.

Recently, a method named Direct Preference Optimization (DPO) (Rafailov et al., 2024) was proposed to integrate Reward Modeling and Policy Optimization into one single procedure using a loss function derived from Proximal Policy Optimization (PPO) (Schulman et al., 2017). This approach demonstrates the potential to unify SFT and RLHF for the first time. Henceforth, many extended methods have been tried to realize this objective by bridging the gap between SFT and DPO. Some of them (Ethayarajh et al., 2024; Hong et al., 2024; Zhang et al., 2024) aim to transform the contrastive loss of DPO into a SFT-like cross-entropy loss, learning positive samples similar to SFT while unlearning negative samples resort to Unlikelihood Training (Welleck et al., 2019). Some others get rid of the preference-labeling process before training, switching to collect samples and labels/rewards in an online manner (Liu et al., 2023a; Yuan et al., 2024; Guo et al., 2024a; Calandriello et al., 2024; Tajwar et al., 2024), or just treating the SFT targets and online policy generations as positive and negative samples respectively (Xiong et al., 2023; Chen et al., 2024; Mitra et al., 2024; Liu et al., 2024). Nevertheless, preference-labeled pairwise data is still essential, and the need for reference model only becomes unnecessary in some cases. Thus the core differences between SFT and Preference Optimization (PO) are not eliminated thoroughly. To address this challenging issue, a deeper and more unified understanding of them are needed.

In this paper, we attempt to explain the similarities and differences between SFT and PO by defining Preference Estimation and Transition Optimization in terms of state-action pairs within the Markov Decision Process (MDP) framework. Through this modeling, we demonstrate that SFT is simply a specialized case of PO with inferior estimation and optimization than other methods. To estimate the policy preference, PO collects sentence-level negative samples from policy for each initial instruction. However, SFT only samples subsequent token for each intermediate state of ground truth answer, which leads to a biased estimation of policy preference and an inferior alignment performance.

Depending on this understanding, we introduce a unified alignment algorithm named Intuitive Fine-Tuning (IFT). Drawing inspiration from the human ability to grasp a intuitive sense of an answer after hearing a question, IFT employs a Temporary Residual Connection across tokens to approximate policy’s entire answer for each instruction. This approach helps IFT better estimate the policy’s preference than SFT, achieving alignment performance comparable or even superior to the sequential recipe of SFT and Preference Optimization. Additionally, IFT requires only a single policy model, and the same volume and format of data as SFT, enjoying both data and computation efficiency. These characteristics of IFT are advantageous in domains where preference data is unavailable or expensive to collect.

Our main contribution are three folds:

(1) Through defining Preference Estimation and Transition Optimization using the MDP, we demonstrate that SFT is only a special case of Preference Optimization. The similarities and differences of SFT, PPO and online/offline DPO are also compared within this framework;   
(2) We introduce Intuitive Fine-tuning (IFT), a deeply unified version of SFT and Preference Optimization. It utilizes temporary residual connections to extract the model’s generation preference given the initial instructions. IFT enjoys the similar efficiency as SFT on negative sampling, but can better estimate and optimize the policy preference.   
(3) Through experiments on several benchmarks, we validate that IFT performs comparably or superiorly to SFT and various Preference Optimization methods. An explainable toy-setting Frozen Lake further demonstrates the effectiveness of IFT.

# 2 Preliminaries

# 2.1 MDP in Language Models

The MDP applied to LMs can be formally described as a tuple $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , \mathcal { T } , r , \rho _ { 0 } )$ , where is the state space comprising ordered permutations of vocabularies, is the action space consisting of vocabularies defined by the tokenizer, $\tau$ is the transition matrix indicating token generation probabilities for given states, r represents rewards for state-action pairs, and $\rho _ { 0 }$ is the initial state typically based on given instructions. See more details in Appendix A.1.

The primary objective of Language Modeling is to train a policy $\pi _ { \theta }$ with $\mathcal { T } _ { \theta }$ to mimic a human policy $\pi ^ { * }$ with $\tau ^ { * }$ , aiming for the two transition matrices to become identical:

$$
\forall s \in \mathcal {S}, a \in \mathcal {A}: \mathcal {T} _ {\theta} (a | s) \rightarrow \mathcal {T} ^ {*} (a | s) \tag {1}
$$

This process can also be expressed using another state-state transition matrix $T )$ :

$$
\forall s, s ^ {\prime} \in S: T _ {\theta} (s ^ {\prime} | s) \rightarrow T ^ {*} (s ^ {\prime} | s) \tag {2}
$$

where $T$ is equivalence to $\tau$ , but instead, indicating the transition probability between states.

![](images/da884a8ebdde3d9b06993403fdea62416f2b4cd21b51834d5d61756b08749e8a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph_SFT["Label"]
        A0["Inst"] --> A1["a₀*"]
        A1 --> A2["a₁*"]
        A2 --> A3["a₂*"]
        A3 --> A4["a₃*"]
    end

    subgraph_PPO["Label"]
        B0["Inst"] --> B1["πθ(s₀)"]
        B1 --> B2["mask"]
        B2 --> B3["mask"]
        B3 --> B4["mask"]
        B4 --> B5["Inst"]
        B5 --> B6["πθ(s₀)"]
        B6 --> B7["mask"]
        B7 --> B8["mask"]
        B8 --> B9["Inst"]
        B9 --> B10["a₀^θ"]
        B10 --> B11["πθ(s₁^θ)"]
        B11 --> B12["mask"]
        B12 --> B13["mask"]
        B13 --> B14["Inst"]
        B14 --> B15["a₀^θ"]
        B15 --> B16["a₁^θ"]
        B16 --> B17["πθ(s₂^θ)"]
        B17 --> B18["mask"]
        B18 --> B19["Inst"]
        B19 --> B20["a₀^θ"]
        B20 --> B21["a₁^θ"]
        B21 --> B22["πθ(s₃^θ)"]
        B22 --> B23["Inst"]
        B23 --> B24["a₀^θ"]
        B24 --> B25["a₁^θ"]
        B25 --> B26["πθ(s₄^θ)"]
    end

    subgraph_DPO["Label"]
        C0["Inst"] --> C1["a₀^θ"]
        C1 --> C2["a₁^θ"]
        C2 --> C3["πθ(s₂^θ)"]
        C3 --> C4["mask"]
        C4 --> C5["Inst"]
        C5 --> C6["a₀^θ"]
        C6 --> C7["a₁^θ"]
        C7 --> C8["πθ(s₃^θ)"]
        C8 --> C9["Inst"]
        C9 --> C10["a₀^θ"]
        C10 --> C11["a₁^θ"]
        C11 --> C12["πθ(s₄^θ)"]
        C12 --> C13["Inst"]
        C13 --> C14["a₀^θ"]
        C14 --> C15["a₁^θ"]
        C15 --> C16["πθ(s₅^θ)"]
    end

    subgraph_IFT["Label"]
        D0["Inst"] --> D1["a₀^θ"]
        D1 --> D2["a₁^θ"]
        D2 --> D3["πθ(s₆^θ)"]
        D3 --> D4["Inst"]
        D4 --> D5["a₀^θ"]
        D5 --> D6["a₁^θ"]
        D6 --> D7["πθ(s₇^θ)"]
    end

    A0 -.->|+| C0
    A0 -.->|+| C1
    A0 -.->|+| C2
    A0 -.->|+| C3
    A0 -.->|+| C4
    A0 -.->|+| C5
    A0 -.->|+| C6
    A0 -.->|+| C7
    A0 -.->|+| C8
    A0 -.->|+| C9
    A0 -.->|+| C10
    A0 -.->|+| C11
    A0 -.->|+| C12
    A0 -.->|+| C13
    A0 -.->|+| C14
    A0 -.->|+| C15
    A0 -.->|+| C16
    A0 -.->|+| C17
    A0 -.->|+| C18
    A0 -.->|+| C19
    A0 -.->|+| C20
    A0 -.->|+| C21
    A0 -.->|+| C22
    A0 -.->|+| C23
    A0 -.->|+| C24
    A0 -.->|+| C25
    A0 -.->|+| C26
    A0 -.->|+| C27
    A0 -.->|+| C28
    A0 -.->|+| C29
    A0 -.->|+| C30
    A0 -.->|+| C31
    A0 -.->|+| C32
    A0 -.->|+| C33
    A0 -.->|+| C34
    A0 -.->|+| C35
    A0 -.->|+| C36
    A0 -.->|+| C37
    A0 -.->|+| C38
    A0 -.->|+| C39
    A0 -.->|+| C40
    A0 -.->|+| C41
    A0 -.->|+| C42
    A0 -.->|+| C43
    A0 -.->|+| C44
    A0 -.->|+| C45
    A0 -.->|+| C46
    A0 -.->|+| C47
    A0 -.->|+| C48
    A0 -.->|+| C49
    A0 -.->|+| C50
    A0 -.->|+| C51
    A0 -.->|+| C52
    A0 -.->|+| C53
    A0 -.->|+| C54
    A0 -.->|+| C55
    A0 -.->|+| C56
    A0 -.->|+| C57
    A0 -.->|+| C58
    A0 -.->|+| C59
    A0 -.->|+| C60
    A0 -.->|+| C61
    A0 -.->|+| C62
    A0 -.->|+| C63
    A0 -.->|+| C64
    A0 -.->|+| C65
    A0 -.->|+| C66
    A0 -.->|+| C67
    A0 -.->|+| C68
    A0 -.->|+| C69
    A0 -.->|+| C70
    A0 -.->|+| C71
    A0 -.->|+| C72
    A0 -.->|+| C73
    A0 -.->|+| C74
    A0 -.->|+| C75
    A0 -.->|+| C76
    A0 -.->|+| C77
    A0 -.->|+| C78
    A0 -.->|+| C79
    A0 -.->|+| C80
    A0 -.->|+| C81
    A0 -.->|+| C82
    A0 -.->|+| C83
    A0 -.->|+| C84
    A0 -.->|+| C85
    A0 -.->|+| C86
    A0 -.->|+| C87
    A0 -.->|+| C88
    A0 -.->|+| C89
    A0 -.->|+| C90
    A0 -.->|+| C91
    A0 -.->|+| C92
    A0 -.->|+| C93
    A0 -.->|+| C94
    A0 -.->|+| C95
    A0 -.->|+| C96
    A0 -.->|+| C97
    A0 -.->|+| C98
    A0 -.->|+| C99
    A0 -.->|- | SFT["Label SFT"]
    
    subgraph_SFT["Label SFT"]
        SFT_["(a) SFT"]
        SFT_["(b) PPO"]
        SFT_["(c) DPO"]
        SFT_["(d) IFT"]
    
    style SFT fill:#f9f,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
```
</details>

Figure 2: The Training Paradigm of Different Methods. Symbol ∗ and θ denote human and model respectively, with $a _ { i } ^ { * } = \pi ^ { * } ( s _ { i } ^ { * } )$ and $s _ { i + 1 } ^ { * } = [ s _ { i } ^ { * } , a _ { i } ^ { * } ]$ , similarly for θ. SFT uses priors deviating from model distribution, resulting in a more biased estimation of model preferences compared to PPO and DPO. IFT achieves a better estimation than SFT by Temporary Residual Connections across tokens. This approach passes the residual embedding from one token to the next, creating a more accurate prior while maintaining the data and computational efficiency of SFT.

# 2.2 Preference Estimation

We define the preference P of policy π given an initial instruction $\rho _ { 0 }$ as a mapping:

$$
\mathcal {P} (\rho_ {0}): \rho_ {0} \rightarrow [ \pi (\rho_ {0}), \pi (s _ {1}), \pi (s _ {2}), \dots ] \tag {3}
$$

where $s _ { i + 1 } = [ s _ { i } , a _ { i } ] , a _ { i } = \pi ( s _ { i } )$ and $s _ { 0 } = \rho _ { 0 }$ .

During alignment, the model preference gradually approaches the human preference:

$$
\mathcal {P} _ {\theta} (\rho_ {0}) \rightarrow \mathcal {P} ^ {*} (\rho_ {0}) \tag {4}
$$

$$
\begin{array}{l}\mathcal {P} _ {\theta} (\rho_ {0}): \rho_ {0} \rightarrow [ \pi_ {\theta} (\rho_ {0}), \pi_ {\theta} (s _ {1} ^ {\theta}), \pi_ {\theta} (s _ {2} ^ {\theta}), \dots ]\\\mathcal {P} ^ {*} (\rho_ {0}): \rho_ {0} \rightarrow [ * (\rho_ {0}), * (\rho_ {0}), * (\rho_ {0}) ]\end{array}\tag {5}
$$

$$
\mathcal {P} ^ {*} (\rho_ {0}): \rho_ {0} \to [ \pi^ {*} (\rho_ {0}), \pi^ {*} (s _ {1} ^ {*}), \pi^ {*} (s _ {2} ^ {*}), \ldots ]
$$

As the truly preferences are difficult to obtain, alignment is usually conducted based on the Preference Estimation of model and human, denoted as $\hat { \mathcal { P } } _ { \theta }$ and $\hat { \mathcal { P } } ^ { * }$ respectively. The estimations from some typical methods are listed in Table 1.

To make preference optimizable, the policy’s preference can also be expressed as follows:

$$
\mathcal {P} (\rho_ {0}) = \{\mathcal {T} (a | s) | \forall a \in \mathcal {A}, s \in \mathcal {S} _ {\rho_ {0}} \} \tag {6}
$$

Here, $ { \boldsymbol } { S } _ { \rho _ { 0 } }$ denotes a conditional state space that constrained by the initial state $\rho _ { 0 }$ , within which each state can only be initially derived from $\rho _ { 0 }$ Consequently, the model preference can be optimized through transition matrix, named Transition Optimization.

# 2.3 Transition Optimization

Ideally, we want to align the state-action transition matrix between model and human in a $\rho _ { 0 } \cdot$ - constrained state space:

$$
\forall a \in \mathcal {A}, s \in \mathcal {S} _ {\rho_ {0}}: \mathcal {T} _ {\theta} (a, s) \rightarrow \mathcal {T} ^ {*} (a, s) \tag {7}
$$

which is equivalent to the following format expressed by state-state transition matrix:

$$
\forall s \in \mathcal {S} _ {\rho_ {0}}: T _ {\theta} (s, \rho_ {0}) \rightarrow T ^ {*} (s, \rho_ {0}) \tag {8}
$$

However, considering the limited data, only matrix elements representing state-action/state-state pairs contained in the dataset  would be aligned. Given a data sample with instruction $\rho _ { 0 }$ and target answer with length-N, the objective would be $\forall a \in A , n \in [ 0 , N ] , \rho _ { 0 } \in \mathcal { D } , s _ { n } ^ { * } \in S _ { \rho _ { 0 } } ^ { * }$ :

$$
\mathcal {T} _ {\theta} (a, s _ {n} ^ {*}) \rightarrow \mathcal {T} ^ {*} (a, s _ {n} ^ {*}) \tag {9}
$$

Or equivalent to $\forall n \in [ 0 , N ] , \rho _ { 0 } \in \mathcal { D } , s _ { n } ^ { * } \in \mathcal { S } _ { \rho _ { 0 } } ^ { * }$

$$
T _ {\theta} (s _ {n} ^ {*}, \rho_ {0}) \to T ^ {*} (s _ {n} ^ {*}, \rho_ {0}) \tag {10}
$$

where $s _ { 0 } ^ { * } = \rho _ { 0 } , T ^ { * } ( \rho _ { 0 } | \rho _ { 0 } ) = T _ { \theta } ( \rho _ { 0 } | \rho _ { 0 } ) = 1$ , and $s _ { i } ^ { * }$ \* denotes the intermediate state of target answer.

Consequently, the loss function can be derived from the disparities of the transition matrices between model and human. Some typical loss function are listed in Appendix A.4.

# 3 From SFT to Preference Optimization

We reformulate SFT, PPO and DPO using the aforementioned framework, detailed in Table 1 and Appendix A.4. A more comprehensible version is presented in Figure 2. To compare the differences between them, we begin by introducing a fundamental theorem and corollary:

Theorem Given a set of events ${ \mathcal { Z } } ,$ , the probability of any event $z \in { \mathcal { Z } }$ is between 0 and 1, i.e., $\forall z \in$ $\mathcal { Z } : 0 \le P ( z ) \le 1$ . If all events are mutually independent, the sum of their probabilities equals $\begin{array} { r } { I , \ i . e . , \ 1 = \ \sum _ { z \in \mathcal { Z } } P ( z ) } \end{array}$ . The event $z ^ { * }$ with the highest probability has a probability greater than or equal to any other event, i.e., $\forall z \in \mathcal { Z } : 0 \leq$ $P ( z ) \leq P ( z ^ { * } ) \leq 1$ .

Corollary LMs consistently assign higher probabilities to their own greedy predictions than to human preference:

$$
\forall s \in S: \mathcal {T} _ {\theta} (\pi^ {*} (s), s) \leq \mathcal {T} _ {\theta} (\pi_ {\theta} (s), s) \leq 1 \tag {11}
$$

thus LMs tend to assign higher probabilities to its own generation than to target answer given the same initial instruction $\forall n \ \in \ [ 0 , N ] , s _ { n } ^ { * } \ \in$ $S _ { \rho _ { 0 } } ^ { * } , s _ { n } ^ { \theta } \in S _ { \rho _ { 0 } } ^ { \theta }$ :

$$
T _ {\theta} (s _ {n} ^ {*}, \rho_ {0}) \leq T _ {\theta} (s _ {n} ^ {\theta}, \rho_ {0}) \leq 1 \tag {12}
$$

where N represents the length when the generation reaches the EOS token or the truncation length.

SFT provides an unbiased estimation of human preference, but a biased estimation for model:

$$
\hat {\mathcal {P}} _ {\theta} (\rho_ {0}): \rho_ {0} \rightarrow [ \pi_ {\theta} (\rho_ {0}), \pi_ {\theta} (s _ {1} ^ {*}), \pi_ {\theta} (s _ {2} ^ {*}), \dots ] \tag {13}
$$

which is caused by wrong prior state when predicting each subsequent token. Consequently, the Transition Optimization objective of SFT:

$$
T _ {\theta} (s _ {n} ^ {*}, s _ {n - 1} ^ {*}) \to T ^ {*} (s _ {n} ^ {*}, s _ {n - 1} ^ {*}) \tag {14}
$$

secretly sets $T _ { \theta } ( s _ { n - 1 } ^ { * } , \rho _ { 0 } ) ~ = ~ 1$ during aligning $T _ { \theta } { \left( s _ { n } ^ { * } , \rho _ { 0 } \right) }$ with $T ^ { * } ( s _ { n } ^ { * } , \rho _ { 0 } )$ . This makes an overestimation of the transition probabilities and preference of model, leading to an inferior optimization progress in SFT. Thus Preference Optimization is needed for further preference alignment.

PPO shows an unbiased estimation of model preference, while employing a progressively unbiased estimation of human preference:

$$
\hat {\mathcal {P}} ^ {*} (\rho_ {0}): \rho_ {0} \rightarrow [ \pi^ {*} (\rho_ {0}), \pi^ {*} (s _ {1} ^ {\theta}), \pi^ {*} (s _ {2} ^ {\theta}), \dots ] \tag {15}
$$

Initially biased, this estimation gradually becomes unbiased as the model aligns with human preference over time. As $T _ { \theta } ( s _ { n } ^ { \theta } , \rho _ { 0 } )$ is consistently closer to 1 than $T _ { \theta } ( s _ { n - 1 } ^ { * } , \rho _ { 0 } )$ , PPO provides an closer approximation than SFT to the actual circumstances of model in Transition Optimization:

$$
T _ {\theta} (\hat {s _ {n} ^ {*}}, s _ {n - 1}) \to T ^ {*} (\hat {s _ {n} ^ {*}}, s _ {n - 1}) \tag {16}
$$

which sets $T _ { \theta } ( s _ { n } ^ { \theta } , \rho _ { 0 } ) = 1$ and $\hat { s _ { n } ^ { * } } = \pi ^ { * } ( s _ { n - 1 } ^ { \theta } )$ . However, estimating $\pi ^ { * } ( s _ { n - 1 } ^ { \theta } )$ is at the expense of preference-labeling, reward modeling and online sampling.

DPO theoretically achieves the best estimation across all scenarios, even without reward modeling. However, obtaining pairwise preference data online is costly, as it requires real-time negative sampling from model and preference labeling by human. Thus, mainstream implementations often rely on off-policy negative samples out-of-distribution from the optimized model, which may yield unstable and sub-optimal results due to biased preference estimation and inferior transition optimization.

<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="2">Preference Estimation</td><td rowspan="2">Transition Optimization</td></tr><tr><td> $s_n^*$  in  $\hat{\mathcal{P}}^*$ </td><td> $s_n^\theta$  in  $\hat{\mathcal{P}}_\theta$ </td></tr><tr><td colspan="2">Truly</td><td> $s_n^*$ </td><td> $s_n^\theta$ </td><td> $T_\theta(s_n^*,\rho_0)\to T^*(s_n^*,\rho_0)$ </td></tr><tr><td colspan="2">SFT</td><td> $s_n^*$ </td><td> $s_n^*$ </td><td> $T_\theta(s_n^*,s_{n-1}^*)\to T^*(s_n^*,s_{n-1}^*)$ </td></tr><tr><td colspan="2">PPO</td><td> $s_n^\theta$ </td><td> $s_n^\theta$ </td><td> $T_\theta(s_n^*,s_{n-1}^\theta)\to T^*(s_n^*,s_{n-1}^\theta)$ </td></tr><tr><td rowspan="2">DPO</td><td>online</td><td> $s_n^*$ </td><td> $s_n^\theta$ </td><td> $T_\theta(s_n^*,\rho_0)\to T^*(s_n^*,\rho_0)$ </td></tr><tr><td>offline</td><td> $s_n^*$ </td><td> $s_n^{\theta^-}$ </td><td> $\hat{T}_\theta(s_n^*,\rho_0)\to T^*(s_n^*,\rho_0)$ </td></tr></table>

Table 1: Reformulation of SFT, PPO and DPO

# 4 Method

While SFT is data and computation-efficient, it has an inferior approximation for both Preference Estimation and Transition Optimization. On the other side, Preference Optimization (represented by PPO and DPO) enjoys better approximation at the expense of preference data construction. We hope to make good use of their strength, using solely target data as SFT but having a similar approximation as Preference Optimization. See pseudo code in Appendix B.3.

# 4.1 Intuitive Preference Estimation

A key distinction between SFT and Preference Optimization is whether the full distribution of model’s preference for each initial instruction is sampled. Preference Optimization samples the policy’s entire answer to estimate its preference, ensuring each generation relies on the prior adheres to the model’s distribution. But SFT only samples subsequent tokens the intermediate state of the target answer, the used prior may be far away from the model preference, leading to inferior preference estimation for model.

To obtain a prior state estimation $\hat { s _ { i } ^ { \theta } }$ closer to model distribution, we introduce a model-based distribution disturbance function $\delta _ { \theta }$ for the biased prior state:

$$
\hat {s _ {i} ^ {\theta}} = \delta_ {\theta} (s _ {i} ^ {*}) = (1 - \lambda) s _ {i} ^ {*} + \lambda \pi_ {\theta} (s _ {i - 1} ^ {*}) \tag {17}
$$

which can also be interpreted as a temporal residual connection that passes the residual embedding from one token to the next. Through this approach, model can predict not only the next token from intermediate state of target answer, but also develop an intuitive sense to the entire answer generation solely based on the initial instruction, deriving more unbiased prior and accurate Preference Estimation for model:

$$
\hat {\mathcal {P}} _ {\theta} (\rho_ {0}) = [ (1 - \lambda) \mathcal {P} _ {\theta} ^ {s f t} + \lambda \mathcal {P} _ {\theta} ^ {t r u l y} ] (\rho_ {0}) \tag {18}
$$

With improved Preference Estimation, we achieve a Transition Optimization process closer to the original objective ∀n $\in [ 0 , N ] , \rho _ { 0 } \in \mathcal { D } , s _ { n } ^ { * } \in \mathcal { S } _ { \rho _ { 0 } } ^ { * }$ :

$$
\hat {T} _ {\theta} (s _ {n} ^ {*}, \rho_ {0}) \to T ^ {*} (s _ {n} ^ {*}, \rho_ {0}) \tag {19}
$$

where $s _ { 0 } ^ { * } = \rho _ { 0 }$ and $\hat { T _ { \theta } } ( s _ { n } ^ { * } , \rho _ { 0 } ) = \prod _ { i = 0 } ^ { n - 1 } T _ { \theta } ( s _ { i + 1 } ^ { * } , \hat { s _ { i } ^ { \theta } } )$ .

This objective can be optimized by the following loss function, which quantifies the disparities of transition between model and human:

$$
\mathcal {L} (\mathcal {T} _ {\theta}, \delta_ {\theta}) = \mathbb {E} \left[ - \sum_ {n = 0} ^ {N} \log \mathcal {T} _ {\theta} (a _ {i} ^ {*}, \delta_ {\theta} (s _ {i} ^ {*})) \right] \tag {20}
$$

where $a _ { i } ^ { * } = \pi ^ { * } ( \delta ^ { * } ( s _ { i } ^ { * } ) ) = \pi ^ { * } ( s _ { i } ^ { * } )$ . See Appendix A.2 for complete derivation.

# 4.2 Dynamic Relation Propagation

The Intuitive Preference Estimation implicitly performs Dynamic Relation Propagation, during which the generation of future tokens will be influenced by the prediction accuracy of current token.

However, limited by the parallel computing mode, the gradient map could only be built on the same time-step. Thus, the current generated tokens is unable to obtain gradient feedback from the future generated tokens. Therefore, we reformulate the loss function by a differentiable cumulativesummation to get around this limitation:

$$
\mathcal {L} _ {\mathrm{IFT}} = \mathbb {E} \left[ - \sum_ {n = 0} ^ {N} \sum_ {i = n} ^ {N} \log \mathcal {T} _ {\theta} (a _ {i} ^ {*}, \delta_ {\theta} (s _ {i} ^ {*})) \right] \tag {21}
$$

This reformulation implicitly satisfies the Bellman Equation for each state, which guarantees the optimization enjoys both of the effectiveness as RLHF and efficiency as SFT:

$$
V _ {\theta} (\hat {s _ {n} ^ {\theta}}) = \exp \left(- \mathcal {L} \big (\hat {T _ {\theta}} (s _ {n} ^ {*}, \rho_ {0}) \big)\right) \tag {22}
$$

The derivation is in Appendix A.3. Additionally, a decay factor can be incorporated to ensure effectiveness in long trajectories, as in the typical Bellman Equation.

# Algorithm 1 The pseudo-code of IFT

1: Input:   
Initial instruction $\rho _ { 0 } .$ , Ground truth $s ^ { * }$ with N tokens: $s ^ { * } [ 1 ] , \ldots , s ^ { * } [ N ]$

2: Step 1: Inference One Step Ahead

3: for t in $[ 1 , N ]$ do

4: Predict the probability distribution of the t-th token: $P _ { t } ^ { \prime } = \pi _ { \theta } ( s ^ { * } [ 0 : t - 1 ] )$

5: Sample tokens: $s ^ { \theta } [ t ] =$ arg max $P _ { t } ^ { \prime }$

6: end for

7: Step 2: Intuitive Preference Estimation

8: Encode $s ^ { * }$ and $s ^ { \theta }$ using Embedding Layer E

9: Compute the fused embedding:

$$
e = (1 - \lambda) E (s ^ {*}) + \lambda E (s ^ {\theta})
$$

10: for t in $[ 1 , N ]$ do

11: Predict the probability distribution of the t-th token: $P _ { t } ^ { \prime \prime } = ( \pi _ { \theta } / E ) ( e [ 0 : t - 1 ] )$

12: Compute token loss: $\mathcal { L } _ { t } = \log ( P _ { t } ^ { \prime \prime } , s ^ { * } [ t ] )$

13: end for

14: Step 3: Dynamic Relation Propagation

15: for t in $[ 1 , N ]$ do

16: Compute the cumsum weight similar to Bellman Equation: $w _ { t } = \sum _ { i = t } ^ { N } \alpha ^ { N - t } \mathcal { L } _ { i }$

17: end for

18: Output: Final loss $\mathcal { L } _ { \mathrm { I F T } } = w \cdot \mathcal { L }$

# 5 Experiments

We conduct experiments mainly on NLP setting. Considering the absence of an optimal policy of human language generation, we also utilize the Frozen Lake environment for further validation.

# 5.1 Settings for NLP

Datasets. Our main experiments use UltraChat-200k (Ding et al., 2023) as the single-target dataset and UltraFeedback-60k (Cui et al., 2023) as the pairwise preference dataset. We also include a variant of UltraFeedback-60k introduced by Meng et al. (2024), which is sampled from Gemma2 and LLaMA3 and labeled with preferences using ArmoRM.

Models. Our main experiments are conducted on Mistral-7B-v0.1 (Jiang et al., 2023) and Mistral-7Bsft-beta (Tunstall et al., 2023), with the latter one fine-tuned from the former using UltraChat-200k.

<table><tr><td>Method</td><td>ARC</td><td>ARC-Gen</td><td>MMLU</td><td>TruthfulQA</td><td>WinoGrande</td><td>GSM8K</td><td>Avg.</td></tr><tr><td>Mistral-7B</td><td>53.07</td><td>73.04</td><td>59.14</td><td>45.29</td><td>77.58</td><td>38.89</td><td>54.79</td></tr><tr><td colspan="8">fine-tuning with UltraFeedback-60k</td></tr><tr><td>+ SFT</td><td>56.49</td><td>74.00</td><td>60.44</td><td>55.57</td><td>77.90</td><td>42.84</td><td>58.65</td></tr><tr><td>+ DPO</td><td>61.86</td><td>73.54</td><td>61.02</td><td>47.98</td><td>76.64</td><td>43.89</td><td>58.28</td></tr><tr><td>+ TDPO</td><td>56.06</td><td>73.72</td><td>60.23</td><td>43.94</td><td>77.03</td><td>41.70</td><td>55.79</td></tr><tr><td>+ ORPO</td><td>56.66</td><td>73.98</td><td>60.57</td><td>51.77</td><td>77.19</td><td>42.30</td><td>57.70</td></tr><tr><td>+ SimPO</td><td>59.90</td><td>73.55</td><td>52.61</td><td>47.25</td><td>78.30</td><td>37.53</td><td>55.15</td></tr><tr><td>+ IFT</td><td>56.74</td><td>74.15</td><td>60.49</td><td>57.65</td><td>78.45</td><td>44.73</td><td>59.61</td></tr><tr><td>Mistral-ORPO- $\alpha$ </td><td>57.25</td><td>73.72</td><td>58.74</td><td>60.59</td><td>73.72</td><td>46.78</td><td>59.41</td></tr><tr><td colspan="8">fine-tuning with Ultrachat-200k + UltraFeedback-60k sequentially</td></tr><tr><td>+ SFT</td><td>57.68</td><td>72.87</td><td>58.25</td><td>45.78</td><td>77.19</td><td>40.94</td><td>55.97</td></tr><tr><td>+ SFT + SFT</td><td>58.10</td><td>72.61</td><td>58.40</td><td>48.59</td><td>76.80</td><td>43.06</td><td>56.99</td></tr><tr><td>+ SFT + DPO</td><td>63.91</td><td>73.98</td><td>59.75</td><td>46.39</td><td>76.06</td><td>41.47</td><td>57.52</td></tr><tr><td>+ SFT + TDPO</td><td>59.13</td><td>73.72</td><td>58.92</td><td>46.63</td><td>76.32</td><td>44.58</td><td>57.12</td></tr><tr><td>+ SFT + ORPO</td><td>58.45</td><td>73.21</td><td>58.80</td><td>50.31</td><td>76.45</td><td>42.76</td><td>57.35</td></tr><tr><td>+ SFT + SimPO</td><td>60.83</td><td>73.63</td><td>59.01</td><td>49.45</td><td>76.95</td><td>38.44</td><td>56.94</td></tr><tr><td>+ SFT + IFT</td><td>58.36</td><td>73.38</td><td>58.45</td><td>52.39</td><td>78.06</td><td>43.82</td><td>58.22</td></tr><tr><td>Zephyr-7B- $\beta$ </td><td>67.41</td><td>72.61</td><td>58.74</td><td>53.37</td><td>74.11</td><td>33.89</td><td>57.50</td></tr></table>

Table 2: Evaluation on Open-LLM Leaderboard with chat template. When fine-tuning with the same recipe, IFT achieves the highest average score across all methods. Directly conducting alignment using IFT showcases the best performance in all recipes with the least data and computation.

We also consider models with different architectures and parameter scales, including Gemma-2B (Team et al., 2024) and LLaMA3-8B (Grattafiori et al., 2024).

Scenarios. We consider two different training scenarios, one using Preference Optimization exclusively, and the other employing sequential recipe of SFT and Preference Optimization. In the first scenario, alignment is conducted directly from base model Mistral-7B-v0.1 using UltraFeedback. In order to ensure balanced data volume between different method, we randomly sample 60k data from UltraChat as supplementary for SFT and IFT, for only the target data are utilized in these two methods. The second scenario is commonly seen, where SFT and Preference Optimization is employed sequentially. For this scenario, we use Mistral-7B-sft-beta as start-point, which has been fine-tuned with UltraChat using SFT. Then we finetune it further with UltraFeedback using Preference Optimization.

Baselines. SFT and DPO (Rafailov et al., 2024) are our main baselines, and we exclude PPO due to computational limitations. We also incorporate three improved versions of DPO: TDPO (Zeng et al., 2024), ORPO (Hong et al., 2024), and SimPO (Meng et al., 2024). TDPO transformers the DPO loss into token-level to make its objective closer to SFT. SimPO adds on a length-normalization term to replace the regularization from reference model. ORPO adds the SFT loss and a DPO-like loss together, achieving alignment directly without SFT and reference model. In addition to reproducing the algorithms mentioned above, we also consider Zephyr-7B-beta (Tunstall et al., 2023) and Mistral-ORPO-alpha (Hong et al., 2024), two opensource checkpoints that utilize sequential and direct recipes respectively. Both of them used start-point models and datasets similar to ours.

Benchmarks. We consider two types of benchmarks. One is from the widely used Open-LLM LeaderBoard, which contains ARC-Challenge(25- shot) (Clark et al., 2018), MMLU(5-shot) (Chung et al., 2024), TruthfulQA(0-shot) (Lin et al., 2021), WinoGrande(5-shot) (Sakaguchi et al., 2021), and GSM8K(5-shot) (Cobbe et al., 2021). The other is LM-based evaluation, including TL;DR (Völske et al., 2017), Alpaca-Eval, and Alpaca-Eval-2 (Dubois et al., 2024). As for TL;DR, we keep the same setting as (Rafailov et al., 2024), using GPT-4 to judge the win-rate between model’s generation and ground truth answer. We utilize chat template for all benchmarks to obtain a more accurate evaluation for chat models.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Reference</td><td colspan="2">Data</td><td colspan="2">Alpaca-Eval</td><td colspan="2">Alpaca-Eval-2</td><td rowspan="2">TL;DR win-rate</td></tr><tr><td>pairwise</td><td>volume</td><td>win-rate</td><td>lc win-rate</td><td>win-rate</td><td>lc win-rate</td></tr><tr><td>Mistral-7B</td><td>-</td><td>-</td><td>120k</td><td>24.72</td><td>11.57</td><td>1.25</td><td>0.35</td><td>92.03</td></tr><tr><td colspan="9">fine-tuning with UltraFeedback-60k</td></tr><tr><td>+SFT</td><td>X</td><td>X</td><td>120k</td><td>82.56</td><td>78.32</td><td>7.09</td><td>8.67</td><td>84.22</td></tr><tr><td>+DPO</td><td>√</td><td>√</td><td>120k</td><td>74.00</td><td>73.12</td><td>9.73</td><td>8.58</td><td>77.25</td></tr><tr><td>+TDPO</td><td>√</td><td>√</td><td>120k</td><td>65.74</td><td>51.41</td><td>4.99</td><td>3.47</td><td>70.82</td></tr><tr><td>+ORPO</td><td>X</td><td>√</td><td>120k</td><td>85.14</td><td>76.60</td><td>8.82</td><td>12.34</td><td>89.24</td></tr><tr><td>+SimPO</td><td>X</td><td>√</td><td>120k</td><td>83.08</td><td>64.30</td><td>24.47</td><td>20.31</td><td>59.13</td></tr><tr><td>+IFT</td><td>X</td><td>X</td><td>120k</td><td>85.18</td><td>78.78</td><td>9.95</td><td>13.27</td><td>92.63</td></tr><tr><td>Mistral-ORPO-α</td><td>X</td><td>√</td><td>120k</td><td>87.92</td><td>-</td><td>-</td><td>11.33</td><td>-</td></tr><tr><td colspan="9">fine-tuning with UltraChat-200k + UltraFeedback-60k sequentially</td></tr><tr><td>+SFT</td><td>X</td><td>X</td><td>200k</td><td>86.69</td><td>77.96</td><td>4.08</td><td>6.43</td><td>98.11</td></tr><tr><td>+SFT + SFT</td><td>X</td><td>X</td><td>260k</td><td>86.34</td><td>76.98</td><td>4.55</td><td>7.14</td><td>97.79</td></tr><tr><td>+SFT + DPO</td><td>√</td><td>√</td><td>320k</td><td>91.62</td><td>81.54</td><td>10.08</td><td>13.72</td><td>99.18</td></tr><tr><td>+SFT + TDPO</td><td>√</td><td>√</td><td>320k</td><td>89.80</td><td>76.44</td><td>9.25</td><td>14.15</td><td>98.89</td></tr><tr><td>+SFT + ORPO</td><td>X</td><td>√</td><td>320k</td><td>86.26</td><td>79.67</td><td>7.40</td><td>12.27</td><td>97.92</td></tr><tr><td>+SFT + SimPO</td><td>X</td><td>√</td><td>320k</td><td>88.79</td><td>68.88</td><td>19.62</td><td>23.94</td><td>98.23</td></tr><tr><td>+SFT + IFT</td><td>X</td><td>X</td><td>260k</td><td>88.37</td><td>81.29</td><td>10.26</td><td>14.34</td><td>98.57</td></tr><tr><td>Zephyr-7B-β</td><td>√</td><td>√</td><td>320k</td><td>90.60</td><td>-</td><td>-</td><td>10.99</td><td>-</td></tr></table>

Table 3: Evaluation on LLM-based Benchmarks. IFT secures top two rankings in nearly all tasks, including conversation and summarization. When fine-tuned on limited data from UltraFeedback, IFT demonstrates a significant lead in TL;DR.

# 5.2 Main Results in NLP Tasks

Effectiveness on Sequential Recipe. In this scenario, IFT demonstrates good performance across benchmarks having standard answers or not (See Table 2 and 3 for details). On Open-LLM Leaderboard, IFT showcases the best average capabilities across all tasks, excelling particularly in tasks requiring generation, reasoning and fact-following abilities, such as TruthfulQA and GSM8K. However, IFT has a relatively large gap between DPO in multi-choice tasks like ARC-Challenge and MMLU. When evaluated for conversation and summarization judged by GPT-4, IFT’s performance is comparable to that of the chosen baselines. Remarkably, IFT achieves these results using the least amount of data and computational resources among all the methods tested.

Effectiveness of Preference Optimization Alone. IFT not only maintains the performance advantages compared with other baselines in this setting. But also, IFT performs comparably or even superiorly to many method in sequential recipe (See Table 2, 3, and Appendix C for details). While DPO, SimPO and TDPO tend to fail under this setting, ORPO remains competitive in its open-source model. However, when constrained in the same experiment setting, the performance of ORPO becomes worse than IFT. Additionally, the reliance on preference data makes ORPO more costly in terms of negative sampling, preference labeling, and GPU memory consumption. Consequently, IFT stands out as a more efficient and cost-effective alternative in this context.

Multi-Choice vs. Generation. IFT performs better on generation tasks but struggles with multichoice, whereas DPO exhibits the opposite performance. This may due to differences in evaluation metrics and training objectives (Zheng et al., 2023; Plaut et al., 2024; Tsvilodub et al., 2024). Multi-choice tasks evaluate log-likelihood for entire answers, while generation tasks require tokenby-token construction for causality and reasoning. DPO aligns the mapping between instructions and complete answers, while IFT emphasizes token-level causal relationships. As a result, DPO tends to excel in multi-choice tasks, while IFT performs better in token-by-token exploration tasks. In an ARC-Challenge adaptation to generation tasks, IFT demonstrates superiority without changing the benchmark’s distribution. Overall, IFT showcases its balanced performance across diverse tasks and achieving the highest average score.

Objective Trade-off between SFT and Preference Optimization. Traditional Preference Optimization methods deliver excellent alignment performance, particularly in enhancing the instructionfollowing ability of language models, as showed in Table 3. However, fitting the different objectives of SFT and Preference Optimization involves trade-offs (Tunstall et al., 2023). Even slight overfitting on SFT may result in reduced effectiveness of Preference Optimization. This phenomenon is also observed in Table 2, where the models trained by sequential recipe of SFT and other Preference Optimization methods showcase obvious inferior results on Open-LLM Leaderboard even worse than SFT alone. Avoiding this trade-off, ORPO and IFT can achieve better and more stable performance by directly conducting alignment on the base model.

Efficiency and Scaling Potential of IFT. Although IFT achieves comparable or superior performance to other methods, it also boasts high efficiency in many aspects. IFT does not require a reference model, which conserves GPU memory and computational resources. Most importantly, IFT and SFT are the only methods that conduct alignment without preference data, offering significant benefits as follows. Firstly, this characteristic eliminates the need for synchronous storage and computation of pairwise data on the GPU, thereby reducing memory consumption and training duration. Secondly, negative sampling from models and human preference-labeling are no longer necessary, eliminating the highest cost associated with alignment, which has been a discarded but fundamental challenge in research so far. Furthermore, using only the target answer brings the potential for scaling in alignment or even in pre-training.

# 5.3 Further Validation in Frozen-Lake Environment

As scores on Open-LLM Leaderboard only partially reflect models’ performance, and GPT-4 inadequately models human language generation, further comparison to a truly optimal policy is necessary. Given the difficulty of obtaining an optimal policy representing human language, we validate our algorithm in a simplified setting called Frozen Lake (Farama, 2023). In this environment, an agent attempts to find a gift on a nearly frozen lake with several holes, terminating the game upon finding the gift or falling into a hole. The limited number of states and actions in this game allows the optimal policy to be easily derived using classical RL methods.

![](images/a429b9f2e16dc8d1488842f84b83138a0436578a31ea61ab0a25a88b2d37fac0.jpg)

<details>
<summary>line</summary>

| Fine-tuning Steps | SFT    | DPO    | ORPO   | IFT    |
| ----------------- | ------ | ------ | ------ | ------ |
| 0                 | 0.0900 | 0.0900 | 0.0900 | 0.0900 |
| 20                | 0.0905 | 0.0905 | 0.0905 | 0.0905 |
| 40                | 0.0910 | 0.0910 | 0.0910 | 0.0910 |
| 60                | 0.0915 | 0.0915 | 0.0915 | 0.0915 |
| 80                | 0.0920 | 0.0920 | 0.0920 | 0.0920 |
| 100               | 0.0925 | 0.0925 | 0.0925 | 0.0925 |
| 120               | 0.0930 | 0.0930 | 0.0930 | 0.0930 |
| 140               | 0.0935 | 0.0935 | 0.0935 | 0.0935 |
| 160               | 0.0940 | 0.0940 | 0.0940 | 0.0940 |
</details>

Figure 3: The Frozen Lake Game. Considering the MSE distance between transition matrices of the trained and optimal policy, IFT performs much better than SFT and ORPO, but slightly worse than DPO.

To simulate parameterized policy alignment, we employ a two-layer fully connected neural network and design the environment with one optimal and one sub-optimal trajectory. The optimal parameterized policy is trained using the previously obtained optimal state-action transition matrix, and various fine-tuning methods from LMs are compared. We evaluate performance by measuring the MSE distance between the transition matrices of the optimal and trained policy. We didn’t count in TDPO and SimPO, as their objectives are similar as DPO in Frozen Lake Game.

In this setting, IFT achieves a significantly better policy than SFT and ORPO, although it performs slightly worse than DPO. This is partly because, in terms of comparing how closely the explored grid aligns with the agent’s preference, the order is DPO > IFT > ORPO > SFT. Although ORPO also considers the negative trajectories sampled from policy, its direct incorporation of SFT loss with a fusion coefficient deviates its preference estimation, partially diminishing its effectiveness. Additionally, DPO, ORPO and IFT explore more grids than SFT, which helps the agent develop a better understanding of the environment.

# 6 Related Work

Classical Reinforcement learning (RL) has demonstrated strong performance in various sequential decision-making and optimal control domains, including robotics (Levine et al., 2018), computer games (Vinyals et al., 2019) and others (Guan et al., 2021). There are two main categories of RL algorithms: value-based and policy-based, depending on whether they learn a parameterized policy. Value-based RL aims to fit an value function defined by Bellman Equation, containing methods such as Monte-Carlo (MC) Learning (Lazaric et al., 2007) and Temporal Difference Learning (Sutton, 1988; Seijen and Sutton, 2014). However, value-based methods struggle in continuous or large discrete space for its greedy objective. Thus, policy-based methods were introduced to model the decision-making process using a parameterized policy. As one of its best-known algorithms, Proximal Policy Optimization (PPO) (Schulman et al., 2017) is widely used in various domains, including Natural Language Processing (NLP).

Alignment for LMs has emerged as a crucial task these years, which adjusts the LMs’ generation distribution in line with human preferences (Bradley and Terry, 1952; Ziegler et al., 2019; Ouyang et al., 2022; Lee et al., 2023). While PPO remains the primary algorithm for alignment, its high demands for computation and memory hinders its broader use. Consequently, many improved methods have been proposed (Dong et al., 2023; Yuan et al., 2023; Zhao et al., 2023). Among them, DPO (Rafailov et al., 2024) unifies reward modeling and policy optimization by utilizing a loss function derived from PPO, training a single model to serve as both a policy model and a reward model. Without sacrificing performance, DPO decrease the costly consumption of PPO through directly value iteration similar to a preference-based format of MC instead of TD. However, it still relies on an expensive preference-labeling process and requires an SFT-based warm-up stage, which may introduce trade-offs when aligning the objectives of SFT and Preference Optimization.

Improved Versions of DPO come out one after another. Efforts such as (Liu et al., 2023b; Khaki et al., 2024; Yin et al., 2024; Guo et al., 2024b; Bansal et al., 2024; Liu et al., 2024) try to enhance the contrastive learning by utilizing better ranking strategies, more informative data, or more number of negative samples. Except for using offline data, (Liu et al., 2023a; Yuan et al., 2024; Guo et al., 2024a; Calandriello et al., 2024; Chen et al., 2024; Mitra et al., 2024) focus on online sampling and automated label/reward collection, reducing the manual cost required for alignment. Methods like (Ethayarajh et al., 2024; Hong et al., 2024) aim to reduce DPO’s dependency on SFT warm-up by transforming its loss functions and data format into a SFT manner. These algorithms handle positive and negative samples using SFT objective and Unlikelihood Training (Welleck et al., 2019), respectively. Recently, (Zeng et al., 2024; Meng et al., 2024) improved the integration of the SFT and DPO by introducing various regularization terms. These terms prevent the policy model from overfitting DPO objective and deviating from SFT objective. However, the actual volume of training data is not decreased in these methods. Also, GPU-memory-consuming pair-wise data is still required, while the need for a reference model and preference-labeling for the entire answer trajectory is only eliminated in limited cases.

# 7 Conclusion

In this paper, we first interpret SFT and typical Preference Optimization methods into a unified framework using Preference Estimation and Transition Optimization. Through this modeling, we found the biased prior used in SFT is one of the main reasons why SFT performs worse than other Preference Optimization methods. Then, we introduce an efficient and effective method called Intuitive Fine-Tuning (IFT), which achieves alignment directly from the base model using non-preferencelabeled data. Finally, experiments on widely used NLP benchmarks and Frozen Lake environment demonstrate the competitive performance of IFT.

# 8 Limitations

Our validation of IFT is limited to the fine-tuning setting, where data volume is constrained, leaving the scalability of IFT unexplored.

# Acknowledgments

This work is supported by the National Science and Technology Major Project (2023ZD0121403), and the Beijing Natural Science Foundation (IS23059). We further extend our gratitude to Yue Yu, Yihao Liu, Che Jiang, Xuekai Zhu, Jingkun Yang, Xuanqi Dong, Hong Liu, and Chushu Zhou for their insightful discussions with us.

# References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.   
Hritik Bansal, Ashima Suvarna, Gantavya Bhatt, Nanyun Peng, Kai-Wei Chang, and Aditya Grover. 2024. Comparing bad apples to good oranges: Aligning large language models via joint preference optimization. arXiv preprint arXiv:2404.00530.   
Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency, pages 610–623.   
Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. 2021. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.   
Ralph Allan Bradley and Milton E Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324– 345.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.   
Daniele Calandriello, Daniel Guo, Remi Munos, Mark Rowland, Yunhao Tang, Bernardo Avila Pires, Pierre Harvey Richemond, Charline Le Lan, Michal Valko, Tianqi Liu, et al. 2024. Human alignment of large language models through online preference optimisation. arXiv preprint arXiv:2403.08635.   
Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. 2024. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335.   
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2024. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53.   
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.   
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias

Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. 2023. Ultrafeedback: Boosting language models with high-quality feedback. arXiv preprint arXiv:2310.01377.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. 2023. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. 2024. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems, 36.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306.

Farama. 2023. Frozen lake. https://gymnasium. farama.org/environments/toy\_text/frozen\_ lake/. Accessed: 2024-05-19.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Yang Guan, Shengbo Eben Li, Jingliang Duan, Jie Li, Yangang Ren, Qi Sun, and Bo Cheng. 2021. Direct and indirect reinforcement learning. International Journal of Intelligent Systems, 36(8):4439–4467.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, et al. 2024a. Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792.

Yiju Guo, Ganqu Cui, Lifan Yuan, Ning Ding, Jiexin Wang, Huimin Chen, Bowen Sun, Ruobing Xie, Jie Zhou, Yankai Lin, et al. 2024b. Controllable preference optimization: Toward controllable multi-objective alignment. arXiv preprint arXiv:2402.19085.

Jiwoo Hong, Noah Lee, and James Thorne. 2024. Reference-free monolithic preference optimization with odds ratio. arXiv preprint arXiv:2403.07691.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.   
Saeed Khaki, JinJin Li, Lan Ma, Liu Yang, and Prathap Ramachandra. 2024. Rs-dpo: A hybrid rejection sampling and direct preference optimization method for alignment of large language models. arXiv preprint arXiv:2402.10038.   
Alessandro Lazaric, Marcello Restelli, and Andrea Bonarini. 2007. Reinforcement learning in continuous action spaces through sequential monte carlo methods. Advances in neural information processing systems, 20.   
Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. 2023. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267.   
Sergey Levine, Peter Pastor, Alex Krizhevsky, Julian Ibarz, and Deirdre Quillen. 2018. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. The International journal of robotics research, 37(4-5):421–436.   
Bo Li, Peng Qi, Bo Liu, Shuai Di, Jingen Liu, Jiquan Pei, Jinfeng Yi, and Bowen Zhou. 2022. Trustworthy ai: From principles to practices. ACM Comput. Surv. Just Accepted.   
Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958.   
Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J Liu, and Jialu Liu. 2023a. Statistical rejection sampling improves preference optimization. arXiv preprint arXiv:2309.06657.   
Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. 2023b. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning. arXiv preprint arXiv:2312.15685.   
Xiao Liu, Xixuan Song, Yuxiao Dong, and Jie Tang. 2024. Extensive self-contrast enables feedbackfree language model alignment. arXiv preprint arXiv:2404.00604.   
Yu Meng, Mengzhou Xia, Danqi Chen, and et al et al. 2024. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734.   
Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. 2024. Orca-math: Unlocking the potential of slms in grade school math. arXiv preprint arXiv:2402.14830.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.   
Benjamin Plaut, Khanh Nguyen, and Tu Trinh. 2024. Softmax probabilities (mostly) predict large language model correctness on multiple-choice q&a. arXiv preprint arXiv:2402.13213.   
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.   
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106.   
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.   
Harm Seijen and Rich Sutton. 2014. True online td (lambda). In International Conference on Machine Learning, pages 692–700. PMLR.   
Richard S Sutton. 1988. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44.   
Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, and Aviral Kumar. 2024. Preference fine-tuning of llms should leverage suboptimal, on-policy data. arXiv preprint arXiv:2404.14367.   
Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.   
Polina Tsvilodub, Hening Wang, Sharon Grosch, and Michael Franke. 2024. Predictions from language models for multiple-choice tasks are not robust under variation of scoring methods. arXiv preprint arXiv:2403.00998.   
Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, et al. 2023. Zephyr: Direct distillation of lm alignment. arXiv preprint arXiv:2310.16944.   
Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. 2019. Grandmaster level in starcraft

ii using multi-agent reinforcement learning. Nature, 575(7782):350–354.   
Michael Völske, Martin Potthast, Shahbaz Syed, and Benno Stein. 2017. Tl; dr: Mining reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization, pages 59– 63.   
Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. 2019. Neural text generation with unlikelihood training. arXiv preprint arXiv:1908.04319.   
Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. 2023. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. In ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models.   
Yueqin Yin, Zhendong Wang, Yi Gu, Hai Huang, Weizhu Chen, and Mingyuan Zhou. 2024. Relative preference optimization: Enhancing llm alignment through contrasting responses across identical and diverse prompts. arXiv preprint arXiv:2402.10958.   
Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. 2024. Self-rewarding language models. arXiv preprint arXiv:2401.10020.   
Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. 2023. Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302.   
Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. 2024. Tokenlevel direct preference optimization. arXiv preprint arXiv:2404.11999.   
Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. 2024. Negative preference optimization: From catastrophic collapse to effective unlearning. arXiv preprint arXiv:2404.05868.   
Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. 2023. Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425.   
Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2023. Large language models are not robust multiple choice selectors. In The Twelfth International Conference on Learning Representations.   
Bowen Zhou and Ning Ding. 2024. Generative ai for complex scenarios: Language models are sequence processors. International Journal of Artificial Intelligence and Robotics Research.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593.

# A Theoretical Details

# A.1 MDP In LMs

$$
\mathcal {M} = (S, A, \mathcal {T}, r, \rho_ {0}) {:}
$$

• A, the concrete action space, consisting of $N _ { A }$ vocabularies as defined by the tokenizer.   
• S, the concrete state space, comprising $N _ { S } =$ $( N _ { A } ) ^ { N }$ elements related to sequence length $N$ Each state represents a ordered permutation of vocabularies.   
• $\rho _ { 0 }$ , the initial state of each generation, typically refers to the given instruction;   
• $\tau \in R ^ { N _ { S } \times N _ { A } }$ , the state-action transition matrix of a given policy, indicating the probability of generating each token given different states;   
• $r ,$ the reward assigned to a particular stateaction pair.

# A.2 Loss Function of IFT

The disparities of transition between model and human can be formalized as follows:

$$
\begin{array}{l} \mathcal {L} (\hat {T} _ {\theta}; T ^ {*}) = \mathbb {E} _ {\rho_ {0} \sim \mathcal {D}} \mathbb {E} _ {s _ {n} ^ {*} \sim \mathcal {S} _ {\rho_ {0}} ^ {*}} \\ \left[ - \sum_ {n = 0} ^ {N} \log \frac {\hat {T} _ {\theta} (s _ {n} ^ {*} , \rho_ {0})}{T ^ {*} (s _ {n} ^ {*} , \rho_ {0})} \right] \tag {23} \\ \end{array}
$$

We make the same hypothesis as SFT that the optimization objective of each target intermediate state has a probability equal to 1, so that $\forall n \in [ 0 , N ] , \rho _ { 0 } \in \mathcal { D } , s _ { n } ^ { * } \in \mathcal { S } _ { \rho _ { 0 } } ^ { * }$ :

$$
T ^ {*} (s _ {n} ^ {*}, \rho_ {0}) = 1 = T ^ {*} (s _ {N} ^ {*}, \rho_ {0}) \tag {24}
$$

Thus, the objective of IFT can be represented directly by the following loss function:

$$
\begin{array}{l} \mathcal {L} (\hat {T} _ {\theta}) = \mathbb {E} _ {\rho_ {0} \sim \mathcal {D}} \mathbb {E} _ {s _ {i} ^ {*} \sim \mathcal {S} _ {\rho_ {0}} ^ {*}} \\ \left[ - \sum_ {n = 0} ^ {N} \log \mathcal {T} _ {\theta} \left(\pi^ {*} \left(\delta^ {*} (s _ {i} ^ {*})\right), \delta_ {\theta} (s _ {i} ^ {*})\right) \right] \tag {25} \\ \end{array}
$$

As the optimal policy enjoys the optimal transition:

$$
s _ {i} ^ {*} = \left[ s _ {i - 1} ^ {*}, a _ {i} ^ {*} \right] = \left[ s _ {i - 1} ^ {*}, \pi^ {*} \left(s _ {i - 1} ^ {*}\right) \right] = \Pi^ {*} \left(s _ {i - 1} ^ {*}\right) \tag {26}
$$

Therefore, the disturbed optimal state keeps similar with the original optimal state:

$$
\delta^ {*} (s _ {i} ^ {*}) = (1 - \lambda) s _ {i} ^ {*} + \lambda \Pi^ {*} (s _ {i - 1} ^ {*}) = s _ {i} ^ {*} \tag {27}
$$

Then, the final loss function can be presented as:

$$
\begin{array}{l} \mathcal {L} (\mathcal {T} _ {\theta}, \delta_ {\theta}) = \mathbb {E} _ {\rho_ {0} \sim \mathcal {D}} \mathbb {E} _ {s _ {i} ^ {*} \sim \mathcal {S} _ {\rho_ {0}} ^ {*}} \\ \left[ - \sum_ {n = 0} ^ {N} \log \mathcal {T} _ {\theta} (a _ {i} ^ {*}, \delta_ {\theta} (s _ {i} ^ {*})) \right] \tag {28} \\ \end{array}
$$

# A.3 Proof for Bellman Equation

Considering only one sampled state $s _ { n } ^ { * }$ constrained by $\rho _ { 0 }$ in the datasets, we have:

$$
\begin{array}{l} \exp \left(- \mathcal {L} (\hat {T} _ {\theta} (s _ {n} ^ {*}, \rho_ {0}))\right) \\ = \mathcal {T} _ {\theta} (a _ {n} ^ {*}, \delta_ {\theta} (s _ {n} ^ {*})) \left(\sum_ {n + 1} ^ {N} \mathcal {T} _ {\theta} (a _ {i} ^ {*}, \delta_ {\theta} (s _ {i} ^ {*}))\right) \tag {29} \\ \end{array}
$$

$$
= \max _ {a} \left[ \mathcal {T} _ {\theta} (a, s _ {n} ^ {*}) \big (r + \gamma V (s _ {n + 1} ^ {\hat {\theta}}) \big) \right]
$$

$$
= V _ {\theta} (\hat {s _ {n} ^ {\theta}})
$$

where $r = ( 1 - \gamma ) V ( s _ { n + 1 } ^ { \hat { \theta } } )$ . This reward function implicitly accounts for the influence of the current prediction on future generations.

# A.4 Reformulation of Typical Methods

We reformulate the loss function of some methods using the disparities of transition matrices as:

SFT

$$
\mathcal {L} _ {\mathrm{SFT}} = \mathbb {E} _ {\rho_ {0} \sim \mathcal {D}} \mathbb {E} _ {s _ {i} ^ {*} \sim \mathcal {S} _ {\rho_ {0}} ^ {*}} \left[ - \sum_ {i = 0} ^ {N} \log \mathcal {T} _ {\theta} (\pi^ {*} (s _ {i} ^ {*}), s _ {i} ^ {*}) \right] \tag {30}
$$

where the human’s preference is unbiasedly estimated, but the model’s preference is inaccurately represented by $s _ { i } ^ { * }$ .

PPO

$$
\mathcal {L} _ {\mathrm{PPO}} = \mathbb {E} _ {\rho_ {0} \sim \mathcal {D}} \mathbb {E} _ {s _ {i} ^ {*} \sim \mathcal {S} _ {\rho_ {0}} ^ {*}} \left[ - \sum_ {i = 0} ^ {N} \mathcal {R} \left(\pi_ {\theta} \left(s _ {i} ^ {\theta}\right), s _ {i} ^ {\theta}\right) \right] \tag {31}
$$

where $\mathcal { R } \in ( - \infty , 0 ]$ denotes the degree of closeness between human preferences and the stateaction pairs chosen by model. The reward and loss will be zero only if the state-action pairs perfectly align with human preferences. Thus, PPO implicitly models the human policy $\pi ^ { * }$ through reward modeling, which can be formulated as follows:

$$
\mathcal {R} = \pi_ {\mathcal {R}} \leftarrow \min _ {\pi} \mathcal {L} _ {\mathcal {R}} \tag {32}
$$

$$
\begin{array}{l} \mathcal {L} _ {\mathcal {R}} = \mathbb {E} _ {\rho_ {0} \sim \mathcal {D}} \mathbb {E} _ {s _ {i} ^ {+} \sim \mathcal {S} _ {\rho_ {0}} ^ {+}, s _ {i} ^ {-} \sim \mathcal {S} _ {\rho_ {0}} ^ {-}} \\ \left[ - \log \sigma \bigg (\sum_ {i = 0} ^ {N} \log \mathcal {T} _ {\mathcal {R}} (\pi^ {+} (s _ {i} ^ {+}) | s _ {i} ^ {+}) \right. \\ \left. \left. - \sum_ {i = 0} ^ {N} \log \mathcal {T} _ {\mathcal {R}} \left(\pi^ {-} \left(s _ {i} ^ {-}\right) \mid s _ {i} ^ {-}\right)\right) \right] \tag {33} \\ \end{array}
$$

# DPO-Online

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DPO}} = \mathbb {E} _ {\rho_ {0} \sim \mathcal {D}} \mathbb {E} _ {s _ {i} ^ {*} \sim \mathcal {S} _ {\rho_ {0}} ^ {*}, s _ {i} ^ {\theta} \sim \mathcal {S} _ {\rho_ {0}} ^ {\theta}} \\ \left[ - \log \sigma \left(\sum_ {i = 0} ^ {N} \log \mathcal {T} _ {\theta} \left(\pi^ {*} \left(s _ {i} ^ {*}\right), s _ {i} ^ {*}\right) \right. \right. \\ \left. \left. - \sum_ {i = 0} ^ {N} \log \mathcal {T} _ {\theta} (\pi_ {\theta} (s _ {i} ^ {\theta}), s _ {i} ^ {\theta})\right) \right] \tag {34} \\ \end{array}
$$

Ideally, this loss function increases the probabilities of state-action pairs preferred by humans and decreases the probabilities of those chosen by the model. It unbiasedly estimate both the human’s and model’s preference.

# DPO-Offline

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DPO}} = \mathbb {E} _ {\rho_ {0} \sim \mathcal {D}} \mathbb {E} _ {s _ {i} ^ {+} \sim \mathcal {S} _ {\rho_ {0}} ^ {+}, s _ {i} ^ {-} \sim \mathcal {S} _ {\rho_ {0}} ^ {-}} \\ \left[ - \log \sigma \bigg (\sum_ {i = 0} ^ {N} \log \mathcal {T} _ {\theta} (\pi^ {+} (s _ {i} ^ {+}), s _ {i} ^ {+}) \right. \\ \left. \left. - \sum_ {i = 0} ^ {N} \log \mathcal {T} _ {\theta} \left(\pi^ {-} \left(s _ {i} ^ {-}\right), s _ {i} ^ {-}\right)\right) \right] \tag {35} \\ \end{array}
$$

In the offline circumstance, the positive samples can still represent the human preference correctly, as $s ^ { + }$ is usually similar to $s ^ { * }$ . However, this is not the case for negative samples.As training progresses, s− becomes more and more out-ofdistributions compared to the model’s preferred state $s ^ { \theta }$ , leading to biased estimations.

# IFT

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{IFT}} = \mathbb {E} _ {\rho_ {0} \sim \mathcal {D}} \mathbb {E} _ {s _ {i} ^ {*} \sim \mathcal {S} _ {\rho_ {0}} ^ {*}} \\ \left[ - \sum_ {n = 0} ^ {N} \sum_ {i = n} ^ {N} \log \mathcal {T} _ {\theta} (a _ {i} ^ {*}, \delta_ {\theta} (s _ {i} ^ {*})) \right] (36) \\ \delta_ {\theta} (s _ {i} ^ {*}) = (1 - \lambda) s _ {i} ^ {*} + \lambda \pi_ {\theta} (s _ {i - 1} ^ {*}) (37) \\ \end{array}
$$

By using a model-based disturbance function, IFT constructs a residual connection in the temporal dimension, providing a better estimation for the model than SFT. Through this approach, IFT implicitly implements a Relation Propagation in the Transition Optimization stage, which considers the influence of current predictions on future outcomes. This propagation also reduces the influence of bias introduced by inaccurate estimations in earlier positions.

# B Implementation Details

# B.1 NLP Settings

For the coefficient β in DPO, TDPO, ORPO and SimPO, we use 0.1, 0.1, 0.25 and 2.0 respectively, as presented in their original papers. For the coefficient $\gamma / \beta$ ration in SimPO, we use 0.8 to keep the same setting in its original papers. For IFT, we choose 0.2 for λ and incorporate a decay factor of 0.95 to fitting better with the Bellman Equation. We save checkpoints every 20k steps and select the results from the checkpoint with the best average score to demonstrate the performance of each method.

<table><tr><td>Name</td><td>Value</td></tr><tr><td>epoch</td><td>3</td></tr><tr><td>mini batch size</td><td>8</td></tr><tr><td>gradient accumulation step</td><td>64</td></tr><tr><td>warmup ratio</td><td>0.1</td></tr><tr><td>scheduler</td><td>cosine</td></tr><tr><td>learning rate</td><td>5e-7</td></tr><tr><td>optimizer</td><td>RMSprop</td></tr><tr><td>precision</td><td>bfloat16</td></tr></table>

Table 4: Hyper-Parameters in NLP Setting

We implement our main experiments on four NVIDIA A6000 GPUs. When using 60k singletarget data, the entire training process for SFT and IFT takes approximately 20 hours, with each epoch lasting 7 hours. When using 60k pair-wise data, the training process for DPO and ORPO takes around 40 hours and 30 hours respectively, due to the differences in requirements for a reference model.

# B.2 Frozen Lake Setting

We keep the similar hyper-parameters as in NLP setting for Frozen Lake game, running this environment on CPUs. Since our designed environment includes an optimal and a sub-optimal trajectory, we select the optimal trajectory as the target for SFT and IFT. For DPO and ORPO, the optimal and sub-optimal trajectories are used as positive and negative samples, respectively.

# C More Experimental Results

<table><tr><td>Method</td><td>ARC</td><td>ARC-Gen</td><td>MMLU</td><td>TruthfulQA</td><td>WinoGrande</td><td>GSM8K</td><td>Avg.</td></tr><tr><td>Gemma-2B</td><td>42.75</td><td>43.17</td><td>35.68</td><td>35.25</td><td>66.46</td><td>16.98</td><td>39.42</td></tr><tr><td colspan="8">fine-tuning with Gemma2-UltraFeedback-armnorm-60k</td></tr><tr><td>+ SFT</td><td>42.06</td><td>42.75</td><td>34.30</td><td>41.49</td><td>64.88</td><td>21.53</td><td>40.85</td></tr><tr><td>+ DPO</td><td>41.30</td><td>40.61</td><td>35.47</td><td>30.11</td><td>65.51</td><td>18.95</td><td>38.27</td></tr><tr><td>+ TDPO</td><td>41.21</td><td>40.70</td><td>35.62</td><td>31.33</td><td>65.04</td><td>18.88</td><td>38.42</td></tr><tr><td>+ ORPO</td><td>41.89</td><td>42.06</td><td>36.43</td><td>41.98</td><td>65.90</td><td>20.54</td><td>41.35</td></tr><tr><td>+ SimPO</td><td>41.38</td><td>40.10</td><td>35.32</td><td>28.76</td><td>65.27</td><td>20.39</td><td>38.22</td></tr><tr><td>+ IFT</td><td>42.49</td><td>42.66</td><td>35.77</td><td>45.41</td><td>66.14</td><td>22.14</td><td>42.39</td></tr><tr><td>LLaMA-3-8B</td><td>49.40</td><td>73.89</td><td>62.17</td><td>46.63</td><td>76.80</td><td>50.26</td><td>57.05</td></tr><tr><td colspan="8">fine-tuning with LLaMA3-UltraFeedback-armnorm-60k</td></tr><tr><td>+ SFT</td><td>52.83</td><td>75.00</td><td>63.24</td><td>50.42</td><td>76.95</td><td>51.09</td><td>58.91</td></tr><tr><td>+ DPO</td><td>51.19</td><td>74.23</td><td>62.21</td><td>36.35</td><td>76.24</td><td>51.25</td><td>55.45</td></tr><tr><td>+ TDPO</td><td>51.37</td><td>74.31</td><td>62.50</td><td>39.66</td><td>76.50</td><td>51.73</td><td>56.36</td></tr><tr><td>+ ORPO</td><td>54.18</td><td>74.98</td><td>63.46</td><td>54.83</td><td>77.06</td><td>51.18</td><td>60.14</td></tr><tr><td>+ SimPO</td><td>53.92</td><td>74.48</td><td>62.76</td><td>38.07</td><td>76.56</td><td>51.93</td><td>56.57</td></tr><tr><td>+ IFT</td><td>54.69</td><td>75.08</td><td>63.20</td><td>57.64</td><td>77.27</td><td>51.78</td><td>60.92</td></tr></table>

Table 5: Evaluation on Open-LLM Leaderboard when fine-tuning with UltraFeedback-60k.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Reference</td><td colspan="2">Data</td><td colspan="2">Alpaca-Eval</td><td colspan="2">Alpaca-Eval-2</td></tr><tr><td>pairwise</td><td>volume</td><td>win-rate</td><td>lc win-rate</td><td>win-rate</td><td>lc win-rate</td></tr><tr><td>Gemma-2B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="8">fine-tuning with UltraFeedback-60k</td></tr><tr><td>+ SFT</td><td>✕</td><td>✕</td><td>120k</td><td>36.53</td><td>30.28</td><td>0.99</td><td>0.57</td></tr><tr><td>+ DPO</td><td>√</td><td>√</td><td>120k</td><td>3.13</td><td>1.18</td><td>0.13</td><td>0.23</td></tr><tr><td>+ TDPO</td><td>√</td><td>√</td><td>120k</td><td>2.14</td><td>0.70</td><td>0.25</td><td>0.10</td></tr><tr><td>+ ORPO</td><td>✕</td><td>√</td><td>120k</td><td>36.62</td><td>34.23</td><td>1.12</td><td>0.59</td></tr><tr><td>+ SimPO</td><td>✕</td><td>√</td><td>120k</td><td>4.48</td><td>2.42</td><td>0.13</td><td>0.15</td></tr><tr><td>+ IFT</td><td>✕</td><td>✕</td><td>120k</td><td>36.74</td><td>39.33</td><td>1.61</td><td>1.23</td></tr><tr><td colspan="8">fine-tuning with Gemma2-UltraFeedback-armnorm-60k</td></tr><tr><td>+ SFT</td><td>✕</td><td>✕</td><td>120k</td><td>39.33</td><td>32.36</td><td>0.86</td><td>0.69</td></tr><tr><td>+ DPO</td><td>√</td><td>√</td><td>120k</td><td>2.83</td><td>0.81</td><td>0.00</td><td>0.00</td></tr><tr><td>+ TDPO</td><td>√</td><td>√</td><td>120k</td><td>2.41</td><td>0.60</td><td>0.00</td><td>0.00</td></tr><tr><td>+ ORPO</td><td>✕</td><td>√</td><td>120k</td><td>43.46</td><td>34.19</td><td>2.06</td><td>1.21</td></tr><tr><td>+ SimPO</td><td>✕</td><td>√</td><td>120k</td><td>3.24</td><td>1.07</td><td>0.00</td><td>0.00</td></tr><tr><td>+ IFT</td><td>✕</td><td>✕</td><td>120k</td><td>51.23</td><td>37.76</td><td>2.14</td><td>1.33</td></tr></table>

Table 6: Evaluation on LLM-based Benchmarks when fine-tuning with UltraFeedback-60k.

<table><tr><td>Method</td><td>ARC</td><td>ARC-Gen</td><td>MMLU</td><td>TruthfulQA</td><td>WinoGrande</td><td>GSM8K</td><td>Avg.</td></tr><tr><td>Mistral-7B</td><td>53.07</td><td>73.04</td><td>59.14</td><td>45.29</td><td>77.58</td><td>38.89</td><td>54.79</td></tr><tr><td>+ SFT</td><td>56.49</td><td>74.00</td><td>60.44</td><td>55.57</td><td>77.90</td><td>42.84</td><td>58.65</td></tr><tr><td>+ IFT</td><td>56.74</td><td>74.15</td><td>60.49</td><td>57.65</td><td>78.45</td><td>44.73</td><td>59.61</td></tr><tr><td>+ IFT with noisy lambda</td><td>61.60</td><td>76.53</td><td>61.11</td><td>57.03</td><td>77.43</td><td>45.64</td><td>60.56</td></tr></table>

Table 7: Evaluation on Open-LLM Leaderboard when fine-tuning with UltraFeedback-60k.