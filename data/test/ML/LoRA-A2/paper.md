# Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients

Jabin Koo\*1, Minwoo Jang\*2, Jungseul Ok†1, 2

1Department of Computer Science and Engineering, POSTECH, South Korea 2Graduate School of Artificial Intelligence, POSTECH, South Korea

{jbkoo, minwoo, jungseul}@postech.ac.kr

# Abstract

Federated fine-tuning for Large Language Models (LLMs) faces significant challenges due to the heavy communication overhead of transmitting large model updates. Although Low Rank Adaptation (LoRA) has been proposed as a solution, yet its application in federated learning is complicated by discordance in aggregation. Existing methods addressing this discordance often suffer from performance degradation at low ranks in heterogeneous data settings. In response, we introduce LoRA-A2 (Low Rank Adaptation with Alternating freeze and Adaptive rank selection), which demonstrates robustness in challenging settings with low ranks and high data heterogeneity. Our experimental findings reveal that LoRA-A2 maintains performance even under extreme heterogeneity and low rank conditions, achieving up to a significant reduction in uploaded parameters compared to full fine-tuning without compromising performance. This adaptive mechanism increases robustness and communication efficiency in federated fine-tuning, enabling the practical deployment of LLMs in resourceconstrained environments.

# 1 Introduction

Large Language Models (LLMs), exemplified by ChatGPT (OpenAI, 2023), Llama (Dubey et al., 2024) and others, represent a hallmark of the current era. These models are being widely applied in real-world scenarios by fine-tuning them on various task-specific datasets (Dodge et al., 2020). With the expansion of edge devices, the potential to leverage rich, privacy-sensitive data for fine-tuning LLMs has shifted the focus toward federated fine-tuning. Despite its potential, this is often infeasible due to the large size of LLMs, which require extensive computational and communication resources from local devices.

Parameter-Efficient Fine-Tuning (PEFT) methods (Lester et al., 2021; Liu et al., 2022) are increasingly being explored in the context of federated fine-tuning. Among these, Low-Rank Adaptation (LoRA) (Hu et al., 2022) is particularly noteworthy for its significant reduction in number of communicated parameters. However, naive application of LoRA in Federated Learning (FL) (McMahan et al., 2017) environment comes with several challenges such as aggregation discordance. Although several solutions have been proposed, they often remain vulnerable to high heterogeneity and low ranks due to a limited parameter space, making it difficult to reduce rank size for communication efficiency in realistic FL scenarios.

To address this, we introduce LoRA-A2 (Low Rank Adaptation with Alternating freeze and Adaptive rank selection), which is robust to both high heterogeneity and low ranks. LoRA-A2 incorporates two main strategies: (1) alternating freeze, which switches between freezing LoRA modules B and A in each round, and (2) adaptive rank selection, which identifies and updates only important ranks in LoRA modules. We conduct experiments across various rank sizes and heterogeneity levels, comparing our algorithm with multiple baselines. Through the experiments, we reveal the vulnerabilities of existing methods and highlight the robustness of LoRA-A2 in challenging conditions, providing analyses of the reasons for its robustness. Additionally, we empirically demonstrate that our approach achieves performance comparable to or exceeding that of full fine-tuning, while uploading less than 0.2% of parameters to the server.

Our contributions can be summarized as follows:

• We address the vulnerabilities of previous federated LoRA methods in high heterogeneity and low-rank settings, and propose a novel algorithm, LoRA-A2, which demonstrates robustness in these challenging conditions.

![](images/3a4b1d0b3f39efc26e2da608b07f85fdd7b41de868555d324fac487d1882fbd9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph Odd Rounds
        A1["Hand with weights W"] --> B1["Orange B"]
        A2["Hand with weights A"] --> B2["Blue A"]
        A3["Hand with weights A"] --> B3["Blue B"]
        A4["Hand with weights B"] <--_B4["Orange B"]
        A5["Hand with weights M"] <--_B5["Blue M"]
    end

    subgraph Section 4.1 Alternating Freeze
        C["Cloud with weight update"] --> D["Pretrained Weights W ∈ ℝ^{d₁×d₂} → B = 0"]
        D --> E["ΔA"]
        D --> F["A = N(0,σ²)"]
        D --> G["A = N(0,σ²)"]
    end

    subgraph Even Rounds
        H["Hand with weights B"] <--_I["Orange B"]
        J["Hand with weights ΔA"] <--_K["Blue B"]
        L["Hand with weights M"] <-- L
        M["Hand with weights B"] <--_N["Blue B"]
        M --> O["Blue A"]
    end

    style Odd Rounds fill:#f9f,stroke:#333
    style Section 4.1 Alternating Freeze fill:#ccf,stroke:#333
    style Even Rounds fill:#cfc,stroke:#333
```
</details>

![](images/5027036bbbc76e0240419484e4a464eab25ad3686a0ad7faba0702f1a7c75c3d.jpg)

<details>
<summary>bar</summary>

| Category | Client | Row Count | Layer Count |
| -------- | ------ | --------- | ----------- |
| Motorcycle | Client 1 | 100 | 0 |
| Motorcycle | Client 2 | 80 | 0 |
| Motorcycle | Client 3 | 90 | 0 |
| Computer | Client 4 | 70 | 0 |
| Computer | Client 5 | 60 | 0 |
| Computer | ... | ... | ... |
| Medicine | Client K | 50 | 0 |
| Medicine | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target Module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target module | ... | ... | ... |
| Target model | ... | ... | ... |
| Target model | ... | ... | ... |
| Target model | ... | ... | ... |
| Target model | ... | ... | ... |
| Target model | ... | ... | ... |
| Target model | ... | ... | ... |
| Target model | ... | ... | ... |
| Target model | ... | ... | ... |
| Target model | ... | ... | ... |
| Target model | ... | ... | ... |
| Target system (labeled) | -      | -         | -           |
| # of selected ranks for each LoRA       |
| rG               |
| # of selected ranks for each LoRA   |
L                L                L
Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             Layer             layer          , 
    Note: The actual values are not provided in the code. The actual values are estimated based on the visual representation of the matrix structure.
</details>

Figure 1: An overview of the proposed method, $\mathrm { L o R A  – A ^ { 2 } }$ . It alternately trains B and A of the LoRA adapters, with each client training only a subset of the downloaded parameters. LoRA-A2 is free from several issues for using LoRA in FL, which are discussed in Section 3. A detailed explanation of the method is provided in Section 4.

• Our algorithm effectively reduces communication costs, achieving a significant reduction in uploaded parameters compared to full fine-tuning, while maintaining its performance.   
• We provide visualization on adaptive rank selection process and a thorough empirical exploration on how important ranks are efficiently trained and transmitted.

# 2 Related Works

LoRA with adaptive rank selection LoRA (Hu et al., 2022) is a widely used PEFT method for LLMs. It tries to approximate the updated part of the pre-trained model with two smaller matrices. This approach is inspired by previous studies (Li et al., 2018; Aghajanyan et al., 2021), which suggest that newly learned parameters for adaptation lie within a low dimensional subspace.

AdaLoRA (Zhang et al., 2023) assumes a scenario where the total parameter budget is limited. It adaptively selects the rank for each LoRA adapter under this constraint, with a criterion for rank selection based on singular values of the updated part.

ALoRA (Liu et al., 2024) utilizes a router for each LoRA adapter. The router determines which part of each LoRA adapter should be either turned on or off, enabling efficient fine-tuning via pruning. Similarly, DoRA (Mao et al., 2024) re-splits LoRA into smaller groups of LoRAs. During the training session, it estimates the importance of each small LoRA, allowing the parts with less contribution across the whole LoRA to be pruned. Our research extends these adaptive rank selection methods in centralized learning to the FL setting so that each client adaptively selects different ranks suitable for their own dataset.

Federated learning with LoRA As training LLMs on mobile devices becomes feasible, finetuning LLMs via FL has recently gained attention. In line with this trend, using LoRA for federated fine-tuning (Babakniya et al., 2023; Kuo et al., 2024; Wang et al., 2024), is also being considered. However, simply adopting LoRA for FL presents several obstacles, which are discussed in Section 3.

HetLoRA (Cho et al., 2023) assumes that each client may have different computational power, which is a common scenario in FL. Based on this assumption, it allows each client to use a LoRA adapter with a different rank. Zero-padding is then applied to align the dimensions of the clientspecific adapters for aggregation.

Sun et al. (2024) point out that aggregating the two matrices of a LoRA adapter separately cannot fully approximate the original LoRA adapter. Based on this finding, they propose FFA-LoRA, which addresses this issue by freezing half of each LoRA throughout the entire fine-tuning session.

FlexLoRA (Bai et al., 2024) aggregates the product of two matrices comprising each LoRA adapter and then decomposes the aggregated parameters back into two smaller matrices via Singular Value Decomposition (SVD). This approach allows FlexLoRA to overcome the challenges addressed by HetLoRA and FFA-LoRA, respectively, though at the expense of increased computational cost on the server-side for the decomposition process.

# 3 Problem Formulation

Low rank adaptation Because LLMs have billions of parameters, fine-tuning them for specific domains demands significant computational power, which may be infeasible in many situations. To address this issue, PEFT techniques such as LoRA (Hu et al., 2022) have recently gained attention, as they can effectively reduce the number of parameters that need to be trained. Specifically, when finetuning a pre-trained weight matrix W0 ∈ Rd1×d2 $\dot { W _ { 0 } } \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ to obtain W , LoRA achieves this by decomposing $\Delta W$ , the update of the weight matrix, into smaller matrices $B \in \mathbb { R } ^ { d _ { 1 } \times r }$ and $A \in \mathbb { R } ^ { r \times d _ { 2 } }$ :

$$
W = W _ {0} + \Delta W = W _ {0} + B A, \tag {1}
$$

where $r \ll \{ d _ { 1 } , d _ { 2 } \}$ denotes the rank of LoRA. With this approximation, the number of trainable parameters is reduced from $d _ { 1 } \cdot d _ { 2 } \tan r \cdot ( d _ { 1 } + d _ { 2 } )$ .

Federated LoRA and discordance problem Consider a global pre-trained model $W _ { 0 }$ and a set of clients $\{ 1 , 2 , \cdots , K \}$ . The objective in federated fine-tuning is to update W that is suitable for all lo $W _ { 0 }$ to obtaidatasets $\{ \mathcal { D } _ { k } \} _ { k = 1 } ^ { K }$ However, fine-tuning LLMs is very expensive for local devices in terms of both computation and communication, as billions of parameters must be trained and transmitted in each round.

LoRA presents a promising approach in FL for reducing communication costs, as only low rank matrices B and A are trained and transmitted, allowing the number of communicated parameters to be linearly reduced by the rank r of LoRA modules. However, the straightforward application of LoRA in FL introduces a significant issue known as discordance (Sun et al., 2024), primarily due to aggregation algorithms. In methods like FedAvg (McMahan et al., 2017), where each weight is aggregated individually, discordance occurs between the actual and aggregated parameters. That is,

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} w _ {k} \Delta W _ {k} = \sum_ {k = 1} ^ {K} w _ {k} B _ {k} A _ {k} \\ \neq \left(\sum_ {k = 1} ^ {K} w _ {k} B _ {k}\right) \left(\sum_ {k = 1} ^ {K} w _ {k} A _ {k}\right) \tag {2} \\ \end{array}
$$

![](images/860fa63d087f8bc3e5ada10f03c9bc5aa72a91a2eb0fa3b9883d854e75cbaad6.jpg)

<details>
<summary>line</summary>

| LoRA Rank | Accuracy (Line 1) | Accuracy (Line 2) |
| --------- | ----------------- | ----------------- |
| 16        | 92                | 80                |
| 8         | 91                | 78                |
| 4         | 90                | 77                |
| 2         | 88                | 75                |
| 1         | 85                | 73                |
</details>

(a) Dir(0.1)

![](images/5e9adadc72195bafc447c50a9570a59e3269e4caf7d8ebdc17d2561f405d39e3.jpg)

<details>
<summary>line</summary>

| LoRA Rank | FL + LoRA | FFA-LoRA | FlexLoRA |
| --------- | --------- | -------- | -------- |
| 16        | 60        | 42       | 75       |
| 8         | 60        | 40       | 70       |
| 4         | 55        | 38       | 55       |
| 2         | 50        | 35       | 50       |
| 1         | 45        | 33       | 45       |
</details>

(b) Dir(0.01)   
Figure 2: Accuracy of previous Federated LoRA methods across different rank sizes in heterogeneous data settings.

in general, where $\textstyle \sum _ { k = 1 } ^ { K } w _ { k } \ = \ 1$ with $w _ { k } ~ \geq ~ 0$ for all $k \in [ K ]$ . One might consider aggregating $\Delta W _ { k } = B _ { k } A _ { k }$ directly to eliminate the discordance, but this approach involves decomposing ∆W = PK $\begin{array} { r } { \Delta W = \sum _ { k = 1 } ^ { K } w _ { k } \Delta W _ { k } } \end{array}$ back into B and A for the next round, which is computationally unstable.

Limited parameter space in low rank and high data heterogeneity This discrepancy can be effectively addressed by either freezing the LoRA module A, as suggested by Sun et al. (2024), or employing SVD decomposition, as outlined by Bai et al. (2024). However, Figure 2 illustrates that the accuracy of these approaches decreases significantly at lower ranks under high heterogeneity. We attribute this decline primarily to the restricted parameter space imposed by LoRA. A limited training parameter space constrains the optimization capabilities for complex FL tasks, and a restricted aggregation parameter space exacerbates conflicts among clients. A detailed analysis of this limited parameter space is provided in Appendix C.

# 4 Proposed Method

To tackle the identified challenges, we propose a novel framework called Low Rank Adaptation with Alternating freeze and Adaptive rank selection for federated learning, or $\mathrm { L o R A { - } A ^ { 2 } }$ , for communication efficient FL with LoRA. LoRA- $\cdot \mathbf { A } ^ { 2 }$ adaptively selects LoRA ranks for local training and transmits only the selected part of each adapter in an alternating way.

# 4.1 Alternating Freeze

$\mathrm { L o R A { – } A { ^ 2 } }$ efficiently addresses the issue of discordance by employing a simple alternating freeze technique to train the LoRA modules B and A. Instead of solely training module B while keeping module A frozen permanently, as suggested by FFA-LoRA (Sun et al., 2024), LoR $\mathsf { A } { - } \mathsf { A } ^ { 2 }$ alternates between the two: LoRA module A is frozen during even rounds, while module B is frozen during odd rounds. This method preserves the optimization space while effectively resolving discordance. Specifically, when freezing A, we have

$$
\begin{array}{l} \Delta W = \sum_ {k = 1} ^ {K} \left(w _ {k} B _ {k}\right) A \\ = \sum_ {k = 1} ^ {K} \left(w _ {k} B _ {k} A _ {k}\right) = \sum_ {k = 1} ^ {K} \left(w _ {k} \Delta W _ {k}\right), \tag {3} \\ \end{array}
$$

and the same applies when freezing B. In this way, $\mathrm { L o R A { - } A ^ { 2 } }$ trains both B and A, ensuring that A does not remain the same as its initial value.

To further enhance the effect of alternating optimization, we adopt different learning rates for B and A, inspired by LoRA+ (Hayou et al., 2024). Figure 6 demonstrates the effectiveness of alternating freeze and learning rate adjustment.

# 4.2 Adaptive Rank Selection

Furthermore, we propose an adaptive rank selection method designed to reduce the number of transmitted parameters while preserving the training and aggregation parameter space. This approach selects important LoRA ranks to match local communication rank budget $r _ { i }$ out of global LoRA adapter with rank $r _ { G }$ adaptively based on the local dataset of each clients. We mainly focus on communication cost for uploading parameters to the server as it is known that upload bandwidth is generally much slower than download bandwidth and is the major part of communication cost (Konecnˇ y et al. \` , 2016; Suresh et al., 2017; Kairouz et al., 2021).

The adaptive rank selection process provides two key benefits: (1) it minimizes client conflicts by allowing each client to select different LoRA ranks in high heterogeneity, and (2) it reallocates rank resources from less important LoRA modules to modules that require more fine-tuning, which is especially effective when the communication rank budget is small.

To quantify which ranks are more important, we introduce our original criterion $S _ { m , i }$ for each rank i within module m as follows:

$$
\begin{array}{l} S _ {m, i} ^ {B _ {k}} = \left\| \Delta B _ {k [:, i ]} A _ {[ i,: ]} \right\| _ {F}. \tag {4} \\ S _ {m, i} ^ {A _ {k}} = \| B _ {[:, i ]} \Delta A _ {k [ i,: ]} \| _ {F} \\ \end{array}
$$

We define the change in $\Delta W$ for each rank i and module m as contribution $( C _ { m , i } )$ , represented as

$\begin{array} { r } { \sum C _ { m , i } = \Delta W _ { k } ^ { t + 1 } - \Delta W _ { k } ^ { t } = \sum ( \Delta B _ { k [ : , i ] } A _ { [ i , : ] } ) } \end{array}$ And define our criterion $S _ { m , i }$ as the Frobenius norm of contribution $( C _ { m , i } )$ . This criterion captures the impact of each rank on model updates based on local gradients. This approach is better suited for LoRA modules than simpler gradient magnitude-based criteria, $| | \Delta B _ { k [ : , i ] } | |$ or $| | \Delta A _ { k [ i , : ] } | |$ , as our criterion explicitly accounts for the interplay between module A and B. The ablation study in Table 9 empirically supports the superiority of this criterion. At each round, participating clients run local training for 1 epoch to obtain $\Delta W$ for calculating the contribution.

After computing $S _ { m , i } ^ { B _ { k } }$ or $S _ { m , i } ^ { A _ { k } }$ for each module $m ,$ , we select top- $. ( r _ { i } \cdot N )$ LoRA ranks from a total of $r _ { G } \cdot N$ based on the scores across the entire model, where N denotes the number of target modules across all the layers of the base model. We refer to the set of selected ranks of client k as $\mathcal { R } _ { k }$ .

Once the ranks are selected, each client defines LoRA module mask $\boldsymbol { M } _ { k } ^ { ( m ) }$ for the module m to be

$$
M _ {k [:, i ]} ^ {(m)} = \left\{ \begin{array}{l l} \mathbf {1} _ {d _ {1}} ^ {T} & \text { if } i \in \mathcal {R} _ {k} \\ \mathbf {0} _ {d _ {1}} ^ {T} & \text { otherwise } \end{array} , \right. \tag {5}
$$

$$
M _ {k [ i,: ]} ^ {(m)} = \left\{ \begin{array}{l l} \mathbf {1} _ {d _ {2}} & \text { if } i \in \mathcal {R} _ {k} \\ \mathbf {0} _ {d _ {2}} & \text { otherwise } \end{array} \right.,
$$

which is producted element-wise to the updated part of $B _ { k }$ (or $A _ { k } )$ . That is, before each backpropagation step, LoRA- $\cdot \mathbf { A } ^ { 2 }$ calculates

$$
\Delta B _ {k} ^ {(m)} \leftarrow \Delta B _ {k} ^ {(m)} \odot M _ {k} ^ {(m)} \tag {6}
$$

$$
\Delta A _ {k} ^ {(m)} \leftarrow \Delta A _ {k} ^ {(m)} \odot M _ {k} ^ {(m)}
$$

for each $B _ { k } \left( \operatorname { o r } A _ { k } \right)$ , where the notation stands for the Hadamard product. After each local training, each client uploads $B _ { k } \odot M _ { k }$ (or $A _ { k } \odot M _ { k } )$ , resulting in sparsification and reducing the number of uploaded parameters. Then, the server aggregates the uploaded ones, which are again added to the $B _ { k } \left( \operatorname { o r } A _ { k } \right)$ saved two rounds before. Algorithm 1 and 2 provides the detailed pseudocode of our $\mathrm { L o R A – A ^ { 2 } }$ algorithm.

# 4.3 Theoretical Insights

In this section, we provide a brief theoretical analysis of the parameter spaces associated with previous methods and our proposed $\mathrm { L o R A { - } A ^ { 2 } }$ framework. To substantiate our approach, we introduce the following proposition:

Algorithm 1 $\mathrm { L o R A – A ^ { 2 } }$   
Initialize $\Delta W = BA$ with $B \in R^{d_{1} \times r_{G}}$ and $A \in R^{r_{G} \times d_{2}}$ for each LoRA adapter

for $t = 1, 2, \cdots, T$ do

Sample participants $\mathcal{K}(t) \subseteq [K]$ for round t $w_{k} = |\mathcal{D}_{k}| / (\sum_{k=1}^{K} |\mathcal{D}_{k}|)$ if $t \% 2 = 1$ then

for $k = 1, 2, \cdots, K$ in parallel do $\Delta B_{k}^{(t+1)} = \text{LocalTraining}(B^{(t)}, t)$ $B^{(t+1)} = B^{(t)} + \sum_{k=1}^{K} w_{k} \Delta B_{k}^{(t+1)}$ $A^{(t+1)} = A^{(t)}$ end for

else

for $k = 1, 2, \cdots, K$ in parallel do $\Delta A_{k}^{(t+1)} = \text{LocalTraining}(A^{(t)}, t)$ $A^{(t+1)} = A^{(t)} + \sum_{k=1}^{K} w_{k} \Delta A_{k}^{(t+1)}$ $B^{(t+1)} = B^{(t)}$ end for

end if

end for

Proposition 1. For a model W , consider LoRAbased FL algorithms which update r rank parameters per round. Let $\Omega _ { \mathcal { A } }$ denote the space of all possible parameter values that an algorithm in $\mathcal { A } \in$ FFA-LoRA, FL+LoRA, FlexLoRA, $\mathrm { L o R A } { \cdot } \mathrm { A } ^ { 2 } \}$ can make. Then, we have the following:

$$
\Omega_ {\mathrm{FFA-LoRA}} \subsetneq \Omega_ {\mathrm{FL+LoRA}} = \Omega_ {\text { FlexLoRA }} \subset \Omega_ {\mathrm{LoRA-A} ^ {2}}.
$$

The proof of the proposition is provided in $\mathsf { A p - }$ pendix D.

Our algorithm is designed to adaptively select the relevant training and aggregation parameter spaces while concurrently reducing the number of parameters that are updated.

# 5 Experiments

In this section, we evaluate the performance of our algorithm against existing FL methods combined with LoRA across various heterogeneity settings and datasets. We assess performance based on accuracy and the total number of uploaded parameters.

# 5.1 Experimental Settings

We mainly adopt pre-trained RoBERTa-base (Liu et al., 2019) as the base model for fine-tuning. The base model has approximately 125M parameters, all of which are frozen during the fine-tuning phase. And a frozen classifier is added upon the model, following Sun et al. (2024). For Table 2 and 3, we adopt RoBERTa-large and DistilBERT(Sanh et al., 2019), respectively. RoBERTa-large has approximately 355M parameters, and DistilBERT has approximately 82M parameters. For fine-tuning, we choose BANKING77 (Casanueva et al., 2020) and 20 Newsgroups (Lang, 1995) datasets. These datasets are chosen for their ability to simulate a controlled level of data heterogeneity using Dirichlet distribution (Hsu et al., 2019). Dataset statistics are reported in Appendix A.

Algorithm 2 LocalTraining   
[Rank Selection]
Calculate importance scores following (4)
Define the mask $M_{k}$ following (5)
[Local Training]
if $t\%2=1$ then $B_{k}^{(t; e-1)} = B^{(t)}$ for $e = 1, 2, \cdots, E$ do $\Delta B_{k}^{(t; e)} = B_{k}^{(t; e)} - B_{k}^{(t; e-1)}$ $\Delta B_{k}^{(t; e)} = \Delta B_{k}^{(t; e)} \odot M_{k}$ Backpropagate $\Delta B_{k}^{(t; e)}$ end for
Return: $B_{k}^{(t; E)} - B^{(t)}$ else $A_{k}^{(t; e-1)} = A^{(t)}$ for $e = 1, 2, \cdots, E$ do $\Delta A_{k}^{(t; e)} = A_{k}^{(t; e)} - A_{k}^{(t; e-1)}$ $\Delta A_{k}^{(t; e)} = \Delta A_{k}^{(t; e)} \odot M_{k}$ Backpropagate $\Delta A_{k}^{(t; e)}$ end for
Return: $A_{k}^{(t; E)} - A^{(t)}$ end if

Unless otherwise stated, we trained 30 local clients under full participation, i.e., $\mathcal { K } ^ { \left( t \right) } = \left[ K \right]$ for all $t \in [ T ]$ . The clients were trained for 50 rounds with 5 local epochs. Detailed hyperparameters for experiments are specified in Appendix B.

For baselines, we adopt four methods that utilize LoRA for federated fine-tuning: FL + LoRA, FFA-LoRA (Sun et al., 2024), FlexLoRA (Bai et al., 2024), and HetLoRA (Cho et al., 2023), where FL + LoRA stands for the naive implementation of LoRA in FedAvg (McMahan et al., 2017).

# 5.2 Main Results

We compare our algorithm with the baseline methods under various data heterogeneity settings in BANKING77 and 20 Newsgroups datasets to demonstrate that our algorithm outperforms previous federated LoRA fine-tuning methods across different non-IID settings and LoRA ranks.

<table><tr><td rowspan="2">Method</td><td colspan="3">BANKING77 Dataset</td><td colspan="3">20 Newsgroups Dataset</td><td rowspan="2">Communicated Parameters*</td></tr><tr><td>Dir(0.5)</td><td>Dir(0.1)</td><td>Dir(0.01)</td><td>Dir(0.5)</td><td>Dir(0.1)</td><td>Dir(0.01)</td></tr><tr><td>FL (w/o LoRA)</td><td> $92.76_{\pm 0.30}$ </td><td> $90.29_{\pm 0.73}$ </td><td> $67.58_{\pm 0.44}$ </td><td> $70.93_{\pm 1.04}$ </td><td> $68.82_{\pm 0.69}$ </td><td> $64.41_{\pm 0.30}$ </td><td>186B</td></tr><tr><td> $FL + LoRA_{(Rank=8)}$ </td><td> $92.80_{\pm 0.24}$ </td><td> $90.47_{\pm 0.53}$ </td><td> $60.96_{\pm 1.47}$ </td><td> $\underline{70.44}_{\pm 0.28}$ </td><td> $\underline{67.33}_{\pm 0.18}$ </td><td> $43.90_{\pm 1.08}$ </td><td>1.99B</td></tr><tr><td> $FFA-LoRA_{(Rank=8)}$ </td><td> $87.20_{\pm 0.57}$ </td><td> $77.44_{\pm 1.28}$ </td><td> $40.88_{\pm 1.04}$ </td><td> $67.00_{\pm 0.67}$ </td><td> $61.27_{\pm 0.71}$ </td><td> $37.34_{\pm 0.30}$ </td><td>0.991B</td></tr><tr><td> $FlexLoRA_{(Rank=8)}$ </td><td> $\underline{93.35}_{\pm 0.24}$ </td><td> $\underline{92.14}_{\pm 0.25}$ </td><td> $\underline{69.84}_{\pm 0.65}$ </td><td> $\underline{70.59}_{\pm 0.22}$ </td><td> $\underline{68.10}_{\pm 0.38}$ </td><td> $\underline{60.41}_{\pm 1.54}$ </td><td>1.99B</td></tr><tr><td>Ours(Rank=8)</td><td> $\underline{93.24}_{\pm 0.27}$ </td><td> $\underline{91.61}_{\pm 0.39}$ </td><td> $\underline{70.13}_{\pm 1.22}$ </td><td> $70.26_{\pm 0.21}$ </td><td> $67.12_{\pm 0.22}$ </td><td> $\underline{54.50}_{\pm 1.44}$ </td><td>1.31B</td></tr><tr><td> $FL + LoRA_{(Rank=4)}$ </td><td> $\underline{92.86}_{\pm 0.08}$ </td><td> $88.11_{\pm 0.88}$ </td><td> $54.99_{\pm 0.59}$ </td><td> $70.33_{\pm 0.12}$ </td><td> $\underline{67.29}_{\pm 0.19}$ </td><td> $43.12_{\pm 2.67}$ </td><td>0.991B</td></tr><tr><td> $FFA-LoRA_{(Rank=4)}$ </td><td> $86.90_{\pm 1.14}$ </td><td> $76.38_{\pm 0.61}$ </td><td> $37.63_{\pm 0.80}$ </td><td> $67.75_{\pm 0.45}$ </td><td> $61.25_{\pm 0.26}$ </td><td> $36.04_{\pm 0.80}$ </td><td>0.497B</td></tr><tr><td> $FlexLoRA_{(Rank=4)}$ </td><td> $92.71_{\pm 0.31}$ </td><td> $\underline{90.53}_{\pm 0.70}$ </td><td> $\underline{57.38}_{\pm 1.30}$ </td><td> $70.05_{\pm 0.14}$ </td><td> $\underline{68.00}_{\pm 0.33}$ </td><td> $\underline{50.50}_{\pm 2.09}$ </td><td>0.991B</td></tr><tr><td>Ours(Rank=4)</td><td> $\underline{93.22}_{\pm 0.24}$ </td><td> $\underline{91.43}_{\pm 0.63}$ </td><td> $\underline{69.63}_{\pm 1.52}$ </td><td> $\underline{70.28}_{\pm 0.32}$ </td><td> $67.12_{\pm 0.60}$ </td><td> $\underline{53.04}_{\pm 1.68}$ </td><td>0.888B</td></tr><tr><td> $FL + LoRA_{(Rank=2)}$ </td><td> $91.97_{\pm 0.43}$ </td><td> $85.59_{\pm 1.13}$ </td><td> $49.08_{\pm 0.56}$ </td><td> $\underline{70.14}_{\pm 0.13}$ </td><td> $65.40_{\pm 0.31}$ </td><td> $39.07_{\pm 2.23}$ </td><td>0.497B</td></tr><tr><td> $FFA-LoRA_{(Rank=2)}$ </td><td> $84.65_{\pm 1.05}$ </td><td> $73.44_{\pm 0.88}$ </td><td> $34.44_{\pm 2.15}$ </td><td> $68.12_{\pm 0.47}$ </td><td> $61.57_{\pm 0.38}$ </td><td> $36.65_{\pm 0.52}$ </td><td>0.249B</td></tr><tr><td> $FlexLoRA_{(Rank=2)}$ </td><td> $\underline{92.22}_{\pm 0.50}$ </td><td> $87.31_{\pm 0.27}$ </td><td> $\underline{55.24}_{\pm 2.19}$ </td><td> $70.03_{\pm 0.31}$ </td><td> $66.17_{\pm 1.70}$ </td><td> $48.23_{\pm 1.73}$ </td><td>0.497B</td></tr><tr><td>Ours(Rank=2)</td><td> $\underline{93.10}_{\pm 0.07}$ </td><td> $\underline{92.02}_{\pm 0.36}$ </td><td> $\underline{69.40}_{\pm 0.48}$ </td><td> $\underline{70.12}_{\pm 0.18}$ </td><td> $\underline{67.02}_{\pm 0.26}$ </td><td> $\underline{52.99}_{\pm 2.56}$ </td><td>0.528B</td></tr><tr><td> $FL + LoRA_{(Rank=1)}$ </td><td> $\underline{90.61}_{\pm 0.10}$ </td><td> $\underline{82.24}_{\pm 1.68}$ </td><td> $\underline{45.78}_{\pm 1.04}$ </td><td> $69.40_{\pm 0.33}$ </td><td> $\underline{63.16}_{\pm 0.53}$ </td><td> $\underline{36.58}_{\pm 0.98}$ </td><td>0.249B</td></tr><tr><td> $FFA-LoRA_{(Rank=1)}$ </td><td> $82.51_{\pm 0.53}$ </td><td> $72.96_{\pm 0.54}$ </td><td> $33.68_{\pm 0.20}$ </td><td> $67.73_{\pm 0.30}$ </td><td> $61.35_{\pm 0.22}$ </td><td> $34.44_{\pm 0.68}$ </td><td>0.124B</td></tr><tr><td> $FlexLoRA_{(Rank=1)}$ </td><td> $90.40_{\pm 0.54}$ </td><td> $82.20_{\pm 0.74}$ </td><td> $42.75_{\pm 0.89}$ </td><td> $\underline{69.53}_{\pm 0.25}$ </td><td> $62.98_{\pm 1.12}$ </td><td> $35.54_{\pm 0.68}$ </td><td>0.249B</td></tr><tr><td>Ours(Rank=1)</td><td> $\underline{93.21}_{\pm 0.13}$ </td><td> $\underline{91.87}_{\pm 0.33}$ </td><td> $\underline{68.88}_{\pm 1.15}$ </td><td> $\underline{70.31}_{\pm 0.24}$ </td><td> $\underline{66.95}_{\pm 0.07}$ </td><td> $\underline{54.84}_{\pm 1.15}$ </td><td>0.270B</td></tr></table>

Table 1: Results with RoBERTa-base on BANKING77 and 20 Newsgroups datasets. Smaller α for $D i r ( \alpha )$ implies that the simulated setting is more heterogeneous. The best results on each dataset are shown in bold and second best is shown by underline. ∗ This column reports the total number of uploaded parameters, averaged across rows.

Robustness of $\mathbf { L o R A { - } A ^ { 2 } }$ in low ranks and high heterogeneity Table 1 highlights the vulnerability of previous methods under conditions of high heterogeneity and low ranks. The accuracy of baseline methods declines significantly as rank decreases, whereas our algorithm maintains its performance, achieving up to a 23% accuracy advantage. This suggests that reducing LoRA ranks is challenging for previous methods under realistic heterogeneous data conditions. Also, our algorithm consistently achieves the highest performance or remains within a 1% margin of the best-performing baselines at rank 8 and 4, while showing a large performance gap in low ranks.

Communication cost reduction by $\mathbf { L o R A { - } A ^ { 2 } }$ Decreasing LoRA ranks in federated LoRA methods reduces the communication cost linearly. Our algorithm achieves performance comparable to or better than fully fine-tuned models even at rank 1, allowing for up to a 99.8% reduction in communicated parameters with minimal performance degradation. This demonstrates that LoRA-A2 effectively solves the significant communication cost challenges of federated fine-tuning on LLMs.

# 5.3 Analysis on Adaptive Rank Selection

In this section, we visualize the process of our adaptive rank selection, and explore how efficiently $\mathrm { L o R A { - } A ^ { 2 } }$ trains and sends important ranks, highlighting the robustness of our algorithm in heterogeneous and low rank environments. To simulate extreme cases of both identical and different client distributions, we test our algorithm on a pathological dataset using the 20 Newsgroups dataset. In this setup, 20 clients each holds data from only two classes, with consecutive pairs sharing the same classes, while others do not. For instance, clients 0 and 1 have classes "medical" and "space," whereas clients 2 and 3 have "motorcycle" and "religions". Detailed settings are shown in Appendix C.

Robustness to low-rank by adaptive module selection In this experiment, our algorithm selects $2 \cdot N ^ { ( m ) }$ ranks from a total of $1 6 \cdot N ^ { ( m ) }$ across the whole RoBERTa model, guided by our importance criterion, and visualizes the adaptive selection of modules. Figure 3 illustrates the number of ranks selected for each module in the model during the training. The figure shows that most modules are allocated zero ranks, indicating either no need for fine-tuning or the insignificance of updates on those modules. This suggests that our adaptive rank selection automatically prunes out modules that do not require additional fine-tuning.

![](images/65d0eee38a9587e4a9f42567f45f4e3d4ee5eb0b0911063bfdc8013984919807.jpg)

<details>
<summary>heatmap</summary>

| | query | key | value | attention | intermediate | output |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| 1 | 0 | 0 | 0 | 0 | 0 | 4 |
| 2 | 0 | 0 | 0 | 6 | 3 | |
| 3 | 0 | 0 | 0 | 1 | 0 | |
| 4 | 0 | 0 | 0 | 6 | 3 | |
| 5 | 0 | 0 | 0 | 5 | 3 | |
| 6 | 0 | 0 | 0 | 10 | 2 | |
| 7 | 0 | 0 | 0 | 11 | 2 | |
| 8 | 0 | 0 | 0 | 7 | 0 | |
| 9 | 0 | 0 | 0 | 15 | 1 | |
| 10 | 1 | 0 | 0 | 14 | 10 | |
| 11 | 0 | 0 | 0 | 4 | 16 | 16 |
</details>

(a) client 0

![](images/ba4cb2e8a0e23df5c04c13c47f739faaa68a8a057d7ea8da2f34bac72b1f84b4.jpg)

<details>
<summary>heatmap</summary>

| | query | key | value |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 |
| 9 | 0 | 0 | 0 |
| 10 | 0 | 0 | 0 |
| 11 | 0 | 0 | 4 |
| 12 | 1 | 1 | 16 |
| 13 | 1 | 15 | 13 |
| 14 | 1 | 2 | 9 |
| 15 | 1 | 5 | 3 |
| 16 | 1 | 5 | 2 |
| 17 | 1 | 5 | 1 |
| 18 | 1 | 5 | 5 |
| 19 | 1 | 5 | 2 |
| 20 | 1 | 5 | 1 |
| 21 | 1 | 5 | 5 |
| 22 | 1 | 5 | 2 |
| 23 | 1 | 5 | 1 |
| 24 | 1 | 5 | 0 |
| 25 | 1 | 5 | 0 |
| 26 | 1 | 5 | 0 |
| 27 | 1 | 5 | 0 |
| 28 | 1 | 5 | 0 |
| 29 | 1 | 5 | 0 |
| 30 | 1 | 5 | 0 |
| 31 | 1 | 5 | 0 |
| 32 | 1 | 5 | 0 |
| 33 | 1 | 5 | 0 |
| 34 | 1 | 5 | 0 |
| 35 | 1 | 5 | 0 |
| 36 | 1 | 5 | 0 |
| 37 | 1 | 5 | 0 |
| 38 | 1 | 5 | 0 |
| 39 | 1 | 5 | 0 |
| 40 | 1 | 5 | 0 |
| Note: The values in the 'attention' and 'output' columns are estimated based on the provided code snippet. The 'intermediate' and 'output' values for the 'attention' column are not explicitly labeled in the image. The 'key' and 'value' columns are estimated based on the given text. There is no additional data series in this image. The 'output' column contains only a few entries in the chart.
</details>

(b) client 1

![](images/7d55e2b5cad43bfac9d556106dea49e82f9c429af342cd43da0c12ce3ae95fb0.jpg)

<details>
<summary>heatmap</summary>

| | query | key | value | attention | intermediate | output |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 1 | 2 | 0 |
| 2 | 0 | 0 | 0 | 0 | 1 | 0 |
| 3 | 0 | 0 | 0 | 1 | 2 | 0 |
| 4 | 0 | 0 | 0 | 2 | 1 | 0 |
| 5 | 0 | 0 | 0 | 2 | 1 | 0 |
| 6 | 1 | 0 | 0 | 6 | 3 | 0 |
| 7 | 0 | 0 | 0 | 5 | 6 | 0 |
| 8 | 0 | 0 | 0 | 13 | 4 | 0 |
| 9 | 0 | 0 | 0 | 13 | 8 | 0 |
| 10 | 9 | 0 | 0 | 12 | 8 | 0 |
| 11 | 7 | 1 | 0 | 16 | 15 | 0 |
The values in the matrix are estimated based on the data provided in the code. The color intensity corresponds to the numeric value in each cell, with darker shades indicating higher values. There is no label for the data series.
</details>

(c) client 2

Figure 3: Visualization on number of selected rank per module. The x-axis shows RoBERTa module types, while the y-axis indicates layer numbers. Experimented on the 20 Newsgroups dataset with a pathological data distribution. Average 2 ranks were selected out of 16 ranks by our adaptive rank selection algorithm.   
![](images/d3af31cb56a3c7e9bad6b14acb66240008ad47db1ac599767ff4487492574630.jpg)

<details>
<summary>bar</summary>

| Layer       | Accuracy |
| ----------- | -------- |
| Layer 0-2   | 66.71    |
| Layer 3-5   | 69.08    |
| Layer 6-8   | 69.55    |
| Layer 9-11  | 70.24    |
</details>

(a) Selected layers

![](images/bdb96d642e9a382a54fb33194de98c2aea310ead06e7c71178ef105ca364c9f3.jpg)

<details>
<summary>bar</summary>

| Category     | Accuracy |
| ------------ | -------- |
| query        | 55.48    |
| key          | 52.73    |
| value        | 66.66    |
| attention    | 67.86    |
| intermediate | 69.37    |
| output       | 69.79    |
</details>

(b) Selected modules   
Figure 4: Ablation analysis on the performance of model when solely fine-tuned on selected layers or types of modules. Experimented on 20 Newsgroups dataset with Dir(0.1) heterogeneity.

To further justify that our adaptive rank selection successfully selects important modules, we conduct an ablation experiment on module selection, following the approach of AdaLoRA (Zhang et al., 2023) but in a federated setting. Figure 4 displays the model’s performance when only specific modules or layers are fine-tuned and other layers are frozen. The ablation experiment demonstrated that last layer in layer experiment and intermediate or output dense modules in module experiment led to the best performance, highlighting their importance for fine-tuning. This aligns with our findings, where the last layers and intermediate / output dense modules are automatically selected through adaptive rank selection, demonstrating the effectiveness of our algorithm in prioritizing essential modules for fine-tuning.

Robustness to data heterogeneity by client clustering Another effect of rank selection is the implicit clustering of clients to minimize conflicts among clients with dissimilar datasets and to enhance cooperation among those with similar ones.

Figure 5 (a) illustrates how much local rank parameters are shared among different clients. The figure shows that clients with similar data distributions tend to share more rank parameters, while those with dissimilar data share fewer. This trend is also evident at the module level in Figure 3, where clients 0 and 1 select a similar number of ranks for each module, differing from client 2, while retaining the tendency to choose more ranks from the last layers or intermediate and output dense modules. These findings suggest that clients with similar datasets converge on the same ranks, facilitating cooperative training, whereas clients with dissimilar datasets select more distinct ranks, resulting in independent parameter updates.

Figure 5 (b) further supports this by visualizing the cosine similarity between clients’ model updates, which approaches 1 for clients with the same classes and remains close to 0 for those without data overlap. This underscores the cooperative nature of updates from similar clients while maintaining independence from dissimilar ones, thereby contributing to the robustness of our algorithm against data heterogeneity.

![](images/60cc7327b52f9ceced0189288a31cdbba5f9a6089f33a00287986d29602551bb.jpg)

<details>
<summary>heatmap</summary>

|        | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 1      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 2      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 3      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 4      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 5      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 6      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 7      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 8      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 9      | 1.15 | -    | -    | -    | -    | -    | -    | -    | -    | -    |
The chart displays a heatmap with color intensity corresponding to the values on the vertical axis (ranging from -1 to +1). The x-axis and y-axis are labeled with integers from -1 to +1 and -1 to +1, respectively, but no explicit numerical data is provided in the image.
</details>

(a) Rank selection similarity

![](images/c7697479767ad0e9e994d348c1d09898e19782851b92c744d9d61a34f851b375.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| 1 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| 2 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| 3 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| 4 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| 5 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| 6 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| 7 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| 8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| 9 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
The heatmap visualizes the intensity of values across a grid layout, with darker shades representing higher values (scale: -1 to +1). The color scale ranges from -1 (lightest) to +1 (darkest), and the grid is symmetrically placed around the origin (x=1, y=1). No explicit title or legend is present; the chart only displays a single continuous grid pattern.
</details>

(b) Cosine similarity of local updates   
Figure 5: Visualization of similarity between clients. the x and y axes represent individual clients trained on 20 Newsgroups dataset with pathologic data distribution.

# 5.4 Ablation Studies

The following ablation studies provide empirical evidence supporting our design choices for aggregation tactics and rank selection criteria.

Efficacy of alternating freeze To address the discordance problem in federated LoRA aggregation, we employ a strategy that alternately freezes LoRA modules B and A, rather than freezing module A only as in FFA-LoRA (Cho et al., 2023). Furthermore, we set the learning rate of module B, ηB, to be five times that of module A, ηA, inspired by LoRA+ (Hayou et al., 2024). This configuration further enhances overall performance and robustness, particularly in highly heterogeneous environments. Figure 6 compares these approaches, showing that solely freezing A is less effective under high data heterogeneity, whereas achieves consistently better performance.

![](images/6425ee6b67c68e746101bf100c9d6e9afd1f6e9abc7585c8dc8c5d5922c35095.jpg)

<details>
<summary>bar</summary>

| Data Heterogeneity | Blue Bar | Orange Bar | Green Bar |
| ----------------- | -------- | ---------- | --------- |
| Dir(0.5)          | 87.19    | 91.86      | 93.25     |
| Dir(0.1)          | 79.77    | 86.41      | 91.71     |
| Dir(0.01)         | 42.58    | 48.85      | 68.82     |
</details>

(a) BANKING77

![](images/044908456771a5452ba80c8e2f96202b5abe905f824f4ba1563637d8c18e1ccc.jpg)

<details>
<summary>bar</summary>

| Data Heterogeneity | FFA-LoRA | FFA-LoRA + Alternating | FFA-LoRA + ηB > ηA |
| ------------------ | -------- | ---------------------- | ------------------ |
| Dir(0.5)           | 68.44    | 69.20                  | 70.25              |
| Dir(0.1)           | 64.44    | 65.11                  | 67.23              |
| Dir(0.01)          | 38.93    | 45.53                  | 53.82              |
</details>

(b) 20 Newsgroups

Figure 6: Ablation analysis for the effect of alternating freeze and learning rate adjustment under varying levels of heterogeneity. 

<table><tr><td rowspan="2"># of Ranks</td><td colspan="4">RoBERTa-Large</td></tr><tr><td>FL+LoRA</td><td>FFA-LoRA</td><td>FlexLoRA*</td><td>Ours</td></tr><tr><td>8</td><td> $\underline{80.15}_{\pm 0.58}$ </td><td> $62.98_{\pm 0.61}$ </td><td>-</td><td> $\mathbf{85.98}_{\pm 0.82}$ </td></tr><tr><td>4</td><td> $\underline{78.97}_{\pm 0.52}$ </td><td> $62.45_{\pm 0.33}$ </td><td>-</td><td> $\mathbf{84.62}_{\pm 0.37}$ </td></tr><tr><td>2</td><td> $\underline{75.09}_{\pm 1.20}$ </td><td> $61.55_{\pm 1.05}$ </td><td>-</td><td> $\mathbf{83.40}_{\pm 0.55}$ </td></tr><tr><td>1</td><td> $\underline{73.75}_{\pm 1.53}$ </td><td> $58.06_{\pm 1.90}$ </td><td>-</td><td> $\mathbf{85.66}_{\pm 0.36}$ </td></tr></table>

Table 2: Experimental results on RoBERTa-Large model. The level of heterogeneity is Dir(0.01).   
∗ FlexLoRA results could not be reported due to an ill-conditioned matrix issue in SVD decomposition.

<table><tr><td rowspan="2"># of Ranks</td><td colspan="4">DistilBERT</td></tr><tr><td>FL+LoRA</td><td>FFA-LoRA</td><td>FlexLoRA</td><td>Ours</td></tr><tr><td>8</td><td> $32.58_{\pm 0.34}$ </td><td> $18.82_{\pm 0.57}$ </td><td> $\underline{51.21}_{\pm 0.51}$ </td><td> $\underline{52.97}_{\pm 0.32}$ </td></tr><tr><td>4</td><td> $36.92_{\pm 0.37}$ </td><td> $16.73_{\pm 0.52}$ </td><td> $\underline{41.26}_{\pm 0.47}$ </td><td> $\underline{51.24}_{\pm 0.44}$ </td></tr><tr><td>2</td><td> $27.14_{\pm 0.92}$ </td><td> $15.49_{\pm 1.24}$ </td><td> $\underline{34.05}_{\pm 0.82}$ </td><td> $\underline{49.97}_{\pm 0.33}$ </td></tr><tr><td>1</td><td> $\underline{21.59}_{\pm 1.12}$ </td><td> $14.29_{\pm 1.34}$ </td><td> $21.01_{\pm 1.23}$ </td><td> $\underline{48.89}_{\pm 0.41}$ </td></tr></table>

Table 3: Experimental results on DistilBERT model. The level of heterogeneity is Dir(0.01).

Scalability and generalizability on model structures To evaluate the scalability and generalizability of our algorithm across various model structures, we present the experimental results on RoBERTa-large (Liu et al., 2019) and DistilBERT (Sanh et al., 2019) models in Table 2 and Table 3, respectively. These tables illustrate the performance of our model when applied to diverse architectures and parameter configurations. The results show that our algorithm achieves superior performance, even on models with a larger number of parameters or different architectures. This highlights the robust scalability and generalizability of our approach across different model structures.

# 5.5 Additional Experiments

Differential privacy According to Sun et al. (2024), discordance problem of federated LoRA intesnsified when Differential Privacy (DP) is applied (Dwork et al., 2006; Abadi et al., 2016), due to the added noise amplifying errors. Specifically, if $\xi _ { B }$ and $\xi _ { A }$ stand for the noise added to B and A, respectively, we have $\Delta W = ( B + \xi _ { B } ) ( A + \xi _ { A } )$ = $B A + B \xi _ { A } + \xi _ { B } A + \xi _ { B } \xi _ { A }$ .

<table><tr><td> $\epsilon$ </td><td>FL+LoRA</td><td>FFA-LoRA</td><td>FlexLoRA</td><td>Ours</td></tr><tr><td> $\infty$ </td><td> $49.08_{\pm 0.56}$ </td><td> $34.44_{\pm 2.15}$ </td><td> $\underline{55.24}_{\pm 2.19}$ </td><td> $\underline{69.40}_{\pm 0.48}$ </td></tr><tr><td>6</td><td> $47.97_{\pm 0.72}$ </td><td> $35.35_{\pm 0.94}$ </td><td> $\underline{50.22}_{\pm 0.56}$ </td><td> $\underline{70.44}_{\pm 1.88}$ </td></tr><tr><td>3</td><td> $44.01_{\pm 0.38}$ </td><td> $31.90_{\pm 0.73}$ </td><td> $\underline{49.62}_{\pm 0.76}$ </td><td> $\underline{68.62}_{\pm 1.61}$ </td></tr><tr><td>1</td><td> $41.05_{\pm 1.11}$ </td><td> $33.78_{\pm 0.75}$ </td><td> $\underline{49.39}_{\pm 1.76}$ </td><td> $\underline{68.70}_{\pm 0.22}$ </td></tr></table>

Table 4: Experiments with differential privacy.

Table 4 represents experiments on BANKING77 dataset with DP. Following Ryu et al. (2022), Laplace mechanism is adopted. The level of heterogeneity is Dir(0.01) and the rank is set to 2 for each method. The clipping constant C is set to either 2 or 5, whichever yields better performance, for each method.

The tables demonstrates that FFA-LoRA (Sun et al., 2024), FlexLoRA (Bai et al., 2024) and Our algorithm effectively mitigate the discordance problem, While FL with LoRA suffers from performance degradation. Moreover our algorithm shows the highest robustness under conditions of severe noise, such as ϵ = 1 and ϵ = 3, outperforming other baseline methods.

Computational overhead Regarding computational overhead, our analysis shows that LoRA-A exhibits a 1.17x increase in computation time compared to standard FL+LoRA, slightly higher than FFA-LoRA (0.93x) and FlexLoRA (1x). This is due to gradient computation for local rank selection. However, we note that communication time, often the dominant bottleneck in federated learning, is significantly reduced by $\mathrm { L o R A { - } A ^ { 2 } }$ , outweighing the modest increase in computation time.

Other experiments We also include further experiments addressing resource heterogeneity settings, pathological distributions, as well as investigations into convergence speed in Appendix C.

# 6 Conclusion

In this work, we tackle the vulnerability of previous methods in high heterogeneity and low ranks by proposing a novel algorithm, $\mathrm { L o R A { - } A ^ { 2 } }$ , which shows robustness in these challenging conditions with alternating freeze and adaptive rank selection. Our approach offers significant improvements in communication efficiency without compromising performance, as demonstrated by a reduction of 99.8% in parameter uploads compared to full fine-tuning. Through extensive experiments, we establish LoR $\mathbf { A } { - } \mathbf { A } ^ { 2 }$ as a superior alternative, providing a practical pathway for efficient and effective federated fine-tuning in diverse and resourceconstrained environments.

# 7 Acknowledgments

This work was supported by Institute of Information $\&$ communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2019-II191906, Artificial Intelligence Graduate School Program (POSTECH); RS-2021-II210739, Development of Distributed/Cooperative AI based 5G+ Network Data Analytics Functions and Control Technology; RS-2024-00457882, AI Research Hub Project; RS-2024-00509258, Global AI Frontier Lab).

# 8 Limitations

LoRA-A2 shows promising results and we plan to distribute the implementation code with detailed instructions for reproducibility. However, several areas remain open for future exploration.

First, our work mainly focuses on classification tasks, primarily due to computational constraints and the use of Dirichlet distribution to simulate non-IID conditions. However, extending LoRA-A2 to more complex tasks, such as natural language generation, could offer additional perspectives. Future work with more resources could explore these broader applications.

Second, our experiments are primarily conducted on comparatively smaller language models, such as RoBERTa-base and RoBERTa-large, due to limited computation resources. Applying LoRA-$\mathrm { A ^ { 2 } }$ to larger models, such as LLaMA or GPT-style architectures, could provide an opportunity to test its scalability. Investigating how well the method handles the increased parameter space of these state-of-the-art models could further demonstrate its efficiency.

Finally, due to the limited access to real world datasets, our current results are mainly based on simulated settings. Extensive research on real world dataset, which typically exhibit more diverse types of noise and heterogeneity would help understand performance and robustness of $\mathrm { L o R A – A ^ { 2 } }$ in practical, dynamic environments.

# References

Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. 2016. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, CCS ’16, page 308–318, New York, NY, USA. Association for Computing Machinery.   
Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. 2021. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7319–7328, Online. Association for Computational Linguistics.   
Sara Babakniya, Ahmed Elkordy, Yahya Ezzeldin, Qingfeng Liu, Kee-Bong Song, MOSTAFA EL-Khamy, and Salman Avestimehr. 2023. SLoRA: Federated parameter efficient fine-tuning of language models. In International Workshop on Federated Learning in the Age of Foundation Models in Conjunction with NeurIPS 2023.   
Jiamu Bai, Daoyuan Chen, Bingchen Qian, Liuyi Yao, and Yaliang Li. 2024. Federated fine-tuning of large language models under heterogeneous tasks and client resources. arXiv preprint arXiv:2402.11505.   
Daniel J Beutel, Taner Topal, Akhil Mathur, Xinchi Qiu, Javier Fernandez-Marques, Yan Gao, Lorenzo Sani, Hei Li Kwing, Titouan Parcollet, Pedro PB de Gusmão, and Nicholas D Lane. 2020. Flower: A friendly federated learning research framework. arXiv preprint arXiv:2007.14390.   
Iñigo Casanueva, Tadas Temcinas, Daniela Gerz, ˇ Matthew Henderson, and Ivan Vulic. 2020. Efficient´ intent detection with dual sentence encoders. In Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, pages 38–45, Online. Association for Computational Linguistics.   
Daoyuan Chen, Liuyi Yao, Dawei Gao, Bolin Ding, and Yaliang Li. 2023. Efficient personalized federated learning via sparse model-adaptation. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 5234–5256. PMLR.   
Yae Jee Cho, Luyang Liu, Zheng Xu, Aldi Fahrezi, Matt Barnes, and Gauri Joshi. 2023. Heterogeneous loRA for federated fine-tuning of on-device foundation models. In International Workshop on Federated Learning in the Age of Foundation Models in Conjunction with NeurIPS 2023.   
Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. 2020. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. 2006. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography, pages 265–284, Berlin, Heidelberg. Springer Berlin Heidelberg.

Soufiane Hayou, Nikhil Ghosh, and Bin Yu. 2024. Lora+: Efficient low rank adaptation of large models. arXiv 2402.12354.

Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. 2019. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. 2021. Advances and open problems in federated learning. Foundations and trends® in machine learning, 14(1–2):1–210.

Jakub Konecnˇ y, H Brendan McMahan, Felix X Yu, Pe- \` ter Richtárik, Ananda Theertha Suresh, and Dave Bacon. 2016. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492.

Kevin Kuo, Arian Raje, Kousik Rajesh, and Virginia Smith. 2024. Federated lora with sparse communication. arXiv preprint arXiv:2406.05233.

Ken Lang. 1995. Newsweeder: Learning to filter netnews. In Proceedings of the Twelfth International Conference on Machine Learning, pages 331–339.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. 2018. Measuring the intrinsic dimension of objective landscapes. arXiv preprint arXiv:1804.08838.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. 2022. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35:1950–1965.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.   
Zequan Liu, Jiawen Lyn, Wei Zhu, Xing Tian, and Yvette Graham. 2024. ALoRA: Allocating low-rank adaptation for fine-tuning large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 622–641, Mexico City, Mexico. Association for Computational Linguistics.   
Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.   
Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. 2022. Peft: State-of-the-art parameter-efficient finetuning methods.   
Yulong Mao, Kaiyu Huang, Changhao Guan, Ganglin Bao, Fengran Mo, and Jinan Xu. 2024. DoRA: Enhancing parameter-efficient fine-tuning with dynamic rank distribution. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11662– 11675, Bangkok, Thailand. Association for Computational Linguistics.   
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pages 1273–1282. PMLR.   
OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.   
Minseok Ryu, Youngdae Kim, Kibaek Kim, and Ravi K Madduri. 2022. Appfl: open-source software framework for privacy-preserving federated learning. In 2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), pages 1074–1083. IEEE.   
Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.   
Youbang Sun, Zitao Li, Yaliang Li, and Bolin Ding. 2024. Improving loRA in privacy-preserving federated learning. In The Twelfth International Conference on Learning Representations.   
Ananda Theertha Suresh, X Yu Felix, Sanjiv Kumar, and H Brendan McMahan. 2017. Distributed mean estimation with limited communication. In International

conference on machine learning, pages 3329–3337. PMLR.

Ziyao Wang, Zheyu Shen, Yexiao He, Guoheng Sun, Hongyi Wang, Lingjuan Lyu, and Ang Li. 2024. Flora: Federated fine-tuning large language models with heterogeneous low-rank adaptations. arXiv preprint arXiv:2409.05976.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023. Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations.

<table><tr><td rowspan="2"></td><td colspan="2">Dir(0.01)</td><td colspan="2">Dir(0.1)</td><td colspan="2">Dir(0.5)</td></tr><tr><td>Train</td><td>Test</td><td>Train</td><td>Test</td><td>Train</td><td>Test</td></tr><tr><td> $\max |\{\mathcal{D}_k\}|_{k\in[K]}$ </td><td>1317</td><td>877</td><td>911</td><td>606</td><td>576</td><td>383</td></tr><tr><td> $\min |\{\mathcal{D}_k\}|_{k\in[K]}$ </td><td>1</td><td>1</td><td>58</td><td>37</td><td>151</td><td>100</td></tr><tr><td> $\max |\{\mathcal{C}_k\}|_{k\in[K]}$ </td><td>5</td><td>5</td><td>12</td><td>12</td><td>20</td><td>14</td></tr><tr><td> $\min |\{\mathcal{C}_k\}|_{k\in[K]}$ </td><td>1</td><td>1</td><td>5</td><td>5</td><td>20</td><td>12</td></tr><tr><td>Number of classes</td><td colspan="6">20</td></tr><tr><td>Number of clients</td><td colspan="6">30</td></tr></table>

Table 5: Statistics of 20 Newsgroups datasets.

<table><tr><td rowspan="2"></td><td colspan="2">Dir(0.01)</td><td colspan="2">Dir(0.1)</td><td colspan="2">Dir(0.5)</td></tr><tr><td>Train</td><td>Test</td><td>Train</td><td>Test</td><td>Train</td><td>Test</td></tr><tr><td> $\max |\{\mathcal{D}_k\}|_{k\in[K]}$ </td><td>639</td><td>212</td><td>672</td><td>185</td><td>473</td><td>133</td></tr><tr><td> $\min |\{\mathcal{D}_k\}|_{k\in[K]}$ </td><td>50</td><td>30</td><td>139</td><td>43</td><td>248</td><td>75</td></tr><tr><td> $\max |\{\mathcal{C}_k\}|_{k\in[K]}$ </td><td>15</td><td>10</td><td>34</td><td>24</td><td>65</td><td>52</td></tr><tr><td> $\min |\{\mathcal{C}_k\}|_{k\in[K]}$ </td><td>2</td><td>2</td><td>18</td><td>15</td><td>37</td><td>31</td></tr><tr><td>Number of intents</td><td colspan="6">77</td></tr><tr><td>Number of clients</td><td colspan="6">30</td></tr></table>

Table 6: Statistics of BANKING77 dataset.

# A Dataset Statistics

BANKING77 (Casanueva et al., 2020) is an intent classification dataset with 77 fine-grained intents related to the banking domain, comprising 10,003 training samples and 3,080 test samples. 20 Newsgroups (Lang, 1995) is a widely used text classification dataset with 20 classes, each representing a unique topic. It contains 11,314 training samples and 7,532 test samples.

We provide the statistics of two datasets in Table 5 and Table $^ { 6 , }$ respectively. $\mathcal { D } _ { k }$ and $| \mathcal { C } _ { k } |$ denotes the local dataset of k and the number of unique classes in $\mathcal { D } _ { k } .$ , respectively. Figure 7 shows the distribution of a local dataset for varying α simulating the Dirichlet distribution.

# B Reproducibility

Hyperparameters When training, we use AdamW (Loshchilov and Hutter, 2019) optimizer with a learning rate of $\eta = 0 . 0 0 0 5$ . For LoR $\mathsf { A } { - } \mathsf { A } ^ { 2 }$ , since B and A of each LoRA module are optimized separately, we use different learning rates for them. Specifically, $\eta _ { A } = \eta$ is used for A and $\eta _ { B } = 5 \cdot \eta _ { A }$ is used for B, which is inspired by LoRA+ (Hayou et al., 2024). For HetLoRA, $\gamma ~ = ~ 0 . 9 9$ is used for the decaying factor as suggested by Cho et al. (2023). When evaluating, we merge the LoRA adapter $\Delta W$ with the pre-trained model $W _ { 0 }$ using a scaling factor, so that $\begin{array} { r } { W _ { f t } = W _ { 0 } + \frac { 1 6 } { r } \Delta W } \end{array}$ .

Implementation details We simulate our FL setup using Flower (Beutel et al., 2020), and utilize HuggingFace PEFT (Mangrulkar et al., 2022) library to train base models with LoRA. The base models are loaded using HuggingFace Transformers (Wolf et al., 2020) library. All experiments are conducted three times to ensure reproducibility, and the code will be released soon to promote transparency and support future research.

![](images/b22f4afb805b9eae5e797965695e415601043573ed643ed7c16a3bf49229d16c.jpg)

<details>
<summary>bar</summary>

| Class | Count |
|---|---|
| 1 | 12 |
| 2 | 3 |
</details>

(a) α = 0.01

![](images/34a03ba24cd669a65d0d76cf0644dbcde7be4f1f44652486264f1206a3219d71.jpg)

<details>
<summary>bar</summary>

| Class | Count |
|---|---|
| 1 | 195 |
| 2 | 70 |
| 3 | 45 |
| 4 | 20 |
| 5 | 5 |
</details>

(b) α = 0.1

![](images/86ab3f5add5c595cb384aa608f57d0057d9381bf448414ea9f7aa7c9ceb796d5.jpg)

<details>
<summary>bar</summary>

| Class | Count |
|---|---|
| 1 | 142 |
| 2 | 58 |
| 3 | 56 |
| 4 | 50 |
| 5 | 42 |
| 6 | 36 |
| 7 | 34 |
| 8 | 32 |
| 9 | 18 |
| 10 | 10 |
| 11 | 6 |
| 12 | 5 |
| 13 | 4 |
| 14 | 2 |
| 15 | 1 |
| 16 | 1 |
| 17 | 1 |
| 18 | 1 |
| 19 | 1 |
| 20 | 1 |
</details>

(c) $\alpha = 0 . 5$

![](images/2bed1a1c64b7653a5a97044703868d45428554a68c590e4249c677e55b5ee3e8.jpg)

<details>
<summary>bar</summary>

| Class | Count |
|---|---|
| 1 | 70 |
| 2 | 60 |
| 3 | 45 |
| 4 | 30 |
| 5 | 28 |
| 6 | 28 |
| 7 | 28 |
| 8 | 25 |
| 9 | 20 |
| 10 | 15 |
| 11 | 10 |
| 12 | 8 |
| 13 | 6 |
| 14 | 5 |
| 15 | 4 |
| 16 | 3 |
| 17 | 2 |
| 18 | 1 |
The data is sorted in descending order based on the count of occurrences. The x-axis represents the class labels (Class), and the y-axis represents the count of occurrences. There is no label for the data series. |
</details>

(d) α = 1

Figure 7: Local distribution of client 0 for different Dir(α) on 20 Newsgroup dataset experiments. 

<table><tr><td>Rank</td><td>FL+LoRA</td><td>FFA-LoRA</td><td>FlexLoRA</td><td>Ours</td></tr><tr><td>8</td><td> $53.80_{\pm 1.44}$ </td><td> $52.60_{\pm 0.96}$ </td><td> $\underline{60.36}_{\pm 1.15}$ </td><td> $\underline{58.74}_{\pm 0.95}$ </td></tr><tr><td>4</td><td> $55.03_{\pm 0.43}$ </td><td> $50.57_{\pm 1.58}$ </td><td> $\underline{59.12}_{\pm 0.98}$ </td><td> $\underline{58.62}_{\pm 1.51}$ </td></tr><tr><td>2</td><td> $50.40_{\pm 0.77}$ </td><td> $48.36_{\pm 0.86}$ </td><td> $\underline{55.46}_{\pm 0.99}$ </td><td> $\underline{59.63}_{\pm 0.59}$ </td></tr><tr><td>1</td><td> $\underline{51.24}_{\pm 3.12}$ </td><td> $46.92_{\pm 1.30}$ </td><td> $51.05_{\pm 0.69}$ </td><td> $\underline{59.11}_{\pm 0.88}$ </td></tr></table>

Table 7: Experiments on pathologic settings.

# C Additional Experiments

Pathologic setting Table 7 provides experiments on pathologic setting, which is also used to generate Figure 5 in Section 5.3, to show the efficacy of adaptive rank selection. In this setting, we have K = 20 clients. And client (2k − 1) and client (2k) exclusively possess half of class (2k  1) and (2k) of 20 Newsgroups datasets, respectively, for k = 1, 2, · · · , 10.

Convergence speed analysis Figure 8 shows the convergence curves of our algorithm and baselines. The figure demonstrates that our algorithm shows similar convergence speed compared to baseline methods in various levels of heterogeneity.

Resource heterogeneity In this experiment, we analyze the efficacy of our algorithm under varying resource constraints across clients. Specifically, we assume that each client has a different communication cost budget (Chen et al., 2023) and has different maximum LoRA rank for its adapter. Following Bai et al. (2024), we simulate three types of resource heterogeneity, as illustrated in Figure 9.

![](images/088c77ed6453f6f673623db7ca4bce583b476c861c444a770c85586621beaa4a.jpg)

<details>
<summary>line</summary>

| Communication Round | Accuracy (Solid Red) | Accuracy (Dashed Blue) | Accuracy (Dash-Dot Red) |
| ------------------- | -------------------- | ---------------------- | ----------------------- |
| 0                   | 0.0                  | 0.0                    | 0.0                     |
| 10                  | 0.8                  | 0.8                    | 0.6                     |
| 20                  | 0.85                 | 0.85                   | 0.7                     |
| 30                  | 0.88                 | 0.88                   | 0.75                    |
| 40                  | 0.9                  | 0.9                    | 0.8                     |
| 50                  | 0.9                  | 0.9                    | 0.85                    |
</details>

(a) Dir(0.5)

![](images/13b9d9a0ed14e2527bacc8c7eb5b5f5d67a5aa4614cf243e76d999d3d6add2ee.jpg)

<details>
<summary>line</summary>

| Communication Round | Accuracy (Solid Red) | Accuracy (Dashed Blue) | Accuracy (Dotted Green) | Accuracy (Dash-Dot Orange) |
| ------------------- | -------------------- | ---------------------- | ----------------------- | -------------------------- |
| 0                   | 0.0                  | 0.0                    | 0.0                     | 0.0                        |
| 10                  | 0.75                 | 0.65                   | 0.70                    | 0.45                       |
| 20                  | 0.85                 | 0.75                   | 0.80                    | 0.55                       |
| 30                  | 0.88                 | 0.80                   | 0.85                    | 0.65                       |
| 40                  | 0.90                 | 0.82                   | 0.88                    | 0.70                       |
| 50                  | 0.92                 | 0.85                   | 0.90                    | 0.72                       |
</details>

(b) Dir(0.1)

![](images/548ab546c0d59d1a332c96896830601ffde409445940b5516c53fc00476d7f78.jpg)

<details>
<summary>line</summary>

| Communication Round | FL + LoRA | FFALoRA | FlexLoRA | Ours |
| ------------------- | --------- | ------- | -------- | ---- |
| 0                   | 0.0       | 0.0     | 0.0      | 0.0  |
| 10                  | 0.3       | 0.2     | 0.4      | 0.4  |
| 20                  | 0.4       | 0.25    | 0.5      | 0.5  |
| 30                  | 0.45      | 0.3     | 0.55     | 0.6  |
| 40                  | 0.5       | 0.35    | 0.6      | 0.65 |
| 50                  | 0.55      | 0.38    | 0.65     | 0.7  |
</details>

(c) Dir(0.01)

Figure 8: Convergence curve of baseline methods in various levels of heterogeneity. Experimented on BANKING77 dataset, with local ranks all set to 2.   
![](images/8b0d372ec07deda24c05ab1821512ef1336ba5f833f2fee66f87c3f8de4ad6cb.jpg)

<details>
<summary>bar</summary>

| Category | LoRA Rank 1 | LoRA Rank 2 | LoRA Rank 4 | LoRA Rank 8 | LoRA Rank 16 | LoRA Rank 32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Uniform | 6 | 6 | 6 | 6 | 6 | 6 |
| Heavy Tail | 16 | 8 | 4 | 2 | 2 | 0 |
| Normal | 3 | 7 | 10 | 7 | 3 | 0 |
</details>

Figure 9: Three types of simulated client rank distributions used to evaluate performance under resource heterogeneity settings.

<table><tr><td rowspan="2">Distribution</td><td rowspan="2">Method</td><td colspan="2">BANKING77 Dataset</td><td rowspan="2">Communicated Parameters</td></tr><tr><td>Dir(0.1)</td><td>Dir(0.01)</td></tr><tr><td rowspan="3">Uniform</td><td>HetLoRA</td><td>86.91±0.43</td><td>68.53±2.14</td><td>3.09B</td></tr><tr><td>FlexLoRA</td><td>73.01±0.69</td><td>45.41±1.60</td><td>3.09B</td></tr><tr><td>Ours</td><td>92.02±0.16</td><td>70.67±0.76</td><td>1.97B</td></tr><tr><td rowspan="3">Heavy Tail</td><td>HetLoRA</td><td>85.82±0.54</td><td>69.57±1.13</td><td>1.06B</td></tr><tr><td>FlexLoRA</td><td>82.69±0.86</td><td>52.46±1.72</td><td>1.06B</td></tr><tr><td>Ours</td><td>91.72±0.07</td><td>69.95±2.23</td><td>0.942B</td></tr><tr><td rowspan="3">Normal</td><td>HetLoRA</td><td>84.57±0.55</td><td>70.34±0.15</td><td>1.34B</td></tr><tr><td>FlexLoRA</td><td>77.08±0.68</td><td>53.37±3.49</td><td>1.34B</td></tr><tr><td>Ours</td><td>92.08±0.18</td><td>69.04±0.64</td><td>0.932B</td></tr></table>

Table 8: Experimental results under resource heterogeneity settings.

In Table 8, we compare our method against Het-LoRA and FlexLoRA, two previous LoRA methods that can handle resource heterogeneity in FL. The experimental results demonstrates that our algorithm shows slightly better or similar performance compared to HetLoRA with less number of communicated parameters.

Efficacy of importance criterion As mentioned in Section 4.2, other criteria such as magnitudebased or importance-based scoring functions can be used for selecting ranks. Table 9 shows that our criterion outperforms others, with less communication than the magnitude-based criterion.

<table><tr><td rowspan="2"></td><td colspan="2">BANKING77 Dataset</td><td rowspan="2">Communicated Parameters</td></tr><tr><td>Dir(0.1)</td><td>Dir(0.01)</td></tr><tr><td>Importance</td><td>91.29±0.76</td><td>66.92±1.58</td><td>0.215B</td></tr><tr><td>Magnitude</td><td>91.71±0.23</td><td>68.00±0.57</td><td>0.651B</td></tr><tr><td>Ours</td><td>92.02±0.36</td><td>69.40±0.48</td><td>0.507B</td></tr></table>

Table 9: Ablation study on scoring functions.

![](images/997d4cb0ef781046ec35910ef55b5f1c39ec5b65c0d038a416d1bf0961bc3af2.jpg)

<details>
<summary>line</summary>

| Heterogeneity | filora | ffalora | ours | ours:even |
| ------------- | ------ | ------- | ---- | --------- |
| IID           | 0.064  | 0.103   | 0.063| 0.060     |
| dir(1)        | 0.063  | 0.086   | 0.061| 0.057     |
| dir(0.5)      | 0.059  | 0.075   | 0.060| 0.059     |
| dir(0.1)      | 0.058  | 0.056   | 0.060| 0.062     |
| dir(0.01)     | 0.051  | 0.043   | 0.057| 0.067     |
</details>

Figure 10: Average Gradient Similarity on various level of heterogeneity. Experimented on 20 Newsgroups dataset and the ranks were all set to 2.

Client drift experiment To thoroughly analyze the impact of data heterogeneity within constrained parameter spaces, we conducted additional experiments that illustrate the local client drift observed in baseline methods operating under these limitations. We quantified the degree of client drift by calculating the "Average Gradient Similarity," defined as follows:

AverageGradientSimilarity =

$$
\frac {1}{n ^ {2}} \sum_ {i} ^ {n} \sum_ {i} ^ {n} \frac {\left(\Delta W _ {i} ^ {t} - \Delta W _ {i} ^ {t - 1}\right) \cdot \left(\Delta W _ {j} ^ {t} - \Delta W _ {j} ^ {t - 1}\right)}{\left| \left| \Delta W _ {i} ^ {t} - \Delta W _ {i} ^ {t - 1} \right| \right| \cdot \left| \left| \Delta W _ {j} ^ {t} - \Delta W _ {j} ^ {t - 1} \right| \right|} \tag {7}
$$

The experimental results presented in Figure 10 indicate a rapid decline in average gradient similarity as the level of heterogeneity increases. In contrast, our method demonstrates greater robustness, exhibiting lower client drift even in rounds where only the LoRA module A is updated. These findings are consistent with the results shown in Figure 2 and Table 1, which illustrate that FFA-LoRA experiences the most significant performance decline between the directional settings of 0.1 and 0.01, while our algorithm maintains its effectiveness in heterogeneous environments.

# D Theoretical Proofs

Here’s brief proof for the proposition made in section 4.3:

Pthe $A _ { i } ^ { \prime } \mathbf { s }$ First, since permanently, $\Omega _ { \mathrm { F F A - L o R A } } ~ = ~ \{ B _ { i } \} _ { i = 1 } ^ { N } .$ Next, since FL + LoRA and FlexLoRA update $B _ { i } { ' } \mathrm { s }$ and $A _ { i } ^ { \prime } \mathbf { s }$ simultaneously, ΩFL + LoRA = $\{ ( B _ { i } , A _ { i } ) \} _ { i = 1 } ^ { N } = \Omega _ { \mathrm { F l e x L o R A } }$ . Finally, $\Omega _ { \mathrm { L o R A - A ^ { 2 } } } =$ $\left\{ \left( \bar { B } _ { i } , \bar { A } _ { i } \right) \right\} _ { i = 1 } ^ { N }$ N , where its subspace ptimized according to $\{ B _ { i } \} _ { i = 1 } ^ { N }$ orlter-$\{ A _ { i } \} _ { i = 1 } ^ { N }$ rithm. Therefore, noting that $r \quad \leq \quad r _ { G }$ , we have $\Omega _ { \mathrm { F F A - L o R A } } \subsetneq \Omega _ { \mathrm { F L + L o R A } } = \Omega _ { \mathrm { F l e x L o R A } } \subset$ $\Omega _ { \mathrm { L o R A - A ^ { 2 } } } . \square$