# NDAD: NEGATIVE-DIRECTION AWARE DECODING FOR LARGE LANGUAGE MODELS VIA CONTROLLABLE HALLUCINATION SIGNAL INJECTION

Panjia Qiu1 Mingyuan Fan1 Cen Chen1∗ Daixin Wang2

School of Data Science and Engineering, East China Normal University 2AntGroup

panjiaqiu@stu.ecnu.edu.cn fmy2660966@gmail.com

cenchen@dase.ecnu.edu.cn daixin.wdx@antgroup.com

# ABSTRACT

Large language models (LLMs) have recently achieved impressive progress in knowledge-intensive and reasoning tasks. However, their tendency to produce fabricated or factually inconsistent content remains a fundamental challenge to their practical deployment. To address this issue, we propose Negative-Direction Aware Decoding (NDAD), a novel decoding method that identifies and exploits hallucination signals as repulsive directions in the model’s representation space, thereby improving factual adherence without retraining. Specifically, NDAD elicits hallucination-leaning signals by selectively masking critical attention heads, which exposes unstable hypotheses that the model would otherwise amplify during generation. To regulate the influence of these signals, NDAD employs two complementary weights: a global alignment weight measuring how well the induced signal aligns with the layer’s native activations (thus quantifying its referential utility) and a local weight estimating whether low-probability tokens in the masked distribution are likely to evolve toward the final output. Based on the weights, we derive a latent hallucination distribution that serves as the negative direction. A lightweight gradient-descent step then subtracts mass from hallucination-prone regions of the output distribution, adjusting the final logits while preserving the model’s high-confidence predictions. Extensive experiments across multiple LLMs and diverse benchmark datasets demonstrate that NDAD consistently enhances factual reliability without requiring additional training or external knowledge.

# 1 INTRODUCTION

In recent years, large language models (LLMs) have achieved remarkable breakthroughs over various tasks (Achiam et al., 2023; Anil et al., 2023; Touvron et al., 2023b;a; Team et al., 2023). However, a pervasive challenge is the phenomenon of hallucination, wherein LLMs generate factually incorrect, fabricated, or nonsensical information with high confidence (Zhang et al., 2023b; Li et al., 2024; Fan et al., 2025).

Current approaches to mitigate hallucinations mainly fall into two categories: retrieval-augmented methods (Li et al., 2023b; Min et al., 2023) and training-based methods (Tian et al., 2023; Rafailov et al., 2023). Retrieval-augmented methods, while effective, often introduce architectural complexity, latency, and dependency on the availability and integrity of external large-scale databases. Trainingbased methods, on the other hand, can be computationally intensive and may struggle to generalize across diverse factual domains. A less explored, yet highly promising, avenue is the optimization of the decoding process itself (Welleck et al., 2024; Shi et al., 2024). Importantly, prior studies suggest that LLMs already encode factual signals within their internal representations as a byproduct of largescale pretraining, though conventional decoding techniques often fail to surface this latent knowledge (Wang et al., 2020; Kadavath et al., 2022; Li et al., 2023a; Saunders et al., 2023). Motivated by this observation, intervention-based decoding methods (Chuang et al., 2023; Li et al., 2023a; 2022; Zhang et al., 2023a) have been developed to exploit these factual signals to alleviate hallucinations.

In this study, we propose a novel yet quite effective intervention decoding method called Negative-Direction Aware Decoding (NDAD). Unlike prior approaches that focus on extracting latent factual cues from early layers, NDAD instead identifies hallucination signals and then leverages them to calibrate the decoding process. As shown in Figure 1, NDAD detects latent distributions that correlate with factually incorrect outputs by masking influential attention heads in the model. We then develop two complementary weights based on the detected distributions to suppress hallucination signals. Specifically, the global weighting component evaluates the alignment between hallucination-oriented logits and earlier-layer logits, estimating whether such trajectories reflect distributions the model is more likely to generate. In parallel, the local weighting component tracks high-risk tail tokens to assess their likelihood of advancing toward the final output. The final token selection is then guided by a single-step gradient-descent adjustment, which penalizes the generation of tokens associated with identified hallucination risks. Our main contributions are as follows:

• We propose NDAD, an innovative decoding approach that introduces hallucination signal to expose the model’s underlying hallucination distribution and applies a negative awareness mechanism for intervention.   
• We incorporate a global weight measuring the directional consistency between hallucination signal and original early-layer logits, and a local weight quantifying the likelihood of tail tokens evolving toward the mature distribution.   
• We perform comprehensive experiments on a diverse set of LLMs with different configurations and scales. The experimental results indicate that NDAD reliably enhances factual accuracy across multiple tasks and benchmark datasets.

# 2 RELATED WORK

Hallucination Mitigation. In LLMs, hallucination refers to the generation of content that diverges from factual knowledge, and it has become a critical bottleneck for ensuring model reliability. Existing research has proposed mitigation strategies along several directions. Retrieval-based methods introduce external knowledge to calibrate factuality, such as Retrieval-Augmented Generation (RAG) (Cheng et al., 2023; Chen et al., 2024a; Fan et al., 2024; Lewis et al., 2020), or enhance attribution by applying retrieval and editing after generation to improve both factuality and traceability (Gao et al., 2022; Mishra et al., 2024). Training- and preference-based methods rely on additional supervised data or human preference signals for optimization, including Supervised Fine-Tuning (SFT) (Tian et al., 2023), Reinforcement Learning with Human Feedback (RLHF) (Ouyang et al., 2022), and Direct Preference Optimization (DPO) (Rafailov et al., 2023), thereby reducing hallucination through parameter updates. Self-evaluation-based methods do not rely on external data; instead, they improve reliability by leveraging multiple inference-time samples and incorporating techniques such as self-criticism (Saunders et al., 2023) and diversified reasoning path sampling (Wang et al., 2022). To further improve efficiency in enhancing factuality, our goal is to directly optimize the output distribution of language models, thereby strengthening their robustness.

Intervention Decoding. In recent years, a line of research has emerged that enhances the factuality of LLMs by intervening during the decoding time. Inference-Time Intervention (ITI) (Li et al., 2023a) identifies attention heads correlated with truthfulness during inference and shifts activations along these “truthful directions”, thereby enhancing the truthfulness of generated outputs. Similarly, Activation Decoding (AD) (Chen et al., 2024b) leverages the model’s internal representations by introducing an entropy-based metric of contextual activation sharpness as a decoding constraint, thereby biasing outputs toward more reliable generations. Inspired by early work on Contrastive Decoding (CD) (Li et al., 2022), which compared strong expert models against weaker amateur models to improve fluency and coherence without addressing factuality, subsequent studies extended the idea of “contrast” to the logits level. For example, Auto-Contrastive Decoding (ACD) (Gera et al., 2023) requires fine-tuning the prediction heads of earlier layers and is therefore mainly applicable to small-scale models. In contrast, Decoding by Contrasting Layers (DoLA) (Chuang et al., 2023) dynamically selects the early layer that exhibits the largest semantic divergence from the final layer, thereby suppressing erroneous tendencies in lower layers. Building upon this, Self Logits Evolution Decoding (SLED) (Zhang et al., 2024) further integrates multiple early layers through weighted combination and employs a gradient-descent procedure to guide the correction of the final logits, resulting in more robust factuality enhancement.

![](images/d6e0b2ac9ae0a3ffacca987e69bce7f6b939d520f57198ddf993aef73a036a1b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Where were the Games of the 30th Olympiad held?"] --> B["Vanilla Forward Propagation"]
    B --> C["Original: logits_L"]
    C --> D{The top I tokens indicate the approximate correctness direction}
    D --> E["logits_l^mask"]
    E --> F["logits_l^mask_l"]
    F --> G["logits_l^mask_l^global"]
    G --> H["logits_8^mask"]
    G --> I["logits_16^mask"]
    G --> J["logits_24^mask"]
    G --> K["logits_L^mask"]
    H --> L["Tail tokens distribution"]
    I --> L
    J --> L
    K --> L
    L --> M["Contrastive Decoding"]
    M --> N["Logits_L^new = logits_L - αVKL(P_{NDAD}, P_{logits_L})"]
    N --> O["True"]
    O --> P["End"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#fcf,stroke:#333
    style H fill:#cff,stroke:#333
    style I fill:#ffc,stroke:#333
    style J fill:#cff,stroke:#333
    style K fill:#ffc,stroke:#333
    style L fill:#cfc,stroke:#333
    style M fill:#fcc,stroke:#333
    style N fill:#ffc,stroke:#333
    style O fill:#cfc,stroke:#333
```
</details>

Figure 1: Overview of NDAD. To enhance factual reliability, we introduce hallucination signals to adjust the final output logits. The weights $\mathcal { W } ^ { \mathrm { g l o b a l } }$ and $\mathcal { W } ^ { \mathrm { l o c a l } }$ jointly regulate the hallucination signals to form the latent allucination distribution $\mathcal { P } _ { \mathrm { N D A D } }$ , from which the model steers its output away.

# 3 METHOD

LLMs are designed to autoregressively predict the next token given a preceding context. Formally, into a sequence of embedding vectors, given an input prefix represented as $\mathbf { x } _ { < t } = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { t - 1 } \}$ $\mathcal { H } _ { 0 } = \{ h _ { 0 } ^ { [ 1 ] } , h _ { 0 } ^ { [ 2 ] } , \dots , h _ { 0 } ^ { [ t - 1 ] } \}$ [2] , , the model first converts these tokens h[t−1]0 }, through an embedding layer. These representations are then updated successively by a stack of L transformer blocks. We denote the hidden state of the t-token at the l-th block as $\bar { h _ { l } ^ { [ t ] } } \in \mathbb { R } ^ { d _ { h } }$ . To generate a probability distribution over the model’s vocabulary $\nu ,$ a shared projection head $\psi : \mathbb { R } ^ { d _ { h } }  \mathbb { R } ^ { d }$ is applied to the hidden states. In detail, from the l-th layer’s hidden state, the unnormalized score vector (logits) for the next token and its corresponding probability distribution are defined as:

$$
P _ {l} ^ {[ t ]} = \text { softmax } (l o g i t s _ {l} ^ {[ t ]}), \text {   where   } l o g i t s _ {l} ^ {[ t ]} = \psi (h _ {l} ^ {[ t ]}), l = 1, \dots , L. \tag {1}
$$

Typically, the logits from the final layer, to generations that are plausible but fact $l o g i t s _ { L } ^ { [ t ] }$ , are used for decoding. However, this can leadorrect or nonsensical. To mitigate this issue, we propose NDAD, which adjusts the logits by leveraging hallucination signal, thereby improving the reliability of the generated text.

# 3.1 HALLUCINATION SIGNAL GENERATION

Unlike prior approaches such as DoLa (Chuang et al., 2023) and SLED (Zhang et al., 2024), which harness early-layer representations as a proxy for faithful evidence to reshape the final token distribution, we instead attempt to explicitly separate the hallucination signal to encourage the final output distribution to diverge from it. Intuitively, this shifts calibration from boosting positives to subtracting negatives. In this way, our method can prevent probability mass from accumulating on spurious or speculative trajectories.

Prior studies (Wu et al., 2024) have demonstrated that certain attention heads in LLMs play a critical role in preserving factuality and stabilizing generation. Once the support of these heads is weakened, the model tends to deviate from factual directions, making the decoding process more susceptible to hallucinations. Building on this insight, we exclusively mask influential heads to isolate a hallucination signal, which serves as a negative direction for contrastive decoding. To determine which heads should be masked, we adopt head importance scores from prior work (Wu et al., 2024)

to evaluate the importance of each head. Furthermore, we take into account the entropy of each layer’s distribution: a lower entropy indicates that the importance is concentrated on a small subset of heads, suggesting that these heads are more influential. By integrating both head importance and layer-level entropy, we achieve a more precise selection of heads to be masked. As illustrated in Figure 2, for each block $l \in L .$ , following Wu et al. (2024), we first obtain a score list of n heads $\{ s _ { l , 1 } , s _ { l , 2 } , . . . , s _ { l , n } \}$ in this block. We then normalize the scores into a probability distribution and compute the layer entropy as follows:

$$
E _ {l} = - \sum_ {i = 1} ^ {n} p _ {l, i} \log p _ {l, i}, \quad p _ {l, i} = \frac {s _ {l , i}}{\sum_ {j = 1} ^ {n} s _ {l , j}}, \quad i = 1, \dots , n, \tag {2}
$$

Here, $p _ { l , i }$ denotes the normalized importance of head i in layer l, and $E _ { l }$ measures the uncertainty of head importance within this layer. Then we select the top K layers with the lowest entropy and mask the top x heads within these layers to separate hallucination signal. We represent the hallucination signal corresponding to l-th block as $l o g i t s _ { l } ^ { \mathrm { m a s k } }$ . The complete algorithmic workflow can be found in Appendix Algorithm 1. After extracting these hallucination signals, the remaining question is how to leverage them to calibrate the model’s final outputs.

![](images/ab980ae642198d2990bbdc33cb05ca4038f9a9b7d4f2d1c7a804f9c5f3bbdd0a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["H"] -->|Capture Information| B["Could the members of The Police perform lawful arrests?"]
    B --> C["E1 E2 E3 ... EL-1 EL"]
    C --> D["Sorted by Entropy"]
    D --> E["E1-1 EL EL 3 ... EL-1 EL"]
    E --> F["HALLCINATION SIGNALS"]
    F --> G["H1-1,1 HL-1,2 HL-1,3 ... HL-1,n-1 HL-1,n"]
    F --> H["H1,1 HL2 HL3 ... HL-n-1 HL,n"]
    F --> I["..."]
    F --> J["H2,1 HL2,2 HL2,3 ... HL-n-1 HL2,n"]
    F --> K["H1,1 HL2 HL3 ... HL-n-1 HL,n"]
```
</details>

Figure 2: Hallucination signal generation. Darker colors indicate larger values, and gray cells correspond to masked attention heads.

# 3.2 DYNAMIC WEIGHTING VIA GLOBAL CONSISTENCY AND LOCAL DIVERGENCE

To exploit the identified negative direction, we propose a dynamic weighting framework that integrates both global and local perspectives.

Global Consistency. At the global level, we evaluate the directional consistency between the hallucination signal and the original early-layer logits from the same layer, which provides a quantitative assessment of the correlation between the original signal and the hallucination signal. Specifically, the directional consistency cl at layer l is measured by computing the cosine similarity between the hallucination signal $l o g i t s _ { l } ^ { \mathrm { { m a s k } } }$ and the original logits logitsl at the same layer l:

$$
\mathcal {W} _ {l} ^ {\text { global }} = \varphi (c _ {l}), \quad c _ {l} = \cos \_ \text { sim } \big (\text { logits } _ {l}, \text { logits } _ {l} ^ {\text { mask }} \big)  . \tag {3}
$$

where $\varphi ( \cdot )$ denotes a linear mapping that scales values into the range [0, 1]. By measuring directional consistency, we assess the correlation between the hallucination signal $\dot { l } o g i t s _ { l } ^ { \mathrm { m a s k } }$ and the model’s original logits $l o g i t s _ { l } ,$ thereby providing a quantitative basis for the referential value of hallucination signal at layer l. A higher consistency indicates that the signal is more closely aligned with the model’s latent hallucination direction. Accordingly, the weighting scheme increases the contribution of more relevant hallucination signals.

Local Divergence. At the local level, we further examine the distribution of low-probability tokens. Consistent with prior studies (Chuang et al., 2023; Zhang et al., 2024), we approximate the final-layer logits $l o g i t s _ { L }$ as the ground-truth distribution. We define the evolution trajectory from the premature to the mature state as $\overline { { { l o g i t s } } } _ { L } - l o g i t s _ { l } ^ { \mathrm { m a s k } }$ . For the final mature layer, we further obtain the probability distribution $\mathcal { P } _ { l o g i t s _ { L } } = s o f t m a x ( l o \dot { g } i t s _ { L } )$ ), select the top-I tokens, and construct I one-hot vectors $\mathcal { T } = \{ \mathcal { P } _ { e _ { 1 } } , \mathcal { P } _ { e _ { 2 } } , . . . , \mathcal { P } _ { e _ { I } } \}$ to serve as I approximate distributions of mature, where the index of the selected token is set to 1 and all others are set to 0. In order to derive the hallucination distribution at layer l, we begin with $\mathcal { P } _ { l o g i t s _ { l } ^ { \mathrm { m a s k } } } = s o f t m a x ( l o g i t s _ { l } ^ { \mathrm { m a s k } } )$ . Mahaut et al. (2024) suggested that lowprobability tokens typically correspond to reduced factuality. Based on this observation, we define our final hallucination distribution by removing the top-I tokens, which encourages the resulting distribution to approximate the negative direction more closely. In particular, by assigning a very small probability $\epsilon  0$ to the top-I tokens in $\mathcal { P } _ { l o g i t s _ { l } ^ { \mathrm { m a s k } } }$ , we are able to derive a cleaner representation of the premature distribution, which is defined as $\mathcal { P } _ { l o g i t s _ { l } ^ { \mathrm { m a s k } } } ^ { \mathrm { t a i l } }$ P taillogitsmask . As illustrated in Figure 1, the vector d1 $d _ { 1 }$ denotes the evolution trajectory from the premature signal $l o g i t s _ { l } ^ { \mathrm { m a s k } }$ at layer l to the mature signal $l o g i t s _ { L }$ , while $d _ { 2 }$ represents the trajectory from the hallucination distribution $P _ { l o g i t s _ { l } ^ { \mathrm { m a s k } } } ^ { t a i l }$ Plogitsmask toward a candidate distribution of correctness $P _ { e _ { i } }$ . Both $d _ { 1 }$ and $d _ { 2 }$ can be interpreted as representations of the the trajectory of factual evolution, and thus we have:

$$
d _ {1} \stackrel {\text { direction }} {\approx} d _ {2}, \text { where } d _ {1} = \operatorname{logits} _ {L} - \operatorname{logits} _ {l} ^ {\text { mask }}, d _ {2} = \nabla K L (\mathcal {P} _ {\operatorname{logits} _ {l} ^ {\text { mask }}} ^ {\text { tail }}, \mathcal {P} _ {e _ {i}}). \tag {4}
$$

Intuitively, if $d _ { 1 }$ and $d _ { 2 }$ are more closely aligned, it indicates that the token in $P _ { l o g i t s _ { l } ^ { \mathrm { m a s k } } } ^ { t a i l }$ P taillogitsmask is more likely to evolve toward the mature output, and therefore a larger weight should be assigned to suppress its evolution. To quantify this evolution trajectory, we define the local weight as:

$$
\mathcal {W} _ {l, i} ^ {\text { local }} = \max \left(\cos_ {-} \operatorname{sim} \left(\operatorname{logits} _ {l} ^ {\text { mask }} - \operatorname{logits} _ {L}, \mathcal {P} _ {\operatorname{logits} _ {l} ^ {\text { mask }}} ^ {\text { tail }} - \mathcal {P} _ {e _ {i}}\right), 0\right), \quad i \in [ 1, I ]. \tag {5}
$$

After deriving both the global and local weights, we integrate them to obtain the final weight for each correctness direction within the top-I tokens. Specifically, for the one-hot vector $P _ { e _ { i } }$ corresponding to the i-th distribution in the correctness, the final weight at layer l is defined as:

$$
\mathcal {W} _ {l, i} = \mathcal {W} _ {l} ^ {\text { global }} \mathcal {W} _ {l, i} ^ {\text { local }}, \quad i \in [ 1, I ]. \tag {6}
$$

To better capture dominant signals and attenuate weak or noisy ones, we apply a squared transformation to the final weight scores. This operation accentuates high-confidence directions while diminishing the influence of marginal ones, thereby producing a sharper weighting distribution (Hinton et al., 2015; Muller et al., 2019; Zhang et al., 2021). Formally, the squared weight is: ¨

$$
\tilde {\mathcal {W}} _ {l, i} = \left(\mathcal {W} _ {l, i}\right) ^ {2}, \quad i \in [ 1, I ] \tag {7}
$$

# 3.3 NEGATIVE-DIRECTION AWARE DECODING

After introducing the global and local weighting mechanisms, we now integrate them into the overall decoding framework. NDAD leverages these weights to controllably exploit the injected hallucination signal and employs an update in the direction of gradient-descent to guide the model away from hallucination directions during generation. The following describes the specific procedure for adjusting the final-layer logits, we first perform intra-layer normalization on the obtained signals, followed by inter-layer aggregation. The squared weights $\tilde { \mathcal { W } } _ { l , i }$ are normalized across the I correctness directions within each layer, resulting in a layer-wise normalized distribution. Formally, the latent distribution of layer l is expressed as:

$$
\mathcal {P} _ {l} = \left(\tilde {\mathcal {W}} _ {l, 1}, \tilde {\mathcal {W}} _ {l, 2}, \dots , \tilde {\mathcal {W}} _ {l, | I |}\right) / \mathcal {Z} _ {l}, \quad \mathcal {Z} _ {l} = \sum_ {i = 1} ^ {I} \tilde {\mathcal {W}} _ {l, i} \tag {8}
$$

We further apply inter-layer weighting to obtain the final NDAD distribution:

$$
\mathcal {P} _ {\mathrm{NDAD}} = \sum_ {l = 1} ^ {L} \mathcal {N} _ {l} \mathcal {P} _ {l}, \quad \text { where } \quad \mathcal {N} _ {l} = \frac {\mathcal {Z} _ {l}}{\sum_ {l = 1} ^ {L} \mathcal {Z} _ {l}}. \tag {9}
$$

Here, $\mathcal { N } _ { l }$ denotes the relative contribution of layer l, ensuring that the aggregation respects the proportional importance of each layer while preserving comparability across layers. By incorporating negative-direction awareness, we obtain a latent hallucination distribution $P _ { \mathrm { N D A D } }$ . To suppress the generation of hallucination-prone tokens, we penalize the divergence between distribution $\mathcal { P } _ { \mathrm { N D A D } }$ and the original distribution $\mathcal { P } _ { l o g i t s _ { L } }$ using the KL divergence term. The procedure is outlined in Algorithm 2. Here, the parameter α, referred to as the Evolution Rate and originally introduced in the (Zhang et al., 2024), controls the magnitude of adjustment applied to the logits along the gradient direction. We then obtain the final adjusted logits as shown below:

$$
\operatorname{logits} _ {L} ^ {\text { new }} = \operatorname{logits} _ {L} - \alpha \nabla K L \left(\mathcal {P} _ {\mathrm{NDAD}}, \mathcal {P} _ {\text { logits } _ {L}}\right) \tag {10}
$$

Table 1: Evaluation results of different methods on Llama models over varying datasets. 

<table><tr><td rowspan="2">Method</td><td colspan="4">TruthfluQA(MC)</td><td>Factor</td><td colspan="2">CoT</td></tr><tr><td>MC1</td><td>MC2</td><td>MC3</td><td>Avg.</td><td>Wiki</td><td>StrQA</td><td>GSM8K</td></tr><tr><td>Llama2-7B-base</td><td>26.58</td><td>41.88</td><td>18.96</td><td>29.14</td><td>58.42</td><td>60.74</td><td>13.95</td></tr><tr><td>+DoLa-low</td><td>33.04</td><td>63.73</td><td>31.25</td><td>42.67</td><td>63.36</td><td>59.56</td><td>14.63</td></tr><tr><td>+DoLa-high</td><td>31.77</td><td>63.26</td><td>30.40</td><td>41.81</td><td>62.56</td><td>60.44</td><td>13.19</td></tr><tr><td>+AD</td><td>32.41</td><td>49.89</td><td>24.03</td><td>35.44</td><td>53.14</td><td>1.97</td><td>2.12</td></tr><tr><td>+SLED</td><td>34.15</td><td>62.57</td><td>31.89</td><td>42.87</td><td>67.00</td><td>61.27</td><td>14.63</td></tr><tr><td>+NDAD</td><td>34.39</td><td>62.62</td><td>31.98</td><td>43.00</td><td>67.30</td><td>61.57</td><td>14.86</td></tr><tr><td>Llama2-7B-chat</td><td>35.62</td><td>57.47</td><td>32.10</td><td>41.73</td><td>56.68</td><td>63.58</td><td>21.23</td></tr><tr><td>+DoLa-low</td><td>34.18</td><td>62.80</td><td>31.00</td><td>42.66</td><td>56.58</td><td>64.59</td><td>21.46</td></tr><tr><td>+DoLa-high</td><td>33.92</td><td>61.75</td><td>30.40</td><td>42.02</td><td>56.25</td><td>64.19</td><td>20.85</td></tr><tr><td>+AD</td><td>32.15</td><td>49.90</td><td>23.99</td><td>35.35</td><td>51.44</td><td>0.48</td><td>1.44</td></tr><tr><td>+SLED</td><td>37.09</td><td>63.83</td><td>32.96</td><td>44.63</td><td>64.80</td><td>64.50</td><td>21.53</td></tr><tr><td>+NDAD</td><td>36.84</td><td>63.42</td><td>32.93</td><td>44.40</td><td>65.06</td><td>64.67</td><td>21.99</td></tr><tr><td>Llama2-13B-base</td><td>27.59</td><td>43.14</td><td>19.53</td><td>30.09</td><td>63.79</td><td>65.98</td><td>28.81</td></tr><tr><td>+DoLa-low</td><td>31.57</td><td>62.48</td><td>30.41</td><td>41.49</td><td>65.70</td><td>66.46</td><td>28.51</td></tr><tr><td>+DoLa-high</td><td>29.38</td><td>63.92</td><td>33.62</td><td>42.31</td><td>52.84</td><td>60.83</td><td>11.90</td></tr><tr><td>+AD</td><td>32.15</td><td>49.90</td><td>23.99</td><td>35.35</td><td>58.18</td><td>2.01</td><td>0.00</td></tr><tr><td>+SLED</td><td>34.76</td><td>63.58</td><td>31.88</td><td>43.41</td><td>70.94</td><td>66.51</td><td>29.19</td></tr><tr><td>+NDAD</td><td>34.88</td><td>63.60</td><td>31.97</td><td>43.48</td><td>71.18</td><td>66.81</td><td>29.26</td></tr><tr><td>Llama2-13B-chat</td><td>36.47</td><td>63.06</td><td>32.77</td><td>44.10</td><td>61.96</td><td>69.65</td><td>36.69</td></tr><tr><td>+DoLa-low</td><td>34.27</td><td>63.27</td><td>31.36</td><td>42.97</td><td>60.69</td><td>69.48</td><td>35.48</td></tr><tr><td>+DoLa-high</td><td>31.82</td><td>62.55</td><td>31.13</td><td>41.83</td><td>54.81</td><td>66.51</td><td>33.21</td></tr><tr><td>+AD</td><td>32.15</td><td>49.90</td><td>23.99</td><td>35.35</td><td>56.71</td><td>23.14</td><td>0.00</td></tr><tr><td>+SLED</td><td>37.45</td><td>63.50</td><td>32.90</td><td>44.62</td><td>67.50</td><td>69.74</td><td>37.15</td></tr><tr><td>+NDAD</td><td>37.58</td><td>63.63</td><td>33.02</td><td>44.74</td><td>67.74</td><td>69.96</td><td>37.30</td></tr></table>

# 4 EXPERIMENTS

# 4.1 EXPERIMENTAL SETUP

Benchmark datasets. We evaluate our approach against strong baselines across both multiple-choice and open-ended generation tasks. For multiple-choice settings, we employ the TruthfulQA (Lin et al., 2021) dataset to measure factuality in short-answer scenarios and the FACTOR (Wiki) (Muhlgay et al., 2023) dataset to assess performance in long-paragraph contexts. For open-ended generation, we consider PopQA (Mallen et al., 2022), NQ-Open (Lee et al., 2019), and TriviaQA (Joshi et al., 2017), as well as reasoning-intensive tasks involving chain-of-thought (CoT), including StrategyQA (Geva et al., 2021) and GSM8K (Cobbe et al., 2021).

Models and Baselines. In our experiments, we adopt a diverse set of representative open-source LLMs, including Llama2-7B (base and chat) (Touvron et al., 2023b), Llama2-13B (base and chat) (Touvron et al., 2023b), Qwen2.5-7B-instruct (Team, 2024), Mistral-7B-instruct (Jiang et al., 2023), and Llama3-8B-instruct (Grattafiori et al., 2024). We compare the following baselines: (1) Greedy Decoding. (2) DoLA-Low (Chuang et al., 2023) subtracts the logits of the most distributionally different layer from the first half of the network from the final-layer logits. (3) DoLA-High (Chuang et al., 2023) subtracts the logits of the most distributionally different layer from the second half of the network from the final-layer logits. (4) AD (Chen et al., 2024b) uses an entropy-based measure of contextual activation sharpness to constrain decoding with the model’s internal representations. (5) SLED (Zhang et al., 2024) integrates multiple early layers via weighted combination and applies a gradient-descent adjustment to refine the final logits for improved factuality.

Metrics and Parameters. For multiple-choice and CoT reasoning tasks, we evaluate factual accuracy following the approach in (Chuang et al., 2023). To assess correctness on TriviaQA, HotpotQA, and NQ-Open, we adopt the Exact Match (EM) metric, consistent with the protocol of (Joshi et al., 2017). The detailed parameter settings are provided in Appendix A.1.

Table 2: Evaluation results on Open-Ended generation tasks. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Llama2-7B-base</td><td colspan="3">Llama2-7B-chat</td></tr><tr><td>TriviaQA</td><td>PopQA</td><td>NQ-Open</td><td>TriviaQA</td><td>PopQA</td><td>NQ-Open</td></tr><tr><td>Greedy</td><td>65.04</td><td>13.67</td><td>21.02</td><td>59.61</td><td>18.55</td><td>23.41</td></tr><tr><td>+DoLa-low</td><td>64.96</td><td>13.88</td><td>20.78</td><td>54.65</td><td>19.64</td><td>23.60</td></tr><tr><td>+DoLa-high</td><td>63.96</td><td>13.41</td><td>19.31</td><td>54.24</td><td>19.48</td><td>23.55</td></tr><tr><td>+AD</td><td>48.78</td><td>15.11</td><td>22.44</td><td>59.64</td><td>18.43</td><td>23.60</td></tr><tr><td>+SLED</td><td>65.10</td><td>25.86</td><td>25.96</td><td>59.61</td><td>19.98</td><td>23.46</td></tr><tr><td>+NDAD</td><td>65.21</td><td>26.00</td><td>26.26</td><td>59.67</td><td>20.13</td><td>23.63</td></tr><tr><td></td><td colspan="3">Llama2-13B-base</td><td colspan="3">Llama2-13B-chat</td></tr><tr><td>Greedy</td><td>68.34</td><td>25.04</td><td>32.71</td><td>66.32</td><td>19.82</td><td>30.03</td></tr><tr><td>+DoLa-low</td><td>68.67</td><td>28.64</td><td>28.78</td><td>65.54</td><td>17.82</td><td>29.14</td></tr><tr><td>+DoLa-high</td><td>62.08</td><td>26.12</td><td>25.68</td><td>61.86</td><td>16.32</td><td>27.42</td></tr><tr><td>+AD</td><td>67.67</td><td>17.91</td><td>30.80</td><td>64.50</td><td>22.91</td><td>34.52</td></tr><tr><td>+SLED</td><td>71.47</td><td>30.53</td><td>32.52</td><td>66.40</td><td>19.84</td><td>29.89</td></tr><tr><td>+NDAD</td><td>71.66</td><td>30.64</td><td>32.88</td><td>66.48</td><td>19.85</td><td>30.11</td></tr></table>

# 4.2 EVALUATION ON DIFFERENT BENCHMARKS

Multiple-Choices Tasks. These tasks are designed to evaluate whether the decoding strategy can more effectively assign higher probabilities to correct answers or reasonable completions, while suppressing its preference for incorrect options. It should be noted that these tasks are essentially distribution-fitting problems, and overfitting to specific tasks often undermines the generalization capability of a decoding method. Since our goal is to enhance factuality and robustness while preserving broad applicability, even when the performance deviation is small or only marginal improvements are achieved, the results remain understandable and acceptable. We validated the effectiveness of the NDAD method through short-answer factuality tests on the TruthfulQA dataset and long-paragraph factuality tests on the FACTOR dataset. The corresponding experimental results are summarized in Table 1, and more detailed analyses are provided in Appendix B.1. Our NDAD method demonstrates strong generalization across different models and datasets, and largely achieves improvements over the baseline SLED. This suggests that the proposed decoding strategy is generally more effective at calibrating probability assignment between correct and incorrect answers.

Chain-of-Thought Reasoning Tasks. This task primarily focuses on evaluating how different decoding methods can be adapted to the CoT strategy to effectively handle complex reasoning problems. The detailed results can be found in Table 1. Our NDAD method consistently outperforms all baselines in decoding performance. At the same time, the limitations of AD become particularly evident on CoT datasets. AD constrains next-token probabilities by incorporating contextual entropy to enhance factuality. However, it falls short on reasoning tasks because tokens in CoT datasets exhibit strong logical dependencies, and relying solely on token-level activation entropy from the context may deviate from the original semantics. Moreover, some intermediate tokens lack contextual support and are prone to being misclassified as hallucinations, thereby impairing reasoning performance.

Open-Ended Generation Tasks. For open-ended tasks, we adopt TriviaQA, PopQA, and NQ-Open datasets. Our NDAD method consistently achieves further improvements over the baselines. Results are shown in Table 2. Since PopQA and NQ-Open are highly knowledge-intensive, models tend to rely more on contextual information during generation. The AD method, which is inherently designed to adjust decoding based on contextual attention, therefore shows exceptionally strong reasoning performance on the Llama-13B-chat model. However, when compared with the results on CoT tasks in Table 1, it becomes evident that AD exhibits substantial variability. Therefore, our NDAD method demonstrates the strongest robustness.

# 4.3 EVALUATION ON DIFFERENT LLMS

We further conduct experiments on a broader range of model architectures, including models from different families as well as different variants within the same family. As reported in Table 3, NDAD consistently delivers state-of-the-art results across all tested configurations, surpassing other baselines. This demonstrates that the proposed method is not only effective for a specific model class but also generalizes well across diverse architectures. Moreover, the performance gains are particularly pronounced on CoT datasets such as GSM8K, where NDAD exhibits substantial improvements over the baselines. This finding highlights the robustness of NDAD in handling complex reasoning tasks. Consequently, these results confirm that NDAD achieves both cross-model generality and strong robustness, making it a versatile and effective decoding strategy.

Table 3: Evaluation results on varying LLMs. 

<table><tr><td rowspan="2">Model</td><td colspan="4">TruthfluQA(MC)</td><td>Factor</td><td>CoT</td></tr><tr><td>MC1</td><td>MC2</td><td>MC3</td><td>Avg.</td><td>Wiki</td><td>GSM8K</td></tr><tr><td>Qwen2.5-7B-instruct</td><td>41.00</td><td>64.59</td><td>38.17</td><td>47.92</td><td>54.54</td><td>84.46</td></tr><tr><td>+DoLa-low</td><td>36.60</td><td>66.03</td><td>34.21</td><td>45.61</td><td>56.08</td><td>83.02</td></tr><tr><td>+DoLa-high</td><td>34.64</td><td>2.37</td><td>34.51</td><td>23.84</td><td>40.85</td><td>76.95</td></tr><tr><td>+SLED</td><td>45.04</td><td>70.37</td><td>39.88</td><td>51.76</td><td>62.99</td><td>84.91</td></tr><tr><td>+NDAD</td><td>45.17</td><td>70.37</td><td>39.89</td><td>51.81</td><td>63.13</td><td>85.14</td></tr><tr><td>Mistral-7B-instruct</td><td>40.27</td><td>68.32</td><td>37.06</td><td>48.55</td><td>60.49</td><td>53.45</td></tr><tr><td>+DoLa-low</td><td>39.53</td><td>68.44</td><td>36.16</td><td>48.04</td><td>64.16</td><td>53.22</td></tr><tr><td>+DoLa-high</td><td>39.53</td><td>68.43</td><td>36.09</td><td>48.02</td><td>64.23</td><td>53.30</td></tr><tr><td>+SLED</td><td>45.41</td><td>71.17</td><td>40.27</td><td>52.28</td><td>67.53</td><td>53.90</td></tr><tr><td>+NDAD</td><td>45.53</td><td>71.31</td><td>40.46</td><td>52.43</td><td>67.70</td><td>54.36</td></tr><tr><td>Llama3-8B-instruct</td><td>38.92</td><td>68.16</td><td>36.56</td><td>47.88</td><td>59.22</td><td>75.97</td></tr><tr><td>+DoLa-low</td><td>35.74</td><td>65.27</td><td>33.60</td><td>44.87</td><td>61.32</td><td>75.82</td></tr><tr><td>+DoLa-high</td><td>35.99</td><td>65.04</td><td>33.72</td><td>44.92</td><td>61.29</td><td>75.51</td></tr><tr><td>+SLED</td><td>41.37</td><td>68.46</td><td>37.61</td><td>49.15</td><td>67.07</td><td>75.82</td></tr><tr><td>+NDAD</td><td>41.37</td><td>69.21</td><td>37.89</td><td>49.49</td><td>67.20</td><td>77.18</td></tr></table>

Table 4: Evaluation results on Llama2-70B.

<table><tr><td>Method</td><td>Factor</td><td>GSM8K</td></tr><tr><td>Llama2-70B</td><td>61.92</td><td>56.10</td></tr><tr><td>+DoLa-Low</td><td>74.05</td><td>57.01</td></tr><tr><td>+DoLa-High</td><td>62.53</td><td>38.21</td></tr><tr><td>+SLED</td><td>77.32</td><td>57.01</td></tr><tr><td>+NDAD</td><td>77.52</td><td>57.54</td></tr></table>

Table 5: Runtime and memory overhead on Llama2-7B-base. 

<table><tr><td>Method</td><td>Runtime (s)</td><td>Memory (MB)</td></tr><tr><td>Greedy</td><td>1.11</td><td>13503.47</td></tr><tr><td>DoLa</td><td>1.17</td><td>15261.98</td></tr><tr><td>SLED</td><td>1.17</td><td>15452.88</td></tr><tr><td>NDAD</td><td>1.34</td><td>17779.01</td></tr></table>

# 4.4 EVALUATION ON LARGER-SCALE LLM

To assess the viability of the method on substantially larger models, we conducted additional experiments using Llama2-70B on the Factor dataset for multiple-choice tasks and GSM8K for chain-of-thought reasoning. The results, presented in Table 4, show that the method continues to deliver strong performance on generative tasks such as GSM8K. The second-best baseline improves by 0.91%, whereas our method achieves an improvement of 1.44%, corresponding to a relative gain of 58%. For the Factor dataset, as discussed in Section 4.2, this task essentially evaluates distribution fitting, where maintaining a smooth upward trend is sufficient. These results demonstrate that the method remains effective when scaled to much larger models and exhibits strong robustness across different model sizes.

# 4.5 ABLATION STUDY

Incorporation of Hallucination Signal. We first demonstrate that our method indeed introduces hallucination signal into the model. To this end, we directly decode the logits obtained after masking the importance attention heads and evaluate their performance. The experimental results are shown in Figure 3. As can be observed, compared with the original decoding, performance consistently drops across different models and datasets, with the most significant decline occurring on the GSM8K dataset. This indicates that complex reasoning tasks heavily rely on the aggregation and inference of internal attention heads, and masking these heads introduces stronger hallucination signal. This observation is consistent with the analysis in Section 4.3, where our NDAD method achieves better results on GSM8K, suggesting that stronger hallucination signal can provide more effective leverage for enhancing the NDAD decoding strategy. Moreover, the ablation experiments in Table 6 based on random head and layer selection further support that hallucination induction guided by head importance and layer-level entropy contributes to the performance gains of NDAD.

![](images/923488a25dab6edadf5f9438860ee59f2abbbf268f17a1a94d7cd7888d3daf12.jpg)

<details>
<summary>bar</summary>

| Category | Method | Original (%) | Mask (%) |
| :--- | :--- | :--- | :--- |
| Llama2-7b-base | Factor | 60 | 30 |
| Llama2-7b-base | StrQA | 60 | 60 |
| Llama2-7b-base | GSM8K | 15 | 5 |
| Llama2-7b-base | TriviaQA | 65 | 55 |
| Llama2-7b-chat | Factor | 55 | 45 |
| Llama2-7b-chat | StrQA | 65 | 60 |
| Llama2-7b-chat | GSM8K | 20 | 5 |
| Llama2-7b-chat | TriviaQA | 60 | 45 |
| Llama2-13b-base | Factor | 65 | 25 |
| Llama2-13b-base | StrQA | 65 | 45 |
| Llama2-13b-base | GSM8K | 30 | 5 |
| Llama2-13b-base | TriviaQA | 65 | 10 |
| Llama2-13b-chat | Factor | 60 | 40 |
| Llama2-13b-chat | StrQA | 70 | 50 |
| Llama2-13b-chat | GSM8K | 35 | 5 |
| Llama2-13b-chat | TriviaQA | 65 | 15 |
</details>

0.00 Figure 3: Results from Decoding Hallucination Signals.

![](images/4ab6d80e99e65565dd63b01267703bb2ea45e1c4e6f86919c39858d402af74e8.jpg)

<details>
<summary>line</summary>

| Number of masked heads | Correct Rate (Factor) | Correct Rate (GSM8K) |
| --- | --- | --- |
| 6 | 67.12 | 14.89 |
| 7 | 67.26 | 14.89 |
| 8 | 67.31 | 14.74 |
| 9 | 67.22 | 14.30 |
| 10 | 67.22 | 14.30 |
| 11 | 67.22 | 14.45 |
| 12 | 67.22 | 14.45 |
</details>

(a) Different number of masked heads

![](images/1e1acf09d19a84b4aab5a88e54702c4177ef989eb669990b57d57ca1c5d42a91.jpg)

<details>
<summary>line</summary>

| Number of masked layers | Factor Correct Rate | GSM8K Correct Rate |
| --- | --- | --- |
| 7 | 67.15 | 14.46 |
| 8 | 67.15 | 14.67 |
| 9 | 66.98 | 14.57 |
| 10 | 67.23 | 14.67 |
| 11 | 67.31 | 14.88 |
| 12 | 67.31 | 14.57 |
</details>

(b) Different number of masked layers   
Figure 4: Different head and layer parameters on the Llama-7B-base.

Importance of Head and Layer Parameters. To effectively introduce hallucination signal, it is necessary to mask more important attention heads. Using the Llama-7B-base model as an example, we present results on the FACTOR and GSM8K datasets under different parameter settings. Figure 4a illustrates the impact on accuracy when varying the number of masked heads while keeping the number of masked layers fixed. Conversely, Figure 4b shows the effect of varying the number of masked layers while fixing the number of masked heads. Overall, the trend generally follows a rising-then-falling pattern. Notably, throughout the experiments, the range of masked heads and layers remained between [6, 13], within which the model consistently achieved relatively strong performance across both datasets. More detailed results are provided in Appendix B.3.

Global and Local Weights. We further analyze the effectiveness of the global and local weighting components in our method. The ablation results based on Llama2-7B-base and Llama2-13B-base are reported in Table 6, and the more comprehensive results and analyses can be found in Appendix B.2. Specifically, w/o global weight indicates removing the measurement of directional consistency between hallucination signal and the original signals, while w/o local weight corresponds to excluding the measurement of consistency between the tail-token evolution and the transition from the premature to the mature state. From the results, it is clear that both weighting mechanisms play a crucial role in enhancing the decoding performance. For example, in the case of Llama2-7B-base, removing either global or local weights leads to a drop in performance. A similar trend is observed for Llama2- 13B-base, where the absence of these weights consistently reduces accuracy across all benchmarks. Importantly, the GSM8K dataset again shows the largest degradation, underscoring that complex reasoning tasks are particularly sensitive to the loss of these weighting mechanisms. These results confirm that both global and local weights contribute complementary benefits, and together they enable NDAD to achieve robust and state-of-the-art performance.

# 4.6 COMPUTATIONAL OVERHEAD ANALYSIS

To evaluate the computational overhead of our method, we measured runtime and memory usage on the Llama2-7B-base model using a single GSM8K sample, and the results are presented in Table 5. As shown, the additional cost introduced by NDAD is relatively lightweight, with the primary overhead arising from the incorporation of the negative-direction signal. Consistent with existing decoding-based approaches, NDAD only modifies the logits of the final layer, requires no additional training, and does not depend on high-quality external data, giving it strong plug-and-play capability. In many real-world applications, safety and factual reliability are often more critical than achieving the absolute lowest decoding latency; thus, a moderate amount of runtime and memory overhead is generally acceptable.

Table 6: Ablation study on the effectiveness of each component in the NDAD method. 

<table><tr><td rowspan="2">Method</td><td colspan="4">TruthfluQA(MC)</td><td>Factor</td><td colspan="2">CoT</td></tr><tr><td>MC1</td><td>MC2</td><td>MC3</td><td>Avg.</td><td>Wiki</td><td>StrQA</td><td>GSM8K</td></tr><tr><td>Llama2-7B-base</td><td>26.58</td><td>41.88</td><td>18.96</td><td>29.14</td><td>58.42</td><td>60.74</td><td>13.95</td></tr><tr><td>random head</td><td>34.15</td><td>62.55</td><td>31.91</td><td>42.87</td><td>67.17</td><td>61.13</td><td>13.95</td></tr><tr><td>random layer</td><td>34.15</td><td>62.61</td><td>31.84</td><td>42.87</td><td>67.10</td><td>61.40</td><td>14.71</td></tr><tr><td>w/o global weight</td><td>34.27</td><td>62.57</td><td>31.93</td><td>42.92</td><td>67.20</td><td>61.09</td><td>14.63</td></tr><tr><td>w/o local weight</td><td>33.90</td><td>61.13</td><td>31.43</td><td>42.15</td><td>67.17</td><td>61.44</td><td>14.10</td></tr><tr><td>NDAD</td><td>34.39</td><td>62.62</td><td>31.98</td><td>43.00</td><td>67.30</td><td>61.57</td><td>14.86</td></tr><tr><td>Llama2-13B-base</td><td>27.59</td><td>43.14</td><td>19.53</td><td>30.09</td><td>63.79</td><td>65.98</td><td>28.81</td></tr><tr><td>random head</td><td>34.88</td><td>63.58</td><td>31.94</td><td>43.47</td><td>71.04</td><td>66.72</td><td>28.13</td></tr><tr><td>random layer</td><td>34.76</td><td>63.56</td><td>31.91</td><td>43.41</td><td>71.01</td><td>66.72</td><td>28.66</td></tr><tr><td>w/o global weight</td><td>34.88</td><td>63.59</td><td>31.93</td><td>43.47</td><td>70.98</td><td>65.41</td><td>28.73</td></tr><tr><td>w/o local weight</td><td>34.76</td><td>63.57</td><td>31.89</td><td>43.41</td><td>70.91</td><td>66.07</td><td>27.98</td></tr><tr><td>NDAD</td><td>34.88</td><td>63.60</td><td>31.97</td><td>43.48</td><td>71.18</td><td>66.81</td><td>29.26</td></tr></table>

# 5 CONCLUSION

We present an innovative decoding strategy NDAD, which explicitly elicits hallucination signal by masking critical attention heads and leverages them as negative directions for contrastive decoding. To controllably leverage these signals, we design a dynamic weighting mechanism: the global weight measures the directional consistency between the hallucination signal and the original earlylayer logits, thereby quantifying the referential value of the current hallucination signal; the local weight characterizes the tendency of low-probability tokens to evolve toward the mature distribution. By suppressing the output probabilities of hallucination-prone tokens through gradient-descent adjustments during decoding, NDAD consistently improves factual reliability across diverse models and benchmarks, demonstrating particularly strong robustness in complex reasoning tasks. In conclusion, NDAD provides a lightweight yet effective solution for optimizing LLM decoding.

# ACKNOWLEDGMENTS

This work was supported by the Guizhou Provincial Program on Commercialization of Scientific and Technological Achievements (Qiankehezhongyindi [2025] No. 006) and Ant Group.

# ETHICAL STATEMENT

This paper presents a decoding strategy designed to improve the factual reliability of LLMs. Our research does not involve human subjects, sensitive personal data, or potentially harmful datasets. All benchmark datasets employed in our experiments are publicly available and widely used within the Natural Language Processing research community.

# REPRODUCIBILITY STATEMENT

To ensure the reproducibility of our experiments, we have provided the source codes in the supplementary materials for review. Upon acceptance of this paper, we will release the codes as open source to enable researchers to replicate and extend our experiments.

# REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.

arXiv preprint arXiv:2303.08774, 2023.   
Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.   
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 17754–17762, 2024a.   
Shiqi Chen, Miao Xiong, Junteng Liu, Zhengxuan Wu, Teng Xiao, Siyang Gao, and Junxian He. In-context sharpness as alerts: An inner representation perspective for hallucination mitigation. arXiv preprint arXiv:2403.01548, 2024b.   
Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, and Rui Yan. Lift yourself up: Retrieval-augmented text generation with self-memory. Advances in Neural Information Processing Systems, 36:43780–43799, 2023.   
Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. arXiv preprint arXiv:2309.03883, 2023.   
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.   
Mingyuan Fan, Chengyu Wang, Cen Chen, Yang Liu, and Jun Huang. On the trustworthiness landscape of state-of-the-art generative models: A survey and outlook. Int. J. Comput. Vis., 133(7): 4317–4348, 2025.   
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pp. 6491–6501, 2024.   
Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Y Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, et al. Rarr: Researching and revising what language models say, using language models. arXiv preprint arXiv:2210.08726, 2022.   
Ariel Gera, Roni Friedman, Ofir Arviv, Chulaka Gunasekara, Benjamin Sznajder, Noam Slonim, and Eyal Shnarch. The benefits of bad advice: Autocontrastive decoding across model layers. arXiv preprint arXiv:2305.01628, 2023.   
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346–361, 2021.   
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.   
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.   
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.   
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. arXiv preprint arXiv:1906.00300, 2019.   
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨ aschel, et al. Retrieval-augmented genera- ¨ tion for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33: 9459–9474, 2020.   
Junyi Li, Jie Chen, Ruiyang Ren, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. The dawn after the dark: An empirical study on factuality hallucination in large language models. arXiv preprint arXiv:2401.03205, 2024.   
Kenneth Li, Oam Patel, Fernanda Viegas, Hanspeter Pfister, and Martin Wattenberg. Inference-time´ intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530, 2023a.   
Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. arXiv preprint arXiv:2210.15097, 2022.   
Xiaonan Li, Changtai Zhu, Linyang Li, Zhangyue Yin, Tianxiang Sun, and Xipeng Qiu. Llatrieval: Llm-verified retrieval for verifiable generation. arXiv preprint arXiv:2311.07838, 2023b.   
Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958, 2021.   
Mateo Mahaut, Laura Aina, Paula Czarnowska, Momchil Hardalov, Thomas M´ uller, and Llu¨ ´ıs Marquez. Factual confidence of llms: on reliability and robustness of current estimators. \` arXiv preprint arXiv:2406.13415, 2024.   
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. arXiv preprint arXiv:2212.10511, 2022.   
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. arXiv preprint arXiv:2305.14251, 2023.   
Abhika Mishra, Akari Asai, Vidhisha Balachandran, Yizhong Wang, Graham Neubig, Yulia Tsvetkov, and Hannaneh Hajishirzi. Fine-grained hallucination detection and editing for language models. arXiv preprint arXiv:2401.06855, 2024.   
Dor Muhlgay, Ori Ram, Inbal Magar, Yoav Levine, Nir Ratner, Yonatan Belinkov, Omri Abend, Kevin Leyton-Brown, Amnon Shashua, and Yoav Shoham. Generating benchmarks for factuality evaluation of language models. arXiv preprint arXiv:2307.06908, 2023.   
Rafael Muller, Simon Kornblith, and Geoffrey E Hinton. When does label smoothing help? ¨ Advances in neural information processing systems, 32, 2019.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.   
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.   
William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. Self-critiquing models for assisting human evaluators, 2022. URL https://arxiv. org/abs/2206.05802, 2023.

Chufan Shi, Haoran Yang, Deng Cai, Zhisong Zhang, Yifan Wang, Yujiu Yang, and Wai Lam. A thorough examination of decoding methods in the era of llms. arXiv preprint arXiv:2402.06925, 2024.   
Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2, 2024.   
Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher Manning, and Chelsea Finn. Fine-tuning language models for factuality. In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following, 2023.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and \` efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.   
Chenguang Wang, Xiao Liu, and Dawn Song. Language models are open knowledge graphs. arXiv preprint arXiv:2010.11967, 2020.   
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.   
Sean Welleck, Amanda Bertsch, Matthew Finlayson, Hailey Schoelkopf, Alex Xie, Graham Neubig, Ilia Kulikov, and Zaid Harchaoui. From decoding to meta-generation: Inference-time algorithms for large language models. arXiv preprint arXiv:2406.16838, 2024.   
Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. arXiv preprint arXiv:2404.15574, 2024.   
Chang-Bin Zhang, Peng-Tao Jiang, Qibin Hou, Yunchao Wei, Qi Han, Zhen Li, and Ming-Ming Cheng. Delving deep into label smoothing. IEEE Transactions on Image Processing, 30:5984– 5996, 2021.   
Jianyi Zhang, Da-Cheng Juan, Cyrus Rashtchian, Chun-Sung Ferng, Heinrich Jiang, and Yiran Chen. Sled: Self logits evolution decoding for improving factuality in large language models. Advances in Neural Information Processing Systems, 37:5188–5209, 2024.   
Yue Zhang, Leyang Cui, Wei Bi, and Shuming Shi. Alleviating hallucinations of large language models through induced hallucinations. arXiv preprint arXiv:2312.15710, 2023a.   
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. Siren’s song in the ai ocean: a survey on hallucination in large language models. arXiv preprint arXiv:2309.01219, 2023b.

# A EXPERIMENTAL SETTINGS

# A.1 PARAMETER SETTINGS

For the parameters α in Equation 10 and the I correctness distributions in Equation 5, we set the default values to α = 2 and I = 10. However, due to dataset uncertainty, additional hyperparameter tuning may be required in special cases. Following the work of (Zhang et al., 2024), we test α from {0.01, 0.1, 1, 2, 5, 10} and I from {5, 10, 20, 50}. During the aforementioned tests, we guarantee that the chosen parameters achieve performance better than greedy decoding. On this basis, we then incorporate our hallucination signal to conduct adaptive negative-direction aware decoding. For the number of masked heads and layers used in introducing hallucination signal, we partly explained this in Section 4.5. In experiments, we usually set the range to [6, 13], which generally yields strong performance.

![](images/ef0dc734a579b4e753ee3441053678326eb4f805ff94fecf62eaad300b901f77.jpg)

<details>
<summary>line</summary>

| Number of masked heads | Correct Rate |
| ---------------------- | ------------ |
| 7                      | 65.07        |
| 8                      | 65.11        |
| 9                      | 65.11        |
| 10                     | 65.07        |
| 11                     | 64.99        |
| 12                     | 64.95        |
</details>

(a) Different number of masked heads

![](images/5306f94661a284d90715978bee161a2c61fd04e631478b5898c2eb13abaa09a9.jpg)

<details>
<summary>line</summary>

| Number of masked heads | Correct Rate |
| ---------------------- | ------------ |
| 7                      | 21.78        |
| 8                      | 22.03        |
| 9                      | 21.57        |
| 10                     | 21.11        |
| 11                     | 21.11        |
| 12                     | 21.13        |
</details>

![](images/9bf0cbf9c704435c53249c6751d64464e8618f050a9aa09f73074fb0cc311549.jpg)

<details>
<summary>line</summary>

| Number of masked layers | Correct Rate |
| ----------------------- | ------------ |
| 7                       | 65.04        |
| 8                       | 65.01        |
| 9                       | 64.95        |
| 10                      | 65.04        |
| 11                      | 65.06        |
| 12                      | 65.03        |
</details>

(b) Different number of masked layers

![](images/c0f302b85ca7f7b80d43488965b94d01e27ac5ec80f4de5a32c1e2d7103e7f09.jpg)

<details>
<summary>line</summary>

| Number of masked layers | Correct Rate |
| ----------------------- | ------------ |
| 8                       | 21.70        |
| 9                       | 21.59        |
| 10                      | 21.78        |
| 11                      | 21.68        |
| 12                      | 22.01        |
| 13                      | 21.59        |
</details>

Figure 5: Different head and layer parameters on the Llama-7B-chat.

# B ADDITIONAL EXPERIMENTAL RESULTS

# B.1 EXTENDED ANALYSIS OF MULTIPLE-CHOICES TASKS

As shown in Table 1, the performance improvements of NDAD on multiple-choice tasks are slightly smaller compared to other task types. This is consistent with the discussion in Section 4.2, where multiple-choice problems essentially reduce to a logits-fitting task; as long as the model achieves stable performance without large fluctuations and delivers moderate gains, the results remain reasonable. Moreover, since the multiple-choice format inherently constrains the output space with a fixed set of candidate answers, the likelihood of hallucination is substantially reduced, leading to weaker hallucination signals and thus smaller benefits from NDAD’s decoding adjustments. Nevertheless, our primary focus is on more complex open-ended generation tasks, where hallucinations are far more prevalent and where NDAD demonstrates clear advantages in suppressing hallucination-prone directions and enhancing factual reliability.

# B.2 EXTENDED ABLATION ANALYSIS

We further conducted ablation experiments on Llama2-7B-chat and Llama2-13B-chat to examine the effect of different components in NDAD, with the experimental setup summarized in Table 7.

Hallucination Signal Induction. During the stage of hallucination signal induction, we observed that the random selection of attention heads or layers occasionally outperformed our guided masking strategy based on head importance and layer-level entropy. This can be attributed to the inherently greedy nature of the masking strategy: although generally effective, it does not fully explore the extensive search space. Consequently, certain random configurations may fortuitously yield superior outcomes. Nonetheless, such instances are expected and do not diminish the overall effectiveness of a principled importance-guided approach.

Global Weighting in Multiple-Choice Tasks. For the global weighting component, the performance on Llama2-7B-chat with the TruthfulQA dataset was slightly better when the global weighting was not applied compared to the full NDAD method. As discussed in Section B.1, these multiple-choice tasks essentially reduce to a logits-fitting problem with a small set of candidate answers. Since all options are inherently more reliable than open-ended generations, the model is less vulnerable to noisy hallucinations in this setting. Consequently, assessing the reliability of hallucination signals becomes less critical, and the global weighting may even introduce unnecessary adjustments that interfere with straightforward logits alignment. By contrast, in open-ended generation tasks, where hallucination is more prevalent, the global and local weighting strategies play a much more important role in enhancing factual reliability.

# B.3 EXTENDED PARAMETER ANALYSIS

We further conducted hyperparameter experiments on Llama2-7B-chat. As shown in Figure 5, for both the number of masked attention heads and the number of masked layers, performance exhibits a general rising-then-falling trend: as the number of masked heads or layers increases, performance initially improves but declines once the masking becomes excessive. The results suggest that the optimal settings typically lie within the range of 6 to 13, where a better balance is achieved between inducing hallucination signals and preserving the original representations.

Table 7: Additional ablation study on the effectiveness of each component in the NDAD method. 

<table><tr><td rowspan="2">Method</td><td colspan="4">TruthfluQA(MC)</td><td>Factor</td><td colspan="2">CoT</td></tr><tr><td>MC1</td><td>MC2</td><td>MC3</td><td>Avg.</td><td>Wiki</td><td>StrQA</td><td>GSM8K</td></tr><tr><td>Llama2-7B-chat</td><td>35.62</td><td>57.47</td><td>32.10</td><td>41.73</td><td>56.68</td><td>63.58</td><td>21.23</td></tr><tr><td>random head</td><td>36.84</td><td>63.38</td><td>32.65</td><td>44.29</td><td>64.93</td><td>64.72</td><td>21.15</td></tr><tr><td>random layer</td><td>36.47</td><td>62.99</td><td>32.59</td><td>44.02</td><td>65.00</td><td>64.72</td><td>20.62</td></tr><tr><td>w/o global weight</td><td>36.84</td><td>63.71</td><td>32.80</td><td>44.45</td><td>64.93</td><td>64.37</td><td>21.00</td></tr><tr><td>w/o local weight</td><td>36.47</td><td>60.82</td><td>32.40</td><td>43.23</td><td>64.96</td><td>63.58</td><td>20.62</td></tr><tr><td>NDAD</td><td>36.84</td><td>63.27</td><td>32.76</td><td>44.29</td><td>65.06</td><td>64.67</td><td>21.99</td></tr><tr><td>Llama2-13B-chat</td><td>36.47</td><td>63.06</td><td>32.77</td><td>44.10</td><td>61.96</td><td>69.65</td><td>36.69</td></tr><tr><td>random head</td><td>37.45</td><td>63.61</td><td>32.95</td><td>44.67</td><td>67.47</td><td>69.91</td><td>35.63</td></tr><tr><td>random layer</td><td>37.70</td><td>63.58</td><td>33.07</td><td>44.78</td><td>67.57</td><td>69.52</td><td>35.78</td></tr><tr><td>w/o global weight</td><td>35.62</td><td>63.91</td><td>32.49</td><td>44.01</td><td>67.60</td><td>69.43</td><td>35.71</td></tr><tr><td>w/o local weight</td><td>37.21</td><td>64.02</td><td>32.90</td><td>44.71</td><td>67.67</td><td>69.65</td><td>37.00</td></tr><tr><td>NDAD</td><td>37.58</td><td>63.63</td><td>33.02</td><td>44.74</td><td>67.74</td><td>69.96</td><td>37.30</td></tr></table>

# B.4 EXTENDED LINGUISTIC QUALITY EVALUATION

To assess whether NDAD introduces any degradation in linguistic quality, we conduct an additional evaluation focusing on fluency, coherence, and comprehensibility. These dimensions reflect whether the generated responses remain natural, logically organized, and easy to understand—qualities that are essential for real-world deployment but are often overlooked in factuality-oriented methods. We generate model outputs using Llama2-70B on GSM8K and obtain linguistic quality scores from the external evaluator Gemini-2.5-Pro. The results are presented in Table 8. As shown, the scores across all methods are highly consistent, and NDAD performs on par with or slightly better than existing decoding strategies, indicating that NDAD does not introduce noticeable negative effects on linguistic quality. This evaluation further demonstrates that NDAD improves factuality while preserving the naturalness and readability of generated text. Table 9 is the full evaluation prompt used for scoring with the Gemini model.

Table 8: Linguistic quality evaluation of different decoding methods using Gemini-2.5-Pro. 

<table><tr><td>Method</td><td>Fluency</td><td>Coherence</td><td>Comprehensibility</td></tr><tr><td>Greedy</td><td>9.37</td><td>7.96</td><td>8.65</td></tr><tr><td>DoLa</td><td>9.29</td><td>7.91</td><td>8.58</td></tr><tr><td>SLED</td><td>9.32</td><td>8.02</td><td>8.69</td></tr><tr><td>NDAD</td><td>9.31</td><td>8.04</td><td>8.67</td></tr></table>

# C ALGORITHM OF NDAD

The entire algorithmic workflow of the NDAD method is presented in Algorithm 1 and 2.

# D CASE STUDY

Table 10 reports the results of the Llama-7B-Base model on the GSM8K dataset under different decoding strategies. The examples demonstrate that our NDAD method is more effective in eliciting factual outputs from the model.

Table 9: Prompt for Gemini-2.5-Pro. 

<table><tr><td>You are an advanced artificial intelligence review system specialized in evaluating the quality of model responses. Your task is to rate the quality from three perspectives: fluency, coherence, and comprehensibility. Please strictly follow the evaluation dimensions below to score each item (range: 0--10, with higher scores indicating better quality).</td></tr><tr><td>[Evaluation Criteria]</td></tr><tr><td>Fluency: Whether the sentence structure of the answer is clear and natural, with no obvious grammatical errors, inappropriate word usage, or issues affecting the reading experience. Higher scores indicate smooth language that can be read without difficulty.</td></tr><tr><td>Coherence: Whether the logical connections between parts of the answer are tight and information flows smoothly. Check for jumps, breaks, contradictions, or repetition that affect logical coherence. Higher scores indicate clear thinking and reasonable structure.</td></tr><tr><td>Comprehensibility: Whether the answer is easy for the target reader to understand. Higher scores indicate clear information delivery, easy understanding, and no ambiguity or obscure expressions.</td></tr><tr><td>[Output Format]</td></tr><tr><td>Please output in the following JSON format:</td></tr><tr><td>{</td></tr><tr><td>&quot;Scores for Each Dimension&quot;: {</td></tr><tr><td>&quot;Fluency&quot;: score,</td></tr><tr><td>&quot;Coherence&quot;: score,</td></tr><tr><td>&quot;Comprehensibility&quot;: score</td></tr><tr><td>},</td></tr><tr><td>&quot;Reason for Scoring&quot;: Explain the reasons for scoring each dimension, and briefly summarize the overall evaluation</td></tr><tr><td>}</td></tr><tr><td>Please validate the question and return the result in JSON format, with no other content except the JSON.</td></tr></table>

Algorithm 1 Hallucination Signal Induction   
1: LLM with $L$ layers, sequence, following the work of (Wu et al., 2024), a original score list of $n$ attention head $\{s_{l,1}, s_{l,2}, ..., s_{l,n}\}$ in layer $l$ , number of masked attention heads $x$ , number of masked layer $K$ .
2: for $l < L$ do
3:    Normalize scores into probability distribution: $p_{l,i} = \frac{s_{l,i}}{\sum_{j=1}^{n} s_{l,j}}, \quad i = 1, \ldots, n$ .
4:    Compute attention head scores distribution entropy: $E_l = -\sum_{i=1}^{n} p_{l,i} \log p_{l,i}$ .
5: end for
6: Obtain the set of distribution entropy $\{E_1, E_2, ..., E_L\}$ .
7: Select the set $\mathcal{L}$ consisting of the $K$ layers $l$ corresponding to the lowest entropy values.
8: for $l \in \mathcal{L}$ do
9:    Set the weights of the top- $x$ scoring attention heads to 0.
10: end for
11: The sequence into the LLM to obtain the hallucination signals logits $_l^{\text{mask}}$ , where $l \leq L$ .
12: Return: {logits $_1^{\text{mask}}$ , logits $_2^{\text{mask}}$ , ..., logits $_L^{\text{mask}}$ }

Algorithm 2 Negative-Direction Aware Decoding   
1: Initialization: LLM with $L$ layers, sequence, $\alpha$ in Equation 10, number of correctness directions $I, \epsilon \to 0$ , $\varphi(\cdot)$ maps values into $[0,1]$ , the one-hot vectors $\mathcal{T} = \{\mathcal{P}_{e_1}, \mathcal{P}_{e_2}, ..., \mathcal{P}_{e_I}\}$ of correctness directions.
2: The sequence into the LLM to obtain the original logits logits $_l$ and hallucination signal logits $_l^{\text{mask}}$ given by Algorithm 1, the probabilities at each layer $l$ denoted as $\mathcal{P}_{\text{logits}_l} = \text{softmax}(\text{logits}_l)$ and $\mathcal{P}_{\text{logits}_l^{\text{mask}}} = \text{softmax}(\text{logits}_l^{\text{mask}})$ , where $l \leq L$ .
3: Identify the tokens with the top- $I$ largest probabilities in $\mathcal{P}_{\text{logits}_L}$ and assign the value 1 to their indices and 0 to the remaining positions.
4: Set the indices of top- $I$ largest probabilities tokens in $\mathcal{P}_{\text{logits}_l^{\text{mask}}} \text{ to } \epsilon: \mathcal{P}_{\text{logits}_l^{\text{mask}}} \to \mathcal{P}_{\text{logits}_l^{\text{mask}}}^{\text{tail}}$ .
5: for $l < L$ do
6: Compute $\mathcal{W}_l^{\text{global}} = \varphi\left(\cos\_sim\left(\text{logits}_l, \text{logits}_l^{\text{mask}}\right)\right)$ .
7: Compute $\mathcal{W}_{l,i}^{\text{local}} = \max\left(\cos\_sim\left(\text{logits}_l^{\text{mask}} - \text{logits}_L, \mathcal{P}_{\text{logits}_l^{\text{mask}}}^{\text{tail}} - \mathcal{P}_{e_i}\right), 0\right), \mathcal{P}_{e_i} \in \mathcal{T}$ .
8: Calculate $\tilde{\mathcal{W}}_{l,i} = \left(\mathcal{W}_l^{\text{global}} \mathcal{W}_{l,i}^{\text{local}}\right)^2, i \in [1, I]$ .
9: end for
10: Obtain the current latent distribution $\mathcal{P}_{\text{NDAD}} = \frac{\sum_{l=1}^{L} \tilde{\mathcal{W}}_{l,i}}{\sum_{l=1}^{L} \sum_{j=1}^{|I|} \tilde{\mathcal{W}}_{l,j}}$ by computing each $i \in [1, I]$ across different layers.
11: Return: logits $_L^{\text{new}} = \text{logits}_L - \alpha \nabla KL(\mathcal{P}_{\text{NDAD}}, \mathcal{P}_{\text{logits}_L})$

Table 10: Case study of Llama-7B-base on the GSM8K Dataset. 

<table><tr><td>Input:</td><td>Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are originally 3 cars. 2 more cars arrive.  $3 + 2 = 5$ . The answer is 5.Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had  $32 + 42 = 74$ . After eating 35, they had  $74 - 35 = 39$ . The answer is 39.Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny  $20 - 12 = 8$ . The answer is 8.Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys.  $5 + 4 = 9$ . The answer is 9.Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So  $5 * 4 = 20$  computers were added.  $9 + 20$  is 29. The answer is 29.Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael started with 58 golf balls. After losing 23 on tuesday, he had  $58 - 23 = 35$ . After losing 2 more, he had  $35 - 2 = 33$  golf balls. The answer is 33.Q: Olivia has 23.Sheboughtfivebagelsfor3 each. How much money does she have left? A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be  $5 \times 3 = 15$  dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.Q: Marcell and Beatrice are having a contest to see who can eat the most fruit roll-ups, so they unroll as many as they can find. Unfortunately, someone makes a mistake and Beatrice&#x27;s was two roll-ups wide and 24 rolls up long while Marcell&#x27;s was 3 roll-ups wide and 14 roll-ups long. If they both ate their entire amount, how many did they eat on average?A:</td></tr><tr><td>Dreedy</td><td>Marcell&#x27;s was 3 wide and 14 long. Beatrice&#x27;s was 2 wide and 24 long. So Marcell&#x27;s was  $3 * 14 = 42$  rolls. Beatrice&#x27;s was  $2 * 24 = 48$  rolls. So Marcell ate 42 rolls and Beatrice ate 48 rolls.  $42 + 48 = 90$ . The answer is 90.</td></tr><tr><td>SLED</td><td>Marcell&#x27;s was 3 wide and 14 long. Beatrice&#x27;s was 2 wide and 24 long. So Marcell&#x27;s was  $3 * 14 = 42$  rolls. Beatrice&#x27;s was  $2 * 24 = 48$  rolls. So Marcell ate 42 rolls and Beatrice ate 48 rolls.  $42 + 48 = 90$  rolls. The answer is 90.</td></tr><tr><td>NDAD</td><td>Marcell&#x27;s was 3 roll-ups wide and 14 roll-ups long. Beatrice&#x27;s was 2 roll-ups wide and 24 roll-ups long. So Marcell&#x27;s was  $3 * 14 = 42$  roll-ups long. Beatrice&#x27;s was  $2 * 24 = 48$  roll-ups long. So Marcell&#x27;s was 42 roll-ups long and Beatrice&#x27;s was 48 roll-ups long. So they both ate  $42 + 48 = 90$  roll-ups.  $90 / 2 = 45$ . The answer is 45.</td></tr></table>