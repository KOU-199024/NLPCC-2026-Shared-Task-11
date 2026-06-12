# MEMAG NIT: RESHAPING LONG-CONTEXT LLM WITH MULTI-CONV RL-BASED MEMORY AGENT

Hongli Yu1,2,3 Tinghong Chen1 Jiangtao Feng1 Jiangjie Chen2,3 Weinan Dai1,2,3

Qiying Yu1,2,3 Ya-Qin Zhang1,3 Wei-Ying Ma1,3 Jingjing Liu1,3 Mingxuan Wang2,3∗

# Hao Zhou1,3∗

1 Institute for AI Industry Research (AIR), Tsinghua University   
2 ByteDance Seed   
3 SIA-Lab of Tsinghua AIR and ByteDance Seed

# ABSTRACT

Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents without performance degradation during extrapolation remains the ultimate challenge in long-text processing. To solve this problem, we introduce a novel agent workflow, MEMAGENT, which processes text in segments and updates memory through an overwrite strategy, addressing the challenge of long-context task through enhanced memory management. We further extend the DAPO algorithm to directly optimize memory ability in an end-to-end fashion, facilitating training via independent-context multi-conversation generation. Experimental results demonstrate that MEMAGENT has superb long-context capabilities, being able to extrapolate from an 8K context to a 3.5M QA task with a performance loss of less than 10% and achieving over 95% on the 512K NIAH test.

![](images/23cb86bfeed34ffddf837aab45370deefc7db553dbcc7b12c328af85335db4d2.jpg)

<details>
<summary>line</summary>

| Context Length in Tokens | RL-Memagent-14B | RL-Memagent-7B | QwenLong-L1-32B | Qwen2.5-Instruct-14B-1M | Qwen2.5-Instruct-7B-1M | DS-Distill-Qwen-32B | DS-Distill-Qwen-14B | DS-Distill-Qwen-7B | Truncation |
| ------------------------ | --------------- | -------------- | --------------- | ------------------------ | ----------------------- | ------------------- | ------------------- | ------------------ | ---------- |
| 7K                       | 80              | 80             | 75              | 60                       | 60                      | 65                  | 65                  | 30                 | 30         |
| 112K                     | 80              | 80             | 70              | 50                       | 50                      | 50                  | 50                  | 10                 | 10         |
| 224K                     | 80              | 80             | 60              | 40                       | 40                      | 40                  | 40                  | 5                  | 5          |
| 448K                     | 80              | 80             | 50              | 30                       | 30                      | 30                  | 30                  | 5                  | 5          |
| 896K                     | 80              | 80             | 40              | 20                       | 20                      | 20                  | 20                  | 5                  | 5          |
| 1.75M                    | 80              | 80             | 30              | 10                       | 10                      | 10                  | 10                  | 5                  | 5          |
| 3.5M                     | 75              | 75             | 25              | 5                        | 5                       | 5                   | 5                   | 5                  | 5          |
</details>

Figure 1: Accuracy scores of RULER-HQA (Hsieh et al., 2024; Yang et al., 2018) . Even models that employ long-context continual pretraining and extrapolation techniques fail to maintain consistent performance. In contrast, MEMAGENT with RL only demonstrates marginal performance dropping.

# 1 INTRODUCTION

While having demonstrated impressive capabilities (OpenAI, 2024; DeepMind, 2024; XAI, 2024; Anthropic, 2024; OpenAI, 2023), industry-level Large Language Model (LLM) systems (Anthropic, 2025; Li et al., 2025a; Liu et al., 2024; Yen et al., 2024) still face a critical challenge: how to handle long contexts effectively - processing an entire book, executing a complex chain of reasoning over many steps, or managing the long-term memory of an agent system - all these complex tasks can generate overflowing text that quickly explodes the typical-size context window of current LLMs.

Existing approaches to long-context tasks are three-pronged. The first involves length extrapolation methods by shifting the positional embeddings in order to extend the context window of the model (Su et al., 2024; bloc97, 2023; Chen et al., 2023; Peng et al., 2023b; An et al., 2024), plus continued pre-training (Liu et al., 2023; Xiong et al., 2023; Gao et al., 2025). Despite promising potential, these methods often suffer from performance degradation and slow processing speed due to O(n2) computational complexity when applied to extremely long text. The second school of methods leverages sparse attention (Beltagy et al., 2020; Zhao et al., 2019; Xiao et al., 2023) and linear attention mechanisms (Child et al., 2019; Katharopoulos et al., 2020) to reduce the complexity of attention for more efficient processing of longer sequences. However, this typically requires training from scratch, with inherent adversities such as linear attention facing difficulties in parallel training or sparse attention depending on human-defined patterns. The last line of inquiry investigates context compression (Jiang et al., 2023; Li et al., 2023; Behrouz et al., 2024; Zhang et al., 2024), which aims to condense information in token-level or external-memory-plugin modules. Such approaches often struggle with extrapolation, and require the integration of additional modules or context operations, which ineluctably disrupts the standard generation process and hinders compatibility as well as parallelization.

Hence, a successful LLM with strong long-context capabilities requires the trinity of: 1) processing infinite length of text; 2) scaling without significant performance drop; and 3) efficient decoding with linear complexity. To pursue this quest, we return to the basic intuition behind long-context modeling (Miller et al., 1956; Hochreiter & Schmidhuber, 1997; Graves et al., 2014; Weston et al., 2014). When humans process long-context information, we tend to abstract out the main revealing conceptions to capture the essence of the whole text, often by making notes of critical details or using short-handed stenograph to record the key points, while discarding redundant and irrelevant data. We do not attempt to memorize every single fact or each small piece of information; instead, we focus our intellectual energy on more important aspects of the task at hand. This selective attention not only simplifies the process but also aids in tackling complex problems more efficiently.

Following this anthropocentric intuition, we propose a novel use of Reinforcement Learning (RL) to equip LLMs with a dynamically updated fixed-length ‘memory’, as illustrated in Figure 2. During inference, the LLM processes the input text segment-by-segment. As it reads each segment, the model proactively and selectively updates the memory, which then contributes to the generation of the final output after all relevant messages are aggregated and synergized in the memory. This clever mechanism allows the LLM to flexibly handle arbitrary text lengths while maintaining a linear time complexity during processing, since the length of the memory is fixed, which leads to a fixed context window size for the model. This segment-based approach generates multiple outputs from a single long-text input, requiring multiple rounds of memory updates and a final round for the generation of the final response. Training this type of agent workflow, which enables dialogues across multiple independent contexts, is still an unexplored territory in current LLM study. Existing systems typically handle workflow trajectories via alternating tool calls or environment feedback by either simply concatenating (Ouyang et al., 2025; Jin et al., 2025) them or using a sliding window (Feng et al., 2025) approach, which lacks flexibility and scalability in practice. Our MEMAGENT approach, instead, proposes that treats each context-independent conversation as an optimization objective. Based on the DAPO (Yu et al., 2025) algorithm, we implement the Multi-Conv DAPO to optimize an arbitrary agent workflow by verifiable outcome reward.

In our experiments, an RL-trained model with a modest 8K context window (with a 1024-token memory and a 5000-token document chunk) trained on 60K length documents exhibits consistently superb capabilities for Question Answering (QA) tasks on documents of up to 3.5 million tokens, without performance drop and with linear computation cost. This demonstratively showcases the efficiency and scalability of our long-context memory approach.

Our major contributions are threefold:

• We introduce a novel approach that enables LLMs to process arbitrarily long inputs within limited context window under linear time complexity during inference, overcoming a significant bottleneck in long-context processing.   
• We design an agent workflow to implement this mechanism and propose an end-to-end training approach using the multi-conversation DAPO algorithm.

![](images/034901bd81cae3a3140810789cfe1548c57556a1f14fdf01e7e9cb5ddc73ccba.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Solving Long-Context Task with Long-Context LLM"] --> B["Long-Context LLM"]
    B --> C["1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ... N-2 N-1 N Q"]
    B --> D["1"]
    B --> E["2"]
    B --> F["3"]
    B --> G["4"]
    B --> H["5"]
    B --> I["6"]
    B --> J["7"]
    B --> K["8"]
    B --> L["9"]
    B --> M["10"]
    B --> N["11"]
    B --> O["12"]
    B --> P["13"]
    B --> Q["14"]
    B --> R["15"]
    B --> S["16"]
    B --> T["..."]
    B --> U["17 ... N-2 N-1 N Q"]
    B --> V["..."]
    B --> W["1"]
    B --> X["2"]
    B --> Y["3"]
    B --> Z["4"]
    B --> AA["5"]
    B --> AB["6"]
    B --> AC["7"]
    B --> AD["8"]
    B --> AE["9"]
    B --> AF["10"]
    B --> AG["11"]
    B --> AH["12"]
    B --> AI["13"]
    B --> AJ["14"]
    B --> AK["15"]
    B --> AL["16"]
    B --> AM["..."]
    B --> AN["17 ... N-2 N-1 N Q"]
    B --> AO["..."]
    B --> AP["1"]
    B --> AQ["2"]
    B --> AR["3"]
    B --> AS["4"]
    B --> AT["5"]
    B --> AU["6"]
    B --> AV["7"]
    B --> AW["8"]
    B --> AX["9"]
    B --> AY["10"]
    B --> AZ["11"]
    B --> BA["12"]
    B --> BB["13"]
    B --> BC["14"]
    B --> BD["15"]
    B --> BE["..."]
    B --> BF["17 ... N-2 N-1 N Q"]
    B --> BG["..."]
    B --> BH["17 ... N-2 N-1 N Q"]
    B --> BI["..."]
    B --> BJ["17 ... N-2 N-1 N Q"]
    B --> BK["..."]
    B --> BL["17 ... N-2 N-1 N Q"]
    B --> BM["..."]
    B --> BN["17 ... N-2 N-1 N Q"]
    B --> BO["..."]
    B --> BP["17 ... N-2 N-1 N Q"]
    B --> BQ["..."]
    B --> BR["17 ... N-2 N-1 N Q"]
    B --> BS["..."]
    B --> BT["17 ... N-2 N-1 N Q"]
    B --> BU["..."]
    B --> BV["17 ... N-2 N-1 N Q"]
    B --> BW["..."]
```
</details>

Figure 2: MEMAGENT is inspired by the way humans process long documents. It divides the document into multiple chunks and allows LLMs to process them iteratively, recording relevant information in memory. Finally, LLMs generate answers based on the information stored in the memory.

![](images/3b543c5afc10f223d0d8980fb8f3ad6433e7942d2a06718cf0450cbe94477612.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["q"] --> B["Policy Model"]
    B --> C["o1"]
    B --> D["o2"]
    B --> E["..."]
    B --> F["og"]
    C --> G["Reference Model"]
    D --> G
    E --> G
    F --> G
    G --> H["Rule-Based Verifier"]
    H --> I["r1"]
    H --> J["r2"]
    H --> K["..."]
    H --> L["rg"]
    I --> M["Group Normalization"]
    J --> M
    K --> M
    L --> M
    M --> N["A1"]
    M --> O["A2"]
    M --> P["..."]
    M --> Q["AG"]
    R["q"] --> S["Policy Model"]
    S --> T["Group of Conversations"]
    T --> U["o1,1"]
    T --> V["o1,2"]
    T --> W["..."]
    T --> X["o1,c1"]
    T --> Y["o2,1"]
    T --> Z["o2,2"]
    T --> AA["..."]
    T --> AB["o2,c2"]
    T --> AC["..."]
    T --> AD["oG,1"]
    T --> AE["oG,2"]
    T --> AF["..."]
    T --> AG["oG,cG"]
    S --> AH["Reference Model"]
    AH --> AI["r1"]
    AH --> AJ["r2"]
    AH --> AK["..."]
    AH --> AL["rg"]
    AI --> AM["Group Normalization"]
    AM --> AN["A1"]
    AM --> AO["A2"]
    AM --> AP["..."]
    AM --> AQ["AG"]
```
</details>

Figure 3: Comparison between vanilla GRPO and Multi-Conv DAPO. During the rollout phase of Multi-conv DAPO, each sample generates multiple conversations. The answer contained in the final conversation is used to compute the reward and advantage, which are then employed to optimize all preceding conversations.

• We empirically demonstrate that our RL-trained method allows models to extrapolate to vastly long documents with minimal performance degradation, pushing the boundaries of what is currently achievable in long-context LLM systems.

# 2 METHODOLOGY

In this section, we describe the details of MEMAGENT approach for solving long-context tasks, including the overall workflow (§ 2.1), Multi-conv RL algorithm for training MEMAGENT (§ 2.2) and the formal modeling of our architecture(§ 2.3).

# 2.1 THE MEMAGENT WORKFLOW: RL-SHAPED MEMORY FOR UNBOUNDED CONTEXTS

As illustrated in Figure 2, MEMAGENT views an arbitrarily long document not as a monolithic block but as a controlled stream of evidence. At every step, the model sees exactly two things: the next chunk of text and a compact, fixed-length memory that summarizes everything deemed important so far. Crucially, the memory is just a sequence of ordinary tokens inside the context window, so the core generation process of the base LLM remains unchanged.

After reading a new chunk, the model overwrites the previous memory with an updated one. This overwrite strategy seems almost too simple, yet it is precisely what enables the system to scale: because memory length never grows, the total compute per chunk stays O(1) and end-to-end complexity is strictly linear to the number of chunks. We formulate the overwrite decision as a reinforcement learning problem: the agent is rewarded for retaining information that will later prove useful and for discarding distractors that would waste precious tokens. By optimizing this objective with our newly introduced multi-conversation DAPO algorithm (detailed in § 2.2), the model learns to compress aggressively while preserving answer-critical facts.

![](images/ff0d66cade249a9a30754bb31e8fa460361760543412f1e44ffc89a7d81caf36.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["External Input"] --> B["Controller"]
    C["External Output"] --> B
    B --> D["Read Head"]
    B --> E["Write Head"]
    D --> F["Memory"]
    E --> F
    B -->|p(c^k | m^{k-1})| D
    B -->|p(m^k | c^k, m^{k-1})| E
```
</details>

Architecture of MemAgent

![](images/1dd11e409d2e2516709c4343541ac89379ade2f9bc32d989f8b7bf77d712e878.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["External Input"] --> B["c¹"]
    B --> C["c²"]
    C --> D["..."]
    D --> E["c^K"]
    E --> F["c^{K+1}"]
    G["∅"] --> H["m¹"]
    H --> I["m²"]
    I --> J["..."]
    J --> K["m^K"]
    L["t = 0"] --> M["m¹"]
    M --> N["m²"]
    N --> O["..."]
    O --> P["m^K"]
    Q["t = 1"] --> R["m¹"]
    R --> S["m²"]
    S --> T["..."]
    T --> U["m^K"]
    V["t = 2"] --> W["m¹"]
    W --> X["m²"]
    X --> Y["..."]
    Y --> Z["m^K"]
    AA["t = K"] --> AB["m¹"]
    AB --> AC["m²"]
    AC --> AD["..."]
    AD --> AE["m^K"]
    AF["t = K + 1"] --> AG["m¹"]
    AG --> AH["m²"]
    AH --> AI["..."]
    AI --> AJ["m^K"]
    AK["External Output"] --> AL["External Input"]
```
</details>

Graphical Model of MemAgent   
Figure 4: The architecture and graphic model of MEMAGENT. The memory is modeled as a latent memory variable, thereby enabling the decomposition of the autoregressive language model into multiple steps of reading from and writing to the memory.

The workflow naturally decomposes inference into two modules. Within the Context-Processing module the model iterates over chunks, updating memory with a prompt template (Table 5, top). Once the stream is exhausted, a final Answer-Generation module is invoked (Table 5, bottom) where the model consults only the problem statement and the memory to produce its boxed answer. Because positional embeddings are never re-scaled or patched, the same tokenization and attention layout apply in both modules, unlocking the model’s latent length-extrapolation capability without any architectural modifications.

MEMAGENT therefore enjoys three benefits from this design: (1) Unlimited length: the document can be millions of tokens because it is processed as a stream; (2) No performance cliff: RL encourages the memory to retain exactly the information needed, yielding near-lossless extrapolation (Figure 1); (3) Linear cost: a constant window size implies decoding time and memory consumption grow linearly with input length (O(N )) (detailed in § B.) This renders a practical recipe for turning any moderately context-sized LLM into an efficient long-context reasoner with minimal engineering overhead.

# 2.2 TRAINING MEMAGENT WITH MULTI-CONV RL

By viewing memory update in context processing for answer-generation tasks as part of the policy to be optimized by RL, we adopt the RLVR recipe (OpenAI, 2024; Guo et al., 2025; Seed et al., 2025) to train MEMAGENT. We adopt DAPO (Yu et al., 2025), an efficient and effective algorithm refined from Group Relative Policy Optimization (GRPO) (Shao et al., 2024), as our training algorithm. Due to the nature of our MEMAGENT approach, which generates multiple context-independent conversations for a single query as illustrated in Figure 2, we treat each conversation as an independent optimization target. This approach necessitates an extension of the loss computation from the conventional (group, token) structure to a new (group, conversation, token) dimensionality, as shown in Figure 3.

Formally, the policy model $\pi _ { \theta _ { \mathrm { o l d } } }$ samples a group of G individual responses $\{ o _ { i , j } \} _ { i = 1 } ^ { G }$ 1 for an input x. Let $n _ { i }$ denote the number of generated conversations $\left( o _ { i , 1 } , o _ { i , 2 } , . . . , o _ { i , n _ { i } } \right)$ for a given sample $( q _ { i } , a _ { i } ) . \ o _ { i , j }$ further decomposes into token-level outputs $\big ( o _ { i , j , 1 } , o _ { i , j , 2 } , . . . , o _ { i , j , | o _ { i , j } | } \big )$ . The advantage value is derived from the conversation that contains the final answer, then uniformly applied to all conversations originating from the same sample, as shown in Eq 1. Following Dr. GRPO (Liu et al.,

Table 1: Main experimental results comparing model performance across various context lengths. All values represent accuracy (%). 

<table><tr><td rowspan="2">Model</td><td colspan="10">Length</td></tr><tr><td>7K</td><td>14K</td><td>28K</td><td>56K</td><td>112K</td><td>224K</td><td>448K</td><td>896K</td><td>1.75M</td><td>3.5M</td></tr><tr><td>QwenLong-L1-32B</td><td>72.66</td><td>75.00</td><td>72.66</td><td>60.94</td><td>31.25</td><td>17.19</td><td>13.28</td><td>11.72</td><td>N/A</td><td>N/A</td></tr><tr><td>Qwen2.5-Instruct-14B-1M</td><td>60.16</td><td>60.94</td><td>50.00</td><td>57.03</td><td>50.00</td><td>37.50</td><td>8.59</td><td>0.00</td><td>N/A</td><td>N/A</td></tr><tr><td>Qwen2.5-Instruct-7B-1M</td><td>61.72</td><td>56.25</td><td>53.91</td><td>55.47</td><td>51.56</td><td>33.59</td><td>12.50</td><td>0.00</td><td>N/A</td><td>N/A</td></tr><tr><td>DS-Distill-Qwen-32B</td><td>70.31</td><td>66.41</td><td>65.62</td><td>46.88</td><td>23.44</td><td>13.28</td><td>7.81</td><td>7.03</td><td>N/A</td><td>N/A</td></tr><tr><td>DS-Distill-Qwen-14B</td><td>64.06</td><td>64.84</td><td>57.03</td><td>40.62</td><td>14.84</td><td>8.59</td><td>3.12</td><td>6.25</td><td>N/A</td><td>N/A</td></tr><tr><td>DS-Distill-Qwen-7B</td><td>30.47</td><td>12.50</td><td>3.12</td><td>0.00</td><td>0.00</td><td>0.78</td><td>0.00</td><td>0.00</td><td>N/A</td><td>N/A</td></tr><tr><td>Qwen2.5-Instruct-32B</td><td>69.53</td><td>64.84</td><td>60.16</td><td>51.56</td><td>44.53</td><td>21.88</td><td>14.06</td><td>7.03</td><td>N/A</td><td>N/A</td></tr><tr><td>Qwen2.5-Instruct-14B</td><td>75.00</td><td>67.19</td><td>57.03</td><td>54.69</td><td>44.53</td><td>21.88</td><td>10.94</td><td>2.34</td><td>N/A</td><td>N/A</td></tr><tr><td>Qwen2.5-Instruct-7B</td><td>52.34</td><td>57.03</td><td>51.56</td><td>44.53</td><td>32.81</td><td>13.28</td><td>6.25</td><td>1.56</td><td>N/A</td><td>N/A</td></tr><tr><td>RL-MEMAGENT-14B</td><td>80.47</td><td>82.03</td><td>82.03</td><td>83.59</td><td>81.25</td><td>77.34</td><td>79.69</td><td>75.78</td><td>78.91</td><td>71.09</td></tr><tr><td>RL-MEMAGENT-7B</td><td>81.25</td><td>81.25</td><td>82.03</td><td>80.47</td><td>79.69</td><td>75.78</td><td>76.56</td><td>74.22</td><td>77.34</td><td>71.88</td></tr></table>

2025), we do not devide the advantage by the standard deviation. Eq 2 describes our loss function.

$$
\hat {A} _ {i, j, t} = R _ {i} - \text { mean } (\{R _ {i} \} _ {i = 1} ^ {G}) \tag {1}
$$

$$
\mathcal {J} _ {\mathrm{DAPO}} (\theta) = \mathbb {E} _ {(q, a) \sim \mathcal {D}, \{o _ {i, j} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | q, o _ {i, j - 1})}
$$

$$
\left[ \frac {1}{\sum_ {i = 1} ^ {G} \sum_ {j = 1} ^ {n _ {i}} \left| o _ {i , j} \right|} \sum_ {i = 1} ^ {G} \sum_ {j = 1} ^ {n _ {i}} \sum_ {t = 1} ^ {\left| o _ {i, j} \right|} \left(\mathcal {C} _ {i, j, t} - \beta D _ {\mathrm{KL}} (\pi_ {\theta} | | \pi_ {\mathrm{ref}})\right) \right] \tag {2}
$$

$\begin{array} { r l } { \mathrm { w h e r e } } & { { } \mathcal { C } _ { i , j , t } = \operatorname* { m i n } \Big ( r _ { i , j , t } ( \theta ) \hat { A } _ { i , j , t } , ~ \mathrm { c l i p } \Big ( r _ { i , j , t } ( \theta ) , 1 - \varepsilon _ { l o w } , 1 + \varepsilon _ { h i g h } \Big ) \hat { A } _ { i , j , t } \Big ) } \end{array}$

$$
r _ {i, j, t} (\theta) = \frac {\pi_ {\theta} (o _ {i , j , t} \mid q , o _ {i , j , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (o _ {i , j , t} \mid q , o _ {i , j , <   t})}.
$$

Following the RLVR recipe (Guo et al., 2025; Jin et al., 2025; Yu et al., 2025), we train the model with a final outcome reward computed by a rule-based verifier:

$$
R (\hat {y}, y) = \mathbf {1} _ {\text { is\_equiv } (y, \hat {y})} \tag {3}
$$

where $\hat { y }$ is the predicted answer while y refers to the ground truth.

# 2.3 RETHINKING MEMAGENT FROM AUTOREGRESSIVE MODELING PERSPECTIVES

Tto get a deeper sense of the MEMAGENT design, we propose to re-think language-model factorization in the following fashion. A standard autoregressive LLM factorizes the joint likelihood of a sequenceleast its h $\mathbf { x } _ { 1 : N }$ as st $\begin{array} { r } { p ( \mathbf { x } _ { 1 : N } ) = \prod _ { n = 1 } ^ { N } p ( x _ { n } \mid \mathbf { x } _ { 1 : n - 1 } ) } \end{array}$ , implicitly assuming that every past token (or att. This is what turns quadratic attention into the long-context bottleneck.

MEMAGENT replaces the unbounded history with a fixed-length memory m $\in \mathbb { V } ^ { M }$ , as shown in Figure 4. The input text is streamed through the model in K contiguous chunks $\mathbf { c } ^ { 1 } , \ldots , \mathbf { c } ^ { K }$ (each of $\mathrm { l e n g t h } \le C )$ . After chunk k is read, the model overwrites the panel with a new vector $\mathbf { m } ^ { k }$ that summarizes all evidence seen so far. Because $| \mathbf { m } ^ { k } | = M$ is constant, both compute and memory per step are $O ( C + M )$ , yielding an overall linear complexity $O ( N )$ .

Introducing the latent sequence $\mathbf { m } ^ { 1 : K - 1 }$ decomposes the original likelihood as

$$
p (\mathbf {x} _ {1: N}) = \sum_ {\mathbf {m} ^ {1: K - 1}} \prod_ {k = 1} ^ {K} \underbrace {p (\mathbf {c} ^ {k} \mid \mathbf {m} ^ {k - 1})} _ {\text { read }} \underbrace {p (\mathbf {m} ^ {k} \mid \mathbf {c} ^ {k} , \mathbf {m} ^ {k - 1})} _ {\text { write }}, \tag {4}
$$

with base case $\mathbf { m } ^ { 0 } = \emptyset$ . Inside each chunk, we still run an ordinary transformer decoder, but conditioned on a constant context window $( \mathbf { c } ^ { k } , \mathbf { m } ^ { k - 1 } )$ . The read path factorizes token-by-token, p(ck | mk−1) =QkCi=(k−1)C+1p(xi | x1:i−1, mk−1), while the write path generates the next memory $p ( \mathbf { c } ^ { k } \mid \mathbf { m } ^ { k - 1 } ) = \prod _ { i = ( k - 1 ) C + 1 } ^ { k C } p ( x _ { i } \mid \mathbf { x } _ { 1 : i - 1 } , \mathbf { m } ^ { k - 1 } )$ in the same autoregressive fashion.

Table 2: Model performance on LongBench-SUM. All values represent recall rates (%). Bold marks the highest value and Underline marks the second highest value in each column. 

<table><tr><td rowspan="2">Model</td><td colspan="4">GOV REPORT</td><td colspan="4">QMSUM</td></tr><tr><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>AVG</td><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>AVG</td></tr><tr><td>Qwen2.5-Instruct-32B</td><td>23.67</td><td>8.46</td><td>12.57</td><td>14.90</td><td>47.77</td><td>11.29</td><td>28.17</td><td>29.08</td></tr><tr><td>Qwen2.5-Instruct-14B</td><td>31.19</td><td>10.96</td><td>14.96</td><td>19.04</td><td>47.53</td><td>11.46</td><td>28.28</td><td>29.09</td></tr><tr><td>Qwen2.5-Instruct-7B</td><td>30.91</td><td>11.68</td><td>15.20</td><td>19.26</td><td>46.64</td><td>12.01</td><td>28.33</td><td>28.99</td></tr><tr><td>QwenLong-L1</td><td>27.60</td><td>8.20</td><td>13.07</td><td>16.29</td><td>39.44</td><td>8.24</td><td>23.56</td><td>23.74</td></tr><tr><td>Qwen2.5-Instruct-14B-1M</td><td>30.58</td><td>11.93</td><td>15.51</td><td> $\underline{19.34}$ </td><td>47.31</td><td>13.13</td><td>29.07</td><td>29.84</td></tr><tr><td>Qwen2.5-Instruct-7B-1M</td><td>31.02</td><td>11.47</td><td>15.30</td><td> $\underline{19.26}$ </td><td>46.72</td><td>12.33</td><td>28.66</td><td>29.24</td></tr><tr><td>DS-Distill-Qwen-32B</td><td>26.13</td><td>8.86</td><td>12.98</td><td>15.99</td><td>39.09</td><td>8.75</td><td>23.96</td><td>23.93</td></tr><tr><td>DS-Distill-Qwen-14B</td><td>28.24</td><td>9.72</td><td>13.78</td><td>17.25</td><td>41.25</td><td>8.95</td><td>25.00</td><td>25.07</td></tr><tr><td>DS-Distill-Qwen-7B</td><td> $\underline{33.30}$ </td><td>9.39</td><td>14.59</td><td>19.10</td><td>34.33</td><td>5.97</td><td>21.57</td><td>20.62</td></tr><tr><td>RL-MEMAGENT-14B</td><td>37.16</td><td> $\underline{12.03}$ </td><td>16.23</td><td>21.80</td><td>50.21</td><td> $\underline{12.70}$ </td><td>31.27</td><td>31.39</td></tr><tr><td>RL-MEMAGENT-7B</td><td>30.28</td><td> $\underline{12.37}$ </td><td> $\underline{15.37}$ </td><td> $\underline{19.34}$ </td><td> $\underline{48.49}$ </td><td> $\underline{14.41}$ </td><td> $\underline{30.91}$ </td><td> $\underline{31.27}$ </td></tr></table>

In our formulation, the model’s reading and writing operations over the context constitute an Markov Decision Process(MDP) and the objective of RL is to optimize the final reward obtained by this MDP. Therefore, MemAgent’s learning objective is to generate a read–write memory trajectory that maximizes the reward, which corresponds to learning an optimal distribution over memory states conditioned on the input context. This further theoretically illustrates the intrinsic unity between our RL formulation and long-text modeling.

# 3 EXPERIMENTS

# 3.1 EXPERIMENTAL SETUP

Training Details. To maintain comparability with previous work, we choose Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct (Yang et al., 2024) as backbone models. We implement the framework for multi-conversation with independent contexts based on verl (Sheng et al., 2024).

We employ a two-stage curriculum RL strategy:

Table 3: Model performance on LongBench-QA. 

<table><tr><td>Method</td><td>2Wiki</td><td>HQA</td><td>MuSiQue</td><td>NQA</td><td>Qasper</td><td>AVG</td></tr><tr><td>QwenLong-L1-32B</td><td>83.0</td><td>69.5</td><td>51.0</td><td>26.0</td><td>24.0</td><td>50.7</td></tr><tr><td>Qwen2.5-Instruct-14B-1M</td><td>70.5</td><td>65.5</td><td>35.0</td><td>22.0</td><td>22.0</td><td>43.0</td></tr><tr><td>Qwen2.5-Instruct-7B-1M</td><td>67.5</td><td>58.5</td><td>26.0</td><td>21.5</td><td>24.0</td><td>39.5</td></tr><tr><td>DS-Distill-Qwen-32B</td><td>83.5</td><td>69.0</td><td>47.5</td><td>24.0</td><td>21.0</td><td>49.0</td></tr><tr><td>DS-Distill-Qwen-14B</td><td>83.5</td><td>67.0</td><td>42.0</td><td>22.0</td><td>22.0</td><td>47.3</td></tr><tr><td>DS-Distill-Qwen-7B</td><td>47.0</td><td>31.5</td><td>7.5</td><td>4.0</td><td>17.5</td><td>21.5</td></tr><tr><td>Qwen2.5-Instruct-32B</td><td>68.5</td><td>66.0</td><td>37.0</td><td>24.5</td><td>22.5</td><td>43.7</td></tr><tr><td>Qwen2.5-Instruct-14B</td><td>67.0</td><td>63.5</td><td>39.0</td><td>20.0</td><td>20.5</td><td>42.0</td></tr><tr><td>Qwen2.5-Instruct-7B</td><td>52.5</td><td>59.0</td><td>24.0</td><td>19.0</td><td>19.0</td><td>34.7</td></tr><tr><td>MEMAGENT-14B</td><td>79.0</td><td>73.0</td><td>52.0</td><td>25.0</td><td>26.0</td><td>51.0</td></tr><tr><td>MEMAGENT-7B</td><td>74.0</td><td>69.5</td><td>47.0</td><td>21.5</td><td>29.0</td><td>48.2</td></tr></table>

1) stage I focuses on enabling the model to acquire fundamental memory capabilities; 2) stage II trains the model to transfer these capabilities to more diverse contexts and challenging tasks. Specific hyperparameters for training are detailed in § A.3.

During training, we intentionally limit the model to an 8K context window to demonstrate its extrapolation capabilities. This 8K window is allocated as follows: 1024 tokens for the query, 5000 tokens for the context chunk, 1024 tokens for the memory, and 1024 tokens for the output, with the remaining tokens reserved for the chat template.

Benchmarks. We conduct comprehensive evaluations on several long-text benchmarks to assess the model’s capabilities across various text types and tasks.

1. RULER-HQA. This benchmark is created using the same synthetic method as in the firststage training data. It consists of tasks with a moderate information density and controllable length, where the context distribution is close to natural language, serving as a quantitative evaluation of extrapolation performance.   
2. LongBench-QA. This benchmark is composed of NarrativeQA (Kociskˇ y et al.\` , 2018), Qasper (Dasigi et al., 2021), HotpotQA (Yang et al., 2018), 2WikiMultihopQA Ho et al. (2020), and MuSiQue (Trivedi et al., 2022). The tasks are relatively short but have a high information density, which severely tests the model’s flexible memory management. It also evaluates the model’s ability to generalize its memory capabilities to various materials, such as novels, news articles, and Wiki items.

![](images/24a14e555266ebf0a2ba01d4bec3fa52f2bb4011fb97415cf33274cf930e3c3d.jpg)  
Figure 5: Performance heatmaps on NIAH benchmark across different context lengths.

3. NIAH. Needle in a haystack (NIAH) (Kamradt, 2023) is a series of extremely long synthetic tasks with very low information density. To succeed, the model must identify key information and maintain its integrity throughout a long process, thereby testing the robustness of memory.   
4. LongBench-SUM. We also adopt two long-context summay tasks, GovReport(Huang et al., 2021) and QMSum(Zhong et al., 2021) from LongBench(Bai et al., 2024) to evaluate the performance in different task category that is different from retrieval QA.

Baselines. We use DeepSeek-R1-Distill-Qwen (Guo et al., 2025), Qwen-2.5-Instruct-1M (Yang et al., 2025) , Qwen-2.5-Instruct (Yang et al., 2024)and QwenLong-L1 (Wan et al., 2025) as baselines. Their generation configurations are shown in Table 6, while MEMAGENT uses the same context management as described previously in Training Details. We also compare MEMAGENT with other agent method, detailed in § D.2.

# 3.2 MAIN RESULTS

RULER-HQA. The results are reported in Table 10. We conduct a comparative analysis of all model performances with context lengths ranging from 7K to 896K. For MEMAGENT, we extend the evaluation to ultra-long contexts of 1.75M and 3.5M to assess its extrapolation capabilities.

MEMAGENT exhibits remarkable length extrapolation capabilities with only marginal performance decay as the input context-length increases. In contrast, baseline models show distinct failure patterns. DS-Distill-Qwen series show rapid performance degradation. QwenLong-L1 maintains reasonable performance within its training length but experiences substantial degradation afterward. The Qwen2.5-Instruct-1M series maintains acceptable performance up to 112K tokens, but the performance deteriorates to zero at 896K tokens, well before reaching their theoretical 1M token capacity. This suggests that despite extended context windows, these models struggle with effective information utilization in ultra-long contexts.

LongBench-QA. The results on the LongBench-QA benchmark are presented in Table 3. MEMA-GENT demonstrates superior overall performance, outperforming larger long-context or reasoning models. Reasoning models such as the DS-Distill families and the QwenLong model which are trained on a complex dataset, exhibit strong performance. In contrast, the Qwen2.5-Instruct-1M series shows limited improvement over its backbone model. This suggests that LongBench-QA emphasize a deeper understanding of text rather than simple retrieval ability. The performance of MEMAGENT demonstrates that the memory capabilities acquired through reinforcement learning are generalizable.

![](images/008baa071092d74a34048eee4e7e48d74e1c91d07c44dbf9c267b29028e4de1b.jpg)

<details>
<summary>line</summary>

| Context Length in Tokens | RL-Memagent-14B | RL-Memagent-7B | MemAgent-14B w/o RL | MemAgent-7B w/o RL | Qwen2.5-Instruct-14B | Qwen2.5-Instruct-7B |
| ------------------------ | --------------- | -------------- | ------------------- | ------------------ | -------------------- | ------------------- |
| 7K                       | 80              | 80             | 60                  | 60                 | 75                   | 50                  |
| 14K                      | 80              | 80             | 55                  | 55                 | 65                   | 55                  |
| 28K                      | 80              | 80             | 50                  | 50                 | 55                   | 50                  |
| 56K                      | 80              | 80             | 45                  | 45                 | 50                   | 45                  |
| 112K                     | 80              | 80             | 40                  | 35                 | 40                   | 30                  |
| 224K                     | 75              | 75             | 35                  | 35                 | 20                   | 15                  |
| 448K                     | 80              | 75             | 30                  | 35                 | 10                   | 5                   |
| 896K                     | 75              | 70             | 25                  | 35                 | 0                    | 0                   |
</details>

Figure 6: Ablation result of RL training on RULER-HQA.

![](images/898ed62ef1d8b0457b84686ca65ba55f813aa5e7d6919d8bde48a565b5c2516f.jpg)

<details>
<summary>bar</summary>

| Model | 7B Instruct | 7B MemAgent w/o RL | 7B MemAgent w/ RL | 14B Instruct | 14B MemAgent w/o RL | 14B MemAgent w/ RL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 2Wiki | 53 | 64 | 75 | 68 | 77 | 80 |
| HQA | 60 | 60 | 70 | 63 | 60 | 74 |
| MuSiQue | 24 | 30 | 47 | 39 | 31 | 53 |
| NQA | 19 | 19 | 22 | 20 | 17 | 25 |
| Qasper | 19 | 18 | 30 | 21 | 15 | 26 |
| AVG | 35 | 39 | 49 | 42 | 40 | 52 |
</details>

Figure 7: Ablation result of RL training on Longbench-QA.

NIAH. We adopt three variants of NIAH from the RULER benchmark Hsieh et al. (2024) with increasing difficulty across three levels. As depicted in Figure 5, the majority of baselines struggle to maintain consistent performance even within a 128K context window, even Qwen2.5-Instruct-1M also experience a performance drop at 512K. RL-MEMAGENT, despite suffering some performance fluctuations, shows only a minimal performance loss of less than 5% at 512K. This robust performance is particularly noteworthy given that the evaluation at 512K involves more than 100 turns of dialogue.

LongBench-SUM. We evaluate summary quality by the recall scores of ROUGE-{1,2,L}. RL-MEMAGENT achieves SOTA on almost all metrics, demonstrating that the model has learned general memory and context management capabilities, rather than abilities specific to the QA task.

# 3.3 ABLATION STUDY

# 3.3.1 RL TRAINING

To investigate the impact of reinforcement learning, we conduct ablation experiments. The results of RULER-HQA and NIAH are presented in Figure 6 and Figure 5, respectively. MEMAGENT without reinforcement learning training outperforms the backbone models; however, it still exhibits a substantial decline in performance as the input length increases. The results of Longbench-QA, shown in Figure 7, demonstrate that directly applying MEMAGENT leads to only marginal or even negative improvements. In contrast, RL-MEMAGENT achieves significant improvements in both evaluation scenarios, indicating reinforcement learning training is essential to develop generalizable and robust memory abilities.

# 3.3.2 MEMORY LENGTH

Selecting an appropriate MEMAGENT setting involves certain trade-offs. A larger memory size allows the model to store more useful information, but it also introduces challenges in memory management and increases the likelihood of redundancy. Conversely situation may lead to insufficient storage capacity, leaving the model without the necessary references.

To achieve a reasonable compression ratio while keeping the total context length within 8,192 tokens, we set the default configuration of MEMAGENT to use a 1,024-token memory and context chunks of 5,000 tokens, based on preliminary validation results.

To investigate the effect of hyperparameter choices, we conduct an ablation study on memory length ranging from 256 to 4096. The results presented in Figure 8 and Figure 9, showing that our chosen configuration constitutes a reasonable sweet spot, and that MEMAGENT ’s performance is robust over different memory size. We further examine the impact of varying the context size in § D.1 and observe similar trends.

![](images/d30d7784f1f7f6b9500e597040f142582ad93f1543de347729db851b1305f09a.jpg)  
Figure 8: Ablation result of memory-length on NIAH

![](images/61345e8a2e900f2305d6987e4831eefcbbe6d01719a9da8b65d861ca644d519e.jpg)

<details>
<summary>line</summary>

| Dataset | Batch Size | 7B   | 14B  |
|---------|------------|------|------|
| 2Wiki   | 256        | 73.0 | 74.0 |
| 2Wiki   | 512        | 72.0 | 78.0 |
| 2Wiki   | 1024       | 74.0 | 79.0 |
| 2Wiki   | 2048       | 71.0 | 78.0 |
| 2Wiki   | 4096       | 70.0 | 79.0 |
| HQA     | 256        | 68.0 | 69.0 |
| HQA     | 512        | 69.0 | 69.0 |
| HQA     | 1024       | 69.0 | 71.0 |
| HQA     | 2048       | 69.0 | 70.0 |
| HQA     | 4096       | 67.0 | 70.0 |
| MuSiQue | 256        | 42.0 | 38.0 |
| MuSiQue | 512        | 43.0 | 45.0 |
| MuSiQue | 1024       | 44.0 | 48.0 |
| MuSiQue | 2048       | 43.0 | 46.0 |
| MuSiQue | 4096       | 44.0 | 46.0 |
| NQA     | 256        | 18.0 | 23.0 |
| NQA     | 512        | 19.0 | 21.0 |
| NQA     | 1024       | 21.0 | 25.0 |
| NQA     | 2048       | 22.0 | 23.0 |
| NQA     | 4096       | 23.0 | 28.0 |
| Qasper  | 256        | 27.0 | 24.0 |
| Qasper  | 512        | 18.0 | 33.0 |
| Qasper  | 1024       | 29.0 | 31.0 |
| Qasper  | 2048       | 27.0 | 26.0 |
| Qasper  | 4096       | 25.0 | 25.0 |
| AVG     | 256        | 45.0 | 45.0 |
| AVG     | 512        | 46.0 | 47.0 |
| AVG     | 1024       | 48.0 | 49.0 |
| AVG     | 2048       | 46.0 | 47.0 |
| AVG     | 4096       | 46.0 | 47.0 |
</details>

Figure 9: Ablation result of memory-length on Longbench

# 3.3.3 CONTEXT DISTRIBUTION

Although our experiments show that MEMAGENT can effectively extrapolate to a length of 3.5M tokens, we still wish to examine whether MEMAGENT is affected by issues such as informationoverwritten and the lost-in-the-middle phenomenon. Our hypothesis is that overcoming such problem is a natural result of end-to-end optimization. During training, the model learns to preserve and track critical information in order to maximize the final reward.

To validate this hypothesis, we carefully design a set of probing experiments based on RULER-HQA, where the context is consist of some key information and many distractors. We divided the key information into two groups and placed them at different positions within the context. We constructed five settings: (0%, 100%), (20%, 80%), (40%, 60%), (0%, 20%), and (80%, 100%), where 0% indicates the beginning of the context and 100% means the end of the context.

For example, in the (0%, 100%) case, the model sees one piece of key information at the very beginning and the other only at the final memory update step. This represents one of the most challenging scenarios for the information-overwritten problem. While (40%, 60%) may serve as a challenging lost-in-the-middle setting.

The results shown in table 4 indicates that MemAgent remains consistently robust across all patterns without exhibiting any catastrophic performance degradation. This strongly supports our hypothesis that the general memory abilities acquired through trial and error are not tied to any particular context pattern.

# 4 RELATED WORK

Long Context LLMs. Extrapolation methods for RoPE-based LLMs (Su et al., 2024), such as NTK (bloc97, 2023), PI (Chen et al., 2023), YaRN (Peng et al., 2023b) and DCA (An et al., 2024), modify the components of positional embeddings, enabling the model to capture long-range semantic dependencies. On the other hand, Linear attention mechanisms (Child et al., 2019; Katharopoulos et al., 2020), Recurrent Neural Networks (RNNs) and State Space Models (SSMs) (Gu et al., 2021; Gu & Dao, 2023; Peng et al., 2023a; De et al., 2024; Feng et al., 2024), sparse attention (Beltagy et al., 2020; Zhao et al., 2019; Xiao et al., 2023; Yuan et al., 2025; Lu et al., 2025) focus on architecture improvements. Chunk strategy have also been explored in long-context modeling (Li et al., 2025b; Liao et al., 2025), while MEMAGENT aims to equip memory ability to any backbone model via post-training with standard RL frameworks without heavily changing on architecture.

Table 4: Probe experiment results. Ctx. Dist. denotes the context distribution, where the two numbers correspond to the relative positions of the two key-information groups within the entire context. 0% means the beginning and 100% means the end. random indicates randomly shuffling all context items, consistent with the setup in the main experiment. The other rows show the performance difference relative to random. All values represent accuracy (%). 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Ctx. Dist.</td><td colspan="8">Length</td></tr><tr><td>7K</td><td>14K</td><td>28K</td><td>56K</td><td>112K</td><td>224K</td><td>448K</td><td>AVG</td></tr><tr><td rowspan="6">14B</td><td>random</td><td>80.47</td><td>82.03</td><td>82.03</td><td>83.59</td><td>81.25</td><td>77.34</td><td>79.69</td><td>75.78</td></tr><tr><td>0% 20%</td><td>+3.91</td><td>-3.91</td><td>+3.13</td><td>+1.57</td><td>0.00</td><td>+4.69</td><td>+3.90</td><td>+1.90</td></tr><tr><td>0% 100%</td><td>+3.12</td><td>+0.78</td><td>+3.13</td><td>-3.12</td><td>+1.56</td><td>+7.82</td><td>+6.25</td><td>+2.79</td></tr><tr><td>20% 80%</td><td>+0.78</td><td>-3.12</td><td>+2.35</td><td>+0.79</td><td>-3.13</td><td>+5.47</td><td>-3.13</td><td>0.00</td></tr><tr><td>40% 60%</td><td>+1.56</td><td>+2.35</td><td>-2.34</td><td>-3.12</td><td>-1.56</td><td>+3.13</td><td>-0.78</td><td>-0.11</td></tr><tr><td>80% 100%</td><td>-2.35</td><td>0.00</td><td>+1.56</td><td>+0.79</td><td>+3.13</td><td>0.00</td><td>+1.56</td><td>+0.67</td></tr><tr><td rowspan="6">7B</td><td>random</td><td>81.25</td><td>81.25</td><td>82.03</td><td>80.47</td><td>79.69</td><td>75.78</td><td>76.56</td><td>79.58</td></tr><tr><td>0% 20%</td><td>-1.56</td><td>-0.78</td><td>+3.13</td><td>+3.91</td><td>+3.12</td><td>+5.47</td><td>+3.13</td><td>+2.35</td></tr><tr><td>0% 100%</td><td>+0.78</td><td>0.00</td><td>+2.35</td><td>+1.56</td><td>+3.90</td><td>+4.69</td><td>+2.35</td><td>+2.23</td></tr><tr><td>20% 80%</td><td>-0.78</td><td>-0.78</td><td>+2.35</td><td>0.00</td><td>0.00</td><td>0.00</td><td>+3.13</td><td>+0.56</td></tr><tr><td>40% 60%</td><td>0.00</td><td>+1.56</td><td>+3.13</td><td>+0.78</td><td>-3.91</td><td>+3.13</td><td>+5.47</td><td>+1.45</td></tr><tr><td>80% 100%</td><td>+1.56</td><td>0.00</td><td>+0.78</td><td>-0.78</td><td>+0.78</td><td>+4.69</td><td>0.00</td><td>+1.00</td></tr></table>

Memory Mechanism. The Long Short-Term Memory (LSTM) mechanism (Hochreiter & Schmidhuber, 1997) achieved significant success in early NLP tasks, while Neural Turing Machines (Graves et al., 2014) and Memory Networks (Weston et al., 2014) demonstrated how to equip neural networks with memory. Existing memory mechanisms integrated to Transformer models are typically realized by adding external memory modules (Martins et al., 2021; Wu et al., 2020; Behrouz et al., 2024; Bulatov et al., 2023) or external database (Zhong et al., 2024; Lu et al., 2023; Modarressi et al., 2023). Recently, retrieval-augmented memory agent (Fang et al., 2025; Chhikara et al., 2025; Zhou et al., 2025) workflows have attracted the community’s attention. The diffrence between MEMAGENT and other agent is that we use reinforcement learning to enable LLM itself the ability to memorize.

Reinforcement Learning for LLMs. In recent RL studies, the reward signals have gradually shifted from human preferences (Ouyang et al., 2022) or reward models distilled from them (Bai et al., 2022) to rule-based feedback, which has demonstrated great potential in enhancing model reasoning capabilities (OpenAI, 2024; Guo et al., 2025; Qwen, 2024; DeepMind, 2024; Team et al., 2025) with GAE (Schulman et al., 2018) based PPO (Schulman et al., 2017) or GRPO (Shao et al., 2024) training. Algorithmic enhancements (Hu, 2025; Yu et al., 2025; Liu et al., 2025) have mostly focused on improving sustainability and efficiency of these algorithms. To further release the potential of RL, recent works such as Search-R1 (Jin et al., 2025), Agent-R1 (Ouyang et al., 2025) and RAGEN (Wang et al., 2025) have explored the training of tool-using agents based on multi-turn chat. GiGPO (Feng et al., 2025) further investigates the use of multiple independent contexts in agent training.

# 5 CONCLUSION

In this paper, we introduce MEMAGENT, a novel long-context method that employs an RL-trained memory module, which enables large language models (LLMs) to selectively record relevant information while disregarding extraneous details. Our experiments demonstrate that when trained on 60K-length sequences, MEMAGENT exhibits remarkable extrapolation, extending its effective context to 3.5M tokens with only 8K context. The model achieves state-of-the-art performance across a diverse range of long-context tasks. Our ablation studies reveal the critical role of RL-based training in achieving these results and how memory capacity influences performance across different task types, providing key insights into the proposed memory mechanism. We hope that this work may lay a strong foundation for developing more advanced memory architectures and training strategies, thereby paving the way for significantly enhancing the long-context capabilities of LLMs.

# REPRODUCIBILITY STATEMENT

For reproducibility, we have provided the inplementation details in (§ A), including the prompt template (§ A.1), pseudocode (§ A.2) and training recipe and algorithm hyperparameter (§ A.3) and evaluation settings (§ A.4). The training and evaluation code, as well as the dataset and model weights, will be available in open-source platforms.

# REFERENCES

Chenxin An, Fei Huang, Jun Zhang, Shansan Gong, Xipeng Qiu, Chang Zhou, and Lingpeng Kong. Training-free long-context scaling of large language models. arXiv preprint arXiv:2402.17463, 2024.   
Anthropic. Claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/ claude-3-5-sonnet.   
Anthropic. Introducing claude 4, 2025. URL https://www.anthropic.com/news/ claude-4.   
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.   
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3119–3137, 2024.   
Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024.   
Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.   
bloc97. NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation., 2023. URL https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware\_ scaled\_rope\_allows\_llama\_models\_to\_have/.   
Aydar Bulatov, Yuri Kuratov, Yermek Kapushev, and Mikhail S Burtsev. Scaling transformer to 1m tokens and beyond with rmt. arXiv preprint arXiv:2304.11062, 2023.   
Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.   
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.   
Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.   
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. arXiv preprint arXiv:2105.03011, 2021.   
Soham De, Samuel L Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, et al. Griffin: Mixing gated linear recurrences with local attention for efficient language models. arXiv preprint arXiv:2402.19427, 2024.   
Google DeepMind. Gemini 2.0 flash thinking, 2024. URL https://deepmind.google/ technologies/gemini/flash-thinking/.   
Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: Exploring agent procedural memory. arXiv preprint arXiv:2508.06433, 2025.   
Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. arXiv preprint arXiv:2505.10978, 2025.

Leo Feng, Frederick Tung, Hossein Hajimirsadeghi, Mohamed Osama Ahmed, Yoshua Bengio, and Greg Mori. Attention as an rnn. arXiv preprint arXiv:2405.13956, 2024.   
Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, and Songlin Hu. Nextlong: Toward effective long-context training without long documents. arXiv preprint arXiv:2501.12766, 2025.   
Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014.   
Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.   
Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured ´ state spaces. arXiv preprint arXiv:2111.00396, 2021.   
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060, 2020.   
Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨ Neural computation, 9(8): 1735–1780, 1997.   
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.   
Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.   
Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization. arXiv preprint arXiv:2104.02112, 2021.   
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for accelerated inference of large language models. arXiv preprint arXiv:2310.05736, 2023.   
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.   
Gregory Kamradt. Needle In A Haystack - pressure testing LLMs. Github, 2023. URL https: //github.com/gkamradt/LLMTest\_NeedleInAHaystack/tree/main.   
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.   
Toma´s Ko ˇ cisk ˇ y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, G \` abor Melis, ´ and Edward Grefenstette. The narrativeqa reading comprehension challenge. Transactions of the Association for Computational Linguistics, 6:317–328, 2018.   
Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, et al. Minimax-01: Scaling foundation models with lightning attention. arXiv preprint arXiv:2501.08313, 2025a.   
Yucheng Li, Bo Dong, Chenghua Lin, and Frank Guerin. Compressing context to enhance inference efficiency of large language models. arXiv preprint arXiv:2310.06201, 2023.

Zhenyu Li, Yike Zhang, Tengyu Pan, Yutao Sun, Zhichao Duan, Junjie Fang, Rong Han, Zixuan Wang, and Jianyong Wang. Focusllm: Precise understanding of long context by dynamic condensing. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 31087–31101, 2025b.   
Zihan Liao, Jun Wang, Hang Yu, Lingxiao Wei, Jianguo Li, and Wei Zhang. E2llm: Encoder elongated large language models for long-context understanding and reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 19212–19241, 2025.   
Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-ofexperts language model. arXiv preprint arXiv:2405.04434, 2024.   
Xiaoran Liu, Hang Yan, Shuo Zhang, Chenxin An, Xipeng Qiu, and Dahua Lin. Scaling laws of rope-based extrapolation. arXiv preprint arXiv:2310.05209, 2023.   
Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.   
Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, et al. Moba: Mixture of block attention for long-context llms. arXiv preprint arXiv:2502.13189, 2025.   
Junru Lu, Siyu An, Mingbao Lin, Gabriele Pergola, Yulan He, Di Yin, Xing Sun, and Yunsheng Wu. Memochat: Tuning llms to use memos for consistent long-range open-domain conversation. arXiv preprint arXiv:2308.08239, 2023.   
Pedro Henrique Martins, Zita Marinho, and Andre FT Martins. ´ ∞-former: Infinite memory transformer. arXiv preprint arXiv:2109.00301, 2021.   
George A Miller et al. The magical number seven, plus or minus two. Psychological review, 63(2): 81–97, 1956.   
Ali Modarressi, Ayyoob Imani, Mohsen Fayyaz, and Hinrich Schutze. Ret-llm: Towards a general ¨ read-write memory for large language models. arXiv preprint arXiv:2305.14322, 2023.   
OpenAI. GPT4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
OpenAI. Learning to reason with llms, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.   
Jie Ouyang, Ruiran Yan, Yucong Luo, Mingyue Cheng, Qi Liu, Zirui Liu, Shuo Yu, and Daoyu Wang. Training powerful llm agents with end-to-end reinforcement learning, 2025. URL https: //github.com/0russwest0/Agent-R1.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 27730–27744. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf.   
Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023a.   
Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023b.

Qwen. Qwq-32b: Embracing the power of reinforcement learning, 2024. URL https://qwenlm. github.io/blog/qwq-32b/.   
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.   
John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation, 2018. URL https://arxiv.org/ abs/1506.02438.   
ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.   
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Y Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv:2409.19256, 2024.   
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.   
Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.   
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.   
Fanqi Wan, Weizhou Shen, Shengyi Liao, Yingcheng Shi, Chenliang Li, Ziyi Yang, Ji Zhang, Fei Huang, Jingren Zhou, and Ming Yan. Qwenlong-l1: Towards long-context large reasoning models with reinforcement learning. arXiv preprint arXiv:2505.17667, 2025.   
Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning, 2025. URL https://arxiv.org/abs/2504. 20073.   
Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. arXiv preprint arXiv:1410.3916, 2014.   
Qingyang Wu, Zhenzhong Lan, Kun Qian, Jing Gu, Alborz Geramifard, and Zhou Yu. Memformer: A memory-augmented transformer for sequence modeling. arXiv preprint arXiv:2010.06891, 2020.   
XAI. Grok 3 beta — the age of reasoning agents, 2024. URL https://x.ai/news/grok-3.   
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.   
Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. Effective long-context scaling of foundation models. arXiv preprint arXiv:2309.16039, 2023.   
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

An Yang, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoyan Huang, Jiandong Jiang, Jianhong Tu, Jianwei Zhang, Jingren Zhou, Junyang Lin, Kai Dang, Kexin Yang, Le Yu, Mei Li, Minmin Sun, Qin Zhu, Rui Men, Tao He, Weijia Xu, Wenbiao Yin, Wenyuan Yu, Xiafei Qiu, Xingzhang Ren, Xinlong Yang, Yong Li, Zhiying Xu, and Zipeng Zhang. Qwen2.5-1m technical report. arXiv preprint arXiv:2501.15383, 2025.   
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.   
Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. Helmet: How to evaluate long-context language models effectively and thoroughly. arXiv preprint arXiv:2410.02694, 2024.   
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.   
Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, YX Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089, 2025.   
Jiaxin Zhang, Yiqi Wang, Xihong Yang, Siwei Wang, Yu Feng, Yu Shi, Ruichao Ren, En Zhu, and Xinwang Liu. Test-time training on graphs with large language models (llms). In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 2089–2098, 2024.   
Guangxiang Zhao, Junyang Lin, Zhiyuan Zhang, Xuancheng Ren, Qi Su, and Xu Sun. Explicit sparse transformer: Concentrated attention through explicit selection. arXiv preprint arXiv:1912.11637, 2019.   
Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, et al. Qmsum: A new benchmark for query-based multidomain meeting summarization. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 5905–5921, 2021.   
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19724–19731, 2024.   
Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. arXiv preprint arXiv:2506.15841, 2025.

# A IMPLEMENTATION DETAILS

# A.1 PROMPT TEMPLATE

You are presented with a problem, a section of an article that may contain the answer, and a previous memory. Please read the section carefully and update the memory with new information that helps to answer the problem, while retaining all relevant details from the previous memory.

```txt
<problem> {prompt} </problem>
<memory> {memory} </memory>
<section> {chunk} </section> 
```

# Updated memory:

You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \boxed {}. <problem> {prompt} </problem> <memory> {memory} </memory>

Your answer:

Table 5: Template of MEMAGENT for context processing (top part) and final answer generation (bottom). Curly-brace placeholders {} will be replaced with actual content.

# A.2 ALGORITHM

Algorithm 1 Multi-conv DAPO   
Require: Policy model $\pi_{\theta}$ , reference model $\pi_{ref}$ (frozen), rule-based verifier V, dataset D, group size G

1: while not converged do
2:    Sample a prompt $q \sim D$ 3:    for g = 1 to G do
4:    Initialize $h_{g,0} \leftarrow [q]$ 5:    for t = 1 to $c_{g}$ do
6:    Sample $o_{g,t} \sim \pi_{\theta}(\cdot \mid h_{g,t-1})$ 7: $h_{g,t} \leftarrow h_{g,t-1} \parallel o_{g,t}$ 8:    end for
9: $y_{g} \leftarrow o_{g,c_{g}}$ 10: $\hat{r}_{g} \leftarrow V(q, y_{g})$ 11: $d_{g} \leftarrow \text{KL}\big(\pi_{\theta}(\cdot \mid h_{g,c_{g}}) \parallel \pi_{\text{ref}}(\cdot \mid h_{g,c_{g}})\big)$ 12: $r_{g} \leftarrow \hat{r}_{g} - \beta d_{g}$ 13: end for
14: $\{A_{g}\}_{g=1}^{G} \leftarrow \text{GroupNorm}\big(\{r_{g}\}_{g=1}^{G}\big)$ 15: for g = 1 to G do
16: $\rho_{g} \leftarrow \frac{\pi_{\theta}(y_{g} \mid h_{g,c_{g}})}{\pi_{\theta_{\text{old}}} (y_{g} \mid h_{g,c_{g}})}$ 17: $J_{g} \leftarrow \min\big(\rho_{g} A_{g}, \text{clip} (\rho_{g}, 1 - \epsilon_{low}, 1 + \epsilon_{high}) A_{g}\big)$ 18: end for
19: $J_{clip} \leftarrow \frac{1}{G} \sum_{g=1}^{G} J_{g}$ 20: $\theta \leftarrow \theta + \eta \nabla_{\theta} J_{clip}$ 21: end while

# A.3 TRAINING

We use the DAPO algorithm for training, applying a KL factor of $1 \times 1 0 ^ { - 3 }$ and disabling the entropy loss. The AdamW optimizer is employed with a constant learning rate of $1 \times 1 0 ^ { - 6 }$ and a linear warmup scheduler, with the wram-up step set to 20. We use a rollout batchsize of 256, with a group size of 16. Note that due to the multi-conversation feature of MEMAGENT, the actual mini-batchsize is not equal to rollout batchsize/16. We utilize off-policy training by fixing the ratio of the sample batch size to the backpropagation batch size is set to 16.

We shift to stage II when stage I are fully converged, which takes about 400 steps. Here is the training data recipe of each stage.

• Stage I We use 32,768 synthetic QA data instances, each approximately 32K tokens in length. These are based on the HotpotQA (Yang et al., 2018) dataset and follow the RULER (Hsieh et al., 2024) methodology, which involves embedding golden paragraphs (containing correct answers) within extensive distractor content sampled from the same dataset.   
• Stage II We use 2,560 training instances with a maximum length of 60K tokens. This set consists of difficult, high-quality long-text QA data from DocQA-RL-1.6K (Wan et al., 2025), mixed with data from the first stage.

Each training sample used in stage I is of 200 articles in HotpotQA, with an approximate total token length of 28K. We thoroughly clean the dataset by filtering out questions where Qwen2.5-7B-Base or Qwen2.5-7B-Instruct achieves 100 % Best-Of-2 score without given any context. These questions likely represent common knowledge already internalized within the models’ memories. 80,000 samples from the HotpotQA training split are processed through this pipeline and approximately 50% of the data are filtered out. We chose the frist 32,768 samples of processed data as our training set.

We then apply a similar approach to synthesize 128 samples from the HotpotQA validation set. For extrapolation performance testing, we synthesize test sets with different context lengths using the same pipeline. The number of wiki items ranges from 50 up to 6400, corresponding to context lengths of approximately 7K to 3.5M tokens.

# A.4 EVALUATION

We extract answers from the model outputs using regular expressions, and we prompt the model to respond in the specified format. The chosen format is ’the answer is ANSWER.’

We employ the sub em score for all benchmarks. This means that an answer is considered correct if it contains all the elements of the ground truth. When an answer consists of multiple parts and the expected response should include all of them, the score corresponds to the proportion of correct parts provided.

Before evaluating the answers, we normalize both the ground truth and the extracted responses. For example, we remove definite articles, ignore case distinctions, and apply similar standard normalization steps following previous work (Wan et al., 2025; Hsieh et al., 2024; Yen et al., 2024).

Table 6 shows the generation configurations of baseline models.

Table 6: Generation configurations of baseline models. 

<table><tr><td>Model</td><td>Context Length</td><td>Input/Output Tokens</td></tr><tr><td>QwenLong-L1 (Wan et al., 2025)</td><td>128K</td><td>120,000 / 10,000</td></tr><tr><td>Qwen2.5-Instruct-1M Series (Yang et al., 2025)</td><td>1M</td><td>990,000 / 10,000</td></tr><tr><td>DeepSeek-R1-Distill-Qwen Series (Guo et al., 2025)</td><td>128K</td><td>120,000 / 10,000</td></tr><tr><td>Qwen2.5-Instruct Series(Yang et al., 2024)</td><td>128K</td><td>120,000 / 10,000</td></tr></table>

NIAH niah single $\left\{ 1 , 2 , 3 \right\}$ in RULER (Hsieh et al., 2024) benchmark are used in our test. The yaml configuration of RULER are presented in 7. In level 1, the ”haystack” consists of repetitive sentences, and the ”needle” is a seven-digit number associated with a magic word. For level 2, the ”haystack” is composed of longer essays. Level 3 goes a step further than Level 2 where the ”needle” is a 36-character UUID string. Question and context are concated as the input of LLMs. We omit the answer prefix provided in original RULER benchmark since it is not compatible with MEMAGENT workflow.

```yaml
niah_single_1:
    task: niah
    args:
    type_haystack: repeat
    type_needle_k: words
    type_needle_v: numbers
    num_needle_k: 1
    num_needle_v: 1
    num_needle_q: 1
niah_single_2:
    task: niah
    args:
    type_haystack: essay
    type_needle_k: words
    type_needle_v: numbers
    num_needle_k: 1
    num_needle_v: 1
    num_needle_q: 1
niah_single_3:
    task: niah
    args:
    type_haystack: essay
    type_needle_k: words
    type_needle_v: uuids
    num_needle_k: 1
    num_needle_v: 1
    num_needle_q: 1 
```  
Table 7: Synthetic Configuration used for NIAH task.

# B COMPUTATION COMPLEXITY

We adopt the floating-point operations (FLOP) estimator for the Qwen2Model from verl Sheng et al. (2024) to compute the FLOP cost of both the baseline model and our proposed method. The results are shown in Figure 10. The baseline model exhibits an ${ \mathrm { O } } ( n ^ { 2 } )$ complexity, while MEMAGENT achieves an O(n) complexity.

![](images/5a5b925d721af9fb415a3752a0c0d49958ee6ce7714dc1bef76b51eb3b58e8d3.jpg)

<details>
<summary>line</summary>

| Context Length | Baseline       | Memory Agent  |
| -------------- | -------------- | ------------- |
| 8K             | 0.0            | 0.0           |
| 16K            | 0.0            | 0.0           |
| 32K            | 0.0            | 0.0           |
| 64K            | 0.0            | 0.0           |
| 128K           | 0.0            | 0.0           |
| 256K           | 0.0            | 0.0           |
| 512K           | 0.0            | 0.0           |
| 1M             | 0.1            | 0.0           |
| 2M             | 0.5            | 0.0           |
| 4M             | 2.2e19         | 0.0           |
</details>

Figure 10: Floating point operations across context lengths from 8K to 4M

For the baseline model, the number of tokens required to process is $\displaystyle q + c + o ,$ where q represents the length for the problem, c is the context length and o represents the output length.

For MEMAGENT, total FLOP cost is the sum of the FLOPs from all stages. The detailed stages involved are as follows:

• Initializing: In the first stage, the model processes an input consisting of $q + 2 0 0 + o ,$ where 200 represents a constant added to prompt the model to follow the MEMAGENT workflow.   
• Memory Updating: The number of repetitions is determined by $k = \left\lceil { \frac { c } { N } } \right\rceil$ , where c is the variable component of the input. Each repetition requires an input of length $\dot { q } + 2 0 0 + N + o .$   
• Final Answering: The final stage processes an input of length $q + 1 0 0 + o ,$ which includes the accumulated output from the previous steps.

We set q = 1024, o = 1024, N = 5000 and c is ranging from 8K to 4M to calculate the final result.

# C CASE STUDY

To further illustrate the proposed memory mechanism in detail, we conduct a case study on a generation trajectory of MEMAGENT-14B. The input question is: The director of the romantic comedy ‘Big Stone Gap’ is based in what New York city? This a 2-hop question with the following relevant Wikipedia entries:

1) Big Stone Gap is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani.   
2) Adriana Trigiani is an Italian American best-selling author of sixteen books, television writer, film director, and entrepreneur based in Greenwich Village, New York City.

In the first round, the model is presented with the entry Ghost, which refers to a production team also based in New York. The model chooses to retain this potentially useful information for future use. In the second round, no relevant context is provided; nevertheless, the model maintains its agent state, demonstrating robustness against distraction. In the third round, both relevant entries are presented. The model correctly identifies critical information and updates its memory accordingly, leading to the correct answer: Greenwich Village, New York City. At this point, the reasoning process is complete. In the remaining rounds, the model’s memory remains unchanged and is used to produce the final response.

# CHUNK1

...

Document 20: Ghost (production team)

Ghost is a Swedish record producing and songwriting team, composed of Ulf Lindstrom and ¨ Johan Ekhe, based in ´ New York City.

...

# MEMORY1

The provided section does not contain any information about the director of the romantic comedy ”Big Stone Gap” or their base in New York City. However, it does mention a production team called ”Ghost” which is based in New York City. Since the problem is specifically asking about the director of ”Big Stone Gap,” and there is no direct information about the director in the given section, we cannot infer anything directly related to the problem from the provided text.

Updated memory:

• Ghost is a Swedish record producing and songwriting team, composed of Ulf Lindstrom and Johan Ekh ¨ e, based in New York City. They are perhaps best known ´ for writing and producing Swedish singer Robyn’s three first studio albums.

The problem remains unanswered based on the provided section. Additional information would be needed to determine the director of ”Big Stone Gap” and their base in New York City.

# CHUNK2

...(No relevant document)

# MEMORY2

The provided section does not contain any information about the director of the romantic comedy ”Big Stone Gap” or their base in New York City. 2 Therefore, the updated memory remains unchanged with respect to the problem at hand. The memory still notes that the Ghost record producing and songwriting team is based in New York City, but it does not provide any information about the director of ”Big Stone Gap.”

Updated memory:

• Ghost is a Swedish record producing and songwriting team, composed of Ulf Lindstrom and Johan Ekh ¨ e, based in New York City. They are perhaps best known ´ for writing and producing Swedish singer Robyn’s three first studio albums.   
• The problem remains unanswered based on the provided section. Additional information would be needed to determine the director of ”Big Stone Gap” and their base in New York City.

# CHUNK3

Document 81: Big Stone Gap (film)

Big Stone Gap is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani and produced by Donna Gigliotti for Altar Identity Studios, a subsidiary of Media Society.

Document 96: Adriana Trigiani

Adriana Trigiani is an Italian American best-selling author of sixteen books, television writer, film director, and entrepreneur based in Greenwich Village, New York City.

# MEMORY3

Updated memory:

• Ghost is a Swedish record producing and songwriting team, composed of Ulf Lindstrom and Johan Ekh ¨ e, based in New York City. They are perhaps best known ´ for writing and producing Swedish singer Robyn’s three first studio albums.   
• Adriana Trigiani is an Italian American best-selling author of sixteen books, television writer, film director, and entrepreneur based in Greenwich Village, New York City. Trigiani has published a novel a year since 2000.   
• Big Stone Gap is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani and produced by Donna Gigliotti for Altar Identity Studios, a subsidiary of Media Society. Based on Trigiani’s 2000 best-selling novel of the same name, the story is set in the actual Virginia town of Big Stone Gap circa 1970s. The film had its world premiere at the Virginia Film Festival on November 6, 2014.

The director of the romantic comedy ”Big Stone Gap” is Adriana Trigiani, who is based in Greenwich Village, New York City.

This trajectory highlights several core capabilities learned by the memory agent: estimating potentially relevant content from query keywords and storing it preemptively, immediately updating memory upon encountering context that matches the query, and remaining unaffected by irrelevant information. Notably, these memory behaviors are not the result of architectural attention mechanisms, but emerge as text generation abilities reinforced through RL.

# D SUPPLEMENTARY EXPERIMENTS

# D.1 ABLATION STUDY ON CONTEXT SIZE AND MEMORY SIZE

In this ablation experiments, we keep the total length of memory size + context chunk size constant, and linearly adjusted the memory size. The goal is to control the total context length per conversation turn. The results are shown in Table 8 and Table 9.

Table 8: Ablation result of memory-size and context-size on NIAH. All number indicates the averaging score of level1˜level3 

<table><tr><td>Method</td><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td><td>256K</td><td>512K</td></tr><tr><td>14B-m4096-c1928</td><td>100.00</td><td>99.22</td><td>98.70</td><td>97.66</td><td>98.96</td><td>99.22</td><td>97.92</td></tr><tr><td>14B-m3072-c2952</td><td>99.74</td><td>100.00</td><td>100.00</td><td>100.00</td><td>98.70</td><td>99.48</td><td>99.48</td></tr><tr><td>14B-m2048-c3976</td><td>99.74</td><td>99.48</td><td>99.48</td><td>100.00</td><td>100.00</td><td>97.13</td><td>97.66</td></tr><tr><td>14B-m1024-c5000</td><td>99.74</td><td>98.70</td><td>100.00</td><td>100.00</td><td>98.96</td><td>97.40</td><td>98.18</td></tr><tr><td>7B-m4096-c1928</td><td>100.00</td><td>99.22</td><td>98.44</td><td>96.35</td><td>98.96</td><td>98.18</td><td>97.14</td></tr><tr><td>7B-m3072-c2952</td><td>99.48</td><td>100.00</td><td>99.74</td><td>99.48</td><td>96.88</td><td>98.18</td><td>96.09</td></tr><tr><td>7B-m2048-c3976</td><td>99.74</td><td>99.48</td><td>99.48</td><td>99.48</td><td>98.96</td><td>94.53</td><td>94.53</td></tr><tr><td>7B-m1024-c5000</td><td>100.00</td><td>98.70</td><td>99.74</td><td>98.96</td><td>97.40</td><td>97.66</td><td>96.62</td></tr></table>

Table 9: Ablation result of memory-size and context-size on Longbench-QA 

<table><tr><td>Method</td><td>2Wiki</td><td>HQA</td><td>MuSiQue</td><td>NQA</td><td>Qasper</td><td>AVG</td></tr><tr><td>14B-m4096-c1928</td><td>74.5</td><td>72.5</td><td>48.5</td><td>21.5</td><td>25.5</td><td>48.5</td></tr><tr><td>14B-m3072-c2952</td><td>76.5</td><td>70.5</td><td>52.5</td><td>24.5</td><td>26.5</td><td>50.1</td></tr><tr><td>14B-m2048-c3976</td><td>74.5</td><td>71.5</td><td>49.5</td><td>23.0</td><td>27.0</td><td>49.1</td></tr><tr><td>14B-m1024-c5000</td><td>79.0</td><td>73.0</td><td>52.0</td><td>25.0</td><td>26.0</td><td>51.0</td></tr><tr><td>7B-m4096-c1928</td><td>70.0</td><td>66.0</td><td>45.5</td><td>19.0</td><td>26.0</td><td>45.3</td></tr><tr><td>7B-m3072-c2952</td><td>72.0</td><td>64.0</td><td>44.0</td><td>20.0</td><td>25.5</td><td>45.1</td></tr><tr><td>7B-m2048-c3976</td><td>75.0</td><td>69.0</td><td>43.5</td><td>23.0</td><td>26.0</td><td>47.3</td></tr><tr><td>7B-m1024-c5000</td><td>74.0</td><td>69.5</td><td>47.0</td><td>21.5</td><td>29.0</td><td>48.2</td></tr></table>

# D.2 AGENT BASELINES

We compare MemAgent against an advanced memory-agent method, Mem0(Chhikara et al., 2025). The Mem0 paper also reports that RAG methods using only top-1 or top-2 retrieval form strong and stable baselines for memory-agent tasks. Therefore, we conduct extensive comparisons against RAG agents under multiple configurations.

For Mem0, we use SOTA OpenAI models, GPT-5.1 and text-embedding-3-large as langugae model and embedding model respectively and we follow the official GitHub repository for memory updating and retrieval. Specifically, during memory creation, we split and processed the entire context in 5,000-token chunks; during retrieval, we selected the top 30 memories.

For RAG, we also use text-embedding-3-large as embedding model and configure it with various chunk size and top-K value.

The results show that MEMAGENT outperforms these methods, demonstrating that end-to-end RL–trained memory provides greater flexibility and coherence compared with retrieval-based strategies.

Table 10: Result versus RAG Agent in RULER-HQA with different top-K settings. We segment the context based on natural semantic units, i.e., each wiki item was treated as a chunk. 

<table><tr><td rowspan="2">Model</td><td colspan="10">Length</td></tr><tr><td>7K</td><td>14K</td><td>28K</td><td>56K</td><td>112K</td><td>224K</td><td>448K</td><td>896K</td><td>1.75M</td><td>3.5M</td></tr><tr><td colspan="11">RAG + Qwen2.5-14B</td></tr><tr><td> $K=2$ </td><td>57.03</td><td>54.69</td><td>51.56</td><td>54.69</td><td>53.12</td><td>50.00</td><td>52.34</td><td>49.22</td><td>48.44</td><td>48.44</td></tr><tr><td> $K=4$ </td><td>66.41</td><td>67.19</td><td>68.75</td><td>67.19</td><td>66.41</td><td>64.06</td><td>66.41</td><td>64.84</td><td>60.94</td><td>59.38</td></tr><tr><td> $K=6$ </td><td>72.66</td><td>75.78</td><td>75.78</td><td>74.22</td><td>69.53</td><td>71.88</td><td>73.44</td><td>67.19</td><td>65.62</td><td>66.41</td></tr><tr><td> $K=8$ </td><td>78.12</td><td>78.91</td><td>77.34</td><td>81.25</td><td>76.56</td><td>78.12</td><td>77.34</td><td>74.22</td><td>70.31</td><td>64.84</td></tr><tr><td>RL-MEMAGENT-14B</td><td>80.47</td><td>82.03</td><td>82.03</td><td>83.59</td><td>81.25</td><td>77.34</td><td>79.69</td><td>75.78</td><td>78.91</td><td>71.09</td></tr><tr><td colspan="11">RAG + Qwen2.5-7B</td></tr><tr><td> $K=2$ </td><td>53.91</td><td>54.69</td><td>53.12</td><td>51.56</td><td>54.69</td><td>51.56</td><td>52.34</td><td>49.22</td><td>48.44</td><td>46.09</td></tr><tr><td> $K=4$ </td><td>67.19</td><td>66.41</td><td>66.41</td><td>67.19</td><td>64.84</td><td>64.06</td><td>62.50</td><td>61.72</td><td>60.94</td><td>59.38</td></tr><tr><td> $K=6$ </td><td>74.22</td><td>73.44</td><td>72.66</td><td>73.44</td><td>70.31</td><td>73.44</td><td>70.31</td><td>67.19</td><td>65.62</td><td>65.62</td></tr><tr><td> $K=8$ </td><td>75.00</td><td>75.00</td><td>75.78</td><td>74.22</td><td>74.22</td><td>77.34</td><td>72.66</td><td>68.75</td><td>64.06</td><td>64.84</td></tr><tr><td>RL-MEMAGENT-7B</td><td>81.25</td><td>81.25</td><td>82.03</td><td>80.47</td><td>79.69</td><td>75.78</td><td>76.56</td><td>74.22</td><td>77.34</td><td>71.88</td></tr></table>

Table 11: Result versus RAG Agent in Longbench-QA with different top-K and Context size settings. We segment the context using fixed-length chunks. For retrieval, we performed top-k matching using cosine similarity scores. 

<table><tr><td>Method</td><td>2Wiki</td><td>HQA</td><td>MuSiQue</td><td>NQA</td><td>Qasper</td><td>AVG</td></tr><tr><td colspan="7">Qwen2.5-14B + RAG</td></tr><tr><td>C=1024 K=2</td><td>51.50</td><td>56.50</td><td>26.50</td><td>15.00</td><td>23.50</td><td>28.83</td></tr><tr><td>C=1024 K=4</td><td>70.00</td><td>64.50</td><td>34.50</td><td>17.50</td><td>27.00</td><td>35.58</td></tr><tr><td>C=1024 K=6</td><td>71.50</td><td>64.00</td><td>41.00</td><td>19.00</td><td>27.50</td><td>37.17</td></tr><tr><td>C=1024 K=8</td><td>72.50</td><td>64.50</td><td>39.00</td><td>17.50</td><td>26.00</td><td>36.58</td></tr><tr><td>C=2048 K=2</td><td>58.50</td><td>61.50</td><td>33.50</td><td>13.50</td><td>25.50</td><td>32.08</td></tr><tr><td>C=2048 K=4</td><td>76.00</td><td>64.00</td><td>36.00</td><td>18.50</td><td>25.00</td><td>36.58</td></tr><tr><td>C=2048 K=6</td><td>73.00</td><td>67.50</td><td>41.50</td><td>21.00</td><td>26.50</td><td>38.25</td></tr><tr><td>C=2048 K=8</td><td>77.50</td><td>68.50</td><td>42.00</td><td>21.00</td><td>27.50</td><td>39.42</td></tr><tr><td>RL-MemAgent-14B</td><td>79.0</td><td>73.0</td><td>52.00</td><td>25.00</td><td>26.00</td><td>51.00</td></tr><tr><td colspan="7">Qwen2.5-7B + RAG</td></tr><tr><td>C=1024 K=2</td><td>41.00</td><td>48.50</td><td>22.00</td><td>14.50</td><td>25.50</td><td>25.25</td></tr><tr><td>C=1024 K=4</td><td>49.00</td><td>56.50</td><td>28.00</td><td>17.00</td><td>28.50</td><td>29.83</td></tr><tr><td>C=1024 K=6</td><td>54.50</td><td>57.50</td><td>29.50</td><td>17.00</td><td>25.00</td><td>30.58</td></tr><tr><td>C=1024 K=8</td><td>50.50</td><td>59.00</td><td>29.50</td><td>18.00</td><td>25.00</td><td>30.33</td></tr><tr><td>C=2048 K=2</td><td>49.50</td><td>51.50</td><td>19.50</td><td>12.50</td><td>27.00</td><td>26.67</td></tr><tr><td>C=2048 K=4</td><td>50.50</td><td>53.00</td><td>26.00</td><td>17.00</td><td>25.50</td><td>28.67</td></tr><tr><td>C=2048 K=6</td><td>50.50</td><td>56.50</td><td>27.50</td><td>22.00</td><td>25.50</td><td>30.33</td></tr><tr><td>C=2048 K=8</td><td>50.50</td><td>58.00</td><td>25.50</td><td>21.00</td><td>27.00</td><td>30.33</td></tr><tr><td>RL-MemAgent-7B</td><td>74.00</td><td>69.50</td><td>47.00</td><td>21.50</td><td>29.00</td><td>48.20</td></tr></table>

# E LLM USAGE

In this section, we report the usage of LLMs in this work. Some sentences in this manuscript are drafted or refined by LLMs, but all text is finalized by human authors. In the experimental process, LLMs assist with code completion, but they do not produce novel ideas or complete experiments.

# F FALIURE PATTERN STUDY

# F.1 INFORMATION OVERWRITTEN

In this example, the model accumulates a large amount of irrelevant memories in the early stage (Turn 60). When the crucial context appears, the model does capture the relevant information (Turn 289), but attempts to append it to the end (Turn 290), where it is truncated due to insufficient memory. Later, the model proactively performs a summarization (Turn 317), which gives it more space to store the second key piece of context (Turn 433). However, since the information from the first context has already been lost, the model incorrectly relies on irrelevant information for reasoning and fails to answer the question correctly.

Question: Where is the company that Sachin Warrier worked for as a software engineer headquartered?

# Relevant Information:

(shown in Turn 289) Tata Consultancy Services Limited (TCS): An Indian multinational information technology (IT) service, consulting and business solutions company. It is a subsidiary of the Tata Group, operates in 46 countries, and is headquartered in Mumbai, Maharashtra.

(shown in Turn 433) Sachin Warrier: A playback singer and composer in the Malayalam cinema industry from Kerala. He became notable with the song ”Muthuchippi Poloru” from the film Thattathin Marayathu, made his debut with Malarvaadi Arts Club, and was once working as a software engineer at Tata Consultancy Services in Kochi (later resigned to focus on music). His latest work is as a composer for the movie Aanandam.

Answer: Mumbai

# Memory Diff between Turn 59 and Turn 60

-The provided section does not contain any information about Sachin Warrier or his work as a software engineer. Therefore, the updated memory remains unchanged with respect to Sachin Warrier. However, it does provide information about Texas Monthly, which is headquartered in Downtown Austin, Texas. Since the problem is specifically asking about a company where Sachin Warrier worked as a software engineer, and no information about Sachin Warrier is provided, the updated memory remains as follows: +The provided section does not contain any information about Sachin Warrier or his work as a software engineer. However, it does provide information about Procter & Gamble, which is an American consumer goods corporation headquartered in downtown Cincinnati, Ohio, United States of America. Since the problem is specifically asking about a company where Sachin Warrier worked as a software engineer, and no information about Sachin Warrier is provided, the updated memory remains as follows:

• Pizza Pizza Ltd. is a franchised Canadian pizza fast-food restaurant, with its headquarters in Etobicoke, Toronto. Its restaurants are mainly in the province of Ontario. Other locations operate in Quebec, Nova Scotia, and in western Canada. Franchises in western Canada are mostly run through Alberta-based subsidiary Pizza 73, and in non-traditional locations such as university campuses and movie theatres throughout Canada. It has over 500 locations, including over 150 non-traditional locations.   
• TI Automotive develops, manufactures, and supplies automotive fluid storage, carrying, and delivery systems. The company has 23,000 employees at 120 locations in 29 countries, and supplies all of the world’s major automobile manufacturers. TI Automotive serves the automotive aftermarket through Bundy, Walbro, and Marwal brands. The company’s headquarters are located in Oxford, UK, with Corporate Offices based in Auburn Hills, Michigan. In 2015 TI Automotive was acquired by Bain Capital.

(...)

# Memory Diff between Turn 282 and Turn 289

-The provided section does not contain any information about Sachin Warrier or the company he worked for as a software engineer. Therefore, the updated memory remains as follows:

+The provided section contains information about Tata Consultancy Services Limited (TCS), which is an Indian multinational information technology (IT) service, consulting and business solutions company. It states that TCS is headquartered in Mumbai, Maharashtra.

+Updated memory:

• Cosmos (formerly Cosmos Tours) is a UK independent tour operator providing a range of package holidays to the UK market. The tour operator is connected to the international Globus Travel Group, founded in 1928, which remains family owned with headquarters in Lugano, Switzerland. Under parent company Cosmos Tours Ltd, the tour operator offers holidays through three brands: Cosmos, Archers Holidays and Avalon Waterways.   
• L&F Jones Holdings Ltd is a company based in Westfield, Somerset, England, that includes a convenience store chain, a wholesale food business, and a Best Western PLUS hotel. The store chain comprises twelve shops in Somerset, Wiltshire, Dorset, South Gloucestershire, and Bristol.   
(...)

Round Table Pizza is a large chain of pizza parlors in the western United States. The company’s headquarters are located in Concord, California.

Round Table Pizza is a large chain of pizza parlors in the western United States. The company’s

• Anton Media Group, formerly Anton Community

# Memory Diff between Turn 289 and Turn 290

-The provided section contains information about Tata Consultancy Services Limited (TCS), which is an Indian multinational information technology (IT) service, consulting and business solutions company. It states that TCS is headquartered in Mumbai, Maharashtra.

+The provided section does not contain any information about Sachin Warrier or Tata Consultancy Services Limited (TCS). Therefore, the updated memory remains as follows:

-Updated memory:

• Cosmos (formerly Cosmos Tours) is a UK independent tour operator providing a range of package holidays to the UK market. The tour operator is connected to the international Globus Travel Group, founded in 1928, which remains family owned with headquarters in Lugano, Switzerland. Under parent company Cosmos Tours Ltd, the tour operator offers holidays through three brands: Cosmos, Archers Holidays and Avalon Waterways.   
• L&F Jones Holdings Ltd is a company based in Westfield, Somerset, England, that includes a convenience store chain, a wholesale food business, and a Best Western PLUS hotel. The store chain comprises twelve shops in Somerset, Wiltshire, Dorset, South Gloucestershire, and Bristol.   
(...)

• Tata Consultancy Services Limited (TCS), an Indian multinational information technology (

# Memory Diff between Turn 316 and Turn 317

The provided section does not contain any information relevant to Sachin Warrier or Tata Consultancy Services Limited (TCS). Therefore, the updated memory remains as follows:

(formerly Cosmos Tours) is a UK independent tour operator providing a range of package holidays to the UK market. The tour operator is connected to the international Globus Travel Group, founded in 1928, which remains family owned with headquarters in Lugano, Switzerland. Under parent company Cosmos Tours Ltd, the tour operator offers holidays through three brands: Cosmos, Archers Holidays and Avalon Waterways.   
• L&F Jones Holdings Ltd is a company based in Westfield, Somerset, England, that includes a convenience store chain, a wholesale food business, and a Best Western PLUS hotel. The store chain comprises twelve shops in Somerset, Wiltshire, Dorset, South Gloucestershire, and Bristol. (...)   
• The company’s corporate headquarters is in Faridabad, Haryana, India.   
• Diageo plc ( or ) is a British multinational alcoholic beverages company, with its headquarters in London, England. It (...)

Since the problem is specifically about the headquarters of the company Sachin Warrier worked for as a software engineer, and the previous memory already states that Tata Consultancy Services Limited (TCS) is headquartered in Faridabad, Haryana, India, the updated memory remains as follows:

\- The company’s corporate headquarters is in Farid

# Memory Diff between Turn 430 and Turn 433

-The provided section does not contain any information about Sachin Warrier or his employment as a software engineer. Therefore, the updated memory remains as follows:

+The provided section contains information about Sachin Warrier, which is useful for answering the problem. Specifically, it states:

\+ Sachin Warrier is a playback singer and composer in the Malayalam cinema industry from Kerala.

\+ He was working as a software engineer in Tata Consultancy Services in Kochi.

\+ Later he resigned from the job to concentrate more on music.

+However, the section does not provide the headquarters information for Tata Consultancy Services. Given the previous memory, we know that the corporate headquarters of Tata Consultancy Services is in Faridabad, Haryana, India. Therefore, the updated memory is as follows:

• The company’s corporate headquarters is in Faridabad, Haryana, India.   
• SpaceX is an American aerospace manufacturer and space transport services company headquartered in Hawthorne, California. (...)   
Sachin Warrier’s company headquarters information is still unknown from the given documents.   
• Sachin Warrier was working as a software engineer in Tata Consultancy Services in Kochi. Later he resigned from the job to concentrate more on music.   
• AVG Technologies is a security software company headquartered in Amsterdam, Netherlands. (...)

-The updated memory reflects that the headquarters of the companies mentioned are in Faridabad, Haryana, India; Hawthorne, California; Amsterdam, Netherlands; Eagan, Minnesota; Pittsburgh, Pennsylvania; and San Antonio, Texas. However, there is no information about the company Sachin Warrier worked for as a software engineer. Therefore, the memory remains unchanged in terms of Sachin Warrier’s company headquarters information.

+The updated memory confirms that Sachin Warrier worked as a software engineer for Tata Consultancy Services in Kochi, and the headquarters of Tata Consultancy Services is in Faridabad, Haryana, India.

# F.2 MISSING CRITICAL INFORMATION

In this example, when the model encounters the first key piece of information (Turn 143), it fails to recognize it as such because this is a multi-hop question and the prerequisite evidence has not yet been observed. Consequently, even though the model becomes aware of the problem upon seeing the second piece of information (Turn 215), it still fails to produce the correct final answer.

Question: who is the younger brother of The episode guest stars of The Hard Easy ?

# Relevant Information:

(shown in Turn 143) Brian Doyle-Murray Brian Doyle-Murray (born Brian Murray, October 31, 1945) is an American actor, voice artist, comedian and screenwriter. He is the older brother of actor/comedian Bill Murray, and the two have acted together in several films...

(shown in Turn 215) The Hard Easy (Adventure Time) ”The Hard Easy” is the twenty-third episode... The episode guest stars Brian Doyle-Murray as Prince Huge and Jonathan Katz as the Mudscamp elder.

Answer: Bill Murray

# Memory Diff between Turn 142 and Turn 143

-The provided section does not contain any information related to ”The Hard Easy” or its guest stars. Therefore, the updated memory remains as follows:

+The provided section does not contain any information related to ”The Hard Easy” or its guest stars. However, it does provide information about Brian Doyle-Murray, who is the older brother of actor/comedian Bill Murray. Since the problem is asking for the younger brother of the episode guest stars of ”The Hard Easy,” and the memory does not contain any information about the guest stars of ”The Hard Easy,” the information about Brian Doyle-Murray and Bill Murray is not directly relevant to the problem at hand.

+Therefore, the updated memory remains as follows:

• (...)   
• Martin Lee Truex Jr. (born June 29, 1980) is an American championship professional stock car racing driver. His younger brother, Ryan, is a champion in the K&N Pro Series East division and currently races in the Camping World Truck Series. (...)

# Memory Diff between Turn 185 and Turn 215

-The provided section does not contain any information related to ”The Hard Easy” or its guest stars. Therefore, the updated memory remains unchanged with respect to the problem at hand. The only relevant piece of information is that the younger brother of Martin Lee Truex Jr. is Ryan Truex.

+The provided section contains information about ”The Hard Easy,” an episode of ”Adventure Time,” which includes the guest stars Brian Doyle-Murray as Prince Huge and Jonathan Katz as the Mudscamp elder. However, it does not provide any information about the younger brother of the episode guest stars.

-Thus, the updated memory is as follows:

+The updated memory is as follows:

• Martin Lee Truex Jr. (born June 29, 1980) is an American championship professional stock car racing driver. His younger brother, Ryan, is a champion in the K&N Pro Series East division and currently races in the Camping World Truck Series. (...)

The younger brother of Martin Lee Truex Jr. is Ryan Truex.

+The provided section does not contain any information about the younger brother of the guest stars of ”The Hard Easy.” Therefore, the updated memory remains unchanged with respect to the problem at hand. The only relevant piece of information is that the younger brother of Martin Lee Truex Jr. is Ryan Truex.

# F.3 PRIMACY BIAS

This example involves a primacy bias. The word country can refer either to a nation or to the countryside. The model initially assumes it refers to a nation (Turn 1) and subsequently reinforces this belief when it encounters additional information about countries (Turn 617). As a result, even though it eventually also attends to the truly critical piece of information (Turn 728), it still fails to answer the question correctly.

Question: Brown State Fishing Lake is in a country that has a population of how many inhabitants ?

# Relevant Information:

(shown in Turn 728) Brown County, Kansas Brown County (county code BR) is a county located in the northeast portion of the U.S. state of Kansas. As of the 2010 census, the county population was 9,984. Its county seat and most populous city is Hiawatha. Brown County is the location of the Kickapoo Indian Reservation of Kansas, the majority of the Sac and Fox Reservation and the majority of the Iowa Reservation of Kansas and Nebraska.

Answer: 9,984

# Memory Diff between Turn 0 and Turn 1

-The provided section does not contain any information about the population of the country where Brown State Fishing Lake is located. However, it does confirm that Brown State Fishing Lake is in Brown County, Kansas in the United States.   
+The provided section does not contain any information about the population of the United States or any country. Therefore, the updated memory remains as follows:

• Brown State Fishing Lake is a protected area in Brown County, Kansas in the United States.   
• The lake is 62 acres (0.25 km2) in area and up to 13 feet (4 m) deep. (...)

-Since the problem is asking for the population of the country where Brown State Fishing Lake is located, and the only country mentioned in the text is the United States, we can infer that the question is seeking the population of the United States. However, the text does not provide the exact population number.

-Given the information available, the updated memory should reflect the following:

+Since the problem is seeking the population of the country where Brown State Fishing Lake is located, and the only country mentioned is the United States, the updated memory reflects that the population of the United States is the required information. However, the text does not provide the exact population number. Therefore, the updated memory retains the above information without changes related to the population.

# Memory Diff between Turn 616 and Turn 617

-The provided section contains information about Kunming, a city in Yunnan Province, Southwest China, which has a population of 6,626,000 as of 2014. However, this information is not directly related to the problem at hand, which is about the population of the country where Brown State Fishing Lake is located.

+... Since the problem is asking for the population of the country where Brown State Fishing Lake is located, and we know that Brown State Fishing Lake is in Brown County, Kansas, in the United States, we can infer that the population of the United States is the answer to the problem.

-The updated memory remains consistent with the previous information, but it now includes the new information about Kunming:

+Updated memory:

• Brown State Fishing Lake is a protected area in Brown County, Kansas in the United States.   
• The lake is 62 acres (0.25 km2) in area and up to 13 feet (4 m) deep. (...)   
• Ruzhou, a city in Henan province, China, has more than 100,000 inhabitants. (...)   
• As of 2018, the United States had a population of around 327,167,434.

+The problem is asking for the population of the country where Brown State Fishing Lake is located, which is the United States. Therefore, the population of the United States is around 327,167,434 as of 2018.

# Memory Diff between Turn 654 and Turn 728

The provided section does not contain any information relevant to the problem at hand. Therefore, the updated memory remains as follows:

• Brown State Fishing Lake is a protected area in Brown County, Kansas in the United States.   
• Brown County, Kansas, as of the 2010 census, had a population of 9,984. (...)   
• As of 2018, the United States had a population of around 327,167,434.

The problem is asking for the population of the country where Brown State Fishing Lake is located, which is the United States. Therefore, the population of the United States is around 327,167,434 as of 2018.