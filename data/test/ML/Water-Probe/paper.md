# CAN WATERMARKED LLMS BE IDENTIFIED BY USERS VIA CRAFTED PROMPTS?

Aiwei Liu1 , Sheng Guan2, Yiming Liu1, Leyi Pan1, Yifei Zhang3, Liancheng Fang4, Lijie Wen1 ∗ Philip S. Yu4 Xuming Hu5

1 Tsinghua University 2 Beijing University of Posts and Telecommunications

3 The Chinese University of Hong Kong 4 University of Illinois at Chicago

5 Hongkong University of Science and Technology (Guangzhou)

liuaw20@mails.tsinghua.edu.cn, guansheng2022@bupt.edu.cn, wenlj@tsinghua.edu.cn

§[Official]:https://github.com/THU-BPM/Watermarked\_LLM\_Identification

# ABSTRACT

Text watermarking for Large Language Models (LLMs) has made significant progress in detecting LLM outputs and preventing misuse. Current watermarking techniques offer high detectability, minimal impact on text quality, and robustness to text editing. However, current researches lack investigation into the imperceptibility of watermarking techniques in LLM services. This is crucial as LLM providers may not want to disclose the presence of watermarks in real-world scenarios, as it could reduce user willingness to use the service and make watermarks more vulnerable to attacks. This work investigates the imperceptibility of watermarked LLMs. We design the first unified identification method called Water-Probe that identifies all kinds of watermarking in LLMs through well-designed prompts. Our key motivation is that current watermarked LLMs expose consistent biases under the same watermark key, resulting in similar differences across prompts under different watermark keys. Experiments show that almost all mainstream watermarking algorithms are easily identified with our well-designed prompts, while Water-Probe demonstrates a minimal false positive rate for non-watermarked LLMs. Finally, we propose that the key to enhancing the imperceptibility of watermarked LLMs is to increase the randomness of watermark key selection. Based on this, we introduce the Water-Bag strategy, which significantly improves watermark imperceptibility by merging multiple watermark keys.

# 1 INTRODUCTION

The rapid advancement of large language models (LLMs) has led to remarkable achievements in tasks such as question answering (Zhuang et al., 2024), programming (Jiang et al., 2024b), and reasoning (Wei et al., 2022), with widespread applications across various scenarios. However, the extensive use of LLMs has also raised concerns regarding copyright protection and misuse. Recent research indicates that malicious attackers can steal LLMs through model extraction techniques (Yao et al., 2024), and some users may abuse LLMs to generate and spread harmful information (Wei et al., 2024).

Text watermarking techniques for LLMs have become an important method to mitigate the above issues by adding detectable features to LLM outputs (Liu et al., 2024b). Recent researches on LLM watermarking have focused on improving watermark detectability (Kirchenbauer et al., 2023a), minimizing impact on generated text (Aaronson & Kirchner, 2022), and enhancing robustness against text modifications (Liu et al., 2024a). However, no work has considered the imperceptibility of watermarked LLMs, i.e., whether users can know if an LLM service is watermarked. In real-world scenarios, LLM service providers may not disclose the existence of watermarks, as it could reduce user willingness to use the service and make it more vulnerable to attacks (Sadasivan et al., 2023). As more LLM services consider implementing watermarks, it is crucial to investigate whether users can identify watermarked LLMs solely through crafted prompts.

![](images/f7839605b0edea37adab877d2b5b2c8904cb84def10a43b4d9315e86fd46c0f7.jpg)

<details>
<summary>flowchart</summary>

Diagram illustrating how LLM blocks highlight key values and their similarity with a fixed watermark, grouped into two prompts with repeated sampling.
</details>

Figure 1: Illustration of our Water-Probe algorithm for identifying watermarked LLMs. We first construct two prompts with similar output distributions, then sample repeatedly using two fixed watermark keys for each prompt. The presence of a watermark is determined by comparing the similarity of distribution differences between the two prompts. Details in §3.

Some studies focus on the imperceptibility of watermarked text, ensuring watermarked and nonwatermarked texts are indistinguishable (Hu et al., 2023; Wu et al., 2023b). However, even if individual watermarked texts are imperceptible, the distribution of numerous watermarked texts may reveal whether the LLM is watermarked, especially when repeatedly sampling with the same watermark key (Wu et al., 2024). While some studies explore cracking watermarks using large volumes of watermarked text (Jovanovic et al. ´ , 2024; Sadasivan et al., 2023; Wu & Chandrasekaran, 2024), they assume the LLM is watermarked and cannot determine if the LLM is watermarked. The most relevant work is Gloaguen et al. (2024), which proposes a black-box detection method for watermarked LLMs. However, their approach uses different detection methods for different watermarks and cannot effectively detect all watermarking techniques.

In this work, we propose Water-Probe, the first unified method for identifying watermarked LLMs that can detect all types of watermarks embedded during the LLM’s text generation process. (see related work section for this type of watermarking) Our motivation stems from a key observation: all current LLM watermarking algorithms expose consistent bias when repeatedly sampled under the same watermark key. Based on this, our Water-Probe algorithm first crafts prompts to perform repeated sampling under the same watermark key, then compares the consistency of sampling distribution differences across different prompts under a pair of watermark keys. Highly consistent differences indicate a watermarked LLM.

In our experiments, we demonstrate that the Water-Probe algorithm achieves high accuracy in detecting various types of watermarked LLMs. We also show its applicability across different LLMs, maintaining a low false positive rate for non-watermarked LLMs. Furthermore, our algorithm exhibits robust performance across different sampling methods and temperature settings.

Finally, we explore methods to enhance the imperceptibility of watermarked LLMs. We find that increasing the randomness of watermark key selection is crucial, as it makes it more difficult to construct prompts for repeated sampling using the same key. Based on this, we propose the Water-Bag algorithm, which combines multiple watermark keys into one, randomly selecting a key for each generation and choosing the highest score for detection. Although increasing key selection randomness often leads to a slight decrease in detectability, it significantly enhances the imperceptibility of watermarked LLMs. Addressing this trade-off should be an important direction for future work.

Our main contributions are summarized as follows:

• We propose Water-Probe, the first unified algorithm that can detect various types of watermarked LLMs by analyzing the consistency of sampling distribution differences across different prompts under fixed watermark keys.   
• Through extensive experiments, we demonstrate that Water-Probe achieves high detection accuracy across different LLMs, watermarking methods, and sampling settings, while maintaining a low false positive rate for non-watermarked LLMs.

• We introduce Water-Bag, a novel algorithm that enhances LLM watermark imperceptibility by combining multiple watermark keys, and analyze the trade-off between watermark detectability and imperceptibility.

# 2 PRELIMINARIES

Definition 1 (Large Language Model). An LLM M is a function that, given an input x and a partial output sequence $y _ { 1 : i - 1 }$ , produces a probability distribution $P _ { M } ( y _ { i } | x , y _ { 1 : i - 1 } )$ over possible next tokens yi. The model generates complete outputs by iteratively sampling from these distributions.

Since this work focuses solely on watermarks embedded in LLM services, we will only consider the watermark during generation type mentioned in $\ S 6 .$ . The definition of the watermark rule is given below.

Definition 2 (Watermark Rule). A watermark rule is typically a function F that adjusts the current LLM’s predicted probability distribution based on a watermark key k to obtain a new probability distribution. Formally, given an LLM $P _ { M }$ and a key k, the watermark rule F modifies the distribution as follows:

$$
P _ {M} ^ {F} \left(y _ {i} \mid x, y _ {1: i - 1}, k\right) = F \left(P _ {M} \left(y _ {i} \mid x, y _ {1: i - 1}\right), k\right) \tag {1}
$$

where $P _ { M } ^ { F }$ is the modified probability distribution for the next token $y _ { i }$

The main difference between watermarking algorithms lies in how they determine the watermark key. Based on this, we categorize watermarking algorithms into n-gram based watermarking and fixed-key-list based watermarking. We will now introduce these two types of watermarking algorithms.

Definition 3 (N-Gram Based Watermarking). In n-gram based watermarking, the watermark key $k _ { i }$ for generating the current token yi is determined by a function f that takes two inputs:

$$
k _ {i} = f (K, y _ {i - n: i - 1}) \tag {2}
$$

where K is a pre-selected master key, and $y _ { i - n : i - 1 }$ represents the previous n tokens.

N-gram based watermarking ensures that for the same n-token prefix, the watermark key for generating the next token remains consistent. This approach is widely used in current watermarking algorithms, including KGW (Kirchenbauer et al., 2023a), KGW-V2 (Kirchenbauer et al., 2023b), Aar (Aaronson & Kirchner, 2022), DiPmark (Wu et al., 2023b), and SIR (Liu et al., 2024a). Next, we define fixed-key-list-based watermarking:

Definition 4 (Fixed-Key-List Based Watermarking). Let $K = \{ k _ { 1 } , k _ { 2 } , . . . , k _ { m } \}$ be a fixed key list. For a given starting index $s \in \{ 1 , . . . , m \}$ , the watermark key ki for generating the i-th token is:

$$
k _ {i} = k _ {\left((s + i - 1) \bmod m\right) + 1} \tag {3}
$$

where the starting index s may be randomly chosen for each generation process.

This approach is employed in algorithms such as Exp-Edit (Kuditipudi et al., 2023), where keys are used sequentially from a potentially random starting position in the key list.

To formalize our approach, we define our goal as follows:

Definition 5 (Black-box Watermark Identification). A function ${ \mathcal { D } } : P _ { M } \to \{ 0 , 1 \}$ that classifies a language model $P _ { M }$ as watermarked (1) or not (0), without access to its internal parameters.

# 3 WATERMARKED LLM IDENTIFICATION

# 3.1 WHY WATERMARKED LLM IDENTIFICATION IS POSSIBLE

Definition 2 implies that LLMs typically introduce some distortion to the distribution. However, there exist distortion-free watermarking algorithms that satisfy Definition 2. We first provide a definition and then demonstrate that LLMs with distortion-free watermarks can still be identified.

Definition 6 (Distortion-Free Watermark). A watermarking algorithm is considered distortion-free if, for all possible inputs x and partial output sequences $y _ { 1 : i - 1 }$ , the expected output distribution of the watermarked model $P _ { M } ^ { F }$ over all possible watermark keys $k \in \mathcal { K }$ is identical to the original model PM : $P _ { M } .$

$$
\mathbb {E} _ {k \in \mathcal {K}} [ P _ {M} ^ {F} (y _ {i} | x, y _ {1: i - 1}, k) ] = P _ {M} (y _ {i} | x, y _ {1: i - 1}). \tag {4}
$$

This equation indicates that the expected output distribution of watermarked text remains unchanged when the watermark key is randomly selected across all possible keys. However, as shown in Equation 1, sampling with a specific watermark key k introduces a difference between $P _ { M } ^ { F } ( y _ { i } | x , y _ { 1 : i - 1 } , \mathbf { \dot { k } } )$ and $P _ { M } ( y _ { i } | x , y _ { 1 : i - 1 } )$ . This observation leads to the following theorem on detectability of watermarked LLMs:

Observation 1 (Distributional Difference of Watermarked LLMs). Let $P _ { M }$ be a language model and F a watermark rule as defined in Definition 2. For a given watermark key $k ,$ the probability distribution of the watermarked model $\bar { P } _ { M } ^ { F } ( y _ { i } | x , y _ { 1 : i - 1 } , k )$ differs from the original distribution $P _ { M } ( y _ { i } | x , y _ { 1 : i - 1 } )$ . This distributional difference suggests the potential detectability of the watermark.

# 3.2 PIPELINE OF WATERMARKED LLM IDENTIFICATION

Observation 1 implies that the key to identifying a watermarked LLM is to construct prompts that allow for multiple samplings using the same watermark key to reveal the difference between $P _ { M } ^ { F } ( y _ { i } | x , y _ { 1 : i - 1 } , k )$ and $P _ { M } ( y _ { i } | x , y _ { 1 : i - 1 } )$ ). However, due to the black-box setting, we cannot directly access the origin logits $P _ { M } ( y _ { i } | x , y _ { 1 : i - 1 } )$ . Instead, we calculate the difference in LLM outputs for two distinct keys, defined as $\Breve { \Delta } ( x , \bar { k } _ { m } , \bar { k _ { n } } ) = P _ { M } ^ { F } ( \cdot | x , k _ { m } ) - P _ { M } ^ { F } ( \cdot | x , k _ { n } )$ , where we use $P _ { M } ^ { \bar { F } } ( \cdot | x , k )$ to represent the output distribution.

If two prompts yield similar output distributions, the same watermark key should have similar effects on both prompts (proven in Theorem 1). We determine if an LLM contains a watermark by comparing the consistency of the effects of two watermark keys on two similar prompts. Specifically, given $x _ { 1 }$ $x _ { 2 } , k _ { 1 }$ , and $k _ { 2 } .$ , we assess the similarity between $\Delta ( x _ { 1 } , k _ { 1 } , k _ { 2 } )$ and $\Delta ( x _ { 2 } , k _ { 1 } , k _ { 2 } )$ . High similarity indicates the presence of a watermark; otherwise, we conclude there is no watermark. Based on the above analysis, we now present the process of the Water-Probe algorithm.

Step 1: Construct highly correlated prompts. Construct N prompts $x _ { 1 } , x _ { 2 } , . . . , x _ { N }$ such that their output probability distributions under M are highly similar, which can be expressed as:

$$
\forall i, j \in \{1, 2,..., N \}, \mathrm{KL} (P _ {M} (\cdot | x _ {i}) | | P _ {M} (\cdot | x _ {j})) \leq \epsilon \text {   and   } x _ {i} \neq x _ {j} \tag {5}
$$

where $\mathrm { K L } ( \cdot | | \cdot )$ is the Kullback-Leibler divergence, $P _ { M } ( \cdot | x _ { i } )$ is the output probability distribution for prompt $x _ { i }$ under the LLM M , and ϵ is a small threshold indicating high similarity between distributions.

Step 2: Sampling with simulated fixed watermark keys. Since we cannot access the logits under a given watermark key, we need to use repeated sampling to estimate the distribution. We construct a set of simulated watermark keys $\boldsymbol { K } \dot { = } \{ k _ { 1 } , k _ { 2 } , . . . , \bar { k } _ { m } \}$ based on our prompt design (detailed in subsequent sections). For each prompt $x _ { i }$ and each simulated key $\bar { k _ { j } } \ \in \bar { K }$ , we estimate the distribution as follows:

$$
\hat {P} _ {M} ^ {F} (y | x _ {i}, k _ {j}) = \frac {1}{W} \sum_ {w = 1} ^ {W} \mathbf {1} _ {y _ {i, j} ^ {w} = y}, \quad \text { where } y _ {i, j} ^ {w} \sim P _ {M} ^ {F} (y | x _ {i}, k _ {j}) \tag {6}
$$

where W is the sample count, $\mathbf { 1 } _ { A }$ is the indicator function, and $y _ { i , j } ^ { w }$ is the w-th sample sampled from $P _ { M } ^ { F } ( y | x _ { i } , k _ { j } )$ . Specific prompt techniques for different watermarking algorithms will be detailed later. Note that our prompt design for simulating watermark keys assumes the target LLM has a watermark. If it doesn’t, then $\begin{array} { r } { { \cal P } _ { M } ^ {  } ( y | x _ { i } , k _ { j } ) = { \cal P } _ { M } ^ {  } ( y | x _ { i } ) } \end{array}$ for all simulated keys.

Step 3: Analyze Cross-Prompt Watermark Consistency. We first assume that the watermark rule satisfies Lipschitz continuity. Based on this assumption, we can deduce that the differences in output distributions produced by a watermark key pair for highly correlated prompts are similar.

Assumption 1 (Lipschitz Continuity of Watermark Rule). For prompts $x _ { 1 }$ and $x _ { 2 }$ satisfying the similarity condition in Equation 5, the watermark rule F satisfies Lipschitz continuity. That $i s ,$ there exists a constant $L > 0$ such that for the probability distributions $P _ { M } ( \cdot | x _ { 1 } ) , P _ { M } ( \cdot | x _ { 2 } )$ and any watermark key $k \in { \mathcal { K } }$ :

$$
\left\| F \left(P _ {M} (\cdot | x _ {1}), k\right) - F \left(P _ {M} (\cdot | x _ {2}), k\right) \right\| _ {1} \leq L \cdot \left\| P _ {M} (\cdot | x _ {1}) - P _ {M} (\cdot | x _ {2}) \right\| _ {1}. \tag {7}
$$

Theorem 1 (Consistency of Watermark Effect). Let $x _ { 1 }$ and x2 be two different prompts satisfying the similarity condition in Equation 5. Let $k _ { 1 }$ and $k _ { 2 }$ be two randomly sampled watermark keys from the key space $\kappa .$ . The effect of applying these keys on the output distribution should be highly consistent across prompts:

$$
\mathbb {E} _ {k _ {1}, k _ {2} \sim \mathcal {K}} [ \operatorname{Sim} (P _ {M} ^ {F} (\cdot | x _ {1}, k _ {1}) - P _ {M} ^ {F} (\cdot | x _ {1}, k _ {2}), P _ {M} ^ {F} (\cdot | x _ {2}, k _ {1}) - P _ {M} ^ {F} (\cdot | x _ {2}, k _ {2})) ] \geq \rho \tag {8}
$$

where $P _ { M } ^ { F }$ is the watermarked distribution, and $S i m ( \cdot , \cdot )$ is a similarity measure (e.g., cosine similarity), and ρ should be a constant significantly greater than 0.

Theorem 1 implies that for prompts with similar output distributions under the LLM, the expected differences introduced by the two watermark keys should also be similar. The formal proof is provided in Appendix A.

Based on Theorem 1, we calculate the average similarity using the estimated distributions from Step 2. To ensure stability across different sampling temperatures, we first apply a rank transformation. Specifically, for a token $y ,$ its rank is defined as the number of tokens with probability greater than or equal to that of $y ,$ denoted as $R ( P ( y | x ) ) = | \{ y ^ { \prime } \in V : P ( y ^ { \prime } | x ) \geq P ( y | x ) \}$ |. We then compute the expected similarity:

$$
\bar {S} = \frac {1}{N} \sum_ {x _ {i} \neq x _ {j} \in \mathcal {X}} \sum_ {k _ {m} \neq k _ {n} \in \mathcal {K}} \operatorname{Sim} (\Delta_ {R} (x _ {i}, k _ {m}, k _ {n}), \Delta_ {R} (x _ {j}, k _ {m}, k _ {n})) \tag {9}
$$

Here, X is the prompt set, K is the watermark key set, $N = | \mathcal { X } | ( | \mathcal { X } | - 1 ) | \mathcal { K } | ( | \mathcal { K } | - 1 )$ , and $\Delta _ { R } ( x , k _ { m } , k _ { n } ) = R ( \hat { P } _ { M } ^ { F } ( \cdot | x , k _ { m } ) ) - R ( \hat { P } _ { M } ^ { F } ( \cdot | x , k _ { n } ) )$ . We verify the importance of rank transformation in Appendix G.

According to Theorem 1, if M contains a watermark, the similarity obtained from Equation 9 should be significantly greater than 0. If M does not contain a watermark, we assume $P _ { M } ^ { F } \dot { = } P _ { M }$ for any $k ,$ so Equation 9 should represent the similarity between two random vectors with zero mean, which should be close to 0. A detailed analysis of the no-watermark case is provided in Appendix B.

Based on this, we design the following z-test to perform hypothesis testing on the average similarity:

$$
z = (\bar {S} - \mu) / \sigma \tag {10}
$$

where $\bar { S }$ is the observed average similarity, σ is the standard deviation of the ${ \bar { S } } ,$ and $\mu$ is the mean of the $\bar { S }$ under the no-watermark case. Theoretically, $\mu$ should be 0 for an unwatermarked LLM. However, in practice, we may choose a value slightly greater than 0 to account for potential biases introduced by our prompt construction method or other factors. The standard deviation σ is estimated through multiple experiments.

We reject the null hypothesis (no watermark) and conclude the LLM is likely watermarked if: $z > z _ { \alpha } ,$ , where $z _ { \alpha }$ is the chosen significance level α. In this work, we consider a z-score between 4 and 10 as moderate confidence, and above 10 as high confidence.

# 3.3 CONSTRUCTING REPEATED SAMPLING WITH SAME WATERMARK KEY

In the previous section, we introduced the basic pipeline of our Water-Probe algorithm. A key challenge is constructing prompts that enable multiple samplings with the same watermark key. This varies for different watermarking algorithms. We’ll discuss approaches for n-gram based and fixed-key-list based methods.

For N-gram based watermarking (Definition 3), since the watermark key is derived from the previous N tokens, we can design prompts that make the LLM generate N irrelevant tokens before following the prompt. An example is provided below:

![](images/236df42d6f4fbf611e372be573b3d62b7af5313b8c7a02972b3667cc25f4451a.jpg)  
Figure 2: Distribution of start keys for identical prefixes in Exp-Edit watermarking. Analysis based on prompts described in Section 3.3 for Watermark-Probe-v2. Each subplot represents a specific prefix(in title).

# Prompt 1: Example Prompt for Watermark-Probe-v1

Please generate abcd before answering the question.

Question: Name a country with a large population.

Answer: abcd India

In the example above, we assume generating abcd does not affect the distribution for answering the question. However, in practice, it’s challenging to ensure completely irrelevant tokens. Consequently, the S¯ for an unwatermarked LLM constructed this way may be slightly above 0. We refer to the Watermark-Probe using prompts similar to the above table as Watermark-Probe-v1.

For fixed-key-set based watermarking, since the start watermark key is randomly selected each time, our approach is to approximate multiple samplings with the same watermark key by exploiting the correlation between the watermark key and the generated tokens. Specifically, we prompt the LLM to perform some quasi-random generation initially. Generally, the same watermark start key will only generate a few fixed sampling results, so we can assume that identical sampling result prefixes are generated by the same watermark key. Here’s a specific example of how we construct prompts for this approach:

# Prompt 2: Example Prompt for Watermark-Probe-v2

Please generate a sentence that satisfies the following conditions: The first word is randomly sampled from A-Z. The second word is randomly sampled from zero to nine. The third word is randomly sampled from cat, dog, tiger and lion. Then answer the question: Name a country with a large population.

Answer: A one cat China

As shown in the example above, we first prompt the LLM to generate N (3 in this case) random tokens before answering the question. Different random token combinations typically correspond to a few watermark keys. Figure 2 illustrates the distribution of watermark key counts for varying numbers of random tokens. As evident from the figure, given a fixed prefix, the vast majority of cases utilize a specific key. So this approach thus approximates sampling with the same watermark key. Similarly, we refer to the prompting method in the above table as Watermark-Probe-v2.

We provide the detailed steps of the Water-Probe algorithm in Algorithm 1 in the appendix.

# 4 EXPERIMENT ON WATERMARKED LLM IDENTIFICATION

# 4.1 EXPERIMENT SETUP

Tested Watermarking Algorithms: We evaluated a diverse range of LLM watermarking algorithms, including N-Gram based watermarking and Fixed-Key-List based watermarking. For N-Gram based watermarking, we tested KGW (Kirchenbauer et al., 2023a) $( \gamma = 0 . 5 , \delta = 2 )$ , Aar (Aaronson & Kirchner, 2022) (N = 1), KGW-Min Kirchenbauer et al. (2023b) (window size of 4), KGW-Skip (Kirchenbauer et al., 2023b) (window of 3), DiPMark (Wu et al., 2023b) (α = 0.45), and γ reweighting (Hu et al., 2023). For Fixed-Key-List based watermarking, we examined EXP-Edit (Kuditipudi et al., 2023) and ITS-Edit (Kuditipudi et al., 2023), both with a key length of 420. Details of these algorithms are provided in Appendix F.

Table 1: Detection similarities for various LLMs with and without different watermarks, calculated using Equation 9 and our two identification methods: Water-Probe-v1 and Water-Probe-v2. indicates high-confidence watermark identification and indicates low-confidence watermark identification while no color indicates no watermark identified. The corresponding z-scores can be found in Table 7 in the Appendix. 

<table><tr><td rowspan="2">LLM</td><td colspan="7">N-Gram</td><td colspan="2">Fixed-Key-List</td></tr><tr><td>Non</td><td>KGW</td><td>Aar</td><td>KGW-Min</td><td>KGW-Skip</td><td>DiPmark</td><td> $\gamma$ -Reweight</td><td>EXP-Edit</td><td>ITS-Edit</td></tr><tr><td colspan="10">Water-Probe-v1 (w. prompt 1)</td></tr><tr><td>Qwen2.5-1.5B</td><td> ${0.02} \pm {0.02}$ </td><td> ${0.37} \pm {0.02}$ </td><td> ${0.88} \pm {0.06}$ </td><td> ${0.37} \pm {0.02}$ </td><td> ${0.39} \pm {0.01}$ </td><td> ${0.55} \pm {0.01}$ </td><td> ${0.55} \pm {0.01}$ </td><td> ${0.01} \pm {0.02}$ </td><td> ${0.00} \pm {0.04}$ </td></tr><tr><td>OPT-2.7B</td><td> ${0.05} \pm {0.01}$ </td><td> ${0.47} \pm {0.01}$ </td><td> ${0.91} \pm {0.01}$ </td><td> ${0.42} \pm {0.02}$ </td><td> ${0.45} \pm {0.01}$ </td><td> ${0.60} \pm {0.01}$ </td><td> ${0.61} \pm {0.01}$ </td><td> ${0.08} \pm {0.02}$ </td><td> ${0.09} \pm {0.01}$ </td></tr><tr><td>Llama-3.2-3B</td><td> ${0.04} \pm {0.02}$ </td><td> ${0.53} \pm {0.01}$ </td><td> ${0.90} \pm {0.01}$ </td><td> ${0.48} \pm {0.00}$ </td><td> ${0.49} \pm {0.01}$ </td><td> ${0.61} \pm {0.01}$ </td><td> ${0.61} \pm {0.01}$ </td><td> ${0.03} \pm {0.01}$ </td><td> ${0.04} \pm {0.01}$ </td></tr><tr><td>Qwen2.5-3B</td><td> ${0.03} \pm {0.01}$ </td><td> ${0.33} \pm {0.02}$ </td><td> ${0.75} \pm {0.05}$ </td><td> ${0.33} \pm {0.02}$ </td><td> ${0.38} \pm {0.00}$ </td><td> ${0.51} \pm {0.01}$ </td><td> ${0.53} \pm {0.01}$ </td><td> ${0.03} \pm {0.01}$ </td><td> ${0.06} \pm {0.02}$ </td></tr><tr><td>Llama2-7B</td><td> ${0.02} \pm {0.01}$ </td><td> ${0.42} \pm {0.01}$ </td><td> ${0.87} \pm {0.01}$ </td><td> ${0.31} \pm {0.01}$ </td><td> ${0.42} \pm {0.01}$ </td><td> ${0.56} \pm {0.01}$ </td><td> ${0.56} \pm {0.04}$ </td><td> ${0.03} \pm {0.02}$ </td><td> ${0.02} \pm {0.00}$ </td></tr><tr><td>Mixtral-7B</td><td> ${0.01} \pm {0.02}$ </td><td> ${0.41} \pm {0.01}$ </td><td> ${0.85} \pm {0.02}$ </td><td> ${0.37} \pm {0.01}$ </td><td> ${0.41} \pm {0.02}$ </td><td> ${0.57} \pm {0.01}$ </td><td> ${0.58} \pm {0.03}$ </td><td> ${0.00} \pm {0.00}$ </td><td> ${0.02} \pm {0.02}$ </td></tr><tr><td>Qwen2.5-7B</td><td> ${0.07} \pm {0.04}$ </td><td> ${0.41} \pm {0.02}$ </td><td> ${0.82} \pm {0.02}$ </td><td> ${0.34} \pm {0.03}$ </td><td> ${0.38} \pm {0.02}$ </td><td> ${0.43} \pm {0.03}$ </td><td> ${0.43} \pm {0.02}$ </td><td> ${0.06} \pm {0.01}$ </td><td> ${0.04} \pm {0.02}$ </td></tr><tr><td>Llama-3.1-8B</td><td> ${0.01} \pm {0.02}$ </td><td> ${0.41} \pm {0.02}$ </td><td> ${0.85} \pm {0.02}$ </td><td> ${0.41} \pm {0.01}$ </td><td> ${0.39} \pm {0.01}$ </td><td> ${0.57} \pm {0.02}$ </td><td> ${0.58} \pm {0.00}$ </td><td> ${0.02} \pm {0.02}$ </td><td> ${0.00} \pm {0.01}$ </td></tr><tr><td>Llama2-13B</td><td> ${0.01} \pm {0.03}$ </td><td> ${0.41} \pm {0.01}$ </td><td> ${0.86} \pm {0.01}$ </td><td> ${0.31} \pm {0.02}$ </td><td> ${0.40} \pm {0.02}$ </td><td> ${0.58} \pm {0.02}$ </td><td> ${0.60} \pm {0.01}$ </td><td> ${0.02} \pm {0.01}$ </td><td> ${0.02} \pm {0.03}$ </td></tr><tr><td>Average</td><td>0.029</td><td>0.418</td><td>0.854</td><td>0.371</td><td>0.412</td><td>0.553</td><td>0.505</td><td>0.031</td><td>0.032</td></tr><tr><td colspan="10">Water-Probe-v2 (w. prompt 2)</td></tr><tr><td>Qwen2.5-1.5B</td><td> ${0.02} \pm {0.02}$ </td><td> ${0.30} \pm {0.01}$ </td><td> ${0.83} \pm {0.01}$ </td><td> ${0.29} \pm {0.01}$ </td><td> ${0.27} \pm {0.02}$ </td><td> ${0.49} \pm {0.02}$ </td><td> ${0.52} \pm {0.03}$ </td><td> ${0.39} \pm {0.03}$ </td><td> ${0.60} \pm {0.00}$ </td></tr><tr><td>OPT-2.7B</td><td> ${0.04} \pm {0.03}$ </td><td> ${0.29} \pm {0.02}$ </td><td> ${0.88} \pm {0.01}$ </td><td> ${0.23} \pm {0.01}$ </td><td> ${0.19} \pm {0.02}$ </td><td> ${0.42} \pm {0.01}$ </td><td> ${0.43} \pm {0.03}$ </td><td> ${0.43} \pm {0.01}$ </td><td> ${0.62} \pm {0.00}$ </td></tr><tr><td>Llama-3.2-3B</td><td> ${0.00} \pm {0.01}$ </td><td> ${0.31} \pm {0.01}$ </td><td> ${0.89} \pm {0.01}$ </td><td> ${0.33} \pm {0.00}$ </td><td> ${0.24} \pm {0.01}$ </td><td> ${0.51} \pm {0.01}$ </td><td> ${0.54} \pm {0.01}$ </td><td> ${0.52} \pm {0.01}$ </td><td> ${0.84} \pm {0.00}$ </td></tr><tr><td>Qwen2.5-3B</td><td> ${0.03} \pm {0.02}$ </td><td> ${0.35} \pm {0.04}$ </td><td> ${0.78} \pm {0.01}$ </td><td> ${0.29} \pm {0.02}$ </td><td> ${0.28} \pm {0.01}$ </td><td> ${0.45} \pm {0.02}$ </td><td> ${0.45} \pm {0.02}$ </td><td> ${0.39} \pm {0.02}$ </td><td> ${0.71} \pm {0.00}$ </td></tr><tr><td>Llama2-7B</td><td> ${0.04} \pm {0.02}$ </td><td> ${0.34} \pm {0.01}$ </td><td> ${0.82} \pm {0.02}$ </td><td> ${0.33} \pm {0.01}$ </td><td> ${0.28} \pm {0.01}$ </td><td> ${0.50} \pm {0.01}$ </td><td> ${0.51} \pm {0.02}$ </td><td> ${0.48} \pm {0.01}$ </td><td> ${0.81} \pm {0.00}$ </td></tr><tr><td>Mixtral-7B</td><td> ${0.09} \pm {0.01}$ </td><td> ${0.34} \pm {0.04}$ </td><td> ${0.83} \pm {0.01}$ </td><td> ${0.29} \pm {0.02}$ </td><td> ${0.24} \pm {0.01}$ </td><td> ${0.51} \pm {0.01}$ </td><td> ${0.53} \pm {0.00}$ </td><td> ${0.42} \pm {0.02}$ </td><td> ${0.81} \pm {0.00}$ </td></tr><tr><td>Qwen2.5-7B</td><td> $- {0.01} \pm {0.04}$ </td><td> ${0.26} \pm {0.02}$ </td><td> ${0.70} \pm {0.00}$ </td><td> ${0.28} \pm {0.02}$ </td><td> ${0.23} \pm {0.01}$ </td><td> ${0.32} \pm {0.03}$ </td><td> ${0.35} \pm {0.02}$ </td><td> ${0.32} \pm {0.02}$ </td><td> ${0.73} \pm {0.00}$ </td></tr><tr><td>Llama-3.1-8B</td><td> ${0.01} \pm {0.00}$ </td><td> ${0.31} \pm {0.01}$ </td><td> ${0.77} \pm {0.01}$ </td><td> ${0.29} \pm {0.02}$ </td><td> ${0.26} \pm {0.00}$ </td><td> ${0.50} \pm {0.01}$ </td><td> ${0.51} \pm {0.01}$ </td><td> ${0.43} \pm {0.01}$ </td><td> ${0.71} \pm {0.00}$ </td></tr><tr><td>Llama2-13B</td><td> ${0.01} \pm {0.02}$ </td><td> ${0.35} \pm {0.01}$ </td><td> ${0.82} \pm {0.02}$ </td><td> ${0.26} \pm {0.02}$ </td><td> ${0.26} \pm {0.01}$ </td><td> ${0.50} \pm {0.01}$ </td><td> ${0.53} \pm {0.01}$ </td><td> ${0.44} \pm {0.02}$ </td><td> ${0.73} \pm {0.00}$ </td></tr><tr><td>Average</td><td>0.026</td><td>0.317</td><td>0.813</td><td>0.288</td><td>0.250</td><td>0.467</td><td>0.486</td><td>0.424</td><td>0.729</td></tr></table>

Tested LLMs: To comprehensively evaluate our algorithm’s effectiveness, we tested a diverse range of LLMs with varying parameter sizes, including Qwen2.5-1.5B (Hui et al., 2024), OPT-2.7B (Zhang et al., 2022), Llama3.2-3B (Meta AI, 2024), Qwen2.5-3B, Llama2-7B (Touvron et al., 2023), Mixtral-7B (Jiang et al., 2024a), Qwen2.5-7B, Llama-3.1-8B (Dubey et al., 2024), and Llama2-13B (Touvron et al., 2023).We evaluated our Water-Probe algorithm on all LLMs, testing its performance under various watermarking schemes and in scenarios without watermarks.

Watermark-Probe Settings: For our Watermark-Probe algorithm, detailed prompts are provided in the appendix C. To calculate the z-score, we repeat each detection experiment 3 times to compute the standard deviation. We set µ = 0.1 for our experiments.

# 4.2 MAIN RESULTS

In Table 1, we present the average similarity and standard deviation obtained using Watermark-Probe-v1 and Watermark-Probe-v2 algorithms for identifying various LLMs under different watermarking conditions and non-watermarked scenarios. For all LLMs, the sampling temperature was set to 1, with the number of samples set to 104. As evident from Table 1, Watermark-Probe-v1 demonstrates high effectiveness for N-gram based watermarking but is not applicable to fixed-key-list based watermarking. In contrast, the Watermark-Probe-v2 algorithm proves effective in identifying all watermarking algorithms tested. Additionally, even watermarking algorithms claiming to be distortion-free, such as Aar (Aaronson & Kirchner, 2022), DiPMark (Wu et al., 2023b), and γ-reweighting (Hu et al., 2023), they can be effectively identified by both versions of Watermark-Probe. Furthermore, our algorithm maintains low similarity for non-watermarked LLMs, ensuring minimal false positive rates. Additionally, we calculated the average similarity for different watermarking algorithms in Table 1 to demonstrate their detection confidence. Among these, the Aar algorithm is the most easily detectable due to its pronounced perturbation for individual keys. Lastly, given the same number of samples, the Watermark-Probe-v1 algorithm yields more significant identification results for N-gram based watermarking algorithms compared to the Watermark-Probe-v2 algorithm.

![](images/bc942709dafc0fbabee4b9e15355e0bf74c5b070ffdc8b398f1a959e46d1b283.jpg)

<details>
<summary>line</summary>

| Temperature | Water-Probe-v1 (KGW) | Water-Probe-v2 (KGW) | Water-Probe-v2 (EXP-Edit) | Water-Probe-v1 (No Watermark) | Water-Probe-v2 (No Watermark) |
| ----------- | --------------------- | --------------------- | -------------------------- | ------------------------------ | ------------------------------ |
| 0.0         | 30                    | 10                    | 0                          | -10                            | -10                            |
| 0.2         | 10                    | 30                    | 10                         | -10                            | -10                            |
| 0.4         | 20                    | 10                    | 10                         | -10                            | -10                            |
| 0.6         | 10                    | 30                    | 10                         | -10                            | -10                            |
| 0.8         | 30                    | 10                    | 10                         | -10                            | -10                            |
| 1.0         | 10                    | 30                    | 10                         | -10                            | -10                            |
| 1.2         | 30                    | 10                    | 10                         | -10                            | -10                            |
| 1.4         | 30                    | 30                    | 10                         | -10                            | -10                            |
</details>

![](images/cb7a0fd5bcfcd22192329ac86f4dcde7e57f12e5d9d222824f949d97cbc1d7d9.jpg)

<details>
<summary>line</summary>

Z-score Comparison of Llama2-7B with Different Sampling Numbers
| Sampling Number | Water-Probe-v2 (KGW) | Water-Probe-v2 (EXP-Edit) | Water-Probe-v1 (KGW) | Water-Probe-v1 (EXP-Edit) | Water-Probe-v1 (No Watermark) | Water-Probe-v2 (No Watermark) |
|---|---|---|---|---|---|---|
| 1000 | 100 | 100 | 100 | 100 | 100 | 100 |
| 2000 | 100 | 100 | 100 | 100 | 100 | 100 |
| 4000 | 100 | 100 | 100 | 100 | 100 | 100 |
| 6000 | 100 | 100 | 100 | 100 | 100 | 100 |
| 8000 | 100 | 100 | 100 | 100 | 100 | 100 |
| 20000 | 100 | 100 | 100 | 100 | 100 | 100 |
| 40000 | 100 | 100 | 100 | 100 | 100 | 100 |
| 60000 | 100 | 100 | 100 | 100 | 100 | 100 |
| 80000 | 100 | 100 | 100 | 100 | 100 | 100 |
| 100000 | 100 | 100 | 100 | 100 | -1.5 | -2.5 |
Threshold (Z-score = 4)
</details>

Figure 3: The left plot shows the variation of z-scores detected by Watermark-Probe-v1 and Watermark-Probe-v2 as a function of sampling temperature. The right plot illustrates the change in z-scores detected by Watermark-Probe-v1 and Watermark-Probe-v2 with different sampling numbers.

# 4.3 FURTHER ANALYSIS

Influence of Sampling Temperature: The results in Table 1 are based on a sampling temperature of 1. To further validate the performance of Watermark-Probe under different sampling temperatures, we show the z-score changes across temperatures in the left plot of Figure 3. Using Llama2-7B as an example, as the sampling temperature increases from 0.1 to 1.5, both Watermark-Probe-v1 and Watermark-Probe-v2 can distinguish between watermarked and unwatermarked LLMs. However, at relatively low temperatures $( T ~ < ~ 0 . 5 )$ , detection of unwatermarked LLMs may show some fluctuations. Since deployed LLMs rarely use very low temperatures, our algorithm can be considered effective for detecting real-world LLM deployments.

Influence of Sampling Number: The right plot in Figure 3 illustrates the impact of the sampling number on the detected z-score. Specifically, we used Llama2-7B as the target LLM with a temperature of 1. We observed that Watermark-Probe-v2 requires more samples compared to Watermark-Probe-v1. With insufficient samples, Watermark-Probe-v2 lacks enough common prefixes to compute Equation 9. In our setting, Watermark-Probe-v1 can achieve stable detection with 1,000 samples, while Watermark-Probe-v2 requires at least 104 samples. For cases where detection is successful, the z-score of watermarked LLMs tends to increase with the number of samples, although this trend exhibits fluctuations.

# 5 ENHANCING THE IMPERCEPTIBILITY OF WATERMARKED LLMS

We have demonstrated that current watermarked LLMs can be identified by our Water-Probe method. In this section, we discuss how to improve the imperceptibility of watermarked LLMs. The core principle is to make it challenging to construct repeated sampling scenarios using two separate keys according to Equation 9.

One design is globally fixed watermark key (e.g., Unigram (Zhao et al., 2023)). While it’s easy to construct repeated sampling scenarios with a single key, we cannot detect stable deviations between different keys as only one exists globally. In the appendix E, we provide an algorithm named Water-Contrast to identify watermarks by comparing the target LLM distribution and a prior distribution. While not theoretically guaranteed (it’s challenging to determine if this bias is from watermarking or inherent to the LLM), it shows practical effectiveness. Meanwhile, Unigram watermarks are susceptible to cracking (Jovanovic et al. ´ , 2024).

The second design aims to increase the randomness of watermark key selection, making it less dependent on N-grams. This approach makes it difficult to construct repeated sampling scenarios using the same key. For Fixed-Key-List Based watermarking, a viable strategy is to increase the length of the key list. Since the initial key position is random, increasing the key list length enhances the randomness of key selection.

Additionally, for N-gram Based watermarking algorithms, we propose an enhanced strategy called Water-Bag, which combines multiple master watermark keys into a key-bag with a key inversion mechanism.

![](images/038afdddbd595a9caf4fbae37e6afd08c03c56c3d31f3c62c5230ac59d6ce7cf.jpg)

<details>
<summary>bar</summary>

| Index | Count |
| ----- | ----- |
| 1     | 6     |
| 2     | 23    |
| 3     | 25    |
| 4     | 9     |
</details>

![](images/6e3f09ca8fc63cb96672937d575476d59933d010011da15f98c4ef026cb9c6cc.jpg)

<details>
<summary>bar</summary>

G nine dog
| Index | Count |
|---|---|
| 1 | 12 |
| 2 | 20 |
| 3 | 14 |
| 4 | 21 |
| 5 | 0 |
</details>

![](images/2d29ad966ea599f3eba4a5752653539d7439c0a5ee64fa0bab50d0e8d584f266.jpg)

<details>
<summary>bar</summary>

A nine lion
| Index | Count |
|---|---|
| 1 | 9 |
| 2 | 16 |
| 3 | 7 |
| 4 | 16 |
| 5 | 0 |
</details>

![](images/a351167793f6ea9d43648bdbb39e5dbc9e28df05f67114a9b2cd0ca9f4103e87.jpg)

<details>
<summary>bar</summary>

| Index (x1e18) | Count |
| ------------- | ----- |
| 1             | 6     |
| 2             | 10    |
| 3             | 17    |
| 4             | 19    |
</details>

Figure 4: Distribution of start keys for identical prefixes in Water-Bag strategy. Analysis based on prompts described in Section 3.3 for Water-Probe-v2. Each subplot represents a specific prefix (showed in title).

Table 2: Performance comparison of Water-Bag Strategy and Exp-Edit algorithm in watermarked LLM identification and watermarked text detection. Water-Bag is evaluated with varying bag sizes, while Exp-Edit is tested with different key lengths. represents high-confidence watermark and represents low-confidence watermark. Detail of z-score could be seen in Appendix H. 

<table><tr><td rowspan="2"></td><td rowspan="2">None</td><td colspan="4">KGW w. Water-Bag</td><td colspan="3">Exp-Edit(Key-len)</td></tr><tr><td> $|K \cup \overline{K}| = 1$ </td><td> $|K \cup \overline{K}| = 2$ </td><td> $|K \cup \overline{K}| = 4$ </td><td> $|K \cup \overline{K}| = 8$ </td><td> $|K| = 420$ </td><td> $|K| = 1024$ </td><td> $|K| = 2048$ </td></tr><tr><td colspan="9">Watermarked LLM Indentification</td></tr><tr><td>Water-Probe-v1(n=3)</td><td> $0.02 \pm 0.01$ </td><td> $0.42 \pm 0.01$ </td><td> $0.05 \pm 0.01$ </td><td> $0.02 \pm 0.01$ </td><td> $0.03 \pm 0.02$ </td><td> $0.03 \pm 0.05$ </td><td> $0.02 \pm 0.01$ </td><td> $0.02 \pm 0.02$ </td></tr><tr><td>Water-Probe-v2(n=3)</td><td> $0.04 \pm 0.01$ </td><td> $0.34 \pm 0.01$ </td><td> $0.34 \pm 0.01$ </td><td> $0.25 \pm 0.01$ </td><td> $0.16 \pm 0.02$ </td><td> $0.48 \pm 0.01$ </td><td> $0.33 \pm 0.01$ </td><td> $0.23 \pm 0.02$ </td></tr><tr><td>Water-Probe-v2(n=5)</td><td> $0.06 \pm 0.06$ </td><td> $0.32 \pm 0.01$ </td><td> $0.18 \pm 0.01$ </td><td> $0.12 \pm 0.02$ </td><td> $0.07 \pm 0.01$ </td><td> $0.64 \pm 0.00$ </td><td> $0.54 \pm 0.01$ </td><td> $0.44 \pm 0.00$ </td></tr><tr><td colspan="9">Watermarked Text Detection</td></tr><tr><td>Detection-F1-score</td><td>-</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.975</td><td>1.0</td></tr><tr><td>PPL</td><td>8.15</td><td>11.93</td><td>11.85</td><td>12.17</td><td>12.50</td><td>16.63</td><td>17.28</td><td>19.06</td></tr><tr><td>Robustness (GPT3.5)</td><td>-</td><td>0.843</td><td>0.849</td><td>0.748</td><td>0.696</td><td>0.848</td><td>0.854</td><td>0.745</td></tr><tr><td>Detection-time (s)</td><td>-</td><td>0.045</td><td>0.078</td><td>0.156</td><td>0.31</td><td>37.87</td><td>108.5</td><td>194.21</td></tr></table>

Definition 7 (Water-Bag Strategy). The Water-Bag strategy extends N-gram based watermarking by using a set of master keys $K = \{ K _ { 1 } , K _ { 2 } , . . . , K _ { n } \}$ and their inversions $\mathbf { \overline { { K } } } = \{ \overline { { K _ { 1 } } } , \overline { { K _ { 2 } } } , . . . , \overline { { K _ { n } } } \}$ . For each generation, a master key $K _ { j }$ or its inversion $\overline { { K _ { j } } }$ is randomly selected:

$$
P _ {M} ^ {W B} (y _ {i} | x, y _ {1: i - 1}, K, \overline {{K}}) = F (P _ {M} (y _ {i} | x, y _ {1: i - 1}), k _ {i}), \quad k _ {i} = f (K _ {j} ^ {*}, y _ {i - n: i - 1}), \quad K _ {j} ^ {*} \sim U n i f o r m (K \cup \overline {{K}}) \tag {11}
$$

where $P _ { M } ^ { W B }$ is the modified probability distribution, $K _ { j } ^ { * }$ is randomly sampled from the combined set of original and inverted keys, and f is the watermark key derivation function. The inverted key $\overline { { K _ { j } } }$ is defined as:

$$
\frac {1}{2} \left(F \left(P _ {M} \left(y _ {i} \mid x, y _ {1: i - 1}\right), f \left(K _ {j}, y _ {i - n: i - 1}\right)\right) + F \left(P _ {M} \left(y _ {i} \mid x, y _ {1: i - 1}\right), f \left(\overline {{{K _ {j}}}}, y _ {i - n: i - 1}\right)\right)\right) = P _ {M} \left(y _ {i} \mid x, y _ {1: i - 1}\right) \tag {12}
$$

This ensures that the average effect of $K _ { j }$ and $\overline { { K _ { j } } }$ on the logits is equivalent to the original logits, which makes our Water-Probe-v1 ineffective against the Water-Bag strategy.

For the watermarked text detection for Water-Bag Strategy, we use the maximum detection score across all master keys in the bag. The text is considered watermarked if this maximum exceeds a threshold.

To validate the effectiveness of the two strategies for enhancing the imperceptibility of watermarked LLMs, we evaluated their performance in Table 2. We examined both watermarked LLM identification and watermarked text detection settings, assessing the new watermarking strategies’ detectability by our Water-Probe algorithm and their impact on watermarked text detection efficacy and performance.

For watermarked text detection, we used OPT-2.7B to generate texts on the C4 dataset (Raffel et al., 2020), using 30 tokens as prompts and generating 200 additional tokens with watermarks. PPL was calculated using Llama2-7B. To assess detection robustness, we computed the F1-score after rewriting texts using GPT-3.5. Detection time for single text was also recorded. We use the KGW algorithm as an implementation example for the water-bag strategy. For watermarked LLM identification, we report results for both Water-Probe-v1 and Water-Probe-v2. For Water-Probe-v2, we present results for $n = 3$ and $n = 5$ , where n is the number of random tokens generated as described in Section 3.3. The n = 3 setting matches Table 1, while prompts for $n = 5$ are provided in Appendix C.

Table 2 demonstrates that Water-Probe-v1 fails to effectively identify both Water-Bag and Exp-Edit algorithms. For Water-Probe-v2, detection difficulty increases with larger bag sizes in Water-Bag and longer key lengths in Exp-Edit. However, Water-Bag proves more challenging to identify. Crucially, Water-Bag’s detectability remains stable as n increases, while Exp-Edit becomes more easy to identify. We analyzed the key distribution of Water-Bag under Water-Probe-v2 for different prefixes in Figure 4. The distribution is notably more uniform compared to Exp-Edit in Figure 2, explaining why Water-Bag is relatively harder to identify. We observe that increasing watermark key randomness reduces watermark robustness for both strategies, with Exp-Edit also significantly increasing detection time. This highlights a trade-off in watermarking algorithms between key randomness and robustness (Liu et al., 2024b). Future work should focus on developing algorithms that enhance randomness without compromising robustness.

# 6 RELATED WORK

Large language model (LLM) watermarking techniques (Liu et al., 2024b) have become crucial for copyright protection (Sander et al., 2024), generated text detection (Wu et al., 2023a), and preventing misuse (Liu et al., 2024c). LLM watermarking can be broadly categorized into two types. The first type is post-processing watermarking, which modifies generated text using format-based (Sato et al., 2023), lexical-based, syntax-based (Wei et al., 2022), or generation-based (Zhang et al., 2024) approaches to add watermark. However, post-processing watermarking methods require waiting for text generation to complete before modifying and adding watermarks, which is not suitable for current LLM services that require real-time text generation. Another category of watermarking algorithms, known as watermarking during generation, typically involves adjusting the distribution of the next generated token based on a watermark key. For instance, the KGW (Kirchenbauer et al., 2023a) algorithm divides the vocabulary into red and green lists, increasing the probability of tokens in the green list. SIR (Liu et al., 2024a) further modifies logits based on semantic information, enhancing watermark robustness. An important objective of these methods is to maintain the imperceptibility of generated text, i.e., watermarked and non-watermarked text should have identical distributions, with some distortion-free algorithms showing promising results (Kuditipudi et al., 2023; Aaronson & Kirchner, 2022). However, previous work has overlooked the imperceptibility of watermarked LLMs themselves, i.e., whether external users can detect if an LLM service contains watermarks without disclosure. This work investigates the imperceptibility of watermarked LLMs. The most relevant work is (Gloaguen et al., 2024), our work differs in several aspects: we propose a unified identification method for all watermarking during generation algorithms without requiring different detection approaches for different watermarks. Our method also supports identification of complex watermark variants (like EXP-Edit with sampling), and we introduce the water-bag algorithm to enhance watermarked LLM imperceptibility.

# 7 CONCLUSION

In this paper, we pioneered the study of identifying watermarked LLMs. We first theoretically demonstrated the basis for identifying watermarked LLMs. We then designed the Water-Probe algorithm, which identifies watermarked LLMs by comparing distribution differences of similar prompts under different watermark keys. Our experiments showed that our algorithm is applicable to all N-Gram and Fixed-Key-List based Watermarking algorithms, independent of sampling temperature. We discussed scenarios where Water-Probe might fail and designed the WaterBag watermarking algorithm, which sacrifices some robustness of watermarked text detection to make watermarked LLMs harder to identify. Future work could focus on watermark concealment as a key research direction, designing more covert watermarking schemes.

# 8 ACKNOWLEDGEMENTS

This work is primarily supported by the Key Research and Development Program of China (No. 2024YFB3309702). Additional support was provided by the National Science Foundation (NSF) under grants III-2106758 and POSE-2346158. This work was also supported by the Guangdong Provincial Department of Education Project (Grant No.2024KQNCX028); Scientific Research Projects for the Higher-educational Institutions (Grant No.2024312096), Education Bureau of Guangzhou Municipality; Guangzhou-HKUST(GZ) Joint Funding Program (Grant No.2025A03J3957), Education Bureau of Guangzhou Municipality.

# REFERENCES

S. Aaronson and H. Kirchner. Watermarking gpt outputs, 2022. https://www.scottaaronson. com/talks/watermark.ppt.   
Imre Csiszár and János Körner. Information theory: coding theorems for discrete memoryless systems. Cambridge University Press, 2011.   
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
Thibaud Gloaguen, Nikola Jovanovic, Robin Staab, and Martin Vechev. Black-box detection of ´ language model watermarks. arXiv preprint arXiv:2405.20777, 2024.   
Zhengmian Hu, Lichang Chen, Xidong Wu, Yihan Wu, Hongyang Zhang, and Heng Huang. Unbiased watermark for large language models. arXiv preprint arXiv:2310.10669, 2023.   
Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.   
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024a.   
Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. A survey on large language models for code generation. arXiv preprint arXiv:2406.00515, 2024b.   
Nikola Jovanovic, Robin Staab, and Martin Vechev. Watermark stealing in large language models. ´ arXiv preprint arXiv:2402.19361, 2024.   
John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In International Conference on Machine Learning, pp. 17061–17084. PMLR, 2023a.   
John Kirchenbauer, Jonas Geiping, Yuxin Wen, Manli Shu, Khalid Saifullah, Kezhi Kong, Kasun Fernando, Aniruddha Saha, Micah Goldblum, and Tom Goldstein. On the reliability of watermarks for large language models. arXiv preprint arXiv:2306.04634, 2023b.   
Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang. Robust distortion-free watermarks for language models. arXiv preprint arXiv:2307.15593, 2023.   
Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, and Lijie Wen. A semantic invariant robust watermark for large language models. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=6p8lpe4MNf.   
Aiwei Liu, Leyi Pan, Yijian Lu, Jingjing Li, Xuming Hu, Xi Zhang, Lijie Wen, Irwin King, Hui Xiong, and Philip Yu. A survey of text watermarking in the era of large language models. ACM Computing Surveys, 2024b.   
Aiwei Liu, Qiang Sheng, and Xuming Hu. Preventing and detecting misinformation generated by large language models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, pp. 3001–3004, New York, NY, USA, 2024c. Association for Computing Machinery. ISBN 9798400704314. doi: 10.1145/3626772.3661377. URL https://doi.org/10.1145/3626772.3661377.   
Meta AI. Llama-3.2-3b. https://huggingface.co/meta-llama/Llama-3.2-3B, 2024.   
Leyi Pan, Aiwei Liu, Zhiwei He, Zitian Gao, Xuandong Zhao, Yijian Lu, Binglin Zhou, Shuliang Liu, Xuming Hu, Lijie Wen, et al. Markllm: An open-source toolkit for llm watermarking. arXiv preprint arXiv:2405.10051, 2024.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.   
Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, and Soheil Feizi. Can ai-generated text be reliably detected? arXiv preprint arXiv:2303.11156, 2023.   
Tom Sander, Pierre Fernandez, Alain Durmus, Matthijs Douze, and Teddy Furon. Watermarking makes language models radioactive. arXiv preprint arXiv:2402.14904, 2024.   
Ryoma Sato, Yuki Takezawa, Han Bao, Kenta Niwa, and Makoto Yamada. Embarrassingly simple text watermarks. arXiv preprint arXiv:2310.08920, 2023.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? Advances in Neural Information Processing Systems, 36, 2024.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F Wong, and Lidia S Chao. A survey on llm-gernerated text detection: Necessity, methods, and future directions. arXiv preprint arXiv:2310.14724, 2023a.   
Qilong Wu and Varun Chandrasekaran. Bypassing llm watermarks with color-aware substitutions. arXiv preprint arXiv:2403.14719, 2024.   
Yihan Wu, Zhengmian Hu, Hongyang Zhang, and Heng Huang. Dipmark: A stealthy, efficient and resilient watermark for large language models. arXiv preprint arXiv:2310.07710, 2023b.   
Yihan Wu, Ruibo Chen, Zhengmian Hu, Yanshuo Chen, Junfeng Guo, Hongyang Zhang, and Heng Huang. Distortion-free watermarks are not truly distortion-free under watermark key collisions. arXiv preprint arXiv:2406.02603, 2024.   
Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, and Yue Zhang. A survey on large language model (llm) security and privacy: The good, the bad, and the ugly. High-Confidence Computing, pp. 100211, 2024.   
Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, and Farinaz Koushanfar. {REMARK-LLM}: A robust and efficient watermarking framework for generative large language models. In 33rd USENIX Security Symposium (USENIX Security 24), pp. 1813–1830, 2024.   
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.   
Xuandong Zhao, Prabhanjan Ananth, Lei Li, and Yu-Xiang Wang. Provable robust watermarking for ai-generated text. arXiv preprint arXiv:2306.17439, 2023.   
Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. Toolqa: A dataset for llm question answering with external tools. Advances in Neural Information Processing Systems, 36, 2024.

# Part I

# Appendix

# Table of Contents

A Proof of Theorem 1 and Additional Analyses 14

B Statistical Analysis of Unwatermarked LLMs Identification 15

C Detailed Prompts for Simulating Watermark Keys 15

D Detailed Algorithm for Water-Probe 18

E Detection of Global-Fixed Key Watermarking in LLMs 18

F Details of Tested Watermarking Algorithms 21

G Ablation of Rank Transformation 22

H Supplementary Z-Scores and P-values 23

I Comparison with Related Work (Gloaguen et al., 2024) 23

J Guidelines for Constructing Watermarked LLM Identification Prompts 25

J.1 Overall Prompt Structure 25

J.2 Question Component Design 25

J.3 Prefix Component Design . . 25

K Threat Model 26

K.1 Detector Capabilities . . 26

K.2 Trust Assumptions . . . 27

K.3 Detection Goals and Constraints . . . 27

L Test on Closed-source Models 27

M Reversion Key Calculation of Water-Bag 27

# A PROOF OF THEOREM 1 AND ADDITIONAL ANALYSES

Proof. Goal: Prove that under the Lipschitz continuity condition, the watermark modification differences for two similar prompts $x _ { 1 }$ and $x _ { 2 }$ have high similarity, with an expected similarity of at least $\rho .$ .

Step 1: Express the watermark modification difference

For a prompt x and keys $k _ { 1 } , k _ { 2 }$ , define the watermark modification difference as:

$$
\Delta_ {x} (k _ {1}, k _ {2}) = P _ {M} ^ {F} (\cdot | x, k _ {1}) - P _ {M} ^ {F} (\cdot | x, k _ {2}) \tag {13}
$$

Step 2: Utilize the Lipschitz continuity condition

By the Lipschitz continuity condition, for any fixed key $k \colon$

$$
\left\| P _ {M} ^ {F} (\cdot | x _ {1}, k) - P _ {M} ^ {F} (\cdot | x _ {2}, k) \right\| _ {1} \leq L \cdot \left\| P _ {M} (\cdot | x _ {1}) - P _ {M} (\cdot | x _ {2}) \right\| _ {1} \tag {14}
$$

Since $x _ { 1 }$ and $x _ { 2 }$ satisfy the similarity condition KL $, ( P _ { M } ( \cdot | x _ { 1 } ) | | P _ { M } ( \cdot | x _ { 2 } ) ) \leq \epsilon$ (Equation 5), by Pinsker’s inequality (Csiszár & Körner, 2011):

$$
\left\| P _ {M} (\cdot | x _ {1}) - P _ {M} (\cdot | x _ {2}) \right\| _ {1} \leq \sqrt {2 \epsilon} \tag {15}
$$

Therefore, for any $k \in \{ k _ { 1 } , k _ { 2 } \}$ :

$$
\left\| P _ {M} ^ {F} (\cdot | x _ {1}, k) - P _ {M} ^ {F} (\cdot | x _ {2}, k) \right\| _ {1} \leq L \cdot \sqrt {2 \epsilon} \tag {16}
$$

Step 3: Analyze the difference of watermark modification differences

Consider the definition of $\Delta _ { x } ( k _ { 1 } , k _ { 2 } )$ and compute $\| \Delta _ { x _ { 1 } } ( k _ { 1 } , k _ { 2 } ) - \Delta _ { x _ { 2 } } ( k _ { 1 } , k _ { 2 } ) \| _ { 1 }$ :

$$
\begin{array}{l} \| \Delta_ {x _ {1}} (k _ {1}, k _ {2}) - \Delta_ {x _ {2}} (k _ {1}, k _ {2}) \| _ {1} = \| (P _ {M} ^ {F} (\cdot | x _ {1}, k _ {1}) - P _ {M} ^ {F} (\cdot | x _ {1}, k _ {2})) - (P _ {M} ^ {F} (\cdot | x _ {2}, k _ {1}) - P _ {M} ^ {F} (\cdot | x _ {2}, k _ {2})) \| _ {1} \\ = \| (P _ {M} ^ {F} (\cdot | x _ {1}, k _ {1}) - P _ {M} ^ {F} (\cdot | x _ {2}, k _ {1})) - (P _ {M} ^ {F} (\cdot | x _ {1}, k _ {2}) - P _ {M} ^ {F} (\cdot | x _ {2}, k _ {2})) \| _ {1} \\ \leq \| P _ {M} ^ {F} (\cdot | x _ {1}, k _ {1}) - P _ {M} ^ {F} (\cdot | x _ {2}, k _ {1}) \| _ {1} + \| P _ {M} ^ {F} (\cdot | x _ {1}, k _ {2}) - P _ {M} ^ {F} (\cdot | x _ {2}, k _ {2}) \| _ {1} \\ \leq L \cdot \sqrt {2 \epsilon} + L \cdot \sqrt {2 \epsilon} \\ = 2 L \cdot \sqrt {2 \epsilon} \\ = \delta^ {\prime} \tag {17} \\ \end{array}
$$

where $\delta ^ { \prime } = 2 L \cdot \sqrt { 2 \epsilon }$ .

Step 4: Relate $L _ { 1 }$ distance to similarity measure

Assume $\mathrm { S i m } ( \cdot , \cdot )$ is cosine similarity, which is negatively correlated with $L _ { 1 }$ distance. Since $\| \Delta _ { x _ { 1 } } ( k _ { 1 } , k _ { 2 } ) - \Delta _ { x _ { 2 } } ( k _ { 1 } , k _ { 2 } ) \| _ { 1 } \leq \delta ^ { \prime }$ , and $\delta ^ { \prime }$ is a small positive number:

$$
\operatorname{Sim} (\Delta_ {x _ {1}} (k _ {1}, k _ {2}), \Delta_ {x _ {2}} (k _ {1}, k _ {2})) \geq \rho^ {\prime} \tag {18}
$$

where $\rho ^ { \prime }$ is a lower bound dependent on $\delta ^ { \prime }$ , approaching 1 as $\delta ^ { \prime }$ decreases.

Step $\mathbf { 5 } \colon$ Calculate expected similarity

Since Sim $( \Delta _ { x _ { 1 } } ( k _ { 1 } , k _ { 2 } ) , \Delta _ { x _ { 2 } } ( k _ { 1 } , k _ { 2 } ) ) \geq \rho ^ { \prime }$ for any $k _ { 1 } , k _ { 2 } \in \mathcal { K }$ , for randomly sampled $k _ { 1 } , k _ { 2 } \colon$

$$
\mathbb {E} _ {k _ {1}, k _ {2} \sim \mathcal {K}} \left[ \operatorname{Sim} (\Delta_ {x _ {1}} (k _ {1}, k _ {2}), \Delta_ {x _ {2}} (k _ {1}, k _ {2})) \right] \geq \rho^ {\prime} \tag {19}
$$

Set $\rho = \rho ^ { \prime }$ , and by choosing a sufficiently small ϵ (making $\delta ^ { \prime }$ small enough), we can ensure $\rho$ is close to 1.

Conclusion: Under the Lipschitz continuity condition, for two similar prompts $x _ { 1 }$ and $x _ { 2 }$ , the expected similarity of their watermark modification differences under randomly sampled watermark keys $k _ { 1 }$ and $k _ { 2 }$ is at least $\rho ,$ where $\rho$ is a large positive number close to 1. This proves the high consistency of watermark effects in similar contexts, ensuring the detectability and robustness of the watermark. □

# B STATISTICAL ANALYSIS OF UNWATERMARKED LLMS IDENTIFICATION

When there’s no watermark present in the LLM, we expect the similarity measure in Equation 9 to be close to 0. This can be mathematically explained as follows:

For an unwatermarked model, $P _ { M } ^ { F } = P _ { M }$ for any k. Let $\hat { R } _ { N } ^ { ( 1 ) } ( \cdot )$ and $\hat { R } _ { N } ^ { ( 2 ) } ( \cdot )$ ) denote two independent empirical estimates of $R ( \cdot )$ using N samples each (because in the black-box setting, we cannot directly access the true $P _ { M } ^ { \dot { F } } )$ . Therefore:

$$
\begin{array}{l} \Delta_ {R} (x _ {i}, k _ {m}, k _ {n}) = \hat {R} _ {N} ^ {(1)} (P _ {M} ^ {F} (\cdot | x _ {i}, k _ {m})) - \hat {R} _ {N} ^ {(2)} (P _ {M} ^ {F} (\cdot | x _ {i}, k _ {n})) \\ = \hat {R} _ {N} ^ {(1)} (P _ {M} (\cdot | x _ {i})) - \hat {R} _ {N} ^ {(2)} (P _ {M} (\cdot | x _ {i})) \tag {20} \\ = [ \hat {R} _ {N} ^ {(1)} (P _ {M} (\cdot | x _ {i})) - R (P _ {M} (\cdot | x _ {i})) ] - [ \hat {R} _ {N} ^ {(2)} (P _ {M} (\cdot | x _ {i})) - R (P _ {M} (\cdot | x _ {i})) ] \\ = \epsilon_ {1} - \epsilon_ {2} \\ \end{array}
$$

where $\epsilon _ { 1 } , \epsilon _ { 2 }$ represent independent sampling errors that follow normal distributions ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ according to the Central Limit Theorem.

Consequently, for any $x _ { i } , x _ { j } , k _ { m } , k _ { n }$ :

$$
\operatorname{Sim} \left(\Delta_ {R} \left(x _ {i}, k _ {m}, k _ {n}\right), \Delta_ {R} \left(x _ {j}, k _ {m}, k _ {n}\right)\right) = \operatorname{Sim} \left(\epsilon_ {1} - \epsilon_ {2}, \epsilon_ {3} - \epsilon_ {4}\right) \tag {21}
$$

where $\epsilon _ { 1 } , \epsilon _ { 2 }$ represent independent sampling errors that follow normal distributions ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ according to the Central Limit Theorem.

Consequently, for any $x _ { i } , x _ { j } , k _ { m } , k _ { n }$ :

$$
\operatorname{Sim} (\Delta_ {R} (x _ {i}, k _ {m}, k _ {n}), \Delta_ {R} (x _ {j}, k _ {m}, k _ {n})) = \operatorname{Sim} (\epsilon_ {1} - \epsilon_ {2}, \epsilon_ {3} - \epsilon_ {4}) \tag {22}
$$

where $\epsilon _ { 1 } , \epsilon _ { 2 } , \epsilon _ { 3 } , \epsilon _ { 4 }$ are independent sampling errors. To understand why this similarity has an expected value of zero, recall that cosine similarity is defined as:

$$
\operatorname{Sim} (a, b) = \frac {a \cdot b}{| | a | | \cdot | | b | |} \tag {23}
$$

Note that $\left( \epsilon _ { 1 } - \epsilon _ { 2 } \right)$ and $( \epsilon _ { 3 } - \epsilon _ { 4 } )$ are differences of independent normal variables, each following ${ \mathcal { N } } ( 0 , 2 \sigma ^ { 2 } )$ . For the numerator:

$$
\begin{array}{l} E [ (\epsilon_ {1} - \epsilon_ {2}) (\epsilon_ {3} - \epsilon_ {4}) ] = E [ \epsilon_ {1} \epsilon_ {3} ] - E [ \epsilon_ {1} \epsilon_ {4} ] - E [ \epsilon_ {2} \epsilon_ {3} ] + E [ \epsilon_ {2} \epsilon_ {4} ] \\ = E \left[ \epsilon_ {1} \right] E \left[ \epsilon_ {3} \right] - E \left[ \epsilon_ {1} \right] E \left[ \epsilon_ {4} \right] - E \left[ \epsilon_ {2} \right] E \left[ \epsilon_ {3} \right] + E \left[ \epsilon_ {2} \right] E \left[ \epsilon_ {4} \right] \tag {24} \\ = 0 \\ \end{array}
$$

where the second equality follows from independence, and the final equality holds because $E [ \epsilon _ { i } ] = 0$ for all i. As these differences are independent and orthogonal in expectation, their cosine similarity has an expected value of zero:

$$
\mathbb {E} \left[ \operatorname{Sim} \left(\epsilon_ {1} - \epsilon_ {2}, \epsilon_ {3} - \epsilon_ {4}\right) \right] = 0 \tag {25}
$$

Therefore, the average similarity $\bar { S }$ in Equation 9 has an expected value of 0.

# C DETAILED PROMPTS FOR SIMULATING WATERMARK KEYS

In this section, we provide details of the prompt used for repeated sampling with the same key, as introduced in Section 3.3.

For the Water-Probe-v1 algorithm, we use the following prompt pair as shown in Prompt 3 and Prompt 4.

# Prompt3: First prompt for Fixed Key List Based Watermarking(Water-Probe-v1)

Please generate some text based on the following instructions(no other words):

First generate the prefix D seven tiger.

Then answer the question: Name a country with a large population.

Example1: D seven tiger China

Example2: D seven tiger India

Example3:

# Prompt4: First prompt for Fixed Key List Based Watermarking(Water-Probe-v1)

Please generate some text based on the following instructions(no other words):

First generate the prefix D seven tiger.

Then answer the question: Name a country with a large area.

Example1: D seven tiger China

Example2: D seven tiger India

Example3:

Here, we use a fixed prefix D seven tiger as an example. In the actual experiment, we used 50 different prefixes to simulate 50 potentially different watermark keys. For the experiment in Table 1, we performed 10,000 samplings, with each prefix sampled 200 times. The complete list of 50 prefixes is shown below.

# All the prefix for Water-Probe-v1

Y three lion, G three lion, U six lion, A eight tiger, K four cat, N seven tiger, K three cat, H five dog, E zero lion, V three dog, W five dog, K one tiger, B two tiger, E two lion, U six dog, A two tiger, D two tiger, I nine dog, F three lion, C three dog, N five cat, L two dog, K zero tiger, E five dog, B five cat, X four tiger, U three dog, K nine dog, P one dog, H zero dog, V eight tiger, S three tiger, P seven cat, S six dog, Y nine cat, J one tiger, C five tiger, A zero lion, L eight dog, X eight dog, I two dog, C eight tiger, O three tiger, L one cat, M five tiger, P five cat, F seven cat, I zero cat, P two lion, L four cat

# Prompt5: First prompt for Fixed Key List Based Watermarking(Water-Probe-v2)

Please generate some text based on the following instructions(no other words):

The first word is randomly sampled from A-Z.

The second word is randomly sampled from zero to nine.

The third word is randomly sampled from cat, dog, tiger and lion.

Then add a separator | and answer the following question: Name a country with a large population.

Example1: D seven tiger | United States

Example2: J five dog | India

Example3: R six cat | China

Example4: T one tiger | Indonesia

Example5: P seven cat | Pakistan

Example6: G six cat | Russia

Example7: R five tiger | India

Example8: L nine cat | Mexico

Example9: T four tiger | United States

Example10: H three dog | Japan

Example11: B three lion | Germany

Example12:

# Prompt6: Second prompt for Fixed Key List Based Watermarking(Water-Probe-v2)

Please generate some text based on the following instructions(no other words):

The first word is randomly sampled from A-Z.

The second word is randomly sampled from zero to nine.

The third word is randomly sampled from cat, dog, tiger and lion.

Then add a separator | and answer the following question: Name a country with a large area.

Example1: L three tiger | United States

Example2: X three cat | India

Example3: A six tiger | China

Example4: W eight lion | Argentina

Example5: D five dog | France

Example6: P one cat | Russia

Example7: E six tiger | Australia

Example8: Z eight lion | Canada

Example9: Q two tiger | United States

Example10: A nine cat | Brazil

Example11: V three dog | Russia

Example12:

Similarly, for Prompt 3 and Prompt 4, we assume that the answer spaces for the questions Name a country with a large population. and Name a country with a large area are highly correlated, i.e., countries with large areas tend to have relatively large populations. In practice, other correlated prompt pairs can be chosen, as long as they satisfy the correlation requirement.

For the Water-Probe-v2 algorithm, we use the following prompt pair shown in Prompt 5 and Prompt 6.

# Prompt7: First prompt for Fixed Key List Based Watermarking(Water-Probe-v2 N=5)

Please generate some text based on the following instructions(no other words):

The first word is randomly sampled from A-Z.

The second word is randomly sampled from zero to nine.

The third word is randomly sampled from cat, dog, tiger and lion.

The fourth word is randomly sampled from apple, banana and orange.

The fifth word is randomly sampled from car, bus and truck.

The sixth entry is the answer to the following question: Name a country with a large population.

Example1: D seven tiger apple car United States

Example2: J five dog banana bus India

Example3: R six cat orange truck China

Example4: T one tiger apple bus Indonesia

Example5: P seven cat orange car Pakistan

Example6: G six cat banana truck Russia

Example7: R five tiger apple bus India

Example8: L nine cat banana car Mexico

Example9: T four tiger orange truck United States

Example10: H three dog apple bus Japan

Example11: B three lion orange truck Germany

Example12:

# Prompt8: Second prompt for Fixed Key List Based Watermarking(Water-Probe-v2 N=5)

Please generate some text based on the following instructions(no other words):

The first word is randomly sampled from A-Z.

The second word is randomly sampled from zero to nine.

The third word is randomly sampled from cat, dog, tiger and lion.

The fourth word is randomly sampled from apple, banana and orange.

The fifth word is randomly sampled from car, bus and truck.

The sixth entry is the answer to the following question: Name a country with a large population.

Example1: D seven tiger apple car United States

Example2: J five dog banana bus India

Example3: R six cat orange truck China

Example4: T one tiger apple bus Argentina

Example5: P seven cat orange car France

Example6: G six cat banana truck Russia

Example7: R five tiger apple bus Australia

Example8: L nine cat banana car Canada

Example9: T four tiger orange truck United States

Example10: H three dog apple bus Brazil

Example11: B three lion orange truck Russia

Example12:

In this case, we selected the same question pair as in Water-Probe-v1. However, for Water-Probe-v2, all prefixes are randomly sampled by the LLM during the generation process. This random sampling by the LLM itself makes it easier to model the correlations between watermark keys at different positions.

Prompt 5 and Prompt 6 both generate prefixes of length 3. Since we further analyzed the case of generating prefixes of length 5 in Section 5, we also provide prompts for prefixes of length 5 in Prompt 7 and Prompt 8. It is worth noting that the longer the required prefix, the more total sampling times are needed, as more samples are required to cover all actually occurring prefixes. In this work, for the case where the prefix length is 3, we typically sampled 10,000 times, and for the case where the prefix length is 5, we typically sampled 100,000 times.

# D DETAILED ALGORITHM FOR Water-Probe

To provide a clearer presentation of our Water-Probe algorithm, we present here a complete algorithmic representation, corresponding to the algorithm pipeline process described in Section 3.2. This algorithm flow can be used for both Water-Probe-v1 and Water-Probe-v2, although the prompt construction process differs between them, as detailed in Appendix C.

# E DETECTION OF GLOBAL-FIXED KEY WATERMARKING IN LLMS

In this section, we explore how to identify LLMs with global-fixed key based watermarking. As discussed in Section 5, since global-fixed key based watermarking uses only one key globally, we cannot compare differences between two different watermark keys. However, as the global-fixed key produces consistent bias across all prompts, we can calculate a prior distribution for each prompt and then verify if the differences from this prior distribution have high similarity across all prompt lists satisfying Equation 5.

Specifically, let $P _ { 1 } , P _ { 2 } , . . . , P _ { N }$ be the constructed prompt list, and $P _ { \mathrm { p r i o r } }$ be their prior distribution. We can modify the average similarity calculation in Equation 9 as follows:

Algorithm 1 Water-Probe Algorithm   
Require: LLM M, significance level $\alpha$ , sampling count W

Ensure: Watermark detection result

1: Construct highly correlated prompts $P_{1}, P_{2}, ..., P_{N}$ satisfying Equation 5
2: Define watermark key list $K = \{k_{1}, k_{2}, ..., k_{m}\}$ 3: for each prompt $P_{i}$ and key $k_{j} \in K$ do
4: Initialize count dictionary $C_{i,j}(y) \leftarrow 0$ for all $y \in V$ 5: for w = 1 to W do
6: Construct sampling prompt $P_{i,j}^{w}$ based on watermarking method (see Section 3.3)
7: Generate output $y_{i,j}^{w} \sim P_{M}^{F}(\cdot | P_{i,j}^{w}, k_{j})$ 8: $C_{i,j}(y_{i,j}^{w}) \leftarrow C_{i,j}(y_{i,j}^{w}) + 1$ 9: end for
10: $\hat{P}_{M}^{F}(y | P_{i}, k_{j}) \leftarrow C_{i,j}(y)/W$ for all $y \in V$ 11: end for
12: Apply rank transformation to all $\hat{P}_{M}^{F}(\cdot | P_{i}, k_{j})$ 13: Calculate average similarity $\bar{S}$ using Equation 9
14: Compute z-score: $z = (\bar{S} - \mu)/\sigma$ 15: if $z > z_{\alpha}$ then
16: return LLM is likely watermarked
17: else
18: return No evidence of watermarking
19: end if

Table 3: Identification of global fixed key watermarking (e.g., Unigram watermarking) in LLMs using a prior distribution computed as the average output distribution of all eight LLMs. indicates high-confidence watermark identification and indicates low-confidence watermark identification while no color indicates no watermark identified. 

<table><tr><td rowspan="2">LLM</td><td colspan="2">Similarity</td><td colspan="2">Z-Score</td></tr><tr><td>Unigram</td><td>Unwatermark</td><td>Unigram</td><td>Unwatermark</td></tr><tr><td>GPT2</td><td> $0.59 \pm 0.018$ </td><td> $0.06 \pm 0.37$ </td><td>32.49</td><td>1.61</td></tr><tr><td>OPT1.3B</td><td> $0.6 \pm 0.02$ </td><td> $0.06 \pm 0.029$ </td><td>28.77</td><td>2.07</td></tr><tr><td>OPT2.7B</td><td> $0.65 \pm 0.009$ </td><td> $0.03 \pm 0.048$ </td><td>74.81</td><td>0.63</td></tr><tr><td>LLama2-7B</td><td> $0.48 \pm 0.042$ </td><td> $0.04 \pm 0.043$ </td><td>11.31</td><td>0.92</td></tr><tr><td>LLama-2-13B</td><td> $0.38 \pm 0.037$ </td><td> $0.08 \pm 0.018$ </td><td>10.17</td><td>4.54</td></tr><tr><td>LLama-3.1-8B</td><td> $0.68 \pm 0.06$ </td><td> $0.12 \pm 0.05$ </td><td>12.09</td><td>2.42</td></tr><tr><td>Mixtral-7B</td><td> $0.87 \pm 0.056$ </td><td> $0.038 \pm 0.052$ </td><td>22.24</td><td>0.74</td></tr><tr><td>Qwen2.5-7B</td><td> $0.53 \pm 0.058$ </td><td> $0.09 \pm 0.028$ </td><td>8.96</td><td>3.29</td></tr></table>

Table 4: Identification of global fixed key watermarking in LLMs using a single proxy model’s output as prior distribution. indicates high-confidence watermark identification and indicates low-confidence watermark identification while no color indicates no watermark identified. 

<table><tr><td rowspan="3">LLM</td><td colspan="4">GPT2</td><td colspan="4">OPT1.3B</td><td colspan="4">OPT2.7B</td><td colspan="4">LLama2-7B</td></tr><tr><td colspan="2">Similarity</td><td colspan="2">Z-Score</td><td colspan="2">Similarity</td><td colspan="2">Z-Score</td><td colspan="2">Similarity</td><td colspan="2">Z-Score</td><td colspan="2">Similarity</td><td colspan="2">Z-Score</td></tr><tr><td>W</td><td>NW</td><td>W</td><td>NW</td><td>W</td><td>NW</td><td>W</td><td>NW</td><td>W</td><td>NW</td><td>W</td><td>NW</td><td>W</td><td>NW</td><td>W</td><td>NW</td></tr><tr><td>GPT2</td><td> $0.48_{\pm 0.009}$ </td><td> $0.005_{\pm 0.025}$ </td><td>49.68</td><td>0.21</td><td> $0.58_{\pm 0.004}$ </td><td> $0.07_{\pm 0.02}$ </td><td>138.24</td><td>3.46</td><td> $0.65_{\pm 0.008}$ </td><td> $-0.07_{\pm 0.015}$ </td><td>77.60</td><td>-0.49</td><td> $0.42_{\pm 0.017}$ </td><td> $0.034_{\pm 0.015}$ </td><td>25.37</td><td>2.25</td></tr><tr><td>OPT1.3B</td><td> $0.71_{\pm 0.07}$ </td><td> $0.09_{\pm 0.04}$ </td><td>9.433</td><td>2.31</td><td> $0.59_{\pm 0.007}$ </td><td> $0.04_{\pm 0.02}$ </td><td>81.44</td><td>1.84</td><td> $0.67_{\pm 0.007}$ </td><td> $0.05_{\pm 0.015}$ </td><td>92.46</td><td>3.43</td><td> $0.45_{\pm 0.008}$ </td><td> $0.11_{\pm 0.04}$ </td><td>53.21</td><td>2.55</td></tr><tr><td>OPT2.7B</td><td> $0.7_{\pm 0.03}$ </td><td> $0.1_{\pm 0.05}$ </td><td>23.89</td><td>1.70</td><td> $0.59_{\pm 0.01}$ </td><td> $0.07_{\pm 0.03}$ </td><td>53.05</td><td>2.41</td><td> $0.64_{\pm 0.007}$ </td><td> $0.025_{\pm 0.025}$ </td><td>88.79</td><td>0.99</td><td> $0.46_{\pm 0.04}$ </td><td> $0.03_{\pm 0.01}$ </td><td>11.16</td><td>2.47</td></tr><tr><td>LLama2-7B</td><td> $0.66_{\pm 0.015}$ </td><td> $0.1_{\pm 0.007}$ </td><td>44.03</td><td>14.08</td><td> $0.57_{\pm 0.025}$ </td><td> $0.07_{\pm 0.02}$ </td><td>22.38</td><td>3.32</td><td> $0.67_{\pm 0.008}$ </td><td> $0.028_{\pm 0.023}$ </td><td>79.72</td><td>1.21</td><td> $0.43_{\pm 0.02}$ </td><td> $0.06_{\pm 0.02}$ </td><td>18.35</td><td>2.60</td></tr></table>

Prompt9: Used Prompt Set for Global-Fixed key Based Watermarking 

<table><tr><td>1: Please generate random number sequence between 0 to9:2,3,9,8,0,4,7,5,6,1,5,8,7,1,2,4,6,0,9,3,2: Please generate random number sequence between 0 to9:3,8,0,7,4,5,6,1,9,2,1,3,7,9,2,0,4,6,8,5,3: Please generate random number sequence between 0 to9:5,8,7,1,2,4,6,0,9,3,2,3,9,8,0,4,7,5,6,1,4: Please generate random number sequence between 0 to9:0,7,2,3,6,5,1,9,8,4,3,8,0,7,4,5,6,1,9,2,5: Please generate random number sequence between 0 to9:1,3,7,9,2,0,4,6,8,5,0,7,2,3,6,5,1,9,8,4,6: Please generate random number sequence between 0 to9:6,0,9,3,2,3,9,8,0,4,7,5,6,1,5,8,7,1,2,4,7: Please generate random number sequence between 0 to9:4,6,0,9,3,2,3,9,8,0,4,7,5,6,1,5,8,7,1,2,8: Please generate random number sequence between 0 to9:9,8,0,4,7,5,6,1,5,8,7,1,2,4,6,0,9,3,1,3,9: Please generate random number sequence between 0 to9:7,4,5,6,1,9,2,1,3,7,9,2,0,4,6,8,5,0,7,2,10: Please generate random number sequence between 0 to9:8,0,7,4,5,6,1,9,2,5,8,7,1,2,4,6,0,9,3,2,</td></tr></table>

$$
\bar {S} = \frac {1}{| \mathcal {P} | (| \mathcal {P} | - 1)} \sum_ {P _ {i} \neq P _ {j} \in \mathcal {P}} \operatorname{Sim} \left(R \left(P _ {M} (\cdot | P _ {i}) - P _ {\text {prior}} (\cdot | P _ {i})\right), R \left(P _ {M} (\cdot | P _ {j}) - P _ {\text {prior}} (\cdot | P _ {j})\right)\right). \tag {26}
$$

For the prior distribution, we select N proxy LLMs and calculate the average output probability distribution under prompt pi as the prior distribution. Specifically:

$$
P _ {\text { prior }} (\cdot | p _ {i}) = \frac {1}{N} \sum_ {j = 1} ^ {N} P _ {M _ {j}} (\cdot | p _ {i}), \tag {27}
$$

where $M _ { j }$ represents the j-th proxy LLM, and N is the total number of proxy LLMs used. Specifically, we used the following proxy LLMs: GPT2, OPT1.3B, OPT2.7B, Llama2-7B, Llama2-13B, Llama3- 8.1B, Mixtral-7B, and Qwen2.5-7B.

Additionally, we utilized 10 distinct prompts for computation to validate our method as shown in Prompt 9.

The intuition behind this method is that if M is watermarked, the differences between its output distributions and those of the proxy model should exhibit consistent patterns across different prompts, resulting in a high correlation. Conversely, for an unwatermarked model, we assume that the differences in bias between different language models are relatively small, and the correlation of these differences should be low.

Table 3 presents the identification results for global-fixed key watermarking using a prior distribution. Our method effectively identifies models employing global-fixed key watermarking, while yielding low z-scores for unwatermarked LLMs.

Table 5: Details of watermarking algorithms tested in our work. 

<table><tr><td>Algorithm Name</td><td>Category</td><td>Methodology</td></tr><tr><td>KGW (Kirchenbauer et al., 2023a)</td><td>N-Gram</td><td>Separate the vocabulary set into two lists: a red list and a green list based on the preceding token, then add bias to the logits of green tokens so that the watermarked text exhibits preference of using green tokens.</td></tr><tr><td>KGW-Min (Kirchenbauer et al., 2023b)</td><td>N-Gram</td><td>Similar to KGW, this approach partitions the vocabulary set based on the minimum token ID within a window of N-gram preceding tokens.</td></tr><tr><td>KGW-Skip (Kirchenbauer et al., 2023b)</td><td>N-Gram</td><td>Similar to KGW, this approach partitions the vocabulary set based on the left-most token ID within a window of N-gram preceding tokens.</td></tr><tr><td>Aar (Aaronson &amp; Kirchner, 2022)</td><td>N-Gram</td><td>Generate a pseudo-random vector  $r_t$  based on the N-gram preceding tokens to guide sampling at position t, and choose the token i that maximize  $r_t(i)^{1/p_t(i)}$  (exponential minimum sampling), where  $p_t$  is the probability produced by LLM.</td></tr><tr><td>γ-Reweight (Hu et al., 2023)</td><td>N-Gram</td><td>Randomly shuffle the probability vector using a seed based on the preceding N-gram tokens. Discard the left half of the vector, doubling the remaining probabilities. Conduct further sampling using this reweighted distribution.</td></tr><tr><td>DiPmark (Wu et al., 2023b)</td><td>N-Gram</td><td>Similar to γ-Reweight, after shuffling, discard the left α portion of the vector and amplify the remaining probabilities by  $1/(1 - \alpha)$ .</td></tr><tr><td>EXP-Edit (Kuditipudi et al., 2023)</td><td>Fixed-Key-List</td><td>Based on the Aar concept, construct a fixed pseudo-random vector list. When generating watermarked text, randomly select a start index in the list. For each watermarked token generation, sequentially use the pseudo-random vectors from this index for exponential minimum sampling. During detection, employ edit distance to calculate the correlation between the pseudo-random vector list and the text.</td></tr><tr><td>ITS-Edit (Kuditipudi et al., 2023)</td><td>Fixed-Key-List</td><td>Similar to EXP-Edit, this method also uses a fixed pseudo-random vector list. However, it uses inverse transform sampling instead of exponential minimum sampling during token selection.</td></tr></table>

Table 6: Z-scores of waterbag method and Exp-Edit method. represents high-confidence watermark and represents low-confidence watermark, while no color means no watermark. This table provides supplementary information on the similarity content in Table 2. 

<table><tr><td rowspan="2"></td><td rowspan="2">None</td><td colspan="4">KGW w. Water-Bag</td><td colspan="3">Exp-Edit(Key-len)</td></tr><tr><td> $|K \cup \overline{K}| = 1$ </td><td> $|K \cup \overline{K}| = 2$ </td><td> $|K \cup \overline{K}| = 4$ </td><td> $|K \cup \overline{K}| = 8$ </td><td>420</td><td>1024</td><td>2048</td></tr><tr><td colspan="9">Watermarked LLM Indentification</td></tr><tr><td>Water-Probe-v1(n=3)</td><td>-8.20</td><td>11.67</td><td>-4.22</td><td>-7.84</td><td>-4.34</td><td>-3.07</td><td>-5.87</td><td>-5.17</td></tr><tr><td>Water-Probe-v2(n=3)</td><td>-2.87</td><td>24.87</td><td>23.49</td><td>16.19</td><td>3.08</td><td>47.74</td><td>18.70</td><td>6.17</td></tr><tr><td>Water-Probe-v2(n=5)</td><td>-100.43</td><td>28.56</td><td>13.68</td><td>1.38</td><td>-4.41</td><td>131.30</td><td>78.63</td><td>69.40</td></tr></table>

To further validate the key factors in using prior distribution for testing, we conducted experiments using a single proxy model as the prior distribution, as shown in Table 4. We performed crossexperiments with different LLMs. These results demonstrate that global fixed-key watermarking can still be detected when using a single LLM as the prior distribution. However, the z-score detection for unwatermarked LLMs exhibits greater fluctuation. This is primarily due to the significant bias in a single proxy model as the prior distribution, affecting the variance of identification and resulting in small z-score.

Although our method using prior distribution achieved good results in our experiments, this identification approach has a limitation in that it can only detect stable biases in LLMs, assuming that a stable bias indicates a watermark. However, this assumption may not hold in real-world scenarios, as it is challenging to distinguish whether the bias is caused by watermarking or inherent to the LLM itself. This is particularly problematic in cases where LLMs are known to have inherent biases. Future work could investigate more interpretable detection methods.

# F DETAILS OF TESTED WATERMARKING ALGORITHMS

To help understand the watermarking algorithms related to the experiments in this paper, we provide detailed information for all watermarking algorithms in Table 5, including their names, references, types, and brief descriptions. All our experiments were conducted using the MarkLLM (Pan et al., 2024) framework.

Table 7: Z-scores for watermark detection on various LLMs with and without watermarks using Watermark-Probe-1 and Watermark-Probe-2. indicates high-confidence watermark identification, indicates low-confidence watermark identification, while no color indicates no watermark identified. This table provides supplementary information on the similarity content in Table 1. 

<table><tr><td rowspan="2">LLM</td><td rowspan="2">Non</td><td colspan="6">N-Gram</td><td colspan="2">Fixed-Key-Set</td></tr><tr><td>KGW</td><td>Aar</td><td>KGW-Min</td><td>KGW-Skip</td><td>DiPmark</td><td> $\gamma$ -Reweight</td><td>EXP-Edit</td><td>ITS-Edit</td></tr><tr><td colspan="10">Watermark-Probe-1 (w. prompt 1)</td></tr><tr><td>Qwen2.5-1.5B</td><td>-5.02</td><td>43.32</td><td>123.57</td><td>14.70</td><td>24.29</td><td>33.35</td><td>55.90</td><td>-5.21</td><td>-25.04</td></tr><tr><td>OPT-2.7B</td><td>-5.99</td><td>49.21</td><td>117.95</td><td>16.69</td><td>35.78</td><td>93.25</td><td>49.98</td><td>-1.32</td><td>-1.27</td></tr><tr><td>Llama-3.2-3B</td><td>-4.52</td><td>49.39</td><td>79.33</td><td>80.94</td><td>71.11</td><td>87.17</td><td>76.35</td><td>-6.70</td><td>-4.29</td></tr><tr><td>Qwen2.5-3B</td><td>-6.70</td><td>48.20</td><td>127.07</td><td>14.73</td><td>625.36</td><td>56.15</td><td>42.96</td><td>-6.18</td><td>-1.97</td></tr><tr><td>Llama2-7B</td><td>-8.20</td><td>30.01</td><td>109.00</td><td>25.51</td><td>29.75</td><td>44.35</td><td>106.37</td><td>-3.07</td><td>-28.05</td></tr><tr><td>Mixtral-7B</td><td>-3.80</td><td>25.03</td><td>40.21</td><td>35.51</td><td>19.99</td><td>36.30</td><td>137.39</td><td>-22.01</td><td>-3.31</td></tr><tr><td>Qwen2.5-7B</td><td>-1.16</td><td>38.25</td><td>30.03</td><td>22.85</td><td>50.34</td><td>47.31</td><td>50.48</td><td>-3.39</td><td>-15.92</td></tr><tr><td>Llama-3.1-8B</td><td>-6.44</td><td>20.40</td><td>143.52</td><td>29.05</td><td>28.19</td><td>29.01</td><td>125.46</td><td>-3.79</td><td>-12.70</td></tr><tr><td>Llama2-13B</td><td>-3.62</td><td>29.23</td><td>79.42</td><td>11.74</td><td>18.01</td><td>30.40</td><td>49.23</td><td>-6.63</td><td>-2.78</td></tr><tr><td colspan="10">Watermark-Probe-2 (w. prompt 2)</td></tr><tr><td>Qwen2.5-1.5B</td><td>-5.06</td><td>34.55</td><td>55.97</td><td>16.39</td><td>8.79</td><td>19.45</td><td>14.28</td><td>10.73</td><td>1840.44</td></tr><tr><td>OPT-2.7B</td><td>-1.95</td><td>42.59</td><td>67.93</td><td>15.17</td><td>4.61</td><td>32.40</td><td>11.04</td><td>26.44</td><td>1073.13</td></tr><tr><td>Llama-3.2-3B</td><td>-12.42</td><td>29.91</td><td>96.14</td><td>50.00</td><td>19.91</td><td>80.04</td><td>67.07</td><td>34.99</td><td>7702.12</td></tr><tr><td>Qwen2.5-3B</td><td>-3.48</td><td>6.35</td><td>108.44</td><td>11.29</td><td>25.92</td><td>39.88</td><td>18.84</td><td>18.06</td><td>8209.12</td></tr><tr><td>Llama2-7B</td><td>-2.87</td><td>24.87</td><td>40.05</td><td>32.68</td><td>15.62</td><td>35.50</td><td>25.11</td><td>47.74</td><td>6885.04</td></tr><tr><td>Mixtral-7B</td><td>-0.87</td><td>6.09</td><td>54.39</td><td>11.14</td><td>13.49</td><td>49.02</td><td>111.12</td><td>14.83</td><td>1812.12</td></tr><tr><td>Qwen2.5-7B</td><td>-2.48</td><td>8.90</td><td>185.88</td><td>10.50</td><td>13.06</td><td>7.64</td><td>12.40</td><td>13.04</td><td>1982.74</td></tr><tr><td>Llama-3.1-8B</td><td>-64.31</td><td>25.24</td><td>104.77</td><td>10.03</td><td>49.23</td><td>38.47</td><td>81.36</td><td>31.35</td><td>12701.95</td></tr><tr><td>Llama2-13B</td><td>-3.98</td><td>20.72</td><td>38.26</td><td>10.36</td><td>16.60</td><td>58.10</td><td>83.27</td><td>47.74</td><td>333.37</td></tr></table>

![](images/cd92b650bca647d9c206aa4efab3df2c4be914bde70c606a61784cf8c49e02d4.jpg)

<details>
<summary>line</summary>

| Temperature | Water-Probe-v1 (KGW) | Water-Probe-v2 (KGW) | Water-Probe-v2 (EXP-Edit) | Water-Probe-v1 (No Watermark) | Water-Probe-v2 (No Watermark) |
| ----------- | --------------------- | --------------------- | -------------------------- | ------------------------------ | ------------------------------ |
| 0.2         | ~30                   | ~150                  | ~10                        | ~10                            | ~0                             |
| 0.4         | ~150                  | ~1000                 | ~50                        | ~10                            | ~5                             |
| 0.6         | ~300                  | ~200                  | ~200                       | ~10                            | ~10                            |
| 0.8         | ~200                  | ~200                  | ~200                       | ~10                            | ~1                             |
| 1.0         | ~200                  | ~300                  | ~100                       | ~10                            | ~-10                           |
| 1.2         | ~200                  | ~200                  | ~200                       | ~10                            | ~-10                           |
| 1.4         | ~200                  | ~300                  | ~300                       | ~10                            | ~-15                           |
| 1.6         | ~200                  | ~150                  | ~200                       | ~1                             | ~-10                           |
</details>

Figure 5: The variation of z-scores at different temperatures when calculating similarity without using rank transformation in Equation 9.

# G ABLATION OF RANK TRANSFORMATION

To illustrate the importance of the rank transformation mentioned in Equation 9, we present in Figure 5 the variation of z-scores at different temperatures without using rank transformation. It can be observed that without rank transformation, the z-scores for Unwatermarked LLMs are significantly higher, especially at lower temperatures. Comparing the left plots in Figures 3 and 5, we can see that rank transformation effectively reduces the z-scores of Unwatermarked LLMs, making the identification and detection more stable.

Table 8: Identification p-value for various LLMs with different watermarks and without watermarks, using our two methods: Watermark-Probe-v1 and Watermark-Probe-v2. represents highconfidence watermark and represents low-confidence watermark, while no color means no watermark. This table provides supplementary information on the similarity content in Table 1. 

<table><tr><td rowspan="2">LLM</td><td rowspan="2">Non</td><td colspan="6">N-Gram</td><td colspan="2">Fixed-Key-Set</td></tr><tr><td>KGW</td><td>Aar</td><td>KGW-Min</td><td>KGW-Skip</td><td>DiPmark</td><td> $\gamma$ -reweight</td><td>EXP-Edit</td><td>ITS-Edit</td></tr><tr><td colspan="10">Watermark-Probe-v1 (w. prompt 1)</td></tr><tr><td>Qwen2.5-1.5B</td><td>1</td><td>2.9e-410</td><td>5.9e-3319</td><td>3.2e-49</td><td>1.3e-130</td><td>3.6e-244</td><td>2.0e-681</td><td>1</td><td>1</td></tr><tr><td>OPT-2.7B</td><td>1</td><td>1.1e-528</td><td>3.4e-3024</td><td>7.7e-63</td><td>1.1e-280</td><td>2.6e-1891</td><td>2.9e-545</td><td>1-9.3e-2</td><td>1-1.0e-1</td></tr><tr><td>Llama-3.2-3B</td><td>1</td><td>1.6e-532</td><td>1.4e-1369</td><td>1.3e-1425</td><td>5.2e-1101</td><td>4.4e-1653</td><td>7.9e-1269</td><td>1</td><td>1</td></tr><tr><td>Qwen2.5-3B</td><td>1</td><td>2.7e-507</td><td>1.8e-3509</td><td>2.1e-49</td><td>8.3e-84925</td><td>1.7e-687</td><td>1.6e-403</td><td>1</td><td>1-2.4e-2</td></tr><tr><td>Llama2-7B</td><td>1</td><td>3.6e-198</td><td>4.3e-2583</td><td>7.6e-144</td><td>8.7e-195</td><td>7.0e-430</td><td>4.4e-2460</td><td>1-1.1e-3</td><td>1</td></tr><tr><td>Mixtral-7B</td><td>1</td><td>1.4e-138</td><td>8.0e-354</td><td>1.7e-276</td><td>3.4e-89</td><td>8.1e-289</td><td>3.9e-4102</td><td>1</td><td>1</td></tr><tr><td>Qwen2.5-7B</td><td>1-1.2e-1</td><td>2.1e-320</td><td>2.0e-198</td><td>7.3e-116</td><td>4.2e-553</td><td>7.9e-489</td><td>3.6e-556</td><td>1</td><td>1</td></tr><tr><td>Llama-3.1-8B</td><td>1</td><td>8.4e-93</td><td>4.4e-4476</td><td>7.7e-186</td><td>3.9e-175</td><td>2.5e-185</td><td>3.6e-3421</td><td>1</td><td>1</td></tr><tr><td>Llama2-13B</td><td>1</td><td>4.0e-188</td><td>1.1e-1372</td><td>4.0e-32</td><td>8.1e-73</td><td>2.7e-203</td><td>4.3e-529</td><td>1</td><td>1-2.7e-3</td></tr><tr><td colspan="10">Watermark-Probe-v2 (w. prompt 2)</td></tr><tr><td>Qwen2.5-1.5B</td><td>1</td><td>7.1e-262</td><td>4.1e-683</td><td>1.1e-60</td><td>7.5e-19</td><td>1.5e-84</td><td>1.5e-46</td><td>3.7e-27</td><td>9.8e-735530</td></tr><tr><td>OPT-2.7B</td><td>1-2.6e-2</td><td>1.2e-396</td><td>5.6e-1005</td><td>2.8e-52</td><td>2.0e-6</td><td>1.4e-230</td><td>1.2e-28</td><td>2.4e-154</td><td>1.2e-250072</td></tr><tr><td>Llama-3.2-3B</td><td>1</td><td>7.3e-197</td><td>3.5e-2010</td><td>1.1e-545</td><td>1.7e-88</td><td>3.7e-1394</td><td>9.2e-980</td><td>1.6e-268</td><td>2.5e-12881755</td></tr><tr><td>Qwen2.5-3B</td><td>1</td><td>1.1e-10</td><td>1.2e-2556</td><td>7.4e-30</td><td>2.0e-148</td><td>4.4e-348</td><td>1.8e-79</td><td>3.3e-73</td><td>7.3e-14633482</td></tr><tr><td>Llama2-7B</td><td>1-2.1e-3</td><td>7.9e-137</td><td>4.9e-351</td><td>1.5e-234</td><td>2.7e-55</td><td>2.5e-276</td><td>1.9e-139</td><td>1.0e-497</td><td>4.3e-10293604</td></tr><tr><td>Mixtral-7B</td><td>1-1.9e-1</td><td>5.6e-10</td><td>3.1e-645</td><td>4.0e-29</td><td>9.0e-42</td><td>1.3e-524</td><td>2.0e-2684</td><td>4.7e-50</td><td>6.5e-713068</td></tr><tr><td>Qwen2.5-7B</td><td>1-6.6e-3</td><td>2.8e-19</td><td>3.9e-7506</td><td>4.3e-26</td><td>2.8e-39</td><td>1.1e-14</td><td>1.3e-35</td><td>3.6e-39</td><td>3.1e-853666</td></tr><tr><td>Llama-3.1-8B</td><td>1</td><td>7.3e-141</td><td>1.0e-2386</td><td>5.6e-24</td><td>4.3e-529</td><td>4.5e-324</td><td>2.0e-1440</td><td>4.9e-216</td><td>7.5e-35034440</td></tr><tr><td>Llama2-13B</td><td>1</td><td>1.1e-95</td><td>1.4e-320</td><td>1.9e-25</td><td>3.5e-62</td><td>6.8e-736</td><td>1.0e-1508</td><td>1.0e-497</td><td>2.0e-24136</td></tr></table>

# H SUPPLEMENTARY Z-SCORES AND P-VALUES

To facilitate a better understanding of the statistical methods used in identifying watermarked LLMs, we provide detailed information including z-scores and p-values for Table 1 and Table 2 in this section.

Specifically, Table 6 provides supplementary z-score information for Table 2, Table 7 provides supplementary z-score information for Table 1, and Table 8 provides supplementary p-value information for Table 1.

For all experiments, we consider a z-score below 4 to indicate no watermark, between 4 and 10 to indicate a watermark with relatively low confidence, and above 10 to indicate a watermark with high confidence.

# I COMPARISON WITH RELATED WORK (GLOAGUEN ET AL., 2024)

1. Universal Detection: Our method (particularly Water-Probe-v2) can detect all current watermarking-during-generation approaches (those that modify generation logits or sampling processes). In contrast, Gloaguen et al. (2024)’s method requires specific designs for different watermarking algorithms:

• Monte Carlo permutation test for red-green watermarking   
• Mann-Whitney U test for EXP-edit watermarking   
• Potential new methods for future watermarking methods

Our approach represents the first universal detection method effective across all current LLM watermarking techniques.

2. Unified Theoretical Foundation: We provide a unified theoretical analysis and explanation for why watermarked LLMs can be detected, specifically demonstrating how watermark key conflicts lead to identifiable characteristics in model outputs. This theoretical framework provides a comprehensive understanding of the detection mechanism.   
3. Imperceptibility Enhancement: Beyond detection methods, we also contribute the Water-Bag approach for improving the imperceptibility of watermarked LLMs, demonstrating significant improvements in watermark concealment while maintaining detectability.

Table 9: Experimental Results with Argmax Sampling On Exp-Edit 

<table><tr><td>Temp</td><td>Watermark F1</td><td>Perplexity</td><td>Gloaguen P-value</td><td>Water-Probe-v2 P-value</td></tr><tr><td>0.2</td><td>0.664</td><td>11.8</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>0.3</td><td>0.666</td><td>11.5</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>0.4</td><td>0.678</td><td>11.2</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>0.5</td><td>0.793</td><td>10.9</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>0.6</td><td>0.907</td><td>10.7</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>0.7</td><td>0.965</td><td>10.5</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>0.8</td><td>0.987</td><td>10.4</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>0.9</td><td>0.987</td><td>10.3</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>1.0</td><td>0.987</td><td>10.3</td><td>&lt;0.001</td><td>&lt;0.001</td></tr></table>

Table 10: Experimental Results with Multinomial Sampling On Exp-Edit, which is the most challenging case for watermarked LLM identification. 

<table><tr><td>Temp</td><td>Watermark F1</td><td>Perplexity</td><td>Gloaguen P-value</td><td>Water-Probe-v2 P-value</td></tr><tr><td>0.2</td><td>0.666</td><td>11.1</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>0.3</td><td>0.662</td><td>11.8</td><td>0.33</td><td>&lt;0.001</td></tr><tr><td>0.4</td><td>0.672</td><td>11.5</td><td>0.83</td><td>&lt;0.001</td></tr><tr><td>0.5</td><td>0.740</td><td>11.2</td><td>1.0</td><td>&lt;0.001</td></tr><tr><td>0.6</td><td>0.877</td><td>11.0</td><td>1.0</td><td>&lt;0.001</td></tr><tr><td>0.7</td><td>0.985</td><td>10.8</td><td>1.0</td><td>&lt;0.001</td></tr><tr><td>0.8</td><td>0.985</td><td>10.7</td><td>1.0</td><td>&lt;0.001</td></tr><tr><td>0.9</td><td>0.987</td><td>10.6</td><td>1.0</td><td>&lt;0.001</td></tr><tr><td>1.0</td><td>0.987</td><td>10.6</td><td>1.0</td><td>&lt;0.001</td></tr></table>

4. Broader Applicability for Challenging Watermarking Variants: Our method supports more challenging watermarking variants. For instance, while Gloaguen et al. (2024)’s experiments with EXP-edit only considered argmax sampling after exponential transformation (limiting a length-N watermark key list to at most N different sampling results), our method requires no such assumptions.

To demonstrate the broader applicability of our method, we conducted experiments with EXP-edit using sampling after exponential transformation. Given logits $l _ { i } ,$ we first compute probabilities $p _ { i }$ through temperature scaling:

$$
p _ {i} = \frac {\exp (l _ {i} / \tau)}{\sum_ {j} \exp (l _ {j} / \tau)} \tag {28}
$$

where $\tau$ is the temperature parameter. While Gloaguen et al. (2024)’s analysis focused on the deterministic argmax sampling variant:

$$
i ^ {*} = \arg \max _ {i} (\xi_ {i} ^ {(j)}) ^ {1 / p _ {i}} \tag {29}
$$

This deterministic approach has a fundamental limitation - for a watermark key list of length N, it can only produce at most N distinct outputs. We instead evaluate multinomial sampling from the distribution:

$$
P (i) \propto (\xi_ {i} ^ {(j)}) ^ {1 / p _ {i}} \tag {30}
$$

Our experiments used the MarkLLM framework with EXP-Edit watermarking (key length = 420), using OPT-1.3B as the base model and LLaMA-7B for perplexity calculations. For Gloaguen et al. (2024)’s testing method, we generated 1,000 text samples of length 200 tokens each. For Water-Probe-v2 testing, we generated 10,000 text samples of length 5 tokens each.

Tables 9 and 10 show that when applying multinomial sampling after EXP transformation, the watermark detection F1 scores and perplexity values remain largely unaffected. However, while our method maintains its effectiveness, Gloaguen et al. (2024)’s method fails to detect the watermark. This demonstrates the broader applicability of our approach to practical watermarking deployments.

# J GUIDELINES FOR CONSTRUCTING WATERMARKED LLM IDENTIFICATION PROMPTS

To help better understand our method, we provide guidelines for constructing watermarked LLM identification prompts, divided into three parts: question space design, answer space design, and implementation and verification protocols.

# J.1 OVERALL PROMPT STRUCTURE

The prompts in our identification method consist of two essential components - a prefix component and a question component. The specific requirements for each component will be detailed in subsequent sections. Here is a basic illustration of the structure:

# Basic Two-Component Prompt Structure

Input Prompt: Please start your answer with "WXYZ" (prefix component) and then answer the question: What is a major city in Asia? (question component)

Response: WXYZ Tokyo

Explanation: The generated prefix would help to fix the watermark key, while the actual answer would reflect the model’s response distribution (achieved by repeated sampling).

# J.2 QUESTION COMPONENT DESIGN

As described in Section 3.2 Step 1, we should first construct highly correlated prompts with significantly overlapping but non-identical answer spaces. This enables easy assessment of how potential watermark keys affect the answer spaces of different prompts.

Here is a list of criteria for selecting questions:

1. Answer Space Similarity: Select questions with overlapping but non-identical answer spaces. For example:

• “Name a country with a large population”   
• “Name a country with a large area”   
• “Name a country with a high GDP”   
• “Name a country with rich natural resources”

These questions typically share common answers (e.g., USA, China, Russia) while maintaining distinct probability distributions over the answer space.

# 2. Structural Requirements:

• Questions should be concise and unambiguous   
• Answers should come from a well-defined finite set (e.g., countries, cities)   
• Questions should maintain comparable difficulty levels   
• The target entity category should remain consistent within a test suite

# J.3 PREFIX COMPONENT DESIGN

# J.3.1 WATERMARK-PROBE-V1 CONSTRUCTION

For Watermark-Probe-v1, simply instruct the LLM to generate a fixed prefix before answering the question through explicit prompt instructions.

Here are the design principles for the prefix component:

1. Use meaningless character sequences (e.g., “abcd”, “wxyz”) that have no semantic meaning in any language   
2. Avoid any sequences that could form acronyms, abbreviations or meaningful patterns   
3. Ensure the prefix is completely unrelated to any potential answers or question domains   
4. Keep the prefix length sufficient for determining the watermark key while maintaining semantic independence

Here is an example of the prefix component:

# Implementation Example for Watermark-Probe-v1

Please generate abcd before answering the question.

Question: Name a country with a large population.

Answer: abcd India

Explanation: The generated prefix is meaningless and unrelated to the question domain, ensuring that it does not introduce any contextual bias.

# J.3.2 WATERMARK-PROBE-V2 CONSTRUCTION

For Watermark-Probe-v2, we need to design a controlled randomization process before answering the question to help fix the watermark key (see Section 3.3 for detailed reasons).

Here are the design principles for the prefix component:

1. Ensure the prefix generation does not influence the answer to the main question   
2. Design multiple choice sets with logically equivalent probabilities   
3. Keep the number of choices moderate and manageable   
4. Maintain clear boundaries between different choice sets

Here is an example of the prefix component:

# Implementation Example for Watermark-Probe-v2

Please generate a sentence that satisfies the following conditions:

– First word: Randomly sampled from A-Z   
– Second word: Randomly sampled from zero to nine   
– Third word: Randomly sampled from {cat, dog, tiger, lion}

Then answer: Name a country with a large population.

Answer: A one cat China

Explanation: All the possible generated prefixes are not related to the question domain, ensuring that they do not introduce any contextual bias.

# K THREAT MODEL

In this section, we outline the threat model under which our watermark identification method (detector) operates. We consider the capabilities and limitations of both the detector and the LLM service provider.

# K.1 DETECTOR CAPABILITIES

We assume the detector:

• Has black-box access to the LLM through standard API interfaces   
• Can only interact with the model through normal prompt-response queries   
• Has no access to model architecture, parameters, or training data   
• Can perform multiple queries   
• Cannot modify or influence the model’s internal state

# K.2 TRUST ASSUMPTIONS

The threat model assumes:

• The LLM service provider may embed watermarks in the model outputs   
• The API interface itself is trustworthy and returns genuine model outputs   
• No man-in-the-middle attacks or response tampering occurs   
• The detection process does not require knowledge of specific watermarking algorithms

# K.3 DETECTION GOALS AND CONSTRAINTS

The primary objectives within this threat model are:

• Determine the presence or absence of watermarks in model outputs   
• Maintain detection accuracy across different sampling temperatures and model configurations

Key constraints include:

• Detection must be performed solely through black-box testing   
• Watermark removal or tampering is outside the scope   
• Detection methods must be robust against normal model output variations

# L TEST ON CLOSED-SOURCE MODELS

We evaluated Water-Probe-V2’s detection capabilities on several closed-source models, including GPT-4o-mini, GPT-4o, GPT-3.5-turbo, Gemini-1.5-flash, and Gemini-1.5-pro. For all experiments, we utilized the latest API versions of these models (as of November 15, 2024) with a temperature setting of 0.7.

Table 11: Watermarked LLM Identification Results on Closed-source Models 

<table><tr><td>Model</td><td>Similarity</td><td>Std Dev</td><td>Z-score</td><td>Watermarked?</td></tr><tr><td>GPT-4o-mini</td><td>-0.005</td><td>0.018</td><td>-5.984</td><td>No</td></tr><tr><td>GPT-4o</td><td>0.017</td><td>0.020</td><td>-4.211</td><td>No</td></tr><tr><td>GPT-3.5-turbo</td><td>0.028</td><td>0.030</td><td>-2.362</td><td>No</td></tr><tr><td>Gemini-1.5-flash</td><td>0.027</td><td>0.049</td><td>-1.474</td><td>No</td></tr><tr><td>Gemini-1.5-pro</td><td>0.018</td><td>0.038</td><td>-2.135</td><td>No</td></tr></table>

Our experimental results provide strong evidence that current closed-source model APIs do not contain watermarks. However, it is important to note that a key limitation of this experiment is our inability to verify ground truth labels, making it impossible to definitively confirm the accuracy of our detection results.

# M REVERSION KEY CALCULATION OF WATER-BAG

In this section, we provide detailed calculations for determining reversion keys that satisfy the constraints in Equation 11. Let $p = P _ { M } ( y _ { i } | x , y _ { 1 : i - 1 } )$ represent the original model distribution, and $q = F ( p , f ( K _ { j } , y _ { i - n : i - 1 } ) )$ represent the distribution after modification using key $K _ { j }$ .

According to Equation 11, we have:

$$
\frac {1}{2} (q + F (p, f (\overline {{K _ {j}}}, y _ {i - n: i - 1}))) = p \tag {31}
$$

Through algebraic manipulation, we can derive the required modification for the reversion key:

$$
F (p, f (\overline {{K _ {j}}}, y _ {i - n: i - 1})) = 2 p - q \tag {32}
$$

This equation provides the concrete method for calculating the reversion key $\overline { { K _ { j } } } .$ . Specifically, for any input sequence $y _ { i - n : i - 1 }$ , the function $f ( \overline { { K _ { j } } } , y _ { i - n : i - 1 } )$ must map the original distribution $p$ to $2 p - q$ to satisfy Equation 11.

It is important to note that a reversion key need not be restricted to numerical values. Any key that produces the required distributional modification qualifies as a valid reversion key, as long as it accurately satisfies the constraint equation.