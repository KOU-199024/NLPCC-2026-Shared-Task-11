# LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models

Yan Wang1\*, Ling Ding1∗, Tien N. Nguyen2∗, Shaohua Wang1†, Yanan Zheng3†

1Central University of Finance and Economics,

2University of Texas at Dallas, 3Yale University

{dayanking,imlingding}@gmail.com, tien.n.nguyen@utdallas.edu,

davidshwang@ieee.org, yanan.zheng@yale.edu

# Abstract

Large Language Models for code often entail significant computational complexity, which grows significantly with the length of the input code sequence. We propose LEANCODE for code simplification to reduce training and prediction time, leveraging code contexts in utilizing attention scores to represent the tokens’ importance. We advocate for the selective removal of tokens based on the average context-aware attention scores rather than average scores across all inputs. LEANCODE uses the attention scores of ‘CLS’ tokens within the encoder for classification tasks, such as code search. It also employs the encoderdecoder attention scores to determine token significance for sequence-to-sequence tasks like code summarization. Our evaluation shows LEANCODE’s superiority over the SOTAs DI-ETCODE and SLIMCODE, with improvements of 60% and 16% for code search, and 29% and 27% for code summarization, respectively.

# 1 Introduction

Pre-trained Large Language Models (PLLMs) demand significant computational resources, often constraining input word or code token lengths. For example, when using CodeBERT (Feng et al., 2020a) locally, there’s a limitation of 512 tokens. CodeT5 (Wang et al., 2021b), CodeGen (Nijkamp et al., 2022), and GPT-4 (ChatGPT) also entail high computational overheads and costs, particularly with longer input code sequences. Code simplification of PLLMs is a practical way to reduce training and prediction time, while maintaining performance of a PLLM as much as possible. Given various pre-trained models and downstream tasks, it is intuitive that not all input tokens play critical roles in downstream-tasks. To tackle this challenge, the state-of-the art approaches, like DietCode (Zhang et al., 2022) and SlimCode (Wang et al., 2024), were proposed to simply the input program of a PLLM.

First, DietCode (Zhang et al., 2022) computes the average self-attention score for each code token across various contexts(global attention scores), treating it as the representative importance score for the token across the entire dataset. It then employs this score to determine whether to eliminate the code token from all inputs. However, due to the inherent nature of a PLLM, the same code token may be associated with different self-attention weights across different contexts. In our experiments, we observed a wide range of self-attention weights assigned to the same code token depending on the contexts in which it appears. Consequently, assigning an average score to each token for its importance is not appropriate.

Second, the way of DietCode using attention scores cannot directly reflect the importance of tokens in downstream tasks. Each input is represented by a special token labeled ‘CLS’. This ‘CLS’ token’s vector is computed based on the vectors of the constituent code tokens within that code snippet and description, determined by their self-attention scores. The vector of the ‘CLS’ token is fed into the final fully-connected layer for classification after the encoder. Consequently, tokens with higher CLS-attention scores hold greater importance for the classification task compared to individual token self-attention scores. However, DietCode assesses the self-attention scores of all input tokens for classification purposes, lacking the focus on the ‘CLS’ token, whose vector essentially encapsulates all necessary information for the classification.

Third, similarly, for code summarization, a transformer-based encoder-decoder architecture translates a given code snippet into a description. The self-attention mechanism within the encoder enables each token in the input to interact with all other tokens, effectively capturing dependencies and contextual information across different positions within the sequence, particularly adept at capturing long-range dependencies. DietCode only uses the attention scores of this encoder to signify the importance of code tokens and discarded these encoder-decoder attention scores.

Alternatively, SLIMCODE (Wang et al., 2024) is the state-of-the-art method based on human knowledge and uses a set of rules to determine the importance for different input code tokens. Specifically, it categorizes tokens into 8 priority levels based on the nature of the code tokens. For example, tokens in method signatures receive the highest priority, while symbol tokens (e.g., brackets, separators, and operators) have the lowest. However, performing better than DietCode, SLIMCODE still has the following issues (Wang et al., 2024). The manually selected priority levels with only 8 tiers result in a large number of tokens having the same priority, thus making token removal lack a solid basis. Secondly, the simplified code is fed into the model to complete downstream tasks, but model cognition can differ from human cognition. Thus, tokens considered important by human knowledge may not necessarily be considered important in model knowledge, leading to unexpected results.

In this paper, we posit that 1) model knowledge is more suitable for code simplification. 2) when leveraging attention scores to indicate the importance of code tokens in code simplification, one needs to consider their appearance contexts, the CLS and encoder-decoder attention are closely tied to downstream tasks, making their scores more appropriate for determining token importance in such tasks. We introduce a novel approach to code simplification, named LEANCODE. LEANCODE’s overarching goal is to streamline computation time, including those for training and inference, in downstream tasks like code search and summarization, while preserving performance to the fullest extent possible. First, we advocate for the removal of a specific instance of token based on the attention score unique to that context of that occurrence, rather than relying on average scores across all inputs. We use the statement type to represent the context of that occurrence of t. Second, we propose integrating the self-attention scores of ‘CLS’ tokens for classification tasks, such as code search. Finally, for sequence-to-sequence tasks, we consider encoder-decoder attention scores to ascertain the significance of input tokens.

The contributions of this paper are as follows:

1. We carried out a systematic analysis of the significant tokens learned by both ‘CLS’ and encoderdecoder attentions, comparing them with the tokens learned only by encoder self-attention.   
2. We present a new context-aware, code simplification, which is used in discriminative and sequence-to-sequence generation tasks.   
3. We evaluate LEANCODE in two downstream tasks and the results show its superiority over DI-ETCODE and SLIMCODE, with improvements of up to 60% and 16% for code search, and 29% and 27% for code summarization, respectively.   
4. We evaluate LeanCode’s cross-model transfer capability by feeding simplified code, generated using CodeT5, into the GPT-4o model to assess downstream task performance.

# 2 Preliminary Empirical Study

We conducted an empirical study on code search and summarization tasks to further investigate the significant tokens identified by the CLS and the Encoder-Decoder attentions. Our main focus is on the token level. The importance of statements and functions can be represented using tokens. We utilize the same models and datasets as DIETCODE and SLIMCODE. Specifically, for the code-search classification task, we use CodeBERT (an encoderonly model) and the encoder of CodeT5 (an encoder–decoder model), each augmented with an additional fully connected layer. For code summarization, we employ CodeBERT with a Transformer decoder as well as the complete CodeT5 model. For all experiments, we measure token importance using CLS (or Encoder-Decoder) attention scores from the final layer of the encoders of CodeBERT and CodeT5 (or the decoders of transformer and CodeT5), since both models compute scores across multiple layers. Token weights from the final layer provide the accurate representations of contextual relationships. Both tasks utilize the CodeSearchNet Corpus (Husain et al., 2019).

(RQ-1) What important tokens do CLS attentions emphasize on? For CodeBERT on code search and summarization, Fig. 1 show the top 10 and bottom 10 tokens with the highest and lowest variances in attention scores provided by Code-BERT. As seen, the tokens with the highest variance typically encompass richer semantic meanings, such as ‘accumulate’, ‘pure’, and ‘commerce’. Conversely, tokens with the lowest variance tend to resemble simple symbols, including numbers, separators, and brackets. Upon examining tokens with high variance, we refer back to the original source code to analyze their positions. It is observed that the tokens receive high attention scores when they appear in method signatures, function invocations, or as variables in return statements. Conversely, they attain low scores when used in conditions, expressions, and similar contexts. Given the substantial variance observed, relying solely on the average of all of its attention scores to gauge a token’s importance across diverse statement types is not reasonable. The attention scores of individual tokens are notably influenced by their positions in the code. Thus, categorizing statements and calculating the average attention score of each token in its contexts, i.e., distinct categories of statements, should be employed, named context-aware, category-local attention average. This aims to diminish such variance and bolster accuracy.

![](images/cf71cee6bb310b975350b2db16e87e192030022452a10dcf97471054a32dac44.jpg)

<details>
<summary>bar</summary>

| Token       | CLS Attention |
|-------------|---------------|
| mul         | 0.4           |
| accumulate  | 0.5           |
| pure        | 0.4           |
| comparable   | 0.4           |
| naive       | 0.1           |
| Async       | 0.6           |
| visit       | 0.5           |
| assist      | 0.3           |
| rename      | 0.4           |
| converge    | 0.2           |
</details>

(a) Top 10 tokens with highest variance in CLS attention for code search.   
![](images/2b50a8ca5b9ae57379e34f9e2cfc95c1117ea1bd8bcc0bc0dfbbbb08fa905565.jpg)

<details>
<summary>boxplot</summary>

| Token     | Encoder-decoder Attention |
| --------- | -------------------------- |
| Gr        | 0.2                        |
| Video     | 0.2                        |
| rm        | 0.2                        |
| Ad        | 0.2                        |
| commerce  | 0.2                        |
| Feed      | 0.2                        |
| VD        | 0.2                        |
| Pro       | 0.2                        |
| Budget    | 0.2                        |
| posal     | 0.2                        |
</details>

(c) Top 10 tokens with highest variance in encoder-decoder attention for code summarization.

![](images/5c917099aae6ea1553e904f81845ed2e9eaf0467a2ec4a0dd56df10404513534.jpg)

<details>
<summary>boxplot</summary>

| Token | CLS Attention |
|-------|---------------|
| e     | 0.00001       |
| i     | 0.00002       |
| i     | 0.00003       |
| e     | 0.00004       |
| vae   | 0.00005       |
| 3333  | 0.00006       |
| ei    | 0.00007       |
| "<    | 0.00008       |
| 998   | 0.00009       |
</details>

(b) Bottom 10 tokens with lowest variance in CLS attention for code search.   
![](images/787b07e9fd4f15edd6d96b6bf5a7a25513d9d34016c90c922d41de2a4aa3e4ca.jpg)

<details>
<summary>boxplot</summary>

| Token | Encoder-decoder Attention |
|-------|---------------------------|
| "<>" | 0.01                      |
| ), ) | 0.02                      |
| ={   | 0.03                      |
| (   | 0.04                      |
| }, , | 0.05                      |
| I[    | 0.06                      |
| )    | 0.07                      |
| :{   | 0.08                      |
| ]    | 0.09                      |
| ></i | 0.10                      |
</details>

(d) Bottom 10 tokens with lowest variance in encoder-decoder attention for code summarization.   
Figure 1: The variance of the top and bottom 10 tokens of CodeBERT for code search and summarization.

(RQ-2) What tokens encoder-decoder attentions emphasize on? Table 1 shows the average of the Encoder-Decoder attention scores of tokens based on statement classes (Zhang et al., 2022). Unlike CLS attention, each token in the input can have multiple Encoder-Decoder (EnDe) attention scores, i.e., for each generated token, the decoder will calculate an attention score for each token in the input. Thus, the largest attention score is selected as the attention score. The EnDe attention scores are generated in conjunction with the description. In the instances where the description contains intricate function details, these tokens garner high attention scores, facilitating the establishment of bi-modal mappings. For code search, the significance of detailed information within the code (e.g., Throw statements) is lower compared to the broader functional description (e.g., ‘Method signature’).

(RQ-3) Do the averages of self-attention scores reflect the CLS attentions and the Encoder-Decoder attentions? Our answer is ‘No’. The accumulated attention scores from the self-attention (as used in DIETCODE) is for pretraining and cannot reflect and substitute for those from the CLS and Encoder-Decoder attentions. i.e., the self-attention is used for pre-trained tasks and vectored general representations, not directly for downstream tasks. For elaboration, these attention schemes are for different tasks. The self attention is for pre-training tasks, while CLS attention is for fine-tuning downstream discriminative tasks, and the Encoder-Decoder attention is for downstream sequence-to-sequence generation tasks. In fact, the encoders of CodeBERT and CodeT5 have been trained in multiple pre-trained tasks. Thus, the averages of self-attention scores cannot replace the CLS and Encoder-Decoder attentions. The latter attentions are directly applied to downstream tasks.

Table 1: (RQ-2) Statistics of encoder-decoder attention scores based on 0.16M training dataset. (Max/Min: the maximum/minimum of encoder-decoder attention scores in each category; Global/Global\_variance: the average/variance of the global attention scores of tokens for each category; Category-local/Local\_variance: the averages/variance of category-local attention scores.) 

<table><tr><td>Category</td><td>Max</td><td>Min</td><td>Global</td><td>Global_variance</td><td>Category-local</td><td>Local_variance</td></tr><tr><td>Annotation</td><td>7.94</td><td>0.32</td><td>2.61</td><td>13.76</td><td>1.55</td><td>0.09</td></tr><tr><td>Arithmetic</td><td>37.44</td><td>0.07</td><td>2.69</td><td>15.41</td><td>2.30</td><td>2.52</td></tr><tr><td>Variable Declaration</td><td>65.54</td><td>0.09</td><td>2.87</td><td>15.63</td><td>2.69</td><td>7.97</td></tr><tr><td>Function Invocation</td><td>63.97</td><td>0.01</td><td>2.86</td><td>15.94</td><td>2.80</td><td>8.54</td></tr><tr><td>Return</td><td>55.23</td><td>0.10</td><td>3.08</td><td>17.61</td><td>4.76</td><td>16.02</td></tr><tr><td>Switch</td><td>30.03</td><td>0.07</td><td>2.71</td><td>16.36</td><td>2.41</td><td>2.63</td></tr><tr><td>Break</td><td>28.02</td><td>0.04</td><td>2.64</td><td>16.43</td><td>2.67</td><td>1.21</td></tr><tr><td>Setter</td><td>69.06</td><td>0.03</td><td>2.85</td><td>17.25</td><td>2.33</td><td>5.10</td></tr><tr><td>Synchronized</td><td>78.09</td><td>0.04</td><td>2.84</td><td>17.08</td><td>3.11</td><td>3.03</td></tr><tr><td>Try</td><td>78.27</td><td>0.03</td><td>2.82</td><td>17.31</td><td>2.46</td><td>2.69</td></tr><tr><td>Catch</td><td>34.99</td><td>0.07</td><td>3.01</td><td>19.80</td><td>2.44</td><td>4.18</td></tr><tr><td>Method Signature</td><td>91.69</td><td>0.14</td><td>3.29</td><td>18.21</td><td>5.91</td><td>30.92</td></tr><tr><td>Finally</td><td>10.49</td><td>0.74</td><td>2.38</td><td>7.78</td><td>2.99</td><td>1.74</td></tr><tr><td>Getter</td><td>68.49</td><td>0.03</td><td>2.88</td><td>16.59</td><td>2.58</td><td>6.42</td></tr><tr><td>Throw</td><td>87.67</td><td>0.06</td><td>2.80</td><td>16.04</td><td>3.10</td><td>8.13</td></tr><tr><td>Case</td><td>23.25</td><td>0.03</td><td>2.75</td><td>16.11</td><td>1.80</td><td>1.55</td></tr><tr><td>While</td><td>67.68</td><td>0.04</td><td>2.70</td><td>15.52</td><td>2.41</td><td>3.14</td></tr><tr><td>Continue</td><td>9.85</td><td>0.27</td><td>2.49</td><td>12.64</td><td>1.73</td><td>0.37</td></tr><tr><td>If Condition</td><td>57.88</td><td>0.05</td><td>2.84</td><td>15.84</td><td>2.50</td><td>5.97</td></tr><tr><td>For</td><td>60.62</td><td>0.03</td><td>2.91</td><td>17.21</td><td>2.99</td><td>6.89</td></tr><tr><td>Logging</td><td>65.63</td><td>0.04</td><td>2.77</td><td>15.53</td><td>2.89</td><td>8.42</td></tr></table>

# 3 LEANCODE: Code Simplification

# 3.1 Code Simplification Problem Formulation

Given a dataset $D = \{ d _ { 1 } , . . . , d _ { m } \}$ with m snippets. Each snippet $d _ { j }$ contains a sequence of $n _ { j }$ tokens. Thus, each code snippet $d _ { j }$ can be denoted as $d _ { j } = \{ t _ { 1 } , \cdot \cdot \cdot , t _ { n _ { j } } \}$ and the index of each token records its position. $w _ { i }$ denotes the importance of each token $t _ { i }$ and $x _ { i }$ is a binary indicator showing whether $t _ { i }$ should be removed or not. With the simplification ratio Simplif iedRatio, the total number of tokens to be removed for $d _ { j }$ is $\mathcal { X } = S i m p l i f i e d R a t i o \times n _ { j }$ . Now, we formulate code simplification as the combinatorial optimization problem as following:

$$
\text { minimize } \sum_ {i = 1} ^ {n _ {j}} w _ {i} x _ {i}, \text { such   that } \sum_ {i = 1} ^ {n _ {j}} x _ {i} = \mathcal {X}. \tag {1}
$$

$x _ { i } \in \{ 0 , 1 \}$ , code simplification aims to minimize the weighted sum of $w _ { i } x _ { i }$ that satisfies the number of tokens to be removed for each $d _ { j }$ .

# 3.2 LEANCODE Algorithm

# 3.2.1 Computation

As mentioned, there are three methods to measure token importance through attention scores: dynamic, global, and category-local methods. Regarding the dynamic method, the CLS and Encoder-Decoder attention scores of the same token will be different for different inputs, which are dynamically generated and can reflect the importance of corresponding tokens in context. However, it is inefficient and impractical to assign a dynamic attention score to each input token in the test dataset using the models. Calculating the dynamic attention scores requires multiple transformer blocks (for example, CodeBERT and the encoder of CodeT5 have a stack of 12 transformer blocks), making it time-consuming. Moreover, once we have obtained the dynamic attention scores, the downstream tasks are nearly complete. Thus, it does not make sense and is unnecessary to reduce the code and redo the downstream tasks.

Regarding the global attention average of each token in the training dataset, DietCode uses it to replace the dynamic attention score in testing dataset, which is computed in Equation (2):

$$
\mu_ {t} = \frac {\sum_ {j = 1} ^ {m} \sum_ {t \in d _ {j} ^ {\prime}} s _ {t}}{n _ {t}} \tag {2}
$$

where $t \in d _ { j } ^ { \prime }$ tmeans that a token t is in $d _ { j } ^ { \prime }$ in a training dataset $\mathbf { \check { \mathit { D } } ^ { \prime } }$ and $s _ { t }$ is the common self-attention score. $n _ { t }$ is the number of the occurrence of token t in the training dataset $D ^ { \prime }$ .

Our empirical study reveals that significant tokens often have high variances in global attention averages. As a result, we propose the categorylocal attention average for each token, defined as

$$
\mu_ {t} ^ {c} = \frac {\sum_ {j = 1} ^ {m} \sum_ {t \in p _ {k} , p _ {k} \in d _ {j} ^ {\prime} , L (p _ {k}) \in c} s _ {t}}{n _ {t} ^ {c}}. \tag {3}
$$

In this equation, $p _ { k }$ is a statement of a code snippet $d _ { j } ^ { \prime } . \ L ( p _ { k } )$ is the label (category) of the statement $p _ { k }$ . $n _ { t } ^ { c }$ refers to the number of occurrences of token t in the statements belonging to the category $c \in$ C . Finally, $s _ { t }$ can be CLS attention or Encoder-Decoder attention scores. The definitions of those scores are provided in Section B.

Algorithm 1 LEANCODE: Code Simplification Algorithm   
INPUT: A dataset $D = \{d_{1}, \ldots, d_{m}\}$ , token scores S, SimplifiedRatio

OUTPUT: A simplified code dataset $D'$ PROCEDURE:

1: $D^{c} \leftarrow D$ 2: for j = 1 to m do

3: removedTokens $\leftarrow \{\}$ 4: $X \leftarrow SimplifiedRatio \times n_{j}$ 5: removedTokenNum $\leftarrow 0$ 6: while removedTokenNum < X do

7: Add {index:token with lowest $s_{t}$ } ( $\in d_{j}^{c}$ , $\notin removedTokens$ ) into removedTokens

8: removedTokenNum updates

9: $d_{j}^{c} = d_{j}^{c}/removedTokens[1 : X]$ 10: return $D^{c}$

# 3.2.2 Removal Algorithm

Algorithm 1 displays LEANCODE algorithm. Unlike DietCode’s removing the less critical statements and proceeding to remove less important tokens from other statements, it exclusively focuses on token-level removal, without initially discarding entire statements. Deleting entire statements would result in the loss of important tokens. Our Algorithm (1) initializes a copy of the original dataset D as the returned simplified dataset (line 1). Next, it iteratively removes the tokens of each snippet in $D ^ { c }$ one by one (lines 2–9) based on their attention scores stored in the dictionary $S = \{ t , c , \mu _ { t } ^ { c } \}$ where c is the category of the statement that token t belongs to, $\mu _ { t } ^ { c }$ is the category-local attention average of the token. At line 3, removedTokens records the pair of the index and the respective token with current lowest score ({index:token}) in $d _ { j } ^ { c }$ at each turn. The number of removed tokens is set (line 4). In line 5, removedTokenNum is the number of currently selected tokens to be removed. At each iteration, LEANCODE repeatedly selects the remaining token (not in removedTokens) with the lowest score in $d _ { j } ^ { c }$ until the number of removed tokens is reached (lines 6–8). Finally, our algorithm returns the simplified dataset $D ^ { \prime }$ .

# 4 Empirical Evaluation

We evaluate our LEANCODE on different code tasks using three PLLMs and GPT-4o.

Downstream Tasks: We choose code search and summarization as the tasks (also used in DIET-CODE and SLIMCODE), which are commonly used in evaluating LLMs in text and code analysis (Ahmed and Devanbu, 2022; Feng et al., 2020a; Jiang et al., 2021; Wang et al., 2021a; Niu et al., 2022; Liu et al., 2019). The goal of code search is to find relevant code snippets from a codebase given a query and code summarization is to generate a natural language summary for a given code.

Models Under Study: We opted for 3 popular models, CodeBERT (Feng et al., 2020a), CodeT5 (Wang et al., 2021b) and GPT-4o (Islam and Moushi, 2024). Unlike DIETCODE (calculating attention scores based on all layers), for ‘CLS’ and Encoder-Decoder attention scores, we obtain them from the last encoder and decoder blocks of the respective model.

CodeBERT-based code search and summarization: We added a fully connected layer on top of the CodeBERT model for binary classification to perform code search. As in DietCode, we added a Transformer decoder for code summarization.

CodeT5-based code search and summarization: For code search, its encoder is separately extracted and joint with a fully connected layer for the classification task. We use CodeT5 directly for code summarization.

GPT4-based code search and summarization: Since GPT-4o is accessible only through programming APIs, we cannot directly access its model. We are limited to using prompts to obtain classification and summrization results and corresponding analyses from GPT-4o in a predefined format.

Baselines: We chose DIETCODE (Zhang et al., 2022) and SLIMCODE (Wang et al., 2024), the SOTA code simplification methods. Diet-Code (Zhang et al., 2022) is based on token weights learned by models and SlimCode (Wang et al., 2024) is based on the nature of tokens.

Datasets: We used code search and summarization datasets from CodeBERT (Feng et al., 2020a) (Details in Table 9 in Appendix). These are the extensions of CodeSearchNet (Husain et al., 2019), which is a collection of datasets and benchmarks for code retrieval using texts. It consists of +2 millions pairs of (comment, code) that were extracted from Github, covering six languages (Python, PHP,

Go, Java, JavaScript, and Ruby). Since Diet-Code (Zhang et al., 2022) reported similar trends for different languages, we conducted experiments only on Java.

Metrics: We use the simplification ratio to measure the degree of simplification of a code snippet. Given a code snippet Code and its simplified one Scode, $\begin{array} { r } { { S i m p l i f i e d R a t i o } = \frac { | C o d e | - | S \bar { c o } d e | } { | C o d e | } \times 1 0 0 . } \end{array}$ . Code and Scode are the numbers of tokens in Code and Scode. The efficiency of simplified code is measured by the time cost it takes for model inference. We use BLEU-4 score and MRR (Mean Reciprocal Rank) for code summarization and search, respectively. For code search by GPT-4o, we use Precision instead of MRR due to the latter’s high computational requirements. Precision is highly correlated with MRR in measuring model effectiveness. Similar to SLIMCODE, we randomly replace the code description in 400 sample pairs of textscode and check if the replacement content matches the code part. The dataset consists of an equal number of matching and non-matching samples.

Implementation: DietCode is realized using the code in (Zhang et al., 2022). We set up CodeBERT and CodeT5 with default hyper-parameters. For optimization, they used Adam optimizer with learning rates of $1 \times 1 0 ^ { - 5 }$ and $5 \times 1 0 ^ { - 5 }$ for downstream tasks. We used a server with 2 CPUs of Intel(R) Xeon(R) Golden 2.40GHz and 2 Nvidia A100s.

# 4.1 Effectiveness of LEANCODE

Tables 2 and 3 present the results. We first use the training dataset (complete code snippets) to train the relevant models, and then use the testing dataset (complete code snippets) to perform inference on a model to obtain the ‘Base’ results. Next, we apply code simplification to the samples in the testing dataset using DietCode, SlimCode, and LEAN-CODE according to the simplifiedRatio. Finally, we input the simplified code into the trained models to perform inference, obtaining results for the cases of the removal percentages of 10%-50%. As seen in columns ‘R-M’ and ‘R-B’, the performance of both methods declines in downstream tasks as the Simplif iedRatio gradually increases. However, even when 50% of the code is removed, the decline at most in the effect of LEANCODE for code search and code summarization are only 5.48% and 11.01%. Meanwhile, LEANCODE enhances DIET-CODE (SLIMCODE) up to 60.37% (15.82%) and 14.12%(6.28%) in terms of MRR and BLEU scores with CodeBERT. Similarly, it improves by 25.84% (10.14%) and 29.36% (27.04%) in terms of MRR and BLEU scores with CodeT5.

We observe two interesting points: 1) Intuitively, one would expect the performance of the removed code for downstream tasks to be worse than the original code. However, LEANCODE and SLIM-CODE can achieve better results on code search. 2) Compared with LEANCODE, SLIMCODE experiences a sharp decline in performance when the simplif iedRatio exceeds 30%.

For the $1 ^ { s t }$ observation, if the code contains more valuable information, the downstream tasks will have better results. Not all code snippets can be fully inserted into models, as the maximum lengths for original codes are limited to 512 tokens. Thus, even if some tokens are removed, tokens towards the end of the code snippets may still be included in the input. If the newly input tokens provide more information than the removed tokens, then the above observation may occur. Specifically, LEANCODE often removes low-quality tokens with lower attention scores (e.g., symbol-like tokens) while allowing high-quality tokens to enter the model input.

For the $2 ^ { n d }$ observation, the main reason lies in the deletion of increasingly important tokens as the process progresses. At this stage, a model’s ability to precisely discern token importance becomes critical for maintaining the task performance. SLIM-CODE, with only 8 levels of token importance, struggles to differentiate between crucial tokens when simplif iedRatio surpasses 30%.

Figure 2 presents an example of our LEANCODE. Subfigure (a) shows the original code, which computes a Bessel function using a recurrence formula. Subfigures (b) and (c) display the results of LEAN-CODE after 30% token removal for code search and summarization, respectively. Both approaches preserved critical elements, such as function signatures and return statements. For code search, structural details like loops, arithmetic operations, and variable names were retained to support precise keyword matching. For summarization, non-essential details such as loop bodies, variable declarations, and calculations were removed, while return statements were emphasized. This demonstrates how return statements play a key role in capturing code intent, enabling concise and clear summaries.

# 4.2 Efficiency of LEANCODE

Tables 2 and 3 show the results on the positive correlations between the simplified ratio and inference time. For instance, for code summarization using

Table 2: Results of Code Search for DIETCODE , SlimCode and LEANCODE. (10%-50%: removing 10%-50% tokens, R-M: Reduced MRR, Time: Inference time, R-T: Reduced Inference time) 

<table><tr><td rowspan="3">Ratio</td><td colspan="4">DIETCODE</td><td colspan="4">SlimCode</td><td colspan="4">LEANCODE</td><td colspan="4">Inference</td></tr><tr><td colspan="2">CodeBERT</td><td colspan="2">CodeT5</td><td colspan="2">CodeBERT</td><td colspan="2">CodeT5</td><td colspan="2">CodeBERT</td><td colspan="2">CodeT5</td><td colspan="2">CodeBERT</td><td colspan="2">CodeT5</td></tr><tr><td>MRR</td><td>R-M</td><td>MRR</td><td>R-M</td><td>MRR</td><td>R-M</td><td>MRR</td><td>R-M</td><td>MRR</td><td>R-M</td><td>MRR</td><td>R-M</td><td>Time</td><td>R-T</td><td>Time</td><td>R-T</td></tr><tr><td>Base</td><td>0.726</td><td>—</td><td>0.747</td><td>—</td><td>0.726</td><td>—</td><td>0.747</td><td>—</td><td>0.726</td><td>—</td><td>0.747</td><td>—</td><td>41m</td><td>—</td><td>40m</td><td>—</td></tr><tr><td>10%</td><td>0.663</td><td>8.67%↓</td><td>0.699</td><td>6.42%↓</td><td>0.731</td><td>0.68%↑</td><td>0.738</td><td>1.2%↓</td><td>0.728</td><td>0.27%↑</td><td>0.743</td><td>0.53%↓</td><td>38m</td><td>7.31%↓</td><td>36m</td><td>10%↓</td></tr><tr><td>20%</td><td>0.598</td><td>17.63%↓</td><td>0.669</td><td>10.44%↓</td><td>0.726</td><td>0.00%↓</td><td>0.733</td><td>1.87%↓</td><td>0.719</td><td>0.96%↓</td><td>0.736</td><td>1.47%↓</td><td>35m</td><td>14.63%↓</td><td>33m</td><td>17.5%↓</td></tr><tr><td>30%</td><td>0.529</td><td>27.13%↓</td><td>0.624</td><td>16.46%↓</td><td>0.70</td><td>3.58%↓</td><td>0.723</td><td>3.21%↓</td><td>0.716</td><td>1.37%↓</td><td>0.724</td><td>3.07%↓</td><td>32m</td><td>21.95%↓</td><td>31m</td><td>22.5%↓</td></tr><tr><td>40%</td><td>0.502</td><td>30.85%↓</td><td>0.602</td><td>19.41%↓</td><td>0.632</td><td>12.94%↓</td><td>0.679</td><td>9.1%↓</td><td>0.697</td><td>3.99%↓</td><td>0.714</td><td>4.41%↓</td><td>29m</td><td>29.27%↓</td><td>28m</td><td>30%↓</td></tr><tr><td>50%</td><td>0.429</td><td>40.90%↓</td><td>0.561</td><td>24.89%↓</td><td>0.594</td><td>18.18%↓</td><td>0.641</td><td>14.19%↓</td><td>0.688</td><td>5.23%↓</td><td>0.706</td><td>5.48%↓</td><td>26m</td><td>36.59%↓</td><td>25m</td><td>37.5%↓</td></tr></table>

Table 3: Results of Code Summarization for DIETCODE , SlimCode and LEANCODE. (10%-50%: removing 10%-50% tokens, R-B: Reduced BLEU, Time: Inference time, R-T: Reduced Inference time) 

<table><tr><td rowspan="3">Ratio</td><td colspan="3">DIETCODE</td><td colspan="3">SlimCode</td><td colspan="3">LEANCODE</td><td colspan="3">Inference</td></tr><tr><td colspan="2">CodeBERT</td><td>CodeT5</td><td colspan="2">CodeBERT</td><td>CodeT5</td><td colspan="2">CodeBERT</td><td>CodeT5</td><td colspan="2">CodeBERT</td><td>CodeT5</td></tr><tr><td>BLUE</td><td>R-B</td><td>BLUE</td><td>R-B</td><td>BLUE</td><td>R-B</td><td>BLUE</td><td>R-B</td><td>BLUE</td><td>R-B</td><td>Time</td><td>R-T</td></tr><tr><td>Base</td><td>18.25</td><td>—</td><td>20.55</td><td>—</td><td>18.25</td><td>—</td><td>20.55</td><td>—</td><td>18.25</td><td>—</td><td>29m</td><td>—</td></tr><tr><td>10%</td><td>16.44</td><td>9.91%↓</td><td>17.27</td><td>15.96%↓</td><td>17.86</td><td>2.13%↓</td><td>20.01</td><td>2.62%↓</td><td>18.08</td><td>0.93%↓</td><td>20.32</td><td>1.11%↓</td></tr><tr><td>20%</td><td>15.68</td><td>14.08%↓</td><td>16.48</td><td>19.80%↓</td><td>17.35</td><td>4.93%↓</td><td>18.68</td><td>4.23%↓</td><td>17.73</td><td>2.84%↓</td><td>20.18</td><td>1.80%↓</td></tr><tr><td>30%</td><td>15.05</td><td>17.53%↓</td><td>15.74</td><td>23.40%↓</td><td>16.8</td><td>7.94%↓</td><td>18.74</td><td>8.80%↓</td><td>17.23</td><td>5.58%↓</td><td>19.82</td><td>3.55%↓</td></tr><tr><td>40%</td><td>14.66</td><td>19.67%↓</td><td>15.11</td><td>26.47%↓</td><td>15.95</td><td>12.60%↓</td><td>16.35</td><td>20.43%↓</td><td>16.71</td><td>8.43%↓</td><td>19.27</td><td>6.22%↓</td></tr><tr><td>50%</td><td>14.23</td><td>22.02%↓</td><td>14.27</td><td>30.55%↓</td><td>15.28</td><td>16.27%↓</td><td>14.53</td><td>29.29%↓</td><td>16.24</td><td>11.01%↓</td><td>18.46</td><td>10.17%↓</td></tr></table>

CodeT5, the inference time is reduced by 40.9% when the simplification ratio is set to 50%.

For both tasks, as the simplified ratio increases, the inference time follows a near-linear descent on both of models. Specifically, for code search, the ratio of Simplif iedRatio and reduce time is about 0.7, meanwhile, for code summarization, the ratio is about 0.5 on CodeBERT and 0.75 on CodeT5, respectively. The ratio on CodeBERT for code summarization is relative lower than others since directly concatenating CodeBERT with a Transformer decoder would result in the model lacking some optimization for inference.

In addition to inference time, each method requires training and pruning time. DietCode and LeanCode also need extra computation for pertoken attention scores. While all methods share identical training times due to using the same training dataset and model, calculating token attention scores adds about 5% to the total training time. For example, in code summarization, standard training across 8 epochs takes 37.5 minutes per epoch, totaling 300 minutes. When attention scores are collected only during the final epoch, the first seven epochs remain unchanged, but the last epoch increases to 53 minutes. This results in a total time of 315.5 minutes,i.e., representing a 5% increase in training time.

We conducted additional pruning experiments for LeanCode and obtained pruning time results for SlimCode and DietCode from SlimCode’s paper. The results are shown in Table 4. From these results, we observed: 1) DietCode’s pruning time is significantly higher than SlimCode and Lean-Code due to its two-step process—selecting highquality statements first, then removing tokens from the remaining ones. This approach leads to complex knapsack optimization problems, resulting in a lower reduction ratio and increased pruning time; 2) LeanCode’s pruning time is approximately 2–4 times longer than SlimCode’s, primarily due to LeanCode’s more complex pruning process, which includes additional steps like tokenizing and statement class matching; 3) The pruning times for LeanCode and SlimCode remain within a comparable range.

Table 4: Pruning times for DietCode(DC), Slim-Code(SC), and LeanCode(LC) on code search and code summarization training datasets(10%-50% removal for each code snippet) 

<table><tr><td rowspan="2">Ratio</td><td colspan="3">Code Search</td><td colspan="3">Code Summarization</td></tr><tr><td>DC</td><td>SC</td><td>LC</td><td>DC</td><td>SC</td><td>LC</td></tr><tr><td>10%</td><td>9h24m</td><td>17m</td><td>46m33s</td><td>1h40m</td><td>45s</td><td>3m32s</td></tr><tr><td>20%</td><td>8h28m</td><td>17m</td><td>46m39s</td><td>1h30m</td><td>53s</td><td>3m37s</td></tr><tr><td>30%</td><td>7h37m</td><td>20m</td><td>47m15s</td><td>1h19m</td><td>59s</td><td>3m41s</td></tr><tr><td>40%</td><td>6h45m</td><td>21m</td><td>47m35s</td><td>1h11m</td><td>66s</td><td>3m45s</td></tr><tr><td>50%</td><td>5h59m</td><td>21m</td><td>47m43s</td><td>1h02m</td><td>69s</td><td>3m50s</td></tr></table>

# 4.3 Model Transferability

We assess whether code simplified by LEANCODE using one model can be applied to another while maintaining previous results and conclusions for a given downstream task. We replicate the analysis procedures in the Section 4.1 with GPT-4o. As GPT-4o only provides access via APIs, we used a program to interact with GPT-4o. We used the following prompts: 1) for code search, “Please check whether the incomplete code snippet is semantically consistent with the given text. Please analyze step-by-step first, and then answer in the following format.", and 2) for code summarization, “Write a short sentence to describe the function of the incomplete code snippet. Answer in the following format." Finally, we analyzed the results of GPT-4o to calculate the number of total tokens, precision, and BLEU-4.

<table><tr><td>1. public static double Y (int n, double x) { 
2. double by, bym, byp, tox; 
3. if (n == 0) return Y0 (x); 
4. if (n == 1) return Y (x); 
5. tox = 2.0 / x; 
6. by = Y(x); 
7. bym = Y0 (x); 
8. for (int j = 1; j &lt; n; j++) { 
9.    byp = j * tox * by - bym; 
10.    bym = by; 
11.    by = byp; 
12. } 
13. return by; 
14. }</td><td>1. public static double Y (int n, double x) { 
2. double by bym byp to; 
3. () return Y0(); 
4. () return Y(); 
5. tox = 2 / x; 
6. by Y(); 
7. bym Y0(); 
8. for () byp = j * tox * by - bym; 
9.    bym = by; 
10.    by = byp; 
11. } 
12. return by; 
13. }</td><td>1. public static double Y (int n, double x) { 
2. , , byp, ; 
3. if (n) return Y0 (x); 
4. if (n) return Y (x); 
5. = 2 x; 
6. by = Y (x); 
7. bym = Y0 (x); 
8. (; &lt; n;)byp = by bym; 
9.    bym = by; 
10.    by = byp; 
11. } 
12. return by; 
13. }</td></tr><tr><td>(a) original source code</td><td>(b) remove 30% tokens with LeanCode for code search</td><td>(c) remove 30% tokens with LeanCode for code summarization</td></tr></table>

Figure 2: The example of LEANCODE for code simplification.

Table 5: Results of removal methods on GPT-4 for Code Search (IT: Input Tokens, R-IT: Reduced Input Tokens (%), OT:Output Tokens, R-OT:Reduced Output Tokens (%); TT:Total Tokens, R-P:Reduced Precision (%) 

<table><tr><td>Removal method</td><td>IT</td><td>R-IT</td><td>OT</td><td>R-OT</td><td>TT</td><td>R-TT</td><td>Precision</td><td>R-P</td></tr><tr><td>Base</td><td>102385</td><td>—</td><td>24535</td><td>—</td><td>127920</td><td>—</td><td>0.82</td><td>—</td></tr><tr><td>DietCode (10%)</td><td>98105</td><td>4.18%↓</td><td>24619</td><td>0.34%↑</td><td>122724</td><td>4.06%↓</td><td>0.776</td><td>5.37%↓</td></tr><tr><td>DietCode (20%)</td><td>93268</td><td>8.23%↓</td><td>24508</td><td>0.11%↓</td><td>117776</td><td>7.92%↓</td><td>0.813</td><td>0.85%↓</td></tr><tr><td>DietCode (30%)</td><td>86672</td><td>14.61%↓</td><td>24211</td><td>1.32%↓</td><td>110883</td><td>13.32%↓</td><td>0.775</td><td>5.49%↓</td></tr><tr><td>DietCode (40%)</td><td>80196</td><td>20.98%↓</td><td>24536</td><td>0.00%↑</td><td>104732</td><td>18.14%↓</td><td>0.78</td><td>4.88%↓</td></tr><tr><td>DietCode (50%)</td><td>73759</td><td>27.76%↓</td><td>24169</td><td>1.49%↓</td><td>97928</td><td>23.44%↓</td><td>0.776</td><td>5.37%↓</td></tr><tr><td>SlimCode (10%)</td><td>96845</td><td>5.41%↓</td><td>25343</td><td>3.29%↑</td><td>122188</td><td>4.48%↓</td><td>0.79</td><td>3.66%↓</td></tr><tr><td>SlimCode (20%)</td><td>91736</td><td>10.40%↓</td><td>25217</td><td>2.99%↑</td><td>116953</td><td>8.57%↓</td><td>0.808</td><td>1.46%↓</td></tr><tr><td>SlimCode (30%)</td><td>85778</td><td>16.22%↓</td><td>24779</td><td>0.99%↑</td><td>110557</td><td>13.57%↓</td><td>0.789</td><td>3.78%↓</td></tr><tr><td>SlimCode (40%)</td><td>79686</td><td>22.17%↓</td><td>24734</td><td>0.81%↑</td><td>104420</td><td>18.37%↓</td><td>0.796</td><td>2.93%↓</td></tr><tr><td>SlimCode (50%)</td><td>73215</td><td>28.49%↓</td><td>24548</td><td>0.05%↑</td><td>97763</td><td>23.57%↓</td><td>0.763</td><td>6.95%↓</td></tr><tr><td>LeanCode (10%)</td><td>96978</td><td>5.28%↓</td><td>25843</td><td>5.33%↑</td><td>122821</td><td>3.99%↓</td><td>0.795</td><td>3.05%↓</td></tr><tr><td>LeanCode (20%)</td><td>91173</td><td>10.95%↓</td><td>25043</td><td>2.07%↑</td><td>116216</td><td>9.15%↓</td><td>0.798</td><td>2.68%↓</td></tr><tr><td>LeanCode (30%)</td><td>85202</td><td>16.78%↓</td><td>25454</td><td>3.74%↑</td><td>110656</td><td>13.50%↓</td><td>0.828</td><td>0.49%↑</td></tr><tr><td>LeanCode (40%)</td><td>79102</td><td>22.74%↓</td><td>25162</td><td>2.55%↑</td><td>104264</td><td>18.49%↓</td><td>0.793</td><td>3.54%↓</td></tr><tr><td>LeanCode (50%)</td><td>72887</td><td>28.81%↓</td><td>24741</td><td>0.84%↑</td><td>97628</td><td>23.68%↓</td><td>0.81</td><td>1.22%↓</td></tr></table>

Tables 5 and 6 present the results of the code simplification methods. We use the total number of tokens (the input and output tokens), instead of the prediction time for two reasons: 1) the fee charged for the usage of computing resource of GPT-4o is based on the total number of tokens (OPENAI-Pricing); 2) GPT-4 only offers API interfaces, making it difficult to accurately measure time due to multiple influencing factors. It’s worth noting that the input tokens include not only the code snippet but also the prompts. From the results, we made the following empirical observations:

a) Saving Computational Resources. As the simplified ratio increases, the total number of tokens decrease for all methods on both tasks in GPT-4o. This finding substantiates that code simplification leads to a reduction in resources required for Prompt-based LLM. This can be due the fact that GPT-4o requires a significant amount of analytical content by following a sequence of statements through reasoning, as opposed to solely producing binary responses. It shows that in the absence of generating analytical texts, the classification performance of GPT-4o is notably deficient (OpenAI, 2023). However, the total number of tokens decreases since the reduction in input tokens outweighs the increase in output tokens.

b) Performance on Code Summarization. Firstly, LEANCODE still slightly outperforms the baselines. We observe that for all methods, the BLEU-4 scores in Table 6 are approximately half of those in Table 3. This indicates that the descriptions from GPT-4o deviate more largely from the ground truth compared to those from CodeT5. GPT-4o lacks the capability to produce descriptions closely resembling the ground truth, as it did not undergo fine-tuning. The overlap of words between the generated description and the ground truth is considerably lower when compared to CodeT5.

Table 6: Results of removal methods on GPT-4 for Code Summarization (IT: Input Tokens, R-IT: Reduced Input Tokens (%), OT: Output Tokens, R-OT: Reduced Output Tokens (%); TT: Total Tokens, R-B: Reduced BLEU (%)) 

<table><tr><td>Removal method</td><td>IT</td><td>R-IT</td><td>OT</td><td>R-OT</td><td>TT</td><td>R-TT</td><td>BLEU</td><td>R-B</td></tr><tr><td>Base</td><td>78246</td><td>—</td><td>7668</td><td>—</td><td>85914</td><td>—</td><td>10.59</td><td>—</td></tr><tr><td>DietCode (10%)</td><td>75217</td><td>3.87%↓</td><td>7340</td><td>4.28%↓</td><td>82557</td><td>3.90%↓</td><td>10.80</td><td>1.98%↑</td></tr><tr><td>DietCode (20%)</td><td>72245</td><td>7.67%↓</td><td>7583</td><td>1.11%↓</td><td>79828</td><td>7.08%↓</td><td>10.21</td><td>3.59%↓</td></tr><tr><td>DietCode (30%)</td><td>66844</td><td>14.58%↓</td><td>7846</td><td>232%↑</td><td>74690</td><td>13.06%↓</td><td>10.12</td><td>4.44%↓</td></tr><tr><td>DietCode (40%)</td><td>61534</td><td>21.36%↓</td><td>8015</td><td>4.52%↑</td><td>69549</td><td>19.06%↓</td><td>9.95</td><td>6.04%↓</td></tr><tr><td>DietCode (50%)</td><td>56162</td><td>28.21%↓</td><td>7748</td><td>1.04%↑</td><td>63910</td><td>25.59%↓</td><td>9.69</td><td>8.50%↓</td></tr><tr><td>SlimCode (10%)</td><td>74912</td><td>4.26%↓</td><td>7426</td><td>3.16%↓</td><td>82338</td><td>4.16%↓</td><td>10.71</td><td>1.14%↑</td></tr><tr><td>SlimCode (20%)</td><td>70734</td><td>9.96%↓</td><td>7715</td><td>0.61%↑</td><td>78449</td><td>8.69%↓</td><td>10.89</td><td>2.83%↑</td></tr><tr><td>SlimCode (30%)</td><td>65648</td><td>16.10%↓</td><td>7633</td><td>0.46%↓</td><td>73281</td><td>14.70%↓</td><td>10.71</td><td>1.14%↑</td></tr><tr><td>SlimCode (40%)</td><td>60585</td><td>22.57%↓</td><td>7773</td><td>1.37%↑</td><td>68358</td><td>20.43%↓</td><td>10.62</td><td>0.28%↑</td></tr><tr><td>SlimCode (50%)</td><td>55335</td><td>29.28%↓</td><td>7496</td><td>2.24%↓</td><td>62831</td><td>26.86%↓</td><td>10.60</td><td>0.09%↑</td></tr><tr><td>LeanCode (10%)</td><td>74938</td><td>4.23%↓</td><td>7487</td><td>2.36%↓</td><td>82425</td><td>4.06%↓</td><td>11.11</td><td>4.91%↑</td></tr><tr><td>LeanCode (20%)</td><td>70207</td><td>10.28%↓</td><td>8278</td><td>7.95%↑</td><td>78485</td><td>8.66%↓</td><td>10.69</td><td>1.09%↑</td></tr><tr><td>LeanCode (30%)</td><td>65322</td><td>16.51%↓</td><td>7607</td><td>0.80%↓</td><td>72929</td><td>15.11%↓</td><td>10.77</td><td>1.70%↑</td></tr><tr><td>LeanCode (40%)</td><td>60296</td><td>22.93%↓</td><td>8163</td><td>6.45%↑</td><td>68459</td><td>20.32%↓</td><td>10.90</td><td>2.93%↑</td></tr><tr><td>LeanCode (50%)</td><td>55261</td><td>29.37%↓</td><td>8442</td><td>10.08%↑</td><td>63703</td><td>25.86%↓</td><td>10.70</td><td>1.04%↑</td></tr></table>

c) Performance on Code Search. From Table 5, the base GPT-4o yielded good results, even without model fine-tuning. Interestingly, as the simplified ratio increases, Precision scores do not show a consistent decline. Moreover, all methods produced closely clustered results, with LEANCODE maintaining a slight edge. A plausible explanation is that GPT-4o was trained on a similar dataset. Consequently, GPT-4o can identify crucial tokens for generating judgments, resulting in better inference.

# 5 Related Work

Pre-trained Models. Pre-trained models have significantly advanced code models (Feng et al., 2020b; Guo et al., 2021; Karampatsis and Sutton, 2020; Guo et al., 2022). They excel in various tasks such as code generation (Wang et al., 2021b; Clement et al., 2020; Wang et al., 2023b), defect detection (Wang et al., 2023a), code summarization (Ahmed and Devanbu, 2022; Jiang et al., 2021), and code search (Wang et al., 2021a; Niu et al., 2022). Researchers have investigated pre-trained models for code understanding (Ahmad et al., 2021; Mastropaolo et al., 2021). For instance, Karmakar and Robbes (Karmakar and Robbes, 2021) conducted four probing tasks on pre-trained models to assess their ability to learn various aspects of source code. Wan et al. (Wan et al., 2022) reported that Transformer attention mechanisms can capture high-level structures within source code. Moreover, Autofocus (Bui et al., 2019) is a method to determine the most relevant code by measuring statement relevance using attention weights from a GGNN (Allamanis et al., 2018).

Program Simplification. SIVAND (Rabin et al., 2021) and P2IM (Suneja et al., 2021) typically build upon the delta debugging prototype (Zeller and Hildebrandt, 2002), which involves treating a code snippet and an auxiliary model (e.g., code2vec) as inputs. The model segments the code snippet into fragments and use each as inputs. If a fragment achieves a satisfactory score, it is further divided. This process continues until the subset’s performance fails to meet the desired score, resulting in the smallest snippet that satisfies the goal.

# 6 Conclusion

In this study, we introduce LEANCODE, a novel code simplification approach that harnesses code contexts to utilize attention scores of pre-trained models for representing the importance levels of each token of input. We advocate for the selective removal of tokens based on the average contextaware attention scores. rather than relying on average scores across all inputs. We evaluated LEANCODE in code search and code summarization tasks, experimental results show its superiority over the SOTA DietCode and SlimCode, achieving improvements of up to 60% and 16% for code search and 29% and 27% for code summarization.

# 7 Limitations

LEANCODE still has the following limitations"

Programming Language: One limitation of our current model is its exclusive application to Java, which restricts its use and effectiveness in other programming languages. Although related literature reports similar effects in other languages, this limitation highlights the need for future work to expand the model’s capabilities across multiple programming languages.

External Validity: Our experiments were on three models (CodeBERT, CodeT5, GPT-4o), and two tasks of code search (text-to-code) and code summarization (code-to-text). Despite the consistent findings, for generalizability, future experiments are needed on a wider variety of models with other paradigms and in other code-related tasks. Our dataset might not be representative. However, for a fair comparison, we used the same datasets and two tasks as in DIETCODE and SLIMCODE.

Internal Validity: When measuring the time, we acknowledge that there might be other external factors involving hardware (e.g., GPUs), operating system delays, etc. However, inference time for the models, CodeBERT and CodeT5 is stable in the same controlled environment.

# Acknowledgments

The third author, Tien N. Nguyen, was supported in part by the US National Science Foundation (NSF) grant CNS-2120386 and the National Security Agency (NSA) grant NCAE-C-002-2021.

# References

Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021. Unified pre-training for program understanding and generation. arXiv preprint arXiv:2103.06333.   
Toufique Ahmed and Premkumar Devanbu. 2022. Multilingual training for software engineering. In Proceedings of the 44th International Conference on Software Engineering, ICSE ’22, page 1443–1455, New York, NY, USA. Association for Computing Machinery.   
Miltiadis Allamanis, Marc Brockschmidt, and Mahmoud Khademi. 2018. Learning to represent programs with graphs. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net.

ChatGPT. OpenAI. https://openai.com/.   
Nghi DQ Bui, Yijun Yu, and Lingxiao Jiang. 2019. Autofocus: interpreting attention-based neural networks by code perturbation. In 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 38–41. IEEE.   
Colin B Clement, Dawn Drain, Jonathan Timcheck, Alexey Svyatkovskiy, and Neel Sundaresan. 2020. Pymt5: multi-mode translation of natural language and python code with transformers. arXiv preprint arXiv:2010.03150.   
Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020a. Codebert: A pre-trained model for programming and natural languages. In Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, volume EMNLP 2020 of Findings of ACL, pages 1536–1547. Association for Computational Linguistics.   
Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020b. Code-BERT: A pre-trained model for programming and natural languages. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1536–1547, Online. Association for Computational Linguistics.   
Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022. Unixcoder: Unified crossmodal pre-training for code representation. arXiv preprint arXiv:2203.03850.   
Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou, Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, Michele Tufano, Shao Kun Deng, Colin B. Clement, Dawn Drain, Neel Sundaresan, Jian Yin, Daxin Jiang, and Ming Zhou. 2021. Graphcodebert: Pre-training code representations with data flow. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.   
Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. Codesearchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436.   
Raisa Islam and Owana Marzia Moushi. 2024. Gpt-4o: The cutting-edge advancement in multimodal llm. Authorea Preprints.   
Xue Jiang, Zhuoran Zheng, Chen Lyu, Liang Li, and Lei Lyu. 2021. Treebert: A tree-based pre-trained model for programming language. In Uncertainty in Artificial Intelligence, pages 54–63. PMLR.   
Rafael-Michael Karampatsis and Charles Sutton. 2020. Scelmo: Source code embeddings from language models. arXiv preprint arXiv:2004.13214.

Anjan Karmakar and Romain Robbes. 2021. What do pre-trained code models know about code? In 2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 1332– 1336. IEEE.   
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.   
Antonio Mastropaolo, Simone Scalabrino, Nathan Cooper, David Nader Palacio, Denys Poshyvanyk, Rocco Oliveto, and Gabriele Bavota. 2021. Studying the usage of text-to-text transfer transformer to support code-related tasks. In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pages 336–347.   
Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint.   
Changan Niu, Chuanyi Li, Bin Luo, and Vincent Ng. 2022. Deep learning meets software engineering: A survey on pre-trained models of source code. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pages 5546–5555. ijcai.org.   
OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.   
OPENAI-Pricing. Openai-pricing. https://openai.com/api/pricing/.   
Md Rafiqul Islam Rabin, Vincent J. Hellendoorn, and Mohammad Amin Alipour. 2021. Understanding neural code intelligence through program simplification. In Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/FSE 2021, page 441–452, New York, NY, USA. Association for Computing Machinery.   
Sahil Suneja, Yunhui Zheng, Yufan Zhuang, Jim A. Laredo, and Alessandro Morari. 2021. Probing model signal-awareness via prediction-preserving input minimization. In Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/FSE 2021, page 945–955, New York, NY, USA. Association for Computing Machinery.   
Yao Wan, Wei Zhao, Hongyu Zhang, Yulei Sui, Guandong Xu, and Hai Jin. 2022. What do they capture? a structural analysis of pre-trained language models for source code. In Proceedings of the 44th International Conference on Software Engineering, ICSE ’22, page 2377–2388, New York, NY, USA. Association for Computing Machinery.

Wenbo Wang, Tien N. Nguyen, Shaohua Wang, Yi Li, Jiyuan Zhang, and Aashish Yadavally. 2023a. Deepvd: Toward class-separation features for neural network vulnerability detection. In Proceedings of the 45th International Conference on Software Engineering, ICSE ’23, page 2249–2261. IEEE Press.   
Xin Wang, Yasheng Wang, Fei Mi, Pingyi Zhou, Yao Wan, Xiao Liu, Li Li, Hao Wu, Jin Liu, and Xin Jiang. 2021a. Syncobert: Syntax-guided multi-modal contrastive pre-training for code representation. arXiv preprint arXiv:2108.04556.   
Yan Wang, Xiaoning Li, Tien N Nguyen, Shaohua Wang, Chao Ni, and Ling Ding. 2024. Natural is the best: Model-agnostic code simplification for pretrained large language models. Proceedings of the ACM on Software Engineering, 1(FSE):586–608.   
Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. 2023b. Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv:2305.07922.   
Yue Wang, Weishi Wang, Shafiq Joty, and Steven C.H. Hoi. 2021b. CodeT5: Identifier-aware unified pretrained encoder-decoder models for code understanding and generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8696–8708, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.   
Andreas Zeller and Ralf Hildebrandt. 2002. Simplifying and isolating failure-inducing input. IEEE Transactions on Software Engineering, 28(2):183–200.   
Zhaowei Zhang, Hongyu Zhang, Beijun Shen, and Xiaodong Gu. 2022. Diet code is healthy: Simplifying programs for pre-trained models of code. In Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/FSE 2022, page 1073–1084, New York, NY, USA. Association for Computing Machinery.

# Appendix

# A DietCode discarded encoder-decoder attention scores

For code summarization like task, DietCode still uses the attention scores of this encoder to signify the importance of code tokens. However, it overlooks a crucial aspect: the encoder-decoder attention. Fig. 3b illustrates this encoder-decoder attention mechanism during the generation for the code in Fig. 4. As shown, the vector $v _ { b u b b l e }$ should receive more attention from the decoder, and the product of $v _ { b u b b l e }$ and normalized attention score $s _ { i + 1 }$ is integrated into the final vector $v _ { a }$ to generate the token ‘bubble’ (after the previous tokens

![](images/4ccc6632872429430db367eeb66657d4ca4a8e98d806e065260d64129b9a624a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Matched/Unmatched"] --> B["fully-connected layer"]
    B --> C["encoder"]
    C --> D["CLS attention in self attention"]
    D --> E["CLS def bubbleSort(arr): n = len(arr)...The method is a bubble...."]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
```
</details>

(a) Example of the CLS-attention. It is the self-attention of the ‘CLS’ token that is calculated by the weighted sum of attention over all tokens in the input sequence to derive the representation of the CLS token. The input is the concatenation of a code snippet and a description, the output is ‘matched’ or ‘unmatched’, v is the vector of each token and ss are the attention scores of ‘CLS’ token on other tokens based on Equation 5.

![](images/39fe57a21f39a2a78c21b63dff16d4888b9358157a5507ec2e4b112c33cc42b3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["&quot;Encoder"] --> B["S_i"]
    B --> C["V_def"]
    B --> D["V_bubble"]
    B --> E["..."]
    B --> F["bubbles"]
    G["decoder self-attention"] --> H["V_a"]
    H --> I["V_{<s>}"]
    H --> J["V_The"]
    H --> K["V_method"]
    H --> L["V_vis"]
    M["def bubble_sort(arr): n = len(arr)..."] --> N["The method is a"]
    style A fill:#f9f,stroke:#333
    style G fill:#ccf,stroke:#333
    style M fill:#cfc,stroke:#333
```
</details>

(b) Example of Encoder-Decoder attention. The attention is computed by paying attention to the encoder’s output while also maintaining self-attention within the decoder layers to generate context-aware representations for the target sequence. The vector $\cdot { \boldsymbol { v } } _ { a } \cdot \mathbf { \ Y }$ first performs self-attention with the generated vectors v<s>, vT he, vmethod, $v _ { i s }$ then absorbs the vectors of input tokens through encoder-decoder’s weights to generate next token "bubble".

Figure 3: CLS and Encoder-Decoder attention scores based on the example in Fig. 4.

have already been generated). Thus, input vectors with higher decoder-attention scores typically hold greater importance for sequence-to-sequence tasks and can effectively discern the tokens’ significance. However, DietCode uses only self-attention scores and discarded these encoder-decoder attention scores.

# B An Example of CLS and encoder-decoder attentions

Let us use an example to illustrate the problem and to motivate our work (Sections 2 and 3).

The state-of-the-art DIETCODE (Zhang et al., 2022) uses the accumulated attention scores to indicate the importance of each input token for downstream tasks. For classification tasks, DI-ETCODE uses a pre-trained encoder (e.g., Code-BERT or CodeT5) in conjunction with a fullyconnected layer for code search. For generation tasks, e.g., code summarization, it uses a sequenceto-sequence structure that combines either the encoder of CodeBERT or CodeT5 with a Transformer decoder or CodeT5 decoder. For both tasks, the calculation of the importance of a token is only based on the self-attention scores of the encoder in DIETCODE. Specifically, the accumulated self attention score of an input token is calculated in Equation 4. Given an input sequence $X = [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ]$ where $x _ { i }$ is a d-dimensional vector. First, Query $( Q = [ q _ { 1 } , q _ { 2 } , . . . , q _ { n } ] )$ , Key $( K = [ k _ { 1 } , k _ { 2 } , \ldots , k _ { n } ] )$ , and Value $( V = [ v _ { 1 } , v _ { 2 } , \ldots ,$ $v _ { n } ] )$ vectors for a token are generated via three linear mappings. These mappings are implemented by the weight matrices $W ^ { \bar { Q } } , W ^ { K }$ , and $W ^ { V }$ , which are learnable parameters. For any token xi, its key vector is dot-producted with every query vector in the sequence, yielding an accumulated attention score that is scaled down by the square root of d as shown in Equation (4).

$$
s _ {i} = \frac {\sum_ {j = 1} ^ {n} q _ {j} \cdot k _ {i}}{\sqrt {d}}. \tag {4}
$$

Each accumulated attention score $s _ { i }$ measures how the corresponding token gains attention from other tokens in the input sequence. The example of the accumulated self-attention based for the example in Fig. 4 is shown in Fig. 6.

For the classification tasks, e.g., code search, only the ‘CLS’ token is sent into the fully-connected layer for label prediction as shown in Fig. 3a. Thus, the tokens to which the ‘CLS’ token pays attention could play more crucial roles in the classification task than other tokens. For the CLS attention score $s _ { i }$ for corresponding token, it can be calculated in Equation 5:

$$
s _ {i} = \frac {q _ {c l s} \cdot k _ {i}}{\sqrt {d}}, (1 \leq i \leq n) \tag {5}
$$

In code search, the model assesses the correspondence between the code and its accompanying description. Via pre-training or fine-tuning, the model acquires an understanding of the interrelation among tokens in bi-modal data via selfattention mechanisms. For instance, the code token $" d e f "$ exhibits considerable attention toward tokens like "the", "method", "is", "a", and "basic" within the description. Conversely, the token "method" in the description concentrates its attention scores on the tokens within the method signature, such as $" d e f "$ , "bubble", and "method". Meanwhile, tokens such as ${ } + , { } = , { }$ and if lack clear respective tokens in

The code of bubble sort:   
```python
def bubbleSort(arr):
    n = len(arr)
    for i in range(n - 1):
    for j in range(n - 1 - i):
    if arr[j] > arr[j + 1]:
    arr[j], arr[j + 1] = arr[j + 1], arr[j] 
```

The description: "The method is a basic implementation of the bubble sort algorithm in Python. Bubble sort works by repeatedly swapping adjacent elements if they are in the wrong order until the entire array is sorted."

Figure 4: An Example of Bubble-Sort Code self attention in encoder   
![](images/bfdd75d800f605fb7c37066231df5f8e56bb5eeb500eff46aac83976bfe76741.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["V_CLS"] --> B["V_def"]
    B --> C["..."]
    C --> D["q_1·k_i"]
    D --> E["V_len"]
    E --> F["..."]
    F --> G["V_bubble"]
    H["s_len = Σ_{j=1}^n q_j·k_i / √d"] --> E
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    note["CLS def bubbleSort(arr): n = len(arr)... The method is a bubble..."
```
</details>

Figure 5: Example of accumulated self attention in DIETCODE on Fig. 4. Dotted lines with different colors represent the selfattentions of different tokens. $s _ { l e n }$ is the accumulated selfattention score of the token ‘len’.   
Figure 6: An Example of Accumulated Self-Attention in DIETCODE on Fig. 4.

the text. Thus, the removal of such tokens may not notably impact the matching outcomes, as the models heavily rely on mapping information.

Figures 7a) and b) represent CLS attention scores on the tokens in the code and the description, respectively. The token <s> serves as the ‘CLS’ token. The deeper the color, the higher the value of the attention score. As seen in Fig. 7a), the CLS attention (the self attention of the token ‘CLS’) assigns greater importance to the tokens with more bi-modal mappings to form its vector representation. The ‘CLS’ token allocates significant attention to most tokens in the textual description, while emphasizing on the code tokens primarily within the method signature.

Similarly, for code summarization, the bi-modal mappings are apparent. In Fig. 7c), when the next generated token is the token "bubble" (part of the method name) in the description, the encoderdecoder attention should emphasize on the tokens of method signature in the code. Fig. 7c) shows the encoder-decoder attention. As seen, all tokens in the method signature are paid much attention than others. More specific, with the current query vector q at position t in the decoder, the Encoder-Decoder score for each input token is calculated via Equation (6):

$$
s _ {i} = \frac {q _ {t} \cdot k _ {i}}{\sqrt {d}}, (1 \leq i \leq n) \tag {6}
$$

Thus, maintaining the mappings between bi-modal data is crucial for bi-modal tasks, which can be accomplished using Encoder-Decoder attention. However, DIETCODE leverages the attention across all tokens, without giving emphasis to bimodal mappings.

# C Detailed Results and Analysis of 3 Research Questions in Preliminary Empirical Study

# C.1 (RQ-1) What important tokens do CLS attentions emphasize on?

In Table 7, the five leftmost columns show the categories, the maximum of CLS attention scores, the minimum of CLS attention scores, the averages and variances of the global attention scores of tokens for each category. The global attention scores are used to calculate the average attention score of each token within a statement category. The last two columns of Table 7 display our proposed averages of category-local attention scores, and the average variances of CLS attention scores of tokens grouped by statement categories. There are 21 categories used by DIETCODE, collectively representing over 95% of the statements in the dataset. In LEANCODE, we use these categories as contexts for tokens. They offer relatively straightforward setups for bi-modal mappings.

![](images/092d9d685b0c5849251eec70a3b59ccae4efa48bca4a7b6aa8c387ac2122cebf.jpg)  
Figure 7: ‘CLS’ and encoder-decoder attentions on the input of the example of bubble sort. (<s> is the CLS token. Deeper color shows higher attention score. Fig. 7a) and b) represent the CLS attentions on the code tokens and the description, and Fig. 7c) shows the encoder-decoder attention for the generation of the "bubble" token.)

Analyzing the columns reveals that the categorylocal attention average of tokens within method signatures significantly surpasses that of others, even exceeding the second largest class (‘return’ statements) by a factor of eight. The Return class typically comprises the names of returned variables, which often convey crucial functional information. Logging and Annotation classes rank third and fourth, respectively, as they frequently encompass function-related data. Variable Declaration and Function Invocation are next, predominantly due to inclusion of variables or callee names.

Conversely, the Break, Case, and Continue exhibit the lowest category-local attention average. This is attributed to the fact that Break and Continue statements have keyword information, while Case statements, though potentially containing conditions, may be too detailed to establish meaningful bimodal mappings with the description. Moreover, the average variance is directly proportional to the average attention scores, indicating that higher attention scores correspond to increased variance.

Comparing our proposed averages (last two columns) with the global averages (the $4 ^ { t h }$ and $5 ^ { t h }$ columns) in Table 7, except Method signature, the averages of attention scores/variances for the context-aware, category-local attentions as we propose are reduced from 0.55/5 times to 3.3/844 times in comparison to the global averages within each statement category. The variance of Method Signature increases 3.3 times when the average of attention scores goes up more than 4 times. Thus, the categories of the statements successfully reflect the context of each token.

# C.2 (RQ-2) What important tokens do encoder-decoder attentions emphasize about?

Table 8 shows the average of the Encoder-Decoder attention scores of tokens based on statement types, called Encoder-Decoder category-local attention average. Unlike CLS attention, each token in the input can have multiple Encoder-Decoder attention scores, i.e., for each generated token, the decoder calculates an attention score for each token in the input. For example, Fig. 3b shows the Encoder-Decoder attention scores for the input at the generation of the token "bubble". Thus, the largest attention score is chosen as the attention score.

As depicted in Table 8, the categories with the highest and lowest importance remain consistent as ‘Method signature’, Return, Continue, and Case, respectively. However, the disparity between them has diminished. Certain categories, e.g., Synchronized, For, and Throw, have surpassed ‘Logging’, ‘Variable Declaration’, and ‘Function Invocation’, securing the $3 ^ { r d }$ , $4 ^ { t h }$ , and $5 ^ { t h }$ places in importance. This shift indicates a redistribution of importance across categories. The Encoder-Decoder attention scores are generated in conjunction with the description. In the instances where the description contains intricate function details, these tokens garner high attention scores, facilitating the establishment of bi-modal mappings. For code search, the significance of details within the code (e.g., Throw statements) is lower compared to the broader functional description (e.g., ‘Method signature’).

Similar to Table 7, except ‘Method’ signature, comparing the $4 ^ { t h }$ and $5 ^ { t h }$ columns with the last

Table 7: (RQ-1) Statistics of CLS attention scores on 0.9M training dataset. (Max/Min: the max/min of CLS attention scores in each category; Global/Global\_variance: the average/variance of global attention scores for each category; Category-local/Local\_variance: the averages/variance of category-local attention scores.) 

<table><tr><td>Category</td><td>Max</td><td>Min</td><td>Global</td><td>Global_variance</td><td>Category-local</td><td>Local_variance</td></tr><tr><td>Annotation</td><td>13.64774</td><td>0.00369</td><td>0.22834</td><td>1.09668</td><td>0.14712</td><td>0.25699</td></tr><tr><td>Arithmetic</td><td>3.83190</td><td>0.00269</td><td>0.23538</td><td>1.15031</td><td>0.05800</td><td>0.00915</td></tr><tr><td>Variable Declaration</td><td>10.49360</td><td>0.00353</td><td>0.24737</td><td>1.18534</td><td>0.10431</td><td>0.10301</td></tr><tr><td>Function Invocation</td><td>5.25585</td><td>0.00349</td><td>0.25464</td><td>1.23739</td><td>0.10638</td><td>0.09292</td></tr><tr><td>Return</td><td>6.59062</td><td>0.00353</td><td>0.27754</td><td>1.33069</td><td>0.20165</td><td>0.17654</td></tr><tr><td>Switch</td><td>4.36353</td><td>0.00352</td><td>0.23530</td><td>1.02716</td><td>0.07194</td><td>0.00936</td></tr><tr><td>Break</td><td>0.83018</td><td>0.00536</td><td>0.23083</td><td>1.02695</td><td>0.04734</td><td>0.00253</td></tr><tr><td>Setter</td><td>2.85442</td><td>0.00196</td><td>0.24565</td><td>1.16452</td><td>0.07033</td><td>0.04076</td></tr><tr><td>Synchronized</td><td>1.02363</td><td>0.00526</td><td>0.23753</td><td>1.00888</td><td>0.08507</td><td>0.01309</td></tr><tr><td>Try</td><td>4.33814</td><td>0.00371</td><td>0.24224</td><td>1.05390</td><td>0.08925</td><td>0.02697</td></tr><tr><td>Catch</td><td>2.07879</td><td>0.00385</td><td>0.24606</td><td>0.90369</td><td>0.05224</td><td>0.00641</td></tr><tr><td>Method Signature</td><td>29.17083</td><td>0.00353</td><td>0.33731</td><td>1.56490</td><td>1.74525</td><td>6.69616</td></tr><tr><td>Finally</td><td>0.53785</td><td>0.00439</td><td>0.22154</td><td>1.41311</td><td>0.09047</td><td>0.00939</td></tr><tr><td>Getter</td><td>7.59034</td><td>0.00373</td><td>0.24690</td><td>1.18191</td><td>0.06407</td><td>0.03392</td></tr><tr><td>Throw</td><td>5.20869</td><td>0.00346</td><td>0.23824</td><td>1.12604</td><td>0.08951</td><td>0.02551</td></tr><tr><td>Case</td><td>2.70762</td><td>0.00338</td><td>0.23824</td><td>1.15401</td><td>0.03953</td><td>0.00673</td></tr><tr><td>While</td><td>2.57405</td><td>0.00311</td><td>0.22953</td><td>1.06577</td><td>0.05870</td><td>0.01045</td></tr><tr><td>Continue</td><td>0.44534</td><td>0.00662</td><td>0.25256</td><td>1.35214</td><td>0.04609</td><td>0.00160</td></tr><tr><td>If Condition</td><td>4.03009</td><td>0.00262</td><td>0.24831</td><td>1.21111</td><td>0.08341</td><td>0.04783</td></tr><tr><td>For</td><td>22.29719</td><td>0.00348</td><td>0.24910</td><td>1.11164</td><td>0.07938</td><td>0.02098</td></tr><tr><td>Logging</td><td>4.90106</td><td>0.00278</td><td>0.24053</td><td>1.15288</td><td>0.14864</td><td>0.11938</td></tr></table>

Table 8: (RQ-2) Statistics of encoder-decoder attention scores based on 0.16M training dataset. (Max/Min: the maximum/minimum of encoder-decoder attention scores in each category; Global/Global\_variance: the average/variance of the global attention scores of tokens for each category; Category-local/Local\_variance: the averages/variance of category-local attention scores.) 

<table><tr><td>Category</td><td>Max</td><td>Min</td><td>Global</td><td>Global_variance</td><td>Category-local</td><td>Local_variance</td></tr><tr><td>Annotation</td><td>7.94088</td><td>0.31593</td><td>2.61080</td><td>13.75766</td><td>1.54646</td><td>0.08766</td></tr><tr><td>Arithmetic</td><td>37.44108</td><td>0.06535</td><td>2.68657</td><td>15.41490</td><td>2.30496</td><td>2.52065</td></tr><tr><td>Variable Declaration</td><td>65.53831</td><td>0.08960</td><td>2.86623</td><td>15.63358</td><td>2.69239</td><td>7.96826</td></tr><tr><td>Function Invocation</td><td>63.96667</td><td>0.00918</td><td>2.86456</td><td>15.94000</td><td>2.80177</td><td>8.54368</td></tr><tr><td>Return</td><td>55.23188</td><td>0.09667</td><td>3.08082</td><td>17.61045</td><td>4.75692</td><td>16.01764</td></tr><tr><td>Switch</td><td>30.02598</td><td>0.07416</td><td>2.70932</td><td>16.36071</td><td>2.40672</td><td>2.63213</td></tr><tr><td>Break</td><td>28.02130</td><td>0.04057</td><td>2.64491</td><td>16.43219</td><td>2.66594</td><td>1.21296</td></tr><tr><td>Setter</td><td>69.05594</td><td>0.02932</td><td>2.84634</td><td>17.25399</td><td>2.32736</td><td>5.09966</td></tr><tr><td>Synchronized</td><td>78.08508</td><td>0.04113</td><td>2.84346</td><td>17.08405</td><td>3.11007</td><td>3.03446</td></tr><tr><td>Try</td><td>78.26600</td><td>0.02602</td><td>2.82429</td><td>17.31111</td><td>2.45927</td><td>2.69108</td></tr><tr><td>Catch</td><td>34.98584</td><td>0.07141</td><td>3.01365</td><td>19.80343</td><td>2.44455</td><td>4.18487</td></tr><tr><td>Method Signature</td><td>91.68832</td><td>0.13817</td><td>3.29116</td><td>18.21317</td><td>5.91461</td><td>30.92008</td></tr><tr><td>Finally</td><td>10.49109</td><td>0.74180</td><td>2.38186</td><td>7.77507</td><td>2.98907</td><td>1.73634</td></tr><tr><td>Getter</td><td>68.48614</td><td>0.02599</td><td>2.87695</td><td>16.58745</td><td>2.57884</td><td>6.41530</td></tr><tr><td>Throw</td><td>87.67154</td><td>0.05743</td><td>2.79540</td><td>16.04224</td><td>3.09842</td><td>8.12932</td></tr><tr><td>Case</td><td>23.25289</td><td>0.03370</td><td>2.74530</td><td>16.10519</td><td>1.79642</td><td>1.54551</td></tr><tr><td>While</td><td>67.68087</td><td>0.04015</td><td>2.69720</td><td>15.52064</td><td>2.40500</td><td>3.14009</td></tr><tr><td>Continue</td><td>9.85333</td><td>0.26809</td><td>2.48510</td><td>12.64186</td><td>1.72747</td><td>0.36762</td></tr><tr><td>If Condition</td><td>57.87812</td><td>0.05160</td><td>2.83536</td><td>15.84334</td><td>2.49570</td><td>5.97373</td></tr><tr><td>For</td><td>60.61899</td><td>0.03448</td><td>2.90877</td><td>17.21421</td><td>2.99445</td><td>6.88533</td></tr><tr><td>Logging</td><td>65.63157</td><td>0.03885</td><td>2.77210</td><td>15.53356</td><td>2.89339</td><td>8.41999</td></tr></table>

two columns, our proposed averages of attention scores/variances for context-aware, category-local attentions are much reduced from 0.1 to 156 times compared to the global averages within each category. However, the change from the global attention average to the category-local average is insignificant. This could be due to 1) our selection of the largest attention score as the representative, and 2) the difference in the Encoder-Decoder attention scores is not substantial across categories.

# C.3 (RQ-3) Do the averages of self-attention scores reflect the CLS attentions and the Encoder-Decoder attentions?

Our answer is ‘No’. The accumulated attention scores from the self-attention (as used in DIETCODE) is for pre-training and cannot reflect and substitute for those from the CLS and Encoder-Decoder attentions. i.e., the self-attention is used for pre-trained tasks and vectored general representations, not directly for downstream tasks. For elaboration, these attention schemes are for different tasks. The self attention is for pre-training tasks, while CLS attention is for fine-tuning downstream discriminative tasks, and the Encoder-Decoder attention is for downstream sequence-to-sequence generation tasks. In fact, the encoders of Code-BERT and CodeT5 have been trained in multiple pre-trained tasks. CodeBERT is pre-trained with two objectives: Masked Language Model (MLM, bimodal data) and Replaced Token Detection (RTD, unimodal data). In MLM, the model is trained to predict the identity of tokens that have been randomly masked in the input sequence. In RTD, the model is given an input sequence where some tokens have been replaced with incorrect ones. The model task is to predict which tokens are the original ones and which have been replaced. These objectives enable the model to capture the long dependencies between tokens in bimodal sequence (MLM) and unimodel (RTD) to obtain general representations via the self attention. Thus, the averages of self-attention scores cannot replace the CLS and Encoder-Decoder attentions. The latter attentions are directly applied to downstream tasks (Fig. 3).

To illustrate our above answer, Figures 8a) and 8b) show the heatmaps of the accumulated selfattention (in DIETCODE) and the Encoder-Decoder attention for the bubblesort example. Each row is one statement without indentation (except that lines 5-6 contain one statement since it is too long to show). The darker color means higher attention scores, i.e., the tokens are more important.

![](images/a1cf541b17d66216368bbd723bd27dcfe459dfc715a815b2d85b40e68e2b32d4.jpg)

![](images/0932a0e42ab6807f41fe66e03f7803fa9eedb3f48e155bd888c14d46fc03bf90.jpg)  
Figure 8: Bubble-sort: Heatmaps of (a) accumulated self-attention scores, (b) Encoder-Decoder attention scores

For comparison, the colors of the cells in Fig. 8a) (accumulated self-attention used by DIETCODE) have the shades in-between the ones of Figure 7a) for ‘CLS’ attention and Fig. 8b) for Encoder-Decoder attention. Specifically, for both method signature and method body, both contain tokens of higher importance as in Fig. 8b). The importance of tokens in the method signature is slightly higher than that of tokens in the method body in Fig. 7a). The above situation mainly stems from MLM and RTD tasks needing to do token-level prediction in code, thus requiring to pay attention on tokens in the method body. Keywords and separators can have high attention weights as they play important roles in predicting tokens (Zhang et al., 2022). In addition, since tokens in the method signature could contain more important ‘guidance’ for prediction, their attention scores could be larger.

In Fig. 8b), at each text token generation iteration, every code token has one attention score, and here the largest attention score in all iterations is used since the largest score represents the most important degree that the token has played in the generation process. As seen, the method body and signature are almost equally important, and import tokens are more evenly distributed for this code snippet. Namely, different generated tokens pay attention to different code tokens. This is largely different from the CLS attention distribution in code search, as shown in Fig. 7a), which emphasizes on the tokens mainly in the method signature.

Table 9: Dataset Statistics (Total: # code snippets, Avg/- Max/Min: the average/max/min # of tokens in a code snippet or description, tr: training, val: validation, test: test dataset, Search: code search, Sum: summarization). 

<table><tr><td rowspan="2">Dataset</td><td colspan="4">Code Snippet</td><td colspan="3">Code Description</td></tr><tr><td>Total</td><td>Avg</td><td>Max</td><td>Min</td><td>Avg</td><td>Max</td><td>Min</td></tr><tr><td>Search_tr</td><td>908,886</td><td>112.67</td><td>68,278</td><td>20</td><td>19.2</td><td>3439</td><td>1</td></tr><tr><td>Search_val</td><td>30,655</td><td>95.57</td><td>3,092</td><td>21</td><td>19.05</td><td>521</td><td>1</td></tr><tr><td>Search_test</td><td>26,909</td><td>113.42</td><td>5,542</td><td>20</td><td>20.22</td><td>709</td><td>1</td></tr><tr><td>Sum_tr</td><td>164,923</td><td>100.99</td><td>512</td><td>17</td><td>13.25</td><td>175</td><td>3</td></tr><tr><td>Sum_val</td><td>5,183</td><td>90.79</td><td>501</td><td>18</td><td>13.39</td><td>147</td><td>3</td></tr><tr><td>Sum_test</td><td>10,955</td><td>100.06</td><td>512</td><td>20</td><td>12.71</td><td>111</td><td>3</td></tr></table>

Table 10: Results of LEANCODE with DIETCODE’s removal algorithm (10%-50% removal for each snippet, BLEU: BLEU-4 values, R-B:Reduced BLEU-4, R-M: Reduced MRR) 

<table><tr><td colspan="5">Code Search</td><td colspan="4">Code Summarization</td></tr><tr><td rowspan="2">Ratio</td><td colspan="2">CodeBERT</td><td colspan="2">CodeT5</td><td colspan="2">CodeBERT</td><td colspan="2">CodeT5</td></tr><tr><td>MRR</td><td>R-M</td><td>MRR</td><td>R-M</td><td>BLUE</td><td>R-B</td><td>BLUE</td><td>R-B</td></tr><tr><td>Base</td><td>0.726</td><td>—</td><td>0.747</td><td>—</td><td>18.25</td><td>—</td><td>20.55</td><td>—</td></tr><tr><td>10%</td><td>0.701</td><td>3.44%↓</td><td>0.723</td><td>3.21%↓</td><td>17.40</td><td>4.66%↓</td><td>18.53</td><td>9.83%↓</td></tr><tr><td>20%</td><td>0.703</td><td>3.17%↓</td><td>0.717</td><td>4.01%↓</td><td>17.34</td><td>4.97%↓</td><td>18.18</td><td>11.53%↓</td></tr><tr><td>30%</td><td>0.702</td><td>3.31%↓</td><td>0.712</td><td>4.69%↓</td><td>17.31</td><td>5.15%↓</td><td>18.08</td><td>12.02%↓</td></tr><tr><td>40%</td><td>0.696</td><td>4.13%↓</td><td>0.713</td><td>4.55%↓</td><td>17.09</td><td>6.35%↓</td><td>18.03</td><td>12.26%↓</td></tr><tr><td>50%</td><td>0.682</td><td>6.06%↓</td><td>0.695</td><td>6.96%↓</td><td>16.73</td><td>8.32%↓</td><td>17.63</td><td>14.20%↓</td></tr></table>

# D Dataset Statistics

Table 9 shows the detailed dataset statistics.

# E Replacement Study

LEANCODE mainly consists of two aspects: token weights and a token removal algorithm (Algorithm 1). Here, we aim to determine how each of them contributes to LEANCODE’s performance. Our procedure is to replace LEANCODE’s removal algorithm in LEANCODE with DIETCODE’s. Then, comparing the replacement results with the DIET-CODE’s results, the performance drop will reflect the contribute of token weights for LEANCODE’ performance due to the same removal algorithm. Similarly, the difference of results of LEANCODE and the replacement method can show the contribute of LEANCODE’s removal algorithm since both share the same token weights.

Table 10 shows the replacement results. Compared to DIETCODE’results in Tables 2 and 3, we

can see that the performance drops of DIETCODE reach up to 37.1% for code search and 19.05% for code summarization. The performance drop conversely demonstrates that more accurate token weights significantly enhanced LEANCODE’s performance. On the other side, compared to LEAN-CODE’ results, we find that the maximal performance drops can be 3.7% for code search and 9.9% for code summarization. It demonstrates that the token-level removal algorithm (Algorithm 1) is much better than that of DIETCODE and significantly contributes to LEANCODE’s performance.