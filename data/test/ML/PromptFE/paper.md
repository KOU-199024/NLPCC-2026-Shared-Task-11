# PromptFE: Automated Feature Engineering by Prompting

Yufeng Zou† Jean Utke§ Diego Klabjan† Han Liu† †Northwestern University

§Allstate

yufeng.zou@u.northwestern.edu

jutke@allstate.com

{d-klabjan, hanliu}@northwestern.edu

# Abstract

Automated feature engineering (AutoFE) liberates data scientists from the burden of manual feature construction. The semantic information of datasets contains rich context information for feature engineering but has been underutilized in many existing AutoFE works. We present PromptFE, a novel AutoFE framework that leverages large language models (LLMs) to automatically construct features in a compact string format and generate semantic explanations based on dataset descriptions. By learning the performance of constructed features in context, the LLM iteratively improves feature construction. We demonstrate through experiments on real-world datasets the superior performance of PromptFE over state-ofthe-art AutoFE methods. We verify the impact of dataset semantic information and provide comprehensive study on the LLM-based feature construction process.

# 1 Introduction

Tabular data, a form of structured data comprising instances and attributes, have extensive use across a broad range of domains including credit assessment, market prediction, and quality control. Traditional machine learning models, especially tree-based models (Breiman, 2001; Ke et al., 2017), have strong performance on tabular datasets of small and medium sizes (Grinsztajn et al., 2022) and good interpretability. Feature engineering refers to the construction of new features from feature attributes of a dataset to enhance the performance of downstream models, which is crucial for traditional ML models as new features extract useful information for target prediction by capturing complex non-linear relationships. Feature engineering by hand demands domain expertise to relieve significant human labor.

Automated feature engineering (AutoFE) employs high-level algorithms and models to automate the feature engineering process and attain performance comparable to domain experts. Prior approaches like (Zhu et al., 2022a,b; Zhang et al., 2023) construct and evaluate an enormous number of features in a trial-and-error manner. While some methods learn to optimize the utility of features during the FE process, they do not utilize domain knowledge to guide feature search. The need to search features from scratch for new datasets and downstream models hampers their efficacy and efficiency. Furthermore, these methods do not offer explanation of the engineered features, undermining interpretability.

Text descriptions of tabular datasets provide rich context for feature engineering. Domain experts consult attribute descriptions to select relevant feature attributes and compute new features useful for target prediction. For example, the square footage of a house times the average housing price per square foot in the neighborhood may be a good predictor of the market value of the house. Pretrained on large volumes of data, large language models (LLMs) (Radford et al., 2019; Brown et al., 2020; OpenAI, 2023; Touvron et al., 2023a,b) handle general language processing tasks and encapsulate extensive domain knowledge. Under proper instructions, an LLM can process dataset semantic information and utilize its knowledge to automatically construct features in a manner similar to domain experts. The work by Hollmann et al. (2023) demonstrates the potential of such research direction but is not sufficiently effective in feature search. Similarly, the work by Nam et al. (2024) suffers from large search space. The works by Han et al. (2024) and Zhang et al. (2024b) do not involve feature learning and improvement.

We present AutoFE by Prompting (PromptFE), a novel AutoFE framework that leverages LLMs for effective, efficient, and interpretable feature engineering, as illustrated in Figure 1. With dataset descriptions and example features in canonical Reverse Polish Notation (cRPN), we prompt the LLM to construct new features. After evaluation, we update the prompt with top-performing features with the evaluation scores and instruct the LLM to construct further features. Iteratively, the LLM explores the feature space and improves solutions by learning good examples in context. The dataset semantic information not only guides feature search, but helps the LLM understand the patterns in example features. Applying domain knowledge, the LLM generates semantically meaningful features and explains their usefulness. Experiments on realworld datasets demonstrate that PromptFE yields over 5% mean performance gain for three downstream models and significantly outperforms stateof-the-art baselines. Furthermore, we show in the ablation study the effects of dataset semantic context and proposed feature canonicalization scheme. We also comprehensively study the behavior of the LLM-based feature construction process.

![](images/d30a64c463ee77ab65bab60fd47004759cde6a6297c237f07b71704ae46f355a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["performance evaluation"] --> B["scores"]
    C["constructed features"] --> B["scores"]
    C --> D["prompt"]
    D --> E["LLM"]
    E --> C
    F["select a set of constructed features"] --> C
    G["dataset descriptions top feature-score pairs"] --> E
```
</details>

Figure 1: Overview of PromptFE: (1) instructing the LLM to construct new features by providing dataset descriptions and example features; (2) evaluating the constructed features; (3) updating the prompt with top-performing features and scores; (4) selecting a set of constructed features to add to the dataset.

Our main contributions are: (1) we introduce a novel LLM-based AutoFE framework utilizing dataset semantic information for automated feature construction, which is the first method capable of generating features in the cRPN format while providing semantic explanations; (2) we benchmark the performance of our approach against state-ofthe-art baselines using both GPT-3.5 and GPT-4; (3) we examine the impact of semantic context and study the behavior of the LLM-based feature construction process, providing a comprehensive view of our approach.

# 2 Related Work

Large Language Models. LLMs are large-scale general-purpose neural networks pretrained on vast corpora of text data, typically built with transformer-based architectures (Vaswani et al., 2017). Generative LLMs, such as the GPT family (Radford et al., 2019; Brown et al., 2020; OpenAI, 2023) and the LLaMA family (Touvron et al., 2023a,b), are pretrained to successively generate the next token given the text input and can be finetuned using reinforcement learning from human feedback (Ziegler et al., 2019; Ouyang et al., 2022). By this means, they acquire the syntactic and semantic knowledge of natural languages and achieve state-of-the-art performance on various tasks including text generation, summarization, and question answering. Prompting techniques (Liu et al., 2023) have been developed to adapt LLMs to downstream tasks without modifying model weights. Few-shot learning (Brown et al., 2020) includes examples in the prompt for the language model to learn in context. Leveraging such capability, an LLM may function as a problem solver (Yang et al., 2024) that iteratively improves candidate solutions according to the task description and performance feedback. Chain-of-thought (Wei et al., 2022; Kojima et al., 2022) strengthens reasoning performance of LLMs through the elicitation of intermediate reasoning steps.

Automated Feature Engineering. AutoFE complements the input dataset with engineered features to enhance the performance of downstream models. Traditional AutoFE approaches include expansion-reduction (Kanter and Veeramachaneni, 2015; Horn et al., 2020; Zhang et al., 2023), evolutionary algorithms (Smith and Bull, 2005; Zhu et al., 2022a), and reinforcement learning (Khurana et al., 2018; Li et al., 2023; Wang et al., 2023). DIFER (Zhu et al., 2022b) utilizes encoderdecoder neural networks to learn the utility of features and optimize features in the embedding space. OpenFE (Zhang et al., 2023) develops a feature boost algorithm to speedup feature evaluation. Nonetheless, these traditional approaches do not incorporate the semantic information of datasets, which hampers the efficacy and interpretability of engineered features.

AutoFE with Domain Knowledge. The benefits of incorporating domain knowledge in AutoFE include: (1) improving effectiveness; and (2) reducing the cost of feature search, especially the feature evaluation overhead. One direction in prior works is to learn transferrable knowledge. LFE (Nargesian et al., 2017) represents features with quantile sketches transferable across datasets and inputs them to a feature transformation recommendation model. FETCH (Li et al., 2023) is an RLbased AutoFE framework that takes tabular data as the state and is generalizable to new data. E-AFE (Wang et al., 2023) pretrains a feature evaluator to help efficiently train the RL-based AutoFE model. The other direction is to leverage the semantic information of datasets. KAFE (Galhotra et al., 2019) employs knowledge graphs to identify semantically informative features relevant to the prediction task. CAAFE (Hollmann et al., 2023) manipulates datasets using the code generated from an LLM based on dataset descriptions. FeatLLM (Han et al., 2024) generates first-order rules for classification tasks. ELF-Gym (Zhang et al., 2024b) generates first feature descriptions and then feature generation code. Neither approach involves feature learning and improvement. OCTree (Nam et al., 2024) relies on external decision tree algorithms to represent features and suffers from a large search space. Differently, we adopt a compact form of feature representation in cRPN with pre-defined transformation operators. Our approach reduces the search space and helps the LLM learn the patterns of useful features, leading to stronger and more robust performance.

# 3 Notations

We denote a tabular dataset with n instances and $d + 1$ columns as $\textit { D } = \{ { \mathbb X } , { \pmb y } \}$ , where $\mathbb { X } =$ $\{ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { d } \}$ is the set of raw features with ${ \pmb x } _ { i } \in$ $\mathbb { R } ^ { n }$ for $i = 1 , \ldots , d$ and $\ b { y } \in \mathbb { R } ^ { n }$ is the target. We construct a new feature $\tilde { \pmb { x } } = t ( \pmb { x } _ { j _ { 1 } } , \ldots , \pmb { x } _ { j _ { o } } )$ （20 by transforming existing features $\pmb { x } _ { j _ { 1 } } , \ldots , \pmb { x } _ { j _ { c } }$ via some operator $t \in \mathbb { R } ^ { n } \times \ldots \times \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ of arity o. Given a set of transformation operators $\mathbb { T } .$ , we define the feature space $\mathbb { X } _ { \mathbb { T } }$ recursively as: for any $\tilde { \textbf { \textit { x } } } \in \ \mathbb { X } _ { \mathbb { T } } .$ , either $\tilde { \textbf { \textit { x } } } \in \ \mathbb { X } ;$ or $\exists t \ \in \ \mathbb { T }$ , s.t., $\tilde { \pmb { x } } = t ( \tilde { \pmb { x } } _ { j _ { 1 } } , \ldots , \tilde { \pmb { x } } _ { j _ { o } } )$ , where $\pmb { \tilde { x } } _ { j _ { 1 } } , \dots , \pmb { \tilde { x } } _ { j _ { o } } \in \mathbb { X } _ { \mathbb { T } }$ . To measure feature complexity, we compute the order of a feature $\tilde { \pmb { x } } \in \mathbb { X } _ { \mathbb { T } }$ as:

$$
\alpha (\tilde {\boldsymbol {x}}) = \left\{ \begin{array}{l l} 0 & \text { if } \tilde {\boldsymbol {x}} \in \mathbb {X}, \\ 1 + \max _ {j} \alpha (\tilde {\boldsymbol {x}} _ {j}) & \text { if } \tilde {\boldsymbol {x}} = t (\tilde {\boldsymbol {x}} _ {j _ {1}}, \dots , \tilde {\boldsymbol {x}} _ {j _ {o}}) \\ & \text { for } t \in \mathbb {T}. \end{array} \right. \tag {1}
$$

The constrained feature space with the order upper bounded by u is denoted as $\mathbb { X } _ { \mathbb { T } } ^ { ( u ) } = \{ \tilde { \pmb { x } } \in \mathbb { X } _ { \mathbb { T } } \ |$ $\alpha ( \tilde { \pmb { x } } ) \le u \}$ .

We denote the performance of a downstream machine learning model algorithm M on the dataset as $\mathcal { E } _ { M } ( \mathbb { X } , \pmb { y } )$ . The objective of AutoFE is to augment the dataset with a set of constructed features $\tilde { \mathbb { X } } ^ { * }$ to optimize the model performance, specifically:

![](images/a0cb155506a280b4ccd6339f70862209fb7b5148f06c0547c0f1c1540d97997f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["col-2"] --> B["+"]
    B --> C["col-1"]
    B --> D["col-0"]
    E["col-2, col-1, col-0, +,*"]
    F["canonilize"] --> G["+"]
    G --> H["col-0"]
    G --> I["col-1"]
    J["col-0, col-1, +, col-2,*"]
```
</details>

Figure 2: We obtain canonical RPN (cRPN) by reordering the nodes of a feature expression tree.

$$
\tilde {\mathbb {X}} ^ {*} = \underset {\tilde {\mathbb {X}} \subset \mathbb {X} _ {\mathbb {T}} ^ {(u)}} {\arg \max} \mathcal {E} _ {M} (\mathbb {X} \cup \tilde {\mathbb {X}}, \boldsymbol {y}). \tag {2}
$$

# 4 Methodology

In this section, we present PromptFE, a novel AutoFE framework leveraging the power of LLMs, particularly, the GPT models (Radford et al., 2019; Brown et al., 2020; OpenAI, 2023). The high-level idea is to provide the LLM with descriptive information of the dataset in the prompt and guide it to search for effective features using examples.

We represent features in a compact form in our prompt. A feature $\tilde { \textbf { \textit { x } } } \in \ \mathbb { X } _ { \mathbb { T } }$ is expressible as a tree, where the leaf nodes are raw features and the internal nodes are operators. However, the expression trees of features containing commutative operators (like addition and multiplication) are not unique since the child nodes of these operators are unordered. We introduce a canonicalization scheme: arranging operator nodes before feature nodes for left skewness and lexicographically sorting the nodes within each group. We then serialize the canonical expression tree into the postorder depth-first traversal string, i.e., cRPN, ensuring the one-to-one mapping between features and string representations. We denote the feature corresponding to a cRPN string f as $\tilde { \boldsymbol { \mathbf { x } } } _ { f }$ and the set of features corresponding to a set of cRPN strings F as $\tilde { \mathbb { X } } _ { \mathbb { F } }$ We make further discussions in Appendix A.

Our prompt contains: (1) a meta description of the dataset; (2) an indexed list of the dataset attributes, with attribute types, value ranges, and descriptions; (3) lists of transformation operators with descriptions, grouped by the arity; (4) a ranked list of example features with performance evaluation scores; (5) feedback of constructed features from the previous iteration; and (6) an output template of new features and explanations. Figure 3 outlines the structure of our prompt. The descriptions of the dataset and attributes provide contextual information for the LLM to understand the data and

(1) Dataset description:

This dataset contains information on default payments, demographic factors, credit data, and history of payment of credit card clients ...

(2) Dataset contains the following columns:

col-0 (int) [10000, 800000]: LIMIT\_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit

col-1 (category) {1, 2}: SEX: Gender (1=male, 2=female)

col-23 (category) {0, 1}: default.payment.next.month: Default payment (1=yes, 0=no)

(3) We have the following unary operators:

log: element-wise logarithm of the absolute value . . .

We have the following binary operators:

+: element-wise addition of two columns

Feature strings are reverse Polish notation (RPN) expressions that operate on the columns of our dataset. Each feature string constructs an extra column that is useful for the downstream model Random Forests to predict the target col-23. The model will be trained on the dataset with the constructed columns and evaluated on a holdout set. The best columns will be selected.

(4) Below are feature strings arranged in ascending order based on their performance scores. Higher scores are better.

Feature

col-17,col-21,\*,col-20,+,sqrt\_abs

Score

0.0011

Feature

col-4,col-6,\*,col-12,col-16,-,sqrt\_abs,\*

Score

0.0014

(5) Previous feature:

col-4,col-5,col-6,+,col-12,col-16,-,\*

Error: invalid RPN expression

Give me a new feature string that is different from all strings above and has a higher score. Use no more than five operators. Make sure all columns and operators exist and do not include the target column. Follow the syntax of RPN.

(6) Output format:

Feature

(Feature description)

Usefulness

(Explanation why this adds useful real world knowledge to predict the target

col-23 according to dataset description)

Figure 3: Prompt template. Sections containing dataset information are marked in blue. The ranked list of feature examples and scores is marked in orange. The feedback message is marked in gray.

apply domain knowledge. The value ranges of attributes are useful for selecting appropriate feature transformations, e.g., min-max normalization when the scale is too large. We include the descriptions of transformation operators as they help the LLM parse example features in the RPN syntax and construct syntactically valid feature strings. The output template not only structures the output but instructs the LLM to reason about the usefulness of the constructed features and offer semantic explanations, utilizing the chain-of-thought technique (Wei et al., 2022; Kojima et al., 2022). We additionally add a constraint instruction to use no more than a certain number of operators, which reduces the search space and regularizes the solutions. Figure 4 shows an example LLM output. The prompt may further include statistics like mean, standard deviation, and skewness of dataset attributes.

We initialize the prompt with k random features from the constrained feature space $\tilde { { \pmb x } } _ { 1 } , \dotsc , \tilde { { \pmb x } } _ { k } \in$ $\mathbb { X } _ { \mathbb { T } } ^ { ( 2 ) }$ represented in cRPN for demonstration, where the feature attributes are sampled per the softmax Feature

col-11,col-11,col-12,-,log,\*

This feature calculates the log of the difference between the September bill statement (col-11) and the August bill statement (col-12), then multiplies it by the September bill amount.

Usefulness

This feature captures the change in the bill amount from August to September in a logarithmic scale, which can effectively highlight significant changes in spending patterns. Large fluctuations in credit card bills could be indicative of financial distress, which may impact the likelihood of defaulting on payments, while the logarithmic transformation allows for handling potential skewness in the data distribution. By incorporating this feature, the model can better understand how temporal changes in spending behavior relate to the probability of default, providing richer contextual information beyond static features.

Figure 4: The LLM constructs a new feature in RPN and explains its usefulness from the semantic perspective.

probabilities of feature importance by fitting the downstream model on the training data. This lets the LLM start the search from a small feature space where it is easier to identify the basic patterns of promising features. Optionally, we can import external example features. We prompt the LLM to construct a fixed number of m new feature in an iteration. For each constructed feature string $f ,$ we first try to obtain the cRPN expression $f ^ { c }$ to check whether $f ^ { c }$ is syntactically valid and not a duplicate of candidate features. If both criteria are met, we evaluate the performance score of the single feature $s = \mathcal { E } _ { M } ( \mathbb { X } \cup \{ \tilde { \pmb { x } } _ { f ^ { c } } \} , \pmb { y } )$ through cross validation on the training data and add $\langle f ^ { c } , s \rangle$ to the candidate set $\mathbb { F } _ { c a n d } .$ . When $f ^ { c }$ is among the top-k candidate features in terms of the score s, we update prompt examples with the top-k pairs $\langle f ^ { \prime } , s ^ { \prime } \rangle \in \mathbb { F } _ { c a n d }$ ranked in the ascending order, taking score increment $s ^ { \prime } - \mathcal { E } _ { M } ( \mathbb { X } , \pmb { y } )$ over the baseline. We also provide the feedback of previous feature constructions with evaluation scores or error messages for performance improvement and error correction. We then instruct the LLM to construct additional features using the updated prompt. To select candidate features, we successively add candidate features to the dataset from the best to the worst and determine the optimal number of features to add based on validation performance, which is evaluated over sets of candidate features and thus takes feature interactions into account.

Algorithm 1 summarizes our methodology. The size of the prompt scales linearly with the number of features in the dataset d and the number of example features k and stays roughly constant across feature construction iterations. Thus, the cost of an LLM generation step in line 3 is almost constant. The computation cost of feature evaluation in line 7 is also constant, preserving the efficiency and scalability of our algorithm. The evaluations in line 7 and at lines 13-16 are parallelizable.

Algorithm 1: AutoFE by Prompting   
Input :Dataset $D = \langle \mathbb{X},\boldsymbol {y}\rangle$ , downstream model $M$ , large language model LLM, the number of features $m$ to construct in a generation, the number of example features $k$ in the prompt, and optionally an external set of features with evaluation scores $\mathbb{F}_{ext}$ Output:A set of engineered features $\mathbb{F}$ 1 Initialize prompt $P$ with dataset descriptions and example features; $\mathbb{F}_{cand}\gets \mathbb{F}_{ext}$ if $\mathbb{F}_{ext}$ is available, otherwise $\mathbb{F}_{cand}\gets \emptyset$ $\mathbb{F}_{set}\gets \emptyset$ 2 repeat   
3 $\mathbb{F}_{LLM} = \{f_1,\dots ,f_m\} \leftarrow LLM(P)$ 4 for each $f\in \mathbb{F}_{LLM}$ do   
5 $f^{c}\gets$ Canonicalize $f$ 6 if $f^c$ is valid and $f^c\notin \mathbb{F}_{cand}$ then   
7 Evaluate cross validation performance $s\gets \mathcal{E}_M(\mathbb{X}\cup \{\pmb{x}_{f^c}\} ,\pmb {y})$ 8 $\mathbb{F}_{cand}\gets \mathbb{F}_{cand}\cup \{\langle f^c,s\rangle \}$ 9 end   
10 end   
11 Update $P$ such that $P$ contains the top- $k\langle f^{\prime},s^{\prime}\rangle \in \mathbb{F}_{cand}$ as ordered by $s^\prime$ and the feedback for $\mathbb{F}_{LLM}$ 12 if feature selection is required then   
13 for $n\gets 1$ to $|\mathbb{F}_{cand}|$ do   
14 $\mathbb{F}_n\gets$ The top- $n$ features in $\mathbb{F}_{cand}$ as ordered by $s$ 15 Evaluate validation performance $s_n\gets \mathcal{E}_M(\mathbb{X}\cup \tilde{\mathbb{X}}_{\mathbb{F}_n},\boldsymbol {y})$ 16 end   
17 $\mathbb{F}_{set}\gets \mathbb{F}_{set}\cup \{\langle \mathbb{F}_n^*,s_n^*\rangle \}$ , with $n^*\gets \arg \max_n s_n$ 18 end   
19 until stopping criteria are met   
20 return F in $\mathbb{F}_{set}$ with the maximum validation score

In our algorithm, the LLM is instructed to perform as a problem solver (Yang et al., 2024). Analogous to evolutionary algorithms that generate new solutions through crossover and mutations on high-fitness candidates (Smith and Bull, 2005; Zhu et al., 2022a; Morris et al., 2024), we provide topperforming features in the prompt. By learning examples and scores in-context (Brown et al., 2020), the LLM recognizes the patterns of promising features and generates new features that are likely to be useful. It can make analogies to, modify, or combine example features in the prompt (Appendix F.2). Early in the search, we expect greater exploration due to the diversity of initial examples. As iterations progress, the LLM exploits promising feature spaces more, gradually refining the search until convergence. The dataset semantic information enhances the effectiveness of feature search through the guidance as a prior. The LLM’s temperature can be adjusted to balance exploration and exploitation, with higher temperatures encouraging more diverse solutions and lower temperatures favoring incremental changes to example features.

We adopt the same set of transformation operators T as those in (Zhu et al., 2022b), including:

• Unary transformations: logarithm, reciprocal, square root, and min-max normalization;   
• Binary transformations: addition, subtraction, multiplication, division, and modulo.

Table 1: Statistics of datasets used in experiments. The datasets cover different domains and tasks, and vary in sizes. 

<table><tr><td>Name</td><td>Task</td><td># Samples</td><td># Features</td></tr><tr><td>Airfoil (AF)</td><td>Regression</td><td>1,503</td><td>5</td></tr><tr><td>Boston Housing (BH)</td><td>Regression</td><td>506</td><td>13</td></tr><tr><td>Bikeshare (BS)</td><td>Regression</td><td>731</td><td>10</td></tr><tr><td>Wine Quality Red (WQR)</td><td>Regression</td><td>1,599</td><td>11</td></tr><tr><td>AIDS Clinical Trials (ACT)</td><td>Classification</td><td>2,139</td><td>23</td></tr><tr><td>Credit Default (CD)</td><td>Classification</td><td>30,000</td><td>23</td></tr><tr><td>German Credit (GC)</td><td>Classification</td><td>1,000</td><td>20</td></tr></table>

In min-max normalization, we collect statistics from the training data. Other transformations require only the information of a single instance. Hence, all transformations can be performed on an individual test instance without leaking other instances’ information. Data leakage (Overman et al., 2024) is an issue that has not been properly addressed in many existing AutoFE works.

# 5 Experiments

# 5.1 Experimental Setup

We benchmark performance on public real-world datasets from Kaggle and UCI repositories covering different domains (Table 1). The descriptive information of datasets and attributes is retrieved from the sources without further processing. The downstream models we evaluate include linear models (LASSO for regression tasks and logistic regression for classification tasks), Random Forests (Breiman, 2001), and LightGBM (Ke et al., 2017). For linear models, we target-encode categorical features and min-max scale all features. We tune downstream model algorithm parameters by randomized search prior to and post AutoFE, because the model may need reconfiguration to accommodate the added features. Data are randomly split into training (64%), validation (16%), and test (20%) sets. We evaluate regression performance with 1 (relative absolute error)1 and classification performance with accuracy. A higher evaluation score indicates better performance.

We compare PromptFE with the following stateof-the-art AutoFE methods: (1) DIFER (Zhu et al., 2022b): A neural network-based method that optimizes features in the embedding space utilizing LSTMs to encode and decode features; (2) OpenFE (Zhang et al., 2023): An expansionreduction method that evaluates features up to a certain order using a feature boost algorithm; (3) CAAFE (Hollmann et al., 2023): An LLM-based method that produces Python code to manipulate datasets stored in Pandas data frames; (4) OC-Tree (Nam et al., 2024): An LLM-based method that produces rules to manipulate datasets and encodes features using decision tree algorithms.

We employ gpt-3.5-turbo-01252 and $\mathtt { g p t - } 4 \mathtt { - } \theta 6 1 3 ^ { 2 }$ as the LLMs. For PromptFE, we include $k = 1 0$ example features in the prompt and set the temperature of LLMs to 1 based on validation performance. We instruct the LLM to construct m = 1 feature in each generation step for the best control of feature generation. We perform feature selection each time 10 new candidate features are constructed and terminate the algorithm once we have 200 candidate features. Parameters of the baseline methods are initialized per the corresponding papers. We make five repeated runs unless stated otherwise.

# 5.2 Performance Comparison

Table 2 compares the overall performance between PromptFE and the baseline methods3. Full results are presented in Appendix D.5. PromptFE attains the best mean performance score and the highest mean rank for all three downstream models, yielding over 5% mean performance gain and over 15% gain for linear models. We observe the greatest gain for linear models because unlike Random Forests and LightGBM, they cannot learn nonlinear relationships themselves. The performance margin between PromptFE and baselines other than DIFER is statistically significant at $p < 0 . 0 1$ by the Friedman-Nemenyi test. PromptFE consistently outperforms CAAFE and OCTree, showing the robustness of PromptFE that reduces the search space with pre-defined operators and represents features in compact cRPN. Post-AutoFE parameter tuning brings the greatest performance improvement to DIFER, as it adds the most features to datasets (Appendix D.9). Compared with DIFER evaluating over 2, 000 candidate features during feature search, PromptFE evaluates only 200 candidate features (Appendix D.10). The higher efficiency of PromptFE is brought by the construction of semantically meaningful and effective features with the guidance of dataset semantic information.

We note that for PromptFE, using GPT-4 yields better performance for linear models but slightly

worse performance for Random Forests than GPT-3.5. We speculate this is because the stronger incontext learning capability of GPT-4 increases the tendency of overfitting example features. One way to address this is to include more example features in the prompt to fully leverage GPT-4’s enhanced in-context learning capability (Appendix D.8).

# 5.3 Effect of Semantic Context

We compare with the blinded version without dataset semantic information (Appendix C.2). From Table 3, PromptFE outperforms the blinded version for all downstream models with statistical significance. The performance difference is more pronounced for Random Forests and LightGBM, likely because the inclusion of non-semantically meaningful features by the blinded version consumes model capacity and causes greater overfitting to the training data. GPT-4 constructs features more efficiently using fewer generations on average than GPT-3.5 due to stronger capabilities. The incorporation of dataset semantic context improves the feature construction efficiency of GPT-3.5 but reduces that of GPT-4, as it guides to more focused feature spaces that increase the chance of duplication with candidate features.

# 5.4 Effect of Feature Expression Canonicalization

We compare with the ablated version without canonicalization of feature expressions. From Table 3, PromptFE outperforms the ablated version for all downstream models. Without canonicalization, we observe a slight decrease in the number of LLM generations. Since a feature can be represented in different expressions, the chance of duplication with the expressions of candidate features during feature search is reduced. However, the effectiveness of the features constructed by the LLM degrades in this setting due to greater difficulty in learning optimal feature patterns.

# 5.5 Feature Attribute Selection

Figure 6 shows the distributions of feature attributes in the constructed features for linear models with GPT-4. Without semantic context, the LLM tends to prioritize earlier feature attributes in the dataset while paying less attention to later ones. In comparison, PromptFE is informed by the semantic context. Specifically, Attribute 19 CD4 at baseline in ACT and Attribute 10 alcohol in WQR, which contain critical information for predicting the respective targets censoring indicator and quality, are consistently among the most frequent ones. This illustrates how the LLM leverages dataset semantic information to construct semantically meaningful and effective features in PromptFE.

Table 2: Comparison of overall performance. For each compared method, the left and right columns show the performance without and with post AutoFE parameter tuning of downstream model algorithms, respectively. The best results are boldfaced, and the second best results are underlined. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Raw</td><td rowspan="2" colspan="2">DIFER</td><td rowspan="2" colspan="2">OpenFE</td><td colspan="4">CAAFE</td><td colspan="2">OCTree</td><td colspan="4">PromptFE (ours)</td></tr><tr><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td><td colspan="2">GPT-4</td><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td></tr><tr><td>Linear</td><td>0.5636</td><td>0.6248</td><td>0.6369</td><td>0.5871</td><td>0.5866</td><td>0.5946</td><td>0.5941</td><td>0.5945</td><td>0.5946</td><td>0.6038</td><td>0.6044</td><td>0.6485</td><td>0.6487</td><td>0.6532</td><td>0.6526</td></tr><tr><td>Model</td><td>14.00</td><td>9.17</td><td>5.83</td><td>10.58</td><td>9.92</td><td>10.00</td><td>9.50</td><td>10.50</td><td>9.83</td><td>8.25</td><td>7.33</td><td>5.08</td><td>3.50</td><td>3.33</td><td>3.17</td></tr><tr><td>Random</td><td>0.7252</td><td>0.7400</td><td>0.7411</td><td>0.7380</td><td>0.7376</td><td>0.7387</td><td>0.7378</td><td>0.7357</td><td>0.7352</td><td>0.7348</td><td>0.7346</td><td>0.7408</td><td>0.7412</td><td>0.7392</td><td>0.7393</td></tr><tr><td>Forests</td><td>12.71</td><td>8.29</td><td>5.86</td><td>8.07</td><td>8.64</td><td>6.29</td><td>7.50</td><td>9.00</td><td>10.79</td><td>9.14</td><td>10.79</td><td>4.43</td><td>4.57</td><td>6.29</td><td>7.64</td></tr><tr><td>Light-</td><td>0.7364</td><td>0.7504</td><td>0.7531</td><td>0.7454</td><td>0.7476</td><td>0.7457</td><td>0.7461</td><td>0.7405</td><td>0.7457</td><td>0.7409</td><td>0.7403</td><td>0.7522</td><td>0.7558</td><td>0.7542</td><td>0.7538</td></tr><tr><td>GBM</td><td>10.43</td><td>8.86</td><td>6.29</td><td>9.50</td><td>9.14</td><td>9.07</td><td>8.14</td><td>9.00</td><td>8.50</td><td>11.21</td><td>12.21</td><td>5.71</td><td>3.57</td><td>3.57</td><td>4.79</td></tr><tr><td>Mean</td><td>0.6806</td><td>0.7091</td><td>0.7140</td><td>0.6953</td><td>0.6958</td><td>0.6979</td><td>0.6976</td><td>0.6950</td><td>0.6967</td><td>0.6976</td><td>0.6975</td><td>0.7171</td><td>0.7185</td><td>0.7187</td><td>0.7183</td></tr><tr><td>Mean Rank</td><td>12.30</td><td>8.75</td><td>6.00</td><td>9.33</td><td>9.20</td><td>8.38</td><td>8.33</td><td>9.45</td><td>9.70</td><td>9.60</td><td>10.25</td><td>5.08</td><td>3.90</td><td>4.45</td><td>5.30</td></tr></table>

Table 3: Comparison of PromptFE with ablated versions. For each compared version, the left and middle columns show the performance without and with post AutoFE parameter tuning of downstream model algorithms, respectively, and the right column shows the number of LLM generations. Statistical significance of performance difference by the Friedman-Nemenyi test is indicated with \* for p < 0.05 and \*\* for p < 0.01. 

<table><tr><td></td><td>Model</td><td colspan="3">w/o Semantic Context</td><td colspan="3">w/o Canonicalization</td><td colspan="3">PromptFE</td></tr><tr><td rowspan="4">GPT-3.5</td><td>Linear Model</td><td>0.6411</td><td>0.6433</td><td>443.4</td><td>0.6471</td><td>0.6486</td><td>349.1</td><td>0.6485</td><td>0.6487</td><td>356.7</td></tr><tr><td>Random Forests</td><td>0.7326**</td><td>0.7328**</td><td>472.5</td><td>0.7372</td><td>0.7373</td><td>358.0</td><td>0.7408</td><td>0.7412</td><td>370.4</td></tr><tr><td>LightGBM</td><td>0.7479*</td><td>0.7494</td><td>490.0</td><td>0.7485</td><td>0.7490</td><td>348.9</td><td>0.7522</td><td>0.7558</td><td>360.2</td></tr><tr><td>Mean</td><td>0.7105**</td><td>0.7118**</td><td>469.9</td><td>0.7141</td><td>0.7148</td><td>352.2</td><td>0.7171</td><td>0.7185</td><td>362.7</td></tr><tr><td rowspan="4">GPT-4</td><td>Linear Model</td><td>0.6437</td><td>0.6461</td><td>253.9</td><td>0.6462</td><td>0.6463</td><td>323.6</td><td>0.6532</td><td>0.6526</td><td>326.3</td></tr><tr><td>Random Forests</td><td>0.7285*</td><td>0.7288*</td><td>262.9</td><td>0.7366</td><td>0.7366</td><td>315.7</td><td>0.7392</td><td>0.7393</td><td>333.0</td></tr><tr><td>LightGBM</td><td>0.7420**</td><td>0.7437</td><td>250.7</td><td>0.7461*</td><td>0.7480</td><td>328.5</td><td>0.7542</td><td>0.7538</td><td>335.7</td></tr><tr><td>Mean</td><td>0.7078**</td><td>0.7092**</td><td>255.9</td><td>0.7128**</td><td>0.7135*</td><td>322.5</td><td>0.7187</td><td>0.7183</td><td>331.9</td></tr></table>

# 5.6 Performance Analysis

We study the performance for linear models with GPT-3.5 from ten repeated runs. Figures 7-10 display the slope and p-value from one-tailed t-tests in OLS regressions, with the shaded area showing one standard deviation above and below the mean.

Feature Learning. We examine the cross validation score of candidate features across iterations. Figure 7 shows a significantly upward trend in the score, with most constructed features improving the performance. This demonstrates that PromptFE effectively improves the quality of constructed features through in-context learning of top-performing examples during feature search.

Feature Complexity. We examine the order of candidate features across iterations. Figure 8 shows that the feature order increases rapidly in early iterations and stabilizes over time. PromptFE effectively constructs complex features within promising feature spaces. Moreover, our constraint instruction offers regularization that prevents the construction of overly complex features.

Feature Divergence. We analyze the divergence of a new candidate feature from previous ones during feature search. We compute the edit distance between canonical feature expression trees using the algorithm by Zhang and Shasha (1989) and normalize the distance by the total number of nodes in both trees. Figure 9 shows the mean normalized tree edit distance between the current candidate feature and the previous five features across iterations. The observed downward trend indicates that feature search converges over iterations.

Feature Construction Efficiency. We examine the number of LLM generations needed to construct new candidate features across iterations. Figure 10 exhibits a slightly upward trend in the number of LLM generations, due to increasing difficulty of constructing non-duplicate features and higher likelihood of producing syntactical errors as features become more complex. Since the increase is nonsignificant, PromptFE remains scalable to a large number of iterations.

# 5.7 Hyperparameter Effect

Number of Examples in Prompt. Table 4 reports the maximum validation score across iterations along with the number of LLM generations by varying the number of example features provided in the prompt. We observe that the best performance is attained with 10 examples. Additionally, feature construction efficiency improves as the number of examples increases, as more examples can help the LLM reduce errors and generate more diverse features. Nonetheless, too many examples hinder the in-context learning of optimal feature patterns, as shown by the performance decline. The performance difference with 10 examples and with 1 example is statistically significant at $p < 0 . 0 5$ by one-tailed paired t-tests.

![](images/8b01d927bfd8ae8099ac2b4eb4e153b22f32ac532b39ea8747555e5307095e96.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.2        |
| 1                       | 0.3        |
| 2                       | 0.5        |
| 3                       | 0.2        |
| 4                       | 0.1        |
| 5                       | 0.3        |
| 6                       | 0.1        |
| 7                       | 0.05       |
| 8                       | 0.1        |
| 9                       | 0.05       |
| 10                      | 0.1        |
| 11                      | 0.05       |
| 12                      | 0.05       |
| 13                      | 0.05       |
| 14                      | 0.05       |
| 15                      | 0.05       |
| 16                      | 0.05       |
| 17                      | 0.05       |
| 18                      | 0.05       |
| 19                      | 0.6        |
| 20                      | 0.4        |
| 21                      | 0.2        |
| 22                      | 0.1        |
</details>

(a) ACT

![](images/cbed2d1ccb481c9015e63da38ab8abe406536d95d00985f8ed4c7f5e20fae9f1.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.2        |
| 1                       | 0.6        |
| 2                       | 0.5        |
| 3                       | 0.4        |
| 4                       | 0.3        |
| 5                       | 0.6        |
| 6                       | 0.5        |
| 7                       | 0.2        |
| 8                       | 0.1        |
| 9                       | 0.1        |
| 10                      | 0.1        |
| 11                      | 0.1        |
| 12                      | 0.1        |
| 13                      | 0.1        |
| 14                      | 0.1        |
| 15                      | 0.0        |
| 16                      | 0.0        |
| 17                      | 0.0        |
| 18                      | 0.0        |
| 19                      | 0.0        |
| 20                      | 0.0        |
</details>

(b) ACT, blinded

![](images/9949bfb876bc88e431c303a37a5bc6c48915e4574f35186f8bda8d513627f6a6.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.5        |
| 1                       | 0.6        |
| 2                       | 0.4        |
| 3                       | 0.4        |
| 4                       | 0.5        |
| 5                       | 0.3        |
| 6                       | 0.2        |
| 7                       | 0.1        |
| 8                       | 0.2        |
| 9                       | 0.2        |
| 10                      | 0.6        |
</details>

(c) WQR

![](images/dc738774d6b971f082298bb6a8ce6f268e58d95e79db7235dcd0276cdb23689c.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.4        |
| 1                       | 0.7        |
| 2                       | 0.6        |
| 3                       | 0.45       |
| 4                       | 0.5        |
| 5                       | 0.5        |
| 6                       | 0.2        |
| 7                       | 0.05       |
| 8                       | 0.05       |
| 9                       | 0.05       |
| 10                      | 0.05       |
</details>

(d) WQR, blinded   
Figure 6: Distributions of feature attributes in the constructed features for linear models with GPT-4.

![](images/d224666fd53f8c7ff88d6c8041fa05d34b871bc96d9aace91e6e9c2017562e88.jpg)

<details>
<summary>line</summary>

| Iterations | Score  |
| ---------- | ------ |
| 0          | 0.748  |
| 50         | 0.751  |
| 100        | 0.754  |
| 150        | 0.753  |
| 200        | 0.752  |
</details>

(a) GC

![](images/8ad4e5ebdc17050b3caba1ec0a529c57fdbf752da513ffd0a165cf1bf95fac8f.jpg)

<details>
<summary>line</summary>

| Iterations | Score |
| ---------- | ----- |
| 0          | 0.243 |
| 50         | 0.246 |
| 100        | 0.248 |
| 150        | 0.247 |
| 200        | 0.246 |
</details>

(b) WQR   
Figure 7: The cross validation score of candidate features on training data across iterations. The dash line indicates the baseline cross validation score with raw dataset features.

![](images/f4683c3c1a9eef741e065f88e46197d773621e590ea0cc57066d33c61466432a.jpg)

<details>
<summary>line</summary>

| Iterations | Distance |
| ---------- | -------- |
| 0          | 0.55     |
| 50         | 0.45     |
| 100        | 0.48     |
| 150        | 0.47     |
| 200        | 0.46     |
</details>

(a) GC

![](images/54ea800d76f53fdc38f80b8b82278dece294c967dbeba77e9bcf5799325ab605.jpg)

<details>
<summary>line</summary>

| Iterations | Distance |
| ---------- | -------- |
| 0          | 0.45     |
| 50         | 0.48     |
| 100        | 0.47     |
| 150        | 0.46     |
| 200        | 0.45     |
</details>

(b) WQR   
Figure 9: The mean normalized edit distance between a candidate feature and previous five candidate features.

![](images/d7e38882d5bace13d63f16a80c7f1ace431af35d3ead6be1fd3db4e8cb62d6f8.jpg)

<details>
<summary>line</summary>

| Iterations | Order |
| ---------- | ----- |
| 0          | 2.0   |
| 50         | 3.5   |
| 100        | 4.0   |
| 150        | 3.8   |
| 200        | 3.6   |
</details>

(a) GC

![](images/3f24bee8f13b2dc2db423f2eb2505450f41d8841c6510de6dbd2b7a6255fc00b.jpg)

<details>
<summary>line</summary>

| Iterations | Order |
| ---------- | ----- |
| 0          | 2.5   |
| 50         | 3.8   |
| 100        | 3.6   |
| 150        | 3.9   |
| 200        | 3.7   |
</details>

(b) WQR   
Figure 8: The order of candidate features across iterations.

![](images/ae0f44e7effece009a45e9d83d600aad7b9ddfa5d992f33ee530d2b88095b159.jpg)

<details>
<summary>line</summary>

| Iterations | LLM Generations |
| ---------- | --------------- |
| 0          | 2               |
| 50         | 3               |
| 100        | 2               |
| 150        | 3               |
| 200        | 2               |
</details>

![](images/7a86803d12b4bfa83d32164ddd34451e846cb7a21f0bfe9a5635fbc033cf4932.jpg)

<details>
<summary>line</summary>

| Iterations | LLM Generations |
| ---------- | --------------- |
| 0          | 2.0             |
| 50         | 1.8             |
| 100        | 2.2             |
| 150        | 2.5             |
| 200        | 2.3             |
</details>

(b) WQR   
Figure 10: The number of LLM generations to construct a new candidate feature across iterations.

Temperature. Table 5 reports the maximum validation score across iterations and the number of LLM generations under different LLM sampling temperatures. The best performance and efficiency are both attained when the temperature is 1. Lower temperatures increase the likelihood of the LLM repeating previously constructed features, while higher temperatures make the LLM more prone to producing errors in the generations, both reducing feature construction efficiency. A temperature at 1 provides the best tradeoff between exploration and exploitation in feature search. The performance difference with the temperature at 1 and at 0.5 is statistically significant at $p < 0 . 0 1$ by one-tailed paired t-tests.

# 6 Conclusion

In this paper, we present a novel LLM-based AutoFE framework for effective, efficient, and interpretable feature engineering that leverages the semantic information of datasets. It features an elegant approach to instructing the LLM to generate semantically meaningful features with explanations by providing dataset descriptions and example features in cRPN expressions. The LLM iteratively explores the feature space and improves feature construction by learning top-performing examples in context. We have demonstrated in extensive experiments that our approach significantly outperforms state-of-the-art AutoFE methods. The incorporation of semantic context from dataset descriptions and the proposed feature canonicalization scheme both contribute to performance improvement. We have also provided comprehensive analyses on the LLM-based feature construction process. Our work opens up new possibilities discussed next for further LLM-driven applications on automated machine learning methodologies and underscores the potential of utilizing semantic information.

Table 4: Effect of the number of example features in the prompt with GPT-3.5. For each compared setting, the left column shows the validation score, and the right column shows the number of LLM generations. The best results are boldfaced. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="8">Number of Examples</td></tr><tr><td colspan="2">1</td><td colspan="2">5</td><td colspan="2">10</td><td colspan="2">20</td></tr><tr><td rowspan="3">RF</td><td>AF</td><td>0.7895</td><td>507.2</td><td>0.7930</td><td>409.2</td><td>0.7914</td><td>393.2</td><td>0.7860</td><td>372.8</td></tr><tr><td>WQR</td><td>0.3897</td><td>339.4</td><td>0.3937</td><td>329.8</td><td>0.3948</td><td>362.6</td><td>0.3940</td><td>330.0</td></tr><tr><td>CD</td><td>0.8212</td><td>480.2</td><td>0.8213</td><td>371.2</td><td>0.8219</td><td>349.8</td><td>0.8218</td><td>343.2</td></tr><tr><td rowspan="3">LGBM</td><td>AF</td><td>0.8421</td><td>440.4</td><td>0.8433</td><td>404.6</td><td>0.8430</td><td>380.2</td><td>0.8420</td><td>384.2</td></tr><tr><td>WQR</td><td>0.4248</td><td>346.0</td><td>0.4294</td><td>334.8</td><td>0.4301</td><td>322.8</td><td>0.4333</td><td>330.4</td></tr><tr><td>CD</td><td>0.8228</td><td>449.4</td><td>0.8224</td><td>361.2</td><td>0.8226</td><td>352.2</td><td>0.8228</td><td>321.2</td></tr><tr><td>Mean</td><td></td><td>0.6817</td><td>427.1</td><td>0.6839</td><td>368.5</td><td>0.6840</td><td>360.1</td><td>0.6833</td><td>347.0</td></tr></table>

Table 5: Effect of the LLM sampling temperature with GPT-3.5. For each compared setting, the left column shows the validation score, and the right column shows the number of LLM generations. The best results are boldfaced. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="6">Temperature</td></tr><tr><td colspan="2">0.5</td><td colspan="2">1</td><td colspan="2">1.5</td></tr><tr><td rowspan="2">RF</td><td>AF</td><td>0.7875</td><td>794.4</td><td>0.7914</td><td>393.2</td><td>0.7916</td><td>609.2</td></tr><tr><td>CD</td><td>0.8211</td><td>823.2</td><td>0.8219</td><td>349.8</td><td>0.8218</td><td>672.6</td></tr><tr><td rowspan="2">LGBM</td><td>AF</td><td>0.8365</td><td>1313.2</td><td>0.8430</td><td>380.2</td><td>0.8418</td><td>627.6</td></tr><tr><td>CD</td><td>0.8225</td><td>519.8</td><td>0.8226</td><td>352.2</td><td>0.8223</td><td>662.6</td></tr><tr><td>Mean</td><td></td><td>0.8169</td><td>862.7</td><td>0.8197</td><td>368.9</td><td>0.8194</td><td>643.0</td></tr></table>

# Limitations

One identified limitation is that our approach requires manual design of the prompt template, which involves trial and error. In the future, we plan to incorporate adaptive techniques for prompt design. Another limitation is that we have only shown the results of our approach with GPT models, which are widely regarded among the most powerful LLMs at the time of experiments. There remains future work to comprehensively examine the performance of our approach with other LLMs, since other LLMs may require different prompt design to achieve the optimal performance.

# Ethical Statement

All datasets used in this work are publicly available, free of personal information, and intended for research purposes only. Our use of OpenAI models complies with the terms and conditions of OpenAI.

# Acknowledgements

Yufeng Zou is partially supported by the Walter P. Murphy Fellowship. We would like to express our sincere gratitude to Allstate for the support of this project. The content is solely the responsibility of the authors and does not necessarily represent the official views of the funding agencies.

# References

Leo Breiman. 2001. Random forests. Machine learning, 45:5–32.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. In Proceedings of Advances in Neural Information Processing Systems, volume 33, pages 1877– 1901.   
Milton Friedman. 1937. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal of the American Statistical Association, 32(200):675–701.   
Sainyam Galhotra, Udayan Khurana, Oktie Hassanzadeh, Kavitha Srinivas, and Horst Samulowitz. 2019. Kafe: Automated feature enhancement for predictive modeling using external knowledge. In Proceedings of NeurIPS 2019 Workshop: Knowledge Representation & Reasoning Meets Machine Learning.   
Leo Grinsztajn, Edouard Oyallon, and Gael Varoquaux. 2022. Why do tree-based models still outperform deep learning on typical tabular data? In Proceedings of the thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track.   
Charles L Hamblin. 1962. Translation to and from polish notation. The Computer Journal, 5(3):210– 213.

Sungwon Han, Jinsung Yoon, Sercan O Arik, and Tomas Pfister. 2024. Large language models can automatically engineer features for few-shot tabular learning. In International Conference on Machine Learning, pages 17454–17479. PMLR.   
Noah Hollmann, Samuel Müller, and Frank Hutter. 2023. Large language models for automated data science: Introducing CAAFE for context-aware automated feature engineering. In Proceedings of the thirtyseventh Conference on Neural Information Processing Systems.   
Franziska Horn, Robert Pack, and Michael Rieger. 2020. The autofeat python library for automated feature engineering and selection. In Proceedings of Machine Learning and Knowledge Discovery in Databases, pages 111–120. Springer.   
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38.   
James Max Kanter and Kalyan Veeramachaneni. 2015. Deep feature synthesis: Towards automating data science endeavors. In Proceedings of 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA), pages 1–10. IEEE.   
Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. Lightgbm: A highly efficient gradient boosting decision tree. In Proceedings of Advances in Neural Information Processing Systems, volume 30.   
Udayan Khurana, Horst Samulowitz, and Deepak Turaga. 2018. Feature engineering for predictive modeling using reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32.   
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems.   
Liyao Li, Haobo Wang, Liangyu Zha, Qingyi Huang, Sai Wu, Gang Chen, and Junbo Zhao. 2023. Learning a data-driven policy network for pre-training automated feature engineering. In Proceedings of the eleventh International Conference on Learning Representations.   
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Comput. Surv., 55(9).   
Clint Morris, Michael Jurado, and Jason Zutty. 2024. Llm guided evolution - the automation of models advancing models. In Proceedings of the Genetic and Evolutionary Computation Conference, GECCO ’24, page 377–384.

Jaehyun Nam, Kyuyoung Kim, Seunghyuk Oh, Jihoon Tack, Jaehyung Kim, and Jinwoo Shin. 2024. Optimized feature generation for tabular data via llms with decision tree reasoning. In Advances in Neural Information Processing Systems, volume 37, pages 92352–92380.   
Fatemeh Nargesian, Horst Samulowitz, Udayan Khurana, Elias B Khalil, and Deepak S Turaga. 2017. Learning feature engineering for classification. In Proceedings of International Joint Conference on Artificial Intelligence, volume 17, pages 2529–2535.   
Peter Bjorn Nemenyi. 1963. Distribution-free multiple comparisons. Princeton University.   
OpenAI. 2023. Gpt-4 technical report. Preprint, arXiv:2303.08774.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. In Proceedings of Advances in Neural Information Processing Systems, volume 35, pages 27730–27744.   
Tom Overman, Diego Klabjan, and Jean Utke. 2024. IIFE: Interaction information based automated feature engineering. In 2024 IEEE International Conference on Data Mining (ICDM), pages 803–808. IEEE.   
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.   
Matthew G Smith and Larry Bull. 2005. Genetic programming with a genetic algorithm for feature construction and selection. Genetic Programming and Evolvable Machines, 6:265–281.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. Preprint, arXiv:2307.09288.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of Advances in Neural Information Processing Systems, volume 30.   
Kafeng Wang, Pengyang Wang, and Chengzhong Xu. 2023. Toward efficient automated feature engineering. In Proceedings of 2023 IEEE 39th International Conference on Data Engineering (ICDE), pages 1625–1637.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems.   
Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. 2024. Large language models as optimizers. In Proceedings of the twelfth International Conference on Learning Representations.   
Kaizhong Zhang and Dennis Shasha. 1989. Simple fast algorithms for the editing distance between trees and related problems. SIAM J. Comput., 18:1245–1262.   
Tianping Zhang, Zheyu Zhang, Zhiyuan Fan, Haoyan Luo, Fengyuan Liu, Qian Liu, Wei Cao, and Jian Li. 2023. Openfe: automated feature generation with expert-level performance. In Proceedings of the 40th International Conference on Machine Learning, ICML’23.   
Xinhao Zhang, Jinghan Zhang, Banafsheh Rekabdar, Yuanchun Zhou, Pengfei Wang, and Kunpeng Liu. 2024a. Dynamic and adaptive feature generation with llm. arXiv preprint arXiv:2406.03505.   
Yanlin Zhang, Ning Li, Quan Gan, Weinan Zhang, David Wipf, and Minjie Wang. 2024b. Elf-gym: Evaluating large language models generated features for tabular prediction. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 5420–5424.   
Guanghui Zhu, Shen Jiang, Xu Guo, Chunfeng Yuan, and Yihua Huang. 2022a. Evolutionary automated feature engineering. In Proceedings of Pacific Rim International Conference on Artificial Intelligence, pages 574–586. Springer.   
Guanghui Zhu, Zhuoer Xu, Chunfeng Yuan, and Yihua Huang. 2022b. Difer: Differentiable automated feature engineering. In Proceedings of the first International Conference on Automated Machine Learning, volume 188 of Proceedings of Machine Learning Research, pages 17/1–17. PMLR.   
Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. Preprint, arXiv:1909.08593.

# A Discussion on Canonical RPN Feature Representation

# A.1 Why RPN

RPN (Hamblin, 1962) provides a compact and unambiguous form of feature representation. In contrast, an infix expression requires extra information such as brackets to determine operator precedence. Without brackets, the feature in infix expression col-0 − ( col-1 + col-2 ) would be indistinguishable from the feature ( col-0 col-1 ) + col-2, while both features are distinctively encoded in RPN. Such compactness and unambiguity of RPN benefit sequential modeling since there is no need to model the extra information, e.g., the positions of brackets.

Compared with other forms of feature representation such as prefix expression of depth-first traversal or breadth-first traversal, RPN better encodes the recursive structure of an expression tree. The bottom-up enumeration of tree nodes makes it easy for an LLM to evaluate the feature expression by scanning the sequence from left to right, for instance, ((col-0 col-1 ) col-2 +) (parentheses denote recursion). Using the prefix expression (+ (− col-0 col-1) col-2) or breadth-first expression (+ ( [col-2] col-0 col-1)), however, the LLM always needs to look back for the operator, which undermines sequential modeling. We find in our experiments that when switching to prefix feature expressions, an LLM encounters difficulty in generating syntactically valid feature expressions.

# A.2 Why Canonicalization

Although there is one-to-one mapping between feature expression trees and RPN expressions, a feature that contains commutative operators (like addition and multiplication) can be represented by different RPN expressions, since the child nodes of these operators are unordered. We introduce a canonicalization scheme: arranging operator nodes before feature nodes and lexicographically sorting the nodes within each group. Through canonicalization, we create one-to-one mapping between features and cRPN expressions. This ensures the consistency of our feature representations and benefits in-context learning of feature patterns.

By arranging operator nodes before feature nodes, we also introduce left skewness to the expression tree that enhances clarity of the recursive structure in cRPN. As illustrated in Figure 2, the original feature expression (col-2 (col-1 col-0 +)

Algorithm 2: Feature Expression Tree to RPN   
Input : A feature expression tree T
Output: An RPN feature string f

1 r ← the root of T
2 Initialize string f ← ε, stack S ← [r], and visited ← ∅
3 repeat
4    u ← S.peek()
5    if u ∈ visited then
6    f.append(u)
7    S.pop()
8    end
9    else
10    for each child v of u in the reverse order do
11    | S.push(v)
12    end
13    visited ← visited ∪ {u}
14    end
15 until S is empty
16 return f

Algorithm 3: RPN to Feature Expression Tree   
Input : An RPN feature string f
Output: The root of a feature expression tree T
1 Initialize stack S ← []
2 for i ← 1 to |f| do
3    u ← the i-th element of f
4    if u is an operator then
5    o ← the arity of u
6    for j ← 1 to o do
7    v ← S.pop()
8    Prepend v to the list of children of u
9    end
10    end
11    S.push(u)
12 end
13 return S.pop()

) switches to ((col-0 col-1 +) col-2 ) after canonicalization, so that the LLM does not need to look back for col-2 when evaluating the expression.

# B Conversion between Feature Expression Tree and RPN

Algorithms 2 and 3 detail the process of conversion between a feature expression tree and a cRPN feature string. We check the syntactical validity of a feature string in Algorithm 3 by checking whether there are enough child nodes in the stack in line 6 and the size of the stack is exactly one (the root) in line 13 returning the output.

# C Example Prompt

# C.1 Full Prompt

Figure 11 shows an example of full prompts used in our main experiments.

Figure 11: Example full prompt on the Credit Default dataset.

# Dataset description:

This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

Dataset contains the following columns:

col-0 (int) [10000, 800000]: LIMIT\_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit

col-1 (category) {1, 2}: SEX: Gender (1=male, 2=female) col-2 (category) {0, 1, 2, 3, 4, 5, 6}: EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)

col-3 (category) {0, 1, 2, 3}: MARRIAGE: Marital status (1=married, 2=single, 3=others)

col-4 (int) [21, 79]: AGE: Age in years

col-5 (category) {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}: PAY\_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, . . . 8=payment delay for eight months, 9=payment delay for nine months and above)

col-23 (category) {0, 1}: default.payment.next.month: Default payment (1=yes, 0=no)

We have the following unary operators:

log: element-wise logarithm of the absolute value sqrt\_abs: element-wise square root of the absolute value min\_max: element-wise min-max normalization reciprocal: element-wise reciprocal

We have the following binary operators:

+: element-wise addition of two columns

: element-wise subtraction of two columns

: element-wise multiplication of two columns

/: element-wise division of two columns

mod\_column: element-wise modulo of two columns

Feature strings are reverse Polish notation (RPN) expressions that operate on the columns of our dataset. Each feature string constructs an extra column that is useful for the downstream model Random Forests to predict the target col-23. The model will be trained on the dataset with the constructed columns and evaluated on a holdout set. The best columns will be selected.

Below are feature strings arranged in ascending order based on their performance scores. Higher scores are better.

Feature

col-17,col-21,\*,col-20,+,sqrt\_abs

Score

0.0011

. . .

Feature

col-4,col-6,\*,col-12,col-16,-,sqrt\_abs,\*

Score

0.0014

Previous feature:

col-4,col-5,col-6,+,col-12,col-16,-,\*

Error: invalid RPN expression

Give me a new feature string that is different from all strings above and has a higher score. Use no more than five operators. Make sure all columns and operators exist and do not include the target column. Follow the syntax of RPN.

# Output format:

Feature

(Feature description)

Usefulness

(Explanation why this adds useful real world knowledge to predict the target col-23 according to dataset description)

# C.2 Semantically Blinded Prompt

Figure 12 shows an example of semantically blinded prompts used in our experiments in Section 5.3.

Figure 12: Example semantically blinded prompt on the Credit Default dataset. 

<table><tr><td>Dataset contains the following columns:</td></tr><tr><td>col-0</td></tr><tr><td>col-1</td></tr><tr><td>col-2</td></tr><tr><td>col-3</td></tr><tr><td>col-4</td></tr><tr><td>col-5</td></tr><tr><td>...</td></tr><tr><td>col-23</td></tr><tr><td>We have the following unary operators:</td></tr><tr><td>log: element-wise logarithm of the absolute value</td></tr><tr><td>sqrt_abs: element-wise square root of the absolute value</td></tr><tr><td>min_max: element-wise min-max normalization</td></tr><tr><td>reciprocal: element-wise reciprocal</td></tr><tr><td>We have the following binary operators:</td></tr><tr><td>+: element-wise addition of two columns</td></tr><tr><td>-: element-wise subtraction of two columns</td></tr><tr><td>*: element-wise multiplication of two columns</td></tr><tr><td>/: element-wise division of two columns</td></tr><tr><td>mod_column: element-wise modulo of two columns</td></tr><tr><td>Feature strings are reverse Polish notation (RPN) expressions that operate on the columns of our dataset. Each feature string constructs an extra column that is useful for the downstream model Random Forests to predict the target col-23. The model will be trained on the dataset with the constructed columns and evaluated on a holdout set. The best columns will be selected.</td></tr><tr><td>Below are feature strings arranged in ascending order based on their performance scores. Higher scores are better.</td></tr><tr><td>Feature</td></tr><tr><td>col-17,col-21,* ,col-20,+,sqrt_abs</td></tr><tr><td>Score</td></tr><tr><td>0.0011</td></tr><tr><td>...</td></tr><tr><td>Feature</td></tr><tr><td>col-4,col-6,* ,col-12,col-16,-,sqrt_abs,*</td></tr><tr><td>Score</td></tr><tr><td>0.0014</td></tr><tr><td>Previous feature:</td></tr><tr><td>col-4,col-6,* ,col-12,col-16,-,sqrt_abs,*</td></tr><tr><td>Error: duplication with candidate features</td></tr><tr><td>Give me a new feature string that is different from all strings above and has a higher score. Use no more than five operators. Make sure all columns and</td></tr></table>

operators exist and do not include the target column. Follow the syntax of RPN.

# Output format:

Feature

(Feature description)

Usefulness

(Explanation why this adds useful real world knowledge to predict the target col-23 according to dataset description)

# D Experimental Detail

# D.1 Dataset Sources

Table 6 summarizes the sources of datasets used in our experiments. Datasets are selected such that they cover different domains and both regression and classification tasks. Most of them have been used in prior works (Zhu et al., 2022a,b; Zhang et al., 2023; Hollmann et al., 2023).

# D.2 Experimental Platform

All experiments are conducted on the Ubuntu 22.04.4 LTS operating system, 16 Intel(R) Core(TM) i7-7820X CPUs, and 4 NVIDIA GeForce RTX 2080 Ti GPUs, with the framework of Python 3.11.9 and PyTorch 1.12.1.

# D.3 Feature Transformation Operators

We list the details of all feature transformation operators used in the experiments below.

Unary transformations:

• Logarithm: Element-wise logarithm of the absolute value;   
• Reciprocal: Element-wise reciprocal;   
• Square root: Element-wise square root of the absolute value;   
• Min-max normalization: Element-wise minmax normalization using the min and max values from the training data.

# Binary transformations:

• Addition: Element-wise addition;   
• Subtraction: Element-wise subtraction;   
• Multiplication: Element-wise multiplication;   
• Division: Element-wise division;   
• Modulo: Element-wise modulo.

Table 6: Sources of datasets used in experiments. 

<table><tr><td>Name</td><td>Source</td></tr><tr><td>Airfoil (AF)</td><td>https://archive.ics.uci.edu/dataset/291/airfoil+self+noise</td></tr><tr><td>Boston Housing (BH)</td><td>https://www.kaggle.com/datasets/arunjangir245/boston-housing-dataset</td></tr><tr><td>Bikeshare (BS)</td><td>https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset</td></tr><tr><td>Wine Quality Red (WQR)</td><td>https://archive.ics.uci.edu/dataset/186/wine+quality</td></tr><tr><td>AIDS Clinical Trials (ACT)</td><td>https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175</td></tr><tr><td>Credit Default (CD)</td><td>https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset</td></tr><tr><td>German Credit (GC)</td><td>https://archive.ics.uci.edu/dataset/573/south+german+credit+update</td></tr></table>

Table 7: Hyperparameter search for downstream models. 

<table><tr><td>Model</td><td>Parameter</td><td>Search Space*</td></tr><tr><td>Linear Model</td><td>regularization</td><td>loguniform(0.00001, 100)</td></tr><tr><td rowspan="4">Random Forests</td><td>num estimators</td><td>randint(5, 250)</td></tr><tr><td>max depth</td><td>randint(1, 250)</td></tr><tr><td>max features</td><td>uniform(0.01, 0.99)</td></tr><tr><td>max samples</td><td>uniform(0.1, 0.9)</td></tr><tr><td rowspan="6">LightGBM</td><td>num estimators</td><td>randint(10, 1000)</td></tr><tr><td>num leaves</td><td>randint(8, 64)</td></tr><tr><td>learning rate</td><td>loguniform(0.001, 1)</td></tr><tr><td>bagging fraction</td><td>uniform(0.1, 0.9)</td></tr><tr><td>feature fraction</td><td>uniform(0.1, 0.9)</td></tr><tr><td>reg lambda</td><td>loguniform(0.001, 100)</td></tr></table>

\* As specified in the scipy.stats package https://docs. scipy.org/doc/scipy/reference/stats.html.

# D.4 Parameter Tuning of Downstream Models

We tune the parameters of downstream models prior to and post AutoFE using randomized search implemented in an Sklearn package4. Table 7 lists the configurations of parameter tuning for each downstream model. We set the number of randomized search iterations to 100.

# D.5 Full Results

Tables 8-10 detail the full experimental results corresponding to the results in Tables 2 and 3. Tables 11-15 report the sample standard deviations corresponding to the experimental results in Tables 8-10 and Tables 4-5, respectively.

# D.6 Relative Performance Improvement

Tables 16 and 17 report the percentage performance improvement of PromptFE over the baseline methods using GPT-3.5 and GPT-4, respectively, corresponding to the experimental results in Tables 2 and 8.

# D.7 Statistical Tests

We perform the Friedman test (Friedman, 1937) to determine whether there is statistically significant difference among the compared AutoFE methods. The Friedman test p-values for the results in Tables 2 and 3 are $4 . 2 6 \times 1 0 ^ { - 5 0 }$ and $3 . 9 5 \times 1 0 ^ { - 3 4 }$ , respectively. Hence, we can reject the null hypothesis that the performance is the same for all methods. We perform the Nemenyi post-hoc test (Nemenyi, 1963) to further determine which AutoFE methods have different performance. Tables 18, 20, and 21 summarize the p-values for the pairwise comparisons in Tables 2 and 3. From Table 18, the performance difference between our method PromptFE and baseline methods other than DIFER (Zhu et al., 2022b) is statistically significant at the $p = 0 . 0 1$ level. From Table 20, the performance difference between the full version of PromptFE and the semantically blinded version is statistically significant at the $p = 0 . 0 1$ level. From Table 21, the performance difference is statistically significant at the p = 0.05 level for the cases with GPT-3.5 and post-AutoFE parameter tuning as well as GPT-4 without post-AutoFE parameter tuning.

To examine the performance difference when using Random Forests and LightGBM, we perform additional statistical tests for the results in Table 2 excluding the linear model results. The Friedman test p-value is $1 . 2 8 \times 1 0 ^ { - 2 5 }$ . Table 19 summarizes the p-values from the Nemenyi post-hoc test for pairwise comparison. We observe that PromptFE with GPT-3.5 and post-AutoFE parameter tuning significantly outperforms all baselines other than DIFER at the $p = 0 . 0 5$ level. With GPT-4, the performance difference between PromptFE and CAAFE (Hollmann et al., 2023) as well as OC-Tree (Nam et al., 2024) is statistically significant at the $p = 0 . 0 5$ level.

# D.8 Additional Hyperparameter Effect

Number of Iterations. Figure 13 shows the validation scores on the AF and CD datasets, which contain the smallest and largest numbers of features, respectively, using Random Forests and LightGBM. The validation score is evaluated after adding the selected set of candidate features to the dataset, as denoted by $s _ { n ^ { * } }$ in line 17 of Algorithm 1. We terminate our algorithm once we have 200 candidate features, as constructing additional features does not substantially enhance the performance, but constructing fewer features degrades the performance in some cases.

![](images/1943e73c25b7ce737080d86b764e0c7456b8f324357c6c99c304ff394fc2cf1a.jpg)

<details>
<summary>line</summary>

| Iterations | Score  |
| ---------- | ------ |
| 0          | 0.780  |
| 50         | 0.783  |
| 100        | 0.786  |
| 150        | 0.784  |
| 200        | 0.782  |
| 250        | 0.783  |
| 300        | 0.786  |
</details>

(a) RF on AF

![](images/2ce0df56d4b54385ddb9c304a7ee9f0468a947f97d8f110204016cefd0531ed1.jpg)

<details>
<summary>line</summary>

| Iterations | Score |
| ---------- | ----- |
| 0          | 0.810 |
| 50         | 0.835 |
| 100        | 0.838 |
| 150        | 0.832 |
| 200        | 0.830 |
| 250        | 0.831 |
| 300        | 0.832 |
</details>

(b) LGBM on AF

![](images/1ca8005fba8e9dc938b7e019a09da9dd92c8ec0493db82fe944dcb49cb4920be.jpg)

<details>
<summary>line</summary>

| Iterations | Score  |
| ---------- | ------ |
| 0          | 0.820  |
| 50         | 0.821  |
| 100        | 0.8215 |
| 150        | 0.821  |
| 200        | 0.8215 |
| 250        | 0.821  |
| 300        | 0.821  |
</details>

(c) RF on CD

![](images/b89b4eb1b0bc94fb5da3cb17e7b6aba595f55bcdb38ebe67b65438e0ebec7d7e.jpg)

<details>
<summary>line</summary>

| Iterations | Score  |
| ---------- | ------ |
| 0          | 0.821  |
| 50         | 0.822  |
| 100        | 0.8215 |
| 150        | 0.822  |
| 200        | 0.8225 |
| 250        | 0.8215 |
| 300        | 0.822  |
</details>

(d) LGBM on CD   
Figure 13: The validation score across iterations using Random Forests and LightGBM.

Number of Examples in Prompt with GPT-4. Table 22 reports the maximum validation score across iterations by varying the number of example features provided in the prompt to GPT-4. We observe improved performance as the number of example features increases. This suggests that providing more example features helps fully leverage GPT-4’s enhanced in-context learning capabilities. In our experiments, we set the number of example features to 10 for a fair comparison with GPT-3.5.

# D.9 Number of Selected Features

Table 23 compares the number of features added to the datasets. Our method PromptFE adaptively determines the number of features and selects fewer features than DIFER (Zhu et al., 2022b), demonstrating the efficacy of the features constructed by our method.

# D.10 Computation Cost

Table 24 compares the number of features evaluated during the feature search process. Guided by domain knowledge, our method PromptFE evaluates much fewer features than DIFER (Zhu et al., 2022b) and OpenFE (Zhang et al., 2023).

Tables 25 and 26 summarize the computation time using gpt-3.5-turbo-0125 as the LLM. For PromptFE, the computation time of LLM generation and feature evaluation is relatively stable across datasets of varying sizes. We note that the LLM generation time can be substantially reduced by instructing the LLM to generate multiple features in a generation step.

The sizes of datasets are listed in Table 1. We observe that only the feature selection time is sensitive to dataset sizes. While in general the cost of downstream model evaluations grows proportionally with the dataset size, the actual cost depends on the hyperparameter of the downstream model, e.g., the maximum tree depth. Based on the observations in Figure 10, the LLM generation time is roughly constant across iterations. The feature evaluation time is also constant. The feature selection time scales quadratically with the number of candidate features but can be computed in parallel.

# E Experiments on Proprietary Datasets

We have conducted experiments on a proprietary real-world dataset containing over 100, 000 samples and over 1, 000 features, where most features contain a substantial proportion of missing values. We select features of top-100 mutual information scores with the target, which filters out features with too many missing values, and perform AutoFE on those features. With data preprocessing, PromptFE brings significant performance improvement to downstream linear models on the proprietary real-world dataset.

# F Additional Analysis

# F.1 Feature Importance

Figure 15 shows feature importance across different datasets and downstream models. For linear models, we use the magnitudes of coefficients; for Random Forests (Breiman, 2001), we use the impurity-based feature importance; for Light-GBM (Ke et al., 2017), we use the total gains of splits. PromptFE enhances the datasets with generated features that extract valuable information for target prediction. Additionally, we observe that Random Forests and LightGBM benefit from features of higher orders compared to linear models, as they are capable of synthesizing simple features internally. Our method adaptively adjusts feature complexity to suit different downstream models.

# F.2 Feature Search Example

Figure 14 shows samples of the LLM-based feature search process from a single run. In iteration 155, the generated feature is derived from the combination of three example features. In iteration 158, the generated feature is derived from the combination of two example features, with the feature attributes modified. Iteration 160 exhibits an instance of modification on the example feature. Iteration 161 exhibits an instance of an analogy of the example feature.

Figure 14: Examples of feature search. The prompt examples that relate to the generated feature are highlighted in red.   
Iteration 155
Generated feature:
6, 12, division, 18, *, min_max
Prompt examples:
4, log, 13, *, 17, min_max, sqrt_abs, +
6, 12, *, 21, sqrt_abs, +
4, 14, +, 18, *, min_max
11, 21, +, 6, *, sqrt_abs
2, 4, +, 6, division, log
4, 6, +, 16, *, 17, +, min_max
6, 21, division, 17, log, *
12, min_max, 18, *, 5, mod_column
2, 6, mod_column, 17, +, sqrt_abs
2, 3, +, 4, sqrt_abs, *
4, 12, division, 16, *, min_max, sqrt_abs

# Iteration 158

Generated feature:

5, log, 19, \*, 15, +, min\_max

Prompt examples:

4, log, 13, \*, 17, min\_max, sqrt\_abs, +

6, 12, \*, 21, sqrt\_abs, +

4, 14, +, 18, \*, min\_max

11, 21, +, 6, \*, sqrt\_abs

2, 4, +, 6, division, log

4, 6, +, 16, \*, 17, +, min\_max

6, 21, division, 17, log, \*

12, min\_max, 18, \*, 5, mod\_column

2, 6, mod\_column, 17, +, sqrt\_abs

2, 3, +, 4, sqrt\_abs, \*

4, 12, division, 16, \*, min\_max, sqrt\_abs

# Iteration 160

Generated feature:

11, 21, \*, 4, +, 14, mod\_column

Prompt examples:

4, log, 13, \*, 17, min\_max, sqrt\_abs, +

6, 12, \*, 21, sqrt\_abs, +

4, 14, +, 18, \*, min\_max

11, 21, +, 6, *, sqrt_abs
2, 4, +, 6, division, log
4, 6, +, 16, *, 17, +, min_max
6, 21, division, 17, log, *
12, min_max, 18, *, 5, mod_column
2, 6, mod_column, 17, +, sqrt_abs
2, 3, +, 4, sqrt_abs, *
4, 12, division, 16, *, min_max, sqrt_abs

# Iteration 161

Generated feature:

6, 12, +, 17, \*, min\_max

Prompt examples:

6, 12, \*, 21, sqrt\_abs, +

4, 14, +, 18, \*, min\_max

11, 21, \*, 4, +, 14, mod\_column

11, 21, +, 6, \*, sqrt\_abs

2, 4, +, 6, division, log

4, 6, +, 16, \*, 17, +, min\_max

6, 21, division, 17, log, \*

12, min\_max, 18, \*, 5, mod\_column

2, 6, mod\_column, 17, +, sqrt\_abs

2, 3, +, 4, sqrt\_abs, \*

4, 12, division, 16, \*, min\_max, sqrt\_abs

# F.3 Potential Failure Modes

One potential failure mode is the generation of features that are duplicates of existing candidate features or syntactically invalid. The third column of each configuration in Table 3 reports the number of LLM generations needed to construct 200 candidate features in our experiments. Specifically, the proportion of valid new features is around 55% using GPT-3.5 and around 60% using GPT-4 on average. Feature search tends to converge per our feature divergence analyses in Section 5.6 and could get stuck in local optima when example features in the prompt are highly similar. From Figure 10, the number of LLM generations needed to construct a new candidate feature remains relatively stable as the algorithm iterates, suggesting low likelihood of getting stuck. The feedback of previous feature constructions helps the LLM avoid duplications and recover syntactical errors, reducing the likelihood of getting stuck. Table 4 shows that including more example features in the prompt improves the success rate of feature construction on average by increasing diversity. Another potential failure mode is that the generated explanation of a constructed feature may be inaccurate, e.g., the column index may be inconsistent with the feature name, a phenomenon known as faithfulness hallucinations (Ji et al., 2023).

# F.4 Additional Feature Attribute Selection Analysis

Figure 16 shows additional distributions of feature attributes in the constructed features for both linear models (LM) and Random Forests (RF) with GPT-4, complementing the results in Figure 6.

# F.5 Additional Performance Analysis

Figures 17-20 show additional performance analysis results for linear models with GPT-3.5, complementing the results in Figures 7-10.

# G Additional Discussion on Differences from Existing Works

Although our work PromptFE and CAAFE (Hollmann et al., 2023) both utilize LLMs to construct new features incorporating dataset semantic information, they differ in several key aspects. We design PromptFE such that it taps into the in-context learning capability of LLMs and performs effective feature search. In PromptFE, we provide topperforming constructed features in the prompt as learning examples, label them with performance scores, and rank them by score. We demonstrate that the LLM learns to optimize feature construction over the course of algorithm. CAAFE instead stores all previous instructions and code snippets in the conversation history, which hinders the incontext learning of optimal feature patterns. It quickly consumes the LLM’s context as the algorithm iterates, incurring more and more LLM generation costs. In comparison, the LLM generation cost of PromptFE stays constant across iterations, without a maximum limit on the number of iterations it can perform. Therefore, PromptFE has stronger capability of performing feature search in large search spaces requiring many iterations, such as datasets with numerous feature attributes.

In PromptFE, we also explore representing features in a different form of cRPN. We refer to Appendix A for further details. Compared with the Python code representation in CAAFE, cRPN is more compact, which not only reduces LLM generation costs but also makes in-context learning of feature patterns easier, and more human interpretable. The use of pre-defined operators reduces the search space and simplifies the learning process for optimizing feature construction. Together, our approach gives a better control than code representation that helps avoid undesirable or unexpected LLM outputs. Another advantage of cRPN is that it is convenient to import external features (as outlined in Algorithm 1) and export the results as individual features, providing compatibility with other feature engineering methods.

The rules generated by FeatLLM (Han et al., 2024) are based on a single raw feature, without considering high-order feature interactions. There is no feedback mechanism to improve the rules either. In comparison, our approach constructs new features that combine multiple raw features and iteratively improves feature construction by learning the performance. Furthermore, FeatLLM can be used for only classification tasks and for a single type of a downstream model. Our approach is effective for both classification and regression tasks, and it adaptively constructs features that are useful for different downstream models.

ELF-Gym (Zhang et al., 2024b) first generates feature descriptions using one LLM and then generates feature code using another LLM, which is less efficient, and it does not have a feedback mechanism to improve the features. Differently, we represent features in the compact form of cRPNs. Our approach iteratively improves feature construction by learning the performance, where the LLM simultaneously generates new features and explanations on its own.

Compared with OCTree (Nam et al., 2024), our approach represents features concisely in the form of canonical RPNs without using external modules like decision trees. The processes of feature encoding, decoding, and validity check detailed in Algorithms 2 and 3 are simple and efficient. We have demonstrated that cRPNs are effective for the LLMs to understand the structure of features and construct new features to improve utilities. The LLMs are also able to semantically explain the constructed features in the context of dataset descriptive information on its own. Our approach gives a better control, such as the number of operators to use to construct features, and facilitates feature search by reducing the search space with a set of pre-defined operators. Moreover, with fewer modules used, our approach is more robust and cost efficient.

While LFG (Zhang et al., 2024a) is also LLM based, it does not utilize the semantic information of datasets. We have shown in Sections 5.3 and 5.5 that the incorporation of dataset semantic information enhances the effectiveness of feature construction of our approach. Informed by dataset semantic information, our approach circumvents exhaustive feature search and evaluates considerably fewer candidate features than traditional approaches, while providing semantic explanations of the constructed features. Another difference is that we represent features in the compact and unambiguous form of cRPNs, which not only reduces LLM generation costs but also facilitates in-context learning of feature patterns. Furthermore, compared with the work in (Zhang et al., 2024a) , we conduct more comprehensive experiments by benchmarking against state-of-the-art LLM- and non-LLM-based AutoFE methods on both regression and classification tasks and perform more detailed performance analyses, such as feature attribute selection, feature complexity, and feature construction efficiency.

More fundamentally, we demonstrate in this work that general-purpose LLMs like GPTs can effectively model recursive tree structures in the form of cRPN feature expressions and reason about the structures in the context of semantic information, shedding light on further LLM-driven applications. We hereby underscore the importance of adopting proper representation for the downstream task to tap into LLMs’ potential.

# H Practical Significance

PromptFE constructs semantically meaningful features that significantly boost the performance of simple predictive models especially linear models and provides semantic explanations. Our cRPN feature representation is concise and easy to interpret. Using our approach, one can enhance the performance of simple predictive models without sacrificing their interpretability. Our toolkit is easy to deploy, requiring only OpenAI APIs.

# I Source Code

The source code and data of this work can be accessed at https://github.com/zyf505/ PromptFE.

![](images/8d58aeceb137e15fbef31f89f025a027531df3387f0d04fa5b902fae6807b1dc.jpg)

<details>
<summary>bar_line</summary>

| Features | Normalized Importance Score |
| -------- | --------------------------- |
| 0        | 0.0                         |
| 5        | 0.2                         |
| 10       | 0.5                         |
| 15       | 0.18                        |
| 20       | 0.4                         |
| 25       | 0.0                         |
| 30       | 0.0                         |
| 35       | 0.45                        |
| 40       | 0.1                         |
| 45       | 0.2                         |
| 50       | 1.0                         |
| 55       | 0.0                         |
| 60       | 0.0                         |
| 65       | 0.1                         |
| 70       | 0.1                         |
| 75       | 0.18                        |
| 80       | 0.0                         |
| 85       | 0.4                         |
| 90       | 0.2                         |
| 95       | 0.2                         |
| 100      | 0.1                         |
| 105      | 0.25                        |
| 110      | 0.0                         |
| 115      | 0.1                         |
| 120      | 0.25                        |
| 125      | 0.1                         |
| 130      | 0.0                         |
| 135      | 0.1                         |
| 140      | 0.1                         |
| 145      | 0.1                         |
| 150      | 0.15                        |
| 155      | 0.08                        |
| 160      | 0.2                         |
| 165      | 0.0                         |
</details>

(a) Logistic regression on BH. Test performance improves from to 0.3776 to 0.5157.

![](images/aeed23980c90052c5563ba342ebd9b0c8ddfc7a55c7d8ec021acbf00f8720cd5.jpg)

<details>
<summary>bar</summary>

| Feature | Normalized Importance Score | Feature Order |
| ------- | --------------------------- | ------------- |
| 1       | 1.0                         | 6             |
| 2       | 0.3                         | 3             |
| 3       | 0.1                         | 3             |
| 4       | 0.25                        | 4             |
| 5       | 0.15                        | 3             |
| 6       | 0.18                        | 3             |
| 7       | 0.2                         | 3             |
| 8       | 0.35                        | 4             |
| 9       | 0.15                        | 3             |
| 10      | 0.65                        | 4             |
| 11      | 0.28                        | 3             |
| 12      | 0.25                        | 3             |
| 13      | 0.35                        | 3             |
| 14      | 0.45                        | 3             |
| 15      | 0.8                         | 4             |
| 16      | 0.2                         | 3             |
| 17      | 0.15                        | 2             |
| 18      | 0.35                        | 3             |
| 19      | 0.4                         | 3             |
| 20      | 0.25                        | 4             |
| 21      | 0.05                        | 2             |
| 22      | 0.05                        | 2             |
| 23      | 0.1                         | 2             |
| 24      | 0.05                        | 2             |
| 25      | 0.05                        | 2             |
| 26      | 0.05                        | 2             |
| 27      | 0.05                        | 2             |
| 28      | 0.05                        | 2             |
| 29      | 0.05                        | 2             |
| 30      | 0.05                        | 2             |
| 31      | 0.05                        | 2             |
| 32      | 0.05                        | 2             |
| 33      | 0.05                        | 2             |
| 34      | 0.05                        | 2             |
| 35      | 0.05                        | 2             |
| 36      | 0.05                        | 2             |
| 37      | 0.05                        | 2             |
| 38      | 0.05                        | 2             |
| 39      | 0.05                        | 2             |
| 40      | 0.05                        | 2             |
| 41      | 0.05                        | 2             |
| 42      | 0.05                        | 2             |
| 43      | 0.05                        | 2             |
| 44      | 0.05                        | 2             |
| 45      | 0.05                        | 2             |
| 46      | 0.05                        | 2             |
| 47      | 0.05                        | 2             |
| 48      | 0.05                        | 2             |
| 49      | 0.05                        | 2             |
| 50      | 0.05                        | 2             |
| Note: The actual values for Feature Order are not provided in the code snippet. The actual values would be the result of the data generation and the number of features that fall below each other's value in the bar chart. [The values for Feature Order] are not explicitly provided in the code.
</details>

(b) Random Forests on GC. Test performance improves from to 0.7450 to 0.7700.

![](images/fff0209b179bb8abc24b8e2e0d601adc5849754348653e346caf64b884ed7eb5.jpg)

<details>
<summary>bar_line</summary>

| Features | Normalized Importance Score | Feature Order |
| -------- | --------------------------- | ------------- |
| 0        | 0.1                         | 2             |
| 5        | 0.05                        | 3             |
| 10       | 0.08                        | 4             |
| 15       | 0.03                        | 5             |
| 20       | 0.06                        | 3             |
| 25       | 0.09                        | 4             |
| 30       | 0.04                        | 5             |
| 35       | 0.07                        | 3             |
| 40       | 0.02                        | 4             |
| 45       | 0.08                        | 5             |
| 50       | 0.1                         | 3             |
| 55       | 0.05                        | 4             |
| 60       | 0.09                        | 5             |
| 65       | 0.12                        | 3             |
| 70       | 0.06                        | 4             |
| 75       | 0.07                        | 5             |
| 80       | 0.04                        | 3             |
| 85       | 0.08                        | 4             |
| 90       | 0.1                         | 5             |
| 95       | 0.05                        | 3             |
| 100      | 0.25                        | 4             |
| 105      | 0.1                         | 5             |
| 110      | 0.15                        | 3             |
| 115      | 0.08                        | 4             |
| 120      | 0.1                         | 5             |
| 125      | 0.2                         | 3             |
| 130      | 0.25                        | 4             |
| 135      | 1.0                         | 6             |
| 140      | 0.05                        | 3             |
| 145      | 0.03                        | 4             |
| 150      | 0.2                         | 3             |
| 155      | 0.5                         | 4             |
| 160      | 0.7                         | 5             |
| 165      | 0.2                         | 3             |
| 170      | 0.1                         | 4             |
| 175      | 0.1                         | 3             |
</details>

(c) LightGBM on WQR. Test performance improves from to 0.3825 to 0.4299.   
Figure 15: The blue bars show the normalized feature importance scores. The red dots show the order of features. The constructed features positioned on the x-axis start at index 0, following raw dataset features.

![](images/f143cdb15df88a06ca81cc6212aaa287c5fedd548a453b5302aab10ecb64bb61.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.2        |
| 1                       | 0.3        |
| 2                       | 0.4        |
| 3                       | 0.5        |
| 4                       | 0.6        |
| 5                       | 0.7        |
| 6                       | 0.8        |
| 7                       | 0.9        |
| 8                       | 1.0        |
| 9                       | 1.1        |
| 10                      | 1.2        |
| 11                      | 1.3        |
| 12                      | 1.4        |
| 13                      | 1.5        |
| 14                      | 1.6        |
| 15                      | 1.7        |
| 16                      | 1.8        |
| 17                      | 1.9        |
| 18                      | 2.0        |
| 19                      | 2.1        |
| 20                      | 2.2        |
</details>

(a) ACT+LM

![](images/db24c33cd94296e74ab063e2c7058774a8623fe421d5e6d5899dd56db55245be.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.2        |
| 1                       | 0.6        |
| 2                       | 0.5        |
| 3                       | 0.4        |
| 4                       | 0.3        |
| 5                       | 0.6        |
| 6                       | 0.5        |
| 7                       | 0.2        |
| 8                       | 0.1        |
| 9                       | 0.1        |
| 10                      | 0.05       |
| 11                      | 0.05       |
| 12                      | 0.05       |
| 13                      | 0.05       |
| 14                      | 0.05       |
| 15                      | 0.05       |
| 16                      | 0.05       |
| 17                      | 0.05       |
| 18                      | 0.05       |
| 19                      | 0.05       |
| 20                      | 0.05       |
</details>

(b) ACT+LM, blinded

![](images/d56fd48805ca12fe599f42a37290f612bbf3a43322dba9a9b269c4aac95cc7f1.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.35       |
| 1                       | 0.38       |
| 2                       | 0.40       |
| 3                       | 0.55       |
| 4                       | 0.38       |
| 5                       | 0.02       |
| 6                       | 0.03       |
| 7                       | 0.02       |
| 8                       | 0.03       |
| 9                       | 0.10       |
| 10                      | 0.18       |
| 11                      | 0.08       |
| 12                      | 0.02       |
| 13                      | 0.01       |
| 14                      | 0.01       |
| 15                      | 0.01       |
| 16                      | 0.01       |
| 17                      | 0.01       |
| 18                      | 0.01       |
| 19                      | 0.75       |
| 20                      | 0.45       |
| 21                      | 0.25       |
</details>

(c) ACT+RF

![](images/0ec40016e532b1184e939b514f49ec5d3308da11e1bf13213da8fa325a2849da.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.7        |
| 1                       | 0.6        |
| 2                       | 0.5        |
| 3                       | 0.4        |
| 4                       | 0.3        |
| 5                       | 0.2        |
| 6                       | 0.1        |
| 7                       | 0.05       |
| 8                       | 0.03       |
| 9                       | 0.02       |
| 10                      | 0.01       |
| 11                      | 0.01       |
| 12                      | 0.01       |
| 13                      | 0.01       |
| 14                      | 0.01       |
| 15                      | 0.01       |
| 16                      | 0.01       |
| 17                      | 0.01       |
| 18                      | 0.01       |
| 19                      | 0.01       |
| 20                      | 0.01       |
</details>

(d) ACT+RF, blinded

![](images/89469e48a35b011b3bbdf84ee75886f125184705018c6a34980f4f7f182d97b1.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.2        |
| 1                       | 0.3        |
| 2                       | 0.2        |
| 3                       | 0.1        |
| 4                       | 0.3        |
| 5                       | 0.8        |
| 6                       | 0.3        |
| 7                       | 0.1        |
| 8                       | 0.1        |
| 9                       | 0.1        |
| 10                      | 0.3        |
| 11                      | 0.2        |
| 12                      | 0.1        |
</details>

(e) BH+LM

![](images/b010b9ecfe88a64b070ec4c8b83a9398a224636037d4234165f3a82f21ad6567.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.1        |
| 1                       | 0.5        |
| 2                       | 0.6        |
| 3                       | 0.4        |
| 4                       | 0.5        |
| 5                       | 0.6        |
| 6                       | 0.4        |
| 7                       | 0.2        |
| 8                       | 0.1        |
| 9                       | 0.1        |
| 10                      | 0.05       |
</details>

(f) BH+LM, blinded

![](images/618348d2b41215e9c0d32a82def94eb58e311fcab376c0d13be42b8a9f13256b.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.4        |
| 1                       | 0.45       |
| 2                       | 0.25       |
| 3                       | 0.15       |
| 4                       | 0.35       |
| 5                       | 0.6        |
| 6                       | 0.25       |
| 7                       | 0.1        |
| 8                       | 0.15       |
| 9                       | 0.15       |
| 10                      | 0.4        |
| 11                      | 0.3        |
| 12                      | 0.2        |
</details>

(g) BH+RF

![](images/2f507a179eeb3de871543eaa7aae9202e899f3ce78a3aafc0532044dbdf83399.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.2        |
| 1                       | 0.5        |
| 2                       | 0.4        |
| 3                       | 0.6        |
| 4                       | 0.3        |
| 5                       | 0.6        |
| 6                       | 0.2        |
| 7                       | 0.1        |
| 8                       | 0.1        |
| 9                       | 0.1        |
| 10                      | 0.0        |
</details>

(h) BH+RF, blinded

![](images/31796107f5ac876c922ae8a66927be81b95e1bb63c28ba4e1c857308ba8c7786.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.2        |
| 1                       | 0.7        |
| 2                       | 0.4        |
| 3                       | 0.8        |
| 4                       | 0.5        |
| 5                       | 0.2        |
| 6                       | 0.1        |
| 7                       | 0.15       |
| 8                       | 0.1        |
| 9                       | 0.15       |
| 10                      | 0.3        |
| 11                      | 0.05       |
| 12                      | 0.05       |
| 13                      | 0.05       |
| 14                      | 0.05       |
| 15                      | 0.05       |
| 16                      | 0.05       |
| 17                      | 0.05       |
| 18                      | 0.05       |
| 19                      | 0.05       |
| 20                      | 0.05       |
</details>

(i) GC+LM

![](images/2c3b62b7ef2fbe700cfb5849dd7792bab522f42aa146d9f7a568736297661c61.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.1        |
| 1                       | 0.7        |
| 2                       | 0.6        |
| 3                       | 0.45       |
| 4                       | 0.55       |
| 5                       | 0.25       |
| 6                       | 0.1        |
| 7                       | 0.05       |
| 8                       | 0.05       |
| 9                       | 0.1        |
| 10                      | 0.05       |
| 11                      | 0.05       |
| 12                      | 0.05       |
| 13                      | 0.05       |
| 14                      | 0.05       |
| 15                      | 0.05       |
</details>

(j) GC+LM, blinded

![](images/0e1195e358ef69fa70b3b28ec7ed3bee61fa269b3f001c2f485c52a8ffb52145.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.8        |
| 1                       | 0.2        |
| 2                       | 0.1        |
| 3                       | 0.5        |
| 4                       | 0.2        |
| 5                       | 0.1        |
| 6                       | 0.2        |
| 7                       | 0.1        |
| 8                       | 0.1        |
| 9                       | 0.1        |
| 10                      | 0.4        |
| 11                      | 0.1        |
| 12                      | 0.1        |
| 13                      | 0.1        |
| 14                      | 0.05       |
| 15                      | 0.05       |
</details>

(k) GC+RF

![](images/5ec133ba4891144a8c76d737f085077f72dfaca2c7a77196f7c2c58511a9d607.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.2        |
| 1                       | 0.5        |
| 2                       | 0.4        |
| 3                       | 0.4        |
| 4                       | 0.4        |
| 5                       | 0.4        |
| 6                       | 0.3        |
| 7                       | 0.2        |
| 8                       | 0.1        |
| 9                       | 0.1        |
| 10                      | 0.05       |
| 11                      | 0.05       |
| 12                      | 0.05       |
| 13                      | 0.05       |
| 14                      | 0.05       |
| 15                      | 0.05       |
</details>

(l) GC+RF, blinded

![](images/1aae76dded755ac840b718ca433c2e75997acfa207ea014be3956ccc1d1dfd4a.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.5        |
| 1                       | 0.6        |
| 2                       | 0.4        |
| 3                       | 0.5        |
| 4                       | 0.3        |
| 5                       | 0.2        |
| 6                       | 0.1        |
| 7                       | 0.2        |
| 8                       | 0.3        |
| 9                       | 0.6        |
| 10                      | 0.5        |
</details>

(m) WQR+LM

![](images/8983ca6e06152075a1060b8944d293a9ad87418cf2e5c42b86c94f7b77e1fe00.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.45       |
| 1                       | 0.75       |
| 2                       | 0.68       |
| 3                       | 0.48       |
| 4                       | 0.52       |
| 5                       | 0.50       |
| 6                       | 0.22       |
| 7                       | 0.05       |
| 8                       | 0.06       |
| 9                       | 0.07       |
| 10                      | 0.01       |
</details>

(n) WQR+LM, blinded

![](images/390e955f832f0d52109fbe100c1d2f5495c4f178a7633071af8b831350c6e160.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.6        |
| 1                       | 0.7        |
| 2                       | 0.3        |
| 3                       | 0.4        |
| 4                       | 0.35       |
| 5                       | 0.3        |
| 6                       | 0.2        |
| 7                       | 0.1        |
| 8                       | 0.25       |
| 9                       | 0.6        |
| 10                      | 0.5        |
</details>

(o) WQR+RF

![](images/af05dc4804ec56b3fbae7aac5deaf9e593955ad9fdd250db35acf51f3eae44d2.jpg)

<details>
<summary>boxplot</summary>

| Feature Attribute Index | Proportion |
| ----------------------- | ---------- |
| 0                       | 0.4        |
| 1                       | 0.65       |
| 2                       | 0.7        |
| 3                       | 0.5        |
| 4                       | 0.6        |
| 5                       | 0.15       |
| 6                       | 0.05       |
| 7                       | 0.02       |
| 8                       | 0.01       |
| 9                       | 0.01       |
| 10                      | 0.01       |
</details>

(p) WQR+RF, blinded   
Figure 16: Distributions of feature attribute selection in the constructed features with GPT-4.

![](images/01f96868c664fb6494abc49230992de2f44a4864115a8544a2d82c8f77b430fd.jpg)

<details>
<summary>line</summary>

| Iterations | Score  |
| ---------- | ------ |
| 0          | 0.871  |
| 50         | 0.871  |
| 100        | 0.871  |
| 150        | 0.874  |
| 200        | 0.871  |
</details>

(a) ACT

![](images/106d46ff7e89c855bc152dd384c63bd1aaa26fdb9fc6ba126d8e53435b5b3d91.jpg)

<details>
<summary>line</summary>

| Iterations | Score |
| ---------- | ----- |
| 0          | 0.52  |
| 50         | 0.53  |
| 100        | 0.54  |
| 150        | 0.53  |
| 200        | 0.54  |
</details>

(b) BH

![](images/b6522487ae250b034fe1986ac30cefeaaefb28e71eefc2aa158084cce8e7611c.jpg)

<details>
<summary>line</summary>

| Iterations | Score  |
| ---------- | ------ |
| 0          | 0.748  |
| 50         | 0.751  |
| 100        | 0.754  |
| 150        | 0.753  |
| 200        | 0.752  |
</details>

(c) GC

![](images/8560d8e7616c41e30a41713405396730c93540599d5398e08d52f87835e3e19f.jpg)

<details>
<summary>line</summary>

| Iterations | Score  |
| ---------- | ------ |
| 0          | 0.243  |
| 50         | 0.246  |
| 100        | 0.249  |
| 150        | 0.246  |
| 200        | 0.243  |
</details>

(d) WQR   
Figure 17: The cross validation score of candidate features on training data across iterations. The dash line indicates the baseline cross validation score with raw dataset features.

![](images/76ac4237a56f1ee9d233738116660a4ab299a10e1f4b2fc076bdbf98ab83083a.jpg)

<details>
<summary>line</summary>

| Iterations | Order |
| ---------- | ----- |
| 0          | 2.0   |
| 50         | 3.5   |
| 100        | 3.0   |
| 150        | 3.8   |
| 200        | 3.2   |
</details>

(a) ACT

![](images/9165c4a9c20ce79c8f98cedab3510b8018eddc8fedf9451f34590c4c6e89d989.jpg)

<details>
<summary>line</summary>

| Iterations | Order |
| ---------- | ----- |
| 0          | 2.0   |
| 50         | 3.5   |
| 100        | 3.0   |
| 150        | 3.2   |
| 200        | 3.0   |
</details>

(b) BH

![](images/3480b9463865dee320f08577402b60f7da854ef7eec39ba34e5e118d0cf886e7.jpg)

<details>
<summary>line</summary>

| Iterations | Order |
| ---------- | ----- |
| 0          | 2.5   |
| 50         | 3.5   |
| 100        | 3.8   |
| 150        | 4.0   |
| 200        | 3.7   |
</details>

(c) GC

![](images/8acc0850f8e6c8c6dca4394d7817c73cd4ebd0d0e3d93c969d5cb51af52cc3ea.jpg)

<details>
<summary>line</summary>

| Iterations | Order |
| ---------- | ----- |
| 0          | 2.5   |
| 50         | 3.8   |
| 100        | 3.6   |
| 150        | 4.2   |
| 200        | 3.7   |
</details>

(d) WQR   
Figure 18: The order of candidate features across iterations.

![](images/daabc84a38f5c2838f3f8d51679b07fa518fe496e6923119eaffd83728aed7eb.jpg)

<details>
<summary>line</summary>

| Iterations | Distance |
| ---------- | -------- |
| 0          | 0.50     |
| 50         | 0.48     |
| 100        | 0.49     |
| 150        | 0.47     |
| 200        | 0.46     |
</details>

(a) ACT

![](images/d1338ee0687e6fc109ba0747a91b868e37e328f3afc69baa0529bca8f501734a.jpg)

<details>
<summary>line</summary>

| Iterations | Distance |
| ---------- | -------- |
| 0          | 0.50     |
| 50         | 0.45     |
| 100        | 0.48     |
| 150        | 0.46     |
| 200        | 0.47     |
</details>

(b) BH

![](images/6ba95cbd085c84f40266019511535d842fea281f4314e71326f681dc0701a61a.jpg)

<details>
<summary>line</summary>

| Iterations | Distance |
| ---------- | -------- |
| 0          | 0.55     |
| 50         | 0.45     |
| 100        | 0.48     |
| 150        | 0.47     |
| 200        | 0.43     |
</details>

(c) GC

![](images/4a44fd53158fef4ac09c5420bce084d1ac3b701a9be42258c6136801a00ce405.jpg)

<details>
<summary>line</summary>

| Iterations | Distance |
| ---------- | -------- |
| 0          | 0.48     |
| 50         | 0.49     |
| 100        | 0.47     |
| 150        | 0.46     |
| 200        | 0.45     |
</details>

(d) WQR   
Figure 19: The mean normalized edit distance between a candidate feature and previous five candidate features across iterations.

![](images/a3b886d95fe21d57644d346414ed2943c1744d08ca6c3bd0cfe819e1c2b78cd3.jpg)

<details>
<summary>line</summary>

| Iterations | LLM Generations |
| ---------- | --------------- |
| 0          | ~2              |
| 50         | ~3              |
| 100        | ~3              |
| 150        | ~3              |
| 200        | ~3              |
</details>

(a) ACT

![](images/925ad87cb108cf570c3663f40e7386b24dfd1a8b0ada11d1ec992e7a6ce065e3.jpg)

<details>
<summary>line</summary>

| Iterations | LLM Generations |
| ---------- | --------------- |
| 0          | 1               |
| 50         | 2               |
| 100        | 3               |
| 150        | 2               |
| 200        | 3               |
</details>

(b) BH

![](images/abe3447e0ebc25fcc433716df0da3242717e3c2817a648d8df3962ff09d15d09.jpg)

<details>
<summary>line</summary>

| Iterations | LLM Generations |
| ---------- | --------------- |
| 0          | 2               |
| 50         | 2               |
| 100        | 2               |
| 150        | 2               |
| 200        | 2               |
</details>

(c) GC

![](images/62502c92941ee0a3bdf36a65123744d030ef2746d2d43dc0b773c5105de8c557.jpg)

<details>
<summary>line</summary>

| Iterations | LLM Generations |
| ---------- | --------------- |
| 0          | 1               |
| 50         | 2               |
| 100        | 3               |
| 150        | 2               |
| 200        | 1               |
</details>

(d) WQR   
Figure 20: The number of LLM generations to construct a new candidate feature across iterations.

Table 8: Full experimental results of Table 2 comparison of overall performance. For each compared method, the left and right columns show the results without and with parameter tuning of the downstream model algorithm post AutoFE, respectively. The best results are boldfaced, and the second best results are underlined. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td rowspan="2">Raw</td><td rowspan="2" colspan="2">DIFER</td><td rowspan="2" colspan="2">OpenFE</td><td colspan="4">CAAFE</td><td colspan="2">OCTree</td><td colspan="4">PromptFE (ours)</td></tr><tr><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td><td colspan="2">GPT-4</td><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td></tr><tr><td rowspan="7">Linear Model</td><td>AF</td><td>0.3474</td><td>0.5870</td><td>0.6090</td><td>0.4300</td><td>0.4303</td><td>0.4011</td><td>0.4016</td><td>0.4376</td><td>0.4378</td><td>0.4698</td><td>0.4698</td><td>0.6612</td><td>0.6616</td><td>0.6649</td><td>0.6647</td></tr><tr><td>BH</td><td>0.3776</td><td>0.5013</td><td>0.4994</td><td>0.3900</td><td>0.3880</td><td>0.4788</td><td>0.4765</td><td>0.4503</td><td>0.4506</td><td>0.4480</td><td>0.4545</td><td>0.4995</td><td>0.5025</td><td>0.5184</td><td>0.5289</td></tr><tr><td>BS</td><td>1.0000</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>WQR</td><td>0.2696</td><td>0.2475</td><td>0.2630</td><td>0.2713</td><td>0.2736</td><td>0.2742</td><td>0.2757</td><td>0.2776</td><td>0.2776</td><td>0.2774</td><td>0.2778</td><td>0.2722</td><td>0.2745</td><td>0.2713</td><td>0.2748</td></tr><tr><td>ACT</td><td>0.8505</td><td>0.8715</td><td>0.8799</td><td>0.8729</td><td>0.8729</td><td>0.8519</td><td>0.8514</td><td>0.8565</td><td>0.8570</td><td>0.8724</td><td>0.8720</td><td>0.8729</td><td>0.8794</td><td>0.8766</td><td>0.8762</td></tr><tr><td>CD</td><td>0.8267</td><td>0.8273</td><td>0.8280</td><td>0.8265</td><td>0.8268</td><td>0.8265</td><td>0.8267</td><td>0.8238</td><td>0.8238</td><td>0.8270</td><td>0.8272</td><td>0.8282</td><td>0.8282</td><td>0.8288</td><td>0.8288</td></tr><tr><td>GC</td><td>0.7100</td><td>0.7140</td><td>0.7420</td><td>0.7320</td><td>0.7280</td><td>0.7350</td><td>0.7330</td><td>0.7210</td><td>0.7210</td><td>0.7280</td><td>0.7250</td><td>0.7570</td><td>0.7460</td><td>0.7590</td><td>0.7420</td></tr><tr><td>Mean</td><td></td><td>0.5636</td><td>0.6248</td><td>0.6369</td><td>0.5871</td><td>0.5866</td><td>0.5946</td><td>0.5941</td><td>0.5945</td><td>0.5946</td><td>0.6038</td><td>0.6044</td><td>0.6485</td><td>0.6487</td><td>0.6532</td><td>0.6526</td></tr><tr><td colspan="2">Mean Rank</td><td>14.00</td><td>9.17</td><td>5.83</td><td>10.58</td><td>9.92</td><td>10.00</td><td>9.50</td><td>10.50</td><td>9.83</td><td>8.25</td><td>7.33</td><td>5.08</td><td>3.50</td><td>3.33</td><td>3.17</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>0.7677</td><td>0.7650</td><td>0.7786</td><td>0.7579</td><td>0.7682</td><td>0.7711</td><td>0.7693</td><td>0.7696</td><td>0.7720</td><td>0.7603</td><td>0.7655</td><td>0.7709</td><td>0.7787</td><td>0.7681</td><td>0.7749</td></tr><tr><td>BH</td><td>0.5426</td><td>0.5718</td><td>0.5701</td><td>0.5658</td><td>0.5620</td><td>0.5556</td><td>0.5556</td><td>0.5512</td><td>0.5492</td><td>0.5519</td><td>0.5497</td><td>0.5549</td><td>0.5533</td><td>0.5543</td><td>0.5522</td></tr><tr><td>BS</td><td>0.9446</td><td>0.9865</td><td>0.9871</td><td>0.9901</td><td>0.9901</td><td>0.9916</td><td>0.9916</td><td>0.9818</td><td>0.9816</td><td>0.9924</td><td>0.9922</td><td>0.9873</td><td>0.9881</td><td>0.9845</td><td>0.9848</td></tr><tr><td>WQR</td><td>0.3662</td><td>0.3838</td><td>0.3832</td><td>0.3753</td><td>0.3729</td><td>0.3718</td><td>0.3718</td><td>0.3693</td><td>0.3693</td><td>0.3655</td><td>0.3656</td><td>0.3862</td><td>0.3845</td><td>0.3810</td><td>0.3810</td></tr><tr><td>ACT</td><td>0.8808</td><td>0.8897</td><td>0.8897</td><td>0.8832</td><td>0.8841</td><td>0.8827</td><td>0.8855</td><td>0.8827</td><td>0.8827</td><td>0.8864</td><td>0.8822</td><td>0.8925</td><td>0.8921</td><td>0.8893</td><td>0.8864</td></tr><tr><td>CD</td><td>0.8293</td><td>0.8285</td><td>0.8291</td><td>0.8287</td><td>0.8285</td><td>0.8291</td><td>0.8289</td><td>0.8294</td><td>0.8287</td><td>0.8291</td><td>0.8289</td><td>0.8295</td><td>0.8294</td><td>0.8295</td><td>0.8276</td></tr><tr><td>GC</td><td>0.7450</td><td>0.7550</td><td>0.7500</td><td>0.7650</td><td>0.7570</td><td>0.7690</td><td>0.7620</td><td>0.7660</td><td>0.7630</td><td>0.7580</td><td>0.7580</td><td>0.7640</td><td>0.7620</td><td>0.7680</td><td>0.7680</td></tr><tr><td>Mean</td><td></td><td>0.7252</td><td>0.7400</td><td>0.7411</td><td>0.7380</td><td>0.7376</td><td>0.7387</td><td>0.7378</td><td>0.7357</td><td>0.7352</td><td>0.7348</td><td>0.7346</td><td>0.7408</td><td>0.7412</td><td>0.7392</td><td>0.7393</td></tr><tr><td colspan="2">Mean Rank</td><td>12.71</td><td>8.29</td><td>5.86</td><td>8.07</td><td>8.64</td><td>6.29</td><td>7.50</td><td>9.00</td><td>10.79</td><td>9.14</td><td>10.79</td><td>4.43</td><td>4.57</td><td>6.29</td><td>7.64</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>0.8375</td><td>0.8285</td><td>0.8411</td><td>0.8188</td><td>0.8244</td><td>0.8364</td><td>0.8348</td><td>0.8430</td><td>0.8426</td><td>0.8234</td><td>0.8190</td><td>0.8311</td><td>0.8392</td><td>0.8366</td><td>0.8395</td></tr><tr><td>BH</td><td>0.5537</td><td>0.5607</td><td>0.5636</td><td>0.5693</td><td>0.5618</td><td>0.5540</td><td>0.5571</td><td>0.5478</td><td>0.5501</td><td>0.5442</td><td>0.5438</td><td>0.5619</td><td>0.5644</td><td>0.5642</td><td>0.5595</td></tr><tr><td>BS</td><td>0.9429</td><td>0.9763</td><td>0.9786</td><td>0.9751</td><td>0.9797</td><td>0.9555</td><td>0.9565</td><td>0.9449</td><td>0.9487</td><td>0.9690</td><td>0.9731</td><td>0.9737</td><td>0.9754</td><td>0.9801</td><td>0.9813</td></tr><tr><td>WQR</td><td>0.3825</td><td>0.4145</td><td>0.4182</td><td>0.3898</td><td>0.3884</td><td>0.4131</td><td>0.4035</td><td>0.3902</td><td>0.3952</td><td>0.3877</td><td>0.3834</td><td>0.4118</td><td>0.4171</td><td>0.4021</td><td>0.4042</td></tr><tr><td>ACT</td><td>0.8832</td><td>0.8794</td><td>0.8827</td><td>0.8808</td><td>0.8799</td><td>0.8822</td><td>0.8860</td><td>0.8827</td><td>0.8818</td><td>0.8879</td><td>0.8822</td><td>0.8888</td><td>0.8925</td><td>0.8902</td><td>0.8925</td></tr><tr><td>CD</td><td>0.8300</td><td>0.8283</td><td>0.8277</td><td>0.8293</td><td>0.8287</td><td>0.8296</td><td>0.8298</td><td>0.8301</td><td>0.8294</td><td>0.8293</td><td>0.8292</td><td>0.8301</td><td>0.8297</td><td>0.8303</td><td>0.8294</td></tr><tr><td>GC</td><td>0.7250</td><td>0.7650</td><td>0.7600</td><td>0.7550</td><td>0.7700</td><td>0.7490</td><td>0.7550</td><td>0.7450</td><td>0.7720</td><td>0.7450</td><td>0.7510</td><td>0.7680</td><td>0.7720</td><td>0.7760</td><td>0.7700</td></tr><tr><td>Mean</td><td></td><td>0.7364</td><td>0.7504</td><td>0.7531</td><td>0.7454</td><td>0.7476</td><td>0.7457</td><td>0.7461</td><td>0.7405</td><td>0.7457</td><td>0.7409</td><td>0.7403</td><td>0.7522</td><td>0.7558</td><td>0.7542</td><td>0.7538</td></tr><tr><td colspan="2">Mean Rank</td><td>10.43</td><td>8.86</td><td>6.29</td><td>9.50</td><td>9.14</td><td>9.07</td><td>8.14</td><td>9.00</td><td>8.50</td><td>11.21</td><td>12.21</td><td>5.71</td><td>3.57</td><td>3.57</td><td>4.79</td></tr><tr><td>Mean</td><td></td><td>0.6806</td><td>0.7091</td><td>0.7140</td><td>0.6953</td><td>0.6958</td><td>0.6979</td><td>0.6976</td><td>0.6950</td><td>0.6967</td><td>0.6976</td><td>0.6975</td><td>0.7171</td><td>0.7185</td><td>0.7187</td><td>0.7183</td></tr><tr><td colspan="2">Mean Rank</td><td>12.30</td><td>8.75</td><td>6.00</td><td>9.33</td><td>9.20</td><td>8.38</td><td>8.33</td><td>9.45</td><td>9.70</td><td>9.60</td><td>10.25</td><td>5.08</td><td>3.90</td><td>4.45</td><td>5.30</td></tr></table>

Table 9: Full experimental results of Table 3 performance comparison of PromptFE with and without dataset semantic context. For each compared version, the left and middle columns show the results without and with parameter tuning of the downstream model algorithm post AutoFE, respectively, and the right column shows the number of LLM generations. The results where the full version outperforms the blinded version are boldfaced. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td rowspan="2">Raw</td><td colspan="6">GPT-3.5</td><td colspan="6">GPT-4</td></tr><tr><td colspan="3">w/o Semantic Context</td><td colspan="3">PromptFE</td><td colspan="3">w/o Semantic Context</td><td colspan="3">PromptFE</td></tr><tr><td rowspan="7">Linear Model</td><td>AF</td><td>0.3474</td><td>0.6613</td><td>0.6602</td><td>450.0</td><td>0.6612</td><td>0.6616</td><td>339.8</td><td>0.6678</td><td>0.6672</td><td>275.0</td><td>0.6649</td><td>0.6647</td><td>371.4</td></tr><tr><td>BH</td><td>0.3776</td><td>0.4678</td><td>0.4794</td><td>438.0</td><td>0.4995</td><td>0.5025</td><td>378.6</td><td>0.4869</td><td>0.4996</td><td>295.6</td><td>0.5184</td><td>0.5289</td><td>335.4</td></tr><tr><td>BS</td><td>1.0000</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>WQR</td><td>0.2696</td><td>0.2643</td><td>0.2733</td><td>442.8</td><td>0.2722</td><td>0.2745</td><td>328.4</td><td>0.2645</td><td>0.2702</td><td>244.6</td><td>0.2713</td><td>0.2748</td><td>312.6</td></tr><tr><td>ACT</td><td>0.8505</td><td>0.8790</td><td>0.8799</td><td>442.8</td><td>0.8729</td><td>0.8794</td><td>372.2</td><td>0.8720</td><td>0.8729</td><td>238.8</td><td>0.8766</td><td>0.8762</td><td>377.4</td></tr><tr><td>CD</td><td>0.8267</td><td>0.8283</td><td>0.8283</td><td>454.8</td><td>0.8282</td><td>0.8282</td><td>342.0</td><td>0.8282</td><td>0.8289</td><td>238.2</td><td>0.8288</td><td>0.8288</td><td>250.4</td></tr><tr><td>GC</td><td>0.7100</td><td>0.7460</td><td>0.7390</td><td>432.2</td><td>0.7570</td><td>0.7460</td><td>379.0</td><td>0.7430</td><td>0.7410</td><td>231.2</td><td>0.7590</td><td>0.7420</td><td>310.6</td></tr><tr><td>Mean</td><td></td><td>0.5636</td><td>0.6411</td><td>0.6433</td><td>443.4</td><td>0.6485</td><td>0.6487</td><td>356.7</td><td>0.6437</td><td>0.6461</td><td>253.9</td><td>0.6532</td><td>0.6526</td><td>326.3</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>0.7677</td><td>0.7644</td><td>0.7743</td><td>425.2</td><td>0.7709</td><td>0.7787</td><td>393.2</td><td>0.7610</td><td>0.7690</td><td>274.2</td><td>0.7681</td><td>0.7749</td><td>314.2</td></tr><tr><td>BH</td><td>0.5426</td><td>0.5483</td><td>0.5483</td><td>479.2</td><td>0.5549</td><td>0.5533</td><td>374.4</td><td>0.5507</td><td>0.5491</td><td>238.4</td><td>0.5543</td><td>0.5522</td><td>278.6</td></tr><tr><td>BS</td><td>0.9446</td><td>0.9628</td><td>0.9628</td><td>510.0</td><td>0.9873</td><td>0.9881</td><td>386.8</td><td>0.9535</td><td>0.9543</td><td>247.4</td><td>0.9845</td><td>0.9848</td><td>255.0</td></tr><tr><td>WQR</td><td>0.3662</td><td>0.3749</td><td>0.3738</td><td>461.4</td><td>0.3862</td><td>0.3845</td><td>362.6</td><td>0.3666</td><td>0.3674</td><td>253.0</td><td>0.3810</td><td>0.3810</td><td>283.2</td></tr><tr><td>ACT</td><td>0.8808</td><td>0.8864</td><td>0.8841</td><td>475.8</td><td>0.8925</td><td>0.8921</td><td>357.6</td><td>0.8874</td><td>0.8841</td><td>222.4</td><td>0.8893</td><td>0.8864</td><td>424.0</td></tr><tr><td>CD</td><td>0.8293</td><td>0.8283</td><td>0.8282</td><td>497.0</td><td>0.8295</td><td>0.8294</td><td>349.8</td><td>0.8291</td><td>0.8286</td><td>375.2</td><td>0.8295</td><td>0.8276</td><td>304.0</td></tr><tr><td>GC</td><td>0.7450</td><td>0.7630</td><td>0.7580</td><td>459.2</td><td>0.7640</td><td>0.7620</td><td>368.2</td><td>0.7510</td><td>0.7490</td><td>229.6</td><td>0.7680</td><td>0.7680</td><td>471.8</td></tr><tr><td>Mean</td><td></td><td>0.6806</td><td>0.7326</td><td>0.7328</td><td>472.5</td><td>0.7408</td><td>0.7412</td><td>370.4</td><td>0.7285</td><td>0.7288</td><td>262.9</td><td>0.7392</td><td>0.7393</td><td>333.0</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>0.8375</td><td>0.8304</td><td>0.8356</td><td>479.6</td><td>0.8311</td><td>0.8392</td><td>380.2</td><td>0.8185</td><td>0.8266</td><td>284.6</td><td>0.8366</td><td>0.8395</td><td>360.6</td></tr><tr><td>BH</td><td>0.5537</td><td>0.5503</td><td>0.5467</td><td>490.8</td><td>0.5619</td><td>0.5644</td><td>342.0</td><td>0.5500</td><td>0.5609</td><td>238.4</td><td>0.5642</td><td>0.5595</td><td>345.6</td></tr><tr><td>BS</td><td>0.9429</td><td>0.9693</td><td>0.9691</td><td>480.2</td><td>0.9737</td><td>0.9754</td><td>380.0</td><td>0.9539</td><td>0.9536</td><td>312.6</td><td>0.9801</td><td>0.9813</td><td>236.8</td></tr><tr><td>WQR</td><td>0.3825</td><td>0.4087</td><td>0.4151</td><td>493.0</td><td>0.4118</td><td>0.4171</td><td>322.8</td><td>0.4057</td><td>0.4050</td><td>246.8</td><td>0.4021</td><td>0.4042</td><td>293.6</td></tr><tr><td>ACT</td><td>0.8832</td><td>0.8864</td><td>0.8883</td><td>513.0</td><td>0.8888</td><td>0.8925</td><td>367.4</td><td>0.8813</td><td>0.8748</td><td>229.0</td><td>0.8902</td><td>0.8925</td><td>359.6</td></tr><tr><td>CD</td><td>0.8300</td><td>0.8284</td><td>0.8292</td><td>490.8</td><td>0.8301</td><td>0.8297</td><td>352.2</td><td>0.8295</td><td>0.8299</td><td>218.6</td><td>0.8303</td><td>0.8294</td><td>371.2</td></tr><tr><td>GC</td><td>0.7250</td><td>0.7620</td><td>0.7620</td><td>482.4</td><td>0.7680</td><td>0.7720</td><td>376.6</td><td>0.7550</td><td>0.7550</td><td>225.0</td><td>0.7760</td><td>0.7700</td><td>382.2</td></tr><tr><td>Mean</td><td></td><td>0.6806</td><td>0.7479</td><td>0.7494</td><td>490.0</td><td>0.7522</td><td>0.7558</td><td>360.2</td><td>0.7420</td><td>0.7437</td><td>250.7</td><td>0.7542</td><td>0.7538</td><td>335.7</td></tr><tr><td>Mean</td><td></td><td>0.6806</td><td>0.7105</td><td>0.7118</td><td>469.9</td><td>0.7171</td><td>0.7185</td><td>362.7</td><td>0.7078</td><td>0.7092</td><td>255.9</td><td>0.7187</td><td>0.7183</td><td>331.9</td></tr></table>

Table 10: Full experimental results of Table 3 performance comparison of PromptFE with and without RPN canonicalization. For each compared version, the left and middle columns show the results without and with parameter tuning of the downstream model algorithm post AutoFE, respectively, and the right column shows the number of LLM generations. The results where the full version outperforms the reduced version are boldfaced. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td rowspan="2">Raw</td><td colspan="6">GPT-3.5</td><td colspan="6">GPT-4</td></tr><tr><td colspan="3">w/o Canonicalization</td><td colspan="3">PromptFE</td><td colspan="3">w/o Canonicalization</td><td colspan="3">PromptFE</td></tr><tr><td rowspan="7">Linear Model</td><td>AF</td><td>0.3474</td><td>0.6679</td><td>0.6688</td><td>338.6</td><td>0.6612</td><td>0.6616</td><td>339.8</td><td>0.6538</td><td>0.6529</td><td>321.2</td><td>0.6649</td><td>0.6647</td><td>371.4</td></tr><tr><td>BH</td><td>0.3776</td><td>0.5048</td><td>0.5076</td><td>351.2</td><td>0.4995</td><td>0.5025</td><td>378.6</td><td>0.4987</td><td>0.5030</td><td>310.8</td><td>0.5184</td><td>0.5289</td><td>335.4</td></tr><tr><td>BS</td><td>1.0000</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>WQR</td><td>0.2696</td><td>0.2702</td><td>0.2735</td><td>336.2</td><td>0.2722</td><td>0.2745</td><td>328.4</td><td>0.2690</td><td>0.2706</td><td>279.0</td><td>0.2713</td><td>0.2748</td><td>312.6</td></tr><tr><td>ACT</td><td>0.8505</td><td>0.8748</td><td>0.8794</td><td>366.4</td><td>0.8729</td><td>0.8794</td><td>372.2</td><td>0.8738</td><td>0.8752</td><td>298.0</td><td>0.8766</td><td>0.8762</td><td>377.4</td></tr><tr><td>CD</td><td>0.8267</td><td>0.8280</td><td>0.8290</td><td>350.4</td><td>0.8282</td><td>0.8282</td><td>342.0</td><td>0.8270</td><td>0.8271</td><td>285.4</td><td>0.8288</td><td>0.8288</td><td>250.4</td></tr><tr><td>GC</td><td>0.7100</td><td>0.7370</td><td>0.7330</td><td>352.0</td><td>0.7570</td><td>0.7460</td><td>379.0</td><td>0.7550</td><td>0.7490</td><td>447.2</td><td>0.7590</td><td>0.7420</td><td>310.6</td></tr><tr><td>Mean</td><td></td><td>0.5636</td><td>0.6471</td><td>0.6486</td><td>349.1</td><td>0.6485</td><td>0.6487</td><td>356.7</td><td>0.6462</td><td>0.6463</td><td>323.6</td><td>0.6532</td><td>0.6526</td><td>326.3</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>0.7677</td><td>0.7628</td><td>0.7762</td><td>358.0</td><td>0.7709</td><td>0.7787</td><td>393.2</td><td>0.7743</td><td>0.7843</td><td>340.2</td><td>0.7681</td><td>0.7749</td><td>314.2</td></tr><tr><td>BH</td><td>0.5426</td><td>0.5573</td><td>0.5573</td><td>364.0</td><td>0.5549</td><td>0.5533</td><td>374.4</td><td>0.5491</td><td>0.5460</td><td>322.4</td><td>0.5543</td><td>0.5522</td><td>278.6</td></tr><tr><td>BS</td><td>0.9446</td><td>0.9804</td><td>0.9807</td><td>372.2</td><td>0.9873</td><td>0.9881</td><td>386.8</td><td>0.9778</td><td>0.9777</td><td>284.4</td><td>0.9845</td><td>0.9848</td><td>255.0</td></tr><tr><td>WQR</td><td>0.3662</td><td>0.3776</td><td>0.3726</td><td>334.6</td><td>0.3862</td><td>0.3845</td><td>362.6</td><td>0.3739</td><td>0.3719</td><td>269.8</td><td>0.3810</td><td>0.3810</td><td>283.2</td></tr><tr><td>ACT</td><td>0.8808</td><td>0.8879</td><td>0.8841</td><td>353.4</td><td>0.8925</td><td>0.8921</td><td>357.6</td><td>0.8841</td><td>0.8864</td><td>327.6</td><td>0.8893</td><td>0.8864</td><td>424.0</td></tr><tr><td>CD</td><td>0.8293</td><td>0.8283</td><td>0.8285</td><td>381.6</td><td>0.8295</td><td>0.8294</td><td>349.8</td><td>0.8290</td><td>0.8287</td><td>297.2</td><td>0.8295</td><td>0.8276</td><td>304.0</td></tr><tr><td>GC</td><td>0.7450</td><td>0.7660</td><td>0.7620</td><td>342.2</td><td>0.7640</td><td>0.7620</td><td>368.2</td><td>0.7680</td><td>0.7610</td><td>368.2</td><td>0.7680</td><td>0.7680</td><td>471.8</td></tr><tr><td>Mean</td><td></td><td>0.6806</td><td>0.7372</td><td>0.7373</td><td>358.0</td><td>0.7408</td><td>0.7412</td><td>370.4</td><td>0.7366</td><td>0.7366</td><td>315.7</td><td>0.7392</td><td>0.7393</td><td>333.0</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>0.8375</td><td>0.8322</td><td>0.8365</td><td>343.6</td><td>0.8311</td><td>0.8392</td><td>380.2</td><td>0.8280</td><td>0.8350</td><td>376.0</td><td>0.8366</td><td>0.8395</td><td>360.6</td></tr><tr><td>BH</td><td>0.5537</td><td>0.5599</td><td>0.5556</td><td>339.2</td><td>0.5619</td><td>0.5644</td><td>342.0</td><td>0.5577</td><td>0.5548</td><td>315.2</td><td>0.5642</td><td>0.5595</td><td>345.6</td></tr><tr><td>BS</td><td>0.9429</td><td>0.9643</td><td>0.9664</td><td>368.8</td><td>0.9737</td><td>0.9754</td><td>380.0</td><td>0.9597</td><td>0.9609</td><td>276.2</td><td>0.9801</td><td>0.9813</td><td>236.8</td></tr><tr><td>WQR</td><td>0.3825</td><td>0.4075</td><td>0.4042</td><td>346.4</td><td>0.4118</td><td>0.4171</td><td>322.8</td><td>0.4036</td><td>0.4032</td><td>288.2</td><td>0.4021</td><td>0.4042</td><td>293.6</td></tr><tr><td>ACT</td><td>0.8832</td><td>0.8813</td><td>0.8860</td><td>342.4</td><td>0.8888</td><td>0.8925</td><td>367.4</td><td>0.8822</td><td>0.8879</td><td>313.2</td><td>0.8902</td><td>0.8925</td><td>359.6</td></tr><tr><td>CD</td><td>0.8300</td><td>0.8302</td><td>0.8291</td><td>355.8</td><td>0.8301</td><td>0.8297</td><td>352.2</td><td>0.8295</td><td>0.8291</td><td>301.6</td><td>0.8303</td><td>0.8294</td><td>371.2</td></tr><tr><td>GC</td><td>0.7250</td><td>0.7640</td><td>0.7650</td><td>346.2</td><td>0.7680</td><td>0.7720</td><td>376.6</td><td>0.7620</td><td>0.7650</td><td>428.8</td><td>0.7760</td><td>0.7700</td><td>382.2</td></tr><tr><td>Mean</td><td></td><td>0.6806</td><td>0.7485</td><td>0.7490</td><td>348.9</td><td>0.7522</td><td>0.7558</td><td>360.2</td><td>0.7461</td><td>0.7480</td><td>328.5</td><td>0.7542</td><td>0.7538</td><td>335.7</td></tr><tr><td>Mean</td><td></td><td>0.6806</td><td>0.7141</td><td>0.7148</td><td>352.2</td><td>0.7171</td><td>0.7185</td><td>362.7</td><td>0.7128</td><td>0.7135</td><td>322.5</td><td>0.7187</td><td>0.7183</td><td>331.9</td></tr></table>

Table 11: Standard deviations of Table 8 comparison of overall performance. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td rowspan="2">Raw</td><td rowspan="2" colspan="2">DIFER</td><td rowspan="2" colspan="2">OpenFE</td><td colspan="4">CAAFE</td><td colspan="2">OCTree</td><td colspan="4">PromptFE (ours)</td></tr><tr><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td><td colspan="2">GPT-4</td><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td></tr><tr><td rowspan="6">Linear Model</td><td>AF</td><td>-</td><td>0.2559</td><td>0.2012</td><td>0.0015</td><td>0.0014</td><td>0.0099</td><td>0.0102</td><td>0.0511</td><td>0.0513</td><td>0.0199</td><td>0.0199</td><td>0.0101</td><td>0.0100</td><td>0.0267</td><td>0.0268</td></tr><tr><td>BH</td><td>-</td><td>0.0092</td><td>0.0153</td><td>0.0169</td><td>0.0188</td><td>0.0196</td><td>0.0184</td><td>0.0408</td><td>0.0419</td><td>0.0502</td><td>0.0516</td><td>0.0111</td><td>0.0149</td><td>0.0254</td><td>0.0184</td></tr><tr><td>WQR</td><td>-</td><td>0.0305</td><td>0.0223</td><td>0.0058</td><td>0.0055</td><td>0.0046</td><td>0.0038</td><td>0.0060</td><td>0.0060</td><td>0.0045</td><td>0.0041</td><td>0.0135</td><td>0.0112</td><td>0.0068</td><td>0.0044</td></tr><tr><td>ACT</td><td>-</td><td>0.0179</td><td>0.0073</td><td>0.0140</td><td>0.0105</td><td>0.0035</td><td>0.0021</td><td>0.0054</td><td>0.0053</td><td>0.0148</td><td>0.0148</td><td>0.0085</td><td>0.0051</td><td>0.0040</td><td>0.0062</td></tr><tr><td>CD</td><td>-</td><td>0.0014</td><td>0.0006</td><td>0.0006</td><td>0.0002</td><td>0.0006</td><td>0.0007</td><td>0.0057</td><td>0.0051</td><td>0.0002</td><td>0.0003</td><td>0.0013</td><td>0.0007</td><td>0.0006</td><td>0.0009</td></tr><tr><td>GC</td><td>-</td><td>0.0272</td><td>0.0104</td><td>0.0097</td><td>0.0076</td><td>0.0100</td><td>0.0125</td><td>0.0134</td><td>0.0108</td><td>0.0084</td><td>0.0079</td><td>0.0120</td><td>0.0213</td><td>0.0108</td><td>0.0152</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>-</td><td>0.0054</td><td>0.0044</td><td>0.0032</td><td>0.0036</td><td>0.0032</td><td>0.0034</td><td>0.0108</td><td>0.0084</td><td>0.0084</td><td>0.0075</td><td>0.0090</td><td>0.0086</td><td>0.0059</td><td>0.0095</td></tr><tr><td>BH</td><td>-</td><td>0.0142</td><td>0.0131</td><td>0.0034</td><td>0.0068</td><td>0.0050</td><td>0.0050</td><td>0.0084</td><td>0.0113</td><td>0.0052</td><td>0.0050</td><td>0.0057</td><td>0.0077</td><td>0.0059</td><td>0.0046</td></tr><tr><td>BS</td><td>-</td><td>0.0128</td><td>0.0113</td><td>0.0003</td><td>0.0003</td><td>0.0003</td><td>0.0003</td><td>0.0208</td><td>0.0207</td><td>0.0016</td><td>0.0014</td><td>0.0088</td><td>0.0070</td><td>0.0157</td><td>0.0154</td></tr><tr><td>WQR</td><td>-</td><td>0.0108</td><td>0.0109</td><td>0.0030</td><td>0.0076</td><td>0.0022</td><td>0.0022</td><td>0.0051</td><td>0.0051</td><td>0.0039</td><td>0.0040</td><td>0.0034</td><td>0.0069</td><td>0.0022</td><td>0.0026</td></tr><tr><td>ACT</td><td>-</td><td>0.0048</td><td>0.0058</td><td>0.0037</td><td>0.0087</td><td>0.0030</td><td>0.0055</td><td>0.0020</td><td>0.0030</td><td>0.0063</td><td>0.0035</td><td>0.0055</td><td>0.0051</td><td>0.0043</td><td>0.0054</td></tr><tr><td>CD</td><td>-</td><td>0.0010</td><td>0.0011</td><td>0.0003</td><td>0.0004</td><td>0.0005</td><td>0.0004</td><td>0.0008</td><td>0.0001</td><td>0.0009</td><td>0.0006</td><td>0.0011</td><td>0.0010</td><td>0.0009</td><td>0.0017</td></tr><tr><td>GC</td><td>-</td><td>0.0184</td><td>0.0177</td><td>0.0154</td><td>0.0110</td><td>0.0082</td><td>0.0076</td><td>0.0065</td><td>0.0164</td><td>0.0160</td><td>0.0130</td><td>0.0114</td><td>0.0067</td><td>0.0097</td><td>0.0097</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>-</td><td>0.0029</td><td>0.0029</td><td>0.0058</td><td>0.0036</td><td>0.0067</td><td>0.0027</td><td>0.0072</td><td>0.0077</td><td>0.0107</td><td>0.0104</td><td>0.0129</td><td>0.0054</td><td>0.0061</td><td>0.0041</td></tr><tr><td>BH</td><td>-</td><td>0.0147</td><td>0.0260</td><td>0.0128</td><td>0.0150</td><td>0.0114</td><td>0.0111</td><td>0.0145</td><td>0.0188</td><td>0.0127</td><td>0.0170</td><td>0.0169</td><td>0.0076</td><td>0.0134</td><td>0.0073</td></tr><tr><td>BS</td><td>-</td><td>0.0092</td><td>0.0070</td><td>0.0007</td><td>0.0004</td><td>0.0159</td><td>0.0198</td><td>0.0056</td><td>0.0139</td><td>0.0162</td><td>0.0174</td><td>0.0151</td><td>0.0139</td><td>0.0033</td><td>0.0034</td></tr><tr><td>WQR</td><td>-</td><td>0.0134</td><td>0.0164</td><td>0.0072</td><td>0.0133</td><td>0.0084</td><td>0.0080</td><td>0.0116</td><td>0.0134</td><td>0.0099</td><td>0.0113</td><td>0.0123</td><td>0.0085</td><td>0.0097</td><td>0.0092</td></tr><tr><td>ACT</td><td>-</td><td>0.0048</td><td>0.0042</td><td>0.0068</td><td>0.0094</td><td>0.0061</td><td>0.0045</td><td>0.0045</td><td>0.0027</td><td>0.0017</td><td>0.0054</td><td>0.0027</td><td>0.0017</td><td>0.0050</td><td>0.0077</td></tr><tr><td>CD</td><td>-</td><td>0.0009</td><td>0.0013</td><td>0.0004</td><td>0.0010</td><td>0.0008</td><td>0.0005</td><td>0.0010</td><td>0.0007</td><td>0.0005</td><td>0.0007</td><td>0.0004</td><td>0.0004</td><td>0.0004</td><td>0.0008</td></tr><tr><td>GC</td><td>-</td><td>0.0141</td><td>0.0184</td><td>0.0184</td><td>0.0184</td><td>0.0222</td><td>0.0166</td><td>0.0079</td><td>0.0199</td><td>0.0146</td><td>0.0152</td><td>0.0076</td><td>0.0045</td><td>0.0096</td><td>0.0146</td></tr></table>

Table 12: Standard deviations of Table 9 performance comparison of PromptFE with and without dataset semantic context. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td rowspan="2">Raw</td><td colspan="6">GPT-3.5</td><td colspan="6">GPT-4</td></tr><tr><td colspan="3">w/o Semantic Context</td><td colspan="3">PromptFE</td><td colspan="3">w/o Semantic Context</td><td colspan="3">PromptFE</td></tr><tr><td rowspan="6">Linear Model</td><td>AF</td><td>-</td><td>0.0147</td><td>0.0156</td><td>36.1</td><td>0.0101</td><td>0.0100</td><td>28.8</td><td>0.0162</td><td>0.0161</td><td>25.8</td><td>0.0267</td><td>0.0268</td><td>92.3</td></tr><tr><td>BH</td><td>-</td><td>0.0444</td><td>0.0519</td><td>39.0</td><td>0.0111</td><td>0.0149</td><td>42.2</td><td>0.0161</td><td>0.0131</td><td>66.7</td><td>0.0254</td><td>0.0184</td><td>58.6</td></tr><tr><td>WQR</td><td>-</td><td>0.0133</td><td>0.0032</td><td>48.9</td><td>0.0135</td><td>0.0112</td><td>15.3</td><td>0.0128</td><td>0.0046</td><td>23.5</td><td>0.0068</td><td>0.0044</td><td>80.6</td></tr><tr><td>ACT</td><td>-</td><td>0.0088</td><td>0.0107</td><td>15.4</td><td>0.0085</td><td>0.0051</td><td>17.5</td><td>0.0056</td><td>0.0085</td><td>15.5</td><td>0.0040</td><td>0.0062</td><td>54.8</td></tr><tr><td>CD</td><td>-</td><td>0.0014</td><td>0.0003</td><td>27.6</td><td>0.0013</td><td>0.0007</td><td>13.1</td><td>0.0021</td><td>0.0011</td><td>13.2</td><td>0.0006</td><td>0.0009</td><td>14.8</td></tr><tr><td>GC</td><td>-</td><td>0.0114</td><td>0.0042</td><td>32.3</td><td>0.0120</td><td>0.0213</td><td>14.3</td><td>0.0125</td><td>0.0114</td><td>11.0</td><td>0.0108</td><td>0.0152</td><td>36.4</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>-</td><td>0.0086</td><td>0.0058</td><td>60.3</td><td>0.0090</td><td>0.0086</td><td>47.3</td><td>0.0092</td><td>0.0079</td><td>27.9</td><td>0.0059</td><td>0.0095</td><td>93.6</td></tr><tr><td>BH</td><td>-</td><td>0.0068</td><td>0.0068</td><td>45.3</td><td>0.0057</td><td>0.0077</td><td>14.5</td><td>0.0142</td><td>0.0132</td><td>24.7</td><td>0.0059</td><td>0.0046</td><td>23.0</td></tr><tr><td>BS</td><td>-</td><td>0.0186</td><td>0.0181</td><td>112.1</td><td>0.0088</td><td>0.0070</td><td>47.8</td><td>0.0103</td><td>0.0088</td><td>38.8</td><td>0.0157</td><td>0.0154</td><td>39.2</td></tr><tr><td>WQR</td><td>-</td><td>0.0078</td><td>0.0081</td><td>40.5</td><td>0.0034</td><td>0.0069</td><td>18.5</td><td>0.0092</td><td>0.0075</td><td>19.1</td><td>0.0022</td><td>0.0026</td><td>45.2</td></tr><tr><td>ACT</td><td>-</td><td>0.0099</td><td>0.0035</td><td>33.7</td><td>0.0055</td><td>0.0051</td><td>13.1</td><td>0.0100</td><td>0.0093</td><td>16.6</td><td>0.0043</td><td>0.0054</td><td>85.7</td></tr><tr><td>CD</td><td>-</td><td>0.0015</td><td>0.0008</td><td>53.3</td><td>0.0011</td><td>0.0010</td><td>14.5</td><td>0.0005</td><td>0.0008</td><td>83.4</td><td>0.0009</td><td>0.0017</td><td>56.9</td></tr><tr><td>GC</td><td>-</td><td>0.0067</td><td>0.0057</td><td>28.9</td><td>0.0114</td><td>0.0067</td><td>17.3</td><td>0.0210</td><td>0.0143</td><td>12.8</td><td>0.0097</td><td>0.0097</td><td>113.1</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>-</td><td>0.0104</td><td>0.0060</td><td>66.8</td><td>0.0129</td><td>0.0054</td><td>21.7</td><td>0.0142</td><td>0.0155</td><td>39.6</td><td>0.0061</td><td>0.0041</td><td>73.1</td></tr><tr><td>BH</td><td>-</td><td>0.0131</td><td>0.0170</td><td>60.7</td><td>0.0169</td><td>0.0076</td><td>20.7</td><td>0.0119</td><td>0.0121</td><td>25.7</td><td>0.0134</td><td>0.0073</td><td>36.1</td></tr><tr><td>BS</td><td>-</td><td>0.0152</td><td>0.0178</td><td>76.3</td><td>0.0151</td><td>0.0139</td><td>31.8</td><td>0.0048</td><td>0.0049</td><td>74.5</td><td>0.0033</td><td>0.0034</td><td>32.1</td></tr><tr><td>WQR</td><td>-</td><td>0.0151</td><td>0.0028</td><td>36.9</td><td>0.0123</td><td>0.0085</td><td>17.3</td><td>0.0195</td><td>0.0190</td><td>21.1</td><td>0.0097</td><td>0.0092</td><td>46.3</td></tr><tr><td>ACT</td><td>-</td><td>0.0021</td><td>0.0030</td><td>44.2</td><td>0.0027</td><td>0.0017</td><td>28.5</td><td>0.0042</td><td>0.0128</td><td>15.7</td><td>0.0050</td><td>0.0077</td><td>49.6</td></tr><tr><td>CD</td><td>-</td><td>0.0011</td><td>0.0011</td><td>59.4</td><td>0.0004</td><td>0.0004</td><td>15.7</td><td>0.0007</td><td>0.0010</td><td>5.6</td><td>0.0004</td><td>0.0008</td><td>85.7</td></tr><tr><td>GC</td><td>-</td><td>0.0130</td><td>0.0148</td><td>41.7</td><td>0.0076</td><td>0.0045</td><td>23.0</td><td>0.0117</td><td>0.0094</td><td>13.7</td><td>0.0096</td><td>0.0146</td><td>46.9</td></tr></table>

Table 13: Standard deviations of Table 10 performance comparison of PromptFE with and without RPN canonicalization. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td rowspan="2">Raw</td><td colspan="6">GPT-3.5</td><td colspan="6">GPT-4</td></tr><tr><td colspan="3">w/o Canonicalization</td><td colspan="3">PromptFE</td><td colspan="3">w/o Canonicalization</td><td colspan="3">PromptFE</td></tr><tr><td rowspan="6">Linear Model</td><td>AF</td><td>-</td><td>0.0117</td><td>0.0111</td><td>24.8</td><td>0.0101</td><td>0.0100</td><td>28.8</td><td>0.0112</td><td>0.0106</td><td>17.3</td><td>0.0267</td><td>0.0268</td><td>92.3</td></tr><tr><td>BH</td><td>-</td><td>0.0081</td><td>0.0138</td><td>22.1</td><td>0.0111</td><td>0.0149</td><td>42.2</td><td>0.0249</td><td>0.0294</td><td>60.6</td><td>0.0254</td><td>0.0184</td><td>58.6</td></tr><tr><td>WQR</td><td>-</td><td>0.0127</td><td>0.0083</td><td>34.7</td><td>0.0135</td><td>0.0112</td><td>15.3</td><td>0.0179</td><td>0.0070</td><td>42.0</td><td>0.0068</td><td>0.0044</td><td>80.6</td></tr><tr><td>ACT</td><td>-</td><td>0.0069</td><td>0.0084</td><td>13.9</td><td>0.0085</td><td>0.0051</td><td>17.5</td><td>0.0074</td><td>0.0069</td><td>22.3</td><td>0.0040</td><td>0.0062</td><td>54.8</td></tr><tr><td>CD</td><td>-</td><td>0.0008</td><td>0.0014</td><td>24.7</td><td>0.0013</td><td>0.0007</td><td>13.1</td><td>0.0016</td><td>0.0012</td><td>22.0</td><td>0.0006</td><td>0.0009</td><td>14.8</td></tr><tr><td>GC</td><td>-</td><td>0.0246</td><td>0.0091</td><td>14.3</td><td>0.0120</td><td>0.0213</td><td>14.3</td><td>0.0132</td><td>0.0042</td><td>99.5</td><td>0.0108</td><td>0.0152</td><td>36.4</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>-</td><td>0.0079</td><td>0.0121</td><td>27.7</td><td>0.0090</td><td>0.0086</td><td>47.3</td><td>0.0080</td><td>0.0036</td><td>32.9</td><td>0.0059</td><td>0.0095</td><td>93.6</td></tr><tr><td>BH</td><td>-</td><td>0.0054</td><td>0.0054</td><td>31.1</td><td>0.0057</td><td>0.0077</td><td>14.5</td><td>0.0095</td><td>0.0055</td><td>56.1</td><td>0.0059</td><td>0.0046</td><td>23.0</td></tr><tr><td>BS</td><td>-</td><td>0.0180</td><td>0.0174</td><td>11.9</td><td>0.0088</td><td>0.0070</td><td>47.8</td><td>0.0211</td><td>0.0210</td><td>25.6</td><td>0.0157</td><td>0.0154</td><td>39.2</td></tr><tr><td>WQR</td><td>-</td><td>0.0036</td><td>0.0051</td><td>17.5</td><td>0.0034</td><td>0.0069</td><td>18.5</td><td>0.0081</td><td>0.0099</td><td>27.7</td><td>0.0022</td><td>0.0026</td><td>45.2</td></tr><tr><td>ACT</td><td>-</td><td>0.0055</td><td>0.0094</td><td>15.2</td><td>0.0055</td><td>0.0051</td><td>13.1</td><td>0.0039</td><td>0.0056</td><td>29.2</td><td>0.0043</td><td>0.0054</td><td>85.7</td></tr><tr><td>CD</td><td>-</td><td>0.0009</td><td>0.0007</td><td>21.1</td><td>0.0011</td><td>0.0010</td><td>14.5</td><td>0.0013</td><td>0.0008</td><td>36.0</td><td>0.0009</td><td>0.0017</td><td>56.9</td></tr><tr><td>GC</td><td>-</td><td>0.0219</td><td>0.0182</td><td>27.2</td><td>0.0114</td><td>0.0067</td><td>17.3</td><td>0.0148</td><td>0.0055</td><td>19.7</td><td>0.0097</td><td>0.0097</td><td>113.1</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>-</td><td>0.0157</td><td>0.0102</td><td>21.4</td><td>0.0129</td><td>0.0054</td><td>21.7</td><td>0.0078</td><td>0.0066</td><td>62.1</td><td>0.0061</td><td>0.0041</td><td>73.1</td></tr><tr><td>BH</td><td>-</td><td>0.0125</td><td>0.0096</td><td>18.3</td><td>0.0169</td><td>0.0076</td><td>20.7</td><td>0.0098</td><td>0.0090</td><td>32.4</td><td>0.0134</td><td>0.0073</td><td>36.1</td></tr><tr><td>BS</td><td>-</td><td>0.0202</td><td>0.0190</td><td>21.5</td><td>0.0151</td><td>0.0139</td><td>31.8</td><td>0.0115</td><td>0.0115</td><td>49.9</td><td>0.0033</td><td>0.0034</td><td>32.1</td></tr><tr><td>WQR</td><td>-</td><td>0.0083</td><td>0.0181</td><td>10.3</td><td>0.0123</td><td>0.0085</td><td>17.3</td><td>0.0100</td><td>0.0092</td><td>26.2</td><td>0.0097</td><td>0.0092</td><td>46.3</td></tr><tr><td>ACT</td><td>-</td><td>0.0078</td><td>0.0065</td><td>12.6</td><td>0.0027</td><td>0.0017</td><td>28.5</td><td>0.0048</td><td>0.0029</td><td>21.3</td><td>0.0050</td><td>0.0077</td><td>49.6</td></tr><tr><td>CD</td><td>-</td><td>0.0005</td><td>0.0011</td><td>14.9</td><td>0.0004</td><td>0.0004</td><td>15.7</td><td>0.0003</td><td>0.0005</td><td>22.3</td><td>0.0004</td><td>0.0008</td><td>85.7</td></tr><tr><td>GC</td><td>-</td><td>0.0096</td><td>0.0184</td><td>14.7</td><td>0.0076</td><td>0.0045</td><td>23.0</td><td>0.0368</td><td>0.0194</td><td>126.6</td><td>0.0096</td><td>0.0146</td><td>46.9</td></tr></table>

Table 14: Standard deviations of Table 4 effect of the number of example features in the prompt. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="8">Number of Examples</td></tr><tr><td colspan="2">1</td><td colspan="2">5</td><td colspan="2">10</td><td colspan="2">20</td></tr><tr><td rowspan="3">RF</td><td>AF</td><td>0.0054</td><td>55.8</td><td>0.0035</td><td>45.0</td><td>0.0042</td><td>47.3</td><td>0.0056</td><td>24.0</td></tr><tr><td>WQR</td><td>0.0088</td><td>19.6</td><td>0.0038</td><td>11.4</td><td>0.0027</td><td>18.5</td><td>0.0096</td><td>29.6</td></tr><tr><td>CD</td><td>0.0005</td><td>46.5</td><td>0.0007</td><td>19.1</td><td>0.0004</td><td>14.5</td><td>0.0006</td><td>17.8</td></tr><tr><td rowspan="3">LGBM</td><td>AF</td><td>0.0065</td><td>103.2</td><td>0.0031</td><td>21.6</td><td>0.0044</td><td>21.7</td><td>0.0044</td><td>56.4</td></tr><tr><td>WQR</td><td>0.0048</td><td>16.9</td><td>0.0057</td><td>32.4</td><td>0.0064</td><td>17.3</td><td>0.0064</td><td>26.5</td></tr><tr><td>CD</td><td>0.0003</td><td>71.2</td><td>0.0002</td><td>39.0</td><td>0.0007</td><td>15.7</td><td>0.0005</td><td>17.5</td></tr></table>

Table 15: Standard deviations of Table 5 effect of temperature. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="6">Temperature</td></tr><tr><td colspan="2">0.5</td><td colspan="2">1</td><td colspan="2">1.5</td></tr><tr><td rowspan="2">RF</td><td>AF</td><td>0.0071</td><td>160.9</td><td>0.0042</td><td>47.3</td><td>0.0040</td><td>34.7</td></tr><tr><td>CD</td><td>0.0005</td><td>324.3</td><td>0.0004</td><td>14.5</td><td>0.0005</td><td>64.1</td></tr><tr><td rowspan="2">LGBM</td><td>AF</td><td>0.0042</td><td>523.3</td><td>0.0044</td><td>21.7</td><td>0.0022</td><td>59.8</td></tr><tr><td>CD</td><td>0.0008</td><td>174.7</td><td>0.0007</td><td>15.7</td><td>0.0005</td><td>73.0</td></tr></table>

Table 16: Percentage performance improvement of PromptFE over the baseline methods with GPT-3.5. For each compared method, the left and right columns show the results without and with parameter tuning of the downstream model algorithm post AutoFE, respectively. 

<table><tr><td>Model</td><td>Dataset</td><td colspan="2">Raw</td><td colspan="2">DIFER</td><td colspan="2">OpenFE</td><td colspan="2">CAAFE</td></tr><tr><td rowspan="6">Linear Model</td><td>AF</td><td>90.34</td><td>90.46</td><td>12.65</td><td>8.64</td><td>53.77</td><td>53.77</td><td>64.86</td><td>64.76</td></tr><tr><td>BH</td><td>32.27</td><td>33.06</td><td>-0.37</td><td>0.61</td><td>28.06</td><td>29.51</td><td>4.32</td><td>5.46</td></tr><tr><td>WQR</td><td>0.96</td><td>1.80</td><td>9.97</td><td>4.37</td><td>0.35</td><td>0.32</td><td>-0.74</td><td>-0.46</td></tr><tr><td>ACT</td><td>2.64</td><td>3.41</td><td>0.16</td><td>-0.05</td><td>0.00</td><td>0.75</td><td>2.47</td><td>3.29</td></tr><tr><td>CD</td><td>0.18</td><td>0.19</td><td>0.10</td><td>0.03</td><td>0.21</td><td>0.17</td><td>0.20</td><td>0.19</td></tr><tr><td>GC</td><td>6.62</td><td>5.07</td><td>6.02</td><td>0.54</td><td>3.42</td><td>2.47</td><td>2.99</td><td>1.77</td></tr><tr><td>Mean</td><td></td><td>22.17</td><td>22.33</td><td>4.76</td><td>2.36</td><td>14.30</td><td>14.50</td><td>12.35</td><td>12.50</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>0.42</td><td>1.44</td><td>0.78</td><td>0.02</td><td>1.72</td><td>1.37</td><td>-0.02</td><td>1.23</td></tr><tr><td>BH</td><td>2.26</td><td>1.97</td><td>-2.95</td><td>-2.95</td><td>-1.92</td><td>-1.55</td><td>-0.13</td><td>-0.41</td></tr><tr><td>BS</td><td>4.52</td><td>4.60</td><td>0.08</td><td>0.10</td><td>-0.29</td><td>-0.21</td><td>-0.43</td><td>-0.35</td></tr><tr><td>WQR</td><td>5.44</td><td>5.00</td><td>0.61</td><td>0.34</td><td>2.89</td><td>3.11</td><td>3.86</td><td>3.42</td></tr><tr><td>ACT</td><td>1.33</td><td>1.27</td><td>0.32</td><td>0.26</td><td>1.06</td><td>0.90</td><td>1.11</td><td>0.74</td></tr><tr><td>CD</td><td>0.02</td><td>0.01</td><td>0.12</td><td>0.04</td><td>0.09</td><td>0.10</td><td>0.05</td><td>0.06</td></tr><tr><td>GC</td><td>2.55</td><td>2.28</td><td>1.19</td><td>1.60</td><td>-0.13</td><td>0.66</td><td>-0.65</td><td>0.00</td></tr><tr><td>Mean</td><td></td><td>2.36</td><td>2.37</td><td>0.02</td><td>-0.08</td><td>0.49</td><td>0.63</td><td>0.54</td><td>0.67</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>-0.76</td><td>0.20</td><td>0.32</td><td>-0.23</td><td>1.51</td><td>1.80</td><td>-0.63</td><td>0.53</td></tr><tr><td>BH</td><td>1.48</td><td>1.94</td><td>0.21</td><td>0.14</td><td>-1.30</td><td>0.47</td><td>1.42</td><td>1.32</td></tr><tr><td>BS</td><td>3.27</td><td>3.45</td><td>-0.27</td><td>-0.32</td><td>-0.15</td><td>-0.43</td><td>1.91</td><td>1.98</td></tr><tr><td>WQR</td><td>7.67</td><td>9.04</td><td>-0.63</td><td>-0.27</td><td>5.66</td><td>7.40</td><td>-0.29</td><td>3.36</td></tr><tr><td>ACT</td><td>0.63</td><td>1.06</td><td>1.06</td><td>1.11</td><td>0.90</td><td>1.43</td><td>0.74</td><td>0.74</td></tr><tr><td>CD</td><td>0.02</td><td>-0.04</td><td>0.22</td><td>0.25</td><td>0.10</td><td>0.12</td><td>0.06</td><td>-0.01</td></tr><tr><td>GC</td><td>5.93</td><td>6.48</td><td>0.39</td><td>1.58</td><td>1.72</td><td>0.26</td><td>2.54</td><td>2.25</td></tr><tr><td>Mean</td><td></td><td>2.60</td><td>3.16</td><td>0.18</td><td>0.32</td><td>1.21</td><td>1.58</td><td>0.82</td><td>1.45</td></tr><tr><td>Mean</td><td></td><td>8.39</td><td>8.63</td><td>1.50</td><td>0.79</td><td>4.88</td><td>5.12</td><td>4.18</td><td>4.49</td></tr></table>

Table 17: Percentage performance improvement of PromptFE over the baseline methods with GPT-4. For each compared method, the left and right columns show the results without and with parameter tuning of the downstream model algorithm post AutoFE, respectively. 

<table><tr><td>Model</td><td>Dataset</td><td colspan="2">Raw</td><td colspan="2">DIFER</td><td colspan="2">OpenFE</td><td colspan="2">CAAFE</td><td colspan="2">OCTree</td></tr><tr><td rowspan="6">Linear Model</td><td>AF</td><td>91.40</td><td>91.34</td><td>13.27</td><td>9.14</td><td>54.62</td><td>54.48</td><td>51.94</td><td>51.82</td><td>41.52</td><td>41.47</td></tr><tr><td>BH</td><td>37.28</td><td>40.06</td><td>3.41</td><td>5.90</td><td>32.92</td><td>36.33</td><td>15.14</td><td>17.39</td><td>15.73</td><td>16.37</td></tr><tr><td>WQR</td><td>0.64</td><td>1.94</td><td>9.62</td><td>4.51</td><td>0.03</td><td>0.46</td><td>-2.25</td><td>-0.99</td><td>-2.18</td><td>-1.07</td></tr><tr><td>ACT</td><td>3.08</td><td>3.02</td><td>0.59</td><td>-0.42</td><td>0.43</td><td>0.37</td><td>2.35</td><td>2.24</td><td>0.48</td><td>0.48</td></tr><tr><td>CD</td><td>0.26</td><td>0.26</td><td>0.18</td><td>0.10</td><td>0.28</td><td>0.25</td><td>0.61</td><td>0.61</td><td>0.22</td><td>0.20</td></tr><tr><td>GC</td><td>6.90</td><td>4.51</td><td>6.30</td><td>0.00</td><td>3.69</td><td>1.92</td><td>5.27</td><td>2.91</td><td>4.26</td><td>2.34</td></tr><tr><td>Mean</td><td></td><td>23.26</td><td>23.52</td><td>5.56</td><td>3.21</td><td>15.33</td><td>15.64</td><td>12.17</td><td>12.33</td><td>10.00</td><td>9.97</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>0.05</td><td>0.93</td><td>0.41</td><td>-0.47</td><td>1.35</td><td>0.86</td><td>-0.20</td><td>0.37</td><td>1.02</td><td>1.22</td></tr><tr><td>BH</td><td>2.16</td><td>1.76</td><td>-3.05</td><td>-3.15</td><td>-2.02</td><td>-1.76</td><td>0.56</td><td>0.54</td><td>0.44</td><td>0.45</td></tr><tr><td>BS</td><td>4.23</td><td>4.25</td><td>-0.20</td><td>-0.23</td><td>-0.57</td><td>-0.54</td><td>0.28</td><td>0.32</td><td>-0.79</td><td>-0.75</td></tr><tr><td>WQR</td><td>4.03</td><td>4.03</td><td>-0.74</td><td>-0.58</td><td>1.51</td><td>2.16</td><td>3.17</td><td>3.17</td><td>4.25</td><td>4.22</td></tr><tr><td>ACT</td><td>0.95</td><td>0.64</td><td>-0.05</td><td>-0.37</td><td>0.69</td><td>0.26</td><td>0.74</td><td>0.42</td><td>0.32</td><td>0.48</td></tr><tr><td>CD</td><td>0.02</td><td>-0.20</td><td>0.13</td><td>-0.18</td><td>0.10</td><td>-0.11</td><td>0.02</td><td>-0.13</td><td>0.05</td><td>-0.15</td></tr><tr><td>GC</td><td>3.09</td><td>3.09</td><td>1.72</td><td>2.40</td><td>0.39</td><td>1.45</td><td>0.26</td><td>0.66</td><td>1.32</td><td>1.32</td></tr><tr><td>Mean</td><td></td><td>2.08</td><td>2.07</td><td>-0.26</td><td>-0.37</td><td>0.21</td><td>0.33</td><td>0.69</td><td>0.76</td><td>0.94</td><td>0.97</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>-0.11</td><td>0.24</td><td>0.98</td><td>-0.19</td><td>2.18</td><td>1.84</td><td>-0.75</td><td>-0.36</td><td>1.61</td><td>2.51</td></tr><tr><td>BH</td><td>1.90</td><td>1.04</td><td>0.63</td><td>-0.74</td><td>-0.89</td><td>-0.41</td><td>3.00</td><td>1.70</td><td>3.68</td><td>2.88</td></tr><tr><td>BS</td><td>3.94</td><td>4.08</td><td>0.38</td><td>0.28</td><td>0.51</td><td>0.17</td><td>3.72</td><td>3.44</td><td>1.14</td><td>0.84</td></tr><tr><td>WQR</td><td>5.12</td><td>5.67</td><td>-2.98</td><td>-3.35</td><td>3.16</td><td>4.08</td><td>3.04</td><td>2.28</td><td>3.71</td><td>5.42</td></tr><tr><td>ACT</td><td>0.79</td><td>1.06</td><td>1.22</td><td>1.11</td><td>1.06</td><td>1.43</td><td>0.85</td><td>1.22</td><td>0.26</td><td>1.17</td></tr><tr><td>CD</td><td>0.04</td><td>-0.07</td><td>0.24</td><td>0.21</td><td>0.12</td><td>0.08</td><td>0.03</td><td>0.00</td><td>0.12</td><td>0.02</td></tr><tr><td>GC</td><td>7.03</td><td>6.21</td><td>1.44</td><td>1.32</td><td>2.78</td><td>0.00</td><td>4.16</td><td>-0.26</td><td>4.16</td><td>2.53</td></tr><tr><td>Mean</td><td></td><td>2.67</td><td>2.60</td><td>0.27</td><td>-0.19</td><td>1.27</td><td>1.03</td><td>2.01</td><td>1.15</td><td>2.10</td><td>2.20</td></tr><tr><td>Mean</td><td></td><td>8.64</td><td>8.69</td><td>1.67</td><td>0.76</td><td>5.12</td><td>5.17</td><td>4.60</td><td>4.37</td><td>4.06</td><td>4.10</td></tr></table>

Table 18: The Nemenyi post-hoc test p-values for pairwise comparison of the methods in Table 2. Results that are significant at the $p = 0 . 0 5$ confidence level are boldfaced. 

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">Raw</td><td rowspan="2" colspan="2">DIFER</td><td rowspan="2" colspan="2">OpenFE</td><td colspan="4">CAAFE</td><td colspan="2">OCTree</td><td colspan="4">PromptFE (ours)</td></tr><tr><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td><td colspan="2">GPT-4</td><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td></tr><tr><td>Raw</td><td></td><td>1.0000</td><td>0.0010</td><td>0.0010</td><td>0.0215</td><td>0.0117</td><td>0.0010</td><td>0.0010</td><td>0.0298</td><td>0.0140</td><td>0.0010</td><td>0.0114</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td rowspan="2">DIFER</td><td></td><td>0.0010</td><td>1.0000</td><td>0.4574</td><td>0.3602</td><td>0.4794</td><td>0.9000</td><td>0.9000</td><td>0.2980</td><td>0.4462</td><td>0.8973</td><td>0.4847</td><td>0.0674</td><td>0.0010</td><td>0.0066</td><td>0.2267</td></tr><tr><td></td><td>0.0010</td><td>0.4574</td><td>1.0000</td><td>0.0010</td><td>0.0010</td><td>0.0513</td><td>0.0500</td><td>0.0010</td><td>0.0010</td><td>0.0017</td><td>0.0010</td><td>0.9000</td><td>0.7955</td><td>0.9000</td><td>0.9000</td></tr><tr><td rowspan="2">OpenFE</td><td></td><td>0.0215</td><td>0.3602</td><td>0.0010</td><td>1.0000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td></td><td>0.0117</td><td>0.4794</td><td>0.0010</td><td>0.9000</td><td>1.0000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td rowspan="3">CAAFE</td><td>GPT-3.5</td><td>0.0010</td><td>0.9000</td><td>0.0513</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.9000</td><td>0.8922</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.0025</td><td>0.0010</td><td>0.0010</td><td>0.0153</td></tr><tr><td></td><td>0.0010</td><td>0.9000</td><td>0.0500</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.8973</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.0024</td><td>0.0010</td><td>0.0010</td><td>0.0148</td></tr><tr><td>GPT-4</td><td>0.0298</td><td>0.2980</td><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>0.8922</td><td>0.8973</td><td>1.0000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td rowspan="3">OCTree</td><td>GPT-4</td><td>0.0140</td><td>0.4462</td><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.9000</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td></td><td>0.0010</td><td>0.8973</td><td>0.0017</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td>GPT-3.5</td><td>0.0114</td><td>0.4847</td><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td rowspan="4">PromptFE</td><td>GPT-4</td><td>0.0010</td><td>0.0674</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0025</td><td>0.0024</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>1.0000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td></tr><tr><td></td><td>0.0010</td><td>0.0010</td><td>0.7955</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.9000</td><td>1.0000</td><td>0.9000</td><td>0.9000</td></tr><tr><td></td><td>0.0010</td><td>0.0066</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.9000</td></tr><tr><td></td><td>0.0010</td><td>0.2267</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0153</td><td>0.0148</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>1.0000</td></tr></table>

Table 19: The Nemenyi post-hoc test p-values for pairwise comparison of the methods in Table 2 excluding linear model results. Results that are significant at the $p = 0 . 0 5$ confidence level are boldfaced. 

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">Raw</td><td rowspan="2" colspan="2">DIFER</td><td rowspan="2" colspan="2">OpenFE</td><td colspan="4">CAAFE</td><td colspan="2">OCTree</td><td colspan="4">PromptFE (ours)</td></tr><tr><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td><td colspan="2">GPT-4</td><td colspan="2">GPT-3.5</td><td colspan="2">GPT-4</td></tr><tr><td colspan="2">Raw</td><td>1.0000</td><td>0.0011</td><td>0.0010</td><td>0.5553</td><td>0.4753</td><td>0.0010</td><td>0.0014</td><td>0.4488</td><td>0.3036</td><td>0.6222</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td rowspan="2" colspan="2">DIFER</td><td>0.0011</td><td>1.0000</td><td>0.9000</td><td>0.7316</td><td>0.8107</td><td>0.9000</td><td>0.9000</td><td>0.8350</td><td>0.9000</td><td>0.6648</td><td>0.1552</td><td>0.5614</td><td>0.0404</td><td>0.2187</td><td>0.9000</td></tr><tr><td>0.0010</td><td>0.9000</td><td>1.0000</td><td>0.0235</td><td>0.0357</td><td>0.9000</td><td>0.9000</td><td>0.0404</td><td>0.0775</td><td>0.0162</td><td>0.0010</td><td>0.9000</td><td>0.8350</td><td>0.9000</td><td>0.9000</td></tr><tr><td rowspan="2" colspan="2">OpenFE</td><td>0.5553</td><td>0.7316</td><td>0.0235</td><td>1.0000</td><td>0.9000</td><td>0.6830</td><td>0.7681</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.0011</td><td>0.0010</td><td>0.0010</td><td>0.0250</td></tr><tr><td>0.4753</td><td>0.8107</td><td>0.0357</td><td>0.9000</td><td>1.0000</td><td>0.7620</td><td>0.8472</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.0019</td><td>0.0010</td><td>0.0010</td><td>0.0380</td></tr><tr><td rowspan="4">CAAFE</td><td rowspan="2">GPT-3.5</td><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>0.6830</td><td>0.7620</td><td>1.0000</td><td>0.9000</td><td>0.7864</td><td>0.9000</td><td>0.6161</td><td>0.1267</td><td>0.6100</td><td>0.0514</td><td>0.2588</td><td>0.9000</td></tr><tr><td>0.0014</td><td>0.9000</td><td>0.9000</td><td>0.7681</td><td>0.8472</td><td>0.9000</td><td>1.0000</td><td>0.8715</td><td>0.9000</td><td>0.7012</td><td>0.1783</td><td>0.5249</td><td>0.0335</td><td>0.1912</td><td>0.9000</td></tr><tr><td rowspan="2">GPT-4</td><td>0.4488</td><td>0.8350</td><td>0.0404</td><td>0.9000</td><td>0.9000</td><td>0.7864</td><td>0.8715</td><td>1.0000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.0022</td><td>0.0010</td><td>0.0010</td><td>0.0430</td></tr><tr><td>0.3036</td><td>0.9000</td><td>0.0775</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.9000</td><td>0.9000</td><td>0.0053</td><td>0.0010</td><td>0.0010</td><td>0.0819</td></tr><tr><td rowspan="2">OCTree</td><td rowspan="2">GPT-4</td><td>0.6222</td><td>0.6648</td><td>0.0162</td><td>0.9000</td><td>0.9000</td><td>0.6161</td><td>0.7012</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0174</td></tr><tr><td>0.9000</td><td>0.1552</td><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>0.1267</td><td>0.1783</td><td>0.9000</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td rowspan="4">PromptFE</td><td rowspan="2">GPT-3.5</td><td>0.0010</td><td>0.5614</td><td>0.9000</td><td>0.0011</td><td>0.0019</td><td>0.6100</td><td>0.5249</td><td>0.0022</td><td>0.0053</td><td>0.0010</td><td>0.0010</td><td>1.0000</td><td>0.9000</td><td>0.9000</td><td>0.9000</td></tr><tr><td>0.0010</td><td>0.0404</td><td>0.8350</td><td>0.0010</td><td>0.0010</td><td>0.0514</td><td>0.0335</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.9000</td><td>1.0000</td><td>0.9000</td><td>0.8229</td></tr><tr><td rowspan="2">GPT-4</td><td>0.0010</td><td>0.2187</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>0.2588</td><td>0.1912</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>1.0000</td><td>0.9000</td></tr><tr><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>0.0250</td><td>0.0380</td><td>0.9000</td><td>0.9000</td><td>0.0430</td><td>0.0819</td><td>0.0174</td><td>0.0010</td><td>0.9000</td><td>0.8229</td><td>0.9000</td><td>1.0000</td></tr></table>

Table 20: The Nemenyi post-hoc test p-values for pairwise comparison of the methods in Table 3 performance comparison of PromptFE with and without dataset semantic context. Results that are significant at the $p = 0 . 0 5$ confidence level are boldfaced. 

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">Raw</td><td colspan="4">GPT-3.5</td><td colspan="4">GPT-4</td></tr><tr><td colspan="2">Blinded</td><td colspan="2">PromptFE</td><td colspan="2">Blinded</td><td colspan="2">PromptFE</td></tr><tr><td>Raw</td><td></td><td>1.0000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0017</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td rowspan="4">GPT-3.5</td><td rowspan="2">Blinded</td><td>0.0010</td><td>1.0000</td><td>0.9000</td><td>0.0062</td><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>0.0010</td><td>0.0057</td></tr><tr><td>0.0010</td><td>0.9000</td><td>1.0000</td><td>0.1775</td><td>0.0066</td><td>0.3858</td><td>0.9000</td><td>0.0105</td><td>0.1677</td></tr><tr><td rowspan="2">PromptFE</td><td>0.0010</td><td>0.0062</td><td>0.1775</td><td>1.0000</td><td>0.9000</td><td>0.0010</td><td>0.0069</td><td>0.9000</td><td>0.9000</td></tr><tr><td>0.0010</td><td>0.0010</td><td>0.0066</td><td>0.9000</td><td>1.0000</td><td>0.0010</td><td>0.0010</td><td>0.9000</td><td>0.9000</td></tr><tr><td rowspan="4">GPT-4</td><td rowspan="2">Blinded</td><td>0.0017</td><td>0.9000</td><td>0.3858</td><td>0.0010</td><td>0.0010</td><td>1.0000</td><td>0.9000</td><td>0.0010</td><td>0.0010</td></tr><tr><td>0.0010</td><td>0.9000</td><td>0.9000</td><td>0.0069</td><td>0.0010</td><td>0.9000</td><td>1.0000</td><td>0.0010</td><td>0.0062</td></tr><tr><td rowspan="2">PromptFE</td><td>0.0010</td><td>0.0010</td><td>0.0105</td><td>0.9000</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>1.0000</td><td>0.9000</td></tr><tr><td>0.0010</td><td>0.0057</td><td>0.1677</td><td>0.9000</td><td>0.9000</td><td>0.0010</td><td>0.0062</td><td>0.9000</td><td>1.0000</td></tr></table>

Table 21: The Nemenyi post-hoc test p-values for pairwise comparison of the methods in Table 3 performance comparison of PromptFE with and without RPN canonicalization. Results that are significant at the $p = 0 . 0 5$ confidence level are boldfaced. 

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">Raw</td><td colspan="4">GPT-3.5</td><td colspan="4">GPT-4</td></tr><tr><td colspan="2">w/o</td><td colspan="2">PromptFE</td><td colspan="2">w/o</td><td colspan="2">PromptFE</td></tr><tr><td>Raw</td><td></td><td>1.0000</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td><td>0.0010</td></tr><tr><td rowspan="4">GPT-3.5</td><td rowspan="2">w/o</td><td>0.0010</td><td>1.0000</td><td>0.9000</td><td>0.2977</td><td>0.0060</td><td>0.9000</td><td>0.9000</td><td>0.0224</td><td>0.4293</td></tr><tr><td>0.0010</td><td>0.9000</td><td>1.0000</td><td>0.6618</td><td>0.0433</td><td>0.8811</td><td>0.8889</td><td>0.1230</td><td>0.7871</td></tr><tr><td rowspan="2">PromptFE</td><td>0.0010</td><td>0.2977</td><td>0.6618</td><td>1.0000</td><td>0.9000</td><td>0.0341</td><td>0.0355</td><td>0.9000</td><td>0.9000</td></tr><tr><td>0.0010</td><td>0.0060</td><td>0.0433</td><td>0.9000</td><td>1.0000</td><td>0.0010</td><td>0.0010</td><td>0.9000</td><td>0.8028</td></tr><tr><td rowspan="4">GPT-4</td><td rowspan="2">w/o</td><td>0.0010</td><td>0.9000</td><td>0.8811</td><td>0.0341</td><td>0.0010</td><td>1.0000</td><td>0.9000</td><td>0.0010</td><td>0.0635</td></tr><tr><td>0.0010</td><td>0.9000</td><td>0.8889</td><td>0.0355</td><td>0.0010</td><td>0.9000</td><td>1.0000</td><td>0.0010</td><td>0.0659</td></tr><tr><td rowspan="2">PromptFE</td><td>0.0010</td><td>0.0224</td><td>0.1230</td><td>0.9000</td><td>0.9000</td><td>0.0010</td><td>0.0010</td><td>1.0000</td><td>0.9000</td></tr><tr><td>0.0010</td><td>0.4293</td><td>0.7871</td><td>0.9000</td><td>0.8028</td><td>0.0635</td><td>0.0659</td><td>0.9000</td><td>1.0000</td></tr></table>

Table 22: Effect of the number of example features in the prompt with GPT-4. For each compared setting, the left column shows the validation score, and the right column shows the number of LLM generations. The best results are boldfaced. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="5">Number of Examples</td></tr><tr><td>1</td><td>5</td><td>10</td><td>20</td><td>30</td></tr><tr><td rowspan="3">RF</td><td>AF</td><td>0.7864</td><td>0.7922</td><td>0.7905</td><td>0.7897</td><td>0.7920</td></tr><tr><td>WQR</td><td>0.3847</td><td>0.3835</td><td>0.3839</td><td>0.3862</td><td>0.3862</td></tr><tr><td>CD</td><td>0.8219</td><td>0.8218</td><td>0.8218</td><td>0.8219</td><td>0.8222</td></tr><tr><td rowspan="3">LGBM</td><td>AF</td><td>0.8387</td><td>0.8413</td><td>0.8401</td><td>0.8433</td><td>0.8411</td></tr><tr><td>WQR</td><td>0.4216</td><td>0.4242</td><td>0.4290</td><td>0.4258</td><td>0.4267</td></tr><tr><td>CD</td><td>0.8231</td><td>0.8234</td><td>0.8227</td><td>0.8229</td><td>0.8231</td></tr><tr><td>Mean</td><td></td><td>0.6794</td><td>0.6810</td><td>0.6813</td><td>0.6816</td><td>0.6819</td></tr></table>

Table 23: Comparison of the number of selected features. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td rowspan="2">DIFER</td><td rowspan="2">OpenFE</td><td colspan="2">PromptFE Blinded</td><td colspan="2">PromptFE</td></tr><tr><td>GPT-3.5</td><td>GPT-4</td><td>GPT-3.5</td><td>GPT-4</td></tr><tr><td rowspan="6">Linear Model</td><td>AF</td><td>310</td><td>10</td><td>167</td><td>165</td><td>162</td><td>183</td></tr><tr><td>BH</td><td>156</td><td>10</td><td>104</td><td>141</td><td>144</td><td>90</td></tr><tr><td>WQR</td><td>109</td><td>10</td><td>57</td><td>80</td><td>43</td><td>55</td></tr><tr><td>ACT</td><td>113</td><td>10</td><td>84</td><td>49</td><td>85</td><td>14</td></tr><tr><td>CD</td><td>157</td><td>10</td><td>92</td><td>68</td><td>74</td><td>74</td></tr><tr><td>GC</td><td>105</td><td>10</td><td>75</td><td>97</td><td>120</td><td>51</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>387</td><td>10</td><td>39</td><td>19</td><td>15</td><td>34</td></tr><tr><td>BH</td><td>186</td><td>10</td><td>4</td><td>6</td><td>19</td><td>77</td></tr><tr><td>BS</td><td>46</td><td>10</td><td>9</td><td>7</td><td>9</td><td>65</td></tr><tr><td>WQR</td><td>63</td><td>10</td><td>9</td><td>44</td><td>39</td><td>45</td></tr><tr><td>ACT</td><td>339</td><td>10</td><td>55</td><td>35</td><td>69</td><td>61</td></tr><tr><td>CD</td><td>178</td><td>10</td><td>97</td><td>74</td><td>94</td><td>89</td></tr><tr><td>GC</td><td>92</td><td>10</td><td>68</td><td>84</td><td>31</td><td>59</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>325</td><td>10</td><td>30</td><td>55</td><td>42</td><td>24</td></tr><tr><td>BH</td><td>118</td><td>10</td><td>15</td><td>17</td><td>16</td><td>25</td></tr><tr><td>BS</td><td>287</td><td>10</td><td>119</td><td>48</td><td>68</td><td>116</td></tr><tr><td>WQR</td><td>454</td><td>10</td><td>64</td><td>29</td><td>129</td><td>128</td></tr><tr><td>ACT</td><td>132</td><td>10</td><td>54</td><td>46</td><td>16</td><td>51</td></tr><tr><td>CD</td><td>409</td><td>10</td><td>68</td><td>53</td><td>12</td><td>50</td></tr><tr><td>GC</td><td>501</td><td>10</td><td>61</td><td>86</td><td>16</td><td>35</td></tr><tr><td>Mean</td><td></td><td>223</td><td>10</td><td>64</td><td>60</td><td>60</td><td>66</td></tr></table>

Table 24: Comparison of the number of evaluated features during feature search. 

<table><tr><td>Model</td><td>Dataset</td><td>DIFER</td><td>OpenFE</td><td>PromptFE</td></tr><tr><td rowspan="6">Linear Model</td><td>AF</td><td>2083</td><td>224</td><td>200</td></tr><tr><td>BH</td><td>2081</td><td>1167</td><td>200</td></tr><tr><td>WQR</td><td>2083</td><td>929</td><td>200</td></tr><tr><td>ACT</td><td>2077</td><td>4310</td><td>200</td></tr><tr><td>CD</td><td>2088</td><td>3385</td><td>200</td></tr><tr><td>GC</td><td>2076</td><td>4169</td><td>200</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>2085</td><td>224</td><td>200</td></tr><tr><td>BH</td><td>2079</td><td>1051</td><td>200</td></tr><tr><td>BS</td><td>2082</td><td>310</td><td>200</td></tr><tr><td>WQR</td><td>2085</td><td>929</td><td>200</td></tr><tr><td>ACT</td><td>2079</td><td>1636</td><td>200</td></tr><tr><td>CD</td><td>2086</td><td>1801</td><td>200</td></tr><tr><td>GC</td><td>2078</td><td>2139</td><td>200</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>2084</td><td>224</td><td>200</td></tr><tr><td>BH</td><td>2080</td><td>1051</td><td>200</td></tr><tr><td>BS</td><td>2083</td><td>310</td><td>200</td></tr><tr><td>WQR</td><td>2084</td><td>929</td><td>200</td></tr><tr><td>ACT</td><td>2079</td><td>1636</td><td>200</td></tr><tr><td>CD</td><td>2087</td><td>1801</td><td>200</td></tr><tr><td>GC</td><td>2078</td><td>2139</td><td>200</td></tr><tr><td colspan="2">Mean</td><td>2082</td><td>1518</td><td>200</td></tr></table>

Table 25: Comparison of computation time, in minutes. 

<table><tr><td>Model</td><td>Dataset</td><td>DIFER</td><td>OpenFE</td><td>CAAFE</td><td>PromptFE</td></tr><tr><td rowspan="6">Linear Model</td><td>AF</td><td>34</td><td>1</td><td>2</td><td>43</td></tr><tr><td>BH</td><td>42</td><td>1</td><td>2</td><td>41</td></tr><tr><td>WQR</td><td>35</td><td>1</td><td>2</td><td>42</td></tr><tr><td>ACT</td><td>45</td><td>1</td><td>2</td><td>44</td></tr><tr><td>CD</td><td>434</td><td>2</td><td>4</td><td>58</td></tr><tr><td>GC</td><td>30</td><td>1</td><td>2</td><td>44</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>179</td><td>1</td><td>5</td><td>63</td></tr><tr><td>BH</td><td>90</td><td>1</td><td>6</td><td>52</td></tr><tr><td>BS</td><td>99</td><td>1</td><td>5</td><td>51</td></tr><tr><td>WQR</td><td>299</td><td>1</td><td>10</td><td>63</td></tr><tr><td>ACT</td><td>79</td><td>1</td><td>4</td><td>45</td></tr><tr><td>CD</td><td>572</td><td>2</td><td>15</td><td>94</td></tr><tr><td>GC</td><td>61</td><td>1</td><td>4</td><td>45</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>302</td><td>1</td><td>6</td><td>63</td></tr><tr><td>BH</td><td>63</td><td>1</td><td>4</td><td>45</td></tr><tr><td>BS</td><td>75</td><td>1</td><td>3</td><td>45</td></tr><tr><td>WQR</td><td>362</td><td>1</td><td>6</td><td>59</td></tr><tr><td>ACT</td><td>37</td><td>1</td><td>2</td><td>43</td></tr><tr><td>CD</td><td>103</td><td>2</td><td>3</td><td>47</td></tr><tr><td>GC</td><td>49</td><td>1</td><td>3</td><td>43</td></tr><tr><td colspan="2">Mean</td><td>149</td><td>1</td><td>4</td><td>52</td></tr></table>

Table 26: Computation time of different components of PromptFE, in minutes. 

<table><tr><td>Model</td><td>Dataset</td><td>LLM Generation</td><td>Feature Evaluation</td><td>Feature Selection</td></tr><tr><td rowspan="6">Linear Model</td><td>AF</td><td>17</td><td>23</td><td>3</td></tr><tr><td>BH</td><td>18</td><td>20</td><td>3</td></tr><tr><td>WQR</td><td>19</td><td>20</td><td>3</td></tr><tr><td>ACT</td><td>19</td><td>21</td><td>4</td></tr><tr><td>CD</td><td>17</td><td>25</td><td>16</td></tr><tr><td>GC</td><td>17</td><td>23</td><td>4</td></tr><tr><td rowspan="7">Random Forests</td><td>AF</td><td>15</td><td>25</td><td>23</td></tr><tr><td>BH</td><td>19</td><td>24</td><td>9</td></tr><tr><td>BS</td><td>15</td><td>25</td><td>11</td></tr><tr><td>WQR</td><td>13</td><td>24</td><td>26</td></tr><tr><td>ACT</td><td>14</td><td>22</td><td>9</td></tr><tr><td>CD</td><td>12</td><td>26</td><td>56</td></tr><tr><td>GC</td><td>15</td><td>22</td><td>8</td></tr><tr><td rowspan="7">Light-GBM</td><td>AF</td><td>17</td><td>21</td><td>25</td></tr><tr><td>BH</td><td>20</td><td>20</td><td>5</td></tr><tr><td>BS</td><td>17</td><td>22</td><td>6</td></tr><tr><td>WQR</td><td>16</td><td>21</td><td>22</td></tr><tr><td>ACT</td><td>19</td><td>20</td><td>4</td></tr><tr><td>CD</td><td>17</td><td>22</td><td>8</td></tr><tr><td>GC</td><td>17</td><td>20</td><td>6</td></tr><tr><td colspan="2">Mean</td><td>17</td><td>22</td><td>13</td></tr></table>