# Reasoning Planning for Language Models

Bao Nguyen1 Hieu Trung Nguyen1 Ruifeng She2 Xiaojin Fu2 Viet Anh Nguyen1

1 The Chinese University of Hong Kong

2 Huawei Noah’s Ark Lab

nbnguyen@se.cuhk.edu.hk, thnguyen@se.cuhk.edu.hk

she.ruifeng@huawei.com, fuxiaojin32@hotmail.com, nguyen@se.cuhk.edu.hk

# Abstract

Selecting an appropriate reasoning method for a given query remains a key challenge in language model generation. Existing approaches typically generate multiple candidate responses and use an aggregation strategy to select the output answer, often assuming that more candidate answers yield higher accuracy. We revisit this assumption through a rigorous theoretical analysis, deriving accuracy bounds for standard aggregation methods under fixed generation distributions and candidate sizes. Building on these insights, we introduce EPIC, an Ensemble PlannIng with Contrastive learning framework to learn a shared representation space that captures both model reasoning abilities and query-method compatibility. EPIC incorporates our probability bounds as a regularizer in a utility-driven optimization that balances accuracy and computational cost. Experiments on diverse mathematical reasoning tasks show that EPIC consistently selects optimal reasoning methods, improving accuracy while reducing computational overhead. Our code can be found at https://github.com/nguyenngocbaocmt02/EPIC.

# 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable abilities to understand and reason in human natural language. These advancements have transformed applications, including travel planning [Xie et al., 2024a], AI teaching platforms [Jin et al., 2024], and human population simulations [Park et al., 2023, Bui et al., 2025]. However, even with a pre-trained LLM, the computational expense of serving LLM-powered systems remains a significant bottleneck due to the massive scale of the models [Lin et al., 2024], the quadratic complexity of the attention mechanism [Dao et al., 2022], and the token-by-token nature of auto-regressive generation [Zhang et al., 2025]. This high computational cost significantly hinders the broader application of LLMs in practical scenarios, particularly in resource-constrained environments such as edge devices, real-time applications, and small-scale businesses.

This challenge becomes even more pronounced in tasks that require advanced reasoning, such as automated theorem proving [Wu et al., 2022], mathematical problem solving [Trinh et al., 2024], code generation [Jiang et al., 2024, Li et al., 2025a], or heuristic discovery [Romera-Paredes et al., 2024]. LLMs often fail to produce accurate responses in these scenarios in a single pass. Instead, they rely on iterative generation strategies combined with aggregation or search techniques, such as best-of-N sampling [Stiennon et al., 2020] or Monte Carlo Tree Search [Xie et al., 2024b], to refine and select the most appropriate response. Throughout this paper, we refer to these iterative strategies as Reasoning Methods.

A key limitation of current approaches lies in their static application of reasoning methods, where the same technique is applied uniformly across all user queries. However, not all reasoning methods are equally effective or efficient for every query. This observation leads to our central research question: Could we select the most suitable reasoning method for a given user query to balance the trade-off between accuracy and efficiency before generating the answer?

As a starting point, we consider a universe of methods, denoted by M. Each technique in M is formally characterized by a tuple (LM, ReStrat, Agg, Config, N), where LM denotes the base language model, ReStrat denotes the reasoning strategy (e.g. Monte Carlo Tree Search, Beam Search, Best-of-N), Config is a collection of relevant configuration parameters (e.g., temperature of a sampling-based decoding strategy), Agg denotes an aggregation technique (e.g., majority voting or score-based voting), and N is the number of candidate answers for aggregation. Importantly, this formulation is broad enough to subsume a wide variety of test-time compute methods [Snell et al., 2025], ranging from simple prompting techniques [Wei et al., 2022, Yao et al., 2023, Brown et al., 2020] and standard decoding strategies [Xie et al., 2024b, Wang et al., 2022] to more specialized intervention-based approaches [Li et al., 2023, Nguyen et al., 2025a,b]. However, in this work, we focus on a representative subset of methods rather than exhaustively covering the entire space.

Contributions. We introduce EPIC, an Ensemble PlannIng with Contrastive learning framework that recommends matching an input question and an appropriate reasoning method in the universe of methods M. EPIC learns jointly the embedding of each reasoning method and a neural mapping from the input question to the embedding space. Two main components guide the learning process:

• a contrastive loss, which pulls the question embedding towards the reasoning method with the highest utility for that question. The utility value is composed of a weighted combination of the accuracy and the inference cost, measured by the number of tokens generated. The user controls the accuracy-cost trade-off through a scalar parameter, balancing the preferences across different conflicting deployment criteria.   
• a regularizer term, which exploits the commonality among methods that share four components (LM, ReStrat, Config, Agg), but differ only by the number of candidate answers N. This regularizer aims to improve the sample efficiency of the training procedure by grounding these methods relatively on the scale of N .

At inference time, EPIC maps the test-time input question to the embedding space and selects the reasoning method with the highest similarity (or scores) for answer generation. Extensive experiments on the MATH dataset demonstrate EPIC’s advantage: compared to individual reasoning models in the universe of methods, EPIC can reduce the number of tokens (or cost) by 75% while maintaining the same level of accuracy.

Our paper unfolds as follows: Section 2 discusses related work on LLM reasoning. Section 3 studies the probabilistic bounds of common aggregation methods. Section 4 delineates our EPIC framework for matching reasoning methods with input questions, and Section 5 presents the extensive numerical results of the mathematical reasoning task.

# 2 Related Work

We review advances in LLM reasoning algorithms and inference-time scaling, highlighting their emerging impact on output quality and computational efficiency.

Reasoning algorithms and inference-time scaling. A naive reasoning process may not generate the correct solutions for complex reasoning tasks. To identify and choose the correct solution within the distribution, Self-Consistency (SC) samples multiple outputs from the LLM and selects the final response by majority voting [Wang et al., 2022]. Another similar approach is best-of-N sampling, which uses a reward model or function to choose the answer with the highest reward [Stiennon et al., 2020]. Both methods enhance the quality of the output, but increase the computational cost by a factor of sampling times. To explore potential reasoning paths, tree-search-based methods are proposed, such as Tree-of-Thought [Yao et al., 2024], Monte Carlo Tree Search (MCTS) [Wan et al., 2024, Zhang et al., 2024, Guan et al., 2025], Forest-of-Thought [Bi et al., 2025]. Damani et al. [2025] indicates that searching over a tree structure is more effective in discovering a correct solution than simply sampling responses in parallel for more complex tasks.

Despite applying different reasoning methods to problems with various levels of complexity, inferencetime scaling on these methods also significantly improves the output quality. Beeching et al. [2024] demonstrates that the accuracy on the MATH-500 benchmark improves as the amount of test-time computation (number of generations per problem) increases for algorithms such as best-of-N, beam search, and diverse verifier tree search (DVTS). Guan et al. [2025] conduct extensive MCTS rollouts and achieve an average accuracy of 53.3% on 15 questions of AIME24 benchmark.

Cost-effective reasoning. Although inference-time scaling significantly enhances LLM’s reasoning capabilities, this approach incurs substantial computational overhead and often leads to inefficient use of computational resources. Recent work finds that performance gains from various inference-time scaling strategies exhibit significant variability across different levels of prompt difficulty [Snell et al., 2025]. Drawing from this evidence, they effectively allocate inference-time compute according to question difficulty, with four times less computation than the best-of-N baseline. However, the method incurs considerable computational costs to assess question difficulty. Damani et al. [2025] train lightweight probes built upon LLM’s hidden representations to quickly predict if allocating more computation to a question will improve the response quality. To efficiently scale best-of-N sampling, Manvi et al. [2024] introduces a highly cost-effective self-evaluation paradigm that does not rely on an external reward model, incurring costs only from generating a single token.

Whereas most studies focus on effectively and efficiently scaling a particular reasoning algorithm, we focus on pairing suitable reasoning methods with various questions, considering both accuracy and cost. We conduct our study based on OpenR [Wang et al., 2024a], an open-source framework for LLM reasoning that integrates multiple strategies, including greedy decoding, best-of-N, beam search, and MCTS.

# 3 Probabilistic Analysis of Aggregation Accuracy

We observe that many methods in the universe M could share common features: they could use the same base language model, reasoning strategy, configuration parameters, and aggregation methods, and they could differ by only the amount of test-time compute, or how many samples N they need to generate before aggregation. To exploit this information, we first need to understand how different sample sizes N affect the quality of the output. We analyze the probabilistic performance of an aggregation method for a specific question q as the number of samples N varies. All probability quantities in this section are conditioned on $q ,$ but this condition is omitted to avoid clutter. Let $\tilde { Y }$ be a random variable representing the final answer extracted from a sampled solution to a question q generated by a model. Importantly, $\tilde { Y }$ refers specifically to the final answer, not the reasoning process or steps leading to it. In practice, the model is trained to enclose $\tilde { Y }$ in a LaTeX box to make extraction easier. We suppose that the support set of $\tilde { Y }$ is finite: $\mathcal { Y } = \{ y _ { 1 } , y _ { 2 } , \dots , y _ { K } \}$ . The stochastic generation process of a model specified by the tuple (LM, ReStrat, Config) produces a probability distribution over $\mathcal { V }$ :

$$
\operatorname * {P r} (\tilde {Y} = y _ {k}) = p _ {k}, \quad \text { where } \quad \sum_ {k = 1} ^ {K} p _ {k} = 1 \text {   and   } p _ {k} \geq 0 \quad \forall k.
$$

Without any loss of generality, we denote $y _ { 1 } \in \mathcal { V }$ as the only correct answer to the question q. After sampling $N$ independent samples following the above distribution, an aggregation method Agg is applied to obtain the output answer. We focus on characterizing the probability that the output answer is $y _ { 1 }$ , which means that the output answer is a correct solution to question $q .$ .

# 3.1 Majority Voting

Given N samples generated by the language model, majority voting counts the frequency of each unique answer among N candidate answers. Then it outputs the answer with the highest count as the output answer. We refer to this aggregation method as Majority Vote. We have the following result.

Theorem 3.1 (Majority voting). $I f p _ { 1 } > p _ { k }$ for all $k = 2 , \ldots , K$ , then

$$
\operatorname * {P r} (M a j o r i t y \_ V o t e p i c k s y _ {1}) \geq 1 - \sum_ {k = 2} ^ {K} e ^ {- N \left(\sqrt {p _ {1}} - \sqrt {p _ {k}}\right) ^ {2}}. \tag {1a}
$$

$f p _ { 1 } < p _ { k } f o r$ some $k = 2 , \ldots , K$ , then

$$
\operatorname * {P r} (M a j o r i t y \_ V o t e p i c k s y _ {1}) \leq e ^ {- N \left(\sqrt {p _ {k}} - \sqrt {p _ {1}}\right) ^ {2}}. \tag {1b}
$$

Note that the bound (1a) approaches 1 as $N \to \infty$ , which implies perfect accuracy. In contrast, the bound (1b) approaches 0 as $N \to \infty$ , implying a complete failure.

# 3.2 Aggregation using Summation of Scores

Given N samples generated by a model, we first pass them through a reward model to obtain a reward score for each sample. Two popular types of reward models are Outcome Reward Models (ORM) [Cobbe et al., 2021, Yu et al., 2023], which provide a single scalar reward for each complete solution trajectory, and Process Reward Models (PRM) [Luo et al., 2024, Lightman et al., 2023a], which provide step-by-step feedback and aggregate it, typically by summing or taking the minimum, to obtain a final score for the sample. While PRM is used throughout this work, our method is flexible and can be applied with any reward models.

For each unique answer, we sum the PRM scores of all samples that generate that answer. The final outcome is selected as the answer with the highest total (summed) reward score across all samples [Wang et al., 2024a, Li et al., 2022]. We suppose that PRM returns a score for answer yk following a Gaussian distribution $\mathcal { N } ( \mu _ { k } , \sigma _ { k } ^ { 2 } )$ for all k. We call this aggregation method PRM Vote. We have the following result.

Theorem 3.2 (Voting with score sum). $H ^ { } p _ { 1 } \mu _ { 1 } > p _ { k } \mu _ { k }$ for all $k = 2 , \ldots , K ,$ , then

$$
\operatorname * {P r} (P R M \_ V o t e p i c k s y _ {1}) \geq 1 - \sum_ {k = 2} ^ {K} \inf _ {t _ {k} > 0} \exp \left(N p _ {1} \left(e ^ {- t _ {k} \mu_ {1} + \frac {1}{2} t _ {k} ^ {2} \sigma_ {1} ^ {2}} - 1\right) + N p _ {k} \left(e ^ {t _ {k} \mu_ {k} + \frac {1}{2} t _ {k} ^ {2} \sigma_ {k} ^ {2}} - 1\right)\right). \tag {2a}
$$

$H p _ { 1 } \mu _ { 1 } < p _ { k } \mu _ { k }$ for some $k = 2 , \ldots , K$ , then

$$
\operatorname * {P r} (P R M \_ V o t e p i c k s y _ {1}) \leq \inf _ {t > 0} \exp \left(N p _ {k} \left(e ^ {- t \mu_ {k} + \frac {1}{2} t ^ {2} \sigma_ {k} ^ {2}} - 1\right) + N p _ {1} \left(e ^ {t \mu_ {1} + \frac {1}{2} t ^ {2} \sigma_ {1} ^ {2}} - 1\right)\right). \tag {2b}
$$

All infimum problems in (2) are convex optimization problems. While no analytical expression for the optimal value t is available, we could tractably find $t _ { k }$ for each term using Newton’s method. Moreover, we could observe a similar conclusion as N tends to infinity: the bound (2a) approaches 1 while the bound (2b) approaches 0.

# 3.3 Aggregation using Maximum of Scores

This aggregation method follows the same setup as in Section 3.2: given N samples, we use the PRM to assign a reward score to each sample. For each unique answer, we take the maximum PRM score among all samples that produce that answer. The final prediction is the answer with the highest such maximum. We suppose that PRM returns a score for answer yk following a Gaussian distribution $\mathcal { N } ( \mu _ { k } , \sigma _ { k } ^ { 2 } )$ ) for all k. We call this aggregation method PRM Max. We have the following result.

Theorem 3.3 (Voting with score maximum). Let

$$
\Phi_ {k} (t) := \Phi \left(\frac {t - \mu_ {k}}{\sigma_ {k}}\right), \quad k = 1, \dots , K,
$$

where Φ is the cumulative distribution function of the standard normal distribution.

$I f \sigma _ { 1 } > \sigma _ { k }$ for all $k = 2 , \ldots , K$ , then

$$
\operatorname * {P r} (P R M \_ M a x p i c k s y _ {1}) \geq 1 - \sum_ {k = 2} ^ {K} \inf _ {t \in \mathbb {R}} \left\{(1 - p _ {1} [ 1 - \Phi_ {1} (t) ]) ^ {N} + 1 - (1 - p _ {k} [ 1 - \Phi_ {k} (t) ]) ^ {N} \right\}. \tag {3a}
$$

If $\sigma _ { k } > \sigma _ { 1 }$ for some k = 2, . . . , K , then

$$
\operatorname * {P r} (P R M \_ M a x p i c k s y _ {1}) \leq \inf _ {t \in \mathbb {R}} \left\{\left(1 - p _ {k} [ 1 - \Phi_ {k} (t) ]\right) ^ {N} + 1 - \left(1 - p _ {1} [ 1 - \Phi_ {1} (t) ]\right) ^ {N} \right\}. \tag {3b}
$$

All infimum problems in (3) are one-dimensional and can be efficiently solved using standard numerical methods. Moreover, we observe similar asymptotic behavior as N increases: if $\sigma _ { 1 } > \sigma _ { k }$ for all $k = 2 , \ldots , K$ , the bound in (3a) approaches 1 as $N \to \infty$ , while if $\sigma _ { k } > \sigma _ { 1 }$ for some $k ,$ the bound in (3b) approaches 0 as $N \to \infty$ .

# 4 Ensemble Planning with Contrastive Learning

Given a universe of methods $\mathcal { M } = \{ 1 , \dots , M \}$ consisting of M reasoning methods in total, EPIC aims to create an ensemble model on M that assigns to any input question x from the test environment an appropriate method $i \in \mathcal { M }$ that could deliver a desirable accuracy-cost trade-off. We first discuss our modeling of the accuracy-cost trade-off in Section 4.1, then we describe the training phase and inference phase in Sections 4.2 and 4.3. We conclude this section by discussing our design choices.

# 4.1 Accuracy-Cost and Utility

We possess a training dataset of n question-answer pairs denoted as $\mathcal { D } = \{ x _ { j } , y _ { j } \} _ { i = 1 } ^ { n }$ , where $x _ { j }$ is a question statement, and $y _ { j }$ is the corresponding true answer. In the training phase, we deploy a reasoning model $\Phi _ { i } , i \in \mathcal { M }$ , to each question $x _ { j }$ . The generated solution is $\Phi _ { i } ( x _ { j } )$ . We record whether $\Phi _ { i } ( x _ { j } )$ is accurate by comparing it to the ground-truth answer $y _ { j }$ , and obtain the accuracy signal

$$
a _ {i, j} = \text { Accuracy } (\Phi_ {i} (x _ {j}), y _ {j}) \in [ 0, 1 ]. \tag {4}
$$

If $\Phi _ { i }$ is a deterministic method, then (4) is a simple binary indicator Accuracy $\left( \Phi _ { i } ( x _ { j } ) , y _ { j } \right) =$ $\mathbb { 1 } ( \Phi _ { i } ( x _ { j } ) = y _ { j } )$ . When $\Phi _ { i }$ is a stochastic method, then we average the accuracy over five seed numbers to get a percentage accuracy. The value $a _ { i , j }$ indicates whether method i succeeds in answering question $j .$ Moreover, we also record how many tokens the method $i \in \mathcal { M }$ costs to generate the answer. This token count is denoted by $\tilde { c } _ { i , j } > 0$ . Because $a _ { i , j } \in [ 0 , 1 ]$ , we normalize the token count by passing $\tilde { c } _ { i , j }$ through a non-decreasing function $\phi : \mathbb { R } _ { + } \to \mathbb { R } _ { + }$ , then dividing by the maximum transformed cost to ensure that the cost $\bar { c _ { i , j } } \in [ 0 , 1 ]$ has the same scale with $a _ { i , j } .$

$$
c _ {i, j} = \frac {\phi (\tilde {c} _ {i , j})}{\max _ {i ^ {\prime} \in \mathcal {M}} (\phi (\tilde {c} _ {i ^ {\prime} , j}))}.
$$

To balance cost and success rate, we establish the utility function that is the convex combination of accuracy $a _ { i , j }$ and normalized cost $c _ { i , j }$ as

$$
u (a _ {i, j}, c _ {i, j}) = \lambda a _ {i, j} + (1 - \lambda) (1 - c _ {i, j}), \tag {5}
$$

where $\lambda \in [ 0 , 1 ]$ is a trade-off parameter, and the utility admits a value between 0 and 1. If $\lambda = 0$ then $\begin{array} { r } { u ( a _ { i , j } , c _ { i , j } ) = 1 - c _ { i , j } , } \end{array}$ , which implies that the utility depends only on the generation cost. In this way, we tend to favor the cheapest reasoning method, regardless of how effective it is at generating accurate answers. On the other end of the spectrum, when $\lambda = 1 , u ( a _ { i , j } , c _ { i , j } ) = a _ { i , j } ,$ implying that the utility is entirely derived from the accuracy. In this way, we tend to favor the most powerful reasoning method, regardless of its cost. To simplify the notation, we omit the parameter λ, and use the shorthand $u _ { i , j } = u ( a _ { i , j } , c _ { i , j } )$ .

The product of the data preparation process is a processed dataset $\{ x _ { j } , ( u _ { i , j } ) _ { i \in \mathcal { M } } \} _ { j = 1 } ^ { n }$ containing the training question and the corresponding utility of each reasoning method for that question. This dataset will be used in the subsequent contrastive learning process.

# 4.2 Contrastive Representation Learning with Probability Regularization

We now describe the core component of our framework that matches the input question with the appropriate reasoning method. We represent each question $x _ { j }$ in the training dataset by its features $f _ { j } \in \mathbb { R } ^ { D }$ . A lightweight neural network $g _ { \theta } : \mathbb { R } ^ { D }  \mathbb { R } ^ { d }$ maps each question feature vector $f _ { j }$ to produce a dense embedding $g _ { \theta } ( f _ { j } )$ in a d-dimensional vector space. EPIC aims for an information compression with $d \ll D$ . Moreover, each reasoning method $i \in \mathcal { M }$ is assigned a trainable embedding vector $v _ { i } \in \mathbb { R } ^ { d }$ , which is the same dimension as the question embeddings. EPIC uses a simple multi-layer perceptron for θ.

We now train the question embedding network parameter θ and the method embedding vectors vi jointly. One component in the training loss is the popular contrastive loss function, InfoNCE loss [Oord et al., 2018]. We identify a positive method for each question $x _ { j }$ , denoted as $m _ { + } ( x _ { j } )$ . Given the utility values defined in (5), we can identify the method with the highest utility for question $x _ { j } { \mathrm { : } }$

$$
m _ {+} (x _ {j}) = \arg \max _ {i \in \mathcal {M}} u (a _ {i, j}, c _ {i, j}),
$$

![](images/395f9e7582bc98819760e92f367b28b553164b2fd2b1d2d2079c8fc171887943.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Find the constant term in the expansion of (10x³ - 1/(2x²))⁵"] --> B["Reasoning Method Embeddings"]
    B --> C["Evaluate log₂64 = (x·x) = gθ(·)"]
    C --> D["Four distinct circles are drawn in a plane. What is the maximum number of points where at least two of the circles intersect?"]
    D --> E["Reasoning method 1: v₁ loss"]
    D --> F["Reasoning method 2: v₂ loss"]
    D --> G["...: v_M loss"]
    D --> H["Reasoning method M: gθ(·)"]
    H --> I["Final result"]
```
</details>

Figure 1: Our method employs the regularized representation learning loss (8) to learn both the reasoning method representation vectors, denoted as $v _ { 1 } , \ldots , v _ { M }$ , and the question embedding network parameters θ. During inference, we route suitable math questions to the appropriate reasoning method by computing the similarity between the input questions and the learned method representations. Color codes on problem difficulty levels are provided for illustration purposes only.

This leads to the contrastive loss component:

$$
\ell_ {\text { contrastive }} (\theta , v _ {1}, \dots , v _ {M}) = \frac {1}{n} \sum_ {j = 1} ^ {n} - \log \left(\frac {\exp (s (g _ {\theta} (f _ {j}) , v _ {m + (x _ {j})}))}{\sum_ {i \in \mathcal {M}} \exp (s (g _ {\theta} (f _ {j}) , v _ {i}))}\right). \tag {6}
$$

Above, s is a similarity score function that measures the similarity of a question embedding $g _ { \boldsymbol { \theta } } ( f _ { j } )$ with the method embedding $v _ { i }$ . Standard choices for s are the dot product similarity measure or the negative 2-norm. The contrastive component (6) aims to pull $g _ { \theta } ( f _ { j } )$ close to the positive method $v _ { m _ { + } ( x _ { j } ) }$ , and push $g _ { \boldsymbol { \theta } } ( f _ { j } )$ far away from the negative methods $i \nearrow _ { m _ { + } ( x _ { j } ) }$ . The loss in (6) is the categorical cross-entropy loss of classifying the positive method, with the fraction inside the logarithm being the model’s prediction.

The second component of the loss function is a regularization term: Two methods that share the same tuple (base model, reasoning strategy, aggregation technique and configuration) but differ only by the compute budget N should conform to a relative performance metric because they both inherit the same stochastic generator. We postulate the following regularization term:

$$
\ell_ {\text { reg }} (\theta , v _ {1}, \dots , v _ {M}) = \frac {1}{n} \sum_ {j = 1} ^ {n} \sum_ {\substack {(i, i ^ {\prime}) \in \mathcal {M} \\ (i, i ^ {\prime}) \text {differ only by} N}} \left(\frac {s \left(g _ {\theta} \left(f _ {j}\right) , v _ {i}\right)}{s \left(g _ {\theta} \left(f _ {j}\right) , v _ {i ^ {\prime}}\right)} - \frac {\operatorname{target} _ {i} ^ {j}}{\operatorname{target} _ {i ^ {\prime}} ^ {j}}\right) ^ {2}. \tag{7}
$$

This regularizer promotes the fraction of the similarities to be close to the fraction of the target quantities. Ideally, we should use targetij = Pr(method i picks the correct answer for question $x _ { j } )$ , which is the intrinsic characteristic of the stochastic generator. However, this probability value is not readily available, therefore we leverage the bounds in Section 3 as target values, and empirically compute these target values as follows: For each question j and core configuration (generation method, temperature, aggregation method, etc.), we generate 80 solutions (5 independent runs of $N = 1 6$ with different seed numbers) to obtain a set of distinct solutions $y _ { 1 } , \ldots , y _ { K }$ . We then estimate the parameters $\hat { p } _ { k } , \hat { \mu } _ { k } , \hat { \sigma } _ { k }$ from these 80 solutions. We can then empirically identify whether the lower or upper bound of the probability is active and assign the target value as either the lower or upper bound with the corresponding size N. The bound provided in Theorem 3.3 is valid when N is large enough. For smaller N, we use the empirical accuracy as an alternative.

Combining two loss terms (6) and (7), we obtain the training problem

$$
\min _ {\theta} \min _ {v _ {1}, \dots , v _ {M} \in \mathbb {R} ^ {d}} \ell_ {\text { contrastive }} (\theta , v _ {1}, \dots , v _ {M}) + \tau \ell_ {\text { reg }} (\theta , v _ {1}, \dots , v _ {M}), \tag {8}
$$

where $\tau > 0$ is a hyperparameter aiming to promote the sample efficiency of the training procedure.

# 4.3 Inference Time Matching

At inference time, we pass any new question $x _ { \mathrm { n e w } } .$ , or equivalently its feature vector $f _ { \mathrm { n e w } }$ , through the trained network gθ to obtain the question embedding $g _ { \theta } ( f _ { \mathrm { n e w } } )$ . We then find the top-1 reasoning method by $m ^ { \star } = \arg \operatorname* { m a x } _ { i \in \mathcal { M } } s ( g _ { \theta } ( f _ { \mathrm { n e w } } ) , v _ { i } )$ , that maximizes the similarity score between the question and the trained representation vector $v _ { 1 } , \ldots , v _ { M }$ of the reasoning models. We then deploy method $m ^ { \star }$ to answer this question.

# 4.4 Discussions

We now discuss the necessity and importance of the design choices of our EPIC method.

Discussion 1 (Questions’ feature vector). There are multiple ways to obtain the feature vector $f _ { j }$ for each question $x _ { j }$ . For example, we can take $f _ { j }$ as the activation of the last token of $x _ { j }$ extracted from one of the layers (potentially the last layer) of the language model. This approach does not incur any additional memory requirement because we do not need to load any auxiliary models onto the device. However, the activation dimension of the language models is usually high: for example, in Qwen2.5-Math-7B-Instruct [Yang et al., 2024a], $D = 3 5 8 4$ . This high dimensionality could prohibit efficient training of the representation parameters θ. Alternatively, we can use a lightweight model to map $x _ { j }$ to $f _ { j }$ . This could incur additional memory overhead but generate a lower D as input to the network $g _ { \theta }$ . In the experiment, we will use a lightweight sentence embedding $\mathsf { a l 1 - M i n i L M - L 6 - v 2 } ^ { 1 }$ that has only 22.6 million parameters and incurs only 80MB of additional VRAM. The corresponding feature dimension is $D = 3 8 4$ .

Discussion 2 (Importance of embedding network $g _ { \theta } )$ . Given the question features $f _ { 1 } , \ldots , f _ { n } \in \mathbb { R } ^ { D }$ , one could simplify the representation learning problem (8) by optimizing the method embedding vectors $v _ { 1 } , \ldots , v _ { M }$ directly on the space of $\mathbb { R } ^ { \breve { D } }$ . This is equivalent to setting $d = D$ , and letting gθ collapse into an identity mapping. However, the proximity between two question features $f _ { j }$ and $f _ { j ^ { \prime } }$ does not convey enough information about the similarity regarding hardness, resource utilization, and suitability with methods. Moreover, learning the method embedding $v _ { j }$ on $\mathbb { R } ^ { D }$ is more difficult than on the smaller dimension space $\mathbb { R } ^ { d }$ . Hence, learning in $\mathbb { R } ^ { D }$ is inefficient. This observation necessitates the use of a lightweight question map gθ.

Discussion 3 (Adaptive method insertion). Given a universe of models M, problem (8) optimizes one vector $v _ { i }$ for a reasoning model $i \in \mathcal { M }$ . Alternatively, we could use another network $h _ { \vartheta }$ that could take a (text) description of a reasoning method and output the respective embedding vector in the representation space $\mathring { \mathbb { R } } ^ { d }$ . Having the second network $h _ { \vartheta }$ could unlock several new capabilities: (i) for a new reasoning method that is not in $\mathcal { M } .$ , we could quickly obtain its embedding and predict its performance on the questions, (ii) we could inverse engineer to design a better reasoning method. Unfortunately, training $h _ { \vartheta }$ requires a meaningful textual description of the reasoning methods. This is currently outside the scope of this paper, and we leave it for subsequent work.

# 5 Numerical Experiments

In this section, we present numerical experiments showcasing the performance of EPIC on the math answering task. Experiments for the code generation task are relegated to Appendix D.4.

Dataset. We use the MATH dataset [Hendrycks et al., 2021] as a training set, utilizing its training split of 7,500 math problems with solutions, as defined in Hendrycks et al. [2021]. For the code generation experiment, we use the LiveCodeBench dataset [Jain et al., 2025]. More details are in Appendix D.4. For evaluation, we test on the MATH500 test split, which contains 500 samples, as defined in Lightman et al. [2023b]. We also use the test set of the GSM8K [Cobbe et al., 2021] dataset to evaluate the transferability of the method embedding vectors learned in Section 4.2.

Base models. We employ $\mathrm { Q w e n } 2 . 5 – \mathrm { M a t h } – 7 \mathrm { B } – \mathrm { I n s t r u c t } ^ { 2 }$ as our generation model, and math-shepherdmistral-7b-prm [Wang et al., 2024b]3 as our reward model in the PRM framework. These models are fixed throughout our main experiments. For transferability experiments, we augment our universe of methods to include Qwen2.5-Math-1.5B. For code generation experiments, we use Qwen2.5-Coder-3B-Instruct and Qwen2.5-Coder-7B-Instruct.

![](images/be76cc95ed43e338d4b62724e2b235d16ffc6ef94a95a47238c0c1bc94e3392f.jpg)

<details>
<summary>scatter</summary>

| Method | Average Success Rate (Accuracy) | Average Cost (Tokens) |
| --- | --- | --- |
| Best-of-2 + PVL | 0.84 | 1500 |
| MCTS | 0.85 | 4500 |
| Offline Ada-BoK | 0.86 | 3000 |
| Best-of-4 + PVL | 0.86 | 2500 |
| Best-of-8 + PVL | 0.86 | 5000 |
| Other Methods | 0.84–0.87 | 10000–10000 |
| Ours (d = 64) | 0.86–0.90 | 2500–7000 |
| λ = 1.0 | 0.89 | 7000 |
| λ = 0.75 | 0.87 | 3500 |
| λ = 0.5 | 0.87 | 3200 |
| λ = 0.25 | 0.86 | 2200 |
| λ = 0.0 | 0.86 | 1200 |
</details>

Figure 2: Average success rate and token counts on the test set with embedding dimension $d = 6 4$ . Our ensemble planner performances with varying $\lambda \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ } are highlighted in red, and individual reasoning models in M are plotted in blue. The boundary of the ensemble planners covers the individual models in the universe M. The Upper Bound (UB) under M is the proportion of questions that at least one method in M could successfully solve.

Performance metrics. We use the accuracy to measure the quality of generation and average token counts to evaluate the efficiency of each method. For accuracy, we use the automatic grading4 provided by previous work Lightman et al. [2023b] to evaluate the accuracy of a generated solution in (4). To measure average token counts, we set the hyperparameter ‘max new token’ to 2048 for all methods and compute the average number of tokens generated.

Universe of methods M. We generate M consisting of 81 distinct methods, spanning a variety of reasoning strategies, aggregation techniques, and parameter configurations. A complete description is provided in Appendix B.

Dataset generation for contrastive learning. For a deterministic method i in $\mathcal { M } ,$ , we run inference on each question $x _ { j }$ once and record the accuracy and number of generated tokens. For the sampling method, we run the inference on each question five times to obtain a mean estimate of $a _ { i , j }$ and $c _ { i , j } .$

Baselines. We compare EPIC against three categories of baselines: (i) individual reasoning methods from the universe M, (ii) strong large-model references including DeepSeek-V3 and OpenAI-o1- mini, and (iii) alternative reasoning selection methods—RA, Offline Ada-BoK [Damani et al., 2025], DRA-λ, and CL-λ. All ensemble baselines and EPIC are trained and evaluated on the same M for fair comparison. Further details are provided in Appendix C.

Reproducibility. All experiments are conducted on a single machine with 8× NVIDIA RTX A5000 GPU and Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz.

# 5.1 Comparison between EPIC and Baselines

In the first experiment, we benchmark how EPIC, an ensemble model, outperforms individual reasoning models in the universe M. Table 1 presents the test performance comparison between our ensemble planner and individual reasoning methods, computed based on average accuracy and the number of generated tokens. We apply the regularization parameter $\tau = 1 0 ^ { - 3 }$ based on its better numerical results than other values shown in Appendix D. Our method with $\lambda = 0 . 2 5$ achieves an accuracy of 86.4%, matching the best-of-16 approach while using significantly fewer tokens: EPIC generates 1859.2 tokens while best-of-16 generates 10036.2 tokens. On a relative scale, this is a 5x reduction in the token counts at the same accuracy level. Compared to beam search, our approach at λ = 0.25 achieves better accuracy with a 29.5% reduction in token usage. With λ = 1.00, our method achieves the highest accuracy (89.4%) at a significantly lower token count (6,921.7).

To better visualize the performance of EPIC, we plot in Figure 2 a scatter plot locating the accuracycost trade-off of EPIC instances and representative reasoning models from M. Our EPIC instances (red) all lie on the frontier, thereby boosting the performance of the inference phase.

Since EPIC is an ensemble method constructed from the universe of reasoning methods M, the performance of EPIC is constrained by the capacity of the universe M itself. We could compute the best possible accuracy of the whole universe M on the test set: M could solve a question if there is at least one method from M that could generate a correct answer. Computing this value yields an upper bound of approximately 91.2%. Figure 2 highlights that our method provides a flexible trade-off between efficiency and accuracy, approaching the upper bound (dashed vertical line) while maintaining computational efficiency.

Table 1: Average accuracy and number of generated tokens on MATH500 using different reasoning methods and language models (full results in Appendix 4). Methods above the blue line are either upper bounds under M or are not included in M. Methods above the green line correspond to individual single-reasoning configurations (without selection modules). Methods above the brown line do not support accuracy–cost trade-offs. We compare EPIC-λ, DRA-λ, and CL-λ at various trade-off settings (groups separated by gray lines). Best results in each section are highlighted. 

<table><tr><td>Method</td><td>Accuracy ↑</td><td>Average Token Count ↓</td></tr><tr><td>OpenAI-o1-mini* [Jaech et al., 2024]</td><td>90.0</td><td>-</td></tr><tr><td>Deepseek-V3* [Liu et al., 2024]</td><td>90.2</td><td>-</td></tr><tr><td>Upper Bound under M</td><td>91.2</td><td>-</td></tr><tr><td>CoT-G</td><td>83.2</td><td>620.4</td></tr><tr><td>Best-of-2</td><td>84.8</td><td>1242.5</td></tr><tr><td>Best-of-4</td><td>86.2</td><td>2499.4</td></tr><tr><td>Best-of-8</td><td>86.6</td><td>4986.8</td></tr><tr><td>Best-of-16</td><td>86.8</td><td>10036.2</td></tr><tr><td>MCTS</td><td>85.4</td><td>4338.1</td></tr><tr><td>Beam-search</td><td>85.2</td><td>2638.1</td></tr><tr><td>RA</td><td>84.4</td><td>1752.4</td></tr><tr><td>Offline Ada-BoK</td><td>87.0</td><td>4095.2</td></tr><tr><td>DRA-0.25</td><td>86.2</td><td>2453.6</td></tr><tr><td>CL-0.25</td><td>86.0</td><td>2275.6</td></tr><tr><td>EPIC-0.25</td><td>86.4</td><td>1859.2</td></tr><tr><td>DRA-0.5</td><td>86.4</td><td>5719.3</td></tr><tr><td>CL-0.5</td><td>86.6</td><td>5320.2</td></tr><tr><td>EPIC-0.5</td><td>86.8</td><td>2482.6</td></tr><tr><td>DRA-0.75</td><td>86.4</td><td>7523.2</td></tr><tr><td>CL-0.75</td><td>87.0</td><td>7524.6</td></tr><tr><td>EPIC-0.75</td><td>87.6</td><td>3192.9</td></tr><tr><td>DRA-1.0</td><td>87.0</td><td>10542.2</td></tr><tr><td>CL-1.0</td><td>87.8</td><td>10923.4</td></tr><tr><td>EPIC-1.0</td><td>89.4</td><td>6921.7</td></tr></table>

∗Method not in M. We obtained results from Liu et al. [2024].

# 5.2 Transferability

We now examine the transferability of EPIC across both model scales and datasets. Table 2 summarizes results for two complementary settings: (a) transferring from the MATH to the GSM8K dataset, and (b) applying EPIC in a cost-aware multi-model environment with Qwen2.5-Math-1.5B and Qwen2.5-Math-7B.

# 5.2.1 Evaluating EPIC with Cost-Aware Multi-Model Reasoning

EPIC remains effective even when reasoning methods use heterogeneous base models. Previously, our universe M contained 81 methods built solely on Qwen2.5-Math-7B-Instruct. We now augment this space with an additional 81 methods using Qwen2.5-Math-1.5B-Instruct. Because larger models are computationally more expensive, we approximate the cost of each method as the product of its parameter count (in billions) and the number of generated tokens. This proxy aligns with real-world inference costs; alternatively, FLOPs or API pricing could be used.

Table 2: EPIC performance comparison across datasets and model sizes. (a) GSM8K results where EPIC is trained on the MATH dataset using $d = 6 4$ and $\lambda = 0 . 2 5$ . (b) Performance and cost comparison for Qwen2.5-Math-1.5B and 7B models. Best results in each column are highlighted.   
(a) GSM8K results 

<table><tr><td>Method</td><td>Accuracy ↑</td><td>Tokens ↓</td></tr><tr><td>CoT-G</td><td>93.5</td><td>297.0</td></tr><tr><td>Best-of-2</td><td>94.0</td><td>594.6</td></tr><tr><td>Best-of-4</td><td>94.0</td><td>1195.6</td></tr><tr><td>Best-of-8</td><td>94.3</td><td>2412.3</td></tr><tr><td>Best-of-16</td><td>94.2</td><td>5019.2</td></tr><tr><td>EPIC</td><td>95.0</td><td>2085.5</td></tr></table>

(b) Performance and cost with Qwen2.5 models 

<table><tr><td>Method</td><td>Accuracy ↑</td><td>Cost ↓</td></tr><tr><td>1.5B-CoT-G</td><td>76.0</td><td>856.5</td></tr><tr><td>1.5Best-of-4</td><td>78.6</td><td>3454.5</td></tr><tr><td>7B-CoT-G</td><td>83.2</td><td>4342.8</td></tr><tr><td>7B-Best-of-4</td><td>86.2</td><td>17495.8</td></tr><tr><td>1.5Best-of-16</td><td>79.4</td><td>62663.2</td></tr><tr><td>7B-Best-of-16</td><td>86.8</td><td>70253.4</td></tr><tr><td>EPIC (λ = 0.25)</td><td>86.2</td><td>8047.8</td></tr><tr><td>EPIC (λ = 1.0)</td><td>89.0</td><td>35705.4</td></tr></table>

Table 2b shows that EPIC adapts effectively across cost regimes. At λ = 0.25, EPIC achieves an accuracy of 86.2, matching the 7B-Best-of-4 method, while reducing the cost by over 50% (8047.8 vs. 17495.8). At λ = 1.0, EPIC attains the highest overall accuracy (89.0), outperforming 7B-Bestof-16 while maintaining roughly half the computational cost (35705.4 vs. 70253.4). These results demonstrate EPIC’s capacity to balance accuracy and cost by dynamically leveraging reasoning methods from different model sizes.

# 5.2.2 Transfer to Another In-Domain Dataset

To further assess generalization, we evaluate EPIC trained on MATH and test it on GSM8K [Cobbe et al., 2021], another widely used arithmetic reasoning benchmark. As shown in Table 2a, GSM8K is a simpler dataset; hence, absolute gains are smaller. Nonetheless, EPIC achieves the best accuracy (95.0%) while requiring fewer tokens than high-cost baselines such as Best-of-8 or Best-of-16. This indicates that EPIC’s learned representations transfer across related reasoning distributions and continue to yield efficient inference-time behavior.

# 5.3 Additional Experiments

We perform ablation studies to better understand the impact of the representation dimension, the regularization parameter τ , and the tradeoff parameter λ, the utility function, the transferability, and the generalization on the code generation task. In the experiment of representation dimension, we fix $\lambda = 0 . 5$ and vary dimension d ∈ {16, 32, 64, 128}. Our results show a general trend: as d increases, accuracy improves, while average token count stabilizes or slightly decreases. This result empirically confirms the expectation that increasing the embedding dimension could boost the performance of our method. In the experiment on the impact of λ, we observe a clear cost-accuracy trade-off as we fix d = 64 and varies λ ∈ {0.00, 0.25, 0.50, 0.75, 1.00}. In the ablation study of the utility function, we switch to an alternative functional form, as shown in equation (5), and observe a decrease in performance, indicating that our design choice is superior. Due to space constraints, further experimental details are provided in Appendix D.

# 6 Conclusion

We introduced EPIC, the Ensemble PlannIng with Contrastive learning framework, a contrastive learning framework that plans optimal reasoning strategies for language models by matching questions to suitable methods. Our analysis established new accuracy bounds for common aggregation techniques, which directly inform a regularization term to guide more sample-efficient learning. Experiments on mathematical reasoning benchmarks demonstrate that EPIC leverages these theoretical insights to achieve strong improvements in both accuracy and inference cost, showcasing the value of principled modeling for reasoning method selection.

Acknowledgments. Viet Anh Nguyen gratefully acknowledges the support from the CUHK’s Improvement on Competitiveness in Hiring New Faculties Funding Scheme, UGC ECS Grant 24210924, and UGC GRF Grant 14208625.

# References

Edward Beeching, Lewis Tunstall, and Sasha Rush. Scaling test-time compute with open models, 2024. URL https://huggingface.co/spaces/HuggingFaceH4/ blogpost-scaling-test-time-compute.   
Zhenni Bi, Kai Han, Chuanjian Liu, Yehui Tang, and Yunhe Wang. Forest-of-thought: Scaling test-time compute for enhancing LLM reasoning. In Forty-second International Conference on Machine Learning, 2025.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901, 2020.   
Ngoc Bui, Hieu Trung Nguyen, Shantanu Kumar, Julian Theodore, Weikang Qiu, Viet Anh Nguyen, and Rex Ying. Mixture-of-personas language models for population simulation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 24761–24778, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/ 2025.findings-acl.1271.   
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.   
Mehul Damani, Idan Shenfeld, Andi Peng, Andreea Bobu, and Jacob Andreas. Learning how hard to think: Input-adaptive allocation of LM computation. In The Thirteenth International Conference on Learning Representations, 2025.   
Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory- ´ efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, volume 35, pages 16344–16359, 2022.   
Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. rStar-Math: Small LLMs can master math reasoning with self-evolved deep thinking. arXiv preprint arXiv:2501.04519, 2025.   
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Advances in Neural Information Processing Systems, 2021.   
Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.   
Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025.   
Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. A survey on large language models for code generation. arXiv preprint arXiv:2406.00515, 2024.   
Hyoungwook Jin, Seonghee Lee, Hyungyu Shin, and Juho Kim. Teach ai how to code: Using large language models as teachable agents for programming education. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, pages 1–28, 2024.

Dacheng Li, Shiyi Cao, Chengkun Cao, Xiuyu Li, Shangyin Tan, Kurt Keutzer, Jiarong Xing, Joseph E Gonzalez, and Ion Stoica. S\*: Test time scaling for code generation. arXiv preprint arXiv:2502.14382, 2025a.   
Dacheng Li, Shiyi Cao, Chengkun Cao, Xiuyu Li, Shangyin Tan, Kurt Keutzer, Jiarong Xing, Joseph E Gonzalez, and Ion Stoica. S\*: Test time scaling for code generation. arXiv preprint arXiv:2502.14382, 2025b.   
Kenneth Li, Oam Patel, Fernanda Viegas, Hanspeter Pfister, and Martin Wattenberg. Inference-time´ intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems, volume 36, pages 41451–41530, 2023.   
Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making large language models better reasoners with step-aware verifier. arXiv preprint arXiv:2206.02336, 2022.   
Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023a.   
Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023b.   
Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. Proceedings of Machine Learning and Systems, 6: 87–100, 2024.   
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.   
Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.   
Rohin Manvi, Anikait Singh, and Stefano Ermon. Adaptive inference-time compute: LLMs can predict if they can do better, even mid-generation. arXiv preprint arXiv:2410.02725, 2024.   
Hieu Trung Nguyen, Bao Nguyen, Binh Nguyen, and Viet Anh Nguyen. Task-driven layerwise additive activation intervention. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 506–513, Albuquerque, New Mexico, April 2025a. Association for Computational Linguistics. ISBN 979-8-89176-190-2. doi: 10.18653/v1/2025.naacl-short.43.   
Hieu Trung Nguyen, Bao Nguyen, and Viet Anh Nguyen. Structured pruning for diverse best-of-n reasoning optimization. In Findings of the Association for Computational Linguistics: ACL 2025, pages 23911–23922, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1225.   
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.   
Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.   
Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang, Omar Fawzi, et al. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025.   
Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.   
Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024.   
Ziyu Wan, Xidong Feng, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like tree-search can guide large language model decoding and training. In Forty-first International Conference on Machine Learning, 2024.   
Jun Wang, Meng Fang, Ziyu Wan, Muning Wen, Jiachen Zhu, Anjie Liu, Ziqin Gong, Yan Song, Lei Chen, Lionel M. Ni, Linyi Yang, Ying Wen, and Weinan Zhang. OpenR: An open source framework for advanced reasoning with large language models. arXiv preprint arXiv:2410.09671, 2024a.   
Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, Bangkok, Thailand, 2024b. Association for Computational Linguistics.   
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in Large Language Models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837, 2022.   
Yuhuai Wu, Albert Qiaochu Jiang, Wenda Li, Markus Rabe, Charles Staats, Mateja Jamnik, and Christian Szegedy. Autoformalization with Large Language Models. In Advances in Neural Information Processing Systems, volume 35, pages 32353–32368, 2022.   
Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. TravelPlanner: A benchmark for real-world planning with language agents. In Forty-first International Conference on Machine Learning, 2024a.   
Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy P Lillicrap, Kenji Kawaguchi, and Michael Shieh. Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning. In The First Workshop on System-2 Reasoning at Scale, NeurIPS, 2024b.   
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024a.   
An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-Math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024b.   
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with Large Language Models. In Advances in Neural Information Processing Systems, volume 36, pages 11809–11822, 2023.   
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024.

Fei Yu, Anningzhe Gao, and Benyou Wang. Ovm, outcome-supervised value models for planning in mathematical reasoning. arXiv preprint arXiv:2311.09724, 2023.   
Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-MCTS∗: LLM self-training via process reward guided tree search. In Advances in Neural Information Processing Systems, volume 37, pages 64735–64772, 2024.   
Xiang Zhang, Tianze Ling, Zhi Jin, Sheng Xu, Zhiqiang Gao, Boyan Sun, Zijie Qiu, Jiaqi Wei, Nanqing Dong, Guangshuai Wang, et al. π-PrimeNovo: an accurate and efficient non-autoregressive deep learning model for de novo peptide sequencing. Nature Communications, 16(1):267, 2025.

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The claims in the abstract and introduction are clearly stated and are fully supported by the theoretical analysis and experimental results presented in the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide the limitations in the impact statement at the end of the main paper.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provide a full set of assumptions and complete proof.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide all necessary information

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Yes, we include an anonymous link to the implementation code for this paper.

# 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We provide information about our experiments in the experimental section.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Due to limited computational resources, we were unable to run multiple experimental trials required to compute error bars or statistical significance.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide information about our machine in experimental section.

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We make sure to preserve anonymity.

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We have an impact statement at the end of the main paper.

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: the paper poses no such risks.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: we cite all existing assets.

# 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: the paper does not release new assets.

# 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

# A Proofs of Section 3

After sampling N independent candidate solutions from the distribution, we aggregate them to obtain the output answer. We focus on characterizing the probability that the output answer is $y _ { 1 }$ , meaning that the output answer is a correct solution to question $q .$ .

Let C˜ (N) $\tilde { \cal C } ^ { ( N ) } = ( \tilde { C } _ { 1 } ^ { ( N ) } , \dots , \tilde { C } _ { K } ^ { ( N ) } )$ = , . . . , denote the counts of each solutions after N samples, so that $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \tilde { C } _ { k } ^ { ( N ) } = N } \end{array}$ . The count random vector $\tilde { \boldsymbol { C } } ^ { ( N ) }$ ) follows the multinomial distribution:

$$
\operatorname * {P r} (\tilde {\boldsymbol {C}} ^ {(N)} = \boldsymbol {c}) = \frac {N !}{c _ {1} ! \dots c _ {K} !} \prod_ {k = 1} ^ {K} p _ {k} ^ {c _ {k}} \tag {9}
$$

for any vector $\pmb { c } = \left( c _ { 1 } , \dots , c _ { K } \right)$ of natural numbers summing up to $N .$

# A.1 Majority Vote Analysis

The majority vote selects $y _ { 1 }$ , the correct answer, with probability one $\mathrm { i f } c _ { 1 } > c _ { k }$ for all $k \neq 1$ , or with some probability $0 < w < 1 \mathrm { i f } c _ { 1 } \geq c _ { k }$ for all $k \neq \bar { 1 }$ with strict equality $c _ { 1 } = c _ { k }$ for some $k \neq 1$ . The value w represents the probability of choosing $y _ { 1 }$ in cases of ties.

Thus, the exact probability that the majority vote selects $y _ { 1 }$ is bounded by

$$
\sum_{\substack{c_{1} + \dots +c_{K} = N\\ c_{1} > c_{k}\forall k\neq 1}}\frac{N!}{c_{1}!\cdots c_{K}!}\prod_{k^{\prime} = 1}^{K}p_{k^{\prime}}^{c_{k^{\prime}}} \leq \Pr (\text{majority vote picks} y_{1}) \leq \sum_{\substack{c_{1} + \dots +c_{K} = N\\ c_{1}\geq c_{k}\forall k\neq 1}}\frac{N!}{c_{1}!\cdots c_{K}!}\prod_{k^{\prime} = 1}^{K}p_{k^{\prime}}^{c_{k^{\prime}}}.
$$

However, these bounds are computationally intractable for large N due to the combinatorial explosion of possible vote count configurations. Moreover, they do not provide clear insight into how the selection probability changes as N varies, limiting their practical utility for analytical understanding or approximation.

We now state the theorem that bounds the probability that the count of one solution is less than or equal to another:

Proposition A.1 (Count upper-bound). Assume $p _ { a } ~ > ~ p _ { b }$ for any pair of distinct indices $a , b \in$ $\{ 1 , \bar { . } . . , K \} , a \neq b .$ . Then, we have

$$
\operatorname * {P r} (\tilde {C} _ {a} ^ {(N)} \leq \tilde {C} _ {b} ^ {(N)}) \leq \exp \left(- N \left(\sqrt {p _ {a}} - \sqrt {p _ {b}}\right) ^ {2}\right). \tag {10}
$$

Proof of Proposition A.1. For each draw $u = 1 , \ldots , N .$ , let $C _ { u }$ denote the selected category with $\operatorname* { P r } ( \dot { C } _ { u } = k ) = p _ { k }$ . Define

$$
\tilde {C} _ {k} ^ {(N)} = \sum_ {u = 1} ^ {N} 1 _ {\{C _ {u} = k \}}, \qquad k = 1, \ldots , K.
$$

Then the difference of counts between bins a and b can be written as

$$
\tilde {C} _ {a} ^ {(N)} - \tilde {C} _ {b} ^ {(N)} = \sum_ {u = 1} ^ {N} X _ {u}, \qquad X _ {u} := 1 _ {\{C _ {u} = a \}} - 1 _ {\{C _ {u} = b \}}.
$$

The variables $( X _ { u } ) _ { u = 1 } ^ { N }$ are i.i.d., therefore for any $t > 0$ , Markov’s inequality implies

$$
\operatorname * {P r} (\tilde {C} _ {a} ^ {(N)} \leq \tilde {C} _ {b} ^ {(N)}) = \operatorname * {P r} \Big (e ^ {- t (\tilde {C} _ {a} ^ {(N)} - \tilde {C} _ {b} ^ {(N)})} \geq 1 \Big) \leq \mathbb {E} \Big [ e ^ {- t (\tilde {C} _ {a} ^ {(N)} - \tilde {C} _ {b} ^ {(N)})} \Big ] = \big (\mathbb {E} [ e ^ {- t X _ {1}} ] \big) ^ {N}.
$$

Conditioning on $C _ { 1 }$ yields

$$
\mathbb {E} [ e ^ {- t X _ {1}} ] = p _ {a} e ^ {- t} + p _ {b} e ^ {t} + (1 - p _ {a} - p _ {b}).
$$

Substituting this expression into the previous bound gives

$$
\operatorname * {P r} (\tilde {C} _ {a} ^ {(N)} \leq \tilde {C} _ {b} ^ {(N)}) \leq \exp \Bigl (N \log \bigl (p _ {a} e ^ {- t} + p _ {b} e ^ {t} + 1 - p _ {a} - p _ {b} \bigr) \Bigr).
$$

To optimize the bound, define $h ( t ) = p _ { a } e ^ { - t } + p _ { b } e ^ { t } + 1 - p _ { a } - p _ { b }$ . Minimizing $h ( t )$ over $t > 0$ leads to the first-order optimality condition

$$
h ^ {\prime} (t) = - p _ {a} e ^ {- t} + p _ {b} e ^ {t} = 0 \quad \Longrightarrow \quad e ^ {2 t ^ {*}} = \frac {p _ {a}}{p _ {b}} \quad \Longrightarrow \quad t ^ {*} = \frac {1}{2} \log \left(\frac {p _ {a}}{p _ {b}}\right).
$$

Since $p _ { a } > p _ { b }$ , we have $t ^ { * } > 0$ , which is valid. Plugging t∗ back into $h ( t )$ , we obtain

$$
h (t ^ {*}) = 1 - p _ {a} - p _ {b} + 2 \sqrt {p _ {a} p _ {b}} = 1 - (\sqrt {p _ {a}} - \sqrt {p _ {b}}) ^ {2}.
$$

Therefore, we obtain

$$
\operatorname * {P r} (\tilde {C} _ {a} ^ {(N)} \leq \tilde {C} _ {b} ^ {(N)}) \leq \left(1 - (\sqrt {p _ {a}} - \sqrt {p _ {b}}) ^ {2}\right) ^ {N}.
$$

Finally, using log $\ u _ { \ u { \ u { \ u { \ u { \ u { \ u { \ u { \chi } } } } } } } } ( 1 - x ) \ d t \leq - x$ for $x \in ( 0 , 1 )$ , we obtain the exponential form

$$
\operatorname * {P r} (\tilde {C} _ {a} ^ {(N)} \leq \tilde {C} _ {b} ^ {(N)}) \leq \exp \left(- N (\sqrt {p _ {a}} - \sqrt {p _ {b}}) ^ {2}\right).
$$

The proof is complete.

![](images/ef6904849765fa69ac41a383a8cb85dab08d3f86ce9037b4e3a0d07c1e9780c2.jpg)

We are now ready to prove Theorem 3.1.

Proof of Theorem 3.1. Recall that we assume $y _ { 1 }$ is the correct solution. Consider the case where $p _ { 1 } > p _ { k }$ for all $k = 2 , \ldots , K$ . We obtain by applying Proposition A.1:

$$
\begin{array}{l} \operatorname * {P r} (\text { majority   vote   picks } y _ {1}) \geq \operatorname * {P r} \left(\bigcap_ {k = 2} ^ {K} \left\{\tilde {C} _ {1} ^ {(N)} > \tilde {C} _ {k} ^ {(N)} \right\}\right) \\ = 1 - \operatorname * {P r} \left(\bigcup_ {k = 2} ^ {K} \left\{\tilde {C} _ {1} ^ {(N)} \leq \tilde {C} _ {k} ^ {(N)} \right\}\right) \\ \geq 1 - \sum_ {k = 2} ^ {K} \operatorname * {P r} \left(\tilde {C} _ {1} ^ {(N)} \leq \tilde {C} _ {k} ^ {(N)}\right) \\ \geq 1 - \sum_ {k = 2} ^ {K} e ^ {- N \left(\sqrt {p _ {1}} - \sqrt {p _ {k}}\right) ^ {2}}. \\ \end{array}
$$

Now consider the case where there exists $k \in \{ 2 , \ldots , K \}$ such that $p _ { k } > p _ { 1 } .$

$$
\begin{array}{l} \operatorname * {P r} (\text { majority   vote   picks } y _ {1}) \leq \operatorname * {P r} \left(\bigcap_ {k = 2} ^ {K} \left\{\tilde {C} _ {1} ^ {(N)} \geq \tilde {C} _ {k} ^ {(N)} \right\}\right) \leq \operatorname * {P r} \left(\tilde {C} _ {1} ^ {(N)} \geq \tilde {C} _ {k} ^ {(N)}\right) \\ \leq e ^ {- N \left(\sqrt {p _ {k}} - \sqrt {p _ {1}}\right) ^ {2}}, \\ \end{array}
$$

where the last inequality follows from Proposition A.1.

![](images/e5cb44277c2bddecfc35c558787b0237831057f474f6b07d40833fac24867ac2.jpg)

# A.2 Aggregation using Summation of Scores

Let $\tilde { C } _ { k } ^ { ( N ) }$ be the count of the solution $y _ { k }$ , and let each of the $\tilde { C } _ { k } ^ { ( N ) }$ PRM scores be independently distributed with the same distribution $\tilde { S } _ { k u } \sim \mathcal N ( \mu _ { k } , \sigma _ { k } ^ { 2 } )$ for all u. Define $\begin{array} { r } { \tilde { U } _ { k } ^ { ( N ) } = \sum _ { u = 1 } ^ { \tilde { C } _ { k } ^ { ( N ) } } \tilde { S } _ { k u } } \end{array}$ C˜(N) S˜ku. We now state the theorem that bounds the probability that the total score for one solution is less than or equal to that of another.

Theorem A.2 (Sum upper-bound). Suppose that $p _ { a } \mu _ { a } > p _ { b } \mu _ { b }$ for some distinct $a , b \in \{ 1 , \dots , K \}$ . Then

$$
\operatorname * {P r} \left(\tilde {U} _ {a} ^ {(N)} \leq \tilde {U} _ {b} ^ {(N)}\right) \leq \inf _ {t > 0} \exp \left(N p _ {a} \left(e ^ {- t \mu_ {a} + \frac {1}{2} t ^ {2} \sigma_ {a} ^ {2}} - 1\right) + N p _ {b} \left(e ^ {t \mu_ {b} + \frac {1}{2} t ^ {2} \sigma_ {b} ^ {2}} - 1\right)\right). \tag {11}
$$

Moreover, under this condition, the right-hand side decays exponentially in N.

Proof. For each draw $u = 1 , \ldots , N$ , let $C _ { u }$ denote the category selected, with $\Pr ( C _ { u } = k ) = p _ { k }$ . Let $\tilde { S } _ { k u } \sim \mathcal N ( \mu _ { k } , \sigma _ { k } ^ { 2 } )$ be independent across both k and u, and independent of $( C _ { u } ) _ { u = 1 } ^ { N }$ . Then

$$
\tilde {U} _ {k} ^ {(N)} = \sum_ {u = 1} ^ {N} \tilde {S} _ {k u}   1 _ {\{C _ {u} = k \}}, \qquad k = 1, \ldots , K,
$$

so that

$$
\tilde {U} _ {a} ^ {(N)} - \tilde {U} _ {b} ^ {(N)} = \sum_ {u = 1} ^ {N} X _ {u}, \quad \text { where } X _ {u} = \tilde {S} _ {a u} 1 _ {\{C _ {u} = a \}} - \tilde {S} _ {b u} 1 _ {\{C _ {u} = b \}}.
$$

The random variables $( X _ { u } ) _ { u = 1 } ^ { N }$ are i.i.d. Hence, for any $t > 0$ , Markov’s inequality yields

$$
\operatorname * {P r} \left(\tilde {U} _ {a} ^ {(N)} \leq \tilde {U} _ {b} ^ {(N)}\right) \leq \mathbb {E} \Big [ e ^ {- t (\tilde {U} _ {a} ^ {(N)} - \tilde {U} _ {b} ^ {(N)})} \Big ] = \left(\mathbb {E} [ e ^ {- t X _ {1}} ]\right) ^ {N}.
$$

Conditioning on $C _ { 1 }$ gives

$$
\mathbb {E} [ e ^ {- t X _ {1}} ] = p _ {a} \mathbb {E} [ e ^ {- t \tilde {S} _ {a 1}} ] + p _ {b} \mathbb {E} [ e ^ {t \tilde {S} _ {b 1}} ] + (1 - p _ {a} - p _ {b}).
$$

Using the moment generating function of a normal random variable gives us

$$
\mathbb {E} [ e ^ {t \tilde {S} _ {k 1}} ] = e ^ {t \mu_ {k} + \frac {1}{2} t ^ {2} \sigma_ {k} ^ {2}}.
$$

Finally, we obtain

$$
\mathbb {E} \Big [ e ^ {- t (\tilde {U} _ {a} ^ {(N)} - \tilde {U} _ {b} ^ {(N)})} \Big ] = \Big (p _ {a} e ^ {- t \mu_ {a} + \frac {1}{2} t ^ {2} \sigma_ {a} ^ {2}} + p _ {b} e ^ {t \mu_ {b} + \frac {1}{2} t ^ {2} \sigma_ {b} ^ {2}} + 1 - p _ {a} - p _ {b} \Big) ^ {N}.
$$

To simplify the expression, we use $\log ( 1 + x ) \leq x { \mathrm { ~ f o r ~ } } x > - 1$ . We obtain

$$
\operatorname * {P r} \left(\tilde {U} _ {a} ^ {(N)} \leq \tilde {U} _ {b} ^ {(N)}\right) \leq \exp \Bigl (N p _ {a} (e ^ {- t \mu_ {a} + \frac {1}{2} t ^ {2} \sigma_ {a} ^ {2}} - 1) + N p _ {b} (e ^ {t \mu_ {b} + \frac {1}{2} t ^ {2} \sigma_ {b} ^ {2}} - 1) \Bigr).
$$

Optimizing over $t > 0 ,$ , we obtain

$$
\operatorname * {P r} \left(\tilde {U} _ {a} ^ {(N)} \leq \tilde {U} _ {b} ^ {(N)}\right) \leq \inf _ {t > 0} \exp \Bigl (N p _ {a} (e ^ {- t \mu_ {a} + \frac {1}{2} t ^ {2} \sigma_ {a} ^ {2}} - 1) + N p _ {b} (e ^ {t \mu_ {b} + \frac {1}{2} t ^ {2} \sigma_ {b} ^ {2}} - 1) \Bigr).
$$

To verify the exponential decay, we define

$$
F (t) = p _ {a} \left(e ^ {- t \mu_ {a} + \frac {1}{2} t ^ {2} \sigma_ {a} ^ {2}} - 1\right) + p _ {b} \left(e ^ {t \mu_ {b} + \frac {1}{2} t ^ {2} \sigma_ {b} ^ {2}} - 1\right).
$$

We have $F ( 0 ) = 0$ and

$$
F ^ {\prime} (0) = - p _ {a} \mu_ {a} + p _ {b} \mu_ {b} <   0,
$$

so for small $t > 0 , F ( t ) < 0$ . Therefore, the exponent is negative and the bound decays exponentially in N . □

We can now prove Theorem 3.2.

Proof of Theorem 3.2. Consider the case where $p _ { 1 } \mu _ { 1 } > p _ { k } \mu _ { k }$ for all $k = 2 , \ldots , K$ . Following an analogous argument as in the proof of Theorem 3.1, we have

$$
\operatorname * {P r} (\text { PRM\_Vote   picks } y _ {1}) \geq 1 - \sum_ {k = 2} ^ {K} \inf _ {t _ {k} > 0} \exp \left(N p _ {1} \left(e ^ {- t _ {k} \mu_ {1} + \frac {1}{2} t _ {k} ^ {2} \sigma_ {1} ^ {2}} - 1\right) + N p _ {k} \left(e ^ {t _ {k} \mu_ {k} + \frac {1}{2} t _ {k} ^ {2} \sigma_ {k} ^ {2}} - 1\right)\right).
$$

Consider the case where there exists k such that $p _ { k } \mu _ { k } > p _ { 1 } \mu _ { 1 }$ . We have

$$
\operatorname * {P r} (\text { PRM\_Vote   picks } y _ {1}) \leq \operatorname * {P r} (\tilde {U} _ {k} ^ {(N)} \leq \tilde {U} _ {1} ^ {(N)})
$$

$$
\leq \inf _ {t > 0} \exp \left(N p _ {k} \left(e ^ {- t \mu_ {k} + \frac {1}{2} t ^ {2} \sigma_ {k} ^ {2}} - 1\right) + N p _ {1} \left(e ^ {t \mu_ {1} + \frac {1}{2} t ^ {2} \sigma_ {1} ^ {2}} - 1\right)\right).
$$

This completes the proof.

![](images/ef8b4e0fe753135d8fcc457f76ab73f1699650cd799eb196890fb85cf4512b12.jpg)

# A.3 Aggregation using Maximum of Scores

To ease the readability, we recall the setup for this section. Let (N) $\tilde { C } _ { k } ^ { ( N ) }$ be the count of the solution $y _ { k } .$ , and let each of the $\tilde { C } _ { k } ^ { ( N ) }$ PRM scores be independently distributed with the same distribution $\tilde { S } _ { k u } \sim \mathcal N ( \mu _ { k } , \sigma _ { k } ^ { 2 } )$ for all u. We define $\tilde { M } _ { k } ^ { ( N ) } = \operatorname* { m a x } _ { 1 \leq u \leq \tilde { C } _ { k } ^ { ( N ) } } \tilde { S } _ { k u }$ .

Proposition A.3 (Max PRM upper bound). Assume $\sigma _ { a } > \sigma _ { b }$ for some distinct pair $a , b \in \{ 1 , \ldots , K \}$ , $a \neq b .$ . Then, for any $N \geq 1$ ,

$$
\operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq \tilde {M} _ {b} ^ {(N)}) \leq \inf _ {t \in \mathbb {R}} \Bigg \{(1 - p _ {a} [ 1 - \Phi (\frac {t - \mu_ {a}}{\sigma_ {a}}) ]) ^ {N} + 1 - (1 - p _ {b} [ 1 - \Phi (\frac {t - \mu_ {b}}{\sigma_ {b}}) ]) ^ {N} \Bigg \}.
$$

Moreover, the right-hand side converges to zero as $N \to \infty$ .

Proof of Proposition A.3. We start by characterizing the distribution of $\tilde { M } _ { b } ^ { ( N ) }$ . Conditional on $\tilde { C } _ { b } ^ { ( N ) } = m$ , we have

$$
\operatorname * {P r} (\tilde {M} _ {b} ^ {(N)} \leq t \mid \tilde {C} _ {b} ^ {(N)} = m) = [ \Phi_ {b} (t) ] ^ {m},
$$

where $\Phi _ { b } ( t ) = \Phi ( ( t - \mu _ { b } ) / \sigma _ { b } )$ is the CDF of $\mathcal { N } ( \mu _ { b } , \sigma _ { b } ^ { 2 } )$ . Substituting this into the law of total probability yields

$$
\begin{array}{l} \operatorname * {P r} (\tilde {M} _ {b} ^ {(N)} \leq t) = \sum_ {m = 0} ^ {N} \operatorname * {P r} (M = m) \operatorname * {P r} (\tilde {M} _ {b} ^ {(N)} \leq t \mid M = m) \\ = \sum_ {m = 0} ^ {N} \binom {N} {m} p _ {b} ^ {m} (1 - p _ {b}) ^ {N - m} [ \Phi_ {b} (t) ] ^ {m} \\ = \sum_ {m = 0} ^ {N} {\binom {N} {m}} (p _ {b} \Phi_ {b} (t)) ^ {m} (1 - p _ {b}) ^ {N - m} \\ = (1 - p _ {b} + p _ {b} \Phi_ {b} (t)) ^ {N} \\ = \left(1 - p _ {b} [ 1 - \Phi_ {b} (t) ]\right) ^ {N}. \\ \end{array}
$$

Taking the complement gives

$$
\operatorname * {P r} (\tilde {M} _ {b} ^ {(N)} > t) = 1 - (1 - p _ {b} [ 1 - \Phi_ {b} (t) ]) ^ {N}.
$$

An identical argument gives

$$
\operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq t) = (1 - p _ {a} [ 1 - \Phi_ {a} (t) ]) ^ {N}, \qquad \Phi_ {a} (t) = \Phi \bigg (\frac {t - \mu_ {a}}{\sigma_ {a}} \bigg).
$$

Next, we have that

$$
\begin{array}{l} \operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq \tilde {M} _ {b} ^ {(N)}) = \operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq t \text {   or   } \tilde {M} _ {b} ^ {(N)} > t \text {   for   all   } t) \\ \leq \operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq t \text {   or   } \tilde {M} _ {b} ^ {(N)} > t \text {   for   any   } t) \\ \leq \operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq t) + \operatorname * {P r} (\tilde {M} _ {b} ^ {(N)} > t), \\ \end{array}
$$

where the last inequality follows from the union bound. Hence we have the following for every $t \in \mathbb { R }$ :

$$
\operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq \tilde {M} _ {b} ^ {(N)}) \leq (1 - p _ {a} [ 1 - \Phi (\frac {t - \mu_ {a}}{\sigma_ {a}}) ]) ^ {N} + 1 - (1 - p _ {b} [ 1 - \Phi (\frac {t - \mu_ {b}}{\sigma_ {b}}) ]) ^ {N}
$$

Since this inequality holds for all t, we may optimize the bound by taking the infimum over $t \in \mathbb { R } :$

$$
\operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq \tilde {M} _ {b} ^ {(N)}) \leq \inf _ {t \in \mathbb {R}} \left\{(1 - p _ {a} [ 1 - \Phi (\frac {t - \mu_ {a}}{\sigma_ {a}}) ]) ^ {N} + 1 - (1 - p _ {b} [ 1 - \Phi (\frac {t - \mu_ {b}}{\sigma_ {b}}) ]) ^ {N} \right\}.
$$

We now analyze the asymptotic decay of the bound as $N \to \infty$ . Let $t _ { N } = \mu _ { b } + \sigma _ { b } \sqrt { 2 ( 1 + \varepsilon ) \log N }$ for some fixed $\varepsilon \in ( 0 , 1 )$ . We define

$$
z _ {b, N} = \frac {t _ {N} - \mu_ {b}}{\sigma_ {b}} = \sqrt {2 (1 + \varepsilon) \log N}, \qquad z _ {a, N} = \frac {t _ {N} - \mu_ {a}}{\sigma_ {a}} = \frac {\sigma_ {b}}{\sigma_ {a}} \sqrt {2 (1 + \varepsilon) \log N} - \frac {\mu_ {a} - \mu_ {b}}{\sigma_ {a}}.
$$

Then, the bound evaluated at $t _ { N }$ becomes:

$$
\operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq \tilde {M} _ {b} ^ {(N)}) \leq \inf _ {t \in \mathbb {R}} \left\{(1 - p _ {a} [ 1 - \Phi (z _ {a, N}) ]) ^ {N} + 1 - (1 - p _ {b} [ 1 - \Phi (z _ {b, N}) ]) ^ {N} \right\}.
$$

We use the standard Mills inequalities, valid for all $x > 0 { : }$

$$
\frac {x}{x ^ {2} + 1} \varphi (x) \leq 1 - \Phi (x) \leq \frac {\varphi (x)}{x}, \quad \varphi (x) = (2 \pi) ^ {- 1 / 2} e ^ {- x ^ {2} / 2}.
$$

We first study $1 - ( 1 - p _ { b } [ 1 - \Phi \left( z _ { b , N } \right) ] ) ^ { N }$ . For any $0 \leq x \leq 1$ it holds that $1 - ( 1 - x ) ^ { N } \leq N x .$

We use it to have

$$
1 - (1 - p _ {b} [ 1 - \Phi (z _ {b, N}) ]) ^ {N} \leq N p _ {b} [ 1 - \Phi (z _ {b, N}) ].
$$

Using the Mills upper bound, we obtain

$$
1 - \Phi (z _ {b, N}) \leq \frac {\varphi (z _ {b , N})}{z _ {b , N}} = \frac {1}{\sqrt {2 \pi} z _ {b , N}} N ^ {- (1 + \varepsilon)}.
$$

Hence, we have

$$
N p _ {b} [ 1 - \Phi (z _ {b, N}) ] \leq \frac {p _ {b}}{\sqrt {2 \pi} z _ {b , N}} N ^ {- \varepsilon} \to 0 \quad \text { as } N \to \infty .
$$

Next we study $( 1 - p _ { a } [ 1 - \Phi ( z _ { a , N } ) ] ) ^ { N }$ . We start with the inequality

$$
0 \leq (1 - p _ {a} [ 1 - \Phi (z _ {a, N}) ]) ^ {N} \leq \exp (- N p _ {a} [ 1 - \Phi (z _ {a, N}) ]),
$$

which holds because $1 - x \leq e ^ { - x }$ for any $x \in [ 0 , 1 ]$ .

$\mathrm { I f } ~ z _ { a , N } \leq 0$ for sufficiently large N , then $1 - \Phi ( z _ { a , N } ) \geq 1 / 2$ , so that

$$
(1 - p _ {a} [ 1 - \Phi (z _ {a, N}) ]) ^ {N} \leq (1 - p _ {a} / 2) ^ {N} \rightarrow 0 \quad \text { as } N \rightarrow \infty .
$$

Otherwise, if $z _ { a , N } > 0$ , the lower Mills bound gives

$$
1 - \Phi (z _ {a, N}) \geq \frac {C}{z _ {a , N}} e ^ {- z _ {a, N} ^ {2} / 2}
$$

for some constant $C > 0$ . Hence,

$$
N p _ {a} [ 1 - \Phi (z _ {a, N}) ] \geq C ^ {\prime} \frac {N}{z _ {a , N}} e ^ {- z _ {a, N} ^ {2} / 2}.
$$

Using the definition of $z _ { a , N }$ , we obtain

$$
z _ {a, N} ^ {2} = \frac {\sigma_ {b} ^ {2}}{\sigma_ {a} ^ {2}} 2 (1 + \varepsilon) \log N + o (\log N).
$$

Substituting into the inequality yields us

$$
N p _ {a} [ 1 - \Phi (z _ {a, N}) ] \geq C ^ {\prime \prime} N ^ {1 - (1 + \varepsilon) \sigma_ {b} ^ {2} / \sigma_ {a} ^ {2}} e ^ {o (\log N)}.
$$

for some constant $C ^ { \prime \prime } > 0$ . Since $\sigma _ { a } > \sigma _ { b }$ , we can choose $\varepsilon > 0$ sufficiently small so that

$$
1 - (1 + \varepsilon) \frac {\sigma_ {b} ^ {2}}{\sigma_ {a} ^ {2}} > 0,
$$

implying that $N p _ { a } [ 1 - \Phi ( z _ { a , N } ) ]  \infty$ , and hence

$$
(1 - p _ {a} [ 1 - \Phi (z _ {a, N}) ]) ^ {N} \to 0.
$$

Combining with the analysis for b, we finally obtain

$$
\operatorname * {P r} (\tilde {M} _ {a} ^ {(N)} \leq \tilde {M} _ {b} ^ {(N)}) \to 0 (N \to \infty).
$$

This completes the proof.

![](images/468bc29098d307b6c5316802315757f5ba440301e8e8e9531a5e0c679d61d16e.jpg)

We are now ready to prove Theorem 3.3.

Proof of Theorem 3.3. Consider the case where $\sigma _ { 1 } ~ > ~ \sigma _ { k }$ for all $k = 2 , \ldots , K$ . Following an analogous argument as in the proof of Theorem 3.1, we use Proposition A.3 to have

$$
\begin{array}{l} \operatorname * {P r} (\text { PRM\_Max   picks } y _ {1}) \geq 1 - \sum_ {k = 2} ^ {K} \inf _ {t \in \mathbb {R}} \left\{(1 - p _ {1} [ 1 - \Phi (\frac {t - \mu_ {1}}{\sigma_ {1}}) ]) ^ {N} \right. \\ \left. + 1 - (1 - p _ {k} [ 1 - \Phi (\frac {t - \mu_ {k}}{\sigma_ {k}}) ]) ^ {N} \right\}, \\ \end{array}
$$

where Φ is the cumulative distribution function of the standard normal distribution. By Proposition A.3, each term in the summation tends to zero as $N  \infty$ . Therefore the probability that PRM Max correctly selects $y _ { 1 }$ tends to 1.

In the alternative case, suppose there exists some k such that $\sigma _ { k } > \sigma _ { 1 }$ Following an analogous argument as in the proof of Theorem 3.1, we use Proposition A.3 to have

$$
\begin{array}{l} \operatorname * {P r} (\text { PRM\_Max   picks } y _ {1}) \leq \inf _ {t \in \mathbb {R}} \left\{(1 - p _ {k} [ 1 - \Phi (\frac {t - \mu_ {k}}{\sigma_ {k}}) ]) ^ {N} \right. \\ \left. + 1 - (1 - p _ {1} [ 1 - \Phi (\frac {t - \mu_ {1}}{\sigma_ {1}}) ]) ^ {N} \right\}. \\ \end{array}
$$

Furthermore, by Proposition A.3, the bound tends to 0 when $N \to \infty$ .

This completes the proof.

![](images/c089045b145dad2f1e14fecf2c4e1b99c6a8c9f9b3665d3568ffbab9a05ffb13.jpg)

# B Universe of Methods

In this section, we provide a systematic description of our universe of methods M as mentioned in Section 5. Each method is represented by a tuple (LM, ReStrat, Agg, Conf, N), where each component is defined as follows.

Language Model (LM). For our main experiments, we consider a single base model: Qwen2.5-Math-7B-Instruct. In Appendix 5.2.1, we extend our pool of methods by additionally using Qwen2.5-Math-1.5B-Instruct.

Reasoning Strategies (ReStrat). We consider four primary groups of reasoning methods:

• Greedy Search: The deterministic strategy denoted as COT-G that always selects the highestprobability next token.   
• Best-of-N: The model samples N complete responses for each question according to the output token distribution, controlled by a temperature hyperparameter. The temperature adjusts the randomness of sampling: lower temperatures make the output more deterministic, while higher temperatures increase diversity. If N = 1, this produces a single, randomly sampled response. When $N > 1$ , aggregation strategies (described below) are applied to select a final answer from the N candidates.   
• Beam Search: The model first generates N distinct first steps, each evaluated by a PRM. The top N/m steps with the highest PRM scores are kept, where $N / m \in \mathbb { Z }$ . For each retained first step, the model generates m second steps, forming N partial solutions. This process repeats until N complete solutions are produced or we reach the maximum number of beam expansions (50 in our case).   
• Monte Carlo Tree Search: This method, denoted as MCTS, formulates response generation as a tree search problem, iteratively exploring possible answers by balancing exploration and exploitation. At each step, MCTS selects the most promising node based on a selection policy (e.g., Upper Confidence Bound), expands new response candidates using the base model, evaluates them through rollouts or verifier models, and back-propagates the scores to refine future selections. The search continues until we obtain N complete solutions.

Configuration (Conf).The configuration presents all the hyperparameters of a reasoning strategy. While almost all of them are fixed as default values suggested by the language model report and repository OpenR [Wang et al., 2024a], we vary the temperature of decoding in the set {0.4, 0.7, 1.0} to explore different levels of diversity.

Aggregation methods (Agg). For reasoning strategies that produce multiple candidate responses, we employ aggregation methods to select the final answer. These methods include:

• Majority Vote (MV): Select the most frequently occurring answer from multiple samples.   
• PRM Vote Min (PVM): For each generation, use the PRM to score each step, select the minimum score within the generation, and ultimately choose the generation with the highest sum of minimum score across all samples.   
• PRM Vote Last (PVL): For each generation, use the PRM to score each step, select the score associated with generating the last step, and ultimately choose the generation with the highest sum of scores across all samples   
• PRM Max Min (PMM): For each generation, use the PRM to score each step, select the minimum score within the generation, and ultimately choose the generation with the highest minimum score across all samples.   
• PRM Max Last (PML): For each generation, use the PRM to score each step, select the score associated with generating the last step, and ultimately choose the generation with the highest score across all samples.

Notably, except for Majority Vote, other methods are required to call the reward model.

Candidate solution size (N). We vary the number of candidate solutions in N ∈ {1, 2, 4, 8, 16}.

Overall, the universe M consists of 81 reasoning models that span different reasoning strategies and configurations. The detailed composition of the universe M is summarized in Table 3. Specifically, we construct M by combining reasoning strategies, decoding settings, and aggregation methods. The full enumeration yields 81 unique configurations: 60 Best-of-N variants, 10 Beam Search variants, 10 MCTS variants, and one deterministic CoT-G configuration.

Table 3: Construction of the universe M (81 methods) by strategy, search budget N, decoding temperature temp, and aggregation Agg ∈ {MV, PVM, PVL, PMM, PML}. 

<table><tr><td>Strategy</td><td>N values</td><td>temp values</td><td>#Agg choices</td><td>Count</td></tr><tr><td>Best-of-N</td><td>{2, 4, 8, 16}</td><td>{0.4, 0.7, 1.0}</td><td>5</td><td> $4 \times 3 \times 5 = 60$ </td></tr><tr><td>Beam Search</td><td>{2, 4}</td><td>0.5</td><td>5</td><td> $2 \times 5 = 10$ </td></tr><tr><td>MCTS</td><td>{2, 4}</td><td>0.5</td><td>5</td><td> $2 \times 5 = 10$ </td></tr><tr><td>CoT-G (greedy)</td><td>1</td><td>(greedy)</td><td>N/A</td><td>1</td></tr><tr><td colspan="3"></td><td>Total</td><td>81</td></tr></table>

# C Baseline Descriptions and Complete Experimental Results

We present an additional ablation study focusing on another critical component of our framework: the design of the reasoning selection module. In particular, we replace our contrastive-learning and two-tower-based reasoning selection model with four baselines. We first consider two baselines that do not support an accuracy-cost trade-off:

• Offline Ada-BoK adapts the approach from Damani et al. [2025], which originally operates on batches of questions and manages resources at the batch level. For fair comparison, we use their ‘Offline allocation’ strategy, modified to operate at the individual question level without requiring access to the entire batch at test time. Here, each allocation corresponds to choosing a specific reasoning configuration.   
• Random Allocation (RA): At inference time, each question randomly selects a reasoning method from the available reasoning configurations.

The next two baselines, along with our method, support an explicit accuracy-cost trade-off, controlled by the hyperparameter λ, which we vary over {0.0, 0.25, 0.5, 0.75, 1.0}.

• Multi-class classifier (CL-λ). In this classifier-based version, the best reasoning method for each question is still determined using the utility-driven labeling (based on the parameter λ), exactly as in our original approach. However, instead of using contrastive loss with a two-tower embedding structure, we adopt a two-layer classifier placed on top of the pretrained sentence transformer network and train with a standard cross-entropy loss, commonly used in classification scenarios. We denote this classification-based ablation as CL-λ.

• Distributional Random Allocation (DRA-λ): Parameter λ matches the trade-off parameter used in our approach. During training, each question is labeled with the reasoning method having the highest λ-adjusted score prediction. At inference, reasoning methods are randomly drawn from the observed training distribution.

Table 4: Average accuracy and the number of generated tokens on MATH500 for different methods and models. Methods above the blue line are either Upper Bound under M or not in M. Methods above the green line are individual single-reasoning configurations (no selection module involved). Methods above the brown line do not support accuracy-cost trade-off. Below, we compare EPIC-λ, DRA-λ, and CL-λ at different trade-off settings (groups separated by gray lines). Best results in each section are in bold. 

<table><tr><td>Base Model</td><td>Method</td><td>Accuracy ↑</td><td>Average Token Count ↓</td></tr><tr><td>Qwen2.5 72B Base</td><td>CoT* [Yang et al., 2024b]</td><td>80.0</td><td>-</td></tr><tr><td>QwQ 32B</td><td>CoT* [Yang et al., 2024b]</td><td>83.2</td><td>-</td></tr><tr><td>OpenAI-o1-mini</td><td>CoT* [Jaech et al., 2024]</td><td>90.0</td><td>-</td></tr><tr><td>Deepseek-V3</td><td>CoT* [Liu et al., 2024]</td><td>90.2</td><td>-</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>Upper Bound under M</td><td>91.2</td><td>-</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>CoT-G</td><td>83.2</td><td>620.4</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>Best-of-2</td><td>84.8</td><td>1242.5</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>Best-of-4</td><td>86.2</td><td>2499.4</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>Best-of-8</td><td>86.6</td><td>4986.8</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>Best-of-16</td><td>86.8</td><td>10036.2</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>MCTS</td><td>85.4</td><td>4338.1</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>Beam-search</td><td>85.2</td><td>2638.1</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>RA</td><td>84.4</td><td>1752.4</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>Offline Ada-BoK</td><td>87.0</td><td>4095.2</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>DRA-0.0</td><td>85.6</td><td>1248.4</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>CL-0.0</td><td>85.2</td><td>606.7</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>EPIC-0.0</td><td>85.8</td><td>892.9</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>DRA-0.25</td><td>86.2</td><td>2453.6</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>CL-0.25</td><td>86.0</td><td>2275.6</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>EPIC-0.25</td><td>86.4</td><td>1859.2</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>DRA-0.5</td><td>86.4</td><td>5719.3</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>CL-0.5</td><td>86.6</td><td>5320.2</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>EPIC-0.5</td><td>86.8</td><td>2482.6</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>DRA-0.75</td><td>86.4</td><td>7523.2</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>CL-0.75</td><td>87.0</td><td>7524.6</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>EPIC-0.75</td><td>87.6</td><td>3192.9</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>DRA-1.0</td><td>87.0</td><td>10542.2</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>CL-1.0</td><td>87.8</td><td>10923.4</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>EPIC-1.0</td><td>89.4</td><td>6921.7</td></tr></table>

∗Method not in M.

We observe from Table 4 that our EPIC-λ strongly outperforms the simpler classification version (CL-λ) and DRA-λ at almost every level of the parameter λ. Still, the CL-λ method remains superior to individual single reasoning configurations, confirming that our labeling strategy based on the proposed utility function is indeed effective. Moreover, our original contrastive-learning approach with a two-tower embedding structure significantly enhances scalability: introducing new reasoning methods simply involves adding a new embedding vector without retraining the entire selection module. This two-tower model and contrastive loss combination have proven highly advantageous over classification-based methods, both in terms of scalability and overall predictive performance.

# D Ablation Studies and Additional Experiments

This section presents additional ablation studies to investigate the impact of various design choices in our framework. First, we examine the effect of representation embeddings and the cost-accuracy trade-off parameter λ. Second, we assess the robustness of our approach by substituting the utility function (5) with an alternative formulation. Third, we evaluate the transferability of our framework across different language models and other in-domain datasets. Finally, we present additional experiments for the code generation task.

# D.1 The Impact of the Representation Dimension

In this experiment, we study the impact of the dimension of the representation space d on the performance of EPIC. One could expect that larger dimensions d will give a higher representation power and thus EPIC could perform better. For simplicity, we conduct the experiments only for $\bar { \lambda } = 0 . 2 5$ . The average test accuracy and token counts are reported in Table 5. We could identify a global trend that, as d increases, the accuracy increases, while the average token count tends to go flat or decrease. This result empirically confirms the expectation that increasing the embedding dimension could increase the ensemble’s performance.

Table 5: Impact of d on test accuracy and average token counts with $\lambda = 0 . 2 5$ . 

<table><tr><td>d</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td>Accuracy ↑</td><td>85.6</td><td>85.4</td><td>86.4</td><td>86.2</td></tr><tr><td>Average token counts ↓</td><td>1828.3</td><td>2271.4</td><td>1859.2</td><td>2004.5</td></tr></table>

# D.2 The Impact of the Trade-off Parameter λ

Table 6 illustrates the impact of λ on accuracy and the number of generated tokens. As λ increases from 0.00 to 1.00, we observe a consistent rise in accuracy from 85.8% to 89.4% and in average token counts from 892.9 to 6921.7. This trend indicates a clear cost-accuracy trade-off and can be visualized in Figure 2, where we can identify an upward trend of the red circles.

Table 6: Impact of λ on test accuracy and average token counts with embedding dimension $d = 6 4$ . 

<table><tr><td>λ</td><td>0.00</td><td>0.25</td><td>0.50</td><td>0.75</td><td>1.00</td></tr><tr><td>Accuracy ↑</td><td>85.8</td><td>86.4</td><td>86.8</td><td>87.6</td><td>89.4</td></tr><tr><td>Average token counts ↓</td><td>892.9</td><td>1859.2</td><td>2482.6</td><td>3192.9</td><td>6921.7</td></tr></table>

# D.3 Ablation on Utility Function

We previously presented two ablation analyses on the embedding dimensionality in Section D.1 and the trade-off parameter λ in Section D.2. To further justify our choice of utility function, we conduct an additional ablation study using an alternative utility formulation:

$$
u (a _ {i, j}, c _ {i, j}) = a _ {i, j} ^ {\lambda} \times (1 - c _ {i, j}) ^ {1 - \lambda},
$$

where $a _ { i , j }$ denotes accuracy and $c _ { i , j }$ denotes cost for the j-th method on the i-th instance. We vary the trade-off parameter λ over the set {0.0, 0.25, 0.5, 0.75, 1.0}.

Notably, for $\lambda = 0 . 0$ and $\lambda = 1 . 0$ , this alternative function reduces to our original utility formulation. Therefore, we focus our comparison on the intermediate values $\lambda \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 \}$ . For a better presentation, we denote our framework with this alternative utility function as PMU (Power Mean Utility).

As shown in Table 7, EPIC consistently outperforms the DRA baseline across all evaluated PMU configurations, demonstrating its robustness under different accuracy-cost trade-offs.

Table 7: Comparison of PMU and EPIC methods at different λ settings. Best results in each section are in bold. 

<table><tr><td>Method</td><td>Accuracy ↑</td><td>Average Token Count ↓</td></tr><tr><td>PMU-0.25</td><td>86.0</td><td>2334.1</td></tr><tr><td>EPIC-0.25</td><td>86.4</td><td>1859.2</td></tr><tr><td>PMU-0.5</td><td>86.4</td><td>3035.4</td></tr><tr><td>EPIC-0.5</td><td>86.8</td><td>2482.6</td></tr><tr><td>PMU-0.75</td><td>87.2</td><td>4724.5</td></tr><tr><td>EPIC-0.75</td><td>87.6</td><td>3192.9</td></tr></table>

# D.4 Results on Code Benchmark

To assess the generality of our proposed method beyond the math domain, we evaluate it on Live-CodeBench [Jain et al., 2025]. The universe of methods includes both the Chain of Thought - Greedy (CoT-G) and Best-of-N sampling strategies. In the Best-of-N approach, the base model generates N candidate responses per question, from which the best is selected, with $N \in \{ 2 , 4 , 8 , 1 6 \}$ and the decoding temperature chosen from 0.2, 0.6. CoT-G produces step-by-step solutions using greedy decoding (temperature set to 0). For evaluation, we adopt the pass@k metric, as described in Li et al. [2025b], and test two LLM base models of differing capacities: Qwen2.5-Coder-3B-Instruct and Qwen2.5-Coder-7B-Instruct [Yang et al., 2024b]. In the code benchmark, we do not consider aggregation methods, so the regularization parameter τ is set to 0 in this experiment.

Table 8 reports the accuracy (pass@k) and average token counts for each method. We observe that our method, EPIC, achieves competitive or superior accuracy to all baselines at both λ = 0.25 and λ = 1.0. For the 7B model, EPIC (λ = 1.0) achieves the highest overall accuracy (61.88%), outperforming Best-of-16, while also consuming fewer tokens. Similarly, on the 3B model, EPIC (λ = 1.0) achieves the best accuracy (48.01%), exceeding the Best-of-16 baseline.

Trade-off Control. EPIC with λ = 0.25 achieves balanced performance, providing better accuracy than CoT-G and Best-of-2, but at a modest computational cost, highlighting the framework’s flexibility in managing the trade-off of accuracy and cost.

Efficiency at Lower Cost. Notably, CoT-G remains the most computationally efficient method, but at the expense of lower accuracy. EPIC offers a favorable balance, substantially improving accuracy while keeping generation costs well below those of aggressive sampling strategies like Best-of-16.

Table 8: Performance comparison of reasoning methods on Qwen2.5-Coder-3B-Instruct and Qwen2.5- Coder-7B-Instruct. The best value in each column is in bold. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen2.5-Coder-3B-Instruct</td><td colspan="2">Qwen2.5-Coder-7B-Instruct</td></tr><tr><td>Accuracy ↑</td><td>Avg. Token Count ↓</td><td>Accuracy ↑</td><td>Avg. Token Count ↓</td></tr><tr><td>CoT - Greedy</td><td>24.85</td><td>580.80</td><td>35.81</td><td>505.15</td></tr><tr><td>Best-of-2 (with best temperature)</td><td>27.40</td><td>1144.73</td><td>41.68</td><td>1000.27</td></tr><tr><td>Best-of-4 (with best temperature)</td><td>32.88</td><td>2299.65</td><td>48.53</td><td>2011.99</td></tr><tr><td>Best-of-8 (with best temperature)</td><td>40.90</td><td>4672.83</td><td>53.82</td><td>4031.65</td></tr><tr><td>Best-of-16 (with best temperature)</td><td>46.38</td><td>9323.22</td><td>58.71</td><td>8034.65</td></tr><tr><td>EPIC  $\lambda = 0.25$ </td><td>30.12</td><td>1025.43</td><td>43.44</td><td>813.42</td></tr><tr><td>EPIC  $\lambda = 1.0$ </td><td>48.01</td><td>8349.12</td><td>61.88</td><td>7013.54</td></tr></table>

# E Qualitative Results

# E.1 A Specific Reasoning Method Favors in Certain Questions

This appendix provides a curated example in Table 9 for qualitative analysis. The goal is to present a case where some individual methods in the universe M fail, but EPIC still produces accurate answers by selecting the most suitable one.

# E.2 Visualization and Analysis of Learned Embedding Space

![](images/bb06c86f5192d54377b2c540f0b0cf7a952c1a6ab9316bc19f25c2698060d49b.jpg)

<details>
<summary>scatter</summary>

PCA Visualization of Question and Reasoning Embeddings
| Label | PCA Dimension 1 | PCA Dimension 2 |
|---|---|---|
| 60:1 | -0.95 | 0.18 |
| 34:4 | -0.55 | -0.12 |
| 24:8 | -0.25 | 0.12 |
| 6:16 | -0.25 | 0.05 |
| 8:16 | -0.25 | 0.02 |
| 5:02 | -0.25 | 0.01 |
| 4:16 | -0.25 | -0.01 |
| 4:26 | 0.45 | 0.45 |
| 3:16 | -0.25 | -0.01 |
| 3:16 | -0.25 | -0.02 |
| 3:8 | -0.25 | -0.22 |
| 3:76 | -0.25 | -0.28 |
| 3:76 | -0.25 | -0.32 |
| 3:8 | -0.25 | -0.34 |
| 19:16 | -0.25 | -0.42 |
| 18:16 | -0.25 | -0.44 |
| 39:4 | -0.15 | -0.58 |
| 58:2 | 0.25 | 0.52 |
| 59:2 | 0.05 | 0.18 |
| 29:8 | 0.35 | -0.05 |
| 47:4 | 0.35 | -0.38 |
| 25:8 | 0.35 | -0.38 |
The image displays a scatter plot with two distinct clusters of data points (gray and red) representing different classes or categories in the PCA space defined by PCA Dimension 1 and PCA Dimension 2.
</details>

(a) λ = 0.25

![](images/446e86551d4ca28cd5c273266a4ca7b3252325c297eebb2ba17600ba2c3281a5.jpg)

<details>
<summary>scatter</summary>

PCA Visualization of Question and Reasoning Embeddings
| Label | PCA Dimension 1 | PCA Dimension 2 |
|---|---|---|
| 4:2 | -0.35 | 0.38 |
| 58:2 | -0.15 | 0.52 |
| 29:8 | -0.25 | 0.22 |
| 47:4 | -0.22 | -0.18 |
| 25:8 | -0.28 | -0.26 |
| 59:2 | 0.35 | 0.18 |
| 63:6 | 0.38 | 0.24 |
| 10:2 | 0.36 | 0.16 |
| 11:2 | 0.34 | 0.14 |
| 12:1 | 0.33 | 0.12 |
| 13:1 | 0.32 | 0.10 |
| 14:1 | 0.31 | 0.08 |
| 16:1 | 0.30 | 0.06 |
| 17:1 | 0.29 | 0.04 |
| 18:1 | 0.28 | -0.34 |
| 19:1 | 0.27 | -0.36 |
| 34:4 | 0.70 | -0.02 |
| 35:4 | 0.72 | -0.38 |
| 36:4 | 0.75 | -0.40 |
| 39:4 | 0.78 | -0.78 |
| 42:1 | 0.76 | -0.24 |
| 43:1 | 0.74 | -0.26 |
| 44:1 | 0.72 | -0.28 |
| 45:1 | 0.70 | -0.30 |
| 46:1 | 0.68 | -0.32 |
| 47:1 | 0.66 | -0.34 |
| 48:1 | 0.64 | -0.36 |
| 49:1 | 0.62 | -0.38 |
| 50:1 | 0.60 | -0.40 |
| 51:1 | 0.58 | -0.42 |
| 52:1 | 0.56 | -0.44 |
| 53:8 | 0.54 | -0.46 |
| 54:8 | 0.52 | -0.48 |
| 55:8 | 0.50 | -0.50 |
| 56:8 | 0.48 | -0.52 |
| 57:8 | 0.46 | -0.54 |
| 58:8 | -0.15 | 0.56 |
| 59:8 | -0.18 | 0.58 |
| 60:1 | 1.05 | -0.22 |
The image displays a scatter plot with two distinct clusters of data points (likely representing different categories or groups) plotted against PCA dimensions for each cluster.
</details>

(b) λ = 0.5   
![](images/1085949c44f3cb43b3f9e9b3c55aaffc8c4bcb0dcf48498907e5bc6e8d203748.jpg)

<details>
<summary>scatter</summary>

PCA Visualization of Question and Reasoning Embeddings
| PCA Dimension 1 | PCA Dimension 2 |
| :--- | :--- |
| -0.15 | 0.65 |
| 0.05 | 0.58 |
| 0.15 | 0.42 |
| 0.25 | 0.38 |
| 0.35 | 0.32 |
| 0.45 | 0.28 |
| 0.55 | 0.22 |
| 0.65 | 0.18 |
| 0.75 | 0.12 |
| 0.85 | 0.08 |
| 0.95 | 0.02 |
| 1.05 | -0.02 |
| 1.15 | -0.08 |
| 1.25 | -0.12 |
| 1.35 | -0.18 |
| 1.45 | -0.22 |
| 1.55 | -0.28 |
| 1.65 | -0.32 |
| 1.75 | -0.38 |
| 1.85 | -0.42 |
| 1.95 | -0.48 |
| 2.05 | -0.52 |
| 2.15 | -0.58 |
| 2.25 | -0.62 |
| 2.35 | -0.68 |
| 2.45 | -0.72 |
| 2.55 | -0.78 |
| 2.65 | -0.82 |
| 2.75 | -0.88 |
| 2.85 | -0.92 |
| 2.95 | -0.98 |
| 3.05 | -1.02 |
| 3.15 | -1.08 |
| 3.25 | -1.12 |
| 3.35 | -1.18 |
| 3.45 | -1.22 |
| 3.55 | -1.28 |
| 3.65 | -1.32 |
| 3.75 | -1.38 |
| 3.85 | -1.42 |
| 3.95 | -1.48 |
| 4.05 | -1.52 |
| 4.15 | -1.58 |
| 4.25 | -1.62 |
| 4.35 | -1.68 |
| 4.45 | -1.72 |
| 4.55 | -1.78 |
| 4.65 | -1.82 |
| 4.75 | -1.88 |
| 4.85 | -1.92 |
| 4.95 | -1.98 |
| 5.05 | -2.02 |
| 5.15 | -2.08 |
| 5.25 | -2.12 |
| 5.35 | -2.18 |
| 5.45 | -2.22 |
| 5.55 | -2.28 |
| 5.65 | -2.32 |
| 5.75 | -2.38 |
| 5.85 | -2.42 |
| 5.95 | -2.48 |
| 6.05 | -2.52 |
| 6.15 | -2.58 |
| 6.25 | -2.62 |
| 6.35 | -2.68 |
| 6.45 | -2.72 |
| 6.55 | -2.78 |
| 6.65 | -2.82 |
| 6.75 | -2.88 |
| 6.85 | -2.92 |
| 6.95 | -2.98 |
| 7.05 | -3.02 |
| 7.15 | -3.08 |
| 7.25 | -3.12 |
| 7.35 | -3.18 |
| 7.45 | -3.22 |
| 7.55 | -3.28 |
| 7.65 | -3.32 |
| 7.75 | -3.38 |
| 7.85 | -3.42 |
| 7.95 | -3.48 |
| 8.05 | -3.52 |
| 8.15 | -3.58 |
| 8.25 | -3.62 |
| 8.35 | -3.68 |
| 8.45 | -3.72 |
| 8.55 | -3.78 |
| 8.65 | -3.82 |
| 8.75 | -3.88 |
| 8.85 | -3.92 |
| 8.95 | -3.98 |
| 9.05 | -4.02 |
| 9.15 | -4.08 |
| 9.25 | -4.12 |
| 9.35 | -4.18 |
| 9.45 | -4.22 |
| 9.55 | -4.28 |
| 9.65 | -4.32 |
| 9.75 | -4.38 |
| 9.85 | -4.42 |
| 9.95 | -4.48 |
| 10.05 | -4.52 |
| 10.15 | -4.58 |
| 10.25 | -4.62 |
| 10.35 | -4.68 |
| 10.45 | -4.72 |
| 10.55 | -4.78 |
| 10.65 | -4.82 |
| 10.75 | -4.88 |
| 10.85 | -4.92 |
| 10.95 | -4.98 |
| 11.05 | -5.02 |
| 11.15 | -5.08 |
| 11.25 | -5.12 |
| 11.35 | -5.18 |
| 11.45 | -5.22 |
| 11.55 | -5.28 |
| 11.65 | -5.32 |
| 11.75 | -5.38 |
| 11.85 | -5.42 |
| 11.95 | -5.48 |
| 12.05 | -5.52 |
| 12.15 | -5.58 |
| 12.25 | -5.62 |
| 12.35 | -5.68 |
| 12.45 | -5.72 |
| 12.55 | -5.78 |
| 12.65 | -5.82 |
| 12.75 | -5.88 |
| 12.85 | -5.92 |
| 12.95 | -5.98 |
| 13.05 | -6.02 |
| 13.< img src="color_formula_zo"
        size=(color_formula_zo)
        color=(color_formula_zo)
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        color=color=color
        style=(color_formula_zo):
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
        colored = color
</details>

(c) λ = 1.0   
Figure 3: PCA visualization of question (grey) and reasoning method (colored crosses) embeddings for three different settings of the utility trade-off λ. Each method is labeled by index:num, where num is the number of generated answers.

To further understand how EPIC organizes and exploits the structure of mathematical reasoning methods and questions, we visualize the learned embedding space using Principal Component Analysis (PCA). Figure 3 presents three PCA plots of the question and method embeddings for different utility trade-off values: $\lambda = 0 . 2 5 , \lambda = 0 . 5$ , and $\lambda = 1 . 0 .$ In these plots, the grey dots represent the embedded math questions, while each colored cross denotes a reasoning method. Each method is annotated with the format index:num, where index is the method identifier and num is the number of generated answers for that method.

Plot 3 illustrate how the utility trade-off parameter λ shapes the structure of the learned embedding space:

When $\lambda = 0 . 2 5$ (cost prioritized), method embeddings are widely scattered and tend to avoid regions dense with question embeddings. Only low-cost methods are positioned near clusters of questions.

When λ = 0.5 (equal weight to accuracy and cost), higher-cost but occasionally effective methods move closer to question clusters. This reflects a balanced trade-off, where moderately accurate and moderately costly methods are preferred.

When λ = 1 (accuracy prioritized), most methods form distinct clusters around their optimal question types. This demonstrates EPIC’s ability to match each question with the most suitable method.

These PCA visualizations confirm that EPIC organizes method and question representations according to the chosen trade-off. This structured alignment helps explain EPIC’s strong performance discussed earlier in the paper.

# Reference Q&A

Q: Evaluate

$$
\sin (\arcsin 0. 4 + \arcsin 0. 5) \cdot \sin (\arcsin 0. 5 - \arcsin 0. 4)
$$

A: From the angle addition and subtraction formulas,

$$
\sin (x + y) = \sin x \cos y + \cos x \sin y, \quad \sin (x - y) = \sin x \cos y - \cos x \sin y,
$$

we get sin(x + y) sin(x − y) = (sin x cos y + cos x sin y)(sin x cos y − cos x sin y) = sin2 x − $\sin ^ { 2 } y .$ For x = arcsin 0.5 and $y = \arcsin 0 . 4 ,$

$$
\sin (\arcsin 0. 5 + \arcsin 0. 4) \cdot \sin (\arcsin 0. 5 - \arcsin 0. 4) = 0. 5 ^ {2} - 0. 4 ^ {2} = 0. 0 9 = \boxed {\frac {9}{1 0 0}}.
$$

# Best-of-16 MV

To evaluate sin(arcsin 0.4 + arcsin 0.5) · sin(arcsin $0 . 5 - \arcsin 0 . 4 ) .$ , let θ = arcsin 0.4 and ϕ = arcsin 0.5, so sin θ = 0.4 and sin $\phi = 0 . 5$ . We need to find si $1 ( \theta + \phi ) \cdot \sin ( \phi - \theta )$ .

Using the sine addition and subtraction formulas:

$$
\sin (\theta + \phi) = \sin \theta \cos \phi + \cos \theta \sin \phi , \quad \sin (\phi - \theta) = \sin \phi \cos \theta - \cos \theta \sin \phi ,
$$

we get:

$$
\sin (\theta + \phi) \cdot \sin (\phi - \theta) = (\sin \theta \cos \phi) ^ {2} - (\cos \theta \sin \phi) ^ {2}.
$$

Now, using $\cos ^ { 2 } \theta = 1 - \sin ^ { 2 } \theta = 0 . 8 4$ and cos $^ { 2 } \phi = 1 - \sin ^ { 2 } \phi = 0 . 7 5 .$ , the expression becomes:

$$
(0. 4 \cdot \sqrt {0 . 7 5}) ^ {2} - (\sqrt {0 . 8 4} \cdot 0. 5) ^ {2} = 0. 1 6 \cdot 0. 7 5 - 0. 2 5 \cdot 0. 8 4 = 0. 1 2 - 0. 2 1 = - 0. 0 9.
$$

Thus, the value of the expression is:

$$
\boxed {- 0. 0 9}.
$$

# EPIC selects Best-of-16 PVL

To evaluate sin(arcsin $0 . 4 + \arcsin 0 . 5 )$ · sin(arcsin $0 . 5 - \arcsin 0 . 4 )$ , let α = arcsin 0.4 and β = arcsin 0.5, so sin $\alpha = 0 . 4$ and sin $\beta = 0 . 5$ . The expression becomes sin $\left( \alpha + \beta \right) \cdot \sin ( \beta - \alpha )$

Using the product-to-sum identity for sines:

$$
\sin x \cdot \sin y = \frac {1}{2} [ \cos (x - y) - \cos (x + y) ],
$$

we apply it to $x = \alpha + \beta$ and $y = \beta - \alpha ,$ yielding:

$$
\sin (\alpha + \beta) \cdot \sin (\beta - \alpha) = \frac {1}{2} [ \cos 2 \alpha - \cos 2 \beta ].
$$

Using the double-angle identity cos $\begin{array} { r l r } { 2 \theta } & { { } = } & { 1 - 2 \sin ^ { 2 } \theta , } \end{array}$ we get: cos $\begin{array} { r l r } { 2 \alpha } & { { } = } & { 1 { \it \Delta \phi } - } \end{array}$ 2 sin2 α and cos $2 { \breve { \beta } } = 1 - 2 \sin ^ { 2 } \beta .$ . Substituting these, the expression becomes:

$$
\sin (\alpha + \beta) \cdot \sin (\beta - \alpha) = \frac {1}{2} \left[ (1 - 2 \sin^ {2} \alpha) - (1 - 2 \sin^ {2} \beta) \right].
$$

Simplifying:

$$
= \frac {1}{2} \cdot 2 (\sin^ {2} \beta - \sin^ {2} \alpha) = \sin^ {2} \beta - \sin^ {2} \alpha .
$$

Substituting sin α = 0.4 and sin $\beta = 0 . 5 \colon \sin ^ { 2 } \beta - \sin ^ { 2 } \alpha = ( 0 . 5 ) ^ { 2 } - ( 0 . 4 ) ^ { 2 } = 0 . 2 5 - 0 . 1 6 = 0 . 0 9 .$

Therefore, the value of the expression is: $\boxed { 0 . 0 9 } .$

Table 9: A curated sample from the MATH benchmark. This is an example where an individual method fails, but EPIC could deliver an accurate answer by selecting the most suitable one.