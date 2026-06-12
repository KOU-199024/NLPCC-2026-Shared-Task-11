# DATA-AWARE AND SCALABLE SENSITIVITY ANALY-SIS FOR DECISION TREE ENSEMBLES

Namrita Varshney, Ashutosh Gupta, Arhaan Ahmad, Tanay V. Tayal, & S. Akshay

Department of Computer Science and Engineering,

Indian Institute of Technology Bombay, Mumbai, India.

namrita@iitb.ac.in, akg@cse.iitb.ac.in, arhaan.ahmad2003@gmail.com,

tanaytayal@cse.iitb.ac.in, akshayss@cse.iitb.ac.in

# ABSTRACT

Decision tree ensembles are widely used in critical domains, making robustness and sensitivity analysis essential to their trustworthiness. We study the feature sensitivity problem, which asks whether an ensemble is “sensitive” to a specified subset of features - such as protected attributes- whose manipulation can alter model predictions. Existing approaches often yield examples of sensitivity that lie far from the training distribution, limiting their interpretability and practical value. We propose a data-aware sensitivity framework that constrains the sensitive examples to remain close to the dataset, thereby producing realistic and interpretable evidence of model weaknesses. To this end, we develop novel techniques for dataaware search using a combination of mixed-integer linear programming (MILP) and satisfiability modulo theories (SMT) encodings. Our contributions are fourfold. First, we strengthen the NP-hardness result for sensitivity verification, showing it holds even for trees of depth 1. Second, we develop MILP-optimizations that significantly speed up sensitivity verification for single ensembles and, for the first time, can also handle multiclass tree ensembles. Third, we introduce a data-aware framework that generates realistic examples close to the training distribution. Finally, we conduct an extensive experimental evaluation on large tree ensembles, demonstrating scalability to ensembles with up to 800 trees of depth 8, achieving substantial improvements over the state of the art. This framework provides a practical foundation for analyzing the reliability and fairness of tree-based models in high-stakes applications.

# 1 INTRODUCTION

Decision tree ensembles are a popular AI model, known for their simplicity, power, and interpretability. They are ubiquitous across multiple industries, ranging from banking (Chang et al., 2018; Madaan et al., 2021) and healthcare (Ghiasi & Zendehboudi, 2021; Kelarev et al., 2012) to water resources engineering (Niazkar et al., 2024) and telecommunication (Shrestha & Shakya, 2022). Given that this class of models forms a cornerstone for automated decision-making in various industries, it is important to be able to trust their answers and provide guarantees on their reliability. Towards this goal, there has been significant research in the past decade on formalizing and verifying various safety properties of tree ensembles.

In this paper, we focus on one such problem: understanding the influence that a particular subset of input features can have on the output of a decision tree classifier. This notion of sensitivity of the model to a feature set has been studied in various contexts in previous works. It has been related to individual fairness and causal discrimination (Dwork et al., 2011; Calzavara et al., 2022; Galhotra et al., 2017; Blockeel et al., 2023), which are central to building responsible AI systems. A model is called sensitive to a specified set of features if the output of the model can be changed by keeping every other feature the same and varying only the specified input features. Thus the problem of feature sensitivity verification (or simply sensitivity) is to check whether a given tree ensemble model E is sensitive to a specified subset of features $F \subseteq { \mathcal { F } } ,$ , i.e, whether there exist two inputs, called a (sensitive) counterexample pair that are identical on ${ \mathcal { F } } \backslash F$ , but on which E gives different outputs. Knowledge of sensitivity to specific subsets of input features is important for understanding and mitigating attacks that aim to change model outputs by manipulating a small set of protected input features. This analysis can also help uncover unwanted patterns in the trained models that may arise from social biases in the training data.

![](images/b7a6e9c8e05664d7eedb4dc139e80d5c02665f8e1c40067643dac8edda1a12ed.jpg)

<details>
<summary>natural_image</summary>

Pixelated grayscale image with no discernible text, symbols, or structured content.
</details>

![](images/b0152799c1bc66d32df6bc0951b45ec5c1f3b8c69aed930ff77c3ba6c1b10fb2.jpg)

<details>
<summary>natural_image</summary>

Pixelated abstract pattern with no discernible text or symbols
</details>

![](images/95c974225d1ba9cb66e496f472f2462f339bea734aa6ca8821ea1902db728de7.jpg)

<details>
<summary>text_image</summary>

3
</details>

![](images/9a9d11b69441155de4b4c42e593346e9e057fa5293f2bfc07c8818f32772ea04.jpg)

<details>
<summary>text_image</summary>

3
</details>

Figure 1: Two counterexample pairs from a tree ensemble trained on MNIST. (Left) A counterexample pair where the left image is classified as 3 and the right as 8; but both are meaningless blobs. (Right) A pair closer to the training distribution. The left image is classified as 3 and the right as 8; where both resemble a 3, but the second is confidently misclassified, providing a more useful witness of sensitivity.

Recently, Ahmad et al. (2025) showed that sensitivity verification is NP-hard for ensembles with trees of maximum depth at least 3, and gave a tool utilizing a pseudo-Boolean encoding to tackle the problem for binary classifiers. However, their NP-hardness proof (via a 3-SAT reduction) does not extend to ensembles of trees with depth at most 1 or 2, leaving the hardness of the sensitivity problem open for such ensembles. Trees of depth 1, also called decision stumps, have been long studied in the literature Wang et al. (2020); Horvath et al. (2022); Mart ´ ´ınez-Munoz et al. (2007) and ˜ are of interest in several applications Chen et al. (2023); Huynh et al. (2018).

A second, major challenge is that sensitive counterexample pairs may a priori lie far from actual data points, providing only weak evidence of a model’s sensitivity. To see an illustrative example of this, consider Figure 1, where a tree ensemble trained on MNIST with 786 features yields two counterexample pairs. In the left pair, the decision flips (3 to 8), but neither image likely appears in the training set, so this does not reveal a model weakness. In contrast, the right pair is informative: the first image closely resembles a real training image (3) and is correctly classified, but modifying 20/786 features causes misclassification to 8 while remaining near the data distribution. Do similar cases occur when $| { \cal F } | = 1 2$ They do, even in tabular datasets, as illustrated in Section C in the Appendix. This raises the question: can we identify sensitive counterexample pairs that are closer to the real data distribution, enabling meaningful conclusions about model sensitivity? Such counterexamples are valuable for downstream tasks, such as retraining or hardening the model, but in this work we focus solely on their identification, which is already a challenging problem.

We start by showing that the sensitivity problem is NP-hard even for ensembles of decision tree stumps (trees of depth 1) via a novel reduction from the subset-sum problem. Next, to find counterexample pairs closer to the data, we develop two complementary strategies - one using a product of marginal distributions as an objective function and another constraint-solver based approach where we avoid regions of sparsely populated data during the search. The pseudo-Boolean approach of Ahmad et al. (2025) is difficult to extend with such objective functions and hence we revisit a mixed integer linear program (MILP) approach for sensitivity verification. However, a baseline MILP implementation, based on the original encoding of Kantchelian et al. (2016) performs worse than the pseudo-Boolean method, highlighting the challenge of this approach. We introduce novel optimizations to the MILP encoding that result in a significant speed up, making it feasible to analyze large ensembles, while guiding the search toward meaningful counterexample pairs (i.e., close to the data distribution). We also show that our new MILP encoding can be extended to obtain to-the-bestof-our-knowledge the first tool for sensitivity verification over multiclass tree ensemble classifiers. Empirically, we demonstrate the effectiveness of our approach on ensembles trained using XGBoost (Chen & Guestrin, 2016), achieving an order of magnitude improvement in runtime compared to earlier methods, as well as higher quality counterexamples, measured by their proximity to the data distribution. Thus, our main contributions are:

• We show that sensitivity verification is NP-hard even for ensembles of depth-1 trees.

• We significantly advance sensitivity verification by enabling discovery of counterexample pairs closer to the training distribution, through two complementary strategies: one using a product-of-marginals objective, and another a novel constraint-solving based approach to compute clause summaries and prune data-sparse regions in the input space.   
• We design a MILP-based encoding with key novel optimizations for sensitivity verification, implemented via a combination of MILP and SMT solvers, and - to the best of our knowledge - are the first to extend sensitivity verification to multiclass decision tree ensembles.   
• We implement our approach in a tool ENSENSE and perform extensive experiments on 18 datasets and 36 tree ensembles for binary and multiclass settings. ENSENSE can verify tree ensembles with up to 800 trees of depth 8, significantly outperforming the state of the art.

Related Work. A closely related problem is local robustness, which involves finding adversarial perturbations that can cause misclassification. In the context of decision trees, this problem was originally defined in Kantchelian et al. (2016) who showed its NP-hardness and used an MILP encoding to solve it. Since then, a rich line of work has emerged for robustness verification (Devos et al., 2021; Chen et al., 2019a; Ranzato & Zanella, 2020; Tornblom & Nadjm-Tehrani, 2019; Wang ¨ et al., 2020), using different techniques, from input-output mappings in Tornblom & Nadjm-Tehrani ¨ (2019) to abstract interpretation in Ranzato & Zanella (2020) to dynamic programming Wang et al. (2020) and clique-based approaches in the state-of-the-art tool, VERITAS (Devos et al., 2021). Most recently, Devos et al. (2024) extended the last approach to local robustness verification for multiclass tree-ensembles. While specific ideas from robustness verification are useful for sensitivity verification (and we do build on some of them), the locality of the robustness problem allows a mixture of simplifying optimizations given the knowledge of one input. In contrast, sensitivity verification involves a universal quantification over two inputs, making it a more complex problem.

# 2 PRELIMINARIES

In a classification problem, we are given an input space $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ defined over a d-dimensional feature space ${ \mathcal { F } } _ { : }$ , and an output space $\mathbf { \bar { \mathcal { Y } } } = \left\{ 0 , 1 , \dotsc , \dotsc , \mathbf { \bar { C } } - 1 \right\}$ , where C is the number of classes. We intend to learn the unknown mapping $\mathcal { E } : \mathcal { X }  \mathcal { Y }$ . For any $x \in \mathcal { X }$ , we will denote the value of feature $f \in { \mathcal { F } }$ for x as $x _ { f }$ . A decision tree is recursively defined as either a leaf node or an internal node. Each leaf node n has a leaf value n.val, which is a scalar in R (for binary classification) or a vector in $\mathbb { R } ^ { C }$ (for multiclass classification). Each internal node n consists of references to child nodes, decision trees n.yes and $n . n o .$ , and a guard n.guard, which is a linear inequality of the form $X _ { f } < \tau$ . Here, $f$ is a feature, $X _ { f }$ denotes the variable for feature $f ,$ and τ is a constant. An input $x \in \mathcal { X }$ is evaluated on the tree $\check { T }$ by a top-down traversal. For each encountered internal node $n ,$ the guard of $n ,$ say $n . g u a r d = X _ { f } < \tau$ is evaluated by substituting $X \gets x$ in the inequality. If the guard inequality evaluates to true, we move to n.yes; otherwise, we move to n.no. This process continues till we reach a leaf node $n ,$ and the output of the tree $T ( x )$ is given by n.val.

To increase the power of a single decision tree, it is common to use ensembling, where multiple decision trees are trained, and the outcomes are aggregated to reach a final decision. Formally, a tree ensemble classifier $\mathcal { E } : \mathcal { X }  \mathcal { Y }$ consists of a set $\tau$ of decision trees. The output of the ensemble is found by aggregating the outputs of each decision tree. There are three notions of outputs from a tree ensemble: $\mathrm { ( i ) } \stackrel { \smile } { \mathcal { E } } _ { c } ^ { r a w } \mathrm { ( } x \mathrm { ) }$ , which represents a linear aggregation of the outputs of the ensemble for class $c ,$ typically a sum over the outputs of all trees in the ensemble, $( \operatorname { i i } ) \mathcal { E } _ { c } ^ { p r o b } ( x ) ;$ the predicted class probability of class c and (iii) the output label $\begin{array} { r } { \mathcal { E } ( x ) = \arg \operatorname* { m a x } _ { c } \mathcal { E } _ { c } ^ { p r o b } ( x ) } \end{array}$ . In the binary classification setting, where $\mathcal { V } = \{ 0 , 1 \}$ , each leaf node stores n.val ∈ R and $\dot { \mathcal { E } } _ { 1 } ^ { r a w } ( x ) =$ $\begin{array} { r } { \sum _ { T \in \mathcal { T } } T ( x ) , \mathcal { E } _ { 0 } ^ { r a w } ( x ) = - \mathcal { E } _ { 1 } ^ { r a w } ( x ) , \mathcal { E } _ { 1 } ^ { p r o b } ( x ) = } \end{array}$ SIGMOID $( \mathcal { E } _ { 1 } ^ { r a w } ( x ) ) , \mathcal { E } _ { 0 } ^ { p r o b } ( x ) = 1 - \mathcal { E } _ { 1 } ^ { p r o b } ( x )$ , where SIGMOID : R → R is the sigmoid function defined as $\mathrm { S I G M O I D } ( x ) \ : = \ : 1 / ( 1 + e ^ { - x } )$ . For convenience, a glossary of the notation used throughout the paper is provided in Appendix A.

# 3 FEATURE SENSITIVITY AND HARDNESS

In this section, we define the sensitivity verification problem and provide our improved hardness results. Given a decision tree ensemble classifier $\mathcal { E } ,$ , our goal is to find two points $x ^ { ( \bar { 1 } ) }$ and $x ^ { ( 2 ) }$ in the input space $x ,$ , such that they differ only in a specified subset of features $F \subseteq { \mathcal { F } }$ and have the same values for all the remaining features, while producing different output labels when passed to the classifier, $\mathrm { i . e . , } \mathcal { E } ( x ^ { ( 1 ) } ) \neq \bar { \mathcal { E } ( x ^ { ( 2 ) } ) }$ . This problem becomes more significant $\operatorname { i f } ,$ not only are their output labels different, but the predicted class probabilities of the outputs are also far apart, indicating a change from a highly confident positive prediction to a highly confident negative prediction.

Definition 3.1. Given a tree ensemble for binary classification ${ \mathcal { E } } : { \mathcal { X } }  \{ 0 , 1 \}$ , a set of features $F \subseteq { \mathcal { F } }$ and a parameter $g \geq 0 , \mathcal { E }$ is said to be $( g , F )$ -sensitive if we can find two inputs $x ^ { ( 1 ) } , x ^ { ( 2 ) }$ (called a counterexample pair) such that $( x ^ { ( 1 ) } ) _ { \mathcal { F } \backslash F } = ( x ^ { ( 2 ) } ) _ { \mathcal { F } \backslash F }$ and $\mathcal { E } _ { 1 } ^ { p r o b } ( x ^ { ( 1 ) } ) \ge 0 . 5 + g$ and $\mathcal { E } _ { 1 } ^ { p r o b } ( x ^ { ( 2 ) } ) \le 0 . 5 - g$ . The sensitivity verification problem asks if a given tree ensemble classifier E is $( g , F ) \mathrm { - s e n s i t i v e }$ .

In what follows, we often fix $g$ and refer to F -sensitivity; when $F$ is clear from context, we simply write sensitivity. Ahmad et al. (2025) showed that sensitivity is NP-Hard for tree ensembles with maximum depth $\geq 3$ for |F | = 1, |F | = constant and $F = { \mathcal { F } }$ . However, as noted by the authors, their reduction (from $3 { \mathrm { - } } \mathrm { S A T } )$ does not work for depth 1 and 2. Our first result is to close this gap by showing NP-hardness at depth 1 using a novel reduction from the subset sum problem, a well-known NP-complete problem Garey & Johnson (1979).

Theorem 3.2. Sensitivity verification with $| F | = 1$ is NP-Hard for tree ensembles with depth $\geq 1$ .

Proof. Consider an instance of the integer subset sum problem, i.e., we are given a set of n integers $\mathcal { U }$ and an integer $k ,$ and we want to find a subset $U \subseteq { \mathcal { U } }$ , such that $\Sigma _ { l \in \bar { U } } = k$ . We call the $i ^ { t h }$ integer $u _ { i }$ where i varies from 0 to $n - 1$ . For every index i, we create a Boolean feature $f _ { i }$ . Then we create a decision tree of depth $^ { 1 , }$ which splits on $f _ { i } ,$ giving output 0 if $f _ { i }$ is false and $u _ { i }$ otherwise. We create another Boolean feature $f ^ { \prime }$ and a decision tree of depth 1, which splits on $f ^ { \prime }$ , giving output $- k - 0 . 5$ if $f ^ { \prime }$ is false and $- k + 0 . 5$ otherwise. We call the ensemble of all these trees to be $\mathcal { E } : \dot { \{ 0 , 1 \} } ^ { n + 1 }  \dot { \{ 0 , 1 \} }$ } with the trees being $T _ { i }$ where i varies from 0 to $n - 1$ and $T ^ { \prime }$ . We claim that E is $\left\{ f ^ { \prime } \right\}$ -sensitive iff there exists $U \subseteq { \bar { u } }$ such that $\textstyle \sum _ { l \in U } l = k$ . With this claim, the theorem immediately follows. A formal proof of the claim is given in Appendix B. □

We remark that when $| { \mathcal { F } } / F |$ is bounded, we can solve the depth 1 problem in polynomial time (see Appendix B). This completes the complexity-theoretic picture for the sensitivity problem.

# 4 DATA-AWARE SENSITIVITY VERIFICATION

Sensitivity, as defined in Definition 3.1, requires finding a counterexample pair showing sensitivity, but does not specify how close the inputs $x ^ { ( 1 ) }$ and $x ^ { ( 2 ) }$ in the pair must be to real-world data. Indeed, this is not surprising, as the definition is itself independent of data (including training data), and only depends on the trained model. But, as a consequence, counterexample pairs may include inputs that are arbitrarily far from the true data distribution, as illustrated in Figure 1 in the Introduction. Additional examples are in Appendix C. Our goal, therefore, is to find more meaningful counterexample pairs, towards which we extend Definition 3.1 with a utility function.

Definition 4.1. Given a (binary) tree ensemble classifier $\mathcal { E } : \mathcal { X } \longrightarrow \{ 0 , 1 \}$ , a set of sensitive features $F \subseteq { \mathcal { F } }$ , a gap parameter $g \geq 0$ and a data distribution/utility function u : ${ \mathcal { X } } \times { \mathcal { X } }  [ 0 , 1 ]$ , we say that E is $( g , F , u )$ -sensitive, if there exist two inputs $x ^ { ( 1 ) } , x ^ { ( 2 ) } \in \mathcal { X }$ such that $x _ { \mathcal { F } \setminus F } ^ { ( 1 ) } = x _ { \mathcal { F } \setminus F } ^ { ( 2 ) } ,$ x F \ F , $\sigma ( \mathcal { E } ( x ^ { ( 1 ) } ) ) \ge 0 . 5 + g , \sigma ( \mathcal { E } ( x ^ { ( 2 ) } ) ) \le 0 . 5 - g$ and $u ( x ^ { ( 1 ) } , x ^ { ( 2 ) } )$ is maximal among all such pairs.

We could also add a threshold parameter $\epsilon \in [ 0 , 1 ]$ and require $u ( x ^ { ( 1 ) } , x ^ { ( 2 ) } ) \ge \epsilon$ . Typically, u serves as a proxy for how similar $x ^ { ( 1 ) } , x ^ { ( 2 ) }$ are to the training distribution. Given a (possibly training) dataset D, we want the utility function u $: \mathcal { X } \times \mathcal { X } \to [ \bar { 0 } , 1 ]$ to represent the likelihood of the input pair being drawn from or close to D. In this work, we investigate two approaches to achieve this.

Utility Function. For simplicity, we first assume that all input features are independent. This allows us to calculate the likelihood of each feature independently and then multiply the results. For a given feature $f ,$ consider the guards in the ensemble that involve $f .$ . Suppose a feature $f$ appears in $K _ { f }$ guards, with sorted thresholds $\tau _ { f 1 } < \cdots < \tau _ { f K _ { f } }$ . We assume that the $X _ { f }$ takes value within range $[ \tau _ { f 1 } , \tau _ { f K _ { f } } )$ , which we can ensure by introducing guards $X _ { f } < - \infty$ and $X _ { f } < \infty$ . This partitions the space of feature f into $K _ { f } - 1$ intervals. We estimate the marginal probability of f lying in each interval $[ \tau _ { f ( k - 1 ) } , \tau _ { f k } )$ by calculating the count of points in D for which f lies in $[ \tau _ { f ( k - 1 ) } , \tau _ { f k } )$ and dividing this by the total count of points in D. That is, for any feature f and real value v, we have,

$$
\pi_ {f} (v) = \sum_ {k = 2} ^ {K _ {f}} \left( \right.\mathbf {1} _ {\left(\tau_ {f (k - 1)} \leq v <   \tau_ {f k}\right)} \cdot \frac {\left| \right.\left\{ \right.x \in \mathcal {D} \mid \tau_ {f (k - 1)} \leq x _ {f} <   \tau_ {f k}\left. \right)\left. \right\}\left. \right|}{| \mathcal {D} |}\left. \right)
$$

And for any input $x = ( x _ { 1 } , x _ { 2 } , x _ { 3 } , \ldots , x _ { d } )$ , assuming independence of features, we define $\pi ( x ) =$ $\pi _ { f _ { 1 } } ( x _ { 1 } ) \cdot \pi _ { f _ { 2 } } ( x _ { 2 } ) \ldots \pi _ { f _ { n } } ( x _ { d } )$ , where $\pi : \mathcal { X }  [ 0 , 1 ]$ is the product distribution estimated from $\mathcal { D }$ . With this the utility function just becomes $\iota ( x ^ { ( 1 ) } , x ^ { ( \bar { 2 } ) } ) = \pi ( x ^ { ( 1 ) } ) \cdot \pi ( x ^ { ( 2 ) } )$ .

Intuitively, under the independence assumption, it measures how likely inputs are to be drawn from the distribution D, guiding our approach to look for more meaningful counterexamples. In the next section, we show how this can be encoded effectively, and our experiments indicate that in many benchmarks it does give better counterexample pairs (i.e., closer to data). However, its effectiveness diminishes in datasets with high feature dependencies, which motivates an orthogonal approach.

Restricting search space using clause summaries. Our second approach attempts to compute summaries of the input space that describe cavities - regions where no data points exist.

Our goal is to ensure that these cavities are excluded from our sensitivity search. For simplicity, we focus on cavities represented as a bounded boxes in the input space. Given a value of w (a width parameter), we create the following template for points in D that fall in a cavity:

$$
\bigwedge_ {i \in [ 1, w ]} (X _ {f _ {i}} \geq l b _ {i} \land X _ {f _ {i}} <   u b _ {i}) \tag {1}
$$

where each $l b _ { i }$ and $u b _ { i }$ take values from one of the guards associated with feature $f _ { i }$ appearing in the tree ensemble. In Figure 2, we illustrate a green cavity in 2-D space, where all the data points are projected in dimensions ${ \dot { f } } _ { i }$ and $f _ { j }$ . Therefore, we avoid finding sensitivity pairs in $\begin{array} { r } { ( X _ { f _ { i } } \geq l b _ { i } \land X _ { f _ { i } } < u b _ { i } ) \land ( X _ { f _ { j } } \geq l b _ { j } \land X _ { f _ { j } } < u b _ { j } ) } \end{array}$ . The difficulty is to find such cavities in the data set. For this, we observe that given a box which is a conjunction of interval regions, its negation is a clause, in a form that can be processed by a constraint solver. Hence, our main idea is to find such cavities in the input data-set using a state-of-the-art Satisfiability Modulo Theories (SMT) solver De Moura & Bjørner (2008), as we detail in Section 5.3.

![](images/22679cea2bad6032dad27861912769cd5602037dc32e747e04790bade3fc4dad.jpg)

<details>
<summary>scatter</summary>

| f_j | f_i |
| --- | --- |
| lb_j | 0 |
| ub_j | 0 |
| 0 | 0 |
</details>

Figure 2: There are no training set data points within the green box.

# 5 AN IMPROVED MILP ENCODING

We build on the MILP encoding for decision trees introduced by Kantchelian et al. (2016). The encoding, when used directly for sensitivity verification, is less efficient than the pseudo-Boolean approach as shown in Ahmad et al. (2025). However, we develop novel optimizations to the encoding for sensitivity analysis, which allow the MILP encoding to outperform the pseudo-Boolean encoding by a large margin, achieving an order-of-magnitude runtime reduction compared to SENSPB.

Base Encoding. The encoding represents the decision tree ensemble as a set of linear inequalities. It uses a set of binary variables $p _ { f k }$ to denote the predicates that appear on the internal node’s guard is true, and a set of continuous variables $0 \leq l _ { n } \leq 1$ 1 to denote which leaf node is visited in each tree. The output of the tree ensemble is then computed as a linear combination of the leaf values, weighted by the predicate variables. For each input feature $f ,$ we ensure consistency across the predicate variables corresponding to the feature, since if $\tau _ { 1 } < \tau _ { 2 }$ , then $X _ { f } ~ < ~ \tau _ { 1 }$ implies $X _ { f } \ < \ \tau _ { 2 } ,$ . Let the corresponding predicate variables be $p _ { f 1 } , p _ { f 2 } , \dotsc , p _ { f K _ { f } } .$ . We require that $p _ { f 1 } = 1 \implies p _ { f 2 } = 1$ , $p _ { f 2 } = 1 \implies p _ { f 3 } = 1 \dots p _ { f ( K _ { f } - 1 ) } = 1 \implies p _ { f K _ { f } } = 1$ . We encode this in MILP as Eq. (2).

$$
p _ {f 1} \leq p _ {f 2} \leq \dots \leq p _ {f K _ {f}} \tag {2}
$$

$$
l _ {1} + l _ {2} + \dots + l _ {N} = 1 \tag {3}
$$

Let $l _ { 1 } , l _ { 2 } , \ldots , l _ { N }$ be the leaf variables corresponding to a tree. We require two leaf consistency conditions. First, we require that exactly one of these leaf variables is set to 1 and every other leaf variable is set to 0, which can be enforced as in Eq. (3). Second, we require that if a leaf variable is set to 1, then every predicate variable in the path to the leaf node needs to be set such that the path is followed. For each internal node n of tth tree, consider the set T Set of leaf nodes in the subtree rooted at n.yes and set $F S e t$ of the leaf nodes of subtree rooted at n.no. Let $X _ { f } < \tau _ { f k }$ be the guard of node n (recall that $X _ { f }$ refers to variables while $x _ { f }$ are concrete input values). If n is a root node, then we add the constraint given in Eq. (4), and for any non-root node, the constraint in Eq. (5).

$$
1 - \sum_ {n \in F S e t} l _ {n} = p _ {f k} = \sum_ {n \in T S e t} l _ {n} \quad (4) \quad 1 - \sum_ {n \in F S e t} l _ {n} \geq p _ {f k} \geq \sum_ {n \in T S e t} l _ {n} \tag {5}
$$

The constraints in Eq. (2) to Eq.(5) are from Kantchelian et al. (2016). To model the sensitivity problem, we create two instances of all the variables to encode the runs of the two differentiating inputs given to the tree ensemble. We will add superscripts to indicate the copies. We need to ensure that the two inputs differ only in the specified set of features $F \subseteq { \mathcal { F } }$ . For this, denoting by $\nu _ { F }$ the set of all predicate variables such that their guards contain some $f \in F$ , we add the constraint in Eq. (6). Finally, for binary trees, $\mathcal { E } ^ { p r o b } ( x ) \stackrel { \smile } { > } 0 . 5 + g \Longleftrightarrow \mathcal { E } ^ { r a w } \bar { ( x ) } > \bar { \mathrm { S I G M O I D } } ^ { - 1 } ( 0 . 5 + g )$ . Let us define $\delta = \mathrm { S I G M O I D } ^ { - 1 } ( 0 . 5 + g )$ . Because of the symmetry of SIGMOID about $y = 0 . 5$ , we have that SIGMOID $^ { - 1 } ( 0 . 5 - g ) \stackrel { \cdot } { = } - \delta$ . Thus, we introduce the constraint described in Eq. (Gap-bin). Recall that for a leaf node n, n.val denotes its value. Let A be the set of all leaves in all trees.

$$
\bigwedge_ {p _ {f k} ^ {(1)} \not \in \mathcal {V} _ {\mathcal {F}}} p _ {f k} ^ {(1)} = p _ {f k} ^ {(2)} \quad (6) \quad \sum_ {n \in \mathcal {A}} l _ {n} ^ {(1)} n. v a l \geq \delta \wedge \sum_ {n \in \mathcal {A}} l _ {n} ^ {(2)} n. v a l \leq - \delta \quad (\text { Gap - bin })
$$

Any feasible solution to these constraints corresponds to two inputs $x ^ { ( 1 ) }$ and $x ^ { ( 2 ) }$ , which differ only in the feature set $F$ and produce outputs such that $\mathcal { E } ( x ^ { ( 1 ) } ) \geq 0 . 5 + g$ and $\mathcal { E } ( x ^ { ( 2 ) } ) \leq 0 . 5 - g$ .

Optimizations to the Encoding. We now describe the novel optimizations that we develop and prove their corrections. Subsequently, we will also explain how we incorporate data-awareness.

# 5.1 CONSTRAINTS ON UNAFFECTED AND AFFECTED LEAVES

For each leaf, we call the set of all the guards in the path from the root to the leaf as the ancestry of that leaf. A leaf is called unaffected if, for each guard in the ancestry of the leaf, the guard predicate does not contain a feature from the set of varying features. Let U denote the set of indices of all unaffected leaves. For each such leaf, we add a constraint on the two variables $l _ { n } ^ { ( 1 ) } , l _ { n } ^ { ( 2 ) }$ as defined in Eq. (UnAff). Intuitively, if a leaf is reached in a run of the first input, then it will also be reached in the second. Adding this constraint explicitly helps the solver reach a feasible solution faster, especially in cases where the sensitive feature only affects a small subset of the leaves. This is particularly important for features that are present in guards that are farther away from root nodes. In practice, a large fraction of leaves belong to U .

Next, given that a significant fraction of the leaves are unaffected in practical scenarios, we also add constraints to ensure that Leaf variables that are not in U (i.e., they correspond to “affected” leaves) are capable of influencing the output. This is done by subtracting the two constraints in Eq. (Gapbin) and using Eq. (UnAff) to remove any terms corresponding to unaffected leaves, leading to the constraints in Eq. (Aff-bin). This constraint has significantly fewer terms than Eq. (Gap-bin) while capturing its essence, leading to significantly faster running times.

$$
\bigwedge_ {l _ {n} \in \mathcal {U}} l _ {n} ^ {(1)} = l _ {n} ^ {(2)} \quad (\text { UnAff }) \quad \sum_ {l _ {n} \not \in \mathcal {U}} \left(l _ {n} ^ {(1)} n. v a l - l _ {n} ^ {(2)} n. v a l\right) \geq 2 \times \delta \quad (\text { Aff - bin })
$$

Crucially, as the following theorem shows (with Proof in Appendix D), adding these optimizations does not change the set of feasible solutions.

Theorem 5.1. Let C denote the MILP equation obtained as a conjunction of Equations (2), (3), (4), $( 5 ) , ( 6 )$ and (Gap-bin). The set of feasible solutions of C and the MILP obtained by conjuncting C with the constraints induced by Eq. (UnAff) and Eq. (Aff-bin) are equal.

# 5.2 OBJECTIVE FUNCTION

Modern MILP solvers are built to optimize over an objective function and are highly engineered with multiple heuristics to traverse the search space while being mindful of the objective function. For our setting, we just need to find a single feasible solution to the set of constraints C as defined in Theorem 5.1. Among these constraints, Eq. (Gap-bin) captures the essence of the two outputs being different and reduces the set of feasible solutions by a significant amount. We can utilize the objective function to amplify the importance of this constraint for the solver by adding a constraint in Eq. Obj-bin), capturing the difference between the two equations in Eq. (Gap-bin).

$$
\text { MAX } \sum_ {n \in \mathcal {A}} \left(l _ {n} ^ {(1)} n. v a l - l _ {n} ^ {(2)} n. v a l\right) \tag {Obj-bin}
$$

This addition leads to significant improvement in running times, as the objective function can guide the MILP solver in choosing the better edge when faced with multiple candidates during the simplexsolving process instead of arbitrarily choosing between each of the available constraints. In Appendix H, we present the full encoding of an illustrative example.

# 5.3 MODIFICATION IN MILP FOR DATA-AWARE SEARCH

Finally, we modify the MILP encoding to solve data-aware sensitivity as defined in Def. 4.1.

Utility Function. Given the utility function as defined in Section 4, we replace the objective function in Eq. (Obj-bin) by the following objective, which maximizes the value of the utility function:

$$
\operatorname{MAX} \sum_ {f \in \mathcal {F}} \sum_ {k = 2} ^ {K _ {f}} (\log (\pi_ {f} (\tau_ {f (k - 1)})) - \log (\pi_ {f} (\tau_ {f k}))) (p _ {f k} ^ {(1)} + p _ {f k} ^ {(2)}). \tag {7}
$$

Importantly, we convert the product of marginals to log values, as MILP solvers only handle additive constraints on the objective function. The following lemma establishes the correctness of the dataaware objective function given in equation 7.

Lemma 5.2. The objective function in 7 maximizes $u ( x ^ { ( 1 ) } , x ^ { ( 2 ) } )$ .

Proof. Let $p _ { f k } ^ { ( 1 ) }$ be true for smallest k. Due to the consistency constraints, all $p _ { f ( k + 1 ) } ^ { ( 1 ) } , . . . , p _ { f K _ { j } } ^ { ( 1 ) }$ are true. Let $p _ { f k ^ { \prime } } ^ { ( 2 ) }$ be true for smallest $k ^ { \prime } .$ . Therefore, the sum will reduce to $\begin{array} { r } { \sum _ { f \in \mathcal { F } } \log \bigl ( \pi _ { f } \bigl ( \tau _ { f ( k - 1 ) } \bigr ) \bigr ) \to } \end{array}$ may replace $\log ( \pi _ { f } ( \tau _ { f ( k ^ { \prime } - 1 ) } ) ) - 2 \log ( \pi _ { f } ( \tau _ { f K _ { f } } ) )$ $\tau _ { f ( k - 1 ) }$ by $ { \boldsymbol { x } } _ { f } ^ { ( 1 ) }$ f because . We may ignore the last term as it is a constant and we $x _ { f } ^ { ( 1 ) } \in [ \tau _ { f ( k - 1 ) } , \tau _ { f k } )$ and $\tau _ { f ( k ^ { \prime } - 1 ) }$ by $x _ { f } ^ { ( 2 ) }$ because $x _ { f } ^ { ( 2 ) } \in$ $[ \tau _ { f ( k ^ { \prime } - 1 ) } , \tau _ { f k ^ { \prime } } )$ . Therefore, the total sum will be $\begin{array} { r } { \sum _ { f \in \mathcal { F } } \log ( \pi _ { f } ( x _ { f } ^ { ( 1 ) } ) ) + \log ( \pi _ { f } ( x _ { f } ^ { ( 2 ) } ) ) } \end{array}$ ), Since the objective function is maximizing the sum, it is maximizing our utility function ${ \dot { u } } ( x ^ { ( 1 ) } , x ^ { ( 2 ) } ) =$ $\Pi _ { f \in \mathcal { F } ^ { \pi _ { f } } } ( x _ { f } ^ { ( 1 ) } ) \cdot \pi _ { f } ( x _ { f } ^ { ( 2 ) } )$ . □

Computing clause summaries. Next we define constraints that can identify cavities in data and their negations, i.e., clauses that guide the sensitivity search. In Eq. (1), we need to learn the features and their bounds. Let $r _ { i f }$ be a Boolean variable indicating $f _ { i }$ is $f ,$ and $s _ { i k }$ indicating $l b _ { i } = \tau _ { f k }$ , and $t _ { i k }$ indicating $u b _ { i } = \tau _ { f k }$ , where $\tau _ { f k }$ is the k-th guard for feature f . For each $x \in \mathcal { D }$ , we add:

$$
\bigvee_ {i \in [ 1, w ]} \left(\bigwedge_ {f \in \mathcal {F}, k, k ^ {\prime} \in [ 1, K _ {f} ]} (r _ {i f} \wedge s _ {i k} \wedge t _ {i k ^ {\prime}} \to ((x _ {f} <   \tau_ {f k} \vee x _ {f} \geq \tau_ {f k ^ {\prime}})))\right)
$$

To avoid redundancies, we enforce the ordering of features: $i < j  f _ { i } < f _ { j }$ . While solving the constraints to find a clause that satisfies all samples, we also add an objective function to guide towards learning tight clauses as: MIN Pi∈[1,w]( $\begin{array} { r } { \left( \sum _ { i \in [ 1 , w ] } ( \sum _ { k = 1 } ^ { | K | } k \cdot s _ { i k } - \sum _ { k ^ { \prime } = 1 } ^ { | K | } k ^ { \prime } \cdot t _ { i k ^ { \prime } } ) \right) } \end{array}$ , where $K =$ $m a x _ { f } ( K _ { f } )$ . This ensures that we select the smallest guard k for the lower bound of some feature and the largest guard $k ^ { \prime }$ for the upper bound of the feature. We iteratively compute one clause at a time and add constraints to exclude solutions corresponding to previously computed clauses.

<table><tr><td colspan="5">Binary classifiers</td></tr><tr><td>SNO</td><td>ModelName</td><td>#Trees</td><td colspan="2">Dep. #Feat.</td></tr><tr><td>1</td><td>breast_cancer_robust</td><td>4</td><td>4</td><td>10</td></tr><tr><td>2</td><td>breast_cancer_unrobust</td><td>4</td><td>5</td><td>10</td></tr><tr><td>3</td><td>diabetes_robust</td><td>20</td><td>4</td><td>8</td></tr><tr><td>4</td><td>diabetes_unrobust</td><td>20</td><td>5</td><td>8</td></tr><tr><td>5</td><td>ijcnn_robust</td><td>60</td><td>8</td><td>22</td></tr><tr><td>6</td><td>ijcnn_unrobust</td><td>60</td><td>8</td><td>22</td></tr><tr><td>7</td><td>adult</td><td colspan="2">{200,300,500}{4,5}</td><td>15</td></tr><tr><td>8</td><td>churn</td><td colspan="2">{200,300,500}{4,5}</td><td>21</td></tr><tr><td>9</td><td>pimadiabetes</td><td colspan="2">{200,300,500}{4,5}</td><td>9</td></tr><tr><td>10</td><td>german_credit</td><td>{500,800}</td><td>{4,5}</td><td>20</td></tr></table>

<table><tr><td colspan="5">Multi classifiers</td></tr><tr><td>SNO</td><td>ModelName</td><td>#Trees</td><td>Dep.</td><td>#Classes</td></tr><tr><td>1</td><td>covtype_robust</td><td>100</td><td>6</td><td>10</td></tr><tr><td>2</td><td>covtype_unrobust</td><td>100</td><td>6</td><td>10</td></tr><tr><td>3</td><td>fashion_robust</td><td>100</td><td>6</td><td>10</td></tr><tr><td>4</td><td>fashion_unrobust</td><td>100</td><td>6</td><td>10</td></tr><tr><td>5</td><td>ori_mnist_robust</td><td>100</td><td>6</td><td>10</td></tr><tr><td>6</td><td>ori_mnist_unrobust</td><td>100</td><td>6</td><td>10</td></tr><tr><td>7</td><td>Iris</td><td>100</td><td>1</td><td>3</td></tr><tr><td>8</td><td>Red-Wine</td><td>100</td><td>4</td><td>5</td></tr></table>

Table 1: Benchmark details. Binary models 1–6 are taken from Chen et al. (2019b); 7–10 are trained on UCI datasets Dua & Graff (2019) using all combination of #Trees ∈ {200, 300, 500} and Depth $\in \ \{ 5 , 6 \}$ (six configurations). Multiclass models 1–6 are from Chen et al. (2019b); models 7–8 are trained on UCI datasets Dua & Graff (2019). Dep. refers to the average depth in the ensemble across all trees of the model.

Initially, we learn clauses of size one and progressively increase the size up to a user-defined limit. The computed clauses provide a summary of D, capturing how the data is distributed across the input space and add the learned clauses to the constraints. Any solution to the query is required to satisfy these clauses, thereby making it more likely to align with the data.

# 6 EXTENSION TO MULTI-CLASS TREE ENSEMBLES

We extend our formalism and encoding to the multi-class setting. Let $\mathcal { Y } = \{ 0 , 1 , \ldots , C - 1 \}$ denote the set of C classes in a multiclass tree ensemble. The set of trees T is partitioned into C equal partitions, one for each class denoted by $\mathcal { T } _ { 0 } , \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { C - 1 }$ . The trees in a partition $\mathcal { T } _ { c }$ are one-vsrest classifiers for the class $c ;$ that is, they consider class c as the positive class, and everything else combined as the negative class and train like a binary classifier. Formally, $\begin{array} { r } { \mathcal { E } _ { c } ^ { r a w } ( x ) = \sum _ { T \in \mathcal { T } _ { c } } \bar { T } ( x ) } \end{array}$ , and $\mathcal { E } _ { c } ^ { p r o b } ( x ) = \mathrm { S O F T M A X } _ { c } \left( \mathcal { E } _ { 0 } ^ { r a w } ( x ) , \ldots , \mathcal { E } _ { C - 1 } ^ { r a w } ( x ) \right)$ , where SOFTMA $\mathbf { X } _ { c } : \mathbb { R } ^ { C } $ R is the softmax function defined as SOFTMthe highest probability, i.e., $\smash { \{ x _ { 0 } , x _ { 1 } , . . . , x _ { C - 1 } \} = e ^ { x _ { c } } / \Sigma _ { k = 0 } ^ { C - 1 } e ^ { x _ { k } } }$ . The output is the class with a tree ensemble for multiclass $\mathcal { E } ( x ) = A r g m a x _ { c \in \mathcal { Y } } \mathcal { E } _ { c } ( x )$ classification $\mathcal { E } : \mathcal { X } \to \{ 0 , 1 , . . . , C - 1 \} , c ^ { ( 1 ) } , c ^ { ( 2 ) } \in \{ 0 , 1 , . . . C - 1 \}$ , we find $x ^ { ( 1 ) } , x ^ { ( 2 ) } \in \mathcal { X }$ such that $\mathcal { E } ( x ^ { ( 1 ) } ) = c ^ { ( 1 ) } \neq \mathcal { E } ( x ^ { ( 2 ) } ) = c ^ { ( 2 ) }$ . We also extend the parameterized version of Def. 3.1 by requiring that the difference between probability of most and second-most probable class is large:

Definition 6.1. Given tree ensemble Y , E is $( g , F , c ^ { ( 1 ) } , c ^ { ( 2 ) } )$ )−sensitive if there exist $\mathcal { E } : \mathcal { X }  \mathcal { Y } , F \subseteq \mathcal { F } , g \geq 0 .$ ${ \bar { x } } ^ { ( 1 ) } , x ^ { ( \overline { { 2 } } ) }$ such that $( x ^ { ( 1 ) } ) _ { \mathcal { F } \backslash F } ~ = ~ ( x ^ { ( 2 ) } ) _ { \mathcal { F } \backslash F } .$ , two classes $c ^ { ( 1 ) } , c ^ { ( 2 ) } \in$ ∀ $: \neq c ^ { ( 1 ) } , \mathcal { E } _ { c ^ { ( 1 ) } } ^ { p r o b } ( x ^ { ( 1 ) } ) \ge g \times \mathcal { E } _ { c } ^ { p r o b } ( x ^ { ( 1 ) } )$ , and ∀c ̸= c(2), $\mathcal { E } _ { c ^ { ( 2 ) } } ^ { p r o b } ( x ^ { ( 2 ) } ) \ge g \times \mathcal { E } _ { c } ^ { p r o b } ( x ^ { ( 2 ) } )$ .

Now to extend our MILP encoding to the multiclass setting it turns out that we only need to modify Eq. (Gap-bin), Eq. (Aff-bin) and Eq. (Obj-bin) (and its data-aware variant). We present the modifications and prove their correctness in Appendix E.

# 7 EXPERIMENTS

We implement the MILP encoding from Section 5 with all structural optimizations and then add our two data-aware objectives in a tool called ENSENSE. We trained models with XGBoost v1.7.1; we evaluate sensitivity using a single CPU core per run with a per-instance 3600 seconds timeout. We use Gurobi Gurobi Optimization, LLC (2024) as the MILP solver. We focus on single-feature sensitivity; results on varying multiple (viz., 2, 3, 4, and 5) features are presented in Appendix 7. We address the following research questions: RQ1: How does our MILP encoding with optimizations fare against the baseline and state-of-the-art for (i) binary and (ii) multi-class classification?

![](images/2ec81aee8ff6fa8713ce35f415ae498e37a54271d2fc1a0539e8ec7f42051c17.jpg)

<details>
<summary>line</summary>

| Instances Solved | ENSENSE | KANT | SENSPB |
| ---------------- | ------- | ---- | ------ |
| 0                | 1       | 1    | 1      |
| 200              | 5       | 10   | 5      |
| 400              | 10      | 20   | 10     |
| 600              | 15      | 30   | 20     |
| 800              | 20      | 50   | 40     |
| 1000             | 30      | 100  | 80     |
| 1200             | 50      | 200  | 150    |
</details>

(a) Binary classification

![](images/cfe4f7e1690aac91e4a5844c3d98941233f33f25e16fd68f92f4403f66780cd1.jpg)

<details>
<summary>line</summary>

| Instances Solved | ENSENSE | KANT   |
| ---------------- | ------- | ------ |
| 0                | 1       | 1      |
| 50               | 20      | 10     |
| 100              | 30      | 20     |
| 150              | 40      | 30     |
| 200              | 50      | 40     |
| 250              | 60      | 50     |
| 300              | 70      | 60     |
| 350              | 80      | 70     |
| 400              | 90      | 80     |
| 450              | 100     | 90     |
| 500              | 120     | 100    |
| 550              | 150     | 120    |
</details>

(b) Multiclass classification   
Figure 3: Cactus plot comparing runtimes of single feature sensitivity for binary and multiclass.

<table><tr><td></td><td>None</td><td>Clause</td><td>Prob</td><td>Probclause</td></tr><tr><td>mean</td><td>0.57</td><td>0.306</td><td>0.26</td><td>0.17</td></tr><tr><td colspan="2">Method</td><td colspan="3">Win%Draw%Loss%</td></tr><tr><td colspan="2">Clause vs None</td><td>80.5</td><td>3.2</td><td>16.1</td></tr><tr><td colspan="2">Prob vs None</td><td>76.6</td><td>1.15</td><td>22.1</td></tr><tr><td colspan="2">Prob vs Clause</td><td>56.6</td><td>1.1</td><td>42.1</td></tr><tr><td colspan="2">Probclause vs None</td><td>86.7</td><td>1.1</td><td>12.1</td></tr><tr><td colspan="2">Probclause vs Prob</td><td>56.4</td><td>15.3</td><td>28.1</td></tr><tr><td colspan="2">Probclause vs Clause</td><td>72.7</td><td>1.7</td><td>25.5</td></tr></table>

![](images/64a277de92128114fc8051f8a333855eab146fe53f048fa66cceccfdd42dfb73.jpg)

<details>
<summary>line</summary>

| Instances Solved | none  | prob  | clause | probclause |
| ---------------- | ----- | ----- | ------ | ---------- |
| 0                | 0.25  | 0.25  | 0.25   | 0.25       |
| 200              | 0.50  | 0.75  | 0.50   | 0.25       |
| 400              | 0.75  | 0.75  | 0.50   | 0.25       |
| 600              | 0.75  | 1.10  | 0.50   | 0.25       |
| 800              | 1.00  | 0.75  | 0.50   | 0.25       |
| 1000             | 1.75  | 1.10  | 0.50   | 0.25       |
</details>

Figure 4: In top-left, we report the mean $\ell _ { 2 }$ distance of all instances for each method. In the bottom-left table, we report the win, draw, and loss rates for all pairwise method comparisons. On the right side, a cactus plot of the distance (from the data set to the counterexample found) across binary benchmarks. Instances are sorted by non data-aware baseline (none, orange, solid); for each position, distances of Prob (red, dotted), Clause (green, dashed), and Probclause (blue, solid) are evaluated on same instance (no re-sorting). Lower curves indicate better quality.

RQ2: How does data-aware sensitivity perform in giving better quality counterexamples? How do we measure it, what do we compare it against and how do each of our strategies help?

Binary Classification. To answer RQ1(i), we compare the performance of ENSENSE against SENSPB the tool using pseudo-Boolean encoding from Ahmad et al. (2025), and KANT, the baseline MILP encoding (adapted from Kantchelian et al. (2016)). We use a wide set of benchmarks mentioned in Table 1 (left), with number of trees ranging from 4 − 800, with depth from 4 − 8. Considering each single-feature variant as a separate instance and with $g a p \in \{ 0 . 5 , 1 , 1 . 5 \}$ this yields a total of 1,290 benchmark instances. Figure 3(a) reports results for the 1,103 instances whose runtime is ≥ 1 s; we omit 187 instances solved in < 1 s for fair comparison, which demonstrates the superior performance of ENSENSE. ENSENSE achieves average speedups of approximately 8× over KANT and 5× over SENSPB, with no timeouts whereas SENSPB times out for 205 and KANT for 153 instances.

Multiclass Classification. For RQ1(ii) we compare ENSENSE against the baseline MILP (still called KANT), as SENSPB does not handle multiclass ensembles. We also repurposed the versatile robustness verification tool VERITAS from Devos et al. (2024) that can handle multiclass ensembles, to solve sensitivity. As it timed out on all except two instances, we detail these results in Appendix F. Table 1(right) lists the multiclass benchmarks. For fashion and ori mnist, which have 784 features, we restricted to single-feature sensitivity for the 100 most frequently used features in the tree model. For the others, we tested on all features, making a total of 538 benchmark instances. Our results are in Fig3(b), where we again dropped 17 instances which were solved in < 1 sec. The results demonstrate that ENSENSE outperforms KANT by roughly 15x average speed up, and does not timeout on any of these benchmarks, whereas KANT times out on 256.

Data-Aware Sensitivity Verification. For RQ2, to evaluate data-aware sensitivity, we compare (i) the baseline (no data-awareness) with (ii) the utility-based objective that steers solutions toward data-dense regions (called Prob), (iii) the clause-summary strategy that prunes empty-data regions (called Clause), (iv) the combination of both (called Probclause), which ENSENSE uses.

To evaluate the performance of data-awareness sensitivity of each method, we compute the $\ell _ { 2 }$ distance over insensitive features from the data to the nearest counterexample region identified. Given an input $x ,$ a counterexample region is a connected subset of the input space containing $x ^ { \prime }$ such that: (i) the tree ensemble’s prediction is constant throughout the region (all points fall into the same combination of leaves across all trees), and (ii) every point in the region has a label y with same probability (e.g., is misclassified). Intuitively, tree ensembles partition $\mathbb { R } ^ { d }$ into axis-aligned polytopes (one per joint leaf pattern); within each polytope, the model’s output does not change. A counterexample region is one of these polytopes (or an intersection thereof with additional constraints) that certifies a whole set of violating inputs, not just a single adversarial point. We focus on regions rather than a single point because if the nearest data point lies in the same counterexample region, the correct distance is zero, whereas distance to any particular witness point inside that region can be nonzero and misleading.

Experiments are conducted on the same binary classification benchmarks (from Table 1) with $g a p \in$ {0.5, 1, 1.5}. We used z3De Moura & Bjørner (2008) to synthesize clauses with a maximum of 3 literals, with counterexample-guided refinement, followed by greedy literal-pruning to minimize each clause without reducing coverage. We also discard clauses that enclose an input point, and limit to 1500 synthesized clauses. Our results in Fig 4 show that Utility-based vs. baseline: wins 76.65% of pairs (losses 22.19%, draws 1.15%), with mean distance advantage 0.435 on wins and mean deficit 0.11 on losses. Clause-summary vs. baseline: wins 80.59% (losses 16.13%, draws 3.26%), with mean advantage 0.34 (wins) and mean deficit 0.09 (losses). Combined (Probclause): strongest overall—wins 86.04% of pairs versus baseline, with mean advantage 0.47 on wins and mean deficit 0.06 on losses. In summary, our results show significant improvement in quality of counterexample pairs (measured by their $\ell _ { 2 }$ distance from data), with best results obtained by probclause used by ENSENSE.

We also performed ablation studies on binary and multiclass ensembles to evaluate the contribution of the MILP optimizations, affected and unaffected constraints that we present in Appendix G.2. More experiments and technical details are given in the ArXiv version Varshney et al. (2026) and code is available at https://github.com/formal-trust-AI/ensense.

# 8 CONCLUSION

In this work, we defined a data-aware variant of the sensitivity problem on tree ensembles and developed two approaches to solve this. We developed a new MILP encoding with several improvements, that allows us to improve the quality of the sensitivity witness reported while at the same time providing upto 5× speedup for binary classification over the existing methods and 15× for multiclass classification. One obvious direction for future work is to develop methods for training tree ensembles such that sensitivity can be reduced. This is analogous to the development of various tools for training decision trees that are hardened for local robustness. The other direction for future work is to go from identifying sensitive pairs to quantifying sensitive areas across the input space.

# ACKNOWLEDGMENTS

We acknowledge the State Bank of India (SBI) Foundation Hub for Data Science & Analytics , IIT Bombay for supporting the work done in this project. We thank Shri Shakeel Ahmed Agasimani, Deputy General Manager, Analytics Department, C Ramesh Chander, Assistant General Manager (Statistician), Muthukumaran M S, Chief Manager (Data Scientist), Komaragiri Srinivas Jagannath, Manager (Data Scientist), Neha Maheswari, Manager (Data Scientist), State Bank of India, for multiple wide-ranging discussions and feedback. We are grateful to Sunita Sarawagi for her valuable discussions and guidance throughout this work.

# REFERENCES

Arhaan Ahmad, Tanay Vineet Tayal, Ashutosh Gupta, and S. Akshay. Sensitivity verification for additive decision tree ensembles. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=h0vC0fm1q7.   
Hendrik Blockeel, Laurens Devos, Benoˆıt Frenay, G ´ eraldin Nanfack, and Siegfried Ni- ´ jssen. Decision trees: from efficient prediction to responsible ai. Frontiers in Artificial Intelligence, 6, 2023. ISSN 2624-8212. doi: 10.3389/frai.2023.1124553. URL https://www.frontiersin.org/journals/artificial-intelligence/ articles/10.3389/frai.2023.1124553.   
Stefano Calzavara, Lorenzo Cazzaro, Claudio Lucchese, and Federico Marcuzzi. Explainable global fairness verification of tree-based classifiers. 2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 1–17, 2022. URL https://api.semanticscholar. org/CorpusID:252545325.   
Yung-Chia Chang, Kuei-Hu Chang, and Guan-Jhih Wu. Application of extreme gradient boosting trees in the construction of credit risk assessment models for financial institutions. Applied Soft Computing, 73:914–920, 2018. ISSN 1568-4946. doi: https://doi.org/10.1016/j.asoc. 2018.09.029. URL https://www.sciencedirect.com/science/article/pii/ S1568494618305465.   
Hongge Chen, Huan Zhang, Duane Boning, and Cho-Jui Hsieh. Robust decision trees against adversarial examples. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 1122–1131. PMLR, 09–15 Jun 2019a. URL https://proceedings.mlr. press/v97/chen19m.html.   
Hongge Chen, Huan Zhang, Si Si, Yang Li, Duane S. Boning, and Cho-Jui Hsieh. Robustness verification of tree-based models. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alche-Buc, Emily B. Fox, and Roman Garnett (eds.), ´ Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 12317–12328, 2019b. URL https://proceedings.neurips.cc/paper/2019/ hash/cd9508fdaa5c1390e9cc329001cf1459-Abstract.html.   
Jing Chen, Lianlian Wu, Kunhong Liu, Yong Xu, Song He, and Xiaochen Bo. EDST: a decision stump based ensemble algorithm for synergistic drug combination prediction. BMC Bioinform., 24(1):325, 2023. doi: 10.1186/S12859-023-05453-3. URL https://doi.org/10.1186/ s12859-023-05453-3.   
Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, pp. 785–794, New York, NY, USA, 2016. ACM. ISBN 978-1-4503-4232-2. doi: 10.1145/ 2939672.2939785. URL http://doi.acm.org/10.1145/2939672.2939785.   
Leonardo De Moura and Nikolaj Bjørner. Z3: an efficient smt solver. In Proceedings of the Theory and Practice of Software, 14th International Conference on Tools and Algorithms for the Construction and Analysis of Systems, TACAS’08/ETAPS’08, pp. 337–340, Berlin, Heidelberg, 2008. Springer-Verlag. ISBN 3540787992.   
Laurens Devos, Wannes Meert, and Jesse Davis. Versatile verification of tree ensembles. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 2654–2664. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/devos21a.html.   
Laurens Devos, Lorenzo Cascioli, and Jesse Davis. Robustness verification of multi-class tree ensembles. Proceedings of the AAAI Conference on Artificial Intelligence, 38(19):21019–21028, Mar. 2024. doi: 10.1609/aaai.v38i19.30093. URL https://ojs.aaai.org/index.php/ AAAI/article/view/30093.

Dheeru Dua and Casey Graff. Uci machine learning repository. https://archive.ics.uci. edu/ml, 2019.   
Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard S. Zemel. Fairness through awareness. CoRR, abs/1104.3913, 2011. URL http://arxiv.org/abs/1104. 3913.   
Sainyam Galhotra, Yuriy Brun, and Alexandra Meliou. Fairness testing: testing software for discrimination. In Proceedings of the 2017 11th Joint Meeting on Foundations of Software Engineering, ESEC/FSE 2017, pp. 498–510, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450351058. doi: 10.1145/3106237.3106277. URL https: //doi.org/10.1145/3106237.3106277.   
M. R. Garey and David S. Johnson. Computers and Intractability: A Guide to the Theory of NP-Completeness. W. H. Freeman, 1979. ISBN 0-7167-1044-7.   
Mohammad M. Ghiasi and Sohrab Zendehboudi. Application of decision tree-based ensemble learning in the classification of breast cancer. Computers in Biology and Medicine, 128:104089, 2021. ISSN 0010-4825. doi: https://doi.org/10.1016/j.compbiomed.2020.104089. URL https: //www.sciencedirect.com/science/article/pii/S0010482520304200.   
Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2024. URL https://www. gurobi.com.   
Miklos Z. Horv ´ ath, Mark Niklas M ´ uller, Marc Fischer, and Martin T. Vechev. (de- ¨ )randomized smoothing for decision stump ensembles. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ 146b4bab3f8536a07905f25d367b4924-Abstract-Conference.html.   
Phuoc-Hai Huynh, Van Hoa Nguyen, and Thanh-Nghi Do. Random ensemble oblique decision stumps for classifying gene expression data. In Proceedings of the Ninth International Symposium on Information and Communication Technology, SoICT 2018, Danang City, Vietnam, December 06-07, 2018, pp. 137–144. ACM, 2018. doi: 10.1145/3287921.3287987. URL https://doi. org/10.1145/3287921.3287987.   
Alex Kantchelian, J. D. Tygar, and Anthony D. Joseph. Evasion and hardening of tree ensemble classifiers. In Maria-Florina Balcan and Kilian Q. Weinberger (eds.), Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19- 24, 2016, volume 48 of JMLR Workshop and Conference Proceedings, pp. 2387–2396. JMLR.org, 2016. URL http://proceedings.mlr.press/v48/kantchelian16.html.   
Andrei V Kelarev, Andrew Stranieri, JL Yearwood, and Herbert F Jelinek. Empirical study of decision trees and ensemble classifiers for monitoring of diabetes patients in pervasive healthcare. In 2012 15th International Conference on Network-Based Information Systems, pp. 441–446. IEEE, 2012.   
Mehul Madaan, Aniket Kumar, Chirag Keshri, Rachna Jain, and Preeti Nagrath. Loan default prediction using decision trees and random forest: A comparative study. IOP Conference Series: Materials Science and Engineering, 1022(1):012042, jan 2021. doi: 10.1088/1757-899X/1022/ 1/012042. URL https://dx.doi.org/10.1088/1757-899X/1022/1/012042.   
Gonzalo Mart´ınez-Munoz, Daniel Hern˜ andez-Lobato, and Alberto Su´ arez. Selection of decision´ stumps in bagging ensembles. In Joaquim Marques de Sa, Lu ´ ´ıs A. Alexandre, Wlodzislaw Duch, and Danilo P. Mandic (eds.), Artificial Neural Networks - ICANN 2007, 17th International Conference, Porto, Portugal, September 9-13, 2007, Proceedings, Part I, volume 4668 of Lecture Notes in Computer Science, pp. 319–328. Springer, 2007. doi: 10.1007/978-3-540-74690-4\ 33. URL https://doi.org/10.1007/978-3-540-74690-4\_33.

Majid Niazkar, Andrea Menapace, Bruno Brentan, Reza Piraei, David Jimenez, Pranav Dhawan, and Maurizio Righetti. Applications of xgboost in water resources engineering: A systematic literature review (dec 2018–may 2023). Environmental Modelling and Software, 174:105971, 2024. ISSN 1364-8152. doi: https://doi.org/10.1016/j.envsoft.2024.105971. URL https:// www.sciencedirect.com/science/article/pii/S136481522400032X.   
Francesco Ranzato and Marco Zanella. Abstract interpretation of decision tree ensemble classifiers. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04):5478–5486, Apr. 2020. doi: 10.1609/aaai.v34i04.5998. URL https://ojs.aaai.org/index.php/ AAAI/article/view/5998.   
Sagar Maan Shrestha and Aman Shakya. A customer churn prediction model using xgboost for the telecommunication industry in nepal. Procedia Computer Science, 215:652–661, 2022. ISSN 1877-0509. doi: https://doi.org/10.1016/j.procs.2022.12.067. URL https://www. sciencedirect.com/science/article/pii/S187705092202138X. 4th International Conference on Innovative Data Communication Technology and Application.   
John Tornblom and Simin Nadjm-Tehrani. An abstraction-refinement approach to formal verifi- ¨ cation of tree ensembles. In Computer Safety, Reliability, and Security: SAFECOMP 2019 Workshops, ASSURE, DECSoS, SASSUR, STRIVE, and WAISE, Turku, Finland, September 10, 2019, Proceedings, pp. 301–313, Berlin, Heidelberg, 2019. Springer-Verlag. ISBN 978-3- 030-26249-5. doi: 10.1007/978-3-030-26250-1 24. URL https://doi.org/10.1007/ 978-3-030-26250-1\_24.   
Namrita Varshney, Ashutosh Gupta, Arhaan Ahmad, Tanay V. Tayal, and S. Akshay. Data-aware and scalable sensitivity analysis for decision tree ensembles, 2026. URL https://arxiv. org/abs/2602.07453.   
Yihan Wang, Huan Zhang, Hongge Chen, Duane S. Boning, and Cho-Jui Hsieh. On lp-norm robustness of ensemble decision stumps and trees. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pp. 10104–10114. PMLR, 2020. URL http: //proceedings.mlr.press/v119/wang20aa.html.

# APPENDIX

The appendix is organized into 8 sections, in the order in which they occurred in the paper:

• In Section A, we provide a table of notation used throughout the paper for easy reference.   
• In Section B, we provide Additional Proofs and theoretical results relevant to Section 3.   
• In Section C, we provide data-aware sensitivity examples from Tabular Data as promised in Section 4.   
• In Section D, we prove the correctness of our encoding optimizations from Section 4.   
• In Section E, we describe the multi-class ensemble encoding into MILP that was left out of Section 5 due to lack of space.   
• In Section F, we describe the difficulty with using the VERITAS Devos et al. (2024) framework for comparison and how we can encode our sensitivity problem in that framework and perform some comparisons.   
• In Section G, we provide additional experimental setup details as well as additional experiments including:   
– (a) an ablation study for our encoding improvements/optimizations on binary and multiclass tree ensembles   
– (b) a multi-feature sensitivity analysis.

• In section H, we provide an illustrative example of our encoding for a small decision tree ensemble.

# A NOTATION TABLE

In this section, we provide a table of notation used throughout the paper for easy reference.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathcal{X}$ </td><td>input space of the classifiers</td></tr><tr><td> $\mathcal{Y}$ </td><td>output space of the classifiers</td></tr><tr><td> $\mathcal{F}$ </td><td>set of all features</td></tr><tr><td> $f$ </td><td>a feature in  $\mathcal{F}$ </td></tr><tr><td> $x$ </td><td>an input in  $\mathcal{X}$ </td></tr><tr><td> $x_f$ </td><td>value of feature  $f$  for input  $x$ </td></tr><tr><td> $T$ </td><td>a decision tree</td></tr><tr><td> $n$ </td><td>a node in a decision tree</td></tr><tr><td> $n.val$ </td><td>leaf value of leaf node  $n$ </td></tr><tr><td> $n.guard = X_f < \tau$ </td><td>guard condition of internal node  $n$ </td></tr><tr><td> $n.yes$ </td><td>child node of internal node  $n$  for true guard evaluation</td></tr><tr><td> $n.no$ </td><td>child node of internal node  $n$  for false guard evaluation</td></tr><tr><td> $T(x)$ </td><td>output of tree  $T$  on input  $x$ </td></tr><tr><td> $X_f$ </td><td>variable for feature  $f$ </td></tr><tr><td> $\mathcal{T}$ </td><td>set of decision trees in the ensemble</td></tr><tr><td> $\mathcal{T}_c$ </td><td>set of decision trees in the ensemble for class  $c$ </td></tr><tr><td> $\mathcal{E}$ </td><td>a decision tree ensemble</td></tr><tr><td> $x^{(1)}, x^{(2)}$ </td><td>inputs to the ensemble</td></tr><tr><td> $F$ </td><td>set of features to check sensitivity against</td></tr><tr><td> $g$ </td><td>minimum probability threshold for data-aware sensitivity</td></tr><tr><td> $u(x^{(1)}, x^{(2)})$ </td><td>utility function to maximize</td></tr><tr><td> $\mathcal{D}$ </td><td>training data samples</td></tr><tr><td> $\tau_{fk}$ </td><td>threshold in the  $k$ th guard of feature  $f$ </td></tr><tr><td> $K_f$ </td><td>the number of guards for feature  $f$ </td></tr><tr><td> $\pi_f$ </td><td>marginal probability function for feature  $f$ </td></tr><tr><td> $w$ </td><td>maximum size of the cavity constraints in Eq. 1.</td></tr><tr><td> $lb_i$ </td><td>lower bound on feature  $f_i$  in the cavity constraints in Eq. 1</td></tr><tr><td> $ub_i$ </td><td>upper bound on feature  $f_i$  in the cavity constraints in Eq. 1</td></tr><tr><td> $p_{fk}$ </td><td>Boolean variable for the truth value of  $k$ th guard of feature  $f$ </td></tr><tr><td> $l_i$ </td><td>Variable denoting leaf  $i$  is visited.</td></tr><tr><td> $\delta$ </td><td> $Sigmod^{-1}(0.5 - g)$ </td></tr><tr><td> $\mathcal{V}_F$ </td><td>set of all predicate variables for features in  $F$ </td></tr><tr><td> $\mathcal{A}$ </td><td>set of all leaf nodes in the ensemble</td></tr><tr><td> $\mathcal{U}$ </td><td>set of all unaffected leaves</td></tr><tr><td> $r_{fi}$ </td><td>Boolean variable to indicate that the feature in  $i$ th conjunct of cavity is feature  $f$ </td></tr><tr><td> $s_{ik}$ </td><td>Boolean variable to indicate that the  $i$ th conjunct of cavity uses  $k$ th guard of its feature as lower bound</td></tr><tr><td> $t_{ik}$ </td><td>Boolean variable to indicate that the  $i$ th conjunct of cavity uses  $k$ th guard of its feature as upper bound</td></tr></table>

# B ADDITIONAL PROOFS AND THEORETICAL RESULTS

Theorem B.1. The feature sensitivity problem with $| F | = 1$ is NP-Hard for tree ensembles with $d e p t h \ge 1$ .

Proof. Consider an instance of the integer subset sum problem, i.e., we are given a set of n integers U and an integer $k ,$ and we want to find a subset $U \subseteq { \bar { \mathcal { U } } }$ such that $\Sigma _ { l \in U } = k$ . We call the $i ^ { t h }$ integer $u _ { i }$ where i varies from 0 to $n - 1$ . For every index $i ,$ we create a Boolean feature $f _ { i } .$ . Then we create a decision tree of depth 1 which splits on $f _ { i }$ giving output 0 if $f _ { i }$ is false and $u _ { i }$ otherwise. We create another Boolean feature $f ^ { \prime }$ and a decision tree of depth 1 which splits on $f ^ { \prime }$ giving output $- k - 0 . 5$ if $f ^ { \prime }$ is false and $- k + 0 . 5$ otherwise. These trees are depicted in Figure 5. We call the ensemble of all these trees to be $\mathcal { E } : \{ 0 , 1 \} ^ { n + 1 }  \{ 0 , 1 \}$ with the trees being $T _ { i }$ where i varies from 0 to $n - 1$ and $T ^ { \prime }$ .

Claim. We claim that $\mathcal { E }$ is $\{ f ^ { \prime } \}$ -sensitive iff there exists $U \subseteq { \mathcal { U } }$ such that $\textstyle \sum _ { l \in U } l = k .$ .

To see this, consider a function $S : \{ 0 , 1 \} ^ { n + 1 } \to { \mathcal { P } } ( { \mathcal { U } } )$ where $\mathcal { P } ( \mathcal { U } )$ denotes the power set of U defined as $S ( x ) = \{ u _ { i } \in \mathcal { U } | x _ { i } = 1$ for some $i \in \{ 0 , 1 , \ldots , n - 1 \} \}$ . By construction of the trees, l∈S(x) i=0 i    x(2) such that E raw(x(1)) < 0 and E raw(x(2)) > 0 and x ⊥f ′ = x )⊥f ′ , where x⊥f ′ refers to input x $\Sigma _ { l \in S ( x ) } l = \Sigma _ { i = 0 } ^ { n - 1 } T _ { i } ( x )$ $x ^ { ( 2 ) }$ $\mathcal { E } ^ { r a w } ( x ^ { ( 1 ) } ) < 0$ where $x \in \{ 0 , 1 \} ^ { n + 1 }$ $\mathcal { E } ^ { r a w } ( x ^ { ( 2 ) } ) > 0$ . If E is sensitive to (1) (2 $x _ { \perp f ^ { \prime } } ^ { ( 1 ) } = x _ { \perp f ^ { \prime } } ^ { ( 2 ) }$ $\{ f ^ { \prime } \}$ , then there exist $x \_ f \prime$ $x ^ { ( 1 ) }$ and projected into all features except $f ^ { \prime } .$ By construction, $x _ { f ^ { \prime } } ^ { ( 1 ) } = 0$ and $x _ { f ^ { \prime } } ^ { ( 2 ) } = 1$ . Then $\mathcal { E } ^ { r a w } ( x ^ { ( 1 ) } ) =$ $\Sigma _ { i = 0 } ^ { n - 1 } T _ { i } ( x ^ { ( 1 ) } ) + T ^ { \prime } ( x ^ { ( 1 ) } ) < 0 \implies \Sigma _ { l \in S ( x ^ { ( 1 ) } ) } l - k - 0 . 5 < 0 \implies \Sigma _ { l \in S ( x ^ { ( 1 ) } ) } l < k + 0 . 5$ . Similarly, $\mathcal { E } ^ { r a w } ( x ^ { ( 2 ) } ) = \Sigma _ { i = 0 } ^ { n - 1 } T _ { i } ( x ^ { ( 2 ) } ) + T ^ { \prime } ( x ^ { ( 2 ) } ) > 0 \implies \Sigma _ { l \in S ( x ^ { ( 2 ) } ) } l - k + 0 . 5 > 0 \implies \Sigma _ { l \in S ( x ^ { ( 2 ) } ) } l > 0 .$ $k - 0 . 5$ . Also, by construction of $S , S ( x ^ { ( 1 ) } ) = S ( x ^ { ( 2 ) } )$ since $x ^ { ( 1 ) }$ and $x ^ { ( 2 ) }$ only differ on $f ^ { \prime }$ and S is independent of that value. Let $S ( x ^ { ( 1 ) } ) = \overline { { S ( x ^ { ( 2 ) } ) } } = U$ . Thus, $k - 0 . 5 < \Sigma _ { l \in U } l < k + 0 . 5$ . As all numbers are integers, $\Sigma _ { l \in U } l = k$ and thus there exists $U \subseteq { \mathcal { U } }$ such that $\Sigma _ { l \in U } l = k$ .

$U \subseteq { \mathcal { U } }$ $\Sigma _ { l \in U } l = k$ $x ^ { ( 1 ) }$ $x ^ { ( 2 ) }$ such that  Also, x(1)f ′ $x _ { i } ^ { ( 1 ) } = x _ { i } ^ { ( 2 ) } = 1$ $u _ { i } \in U$ $x _ { j } ^ { ( 1 ) } = x _ { j } ^ { ( 2 ) } = 0 { \mathrm { i f } } u _ { j } \notin U$ $i , j \in \{ 0 , 1 , \ldots , n - 1 \}$ $x _ { f ^ { \prime } } ^ { ( 2 ) } = 1$ Note ${ \cal S } ( x ^ { ( 1 ) } ) \dot { = } { \cal S } ( x ^ { ( 2 ) } ) = U$ . Thus, $\mathcal { E } ( x ^ { ( 1 ) } ) = \Sigma _ { l \in S ( x ^ { ( 1 ) } ) } l + T ^ { \prime } ( x ^ { ( 1 ) } ) = k - k - 0 . 5 = - 0 . 5 < 0$ and $\mathcal { E } ( x ^ { ( 2 ) } ) = \Sigma _ { l \in S ( x ^ { ( 2 ) } ) } l + T ^ { \prime } ( x ^ { ( 2 ) } ) = k - k + 0 . 5 = 0 . 5 > 0 . \mathrm { ~ A l s o ~ } x _ { \perp f ^ { \prime } } ^ { ( 1 ) } = x _ { \perp f ^ { \prime } } ^ { ( 2 ) }$ . Thus, E is sensitive to $\{ f ^ { \prime } \}$ as the above $x ^ { ( 1 ) }$ and $x ^ { ( 2 ) }$ are a required pair of inputs to show sensitivity.

Thus, by proving in both directions, we have shown if we can solve sensitivity for decision tree ensembles of depth-1, then we can solve integer subset sum problem. Thus, sensitivity is at least as hard as integer subset sum and thus it is NP-Hard for depth 1.

![](images/fc4448cfecb6e0dba794e5ad9487471de99de246d515eb410a1468e4e49922bc.jpg)

![](images/8cfdacf979771be42a4ec4e8ab072454fcee7fdac56fcc5d1a4b818f486504a4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["f' < 1"] -->|F| B["-k + 0.5"]
    A -->|T| C["-k - 0.5"]
```
</details>

Figure 5: Trees for Proof of Theorem B.1

Finally, we show that when $| { \mathcal { F } } / F |$ is bounded, we can solve the depth 1 problem in polynomial time.

Theorem B.2. The feature sensitivity problem with bounded $| { \mathcal { F } } / F |$ is solvable in polynomial time for tree ensembles with depth 1.

Proof. Given a sensitivity problem with decision tree ensemble of trees of depth $1 \ \mathcal { E } \ =$ $\{ T _ { 0 } , T _ { 1 } , \dots \}$ , feature set ${ \mathcal { F } } _ { : }$ , features for checking sensitivity against F and a scalar k such that $| { \mathcal { F } } / F | \leq k .$ , we can solve the problem in polynomial time. Let $| { \mathcal { E } } | = m { \mathrm { ~ i . e ~ } }$ . there are total m decision trees and $| { \mathcal { F } } | = n$ i.e there a total of n features.

As each tree splits on only 1 feature, we can create a set of trees corresponding to each feature. Thus, we create a function $S ( f )$ from the set of feature space to a subset of all trees. Note, it is a partition of all the trees as each tree will split on exactly one feature.

For all $f \in F _ { \cdot }$ , consider the set $S ( f )$ . We will calculate the minimum and maximum value of sum output of trees in $S ( f )$ and their corresponding features values. There are a total of $\vert S ( f ) \vert + 1$ distinct possible values and thus can be found in $O ( m )$ time.

We will add the minimums and maximums found above and get a global minimum and maximum value say m and $M .$ We need to find whether there exists an assignment to features in $\mathcal { F } / F$ such that the sum of output corresponding to trees of these features lies between −M and −m.

For each feature $f \in { \mathcal { F } } / F$ , the possible number of distinct outputs is $\vert S ( f ) \vert + 1$ . Therefore, the total possible number of outputs are $\Pi _ { f \in { \mathcal { F } } / F } ( S ( f ) + 1 )$ which is $O ( m ^ { k } )$ ). Thus, by checking for all possible outputs, we can find whether such output exists or not. If it does, then the ensemble is sensitive otherwise it isn’t. Thus we have a polynomial time algorithm as k is not a parameter but a bound. □

# C DATA-AWARE SENSITIVITY EXAMPLES IN TABULAR DOMAINS

This section presents examples that demonstrate the effectiveness of incorporating data-awareness into sensitivity analysis, using models trained on tabular datasets. In each case, we compare sensitive input pairs discovered with and without data-aware methods, showing how the inclusion of data distribution knowledge leads to more realistic counterexamples. Please note that the IJCNN model is trained by Chen et al. (2019b), and the dataset for this model is unfortunately not available with the original feature names. The feature names are simply mentioned as f1 to f22.

Sensitive Example of IJCCN Chen et al. (2019b): In the examples 1 and example 2 below, we analyse the sensitivity with respect to features 3 and 15 in the model trained on the IJCNN dataset. Both methods detect a sensitive pair by varying only these features respectively, resulting in change in the model’s prediction. However, the distance from the training data distribution reveals a clear difference: The pair found without data-awareness has a distance of 1.077(example 1) and 1.06 (example 2), indicating it is quite far from any possible realistic data point and may not be very helpful in practice. In contrast, the pair identified with data-awareness has a distance of just 0.327(example 1) and 0.34 (example 2), meaning it is much closer to the data distribution and training data.The insensitive features in the training data points that are far away from the sensitive pair are highlighted with cyan color.

Sensitive Examples of Adult: ( based on Adult dataset Dua & Graff (2019)): Here, we present the analysis of the ‘adult’ dataset in examples 3 and 4, this time examining the sensitivity of the features $\mathrm { \dot { \cdot } a g e ^ { \mathrm { \dot { \ } } } }$ and $\cdot _ { \mathrm { s e x } } ,$ . The non-data-aware (baseline) method identifies the sensitive pairs with larger distances of 0.656 and 0.9899 for sensitive features ‘age’ and ‘sex’, respectively. The data-aware method, however, identifies pairs that are much closer to the data distances of 0.019 and 0.108. In example 4, the baseline(non-data-aware) method reports an ‘age’ value of 86, even though the closest datapoint has an age of $\cdot 4 0 ^ { \cdot }$ . The data-aware method, however, identifies a pair with an age value of $\cdot 4 6 ^ { \cdot }$ , very close to the nearest datapoint, which has an age of $\cdot 4 5 '$ . A similar pattern appears for capital-gain and capital-loss. The baseline method(non-data-aware) selects a pair with values 10,585 (capital-gain) and 3,142 (capital-loss), while the nearest datapoint has values 0 and 2,258. The data-aware method, by comparison, identifies a pair where both features are 0, matching the nearest datapoint exactly. The insensitive features in the training data points that are far away from the sensitive pair are highlighted with cyan color.

Sensistive Examples of Pimadiabetes: (Dua & Graff (2019)): Finally, we present one more sensitive pair example, the Pima Diabetes dataset (with no categorical features), in example 5, and examine the sensitivity of feature (‘BloodPressure’). Again, the non-data-aware method identifies the sensitive pairs with larger distances of 0.35 , where almost all feature values are far from the data(which we show in cyan color) . However, the data-aware method again identifies the pairs that are much closer to the data distances of 0.03, where only feature ”Glucose” is far and the others are close. The examples clearly show that data-aware search finds sensitive pairs closer to the data.

# D PROOF OF THEOREM 5.1 (CORRECTNESS OF OPTIMIZATIONS)

In this section, we prove Theorem 5.1, which says that our encoding optimiziations, that gave rise to significant improvement in performance, are sound.

Proof. We first prove that equation UnAff is already subsumed by equation 2-equation 6. This is established using a proof by contradiction. Assume to the contrary and let $n \in \mathcal { U }$ such that $l _ { n } ^ { ( 1 ) } \neq l _ { n } ^ { ( 2 ) }$ . Since the leaf variables are forced to be either 0 or 1 (by equation 4), assume without loss of generality that $l _ { n } ^ { ( 1 ) } = 1$ and $l _ { n } ^ { ( 2 ) } = 0$ .

Example 1: IJCNN ROBUST Chen et al. (2019b)   
Sensitive Point Found Without Data-Awareness Analysis For Sensitive Feature 3   
```txt
Point1: {'f1': 0.0, 'f2': 1.0, 'f3': 0.0, 'f4': 0.0, 'f5': 1.0, 'f6': 1.0, 'f7': 0.0, 'f8': 0.0, 'f9': 1.0, 'f10': 0.0, 'f11': 0.872834, 'f12': 1.21062, 'f13': 0.637325, 'f14': 0.356398, 'f15': 0.482769, 'f16': 0.390789, 'f17': 0.609402, 'f18': 0.558829, 'f19': 0.607626, 'f20': 0.696077, 'f21': 0.448006, 'f22': 0.619263}
Point2: {'f1': 0.0, 'f2': 1.0, 'f3': 1.0, 'f4': 0.0, 'f5': 1.0, 'f6': 1.0, 'f7': 0.0, 'f8': 0.0, 'f9': 1.0, 'f10': 0.0, 'f11': 0.872834, 'f12': 1.210621, 'f13': 0.637325, 'f14': 0.356398, 'f15': 0.482769, 'f16': 0.390789, 'f17': 0.609402, 'f18': 0.558829, 'f19': 0.558829, 'f20': 0.696077, 'f21': 0.448006, 'f22': 0.619263}
Distance from data: 1.07757866169
Nearest Training Datapoint: {'f1': 0.0, 'f2': 1.0, 'f3': 0.0, 'f4': 0.0, 'f5': 0.0, 'f6': 0.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 0.0, 'f11': 0.540556, 'f12': 0.250375, 'f13': 0.467001, 'f14': 0.470297, 'f15': 0.570899, 'f16': 0.527529, 'f17': 0.517118, 'f18': 0.51931, 'f19': 0.51546, 'f20': 0.55852, 'f21': 0.53037, 'f22': 0.52894} 
```

Sensitive Point Found With Data-Awareness Analysis For Sensitive Feature 3   
```csv
Point1: {'f1': 0.0, 'f2': 0.0, 'f3': 0.0, 'f4': 0.0, 'f5': 1.0, 'f6': 0.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 0.0, 'f11': 0.678628, 'f12': 0.541327, 'f13': 0.512386, 'f14': 0.516051, 'f15': 0.516459, 'f16': 0.497491, 'f17': 0.475932, 'f18': 0.495826, 'f19': 0.459603, 'f20': 0.502477, 'f21': 0.50531, 'f22': 0.507861}
Point2: {'f1': 0.0, 'f2': 0.0, 'f3': 1.0, 'f4': 0.0, 'f5': 1.0, 'f6': 0.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 0.0, 'f11': 0.678628, 'f12': 0.541327, 'f 13': 0.512386, 'f14': 0.516051, 'f15': 0.516459, 'f16': 0.497491, 'f17': 0.475932, 'f18': 0.495826, 'f19': 0.459563, 'f20': 0.502477, 'f21': 0.50531, 'f22': 0.507861}
Distance from data: 0.327039
Nearest Training Datapoint: {'f1': 0.0, 'f2': 0.0, 'f3': 0.0, 'f4': 0.0, 'f5': 1.0, 'f6': 0.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 0.0, 'f11': 0 .536865 ,'f12': 0 .250375 ,'f13': 0 .51317 ,'f14': 0 .50743 ,'f15': 0 .526203 ,'f16': 0 .497706 ,'f17': 0 .484023 ,'f18': 0 .524588 ,'f19': 0 .541842 ,'f20': 0 .467001 ,'f21': 0 .470298 ,'f22': 0 .570898} 
```

Example 1: IJCNN ROBUST Chen et al. (2019b)

Example 2: IJCNN ROBUST Chen et al. (2019b)   
Sensitive Point Found Without Data-Awareness Analysis For Sensitive Feature 15   
```txt
Point1: {'f1': 0.0, 'f2': 1.0, 'f3': 1.0, 'f4': 0.0, 'f5': 1.0, 'f6': 1.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 1.0, 'f11': 0.822872, 'f12': 0.276149, 'f13': 0.448491, 'f14': 0.653076, 'f15': 0.64666, 'f16': 0.615426, 'f17': 0.36238, 'f18': 0.561026, 'f19': 0.533231, 'f20': 1.199128, 'f21': 0.394238, 'f22': 0.558881}
Point2: {'f1': 0.0, 'f2': 1.0, 'f3': 1.0, 'f4': 0.0, 'f5': 1.0, 'f6': 1.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 1.0, 'f11': 0 .822872, 'f12': 0.276149, 'f13': 0.448491, 'f14': 0.653076, 'f15': 0.30856, 'f16': 0.615426, 'f17': 0.36238, 'f18': 0.561026, 'f19': 0.533231, 'f20': 1.199128, 'f21': 0.394238, 'f22': 0. 558881}
Distance from data 1.0647264749
Nearest Training Datapoint:: {'f1': 0.0, 'f2': 1.0, 'f3': 0.0, 'f4': 0.0, 'f5': 0.0, 'f6': 0.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 0.0, 'f11': 0 .579145, 'f12': 0 .18406, 'f13': 0 .456363, 'f14': 0 .543959, 'f15': 0 .531502, 'f16': 0 .449082, 'f17': 0 .483391, 'f18': 0 .520606, 'f19': 0 .511848, 'f20': 0 .527679, 'f21': 0 .474918, 'f22': 0 .477095} 
```

Sensitive Point Found With Data-Awareness Analysis For Sensitive Feature 15   
```jsonl
Point1 : {'f1': 0.0, 'f2': 1.0, 'f3': 0.0, 'f4': 0.0, 'f5': 0.0, 'f6': 0.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 0.0, 'f11': 0.797221, 'f12': 0.341174, 'f13': 0.540222, 'f14': 0.380818, 'f15': 1.168982, 'f16': 0.457969, 'f17': 0.368609, 'f18': 0.457974, 'f19': 0.527069, 'f20': 0.399163, 'f21': 0.567169, 'f22': 0.501212}
Point2 : {'f1': 0.0, 'f2': 1.0, 'f3': 0.0, 'f4': 0.0, 'f5': 0.0, 'f6': 0.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 0.0, 'f11': 0 .797221, 'f12': 0.341174, 'f13': 0.540222, 'f14': 0.380818, 'f15': 0.412158, 'f16': 0.457969, 'f17': 0.368609, 'f18': 0.457974, 'f19': 0.527069, 'f20': 0.399163, 'f21': 0.567169, 'f22': 0. 501212}
Distance from data 0.3449
Nearest Training Datapoint:: {'f1': 0.0, 'f2': 1.0, 'f3': 0.0, 'f4': 0.0, 'f5': 0.0, 'f6': 0.0, 'f7': 0.0, 'f8': 0.0, 'f9': 0.0, 'f10': 0.0, 'f11': 0 .579295, 'f12': 0 .16411, 'f13': 0 .44351, 'f14': 0 .45535, 'f15': 0 .542591, 'f16': 0 .501968, 'f17': 0 .500353, 'f18': 0 .5093, 'f19': 0 .495474, 'f20': 0 .515701, 'f21': 0 .511422, 'f22': 0 .517442} 
```

Example 2: IJCNN ROBUST Chen et al. (2019b)

Example 3: Adult Dua & Graff (2019)   
Sensitive Point Found Without Data-Awareness Analysis For Sensitive Feature ‘age’   
```txt
Point1: {'age':66, 'workclass': 'Never-worked', 'fnlwgt': 574792.14018, 'education': 'Some-college', 'education-num': 16, 'marital-status': 'Widowed', 'occupation': 'Transport-moving', 'relationship': 'Unmarried', 'race': 'White', 'sex': 'Male', 'capital-gain': 10585, 'capital-loss': 2309, 'hours-per-week': 92, 'native-country': 'Poland'},  
Point2: {'age':86, 'workclass': 'Never-worked', 'fnlwgt': 574792.14018, 'education': 'Some-college', 'education-num': 16, 'marital-status': 'Widowed', 'occupation': 'Transport-moving', 'relationship': 'Unmarried', 'race': 'White', 'sex': 'Male', 'capital-gain': 10585, 'capital-loss': 1309, 'hours-per-week': 92, 'native-country': 'Poland'}  
Distance from data : 0.6569890  
Nearest Training Datapoint: {'age': 32, 'workclass': 'Private', 'fnlwgt': 226975, 'education': 'Some-college', 'education-num': 10, 'marital-status': 'Never-married', 'occupation': 'Sales', 'relationship': 'Own-child', 'race': 'White', 'sex': 'Male', 'capital-gain': 0, 'capital-loss': 1876, 'hours-per-week': 60, 'native-country': 'United-States'} 
```

Sensitive Point Found With Data-Awareness Analysis For Sensitive Feature ‘age’   
```csv
Point1: {'age': 46, 'workclass': 'Self-emp-inc', 'fnlwgt': 180532.54372, 'education': 'Doctorate', 'education-num': 13.500002, 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Husband', 'race': 'Black', 'sex': 'Female', 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'Puerto-Rico'},  
Point2: {'age': 33, 'workclass': 'Self-emp-inc', 'fnlwgt': 180532.54372, 'education': 'Doctorate', 'education-num': 13.500002, 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Husband', 'race': 'Black', 'sex': 'Female', 'capital-gain': 0. 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'Puerto-Rico'}  
Distance from data 0.0191765741  
Nearest Training DataPoint: {'age': 44, 'workclass': 'Private', 'fnlwgt': 211759, 'education': 'Bachelors', 'education-num': 13, 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Husband', 'race': 'Other', 'sex': 'Male', 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'Puerto-Rico'}} 
```

Example 3: Adult Dua & Graff (2019)

Example 4: Adult Dua & Graff (2019)   
Sensitive Point Found Without Data-Awareness Analysis For Sensitive Feature ‘sex’   
```txt
Point1: {'age': 86, 'workclass': 'Without-pay', 'fnlwgt': 520260.32927, 'education': 'HS-grad', 'education-num': 16.0, 'marital-status': 'Never-married', 'occupation': 'Transport-moving', 'relationship': 'Unmarried', 'race': 'Amer-Indian-Eskimo', 'sex': 'Female', 'capital-gain': 10585, 'capital-loss': 3142, 'hours-per-week': 99, 'native-country': 'Laos'}},  
Point2: {'age': 86, 'workclass': 'Without-pay', 'fnlwgt': 520260.32927, 'education': 'HS-grad', 'education-num': 16.0, 'marital-status': 'Never-married', 'occupation': 'Transport-moving', 'relationship': 'Unmarried', 'race': 'Amer-Indian-Eskimo', 'sex': 'Male', 'capital-gain': 10585, 'capital-loss': 3142, 'hours-per-week': 99, 'native-country': 'Laos'}}  
Distance from data: 0.98998791099  
Nearest Training Datapoint: {'age': 40, 'workclass': 'Private', 'fnlwgt': 287983, 'education': 'Bachelors', 'education-num': 13, 'marital-status': 'Never-married', 'occupation': 'Tech-support', 'relationship': 'Not-in-family', 'race': 'Asian-Pac-Islander', 'sex': 'Female', 'capital-gain': 0, 'capital-loss': 2258, 'hours-per-week': 48, 'native-country': 'Philippines'},} 
```

Sensitive Point Found With Data-Awareness Analysis For Sensitive Feature ‘sex’   
```txt
Point1: {'age': 46, 'workclass': 'Self-emp-inc', 'fnlwgt': 284508.95444, 'education': 'Doctorate', 'education-num': 13, 'marital-status': 'Married-civ-spouse', 'occupation': 'Craft-repair', 'relationship': 'Husband', 'race': 'Black', 'sex': 'Male', 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 50, 'native-country': 'France'},
Point2: {'age': 46, 'workclass': 'Self-emp-inc', 'fnlwgt': 284508.95444, 'education': 'Doctorate', 'education-num': 13, 'marital-status': 'Married-civ-spouse', 'occupation': 'Craft-repair', 'relationship': 'Husband', 'race': 'Black', 'sex': 'Female', 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 50, 'native-country': 'France'}
Distance from data: 0.1081739837784343
Nearest Training Datapoint: {'age': 45, 'workclass': 'Private', 'fnlwgt': 238567, 'education': 'Bachelors', 'education-num': 13, 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'England'} 
```

Example 4: Adult Dua & Graff (2019)

Example 5: Pimadiabetes Dua & Graff (2019)   
Sensitive Point Found Without Data-Awareness Analysis For Sensitive Feature 2   
```javascript
Point1: {'Pregnancies': 17, 'Glucose': 188, 'BloodPressure': 122, 'SkinThickness': 33, 'Insulin': 846, 'BMI': 67.1, 'DiabetesPedigreeFunction': 2.42, 'Age': 81},
Point2: {'Pregnancies': 17.0, 'Glucose': 188, 'BloodPressure': 76, 'SkinThickness': 33, 'Insulin': 846, 'BMI': 67.1, 'DiabetesPedigreeFunction': 2.42, 'Age': 81}
Distance from data: 0.3534358888
Nearest Training Datapoint: {'Pregnancies': 10, 'Glucose': 148, 'BloodPressure': 84, 'SkinThickness': 48, 'Insulin': 237, 'BMI': 37.6, 'DiabetesPedigreeFunction': 1.001, 'Age': 51} 
```

Sensitive Point Found With Data-Awareness Analysis For Sensitive Feature 2   
```txt
Point 1: {'Pregnancies': 2e-06, 'Glucose': 139, 'BloodPressure': 70, 'SkinThickness': 0, 'Insulin': 0, 'BMI': 32.75, 'DiabetesPedigreeFunction': 0.3595, 'Age': 21}
Point2: {'Pregnancies': 2e-06, 'Glucose': 139, 'BloodPressure': 79, 'SkinThickness': 0, 'Insulin': 0, 'BMI': 32.75, 'DiabetesPedigreeFunction': 0.3595, 'Age': 21}
Distance from data: 0.03051399
Nearest Training Datapoint: {'Pregnancies': 0, 'Glucose': 132, 'BloodPressure': 78, 'SkinThickness': 0, 'Insulin': 0, 'BMI': 32.4, 'DiabetesPedigreeFunction': 0.393, 'Age': 21} 
```  
Example 5: Pimadiabetes Dua & Graff (2019)

By equation 3, there exists leaf $n ^ { \prime }$ such that $l _ { n ^ { \prime } } ^ { ( 2 ) } = 1$ n and the leaves n and $n ^ { \prime }$ belong to the same tree. Let $n ^ { \dag \prime }$ be the last common node in the paths from the root to leaves n and $n ^ { \prime } { \mathrm { . } }$ , respectively. Let $n ^ { \prime \prime }$ be labeled with with $X _ { f } < \tau _ { k }$ and $p _ { f k }$ be the corresponding predicate. Since $n ^ { \prime \prime }$ is in the ancestry of $n ^ { \prime }$ and $n \in \mathcal { U }$ , we have that $p _ { f k } ^ { ( 1 ) } = p _ { f k } ^ { ( 2 ) }$ (by equation 6).

Without loss of generality, assume that leaf n is present in the subtree rooted at $n ^ { \prime \prime }$ .no, while leaf n′ is present in the one rooted at $n ^ { \prime \prime } . y e s$ . Since $l _ { n } ^ { ( 1 ) } = 1$ , equation 5 implies that $1 - 1 \ge p _ { f k } ^ { ( 1 ) } \implies$ =⇒ $p _ { f k } ^ { ( 1 ) } = 0 .$ . At the same time, since $l _ { n ^ { \prime } } ^ { ( 2 ) } = 1$ , equation 5 implies that $p _ { f k } ^ { ( 2 ) } \geq 1 \implies p _ { f k } ^ { ( 2 ) } = \bar { 1 } \neq p _ { f k } ^ { ( 1 ) }$ leading to a contradiction. Hence, $l _ { n } ^ { ( 1 ) } = l _ { n } ^ { ( 2 ) } \forall i \in \mathcal { U }$ is implied by equation 2-equation 6 and hence the set of feasible solutions does not change on the addition of equation UnAff.

As mentioned earlier, we show that equation UnAff and equation Gap-bin together imply equation Aff-bin. Subtracting the two inequalities in equation Gap-bin, we get that when equation Gapbin holds, then $\begin{array} { r } { \sum l _ { n } ^ { ( 1 ) } n . v a l - l _ { n } ^ { ( 2 ) } n . v a l \geq 2 \times \delta } \end{array}$ .

However, for the leaves belonging to U, the difference terms are 0 by definition, i.e. $\begin{array} { r } { \sum _ { n \in \mathcal { U } } l _ { n } ^ { ( 1 ) } n . v a l - l _ { n } ^ { ( 2 ) } n . v a l \ = \ 0 } \end{array}$ . Using these two equations we conclude that if equation Un-Aff holds and equation Gap-bin holds, then we must have $\begin{array} { r } { \sum _ { n \notin \mathcal { U } } l _ { n } ^ { ( 1 ) } n . v a l - l _ { n } ^ { ( 2 ) } n . v a l \ge 2 \times \delta , } \end{array}$ which completes the proof. □

# E MILP ENCODING FOR MULTICLASS SENSITIVITY

In the main paper, we had extended the Sensitivity problem from binary to multiclass classification. Here we provide the details regarding how we extend the MILP encoding to tackle the multiclass setting. We observe that for the $\mathsf { \bar { ( } } g , F , \mathsf { \bar { c } } ^ { ( 1 ) } , c ^ { ( 2 ) } )$ −sensitivity problem for a multiclass ensemble, only equation Gap-bin, equation Aff-bin and equation Obj-bin need to be modified. We now describe these changes. Let $\bar { \mathcal { L } _ { c } }$ denote the indices of the leaf variables corresponding to the trees of class c.

The change in equation Gap-bin follows from Definition 6.1. To encode that $\forall c \neq c ^ { ( 1 ) } , \mathcal { E } _ { { c } ^ { ( 1 ) } } ^ { p r o b } ( x ^ { ( 1 ) } ) \ \geq \ g \ \times \mathcal { E } _ { { c } } ^ { p r o b } ( x ^ { ( 1 ) } )$ , we instead move to the space of ${ { \mathcal E } ^ { R a w } }$ , i.e. the values before applying SOFTMAX. Given the definition of SOFTMAX,

$$
\mathcal {E} _ {c ^ {(1)}} ^ {p r o b} (x ^ {(1)}) \geq g \times \mathcal {E} _ {c} ^ {p r o b} (x ^ {(1)})
$$

$$
\Longrightarrow \frac {\operatorname{Exp} (\mathcal {E} _ {c ^ {(1)}} ^ {r a w} (x ^ {(1)}))}{\sum \operatorname{Exp} (\mathcal {E} _ {c _ {i}} ^ {r a w} (x ^ {(1)}))} \geq g \times \frac {\operatorname{Exp} (\mathcal {E} _ {c ^ {r a w} (x ^ {(1)})})}{\sum \operatorname{Exp} (\mathcal {E} _ {c _ {i}} ^ {r a w} (x ^ {(1)}))}
$$

$$
\Longrightarrow \mathcal {E} _ {c ^ {(1)}} ^ {\text { raw }} (x ^ {(1)}) \geq \mathcal {E} _ {c} ^ {\text { raw }} (x ^ {(1)}) + \ln g
$$

We call ln g as η, to get the new gap constraints

$$
\bigwedge_ {c \neq c ^ {(1)}} \sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(1)}}} l _ {n} ^ {(1)} n. v a l > \sum_ {l _ {n} \in \mathcal {L} _ {c}} l _ {n} ^ {(1)} n. v a l + \eta \tag {Gap-multi}
$$

$$
\bigwedge_ {c \neq c ^ {(2)}} \sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(2)}}} l _ {n} ^ {(2)} n. v a l > \sum_ {l _ {n} \in \mathcal {L} _ {c}} l _ {n} ^ {(2)} n. v a l + \eta
$$

With this new constraint gap constraint, we can arrive at a constraint for the affected leaves, similar to equation Aff-bin, by adding the two constraints in equation Gap-multi and using reasoning analogous to that of the binary classification setting.

$$
\sum_{\substack{l_{n}\in \mathcal{L}_{c^{(1)}}\\ n\not\in\mathcal{U}}}\left(l_{n}^{(1)}n.val - l_{n}^{(2)}n.val\right) + \sum_{\substack{l_{n}\in \mathcal{L}_{c^{(2)}}\\ n\not\in\mathcal{U}}}\left(l_{n}^{(2)}n.val - l_{n}^{(1)}n.val\right) > 2\times \eta \qquad (\text{Aff - multi})
$$

An objective function can be formulated as before:

$$
\operatorname{MAX} \sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(1)}}} \left(l _ {n} ^ {(1)} n. v a l - l _ {n} ^ {(2)} n. v a l\right) + \sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(2)}}} \left(l _ {n} ^ {(2)} n. v a l - l _ {n} ^ {(1)} n. v a l\right) \quad (\text { Obj   -   multi })
$$

Theorem E.1. The set of feasible solutions of the MILP defined by equation 2 ∧ equation 3 ∧ equation 4 ∧ equation 5 ∧ equation 6 ∧ equation Gap − multi and that of the MILP defined by adding equation Aff-multi are equal.

The only difficult part in the proof is to see how we obtain equation Aff-multi. Let us choose $c = c ^ { ( 2 ) }$ in the first half and $c = c ^ { ( 1 ) }$ in the second half in equation Gap-multi. We obtain

$$
\sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(1)}}} l _ {n} ^ {(1)} n. v a l > \sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(2)}}} l _ {n} ^ {(1)} n. v a l + \eta \tag {8}
$$

$$
\sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(2)}}} l _ {n} ^ {(2)} n. v a l > \sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(1)}}} l _ {n} ^ {(2)} n. v a l + \eta
$$

Add the two equations

$$
\sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(1)}}} (l _ {n} ^ {(1)} - l _ {n} ^ {(2)}) n. v a l + \sum_ {l _ {n} \in \mathcal {L} _ {c ^ {(2)}}} (l _ {n} ^ {(2)} - l _ {n} ^ {(1)}) n. v a l > 2 \eta \tag {9}
$$

Since for the unaffected leafs $l _ { n } ^ { ( 1 ) } - l _ { n } ^ { ( 2 ) }$ is zero. We derive the desired equation.

$$
\sum_{\substack{l_{n}\in \mathcal{L}_{c^{(1)}}\\ n\not\in\mathcal{U}}}\left(l_{n}^{(1)}n.val - l_{n}^{(2)}n.val\right) + \sum_{\substack{l_{n}\in \mathcal{L}_{c^{(2)}}\\ n\not\in\mathcal{U}}}\left(l_{n}^{(2)}n.val - l_{n}^{(1)}n.val\right) > 2\times \eta \qquad (\text{Aff - multi})
$$

# F COMPARISON WITH VERITAS

One of the comments that we left in the main paper was the comparison or lack there-of with the tool VERITAS, a versatile tool for robustness verification of decision tree ensembles, for which there is a multi-class variant available. In this section, we explain why we cannot easily compare with that tool and also how it can be modified so that we can compare it. Firstly, VERITAS does not solve the problem directly as it is not designed for sensitivity. So as the first step, we modified the tool enable the multiclass sensitivity analysis. Note that VERITAS is a more generalizable tool and we approach the problem differently. To encode the multiclass feature sensitivity problem in VERITAS, we create two instances of a given tree ensemble and optimize the following objective:

$$
\operatorname{MAX} \left(D _ {0} (x ^ {(1)}, x ^ {(2)}) - \operatorname{MAX} _ {c, c \neq 0} (D _ {c} (x ^ {(1)}, x ^ {(2)}))\right) \tag {10}
$$

where D is defined as follows:

$$
D _ {c} (x ^ {(1)}, x ^ {(2)}) = \left\{ \begin{array}{l l} \mathcal {E} _ {c ^ {(1)}} ^ {r a w} (x ^ {(1)}) + \mathcal {E} _ {c ^ {(2)}} ^ {r a w} (x ^ {(2)}), & \text {if c = 0} \\ \mathcal {E} _ {0} ^ {r a w} (x ^ {(1)}) + \mathcal {E} _ {0} ^ {r a w} (x ^ {(2)}), & \text {if c = c^{(1)}} \\ \mathcal {E} _ {c ^ {(2)}} ^ {r a w} (x ^ {(1)}) + \mathcal {E} _ {c ^ {(1)}} ^ {r a w} (x ^ {(2)}), & \text {if c = c^{(2)}} \\ \mathcal {E} _ {c} ^ {r a w} (x ^ {(1)}) + \mathcal {E} _ {c} ^ {r a w} (x ^ {(2)}), & o t h e r w i s e \end{array} \right.
$$

We define the objective value found by VERITAS being ”better than” ENSENSE if the output of VERITAS is greater than $2 \times \eta$ .

Here we provide the proof of correctness of this comparison.

From the equation Gap-multi, we can conclude:

$$
\mathcal {E} _ {c ^ {(1)}} ^ {\text { raw }} (x ^ {(1)}) - \operatorname{MAX} _ {c, c \neq c ^ {(1)}} \mathcal {E} _ {c} ^ {\text { raw }} (x ^ {(1)}) \geq \eta \tag {11}
$$

$$
\mathcal {E} _ {c ^ {(2)}} ^ {\text { raw }} (x ^ {(2)}) - \text { MAX } _ {c, c \neq c ^ {(2)}} \mathcal {E} _ {c} ^ {\text { raw }} (x ^ {(2)}) \geq \eta \tag {13}
$$

Ideally we would like to maximize the sum of LHS of the above equations. We will prove that our objective is an upper bound for the output described above.

Claim. For all $x ^ { ( 1 ) }$ and $\begin{array} { r } { x ^ { ( 2 ) } , ~ : ~ D _ { 0 } ( x ^ { ( 1 ) } , x ^ { ( 2 ) } ) - \mathrm { M A X } _ { c , c \neq 0 } ( D _ { c } ( x ^ { ( 1 ) } , x ^ { ( 2 ) } ) ) \geq \mathcal { E } _ { { c } ^ { ( 1 ) } } ^ { r a w } ( x ^ { ( 1 ) } ) - } \end{array}$ $\mathbf { M } \mathbf { A } \mathbf { X } _ { c , c \neq c ^ { ( 1 ) } } \pmb { \mathcal { E } } _ { c } ^ { r a w } \big ( { x } ^ { ( 1 ) } \big ) + \pmb { \mathcal { E } } _ { c ^ { ( 2 ) } } ^ { r a w } \big ( { x } ^ { ( 2 ) } \big ) - \mathbf { M } \mathbf { A } \mathbf { X } _ { c , c \neq c ^ { ( 2 ) } } \pmb { \mathcal { E } } _ { c } ^ { r a w } \big ( { x } ^ { ( 2 ) } \big )$ .

As $\begin{array} { r } { \mathbf { M } \mathrm { A X } ( a , b ) \ \leq \ \mathbf { M } \mathrm { A X } ( a ) + \mathbf { M } \mathrm { A X } ( b ) , \ \mathbf { M } \mathrm { A X } _ { c , c \neq 0 } ( D _ { c } ( x ^ { ( 1 ) } , x ^ { ( 2 ) } ) ) \ \leq \ \mathbf { M } \mathrm { A X } _ { c , c \neq c ^ { ( 1 ) } } \mathcal { E } _ { c } ^ { r a w } ( x ^ { ( 1 ) } ) \ + \ \underset { ( a , b ) } { \operatorname { M } } \mathrm { A X } _ { c , c \neq c ^ { ( 1 ) } } \mathcal { E } _ { c } ^ { r b } . } \end{array}$ $\mathbf { M A X } _ { c , c \neq c ^ { ( 2 ) } } \mathcal { E } _ { c } ^ { r a w } ( x ^ { ( 2 ) } )$ . Thus negating and adding $D _ { 0 } ( x ^ { ( 1 ) } , x ^ { ( 2 ) } )$ ) to both sides, we arrive at our claim. Hence our claim is true.

As our claim is true for all $x ^ { ( 1 ) }$ and $x ^ { ( 2 ) }$ and $\forall _ { x } f ( x ) \geq g ( x ) \implies \mathbf { M A X } _ { x } ( f ( x ) ) \geq \mathbf { M A X } _ { x } ( g ( x ) )$ , our objective is an upper bound over the ideal objective.

Thus, we can safely say if VERITAS outputs a value less than $2 * \eta$ or it timeouts, while our tool gives a sat output, our tool is better than VERITAS . If our tool gives sat but VERITAS provides a higher output, we deem VERITAS to be better. If our tool gives unsat, then we ignore that instance.

We give VERITAS 1200 seconds to run for each experiment and compare with the best output found till then. For all other tools, we compare time taken for them to find a satisfying pair of examples. The results of the experiments are given in Table 2. The Veritas algorithm finds progressively larger and larger gaps. %V indicates the amount of “gap” found by Veritas during the given time as compared to ENSENSE. For instance, consider the Iris row where we report 2%, which implies that the gap found by ENSENSEis 50 times bigger than the gap found by Veritas.

# G ADDITIONAL EXPERIMENTS AND DETAILS

In this section, we provide more details regarding how we trained the models, how we performed our experiments, and also present additional experimental results. We then explain the counterexample region that is used to evaluate the distances from data. Then we do an ablation study to understand the impact of each improvement both in binary and multiclass tree ensembles. Finally, we show what happens when the sensitive feature set is larger than 1, say 2-4.

<table><tr><td>Dataset</td><td>#Class</td><td>Dep.</td><td>#Trees</td><td>ENSENSE</td><td>KANT</td><td>%V</td></tr><tr><td>covtype_robust</td><td>10</td><td>6</td><td>100</td><td>139.08</td><td>3086.10</td><td>0</td></tr><tr><td>covtype_unrobust</td><td>10</td><td>6</td><td>100</td><td>213.78</td><td>3087.16</td><td>0</td></tr><tr><td>fashion_robust</td><td>10</td><td>6</td><td>100</td><td>118.76</td><td>5667.60</td><td>0</td></tr><tr><td>fashion_unrobust</td><td>10</td><td>6</td><td>100</td><td>67.63</td><td>5001.60</td><td>0</td></tr><tr><td>ori_mnist_robust</td><td>10</td><td>6</td><td>100</td><td>108.76</td><td>3343.97</td><td>0</td></tr><tr><td>ori_mnist_unrobust</td><td>10</td><td>6</td><td>100</td><td>76</td><td>3587.97</td><td>0</td></tr><tr><td>Iris</td><td>3</td><td>1</td><td>100</td><td>0.01</td><td>0.01</td><td>2</td></tr><tr><td>Red-Wine</td><td>3</td><td>6</td><td>100</td><td>3.83</td><td>3.89</td><td>100</td></tr></table>

Table 2: Multi-Class comparison experiments with VERITAS and KANT. The table reports PAR2 runtimes for the experiments in Fig. 3, counting any timeout as 2 × the timeout.

# G.1 TRAINING DETAILS

We trained XGBoost (v1.7.1; binary:logistic) with label encoding for categoricals, rows with missing values removed, and hyperparameters chosen on a 20% validation split (seed = 42) over maximum depth∈ {5,6} and number of boosting rounds ∈ {200,300,500} (benchmarks 7–9) or 500,800 (benchmark 10).

# G.2 ABLATION STUDY

![](images/45f0bffba01df94cb9c1ba74a3255cfd34fbc49e8da2c05c66a21760e24379a5.jpg)

<details>
<summary>line</summary>

| Instances Solved | none | obj | aff+obj | unaff+obj | unaff+aff | unaff+aff+obj |
| ---------------- | ---- | --- | ------- | --------- | --------- | ------------- |
| 0                | 1    | 1   | 1       | 1         | 1         | 1             |
| 200              | 10   | 10  | 10      | 10        | 10        | 10            |
| 400              | 100  | 100 | 100     | 100       | 100       | 100           |
| 600              | 1000 | 1000| 1000    | 1000      | 1000      | 1000          |
| 800              | 10000| 10000| 10000   | 10000     | 10000     | 10000         |
| 1000             | 10000| 10000| 10000   | 10000     | 10000     | 10000         |
| 1200             | 10000| 10000| 10000   | 10000     | 10000     | 10000         |
</details>

(a) Ablation Binary

![](images/62d5627513f954ecf5f85627752eef1ea3992fdc6be2c3fa0bce1a9428e02d87.jpg)

<details>
<summary>line</summary>

| Instances Solved | none   | obj    | aff+obj | unaff+obj | unaff+aff | unaff+aff+obj |
| ---------------- | ------ | ------ | ------- | --------- | --------- | ------------- |
| 0                | 1      | 1      | 1       | 1         | 1         | 1             |
| 100              | 100    | 100    | 100     | 100       | 100       | 100           |
| 200              | 1000   | 1000   | 1000    | 1000      | 1000      | 1000          |
| 300              | 1000   | 1000   | 1000    | 1000      | 1000      | 1000          |
| 400              | 1000   | 1000   | 1000    | 1000      | 1000      | 1000          |
| 500              | 1000   | 1000   | 1000    | 1000      | 1000      | 1000          |
</details>

(b) Ablation Multiclass   
Figure 6: Timing performance of single feature sensitivity for (a) binary ensembles , and (b) multiclass ensembles.

To evaluate the contribution of each component in our sensitivity analysis, we conducted an ablation study by systematically removing key optimizations equation UnAff and equation Aff-bin and evaluating the resulting performance. We present these results in Figure 6, for binary tree ensembles in (left) and multi-class tree ensembles (right). Figure6(left) reports the results for 1102 benchmarks whose runtime is >1s and omitting 188 instances solved under 1s. Figure6(right) reports the results for 517 benchmarks whose runtime is >1s, omitting 21 instances solved under 1s Overall, the added constraints improve solver performance by up to an order of magnitude and dramatically reduce the number of timed-out instances. An interesting observation in both these plots is that when equation Aff-bin is added then equation UnAff do not contribute much in the performance (as can be seen by the overlapping lines). Overall, these results confirm that our enhancements significantly improve the practical feasibility of sensitivity verification in binary and multiclass decision tree ensembles.

# G.3 MULTIFEATURE SENSITIVITY ANALYSIS

To evaluate the ENSENSE’s ability to handle multi-feature sensitivity (i.e., sensitivity wrt change in more than one feature simultaneously $| F | > 1 )$ , we conducted experiments on binary classification models, allowing 1, 2, 3, 4, and 5 features to vary simultaneously. For each benchmark and each m-feature(s) setting, we generate as many test instances as the total number of features. Each instance corresponds to a randomly sampled subset of m-feature(s) from the feature set. Across all benchmarks (binary classifier) in Table1, this results in a total of 430 instances for the m-feature(s). The results, shown in Figure 7, demonstrate that even as the number of varying features increases, our tool remains scalable and even improves in performance. The reason is that the search space explored by the tool decreases as we increase number of sensitive features (since we search in the space of $\mathcal { F } \backslash \dot { F } )$ . These results demonstrate the framework’s scalability and effectiveness in performing multi-feature sensitivity analysis.

![](images/9b5e241b643f805b651145a70bb821b1349020927d8725af8a1752fac4f7a49a.jpg)

<details>
<summary>line</summary>

| Instances Solved | multifeature 1 | multifeature 2 | multifeature 3 | multifeature 4 | multifeature 5 |
| ---------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| 0                | 0.0001         | 0.0001         | 0.0001         | 0.0001         | 0.0001         |
| 50               | 0.1            | 0.2            | 0.3            | 0.2            | 0.1            |
| 100              | 1.0            | 1.5            | 2.0            | 1.5            | 1.0            |
| 150              | 2.0            | 2.5            | 3.0            | 2.5            | 2.0            |
| 200              | 3.0            | 3.5            | 4.0            | 3.5            | 3.0            |
| 250              | 4.0            | 4.5            | 5.0            | 4.5            | 4.0            |
| 300              | 5.0            | 5.5            | 6.0            | 5.5            | 5.0            |
| 350              | 6.0            | 6.5            | 7.0            | 6.5            | 6.0            |
| 400              | 7.0            | 7.5            | 8.0            | 7.5            | 7.0            |
| 420              | 8.0            | 8.5            | 9.0            | 8.5            | 8.0            |
| 430              | 9.0            | 9.5            | 10.0           | 9.5            | 9.0            |
| 440              | 10.0           | 10.5           | 11.0           | 10.5           | 10.0           |
| 450              | 11.0           | 11.5           | 12.0           | 11.5           | 11.0           |
| 460              | 12.0           | 12.5           | 13.0           | 12.5           | 12.0           |
| 470              | 13.0           | 13.5           | 14.0           | 13.5           | 13.0           |
| 480              | 14.0           | 14.5           | 15.0           | 14.5           | 14.0           |
| 490              | 15.0           | 15.5           | 16.0           | 15.5           | 15.0           |
| 500              | 16.0           | 16.5           | 17.0           | 16.5           | 16.0           |
| 510              | 17.0           | 17.5           | 18.0           | 17.5           | 17.0           |
| 520              | 18.0           | 18.5           | 19.0           | 18.5           | 18.0           |
| 530              | 19.0           | 19.5           | 20.0           | 19.5           | 19.0           |
| 540              | 20.0           | 20.5           | 21.0           | 20.5           | 20.0           |
| 550              | 21.0           | 21.5           | 22.0           | 21.5           | 21.0           |
| 560              | 22.0           | 22.5           | 23.0           | 22.5           | 22.0           |
| 570              | 23.0           | 23.5           | 24.0           | 23.5           | 23.0           |
| 580              | 24.0           | 24.5           | 25.0           | 24.5           | 24.0           |
| 590              | 25.0           | 25.5           | 26.0           | 25.5           | 25.0           |
| 600              | 26.0           | 26.5           | 27.0           | 26.5           | 26.0           |
| 610              | 27.0           | 27.5           | 28.0           | 27.5           | 27.0           |
| 620              | 28.0           | 28.5           | 29.0           | 28.5           | 28.0           |
| 630              | 29.0           | 29.5           | 30.0           | 29.5           | 29.0           |
| 640              | 30.0           | 30.5           | 31.0           | 30.5           | 30.0           |
| 650              | 31.0           | 31.5           | 32.0           | 31.5           | 31.0           |
| 660              | 32.0           | 32.5           | 33.0           | 32.5           | 32.0           |
| 670              | 33.0           | 33.5           | 34.0           | 33.5           | 33.0           |
| 680              | 34.0           | 34.5           | 35.0           | 34.5           | 34.0           |
| 690              | 35.0           | 35.5           | 36.0           | 35.5           | 35.0           |
| 700              | 36.0           | 36.5           | 37.0           | 36.5           | 36.0           |
| ...              | ...            | ...            | ...            | ...            | ...            |
| ...              | ...            | ...            | ...            | ...            | ...            |
| ...              | ...            | ...            | ...            | ...            | ...            |
| ...              | ...            | ...            | ...            | ...            | ...            |
| ...              | ...            | ...            | ...            | ...            | ...            |
| ...              | ...            | ...            | ...            | ...            | ...            |
| ...              (repeated)    # value = multifeature_1
        # value = multifeature_1
        # value = multifeature_2
        # value = multifeature_3
        # value = multifeature_4
        # value = multifeature_5
</details>

Figure 7: Timing performance of Multifeature sensitivity for Binary Classification   
![](images/b03fe83cbab4bb13d07af50439ad53317cc63e6a60f463b90d6fc469e883ca26.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["f7 < c1"] -->|F| B["f4 < c2"]
    A -->|T| C["f9 < c3"]
    B -->|F| D["val0"]
    B -->|T| E["val1"]
    C -->|F| F["val2"]
    C -->|T| G["val3"]
```
</details>

![](images/42d7a6c85ca8144417e5dc81daae7827f2b42d1d528b731a8fa6d9f05e650494.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["f7 < c1"] -->|F| B["f9 < c4"]
    A -->|T| C["f9 < c3"]
    B -->|F| D["val4"]
    B -->|T| E["val5"]
    C -->|F| F["val6"]
    C -->|T| G["val7"]
```
</details>

Figure 8: A tree ensemble with two trees $T _ { 1 } , T _ { 2 }$ having real valued (raw) outputs on leaves

# H AN ILLUSTRATIVE EXAMPLE

In Figure 8, we show a tree ensemble with two trees $T _ { 1 }$ and $T _ { 2 } .$ . Each tree has four leaves with real-valued outputs $v a l _ { i } , i = 0 , \ldots , 7 .$ . Let us assume that the sensitive feature set is $\{ f _ { 4 } \}$ . We want to verify if there exists two inputs $x ^ { ( 1 ) }$ and $x ^ { ( 2 ) }$ differing only in feature $f _ { 4 }$ such that the output of the ensemble changes from $0 . 5 - g a p \ : \mathrm { t o } \ : 0 . 5 + g a p$ .

To formulate this as a MILP, we introduce the following variables:

• $p _ { 4 1 } ^ { ( 1 ) } , p _ { 7 1 } ^ { ( 1 ) } , p _ { 9 1 } ^ { ( 1 ) } , p _ { 9 2 } ^ { ( 1 ) }$ p(1)41 , p(1)71 , p(1)91 , p(1)92 : binary variables to represent the decisions at the internal nodes of the trees for input $x ^ { ( 1 ) }$ .

• $p _ { 4 1 } ^ { ( 2 ) } , p _ { 7 1 } ^ { ( 2 ) } , p _ { 9 1 } ^ { ( 2 ) } , p _ { 9 2 } ^ { ( 2 ) }$ p(2)92 : binary variables to represent the decisions at the internal nodes of the trees for input $x ^ { ( 2 ) }$ .   
$l _ { 0 } ^ { ( 1 ) } , l _ { 1 } ^ { ( 1 ) } , l _ { 2 } ^ { ( 1 ) } , l _ { 3 } ^ { ( 1 ) }$ $T _ { 1 }$ $x ^ { ( 1 ) }$   
• $l _ { 4 } ^ { ( 1 ) } , l _ { 5 } ^ { ( 1 ) } , l _ { 6 } ^ { ( 1 ) } , l _ { 7 } ^ { ( 1 ) }$ : binary variables indicating which leaf of tree $T _ { 2 }$ is reached by input $x ^ { ( 1 ) }$   
$l _ { 0 } ^ { ( 2 ) } , l _ { 1 } ^ { ( 2 ) } , l _ { 2 } ^ { ( 2 ) } , l _ { 3 } ^ { ( 2 ) }$ : binary variables indicating which leaf of tree $T _ { 1 }$ is reached by input $x ^ { ( 2 ) }$   
$l _ { 4 } ^ { ( 2 ) } , l _ { 5 } ^ { ( 2 ) } , l _ { 6 } ^ { ( 2 ) } , l _ { 7 } ^ { ( 2 ) }$ : binary variables indicating which leaf of tree $T _ { 2 }$ is reached by input $x ^ { ( 2 ) }$

The objective function in the following is equation Obj-bin for our example.

$$
\begin{array}{l} \max v a l _ {0} l _ {0} ^ {(1)} + v a l _ {1} l _ {1} ^ {(1)} + v a l _ {2} l _ {2} ^ {(1)} + v a l _ {3} l _ {3} ^ {(1)} + v a l _ {4} l _ {4} ^ {(1)} + v a l _ {5} l _ {5} ^ {(1)} \\ + v a l _ {6} l _ {6} ^ {(1)} + v a l _ {7} l _ {7} ^ {(1)} - v a l _ {0} l _ {0} ^ {(2)} - v a l _ {1} l _ {1} ^ {(2)} - v a l _ {2} l _ {2} ^ {(2)} \\ - v a l _ {3} l _ {3} ^ {(2)} - v a l _ {4} l _ {4} ^ {(2)} - v a l _ {5} l _ {5} ^ {(2)} - v a l _ {6} l _ {6} ^ {(2)} - v a l _ {7} l _ {7} ^ {(2)} \tag {14} \\ \end{array}
$$

The above objective is subject to the following constraints. In the guards of $f _ { 9 } ,$ we assume $c _ { 4 } < c _ { 3 }$ . Therefore, the following constraints are due to Equation 2.

$$
\begin{array}{l} p _ {9 1} ^ {(1)} \leq p _ {9 2} ^ {(1)} \\ p _ {9 1} ^ {(2)} \leq p _ {9 2} ^ {(2)} \tag {15} \\ \end{array}
$$

The following constraints are due to Equation 3.

$$
\begin{array}{l} l _ {0} ^ {(1)} + l _ {1} ^ {(1)} + l _ {2} ^ {(1)} + l _ {3} ^ {(1)} = 1 \\ l _ {0} ^ {(2)} + l _ {1} ^ {(2)} + l _ {2} ^ {(2)} + l _ {3} ^ {(2)} = 1 \tag {16} \\ l _ {4} ^ {(1)} + l _ {5} ^ {(1)} + l _ {6} ^ {(1)} + l _ {7} ^ {(1)} = 1 \\ l _ {4} ^ {(2)} + l _ {5} ^ {(2)} + l _ {6} ^ {(2)} + l _ {7} ^ {(2)} = 1 \\ \end{array}
$$

The following constraints are due to Equations 4 and 5.

$$
\begin{array}{l} - p _ {7 1} ^ {(1)} + l _ {0} ^ {(1)} + l _ {1} ^ {(1)} = 0 \quad - p _ {7 1} ^ {(2)} + l _ {0} ^ {(2)} + l _ {1} ^ {(2)} = 0 \\ p _ {7 1} ^ {(1)} + l _ {2} ^ {(1)} + l _ {3} ^ {(1)} = 1 \quad p _ {7 1} ^ {(2)} + l _ {2} ^ {(2)} + l _ {3} ^ {(2)} = 1 \\ - p _ {7 1} ^ {(1)} + l _ {4} ^ {(1)} + l _ {5} ^ {(1)} = 0 \quad - p _ {7 1} ^ {(2)} + l _ {4} ^ {(2)} + l _ {5} ^ {(2)} = 0 \\ p _ {7 1} ^ {(1)} + l _ {6} ^ {(1)} + l _ {7} ^ {(1)} = 1 \quad p _ {7 1} ^ {(2)} + l _ {6} ^ {(2)} + l _ {7} ^ {(2)} = 1 \\ - p _ {4 1} ^ {(1)} + l _ {0} ^ {(1)} \leq 0 \quad - p _ {4 1} ^ {(2)} + l _ {0} ^ {(2)} \leq 0 \\ p _ {4 1} ^ {(1)} + l _ {1} ^ {(1)} \leq 1 \quad p _ {4 1} ^ {(2)} + l _ {1} ^ {(2)} \leq 1 \tag {17} \\ - p _ {9 2} ^ {(1)} + l _ {2} ^ {(1)} \leq 0 \quad - p _ {9 2} ^ {(2)} + l _ {2} ^ {(2)} \leq 0 \\ p _ {9 2} ^ {(1)} + l _ {3} ^ {(1)} \leq 1 \quad p _ {9 2} ^ {(2)} + l _ {3} ^ {(2)} \leq 1 \\ - p _ {9 2} ^ {(1)} + l _ {6} ^ {(1)} \leq 0 \quad - p _ {9 2} ^ {(2)} + l _ {6} ^ {(2)} \leq 0 \\ p _ {9 2} ^ {(1)} + l _ {7} ^ {(1)} \leq 1 \quad p _ {9 2} ^ {(2)} + l _ {7} ^ {(2)} \leq 1 \\ - p _ {9 1} ^ {(1)} + l _ {4} ^ {(1)} \leq 0 \quad - p _ {9 1} ^ {(2)} + l _ {4} ^ {(2)} \leq 0 \\ p _ {9 1} ^ {(1)} + l _ {5} ^ {(1)} \leq 1 \quad p _ {9 1} ^ {(2)} + l _ {5} ^ {(2)} \leq 1 \\ \end{array}
$$

The following constraints are due to Equation 6.

$$
\begin{array}{l} - p _ {9 1} ^ {(1)} + p _ {9 1} ^ {(2)} = 0 \\ - p _ {9 2} ^ {(1)} + p _ {9 2} ^ {(2)} = 0 \tag {18} \\ - p _ {7 1} ^ {(1)} + p _ {7 1} ^ {(2)} = 0 \\ \end{array}
$$

The following constraints are due to equation Gap-bin:

$$
v a l _ {0} l _ {0} ^ {(1)} + v a l _ {1} l _ {1} ^ {(1)} + v a l _ {2} l _ {2} ^ {(1)} + v a l _ {3} l _ {3} ^ {(1)} + v a l _ {4} l _ {4} ^ {(1)} + v a l _ {5} l _ {5} ^ {(1)} + v a l _ {6} l _ {6} ^ {(1)} + v a l _ {7} l _ {7} ^ {(1)} \geq g a p - 0. 5
$$

$$
v a l _ {0} l _ {0} ^ {(2)} + v a l _ {1} l _ {1} ^ {(2)} + v a l _ {2} l _ {2} ^ {(2)} + v a l _ {3} l _ {3} ^ {(2)} + v a l _ {4} l _ {4} ^ {(2)} + v a l _ {5} l _ {5} ^ {(2)} + v a l _ {6} l _ {6} ^ {(2)} + v a l _ {6} l _ {7} ^ {(2)} \leq - 0. 5 - g a p \tag {19}
$$

The following constraints are due to Equations UnAff and Aff-bin respectively, since only feature $f _ { 4 }$ is sensitive.

$$
l _ {2} ^ {(1)} = l _ {2} ^ {(2)}
$$

$$
l _ {3} ^ {(1)} = l _ {3} ^ {(2)}
$$

$$
l _ {4} ^ {(1)} = l _ {4} ^ {(2)} \tag {20}
$$

$$
l _ {5} ^ {(1)} = l _ {5} ^ {(2)}
$$

$$
l _ {6} ^ {(1)} = l _ {6} ^ {(2)}
$$

$$
l _ {7} ^ {(1)} = l _ {7} ^ {(2)}
$$

$$
v a l _ {0} l _ {0} ^ {(1)} + v a l _ {1} l _ {1} ^ {(1)} - v a l _ {0} l _ {0} ^ {(2)} - v a l _ {1} l _ {1} ^ {(2)} \geq 2 * g a p \tag {21}
$$

The above constraints indicates the key reason of the efficiency of our method. The long sum of Equation 19 reduces to much shorter sum . Thereby, MILP solver has easier time solving the problem.