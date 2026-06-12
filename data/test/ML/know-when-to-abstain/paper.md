# KNOW WHEN TO ABSTAIN: OPTIMAL SELECTIVE CLASSIFICATION WITH LIKELIHOOD RATIOS

Alvin Heng1 & Harold Soh1,2

1Department of Computer Science, National University of Singapore   
2Smart Systems Institute, National University of Singapore

{alvinh, harold}@comp.nus.edu.sg

# ABSTRACT

Selective classification enhances the reliability of predictive models by allowing them to abstain from making uncertain predictions. In this work, we revisit the design of optimal selection functions through the lens of the Neyman–Pearson lemma, a classical result in statistics that characterizes the optimal rejection rule as a likelihood ratio test. We show that this perspective not only unifies the behavior of several post-hoc selection baselines, but also motivates new approaches to selective classification which we propose here. A central focus of our work is the setting of covariate shift, where the input distribution at test time differs from that at training. This realistic and challenging scenario remains relatively underexplored in the context of selective classification. We evaluate our proposed methods across a range of vision and language tasks, including both supervised learning and vision-language models. Our experiments demonstrate that our Neyman– Pearson-informed methods consistently outperform existing baselines, indicating that likelihood ratio-based selection offers a robust mechanism for improving selective classification under covariate shifts. Our code is publicly available at https://github.com/clear-nus/sc-likelihood-ratios.

# 1 INTRODUCTION

Machine learning models are inherently fallible and can make erroneous predictions. Unlike humans, who can abstain from answering when uncertain, e.g., by saying “I don’t know”, predictive models typically produce a prediction for every input regardless of confidence. Selective classification aims to address this limitation by enabling models to abstain on uncertain inputs, thereby improving overall performance and robustness, for instance by deferring ambiguous cases to human experts.

A wide range of methods have been proposed to determine whether a model should accept or reject an input. Common post-hoc approaches rely on heuristic confidence estimates, such as the maximum softmax probability (Geifman & El-Yaniv, 2017; Hendrycks & Gimpel, 2017), logit margins (Liang et al., 2024), or Monte Carlo dropout (Geifman & El-Yaniv, 2017; Gal & Ghahramani, 2016). Other techniques assess a sample’s proximity to the training distribution (Lee et al., 2018; Sun et al., 2022), under the assumption that samples farther from the data manifold are more likely to be misclassified. A separate line of work trains models with explicit abstention mechanisms, such as a rejection logit or head (Geifman & El-Yaniv, 2019; Liu et al., 2019; Huang et al., 2020). In this work we focus on post-hoc methods, which are model-agnostic and do not require specialized training.

Despite the rich literature, two important gaps remain. First, while foundational results (such as Chow (1970); Geifman & El-Yaniv (2017)) provide theoretical underpinnings for selective classification, there is a lack of general, principled guidance for designing effective selector functions in the context of modern deep networks. Second, most evaluations are conducted in the i.i.d. setting, where test data is assumed to follow the training distribution. Few works have begun exploring selective classification under distribution shifts (Xia & Bouganis, 2022; Narasimhan et al., 2024), but focus on the common semantic shifts, neglecting the covariate shift setting that is becoming increasingly relevant.

To address these challenges, we propose a new perspective rooted in the Neyman–Pearson lemma, a classical result from statistics that defines the optimal hypothesis test in terms of a likelihood ratio.

We show that existing selectors can be interpreted as approximations to this test, and we use this insight to derive two new selectors, ∆-MDS and ∆-KNN, as well as a simple linear combination strategy. We evaluate our methods on a comprehensive suite of vision and language benchmarks under covariate shift, where the input distribution changes while the label space remains fixed. We focus on covariate shift for two key reasons: first, it is underexplored relative to semantic shifts (Heng & Soh, 2025) (which is well studied in the context of out-of-distribution detection (Hendrycks & Gimpel, 2017; Ming et al., 2022; Lee et al., 2018; Heng et al., 2024)); second, it is increasingly relevant in modern applications such as vision-language models (VLMs), where the label set is large and variable, rendering most practical shifts in deployment covariate in nature. Our results demonstrate that the proposed selectors outperform existing baselines and provide robust performance across distribution shifts, including on powerful VLMs like CLIP.

In summary, our key contributions are:

1. We introduce for the first time a Neyman–Pearson-based framework for defining optimality in selective classification via likelihood ratio tests.   
2. We unify several existing selector methods and propose two new selectors and a linear combination approach under this framework.   
3. We conduct a thorough evaluation under distribution shifts, both covariate and semantic, across vision and language tasks and demonstrate superior performance across both VLMs and traditional supervised models.

# 2 BACKGROUND

Selective Classification. Consider a standard classification problem with input space $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ , label space $\mathcal { V } = \{ { 1 , . . . , K } \}$ , and data distribution $\mathcal { D } _ { \mathcal { X } , \mathcal { Y } }$ over $\mathcal { X } \times \mathcal { V }$ . A selective classifier is a pair $( f , g )$ , where $f : \mathcal { X } \to \mathbb { R } ^ { K }$ is a base classifier, and $g : \mathcal { X }  \{ 0 , 1 \}$ is a selector function that determines whether to make a prediction or abstain. Formally,

$$
(f, g) (\boldsymbol {x}) \triangleq \left\{ \begin{array}{l l} f (\boldsymbol {x}) & \text { if   } g (\boldsymbol {x}) = 1, \\ \text { abstain } & \text { if   } g (\boldsymbol {x}) = 0. \end{array} \right. \tag {1}
$$

That is, the model abstains on input x when $g ( { \pmb x } ) = 0$ . In practice, g is typically implemented by thresholding a real-valued confidence score:

$$
g _ {s, \gamma} (\boldsymbol {x}) = \mathbb {1} [ s (\boldsymbol {x}) > \gamma ], \tag {2}
$$

where $s : \mathcal { X } \to \mathbb { R }$ is a confidence scoring function (often adapted from OOD detection methods; see below), and γ is a tunable threshold. The performance of a selective classifier is typically evaluated using two metrics:

$$
\text { Coverage: } \quad \phi_ {s, \gamma} = \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D} _ {\mathcal {X}, \mathcal {Y}}} [ g _ {s, \gamma} (\boldsymbol {x}) ], \tag {3}
$$

$$
\text { Selective   Risk: } \quad R _ {s, \gamma} = \frac {\mathbb {E} _ {(\boldsymbol {x} , y) \sim \mathcal {D} _ {\mathcal {X} , \mathcal {Y}}} [ \ell (f (\boldsymbol {x}) , y) \cdot g _ {s , \gamma} (\boldsymbol {x}) ]}{\phi_ {s , \gamma}}, \tag {4}
$$

where $\ell ( f ( { \pmb x } ) , { \pmb y } )$ is the $0 / 1$ loss (Geifman & El-Yaniv, 2017). Selective classification aims to optimize the tradeoff between selective risk and coverage by ideally reducing risk while maintaining high coverage. Improvements can stem from enhancing the base classifier $f ,$ or from refining the selector g to better identify error-prone inputs. In this work, we fix f to be a strong pretrained model and focus on designing more effective selector functions g.

Covariate Shift. Covariate shift refers to a scenario where the marginal distribution over inputs, $p ( { \pmb x } )$ , changes between training and testing, while the support of the label distribution $p ( y )$ remains unchanged. For example, a model trained on photographs of cats may face a covariate shift when evaluated on paintings of cats, as the input appearance changes but the semantic categories are preserved. This is in contrast to semantic shift, where both $p ( { \pmb x } )$ and $p ( y )$ change, typically due to the introduction of unseen classes. In this work, we focus on covariate shifts, which are increasingly relevant in modern applications such as VLMs, where the label set is large and can be adjusted to suit a given task. In such settings, distributional changes primarily manifest as covariate shifts, making them a critical yet underexplored challenge for robust selective classification.

Out-of-Distribution (OOD) Detection. OOD detection is closely related to selective classification, as many selector functions $s ( { \pmb x } )$ are derived from or inspired by OOD scoring methods. Given an in-distribution (ID) data distribution $p _ { I D }$ , the goal of OOD detection is to construct a scoring function $s : \mathcal { X } \to \mathbb { R }$ such that $s ( { \pmb x } )$ indicates the likelihood that x originates from $p _ { I D }$ . A higher s(x) corresponds to higher confidence that x is in-distribution. In selective classification, these scores are thresholded to determine whether to accept or abstain on a given input, as formalized in Eq. 2.

# 3 SELECTIVE CLASSIFICATION VIA THE NEYMAN–PEARSON LEMMA

We begin by framing selective classification within the paradigm of hypothesis testing. Let $\mathcal { H } _ { 0 } : \mathcal { C }$ denote the hypothesis that the classifier makes a correct prediction, and $\mathcal { H } _ { 1 } : \neg \mathcal { C }$ that it makes an incorrect one. Selective classification then reduces to deciding, for each input, whether to accept $\mathcal { H } _ { \mathrm { 0 } }$ or reject in favor of $\mathcal { H } _ { 1 } , \mathrm { i . e . }$ , a binary decision problem. This perspective is a natural fit for a foundational result from statistics: the Neyman–Pearson (NP) lemma (Neyman & Pearson, 1933; Lehmann et al., 1986), which characterizes the optimal decision rule between two competing hypotheses.

Lemma 1 (Neyman–Pearson (Neyman & Pearson, 1933; Lehmann et al., 1986)). Let $Z \in \mathbb { R } ^ { d }$ be a random variable, and consider the hypotheses::

$$
\mathcal {H} _ {0}: Z \sim P _ {0} \quad v s. \quad \mathcal {H} _ {1}: Z \sim P _ {1},
$$

where $P _ { 0 }$ and $P _ { 1 }$ have densities $p _ { 0 }$ and $p _ { 1 }$ that are strictly positive on a shared support $\mathcal { Z } \subset \mathbb { R } ^ { d }$ . For any measurable acceptance region $A \subset { \mathcal { Z } }$ under ${ \mathcal { H } } _ { 0 } ,$ , define the type I error (false rejection) as $\alpha ( A ) \stackrel { \cdot } { = } P _ { 0 } ( Z \notin A )$ , and type II error (false acceptance) as ${ \bar { \beta } } ( A ) = { \bar { P } } _ { 1 } ( Z \in A )$ .

Fix a type I error tolerance $\alpha _ { 0 } \in [ 0 , 1 ]$ . Let $\gamma ( \alpha _ { 0 } )$ be the threshold such that

$$
A ^ {*} (\alpha_ {0}) := \left\{z \in \mathcal {Z}: \frac {p _ {0} (z)}{p _ {1} (z)} \geq \gamma (\alpha_ {0}) \right\}
$$

satisfies $\alpha ( A ^ { * } ) = \alpha _ { 0 }$ . Then $A ^ { * } ( \alpha _ { 0 } )$ minimizes the type II error:

$$
\beta (A ^ {*} (\alpha_ {0})) = \min _ {A: \alpha (A) = \alpha_ {0}} \beta (A).
$$

In other words, among all decision rules with the same false rejection rate, the likelihood ratio test minimizes the false acceptance rate.

Applied to selective classification, Lemma 1 suggests the optimal selection score is a likelihood ratio:

$$
s (\pmb {x}) = \frac {p _ {c} (\pmb {x})}{p _ {w} (\pmb {x})},
$$

where $p _ { c } ( { \pmb x } )$ and $p _ { w } ( { \pmb x } )$ denote the probability density associated with the classifier making a correct and wrong prediction respectively. Thresholding this score yields the lowest possible selective risk for any given coverage level.

Corollary 1 (Informal). Any selector score $s ( { \pmb x } )$ that is a monotonic transformation of the likelihood $r a t i o \ { \frac { p _ { 0 } ( { \pmb x } ) } { p _ { 1 } ( { \pmb x } ) } }$ is also optimal under the Neyman–Pearson criterion.

Corollary 1 follows directly from the lemma, since monotonic transformations (e.g., logarithmic or affine maps) preserve the ordering of scores and hence do not alter the resulting acceptance region. We define a score function $s ( { \pmb x } )$ to be Neyman–Pearson optimal if it is a monotonic transformation of the likelihood ratio $p _ { c } ( { \pmb x } ) / p _ { w } ( { \pmb x } )$ . In practice, the true likelihood ratio is not accessible, so we approximate it or construct a monotonic proxy that captures the posterior odds of a correct versus incorrect prediction for a given input.

Note that $p _ { c } ( { \pmb x } )$ and $p _ { w } ( { \pmb x } )$ are general and naturally accounts for distribution shifts; $p _ { c } ( { \pmb x } ) \left( p _ { w } ( { \pmb x } ) \right)$ includes all samples that the classifier classifies correctly (wrongly), regardless of whether they are ID or a distribution shift. Therefore, this simplifies our framework compared to prior works that consider ID and OOD distributions separately (Xia & Bouganis, 2022; Narasimhan et al., 2024).

In what follows, we first reinterpret existing selection scores from the literature as implicit approximations to this likelihood ratio and show conditions under which they are NP optimal. We then introduce two new distance-based selection functions inspired by the NP framework. Finally, we propose a hybrid score that linearly combines multiple selectors which we find performs well in practice. Throughout, the assumptions used in our theoretical results are meant to clarify the connection to NP optimality and the structure of an optimal selector, rather than to prescribe conditions that must hold in practice. Introduction and brief details of the scores discussed in this section are provided in Appendix A.

# 3.1 LOGIT-BASED SCORES AS APPROXIMATIONS TO LIKELIHOOD RATIOS

We consider two popular confidence scores, Maximum Softmax Probability (MSP) (Hendrycks & Gimpel, 2017) and Raw Logits (RLog) (Liang et al., 2024), and interpret them as approximations to likelihood ratio tests in the NP sense. This provides theoretical justification for their empirical success as selector scores in selective classification.

Let $l ^ { ( 1 ) } \ge l ^ { ( 2 ) } \ge \cdots \ge l ^ { ( K ) }$ denote the logits output by a classifier for a sample x, sorted in descending order. Define the corresponding softmax probabilities $d ^ { ( i ) } = \mathrm { s o f t m a x } \bar { ( } l ^ { ( i ) } )$ . The MSP score is given by $s _ { \mathrm { M S P } } ( \pmb { x } ) = d ^ { ( 1 ) }$ , while the RLog score is defined as $s _ { \mathrm { R L o g } } ( { \pmb x } ) = l ^ { ( 1 ) } - l ^ { ( 2 ) }$ . MSP has become a standard baseline for OOD detection and selective classification (Hendrycks & Gimpel, 2017; Geifman & El-Yaniv, 2017), and RLog has been recently proposed as a strong score for selective classification (Liang et al., 2024).

Theorem 1. Let $\hat { y } ( x ) = \arg \operatorname* { m a x } _ { k \in \{ 1 , . . . , K \} } p _ { \theta } ( y = k \mid x )$ be the predicted label and define the event $C : = \{ \hat { y } ( X ) = Y \}$ that the classifier is correct. Assume the classifier is calibrated for top-1 correctness, i.e., $\begin{array} { r } { P ( C \mid X = \pmb { x } ) = \operatorname* { m a x } _ { k } p _ { \theta } ( y = k \mid \pmb { x } ) = : d ^ { ( 1 ) } ( \pmb { x } ) } \end{array}$ . Then MSP is Neyman–Pearson optimal for selective classification. Moreover, under the additional assumption that the softmax distribution is concentrated on the top two classes, i.e., $\begin{array} { r } { L : = \sum _ { i \geq 3 } d ^ { ( i ) } \ll \dot { d } ^ { ( 2 ) } } \end{array}$ , the RLog score is also Neyman–Pearson optimal.

The proof provided in Appendix B shows that under these assumptions, both MSP and RLog are monotonic transformations of the likelihood ratio ${ p _ { c } } / { p _ { w } }$ , and therefore is NP optimal by Corollary 1. Of course, these assumptions are not always satisfied in practice. Prior work (Guo et al., 2017) has shown that modern neural classifiers tend to be poorly calibrated, and has proposed post-hoc calibration methods such as temperature scaling. Notably, RLog has been shown to be invariant to temperature scaling (Liang et al., 2024), making it robust to miscalibration and a compelling choice in practice. This aligns with our empirical findings in Sec. 5, where RLog generally outperforms MSP (which corresponds to temperature scaling with T = 1). While the effect of calibration is an important factor in logit-based methods (Cattelan & Silva, 2023; Fisch et al., 2022), it lies beyond the scope of this work.

# 3.2 NEYMAN–PEARSON OPTIMAL DISTANCE SCORES

The logit-based scores discussed in the previous section rely on classifier logit calibration, a condition often violated in practice (Guo et al., 2017). To avoid this dependency, we consider distance-based methods that make alternative assumptions independent of calibration. As we show below, these methods approximate the likelihood ratio ${ p _ { c } } / { p _ { w } }$ by leveraging spatial relationships in feature space.

Two distance methods widely used in OOD detection are the Mahalanobis distance (MDS) (Lee et al., 2018) and k-Nearest Neighbors (KNN) (Sun et al., 2022). Both rely on computing distances between a test sample and training features (see Appendix A for details). Briefly, MDS is defined a $\begin{array} { r } { \mathrm { ; ~ } s _ { \mathrm { M D S } } ( \pmb { x } ) = \operatorname* { m a x } _ { i } - ( \phi ( \pmb { x } ) - \mu _ { i } ) ^ { \top } \Sigma ^ { - 1 } ( \phi ( \pmb { x } ) - \mu _ { i } ) } \end{array}$ , where ϕ(x) denotes the extracted feature of x, typically from the penultimate or final layer of a trained deep network, $\mu _ { i }$ is the empirical mean feature of class i, and Σ is a shared covariance matrix. In contrast, KNN scores inputs by the negative distance to the k-th nearest training feature vector. We introduce ∆-MDS and $\Delta { \mathrm { - } } \mathrm { K N N }$ , which are modified versions of these scores that explicitly incorporate insights from the NP lemma by estimating separate distributions for correctly and incorrectly classified training samples. Figure 1 gives an overview of our approach, and we provide pseudocode for our proposed methods in Appendix D.

∆-MDS. Instead of estimating a single distribution per class, we maintain two sets of statistics per class: $\{ \mu _ { i } ^ { c } , \Sigma ^ { c } \} _ { i = 1 } ^ { K }$ and $\{ \mu _ { i } ^ { w } , \Sigma ^ { w } \} _ { i = 1 } ^ { K } ,$ , corresponding to the mean and shared covariance of features for training samples that the classifier predicts correctly and wrongly, respectively. These quantities are easily estimated as the true labels are known. We then define the ∆-MDS score as the difference in Mahalanobis distances between the two distributions:

![](images/dfdcbd71ba99f2d1a52fa1547776769fb9efda617bef19af6b289aa4a7b92a4d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph D_c(x1)
        x1["x1"]
        Dw["x1"]
    end
    subgraph_Dw["x2"]
        x2["x2"]
    end
    p_c["x"] --> D_c["x1"]
    p_w["x"] --> Dw["x2"]
    style D_c fill:#f9f,stroke:#333
    style Dw fill:#bbf,stroke:#333
    note_bottom_of_D_c["x1"] -.-> Dw["x1"]
    note_top_of_D_w["x2"] -.-> Dw["x2"]
    note_bottom_of_D_w["x1"] -.-> Dw["x2"]
    note_top_of_D_w["x1"] -.-> Dw["x2"]
    note_bottom_of_D_w["x2"] -.-> Dw["x2"]
    note_bottom_of_D_w["x1"] -.-> Dw["x2"]
    note_top_of_D_w["x1"] -.-> Dw["x2"]
    note_bottom_of_D_w["x2"] -.-> Dw["x2"]
    note_bottom_of_D_w["x1"] -.-> Dw["x2"]
    note_top_of_D_w["x1"] -.-> Dw["x2"]
    note_bottom_of_D_w["x2"] -.-> Dw["-"]
```
</details>

Figure 1: Illustration of our proposed Neyman–Pearson optimal distance-based selective classification methods. We estimate the likelihoods of correct and incorrect predictions $( p _ { c }$ and $p _ { w } )$ as a function of distances to training sets consisting of correctly and incorrectly classified samples: $s ( { \pmb x } ) \ =$ $f ( D _ { c } ( { \pmb x } ) , D _ { w } ( { \pmb x } ) )$ , where $f$ here denotes a function. For example, x1 is “closer” to $p _ { c }$ and “farther” from $p _ { w }$ than $\scriptstyle { \mathbf { { x } } } _ { 2 } .$ , and should therefore receive a higher score.

$$
s _ {\Delta - \mathrm{MDS}} (\boldsymbol {x}) = D _ {\mathrm{MDS}} \left(\boldsymbol {x}; \mu_ {i} ^ {c}, \Sigma^ {c}\right) - D _ {\mathrm{MDS}} \left(\boldsymbol {x}; \mu_ {i} ^ {w}, \Sigma^ {w}\right) \tag {5}
$$

where $D _ { \mathrm { M D S } } ( \pmb { x } ; \mu _ { i } , \Sigma ) = \mathrm { m a x } _ { i } - ( \phi ( \pmb { x } ) - \mu _ { i } ) ^ { \top } \Sigma ^ { - 1 } ( \phi ( \pmb { x } ) - \mu _ { i } )$ is the standard Mahalanobis score. The score intuitively increases when the input is closer (in Mahalanobis sense) to the “correctly classified” region and farther from the “wrongly classified” region in feature space. We now formalize this intuition:

Theorem 2. Let $Z = \phi ( \pmb { x } ) \in \mathbb { R } ^ { d }$ be the feature representation of input x. Let C be the event the classifier makes a correct prediction and ¬C its negation. Assume $\mathsf { \bar { Z } } | \mathcal { C } \sim p _ { c } = \mathcal { N } ( \mu _ { i } ^ { c } , \Sigma ^ { c } )$ and $Z | \lnot { \mathcal { C } } \sim p _ { w } = { \mathcal { N } } ( \mu _ { i } ^ { w } , \Sigma ^ { w } )$ . Then the ∆-MDS score $s _ { \Delta - M D S } ( { \pmb x } )$ is Neyman–Pearson optimal for selective classification.

The proof is provided in Appendix B, which shows that $s _ { \Delta }$ -MDS is a monotonic transformation of the likelihood ratio ${ p _ { c } } / { p _ { w } }$ , and thus is NP optimal as per Corollary 1. The Gaussian assumption on feature representations is supported both empirically and theoretically via connections between Gaussian Discriminant Analysis and softmax classifiers (Lee et al., 2018), making ∆-MDS well-suited for modern deep classifiers trained on standard supervised learning objectives.

∆-KNN. Next we introduce ∆-KNN, a non-parametric distance-based score inspired by the NP framework. Let $A _ { c } = \{ \phi _ { c } ( \pmb { x } _ { 1 } ) , \dots , \phi _ { c } ( \pmb { x } _ { N _ { c } } ) \}$ and $A _ { w } = \{ \phi _ { w } ( { \pmb x } _ { 1 } ) , \dots , \phi _ { w } ( { \pmb x } _ { N _ { w } } ) \}$ denote the feature representations of training samples that the classifier predicted correctly and wrongly respectively, and $N _ { c } = | A _ { c } |$ | and $N _ { w } = | A _ { w } | . \operatorname { L e t } z = \phi ( { \pmb x } )$ be the feature vector of a test input x. Define $u _ { k } ( z )$ and $v _ { k } ( z )$ as the Euclidean distances from z to its k-th nearest neighbors in $A _ { c }$ and $A _ { w }$ . We define the ∆-KNN score as the difference in log-distances:

$$
s _ {\Delta - \mathrm{KNN}} (\boldsymbol {x}) = D _ {\mathrm{KNN}} (\boldsymbol {x}; A _ {c}) - D _ {\mathrm{KNN}} (\boldsymbol {x}; A _ {w}) \tag {6}
$$

where $D _ { \mathrm { K N N } } ( \pmb { x } ; A _ { c } ) = - \log [ u _ { k } ( \phi ( \pmb { x } ) ) ]$ and $D _ { \mathrm { K N N } } ( \pmb { x } ; A _ { w } ) = - \log [ v _ { k } ( \phi ( \pmb { x } ) ) ]$ ]. This score measures how much closer a test point is to the region of correctly classified samples compared to incorrectly classified ones. We now show that ∆-KNN is asymptotically NP optimal:

Theorem 3. Let $Z = \phi ( \pmb { x } ) \in \mathbb { R } ^ { d }$ be the feature representation of input x, and let C denote the event that the classifier makes a correct prediction. Suppose $Z \mid \mathcal { C } \sim p _ { c }$ and $Z | \neg { \mathcal { C } } \sim p _ { w }$ are arbitrary continuous densities bounded away from zero. Let $N _ { c } = \dot { | } A _ { c } |$ and $N _ { w } = | A _ { w } |$ . If k → ∞ while $k / N _ { c }  0$ and $k / N _ { w }  0$ as $N _ { c } , N _ { w }  \infty ,$ , then ${ \pmb S } \Delta { \bf - } K N N ^ { \bf ( } { \pmb x } )$ is a Neyman–Pearson optimal selector.

The proof is provided in Appendix B. As in previous cases, it relies on showing that $s _ { \Delta }$ -KNN is a monotonic transformation of the likelihood ratio ${ p _ { c } } / { p _ { w } }$ . Importantly, this result does not require parametric assumptions on the form of $p _ { c }$ or $p _ { w }$ , unlike ∆-MDS. However, it does depend on asymptotic properties of the k-nearest neighbor density estimator, and the required conditions on k, $N _ { c } ,$ , and $N _ { w }$ may be difficult to satisfy in finite-sample settings. As such, both methods have their tradeoffs in terms of modeling assumptions.

In practice, we replace the single k-th neighbor distance with the average log-distance to the top k neighbors. Specifically, we use: $\begin{array} { r } { D _ { \mathrm { K N N } } ( \pmb { x } ; A _ { c } ) = - \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \log [ u _ { i } ( \phi ( \pmb { x } ) ) ] } \end{array}$ and $D _ { \mathrm { K N N } } ( { \pmb x } ; A _ { w } ) =$ $\begin{array} { r } { - \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \log [ v _ { i } ( \phi ( \pmb { x } ) ) ] } \end{array}$ 1k . We find that this smoother version improves empirical performance, as shown in our ablation studies in Sec. 5.3. While this modification deviates from the form in Theorem 3, we include a discussion in Appendix C suggesting NP optimality holds for the averaged log-distance formulation under standard assumptions.

# 3.3 LINEAR COMBINATIONS OF DISTANCE AND LOGIT-BASED SCORES

The selector scores we discussed rely on different modeling assumptions and exhibit complementary strengths, as discussed in Wang et al. (2022). Logit-based scores utilize the classifier’s learned boundaries, while distance-based methods depend on geometric structures in feature space defined by training samples. We are thus motivated to leverage their respective advantages by proposing a simple yet effective solution: linearly combining selector scores. Intuitively, this allows each score to compensate for the limitations of the other. The following lemma formalizes the NP optimality of such a linear combination:

Lemma 2. Let $s _ { 1 } ( { \pmb x } ) \in \mathbb { R }$ and $s _ { 2 } ( { \pmb x } ) \in \mathbb { R }$ be two selector scores. Assume both are Neyman–Pearson optimal; that $p _ { c } ^ { ( 2 ) } / p _ { w } ^ { ( 2 ) }$ . Then for any scalar (1) (2) $i s , s _ { 1 } ( { \pmb x } )$ is a monotone transform of $\lambda \in \mathbb { R } , t ( { \pmb x } ) = s _ { 1 } ( { \pmb x } ) + \lambda s _ { 2 } ( { \pmb x } )$ $p _ { c } ^ { ( 1 ) } / p _ { w } ^ { ( 1 ) }$ and is a monotonic transformation of $s _ { 2 } ( \pmb { x } )$ is a monotone transform of $p _ { c } ^ { ( 1 ) } ( p _ { c } ^ { ( 2 ) } ) ^ { \lambda } / p _ { w } ^ { ( 1 ) } ( p _ { w } ^ { ( 2 ) } ) ^ { \lambda }$ .

The proof provided in Appendix B follows by expressing t(x) as a log-product of likelihood ratios. Thus, t(x) remains NP optimal under the assumption that the density for each hypothesis takes the form of a multiplicative (or “tilted”) product: $p _ { c } ^ { ( 1 ) } ( p _ { c } ^ { ( 2 ) } ) ^ { \lambda } / Z _ { c }$ and similarly for $p _ { w }$ , where $Z _ { c }$ is a normalization constant. In practice, we find that combining a distance-based score $( \mathrm { e . g . , \Delta . M D S ) }$ with a logit-based score (e.g., RLog) leads to the best performance. We refer to such combinations by concatenating their names, e.g., ∆-MDS-RLog. We discuss fitting parameters like λ in Sec. 5.

# 4 RELATED WORKS

The study of classification with a reject option has a long history, beginning with cost-based formulations (Chow, 1970) and extensions to classical models like SVMs (Fumera & Roli, 2002; Bartlett & Wegkamp, 2008) and nearest neighbors (Hellman, 1970). In deep learning, LeCun et al. (1989) explored rejection via top logit activations. Later, the risk–coverage and classifier–selector frameworks were formalized (El-Yaniv et al., 2010; Geifman & El-Yaniv, 2017), with methods like MSP and Monte Carlo dropout proposed to provide confidence-based selection (Gal & Ghahramani, 2016).

Subsequent works have extended this direction by studying popular logit and distance-based scores, many originally developed for OOD detection. Examples include MSP (Hendrycks & Gimpel, 2017), MaxLogit (Hendrycks et al., 2019), Energy (Liu et al., 2020), MDS (Lee et al., 2018), and KNN (Sun et al., 2022). A limitation of logit-based methods is their reliance on classifier calibration. Although calibration is not the focus of our work, several studies have examined its impact on selective classification performance (Cattelan & Silva, 2023; Galil et al., 2023). An alternative line of research uses conformal prediction to construct calibrated prediction sets with formal guarantees (Vovk et al., 2005; Angelopoulos & Bates, 2021; Bates et al., 2021; Angelopoulos et al., 2024). While such methods could be adapted for selective classification, they differ fundamentally from our goal of designing scoring functions optimized for selective risk. Jiang et al. (2018) proposes a trust score by comparing the model’s prediction with class-conditional KNN distances to estimate if a sample will be correctly classified, but it was shown to be ineffective on high-dimensional images.

Some methods incorporate rejection directly into training. For instance, SelectiveNet (Geifman & El-Yaniv, 2019) adds a dedicated rejection head, while Deep Gamblers (Liu et al., 2019) and Self-Adaptive Training (Huang et al., 2020) introduce a reject class and train the model to abstain. These methods require architectural modifications and joint training. In contrast, our work focuses on post-hoc methods that can be applied to pretrained classifiers without retraining.

Table 1: DFN CLIP AURC (A) and NAURC (N) results on ImageNet and its covariate shifted variants at full coverage. Lower is better. AURC results are on the $1 0 ^ { = 2 }$ scale. Bold and underline denotes the best and second best result respectively. “Avg (1K)" denotes average results over datasets with full 1K-class coverage, while “Avg (all)" is the average result over all datasets. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Im-1K</td><td colspan="2">Im-R</td><td colspan="2">Im-A</td><td colspan="2">ON</td><td colspan="2">Im-V2</td><td colspan="2">Im-S</td><td colspan="2">Im-C</td><td colspan="2">Avg (1K)</td><td colspan="2">Avg (all)</td></tr><tr><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td></tr><tr><td>MSP</td><td>9.08</td><td>0.542</td><td>2.00</td><td>0.344</td><td>2.29</td><td>0.179</td><td>8.87</td><td>0.268</td><td>9.39</td><td>0.521</td><td>12.3</td><td>0.524</td><td>15.1</td><td>0.328</td><td>11.5</td><td>0.479</td><td>8.43</td><td>0.387</td></tr><tr><td>MaxLogit</td><td>9.08</td><td>0.542</td><td>2.21</td><td>0.385</td><td>2.96</td><td>0.249</td><td>12.4</td><td>0.437</td><td>9.35</td><td>0.518</td><td>12.3</td><td>0.525</td><td>17.7</td><td>0.423</td><td>13.9</td><td>0.502</td><td>11.3</td><td>0.440</td></tr><tr><td>Energy</td><td>14.2</td><td>0.901</td><td>5.81</td><td>1.06</td><td>12.2</td><td>1.2</td><td>33.3</td><td>1.43</td><td>14.9</td><td>0.882</td><td>20.8</td><td>0.975</td><td>49.2</td><td>1.59</td><td>24.8</td><td>1.09</td><td>21.5</td><td>1.15</td></tr><tr><td>MDS</td><td>11.3</td><td>0.699</td><td>3.41</td><td>0.608</td><td>3.21</td><td>0.274</td><td>16.4</td><td>0.624</td><td>11.7</td><td>0.672</td><td>15.2</td><td>0.68</td><td>17.6</td><td>0.426</td><td>13.9</td><td>0.619</td><td>11.3</td><td>0.569</td></tr><tr><td>KNN</td><td>10.5</td><td>0.643</td><td>2.51</td><td>0.439</td><td>2.67</td><td>0.219</td><td>11.4</td><td>0.390</td><td>10.9</td><td>0.618</td><td>14.1</td><td>0.619</td><td>16.7</td><td>0.389</td><td>13.1</td><td>0.567</td><td>9.83</td><td>0.474</td></tr><tr><td>RLog</td><td>4.83</td><td>0.246</td><td>0.808</td><td>0.122</td><td>1.59</td><td>0.108</td><td>7.73</td><td>0.214</td><td>5.27</td><td>0.250</td><td>6.84</td><td>0.234</td><td>12.6</td><td>0.226</td><td>7.39</td><td>0.239</td><td>5.67</td><td>0.200</td></tr><tr><td>SIRC</td><td>16.3</td><td>1.04</td><td>3.90</td><td>0.701</td><td>8.48</td><td>0.817</td><td>15.1</td><td>0.564</td><td>17.1</td><td>1.03</td><td>20.7</td><td>0.968</td><td>22.1</td><td>0.612</td><td>19.05</td><td>0.913</td><td>14.8</td><td>0.819</td></tr><tr><td>Δ-MDS</td><td>5.00</td><td>0.257</td><td>2.19</td><td>0.380</td><td>2.43</td><td>0.194</td><td>9.63</td><td>0.304</td><td>5.43</td><td>0.260</td><td>8.28</td><td>0.311</td><td>12.5</td><td>0.224</td><td>7.81</td><td>0.263</td><td>6.50</td><td>0.276</td></tr><tr><td>Δ-KNN</td><td>4.60</td><td>0.230</td><td>1.42</td><td>0.237</td><td>1.99</td><td>0.149</td><td>8.52</td><td>0.252</td><td>4.99</td><td>0.231</td><td>7.55</td><td>0.272</td><td>12.1</td><td>0.207</td><td>7.32</td><td>0.235</td><td>5.89</td><td>0.225</td></tr><tr><td>Δ-MDS-RLog</td><td>4.13</td><td>0.197</td><td>1.09</td><td>0.175</td><td>1.60</td><td>0.109</td><td>7.13</td><td>0.185</td><td>4.52</td><td>0.200</td><td>6.27</td><td>0.204</td><td>11.1</td><td>0.170</td><td>6.51</td><td>0.193</td><td>5.12</td><td>0.177</td></tr><tr><td>Δ-KNN-RLog</td><td>3.98</td><td>0.187</td><td>0.770</td><td>0.115</td><td>1.45</td><td>0.093</td><td>7.14</td><td>0.186</td><td>4.36</td><td>0.190</td><td>6.13</td><td>0.196</td><td>11.3</td><td>0.175</td><td>6.43</td><td>0.187</td><td>5.01</td><td>0.163</td></tr></table>

Related to our work are approaches that combine selective classification and OOD detection, termed SCOD (Xia & Bouganis, 2022; Narasimhan et al., 2024). Such methods consider the ID classification and OOD distributions separately and seek to combine them into a single score function. These approaches are typically designed for semantic shifts and need adaptation for covariate shift. Our formulation avoids this by representing all distribution shifts through the general pair $( p _ { c } , p _ { w } )$ , which does not require distinguishing between shift types.

Finally, most closely related to our work is Liang et al. (2024), who study selective classification under both semantic and covariate shifts and introduce the Raw Logit (RLog) score. Our work differs in several ways: 1) we focus on covariate shifts, which we argue is more relevant in modern settings where large and variable label sets (e.g., from vision-language models) mitigate label drift; 2) we introduce a unified theoretical framework grounded in the Neyman–Pearson lemma, from which we derive new selector scores with formal optimality guarantees; and 3) we evaluate our methods on a broader class of models, including VLMs, whereas Liang et al. (2024) focus exclusively on standard supervised learning paradigms.

# 5 EXPERIMENTS

Datasets. We evaluate our methods across vision and language domains, with a primary focus on the former. For vision tasks, we use ImageNet-1K (Im-1K) and a suite of covariate-shifted variants: 1) ImageNet-Rendition (Im-R) (Hendrycks et al., 2020), 2) ImageNet-A (Im-A) (Hendrycks et al., 2021), 3) ObjectNet (ON) (Barbu et al., 2019), 4) ImageNetV2 (Im-V2) (Recht et al., 2019), 5) ImageNet-Sketch (Im-S) (Wang et al., 2019), and 6) ImageNet-C (Im-C) (Hendrycks & Dietterich, 2019). We group these datasets based on label coverage: full 1000-class coverage (Im-1K, Im-V2, Im-S, Im-C) and subsets of classes (Im-R, Im-A, ON). For language tasks, we evaluate on the Amazon Reviews dataset (Ni et al., 2019; Koh et al., 2021). To simulate realistic deployment scenarios involving distribution shift, following Liang et al. (2024) we evaluate on mixed test sets that combine in-distribution and covariate-shifted samples. For example, results reported on Im-C are computed on a combined test set of Im-1K and Im-C.

Classifiers and Baseline Selector Scores. We consider two families of classifiers for vision experiments, namely CLIP zero-shot VLMs (Radford et al., 2021) and supervised classifiers. Specifically, we use the CLIP model from Data Filtering Networks (DFN) (Fang et al., 2024) and EVA (Fang et al., 2023) for supervised learning, chosen for their state-of-the-art accuracy on ImageNet. Our focus is not on model training but on evaluating selector scores applied post-hoc. Note that for EVA, we restrict evaluation to datasets with full 1K class coverage as the model is trained on the complete ImageNet label set only. In contrast, CLIP can be adapted at inference time to arbitrary label subsets, so we evaluate it across all datasets. For language tasks, we fine-tune a DistilBERT (Sanh et al., 2019) model using LISA (Yao et al., 2022) on the Amazon Reviews training set and evaluate selective classification performance on the full test set.

Table 3: Supervised learning AURC (A) and NAURC (N) results with the EVA model at full coverage. Lower is better. AURC results are on the $1 0 ^ { - 2 }$ scale. Bold and underline denotes the best and second best result respectively. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Im-1K</td><td colspan="2">Im-V2</td><td colspan="2">Im-S</td><td colspan="2">Im-C</td><td colspan="2">Avg (1K)</td></tr><tr><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td></tr><tr><td>MSP</td><td>3.32</td><td>0.256</td><td>3.85</td><td>0.266</td><td>8.15</td><td>0.319</td><td>6.41</td><td>0.215</td><td>5.43</td><td>0.264</td></tr><tr><td>MaxLogit</td><td>4.53</td><td>0.371</td><td>5.16</td><td>0.379</td><td>10.3</td><td>0.437</td><td>7.98</td><td>0.301</td><td>6.99</td><td>0.372</td></tr><tr><td>Energy</td><td>6.82</td><td>0.590</td><td>7.58</td><td>0.589</td><td>14.0</td><td>0.641</td><td>11.1</td><td>0.474</td><td>9.89</td><td>0.573</td></tr><tr><td>MDS</td><td>4.01</td><td>0.322</td><td>4.32</td><td>0.307</td><td>7.26</td><td>0.271</td><td>6.80</td><td>0.236</td><td>5.60</td><td>0.284</td></tr><tr><td>KNN</td><td>4.00</td><td>0.321</td><td>4.31</td><td>0.306</td><td>7.15</td><td>0.265</td><td>6.77</td><td>0.234</td><td>5.56</td><td>0.282</td></tr><tr><td>RLog</td><td>2.33</td><td>0.161</td><td>2.72</td><td>0.168</td><td>5.90</td><td>0.197</td><td>5.50</td><td>0.163</td><td>4.11</td><td>0.172</td></tr><tr><td>SIRC</td><td>3.68</td><td>0.290</td><td>4.23</td><td>0.299</td><td>8.71</td><td>0.350</td><td>6.84</td><td>0.240</td><td>5.87</td><td>0.295</td></tr><tr><td>Δ-MDS</td><td>2.56</td><td>0.183</td><td>2.90</td><td>0.183</td><td>5.76</td><td>0.189</td><td>5.50</td><td>0.164</td><td>4.18</td><td>0.180</td></tr><tr><td>Δ-KNN</td><td>2.60</td><td>0.187</td><td>2.99</td><td>0.191</td><td>5.91</td><td>0.197</td><td>5.74</td><td>0.177</td><td>4.31</td><td>0.188</td></tr><tr><td>Δ-MDS-RLog</td><td>2.26</td><td>0.155</td><td>2.61</td><td>0.158</td><td>5.45</td><td>0.172</td><td>5.12</td><td>0.143</td><td>3.86</td><td>0.157</td></tr><tr><td>Δ-KNN-RLog</td><td>2.31</td><td>0.159</td><td>2.69</td><td>0.165</td><td>5.63</td><td>0.182</td><td>5.38</td><td>0.157</td><td>4.00</td><td>0.166</td></tr></table>

For baseline scores, we compare our proposed ∆-MDS, ∆-KNN, and their linear combinations with common OOD detection and uncertainty-based scores: MSP (Hendrycks & Gimpel, 2017), MCM (for CLIP), MaxLogit (Hendrycks et al., 2019), Energy (Liu et al., 2020), MDS (Lee et al., 2018), KNN (Sun et al., 2022), and RLog (Liang et al., 2024) as well as SIRC (Xia & Bouganis, 2022). As they are functionally similar, we abbreviate MCM as MSP when presenting CLIP results. Details of the baseline are provided in Appendix A.

Evaluation Metrics. We evaluate performance using two metrics: the Area Under the Risk-Coverage Curve (AURC) and the Normalized AURC (NAURC) (Cattelan & Silva, 2023). The AURC captures the joint performance of the classifier and selector across coverage levels. NAURC normalizes AURC to account for the classifier’s base error rate, providing a fairer comparison across models with different accuracies. Formally, NAURC is defined as:

$$
\mathrm{NAURC} (f, g) = \frac {\mathrm{AURC} (f , g) - \mathrm{AURC} (f , g ^ {*})}{R (f) - \mathrm{AURC} (f , g ^ {*})}, \tag {7}
$$

where $g ^ { * }$ denotes an oracle confidence function achieving optimal AURC, and $R ( f )$ is the risk of $f .$ . The oracle can be computed in practice using the ground-truth labels of the evaluation set. Intuitively, NAURC measures how close the selector g gets to the optimal, normalized by the classifier’s total error. Thus, while AURC is useful for understanding overall performance in the context of a specific model, NAURC enables fair selector comparisons across models by factoring out baseline classifier accuracy.

Selecting λ and k. Both λ and k can be selected on a validation set. We found that the simplest recipe to fitting λ is to balance the magnitudes of $s _ { 1 } ( \pmb { x } )$ and $s _ { 2 } ( { \pmb x } )$ , so that neither overpowers the other. For k in KNN-based scores, we find that $k \in [ 2 5 , 5 0 ]$ is a sweet spot. Full experimental settings are provided in Appendix $\mathrm { E , }$ and hyperparameter sensitivity analysis for λ and k is presented in appendix Fig. 2.

# 5.1 IMAGE EXPERIMENTS

We report full selective classification results for CLIP and EVA models in Table 1 and Table 3, respectively. First, let us consider CLIP results. We see that going from MDS and KNN to their NP-informed variants, ∆-MDS and ∆-KNN, leads to roughly 50% reduction in average AURC and NAURC, showing that the assumptions made in the NP-optimality theory hold well in practice. The best average performance is achieved by the linear combinations ∆-KNN-RLog and ∆-MDS-RLog, with ∆-KNN-RLog leading overall in both AURC and NAURC. RLog score ranks third on average, highlighting its strength as a standalone Table 2: Results on Amazon Reviews and its covariate shifted test set at full coverage using Distil-BERT trained with LISA.

<table><tr><td rowspan="2">Method</td><td colspan="2">In-D</td><td colspan="2">Cov Shift</td></tr><tr><td>A</td><td>N</td><td>A</td><td>N</td></tr><tr><td>MSP</td><td>12.2</td><td>0.368</td><td>13.9</td><td>0.401</td></tr><tr><td>MaxLogit</td><td>12.6</td><td>0.384</td><td>14.3</td><td>0.416</td></tr><tr><td>Energy</td><td>12.89</td><td>0.397</td><td>14.6</td><td>0.428</td></tr><tr><td>MDS</td><td>20.6</td><td>0.739</td><td>22.2</td><td>0.750</td></tr><tr><td>KNN</td><td>19.4</td><td>0.686</td><td>21.3</td><td>0.711</td></tr><tr><td>RLog</td><td>12.4</td><td>0.376</td><td>14.1</td><td>0.410</td></tr><tr><td>SIRC</td><td>12.3</td><td>0.370</td><td>14.0</td><td>0.403</td></tr><tr><td>Δ-MDS</td><td>12.7</td><td>0.389</td><td>14.4</td><td>0.422</td></tr><tr><td>Δ-KNN</td><td>12.4</td><td>0.374</td><td>14.2</td><td>0.412</td></tr><tr><td>Δ-MDS-RLog</td><td>12.2</td><td>0.368</td><td>13.9</td><td>0.401</td></tr><tr><td>Δ-KNN-RLog</td><td>12.0</td><td>0.358</td><td>13.8</td><td>0.394</td></tr><tr><td>Δ-MDS-MSP</td><td>11.9</td><td>0.354</td><td>13.6</td><td>0.387</td></tr><tr><td>Δ-KNN-MSP</td><td>12.0</td><td>0.359</td><td>13.8</td><td>0.396</td></tr></table>

Table 4: Ablation experiments on DFN CLIP. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Im-1K</td><td colspan="2">Im-R</td><td colspan="2">Im-A</td><td colspan="2">ON</td><td colspan="2">Im-V2</td><td colspan="2">Im-S</td><td colspan="2">Im-C</td><td colspan="2">Avg (1K)</td><td colspan="2">Avg (all)</td></tr><tr><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td></tr><tr><td colspan="19">Ablations on Δ-KNN</td></tr><tr><td>Δ-KNN no avg</td><td>4.66</td><td>0.234</td><td>1.40</td><td>0.234</td><td>2.16</td><td>0.167</td><td>8.87</td><td>0.268</td><td>5.11</td><td>0.239</td><td>7.63</td><td>0.276</td><td>12.4</td><td>0.216</td><td>7.45</td><td>0.241</td><td>6.03</td><td>0.233</td></tr><tr><td>Δ-KNN w/ avg</td><td>4.60</td><td>0.230</td><td>1.42</td><td>0.237</td><td>1.99</td><td>0.149</td><td>8.52</td><td>0.252</td><td>4.99</td><td>0.231</td><td>7.55</td><td>0.272</td><td>12.1</td><td>0.207</td><td>7.32</td><td>0.235</td><td>5.89</td><td>0.225</td></tr><tr><td colspan="19">Ablations on linear combinations</td></tr><tr><td>Δ-MDS-Δ-KNN</td><td>4.68</td><td>0.235</td><td>1.98</td><td>0.237</td><td>2.26</td><td>0.149</td><td>9.06</td><td>0.252</td><td>5.11</td><td>0.231</td><td>7.90</td><td>0.272</td><td>12.2</td><td>0.207</td><td>7.46</td><td>0.235</td><td>6.16</td><td>0.225</td></tr><tr><td>MSP-RLog</td><td>4.82</td><td>0.245</td><td>0.800</td><td>0.120</td><td>1.56</td><td>0.105</td><td>7.51</td><td>0.204</td><td>5.26</td><td>0.249</td><td>6.81</td><td>0.232</td><td>12.5</td><td>0.222</td><td>7.35</td><td>0.237</td><td>5.61</td><td>0.197</td></tr><tr><td>Δ-KNN-MSP</td><td>4.57</td><td>0.228</td><td>1.22</td><td>0.199</td><td>1.82</td><td>0.131</td><td>7.58</td><td>0.207</td><td>4.94</td><td>0.228</td><td>7.40</td><td>0.264</td><td>11.8</td><td>0.210</td><td>7.18</td><td>0.244</td><td>5.62</td><td>0.253</td></tr><tr><td>Δ-KNN-RLog</td><td>3.98</td><td>0.187</td><td>0.770</td><td>0.115</td><td>1.49</td><td>0.093</td><td>7.14</td><td>0.186</td><td>4.36</td><td>0.190</td><td>6.13</td><td>0.196</td><td>11.3</td><td>0.175</td><td>6.43</td><td>0.187</td><td>5.01</td><td>0.163</td></tr></table>

logit-based selector. Motivated by this strong performance, we use RLog in combination with our distance-based scores. We plot the risk-coverage curves for selected datasets in Fig. 3 of the appendix, showing our methods consistently demonstrate the most favorable trade-off across all coverage levels, remaining stable even at low coverage.

For practitioners aiming to identify the best overall selective classification setup that considers both the base classifier and the selector, one approach is to compare performance using the AURC metric. On the 1K-class datasets, EVA paired with ∆-MDS-RLog achieves an AURC of 3.86, outperforming the DFN CLIP model with ∆-KNN-RLog at 5.01. Despite similar NAURC values, EVA’s higher Im-1K base accuracy (84.33% vs. DFN CLIP’s 80.39%) makes it the preferred choice when considering both components. Intuitively, the optimal setup involves pairing the best selector (here, ∆-MDS-RLog) with the most accurate base classifier.

For EVA, the ranking is reversed: ∆-MDS-RLog achieves the best overall performance, followed by ∆-KNN-RLog. This supports our hypothesis that MDS-based methods are particularly effective for supervised models due to the close connection between softmax classifiers and Gaussian Discriminant Analysis (Lee et al., 2018), which justifies the Gaussian assumptions used in MDS. In contrast, CLIP models trained with contrastive learning (Radford et al., 2021) do not satisfy these assumptions, making the nonparametric ∆-KNN combination more suitable. The bottom row of Fig. 3 of the appendix confirms that ∆-MDS-RLog yields the best risk-coverage behavior for supervised learning across all coverage levels. Comparing average NAURC on the full 1K-class datasets, ∆-MDS-RLog with EVA is the top performing selector score, with a slightly better score (0.157) than the best performer on CLIP (0.163).

To verify that the performance gains of our methods stem from actual algorithmic improvements, rather than large-scale pretraining of CLIP or EVA potentially mitigating distribution shifts, we also evaluate on ResNet50 trained solely on ImageNet-1K. The results in appendix Table 7 show that our methods perform the best, consistent with earlier findings.

Semantic Shift Experiments. For completeness, we also report results for experiments on datasets that are semantic shifts to ImageNet-1K in Appendix Table 8. In agreement with the covariate shift experiments, our proposed methods achieve the best performance on this benchmark.

# 5.2 LANGUAGE EXPERIMENTS

Table 2 presents results on the Amazon Reviews dataset. Unlike the vision tasks, the best-performing method is ∆-MDS-MSP, followed closely by ∆-MDS-RLog and ∆-KNN-MSP. Since LISA (Yao et al., 2022) uses a softmax classification objective, the superiority of MDS-based selectors supports our hypothesis about their suitability for supervised models. Interestingly, MSP outperforms RLog in this domain, resulting in better performance when combined with ∆-MDS. This highlights another important practical insight that the best linear combination often involves pairing the top-performing standalone distance-based and logit-based score.

Table 5: Ablation results using DFN-CLIP on ImageNet-1K where the fraction of labeled samples used in feature computation for our proposed methods are varied. 

<table><tr><td rowspan="2">Method</td><td colspan="2">0.1%</td><td colspan="2">1%</td><td colspan="2">10%</td><td colspan="2">50%</td><td colspan="2">100%</td></tr><tr><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td></tr><tr><td>Δ-MDS-RLog</td><td>-</td><td>-</td><td>10.5</td><td>0.638</td><td>4.19</td><td>0.202</td><td>4.14</td><td>0.198</td><td>4.13</td><td>0.197</td></tr><tr><td>Δ-KNN-RLog</td><td>4.81</td><td>0.245</td><td>4.58</td><td>0.229</td><td>4.17</td><td>0.200</td><td>3.98</td><td>0.188</td><td>3.98</td><td>0.187</td></tr></table>

# 5.3 ABLATIONS

Design Choices. Table 4 summarizes several ablation experiments on the design choices of our proposed methods. First, we justify averaging the top-k nearest neighbor distances in ∆-KNN rather than using the k-th distance alone. This modification yields measurable gains, where average AURC improves from 6.03 to 5.89 and NAURC improves from 0.233 to 0.225. We also investigate various combinations of selector scores. For CLIP, ∆-KNN-RLog remains the best across all configurations, outperforming both double-distance combinations (e.g., ∆-MDS-∆-KNN) and doublelogit combinations (e.g., MSP-RLog). Notably, pairing ∆-KNN with RLog significantly outperforms pairing it with MSP, further validating RLog’s role as a strong logit-based complement.

Sample Efficiency. Although our methods require a one-time feature computation step, this cost is amortized over all future inference runs as the resulting features can be cached. Nevertheless, to evaluate performance in low-data or low-computation resource regimes, we conducted ablations limiting the amount of labeled samples used. The results in Table 5 show that both methods are surprisingly stable. ∆-KNN is especially robust, maintaining strong performance with as little as 0.1% of labeled data. As expected, ∆-MDS degrades at the 1% level due to the difficulty of estimating per-class statistics with so few samples, and is not applicable at 0.1% (roughly 1 image per class). Importantly, ∆-KNN-RLog continues to outperform RLog at 1% and matches it at 0.1% (see Table 1), indicating that our method is still preferable whenever even a small amount of labeled data is accessible.

# 6 CONCLUSION

We presented a framework for designing selector functions for selective classification, grounded in the Neyman–Pearson lemma. This reveals that the optimal selection score is a monotonic transformation of a likelihood ratio, which unifies several existing methods. We proposed two novel distance-based scores and their linear combinations with logit-based baselines. Experiments across vision and language demonstrate that our methods achieve state-of-the-art performance across diverse settings.

Limitations and Future Work. While our focus has been on classification, the Neyman–Pearson framework is general and broadly applicable to other predictive tasks. Exploring selective prediction in settings where uncertainty plays a critical role, such as semantic segmentation and time series forecasting, presents promising future directions. Additionally, extending these ideas to generative models such as LLMs is another exciting avenue for future work.

# ACKNOWLEDGEMENTS

This research / project is supported by A\*STAR under its National Robotics Programme (NRP) (Award M23NBK0053). The authors also acknowledge support from Google (Google South Asia and South-East Asia Award).

# ETHICS STATEMENT

This work does not involve human subjects, and it relies solely on publicly available models and datasets with all attributions provided. Based on the scope of our methods and results, we do not identify ethical issues that require special attention, and we are not aware of immediate harmful applications arising from this research.

# REPRODUCIBILITY STATEMENT

We include the theoretical and experimental details necessary to reproduce our findings. Proofs and supporting discussions for all theoretical claims appear in appendix Sec. B and Sec. C. For implementation clarity, we provide pseudocode for our methods in appendix Sec. D and report experimental setup and hyperparameters in appendix Sec. E.

# LLM USAGE

We used large language models to assist in formulating and checking mathematical proofs and to improve grammar and writing clarity. All outputs from the LLMs were reviewed by the authors before inclusion, and the authors take full responsibility for the paper’s content and the accuracy of the presented results.

# REFERENCES

Anastasios N Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv preprint arXiv:2107.07511, 2021.   
Anastasios Nikolas Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal risk control. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=33XGfHLtZg.   
Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. Advances in neural information processing systems, 32, 2019.   
Peter L Bartlett and Marten H Wegkamp. Classification with a reject option using a hinge loss. Journal of Machine Learning Research, 9(8), 2008.   
Stephen Bates, Anastasios Angelopoulos, Lihua Lei, Jitendra Malik, and Michael Jordan. Distributionfree, risk-controlling prediction sets. Journal of the ACM (JACM), 68(6):1–34, 2021.   
Andy Brock, Soham De, Samuel L Smith, and Karen Simonyan. High-performance large-scale image recognition without normalization. In International conference on machine learning, pp. 1059–1071. PMLR, 2021.   
Luís Felipe P Cattelan and Danilo Silva. How to fix a broken confidence estimator: Evaluating post-hoc methods for selective classification with deep neural networks. arXiv preprint arXiv:2305.15508, 2023.   
Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2818–2829, 2023.

C Chow. On optimum recognition error and reject tradeoff. IEEE Transactions on information theory, 16(1):41–46, 1970.   
Ran El-Yaniv et al. On the foundations of noise-free selective classification. Journal of Machine Learning Research, 11(5), 2010.   
Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander T Toshev, and Vaishaal Shankar. Data filtering networks. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=KAk6ngZ09F.   
Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19358–19369, 2023.   
Adam Fisch, Tommi S. Jaakkola, and Regina Barzilay. Calibrated selective classification. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview. net/forum?id=zFhNBs8GaV.   
Giorgio Fumera and Fabio Roli. Support vector machines with embedded reject option. In Pattern Recognition with Support Vector Machines: First International Workshop, SVM 2002 Niagara Falls, Canada, August 10, 2002 Proceedings, pp. 68–82. Springer, 2002.   
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050–1059. PMLR, 2016.   
Ido Galil, Mohammed Dabbah, and Ran El-Yaniv. What can we learn from the selective prediction and uncertainty estimation performance of 523 imagenet classifiers? In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=p66AzKi6Xim.   
Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. Advances in neural information processing systems, 30, 2017.   
Yonatan Geifman and Ran El-Yaniv. Selectivenet: A deep neural network with an integrated reject option. In International conference on machine learning, pp. 2151–2159. PMLR, 2019.   
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pp. 1321–1330. PMLR, 2017.   
Martin E Hellman. The nearest neighbor classification rule with a reject option. IEEE Transactions on Systems Science and Cybernetics, 6(3):179–185, 1970.   
Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019.   
Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=Hkg4TI9xl.   
Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Joe Kwon, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling out-of-distribution detection for real-world settings. arXiv preprint arXiv:1911.11132, 2019.   
Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Lixuan Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. 2021 ieee. In CVF International Conference on Computer Vision (ICCV), volume 2, 2020.   
Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15262–15271, 2021.

Alvin Heng and Harold Soh. Detecting covariate shifts with vision-language foundation models. In ICLR 2025 Workshop on Foundation Models in the Wild, 2025. URL https://openreview. net/forum?id=SOHJFKH8oc.   
Alvin Heng, Harold Soh, et al. Out-of-distribution detection with a single unconditional diffusion model. Advances in Neural Information Processing Systems, 37:43952–43974, 2024.   
Lang Huang, Chao Zhang, and Hongyang Zhang. Self-adaptive training: beyond empirical risk minimization. Advances in neural information processing systems, 33:19365–19376, 2020.   
Heinrich Jiang, Been Kim, Melody Guan, and Maya Gupta. To trust or not to trust a classifier. Advances in neural information processing systems, 31, 2018.   
Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, et al. Wilds: A benchmark of in-the-wild distribution shifts. In International conference on machine learning, pp. 5637–5664. PMLR, 2021.   
Yann LeCun, Bernhard Boser, John Denker, Donnie Henderson, Richard Howard, Wayne Hubbard, and Lawrence Jackel. Handwritten digit recognition with a back-propagation network. Advances in neural information processing systems, 2, 1989.   
Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. Advances in neural information processing systems, 31, 2018.   
Erich Leo Lehmann, Joseph P Romano, et al. Testing statistical hypotheses, volume 3. Springer, 1986.   
Hengyue Liang, Le Peng, and Ju Sun. Selective classification under distribution shifts. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/ forum?id=dmxMGW6J7N.   
Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. Advances in neural information processing systems, 33:21464–21475, 2020.   
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.   
Ziyin Liu, Zhikang Wang, Paul Pu Liang, Russ R Salakhutdinov, Louis-Philippe Morency, and Masahito Ueda. Deep gamblers: Learning to abstain with portfolio theory. Advances in Neural Information Processing Systems, 32, 2019.   
Don O Loftsgaarden and Charles P Quesenberry. A nonparametric estimate of a multivariate density function. The Annals of Mathematical Statistics, 36(3):1049–1051, 1965.   
Yifei Ming, Ziyang Cai, Jiuxiang Gu, Yiyou Sun, Wei Li, and Yixuan Li. Delving into out-ofdistribution detection with vision-language representations. Advances in neural information processing systems, 35:35087–35102, 2022.   
Harikrishna Narasimhan, Aditya Krishna Menon, Wittawat Jitkrittum, and Sanjiv Kumar. Plugin estimators for selective classification with out-of-distribution detection. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=DASh78rJ7g.   
Jerzy Neyman and Egon Sharpe Pearson. Ix. on the problem of the most efficient tests of statistical hypotheses. Philosophical Transactions of the Royal Society of London. Series A, Containing Papers of a Mathematical or Physical Character, 231(694-706):289–337, 1933.   
Jianmo Ni, Jiacheng Li, and Julian McAuley. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pp. 188–197, 2019.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.   
Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pp. 5389–5400. PMLR, 2019.   
Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.   
Bernard W Silverman. Density estimation for statistics and data analysis. Routledge, 2018.   
Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li. Out-of-distribution detection with deep nearest neighbors. In International Conference on Machine Learning, pp. 20827–20840. PMLR, 2022.   
Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random world, volume 29. Springer, 2005.   
Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. Advances in Neural Information Processing Systems, 32, 2019.   
Haoqi Wang, Zhizhong Li, Litong Feng, and Wayne Zhang. Vim: Out-of-distribution with virtuallogit matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4921–4930, 2022.   
Guoxuan Xia and Christos-Savvas Bouganis. Augmenting softmax information for selective classification with out-of-distribution data. In Proceedings of the Asian Conference on Computer Vision, pp. 1995–2012, 2022.   
Huaxiu Yao, Yu Wang, Sai Li, Linjun Zhang, Weixin Liang, James Zou, and Chelsea Finn. Improving out-of-distribution robustness via selective augmentation. In International Conference on Machine Learning, pp. 25407–25437. PMLR, 2022.   
Puning Zhao and Lifeng Lai. Analysis of knn density estimation. IEEE Transactions on Information Theory, 68(12):7971–7995, 2022.

# Appendix for “Know When to Abstain: Optimal Selective Classification with Likelihood Ratios”

# A DESCRIPTION OF BASELINES

In this section we provide a brief description of each baseline considered in this work.

Maximum Softmax Probability (MSP) (Hendrycks & Gimpel, 2017). Given an input x, let the classifier output logits be $\{ l ^ { ( k ) } \} _ { k = 1 } ^ { K }$ and corresponding softmax probabilities $p _ { \theta } ( y = k | \mathbf { x } ) =$ softmax(l(k)). The MSP score is defined as

$$
s _ {\mathrm{MSP}} (\boldsymbol {x}) = \max _ {k \in \{1, \dots , K \}} p _ {\theta} (y = k | \boldsymbol {x}).
$$

MSP is commonly used as an OOD/confidence score: larger values indicate the model places high probability mass on a single class, and thus the input is treated as more “in-distribution” or “confident”. For selective classification, we threshold this score (and subsequent scores) to form the selector $g _ { s , \gamma } ( \pmb { x } ) = \mathbb { 1 } \big [ s _ { \mathrm { M S P } } ( \pmb { x } ) > \gamma \big ]$ ].

Maximum Logit (MaxLogit) (Hendrycks et al., 2019). The MaxLogit score is defined as

$$
s _ {\mathrm{MaxLogit}} (\boldsymbol {x}) = \max _ {k \in \{1, \dots , K \}} l ^ {(k)}
$$

Intuitively, larger values indicate that at least one class is assigned a large unnormalized score, and thus the model is more confident in its prediction.

Energy (Liu et al., 2020). The energy score is defined from logits as the Helmholtz free energy

$$
E (\boldsymbol {x}) = - T \log \sum_ {k = 1} ^ {K} \exp \left(l ^ {(k)} / T\right),
$$

where $T > 0$ is a temperature parameter (we set $T = 1 )$ . For selective classification we use the negative energy (so that larger values indicate more in-distribution-like inputs):

$$
s _ {\text { Energy }} (\boldsymbol {x}) = \log \sum_ {k = 1} ^ {K} \exp \left(l ^ {(k)}\right).
$$

Mahalanobis Distance (MDS) (Lee et al., 2018). Let $z = \phi ( { \pmb x } )$ denote the feature representation of x (e.g., penultimate features of a deep network). Using training data, we estimate the empirical mean feature $\mu _ { i }$ for each class $i \in \{ 1 , \ldots , K \}$ and a shared (tied) covariance matrix Σ across classes. The MDS score is then

$$
s _ {\mathrm{MDS}} (\boldsymbol {x}) = \max _ {i \in \{1, \dots , K \}} - (\boldsymbol {z} - \mu_ {i}) ^ {\top} \Sigma^ {- 1} (\boldsymbol {z} - \mu_ {i}),
$$

i.e., the negative squared Mahalanobis distance to the closest class centroid. Intuitively, $s _ { \mathrm { M D S } } ( { \pmb x } )$ is large when x lies in a high-density region of some class under the fitted Gaussian discriminant model, and small when x is far from all class clusters.

k-Nearest Neighbors (KNN) (Sun et al., 2022). Let $\{ z _ { j } \} _ { j = 1 } ^ { n }$ denote the set of training features (we normalize the features). Define $r _ { k } ( z )$ as the Euclidean distance from z to its k-th nearest neighbor among $\{ z _ { j } \}$ . The KNN score is

$$
s _ {\mathrm{KNN}} (\boldsymbol {x}) = - r _ {k} (\boldsymbol {z}),
$$

so that inputs lying in denser regions of the training feature manifold (smaller neighbor distances) receive higher scores.

Raw Logits (RLog) (Liang et al., 2024). RLog uses the confidence margin between the top two logits as a scale-robust confidence score. Let $\bar { l } ^ { ( 1 ) } \ge l ^ { ( 2 ) } \ge \cdots \ge l ^ { ( k ) }$ be the logits sorted in descending order for a given input x. We define

$$
s _ {\mathrm{RLog}} (\pmb {x}) = l ^ {(1)} - l ^ {(2)}
$$

Intuitively, $s _ { \mathrm { R L o g } } ( \pmb { x } )$ is large when there is a clear “winner” class, and small when the classifier is uncertain between competing labels. Liang et al. (2024) argue that using a logit-space margin yields a more robust score under classifier miscalibration and post-hoc logit transformations (e.g., temperature scaling), since it depends only on the relative separation between the top classes.

Softmax Information Retaining Combination (SIRC) (Xia & Bouganis, 2022). Let $S _ { 1 } ( \pmb { x } )$ be a primary softmax-derived confidence score, and let $S _ { 2 } ( \pmb { x } )$ be an auxiliary feature-based score. We choose $S _ { 1 }$ to be MSP and $S _ { 2 }$ to be the $L _ { 1 }$ norm, in line with Xia & Bouganis (2022). SIRC combines these via

$$
s _ {\mathrm{SIRC}} (\boldsymbol {x}) = - \big (S _ {1} ^ {\max} - S _ {1} (\boldsymbol {x}) \big) \Big (1 + \exp \big (- b [ S _ {2} (\boldsymbol {x}) - a ] \big) \Big),
$$

where $S _ { 1 } ^ { \mathrm { m a x } }$ is the maximum attainable value of $S _ { 1 } \ ( \mathrm { e . g . } , S _ { 1 } ^ { \mathrm { m a x } } = 1$ for MSP), and $a , b$ control how strongly $S _ { 2 }$ influences the score. We follow Xia & Bouganis (2022) and set a and b using in-distribution statistics of $S _ { 2 } \ : ( a = \mu _ { S _ { 2 } } - 3 \sigma _ { S _ { 2 } }$ and $b = 1 / \sigma _ { S _ { 2 } } )$ .

# B PROOFS

Theorem 1. Let $\hat { y } ( x ) = \arg \operatorname* { m a x } _ { k \in \{ 1 , . . . , K \} } p _ { \theta } ( y = k \mid x )$ be the predicted label and define the event $C : = \{ \hat { y } ( X ) = Y \}$ that the classifier is correct. Assume the classifier is calibrated for top-1 correctness, i.e. $ , P ( C \mid X = \pmb { x } ) = \operatorname* { m a x } _ { k } p _ { \theta } ( y = k \mid \pmb { x } ) = : d ^ { ( 1 ) } ( \pmb { x } )$ . Then MSP is Neyman–Pearson optimal for selective classification. Moreover, under the additional assumption that the softmax distribution is concentrated on the top two classes, i.e., $\begin{array} { r } { L : = \sum _ { i > 3 } d ^ { ( i ) } \ll \dot { d } ^ { ( 2 ) } } \end{array}$ , the RLog score is also Neyman–Pearson optimal.

Proof. Recall that we denote $l ^ { ( 1 ) } \ge \cdots \ge l ^ { ( K ) }$ as the logits predicted by the classifier for a given input x (sorted in descending order) and $d ^ { ( i ) } = \mathrm { s o f t m a x } ( l ^ { ( i ) } )$ the corresponding softmax probabilities. Then $s _ { \mathrm { M S P } } ( \pmb { x } ) = d ^ { ( 1 ) } ( \pmb { x } )$ and $s _ { \mathrm { R L o g } } ( \pmb { x } ) = l ^ { ( 1 ) } ( \pmb { x } ) - l ^ { ( 2 ) } ( \pmb { x } )$ .

Define

$$
p _ {c} (\boldsymbol {x}) := p (\boldsymbol {x} \mid C) \quad \text { and } \quad p _ {w} (\boldsymbol {x}) := p (\boldsymbol {x} \mid \neg C),
$$

i.e., the (test-time) input densities conditioned on the classifier being correct or wrong, respectively. Let $\pi : = P ( C )$ and define the posterior correctness probability $q ( { \pmb x } ) : = P ( C \mid X = \mathbf { \bar { x } } )$ .

MSP Optimality. By Bayes’ rule,

$$
\begin{array}{l} q (\boldsymbol {x}) = \frac {p (\boldsymbol {x} \mid C) P (C)}{p (\boldsymbol {x} \mid C) P (C) + p (\boldsymbol {x} \mid \neg C) P (\neg C)} \\ = \frac {p _ {c} (\boldsymbol {x}) \pi}{p _ {c} (\boldsymbol {x}) \pi + p _ {w} (\boldsymbol {x}) (1 - \pi)}. \\ \end{array}
$$

Equivalently, the posterior odds satisfy

$$
\frac {q (\boldsymbol {x})}{1 - q (\boldsymbol {x})} = \frac {p _ {c} (\boldsymbol {x})}{p _ {w} (\boldsymbol {x})} \cdot \frac {\pi}{1 - \pi}.
$$

Since the classifier is (perfectly) calibrated for top-1 correctness such that $d ^ { ( 1 ) } ( { \pmb x } ) = P ( C \mid { \cal X } =$ ${ \pmb x } ) = q ( { \pmb x } )$ ,

$$
s _ {\mathrm{MSP}} (\boldsymbol {x}) = d ^ {(1)} (\boldsymbol {x}) = q (\boldsymbol {x}) = \frac {\frac {\pi}{1 - \pi} \frac {p _ {c} (\boldsymbol {x})}{p _ {w} (\boldsymbol {x})}}{1 + \frac {\pi}{1 - \pi} \frac {p _ {c} (\boldsymbol {x})}{p _ {w} (\boldsymbol {x})}}. \tag {8}
$$

The mapping $\textstyle h ( z ) = { \frac { z } { 1 + z } }$ is strictly increasing for $z \geq 0$ since $\begin{array} { r } { h ^ { \prime } ( z ) = \frac { 1 } { ( 1 + z ) ^ { 2 } } > 0 } \end{array}$ 1(1+z)2 > 0, and scaling by 1+z the positive constant $\frac { \pi } { 1 - \pi }$ preserves monotonicity. Hence $s _ { \mathrm { M S P } } ( { \pmb x } )$ is a monotone transformation of the likelihood rati o pc(x)pw(x) , and is Neyman–Pearson optimal by Corollary 1. $\frac { p _ { c } ( { \pmb x } ) } { p _ { w } ( { \pmb x } ) }$

RLog Optimality. We can express RLog as the logarithm of the ratio of the top two softmax values:

$$
\frac {d ^ {(1)} (x)}{d ^ {(2)} (\pmb {x})} = \frac {e ^ {l ^ {(1)} (\pmb {x})}}{e ^ {l ^ {(2)} (\pmb {x})}} = e ^ {l ^ {(1)} (\pmb {x}) - l ^ {(2)} (\pmb {x})} = e ^ {s _ {\mathrm{RLog}} (\pmb {x})},
$$

thus $\begin{array} { r } { s _ { \mathrm { R L o g } } ( \pmb { x } ) = \log \frac { d ^ { ( 1 ) } ( \pmb { x } ) } { d ^ { ( 2 ) } ( \pmb { x } ) } } \end{array}$ d(2)(x) . Observe that

$$
\frac {d ^ {(1)} (\boldsymbol {x})}{d ^ {(2)} (\boldsymbol {x})} = \frac {d ^ {(1)} (\boldsymbol {x})}{1 - d ^ {(1)} (\boldsymbol {x})} \left(1 + \frac {L (\boldsymbol {x})}{d ^ {(2)} (\boldsymbol {x})}\right), \tag {9}
$$

where $\begin{array} { r } { L ( { \boldsymbol x } ) = 1 - d ^ { ( 1 ) } ( { \boldsymbol x } ) - d ^ { ( 2 ) } ( { \boldsymbol x } ) = \sum _ { i > 3 } d ^ { ( i ) } ( { \boldsymbol x } ) \geq 0 } \end{array}$ . In the binary classification case, $L ( { \pmb x } ) = 0$ and hence

$$
s _ {\mathrm{RLog}} (\pmb {x}) = \log \frac {d ^ {(1)} (\pmb {x})}{1 - d ^ {(1)} (\pmb {x})} = \log \frac {q (\pmb {x})}{1 - q (\pmb {x})} = \log \frac {p _ {c} (\pmb {x})}{p _ {w} (\pmb {x})} + \log \frac {\pi}{1 - \pi},
$$

which differs from log $\frac { p _ { c } ( { \pmb x } ) } { p _ { w } ( { \pmb x } ) }$ only by an additive constant. Since $\log ( \cdot )$ is strictly increasing and additive constants do not change ordering, $s _ { \mathrm { R L o g } }$ is Neyman–Pearson optimal by Corollary 1.

In the multiclass case, the extra term log $\begin{array} { r } { \left( 1 + \frac { L ( { \pmb x } ) } { d ^ { ( 2 ) } ( { \pmb x } ) } \right) } \end{array}$ may vary across samples and can in principle affect ordering. Under the stated assumption $L ( x ) \ll d ^ { ( 2 ) } ( x )$ , which is empirically supported by high top-5 classification accuracies in prior works (Liu et al., 2021; Brock et al., 2021), this term is small and varies little, so sRLog(x) ≈ log q(x)1−q(x) $\begin{array} { r } { s _ { \mathrm { R L o g } } ( \pmb { x } ) \approx \log \frac { q ( \pmb { x } ) } { 1 - q ( \pmb { x } ) } } \end{array}$ and remains an approximately monotone proxy for lo g pc(x)pw(x) , yielding (approximate) Neyman–Pearson optimality. $\frac { p _ { c } ( \pmb { x } ) } { p _ { w } ( \pmb { x } ) }$ □

Theorem 2. Let $Z = \phi ( \pmb { x } ) \in \mathbb { R } ^ { d }$ be the feature representation of input x. Let C be the event the classifier makes a correct prediction and ¬C its negation. Assume $\mathsf { \bar { Z } } | \mathcal { C } \sim p _ { c } = \mathcal { N } ( \mu _ { i } ^ { c } , \Sigma ^ { c } )$ and $Z | \lnot { \mathcal { C } } \sim p _ { w } = { \mathcal { N } } ( \mu _ { i } ^ { w } , \Sigma ^ { w } )$ . Then the $\Delta { - } M D S$ score $s _ { \Delta - M D S } ( { \pmb x } )$ is Neyman–Pearson optimal for selective classification.

Proof. The likelihood of a multivariate Gaussian in $\begin{array} { r l r l } { \mathbb { R } ^ { d } } & { { } \mathrm { i s } } & { p ( z ; \mu , \Sigma ) } & { } \end{array} =$ $( 2 \pi ) ^ { - d / 2 } \operatorname * { d e t } ( \Sigma ) ^ { - 1 / 2 } \exp \left( - \textstyle { \frac { 1 } { 2 } } ( x - \mu ) ^ { \top } \Sigma ^ { - 1 } ( x - \mu ) \right)$ .

We see that the Mahalanobis distance $D ( z ; \mu , \Sigma )$ is proportional to the log-likelihood of the multivariate Gaussian. As such, assuming that the underlying $p _ { c }$ and $p _ { w }$ follow multivariate Gaussians of the form $\mathcal { N } ( \mu _ { i } ^ { c } , \Sigma ^ { c } )$ and $\mathcal { N } ( \mu _ { i } ^ { w } , \Sigma ^ { w } )$ respectively,

$$
\begin{array}{l} s _ {\Delta \text {-MDS}} (\boldsymbol {x}) = D _ {\mathrm{MDS}} (\boldsymbol {x}; \mu_ {i} ^ {c}, \Sigma^ {c}) - D _ {\mathrm{MDS}} (\boldsymbol {x}; \mu_ {i} ^ {w}, \Sigma^ {w}) \\ = 2 \log \frac {p _ {c} (\boldsymbol {z} ; \mu_ {i} ^ {c} , \Sigma^ {c})}{p _ {w} (\boldsymbol {z} ; \mu_ {i} ^ {w} , \Sigma^ {w})} + \log \frac {\det \Sigma^ {c}}{\det \Sigma^ {w}}. \\ \end{array}
$$

Therefore, $s _ { \Delta - \mathrm { M D S } } ( { \pmb x } )$ is a monotone transform of ${ p _ { c } / p _ { w } }$ and is Neyman–Pearson optimal by Corollary 1.

![](images/9be0676456605dac7e94090b6586297d10029fec079c03aa1496fc7abd1b7fba.jpg)

Theorem 3. Let $Z = \phi ( \pmb { x } ) \in \mathbb { R } ^ { d }$ be the feature representation of input x, and let C denote the event that the classifier makes a correct prediction. Suppose $Z \mid \mathcal { C } \sim p _ { c }$ and $Z | \neg { } C \sim p _ { w }$ are arbitrary continuous densities bounded away from zero. Let $N _ { c } = \vert A _ { c } \vert$ and $N _ { w } = | A _ { w } | . ~ I f k \to \infty$ while $k / N _ { c }  0$ and $k / N _ { w }  0$ as $N _ { c } , N _ { w }  \infty$ , then ${ \pmb s } _ { \Delta - K N N } ( { \pmb x } )$ is a Neyman–Pearson optimal selector.

Proof. The empirical likelihood of the KNN density estimator (Silverman, 2018; Zhao & Lai, 2022) is given by

$$
\hat {p} _ {c} (\boldsymbol {z}) = \frac {k}{N _ {c} V _ {d} (u _ {k} (\boldsymbol {z})) ^ {d}}, \quad \hat {p} _ {w} (\boldsymbol {z}) = \frac {k}{N _ {w} V _ {d} (v _ {k} (\boldsymbol {z})) ^ {d}}, \tag {10}
$$

where $k \geq 2 , u _ { k } ( z )$ and $v _ { k } ( z )$ are the Euclidean distances from z to its k-th nearest neighbor from $A _ { c }$ and $A _ { w }$ and $V _ { d }$ is the unit-ball volume in $\mathbb { R } ^ { d }$ . A classic result of non-parametric nearest neighbor density estimation (Loftsgaarden & Quesenberry, 1965) states that as $k  \infty$ but $k / N _ { c } \ \xrightarrow [ ] { } \ 0$ , $k / N _ { w }  0$ , then $\hat { p } _ { c } ( z )  p _ { c } ( z )$ and $\hat { p } _ { w } ( z ) \to p _ { w } ( z )$ for every z. In other words, the empirical KNN density estimator converges to the true density under the stated asymptotic conditions.

One can see that the difference in log-likelihoods is

$$
\begin{array}{l} \log \hat {p} _ {c} (\boldsymbol {z}) - \log \hat {p} _ {w} (\boldsymbol {z}) \\ = - d \log u _ {k} (\boldsymbol {z}) + d \log v _ {k} (\boldsymbol {z}) + \log \frac {N _ {w}}{N _ {c}}. \tag {11} \\ \end{array}
$$

Therefore

$$
\begin{array}{l} s _ {\Delta \text {-KNN}} (\boldsymbol {z}) \triangleq - \log u _ {k} (\boldsymbol {z}) + \log v _ {k} (\boldsymbol {z}) \\ = \frac {1}{d} \log \frac {\hat {p} _ {c} (\boldsymbol {z})}{\hat {p} _ {w} (\boldsymbol {z})} - \frac {1}{d} \log \frac {N _ {w}}{N _ {c}}. \tag {12} \\ \end{array}
$$

Since the last term is constant, $s _ { \Delta \mathrm { - K N N } } ( z )$ is a monotone transform of $\frac { \hat { p } _ { c } ( z ) } { \hat { p } _ { w } ( z ) }$ . Under the stated conditions on $k , N _ { c }$ and $N _ { w }$ , the empirical likelihoods converge to the true likelihoods $p _ { c }$ and $p _ { w } ,$ thus $s _ { \Delta \mathrm { - K N N } } ( z )$ is also Neyman–Pearson optimal.

Lemma 2. Let $s _ { 1 } ( { \pmb x } ) \in \mathbb { R }$ and $s _ { 2 } ( { \pmb x } ) \in \mathbb { R }$ be two selector scores. Assume both are Neyman–Pearson optimal; that $i s , s _ { 1 } ( { \pmb x } )$ is a monotone transform $o f p _ { c } ^ { ( 1 ) } / p _ { w } ^ { ( 1 ) }$ and $s _ { 2 } ( \pmb { x } )$ is a monotone transform of $p _ { c } ^ { ( 2 ) } / p _ { w } ^ { ( 2 ) }$ y scalar . $\lambda \in \mathbb { R } , t ( { \pmb x } ) = s _ { 1 } ( { \pmb x } ) + \lambda s _ { 2 } ( { \pmb x } )$ is a monotonic transformation of $p _ { c } ^ { ( 1 ) } ( p _ { c } ^ { ( 2 ) } ) ^ { \lambda } / p _ { w } ^ { ( 1 ) } ( p _ { w } ^ { ( 2 ) } ) ^ { \lambda }$

Proof. Let

$$
L _ {1} (\boldsymbol {x}) = \frac {p _ {c} ^ {(1)} (\boldsymbol {x})}{p _ {w} ^ {(1)} (\boldsymbol {x})}, \quad L _ {2} (\boldsymbol {x}) = \frac {p _ {c} ^ {(2)} (\boldsymbol {x})}{p _ {w} ^ {(2)} (\boldsymbol {x})}. \tag {13}
$$

Since each score is already a strictly monotone transform of $L ( { \pmb x } )$ , we are free to re-express the scores in any other convenient monotone scale without affecting relative ordering and thus Neyman–Pearson optimality. Without loss of generality, we will let $\begin{array} { r } { s _ { i } ( { \pmb x } ) = \log \frac { p _ { c } ^ { ( i ) } ( { \pmb x } ) } { p _ { w } ^ { ( i ) } ( { \pmb x } ) } , i = 1 , 2 } \end{array}$ , which are identical to the original scores in terms of sample acceptance and rejection patterns. Given $\lambda \in \mathbb { R }$ ,

$$
\begin{array}{l} t (\boldsymbol {x}) = s _ {1} (\boldsymbol {x}) + \lambda s _ {2} (\boldsymbol {x}) (14) \\ = \log L _ {1} (\boldsymbol {x}) + \lambda \log L _ {2} (\boldsymbol {x}) (15) \\ = \log (L _ {1} (\boldsymbol {x}) L _ {2} (\boldsymbol {x}) ^ {\lambda}) (16) \\ = \log \frac {p _ {c} ^ {(1)} (p _ {c} ^ {(2)}) ^ {\lambda}}{p _ {w} ^ {(1)} (p _ {w} ^ {(2)}) ^ {\lambda}} (17) \\ \end{array}
$$

In other words, $t ( { \pmb x } )$ is a monotone transform of the tilted likelihood ratio $p _ { c } ^ { ( 1 ) } ( p _ { c } ^ { ( 2 ) } ) ^ { \lambda } / p _ { w } ^ { ( 1 ) } ( p _ { w } ^ { ( 2 ) } ) ^ { \lambda }$ . □

Recall $\mathcal { H } _ { 0 }$ and $\mathcal { H } _ { 1 }$ represent the hypotheses that the classifier will make a correct and wrong prediction respectively. Assuming that the density of $\mathcal { H } _ { \mathrm { 0 } }$ takes the form of a tilted likelihood $p _ { c } ^ { ( 1 ) } ( p _ { c } ^ { ( 2 ) } ) ^ { \lambda } / Z _ { c }$ , where $Z _ { c }$ is a normalizing constant, and vice-versa for $\mathcal { H } _ { 1 }$ , then $t ( { \pmb x } )$ is Neyman–Pearson optimal by Corollary 1.

# C AVERAGE TOP k ∆-KNN MODIFICATION

Here we discuss how Neyman–Pearson optimality of the average log-distance formulation of ∆-KNN can hold as described in the main text. Recall that we let $\begin{array} { r } { D _ { \mathrm { K N N } } ( z ; A _ { c } ) = - \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \log ( u _ { i } ( z ) ) } \end{array}$ and vice-versa for $A _ { w }$ .

For concreteness, let us consider distances to the correct set; the derivation is identical for the wrong set. In the asymptotic limit where $N _ { c }$ is large and $k \ll N _ { c }$ , the ball centered at z that just encloses its kth nearest neighbor (i.e., with volume $V _ { d } ( u _ { k } ( z ) ) ^ { d } )$ is so small that the true density is essentially constant over it, so the radii of the first k neighbors are conditionally i.i.d. uniform in the ball.

As such, let us define the normalized variable

$$
U _ {i} = \left(\frac {u _ {i} (\boldsymbol {z})}{u _ {k} (\boldsymbol {z})}\right) ^ {d}, \quad i = 1,..., k. \tag {18}
$$

Note that $U _ { i } \in [ 0 , 1 ]$ for all i. Since the joint distribution of $U _ { i }$ depends only on $k ,$ we know from the i-th order statistics of k i.i.d. Uniform(0, 1) variables that each $U _ { i }$ is Beta-distributed, $U _ { i } \sim$ Beta $( i , k - i + 1 )$ , $0 \leq U _ { 1 } \leq \cdot \cdot \cdot \leq U _ { k } = 1$ . With some algebra, log $\begin{array} { r } { u _ { i } ( z ) = \frac { 1 } { d } \log U _ { i } + \log u _ { k } ( z ) } \end{array}$ . Then,

$$
\frac {1}{k} \sum_ {i = 1} ^ {k} \log (u _ {i} (\boldsymbol {z})) = \log u _ {k} (\boldsymbol {z}) + \frac {1}{k d} \sum_ {i = 1} ^ {k} \log U _ {i} \tag {19}
$$

The second term converges almost surely to

$$
\frac {1}{k d} \sum_ {i = 1} ^ {k} \log U _ {i} \rightarrow \frac {1}{d} \int_ {0} ^ {1} \log x \mathrm{d} x = - \frac {1}{d} \tag {20}
$$

as $k  \infty$ as it is a sum of k i.i.d. Uniform(0, 1) random variables. In other words, in the asymptotic limit the average log-distance is a monotone transform of the log-distance itself. By substituting Eq. 19 back into Eq. 11 for distances to both correct and wrong sets, the modified ∆-KNN formulation remains a monotone transform o ${ p _ { c } } / { p _ { w } }$ , thus suggesting Neyman-Pearson optimality under Corollary 1.

# D ALGORITHM PSEUDOCODE FOR PROPOSED SCORES

Pseudocode for Algorithms 1 (∆-MDS and its linear combination) and 2 (Scoring with ∆-KNN and its linear combination) are shown on the next page.

Algorithm 1 Scoring with ∆-MDS and its linear combination   
Input: Trained classifier f, feature extractor $\phi$ (typically penultimate or final layer of f), training set $\mathcal{D}_{\text{train}} = \{(\boldsymbol{x}_i, y_i)\}$ , test set $\mathcal{D}_{\text{test}} = \{\boldsymbol{x}_j\}$ , optional logit-based score function $s_{\text{logit}}(\boldsymbol{x})$ , combination weight $\lambda$ Output: Selector scores $s(\boldsymbol{x})$ for each $x \in D_{test}$ 1: Initialize $\mathcal{A}_c^{(i)} \leftarrow \emptyset$ and $\mathcal{A}_w^{(i)} \leftarrow \emptyset$ for i = 1 to K ▷ Class-wise correct and incorrect feature sets

2: for each $(\boldsymbol{x}, y)$ in $D_{train}$ do

3: $\hat{y} \leftarrow f(\boldsymbol{x})$ 4: $z \leftarrow \phi(\boldsymbol{x})$ 5: if $\hat{y} = y$ then

6: Add z to $\mathcal{A}_c^{(y)}$ 7: else

8: Add z to $\mathcal{A}_w^{(y)}$ 9: end if

10: end for

11: Compute $\{\mu_i^c, \Sigma^c\}_{i=1}^K$ and $\{\mu_i^w, \Sigma^w\}_{i=1}^K$ from $A_c$ and $A_w$ 12: for each x in $D_{test}$ do

13: $z \leftarrow \phi(\boldsymbol{x})$ 14: $d_c \leftarrow \max_i - (z - \mu_i^c)^{\top} (\Sigma^c)^{-1} (z - \mu_i^c)$ 15: $d_w \leftarrow \max_i - (z - \mu_i^w)^{\top} (\Sigma^w)^{-1} (z - \mu_i^w)$ 16: $s_{\Delta-MDS}(\boldsymbol{x}) \leftarrow d_c - d_w$ 17: if using linear combination then

18: $s(\boldsymbol{x}) \leftarrow s_{\Delta-MDS}(\boldsymbol{x}) + \lambda \cdot s_{\text{logit}}(\boldsymbol{x})$ 19: else

20: $s(\boldsymbol{x}) \leftarrow s_{\Delta-MDS}(\boldsymbol{x})$ 21: end if

22: end for

23: return $\{s(\boldsymbol{x})\}$ for each $x \in D_{test}$

Algorithm 2 Scoring with ∆-KNN and its linear combination   
Input: Trained classifier f, feature extractor $\phi$ (typically penultimate or final layer of f), training set $D_{train} = \{(x_i, y_i)\}$ , test set $D_{test} = \{x_j\}$ , number of neighbors k, optional logit-based score function $s_{\text{logit}}(x)$ , combination weight $\lambda$ Output: Selector scores $s(x)$ for each $x \in D_{test}$ 1: Initialize $A_c \leftarrow \emptyset$ , $A_w \leftarrow \emptyset$ $\triangleright$ Global sets of correct and incorrect features

2: for each $(x, y)$ in $D_{train}$ do

3: $\hat{y} \leftarrow f(x)$ 4: $z \leftarrow \phi(x)$ 5:    if $\hat{y} = y$ then

6:    Add z to $A_c$ 7:    else

8:    Add z to $A_w$ 9:    end if

10: end for

11: for each x in $D_{test}$ do

12: $z \leftarrow \phi(x)$ 13:    Compute $\{u_i\}_{i=1}^k \leftarrow \text{distances from } z \text{ to } k \text{ nearest neighbors in } A_c$ 14:    Compute $\{v_i\}_{i=1}^k \leftarrow \text{distances from } z \text{ to } k \text{ nearest neighbors in } A_w$ 15: $d_c \leftarrow -\frac{1}{k} \sum_{i=1}^k \log(u_i)$ 16: $d_w \leftarrow -\frac{1}{k} \sum_{i=1}^k \log(v_i)$ 17: $s_{\Delta-KNN}(x) \leftarrow d_c - d_w$ 18:    if using linear combination then

19: $s(x) \leftarrow s_{\Delta-KNN}(x) + \lambda \cdot s_{\text{logit}}(x)$ 20:    else

21: $s(x) \leftarrow s_{\Delta-KNN}(x)$ 22:    end if

23: end for

24: return $\{s(x)\}$ for each $x \in D_{test}$

# E EXPERIMENTAL DETAILS

Image Experiments. This section outlines the models and datasets used in the vision experiments in Section 5. For classifiers, we use the ViT-H/14 variant for DFN CLIP and the “Giant” variant of EVA, both with patch size 14. Pretrained weights are obtained from the OpenCLIP1 (Cherti et al., 2023) and timm2 libraries, respectively.

ImageNet and its covariate-shifted variants are downloaded from their respective open-source repositories3456789. For ImageNetV2, we use the MatchedFrequency test set to match the frequency distribution of the original ImageNet. For ImageNet-C, we evaluate using corruption level 5 to simulate the most challenging conditions. The dataset includes four corruption categories: 1) blur, 2) digital, 3) noise, and 4) weather. Each category contains multiple corruption types (e.g., Gaussian, impulse, and shot noise under the noise category). To ensure balanced evaluation, we first average results within each category, then average across the four categories to give equal weight to each corruption type.

For logit-based scores with CLIP, we are required to construct logits over class concepts by taking the dot product between the text embedding Tθ(c) of class concepts c and the image embedding, ϕ(x), where Tθ is the text encoder of CLIP. Given a class label y, we construct the class concept with the template “a real, high-quality, clear and clean photo of a {y}” when computing the confidence scores for logit-based selectors. We found this to improve scores slightly as opposed to using the default template in the original work (Radford et al., 2021). We attribute this to the hypothesis that CLIP should produce lower confidence scores when faced with covariate-shifted inputs, such as sketches or corrupted images, as the error rate on covariate shifts are much higher (Heng & Soh, 2025) than on Im-1K, which are generally clear and well-lit photographs.

Distance-based methods such as MDS, KNN, and our proposed variants compute distances in the feature space. For CLIP, we use the output of the final layer of the vision encoder; for EVA, we use the penultimate layer output.

The hyperparameters λ (for linear combinations) and k (for KNN-based scores) used in the experiments reported in Table 1 and Table 3 are summarized in Table 6. All experiments are conducted on a single NVIDIA A6000 GPU with 48GB of memory.

Language Experiments. For language experiments, we fine-tune a DistilBERT model on the Amazon Reviews dataset using the training pipeline provided in the official LISA repository10, with default hyperparameters. For distance-based selectors, we extract features from the penultimate layer of DistilBERT, consistent with the EVA setting. The values of λ and k used in Table 2 are also listed in Table 6. All language experiments are run on a single NVIDIA A6000 48GB GPU.

# F ADDITIONAL RESULTS

The following provides additional experimental results showing risk-coverage trade-offs, hyperparameter sensitivity, and performance results on both semantic and covariate shift.

$^{1}$ https://github.com/mlfoundations/open_clip $^{2}$ https://github.com/huggingface/pytorch-image-models $^{3}$ https://www.image-net.org/ $^{4}$ https://github.com/hendrycks/imagenet-r $^{5}$ https://github.com/hendrycks/natural-adv-examples $^{6}$ https://objectnet.dev/ $^{7}$ https://imagenetv2.org/ $^{8}$ https://github.com/HaohanWang/ImageNet-Sketch $^{9}$ https://github.com/hendrycks/robustness $^{10}$ https://github.com/huaxiuyao/LISA

Table 6: Values of λ and k for results reported in Sec. 5. 

<table><tr><td>Method</td><td> $\lambda$ </td><td> $k$ </td></tr><tr><td colspan="3">DFN CLIP</td></tr><tr><td>KNN</td><td>-</td><td>50</td></tr><tr><td> $\Delta$ -KNN</td><td>-</td><td>25</td></tr><tr><td> $\Delta$ -MDS-RLog</td><td>10000</td><td>-</td></tr><tr><td> $\Delta$ -KNN-RLog</td><td>10</td><td>25</td></tr><tr><td colspan="3">Eva</td></tr><tr><td>KNN</td><td>-</td><td>50</td></tr><tr><td> $\Delta$ -KNN</td><td>-</td><td>25</td></tr><tr><td> $\Delta$ -MDS-RLog</td><td>1000</td><td>-</td></tr><tr><td> $\Delta$ -KNN-RLog</td><td>0.5</td><td>25</td></tr><tr><td colspan="3">DistilBERT</td></tr><tr><td>KNN</td><td>-</td><td>50</td></tr><tr><td> $\Delta$ -KNN</td><td>-</td><td>25</td></tr><tr><td> $\Delta$ -MDS-RLog</td><td>1000</td><td>-</td></tr><tr><td> $\Delta$ -KNN-RLog</td><td>0.05</td><td>25</td></tr><tr><td> $\Delta$ -MDS-MSP</td><td>1000</td><td>-</td></tr><tr><td> $\Delta$ -KNN-MSP</td><td>0.5</td><td>25</td></tr></table>

![](images/fe04953c5c34af1109c51aeb084dd2e1ccfffb0238281a464995390417e4dc8b.jpg)

![](images/ea83c689308ebfd68548e201d775604dbc3977f1fa1d79340eb6d605d727eb2f.jpg)

<details>
<summary>line</summary>

| λ   | AURC  | NAURC |
| --- | ----- | ----- |
| 1   | 4.38  | 0.215 |
| 2   | 4.23  | 0.205 |
| 5   | 4.03  | 0.190 |
| 10  | 4.00  | 0.190 |
| 20  | 4.12  | 0.195 |
</details>

![](images/8ad152c89948bb211eba729a8a7cf271463a1f2d21e8f23dec92c88f80c1d11d.jpg)

<details>
<summary>line</summary>

| k (log scale) | AURC  | NAURC |
| ------------- | ----- | ----- |
| 5             | 6.1   | 0.325 |
| 10            | 4.6   | 0.225 |
| 25            | 4.0   | 0.200 |
| 50            | 3.9   | 0.195 |
| 100           | 4.0   | 0.195 |
</details>

Figure 2: Hyperparameter sensitivity plots for λ and k for ∆-KNN-RLog with DFN on ImageNet1K. The results for this combination are not highly sensitive to λ, while for k results plateau at around k = 25.

![](images/4b7ec28b2fff69ec5d771f3302a4c38e10059c1d2173248894de2430bac32406.jpg)

<details>
<summary>line</summary>

| Method       | Im-1K | Im-V2 | Im-S | Im-C | Im-4 |
| ------------ | ----- | ----- | ---- | ---- | ---- |
| MCM/MSP      | 0.05  | 0.08  | 0.12 | 0.03 | 0.06 |
| MaxLogit     | 0.07  | 0.10  | 0.15 | 0.05 | 0.09 |
| Energy       | 0.15  | 0.18  | 0.25 | 0.12 | 0.22 |
| MDS          | 0.12  | 0.16  | 0.28 | 0.14 | 0.24 |
| KNN          | 0.08  | 0.14  | 0.26 | 0.13 | 0.23 |
| RLog         | 0.06  | 0.12  | 0.24 | 0.11 | 0.21 |
| Δ-MDS        | 0.04  | 0.10  | 0.22 | 0.10 | 0.20 |
| Δ-KNN        | 0.03  | 0.08  | 0.20 | 0.09 | 0.19 |
| Δ-MDS-RLog   | 0.02  | 0.06  | 0.18 | 0.08 | 0.18 |
| Δ-KNN-RLog   | 0.01  | 0.04  | 0.16 | 0.07 | 0.17 |
</details>

Figure 3: Risk-coverage curves of various selector methods for CLIP (top row) and EVA (bottom row). Our proposed methods consistently achieve the best risk-coverage tradeoff and remain stable at low coverage levels.

Table 7: Additional results under covariate shift using ResNet50 trained on ImageNet1K from the official PyTorch repository. Consistent with the findings on larger models in the main text, our proposed methods outperform all baselines. These results affirm that the gains stem from algorithmic improvements in selective classification, rather than from large-scale pretraining potentially mitigating distribution shifts. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Im-1K</td><td colspan="2">Im-V2</td><td colspan="2">Im-S</td><td colspan="2">Im-C</td><td colspan="2">Avg (1K)</td></tr><tr><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td></tr><tr><td>MSP</td><td>9.67</td><td>0.248</td><td>10.32</td><td>0.257</td><td>23.4</td><td>0.196</td><td>22.6</td><td>0.203</td><td>16.5</td><td>0.226</td></tr><tr><td>MaxLogit</td><td>12.7</td><td>0.379</td><td>13.5</td><td>0.394</td><td>25.3</td><td>0.248</td><td>24.5</td><td>0.260</td><td>19.0</td><td>0.320</td></tr><tr><td>Energy</td><td>16.3</td><td>0.539</td><td>17.6</td><td>0.567</td><td>28.1</td><td>0.328</td><td>27.8</td><td>0.358</td><td>22.5</td><td>0.448</td></tr><tr><td>MDS</td><td>21.2</td><td>0.752</td><td>21.6</td><td>0.736</td><td>58.4</td><td>1.19</td><td>58.3</td><td>1.23</td><td>39.9</td><td>0.977</td></tr><tr><td>KNN</td><td>28.8</td><td>1.08</td><td>30.4</td><td>1.11</td><td>40.2</td><td>0.673</td><td>46.2</td><td>0.900</td><td>36.4</td><td>0.940</td></tr><tr><td>RLog</td><td>8.61</td><td>0.201</td><td>9.14</td><td>0.207</td><td>24.5</td><td>0.225</td><td>22.4</td><td>0.193</td><td>16.2</td><td>0.207</td></tr><tr><td>SIRC</td><td>18.9</td><td>0.648</td><td>19.6</td><td>0.652</td><td>46.2</td><td>0.844</td><td>40.5</td><td>0.729</td><td>31.3</td><td>0.718</td></tr><tr><td>Δ-MDS</td><td>13.0</td><td>0.392</td><td>13.7</td><td>0.399</td><td>30.3</td><td>0.390</td><td>27.3</td><td>0.338</td><td>21.1</td><td>0.380</td></tr><tr><td>Δ-KNN</td><td>15.3</td><td>0.493</td><td>16.2</td><td>0.509</td><td>36.0</td><td>0.552</td><td>33.9</td><td>0.529</td><td>25.3</td><td>0.521</td></tr><tr><td>Δ-MDS-RLog</td><td>8.33</td><td>0.189</td><td>8.86</td><td>0.195</td><td>24.1</td><td>0.215</td><td>21.9</td><td>0.180</td><td>15.8</td><td>0.195</td></tr><tr><td>Δ-KNN-RLog</td><td>8.48</td><td>0.195</td><td>9.02</td><td>0.202</td><td>24.3</td><td>0.221</td><td>22.2</td><td>0.189</td><td>16.0</td><td>0.202</td></tr></table>

Table 8: Additional results on semantic shift datasets, namely ImageNet-O, iNaturalist, SUN and Places. The ID distribution is ImageNet-1K. 

<table><tr><td></td><td colspan="2">ImageNet-O</td><td colspan="2">iNaturalist</td><td colspan="2">SUN</td><td colspan="2">Places</td><td colspan="2">Avg</td></tr><tr><td></td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td><td>A</td><td>N</td></tr><tr><td colspan="11">DFN CLIP</td></tr><tr><td>MSP</td><td>9.70</td><td>0.459</td><td>11.7</td><td>0.273</td><td>12.2</td><td>0.293</td><td>12.8</td><td>0.319</td><td>11.6</td><td>0.336</td></tr><tr><td>MDS</td><td>11.8</td><td>0.584</td><td>14.1</td><td>0.370</td><td>14.8</td><td>0.398</td><td>15.2</td><td>0.414</td><td>14.0</td><td>0.442</td></tr><tr><td>MaxLogit</td><td>9.75</td><td>0.462</td><td>11.9</td><td>0.282</td><td>12.6</td><td>0.311</td><td>13.4</td><td>0.343</td><td>11.9</td><td>0.350</td></tr><tr><td>Energy</td><td>17.7</td><td>0.930</td><td>28.1</td><td>0.935</td><td>33.2</td><td>1.14</td><td>32.9</td><td>1.13</td><td>28.0</td><td>1.03</td></tr><tr><td>KNN</td><td>11.3</td><td>0.552</td><td>13.2</td><td>0.334</td><td>13.9</td><td>0.360</td><td>13.9</td><td>0.362</td><td>13.1</td><td>0.402</td></tr><tr><td>RLog</td><td>6.21</td><td>0.253</td><td>10.0</td><td>0.205</td><td>11.2</td><td>0.254</td><td>11.5</td><td>0.267</td><td>9.73</td><td>0.245</td></tr><tr><td>SIRC</td><td>18.2</td><td>0.958</td><td>32.7</td><td>1.12</td><td>30.2</td><td>1.02</td><td>32.3</td><td>1.11</td><td>28.4</td><td>1.05</td></tr><tr><td>Δ-KNN</td><td>5.93</td><td>0.237</td><td>9.42</td><td>0.181</td><td>10.6</td><td>0.228</td><td>10.4</td><td>0.220</td><td>9.09</td><td>0.217</td></tr><tr><td>Δ-KNN-RLog</td><td>5.20</td><td>0.194</td><td>8.60</td><td>0.148</td><td>9.69</td><td>0.192</td><td>9.76</td><td>0.195</td><td>8.31</td><td>0.182</td></tr><tr><td colspan="11">EVA</td></tr><tr><td>MSP</td><td>4.94</td><td>0.285</td><td>8.46</td><td>0.214</td><td>9.29</td><td>0.251</td><td>10.1</td><td>0.289</td><td>8.20</td><td>0.260</td></tr><tr><td>MDS</td><td>4.58</td><td>0.258</td><td>6.64</td><td>0.133</td><td>7.41</td><td>0.167</td><td>7.78</td><td>0.184</td><td>6.60</td><td>0.186</td></tr><tr><td>MaxLogit</td><td>6.37</td><td>0.392</td><td>10.8</td><td>0.319</td><td>11.5</td><td>0.349</td><td>12.6</td><td>0.399</td><td>10.3</td><td>0.365</td></tr><tr><td>Energy</td><td>8.80</td><td>0.572</td><td>14.5</td><td>0.487</td><td>14.5</td><td>0.484</td><td>15.8</td><td>0.545</td><td>13.4</td><td>0.522</td></tr><tr><td>KNN</td><td>4.63</td><td>0.262</td><td>6.70</td><td>0.135</td><td>7.43</td><td>0.168</td><td>7.70</td><td>0.180</td><td>6.62</td><td>0.186</td></tr><tr><td>RLog</td><td>3.82</td><td>0.201</td><td>6.92</td><td>0.145</td><td>8.25</td><td>0.205</td><td>8.21</td><td>0.203</td><td>6.80</td><td>0.189</td></tr><tr><td>SIRC</td><td>5.41</td><td>0.320</td><td>9.14</td><td>0.245</td><td>10.2</td><td>0.293</td><td>11.1</td><td>0.331</td><td>8.96</td><td>0.297</td></tr><tr><td>Δ-MDS</td><td>3.49</td><td>0.177</td><td>5.63</td><td>0.087</td><td>6.72</td><td>0.136</td><td>6.95</td><td>0.146</td><td>5.70</td><td>0.137</td></tr><tr><td>Δ-MDS-RLog</td><td>3.37</td><td>0.202</td><td>5.64</td><td>0.130</td><td>6.90</td><td>0.187</td><td>7.08</td><td>0.186</td><td>5.75</td><td>0.176</td></tr></table>