# HIDDEN BREAKTHROUGHSIN LANGUAGE MODEL TRAINING

# Sara Kangaslahti

Harvard University

sarakangaslahti@g.harvard.edu

# Elan Rosenfeld

Google Research

elanr@google.com

# Naomi Saphra

Harvard University

nsaphra@fas.harvard.edu

# ABSTRACT

Loss curves are smooth during most of model training, so visible discontinuities stand out as possible conceptual breakthroughs. These breakthroughs enable a deeper understanding of the model’s concept structure, but only when they are properly identified. This paper argues that similar breakthroughs occur frequently throughout training, but they are obscured by a loss metric that collapses all variation into a single scalar. To find these hidden transitions, we introduce POLCA, a method for decomposing changes in loss along arbitrary bases of the low-rank training subspace. We use our method to identify clusters of samples that share similar changes in loss during training, disaggregating the overall loss into that of smaller groups of conceptually similar data. We validate our method on synthetic arithmetic and English language modeling, showing that POLCA recovers clusters that represent interpretable breakthroughs in the model’s capabilities. We demonstrate the promise of these hidden breakthroughs as a tool for unsupervised interpretability.

# 1 INTRODUCTION

As large language models train, various internal structures develop during abrupt breakthroughs. These sudden drops in loss reveal the formation of mechanisms for in-context learning (Olsson et al., 2022b), natural language grammar (Chen et al., 2024a), hierarchical generalization (Murty et al., 2023), and many other concepts (McGrath et al., 2022; Lovering et al., 2022; Power et al., 2022; Abbe et al., 2021). However, the loss curve as a whole remains stubbornly smooth. As a result, these momentary conceptual breakthroughs are treated as isolated curiosities, while the majority of training behavior is considered predictable.

These breakthroughs—often colloquially termed phase transitions (Olsson et al., 2022a; Chen et al., 2024a; Murty et al., 2023)—are extremely consequential for our understanding of neural networks. Phase transitions represent critical periods of learning, so they offer key insights for training and optimization. For instance, introducing noisy data or changing the optimizer during a phase transition can significantly reduce the downstream performance of a model (Achille et al., 2017; Chen et al., 2024a). Their timing is used as evidence the role of specific mechanisms in model capabilities (Zhong et al., 2023; Olsson et al., 2022b; Chen et al., 2024a). If we could find more of these momentary training events, it could expand our understanding even further.

Prior work identifies breakthroughs through a top-down approach by measuring the training dynamics of a predefined concept or skill and searching for sudden changes. We instead propose a bottom-up unsupervised method for finding breakthroughs by grouping data points that have similar training behavior. This data-centric approach can be used to inform optimization choices such as data selection or learning rate scheduling. Like other bottom-up interpretability methods such as SAEs, PCA, and transcoders, our method seeks concepts that are used naturally by the model, rather than imposing an assumed structure onto learning and representation.

This work shows that in fact, a model undergoes many breakthroughs during training, but most are concealed when averaging all data into a single loss curve. Instead of averaging, we divide up the loss curve in two different ways to find hidden breakthroughs. First, we disaggregate the aggregate loss into losses on individual examples. By clustering the individual loss curves, we identify subsets of data that experience synchronized changes in loss, implying that they rely on the same conceptual breakthrough. However, any individual example might benefit from multiple breakthroughs; such an example may undergo changes synchronized with different data subsets at different times. Furthermore, distinct concepts might appear simultaneously, erroneously merging their data clusters. To recover the underlying learned concepts, we may have to identify multiple separate breakthroughs which affect the loss curve of a single example.

![](images/b9fe1a98bb7ed3507c351bd2ee439711af296031b13a7619d1e193f94fac490a.jpg)

<details>
<summary>text_image</summary>

θ·b₂
θ·b₁
</details>

![](images/6ae21aa6eb3f24465b17e797b937d8769cb8aa2710459f7295f8f7499db6a719.jpg)

<details>
<summary>line</summary>

| t    | Average |
| ---- | ------- |
| t1   | 1       |
| t2   | 0.5     |
| t3   | 0.2     |
| t4   | 0.1     |
| t5   | 0.05    |
| t6   | 0.02    |
| t7   | 0.01    |
| t8   | 0.005   |
| t9   | 0.002   |
| t10  | 0.001   |
</details>

Figure 1: A smooth loss function may change sharply for a particular direction or data subset. POLCA works by decomposing and disaggregating the loss to discover these sharp changes. $L e f t { \mathrm { : } }$ Loss $L ( x ; \theta )$ changes as the parameter setting θ moves in a low-rank training subspace. The loss is sigmoidal on each axis, with differently timed inflections along basis vectors $6 +$ and $b _ { 2 }$ . These breakthroughs disappear in the smooth sum of the sigmoids which represents the exact loss. $R i g h t \cdot$ The average of sigmoidal functions—including loss along basis vectors $b _ { 1 }$ and b2—elides individual breakthroughs. The more differently-timed breakthroughs underlie the loss, the more hidden each breakthrough is.

To disentangle these effects for a single sample, we separate the optimization space into specific gradient directions. When the loss changes during training, it is the result of movement across all parameters in a high-dimensional space. We decompose this loss change from an exact trajectory in the full-rank parameter space into a collection of movements along each dimension. By analyzing these loss curves along specific basis vectors, we identify conceptual breakthroughs that rely on particular directions of movement. The latter analysis permits further granularity in clustering data, as final performance on an individual example may rely on multiple conceptual breakthroughs, each corresponding to a particular linear direction in training. In summary:

• We introduce a modified form of Loss Change Allocation (Lan et al., 2020) called Projection Oriented Loss Change Allocation (POLCA) to measure changes in loss due to parameter adjustments in arbitrary directions during training (Section 3.2).   
• We show that some learned concepts can be identified by clustering exact loss, while others cannot (Section 4.2). Using POLCA, we extend our cluster analysis to identify these hidden conceptual breakthroughs obscured in the exact loss curves. We automatically identify specific concepts learned during breakthroughs in both synthetic (Section 4) and natural language settings (Section 5).

# 2 BACKGROUND: HOW MUCH CAN WE LEARN FROM LEARNING DYNAMICS?

Various loss breakthroughs have been interpreted as learning specific concepts. But why expect additional interpretable breakthroughs to underlie periods of undifferentiated, gradual model improvement? Our approach is justified by the nature of the loss surface’s complexity, illustrated by Figure 1, in which a smooth curve emerges by eliding breakthroughs from each dimension.

Why expect multiple breakthroughs? A very early phase transition is to be expected early in training after a brief memorization stage (Shwartz-Ziv & Tishby, 2017). In this sense, the most celebrated breakthroughs—those signaling the formation of induction heads or arithmetic grokking—are not the first drops in their training curves. Some breakthroughs even depend on earlier breakthroughs, as observed in synthetic tasks (Abbe et al., 2021) and in grammar acquisition (Chen et al., 2024a). If one concept depends on another, each must appear at a different timestep, requiring multiple breakthroughs. Furthermore, as shown by Saxe et al. (2019), summing many phase transitions can result in a smooth curve, supporting our hypothesis that these breakthroughs can appear in stable regions of the loss curve.

Multiple breakthroughs can also come from differences in gradient scale along different directions. Ma et al. (2022) even attributed the early edge-of-stability phase transition (Jastrze¸bski et al., 2020; Cohen et al., 2022) to multiscale structure of the loss surface and, furthermore, noted that this multiscale structure emerges at the range where models become singular: their loss lacks a quadratic approximation in terms of model parameters, creating conditions for breakthroughs under Singular Learning Theory (Watanabe, 2010; Wei et al., 2020; Wang et al., 2024). They argued that this structure is the product of both nonuniform data and nonconvex objectives, respectively justifying the disaggregation and decomposition which we apply to interpret training dynamics.

Why disaggregate the aggregate loss? We track learning on training datapoints and subpopulations, rather than the whole training set, because relevant skills can be acquired at different rates (Arora & Goyal, 2023; Chen et al., 2024b). Individual samples thus exhibit changes in loss out of line with the monotonic average trend (Xia et al., 2023; Rosenfeld & Risteski, 2024). In full-batch gradient descent, Cohen et al. (2022) identified non-monotonicity arising from oscillation about the maximum Hessian eigenvector. Rosenfeld & Risteski (2024) demonstrated that these oscillations occur across different axes for different samples and identified the primary cause: surprisingly human-interpretable semantic features. Even when the loss seems stable, performance can oscillate on edge cases until the model develops relevant capabilities (Qin et al., 2024; Bhaskar et al., 2024). We hypothesize that oscillation represents competing skills that are relevant to different subsets of data. To test this hypothesis, and to interpret the meaning of these directions, we disaggregate the loss into clusters with similar dynamics.

Why decompose the exact loss? Michaud et al. (2024) analyzed the scaling behavior of models with respect to individual tokens and identified a limitation of token-wise analysis of breakthroughs, which they called polygenic scaling effects—samples which combine multiple skills and therefore exhibit breakthroughs at multiple scales. Our POLCA decomposition directly addresses this limitation by decomposing the loss of each token along multiple basis vectors. If we assume that a specific skill is enabled by movement along that skill’s basis vector, then the loss change attributed to movement along that vector will accelerate at the moment the skill is acquired, for every sample that requires that skill. In this manner, the sample transitions from early to late dynamics through a basis-specific loss breakthrough. In other words, by monitoring changes in directions corresponding to specific skills, we support the speculation of Nanda et al. (2023) that “phase transitions are everywhere.”

Why is linear decomposition sufficient? In practice, a conceptual breakthrough might not occur in a single direction that persists throughout training. However, there is abundant evidence that linear bases of the low-rank (Gur-Ari et al., 2018) training subspace are conceptually meaningful. In the late stages of training, loss is convex on the line connecting a pair of checkpoints (Frankle et al., 2020) if those checkpoints express similar capabilities (Juneja et al., 2023) and mechanisms (Lubana et al., 2023). If a pair of high-dimensional models lack this linear connection, they still connect nonlinearly (Draxler et al., 2019); however, while parameter settings sampled from their linear connections improve broadly on the capabilities of the original models, those sampled from their nonlinear connections are less robust than the originals (Juneja et al., 2023, ref Appendix G). These observations suggest that linear decomposition should preserve meaningful conceptual features on the loss surface, and our experiments show that the resulting directions are interpretable in practice.

# 3 METHODS

The key to our approach is the separate consideration of each example’s datapoint loss changes throughout training. We contrast this individualized metric with aggregated loss across an entire dataset. Using the datapoint loss, we can cluster individual example x on the basis of its loss $L ( x ; \theta _ { t } )$ , change in loss $L ( x ; \theta _ { t } ) - L ( x ; \theta _ { t - 1 } )$ , or magnitude of change $\check { \vert L ( x ; \theta _ { t } ) - L ( x ; \theta _ { t - 1 } ) \vert }$ | during training.

We next decompose the loss itself into specific directions in the weight space, motivated by several considerations: First, while we have moved from an aggregated loss metric to a more granular datapoint loss metric, we are still only considering breakthroughs that are general enough to be perceived in loss curves. Second, an individual datapoint may benefit from a variety of conceptual breakthroughs, but will not be clustered on the breakthroughs individually. Finally, once we have identified a subset of the data as benefiting from a particular conceptual breakthrough, decomposing into individual weight directions allows us to locate where in the weights the breakthrough occurs and to thereby identify the mechanism involved.

When we break the exact loss curve down by directional movement during training, the resulting decomposed loss will reveal breakthroughs that are specific to a given direction. Our procedure follows three steps: (1) select a basis, (2) decompose the loss along that basis to highlight particular learning events, (3) cluster datapoints according to their shared learning events.

# 3.1 FINDING THE BASIS

Algorithm 1 Finding the decomposed optimization basis   
input: Training set X, Model checkpoints $\{\theta_{t}\}_{t=1}^{T}$ . $B_{0} \leftarrow \emptyset \in R^{d \times 0}.$ for $t = 1 \ldots T$ do $\Pi_{\perp} \leftarrow I - B_{t-1}(B_{t-1}^{\top}B_{t-1})^{-1}B_{t-1}^{\top}$ $\mathcal{H} \leftarrow \nabla_{\theta}^{2}\mathcal{L}(X,\theta).$ Define $B^{+} \in R^{d \times k}$ as the top k eigenvectors of $\Pi_{\perp}H$ (e.g., via the Lanczos method). $B_{t} \leftarrow [B_{t-1}, B^{+}]$ .
end for
return $B_{T}$

To decompose the loss, we first require an interpretable orthogonal basis. We efficiently compute the eigenvectors of the Hessian matrix using CoLA (Potapczynski et al., 2023) to construct a restricted training subspace. We expect this basis to be interpretable because each basis vector captures a large gradient covariance and therefore represents a potential decision boundary. We select this basis for interpretability, but our approach can use an arbitrary choice of basis—for example, one which targets a particular use case or efficiently leverages optimizer preconditioner values.

The basis is constructed as shown in Algorithm 1. Given T intermediate training checkpoints and a number k of eigenvectors to compute at each checkpoint, we seek a low rank T k-dimensional subspace which captures most of the movement during optimization (Gur-Ari et al., 2018). We construct this basis iteratively, starting with $B _ { 0 } = \varnothing ;$ : at each checkpoint t, we take checkpoint weights $\theta _ { t } \in \mathbb { R } ^ { d }$ and project their loss Hessian onto the nullspace of $B \in \mathbb { R } ^ { d \times ( t - 1 ) k }$ . From the resulting projection, we append the top k eigenvectors to $B _ { t - 1 }$ . We compute the eigenvectors using Hessian-vector products Golmant et al. (2018) to avoid explicitly constructing the full Hessian matrix. The resulting basis is designed to include directions of highest curvature at each checkpoint so that it will capture synchronized loss behavior throughout training. Note that the very top eigenvectors are likely to reflect local oscillation, rather than conceptually meaningful long-term movement (Song et al., 2024), but as we continue to add to the low rank basis, we include more directions of long-term stable movement. We discard the oscillatory directions which do not provide an overall decrease in loss over the course of training according to POLCA by removing the directions with an increase in the mean projected loss from checkpoint 1 to T . In this manner, we first construct a basis based on local information, then filter out directions that do not represent long-term movement. This construction finds local high-curvature directions that may be important for breakthroughs in the intermediate stages of training while ensuring that the basis does not overfit to local oscillations.

# 3.2 DECOMPOSING THE LOSS WITH POLCA

To decompose the loss along our basis, we propose a modified version of Loss Change Allocation (LCA) (Lan et al., 2020) which we call Projection-Oriented Loss Change Allocation (POLCA). LCA is a tool for analyzing changes in aggregated loss on dataset X between two checkpoints. The output of LCA is the empirical loss change between a pair of checkpoints which can be attributed to the motion of each individual weight unit. Given two consecutive checkpoints with parameters $\theta _ { t }$ and $\theta _ { t + 1 } , \mathrm { L C A }$ reformulates the change in loss as its first-order Taylor approximation, a sum of the loss changes attributed to the movement of each individual model parameter $\theta$ :

$$
L (X; \theta_ {t + 1}) - L (X; \theta_ {t}) \approx \sum_ {j = 0} ^ {d} (\nabla_ {\theta} L (X; \theta_ {t})) ^ {(j)} (\theta_ {t + 1} ^ {(j)} - \theta_ {t} ^ {(j)}) = \sum_ {j = 0} ^ {d} L C A (X; \theta^ {(j)}) \tag {1}
$$

The POLCA decomposition differs from LCA in three key ways. First, we do not restrict each direction to correspond to a single unit $\theta$ , instead permitting an arbitrary orthonormal basis vector $b \in B _ { T }$ to replace the axis-aligned basis vectors in LCA; we project onto this basis vector using the dot product $\langle b , \cdot \rangle$ . Second, we are interested in changes in the loss on each individual example $x \in X$ , not the entire dataset X. These first two modifications provide the first-order POLCA decomposition:

$$
\begin{array}{l} L (X; \theta_ {t + 1}) - L (X; \theta_ {t}) = \sum_ {x \in X} L (x; \theta_ {t + 1}) - L (x; \theta_ {t}) \\ \approx \sum_ {x \in X} \sum_ {b \in B _ {T}} \langle b, \nabla_ {\theta} L (x; \theta_ {t}) \rangle \langle b, \theta_ {t + 1} - \theta_ {t} \rangle \tag {2} \\ \end{array}
$$

The third key difference is that we use a second-order approximation because this basis is constructed explicitly from the Hessian eigenvectors. To understand why this choice of basis warrants a secondorder approximation, recall that each basis vector b is an eigenvector of the Hessian matrix $\mathcal { H } _ { t ^ { \prime } } ( X )$ at some timestep $t ^ { \prime } ,$ where b is chosen because it has the largest eigenvalue $\lambda _ { t ^ { \prime } } ( X , b )$ over the whole dataset. If we assume that the top eigenvectors of the aggregate Hessian maintain high curvature at other points in training and on individual datapoints, then the scaling factor in the second-order Taylor term will be very large even at the datapoint level. Limiting the approximation to only the first order term gives poor guarantees on error, as the second-order term could be expected to dominate. Although empirically the difference between the first and second-order values is small (see Appendix I), we nonetheless guarantee a better estimate due to lower Lagrange error bounds by computing the second-order approximation below.

Exact computation of the second-order term would be intractable, requiring computation of the top eigenvalues/vectors for each individual datapoint x. Instead, we can approximate it by substituting the true eigenvalue, denoted $\lambda _ { t } ( X , b ) : = b ^ { \top } \dot { \mathcal { H } } _ { t } ( X ) b$ , with the curvature of the individual loss in the direction $b , \mathbf { i . e . } ~ \lambda _ { t } ( x , b ) = b ^ { \top } \mathcal { H } _ { t } ( x ) b .$ If the aggregate Hessian eigenvector b is close to the span of the top eigenvectors of the datapoint-specific Hessian for x, this provides a reasonable estimate while reducing calculation to a single Hessian-vector product per eigenvector. We therefore approximate the basis projection of the datapoint Hessian $h ( x , b , \theta _ { t } )$ as derived in Appendix C.

$$
h (x, b, \theta_ {t}) = \frac {\lambda_ {t} (x , b)}{2} \left\langle \theta_ {t + 1} - \theta_ {t}, b \right\rangle^ {2} \tag {3}
$$

$$
\approx \frac {\lambda_ {t} (X , b)}{2} \cdot \left\langle \theta_ {t + 1} - \theta_ {t}, b \right\rangle^ {2} \times \frac {\left\langle L (x ; \theta_ {t + 1}) - L (x ; \theta_ {t}) , b \right\rangle}{\left\langle L (X ; \theta_ {t + 1}) - L (X ; \theta_ {t}) , b \right\rangle} \tag {4}
$$

$$
= \tilde {h} (x, b, \theta_ {t}) \tag {5}
$$

Equipped with this second-order approximation of the datapoint Hessian’s projection onto our basis, we account for the high curvature and possible domination by the higher order term by modifying Equation 2 into the second-order Taylor expansion using the approximation from Equation 5. We can compute this second-order term with limited additional computational complexity by keeping track of the eigenvalues for each Hessian eigenvector and the aggregate gradient at each checkpoint.

$$
L (X; \theta_ {t + 1}) - L (X; \theta_ {t}) \approx \sum_ {x \in X} \sum_ {b \in B _ {T}} \langle b, \nabla_ {\theta} L (x; \theta_ {t}) \rangle \langle b, \theta_ {t + 1} - \theta_ {t} \rangle + \tilde {h} (x, b, \theta_ {t}) \tag {6}
$$

$$
= \sum_ {x \in X} \sum_ {b \in B _ {T}} P O L C A (x, b; \theta_ {t}) \tag {7}
$$

# 3.3 CLUSTERING THE LOSS

POLCA, above, provides curves that show how a decomposed loss changes with respect to each training example. We assume that if several examples show similarly timed loss changes in the same direction, they likely rely on the same conceptual breakthroughs or learning events; therefore, they are likely to share a required skill, or specific capability needed for a given task. We cluster POLCA training histories to recover these skill groups. For each datapoint x, we compute the total cumulative change in loss along each basis vector b by summing over the previous POLCA values. We denote this sum the projected loss $L _ { b } ( x , \theta _ { t } )$ .

$$
L _ {b} (x, \theta_ {t}) = \sum_ {i = 0} ^ {t - 1} P O L C A (x, b; \theta_ {i}) \tag {8}
$$

We obtain 1d projected loss trajectories for breakthrough clustering by computing $L _ { b } ( x , \theta _ { t } )$ at every time t. We cluster trajectories using Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) Campello et al. (2013) because it distinguishes cluster outliers and discovers clusters with variable density (i.e., similarly shaped curves that lay far apart in their metrics). We cluster the trajectories for each basis vector separately to ensure that the clustering can capture multiple skills per token.

# 3.4 DEFINING AND IDENTIFYING HIDDEN BREAKTHROUGHS

We use POLCA to recover hidden breakthroughs in training, so we must quantify whether clustered trajectories are distinguished by breakthroughs. We use the formulation defined by Chen et al. (2024b) to compute the start of a breakthrough in a given function f for a given datapoint x and basis vector b:

$$
\operatorname{break} (f, x, \Delta) = \underset {t} {\arg \max} [ f (x, t + \Delta) - f (x, t) ] - [ f (x, t) - f (x, t - \Delta) \tag {9}
$$

Here, break $( f , x , \Delta )$ approximates the maximum point of acceleration of x in $f . ~ f$ can be either the projected loss $L _ { b }$ for a given basis vector b or the exact loss L. We define a hidden breakthrough as a breakthrough that occurs in the flat region of the exact loss curve. That is, if we set a threshold τ beyond which the exact loss curve is flat, then a given set $X ^ { \prime } \subseteq X$ has a hidden breakthrough in a metric f if the expected value of the start of breakthroughs in $X ^ { \prime }$ is greater than τ :

$$
\text {hidden} (f, X ^ {\prime}, \Delta) = \mathbf {1} \left\{\mathbb {E} _ {x \in X ^ {\prime}} \left[ \arg \max _ {t} [ f (x, t + \Delta) - f (x, t) ] - [ f (x, t) - f (x, t - \Delta) ] \right] > \tau \right\} \tag {10}
$$

# 4 ARITHMETIC LANGUAGE MODELING

We validate our POLCA clustering method in a synthetic setting using an arithmetic addition task. Our clusters reflect categorical concepts within the data, even when those concepts are not discoverable by clustering directly on loss curves. Specifically, if we cluster on exact loss curves, we recover digit positions—but if we instead cluster on POLCA curves, we also recover the skill of “carrying” a digit.

# 4.1 EXPERIMENTS

Data Our synthetic experiments use data from the arithmetic addition setting in Chen et al. (2024b), where the model is trained to compute the sum of two 3-digit numbers. This setting has 4 skills corresponding to each of the digits in the output sum. Note that the digit in the 1000s place is always a <0> or <1> token since the two input summants are 3 digits long. As shown in Appendix Figure 6 and Chen et al. (2024b), the skills corresponding to the digits have different loss curves, so we will easily recover the digit skill categories by clustering exact loss curves.

Unlike our source material, we also consider an additional skill: arithmetic carries to the output token (Figure 2). This skill corresponds to the case where instead of simply adding the two tokens at the corresponding digit in the input, the model must also carry a 1 from the previous digit. Digit-specific addition skills lead to clearly distinguishable exact loss curves, whereas carrying skills do not (Appendix Figure 7)—but carries will become clear on our decomposed gradient basis. We provide additional skill and labeling details in Appendix E.1.

![](images/be71dd5e13140c4527a4d20c3616000828ad4b41eb21d52bda75c8fb8bbb39a5.jpg)

<details>
<summary>other</summary>

| Output token | 1 | 4 | 3 | 3 |
|---|---|---|---|---|
| Digit skill | 1000s | 100s | 10s | 1s |
| Carry skill | Carry | No carry | Carry | No carry |
</details>

![](images/5262990d155d16db2e14ff6f9cc9a0c65e09513703f9182b0ff23d182b6624c6.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 |
| --------- | --------- | --------- |
| 0         | 3.0       | 3.0       |
| 500       | 0.5       | 2.5       |
| 1000      | 0.1       | 1.0       |
| 1500      | 0.05      | 0.5       |
| 2000      | 0.02      | 0.2       |
| 2500      | 0.01      | 0.1       |
| 3000      | 0.01      | 0.05      |
| 3500      | 0.01      | 0.02      |
| 4000      | 0.01      | 0.01      |
</details>

(a) Mean exact loss trajectories per cluster.

![](images/ffbe78d9cfd45f7cd58f432261bf89350c31fb1afb7557de44560ca263f29424.jpg)

<details>
<summary>bar</summary>

| Cluster | Digit skill | Carrying skill |
| ------- | ----------- | -------------- |
| 1       | 1000s       | 100s           |
| 1       | 100s        | 100s           |
| 1       | 10s         | 10s            |
| 1       | 1s          | No carry       |
| 2       | 1000s       | 100s           |
| 2       | 100s        | 100s           |
| 2       | 10s         | 10s            |
| 2       | 1s          | No carry       |
</details>

(b) Arithmetic skill composition of the clusters.   
Figure 2: Diagram of arithmetic addition task. An example of 3-digit addition, labeled with the skills required for each of the output tokens.   
Figure 3: Exact loss trajectory clustering on the arithmetic task. We use HDBSCAN to cluster the exact loss trajectories. This approach, unlike our POLCA clustering method, fails to recover clusters associated with the carrying skill (the maximum fraction of carries is 0.51).

Setup details We train a 3-layer (9 million parameter) Transformer model with embedding dimension 512, 4 attention heads per layer, and an MLP dimension of 2048, following prior work (Olsson et al., 2022a). We study a validation set with 1250 data points and 5000 output tokens throughout training. We compute the loss and POLCA values for each token at each interval of 20 training steps. The POLCA basis uses the eigenvectors of the Hessian, estimated using a 1250 data point sample of the training set as detailed in Algorithm 1. We compute one new basis vector every 200 training steps, for a total of 50 basis vectors. We provide further ablations on the decomposition strategy and choice of POLCA basis in Appendices G and H. We train the model for 10000 steps, but trim the x-axis of the plots at 4000 to better display the breakthroughs.

Clustering As described in Section 3.1, some of the top Hessian directions may represent directions of oscillation (and not learning) during training. To ensure that we are investigating directions where the model is learning on average, we only consider the basis vectors for which the mean projected loss decreases. Then for each remaining basis vector, we remove all of the tokens for which the decomposed loss increases, therefore retaining only tokens which rely on the vector during training. This removes 2360.8 out of 5000 tokens on average—suggesting that any given direction is irrelevant to learning many examples. We use HDBSCAN to cluster the remaining tokens, discarding the tokens it marks as outliers. Through this process, we find subpopulations of the data that have similar projected loss trajectories.

# 4.2 RESULTS

Comparison to the exact loss In our clustering experiments on arithmetic addition skills, we first consider whether decomposition is necessary for identifying conceptual skills. As a baseline, we therefore cluster tokens solely on their exact loss curves, rather than their decomposed loss curves. According to the exact loss clustering results in Figure 3, we can recover—to a substantial degree—the digit skill by clustering only on the exact loss, likely because the digits have very different loss trajectories. However, as shown in Figure 3 and Table 1, we cannot recover clusters that are homogenous with respect to the carrying skill from the exact loss alone.

<table><tr><td>Decomposition strategy</td><td>Maximum carry homogeneity</td><td>Clusters with hidden breakthroughs</td></tr><tr><td>Exact loss</td><td>0.514</td><td>0.0</td></tr><tr><td>Change in exact loss</td><td>0.524</td><td>0.0</td></tr><tr><td>LCA (Lan et al., 2020)</td><td>0.792</td><td>0.019</td></tr><tr><td>POLCA</td><td>0.973</td><td>0.355</td></tr></table>

Table 1: Cluster quality comparison. We compute the maximum fraction of points within all clusters that contain a carry for the specified digit and the fraction of clusters with hidden breakthroughs past the plateau in the exact loss at τ = 1000. For details and other metrics, see Appendix G.

Recovering concepts with POLCA clustering Because exact loss clustering failed to recover the carrying skill, we will recover it with a different clustering method. Instead of exact loss, we will cluster on each basis vector’s projected loss using the POLCA decomposition. The projected loss value $L _ { b } ( x , \theta _ { t } )$ (Equation 8) represents the cumulative loss change of x attributed to movement along basis vector b. By clustering the projected loss trajectories, we find that the top 2 basis vectors produce two easily-described and near-homogeneous clusters: one which contains examples of the 1000s place digit and one which contains examples of arithmetic carrying for all digits (Figure 4 Appendix Figure 8). We therefore determine that POLCA clustering recovers subtler skills—like carrying—that are challenging to reconstruct from the exact loss or parameter-aligned LCA curves alone (Table 1). In Appendix Table 9, we also use Equation 10 to compare the fraction of clusters with hidden breakthroughs past step 1000 (with ∆ = 100), where the mean exact loss plateaus (Appendix Figure 6), and find in Table 1 that POLCA discovers the highest fraction of clusters with hidden breakthroughs.

![](images/03427b11f0b94124b97ca4dec7c3694c44837c5a2bf5ef30f1bb3fcb26f48024.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 0         | 0         | 0         |
| 4000      | -0.5      | -0.5      | -1.0      |
</details>

(a) Median projected loss of basis vector #1’s POLCA clusters.

![](images/d49255815f988dd7988679a6d98262eab6a35de559df7d121518d3eb6cda9275.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 2.5       | 2.0       | 2.8       |
| 500       | 0.5       | 0.2       | 0.8       |
| 1000      | 0.2       | 0.1       | 0.4       |
| 1500      | 0.1       | 0.05      | 0.2       |
| 2000      | 0.05      | 0.02      | 0.1       |
| 2500      | 0.02      | 0.01      | 0.05      |
| 3000      | 0.01      | 0.005     | 0.02      |
| 3500      | 0.005     | 0.002     | 0.01      |
| 4000      | 0.002     | 0.001     | 0.005     |
</details>

(b) Median exact loss of basis vector #1’s POLCA clusters.

![](images/0503a9b18d594f4fcc7edf2a28c7c692650b5d0094a387368c30a98edc6e1299.jpg)

<details>
<summary>bar</summary>

| Cluster | Digit skill 100s | Digit skill 100s | Digit skill 10s | Digit skill 1s | Carrying skill Carry | Carrying skill No carry |
| ------- | ----------------- | ----------------- | ---------------- | --------------- | -------------------- | ----------------------- |
| 1       | 20                | 500               | 500              | 450             | 150                  | 50                      |
| 2       | 500               | 20                | 20               | 20              | 20                   | 20                      |
| 3       | 20                | 20                | 20               | 120             | 20                   | 20                      |
</details>

(c) Arithmetic skill composition of basis vector #1’s POLCA clusters.

![](images/529194f50c01a5e3f3d67350f4b69553aa45c9be78e0d02769f07e0c7d6521ce.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 0         | 0         | 0         |
| 1000      | -1        | -1        | -2        |
| 2000      | -1        | -1        | -2        |
| 3000      | -1        | -1        | -2        |
| 4000      | -1        | -1        | -2        |
</details>

(d) Median projected loss of basis vector #2’s POLCA clusters.

![](images/e465bb9347c31434d5fafe79a56be2eefa4d2316cae0cc6ad021576e37731405.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 3.0       | 3.0       | 3.0       |
| 500       | 1.0       | 0.5       | 1.5       |
| 1000      | 0.5       | 0.2       | 0.8       |
| 1500      | 0.3       | 0.1       | 0.5       |
| 2000      | 0.2       | 0.05      | 0.3       |
| 2500      | 0.1       | 0.02      | 0.2       |
| 3000      | 0.05      | 0.01      | 0.1       |
| 3500      | 0.02      | 0.005     | 0.05      |
| 4000      | 0.01      | 0.002     | 0.02      |
</details>

(e) Median exact loss of basis vector #2’s POLCA clusters.

![](images/6877ffc631b882a6fb29a575f08ac0bc2f40fe4ad6c1aecd35c05b3592cfab64.jpg)

<details>
<summary>bar</summary>

| Cluster | Digit skill 1000s | Digit skill 100s | Digit skill 10s | Digit skill 1s | Carrying skill Carry | Carrying skill No carry |
| ------- | ----------------- | ---------------- | --------------- | -------------- | -------------------- | ----------------------- |
| 1       | 50                | 250              | 100             | 0              | 0                    | 0                       |
| 2       | 450               | 0                | 20              | 100            | 0                    | 0                       |
| 3       | 0                 | 100              | 80              | 150            | 0                    | 0                       |
</details>

(f) Arithmetic skill composition of basis vector #2’s POLCA clusters.   
Figure 4: Arithmetic data clusters with POLCA. We perform POLCA clustering on the top 2 basis vectors, and report the cluster medoid and quartiles (left), median exact loss (center), and cluster skill composition (right) for each basis vector in order. Vertical lines mark the timestep when the relevant basis vector was sampled; note that a vector’s breakthroughs are not directly associated with this timestep. We find that the first basis vector recovers the digit skill whereas the second basis vector recovers the carrying skill (cluster #1 has homogeneity 0.90). The clusters computed from the POLCA trajectories show changes in the decomposed loss that are obscured in the exact loss curves.

Figure 4 shows POLCA clustering for the first two basis vectors. Along these basis vectors, certain data subpopulations show changes in the projected loss (Figures 4a and 4d) that do not occur as visibly in their smoother exact loss curves (Figures 4b and 4e). We conclude that arithmetic carries rely on breakthroughs along specific dimensions during training, but these breakthroughs may be elided in the exact loss curve.

# 5 ENGLISH LANGUAGE MODELING

We apply our approach to a real-world causal language modeling task and show that POLCA breakthrough clustering recovers interpretable conceptual skills in the natural language setting.

# 5.1 EXPERIMENTS

For the natural language modeling setting, we use the English Wikipedia dataset (Wikimedia Foundation, 2022) from March 2022 to train a 3-layer (40m parameter) model. We use the same POLCA setup as in the arithmetic addition setting (see Appendix E.2 for details). As in the arithmetic setting, we only consider directions where the projected aggregate loss overall decreases to filter out directions of oscillation. We also discard the datapoint-specific POLCA trajectories along the directions where

![](images/36c30601bbe44bc8123600e7580d04c8feed60206dbde27f6d04feec3b14fbc9.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 0.0       | 0.0       | 0.0       |
| 5000      | -0.2      | -0.1      | -0.1      |
| 10000     | -0.4      | -0.1      | -0.1      |
</details>

(i) Projected loss

<table><tr><td>Cluster</td><td>Label</td><td>Contexts closest to medoid</td></tr><tr><td rowspan="3">1</td><td rowspan="3">andafter the first clause in a sentence</td><td>FA intermediate cup twice, in 1942-43 and 1944-45 seasons.\n\nAfter the warthe</td></tr><tr><td>the 19th century. He happened to live through the tense period of the struggle for independence of Perufrom</td></tr><tr><td>Voight), his mentor. While attempting to prevent files containing information on all IMF&#x27;s field agentsfrom</td></tr><tr><td>2</td><td>Repeated newline</td><td>2008, and 2010. In 2000, it was named a National Blue Ribbon School of Excellence.\n\nduplication. The program was first introduced in 1998 and was discontinued on March 31, 2009.\n\nand the protagonist of the Mission: Impossible film series. He is portrayed by Tom Cruise.\n\n</td></tr><tr><td>3</td><td>Comma after parenthetical phrase</td><td>in most countries. The largest episcopal conferences are those of India and China, followed by the Philippines,in 1888 by the Swedish physicist Johannes Rydberg, theoretically by Niels Bohr in 1913,klas Athletic F.C. is a football club based in Leagrave, in Luton.</td></tr></table>

![](images/e8ff491b16448e56d874609ada6c2fe7125b9a3e229dfdf235887b471a174b0a.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 10.0      | 10.0      | 10.0      |
| 5000      | 6.0       | 1.0       | 2.0       |
| 10000     | 5.0       | 0.5       | 1.5       |
</details>

(ii) Exact loss   
(iii) Cluster data examples   
(a) POLCA clusters from basis vector 13.

![](images/49817bd5eab7b6e08e5fa802f63d246cf9ced2c645ea4ebcb9661225589e9d04.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 0.0       | 0.0       | 0.0       |
| 5000      | ~0.1      | ~-0.1     | ~-0.1     |
| 10000     | ~-0.1     | ~-0.1     | ~-0.1     |
| 15000     | ~-0.1     | ~-0.1     | ~-0.1     |
</details>

(i) Projected loss

<table><tr><td colspan="2">Cluster Label</td><td>Contexts closest to medoid</td></tr><tr><td rowspan="3">1</td><td rowspan="3">Appositive noun phrase</td><td>R appear in chapter 3 of Air Force Instruction 36-2406: Officer and Enlisted Evaluation</td></tr><tr><td>this was changed to the current WMCN on July Tin Woodman of Oz: A Faithful</td></tr><tr><td>relative path\n A second derivative in Newton&#x27;s notation\n A diaeresis, a type of diacritic</td></tr><tr><td rowspan="2">2</td><td rowspan="2">Non-appositive phrase preceded by a comma</td><td>year in fantasy, an essay on the year&#x27;s best fantasy books, and introductory notes to the in Saiunkoku Monogat, Gridman in Gridman the Hyper Agent, He</td></tr><tr><td>to calculate the wavelengths of the hydrogen spectral series.\n\nHistory \nIn 1880, R</td></tr><tr><td rowspan="3">3</td><td rowspan="3">Repeated newline</td><td>it actually gathers together pieces originally published during a two-year period, 1973 and 1974.\n</td></tr><tr><td>west part of the village is in Schoolcraft Township and the east part is in Brady Township.\n</td></tr><tr><td>(1984) and its sequel Breakin&#x27; 2: Electric Boogaloo (1984).\n</td></tr></table>

![](images/ba53bbbef00559bb8b17dc1223fab0e085a851c9edec6c5afddb0e28ff7703c0.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 11.0      | 11.0      | 11.0      |
| 5000      | 6.0       | 5.0       | 1.0       |
| 10000     | 5.5       | 4.5       | 0.5       |
</details>

(ii) Exact loss   
(iii) Cluster data examples   
(b) POLCA clusters from basis vector 23.   
Figure 5: Examples of English LM data clusters with POLCA. After clustering on POLCA trajectories for two illustrative basis vectors, we report their average decomposed POLCA trajectories (5a(i) and 5b(i)). Figures 5a(ii) and 5b(ii) show the average of the exact loss trajectories for each of the POLCA trajectory clusters. For each cluster, we provide a label based on the top POS tags and tokens in the cluster and the top 10 contexts closest to its medoid. We report the 3 contexts closest to the cluster medoid and color the corresponding token. Clustering on the decomposed POLCA trajectories reveals low-rank breakthroughs at times when the full-rank exact loss curve remains smooth. See other examples in Appendix K.

the token’s decomposed loss increases, which removes an average 6655.5 out of 12600 tokens per direction. After clustering the remaining token trajectories, we discard HDBSCAN-marked outliers.

Automatic labeling. To analyze the concepts represented by each cluster, we look for syntactic and lexical patterns shared by the cluster data.1 Our automatic labels are based on the target token and its preceding trigram. To obtain automatic labels for a cluster, we compute the frequency of each POS tag (tagged by spacy) in it. We then automatically label each cluster with the smallest set of POS tags required to compose 70% of the cluster’s 4-gram samples. For example, if over 70% of the token instances in a cluster are preceded by a comma <,>, the cluster would be automatically labeled as 1 token after PUNCT. We consider basis vectors with at least one cluster with a simple label—one including at most two POS tags (not counting <PAD> tokens). Starting with 30 basis vectors, we remove 4 because the average decomposed loss increases. We find 22 of the remaining basis vectors have at least one cluster with a simple label as defined. We refine these automatically assigned labels by examining the most frequent tokens in the cluster and the ten examples closest to the cluster medoid and manually selecting a label that is consistent with the automatic POS label and the ten closest examples to the medoid. See Appendix E.2 for further labeling details.

# 5.2 RESULTS

On language models, POLCA clustering again reveals hidden breakthroughs along each basis vector. We show a selection of clusters from specific basis vectors in Figure 5 and others in Appendix K (Figures 10 and 11). Clustering projected loss trajectories along each basis vector, we find groups that correspond to various grammatical constructions. Some show apparent breakthroughs in their projected loss, like the cluster corresponding to predicting < to> and < from> after the first clause in a sentence (Figure 5a(i)). We also observe clusters whose projected loss trajectories move in opposing directions along certain basis vectors. For instance, in Figure 5b, cluster 1 contains appositive noun phrases and cluster 2 has syntactically similar—but non-appositive—noun phrases (such as list items). These clusters visibly mirror each other’s movement—though the clustered decreases in loss are generally larger than their opposing cluster’s increases in loss.

Figure 5 shows that despite their smooth exact loss curves, POLCA clusters have sudden changes in their decomposed loss curves at different points during training. Clusters from the exact loss curves, by contrast, do not reveal breakthroughs except very early in training (Appendix Figure 9). We conclude that POLCA reveals breakthroughs in the decomposed loss that are obscured in the exact loss. Through clustering, POLCA can explain how different skills are learned during training.

# 6 CONCLUSIONS

This work introduces POLCA clustering, a method to identify learned skills from decomposed loss trajectories. POLCA decomposes the loss on two levels: individual datapoints and specific directions in the weight space. We use this decomposition to discover clusters that share breakthroughs obscured by loss metrics. In language modeling and synthetic settings, these clusters recover interpretable skills which appear to emerge at particular moments during training.

These are promising findings for meaningfully interpreting large models. By recovering breakthroughs in identifiable skills, we support the hypothesis that high-dimensional learning typically entails a series of breakthroughs at various scales. When a breakthrough appears in training, it suggests a naturally discrete category; the model either knows the concept or doesn’t know it, with little middle ground. Humans think in categorical concepts, so these breakthroughs are far more interpretable than the continuous data interpolations often assumed in learning theory.

# REPRODUCIBILITY STATEMENT

We implement our models and experiments using open-source libraries and datasets. We provide detailed hyperparameters in Appendix D and a thorough explanation of the experimental setup in Section E. Our code is available at https://github.com/skangasl/POLCA.

# ETHICS STATEMENT

This work provides a method for better understanding the training dynamics of language models. The trained models may contain biases from the training datasets.

# ACKNOWLEDGEMENTS

This work was informed by helpful conversations with Nikhil Vyas, Nicholas Lourie, Mike Lepori, and Ekdeep Singh Lubana. This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE 2140743. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors(s) and do not necessarily reflect the views of the National Science Foundation. This work was enabled in part by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute for the Study of Natural and Artificial Intelligence.

# REFERENCES

Emmanuel Abbe, Enric Boix-Adsera, Matthew Brennan, Guy Bresler, and Dheeraj Nagaraj. The staircase property: How hierarchical structure can guide deep learning, 2021.   
Alessandro Achille, Matteo Rovere, and Stefano Soatto. Critical learning periods in deep neural networks. CoRR, abs/1711.08856, 2017. URL http://arxiv.org/abs/1711.08856.   
Sanjeev Arora and Anirudh Goyal. A theory for emergence of complex skills in language models. arXiv preprint arXiv:2307.15936, 2023.   
Adithya Bhaskar, Dan Friedman, and Danqi Chen. The Heuristic Core: Understanding Subnetwork Generalization in Pretrained Language Models, June 2024. URL http://arxiv.org/abs/2403. 03942. arXiv:2403.03942 [cs].   
Ricardo J. G. B. Campello, Davoud Moulavi, and Joerg Sander. Density-based clustering based on hierarchical density estimates. In Jian Pei, Vincent S. Tseng, Longbing Cao, Hiroshi Motoda, and Guandong Xu (eds.), Advances in Knowledge Discovery and Data Mining, pp. 160–172, Berlin, Heidelberg, 2013. Springer Berlin Heidelberg. ISBN 978-3-642-37456-2.   
Angelica Chen, Ravid Shwartz-Ziv, Kyunghyun Cho, Matthew L. Leavitt, and Naomi Saphra. Sudden drops in the loss: Syntax acquisition, phase transitions, and simplicity bias in mlms, 2024a.   
Mayee Chen, Nicholas Roberts, Kush Bhatia, Jue Wang, Ce Zhang, Frederic Sala, and Christopher Ré. Skill-it! a data-driven skills framework for understanding and training language models. Advances in Neural Information Processing Systems, 36, 2024b.   
Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability, 2022.   
Felix Draxler, Kambis Veschgini, Manfred Salmhofer, and Fred A. Hamprecht. Essentially No Barriers in Neural Network Energy Landscape. arXiv:1803.00885 [cs, stat], February 2019. URL http://arxiv.org/abs/1803.00885. arXiv: 1803.00885.   
Alex Foote, Neel Nanda, Esben Kran, Ioannis Konstas, Shay Cohen, and Fazl Barez. Neuron to graph: Interpreting language model neurons at scale, 2023. URL https://arxiv.org/abs/2305.19911.   
Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In International Conference on Machine Learning, pp. 3259–3269. PMLR, 2020.

Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders, 2024. URL https: //arxiv.org/abs/2406.04093.   
Noah Golmant, Zhewei Yao, Amir Gholami Gholami, Michael Mahoney, and Joseph Gonzalez. pytorch-hessian-eigenthings: efficient pytorch hessian eigendecomposition, October 2018. URL https://github.com/noahgolmant/pytorch-hessian-eigenthings.   
Guy Gur-Ari, Daniel A. Roberts, and Ethan Dyer. Gradient descent happens in a tiny subspace, 2018.   
Stanisław Jastrze¸bski, Maciej Szymczak, Stanislav Fort, Devansh Arpit, Jacek Tabor, Kyunghyun Cho\*, and Krzysztof Geras\*. The break-even point on optimization trajectories of deep neural networks. In International Conference on Learning Representations, 2020.   
Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, and Naomi Saphra. Linear connectivity reveals generalization strategies. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=hY6M0JHl3uL.   
Janice Lan, Rosanne Liu, Hattie Zhou, and Jason Yosinski. Lca: Loss change allocation for neural network training, 2020.   
Charles Lovering, Jessica Forde, George Konidaris, Ellie Pavlick, and Michael Littman. Evaluation beyond task performance: Analyzing concepts in alphazero in hex. Advances in Neural Information Processing Systems, 35:25992–26006, 2022.   
Ekdeep Singh Lubana, Eric J. Bigelow, Robert P. Dick, David Krueger, and Hidenori Tanaka. Mechanistic mode connectivity, 2023. URL https://arxiv.org/abs/2211.08422.   
Chao Ma, Daniel Kunin, Lei Wu, and Lexing Ying. Beyond the quadratic approximation: the multiscale structure of neural network loss landscapes. arXiv preprint arXiv:2204.11326, 2022.   
Thomas McGrath, Andrei Kapishnikov, Nenad Tomašev, Adam Pearce, Martin Wattenberg, Demis Hassabis, Been Kim, Ulrich Paquet, and Vladimir Kramnik. Acquisition of chess knowledge in alphazero. Proceedings of the National Academy of Sciences, 119(47):e2206625119, 2022.   
Eric J. Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural scaling, 2024.   
Shikhar Murty, Pratyusha Sharma, Jacob Andreas, and Christopher D Manning. Grokking of hierarchical structure in vanilla transformers. arXiv preprint arXiv:2305.18741, 2023.   
Neel Nanda and Joseph Bloom. Transformerlens. https://github.com/TransformerLensOrg/ TransformerLens, 2022.   
Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability, 2023. URL https://arxiv.org/abs/2301.05217.   
Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads, 2022a. URL https://arxiv.org/abs/ 2209.11895.   
Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022b.   
Andres Potapczynski, Marc Finzi, Geoff Pleiss, and Andrew Gordon Wilson. CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra. arXiv preprint arXiv:2309.03060, 2023.   
Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177, 2022.

Tian Qin, Naomi Saphra, and David Alvarez-Melis. Sometimes i am a tree: Data drives unstable hierarchical generalization, 2024. URL https://arxiv.org/abs/2412.04619.   
Elan Rosenfeld and Andrej Risteski. Outliers with opposing signals have an outsized effect on neural network optimization. In The Twelfth International Conference on Learning Representations, 2024.   
Andrew M. Saxe, James L. McClelland, and Surya Ganguli. A mathematical theory of semantic development in deep neural networks. Proceedings of the National Academy of Sciences, 116(23): 11537–11546, 2019. doi: 10.1073/pnas.1820226116. URL https://www.pnas.org/doi/abs/ 10.1073/pnas.1820226116.   
Ravid Shwartz-Ziv and Naftali Tishby. Opening the black box of deep neural networks via information, 2017. URL https://arxiv.org/abs/1703.00810.   
Minhak Song, Kwangjun Ahn, and Chulhee Yun. Does SGD really happen in tiny subspaces? In High-dimensional Learning Dynamics 2024: The Emergence of Structure and Reasoning, 2024. URL https://openreview.net/forum?id=iITzMuv9sL.   
George Wang, Jesse Hoogland, Stan van Wingerden, Zach Furman, and Daniel Murfet. Differentiation and specialization of attention heads via the refined local learning coefficient. ArXiv, abs/2410.02984, 2024. URL https://api.semanticscholar.org/CorpusID:273162605.   
Sumio Watanabe. Asymptotic equivalence of bayes cross validation and widely applicable information criterion in singular learning theory. ArXiv, abs/1004.2316, 2010. URL https: //api.semanticscholar.org/CorpusID:15093314.   
Susan Wei, Daniel Murfet, Mingming Gong, Hui Li, Jesse Gell-Redman, and Thomas Quella. Deep learning is singular, and that’s good. IEEE Transactions on Neural Networks and Learning Systems, 34:10473–10486, 2020. URL https://api.semanticscholar.org/CorpusID:225041126.   
Wikimedia Foundation. Wikimedia downloads, 2022. URL https://dumps.wikimedia.org.   
Mengzhou Xia, Mikel Artetxe, Chunting Zhou, Xi Victoria Lin, Ramakanth Pasunuru, Danqi Chen, Luke Zettlemoyer, and Ves Stoyanov. Training trajectories of language models across scales, 2023.   
Ziqian Zhong, Ziming Liu, Max Tegmark, and Jacob Andreas. The clock and the pizza: Two stories in mechanistic explanation of neural networks, 2023. URL https://arxiv.org/abs/2306.17844.

# A LIMITATIONS AND FUTURE WORK

Our method of constructing a basis is inspired by the existing literature on training in restricted subspaces, but represents an obvious site of improvement. The top eigenvectors of the Hessian, like the axis-aligned basis, could represent many concepts in superposition. Therefore, some nonorthogonal basis might represent interpretable concepts more cleanly than our orthogonal basis, though it would no longer provide a low-rank decomposition. Furthermore, our basis is constructed by local curvature and then filtered to favor directions of long-term movement; other bases may favor long-term movement by construction. In general, we consider the ideal basis to be an open question.

Our experiments are limited to small models. The two main challenges with scaling POLCA are the Hessian basis computation and the frequency of checkpoints used to sample the POLCA trajectories. Scaling this work to larger models may require using a basis that is less computationally expensive to compute than Hessian eigenvectors, but our results from Appendix H indicate that this is likely possible with limited impact on the cluster quality. The small scale of models that we use in our experiments allows for very high granularity of checkpoints used to compute both the basis and the POLCA trajectories. For larger models, this may be computationally infeasible and a lower checkpoint frequency may be needed, resulting in less signal for clustering the POLCA trajectories.

Our current experiments are limited to language models. However, in principle our approach is model-agnostic and can be applied to any deep neural network. Applying POLCA to other modalities is an exciting direction for future work.

The labeling approach that we use in the natural language setting relies on POS tagging. This labeling strategy allows for unsupervised, automatic identification of these syntactic skills and ensures strict interpretable labels. However, it fails to capture many human-interpretable language modeling skills. The discarded vectors may (and likely do) contain other interpretable skill clusters that are not found by automated labeling.

# B STATEMENT ON USE OF LARGE LANGUAGE MODELS

We used generative AI for debugging and minor grammar edits when writing. The authors made all significant contributions to the research, analyses, and writing.

# C DERIVATION OF APPROXIMATE SECOND-ORDER TERM

We can approximate the difference between the gradient at time t and $t + 1$ as

$$
g _ {t + 1} (X) - g _ {t} (X) \approx \mathcal {H} _ {t} (X) (\theta_ {t + 1} - \theta_ {t}) \tag {11}
$$

$$
\langle g _ {t + 1} (X) - g _ {t} (X), b \rangle \approx b ^ {\top} \mathcal {H} _ {t} (X) b \langle b, \theta_ {t + 1} - \theta_ {t} \rangle \tag {12}
$$

$$
= \lambda_ {t} (X) \left\langle b, \theta_ {t + 1} - \theta_ {t} \right\rangle \tag {13}
$$

If we assume b to also be an eigenvector of the datapoint Hessians $\mathcal { H } _ { t } ^ { \prime } ( x )$ , we can apply a similar argument for the gradient on the datapoint level.

$$
\langle g _ {t + 1} ^ {\prime} (x) - g _ {t} ^ {\prime} (x), b \rangle \approx b ^ {\top} \mathcal {H} _ {t} ^ {\prime} (x) b \left\langle b, \theta_ {t + 1} - \theta_ {t} \right\rangle \tag {14}
$$

Note that the assumption above (of matching Hessian eigenvectors between data points and their aggregate) is unlikely to be correct. If this assumption is violated, then the scaling factor in the following second-order Taylor term will be minuscule on the datapoint level. In practice, we have found that the second-order term has limited impact at the datapoint level (see Appendix I), but we nonetheless use it to improve our approximation. Then we may approximate it as:

$$
\frac {\langle g _ {t + 1} ^ {\prime} (x) - g _ {t} ^ {\prime} (x) , b \rangle}{\langle g _ {t + 1} (X) - g _ {t} (X) , b \rangle} \approx \frac {b ^ {\top} \mathcal {H} _ {t} ^ {\prime} (x) b \langle b , \theta_ {t + 1} - \theta_ {t} \rangle}{\lambda_ {t} (X , b) \langle b , \theta_ {t + 1} - \theta_ {t} \rangle} \tag {15}
$$

$$
\frac {\left\langle g _ {t + 1} ^ {\prime} (x) - g _ {t} ^ {\prime} (x) , b \right\rangle}{\left\langle g _ {t + 1} (X) - g _ {t} (X) , b \right\rangle} \approx \frac {\left\langle h _ {t} ^ {\prime} (x) , b \right\rangle \left\langle b , \theta_ {t + 1} - \theta_ {t} \right\rangle}{\lambda_ {t} (X , b) \left\langle b , \theta_ {t + 1} - \theta_ {t} \right\rangle} \tag {16}
$$

$$
\lambda_ {t} (X, b) \frac {\langle g _ {t + 1} ^ {\prime} (x) - g _ {t} ^ {\prime} (x) , b \rangle}{\langle g _ {t + 1} (X) - g _ {t} (X) , b \rangle} \approx \langle h _ {t} ^ {\prime} (x), b \rangle \tag {17}
$$

Empirical Validation We run a small-scale empirical study to test the accuracy of this estimation. For a sample of 400 tokens and the first 10 POLCA checkpoints and the top 5 basis vectors in the arithmetic setting, we compute the root mean squared error (RMSE) between the approximated second-order term $\lambda _ { t } ( X , b ) \frac { \langle g _ { t + 1 } ^ { \prime } ( x ) - g _ { t } ^ { \prime } ( x ) , b \rangle } { \langle g _ { t + 1 } ( X ) - g _ { t } ( X ) , b \rangle }$ and the true second-order term $\left. h _ { t } ^ { \prime } ( x ) , b \right.$ averaged across tokens, checkpoints, and basis vectors. We find that our approximation has an RMSE of 0.145 when compared to the ground truth second-order term, indicating that this approximation is close to the real value.

# D HYPERPARAMETERS

In the tables below, we provide the hyperparameters used during training of the models in the synthetic arithmetic and language modeling settings. We use NVIDIA H100 80GB HBM3 GPUs for our experiments and run each training run on a single GPU.

We selected the clustering hyperparameters to maximize empirical performance in the synthetic setting and used similar values relative to the number of tokens in the natural language experiments. We chose the POLCA hyperparameters to maximize the frequency of POLCA checkpoints and basis checkpoints within computational constraints.

Table 2: Hyperparameters for training the synthetic arithmetic model 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Number of Parameters</td><td>9475594</td></tr><tr><td>Steps</td><td>10000</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Number of training tokens</td><td>2560000</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>1e-5</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>Betas</td><td>(0.9, 0.95)</td></tr><tr><td>LR Schedule</td><td>min(i/100, 1.0)</td></tr></table>

Table 3: POLCA hyperparameters for the synthetic setting 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Basis checkpoint interval (steps)</td><td>200</td></tr><tr><td>T</td><td>50</td></tr><tr><td>k</td><td>1</td></tr><tr><td>POLCA checkpoint interval (steps)</td><td>5</td></tr></table>

Table 4: Arithmetic clustering hyperparameters 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Clustering algorithm</td><td>HDBSCAN</td></tr><tr><td>Minimum cluster size</td><td>150</td></tr><tr><td>Minimum samples</td><td>Number of tokens / 15</td></tr></table>

Table 5: Hyperparameters for training the natural language model

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Number of Parameters</td><td>40274737</td></tr><tr><td>Steps</td><td>14000</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Number of training tokens</td><td>114688000</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>1e-5</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>Betas</td><td>(0.9, 0.95)</td></tr><tr><td>LR Schedule</td><td>min(i/100, 1.0)</td></tr></table>

Table 6: POLCA hyperparameters for the natural language setting 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Basis checkpoint interval (steps)</td><td>750</td></tr><tr><td>T</td><td>30</td></tr><tr><td>k</td><td>1</td></tr><tr><td>POLCA checkpoint interval (steps)</td><td>200</td></tr></table>

Table 7: Natural language clustering hyperparameters 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Clustering algorithm</td><td>HDBSCAN</td></tr><tr><td>Minimum cluster size</td><td>300</td></tr><tr><td>Minimum samples</td><td>Number of tokens / 15</td></tr></table>

# E EXPERIMENTAL DETAILS

# E.1 ARITHMETIC SETTING

Setup. In the arithmetic experiments in Section 4, we train a 3-layer (9m parameter) Transformer model with embedding dimension 512, 4 attention heads per layer, and an MLP dimension of 2048 (Nanda & Bloom, 2022). For a validation set with 1250 data points and 5000 output tokens, we compute the loss and POLCA values for each token at intervals of 20 steps throughout training. We compute the POLCA basis using the eigenvectors of the Hessian estimated using a 1250 data point sample of the training set. We compute a new basis vector every 200 steps for a total of 50 basis vectors.

Labeling details. We automatically label each token with the ground truth value of the digit and carry skills using the definition of these two skills. We label the digit skill based on the position of the token in the output: the first token is 1000s, the second token is 100s, the third is 10s, and the fourth is 1s. We label the carry skill by computing the sum up to the next lowest digit place and determining whether it resulted in a carry to the current token. For instance, for an output in the 10s place, if the two inputs in the 1s place sum to 10 or higher, then it will be labeled with "carry", otherwise it will be labeled with "no carry".

# E.2 NATURAL LANGUAGE SETTING

Setup. For the natural language experiments in Section 5, we train on the English Wikipedia dataset (Wikimedia Foundation, 2022) from March 2022. We use the same POLCA setup as in the arithmetic addition setting. We train a 3-layer (40m parameter) transformer model with embedding dimension 512, 4 attention heads, and an MLP dimension of 2048 (Nanda & Bloom, 2022). We compute the loss and POLCA values for each token on a validation set of 12600 output tokens. We analyze intermediate checkpoints at intervals of 200 steps throughout training. We apply POLCA to the basis derived from the eigenvectors of the Hessian estimated using a 1000 data point sample of the training set as detailed in Algorithm 1 with k = 1. We compute a new basis vector every 750 steps.

Labeling details. We label each token and the three tokens before it using spacy for part-of-speech (POS) tagging. This produces a sequence of four POS tags as the automatic POS label for each token. To label a given cluster, we compute the frequency (across the tokens in the cluster) of each POS tag at each index. For each index in the sequence, we then automatically label the cluster with the smallest set of POS tags at that index required to make up 70% of the cluster. We then filter out any labels that require more than 2 POS tags to describe the cluster. To generate the labels reported in Section 5, we manually refine the automatically generated label by looking at the top 10 contexts closest to the medoid of the cluster. Although these manually refined labels are challenging to verify automatically, we ensure that the contexts closest to the medoid follow the manual label and that the manual label follows the automatic label generated using the POS tags.

# F EXACT LOSS TRAJECTORIES FOR THE DIGIT AND CARRY SKILLS

![](images/759f30ac487a6f3aa88aefc2acdc8bbcff88990b699af772e53800fd53c315fd.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1000s | Cluster 100s | Cluster 10s | Cluster 1s |
| --------- | ------------- | ------------ | ----------- | ---------- |
| 0         | ~3.5          | ~2.8         | ~2.7        | ~2.6       |
| 1000      | ~0.1          | ~0.3         | ~0.4        | ~0.5       |
| 5000      | ~0.05         | ~0.1         | ~0.15       | ~0.2       |
| 10000     | ~0.05         | ~0.05        | ~0.05       | ~0.05      |
</details>

Figure 6: Median and quartiles of the loss trajectories for each digit.

![](images/73e7e92974fd9bd06490284c577b9f3bd469845cec86237aef9c7b00e3994365.jpg)

<details>
<summary>line</summary>

| Iteration | Full-rank loss (No carry 1000s) | Full-rank loss (Carry 1000s) |
| --------- | ------------------------------- | ---------------------------- |
| 0         | ~3.5                            | ~3.5                         |
| 1000      | ~0.1                            | ~0.1                         |
| 5000      | ~0.05                           | ~0.05                        |
| 10000     | ~0.05                           | ~0.05                        |
</details>

(a) 1000s place

![](images/10d1d62d84de82e79383d617c99057ec2c8d052e503b306bcfaeba1078c67209.jpg)

<details>
<summary>line</summary>

| Iteration | No carry 100s | Carry 100s |
| --------- | ------------- | ---------- |
| 0         | ~3.5          | ~3.5       |
| 1000      | ~0.5          | ~0.6       |
| 5000      | ~0.1          | ~0.1       |
| 10000     | ~0.0          | ~0.0       |
</details>

(b) 100s place

![](images/646ef1674ecf8aae78bf490c500bacf2b2224f12f5bc33ba1cf22fd6391885cf.jpg)

<details>
<summary>line</summary>

| Iteration | No carry 10s | Carry 10s |
| --------- | ------------ | --------- |
| 0         | ~3.5         | ~3.5      |
| 1000      | ~0.5         | ~0.6      |
| 5000      | ~0.1         | ~0.1      |
| 10000     | ~0.0         | ~0.0      |
</details>

(c) 10s place

![](images/78e748f5f84f13124bc37b1a83431f751b9ba599ab5afb2a7e13cb2056d93bac.jpg)

<details>
<summary>line</summary>

| Iteration | Full-rank loss |
| --------- | -------------- |
| 0         | ~2.5           |
| 1000      | ~0.1           |
| 5000      | ~0.0           |
| 10000     | ~0.0           |
</details>

(d) 1s place   
Figure 7: Median and quartiles of the loss trajectories for each digit and carry combination.

# G DECOMPOSITION STRATEGY COMPARISON

We investigate whether POLCA is required for decomposing the loss. We expand on the results from Table 1 by showing the top three homogeneity scores for the carrying skill and adding empirical Fisher information and first order POLCA results. To compute the homogeneity score for a given basis vector, we take the maximum fraction of carries in any given cluster for that basis vector (or across all of the clusters for exact loss or change in exact loss). We then report the top three homogeneities across the full basis, as well as the recall and F1 score for the corresponding clusters. We note that we exclude HDBSCAN outliers from the recall and F1 computations. Importantly, we only consider clusters for which over 85% of the tokens with the carry skill correspond to the 10s or 100s place, since carrying to the 1000s place corresponds to simply predicting a 0 or 1 in the first position (see Figure 2 for reference) and is recovered with high homogeneity for all trajectory types except exact loss.

We compare carry skill homogeneity across the following trajectory types:

• Loss: Exact loss trajectories.   
• Change in exact loss: We compute the change in exact loss by subtracting the loss at checkpoint t − 1 from the loss at checkpoint t for each timestep $t > 0$ in the exact loss trajectory.   
• Fisher information: We approximate the empirical Fisher Information as $\lVert \nabla _ { \theta } L ( x , \theta _ { t } ) \rVert _ { 2 } ^ { 2 }$ as in Achille et al. (2017). For each basis vector b, the Fisher Information projected onto b is $\langle b , \nabla _ { \theta } L ( x , \theta _ { t } ) \rangle ^ { 2 }$ .

• Loss Change Allocation (LCA): We compute the datapoint-wise LCA trajectories (Equation 1) projected onto the parameters that have the top 50 highest magnitudes at the end of training.   
• First order POLCA: We calculate the POLCA trajectory without the second order term (Equation 2)   
• POLCA: We compute the projected loss (Equation 7).

The results from Table 8 demonstrate that POLCA finds the most homogeneous clusters with respect to the carrying skill. Moreover, POLCA and first order POLCA have comparable F1 scores. Note that many low homogeneity clusters can have high recall because they are large and thus contain most of the carry tokens while having low homogeneity.

Table 8: Carry skill homogeneity comparison. For each type of trajectory, we compute the fraction of points within each cluster that contain a carry to the output token and report the homogeneity, recall, and F1 score for the three clusters with highest homogeneity across all 50 vectors. POLCA recovers carry clusters with the highest homogeneity. 

<table><tr><td rowspan="2">Decomposition strategy</td><td colspan="4">Cluster</td></tr><tr><td>Number</td><td>Homogeneity</td><td>Recall</td><td>F1</td></tr><tr><td>Loss</td><td>1</td><td>0.514</td><td>0.771</td><td>0.617</td></tr><tr><td>Change in exact loss</td><td>1</td><td>0.524</td><td>0.958</td><td>0.678</td></tr><tr><td rowspan="3">Fisher information</td><td>1</td><td>0.664</td><td>0.947</td><td>0.781</td></tr><tr><td>2</td><td>0.643</td><td>0.740</td><td>0.688</td></tr><tr><td>3</td><td>0.637</td><td>0.874</td><td>0.737</td></tr><tr><td rowspan="3">Loss change allocation (LCA) (Lan et al., 2020)</td><td>1</td><td>0.792</td><td>0.772</td><td>0.782</td></tr><tr><td>2</td><td>0.614</td><td>0.873</td><td>0.721</td></tr><tr><td>3</td><td>0.592</td><td>0.626</td><td>0.609</td></tr><tr><td rowspan="3">First order POLCA</td><td>1</td><td>0.948</td><td>0.767</td><td>0.848</td></tr><tr><td>2</td><td>0.928</td><td>0.769</td><td>0.841</td></tr><tr><td>3</td><td>0.887</td><td>0.751</td><td>0.813</td></tr><tr><td rowspan="3">POLCA</td><td>1</td><td>0.973</td><td>0.736</td><td>0.838</td></tr><tr><td>2</td><td>0.946</td><td>0.773</td><td>0.850</td></tr><tr><td>3</td><td>0.903</td><td>0.762</td><td>0.827</td></tr></table>

We also compare the fraction of clusters with hidden breakthroughs for each type of trajectory. To compute whether or not a given cluster has a hidden breakthrough, we use Equation 10 with ∆ = 100 to identify breakthroughs past step τ = 1000 where the exact loss plateaus. We find that POLCA produces the highest fraction of clusters with hidden breakthroughs. We hypothesize that this is mostly due to the basis construction, since the Fisher information and first order POLCA (both of which are computed using the same basis as POLCA) produce the next highest fraction of clusters with hidden breakthroughs.

Table 9: Hidden breakthroughs comparison. For each type of trajectory, we use Equation 10 to compute the fraction of clusters with a hidden breakthrough past the plateau in the exact loss at step τ = 1000. 

<table><tr><td>Decomposition strategy</td><td>Fraction of clusters with hidden breakthroughs</td></tr><tr><td>Loss</td><td>0.0</td></tr><tr><td>Change in exact loss</td><td>0.0</td></tr><tr><td>Fisher information</td><td>0.284</td></tr><tr><td>Loss change allocation (LCA) (Lan et al., 2020)</td><td>0.019</td></tr><tr><td>First order POLCA</td><td>0.307</td></tr><tr><td>POLCA</td><td>0.355</td></tr></table>

# H POLCA BASIS COMPARISON

We test ablated bases to analyze the effect of basis choice on the POLCA breakthrough clustering. To do so, we compute the maximum carry skill homogeneities over all of the clusters when performing POLCA breakthrough clustering. We use the following bases:

• Random orthonormal: randomly sampled orthonormal vectors   
• Random shuffled Hessian: basis computed using Algorithm 1, but randomly shuffling the model checkpoints   
• Top Hessian eigenvectors: basis computed using Algorithm 1

We find in Table 10 that these ablations result in only slightly lower quality clusters with respect to homogeneity (although random orthonormal vectors have lower recall than the other two bases on average), indicating that different bases can be used for larger experiments to trade off between compute and interpretability. We also compute the fraction of clusters with hidden breakthroughs for each basis in Table 11 and find that the random orthonormal basis has a significantly lower fraction of hidden breakthroughs recovered than the other two approaches, indicating that this random orthonormal basis is not sufficient to find hidden breakthroughs late in training.

In addition to these bases, we have tested a variety of additional basis constructions, such as a stacked Jacobian, Hessian computed using a sliding window, and Hessian computed at the end of training, and chose to use the top Hessian eigenvectors since they had the best performance in the arithmetic setting.

Table 10: Carry skill homogeneity comparison. For each basis, we compute the fraction of points within each cluster that contains a carry to the output token and report the homogeneity, recall, and F1 for the clusters with maximum homogeneity. Using the top Hessian eigenvectors recovers slightly more homogeneous carry clusters than the other basis selection strategies. The random orthonormal vectors have high homogeneity but lower recall than the other two bases. 

<table><tr><td rowspan="2">POLCA basis</td><td colspan="4">Cluster</td></tr><tr><td>Number</td><td>Homogeneity</td><td>Recall</td><td>F1</td></tr><tr><td rowspan="3">Random orthonormal</td><td>1</td><td>0.902</td><td>0.655</td><td>0.759</td></tr><tr><td>2</td><td>0.858</td><td>0.533</td><td>0.658</td></tr><tr><td>3</td><td>0.838</td><td>0.586</td><td>0.689</td></tr><tr><td rowspan="3">Random shuffled Hessian</td><td>1</td><td>0.856</td><td>0.699</td><td>0.769</td></tr><tr><td>2</td><td>0.852</td><td>0.734</td><td>0.789</td></tr><tr><td>3</td><td>0.834</td><td>0.730</td><td>0.779</td></tr><tr><td rowspan="3">POLCA</td><td>1</td><td>0.973</td><td>0.736</td><td>0.838</td></tr><tr><td>2</td><td>0.946</td><td>0.773</td><td>0.850</td></tr><tr><td>3</td><td>0.903</td><td>0.762</td><td>0.827</td></tr></table>

Table 11: Hidden breakthroughs basis comparison. For each type of basis, we use Equation 10 to compute the fraction of clusters with a hidden breakthrough past the plateau in the exact loss at step τ = 1000. 

<table><tr><td>POLCA basis</td><td>Fraction of clusters with hidden breakthroughs</td></tr><tr><td>Random orthonormal</td><td>0.031</td></tr><tr><td>Random shuffled Hessian</td><td>0.304</td></tr><tr><td>POLCA</td><td>0.355</td></tr></table>

# I SECOND VERSUS FIRST ORDER POLCA APPROXIMATION

Table 12: Empirical comparison of second and first order POLCA values. For the arithmetic setting, we compute the average cosine similarity and L2 distance between the second (Eq 7) and first (Eq 2) order POLCA trajectory vectors. The first and second-order approximations of the POLCA trajectories are very similar on average.

<table><tr><td>Cosine similarity</td><td>L2 norm</td></tr><tr><td>5.4891 E-4</td><td>0.99987</td></tr></table>

# J ADDITIONAL ARITHMETIC LANGUAGE MODELING CLUSTERS

![](images/3d993a8ddd6c905605e25cb1155cf394c2f6c1573ac7b3cfb530efdcd53b1e25.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 0.0       | 0.0       | 0.0       |
| 1000      | -0.1      | 0.0       | -0.2      |
| 2000      | -0.15     | 0.0       | -0.2      |
| 3000      | -0.15     | 0.0       | -0.2      |
| 4000      | -0.15     | 0.0       | -0.2      |
</details>

(a) Median projected loss of basis vector #3’s POLCA clusters.

![](images/099df7740e0ef852616e6317dff29019e7fcd1ca10bcf3e43ee5570d50c1c077.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 3.0       | 3.0       | 3.0       |
| 500       | 1.0       | 0.5       | 1.5       |
| 1000      | 0.5       | 0.2       | 0.8       |
| 1500      | 0.3       | 0.1       | 0.5       |
| 2000      | 0.2       | 0.05      | 0.3       |
| 2500      | 0.1       | 0.02      | 0.2       |
| 3000      | 0.05      | 0.01      | 0.1       |
| 3500      | 0.02      | 0.005     | 0.05      |
| 4000      | 0.01      | 0.002     | 0.02      |
</details>

(b) Median exact loss of basis vector #1’s POLCA clusters.

![](images/5d6b17205c60e529dc16cc23d03ba8ae3fb91bf89f7cfaf8748e6e9e88ffd51d.jpg)

<details>
<summary>bar</summary>

| Cluster | Digit skill 1000s | Digit skill 100s | Digit skill 10s | Digit skill 1s | Carrying skill Carry | Carrying skill No carry |
| ------- | ----------------- | ---------------- | --------------- | -------------- | -------------------- | ----------------------- |
| 1       | 20                | 160              | 80              | 100            | 20                   | 10                      |
| 2       | 400               | 20               | 10              | 30             | 20                   | 1                       |
| 3       | 20                | 40               | 60              | 160            | 20                   | 1                       |
</details>

(c) Arithmetic skill composition of basis vector #3’s POLCA clusters.   
Figure 8: Arithmetic data clusters with POLCA. We perform POLCA clustering on the third basis vector, and report the cluster medoid and quartiles (left), median exact loss (center), and cluster skill composition (right). Vertical lines mark the timestep when the relevant basis vector was sampled; note that a vector’s breakthroughs are not directly associated with this timestep. We find that the third basis vector recovers the carrying skill in the 1000s place.

# K ADDITIONAL NATURAL LANGUAGE CLUSTERS

![](images/14c4354657c60ce3b70b76472165dc10560fd83d53f16fdbc82e0dcd347f3c5f.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 |
| --------- | --------- | --------- |
| 0         | 11.0      | 11.0      |
| 5000      | 1.0       | 3.0       |
| 10000     | 0.5       | 1.0       |
</details>

(i) Exact loss

<table><tr><td>Cluster</td><td>Label</td><td>Contexts closest to medoid</td></tr><tr><td>1</td><td>Repeatedtokens</td><td>Canterbury, New Zealand\nRivers of New Zealand</td></tr><tr><td>2</td><td>Continuinga noun</td><td>\n Diocese of Sylhet\nEpciscopal Conference of Brunei\nEcclesiastical Province of Brunei and German young people together for dialogue and educational programs.\nHistory\nThe settlement in the Silesian of India and China, followed by the Philippines, and Indonesia.\nList of Roman Catholic dioceses</td></tr></table>

(ii) Cluster data examples   
Figure 9: English language modeling data clusters with the exact loss. We cluster the exact loss trajectories and report the average loss by cluster (9i). For each cluster, we provide a label based on the top POS tags of tokens in the cluster and the top 10 contexts closest to the cluster medoid. We report the 3 contexts closest to the cluster medoid. Clustering on the loss trajectories only discovers a relatively simple skill, continuing nouns composed of multiple tokens. POLCA breakthrough clustering recovers a similar skill in Figures 10i and 10ii as well as discovering other skills.

![](images/5d91fdfe81297c1054cbb5949a701b076235e6a96825d4a1c2e25cadc78c8491.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 0.0       | 0.0       | 0.0       |
| 5000      | -0.2      | -0.8      | -0.5      |
| 10000     | -0.4      | -0.6      | -0.5      |
| 15000     | -0.5      | -0.5      | -0.5      |
</details>

(i) Projected loss

![](images/59256238c0c8d57bc529245b32cb61879a4e4bb0ddb1dad176b5b8a07b646df6.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 10.0      | 10.0      | 10.0      |
| 5000      | 3.0       | 6.0       | 1.0       |
| 10000     | 2.5       | 5.5       | 0.5       |
</details>

(ii) Exact loss

<table><tr><td>Cluster</td><td>Label</td><td>Contexts closest to medoid</td></tr><tr><td>1</td><td>Punctuation after noun phrase</td><td>Pao Newspaper Company Limited () under Ho Man-fat. It was initially published every three days,in the 6th and 5th centuries BC, to denote some type of a sibant sound,president of the United States the commander-in-chief of the United States Armed Forces. Many presidents.</td></tr><tr><td>2</td><td>Noun in noun phrase</td><td>rank (four regular officers, six militia officers, three volunteers). \n\n\n Table of United States presidentsa tournament to have a number of teams that is not a power of two, and gives an extra advantage down from 1,989 at the 2000 census. The Albert Gallatin Area School District serves the township</td></tr><tr><td>3</td><td>&lt; of&gt;</td><td>official from the Electorate of the Palatinate who served Brandenburg-Prussia. He was the son ofand civil parish in Milton Keynes, ceremonial Buckinghamshire, England. For local government purposes, it is part ofering as it extends outward from the coast. Cape Lookout State Park is located on the north side of</td></tr></table>

(iii) Cluster data examples   
(a) POLCA clusters from basis vector 10.

![](images/9bc9cdba0b348c00b73745dae172af16849404c3008e7d7c39144728f2cf8a35.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 0.0       | -0.2      | 0.0       |
| 5000      | -0.2      | -0.2      | -0.2      |
| 10000     | -0.2      | -0.2      | -0.2      |
</details>

(i) Projected loss

![](images/8c930ea4ba56b9d21ab5325c1826ef970ff358683548d454ff87139b7d3a365c.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 | Cluster 3 |
| --------- | --------- | --------- | --------- |
| 0         | 10.0      | 10.0      | 10.0      |
| 5000      | 7.0       | 1.0       | 4.0       |
| 10000     | 6.5       | 0.5       | 3.5       |
</details>

(ii) Exact loss

<table><tr><td>Cluster</td><td>Label</td><td>Contexts closest to medoid</td></tr><tr><td>1</td><td>Starting or continuing foreign language proper noun</td><td>árslevelů (in Hungarian), also called Lipovina (in Slovak), Frunza  $\underline{de}$  time and effort he devoted to medicine. He died in Lima on 1861.\n\CayetanoBeauharnois, Châteauguay-Huntingdon-Laprairie and  $\underline{Saint}$ </td></tr><tr><td>2</td><td>Repeated newline</td><td>was an English churchman who was known for his combative preaching and his Latin poetry.\n $\underline{n}$ (1984) and its sequel Breakin’ 2: Electric Boogaloo (1984).\ $\underline{n}$ acon offers opportunities to participate in student organizations, varsity athletics, community service, and international travel. $\underline{n}$ </td></tr><tr><td>3</td><td>Token after proper noun</td><td>obic bacteria are Staphylococcus spp., Escherichia col, Salmonella, ede Rukawa in Slam Dunk, Ayato Sakamaki in Diabolik Lovers, a 4 piece South African rock band from Johannesburg.\n\They formed in 1996 when Martin, Wade  $\underline{and}$ </td></tr></table>

(iii) Cluster data examples   
(b) POLCA clusters from basis vector 16.   
Figure 10: English language modeling data clusters with POLCA. We compute breakthrough clustering on POLCA trajectories for each vector and report the average decomposed POLCA trajectories (10ai and 10bi). Figures 10aii and 10bii show the average of the per-token loss trajectories for each of the clusters found using the POLCA trajectories. For each cluster, we provide a label based on the top tokens in the cluster and the top 10 contexts closest to its medoid. We then report the 3 contexts closest to the cluster medoid. Clustering on the decomposed POLCA trajectories reveals breakthroughs at points in training where the per-token loss curve remains smooth.

![](images/9e509aeacf0a0d0e55dfabdcc1a93406e5548149198b8598a40266adb10f6f6c.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 |
| --------- | --------- | --------- |
| 0         | 0.0       | 0.0       |
| 5000      | -0.05     | 0.0       |
| 10000     | -0.1      | 0.0       |
| 15000     | -0.15     | 0.0       |
</details>

(i) Projected loss

<table><tr><td>Cluster</td><td>Label</td><td>Contexts closest to medoid</td></tr><tr><td>1</td><td>Verb/noun in sentence-initial phrase</td><td>publishes the pamphlet Les Peintres impressionistes.\n Czech painter Karel Kličperfectsforemost mast are called jibs, headsails, or foresails. The innermost suchsailwon the Miss Minnesota USA pageant and competed at Miss USA.\n\nAnnika Wiese of_Ow</td></tr><tr><td>2</td><td>Repeatedtokens</td><td>\nSee also\nRoyal chapel (disambiguation)\nPalatine</td></tr></table>

(iii) Cluster data examples

![](images/a7236d7df8fd6eba41cfcb62cfdb80302c1699835c5acdb767cc80c31a708e60.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 |
| --------- | --------- | --------- |
| 0         | 10.0      | 10.0      |
| 5000      | 6.0       | 0.0       |
| 10000     | 5.0       | 0.0       |
| 15000     | 4.5       | 0.0       |
</details>

(ii) Exact loss   
(a) POLCA clusters from basis vector 28.

![](images/a4384f5c03dcc109a2606ea47595605d1cd0e296c52c1fdbdc120025109c79d2.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 |
| --------- | --------- | --------- |
| 0         | 0.0       | 0.0       |
| 5000      | -0.1      | 0.0       |
| 10000     | -0.15     | 0.0       |
</details>

(i) Projected loss

<table><tr><td>Cluster</td><td>Label</td><td>Contexts closest to medoid</td></tr><tr><td>1</td><td>Token after &lt;,&gt;</td><td>the Roman goddess Ceres\n Occator (crater), a crater on the planet Ceres,knownof his first novel, The Watsons Go to Birmingham. Bud Caldwell, the main character,travelsbuilt, bypassing Fingal. Later it was joined by the Pere Marquette railway,boost</td></tr><tr><td>2</td><td>Repeatedtokens</td><td></td></tr></table>

(iii) Cluster data examples

![](images/2ebfc6aa7e4ce06e6a1e4cd41898416f6e505db7de8c55794aa2e7f3e06ac269.jpg)

<details>
<summary>line</summary>

| Iteration | Cluster 1 | Cluster 2 |
| --------- | --------- | --------- |
| 0         | 11.0      | 11.0      |
| 5000      | 5.0       | 0.0       |
| 10000     | 4.5       | 0.0       |
</details>

(ii) Exact loss   
(b) POLCA clusters from basis vector 29.   
Figure 11: English language modeling data clusters with POLCA. We compute breakthrough clustering on POLCA trajectories for each vector and report the average decomposed POLCA trajectories (11ai and 11bi). Figures 11aii and 11bii show the average of the per-token loss trajectories for each of the clusters found using the POLCA trajectories. For each cluster, we provide a label based on the top tokens in the cluster and the top 10 contexts closest to its medoid. We then report the 3 contexts closest to the cluster medoid. Clustering on the decomposed POLCA trajectories reveals breakthroughs at points in training where the per-token loss curve remains smooth.