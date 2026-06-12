# FAST ESTIMATION OF WASSERSTEIN DISTANCES VIA REGRESSION ON SLICED WASSERSTEIN DISTANCES

Khai Nguyen∗

Department of Statistics and Data Sciences

University of Texas at Austin

Austin, TX 78713, USA

khainb@utexas.edu

Hai Nguyen∗

Independent Researcher

namhai283287@gmail.com

Nhat Ho

Department of Statistics and Data Sciences

University of Texas at Austin

Austin, TX 78713, USA

minhnhat@utexas.edu

# ABSTRACT

We address the problem of efficiently computing Wasserstein distances for multiple pairs of distributions drawn from a meta-distribution. To this end, we propose a fast estimation method based on regressing Wasserstein distance on sliced Wasserstein (SW) distances. Specifically, we leverage both standard SW distances, which provide lower bounds, and lifted SW distances, which provide upper bounds, as predictors of the true Wasserstein distance. To ensure parsimony, we introduce two linear models: an unconstrained model with a closed-form least-squares solution, and a constrained model that uses only half as many parameters. We show that accurate models can be learned from a small number of distribution pairs. Once estimated, the model can predict the Wasserstein distance for any pair of distributions via a linear combination of SW distances, making it highly efficient. Empirically, we validate our approach on diverse tasks, including Gaussian mixtures, pointcloud classification, and Wasserstein-space visualizations for 3D point clouds. Across various datasets such as MNIST point clouds, ShapeNetV2, MERFISH Cell Niches, and scRNA-seq, our method consistently provides a better approximation of Wasserstein than the state-of-the-art method, Wasserstein Wormhole, and classical methods, particularly in low-data regimes. To illustrate its robustness, we also experiment the method with intra- and inter-class settings. Finally, we demonstrate that RG can accelerate Wasserstein Wormhole training, yielding RG-Wormhole.

# 1 INTRODUCTION

Optimal Transport (OT) and Wasserstein distances (Villani, 2009; Peyré & Cuturi, 2019) have become essential tools in machine learning, widely used for quantifying the similarity or dissimilarity between probability distributions. Fundamentally, the Wasserstein distance measures the minimum cost required to "transport" mass from one distribution to another, effectively capturing the underlying geometry of the data. Thanks to their clear geometric interpretation and mathematical robustness, Wasserstein distances have found applications across various fields, such as generative modeling Genevay et al. (2018), computational biology Bunne et al. (2023), chemistry Wu et al. (2023), and image processing Feydy et al. (2017). Despite its utility, computing the exact Wasserstein distance is computationally expensive. It typically requires solving a large-scale linear program to find an optimal transport plan, with a time complexity of $\mathcal { O } ( n ^ { 3 } \log { n } )$ for discrete distributions of size n. This high cost severely limits its use in large-scale or real-time settings.

In many applications, Wasserstein distances are computed (repeatedly) for many pairs of distributions instead of a single pair, e.g., dataset comparisons (Alvarez-Melis & Fusi, 2020), 3D point-cloud autoencoder (Achlioptas et al., 2018), point-cloud nearest neighbor classification/regression (Rubner et al., 1998), learning embeddings for distributions (Kolouri et al., 2021), density-density regression (Chen et al., 2023), and so on. Therefore, the high computational complexities of the Wasserstein distance become the main bottleneck to scaling up these applications. As a result, speeding up the computation of the Wasserstein distance has become a vital task in practice.

To address this bottleneck, a straightforward improvement is to speed up the computation of the Wasserstein distance. For example, entropic regularization (Cuturi, 2013) enables fast approximation via Sinkhorn iterations, while other methods exploit the structure in the transport plan, such as lowrank approximations (Scetbon et al., 2021). In addition, some approaches rely on strong structural assumptions, such as the Bures-Wasserstein metric (Dowson & Landau, 1982) gives a closed-form solution for the exact 2-Wasserstein distance (W2) under the Gaussian assumption on distributions.

Another approach is to cast computing Wasserstein distances for many pairs of distributions as a learning problem, i.e., learning a model first to predict the Wasserstein distance given any pair of distributions, then use the model later for the mentioned downstream tasks. For example, Deep Wasserstein Embedding (DWE) (Courty et al., 2018) trains a Siamese convolutional network to match OT distances between 2D images, while Wasserstein Wormhole (Haviv et al., 2024) employs transformer-based architectures to learn embeddings of distributions, allowing Euclidean distances in the learned space to approximate Wasserstein distances efficiently. While effective, these deep learning-based methods require significant computational resources and time to train, and their performance may degrade when limited training data are available. Moreover, these approaches are limited to empirical distributions because of the use of neural networks.

In this work, we propose a novel approach to predict the Wasserstein distance without relying on any neural networks or learned embeddings. Moreover, the proposed approach relies on a parsimonious model and can handle both continuous and discrete distributions. In particular, we propose to regress the Wasserstein distance on sliced Wasserstein (SW) distances (Rabin et al., 2010; Mahey et al., 2023; Nguyen & Ho, 2023; Liu et al., 2025; Deshpande et al., 2019; Rowland et al., 2019). In greater detail, we introduce linear models with Wasserstein distances as the response and SW distances as the predictors. We provide estimates of the models via efficient least-squares estimates. In addition, since sliced Wasserstein distances have low computational complexity, the resulting Wasserstein regressor is computationally efficient.

# Contribution: In summary, our main contributions are three-fold:

1. We introduce the first regression framework where the Wasserstein distance serves as the response variable and various sliced Wasserstein (SW) distances act as predictors, in the setting of random pairs of distributions. This framework not only uncovers the relationship between the Wasserstein distance and its SW-based approximations but also enables efficient estimation of the Wasserstein distance. Specifically, we use SW distance (Bonneel et al., 2015), Max-SW (Deshpande et al., 2019), and energy-based SW (Nguyen & Ho, 2023), all of which provide lower bounds on the Wasserstein distance, as predictors. In addition, we incorporate lifted SW distances, which provide upper bounds, including projected Wasserstein (Rowland et al., 2019), minimum SW generalized geodesics (Mahey et al., 2023), and expected sliced distance (Liu et al., 2025).   
2. We propose two linear models for the regression problem and describe their estimation via least-squares. The first model is unconstrained and admits a closed-form least-squares solution. The second model incorporates constraints that leverage the known bounds between SW distances and the Wasserstein distance, thereby reducing the number of parameters by half. Based on these estimations, we obtain a fast method to approximate the Wasserstein distance for any pair of distributions, with the same computational complexity as that of computing SW distances.   
3. Empirically, we demonstrate that our approach yields accurate estimates of the Wasserstein distance, particularly in low-data regimes. We first evaluate its accuracy through simulations with Gaussian mixtures. We then apply the estimated distances to visualize distributional data and to perform k-NN classification on ShapeNetV2 point clouds. Next, we benchmark our method against Wasserstein Wormhole, the state-of-the-art Wasserstein embedding model, across four datasets of increasing dimensionality: MNIST point clouds, ShapeNetV2, MERFISH cell niches, and scRNA-seq.

Finally, we propose RG-Wormhole, a variant of Wasserstein Wormhole that replaces its Wasserstein computations with our estimates, preserving accuracy while substantially reducing training time.

Organization. Section 2 reviews preliminaries on the Wasserstein distance, its sliced variants, and their computation. Section 3 introduces our regression framework for approximating Wasserstein distances from sliced variants, together with both constrained and unconstrained linear models. Section 4 reports the experimental results. The appendices provide supplementary experiments (mixtures of Gaussians and distributional space visualizations), detailed experimental settings, theoretical proofs, and additional related work.

Notations. For any $d \geq 2 .$ , let $\mathbb { S } ^ { d - 1 } : = \{ \theta \in \mathbb { R } ^ { d } : \| \theta \| _ { 2 } = 1 \}$ denote the unit sphere in $\mathbb { R } ^ { d }$ , and let ${ \cal { U } } ( \mathbb { S } ^ { d - 1 } )$ denote the uniform distribution on it. For $p \geq 1$ , we write $\mathcal { P } _ { p } ( \mathcal { X } )$ for the set of all probability measures on X with the finite p th moment. Given two sequences $a _ { n }$ and $b _ { n } .$ the notation $a _ { n } = { \mathcal { O } } ( b _ { n } )$ means that $a _ { n } \leq C b _ { n }$ for all $n \geq 1$ , for some universal constant $C > 0$ . For a measurable map $P _ { - }$ , the notation $P \sharp \mu$ denotes the push-forward of $\mu$ through $P .$ . Additional notation will be introduced as needed.

# 2 PRELIMINARIES

We first review definitions and computational aspects of the Wasserstein distance and its related properties in one dimension.

Wasserstein distance. Wasserstein-p $( p \ge 1 )$ distance Villani (2008); Peyré et al. (2019) between two distributions $\mu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ and $\nu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ ) (dimension $d \geq 1 )$ is defined as:

$$
W _ {p} ^ {p} (\mu , \nu) = \inf _ {\pi \in \Pi (\mu , \nu)} \int_ {\mathbb {R} ^ {d} \times \mathbb {R} ^ {d}} \| x - y \| _ {p} ^ {p} d \pi (x, y), \tag {1}
$$

where $\begin{array} { r } { \Pi ( \mu , \nu ) = \{ \pi \in \mathcal { P } ( \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } ) \} \mid \int _ { \mathbb { R } ^ { d } } d \pi ( x , y ) = \mu ( x ) , \int _ { \mathbb { R } ^ { d } } d \pi ( x , y ) = \nu ( y ) \} } \end{array}$ is the set of all transportation plans i.e., joint distributions which have marginals be two comparing distributions. When and ν are discrete distributions i.e., $\begin{array} { r } { \mu = \sum _ { i = 1 } ^ { n } \alpha _ { i } \delta _ { x _ { i } } \mathbf { \bar { ( } } n \geq 1 ) } \end{array}$ and $\textstyle \nu = \dot { \sum _ { j = 1 } ^ { m } \beta _ { j } \delta _ { y _ { j } } } ( m \geq 1 )$ m $\mu$ where $\begin{array} { r } { \sum _ { i = 1 } ^ { N } \alpha _ { i } = \sum _ { j = 1 } ^ { m } \beta _ { j } = 1 } \end{array}$ and $\alpha _ { i } \geq 0 , \beta _ { j } \geq 0$ for all $i = 1 , \ldots , n$ and $j = 1 , \dots , m$ Wasserstein distance between $\mu$ and ν defined as:

$$
W _ {p} ^ {p} (\mu , \nu) = \min _ {\gamma \in \Gamma (\alpha , \beta)} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \| x _ {i} - y _ {j} \| _ {p} ^ {p} \gamma_ {i j},
$$

where $\Gamma ( \alpha , \beta ) = \lbrace \gamma \in \mathbb { R } _ { + } ^ { n \times m } \ \vert \ \gamma \mathbf { 1 } = \alpha , \gamma ^ { \top } \mathbf { 1 } = \beta \rbrace$ . Without loss of generality, we assume that $n \geq m$ . Therefore, the time complexity for solving this linear programming is $\mathcal { O } ( n ^ { 3 } \log n )$ Peyré & Cuturi (2019) and $\mathcal { O } ( n ^ { 2 } )$ ), which are expensive.

One-dimensional Case. When $d = 1$ , the Wasserstein distance can be efficiently calculated. For the continuous case, Wasserstein-2 distance has the following form: $\begin{array} { r } { W _ { p } ^ { p } ( \mu , \nu ) = \int _ { 0 } ^ { 1 } | F _ { \mu } ^ { - 1 } ( t ) - } \end{array}$ $F _ { \nu } ^ { - 1 } ( t ) | ^ { p } d t$ , where $F _ { \mu } ^ { - 1 }$ and $F _ { \nu } ^ { - 1 }$ denote the quantile functions of $\mu$ and ν respectively. Here, the $\pi _ { ( \mu , \nu ) } = ( F _ { \mu } ^ { - 1 } , F _ { \nu } ^ { - 1 } ) \sharp \mathcal { U } ( [ 0 , 1 ] )$ . When tile func $\mu$ and νons of re discrete distributions, i.e.and ν are: $\begin{array} { r } { \mu = \sum _ { i = 1 } ^ { n } \alpha _ { i } \delta _ { x _ { i } } \left( n \geq \ddot { 1 } \right) } \end{array}$ $\begin{array} { r } { \nu = \sum _ { j = 1 } ^ { m } \beta _ { j } \delta _ { y _ { j } } } \end{array}$ $\mu$

$$
F _ {\mu} ^ {- 1} (t) = \sum_ {i = 1} ^ {n} x _ {(i)} I \left(\sum_ {j = 1} ^ {i - 1} \alpha_ {(j)} <   t \leq \sum_ {j = 1} ^ {i} \alpha_ {(j)}\right), F _ {\nu} ^ {- 1} (t) = \sum_ {j = 1} ^ {m} y _ {(j)} I \left(\sum_ {i = 1} ^ {j - 1} \beta_ {(i)} <   t \leq \sum_ {i = 1} ^ {j} \beta_ {(i)}\right),
$$

where $x _ { ( 1 ) } \ \leq \ . . . \ \leq \ x _ { ( n ) }$ and $y _ { ( 1 ) } \leq \ldots \leq y _ { ( m ) }$ are the sorted supports (or order statistics). Therefore, the one-dimensional Wasserstein distance can be computed in ${ \mathcal { O } } ( n \log n )$ in time and ${ \mathcal { O } } ( n )$ in space (assuming that $n > m )$ .

Random Projection. A key technique that plays a vital role in later discussion is random projection. We consider a function $\dot { P _ { \theta } } : \mathbb { R } ^ { d } \stackrel { \cdot } {  } \mathbb { R }$ where $\theta \sim \sigma ( \theta ) ( \sigma ( \theta ) \sim \mathcal { P } ( \mathbb { S } ^ { d - 1 } ) )$ ) is a random variable. For simplicity, we consider the traditional setup where $\begin{array} { r } { \dot { \theta } \sim \dot { \mathcal { U } } ( \mathbb { S } ^ { d - 1 } ) } \end{array}$ and $\dot { P } _ { \theta } ( x ) = \langle \theta , x \rangle$ (Bonneel et al., 2015; Rabin et al., 2012). However, the following discussion holds for any other types of projections (Kolouri et al., 2019; Bonet et al., 2023b; 2025; 2023c). For $\mu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ and $\nu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ , one-dimensional projected Wasserstein distance with $P _ { \theta }$ is defined as:

$$
\underline {{{W}}} _ {p} ^ {p} (\mu , \nu ; P _ {\theta}) = W _ {p} ^ {p} (P _ {\theta} \sharp \mu , P _ {\theta} \sharp \nu) = \int_ {0} ^ {1} | F _ {P _ {\theta} \sharp \mu} ^ {- 1} (t) - F _ {P _ {\theta} \sharp \nu} ^ {- 1} (t) | ^ {p} d t. \tag {2}
$$

The second approach to construct a Wasserstein-type discrepancy from one-dimensional projection is using lifted transportation plan. There are many ways to construct such lifted plan using disintegration of measures (Muzellec & Cuturi, 2019; Tanguy et al., 2025). In practice, the most used way (Liu et al., 2025; Tanguy et al., 2025) is:

$$
\overline {{W}} _ {p} ^ {p} (\mu , \nu ; P _ {\theta}) = \int_ {\mathbb {R} ^ {d} \times \mathbb {R} ^ {d}} \| x - y \| _ {p} ^ {p} d \pi^ {\theta} (x, y) \tag {3}
$$

$$
= \int_ {\mathbb {R} \times \mathbb {R}} \int_ {P _ {\theta} ^ {- 1} (t _ {1}) \times P _ {\theta} ^ {- 1} (t _ {2})} \| x - y | _ {p} ^ {p} d \mu_ {t _ {1}} \otimes \nu_ {t _ {2}} (x, y) d \pi_ {(P _ {\theta} \sharp \mu , P _ {\theta} \sharp \nu)} (t _ {1}, t _ {2}), \tag {4}
$$

where $\pi ^ { \theta } \in \Pi ( \mu , \nu )$ is the lifted transportation plan, $\pi _ { \left( P _ { \theta } \sharp \mu , P _ { \theta } \sharp \nu \right) }$ is the optimal transport plan between $P _ { \theta } \sharp \mu$ and $P _ { \theta } \sharp \nu , \mu _ { t _ { 1 } }$ and $\nu _ { t _ { 2 } }$ are disintegration of $\mu$ and ν at $t _ { 1 }$ and $t _ { 2 }$ the function $P _ { \theta }$ , and $\otimes$ denotes the product of measures. When dealing with discrete measures $\mu$ and $\nu , \overline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta } )$ can still be computed efficiently (Mahey et al., 2023; Liu et al., 2025) i.e., O(n log n) in time and ${ \mathcal { O } } ( n )$ in space (assumed that $n > m )$ . The quantity $\overline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta } )$ is known as lifted cost (Tanguy et al., 2025) or sliced Wasserstein generalized geodesic (Mahey et al., 2023; Liu et al., 2025). From previous work (Nguyen & Ho, 2023; Mahey et al., 2023; Tanguy, 2023), we know the following relationship ${ \underline { { W } } _ { p } } ( \bar { \mu } , \bar { \nu } ; P _ { \theta } ) \le W _ { p } ( \mu , \nu ) \le \overline { { W } } _ { p } ( \mu , \nu ; P _ { \theta } )$ .

# 3 REGRESSION OF WASSERSTEIN DISTANCE ONTO SLICED OPTIMAL TRANSPORT DISTANCES

In this section, we present a framework for regressing the Wasserstein distance onto sliced Wasserstein distances, propose some models, and discuss related computational properties.

# 3.1 SLICED WASSERSTEIN AND LIFTED SLICED WASSERSTEIN

Sliced Wasserstein distances. Given $\mu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ and $\nu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ , a sliced Wasserstein-p distance can be defined as follows (Rabin et al., 2012; Nguyen, 2025):

$$
S W _ {p} ^ {p} (\mu , \nu ; \sigma) = \mathbb {E} _ {\theta \sim \sigma} \left[ \underline {{W}} _ {p} ^ {p} (\mu , \nu ; P _ {\theta}) \right], \tag {5}
$$

where $P _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R }$ is the projection function, $\underline { W } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta } )$ is the one-dimensional projected Wasserstein distance (equation 2), and $\sigma \in \mathcal { P } ( \mathbb { S } ^ { d - 1 } )$ is the slicing distribution. By changing the slicing distribution, we can obtain variants of SW. There are three main ways:

1. Fixed prior: The simplest way is to choose σ to be a fixed and known distribution, e.g., the uniform distribution ${ \cal { U } } ( \mathbb { S } ^ { d - 1 } )$ as in the conventional SW (Rabin et al., 2012).   
2. Optimization-based: We can also find σ that prioritizes some realizations of θ that satisfies a notion of informativeness. For example, σ can put more masses to realizations of θ where $\underline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta } )$ have high value, i.e., setting informativeness as discriminativeness. For example, we can find σ by solving (Nguyen et al., 2021):

$$
\sup _ {\sigma \in \mathcal {M} (\mathbb {S} ^ {d - 1})} \mathbb {E} _ {\theta \sim \sigma} [ \underline {{W}} _ {p} ^ {p} (\mu , \nu ; P _ {\theta}) ],
$$

where $\mathcal { M } ( \mathbb { S } ^ { d - 1 } ) \subset \mathcal { P } ( \mathbb { S } ^ { d - 1 } )$ be a set of probability measures on $\mathbb { S } ^ { d - 1 }$ . When $\mathcal { M } ( \mathbb { S } ^ { d - 1 } ) = \{ \delta _ { \theta } \ \vert$ $\theta \in \mathbb { S } ^ { d - 1 } \}$ , max sliced Wasserstein distance (Deshpande et al., 2019) is obtained: $\operatorname { M a x } - S W ( \mu , \nu ) =$ $\operatorname* { m a x } _ { \theta \in \mathbb { S } ^ { d - 1 } } \underline { { W } } _ { p } ( \mu , \nu ; P _ { \theta } ) ]$ .

3. Energy-based: An optimization-free way to select σ is to design it as an energy-based distribution with the unnormalized density: $p _ { \sigma } ( \theta ) \propto f ( \underline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta } ) )$ , where f is often chosen to be an increasing function on the positive real line, i.e., an exponential function. This choice of slicing distribution leads to energy-based SW (EBSW) (Nguyen & Ho, 2023).

Empirical estimation. For SW, Monte Carlo estimation is used to approximate the distance:

$$
\widehat {S W} _ {p} ^ {p} (\mu , \nu ; \theta_ {1}, \dots , \theta_ {L}) = \frac {1}{L} \sum_ {l = 1} ^ {L} \underline {{W}} _ {p} ^ {p} (\mu , \nu ; P _ {\theta_ {l}}),
$$

where $\theta _ { 1 } , \dots , \theta _ { L } \overset { i . i . d } { \sim } \mathcal { U } ( \mathbb { S } ^ { d - 1 } ) \left( L > 0 \right)$ are projecting directions (other sampling techniques can also be used (Nguyen et al., 2024; Nguyen & Ho, 2024; Sisouk et al., 2025)). For Max-SW, we can use ${ \hat { \theta } } _ { T }$ which is the solution of an optimization algorithm with $T > 0$ iterations, e.g., projected gradient ascent (Nietert et al., 2022) or Riemannian gradient ascent Lin et al. (2020): $\widehat { \mathbf { M a x } { - } S W } _ { p } ^ { p } ( \mu , \nu ; \hat { \theta } _ { T } ) = W _ { p } ^ { p } ( \mu , \nu ; P _ { \hat { \theta } _ { T } } )$ . For EBSW, one simple way to estimate the distance is to use importance sampling:

$$
\widehat {E B S W} _ {p} ^ {p} (\mu , \nu ; \theta_ {1}, \ldots , \theta_ {L}) = \sum_ {l = 1} ^ {L} \hat {w} _ {l} \underline {{W}} _ {p} ^ {p} (\mu , \nu ; P _ {\theta_ {l}}),
$$

where wˆl = $\begin{array} { r } { \hat { w } _ { l } = \frac { f ( \underline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta _ { l } } ) ) } { \sum _ { l ^ { \prime } = 1 } ^ { L } f ( \underline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta _ { l ^ { \prime } } } ) } } \end{array}$ and $\theta _ { 1 } , \dots , \theta _ { L } \sim \mathcal { U } ( \mathbb { S } ^ { d - 1 } )$ .

Lower bounds. We summarize the connection between SW, Max-SW, EBSW, and Wasserstein distance in the following remark. The detail of the proof can be found in Nguyen & Ho (2023).

Remark 1. Given any $\mu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ and $\nu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ , we have:

$( a ) \ S W _ { p } ( \mu , \nu ) \leq E B S W _ { p } ( \mu , \nu ) \leq M a x { - } S W _ { p } ( \mu , \nu ) \leq W _ { p } ( \mu , \nu ) ,$   
$( b ) \ \widehat { S W } _ { p } ( \mu , \nu ; \theta _ { 1 } , \ldots , \theta _ { L } ) \leq \widehat { E B S W } _ { p } ( \mu , \nu ; \theta _ { 1 } , \ldots , \theta _ { L } ) \leq W _ { p } ( \mu , \nu ) f o r \ a n y \theta _ { 1 } , \ldots , \theta _ { L } \in \mathbb { S } ^ { d - 1 } ,$   
(c $) \widehat { M a x { - } S W } _ { p } ^ { p } ( \mu , \nu ; \hat { \theta } _ { T } ) \leq W _ { p } ( \mu , \nu ) f o r a n y \hat { \theta } _ { T } \in \mathbb { S } ^ { d - 1 } .$

Lifted sliced Wasserstein distances. Given $\mu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ and $\nu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ , a lifted sliced Wasserstein-$p$ distance can be defined as follows (Rowland et al., 2019):

$$
L S W _ {p} ^ {p} (\mu , \nu ; \sigma) = \mathbb {E} _ {\theta \sim \sigma} \left[ \overline {{W}} _ {p} ^ {p} (\mu , \nu ; P _ {\theta}) \right], \tag {6}
$$

where $P _ { \theta } : \mathbb { R } ^ { d } $ R is the projection function, $\overline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta } )$ is the SWGG (equation 3), and $\sigma \in \mathcal { P } ( \mathbb { S } ^ { d - 1 } )$ is the slicing distribution. Similar to SW, we can obtain variants of PW by choosing σ.

1. Fixed prior: The original LSW is introduced as in projected Wasserstein (PW) in Rowland et al. (2019), which uses the uniform distribution ${ \cal { U } } ( \mathbb { S } ^ { d - 1 } )$ .   
2. Optimization-based: In contrast to the case of one-dimensional projected Wasserstein, which is always a lower bound of Wasserstein distance, SWGG is always an upper bound of Wasserstein distance. Therefore, it is desirable to select θ that can minimize the corresponding lifted cost, that leads to min SWGG distance:

$$
\operatorname{Min-} S W G G _ {p} (\mu , \nu) = \min _ {\theta \in \mathbb {S} ^ {d - 1}} \overline {{W}} _ {p} (\mu , \nu ; P _ {\theta})
$$

3. Energy-based: Similar to the case of EBSW, authors in Liu et al. (2025) proposes to choose σ as an energy-based distribution with the unnormalized density: $p _ { \sigma } ( \theta ) \propto f ( - \underline { { \bar { W } _ { p } ^ { p } } } ( \mu , \nu ; P _ { \theta } ) )$ ), where $f$ is often chosen to be an exponential function with temperature. The authors name the distance as expected sliced transport (EST).

Empirical estimation. For PW, Monte Carlo samples are used to approximate the distance: $\begin{array} { r } { \widehat { P W } _ { p } ^ { p } ( \mu , \nu ; \theta _ { 1 } , \dots , \theta _ { L } ) = \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \overline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta _ { l } } ) } \end{array}$ , where $\theta _ { 1 } , \dots , \theta _ { L } \overset { i . i . d } { \sim } \mathcal { U } ( \mathbb { S } ^ { d - 1 } )$ . For Min-SWGG, we can use $\hat { \theta } _ { T }$ which is the solution of an optimization algorithm with $T > 0$ iterations, e.g., simulated annealing (Mahey et al., 2023), gradient ascent with a surrogate objective (Mahey et al., 2023), and differentiable approximation (Chapel et al., 2025): Min- $\widehat { { \cdot } S W } G G _ { p } ^ { p } ( \mu , \nu ; \hat { \theta } _ { T } ) =$ $\overline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \hat { \theta } _ { T } } )$ . For EST, importance sampling estimation is used:

$$
\widehat {E S T} _ {p} ^ {p} (\mu , \nu ; \theta_ {1}, \ldots , \theta_ {L})) = \sum_ {l = 1} ^ {L} \hat {w} _ {l} \overline {{W}} _ {p} ^ {p} (\mu , \nu ; P _ {\theta_ {l}}),
$$

$\begin{array} { r } { \mathrm { w h e r e ~ } \hat { w } _ { l } = \frac { f ( - \overline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta _ { l } } ) ) } { \sum _ { l ^ { \prime } = 1 } ^ { L } f ( - \overline { { W } } _ { p } ^ { p } ( \mu , \nu ; P _ { \theta _ { l ^ { \prime } } } ) } \mathrm { ~ a n d ~ } \theta _ { 1 } , \dots , \theta _ { L } \sim \mathcal { U } ( \mathbb { S } ^ { d - 1 } ) . } \end{array}$ where wˆl =

Upper bounds. We summarize the connection between PW, Min-SWGG, EST, and Wasserstein distance in the following remark. The connection between Min-SWGG, EST, and Wasserstein distance is discussed in Mahey et al. (2023); Liu et al. (2025). The connection between EST and PW can be generalized from the connection between EBSW and SW in Nguyen & Ho (2023).

Remark 2. Given any $\mu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ and $\nu \in \mathcal P _ { p } ( \mathbb { R } ^ { d } )$ , we have:

$( a ) \ : W _ { p } ( \mu , \nu ) \leq M i n { \cdot } S W G G _ { p } ( \mu , \nu ) \leq E S T _ { p } ( \mu , \nu ) \leq P W _ { p } ( \mu , \nu ) ,$   
$( b ) W _ { p } ( \mu , \nu ) \leq \widehat { E S T } _ { p } ( \mu , \nu ; \theta _ { 1 } , \ldots , \theta _ { L } ) \leq \widehat { P W } _ { p } ( \mu , \nu ; \theta _ { 1 } , \ldots , \theta _ { L } ) f o r a n y \theta _ { 1 } , \ldots , \theta _ { L } \in \mathbb { S } ^ { d - 1 } ,$   
$\left( c \right) W _ { p } ( \mu , \nu ) \leq \widehat { M i n { \mathrm { - } } S W G } G _ { p } ^ { p } ( \mu , \nu ; \widehat { \theta } _ { T } ) f o r a n y \widehat { \theta } _ { T } \in \mathbb { S } ^ { d - 1 } .$

# 3.2 REGRESSION OF WASSERSTEIN DISTANCE ON SLICED WASSERSTEIN DISTANCES

We consider the setting where we observe pairs of distributions $( \mu _ { 1 } , \nu _ { 1 } ) , \ldots , ( \mu _ { N } , \nu _ { N } ) \sim \mathbb { P } ( \mu , \nu )$ . Here, $\mathbb { P } ( \mu , \nu )$ is the meta distribution, and we are interested in relating $W _ { p } ( \mu _ { i } , \nu _ { i } )$ with $K > \mathsf { \bar { 0 } } \mathsf { S } \bar { \mathsf { W } }$ distances $S _ { p } ^ { ( 1 ) } ( \mu _ { i } , \nu _ { i } ) , \dots , S _ { p } ^ { ( K ) } ( \mu _ { i } , \nu _ { i } )$ for $i = 1 , \ldots , N .$ . We first start with a general model.

Definition 1 ( Regression of Wasserstein distance onto SW distances). Given a meta distribution $\mathbb { P } ( \mu , \nu ) \in \mathcal { P } ( \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) \times \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) ) , \ K > 0$ SW distances $S _ { p } ^ { ( 1 ) } , \ldots , S _ { p } ^ { ( K ) }$ , a regression model of Wasserstein distance onto SW distances is defined as follows:

$$
W _ {p} (\mu , \nu) = f (S _ {p} ^ {(1)} (\mu , \nu), \dots , S _ {p} ^ {(K)} (\mu , \nu)) + \varepsilon , \tag {7}
$$

where $( \mu , \nu ) \sim \mathbb { P } ( \mu , \nu ) , f \in \mathcal { F }$ is the regression function, and ε is a noise model such that $\mathbb { E } [ \varepsilon ] = 0 .$

To estimate $f ,$ one natural estimator is the least square estimate:

$$
f _ {L S E} = \arg \min _ {f \in \mathcal {F}} \mathbb {E} \left[ \left(f (S _ {p} ^ {(1)} (\mu , \nu), \dots , S _ {p} ^ {(K)} (\mu , \nu)) - W _ {p} (\mu , \nu))\right) ^ {2} \right]. \tag {8}
$$

It is worth noting that the function $f$ can be constructed in both parametric ways (e.g., deep neural networks) or non-parametric ways (e.g., using kernels). However, in order to have a simple and explainable model, we consider linear functions in this work.

Linear Regression of Wasserstein distance onto SW distances. We now propose linear estimations of Wasserstein distances from SW distances.

Definition 2 (Linear Regression of Wasserstein distance onto SW distances). Given a meta distribution $\mathbb { P } ( \mu , \nu ) \in \mathcal { P } ( \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) \times \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) )$ , K > 0 SW distances $S _ { p } ^ { ( 1 ) } , \ldots , S _ { p } ^ { ( K ) }$ , the linear regression model of Wasserstein distance onto SW distances is defined as follows:

$$
W _ {p} (\mu , \nu) = \sum_ {k = 1} ^ {K} \omega_ {k} S _ {p} ^ {(k)} (\mu , \nu) + \varepsilon , \tag {9}
$$

where $( \mu , \nu ) \sim \mathbb { P } ( \mu , \nu )$ and ε is a noise model such that $\mathbb { E } [ \varepsilon ] = 0 .$ .

Again, we use least-squares estimation to obtain an estimate of ω.

Remark 3. The least square estimator admits the following closed form:

$$
\boldsymbol {\omega} _ {L S E} = \mathbb {E} \left[ \boldsymbol {S} _ {p} (\mu , \nu) \boldsymbol {S} _ {p} (\mu , \nu) ^ {\top} \right] ^ {- 1} \mathbb {E} \left[ \boldsymbol {S} _ {p} (\mu , \nu) W _ {p} (\mu , \nu) \right], \tag {10}
$$

where $S _ { p } ( \mu , \nu ) = ( S _ { p } ^ { ( 1 ) } ( \mu , \nu ) , \ldots , S _ { p } ^ { ( K ) } ( \mu , \nu ) ) ^ { \top } .$

The detail of Remark 3 in given in Appendix A.1. In practice, we can sample $( \mu _ { 1 } , \nu _ { 1 } ) , \ldots , ( \mu _ { M } , \nu _ { M } ) \sim \mathbb { P } ( \mu , \nu )$ to approximate the expectation in equation 10. Let $\hat { \pmb { S } } \in \mathbb { R } _ { + } ^ { M \times K }$ be the SW distances matrix i.e., $\hat { \pmb { S } } _ { i k } = S _ { p } ^ { ( k ) } ( \mu _ { i } , \nu _ { i } )$ for $i = 1 , \ldots , M ,$ and $\hat { W } \in \mathbb { R } _ { + } ^ { M }$ be the Wasserstein distances vector i.e., $\hat { W } _ { i } = W _ { p } ( \mu _ { i } , \nu _ { i } ) \mathrm { f o r } i = 1 , \dots , M$ , we have the sample-based least-squares estimate: $\hat { \omega } _ { L S E } = ( \hat { S } ^ { \top } \hat { S } ) ^ { - 1 } \hat { S } ^ { \top } \hat { W }$ , which is an unbiased estimate of ω. It is well-known that the linear model can be seen as L2 projection of the Wasserstein distances vector $\hat { W }$ onto the linear span of the SW distances vectors $\hat { S } ^ { ( \bar { 1 } ) } , \ldots , \hat { S } ^ { ( K ) }$ . We illustrate the idea in the left figure in Figure 1.

![](images/5d1ba6fbd2eda9610b5cee5c1b0855abf1bf592319c7bdaf8b36ca052e9077d9.jpg)

<details>
<summary>text_image</summary>

Span{\hat{S}^{(1)},\ldots,\hat{S}^{(K)}}
\hat{W}
\hat{W}-\omega^{\top}\hat{S}
\omega^{\top}\hat{S}
</details>

![](images/fd5ea58ee8f6d9397c5c6e894d4670147281ae389edb7f3aacb344a61f034f60.jpg)

<details>
<summary>text_image</summary>

SLp(μ,ν) Wp(μ,ν) SUp(μ,ν)
ωSLp(μ,ν) + (1 - ω)SUp(μ,ν)
</details>

Figure 1: Linear regression of the Wasserstein distance vector $\hat { W }$ on sliced Wasserstein (SW) distances $\hat { \pmb { S } } ^ { ( 1 ) } , \dots , \hat { \pmb { S } } ^ { ( K ) }$ . The left figure illustrates a linear model, interpreted as the $\mathbb { L } _ { 2 }$ projection of the Wasserstein distance onto the linear span of the SW distances. The right figure depicts a special case of a constrained linear model with only two SW distances as predictors, which can be seen as a midpoint method.

From Section 3.1, we know that SW distances are either lower bounds or upper bounds of Wasserstein distance. Therefore, natural estimation can be formed using midpoint method. In particular, given a lower bound $S L _ { p } ( \mu , \nu )$ and a upper bound $S U _ { p } ( \mu , \nu )$ , we can predict the Wasserstein distance as $\omega _ { 1 } S L _ { p } ( \mu , \nu ) + \dot { \omega } _ { 2 } S U _ { p } ( \mu , \nu )$ ) with $0 \leq \omega _ { 1 } \leq 1$ and $\omega _ { 2 } = 1 - \omega _ { 1 }$ .

Definition 3 (Constrained Linear Regression of Wasserstein distance onto SW distances). Given a meta distribution $\mathbb { P } ( \mu , \nu ) \in \mathcal { P } ( \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) \times \mathcal { P } _ { p } ( \mathbb { R } ^ { d } ) )$ ), $K > 0$ SW distances $S L _ { p } ^ { ( 1 ) } , \ldots , S L _ { p } ^ { ( K ) }$ (1)p , . . . , SL(K)p w hich are lower bounds of $W _ { p }$ and $K > 0$ SW distances $S U _ { p } ^ { ( 1 ) } , \ldots , S U _ { p } ^ { ( K ) }$ which are lower bounds of $W _ { p } ,$ the constrained linear regression model is defined as follows:

$$
W _ {p} (\mu , \nu) = \frac {1}{K} \sum_ {k = 1} ^ {K} \omega_ {k} S L _ {p} ^ {(k)} (\mu , \nu) + \frac {1}{K} \sum_ {k = 1} ^ {K} (1 - \omega_ {k}) S U _ {p} ^ {(k)} (\mu , \nu) + \varepsilon , \tag {11}
$$

where $0 \leq \omega _ { k } \leq 1 , ( \mu , \nu ) \sim { \mathbb P } ( \mu , \nu )$ and ε is a noise model such that $\mathbb { E } [ \varepsilon ] = 0$ .

To estimate $\omega = ( \omega _ { 1 } , \ldots , \omega _ { K } )$ under the constrained model, we again form the least square estimate, which can be solved using quadratic programming and Monte Carlo estimation. In a special case where $K = 1 , \operatorname { i . e . }$ , having one lower bound and one upper bound, we can have a closed-form.

Remark 4. For the case $K = 1$ with a lower bound $S L _ { p } ( \mu , \nu )$ and an upper bound $S U _ { p } ( \mu , \nu )$ , a closed-form of the least square estimate under the constrained model can be formed:

$$
\hat {\omega} _ {C L S E} = \frac {\mathbb {E} \left[ (S U _ {p} (\mu , \nu) - S L _ {p} (\mu , \nu)) (S U _ {p} (\mu , \nu) - W _ {p} (\mu , \nu)) \right]}{\mathbb {E} [ (S U _ {p} (\mu , \nu) - S L _ {p} (\mu , \nu) ^ {2} ]}. \tag {12}
$$

The detail of Remark 4 in given in Appendix A.2. The corresponding sample-based estimator for the model is: ωˆCLSE = 1M Pmi=1(SUp(µi,νi)−SLp(µi,νi))(SUp(µi,νi)−Wp(µi,νi))1 M . $\begin{array} { r } { \hat { \omega } _ { C L S E } = \frac { \frac { 1 } { M } \sum _ { i = 1 } ^ { \overline { { m } } } ( S U _ { p } ( \mu _ { i } , \nu _ { i } ) - S L _ { p } ( \mu _ { i } , \nu _ { i } ) ) ( S U _ { p } ( \mu _ { i } , \nu _ { i } ) - W _ { p } ( \mu _ { i } , \nu _ { i } ) ) } { \frac { 1 } { M } \sum _ { i = 1 } ^ { M } ( S U _ { p } ( \mu _ { i } , \nu _ { i } ) - S L _ { p } ( \mu _ { i } , \nu _ { i } ) ^ { 2 } } } \end{array}$ We show the idea in the right figure in Figure 1. Compared to the unconstrained model, the constrained model has half of the parameters. In addition, it adds inductive bias to the model, which is often helpful when having limited observed samples.

Wasserstein Distance Estimation with Few-Shot Regression. We recall that we observe $( \mu _ { 1 } , \nu _ { 1 } ) , \ldots , ( \mu _ { N } , \nu _ { N } ) \sim \mathbb { P } ( \mu , \nu )$ in practice. It is not computationally efficient to compute the discussed least square estimates using all N pairs of distributions since those estimates require evaluation of Wasserstein distances. We then sample a subset $( \mu _ { 1 } ^ { \prime } , \nu _ { 1 } ^ { \prime } ) , \dots , ( \mu _ { N } ^ { \prime } , \nu _ { M } ^ { \prime } )$ from the original set with $M < < N$ . After obtaining an estimate ωˆ from $( \mu _ { 1 } ^ { \prime } , \nu _ { 1 } ^ { \prime } ) , \dots , ( \mu _ { N } ^ { \prime } , \nu _ { M } ^ { \prime } )$ , we can form estimations of the Wasserstein distances for other pairs and any new pair of distributions given their SW distances.

Computational complexities. We assume that N pairs of distributions have the number of supports be at most n and in d dimensions. For fitting the estimate on M pairs, we need to compute MK SW distances (using L projecting directions) which costs $\mathcal { O } ( M \dot { K } L n ( \log n + d ) )$ in time and M

Table 1: k-NN accuracy on point-cloud classification on ShapeNetV2 dataset. 

<table><tr><td>Methods</td><td> $R^{2}$ </td><td>k=1</td><td>k=3</td><td>k=5</td><td>k=10</td><td>k=15</td></tr><tr><td>WD</td><td>-</td><td>83.6% ± 0.0%</td><td>83.5% ± 0.0%</td><td>84.2% ± 0.0%</td><td>82.9% ± 0.0%</td><td>79.2% ± 0.0%</td></tr><tr><td>RG-s</td><td>0.868 ± 0.02</td><td>82.1% ± 0.1%</td><td>81.7% ± 0.1%</td><td>80.8% ± 0.1%</td><td>79.4% ± 0.2%</td><td>75.5% ± 0.2%</td></tr><tr><td>RG-e</td><td>0.926 ± 0.04</td><td>82.5% ± 0.1%</td><td>82.2% ± 0.1%</td><td>80.9% ± 0.2%</td><td>79.6% ± 0.3%</td><td>75.7% ± 0.3%</td></tr><tr><td>RG-o</td><td>0.774 ± 0.38</td><td>65.1% ± 0.3%</td><td>67.7% ± 0.3%</td><td>67.6% ± 0.5%</td><td>66.7% ± 0.5%</td><td>66.0% ± 0.5%</td></tr><tr><td>RG-se</td><td>0.935 ± 0.02</td><td>82.5% ± 0.4%</td><td>82.2% ± 0.4%</td><td>82.6% ± 0.5%</td><td>81.9% ± 0.5%</td><td>76.5% ± 0.5%</td></tr><tr><td>RG-seo</td><td>0.937 ± 0.01</td><td>82.8% ± 0.4%</td><td>83.3% ± 0.5%</td><td>83.5% ± 0.7%</td><td>82.3% ± 0.7%</td><td>77.9% ± 0.7%</td></tr></table>

Wasserstein distances which costs $\mathcal { O } ( M n ^ { 2 } ( n \log n + d ) )$ ). Computing the least square estimate has the time complexity of $\mathcal { O } ( M K ^ { 2 } + \dot { K ^ { 3 } } )$ . Then, we compute $( N ^ { \bullet } - M ) K$ SW distances which costs $\mathcal { O } ( ( N - M ) \dot { K } L n ( \log n { + } \dot { d } ) )$ and predict $( N { - } M )$ Wasserstein distances which costs $\mathcal { O } ( ( N - M ) K )$ . Total time complexity is O(N KLn(log $\iota + d ) ) + M n ^ { 2 } ( n \log n + d ) ) + M K ^ { 2 } + K ^ { 3 } + ( N - M ) K )$ compared to $\mathcal { \hat { O } } ( N n ^ { \bar { 2 } } ( n \log n + d ) )$ of computing Wasserstein distances for all N pairs.

Extensions on regression. In this work, we focus on regressing the Wasserstein-p distance. If other ground metrics are used $\mathrm { e . g . }$ , geodesic distances on manifolds, variants of SW distances such as spherical sliced Wasserstein distances (Bonet et al., 2023a; Tran et al., 2024; Quellmalz et al., 2023), hyperbolic sliced Wasserstein distances (Bonet et al., 2023b), sliced Wasserstein for distributions over positive definite matrices (Bonet et al., 2023c), and other non-linear variants of sliced Wasserstein (Bonet et al., 2025; Chapel et al., 2025; Tanguy et al., 2025; Kolouri et al., 2019). However, they might not be upper/lower bounds of the corresponding Wasserstein distances. Moreover, to incorporate uncertainty quantification, we can also perform Bayesian inference (Box & Tiao, 2011), e.g., putting a prior on the regression function.

# 4 EXPERIMENTS

We define some specific model instances: RG-o uses Max-SW and Min-SWGG as predictors; RG-s uses SW and PWD as predictors; RG-e uses EBSW and EST as predictors. We also consider two extensions: RG-se combines SW, EBSW, PWD, and EST, and RG-seo combines all six variants. For each instance, we have a constrained version and an unconstrained version as discussed.

We evaluate our methods in several parts, each with a distinct goal. First, in Section 4.1, we test practical use via k-NN on ShapeNetV2, reporting accuracy under different metrics. Second, in Section 4.2, we benchmark RG variants against Wormhole across MNIST point clouds, ShapeNetV2, MERFISH Cell Niches Zhang et al. (2021), and scRNA-seq atlas Persad et al. (2023), reporting R2/MSE/MAE in low-data regimes. Third, in Section 4.3, we combine our framework with Wormhole to introduce RG-Wormhole, a hybrid that matches Wormhole’s performance while requiring far less training time. We compare training time under varying batch sizes and epochs, as well as embedding, reconstruction, barycenter, and interpolation quality. In Appendix B.1, we run MoG simulations to verify that our methods approximate the true Wasserstein from low to high dimensions. In Appendix B.3, we visualize metric-induced geometry with UMAP McInnes et al. (2018). In Appendix B.6, we compare RG with classical methods in the many-pairs setting. To illustrate robustness, we further evaluate the method under inter-class and intra-class settings in Appendix B.7. Finally, we investigate whether there are consistent patterns in the optimal RG weights across datasets; see Appendix B.8. Throughout, N denotes the number of training-set sizes, and $M _ { 0 }$ the number of samples drawn from the training set, yielding $\begin{array} { r } { M = \frac { M _ { 0 } ( M _ { 0 } - 1 ) } { 2 } } \end{array}$ pairs used to estimate RG coefficients.

# 4.1 POINT CLOUD CLASSIFICATION

We evaluate unconstrained RG variants over a classification task over 10-class ShapeNetV2 with 500 training samples (N=500) and estimate RG weights from 10 samples $( M _ { 0 } { = } 1 0 )$ drawn from the training set. The details of the experimental setting and full results are provided in Appendix B.2.

Results. Table 1 reports k-NN accuracy on ShapeNetV2 under different metrics. As expected, WD achieves the best accuracy, with 84.2% at k=5. Among single sliced-based metrics, SW and EBSW, are the strongest, though they cap at about 72.5% top-1. Our RG methods close much of the gap to Wasserstein. Both RG-s and RG-e consistently achieve around 82.5% top-1 accuracy with high correlation to Wasserstein $( R ^ { 2 } \approx 0 . 9 )$ . The multi-metric extensions further improve stability: $R G \ / { \bf - } s e$ and RG-seo reach up to 83.5% accuracy with $R ^ { 2 }$ as high as 0.93, essentially matching Wasserstein.

Table 2: Approximation quality of Wormhole and RG variants across four datasets under a training set size of 100 samples. Each cell reports R2, MSE, and MA) with respect to the exact Wasserstein distance. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">MNIST Point Cloud</td><td colspan="3">ShapeNetV2</td><td colspan="3">MERFISH</td><td colspan="3">scRNA-seq</td></tr><tr><td> $R^2$ </td><td>MSE</td><td>MAE</td><td> $R^2$ </td><td>MSE</td><td>MAE</td><td> $R^2$ </td><td>MSE</td><td>MAE</td><td> $R^2$ </td><td>MSE</td><td>MAE</td></tr><tr><td>Wormhole</td><td>0.28</td><td> $4.3 \times 10^{-1}$ </td><td> $5.1 \times 10^{-1}$ </td><td>0.65</td><td> $6.6 \times 10^{-2}$ </td><td> $1.8 \times 10^{-1}$ </td><td>-3.6</td><td> $8.0 \times 10^{-4}$ </td><td> $2.1 \times 10^{-2}$ </td><td>0.04</td><td> $7.0 \times 10^{-3}$ </td><td> $7.8 \times 10^{-2}$ </td></tr><tr><td>RG-s (constr.)</td><td>0.84</td><td> $8.9 \times 10^{-2}$ </td><td> $2.3 \times 10^{-1}$ </td><td>0.88</td><td> $2.0 \times 10^{-2}$ </td><td> $1.1 \times 10^{-1}$ </td><td>0.91</td><td> $1.6 \times 10^{-5}$ </td><td> $3.0 \times 10^{-3}$ </td><td>1.00</td><td> $3.7 \times 10^{-5}$ </td><td> $3.0 \times 10^{-3}$ </td></tr><tr><td>RG-e (constr.)</td><td>0.86</td><td> $8.7 \times 10^{-2}$ </td><td> $2.3 \times 10^{-1}$ </td><td>0.90</td><td> $1.7 \times 10^{-2}$ </td><td> $1.0 \times 10^{-1}$ </td><td>0.92</td><td> $1.3 \times 10^{-5}$ </td><td> $3.0 \times 10^{-3}$ </td><td>1.00</td><td> $1.3 \times 10^{-5}$ </td><td> $1.0 \times 10^{-3}$ </td></tr><tr><td>RG-o (constr.)</td><td>0.77</td><td> $1.4 \times 10^{-1}$ </td><td> $2.8 \times 10^{-1}$ </td><td>0.66</td><td> $5.2 \times 10^{-2}$ </td><td> $1.8 \times 10^{-1}$ </td><td>0.75</td><td> $4.8 \times 10^{-5}$ </td><td> $6.0 \times 10^{-3}$ </td><td>0.99</td><td> $6.1 \times 10^{-5}$ </td><td> $6.0 \times 10^{-3}$ </td></tr><tr><td>RG-se (constr.)</td><td>0.84</td><td> $9.8 \times 10^{-2}$ </td><td> $2.4 \times 10^{-1}$ </td><td>0.92</td><td> $1.4 \times 10^{-2}$ </td><td> $9.3 \times 10^{-2}$ </td><td>0.91</td><td> $1.5 \times 10^{-5}$ </td><td> $3.0 \times 10^{-3}$ </td><td>1.00</td><td> $2.4 \times 10^{-5}$ </td><td> $2.0 \times 10^{-3}$ </td></tr><tr><td>RG-seo (constr.)</td><td>0.85</td><td> $9.0 \times 10^{-2}$ </td><td> $2.3 \times 10^{-1}$ </td><td>0.91</td><td> $1.7 \times 10^{-2}$ </td><td> $1.0 \times 10^{-1}$ </td><td>0.92</td><td> $1.3 \times 10^{-5}$ </td><td> $3.0 \times 10^{-3}$ </td><td>1.00</td><td> $2.2 \times 10^{-5}$ </td><td> $2.0 \times 10^{-3}$ </td></tr><tr><td>RG-s (unconstr.)</td><td>0.93</td><td> $4.5 \times 10^{-2}$ </td><td> $1.6 \times 10^{-1}$ </td><td>0.94</td><td> $1.1 \times 10^{-2}$ </td><td> $8.2 \times 10^{-2}$ </td><td>0.96</td><td> $6.3 \times 10^{-6}$ </td><td> $2.0 \times 10^{-3}$ </td><td>0.99</td><td> $8.6 \times 10^{-5}$ </td><td> $7.0 \times 10^{-3}$ </td></tr><tr><td>RG-e (unconstr.)</td><td>0.92</td><td> $5.4 \times 10^{-2}$ </td><td> $1.8 \times 10^{-1}$ </td><td>0.92</td><td> $1.5 \times 10^{-2}$ </td><td> $9.8 \times 10^{-2}$ </td><td>0.96</td><td> $6.9 \times 10^{-6}$ </td><td> $2.0 \times 10^{-3}$ </td><td>0.99</td><td> $7.0 \times 10^{-5}$ </td><td> $6.0 \times 10^{-3}$ </td></tr><tr><td>RG-o (unconstr.)</td><td>0.77</td><td> $1.4 \times 10^{-1}$ </td><td> $3.0 \times 10^{-1}$ </td><td>0.75</td><td> $3.8 \times 10^{-2}$ </td><td> $1.6 \times 10^{-1}$ </td><td>0.89</td><td> $8.7 \times 10^{-4}$ </td><td> $2.9 \times 10^{-2}$ </td><td>0.82</td><td> $2.9 \times 10^{-3}$ </td><td> $5.2 \times 10^{-2}$ </td></tr><tr><td>RG-se (unconstr.)</td><td>0.93</td><td> $4.0 \times 10^{-2}$ </td><td> $1.5 \times 10^{-1}$ </td><td>0.95</td><td> $9.9 \times 10^{-3}$ </td><td> $7.8 \times 10^{-2}$ </td><td>0.98</td><td> $2.9 \times 10^{-6}$ </td><td> $1.0 \times 10^{-3}$ </td><td>1.00</td><td> $3.0 \times 10^{-5}$ </td><td> $4.0 \times 10^{-3}$ </td></tr><tr><td>RG-seo (unconstr.)</td><td>0.93</td><td> $4.0 \times 10^{-2}$ </td><td> $1.5 \times 10^{-1}$ </td><td>0.95</td><td> $9.8 \times 10^{-3}$ </td><td> $7.8 \times 10^{-2}$ </td><td>0.97</td><td> $6.8 \times 10^{-6}$ </td><td> $2.0 \times 10^{-3}$ </td><td>0.99</td><td> $6.8 \times 10^{-5}$ </td><td> $7.0 \times 10^{-3}$ </td></tr></table>

![](images/181db632ff007ff624f6bd0ec702eb40f7a3ebf6210c38c54e7d8cdfc84f3dc4.jpg)

<details>
<summary>scatter</summary>

| Category         | Data Value | W Value |
| ---------------- | ---------- | ------- |
| airplane         | 0.071      | 0.071   |
| cup              | 0.138      | 0.138   |
| lamp             | 0.088      | 0.088   |
| vase             | 0.152      | 0.152   |
| chair            | 0.103      | 0.103   |
</details>

Figure 2: ModelNet40: a RG-Wormhole variant in reconstruction experiment.

# 4.2 COMPARISONS OF RG VARIANTS VS. WORMHOLE IN LOW-DATA REGIMES

We compare our RG framework with Wormhole within the same training sizes, matching the preprocessing of (Haviv et al., 2024) across four datasets spanning dimensionality: MNIST pixel point clouds (2D), ShapeNetV2 point clouds (3D), MERFISH Niche Cells (254D), and scRNA-seq (2,500D). We train on $\bar { N } \in \{ 1 0 , 5 \mathrm { { 0 } , 1 0 0 , 2 0 0 } \}$ random pairs and evaluate $R ^ { 2 } / \mathrm { M S E / M A E }$ against exact WD. For fairness, the number of training pairs for Wormhole equals the number used to estimate the linear coefficients for RG variants, i.e., $M _ { 0 } { = } N$ . Full results appear in Figures 6–13 with settings in Appendix B.4; Table 2 summarizes the $M _ { 0 } { = } 1 0 0$ case, and other $M _ { 0 }$ follow the same pattern.

Results. Across all four datasets, RG variants consistently outperform Wormhole at small training sizes. Wormhole is weaker primarily because it is data hungry and its performance improves as we add samples, yet under comparable budgets it still trails our methods. By contrast, RG variants are already accurate with few pairs, with unconstrained variants are slightly stronger, whereas constrained variants converge faster and are preferable at the very smallest sizes. RG-se and RG-seo are the strongest when given sufficient samples though the latter can lag at the tiniest sizes before its weights settle but becomes top-performing quickly and still requires far fewer pairs than Wormhole.

# 4.3 RG-WORMHOLE: ACCELERATING WORMHOLE WITH REGRESSION OF WASSERSTEIN

The previous comparison reveals a clear trade-off. RG framework is lightweight and data-efficient, but it does not produce Euclidean embeddings and therefore cannot support interpolation experiments. Wormhole, in contrast, learns Euclidean embeddings that enable interpolation and reconstruction, but it is computationally heavy because training requires many Wasserstein evaluations (pairwise distances within each mini-batch and reconstruction losses), which slows and raises training cost.

![](images/57f53d2c82a6fd88db8704e7eca7d406c9fdc0323fdf7f7770364d2332ccb59e.jpg)  
Figure 3: ModelNet40: a RG-Wormhole variant in interpolation experiment.

RG-Wormhole. To combine the strengths of both, we introduce RG-Wormhole. We first calibrate a RG surrogate on a small set of exact Wasserstein pairs from the same data domain and freeze its weights. We then keep the Wormhole architecture, optimizer, and schedule unchanged, and simply replace every use of the Wasserstein distance with the calibrated surrogate in both the encoder (pairwise distances in the batch) and the decoder (reconstruction loss). No other component is modified. This substitution makes each training step much faster while preserving the performance.

We run five experiments of both models to empirically show that RG-Wormhole is much faster than Wormhole while keeping similar effectiveness. First, we measure training time by training Wormhole and RG-Wormhole under the same optimizer and schedule, sweeping batch sizes 4–20 and reporting wall-clock time for training-set sizes $N \in \{ 1 0 , 5 0 , 1 0 0 , 2 0 0 \}$ . Second, we assess encoders via R2/MSE/MAE between learned pairwise distances and exact Wasserstein. Third, we evaluate decoders via the Wasserstein loss between each input shape and its reconstruction. Fourth, we examine barycenters by decoding each class’s mean embedding and visualizing results. Finally, we study interpolation by decoding linear paths between two embeddings and visualizing the trajectories. Across all experiments, hyperparameters match Wormhole; the only change in RG-Wormhole is replacing every use of the Wasserstein distance in the encoder and decoder losses with the calibrated unconstrained RG variants. For RG-Wormhole, we estimate the RG coefficient using 10 random training samples $( M _ { 0 } { = } 1 0 )$ before plugging into Wormhole. We provide some results in Figures2–3 though the details of experimental settings and full results can be found in Appendix B.5.

Results. Replacing every Wasserstein call in Wormhole with a calibrated RG variants preserves performance while cutting compute. First, in the training-time comparison (Figure 14 in Appendix B.5), RG-Wormhole is far faster than Wormhole across all batch sizes and training budgets, with a very large gap. As batch size increases, Wormhole’s time grows almost exponentially, while RG-Wormhole rises only slightly, close to linear or even flat. Next, we verify that the trained models have similar quality. For the encoder, Figures 15 and 16 in Appendix B.5 show pairwise distances that align with the ground-truth Wasserstein and embeddings that match Wormhole, with essentially identical R2, MSE, and MAE. For the decoder, Figures 17 and 18 in Appendix B.5 evaluate reconstructions against the original point clouds using the Wasserstein distance, and both RG-Wormhole and Wormhole produce very small and nearly identical distances. Finally we test whether RG-Wormhole preserves the geometry needed for downstream use. The decoded class barycenters from RG-Wormhole are clean and class consistent and they match those from Wormhole, we refer to Figure19 in Appendix B.5. We also interpolate by moving linearly in the embedding space and decoding along the path, and the trajectories from RG-Wormhole are smooth and semantically meaningful with no visible artifacts, we refer to Figure20 in Appendix B.5. Overall RG-Wormhole matches Wormhole while training much faster, which makes it a practical choice when compute is limited.

# 5 CONCLUSIONS

We introduced a regression framework mapping Wasserstein to sliced Wasserstein distances under a meta-distribution of random distribution pairs. Two simple linear models enable lightweight estimation, leading to the RG framework for few-shot Wasserstein approximation. We derived constrained and unconstrained forms and validated them through Mixture of Gaussian simulations, point cloud classification, and metric-space visualizations, where the surrogate closely matched the exact distance. Compared to Wormhole on MNIST, ShapeNetV2, MERFISH, and scRNA-seq, our method achieved better performance in low-data regimes. Replacing Wasserstein calls in Wormhole with our method yielded RG-Wormhole, preserving accuracy while greatly reducing training time.

# REFERENCES

Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. Learning representations and generative models for 3d point clouds. In International conference on machine learning, pp. 40–49. PMLR, 2018.   
David Alvarez-Melis and Nicolo Fusi. Geometric dataset distances via optimal transport. Advances in Neural Information Processing Systems, 33:21428–21439, 2020.   
Clément Bonet, Paul Berg, Nicolas Courty, François Septier, Lucas Drumetz, and Minh-Tan Pham. Spherical sliced-Wasserstein. International Conference on Learning Representations, 2023a.   
Clément Bonet, Laetitia Chapel, Lucas Drumetz, and Nicolas Courty. Hyperbolic sliced-Wasserstein via geodesic and horospherical projections. In Topological, Algebraic and Geometric Learning Workshops 2023, pp. 334–370. PMLR, 2023b.   
Clément Bonet, Benoıt Malézieux, Alain Rakotomamonjy, Lucas Drumetz, Thomas Moreau, Matthieu Kowalski, and Nicolas Courty. Sliced-Wasserstein on symmetric positive definite matrices for m/eeg signals. In International Conference on Machine Learning, pp. 2777–2805. PMLR, 2023c.   
Clément Bonet, Lucas Drumetz, and Nicolas Courty. Sliced-wasserstein distances and flows on cartan-hadamard manifolds. Journal of Machine Learning Research, 26(32):1–76, 2025.   
Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. Sliced and Radon Wasserstein barycenters of measures. Journal of Mathematical Imaging and Vision, 1(51):22–45, 2015.   
George EP Box and George C Tiao. Bayesian inference in statistical analysis. John Wiley & Sons, 2011.   
Charlotte Bunne, Stefan G Stark, Gabriele Gut, Jacobo Sarabia Del Castillo, Mitch Levesque, Kjong-Van Lehmann, Lucas Pelkmans, Andreas Krause, and Gunnar Rätsch. Learning single-cell perturbation responses using neural optimal transport. Nature methods, 20(11):1759–1768, 2023.   
Laetitia Chapel, Romain Tavenard, and Samuel Vaiter. Differentiable generalized sliced Wasserstein plans. arXiv preprint arXiv:2505.22049, 2025.   
Yaqing Chen, Zhenhua Lin, and Hans-Georg Müller. Wasserstein regression. Journal of the American Statistical Association, 118(542):869–882, 2023.   
Nicolas Courty, Rémi Flamary, and Mélanie Ducoffe. Learning Wasserstein embeddings. In International Conference on Learning Representations, 2018.   
Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems, pp. 2292–2300, 2013.   
Ishan Deshpande, Yuan-Ting Hu, Ruoyu Sun, Ayis Pyrros, Nasir Siddiqui, Sanmi Koyejo, Zhizhen Zhao, David Forsyth, and Alexander G Schwing. Max-sliced Wasserstein distance and its use for GANs. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 10648–10656, 2019.   
DC Dowson and BV666017 Landau. The fréchet distance between multivariate normal distributions. Journal of multivariate analysis, 12(3):450–455, 1982.   
Jean Feydy, Benjamin Charlier, François-Xavier Vialard, and Gabriel Peyré. Optimal transport for diffeomorphic registration. In Medical Image Computing and Computer Assisted Intervention-MICCAI 2017: 20th International Conference, Quebec City, QC, Canada, September 11-13, 2017, Proceedings, Part I 20, pp. 291–299. Springer, 2017.   
Aude Genevay, Gabriel Peyré, and Marco Cuturi. Learning generative models with Sinkhorn divergences. In International Conference on Artificial Intelligence and Statistics, pp. 1608–1617. PMLR, 2018.   
Doron Haviv, Russell Zhang Kunes, Thomas Dougherty, Cassandra Burdziak, Tal Nawy, Anna Gilbert, and Dana Pe’er. Wasserstein wormhole: Scalable optimal transport distance with Transformer. In Forty-first International Conference on Machine Learning, 2024.

Soheil Kolouri, Kimia Nadjahi, Umut Simsekli, Roland Badeau, and Gustavo Rohde. Generalized sliced Wasserstein distances. In Advances in Neural Information Processing Systems, pp. 261–272, 2019.   
Soheil Kolouri, Navid Naderializadeh, Gustavo K. Rohde, and Heiko Hoffmann. Wasserstein embedding for graph learning. In International Conference on Learning Representations, 2021.   
Tianyi Lin, Chenyou Fan, Nhat Ho, Marco Cuturi, and Michael Jordan. Projection robust Wasserstein distance and Riemannian optimization. Advances in Neural Information Processing Systems, 33: 9383–9397, 2020.   
Xinran Liu, Rocio Diaz Martin, Yikun Bai, Ashkan Shahbazi, Matthew Thorpe, Akram Aldroubi, and Soheil Kolouri. Expected sliced transport plans. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= P7O1Vt1BdU.   
Guillaume Mahey, Laetitia Chapel, Gilles Gasso, Clément Bonet, and Nicolas Courty. Fast optimal transport through sliced generalized wasserstein geodesics. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 35350–35385, 2023.   
Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.   
Boris Muzellec and Marco Cuturi. Subspace detours: Building transport plans that are optimal on subspace projections. In Advances in Neural Information Processing Systems, pp. 6917–6928, 2019.   
Khai Nguyen. An introduction to sliced optimal transport: Foundations, advances, extensions, and applications. Foundations and Trends® in Computer Graphics and Vision, 17(3-4):171–391, 2025.   
Khai Nguyen and Nhat Ho. Energy-based sliced Wasserstein distance. Advances in Neural Information Processing Systems, 2023.   
Khai Nguyen and Nhat Ho. Sliced Wasserstein estimator with control variates. International Conference on Learning Representations, 2024.   
Khai Nguyen, Nhat Ho, Tung Pham, and Hung Bui. Distributional sliced-Wasserstein and applications to generative modeling. In International Conference on Learning Representations, 2021.   
Khai Nguyen, Nicola Bariletto, and Nhat Ho. Quasi-monte carlo for 3d sliced Wasserstein. In International Conference on Learning Representations, 2024.   
Sloan Nietert, Ziv Goldfeld, Ritwik Sadhu, and Kengo Kato. Statistical, robustness, and computational guarantees for sliced wasserstein distances. Advances in Neural Information Processing Systems, 35:28179–28193, 2022.   
Sitara Persad, Zi-Ning Choo, Christine Dien, Noor Sohail, Ignas Masilionis, Ronan Chaligné, Tal Nawy, Chrysothemis C Brown, Roshan Sharma, Itsik Pe’er, et al. Seacells infers transcriptional and epigenomic cellular states from single-cell genomics data. Nature biotechnology, 41(12): 1746–1757, 2023.   
Gabriel Peyré and Marco Cuturi. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning, 11(5-6):355–607, 2019.   
Gabriel Peyré, Marco Cuturi, et al. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning, 11(5-6):355–607, 2019.   
Michael Quellmalz, Robert Beinert, and Gabriele Steidl. Sliced optimal transport on the sphere. Inverse Problems, 39(10):105005, 2023.   
Julien Rabin, Julie Delon, and Yann Gousseau. Regularization of transportation maps for color and contrast transfer. In 2010 IEEE International Conference on Image Processing, pp. 1933–1936. IEEE, 2010.

Julien Rabin, Gabriel Peyré, Julie Delon, and Marc Bernot. Wasserstein barycenter and its application to texture mixing. In Scale Space and Variational Methods in Computer Vision: Third International Conference, SSVM 2011, Ein-Gedi, Israel, May 29–June 2, 2011, Revised Selected Papers 3, pp. 435–446. Springer, 2012.   
Mark Rowland, Jiri Hron, Yunhao Tang, Krzysztof Choromanski, Tamas Sarlos, and Adrian Weller. Orthogonal estimation of Wasserstein distances. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 186–195. PMLR, 2019.   
Yossi Rubner, Carlo Tomasi, and Leonidas J Guibas. A metric for distributions with applications to image databases. In Sixth international conference on computer vision (IEEE Cat. No. 98CH36271), pp. 59–66. IEEE, 1998.   
Meyer Scetbon, Marco Cuturi, and Gabriel Peyré. Low-rank sinkhorn factorization. In International Conference on Machine Learning, pp. 9344–9354. PMLR, 2021.   
Keanu Sisouk, Julie Delon, and Julien Tierny. A user’s guide to sampling strategies for sliced optimal transport. Transactions on Machine Learning Research, 2025. ISSN 2835-8856.   
Eloi Tanguy. Convergence of sgd for training neural networks with sliced Wasserstein losses. arXiv preprint arXiv:2307.11714, 2023.   
Eloi Tanguy, Laetitia Chapel, and Julie Delon. Sliced optimal transport plans. arXiv preprint arXiv:2508.01243, 2025.   
Huy Tran, Yikun Bai, Abihith Kothapalli, Ashkan Shahbazi, Xinran Liu, Rocio Diaz Martin, and Soheil Kolouri. Stereographic spherical sliced Wasserstein distances. International Conference on Machine Learning, 2024.   
Cédric Villani. Optimal transport: old and new, volume 338. Springer Science & Business Media, 2008.   
Cédric Villani. Optimal transport: old and new, volume 338. Springer, 2009.   
Fang Wu, Nicolas Courty, Shuting Jin, and Stan Z Li. Improving molecular representation learning with metric learning-enhanced optimal transport. Patterns, 4(4), 2023.   
Meng Zhang, Stephen W Eichhorn, Brian Zingg, Zizhen Yao, Kaelan Cotter, Hongkui Zeng, Hongwei Dong, and Xiaowei Zhuang. Spatially resolved cell atlas of the mouse primary motor cortex by merfish. Nature, 598(7879):137–143, 2021.

# Supplement to “Fast Estimation of Wasserstein Distances via Regression on Sliced Wasserstein Distances"

# A DETAILS

# A.1 DETAILS OF REMARK 3

We derive the gradient:

$$
\begin{array}{l} \nabla_ {\omega} \mathbb {E} \left[ \left\| \boldsymbol {\omega} ^ {\top} \pmb {S} _ {p} ^ {(k)} (\mu , \nu) - W _ {p} (\mu , \nu)) \right\| _ {2} ^ {2} \right] \\ = \nabla_ {\boldsymbol {\omega}} \mathbb {E} \left[ (\boldsymbol {\omega} ^ {\top} \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) - W _ {p} (\mu , \nu))) ^ {\top} (\boldsymbol {\omega} ^ {\top} \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) - W _ {p} (\mu , \nu))) \right] \\ = \nabla_ {\boldsymbol {\omega}} \mathbb {E} \left[ \boldsymbol {\omega} ^ {\top} \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) ^ {\top} \boldsymbol {\omega} \right] - 2 \nabla_ {\boldsymbol {\omega}} \mathbb {E} \left[ \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) ^ {\top} \boldsymbol {\omega} W _ {p} (\mu , \nu) \right] (13) \\ = \mathbb {E} \left[ \nabla_ {\boldsymbol {\omega}} \boldsymbol {\omega} ^ {\top} \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) ^ {\top} \boldsymbol {\omega} \right] - 2 \mathbb {E} \left[ \nabla_ {\boldsymbol {\omega}} \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) ^ {\top} \boldsymbol {\omega} W _ {p} (\mu , \nu) \right] (14) \\ = 2 \mathbb {E} \left[ \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) ^ {\top} \right] \boldsymbol {\omega} - 2 \mathbb {E} \left[ \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) W _ {p} (\mu , \nu) \right] (15) \\ \end{array}
$$

Setting the gradient to 0, we obtain

$$
\hat {\boldsymbol {\omega}} _ {L S E} = \mathbb {E} \left[ \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) ^ {\top} \right] ^ {- 1} \mathbb {E} \left[ \boldsymbol {S} _ {p} ^ {(k)} (\mu , \nu) W _ {p} (\mu , \nu) \right], \tag {16}
$$

which completes the proof.

# A.2 DETAILS OF REMARK 4

From the definition, we recall the model:

$$
W _ {p} (\mu , \nu) = \sum_ {k = 1} ^ {K} \omega_ {k} S L _ {p} ^ {(k)} (\mu , \nu) + \sum_ {k = 1} ^ {K} (1 - \omega_ {k}) S U _ {p} ^ {(k)} (\mu , \nu) + \varepsilon . \tag {17}
$$

With $K = 1$ , we rewrite the model as follows:

$$
W _ {p} (\mu , \nu) = \omega S L _ {p} (\mu , \nu) + (1 - \omega) S U _ {p} (\mu , \nu) + \varepsilon , \tag {18}
$$

which is equivalent to

$$
W _ {p} (\mu , \nu) - S U _ {p} (\mu , \nu) = \omega (S L _ {p} (\mu , \nu) - S U _ {p} (\mu , \nu)) + \epsilon . \tag {19}
$$

Since equation 19 is again an unconstrained linear model, we can obtain the least-squares estimate by following Appendix A.1:

$$
\hat {\omega} _ {C L S E} = \frac {\mathbb {E} \left[ (S U _ {p} (\mu , \nu) - S L _ {p} (\mu , \nu)) (S U _ {p} (\mu , \nu) - W _ {p} (\mu , \nu)) \right]}{\mathbb {E} [ (S U _ {p} (\mu , \nu) - S L _ {p} (\mu , \nu) ^ {2} ]}, \tag {20}
$$

which concludes the proof.

# B EXPERIMENTS

# B.1 GAUSSIAN SIMULATION

We study how a lower–upper bound pair approximates the Wasserstein distance as dimension grows. We simulate 3-component Gaussian mixtures for d=1 . . . 100 (10 seeds), with 200 points per component. For each pair we compute the exact Wasserstein and six sliced-based metrics. Focusing on RG-o, RG-s, and $R G \ – e ,$ , we fit a constrained weight $w \in [ 0 , 1 ]$ ] and report the estimated weight wˆ and $R ^ { 2 }$ versus the exact Wasserstein.

Results. We refer to Figure 4 for the result. The fits are strong for all three methods and all dimensions: $R ^ { 2 }$ is always above 0.8 and quickly rises ${ \mathrm { t o } } \approx 0 . 9 – 1 . 0$ . We also see a clear pattern in the weights: as dimension grows, the weight on the lower bound goes down, so the upper-bound metric gets more weight and eventually dominates. In short, high dimensions favor the upper bound, while lower dimensions rely more on the lower bound.

![](images/6c3c64afe0140731ae3e04d138f3fc19c278813ffbe9b521ac4f8c6f465ede0e.jpg)

<details>
<summary>line</summary>

| Dimension | Optimal w* | R²    |
| --------- | ---------- | ----- |
| 0         | 0.8        | 0.0   |
| 20        | 0.5        | 0.6   |
| 40        | 0.35       | 0.75  |
| 60        | 0.25       | 0.85  |
| 80        | 0.2        | 0.9   |
| 100       | 0.15       | 0.95  |
</details>

![](images/760d17a12b1026f447c4e9cc318bfe35c3796a72eec6a24f1a6bfd4f8daae47e.jpg)

<details>
<summary>line</summary>

| Dimension | Optimal w* | R²     |
| --------- | ---------- | ------ |
| 0         | 0.6        | 0.8    |
| 20        | 0.3        | 0.9    |
| 40        | 0.2        | 0.95   |
| 60        | 0.15       | 0.97   |
| 80        | 0.1        | 0.98   |
| 100       | 0.05       | 0.99   |
</details>

![](images/ae512d41ba249825238e2566d6502c3d80c245dce096e42e48bf0e97354108f5.jpg)

<details>
<summary>line</summary>

| Dimension | ebsw  | est   | -R²   |
| --------- | ----- | ----- | ----- |
| 0         | 0.6   | 0.4   | 0.95  |
| 20        | 0.3   | 0.8   | 0.9   |
| 40        | 0.2   | 0.9   | 0.95  |
| 60        | 0.15  | 0.95  | 0.95  |
| 80        | 0.1   | 0.95  | 0.95  |
| 100       | 0.05  | 0.95  | 0.95  |
</details>

Figure 4: Optimal $w ^ { \ast }$ and $R ^ { 2 }$ in each dimension.

Table 3: k-NN accuracy on point-cloud classification on ShapeNetV2 dataset. 

<table><tr><td>Methods</td><td> $R^{2}$ </td><td>k=1</td><td>k=3</td><td>k=5</td><td>k=10</td><td>k=15</td></tr><tr><td>WD</td><td>-</td><td>83.6% ± 0.0%</td><td>83.5% ± 0.0%</td><td>84.2% ± 0.0%</td><td>82.9% ± 0.0%</td><td>79.2% ± 0.0%</td></tr><tr><td>SWD</td><td>-</td><td>72.4% ± 0.0%</td><td>71.4% ± 0.0%</td><td>70.4% ± 0.0%</td><td>69.0% ± 0.0%</td><td>66.7% ± 0.0%</td></tr><tr><td>PWD</td><td>-</td><td>42.6% ± 0.0%</td><td>42.9% ± 0.0%</td><td>40.4% ± 0.0%</td><td>39.3% ± 0.0%</td><td>39.0% ± 0.0%</td></tr><tr><td>EBSW</td><td>-</td><td>72.5% ± 0.0%</td><td>69.2% ± 0.0%</td><td>60.4% ± 0.0%</td><td>67.9% ± 0.0%</td><td>65.3% ± 0.0%</td></tr><tr><td>EST</td><td>-</td><td>39.1% ± 0.0%</td><td>40.4% ± 0.0%</td><td>40.2% ± 0.0%</td><td>38.0% ± 0.0%</td><td>36.5% ± 0.0%</td></tr><tr><td>Max-SW</td><td>-</td><td>60.3% ± 0.0%</td><td>54.6% ± 0.0%</td><td>57.7% ± 0.0%</td><td>57.6% ± 0.0%</td><td>56.8% ± 0.0%</td></tr><tr><td>Min-SWGG</td><td>-</td><td>36.4% ± 0.0%</td><td>37.6% ± 0.0%</td><td>35.0% ± 0.0%</td><td>32.9% ± 0.0%</td><td>30.8% ± 0.0%</td></tr><tr><td>RG-s</td><td>0.868 ± 0.02</td><td>82.1% ± 0.1%</td><td>81.7% ± 0.1%</td><td>80.8% ± 0.1%</td><td>79.4% ± 0.2%</td><td>75.5% ± 0.2%</td></tr><tr><td>RG-e</td><td>0.926 ± 0.04</td><td>82.5% ± 0.1%</td><td>82.2% ± 0.1%</td><td>80.9% ± 0.2%</td><td>79.6% ± 0.3%</td><td>75.7% ± 0.3%</td></tr><tr><td>RG-o</td><td>0.774 ± 0.38</td><td>65.1% ± 0.3%</td><td>67.7% ± 0.3%</td><td>67.6% ± 0.5%</td><td>66.7% ± 0.5%</td><td>66.0% ± 0.5%</td></tr><tr><td>RG-se</td><td>0.935 ± 0.02</td><td>82.5% ± 0.4%</td><td>82.2% ± 0.4%</td><td>82.6% ± 0.5%</td><td>81.9% ± 0.5%</td><td>76.5% ± 0.5%</td></tr><tr><td>RG-seo</td><td>0.937 ± 0.01</td><td>82.8% ± 0.4%</td><td>83.3% ± 0.5%</td><td>83.5% ± 0.7%</td><td>82.3% ± 0.7%</td><td>77.9% ± 0.7%</td></tr></table>

# B.2 POINT CLOUD CLASSIFICATION

Experimental settings. We construct a 10-class subset, centralize, normalize each shape so that all coordinates lie in $[ - 1 , 1 ] ^ { 3 }$ , and uniformly subsample 2,048 points per shape. For each class we select 50 training examples and 100 test examples. We then compute pairwise distance matrices between train and test sets under different metrics, and evaluate classification accuracy using a k-nearest neighbor classifier with $k \in \{ 1 , 3 , 5 , 1 0 , 1 5 \}$ . Besides the six individual sliced-based metrics, we include all RG variants in unconstrained version. We use 10 samples drawn from the training set to estimate the linear coefficient of RG variants.

# B.3 METRIC SPACE VISUALIZATION

Experimental settings. We visualize the geometry each metric induces on ShapeNetV2. From 10 categories, we randomly sample 500 shapes per class, normalize each shape so that all coordinates lie $\mathrm { i n } [ - 1 , 1 ] ^ { 3 }$ , and keep 2,048 points per shape. For every method, we compute the pairwise distance matrix, then feed to UMAP to obtain 2D embeddings. We use 10 samples drawn from the training set to estimate the linear coefficient of RG variants.

Results. The result is visual in Figures 5. Across methods, the true Wasserstein produces wellseparated class clusters with clear margins. The RG variants produce embeddings that are visually very close to the Wasserstein embeddings, preserving both local compactness and the global arrangement of classes. By contrast, single sliced baselines are weaker. SWD and EBSW keep some structure but blur boundaries, while Max-SW and Min-SWGG show more mixing and noise.

![](images/45fef9262a17b11fa8173f0a1390e36a8fd33b4c3edadbdeacc846db7c077858.jpg)  
Figure 5: Embeddings of methods in ShapeNetV2 dataset.

Table 4: Wormhole training hyperparameters. 

<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Batch size</td><td>10</td></tr><tr><td>Optimizer / LR</td><td>Adam,  $lr = 10^{-4}$ </td></tr><tr><td>LR schedule</td><td>ExponentialLR, final factor ≈ 0.1 over all epochs</td></tr><tr><td>Epochs</td><td>2,000 epochs (20,000 steps)</td></tr><tr><td>Transformer depth</td><td>num_layers = 3</td></tr><tr><td>Attention heads</td><td>num_heads = 4</td></tr><tr><td>Embedding dim</td><td>emb_dim = 128</td></tr><tr><td>MLP hidden dim</td><td>mlp_dim = 512</td></tr><tr><td>Attention dropout</td><td>attention_dropout_rate = 0.1</td></tr><tr><td>Decoder coeff.</td><td>coeff_dec = 0.1</td></tr></table>

# B.4 COMPARISON OF RG VARIANTS VS. WORMHOLE IN LOW-DATA REGIMES

Experimental Settings. We compare our proposed RG framework against Wormhole, a state-of-theart Wasserstein approximation method. To ensure fairness, we follow the exact preprocessing protocol of Haviv et al. (2024). We consider four datasets spanning a wide range of dimensionalities: (i) MNIST point clouds, obtained by thresholding 28 × 28 grayscale images and treating the active pixels as 2D point coordinates; (ii) ShapeNetV2 point clouds, where each CAD model is uniformly sampled into 2,048 points in 3D and normalized; (iii) MERFISH Cell Niches, where each cell is represented by the 50 µm neighborhood of its gene-expression profile embedded in a 254-dimensional space; and (iv) scRNA-seq atlas data, where cells are aggregated into MetaCells that form 2,500-dimensional gene-expression point clouds. We vary the number of training pairs $N \in \{ 1 0 , 5 0 , 1 0 0 , 2 0 0 \}$ by drawing pairs uniformly, and evaluate on 10,000 independently sampled test pairs. For each dataset and training size, we report $R ^ { 2 }$ , MSE, and MAE with respect to the exact Wasserstein.

The original Wormhole codebase is built on JAX and TensorFlow, which are not compatible with our environment. Accordingly, we reimplemented Wormhole in PyTorch.

Data Preprocessing. We follow the same preprocessing pipeline as Haviv et al. (2024).

• MNIST Point Clouds. We turn MNIST 28×28 images into 2D point clouds by thresholding pixel values at 0.5 and keeping the coordinates of the active pixels.   
• ShapeNetV2 Point Clouds. We use ShapeNetCore.v2 with 15k points per shape. Each shape is normalized to fit inside a unit cube with coordinates in [−1, 1]3. We then split each shape into 10k training points and 5k test points, and randomly sample 2,048 points from each point cloud.   
• MERFISH Cell Niches. We scale each gene’s expression to [−1, 1] and divide by ${ \sqrt { d } } ,$ where d is the number of genes. For each cell, we use spatial positions to find its 11 nearest neighbors within a 50 µm radius, keeping only cells with enough neighbors with its cell-type label.   
• scRNA-seq. We select 2,500 highly variable genes, normalize counts (library-size √ $1 0 ^ { 4 }$ and log(1+x)), and scale each gene to [−1, 1] divided by $\sqrt { d }$ (d=2500). We then cluster cells with K-means. For each cluster seed, we consider it as a cloud, labeled by the seed’s annotation.

Wormhole training hyperparameters. We follow the Transformer autoencoder setup of Wormhole with the configuration in Table 4.

![](images/68c78fd5a1e6ee28460f2bdc13a7f04554dd582830e4d0bd7395f22c2fa881ba.jpg)  
Figure 6: MNIST Point Cloud: Wormhole and RG variants (constrained/unconstrained) across training set sizes of 10, 50, 100 and 200.

![](images/4e0125a55ca18969e18754c4c1325952fb59bd6ad8783b6c984c42153c8582ee.jpg)  
Figure 7: MNIST Point Cloud: Wormhole and RG variants (constrained/unconstrained) across training set sizes of 10, 50, 100 and 200.

![](images/8da9df692800708b00f451ceb4e6e8edb00c44b514c1428a7846742a195a26b6.jpg)  
Figure 8: ShapeNetV2 Point Cloud: Wormhole and RG variants (constrained/unconstrained) across training set sizes of 10, 50, 100, and 200.

![](images/5f6c8b9b247d36998cfdb708837d0ed93732fab89a000f8bd9b0bfb8ced34138.jpg)  
Figure 9: ShapeNetV2 Point Cloud: Wormhole and RG variants (constrained/unconstrained) across training set sizes of 10, 50, 100, and 200.

![](images/bb06d8c98fedc8e765f2ee48ccca06ff0084fd058661cc24b8ca981a27f03b1e.jpg)  
Figure 10: MERFISH Cell Niches: Wormhole and RG variants (constrained/unconstrained) across training set sizes of 10, 50, 100, and 200.

![](images/3e0ba7b074784801c65f853b6b49642d9b19e4a8737ee52e0906f99ee8ffe1c8.jpg)  
Figure 11: MERFISH Cell Niches: Wormhole and RG variants (constrained/unconstrained) across training set sizes of 10, 50, 100, and 200.

![](images/8d7c142acacf66295a5bae255e44b42cf704fceba769844d6ad6ff8fa7e45d0a.jpg)  
Figure 12: scRNA-seq: Wormhole and RG variants (constrained/unconstrained) across training set sizes of 10, 50, 100, and 200.

![](images/ea8cc44d811f8d157a128b8e0efe6c2181f1e7ef51b29a6a40a519dd305deffc.jpg)  
Figure 13: scRNA-seq: Wormhole and RG variants (constrained/unconstrained) across training set sizes of 10, 50, 100, and 200.

![](images/296435bed8ad8cd9e37b1789092a403c08ac0f954b96d1c75846d765fb3b0de0.jpg)  
Figure 15: ShapeNetV2: RG-Wormhole (constrained model) vs. Wormhole.

![](images/83dab351996d2fec48712be36906fc8464f1600bc4f86c5fdf8dfa3962baa990.jpg)  
Figure 16: ShapeNetV2: RG-Wormhole (unconstrained model) vs. Wormhole

# B.5 RG-WORMHOLE: ACCELERATING WORMHOLE WITH REGRESSION OF WASSERSTEIN

Experimental Settings. We run five experiments to show that RG-Wormhole is much faster than Wormhole with similar effectiveness. First, we measure training time by training both models under the same optimizer and schedule, sweeping batch sizes from 4 to 20 and reporting wall-clock time for training sets of 10, 50, 100, and 200 pairs. Second, we assess encoders by computing R2/MSE/MAE between pairwise distances in the learned embedding space and exact Wasserstein. Third, we evaluate decoders by reporting the Wasserstein loss between each input and its reconstruction. Fourth, we examine barycenters by decoding the mean embedding of each class and visualizing results. Fifth, we study interpolation by decoding linear paths between two embeddings and illustrating trajectories. Across all experiments, hyperparameters match Wormhole; the only change in RG-Wormhole is replacing Wasserstein in encoder and decoder losses with the calibrated unconstrained RG. We use 10 samples from the training set to estimate RG coefficients. Except for embedding experiment which uses ShapeNetV2 dataset, other experiments use ModelNet40 dataset, same as (Haviv et al., 2024).

![](images/5fd0fc9c992ce48adf6ba1c1fdf96941ed377764128dbc8c309fec85b2f82833.jpg)  
Figure 14: Training time comparison of Wormhole and RG-Wormhole methods on point cloud datasets with varying number of training samples.

# B.6 COMPARISON BETWEEN RG AND CLASSICAL APPROXIMATION METHODS.

Experimental Settings. We conducted direct runtime and accuracy comparisons between our methods and two widely used classical OT accelerations, Sinkhorn and Linear OT, under both singlepair computation and large-scale pairwise computations. For Sinkhorn and Linear OT, no training phase is required. For example, in RG-s, we estimate the optimal regression weight using 5 randomly selected samples (M = 10 training pairs). The training cost (2.891 seconds (s)) is not included in the inference-time table below, as we focus on inference time only. For Sinkhorn, we conduct a grid search over the entropic regularization hyperparameter in {0.01, 0.05, 0.1} and the number of

![](images/8b066dd8dc29d7cf23c186633eeb212afa3b36f7e5022ec1f6e3e539d28445ea.jpg)

<details>
<summary>heatmap</summary>

| Wormhole Type | Category   | W Value |
| ------------- | ---------- | ------- |
| Airplane      | data       | -       |
| Airplane      | reconstruction | -     |
| Airplane      | Reconstruction | -     |
| Cup           | data       | -       |
| Cup           | reconstruction | -     |
| Lamp          | data       | -       |
| Lamp          | reconstruction | -     |
| Vase          | data       | -       |
| Vase          | reconstruction | -     |
| Chair         | data       | -       |
| Chair         | reconstruction | -     |
| Airplane      | W = 0.058  | -       |
| Airplane      | cup        | 0.039   |
| Airplane      | lamp        | 0.044   |
| Airplane      | vase        | 0.047   |
| Airplane      | chair       | 0.053   |
| Reconstruction | data       | -       |
| Reconstruction | reconstruction | -     |
| Reconstruction | Reconstruction | -     |
| Airplane      | W = 0.115  | -       |
| Airplane      | data       | -       |
| Airplane      | Reconstruction | -     |
| Cup           | data       | 0.128   |
| Cup           | reconstruction | -     |
| Lamp          | data       | 0.103   |
| Lamp          | Reconstruction | -     |
| Vase          | data       | 0.094   |
| Vase          | reconstruction | -     |
| Chair         | data       | 0.122   |
| Chair         | reconstruction | -     |
| Airplane      | W = 0.061  | -       |
| Airplane      | data       | -       |
| Airplane      | Reconstruction | -     |
| Cup           | data       | 0.121   |
| Cup           | reconstruction | -     |
| Lamp          | data       | 0.084   |
| Lamp          | Reconstruction | -     |
| Vase          | data       | 0.090   |
| Vase          | reconstruction | -     |
| Chair         | data       | 0.098   |
| Chair         | reconstruction | -     |
| Airplane      | W = 0.106  | -       |
| Airplane      | data       | -       |
| Airplane      | Reconstruction | -     |
| cup           | data       | 0.140   |
| cup           | reconstruction | -     |
| Lamp          | data       | 0.084   |
| Lamp          | Reconstruction | -     |
| Vase          | data       | 0.093   |
| Vase          | reconstruction | -     |
| Chair         | data       | 0.104   |
| Chair         | reconstruction | -     |
</details>

Figure 17: ModelNet40: RG-Wormhole vs Wormhole reconstruction experiment.

![](images/27ccb0264a5ee9bcf7f3d86b50159987420e175101ddf0ffd16ea1408896edcb.jpg)

<details>
<summary>heatmap</summary>

| Wormhole Type | Category | W Value |
| --- | --- | --- |
| RG-se Wormhole | airplane | 0.105 |
| RG-se Wormhole | cup | 0.133 |
| RG-se Wormhole | lamp | 0.079 |
| RG-se Wormhole | vase | 0.156 |
| RG-se Wormhole | chair | 0.098 |
| RG-se reconstruction | airplane | 0.071 |
| RG-se reconstruction | cup | 0.138 |
| RG-se reconstruction | lamp | 0.088 |
| RG-se reconstruction | vase | 0.152 |
| RG-se reconstruction | chair | 0.103 |
The image contains multiple 3D heatmaps with no explicit numerical axis labels provided in the image.
</details>

Figure 18: ModelNet40: RG-Wormhole reconstruction experiment.

iterations in {500, 1000}, and report the best-performing configuration for each dataset. We do not exceed 1,000 iterations because the runtime already surpasses that of exact Wasserstein.

For inference runtime comparison, we use point clouds from ShapeNetV2, where each object consists of 2,048 points in 3D, and measure inference time as the number of pairwise computations in seconds increases (see Table 5). These time measurements are relative and may not be precise due to system conditions.

For performance comparison, we evaluate each method’s approximation quality relative to exact Wasserstein using $R ^ { 2 }$ in Table 6. Each value is computed over 5,000 test pairs across four datasets: MNIST point clouds, ShapeNetV2, MERFISH cell niches, and scRNA-seq.

Results. From the results above, several observations are clear. First, Sinkhorn performs the worst among the compared methods. Even with 1,000 iterations, it fails to converge on MNIST point clouds and ShapeNetV2 (negative $R ^ { 2 } )$ , while also being slower than both Linear OT and RG-s. However, this may be due to a suboptimal choice of the entropic regularization parameter. In theory, the time complexity of Sinkhorn is $\mathcal { O } ( ( N - M ) ^ { 2 } n ^ { 2 } / \epsilon )$ for $( N - M ) ^ { 2 }$ pairs. Therefore, Sinkhorn is computationally expensive in this setting. Second, Linear OT provides reasonable accuracy and moderate speed, but its approximation quality is consistently below that of RG-s, and its runtime grows much faster as the number of distributions increases. As discussed, the time complexity of Linear OT is $\mathcal { O } ( ( N - M ) n ^ { 2 } / \epsilon )$ . Third, the RG methods provide more accurate estimates than both Sinkhorn and Linear OT. In addition, RG-s, RG-e, RG-o, and RG-seo are all faster than Linear OT. RG-seo achieves a speed comparable to Linear OT while offering better approximation quality, whereas optimization-based SW variants require substantially more computation time.

![](images/29a466a13e800b80641c02e602cc5490a9fcd38c26a394a62b6d7f6718c01590.jpg)  
Figure 19: ModelNet40: RG-Wormhole barycenter experiment.

Table 5: Comparison in inference runtime among RG and classical approximation methods. 

<table><tr><td>Method</td><td>2 Samples (1 pair)</td><td>10 Samples (45 pairs)</td><td>50 Samples (1,225 pairs)</td><td>100 Samples (4,950 pairs)</td><td>200 Samples (19,900 pairs)</td></tr><tr><td>Wasserstein</td><td>0.604s</td><td>13.351s</td><td>354.823s</td><td>1,427.276s</td><td>5,030.966s</td></tr><tr><td>Sinkhorn (iters=500)</td><td>0.441s</td><td>10.444s</td><td>286.203s</td><td>1,154.826s</td><td>4,628.393s</td></tr><tr><td>Sinkhorn (iters=1,000)</td><td>0.482s</td><td>21.012s</td><td>572.441s</td><td>2,308.735s</td><td>9,336.621s</td></tr><tr><td>Linear OT</td><td>0.634s</td><td>3.141s</td><td>15.473s</td><td>31.121s</td><td>92.134s</td></tr><tr><td>RG-s</td><td>0.002s</td><td>0.090s</td><td>2.043s</td><td>8.105s</td><td>21.408s</td></tr><tr><td>RG-e</td><td>0.0005s</td><td>0.023s</td><td>1.737s</td><td>7.060s</td><td>18.429s</td></tr><tr><td>RG-o</td><td>0.013s</td><td>0.136s</td><td>6.634s</td><td>15.310s</td><td>54.900s</td></tr><tr><td>RG-se</td><td>0.002s</td><td>0.113s</td><td>3.780s</td><td>15.170s</td><td>39.950s</td></tr><tr><td>RG-seo</td><td>0.013s</td><td>0.147s</td><td>7.017s</td><td>28.840s</td><td>94.200s</td></tr></table>

![](images/b6119fd92126df0e7d48f6ecc6c61b738e758be172b3aa0acd832c37ad03d910.jpg)  
Figure 20: ModelNet40: RG-Wormhole barycenter experiment.

Table 6: Comparison in performance among RG and classical approximation methods. 

<table><tr><td>Method</td><td>MNIST Point Cloud (10,000 test pairs)</td><td>ShapeNetV2 (10,000 test pairs)</td><td>MERFISH Cell Niches (10,000 test pairs)</td><td>scRNA-seq (10,000 test pairs)</td></tr><tr><td>Sinkhorn</td><td>-1.970</td><td>-1.549</td><td>0.764</td><td>0.964</td></tr><tr><td>Linear OT</td><td>0.697</td><td>0.918</td><td>0.842</td><td>0.984</td></tr><tr><td>RG-s</td><td>0.925</td><td>0.942</td><td>0.964</td><td>0.989</td></tr><tr><td>RG-e</td><td>0.910</td><td>0.919</td><td>0.961</td><td>0.991</td></tr><tr><td>RG-o</td><td>0.767</td><td>0.753</td><td>0.809</td><td>0.923</td></tr><tr><td>RG-se</td><td>0.933</td><td>0.948</td><td>0.984</td><td>0.996</td></tr><tr><td>RG-seo</td><td>0.934</td><td>0.948</td><td>0.969</td><td>0.992</td></tr></table>

![](images/7bdcd36d1563eb1bb764fb3a4b3a4a89de41c293f12d684517a2a6a84cf20da6.jpg)  
Figure 21: ModelNet40: Intra-class Experiment.

![](images/fa5e66b4c134d28bc8383467bde08b989b8e496727c3d515be4ee98e2ae7bcf1.jpg)  
Figure 22: ModelNet40: Inter-class Experiment.

# B.7 EXPERIMENT OF INTER-CLASS AND INTRA-CLASS FOR RG.

Experimental Settings. We conducted an experiment to test how well RG generalizes across intraclass and inter-class pairs. We trained RG only on intra-class pairs and tested on inter-class pairs, and vice versa, using five scenarios for each setting on ShapeNetV2.

Concretely, for the intra-class → inter-class setting, $( A , A ) \to ( A , B )$ , we ran five experiments. In each experiment, we sampled 10 intra-class training pairs from (A, A), estimated the regression weight, and then constructed a test set by sampling 50 objects from class A and 50 objects from class B. This yields 100 distributions in total and therefore 4,950 inter-class test pairs $( A , B )$ .

For the inter-class → intra-class setting, $( A , B ) \to ( A , A )$ , we again ran five experiments. In each experiment, we sampled five objects from class A and five objects from class B to form 10 inter-class training pairs $( A , B )$ , estimated the regression weight, and then built a test set by sampling 100 objects from class A, which yields 4,950 intra-class test pairs $( A , A )$ .

Results. We present the results in Figures 21–22. Although this is intentionally suboptimal (the regression weight is learned from a restricted subset of pairs), RG still achieves reasonably high $R ^ { 2 }$ (e.g., 0.902, 0.846, 0.932, 0.805), compared to about 0.95 when training on the full pair distribution. This shows that RG generalizes well to out-of-sample data, with performance degradation that is present but moderate.

# B.8 EXPERIMENT OF OPTIMAL WEIGHTS OF RG.

Results. We visualized the estimated coefficients of the RG methods for all applications in Figures 23– 30. The coefficient patterns vary substantially across datasets, indicating that there is no fixed set of coefficients that universally applies to all cases. This further underscores the need for the proposed regression framework, which captures the dataset-specific uncertainty in the association between Wasserstein and sliced Wasserstein distances (i.e., the meta-distribution) on the full pair distribution.

![](images/3c119d88c96fdaa27b3bce3ac6ed5cabadb71a21456c0f45a1402e1e7090f227.jpg)

<details>
<summary>bar</summary>

|        | SW    | PWD   | EBSW  | EST   | MaxSW | MinSWGG |
| ------ | ----- | ----- | ----- | ----- | ----- | ------- |
| RG-e   | 0.000 | 0.000 | 0.56  | 0.43  | 0.000 | 0.000   |
| RG-o   | 0.000 | 0.000 | 0.000 | 0.97  | 0.02  | 0.000   |
| RG-s   | 0.66  | 0.34  | 0.000 | 0.000 | 0.000 | 0.000   |
| RG-se  | 0.72  | 0.28  | 0.55  | 0.44  | 0.000 | 0.000   |
| RG-seo | 0.67  | 0.32  | 0.38  | 0.61  | 0.88  | 0.11    |
</details>

Figure 23: MNIST Point Cloud: Optimal weight of RG variants (constrained) across different training samples.

![](images/2c095bbfa45796771bd68942112c86977935b4c0bea030f47aa000908d8e6bc3.jpg)

<details>
<summary>bar</summary>

|        | SW    | PWD   | EBSW  | EST   | MaxSW | MinSWGG |
| ------ | ----- | ----- | ----- | ----- | ----- | ------- |
| RG-e   | 0.01  | 0.01  | 1.05  | 0.26  | 0.01  | 0.01    |
| RG-o   | 0.01  | 0.01  | 0.01  | 0.01  | 1.10  | 0.01    |
| RG-s   | 1.53  | 0.12  | 0.01  | 0.01  | 0.01  | 0.01    |
| RG-se  | 0.88  | 0.01  | 0.42  | 0.21  | 0.01  | 0.01    |
| RG-seo | 0.87  | 0.01  | 0.35  | 0.20  | 0.08  | 0.01    |
</details>

Figure 24: MNIST Point Cloud: Optimal weight of RG variants (unconstrained) across different training samples.

![](images/b9d8805f679aee361d80fcdfbb3919a8e8d80a574bad3f9b6afce9cb3ec8d882.jpg)

<details>
<summary>bar</summary>

| Model   | SW    | PWD   | EBSW  | EST   | MaxSW | MinSWGG |
|---------|-------|-------|-------|-------|-------|---------|
| RG-e    | 0.000 | 0.000 | 0.610 | 0.390 | 0.000 | 0.000   |
| RG-o    | 0.000 | 0.000 | 0.880 | 0.120 | 0.000 | 0.000   |
| RG-s    | 0.590 | 0.410 | 0.000 | 0.000 | 0.000 | 0.000   |
| RG-se   | 0.440 | 0.550 | 0.810 | 0.190 | 0.000 | 0.000   |
| RG-seo  | 0.360 | 0.640 | 0.890 | 0.110 | 0.820 | 0.180   |
</details>

Figure 25: ShapeNetV2: Optimal weight of RG variants (constrained) across different training samples.

ShapeNetV2: Optimal weights of RG (unconstr.)   
![](images/fce2a0682b3e71fbd2f0a2ce2adfbd38f6f83d21bf1ab7e21194e099c37dc9c9.jpg)

<details>
<summary>bar</summary>

|        | SW    | PWD   | EBSW  | EST   | MaxSW | MinSWGG |
| ------ | ----- | ----- | ----- | ----- | ----- | ------- |
| RG-e   | 0.00  | 0.00  | 1.05  | 0.25  | 0.00  | 0.00    |
| RG-o   | 0.00  | 0.00  | 0.00  | 0.00  | 0.60  | 0.20    |
| RG-s   | 1.45  | 0.20  | 0.00  | 0.00  | 0.00  | 0.00    |
| RG-se  | 1.65  | 0.45  | -0.15 | -0.30 | 0.00  | 0.00    |
| RG-seo | 1.65  | 0.45  | -0.25 | -0.35 | 0.05  | 0.00    |
</details>

Figure 26: ShapeNetV2: Optimal weight of RG variants (unconstrained) across different training samples.

MERFISH Cell Niches: Optimal weights of RG (constr.)   
![](images/8f2994e877581174a8132e8ed1a0e8bbfbad9e14aa0f64a40cc9fbdeb9689175.jpg)

<details>
<summary>bar</summary>

| Component | Weight |
| --------- | ------ |
| SW        | 0.1    |
| PWD       | 0.9    |
| EBSW      | 0.38   |
| EST       | 0.62   |
| MaxSW     | 0.0    |
| MinSWGG   | 1.0    |
</details>

Figure 27: MERFISH Cell Niches: Optimal weight of RG variants (constrained) across different training samples.

MERFISH Cell Niches: Optimal weights of RG (unconstr.)   
![](images/04077a87ec8b8a744b9478a26415c3953fa9ac34c3ba34c46a0f742bbd71ac3f.jpg)

<details>
<summary>bar</summary>

|        | SW   | PWD  | EBSW | EST  | MaxSW | MinSWGG |
| ------ | ---- | ---- | ---- | ---- | ----- | ------- |
| RG-e   | 0    | 0    | 2    | 0    | 0     | 0       |
| RG-o   | 0    | 0    | 0    | 0    | 0     | 8       |
| RG-s   | 6    | 0    | 0    | 0    | 0     | 0       |
| RG-se  | 4    | -95  | 0    | 70   | 0     | 0       |
| RG-seo | 4    | -95  | 0    | 70   | 0     | 0       |
</details>

Figure 28: MERFISH Cell Niches: Optimal weight of RG variants (unconstrained) across different training samples.

![](images/c00f4f9ca82fcd8aa733c8b3dfc468d8a77273f66de41f3bfe1152b013533209.jpg)

<details>
<summary>bar</summary>

|        | SW    | PWD   | EBSW  | EST   | MaxSW | MinSWGG |
| ------ | ----- | ----- | ----- | ----- | ----- | ------- |
| RG-e   | 0.0   | 0.0   | 0.01  | 1.0   | 0.0   | 1.0     |
| RG-o   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 1.0     |
| RG-s   | 0.01  | 1.0   | 0.0   | 0.0   | 0.0   | 0.0     |
| RG-se  | 0.0   | 1.0   | 0.7   | 0.3   | 0.0   | 0.0     |
| RG-seo | 1.0   | 0.0   | 0.12  | 0.88  | 1.0   | 0.0     |
</details>

Figure 29: scRNA-seq Atlas: Optimal weight of RG variants (constrained) across different training samples.

![](images/87403ececa109b38116bd7bd75548d0165bb07e462d09ead8b80a723733794e3.jpg)

<details>
<summary>bar</summary>

|        | SW   | PWD  | EBSW | EST  | MaxSW | MinSWGG |
| ------ | ---- | ---- | ---- | ---- | ----- | ------- |
| RG-e   | 0    | 0    | 6    | 1    | 0     | 0       |
| RG-o   | 0    | 0    | 0    | 0    | -1    | 2       |
| RG-s   | 12   | 1    | 0    | 0    | 0     | 0       |
| RG-se  | 11   | -12  | -1   | 12   | 0     | 0       |
| RG-seo | 15   | -11  | -2   | 12   | 0     | 0       |
</details>

Figure 30: scRNA-seq Atlas: Optimal weight of RG variants (unconstrained) across different training samples.