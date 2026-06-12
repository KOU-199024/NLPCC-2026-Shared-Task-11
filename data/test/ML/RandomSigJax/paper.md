# RANDOM CONTROLLED DIFFERENTIAL EQUATIONS

Francesco Piatti, Thomas Cass & William F. Turner

Department of Mathematics

Imperial College London

# ABSTRACT

We introduce a training-efficient framework for time-series learning in which large randomly parameterized controlled and rough differential equations act as continuous-time reservoirs. These random dynamical systems map input paths to rich path-dependent representations, while only a linear readout layer is trained, yielding fast, scalable models with strong inductive bias. Building on this foundation, we propose two variants: (i) Random Fourier CDEs (RF-CDEs): these lift the input signal using random Fourier features prior to the dynamics, providing a kernel-free approximation of RBF-enhanced sequence models; (ii) Random Rough DEs (R-RDEs): these operate directly on rough-path inputs via a log-ODE discretisation, using log-signatures to capture higher-order temporal interactions while remaining stable and efficient. We prove that in the infinite-width limit, these models induce the RBF-lifted signature kernel and the rough signature kernel, respectively, offering a unified perspective on random-feature reservoirs, continuous-time deep architectures, and path-signature theory.

We evaluate both models across a range of time-series benchmarks, demonstrating competitive or superior performance. These methods provide a practical alternative to explicit signature computations, retaining their inductive bias while benefiting from the efficiency of random features. Code is publicly available at: https://github.com/FrancescoPiatti/RandomSigJax

# 1 INTRODUCTION

Controlled differential equations (CDEs) generalize ordinary differential equations by allowing dynamics to be driven by an exogenous path $x : [ 0 , T ] \to \mathbb { R } ^ { \bar { d } }$ rather than by time alone. This viewpoint has become central to modern time-series learning: it underlies the continuous-depth limit of residual networks (Cirone et al., 2023), connects naturally to deep state-space models and sequence models with long context (Rangapuram et al., 2018; Gu et al., 2021; Gu & Dao, 2023), and yields the neural CDE paradigm in which the vector field is represented by a neural network and learned from data (Kidger et al., 2020; Jhin et al., 2024). Beyond modelling, CDEs provide a clean analytical lens for studying expressivity and invariances of sequence models.

A complementary analytic tool for path-valued data is the signature of a path, i.e., the sequence of iterated integrals that linearizes CDE solution maps in the tensor algebra (Lyons, 1998). Signatures also induce powerful kernels on path space with universality and stability guarantees. In practice, however, signature features and signature kernels can be computationally demanding at high truncation levels, motivating approximations and random features.

Reservoir computing offers an appealing alternative: use a large, randomly initialized dynamical system to produce rich features of the input, and train only a linear readout (Lukosevi ˇ cius & Jaeger, ˇ 2009). This training-light approach scales well and often works competitively. Yet, principled reservoirs for path data that come with nontrivial statistical or kernel limits – and that retain the continuous-time structure of CDEs – are less developed.

# 1.1 RELATED LITERATURE

A growing body of research has drawn connections between random neural networks and kernel methods. Early works demonstrated that infinitely wide neural networks converge to Gaussian processes (Neal, 1996; Williams, 1996). This idea was later extended to deep networks and their training dynamics: in the infinite-width limit, gradient descent on a network is equivalent to kernel regression with the so-called Neural Tangent Kernel (Jacot et al., 2018). In parallel, random feature models were introduced to approximate kernel maps with random projections (Rahimi & Recht, 2007; 2008). For example, random Fourier features approximate shift-invariant kernels with a linear readout, while Extreme Learning Machines fix hidden weights and train only the output layer, achieving universal approximation under standard conditions on the activation and sufficient width (Huang, 2014). These insights inspire reservoir computing, which leverages large random dynamical systems as feature extractors (Maass et al., 2002; Jaeger, 2007). This approach scales well and has demonstrated strong performance in time-series forecasting and classification tasks.

For sequence data, the path signature provides an expressive feature map with strong guarantees (Lyons, 1998), spawning a body of work on signature features and signature kernels in learning and statistics (Chevyrev & Kormilitzin, 2016; Cochrane et al., 2021; Salvi et al., 2021b; Toth & Oberhauser, 2020). A key advance is the PDE/Volterra characterisation and scalable computation of the untruncated signature kernel (Salvi et al., 2021a), with recent numerical refinements (Cass et al., 2025) and applications across regression, classification, and Bayesian inference (Lemercier et al., 2021a;b).

Randomized signatures compress path information by sampling random linear functionals of the (log-)signature coordinates. Discrete-time constructions with approximation and concentration guarantees were developed by Cuchiero et al. (2021), their practical efficacy as reservoirs for learning rough dynamics has been demonstrated in Compagnoni et al. (2023), while universal approximation on path space via finite mixtures of randomized signature features was established by Biagini et al. (2024).

Of particular relevance, Toth et al. (2025) propose Random Fourier Signature Features (RFSF): the input path is first lifted pointwise into a RBF reproducing kernel Hilbert space via random Fourier features and the signature transform is then approximated in that feature space. We adopt RFSF as a strong baseline and point of comparison for our random differential equation-based reservoirs.

# 1.2 CONTRIBUTIONS

In this work, we bridge these viewpoints by developing random feature models for path-valued data that leverage the continuous-time dynamics of CDEs. Our contributions can be summarized as follows:

• Models. We propose two architectures: (i) RF-CDE, which lifts inputs with random Fourier features and then evolves them through a random CDE; and (ii) R-RDE, which operates directly on geometric rough paths via a log-ODE discretization that uses log-signatures to capture higher-order temporal interactions.   
• Theory. We prove infinite-width limits: RF-CDE converges to the RBF-lifted signature kernel, and R-RDE converges to the rough signature kernel; in both cases, the limiting feature maps induce Gaussian–process priors over path-functionals via the standard kernel–GP correspondence.   
• Efficiency. At finite width, both models require training only a linear readout, yielding fast, scalable pipelines in the spirit of reservoir computing. We provide a user-friendly, optimized JAX implementation, RandomSigJax.   
• Experiments. Across time-series benchmarks, our models are competitive with – or surpass – baselines, including Random Fourier Signature Features.

# 2 MATHEMATICAL BACKGROUND

We use the following notation throughout:

• V and W denote Banach spaces; $M _ { N } ( \mathbb { R } )$ is the set of $N \times N$ real matrices; $C ^ { 1 } ( J ; V )$ is the space of continuously differentiable paths from $J \subset \mathbb { R } ^ { + }$ into V .   
• $\Delta _ { T } : = \{ ( s , t ) \in [ 0 , T ] ^ { 2 } : 0 \leq s \leq t \leq T \}$ is the (time) two–simplex.   
• $\xi _ { N }$ is the Gaussian measure of matrices in $M _ { N } ( \mathbb { R } )$ : if the random matrix $A \sim \xi _ { N }$ then its entries $A _ { i j }$ are i.i.d. according to a standard normal distribution.

# 2.1 CONTROLLED DIFFERENTIAL EQUATIONS

Controlled differential equations (CDEs) describe dynamics driven by a path $x : [ 0 , T ] \to \mathbb { R } ^ { d }$ rather than by time alone:

$$
Z _ {t} = z _ {0} + \sum_ {i = 1} ^ {d} \int_ {0} ^ {t} f _ {i} (Z _ {s}) d x _ {s} ^ {i}, \quad f _ {i}: W \rightarrow W. \tag {1}
$$

They capture how a system $Z _ { t } \in W$ reacts to an external control x, and sit at the heart of continuoustime sequence models.

The main subtlety is how to define the integrals in Eq. 1. If x has bounded variation – i.e. finite total variation, equivalently finite 1-variation (Definition A.1 in Appendix $\mathbf { A } )$ – the integrals are classical Riemann-Stieltjes integrals and the CDE is well-posed under standard Lipschitz conditions on $f _ { i }$ .

If x is rougher but has finite p-variation with $p < 2$ , Young integration applies and Eq. 1 still makes sense provided the vector fields are sufficiently regular. At the stochastic $p \ : = \ : 2$ boundary, the integrals may instead be understood in the Ito or Stratonovich sense. Beyond this thresholdˆ $\left( p > 2 \right)$ , pathwise Riemann–Stieltjes/Young/stochastic integrals break down; we handle this regime by lifting x to a (geometric) rough path carrying iterated integrals and interpret Eq. 1 via rough integration – see the rough-path background in Section 2.4.

# 2.2 SIGNATURE AND SIGNATURE KERNELS

A central tool in the analysis of controlled systems is the path signature, which encodes the essential information of a path through its iterated integrals. Let ${ \dot { x } } : [ 0 , T ] \to V$ be a continuous path of finite p-variation with $p < 2$ . Then, for any $t \in [ 0 , \bar { T } ]$ , the signature $\mathrm { S i g } ( x ) _ { 0 , t }$ is the unique solution to the signature CDE:

$$
d \operatorname{Sig} (x) _ {0, t} = \operatorname{Sig} (x) _ {0, t} \otimes d x _ {t}, \quad \operatorname{Sig} (x) _ {0} = \mathbb {1} := (1, 0, 0, \dots),
$$

taking values in the tensor algebra

$$
T ((V)) := \left\{\mathbb {A} = (a ^ {0}, a ^ {1}, \dots) \mid a ^ {0} \in \mathbb {R},   a ^ {k} \in V ^ {\otimes k} \text {   for   } k \geq 1 \right\}.
$$

equipped with componentwise addition and tensor multiplication ⊗. Explicitly,

$$
\operatorname{Sig} (x) _ {0, T} = \bigl (1, S ^ {1} (x), S ^ {2} (x), \dots \bigr), \quad S ^ {k} (x) = \int_ {0 <   t _ {1} <   \dots <   t _ {k} <   T} d x _ {t _ {1}} \otimes \dots \otimes d x _ {t _ {k}}.
$$

The signature has several important properties, including: (i) it uniquely determines a path up to tree-like equivalence (Hambly & Lyons, 2010); (ii) robustness to missing or irregular samples; and (iii) universality, i.e. any continuous functional of a path can be approximated arbitrarily well by linear functionals of its signature.

As the signature is infinite dimensional, for practical use it is truncated. We define the truncated tensor algebra over $V$ of order $N \in \mathbb N$ as the quotient $T ^ { N } ( V ) : = T ( ( V ) ) / T ^ { > N }$ , by the ideal

$$
T ^ {> N} = \{\mathbb {A} = (a ^ {0}, a ^ {1}, \dots) \in T ((V)): a ^ {0} = \dots = a ^ {N} = 0 \},
$$

and the truncated signature at level N is $\mathrm { S i g } ^ { N } : = \pi _ { \le N } ( \mathrm { S i g } ( \cdot ) )$ , with $\pi _ { \leq N }$ the canonical projection.

Further details on the tensor algebra are given in Appendix A.1, and Appendix A.2 reviews the main properties of the signature.

Signature kernels. Endowing $T ( ( V ) )$ with a suitable inner product yields the signature kernel

$$
K _ {\text { Sig }} ^ {x, y} (s, t) = \langle \text { Sig } (x) _ {0, s}, \text { Sig } (y) _ {0, t} \rangle_ {T ((V))}. \tag {2}
$$

When the inner product on each tensor power $V ^ { \otimes k }$ is chosen to be the Hilbert-Schmidt inner product inas dif $\langle \cdot , \cdot \rangle _ { V } ,$ r product on (a subsect of) . The signature kernel isharacterization as the sol $T ( ( V ) )$ ) can be defined by linearityal and, when the paths are the linear hyperbolic PDE $\begin{array} { r } { \langle v , w \rangle _ { T ( ( V ) ) } = \sum _ { k = 0 } ^ { \infty } \langle v _ { k } , w _ { k } \rangle _ { V ^ { \otimes k } } } \end{array}$ (Salvi et al., 2021a)

$$
\partial_ {s} \partial_ {t} K _ {\mathrm{Sig}} ^ {x, y} (s, t) = \left\langle \dot {x} _ {s}, \dot {y} _ {t} \right\rangle_ {V} K _ {\mathrm{Sig}} ^ {x, y} (s, t), \quad K _ {\mathrm{Sig}} ^ {x, y} (0, t) = K _ {\mathrm{Sig}} ^ {x, y} (s, 0) = 1. \tag {3}
$$

Further details on the signature kernels are provided in Appendix A.3.

# 2.3 RANDOM FOURIER SIGNATURE FEATURES

The RBF kernel admits an explicit feature representation: there is a map $\phi : \mathbb { R } ^ { d }  \mathcal { H }$ into its reproducing kernel Hilbert space (RKHS) – Definition A.2 – such that $\kappa _ { \mathrm { R B F } } ( x , y ) = \langle \phi ( x ) , \phi ( y ) \rangle _ { \mathcal { H } }$ . Equivalently, by Bochner’s theorem, any shift-invariant positive-definite kernel $k ( x , y ) = f ( x - y )$ can be written as

$$
k (x, y) = \int_ {\mathbb {R} ^ {d}} e ^ {i \omega^ {\top} (x - y)} d \mu (\omega),
$$

for a probability measure µ (Gaussian for RBF). This yields Random Fourier Features $( R F F ) { \mathrm { : } }$ sample frequencies ω1, . . . , ωF ∼ µ and define the real map ϕFµ : Rd → R2F $\omega _ { 1 } , \ldots , \omega _ { F } \sim \mu$ $\phi _ { \mu } ^ { F } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { 2 F }$

$$
\phi_ {\mu} ^ {F} (x) = \frac {1}{\sqrt {F}} \bigl (e ^ {i \langle \omega_ {1}, x \rangle}, \dots , e ^ {i \langle \omega_ {F}, x \rangle} \bigr), \qquad \langle \phi_ {\mu} ^ {F} (x), \phi_ {\mu} ^ {F} (y) \rangle \approx \kappa_ {\mathrm{RBF}} (x, y). \tag {4}
$$

where we concatenate the real and imaginary part. When the random Fourier feature map is applied along the path and then the signature is taken we are given the na¨ıve Random Fourier Signature Features (RFSF), first proposed in Toth et al. (2025):

$$
\mathrm{RFSF} _ {\mathrm{N}, \mathrm{F}} (x) := \operatorname{Sig} ^ {N} (\phi_ {\mu} ^ {F} (x)) \in T ^ {N} (\mathbb {R} ^ {2 F}). \tag {5}
$$

As F grows, $\phi _ { \mu } ^ { F }$ approximates the RKHS feature map of RBF; as N grows, $\mathrm { S i g } ^ { N }$ approaches the full signature. Hence taking the inner product $\langle \mathrm { R F S F } _ { \mathrm { N , F } } ( x ) , \mathrm { R F S F } _ { \mathrm { N , F } } ( y ) \rangle _ { T ^ { N } ( \mathbb { R } ^ { 2 F } ) }$ provides a practical approximation to the RBF–lifted signature kernel, which can be defined in the limit as

$$
K _ {\mathrm{Sig-RBF}} ^ {x, y} (s, t) = \langle \operatorname{Sig} (\phi \circ x) _ {s}, \operatorname{Sig} (\phi \circ y) _ {t} \rangle_ {T ((\mathcal {H}))}. \tag {6}
$$

While computing the full signature suffers from the curse of dimensionality, Toth et al. (2025) develops projection schemes that render RFSFs computationally tractable. Recall that, from a computational standpoint, using explicit features rather than kernels often avoids constructing and inverting large Gram matrices, yielding far better scalability.

# 2.4 ROUGH PATHS

Rough path theory generalizes controlled differential equations to paths of limited regularity, including those with finite p-variation for $p > 2$ . In contrast to classical paths, rough paths carry additional algebraic structure encoding iterated integrals, which enables a well-posed theory of integration and differential equations driven by such paths.

Definition 2.1 (Rough Path). Let $p \geq 1$ and let ω be a control (i.e. $\omega : \Delta _ { T }  [ 0 , + \infty )$ is continuous, super-additive, and vanishes on the diagonal). A p-rough path over V controlled by ω is a continuous map $\mathbb { X } : \Delta _ { T }  T ^ { \lfloor p \rfloor } ( V )$ such that:

1. $\mathbb { X } _ { s , t } ^ { 0 } = 1$ for all $s , t \in \Delta _ { T }$   
2. Chen’s identity holds: ${ \mathbb X } _ { s , u } \otimes { \mathbb X } _ { u , t } = { \mathbb X } _ { s , t }$   
3. it has finite p-variation on $\Delta _ { T }$ controlled by ω, in the sense

$$
\| \pi_ {k} (\mathbb {X}) \| _ {V ^ {\otimes k}} \leq \frac {\omega (s , t) ^ {i / p}}{\beta_ {p} \Gamma (i / p)}, \quad \forall s, t \in \Delta_ {T}, \quad \forall k = 1, \ldots , \lfloor p \rfloor ,
$$

where $\beta _ { p } \in \mathbb { R }$ is a constant that depends only on $p ,$ and the norm $\| \cdot \| _ { V ^ { \otimes k } }$ is defined in Eq. 17 in Appendix A.

Definition 2.2 (Geometric rough path). A p-rough path X is called geometric if there exists a sequence of bounded variation paths $( x ^ { ( n ) } ) _ { n \geq 1 }$ such that $\pi _ { \lfloor p \rfloor } \big ( \mathrm { S i g } ( x ^ { ( n ) } ) \big ) \longrightarrow \mathbb { X }$ in the p-variation metric (see Definition A.3 in Appendix A).

We denote by $\Omega _ { p } ( V )$ the space of p-rough paths and by $G \Omega _ { p } ( V ) \subset \Omega _ { p } ( V )$ the geometric ones. Given our definitions $\Omega _ { 1 } ( \mathbb { R } ^ { d } )$ is the space of d-dimensional continuous paths of bounded variation.

The tensor algebra $T ( ( V ) )$ carries the Lie bracket $\left[ \mathbb { A } , \mathbb { B } \right] : = \mathbb { A } \otimes \mathbb { B } - \mathbb { B } \otimes \mathbb { A }$ . The free Lie algebra on V , denoted $\mathcal { L } ( ( V ) )$ , is the smallest Lie subalgebra of $T ( ( V ) )$ containing V ; elements are finite linear combinations of iterated brackets $[ e _ { i _ { 1 } } , [ e _ { i _ { 2 } } , [ \ldots , e _ { i _ { k } } ] \cdot \cdot \cdot ]$ , where $\{ e _ { 1 } , \ldots , e _ { d } \}$ is a basis of V . Its degree-m truncation is ${ \mathcal { L } } ^ { m } ( V ) : = \pi { \dot { \operatorname { \lrcorner } } } m { \bar { ( } } { \bar { \mathcal { L } } } ( { \bar { V } } ) { \bar { ) } }$ .

This allows us to define the log-signature as $\log ( \operatorname { S i g } ( x ) ) \in { \mathcal { L } } ( ( V ) )$ (or $\log _ { n } ( \operatorname { S i g } ( x ) ) \in { \mathcal { L } } ^ { \leq n } ( V )$ at finite level), which collects only Lie monomials and thereby removes algebraic redundancies, yielding a more compact coordinate system than the raw signature. The logarithmic map acting on $\mathrm { \dot { T } ( ( V ) ) }$ is described by Eq. 18 in Appendix A.

For $p \geq 1$ and $q \geq 1$ real numbers, the (rough) signature kernel is the map defined for any two geometric p- and q-rough paths X, Y, by

$$
K _ {\text { Sig }} ^ {\mathbb {X}, \mathbb {Y}} (s, t) = \langle \text { Sig } (\mathbb {X}) _ {0, s}, \text { Sig } (\mathbb {Y}) _ {0, t} \rangle_ {T ((V))}. \tag {7}
$$

Finally, rough paths can drive differential equations, which we use in this paper. Appendix A.5 provides a brief review, including the Universal Limit Theorem (Lyons, 1998), guaranteeing existence and uniqueness of RDEs driven by geometric rough paths under $\operatorname { L i p } ( \gamma )$ vector fields (Definition A.5).

# 3 RANDOM CONTROLLED DIFFERENTIAL EQUATIONS

In this section, we review the random controlled differential equation model of Cirone et al. (2023), then introduce our variants and state the corresponding limit theorems.

# 3.1 R-CDE: RANDOM CONTROLLED DIFFERENTIAL EQUATIONS

Let $x \in C ^ { 1 } ( [ 0 , T ] ; \mathbb { R } ^ { d } )$ and let $\mathcal { D } _ { M } = \{ 0 = t _ { 0 } < \cdots < t _ { M } = T \}$ be any partition of [0, T ]. Consider a 1-layer, randomly initialized, homogeneous ResNet driven by x, with random readout $w \sim \xi _ { N }$ (independent of the dynamics). Its output is

$$
\Psi_ {\varphi} ^ {N} (x) := \frac {1}{\sqrt {N}} \bigl \langle w, Z _ {t _ {M}} ^ {N} (x) \bigr \rangle_ {\mathbb {R} ^ {N}},
$$

where the hidden state evolves by the Euler-type recursion on $\mathcal { D } _ { M }$

$$
Z _ {t _ {i + 1}} ^ {N} (x) = Z _ {t _ {i}} ^ {N} (x) + \frac {1}{\sqrt {N}} \sum_ {k = 1} ^ {d} A _ {k} \varphi \bigl (Z _ {t _ {i}} ^ {N} (x) \bigr) \Delta x _ {t _ {i}} ^ {k}, \qquad Z _ {t _ {0}} ^ {N} (x) = z _ {0} \in \mathbb {R} ^ {N},
$$

$\Delta x _ { t _ { i } } ^ { k } : = x _ { t _ { i + 1 } } ^ { k } - x _ { t _ { i } } ^ { k }$ k $\varphi ,$ $A _ { k } \sim \xi _ { N }$ $M _ { N } ( \mathbb { R } )$

Intuitively, as depth $M  \infty$ (with mesh size $\vert \Delta _ { M } \vert : = \mathrm { m a x } _ { i } ( t _ { i + 1 } - t _ { i } )  0 )$ , this recursion converges to a continuous-time controlled system. We take this limit as the definition of the Random Controlled Differential Equation (R-CDE):

$$
d Z _ {t} ^ {N} (x) = \frac {1}{\sqrt {N}} \sum_ {i = 1} ^ {m} A _ {i} \varphi \left(Z _ {t} ^ {N} (x)\right) d x _ {t} ^ {i}, \quad Z _ {0} ^ {N} (x) = z _ {0} \in \mathbb {R} ^ {N}. \tag {8}
$$

The expected inner product of these features converges to the signature kernel, and the readout converges to a Gaussian process with this covariance – thereby characterizing the joint infinitewidth/continuous-depth limit of controlled ResNets.

Theorem 3.1 (Cirone et al. (2023)). Let $x , y \in C ^ { 1 } ( [ 0 , T ] ; \mathbb { R } ^ { d } )$ and let $Z _ { s } ^ { N } ( x ) , Z _ { t } ^ { N } ( y )$ solve Eq. 8 with the same $( A _ { i } ) _ { i = 1 } ^ { d }$ and $\varphi = \mathrm { i d }$ . Then for all $s , t \in [ 0 , T ]$ ,

$$
\lim _ {N \to \infty} \frac {1}{N} \mathbb {E} _ {\xi_ {N}} \Big [ \big \langle Z _ {s} ^ {N} (x), Z _ {t} ^ {N} (y) \big \rangle_ {\mathbb {R} ^ {N}} \Big ] = K _ {\mathrm{sig}} ^ {x, y} (s, t),
$$

the (Hilbert–Schmidt) signature kernel of $( x , y )$ , defined in Eq. 2. Moreover, with $w \sim \xi _ { N }$ independent $o f \left( A _ { i } \right)$ and $Z ^ { N } ( x )$ ,

$$
\lim _ {N \to \infty} \Psi_ {\varphi} ^ {N} (x) = \mathcal {G P} \big (0, K _ {\mathrm{Sig}} ^ {x, x} \big),
$$

in the sense of finite-dimensional distributions.

Remark 3.1. Our theoretical analysis assumes Gaussian matrices for simplicity; however, the conclusions hold for any ensemble $\xi _ { N }$ with the standard moment/tail conditions (centered, unit variance, sub-Gaussian operator–norm tails), as shown by Cass & Turner (2024).

# 3.2 RF-CDE: RANDOM FOURIER CONTROLLED DIFFERENTIAL EQUATIONS

Motivated by the empirical success of Random Fourier Signature Features (Section 2.3) and by the RBF-lifted signature kernel (Eq. 6), we extend the R-CDE framework by incorporating a random Fourier lift of the driving signal.

Let $\phi _ { \mu } ^ { F } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { 2 F }$ be the RFF map in Eq. 4 and, for $x _ { t } \in C ^ { 1 } ( [ 0 , T ] , \mathbb { R } ^ { d } )$ , we denote the lifted path by

$$
X _ {t} ^ {F} := \phi_ {\mu} ^ {F} (x _ {t}) \in \mathbb {R} ^ {2 F},
$$

where $\mu$ is the Gaussian measure. Notice that $X _ { t } ^ { F }$ is also differentiable as it is a composition of differentiable functions. Then we define the Random Fourier CDE (RF-CDE) as

$$
d Z _ {t} ^ {N, F} (x) = \frac {1}{\sqrt {N}} \sum_ {i = 1} ^ {2 F} A _ {i} \varphi \left(Z _ {t} ^ {N, F} (x)\right) d X _ {t} ^ {F, i}, \quad Z _ {0} ^ {N, F} (x) = z _ {0} \in \mathbb {R} ^ {N}, \tag {9}
$$

where $z _ { 0 } , ( A _ { i } ) \sim \xi _ { N }$ independent across i and from the RFF randomness, and $X _ { t } ^ { F , i }$ denotes the i-th component of the lifted path.

Theorem 3.2. Let $x _ { t } , y _ { t }$ be differentiable paths on [0, T ] and $Z _ { s } ^ { N , F } ( x ) , Z _ { t } ^ { N , F } ( y )$ solve Eq. 9 with $\varphi = \mathrm { i d }$ and the same $A _ { i } \stackrel { i . i . d . } { \sim } \xi _ { N }$ (independent of the RFF draw). Then, for every $s , t \in [ 0 , T ]$

$$
\lim _ {F \to \infty} \lim _ {N \to \infty} \frac {1}{N} \mathbb {E} _ {\xi_ {N}} \left[ \left\langle Z _ {s} ^ {N, F} (x), Z _ {t} ^ {N, F} (y) \right\rangle_ {\mathbb {R} ^ {N}} \right] = K _ {S i g - R B F} ^ {x, y} (s, t),
$$

where $K _ { S i g - R B F } ^ { x , y } ( s , t )$ denotes the RBF–lifted signature kernel $( E q . \ 6 )$

We refer the reader to Appendix B.1 for the proof of this theorem.

Gaussian–Process Interpretation. With a fixed random reservoir and a trained linear readout, RF-CDE implements kernel ridge regression with kernel $N ^ { - 1 } \langle Z _ { s } ^ { N , F } ( x ) , Z _ { t } ^ { N , F } ( y ) \rangle$ ⟩, which converges to the RBF–lifted signature kernel as $N , F \to \infty$ . In this limit, the RF-CDE reservoir defines a Gaussian–process prior over path-functionals with that signature-based covariance – mirroring the GP limit established for R-CDE in Section 3.1. This provides a clear interpretation of the model’s inductive bias: RF-CDE inherits the expressive structure of signature kernels while retaining the scalability of random-feature reservoirs.

Discretization. In practice, we discretize Eq. 9, thereby extending its applicability beyond smooth drivers to piecewise-linear paths. We also include a bias vector $b _ { i } \ \sim \ \xi _ { N }$ together with scaling parameters $\sigma _ { A } , \sigma _ { b } ,$ , and $\sigma _ { 0 }$ (tuned via grid search). Applying an Euler scheme yields

$$
\Delta Z _ {t} ^ {N, F} (x) = \frac {1}{\sqrt {N}} \sum_ {i = 1} ^ {F} \left(\sigma_ {A} A _ {i} \varphi \big (Z _ {t} ^ {N, F} (x) \big) + \sigma_ {b} b _ {i}\right) \Delta X _ {t} ^ {F, i}, \qquad Z _ {0} ^ {N, F} (x) = \sigma_ {0} z _ {0} \in \mathbb {R} ^ {N}, \tag {10}
$$

$\Delta X _ { t } ^ { F , i } = X _ { t _ { i + 1 } } ^ { F , i } - X _ { t _ { i } } ^ { F , i }$ and $X _ { t } ^ { F , i }$ is the i-th coordinate of the lifted path $X _ { t } ^ { F }$

# 3.3 R-RDE: RANDOM ROUGH DIFFERENTIAL EQUATIONS

We now extend the model to non-smooth drivers by working directly with geometric p-rough paths. This serves two purposes: (i) noisy time series often benefit from higher–order information, which signatures/log-signatures provide as stable features; (ii) in many applications the (log-)signature is already available or estimable, so operating in rough-path space avoids information loss.

Let $f \in { \mathrm { H o m } } ( V , W )$ be a continuous linear map. For each $k \geq 1 ,$ , f induces a map

$$
f ^ {\otimes k}: V ^ {\otimes k} \to W ^ {\otimes k} \quad \text { s.t. } \quad f ^ {\otimes k} (v _ {1} \otimes \dots \otimes v _ {k}) := f (v _ {1}) \otimes \dots \otimes f (v _ {k}) \quad \text { with } \quad f ^ {\otimes 0} := \text { Id. }
$$

The elements of $V ^ { \otimes k } ~ \subset ~ T ( ( V ) )$ can be interpreted as functions on words of length k over an alphabet $\mathcal { A } _ { d } = 1 , \ldots , d .$ A word $w = i _ { 1 } i _ { 2 } \dots i _ { k }$ corresponds to the basis element $e _ { i _ { 1 } } \otimes \cdots \otimes e _ { i _ { k } }$ in $V ^ { \otimes k }$ . Denote as $\mathcal { W } _ { d } ^ { m }$ the set of all words formed by letters in $\mathbf { \mathcal { A } } _ { d }$ of length $| w | \leq m ,$ , and $\begin{array} { r } { W _ { d } : = \bigcup _ { m \geq 0 } \mathcal { W } _ { d } ^ { m } } \end{array}$ . Let $( A _ { i } ) \in \operatorname { E n d } ( \mathbb { R } ^ { N } )$ (the algebra of endomorphisms of $\mathbb { R } ^ { N }$ under composition with unit $\operatorname { I d } _ { N } )$ , and let $\Gamma _ { A } : T ( ( \mathbb { R } ^ { d } ) ) \to \operatorname { E n d } ( \mathbb { R } ^ { N } )$ be the unital algebra homomorphism

$$
\Gamma_ {A} (\mathbb {G}) = \sum_ {w \in \mathcal {W} _ {d}} A _ {w} \langle \mathbb {G}, w \rangle , \quad \text { where } A _ {w} := \frac {1}{N ^ {\frac {k}{2}}} A _ {i _ {1}} \dots A _ {i _ {k}} \quad \text { for } w = i _ {1} \dots i _ {k} \in \mathcal {W} _ {d}, \tag {11}
$$

extended multiplicatively by $\Gamma _ { A } ( u v ) = \Gamma _ { A } ( u ) \circ \Gamma _ { A } ( v )$ and $\Gamma _ { A } ( 1 ) = \mathrm { I d } _ { N }$ . Here $\langle \mathbb { G } , w \rangle$ denotes the w–coordinate of the tensor $\vec { \mathbb { G } } \in T ( ( \mathbb { R } ^ { d } ) )$ . The map $\Gamma _ { A }$ is well defined on group-like tensors $G ( \mathbb R ^ { d } ) \subset T ( ( \mathbb R ^ { d } ) )$ (see Appendix A.4) and in particular on signature increments of geometric rough paths; we refer to Lemma 6 in Appendix B.2 for the precise statement and proof.

Linear development along a path. For a bounded-variation path $x : [ 0 , T ] \to \mathbb { R } ^ { d }$ set

$$
S _ {t} ^ {A} (x) := \Gamma_ {A} (\operatorname{Sig} (x) _ {0, t}) \in \operatorname{End} (\mathbb {R} ^ {N}), \tag {12}
$$

Lemma 1. Let $\Gamma _ { A }$ be as in Eq. 11. $\textit { I f x } : [ 0 , T ] \  \ \mathbb { R } ^ { d }$ has bounded variation and $x _ { t } ^ { A } : =$ $\textstyle \sum _ { i = 1 } ^ { d } A _ { i } x _ { t } ^ { i }$ , with $A _ { i } \in \operatorname { E n d } ( \mathbb { R } ^ { N } )$ , then $S _ { t } ^ { A } ( x )$ in Eq. 12 is the unique solution of the linear CDE

$$
d S _ {t} ^ {A} (x) = S _ {t} ^ {A} (x) \circ d x _ {t} ^ {A}, \quad S _ {0} ^ {A} (x) = \operatorname{Id} _ {N} \in \operatorname{End} (\mathbb {R} ^ {N}). \tag {13}
$$

See Appendix B.2 for the proof.

By continuity of the Ito–Lyons map (Theorem A.5), Eq. 13 extends to geometric rough drivers. If ˆ $\bar { \mathbb { X } } \in G \Omega _ { p } ( \mathbb { R } ^ { \bar { d } } )$ , there exists a unique matrix–valued geometric p-rough path $\mathbb { S } ^ { A } \in { \tilde { G } } \Omega _ { p } ^ { \mathbf { \bar { \lambda } } } ( \operatorname { E n d } ( \mathbb { R } ^ { N } ) )$ ) given by the canonical lift of the solution to the rough linear equation

$$
d S _ {t} ^ {A} (\mathbb {X}) = S _ {t} ^ {A} (\mathbb {X}) \circ d \mathbb {X} _ {t} \quad S _ {0} ^ {A} (x) = \operatorname{Id} _ {N} \in \operatorname{End} \left(\mathbb {R} ^ {N}\right), \tag {14}
$$

i.e the first level of the lift: $\pi _ { 1 } ( \mathbb { S } _ { 0 , t } ^ { A } ) : = S _ { t } ^ { A } ( \mathbb { X } ) \in \mathrm { E n d } ( \mathbb { R } ^ { N } )$ .

Random RDE features. Let $( A _ { i } ) _ { i = 1 } ^ { d } \stackrel { \mathrm { i . i . d . } } { \sim } \xi _ { N }$ be Gaussian random matrices in $M _ { N } ( \mathbb { R } )$ , and let $\mathbb { S } ^ { A }$ be the matrix–valued geometric p-rough path associated with Eq. 14. Define the one–form

$$
f: \mathbb {R} ^ {N} \longrightarrow \operatorname{Hom} (\operatorname{End} (\mathbb {R} ^ {N}), \mathbb {R} ^ {N}), \qquad f (z) [ M ] := M \big (\varphi (z) \big),
$$

with non-linearity $\varphi \in \mathrm { L i p } ( \gamma )$ (Definition A.5) and $\gamma > p$ . The random-feature path $Z ^ { N } ( \mathbb { X } )$ : $[ 0 , T ] \to \mathbb { R } ^ { N }$ is then defined as the unique solution of the Random RDE

$$
d Z _ {t} ^ {N} (\mathbb {X}) = f \left(Z _ {t} ^ {N}\right) d \mathbb {S} _ {t} ^ {A}, \quad Z _ {0} ^ {N} = z _ {0} \in \mathbb {R} ^ {N}, \tag {15}
$$

where $z _ { 0 } \sim \xi _ { N }$ is independent of $\{ A _ { i } \} _ { i = 1 } ^ { d }$ .

Remark 3.2. In the smooth case (so $\mathbb { X } \equiv \operatorname { S i g } ( x )$ with x of bounded variation), Eq. 15 yields $\begin{array} { r } { d Z _ { t } = \sum _ { i = 1 } ^ { d } \left( S _ { t } ^ { A } ( x ) A _ { i } \varphi ( Z _ { t } ) \right) } \end{array}$ dxit. A brief derivation is provided in Appendix B.2.

We now state and prove two theorems: the first establishes existence and uniqueness, while the second shows convergence to the rough signature kernel introduced in Section 2.2.

Theorem 3.3 (Existence and uniqueness). Let $\mathbb { X } \in G \Omega _ { p } ( \mathbb { R } ^ { d } )$ with $p \geq 1$ and $\varphi \in \mathrm { L i p } ( \gamma )$ with $\gamma > p .$ Then the R-RDE 15 admits a unique solution $Z ^ { N } { \overset { \cdot } { \in } } C ( [ 0 , T ] ; \mathbb { R } ^ { N } )$ , and the Ito–Lyons map ˆ $( \mathbb { X } , \dot { z _ { 0 } } ) \mapsto Z ^ { N }$ is continuous in the rough-path topology.

Proof. This is a direct application of the Universal Limit Theorem (Theorem A.5) under $\operatorname { L i p } ( \gamma )$ vector fields (Definition A.5) with $\gamma > p$ .

Theorem 3.4. Let $\mathbb { X } \in G \Omega _ { p } ( \mathbb { R } ^ { d } )$ and $\mathbb { Y } \in G \Omega _ { q } ( \mathbb { R } ^ { d } )$ be geometric rough paths. Let $Z _ { s } ^ { N } ( \mathbb { X } )$ and $Z _ { t } ^ { N } ( \mathbb { Y } )$ be the solutions of Eq. 15 with $\varphi = \mathrm { i d }$ and the same matrices $\{ A _ { i } \} _ { i = 1 } ^ { d } ( w i t h A _ { i } \sim \xi _ { N } i . i . d . )$ . Then for all $s , t \in [ 0 , T ]$ ,

$$
\lim _ {N \to \infty} \frac {1}{N} \mathbb {E} _ {\xi_ {N}} \left[ \left\langle Z _ {s} ^ {N} (\mathbb {X}), Z _ {t} ^ {N} (\mathbb {Y}) \right\rangle_ {\mathbb {R} ^ {N}} \right] = K _ {\mathrm{Sig}} ^ {\mathbb {X}, \mathbb {Y}} (s, t),
$$

where where $K _ { \mathrm { S i g } } ^ { \mathbb { X } , \mathbb { Y } }$ denotes the rough signature kernel defined in Eq. 7.

We refer the reader to Appendix B.3 for the proof of this theorem.

Gaussian–process interpretation. Analogously to the RCDE and RF-CDE settings, a fixed R-RDE reservoir with a trained linear readout induces with covariance given by the rough signature kernel $K _ { \mathrm { S i g } } ^ { \mathbb { X } , \mathbb { Y } }$ sian–process prior over path–functionals,.

Log–ODE discretization. For rough drivers, na¨ıve Euler schemes ignore the algebraic structure of the signal, breaking Chen’s multiplicativity. The log–ODE method (Lyons, 2014) addresses this by summarizing each time step $[ t _ { i } , t _ { i + 1 } ]$ via the log-signature

$$
\mathfrak {L} _ {i} := \log_ {m} \left(\mathbb {X} _ {t _ {i}, t _ {i + 1}}\right) \in \mathcal {L} ^ {m} (\mathbb {R} ^ {d}),
$$

which maps increments into the step-m free Lie algebra – see Section 2.4. One then advances the state on $[ t _ { i } , t _ { i + 1 } ]$ by solving an ODE with constant Lie coefficients Li. This preserves the group/Chen structure exactly and removes the algebraic redundancies of the tensor algebra.

Let $\widetilde { \mathcal { W } } _ { d } ^ { m }$ be a fixed Hall/Lyndon basis of Lie words of length $| w | \leq m$ . Given a collection of random matrices $\{ B _ { 1 } , \ldots , B _ { d } \}$ with $B _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } \xi _ { N }$ , define the linear map $\Pi _ { B } : { \mathcal { L } } ^ { m } ( \mathbb { R } ^ { d } ) \to \operatorname { E n d } ( \mathbb { R } ^ { N } )$ by

$$
\Pi_ {B} (\mathbb {G}) := \sum_ {w \in \widetilde {\mathcal {W}} _ {d} ^ {m}} B ^ {(w)} \left\langle \log_ {m} (\mathbb {G} _ {t _ {i}, t _ {i + 1}}), w \right\rangle , \qquad B ^ {(w)} := \frac {1}{N ^ {\frac {k}{2}}} [ \dots [ [ B _ {i _ {1}}, B _ {i _ {2}} ], B _ {i _ {3}} ], \ldots , B _ {i _ {k}} ]
$$

for $w = i _ { 1 } \cdots i _ { k }$ , where $[ A , B ] = A B - B A$ is the Lie bracket.

Given $\mathbb { X } \in G \Omega _ { m } ( \mathbb { R } ^ { d } )$ and a partition $\mathcal { D } _ { M } = \{ 0 = t _ { 0 } < \cdot \cdot \cdot < t _ { M } = T \}$ , we update

$$
\widetilde {Z} _ {t _ {i + 1}} ^ {N} = \widetilde {Z} _ {t _ {i}} ^ {N} + \Pi_ {B} (\mathbb {X} _ {t _ {i}, t _ {i + 1}}) \varphi \big (\widetilde {Z} _ {t _ {i}} ^ {N} \big), \quad \widetilde {Z} _ {t _ {0}} ^ {N} = z _ {0} \in \mathbb {R} ^ {N}, \tag {16}
$$

which uses only the log-signature coefficients $\langle \log _ { m } ( \mathbb { X } _ { t _ { i } , t _ { i + 1 } } ) , w \rangle$ and the corresponding commutators $B ^ { ( w ) }$ ), keeping the discretization faithful to the rough-path algebra while remaining explicit. Scaling and bias hyperparameters are incorporated analogously to the RF-CDE discretization (Eq. 10) and tuned by grid search.

While more computationally expensive than RF–CDE, R–RDE can be advantageous on very long sequences: the log-ODE discretisation operates on log-signatures computed over chunks, thereby shortening the effective trajectory length fed to the random differential equation.

# 4 EXPERIMENTS

In this section, we detail our random differential equation models’ performance for time series classification. We implement RF-CDE via the discretization in Eq. 10 and R-RDE via the log-ODE scheme in Eq. 16. For completeness, we also benchmark the R-CDE of Cirone et al. (2023) – described in Section 2.1 – which, to our knowledge, has not been tested.

Benchmarks. We compare against Random Fourier Signature Features (RFSF) in the two projection variants of Toth et al. (2025) – Diagonal Projection (DP) and Tensorized Random Projection (TRP). We also benchmark the PDE-based signature kernel (SigPDE) with the RBF base kernel (Salvi et al., 2021a), and standard time-series baselines: Random Fourier Features (RFF), RBF, GAK, and Random Warping Series (RWS). For space, full results for RBF, GAK, RWS, and RFF are deferred to Appendix C, as these methods rarely attain state-of-the-art performance on our suite but are included for completeness. We do not include the truncated signature kernel (Kiraly & Ober- ´ hauser, 2019) in our benchmarks, as its exponential feature explosion makes it impractical in all but small and low-dimensional datasets. Neural benchmark such as Neural CDE (Kidger et al., 2020) and Neural RDE (Morrill et al., 2021) have also been included in the ablation study of UEA time series classification and in the Hurst-exponent classification experiment.

Remark 4.1. SigPDE, GAK, and RBF are not random-feature methods: in the SVM setting they require computing and inverting the kernel Gram matrix, which can be a bottleneck as the number of samples grows. By contrast, random-feature models learn only a linear readout on top of fixed random dynamics, avoiding kernel matrix operations and retaining linear-in-samples complexity.

Computational Time. Table 4 summarizes the computational complexity of our signature-based random-feature extractors. These models scale linearly with the sequence length ℓ, unlike kernel baselines such as SigPDE, RBF, and GAK which scale quadratically. Among our methods, R-RDE is typically the slowest due to an additional $O ( N ^ { 3 } )$ component arising from matrix development; however, this cubic term is independent of the batch size B, so it can be precomputed.

<table><tr><td>R-CDE</td><td>RFCDE (ours)</td><td>R-RDE (ours)</td><td>RFSF-DP</td><td>RFSF-TRP</td></tr><tr><td> $O(B\ell N^{2}d)$ </td><td> $O(B\ell F(N^{2}+d))$ </td><td> $O(N^{2}d^{M}(B\ell+N))$ </td><td> $O(B\ell F(Md+2^{M}))$ </td><td> $O(B\ell M(dF+F^{2}))$ </td></tr></table>

Table 1: Asymptotic feature-extraction cost (ignoring the final linear readout). Here B is batch size, F the number of random Fourier features, d the input dimension, ℓ the sequence length, M the truncation level of the signature, and N the (output) feature dimension.

# 4.1 UEA TIME SERIES CLASSIFICATION

UEA Datasets. The UEA archive (Dau et al., 2019) is a collection of datasets for benchmarking classifiers on multivariate time series classification problems. The data modality ranges from various sources e.g. human activity recognition, motion and ECG classification, audio spectra recognition, and others. A summary of the dataset characteristics can be found in Table 2 in Dau et al. (2019).

Experimental Setup. Full details, including the grid-search ranges, are deferred to Appendix C.1.

Results. With 250 random features, RF-CDE is the strongest of random-feature models on average, followed by the non-parametric SigPDE baseline. Averaging across the 16 UEA datasets, RF-CDE attains 0.741 accuracy versus 0.738 for SigPDE, while RFSF variants are slightly behind (0.725–0.726), and R-RDE and R-CDE trail further (0.708 and 0.695, respectively). RF-CDE is particularly competitive on medium-difficulty tasks (e.g., Libras, NATOPS), and R-RDE occasionally leads among random-feature methods on structure-rich datasets (e.g., UWaveGestureLibrary). These trends suggest that injecting an RBF lift before the controlled dynamics (RF-CDE) is an effective way to capture local geometry in continuous time, whereas the rough-path variant (R-RDE) helps when higher-order interactions matter. Table 4.1 reports the results.

<table><tr><td></td><td>R-CDE</td><td>RF-CDE</td><td>R-RDE</td><td>RFSF-DP</td><td>RFSF-TRP</td><td>SigPDE</td></tr><tr><td>ArticularyWordRecognition</td><td>0.950</td><td>0.967</td><td>0.957</td><td>0.977</td><td>0.983</td><td>0.983</td></tr><tr><td>AtrialFibrillation</td><td>0.333</td><td>0.467</td><td>0.467</td><td>0.400</td><td>0.266</td><td>0.333</td></tr><tr><td>BasicMotions</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.975</td><td>0.975</td><td>1.000</td></tr><tr><td>Cricket</td><td>0.972</td><td>0.972</td><td>0.902</td><td>0.972</td><td>0.958</td><td>0.972</td></tr><tr><td>EigenWorms</td><td>0.420</td><td>0.630</td><td>0.612</td><td>0.786</td><td>0.771</td><td>0.794</td></tr><tr><td>Epilepsy</td><td>0.935</td><td>0.971</td><td>0.935</td><td>0.942</td><td>0.942</td><td>0.891</td></tr><tr><td>EthanolConcentration</td><td>0.312</td><td>0.358</td><td>0.373</td><td>0.430</td><td>0.407</td><td>0.460</td></tr><tr><td>FingerMovements</td><td>0.550</td><td>0.550</td><td>0.530</td><td>0.570</td><td>0.530</td><td>0.610</td></tr><tr><td>Handwriting</td><td>0.331</td><td>0.331</td><td>0.331</td><td>0.380</td><td>0.362</td><td>0.409</td></tr><tr><td>Libras</td><td>0.889</td><td>0.911</td><td>0.867</td><td>0.833</td><td>0.906</td><td>0.867</td></tr><tr><td>NATOPS</td><td>0.872</td><td>0.944</td><td>0.906</td><td>0.889</td><td>0.906</td><td>0.928</td></tr><tr><td>RacketSports</td><td>0.809</td><td>0.809</td><td>0.737</td><td>0.829</td><td>0.809</td><td>0.849</td></tr><tr><td>SelfRegulationSCP1</td><td>0.877</td><td>0.877</td><td>0.840</td><td>0.887</td><td>0.904</td><td>0.904</td></tr><tr><td>SelfRegulationSCP2</td><td>0.555</td><td>0.555</td><td>0.567</td><td>0.483</td><td>0.494</td><td>0.544</td></tr><tr><td>StandWalkJump</td><td>0.467</td><td>0.667</td><td>0.400</td><td>0.400</td><td>0.533</td><td>0.400</td></tr><tr><td>UWaveGestureLibrary</td><td>0.853</td><td>0.844</td><td>0.903</td><td>0.856</td><td>0.853</td><td>0.866</td></tr><tr><td>Avg. acc. (↑)</td><td>0.695</td><td>0.741</td><td>0.708</td><td>0.726</td><td>0.725</td><td>0.738</td></tr><tr><td>Avg. rank (↓)</td><td>4.250</td><td>3.062</td><td>4.125</td><td>3.406</td><td>3.594</td><td>2.562</td></tr></table>

Table 2: UEA test accuracies with N = 250 signature-based random-feature models (SigPDE is a kernel baseline: no random features). For each row, the best result is highlighted in bold.

Ablation: Number of Features. We repeat the full protocol with 64 and 500 random features. In the low–budget setting, RF-CDE performs particularly well relative to the RFSF and neural baselines. On the other hand, doubling the feature budget yields modest gains – typically a few percentage points on the more challenging datasets – while leaving easier tasks essentially unchanged. Being a kernel methods, SigPDE is unaffected by this ablation. Results are included in Appendix C.

# 4.2 CLASSIFICATION OF ROUGH SIGNALS VIA HURST EXPONENT RECOVERY

To further assess the ability of our model to extract fine-grained geometric information from irregular time-series data, we introduce a controlled classification task based on synthetic fractional Brownian motion (fBm). Each sample consists of fBm with Hurst exponent $H \in \{ 0 . 0 5 , 0 . 1 5 , \ldots , 0 . 7 5 \}$ , and the goal is to correctly identify the underlying Hurst parameter from the observed path. Recall that H controls the roughness of the process through its p-variation.

We evaluate two variants of the task. V1 uses the raw fBm trajectories. V2 applies per-sample standardisation (zero mean and unit variance), thereby forcing the models to exploit only geometric features and long-range dependence. Full details of this experiment are deferred to Appendix C.5. Table 3 reports classification accuracies across a range of competing baselines. In both settings, our R-RDE model achieves consistently strong performance, and in the most challenging regime (V2, N = 64), it preserves a clear performance margin over alternative approaches.

Table 3: Hurst–exponent classification accuracy. Here N denotes the feature dimension of the random-feature models (with neural baselines matched in parameter count). 

<table><tr><td>Setting</td><td>R-CDE</td><td>RF-CDE</td><td>R-RDE</td><td>RFSF-DP</td><td>RFSF-TRP</td><td>NCDE</td><td>NRDE</td></tr><tr><td> $\mathbf{V1 - N = 64}$ </td><td>0.870</td><td>0.895</td><td> $\mathbf{0.955}$ </td><td>0.840</td><td>0.895</td><td>0.905</td><td>0.920</td></tr><tr><td> $\mathbf{V1 - N = 100}$ </td><td>0.900</td><td>0.945</td><td> $\mathbf{0.950}$ </td><td>0.890</td><td>0.910</td><td>0.895</td><td>0.945</td></tr><tr><td> $\mathbf{V2 - N = 64}$ </td><td>0.635</td><td>0.645</td><td> $\mathbf{0.735}$ </td><td>0.630</td><td>0.650</td><td>0.650</td><td>0.675</td></tr><tr><td> $\mathbf{V2 - N = 100}$ </td><td>0.650</td><td>0.695</td><td> $\mathbf{0.730}$ </td><td>0.675</td><td>0.675</td><td>0.650</td><td>0.685</td></tr></table>

# 4.3 ROBUSTNESS TO MISSING DATA

To assess the robustness of our models to incomplete observations, we perform an additional experiment on multivariate time series from the UEA archive. We synthetically introduce missing data by randomly removing individual time points along the test trajectories. The experiment is described in Appendix C.4, and the corresponding accuracy tables are provided therein (Tables 7 and 8). The results show that our models remains competitive even under substantial information loss: in particular, RF–CDE exhibits the most stable performance as the missingness level increases.

# 5 CONCLUSIONS

We introduced a training-efficient framework for time-series learning based on random continuoustime reservoirs whose infinite-width limits coincide with established path kernels: RF-CDE yields the RBF-lifted signature kernel, and R-RDE yields the rough signature kernel. This places our models on the same framework as infinite-width neural networks. Empirically, with only a few hundred features, both models are competitive on UEA benchmarks while avoiding kernel-matrix inversion and scaling linearly in sequence length. The result is a scalable alternative to explicit signature computation.

Future Directions. Looking forward, we see natural extensions: learn (or sparsify) the spectral measures that define the reservoirs, and couple our continuous-time features with probabilistic heads for calibrated uncertainty and streaming inference. It will also be interesting to study NTK dynamics around the random reservoir, to design adaptive log-structured discretizations for very long contexts.

# ACKNOWLEDGMENTS

TC has been supported in part by UK Research and Innovation (UKRI) through the Engineering and Physical Sciences Research Council (EPSRC) Programme Grant: UKRI1010 - High order mathematical and computational infrastructure for streamed data that enhance contemporary generative and large language models. FP and WFT have been supported by the EPSRC Centre for Doctoral Training in Mathematics of Random Systems: Analysis, Modelling and Simulation (EP/S023925/1). ChatGPT-5 was used by FP to refine the clarity and style of selected paragraphs. For the purpose of open access, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising.

# REFERENCES

Francesca Biagini, Lukas Gonon, and Niklas Walter. Universal randomised signatures for generative time series modelling. arXiv preprint arXiv:2406.10214, 2024.   
Thomas Cass and Cristopher Salvi. Lecture notes on rough paths and applications to machine learning. arXiv preprint arXiv:2404.06583, 2024.   
Thomas Cass and William F Turner. Free probability, path developments and signature kernels as universal scaling limits. arXiv preprint arXiv:2402.12311, 2024.   
Thomas Cass, Francesco Piatti, and Jeffrey Pei. Numerical schemes for signature kernels. arXiv preprint arXiv:2502.08470, 2025.   
Ilya Chevyrev and Andrey Kormilitzin. A primer on the signature method in machine learning. arXiv preprint arXiv:1603.03788, 2016.   
Nicola Muca Cirone, Maud Lemercier, and Cristopher Salvi. Neural signature kernels as infinitewidth-depth-limits of controlled resnets. In International Conference on Machine Learning, pp. 25358–25425. PMLR, 2023.   
Thomas Cochrane, Peter Foster, Varun Chhabra, Maud Lemercier, Terry Lyons, and Cristopher Salvi. Sk-tree: a systematic malware detection algorithm on streaming trees via the signature kernel. In 2021 IEEE international conference on cyber security and resilience (CSR), pp. 35–40. IEEE, 2021.   
Enea Monzio Compagnoni, Anna Scampicchio, Luca Biggio, Antonio Orvieto, Thomas Hofmann, and Josef Teichmann. On the effectiveness of randomized signatures as reservoir for learning rough dynamics. In 2023 International Joint Conference on Neural Networks (IJCNN), pp. 1–8. IEEE, 2023.   
Christa Cuchiero, Lukas Gonon, Lyudmila Grigoryeva, Juan-Pablo Ortega, and Josef Teichmann. Discrete-time signatures and randomness in reservoir computing. IEEE Transactions on Neural Networks and Learning Systems, 33(11):6321–6330, 2021.   
Hoang Anh Dau, Anthony Bagnall, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu, Shaghayegh Gharghabi, Chotirat Ann Ratanamahatana, and Eamonn Keogh. The ucr time series archive. IEEE/CAA Journal of Automatica Sinica, 6(6):1293–1305, 2019.   
RJ Defranco. Gronwall’s inequality for systems of multiple volterra integral equations. Funkcialaj Ekvacioj, 19(1):1–9, 1976.   
Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.   
Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured ´ state spaces. arXiv preprint arXiv:2111.00396, 2021.   
Ben Hambly and Terry Lyons. Uniqueness for the signature of a path of bounded variation and the reduced path group. Annals of Mathematics, pp. 109–167, 2010.

Guang-Bin Huang. An insight into extreme learning machines: random neurons, random features and kernels. Cognitive computation, 6(3):376–390, 2014.   
Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and gen- ´ eralization in neural networks. Advances in neural information processing systems, 31, 2018.   
Herbert Jaeger. Echo state network. scholarpedia, 2(9):2330, 2007.   
Sheo Yon Jhin, Heejoo Shin, Sujie Kim, Seoyoung Hong, Minju Jo, Solhee Park, Noseong Park, Seungbeom Lee, Hwiyoung Maeng, and Seungmin Jeon. Attentive neural controlled differential equations for time-series classification and forecasting. Knowledge and Information Systems, 66 (3):1885–1915, 2024.   
Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural controlled differential equations for irregular time series. Advances in neural information processing systems, 33:6696–6707, 2020.   
Franz J Kiraly and Harald Oberhauser. Kernels for sequentially ordered data. ´ Journal of Machine Learning Research, 20(31):1–45, 2019.   
Maud Lemercier, Cristopher Salvi, Thomas Cass, Edwin V Bonilla, Theodoros Damoulas, and Terry J Lyons. Siggpde: Scaling sparse gaussian processes on sequential data. In International Conference on Machine Learning, pp. 6233–6242. PMLR, 2021a.   
Maud Lemercier, Cristopher Salvi, Theodoros Damoulas, Edwin Bonilla, and Terry Lyons. Distribution regression for sequential data. In International Conference on Artificial Intelligence and Statistics, pp. 3754–3762. PMLR, 2021b.   
Mantas Lukosevi ˇ cius and Herbert Jaeger. Reservoir computing approaches to recurrent neural net- ˇ work training. Computer science review, 3(3):127–149, 2009.   
Terry Lyons. Rough paths, signatures and the modelling of functions on streams. arXiv preprint arXiv:1405.4537, 2014.   
Terry J Lyons. Differential equations driven by rough signals. Revista Matematica Iberoamericana ´ , 14(2):215–310, 1998.   
Wolfgang Maass, Thomas Natschlager, and Henry Markram. Real-time computing without stable ¨ states: A new framework for neural computation based on perturbations. Neural computation, 14 (11):2531–2560, 2002.   
James Morrill, Cristopher Salvi, Patrick Kidger, and James Foster. Neural rough differential equations for long time series. In International Conference on Machine Learning, pp. 7829–7838. PMLR, 2021.   
Radford M Neal. Priors for infinite networks. In Bayesian learning for neural networks, pp. 29–53. Springer, 1996.   
ROYUD Nishino and Shohei Hido Crissman Loomis. Cupy: A numpy-compatible library for nvidia gpu calculations. 31st confernce on neural information processing systems, 151(7), 2017.   
Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. Advances in neural information processing systems, 20, 2007.   
Ali Rahimi and Benjamin Recht. Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. Advances in neural information processing systems, 21, 2008.   
Syama Sundar Rangapuram, Matthias W Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. Advances in neural information processing systems, 31, 2018.   
Cristopher Salvi, Thomas Cass, James Foster, Terry Lyons, and Weixin Yang. The signature kernel is the solution of a goursat pde. SIAM Journal on Mathematics of Data Science, 3(3):873–899, 2021a.

Cristopher Salvi, Maud Lemercier, Chong Liu, Blanka Horvath, Theodoros Damoulas, and Terry Lyons. Higher order kernel mean embeddings to capture filtrations of stochastic processes. Advances in Neural Information Processing Systems, 34:16635–16647, 2021b.   
Csaba Toth and Harald Oberhauser. Bayesian learning from sequential data using gaussian processes with signature covariances. In International Conference on Machine Learning, pp. 9548–9560. PMLR, 2020.   
Csaba Toth, Danilo Jr Dela Cruz, and Harald Oberhauser. A user’s guide to ksig: Gpu-accelerated ´ computation of the signature kernel. arXiv preprint arXiv:2501.07145, 2025.   
Csaba Toth, Harald Oberhauser, and Zoltan Szabo. Random fourier signature features. SIAM Journal on Mathematics of Data Science, 7(1):329–354, 2025.   
Christopher Williams. Computing with infinite networks. Advances in neural information processing systems, 9, 1996.

# A ALGEBRAIC AND ANALYTIC BACKGROUND

This appendix collects the algebraic objects (tensor and free Lie algebras, group-like elements), the analytic objects (signatures, rough paths), and the two key theorems we rely on (the PDE characterization of the signature kernel and Lyons’ Universal Limit Theorem). We keep statements self-contained; detailed proofs can be found in standard references (Cass & Salvi, 2024; Lyons, 1998).

# A.1 TENSOR ALGEBRA

Let V be a Banach space. The spaces of formal polynomials and formal power series over $V$ are defined respectively as

$$
T (V) = \bigoplus_ {k = 0} ^ {\infty} V ^ {\otimes k}, \quad \text { and } \quad T ((V)) = \prod_ {k = 0} ^ {\infty} V ^ {\otimes k},
$$

where $V ^ { \otimes k }$ denotes the k-fold tensor product of V . Both $T ( V )$ and $T ( ( V ) )$ can be endowed with the operations of component-wise addition and multiplication ⊗, the latter defined for any two elements $\mathbb { A } = ( a _ { 0 } , a _ { 1 } , \ldots )$ and $\mathbb { B } = ( b _ { 0 } , b _ { 1 } , \ldots )$ as

$$
\mathbb {A} \otimes \mathbb {B} = (c _ {0}, c _ {1}, c _ {2}, \dots), \quad \text { where } \quad V ^ {\otimes k} \ni c _ {k} = \sum_ {i = 0} ^ {k} a _ {i} \otimes b _ {k - i}, \forall k \geq 0.
$$

When endowed with these two operations and the natural action of R by $\lambda \mathbb { A } = ( \lambda a _ { 0 } , \lambda a _ { 1 } , . . . )$ , $T ( ( V ) )$ becomes a real, non-commutative unital algebra with unit $1 = ( 1 , \dot { 0 } , 0 , \dots )$ called the tensor algebra.

The level-m truncated tensor algebra over V of order $m \in \mathbb { N }$ is the quotient

$$
T ^ {m} (V) := T ((V)) / T ^ {> m} (V) \cong \bigoplus_ {k = 0} ^ {m} V ^ {\otimes k},
$$

where

$$
T ^ {> m} (V) := \{\mathbb {A} = (a ^ {0}, a ^ {1}, \dots) \in T ((V)): a ^ {0} = \dots = a ^ {m} = 0 \}
$$

We denote by $\pi _ { \leq m } : T ( ( V ) ) \to T ^ { m } ( V )$ the canonical projection and by $\pi _ { k } : T ( ( V ) ) \to V ^ { \otimes k }$ th e level maps.

Finally, we can define a norm on $V ^ { \otimes k }$ as

$$
\| v \| _ {V ^ {\otimes k}} = \sqrt {\prod_ {i = 1} ^ {k} \left\langle v _ {i} , v _ {i} \right\rangle_ {V}}, \quad \text { for   } v = v _ {i} \dots v _ {k}. \tag {17}
$$

# A.2 SIGNATURES AND THEIR BASIC PROPERTIES

Signatures extend iterated integrals beyond smooth curves, but to do so we must quantify how “rough” a path is. The right scale is p-variation: it measures the cumulative oscillation of a path and yields a topology under which Young integrals (for $p < 2 )$ and, more generally, rough integrals (for $p \geq 2 )$ are well posed.

Definition $\mathbf { A . 1 } \left( p \mathbf { - } \mathrm { v a r i a t i o n } \right)$ . Let $x : [ 0 , T ] \to V$ be continuous. For any $[ s , t ] \subseteq [ 0 , T ]$ ,

$$
\| x \| _ {p - \text {var}, [ s, t ]} = \left(\sup _ {\mathcal {D} \subset [ s, t ]} \sum_ {t _ {i} \in \mathcal {D}} \left\| x _ {t _ {i + 1}} - x _ {t _ {i}} \right\| ^ {p}\right) ^ {1 / p},
$$

where the supremum is over all partitions D $o f [ s , t ] .$ .

The induced p-variation metric on $C ( [ 0 , T ] ; V )$ is

$$
d _ {p \text {-var}} (x, y) := \| x - y \| _ {p \text {-var}, [ 0, T ]}.
$$

Let $x : [ 0 , T ] \to V$ be a path of bounded variation. Its signature over $[ s , t ]$ is

$$
\operatorname{Sig} (x) _ {s, t} = \bigl (1, S ^ {1} (x), S ^ {2} (x), \dots \bigr), \quad S ^ {k} (x) = \int_ {s <   u _ {1} <   \dots <   u _ {k} <   t} d x _ {u _ {1}} \otimes \dots \otimes d x _ {u _ {k}}.
$$

The signature satisfies Chen’s identity (it is multiplicative):

$$
\operatorname{Sig} (x) _ {s, u} \otimes \operatorname{Sig} (x) _ {u, t} = \operatorname{Sig} (x) _ {s, t}
$$

Lemma 2 (Factorial decay). For x of bounded variation and all $k \geq 1$ ,

$$
\left\| \int_ {s <   u _ {1} <   \dots <   u _ {k} <   t} \dots \int d x _ {u _ {1}} \otimes \dots \otimes d x _ {u _ {k}} \right\| _ {V ^ {\otimes k}} \leq \frac {(\| x \| _ {1 - v a r ; [ s , t ]}) ^ {k}}{k !}.
$$

where the norm on $V ^ { \otimes k }$ is given by Eq. 17 and $\| \cdot \| _ { 1 \cdot \nu a r ; [ s , t ] }$ denotes the 1-variation norm given by Definition A.1.

Three foundational properties make signatures effective for learning:

Theorem A.1 (Uniqueness up to tree-like equivalence). $I f \mathrm { S i g } ( x ) _ { 0 , T } = \mathrm { S i g } ( y ) _ { 0 , T }$ then x and y are tree-like equivalent; conversely, tree-like equivalent paths have equal signatures. On classes where tree-like structure is ruled out (e.g. reduced paths), the signature is injective.

Theorem A.2 (Universality of linear functionals on signatures, (Hambly & Lyons, 2010)). Let K be a compact set of bounded-variation paths (modulo tree-like equivalence). Then the linear span of coordinate iterated integrals $\{ \langle \ell , \mathrm { S i g } \dot { ( \cdot ) } _ { 0 , T } \rangle : \ell \in T ( ( V ) )$ finite} is dense in $C ( \kappa )$ , the space of continuous functions on K with the topology of uniform convergence.

Lemma 3 (Reparameterization invariance). Let $x : [ t _ { 0 } , T ] \to \mathbb { R } ^ { d }$ be a continuous path of bounded variation and let $[ a , b ]$ and $[ c , d ]$ be two subintervals of $[ t _ { 0 } , T ]$ . Let $\lambda : [ c , d ] \to [ a , { \bar { b } } ]$ be a reparameterization. Then $\mathrm { \bar { S i g } } ( x ) _ { a , b } \mathrm { \bar { = } } \bar { S } ( x \circ \lambda ) _ { c , d } .$ .

For some applications it might be important to keep the time parameterization of the path x. In this case, it suffices to add time as an extra coordinate of x to get the time-augmented path $\widehat { x } : t \mapsto \left( t , x _ { t } \right)$ .

# A.3 SIGNATURE KERNELS AND THE PDE CHARACTERIZATION

Kernel methods measure similarity via inner products in a (possibly infinite-dimensional) feature space. This is formalized by the notion of a reproducing kernel Hilbert space (RKHS): a Hilbert space of functions in which point evaluation is a continuous linear functional represented by the kernel. We recall the standard definition below.

Definition A.2. Let X be a nonempty set and k be a positive definite kernel on X . A Hilbert space $\mathcal { H } _ { k }$ of real-valued functions on X equipped with an inner product $\langle \cdot , \cdot \rangle _ { \mathcal { H } _ { k } }$ is called a reproducing kernel Hilbert space (RKHS) with reproducing kernel k, if for any $x \in \mathcal { X }$ and for any $f \in \mathcal { H } _ { k }$ the following two conditions are satisfied:

1. the feature map $\boldsymbol { k } ( \boldsymbol { x } , \cdot ) \in \mathcal { H } _ { k } ;$   
2. the reproducing property $f ( x ) = \langle f , k ( x , \cdot ) \rangle _ { \mathcal { H } _ { k } }$ holds.

Intuitively, the element $k ( x , \cdot ) \in \mathcal { H } _ { k }$ plays the role of an (often infinite-dimensional) feature vector for x. An immediate consequence of the reproducing property is the feature–space inner product identity

$$
k (x, y) = \langle k (x, \cdot), k (y, \cdot) \rangle_ {\mathcal {H} _ {k}}, \qquad x, y \in \mathcal {X}.
$$

In our setting, the feature map of a path is its signature – the sequence of iterated integrals living in the tensor algebra. Given an (Hilbert-Schmidt) inner product $\langle \cdot , \cdot \rangle _ { V }$ on $V ,$ define for $k \geq 1$

$$
\left\langle v _ {1} \otimes \dots \otimes v _ {k}, w _ {1} \otimes \dots \otimes w _ {k} \right\rangle_ {V ^ {\otimes k}} := \prod_ {j = 1} ^ {k} \left\langle v _ {j}, w _ {j} \right\rangle_ {V}.
$$

For A = (a0, a1, . . .) and B = (b0, b1, . . .) in T ((V )), set

$$
\left\langle \mathbb {A}, \mathbb {B} \right\rangle_ {T ((V))} := \sum_ {k = 0} ^ {\infty} \left\langle a _ {k}, b _ {k} \right\rangle_ {V ^ {\otimes k}}.
$$

This induces the signature kernel

$$
K _ {\mathrm{Sig}} ^ {x, y} (s, t) := \left\langle \operatorname{Sig} (x) _ {0, s}, \operatorname{Sig} (y) _ {0, t} \right\rangle_ {T ((V))}.
$$

When $x , y$ are differentiable, $K _ { \mathrm { S i g } } ^ { x , y }$ is characterized as the unique solution to a linear hyperbolic Goursat PDE.

Theorem A.3 (PDE/Volterra characterization, (Salvi et al., 2021a)). Let x, $y \in C ^ { 1 } ( [ 0 , T ] ; V )$ . Then $k ( s , t ) : = K _ { \mathrm { S i g } } ^ { x , y } ( s , t )$ solves

$$
\partial_ {s} \partial_ {t} k (s, t) = \langle \dot {x} _ {s}, \dot {y} _ {t} \rangle_ {V} k (s, t), \quad k (s, 0) = k (0, t) = 1,
$$

equivalently,

$$
k (s, t) = 1 + \int_ {0} ^ {s} \int_ {0} ^ {t} \left\langle \dot {x} _ {u}, \dot {y} _ {v} \right\rangle_ {V} k (u, v) d v d u.
$$

Conversely, the (unique) solution of this problem coincides with $\langle \mathrm { S i g } ( x ) _ { 0 , s } , \mathrm { S i g } ( y ) _ { 0 , t } \rangle$ .

A direct corollary is a universality statement for the induced kernel on path space.

Theorem A.4 (Universality/characteristicness of the signature kernel). On compact sets of paths (modulo tree-like equivalence), the signature kernel is universal (its RKHS is dense in continuous functions) and characteristic (mean embeddings of Borel probability measures are injective).

# A.4 ROUGH PATHS AND LIE ALGEBRAS

For tensor-valued paths, the p-variation extends in the usual rough-path way: apply p-variation levelwise to each homogeneous tensor component and combine (up to an equivalent norm) to obtain the standard p-variation metric on $T ^ { \lfloor p \rfloor } ( V )$ .

Definition A.3 (p-variation metric). The p-variation metric of two p-rough paths $\mathbb { X } , \mathbb { Y } : \Delta _ { T } $ $T ^ { \lfloor p \rfloor } \left( V \right)$ is defined as follows

$$
d _ {p} (\mathbb {X}, \mathbb {Y}) = \max _ {1 \leq k \leq \lfloor p \rfloor} \sup _ {\mathcal {D}} \left(\sum_ {t _ {k} \in \mathcal {D}} \left\| \pi_ {k} (\mathbb {X} _ {t _ {i}, t _ {i + 1}}) - \pi_ {k} (\mathbb {Y} _ {t _ {i}, t _ {i + 1}}) \right\| _ {V ^ {\otimes k}} ^ {p / k}\right) ^ {1 / p},
$$

where the supremum is taken over all partitions D of the interval [0, T ], and the norm on $V ^ { \otimes k }$ is given by Eq. 17.

This metric is the natural topology for rough paths (Definition 2.1): the class of geometric p-rough paths $G \Omega _ { p } ( V )$ is the closure, under $d _ { p } ,$ of truncated signatures of bounded-variation paths (Definition 2.2).

The associative tensor algebra $( T ( \left( V \right) ) , \otimes )$ carries a commutator

$$
[ \mathbb {A}, \mathbb {B} ] = \mathbb {A} \otimes \mathbb {B} - \mathbb {B} \otimes \mathbb {A}.
$$

Equipped with $[ \cdot , \cdot ] .$ , the same underlying vector space becomes a (noncommutative) Lie algebra. Iterated brackets quantify non-commutativity and generate the “Lie words” that will organize the algebraic content of iterated integrals.

Definition A.4 (Lie polynomials and Lie series). Let $L _ { 0 } = 0 , L _ { 1 } = V$ , and $L _ { k + 1 } = [ V , L _ { k } ]$ , with [V, U ] denoting the linear span of all elements of the form $[ e , f ]$ where $( e , f ) \in V \times U$ for any two linear subspaces V, U of T ((V )).

The space of Lie polynomials over V , denoted as ${ \mathcal { L } } ( V )$ , is defined as:

$$
\mathcal {L} (V) = \bigoplus_ {k = 0} ^ {\infty} L _ {k}.
$$

The space of Lie formal series over V , denoted as ${ \mathcal { L } } ( ( V ) ) \subset T ( ( V ) )$ is defined as

$$
\mathcal {L} ((V)) = \left\{\mathbb {L} = \left(l ^ {0}, l ^ {1}, \dots\right) \mid \forall k \geq 0, l ^ {k} \in L _ {k} \right\}.
$$

For any $n \geq 1$ , the step-n free Lie algebra is defined by ${ \mathcal { L } } ^ { n } ( V ) : = \pi _ { \leq n } ( { \mathcal { L } } ( ( V ) ) )$ with elements called Lie polynomials of degree n. Define the (formal) exponential and logarithm w.r.t. tensor multiplication,

$$
\exp (\mathbb {A}) := \sum_ {n = 0} ^ {\infty} \frac {\mathbb {A} ^ {\otimes n}}{n !}, \quad \log (\mathbb {1} + \mathbb {A}) := \sum_ {n = 1} ^ {\infty} \frac {(- 1) ^ {n - 1}}{n} \mathbb {A} ^ {\otimes n}, \tag {18}
$$

and their level-m truncations $\exp _ { m } : = \pi _ { \leq m } \circ \exp , \log _ { m } : = \pi _ { \leq m }$ ◦ log.

The group-like subset

$$
G (V) := \exp \left(\mathcal {L} ((V))\right) \subset T ((V))
$$

is a Lie group under the tensor product; at level n we write $G ^ { n } ( V ) : = \pi _ { \leq n } ( G ( V ) )$ ) with mutually inverse maps

$$
\exp : \mathcal {L} ((V)) \to G (V) \qquad \text { and } \qquad \log : G (V) \to \mathcal {L} ((V)),
$$

and

$$
\exp_ {n}: \mathcal {L} ^ {n} (V) \to G ^ {n} (V) \qquad \text { and } \qquad \log_ {n}: G ^ {n} (V) \to \mathcal {L} ^ {n} (V).
$$

For any bounded-variation path (and, by continuity, for geometric rough paths), the signature $\mathrm { S i g } ( x ) _ { s , t } \in G ( V )$ is group-like. Correspondingly, the log-signature $\log ( \operatorname { S i g } ( x ) _ { s , t } )$ lies in the free Lie algebra $\mathcal { L } ( ( V ) )$ (or its truncation ${ \mathcal { L } } ^ { \bar { n } } ( V )$ at finite step). This identification is what allows Liealgebraic discretizations (e.g. log-ODE schemes) that respect the path’s multiplicative structure.

# A.5 ROUGH DIFFERENTIAL EQUATIONS (RDES)

This section follows Cass & Salvi (2024) and gives a precise pathwise meaning to rough differential equations (RDEs)

$$
d Y _ {t} = f \left(Y _ {t}\right) d \mathbb {X} _ {t}, \quad Y _ {0} = y _ {0} \in \mathbb {R} ^ {N} \tag {19}
$$

driven by a geometric rough path X (Definition 2.2).

The main idea is to approximate the (geometric) rough path $\mathbb { X } \ \in \ G \Omega _ { p } ( V )$ by signatures of smooth/bounded–variation paths in p-variation, solve the ordinary controlled differential equations (CDEs) along those smooth drivers, and define the rough solution as the uniform limit of the smooth solutions. The key tool is Lyons’ Universal Limit Theorem (Theorem A.5), which also yields stability/continuity of the solution map (the Ito –Lyons map). ˆ

Before stating it we recall the definition of γ-Lipschitz function (in the sense of Stein).

Definition ${ \bf A } . { \bf 5 } \ \mathrm { ( L i p } ( \gamma )$ functions). Let $V , W$ be two normed space and let $\gamma > 0 .$ . A function $g : V  W$ is called γ-Lipschitz if g is ⌊γ⌋ times continuously differentiable and such that there exists a constant $M \geq 0$ such that the supremum norm of its $\check { k } ^ { t h }$ derivative, $k = 0 , \ldots , \lfloor \gamma \rfloor$ , and the $( \gamma - \lfloor \gamma \rfloor )$ )-Holder norm of its ¨ $\lfloor \gamma \rfloor ^ { t h }$ derivative are bounded by M. The smallest M satisfying these conditions is the γ Lipschitz norm of g, denoted by $\| g \| _ { L i p } \gamma : = \| g \| _ { L i p } \gamma _ { ( V , W ) }$ . We denote by $L i p ^ { \gamma } ( V , W )$ the space of γ-Lipschitz functions from V to W .

One-forms and the rough integral (informal). In an RDE the map $f : W \to \operatorname { H o m } ( V , W )$ is naturally viewed as a one-form: at each state $y \in W , f ( y )$ is a linear map $V  W$ to be integrated against the (rough) increment of the driver. When $V = \mathbb { R } ^ { d }$ we often write $f = \left( f _ { 1 } , \ldots , f _ { d } \right)$ with ${ \bar { f _ { i } } } : W \to W$ , so that the coordinate form is

$$
d Y _ {t} = \sum_ {i = 1} ^ {d} f _ {i} (Y _ {t}) d X _ {t} ^ {i}.
$$

Since X is rough, the integral cannot be defined by Riemann–Stieltjes sums at level 1 only. Instead one uses all available iterated integrals of X up to level $m = \lfloor p \rfloor$ .

Controlled paths and compensated Riemann sums. A path $Y : [ 0 , T ]  W$ is controlled by X if its increments admit an expansion $Y _ { u , v } = Y _ { u } ^ { \prime } \mathbb { X } _ { u , v } ^ { 1 } + \bar { R } _ { u , v }$ with higher-order consistency and suitable p-variation bounds on the remainder $R _ { u , v } .$ . For such Y and $f \in \mathrm { L i p } ( \gamma )$ with $\gamma > p ,$ the rough integral $\textstyle \int _ { 0 } ^ { t } f ( Y _ { u } ) d \mathbb { X } _ { u }$ is defined as the limit, as $| \mathcal { D } |  0 ,$ of the compensated sums

$$
\sum_ {[ u, v ] \in \mathcal {D}} \Big (f (Y _ {u}) \mathbb {X} _ {u, v} ^ {1} + D f (Y _ {u}) [ f (Y _ {u}) ] \mathbb {X} _ {u, v} ^ {2} + \dots + D ^ {m - 1} f (Y _ {u}) [ \underbrace {f (Y _ {u}) , \ldots , f (Y _ {u})} _ {m - 1 \text {times}} ] \mathbb {X} _ {u, v} ^ {m} \Big),
$$

where $\mathbb { X } _ { u , v } ^ { k } \in V ^ { \otimes k }$ are the level-k increments of the geometric rough path. This construction agrees with the classical Riemann–Stieltjes (or Young) integral when the driver is smooth (or has $p < 2 )$ , and it is the pathwise notion used in Eq. 19.

Theorem A.5 (Universal Limit Theorem (Lyons, 1998)). Let $p \geq 1$ and let $X ^ { n } : [ 0 , T ] \to V$ be a sequence of continuous paths of bounded variation which converges in p-variation to a geometric p-rough path $\mathbb { X } : \Delta _ { T } \overset { \cdot } {  } T ^ { \lfloor p \rfloor } \overset { \cdot } { (} V )$ . Let $f : V \to \operatorname { H o m } ( V , W )$ be a $\operatorname { L i p } ( \gamma )$ function with $\gamma > p .$ Consider the controlled differential equations

$$
d Y _ {t} ^ {n} = f (Y _ {t} ^ {n}) d X _ {t} ^ {n}, \quad Y _ {0} ^ {n} = y _ {0} \in W \tag {20}
$$

Then, there exists a unique geometric rough path $\mathbb { Z } = ( \mathbb { X } , \mathbb { Y } ) : \Delta _ { T }  T ^ { \lfloor p \rfloor } ( V \oplus W )$ such that $Y ^ { n }$ converges to Y in p-variation. Moreover, the Ito map ˆ $I _ { f } : ( y _ { 0 } , \mathbb { X } ) \to \mathbb { Y }$ is continuous in p-variation.

Definition A.6 (RDE solution). Let $\mathbb { X } \in G \Omega _ { p } ( V )$ be a geometric p-rough path. We say that the continuous path ${ \bf \dot { Y } } : [ 0 , T ]  { \bf \dot { W } }$ of finite p-variation is a solution to the RDE

$$
d Y _ {t} = f \left(Y _ {t}\right) d \mathbb {X} _ {t}, \quad Y _ {0} = y _ {0} \in W
$$

if Y belongs to the set of (uniform) limit points constructed in Theorem A.5. In particular, $i f f$ : $\operatorname { \bar { \cal W } } \to \operatorname { H o m } ( V , W )$ is linear or γ-Lipschitz with $\gamma > p ,$ then Y is unique.

The notion of RDE solution presented in Definition A.6 maps a geometric p-rough path to a W - valued continuous path of finite p-variation. However, it might be desirable to construct $\mathrm { ~ a ~ } ^ { 6 } \mathrm { f u l l } ^ { \dag }$ solution also as a geometric rough path. This is the case, for example, if one is interested in using a solution to a first RDE to be the driving signal for a second RDE. More precisely, we will say that $\mathbb { Y } \in G \Omega ( W )$ is the (full) solution to the RDE

$$
d \mathbb {Y} _ {t} = f \left(\mathbb {Y} _ {t}\right) d \mathbb {X} _ {t}, \quad \text {   started   at   } \quad \mathbb {Y} _ {0} \in \operatorname{Sig} ^ {\lfloor p \rfloor} \left(\Omega_ {1} (W)\right)
$$

if there exists a sequence $( X ^ { n } )$ of continuous bounded variation paths such that the sequence of truncated signatures $( \operatorname { S i g } ^ { \left\lfloor p \right\rfloor } ( X ^ { n } ) )$ converges in p-variation to X and such that the sequence $( \mathbb { Y } _ { 0 }$ · $\operatorname { S i g } ^ { \lfloor p \rfloor } ( Y ^ { n } ) )$ ) converges uniformly on [0, T ] to Y as $n  \infty$ , where $\{ Y ^ { n } \}$ are the solutions to the CDEs 20, with $Y _ { 0 } = \pi _ { 1 } ( \mathbb { Y } _ { 0 } )$ .

# B PROOFS

Before proving the main theorems, we record two auxiliary results used in our proofs. First, we show that the inner product of time-derivatives of Random Fourier Feature lifts converges almost surely to the corresponding RKHS cross term; we include a self-contained proof. Second, we invoke a trace–moment identity for products of random matrices from Cass & Turner (2024) (proof therein).

Lemma 4. Let $x , y \in C ^ { 1 } ( [ 0 , T ] , \mathbb { R } ^ { d } )$ and $\phi ^ { F } : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 F }$ be the random Fourier feature map

$$
\phi_ {\mu} ^ {F} (z) := \frac {1}{\sqrt {F}} \Big (\cos (\omega_ {1} ^ {\top} z), \sin (\omega_ {1} ^ {\top} z), \dots , \cos (\omega_ {F} ^ {\top} z), \sin (\omega_ {F} ^ {\top} z) \Big) \in \mathbb {R} ^ {2 F},
$$

where $\{ \omega _ { j } \} _ { j = 1 } ^ { F } \stackrel { i . i . d . } { \sim } \mu$ and µ is the standard Gaussian measure on $\mathbb { R } ^ { d } .$ . Define the lifted curves

$$
X _ {t} ^ {F} := \phi_ {\mu} ^ {F} (x _ {t}), \qquad a n d \qquad Y _ {t} ^ {F} := \phi_ {\mu} ^ {F} (y _ {t}).
$$

Let $\mathcal { H } : = L ^ { 2 } ( \mu ; \mathbb { R } ^ { 2 } )$ and define the (infinite-dimensional) feature map

$$
\phi (z) := \left(\cos (\omega^ {\top} z), \sin (\omega^ {\top} z)\right) \in \mathbb {R} ^ {2}, \qquad \langle u, v \rangle_ {\mathcal {H}} := \int_ {\mathbb {R} ^ {d}} u (\omega) ^ {\top} v (\omega) d \mu (\omega).
$$

Let $X _ { t } : = \phi ( x _ { t } ) \in \mathcal { H } ,$ and $Y _ { t } : = \phi ( y _ { t } ) \in \mathcal { H } ;$ then, for every s, $t \in [ 0 , T ]$ , we have

1. almost sure convergence

$$
\left\langle \dot {X} _ {s} ^ {F}, \dot {Y} _ {t} ^ {F} \right\rangle_ {\mathbb {R} ^ {2 F}} \xrightarrow [ F \to \infty ]{a . s .} \left\langle \dot {X} _ {s}, \dot {Y} _ {t} \right\rangle_ {\mathcal {H}}. \tag {21}
$$

2. convergence in L1:

$$
\int_ {0} ^ {T} \int_ {0} ^ {T} \left|\left\langle \dot {X} _ {s} ^ {F}, \dot {Y} _ {t} ^ {F} \right\rangle_ {\mathbb {R} ^ {2 F}} - \left\langle \dot {X} _ {s}, \dot {Y} _ {t} \right\rangle_ {\mathcal {H}} \right| d t d s \xrightarrow {F \rightarrow \infty} 0. \tag {22}
$$

where $\dot { X } _ { s } ^ { F }$ is the derivative w.r.t. time of $X _ { s } ^ { F }$ , and similarly for $\dot { Y } _ { t } ^ { F } , \dot { X } _ { s }$ , and $\dot { Y } _ { t } .$ .

Proof. Differentiating component-wise,

$$
\frac {d}{d t} \cos (\omega_ {j} ^ {\top} x _ {t}) = - \sin (\omega_ {j} ^ {\top} x _ {t}) \omega_ {j} ^ {\top} \dot {x} _ {t}, \quad \frac {d}{d t} \sin (\omega_ {j} ^ {\top} x _ {t}) = \cos (\omega_ {j} ^ {\top} x _ {t}) \omega_ {j} ^ {\top} \dot {x} _ {t},
$$

and similarly with xt, x˙ t replaced by $y _ { t } , \dot { y } _ { t }$ . Hence

$$
\dot {X} _ {s} ^ {F} = \frac {1}{\sqrt {F}} \Big (- \sin (\omega_ {j} ^ {\top} x _ {s}) \omega_ {j} ^ {\top} \dot {x} _ {s}, \cos (\omega_ {j} ^ {\top} x _ {s}) \omega_ {j} ^ {\top} \dot {x} _ {s} \Big) _ {j = 1} ^ {F},
$$

and analogously for $\dot { Y } _ { t } ^ { F }$ . The Euclidean inner product becomes

$$
\begin{array}{l} \left\langle \dot {X} _ {s} ^ {F}, \dot {Y} _ {t} ^ {F} \right\rangle_ {\mathbb {R} ^ {2 F}} = \frac {1}{F} \sum_ {j = 1} ^ {F} (\omega_ {j} ^ {\top} \dot {x} _ {s}) (\omega_ {j} ^ {\top} \dot {y} _ {t}) \Big (\sin (\omega_ {j} ^ {\top} x _ {s}) \sin (\omega_ {j} ^ {\top} y _ {t}) + \cos (\omega_ {j} ^ {\top} x _ {s}) \cos (\omega_ {j} ^ {\top} y _ {t}) \Big) \\ = \frac {1}{F} \sum_ {j = 1} ^ {F} (\omega_ {j} ^ {\top} \dot {x} _ {s}) (\omega_ {j} ^ {\top} \dot {y} _ {t}) \cos \left(\omega_ {j} ^ {\top} (x _ {s} - y _ {t})\right) =: \frac {1}{F} \sum_ {j = 1} ^ {F} g _ {\omega_ {j}} (s, t). \\ \end{array}
$$

Define

$$
g _ {\omega} (s, t) := (\omega^ {\top} \dot {x} _ {s}) (\omega^ {\top} \dot {y} _ {t}) \cos \big (\omega^ {\top} (x _ {s} - y _ {t}) \big).
$$

Then $\{ g _ { \omega _ { j } } ( s , t ) \} _ { j = 1 } ^ { F }$ 1 are i.i.d. with mean

$$
\mathbb {E} _ {\mu} [ g _ {\omega} (s, t) ] = \int_ {\mathbb {R} ^ {d}} (\omega^ {\top} \dot {x} _ {s}) (\omega^ {\top} \dot {y} _ {t}) \cos \left(\omega^ {\top} (x _ {s} - y _ {t})\right) d \mu (\omega).
$$

On the other hand, in the RKHS model $\mathcal { H } = L ^ { 2 } ( \mu ; \mathbb { R } ^ { 2 } )$ ,

$$
\dot {X} _ {s} (\omega) = \big (- \sin (\omega^ {\top} x _ {s}) \omega^ {\top} \dot {x} _ {s}, \cos (\omega^ {\top} x _ {s}) \omega^ {\top} \dot {x} _ {s} \big),
$$

and similarly for $\dot { Y } _ { t } ( \omega )$ . Therefore

$$
\begin{array}{l} \left\langle \dot {X} _ {s}, \dot {Y} _ {t} \right\rangle_ {\mathcal {H}} = \int_ {\mathbb {R} ^ {d}} (\omega^ {\top} \dot {x} _ {s}) (\omega^ {\top} \dot {y} _ {t}) \Big (\sin (\omega^ {\top} x _ {s}) \sin (\omega^ {\top} y _ {t}) + \cos (\omega^ {\top} x _ {s}) \cos (\omega^ {\top} y _ {t}) \Big) d \mu (\omega) \\ = \int_ {\mathbb {R} ^ {d}} (\omega^ {\top} \dot {x} _ {s}) (\omega^ {\top} \dot {y} _ {t}) \cos (\omega^ {\top} (x _ {s} - y _ {t})) d \mu (\omega) = \mathbb {E} _ {\mu} [ g _ {\omega} (s, t) ]. \\ \end{array}
$$

Thus $\mathbb { E } _ { \mu } [ g _ { \omega } ( s , t ) ] = \langle \dot { X } _ { s } , \dot { Y } _ { t } \rangle _ { \mathcal { H } }$ . By Cauchy–Schwarz and $| \cos | \le 1$

$$
| g _ {\omega} (s, t) | \leq \| \omega \| ^ {2} \| \dot {x} _ {s} \| \| \dot {y} _ {t} \|.
$$

Since

i. the sequence $\{ g _ { \omega _ { j } } ( s , t ) \} _ { j \geq 1 }$ is i.i.d.,

ii. $x , y \in C ^ { 1 } ( [ 0 , T ] , \mathbb { R } ^ { d } )$ implies that $\| \dot { x } _ { s } \| , \| \dot { y } _ { t } \|$ are bounded on [0, T ],

iii. $\begin{array} { r l } & { \int \left\| \omega \right\| ^ { 2 } d \mu ( \omega ) < \infty } \\ & { \mathrm { g r a b l e } , } \end{array}$ as the Gaussian distribution has finite variance, so $g _ { \omega } ( s , t )$ is inte-

we can apply the strong law of large numbers, giving that

$$
\frac {1}{F} \sum_ {j = 1} ^ {F} g _ {\omega_ {j}} (s, t) \xrightarrow [ F \to \infty ]{\mathrm{a.s.}} \mathbb {E} _ {\mu} [ g _ {\omega} (s, t) ] = \left<   \dot {X} _ {s}, \dot {Y} _ {t} \right> _ {\mathcal {H}},
$$

which proves Eq. 21.

For the $L ^ { 1 }$ convergence,

$$
\left| \left\langle \dot {X} _ {s} ^ {F}, \dot {Y} _ {t} ^ {F} \right\rangle_ {\mathbb {R} ^ {2 F}} \right| \leq \left(\frac {1}{F} \sum_ {j = 1} ^ {F} \| \omega_ {j} \| ^ {2}\right) \| \dot {x} _ {u} \| \| \dot {y} _ {v} \| \xrightarrow [ F \to \infty ]{\mathrm{a.s.}} \mathbb {E} _ {\mu} \| \omega \| ^ {2} \| \dot {x} _ {u} \| \| \dot {y} _ {v} \|.
$$

Since $\mathbb { E } _ { \mu } \| \omega \| ^ { 2 } < \infty$ for Gaussian $\mu ,$ there exists $C ( \omega ) < \infty$ such that, for all $F _ { ; }$

$$
| \langle \dot {X} _ {s} ^ {F}, \dot {Y} _ {t} ^ {F} \rangle_ {\mathbb {R} ^ {2 F}} | \leq C (\omega) \| \dot {x} _ {s} \| \| \dot {y} _ {t} \|, \quad \text { and } \quad | \langle \dot {X} _ {s}, \dot {Y} _ {t} \rangle_ {\mathcal {H}} | \leq C (\omega) \| \dot {x} _ {s} \| \| \dot {y} _ {t} \|,
$$

Hence,

$$
\left| \left\langle \dot {X} _ {s} ^ {F}, \dot {Y} _ {t} ^ {F} \right\rangle_ {\mathbb {R} ^ {2 F}} - \left\langle \dot {X} _ {s}, \dot {Y} _ {t} \right\rangle_ {\mathcal {H}} \right| \leq 2 C (\omega) \| \dot {x} _ {s} \| \| \dot {y} _ {t} \| \tag {23}
$$

Since x,˙ $\dot { y } \in L ^ { 1 } ( [ 0 , T ] )$ , the envelope on the right is integrable over $[ 0 , T ] ^ { 2 }$ . Combining the almostsure pointwise convergence with the bound in Eq. 23, and using the continuity (hence measurability) of the maps

$$
(s, t) \mapsto \left\langle \dot {X} _ {s} ^ {F}, \dot {Y} _ {t} ^ {F} \right\rangle_ {\mathbb {R} ^ {2 F}} \quad \text { and } \quad (s, t) \mapsto \left\langle \dot {X} _ {s}, \dot {Y} _ {t} \right\rangle_ {\mathcal {H}}
$$

the hypotheses of the dominated convergence theorem are satisfied. Therefore,

$$
\int_ {0} ^ {T} \int_ {0} ^ {T} \left| \langle \dot {X} _ {s} ^ {F}, \dot {Y} _ {t} ^ {F} \rangle_ {\mathbb {R} ^ {2 F}} - \langle \dot {X} _ {s}, \dot {Y} _ {t} \rangle_ {\mathcal {H}} \right| d t d s \xrightarrow {F \to \infty} 0,
$$

which proves Eq. 22.

□

Lemma 5 (Cass & Turner (2024)). Suppose that for each $N \in \mathbb { N } , \{ A _ { i } ^ { N } : 1 \leq i \leq d \}$ is a collection of d Gaussian matrices. Let $n , m \geq 0$ be non-negative integers and consider the words ${ \mathbf I } = i _ { 1 } \ldots i _ { n } \in \mathcal { W } _ { d } ^ { n }$ , and $\mathbf { J } = j _ { 1 } \ldots j _ { m } \in \mathcal { W } _ { d } ^ { m }$ (where $\mathcal { W } _ { d } ^ { \cdot }$ is defined in Section 3.3), with corresponding matrix products

$$
A _ {\mathbf {I}} ^ {N} := A _ {i _ {1}} ^ {N} \dots A _ {i _ {n}} ^ {N}
$$

$$
A _ {\mathbf {I} \star \mathbf {J}} ^ {N} := \left(A _ {\mathbf {I}} ^ {N}\right) ^ {T} A _ {\mathbf {J}} ^ {N}
$$

Then, setting $k = n + m$ ,

$$
\lim _ {N \to \infty} \frac {1}{N ^ {\frac {k}{2} + 1}} \mathbb {E} \left[ \operatorname{tr} \left(A _ {\mathbf {I} \star \mathbf {J}} ^ {N}\right) \right] = \left\{ \begin{array}{l l} 1, & i f \mathbf {I} = \mathbf {J} \\ 0, & o t h e r w i s e \end{array} \right.
$$

We use the convention that $i f n = 0 ,$ , then $A _ { \mathbf { I } } ^ { N } = \operatorname { I d } _ { N }$ , the $N \times N$ identity matrix.

# B.1 PROOF OF THEOREM 3.2

For convenience we restate Theorem 3.2 and provide its proof.

Theorem. Let $x _ { t } , y _ { t }$ be differentiable paths on [0, T ] and $Z _ { s } ^ { N , F } ( x ) , Z _ { t } ^ { N , F } ( y )$ solve Eq. 9 with $\varphi =$ id and the same $A _ { i } \stackrel { i . i . d . } { \sim } \xi _ { N }$ (independent of the RFF draw). Then, for every $s , t \in [ 0 , T ]$

$$
\lim _ {F \to \infty} \lim _ {N \to \infty} \frac {1}{N} \mathbb {E} _ {\xi_ {N}} \left[ \left\langle Z _ {s} ^ {N, F} (x), Z _ {t} ^ {N, F} (y) \right\rangle_ {\mathbb {R} ^ {N}} \right] = K _ {S i g - R B F} ^ {x, y} (s, t),
$$

where $K _ { S i g - R B F } ^ { x , y } ( s , t )$ denotes the RBF–lifted signature kernel $( E q . \ 6 )$

Proof. Recall from Section 3.2 that we denote

$$
X _ {t} ^ {F} := \phi_ {\mu} ^ {F} (x _ {t}) \in \mathbb {R} ^ {2 F}, \quad \text { and } \quad Y _ {t} ^ {F} := \phi_ {\mu} ^ {F} (y _ {t}) \in \mathbb {R} ^ {2 F}.
$$

where $\phi _ { \mu } ^ { F }$ is the random Fourier map defined in Eq. 4.

For fixed F , the R-CDE limit (Theorem 3.1) applied to the drivers $X ^ { F }$ and $Y ^ { F }$ gives

$$
\lim _ {N \rightarrow \infty} \frac {1}{N} \mathbb {E} _ {\xi_ {N}} \left[\left\langle Z _ {s} ^ {N, F} (x), Z _ {t} ^ {N, F} (y) \right\rangle_ {\mathbb {R} ^ {F}} \right] = k _ {F} (s, t) \quad \text { where } \quad k _ {F} := K _ {\text { sig }} ^ {X ^ {F}, Y ^ {F}}, \tag {24}
$$

Let $k : = K _ { \mathrm { S i g - R B F } } ^ { x , y }$ . By the PDE/Volterra characterization of the signature kernel (Theorem A.3), $k _ { F } , k : [ 0 , T ] ^ { 2 } \to$ R are the unique solutions of the two-parameter Volterra equations

$$
k _ {F} (s, t) = 1 + \int_ {0} ^ {s} \int_ {0} ^ {t} q _ {F} (u, v) k _ {F} (u, v) d v d u,
$$

and

$$
k (s, t) = 1 + \int_ {0} ^ {s} \int_ {0} ^ {t} q (u, v) k (u, v) d v d u,
$$

respectively, with driving kernels $q _ { F } ( u , v ) : = \langle \dot { X } _ { u } ^ { F } , \dot { Y } _ { v } ^ { F } \rangle _ { \mathbb { R } ^ { 2 F } }$ and $q ( u , v ) : = \langle \dot { X } _ { u } , \dot { Y } _ { v } \rangle _ { \mathcal { H } }$ , where H is the RKHS of the RBF feature map.

By Lemma 4, $q _ { F }  q \sin L ^ { 1 } ( [ 0 , T ] ^ { 2 } )$ almost surely (over the RFF draw), and there exists an envelope

$$
| \langle \dot {X} _ {s} ^ {F}, \dot {Y} _ {t} ^ {F} \rangle_ {\mathbb {R} ^ {2 F}} | \leq C (\omega)   \| \dot {x} _ {s} \|   \| \dot {y} _ {t} \|, \quad \text { and } \quad | \langle \dot {X} _ {s}, \dot {Y} _ {t} \rangle_ {\mathcal {H}} | \leq C (\omega)   \| \dot {x} _ {s} \|   \| \dot {y} _ {t} \|,
$$

with $\lVert \dot { x } _ { u } \rVert , \lVert \dot { y } _ { v } \rVert \in L ^ { 1 } ( 0 , T )$ and $C ( \omega ) ~ < \infty$ . The standard Volterra stability estimate via the two-parameter Gronwall inequality (Defranco, 1976), therefore yields ¨

$$
\left\| k _ {F} - k \right\| _ {\infty} \leq \exp \left(C \| \dot {x} _ {s} \| _ {L ^ {1}} \| \dot {y} _ {t} \| _ {L ^ {1}}\right) \| q _ {F} - q \| _ {L ^ {1}}.
$$

Consequently, $k _ { F } ( s , t )  k ( s , t ) = K _ { \mathrm { S i g - R B F } } ^ { x , y } ( s , t )$ uniformly on $[ 0 , T ] ^ { 2 }$ for µ-almost every RFF draw. Taking $F  \infty$ in Eq. 24 and using this uniform convergence (together with the uniform bound implied by the envelope in the stability estimate) proves the theorem by dominated convergence.

![](images/758041841135905503e5e2744f36589e86e9c7419d54ba8c6a179d6c58fce780.jpg)

# B.2 PROOFS AND LEMMAS FOR SECTION 3.3

We first state and prove a lemma ensuring that $\Gamma _ { A }$ is well defined and convergent on group-like elements (signature increments). We then prove Lemma 1 from the main text. Finally, we include a short derivation of the smooth–driver setting mentioned in Remark 3.2.

Lemma 6 (Absolute convergence of $\Gamma _ { A }$ on signature increments). Let $p \ge 1 , d \in \mathbb { N } ,$ and let $\mathbb { X } \in G \Omega _ { p } ( \mathbb { R } ^ { d } )$ with control ω. Fix matrices $A _ { 1 } , \ l . . . , A _ { d } \in \mathrm { E n d } ( \mathbb { R } ^ { N } )$ and set

$$
\kappa := \max _ {1 \leq i \leq d} \frac {1}{\sqrt {N}} \| A _ {i} \| <   \infty .
$$

For a word $w = i _ { 1 } \cdots i _ { k }$ define $A _ { w } : = N ^ { - \frac { k } { 2 } } A _ { i _ { 1 } } \cdot \cdot \cdot A _ { i _ { k } }$ and $\Gamma _ { A }$ by Eq. 11. Then for every $( s , t ) \in$ $\Delta _ { T }$ the series

$$
\sum_ {w \in \mathcal {W} _ {d}} A _ {w} \left\langle \mathbb {X} _ {s, t}, w \right\rangle
$$

converges absolutely in End $. ( \mathbb { R } ^ { N } )$ . Consequently $\Gamma _ { A } ( \mathbb { X } _ { s , t } )$ is well-defined and $( s , t ) \mapsto \Gamma _ { A } ( \mathbb { X } _ { s , t } )$ is continuous.

By sub-multiplicativity and the definition of $\kappa ,$

$$
\left\| A _ {w} \right\| = N ^ {- \frac {k}{2}} \left\| A _ {i _ {1}} \dots A _ {i _ {k}} \right\| \leq N ^ {- \frac {k}{2}} \prod_ {j = 1} ^ {k} \left\| A _ {i _ {j}} \right\| \leq \kappa^ {k} \quad \text { for } | w | = k.
$$

Group the series by word length and use the triangle inequality:

$$
\sum_ {w \in \mathcal {W} _ {d}} \| A _ {w} \| | \langle \mathbb {X} _ {s, t}, w \rangle | \leq \sum_ {k = 0} ^ {\infty} \kappa^ {k} \sum_ {| w | = k} | \langle \mathbb {X} _ {s, t}, w \rangle |.
$$

By Cauchy–Schwarz,

$$
\sum_ {| w | = k} | \langle \mathbb {X} _ {s, t}, w \rangle | \leq d ^ {\frac {k}{2}} \| \pi_ {k} (\mathbb {X} _ {s, t}) \|,
$$

where $\| \cdot \|$ is the Hilbert–Schmidt norm. Geometric p-rough paths satisfy the factorial decay (see Definition 2.1):

$$
\| \pi_ {k} (\mathbb {X} _ {s, t}) \| \leq \frac {C _ {p} \omega (s , t) ^ {\frac {k}{p}}}{\Gamma (\frac {k}{p} + 1)}
$$

for some $C _ { p } > 0$ that depends only on $p .$ Hence

$$
\sum_ {w \in \mathcal {W} _ {d}} \| A _ {w} \| | \langle \mathbb {X} _ {s, t}, w \rangle | \leq C _ {p} \sum_ {k = 0} ^ {\infty} \left(\kappa d ^ {1 / 2}\right) ^ {k} \frac {\omega (s , t) ^ {\frac {k}{p}}}{\Gamma (\frac {k}{p} + 1)}.
$$

By Stirling’s formula, $\Gamma ( k / p + 1 ) \sim ( k / p ) ^ { k / p } e ^ { - k / p } \sqrt { 2 \pi k / p } ,$ , which outgrows any exponential; thus the right-hand series converges for all $\omega ( s , t ) < \infty .$ . Absolute convergence implies the claim.

![](images/6fecc723d6df6d1d471dc6b00d20b5f6a20c895623c348dfb4263fb3e790a67b.jpg)

# B.2.1 PROOF OF LEMMA 1

For convenience we restate Theorem 1 and provide its proof.

Lemma. $L e t \Gamma _ { A }$ be as in Eq. 11. $I f x : [ 0 , T ] \to \mathbb { R } ^ { d }$ has bounded variation and $\begin{array} { r } { x _ { t } ^ { A } : = \sum _ { i = 1 } ^ { d } A _ { i } x _ { t } ^ { i } , } \end{array}$ with $A _ { i } \in \operatorname { E n d } ( \mathbb { R } ^ { N } )$ , then $S _ { t } ^ { A } ( x )$ in $E q .$ . 12 is the unique solution of the linear matrix CDE

$$
d S _ {t} ^ {A} (x) = S _ {t} ^ {A} (x) \circ d x _ {t} ^ {A}, \qquad S _ {0} ^ {A} (x) = \mathrm{Id} _ {N} \in \mathrm{End} (\mathbb {R} ^ {N}).
$$

Proof. For x of bounded variation, the signature solves Chen’s integral equation

$$
\operatorname{Sig} (x) _ {s, t} = \mathbb {1} + \int_ {(s, t ]} \operatorname{Sig} (x) _ {s, u} \otimes d x _ {u},
$$

with $d x _ { u } = ( d x _ { u } ^ { 1 } , \dots , d x _ { u } ^ { d } )$ ). Applying the algebra homomorphism $\Gamma _ { A }$ (which sends $1 \mapsto I _ { N }$ , concatenation ⊗ to composition, and $e _ { i } \mapsto A _ { i } )$ yields

$$
S _ {s, t} ^ {A} (x) := \Gamma_ {A} \big (\operatorname{Sig} (x) _ {s, t} \big) = \operatorname{Id} _ {N} + \int_ {(s, t ]} S _ {s, u} ^ {A} (x) \circ d x _ {u} ^ {A}, \qquad d x _ {u} ^ {A} := \sum_ {i = 1} ^ {d} A _ {i} d x _ {u} ^ {i},
$$

i.e. $d S _ { t } ^ { A } ( x ) = S _ { t } ^ { A } ( x ) \circ d x _ { t } ^ { A }$ with $S _ { 0 } ^ { A } ( x ) = \mathrm { I d } _ { N } \in \mathrm { E n d } ( \mathbb { R } ^ { N } )$ . Uniqueness follows from standard Picard iteration for linear matrix CDEs driven by bounded-variation paths.

![](images/0806fb821ceb700b65fcb9324f1a6e70fe770a758e7d8be195678f626ecaeaba.jpg)

# B.2.2 REMARK 3.2 (SMOOTH-DRIVER DERIVATION)

Assume x has bounded variation so that $\mathbb { X } = \operatorname { S i g } ( x )$ . Starting from

$$
Z _ {t} = Z _ {0} + \int_ {0} ^ {t} d S _ {u} ^ {A} \big (\varphi (Z _ {u}) \big),
$$

expand $d S _ { u } ^ { A }$ using $\begin{array} { r } { S _ { u } ^ { A } = \sum _ { w \in { \mathcal W } _ { d } } A _ { w } \langle \mathrm { S i g } ( x ) _ { 0 , u } , w \rangle } \end{array}$ :

$$
Z _ {t} = Z _ {0} + \int_ {0} ^ {t} \Big (\sum_ {w \in \mathcal {W} _ {d}} A _ {w} d \langle \mathrm{Sig} (x) _ {0, u}, w \rangle \Big) \big (\varphi (Z _ {u}) \big).
$$

Write each non-empty word as $w = \widehat { w } i$ i (last letter i) and use $d \langle \mathrm { S i g } ( x ) _ { 0 , u } , \widehat { w } i \rangle = \langle \mathrm { S i g } ( x ) _ { 0 , u } , \widehat { w } \rangle d x _ { u } ^ { i }$ and $d \langle \mathrm { S i g } ( x ) _ { 0 , u } , \emptyset \rangle \stackrel { - } { = } 0$ to get

$$
Z _ {t} = Z _ {0} + \sum_ {i = 1} ^ {d} \int_ {0} ^ {t} \Big (\sum_ {\widehat {w} \in \mathcal {W} _ {d}} A _ {\widehat {w} i} \left\langle \mathrm{Sig} (x) _ {0, u}, \widehat {w} \right\rangle \Big) \big (\varphi (Z _ {u}) \big) d x _ {u} ^ {i}.
$$

By multiplicativity of $\Gamma _ { A } ( \mathrm { s o } A _ { \widehat { w } i } = A _ { \widehat { w } } A _ { i } )$ ,

$$
\sum_ {\widehat {w}} A _ {\widehat {w} i} \left\langle \operatorname{Sig} (x) _ {0, u}, \widehat {w} \right\rangle = \Big (\sum_ {\widehat {w}} A _ {\widehat {w}} \left\langle \operatorname{Sig} (x) _ {0, u}, \widehat {w} \right\rangle \Big) A _ {i} = S _ {u} ^ {A} A _ {i},
$$

and hence

$$
Z _ {t} = Z _ {0} + \sum_ {i = 1} ^ {d} \int_ {0} ^ {t} \left(S _ {u} ^ {A} A _ {i} \varphi (Z _ {u})\right) d x _ {u} ^ {i}.
$$

# B.3 PROOF OF THEOREM 3.4

For convenience we restate Theorem 3.4 and provide its proof.

Theorem. Let $\mathbb { X } \in G \Omega _ { p } ( \mathbb { R } ^ { d } )$ and $\mathbb { Y } \in G \Omega _ { q } ( \mathbb { R } ^ { d } )$ be geometric rough paths. Let $Z _ { s } ^ { N } ( \mathbb { X } )$ and $Z _ { t } ^ { N } ( \mathbb { Y } )$ be the solutions of Eq. 15 with $\varphi = \mathrm { i d }$ and the same matrices $\{ A _ { i } \} _ { i = 1 } ^ { d }$ (with $A _ { i } \sim \xi _ { N } i . i . d . )$ . Then for all $s , t \in [ 0 , T ]$ ,

$$
\lim _ {N \to \infty} \frac {1}{N} \mathbb {E} _ {\xi_ {N}} \left[ \left<   Z _ {s} ^ {N} (\mathbb {X}), Z _ {t} ^ {N} (\mathbb {Y}) \right> _ {\mathbb {R} ^ {N}} \right] = K _ {\mathrm{Sig}} ^ {\mathbb {X}, \mathbb {Y}} (s, t),
$$

where where $K _ { \mathrm { S i g } } ^ { \mathbb { X } , \mathbb { Y } }$ K denotes the rough signature kernel defined in $E q . \ 7 .$

Proof. As in the smooth/CDE case, the (matrix) Dyson/Chen expansion under $\varphi = \mathrm { i d }$ gives

$$
\lim _ {N \to \infty} \frac {1}{N} \mathbb {E} _ {\xi_ {N}} \left[ \left\langle Z _ {s} ^ {N} (\mathbb {X}), Z _ {t} ^ {N} (\mathbb {Y}) \right\rangle_ {\mathcal {H}} \right] = \lim _ {N \to \infty} \mathbb {E} _ {\xi_ {N}} \left[ \sum_ {\mathbf {I}, \mathbf {J} \in \mathcal {W} _ {d}} ^ {\infty} \frac {1}{N ^ {\frac {| \mathbf {I} ^ {*} \cdot \mathbf {J} |}{2} + 1}} \operatorname{tr} \left(A _ {\mathbf {I} ^ {*} \mathbf {J}} ^ {N}\right) \operatorname{Sig} ^ {\mathbf {I}} (\mathbb {X}) _ {s} \operatorname{Sig} ^ {\mathbf {J}} (\mathbb {Y}) _ {t} \right]
$$

where $\mathcal { W } _ { d }$ is introduced in Section 3.3 and $\mathrm { S i g } ^ { \mathbf { I } } ( \cdot )$ denotes the coordinate of the signature associated to the word $\mathbf { I } \left( \mathrm { i . e . ~ S i g } ^ { \mathbf { I } } ( \cdot ) : = \langle \mathrm { S i g } ( \cdot ) , \mathbf { I } \rangle \right)$ .

From here, the argument mirrors Cass & Turner (2024), which also underpins the alternative proof of Theorem 3.1. In order to apply Lemma 5, we would like to exchange the limit and expectation and the double sum. By Fubini-Tonelli and the dominated convergence theorem, to justify the exchange it is enough to show that

$$
\sum_ {| \mathbf {I} |, | \mathbf {J} | = 0} ^ {\infty} \frac {1}{N ^ {\frac {| \mathbf {I} \star \mathbf {J} |}{2} + 1}} \mathbb {E} _ {\xi_ {N}} \left[ \left| \operatorname{tr} \left(A _ {\mathbf {I} \star \mathbf {J}} ^ {N}\right) \right| \right] \left| \operatorname{Sig} ^ {\mathbf {I}} (\mathbb {X}) _ {s} \operatorname{Sig} ^ {\mathbf {J}} (\mathbb {Y}) _ {t} \right| \tag {25}
$$

is uniformly bounded in N . As the matrices $A _ { i }$ are Gaussian it holds that

$$
\frac {1}{N ^ {\frac {| \mathbf {I} ^ {\star} \mathbf {J} |}{2} + 1}} \mathbb {E} _ {\xi_ {N}} \left[ \left| \operatorname{tr} \left(A _ {\mathbf {I} ^ {\star} \mathbf {J}} ^ {N}\right) \right| \right] \leq \frac {1}{N ^ {\frac {| \mathbf {I} ^ {\star} \mathbf {J} |}{2} + 1}} \mathbb {E} _ {\xi_ {N}} \left[ \left\| A _ {\mathbf {I} ^ {\star} \mathbf {J}} ^ {N} \right\| _ {\mathrm{op}} \right] \leq \kappa^ {| \mathbf {I} ^ {\star} \mathbf {J} |} \Gamma \left(\frac {| \mathbf {I} ^ {\star} \mathbf {J} |}{2} + 1\right)
$$

for a constant κ and where $\| \cdot \| _ { \mathrm { o p } }$ is the operator norm.

Then, by the factorial decay of (the signature of) rough paths (Definition 2.1), and the fact that the $L _ { 2 }$ norm is at least as large as the $L _ { 1 }$ norm on ${ \dot { V } } ^ { \otimes n }$ , there exists some $\omega > 0$ for which Eq. 25 is bounded by

$$
\sum_ {n, m = 0} ^ {\infty} \frac {\Gamma \left(\frac {n + m}{2} + 1\right) (\omega \kappa) ^ {m + n}}{\Gamma (\frac {n}{p} + 1) \Gamma (\frac {m}{q} + 1)} \leq \sum_ {n, m = 0} ^ {\infty} \frac {\sqrt {\Gamma (n + 1) \Gamma (m + 1)} (\omega \kappa) ^ {n + m}}{\Gamma (\frac {n}{p} + 1) \Gamma (\frac {m}{q} + 1)}
$$

where the inequality follows from logarithmic convexity of Γ. By Stirling’s formula,

$$
\Gamma (\alpha n + 1) \asymp \sqrt {2 \pi} (\alpha n) ^ {\alpha n + \frac {1}{2}} e ^ {- \alpha n} \quad \mathrm{as} n \to \infty ,
$$

the numerator grows subfactorially relative to the product $\Gamma ( n / p + 1 ) \Gamma ( m / q + 1 )$ , so the double series converges for any fixed $( \omega , \kappa )$ . Hence, by an exchange of limits and an application of Lemma 5, we see that

$$
\begin{array}{l} \lim _ {N \to \infty} \frac {1}{N} \mathbb {E} _ {\xi_ {N}} \left[ \left\langle Z _ {s} ^ {N} (\mathbb {X}), Z _ {t} ^ {N} (\mathbb {Y}) \right\rangle_ {\mathcal {H}} \right] = \sum_ {| \mathbf {I} |, | \mathbf {J} | = 0} ^ {\infty} \lim _ {N \to \infty} \frac {1}{N ^ {\frac {| \mathbf {I} ^ {*} \cdot \mathbf {J} |}{2} + 1}} \mathbb {E} _ {\xi_ {N}} \left[ \mathrm{tr} \left(A _ {\mathbf {I} \star \mathbf {J}} ^ {N}\right) \right] \mathrm{Sig} ^ {\mathbf {I}} (\mathbb {X}) _ {s} \mathrm{Sig} ^ {\mathbf {J}} (\mathbb {Y}) _ {t} \\ = \sum_ {| \mathbf {I} | = 0} ^ {\infty} \operatorname{Sig} ^ {\mathbf {I}} (\mathbb {X}) _ {s} \operatorname{Sig} ^ {\mathbf {J}} (\mathbb {Y}) _ {t} \\ = \langle \mathrm{Sig} (\mathbb {X}) _ {s} \mathrm{Sig} (\mathbb {Y}) _ {t} \rangle_ {T ((V))} \\ = K _ {\text { Sig }} ^ {\mathbb {X}, \mathbb {Y}} (s, t) \\ \end{array}
$$

which concludes our proof.

![](images/b5d7b924c4c34be6052cc25bc3c8848f025ba645febce80aa745461f5ef05707.jpg)

# C ADDITIONAL EXPERIMENTAL RESULTS AND DETAILS

# C.1 EXPERIMENTAL SETUP

Preprocessing. We use the archive’s pre-specified train/test splits. Each dataset is min–max scaled to [−1, 1], and sequences are represented via piecewise-linear interpolation. We apply time and basepoint augmentation, and tune the inclusion of a lead–lag transform via grid search. All sequences are resampled to length 200 following Toth et al. (2025).

Implementation. Random differential equation models are implemented in Jax in our library RandomSigJax. All the other benchmarks have been evaluated using the KSig library (Toth ´ et al., 2025) built on CuPy (Nishino & Loomis, 2017). Both libraries use CuML to perform SVM/LinearSVM calculations on GPU.

Neural Baseline Configuration. NCDE and NRDE are implemented with an MLP vector field and a linear readout. Their hidden dimensions are chosen so that the total parameter count approximately matches the feature budgets used for the random models, ensuring a fair comparison. Training follows the standard protocol: Adam optimiser, early stopping on validation accuracy.

Compute. All experiments were run on a single NVIDIA RTX 3090 GPU. Each approach is trained and evaluated 3 times on each dataset, then the median test accuracy is taken.

Hyperparameter selection. As documented in prior work on RFSF (Toth et al., 2025), truncated signature kernels (Kiraly & Oberhauser, 2019), and SigPDE (Salvi et al., 2021a), the influence of the ´ scaling parameters introduced in Eq. 10 and in Section 3.3 is strongly data- and task-dependent, and there is currently no principled procedure for selecting them a priori. Consistent with this literature, we treat them as hyperparameters tuned via grid search.

Some general heuristics nevertheless can guide the design of the search ranges: higher truncation orders m are often beneficial for rougher paths (in the sense of larger p-variation), while reasonable choices of $\sigma _ { A } , \sigma _ { B }$ typically ensure that typical increments $\sigma _ { A } \Delta X _ { t }$ remain of order at most one.

Hyperparameter grids. We tune all models via grid search; the swept grids are listed below.

For all models using RFFs, the Fourier frequency scale is swept over {0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100}× the bandwidth suggested by Toth et al. (2025).

Model–specific ranges are:

• RFSF: signature level $M \in \{ 2 , 3 , 4 , 5 \}$ .   
• All Random Differential Equation models:   
– activation ∈ {id, tanh, ReLU};   
– σA ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0};  
– σB ∈ {0.1, 0.25, 0.5};  
$- \ \sigma _ { 0 } \in \{ 0 . 0 , 0 . 5 , 1 . 0 , 1 . 5 \} .$

• R - RDE: signature level $M \in \{ 2 , 3 , 4 , 5 \}$ , capped so that the number of (log-)signature coordinates does not exceed the feature budget.

• RF - CDE: number of Fourier features $F \in \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 , 1 0 2 4 \}$ , capped at 50× the input dimension.

All feature-based models optionally apply feature normalisation (True/False).

Other baselines (SigPDE, RBF/GAK/RWS, SVMs) use standard grids over their key hyperparameters (kernel bandwidths, regularisation, warping parameters).

Remark C.1. SigPDE is evaluated only with an RBF base kernel (Salvi et al., 2021a), due to the superior empirical performance reported therein.

# C.2 ABLATION: NUMBER OF RANDOM FEATURES

We study sensitivity to feature count by repeating the main evaluation with N = 64 and N = 500 random features. Tables 4 and 5 below mirror the main setting (training three runs per dataset and reporting mean accuracy).

<table><tr><td>Dataset</td><td>R-CDE</td><td>RF-CDE</td><td>R-RDE</td><td>RFSF-DP</td><td>RFSF-TRP</td><td>NCDE</td><td>NRDE</td></tr><tr><td>ArticularyWordRecog.</td><td>0.917</td><td>0.963</td><td>0.903</td><td>0.957</td><td>0.957</td><td>0.957</td><td>0.917</td></tr><tr><td>AtrialFibrillation</td><td>0.400</td><td>0.467</td><td>0.533</td><td>0.267</td><td>0.333</td><td>0.467</td><td>0.267</td></tr><tr><td>BasicMotions</td><td>1.000</td><td>1.000</td><td>0.975</td><td>0.975</td><td>1.000</td><td>0.975</td><td>0.975</td></tr><tr><td>Cricket</td><td>0.931</td><td>0.944</td><td>0.917</td><td>0.917</td><td>0.944</td><td>0.917</td><td>0.898</td></tr><tr><td>EigenWorms</td><td>0.420</td><td>0.611</td><td>0.594</td><td>0.701</td><td>0.755</td><td>0.675</td><td>0.420</td></tr><tr><td>Epilepsy</td><td>0.963</td><td>0.963</td><td>0.927</td><td>0.949</td><td>0.927</td><td>0.963</td><td>0.927</td></tr><tr><td>EthanolConcentration</td><td>0.308</td><td>0.338</td><td>0.361</td><td>0.430</td><td>0.418</td><td>0.385</td><td>0.361</td></tr><tr><td>FingerMovements</td><td>0.510</td><td>0.520</td><td>0.500</td><td>0.490</td><td>0.510</td><td>0.500</td><td>0.400</td></tr><tr><td>Handwriting</td><td>0.279</td><td>0.340</td><td>0.302</td><td>0.302</td><td>0.305</td><td>0.305</td><td>0.279</td></tr><tr><td>Libras</td><td>0.827</td><td>0.894</td><td>0.856</td><td>0.817</td><td>0.883</td><td>0.827</td><td>0.894</td></tr><tr><td>NATOPS</td><td>0.900</td><td>0.900</td><td>0.889</td><td>0.789</td><td>0.889</td><td>0.833</td><td>0.789</td></tr><tr><td>RacketSports</td><td>0.750</td><td>0.796</td><td>0.691</td><td>0.747</td><td>0.782</td><td>0.796</td><td>0.747</td></tr><tr><td>SelfRegulationSCP1</td><td>0.825</td><td>0.853</td><td>0.857</td><td>0.880</td><td>0.884</td><td>0.846</td><td>0.857</td></tr><tr><td>SelfRegulationSCP2</td><td>0.522</td><td>0.533</td><td>0.567</td><td>0.550</td><td>0.494</td><td>0.517</td><td>0.550</td></tr><tr><td>StandWalkJump</td><td>0.267</td><td>0.469</td><td>0.400</td><td>0.400</td><td>0.333</td><td>0.333</td><td>0.333</td></tr><tr><td>UWaveGestureLibrary</td><td>0.834</td><td>0.822</td><td>0.897</td><td>0.825</td><td>0.803</td><td>0.789</td><td>0.822</td></tr><tr><td>Avg. acc. (↑)</td><td>0.666</td><td>0.713</td><td>0.698</td><td>0.687</td><td>0.701</td><td>0.692</td><td>0.652</td></tr><tr><td>Avg. rank (↓)</td><td>4.437</td><td>2.500</td><td>4.094</td><td>4.156</td><td>3.375</td><td>4.186</td><td>5.250</td></tr></table>

Table 4: UEA test accuracies with N = 64 random features. The neural baselines NCDE and NRDE use comparable parameter budgets. Best result per row in bold.

<table><tr><td></td><td>R-CDE</td><td>RF-CDE</td><td>R-RDE</td><td>RFSF-DP</td><td>RFSF-TRP</td><td>SigPDE</td></tr><tr><td>ArticularyWordRecognition</td><td>0.973</td><td>0.983</td><td>0.950</td><td>0.973</td><td>0.983</td><td>0.983</td></tr><tr><td>AtrialFibrillation</td><td>0.200</td><td>0.333</td><td>0.400</td><td>0.400</td><td>0.333</td><td>0.333</td></tr><tr><td>BasicMotions</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.975</td><td>1.000</td><td>1.000</td></tr><tr><td>Cricket</td><td>0.986</td><td>0.986</td><td>0.917</td><td>0.958</td><td>0.958</td><td>0.972</td></tr><tr><td>EigenWorms</td><td>0.458</td><td>0.664</td><td>0.612</td><td>0.824</td><td>0.817</td><td>0.794</td></tr><tr><td>Epilepsy</td><td>0.942</td><td>0.971</td><td>0.942</td><td>0.942</td><td>0.957</td><td>0.891</td></tr><tr><td>EthanolConcentration</td><td>0.319</td><td>0.407</td><td>0.375</td><td>0.517</td><td>0.414</td><td>0.460</td></tr><tr><td>FingerMovements</td><td>0.520</td><td>0.610</td><td>0.550</td><td>0.590</td><td>0.600</td><td>0.610</td></tr><tr><td>Handwriting</td><td>0.331</td><td>0.380</td><td>0.362</td><td>0.424</td><td>0.426</td><td>0.409</td></tr><tr><td>Libras</td><td>0.856</td><td>0.911</td><td>0.867</td><td>0.872</td><td>0.900</td><td>0.867</td></tr><tr><td>NATOPS</td><td>0.889</td><td>0.944</td><td>0.906</td><td>0.867</td><td>0.933</td><td>0.928</td></tr><tr><td>RacketSports</td><td>0.809</td><td>0.829</td><td>0.717</td><td>0.889</td><td>0.842</td><td>0.849</td></tr><tr><td>SelfRegulationSCP1</td><td>0.877</td><td>0.881</td><td>0.843</td><td>0.894</td><td>0.881</td><td>0.904</td></tr><tr><td>SelfRegulationSCP2</td><td>0.578</td><td>0.578</td><td>0.557</td><td>0.533</td><td>0.494</td><td>0.544</td></tr><tr><td>StandWalkJump</td><td>0.333</td><td>0.400</td><td>0.467</td><td>0.400</td><td>0.400</td><td>0.400</td></tr><tr><td>UWaveGestureLibrary</td><td>0.850</td><td>0.850</td><td>0.913</td><td>0.897</td><td>0.859</td><td>0.866</td></tr><tr><td>Avg. acc. (↑)</td><td>0.683</td><td>0.733</td><td>0.711</td><td>0.747</td><td>0.737</td><td>0.738</td></tr><tr><td>Avg. rank (↓)</td><td>4.812</td><td>2.812</td><td>4.125</td><td>3.187</td><td>3.031</td><td>3.031</td></tr></table>

Table 5: UEA test accuracies with N = 500 random features. SigPDE is a kernel method (no random features), so its results are unaffected by N . For each row, the best result is highlighted in bold.

Performance at N=64 Random Features Reducing the feature budget from 250 to 64 lowers accuracy across all models, but the relative ordering changes noticeably. RF–CDE achieves both the best average accuracy and the best average rank among all methods (0.713, 2.50), and R–RDE also remains competitive with only a modest drop (0.708 → 0.698). In contrast, the signature-projection baselines exhibit larger declines (RFSF-DP 0.726 → 0.687, RFSF-TRP 0.725 → 0.701). Easy datasets such as BasicMotions and Epilepsy stay saturated, while more complex datasets drive most of the differences across methods. Overall, the small-feature regime highlights that the random differential equation reservoirs retain good performance even when the feature budget is substantially constrained, whereas the neural baselines (NCDE, NRDE) struggle to remain competitive at this scale.

Performance at N=500 Random Features. The largest average gains appear for the randomizedsignature baselines (RFSF-DP 0.726 → 0.747, RFSF-TRP 0.725 → 0.737), while RF-CDE remains essentially flat (0.741 → 0.733) and R-RDE shows a slight lift (0.708 → 0.711), with clearer improvements on harder sets (e.g., EigenWorms for RF-CDE: 0.630 → 0.664, UWaveGestureLibrary for R-RDE: 0.903 → 0.913). Easy tasks (e.g., BasicMotions) are already saturated at near-perfect accuracy. In short, our random differential equation reservoirs already operate efficiently at 250 features; doubling N yields incremental benefits on a subset of challenging datasets.

# C.3 ADDITIONAL BASELINES FOR UEA CLASSIFICATION: CLASSICAL KERNELS AND RFF

For completeness, we report standard time-series baselines: Random Fourier Features (RFF), RBF kernel SVM, Global Alignment Kernel (GAK), and Random Warping Series (RWS). RFF is evaluated at two budgets (250 and 500 features), whereas RWS is reported only for the 250-feature setting. These baselines rarely achieve state-of-the-art performance on our suite but serve as useful reference points.

<table><tr><td></td><td>RFF-250</td><td>RFF-500</td><td>RWS</td><td>GAK</td><td>RBF</td></tr><tr><td>ArticularyWordRecognition</td><td>0.980</td><td>0.980</td><td>0.970</td><td>0.977</td><td>0.977</td></tr><tr><td>AtrialFibrillation</td><td>0.333</td><td>0.333</td><td>0.427</td><td>0.333</td><td>0.267</td></tr><tr><td>BasicMotions</td><td>0.925</td><td>0.925</td><td>0.995</td><td>1.000</td><td>0.975</td></tr><tr><td>Cricket</td><td>0.889</td><td>0.889</td><td>0.958</td><td>0.944</td><td>0.917</td></tr><tr><td>EigenWorms</td><td>0.431</td><td>0.431</td><td>0.578</td><td>0.511</td><td>0.496</td></tr><tr><td>Epilepsy</td><td>0.775</td><td>0.775</td><td>0.925</td><td>0.870</td><td>0.891</td></tr><tr><td>EthanolConcentration</td><td>0.316</td><td>0.316</td><td>0.284</td><td>0.361</td><td>0.346</td></tr><tr><td>FingerMovements</td><td>0.620</td><td>0.620</td><td>0.580</td><td>0.500</td><td>0.620</td></tr><tr><td>Handwriting</td><td>0.247</td><td>0.247</td><td>0.591</td><td>0.481</td><td>0.307</td></tr><tr><td>Libras</td><td>0.783</td><td>0.783</td><td>0.828</td><td>0.767</td><td>0.800</td></tr><tr><td>NATOPS</td><td>0.906</td><td>0.906</td><td>0.900</td><td>0.922</td><td>0.917</td></tr><tr><td>RacketSports</td><td>0.757</td><td>0.757</td><td>0.861</td><td>0.849</td><td>0.809</td></tr><tr><td>SelfRegulationSCP1</td><td>0.894</td><td>0.894</td><td>0.829</td><td>0.915</td><td>0.898</td></tr><tr><td>SelfRegulationSCP2</td><td>0.483</td><td>0.483</td><td>0.456</td><td>0.511</td><td>0.439</td></tr><tr><td>StandWalkJump</td><td>0.267</td><td>0.267</td><td>0.333</td><td>0.267</td><td>0.533</td></tr><tr><td>UWaveGestureLibrary</td><td>0.838</td><td>0.838</td><td>0.897</td><td>0.887</td><td>0.766</td></tr><tr><td>Avg. acc. (↑)</td><td>0.679</td><td>0.679</td><td>0.721</td><td>0.703</td><td>0.706</td></tr></table>

Table 6: Baseline comparison on UEA datasets: Random Fourier Features (250/500), Random Warping Series (RWS) – number of features = 250, Global Alignment Kernel (GAK), and RBF kernel SVM. Entries are test accuracies using the standard splits.

# C.4 ROBUSTNESS TO MISSING OBSERVATIONS

We assess the stability of all models under partial observations using corrupted versions of the UEA multivariate time-series datasets. Importantly, the corruption mechanism is applied only to the test set, while training is always performed on the clean, unmodified training split. This isolates robustness at inference time.

This experiment uses a feature budget of $N \ : = \ : 6 4$ , matching the configuration examined in the ablation study. Accordingly, the numbers reported here are directly comparable to the $N = 6 4$ results (Table 4) presented in Appendix C.2.

Corruption model. Given an input path $X \in \mathbb { R } ^ { T \times d }$ , we generate a binary mask $M \in \{ 0 , 1 \} ^ { T \times d }$ by independently removing entries with fixed probability p. We consider two regimes:

• Low corruption: $p = 2 0 \% ,$ ,   
• High corruption: $p = 4 0 \%$ .

The corrupted path is defined coordinatewise as

$$
\widetilde {X} _ {t, i} = \left\{ \begin{array}{l l} \text { missing }, & M _ {t, i} = 0, \\ X _ {t, i}, & M _ {t, i} = 1. \end{array} \right.
$$

Missing values are then imputed using simple linear interpolation along the time axis.

Evaluation protocol. All models are trained on the full, uncorrupted training set. During testing, the corrupted–interpolated sequences are processed without modifying the training pipeline. Performance degradation as p increases provides a direct measure of robustness to missing observations.

The hyperparameter grids and the general experimental setup follow exactly the Time-Series Classification experiment and are reported in Section C.1.

<table><tr><td></td><td>R-CDE</td><td>RF-CDE</td><td>R-RDE</td><td>RFSF-DP</td><td>RFSF-TRP</td></tr><tr><td>ArticularyWordRecog.</td><td>0.896</td><td>0.953</td><td>0.903</td><td>0.953</td><td>0.943</td></tr><tr><td>AtrialFibrillation</td><td>0.400</td><td>0.400</td><td>0.533</td><td>0.333</td><td>0.200</td></tr><tr><td>BasicMotions</td><td>0.975</td><td>1.000</td><td>0.975</td><td>0.900</td><td>1.000</td></tr><tr><td>Cricket</td><td>0.944</td><td>0.972</td><td>0.875</td><td>0.875</td><td>0.931</td></tr><tr><td>EigenWorms</td><td>0.458</td><td>0.594</td><td>0.594</td><td>0.611</td><td>0.664</td></tr><tr><td>Epilepsy</td><td>0.906</td><td>0.913</td><td>0.768</td><td>0.913</td><td>0.984</td></tr><tr><td>EthanolConcentration</td><td>0.304</td><td>0.312</td><td>0.324</td><td>0.414</td><td>0.407</td></tr><tr><td>FingerMovements</td><td>0.510</td><td>0.520</td><td>0.510</td><td>0.510</td><td>0.520</td></tr><tr><td>Handwriting</td><td>0.279</td><td>0.302</td><td>0.279</td><td>0.300</td><td>0.228</td></tr><tr><td>Libras</td><td>0.867</td><td>0.906</td><td>0.733</td><td>0.806</td><td>0.878</td></tr><tr><td>NATOPS</td><td>0.844</td><td>0.867</td><td>0.844</td><td>0.778</td><td>0.867</td></tr><tr><td>RacketSports</td><td>0.743</td><td>0.763</td><td>0.539</td><td>0.612</td><td>0.697</td></tr><tr><td>SelfRegulationSCP1</td><td>0.836</td><td>0.884</td><td>0.795</td><td>0.877</td><td>0.826</td></tr><tr><td>SelfRegulationSCP2</td><td>0.533</td><td>0.505</td><td>0.550</td><td>0.483</td><td>0.483</td></tr><tr><td>StandWalkJump</td><td>0.267</td><td>0.400</td><td>0.333</td><td>0.267</td><td>0.200</td></tr><tr><td>UWaveGestureLibrary</td><td>0.817</td><td>0.817</td><td>0.857</td><td>0.806</td><td>0.819</td></tr><tr><td>Avg. acc. (↑)</td><td>0.661</td><td>0.694</td><td>0.650</td><td>0.652</td><td>0.665</td></tr><tr><td>Avg. rank (↓)</td><td>3.438</td><td>1.938</td><td>3.406</td><td>3.406</td><td>2.812</td></tr><tr><td>Avg. acc. % decrease (↓)</td><td>0.695</td><td>2.672</td><td>6.777</td><td>5.074</td><td>5.082</td></tr></table>

Table 7: UEA test accuracies under 20% missing observations, where the corruption is applied only to the test set. The number of features For each dataset, the best result is shown in bold.

Results. Across the 20% and 40% missing-value settings, all models naturally experience a drop in accuracy, but the magnitude of this degradation varies substantially. RF-CDE remains the strongest overall performer, retaining the best average accuracy and best average rank in both corrupted regimes, with only moderate decreases (−2.7% at 20% missing, −5.7% at 40%). R-CDE also degrades slowly (−0.7% and -−5.9%), remaining competitive despite its simpler architecture. By contrast, the RFSF baselines exhibit noticeably larger losses, particularly at higher corruption levels.

Unexpectedly, R-RDE does not perform as well in this experiment, despite the theoretical robustness of signature features to missing data. Its accuracy drops more sharply than for CDE-based reservoirs. We suspect this is partly due to the short UEA time series and the relatively small chunk/step size used in the log-ODE discretisation; in longer forecasting settings, or with larger log-signature chunks, we expect the intrinsic stability of signature features to manifest more clearly.

At the per-dataset level, performance declines are mostly monotone as corruption increases, with RF-CDE continuing to dominate on structured motion datasets, while R-RDE remains competitive on a few highly irregular tasks. Overall, the results suggest that CDE-based reservoirs are comparatively robust to missing inputs, while log-ODE RDEs may require longer temporal horizons or coarser signature blocks to fully leverage their theoretical advantages.

The results are reported in Tables 7 and 8.

<table><tr><td></td><td>R-CDE</td><td>RF-CDE</td><td>R-RDE</td><td>RFSF-DP</td><td>RFSF-TRP</td></tr><tr><td>ArticularyWordRecog.</td><td>0.840</td><td>0.940</td><td>0.840</td><td>0.957</td><td>0.937</td></tr><tr><td>AtrialFibrillation</td><td>0.400</td><td>0.333</td><td>0.400</td><td>0.333</td><td>0.200</td></tr><tr><td>BasicMotions</td><td>0.775</td><td>0.975</td><td>0.925</td><td>0.725</td><td>0.800</td></tr><tr><td>Cricket</td><td>0.972</td><td>0.944</td><td>0.847</td><td>0.819</td><td>0.931</td></tr><tr><td>EigenWorms</td><td>0.458</td><td>0.489</td><td>0.489</td><td>0.489</td><td>0.542</td></tr><tr><td>Epilepsy</td><td>0.884</td><td>0.913</td><td>0.645</td><td>0.826</td><td>0.760</td></tr><tr><td>EthanolConcentration</td><td>0.285</td><td>0.304</td><td>0.319</td><td>0.418</td><td>0.414</td></tr><tr><td>FingerMovements</td><td>0.510</td><td>0.520</td><td>0.500</td><td>0.560</td><td>0.540</td></tr><tr><td>Handwriting</td><td>0.279</td><td>0.279</td><td>0.261</td><td>0.267</td><td>0.221</td></tr><tr><td>Libras</td><td>0.830</td><td>0.883</td><td>0.733</td><td>0.783</td><td>0.850</td></tr><tr><td>NATOPS</td><td>0.710</td><td>0.801</td><td>0.755</td><td>0.755</td><td>0.844</td></tr><tr><td>RacketSports</td><td>0.651</td><td>0.697</td><td>0.454</td><td>0.539</td><td>0.579</td></tr><tr><td>SelfRegulationSCP1</td><td>0.843</td><td>0.887</td><td>0.771</td><td>0.867</td><td>0.812</td></tr><tr><td>SelfRegulationSCP2</td><td>0.522</td><td>0.522</td><td>0.517</td><td>0.517</td><td>0.478</td></tr><tr><td>StandWalkJump</td><td>0.267</td><td>0.469</td><td>0.400</td><td>0.333</td><td>0.267</td></tr><tr><td>UWaveGestureLibrary</td><td>0.803</td><td>0.812</td><td>0.821</td><td>0.812</td><td>0.821</td></tr><tr><td>Avg. acc. (↑)</td><td>0.627</td><td>0.673</td><td>0.604</td><td>0.625</td><td>0.624</td></tr><tr><td>Avg. rank (↓)</td><td>3.281</td><td>2.000</td><td>3.594</td><td>3.062</td><td>3.062</td></tr><tr><td>Avg. acc. % decrease (↓)</td><td>5.858</td><td>5.651</td><td>13.36</td><td>9.058</td><td>10.89</td></tr></table>

Table 8: UEA test accuracies under $4 0 \%$ missing observations, where the corruption is applied only to the test set. The number of features For each dataset, the best result is shown in bold.

# C.5 HURST CLASSIFICATION TASK

Fractional Brownian Motion. Fractional Brownian motion (fBm) $\{ B _ { t } ^ { H } \} _ { t \in [ 0 , 1 ] }$ with Hurst exponent $H \in ( 0 , 1 )$ is the unique mean-zero Gaussian process with covariance

$$
\mathbb {E} \left[ B _ {t} ^ {H} B _ {s} ^ {H} \right] = \frac {1}{2} \left(t ^ {2 H} + s ^ {2 H} - | t - s | ^ {2 H}\right). \tag {26}
$$

The parameter H governs the roughness of sample paths:

• $H < \textstyle { \frac { 1 } { 2 } }$ yields negatively correlated increments and rough trajectories;   
• $\begin{array} { r } { H = \frac { 1 } { 2 } } \end{array}$ recovers standard Brownian motion;   
• $H > \frac { 1 } { 2 }$ yields smoother trajectories with persistent increments.

For $H < { \frac { 1 } { 2 } } ,$ the expected p-variation is finite only for $p > 1 / H ,$ so changes in H induce clear geometric differences. Recovering H from sample paths is therefore a natural benchmark for models based on signature features, kernels, or controlled/rough differential equations.

Data Generation. We generate univariate fractional Brownian motion using the standard Davies-Harte method. For each class

$$
H \in \{0. 0 5, 0. 1 5, \dots , 0. 7 5 \},
$$

we produce:

• $n _ { \mathrm { t r a i n } } = 5 0$ samples per class,   
• $n _ { \mathrm { t e s t } } = 2 5$ samples per class,   
• length ℓ = 256 time steps,

• dimension d = 3 by stacking three independent fBm realisations with the same H.

Thus each input sample is a tensor $\boldsymbol { X } \in \mathbb { R } ^ { N \times 3 }$ and the label is the discrete index of the Hurst value.

V1 and V2 Variants. The first variant (V1) uses the raw fBm trajectories. The second variant (V2) applies a per-sample standardisation

$$
\tilde {X} _ {t, i} = \frac {X _ {t , i} - \mu_ {i}}{\sigma_ {i}}, \quad \mu_ {i} = \frac {1}{N} \sum_ {t = 1} ^ {N} X _ {t, i}, \quad \sigma_ {i} ^ {2} = \frac {1}{N} \sum_ {t = 1} ^ {N} (X _ {t, i} - \mu_ {i}) ^ {2}. \tag {27}
$$

This transformation removes global amplitude differences and forces the model to rely purely on geometric and temporal structure. Since fBm trajectories with different H can have substantially different variances, this normalisation creates a more challenging benchmark in which roughness must be inferred independently of scale.

Number of Features. All Random Differential Equation models and RFSF baselines are evaluated with a feature budget of N = 64. The neural baselines (Neural CDE and Neural RDE) are configured to have a comparable number of parameters to ensure a fair comparison across architectures.

Benchmarks. In this experiment we compare our models exclusively with feature-based baselines: Random Fourier Signature Features (RFSF) with diagonal and tensor projections, Neural Controlled Differential Equations (NCDE) (Kidger et al., 2020), and Neural Rough Differential Equations (NRDE) (Morrill et al., 2021). We additionally evaluated plain Random Fourier Features, which consistently failed to exceed 30% accuracy in any configuration and are therefore omitted for clarity.

Results. The results are reported in Table 3 in Section 4.2.