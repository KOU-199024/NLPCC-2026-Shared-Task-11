# GAS: IMPROVING DISCRETIZATION OF DIFFUSION ODES VIA GENERALIZED ADVERSARIAL SOLVER

Aleksandr Oganov∗,1,†, Ilya Bykov∗,1, Eva Neudachina∗,1, Mishan Aliev1, Alexander Tolmachev4, Alexander Sidorov, Aleksandr Zuev, Andrey Okhotin1,2, Denis Rakitin1, Aibek Alanov1,3

1HSE University, Russia   
2Lomonosov Moscow State University, Russia   
3FusionBrain Lab, AXXX, Russia   
4Moscow Independent Research Institute of Artificial Intelligence, Russia

# ABSTRACT

While diffusion models achieve state-of-the-art generation quality, they still suffer from computationally expensive sampling. Recent works address this issue with gradient-based optimization methods that distill a few-step ODE diffusion solver from the full sampling process, reducing the number of function evaluations from dozens to just a few. However, these approaches often rely on intricate training techniques and do not explicitly focus on preserving fine-grained details. In this paper, we introduce the Generalized Solver: a simple parameterization of the ODE sampler that does not require additional training tricks and improves quality over existing approaches. We further combine the original distillation loss with adversarial training, which mitigates artifacts and enhances detail fidelity. We call the resulting method the Generalized Adversarial Solver and demonstrate its superior performance compared to existing solver training methods under similar resource constraints. Code is available at https://github.com/3145tttt/GAS.

# 1 INTRODUCTION

Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b) offer state-of-the-art generation quality in diverse vision problems, including unconditional and conditional (Dhariwal & Nichol, 2021; Ho & Salimans, 2022) generation, text-to-image (Nichol et al., 2021; Ramesh et al., 2022; Saharia et al., 2022; Rombach et al., 2022; Esser et al., 2024), text-to-video (Blattmann et al., 2023; Brooks et al., 2024; Zheng et al., 2024; Chen et al., 2024b) and even text-to-3D (Poole et al., 2022; Wang et al., 2023) generation. One of the reasons for their success consists in satisfying both high sample quality (Dhariwal & Nichol, 2021; Karras et al., 2022) and mode coverage from the generative trilemma (Xiao et al., 2021). In theory, this allows diffusion models to produce desirable samples from the target distribution given unlimited computation time.

Besides, many improvements were made to satisfy the third requirement on generation speed. One way to tackle high inference time is to train a new model that utilizes the pre-trained diffusion and requires fewer inference steps. This may be achieved by straightening the generation trajectories (Liu et al., 2022b; 2023; Wang et al., 2024) or by directly performing diffusion distillation (Salimans & Ho, 2022; Song et al., 2023; Sauer et al., 2023; Yin et al., 2023) into a few-step student. These training-based methods are capable of fast generation with superior quality on large-scale scenarios. Their training procedures, however, are computation and memory-heavy and may be infeasible for users with resource constraints on cutting-edge problems, such as video generation.

Due to the mentioned resource requirements, the lightweight approach of directly accelerating generation is preferable most of the time. Such inference-time methods as designing specific solvers (Song et al., 2020a; Lu et al., 2022a; Zhang & Chen, 2022), caching intermediate steps (Ma et al., 2024; Wimbauer et al., 2024), or performing quantization (Gu et al., 2022; Badri & Shaji, 2023), push the boundaries of the pre-trained model by utilizing its knowledge as much as possible given a fixed computational budget. Among them, specifically designed solvers are mostly theoretically sound and are capable of producing high-quality samples similar to the full-inference model. However, they require significant hyperparameter search (Zhou et al., 2024b; Zhao et al., 2024) for each model and may be suboptimal depending on the particular setting.

![](images/d2de487655fa8d7997d919bffbf15c75fd14103e3cd816885b89ca2e972b2cec.jpg)  
Figure 1: Illustration of the Generalized Adversarial Solver image generation in comparison with the training-free UniPC (Zhao et al., 2024) solver with equal number of function evaluations (NFE). Our method shows superior results that are almost identical to teacher images in terms of generation quality.

A natural improvement of the idea consists of training (hyper-)parameters of the inference-time "student" sampler to match the full-inference "teacher" model. The approach is free-form and allows for optimizing timestep schedule (Sabour et al., 2024; Tong et al., 2024) as well as the sampler coefficients (Kim et al., 2024; Frankel et al., 2025) for each prediction step. Currently existing methods for training the sampler succeed in improving test-time efficiency of the model compared to the standard solvers. At the same time, they do not realize the full potential of the paradigm and tend to have inefficiencies that lead to nuanced and complicated training schemes. Among these are the unstable loss scale (Sabour et al., 2024), limited parameter space (Tong et al., 2024) and disentanglement of the parameter subsets (Frankel et al., 2025), which we find to be harmful for training. Besides, straightforward sampler distillation into a student with limited parameters may be ineffective for preserving the fine-grained details and may interfere with the generation quality.

In this paper, we aim to tackle the aforementioned issues by introducing a simple yet effective sampler parameterization and modifying the distillation loss. Specifically, we construct a sampler that performs each sampling step by calculating a weighted sum of the current velocity direction with all of the points and directions from previous steps. We propose to utilize a pre-defined solver as a time-dependent guidance and learn correction to its theoretically derived weights to facilitate and accelerate training. On top of that, we endow the sampler distillation with the adversarial loss (Goodfellow et al., 2014) to further boost the sampler quality. Most importantly, we

1. Introduce a novel sampler parameterization that we call the Generalized Solver and demonstrate its significant impact on training acceleration;   
2. Combine it with the adversarial training and validate its positive impact on the fine-grained generation details;   
3. Show that the resulting Generalized Adversarial Solver achieves superior results compared to the existing methods of solver/timestep training on several pixel-space and latent-space data sets.

# 2 BACKGROUND

# 2.1 DIFFUSION MODELS

Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b) simulate the data distribution by defining the forward process of gradual data noising and constructing its time reversal. The forward process is commonly defined by a sequence $\{ p _ { t | 0 } \} _ { t \in [ 0 , T ] }$ of transition probabilities $p _ { t | 0 } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) = \mathcal { N } \left( \mathbf { x } _ { t } \mid \alpha _ { t } \mathbf { x } _ { 0 } , \sigma _ { t } ^ { 2 } \mathbf { I } \right)$ . It perturbs the initial data distribution $p _ { \mathrm { d a t a } } ( \mathbf { x } _ { 0 } ) = p _ { 0 } ( \mathbf { x } _ { 0 } )$ by destroying part of its signal and replacing it with the independent Gaussian noise. Here, αt and $\sigma _ { t } ^ { 2 }$ are positive differentiable functions that define the corresponding noise schedule. Typically, their choice ensures that the sequence of the corresponding marginal distributions $p _ { t } ( \mathbf { x } _ { t } )$ converges to a simple and tractable prior distribution $p _ { T } ( \mathbf { x } _ { T } )$ (e.g. standard normal). For each noise schedule one can construct the equivalent Probability Flow ODE (PF-ODE) (Song et al., 2020b)

$$
\mathrm{d} \mathbf {x} _ {t} = \left[ f (t) \mathbf {x} _ {t} - \frac {1}{2} g ^ {2} (t) \nabla_ {\mathbf {x} _ {t}} \log p _ {t} (\mathbf {x} _ {t}) \right] \mathrm{d} t, \tag {1}
$$

where setting

$$
f (t) = \frac {\mathrm{d} \log \alpha_ {t}}{\mathrm{d} t}, \quad g ^ {2} (t) = \mathrm{d} \sigma_ {t} ^ {2} - 2 \frac {\mathrm{d} \log \alpha_ {t}}{\mathrm{d} t} \sigma_ {t} ^ {2} \tag {2}
$$

and sampling the endpoint $\mathbf { x } _ { T }$ from the prior distribution $p _ { T }$ ensures (Lu et al., 2022a) that $\mathbf { x } _ { t } \sim p _ { t }$ for all timesteps. Essentially, ODE formulation allows one to obtain a backward process of data generation by reversing the velocity of the particle given access to the score function $\nabla _ { \mathbf { x } _ { t } }$ log $p _ { t } ( \mathbf { x } _ { t } )$ of the perturbed data distribution. In practice, diffusion models approximate the score function by optimizing the Denoising Score Matching (Vincent, 2011) objective

$$
\min _ {\theta} \int_ {0} ^ {T} \mathbb {E} _ {p _ {0, t} (\mathbf {x} _ {0}, \mathbf {x} _ {t})} \| s _ {\theta} (\mathbf {x} _ {t}, t) - \nabla_ {\mathbf {x} _ {t}} \log p _ {t | 0} (\mathbf {x} _ {t} | \mathbf {x} _ {0}) \| ^ {2} \mathrm{d} t, \tag {3}
$$

where the score functions $\nabla _ { \mathbf x _ { t } }$ t log $p _ { t | 0 } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } )$ of the conditional Gaussian distributions are tractable and equal $\mathrm { t o } - ( \mathbf { x } _ { t } - \alpha _ { t } \mathbf { x } _ { 0 } ) / \sigma _ { t } ^ { 2 }$ . Besides the score networks, one can directly approximate the ODE velocity function by setting $\bar { \mathbf { \boldsymbol { v } } _ { \theta } } ( \mathbf { x } _ { t } , t ) = f ( t ) \mathbf { \boldsymbol { x } } _ { t } - ( 1 / 2 ) g ^ { 2 } ( t ) s _ { \theta } ( \mathbf { x } _ { t } , t )$ .

# 2.2 ODE SOLVERS

Sampling from a diffusion model amounts to numerically approximating the solution of the corresponding PF-ODE (Eq. 1). Standard numerical methods for solving a general-form ODE $\mathrm { d } \mathbf { x } _ { t } = \pmb { v } ( \mathbf { x } _ { t } , t ) \mathrm { d } t$ are mainly based on approximating the direction ${ \bf x } _ { t + h } - { \bf x } _ { t }$ via Taylor expansion.

The first-order Euler scheme makes a step $h \cdot v ( \mathbf { x } _ { t } , t )$ , which is simple, yet has a large discretization error. Its higher-order modifications generally approximate the derivatives with finite differences. This correction allows Runge-Kutta methods to produce high-quality results (Lu et al., 2022a; Zhang & Chen, 2022; Karras et al., 2022). However, these methods require mid-point evaluations, which harms performance in low-NFE regimes (see e.g. (Zhang & Chen, 2022, Table 2)). In contrast, Linear Multistep solvers (Liu et al., 2022a; Zhang & Chen, 2022) use only previously calculated points and directions for the same approximation, thus remain useful in this setting.

Recently designed solvers such as DDIM (Song et al., 2020a), DPM-Solver(++) (Lu et al., 2022a;b), DEIS (Zhang & Chen, 2022), and UniPC (Zhao et al., 2024), exploit the semi-linear nature of the PF-ODE (Hochbruck & Ostermann, 2010). They approximate the integral in the "variation of constants" formula

$$
\mathbf {x} _ {t} = \frac {\alpha_ {t}}{\alpha_ {u}} \mathbf {x} _ {u} - \int_ {u} ^ {t} \frac {\alpha_ {t}}{\alpha_ {\tau}} \cdot \frac {g ^ {2} (\tau)}{2} s (\mathbf {x} _ {\tau}, \tau) \mathrm{d} \tau , \tag {4}
$$

allowing more accurate steps thanks to the non-unit coefficient of $\mathbf { x } _ { u } ,$ and enabling computationally efficient multistep solvers.

Several previous works highlight the importance of choosing the timestep schedule (the set of time points at which function evaluations are performed), which has a significant impact on the image generation quality (see (Karras et al., 2022, Appendix D.1) and (Frankel et al., 2025, Appendix H.3)).

# 2.3 SOLVER AND SCHEDULE DISTILLATION

Several recently introduced acceleration methods outsource the choice of solver coefficients and the timestep schedule to the gradient-based optimization. Specifically, LD3 (Tong et al., 2024) and S4S (Frankel et al., 2025) formulate this as an instance of knowledge distillation (Hinton et al., 2015). Given the pre-trained diffusion model and the corresponding ODE dx $\mathbf { \mu } _ { \mathrm { ~ t ~ } } = \pmb { v } ( \mathbf { x } _ { t } , t ) \mathrm { d } t$ , one can define the complete "teacher" sampler to be the output of a multi-step high-quality approximation of the PF-ODE, which we denote by

$$
\Phi^ {\mathcal {T}} \left(\mathbf {x} _ {T}\right) = \text { ODESolve } \left(\mathbf {x} _ {T}, \boldsymbol {v} (\cdot , \cdot), T \rightarrow 0 \mid \text { Schedule }, \text { Solver }; \text { Params }\right). \tag {5}
$$

Here, $\mathbf { x } _ { T }$ is the initial value, $v ( \cdot , \cdot )$ is the corresponding velocity field and $T  0$ shows the interval, where we solve the ODE. "Solver" and "Schedule" define the sampling scheme and "Params" account for the additional parameters of the scheme. Then, one could take any parameterization of the lightweight "student"

$$
\Phi^ {\mathcal {S}} \left(\mathbf {x} _ {T} \mid \theta , \phi , \xi\right) = \text { ODESolve } \left(\mathbf {x} _ {T}, \boldsymbol {v} (\cdot , \cdot), T \rightarrow 0 \mid \text { Schedule } (\theta), \text { Solver } (\phi); \text { Params } (\xi)\right) \tag {6}
$$

with bounded computational requirements and optimize its parameters by minimizing a distance d between the corresponding outputs

$$
\min _ {\theta , \phi , \xi} \mathcal {L} _ {\text { distill }} (\theta , \phi , \xi) = \min _ {\theta , \phi , \xi} \mathbb {E} _ {p _ {T} (\mathbf {x} _ {T})} \mathbf {d} \left(\Phi^ {\mathcal {S}} (\mathbf {x} _ {T} \mid \theta , \phi , \xi); \Phi^ {\mathcal {T}} (\mathbf {x} _ {T})\right). \tag {7}
$$

In addition, LD3 and S4S account for the limited parameterization of the student and simplify its objective by allowing to slightly adapt the input and facilitate replication of the teacher output

$$
\min _ {\theta , \phi , \xi} \mathcal {L} _ {\text {soft}} (\theta , \phi , \xi) = \min _ {\theta , \phi , \xi} \mathbb {E} _ {p _ {T} \left(\mathbf {x} _ {T}\right)} \min _ {\mathbf {x} _ {T} ^ {\prime} \in \mathcal {B} \left(\mathbf {x} _ {T}, r \sigma_ {T}\right)} \mathbf {d} \left(\Phi^ {\mathcal {S}} \left(\mathbf {x} _ {T} ^ {\prime} \mid \theta , \phi , \xi\right); \Phi^ {\mathcal {T}} \left(\mathbf {x} _ {T}\right)\right), \tag {8}
$$

where $\begin{array} { r } { B ( \mathbf { x } _ { T } , r \sigma _ { T } ) = \{ \mathbf { x } : \| \mathbf { x } - \mathbf { x } _ { T } \| ^ { 2 } \leq r \sigma _ { T } \} } \end{array}$ is the ball centered in $\mathbf { x } _ { T }$ with a radius rσT controlled by the additional hyperparameter r. We thoroughly discuss parameterizations of the methods and compare them with our Generalized Solver in Section 3.1.

# 2.4 ADVERSARIAL TRAINING

Adversarial training (Goodfellow et al., 2014) is a powerful way to guide a free-form generator $G _ { \theta } ( \mathbf { z } )$ towards realistic outputs via optimizing the minimax objective (Nowozin et al., 2016)

$$
\min _ {\theta} \max _ {\psi} \mathbb {E} _ {p (\mathbf {z})} f \left(D _ {\psi} (G _ {\theta} (\mathbf {z}))\right) + \mathbb {E} _ {p _ {\mathrm{data}} (\mathbf {x})} f \left(- D _ {\psi} (\mathbf {x})\right). \tag {9}
$$

Here, $f ( t )$ is commonly equal $\mathrm { t o } - \log ( 1 + e ^ { - t } )$ , the discriminator $D _ { \psi }$ is trained to distinguish real samples from the fake ones, while the generator aims to trick it. Family of the GAN losses with the form of Equation 9 (Nowozin et al., 2016; Mao et al., 2017; Lim & Ye, 2017) suffers from mode collapse (Arjovsky et al., 2017; Gulrajani et al., 2017). One of the alternatives is the relativistic GAN loss (Jolicoeur-Martineau, 2018)

$$
\min _ {\theta} \max _ {\psi} \mathbb {E} _ {p (\mathbf {z}) p _ {\text { data }} (\mathbf {x})} f \left(D _ {\psi} (G _ {\theta} (\mathbf {z})) - D _ {\psi} (\mathbf {x})\right) \tag {10}
$$

that is specifically designed to discourage mode dropping (Sun et al., 2020). Together with the gradient penalty

$$
\mathcal {L} _ {\text { grad }} (\theta , \psi) = \lambda_ {1} \mathbb {E} _ {p _ {\text { data }} (\mathbf {x})} \| \nabla_ {\mathbf {x}} D _ {\psi} (\mathbf {x}) \| ^ {2} + \lambda_ {2} \mathbb {E} _ {p (\mathbf {z})} \| \nabla_ {\mathbf {x}} D _ {\psi} (G _ {\theta} (\mathbf {z})) \| ^ {2} \tag {11}
$$

on discriminator outputs and architecture improvements, relativistic loss allows Huang et al. (2024) to build a novel high-quality GAN baseline R3GAN which we use throughout the paper.

# 3 METHOD

In this section, we construct Generalized Adversarial Solver (GAS): an automatic sampler learning method that combines a simple yet effective parameterization with distillation and adversarial training.

# 3.1 GENERALIZED SOLVER (GS)

In Section 2.2 we have discussed that linear multi-step solvers and their specifically designed diffusion counterparts are the preferable families under strict requirements on computations. Given a timestep schedule $T = t _ { 0 } > \bar { t } _ { 1 } > . . . > t _ { N } = \delta > 0$ and order K they all have the same signature

$$
\mathbf {x} _ {n + 1} = a _ {n} \mathbf {x} _ {n} + \sum_ {j = \max (n - K + 1, 0)} ^ {n} c _ {j, n} \boldsymbol {v} (\mathbf {x} _ {j}, t _ {j}), \tag {12}
$$

where the coefficients $a _ { n } : = a _ { n } ( t _ { n } , t _ { n + 1 } )$ and $c _ { j , n } : = c _ { j , n } ( t _ { j : n + 1 } )$ typically depend on the current and the next timesteps. We propose several modifications to this basic signature. First, we stress that the less restriction on NFE is, the fewer parameters the method has. Second, one can see that depending on the parameterization of the diffusion model the formula may also contain the weighted sum of previous points (e.g., if one substitutes $v ( \mathbf { x } _ { j } , t _ { j } ) = f ( t _ { j } ) \mathbf { x } _ { j } - ( 1 / 2 ) g ^ { 2 } ( t _ { j } ) s ( \mathbf { x } _ { j } , t _ { j } ) )$ ) along with the network predictions. We thus propose to increase the capacity of the signature by adding the weighted sum of all previous points 1 and remove the restriction on the order of the solver:

![](images/cfb8b335a4ac1d61136214147de3989acd26ce1454eb2bcbe0dfc2efa256fb8a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Teacher"] --> B["Teacher0"]
    A --> C["TeacherM"]
    B --> D["Ldistill + Ladv"]
    C --> D
    E["Student"] --> F["Student0"]
    E --> G["Student1"]
    E --> H["Student2"]
    E --> I["Student3"]
    D --> J["GAS"]
    D --> K["GAS"]
    J --> L["×an-2n(φ)"]
    K --> M["×an-1n(φ)"]
    L --> N["×an,n(θ,φ)"]
    M --> O["×cn-1,n(θ,φ)"]
    N --> P["v(·,tn^θ +ξn)"]
    O --> Q["×cn,n(θ,φ)"]
    P --> R["×cn-2n(θ,φ)"]
    Q --> S["v(xn,tn^θ +ξn)"]
    R --> T["GAS"]
    S --> U["GAS"]
```
</details>

Figure 2: Illustration of the Generalized Adversarial Solver. Our student makes each sampling step by calculating the weighted average of all previous points and velocity directions. We train the corresponding weights and timestep schedule via distillation and adversarial loss.

$$
\mathbf {x} _ {n + 1} = \sum_ {j = 0} ^ {n} a _ {j, n} \mathbf {x} _ {j} + \sum_ {j = 0} ^ {n} c _ {j, n} \boldsymbol {v} (\mathbf {x} _ {j}, t _ {j}). \tag {13}
$$

Given this signature, we next define our parameterization that has three sets of parameters: $( \theta , \zeta )$ . The first set θ of parameters defines the timestep schedule via the cumprod transformation: the logits $\theta _ { n }$ are transformed into "stick breaking" portions $\sigma ( \theta _ { n } ) \in [ 0 , 1 ]$ . The timesteps are then defined as

$$
t _ {n} ^ {\theta} = (T - \delta) \prod_ {j = 1} ^ {n} \sigma (\theta_ {j}) + \delta . \tag {14}
$$

The second set ϕ defines the solver coefficients. However, we do not straightforwardly set $a _ { j , n } : = a _ { j , n } ( \phi )$ and $c _ { j , n } : = c _ { j , n } ( \phi )$ . Instead, we use a powerful base multi-step solver (e.g. DPM-Solver++(3M) (Lu et al., 2022b)) as a source of theoreticThis base solver offers time-dependent theoretical coefficients $a _ { n , n } ( t _ { n : n + 1 } ^ { \theta } )$ or the and $c _ { j , n } \big ( t _ { j : n + 1 } ^ { \theta } \big )$ ficients., which we can use as a strong backbone for our solver. We then train additive corrections to these coefficients in the following way. We set

$$
a _ {j, n} (\theta , \phi) := \left\{ \begin{array}{l} a _ {n, n} (t _ {n: n + 1} ^ {\theta}) + \hat {a} _ {n, n} (\phi), \quad j = n; \\ \hat {a} _ {j, n} (\phi), \quad \text { else }, \end{array} \right. \tag {15}
$$

thus adding a trainable scalar $\hat { a } _ { n , n } ( \mathbf { \Sigma } )$ to the current point coefficient $a _ { n , n } \big ( t _ { n : n + 1 } ^ { \theta } \big )$ and training scalars $\hat { a } _ { j , n } ( \mathbf { \Sigma } )$ for all the previous point coefficients.

Next, since the $" o l d "$ velocities (computed more than K steps before) do not have theoretical coefficients, we train one scalar $\hat { c } _ { j , n } ( \phi )$ per timestep $j \le n - K$ and set

$$
c _ {j, n} (\theta , \phi) = \hat {c} _ {j, n} (\phi). \tag {16}
$$

Finally, we define the coefficients before the "recent" velocities (computed less than K steps before). Here, theoretical base coefficients are typically constructed via weighted sum of the approximations $\hat { \pmb v } _ { n } ^ { ( j ) }$ of the higher-order derivatives ${ \pmb v } ^ { ( j ) } ( { \bf x } _ { n } , t _ { n } ^ { \theta } )$ via finite differences (which are themselves weighted sums of previously computed velocities). This leads to the sum of the form $\begin{array} { r l } { ~ } & { { } \sum _ { j = 0 } ^ { K - 1 } \widetilde c _ { j , n } \big ( t _ { j : n + 1 } ^ { \theta } \big ) \cdot \hat { \pmb { v } } _ { n } ^ { ( j ) } } \end{array}$

Table 1: Comparison of solver parameterizations between our GS, LD3 (Tong et al., 2024) and S4S (Frankel et al., 2025). We propose to add additive guidance to several velocity coefficients with a theoretical term from a pre-defined solver instead of just two multiplicative terms $a _ { n }$ and $b _ { n }$ . The guidance is marked by the dependence of coefficients on θ. We add a weighted sum of the previous points to the prediction. 

<table><tr><td>Method</td><td>Parameterization</td></tr><tr><td>LD3</td><td> $\mathbf{x}_{n+1} = a_n(t_n^\theta, t_{n+1}^\theta) \cdot \mathbf{x}_n + \sum_{j=\max(n-K+1,0)}^n c_{j,n}(t_j^\theta, \ldots, t_{n+1}^\theta) \cdot \boldsymbol{v}(\mathbf{x}_j, t_j^\theta + \xi_j)$ </td></tr><tr><td>S4S</td><td> $\mathbf{x}_{n+1} = a_n(t_n^\theta, t_{n+1}^\theta) \cdot \mathbf{x}_n + b_n(t_n^\theta, t_{n+1}^\theta) \cdot \sum_{j=\max(n-K+1,0)}^n c_{j,n}(\phi) \cdot \boldsymbol{v}(\mathbf{x}_j, t_j^\theta + \xi_j)$ </td></tr><tr><td>GS</td><td> $\mathbf{x}_{n+1} = a_{n,n}(\theta, \phi) \cdot \mathbf{x}_n + \sum_{j=0}^{n-1} a_{j,n}(\phi) \cdot \mathbf{x}_j + \sum_{j=0}^n c_{j,n}(\theta, \phi) \cdot \boldsymbol{v}(\mathbf{x}_j, t_j^\theta + \xi_j)$ </td></tr></table>

Combined with the finite-difference approximation of the derivatives $\begin{array} { r } { \hat { \mathbf { v } } _ { n } ^ { ( j ) } = \sum _ { i = n - j } ^ { n } \omega _ { i , n } \cdot \mathbf { v } ( \mathbf { x } _ { i } , t _ { i } ^ { \theta } ) } \end{array}$ we obtain

$$
\sum_ {j = 0} ^ {K - 1} \tilde {c} _ {j, n} (t _ {j: n + 1} ^ {\theta}) \cdot \sum_ {i = n - j} ^ {n} \omega_ {i, n} \cdot \boldsymbol {v} (\mathbf {x} _ {i}, t _ {i} ^ {\theta}). \tag {17}
$$

Here, we train additive corrections $\hat { c } _ { j , n } ( \phi )$ for the coefficients $\tilde { c } _ { j , n } \big ( t _ { j : n + 1 } ^ { \theta } \big )$ corresponding to the derivatives approximation. We thus obtain sum

$$
\sum_ {j = 0} ^ {K - 1} \left[ \tilde {c} _ {j, n} (t _ {j: n + 1} ^ {\theta}) + \hat {c} _ {j, n} (\phi) \right] \cdot \sum_ {i = n - j} ^ {n} \omega_ {i, n} \cdot \boldsymbol {v} (\mathbf {x} _ {i}, t _ {i} ^ {\theta}), \tag {18}
$$

which produces recent velocity coefficients

$$
c _ {i, n} (\theta , \phi) = \omega_ {i, n} \sum_ {j = n - i} ^ {K - 1} \left[ \tilde {c} _ {j, n} (t _ {j: n + 1} ^ {\theta}) + \hat {c} _ {j, n} (\phi) \right]. \tag {19}
$$

We initialize the corrections with zeros to obtain an efficient initialization. By doing this, we ensure that even sudden change of the timesteps does not completely ruin the solver performance due to the meaningful dependence of its coefficients on time. We show the positive impact of the theoretical guidance in Section 4.2.

The last set ξ of parameters acts as a correction to the timesteps that we evaluate the pre-trained model on. Analogous to Tong et al. (2024) and Frankel et al. (2025) we define the decoupled timesteps $t _ { j } ^ { \theta } + \xi _ { j }$ and use them for making predictions with the diffusion model. Combining the signature from Equation 13 with the introduced parameterization, we obtain the Generalized Solver (GS)

$$
\mathbf {x} _ {n + 1} = a _ {n, n} (\theta , \phi) \cdot \mathbf {x} _ {n} + \sum_ {j = 0} ^ {n - 1} a _ {j, n} (\phi) \cdot \mathbf {x} _ {j} + \sum_ {j = 0} ^ {n} c _ {j, n} (\theta , \phi) \cdot \boldsymbol {v} \left(\mathbf {x} _ {j}, t _ {j} ^ {\theta} + \xi_ {j}\right) \tag {20}
$$

and extensively compare it with the parameterizations of LD3 and S4S in Table 1.

Taken together, the following design choices of the Generalized Solver improve its quality over existing approaches:

• the use of theoretical coefficients, which form the basis of GS and improve convergence;   
• the signature of the linear multistep method, on which many theoretical solvers are based, determines the use of the past history of $\mathbf { x } _ { j } ;$ ;   
• additive parameterization, which connects the theoretical and trainable solver coefficients within the signature.

# 3.2 GENERALIZED ADVERSARIAL SOLVER (GAS)

We train the Generalized Solver on the previously established distillation loss from Equation 7. Specifically, we take d from the distillation loss (Equation 7) to be LPIPS in pixel-space and $L _ { 1 }$ in latent-space experiments. We do not use the soft version from Equation 8. It is important to examine the "solver distillation" problem from another perspective. Essentially, it is an instance of the paired translation problem/learning a mapping from its input/output samples. Several works (Isola et al., 2017; Ledig et al., 2017) have shown that the standard regression loss could greatly benefit from adding the adversarial loss on the outputs. Recently, adversarial loss has been established as a powerful tool to boost performance of the diffusion distillation (Kim et al., 2023; Sauer et al., 2023; 2024; Yin et al., 2024) methods.

Given this, we augment distillation-based training of the GS via distillation loss and obtain the Generalized Adversarial Solver (GAS). We denote our solver’s output as

$$
\Phi^ {\mathcal {S}} \left(\mathbf {x} _ {T} | \theta , \phi , \xi\right) = \operatorname{ODESolve} \left(\mathbf {x} _ {T}, \boldsymbol {v} (\cdot , \cdot), T \rightarrow 0 \mid \mathrm{GS} (\theta , \phi , \xi)\right), \tag {21}
$$

where $\operatorname { G S } ( \theta , \phi , \xi )$ defines the Generalized Solver signature and parameterization, defined in Section 3.1 and Equation 20 specifically. We denote the discriminator by $D _ { \psi }$ and train GAS on the sum of distillation and adversarial losses

$$
\left\{ \begin{array}{l} \min _ {\theta , \phi , \xi} \max _ {\psi} \mathcal {L} _ {\mathrm{GAS}} (\theta , \phi , \xi , \psi) = \min _ {\theta , \phi , \xi} \max _ {\psi} \mathcal {L} _ {\text { distill }} (\theta , \phi , \xi) + \mathcal {L} _ {\mathrm{adv}} (\theta , \phi , \xi , \psi); \\ \mathcal {L} _ {\mathrm{adv}} (\theta , \phi , \xi , \psi) = \mathbb {E} _ {p _ {T} (\mathbf {x} _ {T}) p _ {T} (\mathbf {y} _ {T})} f \left(D _ {\psi} \left(\Phi^ {\mathcal {S}} \left(\mathbf {x} _ {T} | \theta , \phi , \xi\right)\right) - D _ {\psi} \left(\Phi^ {\mathcal {T}} (\mathbf {y} _ {T})\right)\right). \end{array} \right. \tag {22}
$$

We note that $\mathbf { x } _ { T }$ and $\mathbf { y } _ { T }$ are different initial noises for student and teacher generation sampled from the same prior distribution. We exploit R3GAN (Huang et al., 2024) relativistic loss with $f ( t ) = - \log ( 1 + e ^ { - t } )$ and add the discriminator gradient penalties from Equation 11 to facilitate its training dynamics.

The incorporation of the adversarial loss is also effective in terms of removing generation artifacts in low NFE regimes, where regression task becomes harder. We will further demonstrate this in Section 4.

# 4 EXPERIMENTS

We demonstrate the efficiency of the proposed method by conducting experiments on several pixel and latent space experiments. We perform evaluation on pixel-space CIFAR10 (32×32) (Krizhevsky & Hinton, 2009), FFHQ (64×64) (Karras et al., 2019), and AFHQv2 (64×64) (Choi et al., 2020). Among latent diffusion models (Rombach et al., 2022) we cover LSUN Bedroom (256×256) (Yu et al., 2015) and the class-conditional ImageNet (256×256) (Russakovsky et al., 2015). Additionally, we assess the Stable Diffusion (Rombach et al., 2022) model on the MSCOCO (512×512) (Lin et al., 2015) text-to-image dataset. We use Karras et al. (2022) and Rombach et al. (2022) pretrained models for pixel and latent space experiments respectively.

We choose distance d (Equation 7) in distillation loss to be LPIPS (Zhang et al., 2018) in pixel-space and $L _ { 1 }$ in latent-space experiments. We initialize timesteps using a time-uniform schedule and utilize the DPM-Solver++(3M) (Lu et al., 2022b) coefficients as the guiding theoretical parameters. For pixel-space models we use a pretrained R3GAN discriminator. For latent experiments we adapt the same discriminator architecture, but train it from scratch. We calculate FID (Heusel et al., 2017) using 50000 samples, unless stated otherwise. The additional training details can be found in Appendix D.

# 4.1 MAIN RESULTS

In Table 2 we illustrate that the proposed methods, GS and GAS, systematically enhance image sampling quality across different solvers, especially in low NFE setups. As an example, the S4S Alt (Frankel et al., 2025) algorithm reports a FID score of 10.63 with NFE=4 on the FFHQ dataset, whereas GAS achieves a significantly better FID score of 7.86 under the same conditions. Our approach outperforms all previously proposed methods across all evaluated datasets. Specifically, GAS achieves a FID score of 4.48 with NFE=4 on the AFHQv2 dataset and 3.79 on the FFHQ dataset using NFE=6. Additionally, we achieve the FID score of 5.38 on the conditional ImageNet dataset with NFE=4, 4.60 on the LSUN Bedrooms dataset with NFE=5, and 14.71 on the MS-COCO dataset with NFE = 4.

Table 2: We evaluate FID score comparison of the proposed GS and GAS methods against existing solvers like UniPC and iPNDM, and alongside training-based approaches such as GITS, DMN, LD3, and S4S. We report the FIDs of the teacher models as those utilized during our training process. The baseline scores were taken from the corresponding papers, unless otherwise noted.† report utilizing teacher model having significantly difference in teacher hyperparameters, thus it cannot be fairly compared to other methods.

(a) Pixel-space datasets include CIFAR10 (32×32), AFHQv2 (64×64), and FFHQ (64×64) 

<table><tr><td>Method</td><td>NFE=4</td><td>NFE=6</td><td>NFE=8</td><td>NFE=10</td></tr><tr><td colspan="5">CIFAR10</td></tr><tr><td colspan="5">Solvers</td></tr><tr><td>DPM++ (3M)</td><td>46.59</td><td>12.16</td><td>4.62</td><td>3.08</td></tr><tr><td>UniPC (3M)</td><td>43.92</td><td>13.12</td><td>4.41</td><td>3.16</td></tr><tr><td>iPNDM (3M)</td><td>35.04</td><td>11.80</td><td>5.67</td><td>3.69</td></tr><tr><td colspan="5">Solver optimization methods</td></tr><tr><td>UniPC [GITS]</td><td>25.32</td><td>11.19</td><td>5.67</td><td>3.70</td></tr><tr><td>UniPC [DMN]</td><td>26.35</td><td>8.09</td><td>5.90</td><td>2.45</td></tr><tr><td>iPNDM [GITS]</td><td>15.63</td><td>6.82</td><td>4.29</td><td>2.78</td></tr><tr><td>iPNDM [DMN]</td><td>28.09</td><td>9.24</td><td>7.68</td><td>3.31</td></tr><tr><td>Best LD3</td><td>9.31</td><td>3.35</td><td>2.81</td><td>2.38</td></tr><tr><td>S4S Alt</td><td>6.35</td><td>2.67</td><td>2.39</td><td>2.18</td></tr><tr><td>GS (Ours)</td><td>4.41</td><td>2.55</td><td>2.25</td><td>2.18</td></tr><tr><td>GAS (Ours)</td><td>4.05</td><td>2.49</td><td>2.24</td><td>2.17</td></tr><tr><td>Teacher</td><td colspan="4">2.03</td></tr><tr><td colspan="5">FFHQ</td></tr><tr><td colspan="5">Solvers</td></tr><tr><td>DPM++ (3M)</td><td>46.14</td><td>14.01</td><td>6.18</td><td>4.18</td></tr><tr><td>UniPC (3M)</td><td>53.25</td><td>11.24</td><td>5.59</td><td>3.90</td></tr><tr><td>iPNDM (3M)</td><td>36.54</td><td>16.44</td><td>8.11</td><td>5.39</td></tr><tr><td colspan="5">Solver optimization methods</td></tr><tr><td>UniPC [GITS]</td><td>21.38</td><td>12.21</td><td>7.84</td><td>4.46</td></tr><tr><td>UniPC [DMN]</td><td>25.82</td><td>9.47</td><td>6.85</td><td>3.54</td></tr><tr><td>iPNDM [GITS]</td><td>18.05</td><td>9.38</td><td>5.72</td><td>3.96</td></tr><tr><td>iPNDM [DMN]</td><td>31.30</td><td>12.12</td><td>11.00</td><td>5.24</td></tr><tr><td>Best LD3</td><td>17.96</td><td>5.97</td><td>3.50</td><td>3.25</td></tr><tr><td>S4S Alt</td><td>10.63</td><td>4.62</td><td>3.15</td><td>2.91</td></tr><tr><td>GS (Ours)</td><td>10.70</td><td>4.49</td><td>2.96</td><td>2.67</td></tr><tr><td>GAS (Ours)</td><td>7.86</td><td>3.79</td><td>2.87</td><td>2.66</td></tr><tr><td>Teacher</td><td colspan="4">2.60</td></tr><tr><td colspan="5">AFHQv2</td></tr><tr><td colspan="5">Solvers</td></tr><tr><td>DPM++ (3M)</td><td>27.82</td><td>10.72</td><td>4.28</td><td>3.19</td></tr><tr><td>UniPC (3M)</td><td>33.78</td><td>8.27</td><td>4.60</td><td>3.81</td></tr><tr><td>iPNDM (3M)</td><td>23.20</td><td>9.55</td><td>4.49</td><td>3.19</td></tr><tr><td colspan="5">Solver optimization methods</td></tr><tr><td>UniPC [GITS]</td><td>12.20</td><td>7.26</td><td>3.86</td><td>2.88</td></tr><tr><td>UniPC [DMN]</td><td>30.32</td><td>14.46</td><td>6.85</td><td>2.94</td></tr><tr><td>iPNDM [GITS]</td><td>12.89</td><td>6.10</td><td>4.03</td><td>3.26</td></tr><tr><td>iPNDM [DMN]</td><td>33.15</td><td>16.01</td><td>10.12</td><td>3.22</td></tr><tr><td>Best LD3</td><td>9.96</td><td>3.63</td><td>2.63</td><td>2.27</td></tr><tr><td>S4S Alt</td><td>6.52</td><td>2.70</td><td>2.29</td><td>2.18</td></tr><tr><td>GS (Ours)</td><td>5.92</td><td>2.87</td><td>2.33</td><td>2.25</td></tr><tr><td>GAS (Ours)</td><td>4.48</td><td>2.66</td><td>2.29</td><td>2.31</td></tr><tr><td>Teacher</td><td colspan="4">2.16</td></tr></table>

(b) Latent diffusion models are tested on the LSUN-Bedroom and ImageNet datasets (256×256). 

<table><tr><td>Method</td><td>NFE=4</td><td>NFE=5</td><td>NFE=6</td><td>NFE=7</td></tr><tr><td colspan="5">LSUN-Bedroom-256 (latent space)</td></tr><tr><td colspan="5">Solvers</td></tr><tr><td>DPM++ (3M)</td><td>48.82</td><td>18.64</td><td>8.50</td><td>5.16</td></tr><tr><td>UniPC (3M)</td><td>39.78</td><td>13.88</td><td>6.57</td><td>4.56</td></tr><tr><td>iPNDM (3M)</td><td>11.93</td><td>6.38</td><td>5.08</td><td>4.39</td></tr><tr><td colspan="5">Solver optimization methods</td></tr><tr><td>UniPC [GITS]</td><td>70.93</td><td>47.37</td><td>22.33</td><td>17.27</td></tr><tr><td>UniPC [DMN]</td><td>29.22</td><td>8.21</td><td> $\underline{4.40}$ </td><td>4.55</td></tr><tr><td>iPNDM [GITS]</td><td>76.86</td><td>59.17</td><td>28.09</td><td>19.54</td></tr><tr><td>iPNDM [DMN]</td><td>11.82</td><td>6.15</td><td>4.71</td><td>5.16</td></tr><tr><td>Best LD3</td><td> $\underline{8.48}$ </td><td>5.93</td><td>4.52</td><td>4.16</td></tr><tr><td>S4S Alt $^{\dagger}$ </td><td>20.89</td><td>13.03</td><td>10.49</td><td>10.03</td></tr><tr><td>GS (Ours)</td><td>9.83</td><td> $\underline{5.32}$ </td><td> $\underline{3.77}$ </td><td> $\underline{3.34}$ </td></tr><tr><td>GAS (Ours)</td><td> $\underline{6.68}$ </td><td> $\underline{4.60}$ </td><td> $\underline{3.77}$ </td><td> $\underline{3.36}$ </td></tr><tr><td>Teacher</td><td colspan="4">3.06</td></tr><tr><td colspan="5">Imagenet-256 (latent space)</td></tr><tr><td colspan="5">Solvers</td></tr><tr><td>DPM++ (3M)</td><td>26.07</td><td>11.91</td><td>7.51</td><td>5.95</td></tr><tr><td>UniPC (3M)</td><td>20.01</td><td>8.51</td><td>5.92</td><td>5.20</td></tr><tr><td>iPNDM (3M)</td><td>13.86</td><td>7.80</td><td>6.03</td><td>5.35</td></tr><tr><td colspan="5">Solver optimization methods</td></tr><tr><td>UniPC [GITS]</td><td>54.88</td><td>34.91</td><td>14.62</td><td>9.04</td></tr><tr><td>UniPC [DMN]</td><td>16.72</td><td>7.96</td><td>7.54</td><td>7.81</td></tr><tr><td>iPNDM [GITS]</td><td>56.00</td><td>43.56</td><td>19.33</td><td>10.33</td></tr><tr><td>iPNDM [DMN]</td><td>10.15</td><td>7.33</td><td>7.25</td><td>7.40</td></tr><tr><td>Best LD3</td><td>9.19</td><td>5.03</td><td>4.46</td><td>4.32</td></tr><tr><td>S4S Alt $^{\dagger}$ </td><td> $\underline{5.13}$ </td><td> $\underline{4.30}$ </td><td> $\underline{4.09}$ </td><td> $\underline{4.06}$ </td></tr><tr><td>GS (Ours)</td><td>7.87</td><td>4.93</td><td> $\underline{4.30}$ </td><td> $\underline{4.17}$ </td></tr><tr><td>GAS (Ours)</td><td> $\underline{5.38}$ </td><td> $\underline{4.87}$ </td><td>4.32</td><td> $\underline{4.17}$ </td></tr><tr><td>Teacher</td><td colspan="4">4.10</td></tr></table>

(c) Training dataset for SD consists of 1000 MS-COCO samples, while FID is computed across 30,000 prompts to generate images with spatial resolution of 512×512. 

<table><tr><td>Method</td><td>NFE=4</td><td>NFE=5</td><td>NFE=6</td><td>NFE=7</td></tr><tr><td colspan="5">MS-COCO (Stable Diffusion v1.5)</td></tr><tr><td>iPNDM (2M)</td><td>17.76</td><td>14.41</td><td>13.86</td><td>13.76</td></tr><tr><td>iPNDM [GITS]</td><td>18.05</td><td>14.11</td><td>12.10</td><td>11.80</td></tr><tr><td>Best LD3</td><td>17.32</td><td>13.07</td><td>12.40</td><td>11.83</td></tr><tr><td> $S4S^†$ </td><td>16.05</td><td>13.26</td><td>11.17</td><td>10.83</td></tr><tr><td>GS (Ours)</td><td>14.94</td><td>11.97</td><td>11.71</td><td>11.32</td></tr><tr><td>GAS (Ours)</td><td>14.71</td><td>11.91</td><td>11.73</td><td>11.36</td></tr><tr><td>Teacher</td><td>14.10</td><td>12.08</td><td>11.80</td><td>11.48</td></tr></table>

# 4.2 ABLATION STUDY

Coefficients parametrization First, we demonstrate significant impact of solver parameterization on training efficiency. Specifically, we show the difference between our parameterization, that represents coefficients as sum of fixed theoretical guidance and explicitly trained additive corrections, and the parameterization from another high-quality method S4S (Frankel et al., 2025). We ablate the theoretical guidance in Appendix B.1 and demonstrate that it yields a substantial improvement in FID. For the purpose of a fair comparison with S4S, we implemented LMS + PC S4S solver type removing a constraint on the solver order. This guarantees that Generalized Solver and S4S have the same number of trainable parameters.

Table 3: We compare our parametrization with S4S variant on CIFAR10 and FFHQ datasets in terms of FID and LPIPS scores. Both setups use batch size of 24, while training dataset consists of 49k samples. Teacher dataset has FID score of 2.03 and 2.60 for CIFAR10 and FFHQ datasets respectively. 

<table><tr><td></td><td colspan="2">NFE=4</td><td colspan="2">NFE=6</td><td colspan="2">NFE=8</td><td colspan="2">NFE=10</td></tr><tr><td></td><td>FID</td><td>LPIPS</td><td>FID</td><td>LPIPS</td><td>FID</td><td>LPIPS</td><td>FID</td><td>LPIPS</td></tr><tr><td colspan="9">CIFAR10</td></tr><tr><td>S4S</td><td>31.44</td><td>0.273</td><td>2.93</td><td>0.073</td><td>2.87</td><td>0.072</td><td>2.26</td><td>0.027</td></tr><tr><td>Our</td><td>4.39</td><td>0.116</td><td>2.51</td><td>0.046</td><td>2.21</td><td>0.017</td><td>2.15</td><td>0.010</td></tr><tr><td colspan="9">FFHQ</td></tr><tr><td>S4S</td><td>24.24</td><td>0.175</td><td>11.08</td><td>0.117</td><td>7.76</td><td>0.098</td><td>3.97</td><td>0.045</td></tr><tr><td>Our</td><td>10.79</td><td>0.116</td><td>4.40</td><td>0.046</td><td>2.97</td><td>0.016</td><td>2.70</td><td>0.005</td></tr></table>

![](images/91c174823e47438c1c4266d067bd7978103e3a5bb4dd6ddcbb39afcbc340c6d5.jpg)

<details>
<summary>line</summary>

| Training iteration | Ours NFE=6 | Ours NFE=8 | Ours NFE=10 | S4S NFE=6 | S4S NFE=8 | S4S NFE=10 |
| ------------------ | ---------- | ---------- | ----------- | --------- | --------- | ---------- |
| 0                  | 0.25       | 0.25       | 0.25        | 0.25      | 0.25      | 0.25       |
| 10000              | 0.05       | 0.03       | 0.02        | 0.12      | 0.11      | 0.10       |
| 20000              | 0.05       | 0.03       | 0.02        | 0.12      | 0.11      | 0.10       |
| 30000              | 0.05       | 0.03       | 0.02        | 0.12      | 0.11      | 0.10       |
| 40000              | 0.05       | 0.03       | 0.02        | 0.12      | 0.11      | 0.10       |
</details>

Figure 3: LPIPS evaluation loss for training iterations comparing S4S and our parametrization. Our method results in more stable training process.

In Table 3 we demonstrate our parameterization’s superior performance on different datasets and NFE. Our results are consistent with the training issue reported in (Frankel et al., 2025). Figure 3 represents the dynamics of LPIPS loss on the evaluation dataset in different training iterations of the experiment. Our parametrization shows a more efficient training process, faster convergence and more stable training behavior. We also compare GS with LMS and LMS+PC under identical configurations in Appendix B.9.

We observe that training with our parametrization for GS and GAS is stable, demonstrating an improvement in FID throughout the training process. More details are provided in Appendix B.3.

Adversarial training Addition of the adversarial training is a crucial part of our contribution, because it significantly improves the image generation quality as seen in Tables 2a, 2b. It is crucial for low NFE setups because a teacher image can be too difficult for the student to replicate, therefore smaller values of the regression loss (LPIPS or L1 for pixel and latent models respectively) do not always correlate with smaller FID scores (as can be seen in Table 4) and occasionally result in visible artifacts. Examples of such behavior are presented in Figure 4. Adding adversarial loss makes the student’s generation closer to teacher’s distribution and thus removes appearing artifacts and makes generation more realistic, in spite of occasionally resulting in bigger LPIPS or L1 losses. Additionally, we discuss three aspects of adversarial training in the appendices: the influence of loss selection in Appendix B.2, the sensitivity to the adversarial loss weight in Appendix B.5 and the impact of the training itself on mode collapse in Appendix B.6.

![](images/c201e8ccd6aaa27d10d95b3f5dd5358d06499b59ad1798843930838a121b27ab.jpg)

<details>
<summary>natural_image</summary>

Grid of nature and animal photos under two conditions: 'w/ Adv' and 'w/o Adv', showing various animals including ducks, peas, and a dog with measurement scales.
</details>

Table 4: Results of 10k training iterations calculated on 1000 validation samples.

Figure 4: Incorporating an adversarial loss into the training process enhances generation quality, reducing occurring image artifacts in low NFE regimes. In this setup, the teacher model uses UniPC (3M) solver with NFE=10, while the student models operate with a reduced NFE=4. 

<table><tr><td></td><td>FID</td><td> $\mathcal{L}_{distill}$ </td></tr><tr><td colspan="3">FFHQ</td></tr><tr><td>GS</td><td>10.70</td><td>0.116</td></tr><tr><td>GAS</td><td>7.86</td><td>0.127</td></tr><tr><td colspan="3">LSUN</td></tr><tr><td>GS</td><td>9.64</td><td>0.172</td></tr><tr><td>GAS</td><td>7.54</td><td>0.174</td></tr></table>

# 4.3 METHOD EFFICIENCY

We next show that GAS is efficient in terms of dataset size and training time.

Dataset size By the dataset we mean the set of samples generated from the teacher model to use in the training process. To this end, we measure method’s performance on the "full" dataset scenario with 49000 samples and find the smaller dataset size that demonstrates equivalent results. First, we observe that the dataset size of 1400 is enough for training GS without adversarial loss. However, the solver’s optimization problem becomes more challenging in low-NFE scenarios with adversarial loss. Here, we expand the dataset from 1400 samples to 5000 and obtain results indistinguishable from the full-dataset scenario in all datasets and settings. Additional information is provided in Appendix C.1.

Performance Without adversarial training, GS converges within 1-2.5 hours depending on the dataset, which is comparable to the most relevant baselines LD3 and S4S. In case of GAS, training time increases to 2-9 hours, which is larger, but still requires similar order. We refer the reader to the Appendix C.2 for the exact comparison of metrics depending on training time and Appendix C.3 for peak-memory usage in the backward pass.

# 5 DISCUSSION

In this paper, we propose Generalized Adversarial Solver, the novel parameterization and training algorithm for automatic gradient-based solver optimization. The main novelty is additive theoretical guidance of solver coefficients and combination of distillation loss with adversarial training. We establish that the introduced Generalized Solver parameterization significantly outperforms existing parameterizations. We show that adding the adversarial loss significantly boosts method’s performance and allows tackling the image artifacts present in simple solver distillation. We extensively compare our method with other solver/timestep training approaches and demonstrate its superior performance on 6 datasets, ranging from 32 × 32 pixel-space CIFAR10 to 256 × 256 latent-space ImageNet and 512 × 512 MS-COCO with Stable Diffusion.

Limitations Our method relies on performing backpropagation through the whole solver inference, which may face scalability issues when applied to larger image sizes and bigger models. We explore the generalizability of our method between different datasets in Section B.4. However, a potential concern remains as to whether GS/GAS requires separate training for each preferred inference NFE. We leave the development of lightweight modifications to our method for future work.

The weights of the pretrained diffusion model remain frozen during solver training. We compare GS and GAS with distillation-based methods and show in Appendix B.7 that our approach better preserves the generative capabilities of the original model. However, due to the limited number of trainable parameters at NFE=1 and 2, our method yields lower quality in these specific settings; perfomance GS and GAS at low NFE is provided in Appendix B.8.

# REPRODUCIBILITY STATEMENT

To ensure the clarity and reproducibility of our work, we provide excessive description of all parts of our method. Appendix D provides the pseudocode of our algorithm, exactly matching the way it appears in our implementation; configurations and hyperparameters of all "teacher" generations and "student" training processes, including batch sizes, optimizer choice and other fine-grained details; and expressions for commonly used timestep schedules mentioned in the paper.

Furthermore, our experiments are built upon publicly available datasets (e.g., CIFAR10, FFHQ) and pre-trained model checkpoints to ensure our experimental setups are accessible and verifiable.

# ACKNOWLEDGMENTS

Aleksandr Oganov wishes to express sincere gratitude to his alma mater, Lomonosov Moscow State University, for providing the foundational education and stimulating environment that made this research possible. Mishan Aliev thanks Yandex Education for supporting him during this research.

We thank Dmitry Baranchuk for fruitful discussions and valuable advice on knowledge distillation techniques. This research was supported in part through computational resources of HPC facilities at HSE University. A special thanks to Maxim Kodryan — his mere existence was contribution enough. The work was supported by the grant for research centers in the field of AI provided by the Ministry of Economic Development of the Russian Federation in accordance with the agreement 000000C313925P4E0002 and the agreement with HSE University №139-15-2025-009. We are thankful for the ICLR reviewers for their detailed suggestions that led to a significant improvement of this paper.

# REFERENCES

Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In International conference on machine learning, pp. 214–223. PMLR, 2017.   
Hicham Badri and Appu Shaji. Half-quadratic quantization of large machine learning models, November 2023. URL https://mobiusml.github.io/hqq\_blog/.   
Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.   
Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. 2024. URL https://openai. com/research/video-generation-models-as-world-simulators, 3:1, 2024.   
Thibault Castells, Hyoung-Kyu Song, Bo-Kyeong Kim, and Shinkook Choi. Ld-pruner: Efficient pruning of latent diffusion models using task-agnostic insights, 2024. URL https://arxiv. org/abs/2404.11936.   
Defang Chen, Zhenyu Zhou, Can Wang, Chunhua Shen, and Siwei Lyu. On the trajectory regularity of ode-based diffusion sampling. arXiv preprint arXiv:2405.11326, 2024a.   
Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7310–7320, 2024b.   
Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8188–8197, 2020.   
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.   
Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.   
Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural pruning for diffusion models, 2023. URL https://arxiv.org/abs/2305.10924.   
Eric Frankel, Sitan Chen, Jerry Li, Pang Wei Koh, Lillian J Ratliff, and Sewoong Oh. S4s: Solving for a diffusion model solver. arXiv preprint arXiv:2502.17423, 2025.   
Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.   
Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Boot: Data-free distillation of denoising diffusion models with bootstrapping. In ICML 2023 Workshop on Structured Probabilistic Inference {\&} Generative Modeling, 2023.

Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10696–10706, 2022.   
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. Advances in neural information processing systems, 30, 2017.   
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.   
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.   
Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
Marlis Hochbruck and Alexander Ostermann. Exponential integrators. Acta Numerica, 19:209–286, 2010.   
Nick Huang, Aaron Gokaslan, Volodymyr Kuleshov, and James Tompkin. The gan is dead; long live the gan! a modern gan baseline. Advances in Neural Information Processing Systems, 37: 44177–44215, 2024.   
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134, 2017.   
Alexia Jolicoeur-Martineau. The relativistic discriminator: a key element missing from standard gan. arXiv preprint arXiv:1807.00734, 2018.   
Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4401–4410, 2019.   
Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.   
Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023.   
Sanghwan Kim, Hao Tang, and Fisher Yu. Distilling ode solvers of diffusion models into smaller steps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9410–9419, 2024.   
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.   
Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photo-realistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4681–4690, 2017.   
Jae Hyun Lim and Jong Chul Ye. Geometric gan. arXiv preprint arXiv:1705.02894, 2017.   
Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft coco: Common objects in context, 2015. URL https://arxiv.org/abs/1405.0312.

Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations, 2022a. URL https: //openreview.net/forum?id=PlKWVd2yBkY.   
Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022b.   
Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023.   
Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022a.   
Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022b.   
Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diffinstruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36, 2024.   
Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15762–15772, 2024.   
Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, pp. 2794–2802, 2017.   
Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, and Jaejun Yoo. Reliable fidelity and diversity metrics for generative models. In International conference on machine learning, pp. 7176–7185. PMLR, 2020.   
Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. arXiv preprint arXiv:2312.05239, 2023.   
Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.   
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization. Advances in neural information processing systems, 29, 2016.   
Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.   
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.   
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015.   
Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your steps: Optimizing sampling schedules in diffusion models. arXiv preprint arXiv:2404.14507, 2024.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.   
Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.   
Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hoogeboom. Multistep distillation of diffusion models via moment matching. arXiv preprint arXiv:2406.04103, 2024.   
Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.   
Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–11, 2024.   
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. pmlr, 2015.   
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020a.   
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.   
Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.   
Ruoyu Sun, Tiantian Fang, and Alexander Schwing. Towards a better global loss landscape of gans. Advances in Neural Information Processing Systems, 33:10186–10198, 2020.   
Vinh Tong, Trung-Dung Hoang, Anji Liu, Guy Van den Broeck, and Mathias Niepert. Learning to discretize denoising diffusion odes. arXiv preprint arXiv:2405.15506, 2024.   
Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23:1661–1674, 2011. URL https://api.semanticscholar.org/CorpusID: 5560643.   
Fu-Yun Wang, Ling Yang, Zhaoyang Huang, Mengdi Wang, and Hongsheng Li. Rectified diffusion: Straightness is not your need in rectified flow. arXiv preprint arXiv:2410.07303, 2024.   
Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36:8406–8441, 2023.   
Daniel Watson, William Chan, Jonathan Ho, and Mohammad Norouzi. Learning fast samplers for diffusion models by differentiating through sample quality. In International Conference on Learning Representations, 2021.   
Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, et al. Cache me if you can: Accelerating diffusion models through block caching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6211–6220, 2024.   
Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion gans. arXiv preprint arXiv:2112.07804, 2021.   
Shuchen Xue, Zhaoqiang Liu, Fei Chen, Shifeng Zhang, Tianyang Hu, Enze Xie, and Zhenguo Li. Accelerating diffusion sampling with optimized time steps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8292–8301, 2024.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. arXiv preprint arXiv:2311.18828, 2023.   
Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024.   
Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.   
Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022.   
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.   
Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictorcorrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36, 2024.   
Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36: 55502–55542, 2023.   
Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, march 2024. URL https://github. com/hpcaitech/Open-Sora, 1(3):4, 2024.   
Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024a.   
Zhenyu Zhou, Defang Chen, Can Wang, and Chun Chen. Fast ode-based sampling for diffusion models in around 5 steps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7777–7786, 2024b.

# A RELATED WORK

Among many inference-time acceleration algorithms, solver-based methods treat diffusion models as ODEs with a (partially) black-box velocity function. Specifically, PNDM (Liu et al., 2022a) and iPNDM (Zhang & Chen, 2022) apply the linear multistep method to the corresponding PF-ODE. DPM-Solver (Lu et al., 2022a), DEIS (Zhang & Chen, 2022) use the variation of constants (Equation 4) and approximate the underlying integral. DPM-Solver++ (Lu et al., 2022b) extends this idea to the multi-step version, and UniPC (Zhao et al., 2024) modifies it with the predictor-corrector framework. Besides the solver distillation loss, introduced for optimizing the timesteps in LD3 (Tong et al., 2024) and used for optimizing both timesteps and solver coefficients in S4S (Frankel et al., 2025), many automatic solver selection methods were proposed. DDSS (Watson et al., 2021) directly optimizes generation quality of the solver. AYS (Sabour et al., 2024) optimizes timesteps to minimize the KL divergence between the backward SDE and the discretization. GITS (Chen et al., 2024a) choose the timesteps by utilizing trajectory structure of the PF-ODE and DMN (Xue et al., 2024) allows for the fast model-free choice of parameters via optimizing an upper-bound on the solution error. Some approaches manipulate diffusion-specific properties and utilize redundancies in their computations. Namely, DeepCache (Ma et al., 2024) and CacheMe (Wimbauer et al., 2024) propose to perform block or layer caching and reuse activations from the previous timesteps. The other directions of acceleration include quantization (Gu et al., 2022; Badri & Shaji, 2023) and pruning (Fang et al., 2023; Castells et al., 2024).

In contrast, diffusion distillation techniques aim at compressing a pre-defined diffusion model by training a few-step student. Several methods learn to mimic solution of the PF-ODE. This includes optimizing the regression loss between the outputs (Salimans & Ho, 2022) or learning the integrator between arbitrary timesteps (Gu et al., 2023; Song et al., 2023; Kim et al., 2023). Others use diffusion models as a training signal that assesses likelihood of the generated images. It is commonly formalized as optimizing the Integrated KL divergence (Luo et al., 2024; Yin et al., 2023; 2024; Nguyen & Tran, 2023) by training an additional "fake" diffusion model on the generator’s output distribution. Other methods consider matching scores (Zhou et al., 2024a) or moments (Salimans et al., 2024) of the corresponding distributions. Many distillation methods enhance student generation quality by adding the adversarial training (Kim et al., 2023; Yin et al., 2024), including discriminator loss on detector (Sauer et al., 2023) or teacher features (Sauer et al., 2024).

# B ADDITIONAL EXPERIMENTS

# B.1 THEORETICAL GUIDANCE FOR THE COEFFICIENTS

The integration of theoretical guidance for the solver coefficients is important for the effective training of GS. It provides a strong inductive bias by embedding knowledge of the theoretically optimal coefficients directly into GS. Thus, in the optimization process, the trained coefficients do not have to learn complex theoretical dependencies, since they are already embedded in the theoretical coefficients.

To empirically validate the contribution, we conducted the ablation by training a version of GS on the FFHQ dataset both with and without this guidance. In both configurations we use hyperparameters from D.4.1, the coefficients were initialized to be equivalent to those of DPM-Solver++(3M). The results, presented in Table 5, demonstrate that the inclusion of theoretical guidance yields a substantial improvement in FID. Furthermore, the flexible signature of GS enables the generalization of various modern theoretical solvers, opening up new avenues for research into different forms of theoretical guidance.

# B.2 ADVERSARIAL LOSS

To better understand the impact of adversarial loss, we compared different GAN losses on ImageNet at NFE=4. We compared the standard GAS, which uses the relativistic loss from Equation 10, with one trained using the traditional loss from Equation 9. In both configurations, we used discriminator gradient penalties and the same hyperparameters from Appendix D.4.2.

Table 5: GS with and without theoretical guidance on FFHQ dataset in terms of FID. Both setups use same hyperparameters and the coefficients were initialized to be equivalent to those of DPM-Solver++(3M). 

<table><tr><td>Parameterization</td><td>NFE=4</td><td>NFE=6</td><td>NFE=8</td><td>NFE=10</td></tr><tr><td>w/ theory</td><td>10.70</td><td>4.49</td><td>2.96</td><td>2.67</td></tr><tr><td>w/o theory</td><td>15.23</td><td>10.53</td><td>5.50</td><td>4.69</td></tr></table>

Table 6: GAS with traditional and relativistic GAN losses on ImageNet dataset at NFE=4 in terms of FID. 

<table><tr><td>GS</td><td>GAS (traditional)</td><td>GAS (relativistic)</td></tr><tr><td>7.87</td><td>6.49</td><td>5.38</td></tr></table>

From the comparison results presented in Table 6, we conclude that employing a GAN loss enhances the final output quality. GAS demonstrates strong performance with both traditional and relativistic losses. Although the relativistic loss is optional, it leads to a superior model.

# B.3 FID PROGRESSION DURING TRAINING

To better understand the training process, we visualize the dynamics of the FID score during the training process.

When comparing the GS and GAS FID scores for FFHQ, as visualized in Figure 5a, we observe that incorporating the adversarial objective requires more training iterations for our method to converge. However, it is more important that, as previously reported in Table 2a, it achieves a significantly lower FID score, allowing for a better trade-off between generation quality and a slight increase in training time.

Figure 5b demonstrates that although GAS achieves excellent FID scores after 30k iterations, it could potentially yield even better results with further training. This is suggested by the continuing decrease in the FID score for NFE of 4 and 5 with each additional training iteration. Scenarios involving a larger number of NFE for model inference do not display this pattern, since they comprise a bigger student’s capacity and lead to easier optimization task and earlier convergence.

![](images/951ad3ad6923945310472a08b9a63dcd42385cc09925694d001b17a21e99882c.jpg)

<details>
<summary>line</summary>

| Training iteration | GS   | GAS  |
| ------------------ | ---- | ---- |
| 0                  | 50   | 50   |
| 1000               | 15   | 20   |
| 2000               | 13   | 12   |
| 3000               | 13   | 11   |
| 4000               | 13   | 11   |
| 5000               | 13   | 11   |
| 6000               | 13   | 11   |
| 7000               | 13   | 11   |
| 8000               | 13   | 11   |
| 9000               | 13   | 11   |
| 10000              | 13   | 11   |
</details>

(a) FID values during training for the FFHQ dataset, using 4 NFE with the GS and GAS. We evaluate FID every 500 training iterations, computing it based on 5000 generated samples.

![](images/7a8ced8f342cb92f409ca4285e84a1912e0ff6d731ffc8eb04f5925ad42f33e5.jpg)

<details>
<summary>line</summary>

| Training iterations | LSUN-Bedrooms-256 NFE=4 | LSUN-Bedrooms-256 NFE=5 | LSUN-Bedrooms-256 NFE=6 | LSUN-Bedrooms-256 NFE=7 | Imagenet-256 NFE=4 | Imagenet-256 NFE=5 | Imagenet-256 NFE=6 | Imagenet-256 NFE=7 |
| ------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| 5000                | 8.9                      | 5.3                      | 4.1                      | 3.7                      | 6.0                 | 5.3                 | 4.3                 | 4.2                 |
| 10000               | 7.6                      | 5.1                      | 4.0                      | 3.7                      | 5.7                 | 5.2                 | 4.3                 | 4.2                 |
| 20000               | 7.1                      | 4.8                      | 3.9                      | 3.7                      | 5.5                 | 4.9                 | 4.3                 | 4.2                 |
| 30000               | 6.9                      | 4.8                      | 3.9                      | 3.7                      | 5.4                 | 4.9                 | 4.3                 | 4.2                 |
</details>

(b) GAS FID training dynamics for latent space datasets for several NFE scenarios. We generate 50k images for each of 5000, 10k, 20k and 30k iterations checkpoints, and evaluate FID scores based on those datasets.   
Figure 5: FID score for several checkpoints during training of GS and GAS for both pixel (FFHQ) and latent space (LSUN, ImageNet) datasets.

# B.4 GENERALIZATION ACROSS DATASETS

Regarding generalization across datasets with significantly different dimensionalities (e.g., CIFAR vs. COCO), the optimal schedule for a smaller resolution may not be optimal for higher resolutions due to simpler denoising tasks at equivalent noise levels (larger images have greater correlation among nearby pixels). To further demonstrate the method’s generalization results, we tested solver transfer between closely related diffusion models (FFHQ and AFHQv2), demonstrating practical generalizability. We thus illustrate its generalization in Table 7.

Table 7: We evaluate FID score comparison of GS and GAS trained on dataset and applied on another against DPM-Solver++, LD3 and S4S. We use the GS and GAS checkpoints as in Table 2a.   
(a) Solvers GS, GAS, trained on FFHQ and applied on AFHQv2 (denoted as GS’ and GAS’) consistently outperform baseline methods. 

<table><tr><td>Method</td><td>NFE=4</td><td>NFE=6</td><td>NFE=8</td><td>NFE=10</td></tr><tr><td>DPM-Solver++</td><td>27.82</td><td>10.72</td><td>4.28</td><td>3.19</td></tr><tr><td>Best LD3</td><td>9.96</td><td>3.63</td><td>2.63</td><td>2.27</td></tr><tr><td>S4S Alt</td><td>6.52</td><td>2.70</td><td>2.29</td><td>2.18</td></tr><tr><td>GS (Ours)</td><td>5.92</td><td>2.87</td><td>2.33</td><td>2.25</td></tr><tr><td>GAS (Ours)</td><td>4.48</td><td>2.66</td><td>2.29</td><td>2.31</td></tr><tr><td>GS&#x27; (Ours)</td><td>6.54</td><td>3.01</td><td>2.41</td><td>2.29</td></tr><tr><td>GAS&#x27; (Ours)</td><td>5.15</td><td>2.81</td><td>2.44</td><td>2.32</td></tr></table>

(b) Solvers GS, GAS, trained on AFHQv2 and applied on FFHQ (denoted as GS’ and GAS’) consistently outperform baseline methods. 

<table><tr><td>Method</td><td>NFE=4</td><td>NFE=6</td><td>NFE=8</td><td>NFE=10</td></tr><tr><td>DPM-Solver++</td><td>46.14</td><td>14.01</td><td>6.18</td><td>4.18</td></tr><tr><td>Best LD3</td><td>17.96</td><td>5.97</td><td>3.50</td><td>3.25</td></tr><tr><td>S4S Alt</td><td>10.63</td><td>4.62</td><td>3.15</td><td>2.91</td></tr><tr><td>GS (Ours)</td><td>10.70</td><td>4.49</td><td>2.96</td><td>2.67</td></tr><tr><td>GAS (Ours)</td><td>7.86</td><td>3.79</td><td>2.87</td><td>2.66</td></tr><tr><td>GS&#x27; (Ours)</td><td>16.01</td><td>5.91</td><td>3.27</td><td>2.70</td></tr><tr><td>GAS&#x27; (Ours)</td><td>9.39</td><td>4.21</td><td>2.92</td><td>2.72</td></tr></table>

# B.5 ADVERSARIAL LOSS WEIGHT

One of the few hyperparameters of GAS is the GAN-weight. Starting from the resolution of $6 4 \times 6 4$ , the weight of the adversarial loss was fixed to 1.0 for all datasets. Figure 6 demonstrates that GAS is insensitive to the GAN-weight selection and achieves similar FID with different weights. This shows that our method achieves strong results without the need for hyperparameter tuning.

![](images/e4ec6af92afd99448e10c625ec521df09f21c67243a53b3224ef58c6515f386c.jpg)

<details>
<summary>line</summary>

| L_adv weight | FID   |
| ------------ | ----- |
| 0            | 10.6  |
| 0.2          | 8.9   |
| 0.3          | 8.3   |
| 0.5          | 8.0   |
| 0.7          | 7.9   |
| 1            | 7.9   |
| 1.2          | 7.9   |
| 1.5          | 7.9   |
| 2            | 7.8   |
| 5            | 7.9   |
| 10           | 7.9   |
| 100          | 7.8   |
| 500          | 7.9   |
</details>

Figure 6: FID values for FFHQ dataset with 4 NFE for different adversarial loss weights. The metric remains stable even at large weight values. Setting $\mathcal { L } _ { \mathrm { a d v } } = 0$ for GAS results in absence of adversarial training, thus is a GS setting.

# B.6 MODE COLLAPSE

To demonstrate that GAS maintains diversity, we computed precision, recall, density, coverage from Naeem et al. (2020) for GS and GAS, comparing their statistics to those of the teacher model. Evaluation results were obtained by comparing 50000 generated samples to 50000 teacher images using T4096(Naeem et al. (2020), Section 4.2). We report results on the ImageNet dataset with NFE=4 in Table 8. We choose this setup because the incorporation of the adversarial loss was significant in this experiment and could raise the greatest concerns regarding mode-collapse. As shown in Table 8, GAS does not suffer from the aforementioned problem and even increases mode coverage and recall compared to GS, which is trained without adversarial loss. Additionally, we provide random samples in Appendix F.

Table 8: Mode collapse ablation of the GAS on ImageNet dataset. 

<table><tr><td>Method</td><td>FID</td><td>precision</td><td>recall</td><td>density</td><td>coverage</td></tr><tr><td>GS</td><td>7.87</td><td>0.90</td><td>0.63</td><td>1.18</td><td>0.90</td></tr><tr><td>GAS</td><td>5.38</td><td>0.90</td><td>0.76</td><td>1.20</td><td>0.97</td></tr></table>

Table 9: Comparison of generative properties of models on CIFAR10 dataset. 

<table><tr><td>Method</td><td>CD, NFE=1</td><td>CD, NFE=4</td><td>GS, NFE=4</td><td>GAS, NFE=4</td><td>Our teacher</td></tr><tr><td>Coverage</td><td>0.942</td><td>0.938</td><td>0.963</td><td>0.961</td><td>0.971</td></tr><tr><td>FID</td><td>3.56</td><td>2.99</td><td>4.41</td><td>4.05</td><td>2.03</td></tr></table>

# B.7 COMPARISON WITH CONSISTENCY DISTILLATION

We compare GS and GAS with distillation-based methods in terms of their ability to preserve the generative properties of the diffusion model. To evaluate diversity, we compare the coverage from Naeem et al. (2020) and FID of our methods (GS and GAS) against the official checkpoint of CD Song et al. (2023) on CIFAR10. We assess generation quality against a teacher solver, which achieves FID=2.03 and coverage=0.971. Throughout, when working with CD, we used the official implementation and ternary search for NFE=4(Song et al. (2023), Section 3). Coverage was measure between 10000 generated images and the CIFAR10 test set using T4096.

The comparison results for FID and coverage are presented in Table 9 and show that GS and GAS have higher coverage than CD. In addition, CD demonstrates low coverage compared to the teacher solver, which indicates a deterioration in the generative properties of the diffusion model. GS and GAS, without changing these parameters, provide a more flexible and property-preserving acceleration method. We compare the efficiency of GS and GAS with that of distillation-based approaches in Appendix C.2.

# B.8 PERFORMANCE AT LOW NFE

We evaluated the performance of GS and GAS at low NFE. We trained GS and GAS on the FFHQ dataset with NFE=1 and NFE=2. For training, we used the hyperparameters from Appendix D.4.1. For comparison, we used DPM-Solver with order set to NFE when NFE < 4, and order=3 when NFE=4. Table 11 shows that GS and GAS improve upon DPM-Solver, but they do not perform well at low NFE as the FID scores are too high.

# B.9 COEFFICIENTS PARAMETRIZATION

In addition to the coefficient parameterization ablation in Section 4.2, we compared GS with the S4S LMS and LMS+PC parameterizations under the same hyperparameters and within our codebase (S4S does not provide an official implementation). In Table 3, we compare GS with LMS+PC when all methods are trained to convergence. Here, we further show that GS outperforms both LMS and LMS+PC on FFHQ under identical training settings from Appendix D.4.1. We initialize LMS and GS so that the corresponding solvers are initially equivalent to DPM-Solver++(3M). Table 10 shows that for all NFE values, GS achieves better generation quality.

Table 10: Comparison of S4S parameterizations with GS on FFHQ dataset in terms of FID. We use same hyperparameters, teachers and codebase. 

<table><tr><td>Parameterization</td><td>NFE=4</td><td>NFE=6</td><td>NFE=8</td><td>NFE=10</td></tr><tr><td>S4S LMS</td><td> $\underline{17.05}$ </td><td> $\underline{5.93}$ </td><td> $\underline{4.37}$ </td><td> $\underline{3.96}$ </td></tr><tr><td>S4S LMS + PC</td><td> $\underline{45.51}$ </td><td> $\underline{24.04}$ </td><td> $\underline{7.95}$ </td><td> $\underline{4.01}$ </td></tr><tr><td>GS</td><td> $\underline{10.70}$ </td><td> $\underline{4.49}$ </td><td> $\underline{2.96}$ </td><td> $\underline{2.67}$ </td></tr></table>

Table 11: Comparison of GS and GAS with DPM-Solver++ on FFHQ dataset in terms of FID. We denoted DPM as DPM-Solver++. 

<table><tr><td></td><td>NFE=1</td><td>NFE=2</td><td>NFE=4</td></tr><tr><td>DPM</td><td>314.95</td><td>134.36</td><td>46.14</td></tr><tr><td>GS</td><td>147.54</td><td>55.98</td><td>10.70</td></tr><tr><td>GAS</td><td>193.22</td><td>60.69</td><td>7.86</td></tr></table>

# C EFFICIENCY OF THE METHOD

# C.1 TRAINING DATASET SIZE

We conduct experiments to assess the efficiency of the proposed methods with respect to the size of the training dataset. We examine several variations of sizes: 49000 as a baseline, 5000 and 1400 as the more lightweight alternatives. For GS, we observe that taking 1400 images and performing 10000 training iterations is sufficient for our method to converge, regardless of NFE. We note that it reaches equivalent or better FID scores compared to a bigger training dataset (see Table 12a).

The same pattern occurs with GAS on CIFAR10. The dataset of 1400 images is optimal for its training. However, starting from the higher-dimensional FFHQ dataset, we observe the typical challenges of adversarial training. As the discriminator used in GAS is trained simultaneously with the other parameters of the solver, it tends to overfit and demands larger dataset size to alleviate this problem.

Adversarial training has demonstrated its effectiveness, especially in scenarios with smaller inference steps. We thus illustrate its performance in Table 12b on NFE = 4 and NFE = 6. It shows that the training dataset size of 5000 is sufficient for matching performance of the model trained on 49000.

Table 12: Comparison of different dataset sizes with several NFE, where N indicates the number of samples in the training dataset. In Table 12a the FID score is calculated after 10k and 20k training iterations to show the early convergence of the GS method. Table 12b presents the results of GAS evaluation after 10k iterations of training.   
(a) Generalized solver 

<table><tr><td></td><td></td><td colspan="2">NFE=4</td><td colspan="2">NFE=10</td></tr><tr><td></td><td>N</td><td>10k</td><td>20k</td><td>10k</td><td>20k</td></tr><tr><td rowspan="2">CIFAR10</td><td>1400</td><td>4.35</td><td>4.35</td><td>2.14</td><td>2.15</td></tr><tr><td>49000</td><td>4.39</td><td>4.39</td><td>2.15</td><td>2.15</td></tr><tr><td rowspan="2">FFHQ</td><td>1400</td><td>10.70</td><td>10.72</td><td>2.71</td><td>2.71</td></tr><tr><td>49000</td><td>10.79</td><td>10.82</td><td>2.70</td><td>2.71</td></tr></table>

(b) Generalized Adversarial solver 

<table><tr><td></td><td>N</td><td>NFE=4</td><td>NFE=6</td></tr><tr><td rowspan="2">CIFAR10</td><td>1400</td><td>3.98</td><td>2.44</td></tr><tr><td>49000</td><td>3.98</td><td>2.48</td></tr><tr><td rowspan="3">FFHQ</td><td>1400</td><td>9.44</td><td>4.48</td></tr><tr><td>5000</td><td>7.83</td><td>3.79</td></tr><tr><td>49000</td><td>7.93</td><td>3.76</td></tr></table>

# C.2 TRAINING TIME

We further investigate GS/GAS training dynamics by estimating their convergence time and comparing their computational efficiency with other methods.

In Table 13a we demonstrate the training time of Progressive Distillation (PD, (Salimans & Ho, 2022)) and Consistency Distillation (CD, (Song et al., 2023)). Those methods focus on training a new generator model that can sample images in a few-NFE manner. Both require days of training time and are computationally demanding.

We also compare our methods with several approaches that involve training certain parameters of solvers. In pixel space GS requires less than an hour of training time on CIFAR10, which is comparable to LD3, S4S and S4S-Alt. Notably, it achieves FID of 2.44 with NFE = 6, while S4S-Alt results in FID score of 2.52 with NFE = 7 and equivalent training time. Adversarial loss extends the training time to up to 2 hours, however, as we report in Table 2a, it achieves superior results in terms of FID score.

In the latent diffusion setting, we compare our method with LD3, which reports convergence within an hour of training time. We observe that GS and GAS require up to 3 hours; however, this is still within the same order (for more details, see Table 13b).

In Table 14 we also provide more details about training time of our methods for both pixel and latent space models.

# C.3 MEMORY USAGE

We are investigating the peak-memory GS/GAS required for training iteration depending on NFE.

In Table 15 we demonstrate the peak-memory usage for GS/GAS compared to LD3. When measuring the memory, we used the config we further report in Appendix D. GS requires the same amount of peak-memory allocated as LD3.

Incorporation of the discriminator loss into the training process of GAS only requires additional less than 4 gigabyte of memory usage, which is a minor overhead, especially considering its efficiency in terms of the final generation quality. This overhead is limited to training at inference time, GAS and GS sample at the same speed. Additionally, storing prior states does not incur additional overhead for peak-memory usage.

Table 13: Comparison of different training-based methods in terms of computational effectiveness across both pixel and latent space selected dataset.   
(a) CIFAR10 

<table><tr><td>Method</td><td>NFE</td><td>FID</td><td>Time</td><td>GPU Type</td></tr><tr><td>CD</td><td>2</td><td>2.93</td><td>8 days</td><td>A100</td></tr><tr><td>PD</td><td>8</td><td>2.47</td><td>8 days</td><td>TPU</td></tr><tr><td>S4S-Alt</td><td>7</td><td>2.52</td><td>&lt; 1 hour</td><td>A100</td></tr><tr><td>S4S</td><td>10</td><td>2.18</td><td>&lt; 1 hour</td><td>A100</td></tr><tr><td>LD3</td><td>10</td><td>2.32</td><td>&lt; 1 hour</td><td>A100</td></tr><tr><td rowspan="2">GS</td><td>6</td><td>2.44</td><td>&lt; 1 hour</td><td>H100</td></tr><tr><td>10</td><td>2.14</td><td>&lt; 1 hour</td><td>H100</td></tr><tr><td>GAS</td><td>4</td><td>3.98</td><td>&lt; 2 hours</td><td>H100</td></tr></table>

(b) Imagenet-256 

<table><tr><td>Method</td><td>NFE</td><td>FID</td><td>Time</td><td>GPU Type</td></tr><tr><td rowspan="4">LD3</td><td>4</td><td>9.19</td><td rowspan="4">&lt; 1 hour</td><td rowspan="4">A100</td></tr><tr><td>5</td><td>5.03</td></tr><tr><td>6</td><td>4.46</td></tr><tr><td>7</td><td>4.32</td></tr><tr><td rowspan="4">GS</td><td>4</td><td>7.97</td><td>&lt; 1.5 hours</td><td rowspan="4">H100</td></tr><tr><td>5</td><td>4.94</td><td>&lt; 2 hours</td></tr><tr><td>6</td><td>4.29</td><td>&lt; 2 hours</td></tr><tr><td>7</td><td>4.16</td><td>&lt; 2.5 hours</td></tr><tr><td>GAS</td><td>4</td><td>6.06</td><td>&lt; 3 hours</td><td>H100</td></tr></table>

Table 14: Approximate training time (in minutes) for 10k iterations scenarios for GS and GAS in both pixel and latent space. For MS-COCO we use 1k iterations scenarios. All the numbers reported are computed using one H100 GPU.   
(a) Pixel space models 

<table><tr><td colspan="2"></td><td>NFE=4</td><td>NFE=6</td><td>NFE=8</td><td>NFE=10</td></tr><tr><td rowspan="3">GS</td><td>CIFAR10</td><td>30m</td><td>40m</td><td>50m</td><td>60m</td></tr><tr><td>FFHQ</td><td>40m</td><td>60m</td><td>80m</td><td>95m</td></tr><tr><td>AFHQv2</td><td>40m</td><td>60m</td><td>80m</td><td>95m</td></tr><tr><td rowspan="3">GAS</td><td>CIFAR10</td><td>85m</td><td>100m</td><td>115m</td><td>130m</td></tr><tr><td>FFHQ</td><td>160m</td><td>185m</td><td>210m</td><td>240m</td></tr><tr><td>AFHQv2</td><td>160m</td><td>185m</td><td>210m</td><td>240m</td></tr></table>

(b) Latent space models 

<table><tr><td></td><td></td><td>NFE=4</td><td>NFE=5</td><td>NFE=6</td><td>NFE=7</td></tr><tr><td rowspan="3">GS</td><td>LSUN</td><td>35m</td><td>45m</td><td>50m</td><td>60m</td></tr><tr><td>ImageNet</td><td>75m</td><td>95m</td><td>115m</td><td>135m</td></tr><tr><td>MS-COCO</td><td>50m</td><td>60m</td><td>70m</td><td>80m</td></tr><tr><td rowspan="3">GAS</td><td>LSUN</td><td>125m</td><td>140m</td><td>150m</td><td>165m</td></tr><tr><td>ImageNet</td><td>185m</td><td>210m</td><td>245m</td><td>270m</td></tr><tr><td>MS-COCO</td><td>60m</td><td>75m</td><td>90m</td><td>105m</td></tr></table>

Table 15: Peak-memory usage (in gigabyte) for training iteration for GS and GAS in CIFAR10 and Imagenet-256. We use LD3 in our implementation. The official implementation uses LPIPS, rather than L1 distance in latent space as we do, which leads to the use of a VAE decoder at each step and incurs additional memory usage.   
(a) CIFAR10 

<table><tr><td></td><td>NFE=4</td><td>NFE=6</td><td>NFE=8</td><td>NFE=10</td></tr><tr><td>GS</td><td>17GB</td><td>23GB</td><td>28GB</td><td>34GB</td></tr><tr><td>GAS</td><td>19GB</td><td>25GB</td><td>30GB</td><td>35GB</td></tr><tr><td>LD3</td><td>17GB</td><td>23GB</td><td>28GB</td><td>34GB</td></tr></table>

(b) Imagenet-256 

<table><tr><td></td><td>NFE=4</td><td>NFE=5</td><td>NFE=6</td><td>NFE=7</td></tr><tr><td>GS</td><td>37GB</td><td>45GB</td><td>54GB</td><td>62GB</td></tr><tr><td>GAS</td><td>41GB</td><td>49GB</td><td>57GB</td><td>66GB</td></tr><tr><td>LD3</td><td>37GB</td><td>45GB</td><td>54GB</td><td>62GB</td></tr></table>

# C.4 INFERENCE TIME

Inference process of our method requires additional operations performed with all prior states. However, they are incomparably computationally simpler than one step of diffusion model (function evaluation). Thus, the wall-clock time of inference for GS is comparable to the solver baselines, which we show in Table 16.

Table 16: Inference time in minutes for ImageNet dataset. We obtain the comparison by generating 1,024 images with batch 64 utilizing a single H100 GPU. GAS differs from GS only in the training process; their inference times are identical.

<table><tr><td>Method</td><td>NFE=4</td><td>NFE=5</td><td>NFE=6</td><td>NFE=7</td></tr><tr><td>UniPC(3M)</td><td>0.36m</td><td>0.46m</td><td>0.55m</td><td>0.64m</td></tr><tr><td>GS (Ours)</td><td>0.36m</td><td>0.45m</td><td>0.55m</td><td>0.64m</td></tr></table>

This pattern does not depend on the model and dataset choice; therefore, our method does not introduce any inference time overhead on both pixel, latent or text-to-image diffusion models.

# D EXPERIMENTAL DETAILS

# D.1 BASELINE DISCRETIZATION HEURISTICS

In this section, we provide the reader with the common timestep schedules, used in the paper.

Polynomial discretization (time-quadratic, time-uniform) defines the timestep schedule via a polynomial function of the uniform sequence. Specifically, it defines

$$
t _ {i} = \left(\frac {i}{N}\right) ^ {\rho} (T - t _ {\mathrm{eps}}) + t _ {\mathrm{eps}}, \quad i = 0, 1, \dots , N. \tag {23}
$$

Here ρ is often set to 1 or 2 (Song et al., 2020b; Ho et al., 2020; Song et al., 2020a) which corresponds to time quadratic and time uniform discretization.

Time logSNR schedule builds on top of the signal-to-noise ratio $\alpha _ { t } ^ { 2 } / \sigma _ { t } ^ { 2 }$ . Specifically, log-SNR uses the transformation $\lambda _ { t } = \log ( \sigma _ { t } / \alpha _ { t } )$ and defines

$$
\lambda (t _ {i}) = \frac {N - i}{N} (\lambda_ {T} - \lambda_ {\mathrm{eps}}) + \lambda_ {\mathrm{eps}}, \quad i = 0, 1, \dots , N. \tag {24}
$$

This schedule offers high generation quality with different versions of the DPM-Solver (Lu et al., 2022a;b; Zheng et al., 2023).

GITS schedule provides an optimized sequence of noise levels for diffusion models, targeting very low NFE. Originally proposed in Chen et al. (2024a) for ODE-based diffusion processes with trajectory regularity constraints. We use optimized timesteps in Stable Diffusion experiments from Tong et al. (2024). Concretely, the timestep schedules are:

$$
\begin{array}{l} \mathrm{NFE} = 4: \quad [ 1, 0. 6 8 3 7, 0. 3 6 7 3, 0. 1 1 7 6, 0. 0 0 1 ]; \\ \mathrm{NFE} = 5: \quad \left[ 1, 0. 7 6 6 9, 0. 4 8 3 9, 0. 2 3 4 1, 0. 0 6 7 6, 0. 0 0 1 \right]; \\ \mathrm{NFE} = 6: \quad \left[ 1, 0. 7 8 3 6, 0. 5 5 0 4, 0. 3 3 4 0, 0. 1 5 0 8, 0. 0 3 4 3, 0. 0 0 1 \right]; \\ \mathrm{NFE} = 7: \quad \left[ 1, 0. 8 5 0 2, 0. 6 0 0 4, 0. 4 0 0 6, 0. 2 1 7 5, 0. 0 8 4 3, 0. 0 1 7 6, 0. 0 0 1 \right]; \\ \mathrm{NFE} = 8: \quad \left[ 1, 0. 8 5 0 2, 0. 6 5 0 4, 0. 4 6 7 2, 0. 3 0 0 7, 0. 1 6 7 5, 0. 0 6 7 6, 0. 0 1 7 6, 0. 0 0 1 \right]. \\ \end{array}
$$

# D.2 TEACHER SOLVER

Data generation For a fair comparison, we follow Tong et al. (2024) to generate the teacher dataset. We choose UniPC with the parameters used in LD3. We utilize class condition of the ImageNet-256 teacher and generate the corresponding dataset with the classifier-free guidance scale of 2.0 and generate 50 images per each of the 1000 classes. We report details in Table 17.

Table 17: Detailed description of the UniPC solver parameters used for a teacher dataset generation consisting of 50000 images for both pixel and latent space scenarios. 

<table><tr><td></td><td>CIFAR10</td><td>FFHQ</td><td>AFHQv2</td><td>LSUN-Bedroom-256</td><td>Imagenet-256</td></tr><tr><td>Order</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>NFE</td><td>20</td><td>20</td><td>20</td><td>20</td><td>10</td></tr><tr><td>Time schedule</td><td>logSNR</td><td>logSNR</td><td>logSNR</td><td>time-uniform</td><td>time-quadratic</td></tr><tr><td> $B(h)$ </td><td>bh1</td><td>bh1</td><td>bh1</td><td>bh2</td><td>bh2</td></tr><tr><td> $t_{eps}$ </td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-3</td><td>1e-3</td></tr><tr><td>FID</td><td>2.03</td><td>2.60</td><td>2.16</td><td>3.06</td><td>4.10</td></tr></table>

Stable Diffusion details Regarding text-to-image generation with Stable Diffusion, we observe that output image distributions of low-NFE students (NFE = 3-5) differ significantly from those of a high-NFE teacher $( \mathrm { e . g . , N F E = 1 0 } )$ . Since such students have very few trainable parameters, direct distillation can be inefficient. The same pattern was found in Tong et al. (2024). For such reason and a fair comparison, we follow identical to the LD3 approach teacher generation protocol. We train student at ${ \mathrm { N F E } } = n$ with the teacher at ${ \mathrm { N F E } } = n + 1$ . This "one-plus" teacher minimizes the gap in noise dynamics and yields smoother, more reliable convergence.

Moreover, in our experiments, we find that FID loses its correlation with perceived fidelity at high NFE, so we treat improvements in that regime with particular caution. Recognizing this unreliability beyond $\mathrm { N F E } \approx 8$ reinforces our choice of simpler teachers as the most robust path to high-quality samples. Further details on teacher parameters are provided in Table 18.

Table 18: Detailed solver parameter settings for teacher-generated dataset using 30000 MS-COCO prompts. 

<table><tr><td>Student&#x27;s NFE</td><td>NFE=4</td><td>NFE=5</td><td>NFE=6</td><td>NFE=7</td></tr><tr><td>Teacher&#x27;s NFE</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>Solver</td><td>IPNDM(2M)</td><td>IPNDM(2M)</td><td>IPNDM(2M)</td><td>IPNDM(2M)</td></tr><tr><td>Time schedule</td><td>GITS</td><td>GITS</td><td>GITS</td><td>GITS</td></tr><tr><td>FID</td><td>14.10</td><td>12.08</td><td>11.80</td><td>11.48</td></tr></table>

# D.3 SOLVER COEFFICIENTS PARAMETERIZATION

The detailed description of the Generalized Solver step is provided in Algorithm 1. Specifically, when all parameters ϕ are set to zero, the GS reduces exactly to DPM-Solver++(3M) (Lu et al., 2022b).

# D.4 PRACTICAL IMPLEMENTATION DETAILS

We define W , H, and C as the width, height, and number of channels of an image, respectively. Similarly, $W ^ { \prime } , H ^ { \prime } ,$ , and $C ^ { \prime }$ represent the corresponding dimensions in the latent space for the Latent Diffusion model (Rombach et al., 2022).

Optimizer and trainable parameters We update three primary parameter sets during training: θ defines the timestep schedule, ϕ defines the solver coefficients and ξ acts as a correction to the timesteps that we evaluate the pre-trained model on. We use one optimizer for all parameter groups. We use time-uniform schedule for the initialization of parameters θ. We initialize $\xi$ and $\hat { c } _ { j , n } ( \phi )$ , $a _ { j , n } ( \phi )$ with zeros. We use the EMA version of the model parameters for evaluation and update the EMA weights after each training iteration.

Evaluation We evaluate our models (Table 2a, 2b) using the FID score with 50 000 randomly generated samples. For ImageNet, we generate an equal number of samples for each class to ensure a balanced FID evaluation. We use EMA weights for evaluations. We calculate FID using reference statistics and code from Karras et al. (2022). For MS-COCO (Table 2c) we obtain the FID score on 30 000 images using the same validation captions and FID reference statistics as in LD3 (Tong et al., 2024).

Algorithm 1 Generalized solver (GS) with theoretical guidance from DPM-Solver++(3M). Denote $h _ { i } = \lambda _ { t _ { i } ^ { \theta } } - \lambda _ { t _ { i - } ^ { \theta } }$ for $i = 1 , \ldots , N$ .   
1: $\psi_{1}\leftarrow e^{-h_{n}}-1$ 2: $\mathbf{x}_{n+1}\leftarrow\left[a_{n,n}(t_{n:n+1}^{\theta})+\hat{a}_{n,n}(\phi)\right]\cdot\mathbf{x}_{n}-\left[\alpha_{t_{n+1}}\phi_{1}+\hat{c}_{n,n}(\phi)\right]\cdot\boldsymbol{v}(\mathbf{x}_{n},t_{n}^{\theta}+\xi_{n})$ 3: if n=1 then
4: $r_{0}\leftarrow\frac{h_{n-1}}{h_{n}}$ 5: $\mathbf{D1}_{0}\leftarrow\frac{1}{r_{0}}\left[\boldsymbol{v}(\mathbf{x}_{n},t_{n}^{\theta}+\xi_{n})-\boldsymbol{v}(\mathbf{x}_{n-1},t_{n-1}^{\theta}+\xi_{n-1})\right]$ 6: $\mathbf{x}_{n+1}\leftarrow\mathbf{x}_{n+1}-\left[\frac{\alpha_{t_{n+1}}\phi_{1}}{2}+\hat{c}_{n-1,n}(\phi)\right]\cdot\mathbf{D1}_{0}$ 7: else if $n\geq2$ then
8: $r_{0}, r_{1}\leftarrow\frac{h_{n-1}}{h_{n}},\frac{h_{n-2}}{h_{n}}$ 9: $\psi_{2}\leftarrow\frac{\psi_{1}}{h}+1$ 10: $\psi_{3}\leftarrow\frac{\psi_{2}}{h}-\frac{1}{2}$ 11: $\mathbf{D1}_{0}\leftarrow\frac{1}{r_{0}}\left[\boldsymbol{v}(\mathbf{x}_{n},t_{n}^{\theta}+\xi_{n})-\boldsymbol{v}(\mathbf{x}_{n-1},t_{n-1}^{\theta}+\xi_{n-1})\right]$ 12: $\mathbf{D1}_{1}\leftarrow\frac{1}{r_{1}}\left[\boldsymbol{v}(\mathbf{x}_{n-1},t_{n-1}^{\theta}+\xi_{n-1})-\boldsymbol{v}(\mathbf{x}_{n-2},t_{n-2}^{\theta}+\xi_{n-2})\right]$ 13: $D1\leftarrow D1_{0}+\frac{r_{0}}{r_{0}+r_{1}}[D1_{0}-D1_{1}]$ 14: $D2\leftarrow\frac{1}{r_{0}+r_{1}}[D1_{0}-D1_{1}]$ 15: $x_{n+1}\leftarrow x_{n+1}+\left[\alpha_{t_{n+1}}\phi_{2}+\hat{c}_{n-1,n}(\phi)\right]\cdot D1-\left[\alpha_{t_{n+1}}\phi_{3}+\hat{c}_{n-2,n}(\phi)\right]\cdot D2$ 16: end if
17: $x_{n+1}\leftarrow x_{n+1}+\sum_{j=0}^{\max(n-1,0)}a_{j,n}(\phi)\cdot x_{j}+\sum_{j=0}^{\max(n-3,0)}c_{j,n}(\phi)\cdot\boldsymbol{v}(\mathbf{x}_{j},t_{j}^{\theta}+\xi_{j})$

# D.4.1 PIXEL SPACE DIFFUSION ON CIFAR10, FFHQ, AND AFHQV2

• Pre-trained diffusion model:

– EDM (Karras et al., 2022);

• Teacher:

– UniPC solver, NFE = 20, logSNR schedule;

• Discriminator R3GAN (Huang et al., 2024):

– Pre-trained CIFAR10 checkpoint for CIFAR10;   
– Pre-trained FFHQ-64 checkpoint for both FFHQ and AFHQv2;   
– Training in pixel space;

• Image resolution:

– W = H = 32, C = 3 for CIFAR10;   
– W = H = 64, C = 3 for FFHQ and AFHQv2;

• Training/validation dataset size:

– CIFAR10: 1400/1000 for GS and GAS;   
– FFHQ and AFHQv2: 1400/1000 for GS; 5000/1000 for GAS;

• Solver training:

– Ldistill is LPIPS;   
– $\mathcal { L } _ { a d v }$ with weight = 0.1 for CIFAR10 and weight = 1.0 for FFHQ and AFHQv2;   
– EMA decay = 0.999;   
– Batch size = 24;   
– Adam optimizer, lr = 0.001, betas = (0.9, 0.999), weight decay = 0.0;   
– Gradients are clipped by the norm of 1.0;

• Discriminator training:

– Batch size = 24;   
– Adam optimizer, lr = 0.00001, betas = (0.9, 0.999), weight decay = 0.0;   
– $\lambda _ { 1 }$ and $\lambda _ { 2 }$ in Equation 11 are equal to 0.1;

• Training duration:

– 10k iterations for GS/GAS;

# D.4.2 LATENT SPACE DIFFUSION ON LSUN-BEDROOM AND IMAGENET

• Pre-trained diffusion model:

– LDM (Rombach et al., 2022);

• Teacher:

– UniPC solver for both LSUN-Bedrooms and ImageNet;   
– NFE = 20 and time-uniform schedule for LSUN;   
– NFE = 10 and time-quadratic schedule for ImageNet;

• Discriminator R3GAN (Huang et al., 2024):

– FFHQ-64 architecture with random initialization;   
– Training in latent space;

• Image resolution:

– W = H = 256, C = 3;   
– W ′ = H ′ = 64, C ′ = 3;

• Guidance scale: 2.0 (for ImageNet);

• Training/validation dataset size:

– 1400/1000 for GS;   
– 5000/1000 for GAS;

• Solver training:

– $\mathcal { L } _ { d i s t i l l }$ is L1 in latent space;   
– Ladv with weight = 1.0;   
– EMA decay = 0.999;   
– Batch size = 8;   
– Adam optimizer, lr = 0.001, betas = (0.9, 0.999), weight decay = 0.0;   
– Gradients are clipped by the norm of 1.0;

• Discriminator training:

– Batch size = 8;   
– Adam optimizer, lr = 0.00001, betas = (0.9, 0.999), weight decay = 0.0;   
– $\lambda _ { 1 }$ and $\lambda _ { 2 }$ in Equation 11 are equal to 0.1;

• Training duration:

– 30k iterations for GS/GAS;

# D.4.3 TEXT-TO-IMAGE GENERATION WITH STABLE DIFFUSION

• Pre-trained diffusion model:

– Stable Diffusion v1.5 (Rombach et al., 2022);   
– Gradient checkpointing at every UNet inference;

• Teacher:

– NFE = n + 1, where n = student NFE;   
– IPNDM(2M) solver with GITS;

• Discriminator R3GAN (Huang et al., 2024):

– FFHQ-64 architecture with random initialization;   
– First convolution layer modified to accept 4-channel latent inputs;   
– Training in latent space;

• Image resolution:

$- \ W \times H = 5 1 2 \times 5 1 2 , C = 3$   
$- \ W ^ { \prime } \times H ^ { \prime } = 6 4 \times 6 4 , C ^ { \prime } = 4$

• Guidance scale: 7.5;

• Training/validation dataset size:

– 1400/128 for GS;   
– 5000/128 for GAS;

• Solver training:

– $\mathcal { L } _ { d i s t i l l }$ is L1 in latent space;   
$- \ \mathcal { L } _ { a d v } \ \mathrm { { w i t h } \ w e i g h t = 1 . 0 ; }$   
$- \mathrm { \ E M A { \ d e c a y } } = 0 . 9 9 9 ;$   
– Batch size = 4;   
– Adam optimizer, lr = 0.001, betas = (0.9, 0.999), weight decay = 0.0;   
– Gradients are clipped by the norm of 1.0;

• Discriminator training:

– Batch size = 4;   
– Adam optimizer, lr = 0.00001, betas = (0.9, 0.999), weight decay = 0.0;   
– $\lambda _ { 1 }$ and $\lambda _ { 2 }$ in Equation 11 are equal to 0.1;

• Training duration:

– 1k iterations for GS;   
– 2k iterations for GAS;

# E TIMESTEPS VISUALIZATION

To validate the adequacy of the learned decoupled timesteps, we provide plots showing both $t ^ { \theta }$ and decoupled $t ^ { \theta } + \xi$ timestep schedules for GS and GAS across several NFE settings on FFHQ and ImageNet (Figure 7).

The decoupled timesteps should remain positive and monotonically decreasing, and we observe that they consistently satisfy these common-sense characteristics without any explicit constraints during training. They never become negative and always decrease monotonically; moreover, they are typically slightly lower than the corresponding unconstrained timesteps.

In one experiment, the first timestep was slightly above 1, but this caused no issues: the denoising model uses continuous positional embeddings and is robust to small shifts, so such a value is not out of distribution.

# F ADDITIONAL SAMPLES

To further demonstrate the method’s competitive results, we provide the reader with the additional samples of GS and GAS, compared to the teacher and the baseline UniPC with the same NFE. For all models/datasets except Stable Diffusion, we choose samples corresponding to 6 random seeds (marked as "random") and 6 samples that are the most distinguishable between GS and GAS in terms of pixel-space $L _ { 1 }$ distance (marked as "selected"). We choose the selected sample seeds at NFE = 4 and report the corresponding samples for all NFE. We report the samples for FFHQ (Figures 8, 9, 10, 11), AFHQv2 (Figures 12, 13, 14, 15), LSUN Bedroom (Figures 16, 17, 18, 19) and ImageNet (Figures 20, 21, 22, 23).

Most random samples show only minor fine-grained differences between GS and GAS (which is still important and has a positive effect on FID, as indicated in Table 2). At the same time, the selected samples fully demonstrate the potential effect of the adversarial loss on the image quality. Most GAS samples at NFE = 4 demonstrate superior image quality compared to GS, while being farther from teacher. This further complements the results demonstrated in Figure 4. At the same time, one could tell that the pictures enhanced by adversarial loss differ depending on NFE: pictures from the same random seeds become significantly closer to the teacher starting from NFE = 6. This also indicates that the effect of the adversarial loss is the most prominent at low NFE, where it is harder for the student to replicate teacher’s performance.

![](images/33b36eab4062d44102654f52915268ade583e588c6723bfaa8d64bad3c118000.jpg)  
Figure 7: Visualization of final timestep schedules with and without additive corrections on FFHQ and ImageNet datasets. It is important to note that they never violate any of the common-sense characteristics expected of timestep schedules.

Mode collapse It is also worth noting that incorporation of the adversarial loss into the training process does not lead to mode collapse — a common concern in such cases — as we explicitly address this issue using the relativistic GAN loss from Huang et al. (2024). The random samples reported in Figures 8- 25 show generation diversity, while low resulting FID values indicate both high quality of our images and the absence of mode collapse.

Stable Diffusion For the Stable Diffusion experiments, we generate images from the 250 MS-COCO-val prompts with both the official LD3 implementation and our GAS method, initializing both with identical random latent noise. From these outputs, we select six images at random (marked "random") and six that best highlight the visual differences between GAS and LD3 (marked "selected").

# Random prompts:

• “A woman sitting on a bench and a woman standing waiting for the bus.”   
• “jumbo jet sits on the tarmac while another takes off”   
• “An old green car parked on the side of the street.”   
• “A gas stove next to a stainless steel kitchen sink and countertop.”   
• “A person walking through the rain with an umbrella.”

# Selected prompts:

• “A man in a wheelchair and another sitting on a bench that is overlooking the water.”   
• “A fireplace with a fire built in it.”   
• “A half eaten dessert cake sitting on a cake plate.”   
• “an airport with one plane flying away and the other sitting on the runway”   
• “A dirt bike rider doing a stunt jump in the air”

The resulting comparisons are shown in Figures 24, 25.

![](images/9fd0b5fa9ed307506cef6ac9ef02c2b27b1edccff33bb145e714cb80dee76ad3.jpg)

<details>
<summary>other</summary>

| Selected | Random |
| -------- | ------ |
| UniPC    | 10     |
| UniPC    | 15     |
| UniPC    | 20     |
| UniPC    | 25     |
| UniPC    | 30     |
| UniPC    | 35     |
| UniPC    | 40     |
| UniPC    | 45     |
| UniPC    | 50     |
| UniPC    | 55     |
| UniPC    | 60     |
| UniPC    | 65     |
| UniPC    | 70     |
| UniPC    | 75     |
| UniPC    | 80     |
| UniPC    | 85     |
| UniPC    | 90     |
| UniPC    | 95     |
| UniPC    | 100    |
| GS       | 10     |
| GS       | 15     |
| GS       | 20     |
| GS       | 25     |
| GS       | 30     |
| GS       | 35     |
| GS       | 40     |
| GS       | 45     |
| GS       | 50     |
| GS       | 55     |
| GS       | 60     |
| GS       | 65     |
| GS       | 70     |
| GS       | 75     |
| GS       | 80     |
| GS       | 85     |
| GS       | 90     |
| GS       | 95     |
| GS       | 100    |
| GAS      | 10     |
| GAS      | 15     |
| GAS      | 20     |
| GAS      | 25     |
| GAS      | 30     |
| GAS      | 35     |
| GAS      | 40     |
| GAS      | 45     |
| GAS      | 50     |
| GAS      | 55     |
| GAS      | 60     |
| GAS      | 65     |
| GAS      | 70     |
| GAS      | 75     |
| GAS      | 80     |
| GAS      | 85     |
| GAS      | 90     |
| GAS      | 95     |
| GAS      | 100    |
| Teacher   | 10     |
| Teacher   | 15     |
| Teacher   | 20     |
| Teacher   | 25     |
| Teacher   | 30     |
| Teacher   | 35     |
| Teacher   | 40     |
| Teacher   | 45     |
| Teacher   | 50     |
| Teacher   | 55     |
| Teacher   | 60     |
| Teacher   | 65     |
| Teacher   | 70     |
| Teacher   | 75     |
| Teacher   | 80     |
| Teacher   | 85     |
| Teacher   | 90     |
| Teacher   | 95     |
| Teacher   | 100    |
</details>

Figure 8: Comparison of GS and GAS with the teacher and UniPC on FFHQ with NFE = 4.

![](images/650af742b14250f1ca543ca81e99eede924e6c6ddbcf9cd45b9d701dec38a5fb.jpg)  
Figure 9: Comparison of GS and GAS with the teacher and UniPC on FFHQ with NFE = 6.

![](images/38a6e6a9a5a4d92a6eabd9ea7fececb139df63712e5f303be85068f44819d074.jpg)  
Figure 10: Comparison of GS and GAS with the teacher and UniPC on FFHQ with NFE = 8.   
Figure 11: Comparison of GS and GAS with the teacher and UniPC on FFHQ with NFE = 10.

![](images/03f413ec6e88e5181419751821bc00e10999024edd481acc7ee75dec94602925.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
Random
</details>

Figure 12: Comparison of GS and GAS with the teacher and UniPC on AFHQv2 with NFE = 4.

![](images/07a1945f5b565c60c930da35acc73850b2792c4c087c982f890b934ccc00e9b6.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
Random
</details>

Figure 13: Comparison of GS and GAS with the teacher and UniPC on AFHQv2 with NFE = 6.

![](images/85ef769b4a9646a3ad0d2216d83177386baf42382538e237806d32f3620376fe.jpg)  
Figure 14: Comparison of GS and GAS with the teacher and UniPC on AFHQv2 with NFE = 8.   
Figure 15: Comparison of GS and GAS with the teacher and UniPC on AFHQv2 with NFE = 10.

![](images/2a84cef6b3e597ce990cd120180df95fbbc7a25c01288fe6a97d230ab0026d69.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
</details>

![](images/a03ba4d0c47389912f765e95665613c25983eb7ddde8a667de408ebae3b94cbf.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
</details>

![](images/b4548785f1489967b87fb3541b92444598a890e4e5f4d953a188151fa08eec9c.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 grayscale photos showing various bedroom layouts and furniture, no visible text or symbols
</details>

![](images/5b9f213d5f5d0a2b3c2fe048f8b8c4a14e22bf269e4b1163966e757d0f4cbdd0.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 identical photos showing a bedroom interior with beds, lamps, and curtains, no visible text or symbols.
</details>

Figure 16: Comparison of GS and GAS with the teacher and UniPC on LSUN-Bedroom with NFE = 4.   
Figure 17: Comparison of GS and GAS with the teacher and UniPC on LSUN-Bedroom with NFE = 5.

![](images/6c1bf2cdca83048b5eb8609ddb9014a2e9db527d08aeafac785bb6d1c7e0638e.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
</details>

![](images/736cc48b0600ce3c938bf086e7646b45df931b82e6bc5ab0d0bc7ce512fb5a5a.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
</details>

![](images/f60be5a251344990afa26df559c772da5019725f6f825f218aedfdcbb5726754.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 grayscale photos showing a bedroom interior with beds, chairs, and windows, no visible text or symbols.
</details>

![](images/3733cbe4df19bcd0117219e4e92ae92c7a0f6669de4c580ff0c1714210909f23.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 identical photos showing a bedroom interior with furniture, bedding, and curtains, no visible text or symbols.
</details>

Figure 18: Comparison of GS and GAS with the teacher and UniPC on LSUN-Bedroom with NFE = 6.   
Figure 19: Comparison of GS and GAS with the teacher and UniPC on LSUN-Bedroom with NFE = 7.

![](images/f8629b8fdf103cad6dc083f571f92f131b4096f6e6afb12d7f81ee82b9c181e9.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
Random
</details>

Figure 20: Comparison of GS and GAS with the teacher and UniPC on ImageNet with NFE = 4.

![](images/f456a6140f45a95d6a0624ae116e88ee758dc614ee144aa18c1374bef26fbdf6.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
Random
</details>

Figure 21: Comparison of GS and GAS with the teacher and UniPC on ImageNet with NFE = 5.

![](images/7dc548b370687f09e2076ac9e9e02deee8c383dd70eb176c841f53648f4ba043.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
Random
</details>

Figure 22: Comparison of GS and GAS with the teacher and UniPC on ImageNet with NFE = 6.

![](images/325898a8c8dc8d0a0059a7cabde25ae0d7278528c7b68bddb9cf7fa39a3844cb.jpg)

<details>
<summary>text_image</summary>

UniPC GS GAS Teacher
Selected
Random
</details>

Figure 23: Comparison of GS and GAS with the teacher and UniPC on ImageNet with NFE = 7.

![](images/7a2b0101fdec3f613236d231487554447d1db6579c9992c66c4929af87c2d30e.jpg)  
Figure 24: Comparison of GAS with LD3 and GITS on MS-COCO with NFE = 5.   
Figure 25: Comparison of GAS with LD3 and GITS on MS-COCO with NFE = 6.