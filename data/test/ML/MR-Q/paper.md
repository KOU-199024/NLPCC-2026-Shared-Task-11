# TOWARDS GENERAL-PURPOSE MODEL-FREE REINFORCEMENT LEARNING

Scott Fujimoto, Pierluca D’Oro, Amy Zhang, Yuandong Tian, Michael Rabbat Meta FAIR

# ABSTRACT

Reinforcement learning (RL) promises a framework for near-universal problemsolving. In practice however, RL algorithms are often tailored to specific benchmarks, relying on carefully tuned hyperparameters and algorithmic choices. Recently, powerful model-based RL methods have shown impressive general results across benchmarks but come at the cost of increased complexity and slow run times, limiting their broader applicability. In this paper, we attempt to find a unifying model-free deep RL algorithm that can address a diverse class of domains and problem settings. To achieve this, we leverage model-based representations that approximately linearize the value function, taking advantage of the denser task objectives used by model-based RL while avoiding the costs associated with planning or simulated trajectories. We evaluate our algorithm, MR.Q, on a variety of common RL benchmarks with a single set of hyperparameters and show a competitive performance against domain-specific and general baselines, providing a concrete step towards building general-purpose model-free deep RL algorithms.

![](images/c9bdc5be0de42f161c18845cdad1a2b7c023e3d2983f905f1e7e7c73137abc7f.jpg)  
Figure 1: Summary of results. Aggregate mean performance across four common RL benchmarks and 118 environments featuring diverse characteristics (e.g., observation and action spaces, task types). Error bars capture a 95% stratified bootstrap confidence interval. Our algorithm, MR.Q, achieves a competitive performance against both state-of-the-art domain-specific and general baselines, while using a single set of hyperparameters. Notably, MR.Q accomplishes this with fewer network parameters and substantially faster training and evaluation speeds than general-purpose model-based methods.

# 1 INTRODUCTION

The conceptual premise of RL is inherently general-purpose—an RL agent can learn optimal behavior with only two basic elements: a well-defined objective and data describing its interactions with the environment. In reality, however, most RL algorithms are anything but general-purpose. Instead, RL algorithms are highly specialized and typically characterized by specific problem classes, such as discrete versus continuous actions or vector versus pixel observations, with each category requiring its own set of algorithmic choices and hyperparameters. For example, Rainbow and TD3 (Hessel et al., 2018; Fujimoto et al., 2018), common methods for Atari and MuJoCo respectively (Bellemare et al., 2013; Todorov et al., 2012), have more differences than similarities in their shared hyperparameters (Table 1)—without accounting for further algorithmic differences.

To some extent, general-purpose algorithms do exist—policy gradient methods (Williams, 1992; Schulman et al., 2015; 2017) and many evolutionary approaches (Rechenberg, 1978; Back, 1996; Rubinstein, 1997; Salimans et al., 2017) require few assumptions on the underlying problem. Unfortunately, these methods often offer poor sample efficiency and asymptotic performance compared more domain-specific approaches, and in some instances, can require extensive re-tuning over numerous implementation-level details (Engstrom et al., 2020; Huang et al., 2022).

Table 1: Hyperparameter differences between Rainbow (Hessel et al., 2018) and TD3 (Fujimoto et al., 2018). TD3 uses an expected moving average (EMA) update with an effective frequency of $\overline { { { \frac { 1 } { 1 - 0 . 9 9 5 } } } } = 2 0 0$ . 

<table><tr><td>Hyperparameter</td><td>Rainbow</td><td>TD3</td></tr><tr><td>Discount factor</td><td>0.99</td><td>0.99</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning Rate</td><td> $6.25 \cdot 10^{-5}$ </td><td> $10^{-3}$ </td></tr><tr><td>Adam  $\epsilon$ </td><td> $1.5 \cdot 10^{-4}$ </td><td> $10^{-8}$ </td></tr><tr><td>Replay buffer size</td><td>1M</td><td>1M</td></tr><tr><td>Minibatch size</td><td>32</td><td>100</td></tr><tr><td>Target network update</td><td>Iterative</td><td>EMA</td></tr><tr><td>Effective target update freq.</td><td>8k</td><td>200</td></tr><tr><td>Initial random steps</td><td>20k</td><td>1k</td></tr></table>

Recently, DreamerV3 (Hafner et al., 2023) and TD-MPC2 (Hansen et al., 2024), have showcased the potential of general-purpose model-based approaches, achieving impressive single-task performance on a diverse set of benchmarks without re-tuning hyperparameters. However, despite their success, model-based methods also introduce substantial algorithmic and computational complexity, making them less practical than lightweight domain-specific model-free algorithms.

This paper presents a general model-free RL algorithm that leverages model-based representations to achieve the sample efficiency and performance of model-based methods, without the computational overhead. A recent surge of high-performing model-free RL algorithms with dynamics-based representations (Guo et al., 2020; 2022; Schwarzer et al., 2020; 2023; Zhao et al., 2023; Fujimoto et al., 2024; Zheng et al., 2024; Scannell et al., 2024) has showcased the potential of this family of algorithms when tailored for a single benchmark. Recognizing the similarity between these modelbased and model-free approaches, our hypothesis is that the true benefit of model-based objectives is in the implicitly learned representation, rather than the model itself, and thus prompting the question:

Can model-based representations alone enable sample-efficient general-purpose learning?

Our proposed approach is based on learning features that approximately capture a linear relationship between state-action pairs and value. To do so, we draw heavily from modern dynamics-based representation learning methods (see Related Work) as well as the work of Parr et al. (2008), who show that both model-based and model-free objectives converge to the same solution in linear space. By mapping states and actions into a single, unified embedding, we eliminate any environmentspecific characteristics of the input space and allow for a standardized set of hyperparameters.

We evaluate our method, MR.Q, on four widely used RL benchmarks and 118 environments, and achieve competitive performance against state-of-the-art domain-specific and general baselines without algorithmic or hyperparameter changes between environments or benchmarks.

# 2 RELATED WORK

General-purpose RL. Although many traditional RL methods are general-purpose in principle, practical constraints often force assumptions about the task domain. For example, algorithms like Q-learning and SARSA (Watkins, 1989; Rummery & Niranjan, 1994) can be conceptually extended to continuous spaces, but are typically implemented using discrete lookup tables. In practice, early examples of general decision-making approaches can be found in on-policy methods with function approximation. For instance, both evolutionary algorithms (Rechenberg, 1978; Back, 1996; Rubinstein, 1997; Salimans et al., 2017) and policy gradient methods (Williams, 1992; Sutton et al., 1999;

Schulman et al., 2015; 2017) offer update rules with convergence guarantees and independence to the input space. However, despite their generality, these methods are also hindered by poor sample efficiency and are prone to local minima, limiting their suitability for many practical applications.

In contrast, the design of deep RL algorithms tends to favor more specialized approaches that align closely with a single benchmark—e.g., DQN Atari (Bellemare et al., 2013; Mnih et al., 2015), DDPG MuJoCo (Todorov et al., 2012; Lillicrap et al., 2015), or AlphaGo Go (Silver et al., 2016). Generalizing beyond these initial benchmarks can often require significant engineering, tuning, or algorithmic discovery (Luong et al., 2019; Schrittwieser et al., 2020; Haydari & Yılmaz, 2020; Ibarz et al., 2021). In imitation learning, GATO achieved generalist behavior, but relied on large expert datasets (Reed et al., 2022). Recently, DreamerV3 (Hafner et al., 2023) demonstrated a strong capability over many benchmarks without re-tuning, but used costly large models and simulated rollouts. Our objective is to discover a lightweight model-free approach to general-purpose learning.

Dynamics-based representation learning. Building representations from system dynamics is a long-standing approach for adaptation, partial observability, and feature selection (Dayan, 1993; Littman & Sutton, 2001; Parr et al., 2008). Numerous model-free methods have been developed to learn representations by predicting future latent states (Munk et al., 2016; Van Hoof et al., 2016; Zhang et al., 2018; Gelada et al., 2019; Lee et al., 2020; Guo et al., 2020; 2022; Schwarzer et al., 2020; 2023; Zintgraf et al., 2021; Yu et al., 2021; 2022; Fujimoto et al., 2021; 2024; McInroe et al., 2021; Seo et al., 2022; Kim et al., 2022; Tang et al., 2023; Zhao et al., 2023; Zheng et al., 2024; Ni et al., 2024; Scannell et al., 2024). Unsurprisingly, these model-free approaches closely relate to model-based counterparts which learn a latent dynamics model for planning or value estimation (Watter et al., 2015; Finn et al., 2016; Karl et al., 2017; Ha & Schmidhuber, 2018; Schrittwieser et al., 2020; 2021; Ye et al., 2021; Hansen et al., 2022; 2024; Hafner et al., 2019; 2023; Wang et al., 2024). Our approach, MR.Q, is most closely related to the state-action representation learning in TD7 (Fujimoto et al., 2024). At a high level, MR.Q differs from TD7 by discarding the original input and including losses over the reward and termination. MR.Q also differs significantly in implementation, drawing inspiration from prior work to determine a set of design choices that performs well across benchmarks, including multi-step returns, unrolled dynamics, and categorical losses.

Our motivation also relates to linear MDPs (Jin et al., 2020; Agarwal et al., 2020) and linear spectral representation (Ren et al., 2022; 2023; Zhang et al., 2022; Shribak et al., 2024). The latter aims to learn a low-rank decomposition of the transition dynamics of the MDP and recover a linear relationship between an embedding and the value function. Similarly, our work connects to two-stage linear RL, where a non-linear embedding is learned for linear RL (Levine et al., 2017; Chung et al., 2019).

State abstraction. Our work is closely related to bisimulation metrics (Ferns et al., 2004; 2011; Castro, 2020) and MDP homomorphisms (Ravindran, 2004; van der Pol et al., 2020a;b; Rezaei-Shoshtari et al., 2022) which rely on measures of similarity in reward and dynamics for state or action abstraction. These concepts have inspired practical approximations to bisimulation metrics as a means of shaping representations in deep RL agents, particularly those using image-based observations (Zhang et al., 2020; Castro et al., 2021; Zang et al., 2022).

# 3 BACKGROUND

Reinforcement learning (RL) problems are described by a Markov Decision Process (MDP) (Bellman, 1957), which we define by a tuple $( S , A , p , R , \gamma )$ of state space S, action space A, dynam-( )ics function p, reward function R and discount factor γ. Value-based RL methods learn a value function $\begin{array} { r } { Q ^ { \pi } \bar { \langle } s , a \rangle : = \mathbb { E } _ { \pi } \big [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { t } \big | s _ { 0 } = s , a _ { 0 } = a \big ] } \end{array}$ that models the expected discounted sum of rewards $\boldsymbol { r } _ { t } \sim R ( s _ { t } , a _ { t } )$ by following a policy π which maps states s to actions a.

The true value function $Q ^ { \pi }$ is estimated by an approximate value function $Q _ { \theta }$ . We use subscripts to indicate the network parameters θ. Target networks, which are used to introduce stationarity in prediction targets, have parameters denoted by an apostrophe, $\mathrm { e . g . , } Q _ { \theta ^ { \prime } }$ . These parameters are periodically synchronized with the current network parameters $( \theta ^ { \prime }  \theta )$ .

# 4 MODEL-BASED REPRESENTATIONS FOR Q-LEARNING

This section presents the MR.Q algorithm (Model-based Representations for Q-learning), a modelfree RL algorithm that learns an approximately linear representation of the value function through model-based objectives. Value-based RL algorithms learn a value function Q that maps state-action pairs $( s , a )$ to values in R and a policy π that maps states s to actions a. Like many representation learning methods for RL, MR.Q adds an initial step that transforms states and state-action pairs into embeddings $\mathbf { z } _ { s }$ and $\mathbf { z } _ { s a : }$ , which serves as inputs to the downstream policy and value function.

$$
f _ {\omega}: s \rightarrow \mathbf {z} _ {s}, \quad g _ {\omega}: (s, a) \rightarrow \mathbf {z} _ {s a}, \tag {1}
$$

$$
\pi_ {\phi}: \mathbf {z} _ {s} \rightarrow a, \quad Q _ {\theta}: \mathbf {z} _ {s a} \rightarrow \mathbb {R}. \tag {2}
$$

While neither the value function nor policy require explicit representation learning, using intermediate embeddings has two main benefits:

1. Introducing an explicit representation learning stage can enable richer alternative learning signals that are grounded in the dynamics and rewards of the MDP, as opposed to relying exclusively on non-stationary value targets used in both value and policy learning.   
2. Representation learning can transform the input into a unified, abstract space that is decoupled from the original input characteristics, e.g., images or action spaces. This abstraction allows us to filter irrelevant or spurious details and use unified downstream architectures, improving robustness to environment variations.

To learn these embeddings, we draw inspiration from linear feature selection, revisiting the work of Parr et al. (2008), as well as MDP homomorphisms (Ravindran & Barto, 2002). In Section 4.1 we highlight how model-based objectives can be used to learn features that share an approximately linear relationship with the true value function. Then in Section 4.2, we relax our theoretical motivation for a practical algorithm based on recent advances in dynamics-based representation learning.

# 4.1 THEORETICAL MOTIVATION

Consider a linear decomposition of the value function, where the value function $Q ( s , a )$ is represented by features $\mathbf { z } _ { s a }$ and linear weights w:

$$
Q (s, a) = \mathbf {z} _ {s a} ^ {\top} \mathbf {w}. \tag {3}
$$

Our primary objective is to learn features $\mathbf { z } _ { s a }$ that share an approximately linear relationship with the true value function $Q ^ { \pi }$ . However, since this relationship is only approximate, we use these features as input to a non-linear function $\hat { Q } ( \mathbf { z } _ { s a } )$ , rather than relying solely on linear function approximation.

We start by exploring how to find features that can linearly represent the true value function. Given a dataset D of tuples $( s , a , r , s ^ { \prime } , a ^ { \prime } )$ , we consider two possible approaches for learning a value func-( )tion Q: A model-free update based on semi-gradient TD (Sutton, 1988; Sutton & Barto, 1998):

$$
\mathbf {w} \leftarrow \mathbf {w} - \alpha \mathbb {E} _ {D} \left[ \nabla_ {\mathbf {w}} \left(\mathbf {z} _ {s a} ^ {\top} \mathbf {w} - | r + \gamma \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \mathbf {w} | _ {\mathrm{sg}}\right) ^ {2} \right]. \tag {4}
$$

A model-based approach to learn $\mathbf { w } _ { \mathrm { m b } }$ , based on rolling out estimates of the dynamics and reward:

$$
\mathbf {w} _ {\mathrm{mb}} := \sum_ {t = 0} ^ {\infty} \gamma^ {t} W _ {p} ^ {t} \mathbf {w} _ {r}, \tag {5}
$$

$$
\mathbf {w} _ {r} := \underset {\mathbf {w}} {\operatorname{argmin}} \mathbb {E} _ {D} \left[ \left(\mathbf {z} _ {s a} ^ {\top} \mathbf {w} - r\right) ^ {2} \right], \quad W _ {p} := \underset {W} {\operatorname{argmin}} \mathbb {E} _ {D} \left[ \left(\mathbf {z} _ {s a} ^ {\top} W - \mathbf {z} _ {s ^ {\prime} a ^ {\prime}}\right) ^ {2} \right]. \tag {6}
$$

Closely following Parr et al. (2008) and Song et al. (2016), we can show that these approaches converge to the same solution (proofs for this section can be found in Appendix A).

Theorem 1. The fixed point of the model-free approach (Equation 4) and the solution of the modelbased approach (Equation 5) are the same.

From the insight of Theorem 1, we can connect the value error VE, the difference between an approximate value function Q and the true value function $Q ^ { \pi }$ ,

$$
\operatorname{VE} (s, a) := Q (s, a) - Q ^ {\pi} (s, a) \tag {7}
$$

to the accuracy of reward and dynamics components of the estimated model (Theorem 2).

Theorem 2. The value error of the solution described by Theorem 1 is bounded by the accuracy of the estimated dynamics and reward:

$$
\left| \mathrm{VE} (s, a) \right| \leq \frac {1}{1 - \gamma} \max _ {(s, a) \in S \times A} \left(\left| \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - \mathbb {E} _ {r | s, a} [ r ] \right| + \max _ {i} \left| \mathbf {w} _ {i} \right| \sum \left| \mathbf {z} _ {s a} ^ {\top} W _ {p} - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} | s, a} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} \right] \right|\right). \tag {8}
$$

Parr et al. (2008) and Song et al. (2016) use a related insight regarding the Bellman error to infer an approach for feature selection. However, with the advent of deep learning, we can instead directly learn the features $\mathbf { z } _ { s a }$ by jointly optimizing them alongside the linear weights ${ \bf w } _ { r }$ and $W _ { p } .$ . This is accomplished by treating the features and linear weights as a unified end-to-end model and balancing the losses in Equation 6 with a hyperparameter λ:

$$
\mathcal {L} \left(\mathbf {z} _ {s a}, \mathbf {w} _ {r}, W _ {p}\right) = \underbrace {\mathbb {E} _ {D} \left[ \left(\mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - r\right) ^ {2} \right]} _ {\text { Reward   learning }} + \underbrace {\lambda \mathbb {E} _ {D} \left[ \left(\mathbf {z} _ {s a} ^ {\top} W _ {p} - \mathbf {z} _ {s ^ {\prime} a ^ {\prime}}\right) ^ {2} \right]} _ {\text { Dynamics   learning }}. \tag {9}
$$

However, the resulting Equation 9 has some notable drawbacks.

Dependency on $\pi .$ The dynamics target $\mathbf { z } _ { s ^ { \prime } a ^ { \prime } }$ depends on an action $a ^ { \prime }$ determined by the policy $\pi .$ In policy optimization problems, this introduces non-stationarity, where the target embedding must be continually updated to reflect changes in the policy. This creates an undesirable interdependence between the policy and encoder.

Undesirable local minima. Jointly optimizing both the features $\mathbf { z } _ { s a }$ and the dynamics target can lead to undesirable local minima, similar to the issues encountered with Bellman residual minimization (Baird, 1995; Fujimoto et al., 2022). This can result in collapsed or trivial solutions when the dataset does not fully cover the state and action space or when the reward is sparse.

To address these issues, we suggest relaxations on our proposed, theoretically grounded approach:

$$
\mathcal {L} \left(\mathbf {z} _ {s a}, \mathbf {w} _ {r}, W _ {p}\right) = \mathbb {E} _ {D} \left[ \left(\mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - r\right) ^ {2} \right] + \lambda \mathbb {E} _ {D} \left[ \left(\mathbf {z} _ {s a} ^ {\top} W _ {p} - \bar {\underline {{\mathbf {z}}} _ {s ^ {\prime}}}\right) ^ {2} \right]. \tag {10}
$$

We propose two key modifications to alleviate the aforementioned issues. Firstly, we use a statedependent embedding $\mathbf { z } _ { s ^ { \prime } }$ as the dynamics target, rather than the state-action embedding $\mathbf { z } _ { s ^ { \prime } a ^ { \prime } }$ . This eliminates any dependency on the current policy while still capturing the environment’s dynamics.

Secondly, to mitigate the issue of local minima, we use a target network $f _ { \omega ^ { \prime } } ( s ^ { \prime } )$ to generate the dynamics target $\bar { \mathbf { z } } _ { s ^ { \prime } }$ , where the parameters $\omega ^ { \prime }$ are periodically updated to track the current network parameters ω. Empirical evidence from prior work suggests that this approach can yield significant performance gains (Grill et al. (2020); Assran et al. (2023), see Related Work), although it no longer guarantees convergence to a fixed point.

Due to these two changes, even if the modified objective defined by Equation 10 is minimized, we can no longer assume there is a linear relationship between the embedding $\mathbf { z } _ { s a }$ and the value function. However, we can instead allow for a non-linear relationship, replacing linear weights w with a non-linear function $\hat { Q } ( \mathbf { z } _ { s a } )$ . We can show that this relationship exists as long as the features are sufficiently rich (i.e., such that a MDP homomorphism is satisfied (Ravindran & Barto, 2002)).

Theorem 3. Given functions $f ( s ) = \mathbf { z } _ { s }$ and $g ( \mathbf { z } _ { s } , a ) = \mathbf { z } _ { s a }$ , then if there exists functions $\hat { p }$ and $\hat { R }$ such that for all $( s , a ) \in S \times A { \mathrm { : } }$ :

$$
\mathbb {E} _ {\hat {R}} \left[ \hat {R} \left(\mathbf {z} _ {s a}\right) \right] = \mathbb {E} _ {R} [ R (s, a) ], \quad \hat {p} \left(\mathbf {z} _ {s ^ {\prime}} \mid \mathbf {z} _ {s a}\right) = \sum_ {\hat {s}: \mathbf {z} _ {\hat {s}} = \mathbf {z} _ {s ^ {\prime}}} p (\hat {s} | s, a), \tag {11}
$$

then for any policy $\pi$ where there exists a corresponding policy ${ \hat { \pi } } ( a | \mathbf { z } _ { s } ) = \pi ( a | s )$ , there exists a function $\hat { Q }$ equal to the true value function $Q ^ { \pi }$ over all possible state-action pairs $( s , a ) \in S \times A$ :

$$
\hat {Q} \big (\mathbf {z} _ {s a} \big) = Q ^ {\pi} (s, a). \tag {12}
$$

Furthermore, Equation 11 guarantees the existence of an optimal policy ${ \hat { \pi } } ^ { * } ( a | \mathbf { z } _ { s } ) = \pi ^ { * } ( a | s )$ .

Consequently, even if the features $\mathbf { z } _ { s a }$ do not linearly represent the true value function, i.e., the loss in Equation 9 cannot be not exactly minimized, $\mathbf { z } _ { s a }$ can still be used in a non-linear relationship to represent the value function. Furthermore, Theorem 3 outlines a similar objective as the original linear objective defined in Equation 9, in learning the reward and dynamics of the MDP.

These results motivates the practical algorithm discussed in the following section. Using the adjusted loss defined in Equation 10, we will aim to learn features with an approximately linear relationship to the true value function, but use a non-linear value function with those features to account for the error induced by our approximations.

# 4.2 ALGORITHM

We now present the details of MR.Q (Model-based Representations for Q-learning). Building on the insights from the previous section, our key idea is to learn a state-action embedding $\mathbf { z } _ { s a }$ that is approximately linear with the true value function $Q ^ { \pi }$ . To account for approximation errors, these features are used with non-linear function approximation to determine the value.

The state embedding vector $\mathbf { z } _ { s }$ is obtained as an intermediate component by training end-to-end with the state-action encoder. MR.Q handles different input modalities by swapping the architecture of the state encoder. Since $\mathbf { z } _ { s }$ is a vector, the remaining networks are independent of the observation space and use feedforward networks.

Given the transition $( s , a , r , d , s ^ { \prime } )$ from the replay buffer:

<table><tr><td colspan="2">Output MR.Q</td><td>Update MR.Q</td></tr><tr><td colspan="2">Trained end-to-end</td><td rowspan="5">if t %  $T_{target} = 0$  thenTarget networks:  $\theta', \phi', \omega' \leftarrow \theta, \phi, \omega$ .Reward scaling:  $\bar{r}' \leftarrow \bar{r}, \bar{r} \leftarrow \text{mean}_D r$ .for  $T_{target}$  time steps doEncoder update: Equation 14.</td></tr><tr><td>State Encoder</td><td> $\mathbf{z}_s = f_\omega(s)$ </td></tr><tr><td>State-Action Encoder</td><td> $\mathbf{z}_{sa} = g_\omega(\mathbf{z}_s, a)$ </td></tr><tr><td>MDP predictor</td><td> $\tilde{\mathbf{z}}_{s'}, \tilde{r}, \tilde{d} = \mathbf{z}_{sa}^\top \mathbf{m}$ </td></tr><tr><td colspan="2">Decoupled RL</td></tr><tr><td>Value</td><td> $\tilde{Q}_i = Q_\theta(\mathbf{z}_{sa})$ </td><td>Value update: Equation 19.</td></tr><tr><td>Policy</td><td> $a_\pi = \pi_\phi(\mathbf{z}_s)$ </td><td>Policy update: Equation 20.</td></tr></table>

The encoder loss is composed of three terms based on the reward, dynamics and terminal signal that are unrolled over a short horizon. The value function and policy are trained independently, using standard losses (Silver et al., 2014; Fujimoto et al., 2018). We use LAP (Fujimoto et al., 2020) to sample transitions with priority according to their TD errors (Schaul et al., 2016), the absolute difference between the predicted value and the target value in Equation 19.

The target network, reward scaling (defined in Equation 19), and the encoder are updated periodically every $T _ { \mathrm { t a r g e t } }$ time steps. This synchronized update schedule keeps the input and target output fixed for the downstream value function and policy within each iteration, thus reducing nonstationarity in the optimization (Fujimoto et al., 2024).

# 4.2.1 ENCODER

The encoder loss is based on unrolling the dynamics of the learned model over a short horizon. Given a subsequence of an episode $( s _ { 0 } , a _ { 0 } , r _ { 1 } , d _ { 1 } , s _ { 1 } , . . . , r _ { { \cal H } _ { \mathrm { E n c } } } , d _ { { \cal H } _ { \mathrm { E n c } } } , s _ { { \cal H } _ { \mathrm { E n c } } } )$ , the model is unrolled by encoding the initial state $s _ { 0 }$ , then by repeatedly applying the state-action encoder $g _ { \omega }$ and linear MDP predictor m:

$$
\tilde {\mathbf {z}} ^ {t}, \tilde {r} ^ {t}, \tilde {d} ^ {t} := g _ {\omega} \left(\tilde {\mathbf {z}} ^ {t - 1}, a ^ {t - 1}\right) ^ {\top} \mathbf {m}, \quad \text { where } \tilde {\mathbf {z}} ^ {0} := f _ {\omega} (s _ {0}). \tag {13}
$$

The final loss is summed over the unrolled model and balanced by corresponding hyperparameters:

$$
\mathcal {L} _ {\text { Encoder }} (f, g, \mathbf {m}) := \sum_ {t = 1} ^ {H _ {\text { Enc }}} \lambda_ {\text { Reward }} \mathcal {L} _ {\text { Reward }} (\tilde {r} ^ {t}) + \lambda_ {\text { Dynamics }} \mathcal {L} _ {\text { Dynamics }} (\tilde {\mathbf {z}} _ {s ^ {\prime}} ^ {t}) + \lambda_ {\text { Terminal }} \mathcal {L} _ {\text { Terminal }} (\tilde {d} ^ {t}). \tag {14}
$$

$\lambda _ { \mathrm { T e r m i n a l } }$ is set to 0 until the first terminal transition $( \mathrm { i . e . , } d = 0 )$ is viewed. This approach is commonly used in model-based RL (Oh et al., 2015; Hafner et al., 2023; Hansen et al., 2024), as well as dynamics-based representation learning (Schwarzer et al., 2020; 2023; Scannell et al., 2024).

Reward loss. While our theoretical analysis suggests using the mean-squared error to train the predicted reward, we find that a categorical representation of the reward is more effective in practice for predicting sparse rewards and is robust to reward magnitude. This empirical benefit is consistent with prior work (Schrittwieser et al., 2020; Hafner et al., 2023; Hansen et al., 2024; Wang et al., 2024). Our reward loss function uses the cross entropy CE between the predicted reward $\tilde { r }$ and a two-hot encoding of the reward r:

$$
\mathcal {L} _ {\text { Reward }} (\tilde {r}) := \mathrm{CE} (\tilde {r}, \text { Two - Hot } (r)). \tag {15}
$$

To handle a wide range of reward magnitudes without prior knowledge, the locations of the two-hot encoding are spaced at increasing non-uniform intervals, according to symexp x = $\mathrm { s i g n } ( x ) ( \exp { ( x ) } ^ { - 1 } )$ (Hafner et al., 2023).

Dynamics loss. The dynamics loss minimizes the mean-squared error between the predicted next state embedding $\tilde { \mathbf { z } } _ { s ^ { \prime } }$ and the next state embedding $\bar { \mathbf { z } } _ { s ^ { \prime } }$ from the target encoder $f _ { \omega ^ { \prime } } :$

$$
\mathcal {L} _ {\text { Dynamics }} \left(\tilde {\mathbf {z}} _ {s ^ {\prime}}\right) := \left(\tilde {\mathbf {z}} _ {s ^ {\prime}} - \bar {\mathbf {z}} _ {s ^ {\prime}}\right) ^ {2}. \tag {16}
$$

As discussed in the previous section, using the next state embedding $\mathbf { z } _ { s ^ { \prime } }$ eliminates the dependency on the policy that would occur when using a state-action embedding target.

Terminal loss. The predicted scalar terminal signal $\tilde { d }$ is trained simply using a MSE loss with the binary terminal signal d:

$$
\mathcal {L} _ {\text { Terminal }} (\tilde {d}) := (\tilde {d} - d) ^ {2}. \tag {17}
$$

# 4.2.2 VALUE FUNCTION

Value learning is primarily based on TD3 (Fujimoto et al., 2018). Specifically, we train two value functions and take the minimum output between their respective target networks to determine the value target. Similar to TD3, the target action is determined by the target policy $\pi _ { \phi ^ { \prime } }$ , perturbed by small amount of clipped Gaussian noise:

$$
a _ {\pi} = \left\{ \begin{array}{l l} \operatorname{argmax} a ^ {\prime} & \text { for   discrete } A, \\ \operatorname{clip} (a ^ {\prime}, - 1, 1) & \text { for   continuous } A, \end{array} \right. \quad \text { where } a ^ {\prime} = \pi_ {\phi^ {\prime}} (s ^ {\prime}) + \operatorname{clip} (\epsilon , - c, c), \quad \epsilon \sim \mathcal {N} (0, \sigma^ {2}). \tag {18}
$$

Discrete actions are represented by a one-hot encoding, where the Gaussian noise is added to each dimension. Action noise and the clipping is scaled according the range of the action space.

We modify the TD3 loss in a few ways. Firstly, following numerous prior work across benchmarks (Hessel et al., 2018; Barth-Maron et al., 2018; Yarats et al., 2022; Schwarzer et al., 2023), we predict multi-step returns over a horizon $H _ { Q }$ . Secondly, we use the Huber loss instead of meansquared error to eliminate bias from prioritized sampling (Fujimoto et al., 2020). Finally, the target value is normalized according to the average absolute reward r¯ in the replay buffer:

$$
\mathcal {L} _ {\text { Value }} \left(\tilde {Q} _ {i}\right) := \operatorname{Huber} \left(\tilde {Q} _ {i}, \frac {1}{\bar {r}} \left(\sum_ {t = 0} ^ {H _ {Q} - 1} \gamma^ {t} r _ {t} + \gamma^ {H _ {Q}} \tilde {Q} _ {j} ^ {\prime}\right)\right), \quad \tilde {Q} _ {j} ^ {\prime} := \bar {r} ^ {\prime} \min _ {j = 1, 2} Q _ {\theta_ {j} ^ {\prime}} \left(\mathbf {z} _ {s _ {H _ {Q}} a _ {H _ {Q}, \pi}}\right). \tag {19}
$$

The value $\bar { r } ^ { \prime }$ captures the target average absolute reward, which is the scaling factor used to the most recently copied value functions $Q _ { \theta _ { i } ^ { \prime } }$ . This value is updated simultaneously with the target networks $\bar { r } ^ { \prime }  \bar { r } .$ Maintaining a consistent reward scale keeps the loss magnitude constant across different benchmarks, thus improving the robustness of a single set of hyperparameters.

# 4.2.3 POLICY

For both continuous and discrete action spaces, the policy is updated using the deterministic policy gradient (Silver et al., 2014):

$$
\mathcal {L} _ {\text { Policy }} \left(a _ {\pi}\right) := - 0. 5 \sum_ {i = \{1, 2 \}} \tilde {Q} _ {i} \left(\mathbf {z} _ {s a _ {\pi}}\right) + \lambda_ {\text { pre   -   activ }} \mathbf {z} _ {\pi} ^ {2}, \quad \text { where } a _ {\pi} = \operatorname{activ} \left(\mathbf {z} _ {\pi}\right). \tag {20}
$$

To make the loss universal between action spaces, we use Gumbel-Softmax (Jang et al., 2017; Lowe et al., 2017; Cianflone et al., 2019) for discrete actions, and Tanh for continuous actions. A small regularization penalty is added to the square of the pre-activations ${ \bf z } _ { \pi }$ before the policy’s final activation to help avoid local minima when the reward, and value, is sparse (Bjorck et al., 2021).

For exploration, Gaussian noise is added to each dimension of the action (or one-hot encoding of the action). Similar to Equation 18, the resulting action vector is clipped to the range of the action space for continuous actions. For discrete actions, the final action is determined by the argmax operation.

![](images/fb14e2cd5b6006b638b0e512a3bba64c893f33808da7b9cd0e6392c9454f4e1e.jpg)  
MR.Q   
DreamerV3   
TD-MPC2   
PPO   
TD7   
DrQ-v2  
Rainbow   
DQN

Figure 2: Aggregate learning curves. Average performance over each benchmark. Results are over 10 seeds. The shaded area captures a 95% stratified bootstrap confidence interval. Due to action repeat, 500k time steps in DMC correspond to 1M frames in the original environment and 2.5M time steps in Atari corresponds to 10M frames in the original environment.

# 5 EXPERIMENTS

We evaluate MR.Q on four popular RL benchmarks and 118 environments, and compare its performance against strong domain-specific baselines, general model-based approaches, DreamerV3 (Hafner et al., 2023) and TD-MPC2 (Hansen et al., 2024), and a general model-free algorithm, PPO (Schulman et al., 2017). Rather than establish MR.Q as the state-of-the-art approach in any particular benchmark, our objective is to demonstrate its broad applicability and effectiveness across a diverse set of tasks with a single set of hyperparameters. The baselines use author-suggested default hyperparameters and are fixed across environments. Additional details can be found in Appendix B.

# 5.1 MAIN RESULTS

Aggregate learning curves are displayed in Figure 2, with full results displayed in Appendix C.

Gym - Locomotion. This subset of the Gym benchmark (Brockman et al., 2016; Towers et al., 2024) considers 5 locomotion tasks in the MuJoCo simulator (Todorov et al., 2012) with continuous actions and low level states. Agents are trained for 1M time steps without any environment preprocessing. We evaluate against three baselines: TD7 (Fujimoto et al., 2024), a state-of-the-art (or near) approach for this benchmark, as well as TD-MPC2, DreamerV3, and PPO. To aggregate results, we normalize using the performance of TD3 (Fujimoto et al., 2018).

DMC - Proprioceptive. The DeepMind Control suite (DMC) (Tassa et al., 2018) is a collection of continuous control robotics tasks built on the MuJoCo simulator. These tasks use the proprioceptive states as the observation space, meaning that the input is a vector, and limit the total reward for each episode at 1000, making it easy to aggregate results. We report results on all 28 default tasks that were used by either TD-MPC2 or DreamerV3. Agents are trained for 500k time steps, equivalent to 1M frames in the original environment due to action repeat. For comparison, we evaluate against the same three algorithms as in the Gym benchmark, with TD-MPC2 considered state-of-the-art (or near) for this benchmark. We also include TD7 due to its strong performance in the Gym benchmark.

DMC - Visual. The visual DMC benchmark includes the same 28 tasks as the proprioceptive benchmark, but uses image-based observations instead. Agents are trained for 500k time steps. For baselines, we include DrQ-v2 (Yarats et al., 2022), given its state-of-the-art (or near) performance in model-free RL, alongside TD-MPC2, DreamerV3, and PPO.

Atari. The Atari benchmark is built on the Arcade Learning Environment (Bellemare et al., 2013). This benchmark uses pixel observations and discrete actions and includes the 57 games used by DreamerV3. We follow standard preprocessing steps, including sticky actions (Machado et al., 2018) (full details in Appendix B.3). Agents are trained for 2.5M time steps (equivalent to 10M frames), a setting which has been considered by prior work (Sokar et al., 2023). For comparison, we evaluate against three baselines: the model-based approach DreamerV3, as well as model-free approaches, DQN (Mnih et al., 2015), Rainbow (Hessel et al., 2018), and PPO. Results are aggregated by normalizing scores against human performance.

Discussion. Throughout our experiments, we find the presence of “no free lunch”, where the topperforming baseline in one benchmark fails to replicate its success in another. Regardless, MR.Q achieves the highest performance in both DMC benchmarks, showcasing its ability to handle different observation spaces. Although it falls slightly behind TD7 in the Gym benchmark, MR.Q is the strongest method overall across all continuous control benchmarks. In Atari, while DreamerV3 outperforms MR.Q, it relies on a model with 40 times more parameters and struggles comparatively in the remaining benchmarks. When compared to the model-free baselines, MR.Q surpasses PPO, DQN, and Rainbow, demonstrating its effectiveness with discrete action spaces.

# 5.2 DESIGN STUDY

To better understand the impact of certain design choices and hyperparameters, we attempt variations of MR.Q, and report the aggregate results in Table 2.

Table 2: Design study. Average difference in normalized performance from varying design choices across each benchmark over 5 seeds. Negative changes are highlighted lightly $[ - 0 . 0 1 , - 0 . 2 )$ . Damaging changes are highlighted moderately [−0.2, −0.5). Catastrophic changes are highlighted boldly ( −0.5). Positive changes are similarly highlighted ( 0.01). 

<table><tr><td>Design</td><td>Gym - Locomotion TD3-Normalized</td><td>DMC - Proprioceptive Reward (1k)</td><td>DMC - Visual Reward (1k)</td><td>Atari - 1M Human-Normalized</td></tr><tr><td colspan="5">Relaxations</td></tr><tr><td>Linear value function</td><td>-1.17 [-1.19, -1.15]</td><td>-0.58 [-0.59, -0.56]</td><td>-0.41 [-0.42, -0.39]</td><td>-1.35 [-1.41, -1.29]</td></tr><tr><td>Dynamics target</td><td>-0.10 [-0.17, -0.04]</td><td>-0.15 [-0.15, -0.15]</td><td>-0.05 [-0.05, -0.04]</td><td>-0.38 [-0.81, 0.05]</td></tr><tr><td>No target encoder</td><td>-0.53 [-0.60, -0.46]</td><td>-0.35 [-0.35, -0.34]</td><td>-0.15 [-0.15, -0.15]</td><td>-0.86 [-0.89, -0.83]</td></tr><tr><td>Revert</td><td>-1.47 [-1.54, -1.39]</td><td>-0.72 [-0.73, -0.72]</td><td>-0.52 [-0.52, -0.51]</td><td>-1.69 [-1.70, -1.67]</td></tr><tr><td>Non-linear model</td><td>-0.01 [-0.07, 0.03]</td><td>-0.00 [-0.02, 0.01]</td><td>-0.01 [-0.02, -0.00]</td><td>-0.07 [-0.32, 0.18]</td></tr><tr><td colspan="5">Loss functions</td></tr><tr><td>MSE reward loss</td><td>0.10 [-0.02, 0.19]</td><td>-0.06 [-0.08, -0.05]</td><td>-0.05 [-0.07, -0.04]</td><td>-0.79 [-0.86, -0.73]</td></tr><tr><td>No reward scaling</td><td>-0.04 [-0.09, 0.02]</td><td>-0.01 [-0.02, 0.00]</td><td>-0.00 [-0.01, 0.01]</td><td>0.18 [-0.25, 0.56]</td></tr><tr><td>No min</td><td>-0.09 [-0.16, -0.01]</td><td>-0.01 [-0.02, 0.01]</td><td>0.00 [-0.01, 0.01]</td><td>0.13 [-0.10, 0.58]</td></tr><tr><td>No LAP</td><td>-0.10 [-0.24, -0.00]</td><td>0.00 [-0.00, 0.01]</td><td>-0.01 [-0.02, -0.01]</td><td>-0.13 [-0.38, 0.14]</td></tr><tr><td>No MR</td><td>-0.56 [-0.69, -0.43]</td><td>-0.19 [-0.19, -0.18]</td><td>-0.07 [-0.09, -0.03]</td><td>-0.78 [-0.88, -0.69]</td></tr><tr><td colspan="5">Horizons</td></tr><tr><td>1-step return</td><td>-0.33 [-0.46, -0.21]</td><td>-0.04 [-0.05, -0.02]</td><td>-0.03 [-0.03, -0.02]</td><td>-0.70 [-0.81, -0.59]</td></tr><tr><td>No unroll</td><td>0.07 [0.01, 0.14]</td><td>-0.01 [-0.01, -0.00]</td><td>-0.04 [-0.06, -0.01]</td><td>-0.33 [-0.41, -0.28]</td></tr></table>

Relaxations. In Section 4.1, we outlined a loss (Equation 9) that, if globally minimized, would provide features that are linear with the true value function. MR.Q in practice relaxes this theoretical result by modifying the loss and using a non-linear value function. In Linear value function, we replace the non-linear value function with a linear function. In Dynamics target, we replace the state embedding dynamics target with a state-action embedding $\bar { \bf z } _ { s ^ { \prime } a ^ { \prime } }$ determined from the target state-action encoder $g _ { \omega }$ . In No target encoder, we use the current encoder to generate the dynamics target ${ \bf z } _ { s ^ { \prime } a ^ { \prime } }$ , and jointly optimize it within the encoder loss. In Revert, we consider all of the aforementioned changes simultaneously, using linear value functions and setting the dynamics target as a state-action embedding determined by the current encoder. In Non-linear model, we replace the linear MDP predictor with individual networks that predict each component separately from $\mathbf { z } _ { s a }$ .

Loss functions. MR.Q’s loss functions use several unconventional choices. In MSE reward loss, we replace the categorical loss function on the predicted reward in Equation 15 with the meansquared error (MSE). In No reward scaling, we remove the reward scaling in Equation 19, setting $\bar { r } = \bar { r } ^ { \prime } = 1$ . In No min, we take the mean over the target value functions instead of the minimum in Equation 19. In No LAP, we remove prioritized sampling (Fujimoto et al., 2020) and use the MSE instead of the Huber loss in the value update. Lastly, in No MR, we remove model-based representation learning and train the encoder end-to-end with the value function.

Horizons. Finally, we consider the role of extended predictions. In 1-step return, we remove multi-step value predictions and use TD learning. In No unroll, we remove the dynamics unrolling in Equation 14, by setting the encoder horizon $\bar { H } _ { \mathrm { E n c } } = 1$ .

Discussion. The results of our design study show the benefit of balancing theory with practical relaxations. The experiments further validate our design choices and hyperparameters. We highlight two results in particular: (1) increasing the model capacity in the “non-linear model” experiment, does not improve performance. This outcome suggests that maintaining an approximately linear relationship with the value function can be more impactful than increased capacity. (2) Our study also reveals a key distinction between the Gym and Atari benchmarks—while the “MSE reward loss” and “No unroll” variants offer moderate performance gains in Gym, they significantly degrade performance in Atari. This discrepancy highlights how hyperparameters can overfit to individual benchmarks, emphasizing the importance of evaluating algorithms across multiple benchmarks.

# 6 DISCUSSION AND CONCLUSION

This paper introduces MR.Q, a general model-free deep RL algorithm that achieves strong performance across diverse benchmarks and environments. Drawing inspiration from the theory of model-based representation learning, MR.Q demonstrates that model-free deep RL is a promising avenue for building general-purpose algorithms that achieve high performance across environments, while being simpler and less expensive than model-based alternatives.

Our work also reveals insights on which design choices matter when building general-purpose model-free deep RL algorithms and how common benchmarks respond to these design choices.

Model-based and model-free RL. MR.Q integrates model-based objectives with a model-free backbone during training, effectively blurring the boundary between traditional model-based and model-free RL. While MR.Q could be extended to the model-based setting by incorporating planning or simulated trajectories with the state-action encoder, these components can add significant execution time and increase the overall complexity and tuning required by a method. Moreover, the performance of MR.Q in these common RL benchmarks demonstrates that these model-based components may be simply unnecessary—suggesting that the representation itself could be the most valuable aspect of model-based learning, even in methods that do use planning. This argument is echoed by DreamerV3 and TD-MPC2, which rely on short planning horizons and trajectory generation, while including both value functions and traditional model-free policy updates. As such, it may be necessary to examine more complex settings, to reliably see a benefit from model-based search or planning, e.g., (Silver et al., 2016).

Universality of RL benchmarks. Our results demonstrate that there is a striking lack of positive transfer between benchmarks. For example, despite the similarities in tasks and the same underlying MuJoCo simulator, the top performers in Gym and DMC fail to replicate their success on the opposing benchmark. Similarly, although DreamerV3 excels at Atari, these performance benefits do not translate to continuous control environments, underperforming TD3 in Gym and outright failing to learn the Dog and Humanoid tasks in DMC (see Appendix C). These findings show the limitations of single-benchmark evaluations, indicating that success on one benchmark may not translate easily to others, and highlights the need for more comprehensive benchmarks.

Limitations. MR.Q is only the first step towards a new generation of general-purpose model-free deep RL algorithms. Many challenges remains for a fully general algorithm. In particular, MR.Q is not equipped to handle settings such as hard exploration tasks or non-Markovian environments. Another limitation is our evaluation only considers standard RL benchmarks. Although this allows direct comparison with other methods, established algorithms such as PPO have demonstrated their effectiveness in highly unique settings, such as team video games (Berner et al., 2019), drone racing (Kaufmann et al., 2023), and large language models (Achiam et al., 2023; Touvron et al., 2023). To demonstrate similar versatility, new algorithms must undergo the same rigorous testing across a range of tasks that is beyond the scope of any single study.

As the community continues to push the boundaries of what is possible with deep RL, we believe that building simpler general-purpose algorithms has the potential to make this technology more accessible to a wider audience, ultimately enabling users to train agents with ease. Perhaps one day — with just the click of a button.

# ACKNOWLEDGMENTS

We would like to thank Brandon Amos, Mikhael Henaff, Luis Pineda, Paria Rashidinejad, and Qinqing Zheng for insightful discussions and comments.

# REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
Alekh Agarwal, Sham Kakade, Akshay Krishnamurthy, and Wen Sun. Flambe: Structural complexity and representation learning of low rank mdps. Advances in neural information processing systems, 33:20095–20107, 2020.   
Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15619–15629, 2023.   
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.   
Thomas Back. Evolutionary algorithms in theory and practice: evolution strategies, evolutionary programming, genetic algorithms. Oxford university press, 1996.   
Adria Puigdom \` enech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, \` Zhaohan Daniel Guo, and Charles Blundell. Agent57: Outperforming the atari human benchmark. In International Conference on Machine Learning, pp. 507–517. PMLR, 2020.   
Leemon Baird. Residual algorithms: Reinforcement learning with function approximation. In Machine Learning Proceedings 1995, pp. 30–37. Elsevier, 1995.   
Gabriel Barth-Maron, Matthew W Hoffman, David Budden, Will Dabney, Dan Horgan, Dhruva TB, Alistair Muldal, Nicolas Heess, and Timothy Lillicrap. Distributional policy gradients. International Conference on Learning Representations, 2018.   
Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47: 253–279, 2013.   
Richard Bellman. A markovian decision process. Journal of mathematics and mechanics, pp. 679– 684, 1957.   
Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemyslaw Debiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, et al. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680, 2019.   
Johan Bjorck, Carla P Gomes, and Kilian Q Weinberger. Is high variance unavoidable in rl? a case study in continuous control. In International Conference on Learning Representations, 2021.   
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.   
Pablo Samuel Castro. Scalable methods for computing state similarity in deterministic markov decision processes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 10069–10076, 2020.   
Pablo Samuel Castro, Subhodeep Moitra, Carles Gelada, Saurabh Kumar, and Marc G Bellemare. Dopamine: A research framework for deep reinforcement learning. arXiv preprint arXiv:1812.06110, 2018.

Pablo Samuel Castro, Tyler Kastner, Prakash Panangaden, and Mark Rowland. Mico: Improved representations via sampling-based state similarity for markov decision processes. Advances in Neural Information Processing Systems, 34:30113–30126, 2021.   
Wesley Chung, Somjit Nath, Ajin Joseph, and Martha White. Two-timescale networks for nonlinear value function approximation. In International Conference on Learning Representations, 2019.   
Andre Cianflone, Zafarali Ahmed, Riashat Islam, Avishek Joey Bose, and William L Hamilton. Discrete off-policy policy gradient using continuous relaxations, 2019.   
Djork-Arne Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network ´ learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015.   
Peter Dayan. Improving generalization for temporal difference learning: The successor representation. Neural Computation, 5(4):613–624, 1993.   
Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry. Implementation matters in deep rl: A case study on ppo and trpo. In International Conference on Learning Representations, 2020.   
Norm Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite markov decision processes. In UAI, volume 4, pp. 162–169, 2004.   
Norm Ferns, Prakash Panangaden, and Doina Precup. Bisimulation metrics for continuous markov decision processes. SIAM Journal on Computing, 40(6):1662–1714, 2011.   
Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, and Pieter Abbeel. Deep spatial autoencoders for visuomotor learning. In 2016 IEEE International Conference on Robotics and Automation (ICRA), pp. 512–519. IEEE, 2016.   
Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning, volume 80, pp. 1587– 1596. PMLR, 2018.   
Scott Fujimoto, David Meger, and Doina Precup. An equivalence between loss functions and nonuniform sampling in experience replay. Advances in Neural Information Processing Systems, 33, 2020.   
Scott Fujimoto, David Meger, and Doina Precup. A deep reinforcement learning approach to marginalized importance sampling with the successor representation. In Proceedings of the 38th International Conference on Machine Learning, volume 139, pp. 3518–3529. PMLR, 2021.   
Scott Fujimoto, David Meger, Doina Precup, Ofir Nachum, and Shixiang Shane Gu. Why should i trust you, bellman? The Bellman error is a poor replacement for value error. In International Conference on Machine Learning, volume 162, pp. 6918–6943. PMLR, 2022.   
Scott Fujimoto, Wei-Di Chang, Edward J. Smith, Shixiang Shane Gu, Doina Precup, and David Meger. For SALE: State-action representation learning for deep reinforcement learning. In Thirtyseventh Conference on Neural Information Processing Systems, 2024.   
Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, and Marc G Bellemare. Deepmdp: Learning continuous latent space models for representation learning. In International Conference on Machine Learning, pp. 2170–2179. PMLR, 2019.   
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249–256. JMLR Workshop and Conference Proceedings, 2010.   
Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin Tallec, Pierre Richemond, Elena ´ Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

Zhaohan Daniel Guo, Bernardo Avila Pires, Bilal Piot, Jean-Bastien Grill, Florent Altche, R ´ emi ´ Munos, and Mohammad Gheshlaghi Azar. Bootstrap latent-predictive representations for multitask reinforcement learning. In International Conference on Machine Learning, pp. 3875–3886. PMLR, 2020.   
Zhaohan Daniel Guo, Shantanu Thakoor, Miruna Pˆıslar, Bernardo Avila Pires, Florent Altche,´ Corentin Tallec, Alaa Saade, Daniele Calandriello, Jean-Bastien Grill, Yunhao Tang, et al. Byolexplore: Exploration by bootstrapped prediction. Advances in neural information processing systems, 35:31855–31870, 2022.   
David Ha and Jurgen Schmidhuber. World models. ¨ arXiv preprint arXiv:1803.10122, 2018.   
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning, volume 80, pp. 1861–1870. PMLR, 2018.   
Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pp. 2555–2565. PMLR, 2019.   
Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.   
Nicklas Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In The Twelfth International Conference on Learning Representations, 2024.   
Nicklas A Hansen, Hao Su, and Xiaolong Wang. Temporal difference learning for model predictive control. In International Conference on Machine Learning, pp. 8387–8406. PMLR, 2022.   
Charles R. Harris, K. Jarrod Millman, Stefan J. van der Walt, Ralf Gommers, Pauli Virtanen, David ´ Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernandez ´ del R´ıo, Mark Wiebe, Pearu Peterson, Pierre Gerard-Marchant, Kevin Sheppard, Tyler Reddy, ´ Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array programming with NumPy. Nature, 585(7825):357–362, September 2020.   
Ammar Haydari and Yasin Yılmaz. Deep reinforcement learning for intelligent transportation systems: A survey. IEEE Transactions on Intelligent Transportation Systems, 23(1):11–32, 2020.   
Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In Thirty-second AAAI conference on artificial intelligence, 2018.   
Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, and Weixun Wang. The 37 implementation details of proximal policy optimization. In ICLR Blog Track, 2022.   
Julian Ibarz, Jie Tan, Chelsea Finn, Mrinal Kalakrishnan, Peter Pastor, and Sergey Levine. How to train your robot with deep reinforcement learning: lessons we have learned. The International Journal of Robotics Research, 40(4-5):698–721, 2021.   
Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations, 2017.   
Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on learning theory, pp. 2137–2143. PMLR, 2020.   
Maximilian Karl, Maximilian Soelch, Justin Bayer, and Patrick van der Smagt. Deep variational bayes filters: Unsupervised learning of state space models from raw data. stat, 1050:3, 2017.   
Elia Kaufmann, Leonard Bauersfeld, Antonio Loquercio, Matthias Muller, Vladlen Koltun, and ¨ Davide Scaramuzza. Champion-level drone racing using deep reinforcement learning. Nature, 620(7976):982–987, 2023.

Kyungsoo Kim, Jeongsoo Ha, and Yusung Kim. Self-predictive dynamics for generalization of vision-based reinforcement learning. In IJCAI, pp. 3150–3156, 2022.   
Arsenii Kuznetsov, Pavel Shvechikov, Alexander Grishin, and Dmitry Vetrov. Controlling overestimation bias with truncated mixture of continuous distributional quantile critics. In International Conference on Machine Learning, pp. 5556–5566. PMLR, 2020.   
Alex X Lee, Anusha Nagabandi, Pieter Abbeel, and Sergey Levine. Stochastic latent actor-critic: Deep reinforcement learning with a latent variable model. Advances in Neural Information Processing Systems, 33:741–752, 2020.   
Nir Levine, Tom Zahavy, Daniel J Mankowitz, Aviv Tamar, and Shie Mannor. Shallow updates for deep reinforcement learning. Advances in Neural Information Processing Systems, 30, 2017.   
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.   
Michael Littman and Richard S Sutton. Predictive representations of state. Advances in neural information processing systems, 14, 2001.   
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.   
Ryan Lowe, Yi I Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch. Multiagent actor-critic for mixed cooperative-competitive environments. Advances in neural information processing systems, 30, 2017.   
Nguyen Cong Luong, Dinh Thai Hoang, Shimin Gong, Dusit Niyato, Ping Wang, Ying-Chang Liang, and Dong In Kim. Applications of deep reinforcement learning in communications and networking: A survey. IEEE communications surveys & tutorials, 21(4):3133–3174, 2019.   
Marlos C Machado, Marc G Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and Michael Bowling. Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents. Journal of Artificial Intelligence Research, 61:523–562, 2018.   
Trevor McInroe, Lukas Schafer, and Stefano V Albrecht. Learning temporally-consistent represen-¨ tations for data-efficient reinforcement learning. arXiv preprint arXiv:2110.04935, 2021.   
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.   
Jelle Munk, Jens Kober, and Robert Babuska. Learning state representation for deep actor-critic ˇ control. In 2016 IEEE 55th Conference on Decision and Control (CDC), pp. 4667–4673. IEEE, 2016.   
Tianwei Ni, Benjamin Eysenbach, Erfan Seyedsalehi, Michel Ma, Clement Gehring, Aditya Mahajan, and Pierre-Luc Bacon. Bridging state and history representations: Understanding selfpredictive rl. In The Twelfth International Conference on Learning Representations, 2024.   
Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard L Lewis, and Satinder Singh. Action-conditional video prediction using deep networks in atari games. Advances in neural information processing systems, 28, 2015.   
Ronald Parr, Lihong Li, Gavin Taylor, Christopher Painter-Wakefield, and Michael L Littman. An analysis of linear models, linear value-function approximation, and feature selection for reinforcement learning. In Proceedings of the 25th international conference on Machine learning, pp. 752–759, 2008.   
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. In Advances in Neural Information Processing Systems, pp. 8024–8035, 2019.

Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1–8, 2021. URL http://jmlr.org/papers/v22/20-1364.html.   
Balaraman Ravindran. An algebraic approach to abstraction in reinforcement learning. University of Massachusetts Amherst, 2004.   
Balaraman Ravindran and Andrew G Barto. Model minimization in hierarchical reinforcement learning. In Abstraction, Reformulation, and Approximation: 5th International Symposium, SARA 2002 Kananaskis, Alberta, Canada August 2–4, 2002 Proceedings 5, pp. 196–211. Springer, 2002.   
Ingo Rechenberg. Evolutionsstrategien. In Simulationsmethoden in der Medizin und Biologie: Workshop, Hannover, 29. Sept.–1. Okt. 1977, pp. 83–114. Springer, 1978.   
Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, ´ Gabriel Barth-maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Ec- ´ cles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, and Nando de Freitas. A generalist agent. Transactions on Machine Learning Research, 2022. ISSN 2835-8856.   
Tongzheng Ren, Tianjun Zhang, Csaba Szepesvari, and Bo Dai. A free lunch from the noise: Prov- ´ able and practical exploration for representation learning. In Uncertainty in Artificial Intelligence, pp. 1686–1696. PMLR, 2022.   
Tongzheng Ren, Chenjun Xiao, Tianjun Zhang, Na Li, Zhaoran Wang, Dale Schuurmans, Bo Dai, et al. Latent variable representation for reinforcement learning. In The Eleventh International Conference on Learning Representations, 2023.   
Sahand Rezaei-Shoshtari, Rosie Zhao, Prakash Panangaden, David Meger, and Doina Precup. Continuous mdp homomorphisms and homomorphic policy gradient. In Advances in Neural Information Processing Systems, 2022.   
Reuven Y Rubinstein. Optimization of computer simulation models with rare events. European Journal of Operational Research, 99(1):89–112, 1997.   
Gavin A Rummery and Mahesan Niranjan. On-line Q-learning using connectionist systems, volume 37. University of Cambridge, Department of Engineering Cambridge, UK, 1994.   
Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864, 2017.   
Aidan Scannell, Kalle Kujanpa¨a, Yi Zhao, Mohammadreza Nakhaei, Arno Solin, and Joni Pajari-¨ nen. iqrl–implicitly quantized representations for sample-efficient reinforcement learning. arXiv preprint arXiv:2406.02696, 2024.   
Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. In International Conference on Learning Representations, Puerto Rico, 2016.   
Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.   
Julian Schrittwieser, Thomas Hubert, Amol Mandhane, Mohammadamin Barekatain, Ioannis Antonoglou, and David Silver. Online and offline reinforcement learning by planning with a learned model. Advances in Neural Information Processing Systems, 34:27580–27591, 2021.   
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning, pp. 1889–1897, 2015.   
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Max Schwarzer, Ankesh Anand, Rishab Goel, R Devon Hjelm, Aaron Courville, and Philip Bachman. Data-efficient reinforcement learning with self-predictive representations. In International Conference on Learning Representations, 2020.   
Max Schwarzer, Johan Samir Obando Ceron, Aaron Courville, Marc G Bellemare, Rishabh Agarwal, and Pablo Samuel Castro. Bigger, better, faster: Human-level atari with human-level efficiency. In International Conference on Machine Learning, pp. 30365–30380. PMLR, 2023.   
Younggyo Seo, Kimin Lee, Stephen L James, and Pieter Abbeel. Reinforcement learning with action-free pre-training from videos. In International Conference on Machine Learning, pp. 19561–19579. PMLR, 2022.   
Dmitry Shribak, Chen-Xiao Gao, Yitong Li, Chenjun Xiao, and Bo Dai. Diffusion spectral representation for reinforcement learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.   
David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. In International Conference on Machine Learning, pp. 387–395, 2014.   
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484–489, 2016.   
Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, and Utku Evci. The dormant neuron phenomenon in deep reinforcement learning. In International Conference on Machine Learning, pp. 32145–32168. PMLR, 2023.   
Zhao Song, Ronald E Parr, Xuejun Liao, and Lawrence Carin. Linear feature encoding for reinforcement learning. Advances in neural information processing systems, 29, 2016.   
Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3 (1):9–44, 1988.   
Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction, volume 1. MIT press Cambridge, 1998.   
Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.   
Yunhao Tang, Zhaohan Daniel Guo, Pierre Harvey Richemond, Bernardo Avila Pires, Yash Chandak, Remi Munos, Mark Rowland, Mohammad Gheshlaghi Azar, Charline Le Lan, Clare Lyle, ´ et al. Understanding self-predictive learning for reinforcement learning. In International Conference on Machine Learning, pp. 33632–33656. PMLR, 2023.   
Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018.   
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5026–5033. IEEE, 2012.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulao, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard˜ interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.

Elise van der Pol, Thomas Kipf, Frans A Oliehoek, and Max Welling. Plannable approximations to mdp homomorphisms: Equivariance under actions. In Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems, pp. 1431–1439, 2020a.   
Elise van der Pol, Daniel Worrall, Herke van Hoof, Frans Oliehoek, and Max Welling. Mdp homomorphic networks: Group symmetries in reinforcement learning. Advances in Neural Information Processing Systems, 33:4199–4210, 2020b.   
Herke Van Hoof, Nutan Chen, Maximilian Karl, Patrick van der Smagt, and Jan Peters. Stable reinforcement learning with autoencoders for tactile and visual data. In 2016 IEEE/RSJ international conference on intelligent robots and systems (IROS), pp. 3928–3934. IEEE, 2016.   
Guido Van Rossum and Fred L Drake Jr. Python tutorial. Centrum voor Wiskunde en Informatica Amsterdam, The Netherlands, 1995.   
Shengjie Wang, Shaohuai Liu, Weirui Ye, Jiacheng You, and Yang Gao. Efficientzero v2: Mastering discrete and continuous control with limited data. arXiv preprint arXiv:2403.00564, 2024.   
Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas. Dueling network architectures for deep reinforcement learning. In International Conference on Machine Learning, pp. 1995–2003, 2016.   
Christopher John Cornish Hellaby Watkins. Learning from delayed rewards. PhD thesis, King’s College, Cambridge, 1989.   
Manuel Watter, Jost Springenberg, Joschka Boedecker, and Martin Riedmiller. Embed to control: A locally linear latent dynamics model for control from raw images. Advances in neural information processing systems, 28, 2015.   
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229–256, 1992.   
Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Mastering visual continuous control: Improved data-augmented reinforcement learning. In International Conference on Learning Representations, 2022.   
Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, and Yang Gao. Mastering atari games with limited data. Advances in neural information processing systems, 34:25476–25488, 2021.   
Tao Yu, Cuiling Lan, Wenjun Zeng, Mingxiao Feng, Zhizheng Zhang, and Zhibo Chen. Playvirtual: Augmenting cycle-consistent virtual trajectories for reinforcement learning. Advances in Neural Information Processing Systems, 34:5276–5289, 2021.   
Tao Yu, Zhizheng Zhang, Cuiling Lan, Yan Lu, and Zhibo Chen. Mask-based latent reconstruction for reinforcement learning. Advances in Neural Information Processing Systems, 35:25117– 25131, 2022.   
Hongyu Zang, Xin Li, and Mingzhong Wang. Simsr: Simple distance-based state representations for deep reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 8997–9005, 2022.   
Amy Zhang, Harsh Satija, and Joelle Pineau. Decoupling dynamics and reward for transfer learning. arXiv preprint arXiv:1804.10689, 2018.   
Amy Zhang, Rowan Thomas McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invariant representations for reinforcement learning without reconstruction. In International Conference on Learning Representations, 2020.   
Tianjun Zhang, Tongzheng Ren, Mengjiao Yang, Joseph Gonzalez, Dale Schuurmans, and Bo Dai. Making linear mdps practical via contrastive representation learning. In International Conference on Machine Learning, pp. 26447–26466. PMLR, 2022.   
Yi Zhao, Wenshuai Zhao, Rinu Boney, Juho Kannala, and Joni Pajarinen. Simplified temporal consistency reinforcement learning. In International Conference on Machine Learning, pp. 42227– 42246. PMLR, 2023.

Ruijie Zheng, Xiyao Wang, Yanchao Sun, Shuang Ma, Jieyu Zhao, Huazhe Xu, Hal Daume III, ´ and Furong Huang. Taco: Temporal latent action-driven contrastive loss for visual reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.   
Luisa Zintgraf, Sebastian Schulze, Cong Lu, Leo Feng, Maximilian Igl, Kyriacos Shiarlis, Yarin Gal, Katja Hofmann, and Shimon Whiteson. Varibad: Variational bayes-adaptive deep rl via meta-learning. Journal of Machine Learning Research, 22(289):1–39, 2021.

# Appendix

A Proofs 19   
B Experimental Details 23

B.1 Hyperparameters 23   
B.2 Network Architecture . . 24   
B.3 Environments 26   
B.4 Baselines . . . 28   
B.5 Software Versions . . . . 28

C Complete Main Results 29

C.1 Gym 29   
C.2 DMC - Proprioceptive 30   
C.3 DMC - Visual 32   
C.4 Atari 34

D Complete Ablation Results 36

D.1 Gym 36   
D.2 DMC - Proprioceptive 37   
D.3 DMC - Visual 39   
D.4 Atari 41

# A PROOFS

Theorem 1. The fixed point of the model-free approach (Equation 4) and the solution of the modelbased approach (Equation 5) are the same.

Proof. Let Z be a matrix containing state-action embeddings $\mathbf { z } _ { s a }$ for each state-action pair $( s , a )$ $S { \times } \bar { A }$ . Let $Z ^ { \prime }$ be the corresponding matrix of next state-action embeddings $\mathbf { z } _ { s ^ { \prime } a ^ { \prime } }$ ( ). Let R be the vector of the corresponding rewards $r ( s , a )$ .

The linear semi-gradient TD update:

$$
\mathbf {w} _ {t + 1} := \mathbf {w} _ {t} - \alpha Z ^ {\top} \left(Z \mathbf {w} _ {t} - \left(R + \gamma Z ^ {\prime} \mathbf {w} _ {t}\right)\right) \tag {21}
$$

$$
= \mathbf {w} _ {t} - \alpha Z ^ {\top} Z \mathbf {w} _ {t} + \alpha Z ^ {\top} R + \alpha \gamma Z ^ {\top} Z ^ {\prime} \mathbf {w} _ {t} \tag {22}
$$

$$
= \left(I - \alpha \left(Z ^ {\top} Z - \gamma Z ^ {\top} Z ^ {\prime}\right)\right) \mathbf {w} _ {t} + \alpha Z ^ {\top} R \tag {23}
$$

$$
= (I - \alpha A) \mathbf {w} _ {t} + \alpha B, \tag {24}
$$

where $A : = Z ^ { \intercal } Z - \gamma Z ^ { \intercal } Z ^ { \prime }$ and $B : = Z ^ { \intercal } R$ .

The fixed point of the system:

$$
\mathbf {w} _ {\mathrm{mf}} = (I - \alpha A) \mathbf {w} _ {\mathrm{mf}} + \alpha B \tag {25}
$$

$$
\mathbf {w} _ {\mathrm{mf}} - (I - \alpha A) \mathbf {w} _ {\mathrm{mf}} = \alpha B \tag {26}
$$

$$
\alpha A \mathbf {w} _ {\mathrm{mf}} = \alpha B \tag {27}
$$

$$
\mathbf {w} _ {\mathrm{mf}} = A ^ {- 1} B. \tag {28}
$$

The least squares solution to $W _ { p }$ and ${ \bf w } _ { r }$

$$
W _ {p} := \left(Z ^ {\top} Z\right) ^ {- 1} Z ^ {\top} Z ^ {\prime} \tag {29}
$$

$$
\mathbf {w} _ {r} := \left(Z ^ {\top} Z\right) ^ {- 1} Z ^ {\top} R \tag {30}
$$

By rolling out $W _ { p }$ and $\mathbf { w } _ { r } .$ , we arrive at a model-based solution:

$$
Q := Z \mathbf {w} _ {\mathrm{mb}} = Z \sum_ {t = 0} ^ {\infty} \gamma^ {t} W _ {p} ^ {t} \mathbf {w} _ {r}. \tag {31}
$$

Simplify $\mathbf { w } _ { \mathrm { m b } }$ :

$$
\mathbf {w} _ {\mathrm{mb}} := \sum_ {t = 0} ^ {\infty} \gamma^ {t} W _ {p} ^ {t} \mathbf {w} _ {r} \tag {32}
$$

$$
\mathbf {w} _ {\mathrm{mb}} = \left(I - \gamma W _ {p}\right) ^ {- 1} \mathbf {w} _ {r} \tag {33}
$$

$$
\mathbf {w} _ {\mathrm{mb}} = \left(I - \gamma (Z ^ {\top} Z) ^ {- 1} Z ^ {\top} Z ^ {\prime}\right) ^ {- 1} (Z ^ {\top} Z) ^ {- 1} Z ^ {\top} R \tag {34}
$$

$$
Z ^ {\top} Z \left(I - \gamma \left(Z ^ {\top} Z\right) ^ {- 1} Z ^ {\top} Z ^ {\prime}\right) \mathbf {w} _ {\mathrm{mb}} = Z ^ {\top} R \tag {35}
$$

$$
\left(Z ^ {\top} Z - \gamma Z ^ {\top} Z ^ {\prime}\right) \mathbf {w} _ {\mathrm{mb}} = Z ^ {\top} R \tag {36}
$$

$$
\mathbf {w} _ {\mathrm{mb}} = A ^ {- 1} B \tag {37}
$$

$$
\mathbf {w} _ {\mathrm{mb}} = \mathbf {w} _ {\mathrm{mf}}. \tag {38}
$$

Theorem 2. The value error of the solution described by Theorem 1 is bounded by the accuracy of the estimated dynamics and reward:

$$
\left| \mathrm{VE} (s, a) \right| \leq \frac {1}{1 - \gamma} \max _ {(s, a) \in S \times A} \left(\left| \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - \mathbb {E} _ {r | s, a} [ r ] \right| + \max _ {i} \left| \mathbf {w} _ {i} \right| \sum \left| \mathbf {z} _ {s a} ^ {\top} W _ {p} - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} | s, a} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} \right] \right|\right). (3 9)
$$

Proof. Let w be the solution described in Theorem 1, i.e. $\mathbf { w } = \mathbf { w } _ { \mathrm { m b } } = \mathbf { w } _ { \mathrm { m f } } .$ Let $p ^ { \pi } ( s , a )$ be the ( )discounted state-action visitation distribution according to the policy π starting from the state-action pair $( s , a )$ .

Firstly from Theorem 1, we can show that

$$
\mathbf {w} = \left(I - \gamma W _ {p}\right) ^ {- 1} \mathbf {w} _ {r} \tag {40}
$$

$$
\Rightarrow (I - \gamma W _ {p}) \mathbf {w} = \mathbf {w} _ {r} \tag {41}
$$

$$
\Rightarrow \mathbf {w} - \gamma W _ {p} \mathbf {w} = \mathbf {w} _ {r}. \tag {42}
$$

Simplify $\textstyle \operatorname { V E } ( s , a )$ :

$$
\operatorname{VE} (s, a) := Q (s, a) - Q ^ {\pi} (s, a) \tag {43}
$$

$$
= Q (s, a) - Q ^ {\pi} (s, a) \tag {44}
$$

$$
= Q (s, a) - \mathbb {E} _ {r, s ^ {\prime}, a ^ {\prime}} \left[ r + \gamma Q ^ {\pi} (s ^ {\prime}, a ^ {\prime}) \right] \tag {45}
$$

$$
= Q (s, a) - \mathbb {E} _ {r, s ^ {\prime}, a ^ {\prime}} \left[ r + \gamma \left(Q \left(s ^ {\prime}, a ^ {\prime}\right) - \mathrm{VE} \left(s ^ {\prime}, a ^ {\prime}\right)\right) \right] \tag {46}
$$

$$
= Q (s, a) - \mathbb {E} _ {r, s ^ {\prime}, a ^ {\prime}} [ r + \gamma Q (s ^ {\prime}, a ^ {\prime}) ] + \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} [ \mathrm{VE} (s ^ {\prime}, a ^ {\prime}) ] \tag {47}
$$

$$
= Q (s, a) - \mathbb {E} _ {r, s ^ {\prime}, a ^ {\prime}} \left[ r - \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} + \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} + \gamma Q \left(s ^ {\prime}, a ^ {\prime}\right) \right] + \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathrm{VE} \left(s ^ {\prime}, a ^ {\prime}\right) \right] \tag {48}
$$

$$
= Q (s, a) - \mathbb {E} _ {r, s ^ {\prime}, a ^ {\prime}} \left[ r - \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} + \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} + \gamma \left(\mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \mathbf {w} - \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w} + \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w}\right) \right] \tag {49}
$$

$$
+ \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathrm{VE} (s ^ {\prime}, a ^ {\prime}) \right]
$$

$$
= \mathbf {z} _ {s a} ^ {\top} \mathbf {w} - \mathbb {E} _ {r, s ^ {\prime}, a ^ {\prime}} \left[ r - \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} + \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} + \gamma \left(\mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \mathbf {w} - \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w} + \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w}\right) \right] \tag {50}
$$

$$
+ \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathrm{VE} (s ^ {\prime}, a ^ {\prime}) \right]
$$

$$
\begin{array}{l} = \mathbf {z} _ {s a} ^ {\top} \mathbf {w} - \mathbb {E} _ {r} \left[ r - \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} + \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} \right] - \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \mathbf {w} - \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w} + \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w} \right] \tag {51} \\ + \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathrm{VE} (s ^ {\prime}, a ^ {\prime}) \right] \\ \end{array}
$$

$$
= \mathbf {z} _ {s a} ^ {\top} \mathbf {w} - \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - \gamma \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w} - \mathbb {E} _ {r} \left[ r - \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} \right] - \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \mathbf {w} - \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w} \right] \tag {52}
$$

$$
+ \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathrm{VE} (s ^ {\prime}, a ^ {\prime}) \right]
$$

$$
= \mathbf {z} _ {s a} ^ {\top} \left(\mathbf {w} - \gamma W _ {p} \mathbf {w} - \mathbf {w} _ {r}\right) - \mathbb {E} _ {r} \left[ r - \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} \right] - \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \mathbf {w} - \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w} \right] \tag {53}
$$

$$
+ \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathrm{VE} (s ^ {\prime}, a ^ {\prime}) \right]
$$

$$
= - \mathbb {E} _ {r} \left[ r - \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} \right] - \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \mathbf {w} - \mathbf {z} _ {s a} ^ {\top} W _ {p} \mathbf {w} \right] + \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathrm{VE} \left(s ^ {\prime}, a ^ {\prime}\right) \right] \tag {54}
$$

$$
= \left(\mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - \mathbb {E} _ {r} [ r ]\right) + \gamma \left(\mathbf {z} _ {s a} ^ {\top} W _ {p} - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \right]\right) \mathbf {w} + \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ \mathrm{VE} \left(s ^ {\prime}, a ^ {\prime}\right) \right]. \tag {55}
$$

Then given the recursive relationship, akin to the Bellman equation (Sutton & Barto, 1998), the value error VE recursively expands to the discounted state-action visitation distribution $p ^ { \pi }$ . For $( \hat { s } , \hat { a } ) \in S \times A \colon$

$$
\mathrm{VE} (\hat {s}, \hat {a}) = \frac {1}{1 - \gamma} \mathbb {E} _ {(s, a) \sim p ^ {\pi} (\hat {s}, \hat {a})} \left[ \left(\mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - \mathbb {E} _ {r | s, a} [ r ]\right) + \gamma \left(\mathbf {z} _ {s a} ^ {\top} W _ {p} - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} | s, a} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \right]\right) \mathbf {w} \right]. \tag {56}
$$

Taking the absolute value:

$$
\left| \mathrm{VE} (\hat {s}, \hat {a}) \right| = \left| \frac {1}{1 - \gamma} \mathbb {E} _ {(s, a) \sim p ^ {\pi} (\hat {s}, \hat {a})} \left[ \left(\mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - \mathbb {E} _ {r | s, a} [ r ]\right) + \gamma \left(\mathbf {z} _ {s a} ^ {\top} W _ {p} - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} | s, a} [ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} ]\right) \mathbf {w} \right] \right| (5 7)
$$

$$
\left| \mathrm{VE} (\hat {s}, \hat {a}) \right| \leq \frac {1}{1 - \gamma} \mathbb {E} _ {(s, a) \sim p ^ {\pi} (\hat {s}, \hat {a})} \left[ \left| \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - \mathbb {E} _ {r | s, a} [ r ] \right| + \gamma \left| \left(\mathbf {z} _ {s a} ^ {\top} W _ {p} - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} | s, a} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \right]\right) \mathbf {w} \right| \right] \tag {58}
$$

$$
= \frac {1}{1 - \gamma} \max _ {(s, a) \in S \times A} \left(\left| \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - \mathbb {E} _ {r | s, a} [ r ] \right| + \gamma \left| \left(\mathbf {z} _ {s a} ^ {\top} W _ {p} - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} | s, a} \left[ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ^ {\top} \right]\right) \mathbf {w} \right|\right) \tag {59}
$$

$$
\leq \frac {1}{1 - \gamma} \max _ {(s, a) \in S \times A} \left(\left| \mathbf {z} _ {s a} ^ {\top} \mathbf {w} _ {r} - \mathbb {E} _ {r | s, a} [ r ] \right| + \max _ {i} | \mathbf {w} _ {i} | \sum \left| \mathbf {z} _ {s a} ^ {\top} W _ {p} - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} | s, a} [ \mathbf {z} _ {s ^ {\prime} a ^ {\prime}} ] \right|\right). (6 0)
$$

Theorem 3. Given functions $f ( s ) = \mathbf { z } _ { s }$ and $g ( \mathbf { z } _ { s } , a ) = \mathbf { z } _ { s a }$ , then if there exists functions $\hat { p }$ and $\hat { R }$ such that for all $( s , a ) \in S \times A { \mathrm { : } }$ :

$$
\mathbb {E} _ {\hat {R}} \left[ \hat {R} \left(\mathbf {z} _ {s a}\right) \right] = \mathbb {E} _ {R} [ R (s, a) ], \quad \hat {p} \left(\mathbf {z} _ {s ^ {\prime}} \mid \mathbf {z} _ {s a}\right) = \sum_ {\hat {s}: \mathbf {z} _ {\hat {s}} = \mathbf {z} _ {s ^ {\prime}}} p (\hat {s} | s, a), \tag {61}
$$

then for any policy π where there exists a corresponding policy ${ \hat { \pi } } ( a | \mathbf { z } _ { s } ) = \pi ( a | s )$ , there exists a function $\hat { Q }$ equal to the true value function $Q ^ { \pi }$ over all possible state-action pairs $( s , a ) \in S \times A { \mathrm { : } }$ :

$$
\hat {Q} \left(\mathbf {z} _ {s a}\right) = Q ^ {\pi} (s, a). \tag {62}
$$

Furthermore, Equation 61 guarantees the existence of an optimal policy ${ \hat { \pi } } ^ { * } ( a | \mathbf { z } _ { s } ) = \pi ^ { * } ( a | s )$ .

Proof. Let

$$
Q _ {h} ^ {\pi} (s, a) = \sum_ {t = 0} ^ {h} \gamma^ {t} \mathbb {E} _ {\pi} [ R (s _ {t}, a _ {t}) | s _ {0} = s, a _ {0} = a ] \tag {63}
$$

$$
\hat {Q} _ {h} \left(\mathbf {z} _ {s a}\right) = \sum_ {t = 0} ^ {h} \gamma^ {t} \mathbb {E} _ {\pi} \left[ \hat {R} \left(\mathbf {z} _ {s _ {t} a _ {t}}\right) \mid s _ {0} = s, a _ {0} = a \right] \tag {64}
$$

Then

$$
Q _ {0} ^ {\pi} (s, a) = \mathbb {E} _ {R} [ R (s, a) ] \tag {65}
$$

$$
= \mathbb {E} _ {\hat {R}} \left[ \hat {R} \left(\mathbf {z} _ {s a}\right) \right] \tag {66}
$$

$$
= \hat {Q} _ {0} \left(\mathbf {z} _ {s a}\right). \tag {67}
$$

Assuming $Q _ { n - 1 } ^ { \pi } ( s , a ) = \hat { Q } _ { n - 1 } ( \mathbf { z } _ { s a } )$ then noting that $\hat { p } ( \mathbf { z } | \mathbf { z } _ { s a } ) = 0$ if z that is not in the image of $f ( s ) = \mathbf { z } _ { s }$ .

$$
Q _ {n} ^ {\pi} (s, a) = \mathbb {E} _ {R} [ R (s, a) ] + \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} [ Q _ {n - 1} ^ {\pi} (s ^ {\prime}, a ^ {\prime}) ] \tag {68}
$$

$$
= \mathbb {E} _ {\hat {R}} [ \hat {R} (s, a) ] + \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} [ \hat {Q} _ {n - 1} (\mathbf {z} _ {s ^ {\prime} a ^ {\prime}}) ] \tag {69}
$$

$$
= \mathbb {E} _ {\hat {R}} [ \hat {R} (s, a) ] + \gamma \sum_ {s ^ {\prime}} \sum_ {a ^ {\prime}} p (s ^ {\prime} | s, a) \pi (a ^ {\prime} | s ^ {\prime}) \hat {Q} _ {n - 1} (\mathbf {z} _ {s ^ {\prime} a ^ {\prime}}) \tag {70}
$$

$$
= \mathbb {E} _ {\hat {R}} [ \hat {R} (s, a) ] + \gamma \sum_ {z _ {s ^ {\prime}}} \sum_ {a ^ {\prime}} \hat {p} (\mathbf {z} _ {s ^ {\prime}} | \mathbf {z} _ {s a}) \hat {\pi} (a ^ {\prime} | \mathbf {z} _ {s ^ {\prime}}) \hat {Q} _ {n - 1} (\mathbf {z} _ {s ^ {\prime} a ^ {\prime}}) \tag {71}
$$

$$
= \hat {Q} _ {n} \left(\mathbf {z} _ {s a}\right). \tag {72}
$$

Thus $\begin{array} { r } { \hat { Q } ( \mathbf { z } _ { s a } ) = \operatorname* { l i m } _ { n  \infty } \hat { Q } _ { n } ( \mathbf { z } _ { s a } ) } \end{array}$ exists, as $\hat { Q } _ { n }$ can be defined as a function of ${ \mathrm { ~  ~ \cdot ~ } } _ { \hat { p } , \hat { R } , }$ and πˆ for all n. Similarly, let π be an optimal policy. Repeating the same arguments we see that

$$
Q _ {n} ^ {\pi} (s, a) = \mathbb {E} _ {R} [ R (s, a) ] + \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} [ Q _ {n - 1} ^ {\pi} (s ^ {\prime}, a ^ {\prime}) ] \tag {73}
$$

$$
= \mathbb {E} _ {R} [ R (s, a) ] + \gamma \sum_ {s ^ {\prime}} p \left(s ^ {\prime} \mid s, a\right) \max _ {a ^ {\prime}} Q _ {n - 1} ^ {\pi} \left(s ^ {\prime}, a ^ {\prime}\right) \tag {74}
$$

$$
= \mathbb {E} _ {\hat {R}} [ \hat {R} (s, a) ] + \gamma \sum_ {z _ {s ^ {\prime}}} \hat {p} (\mathbf {z} _ {s ^ {\prime}} | \mathbf {z} _ {s a}) \max _ {a ^ {\prime}} \hat {Q} _ {n - 1} (\mathbf {z} _ {s ^ {\prime} a ^ {\prime}}) \tag {75}
$$

$$
= \hat {Q} _ {n} (\mathbf {z} _ {s a}). \tag {76}
$$

Thus there exists a function $\hat { Q } ( g ( \mathbf { z } _ { s } , a ) ) = Q ^ { * } ( s , a )$ , consequently, there exists an optimal policy $\hat { \pi } ^ { * } ( a | \mathbf { z } _ { s } ) = \mathrm { a r g m a x } _ { a } \hat { Q } ( s , a )$ .

# B EXPERIMENTAL DETAILS

# B.1 HYPERPARAMETERS

Table 3: MR.Q Hyperparameters. Hyperparameters values are kept fixed across all benchmarks. 

<table><tr><td></td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="6"></td><td>Dynamics loss weight  $\lambda_{\text{Dynamics}}$ </td><td>1</td></tr><tr><td>Reward loss weight  $\lambda_{\text{Reward}}$ </td><td>0.1</td></tr><tr><td>Terminal loss weight  $\lambda_{\text{Terminal}}$ </td><td>0.1</td></tr><tr><td>Pre-activation loss weight  $\lambda_{\text{pre-activ}}$ </td><td>1e-5</td></tr><tr><td>Encoder horizon  $H_{\text{Enc}}$ </td><td>5</td></tr><tr><td>Multi-step returns horizon  $H_Q$ </td><td>3</td></tr><tr><td rowspan="2">TD3(Fujimoto et al., 2018)</td><td>Target policy noise  $\sigma$ </td><td> $\mathcal{N}(0,0.2^2)$ </td></tr><tr><td>Target policy noise clipping  $c$ </td><td>(-0.3, 0.3)</td></tr><tr><td rowspan="2">LAP(Fujimoto et al., 2020)</td><td>Probability smoothing  $\alpha$ </td><td>0.4</td></tr><tr><td>Minimum priority</td><td>1</td></tr><tr><td rowspan="2">Exploration</td><td>Initial random exploration time steps</td><td>10k</td></tr><tr><td>Exploration noise</td><td> $\mathcal{N}(0,0.2^2)$ </td></tr><tr><td rowspan="5">Common</td><td>Discount factor  $\gamma$ </td><td>0.99</td></tr><tr><td>Replay buffer capacity</td><td>1M</td></tr><tr><td>Mini-batch size</td><td>256</td></tr><tr><td>Target update frequency  $T_{\text{target}}$ </td><td>250</td></tr><tr><td>Replay ratio</td><td>1</td></tr><tr><td rowspan="12">Encoder Network</td><td>Optimizer</td><td>AdamW (Loshchilov &amp; Hutter, 2019)</td></tr><tr><td>Learning rate</td><td>1e-4</td></tr><tr><td>Weight decay</td><td>1e-4</td></tr><tr><td> $z_s$  dim</td><td>512</td></tr><tr><td> $z_{sa}$  dim</td><td>512</td></tr><tr><td> $z_a$  dim (only used within architecture)</td><td>256</td></tr><tr><td>Hidden dim</td><td>512</td></tr><tr><td>Activation function</td><td>ELU (Clevert et al., 2015)</td></tr><tr><td>Weight initialization</td><td>Xavier uniform (Glorot &amp; Bengio, 2010)</td></tr><tr><td>Bias initialization</td><td>0</td></tr><tr><td>Reward bins</td><td>65</td></tr><tr><td>Reward range</td><td>[-10, 10] (effective: [-22k, 22k])</td></tr><tr><td rowspan="7">Value Network</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>3e-4</td></tr><tr><td>Hidden dim</td><td>512</td></tr><tr><td>Activation function</td><td>ELU</td></tr><tr><td>Weight initialization</td><td>Xavier uniform</td></tr><tr><td>Bias initialization</td><td>0</td></tr><tr><td>Gradient clip norm</td><td>20</td></tr><tr><td rowspan="7">Policy Network</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>3e-4</td></tr><tr><td>Hidden dim</td><td>512</td></tr><tr><td>Activation function</td><td>ReLU</td></tr><tr><td>Weight initialization</td><td>Xavier uniform</td></tr><tr><td>Bias initialization</td><td>0</td></tr><tr><td>Gumbel-Softmax  $\tau$  (Jang et al., 2017)</td><td>10</td></tr></table>

# B.2 NETWORK ARCHITECTURE

This section describes the networks used in our method using PyTorch code blocks (Paszke et al., 2019). The state encoder and state-action encoder are described as separate networks for clarity but are trained end-to-end as a single network. The value and policy networks are trained independently from the encoders.

Preamble   
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

zs_dim = 512
za_dim = 256
zsa_dim = 512

def ln_activ(self, x):
    x = F.layer_norm(x, (x.shape[-1],))
    return self.activ(x) 
```

# State Encoder f Network

For image inputs, four convolutional layers are used, each with 32 output channels, kernel size of 3, strides of 2, 2, 2, 1 , and ELU activations (Clevert et al., 2015). The convolutional layers are ( )followed by a linear layer taking in the flattened output followed by LayerNorm (Ba et al., 2016) and a final ELU activation.

For vector inputs, a three layer multilayer perceptron (MLP) is used, with hidden dimension 512 and LayerNorm followed by ELU activations after each layer.

The resulting state embedding zs is trained end-to-end with the state-action encoder. It is also used downstream by the policy network (without propagating gradients).

```python
if image_observation_space:
    self.zs_cnn1 = nn.Conv2d(state_channels, 32, 3, stride=2)
    self.zs_cnn2 = nn.Conv2d(32, 32, 3, stride=2)
    self.zs_cnn3 = nn.Conv2d(32, 32, 3, stride=2)
    self.zs_cnn4 = nn.Conv2d(32, 32, 3, stride=1)
    # Assumes 84 x 84 input
    self.zs_lin = nn.Linear(1568, zs_dim)

else:
    self.zs_mlp1 = nn.Linear(state_dim, 512)
    self.zs_mlp2 = nn.Linear(512, 512)
    self.zs_mlp3 = nn.Linear(512, zs_dim)

self.activ = F.elu

def cnn_forward(self, state):
    state = state/255. - 0.5
    zs = self.activ(self.zs_cnn1(state))
    zs = self.activ(self.zs_cnn2(zs))
    zs = self.activ(self.zs_cnn3(zs))
    zs = self.activ(self.zs_cnn4(zs))
    zs = zs.reshape(batch_size, 1568)
    return ln_activ(self.zs_lin(zs))

def mlp_forward(self, state):
    zs = self.ln_activ(self.zs_mlp1(state))
    zs = self.ln_activ(self.zs_mlp2(zs))
    return self.ln_activ(self.zs_mlp3(zs)) 
```

# State-Action Encoder g Network

Action input is processed by a linear layer followed by an ELU activation. Afterwards, the processed action is concatenated with the state embedding and processed by a three layer MLP with hidden dimension 512, and LayerNorm followed by ELU activations after the first two layers.

The resulting state-action embedding $\mathbf { z } _ { s a }$ is used by a linear layer to make predictions about reward, the next state embedding, and the terminal signal. It is also used downstream by the value network (without propagating gradients).

```python
self.za = nn.Linear(action_dim, za_dim)
self.zsa1 = nn.Linear(zs_dim + za_dim, 512)
self.zsa2 = nn.Linear(512, 512)
self.zsa3 = nn.Linear(512, zsa_dim)
self.model = nn.Linear(zsa_dim, output_dim)
self.activ = F.elu

def forward(self, zs, action):
    za = self.activ(self.za(action))
    zsa = torch.cat([zs, za], 1)
    zsa = self.ln_activ(self.zsa1(zsa))
    zsa = self.ln_activ(self.zsa2(zsa))
    zsa = self.zsa3(zsa)
    return self.model(zsa), zsa 
```

# Value Q Networks

The value network is a four layer MLP with hidden dimension 512, and LayerNorm followed by ELU activations after the first three layers.

Two value networks are used with the same network and forward pass.

```python
self.l1 = nn.Linear(zsa_dim, 512)
self.l2 = nn.Linear(512, 512)
self.l3 = nn.Linear(512, 512)
self.l4 = nn.Linear(512, 1)
self.activ = F.elu

def forward(self, zsa):
    q = self.ln_activ(self.l1(zsa))
    q = self.ln_activ(self.l2(q))
    q = self.ln_activ(self.l3(q))
    return self.l4(q) 
```

# Policy π Network

The policy network is a three layer MLP with hidden dimension 512, and LayerNorm followed by ReLU activations after the first two layers.

For discrete actions, the final activation is the Gumbel Softmax with τ  10. For continous actions, the final activation is a tanh function.

```python
self.l1 = nn.Linear(zs_dim, 512)
self.l2 = nn.Linear(hdim, 512)
self.l3 = nn.Linear(512, action_dim)
self.activ = F.relu

if discrete_action_space:
    self.final_activ = partial(F.gumbel_softmax, tau=10)
else:
    self.final_activ = torch.tanh

def forward(self, zs):
    a = self.ln_activ(self.l1(zs))
    a = self.ln_activ(self.l2(a))
    return self.final_activ(self.l3(a)) 
```

# B.3 ENVIRONMENTS

All main experiments were run for 10 seeds (the design study is based on 5 seeds). Evaluations are based on the average performance over 10 episodes, measured every 5k time steps for Gym and DM control and every 100k time steps for Atari.

Gym - Locomotion. For the gym locomotion tasks (Todorov et al., 2012; Brockman et al., 2016; Towers et al., 2024), we choose the five most common environments that appear in prior work (Fujimoto et al., 2018; 2024; Haarnoja et al., 2018; Kuznetsov et al., 2020). We use the -v4 version. No preprocessing is applied. When aggregating scores, we use normalize with the TD3 scores obtained from TD7 (Fujimoto et al., 2024):

$$
\text { TD3 - Normalized } (x) := \frac {x - \text { random   score }}{\text { TD3   score } - \text { random   score }}. \tag {77}
$$

<table><tr><td></td><td>Random</td><td>TD3</td></tr><tr><td>Ant-v4</td><td>-70.288</td><td>3942</td></tr><tr><td>HalfCheetah-v4</td><td>-289.415</td><td>10574</td></tr><tr><td>Hopper-v4</td><td>18.791</td><td>3226</td></tr><tr><td>Humanoid-v4</td><td>120.423</td><td>5165</td></tr><tr><td>Walker2d-v4</td><td>2.791</td><td>3946</td></tr></table>

DM Control Suite. For the DM control suite (Tassa et al., 2018), we choose the 28 default environments that appear either in the evaluation of TD-MPC2 or DreamerV3. We omit any custom environments included by the TD-MPC2 authors. The same subset of tasks are used in the evaluation of proprioceptive and visual control. Like prior work, for both observation spaces, we use an action repeat of 2 (Hansen et al., 2024). For visual control, the state (network input) is composed of the previous 3 observations which are resized to 84 × 84 pixels in RGB format (Tassa et al., 2018).

Atari. For the Atari games (Bellemare et al., 2013; Brockman et al., 2016; Towers et al., 2024), we use the 57 games in the Atari-57 benchmark that appears in prior work (Hessel et al., 2018; Schrittwieser et al., 2020; Badia et al., 2020; Hafner et al., 2023). For DQN and Rainbow, two games (Defender and Surround) are missing from the Dopamine framework (Castro et al., 2018) and are omitted. We use the -v5 version. For MR.Q, we use the common preprocessing steps (Mnih et al., 2015; Machado et al., 2018; Castro et al., 2018), where an action repeat of 4 is used and the observations are grayscaled, resized to 84  84 pixels and set to the max between the 3rd and 4th frame. The state (network input) is composed of the previous 4 observations.

Consider the 16 frame sequence used by a single state, where $f _ { i }$ is the ith grayscaled and resized frame and $o _ { j }$ is the jth observation set to the max of two frames

$$
\overbrace {f _ {0} , f _ {1} , \underbrace {f _ {2} , f _ {3}} _ {o _ {0} = \max (f _ {2} , f _ {3})}} ^ {\text {action} a _ {0}}, \overbrace {f _ {4} , f _ {5} , \underbrace {f _ {6} , f _ {7}} _ {o _ {1} = \max (f _ {6} , f _ {7})}} ^ {\text {action} a _ {1}}, \overbrace {f _ {8} , f _ {9} , \underbrace {f _ {1 0} , f _ {1 1}} _ {o _ {2} = \max (f _ {1 0} , f _ {1 1})}} ^ {\text {action} a _ {2}}, \overbrace {f _ {1 2} , f _ {1 3} , \underbrace {f _ {1 4} , f _ {1 5}} _ {o _ {3} = \max (f _ {1 4} , f _ {1 5})}} ^ {\text {action} a _ {3}}, \tag {78}
$$

then the state is defined as follows:

$$
s = \left[ \begin{array}{l} o _ {0} = \max (f _ {2}, f _ {3}) \\ o _ {1} = \max (f _ {6}, f _ {7}) \\ o _ {2} = \max (f _ {1 0}, f _ {1 1}) \\ o _ {3} = \max (f _ {1 4}, f _ {1 5}) \end{array} \right]. \tag {79}
$$

When aggregating scores, we normalize with Human scores obtained from (Wang et al., 2016):

$$
\text { Human - Normalized } (x) := \frac {x - \text { random   score }}{\text { Human   score } - \text { random   score }}. \tag {80}
$$

<table><tr><td></td><td>Random</td><td>Human</td></tr><tr><td>Alien</td><td>227.8</td><td>7127.7</td></tr><tr><td>Amidar</td><td>5.8</td><td>1719.5</td></tr><tr><td>Assault</td><td>222.4</td><td>742.0</td></tr><tr><td>Asterix</td><td>210.0</td><td>8503.3</td></tr><tr><td>Asteroids</td><td>719.1</td><td>47388.7</td></tr><tr><td>Atlantis</td><td>12850.0</td><td>29028.1</td></tr><tr><td>BankHeist</td><td>14.2</td><td>753.1</td></tr><tr><td>BattleZone</td><td>2360.0</td><td>37187.5</td></tr><tr><td>BeamRider</td><td>363.9</td><td>16926.5</td></tr><tr><td>Berzerk</td><td>123.7</td><td>2630.4</td></tr><tr><td>Bowling</td><td>23.1</td><td>160.7</td></tr><tr><td>Boxing</td><td>0.1</td><td>12.1</td></tr><tr><td>Breakout</td><td>1.7</td><td>30.5</td></tr><tr><td>Centipede</td><td>2090.9</td><td>12017.0</td></tr><tr><td>ChopperCommand</td><td>811.0</td><td>7387.8</td></tr><tr><td>CrazyClimber</td><td>10780.5</td><td>35829.4</td></tr><tr><td>Defender (not used)</td><td>2874.5</td><td>18688.9</td></tr><tr><td>DemonAttack</td><td>152.1</td><td>1971.0</td></tr><tr><td>DoubleDunk</td><td>-18.6</td><td>-16.4</td></tr><tr><td>Enduro</td><td>0.0</td><td>860.5</td></tr><tr><td>FishingDerby</td><td>-91.7</td><td>-38.7</td></tr><tr><td>Freeway</td><td>0.0</td><td>29.6</td></tr><tr><td>Frostbite</td><td>65.2</td><td>4334.7</td></tr><tr><td>Gopher</td><td>257.6</td><td>2412.5</td></tr><tr><td>Gravitar</td><td>173.0</td><td>3351.4</td></tr><tr><td>Hero</td><td>1027.0</td><td>30826.4</td></tr><tr><td>IceHockey</td><td>-11.2</td><td>0.9</td></tr><tr><td>Jamesbond</td><td>29.0</td><td>302.8</td></tr><tr><td>Kangaroo</td><td>52.0</td><td>3035.0</td></tr><tr><td>Krull</td><td>1598.0</td><td>2665.5</td></tr><tr><td>KungFuMaster</td><td>258.5</td><td>22736.3</td></tr><tr><td>MontezumaRevenge</td><td>0.0</td><td>4753.3</td></tr><tr><td>MsPacman</td><td>307.3</td><td>6951.6</td></tr><tr><td>NameThisGame</td><td>2292.3</td><td>8049.0</td></tr><tr><td>Phoenix</td><td>761.4</td><td>7242.6</td></tr><tr><td>Pitfall</td><td>-229.4</td><td>6463.7</td></tr><tr><td>Pong</td><td>-20.7</td><td>14.6</td></tr><tr><td>PrivateEye</td><td>24.9</td><td>69571.3</td></tr><tr><td>Qbert</td><td>163.9</td><td>13455.0</td></tr><tr><td>Riverraid</td><td>1338.5</td><td>17118.0</td></tr><tr><td>RoadRunner</td><td>11.5</td><td>7845.0</td></tr><tr><td>Robotank</td><td>2.2</td><td>11.9</td></tr><tr><td>Seaquest</td><td>68.4</td><td>42054.7</td></tr><tr><td>Skiing</td><td>-17098.1</td><td>-4336.9</td></tr><tr><td>Solaris</td><td>1236.3</td><td>12326.7</td></tr><tr><td>SpaceInvaders</td><td>148.0</td><td>1668.7</td></tr><tr><td>StarGunner</td><td>664.0</td><td>10250.0</td></tr><tr><td>Surround (not used)</td><td>-10.0</td><td>6.5</td></tr><tr><td>Tennis</td><td>-23.8</td><td>-8.3</td></tr><tr><td>TimePilot</td><td>3568.0</td><td>5229.2</td></tr><tr><td>Tutankham</td><td>11.4</td><td>167.6</td></tr><tr><td>UpNDown</td><td>533.4</td><td>11693.2</td></tr><tr><td>Venture</td><td>0.0</td><td>1187.5</td></tr><tr><td>VideoPinball</td><td>16256.9</td><td>17667.9</td></tr><tr><td>WizardOfWor</td><td>563.5</td><td>4756.5</td></tr><tr><td>YarsRevenge</td><td>3092.9</td><td>54576.9</td></tr><tr><td>Zaxxon</td><td>32.5</td><td>9173.3</td></tr></table>

# B.4 BASELINES

DreamerV3. (Hafner et al., 2023). Results for Gym and DMC were obtained by re-running the authors’ code (https://github.com/danijar/dreamerv3 - Commit 251910d04c9f38dd9dc385775bb0d6- efa0e57a95) over 10 seeds, using the author-suggested hyperparameters from the DMC benchmark. Code was modified slightly to match our evaluation protocol. Atari results are based on the authors’ reported results.

DrQ-v2. (Yarats et al., 2022). We use the authors’ reported results whenever possible. For missing any results, we re-ran the authors’ code (https://github.com/facebookresearch/drqv2 - Commit c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef) for 10 seeds.

DQN. (Mnih et al., 2015). Results were obtained from the Dopamine framework (Castro et al., 2018).

PPO. (Schulman et al., 2017). Results were gathered using Stable Baselines 3 (Raffin et al., 2021) and default hyperparameters. The default MLP policy was used for Gym and DMC-proprioceptive and the default CNN policy was used for DMC-visual and Atari.

Rainbow. (Hessel et al., 2018). Results were obtained from the Dopamine framework (Castro et al., 2018).

TD-MPC2. (Hansen et al., 2024). Results for DMC were obtained by re-running the authors’ code on their main branch (https://github.com/nicklashansen/tdmpc2 - Commit 5f6fadec0fec78304b4b53e8171d348b58cac486). As the Gym environments include a termination signal, results for Gym were obtained by running their episodic branch (https://github.com/ nicklashansen/tdmpc2/tree/episodic-rl - Commit 3789fcd5b872079ad610fa3299ff47c3a427a04a). All experiments were run for 10 seeds and use the default author-suggested hyperparameters for all tasks.

TD7. (Fujimoto et al., 2024). Results for Gym were obtained from the authors. Results for DMC were obtained by re-running the authors’ code (https://github.com/sfujim/TD7 - Commit c1c280de1513f474488061b4cf39642b75dd84bd) using our setup for DMC. All experiments use 10 seeds and use the default author-suggested hyperparameters from the Gym benchmark.

# B.5 SOFTWARE VERSIONS

• Gymnasium 0.29.1 (Towers et al., 2024)   
• MuJoCo 3.2.2 (Todorov et al., 2012)   
• NumPy 2.1.1 (Harris et al., 2020)   
• Python 3.11.8 (Van Rossum & Drake Jr, 1995)   
• PyTorch 2.4.1 (Paszke et al., 2019)

# C COMPLETE MAIN RESULTS

# C.1 GYM

Table 4: Gym - Locomotion final results. Final average performance at 1M time steps over 10 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are computed over the TD3-normalized score (see Appendix B.3). 

<table><tr><td>Task</td><td>TD7</td><td>PPO</td><td>TD-MPC2</td><td>DreamerV3</td><td>MR.Q</td></tr><tr><td>Ant</td><td>8509 [8164, 8852]</td><td>1584 [1355, 1802]</td><td>4751 [3012, 6261]</td><td>1947 [1121, 2751]</td><td>6901 [6261, 7482]</td></tr><tr><td>HalfCheetah</td><td>17433 [17284, 17550]</td><td>1744 [1525, 2120]</td><td>15078 [14050, 16012]</td><td>5502 [3887, 7117]</td><td>12939 [11663, 13762]</td></tr><tr><td>Hopper</td><td>3511 [3245, 3746]</td><td>3022 [2587, 3356]</td><td>2081 [1233, 2916]</td><td>2666 [2071, 3201]</td><td>2692 [2131, 3309]</td></tr><tr><td>Humanoid</td><td>7428 [7300, 7555]</td><td>477 [431, 522]</td><td>6071 [5767, 6327]</td><td>4217 [2791, 5481]</td><td>10223 [9929, 10498]</td></tr><tr><td>Walker2d</td><td>6096 [5535, 6521]</td><td>2487 [1875, 3067]</td><td>3008 [1659, 4220]</td><td>4519 [3746, 5190]</td><td>6039 [5644, 6386]</td></tr><tr><td>Mean</td><td>1.57 [1.54, 1.60]</td><td>0.45 [0.41, 0.48]</td><td>1.04 [0.90, 1.16]</td><td>0.76 [0.67, 0.85]</td><td>1.46 [1.41, 1.52]</td></tr><tr><td>Median</td><td>1.55 [1.45, 1.63]</td><td>0.41 [0.36, 0.47]</td><td>1.18 [0.80, 1.23]</td><td>0.81 [0.56, 0.90]</td><td>1.53 [1.43, 1.61]</td></tr><tr><td>IQM</td><td>1.54 [1.49, 1.58]</td><td>0.41 [0.35, 0.46]</td><td>1.05 [0.87, 1.19]</td><td>0.72 [0.62, 0.85]</td><td>1.50 [1.44, 1.55]</td></tr></table>

![](images/815e8581448f86ba0f323a82fd8a594bd6e34af1fc4952bed3c99d7f41a4fcda.jpg)  
Figure 3: Gym - Locomotion learning curves. Results are over 10 seeds. The shaded area captures a 95% boostrap confidence interval.

# C.2 DMC - PROPRIOCEPTIVE

Table 5: DMC - Proprioceptive final results. Final average performance at 500k time steps (1M time steps in the original environment due to action repeat) over 10 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are computed over the default reward. 

<table><tr><td>Task</td><td>TD7</td><td>PPO</td><td>TD-MPC2</td><td>DreamerV3</td><td>MR.Q</td></tr><tr><td>acrobot-swingup</td><td>58 [38, 75]</td><td>39 [33, 45]</td><td>584 [551, 615]</td><td>230 [193, 266]</td><td>567 [523, 616]</td></tr><tr><td>ball_in_cup-catch</td><td>983 [981, 985]</td><td>769 [689, 841]</td><td>984 [982, 986]</td><td>968 [965, 973]</td><td>981 [979, 984]</td></tr><tr><td>cartpole-balance</td><td>999 [998, 1000]</td><td>999 [1000, 1000]</td><td>996 [995, 998]</td><td>998 [997, 1000]</td><td>999 [999, 1000]</td></tr><tr><td>cartpole-balance_sparse</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td><td>999 [1000, 1000]</td><td>1000 [1000, 1000]</td></tr><tr><td>cartpole-swingup</td><td>869 [866, 873]</td><td>776 [661, 853]</td><td>875 [870, 880]</td><td>736 [591, 838]</td><td>866 [866, 866]</td></tr><tr><td>cartpole-swingup_sparse</td><td>573 [333, 806]</td><td>391 [159, 625]</td><td>845 [839, 849]</td><td>702 [560, 792]</td><td>798 [780, 818]</td></tr><tr><td>cheetah-run</td><td>821 [642, 913]</td><td>269 [247, 295]</td><td>917 [915, 920]</td><td>699 [655, 744]</td><td>914 [911, 917]</td></tr><tr><td>dog-run</td><td>69 [36, 101]</td><td>26 [26, 28]</td><td>265 [166, 342]</td><td>4 [4, 5]</td><td>569 [547, 595]</td></tr><tr><td>dog-stand</td><td>582 [432, 741]</td><td>129 [122, 139]</td><td>506 [266, 715]</td><td>22 [20, 27]</td><td>967 [960, 975]</td></tr><tr><td>dog-trot</td><td>21 [13, 30]</td><td>31 [30, 34]</td><td>407 [265, 530]</td><td>10 [6, 17]</td><td>877 [845, 898]</td></tr><tr><td>dog-walk</td><td>52 [19, 116]</td><td>40 [37, 43]</td><td>486 [240, 704]</td><td>17 [15, 21]</td><td>916 [908, 924]</td></tr><tr><td>finger-spin</td><td>335 [99, 596]</td><td>459 [420, 497]</td><td>986 [986, 988]</td><td>666 [577, 763]</td><td>937 [917, 956]</td></tr><tr><td>finger-turn_easy</td><td>912 [774, 983]</td><td>182 [153, 211]</td><td>979 [975, 983]</td><td>906 [883, 927]</td><td>953 [931, 974]</td></tr><tr><td>finger-turn_hard</td><td>470 [199, 727]</td><td>58 [35, 79]</td><td>947 [916, 977]</td><td>864 [812, 900]</td><td>950 [910, 974]</td></tr><tr><td>fish-swim</td><td>86 [64, 120]</td><td>103 [84, 128]</td><td>659 [615, 706]</td><td>813 [808, 819]</td><td>792 [773, 810]</td></tr><tr><td>hopper-hop</td><td>87 [25, 160]</td><td>10 [0, 23]</td><td>425 [368, 500]</td><td>116 [66, 165]</td><td>251 [195, 301]</td></tr><tr><td>hopper-stand</td><td>670 [466, 829]</td><td>128 [56, 216]</td><td>952 [944, 958]</td><td>747 [669, 806]</td><td>951 [948, 955]</td></tr><tr><td>humanoid-run</td><td>57 [23, 92]</td><td>0 [1, 1]</td><td>181 [121, 231]</td><td>0 [1, 1]</td><td>200 [170, 236]</td></tr><tr><td>humanoid-stand</td><td>317 [117, 516]</td><td>5 [5, 6]</td><td>658 [506, 745]</td><td>5 [5, 6]</td><td>868 [822, 903]</td></tr><tr><td>humanoid-walk</td><td>176 [42, 320]</td><td>1 [1, 2]</td><td>754 [725, 791]</td><td>1 [1, 2]</td><td>662 [610, 724]</td></tr><tr><td>pendulum-swingup</td><td>500 [251, 743]</td><td>115 [70, 164]</td><td>846 [830, 862]</td><td>774 [740, 802]</td><td>748 [597, 829]</td></tr><tr><td>quadruped-run</td><td>645 [567, 713]</td><td>144 [122, 170]</td><td>942 [938, 947]</td><td>130 [92, 169]</td><td>947 [940, 954]</td></tr><tr><td>quadruped-walk</td><td>949 [939, 957]</td><td>122 [103, 142]</td><td>963 [959, 967]</td><td>193 [137, 243]</td><td>963 [959, 967]</td></tr><tr><td>reacher-easy</td><td>970 [951, 982]</td><td>367 [188, 558]</td><td>983 [980, 986]</td><td>966 [964, 970]</td><td>983 [983, 985]</td></tr><tr><td>reacher-hard</td><td>898 [861, 936]</td><td>125 [40, 234]</td><td>960 [936, 979]</td><td>919 [864, 955]</td><td>977 [975, 980]</td></tr><tr><td>walker-run</td><td>804 [783, 825]</td><td>97 [91, 104]</td><td>854 [851, 859]</td><td>510 [430, 588]</td><td>793 [765, 815]</td></tr><tr><td>walker-stand</td><td>983 [974, 989]</td><td>431 [363, 495]</td><td>991 [990, 994]</td><td>941 [934, 948]</td><td>988 [987, 990]</td></tr><tr><td>walker-walk</td><td>977 [975, 980]</td><td>283 [253, 312]</td><td>981 [979, 984]</td><td>898 [875, 919]</td><td>978 [978, 980]</td></tr><tr><td>Mean</td><td>566 [544, 590]</td><td>254 [241, 267]</td><td>783 [769, 797]</td><td>530 [520, 539]</td><td>835 [829, 842]</td></tr><tr><td>Median</td><td>613 [548, 718]</td><td>127 [112, 145]</td><td>896 [893, 899]</td><td>700 [644, 741]</td><td>927 [914, 934]</td></tr><tr><td>IQM</td><td>612 [569, 657]</td><td>154 [135, 167]</td><td>868 [860, 880]</td><td>577 [557, 594]</td><td>907 [903, 914]</td></tr></table>

![](images/b8a8820c17d2fe399dd6201b13e7917bd9cfbfb56cb5832dfb5c844fd536978f.jpg)  
MR.Q DreamerV3 TD-MPC2 PPO TD7

Figure 4: DMC - Proprioceptive learning curves. Time steps consider the number of environment interactions, where 500k time steps equals 1M frames in the original environment. Results are over 10 seeds. The shaded area captures a 95% boostrap confidence interval.

# C.3 DMC - VISUAL

Table 6: DMC - Visual final results. Final average performance at 500k time steps (1M time steps in the original environment due to action repeat) over 10 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are computed over the default reward. 

<table><tr><td>Task</td><td>DrQ-v2</td><td>PPO</td><td>TD-MPC2</td><td>DreamerV3</td><td>MR.Q</td></tr><tr><td>acrobot-swingup</td><td>168 [127, 219]</td><td>2 [1, 4]</td><td>197 [179, 217]</td><td>121 [106, 145]</td><td>287 [254, 316]</td></tr><tr><td>ball_in_cup-catch</td><td>909 [821, 973]</td><td>105 [5, 282]</td><td>932 [899, 961]</td><td>971 [969, 973]</td><td>977 [975, 980]</td></tr><tr><td>cartpole-balance</td><td>993 [990, 996]</td><td>353 [231, 485]</td><td>972 [948, 991]</td><td>998 [997, 1000]</td><td>999 [999, 999]</td></tr><tr><td>cartpole-balance_sparse</td><td>962 [887, 1000]</td><td>487 [233, 751]</td><td>1000 [1000, 1000]</td><td>999 [999, 1000]</td><td>1000 [1000, 1000]</td></tr><tr><td>cartpole-swingup</td><td>864 [854, 873]</td><td>596 [437, 723]</td><td>690 [521, 813]</td><td>725 [603, 807]</td><td>868 [860, 875]</td></tr><tr><td>cartpole-swingup_sparse</td><td>774 [741, 805]</td><td>0 [0, 0]</td><td>636 [404, 804]</td><td>547 [351, 726]</td><td>797 [777, 816]</td></tr><tr><td>cheetah-run</td><td>728 [701, 753]</td><td>155 [110, 210]</td><td>431 [267, 556]</td><td>618 [576, 661]</td><td>775 [752, 807]</td></tr><tr><td>dog-run</td><td>10 [9, 12]</td><td>11 [9, 14]</td><td>14 [10, 18]</td><td>9 [6, 14]</td><td>60 [44, 80]</td></tr><tr><td>dog-stand</td><td>43 [37, 49]</td><td>51 [48, 56]</td><td>117 [72, 148]</td><td>61 [30, 92]</td><td>216 [201, 232]</td></tr><tr><td>dog-trot</td><td>14 [11, 18]</td><td>13 [12, 15]</td><td>20 [14, 25]</td><td>14 [13, 16]</td><td>65 [55, 79]</td></tr><tr><td>dog-walk</td><td>22 [18, 29]</td><td>16 [14, 18]</td><td>22 [17, 28]</td><td>11 [11, 12]</td><td>77 [71, 83]</td></tr><tr><td>finger-spin</td><td>860 [787, 922]</td><td>241 [107, 377]</td><td>786 [492, 984]</td><td>656 [544, 765]</td><td>965 [938, 982]</td></tr><tr><td>finger-turn_easy</td><td>503 [399, 615]</td><td>189 [144, 233]</td><td>562 [317, 779]</td><td>491 [447, 542]</td><td>953 [927, 974]</td></tr><tr><td>finger-turn_hard</td><td>223 [121, 340]</td><td>60 [1, 120]</td><td>903 [870, 940]</td><td>494 [401, 571]</td><td>932 [905, 957]</td></tr><tr><td>fish-swim</td><td>84 [65, 107]</td><td>77 [64, 92]</td><td>43 [21, 64]</td><td>90 [84, 96]</td><td>79 [68, 93]</td></tr><tr><td>hopper-hop</td><td>224 [170, 278]</td><td>0 [0, 0]</td><td>187 [119, 238]</td><td>205 [125, 287]</td><td>270 [230, 315]</td></tr><tr><td>hopper-stand</td><td>917 [903, 931]</td><td>1 [0, 2]</td><td>582 [321, 794]</td><td>888 [875, 900]</td><td>852 [703, 930]</td></tr><tr><td>humanoid-run</td><td>1 [1, 1]</td><td>1 [1, 1]</td><td>0 [1, 1]</td><td>1 [1, 1]</td><td>1 [1, 2]</td></tr><tr><td>humanoid-stand</td><td>6 [7, 7]</td><td>6 [6, 7]</td><td>5 [5, 7]</td><td>5 [5, 7]</td><td>7 [7, 8]</td></tr><tr><td>humanoid-walk</td><td>2 [2, 2]</td><td>1 [1, 1]</td><td>1 [1, 2]</td><td>1 [2, 2]</td><td>2 [2, 3]</td></tr><tr><td>pendulum-swingup</td><td>838 [813, 861]</td><td>0 [0, 1]</td><td>748 [574, 850]</td><td>761 [709, 807]</td><td>829 [816, 842]</td></tr><tr><td>quadruped-run</td><td>459 [412, 507]</td><td>118 [98, 139]</td><td>262 [184, 330]</td><td>328 [255, 397]</td><td>498 [476, 522]</td></tr><tr><td>quadruped-walk</td><td>750 [699, 796]</td><td>149 [113, 184]</td><td>246 [179, 310]</td><td>316 [260, 379]</td><td>833 [797, 867]</td></tr><tr><td>reacher-easy</td><td>938 [903, 973]</td><td>113 [55, 192]</td><td>956 [932, 978]</td><td>735 [678, 796]</td><td>979 [978, 982]</td></tr><tr><td>reacher-hard</td><td>705 [580, 831]</td><td>10 [0, 30]</td><td>911 [867, 946]</td><td>338 [227, 461]</td><td>965 [945, 977]</td></tr><tr><td>walker-run</td><td>546 [475, 612]</td><td>39 [35, 44]</td><td>665 [566, 719]</td><td>669 [615, 708]</td><td>615 [571, 655]</td></tr><tr><td>walker-stand</td><td>980 [977, 984]</td><td>253 [210, 310]</td><td>937 [907, 962]</td><td>969 [966, 973]</td><td>980 [977, 985]</td></tr><tr><td>walker-walk</td><td>766 [489, 957]</td><td>47 [40, 56]</td><td>958 [952, 965]</td><td>942 [936, 949]</td><td>970 [968, 973]</td></tr><tr><td>Mean</td><td>510 [497, 523]</td><td>110 [98, 125]</td><td>492 [471, 512]</td><td>463 [452, 475]</td><td>602 [595, 608]</td></tr><tr><td>Median</td><td>626 [528, 665]</td><td>49 [32, 53]</td><td>572 [419, 654]</td><td>493 [420, 532]</td><td>813 [779, 822]</td></tr><tr><td>IQM</td><td>545 [519, 564]</td><td>58 [46, 67]</td><td>501 [458, 537]</td><td>452 [430, 473]</td><td>692 [678, 703]</td></tr></table>

![](images/0c0acb16b58bf7170ace6077d22319ed878d4346114693cc0f04f4680f793519.jpg)  
MR.Q DreamerV3 TD-MPC2 PPO DrQ-v2

Figure 5: DMC - Visual learning curves. Time steps consider the number of environment interactions, where 500k time steps equals 1M frames in the original environment. Results are over 10 seeds. The shaded area captures a 95% boostrap confidence interval.

# C.4 ATARI

Table 7: Atari final results. Final average performance at 2.5M time steps (10M time steps in the original environment due to action repeat) over 10 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are computed over the human-normalized score. 

<table><tr><td>Task</td><td>DQN</td><td>Rainbow</td><td>PPO</td><td>DreamerV3</td><td>MR.Q</td></tr><tr><td>Alien</td><td>925 [879, 968]</td><td>1220 [1191, 1268]</td><td>320 [251, 383]</td><td>4838 [3863, 5813]</td><td>2834 [2241, 3388]</td></tr><tr><td>Amidar</td><td>178 [169, 186]</td><td>301 [280, 330]</td><td>126 [90, 167]</td><td>470 [419, 524]</td><td>595 [525, 657]</td></tr><tr><td>Assault</td><td>988 [957, 1011]</td><td>1430 [1392, 1475]</td><td>423 [271, 581]</td><td>3518 [2969, 4179]</td><td>1296 [1254, 1343]</td></tr><tr><td>Asterix</td><td>2381 [2313, 2469]</td><td>2699 [2598, 2783]</td><td>296 [216, 403]</td><td>7319 [6251, 8354]</td><td>3358 [3004, 3797]</td></tr><tr><td>Asteroids</td><td>423 [408, 436]</td><td>754 [711, 816]</td><td>206 [180, 232]</td><td>1359 [1243, 1482]</td><td>715 [638, 796]</td></tr><tr><td>Atlantis</td><td>7365 [6893, 7742]</td><td>80837 [51139, 126780]</td><td>2000 [2000, 2000]</td><td>664529 [197588, 973362]</td><td>556845 [469425, 660043]</td></tr><tr><td>BankHeist</td><td>474 [448, 493]</td><td>895 [889, 901]</td><td>187 [41, 421]</td><td>801 [691, 1002]</td><td>809 [639, 960]</td></tr><tr><td>BattleZone</td><td>3598 [3235, 3878]</td><td>20209 [17157, 22375]</td><td>2200 [1460, 3100]</td><td>22599 [21055, 24669]</td><td>19880 [13450, 26060]</td></tr><tr><td>BeamRider</td><td>869 [728, 1065]</td><td>5982 [5664, 6268]</td><td>479 [348, 581]</td><td>5635 [3161, 7962]</td><td>2299 [1921, 2813]</td></tr><tr><td>Berzerk</td><td>488 [466, 508]</td><td>443 [413, 484]</td><td>384 [310, 469]</td><td>758 [681, 823]</td><td>523 [456, 588]</td></tr><tr><td>Bowling</td><td>29 [27, 32]</td><td>44 [36, 52]</td><td>51 [38, 60]</td><td>101 [69, 138]</td><td>59 [45, 72]</td></tr><tr><td>Boxing</td><td>37 [31, 44]</td><td>68 [66, 71]</td><td>-3 [-6, 0]</td><td>97 [97, 99]</td><td>96 [95, 97]</td></tr><tr><td>Breakout</td><td>21 [19, 25]</td><td>41 [40, 44]</td><td>9 [8, 11]</td><td>137 [110, 162]</td><td>34 [28, 42]</td></tr><tr><td>Centipede</td><td>2832 [2418, 3215]</td><td>4992 [4784, 5138]</td><td>4239 [2222, 6622]</td><td>20067 [17410, 22758]</td><td>17835 [16161, 19817]</td></tr><tr><td>ChopperCommand</td><td>997 [971, 1022]</td><td>2265 [2160, 2357]</td><td>688 [501, 878]</td><td>15172 [12940, 17219]</td><td>5748 [4822, 6651]</td></tr><tr><td>CrazyClimber</td><td>64611 [46203, 78709]</td><td>103539 [99749, 106850]</td><td>896 [174, 1727]</td><td>132811 [128446, 135930]</td><td>116954 [111371, 122032]</td></tr><tr><td>Defender</td><td>116954 [111371, 122032]</td><td>116954 [111371, 122032]</td><td>1333 [705, 2094]</td><td>34187 [29814, 39261]</td><td>40457 [36892, 43638]</td></tr><tr><td>DemonAttack</td><td>1503 [1282, 1690]</td><td>2477 [2269, 2678]</td><td>139 [116, 165]</td><td>4836 [3443, 6231]</td><td>5924 [4491, 7289]</td></tr><tr><td>DoubleDunk</td><td>-18 [-20, -18]</td><td>-18 [-19, -19]</td><td>-1 [-3, 0]</td><td>21 [20, 22]</td><td>-10 [-15, -9]</td></tr><tr><td>Enduro</td><td>589 [567, 617]</td><td>1601 [1555, 1635]</td><td>13 [9, 17]</td><td>476 [175, 782]</td><td>1845 [1758, 1938]</td></tr><tr><td>FishingDerby</td><td>-42 [-62, -17]</td><td>10 [5, 15]</td><td>-89 [-91, -87]</td><td>40 [32, 47]</td><td>10 [2, 18]</td></tr><tr><td>Freeway</td><td>8 [0, 19]</td><td>32 [32, 32]</td><td>15 [11, 18]</td><td>19 [6, 32]</td><td>32 [32, 32]</td></tr><tr><td>Frostbite</td><td>269 [238, 294]</td><td>2510 [2040, 2823]</td><td>245 [231, 259]</td><td>5183 [2151, 8291]</td><td>4561 [3299, 5740]</td></tr><tr><td>Gopher</td><td>1470 [1316, 1590]</td><td>4279 [4139, 4425]</td><td>126 [80, 174]</td><td>38711 [26066, 48187]</td><td>19174 [14932, 23587]</td></tr><tr><td>Gravitar</td><td>167 [153, 183]</td><td>202 [184, 218]</td><td>63 [31, 98]</td><td>831 [768, 900]</td><td>397 [320, 490]</td></tr><tr><td>Hero</td><td>2679 [2404, 2945]</td><td>9323 [7914, 10863]</td><td>1741 [1062, 2302]</td><td>20582 [19845, 21583]</td><td>13450 [11915, 14781]</td></tr><tr><td>IceHockey</td><td>-9 [-10, -9]</td><td>-5 [-6, -5]</td><td>-8 [-10, -8]</td><td>14 [13, 16]</td><td>0 [-1, 2]</td></tr><tr><td>Jamesbond</td><td>47 [42, 52]</td><td>514 [509, 520]</td><td>85 [62, 106]</td><td>836 [568, 1119]</td><td>624 [588, 662]</td></tr><tr><td>Kangaroo</td><td>539 [525, 553]</td><td>5501 [3853, 7151]</td><td>402 [280, 520]</td><td>8825 [5234, 12418]</td><td>9807 [7851, 11591]</td></tr><tr><td>Krull</td><td>4229 [3942, 4490]</td><td>5972 [5903, 6047]</td><td>421 [136, 735]</td><td>23092 [14679, 28172]</td><td>9309 [8646, 9953]</td></tr><tr><td>KungFuMaster</td><td>15997 [13182, 18813]</td><td>18074 [16041, 20864]</td><td>52 [18, 95]</td><td>70703 [50114, 94578]</td><td>29369 [26954, 31595]</td></tr><tr><td>MontezumaRevenge</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>1310 [598, 2180]</td><td>50 [0, 140]</td></tr><tr><td>MsPacman</td><td>2187 [2121, 2247]</td><td>2347 [2292, 2403]</td><td>457 [352, 578]</td><td>4484 [3539, 5511]</td><td>4922 [4191, 5843]</td></tr><tr><td>NameThisGame</td><td>4000 [3814, 4187]</td><td>8604 [8252, 8931]</td><td>1084 [663, 1501]</td><td>15742 [14542, 17103]</td><td>8693 [8071, 9199]</td></tr><tr><td>Phoenix</td><td>4948 [4236, 5627]</td><td>4830 [4707, 4968]</td><td>101 [81, 120]</td><td>15827 [14903, 16429]</td><td>5173 [5025, 5322]</td></tr><tr><td>Pitfall</td><td>-60 [-89, -35]</td><td>-14 [-29, -6]</td><td>-16 [-38, -2]</td><td>0 [0, 0]</td><td>-20 [-60, 0]</td></tr><tr><td>Pong</td><td>-4 [-14, 3]</td><td>15 [14, 16]</td><td>-5 [-8, -3]</td><td>16 [16, 17]</td><td>17 [16, 19]</td></tr><tr><td>PrivateEye</td><td>118 [78, 181]</td><td>111 [78, 166]</td><td>-17 [-592, 762]</td><td>3046 [975, 5118]</td><td>100 [100, 100]</td></tr><tr><td>Qbert</td><td>1658 [1246, 2139]</td><td>5353 [4363, 6783]</td><td>484 [393, 570]</td><td>16807 [16073, 17564]</td><td>3938 [3210, 4327]</td></tr><tr><td>Riverraid</td><td>3198 [3167, 3222]</td><td>4272 [4060, 4440]</td><td>1045 [833, 1241]</td><td>9160 [8177, 10077]</td><td>10791 [9307, 12511]</td></tr><tr><td>RoadRunner</td><td>27980 [27269, 28692]</td><td>33412 [32459, 34435]</td><td>723 [454, 940]</td><td>66453 [40606, 104163]</td><td>49579 [47425, 51426]</td></tr><tr><td>Robotank</td><td>4 [4, 5]</td><td>19 [18, 20]</td><td>4 [2, 6]</td><td>51 [47, 55]</td><td>13 [12, 15]</td></tr><tr><td>Seaquest</td><td>299 [277, 318]</td><td>1641 [1621, 1661]</td><td>250 [214, 282]</td><td>3416 [2665, 4426]</td><td>3522 [2401, 4850]</td></tr><tr><td>Skiing</td><td>-19568 [-19793, -19362]</td><td>-24070 [-25305, -22667]</td><td>-27901 [-30000, -23704]</td><td>-30043 [-30394, -29764]</td><td>-30000 [-30000, -30000]</td></tr><tr><td>Solaris</td><td>1645 [1480, 1804]</td><td>1289 [1143, 1451]</td><td>0 [0, 2]</td><td>2340 [1882, 2799]</td><td>1103 [799, 1430]</td></tr><tr><td>SpaceInvaders</td><td>663 [651, 675]</td><td>743 [721, 764]</td><td>294 [235, 354]</td><td>1433 [1039, 1943]</td><td>701 [626, 768]</td></tr><tr><td>StarGunner</td><td>692 [662, 719]</td><td>1488 [1470, 1506]</td><td>415 [316, 499]</td><td>2090 [1678, 2649]</td><td>3488 [1032, 8241]</td></tr><tr><td>Surround</td><td>3488 [1032, 8241]</td><td>3488 [1032, 8241]</td><td>-9 [-10, -10]</td><td>5 [4, 7]</td><td>-2 [-4, -2]</td></tr><tr><td>Tennis</td><td>-21 [-24, -19]</td><td>-1 [-2, 0]</td><td>-20 [-22, -19]</td><td>-3 [-11, 0]</td><td>0 [0, 0]</td></tr><tr><td>TimePilot</td><td>1539 [1479, 1613]</td><td>2703 [2627, 2787]</td><td>548 [450, 690]</td><td>7779 [3128, 13016]</td><td>4382 [4208, 4528]</td></tr><tr><td>Tutankham</td><td>112 [97, 123]</td><td>179 [165, 191]</td><td>29 [17, 43]</td><td>253 [240, 269]</td><td>164 [145, 185]</td></tr><tr><td>UpNDown</td><td>7669 [7116, 8147]</td><td>12397 [11489, 13312]</td><td>595 [428, 737]</td><td>284807 [178615, 391388]</td><td>73095 [40836, 108810]</td></tr><tr><td>Venture</td><td>25 [6, 45]</td><td>19 [14, 25]</td><td>2 [0, 6]</td><td>0 [0, 0]</td><td>112 [0, 304]</td></tr><tr><td>VideoPinball</td><td>5129 [4611, 5649]</td><td>26245 [23075, 29067]</td><td>1005 [0, 2485]</td><td>22345 [20669, 23955]</td><td>53826 [40600, 67972]</td></tr><tr><td>WizardOfWor</td><td>481 [396, 542]</td><td>2213 [1827, 2617]</td><td>225 [185, 264]</td><td>7086 [6518, 7730]</td><td>2599 [2259, 2942]</td></tr><tr><td>YarsRevenge</td><td>9426 [9177, 9656]</td><td>10708 [10405, 11071]</td><td>1891 [925, 2964]</td><td>62209 [57783, 67113]</td><td>34861 [29734, 40020]</td></tr><tr><td>Zaxxon</td><td>112 [15, 230]</td><td>3661 [3131, 4192]</td><td>0 [0, 0]</td><td>17347 [15320, 19385]</td><td>8850 [8045, 9740]</td></tr><tr><td>Mean</td><td>0.25 [0.24, 0.26]</td><td>1.08 [1.02, 1.14]</td><td>-0.09 [-0.10, -0.07]</td><td>3.74 [3.29, 4.13]</td><td>2.54 [2.34, 2.75]</td></tr><tr><td>Median</td><td>0.12 [0.10, 0.12]</td><td>0.40 [0.40, 0.47]</td><td>0.01 [0.00, 0.01]</td><td>1.25 [1.11, 1.47]</td><td>0.96 [0.78, 0.98]</td></tr><tr><td>IQM</td><td>0.17 [0.16, 0.17]</td><td>0.61 [0.60, 0.62]</td><td>0.02 [0.01, 0.02]</td><td>1.46 [1.34, 1.51]</td><td>0.90 [0.88, 0.94]</td></tr></table>

![](images/ad5dceb60a23aecfc372fe134ea3d5aedf4fe3ff331d9e738c4c293d7183273b.jpg)  
Figure 6: Atari learning curves. Time steps consider the number of environment interactions, where 2.5M time steps equals 10M frames in the original environment. Results are over 10 seeds. The shaded area captures a 95% boostrap confidence interval.

# D COMPLETE ABLATION RESULTS

In this section, we show a per-environment breakdown of each variation in the design study in Section 5.2. Each table reports the raw score for each environment. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are computed over the the difference in the normalized score. We use TD3 to normalize for Gym, raw scores divided by 1000 for DMC and human scores to normalize for Atari (see Appendix B.3). Highlighting is used to designate the scale of the difference in normalized score:

≤ −0.5   
−0.2, −0.5   
[ )• −0.01, −0.2   
• 0.01, 0.2   
0.2, 0.5   
≥ 0.5

D.1 GYM 

<table><tr><td>Task</td><td>MR.Q</td><td>Linear value function</td><td>Dynamics target</td><td>No target encoder</td></tr><tr><td>Ant</td><td>6901 [6261, 7482]</td><td>1844 [1663, 2018]</td><td>5867 [5543, 6289]</td><td>3970 [2468, 5509]</td></tr><tr><td>HalfCheetah</td><td>12939 [11663, 13762]</td><td>3383 [3054, 3732]</td><td>14019 [13746, 14285]</td><td>12838 [12459, 13266]</td></tr><tr><td>Hopper</td><td>2692 [2131, 3309]</td><td>968 [720, 1210]</td><td>2890 [2030, 3747]</td><td>3007 [2164, 3852]</td></tr><tr><td>Humanoid</td><td>10223 [9929, 10498]</td><td>461 [395, 532]</td><td>8370 [7651, 8988]</td><td>305 [272, 356]</td></tr><tr><td>Walker2d</td><td>6039 [5644, 6386]</td><td>1117 [999, 1238]</td><td>5844 [5146, 6477]</td><td>5944 [5570, 6323]</td></tr><tr><td>Mean</td><td>-</td><td>-1.17 [-1.19, -1.15]</td><td>-0.10 [-0.17, -0.04]</td><td>-0.53 [-0.60, -0.46]</td></tr><tr><td>Median</td><td>-</td><td>-1.25 [-1.28, -1.21]</td><td>-0.05 [-0.23, 0.09]</td><td>-0.02 [-0.16, 0.02]</td></tr><tr><td>IQM</td><td>-</td><td>-1.13 [-1.15, -1.11]</td><td>-0.08 [-0.19, 0.01]</td><td>-0.25 [-0.37, -0.16]</td></tr></table>

<table><tr><td>Task</td><td>MR.Q</td><td>Revert</td><td>Non-linear model</td><td>MSE reward loss</td></tr><tr><td>Ant</td><td>6901 [6261, 7482]</td><td>-422 [-1770, 846]</td><td>7215 [6971, 7466]</td><td>7153 [5991, 7815]</td></tr><tr><td>HalfCheetah</td><td>12939 [11663, 13762]</td><td>-658 [-750, -604]</td><td>13370 [12649, 14053]</td><td>14413 [14096, 14710]</td></tr><tr><td>Hopper</td><td>2692 [2131, 3309]</td><td>103 [39, 189]</td><td>2492 [1835, 3424]</td><td>2869 [2090, 3689]</td></tr><tr><td>Humanoid</td><td>10223 [9929, 10498]</td><td>189 [104, 277]</td><td>10257 [9612, 10688]</td><td>10592 [10017, 10983]</td></tr><tr><td>Walker2d</td><td>6039 [5644, 6386]</td><td>260 [-5, 638]</td><td>5548 [4980, 6117]</td><td>6626 [5256, 7984]</td></tr><tr><td>Mean</td><td>-</td><td>-1.47 [-1.54, -1.39]</td><td>-0.01 [-0.07, 0.03]</td><td>0.10 [-0.02, 0.19]</td></tr><tr><td>Median</td><td>-</td><td>-1.47 [-1.53, -1.37]</td><td>0.01 [-0.03, 0.08]</td><td>0.07 [0.00, 0.17]</td></tr><tr><td>IQM</td><td>-</td><td>-1.51 [-1.58, -1.39]</td><td>-0.01 [-0.05, 0.04]</td><td>0.09 [-0.01, 0.18]</td></tr></table>

<table><tr><td>Task</td><td>MR.Q</td><td>No reward scaling</td><td>No min</td><td>No LAP</td></tr><tr><td>Ant</td><td>6901 [6261, 7482]</td><td>6866 [6227, 7547]</td><td>6936 [6582, 7329]</td><td>6817 [6616, 7039]</td></tr><tr><td>HalfCheetah</td><td>12939 [11663, 13762]</td><td>13502 [13333, 13673]</td><td>14143 [13819, 14515]</td><td>13185 [13085, 13299]</td></tr><tr><td>Hopper</td><td>2692 [2131, 3309]</td><td>2551 [2090, 3064]</td><td>2113 [1728, 2626]</td><td>2681 [1883, 3465]</td></tr><tr><td>Humanoid</td><td>10223 [9929, 10498]</td><td>9515 [8520, 10245]</td><td>10528 [10202, 10837]</td><td>8441 [6206, 9738]</td></tr><tr><td>Walker2d</td><td>6039 [5644, 6386]</td><td>5743 [5362, 6102]</td><td>4293 [3547, 5107]</td><td>5463 [4134, 6376]</td></tr><tr><td>Mean</td><td>-</td><td>-0.04 [-0.09, 0.02]</td><td>-0.09 [-0.16, -0.01]</td><td>-0.10 [-0.24, -0.00]</td></tr><tr><td>Median</td><td>-</td><td>-0.04 [-0.11, 0.04]</td><td>0.01 [-0.08, 0.07]</td><td>-0.02 [-0.12, 0.02]</td></tr><tr><td>IQM</td><td>-</td><td>-0.04 [-0.10, 0.03]</td><td>-0.04 [-0.10, 0.02]</td><td>-0.06 [-0.18, 0.00]</td></tr></table>

<table><tr><td>Task</td><td>MR.Q</td><td>No MR</td><td>1-step return</td><td>No unroll</td></tr><tr><td>Ant</td><td>6901 [6261, 7482]</td><td>4195 [2573, 5819]</td><td>7757 [7729, 7799]</td><td>7528 [7224, 7830]</td></tr><tr><td>HalfCheetah</td><td>12939 [11663, 13762]</td><td>11249 [9238, 12495]</td><td>13123 [10691, 14653]</td><td>14409 [13817, 15002]</td></tr><tr><td>Hopper</td><td>2692 [2131, 3309]</td><td>1877 [1524, 2153]</td><td>2737 [2131, 3343]</td><td>2578 [1857, 3414]</td></tr><tr><td>Humanoid</td><td>10223 [9929, 10498]</td><td>3942 [3262, 4624]</td><td>2328 [1491, 3337]</td><td>10617 [10504, 10731]</td></tr><tr><td>Walker2d</td><td>6039 [5644, 6386]</td><td>4155 [3251, 4897]</td><td>4747 [3197, 6229]</td><td>6077 [5752, 6355]</td></tr><tr><td>Mean</td><td>-</td><td>-0.56 [-0.69, -0.43]</td><td>-0.33 [-0.46, -0.21]</td><td>0.07 [0.01, 0.14]</td></tr><tr><td>Median</td><td>-</td><td>-0.48 [-0.71, -0.27]</td><td>0.01 [-0.21, 0.16]</td><td>0.08 [0.06, 0.16]</td></tr><tr><td>IQM</td><td>-</td><td>-0.47 [-0.66, -0.28]</td><td>-0.10 [-0.32, 0.12]</td><td>0.07 [0.04, 0.15]</td></tr></table>

D.2 DMC - PROPRIOCEPTIVE 

<table><tr><td>Task</td><td>MR.Q</td><td>Linear value function</td><td>Dynamics target</td><td>No target encoder</td></tr><tr><td>acrobot-swingup</td><td>567 [517, 621]</td><td>30 [15, 46]</td><td>626 [578, 684]</td><td>16 [9, 25]</td></tr><tr><td>ball_in_cup-catch</td><td>981 [979, 983]</td><td>820 [658, 922]</td><td>980 [978, 983]</td><td>569 [436, 719]</td></tr><tr><td>cartpole-balance</td><td>999 [999, 1000]</td><td>449 [380, 520]</td><td>999 [999, 1000]</td><td>992 [986, 997]</td></tr><tr><td>cartpole-balance_sparse</td><td>1000 [1000, 1000]</td><td>183 [142, 224]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td></tr><tr><td>cartpole-swingup</td><td>866 [866, 866]</td><td>267 [225, 310]</td><td>869 [866, 876]</td><td>852 [840, 861]</td></tr><tr><td>cartpole-swingup_sparse</td><td>798 [779, 818]</td><td>12 [5, 22]</td><td>817 [809, 832]</td><td>0 [0, 0]</td></tr><tr><td>cheetah-run</td><td>914 [911, 917]</td><td>394 [376, 411]</td><td>919 [919, 921]</td><td>904 [899, 910]</td></tr><tr><td>dog-run</td><td>569 [546, 595]</td><td>11 [5, 18]</td><td>254 [202, 305]</td><td>11 [8, 16]</td></tr><tr><td>dog-stand</td><td>967 [959, 975]</td><td>22 [17, 29]</td><td>672 [520, 830]</td><td>36 [27, 51]</td></tr><tr><td>dog-trot</td><td>877 [845, 898]</td><td>15 [11, 20]</td><td>319 [279, 365]</td><td>12 [8, 19]</td></tr><tr><td>dog-walk</td><td>916 [908, 924]</td><td>13 [11, 18]</td><td>312 [247, 396]</td><td>10 [7, 15]</td></tr><tr><td>finger-spin</td><td>937 [917, 958]</td><td>736 [670, 825]</td><td>942 [916, 971]</td><td>869 [698, 963]</td></tr><tr><td>finger-turn_easy</td><td>953 [928, 975]</td><td>238 [157, 319]</td><td>947 [900, 979]</td><td>624 [509, 756]</td></tr><tr><td>finger-turn_hard</td><td>950 [908, 974]</td><td>23 [1, 63]</td><td>923 [878, 966]</td><td>431 [301, 563]</td></tr><tr><td>fish-swim</td><td>792 [772, 811]</td><td>83 [65, 102]</td><td>410 [307, 493]</td><td>97 [65, 129]</td></tr><tr><td>hopper-hop</td><td>251 [201, 295]</td><td>10 [4, 17]</td><td>199 [131, 265]</td><td>0 [0, 1]</td></tr><tr><td>hopper-stand</td><td>951 [948, 955]</td><td>66 [30, 102]</td><td>639 [413, 866]</td><td>4 [3, 7]</td></tr><tr><td>humanoid-run</td><td>200 [169, 236]</td><td>1 [1, 1]</td><td>1 [1, 1]</td><td>1 [1, 1]</td></tr><tr><td>humanoid-stand</td><td>868 [823, 907]</td><td>6 [5, 7]</td><td>7 [6, 8]</td><td>7 [6, 9]</td></tr><tr><td>humanoid-walk</td><td>662 [609, 721]</td><td>1 [1, 2]</td><td>2 [2, 3]</td><td>2 [2, 3]</td></tr><tr><td>pendulum-swingup</td><td>748 [594, 830]</td><td>357 [114, 617]</td><td>826 [812, 840]</td><td>784 [706, 834]</td></tr><tr><td>quadruped-run</td><td>947 [940, 954]</td><td>172 [93, 252]</td><td>942 [930, 951]</td><td>829 [757, 895]</td></tr><tr><td>quadruped-walk</td><td>963 [959, 968]</td><td>91 [52, 141]</td><td>939 [935, 943]</td><td>952 [946, 960]</td></tr><tr><td>reacher-easy</td><td>983 [983, 985]</td><td>802 [722, 877]</td><td>984 [983, 986]</td><td>983 [981, 986]</td></tr><tr><td>reacher-hard</td><td>977 [975, 979]</td><td>853 [778, 914]</td><td>970 [965, 976]</td><td>975 [972, 979]</td></tr><tr><td>walker-run</td><td>793 [766, 816]</td><td>238 [207, 274]</td><td>730 [585, 814]</td><td>776 [757, 792]</td></tr><tr><td>walker-stand</td><td>988 [987, 990]</td><td>859 [780, 921]</td><td>988 [985, 991]</td><td>988 [986, 991]</td></tr><tr><td>walker-walk</td><td>978 [978, 980]</td><td>504 [397, 613]</td><td>975 [975, 977]</td><td>974 [973, 976]</td></tr><tr><td>Mean</td><td>-</td><td>-0.58 [-0.59, -0.56]</td><td>-0.15 [-0.15, -0.15]</td><td>-0.35 [-0.35, -0.34]</td></tr><tr><td>Median</td><td>-</td><td>-0.58 [-0.64, -0.57]</td><td>-0.01 [-0.02, -0.00]</td><td>-0.22 [-0.23, -0.21]</td></tr><tr><td>IQM</td><td>-</td><td>-0.62 [-0.64, -0.60]</td><td>-0.05 [-0.06, -0.03]</td><td>-0.27 [-0.29, -0.25]</td></tr></table>

<table><tr><td>Task</td><td>MR.Q</td><td>Revert</td><td>Non-linear model</td><td>MSE reward loss</td></tr><tr><td>acrobot-swingup</td><td>567 [517, 621]</td><td>11 [8, 18]</td><td>553 [478, 629]</td><td>577 [547, 612]</td></tr><tr><td>ball_in_cup-catch</td><td>981 [979, 983]</td><td>301 [233, 365]</td><td>982 [982, 984]</td><td>983 [982, 985]</td></tr><tr><td>cartpole-balance</td><td>999 [999, 1000]</td><td>272 [206, 332]</td><td>999 [999, 1000]</td><td>999 [998, 1000]</td></tr><tr><td>cartpole-balance_sparse</td><td>1000 [1000, 1000]</td><td>197 [177, 214]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td></tr><tr><td>cartpole-swingup</td><td>866 [866, 866]</td><td>191 [99, 263]</td><td>866 [866, 867]</td><td>865 [865, 866]</td></tr><tr><td>cartpole-swingup_sparse</td><td>798 [779, 818]</td><td>0 [0, 0]</td><td>824 [809, 839]</td><td>812 [786, 834]</td></tr><tr><td>cheetah-run</td><td>914 [911, 917]</td><td>74 [31, 123]</td><td>909 [906, 913]</td><td>910 [907, 915]</td></tr><tr><td>dog-run</td><td>569 [546, 595]</td><td>3 [4, 4]</td><td>588 [516, 646]</td><td>527 [513, 545]</td></tr><tr><td>dog-stand</td><td>967 [959, 975]</td><td>22 [18, 29]</td><td>962 [938, 982]</td><td>964 [958, 971]</td></tr><tr><td>dog-trot</td><td>877 [845, 898]</td><td>4 [3, 5]</td><td>868 [816, 914]</td><td>861 [831, 886]</td></tr><tr><td>dog-walk</td><td>916 [908, 924]</td><td>4 [4, 6]</td><td>920 [915, 925]</td><td>724 [473, 882]</td></tr><tr><td>finger-spin</td><td>937 [917, 958]</td><td>0 [0, 1]</td><td>868 [767, 951]</td><td>907 [875, 947]</td></tr><tr><td>finger-turn_easy</td><td>953 [928, 975]</td><td>159 [60, 280]</td><td>972 [968, 978]</td><td>935 [894, 977]</td></tr><tr><td>finger-turn_hard</td><td>950 [908, 974]</td><td>60 [20, 100]</td><td>931 [887, 975]</td><td>947 [910, 969]</td></tr><tr><td>fish-swim</td><td>792 [772, 811]</td><td>71 [53, 90]</td><td>790 [754, 824]</td><td>793 [766, 821]</td></tr><tr><td>hopper-hop</td><td>251 [201, 295]</td><td>0 [0, 1]</td><td>288 [222, 332]</td><td>174 [119, 230]</td></tr><tr><td>hopper-stand</td><td>951 [948, 955]</td><td>5 [3, 8]</td><td>848 [664, 945]</td><td>854 [707, 936]</td></tr><tr><td>humanoid-run</td><td>200 [169, 236]</td><td>1 [1, 1]</td><td>205 [191, 221]</td><td>91 [45, 121]</td></tr><tr><td>humanoid-stand</td><td>868 [823, 907]</td><td>7 [7, 8]</td><td>811 [712, 878]</td><td>214 [14, 566]</td></tr><tr><td>humanoid-walk</td><td>662 [609, 721]</td><td>1 [2, 2]</td><td>668 [590, 734]</td><td>77 [3, 224]</td></tr><tr><td>pendulum-swingup</td><td>748 [594, 830]</td><td>61 [20, 103]</td><td>819 [802, 836]</td><td>827 [811, 843]</td></tr><tr><td>quadruped-run</td><td>947 [940, 954]</td><td>87 [40, 145]</td><td>944 [934, 954]</td><td>949 [941, 956]</td></tr><tr><td>quadruped-walk</td><td>963 [959, 968]</td><td>81 [36, 130]</td><td>963 [961, 967]</td><td>966 [961, 971]</td></tr><tr><td>reacher-easy</td><td>983 [983, 985]</td><td>789 [727, 856]</td><td>983 [982, 984]</td><td>964 [927, 984]</td></tr><tr><td>reacher-hard</td><td>977 [975, 979]</td><td>526 [361, 695]</td><td>953 [914, 976]</td><td>976 [974, 978]</td></tr><tr><td>walker-run</td><td>793 [766, 816]</td><td>25 [22, 31]</td><td>795 [784, 811]</td><td>778 [709, 821]</td></tr><tr><td>walker-stand</td><td>988 [987, 990]</td><td>221 [179, 264]</td><td>983 [971, 991]</td><td>988 [987, 990]</td></tr><tr><td>walker-walk</td><td>978 [978, 980]</td><td>33 [23, 46]</td><td>974 [972, 978]</td><td>967 [948, 979]</td></tr><tr><td>Mean</td><td>-</td><td>-0.72 [-0.73, -0.72]</td><td>-0.00 [-0.02, 0.01]</td><td>-0.06 [-0.08, -0.05]</td></tr><tr><td>Median</td><td>-</td><td>-0.78 [-0.80, -0.76]</td><td>-0.00 [-0.00, 0.00]</td><td>-0.00 [-0.01, 0.00]</td></tr><tr><td>IQM</td><td>-</td><td>-0.78 [-0.78, -0.76]</td><td>-0.00 [-0.01, 0.00]</td><td>-0.01 [-0.02, -0.00]</td></tr><tr><td>Task</td><td>MR.Q</td><td>No reward scaling</td><td>No min</td><td>No LAP</td></tr><tr><td>acrobot-swingup</td><td>567 [517, 621]</td><td>593 [532, 672]</td><td>623 [572, 673]</td><td>566 [520, 612]</td></tr><tr><td>ball_in_cup-catch</td><td>981 [979, 983]</td><td>983 [982, 984]</td><td>982 [980, 984]</td><td>981 [980, 984]</td></tr><tr><td>cartpole-balance</td><td>999 [999, 1000]</td><td>998 [999, 999]</td><td>999 [1000, 1000]</td><td>999 [998, 1000]</td></tr><tr><td>cartpole-balance_sparse</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td><td>992 [982, 1000]</td></tr><tr><td>cartpole-swingup</td><td>866 [866, 866]</td><td>865 [864, 866]</td><td>868 [866, 874]</td><td>865 [865, 866]</td></tr><tr><td>cartpole-swingup_sparse</td><td>798 [779, 818]</td><td>647 [318, 822]</td><td>799 [780, 812]</td><td>796 [779, 810]</td></tr><tr><td>cheetah-run</td><td>914 [911, 917]</td><td>911 [909, 914]</td><td>910 [893, 921]</td><td>908 [905, 913]</td></tr><tr><td>dog-run</td><td>569 [546, 595]</td><td>586 [546, 613]</td><td>577 [540, 610]</td><td>536 [499, 573]</td></tr><tr><td>dog-stand</td><td>967 [959, 975]</td><td>959 [940, 979]</td><td>946 [917, 969]</td><td>971 [966, 976]</td></tr><tr><td>dog-trot</td><td>877 [845, 898]</td><td>817 [713, 903]</td><td>846 [767, 906]</td><td>842 [764, 897]</td></tr><tr><td>dog-walk</td><td>916 [908, 924]</td><td>901 [890, 917]</td><td>747 [447, 908]</td><td>899 [886, 914]</td></tr><tr><td>finger-spin</td><td>937 [917, 958]</td><td>873 [768, 947]</td><td>926 [907, 950]</td><td>915 [892, 948]</td></tr><tr><td>finger-turn_easy</td><td>953 [928, 975]</td><td>977 [973, 982]</td><td>976 [972, 980]</td><td>975 [967, 983]</td></tr><tr><td>finger-turn_hard</td><td>950 [908, 974]</td><td>946 [905, 969]</td><td>894 [833, 953]</td><td>949 [909, 972]</td></tr><tr><td>fish-swim</td><td>792 [772, 811]</td><td>745 [663, 809]</td><td>785 [763, 810]</td><td>788 [754, 826]</td></tr><tr><td>hopper-hop</td><td>251 [201, 295]</td><td>343 [263, 477]</td><td>336 [322, 352]</td><td>347 [265, 431]</td></tr><tr><td>hopper-stand</td><td>951 [948, 955]</td><td>934 [912, 948]</td><td>935 [926, 947]</td><td>941 [935, 948]</td></tr><tr><td>humanoid-run</td><td>200 [169, 236]</td><td>184 [149, 214]</td><td>198 [175, 225]</td><td>202 [191, 212]</td></tr><tr><td>humanoid-stand</td><td>868 [823, 907]</td><td>810 [655, 899]</td><td>833 [793, 871]</td><td>880 [856, 900]</td></tr><tr><td>humanoid-walk</td><td>662 [609, 721]</td><td>665 [589, 765]</td><td>597 [292, 808]</td><td>697 [561, 828]</td></tr><tr><td>pendulum-swingup</td><td>748 [594, 830]</td><td>816 [790, 838]</td><td>825 [811, 839]</td><td>815 [792, 836]</td></tr><tr><td>quadruped-run</td><td>947 [940, 954]</td><td>951 [944, 958]</td><td>946 [941, 951]</td><td>937 [927, 947]</td></tr><tr><td>quadruped-walk</td><td>963 [959, 968]</td><td>966 [961, 971]</td><td>959 [942, 972]</td><td>955 [942, 967]</td></tr><tr><td>reacher-easy</td><td>983 [983, 985]</td><td>964 [926, 984]</td><td>983 [981, 986]</td><td>983 [982, 986]</td></tr><tr><td>reacher-hard</td><td>977 [975, 979]</td><td>971 [968, 975]</td><td>978 [974, 982]</td><td>974 [969, 981]</td></tr><tr><td>walker-run</td><td>793 [766, 816]</td><td>804 [783, 820]</td><td>806 [779, 821]</td><td>812 [803, 822]</td></tr><tr><td>walker-stand</td><td>988 [987, 990]</td><td>989 [988, 990]</td><td>989 [986, 992]</td><td>986 [985, 987]</td></tr><tr><td>walker-walk</td><td>978 [978, 980]</td><td>979 [978, 980]</td><td>978 [976, 980]</td><td>977 [974, 980]</td></tr><tr><td>Mean</td><td>-</td><td>-0.01 [-0.02, 0.00]</td><td>-0.01 [-0.02, 0.01]</td><td>0.00 [-0.00, 0.01]</td></tr><tr><td>Median</td><td>-</td><td>-0.00 [-0.00, 0.00]</td><td>-0.00 [-0.00, 0.00]</td><td>-0.00 [-0.00, 0.00]</td></tr><tr><td>IQM</td><td>-</td><td>-0.00 [-0.00, 0.00]</td><td>-0.00 [-0.00, 0.00]</td><td>-0.00 [-0.00, 0.00]</td></tr></table>

<table><tr><td>Task</td><td>MR.Q</td><td>No MR</td><td>1-step return</td><td>No unroll</td></tr><tr><td>acrobot-swingup</td><td>567 [517, 621]</td><td>576 [483, 665]</td><td>440 [360, 528]</td><td>515 [455, 598]</td></tr><tr><td>ball_in_cup-catch</td><td>981 [979, 983]</td><td>981 [980, 984]</td><td>984 [983, 985]</td><td>982 [981, 984]</td></tr><tr><td>cartpole-balance</td><td>999 [999, 1000]</td><td>994 [991, 999]</td><td>999 [1000, 1000]</td><td>999 [999, 1000]</td></tr><tr><td>cartpole-balance_sparse</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td><td>961 [886, 1000]</td><td>1000 [1000, 1000]</td></tr><tr><td>cartpole-swingup</td><td>866 [866, 866]</td><td>870 [864, 878]</td><td>881 [879, 882]</td><td>864 [861, 867]</td></tr><tr><td>cartpole-swingup_sparse</td><td>798 [779, 818]</td><td>684 [528, 814]</td><td>845 [845, 847]</td><td>818 [811, 831]</td></tr><tr><td>cheetah-run</td><td>914 [911, 917]</td><td>871 [823, 907]</td><td>922 [921, 924]</td><td>909 [908, 911]</td></tr><tr><td>dog-run</td><td>569 [546, 595]</td><td>68 [63, 75]</td><td>299 [196, 360]</td><td>514 [473, 554]</td></tr><tr><td>dog-stand</td><td>967 [959, 975]</td><td>494 [452, 530]</td><td>606 [344, 865]</td><td>955 [944, 971]</td></tr><tr><td>dog-trot</td><td>877 [845, 898]</td><td>65 [49, 80]</td><td>725 [679, 756]</td><td>857 [833, 883]</td></tr><tr><td>dog-walk</td><td>916 [908, 924]</td><td>102 [81, 122]</td><td>788 [739, 832]</td><td>920 [905, 934]</td></tr><tr><td>finger-spin</td><td>937 [917, 958]</td><td>888 [731, 975]</td><td>983 [977, 988]</td><td>880 [781, 940]</td></tr><tr><td>finger-turn_easy</td><td>953 [928, 975]</td><td>947 [913, 974]</td><td>980 [979, 982]</td><td>950 [917, 976]</td></tr><tr><td>finger-turn_hard</td><td>950 [908, 974]</td><td>846 [756, 926]</td><td>968 [958, 976]</td><td>947 [907, 971]</td></tr><tr><td>fish-swim</td><td>792 [772, 811]</td><td>706 [683, 727]</td><td>498 [323, 651]</td><td>709 [618, 783]</td></tr><tr><td>hopper-hop</td><td>251 [201, 295]</td><td>85 [33, 142]</td><td>364 [336, 394]</td><td>297 [169, 442]</td></tr><tr><td>hopper-stand</td><td>951 [948, 955]</td><td>365 [233, 491]</td><td>952 [947, 959]</td><td>949 [944, 955]</td></tr><tr><td>humanoid-run</td><td>200 [169, 236]</td><td>1 [1, 2]</td><td>190 [124, 241]</td><td>192 [172, 214]</td></tr><tr><td>humanoid-stand</td><td>868 [823, 907]</td><td>201 [9, 517]</td><td>753 [665, 838]</td><td>858 [806, 913]</td></tr><tr><td>humanoid-walk</td><td>662 [609, 721]</td><td>84 [3, 247]</td><td>761 [689, 827]</td><td>675 [593, 772]</td></tr><tr><td>pendulum-swingup</td><td>748 [594, 830]</td><td>827 [812, 842]</td><td>823 [807, 841]</td><td>819 [793, 843]</td></tr><tr><td>quadruped-run</td><td>947 [940, 954]</td><td>871 [793, 933]</td><td>945 [940, 950]</td><td>950 [944, 955]</td></tr><tr><td>quadruped-walk</td><td>963 [959, 968]</td><td>951 [943, 962]</td><td>962 [958, 968]</td><td>962 [955, 969]</td></tr><tr><td>reacher-easy</td><td>983 [983, 985]</td><td>980 [979, 983]</td><td>984 [984, 986]</td><td>981 [976, 986]</td></tr><tr><td>reacher-hard</td><td>977 [975, 979]</td><td>949 [909, 974]</td><td>980 [979, 983]</td><td>954 [913, 978]</td></tr><tr><td>walker-run</td><td>793 [766, 816]</td><td>780 [769, 790]</td><td>835 [827, 843]</td><td>780 [702, 825]</td></tr><tr><td>walker-stand</td><td>988 [987, 990]</td><td>983 [980, 988]</td><td>990 [990, 992]</td><td>988 [988, 990]</td></tr><tr><td>walker-walk</td><td>978 [978, 980]</td><td>976 [975, 977]</td><td>979 [977, 982]</td><td>975 [969, 981]</td></tr><tr><td>Mean</td><td>-</td><td>-0.19 [-0.19, -0.18]</td><td>-0.04 [-0.05, -0.02]</td><td>-0.01 [-0.01, -0.00]</td></tr><tr><td>Median</td><td>-</td><td>-0.05 [-0.08, -0.01]</td><td>0.00 [0.00, 0.00]</td><td>-0.00 [-0.00, 0.00]</td></tr><tr><td>IQM</td><td>-</td><td>-0.06 [-0.08, -0.05]</td><td>0.00 [-0.00, 0.01]</td><td>-0.00 [-0.01, -0.00]</td></tr></table>

D.3 DMC - VISUAL 

<table><tr><td>Task</td><td>MR.Q</td><td>Linear value function</td><td>Dynamics target</td><td>No target encoder</td></tr><tr><td>acrobot-swingup</td><td>287 [253, 317]</td><td>15 [5, 22]</td><td>296 [281, 323]</td><td>16 [0, 38]</td></tr><tr><td>ball_in_cup-catch</td><td>977 [975, 980]</td><td>644 [328, 904]</td><td>972 [965, 976]</td><td>605 [496, 726]</td></tr><tr><td>cartpole-balance</td><td>999 [999, 999]</td><td>306 [254, 349]</td><td>998 [998, 999]</td><td>978 [947, 997]</td></tr><tr><td>cartpole-balance_sparse</td><td>1000 [1000, 1000]</td><td>243 [183, 327]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td></tr><tr><td>cartpole-swingup</td><td>868 [861, 875]</td><td>229 [181, 294]</td><td>861 [859, 865]</td><td>689 [487, 808]</td></tr><tr><td>cartpole-swingup_sparse</td><td>797 [777, 818]</td><td>4 [0, 14]</td><td>267 [0, 801]</td><td>0 [0, 0]</td></tr><tr><td>cheetah-run</td><td>775 [752, 805]</td><td>230 [159, 294]</td><td>831 [761, 875]</td><td>745 [723, 784]</td></tr><tr><td>dog-run</td><td>60 [44, 80]</td><td>19 [17, 22]</td><td>36 [34, 39]</td><td>10 [6, 20]</td></tr><tr><td>dog-stand</td><td>216 [201, 233]</td><td>76 [70, 82]</td><td>191 [155, 247]</td><td>60 [44, 89]</td></tr><tr><td>dog-trot</td><td>65 [54, 79]</td><td>19 [16, 24]</td><td>46 [42, 53]</td><td>9 [9, 10]</td></tr><tr><td>dog-walk</td><td>77 [70, 83]</td><td>30 [26, 33]</td><td>62 [57, 72]</td><td>16 [10, 21]</td></tr><tr><td>finger-spin</td><td>965 [938, 982]</td><td>789 [598, 923]</td><td>786 [672, 931]</td><td>929 [893, 981]</td></tr><tr><td>finger-turn_easy</td><td>953 [925, 974]</td><td>132 [98, 200]</td><td>876 [691, 969]</td><td>898 [855, 969]</td></tr><tr><td>finger-turn_hard</td><td>932 [905, 957]</td><td>66 [0, 100]</td><td>859 [777, 963]</td><td>492 [385, 577]</td></tr><tr><td>fish-swim</td><td>79 [67, 93]</td><td>69 [45, 109]</td><td>71 [38, 106]</td><td>65 [49, 84]</td></tr><tr><td>hopper-hop</td><td>270 [229, 317]</td><td>1 [0, 2]</td><td>184 [165, 204]</td><td>2 [0, 6]</td></tr><tr><td>hopper-stand</td><td>852 [705, 932]</td><td>7 [3, 11]</td><td>911 [900, 922]</td><td>5 [3, 9]</td></tr><tr><td>humanoid-run</td><td>1 [1, 2]</td><td>1 [1, 1]</td><td>1 [1, 2]</td><td>1 [1, 1]</td></tr><tr><td>humanoid-stand</td><td>7 [7, 8]</td><td>5 [4, 8]</td><td>6 [5, 8]</td><td>7 [6, 8]</td></tr><tr><td>humanoid-walk</td><td>2 [2, 3]</td><td>1 [1, 2]</td><td>2 [1, 3]</td><td>1 [1, 2]</td></tr><tr><td>pendulum-swingup</td><td>829 [815, 842]</td><td>97 [0, 192]</td><td>749 [632, 840]</td><td>191 [93, 287]</td></tr><tr><td>quadruped-run</td><td>498 [474, 523]</td><td>131 [77, 187]</td><td>488 [468, 517]</td><td>575 [566, 594]</td></tr><tr><td>quadruped-walk</td><td>833 [796, 868]</td><td>105 [57, 155]</td><td>717 [445, 895]</td><td>817 [790, 868]</td></tr><tr><td>reacher-easy</td><td>979 [978, 982]</td><td>605 [398, 868]</td><td>977 [973, 981]</td><td>979 [970, 986]</td></tr><tr><td>reacher-hard</td><td>965 [945, 977]</td><td>288 [195, 473]</td><td>975 [971, 978]</td><td>970 [963, 975]</td></tr><tr><td>walker-run</td><td>615 [571, 655]</td><td>158 [133, 193]</td><td>531 [463, 577]</td><td>611 [590, 631]</td></tr><tr><td>walker-stand</td><td>980 [977, 984]</td><td>707 [601, 881]</td><td>982 [981, 984]</td><td>984 [979, 988]</td></tr><tr><td>walker-walk</td><td>970 [968, 973]</td><td>350 [228, 529]</td><td>904 [850, 951]</td><td>965 [957, 972]</td></tr><tr><td>Mean</td><td>-</td><td>-0.41 [-0.42, -0.39]</td><td>-0.05 [-0.05, -0.04]</td><td>-0.15 [-0.15, -0.15]</td></tr><tr><td>Median</td><td>-</td><td>-0.37 [-0.44, -0.37]</td><td>-0.01 [-0.01, -0.00]</td><td>-0.03 [-0.05, -0.03]</td></tr><tr><td>IQM</td><td>-</td><td>-0.42 [-0.43, -0.38]</td><td>-0.02 [-0.02, -0.01]</td><td>-0.05 [-0.06, -0.04]</td></tr></table>

<table><tr><td>Task</td><td>MR.Q</td><td>Revert</td><td>Non-linear model</td><td>MSE reward loss</td></tr><tr><td>acrobot-swingup</td><td>287 [253, 317]</td><td>19 [13, 23]</td><td>279 [235, 314]</td><td>265 [242, 294]</td></tr><tr><td>ball_in_cup-catch</td><td>977 [975, 980]</td><td>195 [91, 297]</td><td>973 [970, 977]</td><td>974 [969, 978]</td></tr><tr><td>cartpole-balance</td><td>999 [999, 999]</td><td>190 [163, 212]</td><td>999 [999, 999]</td><td>998 [998, 1000]</td></tr><tr><td>cartpole-balance_sparse</td><td>1000 [1000, 1000]</td><td>346 [193, 639]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td></tr><tr><td>cartpole-swingup</td><td>868 [861, 875]</td><td>115 [82, 175]</td><td>849 [819, 873]</td><td>876 [875, 878]</td></tr><tr><td>cartpole-swingup_sparse</td><td>797 [777, 818]</td><td>0 [0, 0]</td><td>768 [684, 824]</td><td>33 [0, 60]</td></tr><tr><td>cheetah-run</td><td>775 [752, 805]</td><td>69 [36, 129]</td><td>763 [757, 770]</td><td>732 [712, 757]</td></tr><tr><td>dog-run</td><td>60 [44, 80]</td><td>3 [3, 4]</td><td>37 [36, 40]</td><td>36 [33, 38]</td></tr><tr><td>dog-stand</td><td>216 [201, 233]</td><td>17 [16, 19]</td><td>200 [193, 207]</td><td>195 [187, 208]</td></tr><tr><td>dog-trot</td><td>65 [54, 79]</td><td>5 [4, 6]</td><td>51 [47, 56]</td><td>47 [44, 50]</td></tr><tr><td>dog-walk</td><td>77 [70, 83]</td><td>6 [6, 8]</td><td>69 [62, 78]</td><td>63 [58, 71]</td></tr><tr><td>finger-spin</td><td>965 [938, 982]</td><td>1 [0, 2]</td><td>907 [841, 971]</td><td>924 [884, 980]</td></tr><tr><td>finger-turn_easy</td><td>953 [925, 974]</td><td>133 [99, 200]</td><td>932 [889, 977]</td><td>844 [786, 881]</td></tr><tr><td>finger-turn_hard</td><td>932 [905, 957]</td><td>66 [0, 100]</td><td>938 [903, 967]</td><td>900 [867, 964]</td></tr><tr><td>fish-swim</td><td>79 [67, 93]</td><td>53 [47, 60]</td><td>67 [55, 79]</td><td>75 [59, 105]</td></tr><tr><td>hopper-hop</td><td>270 [229, 317]</td><td>0 [0, 2]</td><td>308 [257, 368]</td><td>102 [14, 151]</td></tr><tr><td>hopper-stand</td><td>852 [705, 932]</td><td>4 [3, 8]</td><td>935 [929, 942]</td><td>919 [914, 925]</td></tr><tr><td>humanoid-run</td><td>1 [1, 2]</td><td>1 [1, 1]</td><td>1 [1, 1]</td><td>1 [1, 1]</td></tr><tr><td>humanoid-stand</td><td>7 [7, 8]</td><td>6 [5, 7]</td><td>6 [6, 8]</td><td>7 [7, 8]</td></tr><tr><td>humanoid-walk</td><td>2 [2, 3]</td><td>1 [1, 2]</td><td>2 [2, 3]</td><td>1 [1, 2]</td></tr><tr><td>pendulum-swingup</td><td>829 [815, 842]</td><td>66 [0, 100]</td><td>820 [795, 844]</td><td>581 [94, 842]</td></tr><tr><td>quadruped-run</td><td>498 [474, 523]</td><td>86 [69, 120]</td><td>555 [514, 578]</td><td>516 [478, 544]</td></tr><tr><td>quadruped-walk</td><td>833 [796, 868]</td><td>76 [34, 100]</td><td>762 [727, 788]</td><td>835 [751, 880]</td></tr><tr><td>reacher-easy</td><td>979 [978, 982]</td><td>583 [395, 684]</td><td>939 [900, 979]</td><td>979 [976, 984]</td></tr><tr><td>reacher-hard</td><td>965 [945, 977]</td><td>33 [0, 100]</td><td>868 [753, 956]</td><td>941 [879, 976]</td></tr><tr><td>walker-run</td><td>615 [571, 655]</td><td>35 [29, 39]</td><td>612 [593, 633]</td><td>596 [505, 643]</td></tr><tr><td>walker-stand</td><td>980 [977, 984]</td><td>280 [269, 294]</td><td>982 [977, 987]</td><td>983 [984, 984]</td></tr><tr><td>walker-walk</td><td>970 [968, 973]</td><td>22 [19, 25]</td><td>951 [917, 972]</td><td>969 [961, 976]</td></tr><tr><td>Mean</td><td>-</td><td>-0.52 [-0.52, -0.51]</td><td>-0.01 [-0.02, -0.00]</td><td>-0.05 [-0.07, -0.04]</td></tr><tr><td>Median</td><td>-</td><td>-0.68 [-0.72, -0.62]</td><td>-0.01 [-0.01, -0.00]</td><td>-0.01 [-0.01, 0.00]</td></tr><tr><td>IQM</td><td>-</td><td>-0.57 [-0.58, -0.56]</td><td>-0.01 [-0.01, -0.00]</td><td>-0.01 [-0.01, -0.00]</td></tr><tr><td>Task</td><td>MR.Q</td><td>No reward scaling</td><td>No min</td><td>No LAP</td></tr><tr><td>acrobot-swingup</td><td>287 [253, 317]</td><td>323 [275, 368]</td><td>332 [301, 391]</td><td>280 [235, 354]</td></tr><tr><td>ball_in_cup-catch</td><td>977 [975, 980]</td><td>973 [971, 977]</td><td>975 [974, 976]</td><td>971 [966, 978]</td></tr><tr><td>cartpole-balance</td><td>999 [999, 999]</td><td>999 [999, 999]</td><td>998 [998, 999]</td><td>998 [997, 999]</td></tr><tr><td>cartpole-balance_sparse</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td></tr><tr><td>cartpole-swingup</td><td>868 [861, 875]</td><td>860 [829, 877]</td><td>879 [879, 880]</td><td>798 [785, 821]</td></tr><tr><td>cartpole-swingup_sparse</td><td>797 [777, 818]</td><td>813 [805, 823]</td><td>805 [763, 829]</td><td>764 [736, 799]</td></tr><tr><td>cheetah-run</td><td>775 [752, 805]</td><td>720 [678, 758]</td><td>751 [734, 762]</td><td>706 [670, 741]</td></tr><tr><td>dog-run</td><td>60 [44, 80]</td><td>61 [49, 73]</td><td>42 [37, 51]</td><td>62 [45, 91]</td></tr><tr><td>dog-stand</td><td>216 [201, 233]</td><td>317 [239, 387]</td><td>228 [224, 232]</td><td>279 [229, 315]</td></tr><tr><td>dog-trot</td><td>65 [54, 79]</td><td>65 [55, 79]</td><td>50 [48, 53]</td><td>58 [56, 61]</td></tr><tr><td>dog-walk</td><td>77 [70, 83]</td><td>89 [83, 96]</td><td>86 [69, 106]</td><td>91 [85, 101]</td></tr><tr><td>finger-spin</td><td>965 [938, 982]</td><td>903 [776, 975]</td><td>870 [709, 982]</td><td>940 [864, 979]</td></tr><tr><td>finger-turn_easy</td><td>953 [925, 974]</td><td>873 [775, 954]</td><td>963 [952, 975]</td><td>844 [785, 879]</td></tr><tr><td>finger-turn_hard</td><td>932 [905, 957]</td><td>923 [885, 962]</td><td>933 [874, 976]</td><td>932 [858, 974]</td></tr><tr><td>fish-swim</td><td>79 [67, 93]</td><td>73 [59, 87]</td><td>54 [49, 61]</td><td>63 [49, 89]</td></tr><tr><td>hopper-hop</td><td>270 [229, 317]</td><td>244 [204, 298]</td><td>255 [244, 275]</td><td>186 [152, 204]</td></tr><tr><td>hopper-stand</td><td>852 [705, 932]</td><td>911 [888, 926]</td><td>923 [902, 945]</td><td>884 [877, 896]</td></tr><tr><td>humanoid-run</td><td>1 [1, 2]</td><td>1 [1, 1]</td><td>1 [1, 1]</td><td>1 [1, 2]</td></tr><tr><td>humanoid-stand</td><td>7 [7, 8]</td><td>7 [6, 8]</td><td>6 [5, 8]</td><td>10 [6, 18]</td></tr><tr><td>humanoid-walk</td><td>2 [2, 3]</td><td>2 [2, 4]</td><td>2 [2, 4]</td><td>2 [2, 3]</td></tr><tr><td>pendulum-swingup</td><td>829 [815, 842]</td><td>823 [798, 846]</td><td>831 [809, 843]</td><td>829 [808, 841]</td></tr><tr><td>quadruped-run</td><td>498 [474, 523]</td><td>505 [471, 545]</td><td>539 [500, 578]</td><td>463 [428, 485]</td></tr><tr><td>quadruped-walk</td><td>833 [796, 868]</td><td>823 [781, 867]</td><td>849 [745, 909]</td><td>799 [713, 905]</td></tr><tr><td>reacher-easy</td><td>979 [978, 982]</td><td>962 [924, 983]</td><td>953 [897, 981]</td><td>948 [885, 980]</td></tr><tr><td>reacher-hard</td><td>965 [945, 977]</td><td>972 [970, 975]</td><td>936 [872, 975]</td><td>973 [973, 974]</td></tr><tr><td>walker-run</td><td>615 [571, 655]</td><td>600 [544, 632]</td><td>666 [643, 682]</td><td>662 [629, 725]</td></tr><tr><td>walker-stand</td><td>980 [977, 984]</td><td>986 [984, 989]</td><td>982 [975, 988]</td><td>984 [984, 985]</td></tr><tr><td>walker-walk</td><td>970 [968, 973]</td><td>970 [968, 974]</td><td>970 [968, 972]</td><td>972 [964, 979]</td></tr><tr><td>Mean</td><td>-</td><td>-0.00 [-0.01, 0.01]</td><td>0.00 [-0.01, 0.01]</td><td>-0.01 [-0.02, -0.01]</td></tr><tr><td>Median</td><td>-</td><td>-0.00 [-0.00, 0.00]</td><td>0.00 [-0.00, 0.00]</td><td>-0.00 [-0.01, 0.00]</td></tr><tr><td>IQM</td><td>-</td><td>-0.00 [-0.00, 0.00]</td><td>0.00 [-0.00, 0.00]</td><td>-0.01 [-0.02, 0.00]</td></tr></table>

<table><tr><td>Task</td><td>MR.Q</td><td>No MR</td><td>1-step return</td><td>No unroll</td></tr><tr><td>acrobot-swingup</td><td>287 [253, 317]</td><td>362 [305, 421]</td><td>91 [76, 112]</td><td>126 [77, 159]</td></tr><tr><td>ball_in_cup-catch</td><td>977 [975, 980]</td><td>898 [746, 977]</td><td>980 [979, 982]</td><td>976 [973, 980]</td></tr><tr><td>cartpole-balance</td><td>999 [999, 999]</td><td>998 [998, 999]</td><td>998 [998, 1000]</td><td>998 [999, 999]</td></tr><tr><td>cartpole-balance_sparse</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td><td>1000 [1000, 1000]</td></tr><tr><td>cartpole-swingup</td><td>868 [861, 875]</td><td>871 [864, 878]</td><td>858 [835, 875]</td><td>872 [863, 879]</td></tr><tr><td>cartpole-swingup_sparse</td><td>797 [777, 818]</td><td>459 [139, 780]</td><td>712 [685, 759]</td><td>529 [0, 816]</td></tr><tr><td>cheetah-run</td><td>775 [752, 805]</td><td>782 [765, 805]</td><td>675 [674, 677]</td><td>753 [679, 845]</td></tr><tr><td>dog-run</td><td>60 [44, 80]</td><td>22 [21, 24]</td><td>30 [21, 43]</td><td>45 [36, 56]</td></tr><tr><td>dog-stand</td><td>216 [201, 233]</td><td>137 [127, 148]</td><td>160 [143, 191]</td><td>209 [195, 217]</td></tr><tr><td>dog-trot</td><td>65 [54, 79]</td><td>32 [29, 36]</td><td>29 [25, 32]</td><td>47 [46, 49]</td></tr><tr><td>dog-walk</td><td>77 [70, 83]</td><td>42 [36, 50]</td><td>55 [33, 67]</td><td>67 [63, 76]</td></tr><tr><td>finger-spin</td><td>965 [938, 982]</td><td>887 [757, 965]</td><td>984 [978, 989]</td><td>738 [588, 960]</td></tr><tr><td>finger-turn_easy</td><td>953 [925, 974]</td><td>694 [539, 805]</td><td>942 [874, 979]</td><td>869 [766, 962]</td></tr><tr><td>finger-turn_hard</td><td>932 [905, 957]</td><td>622 [436, 825]</td><td>908 [872, 974]</td><td>902 [780, 973]</td></tr><tr><td>fish-swim</td><td>79 [67, 93]</td><td>72 [60, 93]</td><td>64 [58, 72]</td><td>67 [55, 90]</td></tr><tr><td>hopper-hop</td><td>270 [229, 317]</td><td>192 [166, 216]</td><td>248 [231, 280]</td><td>242 [219, 270]</td></tr><tr><td>hopper-stand</td><td>852 [705, 932]</td><td>918 [897, 935]</td><td>877 [820, 915]</td><td>925 [907, 940]</td></tr><tr><td>humanoid-run</td><td>1 [1, 2]</td><td>1 [1, 2]</td><td>1 [1, 2]</td><td>1 [1, 1]</td></tr><tr><td>humanoid-stand</td><td>7 [7, 8]</td><td>7 [7, 8]</td><td>7 [6, 9]</td><td>7 [5, 9]</td></tr><tr><td>humanoid-walk</td><td>2 [2, 3]</td><td>2 [2, 3]</td><td>2 [2, 3]</td><td>2 [2, 3]</td></tr><tr><td>pendulum-swingup</td><td>829 [815, 842]</td><td>819 [787, 844]</td><td>828 [811, 839]</td><td>665 [382, 811]</td></tr><tr><td>quadruped-run</td><td>498 [474, 523]</td><td>478 [432, 515]</td><td>456 [424, 476]</td><td>398 [326, 465]</td></tr><tr><td>quadruped-walk</td><td>833 [796, 868]</td><td>701 [663, 731]</td><td>666 [627, 720]</td><td>730 [663, 769]</td></tr><tr><td>reacher-easy</td><td>979 [978, 982]</td><td>978 [976, 981]</td><td>972 [948, 985]</td><td>978 [977, 980]</td></tr><tr><td>reacher-hard</td><td>965 [945, 977]</td><td>545 [214, 858]</td><td>978 [972, 982]</td><td>893 [776, 967]</td></tr><tr><td>walker-run</td><td>615 [571, 655]</td><td>568 [538, 599]</td><td>672 [639, 730]</td><td>656 [619, 696]</td></tr><tr><td>walker-stand</td><td>980 [977, 984]</td><td>974 [962, 986]</td><td>986 [982, 991]</td><td>983 [981, 987]</td></tr><tr><td>walker-walk</td><td>970 [968, 973]</td><td>955 [948, 964]</td><td>971 [967, 978]</td><td>971 [971, 972]</td></tr><tr><td>Mean</td><td>-</td><td>-0.07 [-0.09, -0.03]</td><td>-0.03 [-0.03, -0.02]</td><td>-0.04 [-0.06, -0.01]</td></tr><tr><td>Median</td><td>-</td><td>-0.02 [-0.03, -0.01]</td><td>-0.01 [-0.01, -0.00]</td><td>-0.01 [-0.01, -0.00]</td></tr><tr><td>IQM</td><td>-</td><td>-0.03 [-0.03, -0.01]</td><td>-0.01 [-0.02, -0.01]</td><td>-0.02 [-0.02, -0.01]</td></tr></table>

D.4 ATARI 

<table><tr><td>Task</td><td>MR.Q</td><td>Linear value function</td><td>Dynamics target</td><td>No target encoder</td></tr><tr><td>Alien</td><td>2471 [1848, 3155]</td><td>596 [561, 631]</td><td>1176 [1138, 1215]</td><td>2040 [1585, 2495]</td></tr><tr><td>Amidar</td><td>443 [376, 499]</td><td>48 [38, 61]</td><td>214 [182, 247]</td><td>249 [232, 268]</td></tr><tr><td>Assault</td><td>1125 [1094, 1160]</td><td>366 [359, 374]</td><td>911 [906, 917]</td><td>1057 [880, 1234]</td></tr><tr><td>Asterix</td><td>2216 [2081, 2346]</td><td>810 [585, 1035]</td><td>1940 [1865, 2015]</td><td>2407 [1860, 2955]</td></tr><tr><td>Asteroids</td><td>602 [493, 689]</td><td>609 [595, 623]</td><td>776 [716, 837]</td><td>765 [591, 939]</td></tr><tr><td>Atlantis</td><td>445022 [282338, 630730]</td><td>17683 [13080, 20330]</td><td>529745 [145750, 913740]</td><td>76930 [76090, 77770]</td></tr><tr><td>BankHeist</td><td>542 [348, 749]</td><td>93 [76, 111]</td><td>646 [326, 966]</td><td>40 [38, 43]</td></tr><tr><td>BattleZone</td><td>16520 [10560, 22760]</td><td>11300 [11100, 11500]</td><td>8250 [2100, 14400]</td><td>4600 [2500, 6700]</td></tr><tr><td>BeamRider</td><td>2007 [1855, 2194]</td><td>584 [528, 642]</td><td>1201 [1015, 1387]</td><td>1468 [1298, 1639]</td></tr><tr><td>Berzerk</td><td>430 [383, 472]</td><td>315 [216, 415]</td><td>381 [359, 403]</td><td>359 [275, 444]</td></tr><tr><td>Bowling</td><td>50 [37, 65]</td><td>31 [31, 33]</td><td>81 [81, 82]</td><td>40 [33, 48]</td></tr><tr><td>Boxing</td><td>95 [94, 97]</td><td>39 [37, 42]</td><td>90 [88, 93]</td><td>92 [89, 96]</td></tr><tr><td>Breakout</td><td>25 [21, 32]</td><td>3 [2, 4]</td><td>9 [8, 11]</td><td>1 [1, 2]</td></tr><tr><td>Centipede</td><td>14954 [13541, 16508]</td><td>6709 [6207, 7213]</td><td>7853 [7680, 8026]</td><td>5167 [3661, 6674]</td></tr><tr><td>ChopperCommand</td><td>4348 [3756, 5002]</td><td>890 [780, 1000]</td><td>3055 [2870, 3240]</td><td>1385 [1060, 1710]</td></tr><tr><td>CrazyClimber</td><td>104766 [99290, 109629]</td><td>23016 [21610, 25010]</td><td>92455 [84000, 100910]</td><td>40240 [29150, 51330]</td></tr><tr><td>Defender</td><td>25962 [23406, 29182]</td><td>7825 [5640, 10010]</td><td>17592 [9290, 25895]</td><td>11627 [5745, 17510]</td></tr><tr><td>DemonAttack</td><td>4660 [4072, 5241]</td><td>1608 [1365, 1852]</td><td>281 [242, 322]</td><td>278 [204, 352]</td></tr><tr><td>DoubleDunk</td><td>-9 [-11, -9]</td><td>-18 [-24, -13]</td><td>-15 [-22, -10]</td><td>-20 [-21, -19]</td></tr><tr><td>Enduro</td><td>1480 [1378, 1592]</td><td>343 [319, 368]</td><td>622 [621, 623]</td><td>690 [667, 713]</td></tr><tr><td>FishingDerby</td><td>-34 [-43, -27]</td><td>-90 [-96, -86]</td><td>-64 [-66, -62]</td><td>-81 [-87, -75]</td></tr><tr><td>Freeway</td><td>31 [31, 32]</td><td>20 [19, 23]</td><td>32 [33, 33]</td><td>11 [0, 22]</td></tr><tr><td>Frostbite</td><td>4003 [2871, 5163]</td><td>198 [198, 198]</td><td>268 [267, 269]</td><td>824 [258, 1390]</td></tr><tr><td>Gopher</td><td>4936 [3923, 5730]</td><td>598 [524, 672]</td><td>853 [832, 874]</td><td>4371 [3794, 4948]</td></tr><tr><td>Gravitar</td><td>275 [232, 322]</td><td>190 [130, 250]</td><td>352 [250, 455]</td><td>60 [40, 80]</td></tr><tr><td>Hero</td><td>8391 [6845, 10060]</td><td>615 [112, 1118]</td><td>7560 [7560, 7560]</td><td>2200 [1377, 3024]</td></tr><tr><td>IceHockey</td><td>-2 [-3, -1]</td><td>-15 [-16, -15]</td><td>-6 [-10, -4]</td><td>-8 [-10, -6]</td></tr><tr><td>Jamesbond</td><td>551 [534, 573]</td><td>46 [40, 50]</td><td>412 [310, 515]</td><td>102 [80, 125]</td></tr><tr><td>Kangaroo</td><td>4833 [2716, 7064]</td><td>555 [520, 590]</td><td>6830 [600, 13060]</td><td>685 [590, 780]</td></tr><tr><td>Krull</td><td>8660 [8198, 9147]</td><td>6078 [5777, 6379]</td><td>7460 [6961, 7959]</td><td>9088 [8624, 9552]</td></tr><tr><td>KungFuMaster</td><td>26150 [21973, 30490]</td><td>10400 [9320, 11480]</td><td>17020 [7520, 26520]</td><td>12130 [11280, 12980]</td></tr><tr><td>MontezumaRevenge</td><td>12 [0, 34]</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>0 [0, 0]</td></tr><tr><td>MsPacman</td><td>4395 [3799, 5002]</td><td>826 [757, 896]</td><td>2950 [2490, 3410]</td><td>3171 [1873, 4469]</td></tr><tr><td>NameThisGame</td><td>7511 [7085, 7911]</td><td>2339 [2258, 2436]</td><td>6660 [6568, 6752]</td><td>7015 [6396, 7634]</td></tr><tr><td>Phoenix</td><td>4843 [4635, 5033]</td><td>570 [286, 854]</td><td>3996 [3843, 4150]</td><td>4260 [3944, 4577]</td></tr><tr><td>Pitfall</td><td>-8 [-19, -1]</td><td>-25 [-50, 0]</td><td>0 [0, 0]</td><td>-66 [-122, -12]</td></tr><tr><td>Pong</td><td>15 [13, 17]</td><td>-20 [-21, -20]</td><td>14 [14, 16]</td><td>-10 [-19, -2]</td></tr><tr><td>PrivateEye</td><td>100 [100, 100]</td><td>45 [0, 90]</td><td>100 [100, 100]</td><td>90 [80, 100]</td></tr><tr><td>Qbert</td><td>3600 [2554, 4366]</td><td>256 [228, 285]</td><td>493 [488, 500]</td><td>747 [615, 880]</td></tr><tr><td>Riverraid</td><td>7362 [7062, 7630]</td><td>1997 [1829, 2165]</td><td>6860 [6391, 7330]</td><td>6342 [5450, 7234]</td></tr><tr><td>RoadRunner</td><td>27152 [19731, 34480]</td><td>3245 [2410, 4080]</td><td>21835 [20960, 22710]</td><td>35120 [33050, 37190]</td></tr><tr><td>Robotank</td><td>10 [9, 13]</td><td>7 [3, 11]</td><td>7 [4, 10]</td><td>6 [3, 9]</td></tr><tr><td>Seaquest</td><td>2660 [2055, 3579]</td><td>305 [214, 400]</td><td>895 [834, 956]</td><td>1227 [378, 2076]</td></tr><tr><td>Skiing</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td></tr><tr><td>Solaris</td><td>1262 [863, 1686]</td><td>480 [0, 960]</td><td>710 [378, 1042]</td><td>1219 [1198, 1240]</td></tr><tr><td>SpaceInvaders</td><td>478 [429, 524]</td><td>242 [230, 255]</td><td>303 [263, 344]</td><td>296 [269, 323]</td></tr><tr><td>StarGunner</td><td>1146 [996, 1437]</td><td>1060 [880, 1240]</td><td>960 [920, 1000]</td><td>970 [960, 980]</td></tr><tr><td>Surround</td><td>-6 [-7, -5]</td><td>-9 [-9, -9]</td><td>-7 [-8, -7]</td><td>-7 [-9, -6]</td></tr><tr><td>Tennis</td><td>0 [-1, 0]</td><td>-24 [-24, -24]</td><td>-5 [-10, 0]</td><td>0 [0, 0]</td></tr><tr><td>TimePilot</td><td>3101 [2772, 3482]</td><td>2525 [1430, 3620]</td><td>2525 [2040, 3010]</td><td>1710 [1020, 2400]</td></tr><tr><td>Tutankham</td><td>130 [124, 139]</td><td>90 [86, 95]</td><td>164 [150, 179]</td><td>78 [2, 155]</td></tr><tr><td>UpNDown</td><td>26477 [11956, 43260]</td><td>3180 [2321, 4040]</td><td>3045 [2667, 3424]</td><td>4523 [4422, 4625]</td></tr><tr><td>Venture</td><td>0 [0, 0]</td><td>65 [0, 130]</td><td>0 [0, 0]</td><td>0 [0, 0]</td></tr><tr><td>VideoPinball</td><td>18826 [15048, 23233]</td><td>12994 [10570, 15419]</td><td>9170 [7880, 10460]</td><td>19734 [14479, 24989]</td></tr><tr><td>WizardOfWor</td><td>1918 [1706, 2154]</td><td>635 [480, 790]</td><td>1260 [670, 1850]</td><td>1440 [930, 1950]</td></tr><tr><td>YarsRevenge</td><td>27299 [23434, 30493]</td><td>6404 [6391, 6417]</td><td>23613 [16984, 30244]</td><td>21163 [21047, 21280]</td></tr><tr><td>Zaxxon</td><td>3820 [2577, 4854]</td><td>0 [0, 0]</td><td>690 [0, 1380]</td><td>0 [0, 0]</td></tr><tr><td>Mean</td><td>-</td><td>-1.35 [-1.41, -1.29]</td><td>-0.38 [-0.81, 0.05]</td><td>-0.86 [-0.89, -0.83]</td></tr><tr><td>Median</td><td>-</td><td>-0.42 [-0.55, -0.42]</td><td>-0.16 [-0.16, -0.11]</td><td>-0.18 [-0.19, -0.12]</td></tr><tr><td>IQM</td><td>-</td><td>-0.56 [-0.60, -0.55]</td><td>-0.20 [-0.22, -0.15]</td><td>-0.26 [-0.27, -0.25]</td></tr><tr><td>Task</td><td>MR.Q</td><td>Revert</td><td>Non-linear model</td><td>MSE reward loss</td></tr><tr><td>Alien</td><td>2471 [1848, 3155]</td><td>66 [32, 100]</td><td>2167 [1426, 3169]</td><td>734 [617, 856]</td></tr><tr><td>Amidar</td><td>443 [376, 499]</td><td>39 [19, 60]</td><td>466 [364, 570]</td><td>157 [140, 177]</td></tr><tr><td>Assault</td><td>1125 [1094, 1160]</td><td>366 [359, 374]</td><td>1033 [998, 1068]</td><td>923 [873, 987]</td></tr><tr><td>Asterix</td><td>2216 [2081, 2346]</td><td>492 [425, 560]</td><td>1987 [1560, 2414]</td><td>2503 [2050, 3040]</td></tr><tr><td>Asteroids</td><td>602 [493, 689]</td><td>249 [239, 259]</td><td>563 [424, 740]</td><td>765 [624, 952]</td></tr><tr><td>Atlantis</td><td>445022 [282338, 630730]</td><td>7310 [3140, 11480]</td><td>444370 [241216, 647524]</td><td>87410 [31610, 153510]</td></tr><tr><td>BankHeist</td><td>542 [348, 749]</td><td>14 [0, 28]</td><td>1006 [961, 1042]</td><td>245 [195, 309]</td></tr><tr><td>BattleZone</td><td>16520 [10560, 22760]</td><td>3550 [3300, 3800]</td><td>23820 [20300, 26200]</td><td>4566 [3900, 5200]</td></tr><tr><td>BeamRider</td><td>2007 [1855, 2194]</td><td>546 [510, 582]</td><td>1904 [1777, 2046]</td><td>1489 [1446, 1543]</td></tr><tr><td>Berzerk</td><td>430 [383, 472]</td><td>315 [275, 355]</td><td>527 [498, 579]</td><td>334 [295, 398]</td></tr><tr><td>Bowling</td><td>50 [37, 65]</td><td>41 [27, 55]</td><td>69 [58, 82]</td><td>30 [27, 35]</td></tr><tr><td>Boxing</td><td>95 [94, 97]</td><td>29 [21, 38]</td><td>95 [91, 98]</td><td>93 [89, 98]</td></tr><tr><td>Breakout</td><td>25 [21, 32]</td><td>2 [1, 3]</td><td>17 [17, 18]</td><td>11 [7, 18]</td></tr><tr><td>Centipede</td><td>14954 [13541, 16508]</td><td>2877 [2821, 2933]</td><td>10053 [6514, 13594]</td><td>11624 [8061, 16184]</td></tr><tr><td>ChopperCommand</td><td>4348 [3756, 5002]</td><td>530 [420, 640]</td><td>2918 [2006, 3830]</td><td>2806 [1610, 3570]</td></tr><tr><td>CrazyClimber</td><td>104766 [99290, 109629]</td><td>1700 [0, 3400]</td><td>103950 [98066, 110282]</td><td>107220 [104990, 109150]</td></tr><tr><td>Defender</td><td>25962 [23406, 29182]</td><td>1917 [1780, 2055]</td><td>24283 [22936, 25425]</td><td>15231 [7875, 21485]</td></tr><tr><td>DemonAttack</td><td>4660 [4072, 5241]</td><td>149 [147, 151]</td><td>2467 [1370, 3548]</td><td>311 [288, 331]</td></tr><tr><td>DoubleDunk</td><td>-9 [-11, -9]</td><td>-23 [-24, -23]</td><td>-10 [-14, -8]</td><td>-10 [-14, -9]</td></tr><tr><td>Enduro</td><td>1480 [1378, 1592]</td><td>3 [0, 6]</td><td>1117 [1059, 1173]</td><td>800 [694, 899]</td></tr><tr><td>FishingDerby</td><td>-34 [-43, -27]</td><td>-86 [-91, -82]</td><td>-33 [-36, -30]</td><td>-71 [-75, -66]</td></tr><tr><td>Freeway</td><td>31 [31, 32]</td><td>0 [0, 0]</td><td>31 [31, 32]</td><td>30 [30, 31]</td></tr><tr><td>Frostbite</td><td>4003 [2871, 5163]</td><td>170 [151, 190]</td><td>2693 [834, 4491]</td><td>3954 [266, 7285]</td></tr><tr><td>Gopher</td><td>4936 [3923, 5730]</td><td>385 [280, 490]</td><td>7216 [3645, 11049]</td><td>2484 [886, 3514]</td></tr><tr><td>Gravitar</td><td>275 [232, 322]</td><td>82 [80, 85]</td><td>309 [156, 419]</td><td>140 [50, 310]</td></tr><tr><td>Hero</td><td>8391 [6845, 10060]</td><td>0 [0, 0]</td><td>7635 [7577, 7693]</td><td>933 [0, 2799]</td></tr><tr><td>IceHockey</td><td>-2 [-3, -1]</td><td>-14 [-17, -11]</td><td>-1 [-2, -1]</td><td>-8 [-9, -7]</td></tr><tr><td>Jamesbond</td><td>551 [534, 573]</td><td>60 [45, 75]</td><td>495 [462, 538]</td><td>355 [280, 455]</td></tr><tr><td>Kangaroo</td><td>4833 [2716, 7064]</td><td>80 [0, 160]</td><td>5732 [3148, 7954]</td><td>2793 [540, 7060]</td></tr><tr><td>Krull</td><td>8660 [8198, 9147]</td><td>10 [0, 20]</td><td>8396 [8178, 8593]</td><td>8886 [7771, 9458]</td></tr><tr><td>KungFuMaster</td><td>26150 [21973, 30490]</td><td>1815 [130, 3500]</td><td>24644 [19158, 30310]</td><td>19536 [12710, 31840]</td></tr><tr><td>MontezumaRevenge</td><td>12 [0, 34]</td><td>0 [0, 0]</td><td>240 [80, 400]</td><td>0 [0, 0]</td></tr><tr><td>MsPacman</td><td>4395 [3799, 5002]</td><td>283 [182, 385]</td><td>3721 [3169, 4499]</td><td>1457 [1382, 1549]</td></tr><tr><td>NameThisGame</td><td>7511 [7085, 7911]</td><td>2675 [2609, 2742]</td><td>6162 [5941, 6450]</td><td>6091 [5656, 6475]</td></tr><tr><td>Phoenix</td><td>4843 [4635, 5033]</td><td>728 [517, 939]</td><td>4611 [4300, 4800]</td><td>3638 [3317, 3959]</td></tr><tr><td>Pitfall</td><td>-8 [-19, -1]</td><td>-1012 [-2000, -24]</td><td>-3 [-9, 0]</td><td>-22 [-68, 0]</td></tr><tr><td>Pong</td><td>15 [13, 17]</td><td>-20 [-21, -21]</td><td>16 [15, 19]</td><td>13 [8, 17]</td></tr><tr><td>PrivateEye</td><td>100 [100, 100]</td><td>50 [20, 80]</td><td>100 [100, 100]</td><td>33 [0, 100]</td></tr><tr><td>Qbert</td><td>3600 [2554, 4366]</td><td>137 [125, 150]</td><td>4295 [4006, 4586]</td><td>861 [785, 968]</td></tr><tr><td>Riverrader</td><td>7362 [7062, 7630]</td><td>1660 [1131, 2190]</td><td>6679 [5053, 7770]</td><td>3418 [482, 6556]</td></tr><tr><td>RoadRunner</td><td>27152 [19731, 34480]</td><td>0 [0, 0]</td><td>26678 [19418, 32016]</td><td>24583 [15870, 35950]</td></tr><tr><td>Robotank</td><td>10 [9, 13]</td><td>2 [3, 3]</td><td>10 [8, 13]</td><td>8 [2, 13]</td></tr><tr><td>Seaquest</td><td>2660 [2055, 3579]</td><td>134 [20, 248]</td><td>2344 [1541, 3450]</td><td>1676 [368, 2336]</td></tr><tr><td>Skiing</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td></tr><tr><td>Solaris</td><td>1262 [863, 1686]</td><td>878 [8, 1748]</td><td>1280 [498, 2062]</td><td>870 [370, 1736]</td></tr><tr><td>SpaceInvaders</td><td>478 [429, 524]</td><td>207 [198, 216]</td><td>536 [405, 667]</td><td>253 [235, 278]</td></tr><tr><td>StarGunner</td><td>1146 [996, 1437]</td><td>930 [710, 1150]</td><td>1014 [994, 1048]</td><td>976 [960, 1000]</td></tr><tr><td>Surround</td><td>-6 [-7, -5]</td><td>-9 [-9, -9]</td><td>-5 [-7, -5]</td><td>-8 [-9, -8]</td></tr><tr><td>Tennis</td><td>0 [-1, 0]</td><td>-12 [-24, -2]</td><td>0 [-1, 0]</td><td>-2 [-5, 0]</td></tr><tr><td>TimePilot</td><td>3101 [2772, 3482]</td><td>2195 [2040, 2350]</td><td>3822 [3282, 4418]</td><td>2376 [2070, 2880]</td></tr><tr><td>Tutankham</td><td>130 [124, 139]</td><td>14 [0, 30]</td><td>138 [127, 151]</td><td>37 [0, 112]</td></tr><tr><td>UpNDown</td><td>26477 [11956, 43260]</td><td>1692 [1526, 1859]</td><td>34574 [9812, 71080]</td><td>4568 [4174, 4972]</td></tr><tr><td>Venture</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>74 [0, 210]</td><td>0 [0, 0]</td></tr><tr><td>VideoPinball</td><td>18826 [15048, 23233]</td><td>6961 [6299, 7623]</td><td>14689 [10497, 17911]</td><td>11244 [9717, 12780]</td></tr><tr><td>WizardOfWor</td><td>1918 [1706, 2154]</td><td>575 [550, 600]</td><td>1852 [1650, 2062]</td><td>1190 [1000, 1430]</td></tr><tr><td>YarsRevenge</td><td>27299 [23434, 30493]</td><td>148 [0, 297]</td><td>29495 [25242, 32299]</td><td>14267 [10884, 16770]</td></tr><tr><td>Zaxxon</td><td>3820 [2577, 4854]</td><td>0 [0, 0]</td><td>3144 [1128, 5118]</td><td>0 [0, 0]</td></tr><tr><td>Mean</td><td>-</td><td>-1.69 [-1.70, -1.67]</td><td>-0.07 [-0.32, 0.18]</td><td>-0.79 [-0.86, -0.73]</td></tr><tr><td>Median</td><td>-</td><td>-0.63 [-0.64, -0.63]</td><td>-0.01 [-0.02, 0.00]</td><td>-0.24 [-0.24, -0.17]</td></tr><tr><td>IQM</td><td>-</td><td>-0.67 [-0.69, -0.65]</td><td>-0.02 [-0.05, 0.00]</td><td>-0.23 [-0.24, -0.20]</td></tr><tr><td>Task</td><td>MR.Q</td><td>No reward scaling</td><td>No min</td><td>No LAP</td></tr><tr><td>Alien</td><td>2471 [1848, 3155]</td><td>2074 [1402, 2821]</td><td>2375 [1921, 3052]</td><td>2265 [1784, 2768]</td></tr><tr><td>Amidar</td><td>443 [376, 499]</td><td>402 [345, 454]</td><td>567 [465, 686]</td><td>361 [291, 428]</td></tr><tr><td>Assault</td><td>1125 [1094, 1160]</td><td>1235 [1140, 1330]</td><td>1154 [967, 1254]</td><td>1095 [1027, 1170]</td></tr><tr><td>Asterix</td><td>2216 [2081, 2346]</td><td>2476 [2051, 2901]</td><td>2881 [2600, 3045]</td><td>2503 [2250, 2800]</td></tr><tr><td>Asteroids</td><td>602 [493, 689]</td><td>614 [520, 711]</td><td>769 [713, 878]</td><td>509 [447, 568]</td></tr><tr><td>Atlantis</td><td>445022 [282338, 630730]</td><td>608658 [262742, 946168]</td><td>386213 [77480, 941750]</td><td>329941 [196522, 506576]</td></tr><tr><td>BankHeist</td><td>542 [348, 749]</td><td>490 [176, 806]</td><td>304 [205, 436]</td><td>235 [136, 388]</td></tr><tr><td>BattleZone</td><td>16520 [10560, 22760]</td><td>13660 [8040, 18780]</td><td>21300 [20100, 22200]</td><td>13650 [9270, 17290]</td></tr><tr><td>BeamRider</td><td>2007 [1855, 2194]</td><td>1989 [1947, 2031]</td><td>1989 [1948, 2016]</td><td>1543 [1194, 1840]</td></tr><tr><td>Berzerk</td><td>430 [383, 472]</td><td>427 [361, 524]</td><td>601 [398, 708]</td><td>531 [464, 593]</td></tr><tr><td>Bowling</td><td>50 [37, 65]</td><td>78 [66, 88]</td><td>38 [28, 50]</td><td>68 [58, 78]</td></tr><tr><td>Boxing</td><td>95 [94, 97]</td><td>91 [87, 96]</td><td>95 [92, 97]</td><td>95 [94, 97]</td></tr><tr><td>Breakout</td><td>25 [21, 32]</td><td>22 [20, 26]</td><td>27 [24, 30]</td><td>21 [18, 25]</td></tr><tr><td>Centipede</td><td>14954 [13541, 16508]</td><td>15952 [12806, 19014]</td><td>11288 [8365, 16843]</td><td>12236 [9602, 14654]</td></tr><tr><td>ChopperCommand</td><td>4348 [3756, 5002]</td><td>2796 [2228, 3378]</td><td>3283 [2330, 4770]</td><td>3490 [2724, 4341]</td></tr><tr><td>CrazyClimber</td><td>104766 [99290, 109629]</td><td>105014 [94550, 112412]</td><td>110046 [107390, 113610]</td><td>88805 [77424, 100499]</td></tr><tr><td>Defender</td><td>25962 [23406, 29182]</td><td>30912 [27871, 33695]</td><td>32336 [27280, 38695]</td><td>32385 [29809, 34752]</td></tr><tr><td>DemonAttack</td><td>4660 [4072, 5241]</td><td>4893 [4282, 5345]</td><td>2468 [712, 3513]</td><td>4280 [2797, 6098]</td></tr><tr><td>DoubleDunk</td><td>-9 [-11, -9]</td><td>-11 [-14, -8]</td><td>-9 [-13, -7]</td><td>-11 [-14, -9]</td></tr><tr><td>Enduro</td><td>1480 [1378, 1592]</td><td>1450 [1311, 1593]</td><td>967 [0, 1461]</td><td>1117 [883, 1287]</td></tr><tr><td>FishingDerby</td><td>-34 [-43, -27]</td><td>-22 [-25, -20]</td><td>-17 [-28, -13]</td><td>-35 [-46, -27]</td></tr><tr><td>Freeway</td><td>31 [31, 32]</td><td>25 [13, 32]</td><td>32 [31, 34]</td><td>32 [32, 32]</td></tr><tr><td>Frostbite</td><td>4003 [2871, 5163]</td><td>3247 [1532, 4771]</td><td>2401 [267, 4457]</td><td>1595 [644, 2511]</td></tr><tr><td>Gopher</td><td>4936 [3923, 5730]</td><td>5802 [1467, 13322]</td><td>11774 [7552, 18634]</td><td>11483 [4836, 23229]</td></tr><tr><td>Gravitar</td><td>275 [232, 322]</td><td>256 [215, 305]</td><td>393 [185, 570]</td><td>329 [286, 382]</td></tr><tr><td>Hero</td><td>8391 [6845, 10060]</td><td>8775 [7575, 11125]</td><td>7594 [7525, 7670]</td><td>7519 [7379, 7640]</td></tr><tr><td>IceHockey</td><td>-2 [-3, -1]</td><td>-3 [-5, -2]</td><td>-2 [-3, -1]</td><td>-5 [-7, -4]</td></tr><tr><td>Jamesbond</td><td>551 [534, 573]</td><td>543 [460, 610]</td><td>546 [470, 650]</td><td>471 [406, 528]</td></tr><tr><td>Kangaroo</td><td>4833 [2716, 7064]</td><td>6148 [3412, 8600]</td><td>8033 [5540, 9390]</td><td>4616 [2392, 6998]</td></tr><tr><td>Krull</td><td>8660 [8198, 9147]</td><td>8878 [7898, 9491]</td><td>8430 [6464, 9987]</td><td>8535 [8189, 8924]</td></tr><tr><td>KungFuMaster</td><td>26150 [21973, 30490]</td><td>24292 [18954, 29568]</td><td>25553 [23960, 27040]</td><td>23422 [21084, 25671]</td></tr><tr><td>MontezumaRevenge</td><td>12 [0, 34]</td><td>0 [0, 0]</td><td>190 [100, 370]</td><td>20 [0, 50]</td></tr><tr><td>MsPacman</td><td>4395 [3799, 5002]</td><td>4086 [3847, 4413]</td><td>3860 [3394, 4470]</td><td>3602 [3051, 4116]</td></tr><tr><td>NameThisGame</td><td>7511 [7085, 7911]</td><td>8323 [7308, 9338]</td><td>7992 [7194, 8458]</td><td>7681 [6795, 8351]</td></tr><tr><td>Phoenix</td><td>4843 [4635, 5033]</td><td>4940 [4541, 5255]</td><td>4780 [4605, 5008]</td><td>4717 [4388, 5016]</td></tr><tr><td>Pitfall</td><td>-8 [-19, -1]</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>-1 [-3, 0]</td></tr><tr><td>Pong</td><td>15 [13, 17]</td><td>15 [14, 18]</td><td>18 [19, 19]</td><td>13 [10, 17]</td></tr><tr><td>PrivateEye</td><td>100 [100, 100]</td><td>40 [0, 80]</td><td>125 [100, 177]</td><td>50 [20, 80]</td></tr><tr><td>Qbert</td><td>3600 [2554, 4366]</td><td>2848 [1410, 4287]</td><td>6567 [4638, 7722]</td><td>3365 [2590, 4045]</td></tr><tr><td>Riverraid</td><td>7362 [7062, 7630]</td><td>6669 [5104, 7779]</td><td>7464 [7213, 7688]</td><td>6191 [5522, 6889]</td></tr><tr><td>RoadRunner</td><td>27152 [19731, 34480]</td><td>37306 [32906, 40582]</td><td>38703 [35530, 45050]</td><td>33950 [27807, 39559]</td></tr><tr><td>Robotank</td><td>10 [9, 13]</td><td>15 [13, 17]</td><td>15 [10, 19]</td><td>10 [7, 13]</td></tr><tr><td>Seaquest</td><td>2660 [2055, 3579]</td><td>1998 [1141, 2626]</td><td>2005 [1596, 2348]</td><td>1421 [986, 1782]</td></tr><tr><td>Skiing</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td></tr><tr><td>Solaris</td><td>1262 [863, 1686]</td><td>1175 [660, 1770]</td><td>772 [216, 1606]</td><td>886 [637, 1106]</td></tr><tr><td>SpaceInvaders</td><td>478 [429, 524]</td><td>458 [431, 493]</td><td>515 [488, 532]</td><td>494 [459, 530]</td></tr><tr><td>StarGunner</td><td>1146 [996, 1437]</td><td>1074 [964, 1266]</td><td>11993 [1240, 22010]</td><td>991 [976, 1005]</td></tr><tr><td>Surround</td><td>-6 [-7, -5]</td><td>-5 [-7, -5]</td><td>-5 [-6, -4]</td><td>-7 [-9, -5]</td></tr><tr><td>Tennis</td><td>0 [-1, 0]</td><td>0 [0, 0]</td><td>6 [-4, 20]</td><td>0 [0, 0]</td></tr><tr><td>TimePilot</td><td>3101 [2772, 3482]</td><td>3118 [2620, 3618]</td><td>4433 [4190, 4830]</td><td>3616 [2880, 4259]</td></tr><tr><td>Tutankham</td><td>130 [124, 139]</td><td>144 [87, 184]</td><td>156 [154, 160]</td><td>125 [99, 150]</td></tr><tr><td>UpNDown</td><td>26477 [11956, 43260]</td><td>13859 [6662, 21316]</td><td>37177 [17986, 54020]</td><td>17527 [8653, 28247]</td></tr><tr><td>Venture</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>0 [0, 0]</td></tr><tr><td>VideoPinball</td><td>18826 [15048, 23233]</td><td>18345 [15176, 21920]</td><td>22721 [18131, 28516]</td><td>20490 [14091, 27791]</td></tr><tr><td>WizardOfWor</td><td>1918 [1706, 2154]</td><td>1906 [1070, 2984]</td><td>1790 [1560, 2010]</td><td>1549 [1375, 1708]</td></tr><tr><td>YarsRevenge</td><td>27299 [23434, 30493]</td><td>27358 [21924, 31727]</td><td>26211 [21118, 33567]</td><td>24249 [20860, 27018]</td></tr><tr><td>Zaxxon</td><td>3820 [2577, 4854]</td><td>1336 [0, 3300]</td><td>6746 [4940, 7760]</td><td>1488 [369, 2733]</td></tr><tr><td>Mean</td><td>-</td><td>0.18 [-0.25, 0.56]</td><td>0.13 [-0.10, 0.58]</td><td>-0.13 [-0.38, 0.14]</td></tr><tr><td>Median</td><td>-</td><td>-0.00 [-0.01, 0.00]</td><td>0.01 [0.00, 0.04]</td><td>-0.03 [-0.07, -0.00]</td></tr><tr><td>IQM</td><td>-</td><td>-0.01 [-0.02, 0.03]</td><td>0.03 [0.03, 0.06]</td><td>-0.04 [-0.08, -0.01]</td></tr><tr><td>Task</td><td>MR.Q</td><td>No MR</td><td>1-step return</td><td>No unroll</td></tr><tr><td>Alien</td><td>2471 [1848, 3155]</td><td>2886 [2458, 3315]</td><td>1342 [1226, 1454]</td><td>3056 [2300, 3614]</td></tr><tr><td>Amidar</td><td>443 [376, 499]</td><td>312 [204, 421]</td><td>240 [213, 287]</td><td>385 [335, 471]</td></tr><tr><td>Assault</td><td>1125 [1094, 1160]</td><td>1105 [1075, 1142]</td><td>1045 [904, 1134]</td><td>1079 [1003, 1140]</td></tr><tr><td>Asterix</td><td>2216 [2081, 2346]</td><td>2298 [1920, 2731]</td><td>2708 [2110, 3710]</td><td>2388 [2190, 2740]</td></tr><tr><td>Asteroids</td><td>602 [493, 689]</td><td>485 [387, 584]</td><td>723 [689, 783]</td><td>581 [406, 766]</td></tr><tr><td>Atlantis</td><td>445022 [282338, 630730]</td><td>12834 [9038, 17186]</td><td>60986 [25200, 87460]</td><td>57426 [40850, 88730]</td></tr><tr><td>BankHeist</td><td>542 [348, 749]</td><td>404 [73, 726]</td><td>123 [77, 160]</td><td>783 [231, 1080]</td></tr><tr><td>BattleZone</td><td>16520 [10560, 22760]</td><td>22000 [12100, 29080]</td><td>4700 [3600, 5800]</td><td>23733 [19600, 26100]</td></tr><tr><td>BeamRider</td><td>2007 [1855, 2194]</td><td>1849 [1726, 2004]</td><td>1438 [1213, 1604]</td><td>1389 [1274, 1535]</td></tr><tr><td>Berzerk</td><td>430 [383, 472]</td><td>437 [333, 537]</td><td>361 [290, 400]</td><td>443 [399, 489]</td></tr><tr><td>Bowling</td><td>50 [37, 65]</td><td>84 [73, 95]</td><td>44 [30, 67]</td><td>72 [60, 88]</td></tr><tr><td>Boxing</td><td>95 [94, 97]</td><td>92 [89, 95]</td><td>93 [92, 95]</td><td>95 [94, 96]</td></tr><tr><td>Breakout</td><td>25 [21, 32]</td><td>15 [14, 18]</td><td>14 [7, 23]</td><td>27 [17, 44]</td></tr><tr><td>Centipede</td><td>14954 [13541, 16508]</td><td>10517 [9157, 11949]</td><td>8927 [7098, 10732]</td><td>11984 [11005, 13691]</td></tr><tr><td>ChopperCommand</td><td>4348 [3756, 5002]</td><td>3394 [3140, 3764]</td><td>2520 [1980, 2900]</td><td>2866 [1670, 3990]</td></tr><tr><td>CrazyClimber</td><td>104766 [99290, 109629]</td><td>83734 [71466, 93070]</td><td>66980 [65140, 70490]</td><td>104076 [82650, 114820]</td></tr><tr><td>Defender</td><td>25962 [23406, 29182]</td><td>14469 [9247, 18763]</td><td>9658 [3245, 17870]</td><td>34718 [29120, 44180]</td></tr><tr><td>DemonAttack</td><td>4660 [4072, 5241]</td><td>746 [527, 994]</td><td>1861 [1675, 1990]</td><td>2840 [2064, 3618]</td></tr><tr><td>DoubleDunk</td><td>-9 [-11, -9]</td><td>-14 [-18, -11]</td><td>-7 [-10, -6]</td><td>-11 [-14, -10]</td></tr><tr><td>Enduro</td><td>1480 [1378, 1592]</td><td>897 [784, 1075]</td><td>1064 [1055, 1069]</td><td>1075 [1018, 1131]</td></tr><tr><td>FishingDerby</td><td>-34 [-43, -27]</td><td>-21 [-30, -8]</td><td>-82 [-95, -64]</td><td>-54 [-63, -45]</td></tr><tr><td>Freeway</td><td>31 [31, 32]</td><td>26 [13, 33]</td><td>32 [32, 33]</td><td>32 [32, 33]</td></tr><tr><td>Frostbite</td><td>4003 [2871, 5163]</td><td>3098 [1645, 4358]</td><td>1231 [254, 3182]</td><td>4182 [3473, 5451]</td></tr><tr><td>Gopher</td><td>4936 [3923, 5730]</td><td>1833 [1552, 2114]</td><td>5597 [4242, 6512]</td><td>4746 [2378, 6822]</td></tr><tr><td>Gravitar</td><td>275 [232, 322]</td><td>89 [0, 267]</td><td>166 [70, 310]</td><td>270 [145, 415]</td></tr><tr><td>Hero</td><td>8391 [6845, 10060]</td><td>7584 [7548, 7644]</td><td>7594 [7544, 7676]</td><td>6879 [5384, 7694]</td></tr><tr><td>IceHockey</td><td>-2 [-3, -1]</td><td>-5 [-8, -4]</td><td>-6 [-8, -5]</td><td>-2 [-4, -1]</td></tr><tr><td>Jamesbond</td><td>551 [534, 573]</td><td>433 [406, 462]</td><td>518 [495, 545]</td><td>546 [460, 625]</td></tr><tr><td>Kangaroo</td><td>4833 [2716, 7064]</td><td>7508 [5608, 9500]</td><td>6940 [1520, 10460]</td><td>9166 [8520, 9500]</td></tr><tr><td>Krull</td><td>8660 [8198, 9147]</td><td>8403 [7433, 9087]</td><td>7386 [6611, 7881]</td><td>7785 [7176, 8271]</td></tr><tr><td>KungFuMaster</td><td>26150 [21973, 30490]</td><td>19066 [15450, 22028]</td><td>15300 [14760, 16350]</td><td>29020 [27150, 30700]</td></tr><tr><td>MontezumaRevenge</td><td>12 [0, 34]</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>0 [0, 0]</td></tr><tr><td>MsPacman</td><td>4395 [3799, 5002]</td><td>3297 [2679, 3908]</td><td>2282 [1956, 2592]</td><td>2839 [2612, 2954]</td></tr><tr><td>NameThisGame</td><td>7511 [7085, 7911]</td><td>3638 [3207, 3999]</td><td>5590 [5026, 5941]</td><td>6529 [6069, 6983]</td></tr><tr><td>Phoenix</td><td>4843 [4635, 5033]</td><td>4101 [3752, 4503]</td><td>3825 [3435, 4306]</td><td>4941 [4613, 5193]</td></tr><tr><td>Pitfall</td><td>-8 [-19, -1]</td><td>-27 [-67, -5]</td><td>0 [0, 0]</td><td>0 [0, 0]</td></tr><tr><td>Pong</td><td>15 [13, 17]</td><td>12 [8, 16]</td><td>14 [9, 20]</td><td>12 [8, 16]</td></tr><tr><td>PrivateEye</td><td>100 [100, 100]</td><td>3068 [24, 9080]</td><td>100 [100, 100]</td><td>100 [100, 100]</td></tr><tr><td>Qbert</td><td>3600 [2554, 4366]</td><td>4491 [3362, 5870]</td><td>2517 [2032, 3105]</td><td>4220 [3918, 4508]</td></tr><tr><td>Riverraid</td><td>7362 [7062, 7630]</td><td>7479 [6818, 8239]</td><td>6733 [5928, 7759]</td><td>5856 [3607, 7855]</td></tr><tr><td>RoadRunner</td><td>27152 [19731, 34480]</td><td>29182 [23250, 35114]</td><td>36145 [33590, 38700]</td><td>37636 [33610, 40160]</td></tr><tr><td>Robotank</td><td>10 [9, 13]</td><td>11 [6, 16]</td><td>6 [6, 7]</td><td>12 [11, 17]</td></tr><tr><td>Seaquest</td><td>2660 [2055, 3579]</td><td>1556 [1248, 1866]</td><td>2166 [2154, 2178]</td><td>2194 [2112, 2284]</td></tr><tr><td>Skiing</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td><td>-30000 [-30000, -30000]</td></tr><tr><td>Solaris</td><td>1262 [863, 1686]</td><td>552 [319, 881]</td><td>1107 [662, 1552]</td><td>1088 [638, 1386]</td></tr><tr><td>SpaceInvaders</td><td>478 [429, 524]</td><td>550 [476, 632]</td><td>389 [382, 396]</td><td>551 [502, 596]</td></tr><tr><td>StarGunner</td><td>1146 [996, 1437]</td><td>1272 [1020, 1730]</td><td>1290 [1000, 1580]</td><td>1423 [990, 2220]</td></tr><tr><td>Surround</td><td>-6 [-7, -5]</td><td>-8 [-9, -7]</td><td>-5 [-6, -5]</td><td>-5 [-10, 1]</td></tr><tr><td>Tennis</td><td>0 [-1, 0]</td><td>0 [-2, 0]</td><td>0 [-1, 0]</td><td>0 [-2, 0]</td></tr><tr><td>TimePilot</td><td>3101 [2772, 3482]</td><td>2440 [1540, 3178]</td><td>2535 [2030, 3040]</td><td>2236 [1170, 3810]</td></tr><tr><td>Tutankham</td><td>130 [124, 139]</td><td>112 [65, 148]</td><td>151 [120, 182]</td><td>148 [127, 181]</td></tr><tr><td>UpNDown</td><td>26477 [11956, 43260]</td><td>25451 [17297, 34036]</td><td>4477 [2993, 5962]</td><td>31342 [3170, 84771]</td></tr><tr><td>Venture</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>0 [0, 0]</td><td>0 [0, 0]</td></tr><tr><td>VideoPinball</td><td>18826 [15048, 23233]</td><td>8524 [3899, 12519]</td><td>13506 [7955, 19058]</td><td>27525 [22242, 35815]</td></tr><tr><td>WizardOfWor</td><td>1918 [1706, 2154]</td><td>2058 [1632, 2640]</td><td>1545 [1490, 1600]</td><td>1573 [1430, 1700]</td></tr><tr><td>YarsRevenge</td><td>27299 [23434, 30493]</td><td>30666 [28659, 33225]</td><td>18513 [16940, 20088]</td><td>19082 [11442, 24565]</td></tr><tr><td>Zaxxon</td><td>3820 [2577, 4854]</td><td>5758 [4712, 6962]</td><td>330 [0, 660]</td><td>0 [0, 0]</td></tr><tr><td>Mean</td><td>-</td><td>-0.78 [-0.88, -0.69]</td><td>-0.70 [-0.81, -0.59]</td><td>-0.33 [-0.41, -0.28]</td></tr><tr><td>Median</td><td>-</td><td>-0.06 [-0.10, -0.01]</td><td>-0.12 [-0.12, -0.07]</td><td>-0.00 [-0.02, 0.00]</td></tr><tr><td>IQM</td><td>-</td><td>-0.09 [-0.14, -0.05]</td><td>-0.15 [-0.16, -0.13]</td><td>-0.01 [-0.04, -0.00]</td></tr></table>