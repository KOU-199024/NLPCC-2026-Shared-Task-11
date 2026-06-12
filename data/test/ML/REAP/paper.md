# REAP THE EXPERTS: WHY PRUNING PREVAILS FORONE-SHOT MOE COMPRESSION

Mike Lasby1,2,†, Ivan Lazarevich1, Nish Sinnadurai1, Sean Lie1, Yani Ioannou2, Vithursan Thangarasa1

1Cerebras Systems Inc., 2Schulich School of Engineering, University of Calgary

![](images/a19090ec6789a920fa8ffdeb95ee680d2b3b5bc7e38f4330eefa90b019d5d30d.jpg)

![](images/132f2e51fbb90c071ce332467e43987ad1055dd091754a0d07fde938e25492f6.jpg)

https://github.com/CerebrasResearch/reap

https://hf.co/collections/cerebras/cerebras-reap

# ABSTRACT

Sparsely-activated Mixture-of-Experts (SMoE) models offer efficient pre-training and low latency but their large parameter counts create significant memory overhead, motivating research into expert compression. Contrary to recent findings favouring expert merging on discriminative benchmarks, we find that expert pruning is a superior strategy for generative tasks. We demonstrate that existing merging techniques introduce an irreducible error due to the loss of fine-grained routing control over experts. Leveraging this insight, we propose Router-weighted Expert Activation Pruning (REAP), a novel pruning criterion that considers both router gate-values and expert activation norms to minimize the reconstruction error bound. Across a diverse set of SMoE models ranging from 20B to 1T parameters, REAP consistently outperforms merging and other pruning methods on generative benchmarks, especially at 50% compression. Notably, our method achieves near-lossless compression on code generation tasks with Qwen3-Coder-480B and Kimi-K2, even after pruning 50% of experts.

# 1 INTRODUCTION

Interest in the Sparsely-activated Mixture-of-Experts (SMoE) architecture for Large Language Models (LLMs) surged following the release of DeepSeek-V3 (DeepSeek-AI et al., 2024) and other high-quality open-weight SMoE LLMs (Jiang et al., 2024; Meta AI Team, 2025; Yang et al., 2025a; Zeng et al., 2025; Baidu, 2025; Kimi Team et al., 2025). Compared to dense models, SMoEs offer lower latency and more efficient pre-training (Fedus et al., 2022). However, SMoEs require more parameters than dense models to achieve similar accuracy, resulting in significant memory overhead. Further, expert usage imbalance during inference causes poor accelerator utilization, leading to increased latency or compromises such as dropped tokens (Balmau et al., 2025). Expert usage imbalance also represents an opportunity, motivating prior work which investigates whether experts can be compressed without negatively impairing accuracy (Li et al., 2023; Lu et al., 2024). By eliminating or compressing redundant experts, memory overhead is reduced. A more uniform distribution of expert usage would also improve hardware utilization. Expert compression is particularly valuable for use cases which feature small batch sizes such as local deployments and academic research.

Initial expert compression efforts focused on expert pruning, the removal of experts in their entirety. However, expert pruning is a strong intervention on the model’s weights. Techniques such as quantization, low-rank compression, and expert merging also offer memory savings but maintain a lossy representation of the less important experts. Crucially, expert merging has recently been demonstrated to outperform expert pruning when evaluated with perplexity and on Multiple Choice (MC) question answering benchmarks (Li et al., 2023; Liu et al., 2024b). However, an evaluation comparing these methods on generative benchmarks has yet to be conducted. In this work, we demonstrate that — when paired with a suitable saliency criterion expert pruning outperforms expert merging, particularly on generative benchmark tasks such as code generation, creative writing, and mathematical reasoning. Specifically, our main contributions are as follows:

• We demonstrate that existing expert merging techniques introduce irreducible error due to the loss of the router’s independent, input-dependent modulation of the expert outputs. In high-granularity SMoEs, the loss of fine-grained routing control results in functional subspace collapse;   
• Empirically, we find that expert merging distorts the functional manifold topology due to the introduction of novel functionality. Conversely, as a coordinate subspace operation, pruning preserves the topology;

• We introduce Router-weighted Expert Activation Pruning (REAP), a novel expert pruning saliency criterion. By considering both router gate-values and expert activation norms, REAP explicitly minimizes the upper bound of the reconstruction error derived in our analysis by pruning experts which contribute minimally to the layer output;   
• Across diverse SMoE architectures ranging from 20B to 1T parameters and a suite of generative evaluations, we demonstrate the significant and consistent advantage of REAP over existing expert pruning and merging approaches, particularly at 50% compression. Notably, our method achieves near-lossless compression $( \Delta _ { a c c } \leq 2 \% )$ on code generation tasks after pruning 50% of experts from Qwen3-Coder-480B and Kimi-K2;   
• We open-source our code and select compressed model checkpoints to facilitate further research on compressed SMoEs and their applications.

# 2 RELATED WORK

Sparsely activated SMoE architecture. A Mixture-of-Experts (MoE) layer is comprised of multiple, specialized feed-forward subnetworks known as experts and a router which produces gate-values (i.e., gates) to dynamically modulate the output of the experts based on the input. The architecture was revived in the deep learning era by the introduction of the SMoE by Shazeer et al. (2017). SMoEs layers only select a subset of experts to use for each input, enabling massive scaling of model parameters without a commensurate increase in computational cost (Lepikhin et al., 2021; Fedus et al., 2022). In transformer-based LLMs, SMoE layers are integrated by replacing the traditional feed-forward layers. Further innovations such as auxiliary-loss-free load balancing (DeepSeek-AI et al., 2024), shared experts, and fine-grained experts (Dai et al., 2024) have propelled SMoE architectures to become the de facto standard for LLMs in recent months.

Expert pruning. Although SMoE layers effectively decouple total model parameters from inference costs, their memory overhead has motivated research in expert pruning to reduce total number of parameters. Early efforts demonstrated that progressively pruning experts based on router weights during fine-tuning until a single expert remained could preserve model quality in task-specific settings (Chen et al., 2022). Koishekenov et al. (2023) found expert pruning to be effective without further fine-tuning despite aggressively pruning up to 80% of experts. Muzio et al. (2024) found that global pruning using gate-values as a saliency criterion was more effective than uniform, layer-wise frequency-based pruning. Other sophisticated pruning criteria have been proposed: Lu et al. (2024) introduced an exhaustive search strategy which prunes experts that minimize the reconstruction loss between the original and pruned layer outputs; Liu et al. (2024a) used a gradient-free evolutionary algorithm to prune experts. Both of these works demonstrated significant improvements over naive frequency-based pruning. A comprehensive evaluation of 16 diverse pruning criteria was conducted by Jaiswal et al. (2025). Expert Activation Norm (EAN) was empirically found to be the highest performing criterion and the benefits of iterative pruning were presented. EASY-EP (Dong et al., 2026) proposed expert pruning based on the sum of gate-weighted expert output norms. In contrast, REAP ensures a frequencyagnostic assessment by computing the conditional mean strictly over the tokens where an expert is active.

Expert merging. While the above-noted works prove that expert compression is feasible via pruning, an alternative compression technique is to merge experts. Generally, merging requires both a clustering algorithm and a merging technique. Li et al. (2023) introduced Merge Sparse Mixture of Experts (M-SMoE) which first initializes expert cluster centres by identifying the dominant experts with the highest usage frequency globally across all layers. The remaining non-dominant experts are clustered based on the cosine similarity of router logits. Finally, expert weights are aligned via permutation with the weight matching algorithm (Ainsworth et al., 2023) and merged using frequency-weighted parameter averaging. Li et al. (2023) found that their technique outperformed Chen et al.’s (2022) pruning method on MC benchmarks. Chen et al. (2025) proposed Hierarchical Clustering for Sparsely activated Mixture of Experts (HC-SMoE). HC-SMoE clusters experts based on the euclidean similarity of their representative vectors — the average activation of each expert measured on every token in a calibration dataset — using hierarchical agglomerative clustering. Similar to M-SMoE, HC-SMoE uses frequency-weighted parameter averaging to merge clusters into a single merged expert. Without any fine-tuning, Chen et al. (2025) found that their technique outperformed expert pruning based on router logits (He et al., 2025a), frequency, and Lu et al.’s (2024) method when benchmarked on a suite of MC question answering tasks.

Other compression techniques. In addition to pruning and merging, experts may be compressed through quantization (Huang et al., 2025; Li et al., 2025; Duanmu et al., 2025), low-rank decomposition (Yang et al., 2024a; Gu et al., 2025; He et al., 2025b), weight sparsity (He et al., 2025a), or a combination of any of the above techniques (Liu et al., 2025a). These other approaches are orthogonal to expert pruning and merging; however, note that expert merging necessitates re-quantization for block quantization formats that share common scaling coefficients across a group of weights whereas pruning does not.

Model merging. Model merging aims to combine parameters from multiple trained neural networks and has been rapidly adopted as a cost-effective way to improve model quality across diverse domains. The initial motivation for merging was based on the finding that mode connectivity exists between the loss landscapes of two or more trained neural networks, enabling interpolation of their parameters without incurring an increase in loss (Garipov et al., 2018; Ainsworth et al., 2023; Ito et al., 2024). Simple parameter averaging remains an effective technique; however, more sophisticated strategies based on task vectors have also been proposed to minimize interference in the merged model parameters (Ilharco et al., 2023; Yadav et al., 2023; Yu et al., 2024). Much of the existing literature focuses on the setting in which multiple fine-tunes of a single checkpoint are merged. Non-local merging in which the models do not share a common checkpoint is more closely related to expert merging. Sharma et al. (2024) found that re-scaling of model activations was necessary to achieve high-quality non-local merging.

LLM evaluation. Evaluating LLMs is challenging; prior work demonstrated that simple metrics such as perplexity can be misleading when used to evaluate compressed LLMs (Jaiswal et al., 2024). MC benchmarks typically measure the log-likelihood of answer tokens to determine a model’s response to a question (Gao et al., 2023; Chandak et al., 2025). As such, each response choice is evaluated in a single forward pass, without any tokens being generated by the model. Perplexity and MC accuracy can therefore be viewed as discriminative metrics. In contrast, generative benchmarks require the model to output a response, more closely corresponding with real-world use-cases of LLMs. Tasks such as code generation, mathematical reasoning with structured outputs, and creative writing are examples of generative benchmarks.

# 3 MOTIVATION

Setup. To motivate our proposed expert pruning method, we derive the expected errors of both expert merging and pruning. Consider a SMoE layer with K experts $f _ { 1 } , . . . , f _ { K }$ , each a function $f _ { k } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ . Let $\mathcal T ( \bar { x } )$ denote the set of indices corresponding to the top-k router scores. The router produces a sparse gating vector $\mathbf { g } ( x ) \in \mathbb { R } _ { \geq 0 } ^ { K }$ where $g _ { k } ( x ) > { \dot { 0 } } { \mathrm { i f } } k \in { \mathsf { T } } ( x )$ and $g _ { k } ( x ) = 0$ otherwise. We assume the active gates are normalized such that $\begin{array} { r } { \sum _ { k \in \mathcal { T } ( x ) } g _ { k } ( x ) = 1 } \end{array}$ , an operation commonly included in SMoE architectures.

The output of the layer is

$$
h (x) := \sum_ {k \in \mathcal {T} (x)} g _ {k} (x) f _ {k} (x). \tag {1}
$$

Two operations at fixed compression. To analyse the fundamental difference between compression operations, we focus on the elementary case of reducing two experts, $( f _ { i } , f _ { j } )$ , to one by comparing the mean squared reconstruction error, $\mathcal { E } { = } | | h ( x ) { - } \hat { h } ( x ) | | _ { 2 } ^ { 2 }$ where $\hat { h } ( x )$ is output of the layer after compression. Pruning removes expert $j$ and re-normalizes the router outputs over the remaining $K - 1$ experts. Merging replaces $( f _ { i } , f _ { j } )$ with a new expert $\tilde { \boldsymbol { f } } .$ . Existing one-shot expert merging methods such as HC-SMoE and M-SMoE sum the gates of the original experts $g _ { i } ( x ) + g _ { j } ( x )$ . The pruned, $\bar { h } ( x )$ , and merged, $\tilde { h } ( x )$ , layer outputs are

$$
\bar {h} (x) := \sum_ {k \neq j} \frac {g _ {k} (x)}{1 - g _ {j} (x)} f _ {k} (x), \quad (2) \quad \tilde {h} (x) := \left(g _ {i} (x) + g _ {j} (x)\right) \tilde {f} (x) + \sum_ {k \neq i, j} g _ {k} (x) f _ {k} (x). \tag {3}
$$

# 3.1 MERGING INDUCES AN INPUT-DEPENDENT TARGET A SINGLE EXPERT CANNOT REALIZE

Define the router’s input-dependent mixing ratio $\begin{array} { r } { r ( x ) : = \frac { g _ { i } ( x ) } { g _ { i } ( x ) + g _ { j } ( x ) } \in [ 0 , 1 ] } \end{array}$ locally on the set where $g _ { i } + g _ { j } > 0$ . Substituting $g _ { i } ( x )$ and $g _ { j } ( x )$ in terms of $r ( x )$ , the original contribution of the pair $( i , j )$ can be written as

$$
\begin{array}{l} g _ {i} (x) f _ {i} (x) + g _ {j} (x) f _ {j} (x) = \left[ r (x) \left(g _ {i} (x) + g _ {j} (x)\right) \right] f _ {i} (x) + \left[ (1 - r (x)) \left(g _ {i} (x) + g _ {j} (x)\right) \right] f _ {j} (x) \\ = \left(g _ {i} (x) + g _ {j} (x)\right) \underbrace {\left(r (x) f _ {i} (x) + (1 - r (x)) f _ {j} (x)\right)} _ {\text { The   ideal, input - dependent   target   expert }}. \tag {4} \\ \end{array}
$$

After merging, the router must apply the summed gate, $g _ { i } ( x ) + g _ { j } ( x )$ , to a constant convex combination of the constituent experts which is independent of x. The core issue is that the merged model is forced to approximate the dynamic, input-dependent target expert with a static one. The following quantifies this unavoidable approximation error.

Irreducible error of merging. Let $\widetilde { \boldsymbol f } ( \boldsymbol x ) = \alpha \boldsymbol f _ { i } ( \boldsymbol x ) + ( 1 - \alpha ) \boldsymbol f _ { j } ( \boldsymbol x )$ with a constant $\alpha \in [ 0 , 1 ]$ and define $\Delta _ { i j } : = f _ { i } ( x ) - f _ { j } ( x )$ . This definition of $\tilde { \boldsymbol { f } }$ assumes that the experts are linear functions of x which is generally not the case; however, this simplified model approximates the behaviour of frequency-weighted parameter averaging used by expert merging techniques in practice. $\mathcal { E } _ { m e r g e }$ is minimized when α is chosen to be the expected mixing ratio, $\bar { \alpha } ^ { \star } { : = } \bar { \mathbb { E } } [ r ( \bar { x } ) ^ { \cdot }$ ]. Omitting the argument (x) for brevity, this minimal error is

$$
\left\| \left(g _ {i} + g _ {j}\right) \left(r f _ {i} + (1 - r) f _ {j}\right) - \left(g _ {i} + g _ {j}\right) \left(\alpha^ {\star} f _ {i} + (1 - \alpha^ {\star}) f _ {j}\right) \right\| ^ {2} = \mathbb {E} _ {x} \left[ \underbrace {(g _ {i} + g _ {j}) ^ {2}} _ {\text { router   scale }} \cdot \underbrace {(r - \alpha^ {\star}) ^ {2}} _ {\text { policy   variability }} \cdot \underbrace {\| \Delta_ {i j} \| ^ {2}} _ {\text { expert   gap }} \right]. \tag {5}
$$

In particular, if the router’s policy is not constant $( \mathrm { V a r } [ r ( x ) ] > 0 )$ and the experts are not functionally identical $( \| \Delta _ { i j } \| > 0 )$ , then every constant-α merge incurs positive error. Let $G _ { i j } : = \mathbb { E } _ { x } [ \| \Delta _ { i j } ( x ) \| _ { 2 } ^ { 2 } ]$ ]. Under a simplifying assumption that the router scale, policy variability, and $G _ { i j }$ are weakly correlated across inputs, the error term may be decomposed to:

$$
\mathbb {E} _ {x} \left[ \left(g _ {i} (x) + g _ {j} (x)\right) ^ {2} \left(r (x) - \alpha^ {\star}\right) ^ {2} \| \Delta_ {i j} (x) \| _ {2} ^ {2} \right] \approx \mathbb {E} _ {x} \left[ \left(g _ {i} (x) + g _ {j} (x)\right) ^ {2} \right] \cdot \operatorname{Var} [ r (x) ] \cdot G _ {i j} \tag {6}
$$

Consequences. This is a standard least-squares problem minimized when $\alpha = \mathbb { E } [ r ]$ , and the minimal value is Var[r]. Based on the assumptions noted above, we conclude that merging with summed gates is fundamentally flawed whenever: (i) the router has learned an input-dependent policy for mixing two experts $( \mathrm { V a r } [ r ] \dot { > } 0 )$ ; and (ii) the experts are themselves distinct $( \bar { | | } \bar { \Delta _ { i j } | | } > 0 )$ . Any fixed α cannot overcome the irreducible error bound established in Equation (6).

# 3.2 PRUNING PRESERVES INDEPENDENT CONTROL

Pruning removes one function but importantly does not tie the remaining gates. The router still modulates each surviving expert independently. In contrast, merging removes a degree of freedom in the policy by replacing individual experts with their mergers. For a direct comparison under no fine-tuning, we consider the error of pruning expert j where $j \in \mathcal T ( x )$ ). After pruning, the router promotes previously inactive expert i with the new gate-value of $g _ { i } ^ { \prime } ( x ) \neq 0 .$ , producing the error

$$
\mathcal {E} _ {\text {prune}} = \mathbb {E} _ {x | j \in \mathcal {T} (x)} \left[ \left\| \underbrace {g _ {j} (x) f _ {j} (x) - g _ {i} ^ {\prime} (x) f _ {i} (x)} _ {\text {substitution error}} - \underbrace {\frac {g _ {j} (x) - g _ {i} ^ {\prime} (x)}{1 - g _ {j} (x) + g _ {i} ^ {\prime} (x)} \sum_ {k \neq i , j} g _ {k} (x) f _ {k} (x)} _ {\text {renormalization error}} \right\| _ {2} ^ {2} \right] \tag {7}
$$

Substitution error is the dominant term in the above expression as the renormalization error coefficient simply scales the magnitude of the surviving expert outputs without changing their direction. In contrast, the substitution error includes the output of the promoted expert which may introduce significant error. With top-k routing $g _ { i } ^ { \prime } { \le } g _ { j }$ and the maximum substitution error occurs when $g _ { i } ^ { \prime } { \approx } g _ { j }$ with a magnitude upper bounded by

$$
\left. \left| \left| g _ {j} (x) f _ {j} (x) - g _ {i} ^ {\prime} (x) f _ {i} (x) \right| \right| \leq g _ {j} (x) \left(\left| \left| f _ {j} (x) \right| \right| + \left| \left| f _ {i} (x) \right| \right|\right). \right. \tag {8}
$$

Synthesis. While neither method is clearly superior for all distributions, our simplified analysis above isolates specific sources of error. Merging with summed gates couples the experts, incurring error whenever either expert is active, unless the experts are functionally identical $( \Delta _ { i j } \approx 0 )$ . The router loses the ability to independently modulate the merged experts in an input-dependent manner. Equation (6) establishes that summed gate merging incurs an irreducible error directly proportional to the router’s policy variability $( \mathrm { V a r } [ r ( x ) ] )$ .

In contrast, pruning only incurs errors when the pruned expert is in the top-k set, $j \in \mathcal { T } ( x )$ . Unlike Equation (5), Equation (8) does not penalize policy variability; the router still controls surviving experts independently. The substitution error from pruning (Eq. 7) is proportional to its gate-value $( g _ { j } )$ and is insensitive to policy variability. Highly-granular SMoEs with many experts per layer use highly variable routing policies (high $\mathrm { V a r } [ r ( \dot { x } ) ] )$ to combine many small contributions (small $g _ { j } ( x ) )$ ). In this setting, we expect merging with summed gates to be fundamentally disadvantaged.

Remarks. (i) The constant-mixture model $\tilde { \boldsymbol { f } }$ is mathematically related to the frequency weighted parameter averaging merge used in practice. (ii) Even $\operatorname { i f } { \tilde { f } }$ was dependent on x, the router after merging cannot independently modulate the two latent functions, so the original policy is invalidated. (iii) With top-k routers, the specific irreducible error from policy variability $( \mathrm { V a r } [ r ( x ) ] )$ is generated exclusively on the support where both experts are selected. Outside that support, this component vanishes, leaving only a static error term that depends on the functional expert gap. (iv) See Appendix A for an extension of the above analysis to hierarchical clustering.

# 3.3 EMPIRICAL EVIDENCE FOR LOSS OF INDEPENDENT CONTROL

Setup. We analyse the functional expert output manifolds across four diverse state-of-the-art SMoE architectures by recording mean expert activations from 32 samples of 2048 tokens from the C4 dataset (Raffel et al., 2020).

Functional subspace collapse. By projecting expert activations onto their first two principal components, we visualize how pruning and merging affect the learned representations. Figures 1, A5a and A5b demonstrate a striking progression of functional subspace collapse from early to late layers in high-granularity architectures such as Qwen3 and ERNIE-4.5. In early layers, the original experts form relatively compact manifolds with moderate spread. After pruning, the surviving experts maintain their positions on the original manifold, preserving its geometric structure with reduced density. In contrast, merging produces a visible contraction toward the manifold’s centre. The contrast becomes dramatic in late layers, where experts are more specialized.

The progression from early to late layers validates our theoretical prediction that the irreducible error is proportional to $\mathrm { V a r } [ r ( x ) ]$ . Early layers, which typically learn more generic features, exhibit lower policy variability and thus less dramatic collapse. Late layers, where experts have specialized for distinct computational roles, demonstrate high policy variability, resulting in the severe functional collapse observed when these specialized experts are merged into static averages.

Functional manifold distortion. While collapse is less apparent in low-granularity models, the introduction of novel functions due to merging distorts the topology of the original expert manifold to a greater degree than pruning. To quantitatively measure this phenomenon, we measure the 1-Wasserstein distance (Earth Mover’s distance) between the original and compressed expert output manifolds, see Appendix B.2 for details. As depicted in Figure 2, the merged outputs consistently exhibit a higher transport cost from the original manifold.

Manifold geometry preservation. Across all models and layers, we observe that pruning preserves the topology of the functional manifold while merging fundamentally alters it. The preservation of manifold geometry under pruning reflects the mathematical structure of the operation: the pruned expert class is a coordinate subspace of the original, with the router maintaining independent control over each surviving expert.

In contrast, the subspace collapse observed in merged highly-granular SMoEs visualizes the loss of independent control. When gates $g _ { i }$ and $g _ { j }$ are tied by their sum $( g _ { i } + g _ { j } )$ , the router can no longer independently modulate the two underlying functions, forcing the model to approximate the dynamic mixture $r ( x ) f _ { i } ( x ) + ( 1 - r ( x ) ) f _ { j } ( x )$ with a static merged expert ${ \tilde { \boldsymbol { f } } } .$

With low-granularity SMoEs, such as Llama-4-Scout and Mixtral, functional subspace collapse due to expert merging is less apparent, see Figures A5c to A5f. With few experts per layer and active experts per token, these architectures have less variable routing and higher gate-values, which better preserves the variance of the original manifold. However, the introduction of novel functions by merging introduces greater manifold distortion than the substitution error associated with pruning. These observations reveal that the core issue is not the reduction in the number of experts per se, but rather the qualitative change in the router’s control structure and the introduction of novel functionality. See Appendix B for additional discussion.

# 4 ROUTER-WEIGHTED EXPERT ACTIVATION PRUNING (REAP)

The motivation in Section 3 demonstrates that the functional output space of a SMoE layer is defined by the coordinated behaviour of the router and experts. As established in Equation (8), the magnitude of the substitution error incurred by promoting expert i in lieu of pruned expert j is upper bounded by $g _ { j } ( x ) ( \| f _ { j } ( x ) \| + \| f _ { i } ( x ) \| )$ . Naive frequency-based pruning considers neither the coordination between router and expert $( g _ { j } ( x ) )$ nor the functional properties of the pruned expert $( \parallel f _ { j } ( x ) \parallel )$ , effectively assuming that all active experts contribute equally to the output. By ignoring these terms, frequency-based methods fail to minimize the error bound derived above.

![](images/b911f4d9948afb750c8e8e2a628721f2081f13b980f39fcacbfe6ce079b4d7c1.jpg)

<details>
<summary>scatter</summary>

| Group          | PC1       | PC2       |
| -------------- | --------- | --------- |
| Original Experts | -0.5      | -0.5      |
| Original Experts | 0.0       | 0.0       |
| Original Experts | 0.5       | -0.5      |
| Original Experts | 0.7       | -1.0      |
| Surviving      | -0.5      | -0.5      |
| Surviving      | 0.0       | 0.0       |
| Surviving      | 0.5       | 0.0       |
| Merged         | 0.0       | 0.0       |
| Merged         | 0.2       | 0.0       |
| Merged         | 0.3       | 0.0       |
| Merged         | 0.4       | 0.0       |
| Merged         | 0.5       | 0.0       |
</details>

(a) Qwen3-30B Layer 0

![](images/c1ee42fbbc766e385b2d4ddd83994a45a5c615542635bd24df68d6899a4c981e.jpg)

<details>
<summary>scatter</summary>

| Group          | PC1 Range | PC2 Range |
|----------------|-----------|-----------|
| Original Experts | -100 to 200 | -150 to 50 |
| Surviving      | -100 to 200 | -150 to 50 |
| Merged         | -100 to 200 | -150 to 50 |
</details>

(b) Qwen3-30B Layer 47

Figure 1: (a) Functional subspace (PCA) for early SMoE layers in Qwen3-30B. Pruning (blue) preserves the manifold geometry; merging (green) collapses it toward the centre. (b) Functional subspace (PCA) for late MoE layers. The contraction under merging is dramatically more pronounced, with up to 100× reduction in spread for models with many experts. See Figure A5 for results from other models.   
![](images/47f36a4fa951cb84999f2e61ba26a371bbe6208457aedb02a7e3873a0590e2d6.jpg)

<details>
<summary>line</summary>

| Layer | Pruned | Merged |
|-------|--------|--------|
| 0     | 0.2    | 0.25   |
| 10    | 0.3    | 0.45   |
| 20    | 0.35   | 0.47   |
| 30    | 0.32   | 0.46   |
| 40    | 0.3    | 0.45   |
| 45    | 0.2    | 0.4    |
</details>

(a) Qwen3-30B

![](images/952415c8b348bb48f9ca9a152be761033944ba261c90188aaba43bccdc3e372e.jpg)

<details>
<summary>line</summary>

| Layer | Pruned | Merged |
|-------|--------|--------|
| 0     | 0.25   | 0.25   |
| 5     | 0.27   | 0.41   |
| 10    | 0.28   | 0.45   |
| 15    | 0.29   | 0.47   |
| 20    | 0.28   | 0.46   |
| 25    | 0.29   | 0.40   |
</details>

(b) ERNIE-4.5-21B

![](images/8abb614e055352fcfb38cd8a31009858e90b5de53c2a58eb4caefabf75e78b35.jpg)

<details>
<summary>line</summary>

| Layer | Pruned | Merged |
|-------|--------|--------|
| 0     | 0.08   | 0.08   |
| 5     | 0.25   | 0.32   |
| 10    | 0.24   | 0.50   |
| 15    | 0.26   | 0.48   |
| 20    | 0.15   | 0.47   |
| 25    | 0.12   | 0.38   |
| 30    | 0.08   | 0.24   |
</details>

(c) Mixtral-8x7B

![](images/16ee66b078d0e9c9a582206fd262d44244650e3f5e7f3be63f9b8d6d00e5f746.jpg)

<details>
<summary>line</summary>

| Layer | Pruned | Merged |
|-------|--------|--------|
| 0     | 0.05   | 0.08   |
| 5     | 0.07   | 0.12   |
| 10    | 0.09   | 0.15   |
| 15    | 0.11   | 0.18   |
| 20    | 0.13   | 0.22   |
| 25    | 0.15   | 0.28   |
| 30    | 0.17   | 0.35   |
| 35    | 0.19   | 0.42   |
| 40    | 0.21   | 0.38   |
| 45    | 0.23   | 0.32   |
| 50    | 0.25   | 0.28   |
</details>

(d) Llama-4-Scout   
Figure 2: 1-Wasserstein distance between the compressed and original expert output manifolds measured in normalized angular distance. Expert merging introduces novel functions which distort the manifold.

Since the identity of the promoted expert i (and thus $\Vert f _ { i } ( x ) \Vert )$ varies across tokens, directly minimizing the pruned expert’s impact $g _ { j } ( x ) \lVert f _ { j } ( x ) \rVert$ is an effective heuristic to minimize the total error. This strategy targets the known components of the error bound $( g _ { j } \| f _ { j } \| )$ while simultaneously shrinking the scaling coefficient $( g _ { j } )$ of the unknown component $( \left. f _ { i } \right. )$ . Intuitively, this identifies experts which contribute minimally to the layer output, yielding the minimal difference between the original and pruned layer outputs in expectation. To select which experts to prune, we propose a novel saliency criterion, REAP. Specifically, the saliency score, $S _ { j } { \mathrm { : } }$ , is defined as the average of the expert’s weighted magnitude over tokens for which it is active:

$$
S _ {j} = \frac {1}{| \mathcal {X} _ {j} |} \sum_ {x \in \mathcal {X} _ {j}} g _ {j} (x) \cdot \left\| f _ {j} (x) \right\| _ {2}, \tag {9}
$$

where $\mathcal { X } _ { j }$ is the set of tokens where expert j is active $( { \mathrm { i . e . , ~ } } \chi _ { j } = \{ x | j \in T ( x ) \} )$ . Crucially, calculating this average conditionally over $\mathcal { X } _ { j }$ rather than globally decouples the expert’s functional impact from its frequency of activation. A global average may be dominated by usage frequency and risks pruning specialist experts which are rarely activated but contribute significantly to the layer output when selected. By pruning experts with the lowest $S _ { j }$ , REAP targets those that provide a weak functional contribution even when specifically requested by the router, thereby minimizing the substitution error bound for every active token.

# 5 EXPERIMENTS

Setup. We implement REAP and other expert compression baselines in PyTorch (Ansel et al., 2024). We collect router logits and expert activation data to calibrate the compression algorithms using a variety of general pre-training and domain-specific Supervised Fine-Tuning (SFT) datasets. For calibration, 1,024 samples are randomly selected and packed to 2,048 sequence length for models with ≤ 110B parameters. For models with≥ 110B parameters, we select 12,228 samples with a maximum sequence length of 16,384 tokens without truncation or packing.

Table 1: Comparison of SMoE models included in our study. 

<table><tr><td>Model</td><td>Routed Experts</td><td>Shared Experts</td><td>Top-K</td><td>Sparsity</td><td>Parameters (1e9)</td><td>Active Params. (1e9)</td><td>First layer dense</td><td>Expert Granularity</td></tr><tr><td>ERNIE-4.5-21B-A3B-PT</td><td>64</td><td>2</td><td>6</td><td>87.88%</td><td>21.9</td><td>3</td><td>Yes</td><td>High</td></tr><tr><td>Qwen3-30B-A3B</td><td>128</td><td>0</td><td>8</td><td>93.75%</td><td>30.5</td><td>3</td><td>No</td><td>High</td></tr><tr><td>Mixtral-8x7B-Instruct-v0.1</td><td>8</td><td>0</td><td>2</td><td>75.00%</td><td>46.7</td><td>13</td><td>No</td><td>Low</td></tr><tr><td>GLM-4.5-Air</td><td>128</td><td>1</td><td>8</td><td>93.02%</td><td>106.9</td><td>12</td><td>Yes</td><td>High</td></tr><tr><td>Llama-4-Scout-17B-16E-Instruct</td><td>16</td><td>1</td><td>1</td><td>88.24%</td><td>107.8</td><td>17</td><td>No</td><td>Low</td></tr><tr><td>Qwen3-Coder-480B-A35B-Instruct-FP8</td><td>160</td><td>0</td><td>8</td><td>95.00%</td><td>480.2</td><td>35</td><td>No</td><td>High</td></tr><tr><td>Kimi-K2-Instruct-W4A16 (RedHatAI, 2025)</td><td>384</td><td>1</td><td>8</td><td>97.66%</td><td>1026.4</td><td>32</td><td>Yes</td><td>High</td></tr></table>

We compress models by pruning or merging 25% or 50% of experts in each layer, except for M-SMoE which determines the number of clusters per layer based on global expert usage frequency. When evaluating models with ≤ 50B parameters on coding and MC, we calibrate and compress the models using three different seeds and report the mean. Larger models, creative writing, and mathematical reasoning evaluations are reported using a single seed, except where explicitly noted otherwise. All models are evaluated in the one-shot setting, with no additional fine-tuning after compression.

Models and data. We evaluate the expert compression algorithms on a diverse set of six SMoE architectures covering model sizes from 21B to 1T with varying degrees of sparsity and expert granularity, see Table 1 for details. For MC question answering and code generation benchmarks, we use C4 (Raffel et al., 2020; Allen Institute for AI, 2024) and evol-codealpaca (Chaudhary, 2023; Luo et al., 2024; Tam, 2023) datasets to assess both general and domain-specific calibration. Models with ≥ 110B parameters are additionally calibrated with data from xlam-function-calling (Liu et al., 2024c; Salesforce, 2025) and SWE-smith-trajectories (Yang et al., 2025c;b) datasets. For creative writing and math benchmarks we employ WritingPrompts curated (Pritsker, 2024) and tulu-3-sft-personas-math (Lambert et al., 2025; Allen Institute for AI, 2025), respectively. The default chat template is applied to all SFT datasets and </think> tags are explicitly closed to disable reasoning in hybrid reasoning models.

Evaluation. Compressed SMoE models are evaluated on a suite of benchmarks including MC question answering, code generation, mathematical reasoning, creative writing, and tool calling. See Appendix C for details. We implement HC-SMoE and M-SMoE as expert merging baselines. Average linkage criterion is used for HC-SMoE. M-SMoE does not include low-rank compression from the complete MC-SMoE method. Pruning baselines consist of frequency-based pruning and EAN and experts with the lowest saliency scores according to each method’s criterion are pruned. See Appendix D for formal definitions.

# 5.1 RESULTS

In Table 2 and Figure 3 code generation, creative writing, math reasoning, and MC results are presented for Qwen3-30B and GLM-4.5-Air after calibration with domain-specific datasets. Table 3 contains results for large-scale SMoE pruned models on code generation, tool calling, and MC benchmarks. See Table A6 and Table A7 for detailed MC and code generation results, respectively. Figure A6 depicts coding generation and MC accuracy versus model parameters. See Appendix F for additional results.

Zero-shot MC question answering. Both merging and pruning are capable of producing accurate compressed SMoE models for MC question answering. HC-SMoE and REAP have a mean decrease in accuracy of approximately 4% and 13% for compression ratios of 25% and 50%, respectively, excluding large-scale SMoEs. REAP achieves first or second rank among all methods, models and compression ratios, suggesting strong consistency regardless of specific model architecture. When calibrated on C4, we find slightly improved accuracies for all compression methods with similar rankings as noted above, see Table A8.

Generative benchmarks. Compared to MC, generative benchmarks are more representative of real-world use cases of LLMs. In this setting, pruning emerges as the clearly superior compression method on the generative task benchmarks. Excluding large-scale SMoEs, REAP achieves a mean decrease in accuracy of 1.9% and 6.9% at 25% and 50% compression ratios, respectively, on coding. In comparison, both HC-SMoE and

Table 2: MC and generative benchmark results for Qwen3-30B and GLM-4.5-Air. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Compression</td><td rowspan="2">Technique</td><td rowspan="2">Method</td><td colspan="3">Coding</td><td rowspan="2">Creative Writing WildBench</td><td colspan="3">Math</td><td rowspan="2">MC MC Avg</td></tr><tr><td>Eval+</td><td>LiveCode</td><td>Code Avg</td><td>GSM8K</td><td>MATH-500</td><td>Math Avg</td></tr><tr><td rowspan="11">Qwen3-30B-A3B</td><td>Baseline</td><td></td><td></td><td>0.814</td><td>0.302</td><td>0.558</td><td>0.811</td><td>0.903</td><td>0.872</td><td>0.887</td><td>0.721</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.781</td><td>0.293</td><td>0.537</td><td>0.805</td><td>0.901</td><td>0.872</td><td>0.886</td><td>0.558</td></tr><tr><td>HC-SMoE</td><td>0.752</td><td>0.258</td><td>0.505</td><td>0.497</td><td>0.864</td><td>0.834</td><td>0.849</td><td>0.674</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.805</td><td>0.302</td><td>0.553</td><td>0.807</td><td>0.910</td><td>0.865</td><td>0.888</td><td>0.600</td></tr><tr><td>EAN</td><td>0.797</td><td>0.311</td><td>0.554</td><td>0.811</td><td>0.904</td><td>0.879</td><td>0.892</td><td>0.603</td></tr><tr><td>REAP</td><td>0.797</td><td>0.304</td><td>0.551</td><td>0.804</td><td>0.896</td><td>0.881</td><td>0.888</td><td>0.665</td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.590</td><td>0.205</td><td>0.397</td><td>0.725</td><td>0.824</td><td>0.838</td><td>0.831</td><td>0.451</td></tr><tr><td>HC-SMoE</td><td>0.543</td><td>0.185</td><td>0.364</td><td>0.008</td><td>0.760</td><td>0.696</td><td>0.728</td><td>0.542</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.668</td><td>0.236</td><td>0.452</td><td>0.677</td><td>0.871</td><td>0.860</td><td>0.865</td><td>0.483</td></tr><tr><td>EAN</td><td>0.753</td><td>0.306</td><td>0.530</td><td>0.702</td><td>0.874</td><td>0.855</td><td>0.864</td><td>0.493</td></tr><tr><td>REAP</td><td>0.780</td><td>0.302</td><td>0.541</td><td>0.718</td><td>0.877</td><td>0.838</td><td>0.857</td><td>0.503</td></tr><tr><td rowspan="11">GLM-4.5-Air</td><td>Baseline</td><td></td><td></td><td>0.786</td><td>0.374</td><td>0.580</td><td>0.839</td><td>0.846</td><td>0.918</td><td>0.882</td><td>0.747</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.726</td><td>0.330</td><td>0.528</td><td>0.781</td><td>0.848</td><td>0.880</td><td>0.864</td><td>0.596</td></tr><tr><td>HC-SMoE</td><td>0.737</td><td>0.363</td><td>0.550</td><td>0.788</td><td>0.842</td><td>0.908</td><td>0.875</td><td>0.704</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.759</td><td>0.341</td><td>0.550</td><td>0.793</td><td>0.832</td><td>0.908</td><td>0.870</td><td>0.648</td></tr><tr><td>EAN</td><td>0.768</td><td>0.374</td><td>0.571</td><td>0.824</td><td>0.839</td><td>0.908</td><td>0.874</td><td>0.637</td></tr><tr><td>REAP</td><td>0.759</td><td>0.412</td><td>0.585</td><td>0.831</td><td>0.839</td><td>0.920</td><td>0.879</td><td>0.674</td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.468</td><td>0.099</td><td>0.284</td><td>0.391</td><td>0.465</td><td>0.466</td><td>0.465</td><td>0.444</td></tr><tr><td>HC-SMoE</td><td>0.618</td><td>0.220</td><td>0.419</td><td>0.593</td><td>0.667</td><td>0.732</td><td>0.700</td><td>0.564</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.511</td><td>0.104</td><td>0.308</td><td>0.604</td><td>0.615</td><td>0.612</td><td>0.613</td><td>0.521</td></tr><tr><td>EAN</td><td>0.721</td><td>0.253</td><td>0.487</td><td>0.702</td><td>0.781</td><td>0.838</td><td>0.809</td><td>0.511</td></tr><tr><td>REAP</td><td>0.679</td><td>0.352</td><td>0.515</td><td>0.754</td><td>0.815</td><td>0.900</td><td>0.857</td><td>0.554</td></tr></table>

Table 3: Large-scale pruned SMoEs on agentic, non-agentic coding, tool-use tasks, and MC benchmarks.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Compression</td><td rowspan="2">Method</td><td colspan="3">Non-Agentic Coding</td><td rowspan="2">Agentic CodingSWE-Bench-Verified</td><td colspan="4">Tool-Use (BFCLv3)</td><td rowspan="2">MCMC Avg</td></tr><tr><td>Eval+</td><td>LiveCode</td><td>Code Avg</td><td>Non-Live</td><td>Live</td><td>Multi-Turn</td><td>Overall</td></tr><tr><td rowspan="7">Qwen3-Coder-480B-A35B-Instruct-FP8</td><td>Baseline</td><td></td><td>0.841</td><td>0.431</td><td>0.636</td><td>0.540</td><td>0.866</td><td>0.825</td><td>0.380</td><td>0.690</td><td>0.750</td></tr><tr><td rowspan="3">25%</td><td>Frequency</td><td>0.737</td><td>0.296</td><td>0.516</td><td>0.378</td><td>0.844</td><td>0.763</td><td>0.355</td><td>0.654</td><td>0.606</td></tr><tr><td>EAN</td><td>0.827</td><td>0.419</td><td>0.623</td><td>0.534</td><td>0.831</td><td>0.813</td><td>0.384</td><td>0.676</td><td>0.702</td></tr><tr><td>REAP</td><td>0.831</td><td>0.416</td><td>0.624</td><td>0.540</td><td>0.878</td><td>0.823</td><td>0.392</td><td>0.698</td><td>0.748</td></tr><tr><td rowspan="3">50%</td><td>Frequency</td><td>0.007</td><td>0.012</td><td>0.010</td><td>0.000</td><td>0.200</td><td>0.392</td><td>0.000</td><td>0.197</td><td>0.506</td></tr><tr><td>EAN</td><td>0.777</td><td>0.382</td><td>0.580</td><td>0.536</td><td>0.822</td><td>0.774</td><td>0.383</td><td>0.659</td><td>0.591</td></tr><tr><td>REAP</td><td>0.822</td><td>0.415</td><td>0.619</td><td>0.522</td><td>0.849</td><td>0.801</td><td>0.371</td><td>0.674</td><td>0.692</td></tr><tr><td rowspan="7">Kimi-K2-Instruct-W4A16</td><td>Baseline</td><td></td><td>0.828</td><td>0.434</td><td>0.631</td><td>0.554</td><td>0.840</td><td>0.802</td><td>0.355</td><td>0.666</td><td>0.780</td></tr><tr><td rowspan="3">25%</td><td>Frequency</td><td>0.486</td><td>0.082</td><td>0.284</td><td>0.000</td><td>0.644</td><td>0.603</td><td>0.045</td><td>0.431</td><td>0.604</td></tr><tr><td>EAN</td><td>0.779</td><td>0.379</td><td>0.579</td><td>0.562</td><td>0.819</td><td>0.802</td><td>0.335</td><td>0.652</td><td>0.703</td></tr><tr><td>REAP</td><td>0.840</td><td>0.440</td><td>0.640</td><td>0.580</td><td>0.842</td><td>0.801</td><td>0.263</td><td>0.635</td><td>0.773</td></tr><tr><td rowspan="3">50%</td><td>Frequency</td><td>0.112</td><td>0.000</td><td>0.056</td><td>0.000</td><td>0.255</td><td>0.397</td><td>0.003</td><td>0.218</td><td>0.439</td></tr><tr><td>EAN</td><td>0.722</td><td>0.253</td><td>0.487</td><td>0.576</td><td>0.778</td><td>0.767</td><td>0.173</td><td>0.573</td><td>0.587</td></tr><tr><td>REAP</td><td>0.819</td><td>0.429</td><td>0.624</td><td>0.576</td><td>0.785</td><td>0.743</td><td>0.164</td><td>0.564</td><td>0.643</td></tr></table>

M-SMoE produce mean decreases in accuracy >5% at 25% compression and >20% at 50% compression. Notably, REAP maintains significantly higher accuracy at 50% compression than other pruning methods. M-SMoE achieves significantly better code generation accuracy on low-granularity SMoE architectures.

On creative writing, REAP and EAN are near-lossless at 25% compression with REAP offering improved quality at 50% compression. Merging methods are less consistent across various model architectures and compression ratios. For example, M-SMoE is the best method for Qwen3-30B at 50% compression, but the worst on GLM-4.5-Air. REAP attains the best mathematical reasoning results with a remarkable mean decrease in accuracy of just 0.1% at 25% compression. In comparison, HC-SMoE and M-SMoE offer high accuracy at 25% compression but are significantly less accurate than pruning at 50% compression.

Expert pruning at scale. To assess whether pruning remains viable at scale, we prune Qwen3-Coder-480B and Kimi-K2-Instruct. On MC questions, REAP outperforms other pruning methods. On non-agentic coding tasks, REAP achieves near-lossless accuracy with a 0.16% and 1.2% mean decrease in accuracy compared to baseline at 25% and 50%, respectively, outperforming EAN and frequency-based pruning, particularly at 50% compression. On the challenging SWE-Bench task, both REAP and EAN maintain high accuracy at 25% and 50% compression, with some scores slightly exceeding the baseline. On tool use, EAN and REAP are comparable, with REAP slightly outperforming at 50% compression with a mean decrease in accuracy of 5.9% versus 6.2% for EAN. Frequency-based pruning suffers from a sharp degradation in quality at 50% compression, highlighting the importance of pruning saliency criteria which consider expert activations. Scaling the pruning methods is relatively trivial. Unlike HC-SMoE, calibration for pruning does not require recording activations from every expert for every token, facilitating efficient calibration. Further, pruning can be easily applied to quantized models without any additional steps required to reconcile block scales or re-quantize following compression. See Appendix E for further discussion.

![](images/6a26b50efbd09e2419b0774be5ad8afa9a237656852bb520287b04ef9e2580de.jpg)  
Figure 3: GLM-4.5-Air and Qwen3-30B accuracy vs. task type. REAP offers significant improvements compared to other methods at 50% compression. Note the significant performance drop for merging methods on generative tasks (Coding, Math, Creative Writing) compared to their relative strength on MC.

![](images/beee096cf202f95dec7e0cbbbb07cfe0ad757098a12d74d27e1a1b7449d2c040.jpg)

<details>
<summary>boxplot</summary>

| N-gram size | Baseline | REAP | M-SMoE | HC-SMoE |
| ----------- | -------- | ---- | ------ | ------- |
| 2           | 0.85     | 0.80 | 0.75   | 0.70    |
| 3           | 0.90     | 0.85 | 0.80   | 0.75    |
| 4           | 0.95     | 0.90 | 0.85   | 0.80    |
</details>

(a) N-Gram diversity

![](images/4a337bca9364e5d03373462441c46c819c2bc4c765ab1ef59cec5282f68f034b.jpg)

<details>
<summary>violin</summary>

| Generator model | Cross perplexity |
| --------------- | ---------------- |
| REAP            | 10^0 to 10^1     |
| M-SMoE          | 10^0 to 10^1     |
| HC-SMoE         | 10^0 to 10^1     |
</details>

(b) Cross perplexity

![](images/268617c0b6d3879050c21b14de60796398cdca42b211b0f260fe0175599caa7f.jpg)

<details>
<summary>line</summary>

| Token Position | REAP  | M-SMoE | HC-SMoE |
| -------------- | ----- | ------ | ------- |
| 0              | 0.0   | 0.0    | 0.0     |
| 5              | 0.3   | 0.45   | 0.4     |
| 10             | 0.4   | 0.55   | 0.5     |
| 15             | 0.45  | 0.6    | 0.55    |
| 20             | 0.5   | 0.65   | 0.6     |
| 25             | 0.55  | 0.7    | 0.65    |
</details>

(c) Completion logit JSD

![](images/ef431c2c318616de22bfc888d84acc601ddf4042bc591ca0647467cc958cc5b4.jpg)

<details>
<summary>line</summary>

| Layer | Singular-vector alignment | L2 Distance | Base to IFT | HC-SMoE | M-SMoE | M-SMoE - permuted |
|-------|---------------------------|-------------|-------------|---------|--------|-------------------|
| 0     | 1.4                       | 1.4         | 0.0         | 1.4     | 1.4    | 1.4               |
| 8     | 1.4                       | 1.4         | 0.0         | 1.4     | 1.4    | 1.4               |
| 16    | 1.4                       | 1.4         | 0.0         | 1.4     | 1.4    | 1.4               |
| 24    | 1.4                       | 1.4         | 0.0         | 1.4     | 1.4    | 1.4               |
| 32    | 1.4                       | 1.4         | 0.0         | 1.4     | 1.4    | 1.4               |
| 40    | 1.4                       | 1.4         | 0.0         | 1.4     | 1.4    | 1.4               |
</details>

(d) Expert distance   
Figure 4: (a) & (b) N-Gram diversity and cross-perplexity of compressed Qwen3-30B-A3B models at 50% compression, respectively. (c) Jensen-Shannon Divergence (JSD) of compressed and baseline model logits vs. completion token position for Qwen3-30B-A3B at 50% compression. Initially, all compressed models share close alignment with the baseline model. However, as the completion token position increases the merged models diverge from the baseline more rapidly than the REAP pruned model. (d) The mean relative L2-distance and singular-vector alignment between Qwen3-30B expert weights at 50% compression. Expert merging is more challenging than model merging due to large distances between experts in weight space and low singular-vector alignment.

Quantifying merged SMoE generation quality. While merged expert SMoEs offer reasonable quality for discriminative tasks such as MC question and answering, they fail to remain competitive on generative tasks. To help explain the performance gap of merged models between discriminative and generative tasks, we perform an analysis of the compressed model outputs and compare with REAP pruned models. We prompt 50% compressed Qwen3-30B models with 100 questions randomly sampled from the evol-codealpaca dataset and record their outputs. In Figure 4a, we measure the N-gram diversity and find that the merged models have significantly lower diversity across all N-gram sizes measured. In contrast, the REAP pruned model remains similar to the base model, albeit slightly less diverse. In Figure 4b, we measure the perplexity of the text generated by the compressed models with the original baseline model. The text generated by the merged models has both a higher mean and higher variance than the pruned model generations, suggesting that the REAP pruned model outputs are more closely aligned to the original model. The alignment between the baseline and REAP pruned SMoEs is further supported by Figure 4c, which plots the JSD of the compressed and baseline logits vs. output token position. The merged model logits diverge from the baseline more rapidly than the pruned model.

The challenges of expert merging. Model merging has been widely adopted to facilitate LLM finetuning. Why does expert merging miss the mark? In addition to the loss of the router’s input-dependent modulation of experts explored in Section 3, we argue that the non-local nature of expert merging and high cardinality of expert clusters pose significant unresolved challenges. In Figure 4d, we plot the mean relative L2-distance between experts clustered by HC-SMoE or M-SMoE and compare with the distance between expert weights from the pretrained to Instruct Fine-Tuned (IFT) checkpoints. We find that the distance between clustered experts within the same layer greatly exceeds that of experts in the IFT checkpoint after fine-tuning. Ito et al. (2024) found that weight matching permutations improved alignment of parameters’ singular vectors. Following their approach, we decompose expert weights with Singular Value Decomposition (SVD) and plot the singular-vector alignment in Figure 4d. Even after applying weight matching permutations, the M-SMoE expert clusters remain far apart both in weight space and singular-vector alignment. The relatively poorly aligned experts highlight the considerable challenge of coherently merging their parameters.

When merging works well, it’s more closely related to pruning than one might expect. In Figure A7a, we depict the frequency of singleton clusters — clusters containing a single expert — for both HC-SMoE and M-SMoE. A singleton cluster is directly analogous to an expert that remains after pruning. We find that HC-SMoE in particular has a high prevalence of singleton clusters, leaving important experts unadulterated and compressing the rest into a few mega-clusters containing tens of experts. This is particularly true of the high granularity models which contain more experts per layer. We hypothesize that the cardinality of these mega-clusters poses a challenge for existing merging algorithms and test this intuition in Figure A7b. Unfortunately, even modest restrictions of the maximum cluster size to 32 — half the number of experts to compress — results in large decreases in model quality on coding tasks.

The importance of domain-specific calibration. In Figure A8, we plot the code generation accuracy of the various compression methods and models when calibrated on either C4 or evol-codealpaca. The difference is stark, C4 calibration results in a collapse in accuracy, with several compressed model instances failing to produce coherent outputs, resulting in 0% accuracy. In Figure A9, we compare the accuracy of compressed Qwen3-30B models calibrated with either domain-specific data or the combined calibration data across all generative tasks. While domain-specific calibration results in higher accuracies, REAP best preserves accuracy compared to other compression methods in the combined data calibration setting.

# 6 DISCUSSION

Similar to prior work, we find that expert merging performs reasonably well on MC benchmarks. This may be because MC tasks only require a discriminative function that can be approximated by an average expert. In contrast, merging fails to maintain model quality on generative tasks, particularly at 50% compression and high-granularity architectures. Generative tasks require auto-regressive generation, a capability that is impaired when the router’s fine-grained control is removed or novel expert functions are introduced. Compared to expert pruning, merging is less consistent, exhibiting higher variance across models and compression ratios. The outputs of expert merged models are more repetitive and less closely aligned with the base model compared with pruned model’s outputs. Taken together, these observations are direct evidence of functional manifold distortion of the SMoE layers discussed in Section 3.3.

Overall, expert pruned models offer consistently higher accuracy than merged models on generative tasks. REAP is a robust pruning criterion that generalizes across a wide array of SMoE architectures, compression ratios, and generative tasks. By taking into consideration both the router gate-values and expert activation norms, REAP minimizes the reconstruction error bound by pruning experts which contribute the least to each layers output. REAP is scalable, achieving near-lossless compression on coding tasks with Qwen3-Coder-480B and Kimi-K2. The successes of REAP highlight the crucial importance of preserving coordination between the router and experts. Compression methods which impair the router’s ability to independently modulate expert outputs or distort the original functional manifold are less likely to succeed.

Finally, this work highlights the importance of comprehensive downstream evaluations and the significant challenges involved with evaluating LLMs. Discriminative metrics such as perplexity and log-likelihood based MC benchmarks are not necessarily good proxies for generative model quality.

# 7 CONCLUSION

Our analysis of current SMoE expert merging techniques introduces irreducible error due to the loss of the router’s independent control over experts. In contrast, expert pruning produces a coordinate subspace of the original layer which maintains the topology of the functional manifold. We introduce REAP, a novel expert pruning method which prunes experts that contribute the least to the layer’s output, thereby minimizing the reconstruction error bound. Empirically, we demonstrate that REAP retains remarkably high accuracy on a wide array of generative tasks across a diverse set of model architectures. We hope that this work inspires further compression techniques for SMoEs and facilitates the deployment of accurate, domain-specific models in resource constrained settings.

# ACKNOWLEDGMENTS

We would like to acknowledge the helpful feedback of Valavan Manohararajah, Mohammed Adnan, and Rohan Jain. This work was supported in part by RBC Borealis through the RBC Borealis AI Global Fellowship Award. ML and YI gratefully acknowledge the support of Alberta Innovates (ALLRP 577350-22, ALLRP 600038-24), the Natural Sciences and Engineering Research Council of Canada (NSERC) (RGPIN-2022-03120, DGECR-2022-00358), Defence Research and Development Canada (DGDND-2022-03120), and NSERC/Agence Nationale de la Recherche (ANR) (ALLRP 602719-24). This project was supported thanks to funding from IVADO and the Canada First Research Excellence Fund. This research was enabled in part by support provided by the Digital Research Alliance of Canada (alliancecan.ca). YI is supported by a Schulich Research Chair.

# ETHICS STATEMENT

This research focused on the algorithmic compression of SMoE models and does not involve the use of human subjects, personally identifiable information, or sensitive data. The datasets used for calibration and evaluation (e.g., C4, evol-codealpaca) are publicly available. Our aim is to enable the use of large-scale SMoE models in resource constrained settings. However, we acknowledge that compression techniques such as REAP could potentially facilitate deployment of models for malicious purposes. Further, our compression methods are applied to pre-trained models and any biases related to fairness, discrimination, or representation inherent in the original models may be present in their compressed versions. We make no attempt in this work to mitigate these potential biases. The primary contribution of this paper is technical, and we do not foresee any new, direct ethical concerns arising from our proposed methodology beyond those already associated with the deployment of large language models.

# REPRODUCIBILITY STATEMENT

We are committed to ensuring the reproducibility of our research. We have open-sourced our code and released select compressed model checkpoints to facilitate further research on compressed SMoEs. REAP is formally described in Section 4. The baseline methods we compare against, including frequency-based pruning, EAN, M-SMoE, and HC-SMoE, are formally defined in Appendix D. Section 5 provides a detailed description of our experimental setup, including the specific models used, the calibration and evaluation datasets, and the implementation details for all compression experiments. Further evaluation details are provided in Appendix C.

# REFERENCES

Samuel Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. In Proceedings of the Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=CQsmMYmlP5T.   
Allen Institute for AI. allenai/c4 · Datasets at Hugging Face, August 2024. URL https: //huggingface.co/datasets/allenai/c4.   
Allen Institute for AI. allenai/tulu-3-sft-personas-math · Datasets at Hugging Face, 2025. URL https://huggingface.co/datasets/allenai/tulu-3-sft-personas-math.   
Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, CK Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Michael Suo, Phil Tillet, Eikan Wang, Xiaodong Wang, William Wen, Shunting Zhang, Xu Zhao, Keren Zhou, Richard Zou, Ajit Mathews, Gregory Chanan, Peng Wu, and Soumith Chintala. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’24). ACM, April 2024. doi: 10. 1145/3620665.3640366. URL https://docs.pytorch.org/assets/pytorch2-2.pdf.

Artificial Analysis. Artificial analysis intelligence benchmarking methodology. https: //artificialanalysis.ai/methodology/intelligence-benchmarking, September 2025. Version 3.0.   
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3639–3664, 2025.   
Baidu. Ernie 4.5 technical report, 2025. URL https://yiyan.baidu.com/blog/ publication/ERNIE\_Technical\_Report.pdf.   
Oana Balmau, Anne-Marie Kermarrec, Rafael Pires, André Loureiro Espírito Santo, Martijn de Vos, and Milos Vujasinovic. Accelerating moe model inference with expert sharding. In Proceedings of the 5th Workshop on Machine Learning and Systems, pp. 192–199, 2025.   
Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ2-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982, 2025.   
Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. The fifth pascal recognizing textual entailment challenge. TAC, 7(8):1, 2009.   
Nikhil Chandak, Shashwat Goel, Ameya Prabhu, Moritz Hardt, and Jonas Geiping. Answer Matching Outperforms Multiple Choice for Language Model Evaluation, July 2025. URL http://arxiv.org/abs/2507.02856. arXiv:2507.02856 [cs].   
Sahil Chaudhary. Code alpaca: An instruction-following llama model for code generation. https://github.com/sahil280114/codealpaca, 2023.   
I-Chun Chen, Hsu-Shen Liu, Wei-Fang Sun, Chen-Hao Chao, Yen-Chang Hsu, and Chun-Yi Lee. Retraining-free merging of sparse moe via hierarchical clustering. In Proceedings of the Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=hslOzRxzXL.   
Tianyu Chen, Shaohan Huang, Yuan Xie, Binxing Jiao, Daxin Jiang, Haoyi Zhou, Jianxin Li, and Furu Wei. Task-Specific Expert Pruning for Sparse Mixture-of-Experts, June 2022. URL http://arxiv.org/abs/2206.00277. arXiv:2206.00277 [cs].   
Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1300. URL https://aclanthology.org/N19-1300/.   
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL https://arxiv.org/abs/1803.05457. arXiv:1803.05457 [cs].   
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168. arXiv:2110.14168 [cs].   
Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models, January 2024. URL http://arxiv.org/abs/2401.06066. arXiv:2401.06066 [cs] version: 1.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. DeepSeek-V3 Technical Report, December 2024. URL http://arxiv.org/abs/2412.19437. arXiv:2412.19437 [cs] version: 1.   
Zican Dong, Han Peng, Peiyu Liu, Xin Zhao, Dong Wu, Feng Xiao, and Zhifeng Wang. Domainspecific pruning of large mixture-of-experts models with few-shot demonstrations. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=Vvb27TQzO9.   
Haojie Duanmu, Xiuhong Li, Zhihang Yuan, Size Zheng, Jiangfei Duan, Xingcheng Zhang, and Dahua Lin. MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design. In Proceedings of the Forty-second International Conference on Machine Learning, June 2025. URL https://openreview.net/forum?id=pXoZLGMNDm&noteId=vmMvwkSQDg.   
William Fedus, Barret Zoph, and Noam Shazeer. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, June 2022. URL http://arxiv.org/abs/2101.03961. arXiv:2101.03961 [cs].   
Leo Gao. Multiple Choice Normalization in LM Evaluation, October 2021. URL https://blog.eleuther.ai/multiple-choice-normalization/.   
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL https://github.com/EleutherAI/lm-evaluation-harness.   
Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson. Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information processing systems, 31, 2018.   
Hao Gu, Wei Li, Lujun Li, Zhu Qiyuan, Mark G. Lee, Shengjie Sun, Wei Xue, and Yike Guo. Delta decompression for moe-based LLMs compression. In Proceedings of the Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=ziezViPoN1.   
Shwai He, Daize Dong, Liang Ding, and Ang Li. Towards Efficient Mixture of Experts: A Holistic Study of Compression Techniques, March 2025a. URL http://arxiv.org/abs/2406.02500. arXiv:2406.02500 [cs] version: 3.

Yifei He, Yang Liu, Chen Liang, and Hany Hassan Awadalla. Efficiently Editing Mixture-of-Experts Models with Compressed Experts, March 2025b. URL http://arxiv.org/abs/2503.00634. arXiv:2503.00634 [cs].   
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In Proceedings of the Ninth International Conference on Learning Representations, 2021a. URL https://openreview.net/forum?id=d7KBjmI3GmQ.   
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021b. URL https://openreview.net/forum?id=7Bywt2mQsCe.   
Wei Huang, Yue Liao, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, and XIAOJUAN QI. Mixture compressor for mixture-of-experts LLMs gains more. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=hheFYjOsWO.   
HuggingFace. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github.com/huggingface/open-r1.   
Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=6t0Kwf8-jrj.   
Akira Ito, Masanori Yamada, and Atsutoshi Kumagai. Analysis of Linear Mode Connectivity via Permutation-Based Weight Matching: With Insights into Other Permutation Search Methods. In Proceedings of the Forty-Second International Conference on Machine Learning, October 2024. URL https://openreview.net/forum?id=lYRkGZZi9D.   
Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In Proceedings of the Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=chfJJYC3iL.   
Ajay Jaiswal, Jianyu Wang, Yixiao Li, Pingzhi Li, Tianlong Chen, Zhangyang Wang, Chong Wang, Ruoming Pang, and Xianzhi Du. Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations, April 2025. URL http://arxiv.org/abs/2504.05586. arXiv:2504.05586 [cs].   
Ajay Kumar Jaiswal, Zhe Gan, Xianzhi Du, Bowen Zhang, Zhangyang Wang, and Yinfei Yang. Compressing llms: The truth is rarely pure and never simple. In Proceedings of the Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=B9klVS7Ddk.   
Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of Experts, January 2024. URL http://arxiv.org/abs/2401.04088. arXiv:2401.04088 [cs].   
Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. Swe-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=VTF8yNQM66.   
Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru

Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, Haiming Wang, Jianzhou Wang, Jiaxing Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang, Yuzhi Wang, Zhaoji Wang, Zhengtao Wang, Zhexu Wang, Chu Wei, Qianqian Wei, Wenhao Wu, Xingzhe Wu, Yuxin Wu, Chenjun Xiao, Xiaotong Xie, Weimin Xiong, Boyu Xu, Jing Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ziyao Xu, Junjie Yan, Yuzi Yan, Xiaofei Yang, Ying Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye, Bohong Yin, Longhui Yu, Enming Yuan, Hongbang Yuan, Mengjie Yuan, Haobing Zhan, Dehao Zhang, Hao Zhang, Wanlu Zhang, Xiaobin Zhang, Yangkun Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Haotian Zhao, Yikai Zhao, Huabin Zheng, Shaojie Zheng, Jianren Zhou, Xinyu Zhou, Zaida Zhou, Zhen Zhu, Weiyu Zhuang, and Xinxing Zu. Kimi K2: Open Agentic Intelligence, July 2025. URL http://arxiv.org/abs/2507.20534. arXiv:2507.20534 [cs].   
Yeskendir Koishekenov, Alexandre Berard, and Vassilina Nikoulina. Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3567–3585, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.198. URL https://aclanthology.org/2023.acl-long.198/.   
Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam Stambler, Shyam Upadhyay, and Manaal Faruqui. Fact, fetch, and reason: A unified evaluation of retrieval-augmented generation, 2024. URL https://arxiv.org/abs/2409.12941.   
Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Christopher Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=i1uGbfHHpH.   
Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. In Proceedings of the Ninth International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=qrwe7XHTmYb.   
Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, , and Stanislas Polu. Numinamath tir. [https://huggingface. co/AI-MO/NuminaMath-TIR](https://github.com/project-numina/ aimo-progress-prize/blob/main/report/numina\_dataset.pdf), 2024.   
Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, and Tianlong Chen. Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy. In Proceedings of the Twelfth International Conference on Learning Representations, October 2023. URL https://openreview.net/forum?id=eFWG9Cy3WK.   
Pingzhi Li, Xiaolong Jin, Zhen Tan, Yu Cheng, and Tianlong Chen. QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts, February 2025. URL http://arxiv.org/abs/2406.08155. arXiv:2406.08155 [cs].   
Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild. In Proceedings of the Thirteenth International Conference on Learning Representations, October 2024a. URL https://openreview.net/forum?id=MKEHCx25xp.   
Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for llm compression and acceleration. In MLSys, 2024b.   
Enshu Liu, Junyi Zhu, Zinan Lin, Xuefei Ning, Matthew B. Blaschko, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang. Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs, July 2024a. URL http://arxiv.org/abs/2407.00945. arXiv:2407.00945 [cs].   
James Liu, Pragaash Ponnusamy, Tianle Cai, Han Guo, Yoon Kim, and Ben Athiwaratkun. Training-Free Activation Sparsity in Large Language Models, October 2024b. URL http://arxiv.org/abs/2408.14690. arXiv:2408.14690 [cs].   
Jiacheng Liu, Peng Tang, Wenfeng Wang, Yuhang Ren, Xiaofeng Hou, Pheng-Ann Heng, Minyi Guo, and Chao Li. A Survey on Inference Optimization Techniques for Mixture of Experts Models, January 2025a. URL http://arxiv.org/abs/2412.14219. arXiv:2412.14219 [cs] version: 2.   
Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36:21558–21572, 2023.   
Zechun Liu, Changsheng Zhao, Hanxian Huang, Sijia Chen, Jing Zhang, Jiawei Zhao, Scott Roy, Lisa Jin, Yunyang Xiong, Yangyang Shi, Lin Xiao, Yuandong Tian, Bilge Soran, Raghuraman Krishnamoorthi, Tijmen Blankevoort, and Vikas Chandra. Paretoq: Improving scaling laws in extremely low-bit LLM quantization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b. URL https://openreview.net/forum?id=PMSNd8xTHp.   
Zuxin Liu, Thai Quoc Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Shirley Kokane, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, Rithesh R N, Liangwei Yang, Silvio Savarese, Juan Carlos Niebles, Huan Wang, Shelby Heinecke, and Caiming Xiong. APIGen: Automated PIpeline for generating verifiable and diverse function-calling datasets. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024c. URL https://openreview.net/forum?id=Jfg3vw2bjx.   
Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6159–6172, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10. 18653/v1/2024.acl-long.334. URL https://aclanthology.org/2024.acl-long.334/.   
Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. In Proceedings of the Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=UnUwSIgK5W.   
Meta AI Team. The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation, 2025. URL https://ai.meta.com/blog/llama-4-multimodal-intelligence/.   
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.   
ModelScope Team. EvalScope: Evaluation framework for large models, 2024. URL https://github.com/modelscope/evalscope.   
Alexandre Muzio, Alex Sun, and Churan He. SEER-MoE: Sparse Expert Efficiency through Regularization for Mixture-of-Experts, April 2024. URL http://arxiv.org/abs/2404.05089. arXiv:2404.05089 [cs].

Neil Chowdhury, James Aung, Chan Jun Shern, Oliver Jaffe, Dane Sherburn, Giulio Starace, Evan Mays, Rachel Dias, Marwan Aljubeh, Mia Glaese, Carlos E. Jimenez, John Yang, Leyton Ho, Tejal Patwardhan, Kevin Liu, and Aleksander Madry. Introducing SWE-bench Verified, August 2024. URL https://openai.com/index/introducing-swe-bench-verified/.   
Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive APIs. In Proceedings of the Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=tBRNC6YemY.   
Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=2GmDdhBdDk.   
Jade Pritsker. euclaise/WritingPrompts\_curated · Datasets at Hugging Face, December 2024. URL https://huggingface.co/datasets/euclaise/WritingPrompts\_curated.   
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.   
Red Hat AI and vLLM Project. LLM Compressor, 8 2024. URL https://github.com/ vllm-project/llm-compressor.   
RedHatAI. RedHatAI/Kimi-K2-Instruct-quantized.w4a16 · Hugging Face, September 2025. URL https://huggingface.co/RedHatAI/Kimi-K2-Instruct-quantized.w4a16.   
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021. URL https://dl.acm.org/doi/abs/10.1145/3474381.   
Salesforce. Salesforce/xlam-function-calling-60k · Datasets at Hugging Face, May 2025. URL https: //huggingface.co/datasets/Salesforce/xlam-function-calling-60k.   
Ekansh Sharma, Daniel M. Roy, and Gintare Karolina Dziugaite. The Non-Local Model Merging Problem: Permutation Symmetries and Variance Collapse, October 2024. URL http://arxiv.org/abs/2410.12766. arXiv:2410.12766 [cs].   
Noam Shazeer. GLU Variants Improve Transformer, February 2020. URL http: //arxiv.org/abs/2002.05202. arXiv:2002.05202 [cs].   
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, January 2017. URL http://arxiv.org/abs/1701.06538. arXiv:1701.06538 [cs, stat].   
Shivalika Singh, Freddie Vargus, Daniel Dsouza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Souza Moura, Dominik Krzeminski, Hakimeh ´ Fadaei, Irem Ergün, Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Vu Minh Chien, Sebastian Ruder, Surya Guthikonda, Emad A. Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, Ahmet Üstün, Marzieh Fadaee, and Sara Hooker. Aya dataset: An open-access collection for multilingual instruction tuning, 2024.   
Zhi Rui Tam. theblackcat102/evol-codealpaca-v1 · Datasets at Hugging Face, July 2023. URL https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1.   
Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. TIES-merging: Resolving interference when merging models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=xtaX3WyCj1.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 Technical Report, May 2025a. URL http://arxiv.org/abs/2505.09388. arXiv:2505.09388 [cs].   
Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Yuanlin Duan, Wenqi Jia, Miao Yin, Yu Cheng, and Bo Yuan. MoE-i2: Compressing mixture of experts models through inter-expert pruning and intra-expert low-rank decomposition. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 10456–10466, Miami, Florida, USA, November 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-emnlp.612. URL https://aclanthology.org/2024.findings-emnlp.612/.   
John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Proceedings of the Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b. URL https://openreview.net/forum?id=mXpq6ut8J3.   
John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. SWE-bench/SWE-smith-trajectories · Datasets at Hugging Face, May 2025b. URL https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories.   
John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. SWE-smith: Scaling Data for Software Engineering Agents, May 2025c. URL http://arxiv.org/abs/2504.21798. arXiv:2504.21798 [cs].   
Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In Forty-first International Conference on Machine Learning, 2024.   
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David Traum, and Lluís Màrquez (eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1472. URL https://aclanthology.org/P19-1472/.   
Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, Yean Cheng, Yifan An, Yilin Niu, Yuanhao Wen, Yushi Bai, Zhengxiao Du, Zihan Wang, Zilin Zhu, Bohan Zhang, Bosi Wen, Bowen Wu, Bowen Xu, Can Huang, Casey Zhao, Changpeng Cai, Chao Yu, Chen Li, Chendi Ge, Chenghua Huang, Chenhui Zhang, Chenxi Xu, Chenzheng Zhu, Chuang Li, Congfeng Yin, Daoyan Lin, Dayong Yang, Dazhi Jiang, Ding Ai, Erle Zhu, Fei Wang, Gengzheng Pan, Guo Wang, Hailong Sun, Haitao Li, Haiyang Li, Haiyi Hu, Hanyu Zhang, Hao Peng, Hao Tai, Haoke Zhang, Haoran Wang, Haoyu Yang, He Liu, He Zhao, Hongwei Liu, Hongxi Yan, Huan Liu, Huilong Chen, Ji Li, Jiajing Zhao, Jiamin Ren, Jian Jiao, Jiani Zhao, Jianyang Yan, Jiaqi Wang, Jiayi Gui, Jiayue Zhao, Jie Liu, Jijie Li, Jing Li, Jing Lu, Jingsen Wang, Jingwei Yuan, Jingxuan Li, Jingzhao Du, Jinhua Du, Jinxin Liu, Junkai Zhi, Junli Gao, Ke Wang, Lekang Yang, Liang Xu, Lin Fan, Lindong Wu, Lintao Ding, Lu Wang, Man Zhang, Minghao Li, Minghuan Xu, Mingming Zhao, Mingshu Zhai, Pengfan Du, Qian Dong, Shangde Lei, Shangqing Tu, Shangtong Yang, Shaoyou Lu, Shijie Li, Shuang Li, Shuang-Li, Shuxun Yang, Sibo Yi, Tianshu Yu, Wei Tian, Weihan Wang, Wenbo Yu, Weng Lam Tam, Wenjie Liang, Wentao Liu, Xiao Wang, Xiaohan Jia, Xiaotao Gu, Xiaoying Ling, Xin Wang, Xing Fan, Xingru Pan, Xinyuan Zhang, Xinze Zhang, Xiuqing Fu, Xunkai Zhang, Yabo Xu, Yandong Wu, Yida Lu, Yidong Wang, Yilin Zhou, Yiming Pan, Ying Zhang, Yingli Wang, Yingru Li, Yinpei Su, Yipeng Geng, Yitong Zhu, Yongkun Yang, Yuhang Li, Yuhao Wu, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yuxuan Zhang, Zezhen Liu, Zhen Yang, Zhengda Zhou, Zhongpei Qiao, Zhuoer Feng, Zhuorui Liu, Zichen Zhang, Zihan Wang, Zijun Yao, Zikang Wang, Ziqiang Liu, Ziwei Chai, Zixuan Li, Zuodong Zhao, Wenguang Chen, Jidong Zhai, Bin Xu, Minlie Huang, Hongning Wang,

Juanzi Li, Yuxiao Dong, and Jie Tang. GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models, August 2025. URL http://arxiv.org/abs/2508.06471. arXiv:2508.06471 [cs].   
Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=Bl8u7ZRlbM.

# A EXTENSION TO HIERARCHICAL CLUSTERING

While Equation (5) analyses pairwise merging, practical implementations often employ hierarchical clustering to form groups of experts. Consider a cluster $C { = } \{ f _ { i _ { 1 } } ^ { \mathrm { ' } } , { \ldots } , f _ { i _ { k } } \}$ of k experts merged into a single representative $\tilde { \ b { f } } _ { C }$ . The original contribution of this cluster can be decomposed as:

$$
\sum_ {j \in C} g _ {i _ {j}} (x) f _ {i _ {j}} (x) = \left(\sum_ {j \in C} g _ {i _ {j}} (x)\right) \cdot \underbrace {\sum_ {j \in C} w _ {j} (x) f _ {i _ {j}} (x)} _ {\text { Dynamic,   input - dependent   mixture }} \tag {10}
$$

where wj(x)= $\begin{array} { r } { w _ { j } ( x ) = \frac { g _ { i _ { j } } ( x ) } { \sum _ { l \in C } g _ { i _ { l } } ( x ) } } \end{array}$ are the within-cluster mixing ratios that sum to 1.

After hierarchical merging, the router must apply the summed gate $\textstyle \sum _ { j \in C } g _ { i _ { j } }$ to a single, static cluster representative $\tilde { \ b { f } } _ { C } ,$ , typically computed as a weighted average of the cluster members based on calibration data. This induces an irreducible error.

Hierarchical clustering error. For a cluster C merged into $\scriptstyle { \tilde { \boldsymbol { f } } } _ { C } = \sum _ { j \in C } \alpha _ { j } \ f _ { i _ { j } }$ with fixed weights $\alpha _ { j } \geq 0$ $\textstyle \sum _ { j } \alpha _ { j } = 1$ , the minimal $L ^ { 2 }$ error is:

$$
\min _ {\left\{\alpha_ {j} \right\}} \left\| \sum_ {j \in C} g _ {i _ {j}} f _ {i _ {j}} - \left(\sum_ {j \in C} g _ {i _ {j}}\right) \tilde {f} _ {C} \right\| ^ {2} = \mathbb {E} \left[ \left(\sum_ {j \in C} g _ {i _ {j}}\right) ^ {2} \right] \cdot \operatorname{Var} _ {x} \left[ \sum_ {j \in C} w _ {j} (x) f _ {i _ {j}} (x) \right] \tag {11}
$$

The error grows with both the cluster’s total gate-value and the variance of the dynamic mixture that the cluster must approximate with a static representative.

Implications for cluster formation. The hierarchical error bound reveals a fundamental tension:

• Large clusters (|C| large) aggregate more gate-value $\textstyle \sum _ { j \in C } g _ { i _ { j } }$ , amplifying any approximation error   
• Diverse clusters (high $\lVert \Delta _ { i j } \rVert$ for $i , j \in C )$ increase the variance term, as the static representative must approximate a wider range of functions   
• Imbalanced clustering (many singletons, few mega-clusters) combines the worst aspects: mega-clusters suffer severe collapse while singletons provide minimal compression

Distance metrics like Euclidean distance that consider magnitude can exacerbate these issues by creating clusters based on norm similarity rather than functional role, potentially grouping experts with different specializations but similar scales. The resulting mega-clusters force the router to apply a single control signal to what were previously dozens of independently modulated experts, explaining the catastrophic functional collapse observed empirically in late layers where $\mathrm { V a r } [ w _ { j } ( x ) ]$ is highest.

# B ADDITIONAL EMPIRICAL EVIDENCE FOR LOSS OF INDEPENDENT CONTROL

# B.1 FUNCTIONAL SUBSPACE PCA ANALYSIS

Qualitative evidence of functional subspace collapse. In Figure 1a, Qwen3’s layer 0 exemplifies the contraction of the functional output space by merging in early layers. The original 128 experts span from −0.4 to 1.0 along PC1, pruning maintains this full range with 64 experts, while merging contracts the distribution to approximately [−0.2,0.3], a 5-fold reduction. This contraction is dramatic in late layers, where experts are more specialized, as can be seen in Figure 1b. Figures A5a and A5b exhibit similar contractions of the expert output manifold under merging, whereas pruning often preserves outlier experts and the span of the original expert output manifold.

In Table A4, we tabulate the total cumulative variance explained by PC1+PC2 for the PCA projections in Figures 1 and A5. For low-granularity SMoEs such as Llama-4-Scout and Mixtral, PC1 and PC2 capture most of the variance in the activations. Even in high-granularity SMoEs such as Qwen3 and ERNIE, a large portion of the total variance is captured by PC1 and PC2. The merged variance explained is consistently higher than the baseline, suggesting that the merged outputs have lost some of their high-dimensional complexity. In contrast, the pruned variance explained is consistently lower than the baseline, suggesting that pruning preserves outlier experts and the high-dimensional complexity of the baseline model.

![](images/b0fc2aa0837e201c4873f4706e8d76f0ca69efcab091b4ca08d91388804d433f.jpg)

<details>
<summary>scatter</summary>

| Group          | PC1  | PC2  |
| -------------- | ---- | ---- |
| Original Experts | 0.5  | -1.8 |
| Original Experts | 0.3  | -0.5 |
| Original Experts | 0.7  | 0.2  |
| Original Experts | 0.6  | -0.3 |
| Original Experts | 0.4  | 0.1  |
| Original Experts | 0.8  | -0.7 |
| Original Experts | 0.2  | -1.5 |
| Original Experts | 0.9  | -0.9 |
| Original Experts | 0.1  | -0.2 |
| Original Experts | 1.0  | 0.4  |
| Original Experts | 0.7  | -0.6 |
| Original Experts | 1.1  | -0.4 |
| Original Experts | 0.5  | -0.8 |
| Original Experts | 0.6  | -1.2 |
| Original Experts | 0.3  | -1.6 |
| Original Experts | 0.8  | -0.7 |
| Original Experts | 0.4  | -0.1 |
| Original Experts | 1.2  | -0.5 |
| Original Experts | 0.2  | -1.3 |
| Original Experts | 0.9  | -0.8 |
| Original Experts | 0.7  | -0.3 |
| Original Experts | 1.3  | -0.6 |
| Original Experts | 0.6  | -1.1 |
| Original Experts | 1.4  | -0.9 |
| Original Experts | 0.3  | -0.4 |
| Original Experts | 0.8  | -0.7 |
| Original Experts | 0.5  | -1.4 |
| Original Experts | 1.0  | -0.6 |
| Original Experts | 0.4  | -0.2 |
| Original Experts | 1.1  | -0.5 |
| Original Experts | 0.7  | -1.2 |
| Original Experts | 1.2  | -0.8 |
| Original Experts | 0.5  | -0.3 |
| Original Experts | 0.9  | -0.6 |
| Original Experts | 1.3  | -1.3 |
| Original Experts | 0.2  | -1.7 |
| Original Experts | 0.8  | -0.9 |
| Original Experts | 1.4  | -1.1 |
| Original Experts | 0.6  | -1.5 |
| Original Experts | 1.5  | -0.7 |
| Original Experts | 0.3  | -0.1 |
| Original Experts | 0.9  | -0.4 |
| Original Experts | 1.6  | -1.4 |
| Original Experts | 0.7  | -1.8 |
| Original Experts | 1.7  | -1.0 |
| Original Experts | 0.4  | -1.6 |
| Original Experts | 1.3  | -0.8 |
| Original Experts | 1.8  | -1.2 |
| Original Experts | 0.5  | -2.1 |
| Survival       | -0.5 | -2.5 |
| Surviving      | -0.3 | -2.2 |
| Surviving      | -0.7 | -2.8 |
| Surviving      | -0.4 | -2.4 |
| Surviving      | -0.6 | -2.6 |
| Surviving      | -0.2 | -2.3 |
| Surviving      | -0.8 | -2.9 |
| Surviving      | -0.5 | -2.7 |
| Surviving      | -1.1 | -3.1 |
| Surviving      | -0.3 | -2.5 |
| Surviving      | -0.9 | -2.8 |
| Surviving      | -0.6 | -2.4 |
| Surviving      | -1.3 | -3.3 |
| Surviving      | -0.7 | -2.9 |
| Surviving      | -1.4 | -3.5 |
| Merged         | -0.2 | -2.8 |
| Merged         | -0.5 | -2.4 |
| Merged         | -0.3 | -2.7 |
| Merged         | -0.8 | -3.2 |
| Merged         | -0.6 | -2.9 |
| Merged         | -1.2 | -3.4 |
| Merged         | -0.4 | -2.6 |
| Merged         | -1.5 | -3.6 |
| Merged         | -0.7 | -2.7 |
| Merged         | -1.6 | -3.8 |
| Merged         | -0.5 | -2.9 |
| Merged         | -1.7 | -4.1 |
| Merged         | -0.3 | -2.5 |
| Merged         | -1.8 | -4.3 |
| Merged         | -0.6   | -2.8 |
| Merged         | -1.9   | -4.5 |
| Merged         | -0.7   | -3.1 |
| Merged         | -2.1   | -4.7 |
| Merged         | -1.3   | -3.5 |
| Merged         | -1.8   | -4.9 |
| Merged         | -0.9   | -3.3 |
| Merged         | -2.3   | -5.1 |
| Merged         | -1.4   | -4.7 |
| Merged         | -1.9   | -5.3 |
| Merged         | -1.1   | -4.5 |
| Merged         | -1.6   | -5.6 |
| Merged         | -1   | -3.7 |
| Merged         | -2    | -5    |
| Merged         | -3    | -4    |
| Merged         | -2    | -5    |
| Merged         | -3    | -4    |
| Merged         | -2    | -5    |
| Merged         | -3    | -4    |
| Merged         | -2    | -5    |
| Merged         | -3    | -4    |
| Merged         | -2    | -5    |
| Merged         | -3*   | -4    |
| Merged         | -2*   | -5    |
| Merged         | -3*   | -4    |
| Merged         | -2*   | -5    |
| Merged         | -3*   | -4    |
| Merged         | -2*   | -5    |
| Merged         | -3*   | -4    |
| Merged         | -2*   | -5    |
| Merged         | -3*   | <1     |
| Merged         | <2*   | <2     |
| Merged         | <3*   | <3     |
| Merged         | <2*   | <4     |
| Merged         | <3*   | <5     |
| Merged         | <2*   | <6     |
| Merged         | <3*   | <7     |
| Merged         | <2*   | <8     |
| Merged         | <3*   | <9     |
| Merged         | <2*   | <10    |
| Merged         | <3*   | <11    |
| Merged         | <2*   | <12    |
| Merged         | <3*   | <13    |
| Merged         | <2*   | <14    |
| Merged         | <3*   | <15    |
| Merged         | <2*   | <16    |
| Merged         | <3*   | <17    |
| Merged         | <2*   | <18    |
| Merged         | <3*   | <19    |
| Merged         | <2*   | <2     |
| Merged         | <3*   | <3     |
| Merged         | <2*   | <4     |
| Merged         | <3*   | <5     |
| Merged         | <2*   | <6     |
| Merged         | <3*   | <7     |
| Merged         | <2*   | <8     |
|
| Merged         | <3*   (repeated) for all categories: Original Experts; Surviving; Merged; All categories: Surviving; All categories: Surviving; All categories: Merged.
</details>

(a) ERNIE-4.5-21B Layer 1

![](images/48835c87ca9a469a9b18ca6d57803f98b91985d61f3683dd56e18d05d729a0e2.jpg)

<details>
<summary>scatter</summary>

| Group          | PC1 Range | PC2 Range |
|----------------|-----------|-----------|
| Original Experts | -50 to 0  | 0 to 80   |
| Surviving      | -50 to 0  | 0 to 20   |
| Merged         | -50 to 0  | 0 to 20   |
</details>

(b) ERNIE-4.5-21B Layer 27

![](images/fd920de6eb1fda7b3222e5a683c719de659ab84a2b6f40948757c971d3c63b9b.jpg)

<details>
<summary>scatter</summary>

| Group          | PC1       | PC2       |
| -------------- | --------- | --------- |
| Original Experts | -0.1      | 0.0       |
| Original Experts | 0.0       | 0.2       |
| Original Experts | 0.1       | -0.1      |
| Original Experts | 0.2       | -0.1      |
| Surviving      | -0.1      | 0.0       |
| Surviving      | 0.0       | -0.1      |
| Surviving      | 0.1       | -0.1      |
| Surviving      | 0.2       | -0.1      |
| Merged         | -0.1      | 0.0       |
| Merged         | 0.0       | -0.1      |
| Merged         | 0.1       | -0.1      |
| Merged         | 0.2       | -0.1      |
</details>

(c) Mixtral-8x7B Layer 0

![](images/4cb368326a00964f83de20cdb06c8a84c6d7a1965a61638c59e4b68e3886b7de.jpg)

<details>
<summary>scatter</summary>

| Group          | PC1       | PC2       |
| -------------- | --------- | --------- |
| Original Experts | -100      | 25        |
| Original Experts | 0         | 20        |
| Original Experts | 50        | 25        |
| Original Experts | 100       | 20        |
| Original Experts | -50       | -60       |
| Original Experts | -100      | -40       |
| Original Experts | -50       | -20       |
| Original Experts | 0         | 0         |
| Original Experts | 50        | 20        |
| Original Experts | 100       | 25        |
| Original Experts | -100      | -60       |
| Original Experts | -50       | -40       |
| Original Experts | 0         | -20       |
| Original Experts | 50        | 0         |
| Original Experts | 100       | 20        |
| Survival      | -100      | -40       |
| Survival      | 0         | 0         |
| Survival      | 50        | 20        |
| Survival      | 100       | 25        |
| Survival      | -100      | -20       |
| Survival      | -50       | -40       |
| Survival      | 0         | -20       |
| Survival      | 50        | 0         |
| Survival      | 100       | 20        |
| Merged         | -100      | 0         |
| Merged         | -50       | 20        |
| Merged         | 0         | 40        |
| Merged         | 50        | 60        |
| Merged         | 100       | 80        |
| Merged         | -100      | -40       |
| Merged         | -50       | -60       |
| Merged         | 0         | -40       |
| Merged         | 50        | -20       |
| Merged         | 100       | 0         |
| Merged         | -100      | 20        |
| Merged         | -50       | 40        |
| Merged         | 0         | 60        |
| Merged         | 50        | 80        |
| Merged         | 100       | 100       |
</details>

(d) Mixtral-8x7B Layer 31

![](images/6485186a98792c1eea6a4abade05217a8c0dc328e34c9fc656851c3ef370cec4.jpg)  
(e) Llama-4-Scout Layer 0

![](images/a357fddde24e3196dafb3c22cf23fca3dd492f14fc0791f9f8a25bb4bea47ec8.jpg)

<details>
<summary>scatter</summary>

| Group          | PC1 Range     | PC2 Range     |
| -------------- | ------------- | ------------- |
| Original Experts | -500 to 500   | -200 to 200   |
| Surviving      | -500 to 500   | -200 to 200   |
| Merged         | -500 to 500   | -200 to 200   |
</details>

(f) Llama-4-Scout Layer 47   
Figure A5: (a,c,e) Functional subspace (PCA) for early SMoE layers. Pruning (blue) preserves the manifold geometry; merging (green) collapses it toward the centre. (b,d,f) Functional subspace (PCA) for late MoE layers.

The role of expert granularity. Both Qwen3-30B-A3B and ERNIE-4.5-21B are highly-granular SMoEs containing 128 and 64 routed experts per layer, respectively, and 8 and 6 routed experts per token, respectively. Functional subspace collapse due to expert merging is more pronounced in these models than in low-granularity models such as Mixtral-8x7B and Llama-4-Scout. With fewer experts and a lower amount of experts per token, low-granularity SMoEs appear to better preserve the variance of their expert output manifolds under expert merging. For example, in Figure A5c, the merged manifold spans along PC1 from approximately [−0.1,0.2] whereas the pruned manifold spans from approximately [0.0,0.2] along PC1. Similarly, as depicted in Figure A5e, the pruned and merged manifolds span PC1 along [−0.04,0.0] and [−0.05,0.05], respectively.

However, the merged manifold is distorted by the introduction of novel expert functions. For example, in Figure A5e, expert merging introduces novel functions which occupy approximately [0.05,0.005] and [−0.025,0.005] which are significantly different than any of the original experts. This is best exemplified by Figure 2, which plots the Wasserstein distance between the original and compressed expert output manifolds in terms of normalized angular distance. Compared to the pruned models, the higher distances between the merged and original expert output manifolds suggest a lower degree of similarity. The distorted manifold of the merged expert outputs represents a loss of fidelity with the original manifold which cannot be restored in the one-shot compression setting.

# B.2 WASSERSTEIN DISTANCE

To quantify the distortion of the original expert output manifold, we measure the 1-Wasserstein distance (Earth Mover’s distance) between the original and compressed expert output manifolds, see Figure 2.

The distance is calculated between the discrete empirical distributions of the original expert outputs, $\mathcal { F } { = } \{ f _ { 1 } , . . . , f _ { K } \}$ , and compressed expert outputs, $\hat { \mathcal { F } } { = } \{ \hat { f } _ { 1 } { , } { \ldots } , \hat { f } _ { K / 2 } \}$ , projected onto the unit hypersphere

Table A4: Cumulative variance explained by PC1 and PC2 across compression methods. Compared to pruning, merging results in a consistently higher explained variance suggesting that the merged models have lost some of their high-dimensional complexity. 

<table><tr><td>Model</td><td>Layer</td><td>Baseline</td><td>Merged</td><td>Pruned</td></tr><tr><td>Qwen3-30B-A3B</td><td>0</td><td>0.2343</td><td>0.2700</td><td>0.1845</td></tr><tr><td>Qwen3-30B-A3B</td><td>47</td><td>0.7195</td><td>0.7437</td><td>0.6860</td></tr><tr><td>ERNIE-4.5-21B</td><td>0</td><td>0.3836</td><td>0.2851</td><td>0.2733</td></tr><tr><td>ERNIE-4.5-21B</td><td>26</td><td>0.2563</td><td>0.4599</td><td>0.0785</td></tr><tr><td>Llama-4-Scout</td><td>0</td><td>0.9032</td><td>0.9343</td><td>0.8480</td></tr><tr><td>Llama-4-Scout</td><td>47</td><td>0.9473</td><td>0.9546</td><td>0.8754</td></tr><tr><td>Mixtral-8x7B</td><td>0</td><td>0.6486</td><td>0.8479</td><td>0.4016</td></tr><tr><td>Mixtral-8x7B</td><td>31</td><td>0.8580</td><td>0.8140</td><td>0.7027</td></tr></table>

$$
W _ {1} (\mathcal {F}, \hat {\mathcal {F}}) = \inf _ {\gamma \in \Gamma (\mu , \nu)} \sum_ {i = 1} ^ {K} \sum_ {j = 1} ^ {K / 2} \gamma_ {i j} \frac {1}{\pi} \arccos \left(\frac {f _ {i} \cdot \hat {f} _ {j}}{\| f _ {i} \| \| \hat {f} _ {j} \|}\right) \tag {12}
$$

where $\mu$ and ν are uniform probability measures over the indices of $\mathcal { F }$ and $\hat { \mathcal { F } }$ respectively, $\Gamma ( \mu , \nu )$ is the set of all transport plans (joint distributions) with marginals $\mu$ and ν, and the cost function is defined as the normalized angular distance. This metric quantifies the minimum "work" required to transport the probability mass from the compressed functional manifold to cover the original manifold, thereby penalizing both the contraction of variance (subspace collapse) and the introduction of functionally distinct artifacts (distortion).

# C EVALUATION DETAILS

Multiple choice (MC) evaluation. Following Chen et al. (2025), our MC benchmarks include: AI2 Reasoning Challenge (ARC-c & ARC-e) (Clark et al., 2018), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), MMLU (Hendrycks et al., 2021a), OpenBookQA (OBQA) (Mihaylov et al., 2018), Recognizing Textual Entailment Challenge (RTE) (Bentivogli et al., 2009), and WinoGrande (WinoG.) (Sakaguchi et al., 2021). We evaluate the models in the zero-shot setting using the standard log-likelihood approach with lm-eval-harness (Gao et al., 2023). We report byte-length normalized accuracies for ARC-c, ARC-e, HellaSwag, and OBQA1.

Coding evaluation. For code generation, all models are evaluated on EvalPlus (Liu et al., 2023) and 182 LiveCodeBench (Jain et al., 2025) questions collected between January and April 2025. We extend the original source code for these benchmarks to evaluate our models. We additionally evaluate Kimi-K2-Instruct-W4A16 and Qwen3-Coder-480B on the agentic coding benchmark SWE-Bench (Jimenez et al., 2024) and tool-calling benchmark BFCLv3 (Patil et al., 2025). For BFCLv3, we use the original Gorilla framework for evaluating our models (Patil et al., 2024).

For SWE-Bench evaluation, we run our compressed models with the mini-SWE-agent scaffolding (Yang et al., 2024b) and report the score on the SWE-Bench Verified test set (Neil Chowdhury et al., 2024). We use 4,096 and 16,384 as the maximum number of output tokens for evaluating Qwen3-Coder-480B and Kimi-K2-Instruct-W4A16 on SWE-Bench, respectively. The input context length for both models is limited to 65,536. We do not limit the number of turns in mini-SWE-agent flow, but restart the rollout in cases where the model could not generate a valid patch (that is, in the case when the output of the final turn does not contain a diff --git substring). We set the maximum number of restarts to 20, which we found to be sufficient to generate patches for all samples with pruned models, unless the model produces degenerate responses like repeating strings. We use the cloud-based evaluation provided with the sb-cli tool to get the final scores for all evaluated models.

For $\tau ^ { 2 } .$ -bench Barres et al. (2025), we use greedy decoding and 4,096 as the maximum number of output tokens for each LLM call. For user simulation, we use the gpt-4.1-2025-04-14 model;

maximum number of steps is 100 and number of trials is set to three for each domain. Following Artificial Analysis (2025), we additionally implement an LLM-based repetition checking step. Every 30 steps of the simulation, a model (in our case, $\mathfrak { g p t - 4 . 1 - m i n i - 2 0 2 5 - 0 4 - 1 4 ) }$ is given the past 30 episodes of the conversation trajectory with a repetition checking prompt to determine whether the agent is stuck in the loop or making meaningful progress. This allows early task termination if the agent is stuck. We use the same decoding parameters for the repetition model as for the user and assistant models.

Math and creative writing evaluation. Mathematical reasoning is assessed on GSM8K (Cobbe et al., 2021) and MATH-500 (Hendrycks et al., 2021b; Lightman et al., 2023) benchmarks using the evalscope (ModelScope Team, 2024) framework. To assess creative writing, we use 146 creative writing prompts sampled from WildBench (Lin et al., 2024a) with $\mathfrak { g p t - 4 o - 2 0 2 4 - 0 5 - 1 3 }$ used as the judge to evaluate the model responses. We report normalized scores using the WildBench rubric.

Generation configuration. For models with ≤ 110B parameters, we use greedy sampling (i.e, temperature = 0.0) to evaluate code generation and math reasoning. For creative writing we use the default temperature, top-P, and top-K settings for each respective model. The maximum number of output tokens is extended to 16,384 for all generative tasks to account for the verbosity of some models. For hybrid reasoning models such as Qwen3-30B-A3B, we disable reasoning on all tasks by setting enable\_thinking=False in the chat template.

For larger models with $\ge ~ 1 1 0 \mathsf { B }$ parameters, we use greedy sampling for EvalPlus, SWE-Bench, and BFCLv3. On LiveCodeBench, Qwen3-Coder-480B and Kimi-K2 are evaluated with default sampling parameters and greedy sampling, respectively. We report the mean and standard deviation for Qwen3-Coder-480B on LiveCodeBench over five random seeds. We use a repetition penalty of 1.05 for all large model evaluations. For EvalPlus we use 768 as the maximum number of output tokens and 16,384 for LiveCodeBench. For BFCLv3 we set the maximum number of output tokens to 4,096.

Model details. The Kimi-K2-Instruct-W4A16 model used throughout this study is an INT4 weight-quantized version of Kimi-K2-Instruct released by RedHatAI (2025).

# D BASELINE METHODS

The following formally describes the baselines compression methods we consider.

Notation. Let $\mathcal { X } _ { c a l }$ be a calibration dataset. Consider a SMoE model with n layers, $L _ { n }$ , K experts per layer $f _ { 1 } , \ldots , f _ { K } $ , each a function $f _ { k } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , and a router producing non-negative gates $\begin{array} { r } { \mathbf { \tilde { g } } ( x ) = ( g _ { 1 } ( x ) , . . . , g _ { K } ( x ) ) \in \Delta ^ { K - 1 } } \end{array}$ . The output of layer $L _ { n }$ is

$$
h _ {n} = \sum_ {i} ^ {K} g _ {i} (x) f _ {i} (x).
$$

The expert usage frequency, $\nu _ { i } ,$ for expert $f _ { i }$ is the number of tokens in $\mathcal { X } _ { c a l }$ for which $f _ { i }$ is activated

$$
\nu_ {i} = | \mathcal {X} _ {i} |,
$$

where $\mathcal { X } _ { i } = \{ x \in \mathcal { X } _ { c a l } | i \in \mathrm { T o p K } ( \mathbf { g } ( x ) ) \}$ .

Given saliency scores, $\mathbf { S } \in \mathbb { R } ^ { K }$ , pruning removes experts with the minimum saliency score. For merging, we first cluster experts based on their pairwise distances, $\mathbf { D } \in \mathbb { R } ^ { K \times K }$ , and then merge the parameters of experts contained within each cluster.

Frequency-based pruning. The frequency-based pruning saliency criterion prunes experts with the lowest usage frequency across the calibration dataset. The saliency of $f _ { i }$ is simply $S _ { i } { = } \nu _ { i }$ .

EAN pruning. EAN pruning introduced by Jaiswal et al. (2025) accumulates the activation norm of each expert across tokens for which the expert is activated. The saliency of $f _ { i }$ is

$$
S _ {i} = \sum_ {x \in \mathcal {X} _ {i}} \| f _ {i} (x) \| _ {2}. \tag {13}
$$

M-SMoE merging. Proposed by Li et al. (2023), M-SMoE first uses weight-matching (Ainsworth et al., 2023) to find a permutation matrix $\mathbf { P _ { j } }$ which aligns expert $f _ { j }$ to expert $f _ { i } .$ . In the models we study, each expert is a two-layer feed-forward SwiGLU block (Shazeer, 2020) with up, gate, and down projections: $f _ { j } = \{ W _ { u p } ^ { ( j ) } , W _ { g a t e } ^ { ( j ) } , W _ { d o w n } ^ { ( j ) } \}$ W (j)gate,W (j)down}. The permutation matrix is applied to the intermediate dimension of the experts such that the expert outputs are invariant to the transformation

$$
W _ {u p} ^ {\prime (j)} = W _ {u p} ^ {(j)} \mathbf {P} _ {j}, \qquad \qquad W _ {g a t e} ^ {\prime (j)} = W _ {g a t e} ^ {(j)} \mathbf {P} _ {j}, \qquad \qquad W _ {d o w n} ^ {\prime (j)} = \mathbf {P} _ {j} ^ {T} W _ {d o w n} ^ {(j)}.
$$

The permuted expert is defined as $\tilde { { f } } _ { j } = \{ W _ { u p } ^ { \prime ( j ) } , W _ { g a t e } ^ { \prime ( j ) } , W _ { d o w n } ^ { \prime ( j ) } \}$ W ′(j)down}.

To initialize the expert clusters, M-SMoE identifies the set of m dominant experts $\mathbb { F } _ { d o m }$ , as the experts across all layers with the highest usage frequency ν. The pairwise expert distance is based on the cosine distance of the router gate-values measured on the calibration dataset

$$
D _ {i, j} = \frac {1}{| \mathcal {X} _ {c a l} |} \sum_ {x \in \mathcal {X} _ {c a l}} 1 - \frac {g _ {i} (x) \cdot g _ {j} (x)}{\| g _ {i} (x) \| \| g _ {j} (x) \|}. \tag {14}
$$

Non-dominant expert $j$ is clustered by selecting the dominant expert with the smallest pairwise distance

$$
i ^ {*} = \underset {i \in \mathbb {F} _ {d o m}} {\operatorname{argmin}} D _ {i, j}.
$$

The merged expert $f _ { \alpha }$ is created by calculating the frequency-weighted average of the permuted parameters, $W ^ { \prime }$ , of all experts in the cluster $\bar { \mathbb { C } } _ { \alpha }$

$$
\tilde {W} _ {a} = \frac {\sum_ {i \in \mathbb {C} _ {\alpha}} \nu_ {i} W _ {i} ^ {\prime}}{\sum_ {i \in \mathbb {C} _ {\alpha} \nu_ {i}}}. \tag {15}
$$

HC-SMoE merging. Chen et al. (2025) clusters experts based on their representative vectors, $A _ { i } ,$ , defined as the average activation across every token in the calibration dataset

$$
A _ {i} := \mathbb {E} _ {x \sim \mathcal {X} _ {c a l}} [ f _ {i} (x) ] = \frac {1}{| \mathcal {X} _ {c a l} |} \sum_ {x \in \mathcal {X} _ {c a l}} f _ {i} (x).
$$

The expert pairwise distance is defined as the cosine distance between representative vectors

$$
D _ {i, j} = 1 - \frac {A _ {i} \cdot A _ {j}}{\| A _ {i} \| \| A _ {j} \|}. \tag {16}
$$

Clusters are formed using hierarchical agglomerative clustering with average linkage criterion. We start by initializing each expert as a singleton cluster. At every iteration, the closest pair of clusters, $\mathbb { C } _ { i } ^ { * } , \mathbb { C } _ { j } ^ { * }$ are joined and the pairwise distances updated as the average of the constituents

$$
i ^ {*}, j ^ {*} = \underset {i, j} {\operatorname{argmin}} D _ {i, j}, \qquad \qquad \mathbb {C} _ {\alpha} = \mathbb {C} _ {i ^ {*}} \cup \mathbb {C} _ {j ^ {*}}, \qquad \qquad D _ {a, k} = \frac {\sum_ {i \in \mathbb {C} _ {\alpha}} D _ {i , k}}{| \mathbb {C} _ {\alpha} |}.
$$

The clusters are merged with Equation (15).

# E QUANTIZATION AND EXPERT PRUNING

Pruning is easily combined with quantization, as demonstrated by our large-scale evaluations. We calibrate, prune, and evaluate Qwen3-Coder-480B and Kimi-K2 using FP8 and W4A16 quantization, respectively. In contrast, combining expert merging with quantization is more complex; the fine-grained shared quantization scales across parameter groups must be reconciled during merging, whereas pruning requires no such adjustment

Generally, weight quantization offers superior accuracy at a given compression ratio down to 4-bits. For instance, quantizing Qwen3-30B-A3B to 4-bits using AWQ (Lin et al., 2024b) outperforms a 50% expert-pruned REAP model at 16-bits, despite having half the checkpoint size. However, combining these methods enables compression rates that neither can achieve in isolation. For example, our Kimi-K2 results combine 4-bit weights with 50% expert pruning, representing a total size reduction of 87.5%. Achieving a similar footprint through quantization alone would require 2-bit weights, which are not natively supported on most current accelerators and typically suffer from significant accuracy degradation (Liu et al., 2025b).

To further illustrate this, we calibrate and quantize Qwen3-30B-A3B using the evol-codealpaca-v1 dataset to 4- and 2-bit weights with the LLM Compressor library (Red Hat AI and vLLM Project, 2024). We additionally compress the 4-bit checkpoint by using REAP calibrated on the same dataset to prune half the experts. In Table A5, we report EvalPlus accuracy versus checkpoint size relative to the uncompressed 16-bit model. This preliminary analysis suggests that REAP combined with 4-bit weights outperforms more aggressive quantization. A more thorough exploration of various low-bit schemes and pruning combinations remains an important direction for future work.

<table><tr><td>Quantization</td><td>Num. Experts</td><td>Approx. Relative Size</td><td>Eval+</td></tr><tr><td>W16A16</td><td>128</td><td>100.0%</td><td>81.4</td></tr><tr><td>W16A16</td><td>64</td><td>50.0%</td><td>78.0</td></tr><tr><td>W4A16-G128</td><td>128</td><td>25.0%</td><td>80.5</td></tr><tr><td>W4A16-G128</td><td>64</td><td>12.5%</td><td>77.6</td></tr><tr><td>W2A16-G128</td><td>64</td><td>12.5%</td><td>28.6</td></tr></table>

Table A5: Qwen3-30B-A3B EvalPlus accuracy vs. relative checkpoint size with AWQ quantization, REAP expert pruning, and quantization combined with expert pruning. Overall, quantization outperforms expert pruning up to 4-bit weights. Beyond 4-bit weights, combining quantization with expert pruning yields significantly better accuracy than more aggressive quantization.

# F ADDITIONAL RESULTS

Table A6 shows the full suite of MC question answering benchmarks and the average result across all models and methods. Table A7 tabulates code generation accuracy of compressed SMoE models calibrated on evol-codealpaca. Eval+ is the average of MBPP+ and HE+. The Code Avg column is the average of Eval+ and LiveCodeBench (LiveCode). Table A8 summarizes the accuracy of the various compression methods studied when calibrated with the C4 dataset on coding and MC benchmarks. Notably, while the MC performance is generally slightly higher than models calibrated on evol-codealpaca, the resulting code generation quality is abysmal, with most models failing to generate coherent output.

Figure A6 plots non-agentic coding and MC accuracy versus compressed model size. Figure A7a depicts the proportion of singleton clusters for HC-SMoE and M-SMoE. Figure A7b plots accuracy vs. maximum cluster sizes when the maximum cardinality of clusters is restricted. Figures A8 and A9 show the importance of using domain-specific calibration data, particularly at high compression ratios.

Table A9 presents the complete τ2-bench results across three domains (Retail, Airline, and Telecom) for the baseline model and REAP compression at 25% and 50% levels. The results show passˆk metrics for k=1, 2, and 3, demonstrating the impact of pruning on evaluating conversational agents, specifically designed to test their ability to collaborate with a user in real-world scenarios.

Table A10 presents additional τ2-bench results with REAP compression for a set of non-reasoning and reasoning models (Qwen3-Coder-30B, GLM4.5-Air, MiniMax-M2). Furthermore, Table A11 presents BFCLv3 scores for the same set of REAP-compressed models (with an inclusion of REAP-compressed GLM-4.6-FP8 variants in thinking mode). The calibration data mix used for non-reasoning models in these experiments is the same as for the other large-scale models in this study (12,228 samples including tool calling and agentic trajectory data). For the models that support reasoning, we include an extra 12,228 samples from the Open-R1 Mixture-of-Thoughts dataset (HuggingFace, 2025), drawing 4,096 samples from the Code, Math and Science subsets with a maximum sequence length of 16,384 tokens.

Table A12 shows results for the Kimi-Linear-48B-A3B-Instruct model as an example of REAP compression applied to a hybrid linear attention architecture. The calibration data mix is made up of 24,414 samples with a maximum sequence length of 16,384 tokens. The samples are drawn from evol-codealpaca (Chaudhary, 2023; Luo et al., 2024; Tam, 2023), xlam-function-calling (Liu et al., 2024c; Salesforce, 2025), SWE-smith-trajectories (Yang et al., 2025c;b), WildChat (Zhao et al., 2024), NuminaMath (LI et al., 2024) and Aya (Singh et al., 2024). The calibration data mix composition reflects coding, tool calling, multilingual and general conversational use cases. We evaluate Kimi-Linear-48B-A3B-Instruct on coding and math benchmarks at 30% REAP compression, as well as on long-context question-answering benchmarks LongBench v2 (Bai et al., 2025) and FRAMES (Krishna et al., 2024). For long-context benchmarks, a truncation in the middle is done for input prompts exceeding 131,072 tokens in length. The accuracy scores for FRAMES are estimated with LLM-as-a-judge $( { \tt g p t - 4 . 1 - 2 0 2 5 - 0 4 - 1 4 }$ model as judge). Notably, even though the maximum sequence length during calibration is 16,384 tokens, the REAP-compressed model retains performance at context lengths far exceeding that value.

Table A6: Detailed benchmark results for multiple-choice QA tasks. 

<table><tr><td>Model</td><td>Compression</td><td>Technique</td><td>Method</td><td>ARC-c</td><td>ARC-e</td><td>BoolQ</td><td>Hellaswag</td><td>MMLU</td><td>OBQA</td><td>RTE</td><td>WinoG.</td><td>MC Avg</td></tr><tr><td rowspan="11">ERNIE-4.5-21B-A3B-PT</td><td>Baseline</td><td></td><td></td><td>0.564</td><td>0.782</td><td>0.873</td><td>0.813</td><td>0.737</td><td>0.462</td><td>0.812</td><td>0.724</td><td>0.721</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> ${0.434} \pm {0.006}$ </td><td> ${0.652} \pm {0.008}$ </td><td> ${0.846} \pm {0.001}$ </td><td> ${0.597} \pm {0.002}$ </td><td> ${0.591} \pm {0.001}$ </td><td> ${0.350} \pm {0.006}$ </td><td> ${0.819} \pm {0.010}$ </td><td> ${0.655} \pm {0.003}$ </td><td> ${0.618} \pm {0.002}$ </td></tr><tr><td>HC-SMoE</td><td> ${0.506} \pm {0.000}$ </td><td> ${0.717} \pm {0.001}$ </td><td> ${0.849} \pm {0.001}$ </td><td> ${0.714} \pm {0.001}$ </td><td> ${0.652} \pm {0.002}$ </td><td> ${0.371} \pm {0.002}$ </td><td> ${0.799} \pm {0.002}$ </td><td> ${0.674} \pm {0.004}$ </td><td> ${0.660} \pm {0.001}$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> ${0.486} \pm {0.004}$ </td><td> ${0.711} \pm {0.000}$ </td><td> ${0.852} \pm {0.004}$ </td><td> ${0.675} \pm {0.003}$ </td><td> ${0.628} \pm {0.003}$ </td><td> ${0.373} \pm {0.003}$ </td><td> ${0.780} \pm {0.006}$ </td><td> ${0.676} \pm {0.005}$ </td><td> ${0.648} \pm {0.001}$ </td></tr><tr><td>EAN</td><td> ${0.498} \pm {0.005}$ </td><td> ${0.713} \pm {0.002}$ </td><td> ${0.863} \pm {0.002}$ </td><td> ${0.717} \pm {0.004}$ </td><td> ${0.625} \pm {0.001}$ </td><td> ${0.405} \pm {0.011}$ </td><td> ${0.811} \pm {0.009}$ </td><td> ${0.702} \pm {0.005}$ </td><td> ${0.667} \pm {0.000}$ </td></tr><tr><td>REAP</td><td> ${0.526} \pm {0.004}$ </td><td> ${0.756} \pm {0.006}$ </td><td> ${0.858} \pm {0.003}$ </td><td> ${0.704} \pm {0.001}$ </td><td> ${0.636} \pm {0.002}$ </td><td> ${0.401} \pm {0.001}$ </td><td> ${0.765} \pm {0.010}$ </td><td> ${0.690} \pm {0.002}$ </td><td> ${0.667} \pm {0.000}$ </td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> ${0.294} \pm {0.033}$ </td><td> ${0.452} \pm {0.040}$ </td><td> ${0.764} \pm {0.010}$ </td><td> ${0.341} \pm {0.011}$ </td><td> ${0.385} \pm {0.001}$ </td><td> ${0.270} \pm {0.004}$ </td><td> ${0.687} \pm {0.017}$ </td><td> ${0.529} \pm {0.010}$ </td><td> ${0.465} \pm {0.012}$ </td></tr><tr><td>HC-SMoE</td><td> ${0.411} \pm {0.003}$ </td><td> ${0.641} \pm {0.002}$ </td><td> ${0.822} \pm {0.001}$ </td><td> ${0.523} \pm {0.001}$ </td><td> ${0.495} \pm {0.002}$ </td><td> ${0.330} \pm {0.005}$ </td><td> ${0.742} \pm {0.011}$ </td><td> ${0.587} \pm {0.009}$ </td><td> ${0.569} \pm {0.001}$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> ${0.400} \pm {0.002}$ </td><td> ${0.584} \pm {0.006}$ </td><td> ${0.830} \pm {0.001}$ </td><td> ${0.522} \pm {0.003}$ </td><td> ${0.506} \pm {0.006}$ </td><td> ${0.303} \pm {0.004}$ </td><td> ${0.758} \pm {0.004}$ </td><td> ${0.625} \pm {0.004}$ </td><td> ${0.566} \pm {0.002}$ </td></tr><tr><td>EAN</td><td> ${0.417} \pm {0.005}$ </td><td> ${0.633} \pm {0.005}$ </td><td> ${0.830} \pm {0.003}$ </td><td> ${0.572} \pm {0.001}$ </td><td> ${0.509} \pm {0.002}$ </td><td> ${0.336} \pm {0.003}$ </td><td> ${0.785} \pm {0.014}$ </td><td> ${0.626} \pm {0.003}$ </td><td> ${0.589} \pm {0.003}$ </td></tr><tr><td>REAP</td><td> ${0.407} \pm {0.003}$ </td><td> ${0.628} \pm {0.006}$ </td><td> ${0.820} \pm {0.002}$ </td><td> ${0.551} \pm {0.004}$ </td><td> ${0.491} \pm {0.001}$ </td><td> ${0.331} \pm {0.007}$ </td><td> ${0.767} \pm {0.004}$ </td><td> ${0.614} \pm {0.002}$ </td><td> ${0.576} \pm {0.001}$ </td></tr><tr><td rowspan="11">Qwen3-30B-A3B</td><td>Baseline</td><td></td><td></td><td>0.563</td><td>0.790</td><td>0.887</td><td>0.778</td><td>0.779</td><td>0.454</td><td>0.816</td><td>0.702</td><td>0.721</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> ${0.357} \pm {0.006}$ </td><td> ${0.519} \pm {0.003}$ </td><td> ${0.843} \pm {0.006}$ </td><td> ${0.529} \pm {0.002}$ </td><td> ${0.536} \pm {0.004}$ </td><td> ${0.310} \pm {0.005}$ </td><td> ${0.735} \pm {0.027}$ </td><td> ${0.635} \pm {0.005}$ </td><td> ${0.558} \pm {0.003}$ </td></tr><tr><td>HC-SMoE</td><td> ${0.478} \pm {0.006}$ </td><td> ${0.722} \pm {0.006}$ </td><td> ${0.863} \pm {0.003}$ </td><td> ${0.714} \pm {0.000}$ </td><td> ${0.684} \pm {0.002}$ </td><td> ${0.417} \pm {0.001}$ </td><td> ${0.805} \pm {0.004}$ </td><td> ${0.710} \pm {0.004}$ </td><td> ${0.674} \pm {0.001}$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> ${0.401} \pm {0.011}$ </td><td> ${0.600} \pm {0.016}$ </td><td> ${0.847} \pm {0.003}$ </td><td> ${0.593} \pm {0.005}$ </td><td> ${0.600} \pm {0.004}$ </td><td> ${0.342} \pm {0.012}$ </td><td> ${0.781} \pm {0.002}$ </td><td> ${0.637} \pm {0.005}$ </td><td> ${0.600} \pm {0.005}$ </td></tr><tr><td>EAN</td><td> ${0.406} \pm {0.007}$ </td><td> ${0.603} \pm {0.014}$ </td><td> ${0.847} \pm {0.005}$ </td><td> ${0.607} \pm {0.006}$ </td><td> ${0.600} \pm {0.002}$ </td><td> ${0.337} \pm {0.003}$ </td><td> ${0.764} \pm {0.002}$ </td><td> ${0.660} \pm {0.009}$ </td><td> ${0.603} \pm {0.004}$ </td></tr><tr><td>REAP</td><td> ${0.481} \pm {0.004}$ </td><td> ${0.727} \pm {0.002}$ </td><td> ${0.855} \pm {0.004}$ </td><td> ${0.700} \pm {0.006}$ </td><td> ${0.673} \pm {0.001}$ </td><td> ${0.399} \pm {0.008}$ </td><td> ${0.789} \pm {0.014}$ </td><td> ${0.696} \pm {0.003}$ </td><td> ${0.665} \pm {0.002}$ </td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> ${0.278} \pm {0.003}$ </td><td> ${0.402} \pm {0.003}$ </td><td> ${0.753} \pm {0.004}$ </td><td> ${0.399} \pm {0.002}$ </td><td> ${0.366} \pm {0.004}$ </td><td> ${0.278} \pm {0.002}$ </td><td> ${0.586} \pm {0.014}$ </td><td> ${0.546} \pm {0.004}$ </td><td> ${0.451} \pm {0.002}$ </td></tr><tr><td>HC-SMoE</td><td> ${0.368} \pm {0.002}$ </td><td> ${0.593} \pm {0.003}$ </td><td> ${0.740} \pm {0.003}$ </td><td> ${0.473} \pm {0.002}$ </td><td> ${0.516} \pm {0.003}$ </td><td> ${0.301} \pm {0.007}$ </td><td> ${0.724} \pm {0.004}$ </td><td> ${0.620} \pm {0.005}$ </td><td> ${0.542} \pm {0.001}$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> ${0.285} \pm {0.001}$ </td><td> ${0.424} \pm {0.002}$ </td><td> ${0.779} \pm {0.003}$ </td><td> ${0.458} \pm {0.003}$ </td><td> ${0.397} \pm {0.002}$ </td><td> ${0.286} \pm {0.004}$ </td><td> ${0.659} \pm {0.012}$ </td><td> ${0.570} \pm {0.009}$ </td><td> ${0.483} \pm {0.001}$ </td></tr><tr><td>EAN</td><td> ${0.296} \pm {0.006}$ </td><td> ${0.426} \pm {0.009}$ </td><td> ${0.759} \pm {0.007}$ </td><td> ${0.471} \pm {0.002}$ </td><td> ${0.443} \pm {0.001}$ </td><td> ${0.291} \pm {0.009}$ </td><td> ${0.668} \pm {0.020}$ </td><td> ${0.589} \pm {0.009}$ </td><td> ${0.493} \pm {0.003}$ </td></tr><tr><td>REAP</td><td> ${0.354} \pm {0.006}$ </td><td> ${0.503} \pm {0.008}$ </td><td> ${0.737} \pm {0.009}$ </td><td> ${0.481} \pm {0.004}$ </td><td> ${0.496} \pm {0.003}$ </td><td> ${0.309} \pm {0.001}$ </td><td> ${0.561} \pm {0.020}$ </td><td> ${0.584} \pm {0.004}$ </td><td> ${0.503} \pm {0.002}$ </td></tr><tr><td rowspan="11">Mixtral-8x7B-Instruct-v0.1</td><td>Baseline</td><td></td><td></td><td>0.650</td><td>0.842</td><td>0.887</td><td>0.861</td><td>0.691</td><td>0.496</td><td>0.722</td><td>0.740</td><td>0.736</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> ${0.532} \pm {0.004}$ </td><td> ${0.769} \pm {0.007}$ </td><td> ${0.847} \pm {0.001}$ </td><td> ${0.747} \pm {0.002}$ </td><td> ${0.553} \pm {0.001}$ </td><td> ${0.429} \pm {0.008}$ </td><td> ${0.632} \pm {0.010}$ </td><td> ${0.656} \pm {0.004}$ </td><td> ${0.646} \pm {0.001}$ </td></tr><tr><td>HC-SMoE</td><td> ${0.590} \pm {0.004}$ </td><td> ${0.797} \pm {0.004}$ </td><td> ${0.869} \pm {0.003}$ </td><td> ${0.835} \pm {0.002}$ </td><td> ${0.626} \pm {0.000}$ </td><td> ${0.482} \pm {0.004}$ </td><td> ${0.703} \pm {0.012}$ </td><td> ${0.731} \pm {0.007}$ </td><td> ${0.704} \pm {0.001}$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> ${0.616} \pm {0.014}$ </td><td> ${0.826} \pm {0.007}$ </td><td> ${0.875} \pm {0.001}$ </td><td> ${0.825} \pm {0.002}$ </td><td> ${0.637} \pm {0.003}$ </td><td> ${0.451} \pm {0.003}$ </td><td> ${0.706} \pm {0.017}$ </td><td> ${0.692} \pm {0.005}$ </td><td> ${0.704} \pm {0.002}$ </td></tr><tr><td>EAN</td><td> ${0.607} \pm {0.004}$ </td><td> ${0.831} \pm {0.011}$ </td><td> ${0.884} \pm {0.001}$ </td><td> ${0.836} \pm {0.001}$ </td><td> ${0.646} \pm {0.002}$ </td><td> ${0.484} \pm {0.005}$ </td><td> ${0.700} \pm {0.004}$ </td><td> ${0.732} \pm {0.004}$ </td><td> ${0.715} \pm {0.000}$ </td></tr><tr><td>REAP</td><td> ${0.600} \pm {0.004}$ </td><td> ${0.822} \pm {0.011}$ </td><td> ${0.872} \pm {0.000}$ </td><td> ${0.830} \pm {0.000}$ </td><td> ${0.642} \pm {0.000}$ </td><td> ${0.469} \pm {0.002}$ </td><td> ${0.771} \pm {0.002}$ </td><td> ${0.716} \pm {0.002}$ </td><td> ${0.715} \pm {0.000}$ </td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> ${0.446} \pm {0.005}$ </td><td> ${0.700} \pm {0.011}$ </td><td> ${0.788} \pm {0.003}$ </td><td> ${0.630} \pm {0.002}$ </td><td> ${0.430} \pm {0.011}$ </td><td> ${0.386} \pm {0.003}$ </td><td> ${0.570} \pm {0.000}$ </td><td> ${0.596} \pm {0.005}$ </td><td> ${0.568} \pm {0.011}$ </td></tr><tr><td>HC-SMoE</td><td> ${0.539} \pm {0.003}$ </td><td> ${0.759} \pm {0.011}$ </td><td> ${0.851} \pm {0.011}$ </td><td> ${0.791} \pm {0.011}$ </td><td> ${0.543} \pm {0.011}$ </td><td> ${0.442} \pm {0.011}$ </td><td> ${0.712} \pm {0.012}$ </td><td> ${0.667} \pm {0.011}$ </td><td> ${0.667} \pm {0.011}$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> ${0.541} \pm {0.014}$ </td><td> ${0.781} \pm {0.013}$ </td><td> ${0.824} \pm {0.13}$ </td><td> ${0.759} \pm {0.12}$ </td><td> ${0.516} \pm {0.12}$ </td><td> ${0.411} \pm {0.12}$ </td><td> ${0.787} \pm {0.12}$ </td><td> ${0.655} \pm {0.12}$ </td><td> ${0.649} \pm {0.12}$ </td></tr><tr><td>EAN</td><td> ${0.551} \pm {0.14}$ </td><td> ${0.774} \pm {0.13}$ </td><td> ${0.859} \pm {0.12}$ </td><td> ${0.794} \pm {0.12}$ </td><td> ${0.550} \pm {0.12}$ </td><td> ${0.452} \pm {0.14}$ </td><td> ${0.717} \pm {0.12}$ </td><td> ${0.693} \pm {0.12}$ </td><td> ${0.674} \pm {0.12}$ </td></tr><tr><td>REAP</td><td> ${0.546} \pm {0.13}$ </td><td> ${0.782} \pm {0.13}$ </td><td> ${0.841} \pm {0.13}$ </td><td> ${0.777} \pm {0.12}$ </td><td> ${0.565} \pm {0.12}$ </td><td> ${0.438} \pm {0.13}$ </td><td> ${0.739} \pm {0.12}$ </td><td> ${0.677} \pm {0.12}$ </td><td> ${0.673} \pm {0.12}$ </td></tr><tr><td rowspan="4">GLM-4.5-Air</td><td>Baseline</td><td></td><td></td><td>0.619</td><td>0.825</td><td>0.879</td><td>0.823</td><td>0.803</td><td>0.462</td><td>0.765</td><td>0.692</td><td>0.738</td></tr><tr><td rowspan="3">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> ${0.429} \pm {0.11}$ </td><td> ${0.651} \pm {1.11}$ </td><td> ${0.872} \pm {1.11}$ </td><td> ${0.752} \pm {1.11}$ </td><td> ${0.719} \pm {1.11}$ </td><td> ${0.434} \pm {1.11}$ </td><td> ${0.769} \pm {1.11}$ </td><td> ${0.695} \pm {1.11}$ </td><td> ${0.699} \pm {1.11}$ </td></tr><tr><td>HC-SMoE</td><td> ${0.578} \pm {1.11}$ </td><td> ${0.814} \pm {1.11}$ </td><td> ${1.876} \pm {1.11}$ </td><td> ${1.779} \pm {1.11}$ </td><td> ${1.729} \pm {1.11}$ </td><td> ${1.424} \pm {1.11}$ </td><td> ${1.729} \pm {1.11}$ </td><td> ${1.695} \pm {1.11}$ </td><td> ${1.729} \pm {1.11}$ </td></tr><tr><td>Pruning</td><td>Frequency</td><td> ${0.493} \pm {1.11}$ </td><td>\( {0.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table A7: Detailed benchmark results for non-agentic code generation tasks. Eval+ is the average of MBPP+ and HE+. The Code Avg column is the average of Eval+ and LiveCodeBench (LiveCode). 

<table><tr><td>Model</td><td>Compression</td><td>Technique</td><td>Method</td><td>HE</td><td>HE+</td><td>MBPP</td><td>MBPP+</td><td>Eval+</td><td>LiveCode</td><td>Code Avg</td></tr><tr><td rowspan="11">ERNIE-4.5-21B-A3B-PT</td><td>Baseline</td><td></td><td></td><td>0.902</td><td>0.866</td><td>0.910</td><td>0.765</td><td>0.815</td><td>0.231</td><td>0.523</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.774 \pm 0.011$ </td><td> $0.730 \pm 0.009$ </td><td> $0.768 \pm 0.015$ </td><td> $0.647 \pm 0.017$ </td><td> $0.688 \pm 0.007$ </td><td> $0.194 \pm 0.022$ </td><td> $0.441 \pm 0.011$ </td></tr><tr><td>HC-SMoE</td><td> $0.837 \pm 0.007$ </td><td> $0.805 \pm 0.000$ </td><td> $0.827 \pm 0.003$ </td><td> $0.696 \pm 0.008$ </td><td> $0.750 \pm 0.004$ </td><td> $0.207 \pm 0.008$ </td><td> $0.479 \pm 0.003$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.890 \pm 0.006$ </td><td> $0.846 \pm 0.009$ </td><td> $0.837 \pm 0.010$ </td><td> $0.709 \pm 0.010$ </td><td> $0.777 \pm 0.009$ </td><td> $0.151 \pm 0.096$ </td><td> $0.464 \pm 0.044$ </td></tr><tr><td>EAN</td><td> $0.890 \pm 0.006$ </td><td> $0.848 \pm 0.011$ </td><td> $0.840 \pm 0.006$ </td><td> $0.727 \pm 0.004$ </td><td> $0.787 \pm 0.007$ </td><td> $0.161 \pm 0.111$ </td><td> $0.474 \pm 0.053$ </td></tr><tr><td>REAP</td><td> $0.909 \pm 0.012$ </td><td> $0.868 \pm 0.004$ </td><td> $0.866 \pm 0.006$ </td><td> $0.735 \pm 0.002$ </td><td> $0.801 \pm 0.002$ </td><td> $0.223 \pm 0.008$ </td><td> $0.512 \pm 0.005$ </td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.104 \pm 0.022$ </td><td> $0.100 \pm 0.029$ </td><td> $0.239 \pm 0.036$ </td><td> $0.207 \pm 0.040$ </td><td> $0.153 \pm 0.015$ </td><td> $0.024 \pm 0.008$ </td><td> $0.089 \pm 0.009$ </td></tr><tr><td>HC-SMoE</td><td> $0.425 \pm 0.004$ </td><td> $0.404 \pm 0.007$ </td><td> $0.608 \pm 0.018$ </td><td> $0.511 \pm 0.011$ </td><td> $0.458 \pm 0.008$ </td><td> $0.082 \pm 0.015$ </td><td> $0.270 \pm 0.008$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.699 \pm 0.031$ </td><td> $0.640 \pm 0.022$ </td><td> $0.696 \pm 0.014$ </td><td> $0.584 \pm 0.006$ </td><td> $0.612 \pm 0.014$ </td><td> $0.083 \pm 0.066$ </td><td> $0.348 \pm 0.026$ </td></tr><tr><td>EAN</td><td> $0.675 \pm 0.019$ </td><td> $0.642 \pm 0.009$ </td><td> $0.713 \pm 0.015$ </td><td> $0.591 \pm 0.016$ </td><td> $0.617 \pm 0.012$ </td><td> $0.112 \pm 0.064$ </td><td> $0.364 \pm 0.034$ </td></tr><tr><td>REAP</td><td> $0.787 \pm 0.038$ </td><td> $0.752 \pm 0.035$ </td><td> $0.749 \pm 0.005$ </td><td> $0.638 \pm 0.013$ </td><td> $0.695 \pm 0.024$ </td><td> $0.187 \pm 0.005$ </td><td> $0.441 \pm 0.014$ </td></tr><tr><td rowspan="11">Qwen3-30B-A3B</td><td>Baseline</td><td></td><td></td><td>0.927</td><td>0.884</td><td>0.881</td><td>0.743</td><td>0.814</td><td>0.302</td><td>0.558</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.878 \pm 0.012$ </td><td> $0.833 \pm 0.007$ </td><td> $0.849 \pm 0.007$ </td><td> $0.728 \pm 0.007$ </td><td> $0.781 \pm 0.007$ </td><td> $0.293 \pm 0.017$ </td><td> $0.537 \pm 0.006$ </td></tr><tr><td>HC-SMoE</td><td> $0.866 \pm 0.011$ </td><td> $0.805 \pm 0.016$ </td><td> $0.832 \pm 0.006$ </td><td> $0.698 \pm 0.005$ </td><td> $0.752 \pm 0.006$ </td><td> $0.258 \pm 0.000$ </td><td> $0.505 \pm 0.003$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.921 \pm 0.006$ </td><td> $0.874 \pm 0.007$ </td><td> $0.868 \pm 0.000$ </td><td> $0.735 \pm 0.003$ </td><td> $0.805 \pm 0.005$ </td><td> $0.302 \pm 0.011$ </td><td> $0.553 \pm 0.003$ </td></tr><tr><td>EAN</td><td> $0.909 \pm 0.006$ </td><td> $0.864 \pm 0.004$ </td><td> $0.859 \pm 0.009$ </td><td> $0.729 \pm 0.008$ </td><td> $0.797 \pm 0.005$ </td><td> $0.311 \pm 0.018$ </td><td> $0.554 \pm 0.010$ </td></tr><tr><td>REAP</td><td> $0.911 \pm 0.004$ </td><td> $0.870 \pm 0.004$ </td><td> $0.847 \pm 0.004$ </td><td> $0.725 \pm 0.008$ </td><td> $0.797 \pm 0.004$ </td><td> $0.304 \pm 0.003$ </td><td> $0.551 \pm 0.004$ </td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.687 \pm 0.013$ </td><td> $0.638 \pm 0.004$ </td><td> $0.618 \pm 0.004$ </td><td> $0.541 \pm 0.007$ </td><td> $0.590 \pm 0.005$ </td><td> $0.205 \pm 0.019$ </td><td> $0.397 \pm 0.007$ </td></tr><tr><td>HC-SMoE</td><td> $0.577 \pm 0.023$ </td><td> $0.541 \pm 0.013$ </td><td> $0.631 \pm 0.010$ </td><td> $0.546 \pm 0.004$ </td><td> $0.543 \pm 0.005$ </td><td> $0.185 \pm 0.018$ </td><td> $0.364 \pm 0.007$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.787 \pm 0.016$ </td><td> $0.756 \pm 0.022$ </td><td> $0.692 \pm 0.016$ </td><td> $0.579 \pm 0.016$ </td><td> $0.668 \pm 0.019$ </td><td> $0.236 \pm 0.025$ </td><td> $0.452 \pm 0.022$ </td></tr><tr><td>EAN</td><td> $0.886 \pm 0.025$ </td><td> $0.837 \pm 0.020$ </td><td> $0.798 \pm 0.006$ </td><td> $0.669 \pm 0.008$ </td><td> $0.753 \pm 0.011$ </td><td> $0.306 \pm 0.003$ </td><td> $0.530 \pm 0.004$ </td></tr><tr><td>REAP</td><td> $0.917 \pm 0.013$ </td><td> $0.858 \pm 0.015$ </td><td> $0.818 \pm 0.008$ </td><td> $0.703 \pm 0.004$ </td><td> $0.780 \pm 0.006$ </td><td> $0.302 \pm 0.000$ </td><td> $0.541 \pm 0.003$ </td></tr><tr><td rowspan="11">Mixtral-8x7B-Instruct-v0.1</td><td>Baseline</td><td></td><td></td><td>0.524</td><td>0.476</td><td>0.556</td><td>0.463</td><td>0.469</td><td>0.123</td><td>0.296</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.315 \pm 0.007$ </td><td> $0.270 \pm 0.015$ </td><td> $0.446 \pm 0.007$ </td><td> $0.380 \pm 0.015$ </td><td> $0.325 \pm 0.015$ </td><td> $0.033 \pm 0.010$ </td><td> $0.179 \pm 0.011$ </td></tr><tr><td>HC-SMoE</td><td> $0.439 \pm 0.028$ </td><td> $0.386 \pm 0.020$ </td><td> $0.530 \pm 0.022$ </td><td> $0.441 \pm 0.007$ </td><td> $0.414 \pm 0.007$ </td><td> $0.110 \pm 0.010$ </td><td> $0.262 \pm 0.001$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.400 \pm 0.034$ </td><td> $0.358 \pm 0.035$ </td><td> $0.541 \pm 0.006$ </td><td> $0.453 \pm 0.012$ </td><td> $0.405 \pm 0.019$ </td><td> $0.099 \pm 0.014$ </td><td> $0.252 \pm 0.004$ </td></tr><tr><td>EAN</td><td> $0.413 \pm 0.027$ </td><td> $0.366 \pm 0.024$ </td><td> $0.477 \pm 0.009$ </td><td> $0.409 \pm 0.013$ </td><td> $0.388 \pm 0.015$ </td><td> $0.111 \pm 0.006$ </td><td> $0.249 \pm 0.006$ </td></tr><tr><td>REAP</td><td> $0.445 \pm 0.016$ </td><td> $0.388 \pm 0.025$ </td><td> $0.548 \pm 0.010$ </td><td> $0.470 \pm 0.011$ </td><td> $0.429 \pm 0.008$ </td><td> $0.097 \pm 0.006$ </td><td> $0.263 \pm 0.007$ </td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.085 \pm 0.026$ </td><td> $0.076 \pm 0.022$ </td><td> $0.139 \pm 0.121$ </td><td> $0.118 \pm 0.102$ </td><td> $0.127 \pm 0.011$ </td><td> $0.004 \pm 0.006$ </td><td> $0.063 \pm 0.005$ </td></tr><tr><td>HC-SMoE</td><td> $0.175 \pm 0.015$ </td><td> $0.146 \pm 0.000$ </td><td> $0.335 \pm 0.026$ </td><td> $0.282 \pm 0.031$ </td><td> $0.214 \pm 0.015$ </td><td> $0.013 \pm 0.008$ </td><td> $0.114 \pm 0.007$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.187 \pm 0.015$ </td><td> $0.148 \pm 0.007$ </td><td> $0.342 \pm 0.016$ </td><td> $0.287 \pm 0.012$ </td><td> $0.217 \pm 0.007$ </td><td> $0.023 \pm 0.004$ </td><td> $0.120 \pm 0.004$ </td></tr><tr><td>EAN</td><td> $0.220 \pm 0.006$ </td><td> $0.189 \pm 0.006$ </td><td> $0.375 \pm 0.020$ </td><td> $0.325 \pm 0.015$ </td><td> $0.257 \pm 0.005$ </td><td> $0.031 \pm 0.011$ </td><td> $0.144 \pm 0.006$ </td></tr><tr><td>REAP</td><td> $0.258 \pm 0.019$ </td><td> $0.220 \pm 0.016$ </td><td> $0.381 \pm 0.003$ </td><td> $0.331 \pm 0.008$ </td><td> $0.275 \pm 0.011$ </td><td> $0.055 \pm 0.010$ </td><td> $0.165 \pm 0.001$ </td></tr><tr><td rowspan="11">Llama-4-Scout-17B-16E-Instruct</td><td>Baseline</td><td></td><td></td><td>0.829</td><td>0.768</td><td>0.788</td><td>0.640</td><td>0.704</td><td>0.341</td><td>0.522</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.823$ </td><td> $0.762$ </td><td> $0.786$ </td><td> $0.635$ </td><td> $0.699$ </td><td> $0.324$ </td><td> $0.511$ </td></tr><tr><td>HC-SMoE</td><td> $0.787$ </td><td> $0.738$ </td><td> $0.735$ </td><td> $0.587$ </td><td> $0.663$ </td><td> $0.148$ </td><td> $0.405$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.835$ </td><td> $0.768$ </td><td> $0.788$ </td><td> $0.630$ </td><td> $0.699$ </td><td> $0.317$ </td><td> $0.508$ </td></tr><tr><td>EAN</td><td> $0.823$ </td><td> $0.762$ </td><td> $0.804$ </td><td> $0.648$ </td><td> $0.705$ </td><td> $0.328$ </td><td> $0.517$ </td></tr><tr><td>REAP</td><td> $0.829$ </td><td> $0.787$ </td><td> $0.788$ </td><td> $0.622$ </td><td> $0.704$ </td><td> $0.242$ </td><td> $0.473$ </td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.787$ </td><td> $0.732$ </td><td> $0.762$ </td><td> $0.614$ </td><td> $0.673$ </td><td> $0.187$ </td><td> $0.430$ </td></tr><tr><td>HC-SMoE</td><td> $0.604$ </td><td> $0.530$ </td><td> $0.500$ </td><td> $0.399$ </td><td> $0.465$ </td><td> $0.077$ </td><td> $0.271$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.823$ </td><td> $0.756$ </td><td> $0.751$ </td><td> $0.595$ </td><td> $0.676$ </td><td> $0.223$ </td><td> $0.449$ </td></tr><tr><td>EAN</td><td> $0.805$ </td><td> $0.744$ </td><td> $0.754$ </td><td> $0.601$ </td><td> $0.672$ </td><td> $0.209$ </td><td> $0.441$ </td></tr><tr><td>REAP</td><td> $0.841$ </td><td> $0.768$ </td><td> $0.762$ </td><td> $0.624$ </td><td> $0.696$ </td><td> $0.248$ </td><td> $0.472$ </td></tr><tr><td rowspan="11">GLM-4.5-Air</td><td>Baseline</td><td></td><td></td><td> $0.848$ </td><td> $0.829$ </td><td> $0.860$ </td><td> $0.743$ </td><td> $0.786$ </td><td> $0.374$ </td><td> $0.580$ </td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.866$ </td><td> $0.793$ </td><td> $0.807$ </td><td> $0.659$ </td><td> $0.726$ </td><td> $0.330$ </td><td> $0.528$ </td></tr><tr><td>HC-SMoE</td><td> $0.872$ </td><td> $0.805$ </td><td> $0.825$ </td><td> $0.669$ </td><td> $0.737$ </td><td> $0.363$ </td><td> $0.550$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.848$ </td><td> $0.811$ </td><td> $0.854$ </td><td> $0.706$ </td><td> $0.759$ </td><td> $0.341$ </td><td> $0.550$ </td></tr><tr><td>EAN</td><td> $0.872$ </td><td> $0.817$ </td><td> $0.876$ </td><td> $0.720$ </td><td> $0.768$ </td><td> $0.374$ </td><td> $0.571$ </td></tr><tr><td>REAP</td><td> $0.884$ </td><td> $0.829$ </td><td> $0.839$ </td><td> $0.688$ </td><td> $0.759$ </td><td> $0.412$ </td><td> $0.585$ </td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td> $0.518$ </td><td> $0.500$ </td><td> $0.519$ </td><td> $0.437$ </td><td> $0.468$ </td><td> $0.099$ </td><td> $0.284$ </td></tr><tr><td>HC-SMoE</td><td> $0.707$ </td><td> $0.659$ </td><td> $0.706$ </td><td> $0.577$ </td><td> $0.618$ </td><td> $0.220$ </td><td> $0.419$ </td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td> $0.628$ </td><td> $0.573$ </td><td> $0.534$ </td><td> $0.450$ </td><td> $0.511$ </td><td> $0.104$ </td><td> $0.308$ </td></tr><tr><td>EAN</td><td> $0.841$ </td><td> $0.780$ </td><td> $0.807$ </td><td> $0.661$ </td><td> $0.721$ </td><td> $0.253$ </td><td> $0.487$ </td></tr><tr><td>REAP</td><td> $0.823$ </td><td> $0.780$ </td><td> $0.712$ </td><td> $0.577$ </td><td> $0.679$ </td><td> $0.352$ </td><td> $0.515$ </td></tr><tr><td>Qwen3-Coder-48B-A3B-Instruct-FP8</td><td>Baseline</td><td></td><td></td><td> $0.951$ </td><td> $0.890$ </td><td> $0.923$ </td><td> $0.791$ </td><td> $0.841$ </td><td>\( 0.4</td><td></td></tr></table>

Table A8: C4 calibrated results for coding and MC tasks. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Compression</td><td rowspan="2">Technique</td><td rowspan="2">Method</td><td colspan="3">Coding</td><td rowspan="2">ARC-c</td><td rowspan="2">ARC-e</td><td rowspan="2">BoolQ</td><td rowspan="2">Hellaswag</td><td rowspan="2">MC MMLU</td><td rowspan="2">OBQA</td><td rowspan="2">RTE</td><td rowspan="2">WinoG.</td><td rowspan="2">MC Avg</td></tr><tr><td>Eval+</td><td>LiveCode</td><td>Code Avg</td></tr><tr><td rowspan="11">ERNIE-4.5-21B-A3B-PT</td><td>Baseline</td><td></td><td></td><td>0.815</td><td>0.231</td><td>0.523</td><td>0.564</td><td>0.782</td><td>0.873</td><td>0.813</td><td>0.737</td><td>0.462</td><td>0.812</td><td>0.724</td><td>0.721</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.061</td><td>0.016</td><td>0.039</td><td>0.497</td><td>0.729</td><td>0.860</td><td>0.723</td><td>0.602</td><td>0.424</td><td>0.801</td><td>0.699</td><td>0.667</td></tr><tr><td>HC-SMoE</td><td>0.369</td><td>0.099</td><td>0.234</td><td>0.515</td><td>0.728</td><td>0.860</td><td>0.745</td><td>0.649</td><td>0.428</td><td>0.794</td><td>0.694</td><td>0.677</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.254</td><td>0.000</td><td>0.127</td><td>0.515</td><td>0.735</td><td>0.841</td><td>0.719</td><td>0.588</td><td>0.382</td><td>0.791</td><td>0.683</td><td>0.657</td></tr><tr><td>EAN</td><td>0.262</td><td>0.000</td><td>0.131</td><td>0.528</td><td>0.750</td><td>0.853</td><td>0.790</td><td>0.558</td><td>0.442</td><td>0.783</td><td>0.706</td><td>0.676</td></tr><tr><td>REAP</td><td>0.212</td><td>0.055</td><td>0.133</td><td>0.553</td><td>0.784</td><td>0.843</td><td>0.775</td><td>0.635</td><td>0.454</td><td>0.798</td><td>0.708</td><td>0.694</td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.297</td><td>0.460</td><td>0.674</td><td>0.449</td><td>0.312</td><td>0.280</td><td>0.671</td><td>0.575</td><td>0.465</td></tr><tr><td>HC-SMoE</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.409</td><td>0.615</td><td>0.666</td><td>0.515</td><td>0.489</td><td>0.290</td><td>0.632</td><td>0.580</td><td>0.524</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.393</td><td>0.625</td><td>0.717</td><td>0.569</td><td>0.496</td><td>0.324</td><td>0.758</td><td>0.619</td><td>0.563</td></tr><tr><td>EAN</td><td>0.007</td><td>0.003</td><td>0.005</td><td>0.451</td><td>0.676</td><td>0.742</td><td>0.687</td><td>0.474</td><td>0.398</td><td>0.736</td><td>0.691</td><td>0.607</td></tr><tr><td>REAP</td><td>0.015</td><td>0.000</td><td>0.008</td><td>0.403</td><td>0.562</td><td>0.713</td><td>0.668</td><td>0.391</td><td>0.388</td><td>0.708</td><td>0.669</td><td>0.563</td></tr><tr><td rowspan="11">Qwen3-30B-A3B</td><td>Baseline</td><td></td><td></td><td>0.814</td><td>0.302</td><td>0.558</td><td>0.563</td><td>0.790</td><td>0.887</td><td>0.778</td><td>0.779</td><td>0.454</td><td>0.816</td><td>0.702</td><td>0.721</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.551</td><td>0.768</td><td>0.883</td><td>0.761</td><td>0.733</td><td>0.418</td><td>0.848</td><td>0.701</td><td>0.708</td></tr><tr><td>HC-SMoE</td><td>0.788</td><td>0.269</td><td>0.529</td><td>0.470</td><td>0.713</td><td>0.833</td><td>0.622</td><td>0.646</td><td>0.376</td><td>0.805</td><td>0.665</td><td>0.641</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.548</td><td>0.789</td><td>0.889</td><td>0.775</td><td>0.735</td><td>0.438</td><td>0.801</td><td>0.694</td><td>0.709</td></tr><tr><td>EAN</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.569</td><td>0.802</td><td>0.889</td><td>0.774</td><td>0.735</td><td>0.438</td><td>0.801</td><td>0.697</td><td>0.713</td></tr><tr><td>REAP</td><td>0.763</td><td>0.253</td><td>0.508</td><td>0.555</td><td>0.771</td><td>0.864</td><td>0.740</td><td>0.736</td><td>0.452</td><td>0.805</td><td>0.693</td><td>0.702</td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.262</td><td>0.348</td><td>0.693</td><td>0.479</td><td>0.237</td><td>0.290</td><td>0.523</td><td>0.542</td><td>0.422</td></tr><tr><td>HC-SMoE</td><td>0.688</td><td>0.209</td><td>0.449</td><td>0.316</td><td>0.495</td><td>0.715</td><td>0.354</td><td>0.422</td><td>0.282</td><td>0.603</td><td>0.536</td><td>0.465</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.349</td><td>0.488</td><td>0.782</td><td>0.672</td><td>0.503</td><td>0.364</td><td>0.588</td><td>0.619</td><td>0.545</td></tr><tr><td>EAN</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.480</td><td>0.736</td><td>0.876</td><td>0.760</td><td>0.607</td><td>0.424</td><td>0.762</td><td>0.694</td><td>0.667</td></tr><tr><td>REAP</td><td>0.329</td><td>0.104</td><td>0.217</td><td>0.404</td><td>0.616</td><td>0.828</td><td>0.643</td><td>0.517</td><td>0.360</td><td>0.632</td><td>0.637</td><td>0.580</td></tr><tr><td rowspan="11">Mixtral-8x7B-Instruct-v0.1</td><td>Baseline</td><td></td><td></td><td>0.469</td><td>0.123</td><td>0.296</td><td>0.650</td><td>0.842</td><td>0.887</td><td>0.861</td><td>0.691</td><td>0.496</td><td>0.722</td><td>0.740</td><td>0.736</td></tr><tr><td rowspan="5">25%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.296</td><td>0.044</td><td>0.170</td><td>0.532</td><td>0.775</td><td>0.828</td><td>0.746</td><td>0.529</td><td>0.424</td><td>0.603</td><td>0.632</td><td>0.634</td></tr><tr><td>HC-SMoE</td><td>0.385</td><td>0.121</td><td>0.253</td><td>0.608</td><td>0.811</td><td>0.876</td><td>0.838</td><td>0.631</td><td>0.484</td><td>0.736</td><td>0.726</td><td>0.714</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.370</td><td>0.070</td><td>0.220</td><td>0.612</td><td>0.816</td><td>0.868</td><td>0.836</td><td>0.593</td><td>0.482</td><td>0.675</td><td>0.739</td><td>0.703</td></tr><tr><td>EAN</td><td>0.360</td><td>0.092</td><td>0.226</td><td>0.613</td><td>0.814</td><td>0.875</td><td>0.842</td><td>0.613</td><td>0.498</td><td>0.690</td><td>0.733</td><td>0.710</td></tr><tr><td>REAP</td><td>0.371</td><td>0.088</td><td>0.230</td><td>0.590</td><td>0.810</td><td>0.878</td><td>0.835</td><td>0.638</td><td>0.468</td><td>0.736</td><td>0.710</td><td>0.708</td></tr><tr><td rowspan="5">50%</td><td rowspan="2">Merging</td><td>M-SMoE</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.260</td><td>0.460</td><td>0.614</td><td>0.395</td><td>0.240</td><td>0.302</td><td>0.527</td><td>0.526</td><td>0.416</td></tr><tr><td>HC-SMoE</td><td>0.152</td><td>0.033</td><td>0.093</td><td>0.540</td><td>0.764</td><td>0.862</td><td>0.795</td><td>0.544</td><td>0.448</td><td>0.675</td><td>0.709</td><td>0.667</td></tr><tr><td rowspan="3">Pruning</td><td>Frequency</td><td>0.156</td><td>0.008</td><td>0.082</td><td>0.504</td><td>0.739</td><td>0.793</td><td>0.771</td><td>0.463</td><td>0.426</td><td>0.675</td><td>0.646</td><td>0.627</td></tr><tr><td>EAN</td><td>0.127</td><td>0.008</td><td>0.068</td><td>0.550</td><td>0.756</td><td>0.842</td><td>0.804</td><td>0.529</td><td>0.460</td><td>0.726</td><td>0.716</td><td>0.673</td></tr><tr><td>REAP</td><td>0.121</td><td>0.022</td><td>0.071</td><td>0.531</td><td>0.779</td><td>0.869</td><td>0.787</td><td>0.543</td><td>0.460</td><td>0.773</td><td>0.697</td><td>0.680</td></tr></table>

Table A9: $\tau ^ { 2 } .$ -bench results with REAP compression across different benchmark domains on Qwen3-480B-A35B-Coder-FP8. 

<table><tr><td>Dataset</td><td>Compression</td><td>Method</td><td>pass^1</td><td>pass^2</td><td>pass^3</td></tr><tr><td rowspan="3">Retail</td><td>Baseline</td><td></td><td>0.643</td><td>0.544</td><td>0.500</td></tr><tr><td>25%</td><td>REAP</td><td>0.661</td><td>0.535</td><td>0.465</td></tr><tr><td>50%</td><td>REAP</td><td>0.632</td><td>0.515</td><td>0.456</td></tr><tr><td rowspan="3">Airline</td><td>Baseline</td><td></td><td>0.460</td><td>0.340</td><td>0.280</td></tr><tr><td>25%</td><td>REAP</td><td>0.487</td><td>0.367</td><td>0.320</td></tr><tr><td>50%</td><td>REAP</td><td>0.447</td><td>0.333</td><td>0.280</td></tr><tr><td rowspan="3">Telecom</td><td>Baseline</td><td></td><td>0.500</td><td>0.398</td><td>0.325</td></tr><tr><td>25%</td><td>REAP</td><td>0.529</td><td>0.456</td><td>0.421</td></tr><tr><td>50%</td><td>REAP</td><td>0.471</td><td>0.339</td><td>0.263</td></tr></table>

Table A10: Additional $\tau ^ { 2 } .$ -bench results (passˆ1 scores) across non-reasoning and reasoning models and compression levels with REAP. Interleaved thinking is not applied for MiniMax-M2 (thinking traces from previous turns are discarded in multi-turn trajectories). 

<table><tr><td>Model</td><td>Dataset</td><td>Baseline</td><td>20%</td><td>25%</td><td>30%</td><td>40%</td></tr><tr><td rowspan="3">Qwen3-Coder-30B-A3B</td><td>Airline</td><td>0.393</td><td>0.407</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Retail</td><td>0.626</td><td>0.620</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Telecom</td><td>0.336</td><td>0.322</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="4">GLM-4.5-Air</td><td>Airline (Thinking off)</td><td>0.633</td><td>-</td><td>0.640</td><td>-</td><td>-</td></tr><tr><td>Retail (Thinking off)</td><td>0.728</td><td>-</td><td>0.751</td><td>-</td><td>-</td></tr><tr><td>Telecom (Thinking off)</td><td>0.284</td><td>-</td><td>0.307</td><td>-</td><td>-</td></tr><tr><td>Telecom (Thinking on)</td><td>0.272</td><td>-</td><td>0.269</td><td>-</td><td>-</td></tr><tr><td>MiniMax-M2</td><td>Telecom</td><td>0.591</td><td>-</td><td>0.576</td><td>0.591</td><td>0.553</td></tr></table>

Table A11: Additional BFCLv3 results across across non-reasoning and reasoning models and compression levels with REAP. Interleaved thinking is not applied for MiniMax-M2 (thinking traces from previous turns are discarded in multi-turn trajectories). 

<table><tr><td>Model</td><td>Benchmark</td><td>Baseline</td><td>20%</td><td>25%</td><td>30%</td><td>40%</td></tr><tr><td>Qwen3-Coder-30B-A3B</td><td>BFCLv3</td><td>0.632</td><td>0.622</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GLM-4.5-Air</td><td>BFCLv3 (Thinking)</td><td>0.768</td><td>-</td><td>0.763</td><td>-</td><td>-</td></tr><tr><td>GLM-4.6-FP8</td><td>BFCLv3 (Thinking)</td><td>0.784</td><td>-</td><td>0.773</td><td>0.768</td><td>0.742</td></tr><tr><td>MiniMax-M2</td><td>BFCLv3</td><td>0.626</td><td>-</td><td>0.615</td><td>0.599</td><td>0.579</td></tr></table>

Table A12: Kimi-Linear-48B-A3B-Instruct results at 30% REAP compression across coding, math, and long-context evaluations. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Compression</td><td colspan="3">Coding</td><td colspan="2">Math</td><td colspan="2">Long-Context</td></tr><tr><td>HumanEval+</td><td>MBPP+</td><td>LiveCodeBench</td><td>MATH-500</td><td>GSM8K</td><td>LongBench v2</td><td>FRAMES</td></tr><tr><td rowspan="2">Kimi-Linear-48B-A3B-Instruct</td><td>Baseline</td><td>0.823</td><td>0.669</td><td>0.276</td><td>0.818</td><td>0.873</td><td>0.368</td><td>0.557</td></tr><tr><td>30%</td><td>0.811</td><td>0.693</td><td>0.302</td><td>0.808</td><td>0.858</td><td>0.372</td><td>0.523</td></tr></table>

![](images/a88a8a66cc28049c96d9fa8b71ee780653c0fafadf51e6839830b7df9d320cab.jpg)

<details>
<summary>line</summary>

| Log Scaled Total Parameters (in billions) | Non-Agentic Code Acc. (%) |
| ----------------------------------------- | ------------------------- |
| 10^1                                      | 35                        |
| 10^1                                      | 45                        |
| 10^1                                      | 50                        |
| 10^1                                      | 55                        |
| 10^1                                      | 60                        |
| 10^2                                      | 28                        |
| 10^2                                      | 30                        |
| 10^2                                      | 35                        |
| 10^2                                      | 40                        |
| 10^2                                      | 45                        |
| 10^2                                      | 50                        |
| 10^2                                      | 55                        |
| 10^2                                      | 60                        |
| 10^3                                      | 28                        |
| 10^3                                      | 30                        |
| 10^3                                      | 35                        |
| 10^3                                      | 40                        |
| 10^3                                      | 45                        |
| 10^3                                      | 50                        |
| 10^3                                      | 55                        |
| 10^3                                      | 60                        |
</details>

![](images/17949d828874064204329fd4153d5eb3fcfd42457d6b71f4c4cdb226e872b340.jpg)

<details>
<summary>line</summary>

| Pruning methods | MC Accuracy (%) | Log Scaled Total Parameters (in billions) |
| --- | --- | --- |
| Baseline | 72 | 10^1 |
| REAP (ours) | 73 | 10^2 |
| EAN | 68 | 10^2 |
| Frequency | 60 | 10^2 |
| HC-5MoE | 65 | 10^2 |
| M-SMoE | 60 | 10^2 |
| ERNIE-4.5-21B-A3B | 75 | 10^3 |
| Qwen3-30B-A3B | 70 | 10^3 |
| Mixtral-8x7B-Instruct-v0.1 | 60 | 10^3 |
| LLaMA-4-Scout-17B-16E-Instruct | 65 | 10^3 |
| GLM-4.5-Air | 60 | 10^3 |
| Qwen3-Coder-480B-A35B-Instruct-FP8 | 75 | 10^3 |
| Kimi-K2-Instruct-W4A16 | 70 | 10^3 |
</details>

Figure A6: Coding and MC accuracy across all models vs. parameters. The benefits of REAP over other compression methods are evident at 50% compression. For large-scale SMoEs, REAP is near-lossless whereas the shortcomings of frequency-based pruning become apparent.

![](images/16a7639427f7d30b5ed4cd1e624d3f45e64d099464fed1f65b1715322a85d91c.jpg)

<details>
<summary>bar</summary>

| Model | HC-SMoE (%) | M-SMoE (%) |
|---|---|---|
| ERNIE-4.5-21B-A3B-PT | 97 | 57 |
| Qwen3-30B-A3B | 97 | 64 |
| Mixtral-8x7B-Instruct-v0.1 | 57 | 28 |
| Llama-4-Scout-17B-16E-Instruct | 86 | 74 |
| GLM-4.5-Air | 98 | 66 |
</details>

(a) Singleton cluster proportion

![](images/4a494ef8335787cb095fd167ca7d91febefa9b53562e4dd84c5c28aff5447257.jpg)

<details>
<summary>bar</summary>

| Max cluster size | Coding Accuracy (%) | MC Accuracy (%) |
| :--- | :--- | :--- |
| None | 0.41 | 0.45 |
| 32 | 0.125 | 0.45 |
| 16 | 0.115 | 0.45 |
| 8 | 0.115 | 0.45 |
| 4 | 0.21 | 0.44 |
| 2 | 0.0 | 0.36 |
</details>

(b) Restricted cluster sizes   
Figure A7: (a) Average proportion of singleton clusters vs. model for HC-SMoE and M-SMoE. We find that the clustering algorithms used by our baseline merging methods tend to generate a high proportion of singleton clusters containing just a single expert. In order to achieve the desired compression ratio, the large number of singletons conversely results in some clusters which contain many experts, in some cases $\mathsf { \bar { N } } / 2 + 1$ experts for a layer with N experts are grouped into a single cluster. (b) Accuracy vs. maximum cluster size using M-SMoE to compress 50% of experts in Qwen3-30B. While MC accuracy remains stable up to a maximum cluster size of ${ \bar { 4 } } ,$ generative coding capabilities are severely diminished by restricting the clustering algorithm.

![](images/a3ae4d477ba1bbf2d6fec72e7b7af2c28f36f0e6b564c7ab249dfd89b56d39b9.jpg)  
Figure A8: Coding accuracy vs. calibration dataset. Using domain-specific calibration datasets substantially improves compressed model quality within the target domain. Fine-grained models such as Qwen3-30B and ERNIE suffers greater degradation, with several compression methods failing to produce any coherent output when calibrated on C4.

![](images/51eb636377a53840672fe2844bab2b6ae1bb1880d1cb25ccf8d392c89ab9dd9c.jpg)  
Figure A9: Mean accuracy vs. task type for models calibrated with domain specific data versus general data. The “general” calibration data consists of the combination of evol-codealpaca-v1, Writing-Prompts curated, and tulu-3-sft-personas-math and includes three times the total number of samples as the domain-specific calibration datasets. Compared to other compression methods, REAP best preserves accuracy across tasks when calibrated on the general dataset.