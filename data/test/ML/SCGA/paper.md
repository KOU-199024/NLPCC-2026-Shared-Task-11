# IMPROVING BLACK-BOX GENERATIVE ATTACKS VIA GENERATOR SEMANTIC CONSISTENCY

Jongoh Jeong1, Hunmin Yang1,2, Jaeseok Jeong1, and Kuk-Jin Yoon1,

1Visual Intelligence Lab., Korea Advanced Institute of Science and Technology (KAIST)

2Agency for Defense Development (ADD)

{jeong2, hmyang, jason.jeong, kjyoon}@kaist.ac.kr § Project Page

# ABSTRACT

Transfer attacks optimize on a surrogate and deploy to a black-box target. While iterative optimization attacks in this paradigm are limited by their efficiency and scalability due to multistep gradient updates per input, generative attacks alleviate these by producing adversarial examples in a single forward pass at test time. However, current generative attacks still adhere to optimizing surrogate losses (e.g., feature divergence) and overlook the generator’s internal dynamics, underexploring how the generator’s internal representations shape transferable perturbations. To address this, we enforce semantic consistency by aligning the early generator’s intermediate features to an exponential moving average (EMA) teacher, stabilizing object-aligned representations and improving black-box transfer without inferencetime overhead. To ground the mechanism, we quantify semantic stability as the standard deviation of foreground IoU between cluster-derived activation masks and foreground masks across generator blocks, and observe reduced semantic drift under our method. For more reliable evaluation, we also introduce Accidental Correction Rate (ACR) to separate inadvertent corrections from intended misclassifications, complementing the inherent blind spots in traditional Attack Success Rate (ASR), Fooling Rate (FR), and Accuracy metrics. Across architectures, domains, and tasks, our approach can be seamlessly integrated into existing generative attacks with consistent improvements in black-box transfer, while maintaining test-time efficiency. Code at https://github.com/andyj1/scga.

# 1 INTRODUCTION

Deep neural networks have driven advances in computer vision, natural language processing, and medical diagnosis by learning rich hierarchical representations. At the same time, they remain vulnerable to small human-imperceptible perturbations known as adversarial examples (AE) Szegedy et al. (2013), which can induce confident misclassification and raise safety concerns in real deployments. The risk is amplified in black-box settings, where an attacker has no access to the parameters or architecture of a model. In these scenarios, transfer-based attacks craft perturbations on a surrogate and deploy them against unseen targets, enabling a single perturbation strategy to threaten diverse safety-critical systems such as self-driving and biometrics.

Early white-box iterative attacks (for example, FGSM and its multistep variants Zhang et al. (2021); Dong et al. (2018); Xie et al. (2019); Dong et al. (2019)) rely on direct gradient access. Transfer-based attacks extend this idea by seeking perturbations that generalize across models, often using iterative optimization in the white-box regime Madry et al. (2017); Carlini et al. (2019); Zhang et al. (2021). While effective, they require per-example iterative optimization, whereas generative attacks amortize this cost by producing perturbations in a single forward pass.

Generative transfer attacks train a feedforward perturbation generator against a surrogate and then produce adversarial noise with one forward pass at the test time Xiao et al. (2018); Wang et al. (2018); Baluja & Fischer (2017; 2018); Poursaeed et al. (2018); Naseer et al. (2019); Nakka & Salzmann (2021); Zhang et al. (2022); Aich et al. (2022); Yang et al. (2024a;b); Nakka & Alahi (2025). This design yields fast inference and strong scalability. However, current generative attacks are centered around optimizing the surrogate-level objectives and treat the generator merely as a tool to generate adversarial examples given the adversarial objective, overlooking the progressive AE synthesis process in which perturbations are incrementally formed block by block within the generator. This oversight leaves potential for improving transferability, as the intermediate blocks of the generator are where semantic structure, such as the contour of the object and the coarse shape, is preserved or degraded during synthesis Zhang et al. (2022). As a result, perturbations may disperse onto object-irrelevant regions that are relatively less victim model -agnostic, weakening adversarial transferability. This critically raises the following questions:

Q1. At what stage of perturbation synthesis do semantic cues deteriorate?   
Q2. Which generator blocks most influence transferability?

To investigate the perturbation synthesis in detail, we partition the six intermediate blocks in the generator into three split blocks–early, mid, and late–and find that the early blocks better preserve object-aligned structure than later ones. We substantiate this claim with a diagnostic analysis of the stability of object-aligned perturbation semantics within the generator intermediate blocks As in Fig. 1, lower cross-block variability, and thus higher consistency of object semantics, is associated with higher transferability of AEs.

Guided by this observation, we propose a semantically consistent generative attack (SCGA) that explicitly targets semantic consistency during perturbation synthesis within the generator. Concretely, we use a Mean Teacher pathway in which an Exponential Moving Average (EMA)-updated teacher provides temporally smoothed reference features, and a self-feature consistency loss aligns the student’s early generator block activations with these references while keeping the adversarial objective on the surrogate features unchanged, as shown in Fig. 2. This guidance operates only during training without additional test-time cost, and integrates with existing generative attacks.

Finally, we broaden the evaluation beyond misclassification-based metrics (ASR, FR) and a correctionbased metric (Accuracy) to include our proposed Accidental Correction Rate (ACR). For reliable evaluations, ACR complements these conventional metrics by identifying cases that are inherently likely to be overlooked, such as unintended corrections of initially wrong benign predictions. In a comprehensive evaluation setting, we demonstrate that the internal dynamics within the generator play a critical role in enhancing adversarial transferability between domains, models, and even tasks. We summarize our main contributions as follows:

• Generator–internal evidence for perturbation semantics. To investigate perturbation semantics within the generator, we partition the generator into early/mid/late blocks and quantify objectaligned semantics per block. Our analysis reveals that methods with lower variability in the foreground IoU across the intermediate blocks exhibit higher adversarial transfer. (§2.2)   
• Generator–level semantic consistency guidance. By enforcing training-only semantic consistency at the generator’s early intermediates, we achieve improved adversarial transfer while keeping the adversarial objective on the surrogate unchanged. The guidance can be seamlessly integrated into existing generative attacks without altering the test pipeline at no additional inference cost. (§3)   
• Comprehensive evaluation with an added reliability measure. We conduct a comprehensive transferability evaluation spanning classification (CLS) across architectures, domains, and dense prediction tasks (SS, OD). We also complement conventional Accuracy, ASR, and FR metrics by introducing a novel ACR metric to assess the attack reliability, measured by inadvertent corrections from intended misclassifications. (§4.2).

# 2 BACKGROUND AND MOTIVATION

# 2.1 PRELIMINARIES

Given a pre-trained victim model $\mathcal { F } ^ { t } ( \cdot )$ evaluated on a test distribution $\mathcal { D } _ { \mathrm { t e s t } }$ , the objective is to synthesize human-imperceptible perturbations that transfer across models, domains, and tasks, using only a source domain $\mathcal { D } _ { \mathrm { s r c } }$ and its pre-trained models as substitutes. Generative attack framework employs a generator $\mathcal { G } _ { \theta } ( \cdot )$ that maps a benign input x to an unconstrained adversarial candidate $\tilde { x } ^ { \mathrm { a d v } }$ , followed by a projector $\mathcal { P } ( \cdot )$ that enforces the $\ell _ { \infty }$ -budget, i.e., $\| \mathcal { P } ( \tilde { x } ^ { \mathrm { { a d v } } } ) - x \| _ { \infty } \leq \epsilon$ . Training of $\mathcal { G } _ { \boldsymbol { \theta } } ( \cdot )$ is supervised in white-box fashion by a surrogate model $\mathcal { F } ^ { s } ( \cdot )$ trained on $\mathcal { D } _ { \mathrm { s r c } } .$ , enabling gradientbased updates via backpropagation. The adversarial loss leverages surrogate logits or intermediate features of $\mathcal { F } ^ { s } ( \cdot )$ , e.g. at layer k, to capture model-shared characteristics known to enhance black-box transferability Naseer et al. (2019); Nakka & Salzmann (2021); Zhang et al. (2022); Aich et al. (2022); Yang et al. (2024a;b). Formally, $\mathcal { G } _ { \theta } ( \cdot )$ is optimized to generate AEs that maximize evaluation metrics against victim models $\mathcal { F } ^ { t } ( \cdot )$ and/or relative to ground-truth labels y with:

$$
\text { Metric } \left(x, x _ {a d v}, \mathcal {F} ^ {t} (\cdot), y\right), \text {   with   } \| x _ {a d v} - x \| _ {\infty} \leq \epsilon , \quad \text {(See §4.2 for metric details.)} \tag {1}
$$

where ϵ denotes the maximum perturbation budget that guarantees a minimal change in x. Here, Metric refers to ASR, FR, Acc., and ACR for classification (CLS); mIoU and mAP50 for semantic segmentation (SS) and object detection (OD), respectively.

# 2.2 PERTURBATION SEMANTICS IN GENERATOR-INTERNAL DYNAMICS

![](images/e84b0cadd0dad29eb9b18311552a465171e390b2d779f9417c06d7eadb8956a2.jpg)

<details>
<summary>text_image</summary>

Input
AE Generation Path
Early
Mid
Late
CDA	LTP	BIA	GAMA	FACL	PDCL
</details>

(a)

![](images/825b976749792b62d635b2f4b8ae7744a41b3cf19ec56c75fe3a3127ff4992d8.jpg)

<details>
<summary>natural_image</summary>

Time-lapse sequence of green fluorescent plant cells under GT, Early, and Mid stages (no text or symbols)
</details>

(b)

<table><tr><td>Feature ClusterForegroundIoU</td><td colspan="3">Intermediate block</td><td colspan="2">Std.Dev.(Variability) ↓</td></tr><tr><td>Method</td><td>Early</td><td>Mid</td><td>Late</td><td colspan="2">Baseline → w/ Ours</td></tr><tr><td>CDA</td><td>37.72</td><td>36.74</td><td>32.14</td><td>2.77</td><td>1.51</td></tr><tr><td>LTP</td><td>32.48</td><td>28.16</td><td>28.16</td><td>2.59</td><td>2.98</td></tr><tr><td>BIA</td><td>36.17</td><td>33.79</td><td>30.20</td><td>2.82</td><td>2.06</td></tr><tr><td>GAMA</td><td>36.55</td><td>35.95</td><td>31.57</td><td>2.46</td><td>1.41</td></tr><tr><td>FACL</td><td>36.48</td><td>34.38</td><td>31.76</td><td>2.19</td><td>1.17</td></tr><tr><td>PDCL</td><td>35.31</td><td>33.59</td><td>31.00</td><td>2.08</td><td>0.71</td></tr></table>

(c)   
Figure 1: Our observation on the semantic variability within the perturbation generator. (a) Generator intermediate feature maps for each block partition, (b) predicted masks from intermediate feature clusters on ImageNet-S Gao et al. (2022) from the baseline Zhang et al. (2022), and (c) quantified variability in foreground IoU.

We observe that intermediate features progressively lose semantic recognizability across residual blocks. Figure 1 shows that early maps preserve object contours, while mid and late maps blur them. Using k-means clustering to separate the foreground and background, we also find that stronger attacks preserve the coarse shape earlier and more consistently in stages. To better quantify how much semantic information is retained throughout the intermediates, we define semantic variability as the cross-block standard deviation of foreground IoU between clustered activation masks and foreground masks along perturbation trajectories, where advanced attacks achieve lower variability, suggesting more stable overlap with foreground. These findings are consistent with the well-established premise that the majority of noise being synthesized in the intermediate stage Zhang et al. (2022); Naseer et al. (2019); Nakka & Salzmann (2021); Zhang et al. (2022); Aich et al. (2022); Yang et al. (2024a;b). Based on this evidence, we apply a lightweight EMA teacher to early blocks, leaving inference unchanged, so that subsequent blocks concentrate perturbations on salient regions and black-box transfer improves. Further analysis is provided in Supp. §D.

Crucially, these findings motivate our design to enforce semantic consistency in the intermediate stages of the generator, using an EMA teacher applied in the early blocks to curb semantic drift while leaving the inference path unchanged. By anchoring perturbations to early, semantically consistent features, the later blocks naturally concentrate the generated perturbations in salient object regions, thereby improving black-box transfer between models while preserving internal semantics.

# 3 SEMANTICALLY CONSISTENT GENERATIVE ATTACK

Our semantically consistent generative attack, as described in Alg. 1, augments a standard generative adversarial attack with two key components: a Mean Teacher-based feature smoothing and a selffeature consistency loss that enforces semantic preservation across the intermediate layers of the generator. We base our approach on the baseline work BIA Zhang et al. (2022) as all subsequent works GAMA Aich et al. (2022), FACL Yang et al. (2024a), PDCL Yang et al. (2024b) base their losses on its feature similarity-based adversarial loss, and thus it is adequate to serve as a solid baseline. See Supp. §C for the method distinction.

Role of Mean Teacher. The Mean Teacher (MT) framework Tarvainen & Valpola (2017); Deng et al. (2021); Li et al. (2022); Zhao et al. (2022); Cao et al. (2023); Döbler et al. (2023) has consistently demonstrated robustness in tasks characterized by significant domain shifts between training and testing. Its core mechanism of updating the teacher’s parameters with EMA of the student’s parameters provides a form of temporal ensemble that naturally suppresses instance-specific noise. Intuitively, this EMA update smooths out high-frequency perturbation artifacts, enriching the semantic consistency and stability of the teacher’s intermediate feature maps. As a result, these smoothed features serve as a reliable reference for the student, helping to preserve object contours and shapes throughout adversarial synthesis. To integrate MT, we maintain two generators: a student $G _ { \theta } ( \cdot )$ that is trained via gradient descent, and a teacher $G _ { \theta ^ { \prime } } ( \cdot )$ . We set these mean teacher features as a reference for our self-feature consistency matching. We update $\theta ^ { \prime }$ , per training step t, as follows:

![](images/e94230274fbe91b6272ed87a3dcf3d667d5247cbb7b2a1da18295fdbf597f07c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["x"] --> B["Generator Gθ(·)"]
    B --> C["P(·)"]
    C --> D["x_adv"]
    D --> E["Surrogate Model F(·)"]
    E --> F["f^benign"]
    E --> G["f^adv"]
    F --> H["Training Stage"]
    G --> I["Logits or features"]
    H --> J["Evaluation Stage"]
    I --> K["Victim Model"]
    K --> L["Accuracy ASR + ACR FR"]
    L --> M["Comprehensive Evaluation"]
    
    N["x"] --> O["Conv - IN + ReLU"]
    O --> P["Conv - IN + ReLU"]
    P --> Q["Conv - IN + ReLU"]
    Q --> R["Conv - IN + ReLU + Deprior Conv - IN"]
    R --> S["Conv - IN + ReLU + Deprior Conv - IN"]
    S --> T["Conv - IN + ReLU + Deprior Conv - IN"]
    T --> U["Conv - IN + ReLU + Deprior Conv - IN"]
    U --> V["Conv - IN + ReLU + Deprior Conv - IN"]
    V --> W["Tray + Conv. + In + ReLU"]
    W --> X["Tray + Conv. + In + ReLU"]
    X --> Y["Comp"]
    Y --> Z["P(·)"]
    Z --> AA["x_adv"]
    
    AB["Downsampling"] --> AC["Early"]
    AC --> AD["Mid"]
    AD --> AE["Late"]
    AE --> AF["(a)"]
    AF --> AG["(b)"]
    AG --> AH["(c)"]
    AH --> AI["(d)"]
    AI --> AJ["(e)"]
    AJ --> AK["(f)"]
    
    AL["Residual Learning"] --> AM["Upsampling"]
    
    AN["Intermediate Feature Maps"] --> AO["(a)"]
    AO --> AP["(b)"]
    AP --> AQ["(c)"]
    AQ --> AR["(d)"]
    AR --> AS["(e)"]
    AS --> AT["(f)"]
    
    AU["Baseline Zhang et al. (2022)"] --> AV["w/ Ours"]
```
</details>

Figure 2: Overview of our proposed SCGA framework. Given a benign input image, a perturbation generator produces an adversarial output under the supervision of a Mean Teacher (MT) structure. The student and teacher share the generator architecture, with the teacher updated via EMA. Semantic consistency is enforced by aligning their intermediate features, selectively applied to the early blocks to effectively preserve structural information from the benign input across the residual blocks. The adversarial example is then evaluated against victim models according to the four evaluation metrics. This MT-based design further promotes semantic alignment, combining consistency and integrity, thereby enhancing adversarial transferability across diverse victims.

$$
\theta_ {t} ^ {\prime} \leftarrow \eta \theta_ {t - 1} ^ {\prime} + (1 - \eta) \theta_ {t}, \tag {2}
$$

where $\eta \in [ 0 , 1 ]$ is a smoothing coefficient hyperparameter.

Self-feature consistency. Object-salient intermediate representations have been shown to be critical for adversarial transfer in black-box settings Wu et al. (2020); Byun et al. (2022); Kim et al. (2022); Zhang et al. (2022), and recent work has explored manipulating input or surrogate-level features to this end Huang et al. (2019); Li et al. (2023); Nakka & Alahi (2025). In our generative framework, however, a naïve generator progressively loses semantic integrity in its intermediate layers (Fig. 1), scattering perturbations away from object-salient regions. To preserve these crucial object cues, we introduce a self-feature consistency mechanism grounded in the MT paradigm Grill et al. (2020); Caron et al. (2021); Lee et al. (2023). Concretely, we treat the EMA teacher as the source of temporally smoothed, semantically rich features. At each training iteration, we extract early block activations from both the student and the teacher and enforce semantic consistency via a hinge-based feature consistency loss as follows:

$$
\mathcal {L} _ {\text { cons. }} = \sum_ {\ell = 1} ^ {L _ {\text { early }}} \mathcal {W} _ {\text { cons. }} \cdot \left[ \tau - \frac {\langle \mathbf {g} _ {s} ^ {\ell} , \mathbf {g} _ {t} ^ {\ell} \rangle}{\| \mathbf {g} _ {s} ^ {\ell} \| _ {2} \| \mathbf {g} _ {t} ^ {\ell} \| _ {2}} \right] _ {+}, \tag {3}
$$

where $[ \cdot ] _ { + } : = \operatorname* { m a x } ( 0 , \cdot )$ and τ is the similarity threshold. This loss anchors the student’s edges and shape prior to the smoothed semantics of the teacher, ensuring that subsequent perturbations focus on object-centric regions. $\mathcal { W } _ { \mathrm { c o n s . } } \in \mathbb { R } ^ { | L | }$ denotes the softmax output of a learnable parameter for intermediate block-wise loss weighting. When combined with the adversarial objective, these semantically consistent perturbations that are highly transferable and tightly aligned with the core structure of the image. For fair comparisons with state-of-the-art methods, we adopt adversarial loss in the surrogate feature space as practiced in the baseline, e.g. BIA Zhang et al. (2022):

Algorithm 1: Pseudo-code of SCGA   
Data: Training dataset $D_{src}$ Input: Generator $\mathcal{G}_{\theta}(\cdot)$ , a surrogate model trained on source data $\mathcal{F}^{s}(\cdot)$ , projector $\mathcal{P}(\cdot)$ , perturbation budget $\epsilon$ Output: Optimized teacher perturbation generator $\mathcal{G}_{\theta'}(\cdot)$ 1 Initialize generators: student $\mathcal{G}_{\theta}(\cdot) \leftarrow$ random init., teacher $\mathcal{G}_{\theta'}(\cdot) \leftarrow \mathcal{G}_{\theta}(\cdot)$ 2 repeat

3 Randomly sample a mini-batch $x_{i}$ from $D_{src}$ 4 Acquire student generator intermediate features: $g_{i} \leftarrow \mathcal{G}_{\theta}^{\mathrm{enc}}(x_{i})$ 5 Acquire teacher generator intermediate features: $g_{i}' \leftarrow \mathcal{G}_{\theta'}^{\mathrm{enc}}(x_{i})$ 6 Generate unbounded adversarial examples from student generator intermediate features: $\tilde{x}_{i}^{\mathrm{adv}} \leftarrow \mathcal{G}_{\theta}^{\mathrm{dec}}(\mathbf{g}_{i})$ 7 Bound (project) $\tilde{x}_{i}^{adv}$ using P within the perturbation budget such that $||\mathcal{P}(\tilde{x}_{i}^{\mathrm{adv}}) - x_{i}||_{\infty} \leq \epsilon$ to obtain $x_{i}^{adv}$ 8 Forward pass $x_{i}$ and $x_{i}^{adv}$ through the surrogate model, $F^{s}(\cdot)$ at layer k, to acquire $f_{i}^{benign}, f_{i}^{adv}$ 9 Compute loss using $f_{i}^{benign}, f_{i}^{adv}, g_{i}, g_{i}'$ : $L = L_{adv} + \lambda_{cons.} \cdot L_{cons.}$ // Eq. 5

10 Update student generator parameters via backpropagation

11 EMA update teacher weights with student weights: $\theta \mapsto \theta'$ // Eq. 2

12 until $G_{\theta}(\cdot)$ converges

$$
\mathcal {L} _ {\mathrm{adv}} = \cos (\mathcal {F} _ {k} (x), \mathcal {F} _ {k} (x ^ {a d v})), \tag {4}
$$

where $\scriptstyle \mathtt { C O S } ( \cdot , \cdot )$ denotes cosine similarity.

Final loss objective. Putting the proposed and baseline losses together on the MT framework, we formulate the final loss objective with $\lambda _ { \mathrm { c o n s } }$ . as a weight term for ${ \mathcal { L } } _ { \mathrm { c o n s . } }$ , as follows:

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{adv}} + \lambda_ {\text { cons. }} \cdot \mathcal {L} _ {\text { cons. }}. \tag {5}
$$

# 4 EXPERIMENTS

We refer to Supp. §E.3 for training implementations and computational complexity. For evaluation (Supp. §E.1–2), we conduct cross-setting tests under two black-box protocols. In the cross-model setting, perturbations are crafted on surrogate models trained with the same data distribution (i.e., ImageNet-1K Russakovsky et al. (2015)) and then tested on unseen target model architectures. In the cross-domain/task settings, adversarial examples are to generalize across domain/task shifts without access to any target-distribution samples.

# 4.1 LIMITATIONS IN EXISTING EVALUATION PROTOCOL

Although developing an effective attack mechanism is crucial, it must be validated by fair and comprehensive evaluations. The current evaluation protocols adopted by previous works GAP Poursaeed et al. (2018), CDA Naseer et al. (2019), LTP Nakka & Salzmann (2021), BIA Zhang et al. (2022), GAMA Aich et al. (2022), FACL Yang et al. (2024a),

PDCL Yang et al. (2024b) exhibit three key limitations. (L1) Most studies report only one primary metric (either ASR, FR, or Acc.), offering only a one-dimensional view of attack robustness and neglecting other aspects such as unintended corrections in predictions. (L2) Data sets and sample sizes are often arbitrarily or limited to a single scale, preventing a fair comparison between attacks and undermining statistical significance. (L3) Evaluations in previous work commonly target a narrow set of victim architectures (e.g., mostly CNN-based), lacking the diversity of modern model families, including vision transformers (ViT) and state-space models (SSM), and thus overstating robustness. Although conventional work frames the success of attacks as fooling the target classifier, we contend that evaluation facets should be expanded for a reliable assessment of attacks.

Table 1: Examples of real-world impacts on predictions with different evaluation metrics and attack reliability concerns. 

<table><tr><td colspan="6">Real-world examples:</td></tr><tr><td>Scenario #</td><td>GT Label</td><td>Benign pred.</td><td>Adv. pred.</td><td>Impact</td><td>Captured by</td></tr><tr><td>1</td><td>cat</td><td>cat √</td><td>cat √</td><td>Correct → Correct</td><td>Acc. only</td></tr><tr><td>2</td><td>cat</td><td>cat √</td><td>dog ✗</td><td>Correct → Incorrect</td><td>ASR, FR</td></tr><tr><td>3</td><td>van</td><td>truck ✗</td><td>bus ✗</td><td>Incorrect → Other incorrect</td><td>FR only</td></tr><tr><td>4</td><td>pelagic cormorant</td><td>albatross ✗</td><td>pelagic cormorant √</td><td>Incorrect → Correct</td><td>ACR, FR. Acc.</td></tr><tr><td colspan="6">Reliable attack example:</td></tr><tr><td colspan="2">Cross-Setting</td><td>GT Label</td><td>Benign pred.</td><td>Intended Attack</td><td>Unreliable Attack</td></tr><tr><td colspan="2">ImageNet → FGVC Aircraft</td><td>F-22 Raptor</td><td>F-22 Raptor ✗</td><td>F-18 Hornet ✗</td><td>F-22 Raptor √</td></tr></table>

To address these shortcomings, we introduce, in §4.2 (L1), Accidental Correction Rate (ACR) as a complementary metric that captures the proportion of AEs that inadvertently restore correct predictions, enriching the evaluation of attack efficacy alongside conventional measures (i.e. ASR, FR, Acc.) as demonstrated with practical examples in Table 1. ACR measures a nuanced model behavior often missed by ASR and FR, which is crucial for a complete understanding of robustness in safety-critical systems where any unreliable response to perturbation may pose a risk. We also evaluate AEs on the entire validation set in §4.3 (L2, L3), instead of arbitrary subsets, and cover a wide range of victim models for the classification task. We provide further details in Supp. §E.

# 4.2 EVALUATION METRICS

We tested the effectiveness and transferability of adversarial attacks across model architectures and domain shifts using four key metrics. For notational convenience here, let f(x) denote the predicted label for input x, $f ( x + \delta )$ the prediction after applying adversarial perturbation δ, and y the groundtruth label. The evaluation set is indicated by D, with ${ \mathcal { C } } = \{ x \in { \mathcal { D } } \mid f ( x ) = y \}$ representing correctly classified samples, and ${ \mathcal { T } } = \{ x \in { \dot { \mathcal { D } } } \mid f ( x ) \neq y \}$ denoting misclassified samples under clean inference. We formally define our evaluation metrics (%) as follows:

$$
\begin{array}{l l} \text {Acc.} = | \{x \in \mathcal {D} \mid f (x + \delta) = y \} | / | \mathcal {D} |, & \text {ASR} = | \{x \in \mathcal {C} \mid f (x) = y \wedge f (x + \delta) \neq y \} | / | \mathcal {C} |, \\ \text {FR} = | \{x \in \mathcal {D} \mid f (x) \neq f (x + \delta) \} | / | \mathcal {D} |, & \text {ACR} = | \{x \in \mathcal {I} \mid f (x) \neq y \wedge f (x + \delta) = y \} | / | \mathcal {I} |, \end{array} \tag {6}
$$

where Top-1 Accuracy Zhang et al. (2022); Yang et al. (2024a;b) measures the overall proportion of correctly classified samples under clean or adversarial conditions. It serves as a global performance indicator to assess degraded performance after the attack, orthogonal to FR, ASR, and ACR. Attack Success Rate (ASR) Poursaeed et al. (2018); Naseer et al. (2019) is a subset of FR, which measures the proportion of samples originally correctly classified that are misclassified by adversarial attack. It directly reflects the targeted misclassification. Fooling Rate (FR) Nakka & Salzmann (2021); Nakka & Alahi (2025) quantifies the proportion of adversarial examples that cause a change in the model’s prediction, regardless of correctness. It reflects how often the attack disrupts the original decision and is used as a transferability measure. ACR, also a subset of FR, is a novel metric that quantifies how often misclassified samples are “accidentally” corrected by adversarial perturbations. This unintended side effect provides insight into the nuanced model uncertainty and behavior at the decision boundaries. For SS and OD, we use the standard mIoU and mAP50 metrics, respectively.

# 4.3 EXPERIMENTAL RESULTS

We demonstrate enhanced cross-model attacks in Table 2, wherein augmenting each baseline generative attack with our method yields consistent improvements across various architectures. Although these results confirm the orthogonality and efficacy of our framework, we observe that CLIP-based approaches with objectives similar to ours, e.g. PDCL Yang et al. (2024b), yield only marginal improvements when combined with our method. We conjecture that optimizing for divergence in CLIP’s high-dimensional semantic embedding space may override or dilute the local structural consistency enforced by our early block semantic consistency, thus attenuating incremental gains from preserving fine-grained object contours and textures (see Supp §D for detailed explanation).

Table 3 presents the black-box cross-domain transferability results. In both cross-domain and task, the transferability enhancements become more pronounced. Incorporating MT smoothing and early block consistency steadily enhances the attack performance across unseen domains, architectures, and tasks, demonstrating the broad applicability beyond the source data distribution and task.

With measurable gains in attack accuracy, we visually verify whether our method actually induces the generator to pay more attention to the object-salient regions in Fig. 3. Through Grad-CAM Selvaraju et al. (2017) comparisons against the baseline, ours either reinforces confusion or flips the correctly attending regions (similar to those of benign). Across unseen tasks, we also observe fewer pixels and instances with the correct classifications. We attribute this cross-task generalization to our labelagnostic training pipeline and further validate that our method can be integrated with alternative generator architectures beyond ResNet in Supp. §B.

Table 2: Quantitative cross-model transferability results. We report the improvements (∆ %p) of our method relative to each baseline, with better results marked in a darker color. ‘Avg.’ corresponds to black-box average. 

<table><tr><td colspan="2">Cross-model</td><td colspan="11">CNN</td><td colspan="6">Transformer</td><td colspan="2">Mixer</td><td colspan="2">Mamba</td><td rowspan="2">Avg.</td></tr><tr><td>Method</td><td>Metric</td><td>(a)</td><td>(b)</td><td>(c)</td><td>(d)</td><td>(e)</td><td>(f)</td><td>(g)</td><td>(h)</td><td>(i)</td><td>(j)</td><td>(k)</td><td>(l)</td><td>(m)</td><td>(n)</td><td>(o)</td><td>(p)</td><td>(q)</td><td>(r)</td><td>(s)</td><td>(t)</td><td>(v)</td></tr><tr><td>Benign</td><td>Acc. (%) ↓</td><td>74.60</td><td>77.33</td><td>74.22</td><td>75.74</td><td>76.19</td><td>77.95</td><td>66.50</td><td>55.91</td><td>79.12</td><td>81.49</td><td>75.42</td><td>80.67</td><td>79.28</td><td>81.19</td><td>80.48</td><td>79.10</td><td>57.91</td><td>69.90</td><td>66.53</td><td>66.53</td><td>73.21</td><td>73.77</td></tr><tr><td rowspan="4">CDA w/ Ours</td><td>Acc. (Δ%p) ↓</td><td>-15.93</td><td>-8.39</td><td>-12.93</td><td>-12.70</td><td>-8.41</td><td>-11.21</td><td>-5.09</td><td>-6.48</td><td>-10.17</td><td>-35.91</td><td>-19.74</td><td>-0.12</td><td>-0.14</td><td>+0.72</td><td>+0.03</td><td>+0.06</td><td>+0.09</td><td>+0.53</td><td>+0.73</td><td>+0.06</td><td>+0.29</td><td>-6.89</td></tr><tr><td>ASR (Δ%p) ↑</td><td>+20.13</td><td>+10.35</td><td>+16.52</td><td>+15.96</td><td>+10.37</td><td>+13.75</td><td>+6.92</td><td>+10.35</td><td>+12.19</td><td>+42.13</td><td>+24.37</td><td>+0.09</td><td>+0.10</td><td>-0.95</td><td>-0.05</td><td>-0.05</td><td>-0.23</td><td>-0.67</td><td>-1.04</td><td>-0.13</td><td>-0.51</td><td>+8.55</td></tr><tr><td>FR (Δ%p) ↑</td><td>+17.39</td><td>+9.29</td><td>+14.24</td><td>+13.80</td><td>+8.96</td><td>+11.91</td><td>+5.87</td><td>+7.86</td><td>+11.49</td><td>+38.92</td><td>+21.57</td><td>+0.09</td><td>+0.03</td><td>-0.99</td><td>-0.05</td><td>-0.15</td><td>-0.37</td><td>-0.74</td><td>-0.94</td><td>-0.20</td><td>-0.60</td><td>+7.49</td></tr><tr><td>ACR (Δ%p) ↓</td><td>-3.58</td><td>-1.72</td><td>-2.59</td><td>-2.52</td><td>-2.19</td><td>-2.25</td><td>-1.45</td><td>-1.58</td><td>-2.52</td><td>-7.04</td><td>-4.57</td><td>-0.24</td><td>-0.31</td><td>-0.28</td><td>-0.05</td><td>+0.12</td><td>-0.12</td><td>+0.20</td><td>+0.08</td><td>-0.11</td><td>-0.29</td><td>-1.57</td></tr><tr><td rowspan="4">LTP w/ Ours</td><td>Acc. (Δ%p) ↓</td><td>-8.71</td><td>-9.52</td><td>-8.45</td><td>-10.24</td><td>-4.62</td><td>-10.00</td><td>-5.59</td><td>-9.60</td><td>-9.57</td><td>-5.57</td><td>-5.93</td><td>-1.23</td><td>-1.77</td><td>-7.17</td><td>-1.74</td><td>-3.57</td><td>-5.86</td><td>-5.87</td><td>-9.05</td><td>-3.13</td><td>-5.93</td><td>-6.34</td></tr><tr><td>ASR (Δ%p) ↑</td><td>+11.11</td><td>+11.92</td><td>+10.87</td><td>+12.92</td><td>+5.90</td><td>+12.27</td><td>+8.04</td><td>+15.83</td><td>+11.89</td><td>+6.55</td><td>+7.50</td><td>+1.66</td><td>+2.37</td><td>+8.63</td><td>+2.38</td><td>+4.71</td><td>+9.70</td><td>+8.22</td><td>+13.03</td><td>+4.44</td><td>+7.97</td><td>+8.47</td></tr><tr><td>FR (Δ%p) ↑</td><td>+9.53</td><td>+10.58</td><td>+9.35</td><td>+11.30</td><td>+5.32</td><td>+10.70</td><td>+6.93</td><td>+12.25</td><td>+11.65</td><td>+6.15</td><td>+6.54</td><td>+2.19</td><td>+2.82</td><td>+8.71</td><td>+2.79</td><td>+5.10</td><td>+8.95</td><td>+7.63</td><td>+11.10</td><td>+3.87</td><td>+7.27</td><td>+7.65</td></tr><tr><td>ACR (Δ%p) ↓</td><td>-1.65</td><td>-1.31</td><td>-1.48</td><td>-1.85</td><td>-0.53</td><td>-2.01</td><td>-0.72</td><td>-1.71</td><td>-0.76</td><td>-1.01</td><td>-0.77</td><td>+0.52</td><td>+0.53</td><td>-0.79</td><td>+0.91</td><td>+0.84</td><td>-0.56</td><td>-0.44</td><td>-1.04</td><td>-0.53</td><td>-0.36</td><td>-0.70</td></tr><tr><td rowspan="4">BIA w/ Ours</td><td>Acc. (Δ%p) ↓</td><td>-2.23</td><td>-1.99</td><td>-1.29</td><td>+0.01</td><td>-3.72</td><td>-1.59</td><td>-3.29</td><td>-2.85</td><td>-0.39</td><td>+3.12</td><td>-0.63</td><td>-0.80</td><td>-0.32</td><td>-0.78</td><td>-0.56</td><td>-1.18</td><td>-1.08</td><td>-1.48</td><td>-0.45</td><td>-0.38</td><td>+0.03</td><td>-1.04</td></tr><tr><td>ASR (Δ%p) ↑</td><td>+2.83</td><td>+2.46</td><td>+1.64</td><td>-0.05</td><td>+4.55</td><td>+1.96</td><td>+4.68</td><td>+4.80</td><td>+0.56</td><td>-3.32</td><td>+1.21</td><td>+0.92</td><td>+0.33</td><td>+0.89</td><td>+0.59</td><td>+1.40</td><td>+1.91</td><td>+2.05</td><td>+0.88</td><td>+0.56</td><td>+0.03</td><td>+1.47</td></tr><tr><td>FR (Δ%p) ↑</td><td>+2.57</td><td>+2.28</td><td>+1.48</td><td>-0.06</td><td>+4.20</td><td>+1.73</td><td>+3.78</td><td>+3.69</td><td>+0.56</td><td>-3.10</td><td>+0.69</td><td>+1.01</td><td>+0.44</td><td>+1.00</td><td>+0.57</td><td>+1.44</td><td>+1.62</td><td>+1.82</td><td>+0.81</td><td>+0.35</td><td>+0.04</td><td>+1.28</td></tr><tr><td>ACR (Δ%p) ↓</td><td>-0.45</td><td>-0.36</td><td>-0.28</td><td>-0.09</td><td>-1.09</td><td>-0.27</td><td>-0.53</td><td>-0.38</td><td>+0.24</td><td>+0.33</td><td>-0.37</td><td>-0.33</td><td>-0.27</td><td>-0.29</td><td>-0.45</td><td>-0.30</td><td>+0.06</td><td>-0.12</td><td>+0.40</td><td>-0.03</td><td>+0.19</td><td>-0.21</td></tr><tr><td rowspan="4">GAMA w/ Ours</td><td>Acc. (Δ%p) ↓</td><td>-2.54</td><td>-2.46</td><td>-2.65</td><td>-2.15</td><td>-2.49</td><td>-2.19</td><td>-0.24</td><td>-0.17</td><td>-0.97</td><td>-2.49</td><td>-2.94</td><td>+0.07</td><td>+0.03</td><td>-0.24</td><td>-0.01</td><td>-0.15</td><td>-0.51</td><td>+0.07</td><td>-0.48</td><td>-0.59</td><td>-0.41</td><td>-1.12</td></tr><tr><td>ASR (Δ%p) ↑</td><td>+3.22</td><td>+3.14</td><td>+3.40</td><td>+2.82</td><td>+3.14</td><td>+2.73</td><td>+0.34</td><td>+0.30</td><td>+1.22</td><td>+2.92</td><td>+3.67</td><td>-0.04</td><td>-0.05</td><td>+0.30</td><td>+0.13</td><td>+0.12</td><td>+0.89</td><td>-0.17</td><td>+0.83</td><td>+0.75</td><td>+0.52</td><td>+1.44</td></tr><tr><td>FR (Δ%p) ↑</td><td>+2.81</td><td>+2.87</td><td>+2.91</td><td>+2.56</td><td>+2.76</td><td>+2.53</td><td>+0.24</td><td>+0.21</td><td>+1.20</td><td>+2.67</td><td>+3.23</td><td>+0.05</td><td>+0.03</td><td>+0.27</td><td>+0.20</td><td>+0.05</td><td>+0.73</td><td>-0.21</td><td>+0.73</td><td>+0.59</td><td>+0.44</td><td>+1.28</td></tr><tr><td>ACR (Δ%p) ↓</td><td>-0.58</td><td>-0.14</td><td>-0.51</td><td>-0.08</td><td>-0.43</td><td>-0.31</td><td>-0.03</td><td>0.00</td><td>-0.02</td><td>-0.49</td><td>-0.56</td><td>+0.21</td><td>-0.01</td><td>-0.02</td><td>+0.53</td><td>-0.27</td><td>+0.02</td><td>-0.16</td><td>+0.20</td><td>-0.29</td><td>-0.09</td><td>-0.14</td></tr><tr><td rowspan="4">FACL w/ Ours</td><td>Acc. (Δ%p) ↓</td><td>+0.10</td><td>-0.59</td><td>-3.35</td><td>-1.97</td><td>-4.92</td><td>-0.60</td><td>-3.29</td><td>-0.69</td><td>-2.01</td><td>-1.91</td><td>-2.64</td><td>+0.11</td><td>-0.33</td><td>+0.21</td><td>-0.51</td><td>+0.56</td><td>-0.18</td><td>-0.50</td><td>+0.45</td><td>-0.30</td><td>-0.17</td><td>-1.07</td></tr><tr><td>ASR (Δ%p) ↑</td><td>-0.20</td><td>+0.74</td><td>+4.30</td><td>+2.46</td><td>+6.15</td><td>+0.75</td><td>+4.68</td><td>+1.25</td><td>+2.40</td><td>+2.23</td><td>+3.15</td><td>-0.10</td><td>+0.41</td><td>-0.24</td><td>+0.53</td><td>-0.69</td><td>+0.14</td><td>+0.68</td><td>-0.72</td><td>+0.34</td><td>+0.15</td><td>+1.35</td></tr><tr><td>FR (Δ%p) ↑</td><td>-0.20</td><td>+0.64</td><td>+3.75</td><td>+2.27</td><td>+5.37</td><td>+0.74</td><td>+3.97</td><td>+0.96</td><td>+2.24</td><td>+2.05</td><td>+2.78</td><td>-0.02</td><td>+0.47</td><td>-0.19</td><td>+0.47</td><td>-0.67</td><td>+0.08</td><td>+0.72</td><td>-0.64</td><td>+0.25</td><td>+0.14</td><td>+1.20</td></tr><tr><td>ACR (Δ%p) ↓</td><td>-0.23</td><td>-0.09</td><td>-0.61</td><td>-0.46</td><td>-0.97</td><td>-0.08</td><td>-0.54</td><td>0.00</td><td>-0.52</td><td>-0.41</td><td>-0.96</td><td>+0.16</td><td>-0.02</td><td>+0.05</td><td>-0.46</td><td>+0.09</td><td>-0.23</td><td>-0.09</td><td>-0.09</td><td>-0.24</td><td>-0.24</td><td>-0.28</td></tr><tr><td rowspan="4">PDCL w/ Ours</td><td>Acc. (Δ%p) ↓</td><td>+0.55</td><td>-0.29</td><td>+1.01</td><td>-0.40</td><td>-0.31</td><td>-0.98</td><td>-1.13</td><td>-0.06</td><td>-1.09</td><td>-0.72</td><td>+0.79</td><td>-0.07</td><td>+0.06</td><td>-0.11</td><td>+0.18</td><td>-0.14</td><td>+0.06</td><td>+0.52</td><td>+0.09</td><td>+0.37</td><td>+0.11</td><td>-0.07</td></tr><tr><td>ASR (Δ%p) ↑</td><td>-0.73</td><td>+0.31</td><td>-1.26</td><td>+0.56</td><td>+0.46</td><td>+1.19</td><td>+1.64</td><td>+0.07</td><td>+1.30</td><td>+0.83</td><td>-0.96</td><td>+0.08</td><td>-0.14</td><td>+0.04</td><td>-0.19</td><td>+0.05</td><td>-0.12</td><td>-0.64</td><td>-0.06</td><td>-0.43</td><td>-0.09</td><td>+0.09</td></tr><tr><td>FR (Δ%p) ↑</td><td>-0.68</td><td>+0.27</td><td>-1.08</td><td>+0.36</td><td>+0.45</td><td>+1.09</td><td>+1.42</td><td>+0.13</td><td>+1.23</td><td>+0.81</td><td>-0.88</td><td>+0.22</td><td>-0.10</td><td>+0.09</td><td>-0.03</td><td>+0.09</td><td>-0.22</td><td>-0.44</td><td>+0.05</td><td>-0.33</td><td>-0.10</td><td>+0.11</td></tr><tr><td>ACR (Δ%p) ↓</td><td>+0.03</td><td>-0.18</td><td>+0.29</td><td>+0.09</td><td>+0.18</td><td>-0.22</td><td>-0.12</td><td>-0.05</td><td>-0.33</td><td>-0.22</td><td>+0.21</td><td>-0.06</td><td>-0.23</td><td>-0.42</td><td>+0.08</td><td>-0.47</td><td>-0.04</td><td>+0.22</td><td>+0.15</td><td>+0.27</td><td>+0.16</td><td>-0.03</td></tr></table>

Table 3: Quantitative cross-domain/task transferability results. We report the average improvement (∆ %p) with ours added from each baseline for each domain. Better results in green boldface. 

<table><tr><td rowspan="3">Method\Metric</td><td colspan="13">Cross-domain</td><td colspan="6">Cross-task</td></tr><tr><td colspan="4">CUB-200-2011</td><td colspan="4">Stanford Cars</td><td colspan="4">FGVC Aircraft</td><td rowspan="2">Avg.Acc.</td><td colspan="2">SemSeg (SS)</td><td rowspan="2">Avg.mIoU ↓</td><td colspan="2">ObjDet (OD)</td><td rowspan="2">Avg.mAP50 ↓</td></tr><tr><td>Acc ↓</td><td>ASR ↑</td><td>FR ↑</td><td>ACR ↓</td><td>Acc ↓</td><td>ASR ↑</td><td>FR ↑</td><td>ACR ↓</td><td>Acc ↓</td><td>ASR ↑</td><td>FR ↑</td><td>ACR ↓</td><td>DeepLabV3+</td><td>SegFormer</td><td>Faster R-CNN</td><td>DETR</td></tr><tr><td>Benign</td><td>86.91</td><td>N/A</td><td>N/A</td><td>N/A</td><td>93.56</td><td>N/A</td><td>N/A</td><td>N/A</td><td>92.07</td><td>N/A</td><td>N/A</td><td>N/A</td><td>90.85</td><td>76.21</td><td>71.89</td><td>74.05</td><td>61.01</td><td>62.36</td><td>61.69</td></tr><tr><td>CDA</td><td>67.73</td><td>21.48</td><td>14.16</td><td>26.66</td><td>77.68</td><td>21.88</td><td>15.38</td><td>24.07</td><td>64.42</td><td>27.51</td><td>14.55</td><td>31.13</td><td>69.94</td><td>25.63</td><td>20.16</td><td>22.90</td><td>32.78</td><td>26.29</td><td>29.54</td></tr><tr><td>w/ Ours ( $\Delta\%p$ )</td><td>-16.92</td><td>+21.48</td><td>+20.63</td><td>-3.94</td><td>-5.86</td><td>+2.38</td><td>+2.35</td><td>-0.24</td><td>-22.58</td><td>+27.74</td><td>+26.44</td><td>-6.00</td><td>-15.12</td><td>-0.47</td><td>+0.10</td><td>-0.18</td><td>-0.80</td><td>-0.61</td><td>-0.71</td></tr><tr><td>LTP</td><td>48.74</td><td>45.32</td><td>8.75</td><td>49.31</td><td>57.98</td><td>39.02</td><td>13.03</td><td>40.85</td><td>43.01</td><td>54.15</td><td>8.35</td><td>56.48</td><td>49.91</td><td>23.71</td><td>26.97</td><td>25.34</td><td>29.39</td><td>22.41</td><td>25.90</td></tr><tr><td>w/ Ours ( $\Delta\%p$ )</td><td>-10.43</td><td>+11.72</td><td>+10.87</td><td>-0.90</td><td>-10.62</td><td>+10.98</td><td>+10.68</td><td>-2.16</td><td>-6.41</td><td>+6.66</td><td>+6.35</td><td>-1.41</td><td>-9.15</td><td>-1.44</td><td>-0.29</td><td>-0.86</td><td>-2.54</td><td>-0.23</td><td>-1.39</td></tr><tr><td>BIA</td><td>47.92</td><td>46.13</td><td>8.54</td><td>50.26</td><td>59.89</td><td>37.22</td><td>13.96</td><td>38.97</td><td>45.38</td><td>51.52</td><td>9.24</td><td>54.06</td><td>51.07</td><td>23.89</td><td>25.60</td><td>24.75</td><td>28.43</td><td>21.01</td><td>24.72</td></tr><tr><td>w/ Ours ( $\Delta\%p$ )</td><td>-0.02</td><td>+0.08</td><td>+0.09</td><td>+0.49</td><td>-6.89</td><td>+6.89</td><td>+6.70</td><td>-2.00</td><td>-4.98</td><td>+5.25</td><td>+4.89</td><td>-1.24</td><td>-3.96</td><td>-1.84</td><td>-0.85</td><td>-1.35</td><td>-0.09</td><td>-0.29</td><td>-0.20</td></tr><tr><td>GAMA</td><td>48.72</td><td>45.41</td><td>9.51</td><td>49.67</td><td>54.59</td><td>42.58</td><td>11.94</td><td>44.28</td><td>42.37</td><td>54.46</td><td>7.49</td><td>56.77</td><td>48.56</td><td>23.67</td><td>25.95</td><td>24.81</td><td>28.01</td><td>20.71</td><td>24.36</td></tr><tr><td>w/ Ours ( $\Delta\%p$ )</td><td>-2.41</td><td>+2.59</td><td>+2.30</td><td>-0.67</td><td>-2.33</td><td>+2.24</td><td>+2.10</td><td>-0.60</td><td>-2.68</td><td>+3.06</td><td>+2.87</td><td>+0.14</td><td>-2.47</td><td>-0.43</td><td>-1.58</td><td>-1.01</td><td>-0.41</td><td>+0.08</td><td>-0.16</td></tr><tr><td>FACL</td><td>40.85</td><td>54.36</td><td>7.21</td><td>58.01</td><td>51.23</td><td>48.23</td><td>12.9</td><td>49.71</td><td>40.08</td><td>59.35</td><td>7.34</td><td>61.39</td><td>44.05</td><td>23.75</td><td>26.40</td><td>25.08</td><td>27.94</td><td>20.91</td><td>24.43</td></tr><tr><td>w/ Ours ( $\Delta\%p$ )</td><td>+3.12</td><td>-3.79</td><td>-3.58</td><td>+0.70</td><td>-7.26</td><td>+2.34</td><td>+4.72</td><td>-4.49</td><td>-2.68</td><td>+0.60</td><td>+0.66</td><td>-0.24</td><td>-2.27</td><td>-0.37</td><td>-1.39</td><td>-0.88</td><td>-0.30</td><td>-0.62</td><td>-0.46</td></tr><tr><td>PDCL</td><td>42.36</td><td>52.32</td><td>7.48</td><td>55.93</td><td>50.41</td><td>46.85</td><td>12.31</td><td>48.46</td><td>38.96</td><td>58.23</td><td>6.86</td><td>60.34</td><td>43.91</td><td>24.42</td><td>26.05</td><td>25.24</td><td>28.48</td><td>21.38</td><td>24.93</td></tr><tr><td>w/ Ours ( $\Delta\%p$ )</td><td>-0.46</td><td>+0.61</td><td>+0.66</td><td>+0.40</td><td>-0.71</td><td>+0.75</td><td>+0.69</td><td>-0.32</td><td>-1.38</td><td>+1.52</td><td>+1.42</td><td>+0.14</td><td>-0.85</td><td>-1.91</td><td>-0.17</td><td>-1.04</td><td>-0.82</td><td>-0.65</td><td>-0.73</td></tr></table>

Against robust training (i.e. adversarially trained IncV3 Kurakin et al. (2018),ViT Dosovitskiy et al. (2021), ConvNeXt Singh et al. (2023), and input pre-processing JPEG Guo et al. (2017) BDR Xu et al. (2018), R&P Xie et al. (2018)) techniques, our methods demonstrate superior attacks compared to the baseline as shown in Table 4, reinforcing our hypothesis that enforcing semantic consistency in early generator blocks not only boosts transferability in standard blackbox settings but also produces perturbations capable of further enhancing attacks against defense mechanisms. By anchoring structural cues in the early stages, our self-feature consistency loss yields more potent and robust attacks against adversarially trained models and input pre-processing defenses alike.

Table 4: Superior attack success with our method against robustly trained models including Adversarially trained (AT) models and robust input pre-processing methods. Better averaged results in green boldface. 

<table><tr><td>Method</td><td>Metric</td><td>Adv.IncV3</td><td>Adv.ViT</td><td>Adv.ConvNeXt</td><td>JPEG</td><td>BDR</td><td>R&amp;P</td><td>Avg.</td></tr><tr><td>Benign</td><td>Acc. (%) ↓</td><td>76.33</td><td>48.82</td><td>58.44</td><td>74.68</td><td>74.68</td><td>76.58</td><td>68.26</td></tr><tr><td rowspan="4">Baseline Zhang et al. (2022)</td><td>Acc. (%) ↓</td><td>68.54</td><td>45.64</td><td>53.88</td><td>63.49</td><td>47.82</td><td>44.78</td><td>54.03</td></tr><tr><td>ASR (%) ↑</td><td>14.95</td><td>11.72</td><td>10.26</td><td>20.24</td><td>40.76</td><td>44.59</td><td>23.75</td></tr><tr><td>FR (%) ↑</td><td>24.02</td><td>25.48</td><td>19.40</td><td>28.09</td><td>48.06</td><td>51.60</td><td>32.78</td></tr><tr><td>ACR (%) ↓</td><td>15.30</td><td>4.96</td><td>3.46</td><td>11.45</td><td>11.30</td><td>10.56</td><td>9.51</td></tr><tr><td rowspan="4">w/ Ours</td><td>Acc. (%) ↓</td><td>67.92</td><td>45.33</td><td>53.62</td><td>60.83</td><td>44.07</td><td>39.01</td><td>51.80</td></tr><tr><td>ASR (%) ↑</td><td>15.75</td><td>11.95</td><td>10.65</td><td>23.74</td><td>45.37</td><td>51.63</td><td>26.52</td></tr><tr><td>FR (%) ↑</td><td>24.83</td><td>25.31</td><td>19.60</td><td>31.61</td><td>52.22</td><td>57.86</td><td>35.28</td></tr><tr><td>ACR (%) ↓</td><td>15.23</td><td>4.57</td><td>3.38</td><td>11.48</td><td>10.29</td><td>9.08</td><td>9.01</td></tr></table>

![](images/5ef2dfb7d9721833fd205682eb1bfee1b40309217debaa8f2ae996a216e803d4.jpg)

<details>
<summary>text_image</summary>

(a)
Black-footed
Albatross
Ferrari 458 Italia
Convertible 2012
737-200
Chest
Carpenter's kit
Shopping Basket
(b)
(c)
(d)
Pelagic
Cormorant
Ferrari FF
Coupe 2012
A319
Wallet
Safety Pin
Wool
</details>

![](images/d5e16b5bfed77f2e6a66313b70e7870e16080d0b82ceb62c1510797ffda00727.jpg)

<details>
<summary>text_image</summary>

②
Input
Benign
Baseline
Ours
③
Baseline
Ours
GT
</details>

Figure 3: Qualitative results. Our semantically consistent generative attack successfully guides the generator to focus perturbations particularly on the semantically salient regions, effectively fooling the victim classifier. ⃝1 : (a) benign input image, (b) generated perturbation (normalized for visual purposes only), (c) unbounded adversarial image, and (d) bounded adversarial image across CUB-200-2011 Wah et al. (2011), Stanford Cars Krause et al. (2013), FGVC Aircraft Maji et al. (2013), and ImageNet-1K Russakovsky et al. (2015). The label on top (green) and bottom (orange) denotes the correct label and prediction after the attack, respectively. ⃝2 : We highlight that our method induces Grad-CAM Selvaraju et al. (2017) to focus on drastically different regions in our adversarial examples compared to both the benign image and the adversarial examples crafted by the baseline Zhang et al. (2022). Moreover, our approach noticeably spreads and reduces the high activation regions observed in the benign and baseline cases, enhancing the transferability of our adversarial perturbations. ⃝3 : Cross-task prediction results (SS on top, OD on bottom). Our approach further disrupts the victim models by triggering higher false positive rates and wrong class label predictions. See Supp. §E.4 for additional visualizations.

Interplay with the baselines. The pattern of gains across baselines in Table 3 is largely determined by the level at which each method probes the surrogate (logits, frequency domain, or intermediate features). By enforcing early-block semantic anchoring, our generator produces locally structured, object-aware perturbations. These perturbations move energy away from degenerate high-frequency noise and toward low- and mid-frequency components that align with objects and boundaries. This structural regularization couples most strongly with CNN-centric objectives. When combined with CDA, whose relativistic loss is defined directly on CNN logits, and with frequency- or CNN-priorbased baselines such as FACL and PDCL, our semantics-enhanced perturbations yield the largest improvements on CNN victims. ViT victims, whose global attention patterns and feature geometry differ more from the CNN surrogate, tend to show smaller or more localized changes.

In contrast, mid-layer feature-based attacks such as LTP, BIA, and GAMA rely on intermediate surrogate features that transfer more readily across architectures. These methods benefit more uniformly. Our generator-side semantics act as a complementary regularizer that sharpens feature-space separability on both CNN and ViT targets, with broadly positive or neutral effects. On image classification, the additional gains when combining with PDCL are modest. This behavior is consistent with a saturation regime in which the strong CLIPspace objective already induces powerful global semantic shifts and dominates the joint gradient. Even in this setting, our anchor still rebalances the perturbation spectrum. For localization-oriented downstream tasks such as detection and segmentation, the same local structural consistency produces noticeably larger cross-task improvements. This behavior suggests that Ours refines the global CLIP-driven semantic direction rather than competing with it.

Table 5: Ablation study on the targeted generator intermediate block and the proposed components. Our self-feature consistency strategy on the early intermediate block outperforms matching other block features (a), and the generator trained with all of our components together performs best (b). 

<table><tr><td rowspan="4">Cross-Task</td><td rowspan="4">Metric</td><td rowspan="4">Block</td><td rowspan="4">Early</td><td rowspan="4">Mid</td><td rowspan="4">Late</td><td rowspan="4">All</td><td> $\mathcal{L}_{\text{adv}}$ </td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>PT</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td></tr><tr><td>MT</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td> $\mathcal{L}_{\text{cons.}}$ </td><td>√</td><td>×</td><td>√</td><td>×</td><td>×</td></tr><tr><td rowspan="4">Model</td><td>Acc. (%) ↓</td><td></td><td>44.13</td><td>45.76</td><td>45.79</td><td>51.13</td><td></td><td>44.13</td><td>48.23</td><td>45.11</td><td>46.49</td><td>45.17</td></tr><tr><td>ASR (%) ↑</td><td></td><td>44.02</td><td>41.85</td><td>41.87</td><td>41.67</td><td></td><td>44.02</td><td>33.37</td><td>42.80</td><td>41.02</td><td>42.55</td></tr><tr><td>FR (%) ↑</td><td></td><td>50.66</td><td>48.71</td><td>48.75</td><td>48.57</td><td></td><td>50.66</td><td>44.08</td><td>49.47</td><td>46.99</td><td>49.38</td></tr><tr><td>ACR (%) ↓</td><td></td><td>8.32</td><td>8.59</td><td>8.66</td><td>8.60</td><td></td><td>8.32</td><td>8.71</td><td>8.43</td><td>8.68</td><td>8.53</td></tr><tr><td rowspan="4">Domain</td><td>Acc. (%) ↓</td><td></td><td>47.10</td><td>50.95</td><td>49.03</td><td>51.13</td><td></td><td>47.10</td><td>48.46</td><td>49.57</td><td>51.63</td><td>51.07</td></tr><tr><td>ASR (%) ↑</td><td>(a)</td><td>49.02</td><td>44.91</td><td>47.02</td><td>44.72</td><td>(b)</td><td>49.02</td><td>47.60</td><td>46.35</td><td>44.17</td><td>44.96</td></tr><tr><td>FR (%) ↑</td><td></td><td>51.66</td><td>47.67</td><td>49.75</td><td>47.50</td><td></td><td>51.66</td><td>50.30</td><td>49.10</td><td>47.02</td><td>47.76</td></tr><tr><td>ACR (%) ↓</td><td></td><td>9.66</td><td>10.36</td><td>10.57</td><td>10.36</td><td></td><td>9.66</td><td>9.99</td><td>9.89</td><td>10.73</td><td>10.58</td></tr><tr><td rowspan="2">Task</td><td>SS</td><td>mIoU ↓</td><td>23.40</td><td>24.10</td><td>22.82</td><td>23.92</td><td></td><td>23.40</td><td>23.96</td><td>24.83</td><td>23.73</td><td>24.75</td></tr><tr><td>OD</td><td>mAP50 ↓</td><td>24.52</td><td>24.53</td><td>24.69</td><td>24.52</td><td></td><td>24.52</td><td>24.73</td><td>24.55</td><td>24.41</td><td>24.72</td></tr></table>

Ablation studies. We conducted ablation studies on the intermediate block and our proposed components in Table 5. Across all cross-settings, we observe the highest gains with self-feature consistency applied to the early block compared to those at other and all locations, insinuating the early block matching triggers generator features to place stricter constraint such that perturbations are progressively focused on or around the object.

We also observe performance gains with each component: $\mathcal { L } _ { \mathrm { a d v } } .$ , MT, and ${ \mathcal { L } } _ { \mathrm { c o n s . } }$ ., wherein our consistency of self-features on the intermediate features of the generator serves to widen the transferability gap even further. We attribute this improvement to explicit semantic alignment in the early blocks which complements the effect of implicit smoothing with MT. We also compare against the plain student-copy teacher, as indicated by plain teacher (PT), with and without $\mathcal { L } _ { \mathrm { c o n s . } } ,$ which underperforms our MT configuration. These results validate our hypothesis that anchoring perturbation synthesis on the early intermediate blocks consistently preserves the object semantics the most, and thus guides later blocks to concentrate noise on object-centric regions, maximizing transferability.

# 4.4 GENERATOR INTERMEDIATE BLOCK-LEVEL ANALYSIS

![](images/0a0a8cbbbc19a578be3d74eafc95e0c55089568ec1f6e6a6db585a8e15b1a2de.jpg)  
Baseline vs. w/ Ours

![](images/9d0dcfffa4699b2554e8fa2218da501871e0a3b6c1ea2e604e720d37fc369669.jpg)

![](images/f88c62cbdd65837449444f4f869112f6a1e0026da9590c52afefa32b4b0f3cb9.jpg)

![](images/2780e09bbb4209e0e9d78998031e54410af2e6cf6244c09632238920238a018a.jpg)

![](images/fcd50e465a48e892de7185b5e807e886491a801d3cea6c0fe3e1fa71059b27a3.jpg)

![](images/e91bfa8f1e859a7442d654335179938f358df382547e4286d1604c4f293a021d.jpg)

![](images/aae42d18de48e415c1ab03805c9bb116f7088f8f239fd01c6feb27c1e187bb47.jpg)

![](images/58ffc38bb48dd6e715807895ffea35a5fa0a15e5f400b6933e9e3fd55f2e37bf.jpg)

![](images/8805357ab0acac24a895e3cc116a2144ecccd8b7279748dfe4bc47cbbac218fb.jpg)

![](images/0c825bf71655e1a5b627014a72d31a81256886b748f3c8fc81e5389484ca70d7.jpg)

![](images/c420f43195e004733062562268a165c4d658609f7cf79221b7537e89a4f1a770.jpg)

![](images/8b2da657059d43a7021e4e974ae5ecc9eaa82225496df853bfd34949cc6970c2.jpg)

![](images/f4bb4a3d99cc66a997193e6662ede1a4beb172f4ca92cadeb0260f00db0e5d0e.jpg)

![](images/8f50b70c15cdc34c6998649f3af013eeba0fb4a0a46cbf6bf00f543dcff4c96e.jpg)

![](images/074692159b42a9d6839b3a115157828318b469782791cb20662cbfb8813a8370.jpg)

![](images/6d2af6bb16fb4d7589c1bd676808d6020a35ae6bf0677ff0824b7949f55b9756.jpg)

![](images/66ed85f10e8eb6c54cb4cb6c37cce72efb3d56a2cf30b5b02c9349c8cd55ccd5.jpg)

![](images/d1e15ff2f42a2ff54600c17d04d3c47f9729daaaf8108ebaf7a997fc63d7b5cf.jpg)

![](images/0573e705750c96172dc87cd8fcbd549db9cdc984862373ba955c806c691fdef4.jpg)

![](images/0eac0f0e6dd2bb3a7195dc9f1b8053aaddf667afa98fb7a68b998da50a58c4a3.jpg)

![](images/4db0e70ea93414ec76c943c31f8c435dcad3105b896bc52ec73a817f73a3d6a3.jpg)

![](images/87e95f86655950b974c8c455857cf861942e8fffa8ca6bdfdb80861ae683fda9.jpg)

![](images/67449c4009af02becc598d583d6bb7432d3ea3ff7703aed7db77dc309a2a36f5.jpg)

![](images/c6a34e5ad9fa5ccb0791f4160c9e89e61959907d71c2c280d16ca5f7f1170a16.jpg)

![](images/4070f380eda8a8f81bd1024a7b1ede989c93f1bac67c9123ab3d7caccf9c9c21.jpg)  
Figure 4: Visualization of generator intermediate block-level differences with the baseline Zhang et al. (2022): raw feature differences on bottom, and thresholded on top (normalized for illustration purposes only). With our generator-internal semantic consistency mechanism, we progressively guide adversarial perturbation to focus on the salient object regions initially and gradually disperse to surrounding background regions. See Supp. Fig.S10 for other baseline comparisons.

Feature difference. Following Zhang et al. (2022); Yang et al. (2024a) but generalizing the procedure to all generator layers as follows, for each layer l:

$$
\mathsf {D i f f} (\mathbf {g} _ {\mathrm{baseline}} ^ {l, p o o l e d}, \mathbf {g} _ {\mathrm{ours}} ^ {l, p o o l e d}) = \left\{ \begin{array}{l l} 1, & \mathbf {g} _ {\mathrm{ours}} ^ {l, p o o l e d} - \mathbf {g} _ {\mathrm{baseline}} ^ {l, p o o l e d} > \tau_ {\mathrm{diff}}, \\ 0, & e l s e, \end{array} \right.
$$

$$
\text { with } \quad \mathbf {g} _ {(\cdot)} ^ {l, p o o l e d} = \frac {1}{C} \left| \sum_ {C} G _ {\theta_ {(\cdot)}} ^ {l} (\mathbf {x}) \right|,
$$

we visualized generator block-wise feature difference maps between each baseline with and without our method. At each block, we computed the difference by applying cross-channel average pooling to the activation tensor and then thresholding the resulting map to qualitatively emphasize the added perturbations. As shown in Fig. 4, the thresholded mask (row 1) and the raw feature difference map (row 2) jointly illustrate that, specifically within the targeted resblocks layers in our design, the adversarial signal concentrates on object-salient regions extracted by the preceding downsampling stages. Gradually into the later blocks, the generator learns to craft perturbation not only on object-salient regions but also regions closer to the background, generating more transferable noise. Compared to each baseline alone, our approach more strongly induces perturbations to better align with the semantic characteristics primarily in the intermediate residual blocks.

Spectral energy comparisons. To validate the early-block semantic anchoring hypothesis, we conducted a frequency-domain energy analysis of intermediate feature activations in Table 6, exploiting the link between spectral content and visual structure: low-frequency (LF) components encode coarse shapes and layouts, whereas high frequencies (HF) capture fine texture. By tracking the normalized low-band energy in every block before and after our method, we obtained a quantitative measure of how strongly each block preserves the coarse structure. Anchoring on the early blocks, rather than mid or late, consistently raises low-frequency energy and suppresses superfluous

high-frequency noise downstream, confirming that our method targeting semantic consistency in the early intermediates more effectively propagates the same semantic scaffold through later blocks, yielding higher adversarial transferability.

Table 6: Spectral energy by band (Baseline→w/ Ours). 

<table><tr><td></td><td>Band</td><td>Early</td><td>Mid</td><td>Late</td></tr><tr><td rowspan="2">CDA→w/ Ours</td><td>Low (↑)</td><td>0.82→0.91</td><td>0.75→0.97</td><td>0.77→0.96</td></tr><tr><td>High (↓)</td><td>0.18→0.09</td><td>0.25→0.03</td><td>0.23→0.04</td></tr><tr><td rowspan="2">LTP→w/ Ours</td><td>Low (↑)</td><td>0.73→0.72</td><td>0.78→0.79</td><td>0.95→0.75</td></tr><tr><td>High (↓)</td><td>0.27→0.28</td><td>0.22→0.21</td><td>0.05→0.25</td></tr><tr><td rowspan="2">BIA→w/ Ours</td><td>Low (↑)</td><td>0.56→0.56</td><td>0.53→0.54</td><td>0.53→0.58</td></tr><tr><td>High (↓)</td><td>0.44→0.44</td><td>0.47→0.45</td><td>0.47→0.42</td></tr><tr><td rowspan="2">GAMA→w/ Ours</td><td>Low (↑)</td><td>0.57→0.79</td><td>0.54→0.60</td><td>0.56→0.59</td></tr><tr><td>High (↓)</td><td>0.43→0.21</td><td>0.46→0.40</td><td>0.44→0.41</td></tr><tr><td rowspan="2">FACL→w/ Ours</td><td>Low (↑)</td><td>0.57→0.73</td><td>0.52→0.61</td><td>0.54→0.59</td></tr><tr><td>High (↓)</td><td>0.43→0.27</td><td>0.48→0.39</td><td>0.46→0.45</td></tr><tr><td rowspan="2">PDCL→w/ Ours</td><td>Low (↑)</td><td>0.54→0.62</td><td>0.51→0.59</td><td>0.58→0.59</td></tr><tr><td>High (↓)</td><td>0.46→0.38</td><td>0.49→0.41</td><td>0.42→0.41</td></tr></table>

The pattern reveals how anchoring affects generator’s frequency bias. For band-wise relatively balanced models such as GAMA, the early-block anchor sharply increases low-frequency energy $( 0 . 5 7  0 . 7 9$ ↑), giving later blocks a clearer structural blueprint. When a baseline already overemphasizes low frequencies, as in LTP whose late-block LF reaches 0.95, our method lowers that value to 0.75 ↓, restoring HF detail. This spectral analysis thus reveals that anchoring on the early intermediate features results in perturbations that remain coarse semantic structures aligned and intact within the generator, thereby enhancing transfer effectively across unseen domains and architectures.

Hyperparameter sensitivity. We vary the EMA coefficient (η) and the consistency weight $\lambda _ { \mathrm { c o n s } }$ . and report the cross-setting transfer performance in Table 7. We observe a trade-off between optimizing classification and cross-task scores for both hyperparameters, as no single combination uniformly outperforms the rest. However, maintaining relatively high values for both tends to yield better performance, indicating that each module sufficiently contributes to the overall self-consistency mechanism. Based on this observation, we select $\lambda _ { \mathrm { c o n s . } } = 0 . 7$ and $\eta = 0 . 9 9 9$ as our default configuration, which provides the best overall balance across all cross-setting scenarios.

We define the “early”, “mid”, and “late” stages of the generator intermediates by grouping two consecutive residual blocks based on the observation that perturbations undergo the most noticeable qualitative changes over every two blocks. As illustrated in Fig. 2, the first two blocks (rows (a)–(b)) still closely track the benign image: coarse object shape, foreground–background separation, and large-scale texture are clearly preserved. The next two blocks (rows (c)–(d)) begin to introduce more pronounced distortions and fine-grained variations, while the final two blocks (rows (e)–(f)) predominantly add high-frequency details and noise-like patterns that are no longer easily interpretable as object-level structure. This makes the first two blocks a natural choice for enforcing semantic consistency: they are structurally well-formed and dominantly encode benign scene semantics, before most of the perturbation mass emerges in later stages.

Applying the temporal self-consistency loss only to block 1 or only to block 2 yields some benefits, but using both early blocks jointly provides a better balance across domains, models, and tasks. This pattern aligns with our intuition: anchoring both early blocks preserves coarse semantics at the onset of perturbation generation, which in turn biases later blocks to place perturbations along near-object regions rather than injecting unconstrained noise. As a consequence, the resulting perturbations align more closely with shared, object-level structure across architectures and datasets, thereby enhancing modeland data-agnostic black-box transfer. See Supp.§E for ablations of other components.

Table 7: Hyperparameter $( \lambda _ { \mathrm { c o n s } } , \eta )$ sensitivity and earlyblock selection. (Domain (Acc.), Model (Acc.), SS (mIoU), OD (mAP50)). 

<table><tr><td> $\lambda_{cons}$ </td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.7</td><td>0.9</td></tr><tr><td>Domain</td><td>49.55</td><td>48.29</td><td>48.49</td><td>47.10</td><td>50.08</td></tr><tr><td>Model</td><td>44.84</td><td>44.80</td><td>44.68</td><td>44.13</td><td>45.89</td></tr><tr><td>Task (SS)</td><td>22.59</td><td>23.63</td><td>23.82</td><td>23.40</td><td>22.79</td></tr><tr><td>Task (OD)</td><td>23.96</td><td>24.36</td><td>24.78</td><td>24.52</td><td>24.19</td></tr><tr><td> $\eta$ </td><td>0.9</td><td>0.95</td><td>0.98</td><td>0.99</td><td>0.995</td></tr><tr><td>Domain</td><td>48.72</td><td>50.47</td><td>51.17</td><td>47.97</td><td>48.09</td></tr><tr><td>Model</td><td>45.04</td><td>45.56</td><td>45.84</td><td>44.79</td><td>44.70</td></tr><tr><td>Task (SS)</td><td>24.89</td><td>23.92</td><td>24.23</td><td>23.39</td><td>24.34</td></tr><tr><td>Task (OD)</td><td>24.83</td><td>24.51</td><td>24.73</td><td>24.26</td><td>24.48</td></tr><tr><td>Block</td><td colspan="2">1</td><td>2</td><td colspan="2">1 &amp; 2</td></tr><tr><td>Domain</td><td colspan="2">49.13</td><td>49.88</td><td colspan="2">47.10</td></tr><tr><td>Model</td><td colspan="2">47.62</td><td>48.82</td><td colspan="2">44.13</td></tr><tr><td>Task (SS)</td><td colspan="2">23.78</td><td>22.57</td><td colspan="2">23.40</td></tr><tr><td>Task (OD)</td><td colspan="2">24.47</td><td>24.15</td><td colspan="2">24.52</td></tr></table>

# 5 CONCLUSION

In this paper, we introduce a semantically consistent generative attack leveraging the Mean Teacher and early-block semantic consistency to preserve the object semantics during perturbation generation, thus guiding it towards objectsalient regions to markedly improve black-box transferability as in Fig. 5. With comprehensive evaluations across various models, domains, and tasks, we demonstrate object salient regions play a crucial role within the generator. See Supp. §E.6 for limitations and broader societal impact.

![](images/bdf3b8938f718c1109336de67052a5d72179f7f453d1a45dd5e81c6ac559aff6.jpg)

<details>
<summary>scatter</summary>

| Cross-Domain | ASR (%) | ACR (%) | FR (%) |
| ------------ | ------- | ------- | ------ |
| CDA          | 25      | 15      | 9.5    |
| LTP          | 40      | 11      | 8.75   |
| BIA          | 45      | 10      | 8.5    |
| GAMA         | 45      | 9.5     | 8.25   |
| FACL         | 45      | 9       | 8.0    |
| PDCL         | 45      | 8       | 7.75   |
</details>

Figure 5: Our semantically consistent generative attack effectively exploits the generator intermediates to craft adversarial examples to enhance transferability from the baselines ( → ▼) across domains (a) and models (b).

# ACKNOWLEDGMENTS

This work was supported by the Technology Innovation Program (2410013617,20024355, Development of autonomous driving connectivity technology based on sensor-infrastructure cooperation) funded By the Ministry of Trade, Industry & Energy (MOTIE, Korea) and the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2024-00457882, AI Research Hub Project).

# REFERENCES

Abhishek Aich, Calvin-Khang Ta, Akash Gupta, Chengyu Song, Srikanth Krishnamurthy, Salman Asif, and Amit Roy-Chowdhury. Gama: Generative adversarial multi-object scene attacks. Advances in Neural Information Processing Systems, 35:36914–36930, 2022. 1, 3, 5   
Shumeet Baluja and Ian Fischer. Adversarial transformation networks: Learning to generate adversarial examples. arXiv preprint arXiv:1703.09387, 2017. 1   
Shumeet Baluja and Ian Fischer. Learning to attack: Adversarial transformation networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018. 1   
Junyoung Byun, Seungju Cho, Myung-Joon Kwon, Hee-Seon Kim, and Changick Kim. Improving the transferability of targeted adversarial examples through object-based diverse input. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15244–15253, 2022. 4   
Shengcao Cao, Dhiraj Joshi, Liang-Yan Gui, and Yu-Xiong Wang. Contrastive mean teacher for domain adaptive object detectors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 23839–23848, 2023. 3   
Nicholas Carlini, Anish Athalye, Nicolas Papernot, Wieland Brendel, Jonas Rauber, Dimitris Tsipras, Ian Goodfellow, Aleksander Madry, and Alexey Kurakin. On evaluating adversarial robustness. arXiv preprint arXiv:1902.06705, 2019. 1   
Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021. 4   
Jinhong Deng, Wen Li, Yuhua Chen, and Lixin Duan. Unbiased mean teacher for cross-domain object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4091–4101, 2021. 3   
Mario Döbler, Robert A Marsden, and Bin Yang. Robust mean teacher for continual and gradual test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7704–7714, 2023. 3   
Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li. Boosting adversarial attacks with momentum. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 9185–9193, 2018. 1   
Yinpeng Dong, Tianyu Pang, Hang Su, and Jun Zhu. Evading defenses to transferable adversarial examples by translation-invariant attacks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4312–4321, 2019. 1   
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 7   
Shanghua Gao, Zhong-Yu Li, Ming-Hsuan Yang, Ming-Ming Cheng, Junwei Han, and Philip Torr. Large-scale unsupervised semantic segmentation. 2022. 3

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020. 4   
Chuan Guo, Mayank Rana, Moustapha Cisse, and Laurens Van Der Maaten. Countering adversarial images using input transformations. arXiv preprint arXiv:1711.00117, 2017. 7   
Qian Huang, Isay Katsman, Horace He, Zeqi Gu, Serge Belongie, and Ser-Nam Lim. Enhancing adversarial example transferability with an intermediate level attack. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4733–4742, 2019. 4   
Woo Jae Kim, Seunghoon Hong, and Sung-Eui Yoon. Diverse generative perturbations on attention space for transferable adversarial attacks. In 2022 IEEE international conference on image processing (ICIP), pp. 281–285. IEEE, 2022. 4   
Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In 2013 IEEE International Conference on Computer Vision Workshops, pp. 554–561, 2013. doi: 10.1109/ICCVW.2013.77. 8   
Alexey Kurakin, Ian J Goodfellow, and Samy Bengio. Adversarial examples in the physical world. In Artificial intelligence safety and security, pp. 99–112. Chapman and Hall/CRC, 2018. 7   
Youngwan Lee, Jeffrey Ryan Willette, Jonghee Kim, Juho Lee, and Sung Ju Hwang. Exploring the role of mean teachers in self-supervised masked auto-encoders. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum ?id=7sn6Vxp92xV. 4   
Qizhang Li, Yiwen Guo, Wangmeng Zuo, and Hao Chen. Improving adversarial transferability via intermediate-level perturbation decay. Advances in Neural Information Processing Systems, 36: 32900–32912, 2023. 4   
Yu-Jhe Li, Xiaoliang Dai, Chih-Yao Ma, Yen-Cheng Liu, Kan Chen, Bichen Wu, Zijian He, Kris Kitani, and Peter Vajda. Cross-domain adaptive teacher for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7581–7590, 2022. 3   
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017. 1   
Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew B. Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. ArXiv, abs/1306.5151, 2013. URL https://api.semantic scholar.org/CorpusID:2118703. 8   
Krishna Nakka and Mathieu Salzmann. Learning transferable adversarial perturbations. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=sIDvIyR5I1R. 1, 3, 5, 6   
Krishna Kanth Nakka and Alexandre Alahi. Nat: Learning to attack neurons for enhanced adversarial transferability. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 7593–7604. IEEE, 2025. 1, 4, 6   
Muhammad Muzammal Naseer, Salman H Khan, Muhammad Haris Khan, Fahad Shahbaz Khan, and Fatih Porikli. Cross-domain transferability of adversarial perturbations. Advances in Neural Information Processing Systems, 32, 2019. 1, 3, 5, 6   
Omid Poursaeed, Isay Katsman, Bicheng Gao, and Serge Belongie. Generative adversarial perturbations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4422–4431, 2018. 1, 5, 6   
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Fei-Fei Li. Imagenet large scale visual recognition challenge. IJCV, 2015. 5, 8

Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pp. 618–626, 2017. 6, 8   
Naman D Singh, Francesco Croce, and Matthias Hein. Revisiting adversarial training for imagenet: Architectures, training and generalization across threat models. In NeurIPS, 2023. 7   
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013. 1   
Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems, 30, 2017. 3   
C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technical report, California Institute of Technology, 2011. 8   
Chaoyue Wang, Chang Xu, Chaohui Wang, and Dacheng Tao. Perceptual adversarial networks for image-to-image transformation. IEEE Transactions on Image Processing, 27(8):4066–4079, 2018. 1   
Weibin Wu, Yuxin Su, Xixian Chen, Shenglin Zhao, Irwin King, Michael R Lyu, and Yu-Wing Tai. Boosting the transferability of adversarial samples via attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1161–1170, 2020. 4   
Chaowei Xiao, Bo Li, Jun-Yan Zhu, Warren He, Mingyan Liu, and Dawn Song. Generating adversarial examples with adversarial networks. arXiv preprint arXiv:1801.02610, 2018. 1   
Cihang Xie, Jianyu Wang, Zhishuai Zhang, Zhou Ren, and Alan Yuille. Mitigating adversarial effects through randomization. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=Sk9yuql0Z. 7   
Cihang Xie, Zhishuai Zhang, Yuyin Zhou, Song Bai, Jianyu Wang, Zhou Ren, and Alan L Yuille. Improving transferability of adversarial examples with input diversity. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2730–2739, 2019. 1   
Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples in deep neural networks. In NDSS, 2018. doi: 10.14722/ndss.2018.23295. URL https: //www.ndss-symposium.org/ndss-paper/feature-squeezing-detecting -adversarial-examples-in-deep-neural-networks/. 7   
Hunmin Yang, Jongoh Jeong, and Kuk-Jin Yoon. Facl-attack: Frequency-aware contrastive learning for transferable adversarial attacks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 6494–6502, 2024a. 1, 3, 5, 6, 9   
Hunmin Yang, Jongoh Jeong, and Kuk-Jin Yoon. Prompt-driven contrastive learning for transferable adversarial attacks. In European Conference on Computer Vision, pp. 36–53. Springer, 2024b. 1, 3, 5, 6   
Chaoning Zhang, Adil Karjauv, Philipp Benz, Soomin Ham, Gyusang Cho, Chan-Hyun Youn, and In So Kweon. Is fgsm optimal or necessary for l∞ adversarial attack? In Workshop on Adversarial Machine Learning in Real-World Computer Vision Systems and Online Challenges (AML-CV). Computer Vision Foundation (CVF), IEEE Computer Society, 2021. 1   
Qilong Zhang, Xiaodan Li, Yuefeng Chen, Jingkuan Song, Lianli Gao, Yuan He, and Hui Xue. Beyond imagenet attack: Towards crafting adversarial examples for black-box domains. In International Conference on Learning Representations, 2022. 1, 2, 3, 4, 5, 6, 7, 8, 9   
Shiji Zhao, Jie Yu, Zhenlong Sun, Bo Zhang, and Xingxing Wei. Enhanced accuracy and robustness via multi-teacher adversarial distillation. In European Conference on Computer Vision, pp. 585–602. Springer, 2022. 3