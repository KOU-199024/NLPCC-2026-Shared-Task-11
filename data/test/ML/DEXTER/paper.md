# DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models

Simone Carnemolla1∗ Matteo Pennisi1∗ Sarinda Samarasinghe2 Giovanni Bellitto1 Simone Palazzo1 Daniela Giordano1 Mubarak Shah2 Concetto Spampinato1

1University of Catania 2University of Central Florida simone.carnemolla@phd.unict.it matteo.pennisi@unict.it

# Abstract

Understanding and explaining the behavior of machine learning models is essential for building transparent and trustworthy AI systems. We introduce DEX-TER, a data-free framework that employs diffusion models and large language models to generate global, textual explanations of visual classifiers. DEXTER operates by optimizing text prompts to synthesize class-conditional images that strongly activate a target classifier. These synthetic samples are then used to elicit detailed natural language reports that describe class-specific decision patterns and biases. Unlike prior work, DEXTER enables natural language explanation about a classifier’s decision process without access to training data or groundtruth labels. We demonstrate DEXTER’s flexibility across three tasks—activation maximization, slice discovery and debiasing, and bias explanation—each illustrating its ability to uncover the internal mechanisms of visual classifiers. Quantitative and qualitative evaluations, including a user study, show that DEXTER produces accurate, interpretable outputs. Experiments on ImageNet, Waterbirds, CelebA, and FairFaces confirm that DEXTER outperforms existing approaches in global model explanation and class-level bias reporting. Code is available at https://github.com/perceivelab/dexter.

# 1 Introduction

How can we systematically uncover and explain a deep visual classifier’s decision-making process in a way that is both comprehensive and human-interpretable?

This question is crucial as AI systems are increasingly deployed in high-stakes applications, where interpretability and trust are as important as accuracy. However, the lack of transparency in model reasoning, often exacerbated by the reliance on spurious correlations—irrelevant features that disproportionately influence predictions— undermines confidence in these systems, making decisions difficult to justify. For instance, ImageNet-trained classifiers have been shown to favor background textures or lighting conditions over intrinsic object properties [40, 9]. Addressing these issues requires explainability techniques that extend beyond local attribution, offering a global perspective on a model’s reasoning patterns, biases, and learned representations.

Existing methods for model explainability, such as GradCAM [38] and Integrated Gradients [42] provide local explanations and require data. These methods focus on analyzing individual predictions by attributing importance to specific pixels or regions in an image. While these methods are extremely useful for highlighting areas of interest, they do not offer a global understanding of model behavior. In contrast, activation maximization techniques [26–28] have been instrumental in globally visualizing the features learned by neural network, but the generated images are often abstract and challenging to interpret. While existing methods offer useful insights into model behavior and feature representations, they often fall short of capturing the global reasoning patterns and biases behind predictions. In particular, visual explanations can be hard to interpret and may lack the ability to convey high-level reasoning or reveal subtle spurious correlations—issues that are often better addressed through complementary textual explanations [18, 6, 16, 3].

Text-based explanations offer an accessible complement to visual methods [11], but they often lack global perspective or clarity. Recent work [16, 2, 10], including Natural Language Explanation (NLE) methods [29, 36, 15], combines visual and textual cues to reveal model biases, but typically relies on labeled data, annotations, or pretrained vision-language mappings [2].

We propose DEXTER, a framework that generates human-interpretable textual explanations, uncovering the global reasoning patterns and biases of visual classifiers. DEXTER operates in a fully data-free setting by leveraging the generative capabilities of diffusion models and the reasoning power of large language models (LLMs). At its core, DEXTER optimizes soft prompts, which are mapped to discrete hard prompts, in order to guide a diffusion-based image generation process. This ensures that the generated images are aligned with the outputs of a target classifier, capturing the features and concepts prioritized by the model. The generated images are then analyzed by an LLM, which reasons across them to provide textual explanations of the classifier’s decision-making process. By combining image generation and textual reasoning, DEXTER not only overcomes the interpretability challenges of visual explanations but also provides a global understanding of model behavior, including identifying spurious correlations and biases, without requiring access to any training data or ground truth labels.

Thus, the core main contributions of DEXTER are:

• Global explanations: A high-level understanding of model decisions, identifying key features, biases, and patterns beyond local attribution.   
• Data-free approach: DEXTER requires only the trained model for explanations, unlike existing methods, that employ training or ground-truth data.   
• Bias identification and explanation: Uncovering and describing spurious correlations to support model debiasing.   
• Natural language reasoning: LLM-generated textual explanations that enhance interpretability over purely visual methods.

We evaluate DEXTER across three tasks: (1) activation maximization to reveal model-prioritized features, (2) slice discovery to detect underperforming subpopulations, and (3) bias explanation to identify spurious correlations. Experiments use SalientImageNet [40], Waterbirds [35], CelebA [21], and FairFaces [14]—datasets widely used in fairness and interpretability research. Across tasks, DEXTER demonstrates strong performance, offering meaningful insights into model behavior.

# 2 Related work

Activation Maximization (AM) interprets neural networks by generating inputs that maximize neuron activations [7]. Early methods often produced unrealistic, uninterpretable images [39, 25], later improved through regularization [23] and generative priors [26]. DiffExplainer [30] introduced diffusion-based image generation guided by soft prompts, which improve realism but lack semantic transparency. This work points toward token-wise optimization, but this direction remains underexplored. In contrast, DEXTER builds on this line by replacing soft prompts with discrete (hard) tokens, which are interpretable and enable both visual and textual global explanations. Although once central to model interpretability, AM has seen limited progress in recent years, partly because the resulting images—while optimized for neuron activation—are often difficult to interpret semantically, making it challenging to draw clear, causal insights about model behavior. The field has largely shifted toward attribution methods—e.g., GradCAM [38], Integrated Gradients [42], which are computationally efficient and provide intuitive saliency maps, though they are inherently local and data-dependent. Other efforts in feature visualization [49] offer insights into internal representations but require training data and remain focused on visual output. DEXTER revives AM by leveraging modern diffusion models and discrete prompt optimization to produce class-level, multimodal explanations in a fully data-free setting.

Explanation of visual classifiers. Textual and Multimodal Explanations seek to complement visual saliency with natural language justifications. Approaches like Multimodal Explanations [29, 36], and post-hoc counterfactuals [2] align vision and language to explain model decisions. However, these methods often rely on ground-truth labels or annotated datasets and or limited to instance-level explanations. In contrast, DEXTER provides global, class-wise explanations without supervision or access to data.

Natural Language Explanation (NLE) methods aim to generate human-readable justifications, typically using VQA-style benchmarks [29, 36, 15]. Recent approaches integrate vision and language into unified architectures [37, 36], but remain supervised and local. DEXTER differs by producing global textual reports through unsupervised classifier probing—without labels, data, or task-specific fine-tuning.

Slice Discovery identifies dataset subpopulations where a model underperforms, offering a targeted way to reveal and explain systematic failures in classifier behavior. Traditional methods analyze embeddings, gradients, or misclassifications [1, 24, 41, 48], while recent approaches leverage CLIP’s joint text-visual embeddings [8, 12, 47] or use LLMs to generate captions and extract keywords [44, 16]. Bias-to-Text (B2T)[16] extracts pseudo-bias labels from misclassified image captions, and LADDER[10] generates bias hypotheses from low-confidence predictions, clustering samples via LLM-derived pseudo-attributes. Unlike these data-dependent methods, DEXTER discovers and explains biases in a fully data-free manner, relying only on the classifier’s internal behavior.

# 3 Method

DEXTER’s framework, shown in Fig. 1, integrates three key components: a text pipeline for optimizing prompts, a vision pipeline for the image generation process, and a reasoning module using a vision-language model (VLM). DEXTER begins by optimizing a soft prompt to condition a BERT model [5] to fill in masked tokens in a predefined sentence. The resulting prompt guides the stable diffusion process to generate images that maximize the activation of a set of target neurons (e.g. classification heads) in a given visual classifier. The generated images are then analyzed by the VLM, which reasons across multiple images to provide coherent, human-readable textual explanations of the model’s decision-making process.

# 3.1 Text pipeline

The text pipeline has the goal of optimizing a textual prompt to suitably condition the diffusion model. We pose prompt generation as a masked language modeling task, and employ a pretrained and frozen BERT model for this purpose. The structure of the textual prompt to be produced is fully customizable, and can be controlled by combining portions of fixed text with a set of mask tokens. For the sake of clarity and without loss of generality, we will assume that the textual prompt has the structure of a sequence $\mathbf { t } = [ \mathbf { t } _ { \mathrm { f i x e d } } , m _ { 1 } , m _ { 2 } , \dots , m _ { N } ]$ , where $\mathbf { t } _ { \mathrm { f i x e d } }$ is the portion of fixed text and all $m _ { i }$ are set to BERT’s [MASK]token. Let $\mathbf { t } _ { \mathrm { e m b } }$ be the embedding of t, including positional encoding. In order to alter BERT’s behavior, which would naturally tend to replace $m _ { i }$ with the most likely tokens based on its pretraining, we also prepend to the input sequence a learnable soft prompt $\mathbf { p } \in \mathbb { R } ^ { P \times d }$ , consisting of a sequence of $P$ vectors, with d being the dimensionality of BERT’s embedding space [17]. The full input sequence to BERT is thus $[ \mathbf { p } , \mathbf { t } _ { \mathrm { e m b } } ]$ . We read out the logits corresponding to the masked tokens, i.e., $[ \mathbf { l } _ { 1 } , \ldots , \mathbf { l } _ { N } ]$ , where each $\mathbf { l } _ { i } \in \mathbb { R } ^ { V }$ and V is BERT’s vocabulary size. Each logit vector $\mathbf { l } _ { i }$ is mapped to a differentiable one-hot vector $\mathbf { o } _ { i } \in \left\{ 0 , 1 \right\} ^ { V } , \sum o _ { i , j } = 1$ , through a Gumbel-softmax [22, 13] (with temperature $\tau = 1 )$ , from which the corresponding predicted token $\hat { t } _ { i } \in \{ 1 , \ldots , V \}$ is retrieved. The resulting text prompt can be recovered as $\left[ \mathbf { t } _ { \mathrm { f i x e d } } , \hat { t } _ { 1 } , \dots , \hat { t } _ { N } \right]$ .

At this point, a practical problem arises, in that standard implementations of diffusion models $( \mathrm { e . g . }$ , Stable Diffusion [34]) CLIP’s text encoder [31] is employed to embed textual prompts into a conditioning vector. Unfortunately, BERT’s and CLIP’s vocabulary overlap only partially. To address this issue, we employ a translation matrix $\mathbf { M } \in \{ 0 , 1 \} ^ { V \times W }$ to map each one-hot vector $\mathbf { o } _ { i }$ to its corresponding representation in $\mathrm { C L I P } ^ { \prime } \mathrm { s }$ vocabulary, of size W . In M, each row contains a single 1, indicating the index of the corresponding token in the CLIP vocabulary. Thus, given an original onehot vector $\mathbf { o } _ { i }$ provided by BERT, we can translate it into its CLIP equivalent through $\mathbf { o } _ { i } ^ { ( C ) } = \mathbf { o } _ { i } \mathbf { M }$ , indexing a token $\hat { t } _ { i } ^ { \mathrm { ( C ) } }$ in $\mathrm { C L I P } \mathrm { s }$ vocabulary. For unassigned BERT tokens, the corresponding rows in

![](images/608e40a22fd43bb8c1cceda3e9a4b9bad10e5e75a5813bde591320e5318d38d2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Goal: Investigate class &quot;bee&quot; on Vision Classifier"] --> B["1. Text Pipeline"]
    B --> C["Soft Prompt"]
    C --> D["t_fixed"]
    D --> E["m"]
    E --> F["BERT"]
    F --> G["Masked Token Prediction"]
    G --> H["BERT Logits"]
    H --> I["Gumbel Softmax"]
    I --> J["[FLOWER"] (BERT Token)]
    J --> K["CLIP Text Encoder"]
    K --> L["Translation Matrix M (BERT to CLIP)"]
    L --> M["×"]
    M --> N["t_fixed + [FLOWER"] (CLIP Token)]
    N --> O["2. Vision Pipeline"]
    O --> P["Stable Diffusion"]
    P --> Q["Generated Sample from Text Prompt"]
    Q --> R["Sample Storage"]
    R --> S["Vision Classifier"]
    S --> T["Loss_act with Investigated Class"]
    T --> U["3. Reasoning Module"]
    U --> V["Stored Samples"]
    V --> W["Captioner"]
    W --> X["Captions: A picture of a purple flower, A picture of a flower with a green background ..."]
    X --> Y["VLM"]
    Y --> Z["Bias Report: Based on the analysis of key attributes, it is plausible to hypothesize that there may be a bias in the model towards generating images that heavily emphasize the natural habitat of bees..."]
```
</details>

Figure 1: DEXTER investigates classifier biases by optimizing a learnable soft prompt to generate text prompts. These text prompts condition a diffusion model to generate images that maximize the activation of the target class in the vision classifier. Images that correctly activate the target class are stored and later captioned for Bias Reasoning. A VLM reasons using these captions to produce human-understandable textual explanations of the model’s decisions and potential biases. More details and clarifications about the pipeline can be found in the Appendices A and B.

M are entirely zero. As a result of this design, the model learns to avoid predicting BERT tokens that do not have a valid mapping in CLIP, as they correspond to a zero indexing vector $\mathbf { o } _ { i } ^ { ( C ) }$ , which would provide a meaningless representation when multiplied by the CLIP embedding look-up table.

The textual prompt used to condition the diffusion model is thus $\left\lceil \mathbf { t } _ { \mathrm { f i x e d } } , \hat { t } _ { 1 } ^ { \mathrm { ( C ) } } , \dots , \hat { t } _ { N } ^ { \mathrm { ( C ) } } \right\rceil$ . Thanks to the Gumbel-Softmax activation and the linear mapping between vocabularies, the whole process is fully differentiable, allowing us to optimize the soft prompt p, which is the only learnable parameter, through classic backpropagation.

# 3.2 Vision pipeline

The goal of the vision pipeline is to synthesize a realistic and interpretable image that maximizes the activation of a set of neurons of a visual classifier. In the following, we refer to “neurons” in a generic way, regardless of whether they encode features (i.e., neurons in intermediate layers) or classes (i.e., neurons in the output layer, after applying softmax). Given the predicted textual prompt $\left\lceil \mathbf { t } _ { \mathrm { f i x e d } } , \hat { t } _ { 1 } ^ { \mathrm { ( C ) } } , \dots , \hat { t } _ { N } ^ { \mathrm { ( C ) } } \right\rceil$ (with all tokens ensured to belong to CLIP’s vocabulary as explained in Sect. 3.1), we feed it to $\mathrm { C L } \bar { \mathrm { I P } } ^ { \prime } \mathrm { s }$ text encoder to obtain an embedding vector e. This vector is then used to condition a pretrained and frozen diffusion model d. In this work, we use Stable Diffusion, due to its widespread adoption and versatility in generating high-quality images from textual descriptions. Let $f : { \underline { { \mathcal { T } } } } \to \mathbb { R } ^ { K }$ be the target frozen visual classifier, pretrained over a set of C classes, providing as output the responses of a subset of K selected neurons, whose activations we intend to maximize. Hence, we can obtain the selected activation vector as $\mathbf { n } = f \left( d \left( \mathbf { e } \right) \right)$ .

The whole vision pipeline is differentiable, enabling the definition of an optimization objective for n, which directly affects the learnable soft prompt p in the text pipeline.

In order to guide the generation process to describe the behavior of the visual classifier, we introduce a neuron activation maximization loss $\mathcal { L } _ { \mathrm { a c t } }$ that encourages learning a textual prompt and a synthetic image that maximizes the response of the selected n neurons. We define $\mathcal { L } _ { \mathrm { a c t } }$ as:

$$
\mathcal {L} _ {\mathrm{act}} = \sum_ {i = 1} ^ {K} l _ {\mathrm{act}} (n _ {i}), \tag {1}
$$

where $n _ { i }$ is the activation of the i-th element in n, and $l _ { \mathrm { a c t } }$ depends on whether $n _ { i }$ is a feature neuron or a class neuron :

$$
l _ {\text { act }} (n _ {i}) = \left\{ \begin{array}{l l} - n _ {i}, & \text { if   } n _ {i} \text {   is   a   feature   neuron } \\ - \log n _ {i}, & \text { if   } n _ {i} \text {   is   a   class   neuron } \end{array} \right. \tag {2}
$$

# 3.3 Masked pseudo-labels prediction

In our preliminary experiments, we observed that, when using the activation maximization objective only, the gradient propagated to the learnable soft prompt p in the text pipeline was too small, slowing down or even preventing convergence. To address this issue, we introduce, in the text pipeline, an auxiliary mask prediction task to provide a shorter backpropagation path to p. The design of the auxiliary task gives us the opportunity to add another explainability feature to our approach: associating masked tokens with subsets of target neurons, in order to encourage the mapping between neuron activations and specific portions of the textual prompt.

To facilitate this, we initialize a set of pseudo-labels $y _ { 1 } , \ldots , y _ { N }$ , one for each mask token position $\displaystyle m _ { 1 } , \ldots , m _ { N }$ . Each pseudo-label $y _ { i } \in \left\{ 1 , \ldots , V \right\}$ is associated with a reference loss $L _ { i }$ , initialized $\mathrm { t o } + \infty$ , and with the set of reference neurons $\mathcal { N } _ { i } \subseteq \{ 1 , \ldots , K \}$ . At each optimization step, for each mask token $m _ { i }$ , BERT predicts the logits $\mathbf { l } _ { i } .$ , from which we compute the (standard) softmax vector $\mathbf { s } _ { i }$ and the corresponding predicted token $\hat { t } _ { i }$ , following the notation introduced in Sect. 3.1. We define the aggregated activation loss $\mathcal { L } _ { \mathrm { a g g } , i }$ for the set of associated reference neurons:

$$
\mathcal {L} _ {\mathrm{agg}, i} = \sum_ {j \in \mathcal {N} _ {i}} l _ {\mathrm{act}} (n _ {j}). \tag {3}
$$

Then, if ${ \mathcal { L } } _ { \mathrm { a g g , } }$ i is smaller than the corresponding reference loss $L _ { i }$ , we update both the pseudo-label $y _ { i } \gets { \hat { t } } _ { i }$ and the reference loss $L _ { i } \gets { \mathcal { L } } _ { \mathrm { a g g } , i }$ . If the pseudo-label $y _ { i }$ has been set for $m _ { i }$ (and the corresponding reference loss $L _ { i }$ is finite), we add a cross-entropy loss term $\mathcal { L } _ { \mathrm { m a s k } , i } = - \log s _ { i , y _ { i } }$ , with $s _ { i , j }$ being the j-th element of the softmax vector $\mathbf { s } _ { i }$ . This approach ensures that the pseudo-labels are continually refined to better align with the activation patterns of the target neurons as training progresses, while constraining the model’s parameters to remain within a region of the parameter space that corresponds to meaningful, interpretable configurations. After the first iteration, where all $y _ { i }$ are set, the overall loss function $\mathcal { L }$ is then:

$$
\mathcal {L} = \sum_ {k = 1} ^ {K} l _ {\mathrm{act}} (n _ {k}) - \sum_ {i = 1} ^ {N} \log s _ {i, y _ {i}}. \tag {4}
$$

A possible issue that may arise is the prediction of outlier tokens that could randomly decrease the activation loss, which would possibly update the pseudo-labels to spurious values. To prevent this, for each masked token position $m _ { i }$ , we maintain a history of aggregated losses for each word predicted during the optimization process, using the history mean for comparison with the reference loss $L _ { i }$ . For instance, let us assume that masked token $m _ { i }$ has been mapped by BERT to vocabulary token $t ^ { * }$ in $T$ training iterations, building up a list of corresponding aggregated losses $\left[ \mathcal { L } _ { \mathrm { a g g } , i } ^ { ( 1 ) } , \ldots , \mathcal { L } _ { \mathrm { a g g } , i } ^ { ( T ) } \right]$ Lagg,i Instead of comparing $L _ { i }$ with the most recent aggregated loss (T ) $\mathcal { L } _ { \mathrm { a g g } , i } ^ { ( T ) }$ Lagg,i, we check whether:

$$
\frac {1}{T} \sum_ {j = 1} ^ {T} \mathcal {L} _ {\mathrm{agg}, i} ^ {(j)} <   L _ {i}, \tag {5}
$$

and only in this case do we update $L _ { i } .$ . This approach prioritizes the prediction of words with lower historical activation loss, preventing the selection as pseudo-target outlier words that lead to random loss fluctuations.

# 4 Performance analysis

# 4.1 Datasets

To demonstrate the versatility and effectiveness of DEXTER as a global explanation framework, we design a comprehensive evaluation protocol that reflects the core dimensions of model interpretability: feature relevance, bias identification, and semantic alignment. We evaluate DEXTER across four key tasks: visual explanation, activation maximization, bias discovery, and bias text explanation, using four widely adopted datasets that enable rigorous and complementary assessments of interpretability. Each dataset was selected for its alignment with a specific evaluation goal and its established use in the literature.

SalientImageNet [40] is used to evaluate both visual explanations and activation maximization. It is a curated subset of ImageNet designed to analyze model reasoning through object and context annotations. For each class, the top-5 neural features—highly activated units in the penultimate layer—are annotated as either spurious or core, based on whether they reflect incidental or meaningful correlations with the target label. This enables a fine-grained assessment of DEXTER’s ability to highlight robust, semantically grounded features in both explanation and feature synthesis settings. Waterbirds[35] and CelebA[21] serve as benchmarks for bias discovery. Waterbirds introduces spurious correlations between bird species and backgrounds (e.g., land vs. water), providing a controlled environment for evaluating slice discovery and DEXTER’s ability to surface underperforming subpopulations. CelebA offers over 200,000 annotated face images with 40 binary attributes (e.g., gender, hairstyle, glasses), allowing us to assess how DEXTER identifies biases in classifier behavior related to demographic and semantic features.

FairFaces [14] is finally used for bias text explanation, focusing on the articulation of systematic spurious correlations in facial classification tasks. With over 100,000 images balanced across seven demographic groups, it enables a rigorous evaluation of how DEXTER captures and communicates bias in decision-making across diverse populations.

For each evaluation task, detailed optimization strategies, training procedure, VLM prompts and other information are provided in the appendix.

# 4.2 Results

We evaluate DEXTER’s visual explanations through quantitative metrics and a user study, then assess its performance in bias discovery, mitigation, and explanation, showcasing its ability to identify spurious correlations and enhance fairness. Finally, an ablation study examines the impact of individual components and DEXTER’s effectiveness in optimizing text prompts for activation maximization, comparing it to existing methods.

# 4.2.1 Visual Explanations

We evaluate DEXTER’s visual explanations through qualitative and quantitative comparisons with DiffExplainer, the only prior method that shares a similar diffusion-based generation pipeline, on the SalientImageNet in order to assess the semantic relevance and clarity of the synthesized explanations. For qualitative analysis, we conducted a user study (see Appendix H), involving 100 participants on Amazon Mechanical Turk, to further assess the interpretability of DEXTER’s visual explanations compared to DiffExplainer’s (the two methods with the highest CLIP-IQA scores in Tab. 6). Participants compared images generated by DiffExplainer and DEXTER alongside GradCAM-based attention heatmaps from SalientImageNet, evaluating similarity across three feature categories: perceptual features (shape, texture, color), conceptual features (semantics, context), and cases where no similarity was perceived (None). As shown in Fig. 2, DEXTER was preferred for conceptual features, while DiffExplainer was favored for perceptual attributes. Notably, DEXTER received fewer None responses, indicating stronger alignment with classifier attention regions. A chi-square test $( \chi ^ { 2 } = 1 5 . 3 6 , p = 0 . 0 3 2 )$ confirms a significant difference, with post-hoc analysis highlighting None and conceptual features as key contributors.

Fig. 3 presents an example of visual and textual explanations generated by DEXTER for RobustResNet50 [40], focusing on the 5 most active features in the penultimate layer for the dog sled class, which are all categorized as spurious in SalientImageNet. DEXTER-generated images align more effectively with the classifier’s attended regions compared to those generated by DiffExplainer. Furthermore, DEXTER provides textual descriptions that clearly explain the semantics of each neural feature, offering an interpretable account of the classifier’s reasoning. This approach enhances interpretability and enables users to either identify spurious correlations, such as the ones in the dog sled class where none of the five most important features actually include a sled, or confirm the classifier’s reliance on meaningful, task-relevant attributes.

![](images/4035f8b80ecc0177f50699aeb865fd61783723a8ed72876261bca918e204b159.jpg)

<details>
<summary>bar</summary>

| Feature Group        | DiffExplainer | DEXTER |
| -------------------- | ------------- | ------ |
| Perceptive Features  | 50%           | 50%    |
| Conceptual Features | 48%           | 52%    |
| None                 | 65%           | 35%    |
</details>

Figure 2: User study results on SalientImageNet: Participants evaluated the alignment between classifier attention and visual explanations, categorized into perceptual features (shape, texture, color), conceptual features (context, semantics, multiple elements), and no alignment.

For quantitative evaluation, we compare DEXTER to DiffExplainer using CLIP-IQA and Semantic CLIP-IQA [43] (Appendix D). DEXTER achieves higher scores $( 0 . 9 4 \bar { \pm } 0 . 0 3 / 0 . 9 6 \pm 0 . 0 3 )$ than DiffExplainer $( 0 . 8 9 \pm 0 . 0 9 / 0 . 9 0 \pm 0 . 0 5 )$ ), indicating better semantic alignment and consistency. Additional comparisons with GAN-based methods [45, 23, 26] further confirm the advantage of diffusion-based explanations.

Extended details/examples on the visual explanation task are in Appendix D.

![](images/5ccc78835c5e3459eac3414aa9ec4c6ab0fbec285d1113c6002dcbd38a8123ab.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["winter"] --> B["wolf"]
    B --> C["buck"]
    C --> D["tree"]
    D --> E["glacier"]
    E --> F["DEXTER"]
    F --> G["DiffExplainer"]
    G --> H["Salient Imagenet"]
    H --> I["Feat 0"]
    H --> J["Feat 1"]
    H --> K["Feat 2"]
    H --> L["Feat 3"]
    H --> M["Feat 4"]
```
</details>

Figure 3: DEXTER and DiffExplainer explanations for RobustResNet50 on "dog sled", showing spurious top-5 features as core visual elements.

# 4.2.2 Slice discovery and debiasing

The goal of slice discovery is to identify subgroups of data where the model exhibits worse performance compared to the rest of the dataset. We assess slice discovery and debiasing performance using CelebA and Waterbirds datasets. In the former, the task is “blonde”/“non-blonde” classification, with the “blonde” class including a low proportion of men (representing a slice). The Waterbirds dataset tackles “waterbird“/“landbird” classification, and is built so that a portion of waterbird images feature a land background, and vice versa, thus introducing a slice per class. We perform slice discovery with DEXTER by first identifying words that maximize the activation of a given class. Specifically, we leverage DEXTER’s optimization process to obtain several descriptive words for each class. Following this step, as done by Kim et al. [16], we compute the CLIP similarity between the discovered words and the images in the dataset: images with high similarity are identified as belonging to a slice. We thus train a debiased classifier using Distributionally Robust Optimization (DRO) [35, 32] and evaluate the classification accuracy on the identified slices. We compare this approach with ERM [35], LfF [24], GEORGE [41], JTT [19], CNC [46], DRO [35], LADDER [10] and DRO-B2T [16]. As shown in Tab. 1, DEXTER outperforms state-of-the-art methods on worst-slice prediction for CelebA and achieves comparable results on Waterbirds, deriving descriptive words directly from the model without relying on training data. Additional details on the slice discovery task and results are given in Appendix E.

Table 1: Performance on slice discovery and debiasing. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Data</td><td rowspan="2">GT</td><td colspan="2">CelebA</td><td colspan="2">Waterbirds</td></tr><tr><td>Worst</td><td>Avg.</td><td>Worst</td><td>Avg.</td></tr><tr><td>ERM</td><td>√</td><td>-</td><td>47.7 ± 2.1</td><td>94.9</td><td>62.6 ± 0.3</td><td>97.3</td></tr><tr><td>LfF</td><td>√</td><td>-</td><td>77.2</td><td>85.1</td><td>78.0</td><td>91.2</td></tr><tr><td>GEORGE</td><td>√</td><td>-</td><td>54.9 ± 1.9</td><td>94.6</td><td>76.2 ± 2.0</td><td>95.7</td></tr><tr><td>JTT</td><td>√</td><td>-</td><td>81.5 ± 1.7</td><td>88.1</td><td>83.8 ± 1.2</td><td>89.3</td></tr><tr><td>CNC</td><td>√</td><td>-</td><td>88.8 ± 0.9</td><td>89.9</td><td>88.5 ± 0.3</td><td>90.9</td></tr><tr><td>DRO</td><td>√</td><td>√</td><td>90.0 ± 1.5</td><td>93.3</td><td>89.9 ± 1.3</td><td>91.5</td></tr><tr><td>LADDER</td><td>√</td><td>-</td><td>89.2 ± 0.4</td><td>89.8</td><td>92.4 ± 0.8</td><td>93.1</td></tr><tr><td>DRO-B2T</td><td>√</td><td>-</td><td>90.4 ± 0.9</td><td>93.2</td><td>90.7 ± 0.3</td><td>92.1</td></tr><tr><td>DEXTER(Ours)</td><td>-</td><td>-</td><td>91.3 ± 0.01</td><td>91.7</td><td>90.5 ± 0.1</td><td>92.0</td></tr></table>

# 4.2.3 Bias explanation

To evaluate DEXTER’s ability to identify and explain biases in classifiers, we conduct an analysis using the FairFaces dataset [14]. Specifically, we train two binary classifiers to distinguish between two age groups: 20-29 (class 1) and 50-59 (class 2). These classifiers are trained on two variants of the FairFaces dataset: (1) a balanced dataset with equal male and female representation in both classes, (2) a dataset where males are overrepresented in class 1, and females are overrepresented in class 2. These variations allowed us to systematically assess how biases in the training data are reflected in the classifier’s behavior and how well DEXTER captures these biases.

To generate bias reports for a given classifier, DEXTER produces 50 images that maximize the model’s prediction for the target class. Each image is captioned using a VLM (ChatGPT-4o mini), and the list of captions is used to prompt another instance of the VLM to generate a textual bias report. This approach ensures that the generated reports reflect only the classifier’s internal representations and decision-making processes (more details about VLM hallucination evaluation in appendix I). Figure 4 showcases excerpts of two generated bias reports.

![](images/5beb976dc705c246dbc9f21cfb02aa892e6d71042d039217ee75323b4a8aab25.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["System Prompt"] --> C["LLM"]
    B["Captions"] --> C
    C --> D["Report 1: Overrepresentation of men in training data"]
    C --> E["..."]
    C --> F["3. Gender Representation: The slight overrepresentation of male subjects in the captions could suggest a bias in the training data, which may not equally represent young women."]
    C --> G["..."]
    C --> H["..."]
    C --> I["the model does not appear to exhibit significant bias in identifying the class of 20 to 29 years old individuals."]
    C --> J["..."]
    K["Report 2: No gender bias"] --> C
```
</details>

Figure 4: DEXTER explanation reports. Report 1 analyzes a classifier trained on a FairFaces variant with male overrepresentation in class 1, while Report 2 corresponds to a balanced dataset.

We evaluate DEXTER’s bias reports through two complementary approaches:

1. User Study. In addition to the earlier study, participants evaluated and ranked DEXTER’s bias reports for two classifiers. Informed of dataset biases, they assessed report accuracy, clarity, and interpretability. We report human Mean Opinion Scores $( \mathrm { M O S } _ { \mathrm { h u m a n s } } .$ , 1–5 scale), and complement them with automated evaluations: $\mathbf { M O S } _ { \mathbf { L L M } }$ from a VLM and G-eval [20] metrics, including $\mathrm { G - e v a l _ { c o n } }$ for consistency with classifier behavior. These scores jointly assess the linguistic, structural, and explanatory quality of DEXTER’s outputs.

2. Comparison with training-derived reports. We compare DEXTER’s data-free reports to those generated from training set images using the same pipeline (captioning + LLM-based reporting). Their similarity is quantified via sentence transformer similarity (STS) [33], measuring alignment with data-grounded biases.

Table 2 shows strong agreement between human $( \mathrm { M O S } _ { \mathrm { h u m a n s } } )$ and model $( \mathbf { M O S } _ { \mathbf { L L M } } )$ ratings (statistical validation in appendix F.2), with all scores across MOS and G-eval metrics exceeding 3 and approaching 4—indicating consistent perceptions of the reports as good to excellent. High STS scores further confirm that DEXTER’s reports, generated without data, align closely with training-derived insights. This demonstrates DEXTER’s ability to capture model biases fluently, faithfully, and without supervision. The bias reasoning enabled by DEXTER also facilitates disparity analysis across a variety of classifiers (from transformers to CNNs) as shown in Fig. 8 of the supplementary, where we compare ViT, AlexNet, ResNet50, and RobustResNet50 on ImageNet classes.

Full methodological details and report samples are provided in Appendix F and the supplementary.

Table 2: Evaluation of DEXTER bias reports generated for classifiers trained on FairFaces. The columns “w Bias” and “w/o Bias” refer to models trained on datasets with and without gender bias, respectively, for each age group. 

<table><tr><td rowspan="2">Metric</td><td colspan="2">Class 0 (20-29)</td><td colspan="2">Class 1 (50-59)</td><td rowspan="2">Mean</td></tr><tr><td>w Bias</td><td>w/o Bias</td><td>w Bias</td><td>w/o Bias</td></tr><tr><td>STS</td><td>0.92</td><td>0.85</td><td>0.91</td><td>0.91</td><td>0.90</td></tr><tr><td>G-evalcon</td><td>4.58 ± 1.00</td><td>4.80 ± 0.47</td><td>3.66 ± 0.84</td><td>3.68 ± 0.85</td><td>4.19 ± 0.52</td></tr><tr><td>MOSLLM</td><td>4.29 ± 0.63</td><td>4.80 ± 0.37</td><td>4.63 ± 0.44</td><td>4.19 ± 0.74</td><td>4.48 ± 0.25</td></tr><tr><td>MOSHumans</td><td>4.20 ± 0.63</td><td>3.89 ± 0.64</td><td>4.10 ± 0.67</td><td>3.88 ± 0.79</td><td>4.01 ± 0.69</td></tr></table>

# 4.2.4 Ablation Study

We investigate how prompt design, auxiliary optimization, and multi-target strategies affect DEX-TER’s ability to generate effective and interpretable visual explanations. Using 30 SalientImageNet classes (15 labeled with spurious and 15 with core features), we compare four prompting strategies by generating 100 images per class and measuring classifier activations: (1) class label only, (2) ChatGPT-generated descriptions, (3) captions from DiffExplainer, and (4) DEXTER’s optimized prompts. As shown in Table 3, DEXTER achieves the highest mean score (75.43), effectively maximizing both spurious (63.00) and core (87.86) features. In contrast, the baseline and ChatGPT perform moderately, while DiffExplainer underperforms.

We then ablate the role of prompt length and auxiliary loss $\mathcal { L } _ { \mathrm { m a s k } }$ by comparing single-word vs. multi-word optimization, with and without auxiliary supervision. Single-word prompts lack expressiveness (mean 23.73), and multi-word prompts without auxiliary loss are unstable (mean 11.83). Introducing $\mathcal { L } _ { \mathrm { m a s k } }$ improves both cases, with the best performance achieved when combining multi-word prompts and auxiliary loss (75.43). This highlights the importance of pairing rich prompts with stable optimization to align explanations with classifier behavior. The standard deviations reported in Tables 3 and 4 may appear high due to the variation in activation strength across different classes, as some classes are consistently activated and others only partially. For instance, while some classes were activated in 100 out of 100 generations, others were only in 20 out of 100. As a result, this large variation across classes naturally leads to a high overall standard deviation. Full ablation details are provided in Appendix G.

Finally, we evaluate the faithfulness and robustness of DEXTER’s explanations by testing for bias propagation and LLM-induced hallucinations. In the first test, we injected adversarial cues (e.g., using lion for tiger) into prompt initialization to assess whether upstream biases from BERT or Stable Diffusion would affect outputs. DEXTER consistently recovered class-relevant features, showing robustness to such distortions. In the second test, we extracted the most salient visual cue from each generated text report and added it to the image prompt; the resulting increase in classifier activation confirms that the cues contained in the text explanation reflect model-relevant features rather than hallucinations. Full details and statistical validation are provided in Appendix I.

# 5 Limitations

While DEXTER delivers detailed, data-free global explanations, it is computationally demanding: prompt optimization takes ∼10 minutes per class, though generating 100 images and the final bias report is fast (∼15 seconds without backpropagation), making it suitable for offline use. Since DEXTER relies on Stable Diffusion, there is a risk of NSFW outputs; to mitigate this, we apply its built-in safety checker to filter and discard inappropriate images during generation.

Table 3: Comparison between text-prompting strategies for maximizing the target class. 

<table><tr><td>Method</td><td>Spurious</td><td>Core</td><td>Mean</td></tr><tr><td>Baseline</td><td> $43.06 \pm 38.86$ </td><td> $86.40 \pm 23.83$ </td><td> $64.73 \pm 31.34$ </td></tr><tr><td>ChatGPT description</td><td> $41.20 \pm 40.78$ </td><td> $78.53 \pm 34.02$ </td><td> $59.87 \pm 37.40$ </td></tr><tr><td>DiffExplainer</td><td> $33.20 \pm 41.07$ </td><td> $47.66 \pm 44.80$ </td><td> $39.83 \pm 42.93$ </td></tr><tr><td>DEXTER (ours)</td><td> $63.00 \pm 31.20$ </td><td> $87.86 \pm 15.14$ </td><td> $75.43 \pm 23.17$ </td></tr></table>

Table 4: Ablation results for single-word and multi-word prompt optimization, with/without the auxiliary task ${ \mathcal { L } } _ { \mathrm { m a s k } }$ . 

<table><tr><td>Configuration</td><td>Spurious</td><td>Core</td><td>Mean</td></tr><tr><td>Single-word</td><td> $11.13 \pm 27.38$ </td><td> $36.33 \pm 38.45$ </td><td> $23.73 \pm 32.91$ </td></tr><tr><td> $\hookrightarrow + \mathcal{L}_{\text{mask}}$ </td><td> $34.00 \pm 32.72$ </td><td> $53.86 \pm 44.64$ </td><td> $43.93 \pm 38.68$ </td></tr><tr><td>Multi-word</td><td> $15.53 \pm 27.93$ </td><td> $8.13 \pm 18.74$ </td><td> $11.83 \pm 23.33$ </td></tr><tr><td> $\hookrightarrow + \mathcal{L}_{\text{mask}}$ </td><td> $63.00 \pm 31.20$ </td><td> $87.86 \pm 15.14$ </td><td> $75.43 \pm 23.17$ </td></tr></table>

# 6 Conclusion

We introduced DEXTER, a framework for globally explaining deep visual classifiers by combining diffusion-based activation maximization with textual reasoning. Unlike existing methods, DEXTER operates in a fully data-free setting, deriving insights solely from the classifier’s internal representations. Experiments on SalientImageNet, Waterbirds, CelebA, and FairFaces demonstrate its effectiveness in uncovering spurious and core features, identifying dataset subpopulations, and generating human-readable bias explanations. Ablation studies confirmed the impact of multi-word prompts and auxiliary optimization, while user studies validated the clarity and relevance of DEX-TER’s textual explanations. Future work will extend its capabilities to multimodal models and on refining textual reasoning.

# Acknowledgements

Simone Carnemolla, Matteo Pennisi, Giovanni Bellitto, Simone Palazzo, Daniela Giordano, and Concetto Spampinato have been supported by the European Union – Next Generation EU, Mission 4 Component 2 Line 1.3, through the PNRR MUR project PE0000013 – FAIR “Future Artificial Intelligence Research” (CUP E63C22001940006).

# References

[1] Chirag Agarwal, Daniel D’souza, and Sara Hooker. Estimating example difficulty using variance of gradients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10368–10378, 2022.   
[2] Saeid Asgari, Aliasghar Khani, Amir Hosein Khasahmadi, Aditya Sanghi, Karl D.D. Willis, and Ali Mahdavi Amiri. texplain: Post-hoc textual explanation of image classifiers with pre-trained language models. In ICLR 2024 Workshop on Reliable and Responsible Foundation Models, 2024. URL https://openreview.net/forum?id=0zScQ8kmuu.   
[3] Shi Chen and Qi Zhao. Rex: Reasoning-aware and grounded explanation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15586–15595, 2022.   
[4] Yashar Deldjoo. Understanding biases in chatgpt-based recommender systems: Provider fairness, temporal stability, and recency. ACM Trans. Recomm. Syst., 2024. doi: 10.1145/3690655. URL https://doi.org/10.1145/3690655.   
[5] Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[6] Finale Doshi-Velez and Been Kim. Towards a rigorous science of interpretable machine learning. In arXiv preprint arXiv:1702.08608, 2017.   
[7] Dumitru Erhan, Yoshua Bengio, Aaron Courville, and Pascal Vincent. Visualizing higher-layer features of a deep network. University of Montreal, 1341(3):1, 2009.   
[8] Sabri Eyuboglu, Maya Varma, Khaled Kamal Saab, Jean-Benoit Delbrouck, Christopher Lee-Messer, Jared Dunnmon, James Zou, and Christopher Re. Domino: Discovering systematic errors with cross-modal embeddings. In International Conference on Learning Representations.   
[9] Robert Geirhos, Patrick Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness, 2019.   
[10] Shantanu Ghosh, Rayan Syed, Chenyu Wang, Clare B. Poynton, Shyam Visweswaran, and Kayhan Batmanghelich. Ladder: Language driven slice discovery and error rectification, 2025. URL https://arxiv.org/abs/2408.07832.   
[11] Lisa Anne Hendricks, Zeynep Akata, Marcus Rohrbach, Jeff Donahue, Bernt Schiele, and Trevor Darrell. Generating visual explanations. In European Conference on Computer Vision (ECCV), pages 3–19. Springer, 2016. doi: 10.1007/978-3-319-46493-0\_1.   
[12] Saachi Jain, Hannah Lawrence, Ankur Moitra, and Aleksander Madry. Distilling model failures as directions in latent space. In The Eleventh International Conference on Learning Representations.   
[13] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations, 2017.   
[14] Kimmo Kärkkäinen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1548–1558, 2021.   
[15] Maxime Kayser, Oana-Maria Camburu, Leonard Salewski, Cornelius Emde, Virginie Do, Zeynep Akata, and Thomas Lukasiewicz. e-vil: A dataset and benchmark for natural language explanations in vision-language tasks, 2021. URL https://arxiv.org/abs/2105.03761.   
[16] Younghyun Kim, Sangwoo Mo, Minkyu Kim, Kyungmin Lee, Jaeho Lee, and Jinwoo Shin. Discovering and mitigating visual biases through keyword explanation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11082–11092, 2024.   
[17] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, 2021.   
[18] Zachary C. Lipton. The mythos of model interpretability: In machine learning, the concept of interpretability is both important and slippery. Queue, 16(3):31–57, 2018. doi: 10.1145/ 3236386.3241340.   
[19] Evan Z Liu, Behzad Haghgoo, Annie S Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, and Chelsea Finn. Just train twice: Improving group robustness without training group information. In International Conference on Machine Learning, pages 6781–6792. PMLR, 2021.   
[20] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, 2023.   
[21] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.   
[22] C Maddison, A Mnih, and Y Teh. The concrete distribution: A continuous relaxation of discrete random variables. In Proceedings of the international conference on learning Representations. International Conference on Learning Representations, 2017.

[23] Aravindh Mahendran and Andrea Vedaldi. Visualizing deep convolutional neural networks using natural pre-images. International Journal of Computer Vision, 120:233–255, 2016.   
[24] Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, and Jinwoo Shin. Learning from failure: De-biasing classifier from biased classifier. Advances in Neural Information Processing Systems, 33:20673–20684, 2020.   
[25] Anh Nguyen, Jason Yosinski, and Jeff Clune. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 427–436, 2015.   
[26] Anh Nguyen, Alexey Dosovitskiy, Jason Yosinski, Thomas Brox, and Jeff Clune. Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. In Advances in Neural Information Processing Systems (NeurIPS), pages 3387–3395, 2016.   
[27] Anh Nguyen, Jason Yosinski, Jeff Clune, Alexey Dosovitskiy, and Thomas Brox. Plug & play generative networks: Conditional iterative generation of images in latent space. arXiv preprint arXiv:1612.00005, 2017.   
[28] Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. Feature visualization. Distill, 2(11), 2017. doi: 10.23915/distill.00007.   
[29] Dong Huk Park, Lisa Anne Hendricks, Zeynep Akata, and Trevor Darrell. Multimodal explanations: Justifying decisions and pointing to the evidence. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8779–8788, 2018.   
[30] Matteo Pennisi, Giovanni Bellitto, Simone Palazzo, Isaak Kavasidis, Mubarak Shah, and Concetto Spampinato. Diffexplainer: Towards cross-modal global explanations with diffusion models. Computer Vision and Image Understanding, 262:104559, 2025. ISSN 1077-3142.   
[31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.   
[32] Hamed Rahimian and Sanjay Mehrotra. Distributionally robust optimization: A review. arXiv preprint arXiv:1908.05659, 2019.   
[33] N Reimers. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.   
[34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
[35] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks. In International Conference on Learning Representations, 2019.   
[36] Fawaz Sammani and Nikos Deligiannis. Uni-nlx: Unifying textual explanations for vision and vision-language tasks, 2023. URL https://arxiv.org/abs/2308.09033.   
[37] Fawaz Sammani, Tanmoy Mukherjee, and Nikos Deligiannis. Nlx-gpt: A model for natural language explanations in vision and vision-language tasks, 2022. URL https://arxiv.org/ abs/2203.05081.   
[38] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. 2017 IEEE International Conference on Computer Vision (ICCV), pages 618–626, 2017. doi: 10.1109/ICCV.2017.74.   
[39] K Simonyan, A Vedaldi, and A Zisserman. Deep inside convolutional networks: visualising image classification models and saliency maps. In Proceedings of the International Conference on Learning Representations (ICLR). ICLR, 2014.

[40] Sahil Singla, Sebastian Lapuschkin, Wojciech Samek, Alexander Binder, and Klaus-Robert Muller. Salient imagenet: How to discover spurious features in deep learning? Advances in Neural Information Processing Systems (NeurIPS), 2022.   
[41] Nimit Sohoni, Jared Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. No subclass left behind: Fine-grained robustness in coarse-grained classification problems. Advances in Neural Information Processing Systems, 33:19339–19352, 2020.   
[42] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In International Conference on Machine Learning, pages 3319–3328. PMLR, 2017.   
[43] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023.   
[44] Olivia Wiles, Isabela Albuquerque, and Sven Gowal. Discovering bugs in vision models using off-the-shelf image generation and captioning. In NeurIPS ML Safety Workshop.   
[45] Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. Understanding neural networks through deep visualization, 2015.   
[46] Michael Zhang, Nimit S Sohoni, Hongyang R Zhang, Chelsea Finn, and Christopher Ré. Correct-n-contrast: A contrastive approach for improving robustness to spurious correlations. arXiv preprint arXiv:2203.01517, 2022.   
[47] Yuhui Zhang, Jeff Z HaoChen, Shih-Cheng Huang, Kuan-Chieh Wang, James Zou, and Serena Yeung. Drml: Diagnosing and rectifying vision models using language. In NeurIPS ML Safety Workshop, 2022.   
[48] Bowen Zhao, Chen Chen, Qian-Wei Wang, Anfeng He, and Shu-Tao Xia. Combating unknown bias with effective bias-conflicting scoring and gradient alignment. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 3561–3569, 2023.   
[49] Roland S. Zimmermann, Janis Borowski, Robert Geirhos, Matthias Bethge, Thomas Wallis, and Wieland Brendel. How well do feature visualizations support causal understanding of cnn activations? In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pages 11730–11744, 2021.

# A Glossary

K - The cardinality of the set of neurons of interest, which contains both neurons in intermediate and output layers of the vision classifier

M - A translation matrix that maps the BERT vocabulary to the CLIP embedding space

$\mathcal { L } _ { a c t }$ - Neuron activation maximization loss. Used to determine if the text prompts and images activate the desired neurons of the visual classifier (Eq. 1)

$\mathcal { L } _ { a g g } - \mathbf { A }$ ggregated activation loss. Used also to determine if pseudolabels for $\mathcal { L } _ { m a s k } \left( \mathrm { E q . ~ } 3 \right)$

$\mathcal { L } _ { m a s k } - \mathbf { A }$ cross-entropy loss between the BERT [MASK] token predictions and saved pseudolabels p - The learnable soft prompt in DEXTER

t - A sequence consisting of $\mathrm { \ t { \_ } r { \mathrm { e d } } }$ and N [MASK] tokens

$\mathbf { t _ { e m b } }$ - The embedding of t, including the positional encoding

$\mathbf { t } _ { \mathbf { f i x e d } }$ - Text tokens corresponding to "A picture of $\mathbf { a } "$

V - The BERT vocabulary size, 30522

W - The CLIP vocabulary size, 49408

# B DEXTER Algorithm

Algorithm 1 DEXTER   
Goal: Investigate class c on Vision Classifier f

1: Initialize soft prompt p

2: for iteration = 1 to N do

3: Encode $p + t_{fixed} + [MASK]$ tokens

4: Obtain BERT logits l for masked token prediction

5: Apply Gumbel-Softmax to l to obtain the predicted token $\hat{t}$ 6: Use translation matrix M to convert $\hat{t}$ to the CLIP vocabulary $\hat{t}^{(C)} = \hat{t}M$ 7: Compute CLIP text embedding $e = [t_{fixed}, \hat{t}^{(C)}]$ 8: Generate image with diffusion model d conditioned on e

9: Apply cross-entropy loss $L_{act}$ with ground truth c to $f(d(\mathbf{e}))$ 10: Apply auxiliary loss $L_{mask}$ on l using the previous logit with the highest class activation

11: Update p

12: end for

# C Training hyperparameters

We adopt CLIP as the text encoder and Stable Diffusion v1 $. 4$ as the diffusion model. To reduce inference time, we employ the Latent Consistency Model (LCM) LoRA adapter 3 using 4 inference steps. DEXTER is trained with a batch size of 1 (i.e., one image per iteration) and a learning rate of 0.1 across all tasks. In the following sections (Appendices D, E and F), the term optimization steps refers to the number of iterations performed to optimize the trainable soft prompt. Each optimization step consists of: predicting the masked token, generating an image, and obtaining the visual classifier’s prediction.

In the following, we detail the parameters and hyperparameter settings described in Sections 3.1 and 3.2: for the textual prompt sequence $\left[ \mathbf { t } _ { \mathrm { f i x e d } } , m _ { 1 } , m _ { 2 } , \ldots , { m _ { n } } \right]$ , we use $^ { 6 } \mathsf { a }$ picture of a [MASK].” in the single-word optimization scenario, whereas for multi-word optimization, the fixed prompt is $^ { \mathfrak { s } } \mathfrak { a }$ picture of a [MASK] with [MASK] and [MASK] and [MASK] and [MASK] and [MASK].”

We set the sequence P of soft prompts p to 1 (see Sect. C.1), with each embedding having dimension d = 768, matching BERT’s embedding space. The temperature τ for the Gumbel softmax is kept at

its default value of 1.0.

All experiments ran in half-precision on three H100 GPUs. Prompt optimization takes ∼10 minutes per class, while generating 100 images and the final bias report takes ∼15 seconds without backpropagation. DEXTER is designed for offline, global auditing, with a cost aligned to its goal of delivering comprehensive model insights.

# C.1 Number of soft prompts experiments

We set $P = 1$ to keep the setup minimal and stable. In DEXTER, the soft prompt guides BERT in selecting hard tokens via masked language modeling, rather than encoding semantics directly. Expressiveness comes from composing multi-token prompts (typically 5–6), not from prompt dimensionality. Experiments with P > 1 show that increasing P expands the parameter space and weakens the classifier’s gradient signal, destabilizing optimization in our data-free setting, as shown in Table 5. We report mean and standard deviation for 8 randomly selected classes (4 core, 4 spurious).

Table 5: DEXTER’s performance using diffent number of $P .$ 

<table><tr><td>P</td><td>Spurious</td><td>Core</td><td>Avg</td></tr><tr><td>1</td><td>81.25 ± 18.99</td><td>94.25 ± 4.38</td><td>87.75 ± 11.68</td></tr><tr><td>3</td><td>75.00 ± 29.00</td><td>44.50 ± 35.89</td><td>59.75 ± 32.44</td></tr><tr><td>5</td><td>73.50 ± 25.66</td><td>74.00 ± 42.75</td><td>73.75 ± 34.20</td></tr><tr><td>10</td><td>21.25 ± 31.17</td><td>67.75 ± 40.01</td><td>44.50 ± 35.59</td></tr></table>

# D Visual Explanation Details

This appendix provides additional details on the evaluation of visual explanations generated by DEXTER, complementing the discussion presented in Section 4.2.1. Specifically, we outline the methodology behind CLIP-IQA, our proposed Semantic CLIP-IQA metric and the comparison between DEXTER and prior activation maximization works. Following, a comprehensive analysis of neural feature maximization. These insights further substantiate the findings reported in the main paper and demonstrate the robustness of DEXTER in generating interpretable visual explanations.

# D.1 Quantitative evaluation metrics: CLIP-IQA and Semantic CLIP-IQA

CLIP-IQA [43] is a metric originally introduced to evaluate image quality. In this work, we leverage CLIP-IQA to compare the quality of images generated by DEXTER against those produced by other approaches. Specifically, CLIP-IQA calculates the similarity between generated images and two fixed prompts, returning the probability that an image is closer in similarity to the first prompt rather than the second. The fixed prompts used in the original formulation are “Good photo.” and “Bad photo.”

To incorporate semantic relevance into image quality assessment, we propose Semantic CLIP-IQA. This metric follows the same evaluation protocol as CLIP-IQA but replaces the fixed prompts with class-specific prompts: “Good photo of a [CLASS]” and “Bad photo of a [CLASS]”. This modification ensures that the evaluation captures not only general image quality but also the semantic alignment between the generated images and the target class.

We here also report in Table 6 a quantitative comparison of visual explanation quality across existing activation maximization methods. The comparison highlights the significant performance gap between older GAN-based or optimization-based approaches and diffusion-based methods. In particular, DEXTER demonstrates superior alignment and consistency, validating its ability to generate more meaningful and semantically grounded visual explanations.

# D.2 Neural Feature Maximization

Neural feature maximization refers our strategy to generate images that emphasize the neural features, as defined in SalientImageNet (i.e., the features of the penultimate layer of the model), learned by a neural network for a given class. By optimizing the input image to maximize the activation of specific neural features, this method allows for an interpretable visualization of what the model has learned to recognize as distinctive for a class. This approach is useful for understanding how deep learning models make decisions and can help identify potential biases or spurious correlations in their predictions.

Table 6: Quantitative comparison between DEXTER and existing activation maximization methods for visual explanations 

<table><tr><td>Method</td><td>CLIP-IQA</td><td>Semantic CLIP-IQA</td></tr><tr><td>Yosinski et al. [45]</td><td>0.37 ± 0.19</td><td>0.33 ± 0.23</td></tr><tr><td>Mahendran and Vedaldi [23]</td><td>0.82 ± 0.15</td><td>0.72 ± 0.15</td></tr><tr><td>Nguyen et al. [26]</td><td>0.74 ± 0.21</td><td>0.57 ± 0.30</td></tr><tr><td>DiffExplainer [30]</td><td>0.89 ± 0.09</td><td>0.90 ± 0.05</td></tr><tr><td>DEXTER (ours)</td><td>0.94 ± 0.03</td><td>0.96 ± 0.03</td></tr></table>

In our work, neural feature maximization is performed using 2,000 optimization steps with a single word prompting strategy (see Sect. 4.2.4). We employ RobustResNet50 as the visual classifier for this task. Table 7 reports the 30 classes selected from the SalientImageNet dataset, consisting of 15 classes containing spurious features and 15 with only core features. Specifically, we selected all 15 classes from SalientImageNet in which all top-5 features were marked as spurious. Additionally, we randomly selected another 15 classes from SalientImageNet where all top-5 features were marked as core.

Table 7: Bias and Non-Bias Classes 

<table><tr><td colspan="3">Bias Classes</td><td colspan="3">Non-Bias Classes</td></tr><tr><td>Class_idx</td><td>Class Name</td><td>Bias</td><td>Class_idx</td><td>Class Name</td><td>Bias</td></tr><tr><td>706</td><td>patio</td><td>√</td><td>985</td><td>daisy</td><td>✕</td></tr><tr><td>837</td><td>sunglasses</td><td>√</td><td>291</td><td>lion</td><td>✕</td></tr><tr><td>602</td><td>horizontal bar</td><td>√</td><td>292</td><td>tiger</td><td>✕</td></tr><tr><td>795</td><td>ski</td><td>√</td><td>486</td><td>cello</td><td>✕</td></tr><tr><td>379</td><td>howler monkey</td><td>√</td><td>465</td><td>bulletproof vest</td><td>✕</td></tr><tr><td>890</td><td>volleyball</td><td>√</td><td>574</td><td>golf ball</td><td>✕</td></tr><tr><td>801</td><td>snorkel</td><td>√</td><td>582</td><td>grocery store</td><td>✕</td></tr><tr><td>981</td><td>ballplayer</td><td>√</td><td>635</td><td>magnetic compass</td><td>✕</td></tr><tr><td>746</td><td>puck</td><td>√</td><td>514</td><td>cowboy boot</td><td>✕</td></tr><tr><td>416</td><td>balance beam</td><td>√</td><td>609</td><td>jeep</td><td>✕</td></tr><tr><td>537</td><td>dogsled</td><td>√</td><td>624</td><td>library</td><td>✕</td></tr><tr><td>655</td><td>miniskirt</td><td>√</td><td>764</td><td>rifle</td><td>✕</td></tr><tr><td>810</td><td>space bar</td><td>√</td><td>847</td><td>tank</td><td>✕</td></tr><tr><td>433</td><td>bathing cap</td><td>√</td><td>879</td><td>umbrella</td><td>✕</td></tr><tr><td>785</td><td>seat belt</td><td>√</td><td>971</td><td>bubble</td><td>✕</td></tr></table>

Figure 5 extends Figure 3 by reporting additional comparison between DEXTER and DiffExplainer across different classes (the full set of comparisons is provided in the supplementary). While DiffExplainer effectively maximizes shapes, colors, and textures, its outputs often lack realism, converging on abstract or pattern-like images. In contrast, DEXTER, owing to its textual anchor, more reliably yields semantically coherent representations of the features highlighted by the SalientImageNet heatmaps. This visually confirms the results of the user study reported in Fig. 2.

![](images/18be756cded727e279d90ca1f25f6dd99fca52df53b497995198d21adddfe5d7.jpg)

![](images/75af55bd52d31ae0fd2d5f8d2266cb08077c14a11e05bba27661ecd322971e7c.jpg)

<details>
<summary>text_image</summary>

Label 706 (Patio)
Diffractioner
Different Imagelet
Davier (ours)
Feature 1016
Feature 1633
Feature 194
Feature 451
Feature 654
</details>

![](images/0518b9d95c56816f6cb0641a4f173e69b20bc7a3f48fcd967e9d2514e0afe19d.jpg)

<details>
<summary>text_image</summary>

Label 433 (Bathing Cap)
Differentianer
Differentianer
Differentianer
Dexter (ours)
Feature 121
Feature 1340
Feature 1591
Feature 1609
Feature 755
</details>

![](images/c5c9fbae6264b9111c866e7de4ee425c773c2d0ea88d9b1526f99efe3f5d2687.jpg)

<details>
<summary>text_image</summary>

Label 785 (Seat Belt)
Feature 1010
Feature 108
Feature 116
Feature 1493
Feature 50
Diffexplainer
Salient ImageNet
Dexter (ours)
</details>

Figure 5: Comparison of DiffExplainer and DEXTER with respect to SalientImagenet for the neural feature maximization task, covering different classes. The left column displays classes without spurious features, whereas the right column shows classes with spurious features.

# E Slice Discovery Details

This appendix provides additional details on slice discovery and debiasing using DEXTER, complementing the discussion in Section 4.2.2.

Figure 6 extends Figure 1 to better explain the Slice Discovery pipeline used by DEXTER. We use the Waterbirds dataset as an example, which consists of two classes: Landbirds and Waterbirds. The majority of the landbirds samples are visibly on land, and similarly the majority of the waterbirds are in/near water. While overall this is an easy task for most classifiers, we want to ensure high performance on the small amount of images where the birds are not in their natural habitats (landbirds on water, waterbirds on land).

![](images/055b0b14e960bc3b9ae359937b95ae1caf53738fd89acafe764df82e7eda8e10.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Top K Words"] --> B["Tree forest land leaves"]
    B --> C["Class 0 (Landbirds)"]
    C --> D["Visual Classifier"]
    D --> E["Pond lake sea ocean"]
    E --> F["CLIP Text Encoder"]
    F --> G["Similarity"]
    G --> H["CLIP Image Encoder"]
    H --> I["Dataset"]
    I --> J["Class 0 (Landbirds)"]
    I --> K["Class 1 (Waterbirds)"]
    J --> L["Invistigate Bias of DEXTER"]
    K --> L
```
</details>

Biased and Unbiased slices

Landbird on land

Waterbird on land

Landbird on water

Waterbird on water

Figure 6: Pipeline for Slice Discovery with DEXTER. The top-k words associated with a class are encoded using the CLIP Text Encoder and compared with image encodings from the biased dataset. The appropriate slice is determined based on the similarity between text and image embeddings, along with the ground truth labels of the dataset images. The dataset is then labeled into four categories: waterbird on water, waterbird on land, landbird on land, and landbird on water (see right side).

DEXTER identifies the words explaining an input visual classifier, as presented in Sect. 3. Specifically, given a classifier $f$ and a target class $c ,$ it optimizes a text prompt to extract a set of top-k class words $W _ { C } = w _ { c 1 } , . . . , w _ { c k } \ ( \mathrm { T a b . \ 8 } )$ . Each word $w _ { c i }$ is encoded using CLIP’s text encoder to obtain an embedding $t _ { c i }$ and the class prototype is defined as the average embedding: $\begin{array} { r } { \bar { t _ { c } } = \frac { 1 } { k } \sum t _ { c i } } \end{array}$ Then, following the approach in B2T [16], at inference time, each image $x _ { j }$ of the dataset is encoded into $v _ { j }$ using CLIP’s image encoder. The similarity between the image and the class is then computed as:

$$
\mathrm{CLIP} _ {\text { score }} = \text { similarity } (v _ {j}, \bar {t _ {c}}) \tag {6}
$$

If an image has an higher similarity with the class-words that correspond to its real class it is labeled as unbiased slice (e.g. an image belonging to class landbirds with an higher similarity with landbirds class-words), while if an image has an higher similarity with its counterpart class it is labeled as biased slice (e.g. an image belonging to class landbirds has an higher similarity with waterbirds class-words). Fig. 7 presents the ROC curves obtained using the CLIP-based matching approach on the Waterbirds dataset, in comparison with other SOTA slice discovery methods [16, 8, 12]. The results illustrate how effectively the training images are partitioned into the four groups (waterbird on water, waterbird on land, landbird on land, landbird on water) by the DEXTER-derived words. Finally, we follow the debiased training scheme from [35] to train a debiased classifier on the dataset.

In all our slice discovery experiments, we set k = 4 both for CelebA and Waterbirds. We used 1000 DEXTER optimization steps and single word prompting (see Sect. 4.2.4) as an activation maximization strategy to discover the above words. Tab. 8 reports the discovered words for both datasets, where class 0 of CelebA has been left blank as in B2T [16].

# E.1 Top-k words selection details

The top-k words in Table 8 were obtained by running the DEXTER pipeline multiple times (specifically k=4 timeless). In each run, we recorded the final (single) pseudo-token selected by the masked language model at the end of the optimization. To encourage diversity across runs, we excluded previously selected words from the candidate pool before each new run. However, we also evaluated how using sampling frequency ranking may affect slice discovery results. In particular, we computed results using word frequency–based ranking, evaluating performance with the top ${ \mathrm { k } } = 5$ and ${ \bf k } = 1 0$ most frequent words. We compare these results against our proposed sampling strategy and B2T for reference. As shown in Table 9, our strategy with a single word yields superior performance.

Table 8: Class-wise words discovered by Dexter for Waterbirds and CelebA datasets. 

<table><tr><td>Dataset</td><td>Class 0</td><td>Class 1</td></tr><tr><td>Waterbirds</td><td>fence, woods, jungle, backyard</td><td>seas, sea, lake, harbor</td></tr><tr><td>CelebA</td><td>—</td><td>woman, person, head, girl</td></tr></table>

Table 9: Comparison between multi-run top-k word selection (ours) and single-run frequencyranked top-k selection. 

<table><tr><td>Method</td><td>F1-score Class 0</td><td>F1-score Class 1</td></tr><tr><td>B2T</td><td>0.99</td><td>0.75</td></tr><tr><td>k = 10</td><td>0.98</td><td>0.67</td></tr><tr><td>k = 5</td><td>0.98</td><td>0.66</td></tr><tr><td>ours</td><td>0.99</td><td>0.76</td></tr></table>

![](images/4fc21af1780f478d79bf92dc1b2558ef420edee0e7dddaf0f7d53e7cc2c6d0d9.jpg)

<details>
<summary>line</summary>

| False Positive Rate | ERM Confidence | Domino | Failure Direction | B2T | DEXTER |
| ------------------- | -------------- | ------ | ----------------- | --- | ------ |
| 0.0                 | 0.0            | 0.0    | 0.0               | 0.0 | 0.0    |
| 0.2                 | 0.6            | 0.4    | 0.7               | 0.8 | 0.9    |
| 0.4                 | 0.8            | 0.6    | 0.8               | 0.9 | 0.95   |
| 0.6                 | 0.9            | 0.8    | 0.9               | 0.95| 0.98   |
| 0.8                 | 0.95           | 0.9    | 0.95              | 0.98| 0.99   |
| 1.0                 | 1.0            | 1.0    | 1.0               | 1.0 | 1.0    |
</details>

(b) Waterbird

![](images/9935ffaab032d54ea337fb892ac7c5cff7f0de4bb359cbdd7eb362b3c5bf3b31.jpg)

<details>
<summary>line</summary>

| False Positive Rate | ERM Confidence | Domino | Failure Direction | B2T | DEXTER |
| ------------------- | -------------- | ------ | ----------------- | --- | ------ |
| 0.0                 | 0.0            | 0.0    | 0.0               | 0.0 | 0.0    |
| 0.2                 | 0.8            | 0.4    | 0.9               | 0.95| 0.98   |
| 0.4                 | 0.9            | 0.6    | 0.95              | 0.98| 0.99   |
| 0.6                 | 0.95           | 0.7    | 0.98              | 0.99| 0.995  |
| 0.8                 | 0.98           | 0.8    | 0.99              | 0.995| 0.998 |
| 1.0                 | 1.0            | 1.0    | 1.0               | 1.0 | 1.0    |
</details>

(c) Landbird   
Figure 7: ROC curves for slice discovery on the Waterbirds dataset. Each curve is obtained by comparing the slice assignments predicted by each method with the ground-truth slice labels released with the dataset.

# F Bias Reasoning Details

Given a classifier f and class c, we optimize a prompt $t ^ { * }$ such that a diffusion model generates images $x _ { 1 } , . . . , x _ { M }$ that maximize the classifier’s output for c. Each generated image correctly predicted by the classifier is captioned using a vision-language model $^ { g , }$ producing descriptions $d _ { i } ( g ( x _ { i } ) )$ ). A language model h then summarizes the captions into a single report, describing the features associated with class c and possible biases. This process is fully data-free: both image generation and reasoning depend only on the model’s behavior. For bias reasoning, we generate 50 images classified as the target class using DEXTER over the course of up to 5,000 optimization steps, employing a multi-word prompting strategy. These images are then captioned using ChatGPT-4o mini with the designated Caption System Prompt (detailed below). Subsequently, another instance of ChatGPT-4o mini analyzes these captions using the following Report System Prompt to extract key class attributes and identify potential biases in the model. To generate the captions and the reports with ChatGPT-4o mini we set a temperature of 0.2 to mitigate hallucinations. The max tokens parameter is set to 0, allowing unrestricted response generation. The top\_p is fixed at 1.0, while both the frequency\_penalty and the presence\_penalty are set to 0.0. Finally, the model generates $n = 1$ output.

We conducted multiple experiments on different visual classifiers:

• The reports presented in Fig. 4 and Appendix F.1 were obtained by analyzing a ResNet18 model trained on the FairFaces dataset.   
• Figure 8, instead, provides a bias/not bias categorization for ViT, AlexNet, ResNet50, and RobustResNet50, all trained on the ImageNet-1000 dataset, rather than full bias reasoning reports. Supplementary material contains the full bias reports for all the 30 classes (see Table 7) when analyzing RobustResNet50.

# Caption system prompt

You are an assistant specialized in analyzing pictures.

Your goal is to generate the most accurate and detailed description.

You will be given a picture and you will have to describe it inserting all the key attributes that you find that could be inherent.

Keep the description short (one sentence) while including all the key attributes.

- You have to return only the description of the image. Don’t return other information.   
- Don’t describe the image in a poetic way, just describe the visual elements.   
- Don’t include any information about the image that is not visible.   
- Don’t include any information about the image that is not relevant to the description.   
- If the subject is a person describe the ethnicity features and age features.   
- Keep the description short (one sentence) while including all the key attributes.

Examples:

Input image: [IMG]

Input text: Describe the image

Output: [CAPTION]

# Report system prompt

Context: This task involves analyzing images generated through a process of Class Activation Maximization (CAM). Using a deep neural network trained on a specific dataset, the CAM approach identifies visual patterns that strongly activate a specific class label in the network. To enhance this analysis, we use a trainable soft prompt integrated with a Stable Diffusion model. The soft prompt is optimized to condition Stable Diffusion to create images that maximize the activation of a chosen class. For each generated image, if the predicted class aligns with the target class, a caption describing the image is stored.

Analysis: Given a trainable soft prompt and a Stable diffusion model. We optimize the soft prompt to conditionate stable diffusion to generate pictures that maximize the prediction of a class. If the class predicted is the same of the class we want to maximize, than we store a caption of that picture.

Task: Given a list of sentences that you will receive, you have to reason on the key attributes that the model uses to maximize a specific class. Than you’ve to do a report with the explanation of your reasoning. The report must end saying if the class is biased or not. If you reason that the class is biased you must explain why otherwise you must explain why the class is not biased.

To write the report follow this structure:

\- Title: ### Report on Class Activation Maximization for the Class ’CLASS\_NAME’

\- Hypothesis on Model Bias

the class to analyze will be given by the user with the list of sentences. The shape of the user prompt will be the following:

User: ’Class: [CLASS] - Captions: [LIST OF SENTENCES]’

# F.1 Gender Bias Reports

In this section, we report DEXTER’s reports obtained by analyzing the ResNet18 classifier trained on the FairFaces dataset. As described in Section 4.2.3, we conducted two training procedures for this visual classifier. In the first procedure, we injected a gender bias across the classes, whereas in the second, we employed a balanced training set.

# Report for class 0 (20-29 years) of a biased classifier (overrepresentation of men)

The analysis suggests that the class ’20 to 29 years old people’ may exhibit some bias. While the model captures a range of ethnicities and genders, the emphasis on certain skin tones, facial features, and expressions could lead to a skewed representation of what is considered ’youthful’.   
- Skin Tone Bias: The frequent mention of light brown and medium brown skin tones may indicate a bias towards these tones, potentially neglecting a broader spectrum of skin colors that also represent youth.   
- Facial Feature Bias: The focus on specific facial features that are often idealized in media could lead to a narrow definition of youth, which may not encompass the full diversity of young adults.   
- Gender Representation: The slight focus on the overrepresentation of male subjects could suggest a bias in the training data, which may not equally represent young women.   
In conclusion, while the model successfully identifies attributes associated with the class ’20 to 29 years old people,’ it is essential to recognize the potential biases in representation. The model’s predictions may not fully encompass the diversity of this age group, indicating a need for more balanced training data to ensure equitable representation across all demographics.

# Report for class 0 (20-29 years) of a balanced classifier (equal representation of men and women)

Based on the analysis of the model, it does not appear to exhibit significant bias in identifying the class of 20 to 29 years old individuals. The diversity in hair and eye color, skin tone, and gender representation suggests that the model is capable of recognizing a broad spectrum of individuals within this age group. The focus on well-defined facial features and neutral expressions aligns with common perceptions of young adults, and the inclusion of various ethnic backgrounds indicates an inclusive approach. In conclusion, the class of 20 to 29 years old people, as represented by the model, does not seem to be biased. The attributes identified are consistent with a diverse and inclusive representation of young adults.

# Report for class 1 (50-59 years) of a biased classifier (overrepresentation of women)

Based on the analysis of the model, it can be concluded that the class ’50 to 59 years old people’ is biased. The bias stems from several factors:

- Representation of Age: The model predominantly features youthful characteristics, which may not accurately reflect the physical appearance of individuals aged 50 to 59. This suggests that the model may be influenced by societal standards that equate youthfulness with attractiveness, leading to a misrepresentation of the target age group.   
- Ethnic and Gender Bias: The emphasis on lighter skin tones and the predominance of female subjects indicate a lack of diversity. This could perpetuate stereotypes and fail to represent the true diversity of individuals within the 50 to 59 age range.   
- Societal Stereotypes: The model’s reliance on neutral or serious expressions may reflect societal stereotypes about older individuals, which can further skew the representation of this age group. In conclusion, while the model attempts to maximize the activation of the class ’50 to 59 years old people,’ it does so through a lens that is biased towards certain physical characteristics, expressions, and demographic representations. This highlights the importance of ensuring diverse and representative training data to mitigate bias in machine learning models.

# Report for class 1 (50-59 years) of a balanced classifier (equal representation of men and women)

Based on the analysis, it appears that the class ’50 to 59 years old people’ is not inherently biased. The model successfully captures a range of attributes that reflect the diversity of individuals within this age group. However, there are potential biases in the representation of expressions and the predominance of serious or neutral faces, which may not fully encompass the emotional spectrum of individuals aged 50 to 59. Moreover, the model’s training data could influence its perception of age, potentially leading to stereotypes about how older individuals should look or behave. For instance, the emphasis on serious expressions may inadvertently reinforce the stereotype that older adults are less expressive or joyful. In conclusion, while the class itself is not biased, the representation of certain attributes, particularly expressions and emotional portrayals, may benefit from a broader range of depictions to avoid reinforcing stereotypes associated with aging.

# F.2 Bias Report Quality Assessment Metrics

In this section, we provide additional explanations for the metrics used to evaluate the quality of DEXTER’s bias reports, as shown in Table 2. Following Liu et al. [20], we use the G-eval consistency metric, which involves prompting a large language model (LLM) to assess a text’s factualness, scoring each report on a scale from 1 to 5. Since the original prompts for these metrics were designed for summarization, we create the following G-eval Consistency Prompt to better align with our task.

Following the G-eval format, we use the Mean Opinion Score of LLM $( M O S _ { \mathrm { L L M } } )$ metric. This metric uses the $\bar { M } O S _ { \mathrm { { L L M } } }$ prompt asking the LLM to rate between 1 and 5 the generated reports according to how well the report accurately describes the existence of bias in the classifier. Additionally, the $M O S _ { \mathrm { h u m a n s } }$ score represents the average rating given by participants in the user study.

For both the G-eval and $M O S _ { \mathrm { { L L M } } }$ configurations, we adopt the same settings as Liu et al. [20]. Specifically, we use gpt-4-0613 with a temperature of 2, max\_tokens = 0, top\_p = 1.0, frequency\_penalty = 0, and presence\_penalty = 0.

To enhance response diversity, we generate n = 20 outputs. Furthermore, in both G-eval and MOS LLM system prompts, we provide the same evaluation instructions that were given to human evaluators during the user study, also following the best practices to reduce hallucinations provided by [4].

To assess whether the LLM’s mean-opinion scores are statistically indistinguishable from those assigned by human annotators, we applied an independent-samples two one-sided equivalence test (TOST). The test compared the distribution of ${ \bf M O S } _ { \mathrm { L L M } }$ to that of $\mathrm { M O S } _ { \mathrm { h u m a n s } }$ obtaining a $p \ll 0 . 0 5$ . Thus, within the predefined equivalence bounds (±0.5 for a Likert scale), the LLM’s ratings can be considered statistically equivalent to human judgments, indicating strong agreement between the two sets of scores.

# G-eval Consistency system prompt

\*\*Instructions for Evaluation\*\*: In this task, we trained a visual classifier twice:

1. First, using a dataset with gender-biased images.

2. Second, using a balanced dataset.

Your task is to evaluate how effectively a given textual description identifies the presence or absence of gender bias in the visual classifier:

\- \*\*Visual Classifier\*\*: A machine learning model trained to distinguish between two classes (e.g., Class 1:

"20-29 years old people" vs. Class 2: "50-59 years old people").

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Consistency (1-5) - the factual alignment between the question and the description. A factually consistent description contains statements that are entailed by the question. Annotators were also asked to penalize descriptions that contained hallucinated facts.

Evaluation Steps:

1. Read the question carefully and identify the main topic and key points.   
2. Read the description and compare it to the question. Check if the description covers the main topic and key points of the question, and if it presents them in a clear and logical order.   
3. Assign a score for consistency on a scale of 1 to 5, where 1 is "Bad Description" and 5 is "Excellent Description" based on the Evaluation Criteria.

Example:

Question:

{{Question}}

Description:

{{Description}}

Evaluation Form (scores ONLY):

\- Consistency (1-5):

# MOSLLM system prompt

\*\*Instructions for Evaluation\*\*: In this task, we trained a visual classifier twice:

1. First, using a dataset with gender-biased images.

2. Second, using a balanced dataset.

Your task is to evaluate how effectively a given textual description identifies the presence or absence of gender bias in the visual classifier:

\- \*\*Visual Classifier\*\*: A machine learning model trained to distinguish between two classes (e.g., Class 1: "20-29 years old people" vs. Class 2: "50-59 years old people").

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Mean Opinion score (1-5) - Assign a rating to the description based on its effectiveness:

- 1: Bad description (fails to identify bias or provide relevant details).   
- 2: Poor description (some effort to address bias, but lacks clarity or completeness).   
- 3: Neutral (adequate but not insightful; partially addresses bias).   
- 4: Good description (clear and mostly thorough in addressing bias)   
- 5: Excellent description (comprehensive, clear, and detailed in identifying bias).

Evaluation Steps:

1. Read the question carefully and identify the main topic and key points.   
2. Read the description and compare it to the question. Check if the description covers the main topic and key points of the question, and if it presents them in a clear and logical order.   
3. Assign a score for bias identification on a scale of 1 to 5, where 1 is "Bad Description" and 5 is "Excellent Description" based on the Evaluation Criteria.

Example:

Question:

{{Question}}

Description:

{{Description}}

Evaluation Form (scores ONLY):

\- Rating (1-5):

# F.3 Bias identification of multiple vision classifiers

In Fig. 8 we report an example of disparity analysis across ViT, AlexNet, ResNet50, and RobustResNet50 for multiple ImageNet classes. By revealing class-specific biases and failure patterns, DEXTER helps identify where each model struggles and can guide data collection efforts when biases are consistently observed across classifiers. To select key neural features for these classifiers, we ranked penultimate-layer neurons by their weights to each class and selected the top 5, mirroring the Salient ImageNet method but without any training data.

![](images/de59e0f11b1b25eb52ce0e0ca0fe33ad793d473c5e6b058ccc39a0269feb8c5a.jpg)

<details>
<summary>text_image</summary>

Tiger (292)
Jeep (609)
Snorkel (801)
Baseball player (981)
VIT
Not biased
Not biased
Biased
Biased
AlexNet
Not biased
Biased
Biased
ResNet50
Not biased
Biased
Biased
Biased
RobustResNet50
Not biased
Not biased
Biased
Biased
</details>

Figure 8: Bias analysis across classifiers and SalientImageNet classes. Each cell shows DEXTERgenerated visual explanations and bias assessments (“Biased” or “Not Biased”). While Lion is unbiased across models, Jeep is biased in ResNet50 but not in its robust version. Baseball Player remains biased in all models, suggesting dataset-level bias.

# G Additional details on Ablation Study

![](images/5d842a448abe9481c208920c29206117e53035d34ca81c6a1c0443bda9aebc1c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1st iteration<br>tfixed [Library"]] --> B["CLIP Text Encoder"]
    B --> C["Stable Diffusion"]
    C --> D["Visual Classifier"]
    D --> E["Predicted Class: Library ✓"]
    
    F["ith iteration<br>tfixed [Library"]] --> G["CLIP Text Encoder"]
    G --> H["Stable Diffusion"]
    H --> I["Visual Classifier"]
    I --> J["Predicted Class: Grocery Store ✗"]
    
    K["100th iteration<br>tfixed [Library"]] --> L["CLIP Text Encoder"]
    L --> M["Stable Diffusion"]
    M --> N["Visual Classifier"]
    N --> O["Predicted Class: Library ✓"]
    
    style A fill:#cce5ff,stroke:#333
    style F fill:#cce5ff,stroke:#333
    style K fill:#cce5ff,stroke:#333
```
</details>

Figure 9: Procedure to compute the percentage of activations.

In this section, we provide further details on how we compute the activation score reported in Table 3 and Table 4. As shown in Figure 9, once the word(s) are obtained at the end of the optimization process, we prompt Stable Diffusion to generate 100 distinct images using a fixed prompt $\mathbf { t } _ { \mathrm { f i x e d } }$ (from the optimization stage) concatenated with the discovered word(s) ˆtC. We then sum all inference steps in which the visual classifier predicts the target class. Figure 10 illustrates examples of the different text-prompt strategies, described in Table 3, alongside their corresponding activation scores for the class Tiger and the class Snorkel.

![](images/80e9a7f5169e3495e14377cbf8d4ff604fdc0ca55b7da3fd73245877f6650436.jpg)

<details>
<summary>bar_stacked</summary>

| Method | Class Tiger (Not spurious) | Class Snorkel (Spurious) |
| --- | --- | --- |
| Baseline | 100% | 41% |
| ChatGPT Baseline | 100% | 61% |
| DiffExplainer | 100% | 26% |
| DEXTER_mw(ours) | 100% | 84% |
</details>

Figure 10: Examples of corresponding text prompts and generated images for class activation maximization. For a non-spurious class like tiger, all generated images easily activate the target class. However, in the case of the snorkel class, DEXTER is able to generate significantly more images that maximize the activation, and exposes the other features the model pays attention to.

# H User Study Details

What kind of similarity is there between the red area of the target image and the generated image? Select one of the option below.

![](images/6b2b83c425d46e96aa0bd51f15cc54c7c1e0c5eb02ec09ecfc61bd7582127f21.jpg)

<details>
<summary>text_image</summary>

Target
Generated Image
Shape Colour Texture Material Semantical Context Several elements None
Shape: similar shapes
Colour: similar colour
Texture: similar texture
Material: similar material
Semantical: similar concepts e.g. book, newspaper
Context: Similar context e.g. swimming pool, sea
Several elements: the pictures share several kind of similiraties (e.g. shape, colour)
Please provide a brief explanation of your choice.
</details>

Figure 11: Example of a question from the first section of the user study.

Each user study consisted of two parts. The first part aimed to assess the ability of DEXTER to visually highlight the focus of the target classifier for a specific neural feature, while the second part evaluated the accuracy of the textual bias reports generated by DEXTER.

In the first part of the study, each participant was presented with 15 randomly images (explaining specific feature in SalientImageNet) drawn from the outputs of DiffExplainer and DEXTER. An example is shown in Fig. 11. Participants were asked to assess whether the generated image accurately represented the attended region of the real image based on attributes such as color, shape, texture, material, context, or semantic similarity. Additionally, two extra options were provided: “Several Elements” for cases where multiple similarity criteria were met, and “None” for instances where none of the criteria applied.

The second part of the study aimed to evaluate the ability of DiffExplainer reports to detect the presence of bias in the target classifier. The target classifier was trained both with and without gender bias, and participants were presented with all four reports generated by DEXTER. Examples of questions for class 0 (20–29 years old) are provided in Fig. 12 and Fig. 13, corresponding to the models trained with and without the injected gender bias, respectively. Participants were asked to rate, on a scale from 1 to 5, the extent to which the generated report reflected the presence or absence of bias.

The user study was conducted with 100 distinct MTurk workers, with an average completion time of 10.32 minutes. Each feature explanation in the first part received an average of 10.38 responses for DiffExplainer and 10.82 for DEXTER. To minimize response variability, participants were required to provide a textual justification for their answers in both sections of the questionnaire before submission. The compensation for each worker was \$0.50.

# I Diffusion and LLM’s Hallucination evaluation

# I.1 Randomness in the diffusion generation process

To assess the stability of Stable Diffusion, we performed intra-class evaluation using DEXTER’s final prompt. Across three independent runs (100 images each), we measured activation scores with the target model. If the image generation is unstable, scores will vary significantly. Instead, our consistent results indicate stability. We report mean and standard deviation for 8 randomly selected classes (4 core, 4 spurious) across three independent runs (Tab. 10).

Givena visual classifier trained to recognize individuals aged 20 to 29, where a gender bias was introduced through an overrepresentation of men,does the description below accurately identify the presence of gender bias in your opinion?

Theanalysis suggests that the\_class‘20 to 29 years old people'may exhibit some bias.While the modelcaptures arange of ethnicities and genders,the enphasis oncertain skin tones，facial features,and expressionscould lead toa skevedrepresentation of what isconsidered'youthful'. -SkinTone Bias:The frequent mention of light brownandmediumbron skin tonesmayindicatea biastowardsthese tones,potentially neglectinga broader spectrunof skincolorsthatalso representyouth. -Facial Feature Bias:The focus on specific facial features that are often idealized in nedia could leadtoanarrowdefinitionofvouth，whichmaynotencompassthefulldiversityofyoungadults. -Gender Representation:The slight focus on the overrepresentation of male subjects could suggest a biasin the training data,which nay not equally represent young women. ;Inconclusion，whilethemodel successfullyidentifiesattributesassociated withthe class ‘20 to 29yearsold people,itisessential torecognize thepotential biasesinrepresentation.The model'spredictionsnaynot fullyencompassthediversity of thisagegroup,indicatinganeed for more balanced training data to ensure equitable representation acrossall demographics.

Badepnet

1 Bad description 2 Poor description 3 Average description 4 Good description 5 Excellent description

Figure 12: Example of a question from the second section of the user study (biased).

Givena visual clasifier trained to recognize individuals aged 20 to 29,where gender bias was removed by ensuring equal representation of men and women,does the description below accurately identify the absence of gender bias in your opinion?

Basedontheanalysisoftheodelitdeotaeartoebitsignicantbiasiidentifing theclass of 20to29 yearsold individuals.Thediversityin hairand eye color，skin tone,and genderrepresentationsuggeststhat the model iscapableof recognizinga broad spectrum of individuals withinthisagegroup.Thefocusonwell-defined facialfeaturesandneutral expressions aligns with comnon perceptions of young adults，and the inclusion of various ethnic backgrounds indicatesan inclusive approach.In conclusion,the class of 20 to 29 years old people，as represented by themodel，does not seem tobe biased.Theattributesidentified are consistent with adiverseandinclusiverepresentation ofyoungadults.

Bad description Exelentdescription

1 Bad description 2 Poor description 3Average descrition 4Goodescription 5 Excellent description

Figure 13: Example of a question from the second section of the user study (not biased).   
Table 10: Intra-class evaluation of randomness in Stable Diffusion generation process. 

<table><tr><td>Type</td><td>Class</td><td>Avg</td></tr><tr><td rowspan="4">Spurious</td><td>890</td><td>55.33 ± 2.49</td></tr><tr><td>795</td><td>79.00 ± 1.63</td></tr><tr><td>655</td><td>96.66 ± 0.94</td></tr><tr><td>706</td><td>100.0 ± 0.00</td></tr><tr><td rowspan="4">Core</td><td>291</td><td>100.0 ± 0.00</td></tr><tr><td>486</td><td>95.66 ± 1.24</td></tr><tr><td>514</td><td>86.66 ± 1.24</td></tr><tr><td>624</td><td>95.66 ± 1.88</td></tr></table>

# I.2 Bias propagation

To verify that DEXTER’s outputs align with the classifier—rather than reflecting biases from BERT or Stable Diffusion—we ran a robustness test. We manually injected spurious cues into prompt initialization (e.g., replacing the starting auxiliary pseudo-target with lion for the class tiger) to simulate strong upstream bias. Then, for each class in SalientImageNet subset of our paper (split into “core” and “spurious” categories), we generated 100 images and measured how often the classifier activated the correct class. We report average classifier activation scores (as defined in Tab. 3 of the paper) for spurious, core classes, and their average. These results in Tab. 11 show that even with adversarially biased prompts, DEXTER recovers class-relevant visual patterns, aligning with the classifier. Classifier-driven optimization helps correct upstream bias and grounds both visual and textual outputs in model behavior.

Table 11: Comparison of Spurious, Core, and Average scores under different bias conditions. 

<table><tr><td>Condition</td><td>Spurious</td><td>Core</td><td>Avg</td></tr><tr><td>Injected Bias</td><td>65.3</td><td>88.4</td><td>76.8</td></tr><tr><td>No Bias</td><td>63.0</td><td>87.6</td><td>75.4</td></tr></table>

We also evaluated how DEXTER compensates for bias propagation. Specifically, we recorded the sequence of words selected as pseudotargets during optimization. This was done for both the biased setting and the standard setting, where the initial pseudotarget is chosen randomly. As shown in the Table 12, in both cases DEXTER consistently converges toward the most meaningful word, with the selection path gradually shifting from a random concept to the intended target. This trajectory suggests that the optimization process does not simply lock onto the first spurious or core concept it encounters, but instead explores a range of semantically plausible candidates before progressively refining the prompt toward a stable and interpretable solution, indicating a non-greedy and convergent behavior rather than an early commitment driven by the mask pseudo-label loss. Convergence analysis through keyword trajectory during the DEXTER optimization process. The Table 12 illustrates how DEXTER transitions from initially random concepts to progressively more semantically coherent and domain-specific words, demonstrating convergence toward a stable explanation.

# I.3 Analysis of LLM hallucination impact

Given that large language models can generate statements influenced by their prior knowledge rather than by factual evidence, we systematically assess the fidelity of our reports and how much they are aligned with the visual classifier decision-making process. We aim to ensure that each report highlights genuine visual patterns exploited by the classifier rather than hallucinated elements. Concretely, for each of the 30 SalientImageNet classes in our experimental setup (Tab. 7), we verify whether the single most salient visual cue extracted from the corresponding DEXTER report is truly grounded. Given a class c we form

$$
\begin{array}{c} \text { BASELINE   PROMPT: }" a \text { picture   of } a [ c ]", \\ \text { CUE   PROMPT: }" a \text { picture   of } a [ c ] \text { with } [ \text { cue } ] _ {0} \dots \text { and } ^ {\prime} [ \text { cue } ] _ {n}, \end{array}
$$

where n is the total number of cues obtained automatically from the generated reports via a GPT–4o mini class-cues extractor. Then, we generate 100 images per prompt with Stable–Diffusion and measure the Activation Score (AS) on the frozen RobustResNet50 used throughout the paper (4.2.4). A cue is grounded when $\mathrm { A S _ { c u e } } > \mathrm { A S _ { b a s e l i n e } } ;$ otherwise it is neutral or wrong.

Statistical significance & confidence interval. For each class we measure $\Delta _ { i } = \mathrm { A S } _ { \mathrm { c u e } , i } -$ ${ \mathrm { A S } } _ { \mathrm { b a s e l i n e } , i } .$ . Across the 30 classes we obtain

$$
\bar {\Delta} = 1 6. 3 3 \mathrm{pp}, \quad s _ {\Delta} = 2 3. 8 4 \mathrm{pp}.
$$

The two-sided 95% confidence interval is

$$
\bar {\Delta} \pm t _ {0. 9 7 5, 2 9} \frac {s _ {\Delta}}{\sqrt {3 0}} = [ 7. 4 3, 2 5. 2 4 ] \mathrm{pp}.
$$

Both a paired t-test $( t ( 2 9 ) = 3 . 7 9 , p = 7 . 8 \times 1 0 ^ { - 4 } )$ and a Wilcoxon signed-rank test $( W = 1 9 , p =$ $2 . 9 \times \mathrm { { i 0 ^ { - 4 } } } )$ confirm that the improvement is statistically significant.

Aggregate results. Table 13 contrasts the mean Activation Scores for the two prompts. Adding the report cue raises the mean score from 64.73% to 81.06% ( +16.33 pp).

• 20/30 classes (67%) improve (peak +81 pp for miniskirt), confirming that the cues capture genuinely discriminative evidence.   
• 7 classes already achieve 100 % with the baseline prompt; the cue therefore leaves performance unchanged.

Table 12: Word-wise convergence path for biased vs unbiased initialized prompt. 

<table><tr><td colspan="2">Grocery Store (582)</td><td colspan="2">Patio (706)</td></tr><tr><td>Biased</td><td>Not Biased</td><td>Biased</td><td>Not Biased</td></tr><tr><td>motel</td><td>fan</td><td>restaurant</td><td>head</td></tr><tr><td>library</td><td>coffin</td><td>school</td><td>book</td></tr><tr><td>wedding</td><td>factory</td><td>house</td><td>man</td></tr><tr><td>party</td><td>sale</td><td>pool</td><td>the</td></tr><tr><td>restaurant</td><td>factory</td><td>residence</td><td>republic</td></tr><tr><td>party</td><td>distillery</td><td></td><td>tree</td></tr><tr><td>wedding</td><td>market</td><td></td><td>window</td></tr><tr><td>motel</td><td>supermarket</td><td></td><td>house</td></tr><tr><td>bakery</td><td></td><td></td><td>courtyard</td></tr><tr><td>supermarket</td><td></td><td></td><td>terrace</td></tr><tr><td></td><td></td><td></td><td>patio</td></tr></table>

Table 13: Activation Scores averaged over 30 classes (↑/↓: cue better/worse than baseline). 

<table><tr><td>Prompt</td><td>Mean AS (%)</td><td> $\Delta$ </td><td>Class split</td></tr><tr><td>Baseline</td><td>64.73</td><td>—</td><td></td></tr><tr><td>Cue (report)</td><td>81.06</td><td>+16.33</td><td>20↑ 3↓ 7=30</td></tr></table>

# • 3 classes degrade:

<table><tr><td>bubble</td><td> $\Delta = -5$  pp; the cue is correct but pertains only to a subset of bubble instances.</td></tr><tr><td>tanks</td><td> $\Delta = -1$  pp ( $100 \% \rightarrow 99 \%$ ); a negligible loss.</td></tr><tr><td>rifle</td><td> $\Delta = -17$  pp; the cue “individuals in dark clothing” marking the sole clear failure.</td></tr></table>

Overall assessment. DEXTER explanations are therefore strongly grounded: in 27 out of 30 classes the cue is beneficial or neutral, and only one class (rifle) exhibits evidence of hallucination. In the two minor degradations (bubble,tanks) the reports still isolate genuine—though partial—visual cues.

Evaluation Implications. The consistent increase in classifier confidence when report-derived cues are injected into the prompt demonstrates that DEXTER’s explanations faithfully align with the model’s decision boundary, rather than reflecting spurious or hallucinatory artifacts. Together with the text-based metrics reported in Table 2 for the FairFaces dataset (G-Eval consistency, STS, MOS), these findings confirm that DEXTER delivers genuinely grounded insights into the visual classifier’s decision-making process.

Pipeline Robustness. Furthermore, this evaluation implicitly validates the collaborative operation of all pretrained components (BERT, CLIP, Stable Diffusion, VLM) in our optimization pipeline. The target visual classifier guides the prompt optimization. BERT selects discriminative keywords, the diffusion model generates contextually relevant images, and the captioning VLM produces faithful captions. As a result, the end-to-end process avoids propagating biases or hallucinations from intermediate models and yields explanations that are solidly grounded in the classifier’s behavior.

# I.4 Baseline prompts vs Cue prompts

This section, referring to Sec. I.3, reports the system prompt used to extract the relevant visual cues from DEXTER’s textual explanations. Furthermore, Table 14 compares class-wise activation scores produced by the baseline prompt “A picture of a [CLASS]” and by the CUE-enriched prompt. Large gains appear for classes that were poorly activated in the baseline (e.g., space bar improves from $6 \to 5 7$ , hockey puck from $0  7 9 )$ . The results demonstrate that adding concise semantic cues markedly improves class-specific guidance and reduces hallucinations during image generation.

# Report’s cues extractor system prompt

You are an assistant that reads a bias-analysis report about an ImageNet class and extracts concrete visual cues that the report claims are important for that class. Your task

1. Identify up to 5 key visual phrases (2–5 words each) that:

• are explicitly mentioned in the report;   
• describe tangible elements that can appear in an image (objects, attributes, background, actions);   
• are likely to trigger the classifier according to the report.

2. Return your answer in JSON with two fields:

“‘json "key-phrases": ["phrase1", "phrase2", ...], "full-prompt": "a picture of a <CLASS> with "phrase1" and "phrase2" and

Table 14: Activation maximization scores for each class: comparison between the baseline prompt “A picture of a [CLASS]” and the CUE prompt obtained from the DEXTER’s reports. 

<table><tr><td>Class</td><td>Baseline prompt</td><td>AS</td><td>CUE prompt</td><td>AS</td></tr><tr><td>volleyball</td><td>A picture of a volleyball</td><td>66</td><td>A picture of a volleyball with young women playing and mixed-gender teams and variety of scenarios</td><td>78</td></tr><tr><td>space bar</td><td>A picture of a space bar</td><td>6</td><td>A picture of a space bar grid layouts with framed artworks and black-and-white imagery</td><td>57</td></tr><tr><td>umbrella</td><td>A picture of an umbrella</td><td>95</td><td>A picture of an umbrella with vibrant outdoor scenes and garden scenes</td><td>100</td></tr><tr><td>baseball player</td><td>A picture of a baseball player</td><td>98</td><td>A picture of a baseball player with female athletes and softball context</td><td>100</td></tr><tr><td>bubble</td><td>A picture of a bubble</td><td>100</td><td>A picture of a bubble with whimsical themes</td><td>95</td></tr><tr><td>balance beam</td><td>A picture of a balance beam</td><td>0</td><td>A picture of a balance beam with athletic attire and physical activities and indoor training environments and group dynamics and artistic performances</td><td>27</td></tr><tr><td>cowboy boot</td><td>A picture of a cowboy boot</td><td>83</td><td>A picture of a cowboy boot with various types of boots</td><td>96</td></tr><tr><td>patio</td><td>A picture of a patio</td><td>100</td><td>A picture of a patio with modern architecture and expansive outdoor spaces and affluent homes and specific landscaping styles</td><td>100</td></tr><tr><td>tank</td><td>A picture of a tank</td><td>100</td><td>A picture of a tank with camouflage and military environment and armored vehicle</td><td>99</td></tr><tr><td>dark glasses</td><td>A picture of dark glasses</td><td>8</td><td>A picture of dark glasses with facial hair and an older man</td><td>20</td></tr><tr><td>daisy</td><td>A picture of a daisy</td><td>100</td><td>A picture of a daisy with vibrant colors and natural contexts and red daisies</td><td>100</td></tr><tr><td>howler monkey</td><td>A picture of a howler monkey</td><td>0</td><td>A picture of a howler monkey with lush, green environments and natural habitat</td><td>32</td></tr><tr><td>tiger</td><td>A picture of a tiger</td><td>100</td><td>A picture of a tiger with striped fur and natural habitat and majestic posture</td><td>100</td></tr><tr><td>library</td><td>A picture of a library</td><td>76</td><td>A picture of a library with books and organized indoor area and spacious room with furniture and people</td><td>97</td></tr><tr><td>seat belt</td><td>A picture of a seat belt</td><td>56</td><td>A picture of a seat belt with human subjects</td><td>73</td></tr><tr><td>rifle</td><td>A picture of a rifle</td><td>85</td><td>A picture of a rifle with individuals in dark clothing and jackets and sunglasses and aiming</td><td>68</td></tr><tr><td>grocery store</td><td>A picture of a grocery store</td><td>84</td><td>A picture of a grocery store with diversity of food items and presence of people and colorful displays</td><td>95</td></tr><tr><td>snorkel</td><td>A picture of a snorkel mask</td><td>41</td><td>A picture of a snorkel mask with young individuals and sharks</td><td>94</td></tr><tr><td>dogsled</td><td>A picture of a dogsled</td><td>98</td><td>A picture of a dogsled with snowy landscapes and human-animal interaction and specific dog breeds</td><td>100</td></tr><tr><td>magnetic compass</td><td>A picture of a magnetic compass</td><td>6</td><td>A picture of a magnetic compass with silver compass pendant and craftsmanship and design</td><td>26</td></tr><tr><td>horizontal bar</td><td>A picture of a horizontal bar</td><td>6</td><td>A picture of a horizontal bar in sports and athletic environment</td><td>8</td></tr><tr><td>ski</td><td>A picture of a ski</td><td>56</td><td>A picture of a ski with young individuals with ski attire</td><td>100</td></tr><tr><td>miniskirt</td><td>A picture of a miniskirt</td><td>19</td><td>A picture of a miniskirt with women wearing miniskirts</td><td>100</td></tr><tr><td>lion</td><td>A picture of a lion</td><td>100</td><td>A picture of a lion with distinctive physical traits and social behavior and natural habitat and human interactions</td><td>100</td></tr><tr><td>hockey puck</td><td>A picture of a puck</td><td>0</td><td>A picture of a puck with ice hockey gameplay and player confrontations</td><td>79</td></tr><tr><td>swimming cap</td><td>A picture of a swimming cap</td><td>92</td><td>A picture of a swimming cap with children in joyful scenarios with adults and certain hair types and smiling expressions</td><td>99</td></tr><tr><td>bulletproof vest</td><td>A picture of a bulletproof vest</td><td>67</td><td>A picture of a bulletproof vest with casual attire</td><td>89</td></tr><tr><td>cello</td><td>A picture of a cello</td><td>100</td><td>A picture of a cello with musical instrument</td><td>100</td></tr><tr><td>golf ball</td><td>A picture of a golf ball</td><td>100</td><td>A picture of a golf ball with spherical objects and white color</td><td>100</td></tr><tr><td>jeep</td><td>A picture of a jeep</td><td>100</td><td>A picture of a jeep, landrover with off-road capability</td><td>100</td></tr><tr><td>Mean</td><td>—</td><td>64.73</td><td>—</td><td>81.06</td></tr></table>

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction of the DEXTER paper clearly and accurately reflect the paper’s contributions and scope. The three central use cases mentioned in the abstract (activation maximization, slice discovery and debiasing, and bias explanation) are all thoroughly addressed and empirically validated in the main body. Additionally, the introduction carefully outlines both the limitations of existing work and how DEXTER addresses them, reinforcing that the scope of the claims is realistic and well-motivated.

# Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.   
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have added a "Limitations" section in the document that provides areas where DEXTER could benefit from improvement.

# Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
• The authors are encouraged to create a separate "Limitations" section in their paper.   
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.   
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA]

Justification: The paper does not include any formal theoretical results, theorems, or proofs. Its contributions are methodological and empirical, focusing on the design, implementation, and evaluation of the DEXTER framework rather than on theoretical analysis.

Guidelines:

• The answer NA means that the paper does not include theoretical results.   
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.   
• All assumptions should be clearly stated or referenced in the statement of any theorems.   
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.   
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
• Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper includes all necessary information to reproduce its main experimental results, with detailed implementation instructions provided in the appendix. It specifies datasets, models, evaluation metrics, and experiment settings. The appendix covers optimization strategies, prompts, training procedures, and evaluation protocols,

Guidelines:

• The answer NA means that the paper does not include experiments.   
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.   
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.   
(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.   
(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: While the datasets are publicly available the code will be released upon acceptance.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.   
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.   
• While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.   
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).   
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The whole experimental setting has been defined in both the main paper and the appendix.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.   
• The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All main tables in the paper report results across different random seeds using mean and standard deviation. To assess whether the LLM-generated mean opinion scores $( \mathbf { M O S } _ { \mathbf { L L M } } )$ were statistically equivalent to those provided by human raters $( \mathrm { M O S } _ { \mathrm { h u m a n s } } )$ , we conducted a Two One-Sided Test (TOST) on the paired MOS values presented in Table 2. For the hallucination evaluation (Sec. I.3), we report both statistical significance and confidence intervals, validated using a paired t-test and a Wilcoxon signed-rank test.

# Guidelines:

• The answer NA means that the paper does not include experiments.   
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.   
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
• The assumptions made should be given (e.g., Normally distributed errors).   
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.   
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper provides information about hardware and time execution in the "Limitations" section of the main paper and in the appendix.

# Guidelines:

• The answer NA means that the paper does not include experiments.   
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research adheres to the NeurIPS Code of Ethics. It does not involve human subjects beyond anonymized user studies conducted via established platforms, uses publicly available datasets, and emphasizes transparency, fairness, and bias identification. The proposed method (DEXTER) is explicitly designed to improve model interpretability and promote responsible AI use by revealing biases and spurious correlations in classifiers.

# Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: Although the paper does not explicitly discuss the social impacts of DEXTER, it implicitly highlights, as indicated in the introduction, the growing need for tools capable of interpreting AI models as they become increasingly sophisticated. In this context, DEXTER aims to enhance the trustworthiness of these models, fostering a positive impact on the community that relies on them.

# Guidelines:

• The answer NA means that there is no societal impact of the work performed.   
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.   
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes]

Justification: The paper discusses in the "Limitations" section the potential risk of generating NSFW content with Stable Diffusion and explains that a safety checker has been implemented to mitigate this issue.

# Guidelines:

• The answer NA means that the paper poses no such risks.   
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification:

Table 15: Assets used and licence information. 

<table><tr><td>Asset</td><td>Type</td><td>License</td><td>Github / URL</td><td>Citation/Reference</td></tr><tr><td>SalientImagenet</td><td>data</td><td>non-commercial research</td><td>singlasahil14/salient_imagenet</td><td>[40]</td></tr><tr><td>Waterbirds</td><td>data</td><td>non-commercial research</td><td>kohpangwei/group_DRO</td><td>[35]</td></tr><tr><td>CelebA</td><td>data</td><td>non-commercial research</td><td>https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html</td><td>[21]</td></tr><tr><td>FairFaces</td><td>data</td><td>CC BY 4.0</td><td>joojs/fairface</td><td>[14]</td></tr><tr><td>BERT</td><td>Pretrained Model</td><td>Apache license 2.0</td><td>https://huggingface.co/google-bert/bert-base-uncased</td><td>[5]</td></tr><tr><td>CLIP</td><td>Pretrained Model</td><td>MIT license</td><td>openai/CLIP</td><td>[31]</td></tr><tr><td>Stable Diffusion 1.5</td><td>Pretrained Model</td><td>CreativeML Open RAIL-M</td><td>https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5</td><td>[34]</td></tr><tr><td>GPT-4o</td><td>LLM</td><td>API-based use under OpenAI terms</td><td>https://platform.openai.com/docs/overview</td><td>OpenAI</td></tr></table>

# Guidelines:

• The answer NA means that the paper does not use existing assets.   
• The authors should cite the original paper that produced the code package or dataset.   
• The authors should state which version of the asset is used and, if possible, include a URL.   
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.   
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.   
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.   
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.   
• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

# 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: While the paper introduces DEXTER as a novel framework and reports experimental results using standard datasets, it does not release new datasets or models as our proposed method aims to explain a pretrained model (classifier) using other pretrained frozen models.

# Guidelines:

• The answer NA means that the paper does not release new assets.   
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
• The paper should discuss whether and how consent was obtained from people whose asset is used.   
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

# Answer: [Yes]

Justification: All information about the user study conducted in this work is provided in both the main paper and the appendix, including the results and the questionnaire administered to human evaluators. The appendix contains the full text of the questionnaire as well as details about participant compensation for each HIT.

# Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

# Answer: [NA]

Justification: The paper does not involve research with human subjects.

# Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

# Answer: [Yes]

Justification: The paper explicitly describes the use of large language models (LLMs) as a central and original component of its methodology. In DEXTER, LLMs are used to generate textual prompts and provide human-interpretable explanations of visual classifier behavior.

# Guidelines:

• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.   
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.