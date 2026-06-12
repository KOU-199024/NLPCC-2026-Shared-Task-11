# Mitigating Object Hallucination in Large Vision-Language Models via Image-Grounded Guidance

Linxi Zhao \* 1 Yihe Deng \* 2 Weitong Zhang 3 Quanquan Gu 2

# Abstract

The advancement of Large Vision-Language Models (LVLMs) has increasingly highlighted the critical issue of their tendency to hallucinate non-existing objects in the images. To address this issue, previous works focused on using specially curated datasets or powerful LLMs to rectify the outputs of LVLMs. However, these approaches require either costly training or finetuning, or API access to proprietary LLMs for post-generation correction. In response to these limitations, we propose Mitigating hallucinAtion via image-gRounded guIdaNcE (MARINE), a framework that is both training-free and API-free. MARINE effectively and efficiently reduces object hallucinations during inference by introducing image-grounded guidance to LVLMs. This is achieved by leveraging open-source vision models to extract object-level information, thereby enhancing the precision of LVLM-generated content. Our framework’s flexibility further allows for the integration of multiple vision models, enabling more reliable and robust object-level guidance. Through comprehensive evaluations across 5 popular LVLMs with diverse evaluation metrics and benchmarks, we demonstrate the effectiveness of MARINE, which even outperforms existing fine-tuning-based methods. Remarkably, it reduces hallucinations consistently in GPT-4V-assisted evaluation while maintaining the detailedness of LVLMs’ generations. We release our code at https://github.com/ Linxi-ZHAO/MARINE.

\*Equal contribution . 1Department of Computer Science, Cornell University, Ithaca, NY, USA 2Department of Computer Science, University of California, Los Angeles, CA, USA 3School of Data Science and Society, UNC, Chapel Hill, NC, USA. Correspondence to: Quanquan Gu <qgu@cs.ucla.edu>.

Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

# 1 Introduction

The advent of Large Language Models (LLMs) has motivated advancements in extending their remarkable capabilities to multimodal data. Grounded in the development of pre-trained vision-language models (Radford et al., 2021; Jia et al., 2021; Alayrac et al., 2022) that align visual and textual embedding spaces, Large Vision Language Models (LVLMs) have gained substantial attention in both architectural development (Liu et al., 2023d; Zhu et al., 2023; Ye et al., 2023; Dai et al., 2023a; Gao et al., 2023), alignment (Yu et al., 2024; Zhou et al., 2024; Deng et al., 2024) and benchmarking datasets (Xu et al., 2023; Lu et al., 2024; Zhang et al., 2024a). However, similar to the hallucination issues in textual LLMs (Ji et al., 2023), where irrelevant content is generated with input prompts, LVLMs face a specific challenge known as object hallucination: generating nonexisting objects for a given image (Li et al., 2023b; Wang et al., 2023b; Zhou et al., 2023; Fu et al., 2023; Lovenia et al., 2023; Jing et al., 2023). Such a problem is particularly concerning as it compromises the model’s accuracy and reliability, especially considering the growing application of LVLMs to safety-critical downstream tasks such as medical imaging (Chambon et al., 2022; Bazi et al., 2023).

In response to the pressing issue of object hallucinations in LVLMs, early attempts (Liu et al., 2023a;b; Gunjal et al., 2023; Wang et al., 2023a) focused on addressing the bias by curating high-quality datasets for fine-tuning or leveraging advanced GPT queries (Yin et al., 2023), such as GPT-4, to post-process the generated captions. However, these methods can be infeasible to implement. For instance, creating extensive, high-quality datasets for fine-tuning LVLMs is costly and requires significant human annotation. Additionally, relying on advanced GPT models for post-processing is expensive and can raise privacy concerns, especially in sensitive fields like medical imaging. Most importantly, these approaches do not address the intrinsic causes of object hallucination in LVLMs.

In this paper, we investigate the intrinsic causes of object hallucination in LVLMs. Specifically, these deficiencies may stem from the three main components of the LVLMs: 1) insufficient visual context provided by the visual encoder (Zhang et al., 2023b), 2) distortion or loss of visual information during the projection from vision to text space, and 3) inherent hallucinations common in general language models. To address the first two LVLM-specific causes, we introduce Mitigating hallucinAtion via image-gRounded guIdaNcE (MARINE). MARINE mitigates hallucination issues arising from the visual encoder and information distortion during cross-modal alignment by leveraging external guidance from image-grounded models, such as object detection models. Our approach leverages the inherent advantage of these image-grounded models, which are specifically designed and trained for more detailed visual information extraction. These models provide higher quality, fine-grained visual encoding compared to the standard visual encoders in LVLMs, which are primarily optimized for grasping the overall context of an image. Furthermore, we integrate the guidance from image-grounded models into text descriptions, allowing the LVLM to process the information without requiring additional alignment procedures. As a result, MARINE is a training-free, API-free method that addresses object hallucination at inference time by targeting its two root causes.

![](images/6f2870b11a2ac31579be7c2aadf4c7e1680077d7adb8bb9f5d6d4f15f0175da2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["Vision Encoder (LVLM)"]
    B --> C["Alignment (LVLM)"]
    C --> D["Image- Grounded Guidance"]
    D --> E["focusing on the visible objects in this image"]
    E --> F["LLM Decoder"]
    F --> G["magazine: 0.7\ntelephone: 0.2\ncup: 0.1\n......"]
    F --> H["mouse: 0.5\nmagazine: 0.3\ncup: 0.15\n......"]
    F --> I["magazine: 0.6\ncup: 0.1\ntelephone: 0.1\n......"]
    J["Task: Describe this image in detail."] --> K["Tokenizer"]
    K --> F
    F --> L["Control guidance in the logit space"]
    L --> M["Image Toolbox"]
    M --> N["Guidance Model"]
    N --> O["Input Image"]
```
</details>

Figure 1. Illustration of MARINE framework, which introduces a vision toolbox with one or multiple guidance models to enrich the visual context of the original LVLM. The output logits are controlled to place more importance on the guided generation with the guidance strength γ.

As shown in Figure 1, MARINE incorporates one or more image-grounding models to enrich the visual context of LVLMs. The guidance are then aggregated as prompt input to the LLM decoder to improve the response quality. Empirical evaluations are conducted on five widely-recognized LVLMs across benchmarks including MSCOCO (Lin et al., 2014), LLaVA-QA90 task (Liu et al., 2023d), A-OKVQA (Schwenk et al., 2022), and GQA (Hudson & Manning, 2019). We present results based on guidance from a aggregated source of DEtection TRansformer (DETR) (Carion et al., 2020) and RAM++ (Huang et al., 2023b). We also include ideal results based on ground truth object oracle, denoted as MARINE-Truth. Our experimental results demonstrate that, in comparison with state-of-theart algorithms, MARINE exhibits further reduced hallucination, as measured by popular hallucination metrics such as CHAIR (Rohrbach et al., 2018) and POPE (Li et al., 2023b), as well as GPT-4V’s evaluation. These results confirm that MARINE can effectively mitigate object hallucinations without requiring additional training resources or access to proprietary LLMs. To summarize, our contribution are listed as follows:

• We introduce MARINE, a universal framework and aggregating a toolbox of image-grounded visual models to guide the generation process of LVLMs. MARINE leverages the intrinsic advantages of these visual models in providing the detailed information of the input image and help mitigate the hallucinations in LVLMs.   
• Through extensive evaluations on various datasets, we demonstrate that MARINE consistently outperform the baselines in hallucination mitigation while maintaining overall performance across multiple tasks (image captioning, VQA).   
• MARINE provides a favorable trade-off between latency and accuracy, with the lowest computational overhead compared to existing baselines, which positions MARINE as a practical and scalable solution for real-world applications without significant computational cost.

# 2 Related Work

# 2.1 Object Hallucination in Large Vision-Language Models

The hallucination issue in Large Vision-Language Models (LVLMs) (Liu et al., 2023d; Zhu et al., 2023; Ye et al., 2023; Dai et al., 2023a; Gao et al., 2023) has drawn significant attention, as highlighted by studies (Li et al., 2023b; Wang et al., 2023b; Zhou et al., 2023; Fu et al., 2023; Lovenia et al., 2023). Notably, different from textual LLMs, LVLMs are prone to a unique type of hallucination called ‘object hallucination’ (Rohrbach et al., 2018), where the model falsely perceives the presence of non-existent objects in images. Efforts to address this problem in LVLMs include fine-tuning approaches using vision-language datasets (Liu et al., 2023b; Gunjal et al., 2023), as well as GPT-assisted methods such as those by Zhai et al. (2023). Notably, Yin et al. (2023) proposed a training-free approach using GPT-3.5 for hallucination correction.

Concurrently, Leng et al. (2023) introduced Visual Contrastive Decoding (VCD), a technique that applies noise to image inputs and penalizes logit outputs of these corrupted images. Huang et al. (2023a) enhanced beam-search decoding with the Over-trust Penalty and Retrospection-Allocation Strategy (OPERA), which penalizes over-trust and refines token selection based on previous outputs. HALC (Chen et al., 2024) employs adaptive focal-contrast decoding to encourage LVLMs to focus on fine-grained visual information, while using a computationally intensive beam search algorithm. In addition, BRAVE (Kar et al., 2024) introduces a new architecture that combines features from multiple vision encoders. While not directly targeting hallucination, it shares the key insight of leveraging diverse visual signals to improve grounding.

# 2.2 Controllable Generation

Controllable text generation (Prabhumoye et al., 2020; Hu & Li, 2021; Zhang et al., 2023a) has emerged as a vital research domain, focusing on the generation of natural sentences with controllable attributes such as persona (Prabhumoye et al., 2020; Hu & Li, 2021; Zhang et al., 2023a) and politeness (Niu & Bansal, 2018; Madaan et al., 2020). Among the various approaches, fine-tuning has been recognized as the most straightforward approach, achieved either through full fine-tuning (Li & Liang, 2021; Ouyang et al., 2022; Carlsson et al., 2022) or integrating tunable adaptors (Lin et al., 2021; Ribeiro et al., 2021). While finetuning has been effective in a wide range of applications, it is also expensive in computation as the size of LLMs is growing tremendously. Recently, there has been a development on controllable generation with diffusion models (Li et al., 2022; Lin et al., 2023b), extending to controllable text-to-image generation (Yang et al., 2023). Particularly, the use of classifier guidance (Dhariwal & Nichol, 2021) and classifier-free guidance (Ho & Salimans, 2021) has become prominent in refining the quality of generated outputs. Most recently, Sanchez et al. (2023) applied classifier-free guidance to language models in the single-modal setting to improve their performance at inference time. Our approach methodologically resembles classifier-free guidance for LVLMs’ text generation, while specifically addressing the multi-modal context and focusing on reducing hallucinations.

# 3 Preliminaries

Generative language models. Let pθ denotes an LLM parameterized by θ. Consider a sequence $\mathbf { x } = [ x _ { 1 } , \ldots , x _ { n } ]$ ] as the input prompt, where each $x _ { i }$ is a token from a predefined vocabulary. The LLM then generates the response sequence $\mathbf { y } = [ y _ { 1 } , \dots , y _ { m } ]$ by sampling from the conditional probability distribution $p _ { \pmb { \theta } } ( \cdot | \mathbf { x } )$ , where $y _ { t }$ denotes individual token for $1 \leq t \leq m$ . The conditional distribution $p _ { \pmb { \theta } } ( \mathbf { y } | \mathbf { x } )$ can therefore be expressed as $\begin{array} { r } { p _ { \pmb { \theta } } ( \mathbf { y } | \mathbf { x } ) = \prod _ { t = 1 } ^ { m } p _ { \pmb { \theta } } ( y _ { t } | \mathbf { x } , \mathbf { y } _ { < t } ) } \end{array}$ , where $\mathbf { y } _ { < t } = [ y _ { 1 } , \dots , y _ { t - 1 } ]$ for $t > 1$ and is empty for $t = 1$ . In the case of LVLMs, visual tokens $\mathbf { v } = [ v _ { 1 } , \dots , v _ { k } ]$ are additionally included. These tokens are generated from a pre-trained visual encoder and mapped into the token space through a linear projection. The conditional distribution of output y given the visual tokens v and textual prompt x is expressed as $\begin{array} { r } { p _ { \pmb \theta } ( \mathbf { y } | \mathbf { v } , \mathbf { x } ) = \prod _ { t = 1 } ^ { m } p _ { \pmb \theta } ( y _ { t } | \mathbf { v } , \mathbf { x } , \mathbf { y } _ { < t } ) } \end{array}$ , where pθ is approximated by LVLMs.

Guidance in generative models. The process of a guided generation involves getting the output y conditioned on input x, which encodes the desired properties of the output y. This guidance can be generally added to the model by two distinct approaches: classifier guidance (Dhariwal & Nichol, 2021) and classifier-free guidance (Ho & Salimans, 2021). As a top-level view, both methods formulate the conditional probability distribution of output y conditioned on guidance x as

$$
p (\mathbf {y} | \mathbf {x}) \propto p _ {\boldsymbol {\theta}} (\mathbf {y}) p (\mathbf {x} | \mathbf {y}) ^ {\gamma}, \tag {3.1}
$$

where $p _ { \pmb { \theta } } ( \mathbf { y } )$ is the original generative model and $p ( \mathbf { x } | \mathbf { y } )$ is the posterior distribution of x given y and γ is the guidance strength. In the classifier guidance, the posterior distribution $p ( \mathbf { x } | \mathbf { y } )$ in (3.1) is replaced by a classifier $p _ { \phi } ( \mathbf { x } | \mathbf { y } )$ parameterized by ϕ, which requires additional training step and calculating $\nabla _ { \mathbf { x } } \log p _ { \phi } ( \mathbf { x } | \mathbf { y } )$ . The classifier-free guidance, on the other hand, removes the necessity of the parameterized classifier $f _ { \phi }$ . Instead, according to the Bayes rule, the posterior distribution can be approximated by $p _ { \pmb { \theta } } ( \mathbf { x } | \mathbf { y } ) \propto p _ { \pmb { \theta } } ( \mathbf { y } | \mathbf { x } ) / p _ { \pmb { \theta } } ( \mathbf { y } )$ , where $p _ { \pmb { \theta } } ( \mathbf { y } | \mathbf { x } )$ is the generative model when taking x as prompt input. Plugging this back into (3.1) yields the guided distribution that can be approximated by

$$
\widehat {p} _ {\pmb {\theta}} (\mathbf {y} | \mathbf {x}) \propto p _ {\pmb {\theta}} (\mathbf {y}) \cdot \frac {p _ {\pmb {\theta}} (\mathbf {y} | \mathbf {x}) ^ {\gamma}}{p _ {\pmb {\theta}} (\mathbf {y}) ^ {\gamma}} = \frac {p _ {\pmb {\theta}} (\mathbf {y} | \mathbf {x}) ^ {\gamma}}{p _ {\pmb {\theta}} (\mathbf {y}) ^ {\gamma - 1}}.
$$

As a result, the guided LLM pθ places more importance on the prompt x during generation with the increasing value of $\gamma ,$ thereby producing texts that better align with the desired behavior from the prompt (Sanchez et al., 2023).

# 4 Method

The existing architecture of LVLMs is composed of a visual encoder, a visual and textual domain alignment layer, and the LLM itself. Therefore, besides the inherent language priors of LLMs (Biten et al., 2022), object hallucination may arise from (1) deficiencies in the visual encoder provide insufficient visual information (Zhang et al., 2023b) and (2) distortion or loss of visual information during the projection from vision to language space. To mitigate object hallucinations, we introduce MARINE, a framework containing two major components to address the previous challenges: (1) introducing additional visual information from a set of vision models and (2) using the additional aggregated visual features to guide the LVLM’s generation. In Figure 1, we present the framework overview.

# 4.1 Visual Guidance from Image-Grounded Features

To introduce image-grounded guidance to mitigate hallucinations, our approach integrates additional object detection models, which differ from the visual encoders used in LVLM that are usually pre-trained from CLIP (Radford et al., 2021). This integration leverages object detection models to extract detailed visual information from images. Upon acquiring extra visual information from different image-grounded models, we aggregate and translate the collected information into textual information. This aggregation can be done by the language model (Lin et al., 2023a) or rule-based algorithm (Bird et al., 2009). Such an information aggregation is effective and efficient, as it eliminates the necessity of fine-tuning the alignment layer while retaining the rich information encoded by various of image grounding models. We subsequently employ a simple prompt “focusing on the visible objects in this image:” and concatenate it with the aggregated object information, denoted as the guidance prompt c.

# 4.2 Guided Text Generation with Visual Information

We tackle the object hallucination problem of LVLMs by placing importance on additional image-grounded information. In addition to the visual tokens v extracted from the original LVLM and textual prompt x, we extract the auxiliary visual tokens c from the additional guidance models. The generation of the t-th token in the output y of our classifier-free guided LVLM pθ is expressed as

$$
\widehat {p} _ {\boldsymbol {\theta}} (y _ {t} | \mathbf {v}, \mathbf {c}, \mathbf {x}, \mathbf {y} _ {<   t}) \propto \frac {p _ {\boldsymbol {\theta}} (y _ {t} | \mathbf {v} , \mathbf {c} , \mathbf {x} , \mathbf {y} _ {<   t}) ^ {\gamma}}{p _ {\boldsymbol {\theta}} (y _ {t} | \mathbf {v} , \mathbf {x} , \mathbf {y} _ {<   t}) ^ {\gamma - 1}},
$$

where c denotes our control guidance and γ is the control strength. The sampling of output generation is given by

$$
\begin{array}{l} \widehat {p} _ {\boldsymbol {\theta}} (\mathbf {y} | \mathbf {v}, \mathbf {c}, \mathbf {x}) = \prod_ {t = 1} ^ {m} \widehat {p} _ {\boldsymbol {\theta}} (y _ {t} | \mathbf {v}, \mathbf {c}, \mathbf {x}, \mathbf {y} _ {<   t}) \\ \propto \prod_ {t = 1} ^ {m} \frac {p _ {\boldsymbol {\theta}} (y _ {t} | \mathbf {v} , \mathbf {c} , \mathbf {x} , \mathbf {y} _ {<   t}) ^ {\gamma}}{p _ {\boldsymbol {\theta}} (y _ {t} | \mathbf {v} , \mathbf {x} , \mathbf {y} _ {<   t}) ^ {\gamma - 1}} \\ = \frac {p _ {\boldsymbol {\theta}} (\mathbf {y} | \mathbf {v} , \mathbf {c} , \mathbf {x}) ^ {\gamma}}{p _ {\boldsymbol {\theta}} (\mathbf {y} | \mathbf {v} , \mathbf {x}) ^ {\gamma - 1}}. \\ \end{array}
$$

We can further view MARINE in the logit space, where the t-th token is therefore sampled from the logit space by

$$
\begin{array}{l} \log \widehat {p _ {\boldsymbol {\theta}}} (y _ {t} | \mathbf {v}, \mathbf {c}, \mathbf {x}, \mathbf {y} _ {<   t}) = \gamma \log p _ {\boldsymbol {\theta}} (\mathbf {y} | \mathbf {v}, \mathbf {c}, \mathbf {x}, \mathbf {y} _ {<   t}) \\ + (1 - \gamma) \log p _ {\boldsymbol {\theta}} (\mathbf {y} | \mathbf {v}, \mathbf {x}, \mathbf {y} _ {<   t}). \\ \end{array}
$$

This linear combination of logits implies that the conditional generation on the additional image-grounded guidance acts as a controllable gate. Only objects with relatively high probabilities in both branches could appear at top when sampling. Specifically, setting $\gamma = 0$ recovers the original LLM generation without control guidance and setting $\gamma = 1$ produces the LLM generation entirely based on the control. Meanwhile, for $\gamma \in ( 0 , 1 )$ , MARINE yields a combination of the original generation $p _ { \pmb { \theta } } ( \mathbf { y } | \mathbf { v } , \mathbf { x } )$ and the generation conditioned on the guidance $p _ { \pmb { \theta } } ( \mathbf { y } | \mathbf { v } , \mathbf { c } , \mathbf { x } )$ . This strikes a balance between a better ability to follow instructions to generate high-quality answers and the increased accuracy and detail in image descriptions. The formulation therefore shares resemblance to the classifier-free guidance introduced for LLMs (Sanchez et al., 2023), which places importance on the textual prompt itself to better align the LLM generation with user intention in the single-modal setting. We summarize MARINE in Algorithm 1. In detail, MARINE aggregates the collected visual information $\{ \mathbf { c } _ { i } \}$ i using function Aggr., which can be a small language model for information aggregation (Lin et al., 2023a).

# Algorithm 1 Mitigating hallucinAtion via image-gRounded guIdaNcE (MARINE)

1: Input: LLM parameter θ, input prompt x, visual tokens v from LVLM’s original vision tower   
2: Input: auxiliary visual tokens $\{ { \mathbf { c } } _ { i } \} _ { i = 1 } ^ { M }$ from M image grounding models, guidance scale γ   
3: Initialize empty output $\mathbf { y } = \mathbb { I } .$   
4: Aggregate visual information as textual prompt $\mathbf { c } =$ Aggr. $( \{ \mathbf { c } _ { i } \} _ { i = 1 } ^ { M } )$   
5: for $t = 0 , 1 , \ldots , T$ do   
$\mathbf { x } _ { \mathrm { u n c o n d } } ^ { ( t ) } = \left[ \mathbf { v } , \mathbf { x } , \mathbf { y } _ { < t } \right]$   
7: Generate unconditional output logits using LLM: $\ell _ { \mathrm { u n c o n d } } ^ { ( t ) } = \log p _ { \theta } ( \mathbf { x } _ { \mathrm { u n c o n d } } ^ { ( t ) } ) .$ .   
$\begin{array} { r } { \mathbf { x } _ { \mathrm { c o n d } } ^ { ( t ) } = [ \mathbf { v } , \mathbf { c } , \mathbf { x } , \mathbf { y } _ { < t } ] . } \end{array}$   
9: tional output logits using LLM: . $\ell _ { \mathrm { c o n d } } ^ { ( t ) }$ $= \log p \pmb { \sigma } ( \mathbf { x } _ { \mathrm { c o n d } } ^ { ( t ) } )$   
$\ell ^ { ( t ) } = \gamma \ell _ { \mathrm { c o n d } } ^ { ( t ) } + ( 1 - \gamma ) \ell _ { \mathrm { u n c c } } ^ { ( t ) }$   
11: Sample token yt from logit space denoted by $\ell ^ { ( t ) }$   
12: Let $\mathbf { y } = [ \mathbf { y } , y _ { t } ] .$   
13: end for   
14: Output: y.

# 5 Experiments

In this section, we evaluate MARINE in mitigating object hallucinations across various LVLMs, showing that it outperforms state-of-the-art methods on established metrics across different question formats.

# 5.1 Experiment Setup

Models. To demonstrate the broad applicability of our approach across different LVLM architectures, we apply and evaluate MARINE to widely-used models including LLaVA (Liu et al., 2023d), LLaVA-v1.5 (Liu et al., 2023c), MiniGPT-v2 (Chen et al., 2023), mPLUG-Owl2 (Ye et al., 2023) and InstructBLIP (Liu et al., 2023c). To address the object hallucination problems in text generation, we incorporate the DEtection TRansformer (DETR) (Carion et al., 2020) and RAM++ (Huang et al., 2023b) as the additional vision models for guidance.

Guidance from multiple sources. Our framework’s compatibility with various vision models allows for the incorporation of multiple sources to enhance precision and robustness. By considering object-level information from DETR and RAM++ simultaneously, we generate guidance that reflects consensus across these models. This approach significantly improves the accuracy and reliability of the guidance provided to the LVLM.

Datasets and evaluations. In alignment with established evaluations from previous studies (Dai et al., 2023b; Yin et al., 2023), we assess our method using the following metrics:

• Caption Hallucination Assessment with Image Relevance (CHAIR) (Rohrbach et al., 2018). It involves prompting the LVLMs to generate a description for the input image, and then comparing this generation with ground truth objects present in the image. CHAIR quantifies hallucination both at instance level and sentence level, respectively defined as CHAIRI and CHAIRS:

$$
\begin{array}{c} \mathrm{CHAIR} _ {I} = \frac {\left| \{\text {hallucinated objects} \} \right|}{\left| \{\text {all mentioned objects} \} \right|} \\ \mathrm{CHAIR} _ {S} = \frac {\left| \{\text {captions with hallucinated objects} \} \right|}{\left| \{\text {all captions} \} \right|} \end{array}
$$

In addition to these metrics, we incorporate an instancelevel Recall score in our evaluation to evaluate whether the descriptions accurately include the necessary visual content from the image:

$$
\text { Recall } = \frac {\left| \{\text { non -hallucinated   objects } \} \right|}{\left| \{\text { all   existing   objects } \} \right|}
$$

• Polling-based Object Probing Evaluation (POPE) (Li et al., 2023b). POPE formulates a binary classification task by prompting LVLMs with questions such as “Is there a keyboard in this image?” to answer “yes” or “no”. We specifically focus on the adversarial setting, which is considered the most challenging setting. Results for the random and popular settings are detailed in Appendix C. We report the accuracy and F1 score of the LVLMs’ responses, and the proportion of “yes” answers.

• GPT-4V-aided Evaluation (Yin et al., 2023). The GPT-4V-aided evaluation compares the outputs of two LVLM assistants using GPT-4V as a judge. In this evaluation, we utilize the LLaVA-QA90 task (Liu et al., 2023d) (including conversations, visual perceptions, and complex reasoning tasks) and additionally consider the image captioning task.

Consistent with Li et al. (2023b), we randomly sampled a subset of 500 images from MSCOCO (Lin et al., 2014) dataset for CHAIR evaluation. For the POPE evaluation, we created 3000 questions across three datasets—500 images each from MSCOCO, A-OKVQA (Schwenk et al., 2022), and GQA (Hudson & Manning, 2019). For the GPT-4Vaided evaluation, we utilized 90 questions from the LLaVA-QA90 task and randomly selected 50 MSCOCO images for image captioning task.

Baselines. In addition to comparing with the performance of the original LVLM sampling method, we also consider the following popular methods for mitigating hallucinations.

• Greedy-Decoding, which adopts the greedy sampling strategy, by generating tokens with the highest posterior probability to address hallucinations arising from.   
• LURE (Zhou et al., 2023), which identifies and masks potentially hallucinated words and fine-tune a MiniGPT4 model to rectify object hallucinations in the generated descriptions.   
• Woodpecker (Yin et al., 2023), which leverages GPT-3.5 to correct hallucinations in LVLM generation with five steps toward the correction.   
• VCD (Leng et al., 2023), which distorts the image inputs to impose penalties on logit outputs.   
• OPERA (Huang et al., 2023a), which penalizes logits to mitigate over-trust in beam-search decoding and adjusts token selection.

Lastly, the performance of MARINE improves in correlation with the advancement of the control guidance extractor used. Consequently, to demonstrate the potential upper bound of MARINE’s performance, we consider a version utilizing a ground-truth oracle extractor, which we denote as MARINE-Truth. Further details on model architectures, datasets and evaluation metrics are deferred to Appendix A.

Hyperparameter setting. The hyperparameters for our method are fixed across tasks, with key settings including a guidance strength of 0.7, score threshold for DETR at 0.95, a detection threshold for RAM++ of 0.68, and a greedy sampling approach with a random seed of 242.

# 5.2 Results

Experimental results on object hallucination metrics (CHAIR and POPE) are presented in Table 1 and 2. Overall, MARINE achieves superior performances across different LVLM architectures and evaluation metrics.

Table 1. Evaluation with CHAIR score across multiple LVLM architectures comparing our method with several baselines. We report CHAIRS, CHAIRI and the recall score. The bold numbers indicate the best results among the methods evaluated and the underscored numbers represent the second-best results. We show MARINE-Truth as a reference performance of MARINE. 

<table><tr><td rowspan="2">MethodCHAIR</td><td colspan="3">LLaVA</td><td colspan="3">LLaVA-v1.5</td><td colspan="3">MiniGPTv2</td><td colspan="3">mPLUG-Owl2</td><td colspan="3">InstructBLIP</td><td colspan="3">Average</td></tr><tr><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $R \uparrow$ </td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $R \uparrow$ </td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $R \uparrow$ </td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $R \uparrow$ </td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $R\uparrow$ </td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $R \uparrow$ </td></tr><tr><td>Greedy</td><td>26.6</td><td>10.5</td><td>47.4</td><td>8.8</td><td>4.6</td><td>41.1</td><td>8.2</td><td>4.2</td><td>41.1</td><td>6.2</td><td>3.4</td><td>38.8</td><td>5.0</td><td>3.2</td><td>33.2</td><td>11.0</td><td>5.2</td><td>40.3</td></tr><tr><td>LURE</td><td>33.8</td><td>11.6</td><td>54.8</td><td>38.9</td><td>11.2</td><td>56.3</td><td>36.2</td><td>11.4</td><td>54.6</td><td>33.9</td><td>10.8</td><td>55.9</td><td>38.1</td><td>12.1</td><td>54.5</td><td>36.2</td><td>11.4</td><td>55.2</td></tr><tr><td>Woodpecker</td><td>19.5</td><td>8.9</td><td>44.3</td><td>8.5</td><td>4.5</td><td>38.4</td><td>7.5</td><td>4.5</td><td>37.0</td><td>8.0</td><td>4.3</td><td>37.5</td><td>8.0</td><td>6.2</td><td>32.6</td><td>10.3</td><td>5.7</td><td>38.0</td></tr><tr><td>VCD</td><td>28.1</td><td>11.0</td><td>46.6</td><td>7.3</td><td>4.1</td><td>40.8</td><td>6.8</td><td>3.9</td><td>38.2</td><td>5.9</td><td>3.4</td><td>37.7</td><td>2.4</td><td>1.5</td><td>33.7</td><td>10.1</td><td>4.8</td><td>39.4</td></tr><tr><td>OPERA</td><td>22.4</td><td>9.9</td><td>43.6</td><td>11.0</td><td>6.7</td><td>40.2</td><td>9.2</td><td>5.0</td><td>41.3</td><td>5.8</td><td>3.2</td><td>38.4</td><td>4.6</td><td>2.7</td><td>38.0</td><td>10.6</td><td>5.5</td><td>40.3</td></tr><tr><td>MARINE</td><td>17.8</td><td>7.2</td><td>50.8</td><td>6.2</td><td>3.0</td><td>44.3</td><td>11.8</td><td>4.9</td><td>49.7</td><td>4.2</td><td>2.3</td><td>41.4</td><td>2.2</td><td>1.3</td><td>36.3</td><td>8.4</td><td>3.7</td><td>44.5</td></tr><tr><td>MARINE-Truth</td><td>19.6</td><td>5.1</td><td>79.0</td><td>6.0</td><td>2.5</td><td>55.3</td><td>12.6</td><td>3.8</td><td>70.5</td><td>3.8</td><td>1.7</td><td>48.0</td><td>3.0</td><td>1.8</td><td>35.9</td><td>8.9</td><td>2.9</td><td>57.5</td></tr></table>

Table 2. Evaluation with POPE score in adversarial setting across multiple LVLM architectures comparing our method with several baselines. We report the POPE accuracy (%), F1 score (%) and the yes ratio (%). The ideal yes ratio for a non-biased LVLM is 50%. The bold numbers indicate the best results among the methods evaluated and the underscored numbers represent the second-best results. We show MARINE-Truth as a reference performance of MARINE. 

<table><tr><td>Method</td><td colspan="3">LLaVA</td><td colspan="3">LLaVA-v1.5</td><td colspan="3">MiniGPTv2</td><td colspan="3">mPLUG-Owl2</td><td colspan="3">InstructBLIP</td><td colspan="3">Average</td></tr><tr><td>POPE</td><td>Acc ↑</td><td>F1 ↑</td><td>Yes</td><td>Acc ↑</td><td>F1 ↑</td><td>Yes</td><td>Acc ↑</td><td>F1 ↑</td><td>Yes</td><td>Acc ↑</td><td>F1 ↑</td><td>Yes</td><td>Acc ↑</td><td>F1 ↑</td><td>Yes</td><td>Acc ↑</td><td>F1 ↑</td><td>Yes</td></tr><tr><td>Greedy</td><td>51.8</td><td>67.4</td><td>97.7</td><td>79.4</td><td>81.6</td><td>61.6</td><td>82.7</td><td>81.7</td><td>44.5</td><td>72.5</td><td>77.5</td><td>72.4</td><td>79.8</td><td>81.4</td><td>58.6</td><td>73.2</td><td>77.9</td><td>67.0</td></tr><tr><td>LURE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Woodpecker</td><td>77.5</td><td>77.6</td><td>50.5</td><td>80.5</td><td>80.6</td><td>50.5</td><td>79.5</td><td>77.8</td><td>42.5</td><td>77.5</td><td>76.9</td><td>47.5</td><td>79.0</td><td>78.6</td><td>48.0</td><td>78.8</td><td>78.3</td><td>47.8</td></tr><tr><td>VCD</td><td>54.6</td><td>68.5</td><td>94.0</td><td>78.2</td><td>80.7</td><td>62.8</td><td>81.4</td><td>80.2</td><td>44.1</td><td>72.3</td><td>77.0</td><td>70.5</td><td>79.7</td><td>80.9</td><td>56.7</td><td>73.2</td><td>77.5</td><td>65.6</td></tr><tr><td>OPERA</td><td>51.7</td><td>67.4</td><td>98.0</td><td>77.5</td><td>80.1</td><td>63.2</td><td>82.9</td><td>81.9</td><td>44.3</td><td>70.3</td><td>79.1</td><td>84.6</td><td>79.8</td><td>81.4</td><td>58.6</td><td>72.4</td><td>78.0</td><td>69.7</td></tr><tr><td>MARINE</td><td>66.9</td><td>72.9</td><td>72.3</td><td>85.0</td><td>84.3</td><td>45.7</td><td>83.0</td><td>82.9</td><td>49.4</td><td>82.8</td><td>82.7</td><td>49.2</td><td>81.7</td><td>79.4</td><td>38.8</td><td>79.9</td><td>80.4</td><td>51.1</td></tr><tr><td>MARINE-Truth</td><td>75.6</td><td>80.1</td><td>72.3</td><td>92.0</td><td>92.5</td><td>57.0</td><td>86.9</td><td>88.3</td><td>62.5</td><td>93.4</td><td>93.8</td><td>56.2</td><td>93.8</td><td>93.8</td><td>51.0</td><td>88.3</td><td>89.7</td><td>59.8</td></tr></table>

Table 3. Results of GPT-4V-aided evaluation. The accuracy and detailedness metrics are on a scale of 10, and a higher score indicates better performance. The symbols × and ✓ indicate performance metrics without and with our method, respectively. 

<table><tr><td rowspan="2">Task</td><td rowspan="2">Metrics</td><td colspan="2">LLaVA</td><td colspan="2">mPLUG-Owl2</td></tr><tr><td> $\mathcal{X}$ </td><td> $\checkmark$ </td><td> $\mathcal{X}$ </td><td> $\checkmark$ </td></tr><tr><td rowspan="2">LLaVA-QA90</td><td>Acc ↑</td><td> $5.82_{\pm 0.10}$ </td><td> $\mathbf{5.94}_{\pm 0.05}$ </td><td> $6.03_{\pm 0.13}$ </td><td> $\mathbf{6.35}_{\pm 0.21}$ </td></tr><tr><td>Detail ↑</td><td> $4.59_{\pm 0.08}$ </td><td> $4.59_{\pm 0.08}$ </td><td> $5.06_{\pm 0.05}$ </td><td> $\mathbf{5.16}_{\pm 0.10}$ </td></tr><tr><td rowspan="2">Image Captioning</td><td>Acc ↑</td><td> $5.27_{\pm 0.20}$ </td><td> $\mathbf{6.11}_{\pm 0.23}$ </td><td> $7.97_{\pm 0.25}$ </td><td> $\mathbf{8.63}_{\pm 0.20}$ </td></tr><tr><td>Detail ↑</td><td> $\mathbf{4.39}_{\pm 0.29}$ </td><td> $4.36_{\pm 0.17}$ </td><td> $5.74_{\pm 0.24}$ </td><td> $\mathbf{6.19}_{\pm 0.23}$ </td></tr></table>

Results on CHAIR. CHAIR is a widely adopted benchmark for evaluating caption hallucination in LVLMs, comparing generated descriptions with ground-truth object annotations. It captures object-level precision through CHAIRI (instance-level) and $\mathrm { C H A I R } _ { S }$ (sentence-level), and we further report Recall to assess content coverage.

Table 1 shows that MARINE consistently outperforms existing approaches on all major metrics. It achieves the lowest average CHAIRI and $\mathrm { C H A I R } _ { S }$ scores and ranks second in Recall, reducing hallucination without sacrificing coverage. Compared to the second-best method, MARINE improves CHAIRS by 1.7 points and CHAIRI by 1.1 on average. The gains are particularly strong on LLaVA models, where hallucination drops by up to 8.8 points. In contrast, methods such as LURE and Woodpecker are less effective across model variants.

Importantly, MARINE achieves performance comparable to MARINE -Truth, a variant that uses ground-truth object labels as guidance. This finding suggests that aggregating signals from multiple visual models offers a compelling alternative to manual supervision to reduce hallucination.

Results on POPE. POPE is designed to assess objectlevel grounding in LVLMs by testing their ability to answer yes/no questions about visual content. We focus on the adversarial setting, which presents challenging negatives and helps expose hallucination and biased answering tendencies. In Table 2, MARINE consistently outperforms all baselines, with average improvements of 6.7% in accuracy and 3.5% in F1 score over the original model outputs. Compared to the second-best method, Woodpecker, MARINE still maintains a 1.1% gain in accuracy and a 2.1% gain in F1.

Beyond accuracy, MARINE also reduces the overconfident bias often seen in LVLMs’ outputs. This is reflected in a more balanced “yes” ratio (closer to 50%, reflecting a 15.9% shift towards unbiased answers). This shift suggests that MARINE produces more trustworthy predictions by reducing the tendency toward overconfident affirmative responses.

Table 4. POPE results across three datasets. We report the average score under random, popular, adversarial settings. The detailed POPE results can be found in the appendix C. The bold numbers indicate the best results. The ideal yes ratio for a non-biased LVLM is 50%. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">w/MARINE</td><td colspan="3">LLaVA</td><td colspan="3">mPLUG-Owl2</td></tr><tr><td>Accuracy ↑</td><td>F1 ↑</td><td>Yes(%)</td><td>Accuracy ↑</td><td>F1 ↑</td><td>Yes(%)</td></tr><tr><td rowspan="2">MSCOCO</td><td>✗</td><td>54.2</td><td>68.5</td><td>95.5</td><td>76.7</td><td>80.4</td><td>68.2</td></tr><tr><td>√</td><td>72.2</td><td>76.4</td><td>66.9</td><td>85.5</td><td>85.0</td><td>46.5</td></tr><tr><td rowspan="2">A-OKVQA</td><td>✗</td><td>51.8</td><td>67.5</td><td>97.9</td><td>69.6</td><td>76.5</td><td>78.5</td></tr><tr><td>√</td><td>64.3</td><td>72.8</td><td>80.2</td><td>82.0</td><td>83.5</td><td>57.2</td></tr><tr><td rowspan="2">GQA</td><td>✗</td><td>52.0</td><td>67.6</td><td>97.8</td><td>73.7</td><td>78.7</td><td>72.6</td></tr><tr><td>√</td><td>62.5</td><td>71.8</td><td>81.8</td><td>80.1</td><td>80.6</td><td>51.1</td></tr></table>

Results on GPT-4V-aided evaluation. Following Yin et al. (2023), this GPT-4V-assisted evaluation provides a qualitative perspective that complements the numerical metrics of CHAIR and POPE, offering a more comprehensive assessment of model performance. As shown in Table 3, GPT-4V consistently assigns higher accuracy with equal detailedness scores to models enhanced by MARINE, highlighting its ability to produce more precise and detailed descriptions, which demonstrates the robustness of our method in real-world visual tasks. The evaluation prompt is detailed in Appendix A.5.

Additional results on other vision-language tasks. To further evaluate the generalizability of our approach beyond object hallucination and the MSCOCO dataset, we extended our evaluations to additional datasets including A-OKVQA and GQA and included more general caption quality metrics. As shown in Table 4, the POPE results demonstrate that our method consistently mitigates hallucinations across various datasets with different image distributions. Figure 2 presents a comprehensive evaluation of the image captioning task on MSCOCO and LLaVA-QA90, a comprehensive VQA dataset, using metrics including BLEU (Papineni et al., 2002), ROUGE (Lin, 2004), CIDEr (Vedantam et al., 2015) and SPICE (Anderson et al., 2016). These results demonstrate that, although our method primarily targets hallucination mitigation, it maintains the overall performance of LVLMs on broader tasks, with no significant trade-offs in caption or VQA quality.

Latency analysis Many existing approaches to mitigating object hallucination rely on post-generation correction models (Zhou et al., 2023; Zhai et al., 2023; Yin et al., 2023), external object detectors (Yin et al., 2023), or complex decoding strategies (Huang et al., 2023a; Leng et al., 2023), all of which introduce substantial computational overhead. To assess the practical efficiency of MARINE, we evaluate its latency compared to existing baselines on LLaVA-7B, as shown in Table 5.

Our measurements include the time required for additional forward passes through external vision models. These models contribute only marginal latency relative to the cost of autoregressive decoding in LVLMs. In general, MARINE increases decoding time by just 1.98×, the lowest among all baselines. This demonstrates that MARINE achieves the most favorable trade-off between latency and accuracy, which makes it suitable for real-world use. Detailed settings are provided in Appendix A.6.

![](images/ec37513d16edffa9f7ce51d024e2c8cd8f16461e66b6643cb83fc2b67e5136dc.jpg)

<details>
<summary>radar</summary>

|        | BLEU_1 | BLEU_2 | BLEU_3 | BLEU_4 | ROUGE_L | CIDE | SPICE |
| ------ | ------ | ------ | ------ | ------ | ------- | ---- | ----- |
| LLaVA  | 0.5    | 0.3    | 0.2    | 0.1    | 0.6     | 0.1  | 0.8   |
| mPLUG-Owl2 | 1.0    | 0.9    | 0.9    | 0.9    | 0.9     | 0.9  | 0.9   |
</details>

Figure 2. MARINE maintains or improves overall text quality on general metrics. Solid lines indicate models with MARINE, while dashed lines indicate the original models. Higher scores indicate better textual similarity to the reference outputs.

# 5.3 Ablation Study

Why incorporate multiple image-grounded models? Different image-grounded models excel at capturing different aspects of visual information—some detect objects precisely, while others offer broader, fine-grained context. To understand whether combining these complementary signals leads to better guidance, we conduct an ablation comparing DETR and RAM++ individually versus in combination (Table 6). All variants are evaluated under the same decoding setup to ensure a fair comparison.

DETR allows for highly accurate object detection, while RAM++ excels in extensive recognition tasks, contributing fine-grained visual concepts. Their combination yields consistent improvements on CHAIR metrics, suggesting that aggregating multiple visual perspectives is important for effective hallucination mitigation.

What is the best way to integrate guidance from multiple models? When aggregating the outputs from multiple image-grounding models, the combination method can significantly affect guidance quality. We compare two strategies: taking the intersection or the union of detected objects. As shown in Table 7, the intersection-based approach consistently outperforms the union, significantly reducing hallucination. This suggests that enforcing agreement across models leads to more precise and trustworthy guidance, while union-based aggregation may introduce noisy or spurious information. The detailed experimental setup and prompt templates are provided in Appendix A.

How does control strength affect generation? To understand the impact of guidance strength in our decoding setup, we vary the control weight γ, which balances the influence between the original LVLM generation and the generation conditioned on external image-grounded guidance.

Table 5. Inference latency comparison. We report both the latency and the ratio to the latency of greedy decoding of the original LVLM model. 

<table><tr><td></td><td>Greedy</td><td>LURE</td><td>Woodpecker*</td><td>VCD</td><td>OPERA</td><td>MARINE (ours)</td></tr><tr><td>Training Cost</td><td>0</td><td>10min on A100 80G</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Inference Latency(ms/token)</td><td>26.3 (×1.0)</td><td>179.9 (×6.84)</td><td>94.5 (×3.59)*</td><td>53.4 (×2.03)</td><td>185.1 (×7.0)</td><td>52.2 (×1.98)</td></tr></table>

∗Woodpecker requires GPT API key access and the latency may depend on OPENAI API.

Table 6. Ablation study comparing the performance of combining DETR and RAM++ models versus using individual vision models. This approach leverages multiple object detectors to provide more reliable and robust object-level guidance, resulting in superior performance on CHAIR metrics. 

<table><tr><td rowspan="2">ModelCHAIR</td><td colspan="2">LLaVA</td><td colspan="2">LLaVA-v1.5</td><td colspan="2">mPLUG-Owl2</td></tr><tr><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td></tr><tr><td>Greedy</td><td>26.6</td><td>10.5</td><td>8.8</td><td>4.6</td><td>6.2</td><td>3.4</td></tr><tr><td colspan="7">Ensembling Models</td></tr><tr><td>MARINE</td><td>17.8</td><td>7.2</td><td>6.2</td><td>3.0</td><td>4.2</td><td>2.3</td></tr><tr><td colspan="7">Single Models</td></tr><tr><td>MARINE-DETR only</td><td>27.6</td><td>8.4</td><td>10.5</td><td>4.3</td><td>5.3</td><td>2.7</td></tr><tr><td>MARINE-RAM only</td><td>29.0</td><td>9.1</td><td>6.6</td><td>3.7</td><td>5.2</td><td>2.8</td></tr></table>

Table 7. Effect of Integration Methods for Image-Grounding Models. 

<table><tr><td rowspan="2">ModelCHAIR</td><td colspan="2">LLaVA</td><td colspan="2">LLaVA-v1.5</td><td colspan="2">mPLUG-Owl2</td></tr><tr><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td></tr><tr><td>Greedy</td><td>26.6</td><td>10.5</td><td>8.8</td><td>4.6</td><td>6.2</td><td>3.4</td></tr><tr><td>MARINE-intersection (ours)</td><td>17.8</td><td>7.2</td><td>6.2</td><td>3.0</td><td>4.2</td><td>2.3</td></tr><tr><td>MARINE-union</td><td>30.4</td><td>9.7</td><td>5.4</td><td>2.7</td><td>4.8</td><td>2.7</td></tr></table>

Figure 3 shows that increasing guidance strength from 0 to 1 leads to a notable decrease in CHAIR scores. This trend suggests that higher guidance strength makes LVLMs rely more on image-grounded features, thereby enhancing their ability to produce accurate descriptions. It’s crucial to note that, although some models exhibit optimal performance at a guidance strength of $\gamma = 1$ , excessively strong guidance can adversely affect the models’ ability to adhere to provided instructions. Experimental evidence is detailed in Appendix B.5. This observation highlights the necessity of having a balanced guidance strength that ensures high-quality, accurate outputs while adhering closely to the given instructions. Based on our findings, we recommend a guidance strength within the range of $\gamma \in ( 0 . 3 , 0 . 7 )$ as the most effective for maintaining this balance.

# 6 Conclusions, Limitations and Future Work

In this paper, we introduced a training-free and API-free framework MARINE to mitigate object hallucination in LVLMs during its text generation process. Leveraging a pre-trained object grounding vision encoder for a novel guidance framework in the multi-modal setting, MARINE effectively and cost-efficiently reduces the hallucinations of five widely-used LVLMs, as assessed by various metrics across different tasks. The inherent compatibility of the MARINE with various vision models and projection functions further underscores its flexibility. In contrast to post-generation correction methods, MARINE strikes a balance between efficiency, instruction-following ability and effectiveness in reducing object hallucinations.

![](images/2e56526ae3b50416c522b12dd565f8859517350200c8af2f4d9b25187c7016ba.jpg)

Figure 3. Ablation study on the effect of guidance strength (γ) on the performance of LLaVA, LLaVA-v1.5 and mPLUG-Owl2 using CHAIR metrics, with γ ranging from 0 to 1.   
![](images/20f422ffc9c9872387a290113ad20de9675d3f9faf55408caebdee3230796e3b.jpg)  
Query: Is there a chair in the image?   
LLaVA: Yes, there is a white chair in the image.   
LLaVA w/ MARINE: No, there is no chair in the image. The only object present is a white bird.   
Query: Generate a short caption of the image.

LLaVA-v1.5: A kitchen with a microwave, coffee maker, and toaster.   
LLaVA-v1.5 w/ MARINE: A kitchen counter with a microwave, coffee maker, and a laptop.   
![](images/aee08289a3c3f213146761217b67140d79bbdf014abcdfd734254b7b02cf282d.jpg)  
Figure 4. Hallucination mitigation examples by our proposed MARINE across multiple tasks. Hallucinated objects generated by the LVLM are highlighted in red.

Limitations and future work. While MARINE has demonstrated impressive performance by utilizing guidance from image-grounded models, there remains potential for further improvement through the integration of advanced aggregation methods, such as multi-agent debate (Du et al., 2023), into the MARINE framework. Additionally, although MARINE is specifically designed to mitigate object hallucination, which is the most significant issue in LVLMs, extending its application to address other types of hallucinations in both LLMs and LVLMs across a broader range of benchmarks would be highly advantageous.

# Acknowledgments

We thank anonymous reviewers for their helpful comments. Part of this work was done while WZ was a PhD student at UCLA. WZ and QG are supported in part by NSF grants DMS-2323113, CPS-2312094, IIS-2403400, and the research fund from the UCLA-Amazon Science Hub. WZ was also supported by the UCLA dissertation year fellowship. The views and conclusions contained in this paper are those of the authors and should not be interpreted as representing any funding agencies.

# Impact Statement

This paper introduces research aimed at advancing the field of Large Language Models. We are confident that our work will contribute to significant social benefits, particularly by enhancing the accountability of LLMs through the reduction of hallucinatory outputs. Our proposed method, MARINE, holds the potential to improve the fairness of LLM interactions by effectively reducing biased hallucinations. By mitigating hallucinations, MARINE has the potential to offer a positive social impact by ensuring that LVLMs generate more accountable responses. Despite this merit, MARINE cannot address prejudicial biases inherent in LLM prior knowledge, which could be a focus of future work. To the best of our knowledge, we have not identified any negative effects associated with our research that merit highlighting in this discussion.

# References

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.   
Anderson, P., Fernando, B., Johnson, M., and Gould, S. Spice: Semantic propositional image caption evaluation, 2016.   
Bazi, Y., Rahhal, M. M. A., Bashmal, L., and Zuair, M. Vision–language model for visual question answering in medical imagery. Bioengineering, 10(3):380, 2023.   
Bird, S., Klein, E., and Loper, E. Natural language processing with Python: analyzing text with the natural language toolkit. ” O’Reilly Media, Inc.”, 2009.   
Biten, A. F., Gomez, L., and Karatzas, D. Let there be ´ a clock on the beach: Reducing object hallucination in image captioning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1381–1390, 2022.   
Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S. End-to-end object detection with

transformers. In European conference on computer vision, pp. 213–229. Springer, 2020.

Carlsson, F., Ohman, J., Liu, F., Verlinden, S., Nivre, J., and ¨ Sahlgren, M. Fine-grained controllable text generation using non-residual prompting. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6837–6857, 2022.

Chambon, P., Bluethgen, C., Langlotz, C. P., and Chaudhari, A. Adapting pretrained vision-language foundational models to medical imaging domains. arXiv preprint arXiv:2210.04133, 2022.

Chen, J., Zhu, D., Shen, X., Li, X., Liu, Z., Zhang, P., Krishnamoorthi, R., Chandra, V., Xiong, Y., and Elhoseiny, M. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023.

Chen, Z., Zhao, Z., Luo, H., Yao, H., Li, B., and Zhou, J. Halc: Object hallucination reduction via adaptive focalcontrast decoding. arXiv preprint arXiv:2403.00425, 2024.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, March 2023.

Cho, J., Hu, Y., Garg, R., Anderson, P., Krishna, R., Baldridge, J., Bansal, M., Pont-Tuset, J., and Wang, S. Davidsonian scene graph: Improving reliability in finegrained evaluation for text-to-image generation, 2024. URL https://arxiv.org/abs/2310.18235.

Dai, W., Li, J., Li, D., Tiong, A. M. H., Zhao, J., Wang, W., Li, B., Fung, P., and Hoi, S. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023a.

Dai, W., Liu, Z., Ji, Z., Su, D., and Fung, P. Plausible may not be faithful: Probing object hallucination in visionlanguage pre-training. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 2128–2140, 2023b.

Deng, Y., Lu, P., Yin, F., Hu, Z., Shen, S., Zou, J., Chang, K.-W., and Wang, W. Enhancing large vision language models with self-training on image comprehension. arXiv preprint arXiv:2405.19716, 2024.

Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.   
Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023.   
Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., Huang, T., Wang, X., and Cao, Y. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19358–19369, 2023.   
Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Qiu, Z., Lin, W., Yang, J., Zheng, X., et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.   
Gao, P., Han, J., Zhang, R., Lin, Z., Geng, S., Zhou, A., Zhang, W., Lu, P., He, C., Yue, X., Li, H., and Qiao, Y. Llama-adapter v2: Parameter-efficient visual instruction model, 2023.   
Gunjal, A., Yin, J., and Bas, E. Detecting and preventing hallucinations in large vision language models. arXiv preprint arXiv:2308.06394, 2023.   
Ho, J. and Salimans, T. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.   
Hu, Y., Liu, B., Kasai, J., Wang, Y., Ostendorf, M., Krishna, R., and Smith, N. A. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering, 2023. URL https://arxiv.org/abs/ 2303.11897.   
Hu, Z. and Li, L. E. A causal lens for controllable text generation. Advances in Neural Information Processing Systems, 34:24941–24955, 2021.   
Huang, Q., Dong, X., Zhang, P., Wang, B., He, C., Wang, J., Lin, D., Zhang, W., and Yu, N. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. arXiv preprint arXiv:2311.17911, 2023a.   
Huang, X., Huang, Y.-J., Zhang, Y., Tian, W., Feng, R., Zhang, Y., Xie, Y., Li, Y., and Zhang, L. Open-set image tagging with multi-grained text supervision, 2023b. URL https://arxiv.org/abs/2310.15200.   
Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question

answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700– 6709, 2019.   
Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., and Fung, P. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023.   
Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pp. 4904–4916. PMLR, 2021.   
Jing, L., Li, R., Chen, Y., Jia, M., and Du, X. Faithscore: Evaluating hallucinations in large vision-language models. arXiv preprint arXiv:2311.01477, 2023.   
Kar, O. F., Tonioni, A., Poklukar, P., Kulshrestha, A., Zamir, A., and Tombari, F. Brave: Broadening the visual encoding of vision-language models, 2024. URL https://arxiv.org/abs/2404.07204.   
Leng, S., Zhang, H., Chen, G., Li, X., Lu, S., Miao, C., and Bing, L. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. arXiv preprint arXiv:2311.16922, 2023.   
Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023a.   
Li, X., Thickstun, J., Gulrajani, I., Liang, P. S., and Hashimoto, T. B. Diffusion-lm improves controllable text generation. Advances in Neural Information Processing Systems, 35:4328–4343, 2022.   
Li, X. L. and Liang, P. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 4582–4597, 2021.   
Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models. arXiv preprint arXiv:2305.10355, 2023b.   
Lin, C.-Y. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pp. 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https: //aclanthology.org/W04-1013.

Lin, S.-C., Li, M., and Lin, J. Aggretriever: A simple approach to aggregate textual representations for robust dense passage retrieval. Transactions of the Association for Computational Linguistics, 11:436–452, 2023a.   
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft coco: ´ Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740– 755. Springer, 2014.   
Lin, Z., Madotto, A., Bang, Y., and Fung, P. The adapter-bot: All-in-one controllable conversational model. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 16081–16083, 2021.   
Lin, Z., Gong, Y., Shen, Y., Wu, T., Fan, Z., Lin, C., Duan, N., and Chen, W. Text generation with diffusion language models: A pre-training approach with continuous paragraph denoise. In International Conference on Machine Learning, pp. 21051–21064. PMLR, 2023b.   
Lin, Z., Pathak, D., Li, B., Li, J., Xia, X., Neubig, G., Zhang, P., and Ramanan, D. Evaluating text-to-visual generation with image-to-text generation, 2024. URL https://arxiv.org/abs/2404.01291.   
Liu, F., Lin, K., Li, L., Wang, J., Yacoob, Y., and Wang, L. Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023a.   
Liu, F., Lin, K., Li, L., Wang, J., Yacoob, Y., and Wang, L. Mitigating hallucination in large multi-modal models via robust instruction tuning, 2023b.   
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023c.   
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In NeurIPS, 2023d.   
Liu, S., Ye, H., Xing, L., and Zou, J. Reducing hallucinations in vision-language models via latent space steering, 2024a. URL https://arxiv.org/abs/ 2410.15778.   
Liu, S., Zheng, K., and Chen, W. Paying more attention to image: A training-free method for alleviating hallucination in lvlms. arXiv preprint arXiv:2407.21771, 2024b.   
Lovenia, H., Dai, W., Cahyawijaya, S., Ji, Z., and Fung, P. Negative object presence evaluation (nope) to measure object hallucination in vision-language models. arXiv preprint arXiv:2310.05338, 2023.

Lu, J., Yang, J., Batra, D., and Parikh, D. Neural baby talk. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7219–7228, 2018. doi: 10.1109/ CVPR.2018.00754.   
Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts, 2024.   
Madaan, A., Setlur, A., Parekh, T., Poczos, B., Neubig, G., ´ Yang, Y., Salakhutdinov, R., Black, A. W., and Prabhumoye, S. Politeness transfer: A tag and generate approach. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 1869–1881, 2020.   
Niu, T. and Bansal, M. Polite dialogue generation without parallel data. Transactions of the Association for Computational Linguistics, 6:373–389, 2018.   
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.   
Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318, 2002.   
Petryk, S., Chan, D., Kachinthaya, A., Zou, H., Canny, J., Gonzalez, J., and Darrell, T. ALOHa: A new measure for hallucination in captioning models. In Duh, K., Gomez, H., and Bethard, S. (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pp. 342–357, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-short. 30. URL https://aclanthology.org/2024. naacl-short.30/.   
Prabhumoye, S., Black, A. W., and Salakhutdinov, R. Exploring controllable text generation techniques. In Proceedings of the 28th International Conference on Computational Linguistics, pp. 1–14, 2020.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Ribeiro, L. F., Zhang, Y., and Gurevych, I. Structural adapters in pretrained language models for amr-to-text generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 4269–4282, 2021.   
Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., and Saenko, K. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 4035–4045, 2018.   
Sanchez, G., Fan, H., Spangher, A., Levi, E., Ammanamanchi, P. S., and Biderman, S. Stay on topic with classifier-free guidance. arXiv preprint arXiv:2306.17806, 2023.   
Schwenk, D., Khandelwal, A., Clark, C., Marino, K., and Mottaghi, R. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pp. 146–162. Springer, 2022.   
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
Vedantam, R., Zitnick, C. L., and Parikh, D. Cider: Consensus-based image description evaluation, 2015.   
Wan, D., Cho, J., Stengel-Eskin, E., and Bansal, M. Contrastive region guidance: Improving grounding in vision-language models without training. arXiv preprint arXiv:2403.02325, 2024.   
Wang, B., Wu, F., Han, X., Peng, J., Zhong, H., Zhang, P., Dong, X., Li, W., Li, W., Wang, J., et al. Vigc: Visual instruction generation and correction. arXiv preprint arXiv:2308.12714, 2023a.   
Wang, J., Zhou, Y., Xu, G., Shi, P., Zhao, C., Xu, H., Ye, Q., Yan, M., Zhang, J., Zhu, J., et al. Evaluation and analysis of hallucination in large vision-language models. arXiv preprint arXiv:2308.15126, 2023b.   
Xu, P., Shao, W., Zhang, K., Gao, P., Liu, S., Lei, M., Meng, F., Huang, S., Qiao, Y., and Luo, P. Lvlm-ehub: A comprehensive evaluation benchmark for large visionlanguage models. arXiv preprint arXiv:2306.09265, 2023.   
Yang, L., Zheng, Z., Chen, B., Zhao, Z., Lin, C., and Shen, C. Nullu: Mitigating object hallucinations in large visionlanguage models via halluspace projection, 2025. URL https://arxiv.org/abs/2412.13817.

Yang, Z., Wang, J., Gan, Z., Li, L., Lin, K., Wu, C., Duan, N., Liu, Z., Liu, C., Zeng, M., et al. Reco: Regioncontrolled text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14246–14255, 2023.   
Ye, Q., Xu, H., Xu, G., Ye, J., Yan, M., Zhou, Y., Wang, J., Hu, A., Shi, P., Shi, Y., et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.   
Yin, S., Fu, C., Zhao, S., Xu, T., Wang, H., Sui, D., Shen, Y., Li, K., Sun, X., and Chen, E. Woodpecker: Hallucination correction for multimodal large language models. arXiv preprint arXiv:2310.16045, 2023.   
Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., Hu, J., Liu, Z., Zheng, H.-T., Sun, M., et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13807–13816, 2024.   
Zhai, B., Yang, S., Xu, C., Shen, S., Keutzer, K., and Li, M. Halle-switch: Controlling object hallucination in large vision language models. arXiv e-prints, pp. arXiv–2310, 2023.   
Zhang, H., Song, H., Li, S., Zhou, M., and Song, D. A survey of controllable text generation using transformerbased pre-trained language models. ACM Computing Surveys, 56(3):1–37, 2023a.   
Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.-W., Gao, P., et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024a.   
Zhang, Y., Qian, S., Peng, B., Liu, S., and Jia, J. Prompt highlighter: Interactive control for multi-modal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13215–13224, 2024b.   
Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., and Smola, A. Multimodal chain-of-thought reasoning in language models, 2023b.   
Zhou, Y., Cui, C., Yoon, J., Zhang, L., Deng, Z., Finn, C., Bansal, M., and Yao, H. Analyzing and mitigating object hallucination in large vision-language models. arXiv preprint arXiv:2310.00754, 2023.   
Zhou, Y., Cui, C., Rafailov, R., Finn, C., and Yao, H. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

# A Experiment Setup

We conduct all of the experiments using 8 A6000 GPU with 48GB GPU memory. Each single experiment can be run on a single A6000 GPU.

# A.1 Model Architectures

In Table 8, we provide detailed descriptions of the LVLM architectures used in our experiments. These LVLMs respectively leverage the pre-trained vision encoder of the models we listed, which are all based on the Vision Transformer (ViT) (Dosovitskiy et al., 2020) architecture.

Table 8. Details of the LVLM architectures that we used in our paper. 

<table><tr><td>Model</td><td>Vision encoder</td><td>LLM</td></tr><tr><td>LLaVA (Liu et al., 2023d)</td><td>CLIP-L (Radford et al., 2021)</td><td>LLaMA-2-7B-Chat (Touvron et al., 2023)</td></tr><tr><td>LLaVA-v1.5 (Liu et al., 2023c)</td><td>CLIP-L-336px (Radford et al., 2021)</td><td>Vicuna-v1.5-7B (Chiang et al., 2023)</td></tr><tr><td>MiniGPT-v2 (Chen et al., 2023)</td><td>EVA-G (Fang et al., 2023)</td><td>LLaMA-2-7B-Chat (Touvron et al., 2023)</td></tr><tr><td>mPLUG-OWL2 (Ye et al., 2023)</td><td>CLIP-L (Radford et al., 2021)</td><td>LLaMA-2-7B (Touvron et al., 2023)</td></tr><tr><td>InstructBLIP (Dai et al., 2023a)</td><td>BLIP-2 (Li et al., 2023a)</td><td>Vicuna-v1.1-7B (Chiang et al., 2023)</td></tr></table>

# A.2 Descriptions about Additional Metrics

In Figure 2, we evaluate the text quality of the outputs generated with MARINE using general metrics as follows:

• BLEU (Papineni et al., 2002) measures how well the generated translation matches the reference translations in terms of n-gram overlap.   
• ROUGE-L (Lin, 2004) measures the quality of a machine-generated summary by comparing it to one or more reference summaries.   
• CIDEr (Vedantam et al., 2015) assesses the quality of image captioning models. It focuses on evaluating how well the generated captions align with human consensus.   
• SPICE (Anderson et al., 2016) focuses on assessing the semantic similarity between the generated captions and reference captions.

# A.3 Prompt Templates

For each query, we randomly select a prompt template from the available template list, as shown in Table 9.

# A.4 Details of Baselines

Specifically, the hyperparameters for LURE (Zhou et al., 2023), VCD (Leng et al., 2023), OPERA (Huang et al., 2023a) are reported in Table 10, 11 and 12 respectively. We strictly followed the original implementations and default hyperparameters described in their papers to reproduce the results for each baseline.

# A.5 Experiment Setting for Hallucination Evaluations

Key factors that potentially affect the hallucination evaluation outcomes, including the evaluation dataset and prompt template, LVLM’s sampling strategy and batched generation techniques, and guidance strength, are detailed in this section. The hyper-parameters setting for MARINE and overall experiment settings are shown in Table 13 and 14.

Experiment setting for CHAIR evaluation. We adopt the same prompt “Generate a short caption of the image.” as utilized by Li et al. (2023b). The hyperparameters are fixed, including a guidance strength of 0.7, score threshold for DETR at 0.95, a detection threshold for RAM++ of 0.68, a maximum token length of 64, and a greedy sampling approach with a random seed of 242.

For the calculation of CHAIR metrics, we referenced the 80 object categories annotated in the MSCOCO dataset, following Rohrbach et al. (2018). Besides, we employed the synonym list from Lu et al. (2018) to align synonymous words in the generated text with MSCOCO object categories. Additionally, due to the cost considerations associated with the GPT-3.5 API, we limited our analysis to 200 samples for Woodpecker correction for each model and reported the result in Table 1.

Experiment setting for POPE evaluation. POPE is a flexible approach to evaluating hallucinations in LVLMs, which formulates a binary classification task by prompting LVLMs with questions such as “Is there a keyboard in this image?” to answer “yes” or “no”. Following Li et al. (2023b), we created 3000 POPE questions across three datasets—500 images each from MSCOCO, A-OKVQA, and GQA for the POPE evaluation. We reported the adversarial settings in Table 2, the most challenging setting, which constructs POPE questions from the top-k most frequently co-occurring but absent objects. Additionally, in Table 4, we reported the average scores under random, popular, adversarial settings across MSCOCO,

Table 9. Details of the LVLM architectures that we used in our paper. 

<table><tr><td>Template Type</td><td>Prompt Template</td></tr><tr><td rowspan="4">MARINE-intersec</td><td>This image contains. Based on this,</td></tr><tr><td>The image contains the following objects:. Given these detected objects,</td></tr><tr><td>This image shows the following objects:. Using this information,</td></tr><tr><td>The objects found in this image are:. Considering this list of objects,</td></tr><tr><td rowspan="4">POPE task</td><td>This image contains only the following objects:. Do not assume anything beyond these objects. Based solely on this list,</td></tr><tr><td>The detected objects in the image are:. Answer based only on these objects.</td></tr><tr><td>This image shows the following objects:. You must answer using only the objects in this list. Given these detected objects,</td></tr><tr><td>The objects found in this image are limited to:. You should rely strictly on this list of objects and make no other guesses. Based on this,</td></tr><tr><td rowspan="8">MARINE-union</td><td>List of detected objects in the image:</td></tr><tr><td>Based on the detected objects above,</td></tr><tr><td>The most prominent objects detected are:</td></tr><tr><td>Given these findings,</td></tr><tr><td>The following objects were detected in the image:</td></tr><tr><td>With this information,</td></tr><tr><td>Here is a list of all objects detected in the image:</td></tr><tr><td>Do not infer or hallucinate any additional objects. Using only the detected objects,</td></tr></table>

Table 10. LURE (Zhou et al., 2023) Hyperparameter Settings 

<table><tr><td>Parameters</td><td>Value</td></tr><tr><td>Uncertainty Threshold γ</td><td>0.9</td></tr><tr><td>Position Threshold l</td><td>0.8</td></tr></table>

Table 11. VCD (Leng et al., 2023) Hyperparameter Settings 

<table><tr><td>Parameters</td><td>Value</td></tr><tr><td>Amplification Factor α</td><td>1</td></tr><tr><td>Adaptive Plausibility Threshold</td><td>0.1</td></tr><tr><td>Diffusion Noise Step</td><td>500</td></tr></table>

Table 12. OPERA (Huang et al., 2023a) Hyperparameter Settings 

<table><tr><td>Parameters</td><td>Value</td></tr><tr><td>Self-attention Weights Scale Factor θ</td><td>50</td></tr><tr><td>Attending Retrospection Threshold</td><td>25</td></tr><tr><td>Beam Size</td><td>5</td></tr><tr><td>Attention Candidates</td><td>1</td></tr><tr><td>Penalty Weights</td><td>1</td></tr></table>

Table 13. MARINE Hyperparameter Settings. The settings are fixed depending on the question-answering tasks. 

<table><tr><td>Parameters</td><td>Value</td></tr><tr><td colspan="2">Guidance</td></tr><tr><td>Guidance Strength</td><td>0.7</td></tr><tr><td>score threshold for DETR</td><td>0.95</td></tr><tr><td>Detect Threshold for RAM++</td><td>0.68</td></tr><tr><td colspan="2">Generation</td></tr><tr><td>Max Token Length</td><td>64</td></tr><tr><td>Sampling</td><td>Greedy</td></tr><tr><td>Random Seed</td><td>242</td></tr></table>

Table 14. Batch size for LVLM generation is fixed across all experiments unless otherwise noted. To expedite the evaluation process, we employed the batched generation. We avoid the negative impact of batched generation by adopting left padding if the LVLM does not explicitly assign the padding strategy for inference. 

<table><tr><td>Model</td><td>LLaVA</td><td>LLaVA-v1.5</td><td>MiniGPTv</td><td>mPLUG-Owl2</td><td>InstructBLIP</td></tr><tr><td>Batch Size</td><td>16</td><td>4</td><td>32</td><td>16</td><td>16</td></tr></table>

A-OKVQA, and GQA datasets. The full POPE results are in Tabel 16.

Similarly, we constrained our analysis to 200 samples for Woodpecker correction for each model due to the high costs associated with the GPT API. The outcomes of this analysis are detailed in Table 2.

Experiment setting for GPT-4V-aided evaluation. The GPT-4V-aided evaluation compares the outputs of two LVLM assistants using GPT-4V as a judge. We prompted GPT-4V to assess the quality of the generated outputs, scoring them out of 10 in two aspects:

• Accuracy: how accurately each assistant describes the image;   
• Detailedness: the richness of necessary details in the response.

As shown in Table 15, the assessment prompt template we used is slightly different from that of Yin et al. (2023). Specifically, we include the original question for a task-orientated evaluation and exclude prompts that describe Woodpecker-specific output formats like object bounding boxes.

Experiment setting for ablation study. To explore different methods of integrating image-grounding models, we investigate the intersection and union of detected objects, with integration based on synonyms using the NLTK package. To quantitatively assess the influence of guidance strength, we varied it from 0 to 1, as shown in Figure 7. These quantitative experiments were conducted using the same setting as those in CHAIR evaluation. For qualitative analysis, we selected guidance strength from a recommended range of $\gamma \in ( 0 . 3 , 0 . 7 )$ .

# A.6 Experiment Setting on Other Vision-Language Tasks

Experiment setting for text quality analysis. For text quality analysis, we adopted 90 visual questions from the LLaVA-QA90 1 task (including conversations, visual perceptions, and complex reasoning subtasks), and randomly selected 500 MSCOCO images for image captioning task. Following Liu et al. (2023d), we adpoted the response generated by text-only GPT-4 (0314) with the context captions/boxes provided. answers given by GPT-4 as references for LLaVA-QA90 task and used image captions provided in MSCOCO annotations as references for image captioning task.

In Table 17 and Table 18, we present a detailed evaluation on the image captioning task for both MSCOCO and LLaVA-QA90 using metrics including BLEU, ROUGE, CIDEr and SPICE. The corresponding figure result is shown in Figure 2.

Experiment setting for latency analysis. We compared our method with existing baselines in terms of the trade-off between inference cost and the effectiveness of reducing object hallucinations, as shown in Table 5. For post-correction baselines such as Woodpecker and LURE, we first prompted LLaVA (llava-llama-2-7b-chat-lightning-preview) to generate captions and then measure the latency of generating the corrected outputs. The total latency for post-correction baselines includes both the generation and correction processes. For decoding methods such as VCD, OPERA and our method, we measured the latency of LLaVA generating captions directly.

We prompted the models with “Generate a short caption of the image.” on 500 MSCOCO images with a batch size of 1 and a maximum token length of 64, without any stopping criteria, using a single A6000 GPU. Then latency was calculated as the ratio of the number of output tokens and encoding and generation time.

Table 15. Prompt template for GPT-4V-aided evaluation. {question} is the original instruction; {answer 1} is the original response, and {answer 2} is the response generated by the LVLM using MARINE. 

<table><tr><td>Prompt template for GPT-4V-aided evaluation</td></tr><tr><td>You are required to score the performance of two AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image.</td></tr><tr><td>Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:1. Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinations should be given higher scores.2. Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not count as necessary details.</td></tr><tr><td>Please output a single line for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.</td></tr><tr><td>Question: {question}Assistant 1: {answer 1}Assistant 2: {answer 2}</td></tr><tr><td>Output format:Accuracy:Scores of the two answers:Reason:Detailedness:Scores of the two answers:Reason:</td></tr></table>

Table 16. Detailed POPE (Li et al., 2023b) results on three datasets (MSCOCO (Lin et al., 2014), A-OKVQA (Schwenk et al., 2022), GQA (Hudson & Manning, 2019)). 

<table><tr><td>Dataset</td><td>Type</td><td>Model</td><td>w/MARINE</td><td>Accuracy ↑</td><td>Precision ↑</td><td>Recall ↑</td><td>F1 ↑</td><td>Yes(%)</td></tr><tr><td rowspan="12">MSCOCO</td><td rowspan="4">Adversarial</td><td rowspan="2">LLaVA</td><td>✕</td><td>51.8</td><td>50.9</td><td>99.5</td><td>67.4</td><td>97.7</td></tr><tr><td>√</td><td>66.9</td><td>61.7</td><td>89.1</td><td>72.9</td><td>72.3</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✕</td><td>72.5</td><td>65.5</td><td>94.9</td><td>77.5</td><td>72.4</td></tr><tr><td>√</td><td>82.8</td><td>83.4</td><td>82.0</td><td>82.7</td><td>49.2</td></tr><tr><td rowspan="4">Popular</td><td rowspan="2">LLaVA</td><td>✕</td><td>52.4</td><td>51.2</td><td>99.8</td><td>67.7</td><td>97.4</td></tr><tr><td>√</td><td>71.3</td><td>65.8</td><td>88.9</td><td>75.6</td><td>67.5</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✕</td><td>75.8</td><td>68.7</td><td>94.9</td><td>79.7</td><td>69.0</td></tr><tr><td>√</td><td>85.6</td><td>88.4</td><td>82.0</td><td>85.1</td><td>46.4</td></tr><tr><td rowspan="4">Random</td><td rowspan="2">LLaVA</td><td>✕</td><td>58.3</td><td>54.5</td><td>99.7</td><td>70.5</td><td>91.4</td></tr><tr><td>√</td><td>78.5</td><td>73.4</td><td>89.3</td><td>80.6</td><td>60.8</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✕</td><td>81.8</td><td>75.2</td><td>94.9</td><td>83.9</td><td>63.1</td></tr><tr><td>√</td><td>88.1</td><td>93.4</td><td>81.9</td><td>87.3</td><td>43.9</td></tr><tr><td rowspan="12">A-OKVQA</td><td rowspan="4">Adversial</td><td rowspan="2">LLaVA</td><td>✕</td><td>50.0</td><td>50.0</td><td>99.5</td><td>66.6</td><td>99.5</td></tr><tr><td>√</td><td>56.3</td><td>53.6</td><td>94.3</td><td>68.3</td><td>88.1</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✕</td><td>62.5</td><td>57.3</td><td>98.1</td><td>72.3</td><td>85.6</td></tr><tr><td>√</td><td>74.4</td><td>68.8</td><td>89.3</td><td>77.7</td><td>64.9</td></tr><tr><td rowspan="4">Popular</td><td rowspan="2">LLaVA</td><td>✕</td><td>50.1</td><td>50.1</td><td>99.8</td><td>66.7</td><td>99.7</td></tr><tr><td>√</td><td>63.0</td><td>58.0</td><td>94.5</td><td>71.9</td><td>81.6</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✕</td><td>69.1</td><td>62.1</td><td>97.9</td><td>76.0</td><td>78.9</td></tr><tr><td>√</td><td>82.5</td><td>78.8</td><td>89.1</td><td>83.6</td><td>56.5</td></tr><tr><td rowspan="4">Random</td><td rowspan="2">LLaVA</td><td>✕</td><td>55.4</td><td>52.8</td><td>99.8</td><td>69.1</td><td>94.4</td></tr><tr><td>√</td><td>73.7</td><td>66.7</td><td>94.7</td><td>78.3</td><td>71.0</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✕</td><td>77.2</td><td>69.2</td><td>98.2</td><td>81.2</td><td>71.0</td></tr><tr><td>√</td><td>89.2</td><td>89.2</td><td>89.3</td><td>89.2</td><td>50.1</td></tr><tr><td rowspan="12">GQA</td><td rowspan="4">Adversial</td><td rowspan="2">LLaVA</td><td>✕</td><td>50.3</td><td>50.1</td><td>99.8</td><td>66.8</td><td>99.5</td></tr><tr><td>√</td><td>54.4</td><td>52.5</td><td>93.8</td><td>67.3</td><td>89.4</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✕</td><td>68.4</td><td>63.0</td><td>98.2</td><td>75.6</td><td>79.8</td></tr><tr><td>√</td><td>76.0</td><td>73.6</td><td>81.2</td><td>77.2</td><td>55.2</td></tr><tr><td rowspan="4">Popular</td><td rowspan="2">LLaVA</td><td>✕</td><td>50.1</td><td>50.0</td><td>99.8</td><td>66.7</td><td>99.7</td></tr><tr><td>√</td><td>58.7</td><td>55.1</td><td>94.3</td><td>69.5</td><td>85.5</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✕</td><td>70.6</td><td>63.8</td><td>94.9</td><td>76.3</td><td>74.4</td></tr><tr><td>√</td><td>77.6</td><td>75.6</td><td>81.3</td><td>78.4</td><td>53.8</td></tr><tr><td rowspan="4">Random</td><td rowspan="2">LLaVA</td><td>✕</td><td>55.7</td><td>53.0</td><td>99.8</td><td>69.2</td><td>94.1</td></tr><tr><td>√</td><td>74.3</td><td>67.3</td><td>94.8</td><td>78.7</td><td>70.5</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✕</td><td>82.0</td><td>75.2</td><td>95.5</td><td>84.1</td><td>63.5</td></tr><tr><td>√</td><td>86.8</td><td>91.5</td><td>81.3</td><td>86.1</td><td>44.4</td></tr></table>

Table 17. Performance on general metrics for the image captioning task, including BLEU (Papineni et al., 2002), ROUGE-L (Lin, 2004), CIDEr (Vedantam et al., 2015) and SPICE (Anderson et al., 2016) scores(%). 

<table><tr><td>Model</td><td>w/MARINE</td><td>BLEU_1 (↑)</td><td>BLEU_2 (↑)</td><td>BLEU_3 (↑)</td><td>BLEU_4 (↑)</td><td>ROUGE_L (↑)</td><td>CIDEr (↑)</td><td>SPICE (↑)</td></tr><tr><td rowspan="2">LLaVA</td><td>X</td><td>14.06</td><td>7.12</td><td>3.72</td><td>1.90</td><td>22.06</td><td>0.08</td><td>16.77</td></tr><tr><td>√</td><td>18.59</td><td>9.96</td><td>5.47</td><td>3.04</td><td>26.02</td><td>0.21</td><td>20.58</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>X</td><td>39.91</td><td>25.16</td><td>16.57</td><td>11.24</td><td>36.26</td><td>1.05</td><td>26.82</td></tr><tr><td>√</td><td>39.51</td><td>24.37</td><td>15.93</td><td>10.70</td><td>36.01</td><td>1.03</td><td>27.42</td></tr></table>

# B Additional Experiments

# B.1 Additional Baselines

To further contextualize the effectiveness of MARINE, we conducted additional experiments comparing our approach to a baseline that employs carefully engineered prompts designed to reduce hallucination. Specifically, we used the following prompt:

Table 18. Performance on general metrics for the LLaVA-QA90 task, including BLEU (Papineni et al., 2002), ROUGE-L (Lin, 2004), CIDEr (Vedantam et al., 2015) and SPICE (Anderson et al., 2016) scores(%). 

<table><tr><td>Model</td><td>w/MARINE</td><td>BLEU_1 (↑)</td><td>BLEU_2 (↑)</td><td>BLEU_3 (↑)</td><td>BLEU_4 (↑)</td><td>ROUGE_L (↑)</td><td>CIDEr (↑)</td><td>SPICE (↑)</td></tr><tr><td rowspan="2">LLaVA</td><td>✗</td><td>21.02</td><td>12.91</td><td>8.79</td><td>6.41</td><td>32.30</td><td>0.93</td><td>31.36</td></tr><tr><td>√</td><td>23.37</td><td>14.39</td><td>9.59</td><td>6.83</td><td>33.81</td><td>0.99</td><td>31.91</td></tr><tr><td rowspan="2">mPLUG-Owl2</td><td>✗</td><td>44.50</td><td>28.57</td><td>19.58</td><td>14.43</td><td>40.24</td><td>1.46</td><td>40.51</td></tr><tr><td>√</td><td>45.82</td><td>28.87</td><td>19.24</td><td>13.70</td><td>38.54</td><td>1.29</td><td>38.70</td></tr></table>

Table 19. Comparison against carefully engineered prompts. 

<table><tr><td>Method</td><td colspan="3">LLaVA</td><td colspan="3">LLaVA-v1.5</td><td colspan="3">mPLUG-Owl2</td></tr><tr><td>CHAIR</td><td> $C_s \downarrow$ </td><td> $C_i \downarrow$ </td><td>Recall ↑</td><td> $C_s \downarrow$ </td><td> $C_i \downarrow$ </td><td>Recall ↑</td><td> $C_s \downarrow$ </td><td> $C_i \downarrow$ </td><td>Recall ↑</td></tr><tr><td>Original</td><td>26.6</td><td>10.5</td><td>47.4</td><td>8.8</td><td>4.6</td><td>41.1</td><td>5.0</td><td>3.2</td><td>33.2</td></tr><tr><td>Direct Prompting</td><td>27.2</td><td>11.0</td><td>46.4</td><td>19.6</td><td>8.3</td><td>52.3</td><td>9.0</td><td>5.1</td><td>42.0</td></tr><tr><td>Prompts as Additional Guidance</td><td>37.4</td><td>10.5</td><td>50.4</td><td>12.6</td><td>5.9</td><td>44.6</td><td>6.6</td><td>3.9</td><td>40.4</td></tr><tr><td>MARINE (ours)</td><td>17.8</td><td>7.2</td><td>50.8</td><td>6.2</td><td>3.0</td><td>44.3</td><td>4.2</td><td>2.3</td><td>41.4</td></tr></table>

Table 20. Experiments on dynamic guidance strength based on confidence scores on CHAIR metrics. 

<table><tr><td>Method</td><td colspan="3">LLaVA</td><td colspan="3">mPLUG-Owl2</td></tr><tr><td>CHAIR</td><td> $C_s \downarrow$ </td><td> $C_i \downarrow$ </td><td>Recall ↑</td><td> $C_s \downarrow$ </td><td> $C_i \downarrow$ </td><td>Recall ↑</td></tr><tr><td>Fix Guidance Strength</td><td>17.8</td><td>7.2</td><td>50.8</td><td>4.2</td><td>2.3</td><td>41.4</td></tr><tr><td>Dynamic Guidance Strength</td><td>14.8</td><td>6.5</td><td>49.9</td><td>5.0</td><td>2.6</td><td>41.0</td></tr></table>

Table 21. Experiments on dynamic guidance strength based on confidence scores on POPE metrics. 

<table><tr><td>Method</td><td colspan="3">LLaVA</td><td colspan="3">mPLUG-Owl2</td></tr><tr><td>POPE</td><td>Accuracy ↑</td><td>F1 ↑</td><td>Yes Ratio</td><td>Accuracy ↑</td><td>F1 ↑</td><td>Yes Ratio</td></tr><tr><td>Fix Guidance Strength</td><td>66.9</td><td>72.9</td><td>72.3</td><td>82.8</td><td>82.7</td><td>49.2</td></tr><tr><td>Dynamic Guidance Strength</td><td>71.97</td><td>74.48</td><td>59.83</td><td>83.3</td><td>83.2</td><td>49.4</td></tr></table>

Describe the visible contents of this image in as much detail as possible without adding any information not clearly visible. Only mention objects, colors, shapes, and textures that can be directly observed in the image, avoiding assumptions about materials, functions, or contexts. If there are any uncertainties about what an object is, describe its visual characteristics (e.g., ’a circular object with a smooth surface’) without inferring its purpose or identity. Avoid creative or hypothetical descriptions, and focus on observable details only.

With two different settings:

• Direct Prompting: The original input query was replaced with the prompts as described.   
• Prompts as Additional Guidance: We incorporated the prompt as supplemental context to guide the models in generating outputs.

As shown in Table 19, prompt-based guidance can improve recall for some models (e.g., LLaVA-v1.5), but does not consistently reduce hallucinations across all metrics. In fact, CHAIR scores often worsen. In contrast, MARINE achieves stronger improvements across all models.

We highlight two key differences between MARINE and prompt-based approaches:

• Model Dependence: Prompting methods rely heavily on the instruction-following capabilities of the model. While they may reduce hallucinations slightly for stronger models (e.g., LLaVA-v1.5), they can worsen performance in weaker models (e.g., LLaVA). Additionally, prompt-based approaches may require fine-tuning to be effective (Deng et al., 2024). MARINE, by contrast, improves grounding through explicit visual signals, making it effective even without model retraining.

• Generalization and Efficiency: Prompting methods often require task-specific tuning or dataset-aware phrasing. MARINE generalizes across tasks and models with minimal engineering and no fine-tuning, while offering more consistent hallucination reduction.

# B.2 Dynamic Guidance Strength

We conducted additional experiments to compare fixed and dynamic guidance strength strategies using both CHAIR and POPE metrics (Tables 20 and 21).

• Fix Guidance Strength uses a fixed guidance strength of 0.7, selected to balance hallucination reduction and instructions adherence.   
• Dynamic Guidance Strength adjusts the guidance strength dynamically by mapping the mean confidence score (s) of the image-grounding models to a range of (0.4, 0.8) using the formula

$$
\gamma^ {\prime} = 0. 4 + \frac {(0 . 8 - 0 . 4) \cdot (s - s _ {\mathrm{min}})}{s _ {\mathrm{max}} - s _ {\mathrm{min}}}.
$$

A higher confidence score indicates more reliable grounding, which results in stronger guidance. Empirically, we find that dynamic guidance improves performance for weaker models such as LLaVA, which are more sensitive to noisy signals. For stronger models like mPLUG-Owl2, a fixed guidance strength is already sufficient to reduce object hallucinations effectively.

# B.3 Effect of Sampling Temperature

In our main experiments, we use greedy decoding (temperature = 0) to ensure deterministic outputs and reproducible comparisons—consistent with our primary baseline (VCD) and common practice in hallucination benchmarks. To evaluate robustness under stochastic decoding, we also test with a temperature of 0.6 and report mean ± standard deviation in Table 22. MARINE continues to outperform baseline generations across all hallucination metrics, demonstrating effectiveness regardless of sampling strategy.

Table 22. Object hallucination metrics under temperature = 0.6 sampling. 

<table><tr><td>Method</td><td colspan="3">LLaVA</td><td colspan="3">mPLUG-Owl2</td></tr><tr><td>CHAIR</td><td> $C_s \downarrow$ </td><td> $C_i \downarrow$ </td><td>Recall↑</td><td> $C_s \downarrow$ </td><td> $C_i \downarrow$ </td><td>Recall↑</td></tr><tr><td>Greedy</td><td> $26.1_{\pm 1.6}$ </td><td> $10.8_{\pm 0.5}$ </td><td> $46.0_{\pm 0.8}$ </td><td> $4.9_{\pm 0.6}$ </td><td> $2.8_{\pm 0.3}$ </td><td> $37.7_{\pm 0.6}$ </td></tr><tr><td>MARINE (ours)</td><td> $\mathbf{19.3}_{\pm 0.8}$ </td><td> $\mathbf{7.6}_{\pm 0.1}$ </td><td> $\mathbf{50.6}_{\pm 0.2}$ </td><td> $\mathbf{4.5}_{\pm 0.6}$ </td><td> $\mathbf{2.4}_{\pm 0.2}$ </td><td> $\mathbf{41.1}_{\pm 0.4}$ </td></tr></table>

# B.4 Memory Analysis

We evaluated the peak GPU memory usage during inference on 500 image captioning examples using the LLaVA model, with a batch size of 16 and a maximum generation length of 64 tokens. Results are reported in Table 23. Although MARINE introduces additional vision models, the overall memory footprint increases by only approximately 30% during inference—significantly less than doubling. This is because the object detection models are relatively lightweight compared to the large LLM backbone.

# B.5 Further Study on Guidance Strength

Figure 5 shows how varying the guidance strength γ affects the quality of LLaVA’s output on the LLaVA-QA90 and image captioning tasks (max generation length = 256). We observe that setting $\gamma = 1$ does not yield the best image captioning performance. In the LLaVA-QA90 task, guidance strengths in the range of 0.5 to 0.7 lead to higher output quality. This observation is consistent with prior findings in classifier-free guidance literature: overly strong guidance can dominate the generation process and reduce fluency or instruction adherence.

To further validate these results, we use GPT-4V as an automatic judge to score outputs (on a 10-point scale) for accuracy and detail. The results, summarized in Table 24, show that balancing the original LVLM branch leads to improved generation quality. Finally, Figure 6 provides qualitative examples showing how excessive guidance can reduce instruction alignment, often introducing unnecessary visual details into the response.

Table 23. Peak GPU Memory Usage during Inference (GB) of MARINE compared to greedy decoding and VCD. 

<table><tr><td>Metric</td><td>Greedy</td><td>VCD</td><td>MARINE (Ours)</td></tr><tr><td>Peak GPU Memory Usage</td><td>23.53</td><td>20.73 ( $\times$ 0.88)</td><td>30.78 ( $\times$ 1.30)</td></tr></table>

Table 24. Results of GPT-4V-aided evaluation. The accuracy and detailedness metrics are on a scale of 10, and a higher score indicates better performance. The symbols × and ✓ indicate performance metrics without and with our method, respectively. 

<table><tr><td>Task</td><td>Metric ↑</td><td> $\mathbf{X} (\gamma = 1)$ </td><td> $\checkmark (\gamma = 0.7)$ </td></tr><tr><td rowspan="2">LLaVA-QA90</td><td>Accuracy</td><td>5.52</td><td>5.79</td></tr><tr><td>Detailedness</td><td>4.58</td><td>4.77</td></tr><tr><td rowspan="2">Image Captioning</td><td>Accuracy</td><td>6.06</td><td>6.22</td></tr><tr><td>Detailedness</td><td>5.00</td><td>5.24</td></tr></table>

![](images/7a19ca594a4392042b4fc65352aba1638ffa5133394c446c879a30b0a8a6b573.jpg)

<details>
<summary>line</summary>

| Guidance Strength | BLEU / ROUGE |
| ----------------- | ------------ |
| 0.00              | 37.2         |
| 0.125             | 37.3         |
| 0.25              | 38.0         |
| 0.375             | 38.5         |
| 0.50              | 39.8         |
| 0.625             | 39.9         |
| 0.75              | 39.1         |
| 0.875             | 38.4         |
| 1.00              | 37.4         |
</details>

![](images/1da2f50bd7af773d13a64f210afc18790b4065a6c9e2a7480e811633cc6b23bc.jpg)

<details>
<summary>line</summary>

| Guidance Strength | BLEU / ROUGE |
| ----------------- | ------------ |
| 0.00              | 21.5         |
| 0.25              | 22.5         |
| 0.50              | 24.5         |
| 0.75              | 26.0         |
| 1.00              | 26.5         |
</details>

Bleu-1 ROUGE-L

Figure 5. The impact of guidance strength on the output text quality.

# C Further Analysis

# C.1 Limitations of Hallucination Evaluation

While CHAIR and POPE are widely adopted for evaluating object hallucinations in vision-language models, both have inherent limitations. CHAIR depends on a fixed object vocabulary and synonym list, which may miss rare or fine-grained concepts. POPE relies on the quality of segmentation tools to define ground-truth objects, introducing variability across settings.

To address these limitations, we incorporate ALOHa (Automatic Localized Hallucination) (Petryk et al., 2024), a referencebased metric that evaluates hallucination at both the object level $( A L O H a _ { 0 } )$ and the caption level (ALOHa). We follow the standard ALOHa setup using MSCOCO ground-truth captions and enable reference object detection for more precise and generalizable assessment. As shown in Table 25, MARINE consistently outperforms greedy decoding across all models and both ALOHa metrics.

# C.2 Additional Related Work

Several recent works aim to improve grounding or reduce hallucination in vision-language models. BRAVE (Kar et al., 2024) enhances faithfulness by combining diverse visual sources, similar in spirit to MARINE, but introduces additional trainable modules. MARINE achieves comparable performance with a training-free, modular design.

Other approaches focus on evaluation (Hu et al., 2023; Cho et al., 2024; Lin et al., 2024) or feature-level interventions (Yang et al., 2025; Liu et al., 2024a) to steer models away from hallucinations. Liu et al. (2024b) address text inertia, where models generate similar outputs regardless of image content. Wan et al. (2024) introduce sub-image contrastive alignment, and Zhang et al. (2024b) control generation by adjusting visual attention weights.

These methods highlight complementary strategies to MARINE’s structured, object-level guidance for reducing hallucination.

# C.3 Effect of MARINE on logit distribution.

In Figure 7, we illustrate a specific example that shows how MARINE influences the logit distribution of LVLMs during text generation. Specifically, MARINE is observed to selectively target the potential hallucinated tokens, reducing their original probabilities to mitigate the risk of hallucination in the generated text. For instance, in the provided example, the probability of “fork” is significantly lowered with MARINE, which would have originally resulted in a hallucinated object.

![](images/472d6dbebeb3b4a71c5ac86e601ed555713290d1dca8548685da6dd43e25dc0f.jpg)

<details>
<summary>natural_image</summary>

Aerial view of a large outdoor mural featuring a tiger head sculpture, wooden crates, and people on the ground (no visible text or symbols)
</details>

![](images/998c3dcedbf4df612cd4b8911ca50241b04a253d09e9eaae675ab2c6bfe46d9d.jpg)

# GPT4 response as reference

The art installation, featuring an oversized wooden bench and two massive animal art sculptures, encourages interaction among people by providing a visually intriguing and inviting space to sit or gather. People are naturally drawn to the unique and whimsical nature of these installations, sparking curiosity and conversation among those who encounter them. The oversized bench provides seating for multiple individuals, prompting people to sit down, rest, or socialize while admiring the giant painted feline heads and other artworks on display. Additionally, the installations capture the attention of passersby, creating a shared experience and a sense of community among those who visit the area. The artistic environment stimulates discussion and appreciation of the art, facilitating social interactions and connections among those who gather near the sculptures and oversized bench.

Question: How might the art installation encourage interaction among people?

![](images/5a65332112d89aff4dfc73400f46472870e7026bfc0b2feb25de0378fc8c7965.jpg)

![](images/436647f6f84a3d2cf6b4a081965a580ab7fcaa70bb9ca0f66e29e2c2ed14bcc0.jpg)

# LLaVA w/ MARINE with balance guidance

The art installation, which consists of a large, colorful, and intricately painted tiger sculpture, is likely to encourage interaction among people in several ways. Firstly, the vibrant colors and unique design of the tiger sculpture can attract attention and draw people towards it, making them curious about its origin and significance. This curiosity can lead to conversations and discussions among passersby, fostering social interactions and a sense of community.\n\n\nSecondly, the sculpture's size and prominence in the city's public space can encourage people to take a break from their daily routines and engage with the artwork. The presence of the tiger sculpture can create a sense of awe and wonder, encouraging people to pause and appreciate the beauty of the artwork. This can lead to a more relaxed and enjoyable atmosphere, as people take the time to admire the art and enjoy the surrounding environment.\n\n\nLastly, the sculpture's placement in the city's public space can also serve as a focal point for community events, such as art exhibitions, cultural festivals, or even street performances. The tiger sculpture can become a gather [reach the maximum generation length]

![](images/2c764348a22c9f4190ddbf2afc58bf1c0e9bcfc5343ff40fa665caece33ca228.jpg)

# LLaVA w/ MARINE with overly strong guidance

The art installation, which features a large, colorful tiger sculpture and a car parked next to it, can encourage interaction among people in several ways. Firstly, the vibrant colors and unique design of the tiger sculpture can attract attention and draw people towards it, making them curious about its origin and purpose. This curiosity can lead to conversations and discussions among passersby, encouraging social interaction and engagement. Additionally, the presence of the car parked next to the sculpture can provide a focal point for people to gather and take photos, further fostering social interaction and creating a sense of community. The art installation can also serve as a backdrop for events or gatherings, such as art exhibitions, festivals, or even photo shoots, which can further encourage people to interact with each other and engage with the artwork. Overall, the art installation can serve as a catalyst for social interaction and community engagement, promoting a sense of connection and shared experience among people.

Figure 6. This case highlights that overly strong guidance can induce the model to prioritize providing exhaustive visual details from the image, even when such details are irrelevant to the specific instruction (e.g., “a car parked next to it”). In contrast, balanced guidance enables the model to maintain better adherence to the instruction while still utilizing the visual information effectively.

Conversely, standard language elements such as “various”, an adjective describing the overall image context, and “with”, a crucial preposition, maintain their original probabilities. This selective nature of modulation by MARINE ensures coherent and contextually relevant text generation that adheres to the instruction while effectively reducing hallucinations.

# C.4 Discussion on fine-tuning methods.

The examples shown in Figure 8 illustrate that LURE, at times, fails to adhere to the given instructions when correcting LVLM generations. Despite receiving concise image descriptions generated based on instructions for short responses, LURE predominantly overwrites them with excessively long responses that contain information irrelevant to the instruction. Furthermore, LURE fails to adequately address the binary question format of POPE, as LURE fixates on extended descriptions without responding with “yes” or “no”, making its evaluation using POPE impractical. This issue can be prevalent in small-scale fine-tuning methods, where the limited variety of the specifically tailored fine-tuning dataset harms the model’s performance on other tasks. In contrast, the training-free approach of MARINE demonstrates effective mitigation of hallucinations across a variety of question formats.

# C.5 Extended Analysis in Ablation Study

Additional experimental results explore the score threshold of object grounding features, which are examined across LLaVA, and mPLUG-Owl2, with findings presented in Figures 9, and 10.

This variation is achieved by implementing four confidence thresholds (0.5, 0.7, 0.9, and 0.95) in the DETR model predictions (with MARINE-Truth serving as an ideal reference), where higher thresholds correspond to lesser, yet higher-

Table 25. ALOHa hallucination scores (all values are in %). MARINE improves over greedy decoding across models and metrics. 

<table><tr><td rowspan="2">Method</td><td colspan="2">LLaVA</td><td colspan="2">LLaVA-v1.5</td><td colspan="2">mPLUG-Owl2</td></tr><tr><td> $ALOHa \uparrow$ </td><td> $ALOHa_0 \uparrow$ </td><td> $ALOHa \uparrow$ </td><td> $ALOHa_0 \uparrow$ </td><td> $ALOHa \uparrow$ </td><td> $ALOHa_0 \uparrow$ </td></tr><tr><td>Greedy</td><td>40.1</td><td>70.1</td><td>61.9</td><td>83.1</td><td>70.2</td><td>87.0</td></tr><tr><td>MARINE</td><td>48.7</td><td>76.1</td><td>66.7</td><td>85.6</td><td>72.9</td><td>88.2</td></tr></table>

![](images/58bc540343bcf8216c14b5bfa1b7658eab5265de826dea154f717880869173e7.jpg)  
Generate a short caption of the image.

![](images/b91422a6c4f01b7ee784fe8b1f07963e44bd07cc3652076c6dacfc003b20898c.jpg)

# LLaVA

A table is laden with various dishes, including a plate of scambled eggs, a bowl of fruit, and a bowl of yogurt. There is also a spoon and a fork present.

![](images/b94d45662ae55c4c26f9d0bc9f89b34bd5cc53dbf329a8de084297c76199bbde.jpg)

# LLaVA w/ MARINE

A family is enjoying a meal together on a dining table, with various dishes including a bowl of cake and a cup of dessert. A spoon, a cell phone, and a cup are also present on the table.

![](images/5d0bee3102d87a7c3ce720ea44e142b282f3286dab256915b06a107bd3bc6795.jpg)

<details>
<summary>bar</summary>

|        | original probs | control probs | ouptut probs |
| ------ | -------------- | ------------- | ------------ |
| d      | 0.015          | 0.14          | 0.09         |
| cup    | 0.02           | 0.11          | 0.08         |
| plate  | 0.085          | 0.06          | 0.08         |
| bow    | 0.005          | 0.125         | 0.07         |
| fork   | 0.15           | 0.03          | 0.06         |
</details>

(a) An example of image description where the original LLaVA outputs a hallucinated object, “fork”.   
(b) The probability distributions at the token of the hallucinated word in the original, control, and MARINE outputs. MARINE effectively decreases the the probability of “fork”.

![](images/5fea4552c769738fcfcf06151044f19af64573d3ff0bf29eecc1a40a60a741db.jpg)

<details>
<summary>bar</summary>

| Category | Series 1 | Series 2 | Series 3 |
| :--- | :--- | :--- | :--- |
| a | 100 | 95 | 60 |
| various | 100 | 80 | 60 |
| several | 80 | 45 | 65 |
| multiple | 70 | 90 | 55 |
| the | 20 | 30 | 60 |
</details>

![](images/09ba3b38cc3bcad3bb188bdbbab3952b7de755de2339235adcbcf04dfac5a7a5.jpg)

<details>
<summary>bar</summary>

| Category | Series 1 | Series 2 | Series 3 |
| :--- | :--- | :--- | :--- |
| with | 95 | 94 | 93 |
| surrounded | 96 | 70 | 80 |
| featuring | 40 | 75 | 65 |
| consisting | 45 | 40 | 45 |
| which | 45 | 30 | 35 |
</details>

(c) Probabilities of non-hallucinated words remain the same, highlighting MARINE’s ability to preserve normal outputs.

Figure 7. This sample shows how MARINE controls logit distributions to mitigate hallucinations like “fork” while preserving the probabilities of “with”, “various” during generation.   
![](images/dbd32ec462514d6231253dd95f77cb4d4ba7974923bcfde3ed9dc293e1e121ad.jpg)

<details>
<summary>natural_image</summary>

Interior scene of a bedroom with two children: one in a woven basket and another in a wooden bed, no visible text or symbols.
</details>

Input: Is there a book in the image?

LLaVA: Yes, there is a book in the image, and it is placed on the bed next to the doll.

LURE: There are two dolls in a bedroom scene, with one of them being the focus of attention. One of the dolls is sitting in a crib, while the other is lying in a baby carriage in the same room. A book can be seen on the bed behind the doll in the crib.

Woodpecker: No, there is no book in the image. The image shows two beds([0.294, 0.001, 0.999, 0.998];[0.0, 0.37, 0.658, 0.998]), and a doll([0.078, 0.439, 0.553, 0.92]) is placed on each bed. A teddy bear is next to the doll.

MARINE: No, there is no book in the image. The image features a doll in a crib, a doll on a bed, and a teddy bear.

Figure 8. Example responses to an image-question pair. The LURE-corrected output deviates from the original question, offering irrelevant descriptions without directly addressing the query. Woodpecker hallucinates the existence of two beds while there is only one bed in the figure. In contrast, MARINE maintains the original answer’s style and adheres to the user’s instruction while eliminating hallucination.

quality, visual information. Our findings highlight two significant insights. Firstly, an increase in the quality of visual information correlates with a noticeable decrease in hallucinations produced by the LVLMs. A lower threshold, which allows for more visual information but also includes noisier content, could potentially result in an increased occurrence of hallucinations. Furthermore, lower-quality visual information is associated with enhanced Recall. This suggests that LVLMs under guidance, despite the presence of noisy visual inputs, tend to focus more on the visual details (i.e., objects),

resulting in more elaborate descriptions.

![](images/77ab78e1aedfc4683c714e4757bf31689d926218f4fdda82dd716e957b1863c2.jpg)

<details>
<summary>line</summary>

| x    | Series 1 | Series 2 |
| ---- | -------- | -------- |
| 0.5  | 0.45     | 0.35     |
| 0.7  | 0.38     | 0.32     |
| 0.9  | 0.28     | 0.30     |
| 0.95 | 0.25     | 0.29     |
| Truth| 0.15     | 0.22     |
</details>

(a) CHAIRS

![](images/6cce8b508c9c468cd930bed3797b787b79a627f42a5741a6b542540a88b0f580.jpg)

<details>
<summary>line</summary>

| x    | Series 1 | Series 2 |
| ---- | -------- | -------- |
| 0.5  | 0.11     | 0.12     |
| 0.7  | 0.10     | 0.11     |
| 0.9  | 0.09     | 0.08     |
| 0.95 | 0.09     | 0.07     |
| Truth| 0.06     | 0.03     |
</details>

(b) CHAIRI

![](images/b143bf4dc78d50823985d79d53c12d96ef349e618c5e670548263de1f61ecde9.jpg)

<details>
<summary>line</summary>

| Truth | γ=0.5 | γ=1.0 |
|-------|-------|-------|
| 0.5   | 0.64  | 0.82  |
| 0.7   | 0.61  | 0.76  |
| 0.9   | 0.60  | 0.73  |
| 0.95  | 0.58  | 0.68  |
| Truth | 0.69  | 0.88  |
</details>

(c) Recall

Figure 9. LLaVA’s performance on CHAIR according to different score threshold of object grounding features in MARINE. We consider four confidence thresholds (0.5, 0.7, 0.9, and 0.95) for DETR to vary the score threshold.   
![](images/5f84d4e2562653d8dfbb8db467c1604a853d1c2d589e7a8df748c032b392e7e2.jpg)

<details>
<summary>line</summary>

| x    | Series 1 | Series 2 |
| ---- | -------- | -------- |
| 0.5  | 0.14     | 0.065    |
| 0.7  | 0.11     | 0.063    |
| 0.9  | 0.085    | 0.057    |
| 0.95 | 0.05     | 0.05     |
| Truth| 0.04     | 0.04     |
</details>

(a) CHAIRS

![](images/84394d832d6425d111e4e61be24a5a59b4fa51f82895b321ba24316c009ceee7.jpg)

<details>
<summary>line</summary>

| x    | Blue Line | Purple Line |
| ---- | --------- | ----------- |
| 0.5  | 0.06      | 0.035       |
| 0.7  | 0.05      | 0.034       |
| 0.9  | 0.04      | 0.031       |
| 0.95 | 0.025     | 0.028       |
| Truth| 0.015     | 0.02        |
</details>

(b) CHAIRI

![](images/9e7f0d56d4ffbda5aed192cb6c77965e1c7a51d475d0f663a78c045c51b40d90.jpg)

<details>
<summary>line</summary>

| x     | γ=0.5  | γ=1.0  |
|-------|--------|--------|
| 0.5   | 0.43   | 0.49   |
| 0.7   | 0.425  | 0.51   |
| 0.9   | 0.425  | 0.505  |
| 0.95  | 0.42   | 0.50   |
| Truth | 0.44   |        |
</details>

(c) Recall   
Figure 10. mPLUG-Owl2’s performance on CHAIR according to different score threshold of object grounding features in MARINE. We consider four confidence thresholds (0.5, 0.7, 0.9, and 0.95) for DETR to vary the score threshold, with MARINE-Truth serving as an ideal reference.

# C.6 More Case Studies

In Figures 11, 12 and 13, we present examples of the outputs from LURE (Zhou et al., 2023), Woodpecker (Yin et al., 2023) and MARINE on different tasks further validate our arguments in the paper.

![](images/8690cf6aa7d4b7c11641b02823cd5758f9b422356ea4e223fa48049914ade94d.jpg)

Query: What is the position of the skateboard in the image?

LLaVA: The skateboard is on the ground, with the rider standing on it and holding onto the trucks.

LLaVA w/ MARINE: The skateboard is on the ground, with the person standing on it.

Query: Generate a short caption of the image.

LLaVA-v1.5: A family of four, including a woman and three children, are sitting at a table with a plate of food in front of them.

LLaVA-v1.5 w/ MARINE: A family of four, including two adults and two children, are sitting around a dining table with plates of food in front of them.

![](images/4dac7ece9f411e0db16cd8720e3a9e9099b01484efd03b62e36032ee2b5225e7.jpg)

<details>
<summary>natural_image</summary>

Family photo of a family gathered around a table with food and drinks (no visible text or symbols)
</details>

Figure 11. Hallucination mitigation examples by our proposed MARINE across multiple tasks: LLaVA-QA90 and image captioning. Hallucinated objects generated by the LVLM are highlighted in red.

Query: Generate a short caption of the image.

LLaVA: A black and white dog is sitting on a beach, surrounded by rocks and sand. The dog is looking at its owner who is standing behind it, as if waiting for a command or attention. The scene captures the dog‘s loyalty and affection towards its owner. (Note: 'bird' omitted)

LURE: The scene features a black and white dog sitting on a beach, surrounded by pebbles. The dog is looking at its owner who is standing behind it, as if waiting for attention. The background reveals a vast ocean, and other birds can be seen scattered throughout the area. The presence of these birds gives the impression that the dog is not alone, and they are enjoying the time together on the beach. Overall, the image captures a charming scene on the sandy beach.

Woodpecker: A black and white dog([0.243, 0.662, 0.551, 0.894]) is sitting on the beach([0.002, 0.029, 0.998, 0.999]), surrounded by sand. The dog is looking at its owner([0.712, 0.57, 0.92, 0.684]) who is standing behind it.

LLaVA w/ MARINE:

In the image, a black and white dog is standing on a beach, looking at a bird.

![](images/57c8f2f8b2511d265ec5151084f8915e8a1422f7fcb48a5fc0eb490ac39df4a5.jpg)

<details>
<summary>natural_image</summary>

A dog sitting on a pebble beach with another bird flying nearby (no text or symbols visible)
</details>

Figure 12. A comparison of responses from baseline models and our MARINE in an image description task. It illustrates MARINE’s superior ability to reduce hallucinations, in contrast to LURE and Woodpecker, which fail to effectively address hallucinations and sometimes even increase hallucinated content. This example highlights the strengths of our correct-during-generation framework over post-correction approaches, showcasing its efficiency, preservation of original style, and enhanced adherence to instructions.

Query: Generate a short caption of the image.

MiniGPT-v2: no

Woodpecker: No, there is no toothbrush in the image.

MiniGPT-v2 w/ MARINE: yes

![](images/380e6ac9fea189a37cb8272d6e696f25b3c7b3ae4a0efa778e66ee349e7e6081.jpg)

<details>
<summary>natural_image</summary>

Interior view of a bathroom with beige tiled walls, a white toilet, and a large showerhead near a window (no visible text or symbols)
</details>

Figure 13. A comparison of responses from baseline models and our MARINE in POPE “yes-or-no” task. MiniGPT-v2 provides a concise response without referencing any objects. Under these circumstances, Woodpecker is unable to perform corrections via GPT-3.5 due to missing visual details. MARINE, however, successfully corrects the response while retaining MiniGPT-v2’s style.