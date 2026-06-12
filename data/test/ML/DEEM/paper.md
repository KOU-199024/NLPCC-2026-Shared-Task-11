# DEEM : DIFFUSION MODELS SERVE AS THE EYES OF LARGE LANGUAGE MODELS FOR IMAGE PERCEPTION

Run Luo1,2∗ Yunshui Li1.2∗ Longze Chen1,2∗ Wanwei He1,2 Ting-En Lin3 Ziqiang Liu1,2 Lei Zhang1,2 Zikai Song4 Hamid Alinejad-Rokny5 Xiaobo Xia6,7 Tongliang Liu8 Binyuan Hui9† Min Yang1†

1Shenzhen Key Laboratory for High Performance Data Mining, SIAT, CAS   
2University of Chinese Academy of Sciences 3Tsinghua University   
4Huazhong University of Science and Technology 5 University of New South Wales   
6 School of Computing, National University of Singapore   
7 MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, University of Science and Technology of China 8 The University of Sydney 9Alibaba Group

# ABSTRACT

The development of large language models (LLMs) has significantly advanced the emergence of large multimodal models (LMMs). While LMMs have achieved tremendous success by promoting the synergy between multimodal comprehension and creation, they often face challenges when confronted with out-ofdistribution data, such as which can hardly distinguish orientation, quantity, color, structure, etc. This is primarily due to their reliance on image encoders trained to encode images into task-relevant features, which may lead them to disregard irrelevant details. Delving into the modeling capabilities of diffusion models for images naturally prompts the question: Can diffusion models serve as the eyes of large language models for image perception? In this paper, we propose DEEM , a simple but effective approach that utilizes the generative feedback of diffusion models to align the semantic distributions of the image encoder. This addresses the drawbacks of previous methods that solely relied on image encoders like CLIP-ViT, thereby enhancing the model’s resilience against out-of-distribution samples and reducing visual hallucinations. Importantly, this is achieved without requiring additional training modules and with fewer training parameters. We extensively evaluated DEEM on both our newly constructed RobustVQA benchmark and other well-known benchmarks, POPE and MMVP, for visual hallucination and perception. In particular, DEEM improves LMM’s visual perception performance to a large extent (e.g., 4% ↑ on RobustVQA, 6.5% ↑ on MMVP, and 12.8 % ↑ on POPE ). Compared to the state-of-the-art interleaved content generation models, DEEM exhibits enhanced robustness and a superior capacity to alleviate model hallucinations while utilizing fewer trainable parameters, less pre-training data (10%), and a smaller base model size. Extensive experiments demonstrate that DEEM enhances the performance of LMMs on various downstream tasks without inferior performance in the long term, including visual question answering, image captioning, and text-conditioned image synthesis. The code and benchmark are available at https://github.com/RainBowLuoCS/DEEM

# 1 INTRODUCTION

With the success of large language models (LLMs), large multimodal models (LMMs) built on LLMs have garnered significant attention. Researchers (Liu et al., 2024a; Zhu et al., 2023; Dai et al., 2024; Alayrac et al., 2022; Chen et al., 2023) have attempted to build a bridge between large language models and image encoders through simple mapping modules, and have already made significant progress in multimodal understanding tasks such as visual question answering. Subsequent studies (Yu et al., 2023; Sun et al., 2023b; Dong et al., 2023; Tian et al., 2024) utilize extra advanced diffusion models (DMs) (Rombach et al., 2022) for image generation and train the LMMs on interleaved text-image data in an end-to-end manner. This unified paradigm of multimodal understanding and creation brings various isolated multimodal tasks together, greatly boosting model capabilities and expanding application scenarios.

![](images/ad568b612a29dc40009f9e8ef12b674d3f341d89c43c6c0e0f0b28213c0ab43a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Previous Method"] --> B["Diffusion Model"]
    B --> C{DeEM!}
    C --> D["Sealion"]
    C --> E["Right"]
    C --> F["Two"]
    D --> G["LMM For Text Generation Only"]
    E --> H["LMM For Text And Image Generation"]
    F --> I["LMM For Text Generation Only"]
    J["Visual Robustness"] --> C
    K["Visual Perception"] --> C
    L["Visual Hallucination"] --> C
    M["RobustVQA"] --> N["61.8"]
    O["MMVP"] --> P["25.1"]
    Q["POPE"] --> R["54.5"]
    S["RobustVQA"] --> T["62.9"]
    U["MMVP"] --> V["28.7"]
    W["POPE"] --> X["51.5"]
```
</details>

Figure 1: Illustration of our DEEM . When encountering natural adversarial examples or out-ofdistribution data, DEEM uses the diffusion model to check if the semantic features of the image encoder match the input images. This approach allows DEEM to serve as the ”eyes” of the large language model, proactively identifying and correcting misinterpreted semantic information during training, thereby avoiding the loss of important visual details. This enhances the robustness, hallucination recognition, and foundational visual perception capabilities of LMMs. In contrast, other models rely too heavily on erroneous inputs from the image encoder, making it difficult for them to handle challenges posed by such data.

However, these models commonly rely on encoder architectures like CLIP-ViT (Radford et al., 2021), which suffers from certain perceptual understanding limitations due to the contrastive learning paradigm and the noisy image-text pairs used in training, to encode input images. Additionally, these image encoders are typically trained to encode images into features relevant to downstream tasks, thereby disregarding irrelevant details. Consequently, as shown in Fig. 1, when faced with images outside the training scope, they often capture biased semantic features, resulting in erroneous visual information being perceived by subsequent language models. This accumulation of inaccuracies renders the multimodal model unable to comprehend multimodal context effectively. For this reason, this makes it difficult for previous methods to discern subtle details, thereby hindering their ability to handle tasks related to basic visual perception, visual hallucinations, and visual robustness that are very simple for humans.

On the contrary, the goal of diffusion models (Ho et al., 2020a) is to learn a diffusion process that characterizes a probability distribution for a given dataset, without direct training on the downstream task objective. This enables it to capture finer details of images for better handling of out-ofdistribution data. However, there have been few efforts to integrate the capabilities of the diffusion model into the image perception of large multimodal models.

In this paper, we propose DEEM, a simple but effective approach to leverage the generative feedback of diffusion models for aligning the semantic distributions of image encoders in an elegant self-supervised manner. Building upon this, we introduce an end-to-end interleaved image-text generative modeling approach, where diffusion models serve as additional eyes of large language models for image perception. This addresses the limitations of previous methods that solely relied on image encoders such as CLIP-ViT (Radford et al., 2021), enhancing the model’s robustness against out-of-distribution samples and reducing hallucination perception in multimodal scenarios, without the need for additional training modules and with fewer training parameters. To the best of our knowledge, we are the first to apply diffusion models to large multimodal models for image perception.

Specifically, DEEM takes interleaved image-text pairs as input to the model. It starts by encoding images and text using corresponding visual and text encoders, resulting in image tokens and text tokens. These tokens are then organized according to their original layout and inputted into a large language model to generate corresponding hidden state outputs. The model employs autoregressive modeling for the hidden state outputs of text and utilizes the output hidden states of images, along with the image tokens encoded by the image encoder, as diffusion conditions. These conditions are then fed into a diffusion model for image reconstruction. Through end-to-end training, the model not only acquires the capacity to generate text and images but also employs semantic consistency regularization on the semantic information produced by the image encoder during image reconstruction. This compels the image encoder to incorporate more details into the semantic representation of the image, thereby mitigating the issue of semantic bias in image encoding.

DEEM is trained on a mixture corpora of image-text pairs and interleaved image-text sequences data without extra in-house data following previous solution (Li et al., 2022; 2023a; Dong et al., 2023; Tian et al., 2024). To assess the robustness recognition capability of LMMs, we constructed a new robustness benchmark, RobustVQA, based on existing datasets containing natural adversarial samples and out-of-distribution data. RobustVQA is divided into three parts: RobustVQA-A, RobustVQA-R, and RobustVQA-V, based on different data sources, aiming to provide better insights into the performance of LMMs in real-world scenarios. We conducted extensive evaluations of DEEM on RobustVQA and two widely recognized benchmarks, POPE and MMVP, for visual hallucination and perception respectively. Experimental results indicate that our method exhibits enhanced robustness, a superior capacity to alleviate model hallucinations and better visual perception ability in comparison to the state-of-the-art interleaved image-text modeling model MM-Interleaved (Tian et al., 2024), using a smaller-scale image encoder (CLIP-ConvNext-B (Liu et al., 2022) vs. CLIP-ViT-L (Radford et al., 2021)), a smaller-scale language model (Vicuna 7B vs. Vicuna 13B (Zheng et al., 2024)), and less pre-training data (without Laion-coco (Andreas et al., 2022) & Laion-en (Schuhmann et al., 2022)). DEEM outperforms MM-Interleaved 9.4% on RobustVQA, 17.8% on POPE and 9.1% on MMVP. Moreover, with further enhancement via supervised finetuning, DEEM achieves competitive results on various multimodal tasks, including visual questionanswering, region-level image captioning, and text-to-image generation.

Before delving into details, we summarize our contributions as follows.

• Robustness Benchmark. We design a new robustness benchmark RobustVQA for LMMs based on publicly available ImageNet-A (Hendrycks et al., 2021b), ImageNet-R (Hendrycks et al., 2021a), and ImageNet-V2 (Recht et al., 2019) datasets, which can be utilized to effectively assess the visual robustness capabilities of the multimodal models.   
• Effective Method. We are the first to introduce the diffusion model into the image perception of large language models, to correct potential semantic bias in the image encoder and alleviate the excessive compression of visual details. This approach enhances the model’s robustness and hallucination mitigation capabilities without the need for additional modules or trainable parameters.   
• DEEM Model. Based on the proposed method, we train a multimodal model with end-toend interleaved text-image modeling capabilities. After supervised fine-tuning, DEEM can perform various multimodal tasks in a unified manner, such as visual question answering, text-to-image generation, and region-level image captioning.   
• Comprehensive Experiments. We provide abundant qualitative and quantitative comprehensive experimental results to demonstrate the effectiveness and efficiency of the proposed method.

# 2 METHOD

In this section, we first present our DEEM , starting with an introduction to the overall architecture in Section 2.1, followed by a description of the pipeline in Section 2.2. Finally, we provide details on the training and inference process in Section 2.3.

# 2.1 ARCHITECTURE

In this subsection, we present the multi-modal architecture for processing interleaved image-text data. To excel in both comprehension and creation tasks of text and images, a multi-modal model consists of the following three key components.

VFM-based Image Encoder $\mathcal { E } _ { V }$ which encodes each image $x ^ { V } \in \mathbb { R } ^ { H \times W \times 3 }$ into an image embedding $e ^ { V } \in \mathbb { R } ^ { N \times C }$ , where C is the channel dimension and N is the number of visual tokens in image embedding. LLM-based Multi-modal Decoder $\mathcal { D } _ { \mathbf { L L M } }$ that extracts context features from the interleaved image-text token sequences. Its input sequence $E \in \mathbb { R } ^ { K \times C }$ is a concatenation of embeddings $e _ { n } ^ { V } \in \mathbb { R } ^ { N \times C }$ $( e _ { 1 } , e _ { 2 } , \dots )$ n . K is the total number of input tokens. DM-based Image Decoder DDM that generatesnditioned on image-text sequences context feature. , where $e _ { n }$ is either a word embedding $e _ { n } ^ { L } \in \mathbb { R } ^ { 1 \times C }$ or an image embedding

![](images/ca45b37a351bcce0b199bba48f3833cecb845a63d4ecb12fed5c9cb21a0afbec.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Causal Large Multimodal Model (LMM)
        direction TB
        A["Image Embedding"] --> B["Diffusion Model"]
        B --> C["Image Encoder"]
        C --> D["Image Embedding"]
        D --> E["Diffusion Model"]
        E --> F["Image Encoder"]
        F --> G["Word Embedding"]
        G --> H["Image Restoration"]
    end

    I1["<I like this interesting vegetable clock, made up of various vegetables."] --> A
    I2["Ilike the vegetables are gathered together, it will look more like a complete clock."] --> B
    I3["Imagine when the vegetables are gathered together, it will look more like a complete clock."] --> E
    I4["Now I like it even more. What an interesting clock! </s>"] --> G
    I5["<SOI>"] --> G
    style I1 fill:#f9f,stroke:#333
    style I2 fill:#f9f,stroke:#333
    style I3 fill:#f9f,stroke:#333
    style I4 fill:#f9f,stroke:#333
    style I5 fill:#f9f,stroke:#333
    style I6 fill:#f9f,stroke:#333
    style I7 fill:#f9f,stroke:#333
    style I8 fill:#f9f,stroke:#333
    style I9 fill:#f9f,stroke:#333
    style I10 fill:#ccf,stroke:#333
    style I11 fill:#ccf,stroke:#333
    style I12 fill:#ccf,stroke:#333
    style I13 fill:#ccf,stroke:#333
    style I14 fill:#ccf,stroke:#333
    style I15 fill:#ccf,stroke:#333
    style I16 fill:#ccf,stroke:#333
    style I17 fill:#ccf,stroke:#333
    style I18 fill:#ccf,stroke:#333
    style I19 fill:#ccf,stroke:#333
    style I20 fill:#ccf,stroke:#333
    style I21 fill:#ccf,stroke:#333
    style I22 fill:#ccf,stroke:#333
    style I23 fill:#ccf,stroke:#333
    style I24 fill:#ccf,stroke:#333
    style I25 fill:#ccf,stroke:#333
    style I26 fill:#ccf,stroke:#333
    style I27 fill:#ccf,stroke:#333
    style I28 fill:#ccf,stroke:#333
    style I29 fill:#ccf,stroke:#333
    style I30 fill:#ccf,stroke:#333
    style I31 fill:#ccf,stroke:#333
    style I32 fill:#ccf,stroke:#333
    style I33 fill:#ccf,stroke:#333
    style I34 fill:#ccf,stroke:#333
    style I35 fill:#ccf,stroke:#333
    style I36 fill:#ccf,stroke:#333
    style I37 fill:#ccf,stroke:#333
    style I38 fill:#ccf,stroke:#333
    style I39 fill:#ccf,stroke:#333
    style I40 fill:#ccf,stroke:#333
    style I41 fill:#ccf,stroke:#333
    style I42 fill:#ccf,stroke:#333
    style I43 fill:#ccf,stroke:#333
    style I44 fill:#ccf,stroke:#333
    style I45 fill:#ccf,stroke:#333
    style I46 fill:#ccf,stroke:#333
    style I47 fill:#ccf,stroke:#333
    style I48 fill:#ccf,stroke:#333
    style I49 fill:#ccf,stroke:#333
    style I50 fill:#ccf,stroke:#333
    style I51 fill:#ccf,stroke:#333
    style I52 fill:#ccf,stroke:#333
    style I53 fill:#ccf,stroke:#333
    style I54 fill:#ccf,stroke:#333
    style I55 fill:#ccf,stroke:#333
    style I56 fill:#ccf,stroke:#333
    style I57 fill:#ccf,stroke:#333
    style I58 fill:#ccf,stroke:#333
    style I59 fill:#ccf,stroke:#333
    style I60 fill:#ccf,stroke:#333
```
</details>

Figure 2: Overview of our DEEM framework. Interleaved documents serve as input, decoded to produce outputs. Both text and images are encoded into sequential, discrete token embeddings for the LMM input. Here, we replace the ${ \bf \mathrm { < I M G > } }$ token embedding in the text with the image embedding before inputting it into the LLM. The text is predicted in an autoregressive manner and the images are synthesized by the DM-based image decoder conditioned on holistic historical semantics captured by LMM. Besides, the image token embeddings are fed into DM-based image decoder for consistent image restoration. The start of image token ${ < } \mathrm { S O I { > } }$ is used to determine the starting position of the image, facilitating the natural autoregressive generation of interleaved textimage layouts. Note that our core architecture is presented without the connectors between modules for simplicity.

To provide the conditional inputs for $\mathcal { D } _ { \mathrm { D M } }$ and reduce the number of visual tokens in image embedding $e ^ { V }$ , two different Perceiver Resampler (Alayrac et al., 2022) are employed to map the output features from multi-modal decoder $\mathcal { D } _ { \mathrm { L L M } }$ and image encoder ${ \mathcal { E } } _ { V }$ to a fixed number of conditional tokens, respectively. Additionally, we utilize an extra mask-aware visual extractor ${ \mathcal { E } } _ { \mathrm { M } }$ for extracting region visual information from image embedding $e ^ { V }$ via simple mask-aware operation $\mathcal { E } _ { \mathrm { M } } ( e ^ { V } , \breve { M } ^ { V } )$ , where $\mathcal { M } ^ { V }$ is the corresponding binary mask of image $x ^ { V }$ .

# 2.2 PIPELINE

As shown in Fig. 2, given an interleaved image-text sequence $X = \{ x _ { 1 } , x _ { 2 } , x _ { 3 } , . . . \}$ , where each element $x _ { n }$ is either a text token (denoted as $x _ { n } ^ { L } )$ or a whole image (denoted as $x _ { n } ^ { V } )$ . Text and images are arranged in the order in which they appear in the original content. To build an end-toend generative model for interleaved image-text data, a common practice is to first extract embedding for each text token and each image and then feed them into LLMs, $i . e . , e _ { n } ^ { L } = \mathcal { E } _ { L } ( x _ { n } ^ { L } )$ and $e _ { n } ^ { V } \mathbf { \Psi } = \mathcal { E } _ { \mathrm { M } } ( \mathcal { E } _ { V } ( x _ { n } ^ { V } ) , \mathcal { M } _ { n } ^ { V } )$ , where erceiver $\mathcal { E } _ { L }$ denotes word embedding in LLM. sampler (Alayrac et al., 2022) to m ${ \mathcal { E } } _ { V }$ is typically an imageeach image to a fixed number of visual tokens. As shown in Fig. 3, we introduce a mask-aware visual extractor ${ \mathcal { E } } _ { \mathrm { M } }$ for extracting region visual information from image embedding $e _ { n } ^ { V }$ via simple mask-aware operation $\mathcal { E } _ { \mathrm { M } } ( e _ { n } ^ { V } , \tilde { { \mathcal { M } } } _ { n } ^ { V } )$ , where $\mathcal { M } _ { n } ^ { V }$ is the corresponding binary mask of image $x _ { n } ^ { V }$ and the default value is 1. Then, the interleaved generative modeling is trained to maximize the log-likelihood:

$$
\log p (X) = \sum_ {n} \log p (x _ {n} | e _ {<   n}) = \sum_ {n \in \mathcal {I} _ {L}} \underbrace {\log p (x _ {n} ^ {L} | e _ {<   n})} _ {\text { text   prediction }} + \sum_ {n \in \mathcal {I} _ {V}} \underbrace {\log p (x _ {n} ^ {V} | e _ {<   n})} _ {\text { image   prediction }}, \tag {1}
$$

where $\mathcal { T } _ { L }$ and $\mathcal { T } _ { V }$ represent the index sets for text tokens and images, respectively. That $< n$ in the subscript represents the abbreviation of $\{ 1 , 2 , \ldots , n - 1 \}$ . The following paragraphs provide explanations of Eq. (1).

Text Generation with Multi-modal Condition. $\log p ( x _ { n } ^ { L } | e _ { < n } )$ is similar to traditional causal language modeling, except that the condition also includes previous images. Recent works (Alayrac et al., 2022; Li et al., 2023a; Liu et al., 2024a) have demonstrated the effectiveness of using LLMs for processing additional visual inputs. The loss function for text generation is

$$
\mathcal {L} _ {\mathrm{NTP}} (x _ {n} ^ {L} | e _ {<   n}) = - \log p (x _ {n} ^ {L} | \mathcal {D} _ {\mathrm{LLM}} (e _ {<   n})) \tag {2}
$$

where $\mathcal { D } _ { \mathrm { L L M } }$ denotes the LLM network.

Image Generation with Multi-modal Condition. Maximizing log $p ( x _ { n } ^ { V } | e _ { < n } )$ aligns with the diffusion denoising process, which recently achieved widespread success in image generation. Maximizing the log-likelihood is derived as minimizing the diffusion modeling loss as

$$
\mathcal {L} _ {\mathrm{NIP}} (x _ {n} ^ {V} | e _ {<   n}) = \mathbb {E} _ {\epsilon , t} \left| \left| \epsilon - \mathcal {D} _ {\mathrm{DM}} \big (x _ {n, t} ^ {V}, t, \mathcal {D} _ {\mathrm{LLM}} (e _ {<   n}) \big) \right| \right| ^ {2}, \tag {3}
$$

where $\mathcal { D } _ { D M }$ is the diffusion model for denoising process. That $x _ { n , t } ^ { V }$ is the noisy version of the original image at the denoising step t, and the denoising network $\mathcal { D } _ { D M }$ is trained to predict the noise ϵ.

Consistency Semantic Regularization. In addition to the above text and image generation loss functions, we propose a new consistency semantic constraint term. This term reuses the diffusion model to perform generative checks on the image semantic information extracted by the image encoder, ultimately correcting erroneous knowledge in the pre-trained image encoder. This significantly enhances the out-of-distribution generalization and reduces visual hallucinations in the multi-modal model. The new log-likelihood function can be written as

$$
\log p ^ {\star} (X) = \sum_ {n \in \mathcal {I} _ {L}} \underbrace {\log p \left(x _ {n} ^ {L} \mid e _ {<   n}\right)} _ {\text { text   prediction }} + \sum_ {n \in \mathcal {I} _ {V}} \underbrace {\log p \left(x _ {n} ^ {V} \mid e _ {<   n}\right)} _ {\text { image   prediction }} + \sum_ {n \in \mathcal {I} _ {V}} \underbrace {\log p \left(x _ {n} ^ {V} \mid e _ {n}\right)} _ {\text { image   restoration }}. \tag {4}
$$

Similarly, the corresponding log-likelihood function log $p ( x _ { n } ^ { V } | e _ { n } )$ can be equivalently written as the following loss function used in training:

$$
\mathcal {L} _ {\mathrm{CSR}} (x _ {n} ^ {V} | e _ {n}) = \mathbb {E} _ {\epsilon , t} \left| \left| \epsilon - \mathcal {D} _ {\mathrm{DM}} (x _ {n, t} ^ {V}, t, e _ {n}) \right| \right| ^ {2}. \tag {5}
$$

Note that the new end-to-end modeling framework brings significant improvements to the generalization performance of the model without altering the original modeling flexibility or introducing additional modules.

# 2.3 TRAINING AND INFERENCE

We employ a three-stage training process, consisting of image-text alignment pre-training, imagetext instruction fine-tuning, and mask-text instruction fine-tuning. Image-text alignment pre-training and image-text instruction fine-tuning are designed to validate the effectiveness and efficiency of semantic consistency regularization in enhancing the visual perception capabilities of LMMs. Mask-text instruction fine-tuning is used to verify whether the model trained with semantic consistency regularization negatively impacts the performance of finetuning on downstream tasks in the long term. The image-text alignment pre-training objective is defined as the sum of the next-text prediction loss in Eq. (2), next-image prediction loss in Eq. (3) and consistency semantic regularization loss in Eq. (5) as ${ \mathcal L } _ { S _ { 1 } } = { \mathcal L } _ { \mathrm { N T P } } + \lambda { \mathcal L } _ { \mathrm { N I P } } + \lambda { \mathcal L } _ { \mathrm { C S R } } ,$ , where λ is a coefficient used to determine the relative loss weight between the image and text decoding branches. In order to enable the DEEM to perform general multimodal comprehension and creative tasks following human instructions, we use $\mathcal { L } _ { S _ { 2 } } = \mathcal { L } _ { \mathrm { N T P } } + \lambda \mathcal { L } _ { \mathrm { C S R } }$ to conduct image-text instruction fine-tuning. To further enhance the model’s fine-grained region awareness, we conducted region-level mask-text instruction fine-tuning. Since there is no need to perform text-to-image tasks, we removed the next-image prediction loss and the training objective in mask-text instruction fine-tuning can be defined as $\mathcal { L } _ { S _ { 3 } } = \mathcal { L } _ { \mathrm { N T P } }$ . The whole framework can be optimized end-to-end during the three stages. During inference, the images and texts are generated in an auto-regressive manner. Text tokens are sampled from the distribution predicted by the multi-modal LLM. When the generated token is ${ < S \circ I > }$ , the diffusion model is called for generating the next image.

![](images/7087153945aaa607846af8adf5e33fd246136a7b90d513add926c7736807157e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image Encoder"] --> B["Image"]
    B --> C["LMM"]
    C --> D["A baby elephant"]
    E["Input Image"] --> F["Image"]
    F --> G["LMM"]
    G --> H["Provide a short description for this <ref>region</ref>"]
```
</details>

Figure 3: Pipeline of Mask-Aware Extractor. The mask-aware extractor can be used to extract region-level visual features based on the mask-aware operation. A simple dot product is applied between the mask and the image embedding before being fed into the LLM.

# 3 EXPERIMENT

# 3.1 IMPLEMENTATION DETAILS

In this subsection, we first introduce the network of DEEM and then showcase the three-stage training recipes. More details of datasets and hyper-parameters can be found in Table 11.

Network. Similar to previous work, We leverage Vicuna7B (Zheng et al., 2024) and Stable Diffusion v2.1 (Rombach et al., 2022) as the large language model, and image decoder, respectively. However, unlike their use of a 427M parameter CLIP-ViT-L as the image encoder, we use a smaller 122M parameter CLIP-ConvNeXt-B(Liu et al., 2022). For the multi-modal LLM, two different Perceiver Resamplers (Alayrac et al., 2022) are used to connect diffusion model with image encoder and large language model respectively.

Image-Text Alignment Pre-training. Our model is pre-trained on a mixture of image-text pairs and interleaved image-text sequences, including MMC4-Core (Zhu et al., 2024), LAION-400M (Schuhmann et al., 2021), SBU (Ordonez et al., 2011), and CC-12M (Changpinyo et al., 2021). For LAION-400M (Schuhmann et al., 2021), SBU (Ordonez et al., 2011), and CC-12M (Changpinyo et al., 2021), instead of utilizing the original annotations, we use the version filtered by the pretrained BLIP-2 model (Li et al., 2023a). For simplicity, we refer to it as BLIP-LCS hereafter. ”LCS” abbreviates the LAION, CC, and SBU datasets. The sampling probability of MMC4 is twice that of BLIP-LCS. The images are inserted before or after the corresponding text sentence with equal probability. To optimize training efficiency and data utility, multiple image-text pairs or interleaved image-text sequences are concatenated into extended sequences with the maximum context length.

Image-Text Instruction Fine-tuning. To enable DEEM to perform general multimodal comprehension tasks following human instructions, we utilize publicly available datasets for image-text instruction fine-tuning, including LLaVA-665K (Liu et al., 2024a), COCO Caption (Chen et al., 2015), VQAv2 (Goyal et al., 2017), TextCaps (Sidorov et al., 2020), OCR-VQA (Mishra et al., 2019), GQA (Hudson & Manning, 2019), OK-VQA (Marino et al., 2019), TextVQA (Singh et al., 2019), and AOK-VQA (Schwenk et al., 2022).

Mask-Text Instruction Fine-tuning. At this stage, we use a simple mask-aware visual extractor to capture pixel-level region features and then align mask-based region features with language embeddings. We collect short text and pixel-level mask pairs from the publicly available object-level datasets (COCO (Chen et al., 2015), RefCOCO (Kazemzadeh et al., 2014), RefCOCO+ (Mao et al., 2016), Ref-COCOg (Mao et al., 2016)), part-level datasets (Pascal Part (Chen et al., 2014), Part Imagenet (He et al., 2022)), and multiple region datasets(VCR (Zellers et al., 2019), Visual Genome (Krishna et al., 2017)). Then we conduct mask-text instruction fine-tuning on the mixture of the above text-mask pairs data, enabling DEEM to complete region-level understanding tasks, such as regionlevel image captioning.

![](images/a58496993798542f40d4d1e7cd9918f41577f09c33b149e2bdd03de29105a1b4.jpg)

<details>
<summary>natural_image</summary>

Three images showing bird and grass scenes: ImageNet-R, ImageNet-A, and ImageNet-V2 (no text or symbols)
</details>

Figure 4: Examples from ImageNet-R, ImageNet-A, and ImageNet-V2. These examples share similar backgrounds, rare materials, and unusual textures. They serve as natural adversarial examples and out-of-distribution data, which can be used to test the robustness of models.

# 3.2 EXPERIMENTAL RESULTS

In this study, we evaluate our DEEM model by comparing it with current state-of-the-art (SOTA) models on various tasks including visual robustness , hallucination diagnosis, basic visual perception and image-level visual question answering. Please refer to Appendix C for more experimental results about mask-level visual question answering and text-to-image generation. All metrics and data splits are listed in Table 11 in Appendix E.

Visual Perception Diagnose. We explore the impact of diffusion feedback on the visual perception capabilities of LMMs from three dimensions: visual robustness, visual hallucinations, and basic visual perception. To rigorously assess visual robustness of our model, we design a benchmark called RobustVQA for robustness evaluation based on online datasets, including ImageNet-A (Hendrycks et al., 2021b), ImageNet-R (Hendrycks et al., 2021a) and ImageNet-V2 (Recht et al., 2019). As shown in Fig. 4, these challenging natural adversarial examples and out-of-distribution samples in the original ImageNet dataset can be used to evaluate the neural network robustness of our model. Similar to the POPE and MMVP dataset, we first choose the challenging sample from ImageNet-A, ImageNet-R, and ImageNet-V2 dataset and then convert the them into a VQA format that the multimodal model can evaluate simply and accurately. More details about the new benchmark RobustVQA design can be found in Appendix E.1. For a comprehensive visual robustness and hallucination evaluation, we evaluate our model against other open-source state-of-the-art (SOTA) LMMs for text and image generation, including SEED (Ge et al., 2023), SEED-X (Ge et al., 2024), MM-Interleaved (Tian et al., 2024), and DreamLLM (Dong et al., 2023), on the RobustVQA, POPE (Li et al., 2023c) and MMVP (Tong et al., 2024) dataset with accuracy metric. The results, presented in Table 1, demonstrate that our DEEM model not only exhibits competitive performance compared with existing fine-tuned SOTA models on POPE and MMVP after fine-tuning, but also achieves the best results among visual robustness benchmark only after pre-training. Notably, compared to the larger-scale concurrent SOTA model for interleaved text-image modeling, MM-Interleaved (Tian et al., 2024), our model achieves better results with a smaller scale. DEEM outperforms MM-Interleaved 9.4% on RobustVQA, 17.8% on POPE and 9.1% on MMVP. To ensure a fair comparison and prove the effectiveness of our method, we also train an MM-Interleaved model with the same experimental setting as a baseline. Compared to this baseline, Our method achieves an 4% average gain on RobustVQA, 12.8% average gain on POPE and 6.5% average gain on MMVP, respectively. The experimental results demonstrate the effectiveness of our method for better LMMs’ visual perception capability.

Image-Level Visual Question Answering and Captioning. In order to assess multimodal vision and language capabilities of DEEM , we conduct evaluation against current SOTA LMMs including LLaVA-1.5 (Liu et al., 2023), Qwen-VL (Bai et al., 2023), DreamLLM (Dong et al., 2023) and MM-Interleaved (Tian et al., 2024) across several tasks, including image captioning on COCO (Chen et al., 2015), Image2Paragraph (Krause et al., 2017), visual question answering on VQAv2 (Goyal et al., 2017), OKVQA (Marino et al., 2019), GQA (Hudson & Manning, 2019), VizWiz (Gurari et al., 2018), and VisDial (Das et al., 2017). As demonstrated in Table 2, DEEM exhibits superior or comparable performance relative to SOTA models. In comparison with models for text generation only, our approach consistently achieves competitive performance across various dataset splits. Against models for both image and text generation, DEEM demonstrates enhanced performance in nine dataset splits. Compared to the current state-of-the-art model DreamLLM, DEEM outperforms DreamLLM in six out of the seven shared evaluation dataset splits. It is noteworthy that DEEM is trained with a significantly smaller image encoder CLIP-ConvNeXt-B (Liu et al., 2022), comprising only 122M parameters, in stark contrast to baselines such as DreamLLM (Dong et al., 2023), which utilize larger 427M CLIP-ViT-L (Radford et al., 2021). These results indicate that our method can enhance the model’s robustness performance without compromising the multimodal vision and language capabilities of our model.

Table 1: Zero-shot visual robustness, hallucination and perception evaluation of RobustVQA-A: RVQA-A, RobustVQA-R: RVQA-R, RobustVQA-V: RVQA-V, POPE-Random: POPE-R (Li et al., 2023c), POPE-Popular: POPE-P (Li et al., 2023c), POPE-Adversarial: POPE-A (Li et al., 2023c) and MMVP (Tong et al., 2024) benchmarks. RobustVQA-A, RobustVQA-R, and RobustVQA-V are robustness benchmarks designed by us in Appendix E.1. ”AVG” denotes the overall average accuracy of seven benchmarks. ”SFT” denotes the supervised fine-tuning. ”\*” denotes baseline model without diffusion feedback. The evaluation metrics for each benchmark are listed in Table 12. 

<table><tr><td>Method</td><td>SFT</td><td>Architecture</td><td>RVQA-A</td><td>RVQA-R</td><td>RVQA-V</td><td>POPE-R</td><td>POPE-P</td><td>POPE-A</td><td>MMVP</td><td>AVG</td></tr><tr><td colspan="11">Models for Text-Generation Only</td></tr><tr><td>Shikra (Chen et al., 2023)</td><td>✓</td><td>ViT-L/LLaMA 7B</td><td>33.71</td><td>38.33</td><td>37.45</td><td>86.90</td><td>83.97</td><td>83.10</td><td>22.56</td><td>55.15</td></tr><tr><td>NeXT-Chat (Zhang et al., 2023a)</td><td>✓</td><td>ViT-L/Vicuna 7B</td><td>44.82</td><td>43.67</td><td>47.30</td><td>87.70</td><td>84.57</td><td>81.93</td><td>27.41</td><td>59.62</td></tr><tr><td colspan="11">Models for Text and Image Generation</td></tr><tr><td>MM-Interleaved (Tian et al., 2024)</td><td>✘</td><td>ViT-L/Vicuna 13B</td><td>50.76</td><td>52.71</td><td>50.60</td><td>64.73</td><td>65.33</td><td>65.20</td><td>23.82</td><td>53.31</td></tr><tr><td>Emu-I (Sun et al., 2023b)</td><td>✓</td><td>ViT-L/Vicuna 7B</td><td>46.40</td><td>49.12</td><td>47.36</td><td>61.28</td><td>56.79</td><td>56.01</td><td>22.69</td><td>48.52</td></tr><tr><td>SEED (Ge et al., 2023)</td><td>✓</td><td>ViT-G/Vicuna 7B</td><td>52.06</td><td>59.71</td><td>57.02</td><td>69.84</td><td>56.83</td><td>59.63</td><td>25.62</td><td>54.39</td></tr><tr><td>DreamLLM (Dong et al., 2023)</td><td>✓</td><td>ViT-L/Vicuna 7B</td><td>51.43</td><td>58.96</td><td>57.60</td><td>86.36</td><td>80.07</td><td>72.63</td><td>26.37</td><td>61.84</td></tr><tr><td>SEED-X (Ge et al., 2024)</td><td>✓</td><td>ViT-G/Vicuna 13B</td><td>52.36</td><td>60.27</td><td>59.49</td><td>86.41</td><td>81.43</td><td>74.56</td><td>29.16</td><td>63.39</td></tr><tr><td>DEEM *</td><td>✘</td><td>ConvNext-B/Vicuna 7B</td><td>53.24</td><td>56.06</td><td>54.72</td><td>50.55</td><td>52.00</td><td>51.93</td><td>20.30</td><td>48.40</td></tr><tr><td>DEEM</td><td>✘</td><td>ConvNext-B/Vicuna 7B</td><td>56.86</td><td>68.63</td><td>63.08</td><td>69.93</td><td>70.27</td><td>68.87</td><td>28.74</td><td>60.91</td></tr><tr><td>DEEM -VQA</td><td>✓</td><td>ConvNext-B/Vicuna 7B</td><td>55.22</td><td>64.12</td><td>62.99</td><td>87.40</td><td>82.80</td><td>78.49</td><td>32.89</td><td>65.56</td></tr></table>

Table 2: Multi-modal comprehension evaluation. “ED” denotes using extra in-house data. Benchmarks include COCO (Chen et al., 2015); I2Para.: Image2Paragraph (Krause et al., 2017); $\mathrm { \Delta V Q A ^ { v 2 } } ;$ : VQAv2 (Goyal et al., 2017); OKVQA (Marino et al., 2019); GQA (Hudson & Manning, 2019); VizWiz (Gurari et al., 2018); VisDial (Das et al., 2017); MMBench: MMB (Yu et al., 2024); MMVet (Yu et al., 2024);. The evaluation metrics for each benchmark are listed in Table 12. 

<table><tr><td>Model</td><td>LLM</td><td>VFM</td><td>ED</td><td>COCO</td><td>I2Para.</td><td> $VQA^{v2}$ </td><td>OKVQA</td><td>GQA</td><td>VizWiz</td><td>VisDial</td><td>MMB</td><td>MMVet</td></tr><tr><td colspan="13">Models for Text-Generation Only</td></tr><tr><td>IDEFICS-80B (IDEFICS, 2023)</td><td>LLaMA-65B</td><td>ViT-H</td><td>✗</td><td>91.8</td><td>-</td><td>60.0</td><td>-</td><td>45.2</td><td>36.0</td><td>-</td><td>27.9</td><td>-</td></tr><tr><td>IDEFICS-80B-I (IDEFICS, 2023)</td><td>LLaMA-65B</td><td>ViT-H</td><td>✗</td><td>117.2</td><td>-</td><td>37.4</td><td>-</td><td>-</td><td>26.0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>KOSMOS-1 (Huang et al., 2024)</td><td>MetaLM</td><td>ViT-L</td><td>√</td><td>-</td><td>-</td><td>46.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>KOSMOS-2 (Peng et al., 2023)</td><td>KOSMOS-1</td><td>ViT-L</td><td>√</td><td>-</td><td>-</td><td>45.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Flamingo-9B (Alayrac et al., 2022)</td><td>Chinchilla-7B</td><td>ViT-L</td><td>√</td><td>79.4</td><td>-</td><td>51.8</td><td>44.7</td><td>-</td><td>28.8</td><td>48.0</td><td>7.9</td><td>23.3</td></tr><tr><td>Flamingo-80B (Alayrac et al., 2022)</td><td>Chinchilla-70B</td><td>ViT-H</td><td>√</td><td>84.3</td><td>-</td><td>56.3</td><td>50.6</td><td>-</td><td>31.6</td><td>52.0</td><td>-</td><td>-</td></tr><tr><td>mPLUG-DocOwl (Ye et al., 2023)</td><td>LLaMA-7B</td><td>ViT-L</td><td>✗</td><td>52.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>60.8</td><td>35.7</td></tr><tr><td>BLIP-2 (Li et al., 2023a)</td><td>Vicuna-7B</td><td>ViT-L</td><td>✗</td><td>-</td><td>-</td><td>-</td><td>-</td><td>38.6</td><td>25.3</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BLIP-2 (Li et al., 2023a)</td><td>Vicuna-13B</td><td>ViT-L</td><td>✗</td><td>-</td><td>-</td><td>41.0</td><td>-</td><td>41.0</td><td>19.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td>InstructBLIP (Dai et al., 2024)</td><td>Vicuna-7B</td><td>ViT-L</td><td>✗</td><td>-</td><td>-</td><td>-</td><td>-</td><td>49.2</td><td>34.5</td><td>-</td><td>68.9</td><td>33.1</td></tr><tr><td>InstructBLIP (Dai et al., 2024)</td><td>Vicuna-13B</td><td>ViT-L</td><td>✗</td><td>-</td><td>-</td><td>-</td><td>-</td><td>49.5</td><td>33.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Shikra (Chen et al., 2023)</td><td>Vicuna-13B</td><td>ViT-L</td><td>✗</td><td>117.5</td><td>-</td><td>77.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LLaVA-1.5 (Liu et al., 2023)</td><td>Vicuna-7B</td><td>ViT-L</td><td>✗</td><td>-</td><td>-</td><td>78.5</td><td>-</td><td>62.0</td><td>50.0</td><td>-</td><td>53.1</td><td>32.9</td></tr><tr><td>LLaVA-1.5 (Liu et al., 2023)</td><td>Vicuna-13B</td><td>ViT-L</td><td>✗</td><td>-</td><td>-</td><td>80.0</td><td>-</td><td>63.3</td><td>53.6</td><td>-</td><td>60.6</td><td>35.6</td></tr><tr><td>Qwen-VL (Bai et al., 2023)</td><td>Qwen-7B</td><td>ViT-G</td><td>✗</td><td>-</td><td>-</td><td>78.8</td><td>-</td><td>59.3</td><td>35.2</td><td>-</td><td>32.9</td><td>13.0</td></tr><tr><td>Qwen-VL-Chat (Bai et al., 2023)</td><td>Qwen-7B</td><td>ViT-G</td><td>√</td><td>-</td><td>-</td><td>78.2</td><td>-</td><td>57.5</td><td>38.9</td><td>-</td><td>59.1</td><td>-</td></tr><tr><td colspan="13">Models for both Image and Text Generation</td></tr><tr><td>CM3Leon (Yu et al., 2023)</td><td>-</td><td>-</td><td>√</td><td>61.6</td><td>10.5</td><td>47.6</td><td>23.8</td><td>-</td><td>37.6</td><td>22.6</td><td>-</td><td>-</td></tr><tr><td>Emu (Sun et al., 2023b)</td><td>Vicuna-13B</td><td>ViT-L</td><td>√</td><td>112.4</td><td>-</td><td>52.0</td><td>38.2</td><td>-</td><td>34.2</td><td>47.4</td><td>-</td><td>-</td></tr><tr><td>Emu-I (Sun et al., 2023b)</td><td>Vicuna-13B</td><td>ViT-L</td><td>√</td><td>117.7</td><td>-</td><td>40.0</td><td>34.7</td><td>-</td><td>35.4</td><td>48.0</td><td>-</td><td>-</td></tr><tr><td>Emu2 (Sun et al., 2023a)</td><td>LLaMA-33B</td><td>ViT-L</td><td>√</td><td>-</td><td>-</td><td>33.3</td><td>26.7</td><td>-</td><td>40.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DreamLLM (Dong et al., 2023)</td><td>Vicuna-7B</td><td>ViT-L</td><td>✗</td><td>103.7</td><td>8.4</td><td>72.9</td><td>52.2</td><td>-</td><td>49.3</td><td>-</td><td>58.2</td><td>36.6</td></tr><tr><td>DEEM -VQA</td><td>Vicuna-7B</td><td>ConvNext-B</td><td>✗</td><td>115.4</td><td>22.4</td><td>68.2</td><td>53.4</td><td>55.7</td><td>50.4</td><td>42.1</td><td>60.8</td><td>37.4</td></tr></table>

# 3.3 ABLATION STUDY

In this study, we conduct ablation studies on several key components of the model, including consistency semantic regularization, training latency, scalability and the impact of different architectures. Benchmarks include RobustVQA-A:RVQA-A; RobustVQA-R: RVQA-R; RobustVQA-V:RVQA-V; POPE-R (Li et al., 2023c); POPE-P (Li et al., 2023c); POPE-A (Li et al., 2023c); MMVP (Tong et al., 2024); OK-VQA (Marino et al., 2019). More additional ablation studies can be found in Appendix D.

Consistency Semantic Regularization and Training Latency. To evaluate the effectiveness of the key elements of our design, we conduct the following ablation experiments. We first pre-train a baseline model without using the consistency semantic regularization term under the same training setting for comparison to demonstrate the effectiveness of our architecture. As we can see from Table 3, during the pre-training phase, using our consistency semantic regularization can significantly enhance the model’s performance on both hallucination and robustness benchmarks. Moreover, we load the weights of the pre-trained model for image-text instruction fine-tuning experiments. In the second phase of image-text instruction fine-tuning experiments, we demonstrate the effectiveness of our model design. As detailed in Table 3, we observe that after fine-tuning with image-text instruction data, the model’s visual hallucination ability improves further, but its visual perception robustness decreases. However, using our consistency semantic regularization can mitigate the robustness degradation while further enhancing the model’s visual hallucination ability. To explore the impact of introducing consistency semantic regularization on the training latency in the two stages of training, we conduct corresponding ablation experiments. We present the result in Table 3. Employing consistency semantic regularization adds only a marginal increase in training latency, yet it significantly enhances the model’s robustness capabilities.

Table 3: Ablation study of $\mathcal { L } _ { \mathbf { C S R } }$ and training latency. Using semantic consistency regularization during both the pre-training and supervised fine-tuning phases can significantly enhance the model’s robustness and resistance to hallucinations, while incurring only a marginal additional training cost. 

<table><tr><td>SFT</td><td> $\mathcal{L}_{\text{CSR}}$ </td><td>RVQA-A</td><td>RVQA-R</td><td>RVQA-V</td><td>POPE-R</td><td>POPE-P</td><td>POPE-A</td><td>SPEED</td></tr><tr><td>X</td><td>X</td><td>53.2</td><td>56.1</td><td>54.7</td><td>50.6</td><td>52.0</td><td>51.9</td><td>8.11 s/step</td></tr><tr><td>X</td><td>√</td><td>57.8</td><td>69.0</td><td>64.8</td><td>69.9</td><td>70.3</td><td>68.9</td><td>9.25 s/step</td></tr><tr><td>√</td><td>X</td><td>51.3</td><td>56.5</td><td>57.4</td><td>85.4</td><td>78.8</td><td>76.2</td><td>2.14 s/step</td></tr><tr><td>√</td><td>√</td><td>53.5</td><td>57.6</td><td>58.1</td><td>86.0</td><td>79.2</td><td>77.1</td><td>2.22 s/step</td></tr></table>

Table 4: Ablation study of model scalability. Gradually expanding the training data and model size can further enhance the model’s capabilities, demonstrating the scalability of the approach. 

<table><tr><td>Architecture</td><td>Training Data</td><td>RVQA-A</td><td>RVQA-R</td><td>RVQA-V</td><td>POPE-R</td><td>POPE-P</td><td>POPE-A</td><td>OK-VQA</td></tr><tr><td>ConvNext-B/Vicuna 7B</td><td>32K</td><td>51.86</td><td>54.31</td><td>52.73</td><td>48.44</td><td>50.10</td><td>50.06</td><td>20.74</td></tr><tr><td>ConvNext-B/Vicuna 7B</td><td>96K</td><td>52.31</td><td>57.43</td><td>54.06</td><td>54.42</td><td>57.22</td><td>56.35</td><td>22.33</td></tr><tr><td>ConvNext-B/Vicuna 7B</td><td>160K</td><td>52.89</td><td>58.93</td><td>55.31</td><td>60.28</td><td>60.74</td><td>59.96</td><td>23.65</td></tr><tr><td>ConvNext-L/Vicuna 7B</td><td>160K</td><td>53.23</td><td>60.47</td><td>56.88</td><td>61.12</td><td>62.87</td><td>62.09</td><td>23.87</td></tr><tr><td>ConvNext-B/Vicuna 13B</td><td>160K</td><td>53.92</td><td>61.27</td><td>57.02</td><td>62.60</td><td>64.26</td><td>63.19</td><td>31.11</td></tr></table>

Model Scalability. Although DEEM demonstrates better performance with smaller data count and model sizes, its scalability has yet to be validated. As is well known, scalability is crucial for model performance. We conduct ablation experiments to assess the scalability concerning data count and model size. As shown in Table 4, gradually increasing the training data enables the model to successfully scale while achieving improved results. Additionally, increasing the sizes of both the VFM and LLM leads to sustained performance enhancements, indicating that DEEM possesses good scalability.

Table 5: Ablation study of different architectures. Our method not only significantly enhances the capabilities of LLMs for text and image generation with marginal additional training costs, but it also improves the performance of LLMs for text generation only, validating the generalization ability of the approach. 

<table><tr><td>Name</td><td> $\mathcal{L}_{\text{CSR}}$ </td><td>MMVP</td><td>RVQA-A</td><td>RVQA-R</td><td>RVQA-V</td><td>POPE-R</td><td>POPE-P</td><td>POPE-A</td></tr><tr><td>LLaVA</td><td>✘</td><td>18.6</td><td>54.8</td><td>60.0</td><td>58.7</td><td>55.5</td><td>53.3</td><td>54.6</td></tr><tr><td>LLaVA</td><td>✓</td><td>25.1</td><td>56.7</td><td>66.7</td><td>61.9</td><td>67.9</td><td>68.7</td><td>65.4</td></tr><tr><td>DEEM</td><td>✘</td><td>20.3</td><td>53.2</td><td>56.1</td><td>54.7</td><td>50.6</td><td>52.0</td><td>51.9</td></tr><tr><td>DEEM</td><td>✓</td><td>28.7</td><td>56.9</td><td>68.6</td><td>63.1</td><td>69.9</td><td>70.3</td><td>68.9</td></tr></table>

Impact of Different Architectures. By cleverly reusing the diffusion model from LMMs for image and text generation, we can significantly enhance the model’s foundational visual perception, visual robustness, and anti-hallucination capabilities with only marginal additional training costs. However, whether DEEM possesses sufficient generalization ability to remain effective for LMMs on text generation only has yet to be explored. To validate our hypothesis, we employ the LLaVA (Liu et al., 2024a) architecture and conducted ablation experiments using semantic consistency regularization loss, with results presented in Table 5. We observe that utilizing diffusion feedback to improve the basic perceptual capabilities of LMMs—thus preventing the model from overly compressing visual information and losing sensitivity to subtle details—is a general method that is architecture-agnostic and exhibits good generalization properties. This suggests that the benefits of our approach could extend beyond the specific configurations tested, potentially enhancing a wide range of LMMs in various applications.

# 4 RELATED WORK

# 4.1 DIFFUSION MODELS FOR REPRESENTATION LEARNING

Diffusion models have made significant progress in various generative tasks (Song et al., 2020; Ho et al., 2020b), such as image generation (Betker et al., 2023), video generation (Ho et al., 2022), and object tracking (Luo et al., 2023). In addition to the aforementioned research, many studies focus on leveraging diffusion models for representation learning. Some works utilize the conditional control of pre-trained diffusion models to flexibly address different downstream tasks, including object classification (Xiang et al., 2023), semantic segmentation (Xu et al., 2023), image caption (Wei et al., 2024), and keypoint matching (Nam et al., 2023). Other studies (Li et al., 2023b; Song et al., 2024) design specialized modules and train diffusion models from scratch to further enhance representation capabilities. Although diffusion models have been widely applied in the generative tasks of large multimodal models, the use of diffusion models to optimize the visual representations of large multimodal models has yet to be explored. To our knowledge, we are the first to employ diffusion models in a self-supervised paradigm to optimize the visual representations of large multimodal models, significantly enhancing their perceptual abilities and reliability at minimal cost.

# 4.2 LARGE MULTIMODAL MODEL

Image-to-text large multimodal models (LMMs) (Luo et al., 2025; Liu et al., 2024c; Zhang et al., 2023b; Wang et al., 2024; Zhou et al., 2024; Liu et al., 2024b) inject visual information into large language models (LLMs) through vision foundation models (VFMs), allowing the language models to perceive visual inputs and thus generate captions or answer questions based on the given multimodal content. Flamingo (Alayrac et al., 2022) tries to extract vision features with a resampler, and transfer them into the text features with a cross-attention mechanism. Instead of using crossattention layers, BLIP-2 (Li et al., 2023a) directly feed the visual features into the LLMs as soft prompts and significantly reduce the training cost by reducing the visual token number. LLaVA (Liu et al., 2024a) and MiniGPT-4 (Zhu et al., 2023) construct a small-scale instruction tuning dataset to better align the LMM with the expected output format. Although this unidirectional image-to-text paradigm has achieved tremendous success, it still fails to unify multimodal tasks like text-to-image generation and image-to-text visual question answering, significantly limiting the capabilities of multimodal models.

In order to unify multimodal tasks into a unified manner, some works (Yu et al., 2023; Koh et al., 2024; Sun et al., 2023b; Dong et al., 2023; Tian et al., 2024; Ge et al., 2023; 2024; Luo et al., 2024) attempt to generate images and text in the interleaved context concurrently. The release of some public large-scale interleaved image-text datasets (Laurenc¸on et al., 2024; Zhu et al., 2024) has significantly advanced the development of this field. CM3Leon (Yu et al., 2023) converts images into discrete tokens, facilitating token-level auto-regressive modeling as traditional language modeling. Although CM3Leon showcases competitive image generation capabilities, it exhibits notable weaknesses in image understanding. Emu (Sun et al., 2023b) and DreamLLM (Dong et al., 2023) focus on single-stage end-to-end modeling using raw image pixels as input for interleaved image-text generation modeling, but they feed image information at the input of LMMs, which are limited by the problem that fixed number of visual tokens cannot efficiently describe image details. MM-Interleaved (Tian et al., 2024) addresses this limitation by integrating image details into LMMs via multi-scale visual features. However, when faced with out-of-distribution noisy data, the image encoders used by LMMs often produce incorrect visual information, ultimately leading to erroneous predictions. This significantly limits the application of the models in safety-critical scenarios. Building on an advanced interleaved content modeling mechanism, we propose DEEM , which cleverly reuses DMs to correct the outputs of the VFMs without increasing extra parameter count, thereby enhancing the model’s generalization capabilities and reducing visual hallucinations in a self-supervised manner. Similar to previous work (Liu et al., 2024a; Dong et al., 2023; Tian et al., 2024), after supervised fine-tuning, it achieves competitive performance on multiple downstream multimodal tasks with the smallest scale.

# 5 CONCLUSION

Can diffusion models serve as the eyes of large language models for image perception? In this paper, we answer the question by proposing a novel method called DEEM , which leverages a diffusion model as the eyes for LLMs. This approach enhances the robustness of the multimodal model for interleaved image-text modeling and reduces visual hallucinations without introducing extra modules. Through comprehensive exploratory experiments, we demonstrate the effectiveness of the proposed DEEM method. In addition to its advanced robust performance and visual hallucination handling capabilities, we adopt an additional two-stage instruction fine-tuning process to broaden the application scenarios of our DEEM . This enables DEEM to handle a variety of multimodal tasks, including visual question answering, image captioning, and region-level image reasoning. Besides, this work initiates the first step towards visual robustness via generative feedback in a multimodal model. In the future, we will continue to enhance the model’s ability to conduct better multimodal comprehension and creation tasks. As an end-to-end framework, we hope it will spur further research in the multimodal robustness field, such as multimodal agents that can handle complex tasks that require safety abilities.

# 6 ACKNOWLEDGMENTS

Min Yang is supported by National Key Research and Development Program of China (2022YFF0902100), National Natural Science Foundation of China (Grant No. 62376262), the Natural Science Foundation of Guangdong Province of China (2024A1515030166). Xiaobo Xia is supported by MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, University of Science and Technology of China (Grant No. 2421002).

# REFERENCES

Armen Aghajanyan, Bernie Huang, Candace Ross, Vladimir Karpukhin, Hu Xu, Naman Goyal, Dmytro Okhonko, Mandar Joshi, Gargi Ghosh, Mike Lewis, et al. Cm3: A causal masked multimodal model of the internet. arXiv preprint arXiv:2201.07520, 2022.   
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, pp. 23716–23736, 2022.   
Kopf Andreas, Vencu Richard, Coombes Theo, and Beaumont Romain. Laion coco: 600m synthetic ¨ captions from laion2b-en, 2022.   
Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In ICCV, pp. 2425–2433, 2015.   
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.   
James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.   
Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In CVPR, pp. 3558–3568, 2021.   
Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.   
Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan Yuille. Detect what you can: Detecting and representing objects using holistic models and body parts. In CVPR, pp. 1971–1978, 2014.   
Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and ´ C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.   
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. In NeurIPS, 2024.   
Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, Jose MF Moura, Devi ´ Parikh, and Dhruv Batra. Visual dialog. In CVPR, pp. 326–335, 2017.   
Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. In NeurIPS, pp. 19822–19835, 2021.   
Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. In NeurIPS, pp. 16890–16902, 2022.

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499, 2023.   
Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Makea-scene: Scene-based text-to-image generation with human priors. In ECCV, pp. 89–106, 2022.   
Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large language model. arXiv preprint arXiv:2307.08041, 2023.   
Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.   
Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, pp. 6904–6913, 2017.   
Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, pp. 3608–3617, 2018.   
Ju He, Shuo Yang, Shaokang Yang, Adam Kortylewski, Xiaoding Yuan, Jie-Neng Chen, Shuai Liu, Cheng Yang, Qihang Yu, and Alan Yuille. Partimagenet: A large, high-quality dataset of parts. In ECCV, pp. 128–145, 2022.   
Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV, pp. 8340–8349, 2021a.   
Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In CVPR, pp. 15262–15271, 2021b.   
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, pp. 6840–6851, 2020a.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020b.   
Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.   
Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. In NeurIPS, 2024.   
Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, pp. 6700–6709, 2019.   
IDEFICS. Introducing idefics: An open reproduction of state-of-the-art visual language model. https://huggingface.co/blog/idefics, 2023.   
Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, pp. 787–798, 2014.   
Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. Generating images with multimodal language models. In NeurIPS, 2024.   
Jonathan Krause, Justin Johnson, Ranjay Krishna, and Li Fei-Fei. A hierarchical approach for generating descriptive image paragraphs. In CVPR, pp. 317–325, 2017.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International Journal of Computer Vision, 123:32–73, 2017.   
Hugo Laurenc¸on, Lucile Saulnier, Leo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, ´ Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. In NeurIPS, 2024.   
Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In ICML, pp. 12888–12900, 2022.   
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pp. 19730–19742, 2023a.   
Tianhong Li, Dina Katabi, and Kaiming He. Self-conditioned image generation via generating representations. arXiv preprint arXiv:2312.03701, 2023b.   
Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023c.   
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ´ ECCV, pp. 740–755, 2014.   
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024a.   
Xiaohao Liu, Xiaobo Xia, Zhuo Huang, and Tat-Seng Chua. Towards modality generalization: A benchmark and prospective analysis. arXiv preprint arXiv:2412.18277, 2024b.   
Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, pp. 11976–11986, 2022.   
Ziqiang Liu, Feiteng Fang, Xi Feng, Xinrun Du, Chenhao Zhang, Zekun Wang, Yuelin Bai, Qixuan Zhao, Liyang Fan, Chengguang Gan, Hongquan Lin, Jiaming Li, Yuansheng Ni, Haihong Wu, Yaswanth Narsupalli, Zhigang Zheng, Chengming Li, Xiping Hu, Ruifeng Xu, Xiaojun Chen, Min Yang, Jiaheng Liu, Ruibo Liu, Wenhao Huang, Ge Zhang, and Shiwen Ni. Ii-bench: An image implication understanding benchmark for multimodal large language models, 2024c.   
Run Luo, Zikai Song, Lintao Ma, Jinlin Wei, Wei Yang, and Min Yang. Diffusiontrack: Diffusion model for multi-object tracking. arXiv preprint arXiv:2308.09905, 2023.   
Run Luo, Haonan Zhang, Longze Chen, Ting-En Lin, Xiong Liu, Yuchuan Wu, Min Yang, Minzheng Wang, Pengpeng Zeng, Lianli Gao, et al. Mmevol: Empowering multimodal large language models with evol-instruct. arXiv preprint arXiv:2409.05840, 2024.   
Run Luo, Ting-En Lin, Haonan Zhang, Yuchuan Wu, Xiong Liu, Min Yang, Yongbin Li, Longze Chen, Jiaming Li, Lei Zhang, et al. Openomni: Large language models pivot zero-shot omnimodal alignment across language with real-time self-aware emotional speech synthesis. arXiv preprint arXiv:2501.04561, 2025.   
Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, pp. 11–20, 2016.   
Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, pp. 3195–3204, 2019.

Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, pp. 947–952, 2019.   
Jisu Nam, Gyuseong Lee, Sunwoo Kim, Hyeonsu Kim, Hyoungwon Cho, Seyeon Kim, and Seungryong Kim. Diffusion model for dense matching. arXiv preprint arXiv:2305.19094, 2023.   
Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.   
Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. In NeurIPS, 2011.   
Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.   
Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with localized narratives. In ECCV, pp. 647–664, 2020.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pp. 8748–8763, 2021.   
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, pp. 8821–8831, 2021.   
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In ICML, pp. 5389–5400, 2019.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In CVPR, pp. 10684–10695, 2022.   
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, pp. 36479–36494, 2022.   
Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.   
Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, pp. 25278– 25294, 2022.   
Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In ECCV, pp. 146–162, 2022.   
Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In ECCV, pp. 742–758, 2020.   
Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, pp. 8317–8326, 2019.   
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Zikai Song, Ying Tang, Run Luo, Lintao Ma, Junqing Yu, Yi-Ping Phoebe Chen, and Wei Yang. Autogenic language embedding for coherent point tracking. arXiv preprint arXiv:2407.20730, 2024.   
Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. arXiv preprint arXiv:2312.13286, 2023a.   
Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In ICLR, 2023b.   
Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, et al. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. arXiv preprint arXiv:2401.10208, 2024.   
Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms, 2024.   
Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, pp. 4566–4575, 2015.   
Zhaoqing Wang, Xiaobo Xia, Ziye Chen, Xiao He, Yandong Guo, Mingming Gong, and Tongliang Liu. Open-vocabulary segmentation with unpaired mask-text supervision. arXiv preprint arXiv:2402.08960, 2024.   
Chen Wei, Chenxi Liu, Siyuan Qiao, Zhishuai Zhang, Alan Yuille, and Jiahui Yu. De-diffusion makes text a strong cross-modal interface. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13492–13503, 2024.   
Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. arXiv preprint arXiv:2212.00280, 2022.   
Weilai Xiang, Hongyu Yang, Di Huang, and Yunhong Wang. Denoising diffusion autoencoders are unified self-supervised learners. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15802–15812, 2023.   
Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Openvocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2955–2966, 2023.   
Yinfei Yang, Daniel Cer, Amin Ahmad, Mandy Guo, Jax Law, Noah Constant, Gustavo Hernandez Abrego, Steve Yuan, Chris Tar, Yun-Hsuan Sung, et al. Multilingual universal sentence encoder for semantic retrieval. arXiv preprint arXiv:1907.04307, 2019.   
Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499, 2023.   
Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for contentrich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.   
Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2(3), 2023.   
Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In International conference on machine learning. PMLR, 2024.   
Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In CVPR, pp. 6720–6731, 2019.

Ao Zhang, Yuan Yao, Wei Ji, Zhiyuan Liu, and Tat-Seng Chua. Next-chat: An lmm for chat, detection and segmentation. arXiv preprint arXiv: 2311.04498, 2023a.   
Shaokun Zhang, Xiaobo Xia, Zhaoqing Wang, Ling-Hao Chen, Jiale Liu, Qingyun Wu, and Tongliang Liu. Ideal: Influence-driven selective annotations empower in-context learners in large language models. arXiv preprint arXiv:2310.10873, 2023b.   
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS, 2024.   
Yiwei Zhou, Xiaobo Xia, Zhiwei Lin, Bo Han, and Tongliang Liu. Few-shot adversarial prompt learning on vision-language models. Advances in Neural Information Processing Systems, 37: 3122–3156, 2024.   
Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.   
Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billionscale corpus of images interleaved with text. In NeurIPS, 2024.

# CONTENTS

# 1 Introduction 1

# 2 Method 3

2.1 Architecture 3   
2.2 Pipeline . . . 4   
2.3 Training and Inference 5

# 3 Experiment 6

3.1 Implementation Details 6   
3.2 Experimental Results 6   
3.3 Ablation Study 8

# 4 Related Work 9

4.1 Diffusion Models for Representation Learning . . 9   
4.2 Large Multimodal Model 10

# 5 Conclusion 10

# 6 Acknowledgments 11

A Limitation 18   
B Broader Impacts 18   
C Additional Experiments Results 18

# D Additional Ablation Study 19

D.1 Ablation Study of Input Image Resolution 19   
D.2 Ablation Study of Training Recipes 19

# E Additional Implementation Details 19

E.1 Dataset Construction 19   
E.2 Image-Text Alignment Pre-training . . . 20   
E.3 Image-Text Instruction Fine-tuning 20   
E.4 Mask-Text Instruction Fine-tuning 21   
E.5 Evaluation . 21

# F Additional Visualization Examples 21

F.1 Semantic Image Synthesis 21   
F.2 Text Condition Image Synthesis 23   
F.3 Robustness Comparison . . 24

F.4 Image-Text Multimodal Dialogue 24

F.5 Mask-Text Multimodal Dialogue 24

# A LIMITATION

Although our method significantly enhances the visual robustness of interleaved image-text modeling multimodal models after image-text alignment pre-training, it, unfortunately, cannot eliminate but only alleviate the robustness knowledge forgetting issue caused by subsequent fine-tuning, as shown in the Table 3. Additionally, our model requires using a diffusion model as another eye to correct and update the erroneous knowledge of the image encoder to improve the overall visual robustness of the multimodal model. However, updating larger image encoders such as CLIP-ViT-L and CLIP-ViT-G(Radford et al., 2021) will increase the training overhead, which may limit the application of our model. We hope that in the future, the diffusion model can completely replace the image encoder to further enhance the effectiveness of our method.

# B BROADER IMPACTS

The proposed method introduces a novel strategy to enhance the robustness and generalization capabilities of multimodal models by leveraging a diffusion model as an additional eye for large language models. This strategy allows for the correction and updating of potential semantic errors in the image encoder, leading to significant improvements in handling out-of-distribution data and mitigating visual hallucinations. Overall, our contributions provide a significant step forward in the field of multimodal, offering a robust, efficient, and scalable solution for improving the accuracy and reliability of multimodal models. The broader impacts of this work include the potential for more intelligent and adaptive AI systems that can operate effectively in diverse and challenging environments.

Table 6: Zero-shot region-level image captioning results on ReferCOCOg. 

<table><tr><td>Method</td><td>Type</td><td>METEOR</td><td>CIDEr</td></tr><tr><td>GRIT (Wu et al., 2022)</td><td>Box</td><td>15.2</td><td>71.6</td></tr><tr><td>Kosmos-2 (0-shot) (Peng et al., 2023)</td><td>Box</td><td>12.2</td><td>60.3</td></tr><tr><td>Kosmos-2 (2-shot) (Peng et al., 2023)</td><td>Box</td><td>13.8</td><td>62.2</td></tr><tr><td>Kosmos-2 (4-shot) (Peng et al., 2023)</td><td>Box</td><td>14.1</td><td>62.3</td></tr><tr><td>NeXt-Chat (Zhang et al., 2023a)</td><td>Box</td><td>12.0</td><td>79.0</td></tr><tr><td>DEEM -Mask</td><td>Mask</td><td>14.1</td><td>71.0</td></tr></table>

# C ADDITIONAL EXPERIMENTS RESULTS

Region-Level Image Captioning. In addition to holistic image understanding, we also validate the model’s ability to take region-level image captioning. As shown in Fig. 3, we use a mask-aware extractor to obtain region-level visual features and address region-level image captioning tasks. We adopt the RefCOCOg (Mao et al., 2016) validation set and compare it with other state-of-the-art (SOTA) models, including GRIT (Wu et al., 2022), Kosmos-2 (Peng et al., 2023), and NeXt-Chat (Zhang et al., 2023a). The CIDEr (Vedantam et al., 2015) and METEOR are applied as the evaluation metrics. As shown in Table 6, our model is capable of achieving competitive performance on CIDEr and METEOR across all of the compared methods, which shows the superiority of our DEEM .

Table 7: Zero-shot text-to-image generation FID on MS-COCO and LN-COCO. 

<table><tr><td>Method</td><td>MS-COCO</td><td>LN-COCO</td></tr><tr><td colspan="3">Text-to-Image Specialists</td></tr><tr><td>Retrieval Result</td><td>17.97</td><td>33.59</td></tr><tr><td>DALL-E (Ramesh et al., 2021)</td><td> $\sim 28$ </td><td>-</td></tr><tr><td>CogView (Ding et al., 2021)</td><td>27.10</td><td>-</td></tr><tr><td>CogView2 (Ding et al., 2022)</td><td>24.00</td><td>-</td></tr><tr><td>Stable Diffusion (Rombach et al., 2022)</td><td>12.43</td><td>34.26</td></tr><tr><td>GLIDE (Nichol et al., 2021)</td><td>12.24</td><td>-</td></tr><tr><td>Make-A-Scene (Gafni et al., 2022)</td><td>11.84</td><td>-</td></tr><tr><td>DALL-E 2 (Ramesh et al., 2022)</td><td>10.39</td><td>-</td></tr><tr><td>Muse-3B (Yang et al., 2019)</td><td>7.88</td><td>-</td></tr><tr><td>Imagen-3.4B (Saharia et al., 2022)</td><td>7.27</td><td>-</td></tr><tr><td>Parti-20B (Yu et al., 2022)</td><td>7.23</td><td>15.97</td></tr></table>

Models for both Image and Text Generation 

<table><tr><td>CM3-13B (Aghajanyan et al., 2022)</td><td>29.56</td><td>-</td></tr><tr><td>GILL-8B (Koh et al., 2024)</td><td>12.25</td><td>-</td></tr><tr><td>Emu-13B (Sun et al., 2023b)</td><td>11.66</td><td>-</td></tr><tr><td>CM3Leon-7B (Yu et al., 2023)</td><td>10.82</td><td>-</td></tr><tr><td>DreamLLM-7B (Dong et al., 2023)</td><td>8.76</td><td>22.42</td></tr><tr><td>DEEM -7B (Ours)</td><td>8.89</td><td>24.13</td></tr></table>

Text-to-Image Generation. we evaluate text-conditional image generation on MS-COCO (Lin et al., 2014) and LN-COCO (Pont-Tuset et al., 2020). On MSCOCO, we sample 8 images per text condition and use CLIP-ViT-L (Radford et al., 2021) to rerank based on text-image similarity. CLIP reranking is not used for LN-COCO. FID (Heusel et al., 2017) is used to evaluate both datasets. As shown in Table 7, our model shows competitive text-to-image generation compared to existing image and text generation models. See qualitative results on text-to-image synthesis in Fig. 10 in Appendix F.

Table 8: Ablation study of input image resolution and coefficient λ with 2k training steps and 16 batch size. 

<table><tr><td colspan="3">SFT resolution λ</td><td>RVQA-A</td><td>RVQA-R</td><td>RVQA-V</td><td>POPE-R</td><td>POPE-P</td><td>POPE-A</td><td>OK-VQA</td></tr><tr><td>X</td><td>256</td><td>1</td><td>51.6</td><td>52.0</td><td>49.6</td><td>48.5</td><td>50.0</td><td>50.0</td><td>18.9</td></tr><tr><td>X</td><td>256</td><td>5</td><td>51.9</td><td>54.3</td><td>52.7</td><td>48.4</td><td>50.1</td><td>50.0</td><td>20.7</td></tr><tr><td>X</td><td>256</td><td>10</td><td>51.7</td><td>52.7</td><td>51.9</td><td>48.7</td><td>50.3</td><td>50.3</td><td>20.1</td></tr><tr><td>✓</td><td>256</td><td>5</td><td>51.5</td><td>59.1</td><td>57.9</td><td>85.9</td><td>77.1</td><td>76.4</td><td>38.7</td></tr><tr><td>✓</td><td>448</td><td>5</td><td>52.5</td><td>57.6</td><td>58.1</td><td>86.0</td><td>79.2</td><td>77.1</td><td>41.0</td></tr></table>

# D ADDITIONAL ABLATION STUDY

we provide more ablation studies for DEEM in this section, all of which share the same settings. All the code, models, and data tools will be released soon.

# D.1 ABLATION STUDY OF INPUT IMAGE RESOLUTION

In addition to the aforementioned exploration, we also scale up the input image resolution for performance gain. The performance gain becomes larger when further increasing the input image resolution from 256 to 448 in image-text instruction fine-tuning, as shown in Table 8. Such results indicate our method could better exploit the additional information gained from high resolution. Moreover, we conduct an ablation study on coefficient λ in loss function. As shown in Table 8, setting λ = 5 achieves a better balance between robustness and hallucination empirically.

# D.2 ABLATION STUDY OF TRAINING RECIPES

We also conduct an ablation study to control the trainability of different training modules. As shown in Table 10, we found that freezing the DM (Diffusion Model) while not freezing the VFM (Visual Foundation Model) during training yields the best robustness and hallucination results.

# E ADDITIONAL IMPLEMENTATION DETAILS

# E.1 DATASET CONSTRUCTION

As shown in Fig. 5, we first convert the original ImageNet-A (Hendrycks et al., 2021b), ImageNet-R (Hendrycks et al., 2021a), and ImageNet-V2 (Recht et al., 2019) data into a VQA format that the multimodal model can evaluate. Specifically, we use the CLIP-ViT-L model for hard example mining, predicting the incorrect category label with the highest confidence score apart from the ground truth category label. We then use a pre-defined prompt as: “Is [category label] the main object in this image? Please answer yes or no.” to simultaneously construct a pair of positive and negative example samples, allowing the model to answer “yes” or “no”. By using this design, we can evaluate the robustness of multimodal models in an unbiased manner with the new benchmark called RobustVQA, facilitating both assessment and comparison. It is worth noting that, as shown in Table 9, we find that the yes or no format is more stable than the multiple-choice format and can better evaluate the robustness of multi-modal models.

Table 9: Comparison of different VQA formats. Questions in the yes or no format can well evaluate the performance of the models on the RobustVQA benchmark, while questions in the multiplechoice format are very random, and MM-interleaved tend to output the first option. Therefore, we adopt yes or no format in our experimental settings. More details about the new benchmark RobustVQA design can be found in Appendix E.1. 

<table><tr><td>Format</td><td>Prompt</td><td>RobustVQA-A</td><td>RobustVQA-R</td><td>RobustVQA-V</td></tr><tr><td rowspan="2">multiple-choice</td><td>“What is the main object in this image?”</td><td rowspan="2">44.88</td><td rowspan="2">58.88</td><td rowspan="2">46.86</td></tr><tr><td>“Chose from the list: [false category label,gt category label].”</td></tr><tr><td rowspan="3">multiple-choice</td><td>“What is the main object in this image?”</td><td rowspan="3">84.60</td><td rowspan="3">90.16</td><td rowspan="3">82.92</td></tr><tr><td>“Chose from the list: [gt category label,false category label].”</td></tr><tr><td>“Is [gt category label] the main object in this image?”</td></tr><tr><td rowspan="3">yes or no</td><td>“Please answer yes or no.”</td><td rowspan="3">50.76</td><td rowspan="3">52.71</td><td rowspan="3">50.60</td></tr><tr><td>“Is [false category label] the main object in this image?”</td></tr><tr><td>“Please answer yes or no.”</td></tr></table>

![](images/c607076dd49e4af8db131ce7d378968da462eb25342344d694d058ad4895c889.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Image Collection: {Sealion', 'Drake', 'Goose', 'Killer whale', .....}"] --> B["Hard Example Mining"]
    B --> C{Probability}
    C -->|Incorrect category
Highest confidence| D["Sealion"]
    C -->|Incorrect category
Highest confidence| E["Drake"]
    C -->|Incorrect category
Highest confidence| F["Goose"]
    C -->|Incorrect category
Highest confidence| G["Killer whale"]
    C -->|Positive| H["RobustVQA Format"]
    D --> I["No"]
    E --> J["Yes"]
    F --> K["No"]
    G --> L["Yes"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#fcc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#fff,stroke:#333
```
</details>

Figure 5: Robustness dataset construction process. We use the CLIP-ViT-L model for hard example mining and then transform them into question-answer pairs via a pre-defined template.

# E.2 IMAGE-TEXT ALIGNMENT PRE-TRAINING

We use MMC4-Core (Zhu et al., 2024), LAION-400M (Schuhmann et al., 2021), SBU (Ordonez et al., 2011), and CC-12M (Changpinyo et al., 2021) as the pre-training dataset. For LAION-400M (Schuhmann et al., 2021), SBU (Ordonez et al., 2011), and CC-Table 10: Ablation study of training recipe in image-text alignment pre-training with 10k training steps and 128 batch size.

<table><tr><td>VFM</td><td>DM</td><td>RVQA-A</td><td>RVQA-R</td><td>RVQA-V</td><td>POPE-R</td><td>POPE-P</td><td>POPE-A</td></tr><tr><td>freeze</td><td>unfreeze</td><td>53.2</td><td>56.1</td><td>54.7</td><td>50.6</td><td>52.0</td><td>51.9</td></tr><tr><td>unfreeze</td><td>freeze</td><td>56.8</td><td>68.6</td><td>63.1</td><td>69.9</td><td>70.3</td><td>68.9</td></tr><tr><td>unfreeze</td><td>unfreeze</td><td>50.3</td><td>52.5</td><td>53.1</td><td>54.8</td><td>56.3</td><td>56.1</td></tr></table>

12M (Changpinyo et al., 2021), instead of utilizing the original annotations, we use the version filtered by the pre-trained BLIP-2 model (Li et al., 2023a). For simplicity, we refer to it as BLIP-LCS hereafter. ”LCS” abbreviates the LAION, CC, and SBU datasets. Text prompts with lengths shorter than 10 are also filtered out. Due to network constraints, we only collect approximately 6M of MMC4-Core and 20M of BLIP-LCS data. The sampling probability of MMC4 is twice that of BLIP-LCS. The images are inserted before or after the corresponding text sentence with equal probability. Specifically, images with a CLIP similarity score below 0.24 will be discarded, and only 6 images at most will be kept for each document in MMC4-Core. We also exclude 100% of all documents that do not contain any images, and 50% of documents that contain only 1 image. For image-text-pair BLIP-LCS datasets, we randomly sample multiple image-text pairs from the same dataset and concatenate them to the maximum context length (i.e., 2048) during pre-training. For interleaved image and text MMC4-Core (Zhu et al., 2024) datasets, we also split and concatenate the documents to form the training samples. Such a concatenation strategy can utilize the full context window of Large Language Models and thus achieve high data efficiency. Besides that, for image generation, we ignore the training loss of images which are the first element in the sequence. The text condition of the rest images is dropped with a 10% probability to improve classifier-free guidance sampling. The detailed hyper-parameters of image-text alignment pre-training are listed in Table 11.

# E.3 IMAGE-TEXT INSTRUCTION FINE-TUNING

We utilize public available datasets for supervised fine-tuning, including LLaVA-665K(Liu et al., 2024a), COCO Caption (Chen et al., 2015), VQAv2 (Goyal et al., 2017),TextCaps (Sidorov et al., 2020), OCR-VQA (Mishra et al., 2019), GQA (Hudson & Manning, 2019), OK-VQA (Marino et al., 2019), TextVQA (Singh et al., 2019), and AOK-VQA (Schwenk et al., 2022). We use the following prompt template ‘‘Based on the image, please answer the question. {image} {question}. The answer is: {answer} " to convert the data into a mixture of instruction following forms, resulting in approximately 800K instruction data for the secondstage image-text instruction fine-tuning. The detailed hyper-parameters of image-text instruction fine-tuning are listed in Table 11.

# E.4 MASK-TEXT INSTRUCTION FINE-TUNING

We collect short text and pixel-level mask pairs from the publicly available object-level datasets (COCO, RefCOCO, RefCOCO+) and part-level datasets (Pascal Part, Part Imagenet), then transform them into instruction following data. Moreover, Visual Genome (VG) and Visual Commonsense Reasoning (VCR) datasets are employed to add more multiple region understanding data, resulting in approximately 200K instruction data for the third-stage mask-text instruction fine-tuning. See more hyper-parameters details in Table 11.

# E.5 EVALUATION

As shown in Fig. 6, DEEM achieves the best results on both hallucination and robustness benchmarks even at the smallest scale, demonstrating the efficiency and effectiveness of our approach. In addition to visual robustness and hallucination, we also use various benchmarks and datasets, such as image caption, visual question answering, text-to-image generation and so on, to assess the image-text comprehension capabilities. All these evaluation tasks and metrics are listed in Table 12. The prompt templates for each task are listed in Fig. 8.

![](images/53dc6da6dc18151f79344c322b476e08c690a751d532d4c83465ab2bcfb21476.jpg)

<details>
<summary>radar</summary>

| Model              | RobustVQA-A | RobustVQA-R | RobustVQA-V | POPE-R | POPE-P | POPE-A |
| ------------------ | ----------- | ----------- | ----------- | ------ | ------ | ------ |
| MM-Interleaved Stage1 (ViT-L/Vicuna 13B) | 50         | 55          | 60          | 65     | 70     | 65     |
| NExT-Chat (ViT-L/Vicuna 7B)    | 40          | 45          | 50          | 80     | 85     | 80     |
| Shikra (ViT-L/LLaMA 7B)       | 40          | 45          | 50          | 20     | 80     | 85     |
| LLaVa (ViT-L/Vicuna 13B)      | 50          | 55          | 60          | 40     | 50     | 55     |
| DEEM (ConvNext-B/Vicuna 7B)    | 60          | 65          | 70          | 90     | 95     | 90     |
</details>

Figure 6: Performance on visual robustness and hallucination benchmark. DEEM achieves the best results on robustness benchmark and competitive performance on hallucination even at the smallest scale, demonstrating the efficiency and effectiveness of our approach.

# F ADDITIONAL VISUALIZATION EXAMPLES

# F.1 SEMANTIC IMAGE SYNTHESIS

Dynamic Semantic Bias Erasure. We demonstrate the dynamic semantic bias elimination process through three iterations on the same sample, providing an illustration of the original image alongside its version reconstructed in real-time according to semantic conditions, as shown in Fig. 9. Our method, DEEM , gradually mitigates potential erroneous semantics within the visual encoder through multiple iterations, ultimately enhancing the perceptual capabilities of MLLMs.

Consistency Semantic Image Synthesis We visualize some consistency semantic image synthesis and display both the original images and their reconstructed versions in Fig. 11. DEEM accurately recovers the features of the original images without causing distortion.

<table><tr><td colspan="2">Visual input example, Goose:</td></tr><tr><td></td><td><img src="images/581c9fe1c3b3033bddbf396665b7c6143b4d5a24b4ce2b4dc7d1a8b4a90b5455.jpg"/></td></tr><tr><td>User</td><td>Based on the image, please answer the question. What is the main object in this image?Chose from the list: [&#x27;Sealion&#x27;, &#x27;Drake&#x27;, &#x27;Goose&#x27;, &#x27;Killer whale&#x27;].Let&#x27;s think step by step.</td></tr><tr><td>DEEM (Ours)</td><td>A carved goose sitting on a flat surface. So the answer is goose √</td></tr><tr><td>LLaVA</td><td>The main object in this image is a drake. ✗</td></tr><tr><td>NExT-Chat</td><td>A sculpture of a duck sitting on a table. So the answer is duck. ✗</td></tr><tr><td>MM-Interleaved</td><td>Sealion ✗</td></tr></table>

Figure 7: Case Comparison. Compared to other SOTA models, including LLaVA, NeXt-Chat, and MM-Interleaved, when encountering out-of-distribution data, their models are affected by incorrect semantics from the image encoder and cannot output the correct answer. However, DEEM can output the correct answer via generative feedback.

Table 11: Training recipes for DEEM . The three training stages are introduced in Section 2.3. Stage I: Image-Text Alignment Pre-training, Stage II: Image-Text Instruction Fine-tuning, Stage III: Mask-Text Instruction Fine-tuning. 

<table><tr><td></td><td>Stage I</td><td>Stage II</td><td>Stage III</td></tr><tr><td>Phase</td><td>Image-Text Alignment</td><td>Image-Text SFT</td><td>Mask-Text SFT</td></tr><tr><td colspan="4">Training Hyper-Parameters</td></tr><tr><td>Input image resolution</td><td>256×256</td><td>448×448</td><td>448×448</td></tr><tr><td>Output image resolution</td><td>512×512</td><td>512×512</td><td>512×512</td></tr><tr><td>VFM</td><td>CLIP-ConvNext-B</td><td>CLIP-ConvNext-B</td><td>CLIP-ConvNext-B</td></tr><tr><td>LLM</td><td>Vicuna-7B v1.5</td><td>Vicuna-7B v1.5</td><td>Vicuna-7B v1.5</td></tr><tr><td>DM</td><td>Stable Diffusion v2.1</td><td>Stable Diffusion v2.1</td><td>Stable Diffusion v2.1</td></tr><tr><td>λ</td><td>5</td><td>5</td><td>5</td></tr><tr><td rowspan="2">Learning Rate</td><td>2e-5 (image encoder&amp;decoder)</td><td>1e-6 (language model)</td><td>1e-6 (language model)</td></tr><tr><td>1e-4 (others)</td><td>1e-5 (others)</td><td>1e-5 (others)</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Optimizer hyper-parameters</td><td> $\beta_1, \beta_2, \epsilon = 0.9, 0.995, 1e-6$ </td><td> $\beta_1, \beta_2, \epsilon = 0.9, 0.999, 1e-8$ </td><td> $\beta_1, \beta_2, \epsilon = 0.9, 0.999, 1e-8$ </td></tr><tr><td>Weight Decay</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Training iterations</td><td>10k</td><td>10k</td><td>10k</td></tr><tr><td>Warmup steps</td><td>1k</td><td>500</td><td>500</td></tr><tr><td>Learning Rate Scheduler</td><td>Cosine</td><td>Cosine</td><td>Cosine</td></tr><tr><td>Batch Size Per GPU</td><td>4</td><td>16</td><td>2</td></tr><tr><td>Maximum Token Length</td><td>2048</td><td>2048</td><td>2048</td></tr><tr><td>Augmentation</td><td>CenterCrop</td><td>-</td><td>-</td></tr><tr><td>Unfreeze LLM</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>Unfreeze DM</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>Unfreeze VFM</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td colspan="4">Training Data</td></tr><tr><td>Dataset</td><td>1 MMC42 BLIP-LCS</td><td>1 LLaVA-Mix-665K2 VQA-Mixture3 COCO Caption</td><td>1 COCO/ReferCOCO/ReferCOCO+2 Pascal-Part/Part-ImageNet4 VG/VRC</td></tr><tr><td>Data Size</td><td>~26M</td><td>~800K</td><td>~200K</td></tr><tr><td>Data Type</td><td>Interleave/Pair</td><td>Instruction</td><td>Instruction</td></tr><tr><td colspan="4">Training Cost</td></tr><tr><td>GPU Device</td><td>32×NVIDIA A100</td><td>32×NVIDIA A100</td><td>32×NVIDIA A100</td></tr><tr><td>Training Time</td><td>~30h</td><td>~6h</td><td>~3h</td></tr></table>

Table 12: Overall descriptions of the evaluation benchmarks for evaluating capabilities, including image-level captioning, image-level visual question answering, text-to-image generation, regionlevel image captioning, visual robustness, comprehension, perception and hallucination. 

<table><tr><td></td><td>Dataset</td><td>Task description</td><td>Eval Split</td><td>Metric</td></tr><tr><td rowspan="2">CAP.</td><td>COCO (Chen et al., 2015)</td><td>Scene description</td><td>test</td><td> $CIDEr(\uparrow)$  (Vedantam et al., 2015)</td></tr><tr><td>Image2Paragraph (Krause et al., 2017)</td><td>Scene description</td><td>test</td><td> $CIDEr(\uparrow)$  (Vedantam et al., 2015)</td></tr><tr><td rowspan="5">VQA.</td><td>VQAv2 (Goyal et al., 2017)</td><td>Scene understanding QA</td><td>test-dev</td><td>VQA  $Acc(\uparrow)$  (Antol et al., 2015)</td></tr><tr><td>OKVQA (Marino et al., 2019)</td><td>External knowledge QA</td><td>val</td><td>VQA  $Acc(\uparrow)$  (Antol et al., 2015)</td></tr><tr><td>GQA (Hudson &amp; Manning, 2019)</td><td>Scene understanding QA</td><td>test-dev</td><td>VQA  $Acc(\uparrow)$  (Antol et al., 2015)</td></tr><tr><td>VizWiz (Gurari et al., 2018)</td><td>Scene understanding QA</td><td>test-dev</td><td>VQA  $Acc(\uparrow)$  (Antol et al., 2015)</td></tr><tr><td>VisDial (Das et al., 2017)</td><td>Image dialogue</td><td>val</td><td>NDCG(↑)</td></tr><tr><td rowspan="2">SYN.</td><td>MS-COCO (Lin et al., 2014)</td><td>Text-Conditional Image Synthesis</td><td>val-30K</td><td>FID(↓) (Heusel et al., 2017)</td></tr><tr><td>LN-COCO (Pont-Tuset et al., 2020)</td><td>Text-Conditional Image Synthesis</td><td>val</td><td>FID(↓) (Heusel et al., 2017)</td></tr><tr><td rowspan="3">REF.</td><td>RefCOCO (Kazemzadeh et al., 2014)</td><td>Region-level scene description</td><td>val</td><td> $CIDEr(\uparrow)$  (Vedantam et al., 2015)</td></tr><tr><td>RefCOCO+ (Mao et al., 2016)</td><td>Region-level scene description</td><td>val</td><td> $CIDEr(\uparrow)$  (Vedantam et al., 2015)</td></tr><tr><td>RefCOCOg (Mao et al., 2016)</td><td>Region-level scene description</td><td>val</td><td> $CIDEr(\uparrow)$  (Vedantam et al., 2015)</td></tr><tr><td rowspan="3">OOD.</td><td>RobustVQA-V</td><td>Out-of-Distribution Robustness</td><td>val</td><td>Acc(↑)</td></tr><tr><td>RobustVQA-R</td><td>Out-of-Distribution Robustness</td><td>val</td><td>Acc(↑)</td></tr><tr><td>RobustVQA-A</td><td>Out-of-Distribution Robustness</td><td>val</td><td>Acc(↑)</td></tr><tr><td rowspan="3">Hall.</td><td>POPE-R (Li et al., 2023c)</td><td>Visual Hallucination</td><td>val</td><td>Acc(↑)</td></tr><tr><td>POPE-P (Li et al., 2023c)</td><td>Visual Hallucination</td><td>val</td><td>Acc(↑)</td></tr><tr><td>POPE-A (Li et al., 2023c)</td><td>Visual Hallucination</td><td>val</td><td>Acc(↑)</td></tr><tr><td rowspan="2">CPH.</td><td>MMBench (Yu et al., 2024)</td><td>Visual Comprehension</td><td>val</td><td>Acc(↑)</td></tr><tr><td>MMVet (Yu et al., 2024)</td><td>Visual Comprehension</td><td>val</td><td>Acc(↑)</td></tr><tr><td>PCP.</td><td>MMVP (Tong et al., 2024)</td><td>Visual Perception</td><td>val</td><td>Acc(↑)</td></tr></table>

<table><tr><td colspan="2">System Prompt
You are a helpful assistant.</td></tr><tr><td colspan="2">(a) VQA</td></tr><tr><td colspan="2">USER: Based on the image, please answer the question. {IMAGE} {QUESTION} Please provide an accurate answer within one word.</td></tr><tr><td colspan="2">ASSISTANT: The answer is &lt;ANSWER&gt;</td></tr><tr><td colspan="2">(b) Image Captioning</td></tr><tr><td colspan="2">USER: {IMAGE} Could you provide a short description of the image?</td></tr><tr><td colspan="2">ASSISTANT: A photo of &lt;ANSWER&gt;</td></tr><tr><td colspan="2">(c) Referring</td></tr><tr><td colspan="2">USER: {IMAGE} Provide a short description for this &lt;ref&gt;region1&lt;/ref&gt; {MASK}</td></tr><tr><td colspan="2">ASSISTANT: &lt;ANSWER&gt;</td></tr><tr><td colspan="2">(d) Hallucination</td></tr><tr><td colspan="2">USER: Based on the image, please answer the question. {IMAGE} 
Is there a &lt;OBJECT&gt; in the image? Please answer yes or no.</td></tr><tr><td colspan="2">ASSISTANT: The answer is &lt;ANSWER&gt;</td></tr><tr><td colspan="2">(e) Robustness</td></tr><tr><td colspan="2">USER: Based on the image, please answer the question. {IMAGE} 
Is &lt;OBJECT&gt; the main object in this image? Please answer yes or no.</td></tr><tr><td colspan="2">ASSISTANT: The answer is &lt;ANSWER&gt;</td></tr></table>

Figure 8: Prompt template used for evaluation. (a) VQA includes VQAv2, VizWiz, OKVQA, GQA, VisDial, and MMVP. (b) Image Captioning includes COCO, Image2Paragraph. (c) Regionlevel Image Captioning includes RefCOCOg. (d) Visual hallucination includes POPE. (e) Visual Robustness includes RobustVQA-A, RobustVQA-R, and RobustVQA-V. < IMAGE >denotes the input image representation, < MASK > denotes the mask-level image representation, < QUESTION >denotes each specific question, < ANSWER > is the generated answer, and < OBJECT > is the specific object name in a question of POPE and RobustVQA.

# F.2 TEXT CONDITION IMAGE SYNTHESIS

In Fig. 10, we present some text-to-image synthesis examples from DEEM , demonstrating its capability to generate corresponding images based on given prompts.

![](images/6bc9a719ceafdd26654e154a9e89cfeeb9776acd6326030ed170384ecbfd73de.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["loss"] --> B["Goose seal duck"]
    B --> C["Goose seal duck"]
    C --> D["Goose seal duck"]
    D --> E["Goose seal duck"]
    E --> F["Final Model"]
```
</details>

Figure 9: Dynamic semantic bias elimination process through three iterations on the same sample, diffusion process is conducted by adding 65% noise to the original image as the initial condition.

# F.3 ROBUSTNESS COMPARISON

In Fig. 7, we present a comparative analysis of visual robustness results between our model, DEEM , and other state-of-the-art models: LLaVA (Liu et al., 2024a), NeXt-Chat (Zhang et al., 2023a), and MM-Interleaved (Tian et al., 2024). When encountering natural adversarial samples or out-ofdistribution samples, the image encoder in their models will output incorrect semantic information, leading to incorrect category answers. In contrast, our method uses a diffusion model as the eyes of the large language model to inspect and correct the output features of the image encoder. This process eliminates incorrect semantic outputs from the image encoder, ultimately allowing the large language model to produce the correct category answer. This simple yet effective approach significantly enhances the model’s robustness and generalization capabilities.

# F.4 IMAGE-TEXT MULTIMODAL DIALOGUE

In Fig. 12, we show the image-text dialogue case examples of DEEM . Our model can input any interleaved layout of text-image data and simultaneously understand and generate text-image outputs in any interleaved layout, representing the future of next-generation multimodal dialogue.

# F.5 MASK-TEXT MULTIMODAL DIALOGUE

In addition to image-level input, DEEM also supports mask-text input to perform fine-grained region-level reasoning tasks. As shown in the Fig. 13, DEEM can accurately extract region semantics of the image based on the input mask and complete the corresponding instruction tasks.

(a)   
![](images/c907f75214fbc51d8ecda9420f88bd9d80f75563ab16e8cea8350a9982c92a75.jpg)

<details>
<summary>natural_image</summary>

Close-up of a fluffy gray tabby cat with a wide open mouth, no visible text or symbols
</details>

![](images/619c5663f086f34bde5ca791bd59e134a717936ff28d8c00c746488c6ff91537.jpg)

<details>
<summary>natural_image</summary>

Illustration of a white pala with red and blue accents on a grassy field, surrounded by trees and a yellow umbrella (no text or symbols)
</details>

![](images/318c5dd5cf40e6585106c72e35f7be798b557973a168611da8f026aff7a20430.jpg)

<details>
<summary>natural_image</summary>

Close-up illustration of a bird's head with pink and white feathers against a teal background (no text or symbols)
</details>

![](images/362312fa90f6066479a9355ded45e480680f3c4d750dc30f033780ca67ae4397.jpg)

<details>
<summary>natural_image</summary>

Illustration of a hummingbird perched on a branch with abstract green and blue background (no text or symbols)
</details>

![](images/0e9623b60a8abffb7a4ecfa896c9ba95004506a6c795e61e7723bf65967ebd67.jpg)

<details>
<summary>natural_image</summary>

Close-up of a fluffy gray tabby cat with a wide open mouth, no visible text or symbols
</details>

![](images/90adc66487adadd6a4337eb64d8ff4cf9b6f8b43e72a85b90591b9ad0dff5dce.jpg)

<details>
<summary>natural_image</summary>

Illustration of a colorful outdoor play structure with red pom-pom and blue stripe, surrounded by trees and grass (no text or symbols)
</details>

![](images/fd07516ba6508fcb97840f56daedbe7e6833ca921d50536e5a24d316c6a7cbee.jpg)

<details>
<summary>natural_image</summary>

Close-up illustration of a bird's head with vibrant pink and orange plumage against a teal background (no text or symbols)
</details>

![](images/9c5a862d4820fcf2a1b678c5a41650face4538fa28c8d4afc0b007e61c6c47b4.jpg)

<details>
<summary>natural_image</summary>

Illustration of a hummingbird perched on a branch against a textured blue-green background (no text or symbols)
</details>

(c)   
![](images/f7982e53ed4fe8c49345ec5eec86fbdd1681bd93cb8f90d1cf1e4e10c0924556.jpg)

<details>
<summary>natural_image</summary>

Circular arrangement of assorted food items including vegetables, fruits, and snacks arranged radially (no text or symbols)
</details>

![](images/4b1930abc617c87d7aa73ada278cad916bab8a7307ea639619d5c65c685bd510.jpg)

<details>
<summary>natural_image</summary>

Abstract 3D rendering of swirling, wavy patterns resembling a tree or geological structure (no text or symbols)
</details>

![](images/ed191da79f7749eb213f09165da0044a5500c79ca94a2ec153691a287d440787.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a small wheeled vehicle with wheels and suspension components (no text or symbols visible)
</details>

![](images/63f311664538d259074a1124c44c39d7c2f740cc0538ee91dd409fd63aa9a2c0.jpg)

<details>
<summary>natural_image</summary>

Close-up of a wooden fence with a white light source, partially obscured by tree branches (no text or symbols visible)
</details>

(d)   
![](images/fe284c33a3fbbb91fc19a67553d774b35dfa0062fdc10808fd841b3797a7a746.jpg)

<details>
<summary>natural_image</summary>

Circular arrangement of various food items including vegetables, meatballs, and utensils arranged radially (no text or symbols visible)
</details>

![](images/2d5f606b8f2ecd86615d6717ab1014919e8c833d24d766d5ea64542090e2569d.jpg)

<details>
<summary>natural_image</summary>

3D rendered cylindrical object with swirling black-and-white patterns, no text or symbols visible
</details>

![](images/b448f26f09e5222c7d9c7c0a7fb1faf05b3292f484bf28a642828baded199ffa.jpg)

<details>
<summary>natural_image</summary>

3D model of a small wheeled cargo cart with support legs and wheels (no text or symbols visible)
</details>

![](images/02d4cbf68d9b5f03e3d5f23d34f56362affcf2245cbba2f57f93d6c9a8e80f0a.jpg)

<details>
<summary>natural_image</summary>

Wooden fence with a small white object on top, surrounded by greenery (no text or symbols visible)
</details>

(e)   
![](images/65ed83c853e52a960ea1289363025cd07ba23ae685d83a1175bc1226baf6376d.jpg)

![](images/cef38c2e8a94c26a1f7afc2ba399c39b083b2eb350a3eb93400ed9eeecf894c4.jpg)

<details>
<summary>natural_image</summary>

Close-up illustration of a leopard's head and neck (no text or symbols)
</details>

![](images/5bbba550e94a3833d30a506ee2e2db4a89b157640394022dfa29325eb40516b9.jpg)

<details>
<summary>natural_image</summary>

Painting of a fox standing among green foliage (no text or symbols)
</details>

![](images/e6f4afae284da002f9ccb8c905c9dc9ef51483686473c1c0af71033bef68cc38.jpg)

<details>
<summary>natural_image</summary>

A dog running in a snowy, textured environment with a wall and tree (no visible text or symbols)
</details>

(f)   
![](images/dbaecf97e6bf0a3940af8b95d6e49d11d4ad2b4a9ce8875a100b2dba07e9adde.jpg)

<details>
<summary>natural_image</summary>

Classic black-and-white soccer ball with pentagon-hexagon pattern (no text or symbols)
</details>

![](images/1a1f118eeba29492a4ab58cabc035b578ac17f8eaee6de7d55cbb0d5a3854ebb.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a leopard with textured fur and visible eye eyes (no text or symbols)
</details>

![](images/419acbc0336b0f009d9a400e4d44feab19292c567ad88d25687b0ecff08843b7.jpg)

<details>
<summary>natural_image</summary>

Painting of a wolf in profile, no text or symbols visible
</details>

![](images/00069efffd7d480adf339c3deabe62e4cf3648af296ac4d4943bd1e02eae332e.jpg)

<details>
<summary>natural_image</summary>

Watercolor painting of a snowy landscape with trees and distant structures (no text or symbols)
</details>

Figure 10: image-to-image generation examples with the outputs of image encoder. (a,c,e) are original images and (b,d,f) are synthesis images based on the image embeddings of original images.

![](images/02a395b150ff67b82e7ff8994d9709102186a6aa6d2b3dbfe6c18bf5c671e316.jpg)

<details>
<summary>natural_image</summary>

A small bird standing on a dirt path among green foliage (no text or symbols visible)
</details>

![](images/bef43b734aae19a13878d65c69b01ed716f34a6318aaca8a480648ffb185632f.jpg)

<details>
<summary>natural_image</summary>

Close-up of a small bird with black, gray, and yellow plumage perched on a branch (no text or symbols visible)
</details>

![](images/441bae82736edec86d8d65e47a1d4db9d3c49e7a95eef368855ca6cd5fbdf4e2.jpg)

<details>
<summary>natural_image</summary>

Close-up of a small bird perched on a branch, with gray and yellow plumage (no text or symbols visible)
</details>

![](images/bf403c79ff8ce2e266e28c0865aa8a4acc4ed36ef5ab66fa430552818a6094bf.jpg)

<details>
<summary>natural_image</summary>

Close-up of a bird perched on a branch, wearing orange and black plumage (no text or symbols visible)
</details>

Small bird with a pale yellow underside light brown crown and back gray tail and wing tips tip of tail feather bright yellow black eyes and black strip over eyes

![](images/7706a6e51a394165b90f0c0769bc3023146578c61d84d53d7a30da581c32b549.jpg)

<details>
<summary>natural_image</summary>

Modern armchair with blue cushion, wooden frame, against wooden wall background (no text or symbols)
</details>

![](images/2222f0dd3c4b0d9db66833fee0d63e905ff7db6ea31b3bd350ddd107f420c722.jpg)

<details>
<summary>natural_image</summary>

Green oval-shaped cushion chair on a dirt path, with a wooden fence and greenery in the background (no text or symbols visible)
</details>

![](images/cf314c80a464e980d68ee740106424d034c4a3770bc31ffffaed1927702bde74.jpg)

<details>
<summary>natural_image</summary>

Three glasses of wine on a wooden table, with blurred background lighting and window view (no text or symbols visible)
</details>

![](images/5a764d803afdcf16187bfb2ef48aa97a998fbbf1679c713ca79cbf9eefe6a039.jpg)

<details>
<summary>natural_image</summary>

Two glass glasses filled with orange-colored wine, placed on a wooden table (no text or symbols visible)
</details>

An armchair in the shape of an avocad

A couple of glasses are sitting on a tabl

![](images/d8403212bbdbcb687a96e83aa6ddbcf94f91a819efe4a852e68f689f193b9d78.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a wooden bench with a door and brick wall in the background (no signage or text visible)
</details>

![](images/fb8eae0b48766311b7e62788a4125539d773e2f3e49444ed73ea6e1d597f45bb.jpg)

<details>
<summary>natural_image</summary>

Wooden bench with curved legs placed on a concrete wall against a brick wall (no text or symbols visible)
</details>

![](images/adc50fda9a030571fa040fd9a1407c2f22d3da52f39a48e9bde209dcd4ad2b05.jpg)

<details>
<summary>natural_image</summary>

Wooden bench placed on stone pavement with green ivy-covered wall behind (no text or symbols visible)
</details>

![](images/b917005dc7967b9160267246c3572a855c813c66473649dc5daaa052b7d63ab0.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a wooden bench with slats placed beside a concrete wall and greenery (no text or symbols visible)
</details>

A long wooden bench in front of a brick wall

![](images/c190adbcf9166ac361725406cfa8e6ff69f5c726b10db345bad834bd4934e8dc.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a modern office building (no signage)
</details>

![](images/ef9a18b7e990b3fb344717b821e2f1cbb48efaa842e9025b61b5ee5ff82081cc.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a modern city skyline featuring a prominent red pagoda and skyscrapers under a cloudy sky (no signage or text visible)
</details>

![](images/6eb7c3107d0342f9d3f82fb8b1948540377a943859b9f26df90184e812730983.jpg)

<details>
<summary>natural_image</summary>

Vintage yellow pickup truck parked in a desert landscape with sand dunes and mountains in the background (no visible text or symbols)
</details>

![](images/dd4009fca831ba63d8ddafe80dbd112fe8acf2822f2648ae86380e110b63c878.jpg)

<details>
<summary>natural_image</summary>

Rusty yellow pickup truck parked on a dirt road in a hilly landscape (no visible text or symbols)
</details>

Downtown Beijing at sunrise. detailed ink wash

A beat-up truck in the desert

Figure 11: Text-to-image generation examples with prompts. DEEM can generate vivid images based on input text conditions.

Photograph A:   
![](images/918c53c4f4b5df36c04d978ad76d96741e8b1f5c312ef53f57a7113493c9bb05.jpg)

![](images/a847e59355badd46cffd523d17eaf8da808fa26912553c8d3757d1a7d81c3a97.jpg)

Photograph B:   
![](images/b7bff3fa1be86bc99828691098102a1b411ff4404295ad3ead0d2b352e6c8d4d.jpg)

![](images/3d6b26366e596a148304cc2dde015245a5a420469df2b0a09d6e7c3106088b48.jpg)

What is written in photo A and B? What is the difference between these two photos? Answer in detail about the difference in written words and styles.

![](images/0edb3ed317ed8d567cf4d17867ff50da08d5d99a91dc22844a05ad69e3caf549.jpg)

![](images/136ae865adb5d70dbac777b171df92376518b091ad07af6b018bcc7bcd329cd1.jpg)

In Photograph A, the word \"DEEM\" is written on a pink background with hearts placed around it. This creates an eyecatching and unique display for this particular image of roses against another colorful rose patterned backdrop that complements DEEM's bright colors. On the other hand, in Photograph B, the same words are displayed using vintage typography style and printed over a light blue gradient. The overall effect looks more nostalgic or dreamy compared to Photograph A but has similar visual elements such as flowers arranged beneath DEEM. These differences highlight how design choices can impact mood and perception while retaining essential components like textual content.

help me mix the styles of the above two images to create a new image that combines both modern and romantic styles featuring the word "DEEM":

![](images/58fc86170d444c7d46a910ddd1c54432602f8d5e98d8913d7a858b4b70414693.jpg)

![](images/5dde198b47a243ea45dc4059b276c2dc33e60c1ddd4878cb547f955ee7a1008a.jpg)

![](images/1ed9cef4c465e7b9ebec7ba2f8a003611cc37ddcc294af2e7f61963ea9050e7b.jpg)

This image merges the sleek, contemporary elements with soft, pastel romantic touches, creating a unique blend of both aesthetics.

Figure 12: Examples of image-text multimodal dialogue between human and DEEM . Text and image can be used as inputs or outputs, and multi-round dialogue is shown.

![](images/b0e9035a9fae26e37ea49f04ea355be3b342c12842b864f3e4c9a596ed15975f.jpg)  
Figure 13: Examples of mask-text multimodal dialogue between human and DEEM . Text and mask can be used as inputs and DEEM outputs the corresponding answer, and multi-round dialogue is shown.