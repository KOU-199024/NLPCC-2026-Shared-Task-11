# REAR: RETHINKING VISUAL AUTOREGRESSIVE MODELS VIA GENERATOR-TOKENIZER CONSISTENCY REGULARIZATION

Qiyuan He1, Yicong Li1∗, Haotian Ye2, Jinghao Wang3, Xinyao Liao1 Pheng-Ann Heng3, Stefano Ermon2, James Zou2, Angela Yao1∗

1National University of Singapore 2Stanford University   
3The Chinese University of Hong Kong

# ABSTRACT

Visual autoregressive (AR) generation offers a promising path toward unifying vision and language models, yet its performance remains suboptimal against diffusion models. Prior work often attributes this gap to tokenizer limitations and rasterization ordering. In this work, we identify a core bottleneck from the perspective of generator-tokenizer inconsistency, i.e., the AR-generated tokens may not be well-decoded by the tokenizer. To address this, we propose reAR, a simple training strategy introducing a token-wise regularization objective: when predicting the next token, the causal transformer is also trained to recover the visual embedding of the current token and predict the embedding of the target token under a noisy context. It requires no changes to the tokenizer, generation order, inference pipeline, or external models. Despite its simplicity, reAR substantially improves performance. On ImageNet, it reduces gFID from 3.02 to 1.86 and improves IS to 316.9 using a standard rasterization-based tokenizer. When applied to advanced tokenizers, it achieves a gFID of 1.42 with only 177M parameters, matching the performance with larger state-of-the-art diffusion models (675M).

![](images/2af1140aebaedd55e7130ef2402af86b5ac7701b492c6316d074ef19dd7bcb80.jpg)

<details>
<summary>bar_line</summary>

| Tokenizer Sequence | IS → (higher is better) |
| ------------------ | ---------------------- |
| Tokenizer-friendly Token Sequence | 17 |
| AR-Generated Token Sequence | 17 |
| Decoder-only Transformer | 17 |
| Groundtruth Token Sequence | 17 |
| Tokenizer Codebook | 42 |
| Tokenizer Decoder | 60 |
| Tokenizer Codebook | 98 |
| Tokenizer Friendly Token Sequence | 34 |
| Tokenizer Friendly Token Sequence | 73 |
| Tokenizer Friendly Token Sequence | 20 |
| Tokenizer Friendly Token Sequence | 17 |
| Tokenizer Friendly Token Sequence | 20 |
| Tokenizer Friendly Token Sequence | 34 |
| Tokenizer Friendly Token Sequence | 73 |
| Tokenizer Friendly Token Sequence | 20 |
| Tokenizer Friendly Token Sequence | 20 |
| Tokenizer Friendly Token Sequence | 34 |
| Tokenizer Friendly Token Sequence | 73 |
| Tokenizer Friendly Token Sequence | 20 |
| Tokenizer Friendly Token Sequence | 20 |
| Tokenizer Friendly Token Sequence | 34 |
| Tokenizer Friendly Token Sequence | 73 |
| Tokenizer Friendly Token Sequence | 20 |
| Tokenizer Friendly Token Sequence | 20 |
</details>

(a) Visual autoregressive generation suffers from generator–tokenizer inconsistency: (1) Due to exposure bias, the AR model is more likely to generate token sequences unseen by the tokenizer; (2) Being embedding unaware, the embedding sequence of the generated discrete tokens may also be unfamiliar to the tokenizer, resulting in a cat in an unnatural pose, with its lower body flipped and the belly facing upward. The top and bottom images can still appear similar despite differing token indices, since distinct token sequences may map to nearby embeddings.   
(b) With generator-tokenizer consistency regularization, reAR with fewer parameters significantly improves over vanilla AR (gFID: 3.02 to 1.86, IS: 256.2 to 316.9) and even surpasses methods based on advanced tokenization and sophisticated generative paradigm.   
Figure 1: Generator-tokenizer inconsistency is the bottleneck in the visual autoregressive model.

# 1 INTRODUCTION

Autoregressive (AR) models, using a decoder-only transformer with the objective of next token prediction, are state-of-the-art for natural language generation (Team et al., 2023; Achiam et al., 2023). For image generation, however, AR models are less competitive than diffusion models (Dhariwal & Nichol, 2021; Peebles & Xie, 2023; Ma et al., 2024; Yu et al., 2024c). There is great interest in advancing visual autoregressive models to unify the language and visual modalities into a single generative framework (Bai et al., 2024; Team, 2024; Chung et al., 2024).

Scrutinizing the current design in visual AR, the dominant paradigm is to convert images into discrete tokens and train an autoregressive model on the converted token sequences. Specifically, a tokenizer is trained to split an image (or the feature) into patches and utilizes them into a sequence of discrete tokens (Esser et al., 2021; Sun et al., 2024; Luo et al., 2024), which it can use to reconstruct the original image. A decoder-only transformer using a causal mask is then trained on this token sequence in raster-scan order with the objective of next-token prediction. Unfortunately, this paradigm typically results in suboptimal performance compared to the diffusion model (Dhariwal & Nichol, 2021; Peebles & Xie, 2023; Ma et al., 2024; Yu et al., 2024c). Previous works have analyzed the performance gap from the perspective of tokenization, including token decomposition (Tian et al., 2024; Yu et al., 2024b; Bachmann et al., 2025; Pan et al., 2025) and sequence order (Pang et al., 2025; Yu et al., 2024a), rather than the whole system of visual autoregressive generation.

In this work, we provide a unified perspective on the key bottleneck of visual AR through the lens of generator-tokenizer inconsistency, which refers to the challenge that the autoregressive model might generate a token sequence that is hard for the tokenizer to decode back to an image. Specifically, we examine two sources of the inconsistencies inherited from the generated token sequence.

Firstly, the generated token sequence can be unseen by the tokenizer due to exposure bias. In autoregressive training, each token is predicted given the ground-truth context (teacher forcing), but at inference, the context consists of the model’s own predictions. Early mistakes then compound and lead to sequences never observed during training. While exposure bias is well studied in language (Bengio et al., 2015; Wang & Sennrich, 2020), it is amplified in visual AR. Text tokens are themselves the final output, so even an unseen sequence may still be semantically coherent. By contrast, visual tokens are decoded into images: a single wrong token can corrupt future predictions and decode into a token sequence never seen by the tokenizer during training, spreading structural artifacts across the image. As shown in Figure 1(a), an early misprediction (e.g., 42’→64’) cascades through subsequent tokens and yields a cat in an unnatural pose with a different coat color.

Secondly, the AR model suffers from embedding unawareness. During training, it optimizes only the discrete token indices without considering how these tokens are embedded by the tokenizer. However, the decoded image quality depends on the embeddings of the generated tokens rather than their indices alone, as shown in Figure 1(a). This unawareness leads to two issues: (i) even if two tokens are close in the embedding space, the model can only infer this relation indirectly from co-occurrence statistics, which is data-inefficient. and (ii) the embedding of an incorrect token is unconstrained by the ground-truth embedding, which can cause the overall sequence embedding to drift far from the training distribution of the tokenizer decoder. As illustrated in Figure 1(a), although the purple and red sequences contain the same number of incorrect tokens, the one with embeddings closer to the ground truth generates a decoded image of higher quality.

In this regard, we propose reAR, a unified training framework that explicitly regularizes the model toward tokenizer-friendly behavior. Concretely, we introduce two complementary strategies: 1) Noisy Context Regularization that exposes the model to perturbed context during training, reducing its reliance on clean contexts and improving robustness to imperfect histories at test time, thereby alleviating the model’s tendency to generate unseen token sequence; 2) Codebook Embedding Regularization that aligns the generator’s hidden states with the tokenizer’s embedding space, which encourages the generator to be aware of how tokens are decoded into visual patches. By learning to predict the embeddings rather than only discrete indices, even if the generator generates an unseen token sequence, the corresponding embedding sequence is optimized to be more compatible with the tokenizer. Combining them together, the token-wise consistency regularization can guide visual AR to be friendly to the tokenizer by predicting the visual embedding in a robust manner.

Building on reAR, we conduct extensive experiments comparing it against other generative frameworks. To show that reAR generalizes beyond specific tokenizers, we apply it to non-standard designs such as TiTok (Yu et al., 2024b) and AliTok (Wu et al., 2025). When combined with standard rasterization-order AR, reAR outperforms prior autoregressive methods even when those rely on sophisticated tokenizers, as Figure 1 (b) shows. Under the same model size and training budget, it also surpasses alternative paradigms such as MAR (Li et al., 2024), VAR (Tian et al., 2024), and

![](images/7d630d992a736a82cbe60ac2a8538ed2764d89dd69056ebe2e76848788e46837.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Next Token Prediction Loss"] --> B["Codebook Embedding Regularization"]
    B --> C["MLP"]
    C --> D["Causal Transformer Block"]
    D --> E["X"]
    C --> F["MLP"]
    F --> G["Causal Transformer Block"]
    G --> H["CodeBook Embedding"]
    H --> I["MLP"]
    I --> J["CodeBook Lookup"]
    J --> K["Tokenizer Encoder"]
    K --> L["CodeBook Indices"]
    L --> M["Noisy Context Regularization"]
    M --> N["CodeBook Embedding"]
    N --> O["MLP"]
    O --> P["Causal Transformer Block"]
    P --> Q["X"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#fcf,stroke:#333
    style H fill:#cff,stroke:#333
    style I fill:#ffc,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#fcc,stroke:#333
    style L fill:#cfc,stroke:#333
    style M fill:#fcc,stroke:#333
    style N fill:#cfc,stroke:#333
```
</details>

(a) reAR Training

![](images/34c30e88cd191cab09310420e3a31b3d9a641e26f04089c0d564c036f1600aea.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["17"] --> B["Causal Transformer Block"]
    C["73"] --> B
    D["60"] --> B
    E["98"] --> F["X"]
    G["Tokenizer Decoder"] --> H["Output Image"]
    I["Causal Transformer Block"] --> J["17"]
    K["73"] --> L["60"]
    M["98"] --> N["Output Image"]
    O["Decoder"] --> P["Output Image"]
```
</details>

(b) reAR Inference   
Figure 2: Overview of reAR, a plug-and-play framework that is agnostic to the visual tokenizer.

SiT (Ma et al., 2024). Furthermore, when paired with a tokenizer tailored for causal AR modeling (Wu et al., 2025), reAR achieves FID = 1.42 with only 177M parameters—competitive with the diffusion model REPA, which requires external representations and 675M parameters.

Our contributions can be summarized as follows:

• We identify the inconsistency between generator and tokenizer, where tokenizer fails to decode the generated token sequence, as the bottleneck of visual autoregressive generation;   
• We propose reAR, a plug-and-play training regularization that introduces visual inductive bias from the tokenizer and alleviates exposure bias to train the visual autoregressive model;   
• We demonstrate that reAR significantly improves visual autoregressive generation across different tokenizers (e.g., on VQGAN, FID improves from 3.02 to 1.86) and even surpasses more sophisticated generative models, using far fewer parameters.

# 2 PRELIMINARIES

Visual autoregressive generation is commonly divided into two components: (1) A visual tokenizer to tokenize the image; (2) An autoregressive model to sample the token sequence.

Visual Tokenizer. Visual tokenizers compress image pixels into discrete token sequences. The most commonly adopted methods are patch-based tokenizers (Esser et al., 2021; Sun et al., 2024; Yu et al., 2023; Chang et al., 2022). The tokenizer includes three parts: Encoder E, Quantizer Q and Decoder D. Formally, a given image $\mathbf { I } \in \mathbb { R } ^ { 3 \times H \times W }$ is converted to a feature $\hat { \mathbf { z } } \in \mathbb { R } ^ { \bar { c } \times h \times w }$ with the encoder E where $h \overset { \cdot } { < } H , w < W$ . It’s then processed into quantized embedding $\mathbf { z ^ { q } } \in \mathbb { R } ^ { c \times h \times w }$ via the quantizer Q and decoded back to reconstruct image ˆI by the decoder D:

$$
\hat {\mathbf {z}} = \mathcal {E} (\mathbf {I}), \quad \mathbf {z} ^ {\mathbf {q}} = \mathcal {Q} (\hat {\mathbf {z}}), \quad \hat {\mathbf {I}} = \mathcal {D} (\mathbf {z} ^ {\mathbf {q}}) \tag {1}
$$

The vector quantization is performed element-wise with a codebook ${ \mathcal { Z } } = \left\{ { \bf z } _ { 1 } , { \bf z } _ { 2 } , \ldots , { \bf z } _ { K } \right\} \quad \subset$ $\mathbb { R } ^ { c \times h \times w }$ by looking up the closest entry. Formally:

$$
\mathbf {z} _ {i j} ^ {\mathbf {q}} = \underset {\mathbf {z} _ {k} \in \mathcal {Z}} {\arg \min} \| \hat {\mathbf {z}} _ {i j} - \mathbf {z} _ {k} \|, \quad \mathbf {x} _ {i j} = \underset {k \in \{1, \dots , K \}} {\arg \min} \| \hat {\mathbf {z}} _ {i j} - \mathbf {z} _ {k} \|. \quad i = 1, \dots , h, \quad j = 1, \dots , w. \tag {2}
$$

where $\mathbf { x } _ { i j }$ forms the discrete token (indices such as 17 and 73). In the standard approach, it’s arranged into 1D token sequence via row-major rasterization order, i.e., $\left\{ \mathbf { x } _ { 1 1 } , \ldots , \mathbf { x } _ { 1 w } , \mathbf { x } _ { 2 1 } , \ldots , \mathbf { x } _ { h 1 } , \ldots , \mathbf { x } _ { h w } \right\}$ . The autoregressive model can then be trained on it..

Autoregressive Model. To model the distribution of a sequence of signal $\mathbf { x } _ { 1 : N } = \{ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { N } \}$ , the autoregressive model $p _ { \theta }$ aims to maximize the likelihood of the next token under teacher forcing:

![](images/e33010355bba1d18dc6de9d0c2355fe402ff23d13cf2f84fe9efc0408cadd25a.jpg)

![](images/b90af1556388b0a8187a4ca36c22edde9d6bf36f1c9a64a3e1b10137ca67586c.jpg)

![](images/f54e296e581c81a2eb87846acea34b2054be219e015ed5f085153dd0f8148031.jpg)

![](images/2e7187bf52653948fbdf29896ea48bd6536689a6f1bfb922af8e99f40816c525.jpg)  
(a) Tokenizer is sensitive to the error of generated tokens from exposure bias

![](images/18bdf9ec2f06ef2b3b6e1f938be5a74706703c9e88b2aa80e0a21ad7f66f6c2a.jpg)

![](images/227da861159bad7c8226b6c64f5ae24619366c5f78aebacf9d122cc218739954.jpg)

![](images/7366428b12a9dcbde70bbdc97986899e727dc70a83556a81777342b2868fbe88.jpg)

![](images/19934d2631e29505219c78b26bffecfbe4c01320949e13256ce9428e257d48b1.jpg)  
(b) Tokenizer is sensitive to the embedding of generated tokens   
Figure 3: Token sequence with the same correct token ratio (CTR) under teacher forcing can be decoded into images with different quality. Under the same CTR, (a) The images decoded from imperfect context is much less similar to the ground truth than the one from perfect context; (b) Replacing incorrect token with other incorrect tokens but with more similar embedding of the correct token, the generated image can be more similar to ground truth than original prediction.

$$
\theta = \underset {\theta} {\arg \max} \log p _ {\theta} (\mathbf {x} _ {1: N}) = \underset {\theta} {\arg \max} \sum_ {i = 1} ^ {N} \log p _ {\theta} (\mathbf {x} _ {i} | \mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {i - 1}) \tag {3}
$$

During inference, the model then decodes the sequence one by one. The tth token is sampled from the context $\mathbf { x } _ { 1 : t - 1 }$ by $\mathbf { x } _ { t } \sim p ( \cdot \mid \mathbf { x } _ { 1 : t - 1 } )$ under free running. In visual autoregressive generation, after sampling a sequence ˆx from pθ, it’s decoded into ˆI as the final generated image by the tokenizer decoder D.

# 3 REAR: REGULARIZING CONSISTENCY IN VISUAL AR

Different from natural language, ˆx is not the final generated result in visual autoregressive generation. Therefore, inconsistency between the generator and decoder can lead to unsatisfying results even if the autoregressive model is trained well. For example, when sampling an unseen or rare sequence ˆx in the training dataset of the tokenizer, it’s possible that the sequence ˆx cannot be properly decoded by decoder D and affect the final generated results. We hypothesize that the inconsistency between the tokenizer and generator is the main obstacle to performance. A promising solution is to train the AR model such that it can generate a token sequence that is friendly to the tokenizer.

To verify our hypothesis, we investigate and quantitatively analyze how the existing visual autoregressive model suffers from the inconsistency in Section 3.1. Based on the observations, we propose reAR: regularizing token-wise consistency of visual AutoRegressive generation, a plug-andplay regularized training method designed for a visual autoregressive model. In summary, reAR introduces visual embedding looked up from a discrete tokenizer to the hidden feature of the generator under a noisy context. Despite its simplicity, reAR allows the autoregressive model to leverage visual signals that are compatible with the tokenizer and reduce inconsistent behavior significantly.

# 3.1 UNDERSTANDING THE BOTTLENECK OF VISUAL AUTOREGRESSIVE GENERATION

The performance of an autoregressive model can be assessed through the quality of generated tokens $\hat { x } _ { 1 : n }$ with the ground-truth sequence $x _ { 1 : n }$ by the correct token ratio (CTR), where $\begin{array} { r } { \mathrm { C T R } ( \hat { x } _ { 1 : n } , x _ { 1 : n } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ \hat { x } _ { i } = x _ { i } \} } \end{array}$ . While CTR is widely used to indicate the performance, the token sequence is only an intermediate representation in visual autoregressive generation, and the final output is actually the decoded image. To evaluate end-to-end quality, we instead measure LPIPS (Zhang et al., 2018) between the images decoded from two token sequences. We consider that the inconsistencies between training and inference can be observed from inconsistencies between CTR and LPIPS. In the following, two controlled experiments demonstrate that generated token sequences with similar CTR can result in images of different quality. This inconsistency is also reflected by other metrics for the AR model, such as perplexity, with details in the Appendix B.

Amplified exposure bias. Exposure bias is a well-known issue in sequence models (Bengio et al., 2015; Wang & Sennrich, 2020): during training with teacher forcing, the model predicts the next token given the ground-truth context, whereas at inference it must condition on its own predictions, which may contain errors. In visual autoregressive generation, we hypothesize that the visual tokenizer amplifies this effect since exposure bias leads to more unseen token sequences and spreads structural error in the pixel space. To verify it, consider a token sequence $x _ { 1 : n }$ decoded from an image with a ground-truth token ratio $r \in [ \bar { 0 } , 1 ]$ ]. We compare two decoding protocols: (1) Perfect context (front-loaded). Fix the first $\lfloor r n \rfloor$ tokens to ground truth, $\mathrm { i } . \mathrm { e } . , x _ { 1 : \left\lfloor r n \right\rfloor }$ , and let the AR model generate the remainder. This minimizes exposure bias for a given r, since the context remains clean until step ⌊rn⌋. (2) Imperfect context (uniformly interleaved). Sample a mask $M \subseteq \{ 1 , \dots , n \}$ with $| M | = \bar { \lfloor { r n } \rfloor }$ uniformly at random. During decoding at the tth step, it uses ground truth token $x _ { t }$ if $t \in M$ , otherwise samples the token from the AR model. This introduces earlier contamination of the context, thereby increasing exposure bias compared to Perfect context with similar CTR.

Since both protocols fix the number of ground-truth tokens at $\lfloor r n \rfloor$ , any difference in downstream quality reflects sensitivity to exposure bias rather than token-level accuracy. Results are shown in Figure 3 (a). For comparable CTR, imperfect context consistently yields higher LPIPS than perfect context. Qualitatively, an imperfect context leads to images that deviate significantly from the original, whereas a perfect context yields better prediction, i.e., the layout of the dog is more similar. This highlights that alleviating exposure bias is essential in visual autoregressive generation.

Embedding unawareness. During training, the AR model is optimized only for token correctness, whereas the tokenizer decoder operates in embedding space. We hypothesize that even if a predicted token is incorrect, if its embedding is close to that of the correct token, the decoded image may still retain high visual quality. To verify this, we introduce a replacement ratio $r ^ { \prime }$ . Given a ground-truth sequence $x _ { 1 : n } ,$ , the $\mathbf { A R }$ model predicts $\hat { x } _ { 1 : n }$ with teacher forcing. For each incorrect prediction $( { \hat { x } } _ { i } \neq x _ { i } )$ , we replace ${ \hat { x } } _ { i }$ with probability $r ^ { \prime }$ by another incorrect token $x _ { i } ^ { \prime } \neq x _ { i }$ whose embedding $z ^ { q } { } _ { i } ^ { \prime }$ is closest to the correct embedding $z ^ { q } { } _ { i }$ under cosine similarity $d ( \cdot , \cdot )$ , i.e., $\begin{array} { r } { z ^ { q _ { i } ^ { \prime } } = \arg \operatorname* { m i n } _ { z ^ { q } \in \mathcal { Z } \backslash \{ z ^ { q } { } _ { i } \} } d ( z ^ { q } , z ^ { q } { } _ { i } ) } \end{array}$ . This replacement leaves CTR unchanged.

Figure 3(b) presents the results. As $r ^ { \prime }$ increases, the average embedding similarity improves and LPIPS decreases markedly. Qualitatively, as shown on the right of Figure 3(b), such replacements without altering CTR can yield decoded images more faithful to the ground truth (e.g., clearer prediction of shirts and human legs). This suggests that incorporating tokenizer embeddings into the training of the AR model could potentially improve consistency between them.

A straightforward approach to increase generator-tokenizer inconsistency is to reuse the tokenizer’s codebook embeddings in the embedding layer or prediction head of the AR model. However, this method commonly results in suboptimal performance without sophisticated design of the tokenizer (Weber et al., 2024; Yu et al., 2023). We hypothesize that such a rigid integration is not ideal: it may constrain the scalability of a large AR model with a smaller tokenizer, and the codebook embeddings themselves may not be the optimal representations for the primary task of next-token prediction. It’s required to introduce the embedding into the model in a less constrained manner.

# 3.2 GENERATOR-TOKENIZER CONSISTENCY REGULARIZATION

These findings reveal training–inference inconsistencies: maximizing correctness to predict token indices alone is insufficient for visual AR models. Proper inductive bias is required to train the generator such that the generated token sequence is more consistent with the tokenizer during inference. Meanwhile, injecting this inductive bias should remain lightweight to ensure good cross-architecture generalization and full compatibility with existing AR training and inference pipelines.

To address these challenges, reAR introduces token-wise consistency regularization during training of the visual AR model. Specifically, the decoder-only transformer is trained to perform next-token prediction under noisy contexts, while its hidden representations are regularized by the visual embeddings of the correct current token at a shallow layer and the correct next token at a deep layer. This encourages the AR model to interpret current tokens similar to the tokenizer while improving robustness to exposure bias, then predicting the next token embedding compatible with the decoder. Below we denote the AR model as $p _ { \theta }$ , the tokenizer codebook as $\mathcal { Z } = \mathbf { \bar { \{ z } _ { 1 } , \bar { z } _ { 2 } , \dots , z _ { K } \} }$ , the training dataset as $\chi _ { \mathrm { t r a i n } }$ , and the discrete token sequence as $\mathbf { x } = \{ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { N } \}$ .

Noisy Context Regularization. While techniques such as scheduled sampling (Bengio et al., 2015) can mitigate exposure bias, we choose a simple approach that preserves parallel training of the transformer. Specifically, we apply uniform noise to the input, denoted by $q _ { \epsilon } ( \tilde { \mathbf { x } } \mid \mathbf { x } )$ . Formally:

$$
\tilde {\mathbf {x}} _ {i} = \left(1 - b _ {i}\right) \mathbf {x} _ {i} + b _ {i} \mathbf {u} _ {i}, \quad b _ {i} \sim \text { Bernoulli } (\epsilon), \quad \mathbf {u} _ {i} \sim \text { Uniform } \bigl (\{1, \dots , K \} \bigr). \tag {4}
$$

where $b _ { i }$ is a Bernoulli random variable with probability $\epsilon ,$ and $\mathbf { u } _ { i }$ is sampled uniformly from the codebook indices. In practice, the choice of ϵ strongly affects training stability. To ensure the AR model is exposed to sequences with varying noise levels, we sample $\epsilon \sim U ( 0 , f ( t ) )$ for each token sequence, where $t \in [ \bar { 0 } , 1 ]$ denotes the normalized training progress. Here, $f : [ 0 , 1 ] \to [ 0 , 1 ]$ is an annealing schedule that controls the maximum noise level over training. The AR model is then trained to predict the next correct token based on the noisy context. Formally:

$$
\mathcal {L} _ {\mathrm{AR}} ^ {\prime} (\theta) = - \mathbb {E} _ {\mathbf {x} \in \mathcal {X} _ {\text { train }}, \tilde {\mathbf {x}} \sim q _ {\epsilon} (\cdot | \mathbf {x}), \epsilon \sim U (0, f (t))} \sum_ {i = 1} ^ {N} \log p _ {\theta} (\mathbf {x} _ {i} | \tilde {\mathbf {x}} _ {i - 1}, \dots , \tilde {\mathbf {x}} _ {1}) \tag {5}
$$

Empirically, we found that the annealing uniform noisy augmentation can stabilize training compared to noisy augmentation with a fixed ratio. We provide detailed ablation in Section 4.3.

Codebook Embedding Regularization. Instead of directly applying codebook embedding, we propose to add a regularization task as recover current embedding and predict next embedding. Specifically, we apply a trainable MLP layer $h _ { \phi }$ to project the hidden feature into the target space in the same dimension of visual embedding. For the simplicity of notation, we use $\mathbf { w } _ { \theta } ^ { l } ( \tilde { \mathbf { x } } )$ to represent the feature at the shallow layer l and $\mathbf { w } _ { \theta } ^ { l ^ { ' } } ( \tilde { \mathbf { x } } )$ as the one at the deep layer $l ^ { ' } .$ . To be aligned with the design of decoder-only transformer, the objective of the shallow layer $\mathbf { w } _ { \theta } ^ { l } ( \tilde { \mathbf { x } } )$ is to predict the embedding of current token and $\mathbf { w } _ { \theta } ^ { l ^ { ' } } ( \tilde { \mathbf { x } } )$ is to predict the next token. Formally:

$$
\mathcal{L}_{\mathrm{re}}(\theta ,\phi ;t) = \mathbb{E}_{\substack{\mathbf{x}\sim \mathcal{X}_{\mathrm{train}},\\ \tilde{\mathbf{x}}\sim q_{\epsilon}(\cdot |\mathbf{x}),\\ \epsilon \sim U(0,f(t))}}\sum_{i = 1}^{N - 1}\left[d\big(h_{\phi}^{i}(\mathbf{w}_{\theta}^{l}(\tilde{\mathbf{x}})), z_{\mathbf{x}_{i}}\big) + d\big(h_{\phi}^{i}(\mathbf{w}_{\theta}^{l^{\prime}}(\tilde{\mathbf{x}})),z_{\mathbf{x}_{i + 1}}\big)\right]. \tag{6}
$$

where $d ( \cdot , \cdot )$ is cosine distance to evaluate the distance between different features, $h _ { \phi } ^ { i }$ refers the mapping from the feature of the $i ^ { t h }$ current token to the embedding space, $z _ { \mathbf { x } _ { i } }$ i is the embedding of current token and $z _ { \mathbf { x } _ { i + 1 } }$ is the embedding of the next token looked up from the codebook. In the implementation, we apply the regularization on the layers that are originally most closely to the embedding of the tokenizer in the vanilla AR (i.e, the 1st layer for encoding regularization and the 15th layer for decoding regularization) to avoid potential conflicts on the primary task of next-token prediction. Intuitively, we place the encoding regularization at the first layer to ensure no downstream information required for predicting the next token is suppressed or overwritten, and apply the decoding regularization in a deep but not final layer, since the raw tokenizer embedding is not necessarily the best latent representation for prediction. By default, we regularize at three-quarters of the model depth, which works well across architectures though the exact layer for decoding regularization is flexible. We provide more analysis on the codebook embedding regularization in Section 4.3 and Appendix C.2.

Generator-Tokenizer Consistency Regularization. Combing Noisy Context Regularization and Codebook Embedding Regularization, the object of reAR is:

$$
\mathcal {L} _ {\mathrm{reAR}} (\theta , \phi ; t) = \mathcal {L} _ {\mathrm{AR}} ^ {\prime} (\theta ; t) + \lambda \mathcal {L} _ {\mathrm{re}} (\theta , \phi ; t), \tag {7}
$$

where λ is the weight of the regularization term. Notice that we align the hidden feature of noisy tokens to the embedding of the ground truth token as well, which further encourages the autoregressive model to predict codebook embedding in a robust manner. This joint effect is important to boost the performance of visual autoregressive generation. We provide detailed ablation in Section 4.3.

Table 1: Results on 256×256 class-conditional generation on ImageNet-1K. “Mask.” indicates masked generation; “Tok.” denotes non-standard tokenization; “Rand.” denotes randomized order; “Raster.” denotes rasterization order. “†” indicates that the model is not provided and it’s trained with our implementation. $\mathrm { B P P _ { 1 6 } } = 1 6 \times \mathrm { B P P }$ (bits per pixel) measures the compression rate of discrete tokenizers and is not applicable (“N/A”) to continuous tokenizers. “#Params” is the number of model parameters. “↑” and “↓” indicate whether higher or lower values are better, respectively. 

<table><tr><td>Training Paradigm</td><td>Generation Model</td><td>Tokenizer Type</td><td>Tokenizer  $BPP_{16} \downarrow$ </td><td>Training Epochs</td><td>#Params.↓</td><td>FID↓</td><td>IS↑</td></tr><tr><td rowspan="4">Diffusion</td><td>LDM-4 (Rombach et al., 2022)</td><td>Patch-VAE</td><td>N/A</td><td>200</td><td>400M</td><td>3.60</td><td>247.7</td></tr><tr><td>DiT-XL (Peebles &amp; Xie, 2023)</td><td>Patch-VAE</td><td>N/A</td><td>1400</td><td>675M</td><td>2.27</td><td>278.2</td></tr><tr><td>SiT-XL (Ma et al., 2024)</td><td>Patch-VAE</td><td>N/A</td><td>800</td><td>675M</td><td>2.06</td><td>270.3</td></tr><tr><td>REPA (Yu et al., 2024c)</td><td>Patch-VAE</td><td>N/A</td><td>800</td><td>675M</td><td>1.42</td><td>305.7</td></tr><tr><td rowspan="2">MAR</td><td>MAR-L (Li et al., 2024)</td><td>Patch-VAE</td><td>N/A</td><td>800</td><td>479M</td><td>1.98</td><td>290.3</td></tr><tr><td>MAR-H (Li et al., 2024)</td><td>Patch-VAE</td><td>N/A</td><td>800</td><td>943M</td><td>1.55</td><td>303.7</td></tr><tr><td rowspan="5">Mask.</td><td>MaskGIT-re Chang et al. (2022)</td><td>Patch-VQ</td><td>0.625</td><td>300</td><td>227M</td><td>4.02</td><td>355.6</td></tr><tr><td>MAGVIT-v2 (Yu et al., 2023)</td><td>Patch-VQ</td><td>1.125</td><td>1080</td><td>307M</td><td>1.78</td><td>319.4</td></tr><tr><td>Maskbit (Weber et al., 2024)</td><td>Patch-LFQ</td><td>0.875</td><td>1080</td><td>305M</td><td>1.52</td><td>328.6</td></tr><tr><td>Mask-TiTok-64 (Yu et al., 2024b)</td><td>TiTok</td><td>0.188</td><td>800</td><td>177M</td><td>2.48</td><td>214.7</td></tr><tr><td>Mask-TiTok-128 (Yu et al., 2024b)</td><td>TiTok</td><td>0.375</td><td>800</td><td>287M</td><td>1.97</td><td>281.8</td></tr><tr><td rowspan="2">VAR</td><td>VAR-d20 (Tian et al., 2024)</td><td>VAR</td><td>1.992</td><td>350</td><td>600M</td><td>2.57</td><td>302.6</td></tr><tr><td>VAR-d30 (Tian et al., 2024)</td><td>VAR</td><td>1.992</td><td>350</td><td>2.0B</td><td>1.92</td><td>323.1</td></tr><tr><td rowspan="6">Rand.CausalAR</td><td>RAR-B (Yu et al., 2024a)</td><td>Patch-VQ</td><td>0.625</td><td>400</td><td>261M</td><td>1.95</td><td>290.5</td></tr><tr><td>RAR-L (Yu et al., 2024a)</td><td>Patch-VQ</td><td>0.625</td><td>400</td><td>461M</td><td>1.70</td><td>299.5</td></tr><tr><td>RAR-XL (Yu et al., 2024a)</td><td>Patch-VQ</td><td>0.625</td><td>400</td><td>955M</td><td>1.50</td><td>306.9</td></tr><tr><td>RandAR-L (Pang et al., 2025)</td><td>Patch-VQ</td><td>0.875</td><td>300</td><td>343M</td><td>2.55</td><td>288.8</td></tr><tr><td>RandAR-XL (Pang et al., 2025)</td><td>Patch-VQ</td><td>0.875</td><td>300</td><td>775M</td><td>2.25</td><td>317.8</td></tr><tr><td>RandAR-XXL (Pang et al., 2025)</td><td>Patch-VQ</td><td>0.875</td><td>300</td><td>1.4B</td><td>2.15</td><td>322.0</td></tr><tr><td rowspan="3">Tok.CausalAR</td><td>AR-FlexTok-XL (Bachmann et al., 2025)</td><td>FlexTok</td><td>0.125</td><td>300</td><td>1.3B</td><td>2.02</td><td>-</td></tr><tr><td>AR-GigaTok-XXL (Xiong et al., 2025)</td><td>GigaTok</td><td>0.875</td><td>300</td><td>1.4B</td><td>1.98</td><td>256.8</td></tr><tr><td>AR-WeTok-XL (Zhuang et al., 2025)</td><td>WeTok</td><td>1.667</td><td>300</td><td>1.5B</td><td>2.31</td><td>276.6</td></tr><tr><td rowspan="8">Raster.CausalAR</td><td>VQGAN-re (Esser et al., 2021)</td><td>Patch-VQ</td><td>0.875</td><td>100</td><td>1.4B</td><td>5.20</td><td>280.3</td></tr><tr><td>Open-MAGVIT-v2 (Luo et al., 2024)</td><td>Patch-LFQ</td><td>1.125</td><td>300</td><td>1.5B</td><td>2.33</td><td>271.8</td></tr><tr><td>LlamaGen-XL (Sun et al., 2024)</td><td>Patch-VQ</td><td>0.875</td><td>300</td><td>775M</td><td>2.62</td><td>244.1</td></tr><tr><td>LlamaGen-XXL (Sun et al., 2024)</td><td>Patch-VQ</td><td>0.875</td><td>300</td><td>1.4B</td><td>2.34</td><td>253.9</td></tr><tr><td>AR-L $^†$ (Yu et al., 2024a)</td><td>Patch-VQ</td><td>0.625</td><td>400</td><td>461M</td><td>3.02</td><td>256.2</td></tr><tr><td>reAR-S</td><td>Patch-VQ</td><td>0.625</td><td>400</td><td>201M</td><td>2.00</td><td>295.7</td></tr><tr><td>reAR-B</td><td>Patch-VQ</td><td>0.625</td><td>400</td><td>261M</td><td>1.91</td><td>300.9</td></tr><tr><td>reAR-L (cfg=10.0/11.0)</td><td>Patch-VQ</td><td>0.625</td><td>400</td><td>461M</td><td>1.86/1.90</td><td>316.9/323.2</td></tr></table>

![](images/e67a5c441ca396eea44d6804286f9ea46f0a85d75031cc4b22675fafb90b7179.jpg)

<details>
<summary>line</summary>

| Training Steps (K) | reAR-S | reAR-B | reAR-L |
| ------------------ | ------ | ------ | ------ |
| 50                 | 3.8    | 3.6    | 3.4    |
| 100                | 2.7    | 2.5    | 2.2    |
| 150                | 2.2    | 2.1    | 1.9    |
| 200                | 2.1    | 2.0    | 1.9    |
| 250                | 2.0    | 1.9    | 1.8    |
</details>

Figure 4: Scaling Effect of reAR. As model size increases, the FID at each training step decreases consistently.

Table 2: Superior generalization ability. reAR adapts to different tokenizers and achieves state-of-the-art performance with smaller models. 

<table><tr><td>Model</td><td>Epochs</td><td>Params.</td><td>FID ↓</td></tr><tr><td>Maskbit (Weber et al., 2024)</td><td>1080</td><td>305M</td><td>1.52</td></tr><tr><td>REPA (Yu et al., 2024c)</td><td>800</td><td>675M</td><td>1.42</td></tr><tr><td>AR-TiTok-b64 (Yu et al., 2024b)</td><td>400</td><td>261M</td><td>4.45</td></tr><tr><td>RAR-TiTok-b64 (Yu et al., 2024a)</td><td>400</td><td>261M</td><td>4.07</td></tr><tr><td>reAR-TiTok-b64</td><td>400</td><td>261M</td><td>4.01</td></tr><tr><td>AR-AliTok-B (Wu et al., 2025)</td><td>800</td><td>177M</td><td>1.50</td></tr><tr><td>RAR-B-AliTok (Yu et al., 2024a)</td><td>800</td><td>177M</td><td>1.52</td></tr><tr><td>reAR-B-AliTok</td><td>800</td><td>177M</td><td>1.42</td></tr></table>

# 4 EXPERIMENTS & ANALYSIS

# 4.1 EXPERIMENTAL SETUP

Below we provide a brief of our experimental setup, and more details are in Appendix A.

Dataset and evaluation. We evaluate reAR on ImageNet-1K at 256×256 using the ADM protocol (Dhariwal & Nichol, 2021). Each model generates 50k images with classifier-free guidance (Ho & Salimans, 2022). We report FID (lower is better) (Heusel et al., 2017) and IS (higher is better) (Salimans et al., 2016), and compare training efficiency by epochs and parameters needed to reach the same quality. Baselines span diffusion, masked generation (continuous and discrete), VAR, randomized-order AR, advanced-tokenizer AR, and standard raster AR (see Table 1).

Model configuration. We use MaskGIT VQGAN (Chang et al., 2022) (rFID= 1.97) as a tokenizer and a DiT-style (Peebles & Xie, 2023) AR backbone. We report reAR-S/B/L with 20/24/24 causal Transformer layers and hidden sizes 768/768/1024. To evaluate the generalization of reAR, we also pair it with TiTok (Yu et al., 2024b) and with AliTok (Wu et al., 2025) using their original setting. Additionally, we also verify the effectiveness of our method on non-standard causal AR model such as VAR (Tian et al., 2024) with more details in the Appendix A.

Training. All models are trained for 400 epochs on 8 A800 GPUs (batch size 2048) with AdamW (Loshchilov & Hutter, 2017), gradient clipping (norm= 1), and accumulation. The learning rate warms $\tan 4 \times 1 0 ^ { - 4 }$ over the first 100 epochs, then decays to $1 \times 1 0 ^ { - 5 }$ for the remaining 300 epochs. Class labels are dropped with probability 0.1 to enable classifier-free guidance at inference.

reAR implementation. We apply a linear schedule for annealing noise augmentation. Embedding regularization is implemented using a 2-layer MLP (hidden size 2048, weight λ=1): the shallow layer regularizes the current embeddings at l=0, while the deeper layer regularizes the decoding features at $\textstyle { \frac { 3 } { 4 } }$ depth of the whole transformer $( l ^ { \prime } = 1 5 / 1 8 / 1 8$ for reAR-S/B/L).

# 4.2 MAIN RESULTS

Generation Quality. Table 1 shows that reAR achieves strong results even with a standard rasterorder AR model and a simple 2D patch tokenizer. reAR-S outperforms prior raster AR models like LlamaGen-XL (Sun et al., 2024) (FID 2.00 vs. 2.34; IS 295.7 vs. 253.9) using only 14% of the parameters (201M vs. 1.4B), and surpasses advanced-tokenizer AR models such as WeTok (Zhuang et al., 2025) with just 13–15% of their size. It matches RAR (Yu et al., 2024a) and outperforms RandAR (Pang et al., 2025) under similar scales, and reAR-L exceeds MAR-L and VAR-d30 (Li et al., 2024; Tian et al., 2024). While diffusion and masked-generation models remain strong, reAR narrows the gap with far fewer training epochs. More qualitative results are shown in Appendix F.

Generalization. We also evaluate reAR on non-standard tokenizers TiTok (Yu et al., 2024b) and AliTok (Wu et al., 2025). Unlike RAR (Yu et al., 2024a), which helps mainly on bidirectional tokenization, reAR consistently improves performance on both bidirectional (TiTok: 4.45 → 4.01) and unidirectional (AliTok: 1.50 → 1.42) tokenizers. Notably, it approaches diffusion-based REPA (Yu et al., 2024c) and outperforms Maskbit while using far fewer parameters (177M vs. 675M/305M).

Scaling Effect. We also study if the scaling behavior of the original AR model maintains with reAR. Specifically, we plot the FID under different training epochs for each model size. As Figure 4 shows, the FID consistently decreases as model size and training iteration increase, revealing the potential of reAR on large-scale visual AR models.

Sampling Speed. Like other autoregressive models (Sun et al., 2024; Luo et al., 2024), reAR benefits from KV-cache to achieve high sampling speed. We measure throughput on a single A800 GPU with batch size 128 (Figure 5). With KV-cache, autoregressive models can run much faster than diffusion

![](images/0d576e40407e0c1cde01932ba1c7e89f90e9ce926bcfb9e7a3c0e15d19227015.jpg)

<details>
<summary>scatter</summary>

| Model           | Throughput (Images/s) | FID (lower is better) |
| --------------- | --------------------- | --------------------- |
| MAR-B           | ~1                    | ~2.5                  |
| DIFXL           | ~1                    | ~2.3                  |
| MaskBit         | ~1                    | ~1.5                  |
| Titok-S-128     | ~7                    | ~2.0                  |
| RandAR-XL       | ~16                   | ~2.3                  |
| reAR-S-VQGAN    | ~17                   | ~2.0                  |
| VAR-d30         | ~17                   | ~2.0                  |
| reAR-B-VQGAN    | ~14                   | ~1.9                  |
| reAR-B-AllTok   | ~19                   | ~1.4                  |
</details>

Figure 5: Sampling Speed. Comparison of different methods on FID and throughput (images/sec).

and MAR. Moreover, reAR-B-AliTok achieves lower FID with faster sampling speed even against parallel-decoding approaches such as Maskbit, TiTok, VAR, and RandAR.

# 4.3 ABLATION STUDIES

We conduct ablation studies on the key components of reAR, focusing on the weighting and layer selection for encoding/decoding regularization, as well as the strategy for noise augmentation.

Regularization Layer. We analyzed the optimal layers for embedding regularization using reAR-S trained for 80 epochs without classifier-free guidance (Table 4). We ablated both the presence and placement of regularization and compared with the naive tied embedding strategy (Press & Wolf, 2016; Weber et al., 2024). For decoding regularization, early layers (e.g., layer 10) offer little benefit, while layer 15 performs best; applying it deeper slightly degrades performance. For encoding regularization, the first layer is optimal as it aligns best with the token embeddings, whereas deeper layers harm generation quality. Notably, applying regularization to the layers closest to the target embedding space in vanilla AR yields the best results—encoding at layer 0 and decoding at roughly 34 depth. We hypothesize this placement minimizes interference with next-token prediction. Based on these findings, we use EN@0 + DE@15 for reAR-S and EN@0 + DE@18 for reAR-B/L. We provide a more detailed comparison of different choices of the decoding regularization layer in Appendix C.2.

Table 3: Ablation studies of noisy context regularization with annealing. 

<table><tr><td>Noise Augmentation settings</td><td>FID ↓</td></tr><tr><td> $\epsilon = 0.0$ </td><td>2.12</td></tr><tr><td> $\epsilon = 0.5$ </td><td>3.15</td></tr><tr><td> $\epsilon = 0.25$ </td><td>2.08</td></tr><tr><td> $\epsilon \sim U(0, 0.5)$ </td><td>2.05</td></tr><tr><td> $\epsilon \sim U(0, f(t)), \quad f(t) = 1 - t$ </td><td>2.02</td></tr><tr><td> $\epsilon \sim U(0, f(t)), \quad f(t) = \min(0, 1 - \frac{4}{3}t)$ wo/ embedding regularization</td><td>2.002.18</td></tr></table>

Regularization Weight. As shown in Table 4, regularization weight has a negligible impact on the quality of generation, likely because the AdamW optimizer is insensitive to the scale of the loss (Loshchilov & Hutter, 2017; Zhuang et al., 2022). For simplicity, we use λ = 1.

Noise Augmentation. We further ablate the design of noise augmentation, exploring two strategies: (1) assigning different noise levels to each token sequence, and (2) annealing the maximum noise level during training. Results are summarized in Table 3, based on the default setting with codebook embedding regularization (EN@0 + DE@15 for reAR-S). All models are trained for 400 epochs to evaluate the effect of different schedules. We find that a fixed noise level of ϵ = 0.25 improves FID from 2.12 to 2.08, while a higher level $( \epsilon = 0 . 5 )$ leads to training collapse (FID = 3.15). Randomizing the noise level within [0, 0.5] further improves FID to 2.05. Incorporating an annealing schedule, where $f ( t ) = 1 - t ,$ yields a stronger result (2.02 FID). Finally, using a truncated linear schedule $\begin{array} { r } { f ( t ) = \operatorname* { m a x } ( 0 , 1 - \frac { 4 } { 3 } t ) } \end{array}$ achieves the best performance of 2.00 FID. These results highlight the effectiveness of proper annealing noise augmentation.

Table 4: Ablation studies of embedding regularization. We use ‘EN’ as the encoding regularization and ‘DN’ as the decoding regularization. For example, ‘DN@15’ means applying decoding regularization at the 15th layer of the transformer block. 

<table><tr><td>Regularization settings</td><td>FID ↓</td><td>IS ↑</td></tr><tr><td>Vanilla AR</td><td>21.32</td><td>57.3</td></tr><tr><td>+ tied codebook embedding</td><td>21.08</td><td>57.2</td></tr><tr><td>+ DE@10</td><td>21.29</td><td>57.5</td></tr><tr><td>+ DE@15</td><td>20.03</td><td>61.0</td></tr><tr><td>+ DE@20</td><td>20.28</td><td>61.2</td></tr><tr><td>+ EN@0 + DE@20</td><td>19.83</td><td>61.7</td></tr><tr><td>+ EN@5 + DE@15</td><td>21.36</td><td>57.4</td></tr><tr><td>+ EN@0 + DE@15 (Final choice)</td><td>19.72</td><td>61.3</td></tr><tr><td> $\lambda := 0.5$ </td><td>19.79</td><td>60.9</td></tr><tr><td> $\lambda := 1.5$ </td><td>19.74</td><td>61.5</td></tr></table>

Joint Effect of Consistency Regularization. As shown in Table 3, using only embedding regularization (ϵ=0) yields an FID of 2.12, while using only noise augmentation yields 2.18. In contrast, combining the two further improves performance, reducing the FID of reAR-S to 2.00. This indicates that both noisy context regularization and codebook embedding regularization are important.

# 5 RELATED WORK

Visual AR models generate images by predicting pixels or patch tokens sequentially, each conditioned on previous context (Gregor et al., 2014; Van den Oord et al., 2016; Van Den Oord et al., 2016; Parmar et al., 2018; Chen et al., 2020). In this paper, we refer specifically to the visual AR model as the family using a unidirectional structure. Direct pixel-level modeling is expensive, so patch-based tokenizers (Van Den Oord et al., 2017; Esser et al., 2021) are used to compress local regions into discrete tokens. An AR model then predicts the token sequence (Esser et al., 2021; Sun et al., 2024; Luo et al., 2024). Prior work has focused on modular design, such as reducing quantization errors (Yu et al., 2023; Mentzer et al., 2023; Ma et al., 2025; Li et al., 2024) or exploring tokenization beyond standard 2D grids (Yu et al., 2024b; Miwa et al., 2025; Sargent et al., 2025; Xiong et al., 2025). Others have studied sequence dependencies, imposing causality during tokenizer training (Wu et al., 2025; Bachmann et al., 2025; Pan et al., 2025) or randomizing token order (Pang et al., 2025; Yu et al., 2024a). While these works focus on the flaw of a single component, we provide a novel perspective on the inconsistency between the AR model and the tokenizer.

Other visual generation paradigm has advanced from Variational Autoencoders (VAEs) (Kingma & Welling, 2013) and Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) to modern approaches such as masked generative models (Chang et al., 2022; Yu et al., 2023; Weber et al., 2024) and diffusion-based models (Dhariwal & Nichol, 2021; Peebles & Xie, 2023; Ma et al., 2024; Yu et al., 2024c), apart from AR model. Recently, MAR (Li et al., 2024) was proposed to address quantization errors, and VAR (Tian et al., 2024) for next-scale prediction. However, they are not implemented in a decoder-only transformer, making them harder to incorporate with the standard AR used in large language models. We provide more discussion in Appendix D.

Exposure bias has been extensively studied in the language domain, with methods such as scheduled sampling (Bengio et al., 2015). In the visual domain, RQ-Transformer (Lee et al., 2022) applies scheduled sampling, and IQ-VAE (Zhan et al., 2022) uses Gumbel-softmax to mix ground-truth and predicted tokens, though both approaches compromise the parallel training efficiency of decoderonly Transformers. More recently, video generation works have addressed exposure bias in autoregressive diffusion models (Zhou et al., 2025; Huang et al., 2025), but these strategies are not applicable to discrete token prediction.

Representation Alignment. Representation alignment has been explored in visual generation (Yu et al., 2024c; Leng et al., 2025; Yao et al., 2025; Xiong et al., 2025). For example, REPA (Yu et al., 2024c) incorporates DINO-v2 features to accelerate diffusion training, and Disperse Loss (Wang & He, 2025) applies self-supervised objectives to improve diffusion representations. However, these methods are either designed for encoder-only Transformers and diffusion models or often rely on external visual encoders. In contrast, we aim to align the representations of the tokenizer and the AR model itself, requiring no external models and fitting naturally into the vanilla AR training pipeline.

# 6 CONCLUSION

In this paper, we identify the key bottleneck of visual autoregressive generation as the mismatch between the generator and the tokenizer, where the AR model struggles to produce token sequences that can be effectively decoded back into images. To address this, we propose reAR, a simple regularization method that substantially improves visual AR performance while remaining agnostic to tokenizer design. We hope this work will encourage future research on unifying generators and tokenizers within visual AR models, and more broadly, on developing unified multi-modal models.

# ACKNOWLEDGMENT

We would like to acknowledge that computational work involved in this research is partially supported by NUS IT’s Research Computing group using grant number NUSREC-HPC-00001.

# ETHICS STATEMENT

Our work introduces a regularization strategy to improve visual autoregressive models and contributes toward the broader goal of unifying vision and language generation. While these advances can benefit research on unified multimodal models, we acknowledge the potential risks associated with generative technologies. In particular, improvements in fidelity and scalability may also lower the barrier for misuse, such as the creation of misleading or harmful synthetic media.

# REPRODUCIBILITY STATEMENT

We include all experiment details sufficient for reproducibility in Section. 4, Appendix A, Appendix B, and Appendix C. We provide the anonymous code here and will release the code once the paper is accepted.

# REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, Oguzhan Fatih Kar, Elmira Amir-˘ loo, Alaaeldin El-Nouby, Amir Zamir, and Afshin Dehghan. Flextok: Resampling images into 1d token sequences of flexible length. In Forty-second International Conference on Machine Learning, 2025.   
Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan L Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22861–22872, 2024.   
Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.   
Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11315–11325, 2022.   
Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pp. 1691– 1703. PMLR, 2020.   
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.   
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.   
Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion, 2022. URL https://arxiv.org/abs/2208.01618.   
Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.   
Karol Gregor, Ivo Danihelka, Andriy Mnih, Charles Blundell, and Daan Wierstra. Deep autoregressive networks. In International Conference on Machine Learning, pp. 1242–1250. PMLR, 2014.   
Qiyuan He and Angela Yao. Conceptrol: Concept control of zero-shot personalized image generation, 2025. URL https://arxiv.org/abs/2503.06568.   
Qiyuan He, Jinghao Wang, Ziwei Liu, and Angela Yao. Aid: Attention interpolation of text-to-image diffusion, 2024. URL https://arxiv.org/abs/2403.17924.   
Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control, 2022. URL https://arxiv. org/abs/2208.01626.   
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.   
Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.   
Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. URL https://arxiv.org/abs/2506.08009. Trains with AR rollout and KV-cached self-generated history to directly tackle exposure bias.   
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pp. 3519– 3529. PMlR, 2019.   
Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11523–11532, 2022.   
Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025.   
Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37: 56424–56445, 2024.   
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.   
Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pp. 23–40. Springer, 2024.   
Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR. Open-Review.net, 2022. URL http://dblp.uni-trier.de/db/conf/iclr/iclr2022. html#MengHSSWZE22.   
Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023.   
Keita Miwa, Kento Sasaki, Hidehisa Arai, Tsubasa Takahashi, and Yu Yamaguchi. One-d-piece: Image tokenizer meets quality-controllable compression. arXiv preprint arXiv:2501.10064, 2025.   
Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 16784–16804. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/ v162/nichol22a.html.   
Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, ´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.   
Kaihang Pan, Wang Lin, Zhongqi Yue, Tenglong Ao, Liyu Jia, Wei Zhao, Juncheng Li, Siliang Tang, and Hanwang Zhang. Generative multimodal pretraining with discrete diffusion timestep tokens. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26136–26146, 2025.   
Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T Freeman, and Yu-Xiong Wang. Randar: Decoder-only autoregressive visual generation in random orders. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 45–55, 2025.   
Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International conference on machine learning, pp. 4055– 4064. PMLR, 2018.   
William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.   
Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.   
Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation, 2023. URL https://arxiv.org/abs/2208.12242.   
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.   
Kyle Sargent, Kyle Hsu, Justin Johnson, Li Fei-Fei, and Jiajun Wu. Flow to the mode: Mode-seeking diffusion autoencoders for state-of-the-art image tokenization. arXiv preprint arXiv:2503.11056, 2025.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.   
Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer, 2025. URL https://arxiv.org/abs/ 2411.15098.   
Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.   
Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.   
Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016.   
Aaron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks.¨ In International conference on machine learning, pp. 1747–1756. PMLR, 2016.   
Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.   
Chaojun Wang and Rico Sennrich. On exposure bias, hallucination and domain shift in neural machine translation. arXiv preprint arXiv:2005.03642, 2020.   
Runqian Wang and Kaiming He. Diffuse and disperse: Image generation with representation regularization. arXiv preprint arXiv:2506.09027, 2025.   
Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. arXiv preprint arXiv:2409.16211, 2024.   
Pingyu Wu, Kai Zhu, Yu Liu, Longxiang Tang, Jian Yang, Yansong Peng, Wei Zhai, Yang Cao, and Zheng-Jun Zha. Alitok: Towards sequence modeling alignment between tokenizer and autoregressive model. arXiv preprint arXiv:2506.05289, 2025.   
Tianwei Xiong, Jun Hao Liew, Zilong Huang, Jiashi Feng, and Xihui Liu. Gigatok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation. arXiv preprint arXiv:2504.08736, 2025.   
Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15703–15712, 2025.   
Lijun Yu, Jose Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong ´ Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion– tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.   
Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-Chieh Chen. Randomized autoregressive visual generation. arXiv preprint arXiv:2411.00776, 2024a.   
Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems, 37:128940–128966, 2024b.   
Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024c.

Fangneng Zhan, Yingchen Yu, Rongliang Wu, Jiahui Zhang, Kaiwen Cui, Changgong Zhang, and Shijian Lu. Auto-regressive image synthesis with integrated quantization. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. URL https://www.ecva. net/papers/eccv\_2022/papers\_ECCV/papers/136760106.pdf. Uses Gumbel sampling with reliability-based scheduling to mitigate exposure bias in AR training.   
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.   
Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023.   
Deyu Zhou, Quan Sun, Yuang Peng, Kun Yan, Runpei Dong, Duomin Wang, Zheng Ge, Nan Duan, and Xiangyu Zhang. Taming teacher forcing for masked autoregressive video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7374–7384, 2025.   
Shaobin Zhuang, Yiwei Guo, Canmiao Fu, Zhipeng Huang, Zeyue Tian, Ying Zhang, Chen Li, and Yali Wang. Wetok: Powerful discrete tokenization for high-fidelity visual reconstruction. arXiv preprint arXiv:2508.05599, 2025.   
Zhenxun Zhuang, Mingrui Liu, Ashok Cutkosky, and Francesco Orabona. Understanding adamw through proximal methods and scale-freeness. arXiv preprint arXiv:2202.00089, 2022.

# A ADDITIONAL EXPERIMENTAL DETAILS

Dataset and Evaluation Protocol. For ImageNet evaluation, we follow the ADM protocol (Dhariwal & Nichol, 2021). Specifically, we compute both FID and IS using the ImageNet-1K validation split (50,000 images), and we generate 50,000 synthetic images with our model. We then compute FID between the generated set and the real validation set. During sampling, for classifier-free guidance, we adopt a power-cosine schedule as used in prior work (Zheng et al., 2023). For our reAR-S/B/L models, we set the guidance scale to 22, 14.5, and 10, respectively, and corresponding power scales to 2.75, 2.25, and 1.75. Across all models, we keep the temperature at 1.0 and do not use top-p or top-k sampling, so that improvements reflect model quality rather than sampling tricks. All the images generated and evaluated are fixed at the resolution of 256 × 256.

Comparing methods. We divide the visual generation into seven classes in Table 1: Diffusion model, MAR (continuous masked generation), Mask. (discrete masked generation), VAR (next scale prediction with encoder-only transformer), Rand. Causal AR (introduce randomized order of token sequence), Tok. Causal AR (use an advanced tokenizer that is not rasterization order), Raster. Causal AR (the most standard visual AR based on patch tokens and rasterization order).

Model Configuration. We use the same VQGAN tokenizer from MaskGIT (Chang et al., 2022), a pure CNN that produces feature maps which are patchified into 16 × 16 patches and quantized via a codebook of size 1024. For the autoregressive backbone, we follow the visual transformer (ViT)- based architecture of RAR (Yu et al., 2024a) and DiT (Peebles & Xie, 2023), further inserting class conditioning via AdaLN layers as in DiT. To ensure fair comparison with RAR, we use learnable positional embeddings throughout. We apply dropout with probability 0.1 both in the feed-forward network and in attention layers. Additionally, the MLP ratio is kept as 4.0 in the feed-forward network, and the number of attention heads is fixed to 16 for all different settings. We also include QK-Norm in attention to enhance stability.

Training details. As we mentioned in Section 4.1, all models are trained for 400 epochs with a batch size of 2048 on a single node of 8 A800 GPUs. For reAR-S and reAR-B, we use gradient accumulation as 1, and for reAR-L, we use gradient accumulation over 2 steps with a batch size of 1024 to achieve the same effective batch size. Following prior work (Yu et al., 2024a), we linearly warm up the learning rate to $4 \times 1 0 ^ { - 4 }$ over the first 100 epochs and apply a cosine decay schedule to decrease the learning rate to $1 \times 1 0 ^ { - 5 }$ for the remaining 300 epochs. We use AdamW as the optimizer with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 6$ , and weight decay of 0.03. The gradient clipping is applied with a maximum gradient norm of 1.0. We use mixed precision with bfloat16 to accelerate training.

Table 5: Comparison of computation cost. 

<table><tr><td>Method</td><td>Time / Epoch (min)</td><td>FID</td></tr><tr><td>AR-B</td><td>8.11</td><td>3.12</td></tr><tr><td>reAR-B</td><td>8.14</td><td>1.91</td></tr><tr><td>AR-L</td><td>15.99</td><td>3.02</td></tr><tr><td>reAR-L</td><td>16.05</td><td>1.86</td></tr></table>

Implementation details of reAR. Regarding noisy context regularization, the noise ratio is sampled from a range that is determined by the training procedure. Specifically, the noise ratio is sampled from (0, f (t)), where $\begin{array} { r } { f ( t ) = \operatorname* { m i n } ( 0 , 1 - \frac { 4 } { 3 } t ) } \end{array}$ and t refers to the normalized training progress. For example,

Table 6: Evaluation of reAR on VAR 

<table><tr><td>Method</td><td>FID</td><td>IS</td></tr><tr><td>VAR-d16</td><td>3.55</td><td>274.4</td></tr><tr><td>VAR-d16 (retrained w/ reAR)</td><td>3.39</td><td>276.6</td></tr></table>

the noise ratio is sampled from (0, 1 ) at the 150 epoch where $\textstyle t = { \frac { 3 } { 8 } }$ over total 400 epochs. Regarding codebook embedding regularization, the 2-layer MLP with hidden size as 2048 is equipped with GeLU and maps the generator feature into the dimension of the corresponding codebook embedding. The parameter overhead of the MLP is 2.1M/2.1M/4.2M for reAR-S/B/L. Table 5 shows the training time cost on 8 A800 GPUs. This light-weight design only brings minimal training overhead while achieving superior performance with the same inference cost.

Experiments on VAR (Tian et al., 2024). VAR differs from standard autoregressive models such as VQGAN, TiTok, and AliTok, as it predicts the next scale or resolution and outputs multiple discrete tokens simultaneously rather than using a decoder-only transformer to predict a single next token. These differences lead to training and inference behaviors that diverge from standard AR, and we provide more details in Appendix D. Nevertheless, because VAR still generates discrete tokens autoregressively, it may also benefit from reAR. To test this, we apply reAR to VAR-d16 using the same training settings as in our main experiments and same inference settings as in the original VAR paper. As shown in Table 6, reAR improves performance without tuning any training or inference hyperparameters, demonstrating its generalization ability.

![](images/df40e8994449ceb44c9701435acebfd1fc975f30439e397850c01f6b8131bfbf.jpg)

<details>
<summary>natural_image</summary>

Collage of six natural and symbolic images: volcanic eruption, lighthouse, eagle, fountain, building, and water feature (no text or symbols)
</details>

Figure 6: Qualitative Results of VAR-d16 retrained with reAR.

Ground-truth   
Replace Ratio: 0%   
Replace Ratio: 20%   
Replace Ratio: 40%   
Replace Ratio: 60%   
![](images/623e33c3dbdf226a38e9aec8b069c877864e316a2d1fc307322fef6ece19a67e.jpg)

<details>
<summary>natural_image</summary>

Sequence of photos showing a large dog's head and its butterfly feeding on a flower (no text or symbols)
</details>

Figure 7: Visualization of analysis experiment on replacing tokens with more similar embedding.

# B ANALYSIS DETAILS ON GENERATOR-TOKENIZER INCONSISTENCY

In this section, we present more details on the analysis experiments introduced in Section 3.1 on the generator-tokenizer inconsistency, including (i) evaluation metric (Section B.1), (ii) experiment settings (Section B.2), and (iii) Findings (Section B.3).

# B.1 EVALUATION METRIC FOR STUDYING INCONSISTENCY

We provide additional results on the quantitative evaluation of the inconsistency between token sequences $\mathbf { x } _ { 1 : N }$ and the corresponding decoded images ˆI. We adopt two groups of metrics: (i) for token sequence quality, we use the correct token ratio (CTR) and perplexity, and (ii) for image quality, we use PSNR and LPIPS. Here, the LPIPS and PSNR are different from those in the reconstruction task, since the decoded image is obtained from the generated token sequence under teacher forcing. While this is not a direct evaluation of the generation quality, it serves as an intermediate proxy similar to the correct token ratio and perplexity, but in pixel space.

Evaluation on token sequence. CTR measures the fraction of correctly predicted tokens under teacher forcing, while perplexity reflects the uncertainty of the predicted token distribution. Formally, given ground-truth sequence $\mathbf { x } _ { 1 : N }$ and autoregressive model $p _ { \theta }$ , we define

$$
\mathrm{CTR} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbf {1} \left[ \arg \max _ {v} p _ {\theta} (v \mid \mathbf {x} _ {1: i - 1}) = x _ {i} \right], \tag {8}
$$

$$
\text { Perplexity } = \exp \left(- \frac {1}{N} \sum_ {i = 1} ^ {N} \log p _ {\theta} (x _ {i} \mid \mathbf {x} _ {1: i - 1})\right). \tag {9}
$$

![](images/6703f1ac9ce38e9911ef73a50b4357f1e71b8727aad5c56269b7a45375c29bf0.jpg)

<details>
<summary>line</summary>

| CTR → (higher is better) | Imperfect Context | Perfect Context |
| ------------------------ | ----------------- | --------------- |
| 0.2                      | 0.5               | 0.4             |
| 0.4                      | 0.35              | 0.25            |
| 0.6                      | 0.25              | 0.15            |
| 0.8                      | 0.15              | 0.08            |
</details>

(a)

![](images/7bcee8294b340afe8842560043a10402eb1b2d4c908871659427a3de95b98201.jpg)

<details>
<summary>line</summary>

| CTR (higher is better) | Imperfect Context | Perfect Context |
| ---------------------- | ----------------- | --------------- |
| 0.2                    | 10.0              | 10.5            |
| 0.3                    | 11.0              | 11.5            |
| 0.4                    | 12.0              | 12.5            |
| 0.5                    | 13.0              | 13.5            |
| 0.6                    | 14.0              | 14.5            |
| 0.7                    | 15.0              | 15.5            |
| 0.8                    | 16.0              | 16.5            |
| 0.9                    | 17.0              | 17.5            |
| 1.0                    | 18.0              | 18.5            |
| 1.1                    | 19.0              | 19.5            |
| 1.2                    | 20.0              | 20.5            |
</details>

![](images/0a45d5c497c54fa89486f4913e556862a8b8d99046c1582ced5312f2495027d0.jpg)

<details>
<summary>line</summary>

| Perplexity | Imperfect Context | Perfect Context |
| ---------- | ----------------- | --------------- |
| 1500       | 0.3               | 0.2             |
| 1600       | 0.1               | 0.25            |
| 1700       | 0.5               | 0.1             |
| 1800       | 0.4               | 0.3             |
| 1900       | 0.5               | 0.4             |
</details>

(c)

![](images/ca29f962be1783f17f75811d4a87f684a95c8c9220e0d5cef20784a023a6e64d.jpg)

<details>
<summary>line</summary>

| Perplexity | PSNR - (higher is better) |
| ---------- | -------------------------- |
| 1500       | 16                         |
| 1600       | 20                         |
| 1700       | 14                         |
| 1800       | 12                         |
| 1900       | 10                         |
</details>

![](images/b0feea8fd1fa0a6e2cc1ad3256e7f0da8fc2e5ff2549e64a27546f0678a80eca.jpg)

<details>
<summary>bar_line</summary>

| Replacement Ratio | LPIPS (lower is better) | Embedding Similarity |
|---|---|---|
| 0.0 | 0.317 | 0.52 |
| 0.2 | 0.287 | 0.56 |
| 0.4 | 0.275 | 0.62 |
| 0.6 | 0.272 | 0.70 |
| 0.8 | 0.269 | 0.80 |
</details>

![](images/f70ec01994c9da3d11d67f0b9071f9d813097b7e9c388ceab0b003b13d53c9d2.jpg)

<details>
<summary>bar_line</summary>

| Replacement Ratio | PSNR (higher is better) | Embedding Similarity |
|---|---|---|
| 0.0 | 14.80 | 0.5 |
| 0.2 | 15.47 | 0.6 |
| 0.4 | 15.90 | 0.7 |
| 0.6 | 16.23 | 0.8 |
| 0.8 | 16.31 | 0.9 |
</details>

(e)

![](images/b4c8be2e1fef87737796b508dc60b8b1057bb2bc76dda4e3d98fb390f455edc9.jpg)

<details>
<summary>bar</summary>

| Replacement Ratio | LPIPS   | Δ Perplexity |
| ----------------- | ------- | ------------ |
| 0.0               | 0.317   | 0            |
| 0.2               | 0.287   | 50           |
| 0.4               | 0.275   | 150          |
| 0.6               | 0.272   | 250          |
| 0.8               | 0.269   | 350          |
</details>

(f)

![](images/4f595c4f03ea32a030b2fe630d84b859e147afc07bf7437948bcec27a0502489.jpg)

<details>
<summary>bar_line</summary>

| Replacement Ratio | PSNR | Δ Perplexity |
|---|---|---|
| 0.0 | 14.80 | 25 |
| 0.2 | 15.47 | 60 |
| 0.4 | 15.90 | 120 |
| 0.6 | 16.23 | 250 |
| 0.8 | 16.31 | 400 |
</details>

(g)   
Figure 8: Detailed analysis of generator–tokenizer inconsistency. Results are evaluated with CTR (↑), perplexity (↓), PSNR (↑), and LPIPS (↓). (a–d): Exposure bias analysis—under the same CTR and lower perplexity, imperfect context yields higher LPIPS and lower PSNR. (e–g): Embedding unawareness analysis—∆ Perplexity denotes the increase from original to replaced sequences; even with similar CTR and lower perplexity, original predictions can give worse PSNR/LPIPS, showing that higher-quality token sequences can be decoded into worse images.

Evaluation on decoded image. To assess the quality of the decoded images $\hat { \mathbf { I } } = \mathcal { D } ( \mathbf { z } ^ { \mathrm { q } } )$ , we report peak signal-to-noise ratio (PSNR) and learned perceptual image patch similarity (LPIPS). PSNR is a distortion-based metric that measures reconstruction fidelity relative to the ground-truth image I:

$$
\mathrm{MSE} = \frac {1}{3 H W} \sum_ {c = 1} ^ {3} \sum_ {i = 1} ^ {H} \sum_ {j = 1} ^ {W} \left(I _ {c i j} - \hat {I} _ {c i j}\right) ^ {2}, \tag {10}
$$

$$
\mathrm{PSNR} = 1 0 \cdot \log_ {1 0} \left(\frac {L ^ {2}}{\mathrm{MSE}}\right), \tag {11}
$$

where L is the maximum possible pixel value (e.g., 255 for 8-bit images). A higher PSNR indicates better pixel-wise reconstruction fidelity.

LPIPS, on the other hand, evaluates perceptual similarity by comparing deep features extracted from a pretrained network ϕ:

$$
\operatorname{LPIPS} (\mathbf {I}, \hat {\mathbf {I}}) = \sum_ {l} \frac {1}{H _ {l} W _ {l}} \left\| w _ {l} \odot \left(\phi_ {l} (\mathbf {I}) - \phi_ {l} (\hat {\mathbf {I}})\right) \right\| _ {2} ^ {2}, \tag {12}
$$

where $\phi _ { l } ( \cdot )$ denotes the activation map from layer l, and wl are learned weights that calibrate the contribution of each layer. Lower LPIPS corresponds to higher perceptual similarity.

# B.2 ANALYSIS EXPERIMENT SETTINGS

To analyze the inconsistency between token sequence behavior and decoded image quality, we study the relationship between token-level metrics (CTR, Perplexity) and image-level metrics (LPIPS, PSNR). The key challenge is to design controlled interventions such that one aspect of quality (token sequence or image) can be varied while holding the other approximately fixed, thereby revealing causal effects. In all experiments, we treat correct token ratio (CTR) as the control variable, since it is the most straightforward to manipulate, while Perplexity, LPIPS, and PSNR serve as dependent variables. This setup allows us to investigate how changes in token correctness propagate to perceptual differences in reconstructed images.

Experiments on amplified exposure bias. As discussed in Section 3.1, we design two decoding protocols to vary the amount of exposure bias under the same CTR level:

• Perfect Context (front-loaded): Given a target CTR r, we fix the first $\lfloor r n \rfloor$ tokens to ground truth $x _ { 1 : \lfloor r n \rfloor }$ and let the autoregressive model freely generate the remaining tokens. This minimizes exposure bias, since the context remains error-free until the switch point.   
• Imperfect Context (uniformly interleaved): For the same CTR r, we randomly select ⌊rn⌋ positions in the sequence and load ground-truth tokens only at those positions. At all other positions, tokens are sampled autoregressively. This introduces earlier corruption into the context and amplifies exposure bias.

Both settings guarantee the same number of ground-truth tokens, so any difference in downstream LPIPS/PSNR is attributable to the severity of exposure bias. This isolates the tokenizer’s role in amplifying exposure bias during generation.

Experiments on embedding unawareness. While exposure bias focuses on where ground-truth tokens are inserted, embedding unawareness examines what happens when incorrect tokens are replaced by semantically similar alternatives. During training, the autoregressive model is optimized for exact token prediction, whereas the tokenizer decoder operates in a continuous embedding space. To study this gap, we introduce a replacement ratio $r ^ { \prime } \in [ 0 , 1 ]$ :

1. First, generate predictions $\hat { x } _ { 1 : n }$ with teacher forcing. Identify all positions i where ${ \hat { x } } _ { i } \neq x _ { i }$   
2. For each such incorrect prediction, replace ${ \hat { x } } _ { i }$ with probability $r ^ { \prime }$ by another token $\boldsymbol { x } _ { i } ^ { \prime }$ whose embedding $z ^ { q } { } _ { i } ^ { \prime }$ is the closest to the correct embedding $z ^ { q } { _ i }$ i under cosine similarity, i.e.,

$$
z ^ {q ^ {\prime}} _ {i} = \underset {z ^ {q} \in \mathcal {Z} \setminus \{z ^ {q} _ {i} \}} {\arg \min} d (z ^ {q}, z ^ {q} _ {i}).
$$

3. The CTR remains unchanged, since replacements are only among incorrect predictions, but the embedding similarity of the sequence is improved.

By varying $r ^ { \prime } ,$ we control the degree of embedding similarity while holding CTR constant, and then measure its effect on LPIPS and PSNR of the reconstructed images. This design allows us to directly test whether embedding closeness—rather than token identity alone—affects perceptual quality.

Summary. For both experiments, we additionally evaluate the perplexity of the token sequence under the same CTR and study its correlation with LPIPS / PSNR as well. Together, these controlled settings—Perfect vs. Imperfect Context for exposure bias, and embedding replacement for unawareness—enable a systematic evaluation of how token-level inconsistencies translate into perceptual / pixel-level degradation in decoded images.

# B.3 FINDINGS AND OBSERVATION

Results on exposure bias. As shown in Figure 8(a–b), under the same CTR, sequences generated with imperfect context lead to higher LPIPS and lower PSNR, indicating worse decoded images, especially at low CTR. A similar trend is observed with perplexity. Although perplexity cannot be directly controlled, varying CTR naturally induces different perplexity levels. Thus, in Figure 8(c–d), we plot perplexity against PSNR/LPIPS under matched CTR. Even when the token sequence quality appears worse (higher perplexity), images decoded from tokens generated with perfect context still achieve better visual quality (lower LPIPS, higher PSNR) compared to those from imperfect context. This highlights that a token sequence favored by the autoregressive model does not necessarily yield a better decoded image.

Results on embedding unawareness. As shown in Figure 8(e–g), increasing the replacement ratio $r ^ { \prime }$ improves embedding similarity while keeping CTR unchanged. This leads to consistent improvements in decoded image quality: LPIPS decreases and PSNR increases as more incorrect predictions are replaced with embedding-nearest tokens. Importantly, even though perplexity rises due to these replacements, the resulting images become visually closer to the ground truth. Figure 7 further illustrates this effect—images reconstructed from sequences with higher replacement ratios (20–60%) recover clearer object structures (e.g., sharper outlines of the dog’s ears and the butterfly’s wings)

![](images/1180bfcbeec3b2c9254a2de6dc7bcad92bd083a8394f772332e6200e8698d300.jpg)

<details>
<summary>bar</summary>

| Condition | AR on Trained Data | AR on Unseen Data | reAR on Trained Data | reAR on Unseen Data |
| --------- | ------------------ | ----------------- | -------------------- | ------------------- |
| Clean     | 12.8               | 11.8              | 12.9                 | 12.6                |
| Noisy     | 9.1                | 8.3               | 9.5                  | 9.5                 |
</details>

(a) Correct token ratio of clean and noisy input

![](images/8b4e4d19d187ee29e4c5065862d79a27df646cf38ff01c666c903f10b83d1323.jpg)

<details>
<summary>line</summary>

| Layer Index | (AR) v.s. Encoded Token | (reAR) v.s. Encoded Token |
| ----------- | ------------------------ | -------------------------- |
| 0           | 0.4                      | 0.7                        |
| 5           | 0.15                     | 0.25                       |
| 10          | 0.08                     | 0.15                       |
| 15          | 0.05                     | 0.07                       |
| 20          | 0.03                     | 0.02                       |
</details>

(b) generator feature v.s. encoded embedding

![](images/34cb9489878746e47f9ab86a3bf144a0c70fb4ddacbea6e7c0f4244459e0c9fa.jpg)

<details>
<summary>line</summary>

| Layer Index | (AR) v.s. Decoded Token | (reAR) v.s. Decoded Token |
| ----------- | ------------------------ | -------------------------- |
| 0           | 0.04                     | 0.04                       |
| 5           | 0.07                     | 0.06                       |
| 10          | 0.06                     | 0.07                       |
| 15          | 0.10                     | 0.13                       |
| 20          | 0.07                     | 0.07                       |
</details>

(c) generator feature v.s. decoded embedding   
Figure 9: Mitigating inconsistencies in visual autoregressive generation. (a) reAR narrows the performance gap between trained and unseen data compared to vanilla AR and improves robustness under noisy inputs, indicating better generalization. (b, c) The CKA score demonstrates similarity between the feature and codebook embedding. reAR further aligns hidden features with the embedding of the current token in early layers and with the embedding of the next token in deeper layers.

compared to the 0% baseline. These results demonstrate that token correctness alone is insufficient to guarantee high-quality reconstructions; instead, embedding proximity plays a crucial role in aligning autoregressive predictions with tokenizer decoding.

# C ANALYSIS ON THE EFFECT OF REAR

In this section, we present further analysis on the effect of reAR: (i) its effect on the token space (Section C.1) and (ii) its effect on the hidden features, which also includes the analysis on the choice of regularization layer as mentioned in Section 3.2 and Section 4.3.

# C.1 IMPACT ON SAMPLED TOKEN SEQUENCE

We found that reAR improves the next token prediction on: (i) generalization and (ii) robustness.

Generalization. We compare the correct token ratio (CTR) of vanilla AR and reAR on both trained data1 and unseen validation data from ImageNet-1K as shown in Figure 9(a). On clean inputs, reAR achieves nearly identical performance to vanilla AR on trained data (12.9 vs. 12.8), but obtains higher CTR on unseen data (12.3 vs. 11.8), indicating improved generalization. These results suggest that incorporating codebook embeddings provides a stronger inductive bias for visual signals, enabling the AR model to learn more generalizable representations.

Robustness. To examine the robustness gained from reAR, we randomly replace a fraction of current tokens with noise at a controlled rate. Figure 9 (a) also compares the CTR for clean sequences and for sequences with 10% of tokens replaced uniformly. Compared to vanilla AR, reAR gains higher CTR compared to AR on the noisy trained data (9.5 v.s. 9.1). On the noisy and unseen data, the performance gap is even larger: reAR substantially outperforms vanilla AR (9.0 vs. 8.3). This result shows that reAR is more robust to the possible exposure bias.

# C.2 IMPACT ON HIDDEN FEATURES OF DIFFERENT REGULARIZATION LAYER

To better understand how reAR interacts with hidden representations, we evaluate the similarity between generator features and tokenizer embeddings using centered kernel alignment (CKA) (Kornblith et al., 2019). Specifically, given two sets of feature representations $\textbf { X } \in \ \mathbb { R } ^ { n \times d _ { x } }$ and $\mathbf { Y } \in \mathbb { R } ^ { n \times d _ { y } }$ , we first compute their Gram matrices $\mathbf { K } = \mathbf { X } \mathbf { X } ^ { \top }$ and $\mathbf { L } = \mathbf { Y } \mathbf { Y } ^ { \top }$ , and then center them as $\mathbf { K } _ { c } = \mathbf { H } \mathbf { K } \mathbf { H }$ and $\mathbf { L } _ { c } = \mathbf { H } \mathbf { L } \mathbf { H }$ , where $\begin{array} { r } { { \bf H } = { \bf I } _ { n } - \frac { 1 } { n } { \bf 1 } _ { n } { \bf 1 } _ { n } ^ { \top } } \end{array}$ is the centering matrix. The

CKA score is defined as

$$
\operatorname{CKA} (\mathbf {X}, \mathbf {Y}) = \frac {\left\langle \mathbf {K} _ {c} , \mathbf {L} _ {c} \right\rangle_ {F}}{\left\| \mathbf {K} _ {c} \right\| _ {F} \left\| \mathbf {L} _ {c} \right\| _ {F}}, \tag {13}
$$

where $\langle \cdot , \cdot \rangle _ { F }$ denotes the Frobenius inner product and $\| \cdot \| _ { F }$ is the Frobenius norm. Intuitively, CKA measures the alignment between the pairwise similarity structures of two representations and is invariant to isotropic scaling and orthogonal transformation. A higher CKA score indicates that the hidden features of the generator are more similar to the corresponding tokenizer embeddings.

Analysis Target. We aim to examine how hidden features within the decoder-only transformer correlate with two types of embeddings: the encoded embedding $\mathbf { z } _ { i } ^ { q }$ , representing the codebook vector of the current token, and the decoded embedding $\mathbf { z } _ { i + 1 } ^ { q } ,$ corresponding to the codebook vector of the next token. By comparing generator features against both embeddings, we can assess how the autoregressive model’s hidden representations evolve—capturing alignment with the tokenizer’s codebook while simultaneously encoding the current token and preparing to decode the next one.

Correlation between hidden features and embeddings. To analyze how autoregressive representations evolve with depth, we compute CKA similarity between hidden features of a vanilla AR model and tokenizer embeddings across layers (Figure 9(b–c)). Four key trends emerge: (1) overall, CKA with the decoded embedding is lower than with the encoded embedding, since the current token is known while the next token remains uncertain; (2) similarity to the encoded embedding is highest at the input layer and decreases monotonically with depth; (3) similarity to the decoded embedding gradually increases and peaks around layer 15, roughly three-quarters of the full architecture; and (4) similarity to the decoded embedding

drops again in the final layers. Together, these patterns suggest a natural progression: early layers focus on encoding the current token and aggregating contextual information, while deeper layers shift toward modeling the next-token embedding. The decline in the final layers likely reflects the model’s need to project features onto a decision boundary for prediction, where the codebook embedding itself may not form an optimal target. This also explains why directly tying AR outputs to codebook embeddings can lead to suboptimal performance.

<table><tr><td>Regularization settings</td><td>FID ↓</td><td>IS ↑</td></tr><tr><td>DE@13</td><td>20.47</td><td>59.4</td></tr><tr><td>DE@14</td><td>20.17</td><td>60.8</td></tr><tr><td>DE@15</td><td>20.03</td><td>61.0</td></tr><tr><td>DE@16</td><td>20.11</td><td>60.5</td></tr><tr><td>DE@17</td><td>20.25</td><td>61.1</td></tr></table>

Table 7: Analysis on nearby regularization layer. We use ‘EN’ as the encoding regularization and ‘DN’ as the decoding regularization. For example, ‘DN@15’ means applying decoding regularization at the 15th layer of the transformer block.

Choosing the regularization layer. Motivated by these observations, we design reAR to apply regularization at layers where the CKA similarity is naturally high—early layers for encoded embeddings and later layers for decoded embeddings. Intuitively, this choice minimizes conflict with the primary next-token prediction objective, since these layers are already aligned with the tokenizer. Importantly, we avoid imposing regularization at the very last layer. Instead, we place regularization near the three-quarter depth of the model, where decoded embedding similarity is maximized. Empirically, we find that applying reAR to nearby layers yields similar performance as Table 7, highlighting the flexibility of our method with respect to the choice of regularization layer.

Effect of reAR on feature alignment. After introducing reAR, we observe consistent increases in CKA similarity between generator features and both encoded and decoded embeddings (Figure 9(b–c)) at the target layer. In early layers, reAR strengthens alignment with encoded embeddings, helping the generator encode current tokens similar to the tokenizer. In deeper layers, reAR improves similarity with decoded embeddings, ensuring that hidden features are better aligned with the next token. This result indicates that reAR directly improves the consistency between the hidden feature of the autoregressive model and the tokenizer.

# D ADDITIONAL DISCUSSION ON THE RELATED WORK

In this section, we present a detailed discussion and comparison of related work. Diffusion models have achieved great success in many downstream visual tasks, including image editing (Nichol et al., 2022; Meng et al., 2022; He et al., 2024; Hertz et al., 2022) and personalized image generation (Gal et al., 2022; Ruiz et al., 2023; He & Yao, 2025; Tan et al., 2025). By contrast, visual autoregressive models are less frequently used in these domains, mainly because their generation quality often lags behind that of diffusion models. A growing line of research aims to bridge this gap between visual autoregressive modeling and diffusion-based approaches. In the following, we mainly discuss that how these prior methods can be viewed through a unified lens: they address the inconsistency between the tokenizer (or tokenization scheme) and the autoregressive model. We also discuss how MAR and VAR differ from other autoregressive approaches, and highlight the distinction between our method and REPA, a regularization technique proposed for visual generation.

# D.1 TOKENIZATION WITH RANDOMIZED ORDER

RandAR (Pang et al., 2025) introduces a positional token in front of each patch token to let the token be aware of its position in terms of tokenization. Specifically, given a 256 token sequence, it inserts additional 256 tokens, and the generator is required to learn the distribution of the total 512 tokens under permutation. During training and inference, the token sequence is always shuffled. It enables parallel decoding during inference by inserting multiple positional tokens simultaneously. However, RandAR can double the context and significantly increase the computation budget.

RAR (Yu et al., 2024a) introduces a learnable embedding of target position over each token. During training, it randomly shuffles the token sequence at a given probability, and the token is aware of its own position with the additional positional embedding. It slowly decreases the probability of shuffling and returns to standard rasterization order during training. During inference, it keeps the standard operation for the autoregressive generation.

Summary. Both RandAR and RAR use permutation during training so that the context of each token is not limited to the tokens that are on the left or the top of it, thereby introducing bidirectional context even using a decoder-only transformer. This mitigates the inconsistency between the tokenizer that also models bidirectional context, such as MaskGiT-VQGAN (Chang et al., 2022) or TiTok (Yu et al., 2024b). However, in terms of the advanced tokenizer already introduced, unidirectional dependency, such as AliTok (Wu et al., 2025) and FlexTok (Bachmann et al., 2025), may further amplify the inconsistency as Table 2 in the main text shows.

# D.2 TOKENIZATION WITH 1D SEQUENCE OR UNIDIRECTIONAL DEPENDENCY

TiTok (Yu et al., 2024b) transforms an image into 1D discrete token sequence with query token using ViT. It firstly decouples the number of tokens from the number of patches and can further compress the number of tokens. However, the reconstruction quality of TiTok remains suboptimal compared to the patchify tokenizer. Additionally, although the represented token sequence is 1-dimensional, it’s still in a bidirectional context instead of modeling unidirectional dependency. Therefore, the autoregressive model trained on it remains suboptimal as Table 2 in the main text shows.

GigaTok (Xiong et al., 2025) transforms the image into 1D discrete token sequence as well similar to TiTok. Additionally, it introduces the feature from DINO-v2, similar to REPA (Yu et al., 2024c) to regularize the hidden feature of the tokenizer decoder. This enables the tokenizer to scale up and stabilize training. However, it suffers from the same problem as TiTok, which still models bidirectional dependency.

FlexTok (Bachmann et al., 2025) firstly learns a continuous VAE with high fidelity. It then further resamples 1D discrete tokens from the 2D continuous token obtained from the VAE. Different from TiTok and GigaTok, it additionally employs a causal mask on the 1D sequence to model the unidirectional dependency, which is more consistent with an autoregressive model.

AliTok (Wu et al., 2025) introduces an Aligned Tokenizer that uses 1D sequences instead of the typical 2D patch grid, but with novel training to better align the tokenizer with autoregressive generation. Unlike standard patchified tokenizers, AliTok uses a causal decoder during tokenizer training to enforce unidirectional dependency among encoded tokens, so that tokens depend only on preceding ones. After that, it freezes the encoder and then uses a bidirectional decoder to refine the reconstruction quality. This unidirectional alignment improves compatibility with autoregressive models and leads to state-of-the-art generation metrics — our method still further enhances performance.

Summary. These works (e.g. TiTok (Yu et al., 2024b) and AliTok (Wu et al., 2025)) impose a 1D token sequence or enforce unidirectional dependency in the tokenization stage so that the tokenizer is more aligned with autoregressive models, which shows the importance of consistency between the tokenizer and autoregressive model. In our experiments, we further demonstrate that using generator-tokenizer consistency regularization can further improve upon their performance.

# D.3 REMARKS ON MAR AND VAR

MAR (Li et al., 2024) is a model paradigm that combines masked prediction and autoregressive generation. Rather than generating tokens strictly in a raster (1D) order, MAR predicts multiple masked tokens in parallel across iterations, while still enforcing an ordering among iterations. Importantly, MAR uses continuous tokens instead of discrete ones and employs a diffusion-based head to model the continuous distribution of token predictions.

VAR (Tian et al., 2024) proposes a coarse-to-fine next-scale prediction strategy in image generation: rather than predicting each patch or token in a raster order, VAR generates images scale by scale, first at low resolution and then successively higher resolutions, where each finer scale is conditioned on all previously generated coarser scales. Given tokens of previous scale, the model will provide multiple mask tokens corresponding to the next scale, and decode them in parallel.

Summary. Although MAR and VAR can be regarded as autoregressive since generation proceeds in an autoregressive manner, they implement it with an encoder-only transformer or block causal transformer. In MAR, the model receives masked tokens as input and learns to reconstruct the masked positions, rather than predicting the next token in a decoder-only setup. In VAR, tokens from the previous resolution provide the context for predicting multiple tokens at the next resolution in parallel. Both model are different from standard AR paradigm of next token prediction.

# D.4 REGULARIZATION ON GENERATION TECHNIQUE

REPA (Yu et al., 2024c) is a regularization technique for diffusion-transformer models that aligns noisy intermediate states in the denoising process with clean image features from a pretrained visual encoder. Rather than forcing the model to learn image representations from scratch under noisy conditions, REPA adds a loss that encourages the hidden states of the diffusion model to match the semantic structure of an external teacher (e.g., DINO, DINO-v2).

Comparison. Unlike REPA, which focuses on accelerating the training of diffusion models, reAR is designed to address the inconsistency between autoregressive models and their tokenizers. Moreover, while REPA relies on external feature extractors such as DINO-v2 (Oquab et al., 2023), reAR directly leverages features from the tokenizer, which is already an integral component of the visual generation pipeline. In addition, REPA is tailored to bidirectional transformers and is restricted to 2D tokenizers, whereas reAR is compatible with decoder-only transformers. For these reasons, we do not apply REPA to visual autoregressive models, as it is less generalizable to visual AR training.

# E DISCUSSION ON THE LIMITATION

Our method has several limitations that suggest promising directions for future work. First, the choice of the decoding regularization layer is determined empirically. This issue is not unique to our approach, as prior works that regularize intermediate representations, such as REPA (Yu et al., 2024c) and Dispersive Loss (Wang & He, 2025), also depend on empirically selected layers in the absence of a clear theoretical principle. Developing an adaptive or theoretically grounded strategy for layer selection remains an open challenge and is more closely aligned with ongoing research in automated architecture and hyperparameter search.

Second, our experiments focus primarily on ImageNet, following common practice in foundational visual generative modeling (Esser et al., 2021; Yu et al., 2024b; Wu et al., 2025). While this setup enables controlled comparisons, we did not evaluate reAR on downstream text-guided generation tasks. A comprehensive evaluation on standard text-to-image benchmarks would offer a clearer assessment of practical utility, but is computationally demanding. We leave an expanded downstream study for future work.

Finally, although our empirical results demonstrate the effectiveness and generalization capability of reAR and we provide direct CKA analysis on the hidden feature of transformer layers before and after regularization, we do not provide a deeper theoretical analysis of the geometric factors underlying generator–tokenizer alignment. Understanding properties such as manifold structure or distributional behavior could yield a more principled perspective, but developing such a theoretical framework is non-trivial. We view this as an important direction for future research.

# F QUALITATIVE RESULTS

We present comprehensive generated results of reAR-B-AliTok (Figure 10 to 18) and reAR-L-VQGAN (Figure 19 to 24). All results are generated with a constant guidance scale of 4.0.

![](images/93e680d63f731013581334713c2789c11debacb322767c01f0962814e347fdb3.jpg)

<details>
<summary>natural_image</summary>

Collage of ten coastal and rocky landscapes featuring cliffs, sea, and natural landscapes under clear skies (no text or symbols visible)
</details>

Figure 10: Generated Results of reAR-B-AliTok of class $\bf \delta C l i f f ^ { \prime }$

![](images/7bc15f5777588b9f0126fe132d74089a4e5b9c2ac701498b9cb490b384c82087.jpg)

<details>
<summary>natural_image</summary>

Collage of various fish and aquatic animals including goldfish, goldfish swimming in water, and others in natural scenes (no text or symbols)
</details>

Figure 11: Generated Results of reAR-B-AliTok of class ‘Goldfish’

![](images/b754c50f0ec19e7469d2a3ae4c11eca91a02d85b94391664bf9b1d062a965425.jpg)

<details>
<summary>natural_image</summary>

Collage of eleven different dog breeds and animals, including a black-and-white Labrador Retriever, sitting on grass, and smiling (no text or symbols visible)
</details>

Figure 12: Generated Results of reAR-B-AliTok of class ‘Labrador retriever’

![](images/b7ec89c43d25bfb03c9677c7de4cc7c2123b4650e68acabae0d44891c8ac6de0.jpg)

<details>
<summary>natural_image</summary>

Collage of nine food and beverage items including ice cream, chocolate sauce, and dessert cakes (no visible text or labels)
</details>

Figure 13: Generated Results of reAR-B-AliTok of class ‘Ice cream’

![](images/8e8076b3a5c40616939240533ff6ad9b630f8bcf03d5fdcc1ab92f139feb1516.jpg)

<details>
<summary>natural_image</summary>

Collage of scenic lake views including forests, wetlands, and natural landscapes under clear skies (no text or symbols)
</details>

Figure 14: Generated Results of reAR-B-AliTok of class ‘Lakeshore’

![](images/2d6bf26b616e5088e6dacfc3e98f6b672e095eb59e9185776dbd6bd34c687bd8.jpg)

<details>
<summary>natural_image</summary>

Collage of various hamburger designs and toppings, including cheese, lettuce, cheese bun, and bacon (no text or labels visible)
</details>

Figure 15: Generated Results of reAR-B-AliTok of class ‘Cheeseburger’

![](images/123e460e5372094173529e3f04bd3842fdff5395b307115fa0e551977096a0e0.jpg)

<details>
<summary>natural_image</summary>

Collage of 16 black steel arch bridge designs in various locations, including overpasses, bridges, and waterways under clear sky (no text or symbols visible)
</details>

Figure 16: Generated Results of reAR-B-AliTok of class ‘Bridge’

![](images/f7fffa0fd0cbc46ef428b645c37612295ced88967dc86df23ffb2f76426df0a5.jpg)

<details>
<summary>natural_image</summary>

Collage of colorful hot air balloons against a clear blue sky, no visible text or symbols.
</details>

Figure 17: Generated Results of reAR-B-AliTok of class ‘Balloon’

![](images/b6dfd260df6da227554f6172af96d56d212a079ed73843b5c55853d2ef394b1d.jpg)

<details>
<summary>natural_image</summary>

Grid of 12 photos of small dogs, including one with a knitted collar and another with a blue T-shirt, all in various poses and expressions (no text or symbols visible)
</details>

Figure 18: Generated Results of reAR-B-AliTok of class ‘Chihuahua’

![](images/3f80fbc30899bd9e7763000cce7989549e6e1529689ffdc0f3c1781e63850ac3.jpg)

<details>
<summary>natural_image</summary>

Collage of various roosters in natural habitat, showing detailed head and body features (no text or symbols)
</details>

Figure 19: Generated Results of reAR-L-VQGAN of class ‘Cock’

![](images/1d138b79e5b75e11878071ac974e017afbb072f964d40b7dff9cb5e7bc9d9f51.jpg)

<details>
<summary>natural_image</summary>

Collage of green snake heads and claws, showing detailed scales and eye contours (no text or symbols)
</details>

Figure 20: Generated Results of reAR-L-VQGAN of class ‘Green mamba’

![](images/bf80a1dc00fd3381e540e8a92ea8392d42eadebd8f2f2da1abe15fdbd83cf69e.jpg)

<details>
<summary>natural_image</summary>

Collage of various hand-drawn crabs on sandy beach, showing various sizes and shell patterns (no text or symbols)
</details>

Figure 21: Generated Results of reAR-L-VQGAN of class ‘Hermit crab’

![](images/dc9de479a17fce781a7bb273e72bceadba254604454860679baa7efa8e390287.jpg)

<details>
<summary>natural_image</summary>

Collage of photos of a pink flamingo with visible beak and fanned wings, no text or symbols present.
</details>

Figure 22: Generated Results of reAR-L-VQGAN of class ‘Flamingo’

![](images/6c1f81ce564db614656857a0a80ac5bf5b8ec3060db9489a2c8947670ae03c5b.jpg)

<details>
<summary>natural_image</summary>

Collage of hourglass images showing different stages of writing or processing, with no visible text or symbols.
</details>

Figure 23: Generated Results of reAR-L-VQGAN of class ‘Hourglass’

![](images/3a9ee9434998203714e710045417800bc59fe64a17587d2a878a3d4708af11c9.jpg)

<details>
<summary>natural_image</summary>

Collage of 12 historical sailing ships in various styles and orientations, displayed against a tropical beach backdrop (no visible text or symbols)
</details>

Figure 24: Generated Results of reAR-L-VQGAN of class ‘Pirate’