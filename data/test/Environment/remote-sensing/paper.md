# Remote Sensing Change Detection with Transformers Trained from Scratch

Mubashir Noman1 Mustansar Fiaz1 Hisham Cholakkal1 Sanath Narayan2 Rao Muhammad Anwer1,3 Salman Khan1,4 Fahad Shahbaz Khan1,5

1Mohamed bin Zayed University of AI, UAE 2Technology Innovation Institute, UAE 3Aalto University, Finland 4Australian National University 5Linkoping University, Sweden ¨

# Abstract

Current transformer-based change detection (CD) approaches either employ a pre-trained model trained on large-scale image classification ImageNet dataset or rely on first pre-training on another CD dataset and then finetuning on the target benchmark. This current strategy is driven by the fact that transformers typically require a large amount of training data to learn inductive biases, which is insufficient in standard CD datasets due to their small size. We develop an end-to-end CD approach with transformers that is trained from scratch and yet achieves stateof-the-art performance on four benchmarks. Instead of using conventional self-attention that struggles to capture inductive biases when trained from scratch, our architecture utilizes a shuffled sparse-attention operation that focuses on selected sparse informative regions to capture the inherent characteristics of the CD data. Moreover, we introduce a change-enhanced feature fusion (CEFF) module to fuse the features from input image pairs by performing a per-channel re-weighting. Our CEFF module aids in enhancing the relevant semantic changes while suppressing the noisy ones. Extensive experiments on four CD datasets reveal the merits of the proposed contributions, achieving gains as high as 14.27% in intersection-over-union (IoU) score, compared to the best published results in the literature. Code is available at https://github.com/ mustansarfiaz/ScratchFormer.

# 1. Introduction

Change detection (CD) is a fundamental remote sensing research problem that strives to identify all relevant changes between co-registered satellite images acquired at distinct timestamps. Here, the objective is to detect relevant semantic changes in man-made facilities such as, buildings and other constructions while ignoring noisy changes such as shadows and all types of seasonal and environmental variations. CD plays a crucial role in various remote sensing applications including, disaster management [13], urban planning [23], forestry and ecosystem monitoring [7, 21]. Generally, CD approaches relying on convolutional neural networks (CNN) has shown promising results by utilizing explicit mechanisms such as dilated convolutions, channel and spatial attentions [25, 4, 24, 5, 3]. However, these CNNbased approaches typically struggle to capture long-range dependencies between different image regions, hampering the change detection performance.

Recently, transformer-based CD methods [1, 15, 18, 20, 12, 2] have shown competitive performance on various CD datasets by capturing long-range dependencies between uniformly sampled dense patches through selfattention [19]. Although achieving superior CD performance, state-of-the-art transformer-based methods [1] generally require pre-training based weight initialization for optimal convergence. The pre-training step in existing transformer-based CD methods either involve another CD dataset [1] or an ImageNet pre-trained image classification model [2, 15, 18, 17, 12]. However, the performance of these transformer-based CD methods drastically reduce when directly training from scratch on the target CD dataset. This is likely due to the dense self-attention operation, utilized in these approaches, that has quadratic complexity with respect to tokens, requires longer to converge and prone to over-fitting. In this work, we look into the problem of designing a transformer-based CD approach that is capable of achieving high performance when trained from scratch.

Most existing transformer-based CD approaches employ a two-stream architecture, where features from both streams are combined through simple operations such as, difference, summation and concatenation [8, 1]. However, these approaches do not employ any explicit feature re-weighting between both streams. We argue that such naive feature fusion strategies likely struggle to effectively aggregate semantic changes from each stream. In this work, we set out to address the above issues collectively in a single transformer-based CD architecture.

Contributions: We propose a transformer-based Siamese two-stream CD framework, named ScratchFormer, that is based on a novel shuffled sparse attention (SSA) operation that strives to better attend to sparse informative regions relevant to the CD task. The proposed SSA performs tokenmixing over a sparse subset of shuffled features obtained through a data-dependent feature sampling, enabling optimal CD performance when being trained from scratch directly on the target CD dataset. Furthermore, we introduce a change-enhanced feature fusion module (CEFF) that performs feature fusion based on per-channel re-calibration to enhance the features relevant to the semantic changes, while suppressing the noisy ones.

(a) Pre-change image   
![](images/8dacf3edfbe14c90f817084ef375a6f7edc0793c5101da1bfe4a1e9d477ad393.jpg)

<details>
<summary>natural_image</summary>

Aerial view of agricultural fields with patchwork and green vegetation (no visible text or symbols)
</details>

(b) Post-change image   
![](images/96b7af655c6af5b65bb3e0ad32b616469b9c5169474b42cf9da000f8f7943b1e.jpg)

<details>
<summary>natural_image</summary>

Aerial view of an industrial or construction site with red-labeled factory buildings and surrounding infrastructure (no visible text or symbols)
</details>

![](images/fd5ccdc96cc9ec71a94e1309e56f322b92b92f8629a7b38317d612754884d6b5.jpg)

<details>
<summary>natural_image</summary>

Aerial view of agricultural fields with patchwork patterns and no visible text or symbols.
</details>

![](images/a93941e0025954bf1e05b282e052aba21450c5802da50f21e8f749982d18999f.jpg)

<details>
<summary>natural_image</summary>

Aerial view of a river delta with surrounding landscape features (no visible text or symbols)
</details>

(c) ChangeFormer   
![](images/920b45fe3bd5cc9702e9b08a9e6c4436ef65782bf86baa15538e04052909297b.jpg)

<details>
<summary>natural_image</summary>

Abstract black-and-white graphic with a red rectangle and purple rectangle overlay (no text or symbols)
</details>

![](images/896fdb58f22164fd2fac0824a18486da9d6d4e167f9b1907ff3495b949bcd6ec.jpg)

<details>
<summary>natural_image</summary>

Abstract black background with white cloud and red rectangular shapes (no text or symbols)
</details>

(d) ScratchFormer (Ours)   
![](images/c251967e8613a9f91e9c6ef23c32617ed3880809a3527af4aa3ef19cfcce0a8f.jpg)

<details>
<summary>natural_image</summary>

Abstract black geometric shape on white background (no text or symbols)
</details>

![](images/dd6b2242bb4edf02ad27fb456ea0a2ea9b69e0ed6958ff0cbdb968e60845214f.jpg)

<details>
<summary>natural_image</summary>

Abstract black and white geometric shape with no text or symbols
</details>

(e) Ground Truth   
![](images/49b153609895887d6abdf552ebb7534f1f324db19633a05b66cc367b1e9a9a4f.jpg)

<details>
<summary>natural_image</summary>

Abstract black geometric shape on white background (no text or symbols)
</details>

![](images/92ad845d7945cb53ca7075ede731ffc2ca09b838ff7b6b548b962520f9f29e7e.jpg)

<details>
<summary>natural_image</summary>

Abstract black shape on white background with no text or symbols
</details>

Figure 1. Change detection (CD) performance comparison of (d) our approach (ScratchFormer) with (c) the recent ChangeFormer [1] on examples from DSIFN-CD dataset. Here, (a) the pre-change image and (b) post-change image are shown along with (e) the ground-truth. We show the false positives and false negatives in the purple and red color, respectively. In both rows, the ChangeFormer incorrectly detects a change region (purple box). Similarly, the ChangeFormer fails to detect the change occurring between the pre- and post-images (red box in both rows), as indicated in the ground-truth. Our ScratchFormer achieves improved change detection performance by reducing both false positive and negatives. Best viewed zoomed in. Additional results are in Fig. 4, Fig. 5 and the supplementary.

We perform extensive experiments on four public CD datasets: LEVIR-CD [3], DSIFN-CD [24], WHU-CD [10], and CDD-CD [14]. Our proposed ScratchFormer approach achieves superior performance over the baseline, highlighting the effectiveness of the proposed contributions. Compared to the baseline, our ScratchFormer achieves an absolute gain of 14.27% in terms of intersection-over-union (IoU) on the DSIFN-CD dataset. Furthermore, Scratch-Former sets a new state-of-the-art performance on all four datasets. On the DSIFN-CD, our ScratchFormer achieves an IoU score of 90.75%, outperforming the best existing method [1] published in literature by 14.27%. Fig. 1 shows a qualitative comparison between the recent Change-Former [1] and our ScratchFormer on examples from DSIFN-CD dataset.

# 2. Preliminaries

Problem Formulation: Given $I _ { p r e } , I _ { p o s t } \in \mathbb { R } ^ { 3 \times H \times W }$ as a pair of co-registered satellite images acquired at distinct times $T _ { 1 }$ and $T _ { 2 } ,$ , the objective in change detection (CD)

is to detect relevant semantic changes between $I _ { p r e }$ and $I _ { p o s t }$ while ignoring irrelevant changes. Here, the relevant changes include changes in man-made facilities such as, buildings and other constructions. On the other hand, the irrelevant changes include seasonal variations, illumination changes, building shadows, and atmospheric variations. Consequently, the goal in CD is to predict a binary mask $M \in \bar { \mathbb { R } } ^ { H \times \bar { W } }$ that depicts the semantic (structural) changes between $I _ { p r e }$ and $I _ { p o s t }$ .

# 2.1. Baseline Change Detection Framework

We adapt the recently introduced transformer-based approach [1] as our base framework since it achieves promising performance for the CD task. The base CD framework takes an image pair as input and computes the semantic difference between them using a transformer-based Siamese network. It comprises a transformer encoder, difference feature fusion module, and a decoder. The encoder consists of series of attention layers with each layer comprising the standard self-attention [19] followed by a feed-forward network. The encoder weights are shared and utilized for computing multi-scale features in both streams (pre-change and post-change). For each scale i, the resulting features $F _ { p r e } ^ { i } , F _ { p o s t } ^ { i }$ F post from both the streams are input to a difference feature fusion module, which encodes the semantic changes occurring between the streams in the corresponding scale. The difference feature fusion module comprises a feature concatenation followed by two convolutions with batch normalization and ReLU layers in between. It then outputs the feature $F _ { \mathrm { d i f f } } ^ { i }$ for scale i. These multi-scale features $F _ { \mathrm { d i f f } } ^ { i }$ are then input to the decoder, where they are fused through series of convolution and transpose convolution layers for increasing the spatial resolution of feature maps. Finally, the resulting upsampled features are passed to a mask prediction layer to obtain final semantic binary change map M .

![](images/c3305bd901d8bdd0c50973779516605c3d778dbb1a44c109d0c9da1023b4319b.jpg)

<details>
<summary>line</summary>

| Epochs | IoU_1 Baseline (Scratch) | IoU_1 Baseline (Pretrain) | IoU_1 Our |
| ------ | ------------------------- | -------------------------- | --------- |
| 0      | 40                        | 40                         | 40        |
| 50     | 50                        | 60                         | 70        |
| 100    | 60                        | 75                         | 85        |
| 150    | 65                        | 80                         | 88        |
| 200    | 70                        | 83                         | 90        |
| 250    | 75                        | 84                         | 90        |
| 300    | 78                        | 84                         | 90        |
| 350    | 79                        | 84                         | 90        |
| 400    | 79                        | 84                         | 90        |
</details>

Figure 2. Comparison, in terms of intersection-over-union (IoU) vs. the training epochs, among the baseline trained from scratch, baseline pre-trained† first on another CD dataset and then finetuned, and our approach on the DSIFN-CD. Compared to the baseline employing pre-training, training the baseline from scratch results in inferior convergence in terms of CD performance. Our approach despite being trained from scratch achieves superior convergence in terms of CD performance compared to both variants of baseline. For instance, with only 10% of the training time our approach achieves similar CD performance to that of the final results obtained from the baseline trained from scratch.

Limitations: As discussed above, the base framework employs the transformer encoder with the standard selfattention mechanism to capture long-range dependencies in an image. Here, we argue that the standard self-attention mechanism is sub-optimal for the CD task mainly due to following reason. It operates on uniformly sampled dense patches, thereby requiring large training data for optimal convergence in terms of CD performance (see Fig. 2). The recent ChangeFormer [1] alleviate this issue by performing pre-training on one (source) CD dataset followed by finetuning on another (target) CD data. However, this increases the training time when including the cost of pre-training on another CD dataset as well. Furthermore, despite being trained from scratch the proposed ScratchFormer outperforms our baseline accuracy that is achieved through a pre-training step on another CD dataset, with less than 50% of the training time.

# 3. Method

# 3.1. Motivation

To motivate our proposed approach, we distinguish two properties especially desired when designing a transformerbased change detection (CD) method.

Rethinking Attention for CD Task: As discussed earlier, the conventional self-attention may lead to sub-optimal performance when training from scratch directly on the target CD dataset, likely due to difficulty in capturing the inherent inductive biases in the small CD dataset. Moreover, the standard self-attention typically operates on uniformly sampled dense patches that may have difficulties to learn a rich feature representation encoding diverse shape objects with inconsistent appearance in remote sensing scenes having sparse informative regions. Therefore, rethinking the design of the self-attention is desired to effectively learn a rich feature representation by attending to sparse informative regions in remote sensing CD images.

Semantic Change-enhanced Feature Fusion: While the above requisite focuses on designing a mechanism to attend the sparse informative regions for the CD task, the second desirable characteristic aims at capturing the semantic differences between image pairs while ignoring the irrelevant noisy changes. To this end, a change-enhanced feature fusion module that explicitly models per-channel interdependencies between pre- and post-change images is expected to better ignore the noisy changes while retaining the relevant ones. Next, we present our proposed transformerbased ScratchFormer framework.

# 3.2. Overall Architecture

Fig. 3 shows the overall architecture of our Scratch-Former. The proposed ScratchFormer takes pre- and postchange image pairs $( I _ { p r e } , I _ { p o s t } )$ as input. It comprises a Siamese-based encoder, a change-enhanced feature fusion (CEFF) module, and a decoder for predicting the binary change map M . The encoder computes the features at four stages with different spatial resolutions. At each stage, the features are first spatially downsampled through convolutional layers and then input to the shuffled sparse attention (SSA) layers. The ScratchFormer consists of two parallel identical encoder streams with shared weights to generate pre- and post- change features Fˆipr $\hat { F } _ { p r e } ^ { i } , \hat { F } _ { p o s t } ^ { i } ,$ Fˆipost, respectively at the i-th stage of our multi-stage network. The focus of our design is the introduction of a novel shuffled sparse attention (SSA) layer in the encoder to perform the self-attention on the data-dependent subset of features to effectively capture the semantic changes for CD task. Furthermore, we propose a change-enhanced feature fusion (CEFF) module that re-calibrates the per-channel features of the same scale from both streams $( \hat { F } _ { p r e } ^ { i }$ and $\hat { F } _ { p o s t } ^ { i } )$ and performs enhanced feature fusion to better ignore the noisy changes while retaining the relevant ones.

Our SSA layer comprises a shuffled sparse attention (SSA) and a multi-layer-perceptron (MLP), as shown in Fig. 3-(a). SSA first performs a data-dependent sampling of features to obtain a subset and then performs token-mixing over the selected subset. SSA strives to focus on the sparse informative regions for change detection to achieve optimal convergence with respect to CD performance without requiring pre-training on another CD data. The CEFF modules aims to enhance the semantic changes between pre- and post- change features at each stage of the encoder, while suppressing the noisy changes. The resulting enhanced features from the CEFF module are re-sized to a common spatial resolution and passed to the decoder. The decoder has a series of convolution, transpose convolution, and upsampling layers to increase the spatial resolution of the feature maps. Consequently, these upsampled features are passed to a mask prediction layer to obtain the final binary mask M . Next, we present our SSA layer.

![](images/328a49570e22815c592bab3705cae94c2f68062df52f8638b264f246b9a8decb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph "Encoder"
        A["Pre-change Image"] --> B["Downampling"]
        B --> C["Shuffled Sparse Attention Layer ×3"]
        C --> D["Downampling"]
        D --> E["×3"]
        E --> F["Shuffled Sparse Attention Layer ×3"]
        F --> G["Downampling"]
        G --> H["×9"]
        H --> I["Shuffled Sparse Attention Layer ×3"]
        I --> J["Downampling"]
        J --> K["×3"]
        K --> L["Shuffled Sparse Attention Layer ×3"]
        L --> M["×3"]
    end

    subgraph "Post-change Image"
        N["Pre-change Image"] --> O["Downampling"]
        O --> P["Shuffled Sparse Attention Layer ×3"]
        P --> Q["Downampling"]
        Q --> R["×3"]
        R --> S["Shuffled Sparse Attention Layer ×3"]
        S --> T["Downampling"]
        T --> U["×9"]
        U --> V["Shuffled Sparse Attention Layer ×3"]
        V --> W["Downampling"]
        W --> X["×3"]
    end

    subgraph "Change map M"
        Y["Decoder"] --> Z["Change map M"]
    end

    subgraph "(b) Change-enhanced Features Fusion Module (CEFF)"
        AA["Change-enhanced Features Fusion Module (CEFF)"]
    end

    style A fill:#f0f0f0,stroke:#333
    style B fill:#e0e0e0,stroke:#333
    style C fill:#e0e0e0,stroke:#333
    style D fill:#e0e0e0,stroke:#333
    style E fill:#e0e0e0,stroke:#333
    style F fill:#e0e0e0,stroke:#333
    style G fill:#e0e0e0,stroke:#333
    style H fill:#e0e0e0,stroke:#333
    style I fill:#e0e0e0,stroke:#333
    style J fill:#e0e0e0,stroke:#333
    style K fill:#e0e0e0,stroke:#333
    style L fill:#e0e0e0,stroke:#333
    style M fill:#e0e0e0,stroke:#333
    style N fill:#e0e0e0,stroke:#333
    style O fill:#e0e0e0,stroke:#333
    style P fill:#e0e0e0,stroke:#333
    style Q fill:#e0e0e0,stroke:#333
    style R fill:#e0e0e0,stroke:#333
    style S fill:#e0e0e0,stroke:#333
    style T fill:#e0e0e0,stroke:#333
    style U fill:#e0e0e0,stroke:#333
    style V fill:#e0e0e0,stroke:#333
    style W fill:#e0e0e0,stroke:#333
    style X fill:#e0e0e0,stroke:#333
    style Y fill:#d4edda,stroke:#155724
    style Z fill:#d4edda,stroke:#155724
```
</details>

Figure 3. Overall architecture of our ScratchFormer framework for Change Detection. Our ScratchFormer takes two inputs, pre- and post-change images, and predicts a binary semantic change map for the corresponding image pair. ScratchFormer consists of a Siamesebased hierarchical encoder having four different stages, a change-enhanced feature fusion (CEFF) module, and a decoder for predicting binary change map. The focus of our design is the introduction of a shuffled sparse attention (SSA) layer (Sec. 3.3) in the encoder and a change-enhanced feature fusion (CEFF) module (Sec. 3.4). The SSA layer comprises shuffled sparse attention (SSA) and a MLP, as shown in (a). SSA performs token-mixing over a sparse data-dependent subset of features at each stage. Our ScratchFormer approach computes SSA features from the two streams $\hat { F } _ { p r e } ^ { i }$ and $\hat { F } _ { p o s t } ^ { i }$ at different scales i. The outputs of these stages are fused utilizing the CEFF module, as shown in (b). The CEFF module enhances the semantic changes between the features of the two streams by performing a per-channel re-weighting at each scale and outputs enhanced features $\hat { F } _ { e n h } ^ { i }$ . These enhanced features are then input to the decoder for predicting the final semantic binary change map M .

# 3.3. Shuffled Sparse Attention Layer

We introduce a shuffled sparse attention (SSA) layer within our encoder to capture semantic changes between the input image pairs $I _ { p r e }$ and $I _ { p o s t }$ .

As shown in Fig. 3-(a), it comprises a shuffled sparse attention (SSA) to perform token-mixing, a multi-layer perceptrons (MLP), and layer normalization layers. Our SSA performs token-mixing over a sparse subset of features which are selected based on a data-dependent sampling strategy. Let, $F ^ { i } \in \mathcal { R } ^ { H ^ { i } \times W ^ { i } \times C ^ { i } }$ be the encoder feature at stage i input to SSA. Then, our SSA is computed in two steps. First, we perform a data-dependant sparse sub-sampling of input features with a sparsity factor of $\gamma$ to obtain feature sub-sets $\bar { F } _ { k l } ^ { i } .$ Then, we separately perform self-attention over these $\gamma ^ { 2 }$ feature subsets $\bar { F } _ { k l } ^ { i }$ , where $k = \{ 0 , . . . , \gamma - 1 \}$ and $l ~ = ~ \{ 0 , . . . , \gamma ~ - ~ 1 \}$ . The datadependant sparse spatial sub-sampling of features is performed as follows:

$$
\bar {F} _ {k l} ^ {i} (\bar {x}, \bar {y}) = F ^ {i} (\gamma \bar {x} + k + \Delta x, \gamma \bar {y} + l + \Delta y)
$$

$$
\forall \bar {x} = \{0,..., \frac {H ^ {i}}{\gamma} - 1 \} \text {   and   } \forall \bar {y} = \{0,..., \frac {W ^ {i}}{\gamma} - 1 \} \tag {1}
$$

Here, $( \Delta x , \Delta y )$ , represents the data-dependent position offsets which are predicted using learnable parameters $\theta _ { o f f s e t }$ as $\Delta z = \theta _ { o f f s e t } ( F ^ { i } )$ . The predicted offsets $\Delta z \in \mathcal { R } ^ { H ^ { i } \times \mathbf { \bar { W } } ^ { i } \times 2 }$ have two channels depicting the horizontal and vertical position offsets at each pixel, which are clipped to limit the maximum distance from the current feature location. Then, the position offsets ∆x, $\Delta y$ are obtained as:

$$
\Delta x = \Delta z (\gamma \bar {x} + k, \gamma \bar {y} + l, 1)
$$

$$
\Delta y = \Delta z (\gamma \bar {x} + k, \gamma \bar {y} + l, 2)
$$

The resulting sparse-sampled features $\bar { F } _ { k l } ^ { i }$ are used to compute self-attention [19] (Attention(.)) over the $\gamma ^ { 2 }$ sparse windows as follows:

$$
\hat {F} _ {k l} ^ {i} = \text {Attention} (\bar {F} _ {k l} ^ {i}) \tag {2}
$$

These attended features $\hat { F } _ { k l } ^ { i }$ from $\gamma ^ { 2 }$ feature sub-sets are then shuffled back to full resolution feature map to obtain $\hat { F } ^ { i } \in \mathcal { R } ^ { H ^ { i } \times W ^ { i } \times C ^ { i } }$ . Here, the data-dependent position offsets aids to adaptively sample dense features from regions likely having semantic changes, whereas the sparse sampling helps to efficiently maintain the global receptive field. Due to the sparse sampling, we perform $\gamma ^ { 2 }$ self-attention operations and in each self-attention operation the number of tokens are reduced by a factor of $\gamma ^ { 2 }$ , leading to a $O ( \gamma ^ { 2 } )$ reduction in the overall computation. Our SSA enables faster convergence due to its sparse structure allowing self attention to focus on the sub-sampled relevant features. Our proposed ScratchFormer approach employs SSA layers at each stage of the encoder and computes pre- and post- change features $\hat { F } _ { p r e } ^ { i } , \ : \hat { F } _ { p o s t } ^ { i }$ , from both streams of the encoder. These features are then fused by the changeenhanced feature fusion module described next.

# 3.4. Change Enhanced Features Fusion Module

As discussed earlier, given the diverse nature of the changes in real-world scenarios that can possibly occur in the image pairs, detecting high-level semantic changes while ignoring the noisy ones is one of the major challenges in the CD task. Therefore, it is desired to effectively fuse the features from pre- and post-change feature streams of the encoder. Within several existing transformers-based CD methods [2, 1, 22], multi-level feature fusion between preand post change features is performed through difference, summation or concatenation operations. Similarly, the base framework also introduces a difference module employing concatenation across channel dimension for the feature fusion. We argue that such a fusion of the features from both streams without explicitly re-weighting the channels from each stage is sub-optimal for the CD task. To this end, we introduce a change-enhanced feature fusion module (CEFF) that performs per-channel re-weighting to enhance the channels having higher semantic changes, while suppressing the channels capturing noisy changes.

Fig 3-(b) shows the structure of our change-enhanced feature fusion module (CEFF). The CEFF module is introduced at all four stages of the encoder to fuse the features at each stage. In our CEFF module, we first combine the pre- and post- change features Fˆipr $\hat { F } _ { p r e } ^ { i } , \hat { F } _ { p o s t } ^ { i }$ through addition, and then perform global average pooling (GAP ) to obtain a global feature vector $\mathbf { p } ^ { i }$ as follows:

$$
\mathbf {p} ^ {i} = G A P (\hat {F} _ {p r e} ^ {i} + \hat {F} _ {p o s t} ^ {i}), \tag {3}
$$

We input $\mathbf { p } ^ { i }$ feature vector to shared Conv-ReLU layers to reduce the number of channels. Afterwards, these reduced features are passed to separate $1 \times 1$ conv layers to obtain the channel weights for both streams $\mathbf { v } _ { 1 } ^ { i } , \mathbf { v } _ { 2 } ^ { i }$ as follows:

$$
\bar {\mathbf {p}} ^ {i} = \varphi (\omega_ {1} (\mathbf {p} ^ {i})),
$$

$$
\mathbf {v} _ {1} ^ {i} = \omega_ {2} (\bar {\mathbf {p}} ^ {i}), \mathbf {v} _ {2} ^ {i} = \omega_ {3} (\bar {\mathbf {p}} ^ {i}), \tag {4}
$$

where, $\omega _ { 1 } , \omega _ { 2 }$ , and $\omega _ { 3 }$ are the convolutional weights, and $\varphi$ represents the ReLU activation function. Here, ${ \bf v } _ { 1 } ^ { i } \in  { }$ $\mathcal { R } ^ { C ^ { i } \times 1 }$ , and $\mathbf { v } _ { 2 } ^ { i } \in \mathcal { R } ^ { C ^ { i } \times 1 }$ refers to the un-normalized channels re-weighting factors predicted for the pre- and postchange features at stage $i .$ These un-normalized weights are then normalized by per-channel softmax across both streams. i.e,

$$
\hat {\mathbf {v}} _ {1} ^ {i} (j) = \frac {\exp (\mathbf {v} _ {1} ^ {i} (j))}{\exp (\mathbf {v} _ {1} ^ {i} (j)) + \exp (\mathbf {v} _ {2} ^ {i} (j))}
$$

$$
\hat {\mathbf {v}} _ {2} ^ {i} (j) = \frac {\exp \left(\mathbf {v} _ {2} ^ {i} (j)\right)}{\exp \left(\mathbf {v} _ {1} ^ {i} (j)\right) + \exp \left(\mathbf {v} _ {2} ^ {i} (j)\right)} \tag {5}
$$

$$
\forall j = \{1, \dots , C ^ {i} \}
$$

where, j is the channel index and exp denotes the exponential function. These normalized weights $\hat { \mathbf { v } } _ { 1 } ^ { i } \in \mathcal { R } ^ { C ^ { i } \times 1 }$ and $\hat { \mathbf { v } } _ { 2 } ^ { i } \in \mathcal { R } ^ { C ^ { i } \times 1 }$ are used to perform channel re-weighting of $\hat { F } _ { p r e } ^ { i }$ e and Fˆipo $\hat { F } _ { p o s t } ^ { i }$ followed by feature fusion through addition to generate the enhanced features $\hat { F } _ { e n h } ^ { i }$ Fenh as:

$$
\hat {F} _ {e n h} ^ {i} = \hat {\mathbf {v}} _ {1} ^ {i} \hat {F} _ {p r e} ^ {i} + \hat {\mathbf {v}} _ {2} ^ {i} \hat {F} _ {p o s t} ^ {i}, \tag {6}
$$

The resulting enhanced features from the CEFF module at all stages are then resized to a fixed spatial resolution and passed to the decoder that performs feature upsampling and change map prediction.

# 4. Experiments

# 4.1. Experimental Setup

Datasets: The large-scale LEVIR-CD [3]: dataset is for building change detection. It contains 637 high resolution (0.5m per pixel) image pairs taken from Google Earth with size of 1024x1024. In our experiments, we use the nonoverlapping cropped patches of 256x256, having default data split of train, validation, and test equal to 7120, 1024, and 2048, respectively. The DSIFN-CD [24]: dataset is for binary change detection and contains six high-resolution (2m) satellite image pairs from six cities of China. We used the cropped version of the dataset having image size of 256x256 resulting in train, validation and test set of size 14400, 1360, and 192 image pairs, respectively. The CDD-CD [14]: dataset comprises 11 seasonal varying image pairs including, 7 image pairs of size 4725x2700 pixels and 4 image pairs of size 1900x1000. The image pairs are clipped into 256x256 with data split of 10000, 3000, and 3000 for train, validation, and test set, respectively. The WHU-CD [10]: dataset is for building-related change detection and consists of one high-resolution (0.075m) image pair of size 32507x15354 pixels. This aerial dataset contains a variety of building architectures of different sizes and colors. The dataset is also available with image pairs of size 256x256 pixels having non-overlapping regions and data split of 5947, 743, and 744 image pairs for train, validation and test set, respectively.

Table 1. State-of-the-art comparison on DSIFN-CD, LEVIR-CD, WHU-CD, and CDD-CD datasets. For each dataset, we report the results in terms of F1, IoU and OA metrics. Our ScratchFormer approach that is trained from scratch and does not require any pre-training performs favorably against existing methods and achieves state-of-the-art performance. The best two results are in red and blue, respectively. 

<table><tr><td rowspan="2">Method</td><td colspan="3">DSIFN-CD</td><td colspan="3">LEVIR-CD</td><td colspan="3">WHU-CD</td><td colspan="3">CDD-CD</td></tr><tr><td>F1</td><td>OA</td><td>IoU</td><td>F1</td><td>OA</td><td>IoU</td><td>F1</td><td>OA</td><td>IoU</td><td>F1</td><td>OA</td><td>IoU</td></tr><tr><td>TransUNetCD [15]</td><td>66.62</td><td>-</td><td>57.95</td><td>91.11</td><td>-</td><td>83.67</td><td>93.59</td><td>-</td><td>84.42</td><td>97.17</td><td>-</td><td>94.50</td></tr><tr><td>DASNet [4]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>92.70</td><td>98.2</td><td>-</td></tr><tr><td>H-TransCD [12]</td><td>-</td><td>-</td><td>-</td><td>90.06</td><td>99.00</td><td>81.92</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>STANet [3]</td><td>-</td><td>-</td><td>-</td><td>87.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SNUNet [6]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>83.4</td><td>-</td><td>-</td></tr><tr><td>BIT [2]</td><td>69.26</td><td>89.41</td><td>52.97</td><td>89.31</td><td>98.92</td><td>80.68</td><td>83.98</td><td>98.75</td><td>72.39</td><td>-</td><td>-</td><td>-</td></tr><tr><td>IFNet [24]</td><td>67.33</td><td>88.86</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>90.30</td><td>97.71</td><td>-</td></tr><tr><td>MSTDSNet [18]</td><td>-</td><td>-</td><td>-</td><td>88.10</td><td>-</td><td>78.73</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ChangeFormer [1]</td><td>86.67</td><td>95.56</td><td>76.48</td><td>90.40</td><td>99.04</td><td>82.48</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ScratchFormer (ours)</td><td>95.15</td><td>98.36</td><td>90.75</td><td>91.78</td><td>99.17</td><td>84.81</td><td>92.16</td><td>99.40</td><td>85.46</td><td>98.14</td><td>99.56</td><td>96.34</td></tr></table>

Evaluation Protocol: Following [1], we evaluate change detection (CD) results in terms of change class F1-score, change class Intersection over Union (IoU) and overall accuracy (OA) on all the datasets. Among these evaluation metrics, the change class IoU is the most challenging metric for the CD task.

Implementation Details: Our ScratchFormer takes a pair of images of size $2 5 6 \times 2 5 6 \times 3 ,$ resizes the input images to $5 1 2 \times 5 1 2 \times 3$ and computes the features for the two streams at four stages (having 3, 3, 9, and 3 SSA layers), which outputs the features with 64, 128, 320, and 512 channels, respectively. In the proposed SSA, the sparsity factor is calculated as $\gamma = 2 ^ { n }$ , where $n > 0 .$ . The decoder outputs the prediction maps with two channels having a spatial resolution of $5 1 2 \times 5 1 2$ which is resized to 256 × 256 to compute the metric scores. The model is trained using pixel-wise cross-entropy loss function. During training, we employ standard data augmentations including, random scale crop, Gaussian blur, random flip, random re-scale, and random color jitter. We train our network using random initialization on 4 NVIDIA A100 GPUs. Following [1], we use the AdamW optimizer with a weight decay 0.01 and beta values equal to (0.9, 0.999). We set the batch size 16, initial learning rate to 4.1e-4, and train for 300 epochs. In our experiments, we used linear decay to decrease the learning rate till the last epoch. At the inference stage, we freeze the model weights, resize the input image pair of sizes $2 5 6 \times 2 5 6 \times 3$ to $5 1 2 \times 5 1 2 \times 3$ and pass to our ScratchFormer model to get the $5 1 2 \times 5 1 2 \times 2$ prediction maps which are resized back to original spatial resolution of 256 × 256. The binary change mask M is computed using a pixel-wise argmax operation along the channel dimension. A well-documented code along with trained models will be publicly released.

# 4.2. State-of-the-art Comparison

Comparison on DSIFN-CD: We compare our approach with both CNN-based and transformer-based state-of-theart methods over DSIFN-CD. Tab. 1 presents the results. The CNN-based IFNet [24] obtains F1 score of 67.33%. We observe that recent transformer-based methods achieve better IoU performance. For instance, BIT [2], TransUNetCD [15], and ChangeFormer [1] achieve IoU scores of 52.97%, 57.95%, and 76.48%, respectively. Our Scratch-Former outperforms these recent methods and achieves consistent improvements in terms of all metrics. Notably, our ScratchFormer achieves absolute gains of 8.48%, 2.8%, and 14.27% in terms of F1, OA, and IoU compared to Change-Former [1]. It is worth mentioning that our approach here is trained from scratch without using any pre-training on another CD dataset. On this dataset, ScratchFormer sets a new state-of-the-art performance with a significant gain obtained in the challenging F1 and IoU metrics.

Comparison on LEVIR-CD: Here, we present the stateof-the-art comparison on the LEVIR-CD dataset (Tab. 1). Among recent transformer-based CD methods, H-TransCD [12], BIT [2], and ChangeFormer [1], obtain IoU scores of 81.92%, 80.68%, and 82.48%, respectively. Our Scratch-Former obtains an IoU score of 84.81% with an absolute gain of 2.33% over the recently published method in literature ChangeFormer [1].

Comparison on WHU-CD:Here, we present the stateof-the-art comparison on the WHU-CD dataset (Tab. 1). Among existing transformer-based methods, BIT [2] and TransUNetCD [15] achieve IoU score of 72.39% and 84.42%, respectively. In comparison, our ScratchFormer that is trained from scratch through random initialization on this dataset achieves favorable performance against existing methods with an IoU score of 85.46%.

![](images/d31c51af84bb60b2f79af0fa7f62c11582256e6033ed22bf14eba7bebee31286.jpg)  
Figure 4. Qualitative comparison on DSFIN-CD. we compare our ScratchFormer with BIT [2] and ChangeFormer [1]. We observe Scratch-Former to better detect the semantic changes (marked in red box) with clear boundaries between the pre- and post-change images, compared to other methods. Additional results are presented in the supplementary.

![](images/ebc2ad600945cb20e8dffdbc5f27150eb2d5503d1db5ce6d37f3ddb99c1602cf.jpg)  
Figure 5. Qualitative comparison on LEVIR-CD. We compare our ScratchFormer with BIT [2] and ChangeFormer [1]. Our ScratchFormer provides improved CD performance by accurately detecting the relevant changes (marked in red box) with clear boundaries, compared to existing methods. Additional results are presented in the supplementary.

Comparison on CDD-CD: Lastly, we report results (Tab. 1) on the CDD-CD dataset. Among CNN-based approaches, DASNet [4] achieves F1 score of 92.70%. Among transformer-based methods, TransUNetCD [15] achieves IoU score of 94.50%, which achieves this performance by employing improved ResNet50 backbone. In comparison, our ScratchFormer trained from scratch achieves an IoU score of 96.34%.

Qualitative Comparison: Fig. 4 and 5 present qualitative comparisons of our ScratchFormer with BIT [2] and ChangeFormer [1] on DSIFN-CD and LEVIR-CD datasets, respectively. We observe that our ScratchFormer is able

to accurately detect changes occurring at multiple scales in these complex example scenes. For instance, in Fig. 4 (first row) and Fig. 5 (second row), our method better localizes the semantic changed regions, compared to existing methods, which demonstrates the efficacy of our method due to the employment of our novel shuffled sparse attention.

# 4.3. Ablation Study

Here, we present ablation study to validate the effectiveness of our contributions over DSIFN-CD and LEVIR-CD dataset. Tab. 2 shows the baseline comparison. The baseline approach (Sec. 2.1) when trained from scratch using random initialization achieves IoU score of 78.11% (row4) over DSIFN-CD dataset. The results of the baseline approach are improved to 83.94% (row 3) when first pre-training it on LEVIR-CD and then finetuning it on the DSFIN-CD (target) dataset. When integrating our SSA layer (3.3) into the baseline, the results are significantly improved to 86.62% in terms of IoU score (row 5). Our final ScratchFormer which includes both contributions (SSA and CEFF) and trained from scratch leads to a significant improvement in performance by achieving an IoU score of 90.75%. These results demonstrate the effectiveness of our contributions. Similarly, our method obtains a consistent gain by introducing the proposed SSA and CEFF to the baseline over the LEVIR-CD dataset. In addition to the baseline comparison, we also report the results of Change-Former using both pre-training and training from scratch. Our ScratchFormer achieves consistent gain in performance on all three metrics over the ChangeFormer.

Table 2. Ablation study on the DSIFN and LEVIR datasets. Here, we show the impact of integrating our contributions to the baseline. † denotes that the model is pre-trained first on another CD dataset and then finetuned on the target CD dataset. The integration of our SSA (row 5) into the baseline (row 4) leads to consistent gain in performance. Our final approach ScratchFormer (row 6) that comprises both SSA and CEFF achieves a significant improvement in performance over the baseline. Here, we also report ChangeFormer with and without pre-training. Best two results are in red and blue, respectively. 

<table><tr><td rowspan="2">Method</td><td colspan="3">DSIFN-CD</td><td colspan="3">LEVIR-CD</td></tr><tr><td>F1</td><td>OA</td><td>IoU</td><td>F1</td><td>OA</td><td>IoU</td></tr><tr><td>ChangeFormer [1] †</td><td>86.67</td><td>95.56</td><td>76.48</td><td>90.40</td><td>99.04</td><td>82.48</td></tr><tr><td>ChangeFormer [1]</td><td>81.24</td><td>93.54</td><td>68.41</td><td>84.97</td><td>98.52</td><td>73.86</td></tr><tr><td>Baseline †</td><td>91.27</td><td>97.07</td><td>83.94</td><td>90.84</td><td>99.08</td><td>83.01</td></tr><tr><td>Baseline</td><td>87.71</td><td>95.79</td><td>78.11</td><td>90.68</td><td>99.07</td><td>82.95</td></tr><tr><td>Baseline + SSA (Sec. 3.3)</td><td>92.83</td><td>97.58</td><td>86.62</td><td>91.18</td><td>99.12</td><td>83.79</td></tr><tr><td>Baseline + SSA+ CEFF (ScratchFormer)</td><td>95.15</td><td>98.36</td><td>90.75</td><td>91.78</td><td>99.17</td><td>84.81</td></tr></table>

Table 3. Comparison of the sparsity γ over DSIFN and LEVIR datasets. The sparsity γ = 4 achieves superior performance. The best results are in bold. 

<table><tr><td rowspan="2">γ</td><td colspan="3">DSIFN-CD</td><td colspan="3">LEVIR-CD</td></tr><tr><td>F1</td><td>OA</td><td>IoU</td><td>F1</td><td>OA</td><td>IoU</td></tr><tr><td>γ=2</td><td>93.91</td><td>97.95</td><td>88.53</td><td>91.55</td><td>99.15</td><td>84.42</td></tr><tr><td>γ=4</td><td>95.15</td><td>98.36</td><td>90.75</td><td>91.78</td><td>99.17</td><td>84.81</td></tr><tr><td>γ=8</td><td>94.34</td><td>98.10</td><td>89.29</td><td>91.68</td><td>99.16</td><td>84.64</td></tr></table>

We also conduct an experiment to estimate the optimal sparsity of our SSA by varying the sparsity factor γ (2, 4, and 8) as shown in Table 3. We observe setting the value of γ to 4 to provide optimal performance for DSIFN and LEVIR datasets. Therefore, we fix the γ and use the same value throughout our experiments. We further perform an experiment to compare our CEFF module with standard addition, subtraction, and concatenation based techniques. Here, addition, subtraction, and concatenation are performed for $\hat { F } _ { p r e } ^ { i }$ pre and $\hat { F } _ { p o s t } ^ { i }$ Fˆipost, and passed to two convolutional layers. Tab. 4 shows the comparison. Our CEFF that utilizes feature channel re-weighting achieves superior performance compared to these techniques.

# 5. Relation to Prior Art

CNN-based Approaches: Convolutional neural networks have attained much popularity in remote sensing change detection due to intrinsic properties to capture discriminative features [17]. Chen et al. [4] propose a dual attention mechanism within Siamese CNN to encode long-range dependencies. Fang et al. [6] propose adense Siamese network that uses channel attention to refine features. A feature pyramid with attention mechanism is proposed to encode longrange dependencies in [11]. Liu et al. [16] use multi-scale convolutional attention features to learn the bitemporal feature differences via adversarial learning. Hou et al. [9] employ low rank analysis to benefit from deep features for CD. Chen et al. [3] introduce Siamese-based network to capture spatial–temporal dependencies for CD. Zhang et al. [24] propose a deep supervised image fusion network for CD.

Table 4. Comparison of CEFF with the subtraction, addition, and concatenation based techniques on DSFIN-CD. CEFF achieves superior performance on all metrics. Best two results are in red and blue, respectively. 

<table><tr><td>Method</td><td>F1</td><td>OA</td><td>IoU</td></tr><tr><td>Difference module with Subtraction</td><td>80.23</td><td>88.08</td><td>68.89</td></tr><tr><td>Difference module with Addition</td><td>93.00</td><td>87.34</td><td>78.53</td></tr><tr><td>Difference module with Concatenation</td><td>92.83</td><td>97.58</td><td>86.62</td></tr><tr><td>CEFF</td><td>95.15</td><td>98.36</td><td>90.75</td></tr></table>

Transformers-based Approaches: Recently, transformers [19] have gain popularity for the CD task. Chen et al. [2] introduce a bitemporal image transformer (BIT) to model context information. Li et al. [15] introduce TransUNetCD, which benefits from both transformers and UNet for CD. Song et al. [18] propose a multi-scale Swin transformer that uses refined multi-scale features. Ke et al. [12] propose a hybrid transformer to capture global context dependencies at multiple scales. Bandara et al. [1] propose a hierarchical Siamese transformer to render multi-scale features. Different to existing approaches, we introduce a SSA to effectively capture the inductive CD bias when training from scratch on any change detection (target) dataset. Further, a CEFF module to perform per-channel re-weighting to enhance the feature channels having higher semantic changes, while suppressing the channels encoding noisy changes.

# 6. Conclusion

We propose an transformers-based Siamese architecture, named ScratchFormer, for the problem of remote sensing change detection (CD). Our ScratchFormer introduces a shuffled sparse attention (SSA) to effectively capture the inherent characteristics when training from scratch. We further introduce a change-enhanced feature fusion (CEFF) module to perform per-channel feature weighting to enhance the relevant semantic changes, while suppressing the noisy ones. We validate our approach by conducting extensive experiments on four challenging CD datasets. Our extensive qualitative and quantitative results reveal the benefits of the proposed contributions. The proposed Scratch-Former approach achieves state-of-the-art CD performance on all four CD datasets. A potential future direction is to explore the problem of change detection in natural images and medical imaging.

# References

[1] Wele Gedara Chaminda Bandara and Vishal M Patel. A transformer-based siamese network for change detection. arXiv preprint arXiv:2201.01293, 2022. 1, 2, 3, 5, 6, 7, 8   
[2] Hao Chen, Zipeng Qi, and Zhenwei Shi. Remote sensing image change detection with transformers. IEEE Transactions on Geoscience and Remote Sensing, 60:1–14, 2021. 1, 5, 6, 7, 8   
[3] Hao Chen and Zhenwei Shi. A spatial-temporal attentionbased method and a new dataset for remote sensing image change detection. Remote Sensing, 12(10):1662, 2020. 1, 2, 5, 6, 8   
[4] Jie Chen, Ziyang Yuan, Jian Peng, Li Chen, Haozhe Huang, Jiawei Zhu, Yu Liu, and Haifeng Li. Dasnet: Dual attentive fully convolutional siamese networks for change detection in high-resolution satellite images. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 14:1194–1206, 2020. 1, 6, 7, 8   
[5] Rodrigo Caye Daudt, Bertr Le Saux, and Alexandre Boulch. Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP), pages 4063–4067. IEEE, 2018. 1   
[6] Sheng Fang, Kaiyu Li, Jinyuan Shao, and Zhe Li. Snunet-cd: A densely connected siamese network for change detection of vhr images. IEEE Geoscience and Remote Sensing Letters, 19:1–5, 2021. 6, 8   
[7] Leila MG Fonseca, Thales S Korting, Hugo do N Ben- ¨ dini, Cesare D Girolamo-Neto, Alana K Neves, Anderson R Soares, Evandro C Taquary, and Raian V Maretto. Pattern recognition and remote sensing techniques applied to land use and land cover mapping in the brazilian savannah. Pattern recognition letters, 148:54–60, 2021. 1   
[8] Enqiang Guo, Xinsha Fu, Jiawei Zhu, Min Deng, Yu Liu, Qing Zhu, and Haifeng Li. Learning to measure change: Fully convolutional siamese metric networks for scene change detection. arXiv preprint arXiv:1810.09111, 2018. 1   
[9] Bin Hou, Yunhong Wang, and Qingjie Liu. Change detection based on deep features and low rank. IEEE Geoscience and Remote Sensing Letters, 14(12):2418–2422, 2017. 8   
[10] Shunping Ji, Shiqing Wei, and Meng Lu. Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set. IEEE Transactions on Geoscience and Remote Sensing, 57(1):574–586, 2019. 2, 6   
[11] Huiwei Jiang, Xiangyun Hu, Kun Li, Jinming Zhang, Jinqi Gong, and Mi Zhang. Pga-siamnet: Pyramid feature-based attention-guided siamese network for remote sensing orthoimagery building change detection. Remote Sensing, 12(3):484, 2020. 8   
[12] Qingtian Ke and Peng Zhang. Hybrid-transcd: A hybrid transformer remote sensing image change detection network via token aggregation. ISPRS International Journal of Geo-Information, 11(4):263, 2022. 1, 6, 8

[13] Maja Kucharczyk and Chris H Hugenholtz. Remote sensing of natural hazard-related disasters with small drones: Global trends, biases, and research opportunities. Remote Sensing of Environment, 264:112577, 2021. 1   
[14] Maxim Lebedev, Yu. V. Vizilter, Oleg Vygolov, Vladimir A. Knyaz, and A. Yu. Rubis. Change detection in remote sensing images using conditional adversarial networks. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2018. 2, 6   
[15] Qingyang Li, Ruofei Zhong, Xin Du, and Yu Du. Transunetcd: A hybrid transformer network for change detection in optical remote-sensing images. IEEE Transactions on Geoscience and Remote Sensing, 60:1–19, 2022. 1, 6, 7, 8   
[16] Mengxi Liu, Qian Shi, Andrea Marinoni, Da He, Xiaoping Liu, and Liangpei Zhang. Super-resolution-based change detection network with stacked attention module for images with different resolutions. IEEE Transactions on Geoscience and Remote Sensing, 60:1–18, 2021. 8   
[17] Wenzhong Shi, Min Zhang, Rui Zhang, Shanxiong Chen, and Zhao Zhan. Change detection based on artificial intelligence: State-of-the-art and challenges. Remote Sensing, 12(10):1688, 2020. 1, 8   
[18] Fei Song, Sanxing Zhang, Tao Lei, Yixuan Song, and Zhenming Peng. Mstdsnet-cd: Multiscale swin transformer and deeply supervised network for change detection of the fastgrowing urban regions. IEEE Geoscience and Remote Sensing Letters, 19:1–5, 2022. 1, 6, 8   
[19] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 1, 2, 5, 8   
[20] Guanghui Wang, Bin Li, Tao Zhang, and Shubi Zhang. A network combining a transformer and a convolutional neural network for remote sensing image change detection. Remote Sensing, 14(9):2228, 2022. 1   
[21] Dawei Wen, Xin Huang, Francesca Bovolo, Jiayi Li, Xinli Ke, Anlu Zhang, and Jon Atli Benediktsson. Change detection from very-high-spatial-resolution optical remote sensing images: Methods, applications, and future directions. IEEE Geoscience and Remote Sensing Magazine, 9(4):68– 101, 2021. 1   
[22] Tianyu Yan, Zifu Wan, and Pingping Zhang. Fully transformer network for change detection of remote sensing images. arXiv preprint arXiv:2210.00757, 2022. 5   
[23] Jiadi Yin, Jinwei Dong, Nicholas AS Hamm, Zhichao Li, Jianghao Wang, Hanfa Xing, and Ping Fu. Integrating remote sensing and geospatial big data for urban land use mapping: A review. International Journal of Applied Earth Observation and Geoinformation, 103:102514, 2021. 1   
[24] Chenxiao Zhang, Peng Yue, Deodato Tapete, Liangcun Jiang, Boyi Shangguan, Li Huang, and Guangchao Liu. A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images. ISPRS Journal of Photogrammetry and Remote Sensing, 166:183–200, 2020. 1, 2, 5, 6, 8   
[25] Mengya Zhang, Guangluan Xu, Keming Chen, Menglong Yan, and Xian Sun. Triplet-based semantic relation learning

for aerial remote sensing image change detection. IEEE Geoscience and Remote Sensing Letters, 16(2):266–270, 2018. 1