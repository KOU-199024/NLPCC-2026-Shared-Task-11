# nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer

Hong-Yu Zhou, Student Member, IEEE, Jiansen Guo, Yinghao Zhang, Xiaoguang Han, Lequan Yu, Liansheng Wang, Member, IEEE, and Yizhou Yu, Fellow, IEEE

Abstract— Transformer, the model of choice for natural language processing, has drawn scant attention from the medical imaging community. Given the ability to exploit long-term dependencies, transformers are promising to help atypical convolutional neural networks to overcome their inherent shortcomings of spatial inductive bias. However, most of recently proposed transformer-based segmentation approaches simply treated transformers as assisted modules to help encode global context into convolutional representations. To address this issue, we introduce nnFormer (i.e., not-another transFormer), a 3D transformer for volumetric medical image segmentation. nnFormer not only exploits the combination of interleaved convolution and self-attention operations, but also introduces local and global volume-based self-attention mechanism to learn volume representations. Moreover, nnFormer proposes to use skip attention to replace the traditional concatenation/summation operations in skip connections in U-Net like architecture. Experiments show that nnFormer significantly outperforms previous transformer-based counterparts by large margins on three public datasets. Compared to nnUNet, nnFormer produces significantly lower HD95 and comparable DSC results. Furthermore, we show that nnFormer and nnUNet are highly complementary to each other in model ensembling. Codes and models of nnFormer are available at https://git.io/JSf3i.

Index Terms— Transformer, Attention Mechanism, Volumetric Image Segmentation

# I. INTRODUCTION

Transformer [1], which has become the de-facto choice for natural language processing (NLP) problems, has recently been widely exploited in vision-based applications [2]–[5]. The core idea behind is to apply the self-attention mechanism

(Corresponding author: Liansheng Wang and Yizhou Yu.)

This work was done when Hong-Yu Zhou was a visiting student at Xiamen University.

Hong-Yu Zhou, Jiansen Guo, Yinghao Zhang and Liansheng Wang are with the Department of Computer Science, Xiamen University, Siming District, Xiamen, Fujian Province, P.R. China (email: whuzhouhongyu@gmail.com, jsguo@stu.xmu.edu.cn, zhangyinghao@stu.xmu.edu.cn, lswang@xmu.edu.cn).

Hong-Yu Zhou and Yizhou Yu are with the Department of Computer Science, The University of Hong Kong, Pokfulam, Hong Kong (e-mail: yizhouy@acm.org).

Xiaoguang Han is with the Shenzhen Research Institute of Big Data, The Chinese University of Hong Kong (Shenzhen), Shenzhen, Guangdong Province, P.R. China (email: hanxiaoguang@cuhk.edu.cn).

Lequan Yu is with the Department of Statistics and Actuarial Science, The University of Hong Kong, Pokfulam, Hong Kong (e-mail: lqyu@hku.hk).

First two authors contributed equally.

to capture long-range dependencies. Compared to convolutional neural networks (i.e., convnets [6]), transformer relaxes the inductive bias of locality, making it more capable of dealing with non-local interactions [7]–[9]. It has also been investigated that the prediction errors of transformers are more consistent with those of humans than convnets [10].

Given the fact that transformers are naturally more advantageous than convnets, there are a number of approaches trying to apply transformers to the field of medical image analysis. Chen et al. [11] first time proposed TransUNet to explore the potential of transformers in the context of medical image segmentation. The overall architecture of TransUNet is similar to that of U-Net [12], where convnets act as feature extractors and transformers help encode the global context. In fact, one major characteristic of TransUNet and most of its followers [13]–[16] is to treat convnets as main bodies, on top of which transformers are further applied to capture long-term dependencies. However, such feature may cause a problem, which is the advantages of transformers are not fully exploited. In other words, we believe one- or two-layer transformers are not enough to entangle long-term dependencies with convolutional representations that often contain precise spatial information and provide hierarchical concepts.

To address the above issue, some researchers [17]–[19] started to use transformers as the main stem in segmentation models. Karimi et al. [17] first time introduced a convolution-free segmentation model by forwarding flattened image representations to transformers, whose outputs are then reorganized into 3D tensors to align with segmentation masks. Recently, Swin Transformer [3] showed that by referring to the feature pyramids used in convnets, transformers can learn hierarchical object concepts at different scales by applying appropriate down-sampling to feature maps. Inspired by this idea, SwinUNet [18] utilized hierarchical transformer blocks to construct the encoder and decoder within a U-Net like architecture, based on which DS-TransUNet [19] added one more encoder to accept different-sized inputs. Both SwinUNet and DS-TransUNet have achieved consistent improvements over TransUNet. Nonetheless, they did not explore how to appropriately combine convolution and self-attention for building an optimal medical segmentation network.

In contrast, nnFormer (i.e., not-another transFormer) uses a hybrid stem where convolution and self-attention are interleaved to give full play to their strengths. Figure 1 presents the effects of different components used in the encoder of nn-

![](images/a2e7a1bcd1c876c640469f58899c304a18ed0b5d9de7ba8e836fbc714c55f2e7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1. Precise spatial encoding."] --> B["2. High-resolution low-level features"]
    B --> C["Incorporating long-term dependencies into high-level features."]
    C --> D["3. Modeling object concepts from high-level features at multiple scales."]
    D --> E["4. Convolutional down-sampling"]
    E --> F["5. Transformer blocks"]
    F --> G["6. Convolutional embedding"]
    G --> H["7. Transformer blocks"]
```
</details>

Fig. 1: The interleaved stem used in the encoder of nnFormer.

Former. Firstly, we put a light-weight convolutional embedding layer ahead of transformer blocks. In comparison to directly flattening raw pixels and applying 1D pre-processing in [17], the convolutional embedding layer encodes precise (i.e., pixellevel) spatial information and provides low-level yet highresolution 3D features. After the embedding block, transformer and convolutional down-sampling blocks are interleaved to fully entangle long-term dependencies with high-level and hierarchical object concepts at various scales, which helps improve the generalization ability and robustness of learned representations.

The other contribution of nnFormer lies in proposing a computational-efficient way to leverage inter-slice dependencies. To be specific, nnFormer proposes to jointly use Local Volume-based Multi-head Self-attention (LV-MSA) and Global Volume-based Multi-head Self-attention (GV-MSA) to construct feature pyramids and provide sufficient receptive field for learning representations on both local and global 3D volumes, which are then aggregated to make predictions. Compared to the naive multi-head self-attention (MSA) [1], the proposed strategy can greatly reduce the computational complexity while producing competitive segmentation performance. Moreover, inspired by the attention mechanism used in the task of machine translation [1], we introduce skip attention to replace the atypical concatenation/summation operation in skip connections of U-Net like architecture, which further improves the segmentation results.

To sum up, our contributions can be summarized as follows:

• We introduce nnFormer, a 3D transformer for volumetric medical image segmentation. nnFormer achieves significant improvements over previous transformer-based medical segmentation models on three well-established datasets.   
• Technically, the contributions of nnFormer are three folds: i) an interleaved combination of convolution and selfattention operations. ii) the utilization of both local and global volume-based self-attention to build feature pyramids and provide large receptive fields, respectively. iii) skip attention is proposed to replace traditional concatenation/summation operations in skip connections.   
• Thorough experiments have been conducted to validate the advantages of nnFormer over nnUNet. We show that nnFormer is significantly better than nnUNet in hausdorff distance and achieves slightly better performance in dice coefficient. Moreover, we found that nnFormer and nnUNet are highly complementary to each other as simply averaging their predictions can already greatly boost the overall performance.

# II. RELATED WORK

In this section, we mainly review methodologies that resort to transformers to improve segmentation results of medical images. Since most of them employ hybrid architecture of convolution and self-attention [1], we divide them into two categories based on whether the majority of the stem is convolutional or transformer-based.

Convolution-based stem. TransUNet [11] first time applied transformer to improve the segmentation results of medical images. TransUNet treats the convnet as a feature extractor to generate a feature map for the input slice. Patch embedding is then applied to patches of feature maps in the bottleneck instead of raw images in ViT [2]. Concurrently, similar to TransUNet, Li et al. [20] proposed to use a squeezed attention block to regularize the self-attention modules of transformers and an expansion block to learn diversified representations for fundus images, which are all implemented in the bottleneck within convnets. TransFuse [13] introduced a BiFusion module to fuse features from the shallow convnet-based encoder and transformer-based segmentation network to make final predictions on 2D images. Compared to TransUNet, TransFuse mainly applied the self-attention mechanism to the input embedding layer to improve segmentation models on 2D images. Yun et al. [21] employed transformers to incorporate spectral information, which are entangled with spectral information encoded by convolutional features to address the problem of hyperspectral pathology. Xu et al. [22] extensively studied the trade-off between transformers and convnets and proposed a more efficient encoder named LeViT-UNet. Li et al. [23] presented a new up-sampling approach and incorporated it into the decoder of UNet to model long-term dependencies and global information for better reconstruction results. TransClaw U-Net [15] utilized transformers in UNet with more convolutional feature pyramids. TransAttUNet [16] explored the feasibility of applying transformer self attention with convolutional global spatial attention. Xie et al. [24] adopted transformers to capture long-term dependencies of multi-scale convolutional features from different layers of convnets. TransBTS [25] first utilized 3D convnets to extract volumetric spatial features and down-sample the input 3D images to produce hierarchical representations. The outputs of the encoder in TransBTS are then reshaped into a vector (i.e. token) and fed into transformers for global feature modeling, after which an ordinary convolutional decoder is appended to up-sample feature maps for the goal of reconstruction. Different from these approaches that directly employ convnets as feature extractors, our nnFormer functionally relies on convolutional and transformer-based blocks, which are interleaved to take advantages of each other.

Transformer-based stem. Valanarasu et al. [14] proposed a gated axial-attention model (i.e., MedT) which extends the existing convnet architecture by introducing an summational control mechanism in the self-attention. Karimi et al. [17] removed the convolutional operations and built a 3D segmentation model based on transformers. The main idea is to first split the local volume block into 3D patches, which are then flattened and embedded to 1D sequences and passed to a ViT-like backbone to extract representations. SwinUNet [18] built a U-shape transformer-based segmentation model on top of transformer blocks in [3], where observable improvements were achieved. DS-TransUNet [19] further extended Swin-UNet by adding one more encoder to handle multi-scale inputs and introduced a fusion module to effectively establish global dependencies between features of different scales through the self-attention mechanism. Compared to these transformerbased stems, nnFormer inherits the superiority of convolution in encoding precise spatial information and producing hierarchical representations that help model object concepts at various scales.

# III. METHOD

# A. Overview

The overall architecture of nnFormer is presented in Figure 2, which maintains a similar U shape as that of U-Net [12] and mainly consists of three parts, i.e., the encoder, bottleneck and decoder. Concretely, the encoder involves one embedding layer, two local transformer blocks (each block contains two successive layers) and two down-sampling layers. Symmetrically, the decoder branch includes two transformer blocks, two up-sampling layers and the last patch expanding layer for making mask predictions. Besides, the bottleneck comprises one down-sampling layer, one up-sampling layer and three global transformer blocks for providing large receptive field to support the decoder. Inspired by U-Net [12], we add skip connections between corresponding feature pyramids of the encoder and decoder in a symmetrical manner, which helps to recover fine-grained details in the prediction. However, different from atypical skip connections that often use summation or concatenation operation, we introduce skip attention to bridge the gap between the encoder and decoder.

In the following, we will demonstrate the forward procedure on Synapse. The forward pass on different datasets can be easily inferred based on the procedure on Synapse.

# B. Encoder

The input of nnFormer is a 3D patch $\mathcal { X } \in \mathcal { R } ^ { H \times W \times D }$ (usually randomly cropped from the original image), where H, W and D denote the height, width and depth of each input scan, respectively.

The embedding layer. On Synapse, the embedding block is responsible for transforming each input scan X into a high-dimensional tensor $\begin{array} { r } { \chi _ { e } ^ {  } \ \in \ \mathrm { ~ \mathbb { R } ^ { \frac { H } { 4 } } } \times \frac { w } { 4 } \times \frac { D } { 2 } \times C } \end{array}$ , where $\begin{array} { r } { \frac { H } { 4 } \times \frac { W } { 4 } \times \frac { D } { 2 } } \end{array}$ W4 × represents the number of the patch tokens and C represents the sequence length (these numbers may slightly vary on different datasets). Different from ViT [2] and Swin Transformer [3] that use large convolutional kernels in the embedding block to extract features, we found that applying successive convolutional layers with small convolutional kernels bring more benefits in the initial stage, which could be explained from two perspectives, i.e., i) why applying successive convolutional layers and ii) why using small-sized kernels. For i), we use convolutional layers in the embedding block because they encode pixel-level spatial information, more precisely than patch-wise positional encoding used in transformers. For ii), compared to large-sized kernels, small kernel sizes help reduce computational complexity while providing equal-sized receptive field. As shown in Figure 2b, the embedding block consists of four convolutional layers whose kernel size is 3. After each convolutional layer (except the last one), one GELU [26] and one layer normalization [27] layers are appended. In practice, depending on the size of input patch, strides of convolution in the embedding block may accordingly vary.

# Local Volume-based Multi-head Self-attention (LV-MSA).

After the embedding layer, we pass the high-dimensional tensor $\mathcal { X } _ { e }$ to transformer blocks. The main point behind is to fully entangle the captured long-term dependencies with the hierarchical object concepts at various scales produced by the down-sampling layers and the high-resolution spatial information encoded by the initial embedding layer. Compared to Swin Transformer [3], we compute self-attention within 3D local volumes (i.e., LV-MSA, Local Volume-based Multi-head Self-attention) instead of 2D local windows.

Suppose that $\mathcal { X } _ { \mathrm { L V } } \in \mathcal { R } ^ { L \times C }$ represents the input of the local transformer block, $\mathcal { X } _ { \mathrm { L V } }$ would be first reshaped to $\hat { \mathcal { X } } _ { \mathrm { L V } } ~ \in$ ${ \bf R } ^ { N _ { \mathrm { L V } } \times N _ { T } \times C }$ , where $N _ { \mathrm { { L V } } }$ is a pre-defined number of 3D local volumes and $N _ { T } = S _ { H } \times S _ { W } \times S _ { D }$ denotes the number of patch tokens in each volume. $\{ S _ { H } , ~ S _ { W } , ~ S _ { D } \}$ stand for the size of local volume.

As shown in Figure 3a, we follow [3] to conduct two successive transformer layers in each block, where the second layer can be regarded as a shifted version of the first layer (i.e., SLV-MSA). The main difference lies in that our computation is built on top of 3D local volumes instead of 2D local windows. The computational procedure can be summarized as follows:

$$
\hat {\mathcal {X}} _ {\mathrm{LV}} ^ {l} = \text { LV - MSA } \left(\text { Norm } \left(\mathcal {X} _ {\mathrm{LV}} ^ {l - 1}\right)\right) + \mathcal {X} _ {\mathrm{LV}} ^ {l - 1},
$$

$$
\mathcal {X} _ {\mathrm{LV}} ^ {l} = \operatorname{MLP} \left(\operatorname{Norm} \left(\hat {\mathcal {X}} _ {\mathrm{LV}} ^ {l}\right)\right) + \hat {\mathcal {X}} _ {\mathrm{LV}} ^ {l}, \tag {1}
$$

$$
\hat {\mathcal {X}} _ {\mathrm{LV}} ^ {l + 1} = \text { SLV - MSA } \left(\text { Norm } \left(\mathcal {X} _ {\mathrm{LV}} ^ {l}\right)\right) + \mathcal {X} _ {\mathrm{LV}} ^ {l},
$$

$$
\mathcal {X} _ {\mathrm{LV}} ^ {l + 1} = \operatorname{MLP} \left(\operatorname{Norm} \left(\hat {\mathcal {X}} _ {\mathrm{LV}} ^ {l + 1}\right)\right) + \hat {\mathcal {X}} _ {\mathrm{LV}} ^ {l + 1}.
$$

Here, l stands for the layer index. MLP is an abbreviation for multi-layer perceptron. The computational complexity of LV-MSA on a volume of $h \times w \times d$ patches is:

$$
\Omega (\mathrm{LV-MSA}) = 4 h w d C ^ {2} + 2 S _ {H} S _ {W} S _ {D} h w d C. \tag {2}
$$

SLV-MSA displaces the 3D local volume used in LV-MSA by $( \left\lfloor \frac { S _ { H } } { 2 } \right\rfloor , \left\lfloor \frac { S _ { W } ^ { \star } } { 2 } \right\rfloor , \left\lfloor \frac { S _ { D } } { 2 } \right\rfloor )$ to introduce more interactions between different local volumes. In practice, SLV-MSA has the similar computational complexity as that of LV-MSA.

The query-key-value (QKV) attention [1] in each 3D local volume can be computed as follows:

$$
\text { Attention } (Q, K, V) = \text { softmax } \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}} + B\right) V, \tag {3}
$$

where $Q , K , V \in \mathcal { R } ^ { N _ { T } \times d _ { k } }$ denote the query, key and value matrices. $B ~ \in ~ \mathcal { R } ^ { N _ { T } }$ is the relative position encoding. In practice, we first initialize a smaller-sized position matrix $\hat { B } \ \in \ \mathcal { R } ^ { ( 2 S _ { H } - 1 ) \times ( 2 S _ { W } - 1 ) \times ( 2 S _ { D } - 1 ) }$ and take corresponding values from $\hat { B }$ to build a larger position matrix B.

![](images/dfbe6f8d16fec5a6852569f699b1803aa924c4a8a9de260e508194f4e9e5497b.jpg)  
Fig. 2: Architecture of nnFormer. In (a), we show the overall architecture of nnFormer. In (b), we present more details of the embedding layers on three publicly available datasets. In (c), (d), (e), we display how to implement the down-sampling, up-sampling and expanding layers, respectively. In practice, the architecture may slightly vary depending on the input scan size. In (b)-(e), K denotes the convolutional kernel size, DK stands for the deconvolutional kernel size and S represents the stride. Norm refers to the layer normalization strategy.

![](images/e5e2d323d67917cbfebce56b8653e53b73c1c755aee9667f0556df0268536ee8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Input Layer
        X1["XLV"] --> A1["MLP"]
        A1 --> B1["Norm"]
        B1 --> C1["XLV"]
        C1 --> D1["VL-MSA"]
        D1 --> E1["Norm"]
        E1 --> F1["XLV^l-1"]
    end

    subgraph Output Layer
        X2["XLV"] --> A2["MLP"]
        A2 --> B2["Norm"]
        B2 --> C2["XLV^l+1"]
        C2 --> D2["SLV-MSA"]
        D2 --> E2["Norm"]
        E2 --> F2["XLV^l-1"]
    end

    A1 --> C1
    B1 --> C1
    C1 --> D1
    D1 --> E1
    E1 --> F1
    F1 --> G1["XLV^l+1"]
    style Input Layer fill:#f9f,stroke:#333
    style Output Layer fill:#bbf,stroke:#333
```
</details>

(a) LV-MSA

![](images/e31d8cb3a8c32965c5d6db169e0d057f6a4a70b9b946f6877bb70f8e8cec7c05.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Left_Layer
        A["×^l_GV"] --> B["MLP"]
        B --> C["Norm"]
        C --> D["×^l_GV"]
        D --> E["GV-MSA"]
        E --> F["Norm"]
        F --> G["×^l-GV^l-1"]
    end

    subgraph Right_Layer
        H["×^l+1_GV"] --> I["MLP"]
        I --> J["Norm"]
        J --> K["×^l+1_GV^l+1"]
        K --> L["GV-MSA"]
        L --> M["Norm"]
        M --> N["×^l+1_GV"]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O["×^l-1_GV"]
    style Left_Layer fill:#f9f,stroke:#333
    style Right_Layer fill:#bbf,stroke:#333
```
</details>

(b) GV-MSA

![](images/2fe5fbc3253bfa29ee09fe54f17cb907f99e22009751e5ad368081e39e439ea7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Linear Projection"] --> B["MLP"]
    B --> C["Norm"]
    C --> D["GV/LV-MSA"]
    D --> E["Norm"]
    E --> F["Output"]
    G["\(x_{\text{UP}}^{l+1}\)"] --> H["Feedback Loop"]
    H --> I["MLP"]
    I --> J["Norm"]
    J --> K["GV/LV-MSA"]
    K --> L["Norm"]
    L --> M["Output"]
    N["\(x_{\text{UP}}^{l}\)"] --> O["Feedback Loop"]
    O --> P["MLP"]
    P --> Q["Norm"]
    Q --> R["GV/LV-MSA"]
    R --> S["Norm"]
    S --> T["Output"]
    U["\(x_{\text{UP}}^{l}\)"] --> V["Feedback Loop"]
    V --> W["MLP"]
    W --> X["Norm"]
    X --> Y["GV/LV-MSA"]
    Y --> Z["Norm"]
    Z --> AA["Output"]
```
</details>

(c) Skip Attention   
Fig. 3: Three types of attention mechanism in nnFormer. Norm denotes the layer normalization method. MLP is the abbreviation for multi-layer perceptron, which is a two-layer neural network in practice.

The down-sampling layer. We found that by replacing the patch merging operation in [3] with straightforward strided convolution, nnFormer can provide more improvements on volumetric image segmentation. The intuition behind is that convolutional down-sampling produces hierarchical representations that help model object concepts at multiple scales. As displayed in Figure 2c, in most cases, the down-sampling layer involves a strided convolution operation where the stride is set to 2 in all dimensions. However, in practice, the stride with respect to specific dimension can be set to 1 as the number of slices is limited in this dimension and over-down-sampling (i.e., using a large down-sampling stride) can be harmful.

<table><tr><td></td><td>nnFormer</td><td>nnUNet</td></tr><tr><td>Spacing</td><td>[1.0, 1.0, 1.0]</td><td>[1.0, 1.0, 1.0]</td></tr><tr><td>Median shape</td><td> $138 \times 170 \times 138$ </td><td> $138 \times 170 \times 138$ </td></tr><tr><td>Crop size</td><td> $128 \times 128 \times 128$ </td><td> $128 \times 128 \times 128$ </td></tr><tr><td>Batch size</td><td>2</td><td>2</td></tr><tr><td>DS Str.</td><td>[2, 2, 2], [2, 2, 2], [2, 2, 2],[2, 2, 2], [2, 2, 2]</td><td>[2, 2, 2], [2, 2, 2], [2, 2, 2],[2, 2, 2], [2, 2, 2], [2, 2, 2]</td></tr></table>

(a) Tumor

<table><tr><td></td><td>nnFormer</td><td>nnUNet</td></tr><tr><td>Spacing</td><td>[0.76, 0.76, 3]</td><td>[0.76, 0.76, 3]</td></tr><tr><td>Median shape</td><td>512 × 512 × 148</td><td>512 × 512 × 148</td></tr><tr><td>Crop size</td><td>128 × 128 × 64</td><td>192 × 192 × 48</td></tr><tr><td>Batch size</td><td>2</td><td>2</td></tr><tr><td>DS Str.</td><td>[2, 2, 2], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]</td><td>[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]</td></tr></table>

(b) Synapse

<table><tr><td></td><td>nnFormer</td><td>nnUNet</td></tr><tr><td>Spacing</td><td>[1.52, 1.52, 6.35]</td><td>[1.52, 1.52, 6.35]</td></tr><tr><td>Median shape</td><td> $246 \times 213 \times 13$ </td><td> $246 \times 213 \times 13$ </td></tr><tr><td>Crop size</td><td> $160 \times 160 \times 14$ </td><td> $256 \times 224 \times 14$ </td></tr><tr><td>Batch size</td><td>4</td><td>4</td></tr><tr><td>DS Str.</td><td>[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]</td><td>[2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 1], [2, 2, 1]</td></tr></table>

(c) ACDC   
TABLE I: Network configurations of our nnFormer and nnUNet on three public datasets. We only report the downsampling stride (abbreviated as DS Str.) as the corresponding up-sampling stride can be easily inferred according to symmetrical down-sampling operations. Note that the network configuration of nnUNet is automatically determined based on pre-defined hand-crafted rules (for self-adaptation).

# C. Bottleneck

The original vision transformer (i.e., ViT) [2] employs the naive 2D multi-head self-attention mechanism. In this paper, we extend it to a 3D version (as shown in Figure 3b), whose computational complexity can be formulated as follows:

$$
\Omega (\mathrm{GV-MSA}) = 4 h w d C ^ {2} + 2 (h w d) ^ {2} C. \tag {4}
$$

Compared to $( 2 ) ,$ it is obvious that GV-MSA requires much more computational resources when $\{ h , w , d \}$ are relatively larger (e.g., an order of magnitude larger) than $\left\{ S _ { H } , S _ { W } , S _ { D } \right\}$ . In fact, this is exactly the reason why we use local transformer blocks in the encoder, which are designed to handle large-sized inputs efficiently with the local self-attention mechanism.

However, in the bottleneck, $\{ h , w , d \}$ already become much smaller after several down-sampling layers, making the product of them, i.e. hwd, , have a similar size to that of $S _ { H } S _ { W } S _ { D }$ . This creates the condition for applying GV-MSA, which is able to provide larger receptive field compared to LV-MSA and large receptive field has been proven to be beneficial in different applications [28]–[31]. In practice, we use three global transformer blocks (i.e., six GV-MSA layers) in the bottleneck to provide sufficient receptive field to the decoder.

# D. Decoder

The architecture of two transformer blocks in the decoder is highly symmetrical to those in the encoder. In contrast to the down-sampling blocks, we employ strided deconvolution to up-sample low-resolution feature maps to high-resolution ones, which in turn are merged with representations from the encoder via skip attention to capture both semantic and fine-grained information. Similar to up-sampling blocks, the last patch expanding block also takes the deconvolutional operation to produce final mask predictions.

Skip Attention. Atypical skip connections in convnets [12, 32] adapt either concatenation or summation to incorporate more information. Inspired by the machine translation task in [1], we propose to replace the concatenation/summation with an attention mechanism, which is named as Skip Attention in this paper. To be specific, the output of the l-th transformer block of the encoder, i.e., $\mathcal { X } _ { \{ \mathrm { L V , G V } \} } ^ { l }$ , is transformed and split into a key matrix $K ^ { l ^ { * } }$ and a value matrix $V ^ { l ^ { * } }$ after the linear projection (i.e, a one-layer neural network):

$$
K ^ {l ^ {*}}, V ^ {l ^ {*}} = \mathrm{LP} (\mathcal {X} _ {\{\mathrm{LV}, \mathrm{GV} \}} ^ {l}), \tag {5}
$$

where LP stands for the linear projection. Accordingly, $\chi _ { \mathrm { U P } } ^ { l ^ { * } } .$ the output feature maps after the l∗-th up-sampling layer of the decoder, is treated as the query $Q ^ { l ^ { * } }$ . Then, we can conduct LV/GV-MSA on $Q ^ { l ^ { * } } , K ^ { l ^ { * } }$ and $\dot { V } ^ { l ^ { * } }$ in the decoder like what we have done in (3), i.e.,

$$
\text { Attention } (Q ^ {l ^ {*}}, K ^ {l ^ {*}}, V ^ {l ^ {*}}) = \text { softmax } \left(\frac {Q ^ {l ^ {*}} (K ^ {l ^ {*}}) ^ {T}}{\sqrt {d _ {k} ^ {l ^ {*}}}} + B ^ {l ^ {*}}\right) V ^ {l ^ {*}}, \tag {6}
$$

where $l ^ { * }$ denotes the layer index. $d _ { k } ^ { l ^ { * } }$ and $B ^ { l ^ { * } }$ have the same meaning as those in (3), whose sizes can be easily inferred, accordingly.

# IV. EXPERIMENTS

For thoroughly comparing nnFormer to previous convnetand transformer-based architecture, we conduct experiments on three datasets/tasks: the brain tumor segmentation task in Medical Segmentation Decathlon (MSD) [36], Synapse multiorgan segmentation [37] and Automatic Cardiac Diagnosis Challenge (ACDC) [38]. For each experiment, we repeat it for ten times and report their average results. We also calculate p-values to demonstrate the significance of nnFormer.

Brain tumor segmentation using MRI scans. This task consists of 484 MRI images, each of which includes four channels, i.e., FLAIR, T1w, T1gd and T2w. The data was acquired from 19 different institutions and contained a subset of the data used in the 2016 and 2017 Brain Tumor Segmentation (BraTS) challenges [39]. The corresponding target ROIs were the three tumor sub-regions, namely edema (ED), enhancing tumor (ET), and non-enhancing tumor (NET). To be consistent with those results reported in UNETR [34], we display the experimental results of the whole tumor (WT), enhancing tumor (ET) and tumor core (TC) when comparing our nnFormer with transformer-based models. For the split of data, we follow the instruction of UNETR, where ratios of training/validation/test sets are 80%, 15% and 5%, respectively. As above, we use both HD95 and Dice score as evaluation metrics.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Average</td><td colspan="2">WT</td><td colspan="2">ET</td><td colspan="2">TC</td></tr><tr><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td></tr><tr><td>SETR NUP [33]</td><td>13.78</td><td>63.7</td><td>14.419</td><td>69.7</td><td>11.72</td><td>54.4</td><td>15.19</td><td>66.9</td></tr><tr><td>SETR PUP [33]</td><td>14.01</td><td>63.8</td><td>15.245</td><td>69.6</td><td>11.76</td><td>54.9</td><td>15.023</td><td>67.0</td></tr><tr><td>SETR MLA [33]</td><td>13.49</td><td>63.9</td><td>15.503</td><td>69.8</td><td>10.24</td><td>55.4</td><td>14.72</td><td>66.5</td></tr><tr><td>TransUNet [11]</td><td>12.98</td><td>64.4</td><td>14.03</td><td>70.6</td><td>10.42</td><td>54.2</td><td>14.5</td><td>68.4</td></tr><tr><td>TransBTS [25]</td><td>9.65</td><td>69.6</td><td>10.03</td><td>77.9</td><td>9.97</td><td>57.4</td><td>8.95</td><td>73.5</td></tr><tr><td>CoTr w/o CNN encoder [24]</td><td>11.22</td><td>64.4</td><td>11.49</td><td>71.2</td><td>9.59</td><td>52.3</td><td>12.58</td><td>69.8</td></tr><tr><td>CoTr [24]</td><td>9.70</td><td>68.3</td><td>9.20</td><td>74.6</td><td>9.45</td><td>55.7</td><td>10.45</td><td>74.8</td></tr><tr><td>UNETR [34]</td><td>8.82</td><td>71.1</td><td>8.27</td><td>78.9</td><td>9.35</td><td>58.5</td><td>8.85</td><td>76.1</td></tr><tr><td>Our nnFormer</td><td>4.05</td><td>86.4</td><td>3.80</td><td>91.3</td><td>3.87</td><td>81.8</td><td>4.49</td><td>86.0</td></tr><tr><td>P-values</td><td colspan="8">&lt; 1e-2 (HD95), &lt; 1e-2 (DSC)</td></tr></table>

TABLE II: Comparison with transformer-based models on brain tumor segmentation. The evaluation metrics are HD95 (mm) and DSC in (%). Best results are bolded while second best are underlined. Experimental results of baselines are from [34]. We calculate the p-values between the average performance of our nnFormer and the best performing baseline in both metrics.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Average</td><td rowspan="2">Aotra</td><td rowspan="2">Gallbladder</td><td rowspan="2">Kidney (Left)</td><td rowspan="2">Kidney (Right)</td><td rowspan="2">Liver</td><td rowspan="2">Pancreas</td><td rowspan="2">Spleen</td><td rowspan="2">Stomach</td></tr><tr><td>HD95 ↓</td><td>DSC ↑</td></tr><tr><td>ViT [2] + CUP [11]</td><td>36.11</td><td>67.86</td><td>70.19</td><td>45.10</td><td>74.70</td><td>67.40</td><td>91.32</td><td>42.00</td><td>81.75</td><td>70.44</td></tr><tr><td>R50-ViT [2] + CUP [11]</td><td>32.87</td><td>71.29</td><td>73.73</td><td>55.13</td><td>75.80</td><td>72.20</td><td>91.51</td><td>45.99</td><td>81.99</td><td>73.95</td></tr><tr><td>TransUNet [11]</td><td>31.69</td><td>77.48</td><td>87.23</td><td>63.16</td><td>81.87</td><td>77.02</td><td>94.08</td><td>55.86</td><td>85.08</td><td>75.62</td></tr><tr><td>TransUNet▽ [11]</td><td>-</td><td>84.36</td><td>90.68</td><td>71.99</td><td>86.04</td><td>83.71</td><td>95.54</td><td>73.96</td><td>88.80</td><td>84.20</td></tr><tr><td>SwinUNet [18]</td><td>21.55</td><td>79.13</td><td>85.47</td><td>66.53</td><td>83.28</td><td>79.61</td><td>94.29</td><td>56.58</td><td>90.66</td><td>76.60</td></tr><tr><td>TransClaw U-Net [15]</td><td>26.38</td><td>78.09</td><td>85.87</td><td>61.38</td><td>84.83</td><td>79.36</td><td>94.28</td><td>57.65</td><td>87.74</td><td>73.55</td></tr><tr><td>LeVit-UNet-384s [22]</td><td>16.84</td><td>78.53</td><td>87.33</td><td>62.23</td><td>84.61</td><td>80.25</td><td>93.11</td><td>59.07</td><td>88.86</td><td>72.76</td></tr><tr><td>MISSFormer [35]</td><td>18.20</td><td>81.96</td><td>86.99</td><td>68.65</td><td>85.21</td><td>82.00</td><td>94.41</td><td>65.67</td><td>91.92</td><td>80.81</td></tr><tr><td>UNETR [34]</td><td>22.97</td><td>79.56</td><td>89.99</td><td>60.56</td><td>85.66</td><td>84.80</td><td>94.46</td><td>59.25</td><td>87.81</td><td>73.99</td></tr><tr><td>Our nnFormer</td><td>10.63</td><td>86.57</td><td>92.04</td><td>70.17</td><td>86.57</td><td>86.25</td><td>96.84</td><td>83.35</td><td>90.51</td><td>86.83</td></tr><tr><td>P-values</td><td colspan="10">&lt; 1e-2 (HD95), &lt; 1e-2 (DSC)</td></tr></table>

TABLE III: Comparison with transformer-based models on multi-organ segmentation (Synapse). The evaluation metrics are HD95 (mm) and DSC in (%). Best results are bolded while second best are underlined. 5 denotes TransUNet uses larger inputs, whose size is 512×512. The p-values are calculated based on the average performance of our nnFormer and the best performing baseline in both metrics.

<table><tr><td>Methods</td><td>Average</td><td>RV</td><td>Myo</td><td>LV</td></tr><tr><td>VIT-CUP [2]</td><td>81.45</td><td>81.46</td><td>70.71</td><td>92.18</td></tr><tr><td>R50-VIT-CUP [2]</td><td>87.57</td><td>86.07</td><td>81.88</td><td>94.75</td></tr><tr><td>TransUNet [11]</td><td>89.71</td><td>88.86</td><td>84.54</td><td> $\underline{95.73}$ </td></tr><tr><td>SwinUNet [18]</td><td>90.00</td><td>88.55</td><td>85.62</td><td> $\underline{95.83}$ </td></tr><tr><td>LeViT-UNet-384s [22]</td><td> $\underline{90.32}$ </td><td> $\underline{89.55}$ </td><td> $\underline{87.64}$ </td><td>93.76</td></tr><tr><td>UNETR [34]</td><td>88.61</td><td>85.29</td><td>86.52</td><td>94.02</td></tr><tr><td>nnFormer</td><td> $\underline{92.06}$ </td><td> $\underline{90.94}$ </td><td> $\underline{89.58}$ </td><td>95.65</td></tr><tr><td>P-value</td><td colspan="4">&lt; 1e-2 (DSC)</td></tr></table>

TABLE IV: Comparison with transformer-based models on automatic cardiac diagnosis (ACDC). The evaluation metric is DSC (%). Best results are bolded while second best are underlined. The default evaluation metric is DSC, based on which we calculate the p-value.

Synapse for multi-organ CT segmentation. This dataset includes 30 cases of abdominal CT scans. Following the split used in [11], 18 cases are extracted to build the training set while the rest 12 cases are used for testing. We report the model performance evaluated with the 95% Hausdorff Distance (HD95) and Dice score (DSC) on 8 abdominal organs, which are aorta, gallbladder, spleen, left kidney, right kidney, liver, pancreas and stomach1.

ACDC for automated cardiac diagnosis. ACDC involves 100 patients, with the cavity of the right ventricle, the myocardium of the left ventricle and the cavity of the left ventricle to be segmented. Each case’s labels involve left ventricle (LV), right ventricle (RV) and myocardium (MYO). The dataset is split into 70 training samples, 10 validation samples and 20 test samples. The evaluation metrics include

both HD95 and Dice score2.

# A. Implementation details

We run all experiments based on Python 3.6, PyTorch 1.8.1 and Ubuntu 18.04. All training procedures have been performed on a single NVIDIA 2080 GPU with 11GB memory. The initial learning rate is set to 0.01 and we employ a “poly” decay strategy as described in Equation 7. The default optimizer is SGD where we set the momentum to 0.99. The weight decay is set to 3e-5. We utilize both cross entropy loss and dice loss by simply summing them up. The number of training epochs (i.e., max epoch in Equation 7) is 1000 and one epoch contains 250 iterations. The number of heads of multi-head self-attention used in different encoder stages is [6, 12, 24, 48] on Synapse. In the rest two datasets, the number of heads becomes [3, 6, 12, 24].

$$
\mathrm{lr} = \text { initial\_lr } \times (1 - \frac {\text { epoch\_id }}{\text { max\_epoch }}) ^ {0. 9}. \tag {7}
$$

Pre-processing and augmentation strategies. All images will be first resampled to the same target spacing. Augmentations such as rotation, scaling, gaussian noise, gaussian blur, brightness and contrast adjust, simulation of low resolution, gamma augmentation and mirroring are applied in the given order during the training process.

Deep supervision. We also add deep supervision during the training stage. Specifically, the output of each stage in the decoder is passed to the final expanding block, where cross entropy loss and dice loss would be applied. In practice, given the prediction of one typical stage, we down-sample the ground truth segmentation mask to match the prediction’s resolution. Thus, the final training objective function is the sum of all losses at three resolutions:

$$
\mathcal {L} _ {\text { all }} = \alpha_ {1} \mathcal {L} _ {\{H, W, D \}} + \alpha_ {2} \mathcal {L} _ {\left\{\frac {H}{4}, \frac {W}{4}, \frac {D}{2} \right\}} + \alpha_ {3} \mathcal {L} _ {\left\{\frac {H}{8}, \frac {W}{8}, \frac {D}{4} \right\}}. \tag {8}
$$

Here, $\alpha _ { \{ 1 , ~ 2 , ~ 3 \} }$ denote the magnitude factors for losses in different resolutions. In practice, $\alpha _ { \left\{ 1 , \ 2 , \ 3 \right\} }$ halve with each decrease in resolution, leading to $\alpha _ { 2 } ~ = ~ \frac { \alpha _ { 1 } } { 2 }$ and $\textstyle \alpha _ { 3 } ~ = ~ { \frac { \alpha _ { 1 } } { 4 } }$ . Finally, all weight factors are normalized to 1.

Network configurations. In Table I, we display network configurations of experiments on all three datasets. Compared to nnUNet, in nnFormer, better segmentation results can be achieved with smaller-sized input patches.

# B. Comparison with transformer-based methodologies

Brain tumor segmentation. Table II presents experimental results of all models on the task of brain tumor segmentation. Our nnFormer achieves the lowest HD95 and the highest DSC scores in all classes. Moreover, nnFormer is able to surpass the second best method, i.e., UNETR, by large margins in

2Similar to Synapse, we also follow the evaluation setting of TransUNet. both evaluation metrics. For instance, nnFormer outperforms UNETR by over 4.5 mm in average HD95 and nearly 10 percents in DSC of each class. In comparison to previous transformer-based methods, nnFormer shows more strength in HD95 than in DSC.

![](images/a74177b0101cbdf4a8883cc6b1bc4fa80f72c61a346f91d650dfa58c203c32ca.jpg)

<details>
<summary>heatmap</summary>

| Model   | ED    | NET   | ET    |
|---------|-------|-------|-------|
| Gt      | 100   | 80    | 60    |
| Ours    | 95    | 75    | 55    |
| nnUNet  | 90    | 70    | 50    |
| UNETR   | 85    | 65    | 45    |
</details>

![](images/2c24365bab45578aa4470d952c679045fc07b15611c96625aa9549a7ba2dfe98.jpg)

<details>
<summary>bar_stacked</summary>

| Model   | Spleen | Pancreas | Kidney (right) | Kidney (left) | Gallbladder | Liver | Stomach | Aorta |
|---------|--------|----------|----------------|---------------|-------------|-------|---------|-------|
| Gt      | 0      | 0        | 0              | 0             | 0           | 0     | 0       | 0     |
| Ours    | 0      | 0        | 0              | 0             | 0           | 0     | 0       | 0     |
| nnUNet  | 0      | 0        | 0              | 0             | 0           | 0     | 0       | 0     |
| UNETR   | 0      | 0        | 0              | 0             | 0           | 0     | 0       | 0     |
</details>

![](images/54fe90761b5aa6a65851f8365cff44b66ac27b15e78a5c777aa7d0cb2ec8e9e2.jpg)

<details>
<summary>text_image</summary>

Gt
Ours
nnUNet
UNETR
(c) Automatic cardiac diagnosis (ACDC)
</details>

Fig. 4: Visualization of segmentation results on three wellestablished datasets. We mainly compare nnFormer against nnUNet and UNETR. In addition to segmentation results, we also provide ground truth masks for better comparison.

Multi-organ segmentation (Synapse). As shown in Table III, we make experiments on Synapse and to compare our nnFormer against a variety of transformer-based approaches. As we can see, the best performing methods are LeViT-UNet-384s [22] and TransUNet [11]. LeViT-UNet-384s achieves an average HD95 of 16.84 mm while TransUNet produces

<table><tr><td rowspan="2">Methods</td><td colspan="2">Average</td><td colspan="2">WT</td><td colspan="2">ET</td><td colspan="2">TC</td><td colspan="2">ED</td><td colspan="2">NET</td></tr><tr><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td></tr><tr><td>nnUNet [40]</td><td>4.60</td><td>81.87</td><td>3.64</td><td>91.99</td><td>4.06</td><td>80.97</td><td>4.91</td><td>85.35</td><td>4.26</td><td>84.39</td><td>6.14</td><td>66.65</td></tr><tr><td>Our nnFormer</td><td>4.42</td><td>82.02</td><td>3.80</td><td>91.26</td><td>3.87</td><td>81.80</td><td>4.49</td><td>86.02</td><td>4.17</td><td>83.76</td><td>5.76</td><td>67.29</td></tr><tr><td>P-values</td><td colspan="12">&lt; 1e-2 (HD95), 8.8e-2 (DSC)</td></tr><tr><td>nnAvg</td><td>4.09</td><td>82.65</td><td>3.43</td><td>92.33</td><td>3.69</td><td>82.26</td><td>4.17</td><td>86.14</td><td>3.92</td><td>84.95</td><td>5.23</td><td>67.55</td></tr></table>

(a) Brain tumor segmentation

<table><tr><td rowspan="2">Methods</td><td colspan="2">Average</td><td colspan="2">Aotra</td><td colspan="2">Gallbladder</td><td colspan="2">Kidney (Left)</td><td colspan="2">Kidney (Right)</td><td colspan="2">Liver</td><td colspan="2">Pancreas</td><td colspan="2">Spleen</td><td colspan="2">Stomach</td></tr><tr><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td></tr><tr><td>nnUNet [40]</td><td>10.78</td><td>86.99</td><td>5.91</td><td>93.01</td><td>15.19</td><td>71.77</td><td>18.60</td><td>85.57</td><td>6.44</td><td>88.18</td><td>1.62</td><td>97.23</td><td>4.52</td><td>83.01</td><td>24.34</td><td>91.86</td><td>9.58</td><td>85.26</td></tr><tr><td>Our nnFormer</td><td>10.63</td><td>86.57</td><td>11.38</td><td>92.04</td><td>11.55</td><td>70.17</td><td>18.09</td><td>86.57</td><td>12.76</td><td>86.25</td><td>2.00</td><td>96.84</td><td>3.72</td><td>83.35</td><td>16.92</td><td>90.51</td><td>8.58</td><td>86.83</td></tr><tr><td>P-values</td><td colspan="18">2e-2 (HD95), 7.7e-2 (DSC)</td></tr><tr><td>nnAvg</td><td>7.70</td><td>87.51</td><td>5.90</td><td>93.11</td><td>8.63</td><td>72.08</td><td>18.42</td><td>86.20</td><td>8.56</td><td>87.76</td><td>1.63</td><td>97.20</td><td>3.64</td><td>84.21</td><td>9.42</td><td>91.94</td><td>5.41</td><td>87.60</td></tr></table>

(b) Multi-organ segmentation (Synapse)

<table><tr><td rowspan="2">Methods</td><td colspan="2">Average</td><td colspan="2">RV</td><td colspan="2">Myo</td><td colspan="2">LV</td></tr><tr><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td><td>HD95 ↓</td><td>DSC ↑</td></tr><tr><td>nnUNet [40]</td><td>1.15</td><td>91.61</td><td>1.31</td><td>90.24</td><td>1.06</td><td>89.24</td><td>1.09</td><td>95.36</td></tr><tr><td>Our nnFormer</td><td>1.12</td><td>92.06</td><td>1.23</td><td>90.94</td><td>1.04</td><td>89.58</td><td>1.09</td><td>95.65</td></tr><tr><td>P-values</td><td colspan="8">2e-2 (HD95), &lt; 1e-2 (DSC)</td></tr><tr><td>nnAvg</td><td>1.10</td><td>92.15</td><td>1.19</td><td>91.03</td><td>1.04</td><td>89.75</td><td>1.06</td><td>95.68</td></tr></table>

(c) Automated cardiac diagnosis (ACDC)

TABLE V: Comparison with nnUNet on three public datasets. nnAvg means that we simply average the predictions of nnUNet and nnFormer. Color green denotes the target result of nnAvg is the best among all three approaches. Besides, we also highlight the best results between nnUNet and nnFormer in bold font. We calculate p-values between the average performance of nnUNet and our nnFormer in both metrics on three public datasets.

<table><tr><td>#</td><td>Models</td><td>Average</td><td>RV</td><td>Myo</td><td>LV</td></tr><tr><td>0</td><td>1×LV-MSA + PM [3] + PE [3]</td><td>90.55</td><td>88.59</td><td>88.47</td><td>94.60</td></tr><tr><td>1</td><td>1×LV-MSA + PM [3] + Conv. Embed.</td><td>90.97</td><td>88.94</td><td>88.84</td><td>95.13</td></tr><tr><td>2</td><td>1×LV-MSA + Conv. Down. + Conv. Embed.</td><td>91.26</td><td>89.70</td><td>89.04</td><td>95.04</td></tr><tr><td>3</td><td>1×LV-MSA + 1×GV-MSA + Conv. Down. + Conv. Embed.</td><td>91.46</td><td>89.82</td><td>89.17</td><td>95.39</td></tr><tr><td>4</td><td>1×LV-MSA + 1×GV-MSA + Conv. Down. + Conv. Embed. + Skip Att.</td><td>91.85</td><td>90.41</td><td>89.50</td><td>95.63</td></tr><tr><td>5</td><td>1×LV-MSA + 1×SLV-MSA + 2×GV-MSA + Conv. Down. + Conv. Embed. + Skip Att.</td><td>92.06</td><td>90.94</td><td>89.58</td><td>95.65</td></tr></table>

TABLE VI: Investigation of the impact of different modules used in nnFormer. PM and PE denote the patch merging and patch embedding strategies used in swin transformer [3]. Conv. Embed. and Conv. Down. represent our convolutional embedding and down-sampling layers, respectively. Skip Att. refers to the proposed skip attention mechanism. 1×LV-MSA in lines 0-2 means that each transformer block contains one transformer layer and each layer consists of one LV-MSA. 1×GV-MSA in lines 3-4 denotes that we replace LV-MSA in the bottleneck with GV-MSA. 1×SLV-MSA and 2×GV-MSA in line 5 mean that we increase the number of transformer layers in each transformer block from one to two. To be specific, in the encoder/decoder, each transformer block contains 1×LV-MSA and 1×SLV-MSA while in the bottleneck, there are 2×GV-MSA in each block.

an average DSC of 84.36%. In comparison, our nnFormer is able to outperform LeViT-UNet-384s and TransUNet by over 6 mm and 2 percents in average HD95 and DSC, respectively, which are quite impressive improvements on Synapse. To be specific, nnFormer achieves the highest DSC in six organs, including aotra, kidney (left), kidney (right), liver, pancreas and stomach. Compared to previous transformer-based methods, nnFormer is more advantageous in segmentation pancreas and stomach, both of which are difficult to delineate using past segmentation models.

Automated cardiac diagnosis (ACDC). Table IV displays experimental results on ACDC. We can see that the best transformer-based model is LeViT-UNet-384s, whose average DSC is slightly higher than SwinUNet while TransUNet and SwinUNet are more capable of handling the delineation of the left ventricle (LV). In contrast, nnFormer surpasses LeViT-UNet-384s in all classes and by nearly 1.7 percents in average DSC, which again verifies its advantages over past transformer-based approaches.

Statistical significance. In Table II, III and IV, we employ independent two-sample t-test to calculate p-values between the average performance of our nnFormer and the best performing baseline in both HD95 and DSC. The null hypothesis is that our nnFormer has no advantage over the best performing baseline. As we can see, on all three public datasets, nnFormer produces p-values smaller than 1e-2 under both HD95 and DSC, which indicate strong evidence against the null hypothesis. Thus, nnFormer shows significant improvements over previous transformer-based methods on three different tasks.

# C. Comparison with nnUNet and Discussion

In this section, we compare nnFormer with nnUNet, which has been recognized as one of the most powerful 3D medical image segmentation models [40].

Results. In Table V, we display the class-specific results in both HD95 and DSC metrics to make a thorough comparison. To be specific, from the perspective of the class-specific HD95 results, nnFormer outperforms nnUNet in 11 out of 16 categories. In the class-specific DSC, nnFormer outperforms nnUNet in 9 out of 16 categories. Thus, it seems that nnFormer is more advantageous under HD95, which means nnFormer may better delineate the object boundary. From the view of the average performance, we can see that nnFormer often achieves better average performance. For example, nnFormer outperforms nnUNet on all three public datasets with lower HD95 results, while performing better than nnUNet on two out of three datasets with higher DSC results.

Statistical significance. To further verify the significance of nnFormer over nnUNet, we also calculate the p-values between the average performance of nnFormer and nnUNet. Similar to what we have done in Table II, we provide two p-values based on HD95 and DSC on three public datasets, respectively. The most obvious observation is that nnFormer achieves p-values smaller than 0.05 in HD95 on three public datasets. These results suggest that nnFormer is the first choice when HD95 is treated as the primary evaluation metric. Besides, the p-values based on DSC on tumor and multi-organ segmentation (> 0.05) imply that nnFormer is a model comparable to nnUNet, while the results on ACDC demonstrate the significance of nnFormer. In conclusion, nnFormer has slight advantages over nnUNet under DSC.

Model ensembling. Besides single model performance, we also investigate the diversity between nnFormer and nnUNet, which is a crucial factor in model ensembling. Somewhat surprisingly, we found that by simply averaging the predictions of nnFormer and nnUNet (i.e., nnAvg in Table V), it can already boost the overall performance by large margins. For instance, nnAvg achieves the best results in all classes under HD95 and DSC on tumor segmentation. Moreover, nnAvg brings nearly 30% improvements on Synapse when the evaluation metric is HD95. These results indicate that nnFormer and nnUNet are highly complementary to each other.

# D. Ablation study

Table VI displays our ablation study results towards different modules in nnFormer. For simplicity, we made experiments on ACDC and used DSC as the default evaluation metric.

The most basic baseline in Table VI (line 0) consists of LV-MSA (but without SLV-MSA), the patch merging and embedding layers used in [3]. We can see that such combination can already achieve a higher average DSC than LeViT-UNet-38 [22], which is the best performing baseline in Table IV. We firstly replaced the patch embedding layer, which is implemented with large kernel size and convolutional stride, with our proposed volume embedding layer, i.e., successive convolutional layers with small kernel size and convolutional stride. We found that the introduced convolutional embedding layer improves the average DSC by approximate 0.4 percents.

Next, we removed the patch merging layer and added our convolutional down-sampling layer. We found such simple replacement can further boost the overall performance by 0.3 percents. Then, we replaced LV-MSA in the bottleneck with GV-MSA, where we observed 0.2-percent improvements. This phenomenon indicates that providing sufficient larger receptive field can be beneficial to the segmentation task. Afterwards, we use skip attention to replace traditional concatenation/summation operations. Somewhat surprisingly, we found that the skip attention is able to boost the overall performance by 0.4 percents, which demonstrates that the skip attention may serve as an alternative choice other than traditional skip connections. Last but not the least, we investigate adding more transformer layers to each transformer block by cascading an SLV-MSA layer with every LV-MSA layer as in Swin Transformer and doubling the number of global self-attention layers. We found that introducing more transformer layers does bring more improvements to the overall performance as it entangles more long-range dependencies into the learned volume representations.

# E. Visualization of segmentation results

In Figure 4, we visualize some segmentation results of our nnFormer, nnUNet and UNETR on three public datasets. Compared to UNETR, our nnFormer can greatly reduce the number of false positive predictions. One typical example is the fifth example on ACDC. We can see that UNETR produces a large number of wrong right ventricle pixels outside the myocardium. In contrast, our nnFormer generates no prediction of right ventricle outside the myocardium, which demonstrates that nnFormer is more discriminative than UNETR on ACDC.

On the other hand, we observe that nnUNet displays very competitive segmentation results, much better than UNETR in nearly all examples. However, we still find that nnFormer maintains clear advantages over nnUNet, one of which is that nnFormer is better at dealing with the boundary. In fact, this phenomenon has been reflected in Table VI, where nnFormer is significantly better than nnUNet when HD95 is the default evaluation metric. In Figure 4, we can also observe some evidences. For instance, in the second example on Synapse, nnFormer captures the shape of the left kidney and stomach better than nnUNet. Also, in the third example on brain tumor segmentation, nnUNet misses a major part of the non-enhancing tumor enclosed by the edema. These results verify that our nnFormer has the potential to be treated as an alternative to nnUNet.

# V. CONCLUSION

In this paper, we present a 3D transformer, nnFormer, for volumetric image segmentation. nnFormer is constructed on top of an interleaved stem of convolution and self-attention. Convolution helps encode precise spatial information and builds hierarchical object concepts. For self-attention, nn-Former employs three types of attention mechanism to entangle long-range dependencies. Specifically, local and global volume-based self-attention focus on constructing feature pyramids and providing large receptive field. Skip attention is responsible for bridging the gap between the encoder and decoder. Experiments show that nnFormer maintains great advantages over previous transformer-based models in both HD95 and DSC. Compared to nnUNet, nnFormer is significantly better in HD95 while producing comparable results in DSC. More importantly, we demonstrate that nnFormer and nnUNet can be beneficial to each other in model ensembling, where the simple averaging operation can already produce great improvements.

# REFERENCES

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, pp. 5998–6008, 2017.   
[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.   
[3] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” arXiv preprint arXiv:2103.14030, 2021.   
[4] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, ´ “Masked autoencoders are scalable vision learners,” arXiv preprint arXiv:2111.06377, 2021.   
[5] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in European Conference on Computer Vision, pp. 213–229, Springer, 2020.   
[6] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.   
[7] H.-Y. Zhou, C. Lu, S. Yang, and Y. Yu, “ConvNets vs. Transformers: Whose visual representations are more transferable?,” ICCV Workshop on Deep Multi-Task Learning in Computer Vision, 2021.   
[8] T. Qu, X. Wang, C. Fang, L. Mao, J. Li, P. Li, J. Qu, X. Li, H. Xue, Y. Yu, et al., “M3net: A multi-scale multi-view framework for multiphase pancreas segmentation based on cross-phase non-local attention,” Medical Image Analysis, vol. 75, p. 102232, 2022.   
[9] D. Zhang, G. Huang, Q. Zhang, J. Han, J. Han, and Y. Yu, “Crossmodality deep feature learning for brain tumor segmentation,” Pattern Recognition, vol. 110, p. 107562, 2021.   
[10] S. Tuli, I. Dasgupta, E. Grant, and T. L. Griffiths, “Are convolutional neural networks or transformers more like human vision?,” arXiv preprint arXiv:2105.07197, 2021.   
[11] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, et al., “TransUNet: Transformers make strong encoders for medical image segmentation,” arXiv preprint arXiv:2102.04306, 2021.   
[12] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention, pp. 234–241, Springer, 2015.   
[13] Y. Zhang, H. Liu, and Q. Hu, “TransFuse: Fusing transformers and cnns for medical image segmentation,” arXiv preprint arXiv:2102.08005, 2021.   
[14] J. M. J. Valanarasu, P. Oza, I. Hacihaliloglu, and V. M. Patel, “Medical Transformer: Gated axial-attention for medical image segmentation,” arXiv preprint arXiv:2102.10662, 2021.   
[15] Y. Chang, H. Menghan, Z. Guangtao, and Z. Xiao-Ping, “TransClaw U-Net: Claw u-net with transformers for medical image segmentation,” arXiv preprint arXiv:2107.05188, 2021.   
[16] B. Chen, Y. Liu, Z. Zhang, G. Lu, and D. Zhang, “TransAttUnet: Multi-level attention-guided u-net with transformer for medical image segmentation,” arXiv preprint arXiv:2107.05274, 2021.   
[17] D. Karimi, S. Vasylechko, and A. Gholipour, “Convolution-free medical image segmentation using transformers,” arXiv preprint arXiv:2102.13645, 2021.   
[18] H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, and M. Wang, “Swin-Unet: Unet-like pure transformer for medical image segmentation,” arXiv preprint arXiv:2105.05537, 2021.

[19] A. Lin, B. Chen, J. Xu, Z. Zhang, and G. Lu, “DS-TransUNet: Dual swin transformer u-net for medical image segmentation,” arXiv preprint arXiv:2106.06716, 2021.   
[20] S. Li, X. Sui, X. Luo, X. Xu, Y. Liu, and R. S. M. Goh, “Medical image segmentation using squeeze-and-expansion transformers,” arXiv preprint arXiv:2105.09511, 2021.   
[21] B. Yun, Y. Wang, J. Chen, H. Wang, W. Shen, and Q. Li, “SpecTr: Spectral transformer for hyperspectral pathology image segmentation,” arXiv preprint arXiv:2103.03604, 2021.   
[22] G. Xu, X. Wu, X. Zhang, and X. He, “LeViT-UNet: Make faster encoders with transformer for medical image segmentation,” arXiv preprint arXiv:2107.08623, 2021.   
[23] Y. Li, W. Cai, Y. Gao, and X. Hu, “More than encoder: Introducing transformer decoder to upsample,” arXiv preprint arXiv:2106.10637, 2021.   
[24] Y. Xie, J. Zhang, C. Shen, and Y. Xia, “CoTr: Efficiently bridging cnn and transformer for 3d medical image segmentation,” arXiv preprint arXiv:2103.03024, 2021.   
[25] W. Wang, C. Chen, M. Ding, J. Li, H. Yu, and S. Zha, “TransBTS: Multimodal brain tumor segmentation using transformer,” arXiv preprint arXiv:2103.04430, 2021.   
[26] D. Hendrycks and K. Gimpel, “Gaussian error linear units (gelus),” arXiv preprint arXiv:1606.08415, 2016.   
[27] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” arXiv preprint arXiv:1607.06450, 2016.   
[28] W. Luo, Y. Li, R. Urtasun, and R. Zemel, “Understanding the effective receptive field in deep convolutional neural networks,” in Proceedings of the 30th International Conference on Neural Information Processing Systems, pp. 4905–4913, 2016.   
[29] S. Liu, D. Huang, et al., “Receptive field block net for accurate and fast object detection,” in Proceedings of the European Conference on Computer Vision, pp. 385–400, 2018.   
[30] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 4, pp. 834–848, 2017.   
[31] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1–9, 2015.   
[32] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.   
[33] S. Zheng, J. Lu, H. Zhao, X. Zhu, Z. Luo, Y. Wang, Y. Fu, J. Feng, T. Xiang, P. H. Torr, et al., “Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6881–6890, 2021.   
[34] A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B. Landman, H. R. Roth, and D. Xu, “UNETR: Transformers for 3d medical image segmentation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 574–584, January 2022.   
[35] X. Huang, Z. Deng, D. Li, and X. Yuan, “MISSFormer: An effective medical image segmentation transformer,” arXiv preprint arXiv:2109.07162, 2021.   
[36] M. Antonelli, A. Reinke, S. Bakas, K. Farahani, B. A. Landman, G. Litjens, B. Menze, O. Ronneberger, R. M. Summers, B. van Ginneken, et al., “The medical segmentation decathlon,” arXiv preprint arXiv:2106.05735, 2021.   
[37] B. Landman, Z. Xu, J. E. Igelsias, M. Styner, T. Langerak, and A. Klein, “Miccai multi-atlas labeling beyond the cranial vault–workshop and challenge,” in Proc. MICCAI: Multi-Atlas Labeling Beyond Cranial Vault-Workshop Challenge, 2015.   
[38] O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, X. Yang, P.-A. Heng, I. Cetin, K. Lekadir, O. Camara, M. A. G. Ballester, et al., “Deep learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: Is the problem solved?,” IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514–2525, 2018.   
[39] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, Y. Burren, N. Porz, J. Slotboom, R. Wiest, et al., “The multimodal brain tumor image segmentation benchmark (brats),” IEEE Transactions on Medical Imaging, vol. 34, no. 10, pp. 1993–2024, 2014.   
[40] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnu-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature methods, vol. 18, no. 2, pp. 203–211, 2021.