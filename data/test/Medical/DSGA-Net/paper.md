# DSGA-Net: Deeply separable gated transformer and attention strategy for medical image segmentation network

![](images/5e02a3a85e304e255354d4edc1e0a3d6bdac09f4155dd2e6f165305e924d4f67.jpg)

Junding Sun a , Jiuqiang Zhao a , Xiaosheng Wu a , Chaosheng Tang a , Shuihua Wang a,b,c , Yudong Zhang a,b,c,⇑

a School of Computer Science and Technology, Henan Polytechnic University, Jiaozuo, Henan 454000, PR China   
b School of Computing and Mathematical Sciences, University of Leicester, Leicester LE1 7RH, UK   
c Department of Information Systems, Faculty of Computing and Information Technology, King Abdulaziz University, Jeddah 21589, Saudi Arabia

# a r t i c l e i n f o

Article history:

Received 21 November 2022

Revised 31 March 2023

Accepted 4 April 2023

Available online 11 April 2023

Keywords:

Medical image segmentation

Transformer

Gated attention mechanism

Depth separable

# a b s t r a c t

To address the problems of under-segmentation and over-segmentation of small organs in medical image segmentation. We present a novel medical image segmentation network model with Depth Separable Gating Transformer and a Three-branch Attention module (DSGA-Net). Firstly, the model adds a Depth Separable Gated Visual Transformer (DSG-ViT) module into its Encoder to enhance (i) the contextual links among global, local, and channels and (ii) the sensitivity to location information. Secondly, a Mixed Three-branch Attention (MTA) module is proposed to increase the number of features in the upsampling process. Meanwhile, the loss of feature information is reduced when restoring the feature image to the original image size. By validating Synapse, BraTs2020, and ACDC public datasets, the Dice Similarity Coefficient (DSC) of the results of DSGA-Net reached 81.24%,85.82%, and 91.34%, respectively. Moreover, the Hausdorff Score (HD) decreased to 20.91% and 5.27% on the Synapse and BraTs2020. There are 10.78% and 0.69% decreases compared to the Baseline TransUNet. The experimental results indicate that DSGA-Net achieves better segmentation than most advanced methods.

 2023 The Author(s). Published by Elsevier B.V. on behalf of King Saud University. This is an open access

article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).

# 1. Introduction

Medical image segmentation accurately describes organs, lesions, and other regions in medical images. It can help doctors make accurate and rapid clinical diagnoses and pathological research analyses. In clinical diagnosis, organ labeling requires personnel with solid professional knowledge to do it manually, which is time-consuming, laborious, and tedious. There are many problems to solve in medical image segmentation tasks, such as insufficient data and inaccurate labeling positions. Therefore, it has always been the focus of research in this field to realize medical image segmentation using computational vision (Yao et al., 2020; Liu et al., 2021; Cheng et al., 2022).

In recent years, with the development of deep learning technology, convolutional neural networks (CNN) have been widely used in medical image segmentation (Mu and Li, 2019; Philbrick et al., 2019; Tian et al., 2020). Long et al. (Long et al., 2015) proposed the Full Convolutional Network (FCN), which replaces the full connection layer with the convolution layer to extract image features and uses the upsampling layer to restore the image to its original size to obtain a more refined segmentation effect. Ronneberger et al. (Ronneberger et al., 2015) advance the U-Net network. It is based on the symmetric structure of the En-Decoder. It uses the skip connection to realize the transmission of the characteristic information between the En-Decoder, thus obtaining better segmentation performance. Zhang et al. (Diakogiannis et al., 2020) proposed a Residual Connection UNet (ResUNet), which replaces each submodule of U-Net with a residual connection module to obtain the deeper characteristics of the network, alleviate the problem of gradient disappearance and increase the convergence speed of the network. Oktay et al. (Oktay et al., 2018) proposed the Attention to U-Net (Atten-UNet) to enhance the sensitivity of the model to feature information. Zhan et al. (Zhan et al., 2023) proposed the Multi-view Attention Mechanism and Adaptive Fusion Strategy (CFNet) network. It uses Multi-view Attention Mechanism (MAM) for feature extraction and cross-scale feature

![](images/e73051a75d6fa04dd7400dd202da5c5a7400b83df97ae79fdecbd20457756d8f.jpg)

Production and hosting by Elsevier fusion in skip connection to effectively extract feature information in multiple receptive fields. The Fusion Weight Adaptive Allocation Strategy (FAS) is adopted to effectively guide the cross-scale fusion feature information input to the decoder to solve the semantic difference problem. Although CNN-based networks achieve good segmentation results, due to the innate boundedness of convolution and the difficulty in interacting with remote semantic information. It is impossible to model the feature information of remote dependencies in the feature graph, resulting in insufficient context feature information capture (Gu et al., 2018; Sun et al., 2019; Zhou, 2020).

To make up for the insufficiency of convolution, scholars introduced the Vision Transformer (Dosovitskiy et al., 2020) (ViT) model in the field of Natural Language Processing (NLP) into the field of medical image segmentation. The self-attention mechanism in the Transformer can solve the problem of remote information interaction, thus obtaining global feature information (Bitter et al., 2010; Khurana et al., 2023; Rezaii et al., 2022). Chen et al. (Chen et al., 2021) proposed the Transformers Make Strong Encoders (TansUNet) network, which combines Transformer with CNN to catch global spatial feature information and local feature information at the same time. However, when CNN upsamples the feature map in the Decoder part, the limitation that the convolved receptive field is too small in restoring the global feature to the initial resolution remains unchanged, which will disturb the segmentation of small organs in medical images. Hei et al. (Heidari et al., 2023) proposed the Hierarchical Multi-scale Representations Using the Transformers (HiFormer) network, which combines CNN and Transformer to obtain global and local features and designs a Double-Level Fusion (DLF) module on the hop connection. However, only global feature information can be obtained in the skip connection. It is difficult to obtain local feature information in the skip connection. Yuan et al. (Yuan et al., 2023) proposed CNN and Transformer Comprehensive Network (CTC-Net) based on U-Net. The encoder uses a combination of CNN and Swin-Transformer to propose a Cross-domain Fusion Block (CFB) to fuse the feature information extracted by CNN and Swin Transformer. A Feature Complementary Module (FCM) is proposed to improve feature representation, which combines cross-domain fusion, feature correlation, and dual attention. Gao et al. (Cao et al., 2022) proposed the Unet-like Pure Transformer (SwinUNet) network, which adopts the pure Swin Transformer method in the En-Decoder. However, Swin Transformer focuses on the interaction of global feature information. It neglects the extraction of low-resolution features, resulting in the introduction of a large amount of irrelevant information in the fusion of low-resolution and highresolution. Li et al. (Li et al., 2022) proposed the Separable Vision Transformer (SepViT) network by combining Depth Separable Self-Attention with grouped self-attention. The feature information inside windows and between windows is captured. However, during the upsampling process, many details cannot be recovered, and it isn’t easy to obtain the location information of context features. Gao et al. (Gao et al., 2021) proposed an efficient selfattention mechanism and relative position-coding to polymerize global context information at multiple scales. It can be problemsolving that the medical image processing data set is small and insensitive to location information. However, the resolution of shallow feature information is higher, and more location and detail information is not used effectively. Zhang et al. (Zhang et al., 2021) proposed the Fusing Transformers and CNNs (TransFuse) network architecture, the new fusion technology module is used to integrate the multi-level characteristics of each branch. However, the shallow information of the Encoder is not effectively used during the upsampling process, and the loss of the feature information recovered from the upsampling is not alleviated. Chen et al. (Chen et al., 2021) proposed the Multi-level Attention-guided U-

net with Transformer (TransAttUnet) network, designed multilevel lead attention and multiscale skip connection, which can mitigate the loss of fine details caused by convolution stack and continuous sampling operation. However, the feature information of the same layer with low resolution cannot be obtained at each level of the upsampling.

To solve the above problems, we make the following contributions:

(1) We propose the Deep Separable Gate Visual Transformer (DSG-ViT) Block. It can realize local information interaction within windows, global information between windows, and global information interaction between channels. While extracting features, the Deep Separable Gated Attention mechanism is used to increase the sensitivity of location information, which can solve the feature selection of organ location information and reduce the possibility of the organ being wrongly segmented.   
(2) We propose the Mixed Three Branch Attention (MTA) module, which will extract the rich semantic information of each Encoder layer and the accurate target location information to fuse with the Decoder and then compensate for the detailed information lost in the downsampling process and increase the feature information in the upsampling process.   
(3) We propose a novel medical image segmentation network called DSGA-Net, which uses a 4-layer Depth Separable Gated Visual Transformer (DSG-ViT) module as the Encoder part and a Mixed Three-branch Attention (MTA) module for feature fusion between each layer of the En-Decoder to obtain the final segmentation results, which achieves better results than previous models.

The rest of the paper is summarized as follows: Section 2 introduces the related work of the paper, Section 3 introduces the principles of the methods used and the overall structure of the model, Section 4 introduces the experimental dataset and results required for the experiment, and Section 5 is the conclusion.

# 2. Related work

# 2.1. Medical image segmentation based on CNN

The development of medical image segmentation is a process from manual segmentation to automatic computer segmentation. Currently, most medical treatments use computers to complete image segmentation alone, which has also become a hot issue for scholars at home and abroad to study.

With the wide application of CNN, FCN has achieved excellent results in medical image segmentation. Long et al. (Long et al., 2015) proposed the Full Convolution Network (FCN), which replaces the full connection layer with the convolution layer to extract the image features and uses the upper sampling layer to restore the image to its original size to obtain the segmentation results. The segmentation results of FCN, which is stacked in order, are usually very rough, so scholars have improved the FCN model. U-Net is an FCN model of Encoding and Decoding structure. It uses U-type Encoding and Decoding structures to obtain more feature information and improve segmentation accuracy.

In the CNN-based method, multi-scale fusion has been proven to improve the segmentation performance further. Based on the U-Net structure that has been introduced, Redesigning Skip Connections to Exploit Multiscale Features (U-Net++) (Zhou et al., 2019) and A Full-scale Connected Unet (U-Net 3++) (Huang et al., 2020) are proposed to add multi-scale structure on the jump connection between the Encoder and Decoder to extract richer context information and reduce the semantic difference between the low Encoder and the deep Decoder. Zhan et al. (Zhan et al., 2023) proposed the Multi-view Attention Mechanism and Adaptive Fusion Strategy (CFNet) network, which uses the novel Multi-view Attention Mechanism (MAM) for feature extraction and cross-scale method for feature fusion to obtain more accurate cross-scale fusion features in jump connection. At the same time, the Fusion Weight Adaptive Allocation Strategy (FAS) is adopted to effectively input the cross-scale fusion features to the Decoder to solve the semantic gap problem.

# 2.2. Medical image segmentation based on transformer

Due to the limited receptive field of the CNN model, the longterm dependence on modeling is limited. In order to make up for the limitations of CNN, Transformer’s model has become more and more widely used in medical image segmentation.

Cao et al. (Cao et al., 2022) proposed the SwinUnet model based on a pure Swin-Transformer’s U-shaped structure for segmenting two-dimensional medical images. In addition to applying pure Transformer, Chen et al. (Chen et al., 2021) combined CNN and Transformer to propose the TransUnet structure, by which CNN captures local feature information. Transformer captures global feature information, and the two combine to compensate for each other’s shortcomings. Gao et al. (Gao et al., 2021) proposed a UNEt TRansformers (UNETR) network, using a Transformer-based encoder for feature extraction and CNN based Decoder for final 3D medical image segmentation. In the past, the Encoder combined CNN and Transformer to extract the local and global information, often ignoring the important feature information between channels. The channel and spatial attention based on CNN is used in the skip connection between codecs. Because of the limitations of CNN, some important feature information will be lost.

We advance a novel medical image segmentation network model to solve the above problem with a Depth Separable Gating Transformer and a Three-branch Attention module (DSGA-Net). The model adds a Depth Separatable Gated Visual Transformer (DSG-ViT) module to its Encoder to extract features from global, local, and inter-channel feature information. Secondly, a Mixed Three-branch Attention (MTA) module is proposed to increase the number of features in the upsampling process. At the same time, when the feature image is restored to the original image size, the loss of feature information is reduced, and the accurate segmentation structure is achieved. Through verification on the Synapse, BraTs2020, and ACDC public datasets, the Dice similarity coefficients of DSGA-Net reached 81.24%, 85.82%, and 91.34%, respectively. Moreover, the Hausdorff Score (HD) was reduced to 20.91% and 5.27% on the Synapse and BraTs2020, and there are 10.78%, and 0.69% drops compared to the Baseline TransUNet. Experimental results show that the proposed method achieves better segmentation results compared to similar methods.

# 2.3. Separable visual transformer (SepViT)

Li et al. (Li et al., 2022) advance an efficient Transformer backbone called Separable Visual Transformer (SepViT). Its key design is Separable Self-Attention (Sep-Attention), which is made up of Deepwise Self-Attention (DWA) (Li et al., 2022) and Pointwise Self-Attention (PWA) (Li et al., 2022).

DWA is used to capture the local features inside each window. Each window can be regarded as an input channel of the characteristic diagram. Different windows cover different information and create a win token for each window, which can integrate the spatial information in each channel.

$$
\mathrm{DWA} (z) = \text { Attention } (z \cdot W _ {Q}, z \cdot W _ {K}, z \cdot W _ {V}) \tag {1}
$$

where z is the feature tokens, which consist of the pixel and window tokens. $W _ { Q } , W _ { K } ,$ , and $W _ { V } ,$ denote three linear layers for query, key, and value computation in a regular self-attention. Attention means a standard self-attention operator that works on local windows.

PWA establishes the relationship between windows through window markers, mainly used to fuse cross-window information. The feature map and window mark are extracted from the output of DWA, and the attention map is finally generated through the Normalization Layer (LN) and a Gaussian Error Linear Unit (GELU) (Li et al., 2022) activation function. At the same time, the feature graph is used as a branch of value in PWA, and the attention graph is used to calculate the attention between windows, thus realizing the global information interaction.

$$
\operatorname{PWA} (z, \omega t) = \text { Attention } \left\{\text { GELU } [ \mathrm{LN} (\omega t) ] \cdot W _ {Q}, \text { GELU } [ \mathrm{LN} (\omega t) ] \cdot W _ {K}, z \right\} \tag {2}
$$

where xt denotes the window token. Attention is a standard selfattention operator but works on all of the windows z.

# 2.4. Attention mechanism

Li et al. (Li et al., 2018) proposed the Pyramid Attention Network (PAN) network model. Its key design is the spatial feature pyramid attention module and the global attention upsampling module. The inter-feature pyramid attention module mainly uses different convolution kernels to draw feature information of different scales and then fuses the extracted feature information of different scales. It is helpful to obtain the correlation of adjacent feature information more accurately. Meanwhile, the similarity between features is extracted by multiplying the attention map after high-level semantic information and multi-scale fusion. Finally, the output results are added to the global attention upsampling. Its basic structure is shown in Fig. 1.

The global attention upsampling module comprises the global average pooling layer, 1 1 the convolution layer, and the upsam-pling layer. It uses the global average pooling to obtain the global semantic information.

# 3. Methodology

# 3.1. Proposed deep separable gate visual transformer (DSG-ViT)

Although the SepViT (Li et al., 2022) can learn global feature information within and between windows well by window tokens, SepViT is not accurate in learning the location of organs when training medical image segmentation data. Gao et al. (Gao et al., 2021) believed that Transformer needs a great quantity of data for training if it wants to learn accurate position deviation, and for the experiment of medical image segmentation with a smallscale data set, the position information learned has a large error. Shaw et al. (Shaw et al., 2018) have proved that relative positioncoding can encode the spatial structure of an image, so it is not always accurate in remote interaction. We propose a Deep Separable Gated Visual Transformer module (DSG-ViT) to solve this problem, as shown in Fig. 2.

The left side is the overall framework of DSG-ViT. We input tokens with window marks into the Deep Separable Gated Self-Attention (DSG-Attn) mechanism and then input the results of DSG-Attn into a layer of LN and MLP operations, fusing features of different self-attention forces.

In addition, we use residual structures to mitigate the phenomenon of gradient explosion and disappearance. The right side of Fig. 2 shows the internal structure of DSG-Attn. DSG-Attn consists of Depthwise Gate Self-Attention (DWGA) and Pointwise Gate

![](images/dba97619295f1c3d28e78ac6c13fc3d2c0e47026fa85e11293bbfe4bc3a85d08.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Layer"] --> B["Conv7x7"]
    B --> C["Conv5x5"]
    C --> D["Conv3x3"]
    D --> E["Conv3x3"]
    E --> F["Conv5x5"]
    F --> G["Conv7x7"]
    G --> H["Conv1x1"]
    H --> I["Conv1x1"]
    I --> J["×"]
    J --> K["+"]
    K --> L["Point-wise Add"]
    K --> M["Point-wise Multiply"]
    K --> N["Upsample"]
    L --> O["Output Layer"]
    M --> O
    N --> O
    O --> P["Final Output"]
    Q["Global Pooling"] --> R["Conv1x1"]
    R --> S["Upsample"]
    T["Conv1x1"] --> U["×"]
    U --> V["+"]
    V --> W["Point-wise Add"]
    V --> X["Point-wise Multiply"]
    V --> Y["Upsample"]
```
</details>

Fig. 1. Spatial feature pyramid attention mechanism module.

![](images/c04427f9c37758d5efddc1abd7c2e322ce07248e44ef8e41c074ca51b07df7fa.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["MLP"] --> B["+"]
    C["LN"] --> D["+"]
    E["DSG-Attn"] --> F["Win-tokens"]
    G["Win-tokens"] --> H["Depthwise Gate Self-attention"]
    I["Pointwise Gate Self-attention"] --> J["Win-tokens"]
    K["Win-tokens"] --> L["LN&Act"]
    M["Win-tokens"] --> N["Grid"]
    O["Grid"] --> P["Grid"]
    Q["Grid"] --> R["Grid"]
    S["Grid"] --> T["Grid"]
    U["Grid"] --> V["Grid"]
    W["Wq"] --> X["×"]
    Y["Wk"] --> Z["×"]
    AA["rQ"] --> AB["×"]
    AC["rV"] --> AD["×"]
    AE["Gz"] --> AF["×"]
    AG["Gy"] --> AH["×"]
    AI["softmax"] --> AJ["×"]
    AK["Gz"] --> AL["×"]
    AM["Gy"] --> AN["×"]
    AO["softmax"] --> AP["×"]
    AQ["Gz"] --> AR["×"]
    AS["softmax"] --> AT["×"]
    AU["softmax"] --> AV["×"]
    AW["softmax"] --> AX["×"]
    AY["Wv"] --> AZ["×"]
    BA["rK"] --> BB["×"]
    BC["Gz"] --> BD["×"]
    BE["Gy"] --> BF["×"]
    BG["Gz"] --> BH["×"]
    BI["Gz"] --> BJ["×"]
    BK["Gz"] --> BL["×"]
    BM["Gz"] --> BN["×"]
    BO["Gz"] --> BP["Gz"]
    BQ["Wq"] --> BR["×"]
    BS["Wk"] --> BT["×"]
    BU["rQ"] --> BV["×"]
    BW["rV"] --> BX["×"]
    BY["Gz"] --> BZ["×"]
    CA["Gy"] --> CB["×"]
    CC["Gz"] --> CD["×"]
    CE["Gz"] --> CF["×"]
    DG["Gz"] --> DH["×"]
    DI["Gz"] --> DJ["×"]
    DK["Gz"] --> DL["×"]
    DM["Gz"] --> DJ
    DN["Gz"] --> DJ
    DB["Gz"] --> DC["×"]
    DD["Gz"] --> DC
    DV["Gz"] --> DX["×"]
    DX --> DW["Gz"]
    DX --> DXB["Gz"]
    DXB --> DXC["Gz"]
    DXC --> DXD["Gz"]
    DXD --> DXE["Gz"]
    DXE --> DXF["Gz"]
    DXF --> DXG["Gz"]
    DXG --> DXH["Gz"]
    DXH --> DXI["Gz"]
    DXI --> DXJ["Gz"]
    DXJ --> DXK["Gz"]
    DXK --> DXL["Gz"]
    DXL --> DXM["Gz"]
    DXM --> DXN["Gz"]
    DXN --> DXO["Gz"]
```
</details>

Fig. 2. DSG-ViT.

Self-Attention (PWGA). Its basic principle is that based on Sep-Attn, it introduces the relative position-coding with a gated selfattention mechanism, embeds the position-coding into the DWA and the PWA, and introduces the gating mechanism that can control the position deviation to the global upper and lower coding, forming DWGA and PWGA. The Deeply Separable Gated Self-Attention mechanism (DWGA) is defined as follows:

$$
D _ {o} = z \sum_ {p \in N _ {m \times m} (o)} \operatorname{softmax} _ {p} \left(q _ {o} ^ {T} k _ {p} + G _ {Q} q _ {o} ^ {T} r _ {p - o} ^ {q} + G _ {K} k _ {p} ^ {T} r _ {p - o} ^ {k}\right) \left(G _ {V 1} v _ {p} + G _ {V 2} r _ {p - o} ^ {v}\right) \tag {3}
$$

Given an input feature map $\boldsymbol { X } \in \mathbb { R } ^ { C \times H \times W }$ , C denotes the channel 2number, H denotes the height of the feature map, W denotes the width of the feature map, z denotes the feature tokens, learning the global features of each window. N denotes the pixel of the whole feature map, o denotes a pixel $( i , j )$ in the feature map, $N _ { m \times m } ( \pmb { o } )$ ð Þis the local area of m m size centered on position:

$$
\left\{ \begin{array}{l} o = (i, j) \\ q _ {o} = W _ {Q} x _ {o} \\ k _ {o} = W _ {K} x _ {o} \\ v _ {o} = W _ {V} x _ {o} \end{array} \right. \tag {4}
$$

which is a linear mapping of ${ x _ { o } } \forall o \in N ,$ and $W _ { Q } , W _ { K }$ and $W _ { V }$ are learnable parameters. Since $q _ { o } ^ { T } , k _ { p }$ 8 2and $\upsilon _ { p }$ do not contain any position information, the relative position deviation terms $r _ { p - o } ^ { q } , r _ { p - o } ^ { k }$ and $r _ { p - o } ^ { \nu }$ are added for $q _ { o } ^ { T } , k _ { p }$ and $\nu _ { p } ,$ , respectively. $q _ { o } ^ { T } r _ { p - o }$ indicates the correlation from position $\boldsymbol { p } = ( a , b )$ to position $o = ( i , j ) , ~ G _ { \mathrm { Q } } , ~ G _ { K } ,$ , $G _ { V 1 }$ and $G _ { V 2 }$ ¼ ð Þ ¼ ð Þare the four learnable gated position embeddings. It provides information on whether spatial location can be learned correctly for the positional bias term and controls the effect of the learned relative position encoding on the accuracy of encoding global context information.

If a relative position code is accurately learned, the gating mechanism will give it a higher weight than those codes without accurate learning position information. In the case of the global receptive field, it can obtain more accurate spatial location information.

Li et al. (Li et al., 2022) proposed window markers and used PWA with window markers to fuse cross-window feature information, to better capture the feature information between windows. Position coding and controllable position deviation gating mechanisms are introduced to capture the feature information and accurately obtain the spatial position information between windows. Extract feature map and window tokens xt from the output of ð ÞDWGA, use window tokens and gating mechanism with position coding to build the attention relationship between windows and pose attention map. At the same time, the feature map is directly taken as the value matrix, and the attention calculation is carried out between windows using the attention map and the feature map to realize the global information interaction.

PWGA is calculated as follows:

$$
P _ {o} = z \cdot \omega t \sum_ {p \in N _ {m \times m} (o)} \operatorname{softmax} _ {p} \left(q _ {o} ^ {T} k _ {p} + G _ {Q} q _ {o} ^ {T} r _ {p - o} ^ {q} + G _ {K} k _ {p} ^ {T} r _ {p - o} ^ {k}\right) \tag {5}
$$

where z is the feature token, xt denotes the window token, $q _ { o } = W _ { Q } x _ { o } , \ k _ { o } = W _ { K } x _ { o }$ and $\nu _ { o } = W _ { V } x _ { o }$ is a linear mapping of ${ \boldsymbol { x } } _ { o } \forall 0 \in N .$ .

# 3.2. Proposed mixed three-branch attention mechanism

Mixed Three-branch Attention (MTA) is a mixed attention model which combines channel attention, spatial attention, and global context self-attention. It can map features from the three dimensions of channel, space, and global context, comprehensively improve the loss of extracted feature information and provide accurate feature information for the Decoder in recovering the feature map to the original size. We input the extracted feature map of each DSG layer into the corresponding MTA module for feature extraction and then perform feature fusion with the decoder of the corresponding layer. The structure is shown in Fig. 3.

Channels focus on the feature relationship between channels, mainly capture image texture features, and obtain information about each channel through maximum and average pooling, respectively. Given an input feature map $\boldsymbol { X } \in \mathbb { R } ^ { C \times H \times W }$ , C denotes 2the number of channels, H denotes the height of the feature map, W denotes the width of the feature map. MaxPooling and AveragePooling are used to extract feature information along the spatial axis, then feature maps $C _ { M } \in \mathbb { R } ^ { C \times 1 \times 1 }$ and $C _ { A } \in \mathbb { R } ^ { C \times 1 \times 1 }$ are respectively $1 \times 1$ 2 2 convolved, LeakyReLU activated, and then perform element-based addition operations. Finally, they use Sigmoid functions to obtain channel attention maps. The specific process is as follows:

$$
C _ {M} = \text { LeakyReLU } \{f _ {\text { Conv }} [ \text { MaxPool } (X) ] \}, C _ {M} \in \mathbb {R} ^ {C \times 1 \times 1} \tag {6}
$$

$$
C _ {A} = \text { LeakyReLU } \{f _ {\text { Conv }} [ \text { AveragePool } (X) ] \}, C _ {M} \in \mathbb {R} ^ {C \times 1 \times 1} \tag {7}
$$

$$
C (X) = \text { Sigmoid } (C _ {M} \oplus C _ {A}), C (X) \in \mathbb {R} ^ {C \times 1 \times 1} \tag {8}
$$

where $f _ { C o n t }$ denotes the convolution operation. $C _ { M }$ denotes the feature map of the MaxPooling output, $C _ { A }$ denotes the feature map of the AveragePooling output, C X denotes the channel attention ð Þmap,  denotes summation by pixel.

Since there is a large semantic difference between deep and shallow information in the downsampling process, and it is challenging to obtain spatial location information at different scales in the feature map, we designed multi-scale spatial attention. The input feature map is extracted with different scale feature information through three convolution kernels of different sizes to form a multi-scale pyramid structure. The convolution kernels are $7 \times 7 , 5 \times 5$ , and $3 \times 3 ,$ , respectively. The pyramid can effec-  tively fuse spatial information at different scales so that smaller details in the feature map can be effectively segmented in a larger receptive field, and larger features can be effectively segmented in a smaller receptive field.

Given a feature map $\boldsymbol { X } \in \mathbb { R } ^ { C \times H \times W }$ , the features are extracted 2along the channel axis, the feature vectors at different scales are generated by convolution at three different scales, and the Batch Normalization (BN) and Rectified Linear Units (ReLU) activation function operations are performed after each convolution at different scales. The specific operations are as follows:

$$
S _ {p} = \operatorname{ReLU} \left\{\mathrm{BN} \left[ f _ {1, p} ^ {k \times k} \left\{\operatorname{ReLU} \left[ \mathrm{BN} \left(f _ {2, p} ^ {k \times k} (X)\right) \right] \right\} \right] \right\} \tag {9}
$$

![](images/79549bad33b2d31c489b46f714e5002af4a790459c6524d204f013df40451beb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["GNLU"]
    B --> C["Imagesq"]
    C --> D["Multi-head self-attention"]
    D --> E["GNLU"]
    E --> F["MLP"]
    F --> G["Seq2img"]
    G --> H["Global Attention Map"]
    H --> I["Output"]
    
    subgraph Channel Attention
        J["CN3x3"] --> K["Sigmoid"]
        L["CN1x1"] --> M["Sigmoid"]
        N["Conv7x7"] --> O["Sigmoid"]
        P["Conv5x5"] --> Q["Sigmoid"]
        R["Conv3x3"] --> S["Sigmoid"]
        T["CN1x1"] --> U["Sigmoid"]
        V["Conv7x7"] --> W["Sigmoid"]
        X["Conv5x5"] --> Y["Sigmoid"]
        Z["Conv3x3"] --> AA["Sigmoid"]
    end
    
    subgraph Global Self-Attention
        AB["Global Attention Map"] --> AC["Sigmoid"]
        AD["Point-wise Add"] --> AE["CN1x1"]
        AF["Point-wise Multiply"] --> AG["CN1x1"]
        AH["Token"] --> AI["CN1x1"]
        AJ["Upsample"] --> AK["CN1x1"]
    end
    
    style Channel Attention fill:#f9f,stroke:#333
    style Global Self-Attention fill:#bbf,stroke:#333
```
</details>

Fig. 3. MTA Block.

where $S _ { p }$ denotes the feature vector, $f _ { 2 , p } ^ { k \times k }$ denotes the convolution operation with a convolution kernel of k, the stride of 2, and the padding of p, with a convolution kernel size of $k = \{ 3 , 5 , 7 \}$ , and a corresponding padding size of $p = \{ 1 , 2 , 3 \}$ ¼ f g. The output of the ¼ f gmulti-scale feature pyramid is the sum of the weights at different scales T:

$$
T = \sum_ {p = 1} ^ {3} S _ {p}, T \in \mathbb {R} ^ {1 \times H \times W} \tag {10}
$$

Finally, use the Sigmoid function to obtain the spatial attention map S:

$$
S = \operatorname{Sigmoid} (T), S \in \mathbb {R} ^ {1 \times H \times W} \tag {11}
$$

Global self-attention uses the Vision Transformer (Dosovitskiy et al., 2020) (ViT) network structure. ViT performs global modeling when acquiring feature information. The Multi headed Selfattention mechanisms enable the construction of correlations of global features that can capture global feature information. It can compensate for information loss in spatial and channel attention.

First, the feature map $\boldsymbol { X } \in \mathbb { R } ^ { C \times H \times W }$ is divided into patch sequences $\ b X _ { P } \in \mathbb { R } ^ { \ b C \times P \times P }$ , which $( P , P )$ denotes the size of each patch. 2 ð ÞSince the patch sequence has no location information, it is necessary to add a learnable location code $c _ { l o c } \in \mathbb { R } ^ { C \times P \times P }$ to the patch 2sequence and then obtain the attention sequence through a layer of LayerNormalization (LN). The attention sequence is input to the Multi-Headed Self-Attention (MHSA) to compute the similarity feature representation between patches. Then, a layer of LN and MLP operations are performed to fuse the feature extraction from different self-attentions. In addition, a residual structure is used to mitigate the occurrence of gradient explosion and vanishing phenomena. Finally, the feature sequence is transformed into a global attention map $G \in \mathbb { R } ^ { C \times H \times W }$ . The concrete implementation 2can be expressed as follows:

$$
\begin{array}{l} \mathbf {g} _ {m h s a} = \operatorname{MSHA} \left\{\mathrm{LN} \left[ f _ {\text { flat }} (X) \oplus c _ {\text { loc }} \right] \right\} \oplus \left\{\mathrm{LN} \left[ f _ {\text { flat }} (X) \right] \right\}, \mathbf {g} _ {m h s a} \\ \in \mathbb {R} ^ {C \times P \times P} \tag {12} \\ \end{array}
$$

$$
\mathbf {g} _ {m h s a} ^ {\prime} = \operatorname{MLP} \left[ \mathrm{LN} \left(\mathbf {g} _ {m h s a}\right) \right] \oplus \mathbf {g} _ {m h s a}, \mathbf {g} _ {m h s a} ^ {\prime} \in \mathbb {R} ^ {C \times P \times P} \tag {13}
$$

$$
G = \operatorname{SeqImg} \left(g _ {m h s a} ^ {\prime}\right), G \in \mathbb {R} ^ {C \times H \times W} \tag {14}
$$

where $f _ { f l a t }$ denotes a flatten operation, SeqImg denotes the operation of converting a sequence of features into a global attention map, $G \in \mathbb { R } ^ { C \times H \times W }$ denotes a global attention map,  denotes a 2pixel-by-pixel summation.

Multiply the obtained channel attention map $C \in \mathbb { R } ^ { C \times 1 \times 1 }$ with the spatial attention map $S \in \mathbb { R } ^ { 1 \times H \times W }$ 2to obtain the channel-space attention map $C S \in \mathbb { R } ^ { C \times H \times W }$ 2and finally, add CS with the global 2attention map G to obtain the mixed three-branch attention map $\mathbf { M T A } \in \mathbb { R } ^ { C \times H \times W }$

$$
\mathrm{MTA} = (C \otimes S) \oplus G, \mathrm{MTA} \in \mathbb {R} ^ {C \times H \times W} \tag {15}
$$

where MTA $\in \mathbb { R } ^ { C \times H \times W }$ denotes a mixed three-branch attention 2graph, denotes multiplication using the broadcast mechanism in Python, and denotes summation by elements.

# 3.3. Network structure of proposed DSGA-Net

In response to the problem of insufficient feature location information extracted by the currently available models and the loss of feature information in the upsampling process, we advance the DSGA-Net method. The overall structure of the proposed DSGA-Net consists of three parts: (i) The 4-layer Deep Separable Gate (DSG) Block, which constitutes the Encoder, (ii) the cascaded upsampling Decoder module, and (iii) the Mixed Three-branch Attention (MTA) module between the En-Decoder. The network structure is shown in Fig. 4.

The DSG Block that constitutes the Encoder has four layers, as shown in Fig. 5: DSG Block-1, DSG Block-2, DSG Block-3, and DSG Block-4. Each layer is composed of a different number of DSG-ViTs. The numbers of DSG-ViTs in each layer are 2, 2, 6, and 2, respectively. Each layer has a patch merge layer to downsample the preprocessed image and then input it to the DSG Block in the next layer to acquire the feature form of the feature map to extract the detailed feature information of organs effectively.

Because the shallow feature map is high resolution and contains more detailed features, the output of each layer of DSG Block is sent to the skip connection MTA to capture the feature details at different angles to generate the output of the Decoder. It uses low-level feature information to increase the receptive field at the upsampling stage and irrecoverable details. In the last layer of Encoder, because the underlying feature map is too small but contains rich feature information, only $3 \times 3$ convolution is used to transform the number of channels instead of using the MTA module.

Each layer in the Decoder consists of $\tt { a } 2 \times 2$ upsampling oper-ator, 3 3 convolution, and Relu activation function. Finally, the number of channels in the feature map reconstructed by 1  1 con-volution is reduced to the number of classes, and the final segmentation result is predicted.

# 4. Experiments, results, and analysis

# 4.1. Experimental dataset

Synapse Multi-Organ Segmentation Dataset (https://www. synapse.org/#!Synapse:syn3193805/wiki/217789, n.d.): We used the 30 abdominal CT scanning images and obtained 3779 axial enhanced abdominal clinical CT images. Each abdominal CT was composed of 85 98 slices with a slice pixel 512 512. We ran- domly divided the data according to a certain proportion, including 18 cases (2212 axial slices total). We trained eight abdominal organs (aorta, gallbladder, left kidney, right kidney, liver, pancreas, spleen, stomach), and 12 cases were verified. Among them, the slices of 18 training and 12 validation data are independent and do not overlap.

Automated Cardiac Diagnostic Challenge Dataset (https:// www.creatis.insa-lyon.fr/Challenge/acdc/, n.d.): The ACDC dataset is an electronic MRI image collected while examining different patients with an MRI scanner. Each patient’s MRI image contains $1 8 \sim 2 0$ slices while all being manually labeled by the physician with the true values of the left ventricle (LV), right ventricle (RV), and myocardium (MYO). We randomly divided it into 70 training samples (1930 axial slices), ten validation samples, and 20 test samples. Similar to (Chen et al., 2021), only the mean DSC was used to evaluate our model on this dataset.

Brain Tumor Segmentation Challenge 2020 Dataset (https://www.kaggle.com/datasets/awsaf49/brats20-dataset-trainingvalidation, n.d.): The BraTs2020 dataset has 369 brain tumor cases for model training and 125 cases for model validation. Each case contains four MRI scans of different modes (Flair, T1, T1ce, and T2), which are marked by experts in the field. Labels are divided into four categories: background, NCR/NET, ED, and ET. The evaluation results are based on three different brain tumor regions: the Whole Tumor (WT) region, the Tumor Core (TC) region,

$$
\left\{ \begin{array}{l} \mathrm{WT} = \frac {\mathrm{NCR}}{\mathrm{NET}} + \mathrm{ED} + \mathrm{ET} \\ \mathrm{TC} = \frac {\mathrm{NCR}}{\mathrm{NET}} + \mathrm{ET} \end{array} \right. \tag {16}
$$

and the Enhanced Tumor region (ET).

![](images/c9dfc4e2e7e1a734ad6c18951cceb1241c0ffc968e018b0af0de74b0b07cba09.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["DSG Block-1"]
    B --> C["DSG Block-2"]
    C --> D["DSG Block-3"]
    D --> E["DSG Block-4"]
    B --> F["MTA"]
    C --> G["MTA"]
    D --> H["MTA"]
    E --> I["Conv"]
    F --> J["Downsample"]
    G --> K["Segmentation head"]
    H --> L["Downsample"]
    I --> M["Feature Concatenation"]
    J --> N["Conv3x3,ReLU"]
    K --> O["Conv3x3,ReLU"]
    L --> P["Conv3x3,ReLU"]
    M --> Q["Conv3x3,ReLU"]
    N --> R["(16,H,W)"]
    O --> S["(192,H/8,W/8)"]
    P --> T["(384,H/16,W/16)"]
    Q --> U["(384,H/32,W/32)"]
    R --> V["(96,H/4,W/4)"]
    S --> W["(96,H/4,W/4)"]
    T --> X["(96,H/4,W/4)"]
    U --> Y["(96,H/4,W/4)"]
```
</details>

Fig. 4. Structure of the proposed DSGA-Net.

![](images/dedceacfee3bc8ac6834fdb55b94cd9858af95d909d9581163322e77a922fffa.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["DSG Block-1"] --> B["DSG Block-2"]
    B --> C["DSG Block-3"]
    C --> D["DSG Block-4"]
    style A fill:#cce5ff,stroke:#333
    style B fill:#cce5ff,stroke:#333
    style C fill:#66ccff,stroke:#333
    style D fill:#cce5ff,stroke:#333
    subgraph Encoder
        E["DSG-ViT"]
        F["DSG-ViT"]
    end
    subgraph Downsample
        G["DSG-ViT"]
        H["DSG-ViT"]
    end
```
</details>

Fig. 5. Structure of the Encoder.

# 4.2. Experiment setup and evaluation indicators

We use a 64-bit Windows 10 system with Nvidia GeForce RTX 2070 s graphics card based on Python 3.8 and Python 1.10.1 frameworks. During the training, simple data enhancement operations using flips and rotations were performed for all training cases.

We set the input feature map size to 224  224. The model is trained with the SGD optimizer. The learning rate is set to 0.01. The momentum parameter is set to 0.9. The weight recession is set to 0.0004, and the Batch size is set to 24. The evaluation metrics are used with the average Dice Similarity Coefficient (DSC) and the average Hausdorff Score (HD). DSC is used to measure the similarity of two regions, and the value is in the range of [0,1]. The larger the value, the more similar the two regions are. HD is used to calculate the distance between two regions. The smaller the value, the higher the similarity between the two regions.

# 4.3. Ablation experiments

We conducted ablation experiments under the Synapse multiorgan segmentation dataset to investigate the effect of different factors on the model. Specifically: (1) DSG-ViT with MTA ablation experiments; (2) the effect of the number of skip connections; (3) The effect of input resolution; (4) The effect of the number of DSG-ViT modules.

# 4.3.1. Influence of the DSG-ViT and MTA

To verify the effectiveness of DSG-ViT and MTA modules in DSGA-Net network architecture, we conducted four sets of experiments separately using TransUNet (TU) (Chen et al., 2021) as Baseline. As shown in Table 1, the first group is Baseline, the second group adds the DSG-ViT module to Baseline, the third group adds the MTA module to Baseline, and the fourth group adds both DSG-ViT and MTA modules to Baseline.

Table 1 Four groups of experiments. 

<table><tr><td>Group</td><td>Description</td><td>Short Name</td></tr><tr><td>1</td><td>Baseline</td><td>TU</td></tr><tr><td>2</td><td>Baseline + DSG-ViT</td><td>TU + DSG-VIT</td></tr><tr><td>3</td><td>Baseline + MTA</td><td>TU + MTA</td></tr><tr><td>4</td><td>Baseline + DSG-VIT + MTA</td><td>DSGA-Net</td></tr></table>

The experimental results are shown in Table 2. Compared with the first Baseline model, the second TU+(DSG-ViT) model improved the evaluation index DSC by 1.06% and reduced HD to 23.34%. The evaluation index DSC of the third TU + MTA model improved by 1.27%, and HD decreased to 23.65%. The fourth group is our model DSGA-Net, whose evaluation index DSC improved by 3.76%, and HD decreased to 20.91%.

Fig. 6 shows an example of visual segmentation for the above effect. We can see from the second group that when the MTA module was removed, there was over-segmentation of organs, such as mis-segmentation of the spleen as the liver, mis-segmentation of the left kidney as the right kidney (Fig. 6, Slice I, Group 2) and over-segmentation of the pancreas (Fig. 6, Slice II, Group 2).

The third group shows that when the DSG-ViT module was removed, the prediction of organ edges was not accurate enough, such as over-segmentation of the internal edges of the left kidney, under-segmentation of the edges of the stomach as well as the liver (Fig. 6, Slice III, Group 3), and rough edge segmentation of the pancreas with insufficient segmentation of the edge protrusions (Fig. 6, Slice II, Group 3).

# 4.3.2. Influence of the number of skip connections

To verify the influence of the amount of MTAs between En-Decoder on the segmentation effect, we experimentally tested four cases of no MTA (0-MTA), adding MTA (1-MTA) between DSG Block-1 and Decoder only, adding MTA (2-MTA) between DSG Block-1, DSG Block-2, and Decoder respectively, and adding MTA (3-MTA) between DSG Block-1, DSG Block-2, DSG Block-3, and Decoder respectively.

The experimental results are shown in Fig. 7, which shows the performance of the number of skip connections on the segmentation of 8 organs at the average DSC, and it can be seen from the figure that the 3-MTA case brings better segmentation results. The experimental results also show that the segmentation effect on small organs such as the aorta, gallbladder, kidney, and pancreas has more obvious improvement than large organs such as the liver and stomach. The results fully confirm the capability of the MTA module to draw detailed feature information.

# 4.3.3. Influence of the size of the input resolution

We chose 224 224 and 512 512 two different resolutions  for our experiments, and also compared our method, DSGA-Net, with TransUNet (Chen et al., 2021). The experimental results are shown in Table 3. From the experimental results, we can find out that when we use 512  512 as the input when the size of the segmented slices is consistent, the segmentation effect is improved for both our method and TransUNet (Chen et al., 2021).

This indicates that as the image resolution increases, the segmentation of the network structure also increases. It can also be seen from the experimental results that the change in image resolution has a greater effect on the segmentation effect of the Trans-UNet model and a relatively small effect on the DSGA-Net, which also further indicates that the method in our paper has better robustness. Since high-resolution images increase the computational burden of the network due to the increase in the number of slices, the resolution used in our subsequent experiments is the resolution of 224 224.

# 4.3.4. Influence of the number of DSG-ViTs in the DSG block module

We experimentally verify the impact of different numbers of DSG-ViTs in the Encoder’s DSG Block module on the DSGA-Net models’ segmentation performance. Given that the experimental results of (Liu et al., 2022) show that the number of four-layer modules DSG-ViT forming a ratio of 1:1:3:1 will achieve a better segmentation effect, for this reason, we experimentally tested three cases with the number of four-layer DSG-ViT modules (1, 1, 3, 1), (2, 2, 6, 2), and (3, 3, 9, 3), respectively, noted as ‘‘Base”, ‘‘Middle-100, and ”Middle-200, and the experimental results are shown in Table 4. From the experimental results, we can see that with the number of DSG-ViT modules increasing, the segmentation performance has also been significantly improved by 78.16%, 81.24%, and 82.23%, respectively.

It can also be seen from the experimental results that when the number of DSG-ViT modules is increased from ‘‘Base” to ‘‘Middle-1 , the segmentation performance is improved by 3.08%. When the number of DSG-ViT modules is increased from ”Middle-1 to ‘‘Middle-200, the segmentation performance is improved by 0.99%. The transition from ”Middle-100 to ‘‘Middle-200 will not improve performance, but the increased number of modules will cause greater computing consumption. For this purpose, we use ”Middle-1 as the Encoder, the structure shown in Fig. 3.

# 4.4. Comparison with state-of-the-art methods

# 4.4.1. Results for the synapse dataset

To further validate the segmentation performance of our model, we performed two sets of experiments. The first group compares our mode with CNN-based models, including U-Net (Ronneberger et al., 2015), Atten-Unet (Oktay et al., 2018), R50-UNet (Diakogiannis et al., 2020), and R50 Atten-UNet (Schlemper et al., 2019). The second group is the comparison with ViT (Dosovitskiy et al., 2020), CNN, and ViT combined models for comparison, including ViT (Dosovitskiy et al., 2020); SepViT (Li et al., 2022); R50-ViT (Dosovitskiy et al., 2020), TransUnet (Chen et al., 2021); and SwinUnet (Cao et al., 2022). To prove the correctness of the data results, we trained each set of experiments three times and then weighted them to obtain the final experimental results by averaging them. The experimental results are shown in Tables 5 and $6 ,$ and Figs. 8 and 9 show the segmentation visualization results of each network structure on the Synapse dataset.

Table 2 Validation of different modules of DSGA-Net. 

<table><tr><td rowspan="2">Group</td><td rowspan="2">Baseline</td><td rowspan="2">DSG-ViT</td><td rowspan="2">MTA</td><td colspan="2">Average</td><td rowspan="2">Aorta</td><td rowspan="2">Gallbladder</td><td rowspan="2">Kidney (L)</td><td rowspan="2">Kidney (R)</td><td rowspan="2">Liver</td><td rowspan="2">Pancreas</td><td rowspan="2">Spleen</td><td rowspan="2">Stomach</td></tr><tr><td>DSC↑</td><td>HD↓</td></tr><tr><td>1</td><td>√</td><td></td><td></td><td>77.48</td><td>31.69</td><td>87.23</td><td>63.13</td><td>81.87</td><td>77.02</td><td>94.08</td><td>55.86</td><td>85.08</td><td>75.62</td></tr><tr><td>2</td><td>√</td><td>√</td><td></td><td>78.54</td><td>23.34</td><td>87.48</td><td>65.76</td><td>81.67</td><td>76.76</td><td>95.37</td><td>56.74</td><td>84.86</td><td>79.64</td></tr><tr><td>3</td><td>√</td><td></td><td>√</td><td>78.75</td><td>23.65</td><td>87.79</td><td>68.03</td><td>81.34</td><td>76.88</td><td>94.64</td><td>55.21</td><td>87.79</td><td>78.31</td></tr><tr><td>4</td><td>√</td><td>√</td><td>√</td><td>81.24</td><td>20.91</td><td>88.21</td><td>70.87</td><td>82.67</td><td>82.31</td><td>95.76</td><td>58.49</td><td>90.87</td><td>80.74</td></tr></table>

![](images/9445642df74127fc6e61a0251a177ecb0185e7b343837206e0a2643bd9a16589.jpg)  
aorta gallbladder left kidney right kidney liver pancreas spleen stomach

Fig. 6. Comparison of DSGA-Net modules segmentation.   
![](images/660d0025d4087b01b267a064a45a6efdd2d2bca0e180b4060fc8d26ce67d181f.jpg)  
Fig. 7. Study of the number of skip connections added to DSGA-Net.

Table 3 Ablation experiments with input resolution. 

<table><tr><td>Method</td><td>Resolution</td><td>Average DSC</td><td>Aorta</td><td>Gallbladder</td><td>Kidney(L)</td><td>Kidney(R)</td><td>Liver</td><td>Pancreas</td><td>Spleen</td><td>Stomach</td></tr><tr><td>TransUNet (Chen et al., 2021)</td><td>224</td><td>77.48</td><td>87.23</td><td>63.13</td><td>81.87</td><td>77.02</td><td>94.08</td><td>55.86</td><td>85.08</td><td>75.62</td></tr><tr><td>TransUNet (Chen et al., 2021)</td><td>512</td><td>84.36</td><td>90.68</td><td>71.99</td><td>86.04</td><td>83.71</td><td>95.54</td><td>73.96</td><td>88.80</td><td>84.20</td></tr><tr><td>DSGA-Net</td><td>224</td><td>81.24</td><td>88.21</td><td>70.87</td><td>82.67</td><td>82.31</td><td>95.76</td><td>58.49</td><td>90.87</td><td>80.74</td></tr><tr><td>DSGA-Net</td><td>512</td><td>85.28</td><td>91.31</td><td>72.87</td><td>86.31</td><td>84.90</td><td>96.90</td><td>72.78</td><td>92.69</td><td>84.46</td></tr></table>

Table 4 Ablation experiments for the number of DSG-ViT modules. 

<table><tr><td>DSG-ViT</td><td>Average DSC</td><td>Aorta</td><td>Gallbladder</td><td>Kidney(L)</td><td>Kidney(R)</td><td>Liver</td><td>Pancreas</td><td>Spleen</td><td>Stomach</td></tr><tr><td>Base</td><td>78.16</td><td>87.34</td><td>65.28</td><td>81.37</td><td>80.69</td><td>94.27</td><td>56.64</td><td>86.31</td><td>79.36</td></tr><tr><td>Middle-1</td><td>81.24</td><td>88.21</td><td>70.87</td><td>82.67</td><td>82.31</td><td>95.76</td><td>58.49</td><td>90.87</td><td>80.74</td></tr><tr><td>Middle-2</td><td>82.23</td><td>88.94</td><td>71.31</td><td>84.09</td><td>83.85</td><td>96.22</td><td>59.21</td><td>91.98</td><td>82.21</td></tr></table>

Table 5 Comparison of segmentation performance of different CNN-based network structures on the Synapse dataset. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Average</td><td rowspan="2">FLPOs (G)↓</td><td rowspan="2">Aorta</td><td rowspan="2">Gallbladder</td><td rowspan="2">Kidney (L)</td><td rowspan="2">Kidney (R)</td><td rowspan="2">Liver</td><td rowspan="2">Pancreas</td><td rowspan="2">Spleen</td><td rowspan="2">Stomach</td></tr><tr><td>DSC↑</td><td>HD↓</td></tr><tr><td>U-Net (Ronneberger et al., 2015)</td><td>76.85</td><td>39.70</td><td>65.37</td><td>89.07</td><td>69.72</td><td>77.77</td><td>68.60</td><td>93.43</td><td>53.98</td><td>86.67</td><td>75.58</td></tr><tr><td>R50-UNet (Diakogiannis et al., 2020)</td><td>74.68</td><td>36.87</td><td>62.84</td><td>84.18</td><td>62.84</td><td>79.19</td><td>71.29</td><td>93.35</td><td>48.23</td><td>84.41</td><td>73.92</td></tr><tr><td>R50-AttenUNet (Schlemper et al., 2019)</td><td>75.57</td><td>36.97</td><td>56.31</td><td>55.92</td><td>63.91</td><td>79.20</td><td>72.71</td><td>93.56</td><td>49.37</td><td>87.19</td><td>74.95</td></tr><tr><td>Atten-UNet (Oktay et al., 2018)</td><td>77.77</td><td>36.02</td><td>58.47</td><td>89.55</td><td>68.88</td><td>77.98</td><td>71.11</td><td>93.57</td><td>58.04</td><td>87.30</td><td>75.75</td></tr><tr><td>DSGA-Net(ours)</td><td>81.24</td><td>20.91</td><td>37.26</td><td>88.21</td><td>70.87</td><td>82.67</td><td>82.31</td><td>95.76</td><td>58.49</td><td>90.87</td><td>80.74</td></tr></table>

Table 6 Comparison of segmentation performance of different models of CNN and ViT on the Synapse dataset. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Average</td><td rowspan="2">FLPOs (G)↓</td><td rowspan="2">Aorta</td><td rowspan="2">Gallbladder</td><td rowspan="2">Kidney (L)</td><td rowspan="2">Kidney (R)</td><td rowspan="2">Liver</td><td rowspan="2">Pancreas</td><td rowspan="2">Spleen</td><td rowspan="2">Stomach</td></tr><tr><td>DSC↑</td><td>HD↓</td></tr><tr><td>ViT (Dosovitskiy et al., 2020)</td><td>61.50</td><td>39.61</td><td>50.37</td><td>44.38</td><td>39.59</td><td>67.46</td><td>62.94</td><td>89.21</td><td>43.14</td><td>75.45</td><td>69.78</td></tr><tr><td>R50-ViT (Dosovitskiy et al., 2020)</td><td>71.29</td><td>32.87</td><td>54.07</td><td>73.73</td><td>55.13</td><td>75.80</td><td>72.20</td><td>91.51</td><td>45.99</td><td>81.99</td><td>73.95</td></tr><tr><td>SepViT (Li et al., 2022)</td><td>77.77</td><td>30.37</td><td>41.19</td><td>88.36</td><td>67.49</td><td>80.97</td><td>77.36</td><td>93.21</td><td>53.27</td><td>88.31</td><td>73.21</td></tr><tr><td>TransUnet (Chen et al., 2021)</td><td>77.48</td><td>31.69</td><td>38.52</td><td>87.23</td><td>63.13</td><td>81.87</td><td>77.02</td><td>94.08</td><td>55.86</td><td>85.08</td><td>75.62</td></tr><tr><td>SwinUnet (Cao et al., 2022)</td><td>79.13</td><td>21.55</td><td>42.68</td><td>85.47</td><td>66.53</td><td>83.28</td><td>79.61</td><td>94.29</td><td>54.96</td><td>90.66</td><td>76.60</td></tr><tr><td>DSGA-Net(ours)</td><td>81.24</td><td>20.91</td><td>37.26</td><td>88.21</td><td>70.87</td><td>82.67</td><td>82.31</td><td>95.76</td><td>58.49</td><td>90.87</td><td>80.74</td></tr></table>

![](images/7770e0bfe6067505531c2265a4d3735724fb1db231383632e7e9181ce2e18f7e.jpg)  
aorta gallbladder left kidney right kidneyliver pancreas spleen stomach   
Ground Truth   
DSGA-Net   
U-Net[8]   
R50-UNet[9]   
Atten-UNet[10]   
R50Atten-UNet[36]

Fig. 8. Segmentation results of different CNN-based models on Synapse Dataset.

![](images/7b48d6b2df2c107a8390c68ce0fe17dbfe99c9c4f7c1507f60050dd39489be41.jpg)  
Fig. 9. Segmentation Results of Different Variants of CNN and ViT on Synapse Dataset.

Tables 5 and 6 provide DSC and HD evaluation indicators and computational complexity indicators to reflect the segmentation performance of different methods. Compared to TransUNet and other pure CNN-based frameworks, experimental results show that DSGA-Net pays more attention to boundary information and can achieve better edge prediction.

For pure Transformer methods, DSGA-Net not only ensures sensitivity to boundaries but also prevents feature loss. After adding a deep separable self-attention mechanism and Transformer to traditional CNN, local context information, global context information, and information between channels are fused to improve the expression ability of output features. Then, U shaped structure is used for upsampling recovery to obtain better segmentation results. Our model has the lowest computational complexity using different methods on the Synapse dataset. Still, our model does not significantly reduce computational complexity compared to Trans-UNet, which is also the trend we will study next.

Fig. 8 shows the visualization of segmentation results between our model and other models based only on CNN. Due to the limitations of pure convolution modeling for a long time, the four models are over-segmented, and the segmentation of small organs is not obvious. Specifically, (Fig. 8, Slice I, (b-e)) show that the CNNonly based model identifies the right kidney as the left kidney for segmentation, (Fig. 8, Slice III, (b-e)) shows that the spleen is identified as the liver, and (Fig. 8, Slice II, (b-e)) show that the marginal segmentation of the small organ pancreas is not obvious and appears to be under-segmented.

Fig. 9 shows the segmentation visualization results between our model and the model combined with CNN and Transformer. ViT (Dosovitskiy et al., 2020) and SepViT (Li et al., 2022) are only based on the Transformer model. Because a pure Transformer can exchange global information but has limitations in learning location information; therefore, the two models significantly improved the recognition errors of left and right kidneys in abdominal multiorgan segmentation. However, a small part of the right kidney is still recognized as the left kidney for segmentation (Fig. 9, Slice I, (b-c)), and under-segmentation occurs, such as undersegmentation of the edge details of the spleen and liver (Fig. 9, Slice III, (b-c)). SwinUNet (Cao et al., 2022), TransUNet (Chen et al., 2021), and R50-ViT (Dosovitskiy et al., 2020) are segmentation models based on the combination of CNN and Transformer. It can interact with local and global information, but based on the U-shaped structure, the noise of shallow feature information is not filtered during the jump connection.

These five models make up for the shortcomings of the previous models but have limitations in the accuracy of the segmentation of small organs. For example, the segmentation of small organs such as the pancreas is under-divided, and the segmentation of the stomach is wrong (Fig. 9, Slice III, (d-f)). From (Fig. 9, Slice II, (df)), it can be seen that they are not obvious enough to obtain the location information of the pancreatic edge. Thus, the segmentation of the pancreatic edge is too smooth. The experimental results show that the DSGA-Net network structure is more accurate than other segmentation models.

# 4.4.2. Results for the ACDC dataset

We applied our model’s generalization ability and robustness to the MRI dataset ACDC of automatic heart segmentation for experiments to further verify our model’s generalization ability and robustness. Also, our method is compared with the CNN-based methods R50-UNet (Diakogiannis et al., 2020) and R50 Attend-UNet (Schlemper et al., 2019) and Transformer based methods R50-ViT (Dosovitskiy et al., 2020), SwinUet (Cao et al., 2022) and TransUnet (Chen et al., 2021); respectively, and the experimental results are shown in Table 7. Our model improves the DSC metrics by 3.79%, 4.95%, 3.77%, 1.34%, and 1.63% compared to R50-UNet (Diakogiannis et al., 2020), R50 Atten-UNet (Schlemper et al., 2019), R50-ViT (Dosovitskiy et al., 2020), SwinUnet (Cao et al., 2022) and TransUnet (Chen et al., 2021), respectively. It can be seen that the segmentation performance of our network structure has been significantly improved. In addition, our method has achieved significantly better results in the myocardium (Myo) and left ventricle (LV).

Table 7 Segmentation performance of different methods on the ACDC dataset. 

<table><tr><td>Method</td><td>Average DSC↑</td><td>RV</td><td>Myo</td><td>LV</td></tr><tr><td>R50-UNet (Diakogiannis et al., 2020)</td><td>87.55</td><td>87.10</td><td>80.63</td><td>94.92</td></tr><tr><td>R50 Atten-UNet (Schlemper et al., 2019)</td><td>86.75</td><td>87.58</td><td>79.20</td><td>93.47</td></tr><tr><td>R50-ViT (Dosovitskiy et al., 2020)</td><td>87.57</td><td>86.07</td><td>81.88</td><td>94.75</td></tr><tr><td>SwinUnet (Cao et al., 2022)</td><td>90.00</td><td>88.55</td><td>85.62</td><td>95.83</td></tr><tr><td>TransUnet (Chen et al., 2021)</td><td>89.71</td><td>88.86</td><td>84.53</td><td>95.73</td></tr><tr><td>DSGA-Net(ours)</td><td>91.34</td><td>88.78</td><td>88.39</td><td>96.87</td></tr></table>

Table 8 Segmentation performance of different methods on the BraTs2020 dataset. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Average</td><td rowspan="2">ET</td><td rowspan="2">TC</td><td rowspan="2">WT</td></tr><tr><td>DSC↑</td><td>HD↓</td></tr><tr><td>U-Net (Ronneberger et al., 2015)</td><td>80.12</td><td>8.34</td><td>75.34</td><td>76.39</td><td>88.64</td></tr><tr><td>R50-UNet (Diakogiannis et al., 2020)</td><td>80.70</td><td>7.05</td><td>75.87</td><td>77.34</td><td>88.90</td></tr><tr><td>R50-Atten-UNet (Oktay et al., 2018)</td><td>82.07</td><td>7.00</td><td>76.65</td><td>80.21</td><td>89.34</td></tr><tr><td>SwinUnet (Cao et al., 2022)</td><td>85.08</td><td>6.49</td><td>80.05</td><td>86.11</td><td>89.08</td></tr><tr><td>TransUnet (Chen et al., 2021)</td><td>84.30</td><td>5.96</td><td>79.37</td><td>85.21</td><td>88.31</td></tr><tr><td>DSGA-Net(ours)</td><td>85.82</td><td>5.27</td><td>81.68</td><td>86.00</td><td>89.78</td></tr></table>

# 4.4.3. Results for the BraTs2020 dataset

Our model is tested on the brain tumor segmentation data set BraTs2020 to verify the model’s generalization ability again by comparing with the CNN-based methods U-Net (Ronneberger et al., 2015), R50-UNet (Diakogiannis et al., 2020), and R50- Atten-UNet (Oktay et al., 2018) and Transformer based methods SwinUnet (Cao et al., 2022) and TransUnet (Chen et al., 2021), respectively. The experimental results are shown in Table 8. From the experimental results, the segmentation performance based on the Transformer method model is better than that based on the CNN method. Compared with U-Net (Ronneberger et al., 2015), R50-Unet (Diakogiannis et al., 2020), R50-Atten-Unet (Oktay et al., 2018), SwinUnet (Cao et al., 2022), and TransUnet (Chen et al., 2021), the DSC score of our model increased by 5.70%, 5.12%, 3.75%, 0.74%, and 1.52% respectively, and the HD score of our model decreased by 3.07%, 1.78%, 1.73%, 1.22%, and 0.69% respectively. It can be seen that our network structure has good segmentation performance and strong generalization ability. In addition, our method achieves better segmentation results for the whole tumor region (ET) and enhanced tumor region (WT).

# 5. Conclusion

We propose DSGA-Net, a medical image segmentation network with a Deeply Separable Gating Transformer and Attention strategies. Our approach enhances the contextual connection between global and local as well as between channels by adding a Depth Separable Gated Visual Transformer (DSG-ViT) module to the Encoder. We add a Mixed Three Branch Attention (MTA) module between the Encoder and Decoder to further extract the features of each layer of the Encoder through the spatial, channel, and global attention and input these features into the Decoder, increasing the feature information during the upsampling process. Many tests on medical image segmentation tasks have shown that our method provides a better improvement on the mis-segmentation and under-segmentation of small organs in medical image segmentation. Our method has a better segmentation effect and generalization ability than its counterparts.

However, the issue of a considerable computational burden on computers based on Transformer network structure is yet to be solved. Some devices may not support the global self-attention mechanism, which consumes a large amount of GPU memory and may limit the model’s usability.

In the future, we will go on to develop this algorithm and apply it to other segmentation tasks. We will explore ways to improve segmentation while reducing the number of parameters and increasing the speed of the computer.

# Funding

Anonymized.

# Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Acknowledgments

This work is supported by the National Natural Science Foundation of China (62276092); Key Science and Technology Program of Henan Province (212102310084); Key Scientific Research Projects of Colleges and Universities in Henan Province (22A520027); MRC, UK (MC\_PC\_17171); Royal Society, UK (RP202G0230); BHF, UK (AA/18/3/34220); Hope Foundation for Cancer Research, UK (RM60G0680); GCRF, UK (P202PF11); Sino-UK Industrial Fund, UK (RP202G0289); LIAS, UK (P202ED10, P202RE969); Data Science Enhancement Fund, UK (P202RE237); Fight for Sight, UK (24NN201); Sino-UK Education Fund, UK (OP202006); BBSRC, UK (RM32G0178B8).

# References

Bitter, C., Elizondo, D.A., Yang, Y., 2010. Natural language processing: a prolog perspective. Artif. Intell. Rev. 33 (1–2), 151.

Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., & Wang, M. (2023, February). Swin-unet: Unet-like pure transformer for medical image segmentation. In: Computer Vision–ECCV 2022 Workshops: Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part III (pp. 205-218). Cham: Springer Nature Switzerland   
Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., ... & Zhou, Y. (2021). Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.   
Chen, B., Liu, Y., Zhang, Z., Lu, G., & Kong, A. W. K. (2021). Transattunet: Multi-level attention-guided u-net with transformer for medical image segmentation. arXiv preprint arXiv:2107.05274.   
Cheng, Z., Qu, A., He, X., 2022. Contour-aware semantic segmentation network with spatial attention mechanism for medical image. Vis. Comput. 38 (3), 749–762.   
Diakogiannis, F.I., Waldner, F., Caccetta, P., Wu, C., 2020. ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data. ISPRS J. Photogramm. Remote Sens. 162, 94–114.   
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.   
Gao, Y., Zhou, M., & Metaxas, D. N. (2021). UTNet: a hybrid transformer architecture for medical image segmentation. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part III 24 (pp. 61-71). Springer International Publishing.   
Gu, J., Wang, Z., Kuen, J., Ma, L., Shahroudy, A., Shuai, B., Chen, T., 2018. Recent advances in convolutional neural networks. Pattern Recogn. 77, 354–377.   
Heidari, M., Kazerouni, A., Soltany, M., Azad, R., Aghdam, E. K., Cohen-Adad, J., & Merhof, D. (2023). Hiformer: Hierarchical multi-scale representations using transformers for medical image segmentation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 6202-6212).   
https://www.creatis.insa-lyon.fr/Challenge/acdc/.   
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation, https://www.synapse.org/#!Synapse:syn3193805/wiki/217789.   
Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., ... & Wu, J. (2020, May). Unet 3+: A full-scale connected unet for medical image segmentation. In: ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.   
Khurana, D., Koli, A., Khatter, K., Singh, S., 2023. Natural language processing: state of the art, current trends and challenges. Multimed. Tools Appl. 82 (3), 3713– 3744.   
Li, H., Xiong, P., An, J., & Wang, L. (2018). Pyramid attention network for semantic segmentation. arXiv preprint arXiv:1805.10180.   
Li, W., Wang, X., Xia, X., Wu, J., Xiao, X., Zheng, M., & Wen, S. (2022). Sepvit: Separable vision transformer. arXiv preprint arXiv:2203.15380.   
Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A convnet for the 2020s. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 11976-11986).   
Liu, X., Song, L., Liu, S., Zhang, Y., 2021. A review of deep-learning-based medical image segmentation methods. Sustainability 13 (3), 1224.

Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).   
Mu, C.C., Li, G., 2019. Research progress in medical imaging based on deep learning of neural network. Zhonghua kou Qiang yi xue za zhi= Zhonghua Kouqiang Yixue Zazhi=. Chinese J. Stomatol. 54 (7), 492–497.   
Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Rueckert, D. (2018). Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.   
Philbrick, K.A., Weston, A.D., Akkus, Z., Kline, T.L., Korfiatis, P., Sakinis, T., Erickson, B. J., 2019. RIL-contour: a medical imaging dataset annotation tool for and with deep learning. J. Digit. Imaging 32, 571–581.   
Rezaii, N., Wolff, P., Price, B.H., 2022. Natural language processing in psychiatry: the promises and perils of a transformative approach. Br. J. Psychiatry 220 (5), 251– 253.   
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.   
Schlemper, J., Oktay, O., Schaap, M., Heinrich, M., Kainz, B., Glocker, B., Rueckert, D., 2019. Attention gated networks: Learning to leverage salient regions in medical images. Med. Image Anal. 53, 197–207.   
Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. arXiv preprint arXiv:1803.02155.   
Sun, Y., Xue, B., Zhang, M., Yen, G.G., 2019. Evolving deep convolutional neural networks for image classification. IEEE Trans. Evol. Comput. 24 (2), 394–407.   
Tian, C., Fei, L., Zheng, W., Xu, Y., Zuo, W., Lin, C.W., 2020. Deep learning on image denoising: an overview. Neural Netw. 131, 251–275.   
Yao, X., Song, Y., Liu, Z., 2020. Advances on pancreas segmentation: a review. Multimed. Tools Appl. 79, 6799–6821.   
Yuan, F., Zhang, Z., Fang, Z., 2023. An effective CNN and Transformer complementary network for medical image segmentation. Pattern Recogn. 136, 109228.   
Zhan, B., Song, E., Liu, H., Gong, Z., Ma, G., Hung, C.C., 2023. CFNet: A medical image segmentation method using the multi-view attention mechanism and adaptive fusion strategy. Biomed. Signal Process. Control 79, 104112.   
Zhang, Y., Liu, H., & Hu, Q. (2021). Transfuse: Fusing transformers and cnns for medical image segmentation. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part I 24 (pp. 14-24). Springer International Publishing.   
Zhou, D.X., 2020. Universality of deep convolutional neural networks. Appl. Comput. Harmon. Anal. 48 (2), 787–794.   
Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N., Liang, J., 2019. Unet++: Redesigning skip connections to exploit multiscale features in image segmentation. IEEE Trans. Med. Imaging 39 (6), 1856–1867.