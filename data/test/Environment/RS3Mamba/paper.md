# RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation

Xianping Ma, Xiaokang Zhang, Member, IEEE, and Man-On Pun, Senior Member, IEEE

Abstract—Semantic segmentation of remote sensing images is a fundamental task in geoscience research. However, there are some significant shortcomings for the widely used convolutional neural networks (CNNs) and Transformers. The former is limited by its insufficient long-range modeling capabilities, while the latter is hampered by its computational complexity. Recently, a novel visual state space (VSS) model represented by Mamba has emerged, capable of modeling long-range relationships with linear computability. In this work, we propose a novel dual-branch network named remote sensing images semantic segmentation Mamba (RS3Mamba) to incorporate this innovative technology into remote sensing tasks. Specifically, RS3Mamba utilizes VSS blocks to construct an auxiliary branch, providing additional global information to convolution-based main branch. Moreover, considering the distinct characteristics of the two branches, we introduce a collaborative completion module (CCM) to enhance and fuse features from the dual-encoder. Experimental results on two widely used datasets, ISPRS Vaihingen and LoveDA Urban, demonstrate the effectiveness and potential of the proposed RS3Mamba. To the best of our knowledge, this is the first vision Mamba specifically designed for remote sensing images semantic segmentation. The source code will be made available at https://github.com/sstary/SSRS.

Index Terms—Visual State space model, Remote Sensing, Semantic Segmentation

# I. INTRODUCTION

Modern geoscience research heavily relies on a wide range of remote sensing data collected by satellite or aerial devices. These data play a crucial role in capturing the spectral characteristics of objects on Earth’s surface and offering accurate visual representations of both natural and human-made structures. Semantic segmentation methods aim to classify every pixel in remote sensing images into distinct categories, thereby assisting researchers in thoroughly exploring surface conditions. This automated analysis and interpretation approach supports various downstream tasks and applications, including land cover mapping, environmental monitoring, and disaster management.

In recent years, deep learning-based methods as a datadriven automated technology have revolutionized semantic segmentation methods. Compared with conventional methods

This work was supported in part by the Guangdong Provincial Key Laboratory of Future Networks of Intelligence under Grant 2022B1212010001 and National Natural Science Foundation of China under Grant 42371374 and 41801323. (Corresponding author: Man-On Pun; Xiaokang Zhang)

Xianping Ma and Man-On Pun are with the School of Science and Engineering, the Future Network of Intelligence Institute (FNii), the Chinese University of Hong Kong, Shenzhen, Shenzhen 518172, China (e-mail: xianpingma@link.cuhk.edu.cn; SimonPun@cuhk.edu.cn).

Xiaokang Zhang is with the School of Information Science and Engineering, Wuhan University of Science and Technology, Wuhan 430081, China (e-mail: natezhangxk@gmail.com).

based on expert knowledge and artificial designation, deep learning has the capability to autonomously extract effective features from data and generate predicted probabilities in an end-to-end manner. Currently, mainstream models in remote sensing can be categorized into two main types: CNNs and Transformers. The former can extract image features through stacked convolution operations while the latter can model long-range dependencies based on the self-attention mechanism. Nowadays, these methods have inspired a large number of remote sensing images processing models, including pure convolutional network [1], transformer-based network [2] and hybrid architecture [3].

However, despite their advantages, these models have their limitations when applied to remote sensing images. Compared with natural images, remote sensing images has the characteristics of complex scenes and significant variations in object scales. As a result, CNNs are constrained by their local receptive fields, making it challenging to grasp and learn the intricate representations. On the other hand, although Transformers possess the capability to learn long-range dependencies, their high computational complexity poses a significant challenge when considering model efficiency and memory footprint. Recently, Mamba [4], built upon the state space model (SSM) [5], has emerged as an alternative for establishing long-distance dependency relationships while maintaining linear computational complexity. Subsequently, Vim [6] and VMamba [7] have extended this mechanism into the field of image processing. In remote sensing, Pan-Mamba [8] proposed channel swapping Mamba and cross-modal Mamba for pan-sharpening. RSMamba [9] presented a multi-path VSS Block for large-scale image interpretation. These two methods directly replace the existing network with the VSS blocks. However, since most existing VSS-based models are trained from scratch, they are temporarily hard to compare with fully pre-trained CNNs and Transformers in terms of optimal performance.

To introduce the VSS module into remote sensing images semantic segmentation and cope with the aforementioned challenge, we propose an auxiliary branch strategy that utilizes VSS blocks to supply additional global information, aiding the convolution-based main branch in feature extraction. Furthermore, to address the disparity between global and local semantics, we introduce a CCM module to facilitate crossbranch semantic fusion. The contributions of this work can be outlined as follows:

• We propose RS3Mamba, marking the first exploration of the potential application of VSS-based models in remote sensing images semantic segmentation. It provides valuable insights for the future development of more efficient and effective VSS-based methods for remote sensing tasks.

![](images/2a14e0ea31cba197e260e3801fffcd71b039df61b55742c1b415bc671dffbcd7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Patch Embedding"] --> B["VSS block"]
    B --> C["F^l_a"]
    C --> D["Patch Merging"]
    D --> E["VSS block"]
    E --> F["F^2_a"]
    F --> G["Patch Merging"]
    G --> H["VSS block"]
    H --> I["F^3_a"]
    I --> J["Patch Merging"]
    J --> K["VSS block"]
    K --> L["F^4_a"]
    
    M["Residual block"] --> N["CCM"]
    N --> O["Residual block"]
    O --> P["CCM"]
    P --> Q["Residual block"]
    Q --> R["CCM"]
    R --> S["Residual block"]
    S --> T["CCM"]
    
    U["Decode block"] --> V["Decode block"]
    V --> W["Decode block"]
    W --> X["Decode block"]
    
    Y["X"] --> M
    Y --> U
    Y --> W
    Y --> X
```
</details>

Fig. 1. The overall architecture of RS3Mamba.

• We propose a novel CCM for feature fusion, which can bridge the features from the dual-branch encoder, making them more effective in representation learning of remote sensing images.   
• Extensive experiments on two well-known, publicly available remote sensing datasets, ISPRS Vaihingen and LoveDA Urban confirm that the proposed method holds significant advantages over existing methods based on CNNs and Transformers.

# II. METHODOLOGY

The RS3Mamba consists of three parts, namely the VSS auxiliary encoder, the residual main encoder including the CCM for cross-branch semantic fusion, and the decoder as depicted in Fig. 1. Specifically, the RS3Mamba model extracts image features via the main and auxiliary branches, each comprising four corresponding blocks. The features generated by the auxiliary branch are fed into the CCM of the corresponding scale within the main branch for feature fusion. After the feature extraction and feature fusion at four scales, multiscale features are obtained before they are fed into the decoder with skip connections to generate the prediction map. The decoder in UNetformer [10] is adopted in this work.

# A. Auxiliary encoder

The auxiliary encoder is constructed based on VSS blocks that models long-range dependencies [11]. The detailed structure is presented in Fig. 2(a), where 2D-selective-scan (SS2D) is the core computing unit of the VSS block. The SS2D method extends image patches in four directions to create four separate sequences. These sequences are then individually processed through the SSM. Finally, the resulting features are combined to generate the complete 2D feature map. Given the input feature map x, the output feature map x of SS2D can be formulated as:

![](images/40365b8be3c77a4893941c39037a200d75aba37ff2953dac0da5c1a06ec79741.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a)
        A["Layer Norm"] --> B["Linear"]
        B --> C["DWConv"]
        C --> D["SS2D"]
        D --> E["Layer Norm"]
        E --> F["Linear"]
        F --> G["⊗"]
    end

    subgraph (b)
        H["Layer Norm"] --> I["Global-Local Fuse Attention"]
        I --> J["⊕"]
        J --> K["Layer Norm"]
        K --> L["MLP"]
        L --> M["⊗"]
    end

    I --> N["Windou-based Mutihead Self-Attention"]
    N --> O["Global Branch"]
    O --> P["⊕"]
    P --> Q["DWConv"]
    Q --> R["Batch Norm"]
    R --> S["Conv"]
    S --> T["Conv"]
    T --> U["Batch Norm"]
    U --> V["Conv"]
    V --> W["Local Branch"]
    W --> X["⊕"]
    X --> Y["Conv"]
    Y --> Z["Batch Norm"]
    Z --> AA["Conv"]
    AA --> AB["Local Branch"]
    AB --> AC["⊕"]
    AC --> AD["Conv"]
    AD --> AE["Batch Norm"]
    AE --> AF["Conv"]
    AF --> AG["Local Branch"]
    AG --> AH["⊕"]
    AH --> AI["Conv"]
    AI --> AJ["Batch Norm"]
    AJ --> AK["Conv"]
    AK --> AL["Local Branch"]
    AL --> AM["⊕"]
    AM --> AN["Conv"]
    AN --> AO["Batch Norm"]
    AO --> AP["Conv"]
    AP --> AQ["Local Branch"]
    AQ --> AR["⊕"]
    AR --> AS["Conv"]
    AS --> AT["Batch Norm"]
    AT --> AU["Conv"]
    AU --> AV["Local Branch"]
    AV --> AW["⊕"]
    AW --> AX["Conv"]
    AX --> AY["Batch Norm"]
    AY --> AZ["Conv"]
    AZ --> BA["Local Branch"]
    BA --> BB["⊕"]
    BB --> BC["Conv"]
    BC --> BD["Batch Norm"]
    BD --> BE["Conv"]
    BE --> BF["Local Branch"]
    BF --> BG["⊕"]
    BG --> BH["Conv"]
    BH --> BI["Batch Norm"]
    BI --> BJ["Conv"]
    BJ --> BK["Local Branch"]
    BK --> BL["⊕"]
    BL --> BM["Conv"]
    BM --> BN["Batch Norm"]
    BN --> BO["Conv"]
    BO --> BP["Local Branch"]
    BP --> BQ["⊕"]
    BQ --> BR["Conv"]
    BR --> BS["Batch Norm"]
    BS --> BT["Conv"]
    BT --> BU["Local Branch"]
    BU --> BV["⊕"]
    BV --> BW["Conv"]
    BW --> BX["Batch Norm"]
    BX --> BY["Conv"]
    BY --> BZ["Local Branch"]
```
</details>

Fig. 2. (a) The detailed architecture of VSS block. (b) The detailed architecture of CCM.

$$
x _ {v} = \text { expand } (x, v), \tag {1}
$$

$$
\overline {{x}} _ {v} = S 6 (x _ {v}), \tag {2}
$$

$$
\overline {{x}} = \operatorname{merge} \left(\overline {{x}} _ {1}, \overline {{x}} _ {2}, \overline {{x}} _ {3}, \overline {{x}} _ {4}\right), \tag {3}
$$

where $v \in V = \{ 1 , 2 , 3 , 4 \}$ denotes four different scanning directions. expand(·) and merge(·) denotes the scan expand and scan merge operations in [7]. The selective scan space state sequential model (S6) described in Eq. (2) serves as the core operator of the VSS block. It facilitates interactions between each element in a 1D array and any of the previously scanned samples through a condensed hidden state. Please refer to [7] for more details about S6.

As depicted in Fig. 1, the auxiliary encoder consists of four successive stages, each consisting of a patch operator and a VSS block. The first stage contains one Patch Embedding layer followed by the VSS block, whereas the following three stages are equipped with one Patch Merging layer followed by the VSS block each. Denote by $X \in \bar { \mathbb { R } ^ { H \times W \times Z } }$ the input images. Z denotes the number of image channels, whereas H and W represent the height and width of the image, respectively. X are first split into non-overlapping patches by the Patch Embedding layer before extracting features by VSS block whose output is denoted by $F _ { a } ^ { I } .$ The following three stages perform similar operations and generate $F _ { a } ^ { 2 - 4 }$ . Notably, since the feature flow of the auxiliary branch are not affected by the main branch, we can perform the entire auxiliary branch firstly.

# B. Main encoder and CCM

The ResNet18 is adopted as the main encoder to learn the local representations. As depicted in Fig. 1, it contains four residual blocks and four CCMs. The four residual blocks perform convolution operations and generate multiscale features denoted by $F _ { m } ^ { I - 4 }$ . In contrast to the auxiliary encoder, the main encoder leverages the capabilities of the existing pre-trained model to efficiently extract features from remote sensing images. Hereby we incorporate the features from the auxiliary branch at each scale into the main branch by CCM to compensate for its limitation in extracting global information.

TABLE I EXPERIMENTAL RESULTS ON THE ISPRS VAIHINGEN DATASET. WE PRESENT THE OA OF FIVE FOREGROUND CLASSES AND THREE OVERALL PERFORMANCE METRICS. THE ACCURACY OF EACH CATEGORY IS PRESENTED IN THE F1/IOU FORM. BOLD VALUES ARE THE BEST. 

<table><tr><td>Method</td><td>Backbone</td><td>impervious surface</td><td>building</td><td>low vegetation</td><td>tree</td><td>car</td><td>mF1</td><td>mIoU</td></tr><tr><td>ABCNet [12]</td><td>ResNet-18</td><td>89.78/81.45</td><td>94.30/89.21</td><td>78.49/64.59</td><td>90.08/81.95</td><td>74.05/58.80</td><td>85.34</td><td>75.20</td></tr><tr><td>TransUNet [13]</td><td>R50-ViT-B</td><td>90.77/83.10</td><td>94.32/89.25</td><td>79.02/65.32</td><td>90.53/82.70</td><td>82.66/70.45</td><td>87.46</td><td>78.16</td></tr><tr><td>UNetformer [10]</td><td>ResNet-18</td><td>92.33/85.76</td><td>96.25/92.78</td><td>80.47/67.33</td><td>90.85/83.22</td><td>89.35/80.75</td><td>89.85</td><td>81.97</td></tr><tr><td>CMTFNet [14]</td><td>ResNet-50</td><td>92.53/86.09</td><td>96.95/94.09</td><td>79.98/66.64</td><td>90.22/82.19</td><td>89.87/81.60</td><td>89.91</td><td>82.12</td></tr><tr><td>RS3Mamba</td><td>R18-Mamba-T</td><td>92.83/86.62</td><td>96.82/93.83</td><td>80.84/67.84</td><td>91.10/83.66</td><td>90.09/81.97</td><td>90.34</td><td>82.78</td></tr></table>

The detailed structure of CCM is shown in Fig. 2(b). CCM employwhere cross-branch feature maps . Specifically, the CCM c $F _ { m } ^ { i }$ andrises $F _ { a } ^ { i }$ $i = \{ 1 , 2 , 3 , 4 \}$ two parallel branches, also namely the global branch and the local branch. The former is used to enhance $F _ { m } ^ { i }$ from the main branch, while the latter is used to process $F _ { a } ^ { i }$ from the auxiliary branch. Considering that the features of the main branch are obtained by convolution operations with local properties, we use windou-based multihead self-attention [15] for modeling long-range dependencies. It should be highlighted that this mechanism also maintains linear complexity. On the other hand, considering that the features of auxiliary branches are obtained by VSS with long-range properties, we use convolution to learn the details in local. Therefore, this fusion module is named the collaborative completion module in consideration of our completion of the double-branch features. At each of the four stages in the main branch, a CCM conducts feature fusion with the corresponding scale. The resulting fusion feature, denoted as $F _ { r } ^ { i }$ is then fed into the decoder as the skip connections. The decoder will recover the abstract features and give the final prediction map. The training objective is the cross entropy loss.

# III. EXPERIMENTS AND DISCUSSION

# A. Datasets

1) ISPRS Vaihingen: The ISPRS Vaihingen dataset consists of 16 true orthophotos, each boasting very high-resolution at an average size of 2500 × 2000 pixels. These orthophotos are composed of three channels: Near-Infrared, Red, and Green (NIRRG), with a ground sampling distance of 9 cm. The dataset encompasses five foreground classes, namely impervious surface, building, low vegetation, tree, car, and one background class (clutter). In our experiments, the 16 orthophotos are divided into a training set comprising 12 patches and a test set comprising 4 patches.   
2) LoveDA Urban: The LoveDA dataset offers a comprehensive collection of scenes, including Urban and Rural. In this work, we selected the LoveDA Urban scene due to its diverse distribution of ground objects. The LoveDA Urban scene comprises 1833 high-resolution optical remote sensing images with the size of 1024×1024 pixels. Each image provides three channels, namely Red, Green, and Blue (RGB), with a ground sampling distance of 30 cm. The dataset encompasses seven landcover categories, including background, building, road, water, barren, forest, and agriculture [16]. The 1833 images

are partitioned into two subsets, with 1156 images allocated for training and 677 images for testing.

The two datasets contrast in terms of sampling resolution, ground object categories, and label accuracy. By conducting experiments on both datasets, we can demonstrate the effectiveness and superiority of the RS3Mamba.

# B. Experimental Setup

We adopted UNetformer [10] based on ResNet18 [17] as our baseline model. The proposed RS3Mamba was benchmarked against several state-of-the-art supervised methods, including ABCNet [12], TransUNet [13], UNetformer [10], and CMTFNet [14]. The experiments were conducted using Py-Torch on a single NVIDIA GeForce RTX 4090 GPU equipped with 24GB RAM. Furthermore, Stochastic gradient descent (SGD) served as the optimization algorithm for training all models. A learning rate of 0.01, a momentum of 0.9, a decaying coefficient of 0.0005, and a batch size of 10 were employed. The total epochs were set to 50, with one test per epoch. For quantitative evaluation of the proposed approach, two widely used metrics were recorded: mean F1-score (mF1) and mean intersection over union (mIoU).

![](images/b3cddb2da5ced6cdb6f26c805f658ceae40372f4f3d521d756b7cf94589bbc08.jpg)

<details>
<summary>text_image</summary>

(a)
(b)
(c)
(d)
(e)
(f)
(g)
impervious surface
building
low vegetation
tree
car
</details>

Fig. 3. Qualitative performance comparisons on the ISPRS Vaihaigen with the size of $5 1 2 \times 5 1 { \bar { 2 } } .$ . (a) NIRRG images, (b) Ground truth, (c) ABCNet, (d) TransUNet, (e) UNetformer, (f) CMTFNet and (g) the proposed RS3Mamba. We showcase two samples for each model.

# C. Performance Comparison

1) Performance Comparison on the Vaihingen dataset: As shown in Table I, the proposed RS3Mamba exhibits great improvements in terms of mF1 and mIoU compared to the baseline UNetformer. This confirms that the dual-branch architecture based on VSS blocks effectively enhances feature extraction. When compared with existing state-of-the-art models, RS3Mamba outperforms in four classes: impervious surface, low vegetation, tree and car. Notably, RS3Mamba achieves a performance improvement of 0.30% and 0.53% in F1 and

TABLE II EXPERIMENTAL RESULTS ON THE LOVEDA URBAN DATASET. WE PRESENT THE OA OF FIVE FOREGROUND CLASSES AND THREE OVERALL PERFORMANCE METRICS. THE ACCURACY OF EACH CATEGORY IS PRESENTED IN THE F1/IOU FORM. BOLD VALUES ARE THE BEST. 

<table><tr><td>Method</td><td>Backbone</td><td>background</td><td>building</td><td>road</td><td>water</td><td>barren</td><td>forest</td><td>agriculture</td><td>mF1</td><td>mIoU</td></tr><tr><td>ABCNet [12]</td><td>ResNet-18</td><td>52.02/35.15</td><td>63.36/46.37</td><td>65.42/48.61</td><td>61.42/44.31</td><td>44.27/28.43</td><td>54.63/37.58</td><td>19.98/11.10</td><td>57.30</td><td>40.58</td></tr><tr><td>TransUNet [13]</td><td>R50-ViT-B</td><td>52.00/35.13</td><td>74.14/58.90</td><td>72.87/57.57</td><td>80.68/67.61</td><td>41.19/25.93</td><td>54.03/37.01</td><td>34.50/20.85</td><td>64.37</td><td>49.23</td></tr><tr><td>UNetformer [10]</td><td>ResNet-18</td><td>54.66/37.61</td><td>69.09/52.78</td><td>68.33/51.89</td><td>77.66/63.47</td><td>56.98/39.84</td><td>51.01/34.23</td><td>20.54/11.44</td><td>65.34</td><td>49.12</td></tr><tr><td>CMTFNet [14]</td><td>ResNet-50</td><td>56.09/38.98</td><td>74.18/58.96</td><td>67.11/50.50</td><td>70.3/54.27</td><td>47.00/30.72</td><td>54.45/37.41</td><td>41.92/25.65</td><td>62.95</td><td>46.68</td></tr><tr><td>RS3Mamba</td><td>R18-Mamba-T</td><td>56.86/39.72</td><td>74.02/58.75</td><td>73.35/57.92</td><td>75.78/61.00</td><td>54.27/37.24</td><td>56.80/39.67</td><td>50.72/33.98</td><td>66.86</td><td>50.93</td></tr></table>

![](images/c324cb916d51dde3a574f1d6085a1cd60cb2b20accc33999aae8c3d055be6426.jpg)

<details>
<summary>text_image</summary>

(a)
background
building
road
water
barren
forest
agriculture
(b)
(c)
(d)
(e)
(f)
(g)
</details>

Fig. 4. Qualitative performance comparisons on the LoveDA Urban with the size of 1024 × 1024. (a) NIRRG images, (b) Ground truth, (c) ABCNet, (d) TransUNet, (e) UNetformer, (f) CMTFNet and (g) the proposed RS3Mamba. We showcase two samples for each model.

IoU, respectively, for the impervious surface class compared to CMTFNet. Furthermore, the IoU for low vegetation is enhanced by 0.51%, and the IoU for building is enhanced by 0.44% compared to the baseline UNetformer. In terms of overall performance, RS3Mamba achieves an mF1 score of 90.34% and an mIoU of 82.78%, representing increases of 0.49% and 0.81%, respectively, compared to the corresponding performance of the baseline UNetformer. These improvements are primarily attributed to the global semantic information provided by the auxiliary branch. Fig. 3 illustrates visualization examples of the results obtained by all five methods under consideration. It is evident that RS3Mamba can more accurately segment ground objects with smoother borders and fewer noise points.

2) Performance Comparison on the LoveDA Urban: Despite differences in sample resolution and ground object categories between the two datasets, the experiments on the LoveDA Urban dataset showed similar results on the ISPRS Vaihingen dataset. As shown in Table II, the segmentation performance for background, road, forest and agriculture were 56.86%/39.72%, 73.35%/57.92%, 56.80%/39.67%, and 50.72%/33.98%, respectively, which amounts to an increase of 0.77%/0.74%, 0.48%/0.35%, 2.17%/2.09%, and 8.8%/8.33% in F1 and IoU, respectively, as compared to the state-of-the-art methods. In particular, our approach significantly improves the agriculture category, which is the most difficult to detect among existing methods, proving the potential of RS3Mamba for the recognition of difficult category. In terms of overall performance, RS3Mamba improves over the baseline an mF1 score of 1.52% and an mIoU of 1.81%, respectively. Fig. 4 presents visualization examples from the LoveDA Urban. The more complete identification for ground objects was apparent, which significantly demonstrates the effectiveness and necessity of global auxiliary information for

TABLE III ABLATION STUDIES ON THE DUAL-BRANCH AND CCM. (%). BOLD VALUES ARE THE BEST 

<table><tr><td>Res-b</td><td>VSS-b</td><td>CCM</td><td>mF1</td><td>mIoU</td></tr><tr><td>√</td><td></td><td></td><td>89.85</td><td>81.97</td></tr><tr><td></td><td>√</td><td></td><td>89.04</td><td>80.58</td></tr><tr><td>√</td><td>√</td><td></td><td>90.21</td><td>82.57</td></tr><tr><td>√</td><td>√</td><td>√</td><td>90.34</td><td>82.78</td></tr></table>

the representations learning of remote sensing images.

# D. Ablation Study

To verify the effectiveness of the proposed VSS auxiliary branch and CCM module inRS3Mamba, four ablation experiments were performed as shown in Table III. In addition to the decoder, our network consists of three parts in total, including the residual main branch (Res-b), VSS auxiliary branch (VSS-B), and CCM for feature fusion. The first row is our baseline method, named UNetformer, whereas the second row shows the case for the residual main branch being replaced by the VSS auxiliary branch. It can fully illustrate the current performance of VSS-based encoders on remote sensing images. Furthermore, the third row shows the case in which the dual-branch is adopted but we only use simple element-wise summation for feature fusion instead of CCM. Inspection of the first two rows in Table III suggests that the existing VSS-based backbones are still in the development stage, and due to its insufficient pre-training, it is hard to outperform the classic ResNet. Furthermore, the results in the third row illustrate the validity of our proposed dual-branch structure. As an auxiliary branch, VSS encoder can provide sufficient additional information to assist feature extraction and final semantic recovery. The results of the third and fourth rows demonstrate the effectiveness of our CCM, which further excavates the features effectively according to the structure characteristics of the dual-branch encoder. Table III confirmed that the proposed RS3Mamba is an effective framework to explore the VSS-based methods in remote sensing tasks.

# E. Model Complexity Analysis

The computational complexity of the RS3Mamba is evaluated using three metrics: floating-point operation count (FLOPs), model parameters, and memory footprint. FLOPs accesses into the model’s complexity, while model parameters and memory footprint evaluate the network’s scale and memory requirements, respectively. An ideal model would exhibit lower values in FLOPs, model parameters, and memory footprint.

TABLE IVCOMPUTATIONAL COMPLEXITY ANALYSIS MEASURED BY TWO 256 × 256IMAGES ON A SINGLE NVIDIA GEFORCE RTX 4090 GPU. MIOUVALUES ARE THE RESULTS ON THE ISPRS VAIHINGEN DATASET. BOLDVALUES ARE THE BEST.

<table><tr><td>Model</td><td>FLOPs (G)</td><td>Parameter (M)</td><td>Memory (MB)</td><td>MIoU(%)</td></tr><tr><td>ABCNet</td><td>7.81</td><td>13.67</td><td>1008</td><td>75.20</td></tr><tr><td>TransUNet</td><td>64.55</td><td>105.32</td><td>3122</td><td>78.16</td></tr><tr><td>UNetformer</td><td>5.87</td><td>11.69</td><td>1010</td><td>81.97</td></tr><tr><td>CMTFNet</td><td>17.14</td><td>30.07</td><td>1872</td><td>82.12</td></tr><tr><td>RS3Mamba</td><td>31.65</td><td>43.32</td><td>2332</td><td>82.62</td></tr></table>

Table IV presents the results of complexity analysis for all compared semantic segmentation models considered in this work. Inspection of Table IV shows that our RS3Mamba introduces additional model complexity compared to the baseline UNetformer. This is due to the incorporation of an additional auxiliary branch and the designation of the specific crossbranch feature fusion module. However, it is noteworthy that our approach exhibits a significant reduction in computational complexity and model scale compared to Transformer-based method TransUNet. These outcomes highlight that the effectiveness of our method as a practical way to introduce Mamba into remote sensing tasks. In the early stages of Mamba’s development, our method can provide valuable insights for the future development of Mamba in this field.

# IV. CONCLUSION

In this work, we proposed RS3Mamba that firstly introduce a VSS-based model into remote sensing images semantic segmentation tasks. An auxiliary branch based on VSS blocks is presented to provide additional global awareness information with minimal linear computational complexity. To bridge the distinct features of the two branches, we introduce the CAM module, enhancing the cross-branch features from a global and local perspective, respectively, before they are fused by element-wise addition. Compared with existing methods that directly replace CNNs and Transformers with complete VSS models, our method offers a unique way for integrating Mamba. Experimental evaluations conducted on two distinct remote sensing datasets demonstrate that RS3Mamba outperforms other state-of-the-art semantic segmentation methods based on CNNs and Transformers. We hope that our approach will stimulate further exploration of Mamba applications in remote sensing.

# REFERENCES

[1] F. I. Diakogiannis, F. Waldner, P. Caccetta, and C. Wu, “ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 162, pp. 94–114, 2020.   
[2] Z. Xu, W. Zhang, T. Zhang, Z. Yang, and J. Li, “Efficient transformer for remote sensing image segmentation,” Remote Sensing, vol. 13, no. 18, p. 3585, 2021.   
[3] X. Ma, X. Zhang, M.-O. Pun, and M. Liu, “A multilevel multimodal fusion transformer for remote sensing semantic

segmentation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.   
[4] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.   
[5] A. Gu, K. Goel, and C. Re, “Efficiently modeling long ´ sequences with structured state spaces,” arXiv preprint arXiv:2111.00396, 2021.   
[6] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, “Vision mamba: Efficient visual representation learning with bidirectional state space model,” arXiv preprint arXiv:2401.09417, 2024.   
[7] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, and Y. Liu, “Vmamba: Visual state space model,” arXiv preprint arXiv:2401.10166, 2024.   
[8] X. He, K. Cao, K. Yan, R. Li, C. Xie, J. Zhang, and M. Zhou, “Pan-mamba: Effective pan-sharpening with state space model,” arXiv preprint arXiv:2402.12192, 2024.   
[9] K. Chen, B. Chen, C. Liu, W. Li, Z. Zou, and Z. Shi, “Rsmamba: Remote sensing image classification with state space model,” arXiv preprint arXiv:2403.19654, 2024.   
[10] L. Wang, R. Li, C. Zhang, S. Fang, C. Duan, X. Meng, and P. M. Atkinson, “UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 190, pp. 196–214, 2022.   
[11] J. Liu, H. Yang, H.-Y. Zhou, Y. Xi, L. Yu, Y. Yu, Y. Liang, G. Shi, S. Zhang, H. Zheng, et al., “Swin-umamba: Mambabased unet with imagenet-based pretraining,” arXiv preprint arXiv:2402.03302, 2024.   
[12] R. Li, S. Zheng, C. Zhang, C. Duan, L. Wang, and P. M. Atkinson, “Abcnet: Attentive bilateral contextual network for efficient semantic segmentation of fine-resolution remotely sensed imagery,” ISPRS journal of photogrammetry and remote sensing, vol. 181, pp. 84–98, 2021.   
[13] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, and Y. Zhou, “Transunet: Transformers make strong encoders for medical image segmentation,” arXiv preprint arXiv:2102.04306, 2021.   
[14] H. Wu, P. Huang, M. Zhang, W. Tang, and X. Yu, “CMTFNet: CNN and multiscale transformer fusion network for remote sensing image semantic segmentation,” IEEE Transactions on Geoscience and Remote Sensing, 2023.   
[15] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.   
[16] J. Wang, Z. Zheng, A. Ma, X. Lu, and Y. Zhong, “Loveda: A remote sensing land-cover dataset for domain adaptive semantic segmentation,” arXiv preprint arXiv:2110.08733, 2021.   
[17] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.