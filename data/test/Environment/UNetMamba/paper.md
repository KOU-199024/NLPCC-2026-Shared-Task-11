# UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images

Enze Zhu , Zhan Chen , Dingkai Wang , Hanru Shi, Xiaoxuan Liu , and Lei Wang

Abstract—Semantic segmentation of high-resolution remote sensing images is vital in downstream applications such as land-cover mapping, urban planning and disaster assessment. Existing Transformer-based methods suffer from the constraint between accuracy and efficiency, while the recently proposed Mamba is renowned for being efficient. Therefore, to overcome the dilemma, we propose UNetMamba, a UNet-like semantic segmentation model based on Mamba. It incorporates a mamba segmentation decoder (MSD) that can efficiently decode the complex information within high-resolution images, and a local supervision module (LSM), which is train-only but can significantly enhance the perception of local contents. Extensive experiments demonstrate that UNetMamba outperforms the state-of-the-art methods with mIoU increased by 0.87% on LoveDA and 0.39% on ISPRS Vaihingen, while achieving high efficiency through the lightweight design, less memory footprint and reduced computational cost. The source code is available at https://github.com/EnzeZhu2001/UNetMamba.

Index Terms—Remote sensing, semantic segmentation, mamba

# I. INTRODUCTION

S Purred by the progress in advanced imaging techniques,high-resolution remote sensing images have become increasingly accessible, which can provide richer contents than regular images. Such intricate contents render them crucial for semantic segmentation in several downstream applications, yet simultaneously present significant challenges regarding accuracy and efficiency due to the high complexity of information.

As the foundation models of computer vision continuously evolve, research on semantic segmentation of high-resolution remote sensing images has flourished. The CNN-based method UNet [1] is a milestone in semantic segmentation, which has established the foundational architecture for subsequent works:

This work was supported in part by the Key Laboratory of Target Cognition and Application Technology under Grant 2023-CXPT-LC-005, and in part by the Science and Disruptive Technology Program under Grant AIRCAS2024- AIRCAS-SDTP-03. (Corresponding author: Xiaoxuan Liu.)

E. Zhu, Z. Chen, D. Wang and H. Shi are with the Aerospace Information Research Institute, Chinese Academy of Sciences (CAS), Beijing 100094, China, Key Laboratory of Target Cognition and Application Technology, CAS, Beijing 100190, China, and the School of Electronic, Electrical and Communication Engineering, University of Chinese Academy of Sciences, Beijing 100049, China (e-mail: {zhuenze23, chenzhan21, wangdingkai23, shihanru23}@mails.ucas.ac.cn).

X. Liu and L. Wang are with the Aerospace Information Research Institute, CAS, Beijing 100094, China, and Key Laboratory of Target Cognition and Application Technology, CAS, Beijing 100190, China (e-mail: {liuxiaoxuan, wanglei002931}@aircas.ac.cn).

an encoder-decoder framework with skip connections. With the success of Transformer [2], Wang et al. [3] introduce it into the U-shape framework, significantly improving the semantic segmentation accuracy of high-resolution remote sensing images. Consequently, numerous subsequent works have focused on innovating Transformer-based methods [4]– [6]. Despite the prosperity of Transformer in remote sensing semantic segmentation, its quadratic computational complexity and high parameter count [7] have inevitably limited the application efficiency in high-resolution images.

Recently, Mamba [8] has been proposed and attracted extensive attention. For efficiency, it achieves a linear complexity, while guaranteeing accuracy through its competitive long-distance dependency modeling capability. Subsequently, Mamba-based visual foundation models [9], [10] have been well-designed. Benefiting from such models, a series of Mamba-based UNets [11]–[13] have achieved impressive accuracy in medical imagery segmentation, showing the potential of visual Mamba in segmentation. For remote sensing imagery interpretation, visual Mamba has also demonstrated promising efficiency, with RSMamba [14] for classification, CDMamba [15] for change detection, and RS3Mamba [18] for semantic segmentation. However, these pioneering works have primarily validated the effectiveness of Mamba in their respective tasks, without fully leveraging its dual advantages in accuracy and efficiency through targeted design.

To overcome the dilemma between accuracy and efficiency and then achieve efficient semantic segmentation of highresolution remote sensing images, we propose a Mamba-based UNet-like model named UNetMamba. It consists of three main components: an encoder using ResT [19] backbone, a mamba segmentation decoder (MSD) for efficiently decoding, and a neatly-designed local supervision module (LSM) to enhance the perception of local semantic information. The contributions of our research are summarized as follows:

1) We devise MSD, a plug-and-play Mamba-based decoder for efficiently decoding semantic information from multi-scale feature maps. Specifically, visual state space (VSS) blocks of VMamba [9] are expressly transferred to the decoding side, significantly reducing the parameter count while leveraging the long-distance modeling capability to accurately decode complex information in a global receptive field.

2) We also conceive a CNN-based LSM to address the lack of local information perception in MSD. With two different scales of convolutions and an auxiliary loss function, we enhance the perception of local semantic information through supervised training. The train-only design also saves on inference costs, further ensuring the application efficiency.

![](images/cbb0641173d3fd08324d9bbdc84cf27ebe82bfed644a143ef7b276466d32cfeb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a) UNetMamba
        A["Input ∈ ℝ^H×W×3"] --> B["ResT Stage 1"]
        B --> C["ResT Stage 2"]
        C --> D["ResT Stage 3"]
        D --> E["ResT Stage 4"]
        E --> F["ResT Backbone"]
    end

    subgraph (b) VSS Block of MSD
        G["Linear"] --> H["Layer Norm"]
        H --> I["SS2D"]
        I --> J["SiLU"]
        J --> K["DW Conv"]
        K --> L["Linear"]
        L --> M["Layer Norm"]
        M --> N["SiLU"]
        N --> O["Linear"]
        O --> P["bypass"]

    subgraph (c) LSM Block of LSM
        Q["3×3 Conv"] --> R["Batch Norm"]
        R --> S["Batch Norm"]
        S --> T["ReLU6"]
        T --> U["Dropout"]
        U --> V["1×1 Conv"]
        V --> W["Upsample"]
        W --> X["F̃ ∈ ℝ^H×W×3"]

    B -->|H/4 × W/4 × 64| B
    C -->|H/8 × W/8 × 128| C
    D -->|H/16 × W/16 × 256| D
    E -->|H/32 × W/32 × 512| E
    B -->|VSS Block × 2| F["VSS Block × 2"]
    C -->|VSS Block × 2| G["VSS Block × 2"]
    D -->|VSS Block × 2| H["VSS Block × 2"]
    E -->|VSS Block × 2| I["VSS Block × 2"]
    F --> J
    G --> J
    H --> J
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P

    style (a) UNetMamba fill:#f9f,stroke:#333
    style (b) VSS Block of MSD fill:#ccf,stroke:#333
    style (c) LSM Block of LSM fill:#cfc,stroke:#333
```
</details>

Fig. 1. Framework of the proposed UNetMamba. (a) Overall architecture of UNetMamba. (b) Visual State Space (VSS) block of Mamba Segmentation Decoder (MSD). (c) Local Supervision Module (LSM) Block of LSM.

3) Based on MSD and LSM, we propose UNetMamba, a UNet-like model for efficient semantic segmentation of highresolution remote sensing images. It has the advantages of being lightweight and requiring low costs, while also achieving state-of-the-art (SOTA) performance on two well-known highresolution remote sensing imagery datasets.

# II. METHODOLOGY

As shown in Fig. 1(a), the overall architecture of UNet-Mamba is constructed under a U-shape framework [1], where a pre-trained ResT backbone [19] is selected as encoder while decoder is the proposed MSD. Moreover, the LSM is designed to enhance the perception of local semantic information.

# A. ResT Encoder

As illustrated in Fig. 1(a), with Efficient Transformer Block (ETB) as the core, ResT encoder is consisted of four stages to capture multi-scale feature maps. To avoid the quadratic computational costs of vanilla multi-head self-attention, ETB employs efficient multi-head self-attention (EMSA) [19]:

$$
\operatorname{EMSA} (Q, K, V) = \operatorname{IN} \left(\operatorname{Softmax} \left(\operatorname{Conv} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right)\right)\right) V \tag {1}
$$

where an instance normalization IN(·) and a 1×1 convolution Conv(·) are introduced to improve efficiency. When applied to high-resolution remote sensing images, EMSA can ensure the high encoding efficiency of our UNetMamba.

# B. Mamba Segmentation Decoder

Since high-resolution remote sensing images can be treated as very long sequences when flattened, the linear scaling ability of Mamba [8] has a natural advantage in processing them compared with the quadratic one of Transformer [2].

Actually, in pioneering studies [11]–[13], [18], the benefits of employing Mamba for semantic segmentation have been established. However, models that utilize a pre-trained VMamba [9] backbone (>30M) while encoding tend to be overweight for our goal of high efficiency. Therefore, we incorporate the basic unit of VMamba, namely, VSS block [9], into the decoding side only.

TABLE I QUANTITATIVE COMPARISON RESULTS ON THE LOVEDA DATASET AT A SIZE OF 1024 × 1024 PIXELS. THE BEST VALUES ARE IN BOLD. 

<table><tr><td>Method</td><td>Backbone</td><td>Background</td><td>Building</td><td>Road</td><td>Water</td><td>Barren</td><td>Forest</td><td>Agriculture</td><td>mIoU(%)</td></tr><tr><td>BANet [3]</td><td>ResT-Lite</td><td>41.92</td><td>54.13</td><td>52.39</td><td>76.18</td><td>11.74</td><td>45.07</td><td>51.71</td><td>47.59</td></tr><tr><td>MANet [5]</td><td>ResNet-50</td><td>44.38</td><td>56.20</td><td>56.03</td><td>78.41</td><td>17.06</td><td>45.81</td><td>58.17</td><td>50.87</td></tr><tr><td>DC-Swin [6]</td><td>Swin-T</td><td>40.75</td><td>57.81</td><td>56.55</td><td>79.59</td><td>16.58</td><td>46.49</td><td>57.76</td><td>50.79</td></tr><tr><td>UNetFormer [7]</td><td>ResNet-18</td><td>45.47</td><td>58.75</td><td>56.53</td><td>80.33</td><td>18.90</td><td>45.46</td><td>61.92</td><td>52.48</td></tr><tr><td>E-PyramidMamba [16]</td><td>ResNet-18</td><td>45.60</td><td>56.37</td><td>54.26</td><td>80.33</td><td>15.77</td><td>46.06</td><td>61.73</td><td>51.45</td></tr><tr><td>CM-UNet [17]</td><td>ResNet-18</td><td>43.57</td><td>55.59</td><td>52.85</td><td>77.80</td><td>16.19</td><td>43.15</td><td>58.36</td><td>49.64</td></tr><tr><td>RS $^3$ Mamba [18]</td><td>R18-VMamba-T</td><td>41.60</td><td>58.23</td><td>54.03</td><td>77.34</td><td>17.97</td><td>43.81</td><td>61.37</td><td>50.62</td></tr><tr><td>UNetMamba(ours)</td><td>ResT-Lite</td><td>47.08</td><td>59.16</td><td>56.74</td><td>81.37</td><td>18.15</td><td>46.61</td><td>64.31</td><td>53.35</td></tr></table>

TABLE IIQUANTITATIVE COMPARISON RESULTS ON THE ISPRS VAIHINGEN DATASET AT A SIZE OF 1024 × 1024 PIXELS.THE METRICS ARE MEASURED ON A SINGLE NVIDIA RTX 4090. THE BEST VALUES ARE IN BOLD.

<table><tr><td>Method</td><td>Backbone</td><td>Param(M)</td><td>Memo(MB)</td><td>FLOPs(G)</td><td>mF1(%)</td><td>mIoU(%)</td><td>OA(%)</td></tr><tr><td>BANet [3]</td><td>ResT-Lite</td><td>12.73</td><td>194.61</td><td>85.43</td><td>90.32</td><td>82.45</td><td>91.92</td></tr><tr><td>MANet [4]</td><td>ResNet-50</td><td>35.86</td><td>547.87</td><td>216.82</td><td>90.68</td><td>83.06</td><td>92.28</td></tr><tr><td>DC-Swin [5]</td><td>Swin-T</td><td>45.63</td><td>694.62</td><td>190.04</td><td>90.71</td><td>83.08</td><td>92.30</td></tr><tr><td>UNetFormer [7]</td><td>ResNet-18</td><td>11.72</td><td>179.34</td><td>47.03</td><td>90.59</td><td>82.93</td><td>92.21</td></tr><tr><td>E-PyramidMamba [16]</td><td>ResNet-18</td><td>28.76</td><td>439.15</td><td>75.97</td><td>90.74</td><td>83.08</td><td>92.29</td></tr><tr><td>CM-UNet [17]</td><td>ResNet-18</td><td>13.11</td><td>199.96</td><td>48.66</td><td>90.32</td><td>82.60</td><td>92.02</td></tr><tr><td>RS $^{3}$ Mamba [18]</td><td>R18-VMamba-T</td><td>43.32</td><td>662.21</td><td>157.89</td><td>90.73</td><td>83.08</td><td>92.27</td></tr><tr><td>UNetMamba(ours)</td><td>ResT-Lite</td><td>14.76</td><td>225.71</td><td>100.52</td><td>90.95</td><td>83.47</td><td>92.51</td></tr></table>

The structure of a VSS block is depicted in Fig. 1(b). The 2-D feature map F , which has undergone patch expansion, firstly goes through layer normalization and then splits into two flows. The main flow is further linearly embedded as $F ^ { \prime } { : }$

$$
F ^ {\prime} = \text { LinearEmbed } (\text { LayerNorm } (\text { PatchExp } (F))) \tag {2}
$$

Subsequently, $F ^ { \prime }$ undergoes a 3×3 depth-wise convolution and is then SiLU-activated into $F ^ { \prime \prime }$ :

$$
F ^ {\prime \prime} = \text { SiLU } \left(\text { DWConv } \left(F ^ {\prime}\right)\right) \tag {3}
$$

The 2-D Selective Scan (SS2D) module thereafter scans $F ^ { \prime \prime }$ in four different directions, decoding semantic information under a global receptive field and linear complexity:

$$
F _ {v} = \operatorname{ScanExp} \left(F ^ {\prime \prime}, v\right), \quad v \in \{1, 2, 3, 4 \}
$$

$$
\bar {F} _ {v} = \mathrm{S6} \left(F _ {v}\right), \quad v \in \{1, 2, 3, 4 \} \tag {4}
$$

$$
\bar {F} = \text { ScanMerge } \left(\bar {F} _ {1}, \bar {F} _ {2}, \bar {F} _ {3}, \bar {F} _ {4}\right)
$$

where the expansion and merging operation correspond to VMamba [9], while the S6 operation denotes the selective scan space state sequential model of Mamba [8]. The $\bar { F }$ then goes through layer normalization and an element-wise multiplication with the bypass flow, which has also undergone the same linear embedding and Silu activation. Finally, the VSS block outputs $F _ { M S D }$ through a linear layer with residual connection.

Based on aforementioned operations, the MSD decodes multi-scale feature maps at four different stages, ultimately outputting the semantic segmentation results through a 1×1 convolution head.

# C. Local Supervision Module

The large receptive field of SS2D in VSS blocks tend to be double-edged, causing MSD to partially overlook local semantic information while decoding. However, such details in high-resolution images are key factors to further improve the accuracy of semantic segmentation. Therefore, a CNN-based flexible LSM is proposed to enhance the perception of local details, as depicted in Fig. 1(c).

Considering the relatively smaller scale of land-covers in high-resolution scenarios, we adopt two parallel convolution branches in LSM with the kernel size set as 3 and 1, respectively. Each branch is followed by a batch normalization layer with a ReLU6 activation:

$$
\tilde {F} _ {i} ^ {\prime} = \text { ReLU6 } \left(\text { BatchNorm } \left(\text { Conv } _ {i} \left(F _ {M S D}\right)\right)\right) \tag {5}
$$

where $i \in \{ 1 , 3 \}$ denotes the kernel size, while $F _ { M S D }$ denotes the output feature of MSD at the corresponding stage. The two branches then merge into ${ \tilde { F } ^ { \prime } = \tilde { F } _ { 1 } ^ { ' } \oplus \tilde { \tilde { F } } _ { 3 } ^ { ' } }$ .

Eventually, through a dropout layer, a 1×1 convolution consistent with the segmentation head in MSD, and an upsampling layer to the original image size, LSM obtains F˜ ∈ RH×W ×3: $\tilde { F } \in \dot { \mathbb { R } } ^ { H \times \dot { W } \times 3 } ;$

$$
\tilde {F} = \text { Upsample } \left(\text { Conv } _ {1} \left(\text { Dropout } \left(\tilde {F} ^ {\prime}\right)\right)\right) \tag {6}
$$

During training, we add LSM blocks with different input channels to the decoder at stage 2-4 and then attain the segmentation result by summing up, which is further used in calculating the auxiliary loss function.

# D. Loss Function

The loss function utilized in training UNetMamba consists of two parts, a principal loss $L _ { p }$ for overall optimization and an auxiliary loss $L _ { a }$ for local supervision.

![](images/c94f11fbec17d137197c4401e0368a133e0d662f3fdb5eefe4b16057668bb85c.jpg)  
Fig. 2. Qualitative comparison on the LoveDA dataset at resolution of 1024 × 1024 pixels. (a) Origin, (b) Ground Truth, (c) the proposed UNetMamba, (d) BANet, (e) MANet, (f) DC-Swin, (g) UNetFormer, (h) E-PyramidMamba, (i) CM-UNet and (j) RS3Mamba.

![](images/a741ecc5543c573e1fc449e15745e560e31a4c62b75903949c6d612b55f4d6dd.jpg)  
Fig. 3. Qualitative comparison on the ISPRS Vaihingen dataset at resolution of 1024 × 1024 pixels. (a) Origin, (b) Ground Truth, (c) the proposed UNetMamba, (d) BANet, (e) MANet, (f) DC-Swin, (g) UNetFormer, (h) E-PyramidMamba, (i) CM-UNet and (j) RS3Mamba.

To guarantee the holistic segmentation performance of UNetMamba, two classic loss functions are selected to jointly form the principal loss $L _ { p } \colon$

$$
L _ {d i c e} = 1 - \frac {2}{N} \sum_ {n = 1} ^ {N} \sum_ {k = 1} ^ {K} \frac {\hat {y} _ {k} ^ {(n)} y _ {k} ^ {(n)}}{\hat {y} _ {k} ^ {(n)} + y _ {k} ^ {(n)}} \tag {7}
$$

$$
L _ {c e} = - \frac {1}{N} \sum_ {n = 1} ^ {N} \sum_ {k = 1} ^ {K} \hat {y} _ {k} ^ {(n)} \log y _ {k} ^ {(n)} \tag {8}
$$

where $L _ { d i c e }$ and $L _ { c e }$ represents dice loss and cross-entropy loss, respectively. And N denotes the number of samples while K denotes the number of categories. yˆ(nk $\hat { y } _ { k } ^ { ( n ) }$ is the k-th element in one-hot encoding of the true label for sample n, while $y _ { k } ^ { ( n ) }$ is the confidence of sample n belonging to category k.

Moreover, in order to efficiently achieve the local supervisory role of LSM, the most classic loss function in semantic segmentation $L _ { c e }$ [1] is also used as the auxiliary loss $L _ { a }$ .

Eventually, to balance the effects of $L _ { p }$ and $L _ { a }$ and then achieve optimal performance, the overall loss function for training UNetMamba is formulated as a weighted equation:

$$
L = L _ {p} + \alpha L _ {a} = \left(L _ {d i c e} + L _ {c e}\right) + \alpha L _ {c e} \tag {9}
$$

where the weight factor α is set as 0.4 [6].

# III. EXPERIMENTAL RESULTS

# A. Datasets

1) LoveDA: The LoveDA dataset [20] contains 5987 high-resolution remote sensing images at the same size of 1024×1024 pixels and includes 7 land-cover categories (background, building, road, water, barren, forest and agriculture). The dataset encompasses two scenes (urban and rural) collected from three Chinese cities, which brings considerable challenges due to multi-scale land-covers and complex contents. In our experiments, following the official settings [20], 2522 images were utilized for training, 1669 images for validation and 1796 images for testing without further cropping.

2) ISPRS Vaihingen: The ISPRS Vaihingen dataset contains 33 TOP (True OrthoPhoto) images, each with very high spatial resolution at an average size of 2494×2064 pixels. This widely-used remote sensing dataset involves 6 land-cover categories (impervious surface, tree, building, low vegetation, car and background). In the unbiased experiments, we randomly selected 9 images for testing, 2 images for validating with the remaining images for training, which was repeated for 5 times, and cropped all the 5 different sets of images into patches at a size of 1024×1024 pixels, respectively.

# B. Experimental Settings

All of the experiments were implemented with PyTorch on a single NVIDIA RTX 4090 GPU, and the optimizer was set as AdamW with a 0.0006 learning rate and a 0.00025 weight decay. For both of the datasets, the training epoch was set as 100 while the batch size was 8.

Moreover, the number of model parameters (Param), memory footprint (Memo) and floating point operation count (FLOPs) were selected for efficiency evaluation, while the mean F1-score (mF1), mean intersection over union (mIoU) and overall accuracy (OA) were chosen as accuracy evaluation metrics.

# C. Performance Comparison

We conducted a comprehensive performance comparison on LoveDA and ISPRS Vaihingen datasets. A series of SOTA optical remote sensing imagery semantic segmentation models were selected as our competitors, including Transformer-based models: BANet [3], MANet [4], DC-Swin [5], UNetFormer [6], and Mamba-based models: Efficient PyramidMamba (E-PyramidMamba) [16], CM-UNet [17], RS3Mamba [18].

1) Comparison on the LoveDA Dataset: As shown in Tab. I, the proposed UNetMamba achieved the best accuracy performance in quantitative comparison, with a significant improvement of 0.87% in mIoU. It is note-worthy that our UNetMamba achieved SOTA in six out of seven categories. In background and agriculture, UNetMamba led the second place by 1.48% and 2.39%, respectively. This distinct advantage indicated that, benefiting from MSD, our UNetMamba garnered excellent global perception ability, which made it capable of accurately segmenting large land-covers under such a high-resolution condition. The outstanding performance of UNetMamba in building, road and other small land-cover categories proved that the train-only LSM successfully enhanced the perception of local contents through supervision and generalization. The qualitative comparison results on LoveDA are illustrated in Fig. 2, which intuitively shows that the semantic segmentation mask obtained by UNetMamba is more accurate with less omission and clearer boundaries.

TABLE IIIABLATION STUDY RESULTS ON THE LOVEDA AND ISPRS VAIHINGENDATASET. THE METRICS ARE MEASURED BY A 1024 × 1024 INPUT ONA SINGLE NVIDIA RTX 4090, AND THE BEST VALUES ARE IN BOLD.

<table><tr><td rowspan="2">MSD</td><td rowspan="2">LSM</td><td rowspan="2">Param(M)</td><td rowspan="2">FLOPs(G)</td><td colspan="2">mIoU(%)</td></tr><tr><td>LoveDA</td><td>Vaihingen</td></tr><tr><td colspan="2">Baseline with ResT-Lite</td><td>42.14</td><td>183.64</td><td>52.64</td><td>83.23</td></tr><tr><td>√</td><td></td><td>13.89</td><td>100.52</td><td>52.61</td><td>83.18</td></tr><tr><td>√</td><td>√</td><td>14.76</td><td>100.52</td><td>53.35</td><td>83.47</td></tr></table>

2) Comparison on the ISPRS Vaihingen Dataset: The proposed UNetMamba not only achieved SOTA performance on the challenging LoveDA dataset, but also performed noticeably well on the classic Vaihingen dataset. As shown in Tab. II, in terms of accuracy, our UNetMamba achieved improvements of 0.21% in mF1, 0.39% in mIoU, and 0.21% in OA, while maintaining competitive efficiency through a lightweight model (14.76M) with low costs (225.71MB, 100.52G). Fig. 3 illustrates the qualitative comparison results, with the framed areas further demonstrating the competitiveness of UNetMamba in the perception of local semantic details, which is particularly important for the semantic segmentation accuracy in highresolution remote sensing scenes.

# D. Ablation Studies

To assess the effectiveness of the proposed MSD and LSM in UNetMamba, ablation studies were conducted on both of the datasets, with the results listed in Tab. III. For the sake of credibility, the baseline model, RS3Mamba [17], was updated by replacing its main encoding branch with the ResT [18] backbone consistent with UNetMamba. And after removing its Mamba-based auxiliary branch and upgrading its decoder into our MSD, the model experienced significant reduction in Param and FLOPs, with only minor drop in mIoUs, demonstrating the high efficiency of MSD in decoding complex semantic information. On both datasets, deployment of LSM also led to 0.74% and 0.29% increase in mIoUs, respectively. However, the cost was merely a modest parameter-count increase of 0.87M and a constant computation complexity due to the train-only scheme, which further confirming the high efficiency of LSM in enhancing the perception of local semantic information.

# IV. CONCLUSION

In this paper, an efficient semantic segmentation model called UNetMamba is proposed for high-resolution remote sensing images. Considering the multi-scale land-covers and complex information in such images, our UNetMamba features a plug-and-play Mamba-based segmentation decoder MSD for efficient semantic information decoding, and a train-only local supervision module LSM that can efficiently enhance the perception of local semantic details. Extensive experiments conducted on two well-known remote sensing datasets demonstrate that UNetMamba not only outperforms other SOTA models, but also achieves light weight and low costs. In the future, we will continue to explore promising linear mechanisms to further improve the accuracy and efficiency of our UNetMamba.

# REFERENCES

[1] O. Ronneberger, P. Fischer, T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” Springer International Publishing, Cham., UK, pp. 234–241, 2015.   
[2] A. Vaswani, N. Shazeer, N. Parmar et al., “Attention is all you need,” Adv. Neural Inf. Process. Syst., vol. 30, pp. 5998-6008, 2017.   
[3] L. Wang, R. Li, D. Wang et al., “Transformer meets convolution: A bilateral awareness network for semantic segmentation of very fine resolution urban scene images,” Remote Sens., vol. 13, no. 16: 3065, 2021.   
[4] R. Li, S. Zheng, C. Duan et al., “Multiattention network for semantic segmentation of fine-resolution remote sensing images,” IEEE Trans. Geosci. Remote Sens., vol. 60, 2022, Art no. 5607713.   
[5] L. Wang, R. Li, C. Duan et al., “A novel transformer based semantic segmentation scheme for fine-resolution remote sensing images,” IEEE Geosci. Remote Sens. Lett., vol. 19, 2022, Art no. 6506105.   
[6] L. Wang, R. Li, C. Zhang et al., “UNetFormer: A unet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery,” ISPRS J. Photogramm. Remote Sens., vol. 190, pp. 196-214, Aug. 2022.   
[7] H. Zhang, Y. Zhu, D. Wang et al., “A survey on visual mamba,” Appl. Sci., vol. 14, no. 13: 5683, 2024.   
[8] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” 2023, arXiv:2312.00752.   
[9] Y. Liu, Y. Tian, Y. Zhao et al., “VMamba: Visual state space model,” 2024, arXiv:2401.10166.   
[10] L. Zhu, B. Liao, Q. Zhang et al., “Vision Mamba: Efficient visual representation learning with bidirectional state space model,” 2024, arXiv:2401.09417.   
[11] J. Ruan and S. Xiang, “VM-UNet: Vision mamba unet for medical image segmentation,” 2024, arXiv:2402.02491.   
[12] J. Liu, H. Yang, H.-Y. Zhou et al., “Swin-UMamba: Mamba-based unet with imagenet-based pretraining,” 2024, arXiv:2402.03302.   
[13] Z. Wang, J. Zheng, Y. Zhang et al., “Mamba-UNet: UNet-like pure visual mamba for medical image segmentation,” 2024, arXiv:2402.05079.   
[14] K. Chen, B. Chen, C. Liu et al., “RSMamba: Remote sensing image classification with state space model,” IEEE Geosci. Remote Sens. Lett., vol. 21, 2024, Art no. 8002605.   
[15] H. Zhang, K. Chen, C. Liu et al., “CDMamba: Remote sensing image change detection with mamba,” 2024, arXiv:2406.04207.   
[16] L. Wang, D. Li, S. Dong et al., “PyramidMamba: Rethinking pyramid feature fusion with selective space state model for semantic segmentation of remote sensing imagery,” 2024, arXiv:2406.10828.   
[17] M. Liu, J. Dan, Z. Lu et al., “CM-UNet: Hybrid cnn-mamba unet for remote sensing image semantic segmentation,” 2024, arXiv:2405.10530.   
[18] X. Ma, X. Zhang, M.-O. Pun, “RS3Mamba: Visual state space model for remote sensing image semantic segmentation,” IEEE Geosci. Remote Sens. Lett., vol. 21 , 2024, Art no. 6011405.   
[19] Q. Zhang and Y.-B. Yang. “ResT: An efficient transformer for visual recognition,” Adv. Neural Inf. Process. Syst., vol. 34, pp. 15475-15485, 2021.   
[20] J. Wang, Z. Zheng, A. Ma et al., “LoveDA: A remote sensing land-cover dataset for domain adaptive semantic segmentation,” 2021, arXiv:2110.08733.