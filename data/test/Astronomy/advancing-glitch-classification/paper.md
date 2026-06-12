# Advancing Glitch Classification in Gravity Spy: Multi-view Fusion with Attention-based Machine Learning for Advanced LIGO’s Fourth Observing Run

Yunan Wu1, Michael Zevin2,3, Christopher P. L. Berry4, Kevin Crowston5, Carsten Østerlund5, Zoheyr Doctor3,6, Sharan Banagiri3, Corey B. Jackson7, Vicky Kalogera3,6, Aggelos K. Katsaggelos1,3

1 The Department of Electrical Computer Engineering, Northwestern University, 2145 Sheridan Road, Evanston, 60208, IL, USA   
2 The Adler Planetarium, 1300 South DuSable Lake Shore Drive, Chicago, 60605, IL, USA   
3 Center for Interdisciplinary Exploration and Research in Astrophysics (CIERA), Northwestern University, 1800 Sherman Ave, Evanston, 60201, IL, USA   
4 SUPA, School of Physics and Astronomy, University of Glasgow, Kelvin Building, University Ave, Glasgow, 8QQ, G12, UK   
5 School of Information Studies, Syracuse University, Hinds Hall, Syracuse, 13210, NY, USA   
6 Department of Physics and Astronomy, Northwestern University, 2145 Sheridan Road, Evanston, 60208, IL, USA   
7 Information School, University of Wisconsin–Madison, 600 N Park Street, Madison, 53706, WI, USA

E-mail: christopher.berry.2@glasgow.ac.uk

Abstract. The first successful detection of gravitational waves by ground-based observatories, such as the Laser Interferometer Gravitational-Wave Observatory (LIGO), marked a breakthrough in our comprehension of the Universe. However, due to the unprecedented sensitivity required to make such observations, gravitational-wave detectors also capture disruptive noise sources called glitches, which can potentially be confused for or mask gravitational-wave signals. To address this problem, a community-science project, Gravity Spy, incorporates human insight and machine learning to classify glitches in LIGO data. The machine-learning classifier, integrated into the project since 2017, has evolved over time to accommodate increasing numbers of glitch classes. Despite its success, limitations have arisen in the ongoing LIGO fourth observing run (O4) due to the architecture’s simplicity, which led to poor generalization and inability to handle multi-time window inputs effectively. We propose an advanced classifier for O4 glitches. Using data from previous observing runs, we evaluate different fusion strategies for multi-time window inputs, using label smoothing to counter noisy labels, and enhancing interpretability through attention modulegenerated weights. Our new O4 classifier shows improved performance, and will enhance glitch classification, aiding in the ongoing exploration of gravitational-wave phenomena.

# 1. Introduction

The first discovery of gravitational waves, a pivotal element in Einstein’s theory of general relativity [1], has opened up an entirely new window in the cosmos. The Laser Interferometer Gravitational-Wave Observatory (LIGO) [2] achieved groundbreaking success in detecting these ripples in spacetime for the first time in September 2015 [3]. Since then, the gravitational-wave detector network has been expanded with the addition of Virgo [4], which joined observations in August 2017 [5, 6], and KAGRA [7], which initiated its observing run in April 2020 [8]; this network has collected a large catalog of gravitational-wave observations [9]. However, collecting these observations demands exceptionally sensitive and intricate detectors in order to be able to measure minuscule fluctuations in spacetime [10]. This heightened sensitivity, in turn, leads to the detection of diverse sources of noise, with the potential to obscure or mimic authentic gravitational-wave signals [9, 11–13]. Bursts of non-Gaussian noise caused by environmental or instrumental factors, known as glitches, are particularly disruptive to measuring gravitational waves. The origins of many glitches remain largely unknown [14–16]. The process of characterizing and eliminating these glitches is essential to enhancing the quality of the detection system and the analysis of detector data.

In pursuit of a better understanding of detector noise, Gravity Spy [17, 18], a community-science project, has been launched to leverage the power of both volunteers and machine learning to classify glitches in LIGO data. On one front, the project engages human expertise in identifying and categorizing various glitches collected by LIGO. These labeled glitches then serve as training data to refine machine-learning classifiers, enhancing the accuracy and efficiency of glitch identification. Conversely, the classifiers provide guidance to volunteers, assisting them in identifying existing or new potential glitch classes [19]. Gravity Spy is implemented on the Zooniverse platform [20] on the community-science side, and it has demonstrated significant success by attracting over 30 thousand volunteers making more than 7 million glitch classifications.‡ The outputs are actively used by LIGO detector-characterization experts [12, 21]. One critical component contributing to the results of Gravity Spy is the integration of the machine-learning classifier.

The machine-learning classifier has been seamlessly integrated into the Gravity Spy pipeline for over five years, largely maintaining its original architecture. Periodically, fine-tuning of its final layer has been performed to enhance its adaptability to diversely predicted glitch classes [22]. The input for the classifier remains unchanged, featuring a structure that juxtaposes four spectrograms of the data from different time windows (0.5 s, 1.0 s, 2.0 s and 4.0 s, as shown in Figure 1) of a single glitch into one merged image. The same images are shown to volunteers on Zooniverse for their classifications [17].

The first version of the machine-learning classifier was introduced in the initial Gravity Spy paper [17] and was integrated into the Gravity Spy pipeline after the completion of the first observing run (O1). Designed for a 20-class classification setup, this classifier consisted of two convolutional layers and two fully-connected layers to extract image features. Each convolutional layer was followed by a max pooling layer to reduce the dimensionality of the features.

As the project progressed, during the second observing run (O2), Gravity Spy volunteers identified new glitch classes and two of them, 1080 Lines and 1400 Ripples, were added into the classifier. Therefore, the classifier was optimized by incorporating two additional convolutional layers and max pooling layers before the existing fullyconnected layers, which facilitated the extraction of more discriminative features and adapted the classifier to a 22-class classification task [23].

During the third observing run (O3), two new glitch classes, known as Lowfrequency Blip and Fast Scattering (or Crown), were identified through the collaborative efforts of volunteers and LIGO detector-characterization experts [24]. Alongside this, the decision was made to remove the None of the Above class from the training dataset. The None of the Above class was initially aimed at empowering our volunteers to identify new glitch classes. However, its inclusion confuses the model, especially given morphological similarities with other glitches. As a result, removing this class from the model allows the identification of new glitches more effectively with low predicted confidence [24, 25]. This led to further retraining of the classifier, enabling it to effectively accommodate a 23-class classification task [24].

Currently, the fourth observing run (O4) of the gravitational-wave detector network is in progress [26–28]. With the anticipation of discovering new glitch classes during O4 and subsequent runs, the current Gravity Spy classifier faces major limitations [25, 29, 30]. First, in the glitch classification task, unlike other image processing tasks, location plays a crucial role. For example, distinguishing between a line at the top and a line at the bottom of the image is essential as they represent different glitches at different frequencies. However, the classifier’s relatively straightforward architecture, comprised mainly of a combination of convolutional and fully-connected layers, hinders its capacity to capture unique features in individual glitch classes [31]. Second, due to its shallow layer structure, the classifier requires significant resizing of input image dimensions, resulting in the loss of valuable information. Moreover, previous studies [32] have shown that the classification accuracy of glitches varies depending on the time window used. However, earlier classifier studies [17, 32] relied on a single model to extract features from merged image inputs across multiple time windows, limiting their ability to capture cross-time window correlations [33]. Recognizing these challenges, it becomes apparent that the demand to analyze more data and diverse glitch types exceeds the current model’s capabilities.

Additionally, the current classifier grapples with a confidence overfitting issue [34, 35], displaying excessive certainty in glitch predictions even when errors are present. This tendency can potentially mislead both volunteers and LIGO detectorcharacterization experts, posing a risk to the efficiency and reliability of glitch characterization. Therefore, there is a compelling need for an advanced classifier architecture with robust generalization capabilities in the context of O4.

In this study, we develop a novel machine-learning classifier for glitch classification in the ongoing O4 run to address the challenges faced by Gravity Spy. The proposed classifier employs an attention-based multi-view fusion strategy to capture cross-time window correlations across the four time windows. The incorporation of a regularizer and label smoothing techniques into the loss function improves generalization and mitigates the confidence overfitting issue. The contributions of this paper can be summarized as follows: (1) we comprehensively compare three fusion strategies (early fusion, late fusion, and intermediate fusion) in scenarios with multiple time windows as inputs; (2) we apply label smoothing to mitigate the impact of noisy training labels, and (3) we introduce attention modules in the classifier, enhancing transparency in the glitch decision-making process by identifying specific window time that predominantly influences the glitch classification decision—an exploration undertaken for the first time.

The paper is organized as follows: In Section 2, we provide an overview of the Gravity Spy dataset, introduce three fusion strategies, outline the architecture of the proposed classifier, and elaborate on the conducted statistical analysis. Moving on to Section 3, we present the outcomes obtained from various ablation studies. In Section 4, we delve into a comprehensive discussion, addressing both the strengths and limitations of this study. We conclude in Section 5.

# 2. Materials and methods

In this section, we provide an overview of the Gravity Spy dataset, covering its datageneration process, image preprocessing, and the methodology for data splitting during both training and testing phases. Following this, we delve into the multi-view fusion strategy, discussing its early, intermediate, and late fusion components. Last, we propose a novel classifier, outlining the model architecture, attention modules, and the application of a label-smoothing technique.

# 2.1. Gravity Spy dataset

The Gravity Spy dataset comprises time–frequency spectrograms of glitches.

The glitches are initially identified by the Omicron pipeline [36]. Omicron searches for excess power in the data by performing a constant-Q transform [37], which projects the data onto a basis of Gaussian-windowed complex-valued exponential functions that tile the time–frequency plane (Q refers to the ratio of duration to bandwidth), and then searching for significant clusters of tiles. Omicron calculates signal-to-noise ratios (SNRs) for triggers from the magnitudes of the most significant time–frequency tiles [38]. To concentrate on glitches that pose significant challenges to gravitational-wave analyses (and are visible in spectrograms), the Gravity Spy pipeline only considers triggers with an Omicron SNR greater than 7.5, and where the peak frequency falls between 10 Hz and 2048 Hz.

![](images/056f87410f1d20f006be2367b23afbabd196940b8720ea518ed0fb474534bac6.jpg)  
Figure 1. Time–frequency spectrogram examples of two glitches with four time windows (0.5 s, 1.0 s, 2.0 s and 4.0 s): (a) Blip, characterized by its short duration, and (b) Whistle, characterized by its long duration. These examples illustrate how different time windows enable the analysis of glitches occurring at various temporal scales, providing valuable insights into the morphological characteristics of each glitch type.

Once data of interest have been identified, a constant-Q transform [38, 39] is employed to convert whitened LIGO strain data into time–frequency spectrograms, known as Omega scans [12]. Gravity Spy generates four spectrograms for each glitch triggered, using distinct time windows: 0.5 s, 1.0 s, 2.0 s and 4.0 s. Each window is centred on the Omicron trigger time. The incorporation of multiple time windows serves a crucial purpose, allowing both volunteers and machine-learning models to analyze the morphologies of glitches occurring at different characteristic timescales. For instance, this enables the examination of short- (e.g., Blip) and long-duration (e.g., Whistle) noise transients, as depicted in Figure 1.

The training and testing datasets employed in this study are sourced from a set of Gravity Spy project classifications [40], focusing exclusively on data from the Hanford and Livingston detectors during O3. This classification task has in total 23 glitch classes. To ensure consistency, the names associated with each class and the typical morphology of glitches belonging to each class were determined through a collaborative effort between LIGO scientists and citizen scientists. Examples representing each of the 23 glitch classes are illustrated in Figure 2, and more details are provided in the O1–O3 data release paper [25].

For the purposes of this study, we consider the ground truth for glitches in our dataset to be established through a collaborative effort involving volunteers and the previously implemented machine-learning classifier [17]. However, this ground truth

![](images/94224959b9a99406487bba742bce8bf139274b09a68ee57e74329b0fdc8a0db8.jpg)  
Figure 2. Time–frequency morphologies for each of the 23 glitch classes in the Gravity Spy dataset. Each image is a spectrogram, with all sharing the same axes and scale as for the Violin Mode glitch. These examples provide one possible morphological representation for each class, while each class can exhibit various shapes and patterns.

may exhibit a potential noisy-label issue, that is a few glitches were misclassified by the classifier and some volunteers. To strengthen the confidence in the ground truth used for our new classification model, we adopt a filtering criterion to create the training and testing dataset. Specifically, we included only glitches with consensus confidence scores above 0.9 from both machine-learning models and volunteer classifications, which accounts for 84.5% of the entire dataset. As a consequence, our new model is trained on glitches with relatively reliable labels, effectively reducing confusion during the training process and shielding the model from potentially mislabeled ground truth. Improved training sets for O4 data will be the focus of future studies.

Following the filtering criteria, the Gravity Spy dataset is divided into three distinct sets. The training set consists of 8439 glitches and the validation set consists of 1687 glitches. Additionally, a separate set of 3538 glitches is reserved for testing purposes. Glitches were selected randomly to be partitioned into the three sets. Each set is carefully curated to ensure no overlapping glitch subjects, thereby preventing data leakage between those three phases. A detailed breakdown of the data splitting is presented in Table 1.

The glitches from all three cohorts undergo the same pre-processing steps. Initially, each spectrogram image’s original dimension size is $6 0 0 \times 8 0 0 \times 3$ (height, width, color channel). We first crop the image with a bounding box defined as [topleft vertical:60, top-left horizontal:100, height:580, width:675] to retain only the intensity region of interest and eliminate irrelevant information, such as axes and scaling maps. Next, all cropped glitches are resized to a standardized size of $4 4 8 ~ \times$ $4 4 8 \times 3$ using bilinear interpolation to facilitate uniformity during model training. Finally, min–max normalization is applied to transform each pixel intensity X using $( X - X _ { \operatorname* { m i n } } ) / ( X _ { \operatorname* { m a x } } - X _ { \operatorname* { m i n } } )$ , where $X _ { \mathrm { m i n } }$ and $X _ { \mathrm { m a x } }$ are the minimal and maximum pixel intensities in the dataset. This transformation scales the pixel intensities to a range of 0 to 1, ensuring that all images are equally scaled, and enhancing the model’s ability to effectively learn from the data.

# 2.2. Multi-view fusion strategy

As Gravity Spy generates a dataset of glitches with four different durations, it is crucial to have a method for incorporating glitch information of varying characteristic timescales. Multi-view fusion, a powerful strategy in machine learning, combines information from multiple sources to enhance the representation and performance of a model. Each view contributes unique and complementary insights, leading to improved accuracy, robustness, and generalization of the model [33]. As mentioned above, our Gravity Spy project serves as an example where a single image with four time windows enables models to extract features across diverse timescales. Typically, multi-view fusion can employ one of three fusion architectures with different strategies to combine data from various view images. We review those below: the performance of these three fusion strategies will be compared in Section 3.

Table 1. The distribution of each class in the Gravity Spy O3 dataset across the training, validation, and testing sets. 

<table><tr><td rowspan="2">Gravity Spy class</td><td colspan="3">Dataset size</td></tr><tr><td>Train</td><td>Validation</td><td>Testing</td></tr><tr><td>1080 Lines</td><td>184</td><td>49</td><td>225</td></tr><tr><td>1400 Ripples</td><td>257</td><td>64</td><td>224</td></tr><tr><td>Air Compressor</td><td>293</td><td>57</td><td>225</td></tr><tr><td>Blip</td><td>312</td><td>84</td><td>218</td></tr><tr><td>Blip Low Frequency</td><td>337</td><td>103</td><td>222</td></tr><tr><td>Chirp</td><td>38</td><td>10</td><td>17</td></tr><tr><td>Extremely Loud</td><td>327</td><td>85</td><td>224</td></tr><tr><td>Fast Scattering</td><td>515</td><td>128</td><td>224</td></tr><tr><td>Helix</td><td>46</td><td>10</td><td>71</td></tr><tr><td>Koi Fish</td><td>338</td><td>93</td><td>224</td></tr><tr><td>Light Modulation</td><td>127</td><td>28</td><td>159</td></tr><tr><td>Low-frequency Burst</td><td>378</td><td>106</td><td>223</td></tr><tr><td>Low-frequency Lines</td><td>381</td><td>91</td><td>225</td></tr><tr><td>No Glitch</td><td>311</td><td>78</td><td>225</td></tr><tr><td>Paired Doves</td><td>124</td><td>31</td><td>146</td></tr><tr><td>Power Line</td><td>354</td><td>83</td><td>224</td></tr><tr><td>Repeating Blips</td><td>411</td><td>90</td><td>224</td></tr><tr><td>Scattered Light</td><td>521</td><td>128</td><td>224</td></tr><tr><td>Scratchy</td><td>178</td><td>36</td><td>225</td></tr><tr><td>Tomte</td><td>408</td><td>90</td><td>225</td></tr><tr><td>Violin Mode</td><td>210</td><td>56</td><td>225</td></tr><tr><td>Wandering Line</td><td>25</td><td>8</td><td>25</td></tr><tr><td>Whistle</td><td>462</td><td>129</td><td>224</td></tr></table>

2.2.1. Early fusion Early fusion takes place at the input level, where features from each view are combined in the input layer of the network before being fed into one single model. By merging data from multiple views at this stage, the model has access to a more comprehensive representation of the data. This approach was employed in previous Gravity Spy classifiers [17, 24, 32], which constructed a $2 h \times 2 w$ matrix by juxtaposing four $h \times w$ view images, as illustrated in Figure 3 (a).

Although it offers several advantages, early fusion has some limitations that may impact its practicality, particularly for large-sized images. In previous classifiers [17, 24, 32], the resizing of each view to $1 4 0 \times 1 7 0$ was required to fit within the model’s limitations, leading to a loss of valuable information. Additionally, early fusion assumes equal contributions to the task from all modalities, which may not hold true in this specific project. For instance, Repeating Blips [25] might only appear in the 4 s timewindow view and not in others, necessitating the model to assign greater weight to the

![](images/242af4c127d18e712048983dea041ab3126b7138d9151603cfca32929d54b1b8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image 1"] --> C["CNNs"]
    B["Input Image 2"] --> C["CNNs"]
    D["Input Image 3"] --> C["CNNs"]
    C --> E["Output Image"]
```
</details>

(a) Early Fusion

![](images/6401a94eb24a6a8bdb8eba1133532898f7d6980a39025381d1f8be7f59c5394a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image"] --> B["CNN-1"]
    C["Input Image"] --> D["CNN-2"]
    E["Input Image"] --> F["CNN-3"]
    G["Input Image"] --> H["CNN-4"]
    B --> I["1"]
    D --> J["2"]
    F --> K["3"]
    H --> L["4"]
    I --> M["Output NNs"]
    J --> M
    K --> M
    L --> M
    M --> N["Output"]
```
</details>

(c) Intermediate Fusion

![](images/b1331fa7cd6171ae6fff5b1e4ac60cefa85a2de1fd055dd3872f5354c7eaa0ba.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["CNN-1"] --> B["1"]
    C["CNN-2"] --> D["2"]
    E["CNN-3"] --> F["3"]
    G["CNN-4"] --> H["4"]
    B --> I["Aggregation"]
    D --> I
    F --> I
    H --> I
    I --> J["Output"]
```
</details>

(b) Late Fusion

![](images/d08b9c27fca26f69b09717707418d6eb2aab1dfb05ddc7fcc615778f27b9e9a9.jpg)

<details>
<summary>text_image</summary>

Prediction
Extracted features
Concatenation
</details>

Figure 3. Illustration of three fusion strategies: (a) early fusion; (b) late fusion, and (c) intermediate fusion. CNN: convolutional neural network; NNs: neural networks.

4 s window for accurate classification.

2.2.2. Late fusion Late fusion is a decision-level data-fusion technique, where predictions or outputs from individual models trained on different views are combined at the final stage, as shown in Figure 3 (b). This fusion process often involves aggregation methods, such as majority voting or averaging, to make the final decision. The advantage of late fusion is its flexibility in using different models for each view, enabling the selection of the most suitable model for each input. However, one drawback of late fusion is its potential to overlook cross-modal relationships, as fusion occurs at a higher decision level rather than at the earlier processing stages [41, 42]. Consequently, in our case, this might lead to suboptimal exploitation of valuable complementarity between different time-window views.   
2.2.3. Intermediate fusion Intermediate fusion, also known as feature-level fusion, allows features from each view to be processed separately through different branches of the network, and then combines the resulting feature maps at an intermediate layer, as shown in Figure 3 (c). This technique effectively addresses the limitations of both early fusion and late fusion. Specifically, by leveraging the complementarity of different views and capturing cross-modal relationships at a stage where features are still semantically meaningful, intermediate fusion enables the model to learn complex representations and potentially achieve better performance.

![](images/cc62636f4b78d757f2352d6a4dfa008d422465da325bf64f5c995766bd1d4c2e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph_Time_Series["0.5 s"]
        A1["Pre-Conv Block"] --> B1["Conv Block"] --> C1["Incep-Res Block"] --> D1["x3"]
        D1 --> E1["Post-Conv Block"] --> F1["Dropout"] --> G1["Flatten"] --> H1["FC"] --> I1["Attention Module"]
    end
    subgraph_Time_Series["1.0 s"]
        A2["Pre-Conv Block"] --> B2["Conv Block"] --> C2["Incep-Res Block"] --> D2["x3"]
        D2 --> E2["Post-Conv Block"] --> F2["Dropout"] --> G2["Flatten"] --> H2["FC"] --> I2["Attention Module"]
    end
    subgraph_Time_Series["2.0 s"]
        A3["Pre-Conv Block"] --> B3["Conv Block"] --> C3["Incep-Res Block"] --> D3["x3"]
        D3 --> E3["Post-Conv Block"] --> F3["Dropout"] --> G3["Flatten"] --> H3["FC"] --> I3["Attention Module"]
    end
    subgraph_Time_Series["4.0 s"]
        A4["Pre-Conv Block"] --> B4["Conv Block"] --> C4["Incep-Res Block"] --> D4["x3"]
        D4 --> E4["Post-Conv Block"] --> F4["Dropout"] --> G4["Flatten"] --> H4["FC"] --> I4["Attention Module"]
    end
    subgraph_AttentionModule["Attention Module"]
        I1 --> J1["Softmax"]
        I2 --> J2["Softmax +"]
        J1 --> K1["Dropout"]
        J2 --> K2["FC + Softmax"]
    end
    subgraph_TheSkipConnection["The skip connection"]
        K1 --> L1["Conv Block"]
        K1 --> L2["BN Max Pool"]
        K1 --> L3["Concatenate"]
        L1 --> M1["1x1 Conv"]
        L1 --> M2["1x1 Conv"]
        L1 --> M3["3x3 Conv"]
        L1 --> M4["5x5 Conv"]
        L1 --> M5["Max Pool"]
        L2 --> N1["Concatenate"]
        N1 --> O1["Concatenate"]
    end
```
</details>

Figure 4. Architecture of the proposed machine-learning classifier used in O4. Further elaboration on the hyperparameters can be found in Section 2.3. The ⊕ represents element-wise addition, while the ⊗ represents multiplication by attention weights in Eq. (1). FC: fully connected; BN: batch normalized.

# 2.3. Methods

The proposed classifier distinguishes itself from the past Gravity Spy glitch classifier through several key optimizations, addressing the limitations of previous models [17, 24, 25, 32]. First, it adopts intermediate fusion, effectively combining four views to enhance its representation power by leveraging their complementary information. Second, to handle a larger input dimension (higher resolution images), a deeper network is adopted, featuring the inception residual block [43] to mitigate the vanishing gradient problem and to extract more discriminative features. Additionally, an attention module is introduced at a later stage, providing valuable insights into the model’s focus on specific views during decision-making. Last, the incorporation of smooth labeling during training effectively addresses the noisy label issue. The architecture of the proposed classifier is shown in Figure 4, and detailed explanations of these optimizations are presented next.

2.3.1. Inception residual block The inception residual block combines the ideas of the inception module and the residual block proposed in Szegedy et al. [43]. As shown in Figure 4, it combines these concepts by applying the inception module within a residual block structure. Specifically, the block consists of four parallel convolutional branches with filters of different sizes $( 1 \times 1 , 1 \times 1 , 3 \times 3 , 5 \times 5 )$ . There is a $1 \times 1$ convolution, serving as a dimensionality-reduction layer, before the computationally expensive $3 \times 3$ and $5 \times 5$ convolutions that reduces the computational cost while preserving the network’s capability to learn complex patterns. The outputs of the four branches are concatenated, and the resulting feature map is combined with the original input after a $1 \times 1$ convolutional layer using a skip connection, followed by a rectified linear unit (ReLU) activation function. The use of parallel convolutional filters of different sizes allows the block to capture features at multiple scales, improving the model’s ability to recognize patterns of varying complexities [44]. This is useful in our case, as different glitch classes have time–frequency features at a variety of scales. Moreover, the incorporation of skip connections facilitates the training of deep networks, effectively mitigating the vanishing gradient problem [45].

2.3.2. Attention module To assess the varying importance of each view in the final glitch classification, we introduce an attention module ${ \mathcal { M } } _ { \mathrm { { A t t } } }$ [46] to the latent vectors of four inputs, denoted as $z _ { i } \in \mathbb { R } ^ { 1 \times M } , i = \{ 1 , 2 , 3 , 4 \}$ , extracted after the fully-connected layer. We set the dimension of the latent vectors to $M = 1 2 8$ for this model. The attention weight $\alpha _ { i } \in \mathbb { R } , i = \{ 1 , 2 , 3 , 4 \}$ calculated for each latent vector $z _ { i }$ quantifies the significance of each view in making the glitch classification, with all weights summing up to 1. The attention module is defined as

$$
\mathcal {M} _ {\mathrm{Att}} = \sum_ {i = 1} ^ {4} \alpha_ {i} z _ {i}, \tag {1}
$$

where the attention weight $\alpha _ { i }$ is computed as

$$
\alpha_ {i} = \frac {\exp \left\{w ^ {\mathrm{T}} \left[ \tanh \left(V z _ {i} ^ {\mathrm{T}}\right) \odot \sigma \left(U z _ {i} ^ {\mathrm{T}}\right) \right] \right\}}{\sum_ {j = 1} ^ {4} \exp \left\{w ^ {\mathrm{T}} \left[ \tanh \left(V z _ {j} ^ {\mathrm{T}}\right) \odot \sigma \left(U z _ {j} ^ {\mathrm{T}}\right) \right] \right\}}. \tag {2}
$$

The trainable parameters $V \in \mathbb { R } ^ { L \times M }$ and $w \in \mathbb { R } ^ { L \times 1 }$ are associated with the attention mechanism. Here, L signifies the one dimension of the matrices for transforming the latent vectors (set to $L = 6 4 )$ . The V matrix is instrumental in identifying similarities among different views. The non-linear hyperbolic tangent function tanh(·) includes both positive and negative values in the gradient flow, but it is almost linear in the [−1, 1] range. To handle complex data patterns effectively, we leverage the gated attention mechanism [46], which combines tanh(·) with the element-wise sigmoid function $\sigma ( \cdot )$ . The sigmoid function helps adjust the relevance of information, while the elementwise multiplication $\odot$ facilitates the attention process. The gated attention mechanism introduces additional parameters $U \in \mathbb { R } ^ { L \times M }$ to serve as control gates, mitigating the linearity issue present in tanh(·). The attention module, widely used in multiple-instance learning tasks [46–49], excels in identifying key instances within a bag, i.e., a collection of instances.

In this paper, we evaluate the impact of the attention module on the model’s performance by conducting comparative experiments both with and without the incorporation of attention. For the experiments without the attention module, we employed two approaches: averaging the latent vector (i.e., mean pooling) and maximizing the latent vector (i.e., max pooling).

2.3.3. Label smoothing As mentioned in Section 2.1, the ground-truth labels for the Gravity Spy dataset are generated by the previous classifier and volunteers. Despite filtering the dataset to only use high-confidence predictions, mislabeled glitches can still be present, potentially confusing the model. To address this problem, label smoothing [50], a valuable regularization technique in machine learning, is applied during model training to enhance generalization and prevent overconfidence in predictions, especially when dealing with noisy or uncertain data. This technique introduces a small amount of uncertainty into the training labels. Instead of assigning a hard 1 to the correct class and 0 to other classes, label smoothing assigns a smoothed value, denoted by q, slightly less than 1 to the correct class while distributing the remaining confidence uniformly across the other classes. Mathematically, it can be defined as:

$$
q = (1 - \beta) y + \beta u, \tag {3}
$$

where y represents the true label (encoded as a one-hot vector), u is a uniform distribution over the classes, and $\beta$ is a smoothing parameter between 0 and 1.

In addition, to address the issue of class imbalance (as shown in Table 1), we apply different weights $\gamma _ { i }$ to each class $i = \{ 1 , 2 , \ldots , C \}$ in the loss function during training. These weights are inversely proportional to the number of instances in each glitch class and sum up to 1, ensuring a balanced learning process. Therefore, the loss function of the proposed model is defined as

$$
\mathcal {L} (y, \hat {y}) = - \frac {1}{C} \sum_ {i = 1} ^ {C} \gamma_ {i} q _ {i} \log (\hat {y} _ {i}), \tag {4}
$$

where $\hat { y }$ represents the predicted probability distribution over classes and $C$ is the number of glitch classes. By adjusting the smoothing parameter, the level of smoothing applied to the labels can be controlled. For example, setting $\beta = 0$ (i.e, no smoothing) leads to one-hot encodings and restores the standard categorical cross-entropy loss, while $\beta = 1$ (i.e., completely smooth) results in a uniform distribution over all classes.

In this paper, the impact of different smoothing levels on the model’s performance is also evaluated through different experiments on the value of $\beta .$ .

2.3.4. Architecture of the classifier The overall architecture of the proposed classifier is shown in Figure 4. Each of the four branches shares a common architecture. Within each branch, a pre-convolutional block is followed by three instances of a combination of a convolutional block and an inception residual block (Incep-Res Block) [43], along with a post-convolutional block. These stages effectively extract features from each glitch view. Specifically, all convolutional blocks (i.e., Pre-Conv, Conv and Post-Conv blocks) consist of a convolutional layer, a ReLu activation function, and batch normalization (BN) layer and a max pooling (Max Pool) layer. Subsequently, a flatten layer and a fully-connected (FC) layer reduce the dimensionality of the feature vectors. The attention module [46] then takes these vectors as input, generating a single weighted-averaged feature vector, which is then fed into the final fully-connected layer. The softmax activation function yields the probabilities of the 23 glitch classes. To enhance the model’s generalization and prevent overfitting, we incorporate an L2 regularizer [51] for each convolutional layer and introduce dropout [52] after each fully-connected layer.

For optimization, we use the Adam optimizer [53], with the learning rate starting at $2 \times 1 0 ^ { - 5 }$ . This choice of optimizer helps to efficiently update the model’s parameters during training. To address the potential issue of confidence overfitting, where the model may become overconfident in its predictions on the training data, we employ early stopping with a patience of 10, i.e., if the model does not improve its accuracy on the validation set within 10 training epochs, we stop training to prevent potential overconfidence.

# 2.4. Statistical analysis

To ensure the robustness of the model, five independent runs for all experiments are conducted. Each of the five runs has different random initializations of the network weights and uses different random mini-batch orderings during training, while keeping the same architecture and hyperparameters. For each experiment, we calculate the overall accuracy, precision, F1 score (harmonic mean of the precision and recall), and the area under the receiver operating characteristic curve (AUC) on the held-out test set, where the mean and standard deviation values for each metric are reported. To assess the significance of the differences between different model results, we perform a paired t-test [54] by calculating the differences in scores for each pair, followed by computing the mean and standard deviation of these differences, and determining significance based on a fiducial p-value $\left( p < 0 . 0 5 \right)$ . Furthermore, we employ a reliability plot [55] to evaluate model calibration across different confidence levels. This plot compares predicted probabilities with observed outcomes by binning the probabilities and calculating the average predicted values and observed frequencies for each bin. By juxtaposing these elements, the plot reveals the model’s accuracy in representing uncertainty. In other words, for a well-calibrated model, if it predicts a set of glitches with a 40% probability of being a Blip, the actual frequency of Blips in that set should also be approximately 40%. Additionally, calibration accuracy is quantified through the expected calibration error (ECE) [34], a statistical metric that measures the average deviation between predicted probabilities and the true likelihood of event occurrence.

# 3. Results

In this section, we present an overview of our results. We begin by outlining the design of all experiments conducted, providing a comprehensive understanding of our methods. Next, we compare the classifier’s results under various fusion strategies, evaluate the impact of different smoothing levels, analyze the effects of the gated attention mechanism, and, last, perform a comparative assessment between the proposed model and the previous model implemented in the Gravity Spy pipeline.

# 3.1. Experimental design

In this study, we conduct various ablation studies to assess the model’s performance. First, the effectiveness of different fusion strategies is compared, as shown in Figure 3. Second, we evaluate different values of the smoothing parameter $\beta$ between 0 and 1 with an increment of 0.1, allowing us to identify an optimal value for $\beta .$ Third, we compare the proposed classifier with the attention module and a variant without the attention module (i.e., max pooling and mean pooling) to demonstrate the effectiveness of the attention mechanism. Fourth, we evaluate the performance of our proposed classifier by comparing it with the previous Gravity Spy model architecture [23–25]. Both models are trained from scratch using the same training dataset, and their performance is assessed on the same test set. Training and testing on the same dataset ensures a fair comparison; these have varied in previous studies (along with the list of glitch classes), so it would not be fair to directly compare statistics to those from other works. Finally, to better understand the learned feature representations, we employ t-distributed stochastic neighbor embedding (t-SNE) plots [56] to reduce the dimensionality of feature vectors obtained after the attention module from 128 to 2, offering visualizations for model comparisons in a two-dimensional space. Here, for each experiment, we maintain all other experimental conditions at their optimal settings to ensure a fair comparison.

The experiments and analyses are conducted using TensorFlow 2.11 on two GPUs (NVIDIA Quadro RTX 8000) and scikit-learn, respectively, in Python 3.8. The codes are available via the GitHub of Gravity Spy.

# 3.2. Comparison of three fusion strategies

Table 2 demonstrates the model performance of three fusion strategies on the same test set. The proposed intermediate fusion model achieves the best result with an overall accuracy of 0.941, precision of 0.950, F1 of 0.931 and AUC of 0.965, followed by the late fusion and the early fusion, although the difference between the latter two is not statistically significant. Moreover, the results exhibit robustness, as indicated by the minimal standard deviations derived from five independent experiment runs.

Table 2. Model evaluations (the overall mean ± standard deviation across five independent runs) for three fusion strategies (early fusion, late fusion and intermediate fusion) on the O3 test set. Best results are highlighted in bold, and an asterisk (\*) indicates that the comparison is statistically significant given our chosen threshold $\left( p < 0 . 0 5 \right)$ . 

<table><tr><td rowspan="2">Performance metric</td><td colspan="3">Fusion strategy</td></tr><tr><td>Early fusion(Figure 3 (a))</td><td>Late fusion(Figure 3 (b))</td><td>Intermediate fusion(proposed; Figure 4)</td></tr><tr><td>Accuracy</td><td> $0.911 \pm 0.013$ </td><td> $0.922 \pm 0.022$ </td><td> $\mathbf{0.941} \pm \mathbf{0.021}^{*}$ </td></tr><tr><td>Precision</td><td> $0.918 \pm 0.008$ </td><td> $0.920 \pm 0.011$ </td><td> $\mathbf{0.950} \pm \mathbf{0.010}^{*}$ </td></tr><tr><td>F1</td><td> $0.901 \pm 0.009$ </td><td> $0.901 \pm 0.014$ </td><td> $\mathbf{0.931} \pm \mathbf{0.015}^{*}$ </td></tr><tr><td>AUC</td><td> $0.939 \pm 0.003$ </td><td> $0.940 \pm 0.003$ </td><td> $\mathbf{0.965} \pm \mathbf{0.004}^{*}$ </td></tr></table>

# 3.3. Effects of smoothing labels

Figure 5 presents a comparative analysis of model performance across various smoothing levels of $\beta$ in Eq. (3). At the outset, for $\beta = 0$ , no label smoothing yields a model accuracy of 0.921, precision of 0.915, F1 of 0.908 and AUC of 0.946. With an increase in $\beta _ { ; }$ , the model’s performance exhibits an initial ascent followed by a subsequent decline. When $\beta = 0 . 3$ , the model achieves the optimal performance across all metrics. Additionally, the results indicate that, with the implementation of label smoothing, for $\beta \leqslant 0 . 5$ , the model consistently outperforms the one without label smoothing $( \beta = 0 )$ .

![](images/46dad4f5a32057d6bd3c1085912ef8b5458e5b73bf025895e42a739ad4895a88.jpg)

<details>
<summary>line</summary>

| Smoothing Level | Accuracy | Precision | F1   | AUC  |
| --------------- | -------- | --------- | ---- | ---- |
| 0.0             | 92.1     | 91.5      | 90.7 | 94.6 |
| 0.1             | 92.3     | 93.4      | 91.4 | 94.8 |
| 0.2             | 93.0     | 93.9      | 93.0 | 96.0 |
| 0.3             | 94.1     | 95.0      | 93.1 | 96.5 |
| 0.4             | 92.7     | 94.1      | 92.4 | 96.0 |
| 0.5             | 91.4     | 92.4      | 91.1 | 94.6 |
| 0.6             | 90.6     | 91.3      | 90.9 | 94.7 |
| 0.7             | 90.6     | 90.6      | 89.9 | 93.7 |
| 0.8             | 88.9     | 90.1      | 89.2 | 90.6 |
| 0.9             | 88.3     | 88.6      | 88.9 | 88.9 |
</details>

Figure 5. The evaluation of model performance (accuracy, precision, F1, and AUC scores) on the O3 test set presented across varying smoothing levels (β) ranging from 0 to 1 in increments of 0.1. The results represent the average of five experimental runs, with standard deviations indicated by error bars. Here, $\beta = 0$ represents no smoothing.

# 3.4. Effects of gated attention mechanism

Table 3 compares the performance of models with and without the gated attention mechanism. These models share an identical convolutional neural network architecture for feature extraction from each view but diverge in their feature fusion approaches, such as mean pooling and max pooling. The results demonstrate that the performance of the model with the gated attention mechanism is significantly better than of those without it. Additionally, statistical analysis indicates no significant difference between the performances of mean pooling and max pooling.

The gated attention mechanism also enhances interpretability, rendering the model’s outputs more explainable to humans. Figure 6 shows examples with attention weights assigned by the proposed classifier for four glitch predictions. The magnitude of attention weights positively correlates with the contribution of each time view to the prediction. For instance, in Figure 6 (a), (d) and (e), the model places greater emphasis on long time windows to predict glitches like Repeating Blips and No Glitch, while allocating higher attention to short time windows to predict glitches such as Blip and Koi Fish in Figure 6 (b) and (c).

Table 3. Model evaluations (the overall mean ± standard deviation across five independent runs) for classifiers with gated attention mechanism and without attention mechanism (using max pooling and mean pooling) on the O3 test set using the intermediate fusion structure. Best results are highlighted in bold, and an asterisk (\*) indicates that the comparison is statistically significant given our chosen threshold $\left( p < 0 . 0 5 \right)$ . 

<table><tr><td rowspan="2">Performance metric</td><td colspan="3">Attention mechanism</td></tr><tr><td>Proposed(gated attention)</td><td>No Attention(max pooling)</td><td>No Attention(mean pooling)</td></tr><tr><td>Accuracy</td><td> $0.941 \pm 0.021^{*}$ </td><td> $0.916 \pm 0.013$ </td><td> $0.911 \pm 0.011$ </td></tr><tr><td>Precision</td><td> $0.950 \pm 0.010^{*}$ </td><td> $0.928 \pm 0.010$ </td><td> $0.929 \pm 0.006$ </td></tr><tr><td>F1</td><td> $0.931 \pm 0.015^{*}$ </td><td> $0.908 \pm 0.011$ </td><td> $0.903 \pm 0.008$ </td></tr><tr><td>AUC</td><td> $0.965 \pm 0.004^{*}$ </td><td> $0.948 \pm 0.002$ </td><td> $0.949 \pm 0.001$ </td></tr></table>

# 3.5. Comparison with previous classifiers

We compare the performance of the proposed model with the previously implemented classifier [24]. The previous classifier has been retrained using the same dataset as our new model to enable a clean comparison. Results on our O3 test set are shown in Table 4. The proposed model shows superiority across all metrics, outperforming the previous model by 2.7% to 3.7%. Furthermore, a detailed examination of classwise accuracies reveals that the proposed model provides more accurate predictions for the majority of glitch classes. In particular, the model performs better for classes like Wandering Line, Repeating Blips, and Helix. While there are a few instances for which the previous classifier attains better outcomes, such as Koi Fish and Scattered Light (Slow Scattering), the differences remain relatively modest.

To obtain a deeper insight into this finding, we compare the t-SNE plots of these two classifiers within a two-dimensional space, as shown in Figure 7. Compared to the previous classifier, it is evident that the feature distributions generated by the new classifier exhibit tighter clusters within the same glitch class and distinct separation among clusters of different classes. However, some degree of overlap persists between glitch classes with similar morphologies, such as Extremely Loud and Koi Fish, Blip and Repeating Blips, Scattered Light and Low-frequency Lines.

![](images/ace9a240e5637998c715410c6df55789bdf7c12e9a6ccd53374ff5447dbfe41f.jpg)  
Figure 6. Illustrative examples of glitch classification with attention weights generated by the classifier for each time-window view. Attention weights are shown in white in each subplot. A higher attention weight indicates that the model relies more on that particular view for predicting the glitch class.

Figure 8 presents the reliability plots of two classifiers applied to the test set. Both classifiers are overconfident in their predictions, evidenced by the majority of points residing below the diagonal line. Nevertheless, the proposed model demonstrates better calibration compared to the previous model, particularly in the high confidence range (> 0.8)—a crucial aspect for detector characterization and gravitational-wave data analysis. Additionally, the proposed model achieves a lower expected calibration error of 0.06, compared to the previous model’s 0.09. This discrepancy indicates that the proposed model displays less bias and higher reliability than the previous classifier.

![](images/38658a3bf5afd5786948cb9670048436924ea9bc6f022c05c4448a0c6d5b3a37.jpg)  
Figure 7. Comparison of t-SNE plots in a two-dimensional space (latent features t-SNE 1 and t-SNE 2) between the previous classifier and the newly proposed classifier on the same O3 test set). The proposed classifier (right) generates more tightly-clustered glitch features than the previous one (left) within the latent space. Both plots were generated using the same t-SNE parameters for consistency.

![](images/a53d194fa19ebfdb0644d35f048ec6747b512c68aaeb658dd5be4b69e30a0a25.jpg)  
Figure 8. The reliability plots of the model output probabilities on the O3 test set. The plots illustrate the observed fraction of positives against the predicted fraction of positives. The diagonal dotted line indicates perfect reliability. The expected calibration error (ECE) quantifies the average discrepancy between predicted probabilities and actual accuracy.

Table 4. Comparison of class-wise classification accuracy, and the overall performance (accuracy, precision, F1 and AUC scores) between the previous classifier and the proposed classifier evaluated with our O3 dataset. Better results are highlighted in bold, and an asterisk (\*) indicates that the comparison is statistically significant given our chosen threshold $\left( p < 0 . 0 5 \right)$ . 

<table><tr><td rowspan="2" colspan="2">Performance metric</td><td colspan="2">Classifier</td></tr><tr><td>Previous</td><td>Proposed</td></tr><tr><td rowspan="23">Class accuracy</td><td>1080 Lines</td><td>0.985</td><td>0.991</td></tr><tr><td>1400 Ripples</td><td>0.967</td><td>0.979*</td></tr><tr><td>Air Compressor</td><td>0.921</td><td>0.954*</td></tr><tr><td>Blip</td><td>0.980</td><td>0.991*</td></tr><tr><td>Blip Low Frequency</td><td>0.926</td><td>0.928</td></tr><tr><td>Chirp</td><td>0.913</td><td>0.926*</td></tr><tr><td>Extremely Loud</td><td>0.901</td><td>0.960*</td></tr><tr><td>Fast Scattering</td><td>0.884</td><td>0.890</td></tr><tr><td>Helix</td><td>0.732</td><td>0.891*</td></tr><tr><td>Koi Fish</td><td>0.921</td><td>0.917</td></tr><tr><td>Light Modulation</td><td>0.918</td><td>0.947*</td></tr><tr><td>Low-frequency Burst</td><td>0.962</td><td>0.981*</td></tr><tr><td>Low-frequency Lines</td><td>0.818</td><td>0.916*</td></tr><tr><td>No Glitch</td><td>0.978</td><td>0.984</td></tr><tr><td>Paired Doves</td><td>0.849</td><td>0.882*</td></tr><tr><td>Power Line</td><td>0.987</td><td>0.994*</td></tr><tr><td>Repeating Blips</td><td>0.772</td><td>0.933*</td></tr><tr><td>Scattered Light</td><td>0.929</td><td>0.924</td></tr><tr><td>Scratchy</td><td>0.862</td><td>0.957*</td></tr><tr><td>Tomte</td><td>0.963</td><td>0.984*</td></tr><tr><td>Violin Mode</td><td>0.924</td><td>0.960*</td></tr><tr><td>Wandering Line</td><td>0.520</td><td>0.740*</td></tr><tr><td>Whistle</td><td>0.957</td><td>0.972*</td></tr><tr><td rowspan="4">Overall</td><td>Accuracy</td><td>0.906</td><td>0.941*</td></tr><tr><td>Precision</td><td>0.913</td><td>0.950*</td></tr><tr><td>F1</td><td>0.904</td><td>0.931*</td></tr><tr><td>AUC</td><td>0.937</td><td>0.965*</td></tr></table>

# 4. Discussion

In this study, we introduce a novel classifier for the Gravity Spy project, aiming to enhance the accuracy of glitch classification provided for LIGO. Overall, our proposed classifier achieves accuracy of 0.941, precision of 0.950, F1 score of 0.931 and AUC of 0.965 on the held-out set of glitches, outperforming prior classifiers integrated into the previous Gravity Spy workflow.

The proposed classifier exhibits superior glitch classification through various attributes. First, the use of the inception residual block empowers the model to deepen its architecture without overfitting concerns, thereby facilitating the extraction of more discriminative features unique to different glitch classes. The tightly clustered distributions in Figure 7 further substantiate this finding. Next, the intermediate fusion strategy outperforms early and late fusion due to its ability to capture and combine cross-view correlations, which aligns with findings from prior studies, such as using multiple time scales for anomaly detection [57] and multiple modalities for gesture recognition [58]. Moreover, label smoothing effectively tackles the issue of noisy labels within the training dataset by introducing a controlled level of uncertainty into label distributions during training, resulting in enhanced glitch predictions. However, it is crucial to choose the level of this uncertainty infusion appropriately, as excessive smoothing, illustrated in Figure 5, can lead to label confusion within the model, hindering its ability to learn effectively.

The challenge of the black box issue in deep learning stems from the inherent lack of transparency in comprehending how these models arrive at decisions. The attention mechanism addresses this challenge by enabling the model to selectively focus on specific features of the input data, assigning varying levels of importance to different regions. By highlighting informative features and providing interpretable decisions, the attention mechanism enhances transparency in the decision-making process.

In our study, the model achieves interpretability by discerning which view contributes the most to the classification decision. Consequently, the attention mechanism significantly improves the overall model performance by selectively prioritizing informative aspects of each view and effectively capturing cross-view relationships. For example, as shown in Figure 6, the model’s extraction of more discriminative features from the 2 s and 4 s windows when classifying a glitch as Repeating Blips aligns with volunteers’ experiences, reflecting their awareness that repeating patterns primarily manifest in longer time windows. In contrast, when identifying a single Blip class, the model focuses on extracting more specific features from the 0.5 s window, corresponding to the transient nature of this particular glitch category. Similarly, when dealing with cases resembling Blip or Koi Fish, the model directs greater attention to the 0.5 s window. Concerning No Glitch classification, the model’s emphasis on the 4 s window is intuitive, aligning with the necessity to inspect the longest window time to confirm all background features and the absence of glitches. For Whistle, some attention is given to the 0.5 s window, where the detail of the characteristic up-anddown sweep is most visible, but more is given to the 2 s window. While the illustrative Whistle is a single V-shaped sweep, Whistles can often contain multiple sweeps (a W shape as shown in Figure 1 is common), and these are more apparent in the longer windows: the 2 s window seems to give the best balance between resolving the upand-down structure and the potential repetition of sweeps. The emphasis on specific windows does not diminish the importance of those with low attention weights. Instead, the model effectively captures crucial information from all four windows, enabling it to make optimal glitch predictions. Potentially, the attention mechanism can guide volunteers and LIGO experts, enabling them to enhance their understanding of glitches and thereby facilitating a more effective characterization process.

![](images/c92e586f4a64140764fb9589c450be396053bd8d748d0e3ce56239a7c5c7877a.jpg)  
Figure 9. Examples of glitches with different predicted classes between the previous model and the proposed model. (a) Initially predicted as the Koi Fish class by the previous model, it is now classified as Extremely Loud by the proposed model. (b) Initially classified as the Scattered Light class by the previous model, it is now predicted as Low-frequency Lines by the proposed model.

Given its superior overall performance in comparison to the previous classifier, the proposed classifier has been integrated into the Gravity Spy system for the O4 run. This integration coincides with major infrastructure enhancements introduced in the O4 run [26–28, 59], aimed at achieving higher instrument sensitivity for the detection of gravitational-wave signals. However, this improvement in detector sensitivity is accompanied by a trade-off, i.e., a higher sensitivity to environmental and instrumental artifacts, which potentially leads to new glitch classes.

In this context, the more discriminative features extracted by the proposed classifier, as shown in Figure 7, have the potential to more effectively identify and categorize these new glitch classes. Additionally, Figure 7 also shows distinct clusters within a single class, which suggests the possible existence of sub-classes that may be interesting to explore

An underperformance is observed in certain glitch classes, such as Koi Fish and Scattered Light as presented in Table 4. However, this underperformance is not significant $( p > 0 . 0 5 )$ , approximately 0.5% compared to the previous model, which may potentially be attributed to random sampling variations in our test set. Besides, this underperformance is also influenced by the inherent complexity and variability within these classes. Given their prevalence and diverse morphologies, establishing strict boundaries for these classes poses a challenge, leading to expected variations in classifiers when distinguishing these intricacies. For instance, in Figure 9, Koi Fish can be confused with Extremely Loud when the trigger is very loud [24], and Scattered Light can be confused with Low-frequency Lines as they share characteristics of being low frequency and long duration [60].

![](images/a4372927ce2b035f3bfbe356097b2908d4ca9cbe791c2ae9b4a001de08410633.jpg)

<details>
<summary>violin</summary>

| Category             | Value |
| -------------------- | ----- |
| Air_Compressor       | 0.9   |
| Blip                 | 0.85  |
| Blip_Low_Frequency   | 0.8   |
| Extremely_Loud       | 0.75  |
| Fast_Scattering      | 0.85  |
| Koi_Fish             | 0.8   |
| Light_Modulation     | 0.75  |
| Low_Frequency_Burst  | 0.7   |
| Low_Frequency_Lines  | 0.65  |
| No_Glitch            | 0.6   |
| Paired_Doves         | 0.55  |
| Power_Line          | 0.5   |
| Repeating_Blips      | 0.45  |
| Scattered_Light      | 0.4   |
| Scratchy            | 0.35  |
| Tomte                | 0.3   |
| Violin_Mode          | 0.25  |
| Whistle              | 0.2   |
</details>

(a) Confidence Distribution of the Previous Classifier

![](images/31d89806630dadddca0b68f86491648eee5efde7767bca95dae53da0e9fe79e3.jpg)

<details>
<summary>violin</summary>

| Category | Min    | Q1     | Median | Q3     | Max    |
|----------|--------|--------|--------|--------|--------|
| Group 1  | 0.2    | 0.3    | 0.4    | 0.5    | 0.6    |
| Group 2  | 0.3    | 0.4    | 0.5    | 0.6    | 0.7    |
| Group 3  | 0.4    | 0.5    | 0.6    | 0.7    | 0.8    |
| Group 4  | 0.5    | 0.6    | 0.7    | 0.8    | 0.9    |
| Group 5  | 0.6    | 0.7    | 0.8    | 0.9    | 1.0    |
| Group 6  | 0.7    | 0.8    | 0.9    | 1.0    | 1.1    |
| Group 7  | 0.8    | 0.9    | 1.0    | 1.1    | 1.2    |
| Group 8  | 0.9    | 1.0    | 1.1    | 1.2    | 1.3    |
| Group 9  | 1.0    | 1.1    | 1.2    | 1.3    | 1.4    |
| Group 10 | 1.1    | 1.2    | 1.3    | 1.4    | 1.5    |
| Group 11 | 1.2    | 1.3    | 1.4    | 1.5    | 1.6    |
| Group 12 | 1.3    | 1.4    | 1.5    | 1.6    | 1.7    |
| Group 13 | 1.4    | 1.5    | 1.6    | 1.7    | 1.8    |
| Group 14 | 1.5    | 1.6    | 1.7    | 1.8    | 1.9    |
| Group 15 | 1.6    | 1.7    | 1.8    | 1.9    | 2.0    |
| Group 16 | 1.7    | 1.8    | 1.9    | 2.0    | 2.1    |
| Group 17 | 1.8    | 1.9    | 2.0    | 2.1    | 2.2    |
| Group 18 | 1.9    | 2.0    | 2.1    | 2.2    | 2.3    |
| Group 19 | 2.0    | 2.1    | 2.2    | 2.3    | 2.4    |
| Group 20 | 2.1    | 2.2    | 2.3    | 2.4    | 2.5    |
| Group 21 | 2.2    | 2.3    | 2.4    | 2.5    | 2.6    |
| Group 22 | 2.3    | 2.4    | 2.5    | 2.6    | 2.7    |
| Group 23 | 2.4    | 2.5    | 2.6    | 2.7    | 2.8    |
| Group 24 | 2.5    | 2.6    | 2.7    | 2.8    | 2.9    |
| Group 25 | 2.6    | 2.7    | 2.8    | 2.9    | 3.0    |
| Group 26 | 2.7    | 2.8    | 2.9    | 3.0    | 3.1    |
| Group 27 | 2.8    | 2.9    | 3.0    | 3.1    | 3.2    |
| Group 28 | 2.9    | 3.0    | 3.1    | 3.2    | 3.3    |
| Group 29 | 3.0    | 3.1    | 3.2    | 3.3    | 3.4    |
| Group 30 | 3.1    | 3.2    | 3.3    | 3.4    | 3.5    |
| Group 31 | 3.2    | 3.3    | 3.4    | 3.5    | 3.6    |
| Group 32 | nan    | nan    | nan    | nan    | nan     |
| Group Note: The values in the 'Value' column are estimated based on the provided code (e.g., 'Group' or 'Group' should be 'N'). The actual values may vary due to the random nature of the data generation.
</details>

(b) Confidence Distribution of the Proposed Classifier   
Figure 10. The density distribution plot of predicted confidence for early O4 glitches between (a) the previous model and (b) the proposed model. The wider sections indicating higher densities. The black horizontal bars within each section represent the median confidence score for that glitch class.

The previous Gravity Spy classifier has long grappled with the challenge of confidence overfitting. This issue can be attributed to various factors, including imbalances in our glitch classes, the presence of noisy labels in our training set, and the use of cross-entropy loss for updating the model parameters. These challenges pose the risk of hindering the model’s ability to generalize effectively to new and unseen glitch data. To address this issue, our proposed classifier incorporates several techniques. We apply L2 regularization and incorporate smooth labeling into the cross-entropy loss to penalize extreme confidence. Moreover, we monitor accuracy for updating model parameters, extending beyond the sole reliance on cross-entropy loss, which tends to drive the predicted confidence close to 1.

To demonstrate the impact of confidence overfitting in O4, we conduct a comparative analysis of the predicted confidence distributions between the previous model and the proposed model. This analysis encompasses a total of 2267 glitches extracted from early in O4, from 24 May 2023 to 19 September 19 2023 in Hanford and Livingston data. As illustrated in Figure 10 (a), the previous model exhibits persistent confidence overfitting, as most glitch classes receive predictions with confidence close to 1, except for Paired Doves and Light Modulation. In contrast, the proposed model, as shown in Figure 10 (b), significantly mitigates this issue, providing predictions with varying confidence ranges across glitch classes. Specifically, for glitch classes like Extremely Loud and Low-frequency Lines, which exhibit less variation in morphologies within the O4 distribution compared to O3, the proposed model confidently expresses predictions with relatively high confidence. Conversely, for glitch classes such as Whistle, Wandering Line, Violin Mode, and Scratchy, the proposed model displays lower confidence in predictions, given the visually distinct morphologies in O4 compared to their distributions in O3. This distinction is crucial for LIGO experts during glitch analysis and characterization, allowing the exclusion of glitches with lower confidences to enhance the precision of their analyses. This, in turn, opens avenues to explore the potential discovery of new glitch classes within clusters exhibiting lower confidences. A good example is the identification of a new glitch class, named 589 Hz,§ depicted in Figure 11, which occurred on May 30, 31, and June 1 of 2023 in the Hanford detector. The previous model predicted it as Whistle with an overall high confidence of 0.976(±0.059) on 267 collections, attributing this classification to its high-frequency nature similar to Whistle, while the proposed model predicted it with significantly lower confidence scores of 0.463(±0.138) for 195 cases classified as Whistle, 0.428(±0.137) for 49 cases classified as No Glitch and 0.347(±0.057) for 23 cases classified as Violin Mode, respectively.

![](images/90d78b04679198dbb426259b4d913753374db722d48afd7852330da89a85d123.jpg)  
Figure 11. An example of a new O4 glitch class, named 589 Hz. This class appears as a narrow-band line glitch near 589 Hz lasting less than 0.4 s, appearing in these spectrograms as bright yellow for 0.02 s to 0.2 s. This example was classified as the Whistle class with a high confidence of 0.972 by the previous classifier and a low confidence of 0.348 by the proposed classifier.

This study has some limitations. First, as the classifier is trained under full supervision, its generalization capability to new classes is constrained. Consequently, if a novel glitch class arises during the O4 run, the classifier requires retraining on a dataset that incorporates the new glitch class. This process also involves the participation of volunteers who, with assistance from tools such as the the Gravity Spy similarity search [19], identify and categorize new glitch classes [24]. Next, the training and test set used in this study are still imperfect with noisy labels. To improve the reliability of the model, future efforts should be put into preparing a test set with validated ground truth, as confirmed by LIGO detector-characterization experts. Additionally, there is room for further optimization of the classifier for improved glitch classification. Future studies can enhance the model’s training by incorporating additional information such as sensor locations, temporal attributes, and supplementary auxiliary data [18, 21, 61, 62]. For instance, supplementary auxiliary channels, could assist in reducing potential Scattered Light, as illustrated in Figure 9 (b), by aiding in the identification of ground motion [63]. Last, our model currently provides information on important views but does not specify which parts of the focused image trigger predictions. Our research will explore techniques like Grad-CAM [64] to precisely identify the influential regions within glitch images.

# 5. Conclusion

In this study, we present a novel classifier for the Gravity Spy project specifically designed to enhance the accuracy of glitch classification for the LIGO gravitationalwave detectors. Several factors contribute to the classifier’s distinctiveness. Leveraging the inception residual block allows for deeper architecture exploration, facilitating the extraction of discriminative glitch features. The intermediate fusion strategy captures cross-view correlations more effectively. Label smoothing addresses noisy training labels, enhancing the generalization of glitch predictions. The attention mechanism not only enhances model performance but also provides interpretability, potentially aiding volunteers and experts in understanding glitch classification decisions. Our tests demonstrate that the proposed classifier has exceptional performance compared to the previous classifiers.

The new classifier has successfully been integrated into analysis of the ongoing O4 run, and is actively employed by both volunteers on the Zooniverse platform and LIGO experts. Its capacity to potentially help volunteers identify and categorize new glitch classes aligns with the Gravity Spy project’s larger aim of improving detector sensitivity. A more advanced model is essential for understanding the details of more intricate glitch patterns. Following investigations of new glitch types [21, 24], the new classifier will be retrained with expanded training data for use in future detector-characterization studies.

# Acknowledgments

The authors extend their gratitude to the community-science volunteers of the Gravity Spy project, whose dedicated efforts played a pivotal role in bringing this project to fruition. The authors also thank ManLeong Chan and the anonymous referees for comments that improved the quality of this manuscript, and Shamal Lalvani, Yi Li and Jonathan Stromer-Galley for insightful conversations. This material is based upon work supported by NSF’s LIGO Laboratory which is a major facility fully funded by the National Science Foundation. The authors are grateful for computational resources provided by the LIGO Laboratory and supported by National Science Foundation Grants PHY-0757058 and PHY-0823459. Gravity Spy is supported by NSF grants IIS-

2106882, 2106896, 2107334, and 2106865. CPLB acknowledges support from Science and Technology Facilities Council (STFC) grant ST/V005634/1. ZD acknowledges support from the CIERA Board of Visitors Research Professorship and NSF grant PHY-2207945. This document has been assigned LIGO document number LIGO-P2300458.

# Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Data Availability Statement

The datasets from the Gravity Spy project, including machine learning and volunteer classification information, can be accessed through the Zenodo repositories [40, 65]. The corresponding codes are also accessible on GitHub through Gravity Spy.

# References

[1] Einstein A 1916 Sitzungsber. Preuss. Akad. Wiss. Berlin (Math. Phys.) 1916 688– 696   
[2] Aasi J et al. (LIGO Scientific) 2015 Classical and Quantum Gravity 32 074001 (Preprint arXiv:1411.4547)   
[3] Abbott B P et al. (LIGO Scientific and Virgo Collaboration) 2016 Physical Review Letters 116 061102 (Preprint arXiv:1602.03837)   
[4] Acernese F et al. (Virgo Collaboration) 2015 Classical and Quantum Gravity 32 024001 (Preprint arXiv:1408.3978)   
[5] Abbott B P, et al. (LIGO Scientific and Virgo Collaboration) 2017 Physical Review Letters 119(14) 141101 (Preprint arXiv:1709.09660)   
[6] Abbott B P et al. (LIGO Scientific and Virgo Collaboration) 2019 Physical Review Letters 9 031040 (Preprint arXiv:1811.12907)   
[7] Akutsu T et al. (KAGRA Collaboration) 2019 Nature Astronomy 3 35–40 (Preprint arXiv:1811.08079)   
[8] Abbott R et al. (LIGO Scientific, Virgo and KAGRA Collaboration) 2022 Progress of Theoretical and Experimental Physics 2022 063F01 (Preprint arXiv:2203.01270)   
[9] Abbott R et al. (LIGO Scientific, Virgo and KAGRA Collaboration) 2023 Physical Review X 13 041039 (Preprint arXiv:2111.03606)   
[10] Abbott B P, et al. (LIGO Scientific and Virgo Collaboration) 2020 Classical and Quantum Gravity 37 055002 (Preprint arXiv:1908.11170)   
[11] Abbott B P et al. 2016 Classical and Quantum Gravity 33 134001 (Preprint arXiv:1602.03844)

[12] Davis D, et al. 2021 Classical and Quantum Gravity 38 135014 (Preprint arXiv:2101.11673)   
[13] Berger B K, Areeda J S, Barker J D, Effler A, Goetz E, Helmling-Cornell A F, Lantz B, Lundgren A P, Macleod D M, McIver J and et al 2023 Applied Physics Letters 122 184101   
[14] Davis D and Walker M 2022 Galaxies 10 12   
[15] Nuttall L K 2018 Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 376 20170286 (Preprint arXiv:1804.07592)   
[16] Cabero M et al. 2019 Classical and Quantum Gravity 36 15 (Preprint arXiv:1901.05093)   
[17] Zevin M, Coughlin S, Bahaadini S, Besler E, Rohani N, Allen S, Cabero M, Crowston K, Katsaggelos A K, Larson S L et al. 2017 Classical and Quantum Gravity 34 064003 (Preprint arXiv:1611.04596)   
[18] Zevin M, Jackson C B, Doctor Z, Wu Y, Østerlund C, Johnson L C, Berry C P, Crowston K, Coughlin S B, Kalogera V et al. 2024 European Physical Journal Plus 139 100 (Preprint arXiv:2308.15530)   
[19] Coughlin S, Bahaadini S, Rohani N, Zevin M, Patane O, Harandi M, Jackson C, Noroozi V, Allen S, Areeda J et al. 2019 Physical Review D 99 082002 (Preprint arXiv:1903.04058)   
[20] Fortson L, Wright D, Lintott C and Trouille L 2018 arXiv preprint (Preprint arXiv:1809.09738)   
[21] Soni S et al. 2024 arXiv preprint (Preprint arXiv:2409.02831)   
[22] Dodge J, Ilharco G, Schwartz R, Farhadi A, Hajishirzi H and Smith N 2020 arXiv preprint (Preprint arXiv:2002.06305)   
[23] Bahaadini S, Noroozi V, Rohani N, Coughlin S, Zevin M, Smith J R, Kalogera V and Katsaggelos A 2018 Information Sciences 444 172–186   
[24] Soni S, Berry C P L, Coughlin S B, Harandi M, Jackson C B, Crowston K, Østerlund C, Patane O, Katsaggelos A K, Trouille L et al. 2021 Classical and Quantum Gravity 38 195016 (Preprint arXiv:2103.12104)   
[25] Glanzer J, Banagiri S, Coughlin S, Soni S, Zevin M, Berry C P L, Patane O, Bahaadini S, Rohani N, Crowston K et al. 2023 Classical and Quantum Gravity 40 065004 (Preprint arXiv:2208.12849)   
[26] Abbott B P et al. (KAGRA, LIGO Scientific and Virgo Collaboration) 2020 Living Reviews in Relativity 23 1–69 (Preprint arXiv:1304.0670)   
[27] Abac A G et al. (LIGO Scientific, Virgo and KAGRA Collaboration) 2024 Astrophysic Journal Letters 970 L34 (Preprint arXiv:2404.04248)   
[28] Capote E et al. 2024 arXiv preprint (Preprint arXiv:2411.14607)   
[29] Alvarez-Lopez S, Liyanage A, Ding J, Ng R and McIver J 2024 Classical and Quantum Gravity 41 085007 (Preprint arXiv:2304.09977)

[30] Jarov S, Thiele S, Soni S, Ding J, McIver J, Ng R, Hatoya R and Davis D 2023 arXiv preprint (Preprint arXiv:2307.15867)   
[31] Simonyan K and Zisserman A 2014 arXiv preprint (Preprint arXiv:1409.1556)   
[32] Bahaadini S, Rohani N, Coughlin S, Zevin M, Kalogera V and Katsaggelos A K 2017 Deep multi-view models for glitch classification 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (IEEE) 2931– 2935 (Preprint arXiv:1705.00034)   
[33] Huang S C, Pareek A, Seyyedi S, Banerjee I and Lungren M P 2020 NPJ Digital Medicine 3 136   
[34] Guo C, Pleiss G, Sun Y and Weinberger K Q 2017 On calibration of modern neural networks International conference on machine learning (PMLR) 1321–1330   
[35] Nguyen A, Yosinski J and Clune J 2015 Deep neural networks are easily fooled: High confidence predictions for unrecognizable images Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 427–436 (Preprint arXiv:1412.1897)   
[36] Robinet F, Arnaud N, Leroy N, Lundgren A, Macleod D and McIver J 2020 SoftwareX 12 100620 (Preprint arXiv:2007.11374)   
[37] Brown J C 1991 The Journal of the Acoustical Society of America 89 425–434   
[38] Chatterji S K 2005 The search for gravitational wave bursts in data from the second LIGO science run Ph.D. thesis Massachusetts Institute of Technology   
[39] Chatterji S, Blackburn L, Martin G and Katsavounidis E 2004 Classical and Quantum Gravity 21 S1809 (Preprint arXiv:gr-qc/0412119)   
[40] Zevin M, Coughlin S, Chase E, Allen S, Bahaadini S, Berry C, Crowston K, Harandi M, Jackson C, Kalogera V et al. 2022 Gravity Spy volunteer classifications of LIGO glitches from observing runs O1, O2, O3a, and O3b URL https: //doi.org/10.5281/zenodo.5911227   
[41] Zadeh A, Chen M, Poria S, Cambria E and Morency L P 2017 Tensor fusion network for multimodal sentiment analysis Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing ed Palmer M, Hwa R and Riedel S (Copenhagen, Denmark: Association for Computational Linguistics) 1103–1114 (Preprint arXiv:1707.07250)   
[42] Humbert-Vidan L, Patel V, King A P and Urbano T G 2024 Physics in Medicine and Biology 69 20NT01   
[43] Szegedy C, Ioffe S, Vanhoucke V and Alemi A 2017 Inception-v4, inceptionresnet and the impact of residual connections on learning Proceedings of the AAAI Conference on Artificial Intelligence vol 31 (Preprint arXiv:1602.07261)   
[44] Szegedy C, Liu W, Jia Y, Sermanet P, Reed S, Anguelov D, Erhan D, Vanhoucke V and Rabinovich A 2015 Going deeper with convolutions Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 1–9 (Preprint arXiv:1409.4842)

[45] He K, Zhang X, Ren S and Sun J 2016 Deep residual learning for image recognition Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 770–778 (Preprint arXiv:1512.03385)   
[46] Ilse M, Tomczak J and Welling M 2018 Attention-based deep multiple instance learning International Conference on Machine Learning (PMLR) 2127–2136 (Preprint arXiv:1802.04712)   
[47] Wu Y, Schmidt A, Hern´andez-S´anchez E, Molina R and Katsaggelos A K 2021 Combining attention-based multiple instance learning and Gaussian processes for CT hemorrhage detection International Conference on Medical Image Computing and Computer-Assisted Intervention (Springer) 582–591 (Preprint bioRxiv:2021.07.01.450539)   
[48] Wu Y, Castro-Mac´ıas F M, Morales-Alvarez P, Molina R and Katsaggelos A K ´ 2023 Smooth attention for deep multiple instance learning: Application to CT intracranial hemorrhage detection International Conference on Medical Image Computing and Computer-Assisted Intervention (Springer) 327–337 (Preprint arXiv:2307.09457)   
[49] L´opez-P´erez M, Schmidt A, Wu Y, Molina R and Katsaggelos A K 2022 Computer Methods And Programs In Biomedicine 219 106783   
[50] M¨uller R, Kornblith S and Hinton G E 2019 Advances in Neural Information Processing Systems 32 (Preprint arXiv:1906.02629)   
[51] Cortes C, Mohri M and Rostamizadeh A 2012 arXiv preprint (Preprint arXiv:1205.2653)   
[52] Srivastava N, Hinton G, Krizhevsky A, Sutskever I and Salakhutdinov R 2014 The Journal of Machine Learning Research 15 1929–1958   
[53] Kingma D P and Ba J 2014 arXiv preprint (Preprint arXiv:1412.6980)   
[54] Kim T K 2015 Korean Journal of Anesthesiology 68 540–546   
[55] Br¨ocker J and Smith L A 2007 Weather and Forecasting 22 651–661   
[56] Van der Maaten L and Hinton G 2008 Journal of Machine Learning Research 9 2579–2605   
[57] Wang W, Chang F and Mi H 2021 Neurocomputing 433 37–49   
[58] Roitberg A, Pollert T, Haurilet M, Martin M and Stiefelhagen R 2019 Analysis of deep fusion strategies for multi-modal gesture recognition Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops 198–206   
[59] Cahillane C and Mansell G 2022 Galaxies 10 36 (Preprint arXiv:2202.00847)   
[60] Accadia T et al. (Virgo Collaboration) 2010 Classical and Quantum Gravity 27 194011   
[61] Nguyen P, Schofield R, Effler A, Austin C, Adya V, Ball M, Banagiri S, Banowetz K, Billman C, Blair C et al. 2021 Classical and Quantum Gravity 38 145001 (Preprint arXiv:2101.09935)

[62] Colgan R E, Corley K R, Lau Y, Bartos I, Wright J N, M´arka Z and M´arka S 2020 Physical Review D 101 102003 (Preprint arXiv:1911.11831)   
[63] Soni S, Austin C, Effler A, Schofield R, Gonz´alez G, Frolov V, Driggers J, Pele A, Urban A, Valdes G et al. 2020 Classical and Quantum Gravity 38 025016 (Preprint arXiv:2007.14876)   
[64] Selvaraju R R, Cogswell M, Das A, Vedantam R, Parikh D and Batra D 2017 Grad-CAM: Visual explanations from deep networks via gradient-based localization 2017 IEEE International Conference on Computer Vision (ICCV) 618–626   
[65] Glanzer J, Banagari S, Coughlin S, Zevin M, Bahaadini S, Rohani N, Allen S, Berry C, Crowston K, Harandi M, Jackson C, Kalogera V, Katsaggelos A, Noroozi V, Osterlund C, Patane O, Smith J, Soni S and Trouille L 2021 Gravity Spy Machine Learning Classifications of LIGO Glitches from Observing Runs O1, O2, O3a, and O3b URL https://doi.org/10.5281/zenodo.5649212