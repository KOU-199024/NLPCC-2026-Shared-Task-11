# Accurate and Robust Stellar Rotation Periods catalog for 82771 Kepler stars using deep learning

Ilay Kamai 1 and Hagai B. Perets 1, 2

1Physics Department, Technion: Israel Institute of Technology, Haifa 32000, Israel 2ACRO, Open University of Israel, R’anana, Israel

# ABSTRACT

We propose a new framework to predict stellar properties from light curves. We analyze the lightcurve data from the Kepler space mission and develop a novel tool for deriving the stellar rotation periods for main-sequence stars. Using this tool, we provide rotation periods for more than 80K stars. Our model, LightPred, is a novel deep-learning model designed to extract stellar rotation periods from light curves. The model utilizes a dual-branch architecture combining Long Short-Term Memory (LSTM) and Transformer components to capture temporal and global data features. We train LightPred on self-supervised contrastive pre-training and simulated light curves generated using a realistic spot model. Our evaluation demonstrates that LightPred outperforms classical methods like the Autocorrelation Function (ACF) in terms of accuracy and average error. We apply LightPred to the Kepler dataset, generating the largest catalog to date. Using error analysis based on learned confidence and consistency metric, we were able to filter the predictions and remove stellar types with variability which is different than spot-induced variability. Our analysis shows strong correlations between error levels and stellar parameters. Additionally, we confirm tidal synchronization in eclipsing binaries with orbital periods shorter than 10 days. Our findings highlight the potential of deep learning in extracting fundamental stellar properties from light curves, opening new avenues for understanding stellar evolution and population demographics.

Keywords: astronomy — Machine learning — light curve —star spots

# 1. INTRODUCTION

The study of stellar rotation periods has become increasingly important in our understanding of stellar evolution, activity cycles, and exoplanet detection. With the advent of space-based photometry missions like Kepler, CoRoT, and TESS, we now have access to highprecision, long-duration light curves for hundreds of thousands of stars. These light curves often exhibit periodic variations due to starspots rotating in and out of view, allowing us to measure stellar rotation periods with unprecedented accuracy.

Classical methods for measuring rotation periods, such as Lomb-Scargle periodograms and autocorrelation functions, have been successful (McQuillan et al. (2013), Reinhold et al. (2013), Walkowicz et al. (2013) Nielsen et al. (2013), McQuillan et al. (2014), Garc´ıa et al. (2014), Santos et al. (2019), Santos et al. (2021), Reinhold et al. (2023)) but can be time-consuming and challenging to apply to large datasets. Furthermore, these methods often struggle with complex light curves that exhibit multiple periodicities or evolving spot patterns. In recent years, machine learning techniques have shown promise in automating and improving the accuracy of rotation period measurements. Deep learning models, in particular, have demonstrated the ability to extract complex features from time series data and make accurate predictions across a wide range of stellar types and rotation periods.

In this paper, we present a novel deep-learning approach for measuring stellar rotation periods using Kepler light curves. Our method, which we call LightPred, combines different neural network architectures to capture both local and global temporal features in the light curves. We train our model on a combination of simulated and real Kepler data, allowing it to generalize well to a variety of stellar types and activity patterns.

In the following, we first introduce the necessary background and context for our analysis, discuss previously used methodologies, and then present our analysis and results. Our code is open-sourced at https://github.com/IlayMalinyak/lightPred.

In the following subsections, we present existing works that use classical and data-driven approaches for period determination. We note that this is not intended to be a complete survey of the field, but rather a few pertinent examples.

# 1.1. Classic rotation measurements analysis methods

Various works have been done to directly infer the rotation period from light curves. We refer to those methods, which do not use learning or optimization processes, as ”classical methods”. Basri et al. (2011), Reinhold et al. (2013) and Nielsen et al. (2013) used Lomb–Scargle periodogram (Lomb 1976; Scargle 1982) to detect the period. A different method was used in McQuillan et al. (2013) and McQuillan et al. (2014), and called Autocorrelation function (ACF) - Given a time series data $x = x _ { 1 } . . . x _ { N }$ , ACF measures the correlation between the data and a lagged version of it and is calculated as follows:

$$
r _ {k} = \frac {\sum_ {i = 1} ^ {N - k} (x _ {i} - \bar {x}) (x _ {i + k} - \bar {x})}{\sum_ {i = 1} ^ {N} (x _ {i} - \bar {x}) ^ {2}}.
$$

where $r _ { k }$ is the autocorrelation coefficient at lag k. It can be seen that $r _ { k }$ is high if the input has a periodicity of length k. After applying ACF for a range of lags, the result is another time series data with peaks at the period and its multiplications.

another common method is the wavelet transform (Grossmann & Morlet 1984; Torrence & Compo 1998). Wavelet transform is the result of multiplying a function with some scaled and translated version of a specific response function (usually called the ”wavelet mother”). The Continuous Wavelet Transform (CWT) of a function x(t) at scale a and translation b is the following 2 parameters function:

$$
W (a, b) = \frac {1}{| a | ^ {\frac {1}{2}}} \int_ {- \infty} ^ {\infty} x (t) \psi^ {*} (\frac {t - b}{a}) d t
$$

where $\psi ^ { * } ( t )$ is the complex conjugate of the mother wavelet. Wavelet transform can be seen as Short Time Fourier Transform (STFT) with varying window sizes. To compare different methods in a controlled process, Aigrain et al. (2015) conducted a blind test to compare the different methods for period calculation. It was done using a dataset of 1000 simulated light curves where the underlying parameters are known. Among the suggested methods, the ACF method was the only method that was applied to the entire dataset and achieved superior results relative to other methods. Attempts were also made to combine multiple methods; Garc´ıa et al. (2014),Santos et al. (2019), and Santos et al. (2021) have used both Wavelet transform and ACF. Santos et al. (2021) also used a machine learning algorithm (suggested by Breton et al. (2021)) to select the most reliable period between the different results. The probabilistic approach was also tested; Angus et al. (2017) used Gaussian processes to predict periods of 1132 Kepler object of interest Shapiro et al. (2020) which suggested a variant of the wavelet method is also worth mentioning. This method, called Gradient Power Spectrum (GPS), takes the highest value of the gradient of the wavelet transform (inflection point) as an indicator for the period. Reinhold et al. (2023) used both ACF and GPS to create the largest current sample of stellar periods. They used a scoring function to decide whether to use the ACF prediction or the GPS. This suggests that ACF and GPS are the preferred methods to analyze periods of large samples in a non-learning fashion. One shortcoming of classical methods is their inability to provide uncertainty estimates for predictions, which are essential for evaluating the reliability of the results. To address this, statistical measures are often applied. Nielsen et al. (2013) used the median absolute deviation (MAD) of the quarterly predicted period from the median period to estimate uncertainty. McQuillan et al. (2013) and McQuillan et al. (2014) also used a variant of the MAD estimation. Santos et al. (2019), Santos et al. (2021) used the half width at half maximum (HWHM) of a Gaussian profile fitted to the highest peak of the wavelet power spectrum, and Reinhold et al. (2023) used the difference between the power spectrum at the inflection point and the minimum power as a signal to noise estimation. A different and important example is Angus et al. (2017), which used a probabilistic model that outputs uncertainties inherently.

# 1.2. Machine-learning analysis methods

A different approach is a data-driven approach. In this approach, one uses a dataset of light curves with known parameters to train a model and predict the desired properties. Then, using the trained model, one can predict properties on real data samples. Several works have been done in the field of period predictions: Lu et al. (2020) used a machine learning model that was trained to predict long rotation periods from shortduration measurements. They used MAD as an uncertainty estimation. Breton et al. (2021) used Random Forest algorithms to perform classification and vetting - first predicting if the sample is periodic enough, then predicting if it is not polluted by Pulsators and binaries, and lastly, choosing the best period from different classical methods. For uncertainty estimation, they used HWHM. For the task of period prediction, few attempts were made to use neural networks- Blancato et al. (2022) used a simple CNN network to predict periods among other properties of Kepler stars. For Labels, they used values that were calculated using the ACF method from McQuillan et al. (2014). As such, the model is restricted to the ACF results. Claytor et al. (2022) used simulations to train a 2d CNN model, with wavelet transform as input, and predicted periods of TESS-like simulated samples. based on that work, Claytor et al. (2024) used the same model to predict the period of 7245 TESS samples. Their model is optimized to predict, in addition to the period, uncertainty based on MAD.

Each of the approaches, data-driven and classical, has pros and cons. While the data-driven approach relies on a model that can learn complex high-dimensional functions, it is much less interpretable. It assumes that the simulated data represent the real data sufficiently. Classical methods are more sensitive to parameter tuning and usually less robust as they are applied only to a subset of the samples.

In this work, we suggest a data-driven method to derive the inherent parameters of stars using a deep learning model. Our work brings the following contributions

• We use a simulation-based deep learning model combined with self-supervised learning. Simulation-based learning for period determination was done only by Claytor et al. (2022) and Claytor et al. (2024) for predictions of TESS samples and not for Kepler samples. To the best of our knowledge, the use of self-supervised learning for period determination was not done at all.   
• We use new error metric which combines a learned confidence level, optimized together with the predictions, and a consistency measure based on the ability to reproduce the same predictions on different segments of the same sample.   
• We predicted periods for a large sample of main sequence stars for the first time with better accuracy compared to established methods.

The paper is organized as follows: In section 2, we discuss the creation of the mock dataset for the model’s supervised training, as well as the pre-processing stages for both simulated and Kepler data. Section 3 discusses the model’s architecture. In section 4, we discuss the training pipeline on both simulations and Kepler, and in Section 5, we present the results on both simulated and Kepler data, analyze the model’s errors, discuss possible pollution in the dataset, and different ways to constrain the results. Section 6 contains the conclusion and discussion. The appendix contains supplementary material with further examples, implementation details, additional figures, and tables.

# 2. DATA

# 2.1. Mock Dataset

To create a dataset for supervised training we used simulated light curves using Butterpy, a python package described in Claytor et al. (2022). The simulation is based on a simplistic spots model that alters the flux observed from a star, based on the spot’s activity and configuration, and creates a unique light curve. The spots emerge and decay as Gaussian with characteristic emergence and decay time scales. In addition, each spot is located at a random latitude (within a range that is chosen from prior distribution) and longitude and can either decay towards the equator (to create the known butterfly shape) or not. The suggested model implies an inherent degeneracy between inclination and spot location. This was already observed by Walkowicz et al. (2013), Basri & Shah (2020), and Luger et al. (2021), who suggest that without knowing the full spot distribution of each star, learning the stellar inclination is limited. As stated in Luger et al. (2021), This implies that the model is prior-depended: the distribution of spots in the model has a large effect on the results. Our sun has unique spot distribution - the spots emerge on the latitude of maximum ∼ 30 degrees and decay towards the equator (Hathaway 2011). This decay, with a time scale of the solar cycle and some overlap between cycles, creates the known butterfly effect. However, it is not well known whether this pattern (latitude distribution, decay, and cycle overlap) is consistent with other stars. While butterfly pattern was observed in some stars (Netto & Valio 2020; Bazot et al. 2018), other stars showed no decay pattern and wide latitude distributions. This was found in both solar-like stars (Thomas et al. 2019) and in younger and more active stars (Mackay et al. 2004). Observations of wider latitudes on more rapidly rotating stars (as seen in Mackay et al. 2004) suggest that there’s a relation between stars’ period and spot distribution. Indeed, Morris (2020) suggested a relation between stellar age and spot total coverage. Combining this with the known period-age relationships (Skumanich (1972), Barnes (2003), Barnes (2007), Barnes (2010), Mamajek & Hillenbrand (2008), Angus et al. (2019),Bouma et al. (2023)), can lead to a relation between stellar period and spot total coverage. However, as stated in Morris (2020), this relation is of a statistical nature, and applying this relation to the spot distribution of a specific star is not trivial. As such, it was not implemented in our simulation. We will elaborate on this in section 5.

From the above discussion, it is clear that identifying the line-of-sight inclination of a star is potentially much more challenging compared with identifying a stellar rotation. Although both depend on the spot distribution and behavior, the inclination determination is more sensitive to the spot distribution of a given star, of which we still have a limited understanding. We therefore treat inclination and period predictions separately; in this paper, we investigate the period predictions and treat inclinations as an auxiliary variable, while in a future paper, we will focus on the more complex issue of inclination determination. Nevertheless, the information from the inclination of samples is still valuable for this work. By training a model to predict periods and inclinations at the same time we rely on the fact that mutual learning can benefit both aspects independently. In section 4.3 we show this explicitly for the period determination, but this is also backed with simple intuition: On the one hand, the spots distribution and the stellar inclination will determine which and how many spots will rotate in and out of view and hence determine the amplitude of the periodic-changes in the light curve; these would also affect aspects of differential rotation, as the period might change at different latitudes. When predicting the equatorial rotation, the inclination information might help with ”locating” the equator and determining the noise level from spots, which do not rotate out of view. On the other hand, the information about inclination is affected by the spots that rotate with the stellar period, and the amplitude of the periodic signal related to the ratio between regions in the stars that go in and out of view and those that do not, which depend on the inclination. Indeed, such inclination measurement aspects using the rotation measurement amplitude were demonstrated by Mazeh et al. (2015).

As mentioned above, prior distributions have a large effect on the results and we chose slightly different parameters than Claytor et al. (2022) and Aigrain et al. (2015). specifically, for the inclination of the star, i, we used an isotropic distribution which translates to a uniform distribution in cos(i) instead of an isotropic uniform in $s i n ^ { 2 } ( i )$ , chosen in the original papers. This is a result of observational bias - for isotropic distribution of inclinations, we will see many more equator-on stars than edge-on. For the period, we used the same distribution as in Aigrain et al. (2015). The full parameters of the simulation distributions are specified in Table 1.

# 2.2. pre-processing

We pre-processed each sample in the following way: First, we cut the first 200 days to reach a steady state for spot evolution. Then we randomly chose a 720-day subset of the light curve (corresponding to 8 quarters of Kepler data). Next, we injected noise from real Kepler data. To do this we followed the same procedure as Aigrain et al. (2015) and used the results from McQuillan et al. (2014) which assigned the ’periodicity’ parameter w. w is a parameter between 0 to 1 which is a function of the relative height of the largest peak of the ACF (Local Peak Height - LPH), the effective temperature, and the period predicted by the ACF. In their work, McQuillan et al. (2014) considered only samples with $w > 0 . 2 5$ for period calculation. We chose samples with $w < 0 . 0 4$ which results in 6981 stars which we refer to as ”noise”. For each simulated sample, we randomly chose a star from the ”noise” samples and median normalized it. The addition of the noise was performed in two ways. The first is simply by adding the noise to the sample. The second involves scaling the noise sample such that $s t d ( n o i s e ) = \alpha * s t d ( s a m p l e )$ , where α is a random number between 0.02 − 0.05, before adding it. In this way, we inject Kepler data characteristics into the light curve while keeping the signal-to-noise ratio small. The reason we implemented two approaches is to be able to directly compare our results with Aigrain et al. (2015). We refer to samples that were processed through the first method as ”noisy samples” and samples that were processed using the second method as ”noiseless samples”. We then applied a Savitzky-Golay filter with a window of 13 timepoints (6h) and created a time series of differences between consecutive time stamps to reduce long-term variations. We decided to use the differences of a light curve instead of the original light curve; While computing the time series differences might be a non-trivial choice, it is a known technique to remove trends (Hyndman & Athanasopoulos 2018) and can be thought of as an approximation of the derivative. As such it removes linear trends while keeping the periodicity. We also justify this choice experimentally in 4.3. We refer to the result of this stage as Diff-Lc. Next, we calculated the ACF of the Diff-Lc with all possible lags and filled missing values by linear interpolation. Note that the calculation of ACF on the different light curves is different from that of the common use of ACF (where it is calculated on the light curve itself). We therefore use the term Diff-ACF to emphasize this difference. Lastly, we normalized both the Diff-ACF and Diff-Lc to zero mean and unity standard deviation. The result is a two-channel time series which is the input for the model. An example of the simulated light curve and the pre-processing stages is presented in Figure 1. More examples can be found in Appendix A

# 2.3. Kepler dataset

In addition to the simulated light curves, we also created a dataset from Kepler data. First, we downloaded the long cadence Kepler dataset Data Release 25 (STScI 2016). The data is available at https://doi.org/ 10.17909/T9488N

<table><tr><td>parameter</td><td>range</td><td>distribution</td></tr><tr><td>Inclination i</td><td>0°-90°</td><td>uniform in cos(i)</td></tr><tr><td>Period  $P_{eq}$ </td><td>10-50 (90%) 0-10 (10%) days</td><td>log-uniform</td></tr><tr><td>Spot lifetime  $\tau_{spot}$ </td><td>1-10 Period</td><td>log-uniform</td></tr><tr><td>Activity Cycle length  $T_{cycle}$ </td><td>1-10 Years</td><td>log-uniform</td></tr><tr><td>Activity Cycle overlap  $T_{overlap}$ </td><td>0.1- $T_{cycle}$ </td><td>log-uniform</td></tr></table>

Table 1. Parameters for simulated light curves. parameters that are not specified here are taken from Aigrain et al. (2015).

![](images/f8b31a8e027dc45381a7ddf944b50c0da7ec5a903dcde300b20a66fbd0b13726.jpg)

<details>
<summary>line</summary>

| Time (Days) | Relative Flux (%) |
| ----------- | ----------------- |
| 400         | 0.9980            |
| 600         | 0.9975            |
| 800         | 0.9965            |
| 1000        | 0.9980            |
</details>

![](images/88f1870db3b77d42485b15aebec579ad0e9f8656f8f2985877856787af3b263f.jpg)

<details>
<summary>line</summary>

| Time (Days) | Value  |
| ----------- | ------ |
| 400         | 1.000  |
| 500         | 0.999  |
| 600         | 1.001  |
| 700         | 0.997  |
| 800         | 0.996  |
| 900         | 1.003  |
| 1000        | 1.001  |
</details>

![](images/a2d03b82556dda2c10c4f5b855136b548e2ee4eddea7ff8c5640b2ddb30fa65a.jpg)

<details>
<summary>line</summary>

| Time (Days) | Averaged |
| ----------- | -------- |
| 400         | 0.9982   |
| 500         | 0.9978   |
| 600         | 0.9972   |
| 700         | 0.9965   |
| 800         | 0.9970   |
| 900         | 0.9978   |
| 1000        | 0.9985   |
</details>

![](images/86aaf5d03d1b1acb6acbd7b5abd426df9ee64619abe99f0f3ad3d71b94e75334.jpg)

<details>
<summary>line</summary>

| Time (Days) | Value |
| ----------- | ----- |
| 300         | 10    |
| 350         | -10   |
| 400         | 5     |
| 450         | 10    |
| 500         | -15   |
| 550         | 15    |
| 600         | 5     |
| 650         | 0     |
| 700         | -5    |
| 750         | 0     |
| 800         | 5     |
| 850         | 25    |
| 900         | -25   |
| 950         | 15    |
| 1000        | 0     |
</details>

![](images/21c9ac57b4b88e3a50f2e6a74d4d93e2252177b76d065616667fcc0dee32d8cf.jpg)

<details>
<summary>line</summary>

| Time Lag (Days) | Diff-ACF |
| --------------- | -------- |
| 0               | 0.2      |
| 100             | 0.05     |
| 200             | 0.02     |
| 300             | 0.01     |
| 400             | 0.005    |
| 500             | 0.002    |
| 600             | 0.001    |
| 700             | 0.0      |
</details>

![](images/b5edd025181bcc1438ce0f50e5b6ee672d33d07cacfa9c845dc63925c96c47bf.jpg)

<details>
<summary>line</summary>

| Time (Days) | Normalized |
| ----------- | ---------- |
| 300         | -10        |
| 400         | 5          |
| 500         | -15        |
| 600         | 10         |
| 700         | 0          |
| 800         | -25        |
| 900         | 25         |
| 1000        | -15        |
</details>

Figure 1. Example of a simulated light curve and the different pre-processing stages. The stages are shown sequentially from the upper left to the lower right. On the upper left panel is the raw light curve, randomly cropped to the 720-day segment. The upper middle panel shows the light curve after the addition of Kepler noise, and the upper right panel shows the resulting light curve after averaging with a 6h window. The lower left panel shows the light curve differences (Diff-Lc), and the lower middle panel shows the resulting normalized Diff-ACF. The lower right panel shows the normalized differences light curve. The input for the model is a 2-channel time series corresponding to the last two panels.

Similarly to McQuillan et al. (2014) and Reinhold et al. (2023), we omitted Q0-Q2 and Q17 and used the data corrected for instrumental systematic using PDC-MAP (Stumpe et al. 2012), (Smith et al. 2012). We then divided each quarter by the maximum value, normalized it so that all quarters would have the same average, and stitched together all the quarters. We then preprocessed the sample in the same way as the mock data, except for noise addition. For the contrastive learning setting, which we will elaborate on later, we added the

following data augmentations:

Masking similarly to Morvan et al. (2022), we masked 20% of the time points and replaced 20% of the masked points with a random value from a uniform distribution between −2 and 2.

Gaussian Noise we added Gaussian noise with standard deviation of $1 0 ^ { - 4 }$ .

Shuffle this augmentation shuffles different quarters of the same kepler sample. We acknowledge that shuffling quarters (90 Days) might remove phase information and potentially hinder the detection of long periods. However, we believe that the benefits of shuffling in terms of improving the model’s ability to generalize outweigh the potential drawbacks.

For each sample in a self-supervised contrastive setting, in addition to the pre-processing stage, we chose one of the following augmentations: masking, Gaussian noise, shuffle, or identity (no augmentation). The result is a 2-channel time series data, similar to the simulated data.

# 3. MACHINE LEARNING MODEL

In recent years, deep learning models have shown promising results on a wide variety of time series tasks (for reviews on forecasting and classification you can look at Lara-Ben´ıtez et al. (2021), Ismail Fawaz et al. (2018)). To find an optimal model, we started by using classical architectures for time series analysis, namely, CNN and RNN. CNNs are well known to work well on time series data. Indeed, both Blancato et al. (2022) and Claytor et al. (2024) used them as an architecture to analyze light curves. RNNs (Recurrent Neural Networks), with their sequential nature, are also effective in analyzing time series. however, the vanilla architecture of RNNs suffers from vanishing/exploding gradient as noted by Bengio et al. (1994). An Alternative that aims to improve this problem is the Long-Short-Term Memory (LSTM) network (Hochreiter & Schmidhuber 1997) which adds gates that selectively retain or discard information. Using this intuition, our basic approach used a combination of CNN and bi-directional LSTM. This showed good performance on noiseless samples but was less ideal when adding noise (as explained in 2.2). To better handle noisy samples we added another branch which is transformer-based. Transformers, which revolutionized the field of natural language processing, were also shown to be very effective for time series analysis. This gave rise to a family of transformer-based architectures including Informer (Zhou et al. 2021), Autoformer (Wu et al. 2021), and Conformer (Gulati et al. 2020). Indeed, Morvan et al. (2022) used a transformer-based model to denoise light curves, and Pan et al. (2024) used them to predict logg from light curves. We therefore added a transformer-based branch which is motivated by the work of Pan et al. (2024). The addition of the extra branch performed better than each branch separately and improved the results on noisy samples. In section 4.3 we present the metrics used to evaluate different models and show an ablation study for different parts of the model.

# 3.1. lightPred model

In the following we provide an overview of the model; detailed model schematics can be found in Appendix A. Temporal branch - consists of an ”embedding” convolution block with max pool, dropout, batch normalization, skip connection, a bi-directional LSTM block, and a non-learnable self-attention block. The convolution block gets the Diff-ACF, transforms it into a multi-channel 2d input, and serves as an effective embedding layer. The LSTM block processes the short and medium-range dependencies and the attention block (with the LSTM’s final cell state as queries and the features of the last layer as keys and values) enhances the long-term memory of the network.

Global branch - here we follow the recent work of Pan et al. (2024) called Astroconformer. They used a modified version of the Conformer (Gulati et al. 2020) architecture which combines multi-head self-attention with convolution. The main change in Pan et al. (2024) is the use of rotary positional embedding (Su et al. 2023). We used a slight modification of Astroconformer; in the conformer block, instead of 2 convolution layers after the multi-head self-attention (MHSA), we used one convolution and another MHSA. This showed to have better performance. The input of this branch is Diff-Lc.

The features from the two branches are concatenated and sent into the prediction head; two linear layers with Gaussian Error Linear Unit (GELU) activation function and a dropout between them. GELU (Hendrycks & Gimpel 2016) is an activation function similar to the well-known Rectified Linear Unit (RELU). However, GELU is smoother, has non-zero values on the negative regime, and is differentiable everywhere which makes it a better alternative. The output of the model is four numbers: rotation period, inclination, period confidence score, and inclination confidence score. To find optimal hyper-parameters for the model we followed a two-step search. First, we tuned the temporal branch using optuna (Akiba et al. 2019), a Python package for hyperparameters optimization in deep learning frameworks. Then, we tuned the parameters of the entire model manually. The resulting hyperparameters are listed in Appendix B. Figure 2 shows a high-level architecture of the model. A more detailed diagram of the different parts of the model can be seen in Appendix B.

# 4. TRAINING

# 4.1. Contrastive Learning

Before presenting the training scheme we used, we give a short background on Self Supervised Contrastive Learning. Contrastive learning is a branch of representation learning. The goal of representation learning is to map high-dimensional data into meaningful lowdimensional representation. Maybe one of the most wellknown examples of a simple representation algorithm is principal component analysis (PCA). Since the advances of deep learning models, representation learning has become a very common task for deep neural networks. Generally speaking, deep learning models can be divided into two families - generative and discriminative models. Given a dataset, X, and labels Y , generative models try to learn the probability distribution of the data, $\{ x \in X \}$ , namely $p ( x )$ or $p ( x | y )$ (with or without labels) using latent representation, z, of the input, $x \in X$ . An example of generative models is Generative Adversarial Network (GAN) Goodfellow et al. (2014) and Variational AutoEncoders (VAE) Kingma & Welling (2022)]. Discriminative models, on the other hand, learn the conditional probability $p ( y | x )$ . Contrary to generative models, which can be trained with or without labels, discriminative models require labels. Self-supervised learning tries to apply a discriminative approach to the case where no labels are available. It is done using a pretext task, i.e. a task that the model can solve without the true labels. A good pretext task is a task that will enforce the model to learn meaningful representation during training that will be easily adapted to other downstream tasks. An example of a pretext task is to predict an angle in which an image is rotated (Gidaris et al. 2018), or prediction of masked elements (Devlin et al. 2019). Contrastive learning is a type of self-supervised learning where a specific pretext task is used. The task in contrastive learning is to find representations that are invariant to different views of the input. Usually, input is divided into positive and negative pairs and different views can be seen as different augmentations of the elements in the pair. The network would then be optimized such that positive pairs are close in the latent space and negative pairs are far away. This is usually done using a loss function that maximizes the similarity for positive pairs and minimizes the similarity for negative pairs. A well-known example is the Sim-CLR network (Chen et al. 2020). A modification for the SimCLR network was suggested by Grill et al. (2020) (BYOL) and Chen & He (2020) (SimSiam) which can be seen as ”SimCLR with positive pairs only”. Those networks only try to maximize the similarity between augmented views of positive pairs. To avoid collapsing to a trivial solution, Chen & He (2020) suggested the following architecture (in the following we will use the notations of Chen & He (2020)): given two randomly augmented views $x _ { 1 }$ and x2 of the same input x, the two views are processed through an encoder network, $f ,$ consisting of a backbone and Multi-Layer Perceptron (MLP) head. A prediction head, h, which is another MLP, transforms the output of one view and matches it to the other view. Given $p _ { 1 } = h ( f ( x _ { 1 } )$ and $z _ { 2 } = f ( x _ { 2 } )$ we define a loss function that minimizes the negative cosine similarity between them:

![](images/44538aa6f8c248030ca308411adc3da56f7984c226e1aea89b58736eace91b76.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Signal"] --> B["Diff-ACF"]
    A --> C["Diff Light Curve"]
    B --> D["Temporal Branch"]
    C --> E["Global Branch"]
    D --> F["+"]
    E --> F
    F --> G["Prediction Head"]
    G --> H["Period, Inclination, confidence"]
    style A fill:#f9f,stroke:#333
    style H fill:#ccf,stroke:#333
```
</details>

Figure 2. High-level diagram of the lightPred model.

$$
\mathcal {D} (p _ {1}, z _ {2}) = - \frac {p _ {1}}{\| p _ {1} \| _ {2}} \cdot \frac {z _ {2}}{\| z _ {2} \| _ {2}}
$$

an important character of this loss function is called the ”stop gradient”. It means that gradient would propagate only through $p _ { 1 }$ , and not through $z _ { 2 }$ . The total loss function of the model is the symmetric loss:

$$
\mathcal {L} = \frac {1}{2} \mathcal {D} (p _ {1}, z _ {2}) + \frac {1}{2} \mathcal {D} (p _ {2}, z _ {1})
$$

. In their paper, Chen & He (2020) show that using this loss function together with the stop gradient property the model can converge to a non-trivial solution. Many works explored self-supervised learning for time series data. Recently, Zhang et al. (2024) published a review on the different methods in time series self-supervised learning. In the field of self-supervised learning in astrophysics, most of the work has been done on images and spectra as reviewed in Huertas-Company et al. (2023). It is worthwhile to mention the study by Morvan et al. (2022) which used a self-supervised framework to reduce noise from TESS light curve samples using predictions of masked time stamps.

# 4.2. 2-steps Training

To train the model, we used a 2-step pipeline; First, we trained the model, without the prediction head, in a self-supervised fashion, with Kepler samples. We call this step a Self Supervised Learning (SSL) step. Next, we used the weights from the SSL Step as an initialization for supervised training with the mock dataset.

We call this step Simulation Based Supervised Learning (SBSL) step. The SSL step was done using the Contrastive framework that was suggested by Chen & He (2020) as explained above. As mentioned in section 2.3, for the SSL step we used special augmentations in addition to the regular ones: Shuffle, Gaussian Noise, and Masking. For each sample, we randomly chose 2 special augmentations (including Identity - no augmentation) and created 2 different views of the same sample. It is important to emphasize that we use those augmentations only at the SSL step and not in later steps (SBSL Step and inference on Kepler data). The reason is that those augmentations improve the latent representation founded by the model but it is not intuitive that they would benefit the actual predictions in a supervised setup.

As explained in 4.1, In the SSL step the model consists of the backbone (LightPred without prediction head), projection Multi-Layer Perceptron (MLP), and prediction MLP. For the projection MLP we used 3 Layers with dimension of 64. For the prediction MLP, we used 2 layers with dimensions 32 and 64. Figure 3 shows a diagram of the training pipeline.

After the SSL step we used the trained LightPred backbone as a starting point for the SBSL. This was done on a dataset of 50000 simulated samples of 1000 days that were generated as explained in 2.1 and 2.2. We then split the dataset into training (81%), validation (9%), and test sets (10%). This gave us 5000 samples for testing and 4500 samples for validating the model which is representative. While this is more than the 1000 samples used by Aigrain et al. (2015) to test their models, this is much lower than Claytor et al. (2022) and Claytor et al. (2024) which used 1 million samples for training. We experimented with doubling the dataset size to 100000 samples and found no significant change in the results so we decided to keep the dataset size with 50000 samples. During training, we scaled the ground truth labels to the range 0 − 1 and used the following loss function

$$
\mathcal {L} = \mathcal {L} _ {1} (\hat {c}, c) + \mathcal {L} _ {1} (\hat {p}, p),
$$

where $\mathcal { L } _ { 1 }$ represents the L1 loss, c and p are the absolute deviation and ground truth values respectively, and ${ \hat { c } } ,$ $\hat { p }$ are the models’ predictions for those values. We then defined the confidence of the model as 1 − cˆ.

During training, We used AdamW optimizer (Loshchilov & Hutter 2017). When optimizing a neural network using gradient descent, one of the most critical hyperparameters is the learning rate, which controls the step size in the direction opposite to the gradient. A low learning rate results in slow convergence, while a high learning rate can cause the optimization to oscillate without settling into a minimum. Although various techniques exist to find an optimal learning rate and adjust it dynamically during training, in this work, we used a fixed learning rate of $5 \times 1 0 ^ { - 4 }$ , determined experimentally.

# 4.3. Ablation Study

Before comparing our model to known results, We performed an ablation study to test the different parts of the model. An ablation study is a systematic approach used to understand the contribution of individual components within the model. This is done by selectively removing or modifying specific parts of the model or its training process and evaluating the resulting performance. In this study, we would like to test not only the high-level architecture but also the use of Diff-Lc, Diff-ACF, and the use of inclination as an auxiliary prediction, since they are all nontrivial choices that are worth justification. To perform the study we trained multiple models using the pipeline described in 4.2 on the same dataset, and compared the results. The metric we used to evaluate the models is the 10 percent accuracy - the percentage of points with predictions within 10% absolute error from the ground truth. the different models we tested are:

1. LightPred: The full model as shown in 3.1.   
2. LightPred no-diff: The same as LightPred with a change in the pre-processing; instead of using Diff-Lc and Diff-ACF, we used the regular light curve and ACF.   
3. Temporal Branch: Using only the Temporal branch; LSTM-based network with Diff-ACF as input.   
4. Conformer Branch: Using only the Conformer branch; a convolution-transformer network with Diff-Lc as input.   
5. LightPred period only: the same as LightPred but without inclination predictions. Namely, predictions of only the period.

Table 2 shows the results of the study. We can see that the best performance is achieved by the full model as described in 3.1. In addition, we see that both the use of Diff-Lc and Diff-ACF and the use of inclination predictions improved the performance of the model. In the next sections, by using the term LightPred we mean the full model, with Diff-Lc, Diff-ACF, and inclination predictions as explained in 3.1.

![](images/bc17a0c8bfb3e19e6abc83edaea9c3756a01d8f506176ed7ad29c17a8471c437.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Self_Supervised
        A1["backbone"] --> B1["MLP"]
        A2["backbone"] --> B2["MLP"]
        C1["x₁"] --> A1
        D["x₂"] --> A2
        E["Encoder"] --> F["LightPred"]
        F --> G["Simulated Sample"]
    end
    
    subgraph Supervised
        H1["Supervised"]
        I1["Supervised"]
        J1["Supervised"]
    end
    
    style Self_Supervised fill:#f9f,stroke:#333
    style Supervised fill:#bbf,stroke:#333
    
    note1["x₁"]
    note2["x₂"]
    note3["Similarity"]
    note4["stop-grad"]
    
    note5["Simulated Sample"]
    note6["Period, Inclination + confidence"]
    
    classDef sample fill:#eef,stroke:#333;
    classDef kernel fill:#eef,stroke:#333;
    classDef encoder fill:#eef,stroke:#333;
    classDef period fill:#eef,stroke:#333;
    classDef confidence fill:#eef,stroke:#333;
    class A1,A2,A3,B1,C2,C3,X1,X2,KellerSample
    class B1,B2,B3,B4,B5,B6,B7,KellSample
    class C1,C2,C3,C4,KellSample
    class D,E,F,G,H,I,J,K,K,L,K,N,K,P,Q,R,S,T,U,V,X,Y,X,Z,K,Y,Z,N,K,P,Q,R,S,U,V,X,Y,Z,K,N,K,P,Q,R,S,R,T,U,U,V,X,Z,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,L,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,L,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,L,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,L,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,N,k,L,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,L,K,N,K,P,Q,R,S,R,T,U,V,X,Z,K,L,K,N,k,L,K,N,k,L,K,N,k,L,k,N,k,L,k,N,k,L,k,N,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,L,k,l,k,l,m,n,n,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,m,n,\n\nTheta: Encoder\n\nSupervised: LightPred\n\nTheta: Encoder\n\nReference: Kepler sample\n\nTheta: Encoder\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference: Kepler sample\n\nReference:\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample\nsimulated_sample
```
</details>

Figure 3. Training Pipeline. In the Self Supervised Step, the LightPred model is trained without the prediction head.

<table><tr><td>model</td><td>acc10p (%)</td></tr><tr><td>LightPred no-diff</td><td>76</td></tr><tr><td>Temporal Branch</td><td>75</td></tr><tr><td>Conformer Branch</td><td>76</td></tr><tr><td>LightPred</td><td>78</td></tr><tr><td>LightPred period only</td><td>73</td></tr></table>

Table 2. Ablation study results.

# 5. RESULTS

# 5.1. Results on simulated data

To compare our results with Aigrain et al. (2015) we created another 50000 samples dataset, this time with the same prior distributions as in Aigrain et al. (2015). Note that the main difference in the distributions is in the inclinations distribution, where Aigrain et al. (2015) used uniform in sin2(i), while we used uniform in cos(i). We also adopted the same metrics - the percentage of samples that lie within 10% error (acc10p) and the percentage of samples that lie within 20% error (acc20p). In addition, we used the average absolute error in days (mean err).

We compare our results to the results reported by the Tel-Aviv team in Aigrain et al. (2015). We chose the Tel-Aviv team, as it obtained the best overall results in the comparison study; in addition, they were the only ones who used the ACF method and predicted periods for the entire dataset. The results are shown in Table 3. It can be seen that our results are better for both noisy and noiseless samples.

To take into account changes between the dataset provided in Aigrain et al. (2015) and the dataset we used, and to enable a more detailed comparison, we also implemented ACF and GPS methods and tested them on our test set. Figure 4 shows prediction plots of our model, our ACF, and our GPS implementations. Comparing the plots visually we can see that our model is more robust; both the ACF and the GPS models have more outliers and a non-negligible number of points that were effectively not predicted. In the case of ACF, when there are no high enough peaks we set the period to zero (essentially the period can not be determined). In the case of GPS, many samples were predicted with the maximum possible period (probably due to long-term trends in the data). None of this is apparent in our model’s result. To compare our model and ACF/GPS qualitatively we used two sets - the entire test dataset and a subset of only valid points. In the case of ACF, valid points are points with period > 0 and in the case of GPS we took periods that are lower than the maximal period. Table 3 shows this comparison with ACF and Table 4 shows the same for GPS, both for noisy samples. It can be seen that both ACF and GPS predicted effectively ∼ 85% of the dataset while our model predicted the entire dataset. For a subset of valid predictions, we see that the performance of ACF is comparable to our model in terms of acc10p and acc20p but our model shows significantly lower average error. In the case of GPS, our model is better in all aspects for both the entire dataset and the subset of valid predictions. We conclude that our model is more robust than current established methods. On a subset of points that are detectable by ACF, the performance of our model and that of ACF are comparable, and both our model and ACF have shown to be better than GPS.

An important difference between our model and other methods is the confidence level. In our model, a confidence estimate is a natural product. As explained in 4.2, we train the model to predict, in addition to the period and inclination, the L1 loss, normalized to the range 0 − 1, for each label (period/inclination) and for each sample which we define as ˆc. The confidence is then defined as 1 − cˆ. To see the relation between the confidence value and the model’s performance we show in Figure 5 the period confidence value vs the absolute error (absolute value of the difference between true period and predicted period). The trend of high confidence and low error, which can be easily seen in the figure, implies that the model was able to infer which samples are ”harder” for it to determine the stellar properties. It is possible to get confidence functions also from ACF and GPS, but usually, it requires extra processing of the results.

# 5.2. Inference of stellar rotations in Kepler data

Next, we predicted periods using our trained model on the Kepler dataset. The model was trained as described in 4.2. Graphs of the loss of the SSL and SBSL training steps can be found in Appendix C. We then used the trained model to predict periods from the Kepler dataset. Since the model gets fixed-sized samples and we worked with 8-quarters light curves, we filtered out any sample that did not appear in 8 consecutive quarters. This left us with 130868 which serves both for the SSL step and the inference. As mentioned in 2.2 It is important to note that while we used the same samples for SSL training and inference, the pre-processing steps are different - specifically, we did not use in this step the special augmentations (masking, Shuffle, Gaussian Noise) that we used in the SSL part.

# 5.2.1. Period predictions

For inference, we performed an additional selection process. Both McQuillan et al. (2014) and Reinhold et al. (2023) took stars with $T e f f < 6 5 0 0 K$ . We used a slightly larger threshold of 7000K to include stars with a radiative exterior that tends to rotate faster (Albrecht et al. (2022)). Therefore, Out of the 130868 predictions we filtered out stars with $T e f f < 7 0 0 0 K$ which gave us 126029 samples. Next we filter out known contaminants in the dataset - δ Scuti, γ Durados and Hybrids (Uytterhoeven et al. (2011), Bradley et al. (2015), Murphy et al. (2019), Van Reeth et al. (2018), Li et al. (2019a), Li et al. (2019b)); RR-Lyrae stars (Benk˝o et al. (2010), Nemec et al. (2011), Nemec et al. (2013), Forr´o et al. (2022)); Giants based on the criterion given by Ciardi et al. (2011); this leaves us with 108096 samples.

To estimate the error we used two methods:

• Simulation Error: To relate the error from the simulation to Kepler predictions, we binned the simulation test-set results to 1-day bins and defined the bin error to be the average absolute value difference from the true values per bin. We then rounded the Kepler predictions to integers and found the corresponding bin for each prediction. this way we assigned a period-dependent error for each sample. This approach gives an average error per period bin but does not differentiate between samples with similar periods and is therefore limited.

• Observational Error: Since each Kepler light curve sample is potentially longer than eight quarters, we can divide the full light curve into 8- quarter segments and check the consistency of the model with respect to different segments (for example, the consistency between predictions for Q3- Q10, Q4-Q11, etc.). McQuillan et al. (2014) used a similar approach, with different segment sizes to choose valid ACF predictions. In our case, different segment sizes are not possible since our model uses fixed-length input so we used a fixed size of eight quarters per segment. Since we have seven segments, there are in total, 21 quarter pairs for each sample. Figure 6 shows the distribution of differences in predictions between all the pairs in the dataset. It can be seen that the differences behave similarly to a Normal distribution. We therefore construct the following observational error; for each sample, we fit a normal distribution using all the pairs’ differences and take the 1σ value as the observational error.

To see if the two errors are related, we binned the Kepler predictions to 1-day bin, averaged the observational errors of each bin, and compared them to the simulation error of that bin. Figure 7 shows this comparison. It can be seen that for prediction < 40 days, there is a clear monotonic relationship between simulation and observational errors. For predictions < 25 days, the relation appears linear. This is an important evidence for the

![](images/b3813ca8fa41847d682da6284ca205d7171bf5b98363aa62bb302ef67a7d6eb8.jpg)

<details>
<summary>scatter</summary>

| True (Days) | Predicted (Days) | Confidence |
|-------------|------------------|----------|
| 0           | 0                | 0.825    |
| 10          | 10               | 0.875    |
| 20          | 20               | 0.900    |
| 30          | 30               | 0.925    |
| 40          | 40               | 0.950    |
| 50          | 50               | 0.975    |
</details>

![](images/f344e2bf207d2a333de3b0e3c1f1ca6b01f89327379df993b70c53a77655c1fc.jpg)

<details>
<summary>scatter</summary>

| True (Days) | Predicted (Days) |
| ----------- | ---------------- |
| 0           | 0                |
| 50          | 50               |
</details>

![](images/b845cec80d97f723f8977df63dc0f2a97ecbbc444a10c74e42b4432f3e3f08e9.jpg)

<details>
<summary>scatter</summary>

| True (Days) | Predicted (Days) |
| ----------- | ---------------- |
| 0           | 0                |
| 50          | 40               |
</details>

Figure 4. Results of our model (upper left) vs implementations of ACF (upper right) and GPS (bottom) on our test dataset. on the left panel, the color represents the model’s confidence levels. The sharp drop of samples, seen at 10 days, is a result of the prior distribution of periods (see Table 1). This distribution was also implement in Aigrain et al. (2015) and motivated by the bimodality found by McQuillan et al. (2013), McQuillan et al. (2014)

![](images/bf037df2e84a8e9db0df45e79bf420ecb21b2062baada368be882d4e4cbfb387.jpg)

<details>
<summary>scatter</summary>

| Confidence | Absolute Error (Days) | Counts |
| ---------- | --------------------- | ------ |
| 0.88       | 10                    | 20     |
| 0.90       | 20                    | 40     |
| 0.92       | 30                    | 60     |
| 0.94       | 25                    | 80     |
| 0.96       | 15                    | 100    |
| 0.98       | 5                     | 120    |
| 1.00       | 0                     | 120    |
</details>

Figure 5. Period confidence value vs absolute period error of the model.

<table><tr><td>model</td><td>acc10p (%)</td><td>acc20p (%)</td></tr><tr><td>LightPred Noiseless 8q</td><td>80</td><td>93</td></tr><tr><td>ACF Noiseless Tel-Aviv</td><td>75</td><td>90</td></tr><tr><td>LightPred, Noisy 8q</td><td>76</td><td>90</td></tr><tr><td>ACF Noisy Tel Aviv</td><td>68</td><td>80</td></tr></table>

Table 3. Comparison between the LightPred model and the ACF model as reported by the Tel-Aviv team at Aigrain et al. (2015)

<table><tr><td>model</td><td>percent predicted (%)</td><td>acc10p (%)all (subset)</td><td>acc20p (%)all (subset)</td><td>average error (Days)all (subset)</td></tr><tr><td>LightPred Noisy 8q</td><td>100</td><td>76 (78)</td><td>90 (91)</td><td>1.60 (1.28)</td></tr><tr><td>ACF Noisy Ours</td><td>83</td><td>65 (78)</td><td>76 (92)</td><td>6.86 (1.62)</td></tr></table>

Table 4. Comparison between the LightPred model; and our implementation of the ACF model. Each column presents the result on the entire dataset and the result on a subset of valid ACF predictions (in parenthesis), Best results are bolded.

<table><tr><td>model</td><td>percent predicted (%)</td><td>acc10p (%)all (subset)</td><td>acc20p (%)all (subset)</td><td>average error (Days)all (subset)</td></tr><tr><td>LightPred Noisy 8q</td><td>100</td><td>76 (76)</td><td>90 (90)</td><td>1.60 (1.36)</td></tr><tr><td>GPS Noisy Ours</td><td>86</td><td>61 (62)</td><td>77 (77)</td><td>4.06 (3.65)</td></tr></table>

Table 5. Comparison between the LightPred model and our implementation of the GPS model. Each column presents the result on the entire dataset and the result on a subset of valid GPS predictions (in parenthesis). The best results are emphasized in bold.

![](images/3d34fd26a6f7a0871d8f52a509bc1c530bbffba67350b824b1666fd55f4b4737.jpg)

<details>
<summary>histogram</summary>

| Period Pair Difference (Days) | Frequency |
| ----------------------------- | --------- |
| -20                           | 0.00      |
| -15                           | 0.00      |
| -10                           | 0.00      |
| -5                            | 0.05      |
| 0                             | 0.25      |
| 5                             | 0.10      |
| 10                            | 0.02      |
| 15                            | 0.00      |
| 20                            | 0.00      |
</details>

Figure 6. Distributions of predictions-difference between pairs of different quarters of the same sample for Kepler samples. The orange curve represents the fit to a normal distribution.

relation between simulation and real data predictions. The fact that we see such a clear relation between the two suggests that, at least for periods < 25 days, the Kepler data is sufficiently represented by the simulations for period predictions. For longer periods the correlation is different which might be related to a drop in the activity for older stars.

# 5.2.2. Comparison with ACF

Next, we want to compare our method with classical methods on the Kepler data. Our experiments on mock data with ACF and GPS showed that while GPS is very sensitive to parameter tuning, ACF was more robust and performed better. We therefore choose ACF for that comparison. To compare our method with ACF, we created an ACF prediction on different segments for the entire sample. We chose the parameters of the ACF such that the results are as similar as possible to the samples published by McQuillan et al. (2014) and calculated the observational error for the ACF, as explained 5.2.1. When comparing observational error results, it is worth noting an inherent bias in the ACF method - while our model predicts a value for each sample, ACF is different. As discussed in section 5.1, there are samples for which the ACF is not able to predict any rotation period at all (when the ACF signal has too shallow peaks) in which case we assign a predefined value. The fact that we assign the same value for all the nonpredicted segments creates a falsely high consistency for those samples. This is why we conduct two comparisons with ACF - a comparison where we used all the samples and a comparison where we used a subset of samples where we follow the general selection rules of McQuillan et al. (2014), i.e. we remove planet-hosting stars and eclipsing binaries and take only samples where ACF was able to predict all the segments. The size of the second sub-sample is 36154, i.e. only $\sim \textstyle { \frac { 1 } { 3 } }$ of the data.

Figure 8 shows a comparison of Observational error distribution between the LightPred model and the ACF model for the two sets of samples. In both cases, the ACF shows a tail of very large errors. Our model shows more concentrated distributions and when considering the entire samples (left panel), our model has much lower error. When considering only ACF-valid samples (right panel), The errors are comparable, with LightPred error slightly lower than ACF error. When considering the relative observational error (dividing by the predicted period) we get the same error on average (11%). This suggests for Kepler data what we already saw for simulated data, that is, our model is not only able to determine the rotational period for more samples but is also at least as robust as ACF. Table 6 summarizes this comparison.

We also want to test the efficiency of the selfsupervised phase. Table 7 compares the results both on simulation data and Kepler data for training with and without the self-supervised pre-training phase. We see that self-supervised training improves all the metrics both on simulation and Kepler data. While the improvement on simulation metrics is modest, the improvement on Kepler data is significant (∼ 3 times lower observational error). The significant change in observational error can be understood by the fact that there is a relation between the contrastive task and the observational error - the latter can be seen as a special case of similarity between different views of the same sample. Note that the results of the simulation data in Table 7 are slightly different than the results shown in Section 5.1. This is because here we used different inclination distributions as explained in section 2.1.

Figure 9 shows a comparison between our model and the three largest available catalogs of Kepler stellar periods: McQuillan et al. (2014), Reinhold et al. (2023) and Santos et al. (2021). The colors of the points refer to the learned confidence of the model (the output of the model). It can be seen that in general, more confident samples have better agreement. This implies that the learned confidence represents physical properties and is somehow similar to the scoring functions used in McQuillan et al. (2014) and Reinhold et al. (2023). Another interesting observation is the fact that almost all the points that disagree are located in two distinct regions - the first region is around the half period line Which suggests a ’double period mistake’. The second region is where the reference period is close to zero but our model’s predictions aren’t. This can be seen as a vertical line of points close to $x = 0$ . Both regions are apparent in all three comparisons. We investigate these non-agreement samples more closely in 5.2.5.

# 5.2.3. Error analysis and constraints

Kepler observations are diverse in terms of parameters such as magnitude, distance, stellar activity, etc. Since those parameters affect the observed light curve they affect the predictions. Therefore, we would like to investigate correlations between our errors and different parameters. This is important both as a sanity check for our metrics and to be able to better constrain our results and remove potential biases. In 5.2.1 we saw that the observational error and the simulation error are related and both depend on the period. It is beneficial to normalize the error by the predicted period to remove biases related to the length of the period. In addition, our

![](images/56ab750a44ff7bfb19054ea3a2efa7eea895e74e49853260122503e3ef24bafc.jpg)

<details>
<summary>scatter</summary>

| Simulation Error (Days) | Obs. Error (Averaged over bins) (Days) | Predicted Period (Days) |
| ----------------------- | ------------------------------------- | ----------------------- |
| 0.5                     | 0.5                                   | 5                       |
| 0.6                     | 0.6                                   | 7                       |
| 0.7                     | 0.7                                   | 10                      |
| 0.8                     | 0.8                                   | 12                      |
| 0.9                     | 0.9                                   | 15                      |
| 1.0                     | 1.0                                   | 18                      |
| 1.1                     | 1.1                                   | 20                      |
| 1.2                     | 1.2                                   | 22                      |
| 1.3                     | 1.3                                   | 25                      |
| 1.4                     | 1.4                                   | 28                      |
| 1.5                     | 1.5                                   | 30                      |
| 1.6                     | 1.6                                   | 32                      |
| 1.7                     | 1.7                                   | 35                      |
| 1.8                     | 1.8                                   | 38                      |
| 1.9                     | 1.9                                   | 40                      |
| 2.0                     | 2.0                                   | 42                      |
| 2.1                     | 2.1                                   | 45                      |
| 2.2                     | 2.2                                   | 43                      |
| 2.3                     | 2.3                                   | 40                      |
| 2.4                     | 2.4                                   | 38                      |
| 2.5                     | 2.5                                   | 35                      |
| 2.6                     | 2.6                                   | 32                      |
| 2.7                     | 2.7                                   | 30                      |
| 2.8                     | 2.8                                   | 28                      |
| 2.9                     | 2.9                                   | 25                      |
| 3.0                     | 3.0                                   | 22                      |
| 3.1                     | 3.1                                   | 20                      |
| 3.2                     | 3.2                                   | 18                      |
| 3.3                     | 3.3                                   | 15                      |
| 3.4                     | 3.4                                   | 12                      |
| 3.5                     | 3.5                                   | 10                      |
| 3.6                     | 3.6                                   | 8                       |
| 3.7                     | 3.7                                   | 5                       |
| 3.8                     | 3.8                                   | 7                       |
| 3.9                     | 3.9                                   | 10                      |
| 4.0                     | 4.0                                   | 12                      |
| 4.1                     | 4.1                                   | 15                      |
| 4.2                     | 4.2                                   | 18                      |
| 4.3                     | 4.3                                   | 20                      |
| 4.4                     | 4.4                                   | 22                      |
| 4.5                     | 4.5                                   | 25                      |
</details>

![](images/c6d2453ddfb4aaa51482cc0306755950ec118cdfdcba02ededf62359bc8d1434.jpg)

<details>
<summary>scatter</summary>

| Simulation Error (Days) | Observational Error (Averaged over period bins) | Predicted Period (Days) |
| ----------------------- | ----------------------------------------------- | ----------------------- |
| 0.4                     | 0.4                                             | 5                       |
| 0.6                     | 1.0                                             | 10                      |
| 0.8                     | 1.7                                             | 15                      |
| 1.0                     | 2.3                                             | 20                      |
| 1.2                     | 2.5                                             | 15                      |
| 1.4                     | 2.7                                             | 10                      |
| 1.6                     | 2.9                                             | 5                       |
| 1.8                     | 3.2                                             | 5                       |
| 2.0                     | 3.0                                             | 20                      |
</details>

Figure 7. Simulation error vs Observational error (averaged over 1-day bin). The left panel shows all the samples and the right panel shows only samples with predicted periods smaller than 25 days. The red line represents a line with a slope of 2.

![](images/413e4e428e105b9092a8f144f7762d8186a18a11aee0e025063defdd076dd663.jpg)

<details>
<summary>bar_line</summary>

| Observational Error | ACF Density | LightPred Density |
| ------------------- | ----------- | ----------------- |
| 0                   | 0.15        | 0.30              |
| 5                   | 0.15        | 0.05              |
| 10                  | 0.02        | 0.01              |
| 15                  | 0.01        | 0.00              |
| 20                  | 0.00        | 0.00              |
| 25                  | 0.00        | 0.00              |
| 30                  | 0.00        | 0.00              |
| 35                  | 0.00        | 0.00              |
| 40                  | 0.00        | 0.00              |
| 45                  | 0.00        | 0.00              |
| 50                  | 0.00        | 0.00              |
</details>

![](images/838ae355abe76e09fe1a02bbb1edaa7ab886964f51eeb800fd1411d06d20cfd9.jpg)

<details>
<summary>line</summary>

| Observational Error | ACF Density | LightPred Density |
| ------------------- | ----------- | ----------------- |
| 0                   | 0.24        | 0.40              |
| 5                   | 0.03        | 0.05              |
| 10                  | 0.01        | 0.01              |
| 15                  | 0.00        | 0.00              |
| 20                  | 0.00        | 0.00              |
| 25                  | 0.00        | 0.00              |
| 30                  | 0.00        | 0.00              |
| 35                  | 0.00        | 0.00              |
</details>

Figure 8. Observational error distributions of the LightPred model vs the ACF model. The left panel shows all the samples (108785 samples) and the right panel shows a sub-sample of 36154 points with no planet-hosting stars, eclipsing binaries, and where ACF was able to predict all the segments, i.e. the ACF effectively could not predict a rotation period for the majority of the data.

<table><tr><td>Model</td><td>percent predicted (%)</td><td>Average Observational Error (All Data) (Days)</td><td>Average Observational Error (ACF Data) (Days)</td></tr><tr><td>LIghtPred</td><td>100</td><td>2.01</td><td>1.55</td></tr><tr><td>ACF</td><td>33</td><td>2.98</td><td>1.65</td></tr></table>

Table 6. Prediction results of LightPred model and ACF model. Average errors refer to observational errors. Best results are bolded

<table><tr><td>Method</td><td>acc10p</td><td>acc20p</td><td>Average Simulation Error (Days)</td><td>Average Observational Error (All Data) (Days)</td></tr><tr><td>w/ Self Supervised</td><td>78</td><td>92</td><td>1.58</td><td>2.01</td></tr><tr><td>w/o Self Supervised</td><td>71</td><td>88</td><td>1.80</td><td>6.77</td></tr></table>

Table 7. Comparison between training with and without self-supervised pre-training

![](images/ca9891649abf65db435a531f21a4aa10e47956af8c558e3e310e698ec54c2916.jpg)

<details>
<summary>scatter</summary>

| McQ14 Period (Days) | LightPred Period (Days) | Confidence |
| ------------------- | ----------------------- | ---------- |
| 0                   | 0                       | 0.80       |
| 60                  | 60                      | 0.95       |
</details>

![](images/1b317504954ee9a215aa844a0ec5869a1c589f174870370661521894bb3cc794.jpg)

<details>
<summary>scatter</summary>

| Reinhold23 Period (Days) | LightPred Period (Days) | Confidence |
| ------------------------ | ----------------------- | ---------- |
| 0                        | 0                       | 0.80       |
| 10                       | 10                      | 0.85       |
| 20                       | 20                      | 0.90       |
| 30                       | 30                      | 0.95       |
| 40                       | 40                      | 0.90       |
| 50                       | 50                      | 0.85       |
| 60                       | 60                      | 0.80       |
</details>

![](images/caf4de9d2312204704b78c70629455249ebc437221943aa460d0e79fb019e02d.jpg)

<details>
<summary>scatter</summary>

| Santos21 Period (Days) | LightPred Period (Days) | Confidence |
| ---------------------- | ----------------------- | ---------- |
| 0                      | 0                       | 0.80       |
| 60                     | 60                      | 0.95       |
</details>

Figure 9. Comparison of the period determinations between LightPred model, McQuillan et al. (2014) (left panel), Reinhold et al. (2023) and Santos et al. (2021). Colors represent model confidence scores. The red and orange lines represent slopes of 1 and 0.5 respectively. The inconsistencies, seen on short periods and half period regimes, are discussed in 5.2.5

model outputs confidence for each sample which serves as another ’error’ estimation. The observational error and the model’s confidence represent different types of uncertainties. The observational error represents uncertainty related to the data (sometimes referred to as Aleatoric uncertainty) and the confidence represents uncertainty of the model (sometimes referred to as Epistemic uncertainty). In order to take into account all of the above, we use the following total error :

$$
e _ {t o t} = \frac {e _ {o b s}}{\hat {p} (1 - \hat {c})}
$$

, where $e _ { o b s }$ is the observational error, $\hat { p }$ is the predicted period and $1 - { \hat { c } }$ is the model’s confidence.

We now want to see if the total error correlates with stellar parameters. If such a correlation exists, the total error can help us constrain the predictions to a meaningful subset of the samples. For such comparison we used effective temperature, distance, metallicity, and luminosity from Berger et al. (2020). In addition, we used the Kepler magnitude from the Kepler input catalog (KIC) and two different photometric proxies for activity. The first is called $R _ { v a r }$ and was suggested by Basri et al. (2011); this is the difference between the $5 ^ { t h }$ and the $9 5 ^ { t h }$ percentiles of the flux at a given segment. Indeed, $R _ { v a r }$ was used by Reinhold et al. (2017) to calculate activity cycles for 3203 Kepler stars. To calculate $R _ { v a r } $ we calculated the $5 ^ { t h }$ and $9 5 ^ { t h }$ percentiles on windows of length 90 days with an overlap of 45 days. We then defined $R _ { v a r }$ as the difference between the maximum and the minimum elements in this time series. Another activity proxy is $S _ { p h }$ that was suggested by Mathur et al. (2014) and showed to be correlated with chromospheric activity (Salabert et al. 2016). $S _ { p h }$ is defined as the mean standard deviation of the light curve over a window of five periods.

To see the correlations between different variables and the total error we did the following experiment; for each stellar property, we divided the samples into 200 bins within the range of the property, and calculated the average total error for each bin, neglecting bins with less than 30 points. Figure 10 shows these calculations for the effective temperature, distance, luminosity, and magnitude, $S _ { p h }$ and $R _ { v a r }$ . We see that while there is a clear correlation between the error and some parameters, for others there is weak/no correlation. The most striking example is $R _ { v a r }$ which looks almost periodic with the error. The problem with this test is that there are internal correlations between the properties. To take that into account we need to look at higher dimensional correlations such as 2d correlations. Such correlations were calculated in the following way: for a pair of properties, we first normalized each of the properties by subtracting the minimum value and dividing by the maximum value to get a dimensionless parameter between $0 - 1$ . We then binned the samples with 200 bins for each parameter and calculated the average total error for each 2d bin, again neglecting bins with less than 30 points. We then made use of principal component analysis (PCA) of the binned values and the total error to reduce the dimension. Lastly, we plotted the resulting PCA components for the binned values as a function of the total error. Figure 11 shows this calculation for pairs consisting of $R _ { v a r }$ and all other variables. We see that for all pairs there is a clear correlation with the error. Such correlations can potentially identify the parameter phase space region, in which the predictions are less reliable.

Another possible correlation is with the w parameter from (McQuillan et al. 2014, see 2.2). It is an important sanity check since we used samples with $w < 0 . 0 4$ as noise generators and we want to verify that the model indeed struggles to predict periodicity in those samples. This would reject the assumption of overfitting the model for those samples. The correlation is shown in Figure 12. It can be seen that up to $w \sim 0 . 7 $ , there is a clear monotonic correlation between the total error and w; high error correlates with low w as expected. For $w > 0 . 7$ there is interesting behavior with an opposite trend which is not fully understood. We revisit this inverse relationship in Section 5.2.5. Notably, the error for these particular samples, which constitute 6.6% of the data points, is comparatively small so we can safely reject the possibility of overfitting. The correlation result suggests that the total error indeed represents a physical constraint that affects the predictions. As such, the total error can serve as a reliable tool to constrain the results.

# 5.2.4. Pollution

In 5.2.3 we discussed the effect of stellar parameters on the results but there are other factors that might bias our predictions. It is possible that the observed light curve does not represent a single object but multiple objects. This can be a result of crowding; since Kepler pixels cover $4 ^ { \prime \prime } \mathrm { ~ x ~ } 4 ^ { \prime \prime }$ patch of sky it is plausible that multiple objects would enter a single aperture. Indeed, correction for crowding is one of the main tasks of the pre-search data conditioning pipeline. The pipeline is, of course, not optimal, and indeed, using adaptive optics of a sample of Kepler planet host candidate, Ziegler et al. (2017) found a probability of $1 2 . 6 \% \pm 0 . 9 \%$ for nearby star within $0 . 1 5 ^ { \prime \prime } { - 4 ^ { \prime \prime } }$ . Another possible source of bias is binaries. A system of multiple stars can bias the model towards an incorrect period (the orbital period of the binary for example). Many works investigated the problem of identifying false positive signals in the case of planetary systems (Torres et al. (2011), D´esert et al. (2015), Bryson et al. (2013), Mullally et al. (2016)) which led to multiple vetting procedures (Coughlin et al. (2016), Col´on et al. (2015)) but this is not the case for general Kepler stars. However, Santos et al. (2021) did identify possible contaminants for a sample of general Kepler stars.

![](images/dbcd2b92bc2e3708337259d31188e35f42d47d342b250304437742eb72a7839d.jpg)

<details>
<summary>scatter</summary>

| Teff (K) | total error |
| -------- | ----------- |
| 3500     | 0.275       |
| 4000     | 0.150       |
| 4500     | 0.175       |
| 5000     | 0.200       |
| 5500     | 0.225       |
| 6000     | 0.250       |
| 6500     | 0.225       |
| 7000     | 0.300       |
</details>

![](images/20e55e7feda00d87f9c86382b77dd5ee207c680400c7d9b37b296752e58fca01.jpg)

<details>
<summary>scatter</summary>

| Dist (pc) | total error |
| --------- | ----------- |
| 0         | 0.16        |
| 500       | 0.18        |
| 1000      | 0.20        |
| 1500      | 0.22        |
| 2000      | 0.24        |
| 2500      | 0.26        |
| 3000      | 0.28        |
| 3500      | 0.30        |
| 4000      | 0.26        |
</details>

![](images/f0204d21ff26e0fc039afa8382b60b9dce3b3a1ceb535ccac026887dda6b2829.jpg)

<details>
<summary>scatter</summary>

| kmag | total error |
| ---- | ----------- |
| 9    | 0.28        |
| 10   | 0.24        |
| 11   | 0.22        |
| 12   | 0.20        |
| 13   | 0.22        |
| 14   | 0.24        |
</details>

![](images/7b5f4f09d9906c75e650dd14c646cb1edcf3cdf2742f2f358fe5bf304bdf46bd.jpg)

<details>
<summary>scatter</summary>

| S_ph | total error |
|------|-------------|
| 0.00 | 0.20        |
| 0.01 | 0.19        |
| 0.02 | 0.21        |
| 0.03 | 0.22        |
| 0.04 | 0.20        |
| 0.05 | 0.23        |
| 0.06 | 0.26        |
</details>

![](images/09ea33286bf888068f9b81f7be536b3372a615c6b0abd7f25a59daa529830c17.jpg)

<details>
<summary>scatter</summary>

| Lstar (l☉) | total error |
| ---------- | ----------- |
| -1.5       | 0.30        |
| -1.0       | 0.15        |
| -0.5       | 0.20        |
| 0.0        | 0.25        |
| 0.5        | 0.28        |
| 1.0        | 0.30        |
| 1.5        | 0.35        |
</details>

![](images/1392c7a83e151ed49e8b8016f163ba9284bcfd0f4a1b7653791ed3da2b780036.jpg)

<details>
<summary>scatter</summary>

| R_var | total error |
|-------|-------------|
| 1     | 0.16        |
| 2     | 0.24        |
| 3     | 0.26        |
| 4     | 0.28        |
| 5     | 0.18        |
| 6     | 0.19        |
| 7     | 0.20        |
| 8     | 0.22        |
| 9     | 0.24        |
| 10    | 0.26        |
</details>

Figure 10. Correlation between stellar parameters and the total error on each panel, the Pearson correlation is shown on the legend

![](images/4ac56952c44bd181014a5c899b608d60d264379f0728009f766a3e12c9d39511.jpg)

<details>
<summary>scatter</summary>

| α * kmag + β * R_var | total error |
| --------------------- | ----------- |
| 0.00                  | 0.15        |
| 0.05                  | 0.20        |
| 0.10                  | 0.25        |
| 0.15                  | 0.30        |
</details>

![](images/4c3dfec6215455fb834f6a8999f14d181f130f8feb7fbd89b950bad3b19e1f7d.jpg)

<details>
<summary>scatter</summary>

| α * Dist + β * R_var | total error |
| --------------------- | ----------- |
| -0.20                 | 0.10        |
| -0.15                 | 0.15        |
| -0.10                 | 0.20        |
| -0.05                 | 0.25        |
| 0.00                  | 0.30        |
</details>

![](images/a2e191deb120ffffb7ef109bf92b7d5075d2f11f67b37f2ac953e58844f77c22.jpg)

<details>
<summary>scatter</summary>

| α * FeH + β * R_var | total error |
| ------------------- | ----------- |
| 0.15                | 0.30        |
| 0.20                | 0.25        |
| 0.25                | 0.20        |
| 0.30                | 0.15        |
</details>

![](images/ad5673fff63b7064cd53d21a632b214df13c61ff92528bdf6380054708f36b96.jpg)

<details>
<summary>scatter</summary>

| α * Teff + β * R_var | total error |
| --------------------- | ----------- |
| -0.10                 | 0.20        |
| -0.05                 | 0.25        |
| 0.00                  | 0.30        |
</details>

![](images/103679a7cfc3f01dd15be2cccf7b19bf70c8ebe8f997fdbe80fd1f6723d735ca.jpg)

<details>
<summary>scatter</summary>

| x_value | y_value |
| ------- | ------- |
| 0.10    | 0.30    |
| 0.15    | 0.25    |
| 0.20    | 0.20    |
| 0.25    | 0.15    |
</details>

![](images/a0240c736e68101990e9bd2e15e0b0f325d77b89e28cb850f3709a65c1cbc7ce.jpg)

<details>
<summary>scatter</summary>

| x_value | y_value |
| ------- | ------- |
| -0.25   | 0.10    |
| -0.20   | 0.15    |
| -0.15   | 0.20    |
| -0.10   | 0.25    |
| -0.05   | 0.30    |
| 0.00    | 0.25    |
</details>

Figure 11. 2d correlation between stellar parameters and the total error. On each panel the PCA parameters and Pearson Correlation are shown in the legend

We would like to test the sensitivity of our model to the different types of possible contamination. As a first step, we consider the contaminants specified in Santos et al. (2021). Specifically, we look at samples that have potential pollution in the PDC-MAP signal as defined by Santos et al. (2021). They found 1366 such samples (marked as No Rotation flag 6). When crossed-matched with our samples, we are left with 1046 samples. Figure 13 shows the distributions of periods and total errors for the sample of possible polluted samples and the entire samples. It can be seen that the distributions appear very similar. Kolmogorov-Smirnov (KS) test between the period distributions with a null hypothesis that the two samples were drawn from the same distribution, produced a p-value of $4 . 0 5 * 1 0 ^ { - 3 2 }$ which implies that there is a high probability that the samples were not drawn from the same distribution. However, taking into account the statistical nature of such analysis and the fact that the average error of the possible pollution is similar to that of the entire sample, we conclude that it would be very hard to identify such contaminants based on the error alone. We therefore made a conservative choice and removed those samples from the predictions.

![](images/72eb62a8d54ea289d96285436a7c09dc41b349f2887f3faa5c2ad24058eebb85.jpg)

<details>
<summary>scatter</summary>

| w    | total error |
|------|-------------|
| 0.00 | 0.25        |
| 0.01 | 0.24        |
| 0.02 | 0.23        |
| 0.03 | 0.22        |
| 0.04 | 0.21        |
| 0.05 | 0.20        |
| 0.06 | 0.19        |
| 0.07 | 0.18        |
| 0.08 | 0.17        |
| 0.09 | 0.16        |
| 0.10 | 0.15        |
| 0.11 | 0.14        |
| 0.12 | 0.13        |
| 0.13 | 0.12        |
| 0.14 | 0.11        |
| 0.15 | 0.10        |
| 0.16 | 0.09        |
| 0.17 | 0.08        |
| 0.18 | 0.07        |
| 0.19 | 0.06        |
| 0.20 | 0.05        |
| 0.21 | 0.06        |
| 0.22 | 0.07        |
| 0.23 | 0.08        |
| 0.24 | 0.09        |
| 0.25 | 0.10        |
| 0.26 | 0.11        |
| 0.27 | 0.12        |
| 0.28 | 0.13        |
| 0.29 | 0.14        |
| 0.30 | 0.15        |
| 0.31 | 0.16        |
| 0.32 | 0.17        |
| 0.33 | 0.18        |
| 0.34 | 0.19        |
| 0.35 | 0.20        |
| 0.36 | 0.21        |
| 0.37 | 0.22        |
| 0.38 | 0.23        |
| 0.39 | 0.24        |
| 0.40 | 0.25        |
| 0.41 | 0.24        |
| 0.42 | 0.23        |
| 0.43 | 0.22        |
| 0.44 | 0.21        |
| 0.45 | 0.20        |
| 0.46 | 0.19        |
| 0.47 | 0.18        |
| 0.48 | 0.17        |
| 0.49 | 0.16        |
| 0.50 | 0.15        |
| 0.51 | 0.14        |
| 0.52 | 0.13        |
| 0.53 | 0.12        |
| 0.54 | 0.11        |
| 0.55 | 0.10        |
| 0.56 | 0.09        |
| 0.57 | 0.08        |
| 0.58 | 0.07        |
| 0.59 | 0.06        |
| 0.60 | 0.05        |
| 0.61 | 0.06        |
| 0.62 | 0.07        |
| 0.63 | 0.08        |
| 0.64 | 0.09        |
| 0.65 | 0.10        |
| 0.66 | 0.11        |
| 0.67 | 0.12        |
| 0.68 | 0.13        |
| 0.69 | 0.14        |
| 0.70 | 0.15        |
| 0.71 | 0.16        |
| 0.72 | 0.17        |
| 0.73 | 0.18        |
| 0.74 | 0.19        |
| 0.75 | 0.20        |
| 0.76 | 0.21        |
| 0.77 | 0.22        |
| 0.78 | 0.23        |
| 0.79 | 0.24        |
| 0.80 | 0.25        |
</details>

Figure 12. correlation between w parameter from McQuillan et al. (2014) and the total error

Next we consider binaries. We again rely on the analysis of Santos et al. (2021) which divided samples that were marked as CP/CB (classical pulsator/close-in binary) into four categories:

1. type 1: Samples that showed high amplitude variations.   
2. type 2: Objects whose light curves resemble that of a contact binary signal.   
3. type 3: δ Scuti and/or γ Doradus candidates or alternatively stars whose light curves are polluted by a nearby star of this type.   
4. type 4: Objects whose light curves resemble that of a heartbeat star or a close binary with tidally excited oscillation signals.

In their paper, Santos et al. (2021) predicted periods only for type 1 CP/CD. In Figure 14 we compare the period and error distributions of all the CP/CD types. Interestingly, we find very different period distributions; while type 1 and type 3 show relatively short periods, type 2 shows relatively high periods, and type 4 shows periods similar to the entire sample. To investigate this more deeply, we analyze eclipsing binaries. Eclipsing Binaries (EB) are binaries whose orbital plane is edge-on with respect to the observer. Thus, when they rotate through the center of mass, an ’eclipse’ is seen. This makes them relatively easy to identify using light curves. To date, there are 2920 known EBs in the Villanova1 catalog (Kirk et al. (2016), Abdul-Masih et al. (2016)). It is known that due to tidal forces, those systems show a synchronization of the orbital period and the primary star period for short orbital periods (Hut 1981). Lurie et al. (2017) found that most of the binaries with period < 10 Days are synchronized. On the other hand, Simonian et al. (2020) found that most of the rapid rotators are, in fact, binaries, so we expect most of the rapid rotators to be tidally synchronized binaries. Since we know the orbital period of EB systems, we can test the period predictions in this regime.

Figure 15 shows the orbital period vs the ratio between the orbital period and the predicted period for both our model and an ACF. The colors in the Figure correspond to classes that were assigned to each sample by Lurie et al. (2017) using visual inspection of 2278 EBs. The difference between our model and the ACF is striking. We see a clear separation between samples with starspots modulations (sp) and other types of stars in the predictions made by our model. For $3 < P _ { o r b } < 1 0 .$ , sp samples are very close to the synchronization line (the gray line at $P _ { o r b } / P _ { r o t } = 1 )$ as expected. For $P _ { o r b } < 3$ Days and $P _ { o r b } > 1 0 $ Days we do not see synchronization. While the latter is expected, the former can be assigned to the limitations of the model with very fast rotators, exactly as we saw in Figure 9. We also see that ellipsoidal variations are, in general, overestimated compared to the orbital period, making them appear below the synchronization line, and samples with no apparent periodicity (other than eclipses) are underestimated compared to the orbital period. The ACF created a totally different prediction scheme, falsely recognizing almost every ellipsoidal variation with its orbital periods. We can also see that the synchronization line continues after the 10 Days threshold suggested by Lurie et al. (2017). This reveals a fundamental difference between classical and data-driven methods; our model was trained on data simulated from spots modulations and it implicitly differentiates stars with light curves that are different from the samples it was trained on. This phenomenon is not apparent in the ACF results since it is only able to identify periodicity, without any context. This also suggests that our model is able to distinguish between different stellar types. To test such a distinction, we show in the upper panel of Figure 16 the same plot of $P _ { o r b }$ vs $P _ { o r b } / P _ { r o t }$ colored by the model’s confidence (upper left panel) and the total error as defined in 5.2.3 (upper right panel). We see that while the total error does not separate the data well, the confidence of the model does separate it with very good correspondence to the classes shown in Figure 15. This implies that while different stellar types have the same consistency levels, and therefore a similar total error, the model learned to identify them since their variability structure is inherently different from that of the training data of the model. To find a threshold that can be used to filter the predictions, we constructed the following procedure. We defined a consistent prediction as one with an absolute deviation from the orbital period which is less than 40%, or less than a day. The specific choice of 40% and 1 day was made to create a good separation between synchronized and unsynchronized samples. We then binned the dataset to integer values of orbital periods and, for each bin, we calculated the fraction of consistent samples according to the above criteria. We then took the bin with the largest jump in consistent samples to be the cutoff. As a sanity check, we did the same procedure for the comparison with McQuillan et al. (2014) predictions (using the periods from McQuillan et al. (2014)). The graphs of consistent fraction per bin for both scenarios can be seen in Appendix D. We found that in both scenarios, 3 days seem to be an optimal cutoff, so we adopted this value. The lower panel in Figure 16 shows the following groups for $P _ { o r b } > 3$ days:

• synchronized samples (cyan). These are consistent samples with $3 < P _ { o r b } < 1 0 $ .   
• Unsynchronized samples (red). These are nonconsistent samples with $3 < P _ { o r b } < 1 0$ .

• samples with $P _ { o r b } > 1 0 ~ \mathrm { ( g r a y ) }$ .

The lower right panel shows the confidence histogram for each group and exhibits a very good separation as expected. We therefore decided to take the average confidence value of the unsynchronized group, 0.86, as a filtering threshold.

# 5.2.5. Filtering Predictions

Next, we want to investigate the different inconsistencies between our model and the predictions of McQuillan et al. (2014) as shown in Figure 9. For the fast rotators, as mentioned on 5.2.4, we found the same inconsistency with EBs, and in both cases we found 3 days as a good cutoff for fast rotators. Three examples of fast rotators, as indicated by (McQuillan et al. 2014), can be seen in D. Therefore, we decided to remove all the known EBs with orbital period < 3 days as well as samples where the ACF predicted periods < 3 days. Since fast rotators are dominated by binaries, and since we saw in Figure 15 that in this regime ACF identifies the binary orbital period rather than the stellar period, the two conditions can be understood as pointing to the same phenomena. We found a total of 3106 such samples.

Another group of inconsistent predictions is the samples around the ’half-period’ line in Figure 9. Figure 17, shows on the left panel the same comparison as seen on the upper left panel in Figure 9 with different coloring; colors in Figure 17 represent the w parameter that was given by McQuillan et al. (2014) as a score for periodicity. We see that the inconsistent samples have significantly lower w values, meaning they are the least periodic samples in this subset. This is not surprising given that we saw the same pattern for the model confidence in Figure 9 and the fact that the total error and w are correlated, as seen in Figure 12. Another interesting observation is related to high w values. We now see that those values represent samples with fast predictions; In this regime, we expect tidally synchronized binaries to dominate. We also saw that ACF always finds the orbital period in those cases, while our model separates different stellar classes. This might explain the negative correlation we saw in Figure 12 for very high w values.

To better examine this inconsistency, we visually examined 200 samples randomly sampled from this regime. During this examination, we found an interesting phenomenon that confused our predictions: McQuillan et al. (2013) reported that spots at different hemispheres produce a lower correlation at half the period. This aligns with the findings of Berdyugina & Usoskin (2003) that spots tend to form in active longitudes that are separated by 180◦. To take this into account, they checked if a higher second peak exists in the ACF. If such a

![](images/962aeec156629538ec0aa50685472994a26036d9211e48f4305694edbf58840a.jpg)

<details>
<summary>histogram</summary>

| Period (Days) | 1041 samples | all samples |
| ------------- | ------------ | ----------- |
| 0-5           | 0.03         | 0.01        |
| 5-10          | 0.07         | 0.11        |
| 10-15         | 0.11         | 0.10        |
| 15-20         | 0.05         | 0.04        |
| 20-25         | 0.02         | 0.02        |
| 25-30         | 0.01         | 0.01        |
| 30-35         | 0.00         | 0.00        |
| 35-40         | 0.00         | 0.00        |
</details>

![](images/23141cf96dabbf9c28f15152c9d5b54b991f4ae6c605b5dce318a56a3379fadf.jpg)

<details>
<summary>histogram</summary>

| Total Error Range | 1041 samples Density | all samples Density |
| ----------------- | --------------------- | -------------------- |
| 0.00 - 0.25       | 3.6                   | 2.7                  |
| 0.25 - 0.50       | 2.5                   | 3.1                  |
| 0.50 - 0.75       | 0.7                   | 0.6                  |
| 0.75 - 1.00       | 0.3                   | 0.2                  |
| 1.00 - 1.25       | 0.1                   | 0.1                  |
| 1.25 - 1.50       | 0.1                   | 0.0                  |
| 1.50 - 1.75       | 0.0                   | 0.0                  |
</details>

Figure 13. distributions of the entire sample (red) vs possible contaminants (gray): No Rotation flag 6 in Santos et al. (2021). the upper panel shows the period distributions and the lower panel shows the total error distributions.

![](images/a3e06e5b86c0f5225dbf4d2efd74ddab023b5cd33398c4692c579cb7df024175.jpg)

<details>
<summary>histogram</summary>

| Period (Days) | Density (1364 samples) | Density (all samples) |
| ------------- | ---------------------- | --------------------- |
| 0             | 0.25                   | 0.00                  |
| 5             | 0.15                   | 0.05                  |
| 10            | 0.05                   | 0.10                  |
| 15            | 0.02                   | 0.08                  |
| 20            | 0.01                   | 0.03                  |
| 25            | 0.00                   | 0.01                  |
| 30            | 0.00                   | 0.00                  |
| 35            | 0.00                   | 0.00                  |
| 40            | 0.00                   | 0.00                  |
</details>

![](images/66c87c837ea0b5a0f6ed07139c382949b370fe19bb8a1d074b65edb99c99c6a0.jpg)

<details>
<summary>histogram</summary>

| Period (Days) | 58 samples | all samples |
| ------------- | ---------- | ----------- |
| 0-5           | 0.00       | 0.00        |
| 5-10          | 0.02       | 0.10        |
| 10-15         | 0.04       | 0.06        |
| 15-20         | 0.06       | 0.04        |
| 20-25         | 0.08       | 0.02        |
| 25-30         | 0.06       | 0.01        |
| 30-35         | 0.02       | 0.00        |
| 35-40         | 0.00       | 0.00        |
</details>

![](images/050341758ed15c1825a009ac8e65d261d3cac78d378f5db9996e5be50f7449aa.jpg)

<details>
<summary>histogram</summary>

| Period (Days) | 120 samples | all samples |
| ------------- | ----------- | ----------- |
| 0-5           | 0.10        | 0.01        |
| 5-10          | 0.06        | 0.10        |
| 10-15         | 0.04        | 0.08        |
| 15-20         | 0.03        | 0.05        |
| 20-25         | 0.02        | 0.03        |
| 25-30         | 0.01        | 0.02        |
| 30-35         | 0.005       | 0.01        |
| 35-40         | 0.002       | 0.005       |
</details>

![](images/933619b6557fe7ee1a7cdb371d9be0e4fb87db6c212764c66a9c4c627c42f439.jpg)

<details>
<summary>line</summary>

| Period (Days) | 128 samples | all samples |
| ------------- | ----------- | ----------- |
| 0             | 0.00        | 0.00        |
| 5             | 0.08        | 0.04        |
| 10            | 0.09        | 0.10        |
| 15            | 0.07        | 0.09        |
| 20            | 0.03        | 0.04        |
| 25            | 0.01        | 0.02        |
| 30            | 0.00        | 0.01        |
| 35            | 0.00        | 0.00        |
| 40            | 0.00        | 0.00        |
</details>

![](images/a2387ff6482e23e1aeaa894d55ea4cbe340b29f52b469682445139821118eb0c.jpg)

<details>
<summary>histogram</summary>

| Total Error Range | Density (1364 samples) | Density (all samples) |
| ----------------- | ---------------------- | --------------------- |
| 0.0 - 0.1         | 6.0                    | 2.8                   |
| 0.1 - 0.2         | 1.0                    | 3.0                   |
| 0.2 - 0.3         | 0.5                    | 1.5                   |
| 0.3 - 0.4         | 0.3                    | 0.8                   |
| 0.4 - 0.5         | 0.2                    | 0.5                   |
| 0.5 - 0.6         | 0.1                    | 0.3                   |
| 0.6 - 0.7         | 0.1                    | 0.2                   |
| 0.7 - 0.8         | 0.1                    | 0.1                   |
| 0.8 - 0.9         | 0.1                    | 0.1                   |
| 0.9 - 1.0         | 0.1                    | 0.1                   |
| 1.0 - 1.1         | 0.1                    | 0.1                   |
| 1.1 - 1.2         | 0.1                    | 0.1                   |
| 1.2 - 1.3         | 0.1                    | 0.1                   |
| 1.3 - 1.4         | 0.1                    | 0.1                   |
| 1.4 - 1.5         | 0.1                    | 0.1                   |
</details>

![](images/09b979932d390c7525c04dac1d28f6afe43fe1cb7135b04276f673f8fb8982b3.jpg)

<details>
<summary>histogram</summary>

| Total Error Range | 58 samples Density | all samples Density |
| ----------------- | ------------------- | -------------------- |
| 0.00 - 0.05       | 3.5                 | 2.8                  |
| 0.05 - 0.10       | 4.2                 | 3.0                  |
| 0.10 - 0.15       | 3.8                 | 2.5                  |
| 0.15 - 0.20       | 2.0                 | 1.5                  |
| 0.20 - 0.25       | 1.8                 | 1.2                  |
| 0.25 - 0.30       | 1.5                 | 1.0                  |
| 0.30 - 0.35       | 1.2                 | 0.8                  |
| 0.35 - 0.40       | 1.0                 | 0.6                  |
| 0.40 - 0.45       | 0.8                 | 0.5                  |
| 0.45 - 0.50       | 0.6                 | 0.4                  |
| 0.50 - 0.55       | 0.5                 | 0.3                  |
| 0.55 - 0.60       | 0.4                 | 0.2                  |
| 0.60 - 0.65       | 0.3                 | 0.1                  |
| 0.65 - 0.70       | 0.2                 | 0.1                  |
| 0.70 - 0.75       | 0.1                 | 0.1                  |
| 0.75 - 0.80       | 0.1                 | 0.1                  |
| 0.80 - 0.85       | 0.1                 | 0.1                  |
| 0.85 - 0.90       | 0.1                 | 0.1                  |
| 0.90 - 0.95       | 0.1                 | 0.1                  |
| 0.95 - 1.00       | 0.1                 | 0.1                  |
| 1.00 - 1.05       | 0.1                 | 0.1                  |
| 1.05 - 1.10       | 0.1                 | 0.1                  |
| 1.10 - 1.15       | 0.1                 | 0.1                  |
| 1.15 - 1.20       | 0.1                 | 0.1                  |
| 1.20 - 1.25       | 0.1                 | 0.1                  |
| 1.25 - 1.30       | 0.1                 | 0.1                  |
| 1.30 - 1.35       | 0.1                 | 0.1                  |
| 1.35 - 1.40       | 0.1                 | 0.1                  |
| 1.40 - 1.45       | 0.1                 | 0.1                  |
| 1.45 - 1.50       | 0.1                 | 0.1                  |
| 1.50 - 1.55       | 0.1                 | 0.1                  |
| 1.55 - 1.60       | 0.1                 | 0.1                  |
| 1.60 - 1.65       | 0.1                 | 0.1                  |
| 1.65 - 1.70       | 0.1                 | 0.1                  |
| 1.70 - 1.75       | 0.1                 | 0.1                  |
</details>

![](images/65de552b51bd35d56b98211f47f435480d269a842c9ce407637a094afd1ed9f5.jpg)

<details>
<summary>line</summary>

| Total Error | Density (120 samples) | Density (all samples) |
| ----------- | --------------------- | --------------------- |
| 0.00        | 4.0                   | 2.5                   |
| 0.25        | 3.0                   | 3.0                   |
| 0.50        | 1.5                   | 1.5                   |
| 0.75        | 0.5                   | 0.5                   |
| 1.00        | 0.2                   | 0.2                   |
| 1.25        | 0.1                   | 0.1                   |
| 1.50        | 0.05                  | 0.05                  |
| 1.75        | 0.02                  | 0.02                  |
</details>

![](images/8e9cf6615e997da38b3c9920fbf339197350f4802eb8e979862a1b7f2c118ace.jpg)

<details>
<summary>histogram</summary>

| Total Error Range | Density (128 samples) | Density (all samples) |
| ----------------- | --------------------- | --------------------- |
| 0.00 - 0.25       | 4.0                   | 3.0                   |
| 0.25 - 0.50       | 3.5                   | 2.5                   |
| 0.50 - 0.75       | 1.0                   | 0.5                   |
| 0.75 - 1.00       | 0.5                   | 0.2                   |
| 1.00 - 1.25       | 0.2                   | 0.1                   |
| 1.25 - 1.50       | 0.1                   | 0.05                  |
| 1.50 - 1.75       | 0.05                  | 0.02                  |
</details>

Figure 14. distributions of the entire sample (red) vs possible contaminants (gray): 4 types of CP/CD contaminant. see text for details. the upper panels show the period distributions and the lower panels show the total error distributions.

![](images/95e44886c4fd3097abccf10b19543fdfd0cc7bee3a543fa931c7d334a689bb5b.jpg)

<details>
<summary>scatter</summary>

| Type | P_orb (Days) | P_orb/P_rot |
|------|--------------|-------------|
| sp   | Various      | Various     |
| ev   | Various      | Various     |
| np   | Various      | Various     |
| ce   | Various      | Various     |
| ot   | Various      | Various     |
| pux  | Various      | Various     |
| pu   | Various      | Various     |
</details>

![](images/2ec0ba621a41735f433b71a81cb497c2e36e3c7226b2d9afaa00204c8620a307.jpg)

<details>
<summary>scatter</summary>

| Category | P_orb (Days) | P_orb / P_ACF |
| -------- | ------------ | ------------- |
| sp       | Various      | Various       |
| ev       | Various      | Various       |
| np       | Various      | Various       |
| ce       | Various      | Various       |
| ot       | Various      | Various       |
| pux      | Various      | Various       |
| pu       | Various      | Various       |
</details>

Figure 15. Eclipsing binaries orbital period vs the ratio between the orbital period and stellar period. Horizontal lines corresponds to $P _ { o r b } / P _ { r o t } = 0 . 5 , 1 , 2$ . The upper panel shows the stellar period predicted by LightPred and the lower panel shows the stellar period predicted by the ACF method. The colors represent stellar classes defined by Lurie et al. (2017) sp - starspots modulation; ev - Ellipsoidal variations; np - no periodic out-of-eclipse variability; ce - Targets where starspot modulations appear to have been mistaken for ellipsoidal variation; ot - another periodic out-of-eclipse variability; pux - possible pulsator; pu - likely pulsator.

peak exists, they took it as the period indicator instead of the regular identification process. If our model takes the first peak of the ACF, those cases might result in a double period difference between our model and Mc-Quillan et al. (2014). In our visual inspection, we did find such examples where our model falsely detects half of the true period. One such example can be seen in Figure 18, lower right panel. Another interesting phenomenon are light curves that show dramatic changes in their periodicity; an example can be seen in Figure 18, lower left panel. It can be seen that the light curve in those cases has two distinct regions, one with high amplitude and one with lower. ACF would detect periods related to the high amplitude region only, regardless of its length, while our model usually gives a sort of ’average’ prediction. It is not clear which result is better in general. In addition, we found a significant amount of cases where we believe our model gives a more accurate prediction. In those cases usually McQuillan et al. (2014) took the second peak of the ACF while the first one seemed to be more accurate. Two such examples can be seen in Figure 18, upper panels. Overall, we found that our model seemed more accurate in 57% of the cases that we examined. We conclude that the only cases where we are sure our model is wrong, are cases with higher second ACF peak. To take this into account, we updated our predictions in the following way: in predictions where ACF shows a higher second peak and the difference between our predictions and McQuillan et al. (2014) predictions is a factor of 2 (±20%), we adapt the predictions of McQuillan et al. (2014). We found 2231 such cases. The right panel of Figure 17 shows the same comparison after updating the predictions.

Lastly, we use our model confidence score to filter predictions. We saw in 5.2.4 that the confidence distributions of synchronized EBs and non-synchronized EBs are very different and suggested a value of 0.86 to potentially separate between spots-induced samples from other types of stars. Taking only predictions with confidence higher than 0.86 removed a total of 21173 samples. Figure 19, left panel, is identical to the upper panel of Figure 15 but after removing samples with $P _ { o r b } > 3$ and samples with confidence $< 0 . 8 6$ . The right panel is the same but shows the orbital period vs the rotation period. In this case, we expect the synchronization line to be on the line with slope one. We see that the synchronization line is much clearer with most of the points below it being removed. We also see that the synchronization line disappears naturally at $P _ { o r b } > 1 0 $ . In addition, we see that all the pulsators were removed and out of 350 Ellipsoids that were in the original sample, only five are left, of which four appear to be synchronized. For spots-modulated samples (sp), out of 285 samples with $P _ { o r b } > 3$ in the original sample, 215 (75%) were left after the filtering. For samples with no apparent periodicity (np), out of 295 samples with $P _ { o r b } > 3$ in the original sample, 134 (45%) were left after the filtering. This supports the assumption that the filtering process successfully removes variables that differ from spotinduced stars, mainly pulsators and ellipsoidals. Lurie et al. (2017) used the ratio of the orbital period and the rotation period to define synchronization, finding that 72% of the samples with orbital periods between 2 and 10 days have $0 . 9 2 < P _ { o r b } / P _ { r o t } < 1 . 2$ . We see that in our results, most of the samples lie on a slightly lower line with 72% of the samples with orbital periods between 3 and 10 days having $0 . 6 9 < P _ { o r b } / P _ { r o t } < 1 . 2$ .

# 5.3. Kepler Catalog

We now briefly summarize all the selection processes we made towards obtaining the final sample. After the selections discussed in 5.2.1 we had predictions for 108096 samples. Next, we did the following:

• We removed pulsators identified by Santos et al. (2021) - 1046 samples.   
• We replaced period values with predictions of Mc-Quillan et al. (2014) in case of higher second peak and double period difference - 2231 samples.   
• We removed samples with period < 3 Days as predicted by ACF or McQuillan et al. (2014), and EBs with orbital period $< 3 \mathrm { D a y s - 3 1 0 6 }$ samples.   
• We removed samples with confidence scores lower than 0.86 - 21173 samples.

This procedure left us with 82771 reliable predictions. An example of the final catalog is presented in Table 8. The full catalog is available in machine-readable format.

# 6. DISCUSSION AND SUMMARY

In this study, we introduced LightPred, a novel deeplearning model designed to extract stellar rotation periods from light curves, demonstrating the potential of deep learning in revolutionizing the analysis of stellar light curve data. By leveraging a dual-branch architecture that incorporates LSTM and Transformer components, LightPred excels at capturing both temporal dependencies and global patterns within light curves.

Our training process, combining simulated and real Kepler data, has allowed LightPred to outperform traditional methods like the Autocorrelation Function in terms of accuracy and robustness, particularly for noisy data. This advancement has led to the generation of the

![](images/a6f2f6ae844cffe381228211657e7ae96d24751fe45e659a650d94c8dec34a62.jpg)

<details>
<summary>scatter</summary>

| P_orb (Days) | P_orb/P_rot | confidence |
| ------------ | ----------- | ---------- |
| 10^-1        | 10^-2       | 0.82       |
| 10^0         | 10^-1       | 0.84       |
| 10^1         | 10^0        | 0.86       |
| 10^2         | 10^1        | 0.88       |
| 10^3         | 10^2        | 0.90       |
</details>

![](images/c79e9964f534fa240672b192da5853b378fff83c9d0c1654563b04a3b0b76dd8.jpg)

<details>
<summary>scatter</summary>

| P_orb (Days) | P_orb/P_rot | Total Error |
| ------------ | ----------- | ----------- |
| 10^-1        | 10^-2       | 0.2         |
| 10^0         | 10^-1       | 0.4         |
| 10^1         | 10^0        | 0.8         |
| 10^2         | 10^1        | 1.2         |
| 10^3         | 10^2        | 1.6         |
</details>

![](images/7f1ae90474ac1832c9df0f113d4fecc6d9c805a64208706f2a456a240e7ff79d.jpg)

<details>
<summary>scatter</summary>

| P_orb (Days) | P_orb / P_rot |
| ------------ | ------------- |
| 10^0         | 10^0          |
| 10^1         | 10^0          |
| 10^2         | 10^1          |
| 10^3         | 10^2          |
</details>

![](images/8fd46b21ee1063526918a961662ebe5903105a608c54db0ada6db19a89aa8b6f.jpg)

<details>
<summary>histogram</summary>

| Period (Days) | average 0.88 | average 0.86 | average 0.93 |
| ------------- | ------------ | ------------ | ------------ |
| 0.82          | 1            | 6            | 0            |
| 0.84          | 12           | 30           | 2            |
| 0.86          | 25           | 20           | 4            |
| 0.88          | 8            | 4            | 4            |
| 0.90          | 5            | 5            | 6            |
| 0.92          | 5            | 3            | 12           |
| 0.94          | 6            | 2            | 24           |
| 0.96          | 3            | 1            | 25           |
</details>

Figure 16. Eclipsing binaries orbital period vs the ratio between orbital period and rotational period. Horizontal lines corresponds to $P _ { o r b } / P _ { r o t } = 0 . 5 , 1 , 2 .$ In the upper left panel, colors correspond to the model’s confidence. In the upper right panel, colors correspond to the total error. The lower left panel shows the three classes defined in 5.2.4. The lower right panel shows the histograms of confidence for the classes on the left panel.

![](images/252a659e1894e1741c8d9b4705acf814f98be56d97e94f3df13fcafb908b4097.jpg)

<details>
<summary>scatter</summary>

| McQ14 Period (Days) | LightPred Period (Days) | w     |
| ------------------- | ----------------------- | ----- |
| 0                   | 0                       | 0.25  |
| 10                  | 10                      | 0.30  |
| 20                  | 20                      | 0.35  |
| 30                  | 30                      | 0.40  |
| 40                  | 40                      | 0.45  |
| 50                  | 50                      | 0.50  |
| 60                  | 60                      | 0.55  |
</details>

![](images/857f5ab0235856b0112b8be082f4c02cd639aba07fa50adb720c8d9420e13084.jpg)

<details>
<summary>scatter</summary>

| McQ14 Period (Days) | LightPred Period (Days) | w     |
| ------------------- | ----------------------- | ----- |
| 0                   | 0                       | 0.25  |
| 10                  | 10                      | 0.30  |
| 20                  | 20                      | 0.35  |
| 30                  | 30                      | 0.40  |
| 40                  | 40                      | 0.45  |
| 50                  | 50                      | 0.50  |
| 60                  | 60                      | 0.55  |
</details>

Figure 17. Comparison of our predictions and McQuillan et al. (2014) predictions as seen in figure 9. Colors represent the w parameter as given in McQuillan et al. (2014). The left panel shows the results for $P _ { M c Q 1 4 } > 3$ and the right panel shows the results after the full selection process as described in 5.3.

![](images/85a9ce5679475b19ca264804c80df2cf56cbab5d1ba90e46655715dd4f4ec435.jpg)

<details>
<summary>line</summary>

| Metric | Value |
|--------|-------|
| KIC | 10007265 |
| P_growthal | 11.06 |
| P_RCL14 | 21.76 |
| Confidence | 0.948 |
| KIC | 1865325 |
| P_growthal | 18.62 |
| P_RCL14 | 40.12 |
| Confidence | 0.92 |
</details>

![](images/9c5733389e1e6a9763cb7fa6a94e668c7919e653883066620d22bb5db58f9139.jpg)

<details>
<summary>line</summary>

| Time (Days) | Normalized Flux (%) | Lags (Days) | KIC       | P_ighred | P_reCL14 |
|-------------|---------------------|-------------|-----------|----------|----------|
| 0           | ~0.9960             | 0           | 10015672  | 15.77    | 40.92    |
| 250         | ~0.9955             | 25          | 10331929  | 13.82    | 27.59    |
| 500         | ~0.9958             | 50          | 10331929  | 13.82    | 27.59    |
| 750         | ~0.9960             | 75          | 10331929  | 13.82    | 27.59    |
| 1000        | ~0.9955             | 100         | 10331929  | 13.82    | 27.59    |
| 1250        | ~0.9960             | 125         | 10331929  | 13.82    | 27.59    |
| 1500        | ~0.9958             | 150         | 10331929  | 13.82    | 27.59    |
| 1750        | ~0.9960             | 175         | 10331929  | 13.82    | 27.59    |
| 2000        | ~0.9955             | 200         | 10331929  | 13.82    | 27.59    |
| 2250        | ~0.9960             | 225         | 10331929  | 13.82    | 27.59    |
| 2500        | ~0.9958             | 250         | 10331929  | 13.82    | 27.59    |
| 2750        | ~0.9960             |            |           |          |          |
| 3000        | ~0.9955             |            |           |          |          |
| Note: The data is already in CSV format as it is not available from the original table structure.
</details>

Figure 18. Four light curve examples with inconsistency between our model and McQuillan et al. (2014). For each sample, the lower panel shows the full light curve and the upper left panel shows a zoom-in slice of 166 days. The upper right panel shows the ACF of the light curve. The light curve is averaged over a window of one day. The titles show the Kepler KIC, lightPred period, McQ14 period from McQuillan et al. (2014), and the confidence of the model.

![](images/28d33a9b39c9a7ba9ec12d65f3300d4cc0fd798732ddec472ec7ccef48fb68da.jpg)

<details>
<summary>scatter</summary>

| Type | P_orb (Days) | P_orb/P_rot |
|------|--------------|-------------|
| sp   | 10^1         | 10^0        |
| sp   | 10^2         | 10^1        |
| sp   | 10^3         | 10^2        |
| np   | 10^1         | 10^0        |
| np   | 10^2         | 10^1        |
| np   | 10^3         | 10^2        |
| ev   | 10^1         | 10^0        |
| ev   | 10^2         | 10^1        |
| ev   | 10^3         | 10^2        |
| ot   | 10^1         | 10^0        |
| ot   | 10^2         | 10^1        |
| ot   | 10^3         | 10^2        |
</details>

![](images/d51adf881f70ccd7108fccb979c55495d3a224244eac57cbaca97e22c72c0407.jpg)

<details>
<summary>scatter</summary>

| Group | P_orb (Days) | P_rot |
|-------|--------------|-------|
| sp    | 0            | 0     |
| sp    | 5            | 5     |
| sp    | 10           | 10    |
| sp    | 15           | 15    |
| sp    | 20           | 20    |
| sp    | 25           | 25    |
| sp    | 30           | 30    |
| sp    | 35           | 35    |
| sp    | 40           | 40    |
| sp    | 45           | 45    |
| sp    | 50           | 50    |
| np    | 0            | 0     |
| np    | 5            | 5     |
| np    | 10           | 10    |
| np    | 15           | 15    |
| np    | 20           | 20    |
| np    | 25           | 25    |
| np    | 30           | 30    |
| np    | 35           | 35    |
| np    | 40           | 40    |
| np    | 45           | 45    |
| np    | 50           | 50    |
| ev    | 0            | 0     |
| ev    | 5            | 5     |
| ev    | 10           | 10    |
| ev    | 15           | 15    |
| ev    | 20           | 20    |
| ev    | 25           | 25    |
| ev    | 30           | 30    |
| ev    | 35           | 35    |
| ev    | 40           | 40    |
| ev    | 45           | 45    |
| ev    | 50           | 50    |
| ot    | 0            | 0     |
| ot    | 5            | 5     |
| ot    | 10           | 10    |
| ot    | 15           | 15    |
| ot    | 20           | 20    |
| ot    | 25           | 25    |
| ot    | 30           | 30    |
| ot    | 35           | 35    |
| ot    | 40           | 40    |
| ot    | 45           | 45    |
| ot    | 50           | 50    |
</details>

Figure 19. Left panel: Eclipsing Binaries orbital period vs the ratio between orbital and rotation period. Right panel: Eclipsing Binaries orbital period vs rotation period. Both panels show the results after filtering low confidence and fast rotators as explained in 5.2.5. Colors represents classes defined by Lurie et al. (2017) and explained in Figure 15. Horizontal lines corresponds to $P _ { o r b } / P _ { r o t } = 0 . 5 , 1 , 2$ .

<table><tr><td>KID</td><td>Predicted Period</td><td>Observational Error (%)</td><td>Model Confidence</td><td>Total Error</td><td>Flag</td></tr><tr><td>757450</td><td>19.67</td><td>0.049</td><td>0.942</td><td>0.052</td><td>1</td></tr><tr><td>891901</td><td>20.059</td><td>0.188</td><td>0.921</td><td>0.204</td><td></td></tr><tr><td>891916</td><td>5.846</td><td>0.072</td><td>0.954</td><td>0.075</td><td></td></tr><tr><td>892195</td><td>11.329</td><td>0.187</td><td>0.864</td><td>0.216</td><td>3</td></tr><tr><td>892203</td><td>6.693</td><td>0.29</td><td>0.88</td><td>0.329</td><td>3</td></tr><tr><td>892675</td><td>16.399</td><td>0.103</td><td>0.898</td><td>0.114</td><td>3</td></tr><tr><td>892678</td><td>16.474</td><td>0.308</td><td>0.889</td><td>0.347</td><td>3</td></tr><tr><td>892713</td><td>5.601</td><td>0.192</td><td>0.906</td><td>0.212</td><td></td></tr><tr><td>892718</td><td>11.621</td><td>0.131</td><td>0.893</td><td>0.147</td><td>3</td></tr><tr><td>892772</td><td>10.528</td><td>0.229</td><td>0.871</td><td>0.263</td><td>3</td></tr><tr><td>892832</td><td>12.207</td><td>0.194</td><td>0.867</td><td>0.224</td><td>3</td></tr><tr><td>892834</td><td>13.578</td><td>0.023</td><td>0.959</td><td>0.024</td><td></td></tr><tr><td>892882</td><td>20.415</td><td>0.052</td><td>0.95</td><td>0.055</td><td></td></tr><tr><td>892911</td><td>12.094</td><td>0.305</td><td>0.866</td><td>0.352</td><td>3</td></tr><tr><td>892986</td><td>9.397</td><td>0.21</td><td>0.867</td><td>0.242</td><td>3</td></tr><tr><td>893004</td><td>11.916</td><td>0.228</td><td>0.867</td><td>0.262</td><td>3</td></tr><tr><td>893033</td><td>16.317</td><td>0.335</td><td>0.904</td><td>0.371</td><td></td></tr><tr><td>893165</td><td>8.808</td><td>0.098</td><td>0.864</td><td>0.113</td><td>3</td></tr><tr><td>893209</td><td>5.642</td><td>0.174</td><td>0.934</td><td>0.186</td><td></td></tr></table>

Table 8. An example of the resulting predicted periods on Kepler samples. The Flag column consists of the following values: 1 - confirmed or candidate planet host star; 2 - Eclipsing Binary; 3 - samples where ACF algorithm failed to predict period on one or more segments. The full table is available in machine-readable format. (Kamai & Perets 2024)

largest and most accurate catalog of stellar rotation periods for main-sequence stars to date, comprising over 80000 stars.

Furthermore, our analysis of eclipsing binaries confirms tidal synchronization in systems with orbital periods shorter than 10 days, further validating our approach. Another important result that emerged from the analysis is the ability to cluster the predictions to different stellar types based on the confidence level of the model. Using this property, we were able to filter out false positive predictions.

A comparison between our results and ACF results from McQuillan et al. (2014), reveals a subgroup of slow rotating stars that shows non-consistent predictions among the methods. Those samples have challenging periodicity patterns and can be further analyzed.

Our findings illustrate the potential of a data-driven approach for the analysis of stellar lightcurves and pave the way for future research, including the development of improved spot models for simulation data, particularly to account for the behavior of fast-rotating young stars. We also anticipate further exploration of stellar inclination predictions, a more complex task that is crucial for a comprehensive understanding of stellar systems.

# 7. ACKNOWLEDGMENTS

We would like to thank the anonymous referee for helpful comments that improved the manuscript. We would like to acknowledge the support from the Minerva Center for life under extreme planetary conditions.

# REFERENCES

Abdul-Masih, M., Prˇsa, A., Conroy, K., et al. 2016, AJ,

151, 101, doi: 10.3847/0004-6256/151/4/101

Aigrain, S., Llama, J., Ceillier, T., et al. 2015, MNRAS,

450, 3211, doi: 10.1093/mnras/stv853

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M.

2019, in Proceedings of the 25th ACM SIGKDD

International Conference on Knowledge Discovery and

Data Mining

Albrecht, S. H., Dawson, R. I., & Winn, J. N. 2022, PASP,

134, 082001, doi: 10.1088/1538-3873/ac6c09

Angus, R., Morton, T., Aigrain, S., Foreman-Mackey, D., & Rajpaul, V. 2017, Monthly Notices of the Royal Astronomical Society, 474, 2094–2108, doi: 10.1093/mnras/stx2109   
Angus, R., Morton, T. D., Foreman-Mackey, D., et al. 2019, AJ, 158, 173, doi: 10.3847/1538-3881/ab3c53   
Barnes, S. A. 2003, ApJ, 586, 464, doi: 10.1086/367639 —. 2007, ApJ, 669, 1167, doi: 10.1086/519295 —. 2010, ApJ, 722, 222, doi: 10.1088/0004-637X/722/1/222   
Basri, G., & Shah, R. 2020, ApJ, 901, 14, doi: 10.3847/1538-4357/abae5d   
Basri, G., Walkowicz, L. M., Batalha, N., et al. 2011, AJ, 141, 20, doi: 10.1088/0004-6256/141/1/20   
Bazot, M., Nielsen, M. B., Mary, D., et al. 2018, A&A, 619, L9, doi: 10.1051/0004-6361/201834251   
Bengio, Y., Simard, P., & Frasconi, P. 1994, IEEE Transactions on Neural Networks, 5, 157, doi: 10.1109/72.279181   
Benk˝o, J. M., Kolenberg, K., Szab´o, R., et al. 2010, MNRAS, 409, 1585, doi: 10.1111/j.1365-2966.2010.17401.x   
Berdyugina, S. V., & Usoskin, I. G. 2003, A&A, 405, 1121, doi: 10.1051/0004-6361:20030748   
Berger, T. A., Huber, D., van Saders, J. L., et al. 2020, AJ, 159, 280, doi: 10.3847/1538-3881/159/6/280   
Blancato, K., Ness, M. K., Huber, D., Lu, Y., & Angus, R. 2022, ApJ, 933, 241, doi: 10.3847/1538-4357/ac7563   
Bouma, L. G., Palumbo, E. K., & Hillenbrand, L. A. 2023, ApJL, 947, L3, doi: 10.3847/2041-8213/acc589   
Bradley, P. A., Guzik, J. A., Miles, L. F., et al. 2015, AJ, 149, 68, doi: 10.1088/0004-6256/149/2/68   
Breton, S. N., Santos, A. R. G., Bugnet, L., et al. 2021, A&A, 647, A125, doi: 10.1051/0004-6361/202039947   
Bryson, S. T., Jenkins, J. M., Gilliland, R. L., et al. 2013, PASP, 125, 889, doi: 10.1086/671767   
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. 2020, A Simple Framework for Contrastive Learning of Visual Representations. https://arxiv.org/abs/2002.05709   
Chen, X., & He, K. 2020. https://arxiv.org/abs/2011.10566   
Ciardi, D. R., von Braun, K., Bryden, G., et al. 2011, AJ, 141, 108, doi: 10.1088/0004-6256/141/4/108   
Claytor, Z. R., van Saders, J. L., Cao, L., et al. 2024, The Astrophysical Journal, 962, 47, doi: 10.3847/1538-4357/ad159a   
Claytor, Z. R., van Saders, J. L., Llama, J., et al. 2022, ApJ, 927, 219, doi: 10.3847/1538-4357/ac498f   
Col´on, K. D., Morehead, R. C., & Ford, E. B. 2015, MNRAS, 452, 3001, doi: 10.1093/mnras/stv1382   
Coughlin, J. L., Mullally, F., Thompson, S. E., et al. 2016, ApJS, 224, 12, doi: 10.3847/0067-0049/224/1/12

D´esert, J.-M., Charbonneau, D., Torres, G., et al. 2015, ApJ, 804, 59, doi: 10.1088/0004-637X/804/1/59   
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. 2019, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805   
Forr´o, A., Szab´o, R., B´odi, A., & Cs´asz´ar, K. 2022, ApJS, 260, 20, doi: 10.3847/1538-4365/ac5e9e   
Garc´ıa, R. A., Ceillier, T., Salabert, D., et al. 2014, A&A, 572, A34, doi: 10.1051/0004-6361/201423888   
Garc´ıa, R. A., Ceillier, T., Salabert, D., et al. 2014, Astronomy &amp; Astrophysics, 572, A34, doi: 10.1051/0004-6361/201423888   
Gidaris, S., Singh, P., & Komodakis, N. 2018. https://arxiv.org/abs/1803.07728   
Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., et al. 2014. https://arxiv.org/abs/1406.2661   
Grill, J.-B., Strub, F., Altch´e, F., et al. 2020, Bootstrap your own latent: A new approach to self-supervised Learning. https://arxiv.org/abs/2006.07733   
Grossmann, A., & Morlet, J. 1984, SIAM Journal on Mathematical Analysis, 15, 723, doi: 10.1137/0515056   
Gulati, A., Qin, J., Chiu, C.-C., et al. 2020, Conformer: Convolution-augmented Transformer for Speech Recognition. https://arxiv.org/abs/2005.08100   
Hathaway, D. H. 2011, SoPh, 273, 221, doi: 10.1007/s11207-011-9837-z   
Hendrycks, D., & Gimpel, K. 2016, arXiv e-prints, arXiv:1606.08415, doi: 10.48550/arXiv.1606.08415   
Hochreiter, S., & Schmidhuber, J. 1997, Neural Comput., 9, 1735–1780, doi: 10.1162/neco.1997.9.8.1735   
Huertas-Company, M., Sarmiento, R., & Knapen, J. H. 2023, RAS Techniques and Instruments, 2, 441, doi: 10.1093/rasti/rzad028   
Hut, P. 1981, A&A, 99, 126   
Hyndman, R. J., & Athanasopoulos, G. 2018, Forecasting: principles and practice, OText. https://otexts.com/fpp2/stationarity.html   
Ismail Fawaz, H., Forestier, G., Weber, J., Idoumghar, L., & Muller, P.-A. 2018, arXiv e-prints, arXiv:1809.04356, doi: 10.48550/arXiv.1809.04356   
Kamai, I., & Perets, H. 2024, Rotation Period for 83022 Kepler Stars: A Deep Learning Approach, Zenodo, doi: 10.5281/zenodo.13774785   
Kingma, D. P., & Welling, M. 2022. https://arxiv.org/abs/1312.6114   
Kirk, B., Conroy, K., Prˇsa, A., et al. 2016, AJ, 151, 68, doi: 10.3847/0004-6256/151/3/68

Lara-Ben´ıtez, P., Carranza-Garc´ıa, M., & Riquelme, J. C. 2021, arXiv e-prints, arXiv:2103.12057, doi: 10.48550/arXiv.2103.12057   
Li, G., Bedding, T. R., Murphy, S. J., et al. 2019a, MNRAS, 482, 1757, doi: 10.1093/mnras/sty2743   
Li, G., Van Reeth, T., Bedding, T. R., Murphy, S. J., & Antoci, V. 2019b, MNRAS, 487, 782, doi: 10.1093/mnras/stz1171   
Lomb, N. R. 1976, Ap&SS, 39, 447, doi: 10.1007/BF00648343   
Loshchilov, I., & Hutter, F. 2017, arXiv e-prints, arXiv:1711.05101, doi: 10.48550/arXiv.1711.05101   
Lu, Y., Angus, R., Ag¨ueros, M. A., et al. 2020, AJ, 160, 168, doi: 10.3847/1538-3881/abada4   
Luger, R., Foreman-Mackey, D., Hedges, C., & Hogg, D. W. 2021, AJ, 162, 123, doi: 10.3847/1538-3881/abfdb8   
Lurie, J. C., Vyhmeister, K., Hawley, S. L., et al. 2017, The Astronomical Journal, 154, 250, doi: 10.3847/1538-3881/aa974d   
Mackay, D. H., Jardine, M., Collier Cameron, A., Donati, J. F., & Hussain, G. A. J. 2004, MNRAS, 354, 737, doi: 10.1111/j.1365-2966.2004.08233.x   
Mamajek, E. E., & Hillenbrand, L. A. 2008, ApJ, 687, 1264, doi: 10.1086/591785   
Mathur, S., Garc´ıa, R. A., Ballot, J., et al. 2014, A&A, 562, A124, doi: 10.1051/0004-6361/201322707   
Mazeh, T., Perets, H. B., McQuillan, A., & Goldstein, E. S. 2015, ApJ, 801, 3, doi: 10.1088/0004-637X/801/1/3   
McQuillan, A., Mazeh, T., & Aigrain, S. 2013, ApJL, 775, L11, doi: 10.1088/2041-8205/775/1/L11 —. 2014, ApJS, 211, 24, doi: 10.1088/0067-0049/211/2/24   
Morris, B. M. 2020, The Astrophysical Journal, 893, 67, doi: 10.3847/1538-4357/ab79a0   
Morvan, M., Nikolaou, N., Yip, K., & Waldmann, I. 2022, in Machine Learning for Astrophysics, 11, doi: 10.48550/arXiv.2207.02777   
Mullally, F., Coughlin, J. L., Thompson, S. E., et al. 2016, PASP, 128, 074502, doi: 10.1088/1538-3873/128/965/074502   
Murphy, S. J., Hey, D., Van Reeth, T., & Bedding, T. R. 2019, Monthly Notices of the Royal Astronomical Society, 485, 2380, doi: 10.1093/mnras/stz590   
Nemec, J. M., Cohen, J. G., Ripepi, V., et al. 2013, ApJ, 773, 181, doi: 10.1088/0004-637X/773/2/181   
Nemec, J. M., Smolec, R., Benk˝o, J. M., et al. 2011, Monthly Notices of the Royal Astronomical Society, 417, 1022, doi: 10.1111/j.1365-2966.2011.19317.x   
Netto, Y., & Valio, A. 2020, A&A, 635, A78, doi: 10.1051/0004-6361/201936219

Nielsen, M. B., Gizon, L., Schunker, H., & Karoff, C. 2013, A&A, 557, L10, doi: 10.1051/0004-6361/201321912   
Pan, J.-S., Ting, Y.-S., & Yu, J. 2024, MNRAS, 528, 5890, doi: 10.1093/mnras/stae068   
Reinhold, T., Cameron, R. H., & Gizon, L. 2017, Astronomy &amp; Astrophysics, 603, A52, doi: 10.1051/0004-6361/201730599   
Reinhold, T., Reiners, A., & Basri, G. 2013, A&A, 560, A4, doi: 10.1051/0004-6361/201321970   
Reinhold, T., Shapiro, A. I., Solanki, S. K., & Basri, G. 2023, A&A, 678, A24, doi: 10.1051/0004-6361/202346789   
Salabert, D., Garc´ıa, R. A., Beck, P. G., et al. 2016, A&A, 596, A31, doi: 10.1051/0004-6361/201628583   
Santos, A. R. G., Breton, S. N., Mathur, S., & Garc´ıa, R. A. 2021, ApJS, 255, 17, doi: 10.3847/1538-4365/ac033f   
Santos, A. R. G., Garc´ıa, R. A., Mathur, S., et al. 2019, The Astrophysical Journal Supplement Series, 244, 21, doi: 10.3847/1538-4365/ab3b56   
Scargle, J. D. 1982, ApJ, 263, 835, doi: 10.1086/160554   
Shapiro, A. I., Amazo-G´omez, E. M., Krivova, N. A., & Solanki, S. K. 2020, A&A, 633, A32, doi: 10.1051/0004-6361/201936018   
Simonian, G. V. A., Pinsonneault, M. H., & Terndrup, D. M. 2020, VizieR Online Data Catalog: Kepler rapid rotators and Ks-band excesses (Simonian+, 2019), VizieR On-line Data Catalog: J/ApJ/871/174. Originally published in: 2019ApJ...871..174S, doi: 10.26093/cds/vizier.18710174   
Skumanich, A. 1972, ApJ, 171, 565, doi: 10.1086/151310   
Smith, J. C., Stumpe, M. C., Van Cleve, J. E., et al. 2012, PASP, 124, 1000, doi: 10.1086/667697   
STScI. 2016, Kepler LC, Q0-Q17, STScI/MAST, doi: 10.17909/T9488N   
Stumpe, M. C., Smith, J. C., Van Cleve, J. E., et al. 2012, PASP, 124, 985, doi: 10.1086/667698   
Su, J., Lu, Y., Pan, S., et al. 2023. https://arxiv.org/abs/2104.09864   
Thomas, A. E. L., Chaplin, W. J., Davies, G. R., et al.2019, MNRAS, 485, 3857, doi: 10.1093/mnras/stz672  
Torrence, C., & Compo, G. P. 1998, Bulletin of the American Meteorological Society, 79, 61 , doi: 10.1175/1520-0477(1998)079⟨0061: APGTWA⟩2.0.CO;2   
Torres, G., Fressin, F., Batalha, N. M., et al. 2011, ApJ, 727, 24, doi: 10.1088/0004-637X/727/1/24   
Uytterhoeven, K., Moya, A., Grigahc\`ene, A., et al. 2011, A&A, 534, A125, doi: 10.1051/0004-6361/201117368   
Van Reeth, T., Mombarg, J. S. G., Mathis, S., et al. 2018, A&A, 618, A24, doi: 10.1051/0004-6361/201832718

Walkowicz, L. M., Basri, G., & Valenti, J. A. 2013, ApJS, 205, 17, doi: 10.1088/0067-0049/205/2/17   
Wu, H., Xu, J., Wang, J., & Long, M. 2021, CoRR, abs/2106.13008   
Zhang, K., Wen, Q., Zhang, C., et al. 2024. https://arxiv.org/abs/2306.10125   
Zhou, H., Zhang, S., Peng, J., et al. 2021, Proceedings of the AAAI Conference on Artificial Intelligence, 35, 11106, doi: 10.1609/aaai.v35i12.17325   
Ziegler, C., Law, N. M., Morton, T., et al. 2017, The   
Astronomical Journal, 153, 66,   
doi: 10.3847/1538-3881/153/2/66

# APPENDIX

# A. PRE-PROCESSING EXAMPLES

Examples of simulated light curves and the different preprocessing stages are shown in the figures.

![](images/f3679351b00978d8102d8ad7c9eba4b2d59dd6fd1af693125f7069ef0d74da1d.jpg)

<details>
<summary>line</summary>

| Time (Days) | Relative Flux (%) |
| ----------- | ----------------- |
| 0           | 0.996             |
| 200         | 0.994             |
| 400         | 0.995             |
| 600         | 0.993             |
| 800         | 0.991             |
</details>

![](images/58c9d90a09efa008d93088d252b7975c599b097dccf3f0adaba3c666a649bcda.jpg)

<details>
<summary>line</summary>

| Time (Days) | Value  |
|-------------|--------|
| 0           | 0.997  |
| 100         | 0.996  |
| 200         | 0.995  |
| 300         | 0.996  |
| 400         | 0.997  |
| 500         | 1.002  |
| 600         | 0.996  |
| 700         | 0.994  |
| 800         | 0.995  |
</details>

![](images/80bff8b7eefe10f26d0cb8b005f8ab74039c37669788724663930a4a922fc386.jpg)

<details>
<summary>line</summary>

| Time (Days) | Averaged |
| ----------- | -------- |
| 0           | 0.996    |
| 100         | 0.995    |
| 200         | 0.994    |
| 300         | 0.995    |
| 400         | 0.996    |
| 500         | 0.997    |
| 600         | 0.995    |
| 700         | 0.992    |
| 800         | 0.994    |
</details>

![](images/2ec5f681bfc0c75364b147ea40046eb3fb006d262eb698584d46a14506126f39.jpg)

<details>
<summary>line</summary>

| Time (Days) | Value |
|-------------|-------|
| 0           | 0     |
| 100         | 5     |
| 200         | 0     |
| 300         | 25    |
| 400         | 10    |
| 500         | 30    |
| 600         | 25    |
| 700         | 15    |
| 800         | 5     |
</details>

![](images/3cbbc5eadc3297634dcf1aae8cdf9cec280f1bde7527a390cdea191fdb2d3c67.jpg)

<details>
<summary>line</summary>

| Time Lag (Days) | Diff-ACF |
| --------------- | -------- |
| 0               | 0.0      |
| 100             | 0.0      |
| 200             | 0.0      |
| 300             | 0.0      |
| 400             | 0.0      |
| 500             | 0.0      |
| 600             | 0.0      |
| 700             | 0.0      |
</details>

![](images/399d286052796306d807c0e417c8b34e9c2847173223d5ccb7f42fc42c0e87cb.jpg)

<details>
<summary>line</summary>

| Time (Days) | Normalized |
| ----------- | ---------- |
| 100         | 5          |
| 150         | 8          |
| 200         | 2           |
| 250         | -25        |
| 300         | 22         |
| 350         | 15         |
| 400         | 10         |
| 450         | -15        |
| 500         | 30         |
| 550         | -30        |
| 600         | 25         |
| 650         | -25        |
| 700         | 18         |
| 750         | -20        |
| 800         | 5          |
</details>

![](images/0ce1a98306ab8b26af38a4f4fcc0a34d29605f2a50cab691372a017fffae2221.jpg)

<details>
<summary>line</summary>

| Time (Days) | Relative Flux (%) |
| ----------- | ----------------- |
| 0           | 1.0000            |
| 200         | ~0.9998           |
| 400         | ~0.9992           |
| 600         | ~0.9985           |
| 800         | ~0.9975           |
</details>

![](images/251646c9769d0f1dd393596a425bf7a4d0531182461bb37acd03606f866a8ad0.jpg)

<details>
<summary>line</summary>

| Time (Days) | Value  |
|-------------|--------|
| 0           | 1.000  |
| 100         | 1.002  |
| 200         | 1.001  |
| 300         | 1.003  |
| 400         | 1.012  |
| 500         | 1.004  |
| 600         | 1.017  |
| 700         | 1.006  |
| 800         | 1.002  |
</details>

![](images/9961aae9584c5a7c49a82a50670855a4bf261f2bbaa356ce46e9afa6be3d0632.jpg)

<details>
<summary>line</summary>

| Time (Days) | Value  |
|-------------|--------|
| 0           | 1.000  |
| 200         | 0.998  |
| 400         | 0.999  |
| 600         | 0.997  |
| 800         | 0.998  |
</details>

![](images/68231fc2ff49216a6c968c6821019114f2633bf3fee7ad6bb7607af17c28c56e.jpg)

<details>
<summary>line</summary>

| Time (Days) | Value |
|-------------|-------|
| 0           | 0     |
| 100         | 5     |
| 200         | 0     |
| 300         | 15    |
| 400         | 18    |
| 500         | 5     |
| 600         | 25    |
| 700         | 10    |
| 800         | 5     |
</details>

![](images/b18938b643f16127e6abcda6f8c8de57280df6623ac23bc6051cfae19e2088cb.jpg)

<details>
<summary>line</summary>

| Time Lag (Days) | Diff-ACF |
| --------------- | -------- |
| 0               | 0.0      |
| 100             | 0.0      |
| 200             | 0.0      |
| 300             | 0.0      |
| 400             | 0.0      |
| 500             | 0.0      |
| 600             | 0.0      |
| 700             | 0.0      |
</details>

![](images/a402275e81433ac575985e9cfcb5e3084c8d41f41cbe3c036091ebc8a96553d6.jpg)

<details>
<summary>line</summary>

| Time (Days) | Normalized |
| ----------- | ---------- |
| 0           | 0          |
| 100         | 0          |
| 200         | 0          |
| 300         | 0          |
| 400         | 0          |
| 500         | 0          |
| 600         | 0          |
| 700         | 0          |
| 800         | 0          |
</details>

![](images/33bb76e435c502d895df38a3a42ec8b514e6873006b0c89ba7d2403403a5d545.jpg)

<details>
<summary>line</summary>

| Time (Days) | Relative Flux (%) |
| ----------- | ----------------- |
| 300         | 0.995             |
| 400         | 0.985             |
| 500         | 0.980             |
| 600         | 0.975             |
| 700         | 0.972             |
| 800         | 0.970             |
| 900         | 0.975             |
| 1000        | 0.985             |
</details>

![](images/f0caeaa2c0b66f2f51bc8709eff810b2151566c4c69065bffaa857443032cad8.jpg)

<details>
<summary>line</summary>

| Time (Days) | Value  |
| ----------- | ------ |
| 0           | 0.995  |
| 400         | 0.985  |
| 600         | 0.975  |
| 800         | 0.970  |
| 1000        | 0.985  |
</details>

![](images/4c40fd7477341949238a7535d71b38183a0f99d9f724310b8e9a1b0b6f65c041.jpg)

<details>
<summary>line</summary>

| Time (Days) | Averaged |
| ----------- | -------- |
| 0           | 0.995    |
| 400         | 0.985    |
| 600         | 0.980    |
| 800         | 0.975    |
| 1000        | 0.985    |
</details>

![](images/431968190eea68b49bc81e96172794c024a1704fd149d909bd2fbdcba4791ddb.jpg)

<details>
<summary>line</summary>

| Time (Days) | Diff |
| ----------- | ---- |
| 400         | 0    |
| 500         | 30   |
| 600         | -30  |
| 700         | 0    |
| 800         | 0    |
| 900         | 10   |
| 1000        | 0    |
</details>

![](images/65602296dae97de4f553b35a22884d7717a69cc2ce70383fa1f2b8215739d80d.jpg)

<details>
<summary>line</summary>

| Time Lag (Days) | ACF     |
| --------------- | ------- |
| 0               | 0.5     |
| 50              | -0.2    |
| 100             | 0.2     |
| 150             | 0.1     |
| 200             | 0.15    |
| 250             | 0.05    |
| 300             | 0.1     |
| 350             | 0.05    |
| 400             | 0.08    |
| 450             | 0.03    |
| 500             | 0.02    |
| 550             | 0.01    |
| 600             | 0.01    |
| 650             | 0.01    |
</details>

![](images/0ff81fa54b25b80d447ffdd5d084cdcadf77690f1409031b910f4d8310952044.jpg)

<details>
<summary>line</summary>

| Time (Days) | Normalized |
| ----------- | ---------- |
| 300         | 0          |
| 400         | 0          |
| 500         | 0          |
| 600         | 35         |
| 700         | 0          |
| 800         | 0          |
| 900         | 10         |
| 1000        | 0          |
</details>

Examples of simulated light curve and the different preprocessing stages.

# B. LIGHTPRED MODEL ARCHITECTURE

LightPred utilizes a dual-branch architecture to analyze light curve data comprehensively. The detailed architecture and its schematics are described and shown below, followed by a table of the model parameters.

# • Temporal Branch:

– Input: The Diff-ACF (differenced autocorrelation function) of the light curve. This pre-processed input highlights short-term variations and periodic patterns related to stellar rotation.

– Convolutional Block: A 1D convolutional layer extracts local features and patterns from the Diff-ACF. Max pooling layer downsample the features to reduce dimensionality and computational cost while preserving essential information.

Bi-directional LSTM Block: Long Short-Term Memory (LSTM) layers, arranged in a bi-directional fashion, process the convolutional features to capture both forward and backward temporal dependencies. LSTMs excel at modeling sequential data and remembering information over long durations, making them ideal for identifying the evolving nature of starspot patterns and periodicities in light curves.

– Self-Attention Block: A non-learnable self-attention mechanism refines the LSTM’s final hidden states by weighting the importance of different time steps based on their relevance to the rotation period. This helps focus the model on the most informative segments of the Diff-ACF.

# • Global Branch:

– Input: The Diff-Lc (differenced light curve). This input provides a broader view of the overall flux variations and long-term trends in the light curve.

– AstroConformer Block: This block, a modified version of the Conformer architecture, combines multihead self-attention with convolutional layers.

∗ Multi-head self-attention: Captures long-range dependencies and global context by allowing the model to weigh the importance of different time steps in relation to each other. This is crucial for identifying periodicities that may not be as apparent in the local features extracted by the temporal branch.

∗ Convolutional layers: Extract local features from the light curve, complementing the global context captured by self-attention.

# • Prediction Head:

– Concatenation: Features from both branches are concatenated, combining temporal and global representations.

– Fully Connected Layers: Two dense layers with GELU activation functions and dropout regularization process the concatenated features. This allows the model to learn complex relationships between the features and the target outputs.

– Output: The model produces four values:

∗ Stellar rotation period   
∗ Stellar inclination  
∗ Confidence score for period prediction   
∗ Confidence score for inclination prediction

Kamai & Perets   
![](images/1d56b7508063794c8818870daec668b7b6214a8cbf061d59dd0d1d2658f36eb0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input (B,T)"] --> B["1d Conv"]
    B --> C["(B,C,T)"]
    C --> D["BatchNorm"]
    D --> E["(B,C,T)"]
    E --> F["GELU"]
    F --> G["(B,C,T)"]
    G --> H["MaxPool"]
    H --> I["(B,C,L)"]
    I --> J["Dropout"]
    J --> K["(B,C,L)"]
    
    L["B=Batch size\nT=Time series length\nL=T//stride\nC=Channels\nH=LSTM hidden\ndimension\nN=LSTM layers"] --> M["(B,L,C)"]
    M --> N["Bi-Directional LSTM"]
    N --> O["Output (B,L,2*H)\nLast Cell (2*N,B,H)\nLast Hidden (2*N,B,H)"]
```
</details>

![](images/c0ad93484365508b46f0e341942083a3bb9bdd94e23a30b9608c81a4d8d26f65.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Reshape"] --> B["(B,L,2*H) (N,B,2*H)"]
    B --> C["Softmax(Last Cell*Output.T)"]
    C --> D["Weights (B,1,L)"]
    D --> E["Weights * Output"]
    E --> F["(B,2*H)"]
    A --> G["Output (B,L,2*H)<br>Last Cell (2*N,B,H)"]
```
</details>

Temporal block.

![](images/58997b3aeecf078384bdd417c0a9f59de435e0390e4304006a2527a70c90c6d5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input (B,T)"] --> B["1d Conv"]
    B --> C["(B,D,L)"]
    C --> D["BatchNorm"]
    D --> E["(B,D,L)"]
    E --> F["SiLU"]
    F --> G["(B,D,L)"]
    G --> H["B=Batch size\nT=Time series length\nL=T//stride\nD=Encoder Dimension\nN=Conformer Layers\nM=Attention Heads"]
```
</details>

N Times   
![](images/31ef78fdbc4c456a2a7ccb4ee4005e78b39b3f8f804a9a23f1d72c0783e4247d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph Top_BLK
        A["qkv Projection"] --> B["Positional Encoding"]
        B --> C["MultiHead Self Attention"]
        C --> D["LayerNomr"]
    end

    subgraph Middle_LLK
        E["1d Conv"] --> F["BatchNorm"]
        F --> G["SiLU"]
    end

    subgraph Bottom_LLK
        H["qkv Projection"] --> I["Positional Encoding"]
        I --> J["MultiHead Self Attention"]
        J --> K["LayerNomr"]
    end

    A --> L["(B,L,D)"]
    B --> M["(B,L,D)"]
    C --> N["(B,L,D)"]
    D --> O["(B,L,D)"]
    E --> P["(B,D,L)"]
    F --> Q["(B,D,L)"]
    G --> R["(B,D,L)"]
    H --> S["(B,L,D)"]
    I --> T["(B,L,D)"]
    J --> U["(B,L,D)"]
    K --> V["(B,L,D)"]
```
</details>

![](images/89ca388ed53bdf2f4f7cf21146cca9451aebbe430b1c1fca44051e24dc718cf9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["(B,L,D)"] --> B["Average"]
    C["(B,D)"] --> D["-->"]
```
</details>

Conformer Block.

![](images/8a7b47475f5e0d5a936508f5770ae491647dd3333011adae5902b338b19d9f9b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input (B,2*H+D)"] --> B["Linear"]
    B --> C["(B,K)"]
    C --> D["GELU"]
    D --> E["(B,K)"]
    E --> F["Dropout"]
    F --> G["(B,K)"]
    G --> H["Linear"]
    H --> I["(B,O)"]
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
```
</details>

Prediction head.

<table><tr><td>parameter</td><td>value</td></tr><tr><td>kernel size</td><td>4</td></tr><tr><td>stride-temporal</td><td>4</td></tr><tr><td>stride-conformer</td><td>20</td></tr><tr><td>channels-temporal</td><td>256</td></tr><tr><td>Dropout-temporal</td><td>0.35</td></tr><tr><td>Dropout-conformer</td><td>0.3</td></tr><tr><td>LSTM layers</td><td>5</td></tr><tr><td>LSTM hidden size</td><td>64</td></tr><tr><td>conformer layers</td><td>5</td></tr><tr><td>attention heads</td><td>8</td></tr><tr><td>conformer encoder dim</td><td>128</td></tr><tr><td>fully connected hidden size</td><td>128</td></tr></table>

parameters of the LightPred model

# C. EXAMPLE OF THE COMPARISON BETWEEN SELF-SUPERVISED AND SIMULATION-BASED TRAINING

![](images/b00d18650716d2455171728c7d83231c79f1630def25d18b36471da9bed3321f.jpg)

<details>
<summary>line</summary>

| Iteration # | train  | val    |
| ----------- | ------ | ------ |
| 0           | 0.0    | -0.8   |
| 5000        | -0.9   | -0.9   |
| 10000       | -0.95  | -0.95  |
| 15000       | -0.9   | -0.9   |
| 20000       | -0.85  | -0.85  |
| 25000       | -0.8   | -0.8   |
| 30000       | -0.75  | -0.75  |
| 35000       | -0.7   | -0.7   |
</details>

![](images/71bc8b26d6af80163ff8a43ac0398c70be042197efd63e5d6c330ca7139b5dda.jpg)

<details>
<summary>line</summary>

| Iteration # | train | val   |
| ----------- | ----- | ----- |
| 0           | 0.9   | 0.35  |
| 10000       | 0.2   | 0.15  |
| 20000       | 0.2   | 0.15  |
| 30000       | 0.2   | 0.15  |
| 40000       | 0.2   | 0.15  |
| 50000       | 0.2   | 0.15  |
| 60000       | 0.2   | 0.15  |
| 70000       | 0.2   | 0.15  |
</details>

Self-supervised training (upper panel) and simulation-based training (lower panel). Grey represents the training dataset; Red represents the validation dataset.

# D. FAST ROTATORS

![](images/5376bd03aee2a810594fd13bab032c6f001c82365c545d6fce0e4a7897498b1a.jpg)

<details>
<summary>scatter</summary>

| Period (Days) | Fraction of consistent values |
| ------------- | ----------------------------- |
| 0             | 0.0                           |
| 2             | 0.05                          |
| 4             | 0.35                          |
| 6             | 0.4                           |
| 8             | 0.45                          |
| 10            | 0.6                           |
| 12            | 0.3                           |
| 14            | 0.45                          |
| 16            | 0.7                           |
| 18            | 0.9                           |
| 20            | 0.7                           |
| 22            | 0.65                          |
| 24            | 0.75                          |
| 26            | 0.7                           |
| 28            | 0.6                           |
| 30            | 1.0                           |
| 32            | 0.5                           |
| 34            | 0.25                          |
| 36            | 0.4                           |
| 38            | 0.25                          |
| 40            | 0.5                           |
| 42            | 0.0                           |
| 44            | 0.0                           |
| 46            | 0.0                           |
| 48            | 0.0                           |
| 50            | 0.25                          |
</details>

![](images/d7269a4e2a73b56bf471d4249a5b620f40fdaeb70f419d19fa025c058b006eae.jpg)

<details>
<summary>scatter</summary>

| Period (Days) | Fraction of consistent values |
| ------------- | ---------------------------- |
| 0             | 0.0                          |
| 5             | 0.45                         |
| 10            | 0.98                         |
| 15            | 0.97                         |
| 20            | 0.95                         |
| 25            | 0.75                         |
| 30            | 0.45                         |
| 35            | 0.35                         |
| 40            | 0.25                         |
| 45            | 0.20                         |
| 50            | 0.30                         |
| 55            | 0.25                         |
| 60            | 0.68                         |
| 65            | 0.22                         |
| 70            | 0.50                         |
</details>

consistency curves of Eclipsing Binaries dataset (left) and McQuillan et al. (2014) Dataset (right).

757099,PLightPred:25.27,PMcQ14: 0.37   
![](images/e4009f4caf5dcc8d8cbdf4d90b44d31ff9179039cb4866cdb3693f0ade015242.jpg)

<details>
<summary>line</summary>

| Time (Days) | Normalized Flux (raw) | Normalized Flux (avg) |
|-------------|------------------------|------------------------|
| 0           | 0.90                   | 0.90                   |
| 500         | 0.92                   | 0.91                   |
| 600         | 0.93                   | 0.92                   |
</details>

6449077,PLightPred:5.33,PMcQ14:0.94   
![](images/3234b69cc359828a15119944ac2312ca22b1db331ab2e6d9b5e21f62e00f34d9.jpg)

<details>
<summary>line</summary>

| Time (Days) | Normalized Flux (raw) | Normalized Flux (avg) |
|-------------|------------------------|------------------------|
| 0           | ~0.985                 | ~0.985                 |
| 500         | ~0.985                 | ~0.985                 |
| 400         | ~0.987                 | ~0.986                 |
| 500         | ~0.986                 | ~0.985                 |
</details>

9119108,PLightPred: 22.48,PMcQ14:0.65   
![](images/f12df29533f2f3c45883f35f1add643344b548bfdba28b3ab327a866fd2c8ec8.jpg)

<details>
<summary>line</summary>

| Time (Days) | Normalized Flux (raw) | Normalized Flux (avg) |
|-------------|------------------------|------------------------|
| 0           | 0.94                   | 0.92                   |
| 500         | 0.96                   | 0.94                   |
| 600         | 0.94                   | 0.92                   |
</details>

Three potential wrong predictions of our model. For each sample, the left panel shows the full light curve and the right panel shows a zoom-in slice of 180 days. The gray color represents the raw light curve. The orange color represents an averaged light curve over a window of 1 day. The titles show the Kepler KIC, lightPred period, and McQ14 period fromMcQuillan et al. (2014).