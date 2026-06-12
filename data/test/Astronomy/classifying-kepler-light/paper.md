# Classifying Kepler light curves for 12,000 A and F stars using supervised feature-based machine learning

Nicholas H. Barbara,1,2★ Timothy R. Bedding,1,2† Ben D. Fulcher,1 Simon J. Murphy1,2, and Timothy Van Reeth1,2,3

1Sydney Institute for Astronomy, School of Physics, University of Sydney 2006, Australia   
2Stellar Astrophysics Centre, Department of Physics and Astronomy, Aarhus University, Denmark   
3Institute of Astronomy, KU Leuven, Celestĳnenlaan 200D, B-3001 Leuven, Belgium

Accepted XXX. Received YYY; in original form ZZZ

# ABSTRACT

With the availability of large-scale surveys like Kepler and TESS, there is a pressing need for automated methods to classify light curves according to known classes of variable stars. We introduce a new algorithm for classifying light curves that compares 7000 time-series features to find those which most effectively classify a given set of light curves. We apply our method to Kepler light curves for stars with effective temperatures in the range 6500–10,000 K. We show that the sample can be meaningfully represented in an interpretable five-dimensional feature space that separates seven major classes of light curves (?? Scuti stars, ?? Doradus stars, RR Lyrae stars, rotational variables, contact eclipsing binaries, detached eclipsing binaries, and non-variables). We achieve a balanced classification accuracy of 82% on an independent test set of Kepler stars using a Gaussian mixture model classifier. We use our method to classify 12,000 Kepler light curves from Quarter 9 and provide a catalogue of the results. We further outline a confidence heuristic based on probability density with which to search our catalogue, and extract candidate lists of correctly-classified variable stars.

Key words: methods: data analysis – stars: variables: general – asteroseismology – stars: oscillations – binaries: eclipsing

# 1 INTRODUCTION

The use of machine learning is becoming increasingly common in astronomy (Ball & Brunner 2010; Graff et al. 2014; Ivezić et al. 2019; Baron 2019). In particular, large-scale photometric surveys are producing light curves in numbers too large for humans to manually inspect and analyse. Considerable efforts have gone into using machine learning to classify light curves from large ground-based surveys (e.g., Carrasco-Davis et al. 2019; Tsang & Schultz 2019; Johnston etal.2019a: Cabral et al.2020:Hosenie et al.2020: Jamal & Bloom 2020; Szklenár et al. 2020; Bassi et al. 2021; Zhang & Bloom 2021). Such techniques have also been applied to light curves from NASA’s Kepler and K2 missions (e.g., Blomme et al. 2010, 2011; Debosscher et al. 2011; Bass & Borne 2016; Armstrong et al. 2016; Hon et al. 2017, 2018b,a; Johnston et al. 2019b; Kgoadi et al. 2019; Le Saux et al. 2019; Giles & Walkowicz 2020; Kuszlewicz et al. 2020; Audenaert et al. 2021; Paul & Chattopadhyay 2022).

A range of algorithms have been proposed to classify light-curve databases according to known classes of stars, but these algorithms often rely on black-box machine-learning methods which limits their interpretability and hence ability to drive scientific understanding. Those algorithms that are more interpretable rely on manually selected temporal or spectral features of light curves (e.g., Pashchenko et al. 2017), with minimal comparison to the performance of alternatives from across a highly interdisciplinary time-series analysis liter-

ature. Here we introduce a new algorithm for classifying light-curve databases that searches over 7000 time-series features to automatically find interpretable features relevant to classifying a given set of light curves. Our aim is to develop a simple, efficient classifier that uses these interpretable features to give us new insight into the classification of variable stars, while maintaining comparable performance with existing methods.

Over the course of its four-year mission, the Kepler spacecraft collected light curves for nearly 200,000 stars, most of which show variability. Subsets of Kepler stars have been classified systematically, resulting in catalogues of about 2900 eclipsing binaries (Kirk et al. 2016), 16,000 oscillating red giants (e.g., Yu et al. 2018, and references therein), 2000 pulsating ?? Scuti stars (Murphy et al. 2019), and over 600 ?? Doradus pulsators (Li et al. 2020). Independently, Balona (2018) used visual inspection of light curves and power spectra to classify over 20,000 A and F Kepler stars. Finally, Audenaert et al. (2021) have recently classified 167,000 light curves from one quarter of Kepler data (see Sec. 4.4).

In this paper, we present our methods and provide classifications based on a single three-month quarter (Q9) for approximately 12,000 Kepler stars with effective temperatures in the range 6500–10,000 K. Our primary interest is in pulsating stars and our chosen temperature range covers the classical instability strip, which has the richest variety of variability Kurtz (e.g., 2022). There are relatively few stars in the Kepler sample that are hotter than this range, while pulsations on the cooler side are dominated by a single class (solar-like oscillations) that have already been extensively classified studied (see Jackiewicz

2021 for a recent review). We note that we have previously used our method to identify samples of ?? Scuti stars (Murphy et al. 2020) and ?? Doradus stars (Li et al. 2020). Finally, we compare our classification of Kepler stars to labels assigned by a less interpretable, performance-focused classifier in Audenaert et al. (2021), where we find similar results.

# 2 TRAINING DATA

# 2.1 A Selection of Variable Stars

Selecting an adequate training set to train a feature-based classifier for all 200,000 stars in the Kepler field is the most challenging and time-consuming aspect of developing a general classification algorithm. Many classes of variable stars are rare, while others remain poorly understood, and still others have not yet been identified. Indeed, new classes are occasionally proposed (Debosscher et al. 2011; Bass & Borne 2016; Pietrukowicz et al. 2017). For this reason, a good training set must be representative of a wide range of variable stars to construct a suitably general feature-space representation of Kepler light curves. Rather than attempting to compile a collection of all known classes of variable and non-variable stars, as attempted with limited success by Debosscher et al. (2011), we focused our research on a subset of seven well-studied classes, within the temperature range $6 5 0 0 \mathrm { K } \leq T _ { \mathrm { e f f } } \leq 1 0 , 0 0 0 \mathrm { K } .$ , as an initial demonstration. The seven chosen classes are ?? Scuti stars, ?? Doradus stars, RR Lyrae stars, rotational variables, contact eclipsing binaries, detached eclipsing binaries, and non-variable stars. Typical light curves and power spectra for each class are included in Fig. 1. These classes are commonly represented in the wider Kepler data (with the exception of the RR Lyrae class), are interesting stars that we wish to study in further detail, and are not intermediate or hybrid classes. We excluded hybrid stars to avoid having light curves in the training data that belong to multiple classes.

Two of our seven classes are subclasses of eclipsing binary (EB) systems, which were catalogued in Kepler data by Kirk et al. (2016). EBs can be classed as one of: detached binaries, where the two stars are far from each other to give highly separated eclipses; contact binaries, where there is no space between the two stellar envelopes, producing almost sinusoidal eclipse patterns; and semi-detached binaries, the intermediate class of the two extremes. In accordance with our decision to exclude hybrid classes, only the detached and contact subclasses have been included in our training set.

Rotational variables are most commonly found among cooler stars, whose star spots present darker patches on the surface that rotate in and out of view, but rotational variability is also seen by Kepler across the effective temperature range of our sample (Balona 2013; Sikora et al. 2020). Typically, the spots have lifetimes not much longer than the rotation period, and they may occur at different latitudes, so the variability is only quasi-periodic (Nielsen et al. 2013; McQuillan et al. 2014). The $\scriptstyle \alpha ^ { 2 } \csc \eta _ { \boldsymbol { \Pi } }$ stars are hotter stars whose strong dipolar magnetic fields concentrate certain elements into spots. These also rotate with the star, but occur near the magnetic poles and are much longer lived, leading to light curves that do not change rapidly in period, amplitude, or shape (Wolff 1983). In our chosen temperature range, examples of both are found.

Three classes of pulsating variable star were included (for a recent review of pulsating stars, see Kurtz 2022). RR Lyr variables are bright, evolved stars burning helium in their cores. As they traverse the horizontal branch, they cross the instability strip and pulsate periodically with a characteristic phase curve. Their use as standard candles has allowed measurements of the distance to the Galactic centre and to globular clusters (Oort & Plaut 1975; Walker 1992). The two other pulsating star classes, ?? Doradus and ?? Scuti variables, both comprise A or F-type stars on or near the main sequence, and embody two distinct types of oscillation: g modes, or buoyancy-driven modes sensitive to the near-core region of a star; and p modes, pressuredriven modes most sensitive to the envelope. ?? Doradus stars are multiperiodic g-mode pulsators with periods between approximately 0.3 d and 3 d (Kaye et al. 1999). Despite having periods similar to the RR Lyr variables, the multiperiodic ?? Doradus stars do not have simple phased curves. There are several hundred in the Kepler field (Li et al. 2020), and they have seen substantial recent attention because of their ability to probe internal rotation (Van Reeth et al. 2018; Ouazzani et al. 2019), diffusive mixing (Bouabid et al. 2013), and core overshooting (Mombarg et al. 2019).

Finally, ?? Scuti stars are the most common class of pulsating star at A and F spectral types, with approximately 2000 known in Kepler data alone (Murphy et al. 2019; Guzik 2021). These stars are p-mode oscillators, and with periods between 18 min and 8 hr they are the highest-frequency variables in our sample. For unknown reasons, even in the middle of the ?? Scuti instability strip only half of the stars pulsate as ?? Scuti stars (Murphy et al. 2019), hence we include a nonvariable class in this work. We note that some ?? Scuti stars are known to lie outside of the instability strip (e.g., Bowman & Kurtz 2018), but our classifications are based only on the Kepler light curves and not on parameters such as effective temperature.

# 2.2 Preparing the Training Set

We created a training set across all seven classes by hand-picking 1319 stars from candidate lists according to specific criteria. Stars were restricted to the temperature range $6 5 0 0 \mathrm { K } \le T _ { \mathrm { e f f } } \le 1 0 , 0 0 0 \mathrm { K }$ using effective temperatures from Mathur et al. (2017). We examined one quarter of long-cadence Kepler photometry for each star to prepare the training data. Quarter 9 (Q9) was chosen because it has no prolonged gaps in observation, such as those arising from telescope safe mode events, and no anomalies in data quality. We used light curves made with simple aperture photometry (SAP), downloaded from the Kepler Asteroseismic Science Operations Center (KASOC) website (Data Release 25).1 The choice to examine a single quarter was made to reduce computation time, but this also precludes the analysis of variability on timescales longer than a typical 90-d quarter. While we certainly recommend the investigation of four-year data in future research focused on a wider range of Kepler variables, this will not have a great effect on the stars chosen for our investigation. Of the seven classes, only a handful of detached binaries are known to have a period greater than 90 days (Kirk et al. 2016), and we did not include these in the training set.

The training stars were chosen from lists of possible candidates for each of the seven classes by visually inspecting their light curves and Fourier transforms. This laborious process embodies the motivation for automated variable star classification, and was a necessary task to ensure that the training data were accurate and would not mislead the automated feature-selection process. In the following paragraphs, we describe the selection of stars in training sets for each class. Table 1 summarises the class-specific number of stars in the training set. The full list of stars is provided as supplementary material, with a sample shown in Table 2.

Eclipsing binary systems were selected from the Kepler Eclipsing

![](images/eff0dde3201fa8b423f80f394690e6077d906ade8ba0260b231da69ba1b1395d.jpg)

<details>
<summary>line</summary>

| Time Series | Sample ID       | Value     |
|-------------|-----------------|-----------|
| Detached EB | KIC 8621026    | ~100000   |
| Contact EB  | KIC 8523194    | ~100000   |
| RR Lyrae    | KIC 8344381    | ~100000   |
| γ Dor       | KIC 7977163    | ~100000   |
| δ Scuti     | KIC 7217483    | ~100000   |
| Rotational  | KIC 7810089    | ~100000   |
| Non-variable| KIC 4269712    | ~100000   |
</details>

![](images/bad9a94cb8e12949816ff2c05f8a38f1e730cf14d0df743da314e0e45a8d5273.jpg)  
Figure 1. Examples of variable stars in the detached binary, contact binary, RR Lyr, ?? Doradus, ?? Scuti, rotational, and non-variable stellar classes selected in the training set, respectively. The left panel shows the first 2200 samples (at approx. 30 min per sample) of the light curves of each star, or half a Kepler quarter of data. The right panel shows the corresponding Fourier transforms up to the Nyquist frequency. The vertical axes are scaled for ease of viewing.

Table 1. Breakdown of stars in the training set. 

<table><tr><td>Class</td><td>No. Stars</td></tr><tr><td>Contact EB</td><td>171</td></tr><tr><td>Detached EB</td><td>83</td></tr><tr><td>δ Scuti</td><td>411</td></tr><tr><td>γ Doradus</td><td>262</td></tr><tr><td>Non-variable</td><td>201</td></tr><tr><td>Rotational</td><td>166</td></tr><tr><td>RR Lyrae</td><td>25</td></tr><tr><td>Total</td><td>1319</td></tr></table>

Table 2. The training set of 1319 Kepler stars. An extract of 14 stars is shown, with the full table provided in the supplementary material. 

<table><tr><td>KIC ID</td><td>Class</td></tr><tr><td>10855535</td><td>Contact EB</td></tr><tr><td>9612468</td><td>Contact EB</td></tr><tr><td>3836439</td><td>Detached EB</td></tr><tr><td>9711751</td><td>Detached EB</td></tr><tr><td>9331207</td><td> $\delta$  Scuti</td></tr><tr><td>8376471</td><td> $\delta$  Scuti</td></tr><tr><td>4755510</td><td> $\gamma$  Doradus</td></tr><tr><td>1996456</td><td> $\gamma$  Doradus</td></tr><tr><td>1864603</td><td>Non-variable</td></tr><tr><td>2156425</td><td>Non-variable</td></tr><tr><td>1164109</td><td>Rotational</td></tr><tr><td>1435836</td><td>Rotational</td></tr><tr><td>3733346</td><td>RR Lyrae</td></tr><tr><td>3864443</td><td>RR Lyrae</td></tr></table>

Binary Catalog (Kirk et al. 2016), restricted to periods <90 d and a morphology index of $0 \leq c \leq 0 . 5$ (detached binaries) or $0 . 7 5 \leq c \leq$ 1.0 (contact binaries), as recommended by Matĳevič et al. (2012).

Our selection of ?? Scuti stars began with 2405 stars manually identified as variable at frequencies above $7 \mathrm { d } ^ { - 1 }$ from a preliminary version of the Murphy et al. (2019) catalogue. We randomly selected 1000 of these, and further refined this list to remove any stars that were also ?? Doradus stars (i.e. ?? Doradus/?? Scuti hybrids) by manual inspection. From the same source, we also chose 500 stars that were not ?? Scuti pulsators, and removed stars with low-frequency variability to arrive at the 201-star non-variable class.

We selected the ?? Doradus sample from the Debosscher et al. (2011) catalogue by choosing stars with a label confidence of >95% and an effective temperature in the appropriate range. While the Debosscher et al. catalogue is known to have errors, this approach was taken due to a lack of an available list of $\gamma$ Doradus stars exhibiting a broad range of oscillatory behaviours characteristic of the class — that is, a sample not restricted to neat and well-studied $\gamma$ Doradus stars from which scientific inference has been made (references in Sec. 2.1). The addition of rigorous manual inspection ensured that the ?? Doradus stars included in the final sample were significantly more likely to be correctly classified than in the Debosscher et al. catalogue, and that hybrid pulsators were removed.

Unlike the other classes, RR Lyr variables are not common in the Kepler data set. Of the 47 Kepler RR Lyr stars we found in the literature (Molnár et al. 2018; Nemec et al. 2013; Murphy et al. 2018), only 25 were observed in Q9. We admitted all 25 of these in the hope that we might discover additional RR Lyrae variables when classifying the remainder of the Kepler field (we did not).

The rotational variables were selected after trialling a preliminary version of our classifier, trained on the other six classes, on a test sample of Kepler stars. When visually inspecting the classification results across the six classes, we found that rotational variables constituted a considerable fraction of stars (approximately 15 − 20%).

From these, we added a list of 166 rotational variables to the training set after a second manual verification.

# 2.3 Processing Kepler data

Starting with SAP fluxes from Q9 light curves, we processed the data to remove instrumental variability by eliminating long-period trends in the light curve of each star. Such variability can arise from physical drift of the telescope, causing changes in the flux levels falling in the aperture mask, as well as other instrumental effects distinct from stellar variability. Our processing involved division by a smoothed version of each light curve (smoothed using a Savitzky-Golay filter), removal of single-point outliers more than 3?? from the mean of the smoothed light curve, and converting units to magnitudes.

Any gaps in the data of more than an hour (corresponding to two 29.45-min integrations) were padded with either the mean of the time series for long gaps of four or more integrations, or the mean of the points on either side of smaller gaps. Even in high-quality quarters, long gaps arise from standard telescope operations such as the data downlinks that happen for approximately 24 hrs twice every quarter, while small gaps may be caused by e.g. cosmic ray events. Most machine-learning tools operate as functions of array index rather than explicit functions of time, hence it is imperative that these gaps are filled.

# 3 FEATURE-BASED LIGHT-CURVE CLASSIFICATION

Having constructed a training set, we next aimed to build a classifier to accurately predict the class of a star from features of its lightcurve time series. Our approach involved four steps: (i) mapping each light curve to a large feature vector, where each feature is a single, real-valued summary statistic that captures some interpretable property of the light curve; (ii) learning a classification rule that maps from a reduced subset of extracted features to the class label on a labelled training set; (iii) evaluating the performance of the learned classification rule on an independent test set; and (iv) applying this rule to classify the full Kepler data set.

# 3.1 Feature extraction

The task of selecting relevant properties of a time series for a given application, like light-curve classification, is commonly a manual one performed by a given researcher (e.g., Pashchenko et al. 2017). An alternative approach, termed ‘highly comparative time-series analysis’ (Fulcher et al. 2013; Fulcher & Jones 2014), is to include a large and comprehensive candidate set of possible time-series features, and take a data-driven approach to selecting those that are most relevant to the task at hand. To extract features from a light curve, we used a comprehensive candidate set of over 7000 time-series features from the hctsa software package (v0.96) (Fulcher & Jones 2017).2 The hctsa feature set encompasses a wide range of time-series analysis methods, from properties of: the distribution of time-series values, linear and nonlinear autocorrelation, entropy and complexity measures, stationarity, time-series model fits, wavelet and Fourier basis-function decompositions, and others (Fulcher et al. 2013). This approach allowed us to represent a set of ?? light curves as an $L \times F$ matrix, where ?? is the number of features; applying hctsa to our training data set yielded a 1319 × 7873 light curve × feature matrix,

where each row is labelled according to one of the seven classes listed in Table 1. After performing feature extraction, we excluded features that contained special values (NaN, Inf), returned an error, or produced near-constant outputs (within 10× machine precision) across all 1319 time series, resulting in 6492 features after filtering. As a preprocessing prior to classification, feature values were normalized to the unit interval using a scaled, outlier-robust sigmoidal transformation (Fulcher et al. 2013).

# 3.2 Training and evaluating a classifier

In modern applications of machine learning, complexity is often introduced at the level of the classifier. In this work we instead focused on selecting from a large candidate set of complex features, but using simple classifiers. This has the advantage of yielding features that can provide clear scientific interpretation, and follows the approach of Timmer et al. (1993): “The crucial problem is not the classificator function (linear or nonlinear), but the selection of well-discriminating features. In addition, the features should contribute to an understanding”. For classification, we used a Gaussian Mixture Model (GMM) (McLachlan & Peel 2000) on a labelled time series × feature matrix (described above). We fitted a single Gaussian component to each of the seven training classes in feature space, combining them with equal prior probabilities to form a seven-component Probability Density Function (PDF). While all classes are not equally common, equal priors are the simplest choice without knowing the true distribution of variable stars in the Kepler field. Classification was performed by evaluating the (posterior) probability of a star belonging to each class using the trained PDF, and selecting the class with highest probability. This GMM approach was substantially faster (by factors of approximately 10–100) than alternative algorithms such as nearestneighbour clustering or support vector machines, but achieved similar classification performance on our training set.

We evaluated classification performance as the average balanced accuracy computed using 10-fold stratified cross-validation Hastie et al. (2009). Balanced accuracy, $C _ { \mathrm { b a l } }$ , accounts for class imbalance (the unequal number of observations in each class) in our data set and is defined as:

$$
C _ {\mathrm{bal}} = \frac {1}{m} \sum_ {i = 1} ^ {m} \frac {t _ {i}}{c _ {i}}, \tag {1}
$$

where ?? is the number of classes, ???? is the number of successfully identified time series in the ??th class, and $c _ { i }$ is the total number of time series in this class.

# 3.3 Feature subset selection

To extract a small number of hctsa features that are most informative of the class labels, we used greedy forward feature selection (Hastie et al. 2009; Fulcher & Jones 2014). This simple algorithm iteratively builds a feature set, one feature at a time, with the objective of maximising the balanced classification accuracy, $C _ { \mathrm { b a l } }$ , at each iteration. That is, at iteration ??, the algorithm searches across all individual features for the feature that maximizes $C _ { \mathrm { b a l } }$ when used in combination with the features selected in the $k - 1$ previous iterations.

hctsa was developed to encompass a comprehensive sample of the interdisciplinary time-series analysis literature, and thus contains groups of features with highly correlated behavior (Fulcher et al. 2013; Henderson & Fulcher 2021). When multiple features exhibit similar classification performance, we implemented a simple heuristic constraint to favour the selection of faster-to-compute features: at each iteration, of the features with an accuracy within a margin of 1% of the best-performing feature, the feature with the fastest computation time was selected. The iterative procedure was terminated when the improvement in training-set $C _ { \mathrm { b a l } }$ from adding another feature dropped below 1%. Note that applying this algorithm to the full training data set has the potential to overfit, since the selection step at each iteration (despite using cross-validation for each feature) uses the training set itself to select the best-performing feature. Accordingly, we evaluate the performance of our reduced feature set on an independent test set in Section 4.

# 4 RESULTS & DISCUSSION

# 4.1 Representing light curves in a high-dimensional feature space

We first investigated the structure of the seven labelled classes of 1319 Kepler stars in the 6492-dimensional hctsa feature space. We found that the hctsa feature space is able to capture characteristic properties of the seven labelled classes of stars, obtaining a high mean 10-fold cross-validated balanced accuracy of 95.9% (using a linear support vector machine (Hastie et al. 2009), compared with a chance rate for seven classes of 14.3%). This indicates that each type of star displays distinctive dynamics in ways that can be detected by the features included in hctsa. To better understand the structure of light curves in the high-dimensional hctsa feature space, we inspected a twodimensional ??-SNE visualization (??-distributed stochastic neighbour embedding; Van Der Maaten & Hinton 2008). The result is shown in Fig. 2, where each point is a light curve, and light curves with similar features tend to be positioned closely in the space. While ??-SNE is an unsupervised technique (Fig. 2 was constructed without knowledge of the class labels), stars are meaningfully organized according to their labelled class, with most stars clustering with other stars of the same type. Consistent with the high classification results reported above, this indicates that the hctsa feature space captures distinctive dynamical properties of the light curves corresponding to the seven different types of stars. The plot also reveals scientifically meaningful structure between classes, such as the continuum from non-variable (light orange) stars to rotational-variable (light blue) stars. We also see a small overlap between RR Lyr stars and contact EBs, which reflects the similar morphologies of their light curves (see Fig. 1).

# 4.2 Representing light curves in a reduced feature space

The results above demonstrate that time-series properties in hctsa can capture differences in light-curve dynamics between different types of stars. But which types of individual time-series features are most informative of these differences? To address this question, we aimed to construct a reduced set of hctsa features that display strong classification performance using greedy forward selection (see Sec. 3.3 for details). The cross-validated balanced misclassification rate on the training set is shown as a function of the number of selected features in Fig. 3. This plot reveals that strong in-sample classification performance can be obtained with a relatively small set of well-chosen time-series features, e.g., a balanced accuracy of 95.2% with just three features. According to our termination criterion— when an additional feature provides < 1% marginal improvement in balanced accuracy—we obtained an informative five-dimensional feature space in which to represent Kepler light curves.

To visualize how stars are organized in the reduced feature space, we plotted the training set in the space corresponding to three of the selected features in Fig. 4 (left). Despite a dramatic dimensionality reduction of each time series—from the 4767 data points in a typical Q9 time series to just three extracted summary statistics— the space meaningfully organizes all seven training classes in this low-dimensional feature representation, with each occupying a characteristic region of the space. Much like the ??-SNE construction in Fig. 2, the relative positions of each class are consistent with what we would intuitively expect from their light curves and power spectra in Fig. 1. For example: detached binaries are highly separated from the other classes, as their light curves are the most distinct; non-variable stars blend with rotational variables when the rotations are weak and difficult to distinguish by eye, such that the light curves are almost non-varying; the ?? Doradus and ?? Scuti stars lie on opposite sides of the space, reflecting their contrasting low- and high-frequency pulsations; and the contact binaries are close to the ?? Doradus and RR Lyrae clusters, which all characteristically exhibit regular lowfrequency variability.

![](images/ad083dbd67c018480afd526ed235244ffd54dcbe2b282d0748739a65ffd5de5e.jpg)

<details>
<summary>scatter</summary>

| Category          | tSNE-1 | tSNE-2 |
| ----------------- | ------ | ------ |
| δ Scuti           | -30    | 10     |
| δ Scuti           | -20    | 30     |
| δ Scuti           | -10    | 20     |
| δ Scuti           | 0      | 10     |
| δ Scuti           | 10     | 5      |
| δ Scuti           | 20     | 0      |
| δ Scuti           | 30     | -5     |
| δ Scuti           | 40     | -10    |
| Non variable      | -30    | -30    |
| Non variable      | -20    | -20    |
| Non variable      | -10    | -10    |
| Non variable      | 0      | 0      |
| Non variable      | 10     | 5      |
| Non variable      | 20     | 10     |
| Non variable      | 30     | 15     |
| Non variable      | 40     | 20     |
| Contact EB        | 15     | 20     |
| Contact EB        | 20     | 15     |
| Contact EB        | 25     | 10     |
| Contact EB        | 30     | 5      |
| Contact EB        | 35     | 0      |
| Contact EB        | 40     | -5     |
| Detached EB       | 5      | 5      |
| Detached EB       | 10     | 10     |
| Detached EB       | 15     | 15     |
| Detached EB       | 20     | 20     |
| Detached EB       | 25     | 25     |
| Detached EB       | 30     | 30     |
| Detached EB       | 35     | 35     |
| Detached EB       | 40     | 40     |
| γ Dor             | 40     | -5     |
| γ Dor             | 35     | -10    |
| γ Dor             | 30     | -15    |
| γ Dor             | 25     | -20    |
| γ Dor             | 20     | -25    |
| γ Dor             | 15     | -30    |
| γ Dor             | 10     | -35    |
| γ Dor             | 5      | -40    |
| γ Dor             | 0      | -45    |
| γ Dor             | -5     | -50    |
| γ Dor             | -10    | -55    |
| γ Dor             | -15    | -60    |
| γ Dor             | -20    | -65    |
| γ Dor             | -25    | -70    |
| γ Dor             | -30    | -75    |
| γ Dor             | -35    | -80    |
| γ Dor             | -40    | -85    |
| Rotational variable| 10     | -10    |
| Rotational variable| 15     | -15    |
| Rotational variable| 20     | -20    |
| Rotational variable| 25     | -25    |
| Rotational variable| 30     | -30    |
| Rotational variable| 35     | -35    |
| Rotational variable| 40     | -40    |
| Rotational variable| 45     | -45    |
| Non variable      | -10    | -30    |
| Non variable      | -15    | -25    |
| Non variable      | -20    | -20    |
| Non variable      | -25    | -15    |
| Non variable      | -30    | -10    |
| Non variable      | -35    | -5     |
| Non variable      | -40    | 0      |
RR Lyrae         | 25     | 30     |
RR Lyrae         | 30     | 35     |
RR Lyrae         | 35     | 40     |
| RR Lyrae         | 40     | 40     |
| RR Lyrae         | 45     | 45     |
| RR Lyrae         | 50     | 50     |
| RR Lyrae         | 55     | 55     |
| RR Lyrae         | 60     | 60     |
| RR Lyrae         | 65     | 65     |
| RR Lyrae         | 70     | 70     |
| RR Lyrae         | 75     | 75     |
| RR Lyrae         | 80     | 80     |
| RR Lyrae         | 85     | 85     |
| RR Lyrae         | 90     | 90     |
| RR Lyrae         | 95     | 95     |
| RR Lyrae         | 100    | 100    |
| RR Lyrae         | 105    | 105    |
| RR Lyrae         | 110    | 110    |
| RR Lyrae         | 115    | 115    |
| RR Lyrae         | 120    | 120    |
| RR Lyrae         | 125    | 125    |
| RR Lyrae         | 130    | 130    |
| RR Lyrae         | 135    | 135    |
| RR Lyrae         | 140    | 140    |
| RR Lyrae         | 145    | 145    |
| RR Lyrae         | 150    | 150    |
| RR Lyrae         | 155    | 155    |
| RR Lyrae         | 160    | 160    |
| RR Lyrae         | 165    | 165    |
| RR Lyrae         | 170    | 170    |
| RR Lyrae         | 175    | 175    |
| RR Lyrae         | 180    | 180    |
| RR Lyrae         | 185    | 185    |
| RR Lyrae         | 190    | 190    |
| RR Lyrae         | 195    | 195    |
| RR Lyrae         | 200    | 200    |
| RR Lyrae         | 205    | 205    |
| RR Lyrae         | 210    | 210    |
| RR Lyrae         | 215    | 215    |
| RR Lyrae         | 220    | 220    |
| RR Lyrae         | 225    | 225    |
| RR Lyrae         | 230    | 230    |
| RR Lyrae         | 235    | 235    |
| RR Lyrae         | 240    | 240    |
| RR Lyrae         | 245    | 245    |
| RR Lyrae         | 250    | 250    |
| RR Lyrae         | 255    | 255    |
| RR Lyrae         | 260    | 260    |
| RR Lyrae         | 265    | 265    |
| RR Lyrae         | 270    | 270    |
| RR Lyrae         | 275    | 275    |
| RR Lyrae         | 280    | 280    |
| RR Lyrae         | 285    | 285    |
| RR Lyrae         | 290    | 290    |
| RR Lyrae         | 295    | 295    |
| RR Lyrae         | 300    | 300    |
| RR Lyrae         | 305    | 305    |
| RR Lyrae         | 310    | 310    |
| RR Lyrae         | 315    | 315    |
| RR Lyrae         | 320    | 320    |
| RR Lyrae         | 325    | 325    |
| RR Lyrae         | 330    | 330    |
| RR Lyrae         | 335    | 335    |
| RR Lyrae         | 340    | 340    |
| RR Lyrae         | 345    | 345    |
| RR Lyrae         | 350    | 350    |
| RR Lyrae         | 355    | 355    |
| RR Lyrae         | 360    | 360    |
| RR Lyrae         | 365    | 365    |
| RR Lyrae         | 370    | 370    |
| RR Lyrae         | 375    | 375    |
| RR Lyrae         | 380    | 380    |
| RR Lyrae         | 385    | 385    |
| RR Lyrae         | 390    | 390    |
| RR Lyrae         | 395    | 395    |
| RR Lyrae         | 400    | 400    |
Non variable      |

The chart displays the distribution of tSNE-1 and tSNE-2 genes based on the same variables. The x-axis represents the tSNE-1 values and the y-axis represents the tSNE-2 values. The legend indicates that each color corresponds to a specific category in the data. The color labels are not explicitly provided in the code.
</details>

Figure 2. A two-dimensional ??-SNE projection of our Kepler training set of 1319 stars in the 6492-dimensional hctsa feature space, where each light curve is coloured by its class label. Most stars form clear clusters that match their class identity, indicating that the hctsa features provide a useful space in which to represent Kepler light curves.

![](images/1e7f06aa38a4891954c0efa4fffa82057bcb0c7d2b1da62e9471e40e592160a8.jpg)

<details>
<summary>line</summary>

| No. features added | Misclassification rate (%) |
| ------------------ | -------------------------- |
| 1                  | 34.0                       |
| 2                  | 8.0                        |
| 3                  | 5.0                        |
| 4                  | 3.0                        |
| 5                  | 2.0                        |
| 6                  | 1.5                        |
| 7                  | 1.5                        |
| 8                  | 1.5                        |
| 9                  | 1.0                        |
| 10                 | 1.0                        |
</details>

Figure 3. Classification performance as a function of the number of timeseries features. Balanced misclassification rate on the training set (using a GMM classifier) is plotted as a function of selected features, shown as the mean and standard deviation across 10-fold cross validation.

![](images/1698d28127eb9fc4b13e6c1e830fbe354be5f63c58f45b3515688e380c4cae6f.jpg)

<details>
<summary>scatter</summary>

| Feature | Feature 1 | Feature 2 | Feature 5 |
|---------|-----------|-----------|-----------|
| Contact EB | ~0.8 | ~0.9 | ~0.9 |
| γ Dor | ~0.7 | ~0.6 | ~0.7 |
| RR Lyrae | ~0.6 | ~0.4 | ~0.8 |
| δ Scuti | ~0.7 | ~0.5 | ~0.6 |
| Rotational variable | ~0.6 | ~0.3 | ~0.4 |
| Detached EB | ~0.5 | ~0.2 | ~0.1 |
| Non-variable | ~0.4 | ~0.1 | ~0.05 |
</details>

![](images/b39380389521f6c9c73b227b16842cabead63bb8ab98d2e1d472c7c0a120ee59.jpg)

<details>
<summary>scatter</summary>

| Feature Type         | Feature 1 | Feature 2 | Feature 5 |
|----------------------|-----------|-----------|-----------|
| Contact EB           | [values]  | [values]  | [values]  |
| Detached EB          | [values]  | [values]  | [values]  |
| δ Scuti              | [values]  | [values]  | [values]  |
| γ Dor                | [values]  | [values]  | [values]  |
| Non-variable         | [values]  | [values]  | [values]  |
| Rotational variable  | [values]  | [values]  | [values]  |
| RR Lyrae             | [values]  | [values]  | [values]  |
| Unlabelled           | [values]  | [values]  | [values]  |
</details>

Figure 4. Separation of different classes of stars in the (normalized) space of three time-series features selected from hctsa using greedy forward-feature selection. The two figures show (left) the training set, and (right) the training set alongside unlabelled Kepler data with $6 5 0 0 \mathrm { K } \le T _ { \mathrm { e f f } } \le 1 0 , 0 0 0 \mathrm { K }$ . The three features correspond to AC\_nl\_001, MF\_steps\_ahead\_ar\_best\_6\_mabserr\_5, and SP\_Summaries\_welch\_rect\_peakPower $^ { \cdot 5 , }$ as described in detail in Sec. 4.2.1.

# 4.2.1 The reduced feature set

We have demonstrated the usefulness of representing Kepler light curves in an low-dimensional feature space, but what types of properties are these features measuring, and what can that tell us about how light-curve dynamics differ between the seven classes of stars? In this section, we explain the five features in order of their selection by our greedy forward selection algorithm. Noting the small marginal improvements in accuracy after approximately three features (shown in Fig. 3), we focus in particular on these features. In the following discussion, note that the time series were converted to magnitudes, so that positive excursions correspond to decreases in stellar flux, and vice versa.

The first selected feature (labelled AC\_nl\_001 in hctsa), is a nonlinear autocorrelation statistic that computes the time-average, $\langle x _ { t } ^ { 3 } x _ { t - 1 } \rangle _ { t }$ , of the ??-scored time series $x _ { t } .$ , with a time-lag of 1 sample (approximately 30 minutes in the time domain). Similar to a lag-1 autocorrelation, $\langle x _ { t } x _ { t - 1 } \rangle _ { t }$ , it gives high values to highly autocorrelated light curves, but the modification $( x _ { t } ^ { 3 } )$ accentuates large deviations from the mean. The distribution of this feature’s (sigmoidnormalized) values across the seven classes of stars is shown in Fig. 5. Detached binaries have the largest values of this statistic, driven by large positive excursions from the mean (since the time series are in magnitudes). Autocorrelation arising from slower periodic patterns, as in ?? Doradus, rotational stars, RR Lyr and contact binaries, lead to moderate positive values of AC\_nl\_001, while the non-variable stars have low values (raw values near zero). The high-frequency oscillations seen in some ?? Scuti stars (e.g., Balona et al. 2019; Bedding et al. 2020) resulted in negative values of AC\_nl\_001 (the lowest normalized values).

Feature 2 (labelled MF\_steps\_ahead\_ar\_best\_6\_mabserr\_5

![](images/578715f49fdf6fe85693a9938e5e5d73cbfacfb0a3aae055cc28d15ed4d3b38f.jpg)

<details>
<summary>violin</summary>

| Class     | Value |
| --------- | ----- |
| RRLyr     | 0.08  |
| contact   | 0.07  |
| dSct      | 0.06  |
| detached  | 1.00  |
| gDor      | 0.25  |
| nonvar    | 0.05  |
| rot       | 0.23  |
</details>

Figure 5. Values for Feature 1, which computes $\langle x _ { t } ^ { 3 } x _ { t - 1 } \rangle _ { t }$ (see Sec. 4.2.1). The violin plots show the normalised output of AC\_nl\_001 across the seven classes of stars in the training set. Sigmoidal normalization scaled to the unit interval (see methods) was used to aid visualisation of the large range of raw values of this feature.

in hctsa) uses a linear autoregressive (AR) model to measure how predictable a time series is. This statistic captures how well an AR model (of optimal order, selected in the range 1–10 using Schwartz’s Bayesian Criterion) can predict 5 time steps ahead in the time series. This is measured relative to simple benchmark forecasting methods (including simple mean forecasts and a constant global-mean forecast), calculated as the mean absolute error. The distribution of feature values across the seven classes of stars is shown in Fig. 6. Values near zero indicate strong prediction performance of the AR model relative to simple benchmarks, while values greater than 1 indicate relatively inferior model performance. We see high values for the non-variable stars, detached binaries, rotational stars, and most of the ?? Scuti stars, with RR Lyr stars displaying intermediate values (a few RR Lyr stars with highly symmetric light curves have low values). The ?? Doradus and contact binary light curves exhibit a strong linear correlation structure that allowed the AR models to make strong forecasts of these time series, yielding low values for this statistic.

![](images/ca1bc99f54081d78c309a7190ef83ffc9cc67a82353d156e8bd09271668ac700.jpg)

<details>
<summary>violin</summary>

| Class     | Feature value (MF_steps_ahead_ar_best_6_mabserr_5) |
| --------- | ---------------------------------------------------- |
| RRLyr     | 0.6                                                  |
| contact   | 0.2                                                  |
| dSct      | 0.9                                                  |
| detached  | 1.3                                                  |
| gDor      | 0.7                                                  |
| nonvar    | 1.0                                                  |
| rot       | 0.9                                                  |
</details>

Figure 6. Distribution of Feature 2 values by class (see Sec. 4.2.1), which measures how predictable the time series is using a linear autoregressive (AR) model; high values (near 1) are given to light curves for which the AR model performs worse than simple benchmarks, whereas values near 0 are given when the AR model strongly outperforms the benchmarks. Violin plots are shown for the distribution of this feature across the seven classes of stars.

Feature 3, labelled CO\_trev\_3\_num in hctsa, evaluates the following time-average: $\langle ( x _ { t } - x _ { t - 3 } ) ^ { 3 } \rangle _ { i }$ ?? . This statistic, using a time-lag $\tau \ = \ 3 ,$ , can be thought of as capturing asymmetry in the size of increases $( x _ { t } - x _ { t - 3 } > 0 )$ versus decreases $( x _ { t } - x _ { t - 3 } < 0 )$ . For example, time series with sudden increases but gradual decreases (at a time-lag of 3 samples) will have large values of this feature. RR Lyr are distinguished by negative values of CO\_trev\_3\_num, due to the characteristic asymmetry in the shapes of their light curves (e.g., Catelan & Smith 2015).

Feature 4, labelled ST\_LocalExtrema\_n100\_medianmax in hctsa, captures how positive outliers are distributed through the time series. Operating on the ??-scored time series, this algorithm computes the maximum value in each of 100 overlapping windows (each containing 47 samples corresponding to approximately 23 hours), and outputs the median of these local maxima. For time series with relatively infrequent large positive excursions (like the light curves from many detached binaries, recalling that the calculations are done with magnitudes), most windows will have very low maxima, and thus the median of the maxima will be a low value. But for time series with maxima spaced more evenly throughout time, like most non-variable and ?? Scuti stars, high values are obtained for this statistic.

Feature 5, labelled SP\_Summaries\_welch\_rect\_peakPower\_5 in hctsa, uses Welch’s method and a rectangular window to estimate the power spectrum and returns the proportion of power captured by the five most prominent identified peaks. Broadly, this feature gives high values to time series that are well-captured by a relatively small number of dominant frequencies. The lowest values for this feature were found for non-variable stars and rotational variables, while high values were obtained for contact binaries and RR Lyr stars.

Table 3. The test set of 515 Kepler stars. An extract of 12 stars is shown, with the full table provided in the supplementary material. 

<table><tr><td>KIC ID</td><td>Class</td></tr><tr><td>8282730</td><td>Contact EB</td></tr><tr><td>6957185</td><td>Contact EB</td></tr><tr><td>8953296</td><td>Detached EB</td></tr><tr><td>5090690</td><td>Detached EB</td></tr><tr><td>8585472</td><td> $\delta$  Scuti</td></tr><tr><td>3648131</td><td> $\delta$  Scuti</td></tr><tr><td>6041803</td><td> $\gamma$  Doradus</td></tr><tr><td>8739181</td><td> $\gamma$  Doradus</td></tr><tr><td>5616145</td><td>Non-variable</td></tr><tr><td>8153411</td><td>Non-variable</td></tr><tr><td>3847563</td><td>Rotational</td></tr><tr><td>3967219</td><td>Rotational</td></tr></table>

# 4.2.2 Evaluation on a test data set

Having computed an informative low-dimensional space in which to represent Kepler light curves, we investigated its effectiveness in classifying variable stars outside our training set. We manually compiled a test set of 515 stars in the Kepler field belonging to classes of variable stars in our training set, and with $6 5 0 0 \mathrm { K } \leq T _ { \mathrm { e f f } } \leq$ 10, 000 K. The full list of test stars is provided as supplementary material, with a sample shown in Table 3.

To evaluate classification performance on the test set in the trained five-dimensional feature space, we constructed a GMM consisting of seven Gaussian components, one fitted to each class in our training set (with uniform priors), and used it to classify each of the test stars. Figure 7 summarises our results on the test set as a confusion matrix.

The confusion matrix can be interpreted as follows. Labels on each row were assigned by the GMM classifier, while labels on each column correspond to manually-assigned labels from our test set. Each cell (??, ??) of the confusion matrix shows the number of stars (and percentage of all stars considered) that were classified as category ?? by the GMM, and as category ?? in our test set. For example, there were 18 stars classified as detached binaries by the GMM but labelled as non-variable in the test set. Diagonal elements of the matrix (in green) correspond to correctly classified stars. Summaries in grey on the right of the matrix correspond to the (unbalanced) percentage of correct predictions, while summaries at the bottom are the percentage of each class that was correctly classified. The raw classification accuracy is shown in blue in the bottom-right corner.

Our classifier achieved a balanced accuracy of 81.6% on the test set and performed well on all classes, with two understandable exceptions highlighted in Figure 7:

• Non-variable stars are commonly misclassified as detached binaries (18 misclassifications). Most have sharp transitions in their light curves at the beginning or end of the quarter, or just before or after the Kepler telescope paused observation for data transmission. These transitions appear as sharp peaks or troughs, and are represented similarly to eclipses in our feature space.   
• ?? Doradus stars are commonly misclassified as rotational variables. Both classes have low-frequency variations (e.g. Li et al. 2019) and even for an expert eye, it can be difficult to resolve ?? Doradus

![](images/7230b983121ba47aede7a006fd54559dabe8fb4ff31aab0e8ffbfc22f48fe8ff.jpg)

<details>
<summary>heatmap</summary>

| GMM labels | contact | detached | dSct | gDor | nonvar | rot | RRLyr |
|---|---|---|---|---|---|---|---|
| contact | 43 | 3 | 2 | 1 | 0 | 6 | 0 |
| detached | 0 | 41 | 0 | 1 | 18 | 3 | 0 |
| dSct | 3 | 0 | 72 | 1 | 0 | 3 | 0 |
| gDor | 2 | 0 | 0 | 62 | 0 | 2 | 0 |
| nonvar | 0 | 0 | 0 | 0 | 93 | 3 | 0 |
| rot | 3 | 2 | 3 | 31 | 7 | 105 | 0 |
| RRLyr | 2 | 0 | 3 | 0 | 0 | 0 | 0 |
| contact | 81.1% | 89.1% | 90.0% | 64.6% | 78.8% | 86.1% | 0.0% |
| detached | 18.9% | 10.9% | 10.0% | 35.4% | 21.2% | 13.9% | 0.0% |
| dSct | 21.8% | 21.8% | 21.8% | 21.8% | 21.8% | 21.8% | 21.8% |
| gDor | 65.1% | 65.1% | 65.1% | 65.1% | 65.1% | 65.1% | 65.1% |
| nonvar | 93.9% | 93.9% | 93.9% | 93.9% | 93.9% | 93.9% | 93.9% |
| rot | 69.5% | 69.5% | 69.5% | 69.5% | 69.5% | 69.5% | 69.5% |
| RRLyr | 80.8% | 80.8% | 80.8% | 80.8% | 80.8% | 80.8% | 80.8% |
| contact: detached: dSct: gDor: nonvar: rot: RRLyr: contact: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: detached; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detached: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr.; detachte: Retr./RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: RRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: BRLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CBLyr.: CMLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLYLY L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L Y L<ecel><ecel><ecel><ecel><ecel><ecel><ecel><ecel><nl>
</details>

Figure 7. Confusion matrix summarising GMM classification performance. The GMM and test labels are the classifier predictions and manually-assigned truth labels (respectively) for stars in our test set. Summaries in grey on the right of the matrix correspond to the (unbalanced) percentage of correct predictions, while summaries at the bottom are the (unbalanced) percentage of each class that was correctly classified. The raw classification accuracy is shown in blue (balanced accuracy 81.6%). There were no RR Lyrae stars in our test set.

oscillations from a single quarter of Kepler data. The behaviour may therefore look similar to rotation in our feature space.

Apart from these exceptions, our approach yielded high overall classification accuracies despite relying on very simple methods (greedy forward feature selection and GMM classification), demonstrating the usefulness of the comprehensive hctsa feature space in highlighting high-performing interpretable features for a given problem. We expect that repeating the feature selection and classification procedures with more sophisticated algorithms, while still working with a rich set of interpretable features, would further improve the accuracy reported here using simple methods. However, as discussed in Section 4.3, our method is already a useful tool for classifying and searching large data sets.

# 4.3 Classifying the Kepler field

In this section, we use the low-dimensional feature space learned from the training set and validated on the test set to classify variable stars across the entire Kepler field. Our full classification catalogue is provided as supplementary material (Table 4).

# 4.3.1 Classifying unlabelled stars

We computed the 5-dimensional feature-space representation of all 12,088 stars with Q9 data in the Kepler field and 6500 $\ K \leq T _ { \mathrm { e f f } } \leq$ 10, 000 K (excluding our training set of 1319 stars). These are plotted in the right panel of Fig. 4 as unlabelled stars (grey). Each feature vector was normalized using the same scaled robust sigmoid transformation (including its coefficients) as used on the training set, preserving the structure of our normalized feature space. The grey

unlabelled stars are clearly clustered around the coloured training groups, with the majority of stars residing near the non-variable cluster. This clustering occurs naturally because of our choice of feature space. Intuitively, we might expect that (i) unlabelled stars near each training cluster belong to that respective class; (ii) stars midway between two groups are hybrids of both classes; and (iii) stars far from any group are new classes of variable stars unaccounted for in our training set. We have already verified the first of these hypotheses by applying our GMM classifier to the test set with reasonably high accuracy in Section 4.2.2. We leave investigation of the remaining two claims as future work.

We evaluated our trained GMM classifier on all 12,088 unlabelled stars to generate a catalogue of posterior probabilities, giving a predicted classification for each star. The first ten lines of this catalogue are shown in Table 4, with the full catalogue provided as supplementary material. For each star in the catalogue, its classification is the class with maximum posterior probability. The catalogue is intended as a useful tool in searching for candidate variable stars of interest. We note that this catalogue and our broader methodology have already proven useful in identifying new ?? Doradus stars (Li et al. 2019) and ?? Scuti stars (Murphy et al. 2020) in the Kepler field. We provide suggestions for searching our catalogue in Section 4.3.2.

# 4.3.2 Using probability density as a confidence heuristic

Close examination of Fig. 4 reveals that many unlabelled stars lie in areas between the training set clusters, far from where the GMM classifier was trained. We may therefore ask: how does our classification accuracy improve if we restrict the test set to stars “near” the training distributions? We define $p ( x )$ as the probability density of a star represented by feature vector ??, where the probability density function is the 7-class GMM used for classification. Stars close (in feature space) to the centre of the multivariate Gaussians will have large probability densities, ??(??), while those far from the class centroids will have low ??(??). In this sense, we can use $p ( x )$ as a heuristic measure of how likely a star is to belong to any of the training classes. $p ( x )$ is provided in the final column of Table 4.

As an example, the left and right panels of Fig. 8 show the distributions of $p ( x )$ for all ?? Scuti stars in the training set and the rest of the Kepler field (in our temperature range of interest), respectively. Classifications in the right panel of Fig. 8 are from the (manually-compiled) Murphy et al. (2019) catalogue. As anticipated, misclassification is far more common at low densities. Interestingly, the distribution in Fig. 8 shows that above $p ( x ) \approx 1$ , all predictions of ?? Scuti stars are correct. We would intuitively expect similar $p ( x )$ cutoffs for the other classes, above which we have high confidence in the GMM predictions. However, defining exact cut-offs is impossible without a full labelled catalogue of the Kepler field. Instead, we can define values for $p ( x )$ representing regions of increasing proximity to our trained distribution. The vertical lines in the left panel of Fig. 8 show $p ( x )$ percentile cut-offs, above which a certain percentage of the training data fall. For example, only the 90% “closest” ?? Scuti training stars to the ?? Scuti centroid (in terms of probability density) lie above the blue line in Fig. 8.

The results above, for ?? Scuti stars, suggests that our predictions are more accurate in higher-confidence areas of the feature space, corresponding to areas with higher modelled density for the training set. To test whether this holds more generally, we computed the balanced classification accuracy (across all classes) on the test set for a range of $p ( x )$ percentile thresholds. As shown in Fig. 9, we find that accuracy improves with more stringent restrictions on $p ( x )$ ), demonstrating the usefulness of $p ( x )$ as a proxy for prediction confidence. Even small restrictions in $p ( x )$ , such as the $9 5 ^ { \mathrm { t h } }$ percentile cut-off (green line), improve the classification performance on our test set to approximately 90% accuracy. This is an example of a useful way to search our catalogue and obtain a list of confidently-classified variable stars for further analysis — as ??(??) increases for each class, so too does the confidence of our predictions. We once again stress that such intuitive search criteria are a direct consequence of our choice of feature space and simple classification algorithm. One could achieve even more accurate results with more sophisticated approaches, but this may come at the expense of interpretability of our low-dimensional feature space.

Table 4. Extract of GMM posterior class probabilities and probability density $p ( x )$ for 12,088 unlabelled stars in the Kepler field. The first 10 lines are shown, with the full table provided in the supplementary material. 

<table><tr><td>KIC ID</td><td>Contact EB</td><td>Detached EB</td><td> $\delta$  Scuti</td><td> $\gamma$  Dor</td><td>Non-variable</td><td>Rotational variable</td><td>RR Lyrae</td><td> $p(x)$ </td></tr><tr><td>757280</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.73</td></tr><tr><td>892667</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>9.12</td></tr><tr><td>892828</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.93</td><td>0.07</td><td>0.00</td><td>307.19</td></tr><tr><td>893234</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.99</td><td>0.00</td><td>4.12</td></tr><tr><td>893944</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>3802.55</td></tr><tr><td>1026133</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>1.02</td></tr><tr><td>1026255</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.51</td><td>0.49</td><td>0.00</td><td>0.71</td></tr><tr><td>1026475</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>9.54</td></tr><tr><td>1026861</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.44</td></tr></table>

![](images/6bcc84609486678894c21dbeb4419f3f4ca21a460b1dee72063184b1d9ec8e00.jpg)

<details>
<summary>histogram</summary>

| p(x) Range | δ Scutis Counts | Misclassified Counts |
|------------|-----------------|----------------------|
| 10⁻⁴ to 10⁻³ | 0 | 1 |
| 10⁻³ to 10⁻² | 1 | 0 |
| 10⁻² to 10⁻¹ | 2 | 0 |
| 10⁻¹ to 10⁰ | 3 | 0 |
| 10⁰ to 10¹ | 8 | 0 |
| 10¹ to 10² | 33 | 0 |
</details>

![](images/108a836d95893f78eaa21e9f6243e81bf11ae7ea5cc956edf1cb33aac17ae0ea.jpg)

<details>
<summary>bar</summary>

| p(x) Range       | δ Scutis | Misclassified |
| ---------------- | -------- | ------------- |
| 10⁻⁴ - 10⁻³      | 1        | 2             |
| 10⁻³ - 10⁻²      | 2        | 1             |
| 10⁻² - 10⁻¹      | 5        | 4             |
| 10⁻¹ - 10⁰       | 8        | 3             |
| 10⁰ - 10¹        | 17       | 2             |
| 10¹ - 10²        | 42       | 0             |
</details>

Figure 8. Histograms of GMM probability density $p ( x )$ for all stars classified as ?? Scutis in (left) the training set, and (right) the remainder of the Kepler field with $6 5 0 0 \mathrm { K } \leq T _ { \mathrm { e f f } } \leq 1 0 .$ , 000 K. Lines in the training distribution indicate ?? ( ??) percentile cut-offs. For example, 90% of stars classified as ?? Scuti in the training set lie above the blue line. Stars in the right panel are classified according to Murphy et al. (2019).

# 4.4 Comparison with Audenaert et al. (2021)

When our paper was in the final stages of preparation, a new classification of Kepler light curves was published by Audenaert et al. (2021, hereafter Aud21). Their work was done as part of efforts to design an automated classification algorithm for the TESS mission. Given the complementary nature of Aud21 and our own study, especially given that both were based on Q9 data, it is worthwhile to carry out a brief comparison. We should keep in mind that the emphasis in Aud21 was on providing a high-performance classification pipeline from existing methods, whereas ours involved designing an interpretable classifier from a rich library of time-series features.

![](images/1f217fa814bc99ebcb192cfe1e550abcf93a1df5514431b9a898bcfa3a04db9f.jpg)

<details>
<summary>line</summary>

| p(x) Percentile Cut-Offs (%) | Balanced Accuracy (%) |
| ---------------------------- | --------------------- |
| 0                            | 84.5                  |
| 5                            | 89.0                  |
| 10                           | 92.0                  |
| 15                           | 94.5                  |
| 20                           | 95.5                  |
| 25                           | 95.8                  |
</details>

Figure 9. Balanced classification accuracy as a function of $p ( x )$ percentile cut-offs. Applying the classifier to stars close to regions of feature-space that we trained on significantly improves the overall accuracy. Dotted lines correspond to the same percentile cut-offs overlaid in Fig. 8.

The classification by Aud21 included about 167,000 Kepler Q9 light curves, regardless of effective temperature, whereas our work is restricted to about 12,000 stars with 6500 $\mathsf { K } \le T _ { \mathrm { e f f } } \le 1 0 , 0 0 0$ K. We have compared our classifications with Aud21 in Fig. 10. There is an obvious mapping between most of our classes and those used by Aud21, with the following differences:

• Aud21 combined contact eclipsing binaries and rotational (spotted) variables into a single class.   
• Aud21 included ?? Scuti stars in a class with ?? Cephei stars. These have similar light curves but the ?? Cep pulsators have higher effective temperatures that lie outside the range of our sample. Similarly, Aud21 combined ?? Doradus stars with SPBs (Slowly Pulsating B stars), which are also hotter than our sample.   
• Aud21 included a class for solar-like oscillators, which should not appear in our sample because they occur in stars whose effective temperatures fall below our range.   
• Aud21 also included a class for aperiodic variables.

We see from the confusion matrix in Fig. 10 that there is generally excellent agreement between our results and those of Audenaert et al. (2021). We briefly discuss the areas with the greatest disagreement:

(i) 3094 stars that our classifier labelled as non-variable were classified by Aud21 as rotational/contact EB. We inspected 200 of these light curves (and their Fourier amplitude spectra) and found that most are non-variable, with some showing a weak rotation signal.   
(ii) 637 stars that we labelled as contact EBs or rotational variables were classified by Aud21 as ?? Doradus pulsators. Inspection of 200 light curves shows that most are indeed ?? Doradus stars. This may be a shortcoming of our specific feature space and classifier, particularly when considering Fig. 7, where the same disagreement occurs between our GMM classifications and our independent test set.   
(iii) 153 stars were labelled by us as ?? Doradus stars and by Aud21 as contact EBs or rotational variables. Inspection of these shows that many are indeed ?? Doradus stars, although it is sometimes difficult to be sure.   
(iv) 139 stars in our sample were labelled by Aud21 as having solar-like oscillations, which is not a class that we considered because these oscillations occur in stars below our temperature range. Our classifications for these light curves were mainly as rotational variables, contact EBs or non-variable. We inspected all 139 light curves and found that our classifications were mostly correct.   
(v) 106 stars were labelled by us as contact EBs or rotational variables, and by Aud21 as ?? Scuti stars. We inspected all light curves and found that most have ?? Scuti pulsations, but many also have low-frequency variability.

Finally, we note KIC 10024862, which is one of two stars listed by Aud21 as non-variable and by our algorithm as a detached binary. In fact, Kawahara & Masuda (2019) identified this as a Jupiter-sized exoplanet in a long-period orbit that has only one transit during the 4-year Kepler mission, which happened to be in Q9. This suggests that it might be worthwhile to look in more detail at groups for which classification methods are in disagreement for a small number of stars.

Much like our approach, Aud21 assigned labels to each star according to the class with the highest posterior probability from their classifier. Figure 10 therefore contains samples where either classifier may be confused — for example, a given light curve may have probabilities of 0.34, 0.35, 0.01 split between three classes and the maximum probability (0.35) is relatively low. Not surprisingly, we found that by restricting to stars with a high maximum probability in both samples, the agreement increased between our classification labels and those from Aud21. A detailed comparison of the two catalogues goes beyond the scope of this paper and would require a measure of label confidence from the Aud21 classifier similar to the probability density heuristic from Section 4.3.2.

![](images/3e76a9744df05cf656cd678d45cbe89f82755d3a961f6c11a8a0f863de9ab36c.jpg)  
Figure 10. Confusion matrix comparing the results of the Audenaert et al. (2021) classifier to our own classifications for about 12,000 stars in the Kepler field with 6500 K $\leq T _ { \mathrm { e f f } } \leq$ 10, 000 K. Much like in Fig. 7, the grey summary boxes on the right correspond to the percentage of Aud21 labels that agree with our GMM predictions, while summaries at the bottom are the percentage of our predictions for each class that agree with Aud21.

In general, we conclude that the two approaches produce results that generally agree well. The difference in point (i) reflects the subjectivity in drawing the line between variables and non-variables (and perhaps also different amounts of filtering applied to the light curves). Points (ii) and (iii) reflect the difficulty—especially with short data sets—in deciding whether low-frequency variability is due to pulsation or rotation (e.g., Briquet et al. 2007; Lee 2021; Kurtz 2022).

# 5 CONCLUSIONS

We have used a feature-based machine learning algorithm to classify Kepler light curves for stars with effective temperatures in the range 6500–10,000 K. We first created a training set of 1319 light curves, which we classified into seven classes: ?? Scuti stars, ?? Doradus stars, RR Lyrae stars, rotational variables, contact eclipsing binaries, detached eclipsing binaries, and non-variable stars. We built a classifier using features selected with the hctsa package (highly comparative time-series analysis; Fulcher & Jones 2017), which includes over 7000 time-series features. We found that five features were sufficient to represent the training set with a balanced accuracy of 98%, and a separate test set of 500 stars with a balanced accuracy of 82%.

We used our method to classify Kepler light curves for all 12,000 stars with effective temperatures in the range 6500–10,000 K, and the results are tabulated in the supplementary online material (Table 4). We further outlined a confidence heuristic based on probability density with which to search our catalogue and extract candidate lists of correctly-classified variable stars. We also compared our classifications to recent work on the same light curves by Audenaert et al. (2021) and generally found good agreement.

While many modern approaches to machine-learning focus on performance over interpretability (resulting in the common description of being ‘black box’ algorithms), here we favoured the selection of high-performing and interpretable features with which to meaningfully represent Kepler light curves. Given the ease with which our five features can be computed for a large database of light curves, comparing complex classification algorithms to our methods could provide an independent benchmark for general light-curve classification algorithms, much like we have shown with our comparison to Audenaert et al. (2021).

Further extensions of this work might include using our catalogue to search for rare classes of variable stars, hybrid systems, and new stars entirely different to our training sample. In particular, we expect stars with roughly equal posterior probabilities between two classes to be hybrid systems, and very different stars to have much lower probability density scores than any other star in the Kepler field. Our methods could also be applied to individual classes of variable stars to try to identify interesting or unusual behaviour within a class, such as the recently discovered high-frequency ?? Scuti stars (Bedding et al. 2020). Another possibility is to extend our intuitive feature-based methods by adding more complex feature selection and classification algorithms. Such extensions are likely to improve our already strong classification performance, and strengthen results when applying our methods to even larger photometric surveys, such as that from TESS.

# ACKNOWLEDGEMENTS

We thank the Kepler team for providing such a wonderful data set. We gratefully acknowledge support from the Australian Research Council through DECRA grant DE180101104 and Discovery Project DP210103119, and from the Danish National Research Foundation (Grant DNRF106) through its funding for the Stellar Astrophysics Centre (SAC). TVR gratefully acknowledges support from the Research Foundation Flanders (FWO) under grant agreement number 12ZB620N.

# DATA AVAILABILITY

The data underlying this article are available at the Kepler Asteroseismic Science Operations Center (KASOC), at http://kasoc. phys.au.dk/. The hctsa software is available at https://github. com/benfulcher/hctsa.

# REFERENCES

Armstrong D. J., et al., 2016, MNRAS, 456, 2260   
Audenaert J., et al., 2021, AJ, 162, 209   
Ball N. M., Brunner R. J., 2010, International Journal of Modern Physics D, 19, 1049   
Balona L. A., 2013, MNRAS, 431, 2240   
Balona L. A., 2018, MNRAS, 479, 183   
Balona L. A., Holdsworth D. L., Cunha M. S., 2019, MNRAS, 487, 2117   
Baron D., 2019, Machine Learning in Astronomy: a practical overview (arXiv:1904.07248)

Bass G., Borne K., 2016, MNRAS, 459, 3721   
Bassi S., Sharma K., Gomekar A., 2021, Frontiers in Astronomy and Space Sciences, 8, 168   
Bedding T. R., et al., 2020, Nature, 581, 147   
Blomme J., et al., 2010, ApJ, 713, L204   
Blomme J., et al., 2011, MNRAS, 418, 96   
Bouabid M. P., Dupret M. A., Salmon S., Montalbán J., Miglio A., Noels A., 2013, MNRAS, 429, 2500   
Bowman D. M., Kurtz D. W., 2018, MNRAS, 476, 3169   
Briquet M., Hubrig S., De Cat P., Aerts C., North P., Schöller M., 2007, A&A, 466, 269   
Cabral J. B., Ramos F., Gurovich S., Granitto P. M., 2020, A&A, 642, A58   
Carrasco-Davis R., et al., 2019, PASP, 131, 108006   
Catelan M., Smith H. A., 2015, Pulsating Stars. Wiley-VCH   
Debosscher J., Blomme J., Aerts C., De Ridder J., 2011, A&A, 529, A89   
Fulcher B. D., Jones N. S., 2014, IEEE Trans. Knowl. Data Eng., 26, 3026   
Fulcher B. D., Jones N. S., 2017, Cell Sys., 5, 527   
Fulcher B. D., Little M. A., Jones N. S., 2013, J. Roy. Soc. Interface, 10   
Giles D. K., Walkowicz L., 2020, MNRAS, 499, 524   
Graff P., Feroz F., Hobson M. P., Lasenby A., 2014, MNRAS, 441, 1741   
Guzik J. A., 2021, Frontiers in Astronomy and Space Sciences, 8, 55   
Hastie T., Tibshirani R., Friedman J., 2009, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd edn. Springer   
Henderson T., Fulcher B. D., 2021, An Empirical Evaluation of Time-Series Feature Sets (arXiv:2110.10914)   
Hon M., Stello D., Yu J., 2017, MNRAS, 469, 4578   
Hon M., Stello D., Yu J., 2018a, MNRAS, 476, 3233   
Hon M., Stello D., Zinn J. C., 2018b, ApJ, 859, 64   
Hosenie Z., Lyon R., Stappers B., Mootoovaloo A., McBride V., 2020, MN-RAS, 493, 6050   
Ivezić Ž., Connelly A. J., Vanderplas J. T., Gray A., 2019, Statistics, Data Mining, and Machine Learning in Astronomy. Princeton University Press   
Jackiewicz J., 2021, Frontiers in Astronomy and Space Sciences, 7, 102   
Jamal S., Bloom J. S., 2020, ApJS, 250, 30   
Johnston K. B., Caballero-Nieves S. M., Petit V., Peter A. M., Haber R., 2019a, MNRAS, p. 2752   
Johnston K. B., Haber R., Caballero-Nieves S. M., Peter A. M., Petit V., Knote M., 2019b, Computational Astrophysics and Cosmology, 6, 4   
Kawahara H., Masuda K., 2019, AJ, 157, 218   
Kaye A. B., Handler G., Krisciunas K., Poretti E., Zerbi F. M., 1999, PASP, 111, 840   
Kgoadi R., Whittingham I., Engelbrecht C., 2019, in Griffin R. E., ed., IAU Symposium Vol. 339, Southern Horizons in Time-Domain Astronomy. pp 310–313, doi:10.1017/S1743921318002855   
Kirk B., et al., 2016, AJ, 151, 68   
Kurtz D., 2022, arXiv e-prints, p. arXiv:2201.11629   
Kuszlewicz J. S., Hekker S., Bell K. J., 2020, MNRAS, 497, 4843   
Le Saux A., Bugnet L., Mathur S., Breton S. N., García R. A., 2019, in Di Matteo P., Creevey O., Crida A., Kordopatis G., Malzac J., Marquette J. B., N’Diaye M., Venot O., eds, SF2A-2019: Proceedings of the Annual meeting of the French Society of Astronomy and Astrophysics. (arXiv:1906.09611)   
Lee U., 2021, MNRAS, 505, 1495   
Li G., Van Reeth T., Bedding T. R., Murphy S. J., Antoci V., 2019, MNRAS, 487, 782   
Li G., Van Reeth T., Bedding T. R., Murphy S. J., Antoci V., Ouazzani R.-M., Barbara N. H., 2020, MNRAS, 491, 3586   
Mathur S., et al., 2017, ApJS, 229, 30   
Matĳevič G., Prša A., Orosz J. A., Welsh W. F., Bloemen S., Barclay T., 2012, AJ, 143, 123   
McLachlan G. J., Peel D., 2000, Finite mixture models. Probability and Statistics – Applied Probability and Statistics Section Vol. 299, Wiley, New York   
McQuillan A., Mazeh T., Aigrain S., 2014, The Astrophysical Journal Supplement Series, 211, 24   
Molnár L., Plachy E., Juhász Á. L., Rimoldini L., 2018, A&A, 620, A127   
Mombarg J. S. G., Van Reeth T., Pedersen M. G., Molenberghs G., Bowman D. M., Johnston C., Tkachenko A., Aerts C., 2019, MNRAS, 485, 3248

Murphy S. J., Moe M., Kurtz D. W., Bedding T. R., Shibahashi H., Boffin H. M. J., 2018, MNRAS, 474, 4322   
Murphy S. J., Hey D., Van Reeth T., Bedding T. R., 2019, MNRAS, 485, 2380   
Murphy S. J., Barbara N. H., Hey D., Bedding T. R., Fulcher B. D., 2020, MNRAS, 493, 5382   
Nemec J. M., Cohen J. G., Ripepi V., Derekas A., Moskalik P., Sesar B., Chadid M., Bruntt H., 2013, ApJ, 773, 181   
Nielsen M. B., Gizon L., Schunker H., Karoff C., 2013, A&A, 557, L10   
Oort J. H., Plaut L., 1975, A&A, 41, 71   
Ouazzani R. M., Marques J. P., Goupil M. J., Christophe S., Antoci V., Salmon S. J. A. J., Ballot J., 2019, A&A, 626, A121   
Pashchenko I. N., Sokolovsky K. V., Gavras P., 2017, Monthly Notices of the Royal Astronomical Society, 475, 2326   
Paul S., Chattopadhyay T., 2022, arXiv e-prints, p. arXiv:2201.08755   
Pietrukowicz P., et al., 2017, Nature Astronomy, 1, 0166   
Sikora J., Wade G. A., Rowe J., 2020, MNRAS, 498, 2456   
Szklenár T., Bódi A., Tarczay-Nehéz D., Vida K., Marton G., Mező G., Forró A., Szabó R., 2020, ApJ, 897, L12   
Timmer J., Gantert C., Deuschl G., Honerkamp J., 1993, Biological Cybernetics, 70, 75   
Tsang B. T. H., Schultz W. C., 2019, ApJ, 877, L14   
Van Der Maaten L., Hinton G., 2008, The Journal of Machine Learning Research, 9, 2579   
Van Reeth T., et al., 2018, A&A, 618, A24   
Walker A. R., 1992, ApJ, 390, L81   
Wolff S. C., 1983, The A-type stars: problems and perspectives.. NASA SP-463, Washington D.C.   
Yu J., Huber D., Bedding T. R., Stello D., Hon M., Murphy S. J., Khanna S., 2018, ApJS, 236, 42   
Zhang K., Bloom J. S., 2021, MNRAS, 505, 515

This paper has been typeset from a TEX/LATEX file prepared by the author.