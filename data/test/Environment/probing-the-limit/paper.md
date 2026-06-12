# Probing the limit of hydrologic predictability with the Transformer network

Jiangtao Liu 1, Yuchen Bian 2 and Chaopeng Shen\*,1

1 Civil and Environmental Engineering, The Pennsylvania State University   
1 Amazon Search Science and AI   
\* Corresponding author: Chaopeng Shen, cshen@engr.psu.edu

# Abstract

For a number of years since its introduction to hydrology, recurrent neural networks like long short-term memory (LSTM) have proven remarkably difficult to surpass in terms of daily hydrograph metrics on known, comparable benchmarks. Outside of hydrology, Transformers have now become the model of choice for sequential prediction tasks, making it a curious architecture to investigate. Here, we first show that a vanilla Transformer architecture is not competitive against LSTM on the widely benchmarked CAMELS dataset, and lagged especially for the high-flow metrics due to short-term processes. However, a recurrence-free variant of Transformer can obtain mixed comparisons with LSTM, producing the same Kling-Gupta efficiency coefficient (KGE), along with other metrics. The lack of advantages for the Transformer is linked to the Markovian nature of the hydrologic prediction problem. Similar to LSTM, the Transformer can also merge multiple forcing dataset to improve model performance. While the Transformer results are not higher than current state-of-the-art, we still learned some valuable lessons: (1) the vanilla Transformer architecture is not suitable for hydrologic modeling; (2) the proposed recurrence-free modification can improve Transformer performance so future work can continue to test more of such modifications; and (3) the prediction limits on the dataset should be close to the current state-of-the-art model. As a non-recurrent model, the Transformer may bear scale advantages for learning from bigger datasets and storing knowledge. This work serves as a reference point for future modifications of the model.

# Introduction

Rainfall-runoff modeling is essential for flood prediction, water resource management, and environmental protection (Hrachowitz & Clark, 2017). Rainfall-runoff modeling is a critical aspect of hydrology, as it models the intricate relationships between precipitation, watershed characteristics, and streamflow. The introduction of long short-term memory (LSTM) networks marked a significant advancement in this field for numerous variables of interest including soil moisture (Fang & Shen, 2017; Feng et al., 2020), streamflow (Feng et al., 2020, 2021; Kratzert et al., 2019; Xiang & Demir, 2020), water temperature (Rahmani, Lawson, et al., 2021; Rahmani, Shen, et al., 2021), and groundwater levels (Afzaal et al., 2020; Wunsch et al., 2022). For these applications, LSTM consistently outperformed traditional models such as the autoregressive integrated moving average (ARIMA) and support vector machines (SVM) (Papacharalampous et al., 2018). The LSTM's ability to learn many-step dependencies and handle variable-length input sequences has proven particularly advantageous in capturing the inherent complexity and non-stationarity of hydrological processes (Hochreiter & Schmidhuber, 1997).

As a recurrent neural network (RNN), LSTM has to step through time steps one by one, accumulate changes to states after each step, and apply the neural networks many times, which lead to some limitations. The recurrent nature lends RNNs to an issue called vanishing gradient (Hochreiter, 1991; Hochreiter et al., 2001), where the gradient of the loss with respect to the network weights is very small, making their training extremely slow. This issue limits the length of the training sequence, and reduces the impact of inputs in the longer-term past on present predictions. This could be one of the reasons why baseflow was previously identified as a limitation (Feng et al., 2020). Even though LSTM was developed to mitigate this issue and can suppress it better than the original RNNs, it is not immune to it. Furthermore, recurrence means these time steps must be taken in sequence --- the time steps cannot be run in parallel. This poses a limitation in parallel efficiency and thus the scale of data that can be trained on.

In most applications outside hydrology, the transformer architecture (Vaswani et al., 2017) has demonstrated superior performance over LSTM networks in various domains, including machine translation, speech recognition (Karita et al., 2019), natural language processing and sentiment analysis (Devlin et al., 2019), question answering (Rajpurkar et al., 2018), computer vision (Carion et al., 2020), protein structure prediction (Rives et al., 2021), and music generation (Huang et al., 2018). Transformer model uses the attention mechanism, where each word (or “input token”) is transformed into three different kinds of information: a 'query' that asks how relevant other words are to it, a 'key' that responds to others' queries about its relevance, and a 'value' that carries the word's actual meaning. The model calculates the relevancies between the query and keys of all words, then combines the values of the most relevant words to understand the current word better. With LSTM, the most recent input tokens are always more important than further-away ones, whereas a Transformer could learn to put more focus to further-away tokens (Dehghani et al., 2019; Raganato & Tiedemann, 2018), which makes it ideal for language modeling. Moreover, as it does not have recurrence, a Transformer can run the time steps in parallel and can scale up in parallel computation when more data and more GPUs are available. Considering such success, there should be a heightened interest in harnessing transformers for hydrologic applications. However, only a few studies have employed transformers in hydrology (Li & Yang, 2019; Xu et al., 2021) (which focused on near-term forecast), and no work reported its results on standardized, well-understood benchmark problems like the Catchment Attributes and Meteorology for Large-sample Studies (CAMELS) dataset (Addor et al., 2017, 2017; Newman et al., 2014). It is then intriguing if its advantages over recurrent networks apply to the natural systems, which lacked the irregular sequential structure commonly found in languages.

While some past studies have claimed some architectures’ superior performance compared to LSTM, most of the time the conclusions were highly conditional on using a small dataset for benchmarking (Abed et al., 2022; Amanambu et al., 2022; Ghobadi & Kang, 2022), or using procedures and configurations, e.g., training and test periods, sites, and forcing data, different from published benchmarks (Yin et al., 2022, 2023), or on a case study which were not tested independently by other teams (Koya & Roy, 2023; Liu et al., 2022). In the interest of reproducibility and comparability which underpin scientific progress, it is a good idea to benchmark under the same conditions, on the same (reasonably large) dataset. In addition, data-driven deep learning models enjoy the feature of “data synergy”, where larger and more diverse data leads to stronger and more robust models (Fang et al., 2022; Kratzert et al., 2020; Pasquiou et al., 2022; Yang et al., 2023). Thus small-data comparison results may be no longer valid for the case with more data. Thus far, on the CAMELS dataset (Addor et al., 2017; Newman et al., 2014), both Kratzert et al. (2019) and Feng et al., (2021) reported very comparable metrics with LSTM --- 0.72 for 571 basins with the NLDAS forcing alone, making this a reliable benchmark that is so far not exceeded by other models. Furthermore, Kratzert simultaneously employed multiple forcing dataset (NLDAS, Maurer and Daymet) for LSTM and obtained a Kling-Gupta model efficiency coefficient (KGE) of 0.80, which is the record on this dataset.

In this study, we investigate the performance of the Transformer architecture in rainfall-runoff modeling compared to LSTM, using the CAMELS dataset. We analyze the performance of single models and ensembles for both architectures and examine the models' ability to handle multiple forcings and mixed forcing cases. Even though we had expected the difference to be small compared to LSTM, we aim at establishing a reference point where future studies can contrast and compare to. Our findings contribute to the understanding of the strengths and limitations of both LSTM and Transformer models in hydrological predictions and highlight the potential of the Transformer as an alternative and scalable solution for hydrologic modeling.

Results and Discussion   
![](images/3c9c0f34beb9053c0050b50a9255e3ef607883bf8aad057ab66ea895d862602f.jpg)

<details>
<summary>line</summary>

| NSE   | LSTM single-forcing | LSTM multi-forcing | Transformer single-forcing | Transformer multi-forcing | SAC_SMA |
|-------|---------------------|--------------------|-----------------------------|----------------------------|---------|
| -0.2  | 0.0                 | 0.0                | 0.0                         | 0.0                        | 0.0     |
| 0.0   | 0.0                 | 0.0                | 0.0                         | 0.0                        | 0.0     |
| 0.2   | 0.0                 | 0.0                | 0.0                         | 0.0                        | 0.0     |
| 0.4   | 0.1                 | 0.1                | 0.1                         | 0.1                        | 0.1     |
| 0.6   | 0.3                 | 0.3                | 0.3                         | 0.3                        | 0.3     |
| 0.8   | 0.7                 | 0.7                | 0.7                         | 0.7                        | 0.7     |
| 1.0   | 1.0                 | 1.0                | 1.0                         | 1.0                        | 1.0     |
</details>

![](images/42d941e2656a562183a638255dfabf80a2c2cacf8e0d8fd05734a5fe2085f854.jpg)

<details>
<summary>line</summary>

| KGE   | LSTM single-forcing | LSTM multi-forcing | Transformer single-forcing | Transformer multi-forcing | SAC_SMA |
|-------|---------------------|--------------------|-----------------------------|----------------------------|---------|
| -0.2  | 0.0                 | 0.0                | 0.0                         | 0.0                        | 0.0     |
| 0.0   | 0.0                 | 0.0                | 0.0                         | 0.0                        | 0.0     |
| 0.2   | 0.0                 | 0.0                | 0.0                         | 0.0                        | 0.0     |
| 0.4   | 0.1                 | 0.1                | 0.1                         | 0.1                        | 0.1     |
| 0.6   | 0.3                 | 0.3                | 0.3                         | 0.3                        | 0.3     |
| 0.8   | 0.7                 | 0.7                | 0.7                         | 0.7                        | 0.7     |
| 1.0   | 1.0                 | 1.0                | 1.0                         | 1.0                        | 1.0     |
</details>

![](images/d21c94410bf93ffc60c90c6d30dbbbfb2d2b3d0a43dd8df28f47829c908dfa61.jpg)

<details>
<summary>line</summary>

| FLV  | LSTM single-forcing | LSTM multi-forcing | Transformer single-forcing | Transformer multi-forcing | SAC_SMA |
| ---- | ------------------- | ------------------ | -------------------------- | ------------------------- | ------- |
| -100 | 0.0                 | 0.0                | 0.0                        | 0.0                       | 0.0     |
| -50  | 0.1                 | 0.1                | 0.1                        | 0.1                       | 0.2     |
| 0    | 0.4                 | 0.4                | 0.4                        | 0.4                       | 0.4     |
| 50   | 0.6                 | 0.7                | 0.8                        | 0.9                       | 0.6     |
| 100  | 0.7                 | 0.8                | 0.9                        | 0.95                      | 0.7     |
| 150  | 0.8                 | 0.85               | 0.95                       | 0.98                      | 0.8     |
| 200  | 0.85                | 0.9                | 0.98                       | 0.99                      | 0.85    |
</details>

![](images/2df2e2c655b18709947351300fe4ec903caf45ea13be4bc58a0ed4ae79e1752e.jpg)

<details>
<summary>line</summary>

| FHV  | LSTM single-forcing | LSTM multi-forcing | Transformer single-forcing | Transformer multi-forcing | SAC_SMA |
|------|---------------------|--------------------|----------------------------|---------------------------|---------|
| -60  | 0.0                 | 0.0                | 0.0                        | 0.0                       | 0.0     |
| -40  | 0.1                 | 0.1                | 0.1                        | 0.1                       | 0.1     |
| -20  | 0.4                 | 0.5                | 0.4                        | 0.3                       | 0.6     |
| 0    | 0.8                 | 0.9                | 0.8                        | 0.7                       | 0.9     |
| 20   | 0.95                | 0.98               | 0.95                       | 0.9                       | 0.98    |
| 40   | 1.0                 | 1.0                | 1.0                        | 1.0                       | 1.0     |
</details>

Figure 1. Comparative analysis of Cumulative Density Function(CDF) across various models including LSTM, Transformer, and SAC-SMA, with unit in mm/day. The model encompass single and multi-forcing data for models. The training period ranged from 1 October 1999 to 30 September 2008, while the testing period ranged from 1 October 1989 to 30 September 1999. The figure depict the following comparisons: (a) NSE vs CDF, (b) KGE vs CDF, (c) FLV vs CDF, and (d) FHV vs CDF. Single-forcing models were implemented on a set of 671 basins, whereas multi-forcing models were applied to a subset of 531 basins.

Table 1. Comparative performance metrics for single and multi-forcing experiments of LSTM, vanilla transformer, and modified transformer models. Training dates for the models span from 1 October 1999 to 30 September 2008, while testing dates cover 1 October 1989 to 30 September 1999. We conducted an evaluation of single forcing on 671 basins and multi-forcing on 531 basins, employing the LSTM model results from Kratzert et al., 2019, originally evaluated on 531 basins. To broaden our insights into the impacts of single-forcing on the entire dataset and a fair comparison, we retrained their model on an expanded set of 671 basins with single NLDAS dataset. These numbers are only very slightly different from Kratzert et al., 2019. The means for NSE, KGE, FHV, and FLV are averages from the 10 different ensemble members, each with a different random seed, while the standard deviations for FHV and FLV are calculated for the ensemble members. 

<table><tr><td rowspan="2"></td><td colspan="3">NLDAS</td><td colspan="3">Multi-forcing</td></tr><tr><td>LSTM</td><td>Vanilla Transformer</td><td>Modified Transformer</td><td>LSTM</td><td>Vanilla Transformer</td><td>Modified Transformer</td></tr><tr><td>KGE (mean)</td><td>0.73 ±0.003</td><td>0.71 ±0.007</td><td>0.74 ±0.007</td><td>0.80 ±0.004</td><td>0.77 ±0.016</td><td>0.80 ±0.007</td></tr><tr><td>FHV (mean)</td><td>-17.49 ±0.58</td><td>-26.66 ±2.83</td><td>-18.00 ±2.94</td><td>-11.91 ±0.549</td><td>-21.54 ±2.64</td><td>-9.19 ±4.01</td></tr><tr><td>FLV (mean)</td><td>-2.82</td><td>3.31</td><td>2.28</td><td>2.57</td><td>0.77</td><td>2.72</td></tr><tr><td></td><td>±8.15</td><td>±2.34</td><td>±4.24</td><td>±4.072</td><td>±1.65</td><td>±2.41</td></tr></table>

For the single-forcing CAMELS benchmark (671 basins), the vanilla Transformer is outperformed by LSTM (Table 1; Figure 1). Overall, the vanilla Transformer fell behind LSTM in all metrics, although not by much. Since KGE is a reliable performance metric, we choose to focus on this metric (Knoben et al., 2018, 2019), for which the vanilla Transformer reported 0.71, compared to 0.73 for the LSTM. These results suggest that, without modification, the vanilla Transformer is missing some critical ability to simulate hydrologic processes.

The vanilla Transformer’s under-performance is a curious case as it has been widely recognized that “attention is all you need” (Vaswani et al., 2017) in sequential modeling, and we have several interpretations. First, it is possible that the data size is too small here and Transformer’s advantage would emerge for larger quantities of data. Second, the natural hydrologic process is a “Markovian” system where the states at a time, rather than more remotely-in-the-past steps, completely determine the system’s trajectory for the future time steps along with the forcings. To be more concrete, the soil moisture today, rather than any previous days’, would have far more impact on tomorrow’s discharge. This is in strong contrast to human languages where the order of the words can often be inverted, which would favor the attention-based Transformer architecture. Third, the accumulation of moisture and its nonlinear interactions makes memory effects important, while the Transformer does not have memory and is not necessarily strong at capturing the effects of memory. Regardless of the reason, the results mean that the vanilla Transformer is not optimal and further changes are needed to model the natural systems.

The modified Transformer, on the other hand, shows very similar median metrics as LSTM while the variabilities among ensemble members are different. Its KGE (0.74) is slightly higher than LSTM (0.73), and the differences in FLV and FHV from LSTM’s values are too small to call an advantage considering their variabilities. The ensemble standard deviation of KGE is 0.003 with LSTM and 0.007 with the modified Transformer. We notice that the LSTM has smaller ensemble standard deviation FHV than the modified Transformer, while the opposite is true for FLV. The ensemble standard deviation of median FHV is 0.58 for LSTM and 2.94 for the modified Transformer. That value for the FLV is 8.15 for the LSTM and 4.24 for the modified Transformer. This suggests while we obtain very similar overall metrics, the LSTM and the modified Transformer preferentially address different parts of the hydrograph. LSTM more reliably focus on the high-flow regime (quantified by the ensemble standard deviation of FHV) than the modified Transformer but the latter can more reliably capture the long-term dependence (quantified by ensemble standard deviation of FLV). It seems there is some tradeoff for the different flow regimes.

The multiforcing experiment generally shows similar patterns: the vanilla Transformer falls behind the other two models, which have very similar ensemble-mean performance metrics but different ensemble standard deviations. The high KGE (0.80) and slightly better-than-LSTM FHV (9.19) for the modified Transformer demonstrates that it, too, is able to fuse different forcing dataset as just LSTM. Just as the NLDAS experiment, the modified Transformer has a larger stochastic variability (quantified by ensemble standard deviation) FHV but smaller variability for FLV. Because both FHV and FLV have improved compared to the single forcing, the modified Transformer was able to utilize the short-term and long-term dependencies of multiple forcing datasets. For one particular ensemble member (random seed), the cumulative density plot shows very similar curves between the modified Transformer and LSTM.

The high agreement between the two model architectures, both of which are state-of-the-art, suggest that we are likely at or very close to the predictive limit of the CAMELS dataset for this test (temporal test). We suspect, unless we bring in new information, it is highly unlikely for other models to produce noticeable advantages beyond these two models on this dataset, for the tests presented here. Errors with forcing, attribute and discharge data would prevent high performance. It should be mentioned that for another test, e.g., prediction in ungauged regions or spatial extrapolation test, physics-informed hybrid models (called differentiable models) can actually outperform LSTM (Feng, Beck, et al., 2022; Feng, Liu, et al., 2022; Tsai et al., 2021). Moreover, several issues surrounding the CAMELS dataset include using basin-average attributes that cannot resolve subbasin-level spatial heterogeneity, using daily precipitation that does not represent hourly rainfall intensity, a fraction of basins with major reservoirs, and the inclusion some overly large basins.

Nevertheless, exactly because the Transformer does not have time integration, it can be trained in a highly parallel fashion and is suited to learning from large amounts of data. As the amount of data and the amount of neurons increase, it is possible to observe emergent behaviors of intelligence (Bubeck et al., 2023). This is a property that is worth further exploring in future studies in hydrology and geosciences. We leave to future work how to incorporate more data with the modified Transformer and test the model for spatial extrapolation (under data-scarce scenario) and temporal extrapolation (for multidecadal projection of trends).

# Conclusion

As an initial step, we compared a vanilla Transformer encoder and a modified Transformer to the current state-of-the-art model, LSTM, on the CAMELS benchmark. However, the vanilla Transformer seems to miss some critical functionality so that it is not optimal for simulating discharge. The modified Transformer with no recurrent connection obtains essentially the same metric (albeit only with a slight advantage) as LSTM and works. This means we can continue to search for better architecture to further improve its performance and suitableness for natural physical systems. Our current setup may not be optimal yet. Nevertheless, the differences are overall small between the models, suggesting that we are already close to the optimum for this dataset and this test and more expansion of dataset will be required. We argue the modified Transformer is a viable alternative to LSTM and may find advantages for future, larger datasets.

# Bibliography

Abed, M., Imteaz, M. A., Ahmed, A. N., & Huang, Y. F. (2022). A novel application of transformer

neural network (TNN) for estimating pan evaporation rate. Applied Water Science, 13(2), 31. https://doi.org/10.1007/s13201-022-01834-w   
Addor, N., Newman, A. J., Mizukami, N., & Clark, M. P. (2017). Catchment attributes for large-sample studies [Data set]. UCAR/NCAR. https://doi.org/10.5065/D6G73C3Q   
Afzaal, H., Farooque, A. A., Abbas, F., Acharya, B., & Esau, T. (2020). Groundwater estimation from major physical hydrology components using artificial neural networks and deep learning. Water, 12(1), 5. https://doi.org/10.3390/w12010005   
Amanambu, A. C., Mossa, J., & Chen, Y.-H. (2022). Hydrological Drought Forecasting Using a Deep Transformer Model. Water, 14(22), 3611. https://doi.org/10.3390/w14223611   
Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., et al. (2023, March 27). Sparks of Artificial General Intelligence: Early experiments with GPT-4. arXiv. https://doi.org/10.48550/arXiv.2303.12712   
Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020, May 28). End-to-End Object Detection with Transformers. arXiv. https://doi.org/10.48550/arXiv.2005.12872   
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019, May 24). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv. https://doi.org/10.48550/arXiv.1810.04805   
Fang, K., & Shen, C. (2017). Full-flow-regime storage-streamflow correlation patterns provide insights into hydrologic functioning over the continental US. Water Resources Research, 53(9), 8064–8083. https://doi.org/10.1002/2016WR020283   
Fang, K., Kifer, D., Lawson, K., Feng, D., & Shen, C. (2022). The data synergy effects of time-series deep learning models in hydrology. Water Resources Research, 58(4), e2021WR029583. https://doi.org/10.1029/2021WR029583   
Feng, D., Fang, K., & Shen, C. (2020). Enhancing streamflow forecast and extracting insights using long-short term memory networks with data integration at continental scales.

Water Resources Research, 56(9), e2019WR026793. https://doi.org/10.1029/2019WR026793   
Feng, D., Lawson, K., & Shen, C. (2021). Mitigating prediction error of deep learning streamflow models in large data-sparse regions with ensemble modeling and soft data. Geophysical Research Letters, 48(14), e2021GL092999. https://doi.org/10.1029/2021GL092999   
Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water Resources Research, 58(10), e2022WR032404. https://doi.org/10.1029/2022WR032404   
Feng, D., Beck, H., Lawson, K., & Shen, C. (2022). The suitability of differentiable, learnable hydrologic models for ungauged regions and climate change impact assessment. Hydrology and Earth System Sciences Discussions, 1–28. https://doi.org/10.5194/hess-2022-245   
Ghobadi, F., & Kang, D. (2022). Improving long-term streamflow prediction in a poorly gauged basin using geo-spatiotemporal mesoscale data and attention-based deep learning: A comparative study. Journal of Hydrology, 615, 128608. https://doi.org/10.1016/j.jhydrol.2022.128608   
Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen. Institut fur Informatik, Technische Universitat, Munchen, 1-150. Retrieved from https://www.semanticscholar.org/paper/Untersuchungen-zu-dynamischen-neuronalen-Ne tzen-Hochreiter/3f3d13e95c25a8f6a753e38dfce88885097cbd43   
Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735   
Hochreiter, S., Bengio, Y., Frasconi, P., & Jürgen Schmidhuber. (2001). Gradient Flow in Recurrent Nets: The Difficulty of Learning Long-Term Dependencies. In S. C. Kremer & J. F. Kolen (Eds.), A Field Guide to Dynamical Recurrent Neural Networks (pp.

237–244). Piscataway, NJ, USA: IEEE Press.   
Hrachowitz, M., & Clark, M. P. (2017). HESS Opinions: The complementary merits of competing modelling philosophies in hydrology. Hydrology and Earth System Sciences, 21(8), 3953–3973. https://doi.org/10.5194/hess-21-3953-2017   
Huang, C.-Z. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, C., et al. (2018, December 12). Music Transformer. arXiv. https://doi.org/10.48550/arXiv.1809.04281   
Karita, S., Chen, N., Hayashi, T., Hori, T., Inaguma, H., Jiang, Z., et al. (2019). A Comparative Study on Transformer vs RNN in Speech Applications. In 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) (pp. 449–456). https://doi.org/10.1109/ASRU46091.2019.9003750   
Knoben, W. J. M., Woods, R. A., & Freer, J. E. (2018). A Quantitative Hydrological Climate Classification Evaluated With Independent Streamflow Data. Water Resources Research, 54(7), 5088–5109. https://doi.org/10.1029/2018WR022913   
Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores. Hydrology and Earth System Sciences, 23(10), 4323–4331. https://doi.org/10.5194/hess-23-4323-2019   
Koya, S. R., & Roy, T. (2023, May 20). Temporal Fusion Transformers for Streamflow Prediction: Value of Combining Attention with Recurrence. arXiv. https://doi.org/10.48550/arXiv.2305.12335   
Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., & Nearing, G. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. Hydrology and Earth System Sciences, 23(12), 5089–5110. https://doi.org/10.5194/hess-23-5089-2019   
Kratzert, F., Klotz, D., Hochreiter, S., & Nearing, G. S. (2020). A note on leveraging synergy in multiple meteorological datasets with deep learning for rainfall-runoff modeling. Hydrology and Earth System Sciences, 2020, 1–26.

https://doi.org/10.5194/hess-2020-221   
Li, Y., & Yang, J. (2019). Hydrological time series prediction model based on attention-LSTM neural network. In Proceedings of the 2019 2nd International Conference on Machine Learning and Machine Intelligence (pp. 21–25). New York, NY, USA: Association for Computing Machinery. https://doi.org/10.1145/3366750.3366756   
Liu, C., Liu, D., & Mu, L. (2022). Improved Transformer Model for Enhanced Monthly Streamflow Predictions of the Yangtze River. IEEE Access, 10, 58240–58253. https://doi.org/10.1109/ACCESS.2022.3178521   
Newman, A. J., Sampson, K., Clark, M. P., Bock, A., Viger, R. J., & Blodgett, D. (2014). A large-sample watershed-scale hydrometeorological dataset for the contiguous USA. Boulder. https://doi.org/10.5065/D6MW2F4D   
Papacharalampous, G., Tyralis, H., & Koutsoyiannis, D. (2018). One-step ahead forecasting of geophysical processes within a purely statistical framework. Geoscience Letters, 5(1), 12. https://doi.org/10.1186/s40562-018-0111-1   
Pasquiou, A., Lakretz, Y., Hale, J., Thirion, B., & Pallier, C. (2022, July 7). Neural Language Models are not Born Equal to Fit Brain Data, but Training Helps. arXiv. https://doi.org/10.48550/arXiv.2207.03380   
Rahmani, F., Shen, C., Oliver, S., Lawson, K., & Appling, A. (2021). Deep learning approaches for improving prediction of daily stream temperature in data-scarce, unmonitored, and dammed basins. Hydrological Processes, 35(11), e14400. https://doi.org/10.1002/hyp.14400   
Rahmani, F., Lawson, K., Ouyang, W., Appling, A., Oliver, S., & Shen, C. (2021). Exploring the exceptional performance of a deep learning stream temperature model and the value of streamflow data. Environmental Research Letters, 16(2), 024025. https://doi.org/10.1088/1748-9326/abd501   
Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don’t Know: Unanswerable Questions

for SQuAD. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 784–789). Melbourne, Australia: Association for Computational Linguistics. https://doi.org/10.18653/v1/P18-2124   
Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences, 118(15), e2016239118. https://doi.org/10.1073/pnas.2016239118   
Tsai, W.-P., Feng, D., Pan, M., Beck, H., Lawson, K., Yang, Y., et al. (2021). From calibration to parameter learning: Harnessing the scaling effects of big data in geoscientific modeling. Nature Communications, 12(1), 5988. https://doi.org/10.1038/s41467-021-26107-z   
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., et al. (2017, December 5). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762   
Wunsch, A., Liesch, T., & Broda, S. (2022). Deep learning shows declining groundwater levels in Germany until 2100 due to climate change. Nature Communications, 13(1), 1221. https://doi.org/10.1038/s41467-022-28770-2   
Xiang, Z., & Demir, I. (2020). Distributed long-term hourly streamflow predictions using deep learning – A case study for State of Iowa. Environmental Modelling & Software, 131, 104761. https://doi.org/10.1016/j.envsoft.2020.104761   
Xu, Z., Wang, S., Stanislawski, L. V., Jiang, Z., Jaroenchai, N., Sainju, A. M., et al. (2021). An attention U-Net model for detection of fine-scale hydrologic streamlines. Environmental Modelling & Software, 140, 104992. https://doi.org/10.1016/j.envsoft.2021.104992   
Yang, E., Li, M. D., Raghavan, S., Deng, F., Lang, M., Succi, M. D., et al. (2023). Transformer versus traditional natural language processing: how much data is enough for automated radiology report classification? The British Journal of Radiology, 20220769. https://doi.org/10.1259/bjr.20220769

Yin, H., Guo, Z., Zhang, X., Chen, J., & Zhang, Y. (2022). RR-Former: Rainfall-runoff modeling based on Transformer. Journal of Hydrology, 609, 127781. https://doi.org/10.1016/j.jhydrol.2022.127781   
Yin, H., Zhu, W., Zhang, X., Xing, Y., Xia, R., Liu, J., & Zhang, Y. (2023). Runoff predictions in new-gauged basins using two transformer-based models. Journal of Hydrology, 622, 129684. https://doi.org/10.1016/j.jhydrol.2023.129684