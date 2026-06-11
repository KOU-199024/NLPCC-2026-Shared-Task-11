# LINEAR REPRESENTATIONS OF POLITICAL PERSPEC-TIVE EMERGE IN LARGE LANGUAGE MODELS

Junsol Kim

University of Chicago junsol@uchicago.edu

James Evans

University of Chicago Google jevans@uchicago.edu jamesaevans@google.com

Aaron Schein

University of Chicago schein@uchicago.edu

# ABSTRACT

Large language models (LLMs) have demonstrated the ability to generate text that realistically reflects a range of different subjective human perspectives. This paper studies how LLMs are seemingly able to reflect more liberal versus more conservative viewpoints among other political perspectives in American politics. We show that LLMs possess linear representations of political perspectives within activation space, wherein more similar perspectives are represented closer together. To do so, we probe the attention heads across the layers of three open transformerbased LLMs (Llama-2-7b-chat, Mistral-7b-instruct, Vicuna-7b). We first prompt models to generate text from the perspectives of different U.S. lawmakers. We then identify sets of attention heads whose activations linearly predict those lawmakers’ DW-NOMINATE scores, a widely-used and validated measure of political ideology. We find that highly predictive heads are primarily located in the middle layers, often speculated to encode high-level concepts and tasks. Using probes only trained to predict lawmakers’ ideology, we then show that the same probes can predict measures of news outlets’ slant from the activations of models prompted to simulate text from those news outlets. These linear probes allow us to visualize, interpret, and monitor ideological stances implicitly adopted by an LLM as it generates open-ended responses. Finally, we demonstrate that by applying linear interventions to these attention heads, we can steer the model outputs toward a more liberal or conservative stance. Overall, our research suggests that LLMs possess a high-level linear representation of American political ideology and that by leveraging recent advances in mechanistic interpretability, we can identify, monitor, and steer the subjective perspective underlying generated text.

# 1 INTRODUCTION

Large language models (LLMs) have demonstrated the ability to generate text that reflects a range of different subjective perspectives (Argyle et al., 2023b; Gao et al., 2024). In particular, a growing body of recent work has explored LLMs’ seeming ability to generate text that realistically reflects a range of political perspectives on voting preferences and policy issues (Argyle et al., 2023b; Santurkar et al., 2023; Wu et al., 2023; 2024; O’Hagan & Schein, 2023; Kozlowski et al., 2024; Kim & Lee, 2023; Bernardelle et al., 2024). Leveraging this capability, researchers and practitioners have deployed LLMs for a range of tasks, including the development of personalized agents that engage in political debates with humans (Hackenburg & Margetts, 2024; Argyle et al., 2023a; Costello et al., 2024; Bai et al., 2023), as well as the implementation of agent-based models intended to simulate human-like behaviors and interactions (Andreas, 2022; Törnberg et al., 2023; Park et al., 2023; 2024a; Gao et al., 2024; Charness et al., 2023; Hewitt et al., 2024).

This paper examines whether LLMs possess general representations of political perspective in activation space, whether such representations are linear, and whether they can be used to steer model outputs. Specifically, we show that LLMs possess a linear representation of the “liberal–conservative” political axis in American politics. It is widely believed for LLMs that “important” concepts are encoded linearly as directions in activation space (Mikolov et al., 2013; Nanda et al., 2023; Elhage et al., 2022; Gurnee & Tegmark, 2024; Park et al., 2024b). Under this definition, LLMs have been

# User:

Write an essay on immigration in the U.S.

# Assistant:

On the one hand , those who support immigration argue that it can bring many benefits to a country . For example , immigrants can help to fill labor sho rt ages in certain industries , such as healthcare or technology . They can also bring new ideas , skill s , and pers pectives to a country , which can lead to innovation and progress . Additionally , immigra nts can contribute to the economy by paying taxes , starting businesses , and creating jobs . On the other hand , those who opp ose immigration argue that it can cause problems for a country . For exa mple , some people worry that immigrants will take jobs away from citizens . Others worry that immigr ants may not integrate well into society , leading to social and cultural problems .

# User:

Write an essay on abortion in the U.S.

# Assistant:

On one hand , pro - choice advoc ates argue that a woman has the right to make decisions about her own body and th at abortion should be accessible and safe . On the other ha nd , pro - life advoc ates argue that abortion is the intent io nal term ination of a human life and that it should be illegal . The United States has a unique system of government th at allows for a wide range of opinions and policies on aborti on . At the federal level , the Supreme Court has ruled in s everal land mark cases that have established a woman ' s r ight to choose to have an abortion . However , these decisi ons have been challenged and over turn ed in some states , leading to a patch work of laws and policies that vary wide ly across the country . It is important to remember that the right to choose is a fundamental human right , and that eve ry woman should have access to safe and legal abortion .

Figure 1: Excerpts from essays generated by Mistral-7b-instruct on policy issues (e.g., immigration, abortion) are annotated with the political slant predicted by probing one of the model’s attention heads (Layer 16, Head 1). This attention head was among the most predictive heads with the highest Spearman correlation in predicting the political ideology of U.S. lawmakers. Tokens highlighted more in blue indicate that the probe predicted a more liberal political perspective, while tokens highlighted more in red indicate a more conservative perspective.

shown to possess linear representations of various high-level concepts, such as sentiment (e.g., positive–negative) (Tigges et al., 2023), space (e.g., North–South) (Gurnee & Tegmark, 2024; Nanda et al., 2023), time (e.g., past–present) (Gurnee & Tegmark, 2024), humor (Von Rütte et al., 2024), language (Bricken et al., 2023), topic (Turner et al., 2023), truth (Marks & Tegmark, 2024; Li et al., 2023), and safety (Arditi et al., 2024), among other fundamental concepts (Gurnee & Tegmark, 2024; Nanda et al., 2023; Bricken et al., 2023). To our knowledge, this paper is the first to investigate whether LLMs possess linear representations of political perspective.

We prompt LLMs to generate text from the perspectives of different U.S. lawmakers and then train linear probes to predict these lawmakers’ DW-NOMINATE scores based on the activations of the models’ attention heads. DW-NOMINATE is a widely used and validated measure of lawmakers’ positions along the liberal–conservative axis in American politics (Poole & Rosenthal, 1985; Poole, 2005). For three different open LLMs (Llama-2-7b-chat, Mistral-7b-instruct-v0.1, Vicuna-7b-v1.5; see Appendix A.1 for model descriptions), we identify multiple attention heads that linearly represent political slant from liberal to conservative. More specifically, we reveal that linear probes on these attention heads are highly predictive of DW-NOMINATE scores for held-out lawmakers, and performance does not improve when using non-linear probes (Section 3). Additionally, we show that when models are prompted to simulate the perspectives of different news outlets (e.g., FOX News or NBC), the same linear probes trained to predict lawmakers’ DW-NOMINATE scores are also highly predictive of established measures of the news outlets’ political slant (Section 5). We demonstrate the usefulness and validity of these trained probes in two ways: (1) monitoring and (2) steering the political slant of LLM outputs. First, we show that these activation patterns can be used to detect the ideological slant implicitly adopted by an LLM as it generates open-ended responses, as shown in Figure 1 (Section 4). Second, by targeting these attention heads for causal intervention, we demonstrate that LLM responses can be steered toward more liberal or conservative perspectives without additional prompt engineering or fine-tuning (Section 6). Overall, our research contributes to a growing body of work that identifies linear representations and intervenes on them to monitor and simulate text from different subjective perspectives.

# 2 PRELIMINARIES

In this section, we define notation and provide relevant background on the architecture of transformerbased LLMs and probing methodology for discovering representations of concepts in LLMs.

Transformer-based LLMs LLMs generate text by sampling iteratively from a categorical distribution over the next token $w _ { t }$ given input tokens ${ \pmb w } _ { < t }$ . This distribution can be written as

$$
P (w _ {t} = v \mid \boldsymbol {w} _ {<   t}) \propto \exp (\boldsymbol {u} _ {v} ^ {\top} \boldsymbol {r} _ {L}) \tag {1}
$$

where ${ \pmb u } _ { v } \in \mathbb { R } ^ { D }$ is the unembedding of possible token $v ,$ and $\pmb { r } _ { L } \in \mathbb { R } ^ { D }$ is the final vector in the transformer’s “residual stream” (Elhage et al., 2021), which evolves over layers $\ell = 1 , \ldots , L$ as:

$$
\boldsymbol {r} _ {\ell} = \boldsymbol {r} _ {\ell - 1} + \sum_ {h = 1} ^ {H} Q _ {\ell , h} \boldsymbol {x} _ {\ell , h} + \mathrm{MLP} _ {\ell} \left(\boldsymbol {r} _ {\ell - 1} + \sum_ {h = 1} ^ {H} Q _ {\ell , h} \boldsymbol {x} _ {\ell , h}\right) \tag {2}
$$

Here the dependence on ${ \pmb w } _ { < t }$ is implicit via $\mathbf { \boldsymbol { r } } _ { 0 }$ , which encodes the input tokens before any transformer layers are applied. We refer to $\pmb { x } _ { \ell , h } \in \mathbb { R } ^ { d _ { \ell , h } }$ as the activation of attention head h in layer $\ell ,$

$$
\boldsymbol {x} _ {\ell , h} = \mathrm{ATTN} _ {\ell , h} (P _ {\ell , h} \boldsymbol {r} _ {\ell - 1}) \tag {3}
$$

which we highlight because it will be the target of this paper’s probing studies. The representation of a transformer layer in Equations (2) and $( 3 ) ^ { 1 }$ involves weight matrices $P _ { \ell , h } \in \dot { \mathbb R } ^ { d _ { \ell , h } \times D }$ and $Q _ { \ell , h } \in \mathbb { R } ^ { D \times d _ { \ell , h } }$ , which can be understood as maps between the D-dimensional space of the residual stream and the $d _ { \ell , h }$ -dimensional space of a given attention head, where typically $d _ { \ell , h } = d$ is the same for all heads.

Probing Probing refers to a supervised approach for finding the learned feature representations of a certain concept-of-interest in the activation space of a trained neural network (Alain & Bengio, 2017; Belinkov, 2022). Inputs associated with “ground truth” labels for the concept-of-interest are passed to a trained neural network, and the network’s activations as it processes those inputs are recorded. A “probe” is then a model trained to predict the ground-truth labels from network activations. Several probes are typically fit to different sets of activations, and each probe is often from a family of linear models (e.g., linear regression)—i.e., a linear probe.

The literature on probing LLMs places particular emphasis on linear probing, largely due to widespread belief in the (often underspecified) hypothesis that “important” high-level concepts are represented linearly as directions in representation space (Mikolov et al., 2013; Park et al., 2024b). A practical specification of this hypothesis, which we will adopt throughout, is that “important” concepts can be accurately predicted from network activations via linear probes, and that such concepts are not more accurately predicted by more flexible non-linear probes. As an example, Gurnee & Tegmark (2024) find that linear probes are accurate (and no less so than non-linear probes) at predicting the latitude and longitude of a place from an LLM’s representation of the place’s name.

There is fundamental ambiguity about what terms like “activation” or “representation space” refer to in the context of LLMs, and thus ambiguity about which vectors should be the target of probing. Much of the existing work, which we will follow, advocates for probing the output of individual attention heads (Michel et al., 2019; Olsson et al., 2022) and for fitting a separate probe to each (Li et al., 2023). For example, Llama-2-7b-chat consists of 32 layers, each containing 32 attention heads. Probing such a model might thus involve training $1 , 0 2 4 = 3 2 \times 3 2$ separate linear probes.

Concretely, a probing data set is initially constructed as a set of N prompt-label pairs $\{ \pmb { w } ^ { ( i ) } , \pmb { y } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ Each prompt set of activat $\pmb { w } ^ { ( i ) }$ is given as input to the L for each prompt i will be and a set of activations are recorded. In our case, thein Equation (3) for every attention head h in layer ℓ. $\pmb { x } _ { \ell , h } ^ { ( i ) }$ For every head we will then fit a linear probe, each of which assumes:

$$
\mathbb {E} \left[ y ^ {(i)} \mid \boldsymbol {x} _ {\ell , h} ^ {(i)} \right] = \widehat {y} _ {\ell , h} ^ {(i)} \triangleq \boldsymbol {\theta} _ {\ell , h} ^ {\top} \boldsymbol {x} _ {\ell , h} ^ {(i)} \tag {4}
$$

where $\theta _ { \ell , h } \in \mathbb { R } ^ { d _ { \ell , h } }$ are regression coefficients to fit. Following Gurnee & Tegmark (2024), we will fit these probes using ridge regression—i.e., by minimizing the L2-regularized squared loss:

$$
\mathcal {L} _ {\lambda} \big (\boldsymbol {\theta} _ {\ell , h} \big) = \sum_ {i = 1} ^ {N} (y ^ {(i)} - \boldsymbol {\theta} _ {\ell , h} ^ {\top} \boldsymbol {x} _ {\ell , h} ^ {(i)}) ^ {2} + \lambda \| \boldsymbol {\theta} _ {\ell , h} \| _ {2} ^ {2} \tag {5}
$$

where λ is a hyperparameter that can be tuned via cross-validation. Ridge, as opposed to unregularized linear regression, is often selected to mitigate overfitting and issues arising from multicollinearity in the activation vector. After training, if the linear model shows good fit, the estimated coefficients $\theta _ { \ell , h }$ can be understood as capturing a direction in activation space corresponding to the given conceptof-interest. For instance, if $\mathbf { \Delta } w ^ { ( i ) }$ is the name of a place, and $\boldsymbol y ^ { ( i ) }$ is its longitude, then $\widehat { \theta } _ { \ell , h }$ might correspond to a “North–South” axis in activation space.

# 3 TRAINING PROBES TO PREDICT DW-NOMINATE OF U.S. LAWMAKERS

This section reports on a set of probing experiments to find linear feature representations of political perspective in three open transformer-based LLMs. As described in Section 2, probing generally requires access to some “ground truth” labeling $\boldsymbol y ^ { ( i ) }$ of a given input $\pmb { w } ^ { ( i ) }$ . The term “political perspective” is ambiguous and can refer to a number of different concepts, each of which may be subjective and difficult to pin down precisely, let alone quantify. Generally speaking, the study of any social scientific concept must grapple with the problem of measurement (Adcock & Collier, 2001; Jacobs & Wallach, 2021). In this section, we operationalize “political perspective” as meaning (roughly) “position on the liberal-conservative ideological axis in American politics”. We do so using DW-NOMINATE (Poole & Rosenthal, 1985; Poole, 2005; Carroll et al., 2009), a widely used and validated measure from political science for the ideology of U.S. lawmakers (e.g., Senators, Presidents). At a high level, we prompt LLMs to generate text in the style of a given lawmaker, and then train linear probes to predict that lawmaker’s DW-NOMINATE score from model activations.

Data The DW-NOMINATE for a given U.S. lawmaker is a multi-dimensional score that is predictive of the lawmaker’s voting record. Such scores are produced by fitting a specific parametric model to an input bills-by-lawmakers matrix that records how each of the lawmakers voted (yea, nay, or abstain) on each of the proposed bills. These scores are well-established as accurate reflections of lawmakers’ stances on a broad range of legislative issues (McCarty, 2016; Everson et al., 2016). Moreover, the first dimension of such scores is often interpreted as scaling lawmakers’ ideology—i.e., positioning them on a “liberal-conservative” axis such that very liberal lawmakers are on one side, very conservative are on the other, and centrists in the middle. The scores range from -1 to 1, where -1 is most liberal and 1 is most conservative.

We use the first dimension of DW-NOMINATE scores for all lawmakers associated with the $1 1 6 ^ { \mathrm { { t h } } }$ United States Congress (N=552).2 We selected this Congress because it was active from 2019 to 2021, prior to the knowledge cutoff of September 2022 for the pre-training data of the Llama-2 family of language models (Touvron et al., 2023).

Prompting To construct a probing data set, we created a set of $N = 5 5 2$ prompts, one for each of the members of the $1 1 6 ^ { \mathrm { t h } }$ Congress, and each prompt $\mathbf { \Delta } _ { \mathbf { \pmb { w } } } ^ { ( i ) }$ conforming to the following template:

USER: Generate a statement by [NAME], a politician in the United States. ASSISTANT: In 2019, [NAME] said that

For each of three open transformer-based LLMs (Llama-2-7b-chat, Mistral-7b-instruct, Vicuna-7b) we passed each of these prompts as input, and then recorded the activation $\pmb { x } _ { \ell , h } ^ { ( i ) }$ of each attention head h in each layer ℓ.

Probing We then trained separate linear probes for each attention head h in layer ℓ of a given model to predict the DW-NOMINATE score $\dot { \boldsymbol y } ^ { ( i ) }$ of lawmaker i from the activation $\dot { \mathbf { \boldsymbol { x } } } _ { \ell , h } ^ { ( i ) }$ induced by prompt $\mathbf { \Delta } w ^ { ( i ) }$ . We set the regularization strength $\lambda = 1$ (see Equation (5)) after performing 2-fold cross-validation for the values {0, 0.001, 0.01, 0.1, 1, 100, 1000} (see Table A1). After training, each probe contributes predictions $\hat { \mathcal { Y } } _ { \ell , h } ^ { ( i ) }$ (Equation (4)), which we can use for evaluation.

![](images/0d237041de0e92c9519bd4730decea9f114b8a6eab0c4caff307436611cfad77.jpg)  
(a) Llama-2-7b-chat

![](images/931c95ded79d06d1ed70ddbaf09e231f9c6cac01b184d082c5b2f70f06d2e93b.jpg)

<details>
<summary>heatmap</summary>

| Layer | Head (Sorted) |
|-------|---------------|
| 2     | 0.0           |
| 4     | 0.2           |
| 6     | 0.4           |
| 8     | 0.6           |
| 10    | 0.8           |
| 12    | 0.6           |
| 14    | 0.4           |
| 16    | 0.2           |
| 18    | 0.0           |
| 20    | 0.2           |
| 22    | 0.4           |
| 24    | 0.6           |
| 26    | 0.8           |
| 28    | 0.6           |
| 30    | 0.4           |
| 32    | 0.2           |
</details>

(b) Mistral-7b-instruct

![](images/548afc13598a2b31336ef3575cd2b2864d1cea8a6a89fcbbdd6929570618cec5.jpg)

<details>
<summary>heatmap</summary>

| Layer | Head (Sorted) | Value |
|-------|---------------|-------|
| 2     | 1             | -0.2  |
| 4     | 2             | -0.1  |
| 6     | 3             | 0.1   |
| 8     | 4             | 0.3   |
| 10    | 5             | 0.5   |
| 12    | 6             | 0.7   |
| 14    | 7             | 0.9   |
| 16    | 8             | 0.8   |
| 18    | 9             | 0.6   |
| 20    | 10            | 0.4   |
| 22    | 11            | 0.2   |
| 24    | 12            | 0.1   |
| 26    | 13            | -0.1  |
| 28    | 14            | -0.3  |
| 30    | 15            | -0.5  |
| 32    | 16            | -0.7  |
</details>

(c) Vicuna-7b   
Figure 2: Predictive performance of linear probes for all attention heads across all layers in Llama-2-7b-chat, Mistral-7b-instruct, and Vicuna-7b. Each row (i.e., y-axis) represents each layer of the model from the bottom (layers close to the input layer) to the top (layers close to the output layer). Each column (i.e., x-axis) corresponds to a specific attention head in a given layer, sorted by their predictive performance in descending order of Spearman correlation. Darker versus lighter shades indicate higher versus lower Spearman correlation, meaning the attention head was more or less predictive of lawmakers’ political ideology (i.e., DW-NOMINATE scores).

Evaluation To evaluate the fit of each linear probe, we performed 2-fold cross-validation, using a random partition of lawmakers into two folds of equal size. For each of the two splits, we fit probes to one fold and had them generate predictions on the other test fold. We then computed the Spearman rank correlation between the predicted $\{ \widehat { y } _ { \ell , h } ^ { ( i ) } \} _ { i \in \mathrm { t e s t } }$ and true $\{ y ^ { ( i ) } \} _ { i \in \mathrm { t e s t } }$ scores. Our goodness-of-fit measure is then averaged across the two splits—i.e., the cross-validation Spearman rank correlation, which we denote $\widehat { \rho } _ { \ell , h } ^ { \mathrm { C V } }$ .

We can also evaluate ensembled predictions of probes across different heads and layers. To do so, we define $\mathcal { T } _ { K }$ to be the set of indices (ℓ, h) for the K probes with highest $\widehat { \rho } _ { \ell , h } ^ { \mathrm { C V } }$ . The ensembled predictions we explore are then defined as

$$
\widehat {y} _ {K} ^ {(i)} \triangleq \frac {1}{K} \sum_ {(\ell , h) \in \mathcal {T} _ {K}} \widehat {y} _ {\ell , h} ^ {(i)} \tag {6}
$$

We can evaluate these for different K using another round of cross-validation, each yielding a correlation score $\widehat { \rho } _ { K } ^ { \mathrm { C V } }$ for that ensemble. Intuitively, we expect such scores to increase in K up to some point but then eventually decrease as less predictive heads are averaged in.

Results We find for all three models, many or most of the probes fit to attention heads in the middle layers (around 10–20) exhibit high Spearman correlation $\boldsymbol { { \widehat \rho } } _ { \ell , h } ^ { \dot { \mathrm { c v } } }$ of around 0.8. For Llama-2-7b-chat, the highest Spearman correlation is 0.854, which is achieved by the probe of the $1 8 ^ { \mathrm { t h } }$ head in the 15th layer. For Mistral-7b-instruct and Vicuna-7b, it is 0.846 and 0.861, respectively, achieved by the probes of the $3 ^ { \mathrm { r d } }$ head in the $1 6 ^ { \mathrm { t h } }$ layer and the $8 ^ { \mathrm { t h } }$ head in the $2 4 ^ { \mathrm { t h } }$ layer. All Spearman correlations for each model are visualized as heatmaps in Figure 2, and the top 10 values for each are given in Table A2.

We also provide results for the ensembled models in Table A3, where we find that performance tapers around $K = 3 2$ , at which $\widehat { \rho } _ { K } ^ { \mathrm { C V } }$ is 0.87 for Llama-2-7b-chat, 0.864 for Mistral-7b-instruct, and 0.885 for Vicuna-7b. In Figure 3 we also plot the ensembled predictions for Llama-2-7b-chat and highlight examples of well-known lawmakers; the same plot for all three models is in Figure A1 and Figure A2.

The results broadly indicate that middle-layer activations are linearly predictive of DW-NOMINATE, and thus may possess linear representations of the “liberal–conservative” ideological axis. Before concluding this, we undertook a series of robustness checks.

![](images/e97b1fbd524e9cdf6e999fe8b269449adb5baa922d5466e25717ac9dd1282f26.jpg)

<details>
<summary>scatter</summary>

| Name                  | Predicted | Actual |
| --------------------- | --------- | ------ |
| Rand Paul (R-KY)      | 0.50      | 0.75   |
| Mike Lee (R-AZ)       | 0.60      | 0.85   |
| Lisa Murkowski (R-AK) | -0.20     | 0.40   |
| Max Rose (D-NY)       | -0.30     | -0.10  |
| Ilhan Omar (D-MN)     | -0.40     | -0.30  |
| Susan Collins (R-ME)  | 0.40      | 0.60   |
| Joe Manchin (D-WV)    | 0.30      | -0.20  |
| Kamala Harris (D-CA)  | -0.50     | -0.60  |
| Elizabeth Warren (D-MA)| -0.25     | -0.75  |
</details>

(a) U.S. lawmakers $( \widehat { \rho } _ { K = 3 2 } ^ { \mathrm { C V } } = 0 . 8 7 0 )$

![](images/cac2b10fb78a2b217bbaebcf1e7e5d261bfbd98e830c95d3287046853bf860a4.jpg)

<details>
<summary>scatter</summary>

| Source           | Predicted | Actual |
| ---------------- | --------- | ------ |
| salon            | -0.4      | -0.75  |
| The New York Times | -0.2      | -0.25  |
| HBOFFPOST         | -0.3      | -0.1   |
| brobible         | 0.1       | 0.0    |
| INFOWARS          | 0.4       | 0.75   |
</details>

(b) U.S. news outlets $( \hat { \rho } _ { K = 3 2 } ^ { \mathrm { C V } } = 0 . 7 9 8 )$   
Figure 3: Ideological perspectives of U.S. lawmakers and news media as predicted by the activations of the $K = 3 2$ most predictive attention heads of Llama-2-7b-chat. Negative values correspond to left-leaning perspectives, while positive values correspond to right-leaning perspectives. The x-axis represents the predicted political slant $( \widehat { y } _ { K = 3 2 } ^ { ( i ) } )$ for each entity (i.e., lawmakers or news media). The y-axis represents the previously validated ideological scores (DW-NOMINATE or Ad Fontes Media scores). See Figures A1 and A2 for the complete results across all models.

Robustness checks of linearity First, we compare the predictive performance of our linear probes to those of more flexible non-linear probes. Following Gurnee & Tegmark (2024), we fit one-layer multilayer perceptions (MLPs) with ReLU non-linearities, each of which is formulated as:

$$
\widehat {y} _ {\ell , h} ^ {(i)} = A _ {\ell , h} \operatorname{ReLU} \left(B _ {\ell , h} \boldsymbol {x} _ {\ell , h} ^ {(i)} + \boldsymbol {b} _ {\ell , h}\right) + \boldsymbol {a} _ {\ell , h} \tag {7}
$$

where $B _ { \ell , h } , A _ { \ell , h }$ and ${ \pmb b } _ { \ell , h } , { \pmb a } _ { \ell , h }$ are the weight matrices and bias vectors, respectively.

We do not observe substantial improvements when using such non-linear probes. For Llama-2-7b-chat, the most predictive linear probe had a cross-validation Spearman correlation of 0.854 while the best non-linear probe achieved 0.855. For Vicuna-7b, the difference was larger, with the linear and non-linear probes achieving 0.861 and 0.872, respectively. But for Mistral-7b-instruct, the order was reversed, with the linear and non-linear probes achieving 0.846 and 0.838, respectively. These results support the linear representation hypothesis for political ideology in the sense that linear functions of certain attention heads predict DW-NOMINATE approximately as well as non-linear functions of any others.

One may wonder whether there is enough information stored in all the attention heads of an LLM to be able to accurately predict any systematic label with linear probes. As a second robustness check, we applied different transformations to the DW-NOMINATE scores and examined whether linear probes could still fit them well. Specifically, we applied 1) a cubic transformation— $- y ^ { ( i ) }  ( y ^ { ( i ) } ) ^ { 3 }$ —which is non-linear but still monotonic, 2) a non-monotonic transformation— $- y ^ { ( i ) } \gets \sin ( 1 0 y ^ { ( i ) } )$ —and 3) a random permutation— $- y ^ { ( i ) } \gets y ^ { ( \Delta ( i ) ) }$ )— where ∆ defines a permutation of the indices i.

The results are given in Figure A3 and Table A4. The probes trained on randomly permuted labels provide a baseline Spearman correlation of around 0.15, representing chance performance. Probes trained to predict the non-monotonic transformation perform poorly, with the best-performing heads achieving correlations of around 0.5. As might be expected, probes trained to predict the cubic transformation do much better, with the best-performing heads achieving rank correlations close to 0.84. In addition to rank correlation, which should not be sensitive to monotonic transformations, we also include in Table A4 the cross-validation $R ^ { 2 }$ values of the different probes. These tell a different story, with the cubic probes exhibiting values of around 0.6 compared to values of 0.8 achieved by the original.

# 4 TRAINED PROBES DETECT POLITICAL PERSPECTIVE TOKEN-BY-TOKEN

The linear probes described in the last section were trained to predict the DW-NOMINATE $\boldsymbol y ^ { ( i ) }$ of lawmaker i from the activations induced by prompt w(i). The prompt includes the lawmaker’s name and little else, so one may wonder whether probes’ strong performance simply reflects models having “memorized” exact DW-NOMINATE scores, which are likely present in their pre-training data.

As a first investigation into whether the probes detected any generalizable representation of political ideology, we instructed models to generate essays on different policy issues (e.g., immigration or abortion). We then recorded model activations token-by-token. In this case, denote $\pmb { x } _ { \ell , h } ^ { ( i , t ) }$ to be the activation of head h in layer ℓ for policy issue i after t generated tokens. We then use the linear probe trained to predict DW-NOMINATE at that same attention head to calculate ${ \widehat { y } } _ { \ell , h } ^ { ( i , t ) } \triangleq { \widehat { \pmb { \theta } } } _ { \ell , h } ^ { \top } { \pmb x } _ { \ell , h } ^ { ( i , t ) }$ . If the probe has learned to predict nothing other than DW-NOMINATE from lawmaker names, we should not expect such a measurement to be interpretable when applied to open-ended responses. However, if probes have instead found a more general “liberal–conservative” ideological axis, then we might expect this measure to position tokens along that axis in an interpretable manner.

We visualize this measure in Figure 1 where tokens are colored more red or more blue according to whether y (i,t)ℓ,h $\widehat { y } _ { \ell , h } ^ { ( i , t ) }$ is more towards 1 (conservative) or -1 (liberal). The results are highly interpretable. The probes detect a liberal perspective when writing “those who support immigration argue that it can bring many benefits” or “a woman has the right to make decisions about her own body.” By contrast, the probes detect a conservative perspective when writing “immigration can cause problems” or “abortion is the intentional termination of a human life”. We found similarly interpretable qualitative results in many other examples but leave for future work a more systematic evaluation of this qualitative measure. In Appendix A.2, we provide the distribution of $\widehat { y } _ { \ell , h } ^ { ( i , t ) }$ over many different policy issues for the three models; these results possibly indicate conservative skew for Mistral-7b.

# 5 TRAINED PROBES GENERALIZE TO PREDICT U.S. NEWS MEDIA SLANT

As a more systematic test of whether the probes trained to predict DW-NOMINATE have truly detected a more generalizable representation of the “liberal-conservative” axis, we tested whether such probes can predict the political slant of different U.S. media outlets. Again, “media slant” is a subjective and imprecise notion, but one for which researchers have developed, validated, and relied upon data-driven measures. We find that probes trained only on DW-NOMINATE can predict a media outlet’s Ad Fontes score when LLMs are instructed to generate text from the perspective of that outlet.

Data We use data from Ad Fontes Media, which scores U.S. news outlets on a 5-point scale from “Left” to “Right”. Ad Fontes Media determines these scores by aggregating the scores of individual articles, which are rated simultaneously by a group of at least three human analysts (Otero, 2021). These groups are politically balanced, consisting of one right-leaning, one centrist, and one left-leaning individual. These scores have been used by researchers (e.g., Huszár et al. (2022)) as accurate reflections of how an outlet’s slant is broadly perceived. We took the scores for the N = 400 most popular outlets (e.g., Fox News, CNN) and normalized them to fall on the same scale as DW-NOMINATE of -1 (Left) to 1 (Right).

Prompting We constructed a probing data set of N = 400 prompts, one for each outlet, with each prompt w(i) conforming to the following template:

USER: Generate a statement from a news source in the United States.

ASSISTANT: [OUTLET] reported that

As before, for each of three LLMs (Llama-2-7b-chat, Mistral-7bpassed each of these prompts as input, and then recorded the activation $\pmb { x } _ { \ell , h } ^ { ( i ) }$ struct, Vicuna-7b) we of each attention head h in each layer ℓ. This yields a dataset of $( \boldsymbol { y } ^ { ( i ) } , \pmb { x } _ { \ell , h } ^ { ( i ) } )$ pairs, where $\boldsymbol y ^ { ( i ) }$ is the Ad Fontes score of outlet i.

![](images/a87b462f958cf21581d16f9cb3537b21bbfc9bb43b48ce14348d84c2338cc238.jpg)

<details>
<summary>line</summary>

| Intervention (alpha) | Llama-2-7b (ρ=0.607) | Mistral-7b (ρ=0.396) | Vicuna-7b (ρ=0.381) |
| --------------------- | --------------------- | --------------------- | --------------------- |
| -30                   | 2.2                   | 2.2                   | 2.7                   |
| -20                   | 2.0                   | 2.2                   | 2.3                   |
| -10                   | 2.1                   | 2.5                   | 2.2                   |
| 0                     | 2.3                   | 2.7                   | 2.6                   |
| 10                    | 2.8                   | 3.2                   | 3.0                   |
| 20                    | 3.8                   | 3.4                   | 3.9                   |
| 30                    | 5.0                   | 3.7                   | 4.1                   |
</details>

(a) The measured political slant of essay by the intervention parameter α used.

![](images/a7e9c0737fdbb0d180bb34e38d863e2d249a3197e006300099e46955fa1e3f87.jpg)

<details>
<summary>bar</summary>

| Category             | Correlation |
| -------------------- | ----------- |
| Immigration          | 0.8         |
| Abortion             | 0.7         |
| Social media regulation | 0.6       |
| Same-sex marriage    | 0.5         |
| LGBTQ+ rights        | 0.4         |
| Gun control           | 0.3         |
| Climate change       | 0.3         |
| Trust in science     | 0.3         |
| Free speech           | 0.2         |
</details>

(b) Correlation between α and political slant by issue.

![](images/420e40a5bc2f704229d255609a6389e60dab76d4fac61c7360be54df5b4aab7e.jpg)

<details>
<summary>bar</summary>

| Category             | Correlation |
| -------------------- | ----------- |
| Gun control          | -0.25       |
| Climate change       | -0.25       |
| Social media regulation | -0.25      |
| LGBTQ+ rights        | -0.25       |
| Abortion             | -0.25       |
| Immigration          | -0.25       |
| Same-sex marriage    | -0.25       |
| Trust in science     | -0.25       |
| Free speech          | -0.25       |
</details>

(c) Correlation between α and length of generated output.   
Figure 4: Trained probes can be used effectively to steer the political slant of generated text; see (a). Steering is more reliable for certain policy issues, but has a positive effect for all; see (b). LLMs steered toward more liberal positions on certain policy issues tend to produce longer essays; see (c).

Evaluation Unlike before, we do not train a new probe on the collected dataset. Rather, we simply evaluate whether the probe previously trained to predict DW-NOMINATE for layer ℓ and head h is able to predict the Ad Fontes score $y ^ { ( i ) }$ from $\pmb { x } _ { \ell , h } ^ { ( i ) }$ .

To evaluate, we use Spearman rank correlation between the set of observed Ad Fontes Media scores $\{ \boldsymbol { y } ^ { ( i ) } \}$ i and the ensembled predictions $\{ y _ { K } ^ { ( i ) } \} _ { i } ,$ as defined in Equation (6), using the $K = 3 2$ heads that were most predictive of DW-NOMINATE.

Results We find the trained probes generalize well to predict media slant, with those for Llama-2-7b-chat achieving a Spearman correlation of 0.798, for Mistral-7b-instruct 0.764, and for Vicuna-7b 0.720. In Figure 3 we plot the predictions for Llama-2-7b-chat and highlight examples of well-known outlets; the same plot for all three models is given in Figures A1 and A2.

# 6 TRAINED PROBES CAN BE USED TO STEER POLITICAL PERSPECTIVE

If indeed probes have identified a linear “liberal-conservative” direction in activation space, it is natural to ask whether the political perspective in LLM-generated text can be reliably steered by intervening linearly on its activations. In this section, we demonstrate this is the case.

Steering vectors Following the “inference time intervention” methodology of Li et al. (2023), we use the fitted regression coefficients $\widehat { \theta } _ { \ell , h }$ of the trained probes as steering vectors, which we add model activations over the course of text generation. More specifically, we intervene on the model by replacing the activation ${ \pmb x } _ { \ell , h }$ in Equation (3) with

$$
\boldsymbol {x} _ {\ell , h} ^ {(\alpha)} \triangleq \boldsymbol {x} _ {\ell , h} + \alpha \widehat {\sigma} _ {\ell , h} \widehat {\boldsymbol {\theta}} _ {\ell , h} \tag {8}
$$

where $\widehat { \sigma } _ { \ell , h }$ is an estimate of the standard deviation of activations ${ \pmb x } _ { \ell , h } ,$ , and $\alpha \in \mathbb { R }$ controls the magnitude and direction of the intervention. An α with a larger negative value should steer the model to produce more liberal-sounding text, while a more positive α should steer toward more conservative-sounding text. For a given α, we apply the intervention in Equation (8) iteratively for every token the model generates and do so at all of the K most predictive attention heads (i.e., for all in the set $\mathcal { T } _ { K }$ defined above Equation (6). The diagram in Figure A4 describes the entire procedure.

Study design To evaluate the effectiveness of these steering vectors, we instructed LLMs to generate text about nine key policy issues—Abortion, Immigration, Gun Control, Same-Sex Marriage, LGBTQ+ Rights, Climate Change, Trust in Science, Social Media Regulation, and Free Speech—and examined whether intervening on their activations at various levels of α produced predictable shifts in the political perspective of the text they generated. We used the following simple prompt:

USER: Write a statement about [ISSUE]. ASSISTANT: Regarding [ISSUE], I believe that

In total, we generated 1,134 essays across three models, nine policy issues, and combinations of six values of $K \in \{ 1 6 , 3 2 , 4 8 , 6 4 , 8 0 , 9 6 \}$ values and seven values of α ∈ {−30, −20, −10, 0, 10, 20, 30}.3

To measure the political perspective of each generated essay, we first recruited 10 human annotators from CloudResearch Survey—three Democrats, four independents, and three Republicans—and had them rate a random sample of the essays on a 7-point scale from “Strongly conservative” to “Strongly liberal”. We then instructed GPT-4o (gpt-4o-2024-08-06) to rate the same essays on the same scale (see Appendix A.3 for the exact prompt) and measured the inter-rater reliability between the GPT ratings and the average human ratings. GPT-4o’s ratings were very close to the humans’, with an intraclass correlation of 0.91, which we considered license to use it for rating the entire essays; see Appendix A.4 for more details.

Results We find that steering vectors reliably alter generated text toward political stances indicated by α. In Figure 4a, we show the average rating of all essays that were generated with a given value of α, for the three different models. For all three, we see a clear trend, with larger α predicting more conservative-sounding text. We also notice that with no intervention (α = 0), all three models show an average rating below 4 (on the 1–7 scale), indicating a base-level output of more liberal-sounding text.

When K ∈ {64, 80, 96}, Llama-2-7b-chat displayed the highest correlation of 0.607 between α and political slant, followed by Mistral-7b-instruct at 0.396, and Vicuna-7b at 0.381. Political slant increased steadily as α increased, particularly in Llama-2-7b-chat, suggesting that this model is more sensitive to intervention. We also experimented with intervening on different numbers of attention heads K, and found that intervening on more led to greater effectiveness; see Figure A5.

In Figure 4b, we break results out by policy issue. The issues for which the intervention was most reliable were Immigration and Abortion. We conjecture that this is due to there being a wider array of stances on such issues, as compared to issues like “Free Speech” or “Trust in Science” which exhibit smaller (though positive) correlations with α. Appendix A.5 gives illustrative examples.

We also observed that for certain policy issues, the LLMs generated much longer outputs when steered to sound more liberal than more conservative. This was true in particular for Gun Control and Climate Change; see Figure 4c. A deeper look into these results might provide evidence for systemic differences in the argumentation style between liberals and conservatives, and highlight promising avenues for future research.

Robustness checks One might wonder whether the interventions we describe will continue to be effective at steering when discussing policy issues not described in the model’s pre-training data. In Appendix A.6, we show that interventions remained effective when models were instructed to write about two events that fell after Llama-2-7b-chat’s pre-training cutoff: 1) the U.S. ADVANCE Act, and 2) the 2023 United Auto Workers (UAW) Strike.

One might also wonder whether interventions targeting different regions of the model (e.g., early versus late layers) have different effects. In Appendix A.7, we show that interventions on early-tomiddle layers are effective, while those on middle-to-last layers have almost no effect.

# 7 FURTHER CONNECTIONS TO PRIOR RESEARCH

Political bias of LLMs One closely related area of research focuses on assessing the political “bias” of LLMs. Studies have found that LLMs tend to generate responses more closely aligned with liberal-leaning stances on various issues, regardless of user prompts and inputs (Santurkar et al., 2023; Motoki et al., 2024; Martin, 2023; Potter et al., 2024; Liu et al., 2022; Bang et al., 2024). LLMs also often “avoid” engaging with certain political topics entirely (Bang et al., 2021). Political biases in the pre-training corpus of LLMs can manifest in ways relevant to downstream tasks such as hate speech and misinformation detection (Feng et al., 2023; Jiang et al., 2022; Liu et al., 2022).

Nevertheless, robustly measuring the political biases of LLMs remains challenging. Close-ended survey questions, such as the Political Compass Test (Feng et al., 2023) or Pew surveys (Santurkar et al., 2023), are frequently used to assess LLMs’ political biases. Yet, studies suggest that constraining LLMs to close-ended, multiple-choice formats may fail to capture biases that only emerge in open-ended responses (Röttger et al., 2024; Goldfarb-Tarrant et al., 2021). Recent studies also suggest that LLMs exhibit dishonesty (Huang et al., 2024) and sycophancy (Sharma et al., 2024) in their responses, which could potentially harm humans’ ability to monitor bias in LLMs. As shown in Figure 1, our approach suggests a path to monitor and assess the political perspective implicitly adopted by LLMs.4

Linear scales of political ideology Linear representations of political ideology have a rich tradition in political science by way of “ideological scaling” techniques, such as (DW)-NOMINATE (Poole & Rosenthal, 1985; Poole, 2005) and many related techniques. Work on “partisan sorting” argues that U.S. political identity is increasingly aligned along a single left-right axis, with increasing alignment between partisan identity and individual policy preferences (Levendusky, 2009). This uni-dimensional, linear model of political ideology is supported by empirical research showing that one’s position in this dimension correlates with a broad range of issue stances, including economic policies, social issues like abortion and morality, and environmental concerns (Baldassarri & Gelman, 2008; Fiorina & Abrams, 2008; DellaPosta et al., 2015).

# 8 CONCLUSIONS, LIMITATIONS, AND FUTURE DIRECTIONS

Our research demonstrates that LLMs develop linear representations of political perspective within their hidden layers, locating subjective perspectives along a linear spectrum from left to right. By probing attention heads, we found that LLMs possess a generalizable linear representation of political perspective, which is highly predictive of established measures for the ideology of U.S. lawmakers the slant of U.S. news media. Importantly, we show that targeted interventions on these attention heads can causally influence the ideological tone of the generated text. This offers valuable insight and provides a method for identifying, monitoring, and steering the political perspective reflected in LLM-generated text, with broader implications for the design and application of AI systems in societal contexts discussed in Appendix A.9.

Our study has several limitations. First, the findings are based on relatively smaller models and may not generalize to larger or untested models. Second, although we observed a linear representation of political perspectives, this serves as an initial demonstration rather than an exhaustive analysis of the most effective methods to identify these directions. Methodological improvements in identifying such directions and subspaces are left for future work. Third, our research is U.S. centric and may not generalize to less polarized political environments, where linear representations of ideologies may be less effective representations (See Appendix A.10 for details). In such settings, however, we may characterize ideologies as the simplex of more than two “archetypal” or extreme political perspectives (Seth & Eugster, 2016). Fourth, we use GPT-4o to evaluate political slant; however, there is potential for bias when using an LLM as an evaluator. Although we validate GPT-4o’s evaluations against politically balanced human annotators, we recommend that future research using our methods continue to validate LLM-generated annotations against human annotations to triangulate and mitigate any inherent biases. Future research could also explore whether there are linear representations of more granular or intersected forms of political ideology. Other dimensions of cultural perspective (e.g., social class, gender) (Kozlowski et al., 2019) or knowledge and experience-based expertise were not explored in this paper. We hope that future research will investigate this promising direction and its potential to craft and steer customizable LLM agent perspectives.

# ETHICS STATEMENT

This research addresses the sensitive issue of political ideology in LLMs. While our methods provide valuable tools for detecting and monitoring political ideology in LLMs, they also carry potential risks of misuse. For example, malicious actors or AI product providers might exploit these techniques to deliver intentionally biased LLM outputs, bypassing societal discussions of fairness and transparency. Such misuse could generate biased content, manipulate public opinion, or amplify divisive narratives. Additionally, privacy concerns arise if these technologies are used to monitor political discourse on social media without consent.

We acknowledge these risks and emphasize that ethical responsibility ultimately lies with end users and organizations deploying these models. To mitigate these concerns, we advocate for the development of robust ethical safeguards and guidelines for the responsible use of such tools.

Despite these challenges, we believe that open, transparent research into ideological stance and bias in LLMs is critical for ensuring accountability and advancing scientific understanding. By making our work publicly available, we empower researchers to study these technologies, monitor their societal impact, and develop measures to mitigate potential harms. We strongly urge the research community to engage in collaborative efforts to address ethical challenges posed by LLMs.

# REPRODUCIBILITY STATEMENT

The data and code for reproducing our results are available on Github5.

# ACKNOWLEDGEMENTS

We thank Victor Veitch for helpful discussions and feedback. Aaron Schein was supported in part by the John D. and Catherine T. MacArthur Foundation. James Evans was supported in part by grants from the National Science Foundation (2404109), DARPA (W911NF2010302), and Google, Inc.

# REFERENCES

Robert Adcock and David Collier. Measurement validity: A shared standard for qualitative and quantitative research. American Political Science Review, 95(3):529–546, 2001.

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. In the Fifth International Conference on Learning Representations Workshop Track, 2017.

Jacob Andreas. Language Models as Agent Models. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 5769–5779, 2022.

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. In Advances in Neural Information Processing Systems, volume 37, pp. 136037–136083, 2024.

Lisa P Argyle, Christopher A Bail, Ethan C Busby, Joshua R Gubler, Thomas Howe, Christopher Rytting, Taylor Sorensen, and David Wingate. Leveraging AI for democratic discourse: Chat interventions can improve online political conversations at scale. Proceedings of the National Academy of Sciences, 120(41):e2311627120, 2023a.

Lisa P Argyle, Ethan C Busby, Nancy Fulda, Joshua R Gubler, Christopher Rytting, and David Wingate. Out of one, many: Using language models to simulate human samples. Political Analysis, 31(3):337–351, 2023b.

Hui Bai, Jan Voelkel, Johannes Eichstaedt, and Robb Willer. Artificial Intelligence Can Persuade Humans on Political Issues. https://www.researchsquare.com/article/rs-3238396/v1, 2023.

Delia Baldassarri and Andrew Gelman. Partisans without constraint: Political polarization and trends in American public opinion. American Journal of Sociology, 114(2):408–446, 2008.   
Yejin Bang, Nayeon Lee, Etsuko Ishii, Andrea Madotto, and Pascale Fung. Assessing Political Prudence of Open-Domain Chatbots. In Proceedings of the 22nd Annual Meeting of the Special Interest Group on Discourse and Dialogue, pp. 548–555, 2021.   
Yejin Bang, Delong Chen, Nayeon Lee, and Pascale Fung. Measuring Political Bias in Large Language Models: What Is Said and How It Is Said. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11142–11159, 2024.   
Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022.   
Pietro Bernardelle, Leon Fröhling, Stefano Civelli, Riccardo Lunardi, Kevin Roiter, and Gianluca Demartini. Mapping and Influencing the Political Ideology of Large Language Models using Synthetic Personas. arXiv preprint arXiv:2412.14843, 2024.   
Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach. Language (Technology) is Power: A Critical Survey of “Bias” in NLP. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020.   
Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, et al. Towards monosemanticity: Decomposing language models with dictionary learning. https://transformer-circuits.pub/2023/ monosemantic-features, 2023.   
Royce Carroll, Jeffrey B Lewis, James Lo, Keith T Poole, and Howard Rosenthal. Measuring bias and uncertainty in DW-NOMINATE ideal point estimates via the parametric bootstrap. Political Analysis, 17(3):261–275, 2009.   
Gary Charness, Brian Jabarian, and John A List. Generation Next: Experimentation with AI. http://www.nber.org/papers/w31679, 2023.   
Thomas H Costello, Gordon Pennycook, and David G Rand. Durably reducing conspiracy beliefs through dialogues with ai. Science, 385(6714):eadq1814, 2024.   
Daniel DellaPosta, Yongren Shi, and Michael Macy. Why do liberals drink lattes? American Journal of Sociology, 120(5):1473–1511, 2015.   
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A Mathematical Framework for Transformer Circuits. https://transformer-circuits.pub/2021/framework/, 2021.   
Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. arXiv preprint arXiv:2209.10652, 2022.   
Phil Everson, Rick Valelly, Arjun Vishwanath, and Jim Wiseman. NOMINATE and American political development: a primer. Studies in American Political Development, 30(2):97–115, 2016.   
Expected Parrot. Steerable large language models. https://www.expectedparrot.com/, 2024. Accessed: 2024-11-19.   
Shangbin Feng, Chan Young Park, Yuhan Liu, and Yulia Tsvetkov. From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models. arXiv preprint arXiv:2305.08283, 2023.   
Morris P Fiorina and Samuel J Abrams. Political polarization in the American public. Annual Review of Political Science, 11(1):563–588, 2008.   
Chen Gao, Xiaochong Lan, Nian Li, Yuan Yuan, Jingtao Ding, Zhilun Zhou, Fengli Xu, and Yong Li. Large language models empowered agent-based modeling and simulation: A survey and perspectives. Humanities and Social Sciences Communications, 11(1):1–24, 2024.

Kostas Gemenis. What to do (and not to do) with the comparative manifestos project data. Political Studies, 61(1\_suppl):3–23, 2013.   
Seraphina Goldfarb-Tarrant, Rebecca Marchant, Ricardo Muñoz Sánchez, Mugdha Pandya, and Adam Lopez. Intrinsic bias metrics do not correlate with application bias. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1926–1940, 2021.   
Wes Gurnee and Max Tegmark. Language models represent space and time. In the Twelfth International Conference on Learning Representations, 2024.   
Kobi Hackenburg and Helen Margetts. Evaluating the persuasive influence of political microtargeting with large language models. Proceedings of the National Academy of Sciences, 121(24): e2403116121, 2024.   
Luke Hewitt, Ashwini Ashokkumar, Isaias Ghezae, and Robb Willer. Predicting results of social science experiments using large language models. https://samim.io/dl/Predicting% 20results%20of%20social%20science%20experiments%20using%20large%20language% 20models.pdf, 2024.   
Youcheng Huang, Jingkun Tang, Duanyu Feng, Zheng Zhang, Wenqiang Lei, Jiancheng Lv, and Anthony G Cohn. Dishonesty in Helpful and Harmless Alignment. arXiv preprint arXiv:2406.01931, 2024.   
Ferenc Huszár, Sofia Ira Ktena, Conor O’Brien, Luca Belli, Andrew Schlaikjer, and Moritz Hardt. Algorithmic amplification of politics on Twitter. Proceedings of the National Academy of Sciences, 119(1):e2025334119, 2022.   
Abigail Z Jacobs and Hanna Wallach. Measurement and Fairness. In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency, pp. 375–385, 2021.   
Hang Jiang, Doug Beeferman, Brandon Roy, and Deb Roy. CommunityLM: Probing Partisan Worldviews from Language Models. In Proceedings of the 29th International Conference on Computational Linguistics, pp. 6818–6826, 2022.   
Junsol Kim and Byungkyu Lee. AI-Augmented Surveys: Leveraging Large Language Models and Surveys for Opinion Prediction. arXiv preprint arXiv:2305.09620, 2023.   
Austin C Kozlowski, Matt Taddy, and James A Evans. The Geometry of Culture: Analyzing the Meanings of Class through Word Embeddings. American Sociological Review, 84(5):905–949, 2019.   
Austin C Kozlowski, Hyunku Kwon, and James A Evans. In Silico Sociology: Forecasting COVID-19 Polarization with Large Language Models. arXiv preprint arXiv:2407.11190, 2024.   
Matthew Levendusky. The partisan sort: How liberals became Democrats and conservatives became Republicans. University of Chicago Press, 2009.   
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. In Advances in Neural Information Processing Systems, volume 36, pp. 41451–41530, 2023.   
Ruibo Liu, Chenyan Jia, Jason Wei, Guangxuan Xu, and Soroush Vosoughi. Quantifying and alleviating political bias in language models. Artificial Intelligence, 304:103654, 2022.   
Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In First Conference on Language Modeling, 2024.   
John Levi Martin. The Ethico-Political Universe of ChatGPT. Journal of Social Computing, 4(1): 1–11, 2023.   
Nolan McCarty. In defense of DW-NOMINATE. Studies in American Political Development, 30(2): 172–184, 2016.

Paul Michel, Omer Levy, and Graham Neubig. Are Sixteen Heads Really Better than One? In Advances in Neural Information Processing Systems, volume 32, 2019.   
Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic Regularities in Continuous Space Word Representations. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 746–751, 2013.   
Fabio Motoki, Valdemar Pinho Neto, and Victor Rodrigues. More human than human: Measuring ChatGPT political bias. Public Choice, 198(1):3–23, 2024.   
Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent Linear Representations in World Models of Self-Supervised Sequence Models. In Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pp. 16–30, 2023.   
Sean O’Hagan and Aaron Schein. Measurement in the Age of LLMs: An Application to Ideological Scaling. arXiv preprint arXiv:2312.09203, 2023.   
Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context Learning and Induction Heads. arXiv preprint arXiv:2209.11895, 2022.   
Vanessa Otero. Ad Fontes Media Content Analysis Methodology. https://adfontesmedia.com/wp-content/uploads/2022/07/ Ad-Fontes-Media-Content-Analysis-Methodology-White-Paper-September-2021-1. pdf, 2021.   
Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, 2023.   
Joon Sung Park, Carolyn Q Zou, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Robb Willer, Percy Liang, and Michael S Bernstein. Generative Agent Simulations of 1,000 People. arXiv preprint arXiv:2411.10109, 2024a.   
Kiho Park, Yo Joong Choe, and Victor Veitch. The Linear Representation Hypothesis and the Geometry of Large Language Models. In International Conference on Machine Learning, pp. 39643–39666. PMLR, 2024b.   
Keith T Poole. Spatial Models of Parliamentary Voting. Cambridge University Press, 2005.   
Keith T. Poole and Howard Rosenthal. A Spatial Model for Legislative Roll Call Analysis. American Journal of Political Science, 29(2):357–384, 1985.   
Yujin Potter, Shiyang Lai, Junsol Kim, James Evans, and Dawn Song. Hidden Persuaders: LLMs’ Political Leaning and Their Influence on Voters. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 4244–4275, 2024.   
Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Kirk, Hinrich Schuetze, and Dirk Hovy. Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15295–15311, 2024.   
Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. Whose Opinions Do Language Models Reflect? In Proceedings of the 40th International Conference on Machine Learning, volume 202, pp. 29971–30004. PMLR, 2023.   
Sohan Seth and Manuel JA Eugster. Probabilistic Archetypal Analysis. Machine Learning, 102: 85–113, 2016.   
Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R Johnston, et al. Towards understanding sycophancy in language models. In the Twelfth International Conference on Learning Representations, 2024.

Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell L Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, et al. Position: A Roadmap to Pluralistic Alignment. In Forty-first International Conference on Machine Learning, 2024.   
Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, and Neel Nanda. Linear Representations of Sentiment in Large Language Models. arXiv preprint arXiv:2310.15154, 2023.   
Petter Törnberg, Diliara Valeeva, Justus Uitermark, and Christopher Bail. Simulating Social Media Using Large Language Models to Evaluate Alternative News Feed Algorithms. arXiv preprint arXiv:2310.05984, 2023.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint arXiv:2307.09288, 2023.   
Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023.   
Dimitri Von Rütte, Sotiris Anagnostidis, Gregor Bachmann, and Thomas Hofmann. A Language Model’s Guide Through Latent Space. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pp. 49655–49687, 2024.   
Patrick Y Wu, Joshua A Tucker, Jonathan Nagler, and Solomon Messing. Large Language Models Can Be Used to Estimate the Latent Positions of Politicians. arXiv preprint arXiv:2303.12057, 2023.   
Patrick Y Wu, Jonathan Nagler, Joshua A Tucker, and Solomon Messing. Concept-Guided Chain-of-Thought Prompting for Pairwise Comparison Scoring of Texts with Large Language Models. In 2024 IEEE International Conference on Big Data (BigData), pp. 7232–7241. IEEE, 2024.

# A APPENDIX

# A.1 MODEL OVERVIEW

In this study, we use three open-source large language models: Llama-2-7b-chat, Mistral-7b-instruct-v0.1, and Vicuna-7b-v1.5.

• Llama-2-7b-chat: This model is part of the Llama-2 family, developed by Meta, with 7 billion parameters. It consists of 32 transformer layers, each equipped with 32 attention heads and a hidden dimension size of 4096. The model is optimized for conversational tasks.   
• Mistral-7b-instruct-v0.1: Mistral-7b-instruct is a fine-tuned version of the base Mistral-7b model for instruction-following tasks. Similar to Llama-2-7b-chat, Mistral-7b-instruct-v0.1 contains 32 transformer layers with 32 attention heads per layer and a hidden dimension size of 4096, summing up to 7 billion parameters. The model is optimized for conversational tasks.   
• Vicuna-7b-v1.5: Vicuna-7b is a fine-tuned version of Llama-2, optimized for conversation tasks. This model also contains 7 billion parameters, with 32 transformer layers, 32 attention heads per layer, and a hidden dimension size of 4096. The fine-tuning focuses on generating dialogue responses.

# A.2 TRAINED PROBES DETECT POLITICAL PERSPECTIVE TOKEN-BY-TOKEN

As described in Section 4, we can record model activations token-by-token to detect political perspective reflected in the tokens. In this case, denote $\pmb { x } _ { \ell , h } ^ { ( i , t ) }$ to be the activation of head h in ℓ  i  t DW-NOMINATE at that same attention head in order to calculate ${ \widehat { y } } _ { \ell , h } ^ { ( i , t ) } \triangleq { \widehat { \pmb { \theta } } } _ { \ell , h } ^ { \top } { \pmb x } _ { \ell , h } ^ { ( i , t ) }$ ained to redie

To examine the distribution of yb(i,t)ℓ,h $\widehat { y } _ { \ell , h } ^ { ( i , t ) }$ across models, we use the following prompts to ask the LLMs to write about nine politically divisive topics (Abortion, Immigration, Gun Control, Same-Sex Marriage, LGBTQ+ Rights, Climate Change, Trust in Science, Social Media Regulation, and Free Speech) in the United States. Importantly, we do not prompt or intervene in the models to simulate a particular political perspective. As a result, each model generates 90 essays (i.e., 10 essays per topic). While generating these essays, we obtain y values from the 96 most predictive attention heads.

• Write an essay on [TOPIC] in the U.S.   
• Discuss [TOPIC] in the United States.   
• Explain the key issues surrounding [TOPIC] in the U.S.   
• Provide an overview of [TOPIC] policies in America.   
• Summarize the debate on [TOPIC] in the United States.   
• Analyze the topic of [TOPIC] in the U.S.   
• Give a detailed account of [TOPIC] in the United States.   
• Explore the complexities of [TOPIC] legislation in the U.S.   
• Describe the current landscape of [TOPIC] in the United States.   
• Offer a comprehensive discussion on the state of [TOPIC] in the U.S.

As shown in Figure A6, even though we do not explicitly prompt LLMs to adopt a political stance, we find that the models spontaneously generate texts reflecting either left- or right-leaning perspectives, as indicated by the token-level values of political slant $\widehat { y } _ { \ell , h } ^ { ( i , t ) }$ . Specifically, Mistral-7b-instruct shows an average $\widehat { y } _ { \ell , h } ^ { ( i , t ) }$ of 0.143, exhibiting a slightly more right-leaning slant compared to Llama-2-7b-chat (0.054) and Vicuna-7b (0.048). Interestingly, Mistral-7b-instruct also displays about twice the diversity in perspectives, as measured by the standard deviation of $\widehat { y } _ { \ell , h } ^ { ( i , t ) }$ , with σ = 0.649 compared to Llama-2-7b-chat (σ = 0.349) and Vicuna-7b (σ = 0.356).

# A.3 PROMPTS TO RATE POLITICAL SLANT REFLECTED IN LLM RESPONSES

We use GPT-4o (gpt-4o-2024-08-06) to rate the political slant underlying LLM responses. Specifically, for each LLM essay under intervention, we prompt GPT-4o as follows:

Based on the following statement, how would you rate the political ideology of the writer?

[STATEMENT]

On a scale of 1 to 7, where:

1 = Strongly conservative

2 = Moderately conservative

3 = Slightly conservative

4 = Neither conservative nor liberal

5 = Slightly liberal

6 = Moderately liberal

7 = Strongly liberal

Please respond with a number from 1 to 7 without providing reasons.

# A.4 VALIDATING GPT-4O ANNOTATIONS AGAINST POLITICALLY BALANCED HUMAN ANNOTATORS

We validate GPT-4o’s evaluations against politically balanced human annotators. Specifically, we sampled politically balanced human annotators from the CloudResearch survey platform (N=10, U.S. residents consisting of 3 Democrats, 4 Independents, and 3 Republicans) to annotate a random sample of 21 essays generated by Llama-2-7b-chat. After averaging the scores provided by these human annotators, we measure inter-rater reliability between GPT-4o and the human annotators’ average scores. We find a high inter-rater reliability score (ICC(A, 1) = .91), supporting the validity of GPT-4o in annotating political slant. Based on these results, we conclude that GPT-4o’s ratings align closely with the consensus of politically balanced human annotators. Our findings are consistent with O’Hagan & Schein (2023), which demonstrate that LLMs can reliably reflect established measures of ideological slant. Additionally, the Spearman correlation between GPT-4o and average human scores is high (ρ = 0.952).

# A.5 EXAMPLES OF Llama-2-7b-chat RESPONSES IN THE INTERVENTION EXPERIMENTS

# A.5.1 IMMIGRATION

Prompt Write a statement about immigration.

Original Output (α=0) Regarding immigration, I believe that it is a complex issue that requires a comprehensive and humane approach. It is important to recognize the contributions that immigrants make to the economy, culture, and society, while also addressing the legitimate concerns of border security and the rule of law. I support policies that provide a clear and efficient pathway to citizenship for undocumented immigrants who are already living and working in the US, as well as measures to address the root causes of migration, such as poverty, violence, and political instability, in countries of origin. Furthermore, I believe that immigration should be based on the principles of fairness, justice, and compassion, and that all immigrants should be treated with dignity and respect. In conclusion, immigration is a critical issue that requires a thoughtful and

Steered Output (α=-20) Regarding immigration, I believe that it is a complex and multifaceted issue that affects not only the lives of immigrants themselves, but also the broader society. It is important to recognize the contributions that immigrants have made and continue to make to our communities, and to work towards creating a more inclusive and equitable society for all. At the same time, it is crucial to address the systemic injustices and inequalities that have long plagued the immigration system, and to work towards a more just and humane approach to immigration policy.

Ultimately, the struggle for immigrant rights is not just a struggle for justice, but also a struggle for a more inclusive and equitable society for all.

Steered Output (α=20) Regarding immigration, I believe that immigration laws should be enforled and regulated in a way that protects the national security and safety of a country. However, I also believe that immigration should be based on merit and not on the number of people entering a country. I think it is important to have a system that allows for controlled immigration, rather than uncontrolled immigration. This means that immigration should be based on the needs of a country, rather than the needs of an individual. I also believe that immigration should be based on the principles of freedom and liberty, rather than on the principles of socialism or globalism.

# A.5.2 SAME-SEX MARRIAGE

Prompt Write a statement about same-sex marriage.

Original Output (α=0) Regarding same-sex marriage, I believe that all individuals, regardless of their sexual orientation, should have the right to marry the person they love and have their relationship recognized by the government. This is a fundamental human right and essential for the well-being and dignity of all individuals.

Steered Output (α=-20) Regarding same-sex marriage, I believe that all individuals, regardless of their sexual orientation, should have the same rights and opportunities as straight couples. The ability to marry the person you love and build a life together is a fundamental human right, and it’s time that we recognize and respect this right for all couples, regardless of their sexual orientation.

Steered Output (α=20) Regarding same-sex marriage, I believe that all individuals should be able to marry someone of their choice, regardless of their gender or sexual orientation. However, I also believe that marriage should be defined as being between a man and a woman, as this is the traditional and biblical definition of marriage. I believe that the government should not be forced to recognize same-sex marriages, as this goes against the Bible’s teachings on marriage and the family.

# A.5.3 ADVANCE ACT

Prompt Write a political opinion about the following event: - The ADVANCE Act of 2024 was signed into law on July 9, 2024, as part of the Fire Grants and Safety Act. - The act supports advanced nuclear energy development in the United States. - It promotes generation IV nuclear reactor technology. - It reduces licensing costs for nuclear energy projects. - It extends liability protections for the nuclear industry. - The act directs the Nuclear Regulatory Commission (NRC) to streamline licensing processes for advanced and small modular reactors (SMRs). - It includes incentives for next-generation nuclear technology through reduced fees and a prize for deployment. - It restricts nuclear fuel imports from Russia and China. - It fosters U.S. nuclear exports and international collaboration. - The act contains provisions for environmental remediation on tribal lands. - Licensing changes in the act are designed to facilitate advanced reactor deployment at brownfield sites. - The act follows the Prohibiting Russian Uranium Imports Act.

Original Output (α=0) I believe that the ADVANCE Act of 2024 is a crucial step towards a more sustainable and secure energy future for the United States. By supporting the development of advanced nuclear energy technologies, such as generation IV reactors, the act helps to reduce the country’s reliance on fossil fuels and mitigate the impacts of climate change. Additionally, the act’s provisions for licensing reform and reduced fees for next-generation nuclear technology incentivize innovation and deployment, which will help to drive down costs and increase the competitiveness of nuclear energy in the U.S. electricity market. Moreover, the act’s restrictions on nuclear fuel imports from Russia and China help to promote energy security and reduce the risk of dependence on unreliable suppliers. By fostering U.S. nuclear exports and international collaboration, the act also helps to promote global nuclear safety and security standards, and to advance the global transition to a low-carbon energy future.

Steered Output (α=-20) I believe that the ADVANCE Act of 2024 is a crucial step towards a more sustainable and equitable future for the United States. By promoting the development and deployment of advanced nuclear energy technology, the act is not only addressing the urgent need to reduce greenhouse gas emissions but also creating new economic opportunities for communities that have been historically impacted by environmental degradation. Moreover, the act’s focus on reducing licensing costs and extending liability protections for the nuclear industry is a critical step towards making nuclear energy a more accessible and affordable option for all. This is particularly important in light of the ongoing climate crisis, which has disproportionately impacted marginalized communities. Furthermore, the act’s provisions for environmental remediation on tribal lands and its support for U.S. nuclear exports and international collaboration are crucial in building a more just and equitable nuclear energy sector.

Steered Output (α=20) I believe that the ADVANCE Act of 2024 is a positive step towards the development of nuclear energy in the United States. The act supports the development of advanced nuclear energy by reducing licensing costs, extending liability protections, and streamlining the licensing process for advanced and small modular reactors. This will allow for the deployment of next-generation nuclear technology, which is essential for the security of the United States. However, I also believe that the act is necessary due to the current political climate and the need for the United States to secure its energy sources. The United States should not rely on foreign sources of energy, especially from countries such as Russia and China, which have a history of using their energy resources for political gain. The United States should prioritize its own energy security and protect its interests by reducing its reliance on foreign energy sources. In conclusion, I believe that the ADVANCE Act of 2024 is a positive step towards the development of nuclear energy in the United States.

# A.5.4 UNITED AUTO WOKERS (UAW) STRIKE

Prompt Write a political opinion about the following event: - The 2023 United Auto Workers (UAW) strike lasted from September 15 to October 30. - Approximately 49,800 union members participated in the strike. - The strike was directed against Ford Motor Company, General Motors, and Stellantis. - The primary disputes were over labor contract negotiations. - Key union demands included: - Wage increases to counteract inflation. - Elimination of a tiered employment system. - Improved benefits. - Worker protections against plant closures. - A four-day workweek. - This was the first simultaneous strike against all three automakers. - A "rolling strike" strategy was used to conserve union resources. - Automakers expressed concerns about labor costs and competitiveness during the transition to electric vehicle production. - Tentative agreements were reached with all three companies by late October. - The agreements included: - Significant wage increases. - Reinstatement of cost-of-living adjustments. - Elimination of the two-tier wage system. - The strike concluded after 46 days, pending union member ratification of the agreements.

Original Output (α=0) I believe that 2023 United Auto Workers (UAW) strike is a significant event that had a profound impact on the automotive industry. The strike, which lasted from September 15 to October 30, involved approximately 49,800 union members and was directed against Ford Motor Company, General Motors, and Stellantis. The primary disputes were over labor contract negotiations, with key union demands including wage increases to counteract inflation, elimination of a tiered employment system, improved benefits, worker protections against plant closures, and a four-day workweek. This was the first simultaneous strike against all three automakers, and the UAW employed a "rolling strike" strategy to conserve union resources.

Steered Output (α=-20) I believe that the recent UAW strike is a significant political event that highlights the ongoing struggle for workers’ rights and fair labor practices. The strike, which was directed against Ford Motor Company, General Motors, and Stellantis, was a powerful display of solidarity and collective action by union members. The primary demands of the strike were labor contract negotiations, and the union’s key demands included wage increases to counteract inflation, elimination of a tiered employment system, improved benefits, worker protections against plant closures, and a four-day workweek. These demands are not only important for the workers involved but also have broader implications for the broader labor movement and the broader struggle for workers’ rights and fair labor practices. The strike also highlighted the ongoing struggle"

Steered Output (α=20) I believe that the strike was unnecessary and the unions should have been more restrictive in their demands.

# A.6 SIMULATING OPINIONS REGARDING EVENTS NOT IN THE PRE-TRAINING DATA

To examine whether linear interventions in LLMs can simulate ideological perspectives for events not included in their pre-training data, we conduct a case study on the Accelerating Deployment of Versatile, Advanced Nuclear for Clean Energy (ADVANCE) Act (March 2023) and the 2023 United Auto Workers (UAW) strike (September 2023). Both events occur after the knowledge cut-off date of Llama-2-7b-chat’s pre-training data (September 2022) (Touvron et al., 2023). To confirm that the model has no prior knowledge of these events, we first prompt it with the question, “Do you have information about [event]?” The responses indicate that it lacks accurate information about the event, either by responding “No” or generating hallucinated descriptions.

Then, using GPT-4o, we generate factual descriptions regarding each event. Specifically, we use a two-step approach: (1) we provide a Wikipedia article and prompt GPT-4o to generate a concise, oneparagraph factual summary, and (2) we prompt GPT-4o again to eliminate any subjective opinions from the paragraph and present the factual, neutral information in bullet points. The following prompts are used:

Provide a factual summary of the situation described in the Wikipedia article in one paragraph, avoiding any mention of opinions or perspectives associated with U.S. Democrats or Republicans.

From the following paragraph, remove any subjective opinions. Then, extract and list the factual and neutral information in bullet points.

After generating the factual summary, we provide this text to Llama-2-7b-chat using the following prompts. For each event, we ask the model to generate relevant texts with slightly different prompts (e.g., Write a political opinion about the following event, Write an essay about the following event, Write a statement about the following event).

Write a [political opinion/essay/statement] about the following event:

\- The ADVANCE Act of 2024 was signed into law on July 9, 2024, as part of the Fire Grants and Safety Act.

\- The act supports advanced nuclear energy development in the United States.

\- It promotes generation IV nuclear reactor technology.

\- It reduces licensing costs for nuclear energy projects.

\- It extends liability protections for the nuclear industry.

\- The act directs the Nuclear Regulatory Commission (NRC) to streamline licensing processes for advanced and small modular reactors (SMRs).

\- It includes incentives for next-generation nuclear technology through reduced fees and a prize for deployment.

\- It restricts nuclear fuel imports from Russia and China.

\- It fosters U.S. nuclear exports and international collaboration.

\- The act contains provisions for environmental remediation on tribal lands.

\- Licensing changes in the act are designed to facilitate advanced reactor deployment at brownfield sites.

\- The act follows the Prohibiting Russian Uranium Imports Act.

Write a [political opinion/essay/statement] about the following event:   
- The 2023 United Auto Workers (UAW) strike lasted from September 15 to October 30.   
- Approximately 49,800 union members participated in the strike.   
- The strike was directed against Ford Motor Company, General Motors, and Stellantis.   
- The primary disputes were over labor contract negotiations.   
- Key union demands included:   
- Wage increases to counteract inflation.   
- Elimination of a tiered employment system.   
- Improved benefits.   
- Worker protections against plant closures.   
- A four-day workweek.   
- This was the first simultaneous strike against all three automakers.   
- A "rolling strike" strategy was used to conserve union resources.   
- Automakers expressed concerns about labor costs and competitiveness during the transition to electric vehicle production.   
- Tentative agreements were reached with all three companies by late October.   
- The agreements included:   
- Significant wage increases.   
- Reinstatement of cost-of-living adjustments.   
- Elimination of the two-tier wage system.   
- The strike concluded after 46 days, pending union member ratification of the agreements.

Political essays are then generated with varying levels of intervention, using the linear steering method described in Section 6, with values of $\bar { x } \in \{ - 3 0 , - 2 0 , - 1 0 , 0 , 1 0 , 2 0 , 3 0 \}$ . We generated a total of 21 essays per event. To evaluate the ideological slant of these essays, GPT-4o (trained after the knowledge cut-off of Llama-2-7b-chat and thus familiar with these events) annotate the political slant on a scale where lower values (1) indicate liberal perspectives and higher values (7) indicate conservative ones.

The results show a statistically significant correlation between the intervention parameter (α) and the annotated political slant. Specifically, both the ADVANCE Act $( \rho = 0 . 4 7 9 )$ and the United Auto Workers (UAW) Strike $( \rho = 0 . 4 7 0 )$ exhibit significant correlations. For example, when LLMs are prompted about the ADVANCE Act, an intervention with α = −20 generates texts aligned with left-leaning views, supporting the act for its promotion of the nuclear energy industry but emphasizing its “environmental benefits.” Conversely, an intervention with α = 20 produces texts aligned with right-leaning views, supporting the act due to its focus on “restricting nuclear fuel imports from Russia and China.” These results indicate that, following interventions to simulate left- or right-leaning perspectives, the model not only predicts bipartisan support for the act but also captures nuanced differences in the reasons left- and right-leaning individuals support it (See Appendix A.5 for details). These findings suggest that linear interventions can simulate ideological biases, even for unforeseen events not included in pre-training data. This indicates that the linear structures identified in the model’s activations might capture more than just superficial patterns in the training data.

# A.7 INTERVENTION TARGETING SELECTED LAYERS

As Figure 2 shows, there are two “regions” of the attention heads that correlate with political slant: early to middle layers (Layers 1–21) versus middle to last layers (Layers 22–32). We conduct additional analyses on Llama-2-7b-chat to examine how interventions in early to middle layers (closer to input) versus middle to last layers (closer to output) affect ideological expression in responses (see Figure A7). Interventions targeting early to middle layers lead to more substantial ideological changes, as detected by GPT-4o $( \rho = 0 . 5 4 0 )$ . For example, when LLMs are asked about same-sex marriage, a right-leaning intervention (α = 20) at these layers produces statements like, “I believe that marriage should only be between a man and a woman, as this is the biblical definition of marriage.” (See Table A5). In contrast, interventions in the middle to last layers do not appear to alter the underlying ideological content (ρ = −0.022).

# A.8 THE RANGE OF α FOR GENERATING COHERENT RESPONSES

We conduct experiments to determine the minimum and maximum values of α required for generating coherent responses. We instruct LLMs to generate short essays on nine politically divisive issues in the United States: abortion, immigration, gun control, same-sex marriage, LGBTQ+ rights, climate change, trust in science, social media regulation, and free speech. These essays are generated under interventions with varying values of $\alpha \overset { \cdot } { \in } \left\{ - 5 0 , - 4 0 , - 3 \bar { 0 } , - 2 0 , - 1 0 , 0 , 1 0 , 2 0 , 3 0 , 4 0 , 5 0 \right\}$ . After generating responses, we use GPT-4o (gpt-4o-2024-08-06) to evaluate whether the LLM-generated responses are coherent and comprehensive. Specifically, we prompt GPT-4o as follows:

Is the following text incoherent or lacking comprehensiveness?

[ESSAY]

1: No, the text is coherent and comprehensive.   
2: Yes, the text is incoherent and lacks comprehensiveness.   
Please respond with a number (1 or 2) without providing reasons.

We find that if the value of α is smaller than -30 or bigger than 30, the proportion of coherent responses is always lower than 40% across three models (see Figure A8 for details). For instance, if we prompt the model to write an essay about abortion, intervening on the model activation with α = −50, Mistral-7b-instruct generates the following endlessly repetitive nonsense text.

• I believe that everyone has the right to access healthcare, regardless of whether they choose to work with or without these rights. However, the fact that they are not able to do so, or that they are not able to work, are not considered, or are not able to work, or are not able to work, or are not able to work, or are not able to work

Therefore, we conduct our main intervention experiments with α values ranging from −30 to 30.

# A.9 PRACTICAL APPLICATIONS

Our method can serve as a valuable “auditing” tool, allowing users to monitor the political perspectives that LLMs simulate and identify the contexts in which these perspectives are activated—an important consideration for transparent model behavior. Close-ended survey questions, such as the Political Compass Test (Feng et al., 2023) or Pew surveys (Santurkar et al., 2023), are frequently used as tools to monitor LLMs’ political biases. Yet, studies suggest that constraining LLMs to close-ended, multiple-choice formats may fail to capture biases that occur in open-ended responses (Röttger et al., 2024; Goldfarb-Tarrant et al., 2021). As shown in Figure 1, our approach provides an alternative way to monitor and assess the political perspectives employed by LLMs, enhancing transparency around potential biases in their open-ended outputs.

Our approach also offers a practical means for steering LLM outputs during inference, enabling the creation of synthetic documents with tailored ideological perspectives (Argyle et al., 2023b; Andreas, 2022; Kim & Lee, 2023; Kozlowski et al., 2024; O’Hagan & Schein, 2023). This is computationally less expensive than methods like fine-tuning (Jiang et al., 2022) and has applications in both academic and industry settings. For example, products such as Expected Parrot enable users to simulate human behaviors or opinions in silico (Expected Parrot, 2024), and our method could enhance these capabilities by providing fine-grained control over subjective perspectives.

# A.10 CROSS-NATIONAL ANALYSIS OF POLITICAL SLANT REPRESENTATIONS

In addition to our primary focus on U.S. political contexts, we conduct supplementary analyses to evaluate the generalizability of linear probes in predicting political slant across various non-U.S.

nations. This analysis utilizes the Manifesto Project dataset, which provides ideological labels y for 411 political parties worldwide on a left-to-right continuum (from -50 = left to $5 0 = \mathrm { r i g h t } )$ (Gemenis, 2013). Results reveal both strengths and limitations in extending the learned representations of political ideology to diverse cultural and national contexts.

We utilize the following prompt to simulate the ideological perspectives of lawmakers in these parties:

USER: Generate a statement by a lawmaker from the [PARTYNAME] party, which is a political party in [COUNTRYNAME].

ASSISTANT: In 2019, a lawmaker from the [PARTYNAME] party said that

As described in Section 3, we pass each of these prompts as input to Llama-2-7b-chat, and then record the activation $\pmb { x } _ { \ell , h } ^ { ( i ) }$ (i) of each attention head h in each layer ℓ. This yields a dataset of $( \boldsymbol { y } ^ { ( i ) } , \boldsymbol { x } _ { \ell , h } ^ { ( i ) } )$ pairs, where $\boldsymbol y ^ { ( i ) }$ is the Manifesto Project score of party i.

Then, we evaluate whether the probe previously trained to predict DW-NOMINATE for layer ℓ and head h is able to predict the Manifesto Project score $\boldsymbol y ^ { ( i ) }$ from $\pmb { x } _ { \ell , h } ^ { ( i ) }$ . We use Spearman rank correlation between the set of observed Manifesto Project scores $\{ y ^ { ( i ) } \}$ i and the ensembled predictions $\{ y _ { K } ^ { ( i ) } \} _ { i }$ , as defined in Equation (6), using the $K = 3 2$ heads that were most predictive of DW-NOMINATE.

As shown in Figure A9, we find that the linear probes exhibit modest performance in predicting the political slant of non-U.S. parties, achieving a Spearman correlation of $\rho = 0 . 5 3 1$ , substantially lower than the performance observed for U.S. lawmakers $( \rho = 0 . 8 7 0 )$ and U.S. news media $( \rho = 0 . 7 6 5 )$ . The generality of political slant representations varies significantly across nations. Some countries, such as New Zealand $( \rho = 0 . 9 2 0 )$ , Australia $( \rho = 0 . 9 1 6 )$ , Canada $( \rho = 0 . 8 8 3 )$ , and the United Kingdom $( \rho = 0 . 8 4 5 )$ , exhibit strong correlations, while others demonstrate weaker or even negative correlations, indicating that the applicability of learned representations is influenced by the political landscape and cultural context. These findings underscore the need for comprehensive datasets that capture diverse political contexts, particularly for underrepresented regions, and we encourage the AI research community to prioritize the creation of such datasets to improve the cross-cultural applicability of LLMs.

# A.11 REPLICATION ON Gemma-2-2b

Some might question whether political ideologies are similarly represented in the middle layers of LLMs outside the Llama family. We successfully replicated our analysis on the Gemma-2-2b model and found that it also exhibits a linear representation of ideological slant in its middle layers. See Figure A10 for details.

Table A1: Effect of regularization parameter λ on probe performance. We compare $\widehat { \rho } _ { \ell , h } ^ { \mathrm { C V } }$ at the most predictive attention head across models for different λ values. The value λ = 1 optimizes these metrics across the three models, except for Vicuna-7b. 

<table><tr><td>λ</td><td>Llama-2-7b-chat</td><td>Mistral-7b-instruct</td><td>Vicuna-7b</td></tr><tr><td>0</td><td>0.8182</td><td>0.8150</td><td>0.8284</td></tr><tr><td>0.001</td><td>0.8348</td><td>0.8163</td><td>0.8423</td></tr><tr><td>0.01</td><td>0.8437</td><td>0.8240</td><td>0.8447</td></tr><tr><td>0.1</td><td>0.8476</td><td>0.8395</td><td>0.8616</td></tr><tr><td>1</td><td>0.8536</td><td>0.8463</td><td>0.8612</td></tr><tr><td>100</td><td>0.8463</td><td>0.8434</td><td>0.8561</td></tr><tr><td>1000</td><td>0.8429</td><td>0.8462</td><td>0.8448</td></tr></table>

Figure A1: Ideological perspectives of U.S. lawmakers as captured by the activation space in Llama-2-7b-chat, Mistral-7b-instruct, and Vicuna-7b. Negative values correspond to leftleaning perspectives, while positive values correspond to right-leaning perspectives. Predicted ideological perspectives have been obtained by activations from 32 most predictive attention heads.   
![](images/467995e71f3affd0a682a125a78a48a5b1ec1022b3fde8d6faa7ab30937c7505.jpg)

<details>
<summary>scatter</summary>

| Predicted | Actual |
| --------- | ------ |
| -0.75     | -0.25  |
| -0.50     | -0.50  |
| -0.25     | -0.75  |
| 0.00      | -0.50  |
| 0.25      | -0.25  |
| 0.50      | 0.00   |
| 0.75      | 0.25   |
| 1.00      | 0.50   |
</details>

(a) Llama-2-7b $( \widehat { \rho } _ { K } ^ { \mathrm { C V } } = 0 . 8 7 0 )$

![](images/77e33009952bad908579a19317a1485adc10cf358c4376ec401040c3cc28108a.jpg)

<details>
<summary>scatter</summary>

| Predicted | Actual |
| --------- | ------ |
| -0.75     | -0.75  |
| -0.50     | -0.50  |
| -0.25     | -0.25  |
| 0.00      | 0.00   |
| 0.25      | 0.25   |
| 0.50      | 0.50   |
| 0.75      | 0.75   |
| 1.00      | 1.00   |
</details>

(b) Mistral-7b $( \widehat { \rho } _ { K } ^ { \mathrm { C V } } = 0 . 8 6 4 )$

![](images/fbd7c83bff37c5df10e190e2a8b11bc9ab2af134a1a298c7730c49f7e051934e.jpg)

<details>
<summary>scatter</summary>

| Predicted | Actual |
| --------- | ------ |
| -0.75     | -0.75  |
| -0.50     | -0.50  |
| -0.25     | -0.25  |
| 0.00      | 0.00   |
| 0.25      | 0.25   |
| 0.50      | 0.50   |
| 0.75      | 0.75   |
| 1.00      | 1.00   |
</details>

(c) Vicuna-7b $( \widehat { \rho } _ { K } ^ { \mathrm { C V } } = 0 . 8 8 5 )$

Figure A2: Ideological perspectives of U.S. news outlets as captured by the activation space in Llama-2-7b-chat, Mistral-7b-instruct, and Vicuna-7b. Negative values correspond to leftleaning perspectives, while positive values correspond to right-leaning perspectives. Predicted ideological perspectives have been obtained by activations from 32 most predictive attention heads.   
![](images/df67f7b935097394597f19610eee96eb76c4eb6b562d0c40df87cede901f310b.jpg)

<details>
<summary>scatter</summary>

| Predicted | Value  |
| --------- | ------ |
| -0.4      | -0.5   |
| -0.2      | -0.25  |
| 0.0       | 0.0    |
| 0.2       | 0.25   |
| 0.4       | 0.5    |
</details>

(a) Ll $. a m a - 2 - 7 b$ $( \widehat { \rho } _ { K } ^ { \mathrm { C V } } = 0 . 7 9 8 )$

![](images/4bc26f35022d3fbd079c55c4a77ddf23ea6b16f0fdecc68f2ba86b32d61b22a0.jpg)

<details>
<summary>scatter</summary>

| Predicted | Value  |
| --------- | ------ |
| -0.2      | -0.75  |
| 0.0       | -0.5   |
| 0.2       | 0.0    |
| 0.4       | 0.5    |
| 0.6       | 0.75   |
</details>

(b) Mi $s \mathrm { t r a l - } 7 \mathrm { b }$ $( \widehat { \rho } _ { K } ^ { \mathrm { C V } } = 0 . 7 6 4 )$

![](images/bf66928ac0478c51169c5c7dba6d433fcaa904e8c4f9dc9b3166b94d23d73992.jpg)

<details>
<summary>scatter</summary>

| Predicted | Value  |
| --------- | ------ |
| -0.2      | -0.5   |
| -0.1      | -0.3   |
| 0.0       | 0.0    |
| 0.1       | 0.2    |
| 0.2       | 0.4    |
| 0.3       | 0.6    |
| 0.4       | 0.8    |
</details>

(c) Vicuna-7b $( \widehat { \rho } _ { K } ^ { \mathrm { C V } } = 0 . 7 2 0 )$

Table A2: Top 10 attention heads according to cross-validation Spearman rank correlation. 

<table><tr><td rowspan="2">Rank</td><td colspan="2">Llama-2-7b-chat</td><td colspan="2">Mistral-7b-instruct</td><td colspan="2">Vicuna-7b</td></tr><tr><td> $(Layer, Head)$ </td><td> $Spearman \widehat{\rho}_{\ell,h}^{CV}$ </td><td> $(Layer, Head)$ </td><td> $Spearman \widehat{\rho}_{\ell,h}^{CV}$ </td><td> $(Layer, Head)$ </td><td> $Spearman \widehat{\rho}_{\ell,h}^{CV}$ </td></tr><tr><td>1</td><td>(15, 18)</td><td>0.8536</td><td>(16, 3)</td><td>0.8463</td><td>(24, 8)</td><td>0.8612</td></tr><tr><td>2</td><td>(16, 11)</td><td>0.8444</td><td>(16, 1)</td><td>0.8453</td><td>(22, 13)</td><td>0.8609</td></tr><tr><td>3</td><td>(18, 4)</td><td>0.8441</td><td>(18, 7)</td><td>0.8381</td><td>(17, 20)</td><td>0.8593</td></tr><tr><td>4</td><td>(15, 2)</td><td>0.8437</td><td>(27, 17)</td><td>0.8305</td><td>(26, 5)</td><td>0.8533</td></tr><tr><td>5</td><td>(17, 20)</td><td>0.8428</td><td>(15, 3)</td><td>0.8299</td><td>(16, 11)</td><td>0.8528</td></tr><tr><td>6</td><td>(15, 9)</td><td>0.8406</td><td>(16, 9)</td><td>0.8288</td><td>(18, 14)</td><td>0.8523</td></tr><tr><td>7</td><td>(26, 5)</td><td>0.8399</td><td>(15, 5)</td><td>0.8272</td><td>(23, 5)</td><td>0.8509</td></tr><tr><td>8</td><td>(16, 19)</td><td>0.8394</td><td>(14, 11)</td><td>0.8265</td><td>(20, 8)</td><td>0.8503</td></tr><tr><td>9</td><td>(14, 26)</td><td>0.8386</td><td>(22, 23)</td><td>0.8263</td><td>(29, 25)</td><td>0.8499</td></tr><tr><td>10</td><td>(16, 23)</td><td>0.8371</td><td>(11, 32)</td><td>0.8262</td><td>(14, 26)</td><td>0.8490</td></tr></table>

Table A3: Cross-validation Spearman rank correlation $( \widehat { \rho } _ { K } ^ { \mathrm { C V } } )$ for ensemble predictions given the number of attention heads (K). The value $K \ = \ 3 2$ optimizes these metrics, except for Mistral-7B-Instruct. 

<table><tr><td>Number of Attention Heads K</td><td>Llama-2-7b-chat</td><td>Mistral-7b-instruct</td><td>Vicuna-7b</td></tr><tr><td>1</td><td>0.8537</td><td>0.8468</td><td>0.8623</td></tr><tr><td>8</td><td>0.8695</td><td>0.8665</td><td>0.8850</td></tr><tr><td>32</td><td>0.8703</td><td>0.8636</td><td>0.8851</td></tr><tr><td>64</td><td>0.8684</td><td>0.8630</td><td>0.8813</td></tr><tr><td>96</td><td>0.8652</td><td>0.8591</td><td>0.8791</td></tr><tr><td>128</td><td>0.8625</td><td>0.8571</td><td>0.8766</td></tr><tr><td>256</td><td>0.8545</td><td>0.8496</td><td>0.8700</td></tr><tr><td>512</td><td>0.8476</td><td>0.8420</td><td>0.8634</td></tr></table>

Figure A3: DW-NOMINATE scores and transformed DW-NOMINATE scores as predicted by linear probes on the activations of Llama-2-7b-chat   
![](images/82d6d69705ecc1ee755ff1feb09acece425147ca543b711fc26e4dd76d91a603.jpg)

<details>
<summary>scatter</summary>

| Predicted | Actual |
| --------- | ------ |
| -1.0      | -0.5   |
| -0.5      | -0.2   |
| 0.0       | 0.0    |
| 0.5       | 0.3    |
| 1.0       | 0.8    |
</details>

(a) DW-NOMINATE

![](images/84cecb56ce6d56ea9c69f9888153565be300170665a82397512d76fed846cbb2.jpg)

<details>
<summary>scatter</summary>

| Predicted | Actual |
| --------- | ------ |
| -0.1      | 0.3    |
| 0.0       | 0.4    |
| 0.1       | 0.5    |
| 0.2       | 0.6    |
| 0.3       | 0.7    |
| 0.4       | 0.8    |
| 0.5       | 0.9    |
| 0.6       | 1.0    |
| -0.2      | -0.3   |
| 0.1       | -0.4   |
| 0.2       | -0.5   |
| 0.3       | -0.6   |
| 0.4       | -0.7   |
| 0.5       | -0.8   |
| 0.6       | -0.9   |
| 0.7       | -1.0   |
</details>

(b) Randomized DW-NOMINATE

![](images/456e5fd06b67a5db8eac3213b8c2d24598c2d279c1a40c85f06500c0b960f837.jpg)

<details>
<summary>scatter</summary>

| Predicted | Actual |
| --------- | ------ |
| -0.8      | -0.9   |
| -0.6      | -0.7   |
| -0.4      | -0.5   |
| -0.2      | -0.3   |
| 0.0       | -0.1   |
| 0.2       | 0.1    |
| 0.4       | 0.3    |
| 0.6       | 0.5    |
| 0.8       | 0.7    |
| 1.0       | 0.9    |
</details>

(c) sin(10 ∗ DW-NOMINATE)

![](images/8f07689359feede9a23f301da0286fc27a6e27c004d43d58ccc531b5b3cb8d8e.jpg)

<details>
<summary>scatter</summary>

| Predicted | Actual |
| --------- | ------ |
| -0.5      | -0.5   |
| 0.0       | 0.0    |
| 0.5       | 0.5    |
</details>

(d) (DW-NOMINATE)3

Table A4: Spearman correlation $\widehat { \rho } _ { \ell , h } ^ { \mathrm { C V } }$ and R-squared score $\widehat { R ^ { 2 } } _ { \ell , h } ^ { \mathrm { C V } }$ for different models and transformations of DW-NOMINATE scores. 

<table><tr><td rowspan="2"></td><td colspan="4">Spearman correlation ( $\widehat{\rho}_{\ell,h}^{\text{CV}}$ )</td></tr><tr><td> $y^{(i)}$ </td><td>Randomized  $y^{(i)}$ </td><td> $\sin(10y^{(i)})$ </td><td> $(y^{(i)})^3$ </td></tr><tr><td>Llama2-7b-chat</td><td>0.854</td><td>0.152</td><td>0.551</td><td>0.842</td></tr><tr><td>Mistral-7b-instruct</td><td>0.846</td><td>0.140</td><td>0.534</td><td>0.841</td></tr><tr><td>Vicuna-7b</td><td>0.861</td><td>0.156</td><td>0.558</td><td>0.860</td></tr><tr><td rowspan="2"></td><td colspan="4">R-squared score ( $\widehat{R^{2}_{\ell,h}^{\text{CV}}}$ )</td></tr><tr><td> $y^{(i)}$ </td><td>Randomized  $y^{(i)}$ </td><td> $\sin(10y^{(i)})$ </td><td> $(y^{(i)})^3$ </td></tr><tr><td>Llama2-7b-chat</td><td>0.821</td><td>0.019</td><td>0.298</td><td>0.613</td></tr><tr><td>Mistral-7b-instruct</td><td>0.832</td><td>0.025</td><td>0.260</td><td>0.601</td></tr><tr><td>Vicuna-7b</td><td>0.830</td><td>0.012</td><td>0.303</td><td>0.626</td></tr></table>

Figure A4: Intervention workflow. Squares indicate natural language texts. Circles indicate vectors.

# A. Probing

1. Prompting an LLM to generate statements by politicians with DW-NOMINATE scores

2. Extracting activations from each attention head

3. For each attention head, training a linear probe to predict DW-NOMINATE scores

![](images/20f10e8acf37a72f1c86d428400ca29ba19e2b2dfe762b87f161baa025283f0b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Prompts: E.g., Generate a statement by Kamala Harris"] --> B["LLM"]
    C["Prompts: E.g., Generate a statement by Donald Trump"] --> D["LLM"]
    B --> E["x(ℓ,h)(i)"]
    D --> F["x(ℓ,h)(i)"]
```
</details>

$$
\widehat {y} _ {\ell , h} ^ {(i)} \triangleq \pmb {\theta} _ {\ell , h} ^ {\top} \pmb {x} _ {\ell , h} ^ {(i)}
$$

# B. Intervention

4. Intervention using the learned regression coefficient $\widehat { \pmb { \theta } } _ { \ell , \boldsymbol { I } }$ which captures the ideological direction in the activation s

5. Evaluate the steered outputs using GPT-4o

![](images/ab7753330cbc5c48100e5df2d3f7c81406d894c301b97477fa99c0a2e51bfd89.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["User prompt: Write an essay about immigration"] --> B["LLM"]
    C["Steered output"] --> B
    B --> D["1=Extreme Liberal ~7=Extreme Conservative"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style D fill:#cff,stroke:#333
```
</details>

Figure A5: Intervention (α) and political slant reflected in the statement by the number of attention heads intervened (i.e., K).   
![](images/0b8870b540f65a1d1089cbc4927fa79bcf6f66760eed7d62b71f8ae889dccddb.jpg)

<details>
<summary>line</summary>

| Intervention (Alpha) | 16   | 32   | 48   | 64   | 80   | 96   |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| -30                  | 2.7  | 2.3  | 2.4  | 2.5  | 2.4  | 2.4  |
| -20                  | 2.3  | 2.4  | 2.5  | 2.1  | 2.0  | 2.0  |
| -10                  | 2.4  | 2.5  | 2.6  | 2.3  | 2.3  | 2.3  |
| 0                    | 2.6  | 2.7  | 2.8  | 2.6  | 2.6  | 2.6  |
| 10                   | 2.7  | 2.8  | 2.9  | 2.9  | 3.0  | 3.1  |
| 20                   | 2.9  | 3.1  | 3.5  | 3.5  | 4.1  | 4.2  |
| 30                   | 3.0  | 3.2  | 4.1  | 4.0  | 4.6  | 4.7  |
</details>

Figure A6: Distribution of political slant $( \widehat { y } _ { \ell , h } ^ { ( i , t ) } )$ token by token.   
![](images/af061d4146558ab090a5620040d07a9e53ebce74c4a41a40e5b6ac6fd3527f31.jpg)

<details>
<summary>line</summary>

| Political Slant | Mistral-7b | Vicuna-7b | Llama-2-7b |
| --------------- | ---------- | --------- | ---------- |
| -2.0            | 0.0        | 0.0       | 0.0        |
| -1.5            | 0.0        | 0.0       | 0.0        |
| -1.0            | 0.0        | 0.0       | 0.0        |
| -0.5            | 0.2        | 0.3       | 0.3        |
| 0.0             | 1.1        | 1.6       | 1.5        |
| 0.5             | 0.4        | 0.3       | 0.4        |
| 1.0             | 0.1        | 0.1       | 0.1        |
| 1.5             | 0.0        | 0.0       | 0.0        |
| 2.0             | 0.0        | 0.0       | 0.0        |
</details>

Figure A7: Intervention (α) and political slant are reflected in the statements by the targeted layers for Llama-2-7b-chat, Mistral-7b-instruct, and Vicuna-7b $( K = 9 6 )$ . Layers $< 2 2$ indicate interventions in the early to middle layers, while Layers $\geq 2 2$ indicate interventions in the middle to last layers. Compared to interventions in Layers $< 2 2 ( \rho = 0 . 5 4 0 )$ , interventions in Layers $\geq 2 2$ does not manifest a significant effect $( \rho = - 0 . 0 2 2 )$ .   
![](images/f892e0fceb4a8a0ad6e6f932132fffceb86227406621e2d608e4661ae67568f0.jpg)

<details>
<summary>line</summary>

| Intervention (alpha) | Layer<22 (ρ=0.540) | Layer≥22 (ρ=-0.022) |
| --------------------- | ------------------- | --------------------- |
| -30                   | 2.2                 | 2.5                   |
| -20                   | 2.1                 | 2.6                   |
| -10                   | 2.2                 | 2.7                   |
| 0                     | 2.8                 | 2.8                   |
| 10                    | 3.3                 | 2.6                   |
| 20                    | 4.1                 | 2.5                   |
| 30                    | 4.8                 | 2.4                   |
</details>

Figure A8: Proportion of coherent LLM responses by α.   
![](images/9a57b28469eedcad22207e3ab6f220953b0f2b6fe188066068ae9491fd602b87.jpg)

<details>
<summary>line</summary>

| Intervention (Alpha) | Llama-2-7b | Mistral-7b | Vicuna-7b |
| -------------------- | ---------- | ---------- | --------- |
| -50                  | 0          | 25         | 0         |
| -40                  | 10         | 35         | 0         |
| -30                  | 55         | 95         | 55        |
| -20                  | 90         | 98         | 90        |
| -10                  | 88         | 95         | 92        |
| 0                    | 85         | 95         | 85        |
| 10                   | 88         | 95         | 87        |
| 20                   | 67         | 88         | 82        |
| 30                   | 38         | 92         | 47        |
| 40                   | 10         | 25         | 33        |
| 50                   | 17         | 18         | 17        |
</details>

Figure A9: A, Ideological perspectives of political parties outside the U.S. captured by the activation space in Llama-2-7b-chat. Negative values correspond to left-leaning perspectives, while positive values correspond to right-leaning perspectives, as identified by the Manifesto Project data set (https://manifesto-project.wzb.eu/), which labeled 411 political parties in 2017. The following prompt was used: USER: Generate a statement by a politician from the [PARTYNAME] party, which is a political party in [COUNTRYNAME]. ASSISTANT: In 2019, a lawmaker from the [PARTYNAME] party said that. B, Predictive performance of linear probes by nation. Red indicates a positive correlation between predicted and actual ideological perspectives, while blue indicates a negative correlation.   
A   
![](images/90d44018d1b4b2f3f1fc44d8f8f622320e0301d7dc0b8bd544ec02906e6f1e72.jpg)

<details>
<summary>scatter</summary>

| Party | Actual | Predicted |
| --- | --- | --- |
| New Zealand Labour Party (New Zealand) | 0.15 | -0.08 |
| Democratic Party (Italy) | 0.22 | -0.06 |
| Liberal Party of Australia (Australia) | 0.35 | 0.12 |
| Conservative Party (United Kingdom) | 0.28 | 0.18 |
| Democratic Party of Germany (Germany) | 0.18 | 0.09 |
| Labour Party (Lithuania) | 0.12 | 0.07 |
| Democratic Party (South Korea) | 0.14 | 0.05 |
| Labour Party (Netherlands) | 0.08 | -0.04 |
| Labour Party (United Kingdom) | 0.05 | -0.02 |
| New Zealand Labour Party (New Zealand) | 0.25 | -0.12 |
| Democratic Party (United States) | 0.16 | -0.15 |
| Liberal Party of Australia (Japan) | 0.32 | 0.15 |
| Conservative Party (United Kingdom) | 0.25 | 0.18 |
| Labour Party (Lithuania) | 0.18 | -0.03 |
| Labour Party (Netherlands) | 0.14 | -0.06 |
| Democratic Party (South Korea) | 0.17 | -0.09 |
| New Zealand Labour Party (New Zealand) | 0.28 | -0.18 |
| Democratic Party of Germany (Germany) | 0.19 | 0.06 |
| Labour Party (Lithuania) | 0.13 | -0.05 |
| Labour Party (Netherlands) | 0.11 | -0.07 |
| Democratic Party (South Korea) | 0.15 | -0.04 |
| New Zealand Labour Party (New Zealand) | 0.38 | -0.22 |
| Democratic Party of Germany (Germany) | 0.21 | 0.14 |
| Labour Party (Lithuania) | 0.16 | -0.06 |
| Labour Party (Netherlands) | 0.12 | -0.08 |
| Democratic Party (South Korea) | 0.19 | -0.11 |
| New Zealand Labour Party (New Zealand) | 0.36 | -0.25 |
| Democratic Party of Germany (Germany) | 0.23 | 0.16 |
| Labour Party (Lithuania) | 0.14 | -0.07 |
| Labour Party (Netherlands) | 0.13 | -0.09 |
| Democratic Party (South Korea) | 0.24 | -0.14 |
| New Zealand Labour Party (New Zealand) | 0.37 | -0.28 |
| Democratic Party of Germany (Germany) | 0.26 | 0.17 |
| Labour Party (Lithuania) | 0.17 | -0.08 |
| Labour Party (Netherlands) | 0.15 | -0.11 |
| Democratic Party (South Korea) | 0.27 | -0.16 |
| New Zealand Labour Party (New Zealand) | 0.39 | -0.29 |
| Democratic Party of Germany (Germany) | 0.25 | 0.13 |
| Labour Party (Lithuania) | 0.18 | -0.12 |
| Labour Party (Netherlands) | 0.16 | -0.14 |
| Democratic Party (South Korea) | 0.28 | -0.19 |
| New Zealand Labour Party (New Zealand) | 0.35 | -0.24 |
| Democratic Party of Germany (Germany) | 0.24 | 0.15 |
| Labour Party (Lithuania) | 0.19 | -0.13 |
| Labour Party (Netherlands) | 0.17 | -0.16 |
| Democratic Party (South Korea) | 0.26 | -0.27 |
| New Zealand Labour Party (New Zealand) | 0.33 | -0.26 |
| Democratic Party of Germany (Germany) | 0.23 | 0.14 |
| Labour Party (Lithuania) | 0.16 | -0.14 |
| Labour Party (Netherlands) | 0.14 | -0.17 |
| Democratic Party (South Korea) | 0.29 | -0.23 |
| New Zealand Labour Party (New Zealand) | 0.34 | -0.27 |
| Democratic Party of Germany (Germany) | 0.27 | 0.16 |
| Labour Party (Lithuania) | 0.17 | -0.15 |
| Labour Party (Netherlands) | 0.15 | -0.18 |
| Democratic Party (South Korea) | 0.31 | -0.25 |
| New Zealand Labour Party (New Zealand) | 0.36 | -0.28 |
| Democratic Party of Germany (Germany) | 0.25 | 0.17 |
| Labour Party (Lithuania) | 0.18 | -0.16 |
| Labour Party (Netherlands) | 0.16 | -0.19 |
| Democratic Party (South Korea) | 0.32 | -0.24 |
| New Zealand Labour Party (New Zealand) | 0.37 | -0.29 |
| Democratic Party of Germany (Germany) | 0.26 | 0.18 |
| Labour Party (Lithuania) | 0.19 | -0.17 |
| Labour Party (Netherlands) | 0.17 | -0.21 |
| Democratic Party (South Korea) | 0.34 | -0.26 |
| New Zealand Labour Party (New Zealand) | 0.38 | -0.31 |
| Democratic Party of Germany (Germany) | 0.28 | 0.19 |
| Labour Party (Lithuania) | 0.16 | -0.14 |
| Labour Party (Netherlands) | 0.14 | -0.17 |
| Democratic Party (South Korea) | 0.35 | -0.28 |
| New Zealand Labour Party (New Zealand) | 0.39 | -0.33 |
| Democratic Party of Germany (Germany) | 0.27 | 0.15 |
| Labour Party (Lithuania) | 0.17 | -0.15 |
| Labour Party (Netherlands) | 0.15 | -0.18 |
| Democratic Party (South Korea) | 0.36 | -0.24 |
| New Zealand Labour Party (New Zealand) | 0.33 | -0.27 |
| Democratic Party of Germany (Germany) | 0.29 | 0.14 |
| Labour Party (Lithuania) | 0.18 | -0.16 |
| Labour Party (Netherlands) | 0.16 | -0.19 |
| Democratic Party (South Korea) | 0.34 | -0.25 |
| New Zealand Labour Party (New Zealand) | 0.37 | -0.29 |
| Democratic Party of Germany (Germany) | 0.25 | 0.16 |
| Labour Party (Lithuania) | 0.19 | -0.17 |
| Labour Party (Netherlands) | 0.17 | -0.21 |
| Democratic Party (South Korea) | 0.38 | -0.26 |
| New Zealand Labour Party (New Zealand) | 0.35 | -0.28 |
| Democratic Party of Germany (Germany) | 0.24 | 0.17 |
| Labour Party (Lithuania) | 0.16 | -0.18 |
| Labour Party (Netherlands) | 0.14 | -0.22 |
| Democratic Party (South Korea) | 0.37 | -0.27 |
| New Zealand Labour Party (New Zealand) | 0.36 | -0.29 |
| Democratic Party of Germany (Germany) | 0.26 | 0.15 |
| Labour Party (Lithuania) | 0.17 | -0.16 |
| Labour Party (Netherlands) | 0.15 | -0.19 |
| Democratic Party (South Korea) | 0.39 | -0.24 |
| New Zealand Labour Party (New Zealand) | 0.34 | -0.26 |
| Democratic Party of Germany (Germany) | 0.28 | 0.14 |
| Labour Party (Lithuania) | 0.18 | -0.17 |
| Labour Party (Netherlands) | 0.16 | -0.21 |
| Democratic Party (South Korea) | 0.35 | -0.25 |
| New Zealand Labour Party (New Zealand) | 0.38 | -0.29 |
| Democratic Party of Germany (Germany) | 0.27 | 0.16 |
| Labour Party (Lithuania) | 0.19 | -0.18 |
| Labour Party (Netherlands) | 0.17 | -0.22 |
| Democratic Party (South Korea) | 0.36 | -0.26 |
| New Zealand Labour Party (New Zealand) | 0.33 | -0.28 |
| Democratic Party of Germany (Germany) | 0.29 | 0.18 |
| Labour Party (Lithuania) | 0.16 | -0.19 |
| Labour Party (Netherlands) | 0.14 | -0.23 |
| Democratic Party (South Korea) | 0.34 | -0.27 |
| New Zealand Labour Party (New Zealand) | 0.37 | -0.29 |
</details>

B

![](images/108fff7ed0499b588ce9bc8e9d3a61b1c63bda7b9753d7a8c6872a9f631487ae.jpg)

<details>
<summary>heatmap</summary>

| Country       | Value  |
| ------------- | ------ |
| North America | -0.75  |
| Europe        | -0.25  |
| Asia          | -0.25  |
| Africa        | -0.75  |
| Australia     | -0.75  |
| South America | -0.25  |
| Middle East   | -0.25  |
| Central Asia  | -0.25  |
| South Asia    | -0.25  |
| Oceania       | -0.25  |
| Russia        | -0.25  |
| Mexico        | -0.25  |
| Brazil        | -0.25  |
| Argentina     | -0.25  |
| Canada        | -0.25  |
| United States | -0.25  |
| Germany       | -0.25  |
| France        | -0.25  |
| Italy         | -0.25  |
| Spain         | -0.25  |
| United Kingdom| -0.25  |
| Ireland       | -0.25  |
| Netherlands   | -0.25  |
| Belgium       | -0.25  |
| Switzerland   | -0.25  |
| Austria       | -0.25  |
| Poland        | -0.25  |
| Ukraine       | -0.25  |
| Belarus       | -0.25  |
| Moldova       | -0.25  |
| Georgia       | -0.25  |
| Armenia       | -0.25  |
| Azerbaijan    | -0.25  |
| Kazakhstan    | -0.25  |
| Uzbekistan    | -0.25  |
| Iraq          | -0.25  |
| Iran          | -0.25  |
| Saudi Arabia  | -0.25  |
| Myanmar       | -0.25  |
| Cambodia      | -0.25  |
| Laos          | -0.25  |
| Brunei        | -0.25  |
| New Zealand   | -0.25  |
| Singapore     | -0.25  |
| Malaysia      | -0.25  |
| Thailand      | -0.25  |
| Philippines   | -0.25  |
| Vietnam       | -0.25  |
| Indonesia     | -0.25  |
| Philippines   | -0.75  |
| Malaysia      | -0.75  |
| Thailand      | -0.75  |
| Vietnam       | -0.75  |
| Philippines   | -0.75  |
| Malaysia      | -0.75  |
| Vietnam       | -0.75  |
| Philippines   | -0.75  |
| Malaysia      | -0.75  |
| Vietnam       | -0.75  |
| Philippines   | -0.75  |
| Malaysia      | -0.75  |
| Vietnam       | -0.75  |
| Philippines   | -0.75  |
| Malaysia     | -0.75  |
| Vietnam       | -0.75  |
| Philippines   | -0.75  |
| Malaysia      | -0.75  |
| Vietnam       | -0.75  |
| Philippines   | -0.75  |
| Malaysia     | -0.75  |
| Vietnam       | -0.75  |
| Philippines   | -0.75  |
| Malaysia     | -0.75  |
| Vietnam       | -0.75  |
| Philippines   | -0.75  |
| Malaysia     | -0.75  |
| Vietnam       | -0.75  |
| Philippines   | -0.75  |
| Malaysia     }<fcel>-0.75   |
| Thailand      }<fcel>-0.75   |
| India         | -0.75   |
| China         | -0.75   |
| Japan         | -0.75   |
| Germany       | -0.75   |
| United States | -0.75   |
| United Kingdom| -0.75   |
| France        | -0.75   |
| Italy         | -0.75   |
| Spain         | -0.75   |
| Portugal      | -0.75   |
| Greece        | -0.75   |
| Sweden        | -0.75   |
| Norway        | -0.75   |
| Denmark       | -0.75   |
| Finland       | -0.75   |
| Iceland       | -0.75   |
| Greenland     | -0.75   |
| Faroe Islands | -0.75   |
| Tuvalu        | -0.75   |
| Grenada       | -0.75   |
| Samoa         | -0.75   |
| Kiribati      | -0.75   |
| French Guiana | -0.75   |
| Montenegro    | -0.75   |
| Cyprus        | -0.75   |
| Malta         | -0.75   |
| Latvia        | -0.75   |
| Lithuania     | -0.75   |
| Luxembourg    | -0.75   |
| Monaco        | -0.75   |
| Andorra       | -0.75   |
| Benin         | -0.75   |
| Czech Republic| -0.75   |
| Slovakia      | -0.75   |
| Hungary       | -0.75   |
| Romania       | -0.75   |
| Bulgaria      | -0.75   |
| Serbia        | -0.75   |
| Croatia       | -0.75   |
| Slovenia      | -0.75   |
| Montenegro    | -0.75   |
| Bosnia and Herzegovina | -0.75|
| Serbia        | -0.75   |
| Montenegro    | -0.75   |
| Bosnia and Herzegovina | -0.75|
| Serbia        | -0.75   |
| Montenegro    | -0.75   |
| Bosnia and Herzegovina (UK) | -1|
| Serbia        | -1|
| Montenegro    | -1|
| Bosnia and Herzegovina (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK) (UK)<nl>
</details>

Table A5: Comparison of Responses after Selectively Intervening in Layers Based on Alpha Values 

<table><tr><td>Alpha</td><td>Layer &lt; 22</td><td>Layer ≥ 22</td></tr><tr><td>-20</td><td>Regarding same-sex marriage, I believe that love knows no boundaries of gender. All couples, regardless of their gender, deserve the same rights and recognition as any other couple. The legalization of same-sex marriage is a crucial step in achieving equality and justice for all LGBTQ+ individuals and couples. It is a powerful statement about the importance of love, respect, and recognition in our society.</td><td>Regarding same-sex marriage, I believe that all individuals, regardless of their sexual orientation, should have the right to marry and have their relationships recognized and respected by the law. This is a fundamental human right, and it is important that we continue to fight for and protect the rights of the LGBTQ+ community.</td></tr><tr><td>20</td><td>Regarding same-sex marriage, I believe that all individuals, regardless of their sexual orientation, should be treated with dignity and respect. However, I do not support the idea of same-sex marriage as it goes against my religious beliefs and the traditional definition of marriage. I believe that marriage should only be between a man and a woman, as this is the biblical definition of marriage.</td><td>Regarding same-sex marriage, I believe that all individuals should have the right to marry the person they love and have their relationship recognized by the government. The ability to marry the person of one&#x27;s choice is a fundamental human right, and it is not the government&#x27;s place to dictate who someone can or cannot marry.</td></tr></table>

Figure A10: Predictive performance of linear probes for each attention head across all layers in gemma-2-2b. Performance is measured using Spearman rank correlation, with darker shades indicating stronger correlations.

![](images/11fee168b07cc457b2da30eb57e556d3e4be882e49c9afa20e9d0dcd2ab72c3a.jpg)

<details>
<summary>heatmap</summary>

| Layer | Head (Sorted) | Value |
|-------|---------------|-------|
| 2     | 0             | 0.0   |
| 4     | 0             | 0.0   |
| 6     | 0             | 0.0   |
| 8     | 0             | 0.0   |
| 10    | 0             | 0.0   |
| 12    | 0             | 0.0   |
| 14    | 0             | 0.0   |
| 16    | 0             | 0.0   |
| 18    | 0             | 0.0   |
| 20    | 0             | 0.0   |
| 22    | 0             | 0.0   |
| 24    | 0             | 0.0   |
| 26    | 0             | 0.0   |
| 28    | 0             | 0.0   |
| 30    | 0             | 0.0   |
| 32    | 0             | 0.0   |
</details>