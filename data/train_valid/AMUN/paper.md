# Not All Wrong is Bad: Using Adversarial Examples for Unlearning

Ali Ebrahimpour-Boroojeny 1 Hari Sundaram 1 Varun Chandrasekaran 1

# Abstract

Machine unlearning, where users can request the deletion of a forget dataset, is becoming increasingly important because of numerous privacy regulations. Initial works on “exact” unlearning (e.g., retraining) incur large computational overheads. However, while computationally inexpensive, “approximate” methods have fallen short of reaching the effectiveness of exact unlearning: models produced fail to obtain comparable accuracy and prediction confidence on both the forget and test (i.e., unseen) dataset. Exploiting this observation, we propose a new unlearning method, Adversarial Machine UNlearning (AMUN), that outperforms prior state-of-the-art (SOTA) methods for image classification. AMUN lowers the confidence of the model on the forget samples by fine-tuning the model on their corresponding adversarial examples. Adversarial examples naturally belong to the distribution imposed by the model on the input space; fine-tuning the model on the adversarial examples closest to the corresponding forget samples (a) localizes the changes to the decision boundary of the model around each forget sample and (b) avoids drastic changes to the global behavior of the model, thereby preserving the model’s accuracy on test samples. Using AMUN for unlearning a random 10% of CIFAR-10 samples, we observe that even SOTA membership inference attacks cannot do better than random guessing.

# 1. Introduction

The goal of machine unlearning is to remove the influence of a subset of the training dataset for a model that has been trained on that dataset (Vatter et al., 2023). The necessity for these methods has been determined by privacy regulations such as the European Union’s General Data Protection Act 1University of Illinois at Urbana-Champaign (UIUC), Illinois, USA. Correspondence to: Ali Ebrahimpour-Boroojeny <ae20@illinois.edu>.

Proceedings of the $4 2 ^ { n d }$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

and the California Consumer Privacy Act. Despite early efforts on proposing “exact” solutions to this problem (Cao & Yang, 2015; Bourtoule et al., 2021), the community has favored “approximate” solutions due to their ability to preserve the original model’s accuracy while being more computationally efficient (Chen et al., 2023b; Liu et al., 2024; Fan et al., 2024).

Given a training set D and a subset $\mathcal { D } _ { \mathrm { F } } \subset \mathcal { D }$ of the samples that have to be unlearned from a model trained on D, recent works on unlearning have emphasized the use of evaluation metrics that measure the similarity to the behavior of the models that are retrained from scratch on $\mathcal { D } - \mathcal { D } _ { \mathrm { F } }$ . However, prior unlearning methods do not effectively incorporate this evaluation criterion in the design of their algorithm. In this paper, we first characterize the expected behavior of the retrained-from-scratch models on $\mathcal { D } - \mathcal { D } _ { \mathrm { F } }$ . Using this characterization, we propose Adversarial Machine UNlearning (AMUN). AMUN is a method that, when applied to the models trained on $\mathcal { D } ,$ , replicates that (desired) behavior after a few iterations. The success of AMUN relies on an intriguing observation: fine-tuning a trained model on the adversarial examples of the training data does not lead to a catastrophic forgetting and instead has limited effect on the deterioration of model’s test accuracy.

Upon receiving a request for unlearning a subset $\mathcal { D } _ { \mathrm { F } }$ of the training set $\mathcal { D } ,$ AMUN finds adversarial examples that are as close as possible to the samples in $\mathcal { D } _ { \mathrm { F } }$ . It then utilizes these adversarial examples (with the wrong labels) during fine-tuning of the model for unlearning the samples in $\mathcal { D } _ { \mathrm { F } }$ . Fine-tuning the model on these adversarial examples, which are naturally mispredicted by the model, decreases the confidence of the predictions on $\mathcal { D } _ { \mathrm { F } }$ . This decreased confidence of model’s predictions on $\mathcal { D } _ { \mathrm { F } }$ is similar to what is observed in the models that are retrained on $\mathcal { D } - \mathcal { D } _ { \mathrm { F } }$ . The distance of these adversarial examples to their corresponding samples in $\mathcal { D } _ { \mathrm { F } }$ is much smaller than the distance of $\mathcal { D } _ { \mathrm { F } }$ to other samples in $\mathcal { D } - \mathcal { D } _ { \mathrm { { F } } } ;$ ; this localizes the effect of fine-tuning to the vicinity of the samples in $\mathcal { D } _ { \mathrm { F } }$ and prevents significant changes to the decision boundary of the model and hurting the model’s overall accuracy (see § 3.1).

As we will show in § 6, AMUN outperforms prior stateof-the-art (SOTA) unlearning methods (Fan et al., 2024) in unlearning random subsets of the training data from a trained classification model and closes the gap with the retrained models, even when there is no access to the samples in $\mathcal { D } - \mathcal { D } _ { \mathrm { F } }$ during the unlearning procedure.

To summarize, the main contributions of this work are:

• We observe that neural networks, when fine-tuned on adversarial examples with their wrong labels, have limited test accuracy degradation. While prior research in adversarial robustness fine-tune the models on these samples with their labels corrected, we are the first to utilize this form of fine-tuning to get lower prediction confidence scores on the training samples that are present in the proximity of those adversarial examples.   
• We introduce a new unlearning method, AMUN, for classification models that outperforms prior methods. It does so by replicating the behavior of the retrained models on the test samples and the forget samples.   
• By comparing AMUN to existing unlearning methods using SOTA membership inference attacks (MIAs), we show that it outperforms the other methods in unlearning subsets of training samples of various sizes.

All code can be found in https://github.com/ $\mathtt { A l i - E / A M U N }$

# 1.1. Related Work

Early works in machine unlearning focused on exact solutions (Cao & Yang, 2015; Bourtoule et al., 2021); those ideas were adapted to unlearning in other domains such as graph neural networks (Chen et al., 2022b) and recommendation systems (Chen et al., 2022a). The extensive computational cost and utility loss resulted in the design of approximate methods. An example is the work of Ginart et al. (2019), who provide a definition of unlearning based on differential privacy. Works that followed sought solutions to satisfy those probabilistic guarantees (Ginart et al., 2019; Gupta et al., 2021; Neel et al., 2021; Ullah et al., 2021; Sekhari et al., 2021). However, the methods that satisfy these guarantees were only applied to simple models, such as k-means (Ginart et al., 2019) , linear and logistic regression (Guo et al., 2019; Izzo et al., 2021), convexoptimization problems (Neel et al., 2021), or graph neural networks with no non-linearities (Chien et al., 2022). Additional research was carried out to design more scalable approximate methods, those that can be applied to the models that are used in practice, including large neural networks (Golatkar et al., 2020; Warnecke et al., 2021; Izzo et al., 2021; Thudi et al., 2022; Chen et al., 2023b; Liu et al., 2024; Fan et al., 2024). However, these approximate methods do not come with theoretical guarantees; their effectiveness are evaluated using membership inference attacks (MIAs). MIAs aim to determine whether a specific data sample was used in the training set of a trained model (Shokri et al., 2017; Yeom et al., 2018; Song et al., 2019; Hu et al., 2022; Carlini et al., 2022; Zarifzadeh et al., 2024), and is a common evaluation metric (Liu et al., 2024; Fan et al., 2024). For further discussion on related works, see Appendix B.

# 2. Preliminaries

We begin by introducing the notation we use. We proceed to define various terms in the paper, and conclude by introducing our method.

# 2.1. Notation

Assume a probability distribution $\mathbb { P } _ { \mathcal { X } }$ on the domain of inputs X and m classes $\mathcal { Y } = \{ 1 , 2 , \dots , m \}$ . We consider a multi-class classifier $\mathcal { F } : \mathcal { X }  \mathcal { Y }$ and its corresponding prediction function f (x) which outputs the probabilities corresponding to each class (e.g., the outputs of the softmax layer in a neural network). The loss function for model $\mathcal { F }$ is denoted $\ell _ { \mathcal { F } } : \mathcal { X } \times \mathcal { Y }  \mathbb { R } _ { + } ;$ ; it uses the predicted scores from $f ( x )$ to compute the loss given the true label $y \left( \mathrm { { e . g . } } \right.$ , cross-entropy loss). In the supervised setting we consider here, we are given a dataset $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = \{ 1 , . . . , N \} }$ that contains labeled samples $x _ { i } \sim \mathbb { P } _ { \mathcal { X } }$ with $y _ { i } ~ \in ~ \mathcal { V }$ . The model $\mathcal { F }$ is trained on D using the loss $\ell _ { \mathcal { F } } ~ \mathrm { t o }$ minimize the empirical risk $\mathbb { E } _ { \mathcal { D } } [ \ell _ { \mathcal { F } } ( x , y ) ]$ and a set of parameters $\theta _ { \mathrm { o } } \sim \Theta _ { D }$ is derived for $\mathcal { F } ; \Theta _ { \mathcal { D } }$ is the distribution over the set of all possible parameters Θ when the training procedure is performed on D due to the potential randomness in the training procedure (e.g., initialization and using mini-batch training). We also assume access to a test set $\mathcal { D } _ { \mathrm { T } }$ with samples from the same distribution $\mathbb { P } _ { \mathcal { X } \cdot \mathrm { ~ A ~ } }$ function $g ( x )$ is L-Lipschitz if $\| g ( x ) - g ( x ^ { \prime } ) \| _ { 2 } \leq L \| x - x ^ { \prime } \| _ { 2 } , \forall x , x ^ { \prime } \in \mathcal { X }$ .

# 2.2. Definitions

Definition 2.1 (Attack Algorithm). For a given input/output pair $( x , y ) \in \mathcal { X } \times \mathcal { Y }$ , a model F, and a positive value ϵ, an untargetted attack algorithm $\mathcal { A } _ { \mathcal { F } } ( x , \epsilon ) = x + \delta _ { x }$ minimizes $\ell _ { \mathcal { F } } ( x + \delta _ { x } , y ^ { \prime } \neq y )$ such that $\lVert \delta _ { x } \rVert _ { 2 } \leq \epsilon ,$ where $y ^ { \prime } \in \mathcal { V }$ .

Definition 2.2 (Machine Unlearning). Given the trained model ${ \mathcal F } ,$ , and a subset $\mathcal { D } _ { \mathrm { F } } \subset \mathcal { D }$ known as the forget set, the corresponding machine unlearning method is a function $\mathcal { M } _ { \mathcal { D } , \mathcal { D } _ { \mathrm { F } } } : \Theta \to \Theta$ that gets $\theta _ { \mathrm { o } } \sim \Theta _ { D }$ as input and derives a new set of parameters (aka the unlearned model) $\theta _ { \mathrm { u } } \sim \Theta _ { D _ { \mathrm { F } } }$ , where $\Theta _ { D _ { \mathrm { F } } }$ is the distribution over the set of parameters when $\mathcal { F }$ is trained on $\mathcal { D } - \mathcal { D } _ { \mathrm { F } }$ rather than D.

# 2.3. Approximate Unlearning

Using Definition 2.2, it is clear that the most straightforward, exact unlearning method would be to retrain model F from scratch on $\mathcal { D } - \mathcal { D } _ { \mathrm { F } } ;$ ; this does not even use $\theta _ { \mathrm { o } } .$ However, training deep learning models is very costly, and retraining the models upon receiving each unlearning request would be impractical. Thus, approximate unlearning methods are designed to overcome these computational requirements by starting from $\theta _ { \mathrm { o } }$ and modifying the parameters to derive ${ \theta _ { \mathrm { o } } } ^ { \prime }$ s.t. ${ \pmb \theta } _ { \mathrm { o } } ^ { \prime } \overset {  } { = } { \pmb \theta } _ { \mathrm { u } }$ (i.e., from the same distribution).

In the rest of the paper, we refer to $\mathcal { D } _ { \mathrm { F } }$ as the forget or unlearning set interchangeably. Its complement, $\mathcal { D } _ { \mathrm { ~ R ~ } } =$ $\mathcal { D } - \mathcal { D } _ { \mathrm { F } }$ is the remain set. We will use the behavior of the models retrained from scratch on $\mathcal { D } _ { \mathrm { R } }$ as the goal of approximate unlearning methods, and will refer to them as $\mathcal { F } _ { \mathrm { R } }$ for brevity.

# 2.4. Unlearning Settings

Many of the prior methods on approximate unlearning for classification models require access to $\mathcal { D } _ { \mathrm { R } }$ . However, in practice, this assumption might be unrealistic. The access to $\mathcal { D } _ { \mathrm { R } }$ might be restricted, or might be against privacy regulations. Prior works do not make a clear distinction based on this requirement when comparing different approximate methods. Therefore, to make a clear and accurate comparison, we perform our experiments (see § 6) in two separate settings: one with access to both $\mathcal { D } _ { \mathrm { R } }$ and $\mathcal { D } _ { \mathrm { F } }$ , and the other with access to only $\mathcal { D } _ { \mathrm { F } }$ . We report the results for each setting separately. For comparison with prior methods, we adapt them to both settings whenever possible.

# 3. Motivation

We present the intuition for our proposed unlearning method in § 3.1, and in § 3.2 we describe our observation about finetuning a model on its adversarial examples.

# 3.1. A Guiding Observation

Before designing a new unlearning method, we would like to first characterize the changes we expect to see after a successful unlearning. Because the retrained models are the gold standard of unlearning methods, we first assess their behavior on $\mathcal { D } _ { \mathrm { R } } , \mathcal { D } _ { \mathrm { F } }$ , and $\mathcal { D } _ { \mathrm { T } }$ . To this end, we evaluate the confidence values of $\mathcal { F } _ { \mathrm { R } }$ when predicting labels of $\mathcal { D } _ { \mathrm { R } }$ , $\mathcal { D } _ { \mathrm { F } }$ , and $\mathcal { D } _ { \mathrm { T } }$ . Since samples in $\mathcal { D } _ { \mathrm { T } }$ are drawn from the same distribution as ${ \mathcal { D } } ,$ , we can conclude that samples in $\mathcal { D } _ { \mathrm { T } }$ and $\mathcal { D } _ { \mathrm { F } }$ are from the same distribution. Therefore, we expect $\mathcal { F } _ { \mathrm { R } }$ to have similar accuracy and prediction confidence scores on $\mathcal { D } _ { \mathrm { T } }$ (test set) and $\mathcal { D } _ { \mathrm { F } }$ .

Results: Figure 3 in Appendix F.1, shows the confidence scores (see § F.1 for details) for a ResNet-18 (He et al., 2016) model that has been retrained on $\mathcal { D } - \mathcal { D } _ { \mathrm { F } }$ , where $\mathcal { D }$ is the training set of CIFAR-10 (Alex, 2009) and the size of $\mathcal { D } _ { \mathrm { F } }$ (randomly chosen from D) is 10% and 50% of the size of D (the first and second sub-figures, respectively).

Key Observation 1: The main difference between the predictions on $\mathcal { D } _ { T }$ (unseen samples) and $\mathcal { D } _ { R }$ (observed samples) is that the model’s predictions are much more confident for the samples that it has observed compared to the unseen samples.

This basic observation has either been overlooked by the prior research on approximate machine unlearning or has been treated incorrectly. To make the unlearned models more similar to $\mathcal { F } _ { \mathrm { R } }$ , prior methods have focused on degrading the model’s performance on $\mathcal { D } _ { \mathrm { F } }$ directly by either (a) some variation of fine-tuning on $\mathcal { D } _ { \mathrm { R } }$ (Warnecke et al., 2021; Liu et al., 2024), (b) choosing wrong labels for samples in $\mathcal { D } _ { \mathrm { F } }$ and fine-tuning the model (Golatkar et al., 2020; Chen et al., 2023b; Fan et al., 2024), or (c) directly maximizing the loss with respect to the samples in $\mathcal { D } _ { \mathrm { F } }$ (Thudi et al., 2022). Using the wrong labels for the samples of $\mathcal { D } _ { \mathrm { F } }$ or maximizing the loss on them make these methods very unstable and prone to catastrophic forgetting (Zhang et al., 2024a) because these samples belong to the correct distribution of the data and we cannot force a model to perform wrongly on a portion of the dataset while preserving it’s test accuracy. For these methods, it is important to use a small enough learning rate along with early stopping to prevent compromising the model’s performance while seeking worse prediction confidence values on the samples in $\mathcal { D } _ { \mathrm { F } }$ . Also, most of these methods require access to the set of remaining samples to use it for preventing a total loss of the model’s performance (e.g., by continuing to optimize the model on $\mathcal { D } _ { \mathrm { R } } )$ (Golatkar et al., 2020; Liu et al., 2024).

# 3.2. Fine-tuning on Adversarial Examples

After training a model $\mathcal { F }$ on D, this model imposes a distribution f (x) (e.g., softmax outputs) for all possible labels $y \in \mathcal { V }$ given any $x \in \mathcal { X }$ . Since the model $\mathcal { F }$ is directly optimized on $\mathcal { D } , f ( x )$ becomes very skewed toward the correct class for samples in $\mathcal { D } .$ For a given sample from ${ \mathcal { D } } ,$ , its adversarial examples (see Definition 2.1) are very close in the input space to the original sample. However, $\mathcal { F }$ makes the wrong prediction on these examples. This wrong prediction is the direct result of the learned parameters $\theta _ { \mathrm { o } }$ for the classification model, and these adversarial examples, although predicted incorrectly, belong to the distribution imposed on $\mathcal { X }$ by these learned parameters $( \mathrm { i . e . }$ , even though that is not the correct distribution, that is what the model has learned).

Now, what happens if we insert one adversarial example $( x _ { a d v } , y _ { a d v } )$ that corresponds to the sample $( x , y )$ into $\mathcal { D }$ and make an augmented dataset $\mathcal { D } ^ { \prime }$ for fine-tuning? Even before fine-tuning starts, the model makes the correct prediction on that (adversarial) example (by predicting the wrong label $y _ { a d v } ! )$ , but its confidence might not be as high as the samples in ${ \mathcal { D } } ,$ on which the model has been trained on. Proceeding with fine-tuning of the model on the augmented dataset increases its confidence on $x _ { a d v }$ while making the same wrong prediction $y _ { a d v }$ . However, this fine-tuning does not change the model’s performance because the newly added sample $( x _ { a d v } , y _ { a d v } )$ does not contradict the distribution learned by the model. Since $( x , y ) \in \mathcal { D } ^ { \prime }$ , and x and $x _ { a d v }$ are very close to one another (e.g., very similar images) while having different labels, optimizing the model has to change its decision boundary in that region of the input space to reach small loss for both of these samples. As a result of this balance, the model tends to decrease its confidence on the original sample compared to the model that was solely trained on D because there was no opposing components for its optimization on D. Note that $\| x - x _ { a d v } \| \leq \epsilon ,$ where ϵ is often much smaller than the distance of any pairs of samples in D. This helps to localize this change in the decision boundary during fine-tuning, and prevent changes to models’ behavior in other regions of the input space (Liang et al., 2023). In the following we elaborate on our empirical observations that verify these changes.

![](images/4ff1807ff3117ebc987bb16bda7a26f4694dcc874a79b061b3d0294bf15956fa.jpg)

<details>
<summary>line</summary>

| Epochs | Test Accuracy (Line 1) | Test Accuracy (Line 2) | Test Accuracy (Line 3) | Test Accuracy (Line 4) | Test Accuracy (Line 5) |
| ------ | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| 0      | 100                    | 100                    | 100                    | 100                    | 100                    |
| 5      | 85                     | 75                     | 55                     | 45                     | 20                     |
| 10     | 90                     | 85                     | 55                     | 45                     | 45                     |
| 15     | 90                     | 85                     | 50                     | 45                     | 50                     |
| 20     | 90                     | 85                     | 50                     | 45                     | 50                     |
</details>

![](images/db9a1e4dc6eb7eae9b15308f6cc63953df98539257ef9d094eaf47e301aca63c.jpg)

<details>
<summary>line</summary>

| Epochs | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 0      | 90     | 90     | 90     | 90     | 90     |
| 5      | 85     | 15     | 10     | 15     | 5      |
| 10     | 85     | 15     | 10     | 15     | 5      |
| 15     | 85     | 15     | 10     | 15     | 5      |
| 20     | 85     | 20     | 10     | 15     | 5      |
</details>

![](images/7138a81d46776d1bb3a9c02123efa5aa7e0921c55b20004f265943e784f975b9.jpg)

<details>
<summary>line</summary>

| Epochs | Adv  | Orig | Adv-RS | Adv-RL | Orig-RL | Orig-AdvL |
| ------ | ---- | ---- | ------ | ------ | ------- | --------- |
| 0      | 90   | 90   | 90     | 90     | 90      | 90        |
| 5      | 80   | 85   | 20     | 5      | 5       | 5         |
| 10     | 80   | 85   | 20     | 5      | 5       | 5         |
| 15     | 80   | 85   | 20     | 5      | 5       | 5         |
| 20     | 80   | 85   | 20     | 5      | 5       | 5         |
</details>

Figure 1: Effect of fine-tuning on adversarial examples. This figure shows the effect of fine-tuning on test accuracy of a ResNet-18 model that is trained on CIFAR-10, when the dataset for fine-tuning changes (see § 6.2 for details). Let $\mathcal { D } _ { \mathrm { F } }$ contain 10% of the samples in D and $\mathcal { D } _ { \mathrm { A } }$ be the set of adversarial examples constructed using Algorithm 1. Adv, from the left sub-figure to right one, shows the results when $\mathcal { D } \cup \mathcal { D } _ { \mathrm { A } } , \mathcal { D } _ { \mathrm { F } } \cup \mathcal { D } _ { \mathrm { A } } ,$ and $\mathcal { D } _ { \mathrm { A } }$ is used for fine-tuning the model, respectively. $\mathtt { O r i g , A d v \mathrm { - } R S , A d v \mathrm { - } R L , O r i g \mathrm { - } R L }$ , and $\begin{array} { r l } { { } } & { { } \operatorname { O r i g - A d v I } } \end{array}$ shows the results when $\mathcal { D } _ { \mathrm { A } }$ for each of these sub-figures is replace by $\mathcal { D } _ { \mathrm { F } } , \mathcal { D } _ { \mathrm { A } R S } , \mathcal { D } _ { \mathrm { A } R L } , \mathcal { D } _ { R L }$ , and $\mathcal { D } _ { A d v L }$ , accordingly. As the figure shows, the specific use of adversarial examples with the mis-predicted labels matters in keeping the model’s test accuracy because $\mathcal { D } _ { \mathrm { { A } } } ,$ , in contrast to the other constructed datasets belong to the natural distribution learned by the trained model.

Setup: We consider the training set of CIFAR-10 as D and choose D to be a random subset whose size is 10% of |D|. We also compute a set of adversarial examples (using Algorithm 1) corresponding to $\mathcal { D } _ { \mathrm { F } }$ , which we call $\mathcal { D } _ { \mathrm { A } }$ . Fig. 1 shows the fine-tuning of a trained ResNet-18 model for 20 epochs. Similar experiment for VGG19 models trained on the Tiny Imagenet dataset (Le & Yang, 2015) can be found in In the leftmost sub-figure, the curve presented as ${ \tt O r i g }$ represents the test accuracy of the model when it is finetuned on D. The curve named Adv is fine-tuned on $\mathcal { D } \cup \mathcal { D } _ { \mathrm { { A } } }$ , which has a similar test accuracy to ${ \tt O r i g }$ .

In the second sub-figure, Orig shows the test accuracy of the model when it is fine-tuned on $\mathcal { D } _ { \mathrm { F } }$ (two copies of $\mathcal { D } _ { \mathrm { F } }$ to keep the sample count similar), while Adv represents fine-tuning on $\mathcal { D } _ { \mathrm { F } } \cup \mathcal { D } _ { \mathrm { A } }$ . As the figure shows, Adv has a small degradation in test accuracy compared to Orig.

The rightmost sub-figure shows the case where ${ \tt O r i g }$ is finetuning of the model on $\mathcal { D } _ { \mathrm { F } }$ , and Adv is fine-tuning on only ${ \mathcal { D } } _ { \mathrm { A } } .$ . Although the degradation in test accuracy increases for this case, surprisingly we see that the model still remains noticeably accurate despite being fine-tuned on a set of samples that are all mislabeled. See § 6.2.1 for more details.

Results: As Figure 1 (and Figure 6 and 7 in Appendix F.3) shows, the test accuracy of the model does not deteriorate, even when it is being fine-tuned on only $\mathcal { D } _ { \mathrm { A } }$ (the dataset with wrong labels). See § 6.2.1 for further details.

Key Observation 2: Fine-tuning a model on the adversarial examples does not lead to catastrophic forgetting!

# 4. Adversarial Machine UNlearning (AMUN)

We utilize our novel observation about the effect of finetuning on adversarial examples (see § 3.2) to achieve the intuition we had about the retrained models (see § 3.1). We utilize the existing flaws of the trained model in learning the correct distribution, that appear as adversarial examples in the vicinity of the samples in $\mathcal { D } _ { \mathrm { F } }$ , to decrease its confidence on those samples while maintaining the performance.

Formally, AMUN uses Algorithm 1 to find an adversarial example for any sample in $( x , y ) \in { \mathcal { D } } _ { \mathrm { F } }$ . This algorithm uses a given untargeted adversarial algorithm $\boldsymbol { \mathcal { A } } _ { \mathcal { F } }$ , that finds the solution to Definition 2.1, for finding an adversarial example $x _ { a d v }$ . To make sure ϵ is as small as possible, Algorithm 1 starts with a small ϵ and runs the attack $\boldsymbol { A } _ { \mathcal { F } } ;$ if an adversarial algorithm is not found within that radius, it runs $A _ { \mathcal { F } }$ with a larger ϵ. It continues to perform $\boldsymbol { \mathcal { A } } _ { \mathcal { F } }$ with incrementally increased ϵ values until it finds an adversarial example; it then adds it to $\mathcal { D } _ { \mathrm { A } }$ . The algorithm stops once it finds adversarial examples for all the samples in $\mathcal { D } _ { \mathrm { F } }$ .

The reason behind minimizing the distance of ϵ for each sample is to localize the changes to the decision boundary of the model as much as possible; this prevents changing the model’s behavior on other parts of the input space. For our experiments, we use PGD-50 (Madry, 2017) with $\ell _ { 2 }$ norm bound as $\boldsymbol { \mathcal { A } } _ { \mathcal { F } }$ . We set the step size of the gradient ascent in the attack to $0 . 1 \times \epsilon ,$ which changes with the ϵ value. More details regarding the implementations of AMUN and prior unlearning methods and tuning their hyper-parameters can be found in Appendix C. Also, in Appendix F.5, we will show how using weaker attacks, such as Fast Gradient Sign Method (FGSM) (Goodfellow et al., 2014), might lead to lower performance of AMUN.

Algorithm 1 Build Adversarial Set $( \mathcal { F } , \mathcal { A } , \mathcal { D } _ { \mathrm { F } } , \epsilon _ { i n i t } )$   
1: Input: Model F, Attack algorithm A, Forget set $D_{F}$ , and Initial $\epsilon$ for adversarial attack
2: Output: $D_{A}$ : Adversarial set for $D_{F}$ 3: $D_{A} = \{\}$ 4: for $(x, y)$ in $D_{F}$ do
5: $\epsilon = \epsilon_{init}$ 6: while TRUE do
7: $x_{adv} = \mathcal{A}(x, \epsilon)$ 8: $y_{adv} = \mathcal{F}(x_{adv})$ 9: if $y_{adv} != y$ then
10: Break
11: end if
12: $\epsilon = 2\epsilon$ 13: end while
14: Add $(x_{adv}, y_{adv})$ to $D_{A}$ 15: end for
16: Return $D_{A}$

Once Algorithm 1 constructs $\mathcal { D } _ { \mathrm { { A } } } .$ , AMUN utilizes that to augment the dataset on which it performs the fine-tuning. If $\mathcal { D } _ { \mathrm { R } }$ is available, AMUN fine-tunes the model on $\mathcal { D } _ { \mathbb { R } } \cup$ $\mathcal { D } _ { \mathrm { F } } \cup \mathcal { D } _ { \mathrm { A } }$ and when $\mathcal { D } _ { \mathrm { R } }$ is not accessible, it performs the fine-tuning on $\mathcal { D } _ { \mathrm { F } } \cup \mathcal { D } _ { \mathrm { A } }$ . Also, in the setting where the size of the $\mathcal { D } _ { \mathrm { F } }$ is very large, we noticed some improvement when using only $\mathcal { D } _ { \mathbb { R } } \cup \mathcal { D } _ { \mathrm { A } }$ and $\mathcal { D } _ { \mathrm { A } }$ , for those settings, respectively.

# 4.1. Influencing Factors

We also derive an upper-bound on the 2-norm of the difference of the parameters of the unlearned model and the retrained model (which are gold-standard for unlearning) that illuminated the influencing factors in the effectiveness of AMUN. To prove this theorem, we make assumptions that are common in the certified unlearning literature. The proof is given in Appendix A.

Theorem 4.1. Let $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = \{ 1 , . . . , N \} }$ be a dataset of N samples and without loss of generality let $( x _ { n } , y _ { n } )$ (henceforth represented as $( x , y )$ for brevity) be the sample that needs to be forgotten and $( x _ { a d v } , y _ { a d v } )$ be its corresponding adversarial example used by AMUN such that $\| x - x _ { a d v } \| _ { 2 } = \delta .$ . Let $\hat { \mathcal { R } } ( w )$ represent the (unnormalized) empirical loss on $\mathcal { D } ^ { \prime } = \mathcal { D } \cup \{ ( x _ { a d v } , y _ { a d v } ) \}$ for a model $f$ that is parameterized with w. We assume that $f$ is $L \mathrm { - }$ Lipschitz with respect to the inputs and Rˆ is β-smooth and convex with respect to the parameters. Let $\theta _ { o }$ represent the parameters corresponding to the model originally trained on $\mathcal { D }$ and $\theta _ { u }$ be the parameters derived when the model is trained on $\mathcal { D } - \{ ( x , y ) \}$ }. We also assume that both the original and retrained models achieve near-0 loss on their training sets. After AMUN performs fine-tuning on $\mathcal { D } ^ { \prime }$ using one step of gradient descent with a learning rate of $\frac { 1 } { \beta }$ to derive parameters $\theta ^ { \prime } ,$ we get the following upper-bound for the distance of the unlearned model and the model retrained on $\mathcal { D } - \{ ( x , y ) \}$ (gold standard of unlearning):

$$
\| \theta^ {\prime} - \theta_ {u} \| _ {2} ^ {2} \leq \| \theta_ {o} - \theta_ {u} \| _ {2} ^ {2} + \frac {2}{\beta} (L \delta - C),
$$

$$
\begin{array}{l} \text {where C = \ell(f_{\theta_ {o}}(x_{adv}),y) + \ell(f_{\theta^{\prime}}(x_{adv}),y_{adv}) - } \\ \ell (f _ {\theta_ {u}} (x), y) - \ell (f _ {\theta_ {u}} (x _ {a d v}), y _ {a d v}). \end{array}
$$

According to the bound in Theorem 4.1, a lower Lipschitz constant of the model (L) and adversarial examples that are closer to the original samples (lower value for δ) lead to a smaller upper bound. A larger value of C also leads to a improved upper-bound. In the following we investigate the factors that lead to a larger value for $C ,$ which further clarifies some of influencing factors in the effectiveness of AMUN:

• Higher quality of adversarial example in increasing the loss for the correct label on the original model, which leads to larger value for $\ell ( f _ { \theta _ { o } } ( x _ { a d v } ) , y )$ .   
• Transferability of the adversarial example generated on the original model to the retrained model to decrease its loss for the wrong label, which leads to a lower value for $\ell ( f _ { \theta _ { u } } ( x _ { a d v } ) , y _ { a d v } )$ . This also aligns with lower Lipschitz constant of the model, as shown by prior work (Ebrahimpour-Boroojeny et al.).   
• Early stopping and using appropriate learning rate during fine-tuning phase of unlearning to avoid overfitting to the adversarial example, which does not allow low values for $\ell ( f _ { \theta ^ { \prime } } ( x _ { a d v } ) , y _ { a d v } )$ .

• The generalization of the retrained model to the unseen samples, which leads to a lower value for $\ell ( f _ { \theta _ { u } } ( x ) , y )$ .

Note that the first two implications rely on the strength of the adversarial example in addition to being close to the original sample. The second bullet, which relies on the transferability of adversarial examples, has been shown to improve as the Lipschitz constant decreases (Ebrahimpour-Boroojeny et al., 2024). The third bullet point is a natural implication which also holds for other unlearning methods that rely on the fine-tuning of the model. The fourth bullet point is not relevant to the unlearning method and instead relies on the fact that the retrained model should have good generalizability to unseen samples; it implies that as the size of $\mathcal { D } _ { \mathrm { F } }$ increases (i.e., $| \mathcal { D } _ { \mathrm { R } } |$ decreases) and the performance of the retrained model decreases, the effectiveness of the unlearning model also decreases. This is also intuitively expected in the unlearning process. Hence, the proved theorem also justifies our earlier intuitions about the need for good quality adversarial examples that are as close as possible to the original samples (which is the goal of Algorithm 1).

# 5. Evaluation Setup

In this section we elaborate on the details of evaluating different unlearning methods. More details $( \mathrm { e . g . }$ , choosing the hyper-parameters) can be found in Appendix C.

# 5.1. Baseline Methods

We compare AMUN with FT (Warnecke et al., 2021), RL (Golatkar et al., 2020), GA (Thudi et al., 2022), BS (Chen et al., 2023b), l1-Sparse (Liu et al., 2024), and SalUn (Fan et al., 2024). We also combine the weight saliency idea for masking the model parameters to limit the changes to the parameters during fine-tuning with AMUN and present its results as $\mathbf { A M U N _ { + S a l U n } }$ (see Appendix D for more details). We use the same hyper-parameter tuning reported by prior works. For further details, see Appendix C.

# 5.2. Evaluation Metrics

The metic used by recent works in unlearning to evaluate the unlearning methods (Liu et al., 2024; Fan et al., 2024) considers the models retrained on $\mathcal { D } _ { \mathrm { R } }$ as the goal standard for comparison. They compute the following four values for both the retrained models and the models unlearned using approximate methods:

• Unlearn Accuracy: Their accuracy on $\mathcal { D } _ { \mathrm { F } }$ .   
• Retain Accuracy: Their accuracy on $\mathcal { D } _ { \mathrm { R } }$ .   
• Test Accuracy: Their accuracy on $\mathcal { D } _ { \mathrm { T } }$ .   
• MIA score: Scores returned by membership inference attacks on $\mathcal { D } _ { \mathrm { F } }$

Once these four values are computed, the absolute value of the difference of each of them with the corresponding value for $\mathcal { F } _ { R }$ (the retrained models) is computed. Finally, the average of the four differences (called the Average Gap) is used as the metric to compare the unlearning methods.

The MIAs used in the recent unlearning methods by Liu et al. (2024); Fan et al. (2024) are based on the methods introduced by Yeom et al. (2018); Song et al. (2019). Although these MIAs have been useful for basic comparisons, recent SOTA MIAs significantly outperform their earlier counterparts, albeit with an increase in complexity and computation cost. To perform a comprehensive comparison of the effectiveness of the unlearning methods, we utilized a SOTA MIA called RMIA (Zarifzadeh et al., 2024), in addition to using the MIAs from prior works. In RMIA, the area under the ROC curve (AUC) of the MIA scores for predicting the training samples from the unseen samples is reported. Recall that in machine unlearning, the samples are split to three sets: $\mathcal { D } _ { \mathrm { R } } , \mathcal { D } _ { \mathrm { F } } .$ , and $\mathcal { D } _ { \mathrm { T } }$ . For an unlearning method to be effective, as discussed in § 3.1, we expect the AUC of RMIA for distinguishing the samples in $\mathcal { D } _ { \mathrm { F } }$ from the ones in $\mathcal { D } _ { \mathrm { T } }$ to be the same as random guessing (50% assuming balanced data). As shown in Table 1, this expectation holds for the models retrained on $\mathcal { D } _ { \mathrm { R } }$ .

We report the results of our comparisons for both the MIAs from prior unlearning literature and the new SOTA MIA. We will present the former one as MIS , and the latter one as FT AUC (the AUC of predicting $\mathcal { D } _ { \mathrm { F } }$ from $\mathcal { D } _ { \mathrm { T } } )$ .

# 5.3. Unlearning Settings

Another important factor missing in the comparisons of the unlearning methods in prior works is the possibility of access to $\mathcal { D } _ { \mathrm { R } }$ . So, for our experiments we consider two settings, one with access to $\mathcal { D } _ { \mathrm { R } }$ and one with access to only $\mathcal { D } _ { \mathrm { F } }$ . We adapt each of the unlearning methods to both of these settings, and perform the comparisons in each of these settings separately. The prior unlearning methods that do not adapt to the setting where there is no access to $\mathcal { D } _ { \mathrm { R } }$ (Warnecke et al., 2021; Liu et al., 2024) are excluded for the presented results in that setting.

Therefore, we perform different sets of experiments to evaluate the unlearning methods in both settings, and hope this becomes the norm in future works in machine unlearning. In each of these two settings, we evaluate unlearning of 10% or 50% of the samples randomly chosen from D. For all the experiments we train three models on D. For each size of ${ \mathcal { D } } ,$ we use three random subsets and for each subset, we try three different runs of the unlearning methods. This leads to a total of 27 runs of each unlearning method using different initial models and subsets of D to unlearn.

# 6. Experiments

We wish to answer the following questions:

1. Does AMUN lead to effective unlearning of any random subset of the samples when evaluated by a SOTA MIA?   
2. Does the choice of $\mathcal { D } _ { \mathrm { A } }$ matter in AMUN, or can it be replaced with a dataset that contains different labels or different samples that are within the same distance to the corresponding samples in $\mathcal { D } _ { \mathrm { { F } } } ?$   
3. Is AMUN effective on adversarially robust models?   
4. Does the choice of attack method matter in Algorithm 1 used by AMUN and does transferred attack work as well?   
5. How does AMUN compare to other unlearning methods when used for performing multiple unlearning requests on the same model?

As a quick summary, our results show that: (1) AMUN effectively leads to unlearning the samples in $\mathcal { D } _ { \mathrm { { F } } } \mathbf { ; }$ after unlearning 10% of the samples of CIFAR-10 from a trained ResNet-18, RMIA cannot do better than random guessing (§ 6.1); (2) If we replace $\mathcal { D } _ { \mathrm { A } }$ with any of the aforementioned substitutes, the model’s accuracy significantly deteriorates, especially when there is no access to $\mathcal { D } _ { \mathrm { ~ R ~ } } ( \ S 6 . 2 . 1 )$ ; (3) AMUN is as effective for unlearning on models that are adversarially robust (§ 6.2.2); (4) using weaker attack methods, such as FGSM, in AMUN hurts the effectiveness by not finding the adversarial examples that are very close to the samples in $\mathcal { D } _ { \mathrm { F } }$ . However, they still outperform prior methods (Appendix F.5). The transferred adversarial examples are effective as well (Appendix F.4); and (5) AMUN outperforms other unlearning methods when handling multiple unlearning requests (§ 6.3).

# 6.1. Effectiveness of AMUN

In this subsection we report the results on the comparisons of AMUN to other unlearning methods (see § 5.1). We consider the unlearning settings discussed in § 5.3, and the evaluation metrics discussed in § 5.2. We use ResNet-18 models trained on CIFAR-10 and VGG19 models trained on Tiny Imagenet for this experiment. We also perform an analysis on the computation costs of AMUN (see section E.1 for the details).

Results: Table 1 shows the results of evaluation using RMIA when the unlearning methods have access to $\mathcal { D } _ { R } .$ Table 2 shows these results when there is no access to $\mathcal { D } _ { R }$ . As the results show, AMUN clearly outperforms prior unlearning methods in all settings. This becomes even more clear when there is no access to $\mathcal { D } _ { \mathrm { R } }$ . Note that, for the models retrained on $\mathcal { D } _ { \mathrm { R } }$ , the AUC score of RMIA for predicting $\mathcal { D } _ { \mathrm { R } }$ from $\mathcal { D } _ { \mathrm { T } }$ (which can be considered as the worst case for FT AUC score) are 64.17 and 69.05 for unlearning 10% and 50% accordingly. Similar results for unlearning in VGG19 models trained on Tiny Imagenet when unlearning 10% of $\mathcal { D }$ can be found in section E.

![](images/e8dc3c6a98d60968a8d72be8f3265ea334ca2030bcfbdbb5d056f9cb31941931.jpg)

<details>
<summary>line</summary>

| Request counts | AUC Gap (Line 1) | AUC Gap (Line 2) | AUC Gap (Line 3) | AUC Gap (Line 4) | AUC Gap (Line 5) |
| -------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 1              | 28               | 15               | 10               | 0                | -10              |
| 2              | 22               | 10               | 8                | 0                | -10              |
| 3              | 18               | 8                | 6                | 0                | -10              |
| 4              | 16               | 7                | 5                | 0                | -10              |
| 5              | 14               | 6                | 4                | 0                | -10              |
</details>

![](images/af4a7c4dbd74bc17def7a64325136148c0d05227e418ba477e7a41b5cf127c09.jpg)

<details>
<summary>line</summary>

| Request counts | Amun  | Amun-A | SalUn | RL    | GA    | BS    |
| -------------- | ----- | ------ | ----- | ----- | ----- | ----- |
| 1              | 10.0  | 10.0   | 10.0  | 10.0  | 10.0  | 10.0  |
| 2              | 7.0   | 7.0    | 7.0   | 7.0   | 7.0   | 7.0   |
| 3              | 4.0   | 4.0    | 4.0   | 4.0   | 4.0   | 4.0   |
| 4              | 2.0   | 2.0    | 2.0   | 2.0   | 2.0   | 2.0   |
| 5              | 1.0   | 1.0    | 1.0   | 1.0   | 1.0   | 1.0   |
</details>

Figure 2: Multiple unlearning requests. This figure shows the evaluation of unlearning methods when they are used for unlearning for five times and each time on 2% of the training data. We train a ResNet-18 model on CIFAR-10 when $\mathcal { D } _ { \mathrm { R } }$ is available (left) and when it is not (right). After each step of the unlearning, we use the MIA scores generated by RMIA to derive the area under the ROC curve (AUC) for $\mathcal { D } _ { \mathrm { R } }$ vs. $\mathcal { D } _ { \mathrm { F } }$ and $\mathcal { D } _ { \mathrm { F } }$ vs. $\mathcal { D } _ { \mathrm { T } }$ . The values on the y-axis shows the difference of these two AUC scores. A high value for this gap means the samples in $\mathcal { D } _ { \mathrm { F } }$ are far more similar to $\mathcal { D } _ { \mathrm { T } }$ rather than $\mathcal { D } _ { \mathrm { R } }$ and shows a more effective unlearning.

We also present the results when MIS is used as the evaluation metric in Tables 10 and 11 in Appendix G, which similarly shows AMUN’s dominance in different unlearning settings. Moreover, we evaluate the combination of AMUN and SalUn (see Appendix D for details) and present its results as $\mathbf { A M U N } _ { S a l U n }$ in these tables. $\mathbf { A M U N } _ { S a l U n }$ slightly improves the results of AMUN in the setting where there is no access to ${ \mathcal { D } } _ { \mathrm { R } } ,$ , by filtering the parameters that are more relevant to $\mathcal { D } _ { \mathrm { F } }$ during fine-tuning.

# 6.2. Ablation Studies

In this subsection, we first elaborate on the effect of finetuning a model on its adversarial examples and compare it to the cases where either the samples or labels of this dataset change (§ 6.2.1). We then discuss AMUN’s efficacy on models that are already robust to adversarial examples (§ 6.2.2). We present other ablation studies on using weaker, but faster, adversarial attacks in Algorithm 1 (Appendix F.5). In Appendix F.4, we utilize transferred adversarial examples for unlearning, as this can expedite handling the unlearning from a newly trained model for which adversarial examples on similar architectures are available.

# 6.2.1. FINE-TUNING ON ADVERSARIAL EXAMPLES

We want to verify the importance of $\mathcal { D } _ { \mathrm { A } }$ (created by Algorithm 1) in preserving the model’s test accuracy. To this end, we build multiple other sets to be used instead of $\mathcal { D } _ { \mathrm { A } }$ when fine-tuning. Let us assume that $\mathcal { A } _ { \mathcal { F } } ( x , y ) = ( x _ { a d v } , y _ { a d v } )$ . Then, these other sets are:

<table><tr><td rowspan="2"></td><td colspan="6">RANDOM FORGET (10%)</td><td colspan="5">RANDOM FORGET (50%)</td></tr><tr><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td><td></td></tr><tr><td>RETRAIN</td><td> $94.49 \pm 0.20$ </td><td> $100.0 \pm 0.00$ </td><td> $94.33 \pm 0.18$ </td><td> $50.00 \pm 0.42$ </td><td>0.00</td><td> $92.09 \pm 0.37$ </td><td> $100.0 \pm 0.00$ </td><td> $91.85 \pm 0.33$ </td><td> $50.01 \pm 0.12$ </td><td>0.00</td><td></td></tr><tr><td>FT</td><td> $95.16 \pm 0.29$ </td><td> $96.64 \pm 0.25$ </td><td> $92.21 \pm 0.27$ </td><td> $52.08 \pm 0.34$ </td><td> $2.06 \pm 0.10$ </td><td> $94.24 \pm 0.30$ </td><td> $95.82 \pm 0.31$ </td><td> $91.21 \pm 0.33$ </td><td> $51.74 \pm 0.36$ </td><td> $2.17 \pm 0.13$ </td><td></td></tr><tr><td>RL</td><td> $95.54 \pm 0.14$ </td><td> $97.47 \pm 0.08$ </td><td> $92.17 \pm 0.10$ </td><td> $51.33 \pm 0.63$ </td><td> $1.74 \pm 0.18$ </td><td> $94.83 \pm 0.44$ </td><td> $99.79 \pm 0.04$ </td><td> $90.08 \pm 0.16$ </td><td> $50.78 \pm 0.14$ </td><td> $1.38 \pm 0.09$ </td><td></td></tr><tr><td>GA</td><td> $98.94 \pm 1.39$ </td><td> $99.22 \pm 1.31$ </td><td> $93.39 \pm 1.18$ </td><td> $60.96 \pm 2.93$ </td><td> $4.28 \pm 0.47$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.65 \pm 0.07$ </td><td> $63.39 \pm 0.26$ </td><td> $4.62 \pm 0.05$ </td><td></td></tr><tr><td>BS</td><td> $99.14 \pm 0.31$ </td><td> $99.89 \pm 0.06$ </td><td> $93.04 \pm 0.14$ </td><td> $57.85 \pm 1.12$ </td><td> $3.48 \pm 0.32$ </td><td> $55.24 \pm 5.11$ </td><td> $55.67 \pm 4.90$ </td><td> $50.16 \pm 5.28$ </td><td> $55.19 \pm 0.42$ </td><td> $32.01 \pm 3.86$ </td><td></td></tr><tr><td> $l_{1}$ -SPARSE</td><td> $94.29 \pm 0.34$ </td><td> $95.63 \pm 0.16$ </td><td> $91.55 \pm 0.17$ </td><td> $51.21 \pm 0.32$ </td><td> $2.16 \pm 0.06$ </td><td> $98.00 \pm 0.17$ </td><td> $98.71 \pm 0.13$ </td><td> $92.79 \pm 0.10$ </td><td> $54.44 \pm 0.47$ </td><td> $2.67 \pm 0.11$ </td><td></td></tr><tr><td>SALUN</td><td> $96.25 \pm 0.21$ </td><td> $98.14 \pm 0.16$ </td><td> $93.06 \pm 0.18$ </td><td> $50.88 \pm 0.54$ </td><td> $1.44 \pm 0.12$ </td><td> $96.68 \pm 0.35$ </td><td> $99.89 \pm 0.01$ </td><td> $91.97 \pm 0.18$ </td><td> $50.86 \pm 0.18$ </td><td> $1.36 \pm 0.04$ </td><td></td></tr><tr><td>AMUN</td><td> $95.45 \pm 0.19$ </td><td> $99.57 \pm 0.00$ </td><td> $93.45 \pm 0.22$ </td><td> $50.18 \pm 0.36$ </td><td> $\mathbf{0.62} \pm 0.05$ </td><td> $93.50 \pm 0.09$ </td><td> $99.71 \pm 0.01$ </td><td> $92.39 \pm 0.04$ </td><td> $49.99 \pm 0.18$ </td><td> $\mathbf{0.33} \pm 0.03$ </td><td></td></tr><tr><td> $AMUN_{+SalUn}$ </td><td> $95.02 \pm 0.18$ </td><td> $99.58 \pm 0.04$ </td><td> $93.29 \pm 0.04$ </td><td> $50.72 \pm 0.79$ </td><td> $\underline{0.68} \pm 0.18$ </td><td> $93.56 \pm 0.07$ </td><td> $99.72 \pm 0.02$ </td><td> $92.52 \pm 0.20$ </td><td> $49.81 \pm 0.40$ </td><td> $\underline{0.36} \pm 0.07$ </td><td></td></tr></table>

Table 1: Unlearning with access to $\mathcal { D } _ { \mathbf { R } } .$ . Comparing different unlearning methods in unlearning 10% and 50% of D. Avg. Gap (see § 5.2) is used for evaluation (lower is better). The lowest value is shown in bold while the second best is specified with underscore. As the results show, AMUN outperforms all other methods by achieving lowest Avg. Gap and $\mathbf { A M U N } _ { S a l U n }$ achieves comparable results. 

<table><tr><td rowspan="2"></td><td colspan="5">RANDOM FORGET (10%)</td><td colspan="5">RANDOM FORGET (50%)</td></tr><tr><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td></tr><tr><td>RETRAIN</td><td> $94.49 \pm 0.20$ </td><td> $100.0 \pm 0.00$ </td><td> $94.33 \pm 0.18$ </td><td> $50.00 \pm 0.42$ </td><td>0.00</td><td> $92.09 \pm 0.37$ </td><td> $100.0 \pm 0.00$ </td><td> $91.85 \pm 0.33$ </td><td> $50.01 \pm 0.12$ </td><td>0.00</td></tr><tr><td>RL</td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.45 \pm 0.09$ </td><td> $61.85 \pm 0.25$ </td><td> $4.31 \pm 0.06$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.57 \pm 0.14$ </td><td> $61.99 \pm 0.10$ </td><td> $4.29 \pm 0.03$ </td></tr><tr><td>GA</td><td> $4.77 \pm 3.20$ </td><td> $5.07 \pm 3.54$ </td><td> $5.09 \pm 3.38$ </td><td> $49.78 \pm 0.34$ </td><td> $68.53 \pm 2.45$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $92.65 \pm 0.09$ </td><td> $63.41 \pm 0.24$ </td><td> $5.13 \pm 0.04$ </td></tr><tr><td>BS</td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.48 \pm 0.04$ </td><td> $61.41 \pm 0.29$ </td><td> $4.20 \pm 0.07$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.58 \pm 0.08$ </td><td> $62.43 \pm 0.14$ </td><td> $4.40 \pm 0.05$ </td></tr><tr><td>SALUN</td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.47 \pm 0.10$ </td><td> $61.09 \pm 0.40$ </td><td> $4.11 \pm 0.09$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.59 \pm 0.12$ </td><td> $62.45 \pm 0.37$ </td><td> $4.40 \pm 0.07$ </td></tr><tr><td>AMUN</td><td> $94.28 \pm 0.37$ </td><td> $97.47 \pm 0.10$ </td><td> $91.67 \pm 0.04$ </td><td> $52.24 \pm 0.23$ </td><td> $1.94 \pm 0.13$ </td><td> $92.77 \pm 0.52$ </td><td> $95.66 \pm 0.25$ </td><td> $89.43 \pm 0.19$ </td><td> $52.60 \pm 0.22$ </td><td> $2.51 \pm 0.09$ </td></tr><tr><td>AMUN+SalUn</td><td> $94.19 \pm 0.38$ </td><td> $97.71 \pm 0.06$ </td><td> $91.79 \pm 0.12$ </td><td> $51.93 \pm 0.12$ </td><td> $1.77 \pm 0.06$ </td><td> $91.90 \pm 0.63$ </td><td> $96.59 \pm 0.31$ </td><td> $89.98 \pm 0.44$ </td><td> $52.32 \pm 0.56$ </td><td> $2.00 \pm 0.17$ </td></tr></table>

Table 2: Unlearning with access to only ${ \mathcal { D } } _ { \mathbf { F } } .$ . Comparing different unlearning methods in unlearning 10% and 50% of D. Avg. Gap (see § 5.2) is used for evaluation (lower is better) when only $\mathcal { D } _ { \mathrm { F } }$ is available during unlearning. As the results show, $\mathbf { A M U N } _ { S a l U n }$ significantly outperforms all other methods, and AMUN achieves comparable results.

• $\mathcal { D } _ { A d v L } \colon \{ ( x , y _ { a d v } ) \} _ { \forall ( x , y ) \in \mathcal { D } _ { \mathrm { F } } }$   
$\bullet \mathcal { D } _ { R L } : \{ ( x , y ^ { \prime } ) , \mathrm { s . t . } y ^ { \prime } \neq y , y _ { a d v } \} _ { \forall ( x , y ) \in \mathcal { D } _ { \mathrm { F } } }$   
• $\mathcal { D } _ { \mathrm { A } R L } \colon \{ ( x _ { a d v } , y ^ { \prime } ) , \mathrm { s . t . } y ^ { \prime } \neq y , y _ { a d v } \} _ { \forall ( x , y ) \in \mathcal { D } _ { \mathrm { F } } }$   
• $\mathcal { D } _ { \mathrm { A } R S } \colon \{ ( x ^ { \prime } , y _ { a d v } ) , \mathrm { ~ s . t . ~ } x ^ { \prime } \sim \mathrm { U n i f o r m } ( \mathrm { X } _ { \delta } )$ , where Xδ = {∀xˆ : ∥xδ − x∥2 = δ}}∀(x,y)∈DF

In this experiment, we evaluate the effect of fine-tuning on test accuracy of a ResNet-18 model that is trained on CIFAR-10, when $\mathcal { D } _ { \mathrm { A } }$ is substituted with other datasets that vary in the choice of samples or their labels. We assume that $\mathcal { D } _ { \mathrm { F } }$ contains 10% of the samples in D and $\mathcal { D } _ { \mathrm { A } }$ is the set of corresponding adversarial examples constructed using Algorithm 1.

Results: In Fig. 1, Adv, from the left sub-figure to the right sub-figure, shows the results when $\mathcal { D } \cup \mathcal { D } _ { \mathrm { A } } , \mathcal { D } _ { \mathrm { F } } \cup \mathcal { D } _ { \mathrm { A } }$ , and $\mathcal { D } _ { \mathrm { A } }$ is used for fine-tuning the model, respectively. ${ \tt O r i g }$ , Adv-RS, Adv-RL, Orig-RL, and Orig-AdvL show the results when $\mathcal { D } _ { \mathrm { A } }$ for each of these sub-figures is replaced by $\mathcal { D } _ { \mathrm { F } } , \mathcal { D } _ { \mathrm { A R S } } , \mathcal { D } _ { \mathrm { A R L } } , \mathcal { D } _ { R L }$ , and $\mathcal { D } _ { A d v L }$ , respectively. As the figure shows, the specific use of adversarial examples with the mispredicted labels matters in keeping the model’s test accuracy, especially as we move from the leftmost subfigure (having access to $\mathcal { D } _ { \mathtt { R } } )$ to the right one (only using $\mathcal { D } _ { \mathrm { A } }$ or its substitutes). This is due to the fact that the samples in $\mathcal { D } _ { \mathrm { A } }$ , in contrast to the other constructed datasets, belong to the natural distribution learned by the trained model. Therefore, even if we only fine-tune the ResNet-18 model on $\mathcal { D } _ { \mathrm { A } }$ , we still do not lose much in terms of model’s accuracy on $\mathcal { D } _ { \mathrm { T } }$ . This is a surprising observation, as $\mathcal { D } _ { \mathrm { A } }$ contains a set of samples with wrong predictions! Fig. 6 in Appendix F.3 shows similar results when size of $\mathcal { D } _ { \mathrm { F } }$ is 50% of |D|.

# 6.2.2. ADVERSARIALLY ROBUST MODELS

We evaluate the effectiveness of AMUN when the trained model is adversarially robust. One of the most effective methods in designing robust models is adversarial training which targets smoothing the model’s prediction function around the training samples (Salman et al., 2019). This has been shown to provably enhance the adversarial robustness of the model (Cohen et al., 2019). One of the effective adversarial training methods is by using TRADES loss introduced by (Zhang et al., 2019). We will use adversarially trained ResNet-18 models for unlearning 10% of the samples in CIFAR-10. In addition, we will use another defense mechanism that is less costly and more practical for larger models.

There is a separate line of work that try to achieve the same smoothness in model’s prediction boundary by controlling the Lipschitz constant of the models (Szegedy, 2013). The method proposed by Boroojeny et al. (2024) is much faster than adversarial training and their results show a significant improvement in the robust accuracy. We use their clipping method to evaluate the effectiveness of AMUN for unlearning 10% and 50% of the samples from robust ResNet-18 models trained on CIFAR-10.

Results: Table 7 in Appendix F.2 shows the results for the adversarially trained models. For the models with controlled Lipschitz continuity, the results are shows in Table 3 (no access to $\mathcal { D } _ { \mathtt { R } } )$ and Table 6 in Appendix F.2 (with access to $\mathcal { D } _ { \mathrm { R } } )$ . As the results show, even when there is no access to $\mathcal { D } _ { \mathrm { R } }$ , AMUN still results in effective unlearning for adversarially robust models; RMIA does not do better than random guessing for predicting $\mathcal { D } _ { \mathrm { F } }$ from $\mathcal { D } _ { \mathrm { T } } .$ . As Fig. 3 (right) shows, in the robust models, more than 97% of the adversarial examples are further away from their corresponding training samples, compared to this distance for the original models. However, this does not interfere with the performance of AMUN because these robust models are smoother and tend to be more regularized. This regularization, which prevents them from overfitting to the training samples is in fact a contributing factor to the improved generalization bounds for these models (Bartlett et al., 2017). This in itself contributes to enhanced resilience against MIAs. As seen in Tables 3 and 6, even for the clipped models retrained on $\mathcal { D } _ { \mathrm { R } }$ , the AUC score of RMIA for predicting $\mathcal { D } _ { \mathrm { R } }$ from $\mathcal { D } _ { \mathrm { F } }$ (FR AUC) is very low, which shows that these smoother models .

<table><tr><td rowspan="2"></td><td colspan="3">RANDOM FORGET (10%)</td><td colspan="3">RANDOM FORGET (50%)</td></tr><tr><td>FT AUC</td><td>FR AUC</td><td>TEST ACC</td><td>FT AUC</td><td>FR AUC</td><td>TEST ACC</td></tr><tr><td>RETRAIN</td><td> $49.95 \pm 0.24$ </td><td> $54.08 \pm 0.16$ </td><td> $89.01 \pm 0.21$ </td><td> $50.19 \pm 0.15$ </td><td> $55.61 \pm 0.05$ </td><td> $85.76 \pm 0.41$ </td></tr><tr><td>AMUN</td><td> $49.55 \pm 0.13$ </td><td> $54.01 \pm 0.23$ </td><td> $87.55 \pm 0.44$ </td><td> $49.64 \pm 0.31$ </td><td> $53.23 \pm 0.21$ </td><td> $87.39 \pm 0.61$ </td></tr></table>

Table 3: Unlearning on adversarially robust models. Evaluating the effectiveness of AMUN in unlearning 10% and 50% of the training samples when the models are adversarially robust and there is no access to $\mathcal { D } _ { \mathrm { R } }$ . For this experiment we use models with controlled Lipschitz constant which makes them provably and empirically more robust to adversarial examples.

# 6.3. Continuous Unlearning

We evaluate the performance of the unlearning methods when they are used to perform multiple consecutive unlearning from a trained model. This is a desirable capability for unlearning methods because in real-world applications there might be multiple unlearning requests and it is preferred to minimize the number of times that a model needs to be retrained from scratch. The setting we envision is as follows: models are updated at each request for unlearning. For AMUN, this means that $\mathcal { D } _ { \mathrm { A } }$ is computed on an updated model after each set of unlearning requests (shown as AMUN-A). In addition to comparing AMUN-A to the other unlearning methods, we also compare it to a version (shown as AMUN) that computes all the adversarial examples on the original model so it can handle the unlearning requests faster upon receiving them i.e., $\mathcal { D } _ { \mathrm { A } }$ is not computed on an updated model after each request; the set of requests are batched and $\mathcal { D } _ { \mathrm { A } }$ is computed on the entire batch. For this experiment, we use a ResNet-18 model trained on training set of CIFAR-10 (50K samples). Our goal is to unlearn 10% of the training samples (5K), but this time in 5 consecutive sets of size 2% (1K) each. We then evaluate the effectiveness of unlearning at each step using RMIA.

Results: Fig. 2 shows an overview of the results for both settings of unlearning (with or without access to $\mathcal { D } _ { \mathtt { R } } )$ . This figure shows the effectiveness of unlearning by depicting how the samples in $\mathcal { D } _ { \mathrm { F } }$ are more similar to the test samples $( \mathcal { D } _ { \mathrm { T } } )$ rather than the remaining samples $( \mathcal { D } _ { \mathtt { R } } )$ . The value on the y-axis shows the difference of the area under the ROC curve (AUC) for predicting $\mathcal { D } _ { \mathrm { R } }$ from $\mathcal { D } _ { \mathrm { F } }$ and $\mathcal { D } _ { \mathrm { F } }$ from $\mathcal { D } _ { \mathrm { T } } .$ . For the plots of each of these values separately, see Appendix H. AMUN-A performs better than all the other unlearning methods for all the steps of unlearning. Although AMUN also outperforms all the prior unlearning methods, it slightly under-performs compared to AMUN-A. This is expected, as the model’s decision boundary slightly changes after each unlearning request and the adversarial examples generated for the original model might not be as effective as those ones generated for the new model. Note that for this experiment, we did not perform hyper-parameter tuning for any of the unlearning methods, and used the same ones derived for unlearning 10% of the dataset presented in § 6.1. For further discussion of the results see Appendix H.

# 7. Conclusions

AMUN utilizes our new observation on how fine-tuning the trained models on adversarial examples that correspond to a subset of the training data does not lead to significant deterioration of model’s accuracy. Instead, it decreases the prediction confidence values on the the corresponding training samples. By evaluating AMUN using SOTA MIAs, we show that it outperforms other existing method, especially when unlearning methods do not have access to the remaining samples. It also performs well for handling multiple unlearning requests. This work also raises some questions for future work: (1) Since SOTA MIA methods fail to detect the unlearned samples, can this method be used to provide privacy guarantees for all the training samples?; (2) Can the same ideas be extended to generative models or Large Language Models?; (3) Can we derive theoretical bounds on the utility loss due to fine-tuning on adversarial examples?

# Acknowledgments

This work used Delta computing resources at National Center for Supercomputing Applications through allocation CIS240316 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program (Boerner et al., 2023), which is supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296. This paper was generously supported by NSF award IIS-2312561.

# Impact Statement

This research advances machine unlearning, a critical capability for privacy compliance in AI systems. Traditional exact unlearning methods, such as retraining, are computationally expensive, while approximate methods struggle to maintain model accuracy and confidence. AMUN introduces a novel approach that efficiently lowers model confidence on forget samples by leveraging adversarial examples, ensuring targeted changes to the decision boundary without significantly altering overall model behavior. This breakthrough improves privacy protection by making membership inference attacks ineffective while maintaining test accuracy, setting a new standard for efficient, privacypreserving unlearning in deep learning. The work paves the way for scalable and effective unlearning solutions, addressing a fundamental challenge in AI regulation and ethical machine learning.

# References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pp. 308–318, 2016.   
Alex, K. Learning multiple layers of features from tiny images. https://www. cs. toronto. edu/kriz/learning-features-2009-TR. pdf, 2009.   
Bartlett, P. L., Foster, D. J., and Telgarsky, M. J. Spectrallynormalized margin bounds for neural networks. Advances in neural information processing systems, 30, 2017.   
Boerner, T. J., Deems, S., Furlani, T. R., Knuth, S. L., and Towns, J. Access: Advancing innovation: Nsf’s advanced cyberinfrastructure coordination ecosystem: Services & support. In Practice and Experience in Advanced Research Computing 2023: Computing for the Common Good, PEARC ’23, pp. 173–176, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450399852. doi: 10.1145/ 3569951.3597559. URL https://doi.org/10. 1145/3569951.3597559.

Boroojeny, A. E., Telgarsky, M., and Sundaram, H. Spectrum extraction and clipping for implicitly linear layers. In International Conference on Artificial Intelligence and Statistics, pp. 2971–2979. PMLR, 2024.

Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., Jia, H., Travers, A., Zhang, B., Lie, D., and Papernot, N. Machine unlearning. In 2021 IEEE Symposium on Security and Privacy (SP), pp. 141–159. IEEE, 2021.

Cao, Y. and Yang, J. Towards making systems forget with machine unlearning. In 2015 IEEE symposium on security and privacy, pp. 463–480. IEEE, 2015.

Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., and Tramer, F. Membership inference attacks from first principles. In 2022 IEEE Symposium on Security and Privacy (SP), pp. 1897–1914. IEEE, 2022.

Chen, C., Sun, F., Zhang, M., and Ding, B. Recommendation unlearning. In Proceedings of the ACM Web Conference 2022, pp. 2768–2777, 2022a.

Chen, H., Zhang, Y., Dong, Y., Yang, X., Su, H., and Zhu, J. Rethinking model ensemble in transfer-based adversarial attacks. arXiv preprint arXiv:2303.09105, 2023a.

Chen, M., Zhang, Z., Wang, T., Backes, M., Humbert, M., and Zhang, Y. Graph unlearning. In Proceedings of the 2022 ACM SIGSAC conference on computer and communications security, pp. 499–513, 2022b.

Chen, M., Gao, W., Liu, G., Peng, K., and Wang, C. Boundary unlearning: Rapid forgetting of deep networks via shifting the decision boundary. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7766–7775, 2023b.

Chien, E., Pan, C., and Milenkovic, O. Efficient model updates for approximate unlearning of graph-structured data. In The Eleventh International Conference on Learning Representations, 2022.

Cohen, J., Rosenfeld, E., and Kolter, Z. Certified adversarial robustness via randomized smoothing. In international conference on machine learning, pp. 1310–1320. PMLR, 2019.

Di, Z., Yu, S., Vorobeychik, Y., and Liu, Y. Adversarial machine unlearning. arXiv preprint arXiv:2406.07687, 2024.

Dwork, C. Differential privacy. In International colloquium on automata, languages, and programming, pp. 1–12. Springer, 2006.

Ebrahimpour-Boroojeny, A., Sundaram, H., and Chandrasekaran, V. Training robust ensembles requires rethinking lipschitz continuity. In The Thirteenth International Conference on Learning Representations.

Ebrahimpour-Boroojeny, A., Sundaram, H., and Chandrasekaran, V. Lotos: Layer-wise orthogonalization for training robust ensembles. arXiv preprint arXiv:2410.05136, 2024.   
Fan, C., Liu, J., Zhang, Y., Wei, D., Wong, E., and Liu, S. Salun: Empowering machine unlearning via gradientbased weight saliency in both image classification and generation. In International Conference on Learning Representations, 2024.   
Ginart, A., Guan, M., Valiant, G., and Zou, J. Y. Making ai forget you: Data deletion in machine learning. Advances in neural information processing systems, 32, 2019.   
Golatkar, A., Achille, A., and Soatto, S. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9304–9312, 2020.   
Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.   
Guo, C., Goldstein, T., Hannun, A., and Van Der Maaten, L. Certified data removal from machine learning models. arXiv preprint arXiv:1911.03030, 2019.   
Gupta, V., Jung, C., Neel, S., Roth, A., Sharifi-Malvajerdi, S., and Waites, C. Adaptive machine unlearning. Advances in Neural Information Processing Systems, 34: 16319–16330, 2021.   
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.   
Hu, H., Salcic, Z., Sun, L., Dobbie, G., Yu, P. S., and Zhang, X. Membership inference attacks on machine learning: A survey. ACM Computing Surveys (CSUR), 54(11s):1–37, 2022.   
Izzo, Z., Smart, M. A., Chaudhuri, K., and Zou, J. Approximate data deletion from machine learning models. In International Conference on Artificial Intelligence and Statistics, pp. 2008–2016. PMLR, 2021.   
Jung, Y., Cho, I., Hsu, S.-H., and Hockenmaier, J. Attack and reset for unlearning: Exploiting adversarial noise toward machine unlearning through parameter reinitialization. arXiv preprint arXiv:2401.08998, 2024.   
Le, Y. and Yang, X. Tiny imagenet visual recognition challenge. CS 231N, 7(7):3, 2015.   
Liang, X., Qian, Y., Huang, J., Ling, X., Wang, B., Wu, C., and Swaileh, W. Towards desirable decision boundary by

moderate-margin adversarial training. Pattern Recognition Letters, 173:30–37, 2023.   
Liu, J., Ram, P., Yao, Y., Liu, G., Liu, Y., SHARMA, P., Liu, S., et al. Model sparsity can simplify machine unlearning. Advances in Neural Information Processing Systems, 36, 2024.   
Liu, Y., Chen, X., Liu, C., and Song, D. Delving into transferable adversarial examples and black-box attacks. arXiv preprint arXiv:1611.02770, 2016.   
Łucki, J., Wei, B., Huang, Y., Henderson, P., Tramèr, F., and Rando, J. An adversarial perspective on machine unlearning for ai safety. arXiv preprint arXiv:2409.18025, 2024.   
Madry, A. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.   
Neel, S., Roth, A., and Sharifi-Malvajerdi, S. Descent-todelete: Gradient-based methods for machine unlearning. In Algorithmic Learning Theory, pp. 931–962. PMLR, 2021.   
Papernot, N., McDaniel, P., and Goodfellow, I. Transferability in machine learning: from phenomena to blackbox attacks using adversarial samples. arXiv preprint arXiv:1605.07277, 2016.   
Salman, H., Li, J., Razenshteyn, I., Zhang, P., Zhang, H., Bubeck, S., and Yang, G. Provably robust deep learning via adversarially trained smoothed classifiers. Advances in neural information processing systems, 32, 2019.   
Sekhari, A., Acharya, J., Kamath, G., and Suresh, A. T. Remember what you want to forget: Algorithms for machine unlearning. Advances in Neural Information Processing Systems, 34:18075–18086, 2021.   
Setlur, A., Eysenbach, B., Smith, V., and Levine, S. Adversarial unlearning: Reducing confidence along adversarial directions. Advances in Neural Information Processing Systems, 35:18556–18570, 2022.   
Shokri, R., Stronati, M., Song, C., and Shmatikov, V. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP), pp. 3–18. IEEE, 2017.   
Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.   
Song, L., Shokri, R., and Mittal, P. Privacy risks of securing machine learning models against adversarial examples. In Proceedings of the 2019 ACM SIGSAC conference on computer and communications security, pp. 241–257, 2019.

Szegedy, C. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.   
Thudi, A., Deza, G., Chandrasekaran, V., and Papernot, N. Unrolling sgd: Understanding factors influencing machine unlearning. In 2022 IEEE 7th European Symposium on Security and Privacy (EuroS&P), pp. 303–319. IEEE, 2022.   
Ullah, E., Mai, T., Rao, A., Rossi, R. A., and Arora, R. Machine unlearning via algorithmic stability. In Conference on Learning Theory, pp. 4126–4142. PMLR, 2021.   
Vatter, J., Mayer, R., and Jacobsen, H.-A. The evolution of distributed systems for graph neural networks and their origin in graph processing and deep learning: A survey. ACM Computing Surveys, 56(1):1–37, 2023.   
Warnecke, A., Pirch, L., Wressnegger, C., and Rieck, K. Machine unlearning of features and labels. arXiv preprint arXiv:2108.11577, 2021.   
Wong, E., Rice, L., and Kolter, J. Z. Fast is better than free: Revisiting adversarial training. arXiv preprint arXiv:2001.03994, 2020.   
Yeom, S., Giacomelli, I., Fredrikson, M., and Jha, S. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st computer security foundations symposium (CSF), pp. 268–282. IEEE, 2018.   
Zarifzadeh, S., Liu, P., and Shokri, R. Low-cost high-power membership inference attacks. In Forty-first International Conference on Machine Learning, 2024.   
Zeng, Y., Chen, S., Park, W., Mao, Z. M., Jin, M., and Jia, R. Adversarial unlearning of backdoors via implicit hypergradient. arXiv preprint arXiv:2110.03735, 2021.   
Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., and Jordan, M. Theoretically principled trade-off between robustness and accuracy. In International conference on machine learning, pp. 7472–7482. PMLR, 2019.   
Zhang, J., Wu, W., Huang, J.-t., Huang, Y., Wang, W., Su, Y., and Lyu, M. R. Improving adversarial transferability via neuron attribution-based attacks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14993–15002, 2022.   
Zhang, R., Lin, L., Bai, Y., and Mei, S. Negative preference optimization: From catastrophic collapse to effective unlearning. arXiv preprint arXiv:2404.05868, 2024a.   
Zhang, Y., Chen, X., Jia, J., Zhang, Y., Fan, C., Liu, J., Hong, M., Ding, K., and Liu, S. Defensive unlearning with adversarial training for robust concept erasure in diffusion models. arXiv preprint arXiv:2405.15234, 2024b.

Zhao, Z., Liu, Z., and Larson, M. On success and simplicity: A second look at transferable targeted attacks. Advances in Neural Information Processing Systems, 34: 6115–6128, 2021.

# Appendix

# A. Proofs

Here we provide the proof of Theorem 4.1:

Proof. As we perform the unlearning by fine-tuning and performing a gradient descent update to $\theta _ { o } .$ , we have: $\theta ^ { \prime } =$ $\begin{array} { r } { \theta _ { o } - \frac { 1 } { \beta } \nabla \hat { \mathcal { R } } ( \theta _ { o } ) } \end{array}$ . Therefore, we can write:

$$
\begin{array}{l} \| \theta^ {\prime} - \theta_ {u} \| _ {2} ^ {2} = \| \theta_ {o} - \frac {1}{\beta} \nabla \hat {\mathcal {R}} (\theta_ {o}) - \theta_ {u} \| _ {2} ^ {2} \\ = \| \theta_ {o} - \theta_ {u} \| _ {2} ^ {2} - \frac {2}{\beta} \langle \nabla \hat {\mathcal {R}} (\theta_ {o}), \theta_ {o} - \theta_ {u} \rangle + \frac {1}{\beta^ {2}} \| \nabla \hat {\mathcal {R}} (\theta_ {o}) \| _ {2} ^ {2} \\ \leq \| \theta_ {o} - \theta_ {u} \| _ {2} ^ {2} + \frac {2}{\beta} (\hat {\mathcal {R}} (\theta_ {u}) - \hat {\mathcal {R}} (\theta_ {o})) + \frac {2}{\beta} (\hat {\mathcal {R}} (\theta_ {o}) - \hat {\mathcal {R}} (\theta^ {\prime})) \\ = \| \theta_ {o} - \theta_ {u} \| _ {2} ^ {2} + \frac {2}{\beta} (\hat {\mathcal {R}} (\theta_ {u}) - \hat {\mathcal {R}} (\theta^ {\prime})), \\ \end{array}
$$

where the inequality is derived by using the smoothness property $( \| \nabla \hat { \mathcal { R } } ( \theta _ { o } ) \| _ { 2 } ^ { 2 } \leq 2 \beta ( \hat { \mathcal { R } } ( \theta _ { o } ) - \hat { \mathcal { R } } ( \theta ^ { \prime } ) ) )$ ) and the convexity assumption which leads to the inequality: $\hat { \mathcal { R } } ( \theta _ { o } ) ) \geq \hat { \mathcal { R } } ( \theta _ { u } ) + \langle \nabla \hat { \mathcal { R } } ( \theta _ { o } ) , \theta _ { o } - \bar { \theta } _ { u } \rangle$ .

Next, we derive an upper-bound for $\hat { \mathcal { R } } ( \theta _ { u } ) - \hat { \mathcal { R } } ( \theta ^ { \prime } )$ to replace in the above inequality. By the definition of unnormalized empirical loss on $\mathcal { D } ^ { \prime } \colon$ :

$$
\begin{array}{l} \hat {\mathcal {R}} (\theta_ {u}) - \hat {\mathcal {R}} (\theta^ {\prime}) \\ = \sum_ {i = 1} ^ {n - 1} \ell (f _ {\theta_ {u}} (x _ {i}), y _ {i}) + \ell (f _ {\theta_ {u}} (x), y) + \ell (f _ {\theta_ {u}} (x ^ {\prime}), y ^ {\prime}) - \sum_ {i = 1} ^ {n - 1} \ell (f _ {\theta^ {\prime}} (x _ {i}), y _ {i}) - \ell (f _ {\theta^ {\prime}} (x), y) - \ell (f _ {\theta^ {\prime}} (x ^ {\prime}), y ^ {\prime}) \\ = \ell (f _ {\theta_ {u}} (x), y) + \ell (f _ {\theta_ {u}} (x ^ {\prime}), y ^ {\prime}) - \ell (f _ {\theta^ {\prime}} (x), y) - \ell (f _ {\theta^ {\prime}} (x ^ {\prime}), y ^ {\prime}), \\ \end{array}
$$

where the last equality was derived by the assumption that models are trained until they achieve near-0 loss on their corresponding dataset. Therefore, $\begin{array} { r } { \sum _ { i = 1 } ^ { \bar { n } - 1 } \ell \big ( f _ { \theta _ { u } } \big ( x _ { i } \big ) , y _ { i } \big ) = \sum _ { i = 1 } ^ { n - 1 } \ell \big ( f _ { \theta ^ { \prime } } ( x _ { i } ) , y _ { i } \big ) = 0 } \end{array}$ since the retrained model has been trained on the remaining samples and the unlearned model has been derived by a single step of gradient descent on the original model, that had been trained on D.

To further simplify the derived terms above and reaching at our desired inequality, we focus on the term $- \ell ( f _ { \theta ^ { \prime } } ( x ) , y )$ . By adding and decreasing the term $\ell ( f _ { \theta _ { o } } ( x ^ { \prime } ) , y )$ we get:

$$
\begin{array}{l} - \ell (f _ {\theta^ {\prime}} (x), y) = - \ell (f _ {\theta_ {o}} (x ^ {\prime}), y) + \ell (f _ {\theta_ {o}} (x ^ {\prime}), y) - \ell (f _ {\theta^ {\prime}} (x), y) \\ \leq - \ell (f _ {\theta_ {o}} (x ^ {\prime}), y) + \ell (f _ {\theta_ {o}} (x ^ {\prime}), y) - \ell (f _ {\theta_ {o}} (x), y) - \langle \nabla \ell (f _ {\theta_ {o}} (x), y), \theta^ {\prime} - \theta_ {o} \rangle \\ = - \ell (f _ {\theta_ {o}} (x ^ {\prime}), y) + \ell (f _ {\theta_ {o}} (x ^ {\prime}), y) - \ell (f _ {\theta_ {o}} (x), y) \\ \leq - \ell (f _ {\theta_ {o}} (x ^ {\prime}), y) + L \delta , \\ \end{array}
$$

where the first inequality uses the convexity of the the loss function with respect to the parameters and the third derivations is due to the assumption that the original model achieves a zero loss on its training samples, including $( x , y )$ (hence, $\nabla \ell ( f _ { \theta _ { o } } ( x ) , y ) = 0 )$ . The final inequality is due to the Lipschitzness assumption of model f with respect to the inputs.

□

# B. Related Works (cont.)

To the best of our knowledge AMUN is the first work that considers fine-tuning of a model on the adversarial examples with their wrong labels as a method for unlearning a subset of the samples. However, upon reviewing the prior works in unlearning literature, there are several works that their titles might suggest otherwise. Therefore, here we mention a few of these methods and how they differ from our work.

To improve upon fine-tuning on samples in $\mathcal { D } _ { \mathrm { F } }$ with randomly chosen wrong labels, Chen et al. (2023b) use the labels derived from one step of the FGSM attack to choose the new labels for the samples in $\mathcal { D } _ { \mathrm { F } }$ . This method which was presented as BS in our experiments (§ 6.1), does not use the adversarial examples and only uses their labels as the new labels for samples of $\mathcal { D } _ { \mathrm { F } }$ . This corresponds to the dataset $\mathcal { D } _ { A d v L }$ in § 6.2.1. As our results in Figures 1 and 6 show, fine-tuning the trained model on this dataset leads to catastrophic forgetting even when $\mathcal { D } _ { \mathrm { R } }$ is available. This is simply due to the fact that the samples in $\mathcal { D } _ { A d v L }$ contradict the distribution that the trained models have already learned.

The work by Setlur et al. (2022) is not an unlearning method, despite what the name suggest. They propose a regularization method that tries to maximize the loss on the adversarial examples of the training samples that are relatively at a higher distance to lower the confidence of the model on those examples. The work by Zhang et al. (2024b) proposed a defense method similar to adversarial training for making to unlearned LLMs more robust to jailbreak attacks on the topics that they have unlearned. Łucki et al. (2024) also study the careful application of jailbreak attacks against unlearned models. The work by Jung et al. (2024) investigate computing adversarial noise to mask the model parameters. Many of the works with similar titles, use “adversarial" to refer to minimax optimization (Zeng et al., 2021) or considering a Stackelberg game setting between the source model and the adversary that is trying to extract information (Di et al., 2024).

# C. Implementation Details

For all the experiments we train three models on D. For each size of D (10% or 50%), we use three random subsets and for each subset, we try three different runs of each of the unlearning methods. This leads to a total of 27 runs of each unlearning method using different initial models and subsets of D to unlearn. Hyper-parameter tuning of each of the methods is done on a separate random subset of the same size from ${ \mathcal { D } } ,$ and then the average performance is computed for the other random subsets used as $\mathcal { D } _ { \mathrm { F } }$ . For tuning the hyper-parameters of the models, we followed the same range suggested by their authors and what has been used in the prior works for comparisons. Similar to prior works (Liu et al., 2024; Fan et al., 2024), we performed 10 epochs for each of the unlearning methods, and searched for best learning rate and number of steps for a learning rate scheduler. More specifically, for each unlearning method, we performed a grid search on learning rates within the range of $[ 1 0 ^ { - 6 } , 1 0 ^ { - 1 } ]$ with an optional scheduler that scales the learning rate by 0.1 for every 1 or 5 steps. For SalUn, whether it is used on its own or in combination with AMUN, we searched for the masking ratios in the range [0.1, 0.9].

The original models are ResNet-18 models trained for 200 epochs with a learning rate initialized at 0.1 and using a scheduler that scales the learning rate by 0.1 every 40 epochs. The retrained models are trained using the same hyper-parameters as the original models. For evaluation using RMIA, we trained 128 separate models such that each sample is included in half of these models. As suggested by the authors, we used Soft-Margin Taylor expansion of Softmax (SM-Taylor-Softmax) with a temperature of 2 for deriving the confidence values in attacks of RMIA. We used the suggested threshold of 2 for comparing the ratios in computing the final scores (γ value). For controlling the Lipschitz constant of the ResNet-18 models in § 6.2.2, we used the default setting provided by the authors for clipping the spectral norm of all the convolutional and fully-connected layers of the model to 1. For RMIA evaluations, we trained 128 of these models separately such that each sample appears in exactly half of these models.

# D. AMUN + SalUn

The main idea behind SalUn is to limit the fine-tuning of the model, during unlearning, to only a subset of the parameters of the model, while keeping the rest of them fixed. Fan et al. (2024) show that this technique helps to preserve the accuracy of the model when fine-tuning the model on $\mathcal { D } _ { \mathrm { F } }$ with randomly-chosen wrong labels. More specifically, they compute a mask using the following equation:

$$
\mathbf {m} _ {\mathrm{S}} = \mathbf {1} \left(\left| \nabla_ {\boldsymbol {\theta} _ {\mathrm{o}}} \ell (\boldsymbol {\theta} _ {\mathrm{o}}; \mathcal {D} _ {\mathrm{F}}) \right| \right| \geq \gamma),
$$

which, basically, computes the gradient of the loss function for the current parameters with respect to $\mathcal { D } _ { \mathrm { F } }$ , and uses threshold γ to filter the ones that matter more to the samples in $\mathcal { D } _ { \mathrm { F } }$ . Note that, 1 is an element-wise indicator function. Then, during fine-tuning of the model on $\mathcal { D } _ { \mathrm { F } }$ with random labels they use $\mathbf { m } _ { \mathrm { S } }$ to detect the parameters of $\theta _ { \mathrm { o } }$ that get updated.

![](images/34a57693646c24fda2338a84ddc48d7f9966d09f336a11bde8807951806cc592.jpg)

<details>
<summary>bar_line</summary>

| Confidence | Remain | Test | Forget | Forget |
| ---------- | ------ | ---- | ------ | ------ |
| -15        | 0.0    | 0.0  | 0.0    | 0.0    |
| -10        | 0.0    | 0.0  | 0.0    | 0.0    |
| -5         | 0.0    | 0.0  | 0.0    | 0.0    |
| 0          | 0.0    | 0.0  | 0.0    | 0.0    |
| 5          | 0.1    | 0.1  | 0.1    | 0.1    |
| 10         | 0.3    | 0.2  | 0.2    | 0.2    |
| 15         | 0.0    | 0.0  | 0.0    | 0.0    |
</details>

![](images/7dc84e817a466aeee65963a0735f60c2f06633b2154e369e631f4c8120789d0c.jpg)

<details>
<summary>histogram</summary>

| Confidence Range | Remain | Test | Forget | Forget |
| ---------------- | ------ | ---- | ------ | ------ |
| -15 to -10       | 0.00   | 0.00 | 0.00   | 0.00   |
| -10 to -5        | 0.00   | 0.00 | 0.00   | 0.00   |
| -5 to 0          | 0.00   | 0.00 | 0.00   | 0.00   |
| 0 to 5           | 0.05   | 0.05 | 0.05   | 0.05   |
| 5 to 10          | 0.25   | 0.15 | 0.20   | 0.10   |
| 10 to 15         | 0.15   | 0.10 | 0.15   | 0.05   |
| 15 to 20         | 0.05   | 0.05 | 0.05   | 0.00   |
</details>

![](images/021d43da2b81daa2e743e0ac114797a97406c0d67ce2f448637495edf1f20904.jpg)

<details>
<summary>scatter</summary>

| δ_x Robust | δ_x Non-Robust |
| ---------- | -------------- |
| 0.0        | 0.0            |
| 0.5        | 0.2            |
| 1.0        | 0.4            |
| 1.5        | 0.6            |
| 2.0        | 0.8            |
</details>

Figure 3: (left) These two plots show the histogram of confidence values of the retrained model on its predictions for the remaining set (Remain), test set (Test), and forget set (Forget) during the training, when the size of the forget set is %10 (1st plot) and %50 (2nd plot) of the training set. It also shows the Gaussian distributions fitted to each histogram. As the plots show the models perform similarly on the forget set and test set because to the retrained model they are unseen data from the same distribution. (right) This plot compares the $\delta _ { x }$ value in definition 2.1 for adversarial examples generated on the original ResNet-18 models (x-axis) and clipped ResNet-18 models (y-axis). The dashed line shows $x = y$ line and more than 97% of the values fall bellow this line.

In our experiments, we try combining this idea with AMUN for updating a subset of the parameters that might be more relevant to the samples in $\mathcal { D } _ { \mathrm { F } }$ . We refer to this combination as $\mathbf { A M U N } _ { S a l U n }$ in Tables 1 and 2 in § 6.1 and Tables 10 and 11 in Appendix G. As the results show, $\mathbf { A M U N } _ { S a l U n }$ constantly outperforms SalUn and for the cases that $\mathcal { D } _ { \mathrm { R } }$ is not available it also outperforms AMUN. In the setting where $\mathcal { D } _ { \mathrm { R } }$ is accessible, it performs comparable to AMUN. This is probably due to the fact that when $\mathcal { D } _ { \mathrm { R } }$ is not available and AMUN has access to only the samples $\mathcal { D } _ { \mathrm { F } } \cup \mathcal { D } _ { \mathrm { A } }$ , SalUn acts as a regularization for not allowing all the parameters of the model that might not be relevant to $\mathcal { D } _ { \mathrm { F } }$ be updated. In the setting where $\mathcal { D } _ { \mathrm { R } }$ is available, involving it in fine-tuning will be a sufficient regularization that preserves models’ utility while unlearning $\mathcal { D } _ { \mathrm { F } }$ .

# E. Effectiveness of AMUN (cont.)

In this subsection we report the results on the comparisons of AMUN to other unlearning methods (see § 5.1) for unlearning 10% of the training samples for VGG19 models that are trained on Tiny Imagenet dataset. We consider the unlearning settings discussed in § 5.3, and the evaluation metrics discussed in § 5.2. Table 4 shows the results of evaluation using RMIA when the unlearning methods have access to $\mathcal { D } _ { R }$ . Table 5 shows these results when there is no access to ${ \mathcal { D } } _ { R } .$ . As the results show, AMUN clearly outperforms prior unlearning methods in all settings. This becomes even more clear when there is no access to $\mathcal { D } _ { \mathrm { R } }$ .

<table><tr><td></td><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td></tr><tr><td>RETRAIN</td><td> $55.93 \pm 0.21$ </td><td> $99.98 \pm 0.00$ </td><td> $55.96 \pm 0.74$ </td><td> $50.06 \pm 0.36$ </td><td> $0.00$ </td></tr><tr><td>FT</td><td> $74.99 \pm 1.13$ </td><td> $96.88 \pm 0.48$ </td><td> $56.29 \pm 0.47$ </td><td> $61.66 \pm 0.36$ </td><td> $8.36 \pm 0.37$ </td></tr><tr><td>RL</td><td> $59.66 \pm 1.77$ </td><td> $68.19 \pm 1.55$ </td><td> $50.78 \pm 0.95$ </td><td> $51.38 \pm 0.34$ </td><td> $10.50 \pm 0.07$ </td></tr><tr><td>GA</td><td> $0.46 \pm 0.01$ </td><td> $0.49 \pm 0.02$ </td><td> $0.50 \pm 0.02$ </td><td> $49.80 \pm 0.08$ </td><td> $52.67 \pm 0.03$ </td></tr><tr><td>BS</td><td> $0.50 \pm 0.11$ </td><td> $0.51 \pm 0.01$ </td><td> $0.51 \pm 0.02$ </td><td> $49.80 \pm 0.02$ </td><td> $52.65 \pm 0.04$ </td></tr><tr><td> $l_{1}$ -SPARSE</td><td> $55.27 \pm 1.41$ </td><td> $60.64 \pm 1.24$ </td><td> $49.82 \pm 0.72$ </td><td> $54.49 \pm 0.20$ </td><td> $12.80 \pm 0.52$ </td></tr><tr><td>SALUN</td><td> $66.54 \pm 4.61$ </td><td> $76.63 \pm 3.83$ </td><td> $49.56 \pm 1.19$ </td><td> $54.64 \pm 0.52$ </td><td> $11.24 \pm 0.14$ </td></tr><tr><td>AMUN</td><td> $62.57 \pm 0.62$ </td><td> $93.66 \pm 0.66$ </td><td> $55.52 \pm 0.67$ </td><td> $57.33 \pm 0.14$ </td><td> $\underline{5.17} \pm 0.13$ </td></tr><tr><td> $AMUN+SalUn$ </td><td> $62.96 \pm 0.92$ </td><td> $94.42 \pm 0.49$ </td><td> $55.80 \pm 0.55$ </td><td> $57.65 \pm 0.46$ </td><td> $\underline{5.09} \pm 0.29$ </td></tr></table>

Table 4: Unlearning with access to $\mathcal { D } _ { \mathbf { R } } .$ . Comparing different unlearning methods in unlearning 10% of Tiny Imagenet Dataset (D) from VGG19 models. Avg. Gap is used for evaluation (lower is better). The lowest value is shown in bold while the second best is specified with underscore. As the results show, AMUN outperforms all other methods by achieving lowest Avg. Gap and $\mathbf { A M U N } _ { S a l U n }$ achieves comparable results.

![](images/0ab8c0a6100f0d2b1d573b470a726ad0e644a5ed3d39ef6da16513e5ced9cbbc.jpg)

<details>
<summary>bar</summary>

| Smallest ε | Sample Count |
| ---------- | ------------ |
| 0.04       | 12000        |
| 0.08       | 1500         |
| 0.16       | 22000        |
| 0.32       | 12500        |
| 0.64       | 1500         |
| 1.28       | 500          |
</details>

![](images/e6c843ebb066b126900c513156cbcbbd4459bc1767c26366e8ebbc17723f2650.jpg)

<details>
<summary>bar</summary>

| Smallest ε | Sample Count |
| ---------- | ------------ |
| 0.16       | 45000        |
| 0.32       | 6000         |
| 0.64       | 44000        |
| 1.28       | 4000         |
| 2.56       | 500          |
| 5.12       | 100          |
| 20.48      | 50           |
</details>

![](images/6226a0b032eb593d5ef8060ad3621259d93dcf0869395d4f23a6195bb6c05218.jpg)

<details>
<summary>bar</summary>

| Smallest ε | Sample Count |
| ---------- | ------------ |
| 0.16       | 50000        |
| 0.32       | 4000         |
| 0.64       | 38000        |
| 1.28       | 5000         |
| 2.56       | 1000         |
| 5.12       | 500          |
</details>

Figure 4: Each plot shows the portion of samples for final values of ϵ in Algorithm 1. The smallest value on the x-axis in each plot shows the initial ϵ value chosen for Algorithm 1 based on running this algorithm on a small subset of the dataset. The left-most plot shows this distribution for CIFAR-10 when the chosen attack algorithm is PGD-50. The two right-most plots show the distributions for the Tiny ImageNet dataset when PGD-10 and PGD-20 are used, respectively. Once the initial value is set, a run of PGD attack is performed on all the samples. For the ones that adversarial example is not found within that radius, we perform other runs of PGD attack until adversarial examples are found for all the samples. As the histograms show, the total work is equivalent to less than 3 runs of PGD attack on the whole dataset, which is not computationally expensive.

# E.1. Computational Cost

One option to fast processing of the unlearning request is to run Algorithm 1 on all the training samples after the model is trained; this allows access to $\mathcal { D } _ { \mathrm { A } }$ for any arbitrary $\mathcal { D } _ { \mathrm { F } }$ provided. The alternative, is to run Algorithm 1 on $\mathcal { D } _ { \mathrm { F } }$ once the unlearning request is received. In either of these two cases, note that Algorithm 1 can be run in parallel on all the samples starting from the initial value for ϵ. Then for only the samples that the adversarial example is not found, we perform another iteration of attack with the updated value for ϵ. We continue this procedure until all the samples of interest have a corresponding adversarial example in $\mathcal { D } _ { \mathrm { A } }$ . Choosing a reasonable initial value for ϵ can save the computation time, by avoiding initial iterations on almost all the samples without any outcome. To choose the initial value of ϵ we run Algorithm 1 with a very small ϵ on a small subset of $\mathcal { D } _ { \mathrm { F } } \left( \mathrm { e . g } \right.$ ., 100 samples). Then we choose $\epsilon _ { i n i t }$ such that at least 95% of the samples find their adversarial examples within that distance. Using this strategy, we find that even when running Algorithm 1 on all D, the computation time will be equivalent to running the underlying adversarial attack (e.g., PGD-50) less than 3 times on all the samples in D. Figure 4 shows the histogram for the number of samples in D for each ϵ value for different models and datasets. The smallest value of ϵ in these plots is the $\epsilon _ { i n i t }$ chosen by the sampling procedure mentioned earlier. Note that a large portion of the samples find their corresponding adversarial examples within the first few iterations.

<table><tr><td></td><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td></tr><tr><td>RETRAIN</td><td> $55.93 \pm 0.21$ </td><td> $99.98 \pm 0.00$ </td><td> $55.96 \pm 0.74$ </td><td> $50.06 \pm 0.36$ </td><td> $0.00$ </td></tr><tr><td>RL</td><td> $1.36 \pm 0.21$ </td><td> $1.74 \pm 0.34$ </td><td> $1.27 \pm 0.18$ </td><td> $51.17 \pm 0.42$ </td><td> $52.15 \pm 0.07$ </td></tr><tr><td>GA</td><td> $0.44 \pm 0.05$ </td><td> $0.51 \pm 0.01$ </td><td> $0.50 \pm 0.00$ </td><td> $50.55 \pm 0.97$ </td><td> $52.81 \pm 0.14$ </td></tr><tr><td>BS</td><td> $0.93 \pm 0.24$ </td><td> $0.97 \pm 0.22$ </td><td> $1.02 \pm 0.27$ </td><td> $49.96 \pm 0.20$ </td><td> $52.28 \pm 0.19$ </td></tr><tr><td>SALUN</td><td> $2.46 \pm 1.51$ </td><td> $2.89 \pm 1.68$ </td><td> $2.13 \pm 0.98$ </td><td> $52.20 \pm 0.94$ </td><td> $51.63 \pm 0.82$ </td></tr><tr><td>AMUN</td><td> $58.54 \pm 1.56$ </td><td> $62.34 \pm 1.40$ </td><td> $43.22 \pm 0.68$ </td><td> $63.05 \pm 0.89$ </td><td> $\underline{16.50} \pm 0.13$ </td></tr><tr><td> $AMUN_{+SalUn}$ </td><td> $62.37 \pm 0.61$ </td><td> $68.47 \pm 0.70$ </td><td> $45.44 \pm 0.72$ </td><td> $61.24 \pm 0.63$ </td><td> $\mathbf{14.91} \pm 0.37$ </td></tr></table>

Table 5: Unlearning without access to $\mathcal { D } _ { \mathbf { R } }$ . Comparing different unlearning methods in unlearning $1 0 \%$ of Tiny Imagenet Dataset (D) from VGG19 models. Avg. Gap is used for evaluation (lower is better). The lowest value is shown in bold while the second best is specified with underscore. As the results show, AMUN outperforms all other methods by achieving lowest Avg. Gap and $\mathbf { A M U N } _ { S a l U n }$ achieves comparable results.

![](images/4a2e9baf06b368894a81c39ab4747ddbb7bcda301015dc0b1194b5a213cc692c.jpg)

<details>
<summary>area</summary>

| Confidence | Density (Green Area) | Density (Orange Area) |
| ---------- | -------------------- | --------------------- |
| -15        | 0.0                  | 0.0                   |
| -10        | 0.0                  | 0.0                   |
| -5         | 0.0                  | 0.0                   |
| 0          | 0.0                  | 0.0                   |
| 5          | 0.1                  | 0.1                   |
| 10         | 0.4                  | 0.2                   |
| 15         | 0.0                  | 0.0                   |
</details>

![](images/a27050fffffaa30013daa068c8c318551ea702815f97727998718b5bdd5643fc.jpg)

<details>
<summary>histogram</summary>

| Confidence Range | Remain | Test | Forget |
| ---------------- | ------ | ---- | ------ |
| -15 to -10       | 0.00   | 0.00 | 0.00   |
| -10 to -5        | 0.00   | 0.00 | 0.00   |
| -5 to 0          | 0.00   | 0.00 | 0.00   |
| 0 to 5           | 0.15   | 0.15 | 0.15   |
| 5 to 10          | 0.20   | 0.20 | 0.20   |
| 10 to 15         | 0.00   | 0.00 | 0.00   |
</details>

![](images/14f8979f53be62cc95956058eb69bda7cd0f26a8ce06a9d2aa640197ef7e9acd.jpg)

<details>
<summary>histogram</summary>

| Confidence Range | Frequency |
| ---------------- | --------- |
| -15 to -10       | 0.0       |
| -10 to -5        | 0.0       |
| -5 to 0          | 0.0       |
| 0 to 5           | 0.0       |
| 5 to 10          | 0.4       |
| 10 to 15         | 0.0       |
</details>

![](images/a75b627e154008459892e90462faff284d5d75cf65e5a49611699621c26a0788.jpg)

<details>
<summary>line</summary>

| Confidence | Remain | Test | Forget |
| ---------- | ------ | ---- | ------ |
| -10        | 0.00   | 0.00 | 0.00   |
| 0          | 0.05   | 0.05 | 0.05   |
| 5          | 0.15   | 0.15 | 0.15   |
| 10         | 0.20   | 0.20 | 0.20   |
| 15         | 0.10   | 0.10 | 0.10   |
| 20         | 0.05   | 0.05 | 0.05   |
</details>

Figure 5: The two left-most subplots show the confidence values before and after unlearning (using AMUN) of 10% of the training samples. The two right-most subplots show these confidence values for unlearning 50% of the training samples. In both cases, the confidence values of samples in $\mathcal { D } _ { \mathrm { F } }$ are similar to those of $\mathcal { D } _ { \mathrm { R } }$ and their fitted Gaussian distribution matches as expected. After using AMUN for unlearning the samples in $\mathcal { D } _ { \mathrm { F } }$ , the confidence values on this set gets more similar to the test (unseen) samples.

# F. Ablation Study (cont.)

In this section, we further discuss the ablation studies that were mentioned in § 6.2. We also present other ablation studies on using transferred adversarial examples (Appendix F.4) and weaker adversarial attacks (Appendix F.5) in AMUN.

# F.1. Empirical Behavior of Retrained Models

As discussed in § 3.1, assuming the $\mathcal { D } _ { \mathrm { T } }$ and D come from the same distributions, we expect the prediction confidences of the models retrained on $\mathcal { D } _ { \mathrm { R } }$ to be similar on DF and $\mathcal { D } _ { \mathrm { T } } .$ , because both of these sets are considered unseen samples that belong the the same data distribution. Figure 3 (left) shows the confidence scores for a ResNet-18 model that has been retrained on $\mathcal { D } - \mathcal { D } _ { \mathrm { F } }$ , where $\mathcal { D }$ is the training set of CIFAR-10 and the size of $\mathcal { D } _ { \mathrm { F } }$ (randomly chosen from D) is 10% and 50% of the size of D for the left and right sub-figures, respectively. To derive the confidence values, we use the following scaling on the logit values:

$$
\phi (f (x) _ {y}) = \log \left(\frac {f (x) _ {y}}{1 - f (x) _ {y}}\right),
$$

where $f ( x ) _ { y }$ is the predicted probability for the correct class. This scaling has been used by Carlini et al. (2022) to transform the the prediction probabilities such that they can be better approximated with a normal distribution, which are indeed used by some of the SOTA MIA methods for predicting training samples from the test samples (Carlini et al., 2022). Figure 3 (left) shows these fitted normal distribution as well, which perfectly match for $\mathcal { D } _ { \mathrm { F } }$ and $\mathcal { D } _ { \mathrm { T } }$ .

# F.1.1. CONFIDENCE VALUES IN UNLEARNED MODELS

In this section, we investigate the confidence values of the model, before and after using AMUN for unlearning a subset of 10% or 25% of the training samples. For the original model (before unlearning), we expect the distribution of confidence values of samples in $\mathcal { D } _ { \mathrm { F } }$ to be similar to those of the samples in $\mathcal { D } _ { \mathrm { R } }$ because they were both used as the training data and the model has used them similarly during training. However, this distribution is different for the test samples $( \mathcal { D } _ { \mathrm { T } } )$ , as the model has not seen them during the training phase. After unlearning, as discussed in section 3.1, we expect the distribution of confidence values for $\mathcal { D } _ { \mathrm { F } }$ and $\mathcal { D } _ { \mathrm { T } }$ to become more similar so that MIAs cannot distinguish them from each other. As Figure 5 shows, for both unlearning 10% (two leftmost subplots) and 50% (two rightmost subplots), we observe the same behavior. Fur the original models (1st and 3rd subplot), the distribution for $\mathcal { D } _ { \mathrm { F } }$ and $\mathcal { D } _ { \mathrm { R } }$ mathces exactly, but after using AMUN (2nd and 4th subplot) the distribution for $\mathcal { D } _ { \mathrm { F } }$ shifts toward that of $\mathcal { D } _ { \mathrm { T } }$ .

# F.2. Adversarially Robust Models (cont.)

As discussed in § 6.2.2, we also evaluatee the effectiveness of AMUN when the trained model is adversarially robust. For this experiment, we used the ResNet-18 models with 1-Lipschitz convolutional and fully-connected layers, which are shown to be significantly more robust than the original ResNet-18 models. In Table 3, we showed the results for unlearning 10% and 50% of the samples from the robust ResNet-18 models trained on CIFAR-10, in the case where $\mathcal { D } _ { \mathrm { R } }$ is not accessible. In Table 6, we showed the corresponding results when the unlearning methods have access to $\mathcal { D } _ { \mathrm { { R } } } .$ As the results show, similar to the results discussed in § 6.2.2, AMUN effectively unlearns $\mathcal { D } _ { \mathrm { F } }$ for either of the sizes of the this set.

<table><tr><td rowspan="2"></td><td colspan="3">RANDOM FORGET (10%)</td><td colspan="3">RANDOM FORGET (50%)</td></tr><tr><td>FT AUC</td><td>FR AUC</td><td>TEST ACC</td><td>FT AUC</td><td>FR AUC</td><td>TEST ACC</td></tr><tr><td>RETRAIN</td><td> $49.95 \pm 0.24$ </td><td> $54.08 \pm 0.16$ </td><td> $89.01 \pm 0.21$ </td><td> $50.19 \pm 0.15$ </td><td> $55.61 \pm 0.05$ </td><td> $85.76 \pm 0.41$ </td></tr><tr><td>AMUN</td><td> $49.12 \pm 0.19$ </td><td> $53.60 \pm 0.31$ </td><td> $86.94 \pm 0.56$ </td><td> $49.41 \pm 0.25$ </td><td> $54.22 \pm 0.16$ </td><td> $87.38 \pm 0.39$ </td></tr></table>

Table 6: Unlearning on adversarially robust models. Evaluating the effectiveness of AMUN in unlearning 10% and 50% of the training samples when the models are adversarially robust and we have access to $\mathcal { D } _ { \mathrm { R } } .$ . For this experiment we use models with controlled Lipschitz constant which makes them provably and empirically more robust to adversarial examples.

We also evaluated AMUN for unlearning in models that are adversarially trained. we performed our analysis on ResNet-18 models trained using TRADES loss (Zhang et al., 2019) on CIFAR-10. We performed the experiments for unlearning 10% of the dataset in both cases where $\mathcal { D } _ { \mathrm { R } }$ is accessible and not. As the results in Table 7 show, in both settings AMUN is effective in unlearning the forget samples and achieving a low gap with the retrained models. This gap is obviously smaller when there is access to $\mathcal { D } _ { \mathrm { R } }$ .

<table><tr><td></td><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td></tr><tr><td>RETRAIN</td><td> $82.33 \pm 0.39$ </td><td> $94.22 \pm 0.21$ </td><td> $81.72 \pm 0.36$ </td><td> $50.04 \pm 0.34$ </td><td> $0.00$ </td></tr><tr><td>AMUNWith  $\mathcal{D}_{\text{R}}$ </td><td> $82.65 \pm 0.62$ </td><td> $94.33 \pm 0.84$ </td><td> $84.99 \pm 0.91$ </td><td> $47.18 \pm 0.50$ </td><td> $1.02 \pm 0.18$ </td></tr><tr><td>AMUNNo  $\mathcal{D}_{\text{R}}$ </td><td> $81.38 \pm 0.10$ </td><td> $87.45 \pm 0.54$ </td><td> $79.74 \pm 0.31$ </td><td> $54.61 \pm 0.23$ </td><td> $3.57 \pm 0.24$ </td></tr></table>

Table 7: Unlearning with access to $\mathcal { D } _ { \mathbf { R } }$ . Evaluating AMUN when applied to ResNet-18 models trained using adversarial training. TRADES loss is used to train the models, and the unlearning is done on 10% of CIFAR-10 Dataset (D). Avg. Gap is used for evaluation (lower is better). The result has been reported in two cases: with and without access to $\mathcal { D } _ { \mathrm { R } }$ . As the results show, AMUN is effective in both cases, with slight degradation in the more difficult setting of no access to $\mathcal { D } _ { \mathrm { R } }$ .

# F.3. Fine-tuning on Adversarial Examples (cont.)

As explained in § 6.2.1, we evaluate the effect of fine-tuning on test accuracy of a ResNet-18 model that is trained on CIFAR-10, when $\mathcal { D } _ { \mathrm { A } }$ is substituted with other datasets that vary in the choice of samples or their labels (see § 6.2.1 for details). In Figure 1 we presented the results when $\mathcal { D } _ { \mathrm { F } }$ contains 10% of the samples in D. We also present the results for the case where $\mathcal { D } _ { \mathrm { F } }$ contains 50% of the samples in D in Figure 6. As the figure shows, even for the case where we fine-tune the trained models on only $\mathcal { D } _ { \mathrm { A } }$ which contains the adversarial examples corresponding to 50% of the samples in D (right-most sub-figure), there is no significant loss in models’ accuracy. This is due to the fact that the samples in $\mathcal { D } _ { \mathrm { A } }$ , in contrast to the other constructed datasets, belong to the natural distribution learned by the trained model. To generate the results in both Figures 1 and $^ { 6 , }$ we fine-tuned the trained ResNet-18 models on the all the datasets (see § 6.2.1 for details) for 20 epochs. We used a learning rate of 0.01 with a scheduler that scales the learning rate by 0.1 every 5 epochs.

We also perform the same experiment for VGG19 (Simonyan & Zisserman, 2014) models trained on Tiny Imagenet dataset (Le & Yang, 2015). We evaluate the effect of fine-tuning on test accuracy of these model that is, when $\mathcal { D } _ { \mathrm { F } }$ contains 10% of the samples in D and $\mathcal { D } _ { \mathrm { A } }$ is substituted with other datasets that vary in the choice of samples or their labels (see § 6.2.1 for details). In Figure 7 we presented the results, which similarly show that the specific use of adversarial examples with the mis-predicted labels matters in keeping the model’s test accuracy.

# F.4. Transferred Adversarial Examples

One of the intriguing properties of adversarial attacks is their transferability to other models (Papernot et al., 2016; Liu et al., 2016); Adversarial examples generated on a trained model (source model) mostly transfer successfully to other models (target models). This success rate of the transferred adversarial examples increases if the source model and target model have the same architecture (Papernot et al., 2016). There are other studies that can be used to increase the success rate of this type of attack (Zhao et al., 2021; Zhang et al., 2022; Chen et al., 2023a; Ebrahimpour-Boroojeny et al., 2024). In this section, we are interested to see if using the the adversarial examples generated using Algorithm 1 for a given model trained on some dataset D can be used as the $\mathcal { D } _ { \mathrm { A } }$ dataset for unlearning a portion of D from a separately trained model. The advantage of using adversarial examples generated for another model is saving the computation cost for other trained models. For this purpose, we train three ResNet-18 models separately on CIFAR-10, we generate the adversarial examples for each of these models using Algorithm 1. We use AMUN for unlearning 10% and 50% of CIFAR-10 from either of these models, but instead of their adversarial samples, we use the ones derived from the other models. The results in Table 8 shows that using transferred adversarial examples leads to lower performance, specially for the case where there is no access to $\mathcal { D } _ { \mathrm { { R } } } .$ . All the values for test accuracy are also lower compared to using adversarial examples from the model itself because these adversarial examples from the other models do not all belong to the natural distribution of the model and they do not even always transfer to the other models. Still the results are comparable to the prior SOTA methods in unlearning, and even in the case of no access to $\mathcal { D } _ { \mathrm { R } }$ outperforms all prior methods.

![](images/f5765c9921eebb6eb7b2edab1ec802b1dced15c2d8c0735ff3c4b3efb3667849.jpg)

<details>
<summary>line</summary>

| Epochs | Test Accuracy (Line 1) | Test Accuracy (Line 2) | Test Accuracy (Line 3) | Test Accuracy (Line 4) | Test Accuracy (Line 5) |
| ------ | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| 0      | 90                     | 90                     | 90                     | 90                     | 90                     |
| 5      | 85                     | 85                     | 35                     | 10                     | 5                      |
| 10     | 88                     | 88                     | 60                     | 12                     | 5                      |
| 15     | 90                     | 90                     | 65                     | 15                     | 5                      |
| 20     | 90                     | 90                     | 68                     | 15                     | 5                      |
</details>

![](images/9452d32c3c03f92fd30642f6ba06a0451ebd648ef1859a5e716677f1fd3ea8bf.jpg)

<details>
<summary>line</summary>

| Epochs | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 0      | 90     | 90     | 90     | 90     | 90     |
| 5      | 85     | 85     | 30     | 10     | 5      |
| 10     | 88     | 88     | 50     | 10     | 5      |
| 15     | 89     | 89     | 52     | 10     | 5      |
| 20     | 90     | 90     | 55     | 10     | 5      |
</details>

![](images/ea927634194de8f5a82dd6e4aa26b3a55c50a8b2f6ed5a60fc28e21e5e85f7fa.jpg)

<details>
<summary>line</summary>

| Epochs | Adv  | Orig | Adv-RS | Adv-RL | Orig-RL | Orig-AdvL |
| ------ | ---- | ---- | ------ | ------ | ------- | --------- |
| 0      | 90   | 90   | 90     | 90     | 90      | 90        |
| 5      | 85   | 90   | 10     | 5      | 5       | 5         |
| 10     | 85   | 90   | 10     | 5      | 5       | 5         |
| 15     | 85   | 90   | 10     | 5      | 5       | 5         |
| 20     | 85   | 90   | 10     | 5      | 5       | 5         |
</details>

Figure 6: This figure shows the effect of fine-tuning on test accuracy of a ResNet-18 model that is trained on CIFAR-10, when the dataset for fine-tuning changes (see § 6.2 for details). Let $\mathcal { D } _ { \mathrm { F } }$ contain 50% of the samples in D and $\mathcal { D } _ { \mathrm { A } }$ be the set of adversarial examples constructed using Algorithm 1. Adv, from the left sub-figure to right one, shows the results when $\mathcal { D } \cup \mathcal { D } _ { \mathrm { { A } } } , \mathcal { D } _ { \mathrm { { F } } } \cup \mathcal { D } _ { \mathrm { { A } } } ,$ , and $\mathcal { D } _ { \mathrm { A } }$ is used for fine-tuning the model, respectively. $\mathtt { O r i g } , \mathtt { A d v \mathrm { - } R S } , \mathtt { A d v \mathrm { - } R L } , \mathtt { O r i g \mathrm { - } R L }$ , and $\begin{array} { r l } { { } } & { { } \operatorname { O r i g - A d v I } } \end{array}$ shows the results when $\mathcal { D } _ { \mathrm { A } }$ for each of these sub-figures is replace by $\mathcal { D } _ { \mathrm { F } } , \mathcal { D } _ { \mathrm { A } R S } , \mathcal { D } _ { \mathrm { A } R L } , \mathcal { D } _ { R L }$ , and $\mathcal { D } _ { A d v L } .$ , accordingly. As the figure shows, the specific use of adversarial examples with the mis-predicted labels matters in keeping the model’s test accuracy because $\mathcal { D } _ { \mathrm { A } } ,$ , in contrast to the other constructed datasets belong to the natural distribution learned by the trained model.

![](images/123bc2cc4b87205b4974880ffbf72186cb18edf5bf5c444b7d9bad6cf7653e71.jpg)

<details>
<summary>line</summary>

| Epochs | Test Accuracy (Solid Blue) | Test Accuracy (Dashed Orange) | Test Accuracy (Dash-Dot Red) | Test Accuracy (Dotted Purple) | Test Accuracy (Dash-Dot Green) |
| ------ | -------------------------- | ----------------------------- | ---------------------------- | ----------------------------- | ------------------------------ |
| 0      | 55                         | 55                            | 55                           | 55                            | 55                             |
| 5      | 48                         | 50                            | 38                           | 40                            | 35                             |
| 10     | 52                         | 52                            | 48                           | 45                            | 32                             |
| 15     | 53                         | 53                            | 49                           | 47                            | 33                             |
| 20     | 53                         | 53                            | 49                           | 47                            | 32                             |
</details>

![](images/625356f10ba1d8943190c1b921e8a07137dffc7bfc32e6f0a81ff52a9fe9237d.jpg)

<details>
<summary>line</summary>

| Epochs | Blue Line | Orange Dashed Line | Red Dashed Line | Green Dash-Dot Line |
| ------ | --------- | ------------------ | --------------- | ------------------- |
| 0      | 55        | 55                 | 55              | 55                  |
| 5      | 40        | 10                 | 15              | 10                  |
| 10     | 50        | 10                 | 15              | 10                  |
| 15     | 50        | 10                 | 15              | 10                  |
| 20     | 50        | 10                 | 15              | 10                  |
</details>

![](images/325465866fb99170ca8d633367fac0c86dcab842c3e0cd9de12fdec4bd92157e.jpg)

<details>
<summary>line</summary>

| Epochs | Adv  | Orig | Adv-RS | Adv-RL | Orig-RL | Orig-AdvL |
| ------ | ---- | ---- | ------ | ------ | ------- | --------- |
| 0      | 50   | 50   | 50     | 50     | 50      | 50        |
| 5      | 45   | 50   | 0      | 0      | 0       | 0         |
| 10     | 45   | 50   | 0      | 0      | 0       | 0         |
| 15     | 45   | 50   | 0      | 0      | 0       | 0         |
| 20     | 45   | 50   | 0      | 0      | 0       | 0         |
</details>

Figure 7: Effect of fine-tuning on adversarial examples. This figure shows the effect of fine-tuning on test accuracy of a VGG19 model that is trained on the Tiny ImagenNet dataset, when the dataset for fine-tuning changes for details). Let $\mathcal { D } _ { \mathrm { F } }$ contain 10% of the samples in D and $\mathcal { D } _ { \mathrm { A } }$ be the set of adversarial examples constructed using Algorithm 1. Adv, from the left sub-figure to right one, shows the results when $\mathcal { D } \cup \mathcal { D } _ { \mathrm { A } } , \mathcal { D } _ { \mathrm { F } } \cup \mathcal { D } _ { \mathrm { A } }$ , and $\mathcal { D } _ { \mathrm { A } }$ is used for fine-tuning the model, respectively. $\mathtt { O r i g , A d v \mathrm { - } R S , A d v \mathrm { - } R L , O r i g \mathrm { - } R L }$ , and $\begin{array} { r l } { { } } & { { } \operatorname { O r i g - A d v I } } \end{array}$ shows the results when $\mathcal { D } _ { \mathrm { A } }$ for each of these sub-figures is replace by $\mathcal { D } _ { \mathrm { F } } , \mathcal { D } _ { \mathrm { A R S } } , \mathcal { D } _ { \mathrm { A R L } } , \mathcal { D } _ { R L }$ , and $\mathcal { D } _ { A d v L }$ , accordingly. As the figure shows, the specific use of adversarial examples with the mis-predicted labels matters in keeping the model’s test accuracy because $\mathcal { D } _ { \mathbf { A } } .$ , in contrast to the other constructed datasets belong to the natural distribution learned by the trained model.

<table><tr><td rowspan="3"></td><td colspan="6">WITH ACCESS TO  $\mathcal{D}_{\text{R}}$ </td><td colspan="6">No ACCESS TO  $\mathcal{D}_{\text{R}}$ </td></tr><tr><td colspan="3">RANDOM FORGET (10%)</td><td colspan="3">RANDOM FORGET (50%)</td><td colspan="3">RANDOM FORGET (10%)</td><td colspan="3">RANDOM FORGET (50%)</td></tr><tr><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td><td>TEST AUC</td><td>FT AUC</td><td>AVG. GAP</td><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td><td>TEST AUC</td><td>FT AUC</td><td>AVG. GAP</td></tr><tr><td>SELF</td><td> $93.45 \pm 0.22$ </td><td> $50.18 \pm 0.36$ </td><td> $0.62 \pm 0.05$ </td><td> $92.39 \pm 0.04$ </td><td> $49.99 \pm 0.18$ </td><td> $0.33 \pm 0.03$ </td><td> $91.67 \pm 0.04$ </td><td> $52.24 \pm 0.23$ </td><td> $1.94 \pm 0.13$ </td><td> $89.43 \pm 0.19$ </td><td> $52.60 \pm 0.22$ </td><td> $2.51 \pm 0.09$ </td></tr><tr><td>OTHERS</td><td> $92.64 \pm 0.09$ </td><td> $48.70 \pm 0.59$ </td><td> $1.57 \pm 0.12$ </td><td> $91.49 \pm 0.03$ </td><td> $47.36 \pm 0.63$ </td><td> $1.15 \pm 0.23$ </td><td> $90.56 \pm 0.28$ </td><td> $48.29 \pm 0.22$ </td><td> $3.07 \pm 0.15$ </td><td> $83.61 \pm 0.45$ </td><td> $51.11 \pm 0.04$ </td><td> $6.70 \pm 0.33$ </td></tr></table>

Table 8: Transferred adversarial examples. Comparing the effectiveness of unlearning when instead of using adversarial examples of the model, we use adversarial examples generated using Algorithm 1 on separately trained models with the same architecture. As the results show, relying on transferred adversarial examples in AMUN leads to worse results, specially for test accuracy because the adversarial examples do not necessary belong to the natural distribution learned by the model. However, even by using these transferred adversarial examples AMUN outperforms prior SOTA unlearning methods, specially when there is no access to $\mathcal { D } _ { \mathrm { R } }$ .

![](images/6550a6f5d37f575446a42d4bd36c7fb7020152913f73363e1621b484aa0db526.jpg)

<details>
<summary>line</summary>

| Request counts | FT AUC (Line 1) | FT AUC (Line 2) | FT AUC (Line 3) | FT AUC (Line 4) | FT AUC (Line 5) |
| -------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| 1              | 40              | 47              | 52              | 58              | 62              |
| 2              | 45              | 50              | 53              | 59              | 61              |
| 3              | 47              | 51              | 52              | 58              | 60              |
| 4              | 48              | 52              | 53              | 59              | 61              |
| 5              | 49              | 53              | 54              | 60              | 62              |
</details>

![](images/f97960bc8d6daca2aa951a96ded9128df6676fdea6ec8ce3ecd3464570f7ea1c.jpg)

<details>
<summary>line</summary>

| Request counts | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
| -------------- | ------ | ------ | ------ | ------ | ------ |
| 1              | 70     | 68     | 60     | 58     | 52     |
| 2              | 68     | 66     | 59     | 57     | 51     |
| 3              | 66     | 64     | 58     | 56     | 52     |
| 4              | 64     | 62     | 57     | 55     | 53     |
| 5              | 62     | 60     | 56     | 54     | 52     |
</details>

![](images/4bfe307f6251dc25e720c585b04d7054c17a3aac032f179a0943e3895cdbebeb.jpg)

<details>
<summary>line</summary>

| Request counts | FT AUC (Solid Line) | FT AUC (Dashed Line) |
| -------------- | ------------------- | --------------------- |
| 1              | 48                  | 47                    |
| 2              | 50                  | 49                    |
| 3              | 52                  | 51                    |
| 4              | 53                  | 52                    |
| 5              | 53                  | 51                    |
</details>

![](images/0bd8819026da3d6a0573692b62adea3e14e6f87ce2c8c5c1c1c00fbc7cae7511.jpg)

<details>
<summary>line</summary>

| Request counts | Amun  | Amun-A | SalUn | RL    | GA    | BS    |
| -------------- | ----- | ------ | ----- | ----- | ----- | ----- |
| 1              | 57.5  | 58.5   | 51.0  | 51.0  | 51.0  | 51.0  |
| 2              | 56.0  | 56.0   | 51.0  | 51.0  | 51.0  | 51.0  |
| 3              | 54.5  | 55.0   | 50.5  | 50.5  | 50.5  | 50.5  |
| 4              | 53.5  | 54.0   | 50.0  | 50.0  | 50.0  | 50.0  |
| 5              | 53.0  | 53.5   | 50.0  | 50.0  | 50.0  | 50.0  |
</details>

Figure 8: This figure shows both FT AUC and RF AUC components of the plots presented in Figure 2. The two left-most sub-figures show these values along the number of unlearning requests for the case where there is access to $\mathcal { D } _ { \mathrm { R } }$ and the two right-most ones show these values when there is no access to $\mathcal { D } _ { \mathrm { R } }$ .

# F.5. Weak Attacks

In this section we evaluate the effectiveness of using weaker attacks in Algorithm 1. For this purpose, we perform the unlearning on a ResNet-18 model trained on CIFAR-10 in all unlearning settings mentioned in $\ S \ : 5 ,$ and compare the results with the default choice of PGD-50 in AMUN. The weaker attack that we use is a variation of FFGSM (Wong et al., 2020), which itself is a variant of FGSM (Goodfellow et al., 2014). FGSM takes steps toward the gradient sign at a given sample to find adversarial samples. FFGSM takes a small step toward a random direction first, and then proceeds with FGSM. To adapt these method to the format of Algorithm 1 we start with FGSM attack; we find the gradient sign and start to move toward that direction in steps of size ϵ until we find an adversarial example. If the adversarial example is not found after a few iteration of the While loop, we restart the value of ϵ and add a small random perturbation before the next round of FGSM attack and the $\mathbb { W } \mathrm { h i 1 e }$ loop. We continue this procedure to find an adversarial sample. After deriving a new set of adversarial examples using this methods, we performed a separate round hyper-parameter tuning for unlearning with the new attack to have a fair comparison. It is notable to mention that this leads to a much faster attack because we only compute the gradient once for each round round of FGSM (at the beginning or after each addition of random perturbation and restarting FGSM). Table 9 shows the comparison of the results with the original version of AMUN that uses PGD-50. As the results show, using this weaker attack leads to worse results; however, they still outperform prior SOTA methods in unlearning, specially in the setting where there is no access to $\mathcal { D } _ { \mathrm { R } }$ and the size of $\mathcal { D } _ { \mathrm { F } }$ is 50% of D.

For each image in CIFAR-10, Figure 9 shows $\delta _ { x }$ (see Definition 2.1) for the adversarial examples that Algorithm 1 finds using PGD-50 (x-axis) and FFGSM (y-axis). The dashed line shows the $x = y$ line for the reference. As the figure shows $\delta _ { x }$ is much smaller for PGD-50. This value is smaller for FFGSM for less than 4% of the images, but still even for those images, the value of $\delta _ { x }$ for PGD-50 is very small, compared to the range of values that are required for FFGSM in many cases. This, we believe, is the main reason behind worse performance when using FFGSM. However, still note that the adversarial examples that are found using FFGSM belong to the natural distribution of the trained model and therefore fine-tuning the model on these samples does not lead to noticable deterioration of the test accuracy, while achieving reasonable FT AUC score. Indeed this larger distance of the adversarial examples with the original samples in $\mathcal { D } _ { \mathrm { F } }$ , leads to better performance of AMUN when it does not include $\mathcal { D } _ { \mathrm { F } }$ when fine-tuning the model, because the difference in the predicted logits compared to the $\delta _ { x }$ leads to under-estimation of the local Lipschitz constant and therefore, the model is able to fit perfectly to both the original samples and its corresponding adversarial sample without changing much. This consequently leads to a larger value

of FT AUC score. 

<table><tr><td rowspan="3"></td><td colspan="6">WITH ACCESS TO  $\mathcal{D}_{\text{R}}$ </td><td colspan="6">No ACCESS TO  $\mathcal{D}_{\text{R}}$ </td></tr><tr><td colspan="3">RANDOM FORGET (10%)</td><td colspan="3">RANDOM FORGET (50%)</td><td colspan="3">RANDOM FORGET (10%)</td><td colspan="3">RANDOM FORGET (50%)</td></tr><tr><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td><td>TEST AUC</td><td>FT AUC</td><td>AVG. GAP</td><td>TEST ACC</td><td>FT AUC</td><td>AVG. GAP</td><td>TEST AUC</td><td>FT AUC</td><td>AVG. GAP</td></tr><tr><td>PGD-50</td><td> $93.45 \pm 0.22$ </td><td> $50.18 \pm 0.36$ </td><td> $0.62 \pm 0.05$ </td><td> $92.39 \pm 0.04$ </td><td> $49.99 \pm 0.18$ </td><td> $0.33 \pm 0.03$ </td><td> $91.67 \pm 0.04$ </td><td> $52.24 \pm 0.23$ </td><td> $1.94 \pm 0.13$ </td><td> $89.43 \pm 0.19$ </td><td> $52.60 \pm 0.22$ </td><td> $2.51 \pm 0.09$ </td></tr><tr><td>FGSM</td><td> $93.87 \pm 0.16$ </td><td> $50.64 \pm 0.51$ </td><td> $0.92 \pm 0.25$ </td><td> $89.41 \pm 1.01$ </td><td> $50.93 \pm 0.46$ </td><td> $1.81 \pm 0.77$ </td><td> $92.14 \pm 0.28$ </td><td> $56.58 \pm 1.05$ </td><td> $3.46 \pm 0.36$ </td><td> $90.12 \pm 0.28$ </td><td> $54.54 \pm 0.47$ </td><td> $3.29 \pm 0.10$ </td></tr></table>

Table 9: Using weaker attacks. Comparing the effectiveness of unlearning when PGD-10 in Algorithm 1 is replaced with a variant of FGSM attack, which is considered to be significantly weaker and leads to finding adversarial examples at a much higher distance to the original samples. We evaluate unlearning 10% and 50% of the training samples in CIFAR-10 from a trained ResNet-18 model. As the results show, in both settings of unlearning (with access to $\mathcal { D } _ { \mathrm { R } }$ and no access to $\mathcal { D } _ { \mathtt { R } } )$ , using the weaker attack does not perform as well as the original method. However, it still outperforms prior SOTA unlearning methods.

# G. Comparison Using Prior Evaluation Methods

In this section we perform similar comparisons to what we presented in section 6.1, but instead of FT AUC, we use the same MIA used by prior SOTA methods in unlearning for evaluations. As mentioned in section 5.2, we refer to the score derived by this MIA as MIS.

<table><tr><td rowspan="2"></td><td colspan="5">RANDOM FORGET (10%)</td><td colspan="5">RANDOM FORGET (50%)</td></tr><tr><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>MIS</td><td>AVG. GAP</td><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>MIS</td><td>AVG. GAP</td></tr><tr><td>RETRAIN</td><td> $94.49 \pm 0.20$ </td><td> $100.0 \pm 0.00$ </td><td> $94.33 \pm 0.18$ </td><td> $12.53 \pm 0.32$ </td><td>0.00</td><td> $92.09 \pm 0.37$ </td><td> $100.0 \pm 0.00$ </td><td> $91.85 \pm 0.33$ </td><td> $16.78 \pm 0.37$ </td><td>0.00</td></tr><tr><td>FT</td><td> $95.16 \pm 0.29$ </td><td> $96.64 \pm 0.25$ </td><td> $92.21 \pm 0.27$ </td><td> $11.33 \pm 0.35$ </td><td> $1.84 \pm 0.10$ </td><td> $94.24 \pm 0.30$ </td><td> $95.22 \pm 0.31$ </td><td> $91.21 \pm 0.33$ </td><td> $12.10 \pm 0.72$ </td><td> $3.06 \pm 0.24$ </td></tr><tr><td>RL</td><td> $99.22 \pm 0.19$ </td><td> $99.99 \pm 0.01$ </td><td> $94.10 \pm 0.11$ </td><td> $10.94 \pm 0.45$ </td><td> $1.64 \pm 0.19$ </td><td> $92.98 \pm 1.07$ </td><td> $94.83 \pm 1.04$ </td><td> $89.19 \pm 0.74$ </td><td> $12.48 \pm 0.90$ </td><td> $3.29 \pm 0.04$ </td></tr><tr><td>GA</td><td> $98.94 \pm 1.39$ </td><td> $99.22 \pm 1.31$ </td><td> $93.39 \pm 1.18$ </td><td> $4.21 \pm 5.25$ </td><td> $3.62 \pm 1.04$ </td><td> $99.94 \pm 0.09$ </td><td> $99.95 \pm 0.08$ </td><td> $94.36 \pm 0.31$ </td><td> $0.62 \pm 0.30$ </td><td> $6.64 \pm 0.15$ </td></tr><tr><td>BS</td><td> $99.14 \pm 0.31$ </td><td> $99.89 \pm 0.06$ </td><td> $93.04 \pm 0.14$ </td><td> $5.50 \pm 0.39$ </td><td> $3.27 \pm 0.13$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.62 \pm 0.08$ </td><td> $0.40 \pm 0.05$ </td><td> $6.77 \pm 0.03$ </td></tr><tr><td> $l_1$ -SPARSE</td><td> $94.29 \pm 0.34$ </td><td> $95.63 \pm 0.16$ </td><td> $91.55 \pm 0.17$ </td><td> $12.03 \pm 1.92$ </td><td> $2.26 \pm 0.26$ </td><td> $92.63 \pm 0.13$ </td><td> $95.02 \pm 0.10$ </td><td> $89.56 \pm 0.08$ </td><td> $12.03 \pm 0.39$ </td><td> $3.14 \pm 0.17$ </td></tr><tr><td>SALUN</td><td> $99.25 \pm 0.12$ </td><td> $99.99 \pm 0.01$ </td><td> $94.11 \pm 0.13$ </td><td> $11.29 \pm 0.56$ </td><td> $1.56 \pm 0.20$ </td><td> $95.69 \pm 0.80$ </td><td> $97.26 \pm 0.79$ </td><td> $91.55 \pm 0.59$ </td><td> $11.27 \pm 0.94$ </td><td> $3.06 \pm 0.12$ </td></tr><tr><td>AMUN</td><td> $95.45 \pm 0.19$ </td><td> $99.57 \pm 0.00$ </td><td> $93.45 \pm 0.22$ </td><td> $12.55 \pm 0.08$ </td><td> $0.59 \pm 0.09$ </td><td> $93.50 \pm 0.09$ </td><td> $99.71 \pm 0.01$ </td><td> $92.39 \pm 0.04$ </td><td> $13.53 \pm 0.19$ </td><td> $1.37 \pm 0.07$ </td></tr><tr><td>AMUN+ $SalUn$ </td><td> $94.73 \pm 0.07$ </td><td> $99.92 \pm 0.01$ </td><td> $93.95 \pm 0.18$ </td><td> $14.23 \pm 0.40$ </td><td> $0.60 \pm 0.10$ </td><td> $93.56 \pm 0.07$ </td><td> $99.72 \pm 0.02$ </td><td> $92.52 \pm 0.20$ </td><td> $13.33 \pm 0.10$ </td><td> $1.47 \pm 0.01$ </td></tr></table>

Table 10: Unlearning with access to $\mathcal { D } _ { \mathbf { R } } .$ . Comparing different unlearning methods in unlearning 10% and $5 0 \%$ of D. Avg. Gap (see § 5.2), with MIS as the MIA score, is used for evaluation (lower is better). The lowest value is shown in bold while the second best is specified with underscore. As the results show, AMUN outperforms all other methods by achieving lowest Avg. Gap and $\mathbf { A M U N } _ { S a l U n }$ achieves comparable results.

<table><tr><td rowspan="2"></td><td colspan="5">RANDOM FORGET (10%)</td><td colspan="5">RANDOM FORGET (50%)</td></tr><tr><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>MIA</td><td>AVG. GAP</td><td>UNLEARN ACC</td><td>RETAIN ACC</td><td>TEST ACC</td><td>MIA</td><td>AVG. GAP</td></tr><tr><td>RETRAIN</td><td> $94.49 \pm 0.20$ </td><td> $100.0 \pm 0.00$ </td><td> $94.33 \pm 0.18$ </td><td> $12.53 \pm 0.32$ </td><td>0.00</td><td> $92.09 \pm 0.37$ </td><td> $100.0 \pm 0.00$ </td><td> $91.85 \pm 0.33$ </td><td> $16.78 \pm 0.37$ </td><td>0.00</td></tr><tr><td>RL</td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.45 \pm 0.09$ </td><td> $3.06 \pm 0.63$ </td><td> $3.77 \pm 0.13$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.54 \pm 0.11$ </td><td> $0.40 \pm 0.03$ </td><td> $6.75 \pm 0.02$ </td></tr><tr><td>GA</td><td> $4.77 \pm 3.20$ </td><td> $5.07 \pm 3.54$ </td><td> $5.09 \pm 3.38$ </td><td> $32.63 \pm 50.85$ </td><td> $76.58 \pm 7.73$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.57 \pm 0.06$ </td><td> $0.35 \pm 0.10$ </td><td> $6.77 \pm 0.04$ </td></tr><tr><td>BS</td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.48 \pm 0.04$ </td><td> $1.11 \pm 0.30$ </td><td> $4.27 \pm 0.07$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.59 \pm 0.03$ </td><td> $0.38 \pm 0.02$ </td><td> $6.76 \pm 0.01$ </td></tr><tr><td>SALUN</td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.47 \pm 0.10$ </td><td> $2.39 \pm 0.64$ </td><td> $3.95 \pm 0.14$ </td><td> $100.00 \pm 0.00$ </td><td> $100.00 \pm 0.00$ </td><td> $94.57 \pm 0.12$ </td><td> $0.33 \pm 0.04$ </td><td> $6.77 \pm 0.03$ </td></tr><tr><td>AMUN</td><td> $94.28 \pm 0.37$ </td><td> $97.47 \pm 0.10$ </td><td> $91.67 \pm 0.04$ </td><td> $11.61 \pm 0.60$ </td><td> $1.61 \pm 0.09$ </td><td> $92.77 \pm 0.52$ </td><td> $95.66 \pm 0.25$ </td><td> $89.43 \pm 0.19$ </td><td> $14.13 \pm 0.67$ </td><td> $2.52 \pm 0.16$ </td></tr><tr><td>AMUN+Salun</td><td> $94.19 \pm 0.38$ </td><td> $97.71 \pm 0.06$ </td><td> $91.79 \pm 0.12$ </td><td> $11.66 \pm 0.16$ </td><td> $1.51 \pm 0.02$ </td><td> $91.90 \pm 0.63$ </td><td> $96.59 \pm 0.31$ </td><td> $89.98 \pm 0.44$ </td><td> $13.07 \pm 0.66$ </td><td> $2.35 \pm 0.15$ </td></tr></table>

Table 11: Unlearning with access to only ${ \mathcal { D } } _ { \mathbf { F } } .$ Comparing different unlearning methods in unlearning 10% and 50% of D. Avg. Gap (see § 5.2) is used for evaluation (lower is better) when only $\mathcal { D } _ { \mathrm { F } }$ is available during unlearning. As the results show, $\mathbf { A M U N } _ { S a l U n }$ significantly outperforms all other methods, and AMUN achieves comparable results.

# H. Continuous Unlearning (cont.)

In § 6.3, we showed AMUN, whether with adaptive computation of $\mathcal { D } _ { \mathrm { A } }$ or using the pre-computed ones, outperforms other unlearning methods when handling multiple unlearning requests. Another important observation on the presented results in Figure 2 is that the effectiveness of unlearning decreases with the number of unlearning requests. For the setting with access to $\mathcal { D } _ { \mathrm { R } }$ , this decrease is due to the fact that the $\mathcal { D } _ { \mathrm { F } }$ at each step has been a part of $\mathcal { D } _ { \mathrm { R } }$ at the previous steps; the model has been fine-tuned on this data in all the previous steps which has led to further improving confidence of the modes on predicting those samples. This result also matches the theoretical and experimental results in differential privacy literature as well (Dwork, 2006; Abadi et al., 2016).

This problem does not exist for the setting where there is no access to ${ \mathcal { D } } _ { \mathrm { R } } .$ , but we still see a decrease in the unlearning effectiveness as we increase the number of unlearning requests. The reason behind this deterioration is that the model itself is becoming weaker. As Figure 8 shows, the accuracy on the model on both $\mathcal { D } _ { \mathrm { R } }$ and $\mathcal { D } _ { \mathrm { T } }$ gets worse as it proceeds with the unlearning request; this is because each unlearning step shows the model only 2% (1K) of the samples and their corresponding adversarial examples for fine-tuning. So this deterioration is expected after a few unlearning requests. So when using AMUN in this setting (no access to $\mathcal { D } _ { \mathtt { R } } )$ in practice, it would be better to decrease the number of times that the unlearning request is performed, for example by performing a lazy-unlearning (waiting for a certain number of requests to accumulate) or at least using a sub-sample of $\mathcal { D } _ { \mathrm { R } }$ if that is an option.

![](images/29aa4d7fc83b69522f41516f403970c417a19d9a4c004d934f0e063e8413b06d.jpg)

<details>
<summary>scatter</summary>

| δₓ (PGD-50) | δₓ (FFGSM) |
| ----------- | ---------- |
| 0.0         | 0          |
| 0.5         | 5          |
| 1.0         | 15         |
</details>

Figure 9: For each image in CIFAR-10 the x-axis shows the Euclidean distance of the corresponding adversarial example that is found by using PGD-50 in Algorithm 1. y-axis shows this distance for the adversarial examples found by the variant of FFGSM in Algorithm 1. The dashed line shows the $x = y$ line. As the figure shows, the distance is much larger for weaker attacks and this leads to worse performance of AMUN, as explained by Theorem 4.1.