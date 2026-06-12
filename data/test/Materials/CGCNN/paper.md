# ARTICLE OPEN

? Check for updates

# Atomistic Line Graph Neural Network for improved materials property predictions

Kamal Choudhary 1,2,3,4 ✉ and Brian DeCost 1,4

Graph neural networks (GNN) have been shown to provide substantial performance improvements for atomistic material representation and modeling compared with descriptor-based machine learning models. While most existing GNN models for atomistic predictions are based on atomic distance information, they do not explicitly incorporate bond angles, which are critical for distinguishing many atomic structures. Furthermore, many material properties are known to be sensitive to slight changes in bond angles. We present an Atomistic Line Graph Neural Network (ALIGNN), a GNN architecture that performs message passing on both the interatomic bond graph and its line graph corresponding to bond angles. We demonstrate that angle information can be explicitly and efficiently included, leading to improved performance on multiple atomistic prediction tasks. We ALIGNN models for predicting 52 solid-state and molecular properties available in the JARVIS-DFT, Materials project, and QM9 databases. ALIGNN can outperform some previously reported GNN models on atomistic prediction tasks with better or comparable model training speed.

npj Computational Materials (2021) 7:185 ; https://doi.org/10.1038/s41524-021-00650-1

# INTRODUCTION

Graphs are a powerful non-Euclidean data structure method for establishing relationships between features (nodes) and their relationships (edges)1,2 . Graph neural networks (GNN)3,4 have immense potential for modeling complex phenomena. Common applications of GNNs include community detection and link prediction in social networks5,6 , functional time series on brain structures7 , gene DNA on regulatory networks8 , information flow through telecommunications networks9 , and property prediction for molecular and solid materials10. From a quantum chemistry point of view, GNNs provide a unique opportunity to predict properties of solids, molecules, and proteins in a much faster way rather than by solving the computationally expensive Schrodinger equation11–14.

There has been rapid progress in the development of GNN architectures for predicting material properties such as SchNet10, Crystal Graph Convolutional Neural Networks (CGCNN)15, MatErials Graph Network (MEGNet)16, improved Crystal Graph Convolutional Neural Networks (iCGCNN)17, OrbNet18, and similar variants19–31. This family of models represents a molecule or crystalline material as a graph with one node for each constituent atom and edges corresponding to interatomic bonds. A common theme is the use of elemental properties as node features and interatomic distances and/or bond valences as edge features. Through multiple layers of graph convolution updating node features based on their local chemical environment, these models can implicitly represent many-body interactions. However, many important material properties (especially electronic properties such as band gaps) are highly sensitive to structural features such as bond angles and local geometric distortions. It is possible that these models are not able to efficiently learn the importance of such manybody interactions. Explicit inclusion of angle-based information has already been shown to improve models with hand-crafted features such as classical force-field inspired descriptors (CFID)32. Recently, there has been growing interest in the explicit incorporation of bond angles and other many-body features17,19,20.

In this work, we use line graph neural networks inspired by those proposed in ref. 6 to develop an alternative way to include angular information to provide high accuracy models. Briefly, the line graph L(g) is a graph derived from another graph g that describes the connectivity of the edges in g. While the nodes of an atomistic graph correspond to atoms and its edges correspond to bonds, the nodes of an atomistic line graph correspond to interatomic bonds and its edges correspond to bond angles. Our model alternates between graph convolution on these two graphs, propagating bond angle information through interatomic bond representations to the atom-wise representations and vice versa. We use both the bond distances and angles in the line graph to incorporate finer details of atomic structure which leads to higher model performance. Our Atomistic Line Graph Neural Network (ALIGNN) models are implemented using the deep graph library (DGL)33 which allows efficient construction and neural message passing for different types of graphs. ALIGNN is a part of the Joint Automated Repository for Various Integrated Simulations (JARVIS) infrastructure34. We train ALIGNN models for several crystalline material properties from JARVIS-density functional theory (DFT)34–44 and Materials project45 (MP) datasets as well as molecular properties from QM946 database.

# RESULTS AND DISCUSSION

# Atomistic graph representation

ALIGNN performs Edge-gated graph convolution4 message passing updates on both the atomistic bond graph (atoms are nodes, bonds are edges) and its line graph (bonds are nodes, bond pairs with one common atom are edges). The Edge-gated graph convolution variant has the distinct advantage of updating both node and edge features. Because each edge in the bond graph directly corresponds to a node in the line graph, ALIGNN can aggregate features from bond pairs to efficiently update atom and bond representations by alternating between message passing updates on the bond graph and its line graph.

![](images/6e53cf54be1c79f030bcd49193537f4262a8695e5b7573940215be348e2ddf86.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Node features: v_i"] --> B["Edge features"]
    B --> C["triplet features: t_ijk"]
    C --> D["Nodes are bonds"]
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bfb,stroke:#333
    style D fill:#ffb,stroke:#333
```
</details>

Schematic showing undirected crystal graph representation and corresponding line graph construction for a $\sin _ { 4 }$ polyhedron. For Fig. 1simplicity, only Si–O bonds are illustrated. The ALIGNN convolution layer alternates between message passing on the bond graph (left) and its line graph (or bond adjacency graph, right).

For crystals, we use a periodic 12-nearest-neighbor graph construction. We expand this nearest-neighbor graph to include edges to all atoms in the neighbor shell of the 12th-nearest neighbor. Each node in the atomistic graph is assigned 9 input node features based on its atomic species: electronegativity, group number, covalent radius, valence electrons, first ionization energy, electron affinity, block, and atomic volume. This feature set is inspired by the ${ \mathsf { C G C N N } } ^ { 1 5 }$ model. The initial edge features are interatomic bond distances. We use a radial basis function (RBF) expansion with support between 0 and $8 \mathring { \mathsf { A } }$ for crystals and up to 5 Å for molecules. This undirected graph then can be represented as $G = ( \upsilon , \epsilon )$ where υ are nodes and є are edges i.e., a collection of $( \upsilon _ { \mathrm { i } } , \upsilon _ { \mathrm { j } } )$ linking vertices from $\upsilon _ { \mathrm { i } }$ to $\upsilon _ { \mathrm { j } } . ~ { \sf G }$ has an associated node feature set $H = \{ h _ { 1 } , ~ . . . , ~ h _ { \scriptscriptstyle  { N } } )$ , where $h _ { \mathrm { i } }$ is the feature vector associated with node $\textstyle v _ { \mathrm { i } } .$

# Atomistic line graph representation

The atomistic line graph is derived from the atomistic graph. Each node in the line graph corresponds to an edge in the original atomistic graph; both entities represent interatomic bonds, and in our work, they share latent representations. Edges in the line graph correspond to triplets of atoms or pairs of interatomic bonds. The initial line graph edge features are an RBF expansion of the bond angle cosines: $\begin{array} { r } { \mathsf { \tilde { \theta } } = \mathsf { a r c c o s } ( \frac { r _ { i j } \cdot r _ { j k } } { | r _ { i i } | r _ { i k } } ) . } \end{array}$ rij -rjk r r where $r _ { i j }$ and $r _ { j k }$ j j ij jkare atomic displacement vectors between atoms $i , \ j ,$ and k. A schematic of an atomistic graph and corresponding atomistic line graph is shown in Fig. 1. To avoid ambiguity between the node and edge features of the atomistic graph and its line graph, we write atom, bond, and triplet representations as h, e, and t.

# Edge gated graph convolution

ALIGNN uses Edge-gated graph convolution4 convolution for updating both node and edge features. This convolution is similar to the CGCNN update, except that edge features are only incorporated into normalized edge gates. Furthermore, edge gated graph convolution uses the pre-aggregated edge messages to update the edge representations.

Edge gated graph convolution updates node representations $h ^ { \prime }$ from layer l according to the formula:

$$
h _ {i} ^ {l + 1} = f \left(h _ {j} ^ {i} \left\{h _ {j} ^ {i} \right\} _ {j \in N _ {i}}\right) \tag {1}
$$

$$
h _ {i} ^ {l + 1} = h _ {i} ^ {l} + \text { SiLU } \left(\text { Norm } \left(W _ {s r c} ^ {l} h _ {i} ^ {l} + \sum_ {j \in N _ {i}} \hat {e} _ {i j} ^ {l} W _ {d s t} ^ {l} h _ {j} ^ {l}\right)\right) \tag {2}
$$

$$
\hat {e} _ {i j} ^ {I} = \frac {\sigma (e _ {i j} ^ {I})}{\sum_ {k \in N _ {i}} \sigma (e _ {i k} ^ {I}) + \in} \tag {3}
$$

$$
e _ {i j} ^ {I} = e _ {i j} ^ {I - 1} + \text { SiLU } (\text { Norm } (A ^ {I} h _ {i} ^ {I - 1} + B ^ {I} h _ {j} ^ {I - 1} + C ^ {I} e _ {i j} ^ {I - 1})) \tag {4}
$$

The edge messages in this Eq. (4) are equivalent to the gating term in the CGCNN update15, which coalesces the weight matrices $A , B ,$ and C into $W _ { \mathsf { g a t e } }$ , and the augmented edge representation

$$
z _ {i j} = h _ {i} \oplus h _ {j} \oplus e _ {i j} \tag {5}
$$

$$
e _ {i j} ^ {I} = e _ {i j} ^ {I - 1} + \text { SiLU } \left(\text { Norm } \left(W _ {\text { gate }} ^ {I} z _ {i j} ^ {I - 1}\right)\right) \tag {6}
$$

# ALIGNN update

One ALIGNN layer composes an edge-gated graph convolution on the bond graph (g) with an edge-gated graph convolution on the line graph (L(g)), as illustrated in Fig. 2. To avoid ambiguity between the node and edge features of the atomistic graph and its line graph, we write atom, bond, and triplet representations as h, e, and t. The line graph convolution produces bond messages m that are propagated to the atomistic graph, which further updates the bond features in combination with atom features h.

$$
m ^ {I}, t ^ {I} = \text { Edge   Gated   Graph   Conv } (L (g), e ^ {I - 1}, t ^ {I - 1}) \tag {7}
$$

$$
h ^ {l}, e ^ {l} = \text { Edge   Gated   Graph   Conv } (g, h ^ {l - 1}, m ^ {l}) \tag {8}
$$

# Overall model architecture and training

We use N layers of ALIGNN updates followed by M layers of edgegated graph convolution (GCN) updates on the bond graph. We use Sigmoid Linear Unit (SiLU, also known as Swish) activations instead of rectified linear unit (ReLU) or Softplus because it is twice differentiable like Softplus but can result in a better empirical performance like ReLU on many tasks. After $N + M$ graph convolution layers, our networks perform global average pooling over nodes and finally predict the target properties with a single fully connected regression or classification layers. Table 1 presents the default hyperparameters of the ALIGNN model used to train the models reported in “Model performance” section. These hyperparameters were selected through a combination of hypothesis-driven experiments and random hyperparameter search, as discussed in detail in the “Methods” section. “Model analysis” section provides a detailed analysis of the sensitivity of model performance and computational cost.

![](images/1b793c3996b610cd3b9ccf28ed0ad6361061d1f585d3c8638ef0898668f95ebe.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph EdgeGatedConv on L(g)
        A["e_ij^ℓ"] --> B["W_src^ℓlg"]
        C["{e_jk^ℓ}"] --> D["W_dst^ℓlg"]
        E["{t_ij^ℓ}"] --> F["⊕"]
        G["{e_ij^ℓ}"] --> H["W_gate^ℓlg"]
        I["Σu"] --> J["SILU"]
    end

    subgraph EdgeGatedConv on g
        K["h_i^ℓ"] --> L["W_src^ℓ"]
        M["{h_j^ℓ}"] --> N["W_dst^ℓ"]
        O["Σu"] --> P["W_gate^ℓ"]
        Q["Σu"] --> R["SILU"]
        S["h_i^{ℓ+1}"] --> T["Sum"]
        U["{e_ij^{ℓ+1}"] --> V["Sum_j"]
    end

    B --> W["Sum"]
    D --> X["Sum"]
    F --> Y["⊕"]
    H --> Z["⊕"]
    J --> AA["⊕"]
    L --> AB["Σu"]
    N --> AC["Σu"]
    P --> AD["Σu"]
    Q --> AE["Σu"]
    R --> AF["Σu"]
    AA --> AG["Σu"]
    AB --> AH["Σu"]
    AC --> AI["Σu"]
    AD --> AJ["Σu"]
    AF --> AK["Σu"]
    AG --> AL["Σu"]
    AH --> AM["Σu"]
    AI --> AN["Σu"]
    AJ --> AO["Σu"]
    AK --> AP["Σu"]
    AL --> AQ["Σu"]
    AM --> AR["Σu"]
    AN --> AS["Σu"]
    AO --> AT["Σu"]
    AP --> AU["Σu"]
    AQ --> AV["Σu"]
    AR --> AW["Σu"]
    AS --> AX["Σu"]
    AT --> AY["Σu"]
    AU --> AZ["Σu"]
    AV --> BA["Σu"]
    AW --> BB["Σu"]
    AX --> BC["Σu"]
    AY --> BD["Σu"]
    AZ --> BE["Σu"]
```
</details>

Schematic of the ALIGNN layer structure. The ALIGNN layer Fig. 2first performs edge-gated graph convolution on the line graph to update pair and triplet features. The newly updated pair features are propagated to the edges of the direct graph and further updated with the atom features in a second edge-gated graph convolution applied to the direct graph.

ALIGNN model configuration used for both solid-state and Table 1.molecular machine learning models. 

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>ALIGNN layers</td><td>4</td></tr><tr><td>GCN layers</td><td>4</td></tr><tr><td>Edge input features</td><td>80</td></tr><tr><td>Triplet input features</td><td>40</td></tr><tr><td>Embedding features</td><td>64</td></tr><tr><td>Hidden features</td><td>256</td></tr><tr><td>Normalization</td><td>Batch normalization</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Learning rate</td><td>0.001</td></tr></table>

# Model performance

Model performance can vary substantially depending on the dataset and task. To evaluate the performance of ALIGNN, we currently use two different solid-state property datasets (Materials Project and JARVIS-DFT) as well as molecular property dataset QM9. Because the solid-state datasets are continuously updated, we use time-versioned snapshots of them, specifically selecting the MP version used by previous works to facilitate a direct comparison of model performance with the literature. It is likely that as these dataset sizes increase in the future the performance of the model can be further improved. We select the MP 2018.6.1 version which consists of $^ { 6 9 , 2 3 9 }$ materials with properties such as Perdew Burke-Ernzerhof functional $( \mathsf { P B E } ) ^ { 4 7 }$ bandgaps and formation energies. Similarly, we use 2021.8.18 version of JARVIS-DFT dataset, which consists of 55,722 materials with several properties such as van der Waals correction with optimized Becke88 functional $( \mathsf { O p t B 8 8 v d W } ) ^ { 4 8 }$ bandgaps, formation energies, dielectric constants, Tran-Blaha modified Becke Johnson potential $( M B ) ) ^ { 4 9 }$ bandgaps and dielectric constants, bulk, shear modulus, magnetic moment, density functional perturbation theory (DFPT) based maximum piezoelectric coefficients, Boltztrap50 based Seebeck coefficient, power factor, maximum absolute value of electric field gradient and two-dimensional materials exfoliation energies. All of these properties are critical for functional materials design. For the MP dataset we use a train-validation-test split of 60,000–5000–4239 as used by SchNet10 and MEGNet16. For the JARVIS-DFT dataset and its properties, we use 80 %:10 %: 10 % splits. For QM9 dataset we

<table><tr><td colspan="9">Table 2. Test set performance on the Materials Project dataset.</td></tr><tr><td>Prop</td><td>Unit</td><td>MAD</td><td>CFID</td><td>CGCNN</td><td>MEGNet</td><td>SchNet</td><td>ALIGNN</td><td>MAD: MAE</td></tr><tr><td> $E_{f}$ </td><td>eV/at.</td><td>0.93</td><td>0.104</td><td>0.039</td><td>0.028</td><td>0.035</td><td>0.022</td><td>42.27</td></tr><tr><td> $E_{g}$ </td><td>eV</td><td>1.35</td><td>0.434</td><td>0.388</td><td>0.33</td><td>—</td><td>0.218</td><td>6.19</td></tr><tr><td colspan="9">Predictions on test set are shown in parity plots in Supplementary Figs. 1, 2.</td></tr></table>

use a train-validation-test split of 110,000–10,000–10,829 as used by SchNet10, DimeNet $+ + ^ { 2 0 }$ , and MEGNet16.

Performance of ALIGNN models on MP is shown in Table 2, which shows the regression model performance in terms of mean absolute error metric (MAE). The best MAEs for formation energy (Ef) and band gap $( E _ { \mathsf { g } } )$ with ALIGNN are 0.022 eV(atom)−1 and 0.218 eV, respectively. In terms of $E _ { \mathrm { f } } ,$ ALIGNN outperforms reported values of CGCNN, MEGNet, and SchNet models by 43.6%, 21.4%, and 37.1%, respectively. For $E _ { \mathfrak { g } } ,$ ALIGNN outperforms CGCNN and MEGNet by 43.8% and 33.9%, respectively. Good performance on well-known and well-characterized datasets ensures high prediction accuracy of ALIGNN models. Because each property has different units and in general a different variance, we also report the mean absolute deviation (MAD) for each property to facilitate an unbiased comparison of the model performance between different properties. The MAD values represent the performance of a random guessing model with average value prediction for each data point. We also report the CFID based predictions for comparison. Clearly, all the neural networks, especially ALIGNN, perform much better than the corresponding MAD of the dataset as well as CFID performance. Analyzing the MAD: MAE (ALIGNN) ratio, we observe that the ratio could be as high as 42.27 model. Generally, a model with high MAD:MAE ratio (such as 5 and above) is considered a good predictive model51.

Similarly, we train ALIGNN models on the JARVIS-DFT34–44 dataset which consists of data for $^ { 5 5 , 7 2 2 }$ materials. In addition to properties such as formation energies, and bandgaps it also consists several unique quantities such as solar-cell efficiency (spectroscopic limited maximum efficiency, SLME), topological spin-orbit spillage, dielectric constant with $( \epsilon _ { \times } \left( \mathsf { D F P T } \right) )$ , and without ionic contributions $( \epsilon _ { \times }$ (OPT, MBJ)), exfoliation energies for twodimensional (2D), electric field gradients (EFG), Voigt bulk (Kv) and shear modulus $( \mathsf { G v } ) ,$ , energy above convex hull (ehull), maximum piezoelectric stress $( e _ { \mathrm { i j } } )$ and strain $( d _ { \mathrm { i j } } )$ tensors, n-type and p-type Seebeck coefficient and power factors (PF), crystallographic averages of electron $( m _ { \mathrm { e } } )$ and hole $( m _ { \mathsf { h } } )$ effective masses. As we converge plane wave-cutoff (ENCUT) and k-points used in Brillouin zone integration (Kpoint-length), we attempt to make machine learning predictions on these unique quantities as well. Such a large variety of properties allow a thorough testing of our ALIGNN models. More details for individual properties, its precision with respect to experimental measurements, applicability, and limitations can be found in respective works. However, it is important to mention that many important issues such as tackling systematic underestimation of bandgaps by DFT methods, the inclusion of van der Waals bonding, and the inclusion of spin-orbit coupling interactions, all critically important for materials-design perspective have been key areas of improvements for the JARVIS-DFT dataset. For instance, meta-GGA (generalized gradient approximation) based Tran-Blaha modified Becke Johnson potential (TBmBJ) band gaps are more reliable and comparable to experimental data than Perdew Burke-Ernzerhof functional (PBE) or van der Waals correction with optimized Becke88 functional (OptB88vdW) bandgaps, but their calculations are computationally expensive and hence they are underrepresented in the dataset. In addition to the ALIGNN performance, we also include hand-crafted Classical force-field inspired descriptors (CFID) descriptor and

<table><tr><td colspan="7">Table 3. Regression model performances on JARVIS-DFT dataset for 29 properties using CFID, CGCNN and ALIGNN models on 55,722 materials.</td></tr><tr><td>Property</td><td>Units</td><td>MAD</td><td>CFID</td><td>CGCNN</td><td>ALIGNN</td><td>MAD: MAE</td></tr><tr><td>Formation energy</td><td> $eV(atom)^{-1}$ </td><td>0.86</td><td>0.14</td><td>0.063</td><td>0.033</td><td>26.06</td></tr><tr><td>Bandgap (OPT)</td><td>eV</td><td>0.99</td><td>0.30</td><td>0.20</td><td>0.14</td><td>7.07</td></tr><tr><td>Total energy</td><td> $eV(atom)^{-1}$ </td><td>1.78</td><td>0.24</td><td>0.078</td><td>0.037</td><td>48.11</td></tr><tr><td>Ehull</td><td>eV</td><td>1.14</td><td>0.22</td><td>0.17</td><td>0.076</td><td>15.00</td></tr><tr><td>Bandgap (MBJ)</td><td>eV</td><td>1.79</td><td>0.53</td><td>0.41</td><td>0.31</td><td>5.77</td></tr><tr><td>Kv</td><td>GPa</td><td>52.80</td><td>14.12</td><td>14.47</td><td>10.40</td><td>5.08</td></tr><tr><td>Gv</td><td>GPa</td><td>27.16</td><td>11.98</td><td>11.75</td><td>9.48</td><td>2.86</td></tr><tr><td>Mag. mom</td><td>μB</td><td>1.27</td><td>0.45</td><td>0.37</td><td>0.26</td><td>4.88</td></tr><tr><td>SLME (%)</td><td>No unit</td><td>10.93</td><td>6.22</td><td>5.66</td><td>4.52</td><td>2.42</td></tr><tr><td>Spillage</td><td>No unit</td><td>0.52</td><td>0.39</td><td>0.40</td><td>0.35</td><td>1.49</td></tr><tr><td>Kpoint-length</td><td>Å</td><td>17.88</td><td>9.68</td><td>10.60</td><td>9.51</td><td>1.88</td></tr><tr><td>Plane-wave cutoff</td><td>eV</td><td>260.4</td><td>139.4</td><td>151.0</td><td>133.8</td><td>1.95</td></tr><tr><td> $\epsilon_x$ (OPT)</td><td>No unit</td><td>57.40</td><td>24.83</td><td>27.17</td><td>20.40</td><td>2.81</td></tr><tr><td> $\epsilon_y$ (OPT)</td><td>No unit</td><td>57.54</td><td>25.03</td><td>26.62</td><td>19.99</td><td>2.88</td></tr><tr><td> $\epsilon_z$ (OPT)</td><td>No unit</td><td>56.03</td><td>24.77</td><td>25.69</td><td>19.57</td><td>2.86</td></tr><tr><td> $\epsilon_x$ (MBJ)</td><td>No unit</td><td>64.43</td><td>30.96</td><td>29.82</td><td>24.05</td><td>2.68</td></tr><tr><td> $\epsilon_y$ (MBJ)</td><td>No unit</td><td>64.55</td><td>29.89</td><td>30.11</td><td>23.65</td><td>2.73</td></tr><tr><td> $\epsilon_z$ (MBJ)</td><td>No unit</td><td>60.88</td><td>29.18</td><td>30.53</td><td>23.73</td><td>2.57</td></tr><tr><td>ε (DFPT:elec+ionic)</td><td>No unit</td><td>45.81</td><td>43.71</td><td>38.78</td><td>28.15</td><td>1.63</td></tr><tr><td>Max. piezoelectric strain coeff ( $d_{ij}$ )</td><td> $CN^{-1}$ </td><td>24.57</td><td>36.41</td><td>34.71</td><td>20.57</td><td>1.19</td></tr><tr><td>Max. piezo. stress coeff ( $e_{ij}$ )</td><td> $Cm^{-2}$ </td><td>0.26</td><td>0.23</td><td>0.19</td><td>0.147</td><td>1.77</td></tr><tr><td>Exfoliation energy</td><td> $meV(atom)^{-1}$ </td><td>62.63</td><td>63.31</td><td>50.0</td><td>51.42</td><td>1.22</td></tr><tr><td>Max. EFG</td><td> $10^{21} Vm^{-2}$ </td><td>43.90</td><td>24.54</td><td>24.7</td><td>19.12</td><td>2.30</td></tr><tr><td>avg.  $m_e$ </td><td>electron mass unit</td><td>0.22</td><td>0.14</td><td>0.12</td><td>0.085</td><td>2.59</td></tr><tr><td>avg.  $m_h$ </td><td>electron mass unit</td><td>0.41</td><td>0.20</td><td>0.17</td><td>0.124</td><td>3.31</td></tr><tr><td>n-Seebeck</td><td> $μVK^{-1}$ </td><td>113.0</td><td>56.38</td><td>49.32</td><td>40.92</td><td>2.76</td></tr><tr><td>n-PF</td><td> $μW(mK^2)^{-1}$ </td><td>697.80</td><td>521.54</td><td>552.6</td><td>442.30</td><td>1.58</td></tr><tr><td>p-Seebeck</td><td> $μVK^{-1}$ </td><td>166.33</td><td>62.74</td><td>52.68</td><td>42.42</td><td>3.92</td></tr><tr><td>p-PF</td><td> $μW(mK^2)^{-1}$ </td><td>691.67</td><td>505.45</td><td>560.8</td><td>440.26</td><td>1.57</td></tr></table>

Predictions on test set are shown in Supplementary Figs. 3–31.

CGCNN MAE performances for these properties using identical data-splits.

In Table 3 we show the performance on regression models for different properties in the JARVIS-DFT database. We observe that ALIGNN models outperform CFID descriptors by up to 4 times, suggesting GNNs can be a very powerful method for multiple material property predictions. Also, ALIGNN outperforms CGCNN by more than 2 times (such as for OptB88vdW total energy). Crossdataset comparison of corresponding property entries in Tables 2, 3 shows that generally models generally obtain better performance on the MP dataset, which we attribute primarily to the larger size of MP. For example, the MAE for the formation energy target on MP dataset is 50% lower than for JARVIS-DFT. However, for some targets, the differences in the DFT method and settings, as well as potential differences in the material-space distribution, might significantly contribute to the difficulty of a prediction task. For example, the MAE on high throughput band gaps is lower (by 35.7%) for the JARVIS-DFT dataset, which is interesting in light of MP’s dataset size advantage over JARVIS-DFT. One potential source of this discrepancy is the differing computational methodologies used, such as different functionals (PBE vs OptB88vdW), use of the DFT+U method, and settings for various DFT hyperparameters like smearing and k-point settings, all of which can influence the values of computed bandgaps as discussed in ref. 37. Another potential contributing factor could be differing levels of dataset bias in the MP and JARVIS-DFT

datasets stemming from differing distributions in material space. Clarifying this situation is beyond the scope of the present work, though it is of great importance for the atomistic modeling community to resolve.

Nevertheless, application of ALIGNN models on different datasets shows improvements for materials-property predictions. Both CFID, CGCNN and ALIGNN models’ MAEs are lower than the corresponding MADs. The MAD:MAE ratios can vary for energy related quantities from a high value of 48.11 (total energy), and 26.06 (formation energy model) to low values such as for DFPT based piezoelectric strain coefficients (1.19) and dielectric constant with ionic contributions (1.63). The results indicate that there is still much room for improvement for the GNN models, especially for electronic properties.

As we notice above, the regression tasks for some of the electronic properties do not show very high MAD: MAE. we train classification models for some of them. Classification tasks predict labels such as high value/low value (based on a selected threshold) as 1 and 0 instead of predicting actual data in regression tasks. Such models can be useful for fast screening purposes38 for computationally expensive methods. We evaluate the performance of these classifiers using the receiver operating characteristic curve area under the curve (ROC AUC). A random guessing model has a ROC AUC of 0.5, while a perfect model would be a ROC AUC of 1.0. Interestingly, we notice most of our classification models (as shown in Table 4) have high ROC AUCs, ranging up to a maximum value of 0.94 (for convex hull stability) showing their usefulness for material classification-based applications. All results are based on the performance of 10 % test data which is never used during the training or model selection procedures.

Classification task ROC AUC performance on JARVIS-DFT Table 4.dataset for ALIGNN models. 

<table><tr><td>Model</td><td>Threshold</td><td>ALIGNN</td></tr><tr><td>Metal/non-metal classifier (OPT)</td><td>0.01 eV</td><td>0.92</td></tr><tr><td>Metal/non-metal classifier (MBJ)</td><td>0.01 eV</td><td>0.92</td></tr><tr><td>Magnetic/non-magnetic classifier</td><td>0.05 μB</td><td>0.91</td></tr><tr><td>High/low SLME</td><td>10 %</td><td>0.83</td></tr><tr><td>High/low spillage</td><td>0.1</td><td>0.80</td></tr><tr><td>Stable/unstable (ehull)</td><td>0.1 eV</td><td>0.94</td></tr><tr><td>High/low-n-Seebeck</td><td> $-100 \mu VK^{-1}$ </td><td>0.88</td></tr><tr><td>High/low-p-Seebeck</td><td> $100 \mu VK^{-1}$ </td><td>0.92</td></tr><tr><td>High/low-n-powerfactor</td><td> $1000 \mu W(mK^{2})^{-1}$ </td><td>0.74</td></tr><tr><td>High/low-p-powerfactor</td><td> $1000 \mu W(mK^{2})^{-1}$ </td><td>0.74</td></tr></table>

The ROC curve plots for these models are provided in Supplementary Figs. 32–41.

Regression model performances on QM9 dataset for 11 Table 5.properties using ALIGNN. 

<table><tr><td>Target</td><td>Units</td><td>SchNet</td><td>MEGNet</td><td>DimeNet++</td><td>ALIGNN</td></tr><tr><td>HOMO</td><td>eV</td><td>0.041</td><td>0.043</td><td>0.0246</td><td>0.0214</td></tr><tr><td>LUMO</td><td>eV</td><td>0.034</td><td>0.044</td><td>0.0195</td><td>0.0195</td></tr><tr><td>Gap</td><td>eV</td><td>0.063</td><td>0.066</td><td>0.0326</td><td>0.0381</td></tr><tr><td>ZPVE</td><td>eV</td><td>0.0017</td><td>0.00143</td><td>0.00121</td><td>0.0031</td></tr><tr><td> $\mu$ </td><td>Debye</td><td>0.033</td><td>0.05</td><td>0.0297</td><td>0.0146</td></tr><tr><td> $\alpha$ </td><td>Bohr $^{3}$ </td><td>0.235</td><td>0.081</td><td>0.0435</td><td>0.0561</td></tr><tr><td>R $^{2}$ </td><td>Bohr $^{2}$ </td><td>0.073</td><td>0.302</td><td>0.331</td><td>0.5432</td></tr><tr><td>U0</td><td>eV</td><td>0.014</td><td>0.012</td><td>0.00632</td><td>0.0153</td></tr><tr><td>U</td><td>eV</td><td>0.019</td><td>0.013</td><td>0.00628</td><td>0.0144</td></tr><tr><td>H</td><td>eV</td><td>0.014</td><td>0.012</td><td>0.00653</td><td>0.0147</td></tr><tr><td>G</td><td>eV</td><td>0.014</td><td>0.012</td><td>0.00756</td><td>0.0144</td></tr></table>

These models were trained with same parameters as solid-state databases but for 1000 epochs. Predictions on test set are shown in Supplementary Figs. 42–52.

Next, we evaluate the ALIGNN model on QM9 molecular property dataset (130,829 molecules) and compare it with other well-known models such as SchNet10, MatErials Graph Network (MEGNet)16, and DimeNet++20 as shown in Table 5. The results from models other than ALIGNN are reported as given in corresponding papers, not necessarily reproduced by us. QM9 provides DFT calculated molecular properties such as highest occupied molecular orbital (HOMO), lowest unoccupied molecular orbital (LUMO), energy gap, zero-point vibrational energy (ZPVE), dipole moment, isotropic polarizability, electronic spatial extent, internal energy at 0 K, internal energy at 298 K, enthalpy at 298 K, and Gibbs free energy at 298 K. ALIGNN outperforms competing methods for HOMO and dipole moment tasks while other accuracies are similar to the SchNet model. Most importantly, all ALIGNN results reported here use the same set of hyperparameters obtained by tuning to validation performance on the JARVIS-DFT bandgap target, suggesting that

Effect of changing ALIGNN and GCN layers on machine Table 6.learning models for JARVIS-DFT OptB88vdW formation energy database in ALIGNN models. 

<table><tr><td>Layers</td><td>GCN-0</td><td>GCN-1</td><td>GCN-2</td><td>GCN-3</td><td>GCN-4</td></tr><tr><td>ALIGNN-0</td><td>0.445</td><td>0.065</td><td>0.050</td><td>0.045</td><td>0.044</td></tr><tr><td>ALIGNN-1</td><td>0.064</td><td>0.041</td><td>0.037</td><td>0.036</td><td>0.037</td></tr><tr><td>ALIGNN-2</td><td>0.039</td><td>0.036</td><td>0.034</td><td>0.034</td><td>0.034</td></tr><tr><td>ALIGNN-3</td><td>0.036</td><td>0.034</td><td>0.033</td><td>0.034</td><td>0.034</td></tr><tr><td>ALIGNN-4</td><td>0.034</td><td>0.034</td><td>0.034</td><td>0.034</td><td>0.033</td></tr></table>

The bold values represent the best performing models.

ALIGNN provides robust performance with respect to different datasets and material types.

# Model analysis

We ablate individual components of the ALIGNN model to evaluate their contribution to the overall architecture. Keeping other parameters intact in the ALIGNN model (as specified in Table 1), we vary the number of ALIGNN and GCN layers as shown in Table 6 and Supplementary Table 1 for JARVIS-DFT OptB88vdW formation energies and bandgaps respectively. We find that without any graph convolution layers the MAE for the formation energy and bandgap are 1248.5% and 453.6% higher than the default model. Adding even a single ALIGNN or GCN layer can reduce the MAE by 102.9% illustrating the importance of these layers. However, further increase in ALIGNN/GCN layers doesn’t scale well and performance quickly saturates at a depth of 4. Excluding GCN layers and increasing ALIGNN layers and vice versa show the individual importance of these layers. Performance of GCN-only models saturates at 4 layers with 44 meV/atom MAE on the JARVIS-DFT formation energy task, while ALIGNN-only models saturate at 34 meV(atom)−1 —a relative reduction of 29.14%. Each of these models, along with the other highlighted configurations in Table 6, performs four atom feature updates via graph convolution modules. At least two ALIGNN updates are needed to obtain peak performance. Additional atom feature updates provide little marginal increase in performance. This is consistent with the widely reported difficulty of GCN architectures scaling in depth beyond a few layers52.

Figure 3 shows in detail the tradeoff between the performance benefit of including ALIGNN layers and their computational overhead relative to GCN layers. Per-epoch timing for each configuration is reported in Supplementary Table 2. All GCN-only configurations (annotated with the number of GCN layers) are on the low-computation portion of the pareto frontier, but the highaccuracy portion of the pareto frontier is dominated by ALIGNN/ GCN combinations with at least two ALIGNN updates. The ALIGNN-2/GCN-2 configuration obtains peak performance (again, relative reduction of MAE by 29.14 %) with a computational overhead of roughly 2× relative to the GCN-4 configuration. Supplementary Table 1 and Supplementary Fig. 53 present layer ablation study results yielding similar conclusions on the JARVIS-DFT OptB88vdW band gap target.

This layer ablation study clearly demonstrates that inclusion of bond angle information and propagation of bond and pair features through the node updates improves the generalization ability of atomistic GCN models. This is satisfying from a materials science perspective, as interatomic bonding theory clearly motivates the notion that inclusion of bond angles should improve accuracy of the model.

Similarly, we vary the number of hidden features (i.e., the width of the graph convolution layers), edge input features, and embedding input features to evaluate the MAE performance for

![](images/3963f48f4f4fde42a5f3d7a2dad3b0a372c300a3613be6692fe005c9d250361c.jpg)

<details>
<summary>line</summary>

| Time per epoch (min) | pareto optimal | ALIGNN-only | GCN-only |
| --------------------- | -------------- | ----------- | -------- |
| 0.4                   | -              | -           | 0.065    |
| 0.5                   | -              | 0.065       | 0.050    |
| 0.6                   | -              | -           | 0.045    |
| 0.7                   | -              | -           | 0.044    |
| 0.8                   | -              | -           | -        |
| 0.9                   | -              | -           | -        |
| 1.0                   | 0.037          | 0.039       | -        |
| 1.1                   | 0.035          | 0.036       | -        |
| 1.2                   | 0.034          | 0.035       | -        |
| 1.3                   | 0.033          | 0.034       | -        |
| 1.4                   | 0.032          | 0.033       | -        |
| 1.5                   | 0.031          | 0.032       | -        |
| 1.6                   | 0.031          | 0.031       | -        |
| 1.7                   | 0.031          | 0.031       | -        |
| 1.8                   | 0.031          | 0.031       | -        |
| 1.9                   | 0.031          | 0.031       | -        |
| 2.0                   | 0.031          | 0.031       | -        |
</details>

ALIGNN accuracy-cost ablation study on JARVIS-DFT Fig. 3formation energy target. The red and blue markers represent the number of layers in GCN-only and ALIGNN-only models.

JARVIS-DFT formation energy and bandgap model in comparison with the default model in Table 1. In Supplementary Table 3, we observe that the marginal performance from increasing the hidden features saturates at 256 for both properties. Supplementary Table 4 shows that the number of edge input features is optimal at 80 for formation energy model, while for the bandgap model performance saturates at 40. Similarly, embedding features are optimized at 64 for formation energy while 32 for bandgap model (Supplementary Table 5). Additionally, we tried three different node feature attributes such 1) CFID chemical features (total 438), only atomic number (total 1), and default CGCNN type attributes (total 92) and compared them for formation energy model in Supplementary Table 6. We observe that the default node attributes have the lowest MAE.

Next, we study time taken per epoch of several models for QM9 and JARVIS-DFT formation energy dataset in Supplementary Table 7. To help facilitate fair comparison, we train all models with the same computational resources using the reference implementations and configurations reported in the literature. We note that the timing code for the reference implementations of different methods may include differing amounts of overhead. For example, the ALIGNN timings reported in Supplementary Table 7 amortize the overhead of initial atomistic graph construction across 300 epochs, and each epoch includes the overhead of evaluating the model on the full training and validation sets for performance tracking. Additionally, the computational cost of deep learning models, in general, is not independent of certain hyperparameters; in particular, larger batch sizes can better leverage modern accelerator hardware by exposing more parallelism. We find ALIGNN requires less training time per epoch time compared to other models except DimeNet++ and MEGNet. However, it is important to note that DimeNet++ and other models usually take around 1000 epochs or more to reach desired accuracy, while ALIGNN can converge in about 300 epochs, resulting in lower overall training cost for similar or better accuracy.

While we report timing comparisons using our standard hyperparameter configuration used to train models reported in “Model performance” section, through subsequent model analysis we have identified several strategies that substantially reduce computational workload without incurring a large performance penalty. We observe in Supplementary Fig. 54 that model performance converges after 300 epochs; shorter training budgets incur a modest performance reduction and slightly increased variance with respect to the training split. The performance tradeoff presented in Table 6 and Fig. 3 indicates that switching from the default configuration of 4 layers each of ALIGNN and GCN updates to 2 layers each could offer a speedup of \~1.5× with negligible reduction in accuracy. Finally, we performed a drop-in replacement study comparing batch normalization and layer normalization in Supplementary Table 8, finding that switching to layer normalization provides an additional \~1.7× speedup with a slight degradation in validation loss and negligible degradation in validation MAE. Because the cost of retraining models for all targets reported is still high, and because some of these strategies equally apply to competing models, we defer a more comprehensive performance-cost study to future work.

![](images/3e200a7f855d14754ca3d5cc64081e28d23fa1804b6420b15e5bf6f5dd02fbb7.jpg)

<details>
<summary>scatter</summary>

| N_train | Formation energy MAE eV (atom)⁻¹ |
| ------- | ------------------------------- |
| 1000    | 0.2                             |
| 2000    | 0.15                            |
| 4000    | 0.1                             |
| 8000    | 0.07                            |
| 16000   | 0.05                            |
| 32000   | 0.035                           |
| 64000   | 0.03                            |
</details>

Learning curve for JARVIS-DFT formation energy regres-Fig. 4sion target. Blue markers indicate validation set MAE scores for individual cross-validation iterates. Error bars indicate the mean cross-validation MAE ± one standard error of the mean.

Finally, we simultaneously investigate the effects of dataset size and different train-validation-test splits by performing a learning curve study in cross-validation for the JARVIS-DFT formation energy (Fig. 4 and Supplementary Table 9) and bandgap (Supplementary Fig. 55 and Supplementary Table 9) targets. We perform the cross-validation splitting procedure by merging the standard JARVIS-DFT train and validation sets and randomly sampling without replacement $\mathsf { N } _ { \mathtt { t r a i n } }$ training samples and 5000 validation samples. The learning curve study shows no sign of diminishing marginal returns for additional data up to the full size of the JARVIS-DFT dataset. On the full training set size (44,577) we obtain an average validation MAE of 0.0316 ± 0.0004 eV/at (uncertainty corresponds to the standard error of the mean over five cross-validation (CV) iterates). The standard deviation over CV iterates is 0.0009 eV/at, indicating that model performance is relatively insensitive to the dataset split.

In summary, we have developed an ALIGNN model which uses the line graph neural network that improves the performance of GNN predictions for solids and molecules. We have demonstrated that explicit inclusion of angle-based networks in GNNs can significantly improve model performance. A key contribution of this work is the inclusion and development of both the undirected atomistic graph and its line graph counterpart for solid-state and molecular materials. We develop regression and classification ALIGNN models for some of the well-known preexisting databases and it can be easily applied for other datasets as well. Our model significantly improved accuracies over prior GNN models. We believe the ALIGNN model will rapidly improve the machine learning prediction for several material properties and classes.

# METHODS

# JARVIS-DFT dataset

The JARVIS-DFT34–44 dataset is developed using Vienna Ab-initio simulation package (VASP)53 software (please note commercial software is identified to specify procedures. Such identification does not imply recommendation by National Institute of Standards and Technology (NIST)). Most of the properties are calculated using the OptB88vdW functional48. For a subset of the data we use TBmBJ49 for getting better band gaps. We use density functional perturbation theory (DFPT)54 for predicting piezoelectric and dielectric constants with both electronic and ionic contributions. The linear response theory-based55 frequency based dielectric function was calculated using both OptB88vdW and TBmBJ and the zero-energy values are trained for the machine learning model. Note that the linear response based dielectric constants lack ionic contributions. The TBmBJ frequency dependent dielectric functions are used to calculate the spectroscopic limited maximum efficiency (SLME)38. The magnetic moments are calculated using spin-polarized calculations considering only ferromagnetic initial configurations and neglecting any density functional theory (DFT)+U effects. The thermoelectric coefficients such as Seebeck coefficients and power factors are calculated using BoltzTrap50 software using constant relaxation time approximation. Exfoliation energy for the van der Waals bonded two-dimensional materials are calculated as the energy per atom differences between the bulk and corresponding monolayer counterparts. The spin-orbit spillage40 is calculated using the difference in wavefunctions of a material with and without inclusion of spin-orbit coupling effects. All the JARVIS-DFT data and Classical force-field inspired descriptors (CFID)32 are generated using the JARVIS-Tools package. The CFID baseline models are trained using the LightGBM package (please note commercial software is identified to specify procedures. Such identification does not imply recommendation by National Institute of Standards and Technology (NIST)).56 using the models developed in ref. 32.

# ALIGNN model implementation and training

The ALIGNN model is implemented in PyTorch57 and deep graph library (DGL)33; the training code heavily relies on PyTorch-ignite58. For regression targets we minimize the mean squared error (MSE) loss, and for classification targets we minimize the standard negative log likelihood loss. We train all models for 300 epochs using the AdamW59 optimizer with normalized weight decay of 10−5 and a batch size of 64. The learning rate is scheduled according to the one-cycle policy60 with a maximum learning rate of 0.001. We use the same model configuration for each regression and classification target. We use the initial atom representations from the CGCNN paper, 80 initial bond radial basis function (RBF) features, and 40 initial bond angle RBF features. The atom, bond, and bond angle feature embedding layers produce 64-dimensional inputs to the graph convolution layers. The main body of the network consists of 4 ALIGNN and 4 graph convolution (GCN) layers, each with hidden dimension 256. The final atom representations are reduced by atom-wise average pooling and mapped to regression or classification outputs by a single linear layer. These hyperparameters are selected to optimize validation MAE on the JARVIS-DFT band gap task through a combination of manual hypothesis-driven experiments and random hyperparameter search facilitated and scheduled through Ray Tune61; hyperparameter ranges are given in Supplementary Table 10. The random search results indicate that model performance is most highly sensitive to the learning rate, weight decay, and convolution layer width, and beyond a relatively low threshold is insensitive to the sizes of the initial feature embedding layers.

We used NIST’s Nisaba cluster to train all ALIGNN models, and we reproduce results from the literature using the reference implementations for each competing method on the same hardware. Each model is trained on a single Tesla V100 SXM2 32 gigabyte Graphics processing unit (GPU), with 8 Intel Xeon E5-2698 v4 CPU cores for concurrently fetching and preprocessing batches of data during training (please note commercial software is identified to specify procedures. Such identification does not imply recommendation by National Institute of Standards and Technology (NIST)). For the MP dataset we use a train-validation-test split of 60,000–5000–4239. For the JARVIS-DFT dataset, we use 80%:10%: 10% splits. The 10% test data is never used during training procedures. For QM9 dataset we use a train-validation-test split of 110,000–10,000–10,829.

# DATA AVAILABILITY

All data used in this work is available at Figshare link https://figshare.com/collections/ ALIGNN\_data/5429274. During the training these datasets are accessed using JARVIS-Tools’s figshare module.

# CODE AVAILABILITY

The code and full model and training configurations used in this work are available on GitHub at https://github.com/usnistgov/alignn, along with general tooling at https://github.com/usnistgov/jarvis. An interactive web-app for using ALIGNN models is also made available at https://jarvis.nist.gov/jalignn.

Received: 3 June 2021; Accepted: 2 October 2021;

Published online:15 November 2021

# REFERENCES

1. LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, 436–444 (2015).   
2. Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M. & Monfardini, G. The graph neural network model. IEEE Trans. Neural Netw. 20, 61–80 (2008).   
3. Wu, Z. et al. A comprehensive survey on graph neural networks. IEEE Trans. Neural Netw. Learn. Syst. 32, 4 (2020).   
4. Dwivedi, V. P., Joshi, C. K., Laurent, T., Bengio, Y. & Bresson, X. Benchmarking graph neural networks. arXiv 2003, 00982. Preprint at https://arxiv.org/abs/ 2003.00982 (2020).   
5. Guo, Z. & Wang, H. A deep graph neural network-based mechanism for social recommendations. IEEE Trans. Ind. Inform. 17, 2776 (2020).   
6. Chen, Z., Li, X. & Bruna, J. Supervised community detection with line graph neural networks. arXiv. 1705, 08415. Preprint at https://arxiv.org/abs/1705.08415# (2017).   
7. Li, X. et al. Braingnn: Interpretable brain graph neural network for fmri analysis. Med. Image Anal. 74, 102233 (2021)..   
8. Baumbach, J. CoryneRegNet 4.0–A reference database for corynebacterial gene regulatory networks. BMC Bioinforma. 8, 1–11 (2007).   
9. Wu, K., Chen, Z. & Li, W. A novel intrusion detection model for a massive network using convolutional neural networks. IEEE Access 6, 50850 (2018).   
10. Schütt, K. T. et al. Schnet: a continuous-filter convolutional neural network for modeling quantum interactions. arXiv 1706, 08566. Preprint at https://arxiv.org/ abs/1706.08566 (2017).   
11. Duvenaud, D. et al. Convolutional networks on graphs for learning molecular fingerprints. arXiv 1509, 09292. Preprint at https://arxiv.org/abs/1509.09292 (2015).   
12. Kearnes, S., McCloskey, K., Berndl, M., Pande, V. & Riley, P. Molecular graph convolutions: moving beyond fingerprints. J. Comput. Aided 30, 595–608 (2016).   
13. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl, G. E. Neural message passing for quantum chemistry. PMLR 70, 1263 (2017).   
14. Faber, F. A. et al. Prediction errors of molecular machine learning models lower than hybrid DFT error. J. Chem. Theory Comput. 13, 5255–5264 (2017).   
15. Xie, T. & Grossman, J. C. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Phys. Rev. Lett. 120, 145301 (2018).   
16. Chen, C., Ye, W., Zuo, Y., Zheng, C. & Ong, S. P. Graph networks as a universal machine learning framework for molecules and crystals. Chem. Mater. 31, 3564–3572 (2019).   
17. Park, C. W. & Wolverton, C. Developing an improved crystal graph convolutional neural network framework for accelerated materials discovery. Phys. Rev. Mater. 4, 063801 (2020).   
18. Qiao, Z., Welborn, M., Anandkumar, A., Manby, F. R. & Miller, T. F. III OrbNet: deep learning for quantum chemistry using symmetry-adapted atomic-orbital features. J. Chem. Phys. 153, 124111 (2020).   
19. Klicpera, J., Groß, J. & Günnemann, S. Directional message passing for molecular graphs. arXiv 2003, 03123. Preprint at https://arxiv.org/abs/2003.03123 (2020).   
20. Klicpera, J., Giri, S., Margraf, J. T. & Günnemann, S. Fast and uncertainty-aware directional message passing for non-equilibrium molecules. arXiv 2011, 14115. Preprint at https://arxiv.org/abs/2011.14115 (2020).   
21. Unke, O. T. & Meuwly, M. PhysNet: A neural network for predicting energies, forces, dipole moments, and partial charges. J. Chem. Theory Comput 15, 3678–3693 (2019).   
22. Shui, Z. & George, K. “Heterogeneous molecular graph neural networks for predicting molecule properties”. 2020 IEEE International Conference on Data Mining (ICDM), 492 (2020).   
23. Schütt, K. T., Arbabzadah, F., Chmiela, S., Müller, K. R. & Tkatchenko, A. Quantum-chemical insights from deep tensor neural networks. Nat. Commun. 8, 1–8 (2017).   
24. Anderson, B., Hy, T.-S. & Kondor, R. Cormorant: covariant molecular neural networks. arXiv 1906, 04015. Preprint at https://arxiv.org/abs/1906.04015 (2019).   
25. Zhang, S., Liu, Y. & Xie, L. Molecular mechanics-driven graph neural network with multiplex graph for molecular structures. arXiv 2011, 07457. Preprint at https:// arxiv.org/abs/2011.07457 (2020).   
26. Lubbers, N., Smith, J. S. & Barros, K. Hierarchical modeling of molecular energies using a deep neural network. J. Chem. Phys. 148, 241715 (2018).   
27. Schutt, K. et al. SchNetPack: A deep learning toolbox for atomistic systems. J. Chem. Theory Comput. 15, 448 (2018).   
28. Jha, D. et al. Elemnet: Deep learning the chemistry of materials from only elemental composition. Sci. Rep. 8, 1–13 (2018).   
29. Westermayr, J., Gastegger, M. & Marquetand, P. Combining SchNet and SHARC: The SchNarc machine learning approach for excited-state dynamics. J. Phys. Chem. Lett. 11, 3828 (2020).

30. Wen, M., Blau, S. M., Spotte-Smith, E. W. C., Dwaraknath, S. & Persson, K. A. BonDNet: a graph neural network for the prediction of bond dissociation energies for charged molecules. Chem 12, 1858 (2020).   
31. Isayev, O. et al. Universal fragment descriptors for predicting properties of inorganic crystals. Nat. Commun. 8, 1 (2017).   
32. Choudhary, K., DeCost, B. & Tavazza, F. Machine learning with force-field-inspired descriptors for materials: Fast screening and mapping energy landscape. Phys. Rev. Mater. 2, 083801 (2018).   
33. Wang, M. et al. Deep graph library: a graph-centric, highly-performant package for graph neural networks. arXiv 1909, 01315. Preprit at https://arxiv.org/abs/ 1909.01315 (2019).   
34. Choudhary, K. et al. The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design. Npj Comput. Mater. 6, 1–13 (2020).   
35. Choudhary, K., Cheon, G., Reed, E. & Tavazza, F. Elastic properties of bulk and lowdimensional materials using van der Waals density functional. Phys. Rev. B 98, 014107 (2018).   
36. Choudhary, K., Kalish, I., Beams, R. & Tavazza, F. High-throughput identification and characterization of two-dimensional materials using density functional theory. Sci. Rep. 7, 1–16 (2017).   
37. Choudhary, K. et al. Computational screening of high-performance optoelectronic materials using OptB88vdW and TB-mBJ formalisms. Sci. Data 5, 1–12 (2018).   
38. Choudhary, K. et al. Accelerated discovery of efficient solar cell materials using quantum and machine-learning methods. Chem. Mater. 31, 5900 (2019).   
39. Choudhary, K., Garrity, K. F. & Tavazza, F. High-throughput discovery of topologically non-trivial materials using spin-orbit spillage. Sci. Rep. 9, 1–8 (2019).   
40. Choudhary, K., Garrity, K. F., Ghimire, N. J., Anand, N. & Tavazza, F. Highthroughput search for magnetic topological materials using spin-orbit spillage, machine learning, and experiments. Phys. Rev. B 103, 155131 (2021).   
41. Choudhary, K., Ansari, J. N., Mazin, I. I. & Sauer, K. L. Density functional theorybased electric field gradient database. Sci. Data 7, 1–10 (2020).   
42. Choudhary, K., Garrity, K. F. & Tavazza, F. Data-driven discovery of 3D and 2D thermoelectric materials. J. Condens. Matter Phys. 32, 475501 (2020).   
43. Choudhary, K. et al. High-throughput density functional perturbation theory and machine learning predictions of infrared, piezoelectric, and dielectric responses. Npj Comput. Mater. 6, 1–13 (2020).   
44. Choudhary, K. & Tavazza, F. Convergence and machine learning predictions of Monkhorst-Pack k-points and plane-wave cut-off in high-throughput DFT calculations. Comput. Mater. Sci. 161, 300–308 (2019).   
45. Jain, A. et al. Commentary: The Materials Project: a materials genome approach to accelerating materials innovation. APL Mater. 1, 011002 (2013).   
46. Ramakrishnan, R., Dral, P. O., Rupp, M. & Von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. Sci. Data 1, 1 (2014).   
47. Perdew, J. P., Burke, K. & Ernzerhof, M. Generalized gradient approximation made simple. Phys. Rev. Lett. 77, 3865 (1996).   
48. Klimeš, J., Bowler, D. R. & Michaelides, A. Chemical accuracy for the van der Waals density functional. J. Condens. Matter Phys. 22, 022201 (2009).   
49. Tran, F. & Blaha, P. Accurate band gaps of semiconductors and insulators with a semilocal exchange-correlation potential. Phys. Rev. Lett. 102, 226401 (2009).   
50. Madsen, G. K. & Singh, D. J. BoltzTraP. A code for calculating band-structure dependent quantities. Comput. Phys. Commun. 175, 67–71 (2006).   
51. Ward, L., Agrawal, A., Choudhary, A. & Wolverton, C. A general-purpose machine learning framework for predicting properties of inorganic materials. Npj Comput. Mater. 2, 1 (2016).   
52. Xu, K., Li, C., Tian, Y., Sonobe, T., Kawarabayashi, K. I. & Jegelka, S. Representation learning on graphs with jumping knowledge networks. PMLR 80, 5453 (2018).   
53. Kresse, G. & Furthmüller Efficiency of ab-initio total energy calculations for metals and semiconductors using a plane-wave basis set. Comput. Mater. Sci. 6, 15 (1996).   
54. Baroni, S. & Resta, R. Ab initio calculation of the macroscopic dielectric constant in silicon. Phys. Rev. B 33, 7017 (1986).   
55. Gajdoš, M., Hummer, K., Kresse, G., Furthmüller, J. & Bechstedt, F. Linear optical properties in the projector-augmented wave methodology. Phys. Rev. B 73, 045112 (2006).

56. Ke, G. et al. Lightgbm: A highly efficient gradient boosting decision tree. Adv. Neural Inf. Process. Syst. 30, 3146 (2017).   
57. Paszke, A. et al. Pytorch: an imperative style, high-performance deep learning library. arXiv 1912, 01703. Preprint at https://arxiv.org/abs/1912.01703 (2019).   
58. PyTorch-ignite documentation. https://pytorch.org/ignite/ (2020).   
59. Loshchilov, I. & Hutter, F. Decoupled weight decay regularization. arXiv 1711, 05101. Preprint at https://arxiv.org/abs/1711.05101 (2017).   
60. Smith, L. N. A disciplined approach to neural network hyper-parameters: Part 1-learning rate, batch size, momentum, and weight decay. arXiv 1803, 09820. Preprint at https://arxiv.org/abs/1803.09820 (2018).   
61. Liaw, R., Liang, E., Nishihara, R., Moritz, P., Gonzalez, J. E. & Stoica, I. Tune: a research platform for distributed model selection and training. arXiv 1807, 05118. Preprint at https://arxiv.org/abs/1807.05118 (2018).

# ACKNOWLEDGEMENTS

K.C. and B.D. thank the National Institute of Standards and Technology for funding, computational, and data management resources. Contributions from K.C. were supported by the financial assistance award 70NANB19H117 from the U.S. Department of Commerce, National Institute of Standards and Technology. This work was also supported by the Frontera supercomputer, National Science Foundation OAC-1818253, at the Texas Advanced Computing Center (TACC) at The University of Texas at Austin.

# AUTHOR CONTRIBUTIONS

Both K.C. and B.D. equally contributed to developing the model and writing the manuscript.

# COMPETING INTERESTS

The authors declare no competing interests.

# ADDITIONAL INFORMATION

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41524-021-00650-1.

Correspondence and requests for materials should be addressed to Kamal Choudhary.

Reprints and permission information is available at http://www.nature.com/ reprints

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

![](images/85ca9e69be77c5c968ebe4097ae02c76df1e19afae56410aeed77774d19dff76.jpg)

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing,

adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons. org/licenses/by/4.0/.

This is a U.S. government work and not under copyright protection in the U.S.; foreign copyright protection may apply 2021, corrected publication 2022