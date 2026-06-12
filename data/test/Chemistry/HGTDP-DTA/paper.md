# HGTDP-DTA: Hybrid Graph-Transformer with Dynamic Prompt for Drug-Target Binding Affinity Prediction

Xi Xiao1,⋆, Wentao Wang1,⋆, Jiacheng Xie1, Lijing Zhu3, Gaofei Chen1, Zhengji Li1, Tianyang Wang1, and Min Xu2, ⋆⋆

1 University of Alabama at Birmingham, Birmingham AL 35294, USA   
2 Carnegie Mellon University, Pittsburgh PA 15213, USA   
3 Bowling Green State University, Bowling Green OH 43403, USA

Abstract. Drug target binding affinity (DTA) is a key criterion for drug screening. Existing experimental methods are time-consuming and rely on limited structural and domain information. While learning-based methods can model sequence and structural information, they struggle to integrate contextual data and often lack comprehensive modeling of drug-target interactions. In this study, we propose a novel DTA prediction method, termed HGTDP-DTA, which utilizes dynamic prompts within a hybrid Graph-Transformer framework. Our method generates context-specific prompts for each drug-target pair, enhancing the model’s ability to capture unique interactions. The introduction of prompt tuning further optimizes the prediction process by filtering out irrelevant noise and emphasizing task-relevant information, dynamically adjusting the input features of the molecular graph. The proposed hybrid Graph-Transformer architecture combines structural information from Graph Convolutional Networks (GCNs) with sequence information captured by Transformers, facilitating the interaction between global and local information. Additionally we adopted the multi-view feature fusion method to project molecular graph views and affinity subgraph views into a common feature space, effectively combining structural and contextual information. Experiments on two widely used public datasets, Davis and KIBA, show that HGTDP-DTA outperforms state-of-the-art DTA prediction methods in both prediction performance and generalization ability.

Keywords: drug-target binding affinity · graph convolutional network · transformer · prompt.

# 1 Introduction

In drug development, accurately predicting the binding affinity between a drug molecule and its target protein is a crucial and challenging task [25]. The effectiveness of drug molecules significantly depends on their affinity for target proteins or receptors. However, newly designed drug molecules may sometimes interact with unintended target proteins [27], leading to undesirable side effects. Therefore, it is vital to predict DTA promptly and accurately to ensure the safety and efficacy of new therapeutics.

With the development of high-throughput screening technology, a vast number of molecules that bind to specific targets can be screened more rapidly [7]. However, these techniques frequently require expensive chemicals and equipment. The continued advancement of computational technology has allowed drug-target affinities to be predicted more rapidly and affordably through molecular docking methods [20], which evaluate drug-target interactions using scoring functions [1]. Despite their efficiency, these docking techniques necessitate various biologically significant pre-processing steps, such as hydrogenation and protonation. In recent years, several machine learning-based approaches have emerged. For instance, KronRLS [19] uses Smith-Waterman similarity representations of the targets and compound similarity-based representations of the drugs. SimBoost [6] constructs new features using drug-drug and drug-target similarities. However, these methods often overlook the structural information embedded in the molecules. To extract more relevant information, deep learningbased methods are being increasingly adopted, particularly string-based and graph-based approaches. Methods like DeepDTA [18] employ CNNs to learn feature representations from sequence data. Built upon DeepDTA, AttentionDTA [35] introduces a bilateral multi-headed attention mechanism focusing on key subsequences, which enhances the model’s ability to capture sequence features. Nevertheless, these CNN-based models often fail to capture the characteristics of nearby atoms or amino acids. Consequently, graph-based approaches for DTA prediction have been developed, allowing molecules to be represented as graphs [5]. GraphDTA [17] utilizes GNNs to predict affinities by constructing molecular graphs with atoms as nodes and bonds as edges. DGraphDTA [10] expands this concept by building molecular topological graphs to represent proteins. MGraphDTA [30] constructs ultra-deep GNNs with multiple graph convolution layers to capture both local and global structures of compounds. HSGCL-DTA employs hybrid-scale graph contrastive learning to enhance drug-target binding affinity prediction by integrating multi-scale structural information from both drug and target graphs, leveraging contrastive learning to improve feature representations and prediction accuracy. [31] These approaches demonstrate that GNNs can effectively characterize complex molecular structures.

While the above string and graph-based methods have provided relatively high predictive performance, they often struggle with utilizing limited input data effectively. To address these issues, several fusion-based methods have been developed to integrate additional information. FingerDTA [37] employs molecular fingerprints for drug-target interaction prediction, while MultiscaleDTA [2] utilizes multiscale graph convolutional networks. HGRL-DTA [3] incorporates hierarchical graph representations, and 3DProtDTA [29] leverages 3D protein structures. Additionally, BiComp-DTA [11] and MSF-DTA [15] offer comprehensive graph neural network approaches for predicting drug-target affinities.

However, these fusion-based methods still exhibit certain limitations, such as inadequate integration of structural and contextual information, reliance on static feature representations, and insufficient noise handling capabilities. To overcome these challenges, we propose a novel approach that incorporates dynamic prompt generation into the DTA prediction task. This mechanism enables the model to generate context-specific prompts, enhancing its ability to capture unique interactions between different drugs and targets. By dynamically adjusting to the specific context of each drug-target pair, our model aims to further improve prediction accuracy and adaptability.

Inspired by the success of prompt in computer vision (CV) [9] and natural language processing (NLP) [13,14], we introduce the concept of dynamic prompt generation into the domain of DTA prediction. In CV, prompt-based methods have achieved remarkable success in tasks such as image classification, object detection and segmentation by providing models with context-specific cues that enhance their interpretative capabilities [9]. Similarly, in NLP, prompt techniques have significantly improved performance in tasks like text classification, question answering and machine translation by leveraging prompts to guide the model towards more relevant and accurate responses [13]. The similarity between DTA prediction and these fields lies in the need to capture complex, contextspecific interactions. Dynamic prompt generation methods can be categorized into two categories: context-specific prompt generation and adaptive prompt tuning. Context-specific prompt generation involves creating prompts tailored to the specific drug-target pair, thus capturing unique interactions more effectively [23]. On the other hand, adaptive prompt tuning adjusts the prompts dynamically based on input data characteristics [36], thereby increasing the model’s flexibility and robustness. By maximizing the relevance of generated prompts to the prediction task, dynamic prompt generation can extract task-relevant information while filtering out irrelevant noise. This approach allows the model to better handle the diverse and complex nature of drug-target interactions. Furthermore, we introduce a new strategy where both the molecular graph view and the affinity subgraph view are projected into a common feature space. This unified feature space facilitates the comprehensive learning of information from both views, ensuring a more accurate and robust prediction [33]. Therefore, we propose employing dynamic prompt generation and multi-view feature fusion to address the limitations of existing methods and improve the accuracy and robustness of DTA prediction. The main contributions of this study are summarized as follows:

1) Dynamic Prompt Generation: We introduce dynamic prompt generation into the DTA prediction task for the first time, creating tailored contextspecific prompts for each drug-target pair. This mechanism significantly enhances the model’s ability to capture unique interactions, resulting in more discriminative features and improving the overall prediction accuracy while maintaining global structural information.   
2) Unified Multi-view Feature Fusion: We propose a novel multi-view feature fusion method that projects the molecular graph view and the affinity

subgraph view into a common feature space. This unified feature space effectively combines structural and contextual information, facilitating comprehensive learning and improving the accuracy and robustness of DTA prediction by leveraging the strengths of both views.

3) Hybrid Graph-Transformer with Adaptive Feature Enhancement: We develop a hybrid model that integrates graph convolutional networks (GCNs) to extract structural features from molecular graphs and transformers to capture long-range dependencies in protein sequences. Additionally, we incorporate adaptive feature enhancement, which dynamically adjusts input features to refine the prediction process, ensuring robustness by filtering out irrelevant noise and focusing on critical task-relevant information.

Experimental evaluations conducted on two widely-used public benchmarks across Davis and KIBA demonstrate the effectiveness and superiority of our method compared to the state-of-the-art methods.

# 2 Problem Definition

In the context of drug-target binding affinity (DTA) prediction, the problem is defined as follows [28]: Given a set of drugs D, a set of protein targets $T _ { \mathrm { : } }$ , and a known drug-target affinity matrix $Y \in \mathbf { \mathbb { R } } ^ { | D | \times | T | }$ , where each drug $d _ { i } \in D$ is described by a SMILES string and each target $t _ { j } \in T$ is described by its protein sequence, the objective is to predict the unknown binding affinity $\hat { y } _ { i j }$ between drug $d _ { i }$ and target $t _ { j }$ . This prediction task can be formulated as a regression problem:

$$
\hat {y} _ {i j} = F _ {\theta} (d _ {i}, t _ {j}) \tag {1}
$$

where $\hat { y } _ { i j } \in \mathbb { R }$ denotes the predicted binding affinity, and θ represents the learned parameters of the prediction model F . The goal is to ensure that the predicted affinity $\hat { y } _ { i j }$ closely approximates the true affinity value $y _ { i j }$ .

# 3 Methods

The overall framework of HGTDP-DTA is illustrated in Fig. 1, which consists of six key modules: dynamic prompt generation, adaptive feature enhancement, drug molecule embedding generation, target protein embedding generation, hybrid Graph-Transformer, and binding affinity prediction. HGTDP-DTA integrates various components to predict drug-target binding affinity effectively. Fig. 1 presents three main paths (A, B, C) in our architecture.

# 3.1 Drug Molecule Embedding Generation (Path A)

Drugs are represented as SMILES strings, which are converted into molecular graphs [12]. Each node in the graph represents an atom, and each edge represents a chemical bond. A three-layer Graph Convolutional Network (GCN) is used to obtain the embeddings of the atom nodes. The initial feature matrix $X _ { d _ { i } } ^ { m o l }$ is fed into the GCN to obtain $H _ { d _ { i } } ^ { m o l }$ , where $h _ { d _ { i } , v } ^ { m o l }$ didenotes the potential representation of atomic node $v \in V _ { d _ { i } }$ . The molecular graph embeddings are then combined as z $z _ { d _ { i } } ^ { p r o j }$ e context-specific prompts to form the final drug embeddings (illustratedin Fig. 2).

![](images/b8e541c48f2cb9be928338883b2d6a719018f78b1139db69897714b12292f303.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Drug SMILES"] --> B["Molecular Graph"]
    B --> C["Drug Graph"]
    C --> D["Feature Adjustment"]
    D --> E["Transformer"]
    E --> F["Unified Feature Space"]
    F --> G["Multi-View Fusion"]
    
    subgraph Inputs
        H["SMILES(d_i)"] --> I["X^mol_d_i"]
        J["Low K_d"] --> K["High K_d"]
        L["Target Protein Seq(t_j)"] --> M["Concat Map"]
        N["GCN"] --> O["Molecular Graph"]
    end
    
    subgraph Outputs
        P["z^proj d_i"] --> Q["z^proj_off"]
        R["G^+"] --> S["G^-"]
        T["z^proj t_j"] --> U["H^off + d_i"]
    end
    
    subgraph Legend
        V["Uniified Feature Space"] --> W["Prompt(d_i, t_j)"]
        X["Target Protein Seq(t_j)"] --> Y["Concat Map"]
        Z["GCN"] --> AA["Molecular Graph"]
    end
```
</details>

Fig. 1. Overview of the proposed HGTDP-DTA method.

$$
H _ {d _ {i}} ^ {m o l} = \operatorname{GCN} \left(X _ {d _ {i}} ^ {m o l}\right) \tag {2}
$$

# 3.2 Affinity Graph Embedding Generation (Path B)

Given the known affinity matrix Y , a drug-target affinity graph $G = ( V , E , W )$ can be constructed, where $V = V _ { D } \cup V _ { T }$ denotes the set of nodes of the drug and target, E denotes the set of edges between drugs and targets, and W is the set of corresponding edge weights. For any $v _ { i } \in V _ { D } , v _ { j } \in V _ { T } , e _ { i j } \in E$ denotes the edge between $v _ { i }$ and $v _ { j }$ , and $w _ { i j } \in W$ denotes the affinity value of drug $v _ { i }$ to target $v _ { j }$ .

$$
H _ {d _ {i}} ^ {a f f +} = \Omega^ {+} (G ^ {+}) = \mathrm{ReLU} (\hat {A} ^ {+} \mathrm{ReLU} (\hat {A} ^ {+} X W _ {0}) W _ {1}) \tag {3}
$$

$$
H _ {d _ {i}} ^ {a f f -} = \Omega^ {-} (G ^ {-}) = \mathrm{ReLU} (\hat {A} ^ {-} \mathrm{ReLU} (\hat {A} ^ {-} X W _ {0}) W _ {1}) \tag {4}
$$

where $\hat { A }$ is the regularized adjacency matrix defined as:

$$
\hat {A} = \tilde {D} ^ {- 1 / 2} \tilde {A} \tilde {D} ^ {- 1 / 2}, \quad \tilde {A} = A + I, \quad \tilde {D} = \operatorname{diag} (\sum_ {j} \tilde {A} _ {i j}) \tag {5}
$$

X is the initial feature matrix, F is the dimensionality of the features, $W _ { 0 }$ and $W _ { 1 }$ are the weight parameters of the corresponding layers of GCN. $H ^ { a f f + }$ consists of drug node features $H _ { d _ { i } } ^ { D }$ and target node features $H _ { t _ { j } } ^ { T }$ . After the first layer of GCN encoding, the node can capture information about its one-hop neighbors, i.e., the directly connected heterogeneous nodes. Through the second layer of aggregation, it can obtain information about the same kind of nodes that have the same neighbors as itself.

$$
H _ {d _ {i}} ^ {a f f +} = \operatorname{GCN} (G ^ {+}) \tag {6}
$$

$$
H _ {d _ {i}} ^ {a f f -} = \operatorname{GCN} (G ^ {-}) \tag {7}
$$

# 3.3 Target Protein Embedding Generation (Path C)

The target protein is represented as a sequence of amino acids. We convert the protein sequence into a graph where nodes represent residues and edges represent interactions between residues [10]. Using a similar GCN approach as for drug molecules, we generate the protein embeddings. The initial feature matrix $X _ { t } ^ { p }$ rotj is used to generate Hprott , $H _ { t _ { j } } ^ { p r o t }$ and the embeddings are enhanced with context-specific prompts to capture the unique interactions of the target protein (illustrated as $\bar { z } _ { t _ { j } } ^ { p r o j }$ ztj in Fig. 2).

$$
H _ {t _ {j}} ^ {p r o t} = \operatorname{GCN} (X _ {t _ {j}} ^ {p r o t}) \tag {8}
$$

The target protein is represented as a sequence of amino acids, but it can alternatively be represented as a graph with residues as nodes and contacts of residue pairs as edges. To further investigate the hidden intrinsic structural information in the protein sequence, the sequence of protein $t _ { j }$ is converted into a target graph $G _ { t _ { j } } ^ { m o l } = ( V _ { t _ { j } } ^ { m o l } , E _ { t _ { j } } ^ { m o l } )$ , where $V _ { t _ { j } } ^ { m o l }$ is the set of protein residues and $E _ { t _ { j } } ^ { m o l }$ represents the set of edges between residues. Protein sequence alignment was performed using HHblits [24], and then PconsC4 [16] was used to convert the protein sequence alignment results into a contact map, a residue-residue interaction matrix, whose values represent the Euclidean distance between two residues.

The same encoding is used to generate protein molecule features. After the first layer of GCN, $H _ { t _ { j } } ^ { m o l }$ is obtained, and the target node feature $h _ { t _ { j } } ^ { a f f + }$ of the positive affinity graph are incorporated into each atom feature:

$$
\tilde {h} _ {t _ {j}, v} ^ {m o l} = \left[ h _ {t _ {j}, v} ^ {m o l} \oplus f (h _ {t _ {j}} ^ {a f f +}) \right] \| [ h _ {t _ {j}, v} ^ {m o l} \ominus f (h _ {t _ {j}} ^ {a f f +}) ] \tag {9}
$$

After two more layers of GCN and applying the GMP layer to obtain the final representation of all targets:

$$
Z _ {T} ^ {\text { mol }} = \sum_ {t _ {j} \in T} z _ {t _ {j}} ^ {\text { mol }} \tag {10}
$$

# 3.4 Hybrid Graph-Transformer Architecture with Multi-view Feature Fusion

To effectively combine the information from both the molecular graph view and the affinity subgraph view, we propose a unified feature space where both views are projected. This hybrid architecture integrates GCNs and Transformers to capture both local structural information and long-range dependencies. This is achieved through the following steps:

![](images/bd937f09793692a87873f6c7fdaf42709ede443c676c60927d10e85be74dae9e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Transformer
        A["Feature Adjustment"] --> B["Feature"]
        B --> C["Feature"]
        C --> D["Feature"]
        D --> E["Feature"]
        E --> F["Feature"]
        F --> G["Feature"]
        G --> H["Feature"]
        H --> I["Feature"]
        I --> J["Feature"]
        J --> K["Feature"]
        K --> L["Feature"]
        L --> M["Feature"]
        M --> N["Feature"]
        N --> O["Feature"]
        O --> P["Feature"]
        P --> Q["Feature"]
        Q --> R["Feature"]
        R --> S["Feature"]
        S --> T["Feature"]
        T --> U["Feature"]
        U --> V["Feature"]
        V --> W["Feature"]
        W --> X["Feature"]
        X --> Y["Feature"]
        Y --> Z["Feature"]
        Z --> AA["Feature"]
        AA --> AB["Feature"]
        AB --> AC["Feature"]
        AC --> AD["Feature"]
        AD --> AE["Feature"]
        AE --> AF["Feature"]
        AF --> AG["Feature"]
        AG --> AH["Feature"]
        AH --> AI["Feature"]
        AI --> AJ["Feature"]
        AJ --> AK["Feature"]
        AK --> AL["Feature"]
        AL --> AM["Feature"]
        AM --> AN["Feature"]
        AN --> AO["Feature"]
        AO --> AP["Feature"]
        AP --> AQ["Feature"]
        AQ --> AR["Feature"]
        AR --> AS["Feature"]
        AS --> AT["Feature"]
        AT --> AU["Feature"]
        AU --> AV["Feature"]
        AV --> AW["Feature"]
        AW --> AX["Feature"]
        AX --> AY["Feature"]
        AY --> AZ["Feature"]
        AZ --> BA["Feature"]
        BA --> BB["Feature"]
        BB --> BC["Feature"]
        BC --> BD["Feature"]
        BD --> BE["Feature"]
        BE --> BF["Feature"]
        BF --> BG["Feature"]
        BG --> BH["Feature"]
        BH --> BI["Feature"]
        BI --> BJ["Feature"]
        BJ --> BK["Feature"]
        BK --> BL["Feature"]
        BL --> BM["Feature"]
        BM --> BN["Feature"]
        BN --> BO["Feature"]
        BO --> BP["Feature"]
        BP --> BQ["Feature"]
        BQ --> BR["Feature"]
        BR --> BS["Feature"]
        BS --> BT["Feature"]
        BT --> BU["Feature"]
        BU --> BV["Feature"]
        BV --> BW["Feature"]
        BW --> BX["Feature"]
        BX --> BY["Feature"]
        BY --> BZ["Feature"]
        BZ --> CA["Feature"]
        CA --> CB["Feature"]
        CB --> CC["Feature"]
        CC --> CD["Feature"]
        CD --> CE["Feature"]
        CE --> CF["Feature"]
        CF --> CG["Feature"]
        CG --> CH["Feature"]
        CH --> CI["Feature"]
        CI --> CJ["Feature"]
        CJ --> CK["Feature"]
        CK --> CL["Feature"]
        CL --> CM["Feature"]
        CM --> CN["Feature"]
        CN --> CO["Feature"]
        CO --> CP["Feature"]
        CP --> CQ["Feature"]
        CQ --> CR["Feature"]
        CR --> CS["Feature"]
        CS --> CT["Feature"]
        CT --> CU["Feature"]
        CU --> CV["Feature"]
        CV --> CW["Feature"]
        CW --> CX["Feature"]
        CX --> CY["Feature"]
        CY --> CZ["Feature"]
    end

    subgraph Transformer
        D1["Dynamic Prompt Generate"] --> D2["Dynamic Prompt Generate"] & D3["Dynamic Prompt Generate"] & D4["Dynamic Prompt Generate"] & D5["Dynamic Prompt Generate"] & D6["Dynamic Prompt Generate"] & D7["Dynamic Prompt Generate"] & D8["Dynamic Prompt Generate"] & D9["Dynamic Prompt Generate"] & D10["Dynamic Prompt Generate"] & D11["Dynamic Prompt Generate"] & D12["Dynamic Prompt Generate"] & D13["Dynamic Prompt Generate"] & D14["Dynamic Prompt Generate"] & D15["Dynamic Prompt Generate"] & D16["Dynamic Prompt Generate"] & D17["Dynamic Prompt Generate"] & D18["Dynamic Prompt Generate"] & D19["Dynamic Prompt Generate"] & D20["Dynamic Prompt Generate"] & D21["Dynamic Prompt Generate"] & D22["Dynamic Prompt Generate"] & D23["Dynamic Prompt Generate"] & D24["Dynamic Prompt Generate"] & D25["Dynamic Prompt Generate"] & D26["Dynamic Prompt Generate"] & D27["Dynamic Prompt Generate"] & D28["Dynamic Prompt Generate"] & D29["Dynamic Prompt Generate"] & D30["Dynamic Prompt Generate"] & D31["Dynamic Prompt Generate"] & D32["Dynamic Prompt Generate"] & D33["Dynamic Prompt Generate"] & D34["Dynamic Prompt Generate"] & D35["Dynamic Prompt Generate"] & D36["Dynamic Prompt Generate"] & D37["Dynamic Prompt Generate"] & D38["Dynamic Prompt Generate"] & D39["Dynamic Prompt Generate"] & D40["Dynamic Prompt Generate"] & D41["Dynamic Prompt Generate"] & D42["Dynamic Prompt Generate"] & D43["Dynamic Prompt Generate"] & D44["Dynamic Prompt Generate"] & D45["Dynamic Prompt Generate"] & D46["Dynamic Prompt Generate"] & D47["Dynamic Prompt Generate"] & D48["Dynamic Prompt Generate"] & D49["Dynamic Prompt Generate"] & D50["Dynamic Prompt Generate"] & D51["Dynamic Prompt Generate"] & D52["Dynamic Prompt Generate"] & D53["Dynamic Prompt Generate"] & D54["Dynamic Prompt Generate"] & D55["Dynamic Prompt Generate"] & D56["Dynamic Prompt Generate"] & D57["Dynamic Prompt Generate"] & D58["Dynamic Prompt Generate"] & D59["Dynamic Prompt Generate"] & D60["Dynamic Prompt Generate"] & D61["Dynamic Prompt Generate"] & D62["Dynamic Prompt Generate"] & D63["Dynamic Prompt Generate"] & D64["Dynamic Prompt Generate"] & D65["Dynamic Prompt Generate"] & D66["Dynamic Prompt Generate"] & D67["Dynamic Prompt Generate"] & D68["Dynamic Prompt Generate"] & D69["Dynamic Prompt Generate"] & D70["Dynamic Prompt Generate"] & D71["Dynamic Prompt Generate"] & D72["Dynamic Prompt Generate"] & D73["Dynamic Prompt Generate"] & D74["Dynamic Prompt Generate"] & D75["Dynamic Prompt Generate"] & D76["Dynamic Prompt Generate"] & D77["Dynamic Prompt Generate"] & D78["Dynamic Prompt Generate"] & D79["Dynamic Prompt Generate"] & D80["Dynamic Prompt Generate"] & D81["Dynamic Prompt Generate"] & D82["Dynamic Prompt Generate"] & D83["Dynamic Prompt Generate"] & D84["Dynamic Prompt Generate"] & D85["Dynamic Prompt Generate"] & D86["Dynamic Prompt Generate"] & D87["Dynamic Prompt Generate"] & D88["Dynamic Prompt Generate"] & D89["Dynamic Prompt Generate"] & D90["Dynamic Prompt Generate"] & D91["Dynamic Prompt Generate"] & D92["Dynamic Prompt Generate"] & D93["Dynamic Prompt Generate"] & D94["Dynamic Prompt Generate"] & D95["Dynamic Prompt Generate"] & D96["Dynamic Prompt Generate"] & D97["Dynamic Prompt Generate"] & D98["Dynamic Prompt Generate"] & D99["Dynamic Prompt Generate"] & D100["Dynamic Prompt Generate"]

    subgraph Transformer
    F["Fθ"] --> G1["Hφ^ref = GCN(Xφ^ref)"]
    F1["Hφ^ref = GCN(Xφ^ref)"]
    F2["Hφ^ref = GCN(Xφ^ref)"]
    F3["Hφ^ref = GCN(Xφ^ref)"]
    F4["Hφ^ref = GCN(Xφ^ref)"]
    F5["Hφ^ref = GCN(Xφ^ref)"]
    F6["Hφ^ref = GCN(Xφ^ref)"]
    F7["Hφ^ref = GCN(Xφ^ref)"]
    F8["Hφ^ref = GCN(Xφ^ref)"]
    F9["Hφ^ref = GCN(Xφ^ref)"]
    F10["Hφ^ref = GCN(Xφ^ref)"]
    F11["Hφ^ref = GCN(Xφ^ref)"]
    F12["Hφ^ref = GCN(Xφ^ref)"]
    F13["Hφ^ref = GCN(Xφ^ref)"]
    F14["Hφ^ref = GCN(Xφ^ref)"]
    F15["Hφ^ref = GCN(Xφ^ref)"]
    F16["Hφ^ref = GCN(Xφ^ref)"]
    F17["Hφ^ref = GCN(Xφ^ref)"]
    F18["Hφ^ref = GCN(Xφ^ref)"]
    F19["Hφ^ref = GCN(Xφ^ref)"]
    F20["Hφ^ref = GCN(Xφ^ref)"]
    F21["Hφ^ref = GCN(Xφ^ref)"]
    F22["Hφ^ref = GCN(Xφ^ref)"]
    F23["Hφ^ref = GCN(Xφ^ref)"]
    F24["Hφ^ref = GCN(Xφ^ref)"]
    F25["Hφ^ref = GCN(Xφ^ref)"]
    F26["Hφ^ref = GCN(Xφ^ref)"]
    F27["Hφ^ref = GCN(Xφ^ref)"]
    F28["Hφ^ref = GCN(Xφ^ref)"]
    F29["Hφ^ref = GCN(Xφ^ref)"]
    F30["Hφ^ref = GCN(Xφ^ref)"]
    F31["Hφ^ref = GCN(Xφ^ref)"]
    F32["Hφ^ref = GCN(Xφ^ref)"]
    F33["Hφ^ref = GCN(Xφ^ref)"]
    F34["Hφ^ref = GCN(Xφ^ref)"]
    F35["Hφ^ref = GCN(Xφ^ref)"]
    F36["Hφ^ref = GCN(Xφ^ref)"]
    F37["Hφ^ref = GCN(Xφ^ref)"]
    F38["Hφ^ref = GCN(Xφ^ref)"]
    F39["Hφ^ref = GCN(Xφ^ref)"]
    F40["Hφ^ref = GCN(Xφ^ref)"]
    F41["Hφ^ref = GCN(Xφ^ref)"]
    F42["Hφ^ref = GCN(Xφ^ref)"]
    F43["Hφ^ref = GCN(Xφ^ref)"]
    F44["Hφ^ref = GCN(Xφ^ref)"]
    F45["Hφ^ref = GCN(Xφ^ref)"]
    F46["Hφ^ref = GCN(Xφ^ref)"]
    F47["Hφ^ref = GCN(Xφ^ref)"]
    F48["Hφ^ref = GCN(Xφ^ref)"]
    F49["Hφ^ref = GCN(Xφ^ref)"]
    F50["Hφ^ref = GCN(Xφ^ref)"]
    F51["Hφ^ref = GCN(Xφ^ref)"]
    F52["Hφ^ref = GCN(Xφ^ref)"]
    F53["Hφ^ref = GCN(Xφ^ref)"]
    F54["Hφ^ref = GCN(Xφ^ref)"]
    F55["Hφ^ref = GCN(Xφ^ref)"]
    F56["Hφ^ref = GCN(Xφ^ref)"]
    F57["Hφ^ref = GCN(Xφ^ref)"]
    F58["Hφ^ref = GCN(Xφ^ref)"]
    F59["Hφ^ref = GCN(Xφ^ref)"]
    F60["Hφ^ref = GCN(Xφ^ref)"]
    F61["Hφ^ref = GCN(Xφ^ref)"]
    F62["Hφ^ref = GCN(Xφ^ref)"]
    F63["Hφ^ref = GCN(Xφ^ref)"]
    F64["Hφ^ref = GCN(Xφ^ref)"]
    F65["Hφ^ref = GCN(Xφ^ref)"]
    F66["Hφ^ref = GCN(Xφ^ref)"]
    F67["Hφ^ref = GCN(Xφ^ref)"]
    F68["Hφ^ref = GCN(Xφ^ref)"]
    F69["Hφ^ref = GCN(Xφ^ref)"]
    F70["Hφ^ref = GCN(Xφ^ref)"]
    F71["Hφ^ref = GCN(Xφ^ref)"]
    F72["Hφ^ref = GCN(Xφ^ref)"]
    F73["Hφ^ref = GCN(Xφ^ref)"]
    F74["Hφ^ref = GCN(Xφ^ref)"]
    F75["Hφ^ref = GCN(Xφ^ref)"]
    F76["Hφ^ref = GCN(Xφ^ref)"]
    F77["Hφ^ref = GCN(Xφ^ref)"]
    F78["Hφ^ref = GCN(Xφ^ref)"]
    F79["Hφ^ref = GCN(Xφ^ref)"]
    F80["Hφ^ref = GCN(Xφ^ref)"]
    F81["Hφ^ref = GCN(Xφ^ref)"]
    F82["Hφ^ref = GCN(Xφ^ref)"]
    F83["Hφ^ref = GCN(Xφ^ref)"]
    F84["Hφ^ref = GCN(Xφ^ref)"]
    F85["Hφ^ref = GCN(Xφ^ref)"]
    F86["Hφ^ref = GCN(Xφ^ref)"]
    F87["Hφ^ref = GCN(Xφ^ref)"]
    F88["Hφ^ref = GCN(Xφ^ref)"]
    F89["Hφ^ref = GCN(Xφ^ref)"]
    F90["Hφ^ref = GCN(Xφ^ref)"]

    subgraph Multi-View Fusion
    G1["Pd_i = f_prompt + p_d_i + 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000<br>    G2[Pd_ji = fprompt + p_d_ji + 0.0000000000000000000000000000000000000000000000000000<br>    G3[Pd_iii = fprompt + p_d_iii + 0.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_II[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i"][i["i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i[i Butter i["]n_i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[] i[]
    subgraph Multi-View Fusion
    G2["Pd_i == fprompt + p_d_ji + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii - p_d_ji == fprompt + p_d_ji + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.x_iii + 1.x_iii + 1.x_iii + 1.x_iii - p_d_ji == fprompt + p_d_ji + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.x_iii + 1.x_iii + 1.x_iii - p_d_ji == fprompt + p_d_ji + 1.1x_iii + 1.1x_iii + 1.1x_iii + 1.x_iii + 1.x_II(i[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j[j"][j["kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[Kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj[kj"][kij["x_i"][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij][k_ij}[k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij][k_ij["x_n"][k_ij}[kij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][k_ij][k_ij["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dik["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dikk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][dkk["x_i"][ddkx[X_k,i,j,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k;l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,l,k,r,k,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,G,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,K,L,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,P,D,E,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,g,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,H,F,G,\text{and} \text{if }P\text{ is a duplicate of the other components in the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of the output of all these functions.
End
```
</details>

Fig. 2. Multi-view Feature Fusion with Prompt Integration in HGTDP-DTA. This module combines features from the molecular graph, affinity graph, and protein graph into a unified feature space, integrating context-specific prompts to enhance prediction accuracy.

Feature Projection: For each drug $d _ { i }$ and target $t _ { j } ,$ , we obtain their respective embeddings from the molecular graph view $( H _ { d _ { i } } ^ { m o l } , ~ H _ { t _ { j } } ^ { m o l } )$ and the affinity subgraph view $( H _ { d _ { i } } ^ { a f f + } , \ H _ { t _ { j } } ^ { a f f + } , \ H _ { d _ { i } } ^ { a f f - } , \ H _ { t _ { j } } ^ { a f f - } )$ i j. These embeddings projected into a common feature space using learned projection matrices $W _ { d }$ and $W _ { t } { : }$ :

$$
z _ {d _ {i}} ^ {\text {   proj }} = W _ {d} [ H _ {d _ {i}} ^ {\text {   mol }} \| H _ {d _ {i}} ^ {\text {   aff+ }} \| H _ {d _ {i}} ^ {\text {   aff- }} ] \tag {11}
$$

$$
z _ {t _ {j}} ^ {\text { proj }} = W _ {t} \left[ H _ {t _ {j}} ^ {\text { prot }} \| H _ {t _ {j}} ^ {\text { aff+ }} \| H _ {t _ {j}} ^ {\text { aff- }} \right] \tag {12}
$$

Dynamic Prompt Generation: The next step is to generate dynamic prompts based on the projected features. These prompts are designed to capture the specific context of the drug-target interaction, including drug, target, and affinity subgraphs. The process involves a trained prompt generator, which creates context-specific prompts to enhance the feature representations:

$$
p _ {d _ {i}} = f _ {p r o m p t} (z _ {d _ {i}} ^ {p r o j}) \tag {13}
$$

$$
p _ {t _ {j}} = f _ {\text { prompt }} \left(z _ {t _ {j}} ^ {\text { proj }}\right) \tag {14}
$$

$$
p _ {a f f} = f _ {p r o m p t} (z _ {d _ {i}} ^ {p r o j}, z _ {t _ {j}} ^ {p r o j}) \tag {15}
$$

Prompt Integration: The final integrated features for the drug, target, and affinity are obtained by combining the projected features with the generated prompts. This integration helps to better capture the unique interactions between the drug and target:

$$
z _ {d _ {i}} ^ {\text { final }} = z _ {d _ {i}} ^ {\text { proj }} + p _ {d _ {i}} \tag {16}
$$

$$
z _ {t _ {j}} ^ {\text { final }} = z _ {t _ {j}} ^ {\text { proj }} + p _ {t _ {j}} \tag {17}
$$

$$
z _ {a f f} ^ {f i n a l} = z _ {d _ {i}} ^ {p r o j} + z _ {t _ {j}} ^ {p r o j} + p _ {a f f} \tag {18}
$$

The detailed process is illustrated in Fig. 2.

# 3.5 Drug-target Binding Affinity Prediction

The final integrated embeddings for the drug, target, and affinity are concatenated and fed into a multi-layer perceptron (MLP) to predict the binding affinity score $\hat { y } _ { i j }$ :

$$
\hat {y} i j = M L P (z ^ {\text { final }} \text { fusion }) \tag {19}
$$

The objective is to minimize the mean squared error (MSE) loss between the predicted affinity and the true affinity:

$$
L _ {M S E} = \frac {1}{n} \sum_ {i = 1} ^ {n} (\hat {y} i j - y i j) ^ {2} \tag {20}
$$

The total loss function combines the MSE loss with the prompt integration loss:

$$
L = L _ {M S E} + \alpha L _ {p r o m p t} \tag {21}
$$

where α is a hyperparameter to balance the contributions of the different loss components.

# 4 Experiments

# 4.1 Datasets

Davis: The Davis [4] dataset comprises 68 distinct drugs and 442 unique targets, resulting in a total of 30,056 drug-target interactions quantified by $K _ { d }$ (kinase dissociation constant) values. Following previous studies [4] [19], the $K _ { d }$ values were transformed into logarithmic space as follows:

$$
p K _ {d} = - \log_ {1 0} (K _ {d} / 1 0 ^ {9}) \tag {22}
$$

The transformed affinities range from 5.0 to 10.8. For this study, affinities equal to or below 5.0 were considered as negative interactions, indicating very weak or undetectable binding. The drug molecules in this dataset are represented by SMILES strings sourced from the PubChem compound database, and the protein sequences for targets are derived from the UniProt database using gene names and RefSeq accession numbers. The dataset was divided into a training set of 25,746 interactions and a test set of 5,010 interactions.

KIBA: The KIBA [26] dataset integrates kinase inhibitor bioactivities from various sources, including $K _ { i } , K _ { d } ,$ and IC50 values, to form a comprehensive measure known as the KIBA score. Originally, the dataset contains 52,498 drugs and 467 targets with a total of 246,088 interaction records. For this study, the dataset was filtered to include 2,111 drugs and 229 targets, resulting in 118,254 drug-target pairs with at least 10 interactions each [6]. The affinities in this dataset range from 0.0 to 17.2, with NaN values indicating missing experimental data. SMILES strings for the drugs were obtained from PubChem, and protein sequences were retrieved from UniProt using corresponding identifiers. The dataset was divided into a training set of 98,545 interactions and a test set of 19,709 interactions.

# 4.2 Implementation Details

All experiments are implemented based on PyTorch. Our models are trained on a single Nvidia A100 GPU. The same model configurations are utilized on two datasets. We utilized the Adam optimizer to train the entire framework for 2,000 epochs. The batch size is set to 512. The learning rate was set to 5e-4. The dimension of the embedding was set to 128. For the Davis dataset, the threshold p was set to 6, and for the KIBA dataset, it was set to 11. The hyperparameters α and $\beta$ were both set to 0.2, while τ was set to 0.5.

# 4.3 Comparison with State-of-the-art Methods

We adopt a task-specific paradigm in terms of evaluation metrics in each experiment. Specifically, these metrics include the Mean Squared Error (MSE), Concordance Index (CI), $r _ { m } ^ { 2 }$ , and Pearson’s Correlation (Pearson). To ensure an unprejudiced comparison and demonstrate the effectiveness of our method, we contrast HGTDP-DTA against machine learning, sequence, and graph-based methods, along with the fusion-based methods. To make fair comparisons, we obtained the results with convincing parameters for these methods. We report average results using five different random seeds in Table 1 and Table 2.

Table 1. Comparison results to previous state-of-the-art methods on the Davis dataset. Our results are highlighted in BOLD. The best results are highlighted in BLUE, while the second-best results are marked in RED. 

<table><tr><td>Type</td><td>Method</td><td>MSE ↓</td><td>CI ↑</td><td> $r_{m}^{2}$  ↑</td><td>Pearson ↑</td></tr><tr><td rowspan="2">ML-based</td><td>KronRLS [4] [NAT BIOTECHNOL&#x27;11]</td><td>0.379</td><td>0.871</td><td>0.407</td><td>-</td></tr><tr><td>SimBoost [6] [CHEMINFORMATICS&#x27;17]</td><td>0.282</td><td>0.872</td><td>0.644</td><td>-</td></tr><tr><td rowspan="4">SQ-based</td><td>DeepDTA [18] [BIOINFORMATICS&#x27;18]</td><td>0.261</td><td>0.878</td><td>0.630</td><td>-</td></tr><tr><td>GANsDTA [34] [FRONT GENET&#x27;20]</td><td>0.276</td><td>0.881</td><td>-</td><td>-</td></tr><tr><td>MRBDTA [32] [BRIEF BIONINFORM&#x27;22]</td><td>0.216</td><td>0.901</td><td>0.716</td><td>-</td></tr><tr><td>MFR-DTA [8] [BIOINFORMATICS&#x27;23]</td><td>0.221</td><td>0.905</td><td>0.705</td><td>-</td></tr><tr><td rowspan="6">GR-based</td><td>GraphDTA [17] [BIOINFORMATICS&#x27;21]</td><td>0.229</td><td>0.893</td><td>-</td><td>-</td></tr><tr><td>DGraphDTA [10] [RSC ADV&#x27;20]</td><td>0.203</td><td>0.904</td><td>0.867</td><td>-</td></tr><tr><td>DeepFusionDTA [21] [ACM TCBI&#x27;21]</td><td>0.253</td><td>0.887</td><td>-</td><td>-</td></tr><tr><td>MGraphDTA [30] [CHEM SCI&#x27;22]</td><td>0.207</td><td>0.900</td><td>0.710</td><td>-</td></tr><tr><td>DoubleSG-DTA [22] [PHARMACEUTICS&#x27;23]</td><td>0.219</td><td>0.902</td><td>0.725</td><td>-</td></tr><tr><td>HSGCL-DTA [31] [ICTAI&#x27;23]</td><td>0.155</td><td>0.911</td><td>0.767</td><td>0.899</td></tr><tr><td rowspan="6">FU-based</td><td>FingerDTA [37] [BDMA&#x27;22]</td><td>0.234</td><td>0.895</td><td>0.678</td><td>-</td></tr><tr><td>MultiscaleDTA [2] [METHODS&#x27;22]</td><td>0.200</td><td>0.889</td><td>0.738</td><td>-</td></tr><tr><td>HGRL-DTA [3] [INFORM SCIENCES&#x27;22]</td><td>0.166</td><td>0.911</td><td>0.751</td><td>0.892</td></tr><tr><td>3DProtDTA [29] [RSC ADV&#x27;23]</td><td>0.184</td><td>0.917</td><td>0.722</td><td>-</td></tr><tr><td>BiComp-DTA [11] [PLOS COMPUT BIOL&#x27;23]</td><td>0.237</td><td>0.904</td><td>0.696</td><td>-</td></tr><tr><td>MSF-DTA [15] [JBHI&#x27;23]</td><td>0.194</td><td>0.906</td><td>-</td><td>-</td></tr><tr><td></td><td>HGTDP-DTA [Ours]</td><td>0.142</td><td>0.910</td><td>0.809</td><td>0.905</td></tr></table>

1) Results on Davis Dataset: The quantitative results on the Davis dataset are shown in Table 1. HGTDP-DTA outperforms machine learning-based (ML-based) methods by a large margin, which have limitations in feature extraction and modeling capabilities. The sequence-based (SQ-based) approaches utilize the sequence information of drugs or targets, which could be insufficient in fully capturing the spatial and topological relationships within the molecular structure. Although graph-based (GR-based) methods can capture the topological structure of molecules, they may overlook important features present in the sequence information. Compared to sequence and graph-based methods, our fusion-based (FU-based) HGTDP-DTA leverages both sequence and graph information and shows superior learning ability on almost all evaluation metrics, respectively. Moreover, the success of dynamic prompt generation, unified multi-view feature fusion, and adaptive feature enhancement let HGTDP-DTA overperform other fusion-based methods and achieve a new state-of-the-art.

Table 2. Comparison results to previous state-of-the-art methods on the KIBA dataset. Our results are highlighted in BOLD. The best results are highlighted in BLUE, while the second-best results are marked in RED. 

<table><tr><td>Type</td><td>Method</td><td>MSE ↓</td><td>CI ↑</td><td> $r_{m}^{2}$  ↑</td><td>Pearson ↑</td></tr><tr><td rowspan="2">ML-based</td><td>KronRLS [4] [NAT BIOTECHNOL&#x27;11]</td><td>0.411</td><td>0.782</td><td>0.342</td><td>-</td></tr><tr><td>SimBoost [6] [CHEMINFORMATICS&#x27;17]</td><td>0.222</td><td>0.836</td><td>0.629</td><td>-</td></tr><tr><td rowspan="4">SQ-based</td><td>DeepDTA [18] [BIOINFORMATICS&#x27;18]</td><td>0.194</td><td>0.863</td><td>0.673</td><td>-</td></tr><tr><td>GANsDTA [34] [FRONT GENET&#x27;20]</td><td>0.192</td><td>0.866</td><td>0.756</td><td>-</td></tr><tr><td>MRBDTA [32] [BRIEF BIONINFORM&#x27;22]</td><td>0.146</td><td>0.892</td><td>0.778</td><td>-</td></tr><tr><td>MFR-DTA [8] [BIOINFORMATICS&#x27;23]</td><td>0.136</td><td>0.898</td><td>0.789</td><td>-</td></tr><tr><td rowspan="6">GR-based</td><td>GraphDTA [17] [BIOINFORMATICS&#x27;21]</td><td>0.139</td><td>0.891</td><td>-</td><td>-</td></tr><tr><td>DGraphDTA [10] [RSC ADV&#x27;20]</td><td>0.126</td><td>0.904</td><td>0.903</td><td>-</td></tr><tr><td>DeepFusionDTA [21] [ACM TCBI&#x27;21]</td><td>0.162</td><td>0.887</td><td>-</td><td>-</td></tr><tr><td>MGraphDTA [30] [CHEM SCI&#x27;22]</td><td>0.128</td><td>0.902</td><td>0.801</td><td>-</td></tr><tr><td>DoubleSG-DTA [22] [PHARMACEUTICS&#x27;23]</td><td>0.167</td><td>0.896</td><td>0.787</td><td>-</td></tr><tr><td>HSGCL-DTA [31] [ICTAI&#x27;23]</td><td>0.124</td><td>0.905</td><td>0.803</td><td>0.907</td></tr><tr><td rowspan="7">FU-based</td><td>FingerDTA [37] [BDMA&#x27;22]</td><td>0.150</td><td>0.885</td><td>0.750</td><td>-</td></tr><tr><td>MultiscaleDTA [2] [METHODS&#x27;22]</td><td>0.135</td><td>0.893</td><td>0.793</td><td>-</td></tr><tr><td>HGRL-DTA [3] [INFORM SCIENCES&#x27;22]</td><td>0.166</td><td>0.899</td><td>0.789</td><td>0.907</td></tr><tr><td>3DProtDTA [29] [RSC ADV&#x27;23]</td><td>0.183</td><td>0.893</td><td>0.784</td><td>-</td></tr><tr><td>BiComp-DTA [11] [PLOS COMPUT BIOL&#x27;23]</td><td>0.163</td><td>0.891</td><td>0.757</td><td>-</td></tr><tr><td>MSF-DTA [15][JBHI&#x27;23]</td><td>0.124</td><td>0.899</td><td>-</td><td>-</td></tr><tr><td>HGTDP-DTA [Ours]</td><td>0.119</td><td>0.912</td><td>0.815</td><td>0.913</td></tr></table>

2) Results on KIBA Dataset: To further prove generalizing performances, our model is evaluated on the KIBA dataset, the results are shown in Table 2. Compared to state-of-the-art methods, our HGTDP-DTA achieves the best performances with 0.119 MSE, 0.912 CI, and 0.913 Pearson, and second-best performance with 0.815 $r _ { m } ^ { 2 }$ . Thus, it reveals that our fusion-based HGTDP-DTA is more advantageous to process drug-target affinity data compared to machine learning, sequence, and graph-based models, and consistently outperforms other fusion-based state-of-the-art methods.

# 4.4 Ablation Study

To further explore the contribution of the components in our HGTDP-DTA model, we performed a series of ablation experiments. We repeated the experiment five times on the Davis and KIBA datasets to average the results. The results in Table 3 show that the performance of each variant produces increased performance compared to the basic model. The dynamic prompt generation module has the largest influence on the model performance, suggesting that context-specific prompts are crucial for capturing unique interactions between drug-target pairs. The hybrid Graph-Transformer architecture also contributes substantially to the model’s accuracy, indicating the importance of combining structural and sequential data.

Table 3. Ablation studies were conducted on each component using the Davis and KIBA datasets. The components evaluated include Dynamic Prompt Generation (DP), Graph Convolutional Networks (GCN), and Transformer (Trans). The best results are highlighted in bold. 

<table><tr><td colspan="3">Module Settings</td><td colspan="2">MSE ↓</td><td colspan="2">CI ↑</td><td colspan="2"> $r_{m}^{2}$  ↑</td><td colspan="2">Pearson ↑</td></tr><tr><td>DP</td><td>GCN</td><td>Trans</td><td>Davis</td><td>KIBA</td><td>Davis</td><td>KIBA</td><td>Davis</td><td>KIBA</td><td>Davis</td><td>KIBA</td></tr><tr><td></td><td>√</td><td></td><td>0.180</td><td>0.140</td><td>0.905</td><td>0.895</td><td>0.725</td><td>0.760</td><td>0.875</td><td>0.895</td></tr><tr><td>√</td><td>√</td><td></td><td>0.161</td><td>0.127</td><td>0.907</td><td>0.904</td><td>0.743</td><td>0.785</td><td>0.897</td><td>0.906</td></tr><tr><td></td><td>√</td><td>√</td><td>0.165</td><td>0.129</td><td>0.909</td><td>0.907</td><td>0.748</td><td>0.788</td><td>0.897</td><td>0.906</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.142</td><td>0.119</td><td>0.910</td><td>0.912</td><td>0.809</td><td>0.815</td><td>0.905</td><td>0.913</td></tr></table>

# 4.5 Visualization Analysis

![](images/cb60644ce516ca9c5ece5bf4e152e50771816de36fc9b845f594d7b3adfab4e4.jpg)  
Fig. 3. Visualization of affinity representations. Red: weak affinity. Blue: strong affinity.

In this subsection, we design an additional experiment to explore the representation power of the proposed model from the view of affinity representations.We divide affinities into two clusters using predefined thresholds from previous studies [4,6]: a pKd value of 7 for the Davis dataset and a KIBA score of 12.1 for the KIBA dataset. Affinities below these thresholds are classified as weak, while those above them are classified as strong. This division is applied to the test sets of both benchmark datasets. We use the trained HGTDP-DTA model to extract learned representations of testing samples before the final prediction layer. This analysis assumes that affinities should cluster closely together for similar interactions and separate distinctly for different ones.Fig. 3 shows the t-SNE visualization of affinity representations for the Davis and KIBA datasets generated by the HGTDP-DTA model. Red points indicate weak affinities, while blue points represent strong affinities. The t-SNE plots reveal a clear mix of strong and weak affinity representations in both datasets, highlighting the model’s ability to capture nuanced drug-target interactions.Tables 1 and 2 report the clustering performance of affinity representations for our model and baselines on the two datasets. Our HGTDP-DTA model achieves competitive performance compared to baseline methods. To visualize the affinity representations intuitively, we sample weak and strong affinities in a 1:1 ratio from the KIBA dataset test set and project them into 2D space using t-SNE. As shown in Fig. 4, HGTDP-DTA can effectively distinguish weak affinities (red) from strong ones (blue), indicating that HGTDP-DTA provides more detailed affinity representations, leading to better binding affinity prediction performance.

Overall, the t-SNE visualizations demonstrate that the HGTDP-DTA model effectively learns and distinguishes features of strong and weak affinities. The clear mix and distribution patterns validate the model’s ability to accurately predict affinities and capture unique drug-target interactions.

# 5 Conclusion

In this study, we proposed HGTDP-DTA, a novel model for drug-target binding affinity (DTA) prediction. This model integrates dynamic prompts within a hybrid Graph-Transformer framework, combining structural information of molecules with interaction data from drug-target pairs. By capturing features from multiple perspectives, HGTDP-DTA enhances prediction accuracy and robustness. It utilizes Graph Convolutional Networks (GCNs) for processing molecular graphs and Transformers for sequence information, facilitating feature fusion. Dynamic prompts improve the model’s ability to capture unique interactions by generating context-specific prompts for each drug-target pair. Our experimental results on benchmark datasets, Davis and KIBA, demonstrate that HGTDP-DTA outperforms state-of-the-art methods in terms of both prediction performance and generalization capability. The ablation studies further emphasize the significant contributions of dynamic prompt generation, GCNs, and Transformer modules to the overall model performance. The hybrid Graph-Transformer architecture effectively combines local structural information and long-range dependencies, providing a comprehensive understanding of drug-target interactions. In conclusion, the integration of known drug-target associations and the use of multi-view feature fusion with dynamic prompts significantly enhance DTA prediction. Future work will explore additional heterogeneous graph embedding techniques, optimize model parameters, and accelerate the training process to further improve the efficiency and effectiveness of the HGTDP-DTA model.

# References

1. Pedro J Ballester et al. A machine learning approach to predicting protein– ligand binding affinity with applications to molecular docking. Bioinformatics, 26(9):1169–1175, 2010.   
2. Haoyang Chen et al. MultiscaleDTA: A multiscale-based method with a selfattention mechanism for drug-target binding affinity prediction. Methods, 207:103– 109, 2022.   
3. Zhaoyang Chu et al. Hierarchical graph representation learning for the prediction of drug-target binding affinity. Information Sciences, 613:507–523, 2022.   
4. Mindy I Davis et al. Comprehensive analysis of kinase inhibitor selectivity. Nature biotechnology, 29(11):1046–1051, 2011.   
5. Xumeng Gong et al. Ma-gcl: Model augmentation tricks for graph contrastive learning. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), volume 37, pages 4284–4292, 2023.   
6. Tong He et al. Simboost: a read-across approach for predicting drug–target binding affinities using gradient boosting machines. Journal of cheminformatics, 9:1–14, 2017.   
7. Robert P Hertzberg and Andrew J Pope. High-throughput screening: new technology for the 21st century. Current opinion in chemical biology, 4(4):445–451, 2000.   
8. Yang Hua et al. Mfr-DTA: a multi-functional and robust model for predicting drug–target binding affinity and region. Bioinformatics, 39(2):btad056, 2023.   
9. Menglin Jia et al. Visual prompt tuning. In Proceedings of the European Conference on Computer Vision (ECCV), pages 709–727, 2022.   
10. Mingjian Jiang et al. Drug–target affinity prediction using graph neural network and contact maps. RSC advances, 10(35):20701–20712, 2020.   
11. Mahmood Kalemati et al. BiComp-DTA: Drug-target binding affinity prediction through complementary biological-related and compression-based featurization approach. PLOS Computational Biology, 19(3):e1011036, 2023.   
12. Greg Landrum. Rdkit: Open-source cheminformatics. 2006. Google Scholar, 2006.   
13. Brian Lester et al. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.   
14. Pengfei Liu et al. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35, 2023.   
15. Wenjian Ma et al. Predicting drug-target affinity by learning protein knowledge from biological networks. IEEE Journal of Biomedical and Health Informatics, 27(4):2128–2137, 2023.   
16. Mirco Michel et al. Pconsc4: fast, accurate and hassle-free contact predictions. Bioinformatics, 35(15):2677–2679, 2019.   
17. Thin Nguyen et al. GraphDTA: predicting drug–target binding affinity with graph neural networks. Bioinformatics, 37(8):1140–1147, 2021.   
18. Hakime Öztürk et al. Deepdta: deep drug–target binding affinity prediction. Bioinformatics, 34(17):i821–i829, 2018.   
19. Tapio Pahikkala et al. Toward more realistic drug–target interaction predictions. Briefings in bioinformatics, 16(2):325–337, 2015.   
20. Luca Pinzi and Giulio Rastelli. Molecular docking: shifting paradigms in drug discovery. International journal of molecular sciences, 20(18):4331, 2019.

21. Yuqian Pu et al. DeepFusionDTA: drug-target binding affinity prediction with information fusion and hybrid deep-learning ensemble model. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 19(5):2760–2769, 2021.   
22. Yongtao Qian et al. DoubleSG-DTA: Deep Learning for Drug Discovery: Case Study on the Non-Small Cell Lung Cancer with EGFR T 790 M Mutation. Pharmaceutics, 15(2):675, 2023.   
23. Yongming Rao et al. Denseclip: Language-guided dense prediction with contextaware prompting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18082–18091, 2022.   
24. Michael Remmert et al. Hhblits: lightning-fast iterative protein sequence searching by HMM-HMM alignment. Nature methods, 9(2):173–175, 2012.   
25. Mohammad A. Rezaei et al. Deep Learning in Drug Design: Protein-Ligand Binding Affinity Prediction. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 19(1):407–417, 2022.   
26. Jing Tang et al. Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. Journal of Chemical Information and Modeling, 54(3):735–743, 2014.   
27. Maha Thafar et al. Comparison study of computational prediction tools for drugtarget binding affinities. Frontiers in chemistry, 7:782, 2019.   
28. Andrey A Toropov et al. Simplified molecular input line entry system (smiles) as an alternative for constructing quantitative structure-property relationships (qspr). 2005.   
29. Taras Voitsitskyi et al. 3dprotDTA: a deep learning model for drug-target affinity prediction based on residue-level protein graphs. RSC advances, 13(15):10261– 10272, 2023.   
30. Ziduo Yang et al. MgraphDTA: deep multiscale graph neural network for explainable drug–target binding affinity prediction. Chemical science, 13(3):816–833, 2022.   
31. Hongyan Ye et al. Hsgcl-DTA: Hybrid-scale Graph Contrastive Learning based Drug-Target Binding Affinity Prediction. In 2023 IEEE 35th International Conference on Tools with Artificial Intelligence (ICTAI), pages 947–954, 2023.   
32. Li Zhang et al. Predicting drug–target binding affinity through molecule representation block based on multi-head attention and skip connection. Briefings in Bioinformatics, 23(6):bbac468, 2022.   
33. Jing Zhao et al. Multi-view learning overview: Recent progress and new challenges. Information Fusion, 38:43–54, 2017.   
34. Lingling Zhao et al. GansDTA: Predicting drug-target binding affinity using GANs. Frontiers in genetics, 10:1243, 2020.   
35. Qichang Zhao et al. AttentionDTA: prediction of drug–target binding affinity using attention model. In Proceedings of IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pages 64–69, 2019.   
36. Kaiyang Zhou et al. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337–2348, 2022.   
37. Xuekai Zhu et al. FingerDTA: a fingerprint-embedding framework for drug-target binding affinity prediction. Big Data Mining and Analytics, 6(1):1–10, 2022.