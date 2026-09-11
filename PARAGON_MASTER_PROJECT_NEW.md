# PARAGON — Master Project Document

**Project:** Paraphrase-aware robust detection of AI-generated text using graph-based structural evidence and hybrid detection signals  
**Working name:** PARAGON  
**Academic Context:** Semester Project — CSE-AI, Semester III  
**Project Type:** Research-oriented NLP / Deep Learning project  
**Primary Domain:** Natural Language Processing, AI-generated text detection, Graph Neural Networks, Trustworthy AI

---

## 0. Purpose of This Document

This document is the **single source of truth for the PARAGON project**.

It is intentionally written so that a new team member, faculty member, researcher, coding agent, documentation agent, or other AI agent can understand the project without needing the original conversations, whiteboard notes, or verbal explanations.

It describes:

- the origin and evolution of the project idea;
- the problem being studied;
- the research hypothesis and questions;
- the agreed project boundaries;
- the proposed end-to-end methodology;
- what has already been decided;
- what remains intentionally open for research;
- the dataset and paraphrase-generation strategy;
- the graph/GNN investigation;
- the deterministic graph-feature branch;
- the existing-detector branch;
- the final hybrid/ensemble investigation;
- the evaluation strategy;
- the expected experiments and reporting;
- practical constraints and risks;
- how the project should be implemented and documented.

### Important interpretation rule

This is a **research project**, not a specification of one predetermined neural architecture.

Where this document says that a component is **open for research**, the team must not silently convert an example into a fixed implementation. Candidate methods should first be identified through literature review and/or pilot experiments, then compared, and only then frozen for the final system.

---

# 1. Project Origin and Previous Idea

The original project synopsis proposed a:

> **“Multi-View Graph Representation Learning Framework for Robust Detection of Adversarially Paraphrased AI-Generated Text.”**

The initial synopsis framed the problem as a weakness of existing AI-generated-text detectors: they may perform reasonably on unmodified AI text but can degrade substantially when the text is paraphrased. It proposed exploring a graph-based structural representation, a GNN as a structural detection expert, and a mixture-of-experts-style combination with established detection approaches.

The synopsis deliberately left several architectural choices open, including the exact detector set, graph design, fusion mechanism, and implementation details. It also stated that some proposed components may be simplified, replaced, or dropped according to experimental results.

### Current evolution of the idea

The team has deliberately reduced the original scope.

The project is **not** intended to implement every detector, every attack, every dataset, adversarial training, fairness analysis, deployment, or every component from the earlier large PARAGON playbook.

The current project is centered on one clear research problem:

> **Can structural information extracted from a graph representation of text remain useful for detecting AI-generated text after that text has been adversarially paraphrased, and can such structural evidence complement existing AI-text detectors in a hybrid system?**

The project therefore consists of four main evidence sources:

1. **Controlled paraphrase generation using DIPPER** to create the attack conditions.
2. **Graph-based representations + GNNs** to investigate structural signals.
3. **Deterministic graph-derived mathematical features** to capture interpretable structural patterns.
4. **Existing AI-text detectors** to provide established non-graph detection signals.

These signals will ultimately be studied individually and in combination.

---

# 2. Problem Statement

Large language models can generate fluent text that may be difficult to distinguish from human-written content. Existing AI-generated-text detectors often rely on statistical, lexical, stylistic, model-specific, or other surface-level properties. Such properties can be altered through paraphrasing while preserving much of the underlying meaning.

A major question is therefore whether AI-generated text remains detectable after controlled paraphrasing, and whether a detector that uses **structural information** can retain useful evidence where conventional signals degrade.

The project addresses this problem by constructing controlled paraphrases of AI-generated paragraphs, representing those texts through candidate graph structures, learning graph-level representations with GNNs, extracting deterministic graph features, and comparing these structural signals with established AI-text detectors. The final stage investigates whether the signals are complementary and whether a hybrid/ensemble system is more robust than any individual component.

The focus is therefore **not merely on maximizing clean-text detection accuracy**. The core concern is **robustness under increasing paraphrase intensity**.

---

# 3. Core Research Question

> **How robust is AI-generated-text detection to controlled adversarial paraphrasing, and can graph-based structural evidence provide complementary information that improves robustness?**

## 3.1 Secondary Research Questions

### RQ1 — Graph representation

Which graph-based representations of paragraph-level text retain useful information for distinguishing human-written and AI-generated text under paraphrasing?

### RQ2 — Structural robustness

How does the usefulness of graph-derived structural evidence change as paraphrasing intensity increases?

### RQ3 — GNN effectiveness

Can a GNN learn useful graph-level representations for the binary human-vs-AI detection task under increasingly paraphrased AI text?

### RQ4 — Multi-view representation

Does jointly using multiple complementary graph views provide more robust evidence than using a single graph representation alone?

### RQ5 — Deterministic structural evidence

Do deterministic graph-derived mathematical features provide complementary information to learned GNN embeddings?

### RQ6 — Hybrid detection

Does combining structural evidence with established AI-text detectors improve robustness compared with the individual detectors or structural model alone?

---

# 4. Main Hypothesis

The working hypothesis is:

> **AI-generated text may contain structural patterns that are not completely destroyed by paraphrasing. A graph representation and GNN can potentially learn some of these structural patterns, while deterministic graph statistics may capture additional complementary information. Combining these structural signals with existing detector outputs may therefore improve robustness under paraphrasing.**

This is a **testable hypothesis, not a guaranteed result**.

The project explicitly allows outcomes in which:

- graph methods outperform existing detectors;
- graph methods perform similarly;
- graph methods improve only at certain paraphrase levels;
- deterministic features outperform GNN features for some conditions;
- multi-view graphs help or do not help;
- the final ensemble provides only partial improvement; or
- the graph approach does not provide meaningful improvement.

A negative or mixed result is still a valid research outcome if the experimental design and analysis are rigorous.

---

# 5. What the Project Is and Is Not

## 5.1 The project IS

- A controlled investigation of paraphrase-robust AI-generated-text detection.
- A graph-representation research study.
- A GNN-based classification investigation.
- An experiment comparing structural representations.
- An experiment combining learned and deterministic structural evidence.
- A benchmark against selected existing detectors.
- A robustness study across multiple paraphrase intensities.

## 5.2 The project IS NOT

At the current scope, the project is not committed to:

- a production deployment;
- a real-time detection service;
- a universal detector for every language;
- a claim that AI-text detection is solved;
- a predetermined graph construction before literature research;
- a large-scale commercial system;
- implementing every published AI detector;
- implementing every possible GNN architecture;
- large-scale adversarial training unless later justified;
- a guaranteed improvement over every baseline.

The project's scope remains English-language, paragraph-level, research-oriented detection with paraphrasing as the principal robustness challenge.

---

# 6. High-Level End-to-End Concept

```text
                    ORIGINAL TEXT DATA
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Human-written                 AI-generated
             │                           │
             │                    ┌──────┴──────┐
             │                    │             │
             │                   L0           DIPPER
             │                                  │
             │                     ┌────────────┼────────────┐
             │                     │            │            │
             │                    L1           L2           L3 ... L4
             │                     │            │            │
             └─────────────────────┴────────────┴────────────┘
                                      │
                                MASTER DATASET
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
             EXISTING DETECTORS   GRAPH BRANCH     DETERMINISTIC
                    │                 │            GRAPH FEATURES
           ┌────────┼────────┐        │                 │
           ▼        ▼        ▼        ▼                 │
        RoBERTa  GPTZero  Binoculars  GNN              │
                                      │                 │
                              Graph representation     │
                                      │                 │
                              Learned embedding        │
                                      │                 │
                                      └──────┬──────────┘
                                             ▼
                                      STRUCTURAL EVIDENCE
                                             │
                          ┌──────────────────┴──────────────────┐
                          │                                     │
                          ▼                                     ▼
                    Individual analysis                 Hybrid / Ensemble
                                                                  │
                                                                  ▼
                                                        Final AI probability / class
                                                                  │
                                                                  ▼
                                                        Robustness analysis
                                                        L0 → L1 → L2 → L3 → L4
```

This diagram is the **conceptual architecture**. The precise graph construction and internal model details remain open research decisions.

---

# 7. Dataset Strategy

## 7.1 Initial source dataset

The team has selected the following Hugging Face dataset as the initial human/AI text source:

**AI-human-text dataset**  
https://huggingface.co/datasets/andythetechnerd03/AI-human-text

The source dataset is being treated as the starting point for constructing the project's controlled evaluation corpus.

## 7.2 Initial target size

The current target is:

- **5,000 human-written paragraphs**
- **5,000 AI-generated paragraphs**

Therefore the initial original corpus target is approximately **10,000 paragraphs**.

This is an initial working target rather than an irreversible dataset size. The final size may be adjusted according to data quality, duplicate removal, paraphrase success rate, compute cost, and experimental requirements.

## 7.3 Unit of data

The unit of analysis is a **paragraph**.

The paragraph-level choice is intentional because the project studies paraphrasing of meaningful text structures rather than isolated words or short sentences.

## 7.4 Data lineage

Each original AI sample should retain an identifier linking all of its paraphrased descendants.

Conceptually:

```text
source_id = 1842

1842_L0  → original AI text
1842_L1  → DIPPER paraphrase level 1
1842_L2  → DIPPER paraphrase level 2
1842_L3  → DIPPER paraphrase level 3
1842_L4  → DIPPER paraphrase level 4
```

Human samples do not need paraphrase descendants, but they should have their own stable source IDs.

This lineage is required for correct evaluation and leakage prevention.

---

# 8. Train / Validation / Test Splitting

The split must occur **at the source-document or source-family level**, not independently on every generated variant.

For an AI source paragraph and all of its paraphrases:

```text
Original AI paragraph
        │
   ┌────┼────┬────┬────┐
   L0   L1   L2   L3   L4
```

all variants must belong to the same split.

For example, the following is prohibited:

```text
TRAIN → source 1842 / L1
TEST  → source 1842 / L4
```

because the model could indirectly encounter the same underlying semantic content during training and testing.

The precise split ratio will be finalized after inspecting dataset size and distribution, but the **group-wise split principle is fixed**.

---

# 9. DIPPER Paraphrasing Attack

## 9.1 Why DIPPER is used

The official DIPPER work describes a discourse-level paraphraser capable of paragraph-length paraphrasing and controlled changes in lexical diversity and content order. It was specifically designed to stress-test AI-generated-text detectors.

The corresponding research showed that paraphrasing can substantially reduce the performance of multiple detection methods while preserving much of the original semantics.

Official repository:
https://github.com/martiansideofthemoon/ai-detection-paraphrases

Reference paper:
https://arxiv.org/abs/2303.13408

## 9.2 DIPPER controls

DIPPER exposes two important control dimensions:

- **Lexical diversity (L):** controls how much lexical modification is introduced.
- **Order diversity (O):** controls content reordering.

The DIPPER paper describes lexical diversity using unigram overlap and order diversity using a Kendall-Tau-based token ordering measure, with normalized control values such as 0, 20, 40, 60, 80, and 100.

## 9.3 Four paraphrase levels

The project will construct **four controlled paraphrase conditions**, in addition to the original AI text.

The exact `(L, O)` settings are **not permanently hard-coded in this master document**. Candidate settings will be selected after examining DIPPER's documented configurations and pilot-generation quality.

Conceptually:

```text
L0 = Original AI text
L1 = Low paraphrase intensity
L2 = Moderate paraphrase intensity
L3 = High paraphrase intensity
L4 = High lexical / structural paraphrase intensity
```

The important requirement is that the levels represent **increasing and controlled paraphrase variation**, while maintaining acceptable semantic equivalence.

## 9.4 Paraphrase quality control

Generated paraphrases must not simply be accepted because DIPPER produced them.

The project should evaluate:

- semantic preservation;
- length / validity constraints;
- generation failures;
- duplicate or near-duplicate outputs;
- undesired semantic drift;
- relationship between specified DIPPER controls and actual textual change.

The exact semantic-quality metric or combination of metrics will be finalized during the methodology research stage.

The DIPPER paper itself demonstrates the importance of evaluating semantic similarity alongside detector performance and reports high semantic preservation for much of its attack data.

---

# 10. Primary Classification Task

The final detection task is **binary classification**:

```text
0 → Human-written
1 → AI-generated
```

Paraphrasing does **not** become a third class.

Instead:

```text
AI original → AI label
AI L1       → AI label
AI L2       → AI label
AI L3       → AI label
AI L4       → AI label
```

This preserves the actual research question:

> **Can the detector continue to identify AI authorship when the AI-generated text has been increasingly paraphrased?**

Thus, paraphrase intensity is an **attack condition / robustness variable**, not the classification label.

---

# 11. Graph Representation Research

## 11.1 Core principle

The project will investigate how paragraph-level text can be represented as graph structures that preserve useful linguistic or relational information for AI-text detection.

### The exact graph is intentionally OPEN.

The team has explicitly decided that the following must not be treated as final before literature research:

- node definition;
- edge definition;
- graph type;
- feature representation;
- graph-construction algorithm;
- graph normalization;
- whether a graph is homogeneous or heterogeneous;
- whether semantic and syntactic information should be separated or jointly represented.

## 11.2 Candidate research space

The master project may investigate graph families such as:

- grammar / dependency-driven graphs;
- semantic / relation-driven graphs;
- entity / concept-oriented graphs;
- lexical or co-occurrence graphs;
- heterogeneous graphs;
- multi-view graphs combining complementary graph representations.

These are **examples of research directions, not a preselected final implementation**.

## 11.3 Reference work

A useful reference is:

**“Text Graph Neural Networks for Detecting AI-Generated Content” (GenAIDetect 2025).**

That study constructs a co-occurrence graph in which words are nodes and edges represent word co-occurrence within a context window. It uses edge weights such as frequency and PMI and applies GAT-based graph learning. fileciteturn4file0L121-L146

The paper also compares different node-feature initialization strategies, including Random, Word2Vec, BERT, and RoBERTa features. fileciteturn4file0L245-L260

The paper is therefore a **reference point for prior art**, not the definition of PARAGON's graph.

## 11.4 Graph research objective

The objective is not simply:

> “Create one graph and train one GNN.”

The objective is:

> **Investigate which graph representation(s) preserve useful detection information under paraphrasing and whether multiple complementary structural views provide additional robustness.**

---

# 12. Single-View Graph Experiments

Candidate graph representations will initially be examined independently.

Conceptually:

```text
Same paragraph
      │
 ┌────┼────┐
 ▼    ▼    ▼
 G1   G2   G3   ...
 │    │    │
GNN  GNN  GNN
 │    │    │
S1   S2   S3
```

Each graph/GNN combination can be evaluated independently across the same paraphrase conditions.

This allows the project to answer:

> **Which representation is most useful?**

and:

> **Which representation degrades least as paraphrasing increases?**

The exact number and types of graph variants are deliberately left open until the research phase.

---

# 13. Multi-View Graph Experiments

## 13.1 Definition used by this project

A **multi-view graph approach** means that multiple graph representations of the same text are provided jointly to the learning system so that the model can learn from their complementary structural information.

Conceptually:

```text
                       PARAGRAPH
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
        Graph View A    Graph View B    Graph View C
           │               │               │
        Encoder A       Encoder B       Encoder C
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                    Multi-view fusion
                           │
                          GNN
                           │
                      Graph embedding
                           │
                      Final prediction
```

## 13.2 Why multi-view is investigated

Different graph constructions may encode different aspects of a paragraph.

For example, one view may capture structural/grammatical relations while another may capture lexical or semantic associations.

The project will experimentally investigate whether these views are complementary.

Again, **the exact views and fusion mechanism are open research questions**.

---

# 14. GNN Research Branch

The project will use Graph Neural Networks to learn graph-level representations from the selected graph structures.

### What is fixed

- GNNs are a core research component.
- The task is paragraph-level binary classification.
- Models are evaluated under the same paraphrase conditions.

### What is open

- exact GNN architecture;
- number of message-passing layers;
- hidden dimensions;
- pooling method;
- attention or non-attention mechanism;
- edge-weight handling;
- node-feature initialization;
- regularization;
- exact training objective;
- multi-view fusion mechanism.

Candidate architectures may be investigated based on literature and pilot results; no single GNN architecture is considered mandatory before those studies.

---

# 15. Deterministic Graph-Feature Branch

The project will not rely only on a learned GNN representation.

A second structural branch will calculate deterministic graph-derived features.

The team has already agreed that these features should initially be treated as **additional structural evidence**, not necessarily as a standalone detector.

Potential feature families include, depending on the selected graph representation:

- node-degree statistics;
- degree distribution;
- graph density;
- clustering-related statistics;
- centrality measures;
- connectivity measures;
- shortest-path statistics where meaningful;
- community / modularity-related information where meaningful;
- other graph descriptors justified by the literature.

The final feature list must be research-supported and should remain reasonably small and interpretable.

Conceptually:

```text
Graph
 │
 ├──────────────→ GNN → learned structural embedding
 │
 └──────────────→ deterministic graph features
                              │
                              ▼
                    structural evidence
```

The purpose is to test whether deterministic graph properties add information beyond the learned GNN representation.

---

# 16. Existing AI-Text Detector Layer

The project will include three permanently selected external detector/baseline components:

1. **RoBERTa-based detector**
2. **GPTZero**
3. **Binoculars**

These components serve as established non-graph detection signals and as comparison baselines.

They are not being treated as the project's novelty. Their purpose is to establish:

- how strong established signals are;
- how their performance changes under DIPPER paraphrasing;
- whether graph-based evidence behaves differently;
- whether graph evidence is complementary to existing detectors.

The project should record the exact model/checkpoint/version, interface, score interpretation, minimum text constraints, and configuration used for each detector once implementation is finalized.

---

# 17. Hybrid / Ensemble System

The final stage combines the evidence sources.

Conceptually:

```text
                     INPUT PARAGRAPH
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          RoBERTa       GPTZero       Binoculars
             │             │             │
             └─────────────┼─────────────┘
                           │
                     Existing signals
                           │
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
                  GNN        Graph mathematics
                    │             │
                    └──────┬──────┘
                           │
                    Structural signals
                           │
               ┌──────────┴──────────┐
               ▼                     ▼
          Existing evidence     Structural evidence
               └──────────┬──────────┘
                          ▼
                     Fusion layer
                          │
                          ▼
               Final AI probability/decision
```

The exact fusion mechanism is still open.

Potentially testable strategies include:

- simple weighted fusion;
- concatenation of detector scores/features followed by a meta-classifier;
- learned gating;
- another experimentally justified fusion method.

The team should select the simplest approach that can answer the research question well before adding complexity.

---

# 18. Recommended Experimental Progression

The project should be implemented progressively rather than building the entire hybrid architecture immediately.

## Stage 1 — Reproduce the attack conditions

- obtain and run DIPPER;
- generate a small pilot set;
- confirm the four paraphrase conditions produce increasing diversity;
- validate semantic preservation;
- confirm the existing detectors show meaningful performance degradation.

## Stage 2 — Establish detector baselines

Run:

- RoBERTa detector;
- GPTZero;
- Binoculars;

on:

- human text;
- original AI text;
- each paraphrase level.

This produces the baseline robustness profile.

## Stage 3 — Graph representation research

Conduct the literature review and choose a manageable set of candidate graph representations.

For each candidate:

- construct graphs;
- define node/edge information;
- decide feature initialization;
- train the GNN;
- evaluate across all paraphrase levels.

## Stage 4 — Graph comparison

Compare individual graph approaches across:

- clean detection performance;
- low-FPR performance;
- robustness degradation;
- compute cost;
- stability.

## Stage 5 — Deterministic structural features

Evaluate graph-derived mathematical features independently and with the selected GNN representation.

## Stage 6 — Multi-view graph investigation

Only after the individual graph views are understood, investigate whether jointly using multiple views provides a meaningful improvement.

## Stage 7 — Hybrid system

Combine:

- selected GNN/graph evidence;
- deterministic graph evidence;
- RoBERTa;
- GPTZero;
- Binoculars.

## Stage 8 — Full evaluation and ablation

Compare the complete system with its components and systematically remove components to determine what actually contributes to robustness.

---

# 19. Evaluation Framework

## 19.1 Core classification metrics

The evaluation should include metrics such as:

- AUROC;
- precision;
- recall;
- F1 score;
- confusion matrix where useful.

## 19.2 Low-FPR evaluation

False positives matter strongly in AI-text detection because a human-written text incorrectly classified as AI-generated is a serious error.

Therefore the project should also report low-FPR operating points, particularly:

- TPR at 1% FPR;
- TPR at additional low-FPR operating points where useful.

The DIPPER paper explicitly emphasizes low-FPR evaluation and shows ROC behaviour in the 0–1% FPR region. fileciteturn2file0L299-L308

## 19.3 Robustness evaluation

Every detector/model should be tested across:

```text
L0 → original AI
L1 → paraphrase level 1
L2 → paraphrase level 2
L3 → paraphrase level 3
L4 → paraphrase level 4
```

The critical output is not just a score at L0, but the **performance trajectory**.

## 19.4 Degradation analysis

For a metric `M`:

```text
Degradation(Li) = M(L0) - M(Li)
```

Relative degradation can also be reported when appropriate.

This allows comparison of:

> how much a detector loses as paraphrasing becomes stronger.

## 19.5 Robustness curves

The project should generate curves showing detector performance against paraphrase intensity.

Conceptually:

```text
Detection performance
100% ┤●
     │ ╲
 80% ┤  ●
     │   ╲
 60% ┤    ●
     │      ╲
 40% ┤       ●
     │          ●
 20% ┤
     └────────────────────
       L0  L1  L2  L3  L4
```

The curves should be generated for:

- RoBERTa;
- GPTZero;
- Binoculars;
- candidate graph/GNN systems;
- deterministic structural features;
- multi-view system where applicable;
- final hybrid system.

---

# 20. Core Experimental Matrix

A central results table should eventually resemble the following structure:

| System | L0 | L1 | L2 | L3 | L4 | Degradation Trend |
|---|---:|---:|---:|---:|---:|---|
| RoBERTa |  |  |  |  |  |  |
| GPTZero |  |  |  |  |  |  |
| Binoculars |  |  |  |  |  |  |
| Graph View A + GNN |  |  |  |  |  |  |
| Graph View B + GNN |  |  |  |  |  |  |
| Graph View C + GNN |  |  |  |  |  |  |
| Deterministic graph features |  |  |  |  |  |  |
| GNN + deterministic features |  |  |  |  |  |  |
| Multi-view graph system |  |  |  |  |  |  |
| Final hybrid ensemble |  |  |  |  |  |  |

The exact number of graph rows depends on the literature research.

---

# 21. Ablation Studies

Ablation is required to demonstrate that the final system's behaviour is not simply caused by one component.

At minimum, the final experimental analysis should consider variants such as:

```text
Full system

Full system – GNN
Full system – deterministic graph features
Full system – RoBERTa
Full system – GPTZero
Full system – Binoculars

GNN only
GNN + graph mathematics
Existing detectors only
Existing detectors + structural evidence
```

If multi-view graphs are used, compare:

```text
Best single graph view
vs.
Multi-view graph
```

The exact ablation list may be refined according to the final architecture.

---

# 22. What Counts as a Successful Result?

Success is **not predefined as a specific accuracy number**.

A convincing result would ideally show one or more of the following:

- graph-based signals retain useful detection ability at stronger paraphrase levels;
- some graph representations degrade more slowly than others;
- deterministic structural features add complementary information;
- multi-view graph learning improves over individual views;
- the structural branch provides information not captured by the existing detectors;
- the final hybrid system achieves lower performance degradation than the strongest individual baseline.

A result such as the following would be highly informative even if the clean score were not the highest:

```text
                    L0       L1       L2       L3       L4
Existing detector   92       85       72       54       39
Graph system        89       86       81       74       68
Hybrid              94       90       86       81       76
```

The important observation would be **robustness**, not only the L0 score.

---

# 23. Data Quality and Leakage Controls

The project must explicitly control the following risks:

### 23.1 Source leakage

All descendants of one original source must remain in the same dataset split.

### 23.2 Duplicate leakage

Near-duplicate paragraphs should be detected and handled before the final split where possible.

### 23.3 Paraphrase-family contamination

Do not place different paraphrase levels of the same source into different evaluation partitions.

### 23.4 Label contamination

Do not allow metadata describing `L1`, `L2`, `L3`, `L4`, or the AI/human source to accidentally become an input feature to the classifier.

### 23.5 Evaluation contamination

If model selection decisions use the test set, the experiment becomes biased. Hyperparameter/model selection must use training/validation data, while the final test set is reserved for the final evaluation.

---

# 24. Dataset Metadata Schema

A practical dataset manifest should contain fields similar to:

```text
sample_id
source_id
text
label
generation_type
paraphrase_level
lexical_control
order_control
parent_sample_id
split
language
length_tokens
quality_status
```

These fields are metadata, not necessarily model features.

The exact schema may be expanded after data engineering begins.

---

# 25. Proposed Software/Research Stack

The exact versions should be frozen only after environment testing, but the project is expected to use a Python-based ecosystem.

Likely components include:

- Python;
- PyTorch;
- PyTorch Geometric;
- Hugging Face Transformers;
- NLP parsing libraries appropriate to the selected graph methods;
- graph processing utilities;
- pandas / NumPy for data processing;
- scikit-learn for evaluation and classical components where appropriate;
- experiment tracking / structured logging;
- Git and GitHub.

The exact parser, GNN package versions, pretrained checkpoints, and environment are **implementation decisions**, not conceptual commitments.

---

# 26. Research Literature Plan

Before freezing the graph design, the team should review literature in at least these groups:

1. AI-generated-text detection;
2. robustness of AI detectors under paraphrasing;
3. DIPPER and controlled paraphrase attacks;
4. graph neural networks for text classification;
5. graph representations of text;
6. semantic/dependency graphs for NLP;
7. heterogeneous and multi-view graphs;
8. graph-level classification;
9. graph-based authorship/stylometry or AI-text detection;
10. ensemble / mixture-of-experts approaches for detector combination.

For every candidate graph representation, record:

```text
Paper / method
Text unit
Node definition
Edge definition
Graph type
Node features
Edge features
GNN architecture
Task
Dataset
Strengths
Weaknesses
Computational cost
Relevance to paraphrase robustness
```

The final graph design should emerge from this matrix rather than intuition alone.

---

# 27. Important Reference: Text Graph AI Detection Paper

**Paper:** Text Graph Neural Networks for Detecting AI-Generated Content  
**Venue:** 1st Workshop on GenAI Content Detection, 2025

The paper is relevant because it directly studies text graphs and GNNs for machine-generated-text detection.

Key details from the paper:

- it uses a co-occurrence graph;
- each word is represented as a node;
- co-occurrence within a context window defines an edge;
- edge weights can use frequency or PMI;
- node features can be Random, Word2Vec, BERT, or RoBERTa-derived;
- the graph is processed using a GAT;
- the resulting graph representation is passed to a final classifier. fileciteturn4file0L121-L205

The paper's results also show that node initialization has a meaningful effect and that transformer-derived features improved the GNN results over simpler initializations on its dataset. fileciteturn4file0L245-L280

The paper itself identifies alternative graph representations and heterogeneous graphs as future directions, which is relevant to PARAGON's broader graph-research objective. fileciteturn4file0L301-L321

**Interpretation for PARAGON:** use this work as related work and a possible baseline/reproduction reference, not as a predetermined graph architecture.

---

# 28. Important Reference: DIPPER Attack Paper

**Paper:** Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense  
**Conference:** NeurIPS 2023  
**Repository:** https://github.com/martiansideofthemoon/ai-detection-paraphrases  
**Paper:** https://arxiv.org/abs/2303.13408

Key points relevant to PARAGON:

- DIPPER is a paragraph-level discourse paraphraser;
- lexical diversity and order diversity are controllable;
- increasing diversity can substantially reduce detector performance;
- semantic preservation should be evaluated alongside detector success;
- low-FPR evaluation is important for AI-text detection. fileciteturn2file0L85-L104 fileciteturn2file0L299-L308

The paper reports experiments using controlled configurations such as `20L`, `40L`, `60L`, and `60L,60O`, which can serve as evidence-backed candidate settings during the PARAGON pilot phase. fileciteturn2file0L312-L344

The project should use these as **reference configurations**, not blindly copy them without validating them on PARAGON's chosen dataset.

---

# 29. Major Project Risks

## Risk 1 — Graph design becomes arbitrary

**Problem:** The team selects a graph because it is easy to implement rather than because literature supports it.

**Mitigation:** Maintain a literature matrix and compare candidate representations systematically.

## Risk 2 — Graph complexity becomes too high

**Problem:** Too many graph types, node types, edge types, and GNN architectures produce an unmanageable experiment space.

**Mitigation:** Start with a small number of literature-justified candidates; expand only when a result motivates expansion.

## Risk 3 — Dataset leakage

**Problem:** The model sees one paraphrase of a source during training and another paraphrase of the same source during testing.

**Mitigation:** source-family grouped splitting.

## Risk 4 — Paraphrase quality is poor

**Problem:** High DIPPER diversity produces semantic drift or unusable outputs.

**Mitigation:** automatic quality checks, filtering, pilot testing, and manual spot checks.

## Risk 5 — Existing detectors cannot run reliably

**Problem:** External APIs, checkpoints, versions, or interfaces change.

**Mitigation:** document exact versions/configurations and create a normalized detector interface.

## Risk 6 — GNN performs poorly

**Problem:** A graph representation may not contain enough useful signal, or the chosen GNN may be unstable.

**Mitigation:** compare representations, simplify the model, compare against deterministic graph features, and report negative findings honestly.

## Risk 7 — Ensemble does not improve

**Problem:** Detector signals may be highly correlated.

**Mitigation:** inspect individual score distributions and correlations; evaluate whether structural signals are actually complementary before adding a complex fusion mechanism.

## Risk 8 — Too many experiments

**Problem:** The team tests every imaginable graph × feature × GNN × paraphrase combination.

**Mitigation:** define research gates and freeze only a manageable candidate set before large-scale experiments.

---

# 30. Research Gating / Decision Process

The project should progress through explicit decision gates.

### Gate A — Attack validity

Proceed only if the DIPPER pipeline produces useful paraphrase conditions with acceptable quality.

### Gate B — Baseline validity

Proceed only if RoBERTa, GPTZero, and Binoculars can be evaluated reproducibly on the chosen data.

### Gate C — Graph validity

Proceed with large-scale graph experiments only after the candidate graph representations can be generated consistently.

### Gate D — GNN validity

Proceed to multi-view experiments only after individual graph/GNN experiments produce interpretable results.

### Gate E — Hybrid validity

Build the final hybrid system only after demonstrating that the structural branch contributes measurable information beyond the existing detectors.

---

# 31. Team Development Order

The project should not be developed as one giant monolithic program.

Recommended module structure:

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── paraphrased/
│   └── manifests/
│
├── attacks/
│   └── dipper/
│
├── detectors/
│   ├── roberta/
│   ├── gptzero/
│   └── binoculars/
│
├── graphs/
│   ├── candidate_views/
│   ├── construction/
│   ├── features/
│   └── multiview/
│
├── models/
│   ├── gnn/
│   └── fusion/
│
├── experiments/
│   ├── baselines/
│   ├── graph_comparison/
│   ├── multiview/
│   ├── hybrid/
│   └── ablations/
│
├── evaluation/
│   ├── metrics/
│   ├── robustness/
│   └── plots/
│
├── configs/
├── notebooks/
├── reports/
└── README.md
```

The exact directory structure may change, but the separation of concerns should remain.

---

# 32. Reproducibility Requirements

Every major experiment should record:

- dataset version;
- source split;
- paraphrase controls;
- model name/version/checkpoint;
- graph representation;
- node/edge configuration;
- feature initialization;
- GNN architecture;
- random seed;
- training configuration;
- evaluation metrics;
- output artifacts.

Every result in the final report must be traceable back to an experiment configuration and dataset version.

---

# 33. Required Final Outputs

The final project should produce:

### Research artifacts

- cleaned master dataset;
- controlled paraphrase dataset;
- dataset-generation documentation;
- graph construction implementations;
- trained GNN models;
- deterministic graph-feature pipeline;
- existing detector evaluation pipeline;
- final hybrid/ensemble system;
- complete experiment logs.

### Evaluation artifacts

- detector baseline table;
- graph comparison table;
- multi-view comparison;
- structural-feature comparison;
- hybrid/ensemble comparison;
- ablation studies;
- AUROC curves;
- low-FPR performance;
- paraphrase degradation curves;
- qualitative examples where useful.

### Documentation artifacts

- architecture diagram;
- dataset schema;
- graph definitions used in final experiments;
- methodology;
- experimental protocol;
- limitations;
- final conclusions.

---

# 34. Expected Final Report Structure

The final academic report should roughly follow this logic:

1. Abstract
2. Introduction
3. Problem Statement
4. Motivation
5. Research Questions / Hypothesis
6. Related Work
7. Dataset and Paraphrase Attack Setup
8. Graph Representation Investigation
9. GNN Methodology
10. Deterministic Structural Features
11. Existing Detector Baselines
12. Hybrid / Ensemble Method
13. Experimental Setup
14. Results
15. Robustness and Degradation Analysis
16. Ablation Studies
17. Discussion
18. Limitations
19. Conclusion
20. Future Work
21. References

The final structure can be adjusted to match the institution's required format.

---

# 35. Novelty Positioning

The novelty should **not** be stated as:

> “We invented GNN-based AI-text detection.”

Prior research has already explored text graphs and GNNs for machine-generated-text identification, including co-occurrence graphs and GAT-based systems. fileciteturn4file0L121-L205

Instead, PARAGON's intended contribution is the **combination and experimental framing** around paraphrase robustness:

1. controlled DIPPER-based paraphrase levels;
2. comparative research into graph representations;
3. investigation of single-view and multi-view structural evidence;
4. deterministic graph features alongside learned graph representations;
5. comparison against established detector signals;
6. analysis of how each signal degrades as paraphrasing increases;
7. investigation of whether structural evidence is complementary to existing detectors.

Whether these elements produce a meaningful improvement is an empirical question.

---

# 36. Core Principles for the Team

### Principle 1 — Research before locking architecture

Do not choose graph structures, node types, edge types, or GNNs simply because they are convenient.

### Principle 2 — Robustness is the main story

A clean-text score alone is insufficient.

### Principle 3 — Preserve data lineage

Every paraphrased sample must be traceable to its source.

### Principle 4 — Compare components independently

Do not hide a weak component inside an ensemble and claim overall improvement.

### Principle 5 — Keep the hybrid system explainable

Where possible, record what each detector/feature branch contributes.

### Principle 6 — Simpler is preferred until complexity is justified

A more complicated graph or ensemble is useful only if experiments justify it.

### Principle 7 — Negative results are acceptable

The project is a scientific investigation, not a prewritten success story.

### Principle 8 — Never leak the test set into model selection

Use validation data for development decisions and reserve the test set for final reporting.

---

# 37. Current Locked Decisions

The following decisions are currently considered fixed project-level commitments:

| Area | Current decision |
|---|---|
| Core problem | Robust AI-text detection under paraphrasing |
| Text type | English paragraphs |
| Main attack | DIPPER |
| Attack dimensions | Lexical diversity + order/content diversity |
| Paraphrase conditions | Four controlled levels + original baseline |
| Main task | Binary Human vs AI |
| Initial dataset target | 5K Human + 5K AI paragraphs |
| Split principle | Source-family/grouped split |
| Graph branch | Core research component |
| Graph design | **Open — literature + experiments** |
| Node definition | **Open — literature + experiments** |
| Edge definition | **Open — literature + experiments** |
| Single-view graphs | Yes, to compare candidate representations |
| Multi-view graphs | Yes, to test joint use of graph views |
| Deterministic features | Yes |
| GNN | Yes |
| Existing detectors | RoBERTa + GPTZero + Binoculars |
| Final combination | Hybrid / ensemble investigation |
| Main evaluation | AUROC + low-FPR + robustness/degradation analysis |
| Main research focus | Performance change across paraphrase intensity |

---

# 38. Current Open Decisions

The following should **not** be treated as finalized until the research stage:

- exact DIPPER `(L, O)` values for the four conditions;
- exact graph types;
- exact node definitions;
- exact edge definitions;
- graph weighting methods;
- node feature initialization;
- exact GNN architecture;
- number of GNN layers;
- pooling strategy;
- multi-view fusion architecture;
- deterministic graph-feature subset;
- exact hybrid fusion algorithm;
- precise train/validation/test percentages;
- exact semantic-quality filtering method;
- exact dataset filtering rules.

---

# 39. What a New Agent Should Understand Immediately

If another AI agent receives only this file, it should understand the following:

> PARAGON is a semester-scale research project investigating whether graph-based structural representations of English paragraphs can help detect AI-generated text after the text has been adversarially paraphrased.
>
> The team starts with approximately 5,000 human and 5,000 AI paragraphs. AI paragraphs are transformed with DIPPER into four controlled paraphrase conditions using lexical and order/content-diversity controls. The task remains binary Human vs AI; paraphrasing is the robustness condition, not a separate class.
>
> The research does not assume one graph structure in advance. It will study candidate text-graph representations from the literature, including possible grammar/dependency, semantic/entity, lexical/co-occurrence, heterogeneous, or multi-view structures. Node and edge definitions must be determined through research rather than invented beforehand.
>
> Candidate graph views will first be evaluated individually with GNNs. A multi-view experiment will then investigate whether multiple graph representations can be learned jointly. In parallel, deterministic graph statistics will be evaluated as structural evidence alongside learned GNN representations.
>
> The project also permanently includes three existing detection signals: RoBERTa, GPTZero, and Binoculars. Their role is to establish baselines and provide complementary non-graph evidence.
>
> The final stage investigates a hybrid/ensemble system that combines existing detector signals with GNN-based structural evidence and deterministic graph features.
>
> The primary evaluation question is not simply which model has the highest clean-text accuracy. The central question is which system remains effective as paraphrase intensity increases. Therefore AUROC, low-FPR performance, and L0→L4 degradation curves are central outputs.
>
> The project is explicitly exploratory. If a graph representation, GNN, or ensemble component does not help, it should be simplified or removed rather than protected because it appeared in the original idea.

---

# 40. Source Basis

This master document is grounded in the following project materials:

### A. Original VIT project synopsis

Project title in the synopsis:

> “A Multi-View Graph Representation Learning Framework for Robust Detection of Adversarially Paraphrased AI-Generated Text.”

The synopsis establishes the research-oriented problem, graph/GNN direction, mixture-of-experts framing, paraphrase robustness objective, and deliberately open architectural decisions.

### B. DIPPER / paraphrase attack paper

**Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense**  
NeurIPS 2023  
https://arxiv.org/abs/2303.13408

Official repository:  
https://github.com/martiansideofthemoon/ai-detection-paraphrases

### C. Text Graph Neural Networks for Detecting AI-Generated Content

GenAIDetect Workshop 2025.

This is the main uploaded prior-work reference for graph-based AI-text detection and provides an example of co-occurrence graphs + node features + GAT + graph classification. fileciteturn4file0L21-L30

### D. Initial dataset

https://huggingface.co/datasets/andythetechnerd03/AI-human-text

### E. Earlier project planning material

Earlier internal planning explored much larger detector collections, adversarial training, multiple benchmark datasets, and extensive evaluation. The current project intentionally reduces that scope and should follow this master file rather than automatically inheriting every item from those earlier plans.

---

# 41. Final Project Definition

**PARAGON is a research-oriented framework for studying paraphrase-robust AI-generated-text detection. It constructs a controlled dataset of human, AI-generated, and DIPPER-paraphrased AI paragraphs; investigates alternative graph representations of text; trains GNN-based structural detectors; extracts deterministic graph features; compares these signals with RoBERTa, GPTZero, and Binoculars; and investigates whether combining complementary structural and existing detector signals can improve robustness as paraphrasing intensity increases.**

The project's scientific value lies not in assuming that a particular graph, GNN, or ensemble will work, but in **systematically determining what works, under which paraphrase conditions, by how much, and why**.

---

# 42. Immediate Next Research Tasks

The next work should proceed in this order:

1. Verify and inspect the selected 5K+5K source dataset.
2. Establish the DIPPER generation environment.
3. Generate a small pilot with candidate `(L, O)` combinations.
4. Measure semantic preservation and actual lexical/order changes.
5. Run the three fixed external detectors on the pilot.
6. Build a literature matrix of graph representations used in NLP and AI-text detection.
7. Decide a manageable set of candidate graph views.
8. Define the node/edge representation for each candidate from literature evidence.
9. Implement graph construction.
10. Establish single-view GNN baselines.
11. Add deterministic graph features.
12. Compare graph views under L0→L4.
13. Investigate multi-view graphs.
14. Build and evaluate the hybrid ensemble.
15. Run ablations and robustness analysis.
16. Freeze the final architecture and write the final report from the observed results.

---

**Document status:** Master project specification — working research baseline  
**Important:** Open research decisions must remain open until supported by literature and experiments.  
**Last conceptual update:** 20 August 2026
