# Memory Systems Evaluation

A systematic evaluation of memory systems across three task categories: **Conversational Memory**, **Long-Document Understanding**, and **Agentic Tasks**. All evaluations use 250 questions per dataset.

---

## Groups & Systems

### Group 1 — Conversational Memory
| System | Description |
|--------|-------------|
| [Mem0](https://github.com/mem0ai/mem0) | Graph + vector hybrid memory |
| [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | OS-inspired hierarchical memory |
| [A-Mem](https://github.com/agiresearch/A-MEM) | Agentic memory with graph evolution |
| [Memento](https://github.com/AgentFly/Memento) | CBR-based memory with MCP servers |
| [THEANINE](https://github.com/theanine-eval/theanine) | Timeline-aware memory |
| [MemoryBank](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) | Ebbinghaus forgetting curve memory |
| [Mem1](https://github.com/mem1ai/mem1) | Compressed single-memory approach |
| [MemGPT](https://github.com/cpacker/MemGPT) | OS-inspired virtual context management |

### Group 2 — Long-Document Understanding
| System | Description |
|--------|-------------|
| [ReadAgent](https://github.com/google-deepmind/read_agent) | Gisting-based document reading |
| [A-Mem](https://github.com/agiresearch/A-MEM) | Chunked memory graph for documents |
| [MemTree](https://github.com/memory-tree/MemTree) | Hierarchical memory tree |
| [MemGPT](https://github.com/cpacker/MemGPT) | Archival memory for long documents |
| [MemOS](https://github.com/MemTensor/MemOS) | Cloud-based LLM memory extraction |

### Group 3 — Agentic Tasks
| System | Description |
|--------|-------------|
| [AWM](https://github.com/WangXinglin/AWM) | Agent Workflow Memory with web search |
| [Memento](https://github.com/AgentFly/Memento) | CBR-based agent with DuckDuckGo search |

---

## Datasets

All filtered evaluation sets (250 questions each) are hosted on HuggingFace:

**[darshan3j/memory-systems-eval-datasets](https://huggingface.co/datasets/darshan3j/memory-systems-eval-datasets)**

| Group | Dataset | Source | Metric | Questions |
|-------|---------|--------|--------|-----------|
| G1 | LoCoMo | [LoCoMo](https://github.com/snap-research/locomo) | Contains-Match % | 250 |
| G1 | SimpleQA | [OpenAI SimpleQA](https://github.com/openai/simple-evals) | Contains-Match % | 250 |
| G1 | DeepResearcher | [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) | Contains-Match % | 250 |
| G2 | QuALITY | [QuALITY v1.0.1](https://github.com/nyu-mll/quality) | Accuracy % | 250 |
| G2 | NarrativeQA | [NarrativeQA](https://github.com/google-deepmind/narrativeqa) | Token F1 % | 250 |
| G2 | HaluEval | [HaluEval](https://github.com/RUCAIBox/HaluEval) | Contains-Match % | 250 |
| G3 | Mind2Web | [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) | Avg Action F1 % | 250 |
| G3 | FRAMES | [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) | Accuracy % | 250 |
| G3 | MuSiQue | [MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) | Contains-Match % | 250 |

---

## Results (250 Questions)

### Group 1 — Conversational Memory

| System | LoCoMo (CM%) | SimpleQA (CM%) | DeepResearcher (CM%) | Avg (%) |
|--------|-------------|----------------|----------------------|---------|
| Mem0 | 4.8 | 8.8 | 12.0 | 8.5 |
| MemoryOS | 24.4 | 4.4 | 16.0 | 14.9 |
| A-Mem | 8.4 | 4.4 | 12.0 | 8.3 |
| Memento | 1.2 | 3.6 | 13.6 | 6.1 |
| THEANINE | 13.2 | 29.2 | 22.4 | 21.6 |
| **MemoryBank** | 2.8 | **49.6** | 25.2 | **25.9** |
| **Mem1** | **28.0** | 6.8 | **26.4** | 20.4 |
| MemGPT | 24.8 | 7.2 | 20.0 | 17.3 |

### Group 2 — Long-Document Understanding

| System | QuALITY (Acc%) | NarrativeQA (F1%) | HaluEval (CM%) | Avg (%) |
|--------|---------------|-------------------|----------------|---------|
| **ReadAgent** | **68.0** | 29.2 | 86.4 | 61.2 |
| **A-Mem** | 66.0 | **61.3** | 86.8 | **71.4** |
| MemTree | 53.6 | 16.3 | **88.8** | 52.9 |
| MemGPT | 30.8 | 24.4 | 88.4 | 47.9 |
| MemOS | 54.0 | 35.5 | 76.8 | 55.4 |

### Group 3 — Agentic Tasks

| System | Mind2Web (F1%) | FRAMES (Acc%) | MuSiQue (CM%) | Avg (%) |
|--------|---------------|---------------|----------------|---------|
| **AWM** | **50.2** | **20.4** | **24.0** | **31.5** |
| Memento | 43.8 | 16.0 | 14.8 | 24.9 |

---

## Notebooks

| Notebook | Group | Systems |
|----------|-------|---------|
| [Group1_Conversational_Memory.ipynb](notebooks/Group1_Conversational_Memory.ipynb) | G1 | Mem0, MemoryOS, A-Mem, Memento, THEANINE, MemoryBank, Mem1, MemGPT |
| [Group2_Long_Document_Understanding.ipynb](notebooks/Group2_Long_Document_Understanding.ipynb) | G2 | ReadAgent, A-Mem, MemTree, MemGPT, MemOS |
| [Group3_Agentic_Tasks.ipynb](notebooks/Group3_Agentic_Tasks.ipynb) | G3 | AWM, Memento |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/darshan3j/memory-systems-evaluation.git
cd memory-systems-evaluation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Some systems require specific Python versions:
> - Python 3.13: MemoryOS, A-Mem, MemoryBank, THEANINE, MemTree (sentence-transformers)
> - Python 3.10+: Mem0, MemGPT/Letta
> - Python 3.11 (separate venv): Memento (AgentFly)

### 3. Set up API keys
```bash
cp .env.example .env
# Edit .env and add your keys
```

### 4. Download datasets
Run the dataset download cell at the top of each notebook, or manually download from:
[https://huggingface.co/datasets/darshan3j/memory-systems-eval-datasets](https://huggingface.co/datasets/darshan3j/memory-systems-eval-datasets)

### 5. Run notebooks
Open any notebook in Jupyter and run cells top to bottom. Each system section is self-contained — you can run just one system's section if needed.

---

## Metrics

| Metric | Used For | Description |
|--------|----------|-------------|
| Contains-Match % | G1 all, G2 HaluEval, G3 FRAMES & MuSiQue | Ground truth is a substring of the model answer (case-insensitive) |
| Token F1 % | G2 NarrativeQA | Token overlap between prediction and reference answer |
| Accuracy % | G2 QuALITY | MCQ: model letter (A/B/C/D) matches gold label |
| Avg Action F1 % | G3 Mind2Web | Token F1 on web action representation strings |

---

## Repository Structure

```
memory-systems-evaluation/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── notebooks/
│   ├── Group1_Conversational_Memory.ipynb
│   ├── Group2_Long_Document_Understanding.ipynb
│   └── Group3_Agentic_Tasks.ipynb
├── evaluation/
│   └── metrics.py          # Shared metric implementations
└── results/                # gitignored — results go here when you run
```
