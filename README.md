# Agentic Internal Knowledge System for R&D
 
A retrieval-augmented generation (RAG) system that answers technical questions about machine learning research papers, with citations back to the paper and page each answer comes from. Built by team Zayi 1A for Break Through Tech AI Studio, Fall 2026, with host company Zayi.
 
---
 
### 👥 **Team Members**
 
| Name                | GitHub Handle | Contribution |
|---------------------|---------------|--------------|
| Godwin Idowu        | @Gidowu       | Data exploration notebook, data profiling and cleaning for the 10 papers, fixed page-number artifacts and the ReAct font-encoding cipher, Gemini API key/config handling and tenacity retry with backoff |
| Ivan Ng             | @cybercommits | Turned the cleaning and chunking notebooks into reusable modules, ReAct decoding fix, 30-question retrieval evaluation (hit@k, MRR), end-to-end pipeline tests, Q&A runner over the ground-truth set, project board and PR reviews |
| Nivedha Aravindmali | @NvyGreen     | Document chunking notebook, generation prompt with numbered source citations |
| Tanish Gupta        | @tanish2007   | Embedding notebook with bge-small-en-v1.5, ChromaDB vector store, LangChain retriever and the shared `answer()` RAG chain |
 
---
 
## 🎯 **Project Highlights**
 
- Built a baseline RAG pipeline over 10 foundational ML papers (218 pages, about 122,000 words): text cleaning, chunking into 918 passages, local embeddings with `BAAI/bge-small-en-v1.5`, a ChromaDB vector store, and answer generation with the Gemini API through LangChain.
- Every answer cites its sources by paper ID and page range, and the prompt instructs the model to say when the sources don't contain the answer instead of guessing.
- On our 30-question retrieval evaluation, the correct paper appears in the top 5 results for **97%** of questions, and the exact supporting passage for **57%**.
- Found and fixed a font-encoding issue that had scrambled part of the ReAct paper's appendix text, which raised exact-passage retrieval from 53% to 57%.
- Gemini calls are wrapped in exponential backoff (tenacity) that retries only temporary errors (rate limits and server errors), and the test suite runs with a mocked Gemini so it uses no API quota.
---
 
## 👩🏽‍💻 **Setup and Installation**
 
**Requirements:** Python 3.11 or newer, and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey). Each team member uses their own key.
 
1. **Clone the repository**
```bash
   git clone https://github.com/Break-Through-Tech/Zayi-1A-agentic-internal-knowledge-system-for-r-and-d.git
   cd Zayi-1A-agentic-internal-knowledge-system-for-r-and-d
```
 
2. **Create a virtual environment and install dependencies.** Installing the CPU-only PyTorch first keeps the download small, and the project doesn't need a GPU.
```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt pytest
```
   On WSL, if the project folder is on the Windows drive (`/mnt/c/...`), create the virtual environment in your Linux home folder instead (for example `~/venvs/zayi`). Installing onto the Windows drive is very slow.
 
3. **Add your API key.** Create a file named `.env` in the repo root containing one line:
```
   GOOGLE_API_KEY=your_key_here
```
   `.env` is listed in `.gitignore`, so it is never committed. In Google Colab, add `GOOGLE_API_KEY` under **Secrets** (the key icon on the left) instead.
 
4. **Data.** The papers' pre-extracted text, the cleaned text and the chunks are already in `data/`, so no download is needed. To regenerate them:
```bash
   python src/cleaning.py      # data/curated_papers_text.json -> data/cleaned_papers_text.json
   python src/chunking.py      # -> data/chunks.json
```
 
5. **Build the vector store** (a few minutes on CPU; downloads the embedding model the first time):
```bash
   python src/vector_store.py build
```
 
6. **Ask a question**
```bash
   python src/rag_chain.py "What does LoRA stand for?" -k 5
```
 
7. **Run the tests and evaluations**
```bash
   python -m pytest -v                       # full test suite (Gemini is mocked, no key needed)
   python -m pytest -v -m "not slow"         # skip the tests that build a Chroma store
   python eval/evaluate_retrieval.py         # retrieval metrics on our 30 Q&A pairs
```
 
---
 
## 🏗️ **Project Overview**
 
This project is part of the [Break Through Tech AI Program](https://www.breakthroughtech.org/), where fellows work in teams on an applied machine learning challenge from a host company. Our host company is **Zayi**, a research-led innovation hub that builds AI and software systems for knowledge management and automation.
 
**Objective and scope.** Engineers at Zayi spend significant time manually reading dense research PDFs to extract architectural patterns for their R&D pipeline. Our goal is an internal knowledge system that retrieves the relevant passages from a curated set of ArXiv papers and drafts cited answers and summary memos, running as a local prototype in Google Colab. The project runs in three milestones:
 
| Month     | Milestone                         | Focus |
|-----------|-----------------------------------|-------|
| September | Data Exploration & Preprocessing  | EDA, text cleaning, chunking, embeddings, ChromaDB |
| October   | Modeling & Orchestration          | Baseline RAG chain with LangChain and Gemini, API error handling and backoff |
| November  | Evaluation & Hyperparameter Tuning | Evaluate against the 40-question ground-truth set, tune chunking, prompt and k-value |
 
**Why it matters.** A system that finds the exact supporting passage and cites it lets engineers verify an answer in seconds instead of rereading a paper, and it reduces the risk of acting on a confident but unsupported answer.
 
---
 
## 📊 **Data Exploration**
 
**Dataset.** A curated set of 10 ArXiv machine learning papers provided by Zayi, as pre-extracted per-page text in JSON (`data/curated_papers_text.json`), with links to the original PDFs.
 
| ArXiv ID   | Paper | Topic | Pages |
|------------|-------|-------|------:|
| 2005.11401 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | RAG | 19 |
| 2312.10997 | Retrieval-Augmented Generation for Large Language Models: A Survey | RAG Survey | 21 |
| 2310.11511 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Self-Correction | 30 |
| 2401.15884 | Corrective Retrieval Augmented Generation (CRAG) | Corrective RAG | 16 |
| 2210.03629 | ReAct: Synergizing Reasoning and Acting in Language Models | Agent Reasoning | 33 |
| 2307.03172 | Lost in the Middle: How Language Models Use Long Contexts | Context Windows | 18 |
| 2309.15217 | RAGAS: Automated Evaluation of Retrieval Augmented Generation | Evaluation | 8 |
| 2106.09685 | LoRA: Low-Rank Adaptation of Large Language Models | Model Architecture | 26 |
| 2205.14135 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Systems | 34 |
| 2004.04906 | Dense Passage Retrieval for Open-Domain Question Answering | Embeddings | 13 |
 
Per-paper page, word and character counts are in `data/paper_profile_summary.csv`, and the exploration is in `notebooks/data_exploration.ipynb`.
 
**Preprocessing** (`src/cleaning.py`):
- Removed ArXiv headers, running page headers, and page numbers, including numbers that leaked across page boundaries.
- Rejoined words hyphenated across line breaks and normalized Unicode to plain ASCII.
- Repaired a font-encoding cipher in the ReAct paper's appendix, where the PDF's custom font had turned text into shifted characters.
**Challenges and assumptions.**
- PDF-extracted text is noisy: two-column layouts, headers and footers mixed into body text, and font-encoding artifacts.
- Each paper's References section produces chunks dense with other papers' titles. We tested removing them (918 chunks down to 736) and found no change in retrieval scores, so we kept them for now.
- Page numbers are tracked through cleaning and chunking so every chunk can be cited by page.
---
 
## 🧠 **Model Development**
 
**Pipeline**
 
```
question
  -> ChromaRetriever: top-k chunks by cosine similarity          (src/rag_chain.py, src/vector_store.py)
  -> numbered sources + prompt (answer only from the sources)    (src/generation.py)
  -> Gemini, with exponential backoff on rate limits             (src/gemini_client.py, src/config.py)
  -> answer text + citations (paper ID, title, pages)            (rag_chain.answer)
```
 
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`, chunk size 1,000 characters with 125 characters of overlap, applied to each paper's page text with page numbers mapped back to every chunk (918 chunks, about 950 characters on average).
- **Embeddings:** `BAAI/bge-small-en-v1.5` from sentence-transformers, run locally, with the model's recommended query prefix for questions.
- **Vector store:** ChromaDB, persisted locally, cosine distance.
- **Generation:** Gemini through `langchain-google-genai`, temperature 0. The prompt requires a bracketed citation after every claim and the reply "The provided sources do not contain this information." when the answer isn't in the sources. Citations to source numbers that weren't retrieved are discarded.
- **Error handling:** tenacity retries rate limits (429), server errors (500, 502, 503, 504), timeouts and dropped connections with randomized exponential backoff (up to 6 attempts), and fails immediately on errors that can't succeed on retry, such as a bad key. The API key is read from `.env` or Colab Secrets and never printed.
- **Shared interface:** `answer(question, k=5)` returns the answer, its citations, and every retrieved source, so the evaluation scripts and the Colab notebook all use the same code path.
**Evaluation setup**
- **Retrieval set:** 30 questions across the 10 papers, each with evidence phrases from the paper (`eval/qa_pairs.json`). A retrieved chunk counts as correct if it comes from the expected paper and contains one of the evidence phrases.
- **Ground-truth set:** 40 questions from our Challenge Advisor, 4 per paper, each with a reference answer, expected section and key keywords (`data/ground_truth_qa.json`).
- **Metrics:** hit@1, hit@3 and hit@5 for the exact passage, MRR, paper hit@5 (any chunk from the right paper), and for generated answers: whether the right paper is cited and keyword coverage against the reference answer.
---
 
## 📈 **Results & Key Findings**
 
**Retrieval on our 30-question set** (k = 5, bge-small embeddings, ChromaDB):
 
| Configuration | Chunks | hit@1 | hit@3 | hit@5 | MRR | Paper hit@5 |
|---------------|-------:|------:|------:|------:|----:|------------:|
| Before ReAct text fix | 918 | 0.10 | 0.33 | 0.53 | 0.25 | 0.97 |
| **Current pipeline** | **918** | **0.10** | **0.33** | **0.57** | **0.26** | **0.97** |
| Bibliography chunks removed | 736 | 0.10 | 0.33 | 0.57 | 0.26 | 0.97 |
 
**Key findings**
- **Retrieval finds the right paper almost every time (97%), but not always the right passage (57%).** The exact passage is ranked first only 10% of the time, so the most useful improvement for November is retrieving and ranking the precise passage, which is also the target our Challenge Advisor set.
- **Cleaning matters.** Fixing the scrambled ReAct appendix text alone raised exact-passage retrieval by 4 points.
- **Bibliography chunks were not hurting retrieval** on this set. None appeared in the top 5 for any question, with or without filtering.
- **Answer-level results on the 40-question ground-truth set are in progress** (`eval/run_qa.py`).
---
 
## 🚀 **Next Steps**
 
**Current limitations**
- Exact-passage retrieval is 57% in the top 5, and the best passage is rarely ranked first.
- The evidence-phrase check is strict: a chunk that paraphrases the evidence counts as a miss, so these scores are a lower bound.
- Answer quality (citation accuracy, hallucinations, saying "I don't know") is not yet measured at scale.
**November plan**
- Add answer-level checks: citation correctness, unsupported claims, and correct refusals on questions the papers can't answer.
- Try semantic chunking and tune chunk size, overlap and breakpoint settings.
- Tune the prompt and the number of retrieved chunks (k) to improve citation accuracy.
- Run every configuration through the same evaluation and document the best one.
**With more time:** a local cross-encoder reranker, an agentic self-correction loop that rewrites the query when retrieved context is poor, and a simple Gradio interface for a human-in-the-loop review step.
 
---
 
## 📝 **License**
 
License to be confirmed with our Challenge Advisor.
 
---
 
## 📄 **References**
 
The 10 papers in the dataset:
 
1. Lewis et al., 2020. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
2. Gao et al., 2023. [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
3. Asai et al., 2023. [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
4. Yan et al., 2024. [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
5. Yao et al., 2022. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
6. Liu et al., 2023. [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
7. Es et al., 2023. [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
8. Hu et al., 2021. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
9. Dao et al., 2022. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
10. Karpukhin et al., 2020. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
Tools: [LangChain](https://python.langchain.com/), [ChromaDB](https://docs.trychroma.com/), [sentence-transformers](https://www.sbert.net/) with [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [Gemini API](https://ai.google.dev/), [tenacity](https://tenacity.readthedocs.io/).
 
---
 
## 🙏 **Acknowledgements**
 
Thank you to our Challenge Advisor, **Jose Ambrosio**, for the project design, the curated dataset and the ground-truth Q&A set, and for his guidance throughout; to our AI Studio Coach; and to the Break Through Tech AI Studio team.
