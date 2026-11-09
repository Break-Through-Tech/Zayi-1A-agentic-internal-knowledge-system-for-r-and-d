# Zayi Agentic Internal Knowledge System for R&D

An AI system that reads a set of machine learning research papers and drafts short, cited summary memos, so engineers can get the architectural patterns and answers they need without manually combing through dense technical PDFs. Built with Break Through Tech AI Studio for our host company, **Zayi**.

---

### 👥 **Team Members**

| Name                | GitHub Handle | Contribution                                                                   |
|---------------------|---------------|--------------------------------------------------------------------------------|
| Nivedha Aravindmali | @NvyGreen     | Generation step of the RAG pipeline, citation formatting, retrieval evaluation |
| Tanish Gupta        | @tanish2007   | Embedding and vector storage (ChromaDB), retriever                             |
| Ivan Ng             | @cybercommits | Data cleaning, Gemini client with retry/backoff, evaluation harness            |
| Godwin Idowu        | @Gidowu       | Test and retrieval pipeline, chunking support                                  |

---

## 🎯 **Project Highlights**

- Built a **Retrieval-Augmented Generation (RAG)** pipeline over a curated set of **10 foundational ML research papers** that drafts structured, cited summary memos.
- The system answers **only from retrieved sources** and cites the source paper and page for every claim, directly targeting a citation-precision goal so a reviewing engineer can trust the output.
- Evaluated against **ground-truth Q&A pairs**, measuring retrieval accuracy (hit@1/3/k, MRR) and whether generated answers cite the correct source.
- Reduces the time Zayi's engineers spend manually parsing dense PDFs to extract architectural patterns for their R&D pipeline.

---

## 👩🏽‍💻 **Setup and Installation**

Reproduce the project locally:
 
```bash
# 1. Clone the repository
git clone https://github.com/Break-Through-Tech/zayi-1a-agentic-internal-knowledge-system-for-r-and-d.git
cd zayi-1a-agentic-internal-knowledge-system-for-r-and-d
 
# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Set your Gemini API key (create a .env file at the repo root)
echo "GOOGLE_API_KEY=your_key_here" > .env
 
# 4. Build the vector store from the chunked data
python src/vector_store.py build
 
# 5. Ask a question through the full pipeline
python src/rag_chain.py "What does LoRA stand for?" -k 5
```
 
**Evaluate the model:**
 
```bash
# Retrieval metrics (hit@k, MRR) across configurations
python eval/evaluate_retrieval.py -k 5
 
# Full-pipeline answers + citation scoring against the ground-truth Q&A set
python eval/run_qa.py --limit 3     # try 3 questions first
python eval/run_qa.py               # all 40 questions
```
 
**Run the tests:**
 
```bash
pytest
```

---

## 🏗️ **Project Overview**

This project was built as part of the **Break Through Tech AI Studio** program, which pairs applied machine learning training with a real, industry-sponsored project.
 
Our host company, **Zayi**, is a research-led innovation hub that builds knowledge-management and automation tools. Their engineers spend significant time reading dense technical papers to find a handful of relevant architectural patterns. Our objective was to automate a **"Draft Triage Memo"** workflow: given a library of ML research papers, produce a short, cited memo that answers a specific technical question, grounded entirely in the source documents.
 
The real-world significance: trustworthy, source-cited summaries turn a slow, manual research process into a fast, reliable one, letting engineers act on research with the citations to back it up.

---

## 📊 **Data Exploration**

- **Dataset:** a curated subset of **10 foundational ArXiv machine learning papers** (RAG, Self-RAG, CRAG, ReAct, LoRA, FlashAttention, DPR, and others), provided as pre-extracted JSON text plus full-text PDFs (~18 MB total).
- **Preprocessing:** cleaned and standardized the paper text (fixing encoding artifacts and garbled characters), then split it into overlapping ~1000-character chunks (**recursive character splitting**) so each chunk carries its source paper, page number, and position for accurate citations.
- **Storage:** chunks are embedded with a sentence-transformer model and stored in a local **ChromaDB** vector database for semantic retrieval.
- **Evaluation data:** the central ground-truth set of **40 Q&A pairs** (`data/ground_truth_qa.json`) — each mapping a question to its expected source paper, section, and key keywords — used for full-pipeline answer and citation scoring, plus a 30-pair set with evidence phrases (`eval/qa_pairs.json`) currently used for retrieval metrics.

Data exploration and preprocessing are documented in the notebooks under `notebooks/`.

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

We evaluate the pipeline against the ground-truth Q&A set using:
 
- **Retrieval accuracy** — hit@1, hit@3, hit@k, and MRR (does the correct source chunk come back).
- **Citation correctness** — does the answer cite a source from the expected paper, and does every cited marker map to a real retrieved chunk.
- **Abstention** — when the sources don't contain the answer, does the model correctly say so.
> _Fill in your final numbers here once November evaluation is complete (e.g., hit@5 = X%, citation accuracy = Y%)._
 
Evaluation scripts: `eval/evaluate_retrieval.py` (retrieval metrics) and `eval/run_qa.py` (full-pipeline answers + citation scoring).

---

## 🙋🏽 Individual Contributions
 
**Nivedha Aravindmali (@NvyGreen):**
- Built the **generation step** of the RAG pipeline: the prompt template that answers only from retrieved context and cites the source paper and page for each claim (`src/generation.py`).
- Designed the **citation formatting** so every claim traces back to a specific source, supporting the system's citation-precision goal.
- Contributed to **retrieval evaluation**, measuring accuracy against the ground-truth Q&A set.

> Each teammate: add a short bullet list of your own contributions here.
 
---

## 🚀 **Next Steps**

- Add **agentic self-correction** (LangGraph): an evaluator node that re-queries when retrieved context looks irrelevant.
- Add **local reranking** (e.g., FlashRank) to improve context relevance without extra API calls.
- Build a lightweight **Gradio UI** for the human-in-the-loop memo approval step.
- Switch chunking from recursive to **semantic chunking** and compare retrieval accuracy against the current approach.
- Evaluate all metrics (retrieval and citation) on the full 40-pair ground-truth set.
- Expand the paper corpus and the ground-truth Q&A set.

---

## 📁 Repository Structure
 
```
data/        Curated papers, cleaned text, chunks, and the ground-truth Q&A set
notebooks/   Data exploration, chunking, and embedding/storage notebooks
src/         Pipeline: cleaning, chunking, vector store, retrieval, generation, config, Gemini client
eval/        Retrieval and full-pipeline evaluation scripts
tests/       Pytest suite for the pipeline and components
```
 
---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
