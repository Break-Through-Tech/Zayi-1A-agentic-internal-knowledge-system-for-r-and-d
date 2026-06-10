---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff only — remove before sharing with students)*

| Check | Status | Notes |
|-------|--------|-------|
| Python Compatibility | 🟢 | The tech stack is primarily centered on Python libraries, ensuring optimal compatibility with the ML engineering framework expected in the curriculum. |
| Data Readiness | 🟡 | Data consists of a subset of ArXiv papers and may require preprocessing, especially for PDFs, which could complicate the initial stages of the project. |
| Resource Check | 🟢 | Google Colab should suffice for the provided needs; no specialized hardware is necessary, and API keys can be managed by the organization. |

**Student Fit Score:** 7/10  
**Technical Depth Score:** 8/10  
**Overall Recommendation:** REVISE

**Advisor Feedback Draft:**
The project leverages foundational ML concepts creatively, but the team must ensure clarity in implementing semantic chunking and maintain alignment with the semester timeline. Consider simplifying agentic routing approaches to fit within student capabilities. The team should also establish robust fallback mechanisms for external API dependencies.

---

# Agentic Internal Knowledge System for R&D

**Company / Org:** Zayi  
**Challenge Advisor:** Jose Ambrosio, josentambrosio@outlook.com  
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About Zayi

Zayi is focused on leveraging cutting-edge technology to streamline research and development processes. We operate within the tech industry, specializing in knowledge management and automation solutions.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use a curated subset of publicly available ArXiv Machine Learning research papers and Agentic Retrieval-Augmented Generation (RAG) and semantic chunking to build an autonomous Internal Knowledge Agent that extracts, synthesizes, and drafts structured summary memos with citations from complex technical documentation. This will help our company address the time-intensive process of manually parsing dense PDFs and extracting architectural patterns for our R&D pipeline.

### Success Criteria
Functional local prototype in Google Colab; retrieval accuracy and identification of architectural patterns against ground-truth Q&A; production of structured, cited technical memos with minimal manual corrections.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Understanding | Explore dataset, handle missing values, document findings |
| **October** | Model Development | Train baseline model, experiment with approaches, iterate |
| **November** | Evaluation & Presentation | Finalize model, prepare presentation, document results |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** Curated subset of ArXiv Machine Learning research papers (JSON and PDF format) from Kaggle  
**Format:** JSON, PDF  
**Size:** 1gb to 5gb  
**Location:** [Link to dataset or instructions for accessing it]

### Key Details
- Curated subset of ArXiv Machine Learning research papers (JSON and PDF format) from Kaggle.
- Some PDFs may require preprocessing for effective extraction.
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP

**Recommended Libraries:**
- Agentic RAG
- semantic chunking
- ChromaDB
- LangChain
- Gemini API
- tenacity library
- LangGraph
- FlashRank
- Gradio

**Evaluation Metrics:**
- Accuracy, Retrieval accuracy, Structured memo quality

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [A Comprehensive Guide to Retrieval-Augmented Generation](https://example-link1.com)
- [Understanding Semantic Chunking in NLP](https://example-link2.com)

**Technical Tutorials:**
- [Tutorial on RAG Techniques](https://example-link3.com)
- [Documentation for LangChain](https://example-link4.com)

**Code Examples:**
- [Example implementation of Agentic RAG](https://example-link5.com)
- [Starter implementation using Gradio](https://example-link6.com)

**Other:**
- [Papers on NLP and RAG](https://example-link7.com)
- [Relevant podcasts discussing ML advancements](https://example-link8.com)

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Slack (Break Through Tech workspace)  
**Response time:** Within 48 hours on weekdays  

**Recommended Tools:**
- **Coding:** Google Colab
- **Collaboration:** GitHub, Notion
- **Virtual Meetings:** Zoom

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
