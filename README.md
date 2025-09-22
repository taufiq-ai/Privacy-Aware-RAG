# RAG for E-Commerce/F-Commerce APPs
Accepts: markdown, texts, JSON, Excel files

# Story
We all have personal experience with customer support chatbots in e-commerce, f-commerce, or s-commerce platforms. Have you ever wondered why these platforms still use outdated chatbots that barely understand user queries or just answer predefined questions? What are the difficulties in implementing the RAG chatbot in production that can answer general questions, product/service queries, and user-specific order queries? 

Actually, it comes with some major challenges like latency, accuracy, and data privacy. This type of RAG chatbot should have access to the vector DB of orders, products, services, etc. Where the first concern is if user A pretends to be user B, the chatbot might leak the data of user B. Another concern is that searching through multiple vector DB collections and lots of records might significantly increase latency and decrease accuracy. Although vector databases support filtering with metadata, in an automated system, we can not determine which metadata is present in each query—other than some fixed metadata such as user ID, time, etc.

In this project, I'll demonstrate the following LLM-based techniques using an opensource `EXAONE-3.5-2.4B` model and other commercial APIs to address these key challenges:
**1. Control and Routing:** Analyzes query intent and categorizes it into general, admin, product, order, existing customer support, etc. Then it determines whether to perform RAG or not and prioritizes the vector DB collections.
**2. Metadata Extraction:** Extracts named entities from queries based on vector DB metadata for dynamic filtering.
**3. Search Optimization:** Generates ChromaDB filter snippet and does query fusion.
**4. Retrieval Verification:** Evaluates retrieved knowledge and verifies data leakage is not happening.

**Key takeways:** how to tackle the key challenges in E/F-commerce data and build a production-ready RAG chatbot. 


# Installation
```
git clone https://github.com/taufiq-ai/Privacy-Aware-RAG.git
cd Privacy-Aware-RAG
uv sync
cp .env.example .env
uv run main.py
```

## WorkFlow:
1. Fetch Data such as text, md, excel.
2. Chunking data.
3. Create a VectorDB Collection with Embedding Function.
4. Add/Upsert data into Collection such as chunks*, ids*, metadata, embeddings.
5. Retrieve knowledge via query the VectorDB
6. Pass Retrieved knowledge to LLM Chatbot.
7. Evaluate privacy breaking
8. Response User

## Tools
1. Langchain: Chunking Data
2. ChromaDB: VectorStore
3. LLM_BE: Chatbot Engine

---

## RAG Improvement
### Initial Improvement Area
```
1. After user submit query:
Instead direct query the vector store using default distance metrics,
Filter vector store using query.
For writing vectorstore filter query, we can use llm.

2. When user chat: Add previous query and chat in the messages[] as context history.

3. The retrievar has only 3 records from xl, need more records on related products. So, filter is must.
Now even if we have 10 Asus products, the RAG return k=3, so the LLM think we only have 3 ASUS products. 

4. LLMs answer should be finetuned. 

5. Which XL columns should be used in RAG documents, Which columns should be for metadata (further query) should be well decided.
```
### Bigger Challenges
```
6. Building the full RAG LLMs pipeline for a single data file is a challege but adapting non-tech User in the end-to-end process is bigger challenge. 

7. VectorDB Collection timed out to early
```

## Update
1. NER can help to get query keys and value;
