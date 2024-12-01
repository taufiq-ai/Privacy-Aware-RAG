# RAG for E-Commerce APP
Accepts: markdown, texts, JSON, Excel files

## WorkFlow:
1. Fetch Data such as text, md, excel.
2. Chunking data.
3. Create a VectorDB Collection with Embedding Function.
4. Add/Upsert data into Collection such as chunks*, ids*, metadata, embeddings.
5. Retrieve knowledge via query the VectorDB
6. Pass Retrieved knowledge to LLM Chatbot.

## Tools
1. Langchain: Chunking Data
2. ChromaDB: VectorStore
3. LLM_BE: Chatbot Engine