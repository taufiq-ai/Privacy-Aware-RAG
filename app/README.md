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

5. Building the full RAG LLMs pipeline for a single data file is a challege but adapting non-tech User in the end-to-end process is bigger challenge. 

6. VectorDB Collection timed out to early
```