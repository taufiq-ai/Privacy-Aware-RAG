"""
This is a module that controls few functionalities.

1. Control and Routing (router.py):
Analyzes query intent and categorizes it into general, admin, product, order, existing customer support, etc. Then it determines whether to perform RAG or not and prioritizes the vector DB collections.

2. Metadata Extraction (extractor.py): 
Extracts named entities from queries based on vector DB metadata for dynamic filtering.

3. Search Optimization (coder.py): 
Generates chroma-db filter snippet and does query fusion.

4. Retrieval Verification (verifier.py): 
Evaluates retrieved knowledge and verifies data leakage is not happening.
"""