# Question & Answer
q_and_a = """You are a helpfull chatbot with access to information.
Given context below that includes available information, please answer user's query very concisely with friendly tone based on the given context. 
If you do not find the answer in the context then politely manage user that you do not know and suggest to call +8804896416.  
Context: {context}; 
Query: {query}"""

# Question & Answer
q_and_a_without_context = """Answer user's query very concisely with friendly tone. 
If you find user asking confidential information then politely tell user that you do not know. 

Query: {query}"""

# Named Entity Recognition (NER)
ner = """Given a text, understand it and do Named Entity Recognition (NER). 
Analyze the text to get any of these Fileds: {fileds} in the given Text: {text}. 
If value of any filed is not available in the above text, please do not include the named_entity in the output. 
Return NER result as a JSON. 
Example output: {example_output}.
Please do not write extra speech other than the expected output as JSON."
"""
ner_example_output = "{'field_1': 'value_1', 'filed_N':'value_N'}"

# Text Classification
classification = """Task: Text Classification.
Given a customer message below, classify the message into one of these following classes.
{classes}

Return the result as structured JSON.
Example Output: {example_output}

Given Message: {text}
"""

classification_classes = """Classes: 
Class label | Class Name | Details
0 | CHITCHAT | Basic conversational messages requiring no context
1 | PRESALES_ENQUIRY | Product/service inquiries from potential customers
2 | CUSTOMER_SUPPORT | Existing customer order/service related questions
3 | ACCOUNT_ACTION | Administrative requests requiring user authentication
4 | FLAGGED_CONTENT | Messages violating content policies"""

classification_output = '{"class":"CUSTOMER_SUPPORT", "label":2, "confidence":[0.01, 0.23, 0.74, 0.02, 0.00]}'

# DB Query Optimization
db_query_optimization = """Task: DB Query Filter Operation Optimization.
User Input: 
- text: {text}
- List of Fields: {fields}
- Named Entity Recognition Result: {ner_result}

Assistant Output:
- Chroma DB Metadata Filter Logic

Basic Query Syntax: collection.query(
    query_texts=["Is there any Ryzen 5 laptop under 500USD?"],
    n_results=2,
    where={"price": {"$lte": 500}, "product": {"$in": ["comptuter", "laptop"]}}
)"

Expected Output: {"price": {"$lte": 500}, "product": {"$in": ["comptuter", "laptop"]}}

Our system find the related context from database based on the given text.
You have to write the filter method to find the related context with less huedle.
You see the basic query syntax above. But you have to write the filter logic for the `where` parameter only. Follow, expected output format strictly.

Given a list of metadata fields based on which we can write the filter logic. Only include the named_entity in the filter code that has value in the ner_result. If any filed is not present in the ner_result or any ner named_entity is not present in the filed then do not include it in the filter code.

"""

db_filter = """Task: DB Query Filter Operation Optimization.
User Input: 
- text: {text}
- Fields: {fields}
- NER Result: {ner_result}

Assistant Output: Chroma DB Metadata Filter Logic as JSON. example format: {example_format}

Example Output: {example_output}

Rules:
1. Return only the filter logic 
2. Include only fields present in both fields list and NER results
"""
db_filter_example_output = """{"price": {"$lte": 500}, "product": {"$in": ["comptuter", "laptop"]}}"""
db_filter_example_format = """{"filed_name": {"$operator": "value"}}"""