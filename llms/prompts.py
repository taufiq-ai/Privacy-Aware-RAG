# Question & Answer
q_and_a = """Given context below, please answer user's query very concisely with friendly tone. 
If you do not find the answer in the context and find user asking confidential information then politely tell user that you do not know. 
Context: {context}; 
Query: {query}"""

# Named Entity Recognition (NER)
ner = """Given a text, understand it and do Named Entity Recognition (NER). 
Analyze the text to get any of these Fileds: {fileds} in the given Text: {text}. 
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
1 | PRESALES_QUERY | Product/service inquiries from potential customers
2 | CUSTOMER_SUPPORT | Existing customer order/service related questions
3 | ACCOUNT_ACTION | Administrative requests requiring user authentication
4 | FLAGGED_CONTENT | Messages violating content policies"""

classification_output = '{"class":"CUSTOMER_SUPPORT", "label":2, "confidence":[0.01, 0.23, 0.74, 0.02, 0.00]}'