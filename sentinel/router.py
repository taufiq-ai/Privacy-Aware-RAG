"""
1. Control and Routing (router.py):
Analyzes query intent and categorizes it into general, admin, product, order, existing customer support, etc. Then it determines whether to perform RAG or not and prioritizes the vector DB collections.

1.1. Classify user query into category.
1.2. Decides whether to Answer with RAG, without RAG, Do not Answer.
1.3. Shortlist the existing vector db collections name, based on Category.
"""

from app.llm_engine import classify_text

## NOTE: App Customer can decides categories and add description. AI can also suggest description.
def classify_query(
    query:str,
    categories: dict = {
        "chitchat": {
            "name": "chitchat",
            "label": 0,
            "description": "Basic conversational messages requiring no context"
        },
        "presales_enquiry": {
            "name": "presales_enquiry",
            "label": 1,
            "description": "Product/service inquiries from potential customers"
        },
        "customer_support": {
            "name": "customer_support",
            "label": 2,
            "description": "Existing customer order/service related questions"
        },
        "account_action": {
            "name": "account_action",
            "label": 3,
            "description": "Administrative requests requiring user authentication"
        },
        "flagged_content": {
            "name": "flagged_content",
            "label": 4,
            "description": "Messages violating content policies"
        },
    },
) -> dict:
    result = classify_text(text=query, classes=categories)
    paylod = {
        "class":result.get("class"),
        "label": result.get("label"),
        "confidence": result.get("confidence")
    }
    return paylod




