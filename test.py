from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import spacy
import pdfplumber
import json

# Define the state to track data across the workflow
class RFPState(TypedDict):
    rfp_path: str
    rfp_text: str
    qa_pairs: List[Dict[str, str]]
    response: str

# Initialize NLP and LLM models
nlp = spacy.load("en_core_web_sm")
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-small",  # Lightweight model for demo; replace with larger model or xAI API
    task="text2text-generation",
    pipeline_kwargs={"max_length": 512}
)

# Node 1: Extract text from RFP PDF
def extract_rfp_text(state: RFPState) -> RFPState:
    with pdfplumber.open(state["rfp_path"]) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    state["rfp_text"] = text
    return state

# Node 2: Generate Q&A from RFP text
def generate_qa_pairs(state: RFPState) -> RFPState:
    doc = nlp(state["rfp_text"])
    qa_pairs = []
    
    # Identify requirements (heuristic: sentences with "must", "shall", "required")
    requirements = [
        sent.text.strip() for sent in doc.sents
        if any(keyword in sent.text.lower() for keyword in ["must", "shall", "required"])
    ]
    
    # Use LLM to convert requirements into Q&A
    qa_prompt = PromptTemplate(
        input_variables=["requirement"],
        template="Convert this requirement into a question and answer pair:\nRequirement: {requirement}\nOutput format: Question: [Your question]\nAnswer: [Your answer]"
    )
    
    for req in requirements[:5]:  # Limit to 5 for demo; adjust as needed
        prompt = qa_prompt.format(requirement=req)
        qa_text = llm(prompt)
        # Parse LLM output (assuming it follows the format)
        try:
            q, a = qa_text.split("\n")
            question = q.replace("Question: ", "").strip()
            answer = a.replace("Answer: ", "").strip()
            qa_pairs.append({"question": question, "answer": answer})
        except:
            continue
    
    state["qa_pairs"] = qa_pairs
    return state

# Node 3: Store Q&A (for user review/UI display)
def store_qa_pairs(state: RFPState) -> RFPState:
    # Save Q&A as JSON for user review (and future UI integration)
    with open("rfp_qa.json", "w") as f:
        json.dump(state["qa_pairs"], f, indent=2)
    return state

# Node 4: Generate RFP response from Q&A
def generate_response(state: RFPState) -> RFPState:
    # Create a response template
    response_template = """
    Proposal Response to RFP
    ======================
    Thank you for the opportunity to submit a proposal. Below, we address the key requirements outlined in your RFP:

    {qa_responses}

    We are confident that our solution meets your needs and look forward to further discussions.
    """
    
    # Generate response content from Q&A
    qa_responses = "\n".join([
        f"**Q: {pair['question']}**\nA: {pair['answer']}"
        for pair in state["qa_pairs"]
    ])
    
    # Use LLM to refine the response
    response_prompt = PromptTemplate(
        input_variables=["template", "qa_responses"],
        template="Refine this proposal response to be professional and concise:\n{template}\nQA Responses:\n{qa_responses}"
    )
    response = llm(response_prompt.format(template=response_template, qa_responses=qa_responses))
    
    state["response"] = response
    return state

# Define the LangGraph workflow
workflow = StateGraph(RFPState)

# Add nodes
workflow.add_node("extract_rfp", extract_rfp_text)
workflow.add_node("generate_qa", generate_qa_pairs)
workflow.add_node("store_qa", store_qa_pairs)
workflow.add_node("generate_response", generate_response)

# Define edges
workflow.add_edge("extract_rfp", "generate_qa")
workflow.add_edge("generate_qa", "store_qa")
workflow.add_edge("store_qa", "generate_response")
workflow.add_edge("generate_response", END)

# Set entry point
workflow.set_entry_point("extract_rfp")

# Compile the graph
app = workflow.compile()

# Example usage
if __name__ == "__main__":
    # Initial state with path to RFP PDF
    initial_state = RFPState(rfp_path="sample_rfp.pdf")
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Print results
    print("Generated Q&A Pairs:")
    for pair in final_state["qa_pairs"]:
        print(f"Q: {pair['question']}\nA: {pair['answer']}\n")
    
    print("Generated RFP Response:")
    print(final_state["response"])
    
    # Save response for UI integration
    with open("rfp_response.txt", "w") as f:
        f.write(final_state["response"])