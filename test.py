from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import pdfplumber
import json
import logging
import os
import re
import sys

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the state to track data across the workflow
class RFPState(TypedDict):
    rfp_path: str
    rfp_text: str
    qa_pairs: List[Dict[str, str]]
    response: str

# Verify dependencies
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
    use_spacy = True
except Exception as e:
    logger.warning(f"spaCy not available, using keyword-based extraction: {e}")
    use_spacy = False

try:
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        pipeline_kwargs={"max_length": 512, "truncation": True}
    )
    logger.info("LLM model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load LLM: {e}")
    raise

# Node 1: Extract text from RFP PDF
def extract_rfp_text(state: RFPState) -> RFPState:
    logger.info(f"Attempting to extract text from RFP: {state['rfp_path']}")
    if not os.path.exists(state["rfp_path"]):
        error_msg = (
            f"RFP file not found at: {state['rfp_path']}\n"
            "Please ensure 'sample_rfp.pdf' exists in the project directory "
            "(C:\\Users\\kckde\\Desktop\\apex_hackathon\\RFP-Automation-Agent)."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with pdfplumber.open(state["rfp_path"]) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if not text.strip():
                logger.warning("No text extracted from RFP. The PDF may contain scanned images or no selectable text.")
                state["rfp_text"] = ""
            else:
                logger.info(f"Extracted text: {text[:200]}...")
                state["rfp_text"] = text
                logger.info("RFP text extracted successfully")
    except Exception as e:
        logger.error(f"Error extracting RFP text: {e}")
        state["rfp_text"] = ""
    return state

# Node 2: Generate Q&A from RFP text
def generate_qa_pairs(state: RFPState) -> RFPState:
    logger.info("Generating Q&A pairs from RFP text")
    if not state["rfp_text"]:
        logger.warning("No RFP text available for Q&A generation")
        state["qa_pairs"] = []
        return state

    qa_pairs = []
    
    if use_spacy:
        doc = nlp(state["rfp_text"])
        requirements = [
            sent.text.strip() for sent in doc.sents
            if any(keyword in sent.text.lower() for keyword in ["must", "shall", "required"])
        ]
    else:
        sentences = re.split(r'[.!?]\s+', state["rfp_text"])
        requirements = [
            sent.strip() for sent in sentences
            if any(keyword in sent.lower() for keyword in ["must", "shall", "required"])
        ]
    
    logger.info(f"Found {len(requirements)} potential requirements: {requirements}")
    
    # Simplified prompt for better compatibility with flan-t5-small
    qa_prompt = PromptTemplate(
        input_variables=["requirement"],
        template="Turn this into a question and answer:\n{requirement}\nQuestion: What is the requirement for this?\nAnswer: {requirement}"
    )
    
    for req in requirements[:3]:
        try:
            # Clean up the requirement text for better Q&A generation
            cleaned_req = req.replace('\n-', '').strip()
            if not cleaned_req:
                continue
                
            prompt = qa_prompt.format(requirement=cleaned_req)
            qa_text = llm.invoke(prompt)  # Use invoke instead of deprecated __call__
            logger.info(f"LLM output for requirement '{cleaned_req[:50]}...': {qa_text}")
            
            # Try to parse the LLM output
            try:
                q, a = qa_text.split("\n")
                question = q.replace("Question: ", "").strip()
                answer = a.replace("Answer: ", "").strip()
                if question and answer:
                    qa_pairs.append({"question": question, "answer": answer})
                    logger.info(f"Generated Q&A: Q: {question[:50]}... A: {answer[:50]}...")
                else:
                    logger.warning(f"Empty question or answer for requirement: {cleaned_req[:50]}...")
                    # Fallback: Generate a basic Q&A pair
                    question = f"What is the requirement for {cleaned_req[:20].lower()}...?"
                    answer = cleaned_req
                    qa_pairs.append({"question": question, "answer": answer})
                    logger.info(f"Fallback Q&A: Q: {question[:50]}... A: {answer[:50]}...")
            except:
                logger.warning(f"Failed to parse Q&A for requirement: {cleaned_req[:50]}...")
                # Fallback: Generate a basic Q&A pair
                question = f"What is the requirement for {cleaned_req[:20].lower()}...?"
                answer = cleaned_req
                qa_pairs.append({"question": question, "answer": answer})
                logger.info(f"Fallback Q&A: Q: {question[:50]}... A: {answer[:50]}...")
        except Exception as e:
            logger.error(f"Error generating Q&A for requirement: {cleaned_req[:50]}... {e}")
            continue
    
    state["qa_pairs"] = qa_pairs
    logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
    return state

# Node 3: Store Q&A (for user review/UI display)
def store_qa_pairs(state: RFPState) -> RFPState:
    logger.info("Storing Q&A pairs")
    try:
        with open("rfp_qa.json", "w") as f:
            json.dump(state["qa_pairs"], f, indent=2)
        logger.info("Q&A pairs saved to rfp_qa.json")
    except Exception as e:
        logger.error(f"Error saving Q&A pairs: {e}")
    return state

# Node 4: Generate RFP response from Q&A
def generate_response(state: RFPState) -> RFPState:
    logger.info("Generating RFP response")
    if not state["qa_pairs"]:
        logger.warning("No Q&A pairs available for response generation")
        state["response"] = "No response generated due to missing Q&A pairs."
        try:
            with open("rfp_response.txt", "w") as f:
                f.write(state["response"])
            logger.info("Response saved to rfp_response.txt")
        except Exception as e:
            logger.error(f"Error saving response: {e}")
        return state

    response_template = """
    Proposal Response to RFP
    ======================
    Thank you for the opportunity to submit a proposal. Below, we address the key requirements outlined in your RFP:

    {qa_responses}

    We are confident that our solution meets your needs and look forward to further discussions.
    """
    
    qa_responses = "\n".join([
        f"**Q: {pair['question']}**\nA: {pair['answer']}"
        for pair in state["qa_pairs"]
    ])
    
    response_prompt = PromptTemplate(
        input_variables=["template", "qa_responses"],
        template="Refine this proposal response to be professional and concise:\n{template}\nQA Responses:\n{qa_responses}"
    )
    
    try:
        response = llm.invoke(response_prompt.format(template=response_template, qa_responses=qa_responses))
        state["response"] = response
        logger.info("RFP response generated successfully")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        state["response"] = "Error generating response."
    
    try:
        with open("rfp_response.txt", "w") as f:
            f.write(state["response"])
        logger.info("Response saved to rfp_response.txt")
    except Exception as e:
        logger.error(f"Error saving response: {e}")
    
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
    # Construct absolute path to sample_rfp.pdf
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rfp_path = os.path.join(script_dir, "sample_rfp1.pdf")
    logger.info(f"Looking for RFP file at: {rfp_path}")
    
    # Allow custom path via command-line argument
    if len(sys.argv) > 1:
        rfp_path = sys.argv[1]
        logger.info(f"Using custom RFP path from command line: {rfp_path}")
    
    initial_state = RFPState(rfp_path=rfp_path)
    
    try:
        final_state = app.invoke(initial_state)
        print("Generated Q&A Pairs:")
        for pair in final_state["qa_pairs"]:
            print(f"Q: {pair['question']}\nA: {pair['answer']}\n")
        print("Generated RFP Response:")
        print(final_state["response"])
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        print(f"Error running workflow: {e}")