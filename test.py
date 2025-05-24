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
import pytesseract
from PIL import Image
import io

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
            "Please ensure 'sample_rfp.pdf' exists in the project directory."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        text = ""
        with pdfplumber.open(state["rfp_path"]) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    # Fallback: Try OCR if text extraction fails (e.g., for scanned PDFs)
                    logger.info("No text extracted from page, attempting OCR...")
                    try:
                        # Convert page to image for OCR
                        page_image = page.to_image(resolution=300)
                        img_byte_arr = io.BytesIO()
                        page_image.original.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        img = Image.open(img_byte_arr)
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text:
                            text += ocr_text + "\n"
                            logger.info(f"OCR extracted text: {ocr_text[:100]}...")
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed for page: {ocr_error}")
                        continue
        
        if not text.strip():
            logger.warning("No text extracted from RFP after attempting OCR.")
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

    all_qa_pairs = []
    
    # Step 1: Extract potential requirements
    if use_spacy:
        doc = nlp(state["rfp_text"])
        requirements = [
            sent.text.strip() for sent in doc.sents
            if any(keyword in sent.text.lower() for keyword in ["must", "shall", "required", "scope", "resource", "location", "capacity", "timeline", "duration"])
        ]
    else:
        sentences = re.split(r'[.!?]\s+', state["rfp_text"])
        requirements = [
            sent.strip() for sent in sentences
            if any(keyword in sent.lower() for keyword in ["must", "shall", "required", "scope", "resource", "location", "capacity", "timeline", "duration"])
        ]
    
    # Step 2: Look for numbered clauses (e.g., "1.1.1") to capture more details
    clause_pattern = r'\d+\.\d+(\.\d+)?\s+.*?(?=\n\d+\.\d+|\n[A-Z]|\Z)'
    clauses = re.findall(clause_pattern, state["rfp_text"], re.DOTALL)
    requirements.extend([clause.strip() for clause in clauses if clause.strip()])
    
    # Remove duplicates while preserving order
    seen = set()
    requirements = [req for req in requirements if not (req in seen or seen.add(req))]
    
    logger.info(f"Found {len(requirements)} potential requirements: {requirements[:3]}...")

    # Step 3: Define multiple prompt templates for different types of questions
    qa_prompts = {
        "scope": PromptTemplate(
            input_variables=["requirement"],
            template="Generate a question and answer about the scope of work:\nRequirement: {requirement}\nQuestion: What is the scope of work related to this requirement?\nAnswer: {requirement}"
        ),
        "resource": PromptTemplate(
            input_variables=["requirement"],
            template="Generate a question and answer about the resources needed:\nRequirement: {requirement}\nQuestion: What resources are required for this?\nAnswer: {requirement}"
        ),
        "location": PromptTemplate(
            input_variables=["requirement"],
            template="Generate a question and answer about the work location:\nRequirement: {requirement}\nQuestion: Where must this work be performed?\nAnswer: {requirement}"
        ),
        "timeline": PromptTemplate(
            input_variables=["requirement"],
            template="Generate a question and answer about the timeline:\nRequirement: {requirement}\nQuestion: What is the timeline for this requirement?\nAnswer: {requirement}"
        ),
        "general": PromptTemplate(
            input_variables=["requirement"],
            template="Generate a question and answer:\nRequirement: {requirement}\nQuestion: What is the requirement for this?\nAnswer: {requirement}"
        )
    }
    
    # Step 4: Generate Q&A pairs for all requirements
    for idx, req in enumerate(requirements):
        try:
            # Clean up the requirement text
            cleaned_req = req.replace('\n-', '').strip()
            if not cleaned_req:
                continue
                
            # Determine the type of prompt to use based on keywords
            prompt_type = "general"
            if any(keyword in cleaned_req.lower() for keyword in ["scope", "work", "responsibility"]):
                prompt_type = "scope"
            elif any(keyword in cleaned_req.lower() for keyword in ["resource", "cost", "security", "deposit"]):
                prompt_type = "resource"
            elif any(keyword in cleaned_req.lower() for keyword in ["location", "place", "address", "grid"]):
                prompt_type = "location"
            elif any(keyword in cleaned_req.lower() for keyword in ["timeline", "duration", "date", "period"]):
                prompt_type = "timeline"

            prompt = qa_prompts[prompt_type].format(requirement=cleaned_req)
            qa_text = llm.invoke(prompt)
            logger.info(f"LLM output for requirement {idx + 1} ('{cleaned_req[:50]}...'): {qa_text}")
            
            # Parse the LLM output
            try:
                q, a = qa_text.split("\n")
                question = q.replace("Question: ", "").strip()
                answer = a.replace("Answer: ", "").strip()
                if question and answer:
                    all_qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "prompt_type": prompt_type,
                        "requirement": cleaned_req
                    })
                    logger.info(f"Generated Q&A {idx + 1}: Q: {question[:50]}... A: {answer[:50]}...")
                else:
                    logger.warning(f"Empty question or answer for requirement {idx + 1}: {cleaned_req[:50]}...")
                    # Fallback
                    question = f"What is the detail for {cleaned_req[:20].lower()}...?"
                    answer = cleaned_req
                    all_qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "prompt_type": prompt_type,
                        "requirement": cleaned_req
                    })
                    logger.info(f"Fallback Q&A {idx + 1}: Q: {question[:50]}... A: {answer[:50]}...")
            except:
                logger.warning(f"Failed to parse Q&A for requirement {idx + 1}: {cleaned_req[:50]}...")
                # Fallback
                question = f"What is the detail for {cleaned_req[:20].lower()}...?"
                answer = cleaned_req
                all_qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "prompt_type": prompt_type,
                    "requirement": cleaned_req
                })
                logger.info(f"Fallback Q&A {idx + 1}: Q: {question[:50]}... A: {answer[:50]}...")
        except Exception as e:
            logger.error(f"Error generating Q&A for requirement {idx + 1}: {cleaned_req[:50]}... {e}")
            continue

    # Step 5: Save all Q&A pairs for reference
    try:
        with open("all_rfp_qa.json", "w") as f:
            json.dump(all_qa_pairs, f, indent=2)
        logger.info(f"Saved {len(all_qa_pairs)} Q&A pairs to all_rfp_qa.json")
    except Exception as e:
        logger.error(f"Error saving all Q&A pairs: {e}")

    # Step 6: Prioritize and limit Q&A pairs for the final response
    prioritized_pairs = []
    priority_types = ["scope", "resource", "location", "timeline"]
    
    # First, add pairs from priority categories
    for prompt_type in priority_types:
        for pair in all_qa_pairs:
            if pair["prompt_type"] == prompt_type and len(prioritized_pairs) < 30:
                prioritized_pairs.append({
                    "question": pair["question"],
                    "answer": pair["answer"]
                })
    
    # If we still have room, add general pairs
    for pair in all_qa_pairs:
        if pair["prompt_type"] == "general" and len(prioritized_pairs) < 30:
            prioritized_pairs.append({
                "question": pair["question"],
                "answer": pair["answer"]
            })

    state["qa_pairs"] = prioritized_pairs
    logger.info(f"Selected {len(prioritized_pairs)} prioritized Q&A pairs for the response")
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

# Node 4: Generate RFP response as a PDF
def generate_response(state: RFPState) -> RFPState:
    logger.info("Generating RFP response as a PDF")
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

    # Constructing the LaTeX document for the RFP response
    latex_content = r"""
\documentclass[a4paper,12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{titling}
\usepackage{parskip}
\usepackage{xcolor}
\usepackage{times}

\title{RFP Response: Procurement of Power Project}
\author{Utility Company}
\date{May 24, 2025}

\begin{document}

\begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Huge \textbf{RFP Response: Procurement of Power Project}\par}
    \vspace{1cm}
    {\Large \textbf{Utility Company}\par}
    \vspace{0.5cm}
    {\large Submitted to: Bidder Organization\par}
    \vspace{0.5cm}
    {\large May 24, 2025\par}
    \vfill
\end{titlepage}

\section*{Introduction}
We, the Utility Company, are pleased to submit this response to your Request for Proposal (RFP) for the procurement of power through a Public Private Partnership (PPP) on a Finance, Own, and Operate (FOO) basis. Our organization is committed to ensuring a reliable and sustainable electricity supply to meet the growing demands of our region. This response outlines our approach to fulfilling the requirements set forth in your RFP, detailing our methodology, implementation plan, resource allocation, and pricing structure. We aim to establish a collaborative partnership that ensures the successful execution of this power project.

\section*{Objective Statement}
The primary objective of this response is to demonstrate our capability to finance, construct, operate, and maintain a Power Station that delivers a contracted capacity of *** MW for a period of 5 years, as specified in your RFP (Clause 1.1.1). We intend to supply electricity during peak hours—2 hours up to or before 10:00 AM and 4 hours from or after 5:00 PM—to support the Utility’s distribution network. Our goal is to provide a cost-effective, reliable, and environmentally sustainable solution that aligns with the RFP’s requirements and fosters long-term energy security.

\section*{Readout of Requirements}
Below is a summary of the key requirements extracted from your RFP, ensuring we fully understand and address your expectations:

\begin{itemize}[leftmargin=*]
"""
    # Add Q&A pairs to the requirements section
    for pair in state["qa_pairs"]:
        latex_content += f"    \\item \\textbf{{Q: {pair['question']}}} \\\\ A: {pair['answer']} \n"

    latex_content += r"""
\end{itemize}

\section*{Methodology}
Our approach to fulfilling the RFP requirements involves a structured methodology to ensure the successful financing, construction, operation, and maintenance of the Power Station. The methodology is divided into the following phases:

\begin{enumerate}
    \item \textbf{Financing and Planning}: Secure funding through a combination of equity and debt financing, ensuring compliance with the RFP’s financial requirements such as the Bid Security of Rs. 5 lakh per MW (Clause 1.2.4). We will establish a project management office (PMO) to oversee planning, risk assessment, and stakeholder coordination.
    \item \textbf{Site Selection and Design}: Identify an optimal site for the Power Station, ensuring proximity to the grid point specified in Clause 25 of Appendix-I for efficient electricity delivery. The design phase will involve engineering a Power Station capable of delivering *** MW, with infrastructure to support peak-hour supply (2 hours before 10:00 AM and 4 hours after 5:00 PM).
    \item \textbf{Construction}: Construct the Power Station using modular construction techniques to accelerate timelines. We will deploy solar and wind energy systems to meet sustainability goals, supplemented by battery storage to ensure reliability during peak hours.
    \item \textbf{Operation and Maintenance}: Operate the Power Station with a dedicated team, ensuring 24/7 monitoring and maintenance. We will implement predictive maintenance using IoT sensors to minimize downtime and ensure consistent electricity supply to the Utility’s grid.
    \item \textbf{Grid Integration and Delivery}: Integrate the Power Station with the grid at the RLDC/SLDC-specified point, managing transmission charges and losses as per Clause 26 of Appendix-I. We will use advanced grid synchronization technologies to ensure seamless electricity delivery.
\end{enumerate}

\section*{Implementation Plan}
The implementation plan outlines the key milestones and timelines for the project, ensuring delivery within the *** months specified in Clause 1.1.1 from the date of the RFQ.

\begin{table}[h]
    \centering
    \begin{tabular}{|l|p{8cm}|c|}
        \hline
        \textbf{Phase} & \textbf{Activities} & \textbf{Timeline} \\
        \hline
        Financing and Planning & Secure funding, establish PMO, conduct risk assessment & Months 1--3 \\
        \hline
        Site Selection and Design & Site surveys, engineering design, permitting & Months 4--6 \\
        \hline
        Construction & Build Power Station, install solar/wind systems, battery storage & Months 7--12 \\
        \hline
        Testing and Commissioning & System testing, grid integration, trial runs & Months 13--14 \\
        \hline
        Operation and Maintenance & Begin electricity supply, ongoing monitoring & Month 15 onwards \\
        \hline
    \end{tabular}
    \caption{Implementation Timeline}
\end{table}

\section*{Resources Needed by Role and Phase}
The project requires a diverse team across different phases, with specific roles and responsibilities:

\begin{itemize}
    \item \textbf{Financing and Planning (Months 1--3)}:
        \begin{itemize}
            \item Project Manager (1): Oversees planning and coordination.
            \item Financial Analyst (2): Secures funding and manages budgets.
            \item Legal Advisor (1): Ensures compliance with RFP and regulatory requirements.
        \end{itemize}
    \item \textbf{Site Selection and Design (Months 4--6)}:
        \begin{itemize}
            \item Civil Engineer (2): Conducts site surveys and designs infrastructure.
            \item Electrical Engineer (2): Designs power generation and grid integration systems.
            \item Environmental Consultant (1): Ensures environmental compliance.
        \end{itemize}
    \item \textbf{Construction (Months 7--12)}:
        \begin{itemize}
            \item Construction Manager (1): Manages on-site construction activities.
            \item Construction Workers (20): Build infrastructure and install systems.
            \item Equipment Operators (5): Operate heavy machinery for construction.
        \end{itemize}
    \item \textbf{Testing and Commissioning (Months 13--14)}:
        \begin{itemize}
            \item Testing Engineer (3): Conducts system tests and trial runs.
            \item Grid Integration Specialist (2): Ensures seamless grid connection.
        \end{itemize}
    \item \textbf{Operation and Maintenance (Month 15 onwards)}:
        \begin{itemize}
            \item Operations Manager (1): Oversees daily operations.
            \item Maintenance Technicians (5): Perform regular and predictive maintenance.
            \item Data Analyst (1): Monitors performance using IoT data.
        \end{itemize}
\end{itemize}

\section*{Detailed Pricing}
The pricing is broken down by resource and technology, reflecting the costs associated with each phase of the project. All figures are in INR (Indian Rupees).

\begin{table}[h]
    \centering
    \begin{tabular}{|l|l|r|}
        \hline
        \textbf{Category} & \textbf{Item} & \textbf{Cost (INR)} \\
        \hline
        \multicolumn{3}{|c|}{\textbf{Human Resources}} \\
        \hline
        Project Manager & 1 person, 15 months & 1,800,000 \\
        Financial Analyst & 2 people, 3 months & 600,000 \\
        Legal Advisor & 1 person, 3 months & 300,000 \\
        Civil Engineer & 2 people, 3 months & 600,000 \\
        Electrical Engineer & 2 people, 3 months & 600,000 \\
        Environmental Consultant & 1 person, 3 months & 300,000 \\
        Construction Manager & 1 person, 6 months & 720,000 \\
        Construction Workers & 20 people, 6 months & 3,600,000 \\
        Equipment Operators & 5 people, 6 months & 900,000 \\
        Testing Engineer & 3 people, 2 months & 360,000 \\
        Grid Integration Specialist & 2 people, 2 months & 400,000 \\
        Operations Manager & 1 person, 12 months & 1,200,000 \\
        Maintenance Technicians & 5 people, 12 months & 3,000,000 \\
        Data Analyst & 1 person, 12 months & 720,000 \\
        \hline
        \multicolumn{3}{|c|}{\textbf{Technology and Equipment}} \\
        \hline
        Solar Panels & 500 kW capacity & 25,000,000 \\
        Wind Turbines & 500 kW capacity & 30,000,000 \\
        Battery Storage & 200 kWh capacity & 10,000,000 \\
        IoT Sensors & Monitoring system & 2,000,000 \\
        Grid Integration Tech & Synchronization equipment & 5,000,000 \\
        Construction Equipment & Heavy machinery rental & 5,000,000 \\
        \hline
        \multicolumn{2}{|r|}{\textbf{Total Estimated Cost}} & \textbf{91,110,000} \\
        \hline
    \end{tabular}
    \caption{Detailed Pricing Breakdown}
\end{table}

\section*{Conclusion}
We are confident that our proposed approach to the power project meets the requirements outlined in your RFP. By leveraging a combination of sustainable energy technologies, a skilled workforce, and a well-defined implementation plan, we aim to deliver a reliable electricity supply of *** MW for 5 years, ensuring peak-hour availability as specified. Our competitive pricing and commitment to quality make us a strong partner for this initiative. We look forward to the opportunity to collaborate with your organization and contribute to the region’s energy needs.

\end{document}
"""

    # Save the LaTeX content to a file
    try:
        with open("rfp_response.tex", "w") as f:
            f.write(latex_content)
        logger.info("LaTeX file 'rfp_response.tex' generated successfully")
    except Exception as e:
        logger.error(f"Error saving LaTeX file: {e}")
        state["response"] = "Error generating LaTeX file."
        return state

    # The LaTeX file will be compiled into a PDF using latexmk as per guidelines
    state["response"] = "RFP response generated as 'rfp_response.tex'. This will be compiled into a PDF."
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
    rfp_path = os.path.join(script_dir, "sample_rfp.pdf")
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