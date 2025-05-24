import os
import json
import logging
import shutil
import traceback
import re
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
import pdfplumber
from PIL import Image
import io
import sys
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

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

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyA7QFDytlNKzi_dOcTrriljfUiDpbXVQq4"
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-1.5-pro"  # Using Gemini 1.5 Pro as default
if not GEMINI_API_KEY:
    logger.error("Gemini API key not found. Please set GEMINI_API_KEY in environment or .env file.")
    raise ValueError("Gemini API key is required.")
logger.info(f"Using Gemini API key: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-4:]}")
logger.info(f"Using model: {GEMINI_MODEL}")

def call_gemini_api(prompt: str, model: str = GEMINI_MODEL, max_retries: int = 3) -> str:
    """Call Google Gemini API with retry logic for rate limits and errors."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 512
        }
    }

    logger.info(f"Sending payload to Gemini API: {json.dumps(payload, indent=2)}")
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            # Check if the response contains the expected structure
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                logger.error(f"Unexpected Gemini API response format: {result}")
                return ""
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning(f"Rate limit hit, retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
            elif response.status_code == 401:
                logger.error("Invalid API key. Please verify GEMINI_API_KEY.")
                return ""
            elif response.status_code == 400:
                logger.error(f"Gemini API error: {e}, Response: {response.text}")
                return ""
            else:
                logger.error(f"Gemini API error: {e}, Response: {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return ""
    logger.error("Max retries exceeded for Gemini API.")
    return ""

# Node 1: Extract text from RFP PDF
def extract_rfp_text(state: RFPState) -> RFPState:
    logger.info(f"Attempting to extract text from RFP: {state['rfp_path']}")
    if not os.path.exists(state["rfp_path"]):
        error_msg = (
            f"RFP file not found at: {state['rfp_path']}\n"
            "Please ensure 'apexneural_rfp.pdf' exists in the project directory."
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
                    logger.info("No text extracted from page, skipping OCR for simplicity.")
                    continue
        
        if not text.strip():
            logger.warning("No text extracted from RFP.")
            state["rfp_text"] = ""
        else:
            logger.info(f"Extracted text: {text[:200]}...")
            state["rfp_text"] = text
            logger.info("RFP text extracted successfully")
    except Exception as e:
        logger.error(f"Error extracting RFP text: {e}")
        state["rfp_text"] = ""
    return state

# Node 2: Generate Q&A from RFP text using Gemini API
def generate_qa_pairs(state: RFPState) -> RFPState:
    logger.info("Generating Q&A pairs from RFP text using Gemini API")
    if not state["rfp_text"]:
        logger.warning("No RFP text available for Q&A generation")
        state["qa_pairs"] = []
        return state

    all_qa_pairs = []
    max_pairs = 50
    
    # Step 1: Extract explicit RFP questions (Q1–Q15)
    question_pattern = r'Q\d+\s+([^\n?]+\?)'
    explicit_questions = re.findall(question_pattern, state["rfp_text"], re.MULTILINE)
    # Normalize spacing in extracted questions
    explicit_questions = [re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', q).replace('  ', ' ') for q in explicit_questions]
    logger.info(f"Found {len(explicit_questions)} explicit RFP questions: {explicit_questions[:3]}")

    # Step 2: Generate answers for explicit questions
    for idx, question in enumerate(explicit_questions, 1):
        if len(all_qa_pairs) >= max_pairs:
            break
        try:
            prompt = (
                f"Given this RFP question from ApexNeural's AI automation platform RFP: '{question}', "
                "provide a detailed, complete answer demonstrating technical expertise. "
                "Ensure the answer is specific, avoids truncation, and does not contain '...' placeholders. "
                "Ensure proper spacing between words and sentences for readability. "
                "Context: The RFP requires a cloud-based platform for neural network training with TensorFlow/PyTorch integration, "
                "6-month timeline, $500,000–$1,000,000 budget, GDPR/CCPA compliance. "
                "Format as:\nQuestion: [Question]\nAnswer: [Answer]"
            )
            qa_text = call_gemini_api(prompt)
            logger.info(f"Gemini API output for Q{idx} ('{question[:50]}'): {qa_text[:100]}")
            
            # Parse Gemini output
            try:
                lines = qa_text.split("\n")
                if len(lines) >= 2 and lines[0].startswith("Question: ") and lines[1].startswith("Answer: "):
                    parsed_question = lines[0].replace("Question: ", "").strip()
                    answer = lines[1].replace("Answer: ", "").strip()
                    # Normalize spaces in the answer
                    answer = re.sub(r'\s+', ' ', answer).strip()
                    if parsed_question and answer and "..." not in answer:
                        prompt_type = "general"
                        if any(kw in question.lower() for kw in ["scope", "platform", "work", "dashboard"]):
                            prompt_type = "scope"
                        elif any(kw in question.lower() for kw in ["timeline", "milestone"]):
                            prompt_type = "timeline"
                        elif any(kw in question.lower() for kw in ["resource", "cost", "team", "infrastructure"]):
                            prompt_type = "resource"
                        elif any(kw in question.lower() for kw in ["compliance", "security"]):
                            prompt_type = "compliance"
                        elif any(kw in question.lower() for kw in ["cost", "pricing"]):
                            prompt_type = "pricing"
                        
                        all_qa_pairs.append({
                            "question": parsed_question,
                            "answer": answer,
                            "prompt_type": prompt_type,
                            "requirement": question
                        })
                        logger.info(f"Generated Q&A {idx}: Q: {parsed_question[:50]} A: {answer[:50]}")
                    else:
                        raise ValueError("Invalid or incomplete Q&A")
                else:
                    raise ValueError("Incorrect format")
            except Exception as e:
                logger.warning(f"Failed to parse Gemini Q&A for Q{idx}: {question[:50]}: {e}")
                answer = f"Response to '{question}' requires further clarification from ApexNeural."
                all_qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "prompt_type": "general",
                    "requirement": question
                })
                logger.info(f"Fallback Q&A {idx}: Q: {question[:50]} A: {answer[:50]}")
        except Exception as e:
            logger.error(f"Error generating Q&A for Q{idx}: {question[:50]}: {e}")
            continue

    # Step 3: Extract additional requirements
    if use_spacy:
        doc = nlp(state["rfp_text"])
        requirements = [
            sent.text.strip() for sent in doc.sents
            if any(keyword in sent.text.lower() for keyword in [
                "must", "shall", "required", "scope", "resource", "location", "capacity", 
                "timeline", "duration", "budget", "compliance", "deliverable", "dashboard"
            ])
        ]
    else:
        sentences = re.split(r'[.!?]\s+', state["rfp_text"])
        requirements = [
            sent.strip() for sent in sentences
            if any(keyword in sent.lower() for keyword in [
                "must", "shall", "required", "scope", "resource", "location", "capacity", 
                "timeline", "duration", "budget", "compliance", "deliverable", "dashboard"
            ])
        ]
    
    clause_pattern = r'\d+\.\d+(\.\d+)?\s+.*?(?=\n\d+\.\d+|\n[A-Z]|\Z)'
    clauses = re.findall(clause_pattern, state["rfp_text"], re.DOTALL)
    requirements.extend([clause.strip() for clause in clauses if clause.strip()])
    
    seen = set(explicit_questions)
    requirements = [req for req in requirements if req not in seen and not (req in seen or seen.add(req))]
    # Normalize spacing in requirements
    requirements = [re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', req).replace('  ', ' ') for req in requirements]
    logger.info(f"Found {len(requirements)} additional requirements: {requirements[:3]}")

    # Step 4: Generate Q&A for additional requirements
    qa_prompts = {
        "scope": (
            "Given this RFP requirement: '{requirement}', generate a question and answer about the scope of work. "
            "Ensure the answer is complete, specific, and free of '...' placeholders. "
            "Ensure proper spacing between words and sentences for readability. "
            "Format as:\nQuestion: [Question]\nAnswer: [Answer]"
        ),
        "resource": (
            "Given this RFP requirement: '{requirement}', generate a question and answer about the resources needed. "
            "Ensure the answer is complete, specific, and free of '...' placeholders. "
            "Ensure proper spacing between words and sentences for readability. "
            "Format as:\nQuestion: [Question]\nAnswer: [Answer]"
        ),
        "timeline": (
            "Given this RFP requirement: '{requirement}', generate a question and answer about the timeline. "
            "Ensure the answer is complete, specific, and free of '...' placeholders. "
            "Ensure proper spacing between words and sentences for readability. "
            "Format as:\nQuestion: [Question]\nAnswer: [Answer]"
        ),
        "compliance": (
            "Given this RFP requirement: '{requirement}', generate a question and answer about compliance. "
            "Ensure the answer is complete, specific, and free of '...' placeholders. "
            "Ensure proper spacing between words and sentences for readability. "
            "Format as:\nQuestion: [Question]\nAnswer: [Answer]"
        ),
        "pricing": (
            "Given this RFP requirement: '{requirement}', generate a question and answer about pricing or costs. "
            "Ensure the answer is complete, specific, and free of '...' placeholders. "
            "Ensure proper spacing between words and sentences for readability. "
            "Format as:\nQuestion: [Question]\nAnswer: [Answer]"
        ),
        "general": (
            "Given this RFP requirement: '{requirement}', generate a question and answer about the requirement. "
            "Ensure the answer is complete, specific, and free of '...' placeholders. "
            "Ensure proper spacing between words and sentences for readability. "
            "Format as:\nQuestion: [Question]\nAnswer: [Answer]"
        )
    }
    
    for idx, req in enumerate(requirements, len(explicit_questions) + 1):
        if len(all_qa_pairs) >= max_pairs:
            break
        try:
            cleaned_req = req.replace('\n-', '').strip()
            if not cleaned_req:
                continue
                
            prompt_type = "general"
            if any(keyword in cleaned_req.lower() for keyword in ["scope", "work", "platform", "dashboard"]):
                prompt_type = "scope"
            elif any(keyword in cleaned_req.lower() for keyword in ["resource", "team", "infrastructure"]):
                prompt_type = "resource"
            elif any(keyword in cleaned_req.lower() for keyword in ["timeline", "duration", "milestone"]):
                prompt_type = "timeline"
            elif any(keyword in cleaned_req.lower() for keyword in ["compliance", "security", "gdpr", "ccpa"]):
                prompt_type = "compliance"
            elif any(keyword in cleaned_req.lower() for keyword in ["cost", "budget", "pricing"]):
                prompt_type = "pricing"

            prompt = qa_prompts[prompt_type].format(requirement=cleaned_req)
            qa_text = call_gemini_api(prompt)
            
            try:
                lines = qa_text.split("\n")
                if len(lines) >= 2 and lines[0].startswith("Question: ") and lines[1].startswith("Answer: "):
                    question = lines[0].replace("Question: ", "").strip()
                    answer = lines[1].replace("Answer: ", "").strip()
                    # Normalize spaces in the answer
                    answer = re.sub(r'\s+', ' ', answer).strip()
                    if question and answer and "..." not in answer:
                        all_qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "prompt_type": prompt_type,
                            "requirement": cleaned_req
                        })
                        logger.info(f"Generated Q&A {idx}: Q: {question[:50]} A: {answer[:50]}")
                    else:
                        raise ValueError("Invalid or incomplete Q&A")
                else:
                    raise ValueError("Incorrect format")
            except Exception:
                logger.warning(f"Failed to parse Gemini Q&A for requirement {idx}: {cleaned_req[:50]}")
                question = f"What does the requirement '{cleaned_req[:20]}' entail?"
                answer = cleaned_req
                all_qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "prompt_type": prompt_type,
                    "requirement": cleaned_req
                })
                logger.info(f"Fallback Q&A {idx}: Q: {question[:50]} A: {answer[:50]}")
        except Exception as e:
            logger.error(f"Error generating Q&A for requirement {idx}: {cleaned_req[:50]}: {e}")
            continue

    # Step 5: Save all Q&A pairs
    try:
        with open("all_rfp_qa.json", "w") as f:
            json.dump(all_qa_pairs, f, indent=2)
        logger.info(f"Saved {len(all_qa_pairs)} Q&A pairs to all_rfp_qa.json")
    except Exception as e:
        logger.error(f"Error saving all Q&A pairs: {e}")

    # Step 6: Prioritize and limit Q&A pairs
    prioritized_pairs = []
    priority_types = ["scope", "timeline", "resource", "compliance", "pricing"]
    
    for prompt_type in priority_types:
        for pair in all_qa_pairs:
            if pair["prompt_type"] == prompt_type and len(prioritized_pairs) < max_pairs:
                prioritized_pairs.append({
                    "question": pair["question"],
                    "answer": pair["answer"]
                })
    
    for pair in all_qa_pairs:
        if pair["prompt_type"] == "general" and len(prioritized_pairs) < max_pairs:
            prioritized_pairs.append({
                "question": pair["question"],
                "answer": pair["answer"]
            })

    state["qa_pairs"] = prioritized_pairs
    logger.info(f"Selected {len(prioritized_pairs)} prioritized Q&A pairs for the response")
    return state

# Node 3: Store Q&A
def store_qa_pairs(state: RFPState) -> RFPState:
    logger.info("Storing Q&A pairs")
    try:
        with open("rfp_qa.json", "w") as f:
            json.dump(state["qa_pairs"], f, indent=2)
        logger.info("Q&A pairs saved to rfp_qa.json")
    except Exception as e:
        logger.error(f"Error saving Q&A pairs: {e}")
    return state

# Node 4: Generate RFP response using Gemini API
def generate_response(state: RFPState) -> RFPState:
    logger.info("Generating RFP response as a LaTeX file using Gemini API")
    
    if not state["qa_pairs"]:
        logger.warning("No Q&A pairs available for response generation")
        state["response"] = "No response generated due to missing Q&A pairs."
        try:
            with open("rfp_response.txt", "w", encoding="utf-8") as f:
                f.write(state["response"])
            logger.info("Response saved to rfp_response.txt")
        except Exception as e:
            logger.error(f"Error saving response: {e}")
        return state

    qa_count = len(state["qa_pairs"])
    logger.info(f"Processing {qa_count} Q&A pairs")

    max_pairs = 50
    selected_pairs = state["qa_pairs"][:max_pairs]
    if qa_count > max_pairs:
        logger.info(f"Limiting to {max_pairs} Q&A pairs to manage file size")
    
    def escape_latex(text):
        if not isinstance(text, str):
            text = str(text)
        # Normalize spaces: replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        replacements = {
            '\\': '\\textbackslash{}',
            '{': '\\{',
            '}': '\\}',
            '#': '\\#',
            '%': '\\%',
            '&': '\\&',
            '$': '\\$',
            '_': '\\_',
            '^': '\\^',
            '~': '\\~{}'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    # Categorize Q&A pairs
    scope_pairs = []
    timeline_pairs = []
    resource_pairs = []
    compliance_pairs = []
    pricing_pairs = []
    for pair in selected_pairs:
        question = pair["question"].lower()
        if any(kw in question for kw in ["scope", "platform", "work", "dashboard", "integration"]):
            scope_pairs.append(pair)
        elif any(kw in question for kw in ["timeline", "milestone"]):
            timeline_pairs.append(pair)
        elif any(kw in question for kw in ["resource", "team", "infrastructure"]):
            resource_pairs.append(pair)
        elif any(kw in question for kw in ["compliance", "security"]):
            compliance_pairs.append(pair)
        elif any(kw in question for kw in ["cost", "pricing"]):
            pricing_pairs.append(pair)

    # Extract context
    project_title = "RFP Response: AI-Driven Automation Platform"
    organization = "ApexNeural Inc."
    if state["rfp_text"]:
        title_match = re.search(r'Request for Proposal.*?(?:\n|$)', state["rfp_text"], re.IGNORECASE)
        if title_match:
            project_title = f"RFP Response: {title_match.group(0).replace('Request for Proposal', '').strip()[:100]}"
        org_match = re.search(r'(?:submitted to|issued by|company|organization)\s*[:\s]*(.*?)(?:\n|$)', state["rfp_text"], re.IGNORECASE)
        if org_match:
            organization = org_match.group(1).strip()[:100]

    # Generate response sections with Gemini API
    summary_prompt = (
        "Based on the following Q&A pairs from ApexNeural's AI automation platform RFP, generate a professional RFP response. "
        "Do not directly quote the Q&A. Ensure all answers are complete and free of '...' placeholders. "
        "Ensure proper spacing between words and sentences for readability. "
        "Include these sections, each 2-3 sentences, tailored to the RFP (6-month timeline, $500,000–$1,000,000 budget, TensorFlow/PyTorch integration, GDPR/CCPA compliance):\n"
        "Introduction: [Introduce the response and commitment]\n"
        "Objective: [State project goals]\n"
        "Requirements: [Summarize key requirements from Q&A]\n"
        "Methodology: [Describe approach with 5 specific steps]\n"
        "Implementation Plan: [Summarize 6-month timeline with milestones]\n"
        "Resources: [Detail roles by phase]\n"
        "Pricing: [Provide cost breakdown]\n"
        "Conclusion: [Reinforce partnership]\n\n"
        "Q&A Pairs:\n" +
        "\n".join([f"Q: {p['question']}\nA: {p['answer']}" for p in selected_pairs]) +
        "\n\nFormat as:\n[Section]: [Content]"
    )

    try:
        summary_text = call_gemini_api(summary_prompt)
        logger.info(f"Gemini API summary generated: {summary_text[:200]}")
    except Exception as e:
        logger.error(f"Gemini API summary failed: {e}")
        summary_text = ""

    # Parse Gemini summary
    sections = {
        "Introduction": "We propose a comprehensive solution to develop an AI-driven automation platform for ApexNeural Inc., enhancing neural network training efficiency with robust integration and compliance.",
        "Objective": "The objective is to deliver a scalable, secure platform that automates data preprocessing, model training, and deployment within 6 months to optimize ApexNeural’s AI pipeline.",
        "Requirements": "The platform must integrate with TensorFlow and PyTorch, comply with GDPR and CCPA, support automated workflows, and provide a user-friendly dashboard for monitoring.",
        "Methodology": [
            "Design a cloud-based platform with TensorFlow and PyTorch integration for scalable neural network training.",
            "Develop automated workflows for data preprocessing, including handling missing data and feature scaling.",
            "Implement hyperparameter tuning algorithms to optimize model performance and reduce training time.",
            "Create an intuitive dashboard for real-time pipeline monitoring and user interaction.",
            "Conduct rigorous testing and provide comprehensive training for ApexNeural’s team."
        ],
        "Implementation Plan": "The project will span 6 months, with milestones for design in Month 1, development in Months 2-4, testing in Month 5, and deployment in Month 6.",
        "Resources": "The project will involve AI engineers, DevOps specialists, UI designers, QA engineers, and trainers allocated across design, development, testing, and deployment phases.",
        "Pricing": "The total estimated cost is $750,000, including $400,000 for software development, $200,000 for cloud infrastructure, and $150,000 for support and training services.",
        "Conclusion": "We are committed to partnering with ApexNeural Inc. to deliver a transformative AI automation platform that meets all RFP requirements and drives innovation."
    }

    try:
        current_section = None
        for line in summary_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith(("Introduction:", "Objective:", "Requirements:", "Implementation Plan:", "Resources:", "Pricing:", "Conclusion:")):
                current_section = line.split(":", 1)[0]
                sections[current_section] = line.split(":", 1)[1].strip()
            elif line.startswith("Methodology:"):
                current_section = "Methodology"
                sections[current_section] = []
            elif current_section == "Methodology" and line:
                sections[current_section].append(line.strip())
    except:
        logger.warning("Failed to parse Gemini summary, using defaults")

    # Generate methodology steps if not provided
    if not sections["Methodology"]:
        methodology_prompt = (
            "Based on these scope-related Q&A pairs, generate 5 concise methodology steps (1-2 sentences each, professional tone, no direct quotes, no '...' placeholders) "
            "for developing an AI-driven automation platform. "
            "Ensure proper spacing between words and sentences for readability:\n" +
            "\n".join([f"Q: {p['question']}\nA: {p['answer']}" for p in scope_pairs[:6]])
        )
        try:
            meth_text = call_gemini_api(methodology_prompt)
            sections["Methodology"] = [line.strip() for line in meth_text.split("\n") if line.strip()][:5]
        except:
            sections["Methodology"] = [
                "Design a cloud-based platform with TensorFlow and PyTorch integration for scalable neural network training.",
                "Develop automated workflows for data preprocessing, including handling missing data and feature scaling.",
                "Implement hyperparameter tuning algorithms to optimize model performance and reduce training time.",
                "Create an intuitive dashboard for real-time pipeline monitoring and user interaction.",
                "Conduct rigorous testing and provide comprehensive training for ApexNeural’s team."
            ]

    # Generate timeline table
    timeline_items = []
    for pair in timeline_pairs[:5]:
        try:
            duration_match = re.search(r'(\d+\s*(?:month|week|day)s?)', pair["answer"], re.IGNORECASE)
            timeline = duration_match.group(0) if duration_match else "To be determined"
            timeline_items.append({
                "phase": pair["question"][:50].strip(),
                "activities": pair["answer"][:150],
                "timeline": timeline
            })
        except:
            timeline_items.append({
                "phase": pair["question"][:50].strip(),
                "activities": pair["answer"][:150],
                "timeline": "To be determined"
            })
    
    if not timeline_items:
        timeline_items = [
            {"phase": "Design", "activities": "Define platform architecture and integrate with TensorFlow and PyTorch", "timeline": "Month 1"},
            {"phase": "Development", "activities": "Implement automated workflows and dashboard for neural network training", "timeline": "Months 2-4"},
            {"phase": "Testing", "activities": "Validate platform performance under high compute loads", "timeline": "Month 5"},
            {"phase": "Deployment", "activities": "Deploy platform and train ApexNeural’s team", "timeline": "Month 6"}
        ]

    # Check disk space
    disk_usage = shutil.disk_usage(os.getcwd())
    free_mb = disk_usage.free / (1024 * 1024)
    logger.info(f"Disk space - Free: {free_mb:.2f} MB")
    if free_mb < 10:
        logger.error("Insufficient disk space (<10 MB free)")
        state["response"] = "Error: Insufficient disk space to generate LaTeX file."
        return state

    # Dynamic LaTeX template
    latex_header = rf"""
\documentclass[a4paper,12pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{geometry}}
\geometry{{margin=1in}}
\usepackage{{booktabs}}
\usepackage{{enumitem}}
\usepackage{{titling}}
\usepackage{{parskip}}
\usepackage{{xcolor}}
\usepackage{{times}}

\title{{{escape_latex(project_title)}}}
\author{{[Your Company Name]}}
\date{{May 24, 2025}}

\begin{{document}}

\begin{{titlepage}}
    \centering
    \vspace*{{2cm}}
    {{\Huge \textbf{{{escape_latex(project_title)}}}\par}}
    \vspace{{1cm}}
    {{\Large \textbf{{[Your Company Name]}}\par}}
    \vspace{{0.5cm}}
    {{\large Submitted to: {escape_latex(organization)}\par}}
    \vspace{{0.5cm}}
    {{\large May 24, 2025\par}}
    \vfill
\end{{titlepage}}

\section*{{Introduction}}
{escape_latex(sections["Introduction"])} Our expertise in AI and cloud technologies ensures a solution that meets ApexNeural’s stringent requirements for innovation and efficiency.

\section*{{Objective}}
{escape_latex(sections["Objective"])} This platform will enhance ApexNeural’s ability to deploy high-performance AI models efficiently and securely.

\section*{{Requirements}}
{escape_latex(sections["Requirements"])} Key deliverables include:
\begin{{itemize}}
"""
    for pair in scope_pairs[:6] + compliance_pairs[:3]:
        latex_header += f"    \\item {escape_latex(pair['answer'][:150])}\n"
    latex_header += r"\end{itemize}" + "\n\n"

    latex_methodology = r"""
\section*{Methodology}
Our approach to delivering the AI-driven automation platform includes:
\begin{enumerate}
"""
    for i, step in enumerate(sections["Methodology"], 1):
        latex_methodology += f"    \\item \\textbf{{Phase {i}}}: {escape_latex(step)}\n"
    latex_methodology += r"\end{enumerate}" + "\n\n"

    latex_timeline = rf"""
\section*{{Implementation Plan}}
{escape_latex(sections["Implementation Plan"])} The timeline is structured as follows:

\begin{{table}}[h]
    \centering
    \begin{{tabular}}{{|l|p{{8cm}}|c|}}
        \hline
        \textbf{{Phase}} & \textbf{{Activities}} & \textbf{{Timeline}} \\
        \hline
"""
    for item in timeline_items:
        latex_timeline += f"        {escape_latex(item['phase'])} & {escape_latex(item['activities'])} & {escape_latex(item['timeline'])} \\\\\n        \\hline\n"
    latex_timeline += r"""
    \end{tabular}
    \caption{Implementation Timeline}
\end{table}
""" + "\n\n"

    latex_resources = rf"""
\section*{{Resources}}
{escape_latex(sections["Resources"])} The following roles are allocated:

\begin{{table}}[h]
    \centering
    \begin{{tabular}}{{|l|l|c|}}
        \hline
        \textbf{{Phase}} & \textbf{{Role}} & \textbf{{Count}} \\
        \hline
"""
    resource_roles = [
        {"phase": "Design", "role": "AI Engineers", "count": 3},
        {"phase": "Design", "role": "Solutions Architect", "count": 1},
        {"phase": "Development", "role": "DevOps Specialists", "count": 2},
        {"phase": "Development", "role": "UI Designers", "count": 1},
        {"phase": "Testing", "role": "QA Engineers", "count": 2},
        {"phase": "Deployment", "role": "Trainers", "count": 1}
    ]
    for res in resource_roles:
        latex_resources += f"        {escape_latex(res['phase'])} & {escape_latex(res['role'])} & {res['count']} \\\\\n        \\hline\n"
    latex_resources += r"""
    \end{tabular}
    \caption{Resource Allocation}
\end{table}
""" + "\n\n"

    latex_pricing = rf"""
\section*{{Pricing}}
{escape_latex(sections["Pricing"])} The detailed cost breakdown is:

\begin{{table}}[h]
    \centering
    \begin{{tabular}}{{|l|c|}}
        \hline
        \textbf{{Category}} & \textbf{{Cost (USD)}} \\
        \hline
"""
    pricing_items = [
        {"category": "Software Development", "cost": 400000},
        {"category": "Cloud Infrastructure", "cost": 200000},
        {"category": "Support and Training", "cost": 150000}
    ]
    for item in pricing_items:
        latex_pricing += f"        {escape_latex(item['category'])} & {item['cost']:,} \\\\\n        \\hline\n"
    latex_pricing += r"""
        \textbf{Total} & \textbf{750,000} \\
        \hline
    \end{tabular}
    \caption{Cost Breakdown}
\end{table}
""" + "\n\n"

    latex_conclusion = rf"""
\section*{{Conclusion}}
{escape_latex(sections["Conclusion"])} Our team is eager to collaborate with ApexNeural to deliver a transformative AI automation platform that drives operational excellence.
\end{{document}}
"""

    # Write LaTeX file incrementally
    output_path = os.path.join(os.getcwd(), "rfp_response.tex")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_header)
            f.write(latex_methodology)
            f.write(latex_timeline)
            f.write(latex_resources)
            f.write(latex_pricing)
            f.write(latex_conclusion)
            logger.info("Wrote LaTeX file")
        
        logger.info(f"LaTeX file '{output_path}' generated successfully")
        state["response"] = f"RFP response generated as '{output_path}'. Ready for refinement."
    
    except Exception as e:
        logger.error(f"Error saving LaTeX file: {e}\n{traceback.format_exc()}")
        state["response"] = f"Error generating LaTeX file: {e}"
    
    return state

# Node 5: Refine the RFP response using Gemini API
def refine_response(state: RFPState) -> RFPState:
    logger.info("Refining RFP response using Gemini API for grammar, spelling, and improvement")
    
    if not state["response"].startswith("RFP response generated as"):
        logger.warning("No LaTeX file to refine")
        return state

    # Extract the file path from the state
    output_path_match = re.search(r"'([^']+)'", state["response"])
    if not output_path_match:
        logger.error("Could not find LaTeX file path in state.response")
        state["response"] = "Error: Could not refine LaTeX file due to missing file path."
        return state
    
    output_path = output_path_match.group(1)
    if not os.path.exists(output_path):
        logger.error(f"LaTeX file not found at: {output_path}")
        state["response"] = f"Error: LaTeX file not found at {output_path}."
        return state

    # Read the LaTeX file content
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            latex_content = f.read()
        logger.info(f"Read LaTeX file content: {latex_content[:200]}...")
    except Exception as e:
        logger.error(f"Error reading LaTeX file: {e}")
        state["response"] = f"Error reading LaTeX file for refinement: {e}"
        return state

    # Define sections to refine (excluding tables and structured content)
    sections_to_refine = ["Introduction", "Objective", "Requirements", "Implementation Plan", "Resources", "Pricing", "Conclusion"]
    refined_sections = {}

    # Extract content for each section
    for section in sections_to_refine:
        # Match the section content (from \section*{Section} to the next \section* or \end{table})
        pattern = rf"\\section\*\{{{section}\}}(.*?)(?=\\section\*\{{|\\end\{{table\}}|\\end\{{document\}})"
        try:
            match = re.search(pattern, latex_content, re.DOTALL)
            if match:
                section_content = match.group(1).strip()
                # Remove LaTeX commands (e.g., \item, \textbf) and normalize spaces
                section_text = re.sub(r'\\(?:item|textbf|emph)\s*\{[^}]*\}', '', section_content)
                section_text = re.sub(r'\\[a-zA-Z]+\s*(?:\{[^}]*\})?', '', section_text)
                section_text = re.sub(r'\s+', ' ', section_text).strip()
                if section_text:
                    logger.info(f"Extracted content for {section}: {section_text[:100]}...")
                    # Send to Gemini API for refinement
                    refine_prompt = (
                        f"Refine the following text from an RFP response for grammar, spelling, clarity, and overall professionalism. "
                        f"Ensure proper spacing between words and sentences. "
                        f"Maintain a professional tone suitable for an RFP response. "
                        f"Do not alter the factual content or structure (e.g., do not convert paragraphs to lists). "
                        f"Text: '{section_text}'\n"
                        f"Return the refined text."
                    )
                    try:
                        refined_text = call_gemini_api(refine_prompt)
                        refined_text = re.sub(r'\s+', ' ', refined_text).strip()
                        logger.info(f"Refined {section}: {refined_text[:100]}...")
                        refined_sections[section] = refined_text
                    except Exception as e:
                        logger.warning(f"Failed to refine {section}: {e}. Keeping original content.")
                        refined_sections[section] = section_text
                else:
                    logger.warning(f"No content extracted for {section}. Keeping original.")
                    refined_sections[section] = section_text
            else:
                logger.warning(f"Section {section} not found in LaTeX file. Skipping refinement.")
                refined_sections[section] = ""
        except Exception as e:
            logger.error(f"Error extracting section {section}: {e}")
            refined_sections[section] = ""

    # Update the LaTeX content with refined sections
    updated_latex_content = latex_content
    for section, refined_text in refined_sections.items():
        if refined_text:
            # Escape the refined text for LaTeX
            refined_text = escape_latex(refined_text)
            # Replace the section content
            pattern = rf"(\\section\*\{{{section}\}})(.*?)(?=\\section\*\{{|\\end\{{table\}}|\\end\{{document\}})"
            try:
                # Find the section and its content
                match = re.search(pattern, updated_latex_content, re.DOTALL)
                if match:
                    # Preserve any LaTeX structure (e.g., \begin{itemize}) that follows the section text
                    original_content = match.group(2).strip()
                    # Check if the original content contains structured elements (e.g., itemize, table)
                    structured_part = ""
                    if "\\begin{itemize}" in original_content:
                        structured_part = original_content[original_content.find("\\begin{itemize}"):]
                    elif "\\begin{table}" in original_content:
                        structured_part = original_content[original_content.find("\\begin{table}"):]
                    # Replace the section content, preserving the structured part
                    new_section_content = f"{refined_text}\n{structured_part}" if structured_part else refined_text
                    updated_latex_content = updated_latex_content.replace(match.group(0), f"\\section*{{{section}}}\n{new_section_content}")
                else:
                    logger.warning(f"Could not replace content for {section} during refinement.")
            except Exception as e:
                logger.error(f"Error replacing content for {section}: {e}")

    # Write the updated LaTeX file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(updated_latex_content)
        logger.info(f"Updated LaTeX file '{output_path}' with refined content")
        state["response"] = f"RFP response refined and saved as '{output_path}'. Ready for PDF compilation."
    except Exception as e:
        logger.error(f"Error saving refined LaTeX file: {e}")
        state["response"] = f"Error saving refined LaTeX file: {e}"

    return state

# Define the LangGraph workflow
workflow = StateGraph(RFPState)

# Add nodes
workflow.add_node("extract_rfp", extract_rfp_text)
workflow.add_node("generate_qa", generate_qa_pairs)
workflow.add_node("store_qa", store_qa_pairs)
workflow.add_node("generate_response", generate_response)
workflow.add_node("refine_response", refine_response)

# Define edges
workflow.add_edge("extract_rfp", "generate_qa")
workflow.add_edge("generate_qa", "store_qa")
workflow.add_edge("store_qa", "generate_response")
workflow.add_edge("generate_response", "refine_response")
workflow.add_edge("refine_response", END)

# Set entry point
workflow.set_entry_point("extract_rfp")

# Compile the graph
app = workflow.compile()

# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rfp_path = os.path.join(script_dir, "apexneural_rfp.pdf")
    logger.info(f"Looking for RFP file at: {rfp_path}")
    
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