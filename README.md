# ApexNeural RFP Automation Tool

## Overview
This repository contains a Python-based backend tool developed during a one-day ApexNeural hackathon to automate the processing of Request for Proposal (RFP) documents. The tool extracts text from RFP PDFs, generates question-and-answer (Q&A) pairs using the xAI Grok API, and produces a professional RFP response in LaTeX format. Additionally, the repository includes a basic HTML/CSS/JavaScript frontend, which is currently not integrated with the backend logic, and sample RFP PDFs for testing.

## Features
- **PDF Text Extraction**: Extracts text from RFP PDF files using `pdfplumber`.
- **Q&A Generation**: Identifies explicit questions and implicit requirements in the RFP text, generating detailed Q&A pairs via the xAI Grok API.
- **Response Generation**: Produces a structured RFP response in LaTeX, including sections for introduction, objectives, requirements, methodology, implementation plan, resources, pricing, and conclusion.
- **Workflow Automation**: Utilizes `langgraph` to manage the workflow, chaining tasks from text extraction to response generation.
- **Fallback Mechanisms**: Handles missing dependencies (e.g., `spacy`) and API errors with retry logic and default responses.
- **Sample PDFs**: Includes sample RFP PDFs (`apexneural_rfp.pdf`) for testing.

## Repository Structure
- `main.py`: The core Python script implementing the RFP processing workflow.
- `frontend/`: Directory containing basic HTML, CSS, and JavaScript files for a frontend interface (not currently connected to the backend).
- `sample_pdfs/`: Directory with sample RFP PDFs for testing.
- `rfp_qa.json`: Output file storing prioritized Q&A pairs.
- `all_rfp_qa.json`: Output file storing all generated Q&A pairs.
- `rfp_response.tex`: Output LaTeX file containing the generated RFP response.
- `.env`: Configuration file for storing the xAI API key (not committed).

## Prerequisites
- **Python 3.8+**
- **Dependencies**: Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```
  Required packages include `pdfplumber`, `Pillow`, `requests`, `python-dotenv`, `langgraph`, `langchain`, and optionally `spacy` for advanced text processing.
- **xAI API Key**: Obtain an API key from [xAI](https://x.ai/api) and set it as `XAI_API_KEY` in a `.env` file or environment variable.
- **spaCy Model** (optional): For enhanced text processing, install the `en_core_web_sm` model:
  ```bash
  python -m spacy download en_core_web_sm
  ```
- **LaTeX Environment**: Required to compile the generated `rfp_response.tex` into a PDF (e.g., using `latexmk` with PDFLaTeX).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/apexneural-rfp-tool.git
   cd apexneural-rfp-tool
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up the xAI API key:
   - Create a `.env` file in the root directory with:
     ```
     XAI_API_KEY=your_api_key_here
     ```
4. (Optional) Install the spaCy model for improved requirement extraction:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage
1. Place the RFP PDF (e.g., `apexneural_rfp.pdf`) in the project directory or specify a custom path.
2. Run the main script:
   ```bash
   python main.py [path_to_rfp_pdf]
   ```
   If no path is provided, the script defaults to `./apexneural_rfp.pdf`.
3. The script will:
   - Extract text from the RFP PDF.
   - Generate Q&A pairs, saved as `rfp_qa.json` (prioritized) and `all_rfp_qa.json` (all pairs).
   - Create a LaTeX RFP response file (`rfp_response.tex`).
4. Compile the LaTeX file to generate a PDF:
   ```bash
   latexmk -pdf rfp_response.tex
   ```

## Example Output
- **Q&A Pairs** (`rfp_qa.json`):
  ```json
  [
    {
      "question": "What is the scope of the AI automation platform?",
      "answer": "The platform will integrate TensorFlow and PyTorch for scalable neural network training."
    },
    ...
  ]
  ```
- **LaTeX Response** (`rfp_response.tex`): A professional document with sections like Introduction, Methodology, and Pricing, tailored to the RFP requirements.
- **Logs**: Detailed logs are output to the console for debugging, including API calls and error handling.

## Frontend
The `frontend/` directory contains a basic HTML/CSS/JavaScript interface developed during the hackathon. It is not currently integrated with the backend but provides a starting point for visualizing RFP responses or Q&A pairs. Future work could involve connecting the frontend to the backend using a framework like Flask or FastAPI.

## Hackathon Context
This tool was developed in a single day as part of the ApexNeural hackathon, focusing on automating RFP responses for an AI-driven automation platform. The RFP requirements included TensorFlow/PyTorch integration, a 6-month timeline, a $500,000–$1,000,000 budget, and GDPR/CCPA compliance. The tight timeline led to a focus on robust backend logic, with the frontend remaining a proof-of-concept.

## Limitations
- The frontend is not connected to the backend and requires further development for full integration.
- The tool assumes the RFP PDF contains extractable text; OCR is not implemented for scanned documents.
- The xAI Grok API may encounter rate limits, handled with retry logic but potentially impacting performance.
- The LaTeX output requires a LaTeX environment for compilation, which is not included in the repository.

## Future Improvements
- Integrate the frontend with the backend using a web framework.
- Add OCR support for scanned PDFs using libraries like `pytesseract`.
- Enhance Q&A generation with more sophisticated NLP techniques.
- Implement a configuration file for customizable RFP response templates.
- Add unit tests to ensure robustness across diverse RFP formats.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or contributions, please open an issue or contact [your-email@example.com].