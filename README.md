# QuLab: Lab 29: Summarizing a 10-K with RAG

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Lab 29: Summarizing a 10-K with Retrieval-Augmented Generation (RAG)**

This Streamlit application is part of the QuantUniversity Lab series, specifically Lab 29, focusing on leveraging advanced Natural Language Processing (NLP) techniques to streamline financial analysis. As a CFA Charterholder and Investment Professional, extracting targeted, reliable information from lengthy **10-K annual filings** is a crucial yet time-consuming task. Generic, full-document summaries often lack the precision, detail, and traceability required for robust financial analysis and compliance.

This application transforms the process of analyzing 10-K documents by implementing a sophisticated **Retrieval-Augmented Generation (RAG)** pipeline. It's designed to:

*   **Understand the structure** of 10-K filings through intelligent, section-aware chunking.
*   **Retrieve only the most relevant text passages** from the document in response to specific research questions, utilizing vector embeddings and a vector database (ChromaDB).
*   **Synthesize these retrieved passages into analytical, cited summaries** using a Large Language Model (LLM) like OpenAI's GPT models.
*   **Provide a quantitative comparison** between RAG-based summarization and traditional full-context summarization, highlighting cost savings and improved quality.
*   **Assess summary quality** using an "LLM-as-judge" framework, evaluating faithfulness and coverage.

This methodology dramatically accelerates research, provides auditable trails back to the source document, and significantly reduces LLM operational costs by minimizing token usage.

## Features

This application provides a comprehensive suite of tools for 10-K analysis:

1.  **10-K Document Upload and Processing**:
    *   Securely upload 10-K PDF documents.
    *   Automated parsing of the PDF to identify and extract key sections (e.g., "Item 1. Business", "Item 7. Management's Discussion and Analysis").
    *   Extraction of the full text content of the document.

2.  **Knowledge Base Creation**:
    *   Intelligent chunking of extracted sections to create manageable text segments.
    *   Conversion of text chunks into high-dimensional vector embeddings using OpenAI's embedding models.
    *   Storage of these embeddings and associated metadata (like section names) in a **ChromaDB** vector database, forming a searchable knowledge base.
    *   Visual representation of the mathematical concept of **Cosine Similarity** for semantic retrieval.

3.  **Multi-Aspect Research Brief Generation**:
    *   **Predefined financial topics**: Generate summaries for common analyst questions (e.g., "Revenue Recognition Policies", "Key Risks").
    *   **Custom topic queries**: Users can define their own specific research questions.
    *   **Section filtering**: Limit retrieval to specific 10-K sections for highly targeted analysis.
    *   Generation of concise, analytical summaries with clear citations to the originating 10-K sections.
    *   Real-time tracking of token usage and **estimated cost** for each generated summary and the entire brief.

4.  **RAG vs. Full-Context Summarization Comparison**:
    *   **Quantitative Cost Analysis**: Directly compare the token cost of RAG-based summarization against summarization by feeding the entire (or a large portion of the) 10-K directly to the LLM.
    *   **Summary Compression Ratio**: Calculates how efficiently the summary compresses relevant source material.
    *   Side-by-side display of RAG-generated and full-context summaries for qualitative comparison.
    *   Highlights RAG's advantages in **cost efficiency, traceability, and grounding**.

5.  **Summary Quality Assessment**:
    *   Utilizes an **"LLM-as-judge" framework** to objectively evaluate summary quality.
    *   **Faithfulness Score**: Quantifies the proportion of claims in the summary that are directly supported by the retrieved source chunks, ensuring accuracy and reducing hallucinations.
    *   **Coverage Scores**: Qualitative assessment (1-5 scale) across **Breadth, Depth, and Completeness** to determine how well the summary addresses the topic.
    *   Provides actionable diagnostics to improve RAG performance (e.g., refine query, adjust `k` for retrieval, modify system prompt, adjust LLM temperature).

6.  **Flexible LLM Configuration**:
    *   Select between different OpenAI LLM models (e.g., `gpt-4o`, `gpt-3.5-turbo`).
    *   Secure input field for OpenAI API Key.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   **Python 3.8+**
*   **OpenAI API Key**: You will need an API key from OpenAI to use their LLM and embedding models. You can obtain one from the [OpenAI platform](https://platform.openai.com/api-keys).

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/quolab-lab29-10k-rag.git
    cd quolab-lab29-10k-rag
    ```
    *(Replace `your-username/quolab-lab29-10k-rag` with the actual repository path)*

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `requirements.txt` contains `streamlit`, `pymupdf` (fitz), `pandas`, `numpy`, `chromadb`, `openai`, `tiktoken`, `python-dotenv` if used for environment variables).

4.  **Set your OpenAI API Key** (Optional, but recommended for persistent access):
    You can set your OpenAI API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Alternatively, the application provides a text input field in the sidebar to enter your key directly.

## Usage

1.  **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

2.  **Configure API Key**:
    *   In the sidebar, enter your **OpenAI API Key** if it's not already loaded from an environment variable.

3.  **Page 1: Upload 10-K & Process**:
    *   Navigate to the "Upload 10-K & Process" section using the sidebar.
    *   Upload a 10-K PDF document using the file uploader.
    *   Click the "Process 10-K" button. The application will parse the document, chunk its sections, create embeddings, and build the ChromaDB knowledge base. This may take a few moments.

4.  **Page 2: Multi-Aspect Research Brief**:
    *   Once the 10-K is processed, switch to the "Multi-Aspect Research Brief" section.
    *   Select predefined research topics or add your own custom topic query. You can also filter custom topics to specific 10-K sections.
    *   Click "Generate Multi-Aspect Brief" to get a structured summary across your chosen topics, complete with sources and estimated costs.

5.  **Page 3: Comparison & Quality Assessment**:
    *   After generating a brief, go to the "Comparison & Quality Assessment" section.
    *   Select a topic from your brief to perform a detailed comparison.
    *   Click "Run Comparison & Quality Assessment". The app will then generate a full-context summary for comparison, provide cost metrics, and perform an LLM-based quality assessment of the RAG summary.

## Project Structure

```
quolab-lab29-10k-rag/
├── app.py                  # Main Streamlit application
├── source.py               # Helper functions for PDF parsing, embedding, RAG, etc.
├── requirements.txt        # Python dependencies
├── README.md               # Project README file
├── .env                    # (Optional) For environment variables like OPENAI_API_KEY
├── AAPL_10K.pdf            # Temporary storage for uploaded PDF files
└── LICENSE                 # (Optional) Project license file
```

*   `app.py`: Contains the Streamlit UI logic, session state management, and orchestrates calls to the helper functions in `source.py`.
*   `source.py`: This module is expected to contain the core logic, including:
    *   `parse_10k_sections`: Extracts text and identifies sections from PDF.
    *   `embedder_func`: Handles text embedding using OpenAI.
    *   `chunk_and_embed_sections`: Manages chunking and populating ChromaDB.
    *   `rag_summarize`: Implements the RAG workflow for targeted summarization.
    *   `full_context_summarize`: Performs summarization by sending the full document to the LLM.
    *   `assess_summary_quality`: Implements the LLM-as-judge quality assessment.
    *   `BRIEF_TOPICS`: Predefined list of topics for brief generation.
    *   `client_llm`: OpenAI client instance.

## Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/) (for interactive web applications)
*   **Programming Language**: Python 3.x
*   **PDF Processing**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/) (for storing and querying vector embeddings)
*   **Large Language Models (LLM)**: [OpenAI API](https://openai.com/docs/api/) (GPT-4o, GPT-3.5-turbo)
*   **Embedding Models**: [OpenAI Embeddings](https://openai.com/docs/guides/embeddings/)
*   **Data Handling**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Token Counting**: [Tiktoken](https://github.com/openai/tiktoken)

## Contributing

Contributions to this lab project are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork** the repository.
2.  **Create a new branch** (`git checkout -b feature/your-feature-name` or `fix/bug-description`).
3.  **Make your changes** and ensure the code adheres to existing style guidelines.
4.  **Commit your changes** (`git commit -m 'feat: Add new feature'`).
5.  **Push to the branch** (`git push origin feature/your-feature-name`).
6.  **Open a Pull Request** to the `main` branch of the original repository.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

```
MIT License

Copyright (c) 2024 QuantUniversity

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

For any questions or feedback regarding this lab project, please reach out to:

*   **QuantUniversity**: [www.quantuniversity.com](https://www.quantuniversity.com)
*   **Project Maintainer**: [your-email@example.com](mailto:your-email@example.com) *(Replace with actual contact info)*