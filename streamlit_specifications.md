
## Streamlit Application Specification: 10-K Summarization with RAG

### 1. Application Overview

**Purpose of the Application**

This Streamlit application, designed for **CFA Charterholders and Investment Professionals**, revolutionizes the process of extracting targeted insights from lengthy 10-K annual filings. It implements a **Retrieval-Augmented Generation (RAG)** pipeline to move beyond inefficient manual reviews or generic full-document summaries. The application focuses on **analyst-directed summarization**, enabling users to upload a 10-K, define specific research questions, and receive coherent, analytical summaries grounded in the document, complete with source citations. It also demonstrates the significant cost and quality advantages of RAG over traditional full-context LLM summarization.

**High-Level Story Flow**

1.  **Onboarding & Document Setup:** The analyst first navigates to the "Upload 10-K" page. They upload a 10-K PDF (e.g., AAPL_10K_2024.pdf). The application then processes this document through **Section-Aware Chunking**, which understands and preserves the 10-K's structural boundaries (e.g., Item 1A, Item 7). These section-aware chunks are then converted into **vector embeddings** and stored in a ChromaDB knowledge base, making the document semantically searchable. A visual confirmation of parsing and indexing is provided.

2.  **Multi-Aspect Research Brief Generation:** Once the 10-K is processed, the analyst moves to the "Research Brief" page. Here, they can generate a "Multi-Aspect Research Brief" by selecting from predefined financial topics (e.g., Risk Factors, Liquidity & Capital, Segment Performance) or defining a custom topic. The RAG pipeline executes for each selected topic, retrieving the most relevant chunks and synthesizing them into a focused, cited summary. The application compiles these summaries into a structured brief, displaying the total cost.

3.  **RAG vs. Full-Context Comparison & Quality Assessment:** The final "Comparison & Quality" page allows the analyst to delve deeper into the RAG's efficacy. For a chosen topic from the generated brief, the application compares the RAG-generated summary against a hypothetical full-context summary. This comparison highlights **cost advantages**, token usage, and the critical presence of citations in RAG. Additionally, an "LLM-as-judge" framework assesses the RAG summary's **Coverage** (breadth, depth, completeness) and **Faithfulness** (traceability of claims to source text), providing diagnostic insights into summary quality.

### 2. Code Requirements

```python
import streamlit as st
import os
import fitz # PyMuPDF
import re
import pandas as pd
import json
import numpy as np
import textwrap
from typing import List, Dict, Any, Optional

# Import all functions and necessary classes from source.py
# This assumes source.py is available as a module 'source'
# and contains the necessary imports like OpenAI, SentenceTransformer, etc.
from source import (
    parse_10k_sections,
    chunk_and_embed_sections,
    retrieve, # Though not directly called, it's part of rag_summarize logic
    rag_summarize,
    full_context_summarize,
    assess_summary_quality,
    client_llm, # OpenAI client instance
    embedder_func, # MyEmbeddingFunction instance
    BRIEF_TOPICS # Predefined topics for the research brief
)

# --- st.session_state Initialization ---
# Initialize session state variables if they don't exist
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = os.environ.get("OPENAI_API_KEY", "") # Or get from Streamlit secrets

if 'parsed_sections' not in st.session_state:
    st.session_state['parsed_sections'] = None
if 'full_text' not in st.session_state:
    st.session_state['full_text'] = None
if 'chroma_collection' not in st.session_state:
    st.session_state['chroma_collection'] = None
if 'pdf_uploaded_name' not in st.session_state:
    st.session_state['pdf_uploaded_name'] = None
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False

if 'brief_results' not in st.session_state:
    st.session_state['brief_results'] = []
if 'total_brief_cost' not in st.session_state:
    st.session_state['total_brief_cost'] = 0.0

if 'comparison_topic_label' not in st.session_state:
    st.session_state['comparison_topic_label'] = None
if 'comparison_rag_result' not in st.session_state:
    st.session_state['comparison_rag_result'] = None
if 'comparison_full_summary' not in st.session_state:
    st.session_state['comparison_full_summary'] = None
if 'comparison_full_cost' not in st.session_state:
    st.session_state['comparison_full_cost'] = None
if 'comparison_full_input_tokens' not in st.session_state:
    st.session_state['comparison_full_input_tokens'] = None
if 'quality_assessments' not in st.session_state:
    st.session_state['quality_assessments'] = {}

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Upload 10-K & Process"

# --- UI Interactions and Function Calls ---

st.set_page_config(layout="wide", page_title="10-K RAG Summarizer for Analysts")

st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Choose a section:",
    ["Upload 10-K & Process", "Multi-Aspect Research Brief", "Comparison & Quality Assessment"],
    key='page_select_box'
)
st.session_state['current_page'] = page_selection

st.sidebar.subheader("Configuration")
st.session_state['llm_model'] = st.sidebar.selectbox(
    "Select LLM Model:",
    ['gpt-4o', 'gpt-3.5-turbo'],
    index=1 if os.environ.get("OPENAI_API_KEY") else 0, # Default to cheaper if key not set for demo
    key='llm_model_select'
)
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API Key (e.g., sk-...)")
if api_key_input:
    st.session_state['openai_api_key'] = api_key_input
    client_llm.api_key = api_key_input # Update the client_llm instance directly

# --- Page 1: Upload 10-K & Process ---
if st.session_state['current_page'] == "Upload 10-K & Process":
    st.title("Summarizing a 10-K with RAG: Analyst's Edge")
    st.markdown(f"")
    st.markdown(f"As a **CFA Charterholder and Investment Professional**, extracting targeted information from lengthy **10-K annual filings** is crucial yet time-consuming. Generic, full-document summaries often lack the precision required for robust financial analysis.")
    st.markdown(f"This application transforms this process by implementing **Retrieval-Augmented Generation (RAG)** to:")
    st.markdown(f"- **Understand the structure** of 10-K filings through section-aware chunking.")
    st.markdown(f"- **Retrieve only the most relevant text** for specific research questions.")
    st.markdown(f"- **Synthesize these passages into analytical, cited summaries** using a Large Language Model (LLM).")
    st.markdown(f"This dramatically accelerates research, provides auditable trails, and reduces LLM operational costs.")

    st.subheader("1. Upload 10-K Document")
    uploaded_file = st.file_uploader("Upload a 10-K PDF document", type="pdf")

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state['pdf_uploaded_name']:
            st.session_state['processing_complete'] = False
            st.session_state['chroma_collection'] = None # Reset collection if new file
            st.session_state['brief_results'] = [] # Reset brief results

        # Save the uploaded file temporarily
        with open("temp_10k.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['pdf_uploaded_name'] = uploaded_file.name
        
        st.info(f"File '{uploaded_file.name}' uploaded successfully. Click 'Process 10-K' to continue.")

        if st.button("Process 10-K", disabled=st.session_state['processing_complete'] or not st.session_state['openai_api_key']):
            if not st.session_state['openai_api_key']:
                st.error("Please enter your OpenAI API key in the sidebar to proceed.")
            else:
                with st.spinner("Step 1/2: Parsing 10-K sections and extracting full text..."):
                    try:
                        sections, full_text = parse_10k_sections("temp_10k.pdf")
                        st.session_state['parsed_sections'] = sections
                        st.session_state['full_text'] = full_text
                        st.success("10-K Sections Parsed!")
                        st.markdown(f"**Parsed 10-K sections:**")
                        for name, text in sections.items():
                            st.markdown(f"- `{name}`: {len(text.split()):,} words")

                        st.subheader("2. Building the Knowledge Base: Vector Embeddings and Database")
                        st.markdown(f"Now, we convert each text chunk into a **vector embedding** to capture semantic meaning. These are stored in a **vector database** (ChromaDB) for **Topic-Directed Retrieval**, finding contextually similar chunks to queries.")
                        st.markdown(r"The mathematical foundation for semantic similarity often relies on measures like **cosine similarity**, which quantifies the angular distance between two vectors. Given two embedding vectors, $A$ and $B$, their cosine similarity is calculated as:")
                        st.markdown(r"$$\text{{Cosine Similarity}}(A, B) = \frac{{A \cdot B}}{{\|A\| \|B\|}}$$")
                        st.markdown(r"where $A \cdot B$ is the dot product of $A$ and $B$, and $\|A\|$ and $\|B\|$ are their magnitudes.")
                        st.markdown(f"")
                        st.info("For an analyst, this means we now have a powerful, semantic index. When you ask a question, the system won't just look for keywords; it will understand the *meaning* of your query and retrieve the most relevant passages, especially with section metadata filters.")

                        with st.spinner("Step 2/2: Chunking sections, creating embeddings, and storing in ChromaDB..."):
                            collection_name = f"{uploaded_file.name.replace('.', '_')}_sections"
                            chroma_collection = chunk_and_embed_sections(
                                st.session_state['parsed_sections'],
                                embedder_func,
                                collection_name=collection_name
                            )
                            st.session_state['chroma_collection'] = chroma_collection
                            st.session_state['processing_complete'] = True
                            st.success(f"Knowledge Base Created! Indexed {chroma_collection.count()} chunks from '{uploaded_file.name}'.")
                            st.balloons()

                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        st.session_state['processing_complete'] = False
                        st.session_state['chroma_collection'] = None
        elif st.session_state['processing_complete']:
            st.success(f"10-K '{uploaded_file.name}' is already processed and ready for analysis!")
    elif st.session_state['pdf_uploaded_name'] and st.session_state['processing_complete']:
        st.success(f"Using previously uploaded and processed 10-K: '{st.session_state['pdf_uploaded_name']}'. Navigate to other sections.")

# --- Page 2: Multi-Aspect Research Brief ---
elif st.session_state['current_page'] == "Multi-Aspect Research Brief":
    st.title("Multi-Aspect Research Brief Generation")
    st.markdown(f"As an analyst, a comprehensive research update requires structured overviews across multiple financial dimensions. This section allows you to compile a 'Multi-Aspect Research Brief' by systematically applying our RAG pipeline to predefined or custom financial topics.")
    st.markdown(f"")
    
    if not st.session_state['processing_complete'] or not st.session_state['chroma_collection']:
        st.warning("Please upload and process a 10-K document on the 'Upload 10-K & Process' page first.")
    else:
        st.subheader(f"1. Define Research Topics for '{st.session_state['pdf_uploaded_name']}'")
        st.info("Select predefined topics or add your own custom topic.")
        
        selected_brief_configs = []
        
        # Predefined topics
        st.markdown("**Predefined Topics:**")
        for i, topic_config in enumerate(BRIEF_TOPICS):
            checkbox_key = f"brief_topic_checkbox_{i}"
            if st.checkbox(f"**{topic_config['brief_label']}**", key=checkbox_key, value=True):
                selected_brief_configs.append(topic_config)

        # Custom topic input
        st.markdown("---")
        st.markdown("**Add a Custom Topic:**")
        custom_topic_query = st.text_area("Custom Topic Query:", value="", placeholder="e.g., 'Impact of recent mergers and acquisitions on company strategy'")
        all_sections_available = sorted(list(st.session_state['parsed_sections'].keys())) if st.session_state['parsed_sections'] else []
        custom_topic_sections = st.multiselect(
            "Filter custom topic to specific 10-K sections (optional):",
            options=all_sections_available,
            default=[],
            key='custom_topic_sections_multiselect'
        )
        
        if custom_topic_query.strip():
            custom_config = {
                'topic': custom_topic_query,
                'sections': custom_topic_sections if custom_topic_sections else None,
                'brief_label': "CUSTOM TOPIC"
            }
            if st.button("Add Custom Topic to Brief"):
                selected_brief_configs.append(custom_config)
                st.success("Custom topic added!")

        st.markdown("---")
        
        if st.button("Generate Multi-Aspect Brief", disabled=not selected_brief_configs or not st.session_state['openai_api_key']):
            if not st.session_state['openai_api_key']:
                st.error("Please enter your OpenAI API key in the sidebar to proceed.")
            else:
                st.session_state['brief_results'] = []
                st.session_state['total_brief_cost'] = 0.0
                progress_text = "Generating research brief, please wait..."
                my_bar = st.progress(0, text=progress_text)
                
                brief_output = []
                for i, topic_config in enumerate(selected_brief_configs):
                    my_bar.progress((i + 1) / len(selected_brief_configs), text=f"Generating summary for: {topic_config['brief_label']}")
                    result = rag_summarize(
                        topic_query=topic_config['topic'],
                        collection=st.session_state['chroma_collection'],
                        embedder_func=embedder_func,
                        section_filter=topic_config['sections'],
                        k=8,
                        model=st.session_state['llm_model']
                    )
                    result['label'] = topic_config['brief_label']
                    st.session_state['brief_results'].append(result)
                    st.session_state['total_brief_cost'] += result['cost']
                    brief_output.append(result)

                my_bar.empty()
                st.success("Multi-Aspect Research Brief Generated!")
                st.balloons()
                
                st.subheader(f"2. 10-K RESEARCH BRIEF: {st.session_state['pdf_uploaded_name'].replace('_10K_2024.pdf', '')} (FY2024)")
                st.markdown(f"Generated via RAG | Total cost for brief: **${st.session_state['total_brief_cost']:.4f}**")
                
                for section in st.session_state['brief_results']:
                    st.markdown(f"--- **{section['label']}** ---")
                    st.markdown(f"*(Sources: {', '.join(section['sections_covered'])})*")
                    st.markdown(section['summary'])
                    st.markdown(f"*(Cost: ${section['cost']:.4f})*")
                    st.markdown(f"")
                st.info("This structured brief provides initial summaries, allowing you to rapidly grasp key points, verify sources, and concentrate on deeper qualitative analysis.")

# --- Page 3: Comparison & Quality Assessment ---
elif st.session_state['current_page'] == "Comparison & Quality Assessment":
    st.title("RAG vs. Full-Context Comparison & Summary Quality Assessment")
    st.markdown(f"As a financial analyst, it's crucial to justify RAG's benefits quantitatively. This section compares RAG's cost and quality against full-context summarization and provides an objective assessment of summary quality.")
    st.markdown(f"")

    if not st.session_state['processing_complete'] or not st.session_state['chroma_collection']:
        st.warning("Please upload and process a 10-K document on the 'Upload 10-K & Process' page first.")
    elif not st.session_state['brief_results']:
        st.warning("Please generate a 'Multi-Aspect Research Brief' first to select a topic for comparison.")
    else:
        st.subheader("1. Select a Topic for Detailed Comparison")
        brief_labels = [res['label'] for res in st.session_state['brief_results']]
        selected_topic_label = st.selectbox(
            "Choose a topic from your generated brief:",
            options=brief_labels,
            key='compare_topic_select'
        )
        
        st.session_state['comparison_topic_label'] = selected_topic_label
        selected_brief_result = next((res for res in st.session_state['brief_results'] if res['label'] == selected_topic_label), None)
        st.session_state['comparison_rag_result'] = selected_brief_result
        
        if st.button("Run Comparison & Quality Assessment", disabled=not st.session_state['openai_api_key']):
            if not st.session_state['openai_api_key']:
                st.error("Please enter your OpenAI API key in the sidebar to proceed.")
            else:
                with st.spinner("Running full-context summarization and quality assessment..."):
                    if selected_brief_result:
                        # --- RAG vs. Full-Context Comparison ---
                        st.subheader("2. RAG vs. Full-Context Summarization: Cost & Quality Comparison")
                        st.markdown(f"Here, we compare the `{selected_topic_label}` summary generated by RAG with one produced by feeding the entire 10-K (or a truncated version) directly to the LLM. This highlights RAG's **cost advantage** and its superior ability to produce **grounded and auditable** financial insights.")
                        st.markdown(r"")
                        st.markdown(r"The **RAG Cost Advantage for Targeted Summarization** can be mathematically expressed by comparing the token usage. For a **Full-context summarization**:")
                        st.markdown(r"$$C_{{\text{full}}} = (N_{{\text{10K}\_\text{prompt}}} \times P_{{\text{in}}}) + (N_{{\text{out}}} \times P_{{\text{out}}})$$")
                        st.markdown(r"where $N_{{\text{10K}\_\text{prompt}}}$ is the total input tokens for the full 10-K, $P_{{\text{in}}}$ is input token price, $N_{{\text{out}}}$ is output tokens, and $P_{{\text{out}}}$ is output token price. For a typical 10-K, $N_{{\text{10K}\_\text{prompt}}} \approx 50,000$ tokens.")
                        st.markdown(r"")
                        st.markdown(r"For **RAG-based summarization**:")
                        st.markdown(r"$$C_{{\text{RAG}}} = (k \times N_{{\text{chunk}}} \times P_{{\text{in}}}) + (N_{{\text{out}}} \times P_{{\text{out}}})$$")
                        st.markdown(r"where $k$ is the number of retrieved chunks (e.g., 10), and $N_{{\text{chunk}}}$ is the average tokens per chunk (e.g., 500). Thus, input tokens for RAG are typically $k \times N_{{\text{chunk}}} \approx 5,000$ tokens.")
                        st.markdown(r"")
                        st.markdown(r"The **Summary Compression Ratio** ($CR_{{\text{topic}}}$) quantifies how much the relevant section of the original document is compressed into the summary:")
                        st.markdown(r"$$CR_{{\text{topic}}} = 1 - \frac{{N_{{\text{summary}\_\text{words}}}}}{{\text{N}_{{\text{relevant}\_\text{section}\_\text{words}}}}}$$")
                        st.markdown(r"A higher compression ratio, combined with high quality, indicates efficient summarization.")
                        st.markdown(r"")

                        full_summary, full_cost, full_context_input_tokens = full_context_summarize(
                            topic=selected_brief_result['topic'],
                            full_text=st.session_state['full_text'],
                            model=st.session_state['llm_model']
                        )
                        st.session_state['comparison_full_summary'] = full_summary
                        st.session_state['comparison_full_cost'] = full_cost
                        st.session_state['comparison_full_input_tokens'] = full_context_input_tokens

                        # Comparison Table
                        rag_summary_words = len(selected_brief_result['summary'].split())
                        full_summary_words = len(full_summary.split())
                        
                        # Calculate relevant section words for CR_topic
                        relevant_section_text = ""
                        for sec_name in selected_brief_result['sections_covered']:
                            relevant_section_text += st.session_state['parsed_sections'].get(sec_name, "") + " "
                        n_relevant_section_words = len(relevant_section_text.split()) if relevant_section_text else 1 # Avoid div by zero

                        cr_rag = (1 - (rag_summary_words / n_relevant_section_words)) if n_relevant_section_words > 0 else 0
                        
                        comparison_data = {
                            "Metric": ["Input tokens", "Cost", "Cost ratio (vs. RAG)", "Summary words", "Has citations", "RAG Compression Ratio"],
                            "RAG": [
                                f"{selected_brief_result['input_tokens']:,}",
                                f"${selected_brief_result['cost']:.4f}",
                                "1.0x",
                                f"{rag_summary_words:,}",
                                "Yes",
                                f"{cr_rag:.2f}" if n_relevant_section_words > 0 else "N/A"
                            ],
                            "Full-Context": [
                                f"{full_context_input_tokens:,}",
                                f"${full_cost:.4f}",
                                f"{full_cost / max(0.0001, selected_brief_result['cost']):.1f}x",
                                f"{full_summary_words:,}",
                                "No",
                                "N/A"
                            ]
                        }
                        st.table(pd.DataFrame(comparison_data))

                        st.markdown("---")
                        st.markdown("**RAG Summary:**")
                        st.markdown(selected_brief_result['summary'])
                        st.markdown("---")
                        st.markdown("**Full-Context Summary:**")
                        st.markdown(full_summary)
                        st.markdown("---")
                        st.info("The comparison demonstrates RAG's significant cost reduction and critical traceability through citations, which is non-negotiable for compliance in financial analysis.")

                        # --- Summary Quality Assessment ---
                        st.subheader("3. Summary Quality Assessment: Coverage and Faithfulness")
                        st.markdown(f"The quality of insights is paramount. A summary must be **comprehensive** (Coverage) and **accurate** (Faithfulness). We use an **'LLM-as-judge' framework** to objectively evaluate these dimensions against predefined criteria and the source chunks.")
                        st.markdown(f"")
                        st.markdown(r"**Faithfulness Score ($F$):** The proportion of claims in the summary that are directly supported by the retrieved source chunks. Target: $F > 95\%$. Claims below $F = 90\%$ indicate prompt or retrieval issues or the LLM introducing outside information.")
                        st.markdown(r"$$F = \frac{{\text{|supported claims|}}}{{\text{|total claims in summary|}}}$$")
                        st.markdown(r"")
                        st.markdown(r"**Coverage Scores:** Rated on a 1-5 scale for Breadth, Depth, and Completeness. These are qualitative assessments of how well the summary addresses the topic.")
                        st.markdown(f"")

                        coverage_scores, faithfulness_scores = assess_summary_quality(
                            summary=selected_brief_result['summary'],
                            retrieved_chunks=selected_brief_result['retrieved_chunks'],
                            topic=selected_brief_result['topic'],
                            model=st.session_state['llm_model']
                        )
                        st.session_state['quality_assessments'][selected_topic_label] = (coverage_scores, faithfulness_scores)

                        st.markdown(f"**SUMMARY QUALITY ASSESSMENT for `{selected_topic_label}`:**")
                        st.markdown(f"- **Coverage:**")
                        st.markdown(f"  - Breadth: `{coverage_scores.get('breadth', 'N/A')}/5`")
                        st.markdown(f"  - Depth: `{coverage_scores.get('depth', 'N/A')}/5`")
                        st.markdown(f"  - Completeness: `{coverage_scores.get('completeness', 'N/A')}/5`")
                        if coverage_scores.get('missing_topics'):
                            st.markdown(f"  - *Missing Topics: {', '.join(coverage_scores['missing_topics'])}*")
                        
                        st.markdown(f"- **Faithfulness:**")
                        supported_claims = faithfulness_scores.get('supported_claims', 0)
                        total_claims = faithfulness_scores.get('total_claims', 1)
                        faithfulness_score = supported_claims / max(1, total_claims)
                        st.markdown(f"  - Supported Claims: `{supported_claims}/{total_claims}` ({faithfulness_score:.0%})")
                        if faithfulness_scores.get('unsupported_claims'):
                            st.markdown(f"  - *Unsupported Claims: {'; '.join(faithfulness_scores['unsupported_claims'])}*")
                        
                        st.info("This dual assessment provides actionable diagnostics: Low Coverage suggests improving retrieval (broaden query, increase `k`, adjust `section_filter`). Low Faithfulness suggests refining the system prompt for stricter adherence or reducing `temperature` to prevent hallucinations.")
                    else:
                        st.error("Could not retrieve the selected brief result for comparison.")
```
