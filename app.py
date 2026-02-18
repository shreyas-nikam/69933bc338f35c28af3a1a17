import streamlit as st
import os
import fitz
import pandas as pd
import json
import numpy as np
import base64
import re
from typing import List, Dict, Any, Optional
from source import *

st.set_page_config(
    page_title="QuLab: Lab 29: Summarizing a 10-K with RAG", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 29: Summarizing a 10-K with RAG")
st.divider()

# --- st.session_state Initialization ---
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = os.environ.get("OPENAI_API_KEY", "")

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
if 'llm_model' not in st.session_state:
    st.session_state['llm_model'] = 'gpt-3.5-turbo'
if 'client_llm' not in st.session_state:
    st.session_state.client_llm = None
if 'embedder_func' not in st.session_state:
    st.session_state.embedder_func = None
if 'chroma_persist_dir' not in st.session_state:
    st.session_state['chroma_persist_dir'] = './chroma_db'
if 'custom_topics' not in st.session_state:
    st.session_state['custom_topics'] = []
# --- Sidebar Navigation & Config ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Choose a section:",
    ["Upload 10-K & Process", "Multi-Aspect Research Brief",
        "Comparison & Quality Assessment"],
    key='page_select_box'
)

st.sidebar.subheader("Configuration")
llm_options = ['gpt-4o', 'gpt-3.5-turbo']
st.session_state['llm_model'] = st.sidebar.selectbox(
    "Select LLM Model:",
    llm_options,
    index=1 if os.environ.get("OPENAI_API_KEY") else 0,
    key='llm_model_select'
)

api_key_input = st.sidebar.text_input("OpenAI API Key", type="password",
                                      help="Enter your OpenAI API Key (e.g., sk-...)", value=st.session_state['openai_api_key'])
if api_key_input:
    st.session_state['openai_api_key'] = api_key_input
    st.session_state.client_llm = OpenAI(
        api_key=st.session_state['openai_api_key'],)
    # Initialize embedder_func when API key is set
    st.session_state.embedder_func = MyEmbeddingFunction(
        openai_api_key=st.session_state['openai_api_key'])

    # Auto-load existing collection if available
    if not st.session_state['processing_complete'] and os.path.exists('AAPL_10K.pdf'):
        try:
            from chromadb import PersistentClient
            pdf_path = 'AAPL_10K.pdf'
            base_name = re.sub(
                r'[^a-zA-Z0-9._-]', '_', pdf_path.replace('.pdf', '').replace('.PDF', ''))
            base_name = re.sub(r'^[^a-zA-Z0-9]+', '', base_name)
            base_name = re.sub(r'[^a-zA-Z0-9]+$', '', base_name)
            collection_name = f"{base_name}_sections" if base_name else "document_sections"

            persist_client = PersistentClient(
                path=st.session_state['chroma_persist_dir'])
            existing_collection = persist_client.get_collection(
                name=collection_name,
                embedding_function=st.session_state.embedder_func
            )
            if existing_collection.count() > 0:
                # Load parsed sections and full text
                if st.session_state['parsed_sections'] is None:
                    st.session_state['parsed_sections'] = parse_10k_sections(
                        pdf_path)
                if st.session_state['full_text'] is None:
                    doc = fitz.open(pdf_path)
                    st.session_state['full_text'] = "".join(
                        [page.get_text() for page in doc])
                    doc.close()

                st.session_state['chroma_collection'] = existing_collection
                st.session_state['processing_complete'] = True
                st.session_state['pdf_uploaded_name'] = pdf_path
        except Exception:
            pass  # Collection doesn't exist yet, will be created on first processing

# --- Page 1: Upload 10-K & Process ---
if page_selection == "Upload 10-K & Process":
    st.header("Summarizing a 10-K with RAG: Analyst's Edge")
    st.markdown(f"As a **CFA Charterholder and Investment Professional**, extracting targeted information from lengthy **10-K annual filings** is crucial yet time-consuming. Generic, full-document summaries often lack the precision required for robust financial analysis.")
    st.markdown(
        f"This application transforms this process by implementing **Retrieval-Augmented Generation (RAG)** to:")
    st.markdown(
        f"- **Understand the structure** of 10-K filings through section-aware chunking.")
    st.markdown(
        f"- **Retrieve only the most relevant text** for specific research questions.")
    st.markdown(
        f"- **Synthesize these passages into analytical, cited summaries** using a Large Language Model (LLM).")
    st.markdown(
        f"This dramatically accelerates research, provides auditable trails, and reduces LLM operational costs.")

    st.subheader("1. Working with 10-K Document")

    # Check if AAPL_10K.pdf exists
    pdf_path = "AAPL_10K.pdf"
    if os.path.exists(pdf_path):
        st.success(f"📄 We are working with the 10-K document: **{pdf_path}**")

        # Display PDF viewer
        st.pdf(pdf_path,)

        # Set the uploaded file name if not already set
        if not st.session_state['pdf_uploaded_name']:
            st.session_state['pdf_uploaded_name'] = pdf_path
    else:
        st.error(
            f"❌ The file '{pdf_path}' was not found. Please ensure it exists in the working directory.")
        st.stop()

    if True:  # Always show the process button
        if not st.session_state['openai_api_key']:
            st.warning(
                "Please enter your OpenAI API key in the sidebar to enable processing.")
        if st.button("Process 10-K", disabled=st.session_state['processing_complete'] or not st.session_state['openai_api_key']):
            if not st.session_state['openai_api_key']:
                st.error(
                    "Please enter your OpenAI API key in the sidebar to proceed.")
            else:
                with st.spinner("Step 1/2: Parsing 10-K sections and extracting full text..."):
                    try:
                        sections = parse_10k_sections(
                            "AAPL_10K.pdf")
                        st.session_state['parsed_sections'] = sections

                        # Extract full text from PDF
                        doc = fitz.open("AAPL_10K.pdf")
                        st.session_state['full_text'] = "".join(
                            [page.get_text() for page in doc])
                        doc.close()

                        # Initialize embedder_func if not already done
                        if st.session_state.embedder_func is None:
                            st.session_state.embedder_func = MyEmbeddingFunction(
                                openai_api_key=st.session_state['openai_api_key'])

                        with st.spinner("Step 2/2: Chunking sections, creating embeddings, and storing in ChromaDB..."):
                            # Sanitize collection name for ChromaDB:
                            # Must be 3-512 chars, [a-zA-Z0-9._-], start/end with alphanumeric
                            base_name = re.sub(
                                r'[^a-zA-Z0-9._-]', '_', pdf_path.replace('.pdf', '').replace('.PDF', ''))
                            # Ensure starts with alphanumeric
                            base_name = re.sub(
                                r'^[^a-zA-Z0-9]+', '', base_name)
                            # Ensure ends with alphanumeric
                            base_name = re.sub(
                                r'[^a-zA-Z0-9]+$', '', base_name)
                            collection_name = f"{base_name}_sections" if base_name else "document_sections"

                            # Use persistent ChromaDB client
                            from chromadb import PersistentClient
                            persist_client = PersistentClient(
                                path=st.session_state['chroma_persist_dir'])

                            # Check if collection already exists and has data
                            try:
                                existing_collection = persist_client.get_collection(
                                    name=collection_name,
                                    embedding_function=st.session_state.embedder_func
                                )
                                if existing_collection.count() > 0:
                                    st.info(
                                        f"✅ Found existing knowledge base with {existing_collection.count()} chunks. Using cached embeddings.")
                                    chroma_collection = existing_collection
                                else:
                                    # Collection exists but is empty, populate it
                                    chroma_collection = chunk_and_embed_sections(
                                        st.session_state['parsed_sections'],
                                        st.session_state.embedder_func,
                                        collection_name=collection_name,
                                        chroma_client=persist_client
                                    )
                            except Exception:
                                # Collection doesn't exist, create it
                                chroma_collection = chunk_and_embed_sections(
                                    st.session_state['parsed_sections'],
                                    st.session_state.embedder_func,
                                    collection_name=collection_name,
                                    chroma_client=persist_client
                                )

                            st.session_state['chroma_collection'] = chroma_collection
                            st.session_state['processing_complete'] = True

                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        st.session_state['processing_complete'] = False
                        st.session_state['chroma_collection'] = None

        # Show the explanatory content when processing is complete
        if st.session_state['processing_complete'] and st.session_state['parsed_sections']:
            st.success("10-K Sections Parsed!")

            st.subheader(
                "2. Building the Knowledge Base: Vector Embeddings and Database")
            st.markdown(f"Now, we convert each text chunk into a **vector embedding** to capture semantic meaning. These are stored in a **vector database** (ChromaDB) for **Topic-Directed Retrieval**, finding contextually similar chunks to queries.")
            st.markdown(r"The mathematical foundation for semantic similarity often relies on measures like **cosine similarity**, which quantifies the angular distance between two vectors. Given two embedding vectors, $A$ and $B$, their cosine similarity is calculated as:")
            st.markdown(
                r"""
$$
\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
$$
""")
            st.markdown(
                r"where $A \cdot B$ is the dot product of $A$ and $B$, and $\|A\|$ and $\|B\|$ are their magnitudes.")
            st.markdown(f"")
            st.info("For an analyst, this means we now have a powerful, semantic index. When you ask a question, the system won't just look for keywords; it will understand the *meaning* of your query and retrieve the most relevant passages, especially with section metadata filters.")

            if st.session_state['chroma_collection']:
                st.success(
                    f"Knowledge Base Ready! Indexed {st.session_state['chroma_collection'].count()} chunks from '{pdf_path}'.")

            # Let the user see the content of each section through dropdown
            st.markdown("---")
            st.markdown("**View Section Content:**")
            section_names = list(st.session_state['parsed_sections'].keys())
            selected_section = st.selectbox(
                "Select a section to view its content:", section_names)
            st.text_area(
                "Section Content:", value=st.session_state['parsed_sections'][selected_section], height=300)

    elif st.session_state['pdf_uploaded_name'] and st.session_state['processing_complete']:
        st.success(
            f"✅ Using previously processed 10-K: '{st.session_state['pdf_uploaded_name']}'. Navigate to other sections.")

# --- Page 2: Multi-Aspect Research Brief ---
elif page_selection == "Multi-Aspect Research Brief":
    st.header("Multi-Aspect Research Brief Generation")
    st.markdown(f"As an analyst, a comprehensive research update requires structured overviews across multiple financial dimensions. This section allows you to compile a 'Multi-Aspect Research Brief' by systematically applying our RAG pipeline to predefined or custom financial topics.")
    st.markdown(f"")

    if not st.session_state['processing_complete'] or not st.session_state['chroma_collection']:
        st.warning(
            "Please upload and process a 10-K document on the 'Upload 10-K & Process' page first.")
    else:
        st.subheader(
            f"1. Define Research Topics for '{st.session_state['pdf_uploaded_name']}'")
        st.info("Select predefined topics or add your own custom topic.")

        selected_brief_configs = []

        # Predefined topics
        st.markdown("**Predefined Topics:**")
        for i, topic_config in enumerate(BRIEF_TOPICS):
            checkbox_key = f"brief_topic_checkbox_{i}"
            sections_display = ', '.join(
                topic_config['sections']) if topic_config['sections'] else 'All'
            if st.checkbox(f"**{topic_config['brief_label']}** ({sections_display})", key=checkbox_key, value=True):
                selected_brief_configs.append(topic_config)

        # Display existing custom topics
        if st.session_state['custom_topics']:
            st.markdown("**Custom Topics:**")
            col_clear = st.columns([4, 1])
            with col_clear[1]:
                if st.button("🗑️ Clear All Custom", help="Remove all custom topics"):
                    st.session_state['custom_topics'] = []
                    st.rerun()

            for i, custom_topic in enumerate(st.session_state['custom_topics']):
                col1, col2 = st.columns([4, 1])
                with col1:
                    checkbox_key = f"custom_topic_checkbox_{i}"
                    sections_display = ', '.join(
                        custom_topic['sections']) if custom_topic['sections'] else 'All'
                    if st.checkbox(f"**{custom_topic['brief_label']}** ({sections_display})", key=checkbox_key, value=True):
                        selected_brief_configs.append(custom_topic)
                with col2:
                    if st.button("🗑️", key=f"delete_custom_{i}", help="Remove this custom topic"):
                        st.session_state['custom_topics'].pop(i)
                        st.rerun()

        # Custom topic input
        st.markdown("---")
        st.markdown("**Add a Custom Topic:**")

        with st.form("custom_topic_form", clear_on_submit=True):
            custom_topic_query = st.text_area(
                "Custom Topic Query:", value="", placeholder="e.g., 'Impact of recent mergers and acquisitions on company strategy'")
            all_sections_available = sorted(list(st.session_state['parsed_sections'].keys(
            ))) if st.session_state['parsed_sections'] else []
            custom_topic_sections = st.multiselect(
                "Filter custom topic to specific 10-K sections (optional):",
                options=all_sections_available,
                default=[],
            )
            custom_topic_label = st.text_input(
                "Custom Topic Label (optional):",
                value="",
                placeholder="e.g., 'M&A Impact'"
            )

            submit_button = st.form_submit_button("Add Custom Topic to Brief")

            if submit_button and custom_topic_query.strip():
                custom_config = {
                    'topic': custom_topic_query.strip(),
                    'sections': custom_topic_sections if custom_topic_sections else None,
                    'brief_label': custom_topic_label.strip() if custom_topic_label.strip() else f"CUSTOM TOPIC {len(st.session_state['custom_topics']) + 1}"
                }
                st.session_state['custom_topics'].append(custom_config)
                st.success(
                    f"✅ Custom topic '{custom_config['brief_label']}' added!")
                st.rerun()

        st.markdown("---")

        # Show summary of selected topics
        if selected_brief_configs:
            st.info(
                f"📊 **{len(selected_brief_configs)} topic(s) selected** for analysis")
        else:
            st.warning(
                "⚠️ Please select at least one topic to generate a research brief")

        if st.button("Generate Multi-Aspect Brief", disabled=not selected_brief_configs or not st.session_state['openai_api_key']):
            if not st.session_state['openai_api_key']:
                st.error(
                    "Please enter your OpenAI API key in the sidebar to proceed.")
            else:
                try:
                    st.session_state['brief_results'] = []
                    st.session_state['total_brief_cost'] = 0.0
                    progress_text = "Generating research brief, please wait..."
                    my_bar = st.progress(0, text=progress_text)

                    brief_output = []
                    for i, topic_config in enumerate(selected_brief_configs):
                        my_bar.progress((i + 1) / len(selected_brief_configs),
                                        text=f"Generating summary for: {topic_config['brief_label']}")
                        result = rag_summarize(
                            topic_query=topic_config['topic'],
                            collection=st.session_state['chroma_collection'],
                            embedder_func=st.session_state.embedder_func,
                            llm_client=st.session_state.client_llm,
                            section_filter=topic_config['sections'],
                            k=8,
                            model=st.session_state['llm_model']
                        )
                        result['label'] = topic_config['brief_label']
                        st.session_state['brief_results'].append(result)
                        st.session_state['total_brief_cost'] += result['cost']
                        brief_output.append(result)

                    my_bar.empty()
                    st.success("✅ Multi-Aspect Research Brief Generated!")
                except Exception as e:
                    st.error(f"❌ Error generating brief: {str(e)}")
                    import traceback
                    st.error(f"Details: {traceback.format_exc()}")

        # Display previously generated results
        if st.session_state['brief_results']:
            st.markdown("---")

            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(
                    f"2. 10-K RESEARCH BRIEF: {st.session_state['pdf_uploaded_name'].replace('_10K_2024.pdf', '').replace('.pdf', '')} (FY2024)")
            with col2:
                if st.button("🔄 Clear Results"):
                    st.session_state['brief_results'] = []
                    st.session_state['total_brief_cost'] = 0.0
                    st.rerun()

            st.markdown(
                f"Generated via RAG | Total cost for brief: **${st.session_state['total_brief_cost']:.4f}**")

            for section in st.session_state['brief_results']:
                st.markdown(f"--- **{section['label']}** ---")
                st.markdown(
                    f"*(Sources: {', '.join(section['sections_covered'])})*")
                st.markdown(section['summary'])
                st.markdown(f"*(Cost: ${section['cost']:.4f})*")
                st.markdown(f"")
            st.info("This structured brief provides initial summaries, allowing you to rapidly grasp key points, verify sources, and concentrate on deeper qualitative analysis.")

            # Option to download the brief
            brief_text = f"10-K RESEARCH BRIEF: {st.session_state['pdf_uploaded_name']}\n\n"
            for section in st.session_state['brief_results']:
                brief_text += f"\n--- {section['label']} ---\n"
                brief_text += f"Sources: {', '.join(section['sections_covered'])}\n\n"
                brief_text += f"{section['summary']}\n\n"
                brief_text += f"Cost: ${section['cost']:.4f}\n\n"

            st.download_button(
                label="📥 Download Research Brief",
                data=brief_text,
                file_name=f"research_brief_{st.session_state['pdf_uploaded_name'].replace('.pdf', '')}.txt",
                mime="text/plain"
            )

# --- Page 3: Comparison & Quality Assessment ---
elif page_selection == "Comparison & Quality Assessment":
    st.header("RAG vs. Full-Context Comparison & Summary Quality Assessment")
    st.markdown(f"As a financial analyst, it's crucial to justify RAG's benefits quantitatively. This section compares RAG's cost and quality against full-context summarization and provides an objective assessment of summary quality.")
    st.markdown(f"")

    if not st.session_state['processing_complete'] or not st.session_state['chroma_collection']:
        st.warning(
            "Please upload and process a 10-K document on the 'Upload 10-K & Process' page first.")
    elif not st.session_state['brief_results']:
        st.warning(
            "Please generate a 'Multi-Aspect Research Brief' first to select a topic for comparison.")
    else:
        st.subheader("1. Select a Topic for Detailed Comparison")
        brief_labels = [res['label']
                        for res in st.session_state['brief_results']]
        selected_topic_label = st.selectbox(
            "Choose a topic from your generated brief:",
            options=brief_labels,
            key='compare_topic_select'
        )

        st.session_state['comparison_topic_label'] = selected_topic_label
        selected_brief_result = next(
            (res for res in st.session_state['brief_results'] if res['label'] == selected_topic_label), None)
        st.session_state['comparison_rag_result'] = selected_brief_result

        if st.button("Run Comparison & Quality Assessment", disabled=not st.session_state['openai_api_key']):
            if not st.session_state['openai_api_key']:
                st.error(
                    "Please enter your OpenAI API key in the sidebar to proceed.")
            else:
                with st.spinner("Running full-context summarization and quality assessment..."):
                    if selected_brief_result:
                        # --- RAG vs. Full-Context Comparison ---
                        st.subheader(
                            "2. RAG vs. Full-Context Summarization: Cost & Quality Comparison")
                        st.markdown(
                            f"Here, we compare the `{selected_topic_label}` summary generated by RAG with one produced by feeding the entire 10-K (or a truncated version) directly to the LLM. This highlights RAG's **cost advantage** and its superior ability to produce **grounded and auditable** financial insights.")
                        st.markdown(r"")
                        st.markdown(
                            r"The **RAG Cost Advantage for Targeted Summarization** can be mathematically expressed by comparing the token usage. For a **Full-context summarization**:")
                        st.markdown(
                            r"""
$$
C_{\text{full}} = (N_{\text{10K}\_\text{prompt}} \times P_{\text{in}}) + (N_{\text{out}} \times P_{\text{out}})
$$
""")
                        st.markdown(r"where $N_{\text{10K}\_\text{prompt}}$ is the total input tokens for the full 10-K, $P_{\text{in}}$ is input token price, $N_{\text{out}}$ is output tokens, and $P_{\text{out}}$ is output token price. For a typical 10-K, $N_{\text{10K}\_\text{prompt}} \approx 50,000$ tokens.")
                        st.markdown(r"")
                        st.markdown(r"For **RAG-based summarization**:")
                        st.markdown(
                            r"""
$$
C_{\text{RAG}} = (k \times N_{\text{chunk}} \times P_{\text{in}}) + (N_{\text{out}} \times P_{\text{out}})
$$
""")
                        st.markdown(
                            r"where $k$ is the number of retrieved chunks (e.g., 10), and $N_{\text{chunk}}$ is the average tokens per chunk (e.g., 500). Thus, input tokens for RAG are typically $k \times N_{\text{chunk}} \approx 5,000$ tokens.")
                        st.markdown(r"")
                        st.markdown(
                            r"The **Summary Compression Ratio** ($CR_{\text{topic}}$) quantifies how much the relevant section of the original document is compressed into the summary:")
                        st.markdown(
                            r"""
$$
CR_{\text{topic}} = 1 - \frac{N_{\text{summary}\_\text{words}}}{\text{N}_{\text{relevant}\_\text{section}\_\text{words}}}
$$
""")
                        st.markdown(
                            r"A higher compression ratio, combined with high quality, indicates efficient summarization.")
                        st.markdown(r"")

                        full_context_result = full_context_summarize(
                            topic=selected_brief_result['topic'],
                            full_text=st.session_state['full_text'],
                            llm_client=st.session_state.client_llm,
                            model=st.session_state['llm_model']
                        )
                        st.session_state['comparison_full_summary'] = full_context_result['summary']
                        st.session_state['comparison_full_cost'] = full_context_result['cost']
                        st.session_state['comparison_full_input_tokens'] = full_context_result['input_tokens']

                        # Comparison Table
                        rag_summary_words = len(
                            selected_brief_result['summary'].split())
                        full_summary_words = len(
                            st.session_state['comparison_full_summary'].split())

                        # Calculate relevant section words for CR_topic
                        relevant_section_text = ""
                        for sec_name in selected_brief_result['sections_covered']:
                            relevant_section_text += st.session_state['parsed_sections'].get(
                                sec_name, "") + " "
                        n_relevant_section_words = len(relevant_section_text.split(
                        )) if relevant_section_text else 1  # Avoid div by zero

                        cr_rag = (1 - (rag_summary_words / n_relevant_section_words)
                                  ) if n_relevant_section_words > 0 else 0

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
                                f"{st.session_state['comparison_full_input_tokens']:,}",
                                f"${st.session_state['comparison_full_cost']:.4f}",
                                f"{st.session_state['comparison_full_cost'] / max(0.0001, selected_brief_result['cost']):.1f}x",
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
                        st.markdown(
                            st.session_state['comparison_full_summary'])
                        st.markdown("---")
                        st.info("The comparison demonstrates RAG's significant cost reduction and critical traceability through citations, which is non-negotiable for compliance in financial analysis.")

                        # --- Summary Quality Assessment ---
                        st.subheader(
                            "3. Summary Quality Assessment: Coverage and Faithfulness")
                        st.markdown(f"The quality of insights is paramount. A summary must be **comprehensive** (Coverage) and **accurate** (Faithfulness). We use an **'LLM-as-judge' framework** to objectively evaluate these dimensions against predefined criteria and the source chunks.")
                        st.markdown(f"")
                        st.markdown(r"**Faithfulness Score ($F$):** The proportion of claims in the summary that are directly supported by the retrieved source chunks. Target: $F > 95\%$. Claims below $F = 90\%$ indicate prompt or retrieval issues or the LLM introducing outside information.")
                        st.markdown(
                            r"""
$$
F = \frac{\text{|supported claims|}}{\text{|total claims in summary|}}
$$
""")
                        st.markdown(r"")
                        st.markdown(
                            r"**Coverage Scores:** Rated on a 1-5 scale for Breadth, Depth, and Completeness. These are qualitative assessments of how well the summary addresses the topic.")
                        st.markdown(f"")

                        quality_result = assess_summary_quality(
                            summary=selected_brief_result['summary'],
                            retrieved_chunks=selected_brief_result['retrieved_chunks'],
                            topic=selected_brief_result['topic'],
                            llm_client=st.session_state.client_llm,
                            model=st.session_state['llm_model']
                        )
                        coverage_scores = quality_result['coverage']
                        faithfulness_scores = quality_result['faithfulness']
                        st.session_state['quality_assessments'][selected_topic_label] = (
                            coverage_scores, faithfulness_scores)

                        st.markdown(
                            f"**SUMMARY QUALITY ASSESSMENT for `{selected_topic_label}`:**")
                        st.markdown(f"- **Coverage:**")
                        st.markdown(
                            f"  - Breadth: `{coverage_scores.get('breadth', 'N/A')}/5`")
                        st.markdown(
                            f"  - Depth: `{coverage_scores.get('depth', 'N/A')}/5`")
                        st.markdown(
                            f"  - Completeness: `{coverage_scores.get('completeness', 'N/A')}/5`")
                        if coverage_scores.get('missing_topics'):
                            st.markdown(
                                f"  - *Missing Topics: {', '.join(coverage_scores['missing_topics'])}*")

                        st.markdown(f"- **Faithfulness:**")
                        supported_claims = faithfulness_scores.get(
                            'supported_claims', 0)
                        total_claims = faithfulness_scores.get(
                            'total_claims', 1)
                        faithfulness_score = supported_claims / \
                            max(1, total_claims)
                        st.markdown(
                            f"  - Supported Claims: `{supported_claims}/{total_claims}` ({faithfulness_score:.0%})")
                        if faithfulness_scores.get('unsupported_claims'):
                            st.markdown(
                                f"  - *Unsupported Claims: {'; '.join(faithfulness_scores['unsupported_claims'])}*")

                        st.info("This dual assessment provides actionable diagnostics: Low Coverage suggests improving retrieval (broaden query, increase `k`, adjust `section_filter`). Low Faithfulness suggests refining the system prompt for stricter adherence or reducing `temperature` to prevent hallucinations.")
                    else:
                        st.error(
                            "Could not retrieve the selected brief result for comparison.")

# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
