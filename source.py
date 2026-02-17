import fitz # PyMuPDF (pypdfium2 is its backend)
import re
import os
from openai import OpenAI
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client, Documents, EmbeddingFunction, Embeddings
from chromadb.api.models.Collection import Collection # Import Collection
import tiktoken
import pandas as pd
import json
import numpy as np
import textwrap
from typing import List, Dict, Any, Optional

# Import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# Define global model pricing for convenience
MODEL_PRICING = {
    'gpt-4o': {'input_per_1k_tokens': 0.005, 'output_per_1k_tokens': 0.015},
    'gpt-3.5-turbo': {'input_per_1k_tokens': 0.0005, 'output_per_1k_tokens': 0.0015},
}

class MyEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for ChromaDB to use OpenAIEmbeddings.
    """
    def __init__(self, model_name: str = 'text-embedding-ada-002', openai_api_key: Optional[str] = None):
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass it to MyEmbeddingFunction.")
        self.embedder = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

    def __call__(self, input: Documents) -> Embeddings:
        # Use embed_documents method for OpenAIEmbeddings
        return self.embedder.embed_documents(input)

def parse_10k_sections(filepath: str) -> Dict[str, str]:
    """
    Parses a 10-K PDF into named sections based on Item headers.

    Args:
        filepath (str): The path to the 10-K PDF document.

    Returns:
        Dict[str, str]: A dictionary where keys are section names (e.g., 'Item 1A')
                          and values are the extracted text for that section.
    """
    try:
        doc = fitz.open(filepath)
        full_text = "".join([page.get_text() for page in doc])
        doc.close()
    except Exception as e:
        print(f"Error opening or reading PDF at {filepath}: {e}")
        return {}

    # Common 10-K section headers (regex patterns for robust matching)
    section_patterns = {
        'Item 1': r'Item\s+1[\.\s]+Business',
        'Item 1A': r'Item\s+1A[\.\s]+Risk\s+Factors',
        'Item 7': r'Item\s+7[\.\s]+Management.?s\s+Discussion\s+and\s+Analysis',
        'Item 7A': r'Item\s+7A[\.\s]+Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk',
        'Item 8': r'Item\s+8[\.\s]+Financial\s+Statements\s+and\s+Supplementary\s+Data',
        'Item 9': r'Item\s+9[\.\s]+Changes\s+in\s+and\s+Disagreements\s+With\s+Accountants\s+on\s+Accounting\s+and\s+Financial\s+Disclosure',
        'Item 11': r'Item\s+11[\.\s]+Executive\s+Compensation',
    }

    positions = []
    for name, pattern in section_patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            positions.append((match.start(), name))

    positions.sort()

    sections = {}
    for i, (start, name) in enumerate(positions):
        end = positions[i+1][0] if i+1 < len(positions) else len(full_text)
        sections[name] = full_text[start:end].strip()

    print("Parsed 10-K sections:")
    for name, text in sections.items():
        print(f"  {name}: {len(text.split()):,} words")

    return sections

def _create_dummy_pdf_if_not_exists(filename: str, content: str) -> bool:
    """
    Creates a dummy PDF file for demonstration purposes if it doesn't already exist.
    Returns True if created or exists, False if an error occurred.
    """
    if os.path.exists(filename):
        print(f"PDF file '{filename}' already exists. Skipping dummy creation.")
        return True
    
    try:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 70), content, fontsize=10)
        doc.save(filename)
        doc.close()
        print(f"Created dummy PDF: {filename}")
        return True
    except Exception as e:
        print(f"Error creating dummy PDF: {e}. Please ensure PyMuPDF is installed correctly.")
        return False

def chunk_and_embed_sections(
    sections: Dict[str, str],
    embedder_func: EmbeddingFunction,
    collection_name: str = "tenk_sections",
    chroma_client: Optional[Client] = None
) -> Collection:
    """
    Chunks each section, creates embeddings, and stores them in a ChromaDB collection,
    preserving section metadata.

    Args:
        sections (Dict[str, str]): Dictionary of 10-K sections.
        embedder_func (EmbeddingFunction): Custom embedding function for ChromaDB.
        collection_name (str): Name for the ChromaDB collection.
        chroma_client (Optional[Client]): An initialized ChromaDB client. If None, an in-memory client is used.

    Returns:
        Collection: The ChromaDB collection object.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ""]
    )

    all_chunks = []
    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue
        chunks = splitter.split_text(section_text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'id': f"{section_name}_chunk_{i}",
                'text': chunk,
                'metadata': {
                    'section': section_name,
                    'chunk_idx': i,
                    'n_words': len(chunk.split()),
                    'n_chars': len(chunk)
                }
            })

    client = chroma_client if chroma_client else Client()
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder_func)

    texts = [c['text'] for c in all_chunks]
    metadatas = [c['metadata'] for c in all_chunks]
    ids = [c['id'] for c in all_chunks]

    if texts:
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Indexed {collection.count()} chunks across {len(sections)} sections in collection '{collection_name}'.")
    else:
        print("No chunks to index.")

    return collection

def retrieve(
    topic_query: str,
    collection: Collection,
    embedder_func: EmbeddingFunction,
    section_filter: Optional[List[str]] = None,
    k: int = 8
) -> List[Dict[str, Any]]:
    """
    Retrieves relevant chunks from the ChromaDB collection.

    Args:
        topic_query (str): The user's query about a topic.
        collection (Collection): The ChromaDB collection object.
        embedder_func (EmbeddingFunction): The embedding function to use for the query.
        section_filter (Optional[List[str]]): List of 10-K section names to filter retrieval (e.g., ['Item 1A']).
        k (int): Number of top-k chunks to retrieve.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a retrieved chunk
                                 with its document text, metadata, and distance (relevance).
    """
    query_embedding = embedder_func([topic_query])[0]

    where_filter = {"section": {"$in": section_filter}} if section_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter,
        include=['documents', 'metadatas', 'distances']
    )

    retrieved_chunks = []
    if results['documents'] and results['metadatas']:
        for i in range(len(results['documents'][0])):
            retrieved_chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
    return retrieved_chunks

def rag_summarize(
    topic_query: str,
    collection: Collection,
    embedder_func: EmbeddingFunction,
    llm_client: OpenAI,
    section_filter: Optional[List[str]] = None,
    k: int = 8,
    model: str = 'gpt-4o',
    model_pricing_dict: Dict[str, Dict[str, float]] = MODEL_PRICING
) -> Dict[str, Any]:
    """
    Retrieve-then-summarize: get relevant chunks for a topic,
    then produce a coherent analytical summary.

    Args:
        topic_query (str): The user's query about a topic.
        collection (Collection): The ChromaDB collection object.
        embedder_func (EmbeddingFunction): The embedding function.
        llm_client (OpenAI): The initialized OpenAI client.
        section_filter (Optional[List[str]]): List of 10-K section names to filter retrieval.
        k (int): Number of top-k chunks to retrieve.
        model (str): The OpenAI LLM model to use (e.g., 'gpt-4o', 'gpt-3.5-turbo').
        model_pricing_dict (Dict[str, Dict[str, float]]): Dictionary containing LLM pricing.

    Returns:
        Dict[str, Any]: A dictionary containing the summary, cost, chunks used, etc.
    """
    retrieved_chunks = retrieve(topic_query, collection, embedder_func, section_filter, k)

    if not retrieved_chunks:
        return {
            'topic': topic_query,
            'summary': "No relevant information found for this topic.",
            'n_chunks_used': 0,
            'sections_covered': [],
            'cost': 0.0,
            'input_tokens': 0,
            'retrieved_chunks': []
        }

    SUMMARIZE_SYSTEM = """You are a senior equity research analyst producing targeted analysis from SEC 10-K filings.
RULES:
1. Summarize ONLY from the provided excerpts. Do NOT add outside knowledge.
2. Organize the summary with clear sub-topics.
3. Include specific numbers, percentages, and dollar amounts where mentioned.
4. Note any year-over-year changes or trends.
5. Flag material uncertainties or qualifications ("management noted...", "subject to...").
6. Cite sources as [Section: Item X, Chunk Y] after key claims.
7. If the retrieved excerpts do not adequately cover the topic, state what is missing."""

    context_parts = []
    for chunk_data in retrieved_chunks:
        meta = chunk_data['metadata']
        sim = 1 - chunk_data['distance']
        context_parts.append(
            f"[Section: {meta['section']}, Chunk {meta['chunk_idx']}] "
            f"(relevance: {sim:.2f})\n{chunk_data['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"Topic: {topic_query}\n\nProduce a focused analytical summary of this topic based on the following 10-K excerpts.\n\nEXCERPTS:\n{context}"}
    ]

    current_model_pricing = model_pricing_dict.get(model, model_pricing_dict['gpt-4o'])

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )

        summary = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1000 * current_model_pricing['input_per_1k_tokens']) + \
               (completion_tokens / 1000 * current_model_pricing['output_per_1k_tokens'])

        sections_covered = list(set(c['metadata']['section'] for c in retrieved_chunks))

        return {
            'topic': topic_query,
            'summary': summary,
            'n_chunks_used': len(retrieved_chunks),
            'sections_covered': sections_covered,
            'cost': cost,
            'input_tokens': input_tokens,
            'retrieved_chunks': retrieved_chunks
        }
    except Exception as e:
        print(f"Error during LLM summarization: {e}")
        return {
            'topic': topic_query,
            'summary': f"Error generating summary: {e}",
            'n_chunks_used': len(retrieved_chunks),
            'sections_covered': [],
            'cost': 0.0,
            'input_tokens': 0,
            'retrieved_chunks': retrieved_chunks
        }

def full_context_summarize(
    topic: str,
    full_text: str,
    llm_client: OpenAI,
    model: str = 'gpt-4o',
    model_pricing_dict: Dict[str, Dict[str, float]] = MODEL_PRICING
) -> Dict[str, Any]:
    """
    Summarize by feeding the entire 10-K (if it fits in context) to the LLM.

    Args:
        topic (str): The topic to summarize.
        full_text (str): The entire text of the 10-K.
        llm_client (OpenAI): The initialized OpenAI client.
        model (str): The OpenAI LLM model to use.
        model_pricing_dict (Dict[str, Dict[str, float]]): Dictionary containing LLM pricing.

    Returns:
        Dict[str, Any]: A dictionary containing the summary, cost, input tokens, etc.
    """
    enc = tiktoken.encoding_for_model(model)
    full_text_tokens = enc.encode(full_text)

    # Max input tokens for the LLM (e.g., 128k for gpt-4o, leaving room for prompt and output)
    max_input_llm_tokens = 120000

    truncated_text = full_text
    if len(full_text_tokens) > max_input_llm_tokens:
        truncated_text = enc.decode(full_text_tokens[:max_input_llm_tokens])
        print(f"Warning: Full 10-K truncated from {len(full_text_tokens):,} to {max_input_llm_tokens:,} tokens for full-context summarization.")

    current_model_pricing = model_pricing_dict.get(model, model_pricing_dict['gpt-4o'])

    SUMMARIZE_SYSTEM = "You are a senior equity research analyst producing targeted analysis from SEC 10-K filings."
    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"Topic: {topic}\n\nSummarize this topic from the following 10-K filing.\n\nFILING:\n{truncated_text}"}
    ]

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )
        summary = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1000 * current_model_pricing['input_per_1k_tokens']) + \
               (completion_tokens / 1000 * current_model_pricing['output_per_1k_tokens'])
        return {
            'topic': topic,
            'summary': summary,
            'cost': cost,
            'input_tokens': input_tokens,
            'full_text_tokens': len(full_text_tokens) # Original full text token count
        }
    except Exception as e:
        print(f"Error during full-context LLM summarization: {e}")
        return {
            'topic': topic,
            'summary': f"Error: {e}",
            'cost': 0.0,
            'input_tokens': 0,
            'full_text_tokens': len(full_text_tokens)
        }

def assess_summary_quality(
    summary: str,
    retrieved_chunks: List[Dict[str, Any]],
    topic: str,
    llm_client: OpenAI,
    model: str = 'gpt-4o'
) -> Dict[str, Any]:
    """
    Two-dimensional quality assessment (Coverage and Faithfulness) using LLM-as-judge.

    Args:
        summary (str): The generated summary text.
        retrieved_chunks (List[Dict[str, Any]]): The chunks used to generate the summary.
        topic (str): The original topic query.
        llm_client (OpenAI): The initialized OpenAI client.
        model (str): The OpenAI LLM model to use for judging.

    Returns:
        Dict[str, Any]: A dictionary containing coverage and faithfulness scores.
    """
    JUDGE_SYSTEM_PROMPT = "You are an expert financial analyst critically evaluating summaries based on provided source texts. Your responses must be in JSON format."

    # --- Coverage Assessment ---
    coverage_prompt_user = f"""Given this topic: "{topic}"

And this summary:
{summary}

Rate the summary's coverage on a 1-5 scale:
Breadth: Does the summary cover multiple aspects of the topic? (1=one aspect, 5=comprehensive)
Depth: Does the summary provide specific details, numbers, and trends? (1=vague, 5=specific)
Completeness: Are there obvious gaps in addressing the topic? (1=many gaps, 5=thorough)

Return JSON: {{{{"breadth": X, "depth": X, "completeness": X, "missing_topics": ["..."]}}}}"""

    coverage = {}
    try:
        response_coverage = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": coverage_prompt_user}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=300
        )
        coverage = json.loads(response_coverage.choices[0].message.content)
    except Exception as e:
        print(f"Error during coverage assessment: {e}")
        coverage = {"breadth": "N/A", "depth": "N/A", "completeness": "N/A", "missing_topics": [f"Error: {e}"]}

    # --- Faithfulness Assessment ---
    # Prepare excerpts for faithfulness check (only first few chunks to save tokens)
    source_excerpts = "\n".join([textwrap.dedent(c['text'])[:500] + "..." for c in retrieved_chunks[:5]])

    faithfulness = {}
    if source_excerpts:
        faithfulness_prompt_user = f"""Here are the source excerpts:
{source_excerpts}

Here is a summary based on these excerpts:
{summary}

Identify any claims in the summary that are NOT supported by the provided excerpts.
Return JSON: {{{{"unsupported_claims": ["..."], "total_claims": N, "supported_claims": N}}}}"""

        try:
            response_faithfulness = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": faithfulness_prompt_user}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=500
            )
            faithfulness = json.loads(response_faithfulness.choices[0].message.content)
        except Exception as e:
            print(f"Error during faithfulness assessment: {e}")
            faithfulness = {"unsupported_claims": [f"Error: {e}"], "total_claims": 1, "supported_claims": 0}
    else:
        faithfulness = {"unsupported_claims": ["No source chunks provided for faithfulness assessment."], "total_claims": 0, "supported_claims": 0}


    print(f"SUMMARY QUALITY ASSESSMENT: {topic[:50]}...")
    print(f" Coverage: Breadth={coverage.get('breadth', 'N/A')}/5, Depth={coverage.get('depth', 'N/A')}/5, Completeness={coverage.get('completeness', 'N/A')}/5")
    if coverage.get('missing_topics') and any(coverage['missing_topics']):
        print(f" Missing: {', '.join(topic for topic in coverage['missing_topics'] if topic)}")

    supported_claims = faithfulness.get('supported_claims', 0)
    total_claims = faithfulness.get('total_claims', 1)
    faithfulness_score = supported_claims / max(1, total_claims)
    print(f" Faithfulness: {supported_claims}/{total_claims} claims supported ({faithfulness_score:.0%})")
    if faithfulness.get('unsupported_claims') and any(faithfulness['unsupported_claims']):
        print(f" Unsupported Claims: {'; '.join(claim for claim in faithfulness['unsupported_claims'] if claim)}")
    print("-" * 50)

    return {'coverage': coverage, 'faithfulness': faithfulness}


class TenKAnalyzer:
    """
    A class to encapsulate the workflow for analyzing 10-K filings using RAG and LLM summarization.
    """
    def __init__(self, openai_api_key: str, embedding_model: str = 'text-embedding-ada-002', llm_model: str = 'gpt-4o', collection_name: str = "tenk_sections_default"):
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided.")
        
        os.environ["OPENAI_API_KEY"] = openai_api_key # Set environment variable for OpenAIEmbeddings
        self.openai_api_key = openai_api_key
        self.client_llm = OpenAI(api_key=openai_api_key)
        self.embedder_func = MyEmbeddingFunction(model_name=embedding_model, openai_api_key=openai_api_key)
        self.default_llm_model = llm_model
        
        self.chroma_client = Client() # In-memory client by default, can be configured for persistence
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None
        
        self.sections: Dict[str, str] = {}
        self.full_text: str = ""
        self.full_text_token_count: int = 0

    def load_and_process_document(self, filepath: str, create_dummy_if_missing: bool = False, dummy_content: Optional[str] = None):
        """
        Loads and parses a 10-K PDF document, and then chunks and embeds its sections.
        """
        if create_dummy_if_missing and not os.path.exists(filepath):
            if not dummy_content:
                raise ValueError("Dummy content must be provided if create_dummy_if_missing is True and file is missing.")
            _create_dummy_pdf_if_not_exists(filepath, dummy_content)
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found at {filepath}")

        self.sections = parse_10k_sections(filepath)
        doc = fitz.open(filepath)
        self.full_text = "".join([page.get_text() for page in doc])
        doc.close()
        
        enc = tiktoken.encoding_for_model(self.default_llm_model)
        self.full_text_token_count = len(enc.encode(self.full_text))

        self.collection = chunk_and_embed_sections(
            self.sections, self.embedder_func, self.collection_name, self.chroma_client
        )
        print(f"Document processing complete. Collection '{self.collection_name}' ready.")

    def get_rag_summary(
        self,
        topic_query: str,
        section_filter: Optional[List[str]] = None,
        k: int = 8,
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a summary for a given topic using RAG.
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized. Call load_and_process_document first.")
        
        model_to_use = llm_model if llm_model else self.default_llm_model
        return rag_summarize(
            topic_query=topic_query,
            collection=self.collection,
            embedder_func=self.embedder_func,
            llm_client=self.client_llm,
            section_filter=section_filter,
            k=k,
            model=model_to_use,
            model_pricing_dict=MODEL_PRICING
        )

    def get_full_context_summary(
        self,
        topic: str,
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a summary for a given topic by feeding the full document text to the LLM.
        """
        if not self.full_text:
            raise RuntimeError("Document full text not loaded. Call load_and_process_document first.")
        
        model_to_use = llm_model if llm_model else self.default_llm_model
        return full_context_summarize(
            topic=topic,
            full_text=self.full_text,
            llm_client=self.client_llm,
            model=model_to_use,
            model_pricing_dict=MODEL_PRICING
        )

    def generate_research_brief(
        self,
        brief_topics: List[Dict[str, Any]],
        llm_model_for_brief: Optional[str] = None,
        k_chunks: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Generates a multi-aspect research brief based on predefined topics.
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized. Call load_and_process_document first.")

        research_brief_results = []
        model_to_use = llm_model_for_brief if llm_model_for_brief else 'gpt-3.5-turbo' # Use a cheaper model for brief generation
        
        print("\nGenerating Multi-Aspect Research Brief:")
        print("="*60)

        total_cost_brief = 0
        for topic_config in brief_topics:
            print(f"\n--- Generating summary for: {topic_config.get('brief_label', topic_config['topic'])} ---")
            result = self.get_rag_summary(
                topic_query=topic_config['topic'],
                section_filter=topic_config['sections'],
                k=k_chunks,
                llm_model=model_to_use
            )
            result['label'] = topic_config.get('brief_label', topic_config['topic'])
            research_brief_results.append(result)
            total_cost_brief += result['cost']

            print(f"  Input Tokens: {result['input_tokens']:,}, Cost: ${result['cost']:.4f}")
            print(f"  Sections Used: {', '.join(result['sections_covered'])}")
        
        print("\n" + "="*60)
        print("10-K RESEARCH BRIEF GENERATED")
        print(f"Total cost for brief generation: ${total_cost_brief:.4f}")
        print("="*60)
        
        return research_brief_results

    def compare_summarization_methods(
        self,
        topic_to_compare: str,
        sections_for_rag_filter: List[str],
        llm_model_for_comparison: Optional[str] = None,
        k_rag: int = 10
    ) -> Dict[str, Any]:
        """
        Compares RAG-based summarization with full-context summarization for a given topic.
        """
        if not self.collection or not self.full_text:
            raise RuntimeError("Document not fully processed. Call load_and_process_document first.")

        model_to_use = llm_model_for_comparison if llm_model_for_comparison else self.default_llm_model

        print("\n" + "="*50)
        print(f"RAG vs. FULL-CONTEXT COMPARISON for: {topic_to_compare}")
        print("="*50)

        rag_result = self.get_rag_summary(
            topic_query=topic_to_compare,
            section_filter=sections_for_rag_filter,
            k=k_rag,
            llm_model=model_to_use
        )

        full_context_result = self.get_full_context_summary(
            topic=topic_to_compare,
            llm_model=model_to_use
        )

        # Calculate comparison metrics
        rag_summary_words = len(rag_result['summary'].split())
        full_summary_words = len(full_context_result['summary'].split())
        
        relevant_section_text = ' '.join(self.sections.get(s, '') for s in sections_for_rag_filter)
        n_relevant_section_words = len(relevant_section_text.split())

        comparison_data = {
            'topic': topic_to_compare,
            'rag_result': rag_result,
            'full_context_result': full_context_result,
            'metrics': {
                'rag_input_tokens': rag_result['input_tokens'],
                'full_context_input_tokens': full_context_result['input_tokens'],
                'rag_cost': rag_result['cost'],
                'full_context_cost': full_context_result['cost'],
                'rag_summary_words': rag_summary_words,
                'full_context_summary_words': full_summary_words,
                'total_10k_tokens': self.full_text_token_count,
                'relevant_section_words': n_relevant_section_words
            }
        }
        return comparison_data

    def assess_quality_of_brief_sections(self, brief_results: List[Dict[str, Any]], llm_model_for_judging: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Assesses the quality (coverage and faithfulness) of each section in a research brief.
        """
        quality_results = []
        model_to_use = llm_model_for_judging if llm_model_for_judging else 'gpt-3.5-turbo' # Use a cheaper model for judging

        print("\nAssessing Quality of Each Research Brief Section:")
        print("=" * 60)

        for section_summary in brief_results:
            print(f"\nEvaluating: {section_summary['label']}")
            assessment = assess_summary_quality(
                summary=section_summary['summary'],
                retrieved_chunks=section_summary.get('retrieved_chunks', []), # Pass chunks if available
                topic=section_summary['topic'],
                llm_client=self.client_llm,
                model=model_to_use
            )
            quality_results.append({
                'label': section_summary['label'],
                'topic': section_summary['topic'],
                'assessment': assessment
            })

        print("\n" + "=" * 60)
        print("Summary Quality Assessment Complete")
        print("=" * 60)
        return quality_results

# --- Display Helper Functions ---
def display_comparison_results(comparison_data: Dict[str, Any]):
    """Displays the formatted comparison between RAG and full-context summaries."""
    metrics = comparison_data['metrics']
    rag_result = comparison_data['rag_result']
    full_context_result = comparison_data['full_context_result']
    topic_to_compare = comparison_data['topic']

    print(f"\n{'Metric':<25s} {'RAG':>12s} {'Full-Context':>12s}")
    print("-" * 50)
    print(f"{'Input tokens':<25s} {metrics['rag_input_tokens']:>12,} {metrics['full_context_input_tokens']:>12,}")
    print(f"{'Cost':<25s} ${metrics['rag_cost']:>11.4f} ${metrics['full_context_cost']:>11.4f}")

    cost_ratio = metrics['full_context_cost'] / max(0.0001, metrics['rag_cost'])
    print(f"{'Cost ratio':<25s} {'1.0x':>12s} {cost_ratio:>11.1f}x")

    print(f"{'Summary words':<25s} {metrics['rag_summary_words']:>12,} {metrics['full_context_summary_words']:>12,}")
    print(f"{'Has citations':<25s} {'Yes':>12s} {'No':>12s}") # RAG includes citations, full-context typically doesn't
    print("-" * 50)

    if metrics['relevant_section_words'] > 0:
        cr_rag = 1 - (metrics['rag_summary_words'] / metrics['relevant_section_words'])
        print(f"{'RAG Compression Ratio':<25s} {'':>12s} {cr_rag:>11.2f}")
    else:
        print(f"{'RAG Compression Ratio':<25s} {'':>12s} {'N/A'}")

    print("\n--- RAG Summary ---")
    print(textwrap.fill(rag_result['summary'], width=100))
    print("\n--- Full-Context Summary ---")
    print(textwrap.fill(full_context_result['summary'], width=100))

def display_research_brief(brief_results: List[Dict[str, Any]], company_name: str = "APPLE INC.", filing_year: str = "FY2024"):
    """Displays the formatted research brief."""
    total_cost_brief = sum(r['cost'] for r in brief_results)
    
    print("\n" + "="*60)
    print(f"10-K RESEARCH BRIEF: {company_name} ({filing_year})")
    print(f"Generated via RAG | Total cost for brief: ${total_cost_brief:.4f}")
    print("="*60)

    for section in brief_results:
        print(f"\n--- {section['label']} ---")
        print(f"(Sources: {', '.join(section['sections_covered'])})")
        print(textwrap.fill(section['summary'], width=100))
        print()


# --- Main execution block for demonstration ---
if __name__ == "__main__":
    # Ensure you set your OpenAI API key either as an environment variable
    # or pass it directly to the TenKAnalyzer constructor.
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Uncomment and replace

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY or OPENAI_API_KEY == "OPENAI_API_KEY":
        print("WARNING: OpenAI API key is not set or is a placeholder.")
        print("Please set the OPENAI_API_KEY environment variable or replace 'OPENAI_API_KEY' in the code.")
        print("Exiting demonstration.")
        exit()

    pdf_filename = 'AAPL_10K_2024.pdf'
    dummy_10k_content = """
Item 1. Business
This is a dummy business section for Apple Inc. It covers general company overview, products, and services. The company operates globally. Innovation is a key strategy. It reported $383 billion in net sales last year.

Item 1A. Risk Factors
This is a dummy risk factors section. Key risks include intense competition, reliance on third-party suppliers, macroeconomic conditions, and regulatory changes. Supply chain disruptions can significantly impact operations. Legal and regulatory risks are always present. Geopolitical events also pose a threat. The company faces currency fluctuation risks, as well as risks related to data privacy and security. Global economic downturns could reduce consumer spending on products.

Item 7. Management's Discussion and Analysis
This is a dummy MD&A section. It discusses financial condition and results of operations. Net sales increased by 2% year-over-year. Gross margin was 42%. Operating income was $110 billion. Liquidity remains strong with $150 billion in cash and marketable securities. Cash flows from operations were $120 billion. Capital expenditures were $11 billion. Share repurchases amounted to $90 billion. Research and development expenses were $25 billion, up 10% from the prior year.

Item 7A. Quantitative and Qualitative Disclosures About Market Risk
This section describes dummy market risks, including interest rate risk on debt and foreign currency exchange rate risk. A 10% adverse movement in exchange rates could reduce revenue by $5 billion. Sensitivity analysis shows potential impacts on financial instruments.

Item 8. Financial Statements and Supplementary Data
This section contains dummy financial statements. Accounting policy changes for revenue recognition were adopted in 2023. These changes resulted in a $500 million adjustment to retained earnings. Significant estimates include valuation of inventory and deferred tax assets. Details on liquidity and debt are presented in footnotes.

Item 9. Changes in and Disagreements With Accountants on Accounting and Financial Disclosure
No significant changes or disagreements with accountants.
"""

    analyzer = TenKAnalyzer(openai_api_key=OPENAI_API_KEY, collection_name="aapl_10k_sections_demo", llm_model='gpt-3.5-turbo')

    try:
        analyzer.load_and_process_document(
            filepath=pdf_filename,
            create_dummy_if_missing=True,
            dummy_content=dummy_10k_content
        )
    except Exception as e:
        print(f"Failed to load and process document: {e}")
        print("Using simplified string parsing for demonstration if PDF processing failed.")
        # Fallback to manual string parsing if PDF creation/reading fails
        temp_sections_dict = {}
        current_section = None
        for line in dummy_10k_content.strip().split('\n'):
            match = re.match(r'(Item \d+[\.\s]*[A-Z]?).*', line, re.IGNORECASE)
            if match:
                current_section = match.group(1).strip()
                temp_sections_dict[current_section] = []
            if current_section:
                temp_sections_dict[current_section].append(line)
        analyzer.sections = {k: "\n".join(v) for k, v in temp_sections_dict.items()}
        analyzer.full_text = dummy_10k_content
        print("Parsed 10-K sections (from string directly):")
        for name, text in analyzer.sections.items():
            print(f"  {name}: {len(text.split()):,} words")
        
        # If full text obtained via dummy_10k_content, then chunk and embed manually
        if analyzer.sections:
            analyzer.collection = chunk_and_embed_sections(
                analyzer.sections, analyzer.embedder_func, analyzer.collection_name, analyzer.chroma_client
            )
        else:
            print("Could not parse sections from dummy content. Exiting.")
            exit()
        
        enc = tiktoken.encoding_for_model(analyzer.default_llm_model)
        analyzer.full_text_token_count = len(enc.encode(analyzer.full_text))


    # 1. Generate Research Brief
    BRIEF_TOPICS = [
        {
            'topic': 'Key risk factors including competitive risks, regulatory risks, supply chain risks, and macroeconomic risks',
            'sections': ['Item 1A'],
            'brief_label': 'RISK FACTORS'
        },
        {
            'topic': 'Liquidity position: cash, debt maturity, credit facilities, cash flow adequacy, capital expenditure plans',
            'sections': ['Item 7', 'Item 7A'],
            'brief_label': 'LIQUIDITY & CAPITAL'
        },
        {
            'topic': 'Business segment performance: revenue by segment, growth rates, margin trends, geographic breakdown',
            'sections': ['Item 7'],
            'brief_label': 'SEGMENT PERFORMANCE'
        },
        {
            'topic': 'Competitive position and business strategy: market share, new products, R&D investment, strategic initiatives',
            'sections': ['Item 1', 'Item 7'],
            'brief_label': 'STRATEGY & COMPETITION'
        },
        {
            'topic': 'Accounting policy changes: new standards adopted, revenue recognition changes, significant estimates',
            'sections': ['Item 8'],
            'brief_label': 'ACCOUNTING CHANGES'
        }
    ]

    research_brief_results = analyzer.generate_research_brief(
        brief_topics=BRIEF_TOPICS,
        llm_model_for_brief='gpt-3.5-turbo', # Use a cheaper model for multiple calls
        k_chunks=8
    )
    display_research_brief(research_brief_results, company_name="Apple Inc.", filing_year="FY2024")

    # 2. Compare RAG vs. Full-Context Summarization
    topic_to_compare = "Key risk factors including competitive risks, regulatory risks, supply chain risks, and macroeconomic risks"
    sections_for_rag_filter = ['Item 1A']

    comparison_results = analyzer.compare_summarization_methods(
        topic_to_compare=topic_to_compare,
        sections_for_rag_filter=sections_for_rag_filter,
        llm_model_for_comparison='gpt-3.5-turbo',
        k_rag=10
    )
    display_comparison_results(comparison_results)

    # 3. Assess Quality of Brief Sections
    quality_assessments = analyzer.assess_quality_of_brief_sections(
        brief_results=research_brief_results,
        llm_model_for_judging='gpt-3.5-turbo'
    )
    
    # Example of accessing a specific assessment:
    # print("\nFirst brief section quality:")
    # print(json.dumps(quality_assessments[0], indent=2))
