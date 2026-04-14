#!/usr/bin/env python3
"""Test script for the adapted LLM Wiki Agent.

This script:
1. Creates a test document
2. Ingests it into the wiki
3. Asks a question about the document
4. Prints the answer + trajectory + token usage

Usage:
    python scripts/test_wiki_agent.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm_client import LLMClient
from src.llm_wiki.ingest import WikiIngestor
from src.llm_wiki.query import WikiQuerier
from src.llm_wiki.tracking import TrajectoryLogger
from src.data.models import Document, DocumentPage


def create_test_document() -> Document:
    """Create a simple test document."""
    # Create a minimal document with metadata
    # Note: In production, you'd have actual image paths
    doc = Document(
        doc_id="test-doc-001",
        domain="healthcare",
        pages=[
            DocumentPage(
                image_path="images/healthcare/test-doc-001/test_page_0001.png",
                page_number=1
            )
        ]
    )
    return doc


def create_test_source_content() -> str:
    """Create test source content for ingestion.
    
    This simulates what OCR would extract from the document images.
    """
    return """# Healthcare Policy Document

## Overview
This document outlines the healthcare policy framework for patient data management.

## Key Policies

### Patient Privacy
All patient data must be handled in compliance with HIPAA regulations. 
Patient consent is required before any data sharing.

### Data Retention
Medical records must be retained for a minimum of 7 years after the last patient encounter.
Electronic records should be backed up daily.

### Access Control
Only authorized healthcare providers may access patient records.
Access logs must be maintained for audit purposes.

## Implementation Guidelines

1. All staff must complete privacy training annually
2. Data breaches must be reported within 24 hours
3. Regular audits will be conducted quarterly

## Contact Information
For questions about this policy, contact the Privacy Office at privacy@hospital.org
"""


def main():
    """Run end-to-end test of the wiki agent."""
    print("=" * 60)
    print("LLM Wiki Agent - End-to-End Test")
    print("=" * 60)
    
    # Initialize components
    print("\n[1] Initializing components...")
    client = LLMClient()
    trajectory_logger = TrajectoryLogger()
    ingestor = WikiIngestor(client=client, trajectory_logger=trajectory_logger)
    querier = WikiQuerier(client=client, trajectory_logger=trajectory_logger)
    
    # Create test document
    print("\n[2] Creating test document...")
    document = create_test_document()
    source_content = create_test_source_content()
    print(f"  Document ID: {document.doc_id}")
    print(f"  Domain: {document.domain}")
    print(f"  Pages: {document.page_count}")
    
    # Ingest document
    print("\n[3] Ingesting document into wiki...")
    try:
        result, metadata = ingestor.ingest_document(
            document=document,
            source_content=source_content,
            question_id="test-ingest-001"
        )
        print(f"  Result: {result}")
        print(f"  Metadata:")
        for key, value in metadata.items():
            print(f"    - {key}: {value}")
    except Exception as e:
        print(f"  ERROR during ingestion: {e}")
        print("\nNote: This test requires a working LLM API configuration.")
        print("Make sure your .env file has:")
        print("  OPENAI_BASE_URL=http://az.gptplus5.com/v1")
        print("  OPENAI_API_KEY=<your-key>")
        print("  OPENAI_MODEL=gemini-3-flash-preview")
        return False
    
    # Ask a question
    print("\n[4] Asking a question about the ingested document...")
    question = "What are the data retention requirements mentioned in the healthcare policy?"
    print(f"  Question: {question}")
    
    try:
        answer, query_metadata = querier.query(
            question_text=question,
            question_id="test-query-001"
        )
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
        
        print("\n[5] Query Metadata:")
        for key, value in query_metadata.items():
            print(f"    - {key}: {value}")
        
        # Get final trajectory stats
        messages, metrics = trajectory_logger.end_query()
        
        print("\n[6] Token Usage Summary:")
        print(f"    - Total tokens: {metrics.total_tokens}")
        print(f"    - Prompt tokens: {metrics.prompt_tokens}")
        print(f"    - Completion tokens: {metrics.completion_tokens}")
        print(f"    - Total latency: {metrics.latency_ms:.2f}ms")
        print(f"    - Retrieval count: {metrics.retrieval_count}")
        print(f"    - LLM calls: {metrics.llm_calls}")
        
        print("\n[7] Trajectory logged to:", trajectory_logger.log_dir)
        
        return True
        
    except Exception as e:
        print(f"  ERROR during query: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
