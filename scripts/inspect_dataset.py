#!/usr/bin/env python3
"""
Inspect the UniDoc-Bench dataset from HuggingFace.

This script:
- Loads the dataset using the datasets library
- Prints the schema (column names, types)
- Prints the number of rows and splits
- Prints 2-3 sample rows
- Identifies which fields contain: documents/context, questions, ground truth answers
- Identifies if there is a "healthcare" subset or how to filter for one
- Saves the full schema analysis to docs/dataset_analysis.md
"""

import os
import sys
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "dataset_analysis.md"

def main():
    print("=" * 80)
    print("UniDoc-Bench Dataset Inspection")
    print("=" * 80)
    print()
    
    # Try to load the dataset
    try:
        from datasets import load_dataset
        print("Loading dataset from HuggingFace: Salesforce/UniDoc-Bench...")
        
        # Load the dataset
        dataset = load_dataset("Salesforce/UniDoc-Bench")
        
        print("\n" + "=" * 80)
        print("DATASET STRUCTURE")
        print("=" * 80)
        
        # Print available splits/configs
        print(f"\nDataset type: {type(dataset)}")
        
        if hasattr(dataset, 'keys'):
            # DatasetDict
            splits = list(dataset.keys())
            print(f"\nAvailable splits: {splits}")
            
            for split_name in splits:
                split_data = dataset[split_name]
                print(f"\n--- Split: '{split_name}' ---")
                print(f"  Number of rows: {len(split_data)}")
                print(f"  Column names: {split_data.column_names}")
                print(f"  Features: {split_data.features}")
                
                # Print feature types in detail
                print("\n  Feature details:")
                for col_name, feature in split_data.features.items():
                    print(f"    - {col_name}: {type(feature).__name__} = {feature}")
                
                # Print 2-3 sample rows
                print(f"\n  Sample rows (first 2):")
                for i in range(min(2, len(split_data))):
                    print(f"\n    [Row {i}]")
                    row = split_data[i]
                    for key, value in row.items():
                        # Truncate long values
                        str_value = str(value)
                        if len(str_value) > 500:
                            str_value = str_value[:500] + "... [truncated]"
                        print(f"      {key}: {str_value}")
                        
        elif hasattr(dataset, 'column_names'):
            # Single dataset (no splits)
            print(f"\nNumber of rows: {len(dataset)}")
            print(f"Column names: {dataset.column_names}")
            print(f"Features: {dataset.features}")
            
            print("\nFeature details:")
            for col_name, feature in dataset.features.items():
                print(f"  - {col_name}: {type(feature).__name__} = {feature}")
            
            print("\nSample rows (first 2):")
            for i in range(min(2, len(dataset))):
                print(f"\n  [Row {i}]")
                row = dataset[i]
                for key, value in row.items():
                    str_value = str(value)
                    if len(str_value) > 500:
                        str_value = str_value[:500] + "... [truncated]"
                    print(f"    {key}: {str_value}")
        
        # Analyze field purposes
        print("\n" + "=" * 80)
        print("FIELD PURPOSE ANALYSIS")
        print("=" * 80)
        
        # Get column names from first available split
        if hasattr(dataset, 'keys'):
            first_split = list(dataset.keys())[0]
            columns = dataset[first_split].column_names
            sample_row = dataset[first_split][0]
        else:
            columns = dataset.column_names
            sample_row = dataset[0]
        
        document_fields = []
        question_fields = []
        answer_fields = []
        metadata_fields = []
        
        for col in columns:
            sample_value = sample_row[col]
            sample_str = str(sample_value)[:200]
            
            # Heuristic classification
            col_lower = col.lower()
            
            if any(kw in col_lower for kw in ['document', 'doc', 'context', 'text', 'content', 'passage']):
                document_fields.append(col)
                print(f"\n📄 DOCUMENT FIELD: {col}")
                print(f"   Sample: {sample_str}...")
            elif any(kw in col_lower for kw in ['question', 'query', 'prompt', 'input', 'instruction']):
                question_fields.append(col)
                print(f"\n❓ QUESTION FIELD: {col}")
                print(f"   Sample: {sample_str}...")
            elif any(kw in col_lower for kw in ['answer', 'response', 'output', 'target', 'ground_truth', 'reference']):
                answer_fields.append(col)
                print(f"\n✅ ANSWER FIELD: {col}")
                print(f"   Sample: {sample_str}...")
            else:
                metadata_fields.append(col)
                print(f"\n📋 METADATA FIELD: {col}")
                print(f"   Sample: {sample_str}...")
        
        # Check for healthcare subset
        print("\n" + "=" * 80)
        print("HEALTHCARE SUBSET ANALYSIS")
        print("=" * 80)
        
        # Look for category/domain fields
        category_fields = [col for col in columns if any(kw in col.lower() for kw in 
                          ['category', 'domain', 'topic', 'subject', 'field', 'area', 'type'])]
        
        if category_fields:
            print(f"\nPotential category fields: {category_fields}")
            
            for cat_field in category_fields:
                # Get unique values
                if hasattr(dataset, 'keys'):
                    unique_values = dataset[first_split].unique(cat_field)
                else:
                    unique_values = dataset.unique(cat_field)
                
                print(f"\nUnique values in '{cat_field}':")
                if len(unique_values) <= 20:
                    for val in unique_values:
                        print(f"  - {val}")
                    
                    # Check for healthcare-related categories
                    healthcare_keywords = ['health', 'medical', 'healthcare', 'clinical', 'medicine', 'bio']
                    healthcare_values = [v for v in unique_values if any(kw in str(v).lower() for kw in healthcare_keywords)]
                    
                    if healthcare_values:
                        print(f"\n  🏥 HEALTHCARE-RELATED VALUES FOUND: {healthcare_values}")
                        print(f"     Filter: dataset.filter(lambda x: x['{cat_field}'] in {healthcare_values})")
                else:
                    print(f"  ({len(unique_values)} unique values - too many to list)")
                    print(f"  First 20: {unique_values[:20]}")
        else:
            print("\nNo explicit category/domain field found.")
            
            # Check if any text field contains healthcare keywords
            print("\nSearching for healthcare-related content in text fields...")
            healthcare_keywords = ['health', 'medical', 'patient', 'clinical', 'disease', 'treatment', 
                                   'diagnosis', 'symptom', 'hospital', 'doctor', 'medicine']
            
            for col in document_fields + question_fields:
                if hasattr(dataset, 'keys'):
                    sample_texts = [dataset[first_split][i][col] for i in range(min(10, len(dataset[first_split])))]
                else:
                    sample_texts = [dataset[i][col] for i in range(min(10, len(dataset)))]
                
                for i, text in enumerate(sample_texts):
                    text_lower = str(text).lower()
                    matches = [kw for kw in healthcare_keywords if kw in text_lower]
                    if matches:
                        print(f"  Found healthcare keywords in row {i}, column '{col}': {matches}")
                        break
        
        # Write analysis to markdown file
        print("\n" + "=" * 80)
        print("SAVING ANALYSIS TO docs/dataset_analysis.md")
        print("=" * 80)
        
        write_analysis_report(
            dataset=dataset,
            columns=columns,
            sample_row=sample_row,
            document_fields=document_fields,
            question_fields=question_fields,
            answer_fields=answer_fields,
            metadata_fields=metadata_fields,
            output_file=OUTPUT_FILE
        )
        
        print(f"\n✅ Analysis saved to: {OUTPUT_FILE}")
        
    except ImportError as e:
        print(f"ERROR: Required library not installed: {e}")
        print("\nTo install the datasets library, run:")
        print("  pip install datasets")
        sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nPossible issues:")
        print("  - Dataset name may be incorrect")
        print("  - Network connection required to download from HuggingFace")
        print("  - Authentication may be required for gated datasets")
        sys.exit(1)


def write_analysis_report(dataset, columns, sample_row, document_fields, question_fields, 
                          answer_fields, metadata_fields, output_file):
    """Write comprehensive analysis to markdown file."""
    
    # Get split info
    if hasattr(dataset, 'keys'):
        splits = list(dataset.keys())
        first_split = splits[0]
        num_rows = len(dataset[first_split])
    else:
        splits = ["default"]
        num_rows = len(dataset)
        first_split = "default"
    
    report = f"""# UniDoc-Bench Dataset Analysis

## Overview

**Dataset:** Salesforce/UniDoc-Bench  
**Source:** https://huggingface.co/datasets/Salesforce/UniDoc-Bench  
**GitHub:** https://github.com/SalesforceAIResearch/UniDoc-Bench

---

## Dataset Structure

### Splits
{f"- {splits}" if isinstance(splits, list) else splits}

### Row Count
- **Total rows:** {num_rows} (in first split '{first_split}')

### Columns
{chr(10).join(f'- `{col}`' for col in columns)}

---

## Schema Details

### Column Types

"""
    
    # Add feature types
    if hasattr(dataset, 'keys'):
        features = dataset[first_split].features
    else:
        features = dataset.features
    
    for col_name, feature in features.items():
        report += f"- **{col_name}**: `{type(feature).__name__}`\n"
        report += f"  - Full spec: {feature}\n\n"
    
    report += """
---

## Field Purpose Classification

Based on column names and sample content analysis:

### 📄 Document/Context Fields
These fields contain the source documents or context for retrieval:

"""
    
    for field in document_fields:
        sample = str(sample_row[field])[:300]
        sample = sample.replace('\n', ' ')
        report += f"- **`{field}`**: `{sample}...`\n"
    
    if not document_fields:
        report += "- *No obvious document/context fields identified*\n"
    
    report += """

### ❓ Question Fields
These fields contain queries or questions:

"""
    
    for field in question_fields:
        sample = str(sample_row[field])[:300]
        sample = sample.replace('\n', ' ')
        report += f"- **`{field}`**: `{sample}...`\n"
    
    if not question_fields:
        report += "- *No obvious question fields identified*\n"
    
    report += """

### ✅ Answer/Ground Truth Fields
These fields contain expected answers or reference responses:

"""
    
    for field in answer_fields:
        sample = str(sample_row[field])[:300]
        sample = sample.replace('\n', ' ')
        report += f"- **`{field}`**: `{sample}...`\n"
    
    if not answer_fields:
        report += "- *No obvious answer fields identified*\n"
    
    report += """

### 📋 Metadata Fields
Additional information about each sample:

"""
    
    for field in metadata_fields:
        sample = str(sample_row[field])[:200]
        report += f"- **`{field}`**: `{sample}`\n"
    
    report += f"""

---

## Healthcare Subset Analysis

"""
    
    # Check for category fields
    category_fields = [col for col in columns if any(kw in col.lower() for kw in 
                       ['category', 'domain', 'topic', 'subject', 'field', 'area', 'type'])]
    
    if category_fields:
        report += "### Category/Domain Fields Found\n\n"
        for cat_field in category_fields:
            report += f"- **`{cat_field}`**: Potential domain/category indicator\n"
        
        report += "\n### Filtering Strategy\n\n"
        report += "To filter for healthcare-related samples:\n\n"
        report += "```python\n"
        report += "# Option 1: If explicit healthcare category exists\n"
        report += "healthcare_data = dataset.filter(\n"
        report += "    lambda x: 'health' in str(x.get('category', '')).lower() or\n"
        report += "              'medical' in str(x.get('category', '')).lower()\n"
        report += ")\n\n"
        report += "# Option 2: Keyword-based filtering on document content\n"
        if document_fields:
            doc_field = document_fields[0]
            report += f"healthcare_keywords = ['health', 'medical', 'patient', 'clinical', 'disease', 'treatment']\n"
            report += f"healthcare_data = dataset.filter(\n"
            report += f"    lambda x: any(kw in str(x.get('{doc_field}', '')).lower() for kw in healthcare_keywords)\n"
            report += f")\n"
        report += "```\n"
    else:
        report += "No explicit category/domain field was found in the dataset.\n\n"
        report += "### Recommended Filtering Strategy\n\n"
        report += "Since there's no explicit domain field, healthcare samples can be filtered by:\n\n"
        report += "1. **Keyword matching** on document/question fields\n"
        report += "2. **Semantic search** using embeddings to find healthcare-related content\n\n"
        report += "```python\n"
        report += "healthcare_keywords = [\n"
        report += "    'health', 'medical', 'patient', 'clinical', 'disease', \n"
        report += "    'treatment', 'diagnosis', 'symptom', 'hospital', 'doctor',\n"
        report += "    'medicine', 'pharmaceutical', 'therapy', 'surgery'\n"
        report += "]\n\n"
        if document_fields:
            doc_field = document_fields[0]
            report += f"# Filter by document content\n"
            report += f"healthcare_data = dataset.filter(\n"
            report += f"    lambda x: any(kw in str(x.get('{doc_field}', '')).lower() for kw in healthcare_keywords)\n"
            report += f")\n"
        report += "```\n"
    
    report += f"""

---

## Recommendations for RAG Implementation

### Document Format
Based on the analysis:
- Documents appear to be stored in: {document_fields if document_fields else '*TBD - needs manual inspection*'}
- Typical document length: *Needs further analysis of full dataset*
- Structure: *Needs further analysis*

### Chunking Strategy
*To be determined based on actual document format and length*

### Suggested Next Steps
1. Manually inspect several more samples to understand document structure
2. Analyze document length distribution
3. Identify any special formatting (markdown, HTML, plain text)
4. Determine if documents have inherent section boundaries

---

## Sample Data

"""
    
    # Include 2 full samples
    if hasattr(dataset, 'keys'):
        for i in range(min(2, len(dataset[first_split]))):
            report += f"### Sample {i+1}\n\n"
            row = dataset[first_split][i]
            for key, value in row.items():
                str_val = str(value)
                if len(str_val) > 1000:
                    str_val = str_val[:1000] + "... [truncated]"
                report += f"**{key}**:\n```\n{str_val}\n```\n\n"
    
    report += "---\n\n*Generated by scripts/inspect_dataset.py*\n"
    
    # Write to file
    output_file.write_text(report, encoding='utf-8')


if __name__ == "__main__":
    main()
