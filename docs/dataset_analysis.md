# UniDoc-Bench Dataset Analysis

## Overview

**Dataset:** Salesforce/UniDoc-Bench  
**Source:** https://huggingface.co/datasets/Salesforce/UniDoc-Bench  
**GitHub:** https://github.com/SalesforceAIResearch/UniDoc-Bench

---

## Dataset Structure

### Splits
The dataset contains **8 domain-specific splits**:
- `commerce_manufacturing` (200 rows)
- `construction` (200 rows)
- `crm` (200 rows)
- `education` (200 rows)
- `energy` (200 rows)
- `finance` (200 rows)
- `healthcare` (200 rows) ← **Healthcare subset available!**
- `legal` (200 rows)

**Total:** 1,600 rows across all domains

### Columns
All splits share the same schema:
- `question` (string)
- `answer` (string)
- `gt_image_paths` (list of strings)
- `question_type` (string)
- `answer_type` (string)
- `domain` (string)
- `longdoc_image_paths` (list of strings)

---

## Schema Details

### Column Types

| Column | Type | Description |
|--------|------|-------------|
| `question` | `Value('string')` | The query/question to answer |
| `answer` | `Value('string')` | Ground truth answer text |
| `gt_image_paths` | `List(Value('string'))` | Paths to ground truth image(s) containing the answer |
| `question_type` | `Value('string')` | Category of question (factual_retrieval, summarization, causal_reasoning, comparison, temporal_comparison) |
| `answer_type` | `Value('string')` | Required answer format (image_only, table_required, image_plus_text_as_answer, text_only) |
| `domain` | `Value('string')` | Domain identifier matching the split name |
| `longdoc_image_paths` | `List(Value('string'))` | Paths to all pages of the source long document (multi-page images) |

---

## Field Purpose Classification

### 📄 Document/Context Fields
**`longdoc_image_paths`**: Contains paths to multi-page document images (PNG files). Each sample references a long document split into multiple page images.

Example:
```
['images/commerce_manufacturing/0355944/0355944_page_0001.png', 
 'images/commerce_manufacturing/0355944/0355944_page_0002.png', 
 ...]
```

**Key Observation:** This is a **multimodal document QA dataset**. Documents are stored as PNG images of pages, not text. Each document typically has 5-11 pages.

### ❓ Question Fields
- **`question`**: Natural language questions about the document content
- **`question_type`**: Categorizes the reasoning required:
  - `factual_retrieval`: Direct fact lookup
  - `summarization`: Summarizing content
  - `causal_reasoning`: Understanding cause-effect relationships
  - `comparison`: Comparing entities/concepts
  - `temporal_comparison`: Time-based comparisons

### ✅ Answer/Ground Truth Fields
- **`answer`**: Text answer (natural language)
- **`answer_type`**: Specifies what's needed to answer:
  - `image_only`: Answer comes from image content
  - `text_only`: Answer is purely textual
  - `table_required`: Requires reading tables
  - `image_plus_text_as_answer`: Needs both image and text interpretation
- **`gt_image_paths`**: Specific page(s) containing the answer

### 📋 Metadata Fields
- **`domain`**: Domain category for filtering
- Split name corresponds to domain value

---

## Healthcare Subset Analysis

### ✅ EXPLICIT HEALTHCARE SPLIT AVAILABLE

The dataset has a dedicated **`healthcare`** split with 200 samples.

**To load only healthcare data:**
```python
from datasets import load_dataset

# Load only healthcare split
dataset = load_dataset("Salesforce/UniDoc-Bench", split="healthcare")

# Or load all and filter
full_dataset = load_dataset("Salesforce/UniDoc-Bench")
healthcare_data = full_dataset["healthcare"]
```

### Healthcare Sample Example
```
question: "What was the method of Albendazole administration after surgery 
           for hydatid cysts in the cases reported in the June 2021 volume 
           of Ann Coll Med Mosul, and what outcomes were observed?"

answer: "Albendazole was used in both cases after surgical intervention. 
         In the first case, the patient received 3 cycles of Albendazole 
         of 28 days each with a 14-day rest in between..."

longdoc_image_paths: ['images/healthcare/1851094/1851094_page_0001.png', 
                      'images/healthcare/1851094/1851094_page_0002.png',
                      ...]  # 7 pages total
```

---

## Critical Finding: Multimodal Dataset

⚠️ **This is NOT a text-only RAG dataset.** 

**Document Format:**
- Documents are **PNG images** of pages (scanned documents/PDFs)
- Not plain text or markdown
- Requires OCR or multimodal VLM (Vision-Language Model) to process

**Implications for RAG:**
1. **Standard text-based RAG will NOT work directly** — documents are images
2. Need either:
   - **OCR preprocessing** to extract text from images before indexing
   - **Multimodal embeddings** (e.g., CLIP, ColPali) that can embed images directly
   - **VLM-based retrieval** that can reason over document images

---

## Recommendations for RAG Implementation

### Option 1: OCR + Text-Based RAG
1. Run OCR (Tesseract, Azure OCR, AWS Textract) on all PNG images
2. Extract text per page
3. Store text chunks with metadata linking back to original images
4. Use standard dense retrieval (embeddings + vector store)

### Option 2: Multimodal RAG (Recommended)
1. Use document-aware embedding models:
   - **ColPali**: Vision-language model trained on document retrieval
   - **SigLIP/CLIP**: General vision-language embeddings
2. Embed document page images directly
3. Retrieve relevant pages using image embeddings
4. Pass retrieved images to VLM for answer generation

### Chunking Strategy
Given the dataset structure:
- **Natural chunk boundary = 1 page** (each PNG is a page)
- Documents average 5-11 pages each
- Consider grouping consecutive pages when they form coherent sections

### Document Length Analysis
Based on samples:
- Short documents: ~3 pages
- Medium documents: 5-8 pages  
- Long documents: 10+ pages
- Average: ~7 pages per document

---

## Question/Answer Patterns

### Question Types Distribution
From observed samples:
- `factual_retrieval`: Most common — direct fact lookup
- `comparison`: Comparing two or more entities
- `causal_reasoning`: Understanding relationships
- `summarization`: Condensing information
- `temporal_comparison`: Changes over time

### Answer Types
- `image_only`: Answer found in figures/tables/images
- `text_only`: Answer in prose text
- `table_required`: Requires reading structured tables
- `image_plus_text_as_answer`: Needs synthesis of both

---

## Sample Data

### Healthcare Domain Sample
```json
{
  "question": "What was the method of Albendazole administration after surgery for hydatid cysts...",
  "answer": "Albendazole was used in both cases after surgical intervention...",
  "question_type": "comparison",
  "answer_type": "text_only",
  "domain": "healthcare",
  "longdoc_image_paths": ["images/healthcare/1851094/1851094_page_0001.png", ...],
  "gt_image_paths": ["images/healthcare/1851094/1851094_page_0004.png", ...]
}
```

---

## Suggested Next Steps for Benchmark

1. **Decide on modality approach:**
   - Text-only (requires OCR preprocessing)
   - Multimodal (use VLM + image embeddings)

2. **For text-only approach:**
   - Set up OCR pipeline for the image dataset
   - Verify OCR quality on sample documents
   - Create text corpus with page-level granularity

3. **For multimodal approach:**
   - Evaluate ColPali or similar document VLM models
   - Test retrieval quality with image embeddings
   - Ensure OpenAI-compatible endpoint supports vision inputs

4. **Healthcare-specific evaluation:**
   - Use the `healthcare` split for domain-specific benchmarking
   - Compare performance across question types
   - Measure accuracy on medical terminology handling

---

*Generated by scripts/inspect_dataset.py*
