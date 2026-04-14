# RAG Design for UniDoc-Bench Benchmark

## Document Analysis

### Document Format
Based on dataset inspection, UniDoc-Bench documents have these characteristics:

| Aspect | Finding |
|--------|---------|
| **Storage Format** | PNG images (scanned pages) |
| **Structure** | Multi-page documents (5-11 pages avg) |
| **Content Type** | Mixed: text, tables, figures, charts |
| **Domains** | 8 domains including healthcare, legal, finance |
| **Text Accessibility** | Requires OCR or VLM to extract/read |

### Key Implications
⚠️ **This is a multimodal RAG challenge**, not traditional text-only RAG.

---

## Chunking Strategy

### Recommended Approach: Page-Level Chunking

Given that documents are stored as page images:

```
Document → [Page 1] [Page 2] [Page 3] ... [Page N]
            ↓        ↓        ↓              ↓
         Chunk 1  Chunk 2  Chunk 3       Chunk N
```

**Rationale:**
1. **Natural boundary**: Each PNG file is one page
2. **Preserves layout**: Tables/figures stay intact within pages
3. **Manageable size**: ~7 chunks per document average
4. **Metadata-rich**: Each chunk can retain page number, doc ID

### Alternative: Logical Section Chunking (if OCR used)

If OCR preprocessing is applied:
```python
# Pseudo-code for section-aware chunking
def chunk_document(ocr_text):
    sections = split_by_headers(ocr_text)
    chunks = []
    for section in sections:
        if len(section) > MAX_CHUNK_SIZE:
            chunks.extend(split_by_paragraphs(section))
        else:
            chunks.append(section)
    return chunks
```

**Chunk Size Recommendations:**
- **For page images**: 1 page = 1 chunk
- **For OCR text**: 300-500 tokens per chunk
- **Overlap**: 50-100 tokens between consecutive chunks

---

## Vector Store Selection

### Recommendation: FAISS (Facebook AI Similarity Search)

**Why FAISS:**
1. **Speed**: Optimized for high-dimensional similarity search
2. **Scalability**: Handles millions of vectors efficiently
3. **Flexibility**: Supports multiple index types (IVF, HNSW, etc.)
4. **Integration**: Works seamlessly with LangChain/LlamaIndex
5. **Local**: No external service dependency

### Alternative Options

| Vector Store | Pros | Cons | Best For |
|-------------|------|------|----------|
| **FAISS** | Fast, scalable, local | Memory-bound for huge datasets | Production benchmarks |
| **Chroma** | Simple API, persistent | Slower than FAISS | Development/testing |
| **Qdrant** | Rich filtering, distributed | More complex setup | Large-scale deployments |
| **Pinecone** | Managed service | Cost, vendor lock-in | Cloud-native apps |

### Index Configuration for FAISS

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# For ~10,000 chunks (1600 docs × ~7 pages)
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_base=OPENAI_BASE_URL,
    openai_api_key=OPENAI_API_KEY
)

# Create index
vectorstore = FAISS.from_documents(chunks, embeddings)

# For larger datasets, use IVF index
# index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
```

---

## Embedding Approach

### Given: OpenAI-Compatible Endpoint

The benchmark assumes access to an OpenAI-compatible API endpoint.

### Text Embedding Strategy (if using OCR)

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
```

**Recommended Models:**
- `text-embedding-3-small`: Fast, cost-effective, 1536 dimensions
- `text-embedding-3-large`: Higher quality, 3072 dimensions
- Custom endpoint models: Verify dimensionality compatibility

### Multimodal Embedding Strategy (for image-based retrieval)

Since documents are images, consider these approaches:

#### Option A: ColPali (Document VLM)
```python
# ColPali generates embeddings from document images directly
from colpali_engine.models import ColPali, ColPaliProcessor

model = ColPali.from_pretrained("vidore/colpali-v1.2")
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

def embed_document_image(image_path: str):
    inputs = processor(images=[image_path], return_tensors="pt")
    embeddings = model(**inputs)
    return embeddings.detach().numpy()
```

#### Option B: CLIP/SigLIP for General Images
```python
from PIL import Image
import clip

model, preprocess = clip.load("ViT-B/32")

def embed_image(image_path: str):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.numpy()
```

#### Option C: Hybrid (OCR + Image Embeddings)
- Generate text embeddings from OCR-extracted text
- Generate image embeddings from page PNGs
- Concatenate or fuse embeddings
- Retrieve using combined representation

---

## Retrieval Strategy

### Traditional RAG Pipeline

```
Query → Embed Query → Vector Search → Top-K Chunks → LLM → Answer
```

### Agentic Retrieval (llm-wiki-agent inspired)

```
Query → Agent decides retrieval strategy
          ├─ Keyword search (if index available)
          ├─ Vector search (for semantic match)
          └─ Iterative refinement (if needed)
       → Retrieved chunks → Agent synthesizes answer
```

### Recommended: Hybrid Approach

```python
class HybridRetriever:
    def __init__(self, vector_store, keyword_index=None):
        self.vector_store = vector_store
        self.keyword_index = keyword_index
    
    def retrieve(self, query: str, k: int = 5) -> list:
        # Primary: Dense retrieval
        dense_results = self.vector_store.similarity_search(query, k=k)
        
        # Optional: Sparse/keyword re-ranking
        if self.keyword_index:
            keyword_results = self.keyword_index.search(query, k=k*2)
            # Reciprocal Rank Fusion
            results = rrf_fusion(dense_results, keyword_results)
        else:
            results = dense_results
        
        return results
```

---

## Healthcare-Specific Considerations

### Domain Challenges
1. **Terminology**: Medical jargon, abbreviations, Latin terms
2. **Precision**: High stakes require accurate retrieval
3. **Context**: Symptoms, treatments, dosages need careful handling

### Mitigation Strategies
1. **Domain-specific embeddings**: Fine-tune on medical corpus if possible
2. **Query expansion**: Expand medical abbreviations automatically
3. **Re-ranking**: Use cross-encoder for final ranking of retrieved chunks
4. **Citation tracking**: Ensure answers cite specific source pages

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    UniDoc-Bench Benchmark                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Images     │  │   OCR Text   │  │   Metadata   │      │
│  │  (PNG pages) │  │  (extracted) │  │  (doc/page)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Embedding Layer                            │
│  ┌────────────────────────┐  ┌────────────────────────┐    │
│  │   Text Embeddings      │  │   Image Embeddings     │    │
│  │   (OpenAI-compatible)  │  │   (ColPali/CLIP)       │    │
│  └────────────────────────┘  └────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Vector Store (FAISS)                        │
│  - Index: IVF or HNSW                                        │
│  - Dimension: 1536 (text) or model-specific (images)        │
│  - Metadata: doc_id, page_num, domain, question_type        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Retrieval Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Hybrid Retriever                         │   │
│  │  - Dense retrieval (vector similarity)               │   │
│  │  - Optional: keyword re-ranking                      │   │
│  │  - Domain filtering (healthcare subset)              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Generation Layer (LLM)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Agentic Synthesis                           │   │
│  │  - Receive top-k retrieved chunks                    │   │
│  │  - Synthesize answer with citations                  │   │
│  │  - Handle multi-hop reasoning if needed              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation Layer                           │
│  - Exact match accuracy                                     │
│  - ROUGE/BLEU scores                                        │
│  - Citation correctness                                     │
│  - Domain-specific metrics (healthcare precision)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Notes

### Environment Variables Required
```bash
OPENAI_BASE_URL=https://your-endpoint.com/v1
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini  # or your model
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

### Key Dependencies
```
# Core
litellm>=1.0.0          # Unified LLM client
langchain>=0.1.0        # RAG orchestration
faiss-cpu>=1.7.4        # Vector store

# Dataset
datasets>=2.0.0         # HuggingFace datasets

# Optional: Multimodal
torch>=2.0.0            # For ColPali/CLIP
transformers>=4.30.0    # Model loading
Pillow>=9.0.0           # Image processing

# Optional: OCR
pytesseract>=0.3.0      # Tesseract wrapper
pdf2image>=1.16.0       # PDF to image conversion
```

### Next Steps
1. Decide on modality approach (text-only vs multimodal)
2. Set up data pipeline (download + optional OCR)
3. Implement chunking strategy
4. Configure vector store
5. Build retrieval layer
6. Integrate with LLM for generation
7. Add evaluation metrics

---

*Generated based on analysis of llm-wiki-agent and UniDoc-Bench dataset*
