# Multimodal Vehicle Damage Assessment and Insurance Claim Generator

A Python-based prototype for vehicle damage assessment and insurance claim report generation. The project is designed to integrate vehicle image analysis, repair cost estimation, insurance policy checks, and claim report generation.

## Project Overview

The system aims to help streamline the vehicle insurance claim process by combining damage assessment with structured claim reporting.

**Current implementation:** Streamlit UI, Contract A sample report validation, mock orchestration, JSON/PDF report generation, and a local Role 3 insurance-policy RAG retrieval pipeline.

**Important:** The current prototype uses saved sample data. Uploaded vehicle photos are not analysed by an AI model, repair cost estimation is not connected, and the Role 3 policy retriever is not yet integrated into claim response generation or final claim decisions.

## Features

- Vehicle information form
- Upload 1–5 vehicle photos
- Contract A sample report viewer
- JSON schema validation
- Mock pipeline orchestration
- Structured damage report display
- JSON report download
- PDF assessment report generation
- Automated tests using pytest


## Role 3 — Insurance Policy RAG

The initial Role 3 implementation provides hybrid retrieval over vehicle insurance policy documents.

### Implemented components

- PDF extraction using pypdf and pdfplumber.
- Heading-aware and clause-aware chunking.
- Policy metadata including insurer, UIN, page number and source URL.
- BGE-small-en-v1.5 embeddings and ChromaDB persistent indexing.
- BM25 keyword retrieval and vector search.
- Hybrid retrieval using Reciprocal Rank Fusion (RRF).
- Policy-scope classification and scope-aware reranking.
- Insurer filtering and source-page citation metadata.

### Supported insurers

- TATA AIG
- ACKO
- HDFC ERGO
- SBI General

Policy PDFs, generated chunks and vector databases are stored locally and excluded from Git.

Generate chunks with `python -m src.rag.chunker` and build the retrieval index with `python -m src.rag.retriever --rebuild`.

The document loader, chunker and retriever are implemented. Response generation, the claim writer and Contract B integration remain pending.

## Technology Stack

- Python 3.9+ (tested with Python 3.9.6)
- Streamlit
- JSON Schema
- ReportLab
- Pytest
- pypdf and pdfplumber
- sentence-transformers (BGE-small-en-v1.5)
- ChromaDB
- rank-bm25
- tiktoken

## Project Structure

```text
vehicle-damage-assessment/
├── docs/
│   └── role1/
├── mocks/
├── schemas/
│   ├── contract_a.schema.json
│   └── contract_c.schema.json
├── src/
│   ├── app/
│   │   └── main.py
│   ├── rag/
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   └── retriever.py
│   └── pipeline/
│       ├── contract_a_validator.py
│       ├── mock_pipeline.py
│       └── orchestrator.py
├── tests/
│   ├── test_contract_a_validator.py
│   ├── test_orchestrator.py
│   └── rag/
│       └── test_policy_scope.py
├── requirements.txt
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/gaganaraj/vehicle-damage-assessment.git
cd vehicle-damage-assessment
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run src/app/main.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## Running Tests

Run the test suite with:

```bash
python -m pytest -v
```

Latest local test result: **17 passed** (11 RAG tests and 6 existing project tests).

## Current Pipeline Status

| Component | Status |
|---|---|
| Streamlit user interface | Implemented |
| Contract A sample validation | Implemented |
| Mock orchestration | Implemented |
| JSON report download | Implemented |
| PDF report generation | Implemented |
| Real vehicle photo analysis | Not connected |
| Repair cost estimation | Not connected |
| Policy document extraction and hybrid retrieval | Implemented locally; integration pending |
| RAG response generation and claim writer | Pending |
| Final claim decision | Not connected |

## Disclaimer

This repository contains a project prototype. Sample reports are used for testing the current workflow. The generated reports are not live AI assessments, repair quotations, insurance coverage determinations, or claim approvals.

## Repository

GitHub: https://github.com/gaganaraj/vehicle-damage-assessment