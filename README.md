# PDF-Based RAG Application Using Ollama

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) application that answers questions using information extracted from a PDF document.

The application uses a locally running Large Language Model (LLM) through Ollama. It retrieves relevant text chunks from the PDF and generates answers based on the retrieved context.

The application is developed using Python without LangChain, LangGraph, or external vector databases.

## 2. Technologies Used

* Python
* PyMuPDF for PDF text extraction
* Ollama for local embeddings and LLM inference
* Nomic Embed Text for generating embeddings
* NumPy for cosine similarity calculations
* JSON for storing chunks, embeddings, and metadata
* uv for Python dependency management

## 3. Project Structure

```text
RAG_APP/
├── main.py
├── requirements.txt
├── README.md
├── sample.pdf
└── vector_store.json
```

* `main.py`: Contains the complete RAG application.
* `requirements.txt`: Lists the required Python packages.
* `sample.pdf`: Sample document used for testing.
* `vector_store.json`: Stores text chunks, embeddings, page numbers, and index metadata.
* `README.md`: Project documentation.

The JSON index is generated automatically when the application processes the PDF for the first time.

## 4. Prerequisites

Install Python and Ollama before running the application.

Download the required Ollama models:

```powershell
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
```

The embedding model converts text into numerical vectors. The chat model generates answers from the retrieved context.

## 5. Installation

Open the project directory in the terminal:

```powershell
cd D:\RAG_APP
```

Initialize the project with uv if it has not already been initialized:

```powershell
uv init --bare
```

Create a virtual environment if required:

```powershell
uv venv --python 3.12
```

Install the dependencies:

```powershell
uv add pymupdf ollama numpy
```

Alternatively, install packages from the requirements file:

```powershell
uv pip install -r requirements.txt
```

## 6. Execution

Run the application:

```powershell
uv run main.py
```

On the first run, the application:

1. Extracts text from the PDF.
2. Divides the text into chunks.
3. Generates embeddings using Ollama.
4. Saves the chunks and embeddings in `vector_store.json`.
5. Starts the interactive question-answering session.

On subsequent runs, the application loads the existing JSON index when the PDF and indexing settings match.

## 7. RAG Pipeline

### Step 1: PDF Ingestion

PyMuPDF extracts text from each page of the PDF. The original page number is preserved for source references.

### Step 2: Text Chunking

The extracted text is divided into smaller chunks.

The application uses configurable chunk parameters:

* Chunk size: 500 characters
* Chunk overlap: 100 characters

Overlap preserves some context between adjacent chunks.

### Step 3: Embedding Generation

Ollama uses the `nomic-embed-text` model to convert each text chunk into a numerical embedding vector.

The user's question is converted into an embedding using the same model.

### Step 4: JSON Vector Store

The application stores the following information in `vector_store.json`:

* Chunk ID
* Extracted text
* Source page number
* Embedding vector
* PDF hash
* Embedding model name
* Chunk size and overlap settings

This allows the application to reuse existing embeddings instead of regenerating them on every run.

### Step 5: Cosine Similarity

The application calculates cosine similarity between the question embedding and each stored chunk embedding.

The formula is:

$$
\operatorname{cosine}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
$$

Where:

* A is the question embedding.
* B is a document chunk embedding.
* The dot product measures the alignment between the vectors.
* The vector magnitudes are used for normalization.

Higher similarity scores indicate greater directional similarity between the vectors.

NumPy is used to calculate the dot product and vector norms.

### Step 6: Retrieval

The application sorts the chunks by similarity score and selects the top-k relevant chunks.

Only chunks meeting the configured similarity threshold are passed to the answer-generation stage.

### Step 7: Answer Generation

The retrieved text is supplied to the local Ollama chat model.

The prompt instructs the model to answer using only the provided PDF excerpts and to indicate when the information is unavailable.

Source page numbers are included in the context to support citations.

### Step 8: Interactive Question Answering

Users can ask multiple questions through the terminal. The application retrieves relevant chunks and generates an answer for each question.

Type `exit` to close the application.

## 8. Sample Questions

The following questions can be used to test the sample document:

1. What is Machine Learning?
2. What are the types of Machine Learning?
3. What is Unsupervised Learning?
4. Who invented the telephone?

The first three questions test information available in the sample PDF. The fourth question tests how the application handles information absent from the document.

## 9. Testing

The application was tested for the following requirements:

| Test                                    | Expected Behaviour                        |
| --------------------------------------- | ----------------------------------------- |
| Question answered explicitly in the PDF | Generate an answer from retrieved context |
| Question involving multiple concepts    | Retrieve relevant chunks                  |
| Question about a specific topic         | Ret                                       |
