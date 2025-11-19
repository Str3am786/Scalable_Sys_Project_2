# Enhancing LLM Inference with GraphRAG

This project implements a Graph Retrieval-Augmented Generation (GraphRAG) system using **KuzuDB** and **Llama 3.1**. It features a self-refining Text2Cypher engine that translates natural language into graph queries, validates them against the schema, and fixes errors automatically.

## Architecture
* **App:** Dockerized Python container (Logic, RAG pipeline, KuzuDB driver).
* **LLM:** Native **Ollama** instance on the host (provides hardware/GPU acceleration).

## Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Ollama](https://ollama.com/)
* Python 3.10+ (only for initial data generation)

## Quick Start

### 1. Setup the LLM (Host Machine)
Run the inference engine natively to leverage your local GPU.

1.  **Install Ollama** from [ollama.com](https://ollama.com).
2.  **Start Ollama** (ensure the app is running in the background).
3.  **Download the Model**:
    ```bash
    ollama pull llama3.1:8b
    ```

### 2. Generate the Database (Local)
Build the `nobel.kuzu` graph database before running the container.

```bash
# Create venv and install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the build script
python src/scalable_sys/rag/build_db.py
```

### 3. Run the Project (Docker)
Build and run the application container. It automatically connects to your host's Ollama instance.

```bash
docker-compose up --build
```

Note: The default prompt is configured in docker-compose.yml. To ask a different question, edit the command line in that file or run:

```bash
# Run a custom query
docker-compose run app python -m src.scalable_sys.app --prompt "List all female Physics laureates."
```

### Project Structure

- `src/scalable_sys/rag/prompts.py`: Few-shot exemplars and prompt logic.

- `src/scalable_sys/rag/graph_rag.py`: Core RAG pipeline (Text2Cypher -> Refine -> Execute).

- `data/nobel.kuzu`: The persisted Graph Database.