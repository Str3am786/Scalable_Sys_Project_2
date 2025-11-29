# Enhancing LLM Inference with GraphRAG

This project implements a Graph Retrieval-Augmented Generation (GraphRAG) system using **KuzuDB** and **Llama 3.1**. It features a self-refining Text2Cypher engine that translates natural language into graph queries, validates them against the schema, and fixes errors automatically.

## Architecture
* **App:** Dockerized Python container (Logic, RAG pipeline, KuzuDB driver).
* **LLM:** Native **Ollama** instance on the host (provides hardware/GPU acceleration).

## Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Ollama](https://ollama.com/)
* Python 3.12+

## Quick Start

### 1. Setup the LLM 
Specify your LLM configuration in the .env file. When running the eval pipeline, the judge LLM needs to be specified as well. We recommend using openrouter for quick and easy setup.

### 2. Generate the Database (Local)
Build the `nobel.kuzu` graph database before running the container.

# Run the build script
python src/scalable_sys/rag/build_db.py
```

### 3. Run the Project (Docker)
Build and run the application container. It connects to the LLMs configured via the .env vile. You can launch specific services depending on your needs:

**Main Application**
Run the standard RAG pipeline with the default prompt configured in your compose file:
```bash
docker compose up --build app

Note: The default prompt is configured in docker-compose.yml. To ask a different question, edit the command line in that file or run:

```bash
# Run a custom query
docker compose run app python -m src.scalable_sys.app --prompt "List all female Physics laureates."
```

**Evaluation Pipeline**

Run the full test suite to benchmark the answer accuracies against our curated test set:

```bash
docker compose up --build eval
```

The results of the evaluation will be summarized in results/summary

**LLM Judge**

Run the judge service to compare two specific output files (Plain LLM vs GraphRAG). You must pass the file paths as environment variables:

```bash
PLAIN_FILE="results/plain_output.json" RAG_FILE="results/rag_output.json" docker compose up judge
```

**Cache Tests**

There are two different cache tests you can run:

1. Cold & warm start cache test:

- Evaluates latency for a "cold" start, i.e. when none of the questions in the test set have been cached before. After the "cold" run, the system latency is evaluated again on the same test set, when all questions have been cached. This is the "warm" start evaluation.
- The results for both latency measurements are summarized in the results/cache_test folder.
- Run this test by setting the cache test number flag to "3" in the docker-compose file before running docker-compose up --build cachetest.

2. Cache evaluation on a test set of 60% unique questions and 40% repeated questions.

- The pipeline evaluates the latency with caching on, and then with caching off on the test set with a 60/40 split of unique and repeated questions.
- Set the number flag for the cachetest command to "2" in the docker-compose file to run this version of the cache evaluation.

Run the a cache test to compare the rag pipeline firstly __with__ and secondly __without__ LRU cache. 

The sample questions generated for this evaluation are stored in   __data/test_cache/__
Results, cache log and performance reports will be available in __results/cache_test/__ folder.


```bash
docker compose up --build cachetest
```

Cache is tunable by setting two parameters in the `config.yaml` file:

- `cache_maxsize`: Cache storage dimension.;

- `ache_ttl_seconds:`: Keys Time to Live

### Project Structure

- `src/scalable_sys/rag/prompts.py`: Few-shot exemplars and prompt logic.

- `src/scalable_sys/rag/graph_rag.py`: Core RAG pipeline (Text2Cypher -> Refine -> Execute).

- `data/nobel.kuzu`: The persisted Graph Database.
