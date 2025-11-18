from src.scalable_sys.llm.llama_server import LlamaServer
from .retriever import KuzuRetriever
from prompts import TEXT2CYPHER_PROMPT, ANSWER_PROMPT



class GraphRAG:
    def __init__(self, llm: LlamaServer, db_path: str):
        self.llm = llm
        self.retriever = KuzuRetriever(db_path)
        self.schema = self.retriever.get_schema()

    def query(self, user_question: str):
        # 1. Generate Cypher
        prompt = TEXT2CYPHER_PROMPT.format(schema=self.schema, question=user_question)
        # Note: You might need to clean the output if the LLM is chatty
        cypher_query = self.llm.generate(prompt, max_tokens=200)

        # Simple cleanup (remove markdown code blocks if present)
        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()

        print(f"Generated Cypher: {cypher_query}")

        # 2. Execute Cypher
        results_df = self.retriever.execute_query(cypher_query)

        # Convert dataframe to string context
        context_str = results_df.to_string(index=False) if not isinstance(results_df, str) else results_df

        # 3. Generate Answer
        final_prompt = ANSWER_PROMPT.format(context=context_str, question=user_question)
        answer = self.llm.generate(final_prompt)

        return answer

