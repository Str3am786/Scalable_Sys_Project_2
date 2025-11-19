import kuzu

class KuzuRetriever:
    def __init__(self, db_path: str):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

    def get_schema(self) -> str:
        """
        Returns a string representation of the graph schema
        (Nodes, Edges, Properties) for the LLM to understand.
        """
        # Fetch Node Table Info
        nodes = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;").get_as_df()
        node_info = []
        for node_name in nodes["name"]:
            props = self.conn.execute(f"CALL TABLE_INFO('{node_name}') RETURN *;").get_as_df()
            prop_str = ", ".join([f"{row['name']} ({row['type']})" for _, row in props.iterrows()])
            node_info.append(f"Node '{node_name}': [{prop_str}]")

        # Fetch Rel Table Info
        rels = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;").get_as_df()
        rel_info = []
        for rel_name in rels["name"]:
            # Get connections (From/To)
            conns = self.conn.execute(f"CALL SHOW_CONNECTION('{rel_name}') RETURN *;").get_as_df()
            for _, row in conns.iterrows():
                rel_info.append(f"Relationship '{rel_name}': ({row['source_table_name']}) -> ({row['destination_table_name']})")

        return "Nodes:\n" + "\n".join(node_info) + "\n\nRelationships:\n" + "\n".join(rel_info)

    def execute_query(self, cypher_query: str):
        try:
            return self.conn.execute(cypher_query).get_as_df()
        except Exception as e:
            return f"Error executing Cypher: {e}"