# src/scalable_sys/rag/db.py
from __future__ import annotations

from typing import Dict, List

import kuzu


class KuzuDatabaseManager:
    """Manages Kuzu database connection and schema retrieval."""

    def __init__(self, db_path: str = "nobel.kuzu", read_only: bool = True):
        self.db_path = db_path
        self.db = kuzu.Database(db_path, read_only=read_only)
        self.conn = kuzu.Connection(self.db)

    @property
    def schema_dict(self) -> Dict[str, List[dict]]:
        """
        Return a dictionary describing the labelled property graph schema:
        {
          "nodes": [{ "label": ..., "properties": [...] }, ...],
          "edges": [{ "label": ..., "from": ..., "to": ..., "properties": [...] }, ...],
        }
        """
        response = self.conn.execute(
            "CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;"
        )
        nodes = [row[1] for row in response]

        response = self.conn.execute(
            "CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;"
        )
        rel_tables = [row[1] for row in response]

        relationships: list[dict] = []
        for tbl_name in rel_tables:
            response = self.conn.execute(
                f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;"
            )
            for row in response:
                relationships.append(
                    {"name": tbl_name, "from": row[0], "to": row[1]}
                )

        schema: Dict[str, List[dict]] = {"nodes": [], "edges": []}

        for node in nodes:
            node_schema = {"label": node, "properties": []}
            node_properties = self.conn.execute(
                f"CALL TABLE_INFO('{node}') RETURN *;"
            )
            for row in node_properties:
                node_schema["properties"].append(
                    {"name": row[1], "type": row[2]}
                )
            schema["nodes"].append(node_schema)

        for rel in relationships:
            edge = {
                "label": rel["name"],
                "from": rel["from"],
                "to": rel["to"],
                "properties": [],
            }
            rel_properties = self.conn.execute(
                f"CALL TABLE_INFO('{rel['name']}') RETURN *;"
            )
            for row in rel_properties:
                edge["properties"].append(
                    {"name": row[1], "type": row[2]}
                )
            schema["edges"].append(edge)

        return schema
