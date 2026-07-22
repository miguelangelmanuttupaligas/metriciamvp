from sqlalchemy import text
from sqlalchemy.orm import Session

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.ai import get_embedding_model


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


class PostgresDatasetRetriever(BaseRetriever):
    def __init__(self, db: Session, dataset_id: str, similarity_top_k: int = 8) -> None:
        super().__init__()
        self.db = db
        self.dataset_id = dataset_id
        self.similarity_top_k = similarity_top_k
        self.embed_model = get_embedding_model()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_embedding = self.embed_model.get_query_embedding(query_bundle.query_str)
        rows = self.db.execute(
            text(
                """
                SELECT content, metadata_json, 1 - (embedding <=> CAST(:embedding AS vector)) AS score
                FROM dataset_embeddings
                WHERE dataset_id = :dataset_id
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :top_k
                """
            ),
            {
                "dataset_id": self.dataset_id,
                "embedding": vector_literal(query_embedding),
                "top_k": self.similarity_top_k,
            },
        ).mappings()
        nodes = []
        for row in rows:
            node = TextNode(text=row["content"], metadata=row["metadata_json"] or {})
            nodes.append(NodeWithScore(node=node, score=float(row["score"] or 0)))
        return nodes
