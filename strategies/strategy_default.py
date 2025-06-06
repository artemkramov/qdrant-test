import uuid

from qdrant_client.http.models import UpdateStatus

from chain.tests.performance.qdrant.strategies.strategy_base import StrategyBase, QDrantItemList, QDrantItem
import pandas as pd
from chain.db.qdrant.qdrant_client_extended import QdrantClientExtended
from qdrant_client.http import models
from tqdm import tqdm


class StrategyDefault(StrategyBase):
    NAME = "Default strategy"

    async def execute(
        self, df: pd.DataFrame, qdrant_client: QdrantClientExtended, collection_name: str, *args, **kwargs
    ) -> QDrantItemList:
        response: QDrantItemList = {'dense_items': [], 'sparse_items': []}

        for _, row in df.iterrows():
            item = QDrantItem(uri=row["uri"], chunk_id=row["chunk_id"], content=row["content"], score=row["score"])

            if row["is_sparse"]:
                response["sparse_items"].append(item)
            else:
                response["dense_items"].append(item)

        shard_number = 12
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_client.get_fastembed_vector_params(on_disk=True),
            sparse_vectors_config=qdrant_client.get_fastembed_sparse_vector_params(on_disk=True),
            shard_number=shard_number,
            hnsw_config=models.HnswConfigDiff(on_disk=True),
            on_disk_payload=True,
            quantization_config=None,
        )

        batch_size = 128
        for i in tqdm(range(0, len(response["dense_items"]), batch_size), desc="Processing dense items"):
            try:
                batch = response["dense_items"][i : i + batch_size]
                batch_vectors = qdrant_client.embed_documents(
                    [item['content'] for item in batch], batch_size=len(batch)
                )
                points = []
                for item, vector in zip(batch, batch_vectors, strict=True):
                    payload = {'chunk_id': item['chunk_id'], 'uri': item['uri'], 'score': item['score']}
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector={qdrant_client.get_vector_field_name(): vector},
                            payload=payload,
                        )
                    )

                qdrant_response = await qdrant_client.upsert(collection_name=collection_name, points=points)
                if qdrant_response.status != UpdateStatus.COMPLETED:
                    raise ValueError("Cannot upload all Qdrant points!")
            except Exception as e:
                print(f"Error {str(e)}")

        for i in tqdm(range(0, len(response["sparse_items"]), batch_size), desc="Processing sparse items"):
            try:
                batch = response["sparse_items"][i : i + batch_size]
                batch_vectors = qdrant_client.embed_sparse_documents(
                    [item['content'] for item in batch], batch_size=len(batch)
                )
                points = []
                for item, vector in zip(batch, batch_vectors, strict=True):
                    payload = {'chunk_id': item['chunk_id'], 'uri': item['uri'], 'score': item['score']}
                    sparse_vector_name = qdrant_client.get_sparse_vector_field_name()
                    if sparse_vector_name is None:
                        raise ValueError("Cannot set an embedding model!")
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector={sparse_vector_name: vector},
                            payload=payload,
                        )
                    )

                qdrant_response = await qdrant_client.upsert(collection_name=collection_name, points=points)
                if qdrant_response.status != UpdateStatus.COMPLETED:
                    raise ValueError("Cannot upload all Qdrant sparse points!")
            except Exception as e:
                print(f"Error {str(e)}")

        return response

    async def get_parameters(self) -> dict:
        parameters = await super().get_parameters()
        return parameters
