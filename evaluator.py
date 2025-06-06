import os

from tqdm import tqdm

from chain.db.db_manager import DBManager
from chain.db.qdrant.qdrant_db import QdrantCollectionsMngr, ChunkPoint
from chain.tests.performance.qdrant.file_manager import FileManager, Result
from chain.tests.performance.qdrant.strategies.strategy_base import StrategyBase
from chain.db.qdrant.qdrant_client_extended import QdrantClientExtended

from db.migration import *  # noqa
import pandas as pd
import ast


class Evaluator:
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

    async def evaluate(self, strategy: StrategyBase, skip_data_init: bool = False):
        df_qdrant_questions = self.file_manager.load_qdrant_questions()

        # Init QDrant client
        collection_name = "test_collection"
        qdrant_db = QdrantCollectionsMngr(collection_name)
        qdrant_client: QdrantClientExtended = qdrant_db.qc

        if not skip_data_init:
            # Load Postgres DB
            try:
                is_postgres_loaded = self.file_manager.load_postgres_db()
            except Exception as e:
                print(f"Error loading postgres db: {e}")
                is_postgres_loaded = False

            if not is_postgres_loaded:
                exit(1)

            print("Postgres is loaded!")

            if await qdrant_client.is_collection_exists(collection_name):
                await qdrant_client.delete_collection(collection_name)

        # df_qdrant_questions = df_qdrant_questions[:10]

        # Extract keywords
        db_manager = DBManager()
        all_uris: list[str] = [
            doc.uri for doc in db_manager.vector_store_client.db.docstore.document_store.get_all_documents()
        ]
        top_k_keywords = 500
        all_keywords = db_manager.vector_store_client.db.docstore.document_keyword_store.get_keywords_for_list(
            all_uris, per_uri_limit=top_k_keywords
        )
        keywords_uri_mapping: dict[str, list[str]] = {}
        for keyword in all_keywords:
            uri = keyword['uri']
            if uri not in keywords_uri_mapping:
                keywords_uri_mapping[uri] = []
            keywords_uri_mapping[uri].append(keyword)

        _ = await strategy.execute(
            df_qdrant_questions, qdrant_client, collection_name, keywords_mapping_per_document=keywords_uri_mapping
        )

        memory_data = qdrant_db.get_qdrant_memory_metrics()
        if memory_data is not None:
            strategy.memory_metrics = memory_data

        # Apply the QDrant client to get relevant documents according to the dataset
        df_generated_questions = self.file_manager.load_generated_questions()  # .sample(frac=1, random_state=42)

        top_k = 8
        results = []
        for _, row in tqdm(df_generated_questions.head(300).iterrows()):
            question = row["question"]
            items: list[ChunkPoint] = await qdrant_db._search_relevant_qdrant_points(question, top_k=top_k)
            point_data = dict(row)
            counter = 0
            for item in items:
                all_points = [item.scored_point] + item.related_points
                for point in all_points:
                    if point.payload is not None:
                        chunk_id = point.payload.get('chunk_id', '')
                        uri = point.payload.get('uri', '')
                        point_data[f"top_chunk_{str(counter)}"] = chunk_id
                        point_data[f"top_uri_{str(counter)}"] = uri
                        counter += 1
            results.append(point_data)

        df_results = pd.DataFrame(results)

        # Save the results
        self.file_manager.save_results(await strategy.get_parameters(), df_results)

    def _calculate_hit_rate(self, df: pd.DataFrame, use_uris: bool = False) -> float:
        total_hit_rate = 0.0
        num_rows = len(df)

        if num_rows == 0:
            return 0.0

        for _, row in df.iterrows():
            ground_truth_values = ast.literal_eval(row['uri_list']) if use_uris else ast.literal_eval(row['chunk_ids'])
            hit = False

            # Check all predicted values (up to top_k=8)
            for i in range(8):  # Using 8 as it's the max k value we use
                predicted_col = f"top_uri_{i}" if use_uris else f"top_chunk_{i}"
                if predicted_col in row.index and pd.notna(row[predicted_col]):
                    predicted_value = str(row[predicted_col])
                    if predicted_value in ground_truth_values:
                        hit = True
                        break

            total_hit_rate += 1.0 if hit else 0.0

        return total_hit_rate / num_rows

    def _calculate_precision_recall_at_k(self, df: pd.DataFrame, k: int, use_uris: bool = False) -> tuple[float, float]:
        """
        Calculate standard precision@k and recall@k metrics
        precision@k = # of relevant items in top k / k
        recall@k = # of relevant items in top k / total # of relevant items
        """
        if len(df) == 0:
            return 0.0, 0.0

        total_precision = 0.0
        total_recall = 0.0

        for _, row in df.iterrows():
            ground_truth_values = ast.literal_eval(row['uri_list']) if use_uris else ast.literal_eval(row['chunk_ids'])
            ground_truth_set = set(ground_truth_values)

            # Get predictions from top k
            predictions = []
            for i in range(k):
                predicted_col = f"top_uri_{i}" if use_uris else f"top_chunk_{i}"
                if predicted_col in row.index and pd.notna(row[predicted_col]):
                    predictions.append(str(row[predicted_col]))

            # Remove duplicates
            predictions = list(set(predictions))

            # Count relevant items in top k predictions
            relevant_in_k = sum(1 for pred in predictions if pred in ground_truth_set)

            # Calculate precision and recall for this row
            precision = relevant_in_k / k if k > 0 else 0.0
            recall = relevant_in_k / len(ground_truth_set) if ground_truth_set else 0.0

            total_precision += precision
            total_recall += recall

        # Average across all rows
        avg_precision = total_precision / len(df)
        avg_recall = total_recall / len(df)

        return avg_precision, avg_recall

    def _calculate_average_predictions(self, df: pd.DataFrame, use_uris: bool = False) -> float:
        """
        Calculate the average number of non-null predictions per row
        """
        if len(df) == 0:
            return 0.0

        total_predictions = 0
        for _, row in df.iterrows():
            # Count non-null predictions
            i = 0
            all_values = []
            while True:
                predicted_col = f"top_uri_{i}" if use_uris else f"top_chunk_{i}"
                if predicted_col not in row.index or pd.isna(row[predicted_col]) or len(row[predicted_col]) == 0:
                    break
                all_values.append(row[predicted_col])
                i += 1
            total_predictions += len(set(all_values))

        return total_predictions / len(df)

    def build_metrics(self):
        results: list[Result] = self.file_manager.load_all_results()

        if not results:
            print("No results found to build metrics.")
            return

        k_values = [1, 3, 5, 8]  # Define K values for which to calculate metrics

        # Create a list to store all metrics data
        metrics_data = []

        for result_idx, result in enumerate(results):
            print(f"--- Result Set {result_idx + 1} ---")
            params_str_list = [f"{str(key)}: {str(value)}" for key, value in result['parameters'].items()]
            print(f"Parameters: {', '.join(params_str_list)}")

            df_data = result['results']
            if df_data.empty:
                print("DataFrame is empty for this result set.")
                continue

            # Create a row for this result set
            row_data = {
                'result_set': result['run_name'],
                **result['parameters'],  # Include all parameters
            }

            # Calculate average number of predictions
            avg_chunk_predictions = self._calculate_average_predictions(df_data, use_uris=False)
            avg_uri_predictions = self._calculate_average_predictions(df_data, use_uris=True)
            row_data['avg_chunk_predictions'] = avg_chunk_predictions
            row_data['avg_uri_predictions'] = avg_uri_predictions
            print("  Average Predictions:")
            print(f"    Average Chunk Predictions: {avg_chunk_predictions:.2f}")
            print(f"    Average URI Predictions:   {avg_uri_predictions:.2f}")

            # Calculate hit rates (independent of k)
            chunk_hit_rate = self._calculate_hit_rate(df_data, use_uris=False)
            uri_hit_rate = self._calculate_hit_rate(df_data, use_uris=True)
            row_data['chunk_hit_rate'] = chunk_hit_rate
            row_data['uri_hit_rate'] = uri_hit_rate
            print("  Hit Rates (independent of k):")
            print(f"    Chunk Hit Rate: {chunk_hit_rate:.4f}")
            print(f"    URI Hit Rate:   {uri_hit_rate:.4f}")

            # Calculate and add metrics for each k for both chunk IDs and URIs
            for k in k_values:
                # Calculate metrics for chunk IDs
                precision, recall = self._calculate_precision_recall_at_k(df_data, k, use_uris=False)
                row_data[f'chunk_precision@{k}'] = precision
                row_data[f'chunk_recall@{k}'] = recall
                print(f"  For k={k} (Chunks):")
                print(f"    Precision@{k}: {precision:.4f}")
                print(f"    Recall@{k}:    {recall:.4f}")

                # Calculate metrics for URIs
                precision, recall = self._calculate_precision_recall_at_k(df_data, k, use_uris=True)
                row_data[f'uri_precision@{k}'] = precision
                row_data[f'uri_recall@{k}'] = recall
                print(f"  For k={k} (URIs):")
                print(f"    Precision@{k}: {precision:.4f}")
                print(f"    Recall@{k}:    {recall:.4f}")

            metrics_data.append(row_data)
            print("\n")

        # Create DataFrame from collected metrics
        metrics_df = pd.DataFrame(metrics_data)

        # Save metrics using FileManager
        self.file_manager.save_metrics(metrics_df)
        print(f"Metrics saved to {os.path.join(self.file_manager.dataset_dir, 'metrics.csv')}")

        print(metrics_df)
