import os
import subprocess
from typing import TypedDict

import pandas as pd
import psycopg2
from sqlalchemy import make_url
import yaml
from datetime import datetime

from const.defaults import DB_URL


class Result(TypedDict):
    parameters: dict
    run_name: str
    results: pd.DataFrame


class FileManager:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

        current_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(current_dir, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        self.dataset_dir = os.path.join(datasets_dir, dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

    def _save_df_to_csv(self, df: pd.DataFrame, filename: str):
        df.to_csv(os.path.join(self.dataset_dir, filename), index=False)

    def save_generated_questions(self, df_questions: pd.DataFrame):
        self._save_df_to_csv(df_questions, "questions_generated.csv")

    def save_qdrant_questions(self, df_questions: pd.DataFrame):
        self._save_df_to_csv(df_questions, "questions_qdrant.csv")

    def save_results(self, parameters: dict, df_results: pd.DataFrame):
        # Create a directory for the results
        # Align it with the current datetime
        results_dir = os.path.join(self.dataset_dir, "results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(results_dir, exist_ok=True)

        # Save the parameters as a yaml file
        with open(os.path.join(results_dir, "parameters.yaml"), "w") as f:
            yaml.dump(parameters, f)

        # Save the results
        self._save_df_to_csv(df_results, os.path.join(results_dir, "results.csv"))

    def save_metrics(self, metrics_df: pd.DataFrame):
        """Save metrics DataFrame to CSV file.

        Args:
            metrics_df: DataFrame containing metrics data with columns for parameters,
                       precision@k and recall@k values
        """
        self._save_df_to_csv(metrics_df, os.path.join(self.dataset_dir, "metrics.csv"))

    def load_generated_questions(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dataset_dir, "questions_generated.csv"))

    def load_qdrant_questions(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dataset_dir, "questions_qdrant.csv"))

    def load_all_results(self) -> list[Result]:
        results_dir = os.path.join(self.dataset_dir, "results")

        results = []

        # Iterate over all the files in the results directory
        # Load parameters from the yaml file
        # Load the results from the csv file
        # Set a name of the folder as a run name
        for result_dir in os.listdir(results_dir):
            if os.path.isdir(os.path.join(results_dir, result_dir)):
                with open(os.path.join(results_dir, result_dir, "parameters.yaml")) as f:
                    parameters = yaml.safe_load(f)
                results.append(
                    Result(
                        parameters=parameters,
                        run_name=result_dir,
                        results=pd.read_csv(os.path.join(results_dir, result_dir, "results.csv")),
                    )
                )
        return results

    def load_postgres_db(self):
        BACKUP_FILE = os.path.join(self.dataset_dir, "backup.sql")
        # Parse DB_URL using SQLAlchemy
        url = make_url(DB_URL)
        target_db = str(url.database)
        user = str(url.username)
        password = url.password
        host = url.host or "localhost"
        port = url.port or 5432

        # Step 1: Connect to 'postgres' to drop and create target DB
        conn = psycopg2.connect(dbname="postgres", user=user, password=password, host=host, port=port)
        conn.autocommit = True
        cur = conn.cursor()

        print(f"Dropping and recreating database: {target_db}")
        cur.execute(f"DROP DATABASE IF EXISTS {target_db};")

        # Verify database was dropped
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
        if cur.fetchone() is not None:
            raise Exception(f"Database {target_db} still exists after DROP operation")

        cur.execute(f"CREATE DATABASE {target_db};")

        cur.close()
        conn.close()

        # Step 2: Restore from backup.sql using psql
        print(f"Restoring from {BACKUP_FILE}...")

        env = os.environ.copy()
        env["PGPASSWORD"] = str(password)

        subprocess.run(
            ["psql", "-U", user, "-h", host, "-p", str(port), "-d", target_db, "-f", BACKUP_FILE], check=True, env=env
        )

        # Step 3: Verify database was created and has expected schemas
        conn = psycopg2.connect(dbname=target_db, user=user, password=password, host=host, port=port)
        cur = conn.cursor()

        # Check if database exists and has schemas
        cur.execute("""
            SELECT COUNT(DISTINCT schema_name) 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
        """)

        count = cur.fetchone()
        schema_count = 0
        if count is not None:
            schema_count = count[0]

        cur.close()
        conn.close()

        if schema_count != 3:
            raise Exception(f"Database verification failed: expected 3 schemas, found {schema_count}")

        print("✅ Restore completed successfully.")
        return True
