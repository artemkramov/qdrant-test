from chain.tests.performance.qdrant.evaluator import Evaluator
from chain.tests.performance.qdrant.file_manager import FileManager

if __name__ == "__main__":
    dataset_name = "danby"
    file_manager = FileManager(dataset_name)
    evaluator = Evaluator(file_manager)
    evaluator.build_metrics()
