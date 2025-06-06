import asyncio
from chain.tests.performance.qdrant.file_manager import FileManager
from chain.tests.performance.qdrant.strategies.strategy_base import QuantizationType
from chain.tests.performance.qdrant.evaluator import Evaluator
import os

from chain.tests.performance.qdrant.strategies.strategy_main_questions_keywords_cut import (
    StrategyMainQuestionsKeywordsCut,
)


async def main():
    # Check if the test marker is set
    # We make it to make sure that we use the test data
    is_test = os.getenv("IS_TEST")
    if is_test is None:
        raise Exception("IS_TEST env variable has to be set to run this evaluation")

    dataset_name = "danby"
    file_manager = FileManager(dataset_name)
    evaluator = Evaluator(file_manager)
    # await evaluator.evaluate(StrategyDefault())
    await evaluator.evaluate(
        StrategyMainQuestionsKeywordsCut(num_questions=10, quantization=QuantizationType.INT8), skip_data_init=True
    )


if __name__ == "__main__":
    asyncio.run(main())
