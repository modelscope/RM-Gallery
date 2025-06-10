from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.utils.file import read_jsonl
from rm_gallery.gallery.rm.classification.classification import (
    ClassificationListWiseReward,
)
from tests.test_principle_generator import calc_acc


def test_file(
    file: str = "data/rmb/pair/rmbbenchmark_pairwise_classification_content_categorization_output.jsonl",
):
    llm = OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=True)

    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]

    reward_module = ClassificationListWiseReward(
        llm=llm,
        name="classification_listwise",
        # principles=["Judge according to your own standard."],
    )

    samples = reward_module.evaluate_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=128)
    )
    acc = calc_acc(samples)
    logger.info(f"{file}: {acc}")


test_file(
    "/mnt3/huangsen.huang/codes/RM-Gallery/data/rmb/bon/rmbbenchmark_bestofn_classification_quality_and_compliance_assessment_output.jsonl"
)
