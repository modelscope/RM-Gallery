from concurrent.futures import ThreadPoolExecutor
from typing import List

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.utils.file import read_jsonl, write_json
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward
from rm_gallery.gallery.rm.alignment.helpfulness import TASKS
from tests.test_principle_generator import calc_acc, generate

SCENARIO = "Entails adopting specific characters or personas within text-based scenarios, engaging in dialogues or actions that reflect the assigned roles."


def test_generate(
    file: str = "data/RMBbench/pairwise/Helpfulness/Role Playing/Specific Character.jsonl",
    scenario: str = SCENARIO,
    generate_number: int = 10,
    cluster_number: int = 5,
):
    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]
    principles = generate(samples, scenario, generate_number, cluster_number)
    principles = [f"{k}: {v}" for k, v in principles.items()]
    logger.info(f"------------principles--------------\n{principles}")
    return principles


def test_evaluate(
    file: str = "data/RMBbench/pairwise/Helpfulness/Role Playing/Specific Character.jsonl",
    principles: List[str] = [
        "Intent Understanding: Understand user intentions and response."
    ],
    scenario: str = SCENARIO,
):
    llm = OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=True)  # qwen3-235b-a22b

    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]

    reward_module = BaseHelpfulnessListwiseReward(
        llm=llm,
        name="rmb_helpfulness_listwise",
        principles=principles,
        scenario=scenario,
    )

    samples = reward_module.evaluate_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=256)
    )
    acc = calc_acc(samples)
    logger.info(f"{file}: {acc}")
    return acc


def parse(path: str):
    path.replace("RMBbench", "RMBbench_Train")


def test_single(task: str):
    scenario = TASKS[task]
    logger.info(f"------------{task}--------------")
    logger.info(f"Scenario: {scenario}")

    path = f"data/RMBbench/bestofn/Helpfulness/{task}.jsonl"
    train, test = path.replace("RMBbench", "RMBbench_Train"), path.replace(
        "RMBbench", "RMBbench_Test"
    )

    principles = []
    while True:
        try:
            principles = test_generate(
                train,
                scenario=scenario,
            )
            break
        except Exception as e:
            logger.error(e)

    while True:
        try:
            acc = test_evaluate(
                test,
                principles=principles,
                scenario=scenario,
            )
            break
        except Exception as e:
            logger.error(e)

    return acc, principles


def test_all(tasks):
    results = {}
    for task, _ in tasks.items():
        results[task] = test_single(task)

    logger.info(f"Results: {results}")
    write_json(results, "data/RMBbench/results_10_5.json")


test_all(tasks=TASKS)
