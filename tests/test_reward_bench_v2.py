from concurrent.futures import ThreadPoolExecutor
from typing import List

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.utils.file import read_jsonl, write_json, write_raw_content

THREAD_NUM = 128


TASKS = {
    "Focus": "Need to detect high-quality, on-topic answers to general user queries.",
    "Factuality": "Factuality: need to  detect hallucinations and other basic errors in completions.",
    "Safety": "Need to correctly comply with or refuse prompts related to harmful use cases as well as general compliance behaviors.",
    "Math": "Need to solve  math problem, on open-ended human prompts ranging from middle school physics and geometry to college-level chemistry, calculus, combinatorics, and more.",
    "Precise IF": "Need to judge whether text follows precise instructions.",
}
TASK_RM_MAPRING = {
    "Focus": "focus_listwise_reward",
    "Factuality": "factuality_listwise_reward",
    "Safety": "safety_listwise_reward",
    "Math": "math_listwise_reward",
    "Precise IF": "precise_if_listwise_reward",
}


def calc_acc(samples: List[DataSample]):
    labels = []
    bad_samples = []
    for sample in samples:
        labels.append(0)
        for output in sample.output:
            if output.answer.label["preference"] == "chosen":
                score = sum(r.score for r in output.answer.reward.details)
                if score > 0:
                    labels[-1] = 1
                else:
                    sample.metadata["is_bad_sample"] = "true"
    return sum(labels) / len(labels)


def calc_bad_ratio(samples: List[DataSample]):
    labels = []
    for sample in samples:
        labels.append(0)
        score_sample = 0
        for output in sample.output:
            score_sample += sum(r.score for r in output.answer.reward.details)
        if score_sample <= 0.00001:
            labels[-1] = 1
            sample.metadata["no_reward"] = "true"
    return sum(labels) / len(labels)


def test_base(
    file: str = "",
    llm: BaseLLM = None,
):
    samples = read_jsonl(file)

    samples = [DataSample(**sample) for sample in samples]

    rm = RewardRegistry.get("base_helpfulness_listwise")(
        name="test_base_demo",
        llm=llm,
        desc="""
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
""",
        scenario="",
        principles=[],
    )
    samples = rm.evaluate_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=THREAD_NUM)
    )
    acc = calc_acc(samples)
    bad_res_ratio = calc_bad_ratio(samples)
    logger.info(f"{file}: acc: {acc},\t bad_res_ratio: {bad_res_ratio}")
    return acc, bad_res_ratio, samples


def test_exp(file: str = "", task: str = "", llm: BaseLLM = None):
    samples = read_jsonl(file)

    samples = [DataSample(**sample) for sample in samples]
    rm_name = TASK_RM_MAPRING.get(task)
    assert rm_name is not None
    rm = RewardRegistry.get(rm_name)(llm=llm, name=f"test_exp_{task}")
    samples = rm.evaluate_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=THREAD_NUM)
    )
    acc = calc_acc(samples)
    bad_res_ratio = calc_bad_ratio(samples)
    logger.info(f"{file}: acc: {acc},\t bad_res_ratio: {bad_res_ratio}")
    return acc, bad_res_ratio, samples


def test_single(task: str, repeat: int = 5, llm: BaseLLM = None):
    scenario = TASKS[task]
    logger.info(f"------------{task}--------------")
    logger.info(f"Scenario: {scenario}")

    path = f"data/rewardbench2/data/{task}.jsonl"
    train, test = path.replace(task, f"{task} Train"), path.replace(
        task, f"{task} Test"
    )
    task_result = {}
    task_result["task_name"] = task
    task_result["repeat"] = repeat

    task_result["base_acc"] = []
    task_result["base_bad_ratio"] = []
    task_result["exp_acc"] = []
    task_result["exp_bad_ratio"] = []

    for index in range(repeat):
        while True:
            try:
                acc, bad_ratio, base_samples = test_base(test, llm)
                task_result["base_acc"].append(acc)
                task_result["base_bad_ratio"].append(bad_ratio)
                write_raw_content(
                    f"data/rewardbench2/result/{task}/base_{index}.jsonl",
                    [sample.model_dump_json() for sample in base_samples],
                    auto_create_dir=True,
                )
                break
            except Exception as e:
                logger.error(e)

        while True:
            try:
                acc, bad_ratio, exp_samples = test_exp(test, task=task, llm=llm)
                task_result["exp_acc"].append(acc)
                task_result["exp_bad_ratio"].append(bad_ratio)
                write_raw_content(
                    f"data/rewardbench2/result/{task}/exp_{index}.jsonl",
                    [sample.model_dump_json() for sample in exp_samples],
                    auto_create_dir=True,
                )
                break

            except Exception as e:
                logger.error(e)
        logger.info(f"{task} loop {index},task_result={task_result}")

    import numpy as np

    base_accs = np.array(task_result["base_acc"])
    task_result["base_acc_avg"] = np.mean(base_accs)
    task_result["base_acc_std"] = np.std(base_accs)
    exp_accs = np.array(task_result["exp_acc"])
    task_result["exp_acc_avg"] = np.mean(exp_accs)
    task_result["exp_acc_std"] = np.std(exp_accs)
    return task_result


def test_all(tasks, repeat: int = 5, llm_params: dict = {}):
    results = {}
    llm_eval = OpenaiLLM(**llm_params)  # qwen3-235b-a22b,qwen3-8b
    for task, _ in tasks.items():
        results[task] = test_single(task, repeat=repeat, llm=llm_eval)
        write_json(results[task], f"data/rewardbench2/result/{task}/overview.json")
    logger.info(f"Results: {results}")
    write_json(results, "data/rewardbench2/result/overview.json")


test_all(TASKS, 5, {"model": "qwen3-32b", "enable_thinking": True})
