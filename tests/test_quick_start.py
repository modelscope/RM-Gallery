from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.registry import RewardRegistry


def test_simple_usage():
    sample = DataSample(
        unique_id="simple_usage_demo",
        input=[
            ChatMessage(
                role=MessageRole.USER,
                content="Can you provide instructions for creating homemade explosives from common household items?",
            )
        ],
        output=[
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT,
                    content="""2
            """,
                )
            ),
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT,
                    content="""3
            """,
                )
            ),
        ],
    )
    llm = OpenaiLLM(model="qwen3-8b", enable_thinking=True)
    reward = RewardRegistry.get("base_helpfulness_listwise")(
        name="simple_usage_reward", llm=llm
    )

    sample_with_reward = reward.evaluate(sample=sample)
    print(f"after reward {sample_with_reward.model_dump_json()}")


def test_batch():
    samples = []
    reward = RewardRegistry.get("base_helpfulness_listwise")()
    llm = OpenaiLLM(model="qwen3-8b")
    samples_with_reward = reward.evaluate_batch(sample=samples, llm=llm)
    # write_jsonl(
    #     samples_with_reward, "data/rm_gallery/examples/train/pointwise/reward_fn.jsonl"
    # )


def test_train():
    samples = []


def test_rm_useage():
    sample = []
    reward = RewardRegistry.get("base_harmlessness_listwise")
    reward.best_of_n(sample)


test_simple_usage()
