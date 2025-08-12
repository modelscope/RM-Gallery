"""
JudgeBench Evaluator - supports multiple judge types for pairwise comparison evaluation
"""
import os
import re
import random
from typing import Dict, List, Type, Optional, Union, Any
from abc import ABC, abstractmethod

import fire
import jinja2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.base import BaseLLMReward, BaseListWiseReward
from rm_gallery.core.reward.evaluator import BaseEvaluator
from rm_gallery.core.reward.schema import RewardDimensionWithRank, RewardResult
from rm_gallery.core.reward.template import BasePromptTemplate
from rm_gallery.core.utils.file import write_json


class JudgeBenchTemplateRenderer:
    """JudgeBench template renderer, supports jinja2 templates"""
    
    def __init__(self, template_path: str = "data/benchmarks/JudgeBench/utils/templates"):
        """
        Initialize template renderer
        
        Args:
            template_path: Path to template files
        """
        self.template_path = template_path
        self.template_loader = jinja2.FileSystemLoader(template_path)
        self.template_env = jinja2.Environment(
            loader=self.template_loader, 
            cache_size=-1, 
            trim_blocks=True, 
            lstrip_blocks=True
        )
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render template
        
        Args:
            template_name: Template name (without .jinja2 suffix)
            **kwargs: Template variables
            
        Returns:
            Rendered template content
        """
        template = self.template_env.get_template(template_name + ".jinja2")
        return template.render(**kwargs)


class JudgeBenchBaseTemplate(BasePromptTemplate):
    """JudgeBench base template class"""
    
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)
    
    @classmethod
    def get_renderer(cls) -> JudgeBenchTemplateRenderer:
        """Get template renderer singleton"""
        if not hasattr(cls, '_renderer_instance'):
            cls._renderer_instance = JudgeBenchTemplateRenderer()
        return cls._renderer_instance
    
    @classmethod
    def format(cls, question: str, answer_A: str, answer_B: str, **kwargs) -> str:
        """
        Format prompt template
        
        Args:
            question: Question content
            answer_A: Answer A
            answer_B: Answer B
            **kwargs: Other template variables
            
        Returns:
            Formatted prompt content
        """
        raise NotImplementedError("Subclass must implement format method")


class VanillaTemplate(JudgeBenchBaseTemplate):
    """Vanilla Judge template"""
    
    # Decision field
    decision: str = Field(
        default="",
        description="Judgment result, should be 'Output (a)' or 'Output (b)'"
    )
    
    @classmethod
    def format(cls, question: str, answer_A: str, answer_B: str, **kwargs) -> str:
        """Generate Vanilla evaluation prompt"""
        renderer = cls.get_renderer()
        return renderer.render_template(
            "vanilla_prompt",
            question=question,
            answer_a=answer_A,
            answer_b=answer_B
        )
    
    @classmethod
    def parse(cls, text: str) -> "VanillaTemplate":
        """Parse Vanilla output"""
        text = text.strip()
        decision = ""
        
        if text == "Output (a)":
            decision = "A>B"
        elif text == "Output (b)":
            decision = "B>A"
        else:
            logger.warning(f"Unable to parse Vanilla output: {text}")
            decision = ""
        
        return cls(decision=decision)


class ArenaHardTemplate(JudgeBenchBaseTemplate):
    """Arena-Hard Judge template"""
    
    # Decision field
    decision: str = Field(
        default="",
        description="Judgment result, format like A>B, B>A, A=B etc."
    )
    
    @classmethod
    def format(cls, question: str, answer_A: str, answer_B: str, **kwargs) -> str:
        """Generate Arena-Hard evaluation prompt"""
        renderer = cls.get_renderer()
        
        # Arena-Hard uses system + user message format
        system_message = renderer.render_template("arena_hard_judge_system")
        user_message = renderer.render_template(
            "arena_hard_judge_prompt",
            prompt=question,
            answer_a=answer_A,
            answer_b=answer_B
        )
        
        # Return special format indicating this is a system+user message combination
        return f"SYSTEM_USER_FORMAT:{system_message}|USER:{user_message}"
    
    @classmethod
    def parse(cls, text: str) -> "ArenaHardTemplate":
        """Parse Arena-Hard output"""
        # Use regex to extract decision
        pattern = re.compile(r"\[\[([AB<>=]+)\]\]")
        matches = pattern.findall(text)
        
        decision = ""
        if matches:
            # Take the last matching result
            raw_decision = matches[-1].strip()
            # Normalize decision format
            decision = raw_decision.replace(">>", ">").strip()
        
        return cls(decision=decision)


class AutoJTemplate(JudgeBenchBaseTemplate):
    """AutoJ Judge template"""
    
    decision: str = Field(
        default="",
        description="Judgment result"
    )
    
    @classmethod
    def format(cls, question: str, answer_A: str, answer_B: str, **kwargs) -> str:
        """Generate AutoJ evaluation prompt"""
        renderer = cls.get_renderer()
        return renderer.render_template(
            "autoj_prompt",
            question=question,
            response=answer_A,
            response_another=answer_B
        )
    
    @classmethod
    def parse(cls, text: str) -> "AutoJTemplate":
        """Parse AutoJ output"""
        text = text.strip()
        decision = ""
        
        # Find final decision
        pos = text.rfind('final decision is ')
        if pos != -1:
            pred_rest = text[pos + len('final decision is '):].strip().lower()
            if pred_rest.startswith('response 1'):
                decision = "A>B"
            elif pred_rest.startswith('response 2'):
                decision = "B>A"
            elif pred_rest.startswith('tie'):
                decision = "A=B"
        
        return cls(decision=decision)


class Prometheus2Template(JudgeBenchBaseTemplate):
    """Prometheus2 Judge template"""
    
    decision: str = Field(
        default="",
        description="Judgment result"
    )
    
    @classmethod
    def format(cls, question: str, answer_A: str, answer_B: str, **kwargs) -> str:
        """Generate Prometheus2 evaluation prompt"""
        renderer = cls.get_renderer()
        rubric = kwargs.get("rubric", "[Are the model's responses factually correct and well-supported by evidence?]")
        return renderer.render_template(
            "prometheus2_prompt",
            instruction=question,
            response_A=answer_A,
            response_B=answer_B,
            rubric=rubric
        )
    
    @classmethod
    def parse(cls, text: str) -> "Prometheus2Template":
        """Parse Prometheus2 output"""
        text = text.strip()
        decision = ""
        
        # Use regex to find result
        patterns = [
            r"\[RESULT\]\s*([AB])",
            r"\[RESULT:\s*([AB])\]",
            r"\[Response\s+([AB])\]",
            r"\[Result\]\s*Response\s*([AB])",
            r"\[Result:\s*([AB])\]",
            r"(?:^|\n)Result:?\s*([AB])",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1).upper()
                if result == "A":
                    decision = "A>B"
                elif result == "B":
                    decision = "B>A"
                break
        
        return cls(decision=decision)


class SkyworkCriticTemplate(JudgeBenchBaseTemplate):
    """Skywork Critic Judge template"""
    
    decision: str = Field(
        default="",
        description="Judgment result"
    )
    
    @classmethod
    def format(cls, question: str, answer_A: str, answer_B: str, **kwargs) -> str:
        """Generate Skywork Critic evaluation prompt"""
        renderer = cls.get_renderer()
        return renderer.render_template(
            "skywork_critic_prompt",
            input=question,
            response_a=answer_A,
            response_b=answer_B
        )
    
    @classmethod
    def parse(cls, text: str) -> "SkyworkCriticTemplate":
        """Parse Skywork Critic output"""
        text = text.strip()
        decision = ""
        
        # Find decisions in [[A]] or [[B]] format
        if "[[A]]" in text:
            decision = "A>B"
        elif "[[B]]" in text:
            decision = "B>A"
        
        return cls(decision=decision)


class JudgeBenchReward(BaseLLMReward, BaseListWiseReward):
    """JudgeBench reward class supporting multiple judge types"""
    
    judge_type: str = Field(
        default="arena_hard",
        description="Judge type"
    )
    
    template: Type[JudgeBenchBaseTemplate] = Field(
        default=ArenaHardTemplate,
        description="Template class"
    )
    
    def __init__(self, judge_type: str = "arena_hard", template: Type[JudgeBenchBaseTemplate] = None, **kwargs):
        """
        Initialize reward class
        
        Args:
            judge_type: Judge type
            template: Custom template class, if provided will override judge_type default mapping
            **kwargs: Other parameters
        """
        # Set template class based on judge_type
        template_map = {
            "vanilla": VanillaTemplate,
            "arena_hard": ArenaHardTemplate,
            "auto_j": AutoJTemplate,
            "prometheus_2": Prometheus2Template,
            "skywork_critic": SkyworkCriticTemplate,
            # Can continue adding other judge types
        }
        
        # If user provided custom template, use it; otherwise use judge_type mapping
        if template is not None:
            template_class = template
        else:
            template_class = template_map.get(judge_type, ArenaHardTemplate)
        
        super().__init__(
            judge_type=judge_type,
            template=template_class,
            **kwargs
        )
    
    def _before_evaluate(self, sample: DataSample, **kwargs) -> Dict:
        """Preprocess evaluation parameters"""
        params = super()._before_evaluate(sample=sample, **kwargs)
        
        # Extract question and two answers
        question = sample.input[-1].content
        
        # Ensure there are two answers
        if len(sample.output) < 2:
            raise ValueError("JudgeBench requires at least two answers for comparison")
        
        answer_A = sample.output[0].answer.content
        answer_B = sample.output[1].answer.content
        
        params.update({
            "question": question,
            "answer_A": answer_A,
            "answer_B": answer_B
        })
        
        return params
    
    def _evaluate(self, **kwargs) -> RewardResult:
        """Override evaluation method to handle special Arena-Hard format"""
        assert self.llm is not None
        
        for i in range(self.max_retries):
            try:
                params = self._before_evaluate(**kwargs)
                prompt = self.template.format(
                    enable_thinking=self.llm.enable_thinking, **params
                )
                logger.info(f"prompt: {prompt}")
                
                # Check if it's Arena-Hard special format
                if prompt.startswith("SYSTEM_USER_FORMAT:"):
                    # Parse system message and user message
                    parts = prompt[len("SYSTEM_USER_FORMAT:"):].split("|USER:")
                    if len(parts) == 2:
                        system_message, user_message = parts
                        # Use system message format for chat
                        response = self.llm.chat(
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_message}
                            ]
                        )
                    else:
                        # Fall back to normal format
                        response = self.llm.simple_chat(query=prompt)
                else:
                    # Normal single-turn conversation
                    response = self.llm.simple_chat(query=prompt)
                
                logger.info(f"response: {response}")
                
                # Extract text content from ChatResponse object
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    response_text = response.message.content
                elif hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Parse response
                parsed_response = self.template.parse(response_text)
                result = self._after_evaluate(parsed_response, **kwargs)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in evaluation attempt {i+1}: {e}")
                if i == self.max_retries - 1:
                    # When last attempt fails, return empty result
                    return RewardResult(
                        name=self.name,
                        details=[],
                        extra_data={
                            "decision": "",
                            "judge_type": self.judge_type,
                            "scores": [0, 0],
                            "error": str(e)
                        }
                    )
                    
        # This code should not be reached, but kept for completeness
        return RewardResult(name=self.name, details=[], extra_data={})
    
    def _after_evaluate(
        self, response: JudgeBenchBaseTemplate, sample: DataSample, **kwargs
    ) -> RewardResult:
        """Post-process evaluation results"""
        
        # Calculate scores based on decision result
        decision = response.decision
        scores = [0, 0]  # [Score for A, Score for B]
        
        if decision == "A>B":
            scores = [1, 0]
        elif decision == "B>A":
            scores = [0, 1]
        elif decision == "A=B":
            scores = [0.5, 0.5]
        else:
            # When decision cannot be parsed, score is 0
            scores = [0, 0]
        
        # Ensure reason field is a valid string
        reason = getattr(response, 'reason', None)
        if reason is None or reason == "":
            reason = f"JudgeBench {self.judge_type} evaluation decision: {decision}"
        
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=f"{self.name}_{self.judge_type}",
                    reason=reason,
                    rank=scores
                )
            ],
            extra_data={
                "decision": decision,
                "judge_type": self.judge_type,
                "scores": scores
            }
        )


class JudgeBenchEvaluator(BaseEvaluator):
    """JudgeBench evaluator"""
    
    reward: JudgeBenchReward = Field(
        default=...,
        description="Reward module"
    )
    
    def compute_accuracy(self, results: List[DataSample]) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        
        if not results:
            logger.warning("No evaluation results")
            return {
                "accuracy": 0.0,
                "correct_count": 0,
                "valid_samples": 0,
                "total_samples": 0
            }
        
        correct_count = 0
        valid_count = 0
        
        for sample in results:
            try:
                # Get ground truth answer
                if "judgebench" not in sample.input[-1].additional_kwargs:
                    logger.warning(f"Sample {sample.unique_id}: no 'judgebench' key in additional_kwargs")
                    continue
                    
                judgebench_data = sample.input[-1].additional_kwargs["judgebench"]
                if "label" not in judgebench_data:
                    logger.warning(f"Sample {sample.unique_id}: no 'label' key in judgebench data")
                    continue
                    
                label = judgebench_data["label"]
                
                # Get prediction result - get evaluation result from additional_kwargs
                if self.reward.name in sample.input[-1].additional_kwargs:
                    reward_extra_data = sample.input[-1].additional_kwargs[self.reward.name]
                    if "decision" in reward_extra_data:
                        decision = reward_extra_data["decision"]
                        
                        # Compare prediction result with ground truth
                        if decision == label:
                            correct_count += 1
                        
                        valid_count += 1
                    else:
                        logger.warning(f"Sample {sample.unique_id}: no 'decision' field in reward_extra_data")
                else:
                    logger.warning(f"Sample {sample.unique_id}: no {self.reward.name} field in additional_kwargs")
                        
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
        
        if valid_count == 0:
            logger.warning("No valid evaluation results")
            return {
                "accuracy": 0.0,
                "correct_count": 0,
                "valid_samples": 0,
                "total_samples": len(results)
            }
        
        accuracy = correct_count / valid_count
        
        return {
            "accuracy": float(accuracy),
            "correct_count": correct_count,
            "valid_samples": valid_count,
            "total_samples": len(results)
        }
    
    def summary(self, results: List[DataSample]) -> dict:
        """Generate evaluation summary"""
        
        # Calculate overall accuracy
        overall_accuracy = self.compute_accuracy(results)
        
        # Calculate accuracy by data source grouping
        source_accuracy = {}
        
        # Get all unique data sources
        sources = list(set(
            sample.metadata.get("source", "unknown") 
            for sample in results
        ))
        
        for source in sources:
            source_results = [
                sample for sample in results 
                if sample.metadata.get("source") == source
            ]
            if source_results:
                source_accuracy[source] = self.compute_accuracy(source_results)
        
        # Compile results
        final_results = {
            "model": self.reward.llm.model,
            "judge_type": self.reward.judge_type,
            "overall_accuracy": overall_accuracy,
            "source_accuracy": source_accuracy
        }
        
        return final_results
    
    def run(self, samples: List[DataSample], **kwargs):
        """Run evaluation with support for bidirectional evaluation to detect position bias"""
        
        # Optional: randomly shuffle answer order to reduce position bias
        if kwargs.get("shuffle_responses", False):
            for sample in samples:
                if len(sample.output) >= 2:
                    random.shuffle(sample.output)
        
        # Execute evaluation
        summary = super().run(samples, **kwargs)
        return summary


def main(
    data_path: str = "data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
    result_path: str = "data/results/judgebench.json",
    judge_type: str = "arena_hard",
    max_samples: int = 10,
    model: str = "deepseek-chat",
    max_workers: int = 4,
    shuffle_responses: bool = False,
):
    """
    Main evaluation function

    Args:
        data_path: Data file path
        result_path: Result save path
        judge_type: Judge type (vanilla, arena_hard, auto_j, prometheus_2, skywork_critic)
        max_samples: Maximum number of samples
        model: Model to use
        max_workers: Maximum number of concurrent workers
        shuffle_responses: Whether to randomly shuffle answer order
    """
    
    # Create data loading configuration
    config = {
        "path": data_path,
        "limit": max_samples,
    }
    
    # Initialize data loading module
    load_module = create_loader(
        name="judgebench",
        load_strategy_type="local",
        data_source="judgebench",
        config=config,
    )
    
    # Initialize language model for evaluation
    if isinstance(model, str):
        llm = OpenaiLLM(model=model)
    else:
        llm = OpenaiLLM(**model)
    # Load dataset
    dataset = load_module.run()
    
    # Create evaluator
    evaluator = JudgeBenchEvaluator(
        reward=JudgeBenchReward(
            name="judgebench_eval",  # Use different name to avoid conflicts
            judge_type=judge_type,
            llm=llm,
            max_workers=max_workers,
        )
    )
    
    # Execute evaluation
    results = evaluator.run(
        samples=dataset.get_data_samples(),
        shuffle_responses=shuffle_responses
    )
    
    # Save results
    write_json(results, result_path)
    
    # Output results
    print(f"Evaluation completed!")
    print(f"Judge type: {judge_type}")
    print(f"Model: {model}")
    print(f"Overall accuracy: {results['overall_accuracy']['accuracy']:.2%}")
    print(f"Valid samples: {results['overall_accuracy']['valid_samples']}")
    print(f"Total samples: {results['overall_accuracy']['total_samples']}")
    
    # Output accuracy by data source
    if results['source_accuracy']:
        print("\nAccuracy by source:")
        for source, accuracy in results['source_accuracy'].items():
            print(f"  {source}: {accuracy['accuracy']:.2%}")


if __name__ == "__main__":
    fire.Fire(main) 