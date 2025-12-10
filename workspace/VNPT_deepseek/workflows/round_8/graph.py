from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_8.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.answer_gen = operator.AnswerGenerate(self.llm)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate step-by-step reasoning
        reasoning = await self.answer_gen(input=problem)
        
        # Use reasoning as context for final answer with verification
        solution = await self.custom(
            input=f"Question: {problem}\nReasoning: {reasoning['thought']}",
            instruction=prompt_custom.VERIFY_PROMPT
        )
        
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
