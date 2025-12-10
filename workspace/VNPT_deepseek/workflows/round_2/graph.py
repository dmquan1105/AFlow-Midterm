from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_2.prompt as prompt_custom
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
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # First generate step-by-step reasoning
        reasoning = await self.answer_generate(input=problem)
        
        # Use reasoning in custom prompt for final answer
        solution = await self.custom(
            input=f"Question: {problem}\nReasoning: {reasoning['thought']}", 
            instruction=prompt_custom.FINAL_ANSWER_PROMPT
        )
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
