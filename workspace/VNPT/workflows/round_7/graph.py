from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_7.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        refined_problem = await self.custom(input=problem, instruction=prompt_custom.REFINE_PROMPT)
        solution = await self.answer_generate(input=refined_problem['response'])
        return solution['answer'], self.llm.get_usage_summary()["total_cost"]
