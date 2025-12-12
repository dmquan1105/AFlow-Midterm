from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_3.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

import asyncio

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
        self.sc_ensemble = operator.ScEnsemble()

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate 3 candidate answers in parallel
        solutions = await asyncio.gather(*[self.answer_generate(input=problem) for _ in range(3)])
        # Use self-consistency to select the most frequent answer
        final_solution = await self.sc_ensemble(solutions=[s['answer'] for s in solutions])
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
