from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_9.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple step-by-step reasoning paths
        reasoning_paths = []
        for _ in range(3):
            reasoning = await self.answer_gen(input=problem)
            reasoning_paths.append(reasoning['answer'])
        
        # Use self-consistency to select most frequent answer
        ensemble_answer = await self.sc_ensemble(solutions=reasoning_paths)
        
        # Verify the ensemble answer with reasoning context
        solution = await self.custom(
            input=f"Question: {problem}\nProposed Answer: {ensemble_answer['response']}",
            instruction=prompt_custom.VERIFY_PROMPT
        )
        
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
