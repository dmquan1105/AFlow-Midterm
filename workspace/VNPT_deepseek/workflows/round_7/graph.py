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
        self.sc_ensemble = operator.ScEnsemble()

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple reasoning paths (3 times)
        reasoning_paths = []
        for _ in range(3):
            reasoning = await self.answer_generate(input=problem)
            reasoning_paths.append(reasoning['thought'])
        
        # Generate final answers for each reasoning path
        solutions = []
        for reasoning in reasoning_paths:
            solution = await self.custom(
                input=f"Question: {problem}\nReasoning: {reasoning}",
                instruction="Based on the question and reasoning above, provide the final answer letter (A, B, C, D, or E). Format your response as: 'Answer: [LETTER]'"
            )
            solutions.append(solution['response'])
        
        # Use self-consistency ensemble to select most frequent answer
        final_answer = await self.sc_ensemble(solutions=solutions)
        return final_answer['response'], self.llm.get_usage_summary()["total_cost"]
