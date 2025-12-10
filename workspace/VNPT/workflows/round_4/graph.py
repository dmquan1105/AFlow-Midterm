from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_4.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow with self-consistency
        """
        # Generate multiple reasoning paths (3 times)
        reasoning_paths = []
        for _ in range(3):
            reasoning = await self.answer_generate(input=problem)
            reasoning_paths.append(reasoning['answer'])
        
        # Use ensemble voting to select best answer
        selected_answer = await self.sc_ensemble(solutions=reasoning_paths)
        
        # Generate final formatted answer with reasoning context
        solution = await self.custom(
            input=f"Question: {problem}\nSelected Answer: {selected_answer['response']}", 
            instruction=prompt_custom.FINAL_FORMAT_PROMPT
        )
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
