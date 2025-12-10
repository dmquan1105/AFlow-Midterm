from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_3.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solutions = []
        for _ in range(5):  # Generate 5 responses
            solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
            solutions.append(solution['response'])
        
        # Select the most frequent response
        from collections import Counter
        most_common_response = Counter(solutions).most_common(1)[0][0]
        
        return most_common_response, self.llm.get_usage_summary()["total_cost"]
