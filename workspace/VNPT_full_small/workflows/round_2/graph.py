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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Sử dụng Custom để tạo ra câu trả lời ban đầu
        initial_solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        
        # Sử dụng AnswerGenerate để tạo ra câu trả lời chi tiết hơn
        detailed_solution = await self.answer_generate(input=initial_solution['response'])
        
        # Sử dụng ScEnsemble để chọn ra câu trả lời tốt nhất từ các câu trả lời đã tạo ra
        final_solution = await self.sc_ensemble(solutions=[initial_solution['response'], detailed_solution['answer']])
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
