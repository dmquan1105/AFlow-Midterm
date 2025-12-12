from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_5.prompt as prompt_custom
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
custom_response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
answer_generate_response = await self.answer_generate(input=problem)

# Ensemble các câu trả lời
ensemble_response = self.ensemble(custom_response, answer_generate_response)

return ensemble_response['response'], self.llm.get_usage_summary()["total_cost"]

def ensemble(self, custom_response, answer_generate_response):
"""
Kết hợp các câu trả lời từ Custom và AnswerGenerate
"""
# Sử dụng phương thức ensemble để kết hợp các câu trả lời
ensemble_response = {}
ensemble_response['response'] = custom_response['response'] + "\n" + answer_generate_response['thought'] + "\n" + answer_generate_response['answer']
return ensemble_response
