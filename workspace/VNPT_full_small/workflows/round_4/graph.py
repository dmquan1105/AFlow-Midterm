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

async def __call__(self, problem: str):
"""
Implementation of the workflow
"""
solutions = []
for i in range(3):  # Số lần chạy phương thức Custom
instruction = f"XXX_PROMPT_{i}"  # Sử dụng hướng dẫn khác nhau cho mỗi lần chạy
solution = await self.custom(input=problem, instruction=prompt_custom[instruction])
solutions.append(solution['response'])

final_solution = self.sc_ensemble(solutions)  # Sử dụng ensemble learning để chọn câu trả lời cuối cùng
return final_solution, self.llm.get_usage_summary()["total_cost"]

def sc_ensemble(self, solutions: List[str]) -> str:
"""
Uses self-consistency to select the solution that appears most frequently in the solution list.
"""
from collections import Counter
counter = Counter(solutions)
most_common = counter.most_common(1)[0][0]
return most_common
