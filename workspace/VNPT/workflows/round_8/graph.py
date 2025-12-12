from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_8.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
def init(
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

async def call(self, problem: str):
"""
Implementation of the workflow
"""
preprocessed_problem = await self.preprocess_question(problem)
solution = await self.answer_generate(input=preprocessed_problem)
return solution['answer'], self.llm.get_usage_summary()["total_cost"]

async def preprocess_question(self, problem: str):
"""
Preprocess the question before sending to the language model
"""
# Add code to preprocess the question here
return problem
