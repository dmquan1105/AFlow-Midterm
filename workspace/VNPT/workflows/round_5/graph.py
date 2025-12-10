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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble()

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple reasoning paths
        reasoning_tasks = [self.answer_generate(input=problem) for _ in range(3)]
        reasoning_results = await asyncio.gather(*reasoning_tasks)
        
        # Generate final answers from each reasoning
        answer_tasks = []
        for reasoning in reasoning_results:
            task = self.custom(
                input=f"Question: {problem}\nReasoning: {reasoning['thought']}",
                instruction=prompt_custom.FINAL_ANSWER_PROMPT
            )
            answer_tasks.append(task)
        
        answer_results = await asyncio.gather(*answer_tasks)
        
        # Extract answer letters from each response
        answers = []
        for result in answer_results:
            # Extract answer letter using regex
            match = re.search(r'Answer:\s*([A-E])', result['response'], re.IGNORECASE)
            if match:
                answers.append(match.group(1).upper())
            else:
                # Fallback to last letter in response
                letters = re.findall(r'\b([A-E])\b', result['response'])
                if letters:
                    answers.append(letters[-1].upper())
        
        # Use ensemble voting to select final answer
        if answers:
            final_answer = await self.sc_ensemble(solutions=answers)
            return final_answer['response'], self.llm.get_usage_summary()["total_cost"]
        
        # Fallback to first answer if ensemble fails
        return answer_results[0]['response'], self.llm.get_usage_summary()["total_cost"]
