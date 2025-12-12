from typing import Literal
import workspace.VNPT.workflows.template.operator as operator
import workspace.VNPT.workflows.round_3.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

```python
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
        # Tiền xử lý đầu vào
        context = f"Vấn đề: {problem}\nNgữ cảnh: {self.dataset.context}"
        
        # Gọi custom nhiều lần với các hướng dẫn khác nhau
        responses = []
        for i in range(3):  # Số lần gọi có thể điều chỉnh
            instruction = f"Hướng dẫn {i+1}: {prompt_custom.XXX_PROMPT}"
            response = await self.custom(input=context, instruction=instruction)
            responses.append(response['response'])
        
        # Kết hợp kết quả
        combined_response = "\n".join(responses)
        
        # Sử dụng AnswerGenerate để tạo ra quá trình suy nghĩ từng bước
        step_by_step = await self.answer_generate(input=combined_response)
        
        # Trả về câu trả lời cuối cùng và tổng chi phí
        return step_by_step['answer'], self.llm.get_usage_summary()["total_cost"]
```
