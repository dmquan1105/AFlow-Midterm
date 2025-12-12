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
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        # Thay đổi phương thức lấy câu trả lời từ mô hình ngôn ngữ
        answer = self.extract_answer(solution['response'])
        cost = self.llm.get_usage_summary()["total_cost"]
        return answer, cost

    def extract_answer(self, text: str) -> str:
        """
        Trích xuất câu trả lời từ văn bản đầu ra của mô hình ngôn ngữ
        """
        # Tìm kiếm mẫu câu trả lời trong văn bản
        pattern = r"(?:Answer|Đáp án|answer|đáp án)\s*[:：]\s*(\w+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        else:
            return ""
