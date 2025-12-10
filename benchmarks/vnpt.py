import inspect
import re
from typing import Any, Callable, List, Tuple

import regex
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger

class VNPTBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
    
    def extract_answer_letter(self, text: str, num_choices: int) -> str:
        """Extract answer letter from model output based on number of choices
        
        Args:
            text: Model output text
            num_choices: Number of choices (e.g., 4 for A-D, 5 for A-E, etc.)
            
        Returns:
            Extracted answer letter or empty string if not found
        """
        # Generate valid letters dynamically based on num_choices
        max_letter = chr(65 + num_choices - 1)  # 65 is ASCII for 'A'
        valid_pattern = f"[A-{max_letter}]"
        
        # Try to find answer in format "Answer: X" or "Đáp án: X"
        patterns = [
            rf"(?:Answer|Đáp án|answer|đáp án)\s*[:：]\s*({valid_pattern})",
            rf"\b({valid_pattern})\b(?:\s*[.)]|\s*$)",  # Single letter
            rf"^\s*({valid_pattern})\s*[.)]",  # Letter at start with dot or parenthesis
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, look for last occurrence of valid letters
        matches = re.findall(rf'\b({valid_pattern})\b', text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        return ""
    
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        """Evaluate a single problem"""
        question = problem["question"]
        choices = problem["choices"]
        expected_answer = problem["answer"]  # This is "A", "B", "C", "D", etc.
        num_choices = len(choices)
        
        # Generate dynamic choice range text
        max_letter = chr(65 + num_choices - 1)
        choice_range = f"A-{max_letter}" if num_choices > 1 else "A"
        
        # Format the question with choices
        formatted_question = f"{question}\n\nChoices:\n"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # Convert 0->A, 1->B, etc.
            formatted_question += f"{letter}. {choice}\n"
        formatted_question += f"\nPlease answer with the letter ({choice_range}) of the correct choice."
        
        try:
            output, cost = await self._generate_output(agent, formatted_question)
            score, extracted_answer = self.calculate_score(expected_answer, output, num_choices)
            
            if score == 0:
                self.log_mismatch(
                    question,
                    expected_answer,
                    output,
                    extracted_answer,
                    extract_answer_code=self.get_function_code(self.extract_answer_letter),
                )
            
            return question, output, expected_answer, score, cost
            
        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return question, str(e), expected_answer, 0.0, 0.0

    def calculate_score(self, expected_output: Any, prediction: Any, num_choices: int) -> Tuple[float, Any]:
        """Calculate score based on exact match of answer letter
        
        Args:
            expected_output: Expected answer letter
            prediction: Model prediction text
            num_choices: Number of choices for this question
            
        Returns:
            Tuple of (score, extracted_answer)
        """
        expected_answer = str(expected_output).strip().upper()
        predicted_answer = self.extract_answer_letter(prediction, num_choices)
        
        # Calculate precision: 1 if exact match, 0 otherwise
        score = 1.0 if predicted_answer == expected_answer else 0.0
        
        return score, predicted_answer

    def get_result_columns(self) -> List[str]:
        """Return column names for results CSV"""
        return ["question", "prediction", "expected_output", "score", "cost"]
    
    def get_function_code(self, func):
        """Get source code of a function for logging"""
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"
    
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        """Generate output from the agent with retry logic"""
        return await graph(input_text)