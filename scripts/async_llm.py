# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Zhaoyang
# @Desc    :

from openai import AsyncOpenAI
from scripts.formatter import BaseFormatter, FormatError

import yaml
from pathlib import Path
from typing import Dict, Optional, Any
import aiohttp
import json
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", "https://oneapi.deepwisdom.ai/v1")
        self.top_p = config.get("top_p", 1)
        # VNPT specific fields
        self.api_type = config.get("api_type", "openai")
        self.authorization = config.get("authorization", None)
        self.token_id = config.get("token_id", None)
        self.token_key = config.get("token_key", None)


class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""

    _instance = None  # For singleton pattern if needed
    _default_config = None

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        self.configs = config_dict or {}

    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            # Look for the config file in common locations
            config_paths = [
                Path("config/config2.yaml"),
                Path("config2.yaml"),
                Path("./config/config2.yaml"),
            ]

            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break

            if config_file is None:
                raise FileNotFoundError(
                    "No default configuration file found in the expected locations"
                )

            # Load the YAML file
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Your YAML has a 'models' top-level key that contains the model configs
            if "models" in config_data:
                config_data = config_data["models"]

            cls._default_config = cls(config_data)

        return cls._default_config

    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")

        config = self.configs[llm_name]
        
        # Get api_type to determine how to map fields
        api_type = config.get("api_type", "openai")

        # Create a config dictionary with the expected keys for LLMConfig
        llm_config = {
            "model": llm_name,  # Use the key as the model name
            "temperature": config.get("temperature", 1),
            "base_url": config.get("base_url", "https://oneapi.deepwisdom.ai/v1"),
            "top_p": config.get("top_p", 1),
            "api_type": api_type,
        }
        
        # Handle different API types
        if api_type == "vnpt":
            # VNPT uses authorization, token_id, token_key instead of api_key
            llm_config["authorization"] = config.get("authorization")
            llm_config["token_id"] = config.get("token_id")
            llm_config["token_key"] = config.get("token_key")
            llm_config["key"] = None  # VNPT doesn't use api_key
        else:
            # OpenAI, Gemini, etc. use api_key
            llm_config["key"] = config.get("api_key")

        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)

    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config

    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())


class ModelPricing:
    """Pricing information for different models in USD per 1K tokens"""

    PRICES = {
        # GPT-4o models
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "o3": {"input": 0.003, "output": 0.015},
        "o3-mini": {"input": 0.0011, "output": 0.0025},
        # Google Gemini models
        "gemini-2.0-flash-exp": {"input": 0, "output": 0},  # Free during preview
        "gemini-2.0-flash": {"input": 0, "output": 0},  # Free during preview
        # DeepSeek V3.2-Exp models (giá per 1K tokens)
        "deepseek-chat": {
            "input": 0.00028,
            "output": 0.00042,
        },  # $0.28/$0.42 per 1M tokens
        "deepseek-reasoner": {"input": 0.00028, "output": 0.00042},  # Same pricing
        "vnpt-llm-large": {"input": 0, "output": 0},
        "vnpt-llm-small": {"input": 0, "output": 0},
        "vnpt-llm-embedding": {"input": 0, "output": 0},
    }

    @classmethod
    def get_price(cls, model_name, token_type):
        """Get the price per 1K tokens for a specific model and token type (input/output)"""
        # Try to find exact match first
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]

        # Try to find a partial match (e.g., if model name contains version numbers)
        for key in cls.PRICES:
            if key in model_name:
                return cls.PRICES[key][token_type]

        # Return default pricing if no match found
        return 0


class TokenUsageTracker:
    """Tracks token usage and calculates costs"""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []

    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost

        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output"),
            },
        }

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)

        return usage_record

    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history,
        }


class VNPTAsyncLLM:
    """Async LLM client specifically for VNPT API"""
    
    def __init__(self, config, system_msg: str = None):
        """
        Initialize the VNPTAsyncLLM with a configuration
        
        Args:
            config: LLMConfig instance with VNPT-specific fields
            system_msg: Optional system message to include in all prompts
        """
        self.config = config
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        
        # VNPT API endpoint - dựa trên model name để xác định endpoint
        model_name = config.model.lower()
        if "large" in model_name:
            self.endpoint = f"{config.base_url}/v1/chat/completions/vnptai-hackathon-large"
        elif "small" in model_name:
            self.endpoint = f"{config.base_url}/v1/chat/completions/vnptai-hackathon-small"
        elif "embedding" in model_name:
            self.endpoint = f"{config.base_url}/vnptai-hackathon-embedding"
        else:
            # Default to small model
            self.endpoint = f"{config.base_url}/v1/chat/completions/vnptai-hackathon-small"
    
    @retry(
        stop=stop_after_delay(3800),  # Max delay 3800 seconds
        wait=wait_exponential(multiplier=1, min=4, max=60),  # Exponential backoff: 4s, 8s, 16s, 32s, 60s, 60s...
        retry=retry_if_exception_type((aiohttp.ClientError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
        
    async def __call__(self, prompt):
        """Call VNPT API with the given prompt"""
        message = []
        if self.sys_msg is not None:
            message.append({"content": self.sys_msg, "role": "system"})
        
        message.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": self.config.authorization,
            "Token-id": self.config.token_id,
            "Token-key": self.config.token_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": message,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"VNPT API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                # Extract response content
                if "choices" in result and len(result["choices"]) > 0:
                    ret = result["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"Unexpected VNPT API response format: {result}")
                
                # Track token usage if available in response
                input_tokens = result.get("usage", {}).get("prompt_tokens", 0)
                output_tokens = result.get("usage", {}).get("completion_tokens", 0)
                
                if input_tokens > 0 or output_tokens > 0:
                    usage_record = self.usage_tracker.add_usage(
                        self.config.model, input_tokens, output_tokens
                    )
                    
                    print(ret)
                    print(
                        f"Token usage: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total"
                    )
                    print(
                        f"Cost: ${usage_record['total_cost']:.6f} (${usage_record['input_cost']:.6f} for input, ${usage_record['output_cost']:.6f} for output)"
                    )
                else:
                    print(ret)
                
                return ret
    
    async def call_with_format(self, prompt: str, formatter: BaseFormatter):
        """
        Call the LLM with a prompt and format the response using the provided formatter
        
        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response
            
        Returns:
            The formatted response data
            
        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # Prepare the prompt with formatting instructions
        formatted_prompt = formatter.prepare_prompt(prompt)
        # Call the LLM
        response = await self.__call__(formatted_prompt)
        
        # Validate and parse the response
        is_valid, parsed_data = formatter.validate_response(response)
        
        if not is_valid:
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}. Raw response: {response}")
        
        return parsed_data
    
    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()


class AsyncLLM:
    def __init__(self, config, system_msg: str = None):
        """
        Initialize the AsyncLLM with a configuration

        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)

        # At this point, config should be an LLMConfig instance
        self.config = config
        
        # Only initialize AsyncOpenAI for non-VNPT APIs
        if config.api_type != "vnpt":
            self.aclient = AsyncOpenAI(
                api_key=self.config.key, base_url=self.config.base_url
            )
        else:
            self.aclient = None  # VNPT will use its own client
            
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()

    async def __call__(self, prompt):
        message = []
        if self.sys_msg is not None:
            message.append({"content": self.sys_msg, "role": "system"})

        message.append({"role": "user", "content": prompt})

        response = await self.aclient.chat.completions.create(
            model=self.config.model,
            messages=message,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # Extract token usage from response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Track token usage and calculate cost
        usage_record = self.usage_tracker.add_usage(
            self.config.model, input_tokens, output_tokens
        )

        ret = response.choices[0].message.content
        print(ret)

        # You can optionally print token usage information
        print(
            f"Token usage: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total"
        )
        print(
            f"Cost: ${usage_record['total_cost']:.6f} (${usage_record['input_cost']:.6f} for input, ${usage_record['output_cost']:.6f} for output)"
        )

        return ret

    async def call_with_format(self, prompt: str, formatter: BaseFormatter):
        """
        Call the LLM with a prompt and format the response using the provided formatter

        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response

        Returns:
            The formatted response data

        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # Prepare the prompt with formatting instructions
        formatted_prompt = formatter.prepare_prompt(prompt)
        # Call the LLM
        response = await self.__call__(formatted_prompt)

        # Validate and parse the response
        is_valid, parsed_data = formatter.validate_response(response)

        if not is_valid:
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}. Raw response: {response}")

        return parsed_data

    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()


def create_llm_instance(llm_config):
    """
    Create an AsyncLLM instance using the provided configuration

    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config

    Returns:
        An instance of AsyncLLM or VNPTAsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        if llm_config.api_type == "vnpt":
            return VNPTAsyncLLM(llm_config)
        else:
            return AsyncLLM(llm_config)

    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        config = LLMsConfig.default().get(llm_config)
        if config.api_type == "vnpt":
            return VNPTAsyncLLM(config)
        else:
            return AsyncLLM(config)

    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        config = LLMConfig(llm_config)
        if config.api_type == "vnpt":
            return VNPTAsyncLLM(config)
        else:
            return AsyncLLM(config)

    else:
        raise TypeError(
            "llm_config must be an LLMConfig instance, a string, or a dictionary"
        )
