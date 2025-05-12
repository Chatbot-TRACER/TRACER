from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Union
from uuid import UUID

from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

class TokenUsageTracker(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.successful_calls = 0
        self.failed_calls = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Union[UUID, None] = None, **kwargs: Any
    ) -> None:
        super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.call_count += 1

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Union[UUID, None] = None, **kwargs: Any) -> None:
        super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.successful_calls += 1

        prompt_tokens_api = 0
        completion_tokens_api = 0
        total_tokens_api = 0
        source_of_tokens = "unknown"

        # Attempt 1: Check AIMessage.usage_metadata directly (primary for Gemini invoke results)
        if response.generations and response.generations[0] and response.generations[0][0]:
            first_generation_message = response.generations[0][0].message
            if hasattr(first_generation_message, 'usage_metadata') and first_generation_message.usage_metadata:
                usage_data = first_generation_message.usage_metadata
                if isinstance(usage_data, dict):
                    prompt_tokens_api = usage_data.get('input_tokens', 0)
                    completion_tokens_api = usage_data.get('output_tokens', 0)
                    total_tokens_api = usage_data.get('total_tokens', 0)
                    if prompt_tokens_api or completion_tokens_api or total_tokens_api: # Check if any token info was actually found
                        source_of_tokens = "AIMessage.usage_metadata (Google invoke style)"
                        logger.debug(f"Found tokens directly in AIMessage.usage_metadata: {usage_data}")

        # Attempt 2: Fallback to response.llm_output if no tokens found yet
        if source_of_tokens == "unknown" and response.llm_output:
            # OpenAI-style 'token_usage'
            token_usage_openai = response.llm_output.get('token_usage', {})
            if isinstance(token_usage_openai, dict) and token_usage_openai.get('total_tokens', 0) > 0:
                prompt_tokens_api = token_usage_openai.get('prompt_tokens', 0)
                completion_tokens_api = token_usage_openai.get('completion_tokens', 0)
                total_tokens_api = token_usage_openai.get('total_tokens', 0)
                source_of_tokens = "llm_output.token_usage (OpenAI-style)"

            # Google-style 'usage_metadata' if nested in llm_output (and not found by OpenAI style)
            if source_of_tokens == "unknown": # Check if still not found
                usage_metadata_from_llm_output = response.llm_output.get('usage_metadata', {})
                if isinstance(usage_metadata_from_llm_output, dict) and usage_metadata_from_llm_output.get('total_token_count', 0) > 0:
                    prompt_tokens_api = usage_metadata_from_llm_output.get('prompt_token_count', 0)
                    completion_tokens_api = usage_metadata_from_llm_output.get('candidates_token_count',
                                                              usage_metadata_from_llm_output.get('candidate_token_count', 0))
                    total_tokens_api = usage_metadata_from_llm_output.get('total_token_count', 0)
                    source_of_tokens = "llm_output.usage_metadata (Google-style in llm_output)"

        # Final calculation for total_tokens if not provided directly but components are
        if total_tokens_api == 0 and (prompt_tokens_api > 0 or completion_tokens_api > 0):
            total_tokens_api = prompt_tokens_api + completion_tokens_api

        # Logging and accumulation
        if source_of_tokens != "unknown" or prompt_tokens_api or completion_tokens_api or total_tokens_api :
            self.total_prompt_tokens += prompt_tokens_api
            self.total_completion_tokens += completion_tokens_api
            self.total_tokens += total_tokens_api
            logger.debug(
                f"LLM Call {self.successful_calls} End. Tokens This Call: {total_tokens_api} (P: {prompt_tokens_api}, C: {completion_tokens_api}) from '{source_of_tokens}'. Cumulative Total: {self.total_tokens}"
            )
        else: # No tokens found from any source
            logger.warning(
                f"LLM Call {self.successful_calls} End: Token usage information not found or all zeros. "
                f"llm_output: {str(response.llm_output)[:200]}. "
                f"AIMessage.usage_metadata: {str(getattr(response.generations[0][0].message, 'usage_metadata', None)) if response.generations and response.generations[0] else 'N/A'}. "
                f"AIMessage.response_metadata: {str(getattr(response.generations[0][0].message, 'response_metadata', None)) if response.generations and response.generations[0] else 'N/A'}"
            )

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], *, run_id: UUID, parent_run_id: Union[UUID, None] = None, **kwargs: Any
    ) -> None:
        super().on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.failed_calls += 1
        logger.error(f"LLM Call Error. Total Calls: {self.call_count}, Failed: {self.failed_calls}. Error: {error}")

    def get_summary(self) -> dict:
        return {
            "total_llm_calls": self.call_count,
            "successful_llm_calls": self.successful_calls,
            "failed_llm_calls": self.failed_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens_consumed": self.total_tokens,
        }

    def __str__(self) -> str:
        return (
            f"LLM Token Usage Summary from Tracker:\n"
            f"  Total LLM Calls: {self.call_count}\n"
            f"  Successful Calls: {self.successful_calls}\n"
            f"  Failed Calls: {self.failed_calls}\n"
            f"  Total Prompt Tokens: {self.total_prompt_tokens}\n"
            f"  Total Completion Tokens: {self.total_completion_tokens}\n"
            f"  Total Tokens Consumed: {self.total_tokens}"
        )
