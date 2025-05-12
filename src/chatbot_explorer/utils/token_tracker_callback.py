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
        # self.call_details = [] # Optional for detailed per-call logging

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Union[UUID, None] = None, **kwargs: Any
    ) -> None:
        super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.call_count += 1
        # logger.debug(f"LLM Call {self.call_count} Start...")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Union[UUID, None] = None, **kwargs: Any) -> None:
        super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.successful_calls += 1

        prompt_tokens_api = 0
        completion_tokens_api = 0
        total_tokens_api = 0

        if response.llm_output:
            # OpenAI-style
            token_usage_openai = response.llm_output.get('token_usage', {})
            if isinstance(token_usage_openai, dict) and token_usage_openai:
                prompt_tokens_api = token_usage_openai.get('prompt_tokens', 0)
                completion_tokens_api = token_usage_openai.get('completion_tokens', 0)
                total_tokens_api = token_usage_openai.get('total_tokens', 0)
                # logger.debug("OpenAI-style token usage found.")
            else:
                # Gemini/Vertex-style (and other potential structures)
                usage_metadata = response.llm_output.get('usage_metadata', {}) # Common for Google
                if isinstance(usage_metadata, dict) and usage_metadata:
                    prompt_tokens_api = usage_metadata.get('prompt_token_count', usage_metadata.get('prompt_tokens', 0))
                    completion_tokens_api = usage_metadata.get('candidates_token_count',
                                                              usage_metadata.get('candidate_token_count', # some models use singular
                                                              usage_metadata.get('completion_tokens',
                                                              usage_metadata.get('completion_token_count', 0))))
                    total_tokens_api = usage_metadata.get('total_token_count', 0)
                    # logger.debug("Google/Vertex-style token usage found in 'usage_metadata'.")

                # Fallback for other direct keys if specific structures not found
                if not (prompt_tokens_api or completion_tokens_api or total_tokens_api):
                    # logger.debug("Checking for direct token keys in llm_output.")
                    prompt_tokens_api = response.llm_output.get('prompt_tokens', response.llm_output.get('input_tokens', 0))
                    completion_tokens_api = response.llm_output.get('completion_tokens', response.llm_output.get('output_tokens', 0))
                    total_tokens_api = response.llm_output.get('total_tokens', 0)

            if total_tokens_api == 0 and (prompt_tokens_api > 0 or completion_tokens_api > 0):
                total_tokens_api = prompt_tokens_api + completion_tokens_api

        if not (prompt_tokens_api or completion_tokens_api or total_tokens_api):
            logger.warning(
                f"LLM Call {self.successful_calls} End: Token usage information not found or all zeros in llm_output: {str(response.llm_output)[:200]}"
            )
        else:
            self.total_prompt_tokens += prompt_tokens_api
            self.total_completion_tokens += completion_tokens_api
            self.total_tokens += total_tokens_api
            logger.debug(
                f"LLM Call {self.successful_calls} End. Tokens This Call: {total_tokens_api} (P: {prompt_tokens_api}, C: {completion_tokens_api}). Cumulative Total: {self.total_tokens}"
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
