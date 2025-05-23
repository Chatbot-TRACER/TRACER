from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

# Default costs per 1M tokens for different models (in USD)
DEFAULT_COSTS = {
    # OpenAI models costs per 1M tokens
    "gpt-4o": {"prompt": 5.00, "completion": 20.00},
    "gpt-4o-mini": {"prompt": 0.60, "completion": 2.40},
    "gpt-4.1": {"prompt": 2.00, "completion": 8.00},
    "gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60},
    "gpt-4.1-nano": {"prompt": 0.10, "completion": 0.40},
    # Google/Gemini models costs per 1M tokens
    "gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40},
    "gemini-2.5-flash-preview-05-2023": {"prompt": 0.15, "completion": 0.60},
    # Default fallback rates if model not recognized
    "default": {"prompt": 0.10, "completion": 0.40},
}


class TokenUsageTracker(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.model_names_used = set()  # Track which models were used

        # Track exploration vs. analysis phases
        self.exploration_prompt_tokens = 0
        self.exploration_completion_tokens = 0
        self.exploration_total_tokens = 0
        self.analysis_prompt_tokens = 0
        self.analysis_completion_tokens = 0
        self.analysis_total_tokens = 0
        self._phase = "exploration"  # Start in exploration phase

    def mark_analysis_phase(self):
        """Mark the start of the analysis phase for token tracking"""
        self._phase = "analysis"
        self.exploration_prompt_tokens = self.total_prompt_tokens
        self.exploration_completion_tokens = self.total_completion_tokens
        self.exploration_total_tokens = self.total_tokens
        logger.debug("Marked beginning of analysis phase for token tracking")

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.call_count += 1

        # Track which model is being used
        # Model name can be in 'model' or 'model_name' in kwargs for many Langchain integrations
        model_name_from_serialized_kwargs = None
        if "kwargs" in serialized and isinstance(serialized["kwargs"], dict):
            model_name_from_serialized_kwargs = serialized["kwargs"].get("model") or serialized["kwargs"].get(
                "model_name"
            )

        if model_name_from_serialized_kwargs:
            self.model_names_used.add(model_name_from_serialized_kwargs)
        elif "model_name" in serialized:  # General fallback from top-level of serialized
            self.model_names_used.add(serialized["model_name"])
        # else:
        # logger.debug(f"Model name not found in on_llm_start serialization: {serialized}")
        super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.successful_calls += 1

        prompt_tokens_api = 0
        completion_tokens_api = 0
        total_tokens_api = 0
        source_of_tokens = "unknown"

        # Attempt 1: Check AIMessage.usage_metadata directly (primary for Gemini invoke results)
        if response.generations and response.generations[0] and response.generations[0][0]:
            first_generation_message = response.generations[0][0].message
            if hasattr(first_generation_message, "usage_metadata") and first_generation_message.usage_metadata:
                usage_data = first_generation_message.usage_metadata
                if isinstance(usage_data, dict):
                    prompt_tokens_api = usage_data.get("input_tokens", 0)
                    completion_tokens_api = usage_data.get("output_tokens", 0)
                    total_tokens_api = usage_data.get("total_tokens", 0)
                    if (
                        prompt_tokens_api or completion_tokens_api or total_tokens_api
                    ):  # Check if any token info was actually found
                        source_of_tokens = "AIMessage.usage_metadata (Google invoke style)"
                        logger.debug(f"Found tokens directly in AIMessage.usage_metadata: {usage_data}")

        # Attempt 2: Fallback to response.llm_output if no tokens found yet
        if source_of_tokens == "unknown" and response.llm_output:
            # OpenAI-style 'token_usage'
            token_usage_openai = response.llm_output.get("token_usage", {})
            if isinstance(token_usage_openai, dict) and token_usage_openai.get("total_tokens", 0) > 0:
                prompt_tokens_api = token_usage_openai.get("prompt_tokens", 0)
                completion_tokens_api = token_usage_openai.get("completion_tokens", 0)
                total_tokens_api = token_usage_openai.get("total_tokens", 0)
                source_of_tokens = "llm_output.token_usage (OpenAI-style)"

            # Google-style 'usage_metadata' if nested in llm_output (and not found by OpenAI style)
            if source_of_tokens == "unknown":  # Check if still not found
                usage_metadata_from_llm_output = response.llm_output.get("usage_metadata", {})
                if (
                    isinstance(usage_metadata_from_llm_output, dict)
                    and usage_metadata_from_llm_output.get("total_token_count", 0) > 0
                ):
                    prompt_tokens_api = usage_metadata_from_llm_output.get("prompt_token_count", 0)
                    completion_tokens_api = usage_metadata_from_llm_output.get(
                        "candidates_token_count", usage_metadata_from_llm_output.get("candidate_token_count", 0)
                    )
                    total_tokens_api = usage_metadata_from_llm_output.get("total_token_count", 0)
                    source_of_tokens = "llm_output.usage_metadata (Google-style in llm_output)"

        # Final calculation for total_tokens if not provided directly but components are
        if total_tokens_api == 0 and (prompt_tokens_api > 0 or completion_tokens_api > 0):
            total_tokens_api = prompt_tokens_api + completion_tokens_api

        # Logging and accumulation
        if source_of_tokens != "unknown" or prompt_tokens_api or completion_tokens_api or total_tokens_api:
            self.total_prompt_tokens += prompt_tokens_api
            self.total_completion_tokens += completion_tokens_api
            self.total_tokens += total_tokens_api

            # Track by phase
            if self._phase == "exploration":
                self.exploration_prompt_tokens += prompt_tokens_api
                self.exploration_completion_tokens += completion_tokens_api
                self.exploration_total_tokens += prompt_tokens_api + completion_tokens_api
            else:  # analysis phase
                self.analysis_prompt_tokens += prompt_tokens_api
                self.analysis_completion_tokens += completion_tokens_api
                self.analysis_total_tokens += prompt_tokens_api + completion_tokens_api

            logger.debug(
                f"LLM Call {self.successful_calls} End. Tokens This Call: {total_tokens_api} (P: {prompt_tokens_api}, C: {completion_tokens_api}) from '{source_of_tokens}'. Cumulative Total: {self.total_tokens}"
            )
        else:  # No tokens found from any source
            logger.warning(
                f"LLM Call {self.successful_calls} End: Token usage information not found or all zeros. "
                f"llm_output: {str(response.llm_output)[:200]}. "
                f"AIMessage.usage_metadata: {str(getattr(response.generations[0][0].message, 'usage_metadata', None)) if response.generations and response.generations[0] else 'N/A'}. "
                f"AIMessage.response_metadata: {str(getattr(response.generations[0][0].message, 'response_metadata', None)) if response.generations and response.generations[0] else 'N/A'}"
            )

        # Track model name if available
        if hasattr(response, "model_name") and response.model_name:
            self.model_names_used.add(response.model_name)

    def on_llm_error(
        self, error: Exception | KeyboardInterrupt, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        super().on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.failed_calls += 1
        logger.error(f"LLM Call Error. Total Calls: {self.call_count}, Failed: {self.failed_calls}. Error: {error}")

    def calculate_cost(self) -> dict:
        """Calculate estimated cost based on token usage and models used."""
        cost_model_reported = "default"  # This is what will be shown as "cost_model_used" in the report
        pricing_key_for_lookup = "default"  # This is the key used to get rates from DEFAULT_COSTS

        if self.model_names_used:
            # Attempt to find an exact match in DEFAULT_COSTS from the models that were actually used.
            # If multiple models were used, this prioritizes the first one found with a cost entry.
            found_match_for_pricing = False
            for model_name_actually_used in self.model_names_used:
                if model_name_actually_used in DEFAULT_COSTS:
                    pricing_key_for_lookup = model_name_actually_used
                    cost_model_reported = model_name_actually_used  # Report this specific model
                    found_match_for_pricing = True
                    break

            if not found_match_for_pricing and self.model_names_used:
                # No specific pricing found for any of the used models.
                # Report the first model from the set of used models. Pricing will use 'default' rates.
                cost_model_reported = list(self.model_names_used)[0]
                # pricing_key_for_lookup remains "default"

        cost_info = DEFAULT_COSTS.get(pricing_key_for_lookup, DEFAULT_COSTS["default"])

        # Calculate costs (dividing by 1M to convert from per-million pricing)
        prompt_cost = (self.total_prompt_tokens / 1000000) * cost_info["prompt"]
        completion_cost = (self.total_completion_tokens / 1000000) * cost_info["completion"]
        total_cost = prompt_cost + completion_cost
        # Calculate per phase costs
        exploration_prompt_cost = (self.exploration_prompt_tokens / 1000000) * cost_info["prompt"]
        exploration_completion_cost = (self.exploration_completion_tokens / 1000000) * cost_info["completion"]
        exploration_total_cost = exploration_prompt_cost + exploration_completion_cost

        # For analysis phase, calculate only the tokens used during analysis (subtract exploration tokens)
        analysis_prompt_tokens = self.total_prompt_tokens - self.exploration_prompt_tokens
        analysis_completion_tokens = self.total_completion_tokens - self.exploration_completion_tokens

        analysis_prompt_cost = (analysis_prompt_tokens / 1000000) * cost_info["prompt"]
        analysis_completion_cost = (analysis_completion_tokens / 1000000) * cost_info["completion"]
        analysis_total_cost = analysis_prompt_cost + analysis_completion_cost

        return {
            "prompt_cost": round(prompt_cost, 4),
            "completion_cost": round(completion_cost, 4),
            "total_cost": round(total_cost, 4),
            "cost_model_used": cost_model_reported,
            "exploration_cost": round(exploration_total_cost, 4),
            "analysis_cost": round(analysis_total_cost, 4),
        }

    def get_summary(self) -> dict:
        """Get a summary of token usage statistics and costs"""
        cost_data = self.calculate_cost()

        # Calculate analysis phase tokens (only what was used during analysis)
        analysis_prompt_tokens = self.total_prompt_tokens - self.exploration_prompt_tokens
        analysis_completion_tokens = self.total_completion_tokens - self.exploration_completion_tokens
        analysis_total_tokens = analysis_prompt_tokens + analysis_completion_tokens

        return {
            "total_llm_calls": self.call_count,
            "successful_llm_calls": self.successful_calls,
            "failed_llm_calls": self.failed_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens_consumed": self.total_tokens,
            "models_used": list(self.model_names_used),
            "estimated_cost": cost_data["total_cost"],
            "cost_details": cost_data,
            "exploration_phase": {
                "prompt_tokens": self.exploration_prompt_tokens,
                "completion_tokens": self.exploration_completion_tokens,
                "total_tokens": self.exploration_total_tokens,
                "estimated_cost": cost_data["exploration_cost"],
            },
            "analysis_phase": {
                "prompt_tokens": analysis_prompt_tokens,
                "completion_tokens": analysis_completion_tokens,
                "total_tokens": analysis_total_tokens,
                "estimated_cost": cost_data["analysis_cost"],
            },
        }

    def __str__(self) -> str:
        cost_data = self.calculate_cost()
        current_phase = "Exploration" if self._phase == "exploration" else "Analysis"

        # Calculate only tokens used in this phase for the display
        if current_phase == "Exploration":
            phase_prompt_tokens = self.total_prompt_tokens
            phase_completion_tokens = self.total_completion_tokens
            phase_total_tokens = self.total_tokens
            phase_cost = cost_data["total_cost"]
        else:  # Analysis phase
            phase_prompt_tokens = self.total_prompt_tokens - self.exploration_prompt_tokens
            phase_completion_tokens = self.total_completion_tokens - self.exploration_completion_tokens
            phase_total_tokens = phase_prompt_tokens + phase_completion_tokens
            phase_cost = cost_data["analysis_cost"]

        return (
            f"Token Usage in {current_phase} Phase:\n"
            f"  Prompt tokens:       {phase_prompt_tokens:,}\n"
            f"  Completion tokens:   {phase_completion_tokens:,}\n"
            f"  Total tokens:        {phase_total_tokens:,}\n"
            f"  Estimated cost:      ${phase_cost:.4f} USD"
        )
