"""Token usage tracking callback for LangChain LLM interactions.

This module provides a callback handler that tracks token usage, costs, and model information
across LLM calls, with support for different phases of analysis.
"""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from tracer.utils.logging_utils import get_logger

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
    """Callback handler for tracking LLM token usage and costs.

    Tracks token consumption across different phases of analysis and calculates
    associated costs based on model pricing data.
    """

    def __init__(self) -> None:
        """Initialize the token tracker with zero counters."""
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

    def mark_analysis_phase(self) -> None:
        """Mark the start of the analysis phase for token tracking."""
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
        **kwargs: dict[str, Any],
    ) -> None:
        """Handle the start of an LLM call.

        Args:
            serialized: LLM configuration data
            prompts: List of prompts being sent to the LLM
            run_id: Unique identifier for this LLM run
            parent_run_id: Parent run identifier if applicable
            **kwargs: Additional keyword arguments
        """
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

        super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def _extract_usage_from_aimessage(self, response: LLMResult) -> tuple[int, int, int, str] | None:
        """Extract token usage from AIMessage usage_metadata.

        Args:
            response: The LLM response object

        Returns:
            Tuple of (prompt_tokens, completion_tokens, total_tokens, source) or None
        """
        if (
            response.generations
            and response.generations[0]
            and hasattr(response.generations[0][0].message, "usage_metadata")
        ):
            usage_data = response.generations[0][0].message.usage_metadata
            if usage_data:
                # Handle both dictionary-style and attribute-style access
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

                # Try dictionary-style access first (common with Google/Gemini)
                if isinstance(usage_data, dict):
                    input_tokens = usage_data.get("input_tokens", 0)
                    output_tokens = usage_data.get("output_tokens", 0)
                    total_tokens = usage_data.get("total_tokens", input_tokens + output_tokens)
                # Try attribute-style access as fallback
                elif hasattr(usage_data, "input_tokens") and hasattr(usage_data, "output_tokens"):
                    input_tokens = getattr(usage_data, "input_tokens", 0)
                    output_tokens = getattr(usage_data, "output_tokens", 0)
                    total_tokens = getattr(usage_data, "total_tokens", input_tokens + output_tokens)

                if input_tokens > 0 or output_tokens > 0:  # Accept even if one is zero
                    source = "AIMessage.usage_metadata"
                    logger.debug(
                        "Found tokens in AIMessage.usage_metadata: input=%d, output=%d, total=%d",
                        input_tokens,
                        output_tokens,
                        total_tokens,
                    )
                    return (input_tokens, output_tokens, total_tokens, source)
        return None

    def _extract_usage_from_llm_output(self, response: LLMResult) -> tuple[int, int, int, str] | None:
        """Extract token usage from response.llm_output.

        Args:
            response: The LLM response object

        Returns:
            Tuple of (prompt_tokens, completion_tokens, total_tokens, source) or None
        """
        if not response.llm_output:
            return None

        # Check various possible keys in llm_output
        usage_keys = [
            "token_usage",
            "usage",
            "tokenUsage",
            "token_count",
            "tokens",
        ]

        for key in usage_keys:
            if key in response.llm_output:
                usage_data = response.llm_output[key]
                if isinstance(usage_data, dict):
                    prompt_tokens = usage_data.get("prompt_tokens", 0)
                    completion_tokens = usage_data.get("completion_tokens", 0)
                    total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)

                    if prompt_tokens > 0 and completion_tokens > 0:
                        return (
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            f"response.llm_output['{key}']",
                        )
        return None

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Handle the end of an LLM call and extract token usage.

        Args:
            response: The LLM response containing generations and metadata
            run_id: Unique identifier for this LLM run
            parent_run_id: Parent run identifier if applicable
            **kwargs: Additional keyword arguments
        """
        super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        # Try to extract token usage from different sources
        usage_result = self._extract_usage_from_aimessage(response)
        if not usage_result:
            usage_result = self._extract_usage_from_llm_output(response)

        if usage_result:
            prompt_tokens_api, completion_tokens_api, total_tokens_api, source_of_tokens = usage_result

            # Update counters
            self.total_prompt_tokens += prompt_tokens_api
            self.total_completion_tokens += completion_tokens_api
            self.total_tokens += total_tokens_api
            self.successful_calls += 1

            logger.debug(
                "LLM Call %d End. Tokens This Call: %d (P: %d, C: %d) from '%s'. Cumulative Total: %d",
                self.successful_calls,
                total_tokens_api,
                prompt_tokens_api,
                completion_tokens_api,
                source_of_tokens,
                self.total_tokens,
            )
        else:
            # No tokens found from any source
            logger.warning(
                "LLM Call %d End: Token usage information not found or all zeros. "
                "llm_output: %s. "
                "AIMessage.usage_metadata: %s. "
                "AIMessage.response_metadata: %s",
                self.successful_calls,
                str(response.llm_output)[:200],
                str(getattr(response.generations[0][0].message, "usage_metadata", None))
                if response.generations and response.generations[0]
                else "N/A",
                str(getattr(response.generations[0][0].message, "response_metadata", None))
                if response.generations and response.generations[0]
                else "N/A",
            )

        # Track model name if available
        if hasattr(response, "model_name") and response.model_name:
            self.model_names_used.add(response.model_name)

    def on_llm_error(
        self,
        error: Exception | KeyboardInterrupt,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Handle LLM errors and track failed calls.

        Args:
            error: The exception that occurred
            run_id: Unique identifier for this LLM run
            parent_run_id: Parent run identifier if applicable
            **kwargs: Additional keyword arguments
        """
        super().on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        self.failed_calls += 1
        logger.error(
            "LLM Call Error. Total Calls: %d, Failed: %d. Error: %s", self.call_count, self.failed_calls, error
        )

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
                cost_model_reported = next(iter(self.model_names_used))
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
        """Get a summary of token usage statistics and costs."""
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
        """Return a string representation of the token usage tracker."""
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
