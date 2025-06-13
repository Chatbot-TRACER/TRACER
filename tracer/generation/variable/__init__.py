"""Variable definition and processing module.

This module provides functionality for defining, validating, and processing
variables in user profiles and goals.
"""

from .variable_definition_core import VariableDefinitionContext, define_single_variable_with_retry
from .variable_definition_main import VariableDefinitionConfig, generate_variable_definitions
from .variable_matching import match_variables_to_data_sources_with_llm
from .variable_parameter_extraction import extract_parameter_options_for_profile
from .variable_smart_defaults import generate_smart_default_options
from .variable_validation import validate_semantic_match

__all__ = [
    "VariableDefinitionConfig",
    "VariableDefinitionContext",
    "define_single_variable_with_retry",
    "extract_parameter_options_for_profile",
    "generate_smart_default_options",
    "generate_variable_definitions",
    "match_variables_to_data_sources_with_llm",
    "validate_semantic_match",
]
