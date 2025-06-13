"""Defines the model for representing a discovered chatbot functionality."""

from typing import Any, Optional


class ParameterDefinition:
    """Model representing a parameter with its metadata."""

    def __init__(self, name: str, description: str, options: list[str]) -> None:
        """Initialize one of the parameters of a Functionality Node.

        Args:
            name: Name of the parameter
            description: Description of the parameter/input
            options: What are the available options for the parameter if any (e.g. Small, Medium, Large)
        """
        self.name = name
        self.description = description
        self.options = options

    def to_dict(self) -> dict[str, Any]:
        """Convert the ParameterDefinition to a serializable dict."""
        return {
            "name": self.name,
            "description": self.description,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterDefinition":
        """Create a ParameterDefinition from a dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            options=data["options"],
        )

    def __repr__(self) -> str:
        """Return string representation of ParameterDefinition."""
        opts = f", options={self.options}" if self.options else ""
        return f"ParameterDefinition(name='{self.name}', description='{self.description}'{opts})"


class OutputOptions:
    """Model representing output options provided by the chatbot."""

    def __init__(self, category: str, description: str = "") -> None:
        """Initialize output options provided by the chatbot.

        Args:
            category: Category name of the options (e.g., "pizza_types", "sizes")
            description: Description of what these options represent
        """
        self.category = category
        self.description = description

    def to_dict(self) -> dict[str, Any]:
        """Convert the OutputOptions to a serializable dict."""
        return {
            "category": self.category,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputOptions":
        """Create an OutputOptions from a dictionary."""
        return cls(
            category=data["category"],
            description=data.get("description", ""),  # Handle optional description
        )

    def __repr__(self) -> str:
        """Return string representation of OutputOptions."""
        return f"OutputOptions(category='{self.category}', description='{self.description}')"


class FunctionalityNode:
    """Represents a discovered chatbot functionality node in the graph."""

    def __init__(
        self,
        name: str,
        description: str,
        *,
        parameters: list[ParameterDefinition] | None = None,
        outputs: list[OutputOptions] | None = None,
        parent: Optional["FunctionalityNode"] = None,
    ) -> None:
        """Initialize a FunctionalityNode.

        Args:
            name: The unique name of the functionality.
            description: A description of what the functionality does.
            parameters: Optional list of ParameterDefinition instances.
            outputs: Optional list of OutputOptions instances.
            parent: The parent node in the functionality hierarchy, if any.
        """
        self.name: str = name
        self.description: str = description
        self.parameters: list[ParameterDefinition] = parameters if parameters else []
        self.outputs: list[OutputOptions] = outputs if outputs else []
        self.parent: FunctionalityNode | None = parent
        self.children: list[FunctionalityNode] = []

    def add_child(self, child_node: "FunctionalityNode") -> None:
        """Adds a child node to this node."""
        child_node.parent = self
        if child_node not in self.children:
            self.children.append(child_node)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Converts the FunctionalityNode instance to a serializable dictionary.
        Excludes the 'parent' attribute to prevent circular references.
        Recursively converts children.
        """
        # Ensure outputs and parameters are never null
        outputs = [output.to_dict() for output in self.outputs if output is not None]
        parameters = [param.to_dict() for param in self.parameters if param is not None]

        return {
            "__type__": "FunctionalityNode",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
            "outputs": outputs,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionalityNode":
        """Create a FunctionalityNode from a dictionary."""
        # Reconstruct parameters
        parameters_data = data.get("parameters", [])
        parameters = [ParameterDefinition.from_dict(p_data) for p_data in parameters_data]

        # Reconstruct outputs
        outputs_data = data.get("outputs", [])
        outputs = [OutputOptions.from_dict(o_data) for o_data in outputs_data]

        # Create the node itself (without children initially)
        node = cls(
            name=data["name"],
            description=data["description"],
            parameters=parameters,
            outputs=outputs,
        )

        # Reconstruct and add children
        children_data = data.get("children", [])
        for child_data in children_data:
            child_node = cls.from_dict(child_data)  # Recursive call
            node.add_child(child_node)  # This will set child_node.parent = node
        return node

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the FunctionalityNode."""
        return (
            f"FunctionalityNode(name='{self.name}', desc='{self.description[:20]}...', "
            f"params={len(self.parameters)}, outputs={len(self.outputs)}, children={len(self.children)})"
        )

    def to_detailed_string(self, indent_level: int = 0) -> str:
        """Return a detailed, multi-line string representation of the node and its hierarchy."""
        indent_unit = "  "
        current_indent = indent_unit * indent_level

        parts = [self._format_node_header(current_indent)]

        if self.parameters:
            parts.extend(self._format_parameters(current_indent, indent_unit))

        if self.outputs:
            parts.extend(self._format_outputs(current_indent, indent_unit))

        if self.children:
            parts.extend(self._format_children(current_indent, indent_unit, indent_level))

        return "\n".join(parts)

    def _format_node_header(self, current_indent: str) -> str:
        """Format the node name and description header."""
        node_desc_text = ""
        if self.description:
            node_desc_text = self.description[:20].replace("\n", " ")
        desc_preview = f" (desc: '{node_desc_text}...')" if self.description else ""
        return f"{current_indent}{self.name}:{desc_preview}"

    def _format_parameters(self, current_indent: str, indent_unit: str) -> list[str]:
        """Format the parameters section."""
        parts = [f"{current_indent}{indent_unit}Parameters:"]

        for param in self.parameters:
            param_desc_text = ""
            if param.description:
                param_desc_text = param.description[:20].replace("\n", " ")
            param_desc_preview_str = f" (desc: '{param_desc_text}...')" if param.description else ""
            parts.append(f"{current_indent}{indent_unit * 2}{param.name}:{param_desc_preview_str}")

            if param.options:
                parts.extend(f"{current_indent}{indent_unit * 3}- {option}" for option in param.options)

        return parts

    def _format_outputs(self, current_indent: str, indent_unit: str) -> list[str]:
        """Format the output options section."""
        parts = [f"{current_indent}{indent_unit}Output Options:"]

        for output in self.outputs:
            output_desc_text = ""
            if output.description:
                output_desc_text = output.description[:20].replace("\n", " ")
            output_desc_preview_str = f" (desc: '{output_desc_text}...')" if output.description else ""
            parts.append(f"{current_indent}{indent_unit * 2}{output.category}:{output_desc_preview_str}")

        return parts

    def _format_children(self, current_indent: str, indent_unit: str, indent_level: int) -> list[str]:
        """Format the children section."""
        parts = [f"{current_indent}{indent_unit}Children:"]
        parts.extend(child.to_detailed_string(indent_level + 2) for child in self.children)
        return parts
