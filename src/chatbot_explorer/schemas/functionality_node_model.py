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

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
        return f"OutputOptions(category='{self.category}', description='{self.description}')"


class FunctionalityNode:
    """Represents a discovered chatbot functionality node in the graph."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[ParameterDefinition] | None = None,
        outputs: list[OutputOptions] | None = None,
        parent: Optional["FunctionalityNode"] = None,
        children: list["FunctionalityNode"] | None = None,
    ) -> None:
        """Initialize a FunctionalityNode.

        Args:
            name: The unique name of the functionality.
            description: A description of what the functionality does.
            parameters: Optional list of ParameterDefinition instances.
            outputs: Optional list of OutputOptions instances.
            parent: The parent node in the functionality hierarchy, if any.
            children: A list of child nodes in the functionality hierarchy, if any.
        """
        self.name: str = name
        self.description: str = description
        self.parameters: list[ParameterDefinition] = parameters if parameters else []
        self.outputs: list[OutputOptions] = outputs if outputs else []
        self.parent: FunctionalityNode | None = parent
        self.children: list[FunctionalityNode] = children if children is not None else []

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
        return {
            "__type__": "FunctionalityNode",
            "name": self.name,
            "description": self.description,
            "parameters": [param.to_dict() for param in self.parameters],
            "outputs": [output.to_dict() for output in self.outputs],
            "children": [child.to_dict() for child in self.children],
        }

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

        # Node name and description preview
        node_desc_text = ""
        if self.description:
            node_desc_text = self.description[:20].replace("\n", " ")
        desc_preview = f" (desc: '{node_desc_text}...')" if self.description else ""
        parts = [f"{current_indent}{self.name}:{desc_preview}"]

        # Parameters
        if self.parameters:
            parts.append(f"{current_indent}{indent_unit}Parameters:")
            for param in self.parameters:
                param_desc_text = ""
                if param.description:
                    param_desc_text = param.description[:20].replace("\n", " ")
                param_desc_preview_str = f" (desc: '{param_desc_text}...')" if param.description else ""
                parts.append(f"{current_indent}{indent_unit * 2}{param.name}:{param_desc_preview_str}")
                if param.options:
                    for option in param.options:
                        parts.append(f"{current_indent}{indent_unit * 3}- {option}")

        # Output Options
        if self.outputs:
            parts.append(f"{current_indent}{indent_unit}Output Options:")
            for output in self.outputs:
                output_desc_text = ""
                if output.description:
                    output_desc_text = output.description[:20].replace("\n", " ")
                output_desc_preview_str = f" (desc: '{output_desc_text}...')" if output.description else ""
                parts.append(f"{current_indent}{indent_unit * 2}{output.category}:{output_desc_preview_str}")

        # Children (recursive call)
        if self.children:
            parts.append(f"{current_indent}{indent_unit}Children:")
            for child in self.children:
                # Children are indented further relative to the current node's name
                parts.append(child.to_detailed_string(indent_level + 2))

        return "\n".join(parts)
