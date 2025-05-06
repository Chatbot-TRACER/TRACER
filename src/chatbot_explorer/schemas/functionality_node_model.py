"""Defines the model for representing a discovered chatbot functionality."""

from typing import Any, Optional


class ParameterDefinition:
    """Model representing a parameter with its metadata."""

    def __init__(self, name: str, description: str, options: list[str]) -> None:
        """Initialize one of the parameters of a Functionality Node

        Args:
            name: Name of the parameter
            description: Description of the parameter/input
            options: What are the available options for the parameter if any (e.g. Small, Medium, Large)
        """
        self.name = name
        self.description = description
        self.options = options

    def __repr__(self) -> str:
        opts = f", options={self.options}" if self.options else ""
        return f"ParameterDefinition(name='{self.name}', type='{self.type}', description='{self.description}'{opts})"


class FunctionalityNode:
    """Represents a discovered chatbot functionality node in the graph."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[ParameterDefinition] | None = None,
        parent: Optional["FunctionalityNode"] = None,
        children: list["FunctionalityNode"] | None = None,
    ) -> None:
        """Initialize a FunctionalityNode.

        Args:
            name: The unique name of the functionality.
            description: A description of what the functionality does.
            parameters: Optional list of ParameterDefinition instances.
            parent: The parent node in the functionality hierarchy, if any.
            children: A list of child nodes in the functionality hierarchy, if any.
        """
        self.name: str = name
        self.description: str = description
        self.parameters: list[ParameterDefinition] = parameters if parameters else []
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
            "parameters": [vars(p) for p in self.parameters],
            "children": [child.to_dict() for child in self.children],
        }

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the FunctionalityNode."""
        return (
            f"FunctionalityNode(name='{self.name}', desc='{self.description[:20]}...', children={len(self.children)})"
        )
