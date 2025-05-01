"""Defines the FunctionalityNode class for representing discovered chatbot features."""

from typing import Any, Optional


class FunctionalityNode:
    """Represents a discovered chatbot functionality as a node in a tree structure.

    Attributes:
        name: The name of the functionality (e.g., "order_pizza").
        description: A brief description of what the functionality does.
        parameters: A list of parameters the functionality might take (e.g., size, toppings).
        parent: The parent node in the functionality tree, if any.
        children: A list of child nodes representing sub-functionalities or next steps.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[dict[str, Any]] | None = None,
        parent: Optional["FunctionalityNode"] = None,
        children: list["FunctionalityNode"] | None = None,
    ) -> None:
        """Initializes a FunctionalityNode instance.

        Args:
            name: The name of the functionality.
            description: A description of the functionality.
            parameters: Optional list of parameters associated with the functionality.
            parent: Optional parent node in the hierarchy.
            children: Optional list of child nodes.
        """
        self.name: str = name
        self.description: str = description
        self.parameters: list[dict[str, Any]] = parameters if parameters else []
        self.parent: FunctionalityNode | None = parent
        self.children: list[FunctionalityNode] = children if children is not None else []

    def add_child(self, child_node: "FunctionalityNode") -> None:
        """Adds a child node to this node and sets its parent."""
        child_node.parent = self
        if child_node not in self.children:
            self.children.append(child_node)

    def to_dict(self) -> dict[str, Any]:
        """Converts the node and its children to a serializable dictionary.

        Excludes the 'parent' attribute to prevent circular references during serialization.

        Returns:
            A dictionary representation of the node and its descendants.
        """
        return {
            "__type__": "FunctionalityNode",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            # Recursively convert children
            "children": [child.to_dict() for child in self.children],
            # Excluded parent to avoid circular dependencies
        }

    def __repr__(self) -> str:
        """Returns a concise string representation of the node."""
        return (
            f"FunctionalityNode(name='{self.name}', "
            f"description='{self.description[:20]}...', "
            f"children={len(self.children)})"
        )
