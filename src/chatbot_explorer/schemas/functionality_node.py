from typing import Any, Optional


class FunctionalityNode:
    """Represents a discovered chatbot functionality."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[dict[str, Any]] | None = None,
        parent: Optional["FunctionalityNode"] = None,
        children: list["FunctionalityNode"] | None = None,
    ):
        self.name: str = name
        self.description: str = description
        self.parameters: list[dict[str, Any]] = parameters if parameters else []
        self.parent: FunctionalityNode | None = parent
        self.children: list[FunctionalityNode] = children if children is not None else []

    def add_child(self, child_node: "FunctionalityNode"):
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
            "parameters": self.parameters,
            # Recursively convert children
            "children": [child.to_dict() for child in self.children],
            # Excluded parent to avoid circular dependencies
        }

    def __repr__(self) -> str:
        return (
            f"FunctionalityNode(name='{self.name}', "
            f"description='{self.description[:20]}...', "
            f"children={len(self.children)})"
        )
