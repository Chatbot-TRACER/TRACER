from typing import Any, Dict, List, Optional


class FunctionalityNode:
    """Represents a discovered chatbot functionality."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[List[Dict[str, Any]]] = None,
        parent: Optional["FunctionalityNode"] = None,
        children: Optional[List["FunctionalityNode"]] = None,
    ):
        self.name: str = name
        self.description: str = description
        self.parameters: List[Dict[str, Any]] = parameters if parameters else []
        self.parent: Optional["FunctionalityNode"] = parent
        self.children: List["FunctionalityNode"] = children if children is not None else []

    def add_child(self, child_node: "FunctionalityNode"):
        """Adds a child node to this node."""
        child_node.parent = self
        if child_node not in self.children:
            self.children.append(child_node)

    def to_dict(self) -> Dict[str, Any]:
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
