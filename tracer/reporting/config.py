"""Configuration classes and factory functions for graph generation."""

from dataclasses import dataclass, field
from typing import Any

import graphviz

from .constants import COMPACT_LAYOUT, LARGE_FONT_LAYOUT, LARGE_FONT_THRESHOLD, MEDIUM_FONT_THRESHOLD


@dataclass
class GraphRenderOptions:
    """Options for graph rendering."""

    fmt: str = "pdf"
    graph_font_size: int = 12
    dpi: int = 300
    compact: bool = False
    top_down: bool = False


@dataclass
class FontSizeConfig:
    """Configuration for font sizes in graph nodes.

    Attributes:
        title_font_size: Font size for node titles.
        normal_font_size: Font size for normal text.
        small_font_size: Font size for small text and metadata.
    """

    title_font_size: int
    normal_font_size: int
    small_font_size: int


@dataclass
class TruncationConfig:
    """Configuration for text truncation in graph nodes.

    Attributes:
        title_max_length: Maximum length for node titles.
        desc_max_length: Maximum length for descriptions.
        param_max_length: Maximum length for parameter names.
        output_combined_max_length: Maximum length for combined output text.
        max_params: Maximum number of parameters to show.
        max_outputs: Maximum number of outputs to show.
        max_options: Maximum number of parameter options to show.
    """

    title_max_length: int = 30
    desc_max_length: int = 50
    param_max_length: int = 25
    output_combined_max_length: int = 40
    max_params: int = 3
    max_outputs: int = 3
    max_options: int = 2


@dataclass
class GraphStyleConfig:
    """Configuration for graph visual styling.

    Attributes:
        bgcolor: Background color for the graph.
        fontcolor: Text color for nodes and labels.
        edge_color: Color for graph edges.
        font_family: Font family for text elements.
    """

    bgcolor: str = "#ffffff"
    fontcolor: str = "#333333"
    edge_color: str = "#9DB2BF"
    font_family: str = "Helvetica Neue, Helvetica, Arial, sans-serif"


@dataclass
class GraphLayoutConfig:
    """Configuration for graph layout parameters.

    Attributes:
        pad: Padding around the graph.
        nodesep: Separation between nodes at the same rank.
        ranksep: Separation between ranks.
        splines: Type of spline for edges.
        overlap: How to handle node overlaps.
        node_margin: Margin inside nodes.
    """

    pad: str = "0.7"
    nodesep: str = "0.8"
    ranksep: str = "1.3"
    splines: str = "curved"
    overlap: str = "false"
    node_margin: str = "0.2,0.15"


@dataclass
class GraphBuildContext:
    """Context for tracking graph building state.

    Attributes:
        graph: The main Graphviz Digraph object.
        processed_nodes: Set of node names that have been processed.
        processed_edges: Set of edge tuples that have been processed.
        node_clusters: Dictionary mapping category names to their subgraphs.
    """

    graph: graphviz.Digraph
    processed_nodes: set[str] = field(default_factory=set)
    processed_edges: set[tuple[str, str]] = field(default_factory=set)
    node_clusters: dict[str, graphviz.Digraph] = field(default_factory=dict)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output_dir: str
    graph_font_size: int = 12
    compact: bool = False
    top_down: bool = False
    graph_format: str = "pdf"


@dataclass
class ExecutionResults:
    """Container for execution phase results."""

    exploration_results: dict[str, Any]
    analysis_results: dict[str, Any]
    token_usage: dict[str, Any]


@dataclass
class AddNodeOptions:
    """Options for adding nodes to graph."""

    graph_font_size: int = 12
    compact: bool = False
    target_graph: graphviz.Digraph | None = None
    category_for_label: str | None = None


def create_font_size_config(graph_font_size: int, *, compact: bool) -> FontSizeConfig:
    """Create font size configuration based on graph settings.

    Args:
        graph_font_size: Base font size for the graph.
        compact: Whether to use compact layout.

    Returns:
        FontSizeConfig: Configuration object with calculated font sizes.
    """
    if compact:
        return FontSizeConfig(
            title_font_size=graph_font_size + 1,
            normal_font_size=max(graph_font_size - 1, 8),
            small_font_size=max(graph_font_size - 2, 7),
        )
    return FontSizeConfig(
        title_font_size=graph_font_size + 2,
        normal_font_size=graph_font_size,
        small_font_size=max(graph_font_size - 1, 8),
    )


def create_truncation_config(graph_font_size: int, *, compact: bool) -> TruncationConfig:
    """Create text truncation configuration based on font size and layout.

    Args:
        graph_font_size: Base font size for the graph.
        compact: Whether to use compact layout.

    Returns:
        TruncationConfig: Configuration object with truncation limits.
    """
    if graph_font_size >= LARGE_FONT_THRESHOLD:
        return TruncationConfig(
            title_max_length=25,
            desc_max_length=25,
            output_combined_max_length=30,
            max_params=2,
            max_options=2,
            max_outputs=2,
        )
    if graph_font_size >= MEDIUM_FONT_THRESHOLD:
        return TruncationConfig(
            title_max_length=30,
            desc_max_length=35,
            output_combined_max_length=40,
            max_params=3,
            max_options=3,
            max_outputs=2,
        )
    if compact:
        return TruncationConfig(
            title_max_length=40,
            desc_max_length=45,
            output_combined_max_length=50,
            max_params=3,
            max_options=3,
            max_outputs=3,
        )
    return TruncationConfig(
        title_max_length=60,
        desc_max_length=70,
        output_combined_max_length=70,
        max_params=4,
        max_options=4,
        max_outputs=3,
    )


def create_layout_config(graph_font_size: int, *, compact: bool) -> GraphLayoutConfig:
    """Create graph layout configuration based on font size and compactness.

    Args:
        graph_font_size: Base font size for the graph.
        compact: Whether to use compact layout.

    Returns:
        GraphLayoutConfig: Configuration object with layout parameters.
    """
    if graph_font_size >= LARGE_FONT_THRESHOLD:
        return GraphLayoutConfig(**LARGE_FONT_LAYOUT)
    if compact:
        return GraphLayoutConfig(**COMPACT_LAYOUT)
    return GraphLayoutConfig()  # Use defaults
