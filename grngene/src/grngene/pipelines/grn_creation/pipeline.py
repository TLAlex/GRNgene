"""
This is a boilerplate pipeline 'grn_creation'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_custom_barabasi_albert_graph,
    add_edge_direction,
    add_edge_influence_type,
    nxgraph_to_df,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_custom_barabasi_albert_graph,
                inputs="params:grn_options",
                outputs="undirected_graph",
                name="create_graph_node",
            ),
            node(
                func=add_edge_direction,
                inputs=["undirected_graph", "params:grn_options"],
                outputs="directed_graph",
                name="add_edge_direction_node",
            ),
            node(
                func=add_edge_influence_type,
                inputs=["directed_graph", "params:grn_options"],
                outputs="directed_graph_with_influence_type",
                name="add_edge_influence_node",
            ),
            node(
                func=nxgraph_to_df,
                inputs="directed_graph_with_influence_type",
                outputs="random_graph_edge_list",
                name="graph_to_df_node",
            ),
        ]
    )
