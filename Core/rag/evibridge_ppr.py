#evibridge RAG 策略中的 图计算引擎（Graph Computing Engine）。
# 它的核心作用是：基于前期构建好的树-图混合索引（ EvidenceBridgeIndex ），运行 带权重的个性化 PageRank（Personalized PageRank, PPR）算法 和 最短路径算法 ，从而在复杂的网络中找出与用户问题最相关的证据节点。
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
from pydantic import BaseModel, Field

from Core.Index.EvidenceBridgeIndex import EvidenceBridge, EvidenceBridgeIndex
from Core.rag.evibridge_demand import EvidenceDemand, weights_for_demand


class TypedPPRResult(BaseModel):
    scores: Dict[int, float] = Field(default_factory=dict)
    score_parts: Dict[int, Dict[str, float]] = Field(default_factory=dict)


class ConnectorResult(BaseModel):
    scores: Dict[int, float] = Field(default_factory=dict)
    paths: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


def build_weighted_transition_graph(
    index: EvidenceBridgeIndex,
    demand: EvidenceDemand,
    bridge_types: Optional[Iterable[str]] = None,
    use_typed_weights: bool = True,
    typed_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> nx.DiGraph:
    allowed = set(bridge_types) if bridge_types else None
    type_weights = weights_for_demand(
        demand,
        typed_weights=typed_weights,
        use_typed_weights=use_typed_weights,
    )
    graph = nx.DiGraph()
    for block_id, block in index.blocks.items():
        graph.add_node(block_id, **block.model_dump())
    for bridge in index.bridges:
        if allowed and bridge.bridge_type not in allowed:
            continue
        weight = bridge.weight * type_weights.get(bridge.bridge_type, 0.0)
        if weight <= 0:
            continue
        if graph.has_edge(bridge.source_id, bridge.target_id):
            data = graph[bridge.source_id][bridge.target_id]
            data["weight"] = data.get("weight", 0.0) + weight
            data.setdefault("bridge_types", set()).add(bridge.bridge_type)
            data.setdefault("relation_types", set()).add(bridge.relation_type)
            if bridge.evidence:
                data.setdefault("evidence_items", []).append(bridge.evidence)
        else:
            graph.add_edge(
                bridge.source_id,
                bridge.target_id,
                weight=weight,
                bridge_type=bridge.bridge_type,
                bridge_types={bridge.bridge_type},
                relation_type=bridge.relation_type,
                relation_types={bridge.relation_type},
                evidence=bridge.evidence,
                evidence_items=[bridge.evidence] if bridge.evidence else [],
            )
    for _, _, data in graph.edges(data=True):
        if isinstance(data.get("bridge_types"), set):
            data["bridge_types"] = sorted(data["bridge_types"])
        if isinstance(data.get("relation_types"), set):
            data["relation_types"] = sorted(data["relation_types"])
    return graph


def _pagerank_scores(
    graph: nx.DiGraph,
    seed_scores: Dict[int, float],
    restart_alpha: float,
    max_iter: int,
) -> Dict[int, float]:
    if not graph.nodes:
        return {}
    valid_seed_scores = {
        block_id: max(float(score), 0.0)
        for block_id, score in seed_scores.items()
        if block_id in graph
    }
    if not valid_seed_scores:
        valid_seed_scores = {block_id: 1.0 for block_id in graph.nodes}
    total = sum(valid_seed_scores.values()) or 1.0
    personalization = {
        block_id: valid_seed_scores.get(block_id, 0.0) / total
        for block_id in graph.nodes
    }
    damping = max(0.0, min(1.0, 1.0 - restart_alpha))
    try:
        scores = nx.pagerank(
            graph,
            alpha=damping,
            personalization=personalization,
            weight="weight",
            max_iter=max_iter,
            tol=1.0e-6,
        )
    except nx.PowerIterationFailedConvergence:
        try:
            scores = nx.pagerank(
                graph,
                alpha=damping,
                personalization=personalization,
                weight="weight",
                max_iter=max(max_iter * 5, 200),
                tol=1.0e-5,
            )
        except nx.PowerIterationFailedConvergence:
            scores = personalization

    for block_id, seed_score in valid_seed_scores.items():
        scores[block_id] = scores.get(block_id, 0.0) + 0.15 * (seed_score / total)
    return scores


def run_typed_ppr(
    index: EvidenceBridgeIndex,
    seed_scores: Dict[int, float],
    demand: EvidenceDemand,
    top_k: int = 20,
    restart_alpha: float = 0.15,
    max_iter: int = 50,
    bridge_types: Optional[Iterable[str]] = None,
    use_typed_weights: bool = True,
    typed_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[int, float]:
    return run_typed_ppr_with_details(
        index=index,
        seed_scores=seed_scores,
        demand=demand,
        top_k=top_k,
        restart_alpha=restart_alpha,
        max_iter=max_iter,
        bridge_types=bridge_types,
        use_typed_weights=use_typed_weights,
        typed_weights=typed_weights,
    ).scores


def run_typed_ppr_with_details(
    index: EvidenceBridgeIndex,
    seed_scores: Dict[int, float],
    demand: EvidenceDemand,
    top_k: int = 20,
    restart_alpha: float = 0.15,
    max_iter: int = 50,
    bridge_types: Optional[Iterable[str]] = None,
    use_typed_weights: bool = True,
    typed_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> TypedPPRResult:
    enabled_types = list(bridge_types) if bridge_types else ["context", "semantic", "hierarchy"]
    graph = build_weighted_transition_graph(
        index=index,
        demand=demand,
        bridge_types=enabled_types,
        use_typed_weights=use_typed_weights,
        typed_weights=typed_weights,
    )
    scores = _pagerank_scores(
        graph=graph,
        seed_scores=seed_scores,
        restart_alpha=restart_alpha,
        max_iter=max_iter,
    )
    top_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])

    score_parts: Dict[int, Dict[str, float]] = {
        block_id: {bridge_type: 0.0 for bridge_type in ["context", "semantic", "hierarchy"]}
        for block_id in top_scores
    }
    for bridge_type in ["context", "semantic", "hierarchy"]:
        if bridge_type not in enabled_types:
            continue
        type_graph = build_weighted_transition_graph(
            index=index,
            demand=demand,
            bridge_types=[bridge_type],
            use_typed_weights=use_typed_weights,
            typed_weights=typed_weights,
        )
        type_scores = _pagerank_scores(
            graph=type_graph,
            seed_scores=seed_scores,
            restart_alpha=restart_alpha,
            max_iter=max_iter,
        )
        for block_id in top_scores:
            score_parts[block_id][bridge_type] = round(float(type_scores.get(block_id, 0.0)), 8)

    return TypedPPRResult(
        scores={block_id: round(float(score), 8) for block_id, score in top_scores.items()},
        score_parts=score_parts,
    )


def _bridge_records_for_pair(index: EvidenceBridgeIndex, left_id: int, right_id: int) -> List[EvidenceBridge]:
    records = []
    for bridge in index.bridges:
        if (bridge.source_id == left_id and bridge.target_id == right_id) or (
            bridge.source_id == right_id and bridge.target_id == left_id
        ):
            records.append(bridge)
    return records


def shortest_path_connector(
    index: EvidenceBridgeIndex,
    seed_ids: Iterable[int],
    candidate_ids: Iterable[int],
    max_paths: int = 3,
    bridge_types: Optional[Iterable[str]] = None,
) -> Dict[int, float]:
    return shortest_path_connector_with_paths(
        index=index,
        seed_ids=seed_ids,
        candidate_ids=candidate_ids,
        max_paths=max_paths,
        bridge_types=bridge_types,
    ).scores


def shortest_path_connector_with_paths(
    index: EvidenceBridgeIndex,
    seed_ids: Iterable[int],
    candidate_ids: Iterable[int],
    max_paths: int = 3,
    bridge_types: Optional[Iterable[str]] = None,
) -> ConnectorResult:
    graph = index.to_typed_graph(bridge_types=bridge_types).to_undirected()
    seeds = [block_id for block_id in seed_ids if block_id in graph]
    targets = [block_id for block_id in candidate_ids if block_id in graph and block_id not in seeds]
    connector_scores: Dict[int, float] = {}
    path_records: List[Dict[str, Any]] = []
    edge_records: List[Dict[str, Any]] = []
    paths_added = 0
    for source_id in seeds:
        for target_id in targets:
            if paths_added >= max_paths:
                return ConnectorResult(scores=connector_scores, paths=path_records, edges=edge_records)
            try:
                path = nx.shortest_path(graph, source=source_id, target=target_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(path) <= 2:
                continue
            paths_added += 1
            path_edges: List[Dict[str, Any]] = []
            for left_id, right_id in zip(path, path[1:]):
                bridges = _bridge_records_for_pair(index, left_id, right_id)
                if bridge_types:
                    allowed = set(bridge_types)
                    bridges = [bridge for bridge in bridges if bridge.bridge_type in allowed]
                if not bridges:
                    continue
                bridge = bridges[0]
                edge_record = {
                    "source_id": bridge.source_id,
                    "target_id": bridge.target_id,
                    "bridge_type": bridge.bridge_type,
                    "relation_type": bridge.relation_type,
                    "weight": bridge.weight,
                    "evidence": bridge.evidence,
                }
                edge_records.append(edge_record)
                path_edges.append(edge_record)
            path_records.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "nodes": path,
                    "edges": path_edges,
                    "length": len(path) - 1,
                }
            )
            for depth, block_id in enumerate(path[1:-1], start=1):
                connector_scores[block_id] = max(
                    connector_scores.get(block_id, 0.0),
                    1.0 / (depth + 1),
                )
    return ConnectorResult(scores=connector_scores, paths=path_records, edges=edge_records)
