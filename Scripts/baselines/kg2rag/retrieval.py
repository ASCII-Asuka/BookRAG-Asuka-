from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np


def _normalized(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def expand_one_hop(
    seed_source_ids: Sequence[str],
    triples: Sequence[Mapping[str, Any]],
    hops: int = 1,
) -> dict[str, Any]:
    selected_sources = {str(source_id) for source_id in seed_source_ids}
    selected_triple_indexes = {
        index
        for index, triple in enumerate(triples)
        if str(triple.get("source_id") or "") in selected_sources
    }
    related_entities = {
        str(entity)
        for index in selected_triple_indexes
        for entity in (triples[index].get("head_key"), triples[index].get("tail_key"))
        if entity
    }
    frontier = set(related_entities)
    expansion_edges: list[dict[str, Any]] = []

    for hop in range(max(0, int(hops))):
        matched_indexes: set[int] = set()
        next_entities: set[str] = set()
        for index, triple in enumerate(triples):
            head = str(triple.get("head_key") or "")
            tail = str(triple.get("tail_key") or "")
            if head not in frontier and tail not in frontier:
                continue
            matched_indexes.add(index)
            if head:
                next_entities.add(head)
            if tail:
                next_entities.add(tail)
            source_id = str(triple.get("source_id") or "")
            if source_id:
                selected_sources.add(source_id)
            expansion_edges.append(
                {
                    "hop": hop + 1,
                    "head_key": head,
                    "tail_key": tail,
                    "source_id": source_id,
                    "triple_index": index,
                }
            )
        selected_triple_indexes.update(matched_indexes)
        next_entities.difference_update(related_entities)
        related_entities.update(next_entities)
        frontier = next_entities
        if not frontier:
            break

    return {
        "source_ids": selected_sources,
        "entity_keys": related_entities,
        "triple_indexes": selected_triple_indexes,
        "expansion_edges": expansion_edges,
    }


def _edge_records(graph: nx.MultiGraph) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for head, tail, key, data in graph.edges(keys=True, data=True):
        records.append(
            {
                "head_key": str(head),
                "tail_key": str(tail),
                "key": int(key) if isinstance(key, int) else str(key),
                "head": str(data.get("head") or head),
                "tail": str(data.get("tail") or tail),
                "relation": str(data.get("relation") or ""),
                "source_id": str(data.get("source_id") or ""),
                "weight": float(data.get("weight") or 0.0),
            }
        )
    return records


def _dfs_sources(tree: nx.MultiGraph, root: str) -> tuple[list[str], list[dict[str, Any]]]:
    ordered_sources: list[str] = []
    traversal: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    visited: set[str] = {root}

    def walk(node: str) -> None:
        candidates: list[tuple[float, str, str, Mapping[str, Any]]] = []
        for neighbor, keyed in tree.adj[node].items():
            if neighbor in visited:
                continue
            for key, data in keyed.items():
                candidates.append(
                    (
                        -float(data.get("weight") or 0.0),
                        str(neighbor),
                        str(key),
                        data,
                    )
                )
        for _, neighbor, key, data in sorted(candidates):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            source_id = str(data.get("source_id") or "")
            traversal.append(
                {
                    "from": node,
                    "to": neighbor,
                    "key": key,
                    "source_id": source_id,
                    "weight": float(data.get("weight") or 0.0),
                }
            )
            if source_id and source_id not in seen_sources:
                seen_sources.add(source_id)
                ordered_sources.append(source_id)
            walk(neighbor)

    walk(root)
    return ordered_sources, traversal


def organize_components(
    triples: Sequence[Mapping[str, Any]],
    chunk_scores: Mapping[str, float],
) -> list[dict[str, Any]]:
    graph = nx.MultiGraph()
    for index, triple in enumerate(triples):
        head = str(triple.get("head_key") or "")
        tail = str(triple.get("tail_key") or "")
        source_id = str(triple.get("source_id") or "")
        if not head or not tail or not source_id:
            continue
        graph.add_edge(
            head,
            tail,
            key=index,
            head=str(triple.get("head") or head),
            tail=str(triple.get("tail") or tail),
            relation=str(triple.get("relation") or ""),
            source_id=source_id,
            weight=float(chunk_scores.get(source_id, 0.0)),
        )

    components: list[dict[str, Any]] = []
    connected = sorted(
        nx.connected_components(graph),
        key=lambda nodes: (-len(nodes), min(str(node) for node in nodes)),
    )
    for component_index, nodes in enumerate(connected):
        subgraph = graph.subgraph(nodes).copy()
        tree = nx.maximum_spanning_tree(subgraph, weight="weight")
        edges = sorted(
            _edge_records(tree),
            key=lambda edge: (
                -edge["weight"],
                edge["source_id"],
                edge["head_key"],
                edge["tail_key"],
            ),
        )
        if edges:
            root_edge = edges[0]
            root = min(root_edge["head_key"], root_edge["tail_key"])
            ordered_sources, traversal = _dfs_sources(tree, root)
        else:
            root = min(str(node) for node in nodes)
            ordered_sources, traversal = [], []

        all_sources = sorted(
            {
                str(data.get("source_id") or "")
                for _, _, _, data in subgraph.edges(keys=True, data=True)
                if data.get("source_id")
            },
            key=lambda source_id: (-float(chunk_scores.get(source_id, 0.0)), source_id),
        )
        for source_id in all_sources:
            if source_id not in ordered_sources and not edges:
                ordered_sources.append(source_id)
        triple_text = ", ".join(
            f"<{edge['head']}, {edge['relation']}, {edge['tail']}>" for edge in edges
        )
        components.append(
            {
                "component_id": component_index,
                "entity_keys": sorted(str(node) for node in nodes),
                "all_source_ids": all_sources,
                "mst_edges": edges,
                "root_entity": root,
                "dfs_traversal": traversal,
                "ordered_source_ids": ordered_sources,
                "triple_text": triple_text,
                "max_semantic_score": max(
                    (float(chunk_scores.get(source_id, 0.0)) for source_id in all_sources),
                    default=0.0,
                ),
            }
        )
    return components


def _add_usage(total: dict[str, int], usage: Mapping[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] += int(usage.get(key) or 0)


def _embedding_matrix(index: Mapping[str, Any]) -> np.ndarray:
    if "embedding_matrix" in index:
        return np.asarray(index["embedding_matrix"], dtype=np.float32)
    path = index.get("embedding_path")
    if path:
        return np.load(Path(str(path)), allow_pickle=False)
    raise ValueError("scope index has no embedding_matrix or embedding_path")


def retrieve(
    question: str,
    index: Mapping[str, Any],
    clients: Any,
    *,
    embedding_model: str,
    reranker_model: str,
    seed_topk: int = 10,
    retrieval_topk: int = 20,
    expansion_hops: int = 1,
) -> dict[str, Any]:
    started = time.perf_counter()
    chunks = list(index.get("chunks") or [])
    if not chunks:
        raise ValueError("scope index contains no chunks")
    by_source: dict[str, Mapping[str, Any]] = {}
    source_order: dict[str, int] = {}
    for position, chunk in enumerate(chunks):
        source_id = str(chunk.get("source_id") or "")
        if not source_id or source_id in by_source:
            raise ValueError(f"duplicate or empty source_id: {source_id!r}")
        by_source[source_id] = chunk
        source_order[source_id] = position

    matrix = _embedding_matrix(index)
    if matrix.shape[0] != len(chunks):
        raise ValueError("embedding matrix row count does not match chunks")
    query_matrix, embedding_usage = clients.embed(embedding_model, [question])
    if query_matrix.shape[0] != 1 or query_matrix.shape[1] != matrix.shape[1]:
        raise ValueError("query embedding shape does not match chunk embeddings")
    similarities = (_normalized(matrix) @ _normalized(query_matrix)[0]).tolist()
    chunk_scores = {
        str(chunk["source_id"]): float(similarities[position])
        for position, chunk in enumerate(chunks)
    }
    semantic_order = sorted(
        by_source,
        key=lambda source_id: (-chunk_scores[source_id], source_order[source_id]),
    )
    seed_ids = semantic_order[: max(0, int(seed_topk))]

    triples = list(index.get("triples") or [])
    expansion = expand_one_hop(seed_ids, triples, hops=expansion_hops)
    expanded_ids = set(expansion["source_ids"])
    expanded_triples = [
        triple
        for triple in triples
        if str(triple.get("source_id") or "") in expanded_ids
    ]
    components = organize_components(expanded_triples, chunk_scores)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    _add_usage(usage_total, embedding_usage)

    component_scores: list[float] = []
    if components:
        documents = [
            str(component.get("triple_text") or " ".join(component["entity_keys"]))
            for component in components
        ]
        component_scores, reranker_usage = clients.rerank(
            reranker_model, question, documents
        )
        if len(component_scores) != len(components):
            raise ValueError("component reranker score count mismatch")
        _add_usage(usage_total, reranker_usage)
    for position, component in enumerate(components):
        component["reranker_score"] = float(component_scores[position])

    sorted_components = sorted(
        components,
        key=lambda component: (
            -float(component.get("reranker_score") or 0.0),
            -float(component.get("max_semantic_score") or 0.0),
            int(component["component_id"]),
        ),
    )
    ordered_ids: list[str] = []
    route: dict[str, str] = {}
    component_score_by_source: dict[str, float] = {}
    for component in sorted_components:
        for source_id in component["ordered_source_ids"]:
            if source_id in by_source and source_id not in ordered_ids:
                ordered_ids.append(source_id)
                route[source_id] = "kg_organized"
                component_score_by_source[source_id] = float(
                    component.get("reranker_score") or 0.0
                )

    for source_id in sorted(
        expanded_ids,
        key=lambda item: (-chunk_scores.get(item, 0.0), source_order.get(item, 10**9)),
    ):
        if source_id in by_source and source_id not in ordered_ids:
            ordered_ids.append(source_id)
            route[source_id] = "kg_expanded"

    for source_id in semantic_order:
        if source_id not in ordered_ids:
            ordered_ids.append(source_id)
            route[source_id] = "semantic_tail"
        if len(ordered_ids) >= retrieval_topk:
            break
    ordered_ids = ordered_ids[: max(0, int(retrieval_topk))]

    ranked_results: list[dict[str, Any]] = []
    for rank, source_id in enumerate(ordered_ids, start=1):
        chunk = by_source[source_id]
        ranked_results.append(
            {
                "rank": rank,
                "score": chunk_scores[source_id],
                "semantic_score": chunk_scores[source_id],
                "component_score": component_score_by_source.get(source_id),
                "retrieval_route": route[source_id],
                "source_id": source_id,
                "content": str(chunk.get("content") or ""),
                "metadata": dict(chunk.get("metadata") or {}),
            }
        )

    elapsed = time.perf_counter() - started
    return {
        "question": question,
        "semantic_seeds": [
            {"source_id": source_id, "score": chunk_scores[source_id]}
            for source_id in seed_ids
        ],
        "expanded_source_ids": sorted(
            expanded_ids, key=lambda item: source_order.get(item, 10**9)
        ),
        "expansion_edges": expansion["expansion_edges"],
        "components": sorted_components,
        "ranked_results": ranked_results,
        "retrieval_time": elapsed,
        "retrieval_token_cost": {**usage_total, "time": elapsed},
        "dense_fallback": not bool(components),
        "dense_fallback_reason": "no_graph_component" if not components else None,
    }
