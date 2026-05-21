import logging
import os
import shutil

from Core.Index.HRIIndex import HRIIndex
from Core.Index.Tree import DocumentTree
from Core.configs.system_config import SystemConfig
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.vdb import VectorStore

log = logging.getLogger(__name__)


def build_hri_index(tree_index: DocumentTree, cfg: SystemConfig) -> HRIIndex:
    """
    Build the hydro regulation HRI index from an existing DocumentTree.

    The implementation uses deterministic, high-confidence rules for node
    typing, evidence anchors, semantic target anchors, and relation extraction.
    """
    log.info("Starting Hydro HRI index construction...")

    hri_index = HRIIndex.from_tree(tree=tree_index, save_dir=cfg.save_path)
    bm25 = hri_index.build_bm25()
    hri_index.save_bm25(bm25)
    hri_index.save_to_dir()
    _build_hri_vector_index(hri_index=hri_index, cfg=cfg)

    log.info(
        "Hydro HRI index saved with %s anchors and %s relations.",
        len(hri_index.anchors),
        len(hri_index.relations),
    )
    return hri_index


def _build_hri_vector_index(hri_index: HRIIndex, cfg: SystemConfig) -> None:
    rag_config = getattr(cfg.rag, "strategy_config", None)
    if not getattr(rag_config, "enable_vector_recall", False):
        return

    vdb_cfg = rag_config.hri_vdb_config
    vdb_dir = _resolve_vdb_path(cfg.save_path, vdb_cfg.vdb_dir_name)
    if vdb_cfg.force_rebuild and os.path.exists(vdb_dir):
        log.info("Removing existing HRI vector database at %s", vdb_dir)
        shutil.rmtree(vdb_dir)
    os.makedirs(os.path.dirname(vdb_dir), exist_ok=True)

    embed_cfg = vdb_cfg.embedding_config
    embedder = TextEmbeddingProvider(
        model_name=embed_cfg.model_name,
        backend=embed_cfg.backend,
        device=embed_cfg.device,
        max_length=embed_cfg.max_length,
        api_base=embed_cfg.api_base,
        api_key=embed_cfg.api_key,
    )
    vdb = VectorStore(
        embedding_model=embedder,
        db_path=vdb_dir,
        collection_name=vdb_cfg.collection_name,
    )
    docs = hri_index.iter_vector_documents()
    vdb.add_texts(
        texts=[doc["text"] for doc in docs],
        metadatas=[doc["metadata"] for doc in docs],
    )
    embedder.close()
    log.info("Hydro HRI vector index saved to %s with %s anchors.", vdb_dir, len(docs))


def _resolve_vdb_path(save_path: str, vdb_dir_name: str) -> str:
    if os.path.isabs(vdb_dir_name) or save_path in vdb_dir_name:
        return vdb_dir_name
    return os.path.join(save_path, vdb_dir_name)
