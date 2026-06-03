import logging
import os
import shutil

from Core.Index.EvidenceBridgeIndex import EvidenceBridgeIndex
from Core.Index.Tree import DocumentTree
from Core.configs.system_config import SystemConfig

log = logging.getLogger(__name__)


def build_evibridge_index(tree_index: DocumentTree, cfg: SystemConfig) -> EvidenceBridgeIndex:
    log.info("Starting EviBridge index construction...")
    evibridge_index = EvidenceBridgeIndex.from_tree(tree=tree_index, save_dir=cfg.save_path)
    bm25 = evibridge_index.build_bm25()
    evibridge_index.save_bm25(bm25)
    evibridge_index.save_to_dir()
    _build_evibridge_vector_index(evibridge_index=evibridge_index, cfg=cfg)
    log.info(
        "EviBridge index saved with %s blocks and %s bridges.",
        len(evibridge_index.blocks),
        len(evibridge_index.bridges),
    )
    return evibridge_index


def _build_evibridge_vector_index(evibridge_index: EvidenceBridgeIndex, cfg: SystemConfig) -> None:
    rag_config = getattr(cfg.rag, "strategy_config", None)
    if not getattr(rag_config, "enable_vector_recall", False):
        return

    vdb_cfg = rag_config.evibridge_vdb_config
    vdb_dir = _resolve_vdb_path(cfg.save_path, vdb_cfg.vdb_dir_name)
    if vdb_cfg.force_rebuild and os.path.exists(vdb_dir):
        log.info("Removing existing EviBridge vector database at %s", vdb_dir)
        shutil.rmtree(vdb_dir)
    os.makedirs(os.path.dirname(vdb_dir), exist_ok=True)

    from Core.provider.embedding import TextEmbeddingProvider
    from Core.provider.vdb import VectorStore

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
    docs = evibridge_index.iter_vector_documents()
    vdb.add_texts(
        texts=[doc["text"] for doc in docs],
        metadatas=[doc["metadata"] for doc in docs],
    )
    embedder.close()
    log.info("EviBridge vector index saved to %s with %s blocks.", vdb_dir, len(docs))


def _resolve_vdb_path(save_path: str, vdb_dir_name: str) -> str:
    if os.path.isabs(vdb_dir_name) or save_path in vdb_dir_name:
        return vdb_dir_name
    return os.path.join(save_path, vdb_dir_name)
