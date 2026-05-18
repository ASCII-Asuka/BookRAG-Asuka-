import logging

from Core.Index.HRIIndex import HRIIndex
from Core.Index.Tree import DocumentTree
from Core.configs.system_config import SystemConfig

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

    log.info(
        "Hydro HRI index saved with %s anchors and %s relations.",
        len(hri_index.anchors),
        len(hri_index.relations),
    )
    return hri_index
