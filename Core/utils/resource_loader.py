from typing import Dict, Any
from Core.configs.system_config import SystemConfig
import logging


log = logging.getLogger(__name__)


def prepare_rag_dependencies(cfg: SystemConfig) -> Dict[str, Any]:
    """
    根据配置加载并准备RAG agent所需的依赖项。
    这是一个调度函数，它知道哪种策略需要哪种资源。
    """

    rag_config = cfg.rag.strategy_config
    strategy_name = rag_config.strategy
    log.info(f"Preparing dependencies for RAG strategy: '{strategy_name}'")

    dependencies = {}

    if strategy_name == "traverse":
        from Core.Index.Tree import DocumentTree

        # 加载 TraverseAgent 需要的 tree_index
        tree_index_path = DocumentTree.get_save_path(cfg.save_path)
        tree_index = DocumentTree.load_from_file(tree_index_path)
        log.info(f"Successfully loaded tree index from {tree_index_path}")
        dependencies["tree_index"] = tree_index

    elif strategy_name == "hri":
        import os
        from Core.Index.Tree import DocumentTree
        from Core.Index.HRIIndex import HRIIndex

        tree_index_path = DocumentTree.get_save_path(cfg.save_path)
        tree_index = DocumentTree.load_from_file(tree_index_path)
        hri_index = HRIIndex.load_from_dir(cfg.save_path)
        bm25 = HRIIndex.load_bm25(cfg.save_path)
        log.info(f"Successfully loaded tree index from {tree_index_path}")
        log.info(f"Successfully loaded HRI index from {HRIIndex.get_index_path(cfg.save_path)}")
        log.info(f"Successfully loaded HRI BM25 from {HRIIndex.get_bm25_path(cfg.save_path)}")
        dependencies["tree_index"] = tree_index
        dependencies["hri_index"] = hri_index
        dependencies["bm25"] = bm25
        if getattr(rag_config, "enable_vector_recall", False):
            from Core.provider.embedding import TextEmbeddingProvider
            from Core.provider.vdb import VectorStore

            vdb_cfg = rag_config.hri_vdb_config
            embed_cfg = vdb_cfg.embedding_config
            hri_vdb_path = vdb_cfg.vdb_dir_name
            if not os.path.isabs(hri_vdb_path) and cfg.save_path not in hri_vdb_path:
                hri_vdb_path = os.path.join(cfg.save_path, hri_vdb_path)
            embed_model = TextEmbeddingProvider(
                model_name=embed_cfg.model_name,
                backend=embed_cfg.backend,
                device=embed_cfg.device,
                max_length=embed_cfg.max_length,
                api_base=embed_cfg.api_base,
                api_key=embed_cfg.api_key,
            )
            hri_vector_store = VectorStore(
                embedding_model=embed_model,
                db_path=hri_vdb_path,
                collection_name=vdb_cfg.collection_name,
            )
            log.info(f"Successfully loaded HRI vector store from {hri_vdb_path}")
            dependencies["hri_vector_store"] = hri_vector_store

    elif strategy_name == "evibridge":
        import os
        from Core.Index.EvidenceBridgeIndex import EvidenceBridgeIndex

        evibridge_index = EvidenceBridgeIndex.load_from_dir(cfg.save_path)
        bm25 = EvidenceBridgeIndex.load_bm25(cfg.save_path)
        log.info(
            f"Successfully loaded EviBridge index from {EvidenceBridgeIndex.get_index_path(cfg.save_path)}"
        )
        log.info(
            f"Successfully loaded EviBridge BM25 from {EvidenceBridgeIndex.get_bm25_path(cfg.save_path)}"
        )
        dependencies["evibridge_index"] = evibridge_index
        dependencies["bm25"] = bm25
        if getattr(rag_config, "enable_vector_recall", False):
            from Core.provider.embedding import TextEmbeddingProvider
            from Core.provider.vdb import VectorStore

            vdb_cfg = rag_config.evibridge_vdb_config
            embed_cfg = vdb_cfg.embedding_config
            evibridge_vdb_path = vdb_cfg.vdb_dir_name
            if not os.path.isabs(evibridge_vdb_path) and cfg.save_path not in evibridge_vdb_path:
                evibridge_vdb_path = os.path.join(cfg.save_path, evibridge_vdb_path)
            embed_model = TextEmbeddingProvider(
                model_name=embed_cfg.model_name,
                backend=embed_cfg.backend,
                device=embed_cfg.device,
                max_length=embed_cfg.max_length,
                api_base=embed_cfg.api_base,
                api_key=embed_cfg.api_key,
            )
            evibridge_vector_store = VectorStore(
                embedding_model=embed_model,
                db_path=evibridge_vdb_path,
                collection_name=vdb_cfg.collection_name,
            )
            log.info(f"Successfully loaded EviBridge vector store from {evibridge_vdb_path}")
            dependencies["evibridge_vector_store"] = evibridge_vector_store
        if getattr(rag_config, "enable_candidate_rerank", False) or getattr(
            rag_config,
            "enable_supporting_rerank",
            False,
        ) or getattr(
            rag_config,
            "enable_answer_conditioned_support_rerank",
            False,
        ):
            from Core.provider.rerank import TextRerankerProvider

            reranker_cfg = rag_config.reranker_config
            reranker = TextRerankerProvider(
                model_name=reranker_cfg.model_name,
                device=reranker_cfg.device,
                max_length=reranker_cfg.max_length,
                backend=reranker_cfg.backend,
                api_base=reranker_cfg.api_base,
                api_key=reranker_cfg.api_key,
                max_retries=getattr(reranker_cfg, "max_retries", 3),
                retry_backoff=getattr(reranker_cfg, "retry_backoff", 0.5),
                request_timeout=getattr(reranker_cfg, "request_timeout", 60.0),
            )
            log.info(f"Successfully loaded EviBridge reranker: {reranker_cfg.model_name}")
            dependencies["reranker"] = reranker

    elif strategy_name == "lightrag":
        from Core.Index.Tree import DocumentTree

        tree_index_path = DocumentTree.get_save_path(cfg.save_path)
        tree_index = DocumentTree.load_from_file(tree_index_path)
        log.info(f"Successfully loaded tree index from {tree_index_path}")
        dependencies["tree_index"] = tree_index
        dependencies["save_path"] = cfg.save_path

    elif strategy_name == "hipporag":
        from Core.Index.Tree import DocumentTree

        tree_index_path = DocumentTree.get_save_path(cfg.save_path)
        tree_index = DocumentTree.load_from_file(tree_index_path)
        log.info(f"Successfully loaded tree index from {tree_index_path}")
        dependencies["tree_index"] = tree_index
        dependencies["save_path"] = cfg.save_path

    elif strategy_name == "gbc":
        from Core.Index.GBCIndex import GBC

        gbc_index = GBC.load_gbc_index(cfg)
        log.info(f"Successfully loaded GBC index from {cfg.save_path}")
        dependencies["gbc_index"] = gbc_index
    elif strategy_name == "graph":
        from Core.Index.GBCIndex import GBC

        gbc_index = GBC.load_gbc_index(cfg)
        log.info(f"Successfully loaded GBC index from {cfg.save_path}")
        dependencies["gbc_index"] = gbc_index

    elif strategy_name == "vanilla":
        import os
        from Core.Index.Tree import DocumentTree
        from Core.configs.vdb_config import VDBConfig

        retrieval_method = rag_config.retrieval_method

        def _resolve_store_path(path: str) -> str:
            resolved = path
            if not os.path.isabs(resolved) and cfg.save_path not in resolved:
                resolved = os.path.join(cfg.save_path, resolved)
            return resolved

        def _load_bm25(vdb_dir_name: str):
            from Core.utils.bm25 import BM25

            bm25_dir = _resolve_store_path(vdb_dir_name)
            bm25_path = os.path.join(bm25_dir, "bm25_index.pkl")
            bm25 = BM25.load(bm25_path)
            log.info(f"Successfully loaded BM25 index from {bm25_path}")
            return bm25

        def _load_vector_store(vdb_cfg: VDBConfig):
            from Core.provider.embedding import TextEmbeddingProvider
            from Core.provider.vdb import VectorStore

            vdb_store_path = _resolve_store_path(vdb_cfg.vdb_dir_name)
            embed_cfg = vdb_cfg.embedding_config
            embed_model = TextEmbeddingProvider(
                model_name=embed_cfg.model_name,
                backend=embed_cfg.backend,
                device=embed_cfg.device,
                max_length=embed_cfg.max_length,
                api_base=embed_cfg.api_base,
                api_key=embed_cfg.api_key,
            )
            vdb = VectorStore(
                embedding_model=embed_model,
                db_path=vdb_store_path,
                collection_name=vdb_cfg.collection_name,
            )
            log.info(f"Successfully loaded vector store from {vdb_store_path}")
            return vdb

        if retrieval_method == "bm25":
            dependencies["bm25"] = _load_bm25(rag_config.vdb_config.vdb_dir_name)
        elif retrieval_method == "hybrid":
            dependencies["bm25"] = _load_bm25(rag_config.bm25_vdb_dir_name)
            dependencies["vector_store"] = _load_vector_store(rag_config.vdb_config)
        elif retrieval_method == "bm25_rerank":
            from Core.provider.rerank import TextRerankerProvider

            dependencies["bm25"] = _load_bm25(rag_config.bm25_vdb_dir_name)
            reranker_cfg = rag_config.reranker_config
            reranker = TextRerankerProvider(
                model_name=reranker_cfg.model_name,
                device=reranker_cfg.device,
                max_length=reranker_cfg.max_length,
                backend=reranker_cfg.backend,
                api_base=reranker_cfg.api_base,
                api_key=reranker_cfg.api_key,
                max_retries=getattr(reranker_cfg, "max_retries", 3),
                retry_backoff=getattr(reranker_cfg, "retry_backoff", 0.5),
                request_timeout=getattr(reranker_cfg, "request_timeout", 60.0),
            )
            log.info(f"Successfully loaded reranker: {reranker_cfg.model_name}")
            dependencies["reranker"] = reranker
        elif retrieval_method in {"ircot", "react"}:
            dependencies["bm25"] = _load_bm25(rag_config.bm25_vdb_dir_name)
        elif retrieval_method in {
            "abstract_only",
            "lead_only",
            "full_document",
            "longrag",
        }:
            tree_index_path = DocumentTree.get_save_path(cfg.save_path)
            tree_index = DocumentTree.load_from_file(tree_index_path)
            log.info(f"Successfully loaded tree index from {tree_index_path}")
            dependencies["tree_index"] = tree_index
        else:
            dependencies["vector_store"] = _load_vector_store(rag_config.vdb_config)

    elif strategy_name == "gbcvanilla":
        import os
        from Core.configs.embedding_config import EmbeddingConfig
        from Core.configs.vdb_config import VDBConfig
        from Core.provider.vdb import VectorStore
        from Core.provider.embedding import TextEmbeddingProvider

        embed_cfg: EmbeddingConfig = rag_config.tree_vdb_config.embedding_config
        
        tree_vdb_cfg: VDBConfig = rag_config.tree_vdb_config
        tree_vdb_store_path = tree_vdb_cfg.vdb_dir_name
        if cfg.save_path not in tree_vdb_store_path:
            tree_vdb_store_path = os.path.join(cfg.save_path, tree_vdb_store_path)
            
        graph_vdb_cfg: VDBConfig = rag_config.graph_vdb_config
        graph_vdb_store_path = graph_vdb_cfg.vdb_dir_name
        if cfg.save_path not in graph_vdb_store_path:
            graph_vdb_store_path = os.path.join(cfg.save_path, graph_vdb_store_path)

        embed_model = TextEmbeddingProvider(
            model_name=embed_cfg.model_name,
            backend=embed_cfg.backend,
            device=embed_cfg.device,
            max_length=embed_cfg.max_length,
            api_base=embed_cfg.api_base,
            api_key=embed_cfg.api_key,
        )
        
        tree_vdb = VectorStore(
            embedding_model=embed_model,
            db_path=tree_vdb_store_path,
            collection_name=tree_vdb_cfg.collection_name,
        )
        log.info(f"Successfully loaded tree VDB from {tree_vdb_store_path}")
        
        graph_vdb = VectorStore(
            embedding_model=embed_model,
            db_path=graph_vdb_store_path,
            collection_name=graph_vdb_cfg.collection_name,
        )
        log.info(f"Successfully loaded graph VDB from {graph_vdb_store_path}")
        dependencies["tree_vdb"] = tree_vdb
        dependencies["graph_vdb"] = graph_vdb

    elif strategy_name == "mmr":
        from Core.configs.embedding_config import EmbeddingConfig
        from Core.provider.vdb import VectorStore

        embed_cfg: EmbeddingConfig = rag_config.vdb_config.embedding_config
        embed_model_type = embed_cfg.type
        if embed_model_type == "text":
            from Core.provider.embedding import TextEmbeddingProvider

            embed_model = TextEmbeddingProvider(
                model_name=embed_cfg.model_name,
                backend=embed_cfg.backend,
                device=embed_cfg.device,
                max_length=embed_cfg.max_length,
                api_base=embed_cfg.api_base,
                api_key=embed_cfg.api_key,
            )
        elif embed_model_type == "gme":
            from Core.provider.embedding import GmeEmbeddingProvider

            embed_model = GmeEmbeddingProvider(
                model_name=embed_cfg.model_name,
                device=embed_cfg.device,
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {embed_model_type}")

        import os
        from Core.configs.vdb_config import VDBConfig

        vdb_cfg: VDBConfig = rag_config.vdb_config
        vdb_store_path = vdb_cfg.vdb_dir_name
        if cfg.save_path not in vdb_store_path:
            vdb_store_path = os.path.join(cfg.save_path, vdb_store_path)

        vdb = VectorStore(
            embedding_model=embed_model,
            db_path=vdb_store_path,
            collection_name=vdb_cfg.collection_name,
        )
        log.info(f"Successfully loaded vector store from {vdb_store_path}")
        dependencies["vector_store"] = vdb
    else:
        raise ValueError(f"Unknown or unsupported RAG strategy: '{strategy_name}'")

    return dependencies
