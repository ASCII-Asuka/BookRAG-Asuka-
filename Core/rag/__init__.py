from typing import Any, Union

from Core.configs.llm_config import LLMConfig
from Core.configs.rag import ALL_STRATEGY_CONFIGS
from Core.configs.rag.evibridge_config import EviBridgeRAGConfig
from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.configs.rag.gbc_vanilla_config import GBCVanillaConfig
from Core.configs.rag.graph_config import GraphRAGConfig
from Core.configs.rag.hri_config import HRIRAGConfig
from Core.configs.rag.hipporag_config import HippoRAGConfig
from Core.configs.rag.lightrag_config import LightRAGConfig
from Core.configs.rag.mm_config import MMConfig
from Core.configs.rag.traverse_config import TraverseRAGConfig
from Core.configs.rag.vanilla_config import VanillaConfig
from Core.configs.vlm_config import VLMConfig

StrategyConfig = Union[*ALL_STRATEGY_CONFIGS]
LightRAGRAG = None
HippoRAGRAG = None


def create_rag_agent(
    strategy_config: StrategyConfig,
    llm_config: LLMConfig,
    vlm_config: VLMConfig,
    **dependencies: Any,
) -> Any:
    """
    Factory function to create a RAG agent based on the provided strategy configuration.

    Strategy implementations are imported lazily so lightweight modules and tests do
    not need optional dependencies for every baseline.
    """
    strategy_name = strategy_config.strategy
    print(f"INFO: Creating RAG agent with strategy: '{strategy_name}'")

    from Core.provider.llm import LLM
    from Core.provider.vlm import VLM

    llm_client = LLM(llm_config)
    vlm_client = VLM(vlm_config)

    if isinstance(strategy_config, TraverseRAGConfig):
        from Core.rag.traverse_agent import TraverseAgent

        tree_index = dependencies.get("tree_index")
        if not tree_index:
            raise ValueError("TraverseAgent requires a 'tree_index' in dependencies.")
        return TraverseAgent(
            config=strategy_config,
            llm=llm_client,
            vlm=vlm_client,
            tree_index=tree_index,
        )

    if isinstance(strategy_config, GBCRAGConfig):
        from Core.rag.gbc_rag import GBCRAG

        return GBCRAG(
            llm=llm_client,
            vlm=vlm_client,
            config=strategy_config,
            gbc_index=dependencies.get("gbc_index"),
        )

    if isinstance(strategy_config, GraphRAGConfig):
        from Core.rag.graph_rag import GraphRAG

        return GraphRAG(
            llm=llm_client,
            vlm=vlm_client,
            config=strategy_config,
            gbc_index=dependencies.get("gbc_index"),
        )

    if isinstance(strategy_config, VanillaConfig):
        from Core.rag.vanilla_rag import VanillaRAG

        return VanillaRAG(
            config=strategy_config,
            llm=llm_client,
            vector_store=dependencies.get("vector_store"),
            bm25=dependencies.get("bm25"),
            reranker=dependencies.get("reranker"),
            tree_index=dependencies.get("tree_index"),
        )

    if isinstance(strategy_config, GBCVanillaConfig):
        from Core.rag.gbc_vanilla_rag import GBCVanillaRAG

        return GBCVanillaRAG(
            llm=llm_client,
            vlm=vlm_client,
            config=strategy_config,
            tree_vdb=dependencies.get("tree_vdb"),
            graph_vdb=dependencies.get("graph_vdb"),
        )

    if isinstance(strategy_config, MMConfig):
        from Core.rag.mm_rag import MMRAG

        vector_store = dependencies.get("vector_store")
        if not vector_store:
            raise ValueError("MMRAG requires a 'vector_store' in dependencies.")
        return MMRAG(
            config=strategy_config,
            llm=llm_client,
            vlm=vlm_client,
            vector_store=vector_store,
            topk=strategy_config.topk if hasattr(strategy_config, "topk") else 3,
        )

    if isinstance(strategy_config, HRIRAGConfig):
        from Core.rag.hri_rag import HRIRAG

        tree_index = dependencies.get("tree_index")
        hri_index = dependencies.get("hri_index")
        bm25 = dependencies.get("bm25")
        hri_vector_store = dependencies.get("hri_vector_store")
        if not tree_index or not hri_index or not bm25:
            raise ValueError("HRIRAG requires 'tree_index', 'hri_index', and 'bm25'.")
        return HRIRAG(
            config=strategy_config,
            llm=llm_client,
            tree_index=tree_index,
            hri_index=hri_index,
            bm25=bm25,
            hri_vector_store=hri_vector_store,
        )

    if isinstance(strategy_config, EviBridgeRAGConfig):
        from Core.rag.evibridge_rag import EviBridgeRAG

        evibridge_index = dependencies.get("evibridge_index")
        bm25 = dependencies.get("bm25")
        evibridge_vector_store = dependencies.get("evibridge_vector_store")
        reranker = dependencies.get("reranker")
        if not evibridge_index or not bm25:
            raise ValueError("EviBridgeRAG requires 'evibridge_index' and 'bm25'.")
        return EviBridgeRAG(
            config=strategy_config,
            llm=llm_client,
            evibridge_index=evibridge_index,
            bm25=bm25,
            evibridge_vector_store=evibridge_vector_store,
            reranker=reranker,
        )

    if isinstance(strategy_config, LightRAGConfig):
        global LightRAGRAG
        if LightRAGRAG is None:
            from Core.rag.lightrag_rag import LightRAGRAG as _LightRAGRAG

            LightRAGRAG = _LightRAGRAG
        tree_index = dependencies.get("tree_index")
        save_path = dependencies.get("save_path")
        if tree_index is None or not save_path:
            raise ValueError("LightRAGRAG requires 'tree_index' and 'save_path'.")
        return LightRAGRAG(
            config=strategy_config,
            llm=llm_client,
            tree_index=tree_index,
            save_path=save_path,
        )

    if isinstance(strategy_config, HippoRAGConfig):
        global HippoRAGRAG
        if HippoRAGRAG is None:
            from Core.rag.hipporag_rag import HippoRAGRAG as _HippoRAGRAG

            HippoRAGRAG = _HippoRAGRAG
        tree_index = dependencies.get("tree_index")
        save_path = dependencies.get("save_path")
        if tree_index is None or not save_path:
            raise ValueError("HippoRAGRAG requires 'tree_index' and 'save_path'.")
        return HippoRAGRAG(
            config=strategy_config,
            llm=llm_client,
            tree_index=tree_index,
            save_path=save_path,
        )

    raise NotImplementedError(
        f"RAG agent for strategy '{strategy_name}' is not implemented."
    )
