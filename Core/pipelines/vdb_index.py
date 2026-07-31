import os
from typing import Any, Dict, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from Core.Index.Tree import DocumentTree, NodeType
from Core.configs.vdb_config import VDBConfig
from Core.configs.system_config import SystemConfig
from Core.utils.utils import TextProcessor
from Core.utils.bm25 import BM25
import json
import logging

log = logging.getLogger(__name__)

save_path = "/home/wangshu/multimodal/GBC-RAG/test/sf/"


def process_tree_nodes(tree: DocumentTree) -> Tuple[Dict[str, List], Dict[str, List]]:
    text_list = []
    text_meta_data = []
    image_list = []
    image_meta_data = []
    image_str_list = []
    for node in tree.nodes:
        if node == tree.root_node:
            continue

        node_type = node.type
        meta_data = {
            "node_id": node.index_id,
            "pdf_id": node.meta_info.pdf_id,
        }

        if node_type == NodeType.IMAGE:
            image_path = node.meta_info.img_path
            image_str = node.meta_info.caption + node.meta_info.footnote
            text_list.append(image_str)
            text_meta_data.append(meta_data)

            # Check if the image path exists before adding it
            if image_path and os.path.exists(image_path):
                image_list.append(image_path)
                image_meta_data.append(meta_data)
                image_str_list.append(image_str)
        elif node_type == NodeType.TABLE:
            table_str = node.meta_info.content
            table_body = node.meta_info.table_body
            if table_body:
                table_str += table_body
            text_list.append(table_str)
            text_meta_data.append(meta_data)

            table_img = node.meta_info.img_path
            if table_img and os.path.exists(table_img):
                image_list.append(table_img)
                image_meta_data.append(meta_data)
                image_str_list.append(table_str)
        elif (
            node_type == NodeType.TEXT
            or node_type == NodeType.TITLE
            or node_type == NodeType.EQUATION
        ):
            text_content = node.meta_info.content
            if text_content:
                text_list.append(text_content)
                text_meta_data.append(meta_data)

    text_dict = {"text": text_list, "meta": text_meta_data}
    image_dict = {
        "image": image_list,
        "meta": image_meta_data,
        "image_str": image_str_list,
    }
    return text_dict, image_dict


def build_vdb_index(tree: DocumentTree, vdb_cfg: VDBConfig):
    from Core.provider.embedding import GmeEmbeddingProvider, TextEmbeddingProvider
    from Core.provider.vdb import VectorStore

    if vdb_cfg.mm_embedding:
        embedder = GmeEmbeddingProvider(
            model_name=vdb_cfg.embedding_config.model_name,
            device=vdb_cfg.embedding_config.device,
        )
        log.info("Using GME multi-modal embedding model for vector database.")
    else:
        embedder = TextEmbeddingProvider(
            model_name=vdb_cfg.embedding_config.model_name,
            device=vdb_cfg.embedding_config.device,
            backend=vdb_cfg.embedding_config.backend,
            api_base=vdb_cfg.embedding_config.api_base,
            api_key=vdb_cfg.embedding_config.api_key,
            max_length=vdb_cfg.embedding_config.max_length,
        )
        log.info("Using text embedding model for vector database.")

    vdb = VectorStore(
        embedding_model=embedder,
        db_path=vdb_cfg.vdb_dir_name,
        collection_name=vdb_cfg.collection_name,
    )

    try:
        text_dict, image_dict = process_tree_nodes(tree)

        text, text_meta = text_dict["text"], text_dict["meta"]
        vdb.add_texts(texts=text, metadatas=text_meta)

        mm_vdb = vdb_cfg.mm_embedding
        if mm_vdb is True:
            image, img_meta, img_str = (
                image_dict["image"],
                image_dict["meta"],
                image_dict["image_str"],
            )
            vdb.add_images(image_paths=image, metadatas=img_meta, image_str=img_str)
            log.info("Images added to vector database successfully.")

        log.info("Vector database index built successfully.")
    finally:
        vdb.close()  # Close the vector store and embedding model to free resources.
    return


def get_input_text(cfg: SystemConfig) -> str:
    pdf_path = cfg.pdf_path
    file_name = str(Path(pdf_path).stem)
    method = getattr(getattr(cfg, "mineru", None), "method", None)
    candidate_methods = []
    for name in [method, "auto", "vlm", "ocr", "txt"]:
        if name and name not in candidate_methods:
            candidate_methods.append(name)

    candidate_paths = [
        os.path.join(cfg.save_path, method_name, f"{file_name}.md")
        for method_name in candidate_methods
    ]
    for md_file in candidate_paths:
        if os.path.exists(md_file):
            with open(md_file, "r", encoding="utf-8") as f:
                return f.read()

    candidates = "\n".join(f"- {path}" for path in candidate_paths)
    raise FileNotFoundError(
        "Could not find MinerU markdown output for vanilla indexing. "
        f"Checked candidate paths:\n{candidates}"
    )


def _node_type_value(node_type: Any) -> str:
    return getattr(node_type, "value", str(node_type))


def _node_text(node) -> str:
    node_type = node.type
    if node_type == NodeType.IMAGE:
        return f"{node.meta_info.caption or ''}{node.meta_info.footnote or ''}".strip()
    if node_type == NodeType.TABLE:
        return f"{node.meta_info.content or ''}\n{node.meta_info.table_body or ''}".strip()
    return (node.meta_info.content or "").strip()


def _title_path(tree: DocumentTree, node) -> List[str]:
    path_nodes = tree.get_path_from_root(node.index_id)
    titles: List[str] = []
    for path_node in path_nodes:
        content = (path_node.meta_info.content or "").strip()
        if not content:
            continue
        if path_node.outline_node or path_node.type == NodeType.TITLE:
            titles.append(content)
    return titles


def _page_number(page_idx: Any):
    if page_idx is None:
        return None
    try:
        return int(page_idx) + 1
    except (TypeError, ValueError):
        return None


def _metadata_without_none(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in metadata.items() if value is not None}


def _strategy_config(cfg: SystemConfig) -> Any:
    return getattr(getattr(cfg, "rag", None), "strategy_config", None)


def _use_paragraph_bm25_corpus(cfg: SystemConfig) -> bool:
    strategy_config = _strategy_config(cfg)
    return (
        getattr(cfg, "index_type", None) == "bm25"
        and getattr(strategy_config, "bm25_corpus", "chunk") == "paragraph"
    )


def _use_paragraph_corpus(cfg: SystemConfig) -> bool:
    strategy_config = _strategy_config(cfg)
    return _use_paragraph_bm25_corpus(cfg) or (
        getattr(cfg, "index_type", None) in {"vanilla", "raptor"}
        and getattr(strategy_config, "corpus_unit", "chunk") == "paragraph"
    )


def _get_tree_paragraphs(tree: DocumentTree) -> Tuple[List[str], List[Dict[str, Any]]]:
    paragraphs: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    chunk_id = 0
    for node in tree.get_nodes(hasRoot=False):
        if node.type != NodeType.TEXT:
            continue
        text = str(node.meta_info.content or "")
        if not text.strip():
            continue
        title_path = _title_path(tree, node)
        section_id = title_path[-1] if title_path else ""
        paragraph_id = node.index_id
        source_metadata = (
            node.meta_info.pdf_para_block
            if isinstance(node.meta_info.pdf_para_block, dict)
            else {}
        )
        paragraphs.append(text)
        metadatas.append(
            _metadata_without_none(
                {
                    "source": source_metadata.get("source") or "qasper_paragraph",
                    "chunk_id": chunk_id,
                    "source_chunk_index": 0,
                    "node_id": node.index_id,
                    "source_node_id": node.index_id,
                    "paragraph_id": paragraph_id,
                    "evidence_id": paragraph_id,
                    "pdf_id": node.meta_info.pdf_id,
                    "page": _page_number(node.meta_info.page_idx),
                    "node_type": _node_type_value(node.type),
                    "section_id": section_id,
                    "section": section_id,
                    "title_path": " > ".join(title_path),
                    "qasper_evidence_text": text,
                    "hotpot_title": source_metadata.get("hotpot_title"),
                    "hotpot_sent_id": source_metadata.get("hotpot_sent_id"),
                }
            )
        )
        chunk_id += 1
    return paragraphs, metadatas


def get_tree_chunks(cfg: SystemConfig) -> Tuple[List[str], List[Dict[str, Any]]]:
    tree_path = DocumentTree.get_save_path(cfg.save_path)
    if not os.path.exists(tree_path):
        return [], []

    tree = DocumentTree.load_from_file(tree_path)
    if _use_paragraph_corpus(cfg):
        return _get_tree_paragraphs(tree)

    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    chunk_id = 0
    for node in tree.get_nodes(hasRoot=False):
        text = _node_text(node)
        if not text:
            continue
        title_path = _title_path(tree, node)
        section_id = title_path[-1] if title_path else ""
        node_chunks = TextProcessor.split_text_into_chunks(text=text, max_length=500)
        for source_chunk_index, chunk in enumerate(node_chunks):
            chunks.append(chunk)
            metadatas.append(
                _metadata_without_none(
                    {
                        "source": "tree",
                        "chunk_id": chunk_id,
                        "source_chunk_index": source_chunk_index,
                        "node_id": node.index_id,
                        "source_node_id": node.index_id,
                        "pdf_id": node.meta_info.pdf_id,
                        "page": _page_number(node.meta_info.page_idx),
                        "node_type": _node_type_value(node.type),
                        "section_id": section_id,
                        "section": section_id,
                        "title_path": " > ".join(title_path),
                    }
                )
            )
            chunk_id += 1
    return chunks, metadatas


def get_all_chunks(cfg: SystemConfig):
    index_type = cfg.index_type
    tree_chunks: List[str] = []
    tree_metadatas: List[Dict[str, Any]] = []
    if index_type in ["vanilla", "bm25", "raptor"]:
        tree_chunks, tree_metadatas = get_tree_chunks(cfg)

    if index_type in ["vanilla", "bm25"]:
        if tree_chunks:
            log.info(
                "Using DocumentTree nodes with metadata for %s index (%s chunks).",
                index_type,
                len(tree_chunks),
            )
            return tree_chunks, tree_metadatas

    base_metadatas = None
    if index_type == "raptor" and tree_chunks:
        log.info(
            "Using DocumentTree nodes with metadata for RAPTOR index (%s chunks).",
            len(tree_chunks),
        )
        chunks = tree_chunks
        base_metadatas = tree_metadatas
    else:
        corpus_text = get_input_text(cfg)
        chunks = TextProcessor.split_text_into_chunks(text=corpus_text, max_length=500)

    if index_type == "vanilla" or index_type == "bm25":
        meta_datas = [{"source": "document", "chunk_id": i} for i in range(len(chunks))]
        return chunks, meta_datas
    elif index_type == "raptor":
        from Core.provider.embedding import TextEmbeddingProvider
        from Core.provider.llm import LLM
        from Core.utils.raptor_utils import raptor_tree

        llm = LLM(cfg.llm)
        embed_cfg = cfg.vdb.embedding_config
        embedder = TextEmbeddingProvider(
            model_name=embed_cfg.model_name,
            device=embed_cfg.device,
            backend=embed_cfg.backend,
            api_base=embed_cfg.api_base,
            api_key=embed_cfg.api_key,
            max_length=embed_cfg.max_length,
        )

        all_tree_text, all_meta_data = raptor_tree(
            chunks, embedder=embedder, llm=llm, base_metadatas=base_metadatas
        )
        embedder.close()
        return all_tree_text, all_meta_data


def build_other_vdb_index(cfg: SystemConfig):
    vdb_dir = os.path.join(cfg.save_path, cfg.vdb.vdb_dir_name)
    if os.path.exists(vdb_dir):
        if cfg.vdb.force_rebuild:
            import shutil

            shutil.rmtree(vdb_dir)
            log.info(
                f"Vector database path already exists: {vdb_dir}. Remove and rebuild"
            )
        else:
            log.info(f"Vector database path already exists: {vdb_dir}. Skip")
            return

    os.makedirs(vdb_dir, exist_ok=True)
    all_chunks, meta_datas = get_all_chunks(cfg)

    if cfg.index_type == "bm25":
        save_path = os.path.join(vdb_dir, "bm25_index.pkl")
        bm25 = BM25(all_chunks, metadatas=meta_datas)
        bm25.initialize()
        # test
        query = "quick"
        results = bm25.search(query, top_k=2)
        log.info("BM25 smoke test completed with %s results", len(results))

        bm25.save(save_path)
        log.info(f"BM25 index saved to {save_path}")
    else:
        from Core.provider.embedding import TextEmbeddingProvider
        from Core.provider.vdb import VectorStore

        vdb_config = cfg.vdb
        vdb = VectorStore(
            embedding_model=TextEmbeddingProvider(
                model_name=vdb_config.embedding_config.model_name,
                device=vdb_config.embedding_config.device,
                backend=vdb_config.embedding_config.backend,
                api_base=vdb_config.embedding_config.api_base,
                api_key=vdb_config.embedding_config.api_key,
                max_length=vdb_config.embedding_config.max_length,
            ),
            db_path=vdb_dir,
            collection_name=vdb_config.collection_name,
        )
        try:
            vdb.add_texts(texts=all_chunks, metadatas=meta_datas)
            log.info("Vector database index built successfully.")
        finally:
            vdb.close()  # Close the vector store and embedding model to free resources.


def load_pdf_lists_from_dir(save_dir):
    res_list = []
    pdf_list_json_files = os.listdir(save_dir)
    for pdf_list_json_file in pdf_list_json_files:
        if not pdf_list_json_file.endswith(".json"):
            continue
        pdf_list_path = os.path.join(save_dir, pdf_list_json_file)
        with open(pdf_list_path, "r", encoding="utf-8") as f:
            pdf_list = json.load(f)
        tmp_dict = {"pdf_list": pdf_list, "pdf_list_path": pdf_list_path}
        res_list.append(tmp_dict)

    return res_list


def compute_mm_embedding(cfg: SystemConfig, tree_index: DocumentTree):
    from Core.provider.embedding import GmeEmbeddingProvider

    embedder_cfg = cfg.vdb.embedding_config
    embedder = GmeEmbeddingProvider(
        model_name=embedder_cfg.model_name,
        device=embedder_cfg.device,
    )

    text_only_group_data = []
    image_only_group_data = []
    fused_group_data = []

    all_node_data = []
    all_embeddings_values = []

    for i, node in enumerate(tree_index.nodes):
        if node == tree_index.root_node:
            continue

        node_id = node.index_id
        node_type = node.type
        content = node.meta_info.content
        img_path = (
            node.meta_info.img_path
            if (node_type == NodeType.IMAGE or node_type == NodeType.TABLE)
            else None
        )

        node_info = {
            "node_id": node_id,
            "node_type": node_type,
            "content": content,
            "img_path": img_path,
            "embedding_idx": None,
        }
        current_node_data_idx = len(all_node_data)
        all_node_data.append(node_info)

        if content and img_path:
            fused_group_data.append(
                {
                    "original_node_data_idx": current_node_data_idx,
                    "text": content,
                    "image": img_path,
                }
            )
        elif content:
            text_only_group_data.append(
                {"original_node_data_idx": current_node_data_idx, "text": content}
            )
        elif img_path:
            image_only_group_data.append(
                {"original_node_data_idx": current_node_data_idx, "image": img_path}
            )

    if text_only_group_data:
        texts = [item["text"] for item in text_only_group_data]
        text_embeddings = embedder.embed_texts(texts)
        for i, item in enumerate(text_only_group_data):
            original_node_data_idx = item["original_node_data_idx"]
            embedding = text_embeddings[i]

            embedding_idx = len(all_embeddings_values)
            all_embeddings_values.append(embedding)

            all_node_data[original_node_data_idx]["embedding_idx"] = embedding_idx

    if image_only_group_data:
        images = [item["image"] for item in image_only_group_data]
        image_embeddings = embedder.embed_images(images)
        for i, item in enumerate(image_only_group_data):
            original_node_data_idx = item["original_node_data_idx"]
            embedding = image_embeddings[i]

            embedding_idx = len(all_embeddings_values)
            all_embeddings_values.append(embedding)

            all_node_data[original_node_data_idx]["embedding_idx"] = embedding_idx

    if fused_group_data:
        texts = [item["text"] for item in fused_group_data]
        images = [item["image"] for item in fused_group_data]
        fused_embeddings = embedder.embed_fused(texts=texts, images=images)
        for i, item in enumerate(fused_group_data):
            original_node_data_idx = item["original_node_data_idx"]
            embedding = fused_embeddings[i]

            embedding_idx = len(all_embeddings_values)
            all_embeddings_values.append(embedding)

            all_node_data[original_node_data_idx]["embedding_idx"] = embedding_idx
    embedder.clear_cache()

    # --- 保存所有节点元数据到JSON文件 ---
    save_dir = cfg.save_path
    os.makedirs(save_dir, exist_ok=True)  # 确保保存路径存在
    metadata_filepath = os.path.join(save_dir, "mm_node_metadata.json")
    embeddings_filepath = os.path.join(save_dir, "mm_embeddings.npy")

    with open(metadata_filepath, "w", encoding="utf-8") as f:
        json.dump(all_node_data, f, ensure_ascii=False, indent=4)

    if all_embeddings_values:
        final_embeddings_array = np.array(all_embeddings_values)
        np.save(embeddings_filepath, final_embeddings_array)
        log.info(f"All embeddings saved to: {embeddings_filepath}")
    else:
        log.warning("No embeddings were computed, .npy file not saved.")

    log.info(f"All node metadata saved to: {metadata_filepath}")


def compute_mm_embedding_question(cfg: SystemConfig, group: pd.DataFrame):
    from Core.provider.embedding import GmeEmbeddingProvider

    embedder_cfg = cfg.vdb.embedding_config
    embedder = GmeEmbeddingProvider(
        model_name=embedder_cfg.model_name,
        device=embedder_cfg.device,
    )

    group_dedup = group.drop_duplicates(subset=["question"], keep="first")
    questions = group_dedup["question"].tolist()
    RERANKER_INSTRUCTION = "Retrieve the most relevant document for the given query."

    # add instruction for gme model
    question_embeddings_raw = embedder.embed_texts(
        questions, instruction=RERANKER_INSTRUCTION
    )

    all_question_embeddings = []
    question_embedding_indices = []

    for i, embedding in enumerate(question_embeddings_raw):
        all_question_embeddings.append(embedding)
        question_embedding_indices.append(len(all_question_embeddings) - 1)

    group_dedup["question_embedding_idx"] = question_embedding_indices

    save_dir = cfg.save_path
    os.makedirs(save_dir, exist_ok=True)

    question_metadata_filepath = os.path.join(save_dir, "mm_question_metadata.json")
    question_embeddings_filepath = os.path.join(save_dir, "mm_question_embeddings.npy")

    group_dedup.to_json(
        question_metadata_filepath, orient="records", force_ascii=False, indent=4
    )

    if all_question_embeddings:
        final_question_embeddings_array = np.array(all_question_embeddings)
        np.save(question_embeddings_filepath, final_question_embeddings_array)
        log.info(f"All question embeddings saved to: {question_embeddings_filepath}")
    else:
        log.warning("No question embeddings were computed, .npy file not saved.")

    log.info(f"All question metadata saved to: {question_metadata_filepath}")


if __name__ == "__main__":
    # tmp_tree_path = f"{save_path}/sftree.pkl"
    # tree_index = DocumentTree.load_from_file(tmp_tree_path)
    # print(f"Loaded tree index from: {tmp_tree_path}")
    # vector_store = build_vdb_from_tree(tree_index)
    print("test")
