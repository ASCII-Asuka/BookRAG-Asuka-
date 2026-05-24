import math
import pickle
import re
from collections import Counter
from typing import List

import tqdm


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75, metadatas=None):
        self.original_docs = docs
        self.metadatas = metadatas or [{} for _ in docs]
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [self._tokenize(doc) for doc in self.original_docs]
        self.doc_len = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = (
            sum(self.doc_len) / len(self.tokenized_docs) if self.tokenized_docs else 0
        )
        self.doc_freqs = []
        self.idf = {}

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens: List[str] = []
        for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text):
            if re.fullmatch(r"[\u4e00-\u9fff]+", part):
                tokens.extend(part)
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
            else:
                tokens.append(part)
        return tokens

    def initialize(self):
        if not self.tokenized_docs:
            return

        df = {}
        self.doc_freqs = []
        for doc in self.tokenized_docs:
            self.doc_freqs.append(Counter(doc))
            for word in set(doc):
                df[word] = df.get(word, 0) + 1

        num_docs = len(self.tokenized_docs)
        for word, freq in tqdm.tqdm(df.items(), total=len(df), desc="Calculating IDF"):
            self.idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

    def _get_score(self, doc_index: int, query_tokens: List[str]) -> float:
        score = 0.0
        doc_freq = self.doc_freqs[doc_index]
        for word in query_tokens:
            if word in doc_freq:
                freq = doc_freq[word]
                score += (self.idf.get(word, 0) * freq * (self.k1 + 1)) / (
                    freq
                    + self.k1
                    * (1 - self.b + self.b * self.doc_len[doc_index] / self.avgdl)
                )
        return score

    def search(self, query_text: str, top_k: int) -> List[dict]:
        query_tokens = self._tokenize(query_text)
        scores = [
            (self._get_score(i, query_tokens), i)
            for i in range(len(self.tokenized_docs))
        ]
        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scores[:top_k]:
            metadata = (
                self.metadatas[idx]
                if hasattr(self, "metadatas") and idx < len(self.metadatas)
                else {}
            )
            results.append(
                {
                    "id": idx,
                    "score": score,
                    "content": self.original_docs[idx],
                    "metadata": metadata,
                }
            )
        return results

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            bm25 = pickle.load(f)
        if not hasattr(bm25, "metadatas"):
            bm25.metadatas = [{} for _ in getattr(bm25, "original_docs", [])]
        return bm25

    def close(self):
        if hasattr(self, "original_docs"):
            del self.original_docs
        if hasattr(self, "metadatas"):
            del self.metadatas
