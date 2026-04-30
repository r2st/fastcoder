"""TF-IDF based semantic search for code chunks."""

import math
import re
from collections import Counter
from typing import Optional

from fastcoder.types.codebase import CodeChunk, SearchResult


class SemanticSearch:
    """TF-IDF based semantic search engine for code."""

    def __init__(self) -> None:
        """Initialize the semantic search engine."""
        self.chunks: list[CodeChunk] = []
        self.inverted_index: dict[str, list[tuple[int, float]]] = {}  # token -> [(chunk_idx, tfidf)]
        self.doc_freq: dict[str, int] = {}  # token -> document frequency
        self.search_cache: dict[str, list[SearchResult]] = {}

    def index(self, chunks: list[CodeChunk]) -> None:
        """
        Build inverted index with TF-IDF weights.

        Args:
            chunks: List of CodeChunk objects to index
        """
        self.chunks = chunks
        self.inverted_index.clear()
        self.doc_freq.clear()
        self.search_cache.clear()

        # First pass: build document frequency
        all_tokens: list[set[str]] = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            all_tokens.append(tokens)
            for token in tokens:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        # Second pass: build inverted index with TF-IDF
        for chunk_idx, chunk in enumerate(chunks):
            tokens = all_tokens[chunk_idx]
            token_counts = Counter(tokens)
            total_tokens = len(tokens) if tokens else 1

            for token, count in token_counts.items():
                # TF: term frequency
                tf = count / total_tokens
                # IDF: inverse document frequency
                idf = math.log(len(chunks) / (self.doc_freq[token] + 1)) + 1
                # TF-IDF score
                tfidf = tf * idf

                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((chunk_idx, tfidf))

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search for code chunks matching the query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects
        """
        # Check cache first
        if query in self.search_cache:
            return self.search_cache[query][:top_k]

        # Try exact text search first
        text_results = self._exact_text_search(query)
        if text_results:
            results = text_results[:top_k]
            self.search_cache[query] = results
            return results

        # Fall back to semantic search
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Calculate query vector (TF-IDF)
        query_vector: dict[str, float] = {}
        for token in query_tokens:
            query_vector[token] = query_vector.get(token, 0) + 1

        # Normalize query vector
        query_norm = math.sqrt(sum(v * v for v in query_vector.values())) if query_vector else 1
        if query_norm > 0:
            query_vector = {k: v / query_norm for k, v in query_vector.items()}

        # Calculate similarity with each document
        chunk_scores: dict[int, float] = {}

        for token, query_weight in query_vector.items():
            if token in self.inverted_index:
                for chunk_idx, tfidf in self.inverted_index[token]:
                    if chunk_idx not in chunk_scores:
                        chunk_scores[chunk_idx] = 0
                    chunk_scores[chunk_idx] += query_weight * tfidf

        # Sort by score and return top results
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        results: list[SearchResult] = []

        for chunk_idx, score in sorted_chunks[:top_k]:
            if chunk_idx < len(self.chunks):
                results.append(
                    SearchResult(
                        chunk=self.chunks[chunk_idx],
                        score=score,
                        match_type="semantic",
                    )
                )

        self.search_cache[query] = results
        return results

    def add_chunk(self, chunk: CodeChunk) -> None:
        """
        Add a single chunk to the index.

        Args:
            chunk: CodeChunk object to add
        """
        # For simplicity, rebuild the entire index
        self.chunks.append(chunk)
        self.index(self.chunks)
        self.search_cache.clear()

    def remove_chunks_for_file(self, file_path: str) -> None:
        """
        Remove all chunks for a file from the index.

        Args:
            file_path: Path to the file
        """
        self.chunks = [c for c in self.chunks if c.file != file_path]
        self.index(self.chunks)
        self.search_cache.clear()

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text with support for camelCase, snake_case, etc.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Split on camelCase boundaries
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # Split on snake_case
        text = text.replace("_", " ")

        # Split on special characters and whitespace
        tokens = re.findall(r"\b\w+\b", text.lower())

        # Filter out very short tokens (noise)
        return [t for t in tokens if len(t) > 2]

    def _exact_text_search(self, query: str) -> list[SearchResult]:
        """
        Perform exact text search as fallback.

        Args:
            query: Search query

        Returns:
            List of SearchResult objects with text matches
        """
        results: list[SearchResult] = []
        query_lower = query.lower()

        for idx, chunk in enumerate(self.chunks):
            if query_lower in chunk.content.lower():
                # Calculate score based on position and frequency
                matches = len(re.findall(re.escape(query_lower), chunk.content.lower()))
                score = min(1.0, matches / 10.0)  # Cap at 1.0
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=score,
                        match_type="text",
                    )
                )

        return sorted(results, key=lambda r: r.score, reverse=True)

    def get_index_stats(self) -> dict:
        """
        Get statistics about the search index.

        Returns:
            Dict with index statistics
        """
        return {
            "total_chunks": len(self.chunks),
            "total_tokens": len(self.inverted_index),
            "avg_chunk_size": sum(len(c.content) for c in self.chunks) / len(self.chunks) if self.chunks else 0,
            "cache_size": len(self.search_cache),
        }
