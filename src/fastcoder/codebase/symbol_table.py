"""Global symbol registry and lookup for codebase symbols."""

import re
from typing import Optional

from fastcoder.types.codebase import ASTNode, SymbolInfo


class SymbolTable:
    """Global registry of exported symbols across the codebase."""

    def __init__(self) -> None:
        """Initialize the symbol table."""
        self.symbols: list[SymbolInfo] = []
        self.file_symbols: dict[str, list[SymbolInfo]] = {}

    def build(self, ast_index: dict[str, list[ASTNode]]) -> None:
        """
        Populate symbol table from AST index.

        Args:
            ast_index: Dict mapping file paths to lists of ASTNode objects
        """
        self.symbols.clear()
        self.file_symbols.clear()

        for file_path, nodes in ast_index.items():
            file_symbols: list[SymbolInfo] = []

            for node in nodes:
                if node.exported and node.name:
                    symbol = SymbolInfo(
                        name=node.name,
                        kind=node.type,
                        file=file_path,
                        line=node.start_line,
                        exported=True,
                        type_signature=node.signature,
                    )
                    file_symbols.append(symbol)
                    self.symbols.append(symbol)

            if file_symbols:
                self.file_symbols[file_path] = file_symbols

    def add_symbol(self, symbol: SymbolInfo) -> None:
        """
        Add a symbol to the table.

        Args:
            symbol: SymbolInfo object to add
        """
        self.symbols.append(symbol)
        if symbol.file not in self.file_symbols:
            self.file_symbols[symbol.file] = []
        self.file_symbols[symbol.file].append(symbol)

    def remove_symbols_for_file(self, file_path: str) -> None:
        """
        Remove all symbols for a file.

        Args:
            file_path: Path to the file
        """
        if file_path in self.file_symbols:
            for symbol in self.file_symbols[file_path]:
                if symbol in self.symbols:
                    self.symbols.remove(symbol)
            del self.file_symbols[file_path]

    def lookup(self, name: str) -> list[SymbolInfo]:
        """
        Exact symbol lookup by name.

        Args:
            name: Symbol name

        Returns:
            List of matching SymbolInfo objects
        """
        return [s for s in self.symbols if s.name == name]

    def lookup_by_file(self, file_path: str) -> list[SymbolInfo]:
        """
        Get all symbols in a file.

        Args:
            file_path: Path to the file

        Returns:
            List of SymbolInfo objects in the file
        """
        return self.file_symbols.get(file_path, [])

    def resolve_import(self, symbol_name: str) -> Optional[SymbolInfo]:
        """
        Resolve an imported symbol name to its definition.

        Args:
            symbol_name: Name of the symbol

        Returns:
            SymbolInfo if found, None otherwise
        """
        matches = self.lookup(symbol_name)
        if matches:
            return matches[0]
        return None

    def search_symbols(self, query: str) -> list[SymbolInfo]:
        """
        Fuzzy search symbols by name.

        Args:
            query: Search query (case-insensitive substring)

        Returns:
            List of matching SymbolInfo objects
        """
        query_lower = query.lower()
        results: list[SymbolInfo] = []

        for symbol in self.symbols:
            if query_lower in symbol.name.lower():
                results.append(symbol)

        # Sort by relevance (exact match, starts with, then substring)
        def relevance_score(symbol: SymbolInfo) -> tuple[int, int]:
            if symbol.name.lower() == query_lower:
                return (0, 0)
            elif symbol.name.lower().startswith(query_lower):
                return (1, len(symbol.name))
            else:
                return (2, len(symbol.name))

        results.sort(key=relevance_score)
        return results

    def increment_usage(self, symbol_name: str) -> None:
        """
        Increment usage count for a symbol.

        Args:
            symbol_name: Name of the symbol
        """
        for symbol in self.symbols:
            if symbol.name == symbol_name:
                symbol.usage_count += 1

    def get_stats(self) -> dict:
        """
        Get statistics about the symbol table.

        Returns:
            Dict with symbol statistics
        """
        return {
            "total_symbols": len(self.symbols),
            "total_files": len(self.file_symbols),
            "symbols_by_kind": self._count_by_kind(),
            "most_used": self._get_most_used_symbols(10),
        }

    def _count_by_kind(self) -> dict[str, int]:
        """Count symbols by kind."""
        counts: dict[str, int] = {}
        for symbol in self.symbols:
            counts[symbol.kind] = counts.get(symbol.kind, 0) + 1
        return counts

    def _get_most_used_symbols(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most used symbols."""
        sorted_symbols = sorted(self.symbols, key=lambda s: s.usage_count, reverse=True)
        return [(s.name, s.usage_count) for s in sorted_symbols[:limit]]
