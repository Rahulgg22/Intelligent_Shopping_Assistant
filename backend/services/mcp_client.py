from __future__ import annotations

import logging
import time
from typing import Any

from backend.mcp_server import mcp

logger = logging.getLogger("mcp_client")


class MCPClient:
    """Client for discovering and invoking tools registered on the PS-12 MCP Server."""

    def __init__(self, server=None):
        self.server = server or mcp

    def discover_tools(self) -> list[dict[str, Any]]:
        """Return list of discovered tools with metadata and parameter schemas."""
        tools = self.server._tool_manager.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.parameters or {},
            }
            for t in tools
        ]

    def has_tool(self, tool_name: str) -> bool:
        return any(t.name == tool_name for t in self.server._tool_manager.list_tools())

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Invoke an MCP tool by name with arguments.
        Logs execution duration, provider used, status, and result count.
        Never logs credentials or sensitive conversation details.
        """
        args = arguments or {}
        tool = self.server._tool_manager.get_tool(tool_name)
        if not tool:
            logger.error("mcp_tool_invocation: tool=%s status=failed error=tool_not_found", tool_name)
            raise ValueError(f"MCP tool '{tool_name}' not found on server")

        provider_used = "deterministic"
        if tool_name in ("search_products", "get_product_details", "get_provider_status", "find_product_alternatives"):
            provider_used = "composite_search"
        elif tool_name == "search_product_reviews":
            provider_used = "tavily"

        start_time = time.perf_counter()
        try:
            result = tool.fn(**args)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Count results based on tool output shape
            results_count = 0
            if isinstance(result, dict):
                if "products" in result and isinstance(result["products"], list):
                    results_count = len(result["products"])
                elif "comparison" in result and isinstance(result["comparison"], list):
                    results_count = len(result["comparison"])
                elif "reviews" in result and isinstance(result["reviews"], list):
                    results_count = len(result["reviews"])
                elif "alternatives" in result and isinstance(result["alternatives"], list):
                    results_count = len(result["alternatives"])
                elif "product" in result and result["product"]:
                    results_count = 1

            logger.info(
                "mcp_tool_invocation: tool=%s status=success duration_ms=%.1f results_count=%d provider=%s error=none",
                tool_name, duration_ms, results_count, provider_used
            )
            return result
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_category = type(exc).__name__
            logger.error(
                "mcp_tool_invocation: tool=%s status=failed duration_ms=%.1f results_count=0 provider=%s error=%s",
                tool_name, duration_ms, provider_used, error_category
            )
            raise


_default_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    global _default_client
    if _default_client is None:
        _default_client = MCPClient()
    return _default_client
