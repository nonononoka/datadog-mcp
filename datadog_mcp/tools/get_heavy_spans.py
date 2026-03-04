"""
Get heavy span events tool
"""

import json
from typing import Any, Dict, List

from mcp.types import CallToolRequest, CallToolResult, Tool, TextContent

from ..utils.datadog_client import fetch_span_events


def get_tool_definition() -> Tool:
    """Get the tool definition for get_heavy_spans."""
    return Tool(
        name="get_heavy_spans",
        description=(
            "Search span events by resource_name with a duration threshold using the"
            " Datadog v2 spans/events endpoint. Useful for finding slow traces."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "resource_name": {
                    "type": "string",
                    "description": "Span resource_name to match (exact).",
                },
                "env": {
                    "type": "string",
                    "description": "Environment tag to filter (e.g., production, staging).",
                    "default": "production",
                },
                "service": {
                    "type": "string",
                    "description": "Service name to filter.",
                    "default": "lips",
                },
                "operation_name": {
                    "type": "string",
                    "description": "operation_name to filter (e.g., rack.request).",
                    "default": "rack.request",
                },
                "min_duration_ms": {
                    "type": "integer",
                    "description": "Minimum duration in milliseconds (used in @duration filter).",
                    "default": 1000,
                    "minimum": 1,
                    "maximum": 600000,
                },
                "time_from": {
                    "type": "string",
                    "description": "Start of window (relative like 'now-1h' or RFC3339).",
                    "default": "now-1h",
                },
                "time_to": {
                    "type": "string",
                    "description": "End of window (relative like 'now' or RFC3339).",
                    "default": "now",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of spans to return (Datadog max 1000).",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 1000,
                },
                "cursor": {
                    "type": "string",
                    "description": "Pagination cursor from previous response.",
                    "default": "",
                },
                "format": {
                    "type": "string",
                    "description": "Output format: table (default) or json.",
                    "enum": ["table", "json"],
                    "default": "table",
                },
            },
            "additionalProperties": False,
            "required": ["resource_name"],
        },
    )


def _extract_span_info(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract a concise view of span events."""
    results = []
    for span in spans:
        attrs = span.get("attributes", {}) or {}
        inner_attrs = attrs.get("attributes", {}) if isinstance(attrs.get("attributes"), dict) else {}
        merged = {**attrs, **inner_attrs}
        meta = merged.get("meta") if isinstance(merged.get("meta"), dict) else {}

        # Prefer custom.duration when present, otherwise duration
        raw_duration = None
        if isinstance(merged.get("custom"), dict) and "duration" in merged["custom"]:
            raw_duration = merged["custom"]["duration"]
        else:
            raw_duration = merged.get("duration")

        duration_ms = None
        if raw_duration is not None:
            duration_ms = raw_duration / 1_000_000 if raw_duration > 10000 else raw_duration

        results.append(
            {
                "trace_id": merged.get("trace_id") or meta.get("trace_id") or span.get("trace_id", ""),
                "span_id": merged.get("span_id") or meta.get("span_id") or span.get("id", ""),
                "service": merged.get("service") or meta.get("service") or "",
                "resource": merged.get("resource") or merged.get("resource_name") or meta.get("resource_name") or "",
                "operation": merged.get("name") or merged.get("operation_name") or meta.get("operation_name") or "",
                "duration_ms": duration_ms,
                "start": merged.get("start") or merged.get("start_time") or meta.get("_timestamp") or "",
            }
        )

    return results


def _format_spans_as_table(spans: List[Dict[str, Any]]) -> str:
    """Render spans as a simple fixed-width table."""
    if not spans:
        return "No spans found for the given query."

    # Truncate long values for readability
    display_spans = []
    for span in spans:
        span_copy = span.copy()
        if len(span_copy.get("resource", "")) > 60:
            span_copy["resource"] = span_copy["resource"][:57] + "..."
        display_spans.append(span_copy)

    trace_w = max(len("Trace ID"), max(len(str(s.get("trace_id", ""))) for s in display_spans))
    span_w = max(len("Span ID"), max(len(str(s.get("span_id", ""))) for s in display_spans))
    service_w = max(len("Service"), max(len(str(s.get("service", ""))) for s in display_spans))
    resource_w = max(len("Resource"), max(len(str(s.get("resource", ""))) for s in display_spans))
    dur_w = len("Duration(ms)")

    header = f"| {'Trace ID':<{trace_w}} | {'Span ID':<{span_w}} | {'Service':<{service_w}} | {'Resource':<{resource_w}} | {'Duration(ms)':>{dur_w}} |"
    separator = f"|{'-' * (trace_w + 2)}|{'-' * (span_w + 2)}|{'-' * (service_w + 2)}|{'-' * (resource_w + 2)}|{'-' * (dur_w + 2)}|"

    lines = [header, separator]
    for span in display_spans:
        duration_ms = span.get("duration_ms")
        duration_str = f"{duration_ms:.1f}" if duration_ms is not None else "N/A"
        lines.append(
            f"| {span.get('trace_id', ''):<{trace_w}} | {span.get('span_id', ''):<{span_w}} | "
            f"{span.get('service', ''):<{service_w}} | {span.get('resource', ''):<{resource_w}} | "
            f"{duration_str:>{dur_w}} |"
        )

    return "\n".join(lines)


async def handle_call(request: CallToolRequest) -> CallToolResult:
    """Handle the get_heavy_spans tool call."""
    try:
        args = request.arguments or {}

        resource_name = args.get("resource_name")
        if not resource_name:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: resource_name is required")],
                isError=True,
            )

        env = args.get("env", "production")
        service = args.get("service", "lips")
        operation_name = args.get("operation_name", "rack.request")
        min_duration_ms = args.get("min_duration_ms", 1000)
        time_from = args.get("time_from", "now-1h")
        time_to = args.get("time_to", "now")
        limit = args.get("limit", 100)
        cursor = args.get("cursor", "")
        output_format = args.get("format", "table")

        # Build query string
        query_parts = []
        if env:
            query_parts.append(f"env:{env}")
        if service:
            query_parts.append(f"service:{service}")
        if operation_name:
            query_parts.append(f"operation_name:{operation_name}")
        query_parts.append(f'resource_name:"{resource_name}"')
        query_parts.append(f"@duration:>={min_duration_ms}ms")
        query = " ".join(query_parts)

        response = await fetch_span_events(
            query=query,
            time_from=time_from,
            time_to=time_to,
            limit=limit,
            cursor=cursor if cursor else None,
        )

        spans_raw = response.get("data", [])
        spans = _extract_span_info(spans_raw)

        meta = response.get("meta", {}) or {}
        page = meta.get("page", {}) if isinstance(meta, dict) else {}
        next_cursor = page.get("after")

        if output_format == "json":
            content = {
                "query": query,
                "time_from": time_from,
                "time_to": time_to,
                "count": len(spans),
                "spans": spans,
                "pagination": {"next_cursor": next_cursor},
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(content, indent=2))],
                isError=False,
            )

        table = _format_spans_as_table(spans)
        summary = f"Found {len(spans)} span(s) | Query: {query} | Window: {time_from} -> {time_to}"
        if cursor:
            summary += " | Used cursor pagination"
        if next_cursor:
            summary += f" | Next cursor: {next_cursor}"

        final_content = f"{summary}\n{'=' * len(summary)}\n\n{table}"

        return CallToolResult(
            content=[TextContent(type="text", text=final_content)],
            isError=False,
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )
