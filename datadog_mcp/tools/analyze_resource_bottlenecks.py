"""
Analyze bottleneck spans for a specific resource.

Workflow:
1) Find heavy spans (by duration) for the target resource_name to collect trace IDs.
2) Fetch all spans for those trace IDs.
3) Group by trace and surface the longest spans to spot bottlenecks.
"""

import json
from typing import Any, Dict, List, Tuple

from mcp.types import CallToolRequest, CallToolResult, Tool, TextContent

from ..utils.datadog_client import fetch_span_events


def get_tool_definition() -> Tool:
    """Get the tool definition for analyze_resource_bottlenecks."""
    return Tool(
        name="analyze_resource_bottlenecks",
        description=(
            "Identify bottleneck spans for a given resource by first finding slow spans, "
            "then expanding to all spans in those traces to show the longest spans per trace."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "resource_name": {
                    "type": "string",
                    "description": "Span resource_name to analyze (exact match).",
                }
            },
            "additionalProperties": False,
            "required": ["resource_name"],
        },
    )


def _as_ms(duration_ns: Any) -> float:
    """Convert Datadog span duration (ns) to ms, tolerant of already-ms values."""
    if duration_ns is None:
        return None
    try:
        duration_ns = float(duration_ns)
    except Exception:
        return None
    return duration_ns / 1_000_000

def _extract_span(span: Dict[str, Any]) -> Dict[str, Any]:
    # Span events sometimes nest real data under attributes["attributes"]
    attrs = span.get("attributes", {}) or {}
    inner_attrs = attrs.get("attributes", {}) if isinstance(attrs.get("attributes"), dict) else {}
    # Merge inner over outer so inner overrides when present
    merged = {**attrs, **inner_attrs}

    meta = merged.get("meta") if isinstance(merged.get("meta"), dict) else {}

    custom = merged.get("custom") if isinstance(merged.get("custom"), dict) else {}

    return {
        "trace_id": merged.get("trace_id") or meta.get("trace_id") or span.get("trace_id") or "",
        "span_id": merged.get("span_id") or meta.get("span_id") or span.get("id") or "",
        "service": merged.get("service") or meta.get("service") or "",
        "resource": merged.get("resource") or merged.get("resource_name") or meta.get("resource_name") or "",
        "operation": custom.get("operation")
        or merged.get("name")
        or merged.get("operation_name")
        or meta.get("operation_name")
        or "",
        "duration_ms": _as_ms(custom.get("duration")) or _as_ms(merged.get("duration")),
    }


def _group_top_spans(spans: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """Group spans by trace and compute per-trace tops + aggregate resource stats."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    resource_stats: Dict[str, Dict[str, Any]] = {}

    for span in spans:
        tid = span.get("trace_id")
        if not tid:
            continue
        grouped.setdefault(tid, []).append(span)

        op = span.get("operation") or "(unknown)"
        res = span.get("resource") or "(unknown)"
        key = (op, res)
        stats = resource_stats.setdefault(key, {"operation": op, "resource": res, "count": 0, "total_ms": 0.0})
        stats["count"] += 1
        if span.get("duration_ms") is not None:
            stats["total_ms"] += span["duration_ms"]

    return grouped, resource_stats


def _format_resource_table(resource_stats: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    if not resource_stats:
        return "No span data to summarize."

    rows = []
    for (op, res), stats in resource_stats.items():
        total = stats["total_ms"]
        count = stats["count"]
        avg = total / count if count else 0
        resource_label = f"{op} {res}"
        rows.append((resource_label, count, total, avg))

    rows.sort(key=lambda r: r[2], reverse=True)  # sort by total_ms

    res_w = max(len("Resource"), max(len(r[0]) for r in rows))
    cnt_w = len("Count")
    tot_w = len("Total(ms)")
    avg_w = len("Avg(ms)")

    header = f"| {'Resource':<{res_w}} | {'Count':>{cnt_w}} | {'Total(ms)':>{tot_w}} | {'Avg(ms)':>{avg_w}} |"
    sep = f"|{'-' * (res_w + 2)}|{'-' * (cnt_w + 2)}|{'-' * (tot_w + 2)}|{'-' * (avg_w + 2)}|"
    lines = [header, sep]
    for res, count, total, avg in rows:
        lines.append(f"| {res:<{res_w}} | {count:>{cnt_w}} | {total:>{tot_w}.1f} | {avg:>{avg_w}.1f} |")
    return "\n".join(lines)


def _format_trace_tables(grouped: Dict[str, List[Dict[str, Any]]]) -> str:
    if not grouped:
        return "No traces to display."

    parts = []
    for tid, spans in grouped.items():
        service_w = max(len("Service"), max(len(s.get("service", "")) for s in spans))
        res_w = max(len("Resource"), max(len(s.get("resource", "")) for s in spans))
        dur_w = len("Duration(ms)")

        # Datadog APM trace deep link so the user can jump into the UI
        trace_link = f"https://app.datadoghq.com/apm/trace/{tid}"

        header = f"| {'Service':<{service_w}} | {'Resource':<{res_w}} | {'Duration(ms)':>{dur_w}} |"
        sep = f"|{'-' * (service_w + 2)}|{'-' * (res_w + 2)}|{'-' * (dur_w + 2)}|"
        lines = [header, sep]
        for span in spans:
            dur = span.get("duration_ms")
            dur_str = f"{dur:.1f}" if dur is not None else "N/A"
            lines.append(
                f"| {span.get('service',''):<{service_w}} | {span.get('resource',''):<{res_w}} | {dur_str:>{dur_w}} |"
            )
        parts.append(f"Trace {tid}\nLink: {trace_link}\n{header}\n{sep}\n" + "\n".join(lines[2:]))

    return "\n\n".join(parts)


def _format_trace_links(grouped: Dict[str, List[Dict[str, Any]]]) -> str:
    """Render a compact table of trace IDs with their Datadog APM deep links."""
    if not grouped:
        return "No traces to display."

    rows = []
    for tid in grouped.keys():
        rows.append((tid, f"https://app.datadoghq.com/apm/trace/{tid}"))

    tid_w = max(len("Trace ID"), max(len(t[0]) for t in rows))
    url_w = len("Link")

    header = f"| {'Trace ID':<{tid_w}} | {'Link':<{url_w}} |"
    sep = f"|{'-' * (tid_w + 2)}|{'-' * (url_w + 2)}|"
    lines = [header, sep]
    for tid, url in rows:
        lines.append(f"| {tid:<{tid_w}} | {url:<{url_w}} |")

    return "\n".join(lines)


async def handle_call(request: CallToolRequest) -> CallToolResult:
    try:
        args = request.arguments or {}

        resource_name = args.get("resource_name")
        if not resource_name:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: resource_name is required")],
                isError=True,
            )

        # Only resource_name comes from the model; all other parameters stay fixed to defaults
        env = "production"
        service = "lips"
        operation_name = "rack.request"
        min_duration_ms = 1000
        time_from = "now-1d"
        time_to = "now"
        heavy_limit = 100
        trace_limit = 50
        output_format = "table"

        # Step 1: fetch heavy spans to collect trace IDs
        heavy_query_parts = [
            f"env:{env}" if env else "",
            f"service:{service}" if service else "",
            f"operation_name:{operation_name}" if operation_name else "",
            f'resource_name:\"{resource_name}\"',
            f"@duration:>={min_duration_ms}ms",
        ]
        heavy_query = " ".join(part for part in heavy_query_parts if part).strip()

        heavy_resp = await fetch_span_events(
            query=heavy_query,
            time_from=time_from,
            time_to=time_to,
            limit=heavy_limit,
        )

        heavy_spans_raw = heavy_resp.get("data", []) or []
        heavy_spans = [_extract_span(s) for s in heavy_spans_raw]
        trace_ids: List[str] = []
        for s in heavy_spans:
            tid = s.get("trace_id")
            if tid and tid not in trace_ids:
                trace_ids.append(tid)
            if len(trace_ids) >= trace_limit:
                break

        if not trace_ids:
            return CallToolResult(
                content=[TextContent(type="text", text="No heavy spans found for the given resource/time window.")],
                isError=False,
            )

        # Step 2: fetch all spans for those trace IDs
        trace_query = " OR ".join([f"trace_id:{tid}" for tid in trace_ids])
        # Auto-paginate so we don't drop spans when a trace fan-outs past 1000 events.
        trace_resp = await fetch_span_events(
            query=trace_query,
            time_from=time_from,
            time_to=time_to,
            limit=1000,
            auto_paginate=True,
        )

        all_spans_raw = trace_resp.get("data", []) or []
        all_spans = [_extract_span(s) for s in all_spans_raw]

        grouped, resource_stats = _group_top_spans(all_spans)

        if output_format == "json":
            payload = {
                "resource_name": resource_name,
                "heavy_query": heavy_query,
                "trace_query": trace_query,
                "trace_ids": trace_ids,
                "traces": grouped,
                "resource_stats": resource_stats,
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(payload, indent=2))],
                isError=False,
            )

        # Table/text output
        summary = (
            f"Resource '{resource_name}' | heavy spans: {len(heavy_spans)} | traces expanded: {len(grouped)} | "
            f"window: {time_from} -> {time_to}"
        )
        resource_table = _format_resource_table(resource_stats)
        trace_links = _format_trace_links(grouped)
        # trace_tables = _format_trace_tables(grouped)

        final = (
            f"{summary}\n{'=' * len(summary)}\n\n"
            f"Top resources by total duration\n{resource_table}\n\n"
            f"Trace links\n{trace_links}\n\n"
            # f"Top spans per trace\n{trace_tables}"
        )

        return CallToolResult(content=[TextContent(type="text", text=final)], isError=False)

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )
