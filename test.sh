uv run python - <<'PY'
import asyncio
from types import SimpleNamespace
from datadog_mcp.tools import analyze_resource_bottlenecks

req = SimpleNamespace(arguments={"resource_name": "Api::ProductsController#index"})
result = asyncio.run(analyze_resource_bottlenecks.handle_call(req))
print(result.content[0].text if result.content else result)
PY