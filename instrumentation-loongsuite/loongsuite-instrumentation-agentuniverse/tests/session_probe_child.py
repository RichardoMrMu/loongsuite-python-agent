# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run agentUniverse's own session pipeline and report what it produced.

This must run as its own process. ``TelemetryManager.init_from_config`` calls
``trace.set_tracer_provider`` and ``propagate.set_global_textmap``, and both are
one-shot global side effects that must never leak into the pytest process --
the bridge under test is required not to touch them itself. The script prints a
single JSON object on stdout; ``test_instrumentor.py`` asserts its contents.
"""

import json
import os
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_PACKAGE_ROOT), str(_PACKAGE_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
)

from agentuniverse.base.config.application_configer.app_configer import (  # noqa: E402
    AppConfiger,
)
from agentuniverse.base.config.application_configer.application_config_manager import (  # noqa: E402
    ApplicationConfigManager,
)
from agentuniverse.base.tracing.au_trace_manager import (  # noqa: E402
    get_session_id,
    set_session_id,
)
from agentuniverse.base.tracing.otel.telemetry_manager import (  # noqa: E402
    TelemetryManager,
)

from opentelemetry import propagate, trace  # noqa: E402
from opentelemetry.baggage import get_baggage  # noqa: E402

SESSION = "session-abc123"
CARRIER_SESSION = "session-from-carrier"
_INSTRUMENTATION_ROOT = "agentuniverse.base.tracing.otel.instrumentation"

# ``TelemetryManager`` is documented for exactly this: the three native
# instrumentors plus the LoongSuite bridge, wired by class path.
TELEMETRY_CONFIG = {
    "service_name": "agentuniverse-session-probe",
    "processors": [
        {
            "class": "opentelemetry.sdk.trace.export.SimpleSpanProcessor",
            "exporter": {
                "class": "tests.session_probe_support.RecordingSpanExporter"
            },
        }
    ],
    "metric_readers": [
        {"class": "tests.session_probe_support.RecordingMetricReader"}
    ],
    "instrumentations": [
        f"{_INSTRUMENTATION_ROOT}.llm.llm_instrumentor.LLMInstrumentor",
        f"{_INSTRUMENTATION_ROOT}.tool.tool_instrumentor.ToolInstrumentor",
        f"{_INSTRUMENTATION_ROOT}.agent.agent_instrumentor.AgentInstrumentor",
        "opentelemetry.instrumentation.agentuniverse:AgentUniverseInstrumentor",
    ],
}


def _metric_sum(reader: object, name: str) -> object:
    """The recorded value of one metric; native token metrics are histograms."""
    for resource_metrics in reader.get_metrics_data().resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != name:
                    continue
                for point in getattr(metric.data, "data_points", []):
                    value = getattr(point, "sum", None)
                    if value is None:
                        value = getattr(point, "value", None)
                    return value
    return None


def main() -> dict:
    # Imported here so the paths above are already on sys.path, which is also
    # how TelemetryManager will import this module's exporters.
    from tests.conftest import RichAgent, build_agent

    from tests import session_probe_support as support

    ApplicationConfigManager().app_configer = AppConfiger()

    manager = TelemetryManager()
    manager.init_from_config(TELEMETRY_CONFIG)

    # The real session path: the framework's own setter, before the run, so
    # every span the run produces is stamped on start.
    set_session_id(SESSION)
    build_agent("session_agent", RichAgent).run(input="hello")
    trace.get_tracer_provider().force_flush()

    spans = {
        span.name: dict(span.attributes or {})
        for span in support.SPAN_EXPORTERS["exporter"].get_finished_spans()
    }

    carrier: dict = {}
    propagate.inject(carrier)

    # ``extract`` returns a context and also sets the ambient session id, so
    # both the returned context and the recovered id are worth reporting.
    extracted_context = propagate.extract({"AU-SessionId": CARRIER_SESSION})

    return {
        "initialized": manager._initialized,
        "propagator": type(propagate.get_global_textmap()).__name__,
        "propagator_fields": sorted(propagate.get_global_textmap().fields),
        "session_id": get_session_id(),
        "span_names": sorted(spans),
        "session_attributes": {
            name: attributes.get("au.trace.session.id")
            for name, attributes in spans.items()
        },
        "gen_ai_span_kinds": {
            name: attributes.get("gen_ai.span.kind")
            for name, attributes in spans.items()
        },
        "carrier": carrier,
        "extracted_session_id": get_session_id(),
        "extracted_baggage_in_returned_context": get_baggage(
            "AU-SessionId", extracted_context
        ),
        "agent_total_tokens": _metric_sum(
            support.METRIC_READERS["reader"], "agent_total_tokens"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), default=str))
