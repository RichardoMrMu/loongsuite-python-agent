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

"""Exporter and reader that publish themselves for the session test.

``TelemetryManager`` builds its span exporter and metric reader from class
paths, so those classes have to live in an importable module and hand the
instances they create back to the test. That is all this module is for.
"""

from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

SPAN_EXPORTERS: dict = {}
METRIC_READERS: dict = {}


class RecordingSpanExporter(InMemorySpanExporter):
    """An ``InMemorySpanExporter`` that publishes itself on construction."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        SPAN_EXPORTERS["exporter"] = self


class RecordingMetricReader(InMemoryMetricReader):
    """An ``InMemoryMetricReader`` that publishes itself on construction."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        METRIC_READERS["reader"] = self
