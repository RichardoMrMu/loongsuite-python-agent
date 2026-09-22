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

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import json
import unittest
from unittest.mock import patch

from google.genai import types as genai_types

from opentelemetry._logs import get_logger_provider
from opentelemetry.instrumentation.google_genai import tool_call_wrapper
from opentelemetry.instrumentation.google_genai._compat import TelemetryHandler
from opentelemetry.metrics import get_meter_provider
from opentelemetry.trace import get_tracer_provider

from ..common import otel_mocker


class TestCase(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(
            "os.environ",
            {
                "OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental",
            },
        )
        self.env_patcher.start()
        self._otel = otel_mocker.OTelMocker()
        self._otel.install()
        self._otel_wrapper = TelemetryHandler(
            tracer_provider=get_tracer_provider(),
            logger_provider=get_logger_provider(),
            meter_provider=get_meter_provider(),
        )

    def tearDown(self):
        self._otel.uninstall()
        self.env_patcher.stop()

    @property
    def otel(self):
        return self._otel

    @property
    def otel_wrapper(self):
        return self._otel_wrapper

    def wrap(self, tool_or_tools):
        return tool_call_wrapper.wrapped_tool(tool_or_tools, self.otel_wrapper)

    def test_wraps_none(self):
        result = self.wrap(None)
        self.assertIsNone(result)

    def test_wraps_multiple_tool_functions_as_list(self):
        def somefunction():
            pass

        def otherfunction():
            pass

        wrapped_functions = self.wrap([somefunction, otherfunction])
        wrapped_somefunction = wrapped_functions[0]
        wrapped_otherfunction = wrapped_functions[1]
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        somefunction()
        otherfunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_otherfunction()
        self.otel.assert_has_span_named("execute_tool otherfunction")

    def test_wraps_multiple_tool_functions_as_dict(self):
        def somefunction():
            pass

        def otherfunction():
            pass

        wrapped_functions = self.wrap(
            {"somefunction": somefunction, "otherfunction": otherfunction}
        )
        wrapped_somefunction = wrapped_functions["somefunction"]
        wrapped_otherfunction = wrapped_functions["otherfunction"]
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        somefunction()
        otherfunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        self.otel.assert_does_not_have_span_named("execute_tool otherfunction")
        wrapped_otherfunction()
        self.otel.assert_has_span_named("execute_tool otherfunction")

    def test_wraps_async_tool_function(self):
        async def somefunction():
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        asyncio.run(somefunction())
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        asyncio.run(wrapped_somefunction())
        self.otel.assert_has_span_named("execute_tool somefunction")

    def test_preserves_tool_dict(self):
        tool_dict = genai_types.ToolDict()
        wrapped_tool_dict = self.wrap(tool_dict)
        self.assertEqual(tool_dict, wrapped_tool_dict)

    def test_does_not_have_description_if_no_doc_string(self):
        def somefunction():
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        self.assertNotIn("gen_ai.tool.description", span.attributes)

    def test_has_description_if_doc_string_present(self):
        def somefunction():
            """An example tool call function."""

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction()
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction()
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        self.assertEqual(
            span.attributes["gen_ai.tool.description"],
            "An example tool call function.",
        )

    # Capture content must be enabled to get arguments
    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_AND_EVENT",
        },
    )
    def test_handles_various_arg_types(self):
        def somefunction(
            primitive_int=None,
            dict_arg=None,
            list_arg=None,
            heterogenous_list_arg=None,
        ):
            pass

        wrapped_somefunction = self.wrap(somefunction)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        somefunction(12345)
        self.otel.assert_does_not_have_span_named("execute_tool somefunction")
        wrapped_somefunction(12345, {"key": "value"}, [1, 2, 3], [123, "abc"])
        self.otel.assert_has_span_named("execute_tool somefunction")
        span = self.otel.get_span_named("execute_tool somefunction")
        arguments = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        self.assertEqual(
            arguments["code.function.parameters.primitive_int.type"], "int"
        )
        self.assertEqual(span.attributes["gen_ai.tool.name"], "somefunction")
        self.assertEqual(
            arguments["code.function.parameters.primitive_int.value"], 12345
        )
        self.assertEqual(
            arguments["code.function.parameters.dict_arg.type"], "dict"
        )
        self.assertEqual(
            arguments["code.function.parameters.dict_arg.value"],
            {"key": "value"},
        )
        self.assertEqual(
            arguments["code.function.parameters.list_arg.type"], "list"
        )
        self.assertEqual(
            arguments["code.function.parameters.list_arg.value"], [1, 2, 3]
        )
        self.assertEqual(
            arguments["code.function.parameters.heterogenous_list_arg.type"],
            "list",
        )
        self.assertEqual(
            arguments["code.function.parameters.heterogenous_list_arg.value"],
            [123, "abc"],
        )

    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT",
        },
    )
    def test_with_capture_content_disabled(self):
        def somefunction(arg=None):
            return arg

        wrapped_somefunction = self.wrap(somefunction)
        wrapped_somefunction("a string value")
        span = self.otel.get_span_named("execute_tool somefunction")

        self.assertNotIn(
            "gen_ai.tool.call.arguments",
            span.attributes,
        )
        self.assertNotIn(
            "gen_ai.tool.call.result",
            span.attributes,
        )

    def test_function_that_throws_exception(self):
        def somefunction(arg=None):
            raise Exception("Something went wrong")

        wrapped_somefunction = self.wrap(somefunction)
        try:
            wrapped_somefunction(12345)
        except Exception:
            span = self.otel.get_span_named("execute_tool somefunction")
            self.assertEqual(span.attributes["error.type"], "Exception")

    def test_parallel_tool_calls_share_parent_trace(self):
        # Regression for #38: an agent runs wrapped tools concurrently in a
        # ThreadPoolExecutor. Worker threads do not inherit contextvars, so
        # without context propagation each tool span starts its own root trace
        # instead of joining the active agent span's trace.
        tracer = get_tracer_provider().get_tracer("test-#38")

        def get_weather():
            pass

        def get_stock():
            pass

        with tracer.start_as_current_span("invoke_agent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            wrapped_weather = self.wrap(get_weather)
            wrapped_stock = self.wrap(get_stock)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2
            ) as executor:
                futures = [
                    executor.submit(wrapped_weather),
                    executor.submit(wrapped_stock),
                ]
                for future in futures:
                    future.result()

        weather_span = self.otel.get_span_named("execute_tool get_weather")
        stock_span = self.otel.get_span_named("execute_tool get_stock")
        # Both tool spans must belong to the agent's trace, not new roots.
        self.assertEqual(
            weather_span.context.trace_id,
            parent_trace_id,
            "get_weather tool span started a new trace (context lost across "
            "the executor worker)",
        )
        self.assertEqual(
            stock_span.context.trace_id,
            parent_trace_id,
            "get_stock tool span started a new trace (context lost across "
            "the executor worker)",
        )

    def test_run_in_executor_tool_call_shares_parent_trace(self):
        # Regression for #38 via the asyncio.run_in_executor path named in the
        # issue: the coroutine offloads a sync tool to the default executor.
        tracer = get_tracer_provider().get_tracer("test-#38-async")

        def get_weather():
            pass

        async def drive():
            loop = asyncio.get_event_loop()
            wrapped_weather = self.wrap(get_weather)
            await loop.run_in_executor(None, wrapped_weather)

        with tracer.start_as_current_span("invoke_agent") as parent:
            parent_trace_id = parent.get_span_context().trace_id
            asyncio.run(drive())

        weather_span = self.otel.get_span_named("execute_tool get_weather")
        self.assertEqual(
            weather_span.context.trace_id,
            parent_trace_id,
            "run_in_executor tool span started a new trace (context lost "
            "across the executor worker)",
        )
