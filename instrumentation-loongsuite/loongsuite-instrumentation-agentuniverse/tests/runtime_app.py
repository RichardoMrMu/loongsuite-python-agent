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

"""Real native tool loop and SDK parsing; only the HTTP provider is scripted."""

import asyncio
import threading
import time
from collections import Counter
from queue import Queue
from typing import Any, ClassVar

import httpx
from agentuniverse.agent.action.tool.tool import Tool
from agentuniverse.agent.action.tool.tool_manager import ToolManager
from agentuniverse.agent.agent import Agent
from agentuniverse.agent.agent_model import AgentModel
from agentuniverse.agent.memory.enum import ChatMessageEnum
from agentuniverse.agent.memory.message import Message
from agentuniverse.ai_context.agent_context import AgentContext
from agentuniverse.llm.llm import LLM
from agentuniverse.llm.llm_channel.llm_channel import LLMChannel
from openai import AsyncOpenAI, OpenAI

from opentelemetry import context

from .runtime_provider import MODEL, Provider, answer, prompt


class ChannelLLM(LLM):
    def _call(self, *args, **kwargs):
        raise AssertionError("The real LLM channel should handle the call")

    async def _acall(self, *args, **kwargs):
        raise AssertionError("The real LLM channel should handle the call")

    def get_num_tokens(self, text):
        return len(text.split())


class InventoryTool(Tool):
    workload: ClassVar[Any] = None

    def execute(self, **kwargs):
        work = self.workload
        with work.lock:
            work.active += 1
            work.max_active = max(work.max_active, work.active)
        try:
            time.sleep(0.008)
            value = (
                7
                if self.name == "stock_lookup"
                else 12
                if self.name == "unit_price"
                else kwargs["a"] * kwargs["b"]
            )
            work.tool_calls.append(
                {"name": self.name, "args": kwargs, "result": value}
            )
            return str(value)
        finally:
            with work.lock:
                work.active -= 1


class InventoryAgent(Agent):
    history: list = []
    configured_llm: Any = None
    stream_queue: Any = None
    schemas: list = []
    observed_reasoning: list = []

    def input_keys(self):
        return ["input"]

    def output_keys(self):
        return ["output"]

    def parse_input(self, input_object, agent_input):
        return agent_input

    def parse_result(self, agent_result):
        return agent_result

    def make_context(self, input_object):
        return AgentContext(
            agent_id=self.agent_model.info["name"],
            session_id=input_object.get_data("session_id"),
            agent_model=self.agent_model,
            tools_schema=self.schemas,
            chat_history=list(self.history),
            output_stream=self.stream_queue,
            current_messages=[
                Message(
                    type=ChatMessageEnum.USER,
                    content=input_object.get_data("input"),
                )
            ],
        )

    def finish(self, current, output):
        self.observed_reasoning.append(output.reasoning_text)
        self.history = (
            current.chat_history + current.current_messages + [output.message]
        )
        return {"output": output.text}

    def execute(self, input_object, agent_input):
        current = self.make_context(input_object)
        output = self.run_tool_calling_loop(
            self.configured_llm, current, input_object, max_iterations=4
        )
        return self.finish(current, output)

    async def async_execute(self, input_object, agent_input):
        current = self.make_context(input_object)
        output = await self.async_run_tool_calling_loop(
            self.configured_llm, current, input_object, max_iterations=4
        )
        return self.finish(current, output)


class Workload:
    def __init__(self):
        InventoryTool.workload = self
        self.backend = Provider()
        self.tool_calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.schemas = []
        for name, fields in [
            ("stock_lookup", {"sku": {"type": "string"}}),
            ("unit_price", {"sku": {"type": "string"}}),
            ("multiply", {"a": {"type": "integer"}, "b": {"type": "integer"}}),
        ]:
            properties = dict(
                worker={"type": "string"}, turn={"type": "integer"}, **fields
            )
            tool = InventoryTool(
                name=name,
                description="Synthetic inventory tool",
                input_keys=list(properties),
            )
            ToolManager().register(tool.get_instance_code(), tool)
            self.schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "Synthetic inventory tool",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": list(properties),
                        },
                    },
                }
            )

    def agent(self, label, streaming):
        channel = LLMChannel(
            channel_name="openai",
            channel_model_name=MODEL,
            channel_api_base="http://provider.invalid/v1",
            ext_params={
                "reasoning_effort": "high",
                "stream_options": {"include_usage": True},
            },
        )
        channel.channel_model_config = {
            "model": MODEL,
            "streaming": streaming,
            "temperature": 0,
        }
        channel.client = OpenAI(
            api_key="synthetic",
            base_url="http://provider.invalid/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(self.backend.respond)
            ),
        )
        channel.async_client = AsyncOpenAI(
            api_key="synthetic",
            base_url="http://provider.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(self.backend.arespond)
            ),
        )
        llm = ChannelLLM(
            name="inventory_llm", model_name=MODEL, streaming=streaming
        )
        llm._channel_instance = channel
        agent = InventoryAgent()
        agent.agent_model = AgentModel(info={"name": "inventory_" + label})
        agent.configured_llm = llm
        agent.stream_queue = Queue()
        agent.history = []
        agent.observed_reasoning = []
        agent.schemas = self.schemas
        return agent

    async def run(self, tracer_provider):
        records = []

        async def worker(label, streaming, synchronous=False):
            agent = self.agent(label, streaming)
            with tracer_provider.get_tracer("app").start_as_current_span(
                "au-session." + label
            ):
                before = context.get_current()
                for turn in (1, 2):
                    kwargs = dict(
                        input=prompt(label, turn),
                        session_id=label,
                        output_stream=agent.stream_queue,
                    )
                    output = (
                        agent.run(**kwargs)
                        if synchronous
                        else await agent.async_run(**kwargs)
                    )
                    text = output.get_data("output")
                    assert text == answer(label, turn)
                    assert context.get_current() is before
                    records.append(
                        {
                            "worker": label,
                            "turn": turn,
                            "streaming": streaming,
                            "text": text,
                            "reasoning": agent.observed_reasoning[-1],
                        }
                    )
                if streaming:
                    assert agent.stream_queue.qsize() >= 4
                agent.configured_llm._channel_instance.client.close()
                await (
                    agent.configured_llm._channel_instance.async_client.close()
                )

        await worker("auSync", False, True)
        await asyncio.gather(
            worker("auStreamA", True), worker("auStreamB", True)
        )
        counts = Counter(
            (item["args"]["worker"], item["args"]["turn"], item["name"])
            for item in self.tool_calls
        )
        for label in ("auSync", "auStreamA", "auStreamB"):
            for turn in (1, 2):
                for name in ("stock_lookup", "unit_price", "multiply"):
                    assert counts[label, turn, name] == 1
        assert self.max_active >= 2
        return dict(
            self.backend.verify(),
            scenarios=records,
            tool_calls=self.tool_calls,
            max_active_tools=self.max_active,
        )
