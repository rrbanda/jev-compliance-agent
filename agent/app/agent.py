# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-border data-policy router — orchestrator and root agent definition.

Pipeline with confidence gating:
  1. Laya (System 1, fast): one Jev /v1/systemone call, five typed
     questions, ~150ms.  If confidence is low, DiffusionGemma 26B is
     called automatically with the same questions for a deeper read.
  2. Application policy: deterministic rules that map calibrated
     probabilities to routing requirements (residency, exclusions,
     human review flags).
  3. Gemini (System 2): reasons about the decision, explains it to the
     user, and routes to the correct regional sub-agent.
"""

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.tools.agent_tool import AgentTool

from .model import get_model
from .prompt import ORCHESTRATOR_INSTRUCTION
from .sub_agents.eu_processor.agent import eu_processor_agent
from .sub_agents.uk_processor.agent import uk_processor_agent
from .sub_agents.us_processor.agent import us_processor_agent
from .tools.laya_tool import classify_with_laya
from .tools.compliance_policy import apply_compliance_policy
from .tools.routing_tool import evaluate_and_route


def create_agent() -> Agent:
    """Creates a fresh, isolated instance of the Agent."""
    return Agent(
        name="root_agent",
        model=get_model(),
        description=(
            "Routes data-processing requests to a jurisdiction-compliant "
            "regional agent.  Uses Laya (fast) and DiffusionGemma (deep) "
            "as Jev decision models with confidence gating, an application "
            "policy for deterministic compliance rules, and Gemini for reasoning."
        ),
        instruction=ORCHESTRATOR_INSTRUCTION,
        tools=[
            classify_with_laya,
            apply_compliance_policy,
            evaluate_and_route,
            AgentTool(agent=eu_processor_agent),
            AgentTool(agent=uk_processor_agent),
            AgentTool(agent=us_processor_agent),
        ],
    )


root_agent = create_agent()

app = App(
    root_agent=root_agent,
    name="app",
)
