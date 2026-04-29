"""
LangGraph node functions for the Clinician Chatbot.
Each node takes the state as a dict, calls the LLM with structured output,
and returns partial state updates.
"""

import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

from agent import STAGES
from schemas import StageDecision, StageResponse, ClinicalSummary
from prompts import (
    SYSTEM_PROMPT,
    ROUTER_PROMPT,
    STAGE_PROMPTS,
    STAGE_EXECUTOR_PROMPT,
)

logger = logging.getLogger(__name__)
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def _format_conversation(messages: list) -> str:
    """Format message history into a readable string for prompt injection."""
    lines = []
    for msg in (messages or [])[-10:]:
        if isinstance(msg, dict):
            role = msg.get("type", msg.get("role", ""))
            content = msg.get("content", "")
        elif isinstance(msg, (HumanMessage, AIMessage)):
            role = "human" if isinstance(msg, HumanMessage) else "ai"
            content = msg.content
        else:
            continue

        if role in ("human", "user"):
            lines.append(f"Patient: {content}")
        elif role in ("ai", "assistant"):
            lines.append(f"Clinician: {content}")
    return "\n".join(lines) if lines else "(No conversation yet)"


def _format_collected_data(data: Dict[str, Any]) -> str:
    """Format collected data dict into readable string."""
    if not data:
        return "(No data collected yet)"
    lines = [f"- {key}: {value}" for key, value in data.items()]
    return "\n".join(lines)


def _format_all_stages() -> str:
    """Format all stages into a readable list."""
    return "\n".join(
        f"- {sid}: {s.description}" for sid, s in STAGES.items()
    )


def router_node(state: dict) -> dict:
    """
    Uses structured output (StageDecision) to decide which stage to transition to.
    Returns partial state updates for trace.
    """
    trace = state.get("trace", {})
    current_stage = trace.get("current_stage") or "greet"
    visited_stages = trace.get("visited_stages", [])

    current_stage_obj = STAGES.get(current_stage)
    next_stages = current_stage_obj.next_stages if current_stage_obj else []

    context = state.get("context", {})
    user_input = context.get("user_input")

    # Build the router prompt
    prompt_text = ROUTER_PROMPT.format(
        current_stage=current_stage,
        visited_stages=", ".join(visited_stages) or "none",
        next_stages=", ".join(next_stages) or "none (terminal stage)",
        all_stages=_format_all_stages(),
        user_input=user_input or "(initial greeting — no user input yet)",
        conversation_summary=_format_conversation(state.get("messages", [])),
        collected_data=_format_collected_data(state.get("collected_data", {})),
    )

    # Call LLM with structured output using function_calling to bypass strict dict schema validation
    structured_llm = model.with_structured_output(StageDecision, method="function_calling")
    decision: StageDecision = structured_llm.invoke([
        SystemMessage(content="You are a clinical conversation router. Output valid JSON."),
        HumanMessage(content=prompt_text),
    ])

    logger.info(f"Router -> stage: '{decision.stage_id}' (reason: {decision.reasoning})")

    # Validate the stage exists
    selected_stage = decision.stage_id if decision.stage_id in STAGES else current_stage

    # Build updated trace
    visited = list(visited_stages)
    if selected_stage not in visited:
        visited.append(selected_stage)

    new_state = dict(state)
    new_state["trace"] = {
        "current_stage": selected_stage,
        "previous_stage": current_stage,
        "visited_stages": visited,
    }
    return new_state

def stage_executor_node(state: dict) -> dict:
    """
    Uses structured output (StageResponse) to generate the clinician's
    response and extract clinical data from the patient's input.
    """
    trace = state.get("trace", {})
    current_stage = trace.get("current_stage") or "greet"
    stage_obj = STAGES.get(current_stage)
    stage_instructions = STAGE_PROMPTS.get(current_stage, "Proceed with clinical best practices.")

    collected_data = state.get("collected_data", {})

    # Format the patient's name into stage instructions if available
    patient_name = collected_data.get("patient_name", "")
    if patient_name:
        stage_instructions = stage_instructions.replace("{patient_name}", patient_name)
    else:
        stage_instructions = stage_instructions.replace("{patient_name}", "the patient")

    # Build the executor prompt
    prompt_text = STAGE_EXECUTOR_PROMPT.format(
        stage_id=current_stage,
        stage_description=stage_obj.description if stage_obj else "",
        stage_instructions=stage_instructions,
        collected_data=_format_collected_data(collected_data),
        conversation_history=_format_conversation(state.get("messages", [])),
    )

    # Call LLM with structured output using function_calling to bypass strict dict schema validation
    structured_llm = model.with_structured_output(StageResponse, method="function_calling")
    result: StageResponse = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ])
    logger.info(f"Result: {result}")
    logger.info(
        f"Stage '{current_stage}' -> data: {result.data_extracted}, "
        f"complete: {result.stage_complete}"
    )

    # Merge extracted data into collected_data
    updated_data = dict(collected_data)
    if result.data_extracted:
        for item in result.data_extracted:
            updated_data[item.key] = item.value

    # Append the AI response to messages
    updated_messages = list(state.get("messages", []))
    updated_messages.append(AIMessage(content=result.response_to_patient))

    new_state = dict(state)
    new_state["messages"] = updated_messages
    new_state["response"] = result.response_to_patient
    new_state["collected_data"] = updated_data
    return new_state


def summary_node(state: dict) -> dict:
    """
    Generates a natural language clinical summary from collected data only.
    Invoked when current stage is 'scene_7_summary'.
    Does NOT use structured output — just generates a plain text summary.
    """
    collected_data = state.get("collected_data", {})
    stage_instructions = STAGE_PROMPTS.get("scene_7_summary", "")

    prompt_text = f"""{stage_instructions}

PATIENT DATA COLLECTED SO FAR:
{_format_collected_data(collected_data)}

Write the summary now. Use ONLY the data above. Do not add anything else."""

    result = model.invoke([
        SystemMessage(content="You are a clinical documentation assistant. Summarize ONLY the data provided. Do NOT add diagnoses, assessments, plans, or any information not explicitly listed."),
        HumanMessage(content=prompt_text),
    ])

    summary_text = result.content
    logger.info("Clinical summary generated from collected data.")

    updated_messages = list(state.get("messages", []))
    updated_messages.append(AIMessage(content=summary_text))

    new_state = dict(state)
    new_state["messages"] = updated_messages
    new_state["response"] = summary_text
    return new_state