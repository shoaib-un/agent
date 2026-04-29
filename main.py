"""
CLI entry point for the Clinician Chatbot.
Runs a conversational while-loop, invoking the LangGraph once per turn.
"""

import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import ClinicianAgentState, STAGES, ExecutionTrace, DecisionContext
from graph import clinician_graph

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_initial_state() -> dict:
    """Create a fresh state dict for a new conversation."""
    state = ClinicianAgentState(
        stages=STAGES,
        trace=ExecutionTrace(),
        context=DecisionContext(),
    )
    return state.model_dump()


def run_conversation():
    """Main conversation loop."""
    print("\n" + "=" * 60)
    print("  🏥  CLINICIAN CHATBOT — Patient Intake Assistant")
    print("=" * 60)
    print("  Type 'quit' or 'exit' to end the conversation.\n")

    # Initialize state
    state = create_initial_state()
    logger.info(f"Initialized conversation state: {state}")

    # --- First turn: greeting (no user input needed) ---
    logger.info("Starting conversation — greeting phase")
    state = clinician_graph.invoke(state)

    if state.get("response"):
        print(f"\n🤖 Clinician: {state['response']}\n")

    # --- Conversation loop ---
    while True:
        # Check if we've reached the terminal stage
        current_stage = state.get("trace", {}).get("current_stage", "")
        if current_stage == "human_review":
            print("\n" + "=" * 60)
            print("  ✅  Session complete — forwarded to clinician for review.")
            print("=" * 60 + "\n")
            break

        # Get user input
        try:
            user_input = input("👤 Patient: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended by user.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nSession ended. Goodbye!")
            break

        # Update state with user input
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=user_input))
        state["messages"] = messages
        state["context"] = {
            **state.get("context", {}),
            "user_input": user_input,
        }

        # Invoke graph for one turn
        logger.info(f"Invoking graph — current stage: {current_stage}")
        try:
            state = clinician_graph.invoke(state)
        except Exception as e:
            logger.error(f"Graph invocation error: {e}", exc_info=True)
            print(f"\n⚠️  Something went wrong: {e}")
            print("Let me try again...\n")
            continue

        # Display the response
        if state.get("response"):
            print(f"\n🤖 Clinician: {state['response']}\n")

        # Log state for debugging
        trace = state.get("trace", {})
        logger.info(
            f"Turn complete — stage: {trace.get('current_stage')} | "
            f"visited: {trace.get('visited_stages')} | "
            f"data keys: {list(state.get('collected_data', {}))}"
        )

    # Print final collected data summary
    collected = state.get("collected_data", {})
    if collected:
        print("\n📊 Data Collected During Session:")
        print("-" * 40)
        for key, value in collected.items():
            if key != "clinical_summary":
                print(f"  {key}: {value}")
        print("-" * 40)


if __name__ == "__main__":
    run_conversation()
