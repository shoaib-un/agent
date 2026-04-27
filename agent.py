from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage


class PatientHistory(BaseModel):
    """Represents the medical history of a patient."""
    patient_id: str = Field(default="", description="The unique identifier for the patient.")
    patient_name: str = Field(default="", description="The name of the patient.")
    medical_conditions: List[str] = Field(default_factory=list, description="A list of medical conditions the patient has.")
    medications: List[str] = Field(default_factory=list, description="A list of medications the patient is taking.")
    allergies: List[str] = Field(default_factory=list, description="A list of allergies the patient has.")
    past_procedures: List[str] = Field(default_factory=list, description="A list of past medical procedures the patient has undergone.")
    family_history: Dict[str, Any] = Field(default_factory=dict, description="The family medical history of the patient.")


class Stage(BaseModel):
    stage_id: str
    description: str
    next_stages: List[str] = Field(default_factory=list)


class ExecutionTrace(BaseModel):
    visited_stages: List[str] = Field(default_factory=list)
    current_stage: Optional[str] = None
    previous_stage: Optional[str] = None


class DecisionContext(BaseModel):
    user_input: Optional[str] = None
    extracted_intent: Optional[str] = None
    clinical_flags: Dict[str, Any] = Field(default_factory=dict)


class ClinicianAgentState(BaseModel):
    """The full state of the clinician agent, persisted across conversation turns."""
    stages: Dict[str, Stage]
    trace: ExecutionTrace = Field(default_factory=ExecutionTrace)
    context: DecisionContext = Field(default_factory=DecisionContext)
    patient_history: PatientHistory = Field(default_factory=PatientHistory)

    # --- New fields for conversational loop ---
    messages: List[Any] = Field(
        default_factory=list,
        description="Conversation history as LangChain BaseMessage objects."
    )
    response: Optional[str] = Field(
        default=None,
        description="The latest agent response to display to the user."
    )
    collected_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated structured data extracted across all stages."
    )


STAGES = {
    "greet": Stage(
        stage_id="greet",
        description="Greet patient and confirm identity",
        next_stages=["confirm_reason"]
    ),
    "confirm_reason": Stage(
        stage_id="confirm_reason",
        description="Confirm reason for visit from referral/context",
        next_stages=["expand_symptoms"]
    ),
    "expand_symptoms": Stage(
        stage_id="expand_symptoms",
        description="Ask open-ended questions to expand symptoms",
        next_stages=["quantify_severity"]
    ),
    "quantify_severity": Stage(
        stage_id="quantify_severity",
        description="Assess severity (scale, impact, frequency)",
        next_stages=["associated_symptoms"]
    ),
    "associated_symptoms": Stage(
        stage_id="associated_symptoms",
        description="Check for related symptoms (e.g., dyspnea, nausea)",
        next_stages=["timeline"]
    ),
    "timeline": Stage(
        stage_id="timeline",
        description="Establish onset, duration, progression",
        next_stages=["medications"]
    ),
    "medications": Stage(
        stage_id="medications",
        description="Review current medications",
        next_stages=["allergies"]
    ),
    "allergies": Stage(
        stage_id="allergies",
        description="Check known allergies",
        next_stages=["social_history"]
    ),
    "social_history": Stage(
        stage_id="social_history",
        description="Capture smoking, alcohol, lifestyle",
        next_stages=["family_history"]
    ),
    "family_history": Stage(
        stage_id="family_history",
        description="Capture relevant family medical history",
        next_stages=["explain_procedure"]
    ),
    "explain_procedure": Stage(
        stage_id="explain_procedure",
        description="Explain planned procedure or evaluation",
        next_stages=["explain_risks"]
    ),
    "explain_risks": Stage(
        stage_id="explain_risks",
        description="Explain risks and benefits",
        next_stages=["answer_questions"]
    ),
    "answer_questions": Stage(
        stage_id="answer_questions",
        description="Address patient questions",
        next_stages=["consent"]
    ),
    "consent": Stage(
        stage_id="consent",
        description="Obtain informed consent",
        next_stages=["generate_summary"]
    ),
    "generate_summary": Stage(
        stage_id="generate_summary",
        description="Generate structured clinical summary",
        next_stages=["human_review"]
    ),
    "human_review": Stage(
        stage_id="human_review",
        description="Send case for clinician review",
        next_stages=[]
    ),
}