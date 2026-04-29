from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal


class PatientDataCaptured(BaseModel):
    """Represents detailed clinical data captured from the patient during the conversation."""
    chief_complaint : Optional[str] = Field(description="The patient's main reason for seeking care, in their own words.", default=None)
    pain_character : Optional[Literal["pressure", "squeezing", "burning", "sharp", "dull", "other"]] = Field(description="The character of the patient's chest pain.", default=None)
    location : Optional[str] = Field(description="The location of the patient's chest pain.", default=None)
    radiation : Optional[str] = Field(description="Whether the pain radiates to other areas.", default=None)
    severity : Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] = Field(description="The severity of the patient's chest pain on a 0-10 scale.", default=None)
    timing : Optional[Literal["exertional", "rest", "nocturnal"]] = Field(description="When the patient experiences the chest pain.", default=None)
    functional_capacity : Optional[str] = Field(description="The patient's functional capacity if exertional.", default=None)
    relieving_factors : Optional[str] = Field(description="Factors that relieve the patient's chest pain.", default=None)
    dyspnea : Optional[Literal["exertional", "orthopnea"]] = Field(description="Whether the patient has dyspnea.", default=None)
    fatigue : Optional[bool] = Field(description="Whether the patient has fatigue.", default=None)
    palpitations : Optional[bool] = Field(description="Whether the patient has palpitations.", default=None)
    lightheadedness : Optional[bool] = Field(description="Whether the patient has lightheadedness.", default=None)
    syncope : Optional[bool] = Field(description="Whether the patient has syncope.", default=None)
    edema : Optional[bool] = Field(description="Whether the patient has edema.", default=None)
    patient_concerns : Optional[str] = Field(description="The patient's concerns at the end of the conversation.", default=None)


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
    patient_data: PatientDataCaptured = Field(default_factory=PatientDataCaptured)

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
    "scene_1_greeting": Stage(
        stage_id="scene_1_greeting",
        description="Initial greeting, introduce role, explain angiogram, wait for acknowledgment",
        next_stages=["scene_2_open_conversation"]
    ),

    "scene_2_open_conversation": Stage(
        stage_id="scene_2_open_conversation",
        description="Extract patient acknowledgment, ask open-ended question about what's been bothering them",
        next_stages=["scene_3a_pain_character"]
    ),

    "scene_3a_pain_character": Stage(
        stage_id="scene_3a_pain_character",
        description="Extract chief complaint, ask patient to describe the feeling in their chest",
        next_stages=["scene_3b_location_radiation"]
    ),

    "scene_3b_location_radiation": Stage(
        stage_id="scene_3b_location_radiation",
        description="Extract pain character, ask where they feel it and if it moves anywhere",
        next_stages=["scene_3c_severity"]
    ),

    "scene_3c_severity": Stage(
        stage_id="scene_3c_severity",
        description="Extract location and radiation, ask pain severity on 0-10 scale",
        next_stages=["scene_3d_timing"]
    ),

    "scene_3d_timing": Stage(
        stage_id="scene_3d_timing",
        description="Extract severity, ask about timing — exertional, rest, or nocturnal",
        next_stages=["scene_3e_relieving_factors"]
    ),

    "scene_3e_relieving_factors": Stage(
        stage_id="scene_3e_relieving_factors",
        description="Extract timing, ask what makes the pain go away — rest, nitroglycerin, etc.",
        next_stages=["scene_4a_dyspnea"]
    ),

    "scene_4a_dyspnea": Stage(
        stage_id="scene_4a_dyspnea",
        description="Extract relieving factors, ask about shortness of breath",
        next_stages=["scene_4b_fatigue"]
    ),

    "scene_4b_fatigue": Stage(
        stage_id="scene_4b_fatigue",
        description="Extract dyspnea data, ask about fatigue and energy levels",
        next_stages=["scene_4c_palpitations"]
    ),

    "scene_4c_palpitations": Stage(
        stage_id="scene_4c_palpitations",
        description="Extract fatigue data, ask about heart fluttering or racing",
        next_stages=["scene_4d_lightheadedness_syncope"]
    ),

    "scene_4d_lightheadedness_syncope": Stage(
        stage_id="scene_4d_lightheadedness_syncope",
        description="Extract palpitations data, ask about lightheadedness and blackouts",
        next_stages=["scene_4e_edema"]
    ),

    "scene_4e_edema": Stage(
        stage_id="scene_4e_edema",
        description="Extract lightheadedness/syncope data, ask about leg/ankle swelling",
        next_stages=["scene_5_close"]
    ),

    "scene_5_close": Stage(
        stage_id="scene_5_close",
        description=(
            "Extract associated symptom data from Scene 4, close interaction, ask for patient concerns"
        ),
        next_stages=["scene_6_farewell"]
    ),

    "scene_6_farewell": Stage(
        stage_id="scene_6_farewell",
        description=(
            "Extract patient concerns from Scene 5, deliver farewell message, handoff to clinical team"
        ),
        next_stages=["scene_7_summary"]
    ),

    "scene_7_summary": Stage(
        stage_id="scene_7_summary",
        description="Generate a natural language clinical summary from all collected data",
        next_stages=[]
    ),
}