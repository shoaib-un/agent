"""
Structured output schemas for LLM calls.
Each schema is used with `model.with_structured_output(Schema)` to get
type-safe, validated responses from the LLM.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StageDecision(BaseModel):
    """
    Router output: the LLM decides which stage to transition to next.
    Used by the router_node to dynamically select the conversation stage.
    """
    stage_id: str = Field(
        description="The ID of the next stage to transition to. Must be a valid stage_id from the stages dictionary."
    )
    reasoning: str = Field(
        description="Brief explanation of why this stage was selected based on the patient's input and current context."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) for this routing decision."
    )


class StageResponse(BaseModel):
    """
    Stage executor output: the LLM generates a clinician response for
    the current stage and extracts any relevant data from the conversation.
    """
    response_to_patient: str = Field(
        description="The clinician's natural language response to the patient. Should be empathetic, clear, and professional."
    )
    data_extracted: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs of clinical data extracted from the patient's response in this turn. "
                    "Examples: {'symptom': 'chest pain', 'onset': '2 days ago', 'severity': '7/10'}"
    )
    stage_complete: bool = Field(
        default=False,
        description="Whether this stage has gathered all necessary information and is ready to advance."
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="If stage_complete is False, the specific follow-up question embedded in the response."
    )


class ClinicalSummary(BaseModel):
    """
    Final output: structured clinical summary generated at the generate_summary stage.
    """
    patient_name: str = Field(default="", description="Patient's full name")
    chief_complaint: str = Field(default="", description="Primary reason for visit")
    history_of_present_illness: str = Field(default="", description="Detailed HPI narrative")
    symptoms: List[str] = Field(default_factory=list, description="List of reported symptoms")
    severity: Optional[str] = Field(default=None, description="Severity assessment")
    onset_and_duration: Optional[str] = Field(default=None, description="When symptoms started and how long")
    medications: List[str] = Field(default_factory=list, description="Current medications")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    social_history: Optional[str] = Field(default=None, description="Smoking, alcohol, lifestyle")
    family_history: Optional[str] = Field(default=None, description="Relevant family medical history")
    assessment: Optional[str] = Field(default=None, description="Clinical assessment")
    plan: Optional[str] = Field(default=None, description="Recommended plan/next steps")