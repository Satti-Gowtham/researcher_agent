from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class InputSchema(BaseModel):
    topic: str = Field(..., title="Topic to research")
    context: Dict[str, Any] = Field(default_factory=dict, title="Current research context")
    round: int = Field(default=1, title="Current research round")
    system_prompt: Optional[str] = Field(default=None, title="Custom system prompt for the agent")
    questions_to_answer: Optional[List[str]] = Field(default=None, title="Specific questions to answer")
    temperature: Optional[float] = Field(default=None, title="Temperature for LLM response")
    max_tokens: Optional[int] = Field(default=None, title="Maximum tokens for LLM response")


class ResearchOutput(BaseModel):
    findings: List[Dict[str, Any]] = Field(default_factory=list, title="Research findings")
    questions: List[str] = Field(default_factory=list, title="Questions for further research")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, title="Additional metadata")
    summary: str = Field(default="", title="Summary of research findings") 