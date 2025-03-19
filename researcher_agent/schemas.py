from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class ResearchInput(BaseModel):
    topic: str = Field(..., title="Topic to research")
    context: Dict[str, Any] = Field(default_factory=dict, title="Current research context")
    round: int = Field(default=1, title="Current research round")


class ResearchOutput(BaseModel):
    findings: List[Dict[str, Any]] = Field(default_factory=list, title="Research findings")
    questions: List[str] = Field(default_factory=list, title="Questions for further research")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, title="Additional metadata") 