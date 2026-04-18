from pydantic import BaseModel, Field, field_validator


class PromptOptimizationRequest(BaseModel):
    goal: str
    context: str = ""
    audience: str = ""
    tone: str = ""
    output_format: str = ""
    constraints: str = ""
    role: str = ""
    target_agent: str = "generic"

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("goal is required")
        return clean

    @field_validator("context", "audience", "tone", "output_format", "constraints", "role", "target_agent")
    @classmethod
    def strip_optional_fields(cls, value: str) -> str:
        return value.strip()


class PromptOptimizationResponse(BaseModel):
    optimized_prompt: str
    quality_score: int = Field(ge=0, le=100)
    completeness: int = Field(ge=0, le=100)
    missing_fields: list[str] = Field(default_factory=list)
    optimization_notes: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    model: str
