from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request

from app.core.config import settings
from app.schemas.prompt import PromptOptimizationRequest

ALLOWED_FIELDS = {"goal", "context", "audience", "tone", "output_format", "constraints", "role"}


class PromptOptimizerConfigError(RuntimeError):
    pass


class PromptOptimizerUpstreamError(RuntimeError):
    pass


@dataclass
class PromptOptimizationResult:
    optimized_prompt: str
    quality_score: int
    completeness: int
    missing_fields: list[str]
    optimization_notes: list[str]
    clarifying_questions: list[str]
    model: str


class PromptOptimizerService:
    _endpoint = "https://api.openai.com/v1/chat/completions"

    def optimize(self, payload: PromptOptimizationRequest) -> PromptOptimizationResult:
        if not settings.openai_api_key:
            return self._fallback_result(
                payload, "OpenAI API key is not configured. Using local optimization fallback."
            )

        system_prompt = (
            "You are an expert prompt engineer. Rewrite user intent into a high-quality prompt "
            "for general everyday use. Optimize the structure for the selected target agent "
            "(for example ChatGPT, Gemini, Claude). Return only valid JSON with this exact schema: "
            "{"
            '"optimized_prompt": string, '
            '"quality_score": integer 0-100, '
            '"completeness": integer 0-100, '
            '"missing_fields": string[], '
            '"optimization_notes": string[], '
            '"clarifying_questions": string[]'
            "}. "
            "The optimized_prompt must be a final, ready-to-use prompt only. "
            "Do not include meta commentary about prompt engineering. "
            "Rules: missing_fields may only include goal, context, audience, tone, output_format, "
            "constraints, role. Keep optimization_notes short and practical. "
            "If details are complete, return an empty clarifying_questions list."
        )

        user_prompt = (
            f"goal: {payload.goal}\n"
            f"context: {payload.context}\n"
            f"audience: {payload.audience}\n"
            f"tone: {payload.tone}\n"
            f"output_format: {payload.output_format}\n"
            f"constraints: {payload.constraints}\n"
            f"role: {payload.role}\n"
            f"target_agent: {payload.target_agent or 'generic'}\n"
        )

        body = {
            "model": settings.openai_model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            payload_json = self._post_json(body)
            return self._parse_response(payload_json)
        except PromptOptimizerUpstreamError as exc:
            return self._fallback_result(payload, str(exc))

    def _post_json(self, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = request.Request(
            self._endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise PromptOptimizerUpstreamError(
                f"OpenAI API returned HTTP {exc.code}. Detail: {detail}"
            ) from exc
        except error.URLError as exc:
            raise PromptOptimizerUpstreamError(f"Unable to reach OpenAI API: {exc.reason}") from exc

    def _parse_response(self, raw_response: dict) -> PromptOptimizationResult:
        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PromptOptimizerUpstreamError("OpenAI response format was invalid.") from exc

        if not isinstance(content, str):
            raise PromptOptimizerUpstreamError("OpenAI response content was not text JSON.")

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise PromptOptimizerUpstreamError("OpenAI response was not valid JSON.") from exc

        optimized_prompt = str(parsed.get("optimized_prompt", "")).strip()
        if not optimized_prompt:
            raise PromptOptimizerUpstreamError("OpenAI returned an empty optimized prompt.")

        missing_fields = [
            str(item).strip()
            for item in parsed.get("missing_fields", [])
            if str(item).strip() in ALLOWED_FIELDS
        ]

        return PromptOptimizationResult(
            optimized_prompt=optimized_prompt,
            quality_score=self._clamp_int(parsed.get("quality_score"), fallback=60),
            completeness=self._clamp_int(parsed.get("completeness"), fallback=60),
            missing_fields=missing_fields,
            optimization_notes=self._normalize_string_list(parsed.get("optimization_notes")),
            clarifying_questions=self._normalize_string_list(parsed.get("clarifying_questions"), limit=3),
            model=str(raw_response.get("model") or settings.openai_model),
        )

    def _normalize_string_list(self, value: object, limit: int = 6) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text)
            if len(cleaned) >= limit:
                break
        return cleaned

    def _clamp_int(self, value: object, fallback: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = fallback
        return max(0, min(100, number))

    def _fallback_result(self, payload: PromptOptimizationRequest, reason: str) -> PromptOptimizationResult:
        goal = payload.goal.strip()
        context = payload.context.strip()
        audience = payload.audience.strip()
        tone = payload.tone.strip()
        output_format = payload.output_format.strip()
        constraints = payload.constraints.strip()
        role = payload.role.strip()
        target_agent = payload.target_agent.strip() or "Any AI assistant"

        missing_fields: list[str] = []
        if not context:
            missing_fields.append("context")
        if not audience:
            missing_fields.append("audience")
        if not tone:
            missing_fields.append("tone")
        if not output_format:
            missing_fields.append("output_format")
        if not constraints:
            missing_fields.append("constraints")

        clarifying_questions: list[str] = []
        if not context:
            clarifying_questions.append("What background details should the assistant know?")
        if not audience:
            clarifying_questions.append("Who is this response for?")
        if not constraints:
            clarifying_questions.append("Any limits like length, budget, or deadline?")

        lines: list[str] = []
        if role:
            lines.append(f"You are {role}.")
        else:
            lines.append(f"You are a high-precision assistant optimized for {target_agent}.")
        lines.append(f"Task: {goal}")
        if context:
            lines.append(f"Context: {context}")
        if audience:
            lines.append(f"Audience: {audience}")
        if tone:
            lines.append(f"Tone: {tone}")
        else:
            lines.append("Tone: clear, practical, and concise.")
        if constraints:
            lines.append(f"Constraints: {constraints}")
        else:
            lines.append("Constraints: preserve intent and avoid unnecessary filler.")
        if output_format:
            lines.append(f"Output format: {output_format}")
        else:
            lines.append("Output format: optimized prompt only.")

        completeness = self._clamp_int(100 - len(missing_fields) * 15, fallback=40)
        quality_score = self._clamp_int(completeness - 5, fallback=35)

        notes = [
            "Used local optimization fallback due to upstream API unavailability.",
            "Added clearer task framing and explicit output expectations.",
            f"Fallback reason: {reason[:180]}",
        ]

        return PromptOptimizationResult(
            optimized_prompt="\n\n".join(lines),
            quality_score=quality_score,
            completeness=completeness,
            missing_fields=missing_fields,
            optimization_notes=notes,
            clarifying_questions=clarifying_questions[:3],
            model="fallback-local",
        )
