export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api";

export async function uploadDataset(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/datasets`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}

export async function listDatasets() {
  const res = await fetch(`${API_BASE}/datasets`);
  if (!res.ok) throw new Error("Failed to load datasets");
  return res.json();
}

export async function fetchProfile(datasetId: number) {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/profile`);
  if (!res.ok) throw new Error("Failed to profile dataset");
  return res.json();
}

export async function imputeDataset(datasetId: number) {
  const res = await fetch(`${API_BASE}/impute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId }),
  });
  if (!res.ok) throw new Error("Imputation failed");
  return res.json();
}

export async function synthesizeDataset(datasetId: number, samples: number) {
  const res = await fetch(`${API_BASE}/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, samples }),
  });
  if (!res.ok) throw new Error("Synthesis failed");
  return res.json();
}

export async function fetchMetrics(datasetId: number) {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/metrics`);
  if (!res.ok) throw new Error("Metrics unavailable");
  return res.json();
}

export interface PromptOptimizationRequest {
  goal: string;
  context?: string;
  audience?: string;
  tone?: string;
  output_format?: string;
  constraints?: string;
  role?: string;
  target_agent?: string;
}

export interface PromptOptimizationResponse {
  optimized_prompt: string;
  quality_score: number;
  completeness: number;
  missing_fields: string[];
  optimization_notes: string[];
  clarifying_questions: string[];
  model: string;
}

export async function optimizePrompt(
  payload: PromptOptimizationRequest
): Promise<PromptOptimizationResponse> {
  const res = await fetch(`${API_BASE}/prompts/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let message = "Prompt optimization failed";
    try {
      const data = await res.json();
      if (typeof data?.detail === "string" && data.detail.trim()) {
        message = data.detail;
      }
    } catch {
      // Keep default fallback message if backend error body is not JSON.
    }
    throw new Error(message);
  }

  return res.json();
}
