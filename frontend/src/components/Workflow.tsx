import { useMemo, useState } from "react";
import StepCard from "./StepCard";
import { optimizePrompt } from "../lib/api";

const quickExamples = [
  "write a short leave request message for tomorrow",
  "plan a simple high-protein vegetarian meal for one day",
  "help me ask interview questions for a frontend developer",
  "explain this topic in very simple language",
];

function buildDraftPrompt(userMessage: string): string {
  const clean = userMessage.trim();
  if (!clean) {
    return "Type your normal message and click Optimize Prompt to get a better final prompt.";
  }

  return [
    "You are a helpful assistant.",
    `Task: ${clean}`,
    "Rewrite this into a clearer, specific, high-quality prompt while preserving intent.",
    "Return the final optimized prompt only.",
  ].join("\n\n");
}

export default function Workflow() {
  const [message, setMessage] = useState("");
  const [optimizedPrompt, setOptimizedPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const draftPrompt = useMemo(() => buildDraftPrompt(message), [message]);

  async function handleOptimize() {
    const cleanMessage = message.trim();
    if (!cleanMessage) {
      setError("Type your message first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await optimizePrompt({
        goal: cleanMessage,
        context: "Convert this normal message into a stronger final prompt.",
        audience: "AI assistant",
        tone: "Clear, natural, and direct",
        output_format: "Optimized prompt only",
        constraints: "Keep original intent, improve clarity and specificity, avoid extra fluff.",
        role: "an expert prompt engineer",
        target_agent: "generic",
      });

      setOptimizedPrompt(result.optimized_prompt);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Prompt optimization failed";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function handleCopy(value?: string) {
    const text = value ?? (optimizedPrompt || draftPrompt);
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  return (
    <section className="grid lg:grid-cols-[1fr_1fr] gap-8">
      <div className="space-y-6">
        <StepCard title="Type Your Message" subtitle="Step 01" tag="Input">
          <p className="text-slate-300">Write your request naturally, exactly how you think.</p>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={6}
            placeholder="Example: make this resume summary stronger for a data analyst role"
            className="mt-3 w-full rounded-2xl border border-ink-700 bg-ink-900/70 px-4 py-3 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-mint-500"
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {quickExamples.map((example) => (
              <button
                key={example}
                onClick={() => setMessage(example)}
                className="px-3 py-2 rounded-full border border-ink-700 text-xs text-slate-200 hover:border-mint-500 hover:text-mint-400 transition"
              >
                {example}
              </button>
            ))}
          </div>
        </StepCard>

        <StepCard title="Generate Final Prompt" subtitle="Step 02" tag="Optimize">
          <div className="flex flex-wrap gap-3 items-center">
            <button
              className="px-4 py-2 rounded-full bg-ember-500 text-ink-950 font-semibold disabled:opacity-60"
              onClick={handleOptimize}
              disabled={loading}
            >
              {loading ? "Optimizing..." : "Optimize Prompt"}
            </button>
          </div>

          {error && (
            <p className="mt-3 rounded-xl border border-coral-500/60 bg-coral-500/10 px-3 py-2 text-xs text-coral-500">
              {error}
            </p>
          )}
        </StepCard>
      </div>

      <div className="space-y-6">
        <StepCard title="Final Optimized Prompt" subtitle="Step 03" tag="Result">
          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-400">Your web app generated this final prompt.</p>
            <button
              className="px-4 py-2 rounded-full bg-mint-500 text-ink-950 font-semibold"
              onClick={() => handleCopy()}
            >
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <textarea
            value={optimizedPrompt || draftPrompt}
            readOnly
            rows={14}
            className="mt-3 w-full rounded-2xl border border-ink-700 bg-ink-950/70 px-4 py-3 text-sm text-slate-100 focus:outline-none"
          />
        </StepCard>
      </div>
    </section>
  );
}
