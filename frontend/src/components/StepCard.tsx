import { ReactNode } from "react";

interface StepCardProps {
  title: string;
  subtitle: string;
  tag: string;
  children: ReactNode;
}

export default function StepCard({ title, subtitle, tag, children }: StepCardProps) {
  return (
    <div className="glass step-card rounded-2xl p-6 shadow-card">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-mint-500 font-mono">{tag}</p>
          <h3 className="text-xl font-display mt-2">{title}</h3>
        </div>
        <span className="text-xs px-3 py-1 rounded-full bg-ink-700 text-slate-200">{subtitle}</span>
      </div>
      <div className="mt-4 text-sm text-slate-200 leading-relaxed">{children}</div>
    </div>
  );
}
