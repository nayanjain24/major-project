interface MetricPillProps {
  label: string;
  value: string;
}

export default function MetricPill({ label, value }: MetricPillProps) {
  return (
    <div className="glass rounded-xl px-4 py-3">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="text-lg font-display text-white mt-1">{value}</p>
    </div>
  );
}
