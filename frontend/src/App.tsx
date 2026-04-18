import Workflow from "./components/Workflow";

export default function App() {
  return (
    <div className="min-h-screen bg-signal-grid">
      <header className="px-6 py-10 max-w-6xl mx-auto">
        <div className="glass rounded-3xl px-6 py-8 md:px-10 md:py-10 shadow-glow">
          <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-sand-200">PromptPilot AI</p>
              <h1 className="text-3xl md:text-4xl font-display mt-3">
                Type normally. Get one final optimized prompt.
              </h1>
              <p className="text-sm text-slate-200 mt-3 max-w-2xl">
                Enter your rough message and get a cleaner, stronger prompt generated directly by this app.
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="px-6 pb-16 max-w-6xl mx-auto animate-float-in">
        <Workflow />
      </main>
    </div>
  );
}
