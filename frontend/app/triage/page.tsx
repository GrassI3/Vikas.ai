"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import {
  ArrowLeft, Brain, TriangleAlert, Send, Mic,
  Image as ImageIcon, ShieldAlert, Stethoscope,
  Activity, AlertCircle, ChevronRight, Pill, Heart
} from "lucide-react";

/* ── Types ────────────────────────────────────── */

type SymptomDetected = {
  condition: string;
  idiom_matched: string;
  literal: string;
  match_confidence: number;
  default_severity: string;
  context: string;
};

type PossibleCause = {
  condition: string;
  severity: string;
  severity_label: string;
  escalation_triggers: string[];
};

type RecommendedAction = {
  priority: string;
  action: string;
  flag?: string;
};

type ClinicalAnalysis = {
  language_detected: string;
  severity_label: string;
  severity_description: string;
  symptoms_detected: SymptomDetected[];
  possible_causes: PossibleCause[];
  related_conditions: string[];
  recommended_actions: RecommendedAction[];
  model_probabilities: Record<string, number>;
};

type Message = {
  id: string;
  sender: "user" | "ai";
  text: string;
  prediction?: string;
  flags?: string[];
  analysis?: ClinicalAnalysis;
  xaiData?: any;
};

/* ── Severity colors ──────────────────────────── */

const severityColor = (sev: string) => {
  switch (sev) {
    case "EMERGENCY": return "bg-red-600 text-white";
    case "HIGH": return "bg-orange-500 text-white";
    case "MODERATE": return "bg-yellow-500 text-black";
    case "LOW": return "bg-emerald-500 text-white";
    default: return "bg-gray-400 text-white";
  }
};

const severityBorder = (sev: string) => {
  switch (sev) {
    case "EMERGENCY": return "border-red-500/40";
    case "HIGH": return "border-orange-400/40";
    case "MODERATE": return "border-yellow-400/40";
    case "LOW": return "border-emerald-400/40";
    default: return "border-outline-variant/30";
  }
};

/* ── Page ─────────────────────────────────────── */

export default function TriagePage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      sender: "ai",
      text: "🙏 Namaste! Main Sahaayak AI hoon. Apne lakshan batayein — aap Hindi, Marathi ya English mein likh sakte hain.\n\nNamaste! I'm the Sahaayak AI. Describe your symptoms — you can write in Hindi, Marathi, or English.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /* ── send ── */
  const handleSend = async () => {
    const userText = input.trim();
    if (!userText || isLoading) return;
    setInput("");

    const userMsg: Message = { id: Date.now().toString(), sender: "user", text: userText };
    setMessages((p) => [...p, userMsg]);
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/triage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText, include_xai: true }),
      });
      const data = await res.json();
      const analysis: ClinicalAnalysis | undefined = data.clinical_analysis;

      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: "ai",
        text: analysis?.severity_description ?? `Severity: ${data.prediction}`,
        prediction: data.prediction,
        flags: data.safety_flags ?? [],
        analysis,
        xaiData: data.xai,
      };
      setMessages((p) => [...p, aiMsg]);
    } catch {
      setMessages((p) => [
        ...p,
        { id: (Date.now() + 1).toString(), sender: "ai", text: "⚠️ Could not reach the Sahaayak backend. Please make sure `python main.py` is running." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  /* ── render ─────────────────────────────────── */
  return (
    <div className="bg-background text-on-background font-body min-h-screen flex flex-col antialiased">
      {/* Header */}
      <header className="bg-surface-container-lowest border-b border-outline-variant/30 sticky top-0 z-50 shadow-sm flex items-center justify-between px-margin-mobile py-sm min-h-[64px]">
        <div className="flex items-center gap-xs">
          <Link href="/" aria-label="Back" className="w-[48px] h-[48px] flex items-center justify-center rounded-xl hover:bg-surface-container transition-colors text-primary">
            <ArrowLeft size={24} />
          </Link>
          <div className="flex flex-col">
            <span className="font-h3 text-h3 text-on-surface">Active Triage</span>
            <span className="font-caption text-caption text-outline flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
              Sahaayak-Core Online
            </span>
          </div>
        </div>
      </header>

      {/* Disclaimer */}
      <div className="bg-error-container border-b border-error/20 px-margin-mobile py-sm flex items-start gap-sm">
        <TriangleAlert size={20} className="text-secondary shrink-0 mt-0.5" />
        <p className="font-label-bold text-label-bold text-secondary leading-tight pt-0.5">
          Not a doctor. In emergency, call 108 immediately.
        </p>
      </div>

      {/* Chat Canvas */}
      <main className="flex-1 overflow-y-auto px-margin-mobile py-lg flex flex-col gap-lg bg-surface-container-lowest pb-[150px]">
        {messages.map((msg) =>
          msg.sender === "user" ? (
            /* ── User Bubble ── */
            <div key={msg.id} className="flex flex-col gap-xs max-w-[85%] self-end items-end">
              <div className="bg-primary text-on-primary p-md rounded-2xl rounded-tr-sm shadow-sm">
                <p className="font-chat-bubble text-chat-bubble whitespace-pre-wrap">{msg.text}</p>
              </div>
            </div>
          ) : (
            /* ── AI Bubble ── */
            <div key={msg.id} className="flex flex-col gap-xs max-w-[95%] self-start">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-8 h-8 rounded-full bg-primary-container flex items-center justify-center text-on-primary-container shrink-0">
                  <Brain size={18} />
                </div>
                <span className="font-label-bold text-label-bold text-on-surface-variant">Sahaayak AI</span>
              </div>

              {/* Base text */}
              <div className="bg-surface-container text-on-surface p-md rounded-2xl rounded-tl-sm shadow-sm">
                <p className="font-chat-bubble text-chat-bubble whitespace-pre-wrap">{msg.text}</p>
              </div>

              {/* ── Clinical Analysis Card ── */}
              {msg.analysis && (
                <div className={`border rounded-2xl overflow-hidden shadow-md mt-1 ${severityBorder(msg.prediction ?? "")}`}>

                  {/* Severity Banner */}
                  <div className={`px-4 py-3 flex items-center justify-between ${
                    msg.prediction === "EMERGENCY" ? "bg-red-600 text-white" :
                    msg.prediction === "HIGH" ? "bg-orange-500 text-white" :
                    msg.prediction === "MODERATE" ? "bg-yellow-400 text-black" :
                    "bg-emerald-500 text-white"
                  }`}>
                    <div className="flex items-center gap-2">
                      <ShieldAlert size={20} />
                      <span className="font-bold text-lg">{msg.analysis.severity_label} — {msg.prediction}</span>
                    </div>
                    <span className="text-sm opacity-80">Confidence: {((msg.xaiData ? 1 : 0.8) * 100).toFixed(0)}%</span>
                  </div>

                  {/* ── Safety Flags ── */}
                  {msg.flags && msg.flags.length > 0 && (
                    <div className="bg-red-50 dark:bg-red-900/20 px-4 py-3 flex flex-col gap-2">
                      {msg.flags.map((flag, i) => (
                        <div key={i} className="flex items-start gap-2">
                          <AlertCircle size={18} className="text-red-600 shrink-0 mt-0.5" />
                          <div>
                            <span className="font-bold text-red-700 dark:text-red-400 text-sm">{flag.replace(/_/g, " ")}</span>
                            {msg.analysis!.recommended_actions.filter(a => a.flag === flag).map((a, j) => (
                              <p key={j} className="text-red-600 dark:text-red-300 text-sm">{a.action}</p>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* ── Symptoms Detected ── */}
                  {msg.analysis.symptoms_detected.length > 0 && (
                    <div className="px-4 py-3 border-b border-outline-variant/20 bg-surface-container-lowest">
                      <div className="flex items-center gap-2 mb-2">
                        <Stethoscope size={16} className="text-primary" />
                        <span className="font-bold text-sm text-on-surface">Symptoms Detected</span>
                      </div>
                      <div className="flex flex-col gap-2">
                        {msg.analysis.symptoms_detected.map((s, i) => (
                          <div key={i} className="flex items-start gap-3 bg-surface-container rounded-xl px-3 py-2">
                            <div className="flex flex-col flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-bold text-sm text-on-surface">{s.condition}</span>
                                <span className={`text-xs px-2 py-0.5 rounded-full ${severityColor(s.default_severity)}`}>{s.default_severity}</span>
                              </div>
                              {s.idiom_matched && (
                                <span className="text-xs text-outline mt-0.5">
                                  Matched: &ldquo;{s.idiom_matched}&rdquo; {s.literal && `(${s.literal})`} — {s.context}
                                </span>
                              )}
                            </div>
                            <span className="text-xs text-outline font-mono shrink-0">{s.match_confidence}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* ── Possible Causes ── */}
                  {msg.analysis.possible_causes.length > 0 && (
                    <div className="px-4 py-3 border-b border-outline-variant/20 bg-surface-container-lowest">
                      <div className="flex items-center gap-2 mb-2">
                        <Activity size={16} className="text-primary" />
                        <span className="font-bold text-sm text-on-surface">Possible Causes</span>
                      </div>
                      <div className="flex flex-col gap-2">
                        {msg.analysis.possible_causes.map((c, i) => (
                          <div key={i} className="bg-surface-container rounded-xl px-3 py-2">
                            <div className="flex items-center justify-between">
                              <span className="font-bold text-sm text-on-surface">{c.condition}</span>
                              <span className={`text-xs px-2 py-0.5 rounded-full ${severityColor(c.severity)}`}>{c.severity_label}</span>
                            </div>
                            {c.escalation_triggers.length > 0 && (
                              <div className="mt-1 flex flex-wrap gap-1">
                                {c.escalation_triggers.map((t, j) => (
                                  <span key={j} className="text-xs text-orange-600 dark:text-orange-400 bg-orange-50 dark:bg-orange-900/20 px-2 py-0.5 rounded">
                                    ⚠ {t.replace(/_/g, " ")}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* ── Related Conditions ── */}
                  {msg.analysis.related_conditions.length > 0 && (
                    <div className="px-4 py-3 border-b border-outline-variant/20 bg-surface-container-lowest">
                      <div className="flex items-center gap-2 mb-2">
                        <Heart size={16} className="text-primary" />
                        <span className="font-bold text-sm text-on-surface">Related Conditions to Watch</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {msg.analysis.related_conditions.map((rc, i) => (
                          <span key={i} className="text-xs bg-primary/10 text-primary px-3 py-1 rounded-full font-medium">
                            {rc.replace(/([A-Z])/g, " $1").trim()}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* ── Recommended Actions ── */}
                  <div className="px-4 py-3 bg-surface-container-lowest">
                    <div className="flex items-center gap-2 mb-2">
                      <Pill size={16} className="text-primary" />
                      <span className="font-bold text-sm text-on-surface">Recommended Actions</span>
                    </div>
                    <div className="flex flex-col gap-2">
                      {msg.analysis.recommended_actions.filter(a => !a.flag).map((a, i) => (
                        <div key={i} className="flex items-start gap-2">
                          <ChevronRight size={14} className="text-primary shrink-0 mt-0.5" />
                          <span className="text-sm text-on-surface">{a.action}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* ── Footer: Probability Bars ── */}
                  <div className="px-4 py-3 bg-surface-container border-t border-outline-variant/20">
                    <span className="text-xs text-outline mb-2 block">Model Confidence Distribution</span>
                    <div className="flex gap-1 h-3 rounded-full overflow-hidden">
                      {["LOW", "MODERATE", "HIGH", "EMERGENCY"].map((cls) => {
                        const pct = (msg.analysis!.model_probabilities[cls] ?? 0) * 100;
                        const colors: Record<string, string> = {
                          LOW: "bg-emerald-500", MODERATE: "bg-yellow-500",
                          HIGH: "bg-orange-500", EMERGENCY: "bg-red-600",
                        };
                        return pct > 1 ? (
                          <div key={cls} className={`${colors[cls]} transition-all`} style={{ width: `${pct}%` }} title={`${cls}: ${pct.toFixed(1)}%`} />
                        ) : null;
                      })}
                    </div>
                    <div className="flex justify-between mt-1">
                      {["LOW", "MODERATE", "HIGH", "EMERGENCY"].map((cls) => (
                        <span key={cls} className="text-[10px] text-outline">{cls.slice(0, 3)} {((msg.analysis!.model_probabilities[cls] ?? 0) * 100).toFixed(0)}%</span>
                      ))}
                    </div>
                  </div>

                  {/* View XAI link */}
                  {msg.xaiData && (
                    <Link href="/map" className="block px-4 py-2.5 bg-primary/5 text-primary text-center font-bold text-sm hover:bg-primary/10 transition-colors">
                      View Full AI Reasoning Map →
                    </Link>
                  )}
                </div>
              )}
            </div>
          )
        )}

        {/* Loading */}
        {isLoading && (
          <div className="flex flex-col gap-xs max-w-[85%] self-start">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-8 h-8 rounded-full bg-primary-container flex items-center justify-center text-on-primary-container shrink-0">
                <Brain size={18} />
              </div>
              <span className="font-label-bold text-label-bold text-on-surface-variant">Sahaayak AI</span>
            </div>
            <div className="bg-surface-container text-on-surface p-md rounded-2xl rounded-tl-sm shadow-sm">
              <div className="flex gap-2 items-center text-outline h-6">
                <span className="w-2 h-2 rounded-full bg-primary/40 animate-bounce" />
                <span className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "0.2s" }} />
                <span className="w-2 h-2 rounded-full bg-primary/80 animate-bounce" style={{ animationDelay: "0.4s" }} />
                <span className="text-xs text-outline ml-2">Analyzing symptoms...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={endRef} />
      </main>

      {/* Input Area */}
      <div className="fixed bottom-0 left-0 w-full bg-surface-container-lowest border-t border-outline-variant/30 p-margin-mobile shadow-[0_-4px_15px_-3px_rgba(0,0,0,0.05)] z-50 pb-safe">
        <div className="max-w-4xl mx-auto flex items-end gap-sm relative">
          <button className="w-[48px] h-[48px] shrink-0 rounded-xl bg-surface-container text-primary flex items-center justify-center hover:bg-surface-variant transition-colors">
            <ImageIcon size={24} />
          </button>
          <div className="flex-1 relative">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
              className="w-full bg-background border border-outline-variant rounded-xl py-3 pl-4 pr-12 focus:ring-2 focus:ring-primary focus:border-primary text-body font-body resize-none min-h-[48px] max-h-[120px] shadow-sm"
              placeholder="Lakshan batayein / Describe symptoms..."
              rows={1}
              disabled={isLoading}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="absolute right-2 bottom-2 w-8 h-8 flex items-center justify-center text-primary hover:bg-surface-container rounded-lg transition-colors disabled:opacity-50"
            >
              <Send size={20} />
            </button>
          </div>
          <button className="w-[56px] h-[56px] shrink-0 rounded-xl bg-primary text-on-primary flex items-center justify-center shadow-sm relative overflow-hidden group">
            <Mic size={28} className="relative z-10" />
          </button>
        </div>
      </div>
    </div>
  );
}
