import Link from "next/link";
import { ArrowLeft, Languages, Brain, Thermometer, Activity, TriangleAlert, Stethoscope, Send } from "lucide-react";
import MobileNav from "@/components/ui/MobileNav";

export default function MapPage() {
  return (
    <div className="bg-background text-on-background font-body text-body min-h-screen flex flex-col">
      {/* TopAppBar */}
      <header className="bg-surface-container-lowest antialiased docked full-width top-0 sticky border-b border-outline-variant/30 shadow-sm flex justify-between items-center w-full px-6 py-3 z-50">
        <div className="flex items-center gap-4">
          <Link href="/triage" className="text-outline hover:bg-surface-container transition-colors active:scale-95 duration-150 p-2 rounded-full hidden md:flex items-center justify-center">
            <ArrowLeft size={24} />
          </Link>
          <div className="text-xl font-bold tracking-tight text-primary">Sahaayak</div>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-6">
            <Link className="text-outline hover:bg-surface-container transition-colors active:scale-95 duration-150 px-3 py-2 rounded-lg font-label-bold text-label-bold" href="/">Home</Link>
            <Link className="text-outline hover:bg-surface-container transition-colors active:scale-95 duration-150 px-3 py-2 rounded-lg font-label-bold text-label-bold" href="/triage">Triage</Link>
            <Link className="text-primary font-semibold hover:bg-surface-container transition-colors active:scale-95 duration-150 px-3 py-2 rounded-lg font-label-bold text-label-bold" href="/map">Map</Link>
            <Link className="text-outline hover:bg-surface-container transition-colors active:scale-95 duration-150 px-3 py-2 rounded-lg font-label-bold text-label-bold" href="#">Profile</Link>
          </div>
          <div className="text-sm text-outline font-medium hidden md:block">EN | हिंदी</div>
          <button className="text-outline hover:bg-surface-container transition-colors active:scale-95 duration-150 p-2 rounded-full flex items-center justify-center">
            <Languages size={20} />
          </button>
          <button className="text-outline hover:bg-surface-container transition-colors active:scale-95 duration-150 p-2 rounded-full flex items-center justify-center">
            <Brain size={20} />
          </button>
        </div>
      </header>

      {/* Main Canvas */}
      <main className="flex-grow flex flex-col p-margin-mobile md:p-xl gap-xl max-w-5xl mx-auto w-full pb-[100px] md:pb-xl">
        <div className="flex flex-col gap-sm">
          <h1 className="font-h1 text-h1 text-on-surface">AI Reasoning Map</h1>
          <p className="font-body text-body text-outline">Here is how Sahaayak arrived at your recommendation based on the symptoms you shared.</p>
        </div>

        {/* Decision Tree Area */}
        <div className="relative flex flex-col items-center py-lg">
          {/* connecting line */}
          <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-surface-variant -translate-x-1/2 z-0"></div>

          {/* Node 1 */}
          <div className="relative z-10 bg-surface-container-lowest p-md rounded-xl shadow-sm border border-surface-variant w-full max-w-sm mb-xl flex flex-col gap-xs">
            <div className="flex items-center gap-xs text-primary">
              <Thermometer size={20} />
              <span className="font-label-bold text-label-bold uppercase tracking-wide text-xs">Reported Symptom</span>
            </div>
            <div className="font-h3 text-h3 text-on-surface">High Fever & Chills</div>
            <p className="font-caption text-caption text-outline">User reported starting 24 hours ago.</p>
          </div>

          {/* Node 2 */}
          <div className="relative z-10 bg-surface-container-lowest p-md rounded-xl shadow-sm border border-surface-variant w-full max-w-sm mb-xl flex flex-col gap-xs ml-auto md:ml-[50%] md:translate-x-4">
            <div className="flex items-center gap-xs text-tertiary">
              <Activity size={20} />
              <span className="font-label-bold text-label-bold uppercase tracking-wide text-xs">AI Analysis</span>
            </div>
            <div className="font-h3 text-h3 text-on-surface">Duration &gt; 24hrs</div>
            <p className="font-caption text-caption text-outline">Fever persisting over 24 hours with chills indicates potential infection requiring medical review.</p>
          </div>

          {/* Node 3 */}
          <div className="relative z-10 bg-surface-container-lowest p-md rounded-xl shadow-sm border border-surface-variant w-full max-w-sm flex flex-col gap-xs mr-auto md:mr-[50%] md:-translate-x-4">
            <div className="flex items-center gap-xs text-secondary">
              <TriangleAlert size={20} />
              <span className="font-label-bold text-label-bold uppercase tracking-wide text-xs">Risk Assessment</span>
            </div>
            <div className="font-h3 text-h3 text-on-surface">Moderate Risk Level</div>
            <p className="font-caption text-caption text-outline">Not a critical emergency, but requires timely professional consultation.</p>
          </div>
        </div>

        {/* Recommendation Card */}
        <div className="bg-primary-container p-lg rounded-xl shadow-sm border border-primary-fixed-dim flex flex-col gap-md">
          <div className="flex items-center gap-sm">
            <div className="bg-primary text-on-primary w-10 h-10 rounded-full flex items-center justify-center shadow-sm">
              <Stethoscope size={24} />
            </div>
            <h2 className="font-h2 text-h2 text-on-primary-container">Recommended Action Plan</h2>
          </div>
          
          <div className="flex flex-col gap-sm">
            {/* Step 1 */}
            <div className="bg-surface-container-lowest p-md rounded-lg flex items-start gap-sm shadow-sm">
              <div className="bg-surface-variant text-on-surface w-8 h-8 rounded-full flex items-center justify-center font-label-bold text-label-bold flex-shrink-0">1</div>
              <div className="flex flex-col">
                <span className="font-label-bold text-label-bold text-on-surface">Hydrate & Rest</span>
                <span className="font-caption text-caption text-outline">Drink plenty of water. Take paracetamol if fever exceeds 101°F.</span>
              </div>
            </div>

            {/* Step 2 */}
            <div className="bg-surface-container-lowest p-md rounded-lg flex items-start gap-sm shadow-sm">
              <div className="bg-surface-variant text-on-surface w-8 h-8 rounded-full flex items-center justify-center font-label-bold text-label-bold flex-shrink-0">2</div>
              <div className="flex flex-col">
                <span className="font-label-bold text-label-bold text-on-surface">Find a Local Clinic</span>
                <span className="font-caption text-caption text-outline">Visit a general physician today.</span>
              </div>
            </div>

            {/* Step 3 */}
            <div className="bg-surface-container-lowest p-md rounded-lg flex items-start gap-sm shadow-sm">
              <div className="bg-surface-variant text-on-surface w-8 h-8 rounded-full flex items-center justify-center font-label-bold text-label-bold flex-shrink-0">3</div>
              <div className="flex flex-col">
                <span className="font-label-bold text-label-bold text-on-surface">Share Summary</span>
                <span className="font-caption text-caption text-outline">Provide this triage summary to your doctor.</span>
              </div>
            </div>
          </div>
          
          <button className="mt-sm bg-primary text-on-primary hover:bg-on-primary-fixed-variant transition-colors active:scale-95 duration-150 font-label-bold text-label-bold min-h-[48px] rounded-xl flex items-center justify-center gap-xs shadow-sm w-full md:w-auto md:self-end px-lg">
            <Send size={20} />
            Warm Handoff (WhatsApp)
          </button>
        </div>
      </main>

      <MobileNav />

      {/* Footer */}
      <footer className="bg-surface-container text-sm font-['Inter'] text-outline w-full mt-auto border-t border-outline-variant/30 px-8 py-12 flex-col md:flex-row justify-between items-center gap-4 hidden md:flex">
        <div>© 2024 Sahaayak AI. Professional Medical Assistance.</div>
        <div className="flex gap-6">
          <Link className="text-outline hover:underline" href="#">Emergency Support</Link>
          <Link className="text-outline hover:underline" href="#">Privacy Policy</Link>
          <Link className="text-outline hover:underline" href="#">Terms of Service</Link>
        </div>
      </footer>
    </div>
  );
}
