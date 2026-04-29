import { Languages, Brain } from "lucide-react";
import Link from "next/link";

export default function Header() {
  return (
    <header className="bg-surface-container-lowest top-0 sticky border-b border-outline-variant/30 shadow-sm z-50">
      <div className="flex justify-between items-center w-full px-6 py-3">
        <div className="flex items-center gap-2">
          <Link href="/" className="text-xl font-bold tracking-tight text-primary">
            Sahaayak
          </Link>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 text-on-surface-variant font-caption text-caption">
            <span>EN | हिंदी | मराठी</span>
          </div>
          <div className="flex items-center gap-3">
            <button className="text-on-surface-variant hover:bg-surface-container transition-colors active:scale-95 duration-150 p-2 rounded-full flex items-center justify-center">
              <Languages size={20} />
            </button>
            <button className="text-on-surface-variant hover:bg-surface-container transition-colors active:scale-95 duration-150 p-2 rounded-full flex items-center justify-center">
              <Brain size={20} />
            </button>
            <div className="hidden md:flex items-center gap-2 border-l border-outline-variant pl-4 ml-2">
              <span className="font-caption text-caption text-outline">Cognitive Load</span>
              <div className="w-10 h-5 bg-surface-variant rounded-full relative cursor-pointer">
                <div className="w-4 h-4 bg-primary rounded-full absolute left-1 top-0.5 shadow-sm"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
