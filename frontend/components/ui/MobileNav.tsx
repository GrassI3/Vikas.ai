import Link from "next/link";
import { Home, MessageCircle, Network, User } from "lucide-react";

export default function MobileNav() {
  return (
    <nav className="md:hidden bg-surface-container-lowest text-caption font-caption fixed bottom-0 w-full rounded-t-xl border-t border-outline-variant/30 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-50 flex justify-around items-center px-4 pb-safe pt-2 backdrop-blur-md">
      <Link href="/" className="flex flex-col items-center justify-center text-outline hover:text-primary touch-manipulation cursor-pointer min-h-[48px] flex-1">
        <Home size={24} className="mb-1" />
        <span>Home</span>
      </Link>
      <Link href="/triage" className="flex flex-col items-center justify-center text-primary bg-primary-fixed rounded-xl px-3 py-1 hover:text-primary touch-manipulation cursor-pointer min-h-[48px] flex-1">
        <MessageCircle size={24} className="mb-1" />
        <span>Triage</span>
      </Link>
      <Link href="/map" className="flex flex-col items-center justify-center text-outline hover:text-primary touch-manipulation cursor-pointer min-h-[48px] flex-1">
        <Network size={24} className="mb-1" />
        <span>Map</span>
      </Link>
      <Link href="#" className="flex flex-col items-center justify-center text-outline hover:text-primary touch-manipulation cursor-pointer min-h-[48px] flex-1">
        <User size={24} className="mb-1" />
        <span>Profile</span>
      </Link>
    </nav>
  );
}
