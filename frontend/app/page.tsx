import Header from "@/components/ui/Header";
import Footer from "@/components/ui/Footer";
import MobileNav from "@/components/ui/MobileNav";
import { Stethoscope, HeartHandshake, Accessibility, Phone } from "lucide-react";
import Link from "next/link";

export default function LandingPage() {
  return (
    <>
      <Header />
      <main className="flex-grow flex flex-col items-center justify-center px-margin-mobile py-xl md:py-[64px] max-w-7xl mx-auto w-full gap-xl">
        <section className="text-center max-w-2xl mx-auto mb-lg">
          <h1 className="font-h1 text-h1 text-on-surface mb-md">Clear next steps when it matters most.</h1>
          <p className="font-body text-body text-outline">Select the area where you need assistance to begin.</p>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-3 gap-gutter w-full max-w-5xl">
          <Link href="/triage" className="bg-surface-container-lowest border border-outline-variant rounded-xl p-lg flex flex-col items-center justify-center text-center shadow-sm hover:border-primary hover:shadow-md transition-all group min-h-[200px]">
            <div className="w-16 h-16 rounded-full bg-primary-fixed flex items-center justify-center mb-md group-hover:scale-105 transition-transform">
              <Stethoscope size={32} className="text-primary" />
            </div>
            <h2 className="font-h3 text-h3 text-on-surface mb-xs">Medical Triage</h2>
            <p className="font-caption text-caption text-outline">Assess symptoms and find care.</p>
          </Link>

          <button className="bg-surface-container-lowest border border-outline-variant rounded-xl p-lg flex flex-col items-center justify-center text-center shadow-sm hover:border-primary hover:shadow-md transition-all group min-h-[200px]">
            <div className="w-16 h-16 rounded-full bg-surface-container-high flex items-center justify-center mb-md group-hover:scale-105 transition-transform">
              <HeartHandshake size={32} className="text-on-surface-variant" />
            </div>
            <h2 className="font-h3 text-h3 text-on-surface mb-xs">Mental Wellness</h2>
            <p className="font-caption text-caption text-outline">Support for emotional health.</p>
          </button>

          <button className="bg-surface-container-lowest border border-outline-variant rounded-xl p-lg flex flex-col items-center justify-center text-center shadow-sm hover:border-primary hover:shadow-md transition-all group min-h-[200px]">
            <div className="w-16 h-16 rounded-full bg-surface-container-high flex items-center justify-center mb-md group-hover:scale-105 transition-transform">
              <Accessibility size={32} className="text-on-surface-variant" />
            </div>
            <h2 className="font-h3 text-h3 text-on-surface mb-xs">Accessibility Help</h2>
            <p className="font-caption text-caption text-outline">Tools for easier navigation.</p>
          </button>
        </section>

        <section className="w-full max-w-sm mx-auto mt-xl text-center">
          <button className="w-full min-h-[48px] bg-secondary hover:bg-secondary-container text-on-error rounded-xl font-label-bold text-label-bold shadow-sm transition-colors flex items-center justify-center gap-2 px-6 py-3 active:scale-95 duration-150">
            <Phone size={20} />
            Emergency Hotline (108/102)
          </button>
        </section>
      </main>
      <Footer />
      <MobileNav />
    </>
  );
}
