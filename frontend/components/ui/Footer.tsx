import Link from "next/link";

export default function Footer() {
  return (
    <footer className="bg-surface-container w-full mt-auto border-t border-outline-variant/30 hidden md:flex">
      <div className="w-full px-8 py-12 flex flex-col md:flex-row justify-between items-center gap-4">
        <p className="text-sm font-body text-outline">© 2024 Sahaayak AI. Professional Medical Assistance.</p>
        <div className="flex gap-6">
          <Link href="#" className="text-sm font-body text-outline hover:underline">Emergency Support</Link>
          <Link href="#" className="text-sm font-body text-outline hover:underline">Privacy Policy</Link>
          <Link href="#" className="text-sm font-body text-outline hover:underline">Terms of Service</Link>
        </div>
      </div>
    </footer>
  );
}
