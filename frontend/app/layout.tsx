import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Sahaayak - Professional Medical Assistance",
  description: "Clear next steps when it matters most.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} antialiased h-full`}
    >
      <body className="min-h-full flex flex-col bg-surface-container-lowest text-on-surface font-body">
        {children}
      </body>
    </html>
  );
}
