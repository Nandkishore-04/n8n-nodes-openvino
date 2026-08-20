import type { Metadata } from 'next';
import './globals.css';
import { Nav } from '@/components/nav';

export const metadata: Metadata = {
  title: 'OpenVINO AI Suite',
  description:
    'Process documents and ask questions about them — running entirely on Intel NPU/GPU/CPU, no cloud.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="flex h-screen flex-col overflow-hidden">
        <Nav />
        <main className="min-h-0 flex-1">{children}</main>
      </body>
    </html>
  );
}
