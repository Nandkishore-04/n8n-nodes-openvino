'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { FileText, Home, MessagesSquare } from 'lucide-react';
import clsx from 'clsx';

const LINKS = [
  { href: '/', label: 'Home', Icon: Home },
  { href: '/documents', label: 'Documents', Icon: FileText },
  { href: '/chat', label: 'Ask AI', Icon: MessagesSquare },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <header className="z-40 shrink-0 border-b border-white/[0.07] bg-black/70 backdrop-blur-xl">
      <div className="mx-auto flex h-14 w-full max-w-7xl items-center justify-between px-5">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-brand-400 to-brand-600 text-[13px] font-bold text-white shadow-md shadow-brand-500/30">
            OV
          </span>
          <span className="text-sm font-semibold tracking-tight text-white">
            OpenVINO <span className="text-brand-300">AI Suite</span>
          </span>
        </Link>

        <nav className="flex items-center gap-1 rounded-full border border-white/[0.07] bg-white/[0.03] p-1">
          {LINKS.map(({ href, label, Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={clsx(
                  'relative flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs font-medium transition-colors',
                  active ? 'text-white' : 'text-white/45 hover:text-white/85',
                )}
              >
                {active && (
                  <motion.span
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-full bg-brand-500/25 ring-1 ring-inset ring-brand-400/40"
                    transition={{ type: 'spring', bounce: 0.25, duration: 0.5 }}
                  />
                )}
                <Icon size={13} className="relative" />
                <span className="relative hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </nav>

        <span className="hidden items-center gap-1.5 rounded-full border border-white/[0.08] px-2.5 py-1 text-[10px] text-white/40 md:flex">
          <span className="h-1.5 w-1.5 rounded-full bg-brand-400" />
          100% on-device
        </span>
      </div>
    </header>
  );
}
