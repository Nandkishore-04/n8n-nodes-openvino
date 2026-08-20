import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-sans)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      colors: {
        // OpenVINO brand purple (#6c24f0 — sampled from the official logo)
        brand: {
          300: '#b794f6',
          400: '#9d5cf5',
          500: '#6c24f0',
          600: '#5a1cd0',
          700: '#4a17ab',
        },
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(-4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        blink: {
          '0%, 80%, 100%': { opacity: '0.25', transform: 'scale(0.85)' },
          '40%': { opacity: '1', transform: 'scale(1)' },
        },
        'pulse-ring': {
          '0%': { boxShadow: '0 0 0 0 rgba(108, 36, 240, 0.35)' },
          '70%': { boxShadow: '0 0 0 8px rgba(108, 36, 240, 0)' },
          '100%': { boxShadow: '0 0 0 0 rgba(108, 36, 240, 0)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.25s ease-out',
        'fade-up': 'fade-up 0.3s ease-out',
        blink: 'blink 1.2s infinite ease-in-out',
        'pulse-ring': 'pulse-ring 1.8s infinite',
      },
    },
  },
  plugins: [],
};

export default config;
