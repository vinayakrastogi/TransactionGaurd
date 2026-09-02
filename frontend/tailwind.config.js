/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        guard: {
          50:  '#f0f4ff',
          100: '#dde6ff',
          200: '#c3d0ff',
          300: '#9fb1ff',
          400: '#7888ff',
          500: '#5561fa',
          600: '#4248ef',
          700: '#3637d3',
          800: '#2d2fab',
          900: '#2b2f87',
          950: '#1a1c52',
        },
        danger: {
          50:  '#fff1f1',
          100: '#ffe0e0',
          200: '#ffc7c7',
          300: '#ff9f9f',
          400: '#ff6767',
          500: '#ff3636',
          600: '#ef1515',
          700: '#c90d0d',
          800: '#a51010',
          900: '#881414',
        },
        success: {
          500: '#22c55e',
          600: '#16a34a',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'hero-gradient': 'linear-gradient(135deg, #1a1c52 0%, #2d2fab 50%, #4248ef 100%)',
        'card-gradient': 'linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%)',
        'danger-gradient': 'linear-gradient(135deg, #c90d0d 0%, #ef1515 100%)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.4s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(16px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
