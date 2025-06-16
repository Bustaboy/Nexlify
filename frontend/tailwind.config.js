// frontend/tailwind.config.js

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Night City's neon palette - straight from the streets
        cyber: {
          black: '#0a0a0a',
          dark: '#151515',
          darker: '#1f1f1f',
          gray: '#2a2a2a',
          light: '#3a3a3a'
        },
        neon: {
          cyan: '#00ffff',
          magenta: '#ff00ff',
          green: '#00ff00',
          yellow: '#ffff00',
          orange: '#ff6600',
          red: '#ff0000',
          blue: '#0080ff',
          purple: '#9d00ff'
        },
        // Muted variants for when you need to lay low
        muted: {
          cyan: 'rgba(0, 255, 255, 0.6)',
          magenta: 'rgba(255, 0, 255, 0.6)',
          green: 'rgba(0, 255, 0, 0.6)',
          red: 'rgba(255, 0, 0, 0.6)'
        },
        // Glass morphism effects
        glass: {
          light: 'rgba(255, 255, 255, 0.05)',
          medium: 'rgba(255, 255, 255, 0.1)',
          heavy: 'rgba(255, 255, 255, 0.2)'
        }
      },
      fontFamily: {
        'mono': ['Consolas', 'Monaco', 'Courier New', 'monospace'],
        'cyber': ['Orbitron', 'Rajdhani', 'sans-serif'],
        'display': ['Share Tech Mono', 'monospace']
      },
      backgroundImage: {
        // Cyberpunk gradients - each tells a story
        'neon-pulse': 'linear-gradient(45deg, #00ffff, #ff00ff, #00ffff)',
        'corpo-fade': 'linear-gradient(180deg, #0a0a0a 0%, #1f1f1f 100%)',
        'street-glow': 'radial-gradient(circle at center, rgba(0, 255, 255, 0.1) 0%, transparent 70%)',
        'danger-alert': 'radial-gradient(circle at center, rgba(255, 0, 0, 0.2) 0%, transparent 70%)',
        'matrix-rain': `url("data:image/svg+xml,%3Csvg width='40' height='40' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h40v40H0z' fill='%230a0a0a'/%3E%3Cpath d='M0 0l20 20L0 40M20 0l20 20-20 20' stroke='%2300ffff' stroke-opacity='0.05' fill='none'/%3E%3C/svg%3E")`
      },
      boxShadow: {
        'neon-cyan': '0 0 20px rgba(0, 255, 255, 0.5)',
        'neon-magenta': '0 0 20px rgba(255, 0, 255, 0.5)',
        'neon-green': '0 0 20px rgba(0, 255, 0, 0.5)',
        'neon-red': '0 0 20px rgba(255, 0, 0, 0.5)',
        'glass': '0 8px 32px 0 rgba(0, 255, 255, 0.1)',
        'corpo': '0 10px 40px rgba(0, 0, 0, 0.8)'
      },
      animation: {
        'pulse-neon': 'pulse-neon 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glitch': 'glitch 3s infinite',
        'matrix-fall': 'matrix-fall 10s linear infinite',
        'scan-line': 'scan-line 8s linear infinite',
        'flicker': 'flicker 4s linear infinite',
        'chrome-shift': 'chrome-shift 3s ease-in-out infinite'
      },
      keyframes: {
        'pulse-neon': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 }
        },
        'glitch': {
          '0%, 100%': { transform: 'translate(0)' },
          '20%': { transform: 'translate(-2px, 2px)' },
          '40%': { transform: 'translate(-2px, -2px)' },
          '60%': { transform: 'translate(2px, 2px)' },
          '80%': { transform: 'translate(2px, -2px)' }
        },
        'matrix-fall': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' }
        },
        'scan-line': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' }
        },
        'flicker': {
          '0%, 100%': { opacity: 1 },
          '11.11%': { opacity: 0.4 },
          '22.22%': { opacity: 1 },
          '33.33%': { opacity: 0.6 },
          '44.44%': { opacity: 1 },
          '55.55%': { opacity: 0.3 },
          '66.66%': { opacity: 1 },
          '77.77%': { opacity: 0.7 },
          '88.88%': { opacity: 1 }
        },
        'chrome-shift': {
          '0%, 100%': { filter: 'hue-rotate(0deg)' },
          '50%': { filter: 'hue-rotate(180deg)' }
        }
      },
      backdropFilter: {
        'glass': 'blur(10px) saturate(180%)',
        'heavy-glass': 'blur(20px) saturate(200%)'
      },
      transitionTimingFunction: {
        'cyber': 'cubic-bezier(0.25, 0.46, 0.45, 0.94)'
      },
      gridTemplateColumns: {
        'trading': 'minmax(250px, 300px) 1fr minmax(300px, 350px)'
      }
    },
  },
  plugins: [
    // Custom utilities for that extra chrome
    function({ addUtilities, theme }) {
      const newUtilities = {
        '.text-neon-glow': {
          textShadow: '0 0 10px currentColor, 0 0 20px currentColor'
        },
        '.border-neon-glow': {
          borderColor: theme('colors.neon.cyan'),
          boxShadow: `inset 0 0 10px ${theme('colors.neon.cyan')}40, 0 0 10px ${theme('colors.neon.cyan')}40`
        },
        '.cyber-grid': {
          backgroundImage: `linear-gradient(${theme('colors.neon.cyan')}10 1px, transparent 1px), linear-gradient(90deg, ${theme('colors.neon.cyan')}10 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        },
        '.scan-lines': {
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `linear-gradient(transparent 50%, ${theme('colors.neon.cyan')}05 50%)`,
            backgroundSize: '100% 4px',
            pointerEvents: 'none'
          }
        },
        '.glitch-text': {
          position: 'relative',
          '&::before, &::after': {
            content: 'attr(data-text)',
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%'
          },
          '&::before': {
            animation: 'glitch 3s infinite',
            color: theme('colors.neon.cyan'),
            zIndex: -1
          },
          '&::after': {
            animation: 'glitch 3s infinite reverse',
            color: theme('colors.neon.magenta'),
            zIndex: -2
          }
        }
      }
      addUtilities(newUtilities)
    }
  ],
}
