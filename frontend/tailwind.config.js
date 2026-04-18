export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["\"Space Grotesk\"", "sans-serif"],
        body: ["\"DM Sans\"", "sans-serif"],
        mono: ["\"IBM Plex Mono\"", "monospace"],
      },
      colors: {
        ink: {
          950: "#0b0b11",
          900: "#12121a",
          800: "#1d1d28",
          700: "#2c2c3a",
        },
        mint: {
          500: "#3ef0b7",
          600: "#2bd39d",
        },
        coral: {
          500: "#ff7a59",
          600: "#ff5b32",
        },
        sky: {
          500: "#57b7ff",
        },
        sand: {
          200: "#f6e3c1",
          300: "#f0d2a4",
        },
        ember: {
          500: "#f2a35a",
          600: "#e2853a",
        },
      },
      boxShadow: {
        glow: "0 0 40px rgba(242, 163, 90, 0.25)",
        card: "0 20px 40px rgba(8, 10, 20, 0.45)",
      },
      backgroundImage: {
        "signal-grid":
          "radial-gradient(circle at 15% 20%, rgba(87, 183, 255, 0.2), transparent 45%), radial-gradient(circle at 85% 0%, rgba(242, 163, 90, 0.25), transparent 40%), radial-gradient(circle at 70% 80%, rgba(62, 240, 183, 0.18), transparent 40%), linear-gradient(180deg, rgba(11, 11, 17, 0.96), rgba(18, 18, 30, 0.92))",
      },
    },
  },
  plugins: [],
};
