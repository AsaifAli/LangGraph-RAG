import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const styles = readFileSync(new URL("./styles.css", import.meta.url), "utf8");
const app = readFileSync(new URL("./App.tsx", import.meta.url), "utf8");
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

describe("EvidenceFlow spatial theme system", () => {
  it("uses layered surfaces and restrained gradient lighting instead of flat panels", () => {
    expect(styles).toMatch(/--surface-main:/);
    expect(styles).toMatch(/--surface-raised:/);
    expect(styles).toMatch(/radial-gradient\(ellipse 66% 48%/);
    expect(styles).toMatch(/backdrop-filter:\s*blur\(24px\)/);
  });

  it("defines a dedicated dark palette with readable raised and inset surfaces", () => {
    expect(styles).toMatch(/:root\[data-theme=\"dark\"\]\s*\{/);
    expect(styles).toMatch(/--surface-main:\s*#1b212a/);
    expect(styles).toMatch(/--surface-input:\s*#171d25/);
    expect(styles).toMatch(/:root\[data-theme=\"dark\"\] \.welcome-core/);
  });

  it("uses a neutral graphite foundation in dark mode with green kept as an accent", () => {
    const graphite = styles.split("/* Graphite dark mode — neutral charcoal surfaces, restrained brand accents. */").at(-1) ?? "";
    expect(graphite).toMatch(/--canvas:\s*#101319/);
    expect(graphite).toMatch(/--surface-main:\s*#1b212a/);
    expect(graphite).toMatch(/--surface-raised:\s*#252c36/);
    expect(graphite).toMatch(/linear-gradient\(145deg, #191f29 0%, #11151c 58%, #0e1218 100%\)/);
    expect(graphite).toMatch(/--accent-soft:\s*rgba\(126, 157, 137, \.14\)/);
    expect(graphite).toMatch(/\.sidebar \{[\s\S]*?linear-gradient\(154deg, #202630 0%, #191f28 52%, #141922 100%\)/);
    expect(graphite).not.toContain("#1c2922");
    expect(graphite).not.toContain("rgba(72, 112, 82");
  });

  it("follows the operating system when System appearance is selected", () => {
    expect(styles).toMatch(/@media \(prefers-color-scheme: dark\)/);
    expect(styles).toMatch(/:root\[data-theme=\"system\"\]/);
    expect(html).toContain('root.style.colorScheme = preference === "system" ? "light dark" : preference;');
  });

  it("wires the persistent appearance control into the existing workspace toolbar", () => {
    expect(app).toMatch(/useState<ThemePreference>\(getInitialThemePreference\)/);
    expect(app).toMatch(/applyThemePreference\(themePreference/);
    expect(app).toMatch(/<AppearanceControl value=\{themePreference\} onChange=\{setThemePreference\}/);
    expect(html).toMatch(/localStorage\.getItem\(\"evidenceflow\.theme\"\)/);
  });

  it("keeps nonessential movement short and obeys reduced-motion preferences", () => {
    expect(styles).toMatch(/\.welcome-state \{ animation-duration: \.4s; \}/);
    expect(styles).toMatch(/\.message-row \{ animation-duration: \.24s; \}/);
    expect(styles).toMatch(/@media \(prefers-reduced-motion:\s*reduce\)[\s\S]*?animation-duration:\s*\.01ms/);
  });
});
