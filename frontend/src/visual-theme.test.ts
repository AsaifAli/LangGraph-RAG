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
    expect(styles).toMatch(/--surface-main:\s*rgba\(26, 38, 31/);
    expect(styles).toMatch(/--surface-input:\s*rgba\(20, 32, 25/);
    expect(styles).toMatch(/:root\[data-theme=\"dark\"\] \.welcome-core/);
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
