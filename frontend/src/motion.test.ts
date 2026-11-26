import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const styles = readFileSync(new URL("./styles.css", import.meta.url), "utf8");

describe("EvidenceFlow motion system", () => {
  it("defines restrained entrance motion for workspace views and messages", () => {
    expect(styles).toMatch(/\.view-enter[\s\S]*?animation:/);
    expect(styles).toMatch(/\.message-enter[\s\S]*?animation:/);
    expect(styles).toMatch(/@keyframes\s+ef-enter/);
  });

  it("stages evidence-card entrances without animating layout properties", () => {
    expect(styles).toMatch(/\.evidence-card[\s\S]*?animation:/);
    expect(styles).toMatch(/@keyframes\s+ef-evidence-in/);
    expect(styles).toMatch(/animation-delay:[^;]*var\(--motion-index/);
  });

  it("provides a deliberate streaming state and drawer transition", () => {
    expect(styles).toMatch(/\.streaming-indicator[\s\S]*?animation:/);
    expect(styles).toMatch(/@keyframes\s+ef-stream-pulse/);
    expect(styles).toMatch(/\.inspector-mobile-open[\s\S]*?animation:/);
  });

  it("respects reduced-motion preferences for nonessential animation", () => {
    expect(styles).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*?animation-duration:\s*\.01ms/);
  });
});
