import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const styles = readFileSync(new URL("./styles.css", import.meta.url), "utf8");

describe("EvidenceFlow motion system", () => {
  it("gives workspace surfaces and the welcome state a restrained entrance", () => {
    expect(styles).toMatch(/\.research-heading, \.page-view\s*\{[\s\S]*?animation:\s*surface-arrive/);
    expect(styles).toMatch(/\.welcome-state\s*\{[\s\S]*?animation:\s*welcome-arrive/);
    expect(styles).toMatch(/@keyframes\s+surface-arrive/);
    expect(styles).toMatch(/@keyframes\s+welcome-arrive/);
  });

  it("stages chat messages and evidence cards with transform and opacity motion", () => {
    expect(styles).toMatch(/\.message-row\s*\{[\s\S]*?animation:\s*message-arrive/);
    expect(styles).toMatch(/\.evidence-card\s*\{[\s\S]*?animation:\s*card-arrive/);
    expect(styles).toMatch(/@keyframes\s+message-arrive\s*\{[^}]*opacity:[^}]*transform:/);
    expect(styles).toMatch(/@keyframes\s+card-arrive\s*\{[^}]*opacity:[^}]*transform:/);
  });

  it("adds subtle feedback to streaming and opens the mobile evidence drawer", () => {
    expect(styles).toMatch(/\.thinking-status svg\s*\{[\s\S]*?animation:\s*rotate/);
    expect(styles).toMatch(/\.inspector\.inspector-mobile-open\s*\{[\s\S]*?animation:\s*evidence-drawer-in/);
    expect(styles).toMatch(/@keyframes\s+evidence-drawer-in/);
  });

  it("respects reduced-motion preferences across animations and transitions", () => {
    expect(styles).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*?animation-duration:\s*\.01ms/);
    expect(styles).toMatch(/\.inspector\.inspector-mobile-open\s*\{\s*animation:\s*none/);
  });
});
