import { describe, expect, it, vi } from "vitest";
import { applyThemePreference, parseThemePreference } from "./theme";

describe("theme preference", () => {
  it.each(["system", "light", "dark"] as const)("accepts the %s preference", (mode) => {
    expect(parseThemePreference(mode)).toBe(mode);
  });

  it("falls back to system for missing or unknown values", () => {
    expect(parseThemePreference(null)).toBe("system");
    expect(parseThemePreference("auto-dark-ish")).toBe("system");
  });

  it("applies and persists a selected theme without replacing other root attributes", () => {
    const root = { dataset: { existing: "kept" } as DOMStringMap, style: { colorScheme: "" } };
    const storage = { setItem: vi.fn() };

    applyThemePreference("dark", root, storage);

    expect(root.dataset).toEqual({ existing: "kept", theme: "dark" });
    expect(root.style.colorScheme).toBe("dark");
    expect(storage.setItem).toHaveBeenCalledWith("evidenceflow.theme", "dark");
  });

  it("lets system mode follow the operating system color scheme", () => {
    const root = { dataset: {} as DOMStringMap, style: { colorScheme: "" } };
    applyThemePreference("system", root, { setItem: vi.fn() });
    expect(root.dataset.theme).toBe("system");
    expect(root.style.colorScheme).toBe("light dark");
  });

  it("still applies the theme when browser storage is unavailable", () => {
    const root = { dataset: {} as DOMStringMap, style: { colorScheme: "" } };
    const storage = { setItem: () => { throw new Error("storage disabled"); } };
    expect(() => applyThemePreference("light", root, storage)).not.toThrow();
    expect(root.dataset.theme).toBe("light");
  });
});
