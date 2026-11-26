export type ThemePreference = "system" | "light" | "dark";

export const THEME_STORAGE_KEY = "evidenceflow.theme";

export function parseThemePreference(value: string | null): ThemePreference {
  return value === "light" || value === "dark" || value === "system" ? value : "system";
}

export function getInitialThemePreference(): ThemePreference {
  if (typeof window === "undefined") return "system";
  try {
    return parseThemePreference(window.localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return "system";
  }
}

type ThemeRoot = {
  dataset: DOMStringMap;
  style: { colorScheme: string };
};

type ThemeStorage = Pick<Storage, "setItem">;

export function applyThemePreference(
  preference: ThemePreference,
  root: ThemeRoot,
  storage?: ThemeStorage,
): void {
  root.dataset.theme = preference;
  root.style.colorScheme = preference === "system" ? "light dark" : preference;

  try {
    storage?.setItem(THEME_STORAGE_KEY, preference);
  } catch {
    // Theme selection still works when browser storage is disabled.
  }
}
