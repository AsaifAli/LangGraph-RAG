import { Monitor, Moon, Sun } from "lucide-react";
import type { ThemePreference } from "./theme";

interface AppearanceControlProps {
  value: ThemePreference;
  onChange: (preference: ThemePreference) => void;
}

const choices = [
  { value: "system", label: "Use system appearance", title: "System", Icon: Monitor },
  { value: "light", label: "Use light appearance", title: "Light", Icon: Sun },
  { value: "dark", label: "Use dark appearance", title: "Dark", Icon: Moon },
] as const;

export function AppearanceControl({ value, onChange }: AppearanceControlProps) {
  return (
    <div className="appearance-control" role="group" aria-label="Appearance">
      {choices.map(({ value: choice, label, title, Icon }) => (
        <button
          key={choice}
          type="button"
          className={value === choice ? "appearance-option active" : "appearance-option"}
          aria-label={label}
          aria-pressed={value === choice}
          title={title}
          onClick={() => onChange(choice)}
        >
          <Icon size={14} strokeWidth={1.8} aria-hidden="true" />
        </button>
      ))}
    </div>
  );
}
