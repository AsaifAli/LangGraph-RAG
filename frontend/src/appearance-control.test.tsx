// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(cleanup);
import { AppearanceControl } from "./AppearanceControl";

describe("AppearanceControl", () => {
  it("exposes System, Light and Dark as a labelled mutually exclusive control", () => {
    const onChange = vi.fn();
    render(<AppearanceControl value="system" onChange={onChange} />);

    expect(screen.getByRole("group", { name: "Appearance" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Use system appearance" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "Use light appearance" }).getAttribute("aria-pressed")).toBe("false");
    expect(screen.getByRole("button", { name: "Use dark appearance" }).getAttribute("aria-pressed")).toBe("false");
  });

  it.each([
    ["Use system appearance", "system"],
    ["Use light appearance", "light"],
    ["Use dark appearance", "dark"],
  ] as const)("selects %s", (label, expected) => {
    const onChange = vi.fn();
    render(<AppearanceControl value="system" onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: label }));
    expect(onChange).toHaveBeenCalledWith(expected);
  });
});
