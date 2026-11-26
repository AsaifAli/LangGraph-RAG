import { describe, expect, it } from "vitest";
import { decodeSseBlock } from "./api";

describe("decodeSseBlock", () => {
  it("decodes JSON event payloads", () => {
    expect(decodeSseBlock('event: token\ndata: {"text":"hello"}')).toEqual({
      event: "token",
      data: { text: "hello" },
    });
  });

  it("joins multi-line data fields before parsing JSON", () => {
    expect(decodeSseBlock('event: final\ndata: {"answer":\ndata: "verified"}')).toEqual({
      event: "final",
      data: { answer: "verified" },
    });
  });

  it("ignores heartbeat/comment-only blocks", () => {
    expect(decodeSseBlock(": keep-alive")).toBeNull();
  });
});
