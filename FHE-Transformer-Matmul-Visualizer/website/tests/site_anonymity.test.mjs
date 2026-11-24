import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const pages = ["index.html", "transformer_visualizer.html", "visualizer.html"];

test("public pages use the survey identity and expose no upstream publication or repository links", () => {
  for (const page of pages) {
    const html = readFileSync(`${root}/${page}`, "utf8");
    assert.match(html, /Secure Transformer Survey/, page);
    assert.doesNotMatch(html, /eprint\.iacr\.org\/2025\/1319/i, page);
    assert.doesNotMatch(html, /github\.com\/cmu-cryptosystems/i, page);
    assert.doesNotMatch(html, />\s*Rotom\s*</i, page);
    assert.doesNotMatch(html, />\s*(Product|Contribute|Research|Repository|Organization|Paper)\s*</i, page);
  }
});

test("home page contains no citation or bibliography section", () => {
  const html = readFileSync(`${root}/index.html`, "utf8");
  assert.doesNotMatch(html, /id=["']paper["']/i);
  assert.doesNotMatch(html, /citation|bibtex/i);
  assert.match(html, /href=["']\.\/transformer_visualizer\.html["']/);
});
