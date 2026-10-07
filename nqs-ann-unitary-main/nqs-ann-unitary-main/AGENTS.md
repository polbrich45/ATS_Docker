# Repository guidance

Global defaults apply. This file adds only project-specific conventions.

## Scientific Python

- Prefer the smallest correct change. Reuse existing code and avoid speculative abstractions, compatibility aliases, and unrelated refactors.
- Match the repository's space-based (four-space) indentation and align related fields when it improves scanning.
- Keep the adaptive workflow visible and ordered: train, evaluate, rotate, final Ritz, teacher, accept, summarize.
- Use short separator banners and direct physics-oriented comments or docstrings. State non-obvious basis, sign, normalization, unit, or ordering conventions.
- Preserve exact results, approximations, heuristics, and assumptions as distinct cases.

## Theory notes and readmes

- Use compact Physical Review-style prose. Define every symbol at first use, introduce the setting before the result, and stop when the requested point is resolved.
- Use `$...$` for inline math and `$$...$$` for display math on separate lines. Display equations end with `\;.` or `\;,` as appropriate; do not introduce them with a colon.
- Check signs, normalization, dimensions, operator ordering, Hermiticity, basis conventions, index ranges, and limiting cases.
- Make claims conservatively. Never invent sources, identifiers, quotations, or results; flag anything that cannot be verified.
- Use ASCII hyphens, `- ` bullets, and natural prose flow. Do not add artificial line wrapping.
