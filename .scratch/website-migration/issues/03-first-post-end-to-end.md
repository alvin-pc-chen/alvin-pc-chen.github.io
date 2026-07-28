# 03 — First Post end-to-end: content collection + layout + build-output test

**What to build:** The Post content collection schema (frontmatter: title, date, `draft`, `pubDatetime`, tags). A post layout with nested TOC generation (from headings — not the distill layout's disliked behavior), image rendering, syntax-highlighted code blocks, and a Vega-Lite chart component embeddable via MDX. Migrate the `probing` post as the test case. The build-output seam (seam 1): a test asserting the post renders with TOC, image, code, and chart markup, and that a `draft: true` entry is excluded from the production build.

**Blocked by:** 02 — CI gates + pre-commit hook.

**Status:** ready-for-agent

- [ ] The Post content collection schema (title, date, draft, pubDatetime, tags) is defined and `astro check` validates frontmatter against it
- [ ] A post layout renders a nested TOC from headings, images, syntax-highlighted code blocks, and a Vega-Lite chart component (via MDX)
- [ ] The `probing` post is migrated and live at `/blog/probing` with TOC, image, code, and chart markup
- [ ] A build-output test asserts the migrated post renders with all expected elements
- [ ] A `draft: true` post is excluded from the production build (asserted by the build-output test)
