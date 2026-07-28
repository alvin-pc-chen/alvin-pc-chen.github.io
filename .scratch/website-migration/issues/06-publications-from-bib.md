# 06 — Publications from `.bib` + scholarly layout

**What to build:** Port `_bibliography/papers.bib` (4 entries) as the source of truth. A BibTeX parser runs at build time; a custom Astro layout renders the scholarly format (author lists, venue, links, `selected` highlighting). A `/publications` page. A build-output assertion that the page contains all 4 entries with the scholarly format and selected highlighting.

**Blocked by:** 02 — CI gates + pre-commit hook.

**Status:** ready-for-agent

- [ ] `_bibliography/papers.bib` (4 entries) is ported and parsed at build time
- [ ] A custom layout renders the scholarly format: author lists, venue, links, and `selected` highlighting
- [ ] The `/publications` page renders all 4 entries
- [ ] A build-output test asserts the page contains all 4 entries with the scholarly format and selected highlighting
