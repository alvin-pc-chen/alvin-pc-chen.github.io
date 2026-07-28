# Personal Website

Alvin's personal website — a blog-first site for research writeups, projects,
and short news updates, optimized for agent-authored infrastructure and a
draft → review → publish workflow. Migrated off a forked al-folio (Jekyll)
template into a new greenfield repo.

## Language

**News**:
A short, at-most-couple-sentences update shown at the top of the homepage /
under the bio. May or may not carry a link.
_Avoid_: announcement, status update

**Post**:
A long-form blog entry — the unified home for both traditional blog writing and
what used to be al-folio "projects" (non-interactive writeups). Supports images,
code blocks, a Vega-Lite-class chart, and a (possibly nested) TOC.
_Avoid_: article, project writeup

**Project**:
An interactive piece — a small Python (simple UI) or JS/TS app — hosted on the
site separately from Posts. Non-interactive "projects" from the old site are
reclassified as Posts; only interactive ones stay Projects.
_Avoid_: app, demo, tool

**Palette**:
The site's custom color scheme and fonts, already diverged from al-folio
defaults. Must carry over to the new repo. Independent of layout.
_Avoid_: theme (overloaded with framework theme)

**Distill layout**:
al-folio's styled post layout (d3 footnotes, sidebar TOC, inline citations).
NOT load-bearing — the user dislikes its TOC behavior. Only the underlying
capabilities (nested TOC, charts, images, code) are required, not distill itself.
_Avoid_: distill (when meaning just "the blog post style")

**Replay toy**:
The shape most Projects take: a static-replay of pre-computed traces, not live
computation. The A* visualizer is the canonical example — traces are baked into
JSON offline, the frontend just steps through them. Implies Tier 1 hosting
(client-side only, no server).
_Avoid_: simulation, interactive demo (too vague about where compute lives)

**Manifest**:
A small, in-git `data/manifest.json` per Project listing the data files
expected on R2 plus content hashes. Restores the atomicity that data-in-git
would give for free: CI verifies live R2 files match the manifest before
deploy. Also drives cleanup (sync deletes orphaned R2 objects).
_Avoid_: index (too generic), data list

## Decisions

- [accepted] **New greenfield repo**, migrate content only. Forked-template upstream sync pain is structural and unsolvable on a fork.
- [accepted] **Cloudflare Pages** hosts the static site (blog + news + Projects-as-JS). Custom domain `alvin-pc-chen.com` is one click since the domain is already on Cloudflare. Built-in unlimited preview deployments = the "push to a dev site" review mode.
- [accepted] **Three-tier Project hosting**: (1) client-side JS/TS bundled into the Pages site at `/projects/<name>/` — default, for replay toys; (2) Cloudflare Worker backend — only when a server-side secret or genuine server compute is needed (e.g. an LLM API key); (3) separate repo + `<project>.alvin-pc-chen.com` subdomain — for large-scale/auth/complex Projects. Most Projects are Tier 1.
- [accepted] **pnpm-workspace monorepo**, one package per Project + `packages/shared/` only when real sharing emerges. Per-project `README.md` for agent context. No build orchestrator (Turborepo/Nx) until build time complains.
- [accepted] **Cloudflare R2 + per-project `manifest.json`** for Project data (see ADR-0001). Data dirs gitignored; manifest in git restores atomicity and drives a CI integrity check + R2 cleanup.
- [accepted] **Astro** as the SSG, starting from the *Empty* template (not Astro-Sphere or any blog starter) with official integrations for plumbing only (`@astrojs/mdx`, `@astrojs/rss`, `@astrojs/sitemap`, `@astrojs/cloudflare`). Layout, palette/fonts, and nav authored from scratch from the existing al-folio-diverged assets. Solves both the Astro-Sphere customization friction and the forked-template upstream-sync pain. See ADR-0002.
- [accepted] **Medium syndication = semi-manual URL-import** (Medium's "Import a story"), which sets `canonical` back to the site (POSSE). The manual paste doubles as a visual-inspection gate. Interactive charts need a static-image fallback in Posts because Medium's importer doesn't run JS. Fully-automatic API syndication deprioritized by Medium; not relied on.
- [accepted] **Private GitHub repo** + Cloudflare Pages deployment. CF Pages is decoupled from GitHub repo visibility (unlike the old GitHub Pages setup), so source stays private while the deployed site is public. Preview deploy URLs (`*.pages.dev`) are still publicly accessible by link — fine for review; just don't put secrets in drafts.
- [accepted] **In-repo drafts via Astro content collections** (`draft: true` frontmatter). Prose may originate anywhere (notes, Obsidian, in-repo) but the canonical draft is the repo file; the agent enhances from finalized markdown (including placement notes for figures/code).
- [accepted] **Git workflow: branch-per-draft** (and per-Project, per-infra-change), with prefixed names (`draft/`, `projects/`, `infra/`). `main` = published only. No squash-merge — full per-commit history is preserved on `main` (version-control capability prioritized over a tidy log). Auto-delete head branches and stale-branch housekeeping are deferred, not built up front. When the stale-branch action is eventually built, it must **only delete branches that have been merged to `main` and never auto-delete unmerged branches** (safe rule; consistent with the no-squash rationale).
- [accepted] **Scheduled publish via `draft: true` + `pubDatetime` + a daily scheduled Action.** A post merged to `main` with `draft: true` and a future `pubDatetime` (UTC) is not rendered; a daily Action flips `draft: false` when `pubDatetime` passes, triggering the deploy. "Set and forget." Normal publish (merge when ready, live immediately) remains the default; scheduled is opt-in per post.
- [accepted] **CI gates.** Blocking on every PR: `astro build`, `astro check` (typecheck), `biome check` (lint/format), broken-internal-links (covered by build). Pre-commit hook (Husky + lint-staged) auto-formats staged files via Biome so agent-authored code lands formatted; CI format-check is a backstop. Affected-selected on PRs (via `pnpm --filter="...[origin/main]" -r`): per-project unit tests (vitest) and per-project R2 manifest verification. Site-wide gates run every PR because there is only one site; project tests + manifest checks run only for changed projects and their dependents. Post-merge push to `main` is deploy-only (no re-test). Turborepo deferred — pnpm filter gives affected selection now; caching is the retrofit if selection alone proves slow. Non-blocking / scheduled separately: external link-check (lychee, weekly cron), Lighthouse/axe (informational cron reports).
- [accepted] **Dependabot.** Scope: npm deps + GitHub Actions versions. Grouped minor/patch updates (one PR per ecosystem, weekly cadence); major updates as individual PRs. Auto-merge minor/patch after CI green; major updates stay manual. Security advisories bypass the weekly cadence.
- [accepted] **Cron monitoring — detection + notification only, deferred.** The approach is decided but the whole monitoring layer is NOT built in the initial migration; added later. Decided shape: weekly lychee broken-link check, weekly Lighthouse/axe digest, daily post-publish verification, weekly R2 manifest drift check — each opens a GitHub Issue / notification. Agent auto-repair is phase 2 (broken links only, the one problem with judgment-required repair + clear success criterion); agent PRs require human review (not auto-merge) because agent fixes involve judgment CI can't validate. Deploy failures, manifest drift, and missed scheduled publishes are notification-only (deterministic non-agent fixes).
- [accepted] **Publications: keep `_bibliography/papers.bib` as source of truth** + a BibTeX parser at build time (`@retorquere/bibtex-parse` or similar) + a custom Astro layout for the scholarly rendering (author lists, venue, links, selected highlighting). Rationale: new publications arrive as BibTeX from arXiv/ACL, so "paste BibTeX → rendered page" preserves the path of least resistance; the user explicitly does not want to hand-maintain entries.

## Deferred decisions (decide later, reversible)

- **Island component framework** for Project interactivity (vanilla JS default vs Svelte/React when reactive). Astro supports mixing, so this is reversible and can be decided per-Project as patterns emerge.
