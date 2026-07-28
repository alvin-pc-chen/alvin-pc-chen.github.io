Status: ready-for-agent

# Spec: Website migration to a new greenfield Astro repo

Synthesized from the `/grill-with-docs` session that produced `CONTEXT.md` and ADRs `0001` (hosting + data architecture) and `0002` (Astro as the SSG). Those documents are the source of truth for the decisions; this spec turns them into buildable work. The domain glossary vocabulary (News, Post, Project, Replay toy, Manifest, Palette, Distill layout) is used throughout.

## Problem Statement

My personal website is a fork of the al-folio Jekyll template. Three problems have made it unsustainable:

1. I want to switch to a custom domain (`alvin-pc-chen.com`, registered on Cloudflare), but the current GitHub Pages deploy couples "public source repo" to "hosted site," and I'd prefer the source stay private.
2. Keeping up with al-folio upstream is structurally painful — it's a forked template, not a library, so every customization diverges and rebasing `_includes`/`_layouts`/`_sass` overrides against upstream is miserable. No automation fixes this; it's the architecture.
3. I've left academia but still produce research and personal projects I want to share and blog about. I want to combine my (non-interactive) projects and blog into a single blog, cross-post to Medium via POSSE for visibility, and also host new interactive projects. The current al-folio setup isn't shaped for any of this, and isn't optimized for agent-authored infrastructure.

I want a new repo that I don't have to hand-fix, with a streamlined draft → review → publish pipeline and CI that catches errors.

## Solution

Build a new, private greenfield repo from base Astro (the Empty template, no blog starter), migrating content only. The deployed site is a static site on Cloudflare Pages (custom domain via the existing Cloudflare account), with built-in preview deployments as the "push to a dev site" review mode. Interactive Projects are client-side JS/TS apps bundled into the site (Replay toys), with Cloudflare Workers reserved for the rare server-side-secret case and separate subdomain repos for large-scale projects. Project data lives on Cloudflare R2 with per-project in-git manifests restoring atomicity. A pnpm-workspace monorepo keeps each Project isolated for agent work. The draft → review → publish pipeline is branch-per-draft, with CI gates (build, typecheck, lint, affected tests, manifest verification) blocking every PR, Dependabot auto-merging routine updates on green, and scheduled-publish support for future-dated Posts. Medium syndication is semi-manual URL-import (POSSE, canonical back to the site).

## User Stories

### Authoring & content

1. As the site owner, I want to draft a Post in Markdown (originating in a notes app, Obsidian, or in-repo) and finalize it as a Markdown file in the repo, so that the canonical draft lives where the agent can assist.
2. As the site owner, I want to hand the agent a finalized Markdown draft with placement notes for figures/code/charts, so that the agent can add the missing images, Vega-Lite charts, and code blocks and tidy formatting.
3. As the site owner, I want a Post to support a nested TOC, so that long research writeups are navigable without the distill layout's disliked TOC behavior.
4. As the site owner, I want a Post to render images, syntax-highlighted code blocks, and Vega-Lite charts, so that research writeups look right.
5. As the site owner, I want News items (at most a couple sentences, optionally linked) shown at the top of the homepage / under the bio, so that visitors see recent updates first.
6. As the site owner, I want to add a News item by creating a short Markdown file, so that updating the feed is trivial.
7. As the site owner, I want my existing non-interactive projects (cnn, irony) reclassified as Posts, so that the blog is the single home for long-form writeups.
8. As the site owner, I want my Publications rendered from `_bibliography/papers.bib` in a scholarly format (author lists, venue, links, `selected` highlighting), so that I can paste new BibTeX from arXiv/ACL and have it render without hand-maintaining entries.
9. As the site owner, I want my custom Palette (colors and fonts, already diverged from al-folio defaults) carried over, so that the new site looks like my site.
10. As the site owner, I want a centered single-column layout with a top nav bar and minimal removable UI flourish, so that the design is clean and not imposed by a template.
11. As the site owner, I want social links (email, GitHub, LinkedIn, Medium, and Bluesky) shown in the nav/footer, so that visitors can find me elsewhere.

### Review & publish pipeline

12. As the site owner, I want to run a local dev server that renders a draft exactly as it will appear in production, so that I can review before pushing.
13. As the site owner, I want each branch to deploy to its own preview URL on Cloudflare Pages, so that I can review the exact production layout on any device and share a link.
14. As the site owner, I want to open a PR manually when a draft is ready (not auto-PR on push), so that iterating locally doesn't create PR noise.
15. As the site owner, I want the agent to enhance a draft on its branch *before* I push for preview, so that the preview shows the enhanced version, not my pre-enhancement draft.
16. As the site owner, I want to work on multiple drafts and Projects concurrently on separate prefixed branches (`draft/`, `projects/`, `infra/`), so that concurrent work doesn't block independent publishing.
17. As the site owner, I want to publish a Post by merging its branch to `main`, so that `main` = published only and merges are independent.
18. As the site owner, I want full per-commit history preserved on `main` (no squash-merge), so that version-control capability isn't reduced.
19. As the site owner, I want to schedule a Post for future publication by merging it to `main` with `draft: true` and a future `pubDatetime`, so that a daily Action flips it live at the right time without me babysitting it.
20. As the site owner, I want to syndicate a Post to Medium by pasting its URL into Medium's "Import a story," so that the cross-post carries `canonical` back to my site (POSSE) and I can visually inspect the Medium render before it goes public.

### Projects (interactive)

21. As the site owner, I want to host an interactive Project as a client-side JS/TS app at `/projects/<name>/`, so that visitors can use a toy (e.g., an A\* search visualizer) with no server.
22. As the site owner, I want each Project to be a self-contained pnpm-workspace package with its own `README.md` and (when reactive) its own component code, so that an agent can work on one Project without reading the others.
23. As the site owner, I want Project data files to live on Cloudflare R2 (gitignored locally), so that large datasets don't bloat the repo and volume growth is unconstrained by Git LFS bandwidth.
24. As the site owner, I want each Project to carry an in-git `data/manifest.json` listing its expected R2 files plus content hashes, so that CI can verify data exists and matches before deploy, and a publish script can sync-delete orphaned R2 objects.
25. As the site owner, I want to publish a Project's data to R2 with one command, so that "publish data before merging data-dependent code" is a trivial action.
26. As the site owner, I want the agent to be able to pull a Project's data on demand from R2 into the gitignored local `data/` dir, so that it can test data-dependent behavior without the data being in git.
27. As the site owner, I want Projects I build separately (e.g., the A\* visualizer) to be migrated into the site when ready, at which point they gain their own per-Project test seam.
28. As a visitor, I want a Project page to load only that Project's JS (not every Project's), so that opening one toy is fast regardless of how many Projects exist.

### Visitors

29. As a visitor, I want to read the blog, see the news feed, browse Publications, and use interactive Projects, all at `alvin-pc-chen.com` over HTTPS, so that the site is fast and reachable.
30. As a visitor, I want only published Posts to appear on the live site (drafts excluded), so that I never see half-finished work.
31. As a visitor, I want an RSS feed and a sitemap, so that I can subscribe and the site is discoverable.
32. As a visitor, I want a 404 page, so that broken URLs are handled gracefully.

### Maintenance & CI

33. As the site owner, I want every PR to be blocked unless `astro build`, `astro check` (typecheck), and `biome check` (lint/format) pass, so that broken code never reaches `main`.
34. As the site owner, I want a pre-commit hook (Husky + lint-staged) that auto-formats staged files with Biome, so that agent-authored code lands formatted and I never review formatting.
35. As the site owner, I want per-Project unit tests (vitest) and per-Project R2 manifest verification to run only for changed projects and their dependents (`pnpm --filter="...[origin/main]"`), so that CI doesn't re-run all 50 Projects' tests on every commit.
36. As the site owner, I want external link-checking (lychee) to run on a weekly schedule (not blocking PRs), so that link rot is caught as decay rather than mistaken for a PR regression.
37. As the site owner, I want Dependabot to open grouped minor/patch PRs (npm + GitHub Actions) weekly and auto-merge them when CI is green, so that routine updates flow without my involvement.
38. As the site owner, I want Dependabot to open major updates as individual PRs that require my review, so that potentially-breaking changes get my eyes.
39. As the site owner, I want a push to `main` to deploy to production (no re-test), so that merging a green PR is the publish action.
40. As an AFK agent working on a ticket, I want to read `CONTEXT.md`, `docs/adr/`, and `docs/agents/` to understand the domain and conventions before touching code, so that my work matches the project's vocabulary and decisions.
41. As an AFK agent, I want affected-project test selection to mean I only run tests for the packages my change touches (plus dependents), so that my feedback loop is fast.

## Implementation Decisions

### Repo & monorepo shape

- New, **private** GitHub repo. Deployed via Cloudflare Pages, which is decoupled from GitHub repo visibility (unlike the old GitHub Pages setup).
- **pnpm-workspace monorepo**: one package per Project; a `packages/shared/` only when real sharing emerges. No build orchestrator (Turborepo/Nx) initially.
- Per-Project `README.md` for agent context. Project data dirs are gitignored; a per-project in-git `data/manifest.json` (files + content hashes) restores atomicity.
- Git workflow: branch-per-draft and per-Project/per-infra-change, with prefixed names (`draft/`, `projects/`, `infra/`). `main` = published only. No squash-merge (full history preserved). Auto-delete head branches and stale-branch housekeeping are deferred; the eventual stale-branch action deletes merged-only, never unmerged.
- See ADR-0001 (hosting + data architecture) and ADR-0002 (Astro as the SSG).

### SSG & content

- **Astro** from the Empty template, with official integrations for plumbing only (`@astrojs/mdx`, `@astrojs/rss`, `@astrojs/sitemap`, `@astrojs/cloudflare`). Layout, Palette, and nav authored from scratch from the ported al-folio-diverged assets. No blog starter template (Astro-Sphere/Paper rejected — they impose visual/structural opinions to undo).
- **Posts** are an Astro content collection of Markdown/MDX entries with frontmatter (title, date, `draft`, `pubDatetime`, tags). MDX permits embedding Vega-Lite chart components and interactive figures. Nested TOC generated from headings (not the distill layout). The `distill layout` is NOT carried over — only its capabilities (TOC, charts, images, code) are required.
- **News** is a content collection of short inline items rendered on the homepage / under the bio.
- **Projects** are interactive client-side apps, one per workspace package, routed at `/projects/<name>/`. Island component framework choice (vanilla JS default vs Svelte/React when reactive) is deferred and decided per-Project (Astro supports mixing).
- The two existing non-interactive projects (cnn, irony) are migrated as **Posts**, not Projects.
- Route-level code splitting is automatic (Astro default): visiting one Project loads only that Project's JS.

### Publications

- Keep `_bibliography/papers.bib` as the source of truth. A BibTeX parser runs at build time; a custom Astro layout renders the scholarly format (author lists, venue, links, `selected` highlighting). New publications arrive as BibTeX from arXiv/ACL — paste and render.

### Data layer (R2)

- Project data lives on **Cloudflare R2**, served via a custom subdomain (e.g., `data.alvin-pc-chen.com`) on Cloudflare's CDN (free egress).
- The frontend fetches data from the R2 URL (not a local path). For replay toys, the data-shaping convention is "small index first, lazy chunks" so first paint doesn't wait on a multi-MB fetch.
- A `scripts/publish-data` CLI syncs a Project's `data/` dir to `r2://projects-data/<project>/`, with sync-delete for cleanup.
- A `scripts/fetch-data` CLI lets an agent pull a Project's data on demand into the gitignored local `data/` for testing.
- The in-git `data/manifest.json` per Project lists expected files + content hashes.

### Deploy

- **Cloudflare Pages** builds on push to `main` → production at `alvin-pc-chen.com` (custom domain via the existing Cloudflare account; DNS + SSL automatic). Preview deployments per branch/PR at `*.pages.dev` (publicly accessible by link — fine for review; no secrets in drafts).
- Post-merge push to `main` is deploy-only (no re-test).
- Three-tier Project hosting: (1) client-side JS/TS (default, Replay toys); (2) Cloudflare Worker (only for server-side secrets/compute, e.g., an LLM API key); (3) separate repo + `<project>.alvin-pc-chen.com` subdomain (large-scale/auth/complex). Most Projects are Tier 1.

### Scheduled publish

- A Post merged to `main` with `draft: true` and a future `pubDatetime` (UTC) is not rendered. A daily scheduled GitHub Action finds such Posts, flips `draft: false` when `pubDatetime` passes, and commits — triggering the deploy. Normal publish (merge ready, live immediately) is the default; scheduled is opt-in per Post. A follow-up check verifies the publish ran; if not, it opens a notification (deferred — cron layer not built initially).

### CI gates (blocking on every PR)

- `astro build` (catches broken imports, bad frontmatter, content-collection schema violations, broken internal links).
- `astro check` (typecheck).
- `biome check` (lint/format backstop; pre-commit hook is the primary formatter).
- Affected-selected via `pnpm --filter="...[origin/main]" -r`: per-Project vitest tests + per-Project R2 manifest verification. Site-wide gates run every PR; project tests + manifest checks run only for changed projects and dependents.

### Medium syndication

- Semi-manual URL-import: publish on the site → paste the Post URL into Medium's "Import a story" → Medium sets `canonical` back to the site (POSSE). Interactive charts need a static-image fallback in Posts because Medium's importer doesn't run JS. Fully-automatic API syndication is not relied on (Medium has deprioritized it).

### Dependabot

- Scope: npm deps + GitHub Actions versions. Grouped minor/patch (one PR per ecosystem, weekly); major as individual PRs. Auto-merge minor/patch on CI green; major stays manual. Security advisories bypass the weekly cadence.

## Testing Decisions

### What makes a good test

Test external behavior, not implementation details. A good test asserts what a user/visitor experiences and survives internal refactors. Go to the highest seam possible; the fewer seams, the better.

### Seams (confirmed with the user)

1. **Build output — one site-wide seam (highest).** After `astro build`, assert properties of the rendered `dist/` output. Covers: Posts render with correct frontmatter and contain images/code-blocks/Vega-Lite chart markup; News feed renders the migrated items on the homepage; Publications page renders all `.bib` entries with scholarly format and `selected` highlighting; socials render (incl. Bluesky); custom Palette/fonts applied; centered single-column + top nav layout; drafts (`draft: true`) excluded from the production build; internal links valid; RSS + sitemap generated; 404 renders. This seam rides on top of the `astro build` + `astro check` CI gates, adding behavioral assertions (e.g., "the publications page contains N entries," "drafts are absent from the prod build").

2. **Project logic — one seam per Project (function-level, added when the Project is migrated in).** Each interactive Project owns its own vitest suite asserting algorithm correctness (e.g., A\* replay produces the correct open/closed sets for a given trace; graph search returns the correct path). Build output cannot validate algorithm correctness, so this lower seam is necessary. It is per-project (one seam per package), not a second site-wide seam. Projects built separately (e.g., the A\* visualizer) gain this seam when migrated into the site.

### Modules tested

- The site (layout, content collections, Publications, News, socials, deploy output) — at the build-output seam.
- Each Project package — at its own function-level seam, when migrated.

### Prior art

- No existing test suite in the current al-folio repo — all seams are new.
- The separately-built A\* project repo (`interp_gridworld`, with a `tests/` dir) is the pattern reference for Project-level testing when it is migrated in.

## Out of Scope

- **Cron monitoring + agent auto-repair** — deferred entirely (not built in the initial migration). The decided shape: weekly lychee broken-link check, weekly Lighthouse/axe digest, daily post-publish verification, weekly R2 manifest drift check — each opening a notification. Agent auto-repair for broken links is phase 2; agent PRs require human review. Added later.
- **Turborepo** — retrofit only if affected-selection proves slow.
- **Stale-branch action + auto-delete head branches** — retrofit only if branch clutter hurts. The eventual stale action deletes merged-only, never unmerged.
- **Island component framework choice** (vanilla JS vs Svelte/React) — decided per-Project when migrated; Astro supports mixing.
- **The A\* visualizer and other separately-built Projects** — built in their own repos, migrated into the site when ready (at which point they gain the per-Project test seam).
- **Tier 2 (Cloudflare Worker for LLM features) and Tier 3 (large-scale/auth/complex projects on subdomains)** — built only when a specific Project forces them.
- **The prose content of blog Posts** — written by hand by the owner; the agent assists with formatting/images/charts/code, not authorship.
- **Lighthouse/axe as blocking gates** — informational cron only, not merge gates.

## Further Notes

- The authoritative decisions live in `CONTEXT.md` (glossary + decision log) and `docs/adr/0001-hosting-and-data-architecture.md` and `docs/adr/0002-astro-as-the-ssg.md`. If implementation contradicts an ADR, surface it explicitly rather than silently overriding.
- Content to migrate from the current repo: 3 live Posts (`umr_annotation`, `probing`, `probing_pt2`) + 3 drafts (`ai_coding`, `graphspect`, `probing-old`/`probing-page`); 2 non-interactive projects reclassified as Posts (`cnn`, `irony`); 2 News items; `_bibliography/papers.bib` (4 entries); socials config (add Bluesky); the custom Palette/fonts from `_sass/` and assets.
- This spec describes the **target** (the new repo). It is being written into the **old** repo's `.scratch/` per the sequencing: the grilling, spec, and tickets stay in one context window through `/to-tickets`, then the artifacts (`CONTEXT.md`, `AGENTS.md`, `docs/`, `.scratch/`) move to the new private repo when it is created.
- The new repo is the implementation surface; `/to-tickets` next splits this spec into tracer-bullet tickets with blocking edges, each `Status: ready-for-agent`, worked blockers-first.
