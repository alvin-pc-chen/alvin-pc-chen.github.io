# 02 — CI gates + pre-commit hook

**What to build:** The safety net. A CI workflow with blocking gates on every PR: `astro build`, `astro check` (typecheck), and `biome check` (lint/format). A Husky + lint-staged pre-commit hook that auto-formats staged files with Biome so agent-authored code lands formatted. pnpm-filter scaffolding for affected project-test selection (a no-op until a Project lands; manifest verification added when the data layer lands). A push to `main` is deploy-only (no re-test).

**Blocked by:** 01 — Bootstrap the new repo with a deployable Astro site + ported palette.

**Status:** ready-for-agent

- [ ] A PR that breaks `astro build`, `astro check`, or `biome check` is blocked from merging
- [ ] A commit made with staged files lands formatted (the pre-commit hook auto-formats via Biome)
- [ ] Affected-project test selection via `pnpm --filter="...[origin/main]"` is wired (no-op until a Project package exists)
- [ ] A merged PR triggers a production deploy with no re-test on `main`
