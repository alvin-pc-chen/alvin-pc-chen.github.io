# 09 — Dependabot config

**What to build:** `.github/dependabot.yml` for npm deps + GitHub Actions versions. Grouped minor/patch (one PR per ecosystem, weekly cadence); major updates as individual PRs. Auto-merge minor/patch on CI green; major stays manual. Security advisories bypass the weekly cadence.

**Blocked by:** 02 — CI gates + pre-commit hook.

**Status:** ready-for-agent

- [ ] Dependabot is configured for npm deps and GitHub Actions versions
- [ ] Minor/patch updates are grouped into one PR per ecosystem on a weekly cadence; major updates are individual PRs
- [ ] A routine minor/patch PR auto-merges when CI is green
- [ ] A major-update PR requires manual review (does not auto-merge)
- [ ] Security advisories open PRs immediately, bypassing the weekly cadence
