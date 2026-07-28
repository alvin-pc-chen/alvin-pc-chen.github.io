# 08 — Scheduled publish (daily Action + `draft`/`pubDatetime`)

**What to build:** A daily scheduled GitHub Action that finds Posts on `main` with `draft: true` and a past `pubDatetime` (UTC), flips `draft: false`, and commits — triggering the production deploy. Normal publish (merge ready, live immediately) is the default; scheduled is opt-in per Post.

**Blocked by:** 03 — First Post end-to-end: content collection + layout + build-output test.

**Status:** ready-for-agent

- [ ] A daily scheduled GitHub Action runs and finds Posts on `main` with `draft: true` and a past `pubDatetime`
- [ ] The Action flips `draft: false` for due Posts and commits, triggering the production deploy
- [ ] A future-dated draft (future `pubDatetime`) stays hidden on the live site until its `pubDatetime` passes, then goes live
- [ ] Normal publish (merge with `draft: false`) goes live immediately, unchanged
