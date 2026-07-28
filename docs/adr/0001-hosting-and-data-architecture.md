# Hosting and data architecture

The static site (blog, news, Projects-as-JS) is hosted on **Cloudflare Pages** with the custom domain `alvin-pc-chen.com` (the domain is already on Cloudflare, so DNS + SSL are one click). Cloudflare Pages' built-in unlimited preview deployments serve as the "push to a dev site" review mode — no `gh-pages` branch gymnastics. Interactive Projects follow a three-tier model: (1) client-side JS/TS bundled into the Pages site at `/projects/<name>/` — the default, for replay toys (pre-computed traces replayed in the browser, no live compute); (2) a **Cloudflare Worker** — only when a server-side secret or genuine server compute is needed (e.g. an LLM API key that cannot live client-side); (3) a separate repo + `<project>.alvin-pc-chen.com` subdomain — for large-scale/auth/complex Projects.

Project **data lives on Cloudflare R2**, not in git, because total volume is uncertain and R2's free egress beats Git LFS's 1GB/month bandwidth cap. Per-project `data/` dirs are gitignored; a per-project in-git `data/manifest.json` (files + content hashes) restores the atomicity that in-git data would give for free. A CI check verifies live R2 files match the manifest before deploy, and a publish script syncs deletes so orphaned R2 objects don't accumulate.

## Considered Options

- **GitHub Pages for prod** — familiar and already in use, but no native preview deploys (would need a second repo/branch hack) and the domain is on Cloudflare anyway. Strictly less capable than Cloudflare Pages for the stated needs.
- **Netlify / Vercel** — equally good on preview deploys, but using Cloudflare Pages keeps DNS + hosting + R2 + Workers in one account.
- **Self-hosted VPS** — only justified if interactive Python Projects must run server-side and Pyodide/PyScript is unacceptable. Deferred until a specific Project forces it.
- **Git LFS for data** — simpler "everything in git" story, but the 1GB/month free bandwidth cap is easily burned by agent sessions doing shallow clones, and the user already operates R2. Rejected for volume uncertainty + bandwidth economics.
- **Workers Python for all interactive Projects** — rejected. Most Projects are replay toys with no live compute; Workers Python (Pyodide/WASM) adds cold-start and package-compatibility limits for no benefit. Workers are reserved for the secret/compute case (Tier 2).
