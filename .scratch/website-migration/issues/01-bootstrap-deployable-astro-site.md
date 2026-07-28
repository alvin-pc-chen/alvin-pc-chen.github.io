# 01 — Bootstrap the new repo with a deployable Astro site + ported palette

**What to build:** A new private GitHub repo, initialized from `npm create astro@latest` (Empty template) with pnpm and the official integrations (`@astrojs/mdx`, `@astrojs/rss`, `@astrojs/sitemap`, `@astrojs/cloudflare`). The custom Palette (colors and fonts, ported from the al-folio-diverged assets) as a minimal stylesheet, plus a bare BaseLayout (centered single-column + top nav placeholder). Cloudflare Pages connected to the repo, the custom domain `alvin-pc-chen.com` live, and a placeholder homepage deployed. Preview deployments work per-branch. This is the thinnest end-to-end slice: repo → build → deploy → live site. Everything else builds on it.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] A new private GitHub repo exists, initialized from Astro's Empty template with pnpm
- [ ] The official integrations (MDX, RSS, sitemap, Cloudflare) are installed
- [ ] The custom Palette (colors and fonts) is ported into a stylesheet; a bare BaseLayout (centered single-column + top nav placeholder) renders the homepage using it
- [ ] Cloudflare Pages builds the site on push and deploys to the custom domain `alvin-pc-chen.com`; visiting it shows the placeholder homepage with the ported palette
- [ ] A pushed branch gets its own preview deployment URL at `*.pages.dev` that renders correctly
