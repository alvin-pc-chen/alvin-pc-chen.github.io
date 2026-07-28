# Astro as the static site generator

Use **Astro** as the SSG, starting from the *Empty* template (`npm create astro@latest`, choose Empty) — not Astro-Sphere or any blog starter — with official integrations for plumbing only (`@astrojs/mdx`, `@astrojs/rss`, `@astrojs/sitemap`, `@astrojs/cloudflare`). Layout, palette/fonts, and nav are authored from scratch, porting the existing al-folio-diverged visual identity. Rationale: Astro's islands architecture matches the site's shape (content-first blog + news, with interactive Project islands that ship JS only on their route); MDX gives native embedding of Vega-Lite charts, code blocks, and images; file-based routing and TSX/MDX are agent-friendly compared to Jekyll + Liquid; and base Astro ships no visual or structural opinions, which both solves the customization friction hit with Astro-Sphere and avoids reintroducing the forked-template upstream-sync pain (pain #2). Base Astro is a normal versioned npm dependency with no upstream template to rebase against — the root cause of pain #2 is gone by construction.

Medium syndication is orthogonal to the SSG: Medium's "Import a story" consumes a public URL with clean HTML, which Astro produces. The import sets `canonical` back to the site (POSSE). Interactive charts need a static-image fallback in Posts because Medium's importer does not run JS; this is a per-post content convention, not a framework concern.

## Considered Options

- **Astro-Sphere / Astro-Paper (blog starter templates)** — rejected. They impose visual + structural + plumbing opinions that must be undone, reproducing the forked-template pain the migration is meant to escape. The customization friction the user hit with Astro-Sphere is the same shape as al-folio, just smaller.
- **Next.js** — overkill. An app framework that happens to do SSG; carries React App Router complexity and heavier builds for a site that's mostly static content. No upside for the content-to-interactivity ratio here.
- **Eleventy (11ty)** — Markdown-first and simple, but no component/islands model. Interactive Projects would be hand-rolled HTML+JS glued in, which is the indirection being escaped.
- **Hugo** — fast, but its Go-template DSL is agent-hostile and Projects would be hand-rolled. No.
- **SvelteKit** — excellent for interactivity, but an app framework first and a blog second; inverts the actual content-to-interactivity ratio.
