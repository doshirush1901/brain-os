# Brain OS — visual identity (public)

Public-safe brand tokens for **brain-os** README and GitHub social preview.
Voice and operator behavior for your private OEM stack live in your fork's `SOUL.md`.

## Assets (this repo)

| File | Use |
|:-----|:----|
| `docs/assets/readme-hero.png` | **README hero** (~480 KB, 1840w) — neofetch layout |
| `docs/assets/social-preview.png` | **GitHub social preview** (1280×640, ~190 KB) |
| `docs/assets/readme-hero-neural.png` | Optional alt hero (~730 KB) |

Regenerate from masters: `private-brand-assets/branding/optimize_brand_assets.sh` (requires `pngquant`).
| `docs/assets/brain-os-banner.svg` | Lightweight fallback if PNG too heavy for a fork |
| `docs/assets/brain-os-logo.svg` | Compact logo / docs |
| `docs/assets/brain-os-icon.svg` | Footer icon / favicon base |
| `docs/assets/social-preview.svg` | Vector fallback for social card |

Source masters live in the private maintainer export tree under `private-brand-assets/branding/`.

## Color tokens

| token | hex | use |
|:------|:----|:----|
| ink | `#0c1014` | Dark backgrounds |
| paper | `#fafaf7` | Light surfaces |
| muted | `#5b6470` | Secondary text |
| accent | `#c8332a` | Triangulation / CTA strip |
| accent-3 | `#264653` | Graph / secondary accent |
| good | `#2a9d8f` | Success / healthy status |

Swatches on `readme-hero.png` match this table.

## Mark

Triangle + three nodes = **triangulation** (intent, relationship, identity) — product-neutral wordmark **BRAIN OS**.

## Maintainer-only media

- **private-brand-assets/** in the maintainer tree: ad concepts + `branding/` PNGs — video keyframes in `keyframes/`.
- Do not put real customer photos from a private operator repo in this public tree.

## Do not

- Use third-party wordmarks on Brain OS customer forks without trademark review.
- Upload raw Mac desktop screenshots with real paths to the public repo.
