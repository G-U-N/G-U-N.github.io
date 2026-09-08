# Shared blog components

Every article uses the same components. Readers can switch **Classic / Paper**;
fonts, Colab entries, references, and navigation do not have separate settings.

- **`blog-common.css`**: font loading and font families, type sizes, page width,
  headings, contents, Colab, equations, figures, code, references, and footer navigation.
  Edit the variables at the top for routine adjustments.
- **`blog-themes.css`**: background, article card, and accent-color presets.
- **`blog-theme.js`**: the shared theme button and saved preference.
- `distributional-distillation.css`: existing base rules and shared derivation /
  figure helpers; still required by the distillation, attention, RL, and MoE posts.
- `scaling-long-context-attention.css`: code and attention-layout helpers, also
  used by MoE. `rl-opd-vsd.css` and `mixture-of-experts-figures.css` hold their
  articles' diagrams and interactions.

Add new shared component styles to `blog-common.css`, not to article stylesheets.
`blog-refinements.css` was superseded by the common styles and has been removed.

## New article

Use `<html lang="en" class="blog-page">`. After article-specific styles, include:

```html
<link rel="stylesheet" href="blog-common.css">
<link rel="stylesheet" href="blog-themes.css">
<script src="blog-theme.js"></script>
```

The script adds the theme button beside `.back-link` in `.blog-header`.
Do not add per-page font imports or override common components in article CSS.

## Colab

```html
<a class="colab-callout" href="NOTEBOOK_URL" target="_blank" rel="noopener">
  <span class="colab-label">Open in Colab</span>
  <span><strong>Notebook title</strong><small>Brief description.</small></span>
  <i class="fas fa-arrow-up-right-from-square" aria-hidden="true"></i>
</a>
```

## References and navigation

Use one flat, numbered list in every article, including Markdown-rendered articles.
Keep existing citation numbers, order, IDs, links, and explanatory notes. Put the
full paper title first and author/source/year below it. Every entry needs a primary
source link and complete bibliographic metadata, not just a method name or surname.
Do not add topic groups or separate cards. Number new entries consecutively.

- Papers: use authors' full names in the published order. List all authors up to
  ten; for longer lists, use the first three names followed by “et al.” Use the
  credited team name for reports authored by a collective such as DeepSeek-AI.
- Cite the confirmed conference/journal and its publication year. Otherwise use
  the arXiv identifier and preprint year; do not guess a conference from a review
  submission. Keep explicitly cited versions when section pointers depend on them.
- Books: include the edition, publisher, and year, with a publisher or DOI link.
- Code and documentation: credit the maintaining project, name the resource type,
  and include the cited commit/version when available. Do not add access dates.
- Repeated references across articles must have the same title, authors, and
  publication metadata. Article-specific reading notes can differ.

```html
<section class="references" aria-labelledby="references">
  <h2 id="references">References</h2>
  <ol class="reference-list" role="list">
    <li id="ref-example">
      <span class="reference-number">[1]</span>
      <div class="reference-body">
        <span class="reference-title"><a href="PAPER_URL">Paper title</a></span>
        <span class="reference-meta">Author. Venue, year. Optional note.</span>
      </div>
    </li>
  </ol>
</section>
<nav class="article-nav" aria-label="Blog navigation">
  <a href="../index.html#blogs">All posts</a>
</nav>
```

Number alignment, title weight, metadata color, and separators are defined once in
`blog-common.css`. Keep these shared styles out of article-specific CSS.
