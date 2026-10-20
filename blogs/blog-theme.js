/* Minimalist Reader Enhancements (Sander.ai style)
 * Zero toolbar buttons or visual clutter.
 * Provides:
 * 1. Reading time appended quietly to .blog-meta
 * 2. Top-left unobtrusive corner menu (Home, All posts, and section jump links)
 * 3. Quiet hover popovers for citations ([1]) and equation references ((1))
 * 4. Click-to-zoom figure lightbox
 */
(() => {
    function extractCleanText(el) {
        if (!el) return '';
        const clone = el.cloneNode(true);
        clone.querySelectorAll('.katex-mathml, script, style').forEach(n => n.remove());
        return (clone.textContent || '').replace(/\s+/g, ' ').trim();
    }

    function slugify(text, fallbackIndex) {
        const base = (text || '')
            .toLowerCase()
            .replace(/[^\w\u4e00-\u9fa5\s-]/g, '')
            .trim()
            .replace(/\s+/g, '-')
            .slice(0, 48);
        return base || `section-${fallbackIndex}`;
    }

    function init() {
        const content = document.querySelector('.blog-content');
        const header = document.querySelector('.blog-header');
        if (!content) return;

        const isIndexPage = content.classList.contains('post-index');
        if (isIndexPage) {
            setupCornerMenu([]);
            return;
        }

        // Defer a tick so dynamic articles (consolver-theory, rectified-diffusion, cm-meanflow) finish rendering
        setTimeout(() => enhanceArticle(content, header), 20);
    }

    function enhanceArticle(content, header) {
        if (content.dataset.sanderEnhanced === 'true') return;
        content.dataset.sanderEnhanced = 'true';

        // Move .dek, .colab-callout, .header-insight, and .series-nav into the top of the white .blog-content sheet
        // so the header above the sheet only has the clean centered Title + Meta (just like sander.ai)
        const container = content.parentElement;
        const seriesNav = container?.querySelector(':scope > .series-nav');
        const colab = header?.querySelector('.colab-callout');
        const insight = header?.querySelector('.header-insight');
        const dek = header?.querySelector('.dek');

        if (seriesNav) content.prepend(seriesNav);
        if (colab) content.prepend(colab);
        if (insight) content.prepend(insight);
        if (dek) content.prepend(dek);

        // 1. Append quiet reading time to .blog-meta (like Sander's "Reading time ~X minutes")
        const meta = header?.querySelector('.blog-meta');
        if (meta && !meta.textContent.includes('min read') && !meta.textContent.includes('Reading time')) {
            const words = (content.innerText || '').trim().split(/\s+/).filter(Boolean).length;
            const mins = Math.max(3, Math.round(words / 210));
            const span = document.createElement('span');
            span.innerHTML = ` &middot; Reading time ~${mins} minutes`;
            meta.appendChild(span);
        }

        // 2. Ensure h2 headings have IDs and build unobtrusive top-left corner menu (like Sander's #dl-menu)
        const h2s = [...content.querySelectorAll(':scope > h2, section.references > h2')];
        h2s.forEach((h2, idx) => {
            if (!h2.id) h2.id = slugify(extractCleanText(h2), idx + 1);
        });
        setupCornerMenu(h2s);

        // 3. Quiet hover popovers for citations and equation references
        const eqMap = indexEquations(content);
        setupHoverPopovers(content, eqMap);

        // 4. Click-to-zoom figure lightbox
        setupFigureLightbox(content);
    }

    function setupCornerMenu(h2s) {
        if (document.querySelector('.sander-corner-nav')) return;
        const nav = document.createElement('nav');
        nav.className = 'sander-corner-nav';
        nav.setAttribute('aria-label', 'Quick navigation');

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'sander-menu-btn';
        btn.setAttribute('aria-label', 'Open Menu');
        btn.title = 'Menu & Contents';
        btn.innerHTML = '<span class="sander-menu-bars"></span>';

        const panel = document.createElement('div');
        panel.className = 'sander-menu-panel';

        const homeLink = document.createElement('a');
        homeLink.href = '../index.html';
        homeLink.textContent = 'Home';
        panel.appendChild(homeLink);

        const postsLink = document.createElement('a');
        postsLink.href = '../index.html#blogs';
        postsLink.textContent = 'All Posts';
        panel.appendChild(postsLink);

        if (h2s.length > 0) {
            const div = document.createElement('div');
            div.className = 'sander-menu-divider';
            panel.appendChild(div);

            const heading = document.createElement('div');
            heading.className = 'sander-menu-heading';
            heading.textContent = 'Contents';
            panel.appendChild(heading);

            h2s.forEach(h2 => {
                const a = document.createElement('a');
                a.href = `#${h2.id}`;
                a.textContent = extractCleanText(h2);
                a.addEventListener('click', () => nav.classList.remove('is-open'));
                panel.appendChild(a);
            });
        }

        btn.addEventListener('click', e => {
            e.stopPropagation();
            nav.classList.toggle('is-open');
        });
        document.addEventListener('click', e => {
            if (!nav.contains(e.target)) nav.classList.remove('is-open');
        });

        nav.append(btn, panel);
        document.body.appendChild(nav);
    }

    function indexEquations(content) {
        const eqMap = new Map();
        const cards = [...content.querySelectorAll('.formula-card')];
        cards.forEach((card, idx) => {
            const labelEl = card.querySelector('.equation-label');
            if (!labelEl) return;
            const match = labelEl.textContent.trim().match(/^\(([^)]+)\)$/);
            if (!match) return;
            const key = match[1].trim();
            if (!card.id) card.id = `eq-auto-${key.replace(/\W+/g, '-')}-${idx}`;
            eqMap.set(key, { id: card.id, html: card.innerHTML });
        });
        return eqMap;
    }

    function setupHoverPopovers(content, eqMap) {
        const popover = document.createElement('div');
        popover.className = 'blog-hover-popover';
        document.body.appendChild(popover);

        const refItems = [...content.querySelectorAll('.references .reference-list > li, .footnote ol > li')];
        const refByNumber = new Map();
        const refById = new Map();

        refItems.forEach((li, idx) => {
            const numText = (li.querySelector('.reference-number')?.textContent || `[${idx + 1}]`).trim();
            const numMatch = numText.match(/\d+/);
            const clone = li.cloneNode(true);
            clone.querySelectorAll('.footnote-backref').forEach(b => b.remove());
            const html = clone.innerHTML;
            if (numMatch) refByNumber.set(numMatch[0], html);
            if (li.id) refById.set(li.id, html);
        });

        function showPopover(anchorEl, htmlContent) {
            if (!htmlContent) return;
            popover.innerHTML = htmlContent;
            popover.classList.add('is-visible');

            const rect = anchorEl.getBoundingClientRect();
            const popRect = popover.getBoundingClientRect();
            let top = rect.top - popRect.height - 8;
            if (top < 12) top = rect.bottom + 8;
            let left = rect.left + rect.width / 2 - popRect.width / 2;
            left = Math.max(12, Math.min(window.innerWidth - popRect.width - 12, left));

            popover.style.top = `${top}px`;
            popover.style.left = `${left}px`;
        }

        function hidePopover() {
            popover.classList.remove('is-visible');
        }

        const interactiveLinks = [...content.querySelectorAll('a.source-cite, a.footnote-ref, a[data-reference], a[href^="#ref-"], a[href^="#fn:"], a[href^="#eq-"]')];
        interactiveLinks.forEach(a => {
            a.addEventListener('mouseenter', () => {
                const refId = a.dataset.reference || (a.getAttribute('href')?.startsWith('#') ? a.getAttribute('href').slice(1) : '');
                if (refId && refById.has(refId)) {
                    showPopover(a, refById.get(refId));
                    return;
                }
                if (refId) {
                    const targetEl = document.getElementById(refId);
                    const card = targetEl?.closest('.formula-card');
                    if (card) {
                        showPopover(a, card.innerHTML);
                        return;
                    }
                }
                const numText = a.querySelector('.source-cite-number')?.textContent || a.textContent || '';
                const m = numText.match(/\[(\d+)\]/);
                if (m && refByNumber.has(m[1])) {
                    showPopover(a, refByNumber.get(m[1]));
                    return;
                }
                if (a.title) {
                    showPopover(a, `<span class="reference-title">${a.title}</span>`);
                }
            });
            a.addEventListener('mouseleave', hidePopover);
            a.addEventListener('blur', hidePopover);
        });
    }

    function setupFigureLightbox(content) {
        const figures = [...content.querySelectorAll('.plot-figure, figure')];
        if (figures.length === 0) return;

        const lightbox = document.createElement('div');
        lightbox.className = 'blog-lightbox';
        lightbox.setAttribute('role', 'dialog');
        lightbox.setAttribute('aria-modal', 'true');
        lightbox.innerHTML = `<img class="blog-lightbox-img" alt=""><div class="blog-lightbox-caption"></div>`;
        document.body.appendChild(lightbox);

        const lbImg = lightbox.querySelector('.blog-lightbox-img');
        const lbCap = lightbox.querySelector('.blog-lightbox-caption');

        const closeLightbox = () => lightbox.classList.remove('is-open');
        lightbox.addEventListener('click', closeLightbox);
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && lightbox.classList.contains('is-open')) closeLightbox();
        });

        figures.forEach(fig => {
            const img = fig.querySelector('img');
            if (!img) return;
            const link = img.closest('a');
            const capHtml = fig.querySelector('figcaption')?.innerHTML || '';
            const trigger = link || img;

            trigger.addEventListener('click', e => {
                if (e.metaKey || e.ctrlKey || e.shiftKey) return;
                e.preventDefault();
                lbImg.src = link?.getAttribute('href') || img.currentSrc || img.src;
                lbImg.alt = img.alt || '';
                lbCap.innerHTML = capHtml;
                lightbox.classList.add('is-open');
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();
