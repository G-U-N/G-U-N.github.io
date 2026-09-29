/* Disqus discussion for the published article pages. */
(() => {
    const DISQUS_SHORTNAME = 'fuyunwang-blog';
    const PUBLISHED_ORIGIN = 'https://g-u-n.github.io';

    if (!/^[a-z0-9-]+$/i.test(DISQUS_SHORTNAME)) return;
    if (location.origin !== PUBLISHED_ORIGIN) return;

    const stylesheet = document.createElement('link');
    stylesheet.rel = 'stylesheet';
    stylesheet.href = new URL('blog-discussion.css', document.currentScript.src).href;
    document.head.append(stylesheet);

    function mountDiscussion() {
        const article = document.querySelector('.blog-content') || document.querySelector('article');
        if (!article || document.getElementById('disqus_thread')) return;

        const section = document.createElement('section');
        section.className = 'blog-discussion';
        section.setAttribute('aria-label', document.documentElement.lang.startsWith('zh') ? '评论' : 'Discussion');
        const heading = document.createElement('h2');
        heading.textContent = document.documentElement.lang.startsWith('zh') ? '评论' : 'Discussion';
        const thread = document.createElement('div');
        thread.id = 'disqus_thread';
        section.append(heading, thread);
        article.insertAdjacentElement('afterend', section);

        const pageUrl = new URL(location.pathname, PUBLISHED_ORIGIN).href;
        const pageId = location.pathname;
        const pageTitle = document.querySelector('h1')?.textContent.trim() || document.title;
        window.disqus_config = function () {
            this.page.url = pageUrl;
            this.page.identifier = pageId;
            this.page.title = pageTitle;
        };

        const script = document.createElement('script');
        script.src = `https://${DISQUS_SHORTNAME}.disqus.com/embed.js`;
        script.async = true;
        script.setAttribute('data-timestamp', String(Date.now()));
        document.head.append(script);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mountDiscussion, { once: true });
    } else {
        mountDiscussion();
    }
})();
