/* Shared behavior only. Edit colors and card appearance in blog-themes.css.
 * Load this small script in <head> without defer to restore the theme before paint.
 * New article: include blog-themes.css last, then this script, in its <head>.
 */
(() => {
    const STORAGE_KEY = 'blog-theme';
    const DEFAULT_THEME = 'classic';
    const THEMES = [
        { id: 'classic', label: 'Classic' },
        { id: 'paper', label: 'Paper' }
    ];
    const normalize = value => THEMES.some(theme => theme.id === value) ? value : DEFAULT_THEME;
    let selected = DEFAULT_THEME;
    let button;
    function readSaved() {
        try { return normalize(localStorage.getItem(STORAGE_KEY)); }
        catch { return selected; }
    }
    function apply(value) {
        selected = normalize(value);
        document.documentElement.dataset.blogTheme = selected;
        if (!button) return;
        const index = THEMES.findIndex(theme => theme.id === selected);
        const current = THEMES[index], next = THEMES[(index + 1) % THEMES.length];
        button.textContent = `Theme: ${current.label}`;
        button.setAttribute('aria-label', `Theme: ${current.label}. Switch to ${next.label}`);
        button.title = `Switch to ${next.label} theme`;
    }
    apply(readSaved());

    function mount() {
        const header = document.querySelector('.blog-header');
        if (!header || header.querySelector('.blog-theme-toggle')) return;
        const toolbar = document.createElement('div');
        toolbar.className = 'blog-tools';
        const back = header.querySelector('.back-link');
        if (back) toolbar.append(back);
        button = document.createElement('button');
        button.type = 'button';
        button.className = 'blog-theme-toggle';
        button.addEventListener('click', () => {
            const index = THEMES.findIndex(theme => theme.id === selected);
            apply(THEMES[(index + 1) % THEMES.length].id);
            try { localStorage.setItem(STORAGE_KEY, selected); }
            catch { /* Switching still works when storage is unavailable. */ }
        });
        toolbar.append(button);
        header.prepend(toolbar);
        apply(selected);
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, { once: true });
    else mount();
    window.addEventListener('storage', event => {
        if (event.key === STORAGE_KEY || event.key === null) apply(readSaved());
    });
    window.addEventListener('pageshow', () => apply(readSaved()));
})();
