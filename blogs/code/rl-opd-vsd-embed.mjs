// Load the animation only when its disclosure is opened.
const disclosure = document.getElementById('watch-updates');
if (disclosure) {
  const frame = disclosure.querySelector('iframe');
  const syncVisibility = () => {
    if (disclosure.open && !frame.hasAttribute('src')) frame.src = frame.dataset.src;
    if (frame.hasAttribute('src')) frame.contentWindow?.postMessage({
      type: 'rl-opd-vsd-visibility', visible: disclosure.open
    }, location.origin);
  };
  disclosure.addEventListener('toggle', syncVisibility);
  frame.addEventListener('load', syncVisibility);
  window.addEventListener('message', event => {
    if (event.origin !== location.origin || event.source !== frame.contentWindow) return;
    if (event.data?.type !== 'rl-opd-vsd-height') return;
    const height = event.data.height;
    if (Number.isFinite(height) && height >= 200 && height <= 2400) {
      frame.style.height = `${Math.ceil(height)}px`;
    }
  });
  syncVisibility();
}
