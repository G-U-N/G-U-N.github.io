// A scalar top-2 example. Both the drawing and the combined values use this data.
export const tokens = [1, 2, 3, 4];
export const selections = [[[0, .7], [1, .3]], [[0, .4], [2, .6]], [[1, .25], [2, .75]], [[0, .6], [3, .4]]];
export const experts = [x => x + 1, x => 2 * x + 6, x => 3 * x, x => 5 - x];
export function dispatch() {
  return experts.map((fn, expert) => tokens.flatMap((x, token) =>
    selections[token].flatMap(([e, weight], slot) => e === expert
      ? [{ token, slot, expert, weight, source: Math.floor(token / 2), owner: Math.floor(expert / 2), x, value: fn(x) }]
      : [])));
}
export function combine(batches) {
  const output = tokens.map(() => 0);
  for (const batch of batches) for (const row of batch) output[row.token] += row.weight * row.value;
  return output;
}

if (typeof document !== 'undefined') {
  const fold = document.getElementById('moe-dispatch');
  let mounted = false;
  fold?.addEventListener('toggle', function init() {
    if (!fold.open || mounted) return;
    fold.removeEventListener('toggle', init);
    mounted = true;
    mount(fold);
  });
  if (fold?.open) { mounted = true; mount(fold); }
}

function mount(fold) {
  const root = fold.querySelector('.moe-animation');
  const NS = 'http://www.w3.org/2000/svg';
  const colors = ['#167d87', '#b65846', '#7954a1', '#a0711f'];
  const tints = ['#edf7f7', '#fcf1ed', '#f4eff9', '#faf4e8'];
  const sub = n => String(n).replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d]);
  const format = n => Number(n.toFixed(2)).toString();
  const steps = ['Tokens', 'Top-2', 'Dispatch', 'Experts', 'Return', 'Combine'];
  const notes = [
    'Each GPU starts with two tokens and owns two experts.',
    'Make two copies of each token, one for each selected expert.',
    'Send copies to their expert owners. Local routes stay on the same GPU.',
    'Each expert processes its own batch: 3, 2, 2 and 1 rows.',
    'Return the expert outputs to the GPU where each token started.',
    'Multiply by the routing weights, then sum to get one output per token.'
  ];
  root.innerHTML = `<div class="moe-animation-heading"><span class="moe-animation-kicker">FOLLOW THE TOKENS</span><h3>One trip through Expert Parallel</h3></div>
    <div class="moe-animation-steps" role="group" aria-label="Animation steps">${steps.map((s, i) => `<button type="button" data-step="${i}">${i + 1}. ${s}</button>`).join('')}</div>
    <p class="moe-animation-note" aria-live="polite"></p>
    <svg viewBox="0 0 540 556" role="img" aria-labelledby="moe-animation-title moe-animation-desc"><title id="moe-animation-title">Token dispatch, expert computation and weighted combine</title><desc id="moe-animation-desc">Four tokens select two experts each. Experts 0 and 1 live on GPU 0; experts 2 and 3 live on GPU 1. Outputs return to their source GPU and are combined using their routing weights.</desc></svg>
    <div class="moe-animation-example" aria-live="polite"></div>
    <div class="moe-animation-controls"><button type="button" data-play>Play</button><button type="button" data-replay>Replay</button><label>Progress <input type="range" min="0" max="5" step="0.01" value="0" aria-label="Animation progress"></label></div>`;
  const svg = root.querySelector('svg');
  function el(tag, attrs, parent = svg, value) {
    const node = document.createElementNS(NS, tag);
    for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, val);
    if (value !== undefined) node.textContent = value;
    parent.append(node);
    return node;
  }
  const txt = (x, y, value, attrs = {}, parent = svg) => el('text', { x, y, fill: '#243047', 'font-size': 18, ...attrs }, parent, value);
  for (let gpu = 0; gpu < 2; gpu++) {
    const x = 10 + gpu * 278;
    el('rect', { x, y: 12, width: 242, height: 531, rx: 16, fill: '#fafbfe', stroke: '#dde3ed' });
    txt(x + 16, 45, `GPU ${gpu}`, { 'font-weight': 700, 'font-size': 23 });
    txt(x + 16, 71, `owns e${sub(2 * gpu)}, e${sub(2 * gpu + 1)}`, { fill: '#657086' });
    txt(x + 16, 103, 'Token inputs / outputs', { 'font-size': 16, fill: '#657086' });
    for (let local = 0; local < 2; local++) {
      const expert = 2 * gpu + local, y = 286 + local * 132;
      el('rect', { x: x + 12, y, width: 218, height: 114, rx: 10, fill: tints[expert], stroke: colors[expert] });
      txt(x + 25, y + 26, `Expert ${expert}`, { fill: colors[expert], 'font-size': 19, 'font-weight': 600 });
    }
  }
  // Thin lanes locate the two tokens on each source GPU throughout the animation.
  const sourceCenter = token => ({ x: 131 + Math.floor(token / 2) * 278, y: 149 + (token % 2) * 85 });
  const originals = tokens.map((x, token) => {
    const p = sourceCenter(token);
    el('rect', { x: p.x - 109, y: p.y - 23, width: 218, height: 47, rx: 10, fill: 'white', stroke: '#e2e6ee', 'stroke-dasharray': '4 4' });
    const node = el('g', {});
    el('rect', { x: p.x - 54, y: p.y - 18, width: 108, height: 36, rx: 9, fill: 'white', stroke: '#7b879d' }, node);
    const label = txt(p.x, p.y + 6, `x${sub(token)} = ${x}`, { 'text-anchor': 'middle', 'font-weight': 600, 'font-size': 20 }, node);
    return { node, label };
  });
  const batches = dispatch(), output = combine(batches);
  const copies = batches.flatMap((batch, expert) => batch.map((row, index) => {
    const group = el('g', {});
    el('rect', { x: -48, y: -12, width: 96, height: 24, rx: 6, fill: 'white', stroke: colors[expert], 'stroke-width': 1.5 }, group);
    const label = txt(0, 6, '', { fill: colors[expert], 'text-anchor': 'middle', 'font-size': 17, 'font-weight': 600 }, group);
    const center = sourceCenter(row.token);
    return { ...row, group, label, center,
      start: { x: center.x + (row.slot ? 55 : -55), y: center.y },
      end: { x: 131 + row.owner * 278, y: 330 + (expert % 2) * 132 + index * 25 }
    };
  }));
  const play = root.querySelector('[data-play]'), range = root.querySelector('input');
  const note = root.querySelector('.moe-animation-note'), example = root.querySelector('.moe-animation-example');
  const buttons = [...root.querySelectorAll('[data-step]')];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let progress = 0, playing = false, frame = 0, lastTime = null, currentStep = -1;
  const ease = x => { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };
  const lerp = (a, b, u) => ({ x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u });
  function draw() {
    const step = Math.min(5, Math.ceil(progress));
    if (step !== currentStep) {
      currentStep = step;
      note.textContent = notes[step];
      buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(i === step)));
      example.textContent = step < 3
        ? 'Trace x₀: 70% to expert 0 + 30% to expert 1.'
        : step < 5 ? 'For x₀ = 1, the toy experts return 2 and 8.'
        : 'y₀ = 0.7 × 2 + 0.3 × 8 = 3.8';
    }
    originals.forEach(({ node, label }, token) => {
      node.setAttribute('opacity', progress < 1 ? 1 - ease(progress) : ease(progress - 4));
      label.textContent = progress < 1 ? `x${sub(token)} = ${tokens[token]}` : `y${sub(token)} = ${format(output[token])}`;
    });
    copies.forEach(c => {
      let p;
      if (progress < 1) p = lerp(c.center, c.start, ease(progress));
      else if (progress < 2) p = lerp(c.start, c.end, ease(progress - 1));
      else if (progress < 3) p = c.end;
      else if (progress < 4) p = lerp(c.end, c.start, ease(progress - 3));
      else p = lerp(c.start, c.center, ease(progress - 4));
      c.group.setAttribute('transform', `translate(${p.x},${p.y})`);
      c.group.setAttribute('opacity', progress < 1 ? ease(progress) : 1 - ease(progress - 4));
      c.label.textContent = progress < 2.5 ? `x${sub(c.token)} → e${sub(c.expert)}` : `x${sub(c.token)}·e${sub(c.expert)}: ${format(c.value)}`;
      c.label.setAttribute('font-size', progress < 2.5 ? '17' : '16');
    });
    range.value = String(progress);
  }
  function pause() { playing = false; cancelAnimationFrame(frame); lastTime = null; play.textContent = reduced.matches ? 'Next step' : 'Play'; }
  function tick(now) {
    if (!playing) return;
    if (lastTime !== null) progress = Math.min(5, progress + Math.min(now - lastTime, 80) / 2200);
    lastTime = now;
    draw();
    if (progress >= 5) pause(); else frame = requestAnimationFrame(tick);
  }
  function start() {
    if (reduced.matches) { pause(); progress = Math.min(5, Math.floor(progress) + 1); draw(); return; }
    if (progress >= 5) progress = 0;
    playing = true; lastTime = null; play.textContent = 'Pause'; frame = requestAnimationFrame(tick);
  }
  play.addEventListener('click', () => playing ? pause() : start());
  root.querySelector('[data-replay]').addEventListener('click', () => { pause(); progress = 0; draw(); if (!reduced.matches) start(); });
  buttons.forEach((b, i) => b.addEventListener('click', () => { pause(); progress = i; draw(); }));
  range.addEventListener('input', () => { pause(); progress = Number(range.value); draw(); });
  fold.addEventListener('toggle', () => { if (!fold.open) pause(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  reduced.addEventListener('change', pause);
  new IntersectionObserver(entries => { if (!entries[0].isIntersecting) pause(); }).observe(root);
  pause(); draw();
}
