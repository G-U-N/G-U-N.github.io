// Deterministic calculations, shared by the preview and numerical verification.
const normal = (x, mu = 0, sigma = 1) => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
const sum = xs => xs.reduce((a, b) => a + b, 0);
const normalize = xs => { const total = sum(xs); return xs.map(x => x / total); };
const softmax = xs => normalize(xs.map(x => Math.exp(x - Math.max(...xs))));
export function probabilityFrames(steps = 360) {
  const positions = Array.from({length: 13}, (_, i) => -3 + i * 0.5);
  const target = normalize(positions.map(x => normal(x, 1, 0.85)));
  // A small uniform component keeps every categorical outcome represented initially.
  const initial = normalize(positions.map(x => normal(x, -1.2, 0.85)));
  let logits = initial.map(p => Math.log(.97 * p + .03 / positions.length));
  const frames = [];
  for (let k = 0; k <= steps; k++) {
    const q = softmax(logits), reward = q.map((v, i) => Math.log(target[i] / v));
    const baseline = sum(q.map((v, i) => v * reward[i]));
    frames.push({q, reward, kl: -baseline});
    logits = logits.map((v, i) => v + 2 * q[i] * (reward[i] - baseline));
  }
  return {positions, target, frames};
}
export function gaussianSamples(count = 512, seed = 1739) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let total = 0;
  return Array.from({length: count}, (_, i) => {
    const epsilon = Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
    const cost = 0.5 - epsilon, gradient = cost * epsilon;
    total += gradient;
    return {epsilon, cost, gradient, average: total / (i + 1)};
  });
}

if (typeof document !== 'undefined') {
  let hostVisible = true;
  const embedded = window.parent !== window && document.documentElement.classList.contains('embed-views');
  if (embedded) {
    window.addEventListener('message', event => {
      if (event.origin === location.origin && event.source === window.parent && event.data?.type === 'rl-opd-vsd-visibility') {
        hostVisible = event.data.visible === true;
      }
    });
    const reportHeight = () => window.parent.postMessage({
      type: 'rl-opd-vsd-height', height: document.querySelector('main').getBoundingClientRect().height
    }, location.origin);
    new ResizeObserver(reportHeight).observe(document.querySelector('main'));
    window.addEventListener('load', reportHeight);
    document.fonts.ready.then(reportHeight);
  }
  const NS = 'http://www.w3.org/2000/svg';
  const colors = {purple:'#7863bb', green:'#258571', coral:'#cd815c', gray:'#737b8d'};
  const $ = id => document.getElementById(id);
  const set = (node, attrs) => { for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v); return node; };
  const el = (parent, type, attrs = {}, text) => { const n = set(document.createElementNS(NS, type), attrs); if (text !== undefined) n.textContent = text; parent.append(n); return n; };
  const line = (parent, x1, y1, x2, y2, attrs = {}) => el(parent, 'line', {x1,y1,x2,y2,...attrs});
  const text = (parent, x, y, value, attrs = {}) => el(parent, 'text', {x,y,...attrs}, value);
  const path = points => points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');
  const fixed = n => (Math.abs(n) < .0005 ? 0 : n).toFixed(3);
  const signed = n => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(2)}`;
  const marker = (svg, id, color) => { const defs = el(svg, 'defs'); const m = el(defs, 'marker', {id,viewBox:'0 0 8 8',refX:7,refY:4,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});el(m,'path',{d:'M0 0 L8 4 L0 8 Z',fill:color}); };
  const gaussianCurve = (mu, sigma, X, Y, lo = -4, hi = 4) => Array.from({length:161},(_,i)=>{const x=lo+(hi-lo)*i/160;return [X(x),Y(normal(x,mu,sigma))];});

  const {positions, target, frames} = probabilityFrames();
  const prob = $('probability-plot'), particle = $('particle-plot');
  const X = x => 42 + (x + 4) / 8 * 386;
  const barY = p => 182 - p / .52 * 150;
  [0,.2,.4].forEach(p => {line(prob,34,barY(p),438,barY(p),{class:p?'grid':'axis'});text(prob,25,barY(p)+4,p.toFixed(1),{'text-anchor':'end',class:'small'});});
  const bars = positions.map((x,i)=>{
    const targetBar = el(prob,'rect',{x:X(x)-11,y:barY(target[i]),width:22,height:182-barY(target[i]),rx:3,fill:'#25857108',stroke:colors.green,'stroke-width':1.3});
    const current = el(prob,'rect',{x:X(x)-7,y:182,width:14,height:0,rx:2,fill:colors.purple,opacity:.88});
    return {targetBar,current};
  });
  [-3,-2,-1,0,1,2,3].forEach(x=>text(prob,X(x),202,x,{'text-anchor':'middle',class:'small'}));
  text(prob,42,230,'Log-ratio reward', {class:'small'});
  text(prob,428,230,'+ favors · − suppresses',{'text-anchor':'end',class:'small'});
  line(prob,42,258,428,258,{class:'axis'});
  const rewardBars = positions.map(x=>el(prob,'rect',{x:X(x)-4,width:8,rx:2,y:258,height:0}));
  text(prob,235,285,'Fixed outcome positions',{'text-anchor':'middle',class:'small'});

  const densityY = y => 176 - y * 300;
  [0,.2,.4].forEach(p=>{line(particle,34,densityY(p),438,densityY(p),{class:p?'grid':'axis'});text(particle,25,densityY(p)+4,p.toFixed(1),{'text-anchor':'end',class:'small'});});
  const fillPath = el(particle,'path',{fill:'#7863bb14',stroke:'none'});
  el(particle,'path',{d:path(gaussianCurve(1,.85,X,densityY)),fill:'none',stroke:colors.green,'stroke-width':2,'stroke-dasharray':'5 4'});
  const studentPath = el(particle,'path',{fill:'none',stroke:colors.purple,'stroke-width':2.4});
  [-3,-2,-1,0,1,2,3].forEach(x=>text(particle,X(x),196,x,{'text-anchor':'middle',class:'small'}));
  text(particle,42,224,'Same particles, new positions',{class:'small'});
  // Numerically invert the normal CDF on a fine grid for a stable, representative particle set.
  const quantiles = [];
  let cdf = 0, qi = 0;
  for(let x=-5;x<=5 && qi<35;x+=.001){cdf+=normal(x)*.001;if(cdf >= (qi+.5)/35){quantiles.push(x);qi++;}}
  const dots = quantiles.map((z,i)=>el(particle,'circle',{r:3.1,cy:249+(i%3-1)*7,fill:colors.purple,opacity:.8}));
  marker(particle,'motion-arrow',colors.purple);
  const arrows = [-1.1,0,1.1].map(z=>({z,node:line(particle,0,275,0,275,{stroke:colors.purple,'stroke-width':1.5,'marker-end':'url(#motion-arrow)'})}));
  function drawViews(progress) {
    // Interpolate between solver steps for smooth playback; spend more time on early steps.
    const k = frames.length ** progress - 1, index = Math.min(frames.length - 1, Math.floor(k));
    const next = frames[Math.min(index + 1, frames.length - 1)], fraction = k - index;
    const q = frames[index].q.map((v, i) => v + fraction * (next.q[i] - v));
    const reward = q.map((v, i) => Math.log(target[i] / v));
    const f = {q, reward, kl: -sum(q.map((v, i) => v * reward[i]))};
    bars.forEach(({current},i)=>set(current,{y:barY(f.q[i]),height:182-barY(f.q[i])}));
    rewardBars.forEach((b,i)=>{const h=Math.min(19,Math.abs(f.reward[i])*2);set(b,{y:f.reward[i]>=0?258-h:258,height:h,fill:f.reward[i]>=0?colors.green:colors.coral});});
    $('discrete-kl').textContent = fixed(f.kl);
    const steps = progress*65, mu = 1-2.2*(1-.07/.85**2)**steps;
    const pts=gaussianCurve(mu,.85,X,densityY);
    set(studentPath,{d:path(pts)});set(fillPath,{d:path(pts)+` L${X(4)},176 L${X(-4)},176 Z`});
    dots.forEach((d,i)=>set(d,{cx:X(mu+.85*quantiles[i])}));
    arrows.forEach(({z,node})=>{const start=X(mu+.85*z),len=(1-mu)/.85**2*12;set(node,{x1:start,x2:start+len,opacity:Math.min(1,len/4)});});
    $('continuous-kl').textContent = fixed((mu-1)**2/(2*.85**2));
  }

  const samples = gaussianSamples();
  const sampleSvg=$('sample-plot'), avgSvg=$('average-plot');
  const SX=x=>42+(x+4)/8*386, SY=y=>149-y*265;
  line(sampleSvg,34,149,438,149,{class:'axis'});
  el(sampleSvg,'path',{d:path(gaussianCurve(0,1,SX,SY))+` L${SX(4)},149 L${SX(-4)},149 Z`,fill:'#7863bb12'});
  el(sampleSvg,'path',{d:path(gaussianCurve(1,1,SX,SY)),fill:'none',stroke:colors.green,'stroke-width':2,'stroke-dasharray':'5 4'});
  el(sampleSvg,'path',{d:path(gaussianCurve(0,1,SX,SY)),fill:'none',stroke:colors.purple,'stroke-width':2});
  text(sampleSvg,SX(0)-13,32,'q',{'text-anchor':'end',style:`fill:${colors.purple}`});text(sampleSvg,SX(1)+13,32,'p',{style:`fill:${colors.green}`});
  [-3,-2,-1,0,1,2,3].forEach(x=>text(sampleSvg,SX(x),167,x,{'text-anchor':'middle',class:'small'}));
  const sampleStem=line(sampleSvg,0,149,0,149,{stroke:colors.coral,'stroke-width':1,'stroke-dasharray':'3 3'});
  const sampleDot=el(sampleSvg,'circle',{r:4.6,fill:colors.coral,stroke:'white','stroke-width':2});
  const sampleLabel=text(sampleSvg,0,0,'sample a',{style:`fill:${colors.coral}`,class:'small'});
  const gradientLo = Math.floor(Math.min(-2,...samples.map(s=>s.gradient)))-1;
  const GX=g=>42+(g-gradientLo)/(1-gradientLo)*386;
  text(sampleSvg,42,204,'A single gradient can fluctuate widely',{class:'small'});
  line(sampleSvg,42,242,428,242,{class:'axis'});
  [gradientLo,-1,0,1].forEach(g=>{line(sampleSvg,GX(g),238,GX(g),247,{stroke:'#c9ccd8'});text(sampleSvg,GX(g),266,g,{'text-anchor':'middle',class:'small'});});
  line(sampleSvg,GX(-1),222,GX(-1),251,{stroke:colors.green,'stroke-dasharray':'3 3','stroke-width':1.3});
  marker(sampleSvg,'gradient-arrow',colors.coral);
  const gradientArrow=line(sampleSvg,GX(0),232,GX(0),232,{stroke:colors.coral,'stroke-width':2.4,'marker-end':'url(#gradient-arrow)'});

  const AX=n=>49+Math.log(n)/Math.log(512)*375;
  const low=Math.min(-1.7,Math.floor(Math.min(...samples.map(s=>s.average))*2)/2-.25);
  const high=Math.max(.25,Math.ceil(Math.max(...samples.map(s=>s.average))*2)/2+.25);
  const AY=g=>238-(g-low)/(high-low)*202;
  for(let g=Math.ceil(low*2)/2;g<=high;g+=.5){line(avgSvg,49,AY(g),424,AY(g),{class:'grid'});text(avgSvg,39,AY(g)+4,g.toFixed(1),{'text-anchor':'end',class:'small'});}
  line(avgSvg,49,238,424,238,{class:'axis'});
  [1,8,64,512].forEach(n=>{line(avgSvg,AX(n),238,AX(n),242,{class:'axis'});text(avgSvg,AX(n),258,n,{'text-anchor':'middle',class:'small'});});
  text(avgSvg,237,282,'Number of samples · log scale',{'text-anchor':'middle',class:'small'});
  line(avgSvg,49,AY(-1),424,AY(-1),{stroke:colors.green,'stroke-width':1.5,'stroke-dasharray':'5 4'});
  text(avgSvg,422,AY(-1)-9,'MSE = −1',{'text-anchor':'end',style:`fill:${colors.green}`,class:'small'});
  const averagePath=el(avgSvg,'path',{fill:'none',stroke:colors.purple,'stroke-width':2.5,'stroke-linejoin':'round'});
  const averageDot=el(avgSvg,'circle',{r:4.3,fill:colors.purple,stroke:'white','stroke-width':1.5});
  function drawGaussian(progress) {
    const n=Math.max(1,Math.min(512,Math.round(512**progress))), s=samples[n-1];
    $('sample-count').textContent=n;$('noise-value').textContent=signed(s.epsilon);$('cost-value').textContent=signed(s.cost);$('gradient-value').textContent=signed(s.gradient);$('average-value').textContent=signed(s.average);
    const x=SX(s.epsilon),y=SY(normal(s.epsilon));
    set(sampleStem,{x1:x,x2:x,y2:y});set(sampleDot,{cx:x,cy:y});set(sampleLabel,{x:x>365?x-9:x+9,y:y-12,'text-anchor':x>365?'end':'start'});
    set(gradientArrow,{x2:GX(s.gradient),opacity:Math.abs(s.gradient)<.025?0:1});
    set(averagePath,{d:path(samples.slice(0,n).map((v,i)=>[AX(i+1),AY(v.average)]))});
    set(averageDot,{cx:AX(n),cy:AY(s.average)});
  }

  class Player {
    constructor(root,draw,duration){
      this.root=root;this.draw=draw;this.duration=duration;this.progress=0;this.playing=false;this.visible=false;this.started=false;this.last=0;
      this.play=root.querySelector('[data-action=play]');this.replay=root.querySelector('[data-action=replay]');this.slider=root.querySelector('input');this.label=root.querySelector('.progress-label');
      this.play.addEventListener('click',()=>{this.started=true;if(this.progress===1)this.progress=0;this.playing=!this.playing;this.update();});
      this.replay.addEventListener('click',()=>{this.started=true;this.progress=0;this.playing=true;this.update();});
      this.slider.addEventListener('input',()=>{this.started=true;this.playing=false;this.progress=Number(this.slider.value)/1000;this.update();});
      new IntersectionObserver(([entry])=>{this.visible=entry.isIntersecting;if(this.visible&&!this.started){this.started=true;this.playing=!matchMedia('(prefers-reduced-motion: reduce)').matches;this.update();}},{threshold:.1}).observe(root);
      this.update();requestAnimationFrame(t=>this.tick(t));
    }
    update(){this.draw(this.progress);this.slider.value=Math.round(this.progress*1000);this.label.textContent=`${Math.round(this.progress*100)}%`;this.play.textContent=this.playing?'Pause':this.progress===1?'Play again':'Play';this.play.setAttribute('aria-label',`${this.playing?'Pause':'Play'} ${this.root.dataset.animation==='views'?'probability and motion':'Gaussian gradient'} animation`);}
    tick(t){const dt=Math.min(100,t-this.last);this.last=t;if(this.playing&&this.visible&&hostVisible&&!document.hidden){this.progress=Math.min(1,this.progress+dt/this.duration);if(this.progress===1)this.playing=false;this.update();}requestAnimationFrame(t=>this.tick(t));}
  }
  new Player(document.querySelector('[data-animation=views]'),drawViews,18000);
  new Player(document.querySelector('[data-animation=gaussian]'),drawGaussian,28000);
}
