(() => {
  const experience = document.getElementById('experience');
  const workspace = document.getElementById('workspace');
  const stage = experience?.querySelector('.experience-stage');
  const launch = document.getElementById('launch-screen');
  const enter = document.getElementById('launch-enter');
  const skip = document.getElementById('launch-skip');
  if (!experience || !workspace || !stage || !launch || !enter || !skip) return;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const clamp = (value, low = 0, high = 1) => Math.min(high, Math.max(low, value));
  const smoothstep = (start, end, value) => {
    const t = clamp((value - start) / (end - start));
    return t * t * (3 - 2 * t);
  };
  let animationFrame = 0;

  function render() {
    animationFrame = 0;
    if (reducedMotion.matches) return;
    const travel = Math.max(1, experience.offsetHeight - window.innerHeight);
    const progress = clamp(-experience.getBoundingClientRect().top / travel);
    const sceneOne = 1 - smoothstep(.08, .28, progress);
    const sceneTwo = smoothstep(.20, .34, progress) * (1 - smoothstep(.48, .65, progress));
    const sceneThree = smoothstep(.56, .73, progress) * (1 - smoothstep(.91, 1, progress));
    const icebergFade = 1 - smoothstep(.76, .96, progress);
    const exit = smoothstep(.90, 1, progress);
    const values = {
      '--orbit-shift': progress,
      '--scene-one': sceneOne,
      '--scene-two': sceneTwo,
      '--scene-three': sceneThree,
      '--earth-dim': smoothstep(.28, .75, progress) * .35,
      '--hero-exit': exit,
      '--iceberg-opacity': icebergFade,
      '--iceberg-x': smoothstep(.08, .72, progress) * -5,
      '--iceberg-y': -smoothstep(.12, .8, progress) * 7,
      '--iceberg-scale': .9 + smoothstep(0, .82, progress) * .32,
      '--sea-shift': smoothstep(0, 1, progress) * 8
    };
    for (const [name, value] of Object.entries(values)) stage.style.setProperty(name, value.toFixed(4));
  }

  function scheduleRender() {
    if (!animationFrame) animationFrame = requestAnimationFrame(render);
  }

  function closeLaunch() {
    document.body.classList.add('launch-entered');
    document.body.classList.remove('launch-locked');
    experience.inert = false;
    workspace.inert = false;
    launch.inert = true;
    launch.setAttribute('aria-hidden', 'true');
    scheduleRender();
  }

  enter.addEventListener('click', () => {
    closeLaunch();
    window.scrollTo({ top: 0, behavior: 'instant' });
    experience.setAttribute('tabindex', '-1');
    experience.focus({ preventScroll: true });
  });
  skip.addEventListener('click', () => {
    closeLaunch();
    workspace.setAttribute('tabindex', '-1');
    window.setTimeout(() => workspace.focus({ preventScroll: true }), 100);
  });
  window.addEventListener('scroll', scheduleRender, { passive: true });
  window.addEventListener('resize', scheduleRender, { passive: true });
  reducedMotion.addEventListener?.('change', () => {
    if (reducedMotion.matches && document.body.classList.contains('launch-ready')) closeLaunch();
    scheduleRender();
  });
  scheduleRender();

  if (reducedMotion.matches || location.hash === '#workspace' || location.hash.startsWith('#review')) return;
  document.body.classList.add('experience-loading');
  const imageReady = path => {
    const image = new Image();
    image.src = path;
    return image.decode ? image.decode().catch(() => {}) : Promise.resolve();
  };
  Promise.all([
    imageReady('antarctic-sea.png'),
    imageReady('hero-iceberg.png'),
    new Promise(resolve => window.setTimeout(resolve, 800))
  ]).then(() => {
    document.body.classList.add('experience-loaded', 'launch-ready', 'launch-locked');
    experience.inert = true;
    workspace.inert = true;
    enter.focus({ preventScroll: true });
    window.setTimeout(() => document.body.classList.remove('experience-loading'), 750);
  });
})();
