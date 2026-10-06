(() => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const soundButton = document.getElementById('sound-toggle');
  let soundEnabled = false;
  let audioContext = null;
  let lastSplashAt = 0;

  function playWaterDrop() {
    if (!soundEnabled) return;
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    try {
      audioContext ||= new AudioContextClass();
      if (audioContext.state === 'suspended') audioContext.resume().catch(() => {});
      const now = audioContext.currentTime;
      const oscillator = audioContext.createOscillator();
      const gain = audioContext.createGain();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(610, now);
      oscillator.frequency.exponentialRampToValueAtTime(205, now + .17);
      gain.gain.setValueAtTime(.0001, now);
      gain.gain.exponentialRampToValueAtTime(.065, now + .016);
      gain.gain.exponentialRampToValueAtTime(.0001, now + .22);
      oscillator.connect(gain);
      gain.connect(audioContext.destination);
      oscillator.start(now);
      oscillator.stop(now + .24);
    } catch (_) {
      // Animation and navigation remain usable if audio is unavailable.
    }
  }

  function splash(x, y) {
    const now = performance.now();
    if (now - lastSplashAt < 120) return;
    lastSplashAt = now;
    if (!reducedMotion.matches) {
      const drop = document.createElement('span');
      drop.className = 'marine-splash';
      drop.setAttribute('aria-hidden', 'true');
      drop.style.left = `${x}px`;
      drop.style.top = `${y}px`;
      drop.innerHTML = '<i></i><i></i><b></b>';
      document.body.appendChild(drop);
      window.setTimeout(() => drop.remove(), 820);
    }
    playWaterDrop();
  }

  soundButton?.addEventListener('click', () => {
    soundEnabled = !soundEnabled;
    soundButton.setAttribute('aria-pressed', String(soundEnabled));
    soundButton.setAttribute('aria-label', soundEnabled ? 'Disable water sounds' : 'Enable water sounds');
    soundButton.title = soundEnabled ? 'Water sounds on' : 'Water sounds off';
    const label = soundButton.querySelector('.sound-toggle__label');
    if (label) label.textContent = soundEnabled ? 'SOUND ON' : 'SOUND OFF';
    if (soundEnabled) playWaterDrop();
  });

  document.addEventListener('pointerdown', event => {
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    if (!event.target.closest('button, a, summary, select, input[type="range"]')) return;
    splash(event.clientX, event.clientY);
  }, { passive: true });

  document.addEventListener('keydown', event => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const target = document.activeElement;
    if (!target?.matches('button, a, summary')) return;
    const bounds = target.getBoundingClientRect();
    splash(bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
  });

  const revealTargets = document.querySelectorAll('.workspace-head, .map-panel, .insights-column > .panel, .timeline, .replay-evidence, .review-view-heading, .review-workflow');
  if ('IntersectionObserver' in window && !reducedMotion.matches) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add('marine-revealed');
        observer.unobserve(entry.target);
      }
    }, { rootMargin: '0px 0px -28px 0px', threshold: .06 });
    revealTargets.forEach(element => {
      element.classList.add('marine-reveal');
      observer.observe(element);
    });
  }
})();
