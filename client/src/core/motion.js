/**
 * 화면 움직임 — 등장, 스크롤 연동, 숫자 올라가기, 흐르는 글자
 *
 * 스크롤은 창이 아니라 .view 안에서 일어나므로 관찰 기준(root)을 .view 로 둔다.
 * '동작 줄이기'(prefers-reduced-motion) 설정이면 모든 효과를 건너뛰고 최종 상태만 보여 준다.
 */

export const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

/** 글자를 단어 단위로 감싸 차례로 떠오르게 한다. <br> 은 줄바꿈으로 유지. */
export function splitWords(el, { delay = 0, step = 70 } = {}) {
  const nodes = [...el.childNodes];
  el.textContent = '';
  let i = 0;
  for (const n of nodes) {
    if (n.nodeType === Node.TEXT_NODE) {
      for (const part of n.textContent.split(/(\s+)/)) {
        if (!part) continue;
        if (/^\s+$/.test(part)) { el.append(' '); continue; }
        const outer = document.createElement('span');
        outer.className = 'w';
        const inner = document.createElement('span');
        inner.textContent = part;
        inner.style.transitionDelay = `${delay + i++ * step}ms`;
        outer.append(inner);
        el.append(outer);
      }
    } else {
      el.append(n);
    }
  }
  el.classList.add('split');
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('in')));
}

/** [data-reveal] 요소가 화면에 들어오면 .in 을 붙인다 (같은 부모 안에서는 조금씩 늦게). */
export function reveal(root, scroller) {
  const items = [...root.querySelectorAll('[data-reveal]')];
  if (reduced() || !('IntersectionObserver' in window)) {
    items.forEach((el) => el.classList.add('in'));
    return () => {};
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const el = e.target;
      const sibs = [...el.parentElement.children].filter((c) => c.hasAttribute('data-reveal'));
      el.style.transitionDelay = `${Math.min(sibs.indexOf(el), 6) * 60}ms`;
      el.classList.add('in');
      io.unobserve(el);
    }
  }, { root: scroller, rootMargin: '0px 0px -8% 0px', threshold: 0.12 });
  items.forEach((el) => io.observe(el));
  return () => io.disconnect();
}

/** 스크롤 위치(0~1, 요소가 화면을 지나가는 정도)를 CSS 변수 --p 로 넘긴다. */
export function scrollProgress(scroller, targets) {
  if (reduced()) return () => {};
  let raf = 0;
  const update = () => {
    raf = 0;
    const vh = scroller.clientHeight;
    const top0 = scroller.getBoundingClientRect().top;
    for (const el of targets) {
      const r = el.getBoundingClientRect();
      const p = Math.min(1, Math.max(0, (vh - (r.top - top0)) / (vh + r.height)));
      el.style.setProperty('--p', p.toFixed(4));
    }
  };
  const onScroll = () => { if (!raf) raf = requestAnimationFrame(update); };
  scroller.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  update();
  return () => {
    scroller.removeEventListener('scroll', onScroll);
    window.removeEventListener('resize', onScroll);
    cancelAnimationFrame(raf);
  };
}

/** 숫자가 화면에 들어오면 0 에서 목표값까지 올라간다. */
export function countUp(el, value, scroller) {
  const target = Number(value);
  if (!Number.isFinite(target) || reduced()) { el.textContent = String(value); return; }
  el.textContent = '0';
  const io = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    io.disconnect();
    const t0 = performance.now();
    const dur = 900;
    const tick = (now) => {
      const k = Math.min(1, (now - t0) / dur);
      const ease = 1 - Math.pow(1 - k, 3);
      el.textContent = String(Math.round(target * ease));
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, { root: scroller, threshold: 0.6 });
  io.observe(el);
}

/** 끝없이 흐르는 글자 띠. 스크롤이 빠를수록 잠깐 빨라진다. */
export function marquee(track, scroller) {
  if (reduced()) return () => {};
  let x = 0, boost = 0, last = scroller.scrollTop, raf = 0, prev = performance.now();
  const onScroll = () => {
    const d = scroller.scrollTop - last;
    last = scroller.scrollTop;
    boost = Math.max(-6, Math.min(6, boost + d * 0.08));
  };
  const loop = (now) => {
    const dt = Math.min(50, now - prev);
    prev = now;
    const half = track.scrollWidth / 2;
    x -= (0.04 + Math.abs(boost) * 0.05) * dt * (boost < -0.5 ? -1 : 1);
    if (half > 0) { if (-x >= half) x += half; if (x > 0) x -= half; }
    boost *= 0.92;
    track.style.transform = `translate3d(${x}px,0,0)`;
    raf = requestAnimationFrame(loop);
  };
  scroller.addEventListener('scroll', onScroll, { passive: true });
  raf = requestAnimationFrame(loop);
  return () => { cancelAnimationFrame(raf); scroller.removeEventListener('scroll', onScroll); };
}
