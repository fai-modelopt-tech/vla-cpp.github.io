(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const header = document.querySelector(".site-header");

  const updateHeader = () => {
    if (!header) return;
    header.classList.toggle("is-scrolled", window.scrollY > 8);
  };

  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });

  const revealSelectors = [
    ".section h2",
    ".section-intro",
    ".media-main",
    ".media-card",
    ".keyword-list span",
    ".contribution-grid article",
    ".metric",
    ".figure",
    ".feature-list article",
    ".table-block",
    ".libero-video-card",
    ".aloha-highlight",
    ".chart-card",
    ".latency-grid article",
    ".video-evidence-panel",
    ".result-panel",
    ".memory-card",
    ".comparison-grid article",
    ".citation-box",
  ];

  const elements = Array.from(document.querySelectorAll(revealSelectors.join(",")));

  elements.forEach((element, index) => {
    element.classList.add("reveal");
    element.style.setProperty("--reveal-delay", `${Math.min(index % 8, 7) * 45}ms`);
  });

  document.querySelectorAll(".chart-card").forEach((card) => {
    card.querySelectorAll(".bar-fill").forEach((bar, index) => {
      bar.style.setProperty("--bar-delay", `${index * 70}ms`);
    });
  });

  if (reduceMotion || !("IntersectionObserver" in window)) {
    elements.forEach((element) => element.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    {
      rootMargin: "0px 0px -8% 0px",
      threshold: 0.12,
    },
  );

  elements.forEach((element) => observer.observe(element));
})();
