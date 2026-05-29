(function () {
  const evidence = Array.isArray(window.ALOHA_EVIDENCE) ? window.ALOHA_EVIDENCE : [];
  const gallery = document.getElementById("evidence-gallery");
  const count = document.getElementById("evidence-count");
  const modal = document.getElementById("evidence-modal");
  const modalMedia = document.getElementById("modal-media");
  const modalTitle = document.getElementById("modal-title");
  const modalMeta = document.getElementById("modal-meta");
  const prevButton = document.getElementById("evidence-prev");
  const nextButton = document.getElementById("evidence-next");
  const pageStatus = document.getElementById("evidence-page");
  const filters = {
    task: document.getElementById("filter-task"),
    result: document.getElementById("filter-result"),
    setup: document.getElementById("filter-setup"),
  };

  if (!gallery || !count || !modal || !modalMedia || !modalTitle || !modalMeta) return;

  let pageIndex = 0;

  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const variantOrder = {
    "Baseline-Pytorch-BF16": 0,
    "vla.cpp-BF16": 1,
  };

  const taskOrder = {
    "Task 1": 0,
    "Task 2": 1,
  };

  const variantLabels = {
    "Baseline-Pytorch-BF16": "pytorch",
    "vla.cpp-BF16": "vla.cpp",
  };

  const cameraViews = [
    { label: "overview", suffix: "overview" },
    { label: "wristleft", suffix: "wristleft" },
    { label: "high", suffix: "high" },
  ];

  const compareEvidence = (a, b) =>
    (taskOrder[a.task] ?? 99) - (taskOrder[b.task] ?? 99) ||
    Number(a.trial) - Number(b.trial) ||
    (variantOrder[a.variant] ?? 99) - (variantOrder[b.variant] ?? 99);

  const variantName = (variant) => (variant === "vla.cpp-BF16" ? "vla.cpp" : "pytorch");
  const trialTitle = (item) =>
    `${variantName(item.variant)} · ${String(item.task).toLowerCase()} · trial ${String(item.trial).padStart(2, "0")}`;

  const selected = (key) => {
    const control = filters[key];
    return control && control.value !== "all" ? control.value : "";
  };

  const taskSlug = (task) => (task === "Task 1" ? "task1" : "task2");
  const setupImage = (task) => `assets/aloha-setup/${taskSlug(task)}-evidence.jpeg`;

  const trialVideoPath = (item, view, ext = "mp4") => {
    const replacement = `-${view.suffix}.${ext}`;
    return String(item?.video ?? "")
      .replace(/-iphone\.mp4$/, replacement)
      .replace(/-overview\.mp4$/, replacement)
      .replace(/-wristleft\.mp4$/, replacement)
      .replace(/-high\.mp4$/, replacement);
  };

  const groupEvidence = () => {
    const base = evidence.filter((item) => {
      if (selected("task") && item.task !== selected("task")) return false;
      if (selected("setup") && item.setup !== selected("setup")) return false;
      return true;
    });

    const groups = new Map();
    base.sort(compareEvidence).forEach((item) => {
      const key = `${item.task}::${item.trial}`;
      if (!groups.has(key)) {
        groups.set(key, {
          task: item.task,
          trial: Number(item.trial),
          setup: item.setup,
          items: [],
        });
      }
      groups.get(key).items.push(item);
    });

    return Array.from(groups.values())
      .filter((group) => {
        if (!selected("result")) return true;
        return group.items.some((item) => item.result === selected("result"));
      })
      .sort(
        (a, b) =>
          (taskOrder[a.task] ?? 99) - (taskOrder[b.task] ?? 99) ||
          Number(a.trial) - Number(b.trial),
      );
  };

  const videoCell = (item, view) => {
    if (!item) {
      return '<article class="trial-video-card trial-video-card-empty"><div class="trial-video-pending">missing trial</div></article>';
    }

    const path = trialVideoPath(item, view);
    const label = `${variantLabels[item.variant] ?? item.variant} · ${view.label}`;
    const resultClass = item.result === "PASS" ? "pass" : "fail";

    return `
      <article class="trial-video-card">
        <video controls preload="metadata" data-pending-path="${escapeHtml(path)}">
          <source src="${escapeHtml(path)}" type="video/mp4">
          your browser does not support the video tag.
        </video>
        <div class="trial-video-caption">
          <span>${escapeHtml(label)}</span>
          <em class="${resultClass}">${escapeHtml(String(item.result).toLowerCase())}</em>
        </div>
      </article>
    `;
  };

  const bindInlineVideoFallbacks = () => {
    gallery.querySelectorAll("video[data-pending-path]").forEach((video) => {
      const path = video.dataset.pendingPath;
      const fallback = () => {
        const card = video.closest(".trial-video-card");
        if (!card || card.dataset.pending === "true") return;
        card.dataset.pending = "true";
        const caption = card.querySelector(".trial-video-caption");
        card.innerHTML = `
          <div class="trial-video-pending">
            <span>pending</span>
          </div>
          ${caption ? caption.outerHTML : ""}
        `;
        card.dataset.path = path;
      };

      video.addEventListener("error", fallback, { once: true });
      const source = video.querySelector("source");
      if (source) source.addEventListener("error", fallback, { once: true });
    });
  };

  const render = () => {
    const groups = groupEvidence();
    const pageCount = Math.max(1, groups.length);
    pageIndex = Math.min(pageIndex, pageCount - 1);
    const group = groups[pageIndex];

    count.textContent = groups.length ? `1 trial shown · ${groups.length} matched` : "0 trials";
    if (pageStatus) pageStatus.textContent = groups.length ? `page ${pageIndex + 1} of ${pageCount}` : "page 0 of 0";
    if (prevButton) prevButton.disabled = pageIndex === 0;
    if (nextButton) nextButton.disabled = pageIndex >= pageCount - 1 || groups.length === 0;

    if (!group) {
      gallery.innerHTML = '<div class="evidence-empty">no trial evidence matches the selected filters.</div>';
      return;
    }

    const itemsByVariant = Object.fromEntries(group.items.map((item) => [item.variant, item]));
    const title = `${String(group.task).toLowerCase()} · trial ${String(group.trial).padStart(2, "0")}`;

    gallery.innerHTML = `
      <section class="trial-comparison" style="--card-delay: 0ms">
        <div class="trial-comparison-header">
          <div>
            <span>${escapeHtml(title)}</span>
            <h4>${escapeHtml(String(group.setup).toLowerCase())}</h4>
          </div>
        </div>
        <figure class="trial-setup-card">
          <span class="trial-setup-frame ${escapeHtml(taskSlug(group.task))}">
            <img src="${escapeHtml(setupImage(group.task))}" alt="${escapeHtml(title)} setup image.">
          </span>
        </figure>
        <div class="trial-video-grid" aria-label="${escapeHtml(title)} video comparison">
          <span class="trial-video-head"></span>
          <span class="trial-video-head">pytorch</span>
          <span class="trial-video-head">vla.cpp</span>
          ${cameraViews
            .map(
              (view) => `
                <span class="trial-camera-label">${escapeHtml(view.label)}</span>
                ${videoCell(itemsByVariant["Baseline-Pytorch-BF16"], view)}
                ${videoCell(itemsByVariant["vla.cpp-BF16"], view)}
              `,
            )
            .join("")}
        </div>
      </section>
    `;

    bindInlineVideoFallbacks();
  };

  const showPending = (item) => {
    modalMedia.innerHTML = `
      <div class="modal-pending">
        <div>
          <strong>evidence pending</strong>
          <p>upload the video to <span class="mono">${escapeHtml(item.video)}</span> and reopen this card.</p>
        </div>
      </div>
    `;
  };

  const openModal = (item) => {
    modalTitle.textContent = trialTitle(item);
    modalMeta.textContent = "";

    modalMedia.innerHTML = `
      <video controls preload="metadata" poster="${escapeHtml(item.poster)}">
        <source src="${escapeHtml(item.video)}" type="video/mp4">
        your browser does not support the video tag.
      </video>
    `;

    const video = modalMedia.querySelector("video");
    const source = modalMedia.querySelector("source");
    const fallback = () => showPending(item);
    video.addEventListener("error", fallback, { once: true });
    source.addEventListener("error", fallback, { once: true });

    modal.hidden = false;
    document.body.classList.add("modal-open");
  };

  const closeModal = () => {
    modal.hidden = true;
    modalMedia.innerHTML = "";
    document.body.classList.remove("modal-open");
  };

  Object.values(filters).forEach((control) => {
    if (control) {
      control.addEventListener("change", () => {
        pageIndex = 0;
        render();
      });
    }
  });

  if (prevButton) {
    prevButton.addEventListener("click", () => {
      pageIndex = Math.max(0, pageIndex - 1);
      render();
    });
  }

  if (nextButton) {
    nextButton.addEventListener("click", () => {
      pageIndex += 1;
      render();
    });
  }

  modal.querySelectorAll("[data-close-modal]").forEach((control) => {
    control.addEventListener("click", closeModal);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.hidden) closeModal();
  });

  render();
})();
