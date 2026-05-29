(function () {
  const VARIANTS = {
    pytorch: {
      label: "Baseline-Pytorch-BF16",
      dir: "pytorch",
    },
    vlacpp: {
      label: "vla.cpp-BF16",
      dir: "vlacpp",
    },
  };

  const CAMERA = "iPhone Overview";

  const trial = (id, setup, result, time = "", note = "") => ({
    id,
    setup: `Setup ${setup}`,
    result,
    time,
    note,
  });

  const DATA = [
    {
      variant: VARIANTS.pytorch,
      task: "Task 1",
      slug: "task1",
      rows: [
        trial(1, 1, "PASS", "1m15s"),
        trial(2, 1, "PASS", "1m22s"),
        trial(3, 1, "FAIL", "", "Task failed completely"),
        trial(4, 1, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(5, 2, "PASS", "0m57s"),
        trial(6, 2, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(7, 2, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(8, 2, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(9, 3, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(10, 3, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(11, 3, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(12, 3, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(13, 4, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(14, 4, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(15, 4, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(16, 4, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(17, 5, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(18, 5, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(19, 5, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(20, 5, "FAIL", "", "Task failed partly (Timeout/Retry)"),
      ],
    },
    {
      variant: VARIANTS.pytorch,
      task: "Task 2",
      slug: "task2",
      rows: [
        trial(1, 1, "PASS", "0m24s"),
        trial(2, 1, "PASS", "0m25s"),
        trial(3, 1, "PASS", "0m23s"),
        trial(4, 1, "PASS", "0m24s"),
        trial(5, 2, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(6, 2, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(7, 2, "FAIL", "", "Task failed completely"),
        trial(8, 2, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(9, 3, "PASS", "0m31s"),
        trial(10, 3, "PASS", "0m29s"),
        trial(11, 3, "PASS", "0m29s"),
        trial(12, 3, "PASS", "0m34s"),
        trial(13, 4, "PASS", "0m46s"),
        trial(14, 4, "PASS", "0m29s"),
        trial(15, 4, "PASS", "0m28s"),
        trial(16, 4, "PASS", "0m28s"),
        trial(17, 5, "PASS", "0m35s"),
        trial(18, 5, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(19, 5, "FAIL", "", "Task failed completely"),
        trial(20, 5, "FAIL", "", "Task failed partly (Timeout/Retry)"),
      ],
    },
    {
      variant: VARIANTS.vlacpp,
      task: "Task 1",
      slug: "task1",
      rows: [
        trial(1, 1, "PASS", "1m29s"),
        trial(2, 1, "PASS", "0m47s"),
        trial(3, 1, "PASS", "0m46s"),
        trial(4, 1, "PASS", "0m44s"),
        trial(5, 2, "PASS", "0m44s"),
        trial(6, 2, "PASS", "0m55s"),
        trial(7, 2, "PASS", "0m43s"),
        trial(8, 2, "PASS", "0m52s"),
        trial(9, 3, "PASS", "0m52s"),
        trial(10, 3, "PASS", "0m42s"),
        trial(11, 3, "PASS", "0m43s"),
        trial(12, 3, "PASS", "0m55s"),
        trial(13, 4, "PASS", "0m43s"),
        trial(14, 4, "PASS", "0m44s"),
        trial(15, 4, "PASS", "0m51s"),
        trial(16, 4, "PASS", "0m43s"),
        trial(17, 5, "PASS", "1m00s"),
        trial(18, 5, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(19, 5, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(20, 5, "PASS", "0m35s"),
      ],
    },
    {
      variant: VARIANTS.vlacpp,
      task: "Task 2",
      slug: "task2",
      rows: [
        trial(1, 1, "PASS", "0m24s"),
        trial(2, 1, "PASS", "0m31s"),
        trial(3, 1, "PASS", "0m24s"),
        trial(4, 1, "PASS", "0m24s"),
        trial(5, 2, "PASS", "0m27s"),
        trial(6, 2, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(7, 2, "PASS", "0m29s"),
        trial(8, 2, "PASS", "0m30s"),
        trial(9, 3, "PASS", "0m30s"),
        trial(10, 3, "PASS", "0m27s"),
        trial(11, 3, "PASS", "0m28s"),
        trial(12, 3, "PASS", "0m25s"),
        trial(13, 4, "PASS", "0m25s"),
        trial(14, 4, "FAIL", "", "Task failed partly (Timeout/Retry)"),
        trial(15, 4, "PASS", "0m30s"),
        trial(16, 4, "PASS", "0m26s"),
        trial(17, 5, "PASS", "0m28s"),
        trial(18, 5, "PASS", "0m42s"),
        trial(19, 5, "PASS", "0m33s"),
        trial(20, 5, "FAIL", "", "Task failed completely"),
      ],
    },
  ];

  window.ALOHA_EVIDENCE = DATA.flatMap((group) =>
    group.rows.map((row) => {
      const trialId = String(row.id).padStart(2, "0");
      const base = `assets/aloha-videos/${group.variant.dir}/${group.slug}/trial-${trialId}-iphone`;
      return {
        variant: group.variant.label,
        task: group.task,
        trial: row.id,
        setup: row.setup,
        result: row.result,
        time: row.time,
        note: row.note,
        camera: CAMERA,
        video: `${base}.mp4`,
        poster: `${base}.jpg`,
      };
    }),
  );
})();
