(function () {
  const VARIANTS = {
    vlacpp: {
      label: "vla.cpp-BF16",
      dir: "vlacpp",
    },
  };

  const CAMERA = "iPhone Overview";

  const trial = (id, setupIndex, result, time = "", note = "") => ({
    id,
    setupIndex,
    setup: `Setup ${setupIndex}`,
    result,
    time,
    note,
  });

  const DATA = [
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
      return {
        variant: group.variant.label,
        variantDir: group.variant.dir,
        task: group.task,
        taskSlug: group.slug,
        trial: row.id,
        setupIndex: row.setupIndex,
        setupTrialIndex: ((row.id - 1) % 4) + 1,
        setup: row.setup,
        result: row.result,
        time: row.time,
        note: row.note,
        camera: CAMERA,
        overviewVideo: "",
        overviewPoster: "",
        video: "",
        poster: "",
      };
    }),
  );
})();
