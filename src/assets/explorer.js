/* Model x backend explorer. Every latency and memory figure is copied from the per-device reports
   of the runtime's latest benchmark round (vla-bench, September-November 2026), every task-success
   rate from its LIBERO report, and every fixed-input check from its backend notes. These postdate
   the paper and use a different protocol, so they do not match its tables cell for cell.
   Nothing is interpolated: a pair without a number carries the report's reason instead. */
(function () {
  'use strict';

  /* Configuration from the reports' model table. ok is LIBERO-Object successes out of 200 and
     replay the actions executed per query, both from the LIBERO report (RTX 3090, defaults); null
     for ACT, whose checkpoint is an ALOHA policy with no LIBERO run. */
  var MODELS = [
    { id:'octo', name:'Octo-Small', head:'DDPM diffusion', steps:20, views:2, input:'256 + 128',
      chunk:4, replay:4, ok:16,
      note:'A diffusion head run for 20 steps over a 256 px primary view and a 128 px wrist view, ' +
           'with F32 weights as shipped. The fastest model on every device that runs it. OpenVINO ' +
           'cannot translate one of its ops yet, so it has no OpenVINO rows. Its 8% on ' +
           'LIBERO-Object is well below published results for this checkpoint and has not been ' +
           'investigated.' },

    { id:'act', name:'ACT', head:'transformer decoder', steps:1, views:1, input:'640 × 480',
      chunk:100, replay:null, ok:null,
      note:'Action Chunking with Transformers: a ResNet-18 and a transformer encoder–decoder with ' +
           'no language input, run on its ALOHA sim transfer-cube checkpoint over one 640×480 view, ' +
           'predicting a 100-step chunk of a 14-dim bimanual action. It has no LIBERO checkpoint, so ' +
           'no success rate here. Only the Dragonwing IQ-10 report measures it so far.' },

    { id:'turbo', name:'TurboVLA', head:'single-pass decoder', steps:1, views:2, input:'256',
      chunk:12, replay:12, ok:200,
      note:'A single-pass decoder over two 256 px views, with F32 weights as shipped. Lower-precision ' +
           'weights help it a lot: F16 weights with flash attention cut its mean by 37–38% on the ' +
           'three measured RTX cards and by 76% on the Snapdragon X Hexagon NPU, where the F32 weights ' +
           'otherwise stay F32; SYCL and the IQ-10 build make them the defaults. It clears all 200 ' +
           'LIBERO-Object episodes.' },

    { id:'jepa', name:'VLA-JEPA', head:'flow matching (DiT)', steps:4, views:1, input:'256',
      chunk:7, replay:7, ok:200,
      note:'A Qwen3-VL tower and Qwen3 LM ahead of a DiT flow-matching head; its prompt carries 32 ' +
           'extra <code>&lt;embodied&gt;</code> tokens. It clears all 200 LIBERO-Object episodes.' },

    { id:'adapter', name:'VLA-Adapter', head:'bridge-attention policy', steps:1, views:1, input:'224',
      chunk:8, replay:8, ok:191,
      note:'DINOv2 and SigLIP towers and a Qwen2.5 LM, with a bridge-attention policy that predicts ' +
           'the 8-action chunk in one pass rather than refining it over solver steps.' },

    { id:'n15', name:'GR00T N1.5', head:'flow matching (DiT)', steps:4, views:1, input:'224',
      chunk:16, replay:16, ok:194,
      note:'A SigLIP tower, a Qwen3 LM and a VL self-attention block ahead of a DiT flow-matching ' +
           'head. Its action chunk is 16 steps of a 32-wide padded action.' },

    { id:'n16', name:'GR00T N1.6', head:'flow matching (DiT)', steps:4, views:1, input:'224',
      chunk:50, replay:16, ok:173,
      note:'The same SigLIP, Qwen3 and DiT layout as N1.5, with a 50-step chunk 128 wide. At ' +
           '173 of 200 it is the lowest of the three GR00T checkpoints on LIBERO-Object. It is also ' +
           'the ALOHA policy in the robot section below.' },

    { id:'n17', name:'GR00T N1.7', head:'flow matching (DiT)', steps:4, views:1, input:'256',
      chunk:40, replay:16, ok:196, suites:[95.5, 98.0, 97.5, 90.0, 95.3],
      note:'A Qwen3-VL tower and Qwen3 LM ahead of a DiT head, over a 40-step chunk 132 wide. ' +
           'Besides LIBERO-Object it is measured on all four LIBERO suites.' },

    { id:'bitvla', name:'BitVLA', head:'parallel decoding', steps:1, views:1, input:'224',
      chunk:8, replay:8, ok:200, suites:[95.5, 100.0, 91.5, 89.5, 94.1],
      note:'Ternary weights. The only published GGUF packs them as int2, and only CUDA builds load ' +
           'it, so BitVLA runs on the four NVIDIA cards and the three Jetsons and nowhere else. On ' +
           'CUDA it has its own pipeline: ternary GEMMs on tensor cores, whose int32 sums are exact ' +
           'in any order.' },

    { id:'smolvla', name:'SmolVLA', head:'flow matching', steps:10, views:2, input:'512',
      chunk:50, replay:10, ok:179,
      note:'A SigLIP tower over two 512 px views and a flow-matching expert. In LIBERO it replays 10 ' +
           'of each 50-action chunk, the best setting in its paper’s ablation, where upstream ' +
           'replans every step. <code>--flash-attn --mm-prec default</code> is its fastest flag set ' +
           'on seven of the GPUs, by 14–39%, but it moves the action chunk.' },

    { id:'pi0', name:'π0', head:'flow matching', steps:10, views:2, input:'224',
      chunk:50, replay:50, ok:144,
      note:'A SigLIP tower and a Gemma prefix with an action expert. It replays the whole 50-action ' +
           'chunk in LIBERO, as its LeRobot config does, where openpi’s eval replans every 5. At ' +
           '144 of 200 it is well below published results for this checkpoint, which has not been ' +
           'investigated yet.' },

    { id:'pi05', name:'π0.5', head:'flow matching', steps:10, views:2, input:'224',
      chunk:50, replay:10, ok:195,
      note:'The π0 layout, with a prompt whose length follows the robot state.' },

    { id:'oft', name:'OpenVLA-OFT', head:'L1 regression head', steps:1, views:1, input:'224',
      chunk:8, replay:8, ok:197,
      note:'DINOv2 and SigLIP towers and a Llama-2 LM: the heaviest model, at about 15 GB of BF16 ' +
           'weights. It runs out of memory on the 12 GB RTX 3060, the 8 GB RTX 5070 Laptop and the ' +
           '6 GB Arc A380, and was not attempted on the 8 and 16 GB Jetsons or on hosts with 16 GB of ' +
           'RAM or less. The 16 GB RX 9060 XT holds it.' },

    { id:'evo1', name:'Evo-1', head:'flow matching', steps:32, views:1, input:'448',
      chunk:50, replay:14, ok:197,
      note:'An InternViT tower and a Qwen2 LM ahead of a 32-step flow-matching head, on one 448 px ' +
           'view. The slowest model on nine of the twenty-two rows: eight GPUs and the IQ-10’s NPU.' }
  ];

  var SUITES = ['Spatial', 'Object', 'Goal', 'Long', 'Average'];

  /* mem names the report's memory columns, in order; chk says what a fixed-input check compares. */
  var DEVICES = [
    { id:'rtx5090', group:'Discrete GPUs', name:'RTX 5090', short:'RTX 5090', be:'CUDA', tag:'cuda',
      sub:'CUDA · 32 GB GDDR7', date:'2026-09-28', mem:['VRAM', 'Host RSS'], minOnly:true,
      note:'Contributed numbers, not re-measured: the minimum over five rounds, with no mean, ' +
           'percentiles, vision time or memory, and no flag screen. Its cells show that minimum.' },

    { id:'rtx3090', group:'Discrete GPUs', name:'RTX 3090', short:'RTX 3090', be:'CUDA', tag:'cuda',
      sub:'CUDA · 24 GB GDDR6X', date:'2026-09-30', mem:['VRAM', 'Host RSS'],
      note:'One of two cards in an i7-14700F host, the other idle. The LIBERO success rates come ' +
           'from this card. VRAM is the process’s device memory from nvidia-smi, sampled every ' +
           '0.5 s; host RSS is its peak resident set.' },

    { id:'rtx3060', group:'Discrete GPUs', name:'RTX 3060', short:'RTX 3060', be:'CUDA', tag:'cuda',
      sub:'CUDA · 12 GB GDDR6', date:'2026-09-30', mem:['VRAM', 'Host RSS'],
      note:'In an i5-12400F host. VRAM is the process’s device memory from nvidia-smi, sampled ' +
           'every 0.5 s; host RSS is its peak resident set.' },

    { id:'rtx5070l', group:'Discrete GPUs', name:'RTX 5070 Laptop', short:'RTX 5070 Laptop', be:'CUDA',
      tag:'cuda', sub:'CUDA · 8 GB GDDR7', date:'2026-09-30', mem:['VRAM', 'Host RSS'],
      note:'In an i9-14900HX laptop on AC power, performance profile. VRAM is the process’s device ' +
           'memory from nvidia-smi, sampled every 0.5 s; host RSS is its peak resident set.' },

    { id:'rx9060xt', group:'Discrete GPUs', name:'Radeon RX 9060 XT', short:'RX 9060 XT', be:'ROCm',
      tag:'rocm', sub:'ROCm · 16 GB GDDR6', date:'2026-10-28', mem:['VRAM', 'Host RSS'],
      chk:'HIP against the CPU build, both at F32 weights, which leaves only summation order',
      note:'RDNA 4 at its default 160 W limit. HIP runs the plain ggml graph: the fused CUDA ' +
           'nodes, BF16 activations and BitVLA’s int2 kernels are CUDA-only. Every model was checked ' +
           'against the CPU build on fixed inputs. VRAM is from the amdgpu KFD counters.' },

    { id:'a380', group:'Discrete GPUs', name:'Arc A380', short:'Arc A380', be:'SYCL', tag:'sycl',
      sub:'SYCL · 6 GB GDDR6', date:'2026-09-30', mem:['VRAM', 'Host RSS'],
      note:'In a Ryzen 5 5500 host, through Level Zero. VRAM is the device-local memory from the ' +
           'i915 fdinfo counters.' },

    { id:'agx', group:'Jetson boards', name:'Jetson AGX Orin', short:'AGX Orin', be:'CUDA', tag:'cuda',
      sub:'CUDA · 64 GB unified', date:'2026-10-22', mem:['CUDA'],
      note:'MAXN, on the bit-exact CUDA path: the defaults return the same bits as the plain ggml ' +
           'graph, at 23–40% lower mean latency than the commit before it for every model except ' +
           'Octo-Small. Flags that change the numerics leave that path, so most are slower here. ' +
           'CUDA is the device memory the process holds, counted by a preload shim; host memory is ' +
           'not included.' },

    { id:'nx', group:'Jetson boards', name:'Jetson Orin NX 16 GB', short:'Orin NX', be:'CUDA',
      tag:'cuda', sub:'CUDA · 16 GB unified', date:'2026-10-01', mem:[],
      note:'MAXN_SUPER. Latency only: on this board the resident set did not reliably capture the ' +
           'CUDA buffers. GPU load stayed at 94–98% with no throttling, so these are compute-bound.' },

    { id:'nano', group:'Jetson boards', name:'Jetson Orin Nano Super', short:'Orin Nano', be:'CUDA',
      tag:'cuda', sub:'CUDA · 8 GB unified', date:'2026-09-30', mem:['RSS'],
      note:'MAXN_SUPER. Memory is unified and the CUDA buffers are mapped into the process, so peak ' +
           'RSS is its whole footprint.' },

    { id:'m4', group:'Integrated GPUs', name:'Apple M4', short:'Apple M4', be:'Metal', tag:'metal',
      sub:'Metal · 24 GB unified', date:'2026-09-30', mem:['RSS'],
      note:'10-core GPU. Memory is unified, so peak RSS includes the Metal buffers.' },

    { id:'b390sycl', group:'Integrated GPUs', name:'Arc B390 iGPU', short:'Arc B390', be:'SYCL',
      tag:'sycl', sub:'SYCL · Core Ultra X7 358H', date:'2026-11-04', mem:['Device', 'RSS'],
      note:'The November round, through the OpenCL adapter: the Level Zero adapter crashes with that ' +
           'machine’s loader. 25 W long-term power limit. The SYCL defaults run fused graphs for ' +
           'SmolVLA, VLA-Adapter, GR00T N1.5 and GR00T N1.6, and flash attention for SmolVLA, Evo-1, ' +
           'π0.5 and TurboVLA; Evo-1 and π0.5 fuse only with BF16 activations, their fastest flags. ' +
           'Device is the GPU buffer memory from the xe fdinfo counters.' },

    { id:'b390ov', group:'Integrated GPUs', name:'Arc B390 iGPU', short:'Arc B390', be:'OpenVINO',
      tag:'ov', sub:'OpenVINO GPU · Core Ultra X7 358H', date:'2026-11-04', mem:['Device', 'RSS'],
      chk:'the GPU plugin against the ggml CPU build at F32 weights, on the continuous action dims of six fixed inputs',
      note:'OpenVINO’s GPU plugin on the same chip. Each graph compiles on the first prediction, ' +
           'which the warmups absorb. Device is the GPU buffer memory from the xe fdinfo counters.' },

    { id:'hexnpu', group:'NPUs', name:'Snapdragon X', short:'Snapdragon X', be:'Hexagon', tag:'npu',
      sub:'Hexagon NPU · CPU fallback', date:'2026-09-30', mem:['Working set', 'Private'],
      chk:'the NPU against a CPU F32 reference, at a slightly earlier llama.cpp build',
      note:'A Snapdragon X X1-26-100 laptop on Windows 11, Balanced power plan. Ops the NPU rejects ' +
           'run on the Oryon CPU. F16 is the default weight type in this build. The weights live in ' +
           'shared memory, which counts toward the working set but not the private commit.' },

    { id:'iq10', group:'NPUs', name:'Dragonwing IQ-10', short:'Dragonwing IQ-10', be:'Hexagon', tag:'npu',
      sub:'Hexagon v81 NPU · CPU fallback · Linux', date:'2026-11-05', mem:['RSS', 'System RAM'],
      chk:'the NPU against the same build on the CPU at F32 weights without flash attention, on the views the LIBERO client sends',
      note:'A Qualcomm Dragonwing IQ-10 (IQX EVK, QCS8797) under Ubuntu 24.04, one of its four Hexagon ' +
           'sessions in use, with 16 Oryon threads for the ops the NPU rejects. F16 weights are the ' +
           'default; the NPU has no BF16 kernels. RSS leaves out the weights, which live in FastRPC ' +
           'buffers mapped for the NPU; system RAM is the drop in MemAvailable while the process ran, ' +
           'which counts them.' },

    { id:'ovnpu', group:'NPUs', name:'AI Boost NPU', short:'AI Boost', be:'OpenVINO', tag:'npu',
      sub:'OpenVINO NPU · Core Ultra X7 358H', date:'2026-11-04', mem:['Device', 'RSS'],
      chk:'the NPU against the ggml CPU build at F32 weights, on the continuous action dims of six fixed inputs (48 for π0)',
      note:'Runs eleven of the thirteen graphs since the November round; the compiler failures on ' +
           'dynamic shapes are gone. It computes in F16 only, so a latency is not a correctness ' +
           'check here. Device is the NPU buffer memory from the intel_vpu fdinfo counters.' },

    { id:'i7', group:'CPUs', name:'Core i7-14700F', short:'i7-14700F', be:'CPU', tag:'cpu',
      sub:'CPU · 16 threads', date:'2026-09-30', mem:['RSS'],
      note:'65 W long-term power limit with no short-term cap, so the first calls of a process can ' +
           'run at boost power.' },

    { id:'i9', group:'CPUs', name:'Core i9-14900HX', short:'i9-14900HX', be:'CPU', tag:'cpu',
      sub:'CPU · 16 threads', date:'2026-09-30', mem:['RSS'],
      note:'Laptop on AC power, performance profile, 250 W power limits.' },

    { id:'x7cpu', group:'CPUs', name:'Core Ultra X7 358H', short:'Ultra X7 358H', be:'CPU', tag:'cpu',
      sub:'ggml CPU · 16 threads', date:'2026-11-04', mem:['RSS'],
      note:'ggml’s own CPU backend under a 25 W long-term power limit. The first calls of a process ' +
           'can run at boost power, so read p50 for sustained latency.' },

    { id:'x7ovcpu', group:'CPUs', name:'Core Ultra X7 358H', short:'Ultra X7 358H', be:'OpenVINO',
      tag:'ov', sub:'OpenVINO CPU · 16 threads', date:'2026-11-04', mem:['RSS'],
      chk:'the CPU plugin against the ggml CPU build at F32 weights, on the continuous action dims of six fixed inputs',
      note:'OpenVINO’s CPU plugin on the same chip. At p50 it is 9–36% slower than ggml’s CPU ' +
           'backend on every model. It is where a translation is checked, since it computes in F32.' },

    { id:'i5', group:'CPUs', name:'Core i5-12400F', short:'i5-12400F', be:'CPU', tag:'cpu',
      sub:'CPU · 12 threads', date:'2026-09-30', mem:['RSS'],
      note:'Six cores and 15.4 GiB of RAM, with power limits effectively unlimited.' },

    { id:'r5', group:'CPUs', name:'Ryzen 5 5500', short:'Ryzen 5 5500', be:'CPU', tag:'cpu',
      sub:'CPU · 12 threads', date:'2026-09-30', mem:['RSS'],
      note:'Zen 3 at stock power, 15.5 GiB of RAM.' },

    { id:'oryon', group:'CPUs', name:'Snapdragon X', short:'Snapdragon X', be:'CPU', tag:'cpu',
      sub:'Oryon CPU · 8 threads · Windows', date:'2026-09-30', mem:['Working set', 'Private'],
      note:'The Hexagon build with the NPU switched off. It defaults to F16 weights; Oryon has no ' +
           'BF16 matmul, and BF16 weights run about 3× slower. Balanced power plan with clocks ' +
           'uncontrolled, so these rows vary more than on the desktop hosts.' }
  ];

  /* ---------------- measurements ---------------- */

  /* Defaults, per device and model: [min, mean, p50, p90, vision] in ms, then the device's memory
     columns in MiB, from each report's latency table. null where the report gives no number.
     T is the fastest flag set where it beats the defaults: [flags, min, mean, p50, p90, vision,
     change in mean, %]. Both are generated from the reports; do not edit by hand. */
  var R = {
    rtx5090: { octo:[2.69,null,null,null,null,null,null], turbo:[4.88,null,null,null,null,null,null], jepa:[14.3,null,null,null,null,null,null], adapter:[16,null,null,null,null,null,null], n15:[16.1,null,null,null,null,null,null], n16:[19.4,null,null,null,null,null,null], n17:[19.5,null,null,null,null,null,null], bitvla:[20.8,null,null,null,null,null,null], pi0:[28.8,null,null,null,null,null,null], pi05:[29.2,null,null,null,null,null,null], oft:[34.5,null,null,null,null,null,null], smolvla:[38.2,null,null,null,null,null,null], evo1:[49.1,null,null,null,null,null,null] },
    rtx3090: { octo:[4.3,4.4,4.3,4.4,0.5,810,501], turbo:[9,9.3,9.4,9.4,null,1152,496], bitvla:[26.9,27.1,27.2,27.3,7.3,1460,1180], jepa:[27.3,27.8,27.6,28.1,8.9,4138,634], adapter:[34.2,34.5,34.5,34.7,17,3032,1063], n17:[35.3,35.7,35.4,36.1,9,5204,650], n16:[36.3,36.9,36.6,37.5,9.5,4854,643], n15:[36.6,36.9,36.7,37.2,9.5,3804,692], smolvla:[61,61.3,61.3,61.6,28.9,2216,734], pi0:[97.5,98,98,98.5,18.7,5670,712], pi05:[97.8,99.6,99.7,99.9,18.5,5880,735], evo1:[108.6,109,108.9,109.5,39.8,1746,707], oft:[134.2,135.2,135.2,135.8,17.6,14768,1036] },
    rtx3060: { octo:[8.3,8.3,8.3,8.4,1,626,501], turbo:[21.6,22.2,22.1,22.7,null,982,496], jepa:[57.1,58.5,58.1,60.7,21.8,3968,631], bitvla:[59.1,59.8,59.4,61.1,11.9,1308,1180], adapter:[74.7,75.4,74.8,76.9,41.8,2862,1064], n16:[78.1,78.9,78.2,81.6,22.7,4682,636], n17:[82.8,83.7,83,85.8,21.7,5032,647], n15:[86.8,87.5,86.9,89.4,22.9,3632,688], smolvla:[125.1,126,125.3,128,71.6,2038,721], pi0:[228.2,228.8,228.6,229.2,44.9,5516,710], pi05:[229.9,230.2,230.1,230.3,45,5728,733], evo1:[230.8,231.2,231.1,231.4,108,1594,705] },
    rtx5070l: { octo:[4.7,5,5,5,0.6,734,589], turbo:[14.1,14.5,14.5,14.8,null,1030,583], jepa:[38,40.8,38.6,50.5,14.2,4006,728], bitvla:[54.4,60,55.8,71.8,11.9,1328,1178], adapter:[56.3,63,59.4,75.9,34.1,2896,1071], n17:[63.3,68,64.3,81.6,16.1,5064,750], n16:[65.5,72.3,68.6,87.8,21.5,4708,740], n15:[73,79.2,74.4,95,21.2,3668,765], smolvla:[89.3,98.7,96.5,111.3,54.2,2056,805], pi0:[177.1,195.7,190.7,212.7,41.3,5534,788], pi05:[180.5,196.7,192.1,209.8,40.9,5528,809], evo1:[184.6,201.6,201.7,217.7,92.4,1604,805] },
    rx9060xt: { octo:[14,14.1,14,14.2,1.9,817,648], turbo:[33.8,34,34,34.2,null,1112,662], jepa:[65.9,66.5,66.4,67,22.2,4167,941], adapter:[70.1,70.9,70.9,71.5,34.9,3144,1208], n15:[79.5,80.1,80.1,80.6,21.1,3925,1203], n16:[94.6,95.5,95.5,96.3,20.6,4871,945], n17:[97.1,98.2,98.2,99.1,22.2,5225,954], oft:[182.2,183,183.2,183.4,36.1,14899,1191], smolvla:[207.7,208.5,208.7,209.1,51.9,2303,1211], pi0:[217.9,219.5,219,221.5,40.7,5791,1207], pi05:[220.6,222.5,221.8,225.2,40.5,5793,1221], evo1:[233.6,234,234,234.6,79.5,1871,1217] },
    agx: { octo:[14.3,20.2,17.9,28.1,2.9,554], turbo:[28,29.9,28.4,35,null,863], jepa:[60.6,61.4,61.3,61.7,19.4,3890], bitvla:[76.8,80,77.3,87.3,15.9,1037], adapter:[79.1,79.7,79.7,80,39.5,2792], n17:[93,94,94.1,94.2,null,4953], n15:[96.7,98,97.9,98.7,21.2,3578], n16:[101.2,102.1,102.2,102.3,21.4,4570], smolvla:[164.3,165.2,165,166.5,72.6,1896], pi05:[266.6,267.1,267.2,267.4,42.2,5630], pi0:[269,269.4,269.5,269.8,41.6,5521], oft:[297.4,297.7,297.7,298,40.9,14648], evo1:[298.5,299,298.8,299.7,124,1566] },
    nx: { octo:[27.4,27.5,27.5,27.5,2.9], turbo:[79.7,79.8,79.8,79.8,null], jepa:[176.7,176.8,176.7,176.8,73.6], n17:[251.4,251.8,251.6,251.9,73.5], n16:[252.1,252.3,252.2,253.1,76.5], n15:[260.8,261.1,261,261.7,76.5], bitvla:[269.1,269.2,269.2,269.2,50.6], adapter:[270.7,270.8,270.8,270.9,151.8], smolvla:[411.2,411.5,411.4,411.8,261.7], pi0:[739.6,740,740,740.2,152.7], pi05:[742.9,743.2,743.1,743.4,152.2], evo1:[893.2,894,894.1,894.2,454] },
    nano: { octo:[31.9,34.4,33.1,36.3,4.1,923], turbo:[89.3,90.2,89.6,89.9,null,1215], jepa:[193.9,194.6,194.3,196,82.4,4332], n17:[271.7,272.1,272,272.2,82.3,5413], n16:[272.3,272.6,272.5,272.9,85,5051], n15:[293.4,295,295.2,296,84.8,4048], adapter:[297.5,297.9,297.9,298.1,167.4,3669], bitvla:[335,335.5,335.3,336,63.6,2193], smolvla:[462.8,464.2,463.9,466.1,298.6,2488], pi0:[845.2,846.1,846.1,846.7,169.2,5961], pi05:[849.7,850.2,850.2,850.5,169.3,5982], evo1:[1011.5,1012.5,1012.6,1013,523.3,2048] },
    m4: { octo:[22.6,22.9,22.9,23,5.7,838], turbo:[65.2,65.4,65.4,65.5,null,1127], jepa:[248.7,249,248.9,249.2,86,4043], adapter:[315.7,316.3,316.3,316.5,174.1,3712], n17:[350.8,355,355.2,355.6,85.9,5125], smolvla:[357.6,358.1,358.2,358.4,215.8,1208], n16:[358.9,361.6,361.7,362,100.3,4762], n15:[401.5,401.9,401.9,402.2,99.4,3753], evo1:[829.3,830.1,830.1,830.5,382.1,1637], pi0:[1141.1,1149.7,1146.3,1159.6,199.5,5549], pi05:[1153.8,1165.1,1161.1,1180.4,199.3,5963], oft:[1650.7,1716.1,1740,1751.1,184.2,15479] },
    b390sycl: { octo:[15.5,15.8,15.6,16.2,1.6,615,444], turbo:[25,25.6,25.7,25.8,null,560,445], smolvla:[61.7,62.3,62.3,62.8,16.4,1107,431], adapter:[62.6,63.1,63.1,63.5,22.8,2751,1065], n15:[101.2,101.8,101.9,102.1,12.9,3535,423], jepa:[102.1,102.6,102.5,103,39.4,3879,422], n16:[106.1,106.5,106.5,106.8,13,4566,410], n17:[178.4,179.4,179.3,180.3,39.6,4944,406], pi0:[269.3,270,269.9,270.7,57.4,5433,476], evo1:[270.4,271.8,271.8,272.2,76.6,1456,516], pi05:[275.2,276.4,276.4,277.2,56.7,5665,547], oft:[434.9,438.5,438.3,441.3,79.9,14534,1035] },
    b390ov: { turbo:[28.5,29.7,29.6,30.4,null,1080,1540], n15:[106,107.1,107.2,107.4,38.1,10575,6630], jepa:[116.3,117,117.1,117.8,19.9,9387,5010], n16:[122.8,131.2,132.6,134.5,15.6,14698,9922], n17:[131.4,132,131.8,132.8,20.1,15035,10999], adapter:[170.2,185.4,172.4,207.7,97.8,7431,3621], smolvla:[213.1,214.9,215,216,111.7,4211,3616], pi05:[232.5,233.7,233.5,234.5,36.5,15964,11464], evo1:[566,568.2,568.2,569.5,227.9,4982,3279], pi0:[620.5,625.1,624.6,630.1,78.7,25056,16936], oft:[1262.5,1334.9,1326.8,1419.9,139.5,41761,15264] },
    a380: { octo:[60.5,61,61,61.3,4.8,585,497], turbo:[103.5,103.6,103.6,103.8,null,885,561], jepa:[355.9,356.8,356.8,357.3,109.3,3855,468], adapter:[408.9,410,410.2,410.4,193.7,2750,1102], n15:[472,472.7,472.7,473.3,114.8,3531,496], n17:[615.3,618.4,618.2,620.9,109.9,4913,517], n16:[646,647.6,647.9,648.4,114.8,4558,493], smolvla:[734.5,736.5,736.5,738,231.6,1159,508], pi0:[921.9,924.2,924.1,925.4,228.3,5366,513], pi05:[945.6,948.1,948.2,950,228.3,5659,513], evo1:[1047.3,1048.8,1048.8,1049.8,354.6,1467,544] },
    hexnpu: { octo:[239.7,266.3,265.4,288.4,51.2,679,218], jepa:[821,846.2,846,859,369.2,3900,1105], smolvla:[1239.3,1246.9,1247.7,1249.5,430.6,1129,334], n17:[1400.2,1421.3,1417.2,1438.2,367.9,4964,625], turbo:[2280.8,2292.7,2289.2,2305.5,null,948,208], n15:[2755.7,2773.9,2773,2784.5,1802.1,3560,151], n16:[2940,2957.1,2956.9,2970.7,1861.1,4601,164], adapter:[4072.1,4099.2,4101.9,4114.7,2888.8,2796,861], pi0:[5730,5739.3,5740.4,5743.9,3614.4,5384,230], evo1:[7208.6,7292.2,7302.9,7344.8,1054.1,1499,276], pi05:[9321.7,9363.2,9367.4,9395.1,4531.4,5701,230] },
    iq10: { octo:[117.6,119.4,118.4,120.7,21.2,253,723], act:[148.5,152.9,152.5,157.3,null,194,467], turbo:[188.1,190.8,188.6,194.9,null,249,1075], jepa:[505.4,512.6,510.1,522.2,196.4,184,3969], smolvla:[583.6,584.4,584.4,585.1,229.9,145,1196], n15:[591.6,594.9,593.4,597.1,138.9,162,3639], n16:[696.5,699.3,699.2,702.2,145.1,147,4692], adapter:[703.2,708.4,708.2,715.3,258.1,872,3566], n17:[759.5,766.3,766,773.4,197.6,191,5113], pi0:[1114,1116.6,1116.4,1118.2,277.1,284,5537], pi05:[1113,1117.3,1116.6,1120.7,277.9,285,5826], oft:[1720.3,1725.5,1724.7,1731,263.4,844,15451], evo1:[2333.6,2348.3,2349.9,2360.1,820.6,319,1757] },
    ovnpu: { turbo:[23.5,25.1,24,28.8,null,486,1790], jepa:[90.7,91.3,91.3,91.9,22.7,3953,8991], adapter:[126.7,128.5,128,130.3,49.8,3129,6925], n15:[134,135.2,135,136.1,28.5,3752,9727], n17:[144.9,145.8,145.8,146.3,22.8,5083,12417], n16:[161.1,161.9,161.8,162.6,28.4,4923,12108], smolvla:[161.3,165,164.9,166.4,55.4,1191,4975], evo1:[256.3,258.2,258.1,259.2,78.7,1528,5579], oft:[405.8,408.4,408.4,410.2,51.6,14975,35112], pi05:[430.2,438.1,438,441.9,57.1,5784,14942], pi0:[438.3,444.6,445.4,447.1,57.4,5594,14214] },
    i7: { octo:[39.8,39.9,39.9,39.9,5.4,622], turbo:[195.1,257.5,285.7,294.8,null,854], jepa:[825.5,1192.8,1211.7,1213.2,394.6,3822], adapter:[1140.6,1585.4,1599.6,1602.2,918.7,2707], n17:[1453.4,1455.2,1455.5,1456.2,403.6,4882], n16:[1627.9,1631.5,1631.1,1633.8,500.6,4524], smolvla:[1689.2,1691.4,1691.5,1692.7,1083.6,1100], n15:[1913.8,1915.1,1915.1,1916.2,500.4,3499], evo1:[4159.6,4164.7,4164.5,4168.4,1941.3,1431], pi05:[6068.1,6078,6077.8,6083.2,993.9,5657], pi0:[6455.7,6465.7,6467.3,6471.3,1022.3,5355], oft:[9869,9881.1,9882,9892.2,963.3,14797] },
    i9: { octo:[67,71.8,72.2,75,8.4,622], turbo:[314.7,325.9,325.5,333.7,null,854], jepa:[1227.2,1273.3,1280.7,1294.7,400.9,3822], n17:[1505.5,1530.8,1533,1545.2,400.5,4883], adapter:[1659.3,1689.6,1684.4,1729.9,980.8,2706], n16:[1688.9,1721.9,1717.9,1745.8,496.1,4524], smolvla:[1822.6,1855.6,1854.4,1874.7,1179.1,1099], n15:[1935.4,1966.1,1968.8,1983,494,3499], evo1:[4118.2,4155.7,4152.5,4184.5,1875.2,1431], pi05:[5735,5913.9,5799.2,6092.9,1001.6,5657], pi0:[6182.1,6342.8,6250.5,6507.3,1035.7,5353], oft:[9126.4,9636.7,9836.8,9854.8,1008.6,14796] },
    x7cpu: { octo:[48,50.7,50.8,51.6,5.8,622], turbo:[199,231.6,233.6,237.1,null,855], jepa:[1258.1,1265,1264.8,1269.8,421,3823], n17:[1532.5,1538.6,1537.5,1544,427.7,4884], adapter:[1645.4,1665.8,1667.4,1671.5,964.4,2707], n16:[1697.3,1705.1,1703.9,1711.2,519,4524], smolvla:[1758,1763.2,1763,1767.6,1072.9,1101], n15:[1979.1,1986.3,1987.4,1991.4,522.3,3500], evo1:[4025.8,4121.8,4138.3,4145.1,2051.2,1432], pi0:[6083.9,6160.1,6166.6,6183.4,1050.5,5356], pi05:[6120.3,6134.4,6133.9,6142.2,1046.5,5657], oft:[9894.3,9989.6,9995.9,10004.3,1017.9,14797] },
    x7ovcpu: { turbo:[238.4,281.2,260.7,301.1,null,1816], jepa:[1407.2,1413.3,1411.8,1415.3,491.4,11694], adapter:[1831.2,1838.1,1837.4,1842.8,1104.5,7890], smolvla:[1899.5,1913.8,1914.2,1923.4,1242.1,4186], n17:[2076.5,2083.7,2081.9,2086.3,488.3,14927], n16:[2300.4,2311.6,2312.3,2317.3,558.6,13833], n15:[2456.6,2466.9,2465.6,2477.2,629.8,10635], evo1:[4626.5,4650.7,4638.1,4689.2,2031.6,4765], pi05:[7522.7,7551,7544.4,7582.7,1265.5,16600], pi0:[7537.1,7584.4,7585.8,7607.5,1291.6,15828], oft:[11651.3,11721.3,11730.9,11750.1,1156.9,43074] },
    i5: { octo:[78.1,78.9,78.5,80.4,9.5,623], turbo:[392.7,394.9,393.9,396.6,null,855], jepa:[1668.5,1674.6,1674.4,1680.9,568.6,3823], n17:[1973.6,1982.7,1981.6,1991.9,567.5,4884], adapter:[2165.7,2179.3,2177.6,2186.4,1285.5,2707], n16:[2216.8,2228.1,2227,2234.8,699.7,4524], smolvla:[2288.4,2304.3,2296.6,2338.6,1520.8,1101], n15:[2681.4,2692.9,2691.6,2702.3,697.4,3500], evo1:[5435,5474.8,5456.3,5521.2,2682.3,1432], pi05:[8637.9,8653.7,8651.9,8671.8,1388.6,5657], pi0:[8953.5,8979.1,8975.9,8994.5,1413.9,5356] },
    r5: { octo:[96.6,97.2,97.1,97.5,11.2,622], turbo:[475.7,476.1,476.1,476.4,null,855], jepa:[2133,2147.9,2136.5,2144.8,726.6,3823], n17:[2518.8,2525.3,2522.6,2534.5,728,4884], adapter:[2781.7,2804.8,2798.3,2803.6,1623,2707], n16:[2826,2833.3,2827.5,2842.3,892.4,4524], smolvla:[3124.3,3131.8,3131.5,3138.8,2125,1101], n15:[3447,3461.6,3450.2,3471.3,887.2,3500], evo1:[7347.2,7385.4,7381.8,7400.9,3651.6,1432], pi05:[11225.6,11333.6,11342.5,11402.1,1788.3,5656], pi0:[11321.3,11439.9,11443.3,11505.5,1810.5,5355] },
    oryon: { octo:[109.5,111.6,111.1,114.3,13.1,623,729], turbo:[548.9,591.2,617.6,625.4,null,847,1020], jepa:[1656.5,1668.1,1664.1,1674.2,575.5,3826,4886], n17:[1988.1,2018,2002.6,2043.9,570.8,4886,5460], adapter:[2013.1,2181.2,2180.4,2209.5,1279.4,2708,3508], smolvla:[2080.9,2312.7,2321.2,2337.6,1540.8,1063,1352], n16:[2226,2352.5,2272.7,2524.1,717.9,4524,4648], n15:[2736.6,2863.4,2761.5,3080.6,734.4,3488,3599], evo1:[5518.9,5885.5,6116.9,6142.7,2785.1,1363,1493], pi05:[9140.6,10325.4,10198.6,11022.6,1614.3,5643,5674], pi0:[9224.5,10401.7,10298.1,11023.1,1556.6,5321,5472] }
  };

  var T = {
    rtx5090: {  },
    rtx3090: { turbo:['--weight-dtype f16 --flash-attn',5.7,5.8,5.8,5.8,null,-38], jepa:['--weight-dtype f16',23.1,23.8,23.4,25.1,8,-14], adapter:['--weight-dtype f16 --flash-attn',29.3,29.5,29.4,29.6,14,-14], n15:['--weight-dtype f16 --flash-attn',30.3,30.6,30.5,30.8,8.1,-17], n17:['--weight-dtype f16 --flash-attn',33.4,33.9,33.5,35.7,7.7,-5], n16:['--weight-dtype f16 --flash-attn',33.5,33.7,33.6,33.8,7.8,-9], smolvla:['--flash-attn --mm-prec default',47.7,47.9,47.9,48.2,18.8,-22], pi0:['--weight-dtype f16 --flash-attn',76.6,77.1,77.2,77.5,16,-21], evo1:['--weight-dtype f16 --flash-attn',82.2,82.8,82.8,83.2,21.7,-24], pi05:['--weight-dtype f16',84.3,84.6,84.6,84.8,15.9,-15], oft:['--weight-dtype f16',88.6,88.9,88.8,89.2,16.2,-34] },
    rtx3060: { turbo:['--weight-dtype f16 --flash-attn',13.7,13.7,13.7,13.8,null,-38], jepa:['--weight-dtype f16 --flash-attn',46.9,47.6,47.6,48.1,17.5,-19], adapter:['--weight-dtype f16',65.6,66.3,65.9,68.2,35.9,-12], n17:['--weight-dtype f16 --flash-attn',68.2,68.8,68.3,69.9,17.2,-18], n15:['--weight-dtype f16 --flash-attn',69.3,69.9,69.6,71.4,19.6,-20], n16:['--weight-dtype f16 --flash-attn',72.1,72.6,72.2,74,19.4,-8], smolvla:['--flash-attn --mm-prec default',91.3,92.7,92,94.8,45.4,-26], pi0:['--weight-dtype f16 --flash-attn',172.1,172.3,172.3,172.5,38.1,-25], evo1:['--act-dtype bf16 --flash-attn',176.2,176.8,176.4,176.9,56.2,-24], pi05:['--weight-dtype f16',191.3,191.7,191.6,192,38.7,-17] },
    rtx5070l: { octo:['--flash-attn',4.4,4.7,4.8,4.8,0.6,-6], turbo:['--weight-dtype f16 --flash-attn',8.6,9.1,9.1,9.3,null,-37], adapter:['--weight-dtype f16 --flash-attn',48.1,53.6,49.6,63.7,27.4,-15], n15:['--weight-dtype f16 --flash-attn',59.8,64.8,61.2,76.4,15.6,-18], smolvla:['--flash-attn --mm-prec default',60.4,65.3,61.4,77.5,29.4,-34], evo1:['--weight-dtype f16 --flash-attn',129,141.8,138.4,153.9,36,-30], pi0:['--weight-dtype f16 --flash-attn',135.4,149,145.3,169.3,31.5,-24], pi05:['--weight-dtype f16 --flash-attn',147.1,161,156.6,174.9,29,-18] },
    rx9060xt: { turbo:['--weight-dtype f16 --flash-attn',31.7,31.8,31.8,31.9,null,-6], n15:['--flash-attn',75.9,76.5,76.6,77.1,19.8,-4], smolvla:['--flash-attn --mm-prec default',127.3,127.7,127.8,128,43.1,-39], evo1:['--weight-dtype f16 --flash-attn',214.7,215.5,215.4,216,62.8,-8] },
    agx: { octo:['--flash-attn',13.3,17.8,16,23.2,2.5,-12], smolvla:['--flash-attn --mm-prec default',140.5,141.6,141.1,143.4,72.4,-14], evo1:['--flash-attn --mm-prec default',282.9,283.3,283.1,284.1,107.6,-5] },
    nx: { turbo:['--weight-dtype bf16 --flash-attn',46.7,46.8,46.8,46.8,null,-41], jepa:['--weight-dtype f16 --flash-attn',154.9,155,155,155.1,62,-12], n17:['--weight-dtype f16 --flash-attn',233,233.1,233,233.1,62,-7], adapter:['--weight-dtype f16',236.7,236.9,236.8,237.3,131.2,-13], n16:['--weight-dtype f16',243.1,243.3,243.2,244.1,68.3,-4], n15:['--weight-dtype f16 --flash-attn',249.2,249.5,249.4,250.1,70.9,-4], smolvla:['--flash-attn --mm-prec default',254.1,254.3,254.2,254.7,151.8,-38], pi0:['--weight-dtype f16 --flash-attn',597.6,597.9,597.8,598.2,140,-19], evo1:['--act-dtype bf16 --flash-attn',626,626.3,626.2,626.4,196,-30], pi05:['--weight-dtype f16 --flash-attn',664.7,665.1,665.1,665.6,135.8,-11] },
    nano: { turbo:['--weight-dtype bf16 --flash-attn',52.4,53.1,53.1,53.2,null,-41], jepa:['--weight-dtype f16 --flash-attn',173.7,174.8,174.6,176.6,70.6,-10], n17:['--weight-dtype f16 --flash-attn',254.5,255,254.9,255.6,70.4,-6], adapter:['--weight-dtype f16',266.2,266.6,266.5,266.8,148.1,-11], smolvla:['--flash-attn --mm-prec default',286.7,287.8,287.3,289.9,173.1,-38], pi0:['--weight-dtype f16 --flash-attn',688.2,688.7,688.8,689,160,-19], evo1:['--act-dtype bf16 --flash-attn',720.8,722.7,722.7,723.6,233.4,-29], pi05:['--weight-dtype f16 --flash-attn',764.8,765.8,765.7,766.4,153.2,-10] },
    m4: { turbo:['--weight-dtype f16 --flash-attn',54.8,55,55,55.2,null,-16], smolvla:['--weight-dtype f16 --flash-attn',325.6,326,325.9,326.2,185.9,-9], evo1:['--weight-dtype f16 --flash-attn',771.4,772.1,772.2,772.4,327.1,-7] },
    b390sycl: { n15:['--weight-dtype bf16 --flash-attn',87.8,88.2,88.3,88.5,13,-13], jepa:['--flash-attn',93.1,93.8,93.7,94.4,32.4,-9], evo1:['--act-dtype bf16 --flash-attn',142.9,144.2,144,144.9,26.2,-47], pi05:['--act-dtype bf16 --flash-attn',164.9,165.5,165.4,165.9,15.2,-40], n17:['--weight-dtype bf16 --flash-attn',167.9,168.7,168.7,169.5,32.2,-6], pi0:['--act-dtype bf16 --flash-attn',243.6,244.6,244.5,245.3,46.3,-9] },
    b390ov: { turbo:['--weight-dtype bf16 --flash-attn',25.7,26.6,26.6,27.2,null,-10], n16:['--weight-dtype f16 --flash-attn',121.9,123.4,123.5,124.1,14.6,-6], smolvla:['--weight-dtype bf16 --flash-attn',169.6,170.2,170.2,170.6,65.3,-21], evo1:['--weight-dtype f16 --flash-attn',467.4,472.7,472.5,477,135.3,-17] },
    a380: { turbo:['--weight-dtype f16',89.7,89.9,89.9,90.1,null,-13] },
    hexnpu: { turbo:['--weight-dtype f16 --flash-attn',490.1,554.7,541.9,600.9,null,-76] },
    iq10: { jepa:['--flash-attn --mm-prec default',442.1,452.4,450.2,460.6,198.3,-11.7], n17:['--flash-attn --mm-prec default',727.8,734.1,733.2,742.4,196.2,-4.2] },
    ovnpu: {  },
    i7: { smolvla:['--weight-dtype bf16 --flash-attn',1546.8,1548.7,1547.7,1549.2,938.6,-8], evo1:['--flash-attn --mm-prec default',3893.4,3897,3896.2,3900.1,1673.1,-6] },
    i9: { smolvla:['--flash-attn',1630.3,1661.5,1658.8,1681.5,982.4,-10], evo1:['--flash-attn --mm-prec default',3906.7,3952.3,3951.2,3981.4,1711.6,-5] },
    x7cpu: { turbo:['--flash-attn',184.8,220.4,221.7,229.2,null,-5], jepa:['--weight-dtype f16',1020.9,1026.4,1026.3,1030.3,344.4,-19], n17:['--weight-dtype f16',1251,1255.7,1255.1,1258.8,348.4,-18], adapter:['--weight-dtype f16 --flash-attn',1341.4,1368,1370.5,1376.3,800.2,-18], n16:['--weight-dtype f16 --flash-attn',1363.4,1368,1367,1371.2,392.9,-20], smolvla:['--weight-dtype f16 --flash-attn',1388.5,1399.4,1399.2,1406.6,799.3,-21], n15:['--weight-dtype f16 --flash-attn',1564.6,1568.8,1568.9,1571.5,390.4,-21], evo1:['--weight-dtype f16 --flash-attn',3256,3283.5,3286.3,3296.1,1503.2,-20], pi05:['--weight-dtype f16',4783.6,4891.5,4902.1,4917.8,842.3,-20], pi0:['--weight-dtype f16',4835.2,4919,4922.9,4955.6,850.1,-20], oft:['--weight-dtype f16 --flash-attn',7879.9,7922.2,7915.3,7940.6,847.6,-21] },
    x7ovcpu: { n15:['--flash-attn --mm-prec default',2380.8,2391,2388.7,2400.8,559.7,-3] },
    i5: { turbo:['--weight-dtype f16 --flash-attn',357.7,361,359.6,364.8,null,-9], smolvla:['--weight-dtype f16 --flash-attn',2044.6,2064.2,2057.7,2082.8,1274.9,-10], evo1:['--flash-attn --mm-prec default',5061.5,5081.8,5076.1,5105.2,2299.2,-7] },
    r5: { turbo:['--weight-dtype f16 --flash-attn',452.7,453.2,453.1,453.6,null,-5], smolvla:['--weight-dtype f16 --flash-attn',2684.7,2687.5,2687.5,2688.7,1674.1,-14], evo1:['--weight-dtype f16 --flash-attn',6798.7,6813.3,6802.8,6808,3057.5,-8] },
    oryon: { turbo:['--weight-dtype f16',358.8,363.3,363.1,367.3,null,-39] }
  };

  /* Why a pair has no number, from each report's "Not run" section. The kind picks the cell style:
     na, no path on this backend; oom, ran out of memory; big, not attempted because the weights
     exceed the memory; fail, errors out; nr, not run. */
  var WHY = {
    int2:  ['na',  'No path. The only published BitVLA GGUF is int2-packed, and only CUDA builds load it.'],
    ovop:  ['na',  'No path yet. OpenVINO has no translation for one of Octo-Small’s ops.'],
    oom:   ['oom', 'Does not fit. Loading the weights runs out of device memory.'],
    oomrun:['oom', 'Does not fit. The weights load, but the first prediction runs out of device memory.'],
    big8:  ['big', 'Not attempted. Its 15.1 GB of weights exceed the board’s 8 GB.'],
    big16: ['big', 'Not attempted. Its 15.1 GB of weights do not fit in the 15.3 GiB the board exposes.'],
    ram15: ['big', 'Not attempted. Its 14 GiB of weights do not fit in 15 GiB of RAM.'],
    win16: ['big', 'Not attempted. Its 14 GiB of weights do not fit next to Windows in 16 GB of RAM.'],
    actnr: ['nr',  'Not run. Only the Dragonwing IQ-10 report measures ACT so far.']
  };

  var SKIP = {
    rtx3060: { oft:'oom' }, rtx5070l: { oft:'oom' },
    rx9060xt:{ bitvla:'int2' }, a380: { bitvla:'int2', oft:'oomrun' },
    nx: { oft:'big16' }, nano: { oft:'big8' }, m4: { bitvla:'int2' },
    b390sycl:{ bitvla:'int2' }, b390ov: { bitvla:'int2', octo:'ovop' },
    hexnpu: { bitvla:'int2', oft:'win16' }, iq10: { bitvla:'int2' },
    ovnpu: { octo:'ovop', bitvla:'int2' },
    i7: { bitvla:'int2' }, i9: { bitvla:'int2' }, x7cpu: { bitvla:'int2' },
    x7ovcpu: { bitvla:'int2', octo:'ovop' }, i5: { bitvla:'int2', oft:'ram15' },
    r5: { bitvla:'int2', oft:'ram15' }, oryon: { bitvla:'int2', oft:'win16' }
  };

  /* Fixed-input checks from the backend notes: the largest absolute difference in the action chunk
     against the reference the device's chk names, on fixed images, tokens, state and noise. Kept
     as printed; the OpenVINO ones are on the continuous action dims. The notes hold the other
     backends to 2.9e-3. */
  var BAR = 2.9e-3;
  var CHECK = {
    rx9060xt:{ octo:'2.98e-7', turbo:'7.23e-7', jepa:'9.14e-5', adapter:'1.55e-6', n15:'3.35e-4',
               n16:'3.39e-4', n17:'3.86e-4', smolvla:'1.74e-4', pi0:'4.79e-4', pi05:'4.62e-5',
               evo1:'2.11e-6' },
    b390ov:  { turbo:'2.7e-6', jepa:'1.4e-2', adapter:'4.8e-5', n15:'8.1e-4', n16:'2.2e-3', n17:'1.7e-2',
               smolvla:'1.2e-2', pi0:'7.3e-4', pi05:'4.2e-4', oft:'4.1e-5', evo1:'5.8e-5' },
    x7ovcpu: { turbo:'3.1e-6', jepa:'1.4e-3', adapter:'4.7e-5', n15:'4.9e-4', n16:'2.5e-3', n17:'6.6e-4',
               smolvla:'7.6e-4', pi0:'1.3e-4', pi05:'8.2e-5', oft:'4.0e-5', evo1:'5.7e-5' },
    ovnpu:   { turbo:'8.1e-3', jepa:'1.4e-2', adapter:'7.6e-3', n15:'8.8e-3', n16:'4.9e-3', n17:'2.2e-2',
               smolvla:'1.8e-2', pi0:'3.7e-2', pi05:'5.9e-3', oft:'8.1e-4', evo1:'3.8e-2' },
    iq10:    { octo:'5.3e-4', act:'6.1e-4', turbo:'3.7e-4', jepa:'2.2e-3', adapter:'5.4e-4', n15:'2.4e-3',
               n16:'1.4e-3', n17:'2.0e-3', smolvla:'7.2e-4', pi0:'1.2e-3', pi05:'4.8e-4', oft:'3.2e-4',
               evo1:'5.5e-4' },
    hexnpu:  { octo:'5.2e-4', turbo:'6.0e-4', jepa:'4.8e-3', adapter:'7.0e-4', n15:'1.5e-3',
               n16:'4.1e-4', n17:'1.6e-3', smolvla:'2.0e-3', pi0:'5.6e-3', pi05:'7.1e-4',
               evo1:'4.1e-4' }
  };

  /* What a report says about one pair in particular. */
  var PAIR = {
    'rx9060xt.oft':  'It fits: its BF16 weights take 14.9 GB of the 16 GB. There is no CPU check, ' +
                     'because they exceed the host’s 15.4 GiB of RAM.',
    'rx9060xt.smolvla': 'Keeps its LM weights at F32 under the default precision, as on CUDA: 208.5 ms ' +
                     'mean instead of 228.1, with byte-identical actions, for 2303 MiB of VRAM instead of 1541.',
    'agx.octo':      'Bound by the host: its graphs are small, and the time goes to launching them. Its ' +
                     'mean moves by about 15% between runs of the same build.',
    'agx.n17':       'The vision tower runs inside the main graph on this path, so there is no separate ' +
                     'vision time.',
    'b390sycl.pi0':  'Its three process means were 270.0, 282.5 and 270.7 ms; the row is the fast one.',
    'b390sycl.pi05': 'Its three process means were 276.4, 277.0 and 295.5 ms; the row is the fast one.',
    'b390sycl.oft':  'Its three process means were 464.1, 438.5 and 446.6 ms; the row is the fast one.',
    'b390sycl.n17':  'Its fused graph is opt-in and not in this table: with it, 114.0 ms mean against 179.4.',
    'b390ov.adapter':'Its three process means were 185.4, 216.8 and 215.6 ms; the row is the fast one, ' +
                     'and the report found the spread changes from run to run.',
    'b390ov.oft':    'Its first process ran slow (1749.9 ms against 1351.5 and 1334.9); it holds 41 GB of ' +
                     'device buffers out of 62 GiB of shared RAM.',
    'b390ov.pi0':    'π0 alone runs the GPU at F32 inference precision by default, which costs about ' +
                     '2.5×: at F16 it runs in 254.6 ms, but its gripper then flips a step early in one ' +
                     'chunk in 48.',
    'b390ov.n17':    'One of the six fixed chunks flips its gripper a step early, as ggml’s own BF16 run ' +
                     'does on these inputs.',
    'b390ov.smolvla':'One of the six fixed chunks flips its gripper a step early, as ggml’s own BF16 run ' +
                     'does on these inputs.',
    'ovnpu.n17':     'One of the six fixed chunks flips its gripper a step early, as ggml’s own BF16 run ' +
                     'does on these inputs.',
    'ovnpu.smolvla': 'One of the six fixed chunks flips its gripper a step early, as ggml’s own BF16 run ' +
                     'does on these inputs.',
    'hexnpu.turbo':  'Its F32 weights stay F32 at the defaults; F16 weights with flash attention cut the ' +
                     'mean by 76%.',
    'hexnpu.jepa':   'Precision-sensitive: its own CPU run on the same F16 weights lands further out, at ' +
                     '8.5×10⁻³.',
    'hexnpu.pi0':    'Precision-sensitive: its own CPU run on the same F16 weights lands further out, at ' +
                     '1.2×10⁻².',
    'hexnpu.pi05':   'Checked against the CPU BF16 run: an F32 reference needs more memory than the ' +
                     '8 GB budget.',
    'hexnpu.n15':    'Checked before flash attention reached the vision tower; the NPU now turns it on ' +
                     'by default, and this was not re-checked.',
    'hexnpu.n16':    'Checked before flash attention reached the vision tower; the NPU now turns it on ' +
                     'by default, and this was not re-checked.',
    'ovnpu.pi0':     'A latency only. On 48 fixed inputs one chunk flips its gripper a step early, as ' +
                     'the GPU plugin at F16 does; the NPU has no F32 mode.',
    'iq10.act':      'An ALOHA checkpoint on one 640×480 view, no language. Its ResNet’s im2col matrices ' +
                     'are zero-padded to a multiple of 32 so the convolutions’ matmuls take the HMX; the ' +
                     'board’s CPU runs it in 170.8 ms.',
    'iq10.jepa':     'The defaults leave its LM off flash attention, which lands at 6.7×10⁻³ on a served ' +
                     'observation; the fastest flags put it back on.',
    'iq10.n17':      'The defaults leave its LM off flash attention, which lands at 5.4×10⁻³ on fixed ' +
                     'inputs; the fastest flags put it back on.',
    'iq10.n15':      'On a served LIBERO observation it lands at 1.1×10⁻², above the bar but closer to ' +
                     'the reference than the CPU at F16: sensitive to 16-bit precision, not miscomputed.',
    'iq10.n16':      'On a served LIBERO observation it lands at 1.3×10⁻², above the bar and slightly ' +
                     'further than the CPU at F16 (1.1×10⁻²). One view here: with two, one batched ' +
                     'matmul of its SigLIP2 tower still runs on the CPU.',
    'i7.turbo':      'Its min catches boost power at the start of the process and sits about 30% below p50.',
    'i7.jepa':       'Its min catches boost power at the start of the process and sits about 30% below p50.',
    'i7.adapter':    'Its min catches boost power at the start of the process and sits about 30% below p50.',
    'x7cpu.octo':    'Runs at one of two speeds set by the machine’s state: this row is a fast process ' +
                     '(50.7 ms mean); the other two gave 57.2 and 59.8 ms.',
    'x7cpu.turbo':   'Its min sits 15% below its p50: the first calls run at boost power. It also runs ' +
                     'at one of two speeds; this row is the slow one, and the September report’s ' +
                     '163 ms was the fast one.',
    'x7ovcpu.adapter':'About 5% slower than in the September round: the 2D im2col fold costs its vision ' +
                     'stage 110 ms on the CPU plugin, while it is 5–7% faster on the GPU plugin.'
  };

  /* Cell shading: p50 bins in ms, on a log scale. Six steps of one teal, darker is faster. */
  var BINS = [30, 100, 300, 1000, 3000];
  var LABEL = { na:'n/a', oom:'OOM', big:'too big', fail:'fails', nr:'n/r' };
  var SAY   = { na:'no path', oom:'does not fit', big:'not attempted', fail:'fails', nr:'not run' };

  var dev  = 'agx';
  var mod  = 'n17';
  var sort = 'lat';
  var el   = {};

  /* ---------------- helpers ---------------- */

  function model(id) { for (var i = 0; i < MODELS.length; i++) if (MODELS[i].id === id) return MODELS[i]; return null; }
  function device(id) { for (var i = 0; i < DEVICES.length; i++) if (DEVICES[i].id === id) return DEVICES[i]; return null; }
  function row(d, m) { return (R[d] && R[d][m]) || null; }

  /* The headline: p50, or the minimum where that is all a report gives. */
  function lat(d, m) { var r = row(d, m); return r ? (r[2] != null ? r[2] : r[0]) : null; }

  function why(d, m) {
    var k = SKIP[d] && SKIP[d][m];
    return k ? WHY[k] : (m === 'act' ? WHY.actnr : ['nr', 'Not run.']);
  }

  /* One decimal, as the reports print, or two where a report gives two (the RTX 5090). */
  function fix(v) { var s = v.toFixed(1); return Math.abs(parseFloat(s) - v) > 1e-9 ? v.toFixed(2) : s; }
  function ms(v) { return v == null ? '<span class="xp-dim">—</span>' : fix(v); }
  function pct(v) { return (v < 0 ? '−' : '+') + Math.abs(v) + '%'; }
  function mib(v) { return v == null ? '<span class="xp-dim">—</span>' : String(Math.round(v)); }
  function short(v) { return v < 100 ? v.toFixed(1) : String(Math.round(v)); }

  function bin(v) { for (var i = 0; i < BINS.length; i++) if (v < BINS[i]) return i; return BINS.length; }

  var SUP = { '-':'⁻', '0':'⁰', '1':'¹', '2':'²', '3':'³', '4':'⁴', '5':'⁵', '6':'⁶', '7':'⁷', '8':'⁸', '9':'⁹' };
  function sci(s) {
    var p = s.split('e');
    if (p.length < 2) return s;
    return p[0] + '×10' + p[1].replace(/./g, function (c) { return SUP[c] || c; });
  }

  function wilson(k, n) {
    var z = 1.959964, p = k / n, d = 1 + z * z / n;
    var c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.round(1000 * (c - h)) / 10, Math.round(1000 * (c + h)) / 10];
  }

  /* Columns for the device table: the latency fields and memory columns this device reports. */
  function columns(d) {
    var p = device(d), out = [], base = [
      { h:'p50', i:2, lat:true }, { h:'p90', i:3 }, { h:'min', i:0, lat:!!p.minOnly }, { h:'vision', i:4 }
    ];
    for (var k = 0; k < p.mem.length; k++) base.push({ h:p.mem[k], i:5 + k, mem:true });
    for (var j = 0; j < base.length; j++) {
      for (var m in R[d]) if (R[d][m][base[j].i] != null) { out.push(base[j]); break; }
    }
    return out;
  }

  function sorted(d) {
    var yes = [], no = [];
    for (var i = 0; i < MODELS.length; i++) (row(d, MODELS[i].id) ? yes : no).push(MODELS[i]);
    yes.sort(function (a, b) {
      if (sort === 'succ') return (b.ok == null ? -1 : b.ok) - (a.ok == null ? -1 : a.ok) || lat(d, a.id) - lat(d, b.id);
      return lat(d, a.id) - lat(d, b.id);
    });
    return { yes: yes, no: no };
  }

  /* ---------------- panes ---------------- */

  function renderMatrix() {
    var out = '<div class="xp-mgrid"><div class="xp-mc"></div>', g = null, i, j;
    for (j = 0; j < MODELS.length; j++)
      out += '<button type="button" class="xp-mh' + (MODELS[j].id === mod ? ' on' : '') +
             '" data-mod="' + MODELS[j].id + '">' + MODELS[j].name + '</button>';

    for (i = 0; i < DEVICES.length; i++) {
      var p = DEVICES[i];
      if (p.group !== g) { g = p.group; out += '<div class="xp-mg">' + g + '</div>'; }
      out += '<button type="button" class="xp-mr' + (p.id === dev ? ' on' : '') + '" data-dev="' + p.id +
             '"><b>' + p.short + '</b><span>' + p.be + '</span></button>';
      for (j = 0; j < MODELS.length; j++) {
        var m = MODELS[j], v = lat(p.id, m.id), sel = (m.id === mod && p.id === dev) ? ' sel' : '';
        var what = m.name + ' on ' + p.name + ' (' + p.be + ')';
        if (v != null) {
          var key = p.minOnly ? 'min ' : 'p50 ';
          out += '<button type="button" class="xp-c b' + bin(v) + (p.minOnly ? ' mo' : '') + sel +
                 '" data-dev="' + p.id + '" data-mod="' + m.id + '" title="' + what + ': ' + key +
                 fix(v) + ' ms">' + short(v) + '</button>';
        } else {
          var w = why(p.id, m.id);
          out += '<button type="button" class="xp-c s-' + w[0] + sel + '" data-dev="' + p.id +
                 '" data-mod="' + m.id + '" title="' + what + ': ' + w[1] + '">' + LABEL[w[0]] + '</button>';
        }
      }
    }
    out += '</div>';

    var ramp = '', ticks = '';
    for (i = 0; i <= BINS.length; i++) ramp += '<span class="b' + i + '"></span>';
    for (i = 0; i < BINS.length; i++)
      ticks += '<span style="left:' + (100 * (i + 1) / (BINS.length + 1)).toFixed(3) + '%">' +
               (BINS[i] < 1000 ? BINS[i] : BINS[i] / 1000 + ' s') + '</span>';
    out += '<div class="xp-key"><div class="xp-scale6"><div class="xp-ramp">' + ramp + '</div>' +
           '<div class="xp-rampt">' + ticks + '</div>' +
           '<p class="xp-slg">p50 at the defaults, ms · darker is faster</p></div>' +
           '<ul class="xp-leg">' +
           '<li><i class="xp-c s-na">n/a</i>no path on this backend</li>' +
           '<li><i class="xp-c s-oom">OOM</i><i class="xp-c s-big">too big</i>does not fit</li>' +
           '<li><i class="xp-c s-nr">n/r</i>not run</li>' +
           '<li><i class="xp-c b2 mo">20.8</i>min only</li></ul></div>' +
           '<p class="xp-cap">Engine latency from <code>vla-bench</code>: <code>predict()</code> timed in ' +
           'process on synthetic inputs, with no transport, simulator or task. Each cell is the p50 of 20 ' +
           'timed calls after 3 warmups, from the fastest of three processes, at the default runtime ' +
           'flags. Click a cell, a device or a model.</p>';
    el.matrix.innerHTML = out;
  }

  function renderModel() {
    var m = model(mod), p = device(dev), r = row(dev, mod), i;
    var out = '<div class="xp-mname">' + m.name + '</div>' +
              '<div class="xp-mhead">' + m.head + (m.steps > 1 ? ' · ' + m.steps + ' steps' : '') + ' · ' +
              m.views + ' × ' + m.input + ' px · ' + m.chunk + '-action chunk</div>';

    /* LIBERO-Object, 200 episodes: success with its Wilson interval. */
    if (m.ok == null) {
      out += '<div class="xp-sec">LIBERO-Object</div>' +
             '<p class="xp-nm">No success rate: its checkpoint is an ALOHA policy, not a LIBERO one.</p>';
    } else {
    var rate = m.ok / 2, ci = wilson(m.ok, 200);
    out += '<div class="xp-sec">LIBERO-Object · 200 episodes</div>' +
           '<div class="xp-scale"><div class="xp-track">' +
           '<span class="xp-fill" style="width:' + rate + '%"></span></div>' +
           '<div class="xp-whisk"><span style="left:' + ci[0] + '%;right:' + (100 - ci[1]) + '%"></span></div>' +
           '<div class="xp-ticks"><span>0</span><span>50</span><span>100%</span></div>' +
           '<p class="xp-slg">bar: success · caps: 95% interval</p></div>' +
           '<table class="xp-kv"><tbody>' +
           '<tr><td>success</td><td><b>' + m.ok + '/200</b> · ' + rate.toFixed(1) + '% ' +
           '<span class="xp-dim">[' + ci[0].toFixed(1) + ', ' + ci[1].toFixed(1) + ']</span></td></tr>' +
           '<tr><td>replay</td><td>' + m.replay + ' of ' + m.chunk + ' actions per query</td></tr>';
    out += '</tbody></table>';
    if (m.suites) {
      out += '<div class="xp-sub">four LIBERO suites, %</div><table class="xp-suites"><thead><tr>';
      for (i = 0; i < SUITES.length; i++) out += '<th>' + SUITES[i] + '</th>';
      out += '</tr></thead><tbody><tr>';
      for (i = 0; i < SUITES.length; i++) out += '<td class="xp-num">' + m.suites[i].toFixed(1) + '</td>';
      out += '</tr></tbody></table>';
    }
    out +=
           '<p class="xp-fn">On the RTX 3090 at the defaults, through the server and the LIBERO ' +
           'simulator. Success belongs to the checkpoint; it was not re-run on other devices.</p>';
    }

    /* This model on the selected device. */
    out += '<div class="xp-sec">on ' + p.name + ' · ' + p.be + '</div>';
    if (!r) {
      var w = why(dev, mod);
      out += '<p class="xp-nm' + (w[0] === 'na' || w[0] === 'nr' ? '' : ' bad') + '">' + w[1] + '</p>';
    } else {
      var kv = [['p50', 2, 'ms'], ['p90', 3, 'ms'], ['min', 0, 'ms'], ['mean', 1, 'ms'], ['vision', 4, 'ms']];
      for (i = 0; i < p.mem.length; i++) kv.push([p.mem[i], 5 + i, 'MiB']);
      out += '<table class="xp-kv"><tbody>';
      for (i = 0; i < kv.length; i++) {
        var v = r[kv[i][1]];
        if (v == null) continue;
        out += '<tr><td>' + kv[i][0] + ' <span class="xp-dim">(' + kv[i][2] + ')</span></td>' +
               '<td class="xp-num' + (kv[i][1] === (p.minOnly ? 0 : 2) ? ' ours' : '') + '">' +
               (kv[i][2] === 'MiB' ? mib(v) : ms(v)) + '</td></tr>';
      }
      out += '</tbody></table>';

      var t = T[dev] && T[dev][mod];
      out += '<div class="xp-sub">fastest flags</div>';
      if (p.minOnly) out += '<p class="xp-nm">No flag screen reported.</p>';
      else if (!t) out += '<p class="xp-nm">The defaults are the fastest flag set here.</p>';
      else out += '<p class="xp-flags"><code>' + t[0] + '</code></p>' +
                  '<table class="xp-kv"><tbody><tr><td>p50 <span class="xp-dim">(ms)</span></td>' +
                  '<td class="xp-num">' + ms(t[3]) + '</td></tr><tr><td>mean against defaults</td>' +
                  '<td class="xp-num">' + pct(t[6]) + '</td></tr></tbody></table>' +
                  '<p class="xp-fn">These flags can move the action chunk, so the success rate above ' +
                  'does not carry over to them.</p>';

      var c = CHECK[dev] && CHECK[dev][mod];
      if (c) {
        var over = parseFloat(c) > BAR;
        out += '<div class="xp-sub">fixed-input check</div>' +
               '<table class="xp-kv"><tbody><tr><td>max |Δ|</td><td class="xp-num' + (over ? ' bad' : '') +
               '">' + sci(c) + '</td></tr></tbody></table>' +
               '<p class="xp-fn">Action chunk from ' + p.chk + '.' +
               (over ? ' Above the 2.9×10⁻³ bar the backend notes hold the others to.' : '') + '</p>';
      }
    }
    var pn = PAIR[dev + '.' + mod];
    if (pn) out += '<p class="xp-fn">' + pn + '</p>';
    out += '<p class="xp-note">' + m.note + '</p>';
    el.model.innerHTML = out;
  }

  function renderTable() {
    var p = device(dev), cols = columns(dev), r = sorted(dev), t = T[dev] || {}, i, j;
    var tuned = !p.minOnly, max = 0;
    for (i = 0; i < r.yes.length; i++) max = Math.max(max, lat(dev, r.yes[i].id));

    var out = '<div class="xp-dhead"><b>' + p.name + '</b><i class="xp-b ' + p.tag + '">' + p.be +
              '</i><span>' + p.sub + ' · measured ' + p.date + '</span></div>' +
              '<table><thead><tr><th>Model</th>';
    for (i = 0; i < cols.length; i++)
      out += '<th>' + cols[i].h + '<span class="xp-u"> ' + (cols[i].mem ? 'MiB' : 'ms') + '</span></th>';
    if (tuned) out += '<th title="Change in mean from the fastest flag set">tuned<span class="xp-u"> Δ</span></th>';
    out += '<th class="xp-barh">' + (p.minOnly ? 'min' : 'p50') + '</th></tr></thead><tbody>';

    for (i = 0; i < r.yes.length; i++) {
      var m = r.yes[i], v = R[dev][m.id];
      out += '<tr data-mod="' + m.id + '" tabindex="0"' + (m.id === mod ? ' class="xp-sel"' : '') + '><th>' + m.name + '</th>';
      for (j = 0; j < cols.length; j++)
        out += '<td class="xp-num' + (cols[j].lat ? ' ours' : '') + '">' +
               (cols[j].mem ? mib(v[cols[j].i]) : ms(v[cols[j].i])) + '</td>';
      if (tuned)
        out += '<td class="xp-num"' + (t[m.id] ? ' title="' + t[m.id][0] + '">' + pct(t[m.id][6]) :
               '><span class="xp-dim">—</span>') + '</td>';
      var w = Math.max(2, Math.round(100 * lat(dev, m.id) / max));
      out += '<td class="xp-bar"><span style="width:' + w + '%"></span></td></tr>';
    }

    for (i = 0; i < r.no.length; i++) {
      var n = r.no[i], k = why(dev, n.id);
      out += '<tr class="xp-none' + (n.id === mod ? ' xp-sel' : '') + '" data-mod="' + n.id + '" tabindex="0"' +
             '><th>' + n.name + '</th><td colspan="' + (cols.length + (tuned ? 2 : 1)) + '">' + k[1] + '</td></tr>';
    }

    out += '</tbody></table><p class="xp-fn">' + p.note + '</p>';
    if (tuned) out += '<p class="xp-fn"><b>tuned Δ</b> is the change in mean latency from the fastest ' +
                      'flag set that beats the defaults by more than 3%, among flash attention, matmul ' +
                      'precision and F16 or BF16 weights. Hover it for the flags.</p>';
    el.table.innerHTML = out;
  }

  function renderVerdict() {
    var m = model(mod), p = device(dev), r = row(dev, mod), out;

    if (!r) {
      var w = why(dev, mod);
      out = '<b>' + m.name + ' on ' + p.name + ' (' + p.be + '): ' + SAY[w[0]] + '.</b>' +
            '<span class="xp-vsub">' + w[1] + '</span>';
    } else if (p.minOnly) {
      out = '<b>' + m.name + ' on ' + p.name + ': ' + fix(r[0]) + ' ms at best.</b>' +
            '<span class="xp-vsub">contributed number · minimum over five rounds · no p50 reported</span>';
    } else {
      var v = r[2], t = T[dev] && T[dev][mod], bits = [];
      bits.push((1000 / v).toFixed(1) + ' predictions per second');
      bits.push(m.chunk + '-action chunk');
      for (var k = 0; k < p.mem.length; k++)
        if (r[5 + k] != null) bits.push(Math.round(r[5 + k]) + ' MiB ' + p.mem[k]);
      out = '<b>' + m.name + ' on ' + p.name + ' (' + p.be + '): ' + v.toFixed(1) + ' ms p50.</b>' +
            '<span class="xp-vsub">' + bits.join(' · ') + '</span>';
      if (t)
        out += '<span class="xp-vnote">With <code>' + t[0] + '</code>: ' + t[3].toFixed(1) + ' ms p50, ' +
               pct(t[6]) + ' mean, on numerics that can move the action chunk.</span>';
    }
    if (m.ok != null) out += '<span class="xp-vnote">LIBERO-Object: ' + m.ok + ' of 200 on the RTX 3090.</span>';
    el.verdict.innerHTML = out;
  }

  function render() {
    renderMatrix(); renderModel(); renderTable(); renderVerdict();
    var sb = document.querySelectorAll('#xp-app [data-sort]');
    for (var i = 0; i < sb.length; i++)
      sb[i].setAttribute('aria-pressed', sb[i].dataset.sort === sort ? 'true' : 'false');
  }

  /* ---------------- wiring ---------------- */

  function init() {
    var root = document.getElementById('xp-app');
    if (!root) return;
    ['matrix', 'model', 'table', 'verdict'].forEach(function (k) {
      el[k] = root.querySelector('[data-pane="' + k + '"]');
    });

    /* Re-rendering replaces the panes, so put focus back on the control that was used. */
    function again(sel) {
      render();
      var n = sel && root.querySelector(sel);
      if (n) n.focus({ preventScroll: true });
    }

    root.addEventListener('click', function (e) {
      var c = e.target.closest('button.xp-c');
      if (c) { dev = c.dataset.dev; mod = c.dataset.mod;
               return again('button.xp-c[data-dev="' + dev + '"][data-mod="' + mod + '"]'); }
      var d = e.target.closest('[data-dev]');
      if (d) { dev = d.dataset.dev; return again('.xp-mr[data-dev="' + dev + '"]'); }
      var m = e.target.closest('[data-mod]');
      if (m) { mod = m.dataset.mod;
               return again((m.tagName === 'TR' ? 'tr' : '.xp-mh') + '[data-mod="' + mod + '"]'); }
      var sb = e.target.closest('[data-sort]');
      if (sb) { sort = sb.dataset.sort; return again('[data-sort="' + sort + '"]'); }
    });

    root.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      var r = e.target.closest('tr[data-mod]');
      if (r) { e.preventDefault(); mod = r.dataset.mod; again('tr[data-mod="' + mod + '"]'); }
    });

    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
