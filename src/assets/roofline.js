/* Roofline of the runtime's latest benchmark round. Achieved throughput is the work in one predict()
   divided by its p50 latency at the defaults (the explorer's numbers: vla-bench, September-October
   2026). The work is counted, not estimated: each model's ggml graphs were run on the CPU backend
   with a counter added to ggml, at the benchmark's views, input size and prompt length, and one
   predict() is the difference between a two-call and a one-call run. FLOPs are 2 x M x N x K per
   matrix multiply, attention included. Bytes are the stored operands those multiplies read: weights
   at their resident dtype, once per use (an action head's once per solver step), plus cached keys
   and values. Activations are not counted, so every intensity is an upper bound. OpenVLA-OFT is
   counted by hand under the same rules (hand:true): its 14 GB of weights did not fit the counting
   host. The ceilings are the paper's: measured BF16 GEMM rates and nominal bandwidth.
   The roof is a BF16 roof, so only models whose multiplies run in BF16 are on it. BitVLA multiplies
   in int8 on the tensor cores and Octo-Small and TurboVLA in F32; each has a different ceiling, and
   against this one BitVLA would sit above 100%. */
(function () {
  'use strict';

  var DEVICES = [
    { id:'rtx3060', name:'RTX 3060', peak:25.6, bw:360 },
    { id:'agx',     name:'Jetson AGX Orin', peak:21.3, bw:204.8 }
  ];

  /* gflop and gb per predict(); p50 in ms per device, copied from explorer.js. π0 and π0.5 land on
     the same spot, so they share one label (short, nolab). */
  var MODELS = [
    { id:'jepa',    name:'VLA-JEPA',    gflop:528.8,  gb:4.568,  p50:{ rtx3060:58.1,  agx:61.3 } },
    { id:'adapter', name:'VLA-Adapter', gflop:660.3,  gb:2.506,  p50:{ rtx3060:74.8,  agx:79.7 } },
    { id:'n15',     name:'GR00T N1.5',  short:'N1.5', gflop:863.2, gb:6.013, p50:{ rtx3060:86.9, agx:97.9 } },
    { id:'n16',     name:'GR00T N1.6',  short:'N1.6', gflop:703.1, gb:9.462, p50:{ rtx3060:78.2, agx:102.2 } },
    { id:'n17',     name:'GR00T N1.7',  short:'N1.7', gflop:621.9, gb:9.823, p50:{ rtx3060:83.0, agx:94.1 } },
    { id:'smolvla', name:'SmolVLA',     gflop:650.1,  gb:2.912,  p50:{ rtx3060:125.3, agx:165.0 } },
    { id:'pi0',     name:'π0',          short:'π0, π0.5', gflop:2821.2, gb:11.763, p50:{ rtx3060:228.6, agx:269.5 } },
    { id:'pi05',    name:'π0.5',        gflop:2810.9, gb:11.636, p50:{ rtx3060:230.1, agx:267.2 }, nolab:true },
    { id:'oft',     name:'OpenVLA-OFT', short:'OFT', gflop:4737.2, gb:14.807, p50:{ rtx3060:null, agx:297.7 }, hand:true },
    { id:'evo1',    name:'Evo-1',       gflop:1492.8, gb:7.981,  p50:{ rtx3060:231.1, agx:298.8 } }
  ];

  var X0 = 10, X1 = 1000, Y0 = 1, Y1 = 40;
  var W = 440, H = 340, ML = 40, MR = 8, MT = 10, MB = 38;
  var NS = 'http://www.w3.org/2000/svg';

  function lx(v) { return ML + Math.log(v / X0) / Math.log(X1 / X0) * (W - ML - MR); }
  function ly(v) { return H - MB - Math.log(v / Y0) / Math.log(Y1 / Y0) * (H - MT - MB); }
  function el(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function tf(v) { return v < 10 ? v.toFixed(2) : v.toFixed(1); }
  function gf(v) { return v < 100 ? v.toFixed(1) : String(Math.round(v)); }
  function decades(a, b) {
    var out = [];
    for (var p = Math.ceil(Math.log10(a)); Math.pow(10, p) <= b; p++) out.push(Math.pow(10, p));
    return out;
  }
  function minors(a, b) {
    var out = [];
    for (var p = Math.floor(Math.log10(a)); Math.pow(10, p) <= b; p++)
      for (var m = 2; m <= 9; m++) { var v = m * Math.pow(10, p); if (v > a && v < b) out.push(v); }
    return out;
  }

  // One point per model with a p50 on this device: intensity, achieved TFLOP/s, share of the roof.
  function points(dev) {
    var out = [];
    MODELS.forEach(function (m) {
      var p50 = m.p50[dev.id];
      if (p50 == null) return;
      var I = m.gflop / m.gb, P = m.gflop / p50, roof = Math.min(dev.peak, dev.bw * I / 1000);
      out.push({ m:m, I:I, P:P, pct:Math.round(P / roof * 100), p50:p50, x:lx(I), y:ly(P) });
    });
    return out;
  }

  function panel(dev) {
    var ridge = dev.peak * 1000 / dev.bw;
    var svg = el('svg', { viewBox:'0 0 ' + W + ' ' + H, class:'rf-svg', role:'img',
      'aria-label':'Roofline for ' + dev.name + ': ' + dev.peak + ' TFLOP/s compute ceiling, ' + dev.bw +
        ' GB/s bandwidth ceiling, ridge at ' + Math.round(ridge) + ' FLOP per byte. The table below lists every point.' });
    var px0 = lx(X0), px1 = lx(X1), py0 = ly(Y0), py1 = ly(Y1);

    svg.appendChild(el('rect', { x:px0, y:py1, width:lx(ridge) - px0, height:py0 - py1, class:'rf-mem' }));
    minors(X0, X1).forEach(function (v) { svg.appendChild(el('line', { x1:lx(v), x2:lx(v), y1:py1, y2:py0, class:'rf-g2' })); });
    minors(Y0, Y1).forEach(function (v) { svg.appendChild(el('line', { x1:px0, x2:px1, y1:ly(v), y2:ly(v), class:'rf-g2' })); });
    decades(X0, X1).forEach(function (v) {
      svg.appendChild(el('line', { x1:lx(v), x2:lx(v), y1:py1, y2:py0, class:'rf-g' }));
      svg.appendChild(el('text', { x:lx(v), y:py0 + 15, class:'rf-tick', 'text-anchor':'middle' }, String(v)));
    });
    decades(Y0, Y1).forEach(function (v) {
      svg.appendChild(el('line', { x1:px0, x2:px1, y1:ly(v), y2:ly(v), class:'rf-g' }));
      svg.appendChild(el('text', { x:px0 - 6, y:ly(v) + 4, class:'rf-tick', 'text-anchor':'end' }, String(v)));
    });
    svg.appendChild(el('text', { x:(px0 + px1) / 2, y:H - 4, class:'rf-axis', 'text-anchor':'middle' },
      'Arithmetic intensity (FLOP/byte)'));
    var ym = (py0 + py1) / 2;
    svg.appendChild(el('text', { x:11, y:ym, class:'rf-axis', 'text-anchor':'middle', transform:'rotate(-90 11 ' + ym + ')' },
      'Achieved TFLOP/s'));
    svg.appendChild(el('text', { x:px0 + 6, y:py1 + 15, class:'rf-reg' }, 'memory-bound'));
    svg.appendChild(el('text', { x:px1 - 6, y:py0 - 8, class:'rf-reg', 'text-anchor':'end' }, 'compute-bound'));

    // The roof: bandwidth slope up to the ridge, then the measured GEMM rate.
    var xs = Math.max(X0, Y0 * 1000 / dev.bw);
    svg.appendChild(el('line', { x1:lx(ridge), x2:lx(ridge), y1:ly(dev.peak), y2:py0, class:'rf-ridge' }));
    svg.appendChild(el('path', { class:'rf-roof',
      d:'M' + lx(xs) + ',' + ly(dev.bw * xs / 1000) + 'L' + lx(ridge) + ',' + ly(dev.peak) + 'L' + px1 + ',' + ly(dev.peak) }));
    svg.appendChild(el('text', { x:lx(ridge) + 6, y:ly(dev.peak) - 6, class:'rf-ceil' }, dev.peak + ' TFLOP/s (BF16 GEMM)'));
    var ang = Math.atan2(ly(dev.peak) - ly(dev.bw * xs / 1000), lx(ridge) - lx(xs)) * 180 / Math.PI;
    var bx = Math.pow(xs, 0.6) * Math.pow(ridge, 0.4), by = dev.bw * bx / 1000, tx = lx(bx), ty = ly(by) - 6;
    svg.appendChild(el('text', { x:tx, y:ty, class:'rf-ceil', 'text-anchor':'middle',
      transform:'rotate(' + ang + ' ' + tx + ' ' + ty + ')' }, dev.bw + ' GB/s'));
    svg.appendChild(el('text', { x:lx(ridge) + 4, y:py0 - 8, class:'rf-reg' }, 'ridge ' + Math.round(ridge)));

    var pts = points(dev), g = el('g', {});
    pts.forEach(function (p) {
      g.appendChild(el('circle', { cx:p.x, cy:p.y, r:4.5, class:'rf-pt' }));
      var hit = el('circle', { cx:p.x, cy:p.y, r:10, class:'rf-hit', tabindex:0,
        'aria-label':p.m.name + ': ' + Math.round(p.I) + ' FLOP per byte, ' + tf(p.P) + ' TFLOP/s, ' + p.pct + '% of the roof' });
      hit.setAttribute('data-i', pts.indexOf(p));
      g.appendChild(hit);
    });
    svg.appendChild(g);
    return { svg:svg, pts:pts, roof:[[lx(xs), ly(dev.bw * xs / 1000)], [lx(ridge), ly(dev.peak)], [px1, ly(dev.peak)]] };
  }

  /* Labels are placed once the SVG is in the page, so their real widths can be measured. Each tries
     spots on rings around its point, nearest first, and takes the cheapest: overlap with points,
     other labels, leaders, the roof, the region labels or the plot's edge costs, and so does a spot
     nearer another point than its own. Beyond the first ring a thin leader joins label and point.
     The most crowded points choose first. */
  var RINGS = [9, 20, 32];

  function dist(p, b) {
    var dx = Math.max(b.x - p.x, 0, p.x - b.x - b.w), dy = Math.max(b.y - p.y, 0, p.y - b.y - b.h);
    return Math.sqrt(dx * dx + dy * dy);
  }
  function overlap(a, b) {
    var w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x), h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
    return w > 0 && h > 0 ? w * h : 0;
  }
  function inside(q, b, pad) { return q[0] > b.x - pad && q[0] < b.x + b.w + pad && q[1] > b.y - pad && q[1] < b.y + b.h + pad; }
  function samples(a, b, step) {
    var n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / step)), out = [];
    for (var k = 0; k <= n; k++) out.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
    return out;
  }
  function nearest(p, b) { return [Math.min(Math.max(p.x, b.x), b.x + b.w), Math.min(Math.max(p.y, b.y), b.y + b.h)]; }

  function placeLabels(r) {
    var svg = r.svg, sb = svg.getBoundingClientRect(), s = sb.width / W;
    var px0 = lx(X0), px1 = lx(X1), py0 = ly(Y0), py1 = ly(Y1);
    var taken = [], leads = [], roof = [];
    svg.querySelectorAll('.rf-reg, .rf-ceil').forEach(function (t) {
      var b = t.getBoundingClientRect();
      taken.push({ x:(b.left - sb.left) / s, y:(b.top - sb.top) / s, w:b.width / s, h:b.height / s });
    });
    r.pts.forEach(function (p) { taken.push({ x:p.x - 6, y:p.y - 6, w:12, h:12 }); });
    for (var i = 0; i + 1 < r.roof.length; i++) roof = roof.concat(samples(r.roof[i], r.roof[i + 1], 3));

    var order = r.pts.filter(function (p) { return !p.m.nolab; });
    order.forEach(function (p) {
      p.crowd = r.pts.filter(function (q) { return Math.hypot(q.x - p.x, q.y - p.y) < 45; }).length;
    });
    order.sort(function (a, b) { return b.crowd - a.crowd; });

    var g = el('g', { class:'rf-labs' });
    svg.appendChild(g);
    order.forEach(function (p) {
      var t = el('text', { class:'rf-lab' }, p.m.short || p.m.name);
      g.appendChild(t);
      var best = null;
      RINGS.forEach(function (rad, ring) {
        for (var a = 0; a < 360; a += 30) {
          var c = Math.cos(a * Math.PI / 180), sn = Math.sin(a * Math.PI / 180);
          var ax = p.x + rad * c, ay = p.y + rad * sn;
          var anchor = c > 0.3 ? 'start' : c < -0.3 ? 'end' : 'middle';
          t.setAttribute('x', ax); t.setAttribute('text-anchor', anchor);
          t.setAttribute('y', anchor !== 'middle' ? ay + 4 : sn < 0 ? ay : ay + 9);
          var bb = t.getBBox(), box = { x:bb.x - 1, y:bb.y, w:bb.width + 2, h:bb.height };
          var cost = ring * 30 + rad + (c < 0 ? 2 : 0);
          taken.forEach(function (q) { cost += overlap(box, q) * 3; });
          roof.forEach(function (q) { if (inside(q, box, 2)) cost += 40; });
          leads.forEach(function (seg) { seg.forEach(function (q) { if (inside(q, box, 1)) cost += 60; }); });
          cost += 6 * (Math.max(0, px0 - box.x) + Math.max(0, box.x + box.w - px1) +
                       Math.max(0, py1 - box.y) + Math.max(0, box.y + box.h - py0)) * box.h;
          var lead = null;
          if (ring > 0) {
            var e = nearest(p, box), d = Math.hypot(e[0] - p.x, e[1] - p.y);
            lead = samples([p.x + (e[0] - p.x) * 6 / d, p.y + (e[1] - p.y) * 6 / d], e, 2);
            r.pts.forEach(function (q) {
              if (q !== p && lead.some(function (u) { return Math.hypot(u[0] - q.x, u[1] - q.y) < 7; })) cost += 300;
            });
            taken.forEach(function (q) { if (lead.some(function (u) { return inside(u, q, 0); })) cost += 150; });
          } else {
            // Without a leader, a label reads as its nearest point's, so no other point may be as near.
            var own = dist(p, box);
            r.pts.forEach(function (q) { if (q !== p && Math.hypot(q.x - p.x, q.y - p.y) > 2 && dist(q, box) < own + 5) cost += 400; });
          }
          if (!best || cost < best.cost) best = { cost:cost, x:t.getAttribute('x'), y:t.getAttribute('y'), anchor:anchor, box:box, lead:lead };
        }
      });
      t.setAttribute('x', best.x); t.setAttribute('y', best.y); t.setAttribute('text-anchor', best.anchor);
      taken.push(best.box);
      if (best.lead) {
        var u = best.lead[0], v = best.lead[best.lead.length - 1];
        g.insertBefore(el('line', { x1:u[0], y1:u[1], x2:v[0], y2:v[1], class:'rf-lead' }), g.firstChild);
        leads.push(best.lead);
      }
    });
  }

  function tipHTML(p, dev) {
    return '<b>' + p.m.name + '</b>' +
      '<span>' + gf(p.m.gflop) + ' GFLOP, ' + p.m.gb.toFixed(2) + ' GB per call' + (p.m.hand ? ' (counted by hand)' : '') + '</span>' +
      '<span>' + Math.round(p.I) + ' FLOP/byte; p50 ' + p.p50 + ' ms on ' + dev.name + '</span>' +
      '<span>' + tf(p.P) + ' TFLOP/s, ' + p.pct + '% of the roof</span>';
  }

  function init() {
    var root = document.getElementById('rf-app');
    if (!root) return;
    DEVICES.forEach(function (dev) {
      var box = document.createElement('div');
      box.className = 'rf-p';
      box.innerHTML = '<div class="rf-t">' + dev.name + '</div>';
      var r = panel(dev), tip = document.createElement('div');
      tip.className = 'rf-tip';
      box.appendChild(r.svg);
      box.appendChild(tip);
      function show(e) {
        var i = e.target.getAttribute && e.target.getAttribute('data-i');
        if (i == null) return;
        var p = r.pts[+i], sb = r.svg.getBoundingClientRect(), bb = box.getBoundingClientRect(), s = sb.width / W;
        tip.innerHTML = tipHTML(p, dev);
        var half = tip.offsetWidth / 2, x = sb.left - bb.left + p.x * s;
        tip.style.left = Math.min(Math.max(x, half), bb.width - half) + 'px';
        tip.style.top = (sb.top - bb.top + p.y * s - 10) + 'px';
        tip.classList.add('on');
      }
      function hide() { tip.classList.remove('on'); }
      r.svg.addEventListener('mouseover', show);
      r.svg.addEventListener('focusin', show);
      r.svg.addEventListener('mouseout', hide);
      r.svg.addEventListener('focusout', hide);
      root.appendChild(box);
      placeLabels(r);
      // Labels are measured in the page's font; place them again once a late web font arrives.
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () {
        var old = r.svg.lastChild;
        if (old && old.getAttribute('class') === 'rf-labs') { r.svg.removeChild(old); placeLabels(r); }
      });
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
