/* Execution-horizon sweep, redrawn in the page's colours from the paper's table (SmolVLA, LIBERO-
   Object, Orin Nano, H = 50, 50 episodes per setting). Success carries 95% Wilson intervals;
   throughput is S/L, inference amortised over the S executed actions, excluding execution and
   transport. The chart is drawn at the container's own pixel width, so its text never scales. */
(function () {
  'use strict';

  // S, successes out of N, Wilson interval (%), median and p95 request latency (ms), S/L (Hz).
  var N = 50, H_PRED = 50;
  var ROWS = [
    { s:1,  k:41, ci:[69.2, 90.2], med:358.4, p95:370.5, hz:2.8 },
    { s:2,  k:43, ci:[73.8, 93.0], med:352.7, p95:363.4, hz:5.7 },
    { s:4,  k:47, ci:[83.8, 97.9], med:356.6, p95:383.2, hz:11.2 },
    { s:8,  k:44, ci:[76.2, 94.4], med:391.7, p95:429.9, hz:20.4 },
    { s:16, k:49, ci:[89.5, 99.6], med:447.0, p95:463.3, hz:35.8 },
    { s:32, k:40, ci:[67.0, 88.8], med:456.7, p95:466.8, hz:70.1 },
    { s:50, k:19, ci:[25.9, 51.8], med:457.0, p95:467.9, hz:109.4 }
  ];
  var HZ_MAX = 120;
  var NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function draw(root, tip) {
    var W = root.clientWidth, H = Math.round(Math.min(330, Math.max(250, W * 0.56)));
    var ML = 44, MR = 46, MT = 10, MB = 42, n = ROWS.length;
    var bw = (W - ML - MR) / n;
    function x(i) { return ML + bw * (i + 0.5); }
    function yl(v) { return H - MB - v / 100 * (H - MT - MB); }
    function yr(v) { return H - MB - v / HZ_MAX * (H - MT - MB); }

    var svg = el('svg', { width:W, height:H, viewBox:'0 0 ' + W + ' ' + H, class:'rf-svg', role:'img',
      'aria-label':'Success and throughput against the replan interval S. Success stays between 80 and 98% ' +
        'up to S = 32 and falls to 38% at S = 50; throughput rises from 2.8 to 109.4 Hz.' });
    var top = yl(100), bot = yl(0), right = W - MR;

    // S = H: the robot runs the whole chunk blind.
    var xb = ML + bw * (n - 1);
    svg.appendChild(el('rect', { x:xb, y:top, width:right - xb, height:bot - top, class:'rf-mem' }));
    svg.appendChild(el('line', { x1:xb, x2:xb, y1:top, y2:bot, class:'rf-ridge' }));
    svg.appendChild(el('text', { x:right - 5, y:bot - 8, class:'rf-reg', 'text-anchor':'end' }, 'S = H'));

    [0, 20, 40, 60, 80, 100].forEach(function (v) {
      svg.appendChild(el('line', { x1:ML, x2:right, y1:yl(v), y2:yl(v), class:v ? 'rf-g2' : 'rf-g' }));
      svg.appendChild(el('text', { x:ML - 6, y:yl(v) + 4, class:'rf-tick', 'text-anchor':'end' }, String(v)));
    });
    [0, 20, 40, 60, 80, 100, 120].forEach(function (v) {
      svg.appendChild(el('text', { x:right + 6, y:yr(v) + 4, class:'rf-tick' }, String(v)));
    });
    ROWS.forEach(function (r, i) {
      svg.appendChild(el('text', { x:x(i), y:bot + 16, class:'rf-tick', 'text-anchor':'middle' }, String(r.s)));
    });
    var ym = (top + bot) / 2;
    svg.appendChild(el('text', { x:11, y:ym, class:'rf-axis', 'text-anchor':'middle', transform:'rotate(-90 11 ' + ym + ')' },
      'Task success (%)'));
    svg.appendChild(el('text', { x:W - 6, y:ym, class:'rf-axis', 'text-anchor':'middle', transform:'rotate(90 ' + (W - 6) + ' ' + ym + ')' },
      'Throughput S/L (Hz)'));
    svg.appendChild(el('text', { x:(ML + right) / 2, y:H - 6, class:'rf-axis', 'text-anchor':'middle' },
      'Replan interval S (actions executed per inference call)'));

    // throughput under success, so the success marks stay on top
    svg.appendChild(el('polyline', { class:'hz-thr', points:ROWS.map(function (r, i) { return x(i) + ',' + yr(r.hz); }).join(' ') }));
    ROWS.forEach(function (r, i) {
      svg.appendChild(el('rect', { x:x(i) - 4, y:yr(r.hz) - 4, width:8, height:8, class:'hz-sq' }));
    });
    ROWS.forEach(function (r, i) {
      var cx = x(i);
      svg.appendChild(el('line', { x1:cx, x2:cx, y1:yl(r.ci[0]), y2:yl(r.ci[1]), class:'hz-err' }));
      [r.ci[0], r.ci[1]].forEach(function (v) {
        svg.appendChild(el('line', { x1:cx - 4, x2:cx + 4, y1:yl(v), y2:yl(v), class:'hz-err' }));
      });
    });
    svg.appendChild(el('polyline', { class:'hz-suc', points:ROWS.map(function (r, i) { return x(i) + ',' + yl(r.k / N * 100); }).join(' ') }));
    ROWS.forEach(function (r, i) {
      svg.appendChild(el('circle', { cx:x(i), cy:yl(r.k / N * 100), r:4.5, class:'rf-pt' }));
    });

    // one hover column per setting
    ROWS.forEach(function (r, i) {
      var c = el('rect', { x:ML + bw * i, y:top, width:bw, height:bot - top, class:'rf-hit', tabindex:0, 'data-i':i,
        'aria-label':'S = ' + r.s + ': ' + r.k + ' of ' + N + ' succeed, ' + r.hz + ' Hz' });
      svg.appendChild(c);
    });

    function show(e) {
      var i = e.target.getAttribute && e.target.getAttribute('data-i');
      if (i == null) return;
      var r = ROWS[+i];
      tip.innerHTML = '<b>S = ' + r.s + (r.s === H_PRED ? ' (= H)' : '') + '</b>' +
        '<span>' + r.k + '/' + N + ' succeed (' + Math.round(r.k / N * 100) + '%), 95% CI ' + r.ci[0] + '–' + r.ci[1] + '%</span>' +
        '<span>' + r.hz.toFixed(1) + ' Hz; latency median ' + r.med.toFixed(1) + ' ms, p95 ' + r.p95.toFixed(1) + ' ms</span>';
      var half = tip.offsetWidth / 2;
      tip.style.left = Math.min(Math.max(x(+i), half), W - half) + 'px';
      tip.style.top = (Math.min(yl(r.ci[1]), yr(r.hz)) - 8) + 'px';
      tip.classList.add('on');
    }
    function hide() { tip.classList.remove('on'); }
    svg.addEventListener('mouseover', show);
    svg.addEventListener('focusin', show);
    svg.addEventListener('mouseout', hide);
    svg.addEventListener('focusout', hide);
    return svg;
  }

  function init() {
    var root = document.getElementById('hz-app');
    if (!root) return;
    var tip = document.createElement('div'), svg = null, last = 0;
    tip.className = 'rf-tip';
    root.appendChild(tip);
    function render() {
      if (root.clientWidth === last) return;
      last = root.clientWidth;
      if (svg) root.removeChild(svg);
      svg = draw(root, tip);
      root.insertBefore(svg, tip);
    }
    render();
    var t = 0;
    window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(render, 100); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
