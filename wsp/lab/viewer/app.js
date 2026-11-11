import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ----------------------------------------------------------------- state
const S = {
  mode: "weights",
  gain: 1.0,
  lo: 0,
  hi: 0,
  meta: null,      // {layers, neurons, heads}
  mlp: null,       // Float32Array [layers * neurons]
  head: null,      // Float32Array [layers * heads]
  nEvents: 0,
  selected: null,
};

const MODE_HELP = {
  weights: "‖column‖ of mlp.down_proj per neuron (what each neuron can write); attention heads by ‖o_proj block‖.",
  edits: "‖ΔW‖ against the base model. Dark = untouched. Updates as the analyst edits.",
  activations: "mean |activation| per neuron from the last forward_read that captured mlp_act.",
};

const $ = (id) => document.getElementById(id);

// Live lab: same-origin /viz endpoints. Static replay (GitHub Pages): JSON files under WSP_STATIC.
const STATIC = window.WSP_STATIC || null;
let staticState = null, staticNeurons = null;
const api = {
  async layers(mode) {
    return (await fetch(STATIC ? `${STATIC}layers-${mode}.json` : `viz/layers?mode=${mode}`)).json();
  },
  async state(since) {
    if (!STATIC) return (await fetch(`viz/state?since=${since}`)).json();
    staticState ??= await (await fetch(`${STATIC}state.json`)).json();
    return since > 0 ? { ...staticState, events: [] } : staticState;
  },
  async neuron(layer, idx) {
    if (!STATIC) return (await fetch(`viz/neuron?layer=${layer}&idx=${idx}`)).json();
    staticNeurons ??= await (await fetch(`${STATIC}neurons.json`)).json();
    return staticNeurons[`${layer}-${idx}`] || null;
  },
};

// ----------------------------------------------------------------- scene
const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090d);
scene.fog = new THREE.Fog(0x07090d, 160, 420);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(40, 120, 60);
scene.add(sun);

const GRID_W = 48;
const LAYER_GAP = 3.2;
let geo = null, points = null, heads = null, slabs = null, layout = null;

const pointMat = new THREE.ShaderMaterial({
  uniforms: { scale: { value: 400 } },
  vertexShader: `
    attribute float size;
    attribute vec3 color;
    varying vec3 vColor;
    uniform float scale;
    void main() {
      vColor = color;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size * scale / -mv.z;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    varying vec3 vColor;
    void main() {
      vec2 c = gl_PointCoord - 0.5;
      if (dot(c, c) > 0.25) discard;
      gl_FragColor = vec4(vColor, 1.0);
    }`,
});

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  pointMat.uniforms.scale.value = h / (2 * Math.tan((camera.fov * Math.PI) / 360));
}
addEventListener("resize", resize);

function build(meta) {
  if (points) { scene.remove(points); geo.dispose(); }
  if (heads) { scene.remove(heads); }
  if (slabs) { scene.remove(slabs); }
  const { layers: L, neurons: I, heads: H } = meta;
  const cols = Math.ceil(Math.sqrt(I * 1.7));
  const rows = Math.ceil(I / cols);
  const cell = GRID_W / cols;
  const depth = rows * cell;
  layout = { cols, rows, cell, depth };

  const pos = new Float32Array(L * I * 3);
  for (let l = 0; l < L; l++) {
    for (let i = 0; i < I; i++) {
      const k = (l * I + i) * 3;
      pos[k] = ((i % cols) + 0.5) * cell - GRID_W / 2;
      pos[k + 1] = l * LAYER_GAP;
      pos[k + 2] = (Math.floor(i / cols) + 0.5) * cell - depth / 2;
    }
  }
  geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(L * I * 3), 3));
  geo.setAttribute("size", new THREE.BufferAttribute(new Float32Array(L * I), 1));
  points = new THREE.Points(geo, pointMat);
  scene.add(points);

  // attention heads: a column of cubes to the left of each layer
  if (H > 0) {
    const hs = Math.min(depth / H, 1.6) * 0.8;
    const box = new THREE.BoxGeometry(hs, hs * 0.6, hs);
    heads = new THREE.InstancedMesh(box, new THREE.MeshLambertMaterial(), L * H);
    const m = new THREE.Matrix4();
    for (let l = 0; l < L; l++) {
      for (let h = 0; h < H; h++) {
        m.makeTranslation(-GRID_W / 2 - 3, l * LAYER_GAP, (h + 0.5) * (depth / H) - depth / 2);
        heads.setMatrixAt(l * H + h, m);
        heads.setColorAt(l * H + h, new THREE.Color(0x223044));
      }
    }
    scene.add(heads);
  }

  // faint slab per layer
  slabs = new THREE.Group();
  const slabGeo = new THREE.PlaneGeometry(GRID_W + 8, depth + 2);
  const slabMat = new THREE.MeshBasicMaterial({ color: 0x1a2333, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false });
  for (let l = 0; l < L; l++) {
    const s = new THREE.Mesh(slabGeo, slabMat);
    s.rotation.x = -Math.PI / 2;
    s.position.set(-2, l * LAYER_GAP - 0.05, 0);
    slabs.add(s);
  }
  scene.add(slabs);

  // frame the whole stack: distance from the larger of height and width
  const top = (L - 1) * LAYER_GAP;
  const span = Math.max(top, GRID_W * 1.4);
  controls.target.set(0, top / 2, 0);
  camera.position.set(span * 0.95, top / 2 + span * 0.42, span * 1.1);
  raycaster.params.Points.threshold = cell * 0.6;
}

// -------------------------------------------------------------- coloring
const C = (h) => new THREE.Color(h);
const RAMPS = {
  weights: [C(0x0b1a2e), C(0x1f5f99), C(0x5ac8fa), C(0xeaf6ff)],
  edits: [C(0x5a1e00), C(0xff7a1a), C(0xffd166), C(0xffffff)],
  activations: [C(0x2b0f45), C(0x2a6f8f), C(0x3fbf7f), C(0xf5e663)],
};
const DIM = C(0x1c2532);

function ramp(stops, t) {
  t = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(t), stops.length - 2);
  return stops[i].clone().lerp(stops[i + 1], t - i);
}

function quantiles(arr, qs) {
  const nz = [];
  const step = Math.max(1, Math.floor(arr.length / 60000));
  for (let i = 0; i < arr.length; i += step) if (arr[i] > 0) nz.push(arr[i]);
  if (!nz.length) return qs.map(() => 0);
  nz.sort((a, b) => a - b);
  return qs.map((q) => nz[Math.min(nz.length - 1, Math.floor(nz.length * q))]);
}

// value -> [0, 1] per mode: weights use a percentile stretch (norms are near-uniform),
// activations a log stretch (heavy-tailed), edits a linear scale to the largest edit.
function mapper(arr, mode) {
  if (mode === "edits") {
    const [hi] = quantiles(arr, [1.0]);
    return (v) => (hi > 0 ? Math.sqrt(v / hi) : 0);
  }
  if (mode === "activations") {
    const [mid, hi] = quantiles(arr, [0.5, 0.998]);
    const den = Math.log1p(hi / Math.max(mid, 1e-12)) || 1;
    return (v) => Math.log1p(v / Math.max(mid, 1e-12)) / den;
  }
  const [lo, hi] = quantiles(arr, [0.02, 0.995]);
  return (v) => (v - lo) / Math.max(hi - lo, 1e-12);
}

function recolor() {
  if (!S.meta || !S.mlp) return;
  const { layers: L, neurons: I, heads: H } = S.meta;
  const col = geo.attributes.color.array;
  const size = geo.attributes.size.array;
  const stops = RAMPS[S.mode];
  const map = mapper(S.mlp, S.mode);
  const g = 1 / S.gain;
  const cell = layout.cell;
  for (let l = 0; l < L; l++) {
    const vis = l >= S.lo && l <= S.hi;
    for (let i = 0; i < I; i++) {
      const p = l * I + i;
      const v = S.mlp[p];
      let c, s;
      if (!vis) { s = 0; c = DIM; }
      else if (S.mode !== "weights" && v === 0) { c = DIM; s = cell * 0.45; }
      else {
        const t = Math.pow(Math.min(Math.max(map(v), 0), 1), g);
        c = ramp(stops, t);
        s = S.mode === "edits" ? cell * (3.2 + 2.5 * t) : cell * (0.5 + 0.7 * t);
      }
      if (S.selected === p) { c = C(0xff3b6b); s = cell * 2.4; }
      col[p * 3] = c.r; col[p * 3 + 1] = c.g; col[p * 3 + 2] = c.b;
      size[p] = s;
    }
  }
  geo.attributes.color.needsUpdate = true;
  geo.attributes.size.needsUpdate = true;

  if (heads && S.head) {
    const hmap = mapper(S.head, S.mode);
    const m = new THREE.Matrix4();
    for (let l = 0; l < L; l++) {
      const vis = l >= S.lo && l <= S.hi;
      for (let h = 0; h < H; h++) {
        const v = S.head[l * H + h];
        const t = Math.pow(Math.min(Math.max(hmap(v), 0), 1), g);
        heads.setColorAt(l * H + h, S.mode !== "weights" && v === 0 ? DIM : ramp(stops, t));
        m.makeTranslation(-GRID_W / 2 - 3, l * LAYER_GAP, (h + 0.5) * (layout.depth / H) - layout.depth / 2);
        if (!vis) m.scale(new THREE.Vector3(0, 0, 0));
        heads.setMatrixAt(l * H + h, m);
      }
    }
    heads.instanceColor.needsUpdate = true;
    heads.instanceMatrix.needsUpdate = true;
  }
  $("layer-range").textContent = `showing layers ${S.lo}–${S.hi} of ${L}`;
}

// ------------------------------------------------------------------ data
function f32(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Float32Array(buf.buffer);
}

async function loadLayers() {
  const d = await api.layers(S.mode);
  const first = !S.meta;
  if (first || S.meta.layers !== d.layers || S.meta.neurons !== d.neurons) {
    S.meta = { layers: d.layers, neurons: d.neurons, heads: d.heads };
    build(S.meta);
    S.lo = 0; S.hi = d.layers - 1;
    for (const id of ["lo", "hi"]) { $(id).max = d.layers - 1; }
    $("lo").value = 0; $("hi").value = d.layers - 1;
  }
  S.mlp = f32(d.mlp);
  S.head = f32(d.head);
  S.stale = !!d.busy;   // lab was mid-pass: re-fetch when it frees up
  recolor();
}

function fmtSecs(t) {
  return t < 60 ? `${Math.round(t)}s` : `${Math.floor(t / 60)}m ${Math.round(t % 60)}s`;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

const EDIT_TOOLS = new Set(["apply_patch", "restore", "python"]);

async function poll() {
  try {
    const st = await api.state(S.nEvents);
    $("model").textContent = `${st.model_id} · ${st.session}`;
    const b = st.budget;
    $("budget-text").textContent = `${b.used} / ${b.total}`;
    $("budget-bar").style.width = `${(100 * b.used) / Math.max(1, b.total)}%`;
    $("n-events").textContent = `${st.n_events} calls`;
    $("status").textContent = st.running ? `running ${st.running.tool} · ${fmtSecs(st.running.secs)}` : "";
    if (!st.running && S.stale && !STATIC) await loadLayers();
    const fresh = st.events;
    if (fresh.length) {
      const ol = $("events");
      for (const ev of fresh) {
        const li = document.createElement("li");
        const isErr = typeof ev.result === "string" && ev.result.startsWith('{"error"');
        li.innerHTML = `<span class="tool ${EDIT_TOOLS.has(ev.tool) ? "edit" : ""}">${esc(ev.tool)}</span>
          <span class="muted">${esc(ev.t.slice(11))} · ${ev.secs}s</span>
          ${isErr ? `<span class="err"> error</span>` : ""}
          <span class="args">${esc(ev.args)}</span>`;
        ol.prepend(li);
      }
      while (ol.children.length > 80) ol.lastChild.remove();
      S.nEvents = st.n_events;
      if (S.mode !== "weights" && !STATIC) await loadLayers();
    }
    $("notes").innerHTML = st.notes
      .slice()
      .reverse()
      .map((n) => `<li><span class="kind">${esc(n.kind)}</span>${esc(n.text)}</li>`)
      .join("");
  } catch (e) {
    $("model").textContent = "lab unreachable";
  }
}

// ------------------------------------------------------------- picking
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let downAt = null;

renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const I = S.meta.neurons;
  const hit = raycaster
    .intersectObject(points)
    .find((h) => { const l = Math.floor(h.index / I); return l >= S.lo && l <= S.hi; });
  if (hit) inspect(Math.floor(hit.index / I), hit.index % I, hit.index);
});

async function inspect(layer, idx, p) {
  S.selected = p;
  recolor();
  const d = await api.neuron(layer, idx);
  $("inspect").classList.remove("hidden");
  if (d && d.busy) {
    const r = d.running ? `${d.running.tool} (${fmtSecs(d.running.secs)})` : "a forward pass";
    $("inspect-body").innerHTML = `<p class="small muted">The lab is busy running ${esc(r)}.
      Click the neuron again when it finishes.</p>`;
    return;
  }
  if (!d) {
    $("inspect-body").innerHTML = `<p class="small muted">layer ${layer} · neuron ${idx} is not in this snapshot.
      The replay includes edited neurons and the most active / strongest neurons per layer.
      Run the lab locally to inspect any neuron.</p>`;
    return;
  }
  const toks = (xs, neg) => xs.map(([t, s]) => `<span class="tok ${neg ? "neg" : ""}" title="${s}">${esc(JSON.stringify(t).slice(1, -1))}</span>`).join("");
  $("inspect").classList.remove("hidden");
  $("inspect-body").innerHTML = `
    <table>
      <tr><td>layer · neuron</td><td>${d.layer} · ${d.neuron}</td></tr>
      <tr><td>‖write vector‖</td><td>${d.norm}</td></tr>
      <tr><td>edited</td><td>${d.edited ? `yes · ‖Δ‖ ${d.delta_norm}` : "no"}</td></tr>
      <tr><td>last mean |act|</td><td>${d.last_mean_abs_act ?? "–"}</td></tr>
    </table>
    <div class="small muted" style="margin-top:6px">promotes (logit lens of its down_proj column)</div>
    <div class="toks">${toks(d.promotes)}</div>
    <div class="small muted" style="margin-top:6px">suppresses</div>
    <div class="toks">${toks(d.suppresses, true)}</div>
    <div class="small muted mono" style="margin-top:6px">${esc(d.tensor)}[:, ${d.neuron}]</div>`;
}

// ------------------------------------------------------------------ ui
for (const btn of document.querySelectorAll("#modes button")) {
  btn.addEventListener("click", async () => {
    document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b === btn));
    S.mode = btn.dataset.mode;
    $("mode-help").textContent = MODE_HELP[S.mode];
    await loadLayers();
  });
}
$("mode-help").textContent = MODE_HELP[S.mode];
$("gain").addEventListener("input", (e) => { S.gain = +e.target.value; recolor(); });
for (const id of ["lo", "hi"]) {
  $(id).addEventListener("input", () => {
    let lo = +$("lo").value, hi = +$("hi").value;
    if (lo > hi) [lo, hi] = id === "lo" ? [lo, lo] : [hi, hi];
    $("lo").value = lo; $("hi").value = hi;
    S.lo = lo; S.hi = hi;
    recolor();
  });
}

function hud() {
  if (!S.meta) return;
  const tot = S.meta.layers * S.meta.neurons;
  $("hud").textContent = `${S.meta.layers} layers × ${S.meta.neurons} MLP neurons (${tot.toLocaleString()} points) + ${S.meta.heads} heads/layer · drag to orbit · click a neuron`;
}

// ---------------------------------------------------------------- start
resize();
await poll();          // never blocks: the header and activity feed show up immediately
await loadLayers();
hud();
if (!STATIC) setInterval(poll, 2000);
else $("replay").classList.remove("hidden");
window.wspViewer = { S, inspect, loadLayers, camera, controls,
  setMode: (m) => document.querySelector(`#modes button[data-mode="${m}"]`).click() };
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
