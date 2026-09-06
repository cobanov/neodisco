const $ = (id) => document.getElementById(id);

const FIXED = {
  image_size: 512,
  steps: 250,
  skip_steps: 10,
  eta: 0.8,
  clamp_max: 0.05,
  clip_scale: 5000,
  range_scale: 150,
  tv_scale: 0,
  sat_scale: 0,
  cutn_batches: 4,
  cut_overview: "[12]*400+[4]*600",
  cut_innercut: "[4]*400+[12]*600",
  cut_icgray_p: "[0.2]*400+[0]*600",
  inner_size_pow: 1,
  clip_models: ["ViTB32", "ViTB16", "RN50"],
  use_secondary: true,
  seed: -1,
};

const CLIP_NAMES = {
  ViTB32: "ViT-B/32",
  ViTB16: "ViT-B/16",
  ViTL14: "ViT-L/14",
  RN50: "RN50",
  RN101: "RN101",
  RN50x4: "RN50x4",
  RN50x16: "RN50x16",
  RN50x64: "RN50x64",
};

// Dönemin kalıbı: konu + sanatçı isimleri + trending on artstation. İsimler o zaman
// işin yarısını taşıyordu.
const EXAMPLES = [
  "A colossal derelict starship drifting past a gas giant, galactic soldiers on the hull, by greg rutkowski and john berkey and thomas kinkade, Trending on artstation.",
  "An enormous war fleet emerging from hyperspace above a ringed planet, epic scale, by ralph mcquarrie and greg rutkowski and john harris, matte painting, Trending on artstation.",
  "A titanic space station orbiting a dying star with tiny fighters swarming its spine, by john berkey and syd mead and thomas kinkade, cinematic, Trending on artstation.",
  "An ancient alien megastructure rising above a storm ocean, lightning between its towers, by zdzislaw beksinski and greg rutkowski, dramatic, Trending on artstation.",
  "A dreadnought breaking through the cloud layer above a burning city, by john harris and ralph mcquarrie and greg rutkowski, epic, Trending on artstation.",
  "The cathedral-sized engines of a generation ship, crew silhouettes against the glow, volumetric light, by john berkey and thomas kinkade, Trending on artstation.",
  "A black hole devouring a shattered moon, warships silhouetted against the accretion disk, by chesley bonestell and greg rutkowski and john harris, Trending on artstation.",
  "A frozen orbital shipyard on an ice world, colossal hulls under construction, by syd mead and simon stalenhag and greg rutkowski, Trending on artstation.",
  "A cathedral of glowing coral grown over a sunken cruiser, shafts of light, by zdzislaw beksinski and thomas kinkade, Trending on artstation.",
  "A lone walker crossing the shadow of an orbital ring at dusk, by simon stalenhag and john harris, Trending on artstation.",
];

// Composition is deliberately limited to a prompt and a preset frame.
let next = { w: 1280, h: 768 };
let current = null,
  active = null,
  jobs = [],
  viewVersion = 0,
  pollTimer = null;
let busy = false,
  previewVersion = 0,
  connectionLost = false;
const fmtTime = (sec) => {
  const n = Math.max(0, Math.round(sec || 0));
  return n >= 60
    ? `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`
    : `${n}s`;
};
const store = (key, value) => {
  try {
    value === null
      ? localStorage.removeItem(key)
      : localStorage.setItem(key, value);
  } catch {}
};
const recall = (key) => {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
};
const announce = (message) => {
  $("announcement").textContent = message;
};
const error = (message) => {
  $("err").textContent = message;
  $("err").hidden = !message;
};
const grow = () => {
  $("prompt").style.height = "auto";
  $("prompt").style.height =
    Math.min(116, Math.max(56, $("prompt").scrollHeight)) + "px";
};

$("prompt").placeholder = "A world, a feeling, a scene. What do you imagine?";

function fitFrame() {
  const canvas = $("canvas");
  const scale = Math.min(
    canvas.clientWidth / next.w,
    canvas.clientHeight / next.h,
    1,
  );
  if (scale <= 0) return;
  $("frame-preview").style.width = Math.round(next.w * scale) + "px";
  $("frame-preview").style.height = Math.round(next.h * scale) + "px";
}
new ResizeObserver(fitFrame).observe($("canvas"));

function lock(on) {
  busy = on;
  $("waiting").hidden = !on;
  for (const id of ["go", "ex", "new", "prompt"]) $(id).disabled = on;
  for (const b of $("ratios").children) b.disabled = on;
  for (const b of $("rail").children) b.disabled = on;
  $("go-label").textContent = on ? "Creating" : "Create";
  $("canvas").setAttribute("aria-busy", String(on));
}
function compose() {
  viewVersion++;
  previewVersion++;
  current = null;
  $("stage").style.setProperty("--canvas-ratio", next.w / next.h);
  $("empty").hidden = false;
  $("waiting").hidden = true;
  $("image-area").hidden = true;
  $("artwork").removeAttribute("src");
  $("result-actions").hidden = true;
  $("progress").hidden = true;
  fitFrame();
  paintPending();
  paintHistory();
}
function setSize(w, h) {
  next = { w, h };
  $("next-size").textContent = `${w} × ${h}`;
  $("frame-size").textContent = `${w} × ${h}`;
  $("frame-preview").setAttribute("aria-label", `${w} by ${h} pixel frame`);
  fitFrame();
  if (!current && !busy) paintPending();
  for (const b of $("ratios").children)
    b.setAttribute(
      "aria-pressed",
      String(+b.dataset.w === w && +b.dataset.h === h),
    );
}
$("ratios").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || busy) return;
  setSize(+b.dataset.w, +b.dataset.h);
  // Leave the previous result in history and show the next frame immediately.
  compose();
  error("");
});
$("new").addEventListener("click", () => {
  if (busy) return;
  compose();
  error("");
  $("prompt").value = "";
  paintPending();
  grow();
  $("prompt").focus();
});
$("prompt").addEventListener("input", () => {
  grow();
  if (!current && !busy) paintPending();
});
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !e.isComposing) {
    e.preventDefault();
    if (!busy) $("form").requestSubmit();
  }
});
$("ex").addEventListener("click", () => {
  const options = EXAMPLES.filter((x) => x !== $("prompt").value);
  $("prompt").value = options[Math.floor(Math.random() * options.length)];
  grow();
  if (!current) paintPending();
  $("prompt").focus();
});

function rows(id, pairs) {
  const dl = $(id);
  dl.replaceChildren();
  for (const [key, value] of pairs) {
    const dt = document.createElement("dt"),
      dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = String(value ?? "-");
    dl.append(dt, dd);
  }
}
function paintPending(job = null) {
  $("rec-title").textContent = job
    ? job.state === "running"
      ? "Rendering"
      : "In queue"
    : "Next image";
  $("rec-id").textContent = job?.id?.slice(0, 8) || "";
  $("rec-prompt").textContent =
    $("prompt").value.trim() || "Write a prompt to begin.";
  rows("rec-main", [
    ["Size", `${job?.width || next.w} × ${job?.height || next.h}`],
    ["Seed", job?.seed ?? "Random"],
    [
      "Steps",
      job
        ? `${job.step || 0} / ${job.total || 240}`
        : FIXED.steps - FIXED.skip_steps,
    ],
    ["Time", job?.elapsed > 0 ? fmtTime(job.elapsed) : "-"],
  ]);
  rows("rec-tech", [
    ["Model", `${FIXED.image_size} uncond`],
    ["CLIP", FIXED.clip_models.map((n) => CLIP_NAMES[n] || n).join("\n")],
    ["CLIP guidance", FIXED.clip_scale],
    ["Range", FIXED.range_scale],
    ["Eta", FIXED.eta],
    ["Clamp", FIXED.clamp_max],
    ["Cut batches", FIXED.cutn_batches],
  ]);
  $("result-actions").hidden = true;
}

function paintResult() {
  const { job, cfg } = current;
  $("rec-title").textContent = "Image details";
  $("rec-id").textContent = job.id.slice(0, 8);
  $("rec-prompt").textContent =
    cfg?.prompts?.join("\n") || "Prompt unavailable.";
  rows("rec-main", [
    ["Size", `${job.width} × ${job.height}`],
    ["Seed", job.seed ?? cfg?.seed],
    [
      "Steps",
      cfg?.actual_steps ??
        (cfg?.steps ?? FIXED.steps) - (cfg?.skip_steps ?? FIXED.skip_steps),
    ],
    ["Time", job.elapsed > 0 ? fmtTime(job.elapsed) : "-"],
  ]);
  rows(
    "rec-tech",
    cfg
      ? [
          ["Model", `${cfg.image_size ?? 512} uncond`],
          [
            "CLIP",
            (cfg.clip_models || []).map((n) => CLIP_NAMES[n] || n).join("\n"),
          ],
          ["CLIP guidance", cfg.clip_scale],
          ["Range", cfg.range_scale],
          ["Eta", cfg.eta],
          ["Clamp", cfg.clamp_max],
          ["Cut batches", cfg.cutn_batches],
        ]
      : [["Settings", "Unavailable"]],
  );
  $("rec-json").hidden = !cfg;
  $("reuse").disabled = !cfg?.prompts?.length;
  $("rec-json").href = `/api/result/${encodeURIComponent(job.id)}.json`;
  $("rec-json").download = `neodisco-${job.id}.json`;
  $("rec-png").href = `/api/result/${encodeURIComponent(job.id)}.png`;
  $("rec-png").download = `neodisco-${job.id}.png`;
}
async function view(job) {
  if (busy) return;
  $("stage").style.setProperty("--canvas-ratio", job.width / job.height);
  const version = ++viewVersion;
  previewVersion++;
  error("");
  $("progress").hidden = true;
  $("result-actions").hidden = true;
  $("empty").hidden = true;
  $("image-area").hidden = false;
  $("image-loading").hidden = false;
  $("artwork").removeAttribute("src");
  $("rec-title").textContent = "Loading image";
  $("rec-id").textContent = job.id.slice(0, 8);
  $("rec-prompt").textContent = "";
  rows("rec-main", [
    ["Size", `${job.width} × ${job.height}`],
    ["Seed", job.seed],
  ]);
  rows("rec-tech", []);
  const config = fetch(`/api/result/${encodeURIComponent(job.id)}.json`)
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
  const image = new Image();
  const loaded = new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () =>
      reject(new Error("Image could not be loaded. Try selecting it again."));
  });
  image.src = `/api/result/${encodeURIComponent(job.id)}.png`;
  try {
    const [, cfg] = await Promise.all([loaded, config]);
    if (version !== viewVersion) return;
    current = { job, cfg };
    $("artwork").src = image.src;
    $("artwork").alt = cfg?.prompts?.join(" ") || "Generated image";
    $("image-loading").hidden = true;
    $("result-actions").hidden = false;
    paintResult();
    paintHistory();
  } catch (e) {
    if (version !== viewVersion) return;
    compose();
    error(e.message);
  }
}
function paintHistory() {
  const rail = $("rail");
  const scroll = rail.scrollLeft;
  const scrollTop = rail.scrollTop;
  rail.replaceChildren();
  $("history-count").textContent = jobs.length ? String(jobs.length) : "";
  $("history-empty").hidden = jobs.length > 0;
  for (const job of jobs) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "thumb";
    b.disabled = busy;
    b.setAttribute(
      "aria-label",
      `${"View image"} ${job.id.slice(0, 8)}, ${job.width} × ${job.height}`,
    );
    b.setAttribute("aria-pressed", String(current?.job.id === job.id));
    b.title = `${job.width} × ${job.height}`;
    const img = new Image();
    img.src = `/api/thumb/${encodeURIComponent(job.id)}.jpg`;
    img.alt = "";
    img.loading = "lazy";
    b.append(img);
    b.addEventListener("click", () => view(job));
    rail.append(b);
  }
  rail.scrollLeft = scroll;
  rail.scrollTop = scrollTop;
}
async function loadHistory() {
  try {
    const r = await fetch("/api/jobs");
    if (!r.ok) throw new Error();
    const all = await r.json();
    if (!Array.isArray(all)) throw new Error();
    jobs = all
      .filter((j) => j.state === "done" && j.width && j.height)
      .slice(0, 24);
    paintHistory();
    $("refresh").textContent = "Refresh";
  } catch {
    $("refresh").textContent = "Retry history";
  }
}
$("refresh").addEventListener("click", loadHistory);
$("reuse").addEventListener("click", () => {
  if (!current?.cfg?.prompts) return;
  const { cfg, job } = current;
  // Reuse text and supported framing, never silently reapply hidden historical settings.
  $("prompt").value = cfg.prompts
    .map((p, i) =>
      cfg.weights?.[i] != null && cfg.weights[i] !== 1
        ? `${p}:${cfg.weights[i]}`
        : p,
    )
    .join("\n");
  if (
    [...$("ratios").children].some(
      (b) => +b.dataset.w === job.width && +b.dataset.h === job.height,
    )
  )
    setSize(job.width, job.height);
  grow();
  $("prompt").focus();
});

function paintProgress(job) {
  $("stage").style.setProperty(
    "--canvas-ratio",
    (job.width || next.w) / (job.height || next.h),
  );
  paintPending(job);
  $("progress").hidden = false;
  const running = job.state === "running";
  $("progress-title").textContent = connectionLost
    ? "Reconnecting…"
    : running
      ? "Taking shape"
      : "In the queue";
  $("bar").max = job.total || 240;
  $("bar").value = job.step || 0;
  $("progress-step").textContent = running
    ? `${job.step || 0} / ${job.total || 240}`
    : `#${job.position || 1}`;
  $("progress-time").textContent =
    job.eta > 0 ? `${fmtTime(job.eta)} ${"left"}` : "Please wait";
}
function preview(job) {
  const version = ++previewVersion;
  const id = job.id;
  const img = new Image();
  img.onload = () => {
    if (!busy || active?.id !== id || version !== previewVersion) return;
    $("empty").hidden = true;
    $("waiting").hidden = true;
    $("image-area").hidden = false;
    $("image-loading").hidden = true;
    $("artwork").src = img.src;
    $("artwork").alt = "Image taking shape";
  };
  img.src = `/api/preview/${encodeURIComponent(id)}.jpg?n=${job.preview}`;
}
function finish() {
  clearTimeout(pollTimer);
  active = null;
  previewVersion++;
  store("neodisco.active", null);
  lock(false);
  $("progress").hidden = true;
}
async function watch(id) {
  let lastPreview = 0;
  async function poll() {
    if (!busy || active?.id !== id) return;
    try {
      const res = await fetch(`/api/job/${encodeURIComponent(id)}`);
      if (res.status === 404) {
        finish();
        compose();
        error("This job is no longer available. You can create it again.");
        return;
      }
      if (!res.ok) throw new Error();
      const j = await res.json();
      if (!["queued", "running", "done", "error"].includes(j.state))
        throw new Error();
      if (!busy || active?.id !== id) return;
      active = j;
      connectionLost = false;
      if (j.state === "done") {
        finish();
        await view(j);
        await loadHistory();
        announce("Your image is ready.");
        return;
      }
      if (j.state === "error") {
        finish();
        compose();
        error(j.error || "Generation failed. Try again.");
        return;
      }
      paintProgress(j);
      if (j.preview && j.preview !== lastPreview) {
        lastPreview = j.preview;
        preview(j);
      }
    } catch {
      connectionLost = true;
      if (active) paintProgress(active);
    }
    pollTimer = setTimeout(poll, 1200);
  }
  await poll();
}
$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy) return;
  const text = $("prompt").value.trim();
  if (!text) {
    $("prompt").focus();
    return;
  }
  error("");
  compose();
  lock(true);
  $("empty").hidden = true;
  $("progress").hidden = false;
  $("progress-title").textContent = "Starting…";
  $("progress-step").textContent = "";
  $("progress-time").textContent = "";
  $("bar").value = 0;
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...FIXED,
        prompt_text: text,
        width: next.w,
        height: next.h,
      }),
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(
        typeof d.detail === "string"
          ? d.detail
          : JSON.stringify(d.detail || res.statusText),
      );
    }
    const j = await res.json();
    if (!j.id) throw new Error("The server did not return a job.");
    active = j;
    store(
      "neodisco.active",
      JSON.stringify({ id: j.id, text, width: next.w, height: next.h }),
    );
    paintProgress(j);
    announce("Creating your image.");
    watch(j.id);
  } catch (e) {
    finish();
    compose();
    error(e.message || "Could not connect. Try again.");
  }
});
compose();
grow();
loadHistory();
// Restore only an unfinished job on this browser; do not carry a previous visitor's draft.
try {
  const saved = JSON.parse(recall("neodisco.active") || "null");
  if (saved?.id) {
    $("prompt").value = saved.text || "";
    if (saved.width && saved.height) setSize(saved.width, saved.height);
    grow();
    lock(true);
    $("empty").hidden = true;
    active = { id: saved.id, state: "queued" };
    paintProgress(active);
    watch(saved.id);
  }
} catch {
  store("neodisco.active", null);
}
