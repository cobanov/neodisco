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

// Keep the original Disco structure, with compatible subjects/settings/artists.
// A shuffled deck visits every subject before repeating, without extra UI.
const EXAMPLE_WORLDS = [
  {
    subjects: [
      "A colossal derelict starship",
      "A sprawling orbital shipyard",
      "An ancient alien megastructure",
      "A fleet of interstellar arks",
    ],
    settings: [
      "above a ringed gas giant",
      "at the edge of a luminous nebula",
      "beside a shattered moon",
      "in orbit around a dying star",
    ],
    details: [
      "tiny exploration vessels revealing the immense scale",
      "intricate machinery silhouetted against the void",
      "fragments of ancient technology drifting nearby",
      "glowing structures stretching into the distance",
    ],
    artists: [
      "john berkey",
      "john harris",
      "ralph mcquarrie",
      "chesley bonestell",
      "syd mead",
    ],
  },
  {
    subjects: [
      "A monumental alien observatory",
      "A caravan of towering mechanical walkers",
      "A forgotten city of crystal spires",
      "A vast temple carved into a meteorite",
    ],
    settings: [
      "on a windswept desert planet",
      "beneath the rings of an alien sky",
      "on the rim of a volcanic crater",
      "across a frozen extraterrestrial plain",
    ],
    details: [
      "a lone explorer in the foreground",
      "ancient geometric markings covering the surfaces",
      "distant mountains revealing a monumental scale",
      "delicate mineral formations surrounding the scene",
    ],
    artists: [
      "moebius",
      "roger dean",
      "ralph mcquarrie",
      "john harris",
      "simon stalenhag",
    ],
  },
  {
    subjects: [
      "A cathedral of luminous coral",
      "A forgotten palace of pearl and glass",
      "A colossal sunken ocean liner",
      "An underwater garden of giant anemones",
    ],
    settings: [
      "on the floor of a deep ocean trench",
      "beneath a canopy of drifting jellyfish",
      "among the ruins of a submerged city",
      "inside a vast underwater cavern",
    ],
    details: [
      "schools of tiny fish winding through the scene",
      "delicate sea fans growing over every surface",
      "suspended particles tracing the ocean currents",
      "a small diver revealing the immense scale",
    ],
    artists: [
      "ernst haeckel",
      "ivan aivazovsky",
      "james gurney",
      "roger dean",
      "thomas kinkade",
    ],
  },
  {
    subjects: [
      "A labyrinth of impossible staircases",
      "A floating palace of carved stone",
      "A deserted library with towering arches",
      "A clockwork cathedral with mirrored towers",
    ],
    settings: [
      "above a sea of clouds",
      "at the edge of an endless salt flat",
      "within a dreamlike mountain valley",
      "beside a perfectly still lake",
    ],
    details: [
      "a solitary figure wandering through the scene",
      "long shadows forming intricate geometric patterns",
      "delicate bridges connecting distant structures",
      "weathered sculptures guarding forgotten entrances",
    ],
    artists: [
      "giorgio de chirico",
      "rene magritte",
      "m. c. escher",
      "zdzislaw beksinski",
      "giovanni battista piranesi",
    ],
  },
  {
    subjects: [
      "An ancient forest of towering redwoods",
      "A hidden garden of giant flowers",
      "A ruined sanctuary reclaimed by moss",
      "A village woven into enormous tree roots",
    ],
    settings: [
      "beside a cascading mountain waterfall",
      "in a secluded alpine valley",
      "along the shore of a quiet lake",
      "on a misty island of steep cliffs",
    ],
    details: [
      "a narrow path inviting the viewer into the distance",
      "delicate leaves scattered across weathered stones",
      "tiny birds circling above the canopy",
      "wildflowers emerging from cracks in the stone",
    ],
    artists: [
      "albert bierstadt",
      "caspar david friedrich",
      "ivan shishkin",
      "thomas kinkade",
      "james gurney",
    ],
  },
  {
    subjects: [
      "A retrofuturistic railway terminal",
      "A sprawling city of elevated gardens",
      "A giant abandoned research machine",
      "A hillside settlement of modular towers",
    ],
    settings: [
      "overlooking a rain-soaked metropolis",
      "along a remote northern coastline",
      "at the boundary between a city and a forest",
      "beneath a vast network of suspended bridges",
    ],
    details: [
      "small human silhouettes among monumental structures",
      "intricate cables and walkways connecting the scene",
      "weathered metal contrasting with lush vegetation",
      "distant windows glowing through the atmosphere",
    ],
    artists: [
      "syd mead",
      "simon stalenhag",
      "moebius",
      "hugh ferriss",
      "ralph mcquarrie",
    ],
  },
];
const EXAMPLE_PALETTES = [
  "cobalt blue and warm amber",
  "emerald green and antique gold",
  "dusty rose and slate gray",
  "deep violet and pale peach",
  "burnt orange and muted teal",
  "ivory and charcoal",
  "copper and midnight blue",
  "sage green and soft lavender",
];
const EXAMPLE_LIGHTING = [
  "soft diffused light",
  "dramatic backlighting",
  "a gentle luminous haze",
  "subtle rim lighting and deep shadows",
  "delicate light revealing intricate textures",
];
const EXAMPLE_FINISHES = [
  "a detailed matte painting",
  "a painterly cinematic composition",
  "an atmospheric illustration with intricate detail",
];
const pickExamplePart = (items) =>
  items[Math.floor(Math.random() * items.length)];
function shuffleExamples(items) {
  const shuffled = [...items];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}
let exampleDeck = [];
let previousExampleSubject = null;
function buildExamplePrompt() {
  if (!exampleDeck.length) {
    exampleDeck = shuffleExamples(
      EXAMPLE_WORLDS.flatMap((world) =>
        world.subjects.map((subject) => ({ world, subject })),
      ),
    );
    // Avoid repeating a subject at the boundary between two decks as well.
    const last = exampleDeck.length - 1;
    if (exampleDeck[last].subject === previousExampleSubject) {
      [exampleDeck[0], exampleDeck[last]] = [exampleDeck[last], exampleDeck[0]];
    }
  }
  const { world, subject } = exampleDeck.pop();
  previousExampleSubject = subject;
  const artists = shuffleExamples(world.artists).slice(0, 2).join(" and ");
  return `${subject} ${pickExamplePart(world.settings)}, ${pickExamplePart(world.details)}, ${pickExamplePart(EXAMPLE_LIGHTING)}, ${pickExamplePart(EXAMPLE_PALETTES)} color palette, by ${artists}, ${pickExamplePart(EXAMPLE_FINISHES)}, Trending on artstation.`;
}

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
  if (busy) return;
  $("prompt").value = buildExamplePrompt();
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
