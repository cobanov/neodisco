// neodisco — tek ekran.
//
// İki mod var ve birbirine karışmazlar:
//   compose  seçili orandaki boş plaka. Sayfa hep böyle açılır, çünkü yeni bir görüntü
//            üretmeden önce bilmek istediğin şey onun ne boyda olacağı.
//   view     biten bir işi seyretme. Görüntü kendi oranında durur ve künyesi sağda.
// Oran düğmesi yalnız SIRADAKİ üretimin boyunu değiştirir ve seni compose'a döndürür;
// ekranda duran bir görüntünün biçimine asla dokunmaz.
//
// Ayarların geri kalanı Disco'nun kendi varsayılanlarında sabit. Bu sayfanın işi ayar
// yapmak değil, resmi göstermek; ama ne kullanıldığı künyede yazılı.

const $ = (id) => document.getElementById(id);
const root = document.documentElement;

const FIXED = {
  image_size: 512, steps: 250, skip_steps: 10, eta: 0.8, clamp_max: 0.05,
  clip_scale: 5000, range_scale: 150, tv_scale: 0, sat_scale: 0, cutn_batches: 4,
  cut_overview: '[12]*400+[4]*600', cut_innercut: '[4]*400+[12]*600',
  cut_icgray_p: '[0.2]*400+[0]*600', inner_size_pow: 1,
  clip_models: ['ViTB32', 'ViTB16', 'RN50'], use_secondary: true, seed: -1,
};

const CLIP_NAMES = {
  ViTB32: 'ViT-B/32', ViTB16: 'ViT-B/16', ViTL14: 'ViT-L/14',
  RN50: 'RN50', RN101: 'RN101', RN50x4: 'RN50x4', RN50x16: 'RN50x16', RN50x64: 'RN50x64',
};

// Dönemin kalıbı: konu + sanatçı isimleri + trending on artstation. İsimler o zaman
// işin yarısını taşıyordu.
const EXAMPLES = [
  'A colossal derelict starship drifting past a gas giant, galactic soldiers on the hull, by greg rutkowski and john berkey and thomas kinkade, Trending on artstation.',
  'An enormous war fleet emerging from hyperspace above a ringed planet, epic scale, by ralph mcquarrie and greg rutkowski and john harris, matte painting, Trending on artstation.',
  'A titanic space station orbiting a dying star with tiny fighters swarming its spine, by john berkey and syd mead and thomas kinkade, cinematic, Trending on artstation.',
  'An ancient alien megastructure rising above a storm ocean, lightning between its towers, by zdzislaw beksinski and greg rutkowski, dramatic, Trending on artstation.',
  'A dreadnought breaking through the cloud layer above a burning city, by john harris and ralph mcquarrie and greg rutkowski, epic, Trending on artstation.',
  'The cathedral-sized engines of a generation ship, crew silhouettes against the glow, volumetric light, by john berkey and thomas kinkade, Trending on artstation.',
  'A black hole devouring a shattered moon, warships silhouetted against the accretion disk, by chesley bonestell and greg rutkowski and john harris, Trending on artstation.',
  'A frozen orbital shipyard on an ice world, colossal hulls under construction, by syd mead and simon stalenhag and greg rutkowski, Trending on artstation.',
  'A cathedral of glowing coral grown over a sunken cruiser, shafts of light, by zdzislaw beksinski and thomas kinkade, Trending on artstation.',
  'A lone walker crossing the shadow of an orbital ring at dusk, by simon stalenhag and john harris, Trending on artstation.',
];

/* ── durum ────────────────────────────────────────────────────────────────── */

// next: bir sonraki üretimin ölçüsü, oran düğmeleri bunu değiştirir.
// shown: plakada duran görüntünün ölçüsü. İkisi ayrı tutulduğu için oran düğmesi
// duran görüntüyü hiçbir koşulda bozamaz.
let next = { w: 1280, h: 768 };
let shown = null;
let mode = 'compose';          // compose | busy | view
let live = null;               // o an izlenen iş
let current = null;            // künyesi gösterilen bitmiş iş
let polling = null;

/* ── dil ──────────────────────────────────────────────────────────────────── */

const langBtn = $('lang');
const setLang = (lang) => {
  if (lang === 'tr') root.setAttribute('data-lang', 'tr'); else root.removeAttribute('data-lang');
  root.lang = lang;
  langBtn.textContent = lang === 'tr' ? 'EN' : 'TR';
  try { localStorage.setItem('lang', lang); } catch (e) {}
};
const t = (en, tr) => (root.lang === 'tr' ? tr : en);
setLang(root.getAttribute('data-lang') === 'tr' ? 'tr' : 'en');
// render() burada cagrilmaz: bu satir modulun tepesinde kosuyor ve render'in kullandigi
// yardimcilar (fmtSize, showRecordPending) henuz tanimlanmadi. Ilk boyama en altta,
// toCompose() ile yapiliyor; dil degisiminde de oradan tazeleniyor.
langBtn.addEventListener('click', () => { setLang(root.lang === 'tr' ? 'en' : 'tr'); render(); });

/* ── plaka ölçüsü ─────────────────────────────────────────────────────────── */

// Kullanılabilir kutuyu ölç, orana sığan en büyüğünü plakaya ver. Hayalet de görüntü de
// aynı hücreyi doldurduğu için biri diğerinin yerine birebir oturur. Yüzdeli max-height
// burada iş görmüyor: yüksekliği auto olan bir kapsayıcıda yüzde çözülemez.
const fitPlate = () => {
  const box = $('plate-wrap');
  const size = shown || next;
  const readout = $('readout').offsetHeight + 12;
  const availW = box.clientWidth;
  const availH = box.clientHeight - readout - ($('err').hidden ? 0 : $('err').offsetHeight + 12);
  if (availW <= 0 || availH <= 0) return;
  const scale = Math.min(availW / size.w, availH / size.h, 1);
  const plate = $('plate');
  const w = Math.round(size.w * scale) + 'px';
  const h = Math.round(size.h * scale) + 'px';
  if (plate.style.width !== w) plate.style.width = w;
  if (plate.style.height !== h) plate.style.height = h;
};

if (window.ResizeObserver) new ResizeObserver(fitPlate).observe($('plate-wrap'));
addEventListener('resize', fitPlate);

/* ── künye ────────────────────────────────────────────────────────────────── */

const fmtClock = (sec) => {
  const n = Math.max(0, Math.round(sec));
  return n >= 60 ? `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}` : `${n}s`;
};
const fmtSize = (w, h) => `${w} × ${h}`;

const rows = (dl, pairs) => {
  dl.textContent = '';
  for (const [k, v, strong] of pairs) {
    if (v === null || v === undefined || v === '') continue;
    const dt = document.createElement('dt');
    dt.textContent = k;
    const dd = document.createElement('dd');
    if (strong) dd.className = 'strong';
    dd.textContent = v;
    dl.append(dt, dd);
  }
};

const clipList = (names) => (names || FIXED.clip_models).map((n) => CLIP_NAMES[n] || n).join('\n');

// Künye her zaman dolu: compose'da sıradaki üretimin ayarları, üretim sırasında canlı
// sayılar, bitince diskteki settings dosyası. Boş panel diye bir hâl yok.
function paintRecord(cfg) {
  const s = cfg || {};
  // CLIP, Clamp, Eta, Cutn, Overview, Inner çevrilmiyor: bunlar Disco'nun settings
  // dosyasındaki anahtar adları, Türkçeleştirmek tanınmalarını bozar.
  const tech = [
    [t('Model', 'Model'), `${s.image_size || FIXED.image_size} uncond`],
    [t('Secondary', 'İkincil'), (s.use_secondary ?? FIXED.use_secondary) ? t('on', 'açık') : t('off', 'kapalı')],
    ['CLIP', clipList(s.clip_models)],
    [t('Scale', 'Ölçek'), s.clip_scale ?? FIXED.clip_scale],
    [t('Range', 'Aralık'), s.range_scale ?? FIXED.range_scale],
    ['Clamp', s.clamp_max ?? FIXED.clamp_max],
    ['Eta', s.eta ?? FIXED.eta],
    ['Cutn', s.cutn_batches ?? FIXED.cutn_batches],
    ['Overview', s.cut_overview || FIXED.cut_overview],
    ['Inner', s.cut_innercut || FIXED.cut_innercut],
  ];
  rows($('rec-tech'), tech);
}

function showRecordFor(job, cfg) {
  const steps = (cfg?.steps ?? FIXED.steps) - (cfg?.skip_steps ?? FIXED.skip_steps);
  $('rec-title').textContent = t('Record', 'Künye');
  $('rec-id').textContent = job.id.slice(0, 8);
  const p = $('rec-prompt');
  p.classList.remove('empty');
  p.textContent = (cfg?.prompts || []).join('\n') || t('no prompt stored', 'prompt kaydı yok');
  rows($('rec-main'), [
    [t('Size', 'Ölçü'), fmtSize(job.width, job.height), true],
    [t('Seed', 'Tohum'), String(job.seed || cfg?.seed || '-'), true],
    [t('Steps', 'Adım'), String(steps)],
    [t('Time', 'Süre'), job.elapsed > 0 ? fmtClock(job.elapsed) : '-'],
  ]);
  paintRecord(cfg);
  $('rec-actions').hidden = false;
  $('rec-png').href = `/api/result/${job.id}.png`;
  $('rec-png').setAttribute('download', `neodisco-${job.id.slice(0, 8)}.png`);
}

function showRecordPending() {
  $('rec-title').textContent = t('Next render', 'Sıradaki üretim');
  $('rec-id').textContent = '';
  const p = $('rec-prompt');
  p.classList.add('empty');
  // Bu paragraf kunyenin prompt alani; compose'da henuz bir prompt yok, o yuzden
  // bos halini yaziyor. "Henuz uretim yok" demek yanlisti: seritte gecmis durabilir.
  p.textContent = t('Waiting for a prompt.', 'Bir prompt bekliyor.');
  rows($('rec-main'), [
    [t('Size', 'Ölçü'), fmtSize(next.w, next.h), true],
    [t('Seed', 'Tohum'), t('random', 'rastgele')],
    [t('Steps', 'Adım'), String(FIXED.steps - FIXED.skip_steps)],
  ]);
  paintRecord(null);
  $('rec-actions').hidden = true;
}

function showRecordLive(job, text) {
  $('rec-title').textContent = job.state === 'queued' ? t('In queue', 'Sırada') : t('Rendering', 'Üretiliyor');
  $('rec-id').textContent = job.id.slice(0, 8);
  const p = $('rec-prompt');
  p.classList.remove('empty');
  p.textContent = text;
  rows($('rec-main'), [
    [t('Size', 'Ölçü'), fmtSize(job.width, job.height), true],
    [t('Seed', 'Tohum'), String(job.seed || '-'), true],
    [t('Steps', 'Adım'), `${job.step} / ${job.total}`],
    [t('Time', 'Süre'), job.elapsed > 0 ? fmtClock(job.elapsed) : '-'],
  ]);
  paintRecord(null);
  $('rec-actions').hidden = true;
}

/* ── plakanın altyazısı ───────────────────────────────────────────────────── */

function readout(pairs) {
  const el = $('readout');
  el.textContent = '';
  for (const [k, v] of pairs) {
    const wrap = document.createElement('span');
    if (k) {
      const label = document.createElement('span');
      label.className = 'label';
      label.textContent = k + ' ';
      wrap.append(label);
    }
    wrap.append(document.createTextNode(v));
    el.append(wrap);
  }
}

// Dil değişince ekranda yazan her şeyi tazele.
function render() {
  // Altyazi yalnizca CANLI sayilari tasir: adim ve kalan sure. Compose'da olcu zaten
  // plakanin ortasinda, view'da kunyede yaziyor; ucuncu bir kere yazmak gurultu olurdu.
  // Satir her modda yer kapliyor ki uretim baslayinca plaka sicramasin.
  if (mode === 'compose') {
    readout([]);
    showRecordPending();
  } else if (mode === 'view' && current) {
    readout([]);
    showRecordFor(current.job, current.cfg);
  }
  fitPlate();
}

/* ── modlar ───────────────────────────────────────────────────────────────── */

function clearPeek() {
  const peek = $('peek');
  peek.classList.remove('on');
  peek.removeAttribute('src');
}

function dropShot() {
  const old = $('plate').querySelector('img.shot');
  if (old) old.remove();
}

// Boş plakaya dön: seçili oran, hedef ölçü yazılı, künyede sıradaki üretimin ayarları.
function toCompose() {
  mode = 'compose';
  shown = null;
  current = null;
  dropShot();
  clearPeek();
  $('ghost').hidden = false;
  $('ghost-dim').innerHTML = `${next.w} &times; ${next.h}`;
  [...$('rail').children].forEach((b) => b.classList.remove('on'));
  render();
}

// Biten bir işi göster. Görüntü kendi ölçüsünde durur, künye o işin dosyasından gelir.
async function toView(job, button) {
  mode = 'view';
  shown = { w: job.width, h: job.height };
  const cfg = await fetch(`/api/result/${job.id}.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  current = { job, cfg };
  fitPlate();

  const img = new Image();
  img.className = 'shot';
  img.alt = t('Generated image', 'Üretilen görüntü');
  img.addEventListener('load', () => {
    $('ghost').hidden = true;
    clearPeek();
    dropShot();
    $('plate').appendChild(img);
    requestAnimationFrame(() => img.classList.add('in'));
  });
  img.src = `/api/result/${job.id}.png?t=${job.finished || ''}`;

  [...$('rail').children].forEach((b) => b.classList.remove('on'));
  if (button) button.classList.add('on');
  render();
}

/* ── oranlar ──────────────────────────────────────────────────────────────── */

const ratios = $('ratios');
ratios.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b || b.disabled) return;
  [...ratios.children].forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  next = { w: Number(b.dataset.w), h: Number(b.dataset.h) };
  toCompose();
});

const lockRatios = (on) => [...ratios.children].forEach((b) => { b.disabled = on; });

/* ── prompt ───────────────────────────────────────────────────────────────── */

const prompt = $('prompt');
const grow = () => { prompt.style.height = 'auto'; prompt.style.height = Math.min(prompt.scrollHeight, 132) + 'px'; };
prompt.addEventListener('input', grow);
prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('form').requestSubmit(); }
});

$('ex').addEventListener('click', () => {
  let pick = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
  if (pick === prompt.value) pick = EXAMPLES[(EXAMPLES.indexOf(pick) + 1) % EXAMPLES.length];
  prompt.value = pick;
  grow();
  prompt.focus();
});

const showError = (msg) => {
  const box = $('err');
  box.textContent = msg || '';
  box.hidden = !msg;
  fitPlate();
};

/* ── şerit ────────────────────────────────────────────────────────────────── */

const RAIL_MAX = 24;

async function loadRail(activeId) {
  const jobs = await fetch('/api/jobs').then((r) => r.json()).catch(() => null);
  if (!jobs) return [];
  const done = jobs.filter((j) => j.state === 'done' && j.width && j.height).slice(0, RAIL_MAX);
  const rail = $('rail');
  rail.textContent = '';
  root.style.setProperty('--rail-w', done.length ? '112px' : '0px');
  rail.hidden = done.length === 0;
  for (const j of done) {
    const b = document.createElement('button');
    b.type = 'button';
    b.title = fmtSize(j.width, j.height);
    if (j.id === activeId) b.classList.add('on');
    const im = new Image();
    im.src = `/api/thumb/${j.id}.jpg`;
    im.alt = '';
    im.loading = 'lazy';
    b.appendChild(im);
    b.addEventListener('click', () => toView(j, b));
    rail.appendChild(b);
  }
  fitPlate();
  return done;
}

/* ── üretim ───────────────────────────────────────────────────────────────── */

// Bulanıklık ilerlemeyle çözülüyor. Üstel eğri, ilk karelerde yüksek tutup sona doğru
// hızla sıfıra iniyor: erken adımlarda zaten okunacak bir şey yok.
const peekBlur = (p) => 30 * Math.pow(1 - Math.min(Math.max(p, 0), 1), 1.7);

const setPeek = (id, n, p) => {
  const peek = $('peek');
  const img = new Image();
  img.onload = () => {
    peek.src = img.src;
    peek.style.filter = `blur(${peekBlur(p).toFixed(1)}px)`;
    peek.classList.add('on');
    $('ghost').hidden = true;
  };
  img.src = `/api/preview/${id}.jpg?n=${n}`;
};

function watch(id, text) {
  clearInterval(polling);
  let seenPreview = 0;
  polling = setInterval(async () => {
    // Ağ hatası geçicidir, yoklamaya devam edilir. 404 kalıcıdır: sunucu yeniden
    // başlamış ya da iş geçmişten düşmüştür, o kimlik bir daha dönmez. Ayırmazsak 404
    // gövdesinde state alanı olmadığı için hiçbir dal tutmuyor ve sayfa son sayılarda
    // donup kalıyor, Üret düğmesi sonsuza kadar kapalı kalıyordu.
    let res;
    try { res = await fetch(`/api/job/${id}`); } catch (err) { return; }
    if (res.status === 404) {
      clearInterval(polling);
      finish();
      showError(t('That job is gone, the server restarted. Render again.',
                  'O iş kayboldu, sunucu yeniden başlamış. Yeniden üret.'));
      toCompose();
      return;
    }
    if (!res.ok) return;
    const j = await res.json().catch(() => null);
    if (!j || !j.state) return;
    live = j;

    if (j.state === 'queued') {
      readout([[t('In queue', 'Sırada'), String(j.position)]]);
      showRecordLive(j, text);
    } else if (j.state === 'running') {
      const p = j.total ? j.step / j.total : 0;
      $('bar').style.width = p * 100 + '%';
      readout([
        [t('Step', 'Adım'), `${j.step} / ${j.total}`],
        [t('Left', 'Kalan'), j.eta > 0 ? fmtClock(j.eta) : '-'],
      ]);
      showRecordLive(j, text);
      if (j.preview && j.preview !== seenPreview) {
        seenPreview = j.preview;
        setPeek(id, j.preview, p);
      } else if ($('peek').classList.contains('on')) {
        $('peek').style.filter = `blur(${peekBlur(p).toFixed(1)}px)`;
      }
    } else if (j.state === 'done') {
      clearInterval(polling);
      $('bar').style.width = '100%';
      finish();
      await toView(j, null);
      loadRail(j.id).then(() => {
        const first = $('rail').firstElementChild;
        if (first) first.classList.add('on');
      });
    } else if (j.state === 'error') {
      clearInterval(polling);
      finish();
      showError(j.error);
      toCompose();
    }
  }, 1200);
}

function finish() {
  live = null;
  $('plate').classList.remove('busy');
  $('go').disabled = false;
  lockRatios(false);
  setTimeout(() => { $('bar').style.width = '0'; }, 900);
}

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  showError('');
  const text = prompt.value.trim() || EXAMPLES[0];
  prompt.value = text;
  grow();

  // Üretim boyunca seçilen oranın boş plakası durur, görüntü onun içinde açılır.
  toCompose();
  mode = 'busy';
  $('go').disabled = true;
  lockRatios(true);
  $('plate').classList.add('busy');
  $('bar').style.width = '0';
  readout([[t('Sending', 'Gönderiliyor'), '']]);

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...FIXED, prompt_text: text, width: next.w, height: next.h }),
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.detail || res.statusText);
    }
    const job = await res.json();
    showRecordLive(job, text);
    watch(job.id, text);
  } catch (err) {
    finish();
    showError(err.message);
    toCompose();
  }
});

/* ── künye eylemleri ──────────────────────────────────────────────────────── */

$('rec-copy').addEventListener('click', async (e) => {
  if (!current) return;
  const btn = e.currentTarget;
  const before = btn.innerHTML;
  try {
    await navigator.clipboard.writeText(JSON.stringify(current.cfg, null, 2));
    btn.textContent = t('Copied', 'Kopyalandı');
  } catch (err) {
    btn.textContent = t('Copy failed', 'Kopyalanmadı');
  }
  setTimeout(() => { btn.innerHTML = before; }, 1600);
});

// Dar ekranda künye bir çekmece.
const setDrawer = (open) => {
  root.setAttribute('data-record', open ? 'open' : 'closed');
  $('rec-toggle').setAttribute('aria-expanded', String(open));
};
$('rec-toggle').addEventListener('click', () => setDrawer(root.getAttribute('data-record') !== 'open'));
// Sahneye dokunmak çekmeceyi kapatır; açık kalıp konsolun üstünde durmasın.
$('stage').addEventListener('pointerdown', () => {
  if (root.getAttribute('data-record') === 'open') setDrawer(false);
});
addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && root.getAttribute('data-record') === 'open') setDrawer(false);
});

/* ── açılış ───────────────────────────────────────────────────────────────── */

// Sayfa boş plakayla açılır. Son üretim şeridin ilk karesinde duruyor, bir tık uzakta;
// ama ekranı o kaplamaz, çünkü buraya yeni bir şey üretmeye gelindi.
toCompose();
loadRail();
