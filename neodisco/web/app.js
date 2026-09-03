// Tek ekran: oran seç, prompt yaz, üret. Geri kalan her ayar Disco'nun kendi
// varsayılanlarında sabit, çünkü bu sayfanın işi ayar yapmak değil resmi göstermek.
const $ = (id) => document.getElementById(id);
const root = document.documentElement;

const FIXED = {
  image_size: 512, steps: 250, skip_steps: 10, eta: 0.8, clamp_max: 0.05,
  clip_scale: 5000, range_scale: 150, tv_scale: 0, sat_scale: 0, cutn_batches: 4,
  cut_overview: '[12]*400+[4]*600', cut_innercut: '[4]*400+[12]*600',
  cut_icgray_p: '[0.2]*400+[0]*600', inner_size_pow: 1,
  clip_models: ['ViTB32', 'ViTB16', 'RN50'], use_secondary: true, seed: -1,
};

const langBtn = $('lang');
const setLang = (lang) => {
  if (lang === 'tr') root.setAttribute('data-lang', 'tr'); else root.removeAttribute('data-lang');
  root.lang = lang;
  langBtn.textContent = lang === 'tr' ? 'EN' : 'TR';
  try { localStorage.setItem('lang', lang); } catch (e) {}
};
setLang(root.getAttribute('data-lang') === 'tr' ? 'tr' : 'en');
langBtn.addEventListener('click', () => setLang(root.lang === 'tr' ? 'en' : 'tr'));
const t = (en, tr) => (root.lang === 'tr' ? tr : en);

// Ornek promptlar: donemin kalibi, konu + sanatci isimleri + trending on artstation,
// ikinci satirda ayri bir renk semasi. Isimler o zaman isin yarisini tasiyordu.
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

let size = { w: 1280, h: 768 };
const ghost = $('ghost');
// Konsol prompt uzadikca buyuyor. Sahnenin alt bosluğunu onun gercek yuksekligine
// bagla, yoksa cerceve konsolun uzerine biner.
const fitStage = () => {
  const c = document.querySelector('.console');
  const bottom = Math.round(window.innerHeight - c.getBoundingClientRect().top) + 28;
  document.documentElement.style.setProperty('--stage-bottom', bottom + 'px');
};
const setGhost = () => {
  fitStage();
  // Kullanilabilir alani olcup orana sigan en buyuk kutuyu ver. Goruntu de ayni
  // kutuya oturdugu icin cerceve ile sonuc birebir ayni yerde duruyor.
  const stage = $('stage');
  const cs = getComputedStyle(stage);
  const availW = stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const availH = stage.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  const scale = Math.min(availW / size.w, availH / size.h, 1);
  const frame = $('frame');
  frame.style.width = Math.round(size.w * scale) + 'px';
  frame.style.height = Math.round(size.h * scale) + 'px';
  $('ghost-dim').innerHTML = `${size.w} &times; ${size.h}`;
  $('m-dim').textContent = `${size.w}×${size.h}`;
};
setGhost();
addEventListener('resize', setGhost);

const ratios = $('ratios');
ratios.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  [...ratios.children].forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  size = { w: Number(b.dataset.w), h: Number(b.dataset.h) };
  setGhost();
  // Bir goruntu duruyorsa onu birakip cerceveye donmek yerine goruntu kalir; yeni oran
  // bir sonraki uretimde devreye girer.
  if (!$('frame').querySelector('img:not(#peek)')) ghost.hidden = false;
});

$('dice').addEventListener('click', () => {
  let pick = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
  if (pick === prompt.value) pick = EXAMPLES[(EXAMPLES.indexOf(pick) + 1) % EXAMPLES.length];
  prompt.value = pick;
  grow();
  setGhost();
  prompt.focus();
});

const prompt = $('prompt');
const grow = () => { prompt.style.height = 'auto'; prompt.style.height = Math.min(prompt.scrollHeight, 96) + 'px'; };
prompt.addEventListener('input', () => { grow(); setGhost(); });
if (window.ResizeObserver) new ResizeObserver(() => setGhost()).observe(document.querySelector('.console'));
prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('form').requestSubmit(); }
});

const showError = (msg) => {
  const box = $('err');
  box.textContent = msg || '';
  box.hidden = !msg;
};

let polling = null;

// Kalan sureyi dakika:saniye olarak yaz, 60 saniyenin altinda saniye olarak.
const fmtLeft = (sec) => {
  const n = Math.max(0, Math.round(sec));
  return n >= 60 ? `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}` : `${n}s`;
};

// Bulaniklik ilerlemeyle birlikte cozuluyor. Ustel egri, ilk karelerde yuksek tutup
// sona dogru hizla siniri geciyor: erken adimlarda zaten okunacak bir sey yok.
const peekBlur = (p) => 30 * Math.pow(1 - Math.min(Math.max(p, 0), 1), 1.7);

const setPeek = (id, n, p) => {
  const peek = $('peek');
  const img = new Image();
  img.onload = () => {
    peek.src = img.src;
    peek.style.filter = `blur(${peekBlur(p).toFixed(1)}px)`;
    peek.classList.add('on');
    ghost.hidden = true;
  };
  img.src = `/api/preview/${id}.jpg?n=${n}`;
};

const clearPeek = () => {
  const peek = $('peek');
  peek.classList.remove('on');
  peek.removeAttribute('src');
};

function reveal(id, job) {
  const frame = $('frame');
  const img = new Image();
  img.alt = t('Generated image', 'Üretilen görüntü');
  img.addEventListener('load', () => {
    ghost.hidden = true;
    ghost.classList.remove('busy');
    clearPeek();
    // #peek de bir img: onizleme katmanini silmemek icin disarida birakiliyor.
    const old = frame.querySelector('img:not(#peek)');
    if (old) old.remove();
    frame.appendChild(img);
    requestAnimationFrame(() => { img.classList.add('in'); frame.classList.add('marked'); });
  });
  img.src = `/api/result/${id}.png?t=${Date.now()}`;
  $('m-seed').textContent = job.seed || '-';
  $('m-time').textContent = job.elapsed > 0 ? `${Math.round(job.elapsed)}s` : '-';
}

// Onceki uretimler soldaki seritte. Sunucu gecmisi diskten kuruyor, yani yeniden
// baslatma bunlari silmiyor.
const RAIL_MAX = 24;

function pick(job, button) {
  size = { w: job.width, h: job.height };
  [...ratios.children].forEach((x) => x.setAttribute('aria-pressed',
    String(Number(x.dataset.w) === job.width && Number(x.dataset.h) === job.height)));
  setGhost();
  clearPeek();
  reveal(job.id, job);
  $('state').textContent = t('done', 'bitti');
  [...$('rail').children].forEach((b) => b.classList.remove('on'));
  if (button) button.classList.add('on');
}

async function loadRail(activeId) {
  const jobs = await fetch('/api/jobs').then((r) => r.json()).catch(() => null);
  if (!jobs) return [];
  const done = jobs.filter((j) => j.state === 'done' && j.width && j.height).slice(0, RAIL_MAX);
  const rail = $('rail');
  rail.textContent = '';
  document.documentElement.style.setProperty('--rail-w', done.length ? '104px' : '0px');
  rail.hidden = done.length === 0;
  for (const j of done) {
    const b = document.createElement('button');
    b.type = 'button';
    b.title = `${j.width}\u00d7${j.height}`;
    if (j.id === activeId) b.classList.add('on');
    const im = new Image();
    im.src = `/api/thumb/${j.id}.jpg`;
    im.alt = '';
    im.loading = 'lazy';
    b.appendChild(im);
    b.addEventListener('click', () => pick(j, b));
    rail.appendChild(b);
  }
  setGhost();
  return done;
}

function watch(id) {
  clearInterval(polling);
  let seenPreview = 0;
  polling = setInterval(async () => {
    // Ag hatasi gecicidir, yoklamaya devam edilir. 404 kalicidir: sunucu yeniden
    // baslamis ya da is gecmisten dusmustur, o kimlik bir daha donmez. Ayirmazsak
    // 404 govdesinde state alani olmadigi icin asagidaki dallarin hicbiri tutmuyor,
    // sayfa son sayilarda donup kaliyor ve Generate sonsuza kadar kapali kaliyordu.
    let res;
    try { res = await fetch(`/api/job/${id}`); } catch (err) { return; }
    if (res.status === 404) {
      clearInterval(polling);
      $('state').textContent = t('lost', 'kayıp');
      $('bar').style.width = '0';
      $('m-eta').textContent = '-';
      clearPeek();
      showError(t('That job is gone, the server restarted. Generate again.',
                  'O iş kayboldu, sunucu yeniden başlamış. Yeniden üret.'));
      ghost.classList.remove('busy');
      $('ghost-note').textContent = t('write a prompt', 'bir prompt yaz');
      $('go').disabled = false;
      return;
    }
    if (!res.ok) return;
    const j = await res.json().catch(() => null);
    if (!j || !j.state) return;
    if (j.state === 'queued') {
      $('state').textContent = t(`queued ${j.position}`, `sırada ${j.position}`);
      $('ghost-note').textContent = t(`queued ${j.position}`, `sırada ${j.position}`);
    } else if (j.state === 'running') {
      const p = j.total ? j.step / j.total : 0;
      $('state').textContent = t(`rendering ${Math.round(p * 100)}%`,
                                 `üretiliyor %${Math.round(p * 100)}`);
      $('bar').style.width = p * 100 + '%';
      $('m-step').textContent = `${j.step}/${j.total}`;
      $('m-time').textContent = `${Math.round(j.elapsed)}s`;
      $('m-eta').textContent = j.eta > 0 ? fmtLeft(j.eta) : '-';
      // Tohum uretim baslar baslamaz belli; bekletirsek HUD bir onceki isi gosteriyor.
      $('m-seed').textContent = j.seed || '-';
      $('ghost-note').textContent = j.eta > 0
        ? t(`about ${fmtLeft(j.eta)} left`, `yaklaşık ${fmtLeft(j.eta)} kaldı`)
        : t('starting', 'başlıyor');
      if (j.preview && j.preview !== seenPreview) {
        seenPreview = j.preview;
        setPeek(id, j.preview, p);
      } else if ($('peek').classList.contains('on')) {
        $('peek').style.filter = `blur(${peekBlur(p).toFixed(1)}px)`;
      }
    } else if (j.state === 'done') {
      clearInterval(polling);
      $('state').textContent = t('done', 'bitti');
      $('bar').style.width = '100%';
      $('m-eta').textContent = '-';
      $('m-step').textContent = j.total;
      reveal(j.id, j);
      loadRail(j.id);
      $('go').disabled = false;
      setTimeout(() => { $('bar').style.width = '0'; }, 900);
    } else if (j.state === 'error') {
      clearInterval(polling);
      $('state').textContent = t('error', 'hata');
      $('bar').style.width = '0';
      $('m-eta').textContent = '-';
      clearPeek();
      showError(j.error);
      ghost.classList.remove('busy');
      $('go').disabled = false;
    }
  }, 1200);
}

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  showError('');
  const text = prompt.value.trim() || EXAMPLES[0];
  $('go').disabled = true;
  $('state').textContent = t('sending', 'gönderiliyor');
  $('bar').style.width = '0';
  // Uretim boyunca secilen oranin bos cercevesi durur, goruntu onun icine acilir.
  const stale = $('frame').querySelector('img:not(#peek)');
  if (stale) stale.remove();
  clearPeek();
  $('frame').classList.remove('marked');
  setGhost();
  ghost.hidden = false;
  ghost.classList.add('busy');
  $('ghost-note').textContent = t('rendering', 'üretiliyor');
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...FIXED, prompt_text: text, width: size.w, height: size.h }),
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.detail || res.statusText);
    }
    watch((await res.json()).id);
  } catch (err) {
    $('state').textContent = t('error', 'hata');
    showError(err.message);
    ghost.classList.remove('busy');
    $('go').disabled = false;
  }
});

// Son biten işi ekrana koy, sayfa boş açılmasın.
loadRail().then((done) => {
  const last = done[0];
  if (!last) return;
  pick(last, $('rail').firstElementChild);
});
