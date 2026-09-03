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

let size = { w: 1280, h: 768 };
const ratios = $('ratios');
ratios.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  [...ratios.children].forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  size = { w: Number(b.dataset.w), h: Number(b.dataset.h) };
  $('m-dim').textContent = `${size.w}×${size.h}`;
});

const prompt = $('prompt');
const grow = () => { prompt.style.height = 'auto'; prompt.style.height = Math.min(prompt.scrollHeight, 96) + 'px'; };
prompt.addEventListener('input', grow);
prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('form').requestSubmit(); }
});

const showError = (msg) => {
  const box = $('err');
  box.textContent = msg || '';
  box.hidden = !msg;
};

let polling = null;

function reveal(id, job) {
  const frame = $('frame');
  const img = new Image();
  img.alt = t('Generated image', 'Üretilen görüntü');
  img.addEventListener('load', () => {
    const ph = $('placeholder');
    if (ph) ph.remove();
    const old = frame.querySelector('img');
    if (old) old.remove();
    frame.appendChild(img);
    requestAnimationFrame(() => { img.classList.add('in'); frame.classList.add('marked'); });
  });
  img.src = `/api/result/${id}.png?t=${Date.now()}`;
  $('m-seed').textContent = job.seed;
  $('m-time').textContent = `${Math.round(job.elapsed)}s`;
}

function watch(id) {
  clearInterval(polling);
  polling = setInterval(async () => {
    const j = await fetch(`/api/job/${id}`).then((r) => r.json()).catch(() => null);
    if (!j) return;
    if (j.state === 'queued') {
      $('state').textContent = t(`queued ${j.position}`, `sırada ${j.position}`);
    } else if (j.state === 'running') {
      $('state').textContent = t('rendering', 'üretiliyor');
      $('bar').style.width = (j.total ? (j.step / j.total) * 100 : 0) + '%';
      $('m-step').textContent = `${j.step}/${j.total}`;
      $('m-time').textContent = `${Math.round(j.elapsed)}s`;
    } else if (j.state === 'done') {
      clearInterval(polling);
      $('state').textContent = t('done', 'bitti');
      $('bar').style.width = '100%';
      $('m-step').textContent = j.total;
      reveal(j.id, j);
      $('go').disabled = false;
      setTimeout(() => { $('bar').style.width = '0'; }, 900);
    } else if (j.state === 'error') {
      clearInterval(polling);
      $('state').textContent = t('error', 'hata');
      $('bar').style.width = '0';
      showError(j.error);
      $('go').disabled = false;
    }
  }, 1200);
}

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  showError('');
  const text = prompt.value.trim() || prompt.placeholder;
  $('go').disabled = true;
  $('state').textContent = t('sending', 'gönderiliyor');
  $('bar').style.width = '0';
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
    $('go').disabled = false;
  }
});

// Son biten işi ekrana koy, sayfa boş açılmasın.
fetch('/api/jobs').then((r) => r.json()).then((jobs) => {
  const last = (jobs || []).find((j) => j.state === 'done');
  if (last) { reveal(last.id, last); $('state').textContent = t('done', 'bitti'); }
}).catch(() => {});
