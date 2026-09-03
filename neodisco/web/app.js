// neodisco arayüzü: form -> /api/generate, sonra iş bitene kadar yoklama.
const $ = (id) => document.getElementById(id);
const root = document.documentElement;

/* dil */
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

/* durum */
const state = $('state'), bar = $('bar'), nums = $('nums'), stage = $('stage');
const errorBox = $('error'), gallery = $('gallery');
const download = $('download'), settingsLink = $('settings_link');
let polling = null;

const setState = (s, label) => { state.dataset.state = s; state.textContent = label || s; };
const showError = (msg) => {
  errorBox.innerHTML = '';
  if (!msg) return;
  const d = document.createElement('div');
  d.className = 'error-box';
  d.textContent = msg;
  errorBox.appendChild(d);
};

const show = (id) => {
  stage.innerHTML = '';
  const img = new Image();
  img.src = `/api/result/${id}.png?t=${Date.now()}`;
  img.alt = t('Generated image', 'Üretilen görüntü');
  stage.appendChild(img);
  download.href = img.src;
  download.download = `neodisco-${id}.png`;
  download.hidden = false;
  settingsLink.href = `/api/result/${id}.json`;
  settingsLink.hidden = false;
  [...gallery.querySelectorAll('button')].forEach((b) => b.setAttribute('aria-current', String(b.dataset.id === id)));
};

async function refreshGallery(currentId) {
  const jobs = await fetch('/api/jobs').then((r) => r.json()).catch(() => []);
  gallery.innerHTML = '';
  jobs.filter((j) => j.state === 'done').slice(0, 18).forEach((j) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.id = j.id;
    b.setAttribute('aria-current', String(j.id === currentId));
    b.title = `${j.seed}`;
    const img = new Image();
    img.src = `/api/result/${j.id}.png`;
    img.alt = '';
    b.appendChild(img);
    b.addEventListener('click', () => show(j.id));
    gallery.appendChild(b);
  });
}

function watch(id) {
  clearInterval(polling);
  polling = setInterval(async () => {
    const j = await fetch(`/api/job/${id}`).then((r) => r.json()).catch(() => null);
    if (!j) return;
    if (j.state === 'queued') {
      setState('queued', t(`queued · ${j.position}`, `sırada · ${j.position}`));
      nums.textContent = '';
    } else if (j.state === 'running') {
      const pct = j.total ? Math.round((j.step / j.total) * 100) : 0;
      setState('running', t('running', 'çalışıyor'));
      bar.style.width = pct + '%';
      nums.textContent = `${j.step}/${j.total} · ${Math.round(j.elapsed)}s`;
    } else if (j.state === 'done') {
      clearInterval(polling);
      setState('done', t('done', 'bitti'));
      bar.style.width = '100%';
      nums.textContent = `${Math.round(j.elapsed)}s · seed ${j.seed}`;
      show(j.id);
      refreshGallery(j.id);
      $('go').disabled = false;
    } else if (j.state === 'error') {
      clearInterval(polling);
      setState('error', t('error', 'hata'));
      bar.style.width = '0';
      showError(j.error);
      $('go').disabled = false;
    }
  }, 1200);
}

async function uploadFile(input) {
  if (!input.files || !input.files[0]) return null;
  const fd = new FormData();
  fd.append('file', input.files[0]);
  const r = await fetch('/api/upload', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(t('upload failed', 'yükleme başarısız'));
  return (await r.json()).path;
}

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  showError('');
  $('go').disabled = true;
  setState('queued', t('submitting', 'gönderiliyor'));
  bar.style.width = '0';
  try {
    const num = (id) => Number($(id).value);
    const payload = {
      prompt_text: $('prompt_text').value,
      width: num('width'), height: num('height'), image_size: num('image_size'),
      steps: num('steps'), skip_steps: num('skip_steps'), seed: num('seed'),
      clamp_max: num('clamp_max'), eta: num('eta'), cutn_batches: num('cutn_batches'),
      clip_scale: num('clip_scale'), range_scale: num('range_scale'),
      tv_scale: num('tv_scale'), sat_scale: num('sat_scale'),
      cut_overview: $('cut_overview').value, cut_innercut: $('cut_innercut').value,
      cut_icgray_p: $('cut_icgray_p').value, inner_size_pow: num('inner_size_pow'),
      init_scale: num('init_scale'),
      use_secondary: $('use_secondary').checked, fp16: $('fp16').checked,
      clip_models: [...document.querySelectorAll('input[name="clip"]:checked')].map((c) => c.value),
    };
    const initPath = await uploadFile($('init_image'));
    if (initPath) payload.init_image = initPath;
    const jsonFile = $('disco_json').files[0];
    if (jsonFile) payload.disco_json = await jsonFile.text();

    const res = await fetch('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || res.statusText);
    }
    watch((await res.json()).id);
  } catch (err) {
    setState('error', t('error', 'hata'));
    showError(err.message);
    $('go').disabled = false;
  }
});

refreshGallery(null);
