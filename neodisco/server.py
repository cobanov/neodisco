"""HTTP server for the neodisco web interface.

One GPU means one render at a time, so requests go into a queue and a single worker
thread walks it. The browser polls for progress rather than holding a connection open
for the several minutes a 1280x768 run takes.
"""

import argparse
import gc
import io
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from PIL import Image

from .clip_bank import ClipBank
from .cutouts import MakeCutouts
from .guidance import PromptGuidance
from .backends.pixel import PixelBackend
from . import disco_config
from .runtime import auto_cut_batch, prepare_reference_runtime, resolve_runtime
from .settings import WEB_DEFAULTS, effective_record, normalise_settings

WEB = Path(__file__).parent / 'web'


@dataclass
class Job:
    id: str
    settings: dict
    state: str = 'queued'          # queued | running | done | error
    step: int = 0
    total: int = 0
    started: float = 0.0
    finished: float = 0.0
    error: str = ''
    position: int = 0
    seed: int = 0
    width: int = 0
    height: int = 0
    first_step: float = 0.0        # ilk adimin saati, kalan sure buradan olculuyor
    preview: int = 0               # son yazilan onizlemenin adimi, 0 ise henuz yok

    def public(self):
        d = asdict(self)
        d.pop('settings')
        d['elapsed'] = (self.finished or time.time()) - self.started if self.started else 0
        # Kalan sure adim hizindan. Isin baslangicindan olcmek yaniltiyordu: ilk kosuda
        # agirliklarin yuklenmesi yirmi saniye suruyor ve o sure adim maliyetiymis gibi
        # sayilip tahmini iki katina cikariyordu. Saat ilk adimda baslatiliyor.
        d['eta'] = 0.0
        if self.state == 'running' and self.step > 1 and self.total and self.first_step:
            pace = (time.time() - self.first_step) / (self.step - 1)
            d['eta'] = max(0.0, pace * (self.total - self.step))
        return d


class Runner:
    """Owns the models and the queue. Everything GPU-touching happens on one thread."""

    def __init__(self, weights_dir, out_dir, runtime_overrides=None, start_worker=True):
        self.weights_dir = weights_dir
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self._backends: dict[tuple, PixelBackend] = {}
        self._banks: dict[tuple, ClipBank] = {}
        self.runtime_overrides = dict(runtime_overrides or {})
        self.reference_ready = False
        if self.runtime_overrides.get('deterministic'):
            policy = resolve_runtime(
                self.runtime_overrides.get('device', 'auto'),
                self.runtime_overrides.get('precision', 'auto'))
            prepare_reference_runtime(policy.device)
            self.reference_ready = True
        self._restore()
        if start_worker:
            threading.Thread(target=self._loop, daemon=True).start()

    def _restore(self):
        """Rebuild finished jobs from disk.

        The queue used to live only in memory, so a restart wiped every result the
        browser knew about: the page opened empty even though the PNGs were still
        sitting in the output directory. The settings written next to each render
        carry everything the listing needs.
        """
        found = []
        for meta in self.out_dir.glob('*.json'):
            png = meta.with_suffix('.png')
            if not png.exists():
                continue
            try:
                cfg = json.loads(meta.read_text(encoding='utf-8'))
            except Exception:
                continue
            found.append((png.stat().st_mtime, meta.stem, cfg))
        for mtime, job_id, cfg in sorted(found):
            job = Job(id=job_id, settings=cfg, state='done',
                      total=int(cfg.get('steps', 0)), step=int(cfg.get('steps', 0)),
                      started=mtime, finished=mtime, seed=int(cfg.get('seed', 0)),
                      width=int(cfg.get('width', 0)), height=int(cfg.get('height', 0)))
            self.jobs[job_id] = job
            self.order.append(job_id)

    def submit(self, settings):
        job = Job(id=uuid.uuid4().hex[:12], settings=settings,
                  total=max(1, int(settings['steps']) - int(settings['skip_steps'])),
                  width=int(settings['width']), height=int(settings['height']),
                  seed=int(settings['seed']))
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.q.put(job.id)
        self._renumber()
        return job

    def _renumber(self):
        with self.lock:
            waiting = [j for j in (self.jobs[i] for i in self.order) if j.state == 'queued']
            for n, j in enumerate(waiting, start=1):
                j.position = n

    def _backend(self, settings):
        policy = resolve_runtime(settings['device'], settings['precision'])
        key = (settings['image_size'], settings['width'], settings['height'],
               str(policy.device), policy.precision,
               settings['attention'], settings['compile_mode'], settings['grad_checkpoint'],
               settings['use_secondary'])
        if key not in self._backends:
            self._backends.clear()
            gc.collect()
            if policy.device.type == 'cuda':
                torch.cuda.empty_cache()
            self._backends[key] = PixelBackend(
                PixelBackend.default_path(settings['image_size'], self.weights_dir),
                image_size=settings['image_size'], device=policy.device,
                fp16=policy.model_fp16, use_checkpoint=settings['grad_checkpoint'],
                secondary_path=(PixelBackend.default_secondary_path(self.weights_dir)
                                if settings['use_secondary'] else None),
                fast_attention=settings['attention'] == 'sdpa',
                autocast_dtype=policy.autocast_dtype,
                compile_mode=settings['compile_mode'])
        return self._backends[key], policy

    def _bank(self, names, device):
        key = (str(device), *tuple(names))
        if key not in self._banks:
            self._banks.clear()
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            self._banks[key] = ClipBank(
                [disco_config.CLIP_NAMES[n] for n in names], device=device)
        return self._banks[key]

    def _loop(self):
        while True:
            job_id = self.q.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            job.state, job.started = 'running', time.time()
            self._renumber()
            try:
                self._run(job)
                job.state = 'done'
            except Exception as exc:  # surfaced to the browser as-is
                job.state, job.error = 'error', f'{type(exc).__name__}: {exc}'
            finally:
                job.finished = time.time()
                torch.cuda.empty_cache()

    def _run(self, job):
        s = dict(job.settings)
        backend, policy = self._backend(s)
        s['device'], s['precision'] = str(policy.device), policy.precision
        if s['cut_batch'] == 'auto':
            s['cut_batch'] = auto_cut_batch(policy.device)
        bank = self._bank(s['clip_models'], policy.device)
        cutouts = MakeCutouts(
            bank.cut_size, inner_size_pow=1.0,
            augment=bool(s['augment']))
        guidance = PromptGuidance(
            bank, cutouts, s['prompts'], s['weights'], clip_scale=float(s['clip_scale']),
            tv_scale=float(s['tv_scale']), range_scale=float(s['range_scale']),
            sat_scale=float(s['sat_scale']), clamp_max=float(s['clamp_max']))
        seed = int(s['seed'])
        job.seed = seed
        # Atlanan adimlar hic kosulmuyor; sayaci gercek yineleme sayisina kuruyoruz,
        # yoksa bar 240/250'de "bitti" diyor.
        job.total = max(1, int(s['steps']) - int(s['skip_steps']))

        preview_path = self.out_dir / f'{job.id}_p.jpg'
        last = [0.0]

        def write_preview(n, pred):
            # Saniyede birden sik yazmanin anlami yok: tarayici zaten periyodik yokluyor
            # ve JPEG kodlamasi GPU adimindan calmasin.
            now = time.time()
            if now - last[0] < 2.5 and n != job.total:
                return
            last[0] = now
            try:
                arr = backend.to_uint8(pred.clamp(-1, 1))[0]
                tmp = preview_path.with_suffix('.tmp.jpg')
                Image.fromarray(arr).save(tmp, quality=82)
                os.replace(tmp, preview_path)
                job.preview = n
            except Exception:
                pass       # onizleme kozmetik, uretimi asla dusurmesin

        class Ticker:
            """Stands in for tqdm so the queue can report progress to the browser."""

            def __init__(self, it):
                self.it = it

            def __iter__(self):
                for n, v in enumerate(self.it, start=1):
                    if n == 1:
                        job.first_step = time.time()
                    job.step = n
                    yield v

        pixels = backend.sample(
            guidance=guidance, batch_size=int(s['batch_size']), steps=int(s['steps']), seed=seed,
            width=int(s['width']), height=int(s['height']), eta=float(s['eta']),
            skip_steps=int(s['skip_steps']), cut_overview=s['cut_overview'],
            cut_innercut=s['cut_innercut'], cut_icgray_p=s['cut_icgray_p'],
            cut_ic_pow=s['inner_size_pow'],
            cutn_batches=int(s['cutn_batches']), cut_batch=int(s['cut_batch']),
            clip_denoised=bool(s['clip_denoised']), use_secondary=bool(s['use_secondary']),
            init_image=s.get('init_image') or None,
            init_scale=float(s.get('init_scale') or 0), progress=Ticker,
            preview=write_preview, deterministic=bool(s['deterministic']),
            finite_check=bool(s['finite_check']))
        Image.fromarray(backend.to_uint8(pixels)[0]).save(self.out_dir / f'{job.id}.png')
        out = effective_record(dict(
            s, seed=seed, actual_steps=job.total,
            guidance_nan_steps=list(getattr(backend, "guidance_nan_steps", [])),
            compile_mode_requested=s['compile_mode'],
            compile_mode_effective=backend.effective_compile_mode))
        job.settings = out
        (self.out_dir / f'{job.id}.json').write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


DEFAULTS = WEB_DEFAULTS


def build_app(runner, uploads):
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title='neodisco')

    @app.post('/api/generate')
    async def generate(payload: dict):
        try:
            source = None
            if payload.get('disco_json'):
                raw = json.loads(payload['disco_json'])
                source = disco_config.from_mapping(raw) if 'text_prompts' in raw else raw
            overrides = {k: v for k, v in payload.items() if k in DEFAULTS and v is not None}
            text = (payload.get('prompt_text') or '').strip()
            if text:
                pairs = [disco_config.split_prompt(line.strip()) for line in text.splitlines()
                         if line.strip()]
                overrides['prompts'] = [pair[0] for pair in pairs]
                overrides['weights'] = [pair[1] for pair in pairs]
            overrides.update(runner.runtime_overrides)
            settings = normalise_settings(
                source, overrides, defaults=DEFAULTS, resolve_random_seed=True)
            if settings['batch_size'] != 1:
                raise ValueError('the web API currently supports batch_size=1')
            if settings['deterministic'] and not getattr(runner, 'reference_ready', False):
                raise ValueError('restart neodisco-web with --deterministic for reference mode')
            # Resolve device/precision before queuing, so invalid requests fail as 4xx
            # without loading checkpoints or CLIP towers.
            resolve_runtime(settings['device'], settings['precision'])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return runner.submit(settings).public()

    @app.post('/api/upload')
    async def upload(file: UploadFile = File(...)):
        name = f'{uuid.uuid4().hex}{Path(file.filename or "").suffix or ".png"}'
        path = uploads / name
        path.write_bytes(await file.read())
        return {'path': str(path)}

    @app.get('/api/preview/{job_id}.jpg')
    async def preview(job_id: str):
        path = runner.out_dir / f'{job_id}_p.jpg'
        if not path.exists():
            raise HTTPException(404, 'no preview yet')
        return FileResponse(path, media_type='image/jpeg',
                            headers={'Cache-Control': 'no-store'})

    @app.get('/api/thumb/{job_id}.jpg')
    async def thumb(job_id: str):
        src = runner.out_dir / f'{job_id}.png'
        if not src.exists():
            raise HTTPException(404, 'not ready')
        dst = runner.out_dir / f'{job_id}_t.jpg'
        # Tam boy PNG'yi seride basmak birkac megabayt bosa trafik; kucugu bir kere
        # uretip yaninda tutuyoruz.
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            im = Image.open(src).convert('RGB')
            im.thumbnail((320, 320), Image.LANCZOS)
            im.save(dst, quality=80)
        return FileResponse(dst, media_type='image/jpeg')

    @app.get('/api/jobs')
    async def jobs():
        with runner.lock:
            ids = runner.order[-40:]
            return [runner.jobs[i].public() for i in reversed(ids)]

    @app.get('/api/job/{job_id}')
    async def job(job_id: str):
        j = runner.jobs.get(job_id)
        if not j:
            raise HTTPException(404, 'no such job')
        return j.public()

    @app.get('/api/result/{job_id}.png')
    async def result(job_id: str):
        path = runner.out_dir / f'{job_id}.png'
        if not path.exists():
            raise HTTPException(404, 'not ready')
        return FileResponse(path, media_type='image/png')

    @app.get('/api/result/{job_id}.json')
    async def result_settings(job_id: str):
        path = runner.out_dir / f'{job_id}.json'
        if not path.exists():
            raise HTTPException(404, 'not ready')
        return JSONResponse(json.loads(path.read_text()))

    # index.html'i StaticFiles'tan once yakalayip app.css/app.js baglantilarina dosya
    # mtime'ini damgaliyoruz. Damga olmadan tarayici deploy sonrasi eski stylesheet'i
    # onbellekten servis ediyor: yeni JS ile eski CSS karisiyor ve sayfa bozuk gorunuyor.
    @app.get('/')
    async def index():
        html = (WEB / 'index.html').read_text(encoding='utf-8')
        for asset in ('app.css', 'app.js'):
            stamp = int((WEB / asset).stat().st_mtime)
            html = html.replace(f'"{asset}"', f'"{asset}?v={stamp}"')
        return HTMLResponse(html, headers={'Cache-Control': 'no-cache'})

    app.mount('/', StaticFiles(directory=str(WEB), html=True), name='web')
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/disco')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7870)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--precision', choices=['auto', 'fp32', 'bf16', 'fp16'], default='auto')
    ap.add_argument('--cut-batch', default='auto')
    ap.add_argument('--attention', choices=['original', 'sdpa'], default='sdpa')
    ap.add_argument('--compile', dest='compile_mode',
                    choices=['eager', 'default', 'reduce-overhead', 'max-autotune'],
                    default='eager')
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()

    from .cli import _raise_fd_limit
    _raise_fd_limit()
    uploads = Path(args.out) / 'uploads'
    uploads.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.weights, args.out, runtime_overrides={
        'device': args.device, 'precision': args.precision, 'cut_batch': args.cut_batch,
        'attention': args.attention, 'compile_mode': args.compile_mode,
        'deterministic': args.deterministic})
    import uvicorn
    uvicorn.run(build_app(runner, uploads), host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
