"""HTTP server for the neodisco web interface.

One GPU means one render at a time, so requests go into a queue and a single worker
thread walks it. The browser polls for progress rather than holding a connection open
for the several minutes a 1280x768 run takes.
"""

import argparse
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

    def public(self):
        d = asdict(self)
        d.pop('settings')
        d['elapsed'] = (self.finished or time.time()) - self.started if self.started else 0
        return d


class Runner:
    """Owns the models and the queue. Everything GPU-touching happens on one thread."""

    def __init__(self, weights_dir, out_dir):
        self.weights_dir = weights_dir
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self._backends: dict[tuple, PixelBackend] = {}
        self._banks: dict[tuple, ClipBank] = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, settings):
        job = Job(id=uuid.uuid4().hex[:12], settings=settings, total=int(settings['steps']))
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

    def _backend(self, image_size, fp16):
        key = (image_size, fp16)
        if key not in self._backends:
            self._backends.clear()
            torch.cuda.empty_cache()
            self._backends[key] = PixelBackend(
                PixelBackend.default_path(image_size, self.weights_dir),
                image_size=image_size, fp16=fp16, use_checkpoint=True,
                secondary_path=PixelBackend.default_secondary_path(self.weights_dir),
                autocast_dtype=torch.bfloat16)
        return self._backends[key]

    def _bank(self, names):
        key = tuple(names)
        if key not in self._banks:
            self._banks[key] = ClipBank([disco_config.CLIP_NAMES[n] for n in names])
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
        s = job.settings
        bank = self._bank(s['clip_models'])
        cutouts = MakeCutouts(bank.cut_size, inner_size_pow=float(s['inner_size_pow']))
        guidance = PromptGuidance(
            bank, cutouts, s['prompts'], s['weights'], clip_scale=float(s['clip_scale']),
            tv_scale=float(s['tv_scale']), range_scale=float(s['range_scale']),
            sat_scale=float(s['sat_scale']), clamp_max=float(s['clamp_max']))
        backend = self._backend(int(s['image_size']), bool(s.get('fp16')))
        seed = int(s['seed']) if int(s['seed']) >= 0 else int(torch.randint(0, 2 ** 31, ()))
        job.seed = seed

        class Ticker:
            """Stands in for tqdm so the queue can report progress to the browser."""

            def __init__(self, it):
                self.it = it

            def __iter__(self):
                for n, v in enumerate(self.it, start=1):
                    job.step = n
                    yield v

        pixels = backend.sample(
            guidance=guidance, steps=int(s['steps']), seed=seed,
            width=int(s['width']), height=int(s['height']), eta=float(s['eta']),
            skip_steps=int(s['skip_steps']), cut_overview=s['cut_overview'] or None,
            cut_innercut=s['cut_innercut'] or None, cut_icgray_p=s['cut_icgray_p'] or None,
            cutn_batches=int(s['cutn_batches']), cut_batch=64,
            use_secondary=bool(s['use_secondary']), init_image=s.get('init_image') or None,
            init_scale=float(s.get('init_scale') or 0), progress=Ticker)
        Image.fromarray(backend.to_uint8(pixels)[0]).save(self.out_dir / f'{job.id}.png')
        out = dict(s, seed=seed)
        out.pop('init_image', None)
        (self.out_dir / f'{job.id}.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))


DEFAULTS = dict(prompts=[], weights=[], clip_models=['ViTB32', 'ViTB16', 'RN50'],
                image_size=512, width=1280, height=768, steps=250, skip_steps=10, seed=-1,
                eta=0.8, clamp_max=0.05, clip_scale=5000.0, tv_scale=0.0, range_scale=150.0,
                sat_scale=0.0, cutn_batches=4, cut_overview='[12]*400+[4]*600',
                cut_innercut='[4]*400+[12]*600', cut_icgray_p='[0.2]*400+[0]*600',
                inner_size_pow=1.0, use_secondary=True, fp16=False, init_image=None,
                init_scale=0.0)


def build_app(runner, uploads):
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title='neodisco')

    @app.post('/api/generate')
    async def generate(payload: dict):
        settings = dict(DEFAULTS)
        if payload.get('disco_json'):
            try:
                tmp = uploads / f'{uuid.uuid4().hex}.json'
                tmp.write_text(payload['disco_json'])
                settings.update(disco_config.load(str(tmp)))
                tmp.unlink(missing_ok=True)
            except Exception as exc:
                raise HTTPException(400, f'settings file could not be read: {exc}')
        for k, v in payload.items():
            if k in settings and k != 'prompts' and v is not None:
                settings[k] = v
        text = (payload.get('prompt_text') or '').strip()
        if text:
            prompts, weights = [], []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                p, w = disco_config.split_prompt(line)
                prompts.append(p)
                weights.append(w)
            settings['prompts'], settings['weights'] = prompts, weights
        if not settings['prompts']:
            raise HTTPException(400, 'write at least one prompt')
        for axis in ('width', 'height'):
            if int(settings[axis]) % 64:
                raise HTTPException(400, f'{axis} must be a multiple of 64')
        return runner.submit(settings).public()

    @app.post('/api/upload')
    async def upload(file: UploadFile = File(...)):
        name = f'{uuid.uuid4().hex}{Path(file.filename or "").suffix or ".png"}'
        path = uploads / name
        path.write_bytes(await file.read())
        return {'path': str(path)}

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
    args = ap.parse_args()

    from .cli import _raise_fd_limit
    _raise_fd_limit()
    uploads = Path(args.out) / 'uploads'
    uploads.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.weights, args.out)
    import uvicorn
    uvicorn.run(build_app(runner, uploads), host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
