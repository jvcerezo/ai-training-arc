"""Live web dashboard for an autonomous training run.

Reads the `metrics.jsonl` / `training.log` / `best.pt` / `last.pt` that
``thcm.training.auto`` writes and serves a self-refreshing page: a progress bar +
ETA, status cards, charts (val/train loss, next-concept accuracy, learning rate),
and a live tail of the training log. Pure standard library — no torch, no web
framework — so it runs in its own process and can never disturb training.

    .\\.venv\\Scripts\\python.exe -m thcm.training.dashboard --ckpt-dir checkpoints
    # then open http://localhost:8000

Point ``--ckpt-dir`` at the directory the trainer writes to; the page polls every
few seconds, tracking a live run (or showing the final state of a finished one).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_CKPT_DIR = "checkpoints"


def read_log(ckpt_dir: str, max_lines: int = 400, max_bytes: int = 96_000) -> list[str]:
    """Tail of training.log (reads only the final chunk so it scales to long runs)."""
    path = os.path.join(ckpt_dir, "training.log")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        data = fh.read().decode("utf-8", errors="replace")
    lines = data.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]                       # drop the partial first line
    return lines[-max_lines:]


def read_metrics(ckpt_dir: str) -> dict:
    """Parse metrics + log + checkpoint state into a JSON-serializable snapshot."""
    path = os.path.join(ckpt_dir, "metrics.jsonl")
    records: list[dict] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as fh:   # tolerate a stray BOM
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue                    # tolerate a half-written last line

    def finfo(name: str) -> dict:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            st = os.stat(p)
            return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}
        return {"exists": False}

    def last_with(key: str):
        return next((r[key] for r in reversed(records) if r.get(key) is not None), None)

    evals = [r for r in records if r.get("val_loss") is not None]
    latest = evals[-1] if evals else {}
    first_t = records[0].get("t") if records else None
    last_t = records[-1].get("t") if records else None
    status = {
        "step": last_with("step"),
        "max_steps": last_with("max_steps"),
        "val_loss": latest.get("val_loss"),
        "val_acc": latest.get("val_acc"),
        "train_loss": last_with("train_loss"),
        "lr": last_with("lr"),
        "best_val": min((r["val_loss"] for r in evals), default=None),
        "steps_per_sec": last_with("steps_per_sec"),
        "last_event": records[-1].get("event") if records else None,
        "n_evals": len(evals),
        "started": first_t,
        "updated": last_t,
        "elapsed": (last_t - first_t) if (first_t and last_t) else None,
        "now": time.time(),
    }
    series = {
        "step": [r.get("step") for r in evals],
        "val_loss": [r.get("val_loss") for r in evals],
        "train_loss": [r.get("train_loss") for r in evals],
        "val_acc": [r.get("val_acc") for r in evals],
        "lr": [r.get("lr") for r in evals],
    }
    return {"status": status, "series": series, "log": read_log(ckpt_dir),
            "checkpoints": {"best": finfo("best.pt"), "last": finfo("last.pt")}}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>T-HCM training</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--muted:#8b949e;--fg:#e6edf3;
   --blue:#58a6ff;--green:#3fb950;--amber:#d29922;--red:#f85149;--pink:#f778ba}
 *{box-sizing:border-box}
 body{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
 header{position:sticky;top:0;z-index:5;background:rgba(13,17,23,.9);backdrop-filter:blur(6px);
   border-bottom:1px solid var(--line);padding:14px 24px;display:flex;align-items:center;gap:14px}
 h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.02em}
 .badge{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;text-transform:uppercase;
   letter-spacing:.04em;background:#21262d;color:var(--muted)}
 #dot{width:9px;height:9px;border-radius:50%;background:var(--muted);box-shadow:0 0 0 0 rgba(63,185,80,.5)}
 #dot.live{background:var(--green);animation:pulse 2s infinite}
 @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,185,80,.5)}70%{box-shadow:0 0 0 7px rgba(63,185,80,0)}}
 #fresh{color:var(--muted);font-size:13px;margin-left:auto}
 main{padding:20px 24px;max-width:1400px;margin:0 auto}
 .prog{height:8px;border-radius:6px;background:#21262d;overflow:hidden;margin:4px 0 6px}
 .prog>div{height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .5s}
 .progmeta{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:18px}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
 .card .label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
 .card .value{font-size:23px;font-weight:650;margin-top:5px;font-variant-numeric:tabular-nums}
 .card .sub{font-size:12px;color:var(--muted);margin-top:2px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
 .panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 8px}
 #logwrap{grid-column:1/-1}
 #log{height:300px;overflow:auto;background:#0a0d12;border:1px solid var(--line);border-radius:8px;
   padding:10px 12px;font-family:"Cascadia Code",Consolas,monospace;font-size:12.5px;line-height:1.5}
 #log .l{white-space:pre-wrap;word-break:break-word}
 .c-green{color:var(--green)}.c-blue{color:var(--blue)}.c-amber{color:var(--amber)}
 .c-red{color:var(--red)}.c-muted{color:var(--muted)}
 @media(max-width:880px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header>
 <div id="dot"></div><h1>Tokenless T-HCM</h1>
 <span class="badge" id="state">&mdash;</span>
 <span id="fresh">connecting&hellip;</span>
</header>
<main>
 <div class="prog"><div id="bar"></div></div>
 <div class="progmeta"><span id="progL">step &mdash;</span><span id="progR"></span></div>
 <div class="cards" id="cards"></div>
 <div class="grid">
  <div class="panel"><h2>loss</h2><canvas id="loss" height="150"></canvas></div>
  <div class="panel"><h2>next-concept accuracy</h2><canvas id="acc" height="150"></canvas></div>
  <div class="panel"><h2>learning rate</h2><canvas id="lr" height="150"></canvas></div>
  <div class="panel"><h2>checkpoints</h2><div id="ckpt" style="font-size:13px;line-height:1.9"></div></div>
  <div class="panel" id="logwrap"><h2>training log</h2><div id="log"></div></div>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
const NA='—';
const fmt=(v,d=3)=>v==null?NA:Number(v).toFixed(d);
const sci=v=>v==null?NA:Number(v).toExponential(1);
function dur(s){if(s==null)return NA;s=Math.round(s);const h=s/3600|0,m=(s%3600)/60|0,x=s%60;
 return h?`${h}h ${m}m`:(m?`${m}m ${x}s`:`${x}s`);}
const BADGE={improve:'var(--green)',start:'var(--blue)',stall:'var(--amber)',
 plateau:'var(--amber)',diverge:'var(--red)',converged:'var(--blue)',finish:'var(--muted)'};
const C='#8b949e', G='#21262d';
const base=(title,log)=>({type:'line',data:{labels:[],datasets:[]},options:{responsive:true,
 maintainAspectRatio:false,animation:false,interaction:{intersect:false,mode:'index'},
 plugins:{legend:{labels:{color:C,boxWidth:12}}},
 scales:{x:{ticks:{color:C,maxTicksLimit:8},grid:{color:G}},
  y:{ticks:{color:C},grid:{color:G},...(log?{type:'logarithmic'}:{})}}}});
const lossC=new Chart($('loss'),base('loss')),accC=new Chart($('acc'),base('acc')),
      lrC=new Chart($('lr'),base('lr',true));
const ds=(label,data,c)=>({label,data,borderColor:c,backgroundColor:c,pointRadius:0,borderWidth:2,tension:.25});
function logClass(t){t=t.toLowerCase();
 if(t.includes('improved')||t.includes('saving best'))return'c-green';
 if(t.includes('converged'))return'c-blue';
 if(t.includes('non-finite')||t.includes('warning')||t.includes('error'))return'c-red';
 if(t.includes('plateau'))return'c-amber';
 if(t.includes('no improvement'))return'c-muted';return'';}
async function tick(){
 let d; try{d=await(await fetch('/api/status')).json()}catch(e){$('fresh').textContent='server offline';return}
 const s=d.status;
 $('state').textContent=s.last_event||NA;
 $('state').style.color=BADGE[s.last_event]||'var(--muted)';
 const age=s.updated?(s.now-s.updated):1e9, live=age<30;
 $('dot').className=live?'live':'';
 $('fresh').textContent=s.updated?(live?'live':`idle ${dur(age)}`):'waiting for metrics…';
 // progress + ETA
 const have=s.step!=null, max=s.max_steps;
 const pct=(have&&max)?Math.min(100,100*s.step/max):0;
 $('bar').style.width=pct+'%';
 $('progL').textContent=have?`step ${s.step.toLocaleString()}${max?' / '+max.toLocaleString():''}`:'step '+NA;
 let eta=NA; if(have&&max&&s.steps_per_sec)eta=dur((max-s.step)/s.steps_per_sec);
 $('progR').textContent=(max?pct.toFixed(1)+'%  ·  ':'')+`elapsed ${dur(s.elapsed)}  ·  ETA ${eta}`;
 // cards
 const cards=[['best val',fmt(s.best_val),'lowest held-out loss'],
  ['val loss',fmt(s.val_loss),'last eval'],['val acc',fmt(s.val_acc),'next-concept top-1'],
  ['train loss',fmt(s.train_loss),'latest step'],['learning rate',sci(s.lr),''],
  ['steps/sec',fmt(s.steps_per_sec,2),'throughput'],['evals',s.n_evals??0,''],
  ['elapsed',dur(s.elapsed),'']];
 $('cards').innerHTML=cards.map(c=>`<div class="card"><div class="label">${c[0]}</div>
  <div class="value">${c[1]}</div><div class="sub">${c[2]||''}</div></div>`).join('');
 // charts
 const x=d.series.step;
 lossC.data.labels=x;lossC.data.datasets=[ds('val',d.series.val_loss,'#58a6ff'),ds('train',d.series.train_loss,'#f778ba')];lossC.update();
 accC.data.labels=x;accC.data.datasets=[ds('val acc',d.series.val_acc,'#3fb950')];accC.update();
 lrC.data.labels=x;lrC.data.datasets=[ds('lr',d.series.lr,'#d29922')];lrC.update();
 // checkpoints
 const ck=d.checkpoints,mb=b=>b?(b/1e6).toFixed(1)+' MB':'';
 $('ckpt').innerHTML=['best','last'].map(k=>{const c=ck[k];
  return `<div><b>${k}.pt</b> &mdash; ${c.exists?`saved, ${mb(c.size)}`:'<span class="c-muted">not yet</span>'}</div>`;}).join('');
 // log (auto-scroll when already near the bottom)
 const box=$('log'),atBottom=box.scrollHeight-box.scrollTop-box.clientHeight<40;
 box.innerHTML=(d.log||[]).map(l=>`<div class="l ${logClass(l)}">${l.replace(/[<>&]/g,m=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[m]))}</div>`).join('');
 if(atBottom)box.scrollTop=box.scrollHeight;
}
tick();setInterval(tick,3000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):                     # silence per-request logging
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/api/status"):
            body = json.dumps(read_metrics(_CKPT_DIR)).encode("utf-8")
            self._send(200, body, "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")


def serve(ckpt_dir: str, host: str, port: int) -> None:
    global _CKPT_DIR
    _CKPT_DIR = ckpt_dir
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"dashboard on http://{host}:{port}  (watching {ckpt_dir})  — Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main() -> None:
    p = argparse.ArgumentParser(description="Live web dashboard for T-HCM training.")
    p.add_argument("--ckpt-dir", default="checkpoints", help="dir the trainer writes to")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    serve(args.ckpt_dir, args.host, args.port)


if __name__ == "__main__":
    main()
