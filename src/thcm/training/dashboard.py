"""Live web dashboard for an autonomous training run.

Reads the `metrics.jsonl` / `best.pt` / `last.pt` that ``thcm.training.auto``
writes and serves a self-refreshing page of status cards + charts (val/train
loss, next-concept accuracy, learning rate). Pure standard library — no torch,
no web framework — so it runs in its own process and can never disturb training.

    .\\.venv\\Scripts\\python.exe -m thcm.training.dashboard --ckpt-dir checkpoints
    # then open http://localhost:8000

Point ``--ckpt-dir`` at the same directory the trainer is writing to; the page
polls the server every few seconds, so it tracks a live run (or shows the final
state of a finished one).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_CKPT_DIR = "checkpoints"


def read_metrics(ckpt_dir: str) -> dict:
    """Parse metrics + checkpoint state into a JSON-serializable snapshot."""
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
                    continue                      # tolerate a half-written last line

    def finfo(name: str) -> dict:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            st = os.stat(p)
            return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}
        return {"exists": False}

    evals = [r for r in records if r.get("val_loss") is not None]
    latest = evals[-1] if evals else {}
    best_val = min((r["val_loss"] for r in evals), default=None)
    status = {
        "step": latest.get("step"),
        "val_loss": latest.get("val_loss"),
        "val_acc": latest.get("val_acc"),
        "train_loss": latest.get("train_loss"),
        "lr": latest.get("lr"),
        "best_val": best_val,
        "steps_per_sec": latest.get("steps_per_sec"),
        "last_event": records[-1].get("event") if records else None,
        "n_evals": len(evals),
        "updated": records[-1].get("t") if records else None,
        "now": time.time(),
    }
    series = {
        "step": [r.get("step") for r in evals],
        "val_loss": [r.get("val_loss") for r in evals],
        "train_loss": [r.get("train_loss") for r in evals],
        "val_acc": [r.get("val_acc") for r in evals],
        "lr": [r.get("lr") for r in evals],
    }
    return {"status": status, "series": series,
            "checkpoints": {"best": finfo("best.pt"), "last": finfo("last.pt")}}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>T-HCM training</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#0d1117;color:#e6edf3}
 header{padding:16px 24px;border-bottom:1px solid #30363d;display:flex;align-items:center;gap:16px}
 h1{font-size:18px;margin:0;font-weight:600}
 #dot{width:10px;height:10px;border-radius:50%;background:#888}
 .cards{display:flex;flex-wrap:wrap;gap:12px;padding:20px 24px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;min-width:120px}
 .card .label{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8b949e}
 .card .value{font-size:22px;font-weight:600;margin-top:4px}
 .charts{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:0 24px 24px}
 .chart{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}
 @media(max-width:900px){.charts{grid-template-columns:1fr}}
 .ev{padding:2px 8px;border-radius:6px;font-size:12px;background:#21262d}
</style></head><body>
<header><div id="dot"></div><h1>Tokenless T-HCM &mdash; training monitor</h1>
 <span class="ev" id="state">&mdash;</span><span id="fresh" style="color:#8b949e;font-size:13px"></span></header>
<div class="cards" id="cards"></div>
<div class="charts">
 <div class="chart"><canvas id="loss"></canvas></div>
 <div class="chart"><canvas id="acc"></canvas></div>
 <div class="chart"><canvas id="lr"></canvas></div>
</div>
<script>
const fmt=(v,d=3)=>v==null?'\\u2014':Number(v).toFixed(d);
const mk=(id,label,color,opts={})=>new Chart(document.getElementById(id),{type:'line',
 data:{labels:[],datasets:[]},options:{responsive:true,animation:false,
 plugins:{title:{display:true,text:label,color:'#e6edf3'},legend:{labels:{color:'#8b949e'}}},
 scales:{x:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}},
         y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'},...(opts.y||{})}}}});
const lossC=mk('loss','loss'),accC=mk('acc','next-concept accuracy'),
      lrC=mk('lr','learning rate',{},{y:{type:'logarithmic'}});
function ds(label,data,color){return {label,data,borderColor:color,backgroundColor:color,
 pointRadius:0,borderWidth:2,tension:.2}}
async function tick(){
 let d; try{d=await (await fetch('/api/status')).json()}catch(e){return}
 const s=d.status, x=d.series.step;
 const cards=[['state',s.last_event||'\\u2014'],['step',s.step??'\\u2014'],
  ['best val',fmt(s.best_val)],['val loss',fmt(s.val_loss)],
  ['val acc',fmt(s.val_acc)],['lr',s.lr==null?'\\u2014':s.lr.toExponential(1)],
  ['steps/sec',fmt(s.steps_per_sec,2)],['evals',s.n_evals??0]];
 document.getElementById('cards').innerHTML=cards.map(c=>
  `<div class="card"><div class="label">${c[0]}</div><div class="value">${c[1]}</div></div>`).join('');
 document.getElementById('state').textContent=s.last_event||'\\u2014';
 const age=s.updated?(s.now-s.updated):1e9, live=age<30;
 document.getElementById('dot').style.background=live?'#3fb950':'#8b949e';
 document.getElementById('fresh').textContent=s.updated?
  (live?'live':`idle (${Math.round(age)}s)`):'waiting for metrics\\u2026';
 lossC.data.labels=x; lossC.data.datasets=[ds('val',d.series.val_loss,'#58a6ff'),
   ds('train',d.series.train_loss,'#f778ba')]; lossC.update();
 accC.data.labels=x; accC.data.datasets=[ds('val acc',d.series.val_acc,'#3fb950')]; accC.update();
 lrC.data.labels=x; lrC.data.datasets=[ds('lr',d.series.lr,'#d29922')]; lrC.update();
}
tick(); setInterval(tick,3000);
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
