"""Local web dashboard — live risk meter, branch scores, transcripts, escalation log.

Stdlib http.server only (no framework dependency), single self-contained page, no external
assets: the app must run on a laptop with no network. Binds to 127.0.0.1 by default — this
streams what the machine is playing, so it must not be reachable from the LAN.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Harm monitor</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--fg:#e7e9ee;--dim:#8b93a3;--ok:#37b26a;--watch:#e0a83a;--alert:#e0533a}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 system-ui,-apple-system,"Helvetica Neue",sans-serif;background:var(--bg);color:var(--fg)}
header{padding:14px 20px;border-bottom:1px solid #232733;display:flex;gap:16px;align-items:baseline}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.02em}
.sub{color:var(--dim);font-size:12px}
main{padding:20px;display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));max-width:1200px}
.card{background:var(--card);border:1px solid #232733;border-radius:10px;padding:16px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:0 0 12px}
.big{font-size:42px;font-weight:600;line-height:1;font-variant-numeric:tabular-nums}
.meter{height:10px;background:#232733;border-radius:5px;overflow:hidden;margin-top:12px}
.meter>div{height:100%;transition:width .3s,background .3s}
.badge{display:inline-block;padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600}
.ok{background:rgba(55,178,106,.15);color:var(--ok)}
.watch{background:rgba(224,168,58,.15);color:var(--watch)}
.alert{background:rgba(224,83,58,.15);color:var(--alert)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);padding:0 8px 6px 0;font-weight:500}
td{padding:4px 8px 4px 0;border-top:1px solid #232733;font-size:13px}
.wrap{max-height:280px;overflow-y:auto}
.kv{display:flex;justify-content:space-between;padding:3px 0;font-size:13px}
.kv span:first-child{color:var(--dim)}
.tx{color:var(--fg);font-size:13px;padding:6px 0;border-top:1px solid #232733;white-space:pre-wrap}
.tx small{color:var(--dim)}
.muted{color:var(--dim)}
</style></head><body>
<header><h1>Playback harm monitor</h1><span class="sub" id="src">connecting…</span></header>
<main>
 <section class="card"><h2>Risk (smoothed)</h2>
   <div class="big" id="risk">–</div><div class="meter"><div id="bar" style="width:0%"></div></div>
   <div style="margin-top:12px"><span class="badge ok" id="lvl">ok</span></div></section>
 <section class="card"><h2>Latest window</h2>
   <div class="kv"><span>acoustic (violence)</span><b id="a">–</b></div>
   <div class="kv"><span>text (any-harm)</span><b id="t">–</b></div>
   <div class="kv"><span>rms</span><b id="rms">–</b></div>
   <div class="kv"><span>compute / window</span><b id="lat">–</b></div></section>
 <section class="card"><h2>Session</h2>
   <div class="kv"><span>windows</span><b id="w">0</b></div>
   <div class="kv"><span>events</span><b id="e">0</b></div>
   <div class="kv"><span>ASR runs</span><b id="ar">0</b></div>
   <div class="kv"><span>ASR backend</span><b id="asrb">–</b></div>
   <div class="kv"><span>Demucs runs</span><b id="dr">0</b></div>
   <div class="kv"><span>Demucs / Whisper time</span><b id="ptime">–</b></div>
   <div class="kv"><span>server</span><b id="srv">local only</b></div>
   <div class="kv"><span>thresholds a/t</span><b id="thr">–</b></div></section>
 <section class="card" style="grid-column:1/-1"><h2>Harm events (one per incident, sent to the server)</h2>
   <div class="wrap" id="evs"><div class="muted">none yet</div></div></section>
 <section class="card" style="grid-column:1/-1"><h2>Transcripts (on-device ASR)</h2>
   <div class="wrap" id="txs"><div class="muted">no speech yet</div></div></section>
 <section class="card" style="grid-column:1/-1"><h2>Recent windows</h2>
   <div class="wrap"><table><thead><tr><th>t</th><th>acoustic</th><th>text</th><th>level</th><th>reasons</th><th>ms</th></tr></thead>
   <tbody id="rows"></tbody></table></div></section>
</main>
<script>
const f=(v,d=3)=>v==null?"–":(+v).toFixed(d);
async function tick(){
 let s; try{ s=await (await fetch("/api/state")).json(); }catch(e){ document.getElementById("src").textContent="disconnected"; return; }
 document.getElementById("src").textContent=s.source+" · "+s.stats.windows+" windows";
 const r=s.risk??0, rs=s.results||[], last=rs[rs.length-1]||{};
 document.getElementById("risk").textContent=f(r,2);
 const bar=document.getElementById("bar"); bar.style.width=Math.min(100,r*100)+"%";
 const lvl=last.level||"ok", col=getComputedStyle(document.documentElement).getPropertyValue("--"+lvl);
 bar.style.background=col.trim()||"#37b26a";
 const b=document.getElementById("lvl"); b.textContent=lvl; b.className="badge "+lvl;
 document.getElementById("a").textContent=f(last.acoustic);
 document.getElementById("t").textContent=f(last.text);
 document.getElementById("rms").textContent=f(last.rms,4);
 document.getElementById("lat").textContent=last.latency_ms!=null?last.latency_ms+" ms":"–";
 document.getElementById("w").textContent=s.stats.windows;
 document.getElementById("e").textContent=(s.stats.events??0)+" ("+s.stats.escalations+" windows)";
 const evAll=(s.events||[]).concat(s.open_event?[Object.assign({open:1},s.open_event)]:[]);
 document.getElementById("evs").innerHTML=evAll.length?evAll.slice(-12).reverse().map(x=>
   `<div class="tx"><small>${f(x.start,0)}–${f(x.end,0)}s · ${f(x.duration,0)}s · ${x.windows} windows ·
    ${(x.reasons||[]).join(", ")} · peak ${f(x.peak_score,2)}${x.open?' · <b>ongoing</b>':''}</small>
    ${(x.transcripts||[]).length?"<br>"+x.transcripts[x.transcripts.length-1].replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c])):""}</div>`).join("")
   :'<div class="muted">none yet</div>';
 document.getElementById("ar").textContent=s.stats.asr_runs;
 document.getElementById("asrb").textContent=s.stats.asr_backend||"–";
 document.getElementById("dr").textContent=s.stats.demucs_runs||0;
 document.getElementById("ptime").textContent=(s.stats.demucs_seconds!=null||s.stats.whisper_seconds!=null)
   ?f(s.stats.demucs_seconds||0,2)+"s / "+f(s.stats.whisper_seconds||0,2)+"s":"–";
 const tp=s.transport||{};
 document.getElementById("srv").textContent=tp.server_url?(tp.server_url+" · sent "+tp.sent+" / failed "+tp.failed):"local only";
 document.getElementById("thr").textContent=f(s.thresholds.acoustic,2)+" / "+f(s.thresholds.text,2);
 document.getElementById("rows").innerHTML=rs.slice(-40).reverse().map(x=>
   `<tr><td>${f(x.t_start,0)}s</td><td>${f(x.acoustic)}</td><td>${f(x.text)}</td>
    <td><span class="badge ${x.level}">${x.level}</span></td><td>${(x.reasons||[]).join(", ")||"–"}</td><td>${x.latency_ms}</td></tr>`).join("");
 const txs=rs.filter(x=>x.transcript&&x.transcript.trim()).slice(-12).reverse();
 document.getElementById("txs").innerHTML=txs.length?txs.map(x=>
   `<div class="tx"><small>${f(x.t_start,0)}s · text ${f(x.text)}</small><br>${x.transcript.replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}</div>`).join("")
   :'<div class="muted">no speech yet</div>';
}
tick(); setInterval(tick,1000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    engine = None
    source_name = "?"

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/state"):
            state = self.engine.snapshot()
            state["source"] = self.source_name
            self._send(200, "application/json", json.dumps(state).encode())
        elif self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):      # keep the console clean for engine output
        return


def serve(engine, source_name: str = "?", host: str = "127.0.0.1", port: int = 8765):
    """Start the dashboard on a daemon thread; returns the server (call shutdown() to stop)."""
    handler = type("Handler", (_Handler,), {"engine": engine, "source_name": source_name})
    httpd = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=httpd.serve_forever, name="dashboard", daemon=True).start()
    return httpd
