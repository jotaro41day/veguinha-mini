"""Read-only, stdlib progress page. Does not import Brian2 or run simulations."""
import argparse
import json
import os
from pathlib import Path
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT=Path(__file__).parent/'results/leg_reflex_v1'
PAGE=Path(__file__).parent/'web/progress.html'


def read_json(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def read_status(root=ROOT):
    rows=[]
    for p in root.glob('*.json'):
        r=read_json(p,{})
        if isinstance(r,dict) and all(k in r for k in ('profile','seed','joints')):
            r={**r,'id':p.stem,'modified':p.stat().st_mtime};rows.append(r)
    rows.sort(key=lambda r:r['modified'],reverse=True)
    progress=read_json(root/'progress.json',{})
    pid=progress.get('pid',0);alive=False
    if isinstance(pid,int) and pid>1:
        try:os.kill(pid,0);alive=True
        except (OSError,OverflowError):pass
    progress['alive']=alive
    if progress.get('state') in ('preparando','compilando','executando') and not alive:
        progress['state']='interrompido'
    try:log=(root/'unit_tests.log').read_text()[-12000:]
    except OSError:log='Testes automáticos ainda não executados nesta rodada.'
    n=re.search(r'Ran (\d+) tests? in',log)
    passed=bool(n and re.search(r'^OK\s*$',log,re.M))
    state='passed' if passed else 'failed' if 'FAILED (' in log else 'pending'
    task=read_json(root/'task.json',{'title':'Ajustando o módulo de uma perna','detail':'Analisando os resultados salvos.'})
    return dict(task=task,progress=progress,results=rows,
                gait=read_json(root.parent/'gait_loop_v1/report.json',{}),
                motor_pool=read_json(root.parent/'motor_pool_v1/pool_summary.json',{'groups':[]}),
                motor_pool_legs=read_json(root.parent/'motor_pool_v1/leg_summary_42_44.json',{'groups':[]}),
                motor_pool_confirmation=read_json(root.parent/'motor_pool_v1/leg_summary_1001_1010.json',{'groups':[]}),
                unit_tests=dict(state=state,passed=int(n[1]) if passed else 0,log=log),
                now=time.time(),device='Celular deste chat · 25078PC3EG')


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/':body=PAGE.read_bytes();kind='text/html; charset=utf-8'
        elif path=='/status':body=json.dumps(read_status(),allow_nan=False).encode();kind='application/json'
        elif path=='/gait.png' and (ROOT.parent/'gait_loop_v1/gait_comparison.png').exists():
            body=(ROOT.parent/'gait_loop_v1/gait_comparison.png').read_bytes();kind='image/png'
        elif path=='/favicon.ico':self.send_response(204);self.end_headers();return
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8000);a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);server.daemon_threads=True
    (ROOT/'progress_server.pid').write_text(str(os.getpid()))
    print(f'Painel de correções: http://127.0.0.1:{a.port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
