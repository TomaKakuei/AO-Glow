"""Small persistent observation-to-command HTTP endpoint, no simulation truth API."""
from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import json
import os
import hmac
from threading import Lock
import numpy as np
from .predictors import Predictor

def serve(system,profile,device,host,port):
    models={};lock=Lock();token=os.environ.get('DAO_API_TOKEN')
    branches={'mod3':('front','rear'),'trepan':('front','rear'),'kla':('module2','module3','module4'),'nikon':('joint',)}[system]
    for branch in branches:models[branch]=Predictor(system,None if system=='nikon' else branch,profile,device)
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,payload):
            raw=json.dumps(payload).encode('utf-8');self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.path=='/health':return self.reply(200,{'system':system,'profile':profile,'branches':branches})
            self.reply(404,{'error':'not found'})
        def do_POST(self):
            if token and not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):return self.reply(401,{'error':'unauthorized'})
            if not self.path.startswith('/predict/'):return self.reply(404,{'error':'not found'})
            branch=self.path.rsplit('/',1)[-1]
            if branch not in models:return self.reply(400,{'error':'unknown branch'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4*1024*1024:raise ValueError('Expected an NPZ observation <=4 MiB')
                with np.load(io.BytesIO(self.rfile.read(length)),allow_pickle=False) as z:image=z['observation']
                with lock:
                    estimate=models[branch](image)
                    command=-np.clip(estimate,-1,1) if system=='mod3' else (-np.clip(estimate,-1.5,1.5) if system=='nikon' else -estimate)
                self.reply(200,{'predicted_pose':estimate.tolist(),'command':command.tolist(),'system':system,'branch':branch,'profile':profile})
            except (ValueError,KeyError,TypeError) as e:self.reply(400,{'error':str(e)})
    server=HTTPServer((host,port),Handler)
    print(f'DAO service: http://{host}:{port}; {system}/{profile}',flush=True)
    try:server.serve_forever()
    finally:server.server_close()
