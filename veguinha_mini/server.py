"""HTTP local, SSE/POST e WebSocket opcional para o laboratório."""

import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


PAGE = Path(__file__).with_name("web") / "index.html"


def make_handler(simulation):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def json_response(self, status, payload):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/state":
                self.json_response(200, simulation.initial())
            elif path == "/viewer-stats":
                bridge = getattr(simulation, "viewer_bridge", None)
                clients = bridge.server.get_clients() if bridge else {}
                self.json_response(200, {"enabled": bridge is not None,
                                         "clients": len(clients),
                                         "frames": bridge.frames if bridge else 0,
                                         "render_seconds": round(bridge.render_seconds, 3) if bridge else 0,
                                         "camera": [{"position": list(c.camera.position),
                                                     "look_at": list(c.camera.look_at)}
                                                    for c in clients.values()]})
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    self.wfile.write(b"data:" + json.dumps(simulation.initial()).encode() + b"\n\n")
                    self.wfile.flush()
                    last_sequence = -1
                    while True:
                        frame = simulation.latest()
                        sequence = frame.get("seq", 0)
                        if sequence != last_sequence:
                            self.wfile.write(b"data:" + json.dumps(frame, separators=(",", ":")).encode() + b"\n\n")
                            self.wfile.flush()
                            last_sequence = sequence
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            else:
                self.send_error(404)

        def do_POST(self):
            if urlsplit(self.path).path != "/command":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("tamanho inválido")
                command = json.loads(self.rfile.read(length))
                simulation.submit(command)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.json_response(400, {"error": str(error)})
                return
            self.json_response(202, {"queued": True})

    return Handler


def start_websocket(simulation, port):
    try:
        from websockets.sync.server import serve
        from websockets.exceptions import ConnectionClosed
    except ImportError:
        return False

    def handler(connection):
        connection.send(json.dumps(simulation.initial(), separators=(",", ":")))
        stopped = threading.Event()

        def sender():
            last_sequence = -1
            while not stopped.is_set():
                frame = simulation.latest()
                sequence = frame.get("seq", 0)
                if sequence != last_sequence:
                    try:
                        connection.send(json.dumps(frame, separators=(",", ":")))
                    except Exception:
                        break
                    last_sequence = sequence
                stopped.wait(0.05)

        worker = threading.Thread(target=sender, daemon=True)
        worker.start()
        try:
            for message in connection:
                try:
                    simulation.submit(json.loads(message))
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    connection.send(json.dumps({"type": "error", "message": str(error)}))
        except ConnectionClosed:
            pass
        finally:
            stopped.set()
            worker.join(timeout=1)

    def run():
        with serve(handler, "127.0.0.1", port) as server:
            server.serve_forever()

    threading.Thread(target=run, daemon=True).start()
    return True


def serve_local(simulation, port=8000, viewer=True):
    bridge = None
    if viewer:
        from .mjviser_bridge import MjviserBridge
        bridge = MjviserBridge(simulation, port=port + 2)
        simulation.viewer_bridge = bridge
        simulation.viewer_port = port + 2
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(simulation))
    simulation.start()
    if bridge:
        bridge.start()
    websocket = start_websocket(simulation, port + 1)
    print(f"Abra http://127.0.0.1:{port}/ (WebSocket {'ativo' if websocket else 'indisponível; SSE ativo'}; "
          f"mjviser {'em '+str(port+2) if bridge else 'desligado'})", flush=True)
    prior_handler = signal.getsignal(signal.SIGINT)

    def request_shutdown(_number, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGINT, prior_handler)
        server.server_close()
        if bridge:
            bridge.stop()
        simulation.stop()
