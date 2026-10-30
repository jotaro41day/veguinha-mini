import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from veguinha_mini.server import make_handler


class FakeSimulation:
    def __init__(self):
        self.commands = []

    def submit(self, command):
        self.commands.append(command)

    def initial(self):
        return {"type": "init", "regions": ["kc"]}

    def latest(self):
        return {"type": "frame", "rates": [1.0]}


class ServerTests(unittest.TestCase):
    def test_http_page_state_and_post_fallback(self):
        simulation = FakeSimulation()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(simulation))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/") as response:
                self.assertIn(b"<canvas", response.read())
            with urlopen(base + "/state") as response:
                self.assertEqual(json.load(response)["regions"], ["kc"])
            with urlopen(base + "/events", timeout=2) as response:
                self.assertEqual(response.headers.get_content_type(), "text/event-stream")
                self.assertIn(b'"type": "init"', response.readline())
            payload = json.dumps({"cmd": "pause", "value": True}).encode()
            request = Request(base + "/command", data=payload,
                              headers={"Content-Type": "application/json"})
            with urlopen(request) as response:
                self.assertEqual(response.status, 202)
            self.assertEqual(simulation.commands, [{"cmd": "pause", "value": True}])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
