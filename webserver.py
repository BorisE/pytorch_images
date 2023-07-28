from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import json


hostName = "localhost"
serverPort = 8080


def getResponseSt():
    response = ["update", {"bar": ["baz", None, 1.0, 2], "time": time.time()}]
    responseSt_parsed = json.dumps(response)
    return responseSt_parsed


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(bytes(getResponseSt(), "utf-8"))
        #self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        #self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        #self.wfile.write(bytes("<body>", "utf-8"))
        #self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        #self.wfile.write(bytes("</body></html>", "utf-8"))
        self.consoleprint("run")
        
    def consoleprint(self, st):
        print (" "*3 + str(st))


if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")