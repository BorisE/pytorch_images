################################################################
# TELESCOPE PARK CLASSIFICATION MODULE
# 2023 By BorisE
# based on https://github.com/spmallick/learnopencv/blob/master/Image-Classification-in-PyTorch/image_classification_using_transfer_learning_in_pytorch.ipynb
#
# Webserver wrapper for test_park classification AI model
# USAGE: http://localhost:8080/?AstroCam_1.jpg
# RESONSE: {"status": "park", "score": "0.99993515", "duration": "0:00:02.754040"}
################################################################
from datetime import datetime
start_time = datetime.now()

from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from park_class_module import *

################################################################
# Settings
hostName = "localhost"
serverPort = 8080


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        start_time = datetime.now()

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        #self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        #self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        #self.wfile.write(bytes("<body>", "utf-8"))
        #self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        #self.wfile.write(bytes("</body></html>", "utf-8"))
        
        self.consoleprint("run")

        if self.path.startswith("/?"):
            filename = self.path[2:]
            model = torch.load(dataset + "/" + "{}_model_best.pt".format(dataset))
            returnDict = predict(model, filename, silent=False) #'AstroCam_1.jpg'
            returnDict["duration"] = str(datetime.now() - start_time) 

            responseSt = json.dumps(returnDict) 
        else:
            responseSt = "file not found"
            self.consoleprint(responseSt)

        self.wfile.write(bytes(responseSt, "utf-8"))
                
        print('Duration: {}'.format(datetime.now() - start_time))

        
    def consoleprint(self, st):
        print (" "*3 + str(st))


if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))
    print('Startup duration: {}'.format(datetime.now() - start_time))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")