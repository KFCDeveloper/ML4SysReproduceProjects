from http.server import HTTPServer, BaseHTTPRequestHandler
import json

def jsonpy(params):
    try:
        length = len(params['coordinates'])
    except:
        return {'Error' : 'Input parameters should include coordinates.'}
    x = 0
    y = 0
    z = 0

    for coord in params['coordinates']:
        x += coord['x']
        y += coord['y']
        z += coord['z']

    return {'x' : x/length, 'y' : y/length, 'z' : z/length}

class FunctionServer(BaseHTTPRequestHandler):
    def do_POST(self):
        # inherited from BaseHTTPRequestHandler
        content_length = int(self.headers['Content-Length'])
        data = json.loads(self.rfile.read(content_length))

        # execute the function
        results = jsonpy(data)

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html>\n<head><title>Function Execution Results</title></head>\n", "utf-8"))
        self.wfile.write(bytes("<body>\n", "utf-8"))
        self.wfile.write(bytes("<p>%s</p>\n" % results, "utf-8"))
        self.wfile.write(bytes("</body>\n</html>", "utf-8"))

def run(server_class=HTTPServer, handler_class=FunctionServer):
    server_address = ("0.0.0.0", 8000)
    httpd = server_class(server_address, handler_class)
    print("Launching server...")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print("\nServer stopped.")

if __name__ == "__main__":
    run()
