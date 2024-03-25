from http.server import HTTPServer, BaseHTTPRequestHandler
import base64

def base64py():
    STR_SIZE = 1000000
    TRIES = 100
    str1 = b"a" * STR_SIZE
    str2 = b""
    s_encode = 0
    for _ in range(0, TRIES):
        str2 = base64.b64encode(str1)
        s_encode += len(str2)
    
    s_decode = 0
    for _ in range(0, TRIES):
        s_decode += len(base64.b64decode(str2))

    result = {'s_encode' : str(s_encode), 's_decode' : str(s_decode)}
    
    return result

class FunctionServer(BaseHTTPRequestHandler):
    def do_GET(self):
        # inherited from BaseHTTPRequestHandler 
        # execute the function
        result = base64py()

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html>\n<head><title>Function Execution Results</title></head>\n", "utf-8"))
        self.wfile.write(bytes("<body>\n", "utf-8"))
        self.wfile.write(bytes("<p>Encoded: %s</p>\n" % result['s_encode'], "utf-8"))
        self.wfile.write(bytes("<p>Decoded: %s</p>\n" % result['s_decode'], "utf-8"))
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
