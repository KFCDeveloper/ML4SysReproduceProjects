from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

def primespy():
    n = 10000000

    if n < 2:
        return {'Number of primes found': 0}
    if n == 2:
        return {'Number of primes found': 2}
    # do only odd numbers starting at 3
    if sys.version_info.major <= 2:
        s = range(3, n + 1, 2)
    else:  # Python 3
        s = list(range(3, n + 1, 2))
    # n**0.5 simpler than math.sqr(n)
    mroot = n ** 0.5
    half = len(s)
    i = 0
    m = 3
    while m <= mroot:
        if s[i]:
            j = (m * m - 3) // 2  # int div
            s[j] = 0
            while j < half:
                s[j] = 0
                j += m
        i = i + 1
        m = 2 * i + 3
    res = [2] + [x for x in s if x]
    return {'Number of primes found': len(res)}

class FunctionServer(BaseHTTPRequestHandler):
    def do_GET(self):
        # inherited from BaseHTTPRequestHandler 
        # execute the function
        result = primespy()

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html>\n<head><title>Function Execution Results</title></head>\n", "utf-8"))
        self.wfile.write(bytes("<body>\n", "utf-8"))
        self.wfile.write(bytes("<p>%s</p>\n" % result, "utf-8"))
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
