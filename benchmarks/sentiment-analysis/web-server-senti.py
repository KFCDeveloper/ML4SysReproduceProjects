from http.server import HTTPServer, BaseHTTPRequestHandler
from textblob import TextBlob
import json

def sentimentpy(params):
    try:
        analyse = TextBlob(params['analyse'])
    except:
        return {'Error' : 'Input parameters should include a string to sentiment analyse.'}

    sentences = len(analyse.sentences)

    retVal = {}

    retVal["subjectivity"] = sum([sentence.sentiment.subjectivity for sentence in analyse.sentences]) / sentences
    retVal["polarity"] = sum([sentence.sentiment.polarity for sentence in analyse.sentences]) / sentences
    retVal["sentences"] = sentences

    return retVal

class FunctionServer(BaseHTTPRequestHandler):
    def do_POST(self):
        # inherited from BaseHTTPRequestHandler 
        content_length = int(self.headers['Content-Length'])
        data = json.loads(self.rfile.read(content_length))

        # execute the function
        result = sentimentpy(data)

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
