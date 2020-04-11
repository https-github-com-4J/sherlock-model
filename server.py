from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import detector

port = 8085
class S(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/execute/sherlock':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            detector.execute()

            to_return = "{\"response\": \"The image was cropped successfully\"}"
            self.wfile.write(bytes(to_return, "utf-8"))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes("{\"message\": \"Not found\"}", "utf-8"))

def run(server_class=HTTPServer, handler_class=S, port=port):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info(' sherlock-model listening at port ' + str(port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()