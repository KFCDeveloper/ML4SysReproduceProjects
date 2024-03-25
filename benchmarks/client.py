import requests as req
import json
import sys

host = 'localhost'
port = '30001'
function = 'sentiment'

if len(sys.argv) != 3:
    print('Usage: python3 client.py <port> <function>')
    exit()
port = sys.argv[1]
function = sys.argv[2]

if function == 'base64':
    # GET - base64
    resp = req.request(method='GET', url='http://'+host+':'+port)
    print(resp.text)
elif function == 'json':
    # POST - json
    # data = {'name': 'value'}
    with open('json-data.json') as f:
        data = json.load(f)
    resp = req.post('http://'+host+':'+port, json=data)
    print(resp.text)
elif function == 'primes':
    # GET - primes
    resp = req.request(method='GET', url='http://'+host+':'+port)
    print(resp.text)
elif function == 'sentiment':
    # POST - sentiment
    with open('senti-data.json') as f:
        data = json.load(f)
    resp = req.post('http://'+host+':'+port, json=data)
    print(resp.text)
