import requests as req
import json
import sys

host = 'localhost'
port = '8000'
function = 'sentiment'

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
