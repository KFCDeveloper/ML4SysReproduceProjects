import requests
session = requests.Session()
print("——————————————————————————————————————————————")
print("——————————————————————————————————————————————")
addr = "http://localhost:8080"
r = session.post(addr + "/wrk2-api/user/register", data="{'first_name': 'first_name_1', 'last_name': 'last_name_1', 'username': 'username_1', 'password': 'password_1', 'user_id': '1'}")
print(r.status_code)