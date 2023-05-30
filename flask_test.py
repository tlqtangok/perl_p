#from flask import Flask
from flask import Flask, session, redirect, url_for, escape, request,jsonify
app = Flask(__name__)

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
def data_to_username(data_):
    id_list= data_["username"]
    print("- len id_list:", len(id_list))
    #print("- id_list:", id_list)
    return id_list


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')
def about():
    return 'The about page'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        #a = "ABC"
        #a = request.form["username"]
        #session["username"] = request.form["username"]
        data = request.get_json()
        a="abcdefg"
        #print (data)
        #return  data["username"]
        return data_to_username(data) 

    return ''' login() '''


"""

#curl -H 'Content-Type: application/json' -X POST -d '{"username": "jidortang_1:2:3:4"}' 'localhost:5000/login'
#curl -H 'Content-Type: application/json' -X POST -d '{"username": "jidortang_0:1,2,3"}' 'localhost:5000/login'
curl -H 'Content-Type: application/json' -X POST -d '@json.json' 'localhost:5000/login'



{
"username":[
    1.2,4,5,6,
    4,5
]
}

"""
