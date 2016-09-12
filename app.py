from flask import Flask, render_template, url_for, request, session, redirect
from flask.ext.pymongo import PyMongo 
from pymongo import MongoClient
from werkzeug import secure_filename
import bcrypt
from newthing import a
from flask import send_from_directory
connection = MongoClient('localhost',27017)
db=connection.monogologinexample
flag=0
ALLOWED_EXTENSIONS = set(['txt'])
app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = set(['txt'])
#app.config['MONGO_DBNAME'] = connection.mongologinexample
#app.config['MONGO_URI'] = 'mongodb://pretty:printed@ds021731.mlab.com:21731/mongologinexample'

mongo = PyMongo(app)

@app.route('/')
def index():
    if 'username' in session:
        # return 'You are logged in as ' + session['username']
        return render_template('upload.html')
    return render_template('index.html')
	
	

@app.route('/login', methods=['POST'])
def login():
    users = db.users
    login_user = users.find_one({'name' : request.form['username']})

    if login_user:
        if bcrypt.hashpw(request.form['pass'].encode('utf-8'), login_user['password'].encode('utf-8')) == login_user['password'].encode('utf-8'):
            session['username'] = request.form['username']
            return render_template('upload.html')

    return 'Invalid username/password combination'

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        users =db.users
        existing_user = users.find_one({'name' : request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['pass'].encode('utf-8'), bcrypt.gensalt())
            users.insert({'name' : request.form['username'], 'password' : hashpass})
            session['username'] = request.form['username']
            return render_template('upload.html')
        
        return 'That username already exists!'

    return render_template('register.html')


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename('abcd.txt'))
        flag =1
        thefile = open('C:\Users\Krishna\Desktop\SemEval\static\output.txt', 'w')
        # b=['abc','def']
        for item in range(len(a)):
            thefile.write("%s \n" % a[item] )

            # count2=0
            # thefile.write("%s\n" % b[item])
            # for j in b[item]:
            #     count+=1
            #     thefile.write("(%s)" %j )
            #     if count>3:
            #         break
        thefile.close()
        return send_from_directory(directory='.\static', filename='output.txt')

#         return 'upload successful'
# @app.route('/download')
# def downl():
#     if flag==1:

    return render_template('upload.html')

if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(debug=True)
