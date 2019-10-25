import os
from flask import Flask, escape, request, redirect, flash
from process import startprocess


app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_ml_file():
    if request.method == 'POST':
        print(request.args)
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No selected file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('filename is empty')
            return redirect(request.url)
        print("file recieved :"+ file.filename)
        print("saving to : uploads/1.mat")
        file.save(os.path.join('./uploads','1.mat'))
        result = startprocess('./uploads/1.mat',request.args["ll"])
        return result

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.debug = False
    app.run()