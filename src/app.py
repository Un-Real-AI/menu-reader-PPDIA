from flask import Flask, make_response, request
from flask_cors import CORS, cross_origin

import modules.reader as reader
import modules.upload as upload
import threading
from dotenv import load_dotenv, find_dotenv
from werkzeug.utils import secure_filename

load_dotenv(find_dotenv())

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

BUCKET='contler-prueba-2023'

#Run menu image through the model pipeline
@app.route('/api/menu_reader', methods=['POST'])
@cross_origin()
def read_menu():
    body=request.json
    return reader.read_menu(body['path'], body['language'],BUCKET)

def read_menu_trigger(path,language='eng'):
    return reader.read_menu(path,language,BUCKET)

#Upload an image on AWS S3
@app.route('/api/upload_image', methods=['POST'])
@cross_origin()
def upload_image():
    if "file" not in request.files:
        return "No file key in request.files"
    file = request.files["file"]
    if file.filename == "":
        return "Please select a file"
    if file:
        file.filename = secure_filename(file.filename)
        lang=request.form["language"]
        output,file_name = upload.sent_to_s3(file, BUCKET)
        threading.Thread(target=read_menu_trigger, args=(file_name,lang)).start()
        return output

#Status checking of the process
@app.route('/api/check_status/<process_id>', methods=['GET'])
@cross_origin()
def check_status(process_id):
    return upload.check_status(process_id,BUCKET)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
