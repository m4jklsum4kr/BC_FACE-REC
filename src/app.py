# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Project     : Privacy-preserving face recognition
# Created By  : Elodie CHEN - Bruce L'HORSET - Charles MAILLEY
# Created Date: 11/02/2025
# Referent: Sara Ricci - ricci@vut.cz
# version ='1.0'
# ---------------------------------------------------------------------------
"""
This project will explore the intersection of Machine Learning (ML) and data privacy.
The student will investigate data anonymization techniques, such as differential privacy and k-anonymity, to enhance the privacy of ML models for facial recognition.
The aim of the project is the development a prototype that take a photo and match it with the one in the anonymized database.
"""
# ---------------------------------------------------------------------------
# Usefully links:
# * https://www.geeksforgeeks.org/single-page-portfolio-using-flask/
# * https://realpython.com/flask-blueprint/
# * https://www.geeksforgeeks.org/flask-rendering-templates/
# Usefully commands
# $ pip freeze > requirements.txt; poetry init
# ---------------------------------------------------------------------------
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
from config import Config
import modules.utils as utils

import os
from PIL import Image
import io
import pandas as pd
import base64


app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = b'\x1f\x0e\x0c\xa6\xdbt\x01S\xa0$r\xf8$\xb4\xe3\x8a\xcf\xe0\\\x00M0H\x01'
#app.config['UPLOAD_FOLDER'] = os.path.join('src', 'static', 'uploads')


# ---------------------------------------------------------------------------
# ------------------------- WEB PAGE ----------------------------------------
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search_people")
def search_people():
    return render_template("search_people.html")

@app.route("/new_people")
def new_people():
    return render_template("new_people.html")


@app.route("/show_database")
def show_database():
    return render_template("show_database.html")

@app.route("/new_people", methods=['POST'])
def new_people_processing():
    print("ddd")
    if not request.files.getlist('fileInput'):
        print("No file part")
        error = "No file part"
        return  render_template('new_people.html', error=error)

    files = request.files.getlist('fileInput')
    print(len(request.form), request.form)
    print(len(files), type(files[0]), files)

    file_urls = []
    for file in files:
        pillow_image = Image.open(io.BytesIO(file.read()))
        file_urls.append(pillow_image)
    print(file_urls)

    df_images = pd.DataFrame({'images':file_urls, "idUser": 16, "idNumber":range(1, len(file_urls)+1)})
    print(df_images)

    #workflow = PeepWorkflow(image_folder)
    #return workflow.getEngeiface(), workflow.getBruitedimage()
    #return render_template('new_people.html', var1=workflow.getEngeiface())

    # Converti image pillow en binaire pour ne pas avoir a les save en memoire
    df_end = []
    for image in df_images['images'].to_list():
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        df_end.append(img_str)
    print(len(df_end), df_end)


    return render_template("result.html", df_end=df_end)

# ---------------------------------------------------------------------------
# ------------------------- BACK FUNCTIONS ----------------------------------
# ---------------------------------------------------------------------------

@app.route('/api/check_photo', methods=['POST'])
def check_photo():
    return jsonify({'result': utils.random_bool()})

#@app.route('/api/db_search_all', methods=['POST'])
#def db_search_all():
    #return jsonify({'result': Config.db.select_all()})

#@app.route('/api/db_search_image', methods=['POST'])
#def db_search_image(id_subject, id_image):
    #return jsonify({'result': Config.db.select_image(id_subject, id_image)})

#@app.route('/static/uploads/<filename>')
#def uploaded_file(filename):
    #return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# ---------------------------------------------------------------------------
# ------------------------- MAIN --------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)


