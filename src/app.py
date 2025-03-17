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
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory, session
from config import *

import modules.utils as utils
from modules.gui_controller import GUIController
from modules.peep import Peep
from modules.main import Main

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
    if not request.files.getlist('fileInput'):
        print("No file part")
        error = "No file part"
        return  render_template('new_people.html', error=error)
    files = request.files.getlist('fileInput')

    c = GUIController(files)
    c.s1_apply_k_same_pixel()
    c.s2_resize_images((100, 100))
    c.s3_generate_pca_components()
    c.s4_apply_differential_privacy(5)

    #session['GUIController'] = c # Save session object
    #c = session.get('my_object', None)  # Retrieve session object


    renderer = c.get_image_source() + c.get_image_eigenface() + c.get_image_noised()
    return render_template("result.html", eigenfaces_list=renderer)

# ---------------------------------------------------------------------------
# ------------------------- BACK FUNCTIONS ----------------------------------
# ---------------------------------------------------------------------------

@app.route('/api/check_photo', methods=['POST'])
def check_photo():
    return jsonify({'result': utils.random_bool()})


# ---------------------------------------------------------------------------
# ------------------------- MAIN --------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

