"""
This script serves for testing on visualizing the resulting image with masks printed on and would be chosen as the
candidate script for submission.
"""

from flask import Flask, render_template, redirect, url_for, send_from_directory, send_file, request, jsonify
from flask_uploads import IMAGES, UploadSet, configure_uploads
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField, FileRequired, FileAllowed

import numpy as np
# from app.utils.model import run_model, get_model
from redis import Redis
import os
import time

from rq import Queue
from rq.job import Job
from app.worker import conn
import pathlib2
from app.config import cfg  # since app is the base package
from app.mmdet_utils import get_prediction  # since app is the base package


current_dir = pathlib2.Path.cwd()
project_dir = current_dir.parent
ds_path = project_dir / 'test_imgs'
print(f"the path to the folder holding the testing imgs: {ds_path}")

app = Flask(__name__)

# app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 0)
app.config['UPLOADED_IMAGES_DEST'] = 'app/templates/img/'
app.config['DOWNLOAD_IMAGES_DEST'] = 'templates/img/'
app.config['PREFERRED_URL_SCHEME'] = 'https'

task_queue = Queue('detector', connection=conn)

images = UploadSet('images', IMAGES)
configure_uploads(app, images)


class Upload(FlaskForm):
    image = FileField('image', validators=[
        FileRequired(),
        FileAllowed(images, 'Images only!')])
    submit = SubmitField('Submit')


class Download(FlaskForm):
    submit = SubmitField('Download')


@app.route('/', methods=['GET', 'POST'])
def index():
    upload_form = Upload()
    if request.method == 'POST':
        if upload_form.validate_on_submit():
            filename = images.save(upload_form.image.data)
            processed_filename = f"detected_{filename}"

            img = f"{app.config['UPLOADED_IMAGES_DEST']}{filename}"
            processed_img = f"{app.config['UPLOADED_IMAGES_DEST']}{processed_filename}"

            job = task_queue.enqueue('app.mmdet_utils.get_prediction',
                                     config=cfg.CONFIG,
                                     checkpoint=cfg.CHECKPOINT,
                                     img=(ds_path / '7ae19de7bc2a.png').as_posix(),
                                     processed_filename=processed_filename,
                                     processed_img="/".join(processed_img.split('/')[1:]),
                                     result_ttl=86400)
            job_id = job.get_id()

            return redirect(url_for('show', picture=filename, job_id=job_id))
    else:
        return render_template('index.html', form=upload_form)


@app.route('/show/<picture>/<job_id>', methods=['GET', 'POST'])
def show(picture, job_id):
    download_form = Download()

    job = Job.fetch(job_id, connection=conn)
    job_complete = job.is_finished

    filename = f"{picture}"
    processed_filename = f"detected_{picture}"
    img = f"{app.config['UPLOADED_IMAGES_DEST']}{filename}"
    processed_img = f"{app.config['UPLOADED_IMAGES_DEST']}{processed_filename}"

    if download_form.validate_on_submit() and job_complete:
        print(processed_img)
        return send_from_directory(app.config['DOWNLOAD_IMAGES_DEST'], processed_filename, as_attachment=True)

    return render_template('show.html', pic=filename, job_id=job_id, job_complete=job_complete, form=download_form)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=True)