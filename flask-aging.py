import os
import glob
import random
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from torchvision.datasets.folder import pil_loader
from model import Net
from utils import pil_to_model_tensor_transform
import consts

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# UTKFace constants
MALE = 0
FEMALE = 1
WHITE = 0
BLACK = 1
ASIAN = 2
INDIAN = 3
OTHER = 4

# User constants
dset_path = os.path.join('.', 'data', 'UTKFace', 'unlabeled')

# Initialize the model
consts.NUM_Z_CHANNELS = 100
net = Net()
load_path = {100: r"trained_models/100_Z_channels_200th_epoch"}[consts.NUM_Z_CHANNELS]
net.load(load_path, slim=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/age_progression', methods=['GET', 'POST'])
def age_progression():
    if request.method == 'POST':
        # Get the user inputs
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        race = int(request.form['race'])

        # Select a random image path based on the user inputs
        image_path = random.choice(glob.glob(os.path.join(dset_path, '{a}_{g}_{r}*'.format(a=age, g=gender, r=race))))

        # Load the image and generate the age progression
        image_tensor = pil_to_model_tensor_transform(pil_loader(image_path))
        result_path = net.test_single(image_tensor=image_tensor, age=age, gender=gender, target='static', watermark=False)

        return render_template('age_progression.html', original_image=image_path, result_image=result_path)
    return render_template('age_progression.html')

@app.route('/morphing', methods=['GET', 'POST'])
def morphing():
    if request.method == 'POST':
        # Get the user inputs for the first person
        age_1 = int(request.form['age_1'])
        gender_1 = int(request.form['gender_1'])
        race_1 = int(request.form['race_1'])

        # Get the user inputs for the second person
        age_2 = int(request.form['age_2'])
        gender_2 = int(request.form['gender_2'])
        race_2 = int(request.form['race_2'])

        # Select random image paths based on the user inputs
        image_path_1 = random.choice(glob.glob(os.path.join(dset_path, '{a}_{g}_{r}*'.format(a=age_1, g=gender_1, r=race_1))))
        image_path_2 = random.choice(glob.glob(os.path.join(dset_path, '{a}_{g}_{r}*'.format(a=age_2, g=gender_2, r=race_2))))

        # Load the images and generate the morphing sequence
        image_tensor_1 = pil_to_model_tensor_transform(pil_loader(image_path_1))
        image_tensor_2 = pil_to_model_tensor_transform(pil_loader(image_path_2))
        result_path = net.morph(image_tensors=(image_tensor_1, image_tensor_2), ages=(age_1, age_2), genders=(gender_1, gender_2), length=10, target='static')

        return render_template('morphing.html', image_1=image_path_1, image_2=image_path_2, result_image=result_path)
    return render_template('morphing.html')

@app.route('/kids', methods=['GET', 'POST'])
def kids():
    if request.method == 'POST':
        # Get the user inputs for the first person
        age_1 = int(request.form['age_1'])
        gender_1 = int(request.form['gender_1'])
        race_1 = int(request.form['race_1'])

        # Get the user inputs for the second person
        age_2 = int(request.form['age_2'])
        gender_2 = int(request.form['gender_2'])
        race_2 = int(request.form['race_2'])

        # Select random image paths based on the user inputs
        image_path_1 = random.choice(glob.glob(os.path.join(dset_path, '{a}_{g}_{r}*'.format(a=age_1, g=gender_1, r=race_1))))
        image_path_2 = random.choice(glob.glob(os.path.join(dset_path, '{a}_{g}_{r}*'.format(a=age_2, g=gender_2, r=race_2))))

        # Load the images and generate the kids sequence
        image_tensor_1 = pil_to_model_tensor_transform(pil_loader(image_path_1))
        image_tensor_2 = pil_to_model_tensor_transform(pil_loader(image_path_2))
        result_path = net.kids(image_tensors=(image_tensor_1, image_tensor_2), length=10, target='static')

        return render_template('kids.html', image_1=image_path_1, image_2=image_path_2, result_image=result_path)
    return render_template('kids.html')

if __name__ == '__main__':
    app.run(debug=True)