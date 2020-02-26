import json
import numpy as np

from PIL import Image
from io import BytesIO

import utils

class CarDamageClassification(object):
    def __init__(self,models_path):
        self.models = utils.load_models(models_path)

    def image_loader(self, image_decod):
        """load image"""
        img = Image.open(BytesIO(image_decod))
        img = utils.preprocess_image(img)
        return img

    def predict(self, image):
        img_input = self.image_loader(image)
        trained_models = self.models
        response = utils.run_models(img_input, trained_models)
        return json.dumps(response)
