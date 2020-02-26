# Load libraries
import requests
import json
import base64

from PIL import Image
from os.path import join
from io import BytesIO


# Image path
image_path = 'test_image.JPEG'

# Endpoint
ip='localhost'

URL = f"http://{ip}:7011/"
ENDPOINT = join(URL, "predict")

# Prepare image
im = Image.open(image_path)

buffered = BytesIO()
im.save(buffered, format="JPEG")
data = base64.b64encode(buffered.getvalue())

# Send post request
headers = {'Content-Type': 'application/image'}
response = requests.post(ENDPOINT, headers = headers, data=data)
result = response.json()
print(json.loads(result))