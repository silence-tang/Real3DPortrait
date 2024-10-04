from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import os
from datetime import datetime
import os
import argparse
import sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import RandomSampler, DataLoader

app = Flask(__name__)

face_swapping = None

def my_responese(status = "success", message = "", data = ""):
    return  jsonify({"status": status,
            "message": message,
            "data": data}), 200


def my_swap(soruce_path, target_path, output_path, output_real):
    global face_swapping
    face_swapping(soruce_path, target_path, output_path)
    video_first_frame_to_image(output_path, output_real)

@app.route('/api/relace-face/sample', methods=['POST'])
def swap_known_face():
    try:
        person_id = request.form.get('person_id', None)
        source = request.form.get('source', None)
        target = request.form.get('target', None)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if person_id is None and source is None:
            return my_responese(message = "Must have one of person_id or source image!")
        
        if person_id not in person_dict.keys():
            return my_responese(message = "Only can choose from this list: [mahuateng, masike, telangpu, anbei]")
        
        if target is None:
            return my_responese(message="Must have a target!")
        
        if person_id is not None:
            src = person_dict[person_id]
        else:
            src = os.path.join(folder_path, f"{timestamp}_src{file_extension}")
            image_tmp = Image.open(BytesIO(base64.b64decode(source)))
            image_tmp.save(src)
        tgt = os.path.join(folder_path, f"{timestamp}_tgt{file_extension}")
        image_tmp = Image.open(BytesIO(base64.b64decode(target)))
        image_tmp.save(tgt)

        output_tmp = os.path.join(folder_path, f"{timestamp}_op.mp4")
        output_real = os.path.join(folder_path, f"{timestamp}_op.jpg")

        my_swap(soruce_path=src, target_path=tgt, output_path=output_tmp, output_real=output_real)

        img = Image.open(output_real)
        buffered = BytesIO()
        img.save(buffered, format="JPG")
        img_byte_data = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte_data).decode('utf-8')
        return my_responese(data=img_base64)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    face_swapping = FaceSwapping()
    app.run(debug=True)