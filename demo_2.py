from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import os
from datetime import datetime
import argparse
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from inference.real3d_infer import GeneFace2Infer

app = Flask(__name__)

# 服务器端的数据库
person_list = ['person1', 'person2', 'person3']
bg_list = ['bg1', 'bg2', 'bg3']
aud_list = ['aud1', 'aud2', 'aud3']
root_path = '/root/autodl-tmp/.autodl/Real3DPortrait'

# 提前实例化推理类，并加载模型
inp = {
    'a2m_ckpt': 'checkpoints/240210_real3dportrait_orig/audio2secc_vae',
    'head_ckpt': '',
    'torso_ckpt': 'checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig',
    # 'src_image_name': person + '.png',
    # 'bg_image_name': bg + '.png',
    # 'drv_audio_name': aud + '.mp3',
    'drv_pose_name': 'static',
    'blink_mode': 'period',
    'temperature': 0.2,
    'mouth_amp': 0.45,
    # 'out_name': video_name,
    'out_mode': 'final',
    'map_to_init_pose': 'True',
    'head_torso_threshold': None,
    'seed': None,
    'min_face_area_percent': 0.2,
    'low_memory_usage': True,
}

# 加载模型
video_forging = GeneFace2Infer(inp['a2m_ckpt'], inp['head_ckpt'], inp['torso_ckpt'], inp=inp)
# 设置随机种子
video_forging.prepare_for_infer(inp)

def my_responese(status = "success", message = "", data = ""):
    return jsonify({"status": status, "message": message, "data": data}), 200


def my_forge(cur_inp):
    inp.update(cur_inp)
    video_forging.infer_once(inp)

@app.route('/api/video-forge/sample', methods=['POST'])
def video_forge():
    try:
        person = request.form.get('person', None)             # 必为None, 'person{1,2,3}', base64这三者之一
        bg = request.form.get('background', None)             # 必为None, 'bg{1,2,3}', base64这三者之一
        aud = request.form.get('audio', None)                 # 必为None, 'aud{1,2,3}', base64这三者之一
        person_file_name = person
        bg_file_name = bg
        aud_file_name = aud
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 用于唯一保存各时刻用户上传的文件
        
        # 判断人物是否为空&是否在指定列表内
        if person is None:
            return my_responese(status="error", message="Must specifiy a person.")
        else:
            # 若person不为空且不为'person{1,2,3}', 尝试解码并打开person图像, 若成功则存图, 否则返回提示信息
            if person not in person_list:
                try:
                    image_tmp = Image.open(BytesIO(base64.b64decode(person)))
                    person_path = os.path.join(root_path, f"person_{timestamp}.png")
                    image_tmp.save(person_path)
                    person_file_name = f"person_{timestamp}"
                except Exception:
                    return my_responese(status="error", message="Fail to resolve the given base64 person image, please try again.")

            
        # 判断背景是否为空&是否在指定列表内
        if bg is None:
            return my_responese(status="error", message="Must specifiy a background.")
        else:
            if bg not in bg_list:
                try:
                    bg_tmp = Image.open(BytesIO(base64.b64decode(bg)))
                    bg_path = os.path.join(root_path, f"bg_{timestamp}.png")
                    bg_tmp.save(bg_path)
                    bg_file_name = f"bg_{timestamp}"
                except Exception:
                    return my_responese(status="error", message="Fail to resolve the given base64 background image, please try again.")

        # 判断音频是否为空&是否在指定列表内
        if aud is None:
            return my_responese(status="error", message="Must specifiy a audio.")
        else:
            if aud not in aud_list:
                try:
                    aud_tmp = AudioSegment.from_file(BytesIO(base64.b64decode(aud)), format="mp3")
                    aud_path = os.path.join(root_path, f"aud_{timestamp}.mp3")
                    aud_tmp.export(aud_path, format="mp3")
                    aud_file_name = f"aud_{timestamp}"
                except Exception:
                    return my_responese(status="error", message="Fail to resolve the given base64 driving audio, please try again.")

        video_name = f"{timestamp}.mp4"

        # 手动补充其他推理所需的args
        cur_inp = {
            'src_image_name': person_file_name + '.png',
            'bg_image_name': bg_file_name + '.png',
            'drv_audio_name': aud_file_name + '.mp3',
            'out_name': video_name,
        }

        # 调用函数执行前向过程
        my_forge(cur_inp)

        video_path = os.path.join(root_path, video_name)
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
        
        base64_video_data = base64.b64encode(video_data).decode('utf-8')
        
        # 返回编码后的视频数据
        return my_responese(status="success", message="", data=base64_video_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':

    app.run(host="127.0.0.1", port=6006)



