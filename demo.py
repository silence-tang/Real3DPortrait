from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import os
from datetime import datetime
import argparse
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


@app.route('/video-forge/sample', methods=['POST'])
def video_forge():
    try:
        # person = request.get_json().get('person', None)
        # bg = request.get_json().get('background', None)
        # aud = request.get_json().get('audio', None)
        person = request.form.get('person', None)
        bg = request.form.get('background', None)
        aud = request.form.get('audio', None)
        print('person:', person)
        print('background:', bg)
        print('audio:', aud)

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 用于唯一保存每次用户上传的文件
        
        # 判断人物是否为空&是否在指定列表内
        if person is None:
            return my_responese(status="error", message="Must specifiy a person.")
        else:
            if person not in person_list:
                return my_responese(status="error", message="No such person in database, choose from this list: [person1, person2, person3].")
        
        # 判断背景是否为空&是否在指定列表内
        if bg is None:
            return my_responese(status="error", message="Must specifiy a background.")
        else:
            if bg not in bg_list:
                return my_responese(status="error", message="No such background in database, choose from this list: [bg1, bg2, bg3].")

        # 判断音频是否为空&是否在指定列表内
        if aud is None:
            return my_responese(status="error", message="Must specifiy a audio.")
        else:
            if aud not in aud_list:
                return my_responese(status="error", message="No such audio in database, choose from this list: [aud1, aud2, aud3].")

        # TODO: 处理上传图片/音频的情况
        # if person is not None:
        #     src = person_dict[person]
        
        # else:
        #     # 根据post过来的base64解码出用户上传的图片并保存在服务器端
        #     src = os.path.join(folder_path, f"{timestamp}_src{file_extension}")
        #     image_tmp = Image.open(BytesIO(base64.b64decode(source)))
        #     image_tmp.save(src)
        
        # 根据post过来的base64解码出用户上传的图片并保存在服务器端
        # tgt = os.path.join(folder_path, f"{timestamp}_tgt{file_extension}")
        # image_tmp = Image.open(BytesIO(base64.b64decode(target)))
        # image_tmp.save(tgt)

        # output_tmp = os.path.join(folder_path, f"{timestamp}_op.mp4")
        # output_real = os.path.join(folder_path, f"{timestamp}_op.jpg")


        # 根据解析得到的字段名取得相应的文件路径
        # person_path = os.path.join(root_path, person) + '.jpg'
        # bg_path = os.path.join(root_path, bg) + '.jpg'
        # aud_path = os.path.join(root_path, aud) + '.mp3'
        video_name = person + '_' + bg + '_' + aud + '.mp4'
        
        # 手动补充其他推理所需的args
        cur_inp = {
            'src_image_name': person + '.png',
            'bg_image_name': bg + '.png',
            'drv_audio_name': aud + '.mp3',
            'out_name': video_name,
        }

        # 调用函数执行前向过程
        my_forge(cur_inp)

        # img = Image.open(output_rea)
        # buffered = BytesIO()
        # img.save(buffered, format="JPG")
        # img_byte_data = buffered.getvalue()
        # img_base64 = base64.b64encode(img_byte_data).decode('utf-8')

        # video_path = os.path.join(root_path, video_name)
        # video_clip = VideoFileClip(video_path)

        # # 将视频数据存入字节流
        # buffered = BytesIO()
        # video_clip.write_videofile(buffered, codec="libx264", audio_codec="aac", format="mp4")
        # # 获取视频的字节数据
        # video_byte_data = buffered.getvalue()
        # # 将字节数据编码为Base64
        # video_base64 = base64.b64encode(video_byte_data).decode('utf-8')

        video_path = os.path.join(root_path, video_name)
        print('video_path', video_path)

        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()

        base64_video_data = base64.b64encode(video_data).decode('utf-8')
        
        # 返回编码后的视频数据
        return my_responese(status="success", message="", data=base64_video_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    
    # app.run(host="127.0.0.1", port=6006)
    app.run(host="127.0.0.1", port=5002)