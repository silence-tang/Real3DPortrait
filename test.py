from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

def my_responese(status = "success", message = "", data = ""):
    return jsonify({"status": status, "message": message, "data": data}), 200

@app.route('/greet', methods=['POST'])
def greet_user():
    # 从请求的JSON数据中解析出'name'字段
    data = request.get_json()
    name = data.get('name')
    
    # 构建响应信息
    response = f"Hello, {name}"

    with open('/root/autodl-tmp/.autodl/Real3DPortrait/id1_bg1_aud1.mp4', 'rb') as video_file:
        video_data = video_file.read()

    base64_data = base64.b64encode(video_data).decode('utf-8')
    
    # 返回响应
    return jsonify(status="success", message=response, data=base64_data)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=6006)
