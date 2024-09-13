from flask import Flask, request, jsonify, render_template, Response, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from yolo import *
from flask_cors import CORS
import base64
from flask_mail import Mail, Message
app = Flask(__name__)
CORS(app)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Your mail server
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '*******@gmail.com'
app.config['MAIL_PASSWORD'] = 'zmhb **** **** evhv'
app.config['MAIL_DEFAULT_SENDER'] = '********@gmail.com'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
           
mail = Mail(app)

# def send_email(subject, recipient, body):
#     msg = Message(subject, recipients=[recipient])
#     msg.body = body
#     mail.send(msg)

mail_alert = None

email_sent = False

@app.route('/send_email', methods=['POST'])
def send_email():
    global email_sent, mail_alert
    # try:
    if email_sent:
        # If an email has already been sent, do not send another one.
        return jsonify({'message': 'Email already sent'}), 200

    if  mail_alert != None:
        message_txt = "ACTION Alert....!!!!"
        body_txt = "A Harmful Event has been detected."
        msg = Message(message_txt, 
                    # recipients=["shenoyradhikav@gmail.com"],
                    recipients=["thejeshnaidu555@gmail.com"],
                    body=body_txt)
        mail.send(msg)
        email_sent = True
        return jsonify({'message': 'Email sent successfully'}), 200
    # except Exception as e:
    else:
        return jsonify({'message': 'Email failed to send', 'error': str(e)}), 500


def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/video_page')
def video_page():
    return render_template('video.html')


# Configuration
UPLOAD_FOLDER = 'uploads'
latest_detection = None
fire_detection_count = 0
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','mp4','mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/latest_detection')
def get_latest_detection():
    global latest_detection
    if latest_detection is None:
        return jsonify({'error': 'No detection available'}), 404
    return jsonify({'latest_detection': latest_detection})

@app.route('/fire_detection_count')
def get_fire_detection_count():
    global fire_detection_count
    return jsonify({'fire_detection_count': fire_detection_count})


def gen():
    global video_capture
    global latest_detection
    global fire_detection_count, mail_alert
    send = False
    model, classes, colors, output_layers = load_yolo()
    while True:
        success, frame = video_capture.read()
        if frame is None:
            fire_detection_count = 0
            break

        class_label = 'None'
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        
        if class_ids != []:
            class_label = classes[class_ids[0]]
            latest_detection = class_label
            mail_alert =  class_label
            fire_detection_count += 1
        print(class_label)

        # if class_label == "Fire" and send != True:
        #     send_email("Fire Alert", "thejeshnaidu555@gmail.com", "A fire has been detected.")
        #     send = True
        try:
            img = draw_labels(boxes, confs, colors, class_ids, classes, frame)
        except:
            continue
        ret, jpeg = cv2.imencode('.jpg', img)
        if not ret:
            latest_detection = None
            fire_detection_count = 0
            break
        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            latest_detection = None
            fire_detection_count = 0
            break

        frame = jpeg.tobytes()

        b64_frame = base64.b64encode(frame).decode('utf-8')
        # yield f'data:{{"frame":"{b64_frame}", "class_label":"{class_label}"}}\n\n'
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/upload', methods=['POST'])
def upload_file():
    global video_capture, fire_detection_count, latest_detection, mail_alert, email_sent
    mail_alert = None
    email_sent = False
    fire_detection_count = 0
    latest_detection = None
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # [os.remove(file) for file in os.listdir('.') if file.endswith('.jpg')]

        video_capture = cv2.VideoCapture(filepath)

        return jsonify({'message': 'File uploaded successfully'}), 200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/video_feed', methods=['GET'])
def video_feed():
    global video_capture
    if video_capture is None:
        return jsonify({'error': 'No video uploaded'}), 400

    # return Response(gen(), mimetype='text/event-stream')
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


