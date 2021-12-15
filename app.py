from flask import Flask, json, render_template,Response
import cv2

from detect_mask_video import detect_and_predict_mask

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'secret!'

camera = False
detection_status = True
camera_status = False
def generate_frames():
    while True:
        global camera_status
        success, frame = camera.read()
        count = 0

        if not success:
            camera_status = False
            break
        else:
            camera_status = True

            (locs, preds) = detect_and_predict_mask(frame)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                
                if withoutMask > mask:
                    count = count + 1

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            global detection_status
            if count > 0:
                detection_status = False
                count = 0
            else:
                detection_status = True

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def init_camera():
    global camera
    camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    init_camera()
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def update_detection_status():
    response = app.response_class(
        response=json.dumps({'status': detection_status}),
        mimetype='application/json'
    )
    return response

@app.route('/camera')
def update_camera_status():
    response = app.response_class(
        response=json.dumps({'status': camera_status}),
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    app.run(debug = True)
    