import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load a lightweight YOLOv8 model for speed
model = YOLO("yolov8n.pt") 

def generate_frames():
    # Open the webcam (0 is usually the built-in webcam)
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Run YOLO inference on the live frame
            results = model(frame, stream=True)
            
            # Plot the bounding boxes directly onto the frame
            for r in results:
                frame = r.plot() 
                
            # Encode the processed frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Use 'multipart/x-mixed-replace' to stream frames continuously
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. The HTML img tag points to this."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run the web server locally
    app.run(host='0.0.0.0', port=5000, debug=True)
