"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
from itertools import count
import os
from unicodedata import name
from PIL import Image, ImageDraw, ImageFont
import cv2
import flask
import numpy as np
import time

import torch

from flask import Flask, render_template, request, redirect, Response

app = Flask(__name__)


#'''
# Load Pre-trained Model
# model = torch.hub.load(
#         "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
#         )#.autoshape()  # force_reload = recache latest code
#'''
# Load Custom Model
model = torch.hub.load("ultralytics/yolov5", "custom", path = "./yolov5s.pt", force_reload=True)
# model = torch.hub.load("ultralytics/yolov5", "custom", path = "./best.pt", force_reload=True)

# Set Model Settings
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.4  # NMS IoU threshold (0-1) 

from io import BytesIO

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判斷是否OpenCV圖片型別
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "msjhbd.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def gen():
    # cap=cv2.VideoCapture('n.jpg')
    
    cap=cv2.VideoCapture('queue02.mp4')
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            
            
            # print(results)            
            # print(results.pandas().xyxy[0])
            df=results.pandas().xyxy[0]
            # print(df)
            
            df=df[df['name']=='person'].loc[:,['name']].count()
            # print(df)
            for name in df:
            
                print(name)
            
            # print(name)
            # results.pandas()xyxy[0].to_json(orient="records")
            #results.render()  # updates results.imgs with boxes and labels
            # print(results.render())
            # results.print()  # print results to screen

            # results.show() 
            # print(results.imgs)
            #print(type(img))
            #print(results)
            # plt.imshow(np.squeeze(results.render()))
            # print(type(img))
            # print(img.mode)
            
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
           
            img_BGR=cv2ImgAddText(img_BGR, f'場域內共{name}人', 10, 10, (255, 0, 0), 40)
            #print(type(img))
            #print(img.shape)
            #frame = img
            #ret,buffer=cv2.imencode('.jpg',img)
            #frame=buffer.tobytes()
            #print(type(frame))
            #for img in results.imgs:
                #img = Image.fromarray(img)
            #ret,img=cv2.imencode('.jpg',img)
            #img=img.tobytes()

            #encode output image to bytes
            #img = cv2.imencode('.jpg', img)[1].tobytes()
            #print(type(img))
        else:
            break
        #print(cv2.imencode('.jpg', img)[1])

        #print(b)
        #frame = img_byte_arr
        
        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
       



@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(gen(),mimetype='text/plain')

'''                        
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
'''
'''
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=640)

        # for debugging
        # data = results.pandas().xyxy[0].to_json(orient="records")
        # return data

        results.render()  # updates results.imgs with boxes and labels
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")

    return render_template("index.html")
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    '''
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
    ).autoshape()  # force_reload = recache latest code
    model.eval()
    '''
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

# Docker Shortcuts
# docker build --tag yolov5 .
# docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device="/dev/video0:/dev/video0" yolov5
