import cv2
import os
import sys

if len(sys.argv) < 3:
    print('rutas de videos no entregadas')
    exit(-1)

DEBUG = bool(int(0 if os.getenv('DEBUG') is None else os.getenv('DEBUG')))
BLUR = int(23 if os.getenv('BLUR') is None else os.getenv('BLUR'))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
head_cascade = cv2.CascadeClassifier('HS.xml')
face2_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

video = cv2.VideoCapture(sys.argv[1])

x_offset = 0
y_offset = 0

frame_index = 0

frame_w = int(video.get(3))
frame_h = int(video.get(4))

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

video_output = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_w, frame_h))

def limit(x, offset, _max, _type):
    if x + offset > _max:
        return _max
    elif x - offset < 0:
        return 0
    else:
        if _type == 1:
            return x + offset
        else:
            return x - offset


while video.isOpened():
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 8)
    faces2 = face2_cascade.detectMultiScale(gray, 1.1, 2)
    heads = head_cascade.detectMultiScale(gray, 1.1, 1)

    for (x, y, w, h) in [*faces, *faces2, *heads]:
        sub_face = frame[
            limit(y, y_offset, frame_h, 0):limit(y + h, y_offset, frame_h, 1),
            limit(x, x_offset, frame_w, 0):limit(x + w, x_offset, frame_w, 1)
        ]
        sub_face = cv2.GaussianBlur(sub_face, (BLUR, BLUR), 30)
        frame[limit(y, y_offset, frame_h, 0):limit(y, y_offset, frame_h, 0)+sub_face.shape[0], limit(x, x_offset, frame_w, 0):limit(x, x_offset, frame_w, 0)+sub_face.shape[1]] = sub_face

    video_output.write(frame)
    if DEBUG:
        cv2.imshow('video', frame)
        cv2.waitKey(1)

video.release()
video_output.release()
cv2.destroyAllWindows()