import cv2
import os
import sys
from faced import FaceDetector
from faced.utils import annotate_image

if len(sys.argv) < 3:
    print('rutas de videos no entregadas')
    exit(-1)

face_detector = FaceDetector()

DEBUG = bool(int(0 if os.getenv('DEBUG') is None else os.getenv('DEBUG')))
BLUR = int(23 if os.getenv('BLUR') is None else os.getenv('BLUR'))

video = cv2.VideoCapture(sys.argv[1])

x_offset = 50
y_offset = 50

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

video_output = cv2.VideoWriter(sys.argv[2], 0x7634706d, fps, (frame_w, frame_h))

def limit(x, offset, _max, _type):
    if x + offset > _max:
        return (_max - 1)
    elif x - offset < 0:
        return 0
    else:
        if _type == 1:
            return x + offset
        else:
            return x - offset


while video.isOpened():
    ret, frame = video.read()

    if ret is False:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bboxes = face_detector.predict(frame, 0.6)

    for (x, y, w, h, acc) in bboxes:
        x = x - (w // 2)
        y = y - (h // 2)
        sub_face = frame[
            limit(y, y_offset, frame_h, 0):limit(y + h, y_offset, frame_h, 1),
            limit(x, x_offset, frame_w, 0):limit(x + w, x_offset, frame_w, 1)
        ]
        sub_face = cv2.GaussianBlur(sub_face, (BLUR, BLUR), 30)
        frame[limit(y, y_offset, frame_h, 0):limit(y + h, y_offset, frame_h, 1), limit(x, x_offset, frame_w, 0):limit(x + w, x_offset, frame_w, 1)] = sub_face

    video_output.write(frame)
    if DEBUG:
        cv2.imshow('video', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

video.release()
video_output.release()
cv2.destroyAllWindows()