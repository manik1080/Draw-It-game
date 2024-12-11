import cv2 as cv
from datetime import datetime
import numpy
import mediapipe
import random
import tensorflow as tf
import numpy

decoder = {'airplane': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'ambulance': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'apple': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'axe': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'banana': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'basket': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
 'bed': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
 'carrot': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
 'cat': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
 'fish': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}

# model = tf.keras.models.load_model('drawGuesser_lite.tflite', compile=False)
interpreter = tf.lite.Interpreter(model_path="model/drawGuesser_lite.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def decode(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (64, 64))
    cv.imshow('img', img)
    img = numpy.expand_dims(img, 0)
    img = numpy.expand_dims(img, -1)
    input_shape = input_details[0]['shape']
    input_data = img.astype(numpy.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    model_out = interpreter.get_tensor(output_details[0]['index'])[0]
    prediction = numpy.argmax(model_out, axis=0)
    prediction_score = model_out[prediction]
    return (list(decoder.keys())[prediction], prediction_score)


if __name__ == '__main__':

    camw, camh = 640, 480
    cam = cv.VideoCapture(0)
    cam.set(3, camw)
    cam.set(4, camh)

    hand_tracker = mediapipe.solutions.hands
    hands = hand_tracker.Hands(min_detection_confidence=0.8)
    duration = 20
    press = 0
    start_time = datetime.now()

    def find(ind, Hand):
        id, lm = tuple(enumerate(Hand.landmark))[ind]
        h, w, c = img.shape
        X, Y = int(lm.x * w), int(lm.y * h)
        return [X, Y]


    xp, yp = 0, 0
    drawColour = (255, 255, 255)
    thickness = 5
    score = 0
    imgCanvas = numpy.zeros((camh, camw, 3), numpy.uint8)
    choice = random.choice(list(decoder.keys()))
    while True:
        stat, img = cam.read()
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        img = cv.flip(cv.addWeighted(img, 0.5, imgCanvas, 0.8, 0), 1)
        cv.putText(img, str(score), (450,30), cv.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv.LINE_AA)

        if press:
            diff = (datetime.now() - start_time).seconds
            cv.putText(img, str(diff), (310,32), cv.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv.LINE_AA)

            if diff<=duration:
                diff = (datetime.now() - start_time).seconds
                if results.multi_hand_landmarks:

                    hand = results.multi_hand_landmarks[0]
                    tip, bottom = find(8, hand), find(6, hand)  # index finger
                    head2, butt2 = find(16, hand)[1], find(14, hand)[1]  # ring finger

                    if tip[1] < bottom[1] and head2 > butt2:
                        head1, butt1 = find(12, hand)[1], find(10, hand)[1]  # middle finger
                        if head1 < butt1 and head2 > butt2:
                            drawColour = (0, 0, 0)
                            xp, yp = tip[0], tip[1]
                            thickness = 55
                        else:
                            drawColour = (255, 255, 255)
                            thickness = 15
                        if xp == yp == 0:
                            xp, yp = tip[0], tip[1]

                        # cv.circle(img, (tip[0], tip[1]), thickness+2, drawColour, cv.FILLED)
                        # cv.line(img, (xp, yp), (tip[0], tip[1]), drawColour, thickness)
                        cv.line(imgCanvas, (xp, yp), (tip[0], tip[1]), drawColour, thickness)

                        xp, yp = tip[0], tip[1]
                    else:
                        xp, yp = 0, 0

                cv.putText(img, choice, (230,390), cv.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv.LINE_AA)

            else:
                pred, pred_score = decode(imgCanvas)
                print(pred, pred_score)
                if pred==choice and pred_score>0.4:
                    score += 1
                    print("WELL DONE! Updated score: ", score)
                else:
                    print("Oopsie! Current score: ", score)
                choice = random.choice(list(decoder.keys()))
                imgCanvas = numpy.zeros((camh, camw, 3), numpy.uint8)
                press = 0
            
        else:
            start_time = datetime.now()
            cv.putText(img, 'press r for new word, q to quit', (110,220), cv.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow("Video Capture", img)
        k = cv.waitKey(10)
        if k & 0xFF == ord("r"):
            press = 1
        if k & 0xFF == ord("q"):
            qu = 1
            break

    cam.release()
    cv.destroyAllWindows()










    
