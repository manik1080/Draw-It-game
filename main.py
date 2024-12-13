import cv2 as cv
from datetime import datetime
import numpy as np
import mediapipe as mp
import random
import tensorflow as tf

class DrawGuesserGame:
    def __init__(self):
        self.camw, self.camh = 640, 480
        self.cam = cv.VideoCapture(0)
        self.cam.set(3, self.camw)
        self.cam.set(4, self.camh)
        self.hand_tracker = mp.solutions.hands.Hands(min_detection_confidence=0.8)
        self.interpreter = tf.lite.Interpreter(model_path=r"model/drawGuesser_lite.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.xp, self.yp = 0, 0
        self.drawColour = (255, 255, 255)
        self.thickness = 5
        self.score = 0
        self.imgCanvas = np.zeros((self.camh, self.camw, 3), np.uint8)
        self.decoder = {
            'airplane': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'ambulance': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'apple': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'axe': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'banana': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'basket': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            'bed': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'carrot': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            'cat': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            'fish': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        }

    def decode(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (64, 64))
        img = np.expand_dims(img, axis=(0, -1))
        input_data = img.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        model_out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        prediction = np.argmax(model_out)
        prediction_score = model_out[prediction]

        return list(self.decoder.keys())[prediction], prediction_score

    @staticmethod
    def find(ind, hand, img):
        lm = hand.landmark[ind]
        h, w, _ = img.shape
        return int(lm.x * w), int(lm.y * h)

    def run(self):
        duration = 20
        press = 0
        start_time = datetime.now()

        choice = random.choice(list(self.decoder.keys()))

        while True:
            stat, img = self.cam.read()
            imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = self.hand_tracker.process(imgRGB)
            img = cv.flip(cv.addWeighted(img, 0.5, self.imgCanvas, 0.8, 0), 1)

            cv.putText(img, str(self.score), (450, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            if press:
                diff = (datetime.now() - start_time).seconds
                cv.putText(img, str(diff), (310, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

                if diff <= duration:
                    if results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]
                        tip, bottom = self.find(8, hand, img), self.find(6, hand, img) # index finger
                        head2, butt2 = self.find(16, hand, img)[1], self.find(14, hand, img)[1] # ring finger

                        if tip[1] < bottom[1] and head2 > butt2:
                            head1, butt1 = self.find(12, hand, img)[1], self.find(10, hand, img)[1] # middle finger
                            if head1 < butt1 and head2 > butt2:
                                self.drawColour = (0, 0, 0)
                                self.thickness = 55
                            else:
                                self.drawColour = (255, 255, 255)
                                self.thickness = 15

                            if self.xp == self.yp == 0:
                                self.xp, self.yp = tip

                            cv.line(self.imgCanvas, (self.xp, self.yp), tip, self.drawColour, self.thickness)
                            self.xp, self.yp = tip
                        else:
                            self.xp, self.yp = 0, 0

                    cv.putText(img, choice, (230, 390), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

                else:
                    pred, pred_score = self.decode(self.imgCanvas)
                    print(pred, pred_score)
                    if pred == choice and pred_score > 0.4:
                        self.score += 1
                        print("WELL DONE! Updated score:", self.score)
                    else:
                        print("Oopsie! Current score:", self.score)

                    choice = random.choice(list(self.decoder.keys()))
                    self.imgCanvas = np.zeros((self.camh, self.camw, 3), np.uint8)
                    press = 0
            else:
                start_time = datetime.now()
                cv.putText(img, 'Press R for new word, Q to quit', (110, 220), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow("Video Capture", img)
            k = cv.waitKey(10)

            if k & 0xFF == ord("r"):
                press = 1
            elif k & 0xFF == ord("q"):
                break

        self.cam.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    game = DrawGuesserGame()
    game.run()
