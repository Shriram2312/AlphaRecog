import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc, mediapipe_utils
from model import KeyPointClassifier

import csv
import copy
import argparse
import itertools

from collections import deque
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

    
def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    sign_history = deque(maxlen=30)

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/labels_alpha.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    mode = 0
    fc = 0
    word = []
    no_hand = 0
    while True:
        fps = cvFpsCalc.get()

        
        key = cv.waitKey(10)
        if key == 27:  
            break

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  
        debug_image = copy.deepcopy(image)

        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            hand = results.multi_handedness
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):

                brect = mediapipe_utils.calc_bounding_rect(debug_image, hand_landmarks)
                
                landmark_list = mediapipe_utils.calc_landmark_list(debug_image, hand_landmarks)

                
                pre_processed_landmark_list = mediapipe_utils.pre_process_landmark(
                    landmark_list)

                debug_image = mediapipe_utils.draw_landmarks(debug_image, landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)


                debug_image = mediapipe_utils.draw_info_text(
                    debug_image,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id]
                )
                sign_history.append(keypoint_classifier_labels[hand_sign_id])
            if fc !=0 and fc % 64 == 0:
                most_common_letter = Counter(sign_history).most_common()
                if len(most_common_letter):
                    word.append(most_common_letter[0][0])                
                sign_history.clear()
                # print("Sign history most common....", sign_history)
            if fc % 16 == 0 and hand[0].classification[0].label[0:] == "Right":
                # Raise right hand to delete a letter
                word = word[:-1]
        else:
            no_hand+=1

        

        if no_hand == 100:
            no_hand = 0
            sign_history.clear()

        debug_image = mediapipe_utils.draw_word(debug_image,word)
        # debug_image = mediapipe_utils.draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand shape recognition', debug_image)
        fc+=1

    cap.release()
    cv.destroyAllWindows()

                


if __name__ == "__main__":
    main()

