import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

def box_plot(img): # function for finding area that represents face detection and blur of that area
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faces.process(img_rgb) # function process needs RGB image
    if result.detections is not None:
        print(f'Number of registered faces: {len(result.detections)}')
        for detection in result.detections:
            location_data = detection.location_data
            box = location_data.relative_bounding_box

            H, W, _ = img.shape

            x, y, w, h = box.xmin, box.ymin, box.width, box.height

            x1 = int(x * W)
            y1 = int(y * H)
            w = int(w * W)
            h = int(h * H)

            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (25, 25)) # Area in which face is found will be blurred
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 0, 255), 4)
    else:
        print('Number of registered faces: 0')
    return img


mp_faces_detect = mp.solutions.face_detection
faces = mp_faces_detect.FaceDetection(min_detection_confidence=0.5) # This object will be used for detecting face

cond = 0 # Help variable for while loop condition. Varibale cond will be set to 1 if anonymisation is done
while cond == 0: # This condition assures that user must type one of options described in input
    user_input = input('Types of data that can be anonymised:\n' 
                       '- webcam \n- video \n- image?'
                       '\nType your selection: ')

    if user_input.lower() == "webcam": # User wants to anonymise data from webcam
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()

            img = box_plot(img)
            cv2.imshow('RESULT', img)
            if cv2.waitKey(5) == ord('q'): # User needs to type q on keyboard to exit webcam mode
                break

        cap.release()
        cv2.destroyAllWindows()
        cond = 1
    elif user_input.lower() == 'video': # User wants to anonymise data from video
        cap = cv2.VideoCapture('./content/vukasin.mp4')
        ret = True
        while ret:
            ret, img = cap.read()

            img = box_plot(img)
            cv2.imshow('RESULT', img)
            if cv2.waitKey(20) == ord('q'): # User can type q on keyboard to exit video mode
                break

        cap.release()
        cv2.destroyAllWindows()
        cond = 1
    elif user_input.lower() == 'image': # User wants to anonymise data from image
        image = cv2.imread('./content/img.jpg')

        image = box_plot(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(image)
        plt.show()
        cond = 1
    else:
        print('You typed unsupported data type. Try again!')
