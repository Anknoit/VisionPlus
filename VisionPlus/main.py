import cv2 as cv
import imutils
import numpy as np
import argparse

# HOGCV =cv.HOGDescriptor()   #Pretrained model for humandeeector of OpenCV
# HOGCV.setSVMDetector(cv.HOGDescripor_getDefaultPeopleDetector()) #Feeding Support Vector Machine(SVM) with it
# #SVM is a supervised algorithm that differentiate the dataset by plotting each feature of dataset in  x-y plane and classify using linear or any model that differentiate itperfectly

HOGCV = cv.HOGDescriptor()
HOGCV.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride = (4,4), padding = (8,8), scale=1.03)
    person =1
    for x,y,w,h in bounding_box_cordinates:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv.putText(frame, f'person {person}', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        person +=1
    cv.putText(frame, 'Status: Detecting People', (40,40), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 0,0), 2)
    cv.putText(frame, f'TOTAL People: {person-1}', (40,70), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 0,0), 2)
    cv.imshow('output', frame)

    return frame

def humanDetector(args):
    image_path = args["image"]
    video_path = args["video"]
    if str(args["camera"]) == 'True' : camera=True
    else :camera =False
    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv.VideoWriter(args['output'],cv.VideoWriter_fourcc(*'MJPG'), 10, (600,600))
    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])


def detectByPathVideo(path, writer):

    video = cv.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while video.isOpened():
        #check is True if reading was successful 
        check, frame =  video.read()

        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            frame = detect(frame)
            
            if writer is not None:
                writer.write(frame)
            
            key = cv.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()

def detectByCamera(writer):   
    video = cv.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame = video.read()

        frame = detect(frame)
        if writer is not None:
            writer.write(frame)

        key = cv.waitKey(1)
        if key == ord('q'):
                break

    video.release()
    cv.destroyAllWindows()
def detectByPathImage(path, output_path):
    image = cv.imread(path)
    image = imutils.resize(image, width = min(800, image.shape[1])) 
    result_image = detect(image)
    if output_path is not None:
        cv.imwrite(output_path, result_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())
    return args
if __name__ == "__main__":

    HOGCV = cv.HOGDescriptor()
    HOGCV.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    args = argsParser()
    humanDetector(args)

# 1. To give video file as input:
# python main.py -v ‘Path_to_video’

# 2. To give image file as input:
# python main.py -i ‘Path_to-image’

# 3. To use the camera:
# python main.py -c True

# 4. To save the output:
# Python main.py -c True -o ‘file_name’    