import cv2 as cv
import asyncio
import threading
import time
import argparse

from face_lib import *

def main(args):
    ## select face detection model
    # todo : arg parser
    backend = select_backend('mtcnn')
    if not args.net == 'mobilefacenet':
        verification_method= select_method('classification')
        model, targets, names = None,None,None
    else:
        model, targets, names, verification_method= select_method('arcface',args)
    
    if args.type == 'webcam':
        ## cam load
        cam = open_webcam(1)
        
        ## main loop
        while(True):
            t1 = time.time()
            ret, frame = cam.read()
            if not ret:
                raise SystemError("Could not read frame")

            ## detection ouput : NDarray crop images
            if not args.net == 'mobilefacenet':
                detected = backend(frame)
            else: detected = None
            
            ## verification 
            frame = verification_method(frame,detected,model,targets,names,args)

            ## show
            cv.imshow('frame',frame)

            t2 = time.time()
            print('1 iter time : ',t2-t1)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv.destroyAllWindows()
    else:
        test_image = cv.imread(r'build\facebank\id2\2023-02-15-15-01-07.jpg')
        # test_image = cv.cvtColor(test_image,cv.COLOR_BGR2RGB)
        if not args.net == 'mobilefacenet':
            detected = backend(test_image)
        else: 
            detected = None
            
        ## verification 
        frame = verification_method(test_image,detected,model,targets,names,args)

        ## show
        while True:
            cv.imshow('frame',frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows()

if __name__ == '__main__':
    ## args 
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-t", "--type", help="input type [webcam, image]",default='image',type=str)
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-n", "--net", help="which network, [mobilefacenet, ]",default='mobilefacenet', type=str) # todo : classification method
    args = parser.parse_args()

    main(args)