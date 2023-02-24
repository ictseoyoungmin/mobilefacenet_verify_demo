import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import torch
import cv2 as cv
from PIL import Image
# import cvlib as cvl

# my custom
from face_lib import *

# [insightface]
from mtcnn import *
from model import MobileFaceNet
from config import get_config

conf = get_config(False)

device = conf.device#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
mtcnn = MTCNN()
print(device)

def draw_box_name(bbox,frame,name='None'):
    frame = cv.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
    frame = cv.putText(frame,
                    name,
                    (bbox[0],bbox[1]), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (0,255,0),
                    2,
                    cv.LINE_AA)
    return frame


if __name__ == '__main__':
    backend = MTCNN()
    model = MobileFaceNet(512).to(conf.device)
    model.load_state_dict(torch.load(r'build\pre_trained\mfn_2023-02-02_acc0.9290.pth'))
    tta = True
    conf.threshold = 0.8
    targets, names = prepare_facebank(conf, model, backend, tta = tta)
    ## cam load
    cam = open_webcam(1)
    
    ## main loop
    while(True):
        ret, frame = cam.read()
        if not ret:
            raise SystemError("Could not read frame")

        ## detection ouput : NDarray crop images
        img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        ##############################################

        boxes, landmarks = backend.detect_faces(img)
        print(np.array(boxes).shape)

        #################################################
       
        if len(boxes) > 0:
            faces = []
            for landmark in landmarks:
                facial5points = [[landmark[j],landmark[j+5]] for j in range(5)]
                warped_face = warp_and_crop_face(np.array(img), facial5points, get_reference_facial_points(default_square= True), crop_size=(112,112))
                faces.append(Image.fromarray(warped_face))
            
            results, score = infer(model,conf,faces,targets,tta=tta)
            for idx,face in enumerate(boxes):
                face = face.astype(np.int16) 
                # print(names)
                frame = draw_box_name(face,frame,names[results[idx]+1] + '_{:.2f}'.format(score[idx]))
        ## show
        cv.imshow('frame',frame)


        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()