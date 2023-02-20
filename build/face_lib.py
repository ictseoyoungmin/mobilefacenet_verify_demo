import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import torch
from torchvision import transforms
import cv2 as cv
from PIL import Image
import cvlib as cvl

# my custom
# Use '.' to reference in ipynb.
from nets import OOnet,OOnetFCN,CustomResNet

# [insightface]
from mtcnn import MTCNN
from model import MobileFaceNet,Arcface
from config import get_config
conf = get_config(False)

device = conf.device#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
mtcnn = MTCNN()
print(device)

clf_mode = 1

### clf model
if clf_mode == 1:
    clf_model = OOnetFCN(3).to(device)
    clf_model.load_state_dict(torch.load(r'build/pre_trained\oonet_123.pth'))
    # clf_model.load_state_dict(torch.load(r'result\012\oonet_0.8281.pth'))
elif clf_mode == 2:
    clf_model = OOnet(10).to(device)
    clf_model.load_state_dict(torch.load(r'build/pre_trained\oonet_fc_0.8697_4.pth'))
elif clf_mode == 3:
    clf_model = CustomResNet(10).to(device)
    clf_model.load_state_dict(torch.load(r'build/pre_trained\oonet_fc_0.6380_49.pth'))
# clf_model.to(device)

### arc model
# mobilefacenet = MobileFaceNet(512).to(device)


# 보류
{
# async def main():
#     ## cam load
#     cam = open_webcam()
    
#     ## select face detection model
#     backend = select_backend('mtcnn')

#     ## main loop
#     while(True):
#         t1 = time.time()
#         ret, frame = cam.read()

#         if not ret:
#             print("Could not read frame")
#             exit()

#         ## detection ouput : NDarray
#         detected = backend(frame)

#         ## verification 


#         ## show
#         cv.imshow('frame',detected)

#         t2 = time.time()
#         print('1 iter time : ',t2-t1)
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break

#     cam.release()
#     cv.destroyAllWindows()

# async def open_loop():
#     asyncio.create_task(main())

# def start():
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(open_loop())
#     loop.close()
}


def open_webcam(device_number=1):
    # open webcam
    webcam = cv.VideoCapture(device_number) # 0 or 1
    webcam.set(3, 640) 
    webcam.set(4, 400) 
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()
    
    return webcam

def detect_face(frame):
    ## face detection
    face, confidence = cvl.detect_face(frame)
    ##
    for idx, f in enumerate(face):
                
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        
        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
            
            ## detected face frame and preprocessing for network input
            face_region = frame[startY:endY, startX:endX]
            face_region1 = cv.resize(face_region, (112, 112), interpolation = cv.INTER_AREA)
            ##

            ## face verification
            prediction =  1#np.argmax(model.predict(x))
            
            if prediction == 2: # 마스크 미착용으로 판별되면, 
                cv.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No Mask " #({:.2f}%)".format((1 - prediction[0][0])*100)
                cv.putText(frame, text, (startX,Y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
            elif prediction == 1: # 마스크 착용으로 판별되면
                cv.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Mask "#({:.2f}%)".format(prediction[0][0]*100)
                cv.putText(frame, text, (startX,Y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            else : # 올바르지 않는 착용
                cv.rectangle(frame, (startX,startY), (endX,endY), (0,255,255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Incorrect Mask "
                cv.putText(frame, text, (startX,Y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

def backend_mtcnn(frame):
    # img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    img = frame
    pil = Image.fromarray(img)
    bounding_boxes, landmarks = mtcnn.detect_faces(pil)
    bounding_boxes = np.array([i for i in bounding_boxes if i is not None],dtype=np.int16)
    # for  f in bounding_boxes:
                
    #     (startX, startY) = f[0], f[1]
    #     (endX, endY) = f[2], f[3]
    #     print(endX-startX,endY-startY)
    #     cv.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
    return bounding_boxes

def backend_cvlib(frame):
    faces, confidence = cvl.detect_face(frame)
    # for  f in face:
                
    #     (startX, startY) = f[0], f[1]
    #     (endX, endY) = f[2], f[3]
    #     cv.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
    #     print(endX-startX,endY-startY)
    return faces

def select_backend(backend_model:str):
    if backend_model == 'mtcnn':
        return backend_mtcnn
    elif backend_model == 'cvlib': # to be removed
        return backend_cvlib
    else:
        print(f'not support {backend_model}.')
        exit()

def clf_verification(croped_image:np.ndarray):
    """
    return : predict id, confidence
    """
    image = croped_image/255.
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float().to(device)
    image = transforms.Resize((196,196))(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    
    with torch.no_grad():
        pred = clf_model(image)
    
    pred = torch.softmax(pred,dim=1)
    confidence,pred = torch.max(pred,1)
    print("pred : ",pred.item(),confidence.item())

    return(pred.cpu().numpy(),confidence.cpu().numpy())

def clf_fcn_verification(croped_image:np.ndarray):
    """
    return : predict id, confidence
    """
    image = croped_image/255.
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()
    image = transforms.Resize((112,112))(image)
    
    with torch.no_grad():
        pred = clf_model(image)
    
    pred = torch.softmax(pred,dim=1)
    confidence,pred = torch.max(torch.mean(pred,dim=(2,3)),1)

    return(pred.cpu().numpy(),confidence.cpu().numpy())

def clf_verify(croped_image:np.ndarray):
    if clf_mode == 1:
        return clf_fcn_verification(croped_image)
    elif clf_mode == 2 or 3:
        return clf_verification(croped_image)
    else:
        raise Exception('clf mode Exception.')

def classification_module(frame,faces,*args):
    for  f in faces:
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
            ## verification moudle 
            id,conf = clf_verify(frame[startY:endY,startX:endX,:])

            ## master
            if conf > 0.85 and id == 1:
                cv.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = f"My Master {conf*100}"
                cv.putText(frame, text, (startX,Y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            ## master?
            elif conf > 0.83 and id == 1:
                cv.rectangle(frame, (startX,startY), (endX,endY), (255,50,0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = f"My Master?? {conf*100}"
                cv.putText(frame, text, (startX,Y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,50,0), 2)
            ## other
            else:
                cv.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = f"Another {conf*100}"
                cv.putText(frame, text, (startX,Y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return frame

# 일단 그냥 conf, args 다씀
def update_config(conf,args):
    conf.threshold = args.threshold
    conf.net = args.net

    return conf
    pass

def load_facebank(conf):
    embeddings = torch.load(conf.facebank_path/'facebank.pth')
    names = np.load(conf.facebank_path/'names.npy')
    return embeddings, names

def infer(model, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = transforms.functional.hflip(img)
                emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > conf.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               

def arc_verification(frame,detected,model,targets,names,args):
    frame_ = frame
    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
    try:
        print(np.array(faces))
        bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
        bboxes = bboxes.astype(np.int16)
        bboxes = bboxes + [-1,-1,1,1] # personal choice    
        results, score = infer(model,conf, faces, targets, args.tta)
        print(names)
        for idx,bbox in enumerate(bboxes):
            if args.score:
                frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame_)
            else:
                frame = draw_box_name(bbox, names[results[idx] + 1], frame_)
    except:
        print('detect error')
        # bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)

    return frame

def arcface_module(conf,args):
    mobilefacenet = MobileFaceNet(512).to(device)
    mobilefacenet.load_state_dict(torch.load(conf.pre_trained_path/conf.mfn_model))
    mobilefacenet.eval()
    if args.update:
        targets, names = prepare_facebank(conf, mobilefacenet, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    return mobilefacenet, targets, names, arc_verification


def select_method(method='classification',args=None):
    allow_method = ['classification','arcface']
    if method not in allow_method:
        raise NameError(f'Current allowed methods are {allow_method}.')

    # pass
    # if not args == None:
    #     conf = update_config(conf,args)

    if method == allow_method[0]:
        return classification_module
    elif method == allow_method[1]:
        return arcface_module(conf,args)


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def prepare_facebank(conf, model, mtcnn, tta = True):
    model.eval()
    embeddings =  []
    names = ['Unknown']
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        inp = Image.open(file).convert('RGB')
                    except:
                        continue
                    if inp.size != (112, 112):
                        img = mtcnn.align(inp)
                        if img == None:
                            print(np.shape(inp))
                            continue
                    with torch.no_grad():
                        if tta:
                            mirror = transforms.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:                        
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0,keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    print(embeddings)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path/'facebank.pth')
    np.save(conf.facebank_path/'names', names)
    
    return embeddings, names

def draw_box_name(bbox,name,frame):
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