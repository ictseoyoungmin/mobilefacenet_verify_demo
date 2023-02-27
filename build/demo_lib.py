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
# import cvlib as cvl


# [insightface]
from mtcnn import MTCNN
from model import MobileFaceNet,Arcface
from config import get_config
conf = get_config(False)

device = conf.device#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
mtcnn = MTCNN()
print(device)

def open_webcam(device_number=1):
    # open webcam
    webcam = cv.VideoCapture(device_number) # 0 or 1
    webcam.set(3, 640) 
    webcam.set(4, 400) 
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()
    
    return webcam

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