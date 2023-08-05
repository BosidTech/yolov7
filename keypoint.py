import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device).eval()
    
import os
names = [name.split('.')[0].split('_')[1] for name in os.listdir('runs/detect/exp11/labels/') if '.txt' in name]

video = '07'
if not os.path.exists(video + '_pose'):
    os.mkdir(video + '_pose')
folder_name = video + '_pose'
with torch.no_grad():
    for name in names:
        #print(name)
        try:
            image = cv2.imread('./' + video + '_frames/' + video + '_' + name +'.jpg')
            image = letterbox(image, 960, stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            output, _ = model(image.half().to(device))

            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            with torch.no_grad():
                output = output_to_keypoint(output)
            txt_path = folder_name + '/' + '7-' + name
            print(txt_path)
            if len(output) > 0:
                for op in output:
                    line = str(op.tolist())  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(line + '\n')
            torch.cuda.empty_cache()
        except:
            print('error')

   #if torch.cuda.is_available():
        #image = image.half().to(device)
    """   
    output, _ = model(image.half().to(device))

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    print(len(output))
    #nimg = image[0].permute(1, 2, 0) * 255
    #nimg = nimg.cpu().numpy().astype(np.uint8)
    #nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    #for idx in range(output.shape[0]):
    #    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    # Write to filet
    txt_path = folder_name + '/' + '7-' + name
    print(txt_path)
    if len(output) > 0:
        for op in output:
            line = str(op.tolist())  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(line + '\n')
    torch.cuda.empty_cache()
    """