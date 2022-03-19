import os
from PIL import Image
from numpy import *
import matplotlib.pyplot as plt
import pickle
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True)
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.output): os.mkdir(args.output)

def cover(image, mask):
    if len(mask.shape)==3: mask = mask[:,:,0]
    mask = mask.astype(float)/255.0 if mask.max()>1 else mask.astype(float)
    mask = (mask > 0.50)*1.0 ## Threshold
    r, g, b = image[:,:,0],image[:,:,1],image[:,:,2]
    
    fac = 1.0
    r = ((1.0 - mask) * r * fac + mask * 0 + mask * r * 0.25 ).astype(uint8)
    g = ((1.0 - mask) * g * fac + mask * 190 + mask * g * 0.25 ).astype(uint8)
    b = ((1.0 - mask) * b * fac + mask * 0 + mask * b * 0.25 ).astype(uint8)
    
    covimg = stack([r,g,b], axis=2)
    return covimg.astype(uint8)

inputs = os.listdir(args.image)
for x in tqdm.tqdm(inputs):
    name = ".".join(x.split(".")[0:-1])
    pngname = name+".png"
    if os.path.exists(os.path.join(args.mask, pngname)):
        visual = cover(
            array(Image.open(os.path.join(args.image, x))), 
            array(Image.open(os.path.join(args.mask, pngname)))
        )
        Image.fromarray(visual.astype(uint8)).save(os.path.join(args.output, pngname))
print( "Done, please check %s for the results!"%args.output )