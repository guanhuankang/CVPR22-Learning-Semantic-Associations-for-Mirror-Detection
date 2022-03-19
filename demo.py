import os, argparse, torch, numpy, ntpath, tqdm
from PIL import Image
from model.architecture import Architecture

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoints", type=str, required=True)
parser.add_argument("-i", "--image", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-nocrf", "--nocrf", action="store_true")
args = parser.parse_args()

print(
    "Load checkpoints:", args.checkpoints, 
    "\nInput:", args.image, 
    "\nOutput:", args.output,
    "\nCRF:", not args.nocrf)

model = Architecture(training = False).cuda()
model.load_state_dict(torch.load(args.checkpoints))
model.eval()

images = [args.image]
if os.path.isdir(args.image):
    images = os.listdir(args.image)
    images = [ os.path.join(args.image, x) for x in images if x[-4::] in [".jpg",".png",".JPG",".PNG","jpeg"]]

if not os.path.exists(args.output): os.mkdir(args.output)

for imgpath in tqdm.tqdm(images):
    torch_img = torch.from_numpy(numpy.array(Image.open(imgpath), dtype=numpy.uint8)) # H, W, C
    if len(torch_img.shape)==2:
        torch_img = torch.stack([torch_img,torch_img,torch_img], dim=2)
    if len(torch_img.shape)==3 and torch_img.shape[-1]>=4:
        torch_img = torch_img[:,:,0:3]
    img = torch_img.numpy()


    size = torch_img.shape[0:2]
    torch_img = torch.nn.functional.interpolate(
        torch_img.permute(2, 0, 1).unsqueeze(0), 
        size=(384,384), 
        mode="nearest")
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
    
    torch_img = (torch_img/255. - mean)/std
    out = model(torch_img.cuda())["final"].cpu().detach()
    out = torch.sigmoid(torch.nn.functional.interpolate(out, size=size, mode="bilinear"))
    out = (out.numpy()*255.).astype(numpy.uint8)[0, 0]

    if not args.nocrf:
        from crf import crf_refine
        out = crf_refine(img.astype(numpy.uint8), out)
    
    Image.fromarray(out.astype(numpy.uint8)).save( os.path.join(
        args.output, ntpath.basename(imgpath).replace(".jpg", ".png")
    ) )
    # print("running ", imgpath, end="\r")
print("Done", "see '%s' for the output"%args.output )