import numpy as np
from PIL import Image
import torch

k = 10

imagenet_labels = dict(enumerate(open("imagenet-classes.txt")))

model = torch.load("model.pth", weights_only=False)
model.eval()

img = Image.open("cat.png")

width, height = img.size

if (width != height) or (width != 384):
    left = 4
    top = height / 5
    right = 154
    bottom = 3 * height / 5
    img = img.crop((left, top, right, bottom))
    newsize = (384, 384)
    img = img.resize(newsize)

img = (np.array(img) / 128) - 1


inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
logits = model(inp)
probs = logits.softmax(dim=-1)

top_probs, top_ixs = probs[0].topk(k)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()
    print(f"{i}: {cls:<45} ---- {prob*100:.4f}%")
