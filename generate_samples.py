import os
import random
import torch
from torchvision import transforms, datasets 
from torchvision.utils import make_grid
from torchvision.io import read_image, write_png
from modules import VectorQuantizedVAE
from datasets import MiniImagenet
from tensorboardX import SummaryWriter
from PIL import Image

writer = SummaryWriter('./')

action1 = '/home/ju/truenas/Jonathan/jonathan/hand-data-fall2021/image-data/train/rub_back/'
action2 = '/home/ju/truenas/Jonathan/jonathan/hand-data-fall2021/image-data/train/rub_palm/'
action3 = '/home/ju/truenas/Jonathan/jonathan/hand-data-fall2021/image-data/train/rub_tips/'
action4 = '/home/ju/truenas/Jonathan/jonathan/hand-data-fall2021/image-data/train/rub_thumb/'
action_list = [action1, action2, action3, action4]
sample_orig_imgs = []
for action in action_list:
    all_imgs = os.listdir(action)
    selected_imgs = random.sample(all_imgs, 4)
    for img in selected_imgs:
        sample_orig_imgs.append(Image.open(os.path.join(action, img)))

# orig_grid = make_grid(sample_orig_imgs, nrow=8)
# write_png(orig_grid, 'original.png')
device = 'cuda:0'
model = VectorQuantizedVAE(3,256,512).to(device)
model.load_state_dict(torch.load('models/models/vqvae/best.pt'))
model.eval()
transform = transforms.Compose([transforms.RandomResizedCrop(128),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
images = torch.zeros((16,3,128,128))
for i, img in enumerate(sample_orig_imgs):
    images[i] = transform(img)
# images = transform(images)
orig_grid = make_grid(images, nrow=8, range=(-1,1), normalize=True)
writer.add_image('original', orig_grid, 0)
# images = images.to(device)
# test_dataset = MiniImagenet('~/truenas/Jonathan/jonathan/data/miniimagenet/', test=True,download=False, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=16, shuffle=True)
# images, _ = next(iter(test_loader))
# images = Image.open('/home/ju/truenas/Jonathan/jonathan/hand-data-fall2021/image-data/train/rub_back/image3310.jpg')
# images = transform(images)
# images = torch.unsqueeze(images, 0)
print(images.shape)
images = images.to(device)
with torch.no_grad():
    images_tilde, _, _ = model(images)
images_tilde = images_tilde.cpu()
# images_tilde = images_tilde.type(torch.uint8)
re_grid = make_grid(images_tilde, nrow=8, range=(-1,1), normalize=True)
# re_grid = re_grid.type(torch.uint8)
# write_png(re_grid, 'reconstructed.png')
writer.add_image('reconstruction', re_grid, 0)
