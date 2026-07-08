import glob

from util.common_utils import *
from torchvision import transforms
from torch.utils.data import Dataset

mergeMertens = cv2.createMergeMertens()


class MEFB(Dataset):
    def __init__(self, root_dir):
        self.img_paths = sorted(glob.glob(root_dir + '*'))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path1 = glob.glob(self.img_paths[idx] + '/*_A*')[0]
        path2 = glob.glob(self.img_paths[idx] + '/*_B*')[0]
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        mergeMertens = cv2.createMergeMertens()
        img = mergeMertens.process([img1, img2])

        img1 = self.transform(img1).type(torch.cuda.FloatTensor)
        img2 = self.transform(img2).type(torch.cuda.FloatTensor)
        img = self.transform(img).type(torch.cuda.FloatTensor)

        imgs = [img1, img2]
        imgs = torch.stack(imgs, 0).contiguous()
        #img = (img1 + img2) / 2.0
        return img, imgs, self.img_paths[idx].split('/')[-1]


class MEF_dataset(Dataset):
    def __init__(self, root_dir):
        self.img_paths = sorted(glob.glob(root_dir + '*'))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        imgs_path = sorted(glob.glob(self.img_paths[idx] + '/*'))
        num = len(imgs_path)
        imgs = []
        for i in range(num):
            img = cv2.imread(imgs_path[i])
            imgs.append(img)

        input_img = mergeMertens.process(imgs)
        #cv2.imwrite('./merge.png', input_img * 255.0)
        input_img = self.transform(input_img).type(torch.cuda.FloatTensor)

        imgs = [self.transform(x / 255.0).type(torch.cuda.FloatTensor) for x in imgs]
        imgs = torch.stack(imgs, 0).contiguous()
        return input_img, imgs, self.img_paths[idx].split('/')[-1]

