import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as transforms
import os
import os.path

from torchvision.models.inception import inception_v3
import numpy as np
import math
from PIL import Image
import argparse




IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_paths.append(path)

    return img_paths, len(img_paths)

def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths, len(img_paths)


def make_dataset(path_files):
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(path_files)
    else:
        paths, size = make_dataset_dir(path_files)

    return paths, size


def get_inception_score(imgs, batch_size=32, resize=True, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = imgs.shape[0]

    assert batch_size > 0
    assert N > batch_size

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=True)
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)
    if torch.cuda.is_available():
        inception_model.cuda(0)
        up.cuda(0)
    inception_model.eval()

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    n_batches = int(math.ceil(float(N) / float(batch_size)))

    for i in range(n_batches):
        batch = torch.from_numpy(imgs[i * batch_size:min((i + 1) * batch_size, N)])
        batchv = Variable(batch)
        if torch.cuda.is_available():
            batchv = batchv.cuda(0)

        preds[i * batch_size:min((i + 1) * batch_size, N)] = get_pred(batchv)

    # Now compute the mean kl-div
    scores = []

    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
    parser.add_argument('--save_path', type=str,
                        default='noisy/',
                        help='path to save the test dataset')
    parser.add_argument('--num_test', type=int, default=400,
                        help='how many images to load for each test')
    args = parser.parse_args()


    img_paths, img_size = make_dataset(args.save_path)
    print(len(img_paths))

    iters = int(10000 / args.num_test)

    u = np.zeros(iters, np.float32)
    sigma = np.zeros(iters, np.float32)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    for i in range(iters):
        all_samples = []
        num = i*args.num_test

        for j in range(args.num_test):
            index = num+j
            image = Image.open(img_paths[index]).resize([256,256]).convert('RGB')
            all_samples.append(transform(image).reshape(-1,3,256,256))
        all_samples = np.concatenate(all_samples, axis=0)
        u_iter,sigma_iter = get_inception_score(all_samples, batch_size=8)

        u[i] = u_iter
        sigma[i] = sigma_iter

        print(i)
        print('{:10.4f},{:10.4f}'.format(u_iter, sigma_iter))

    print('{:>10},{:>10}'.format('u', 'sigma'))
    print('{:10.4f},{:10.4f}'.format(u.mean(), sigma.mean()))
