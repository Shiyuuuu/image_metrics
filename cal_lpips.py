import lpips
from utils import tensor2img, img2tensor

'''
https://github.com/richzhang/PerceptualSimilarity

@inproceedings{zhang2018perceptual,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={CVPR},
  year={2018}
}
'''


loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

if __name__ =='__main__':
    import torch

    img0 = torch.randn(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = torch.randn(1, 3, 64, 64)
    d = loss_fn_alex(img0, img1)

    print(d)

    from skimage import io

    clean = io.imread('clean/2762.png')
    noisy = io.imread('noisy/2762.png')
    print(loss_fn_alex(img2tensor(clean), img2tensor(noisy)))
    print(loss_fn_vgg(img2tensor(clean), img2tensor(noisy)))
