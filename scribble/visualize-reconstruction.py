import torch

x, channel_ids, mask = batch

x = x[0].unsqueeze(0)
channel_ids = channel_ids[0].unsqueeze(0)
mask = mask[0].unsqueeze(0)

rec = self.model.reconstruct(x=x, channel_ids=channel_ids, mask=mask)
rec = einops.rearrange(rec, "b c h w (kh kw) -> b c (h kh) (w kw)", kh=8, kw=8)
rec = rec.squeeze(0)

# from torchvision.utils import make_grid, save_image
# import numpy as np
# tmp = np.load('/work/FAC/FBM/DBC/mrapsoma/prometex/data/virtues/datasets/PCa/random_crops_self_std_256/231204_001_0.npy')
# save_image(g0, 'test.png')

# images
image = einops.rearrange(x, "b c h w (kh kw) -> b c (h kh) (w kw)", kh=8, kw=8)
image = image.squeeze(0)
h, w = image.shape[-2:]

# masks
m = torch.zeros_like(x)
m[mask] = 1
m = einops.rearrange(m, "b c h w (kh kw) -> b c (h kh) (w kw)", kh=8, kw=8)
m = m.squeeze(0)

# masked image
images_masked = image.clone()
images_masked[m.bool()] = images_masked.min()

num_channels = rec.shape[0]
labels = [str(i) for i in range(num_channels)]

normalize = True
g0, nrows, _ = image_to_channels_grid(image, nrow=1, padding=padding, pad_value=0.5, normalize=normalize,
                                      scale_each=True)
g0 = label_grid(g0, channel_labels=labels, nrows=nrows, ncols=1, height=h, width=w, padding=padding)

g1, nrows, _ = image_to_channels_grid(images_masked, nrow=1, padding=padding, pad_value=0.5, normalize=normalize,
                                      scale_each=True)
g1 = label_grid(g1, channel_labels=labels, nrows=nrows, ncols=1, height=h, width=w, padding=padding)

g2, nrows, _ = image_to_channels_grid(rec, nrow=1, padding=padding, pad_value=0.5, normalize=normalize,
                                      scale_each=True)
g2 = label_grid(g2, channel_labels=labels, nrows=nrows, ncols=1, height=h, width=w, padding=padding)

grid = torch.cat([g0, g1, g2], dim=-1)

