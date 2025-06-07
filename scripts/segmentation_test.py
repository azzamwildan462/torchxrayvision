'''
@author AWM and the Dangereous

Run seperti biasa:
cd scripts && python3 segmentation_test.py

Gunakan arah kanan atau arah kiri untuk mengganti gambar 

Notes:
Gambar yang ingin diuji bisa ditambahkan dalam folder ../imgs dengan format .png atau .jpg atau .jpeg
'''

import os
import numpy as np
import skimage.io
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchxrayvision as xrv
import time

# Load model
model = xrv.baseline_models.chestx_det.PSPNet()
model.eval()

# Image folder path
image_dir = "../imgs"
image_paths = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(('.png', '.jpg', '.jpeg'))
])

# Transforms
resize = T.Resize((514, 514))

# State
current_index = 0
fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
axes = axes.flatten()
fig.tight_layout()

def run_inference(index):
    img = skimage.io.imread(image_paths[index])

    if img.ndim == 3 and img.shape[2] == 3:
        img = img.mean(axis=2)

    img = xrv.datasets.normalize(img, 255)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    img_tensor = resize(img_tensor)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.nn.functional.interpolate(pred, size=[498, 498])
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    masks = {
        'Image': img_tensor.squeeze().numpy(),
        'Clavicles': torch.logical_or(pred[0,model.targets.index('Left Clavicle')], pred[0,model.targets.index('Right Clavicle')]),
        'Scapulas': torch.logical_or(pred[0,model.targets.index('Left Scapula')], pred[0,model.targets.index('Right Scapula')]),
        'Lungs': torch.logical_or(pred[0,model.targets.index('Left Lung')], pred[0,model.targets.index('Right Lung')]),
        'Hilus Pulmonis': torch.logical_or(pred[0,model.targets.index('Left Hilus Pulmonis')], pred[0,model.targets.index('Right Hilus Pulmonis')]),
        'Heart': pred[0,model.targets.index('Heart')],
        'Aorta': pred[0,model.targets.index('Aorta')],
    }

    return masks

def display_overlay(ax, base_image, masks_dict):
    ax.imshow(base_image, cmap='gray')  # Grayscale X-ray
    cmap_list = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'pink', 'YlOrBr']
    alpha = 0.4
    for i, (label, mask) in enumerate(masks_dict.items()):
        if label == 'Image':
            continue
        cmap = plt.get_cmap(cmap_list[i % len(cmap_list)])
        ax.imshow(mask.numpy(), cmap=cmap, alpha=alpha)
    ax.set_title("Overlay Segmentation")
    ax.axis('off')

def update_display(index):
    
    time_start = time.time()
    masks = run_inference(index)
    time_end = time.time()
    print(f"Time taken for inference: {time_end - time_start:.2f} seconds")
    titles = list(masks.keys())

    # Show individual masks
    for i in range(len(titles)):
        axes[i].clear()
        axes[i].imshow(masks[titles[i]], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    # Overlay on last subplot
    if len(axes) > len(titles):
        axes[-1].clear()
        display_overlay(axes[-1], masks['Image'], masks)

    # Hide any extra axes
    for i in range(len(titles), len(axes) - 1):
        axes[i].axis('off')

    fig.canvas.draw_idle()

def callback_keypress(event):
    global current_index
    if event.key in ['right', ' ']:
        current_index = (current_index + 1) % len(image_paths)
        update_display(current_index)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(image_paths)
        update_display(current_index)

# Initial display
update_display(current_index)

# Keyboard event binding
fig.canvas.mpl_connect('key_press_event', callback_keypress)
plt.show()


'''
References:

Joseph Paul Cohen and Mohammad Hashir and Rupert Brooks and Hadrien Bertrand
On the limits of cross-domain generalization in automated X-ray prediction. 
Medical Imaging with Deep Learning 2020 (Online: https://arxiv.org/abs/2002.02497)

@inproceedings{cohen2020limits,
  title={On the limits of cross-domain generalization in automated X-ray prediction},
  author={Cohen, Joseph Paul and Hashir, Mohammad and Brooks, Rupert and Bertrand, Hadrien},
  booktitle={Medical Imaging with Deep Learning},
  year={2020},
  url={https://arxiv.org/abs/2002.02497}
}

Joseph Paul Cohen, Joseph D. Viviano, Paul Bertin, Paul Morrison, Parsa Torabian, Matteo Guarrera, Matthew P Lungren, Akshay Chaudhari, Rupert Brooks, Mohammad Hashir, Hadrien Bertrand
TorchXRayVision: A library of chest X-ray datasets and models. 
Medical Imaging with Deep Learning
https://github.com/mlmed/torchxrayvision, 2020


@inproceedings{Cohen2022xrv,
title = {{TorchXRayVision: A library of chest X-ray datasets and models}},
author = {Cohen, Joseph Paul and Viviano, Joseph D. and Bertin, Paul and Morrison, Paul and Torabian, Parsa and Guarrera, Matteo and Lungren, Matthew P and Chaudhari, Akshay and Brooks, Rupert and Hashir, Mohammad and Bertrand, Hadrien},
booktitle = {Medical Imaging with Deep Learning},
url = {https://github.com/mlmed/torchxrayvision},
arxivId = {2111.00595},
year = {2022}
}

'''