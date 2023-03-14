import os
import numpy as np
import torch
import streamlit as st
import random

from PIL import Image

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
#import torch
from datasets import grid,ex4

MIN_OFFSET = 0
MAX_OFFSET = 8
MIN_SPACING = 2
MAX_SPACING = 6

# st.set_page_config(layout="wide")


def load_image(image_file):
    image = Image.open(image_file)
    image = np.asarray(image, dtype=np.float32)
    return image


transforms = A.Compose(
    [
        A.Resize(
            100, 100, interpolation=cv2.INTER_LINEAR, always_apply=True, p=1
        ),  # resize the image to 100x100
        ToTensorV2(),  # convert the image to a tensor
        
    ]
)


st.header("Image Inpainting Project Demo Page")
"""
This is a demo page for an image inpainting project. \n
Repository: https://github.com/HectorAuvinen/Image-Inpainting-Project
"""
#st.image("./recoursces/grid_specifications.png")


OFFSET = st.slider(
    "Offset", MIN_OFFSET, MAX_OFFSET, (0, 1)
)  # offset in the N-direction
SPACING = st.slider(
    "Spacing ", MIN_SPACING, MAX_SPACING, (2, 3)
)  # spacing in the M-direction

clicked_random_OS = st.button(
    "Random Offset and Spacing"
)  # random offset and spacing

model = st.selectbox(
    'Which model architecture would you like to use?',
    ('5 Hidden Layers', ' 6 Hidden Layers'))
st.write('You selected:', model)

model_path = r"./results/models/6_hidden/best_model.pt"

if model == '5 Hidden Layers':
    model_path = r"./results/models/5_hidden/best_model.pt"
elif model == '6 Hidden Layers':
    model_path = r"./results/models/6_hidden/best_model.pt"

image_file = st.file_uploader(
    "Upload Images", type=["png", "jpg", "jpeg"]
)  # upload image


clicked_random_image = st.button("Random Image")

if clicked_random_image:
    image_file = Image.open(
        random.choice(
            [os.path.abspath(os.path.join("./pics", p)) for p in os.listdir("./pics")]
        )
    )
    image_array = np.asarray(image_file, dtype=np.float32)


if image_file is not None:
    if not clicked_random_image:
        image_array = load_image(image_file)  # load image

    image_file = transforms(image=image_array)["image"]  # apply the transform

    if clicked_random_OS:
        offset = np.random.randint(MIN_OFFSET, MAX_OFFSET, size=2)  # random offset
        spacing = np.random.randint(MIN_SPACING, MAX_SPACING, size=2)  # random spacing
    else:
        offset = OFFSET
        spacing = SPACING

    target, image_array, known_array, _ = grid(
        np.asarray(image_file, dtype=np.float32), offset, spacing
    )  # apply the grid
    known_array = known_array[
        0:1, ::, ::
    ]  # remove the channel dimension (3, 256, 256) -> (256, 256)

    full_image = torch.cat(
        (torch.from_numpy(image_array), torch.from_numpy(known_array)), 0
    )  # concatenate the image and the known array

    col1, col2, col3 = st.columns(3, gap="small")  # split the screen into 3 columns

    with col1:
        st.subheader("Input Image ")
        st.image(
            np.transpose(image_array.astype("uint8"), (1, 2, 0)), width=200
        )  # show the image

    with col2:
        st.subheader("Target Image (Original Size) ")
        st.image(
            np.transpose(target.astype("uint8"), (1, 2, 0)), width=200
        )  # show the target image

    with st.spinner("Doing Inference..."):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # use GPU if available
        if model_path == None:
            model_path = r"./results/models/6_hidden/best_model.pt"
        model = torch.load(
#C:\Users\Hector Auvinen\Documents\image_inpainting\saved_models\32_batchsize_7_hidden_64_kernels_3_kernel_size_130722
            #r"./results/best_model.pt", map_location=torch.device("cpu")
            #r"./results/models/model_1.pt", map_location=torch.device("cpu")
            model_path, map_location=torch.device("cpu")
        )  # load the model
        model.to(device)  # move the model to the GPU
        with torch.no_grad():  # do not compute gradients
            output = model(full_image.to(device))  # predict the image
        output = output.detach().cpu().numpy()  # move the output to the CPU
    st.success("Done!")  # show the success message

    with col3:
        st.subheader("Output Image")
        st.image(
            np.transpose(output.astype("uint8"), (1, 2, 0)), width=200
        )  # show the output image