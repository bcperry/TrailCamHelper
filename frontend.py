import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import zipfile
import tempfile
import gc

import cv2

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

# start with important models and helpers
@st.cache_resource
def load_model():
    st.header("Deer Detection Model")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    model.eval()

    return model

# start with important models and helpers
@st.cache_resource
def load_processor():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    return processor


large_animal_list = [19, 20, 21, 23, 24, 25]


# https://docs.streamlit.io/library/api-reference

st.title('TrailCam Helper')


# def multi_inference(files):
#     url = "http://localhost:8000/uploadfiles/"
#     multi_files = (('files', x) for x in files)  # need to convert to tuples for the requests library

#     response = requests.post(url=url, files=multi_files)
#     return response

model = load_model()
processor = load_processor()

confidence = st.slider('What level of confidence to find deer', 0, 100, 80, help="Lower confidence will find more animals, but may have false positives.")

data = st.file_uploader("Upload Trail Camera Images", 
                        type=['png', 'jpg', 'webp', 'mp4'], 
                        accept_multiple_files=True,
                        help="Select the files you want to check for deer")


if len(data) > 0:
# if data is not None:
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    
    # res = multi_inference(data)

    # # for debugging 
    # st.subheader("Request POST Header - just for debugging")
    # st.json(dict(res.request.headers))
    # st.subheader("Response Status Code - just for debugging")
    # st.info(f'Status Code: {res.status_code}')
    # st.subheader("Response Header - just for debugging")
    # st.json(dict(res.headers))
    # st.subheader("Response Content - just for debugging")
    # st.write(res.content)

    data_load_state.text("Data Loaded")

    detection_list = []

    for file in data:

        # check if the file is a video
        if 'video' in file.type:
            st.write('video loading...')

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            vidcap = cv2.VideoCapture(tfile.name)

            cur_frame = 0

            success, frame = vidcap.read() # get next frame from video
            image = Image.fromarray(frame[:,:,[2,1,0]]) # convert opencv frame (with type()==numpy) into PIL Image

        else:

            image = Image.open(file)



        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)



        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=(confidence/100))[0]
        # st.write(results)

        # check if the image contains any interesting detections
        if bool(set(large_animal_list).intersection(results['labels'].tolist())):
            st.write(f"Animal found in {file.name}")

            # if an animal is found, add it to the list for later download
            detection_list.append(file)


            # filter the results to just the animals of interest
            animal_of_interest = np.where(np.isin(results['labels'], large_animal_list))[0]


            im = to_pil_image(
                draw_bounding_boxes(
                    pil_to_tensor(image),
                    results['boxes'][animal_of_interest],
                    colors="red",
                    # labels = [f"{model.config.id2label[x]}: conf. {round(results['scores'][animal_of_interest].item(), 3)}" for x in results['labels'][animal_of_interest].tolist()]
                    labels = [f"Confidence: {round(x*100)}" for x in results['scores'][animal_of_interest].tolist()]
                            )
                        )
            
            st.image(im)
    

    # write the iamges to a zip folder for download

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file in detection_list:
            zip_file.writestr(file.name, file.getvalue())
    

    if st.download_button("Download Images/Videos with Deer",
                       file_name="deerImages.zip",
                       mime="application/zip",
                       data=zip_buffer
                       ):

        gc.collect()

    
    
# hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# st.map(filtered_data)