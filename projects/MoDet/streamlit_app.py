# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import time
import os, urllib, cv2
import argparse, json
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.data import MetadataCatalog

from detectron2.engine.defaults import DefaultPredictor

DETECTRON2_DATASETS = "/media/rahul/Karmic/data"
COCO_DATASET  = DETECTRON2_DATASETS+"/coco"
coco_id_label_categories = {}

# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("streamlit/instructions.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("app.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()


def load_master_coco_categories(basepath, categories_file):
    id_label_categories = {}
    id_label_super_categories = {}
    label_colors = {}
    with open(os.path.join(basepath, categories_file), 'r') as catfile:
        data = catfile.read()
        cobj = json.loads(data)
        for im in cobj:
            if im["isthing"]:
                id_label_categories[im["id"]] = im["name"]
                id_label_super_categories[im["id"]] = im["supercategory"]
                label_colors[im["name"]] = im["color"]
                
    return id_label_categories, id_label_super_categories, label_colors



def load_images_ann_pandas(basepath, instances_annotation_file):
    dataset_instances = pd.DataFrame(columns=['id','frame', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'coco_url'])
    with open(os.path.join(basepath, instances_annotation_file), 'r') as annfile:
        data = annfile.read()
        cobj = json.loads(data)
        images = {}
        for im in cobj["images"]:
            images[im["id"]] = im
            
        for ann in cobj["annotations"]:
            (x, y, w, h) = ann["bbox"]
            segmentations = ann["segmentation"]
            cls_name = coco_id_label_categories[ann["category_id"]]
            cls_super = coco_id_label_super_categories[ann["category_id"]]
            dataset_instances = dataset_instances.append({'id': ann["image_id"], 
                                        'frame': images[ann["image_id"]]["file_name"], 
                                        'xmin': int(x), 'ymin': int(y), 'xmax': int(x+w), 'ymax': int(y+h), 
                                        'label': cls_name, 'coco_url': images[ann["image_id"]]["coco_url"],
                                        'label_super': cls_super, 'segmentation': []
                                        }, ignore_index=True)
    return dataset_instances

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache(allow_output_mutation=True)
    def load_metadata():
        """ 
            id,     frame                  , xmin, ymin, xmax, ymax, label,      coco_url,  label_super,  segmentation_bounds
            12,     1478019952686311006.jpg,  254,  153,  269,  166, car   ,             ,      vehicle, 
            12,     1478019952686311006.jpg,  468,  128,  487,  199, person,             ,       person, 
            14,     1478019953180167674.jpg,  233,  157,  248,  169, car   ,             ,      vehicle, 
        """
        instances_annotation_file = "annotations/instances_val2017_100.json"
        print("\n", "[COCO]\t",COCO_DATASET, "\t", instances_annotation_file)
        annotations = load_images_ann_pandas(COCO_DATASET, instances_annotation_file)
        print(annotations.head(2))
        return annotations

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.cache
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label","label_super","coco_url"]], columns=["label_super"])
        summary = one_hot_encoded.groupby(["frame", "coco_url"]).sum().rename(columns={
            "label_super_person" : "person",
            "label_super_vehicle" : "vehicle"
        })
        print(summary)
        return summary

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    metadata = load_metadata()
    summary = create_summary(metadata)

    # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    selected_frame_index, selected_frame = frame_selector_ui(summary)
    print("----> ", selected_frame_index,selected_frame)
    if selected_frame_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()

    # Load the image from Local.
    image_url = os.path.join(COCO_DATASET,"val2017", selected_frame[0])
    image = load_image(image_url)
    

    # A.) Get the boxes for the objects detected by YOLO by running the YOLO model.
    boxes = run_on_image(image)
    #draw_image_with_boxes(image, boxes, "Real-time Road Damage Detection",
    #    "**Faster RCNN Resnet 50 Model** (overlap `%3.1f`) (confidence `%3.1f`) -`%s`" % (overlap_threshold, confidence_threshold, selected_frame[0]))

    # B.) Uncomment these lines to peek at these DataFrames.
    st.write('## Summary', summary[:10], '## Metadata', metadata[:10])

    # C.) Add boxes for objects on the image. These are the boxes for the ground image.
    boxes = metadata[metadata.frame == selected_frame[0]][['xmin', 'ymin', 'xmax', 'ymax', 'label']]
    draw_image_with_boxes(image, boxes, "Ground Truth",
        "**Human-annotated data** (frame `%i`-`%s`)" % (selected_frame_index, selected_frame[0]))



# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(summary):
    st.sidebar.markdown("# Frame")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)

    # The user can select a range for how many of the selected objecgt should be present.
    min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 25, [1, 15])
    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
    if len(selected_frames) < 1:
        return None, None
    
    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)

    # Draw an altair chart in the sidebar with information on the frame.
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y("%s:Q" % object_type))
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(
        alt.X("selected_frame:Q", axis=None)
    )
    st.sidebar.altair_chart(alt.layer(chart, vline))
    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame

# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image, boxes, header, description):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    image_with_boxes = image.astype(np.float64)
    if 'scores' in boxes.columns:
        for _, (xmin, ymin, xmax, ymax, label, score) in boxes.iterrows():
            cv2.putText(image_with_boxes, text="{0} ({1:1.2f})".format(label, score), org=(int(xmin+2),int(ymin+10)),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,255),
                thickness=1, lineType=cv2.LINE_AA)
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2
    else:
        for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
            cv2.putText(image_with_boxes, text=label, org=(int(xmin+2),int(ymin+10)),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,255),
                thickness=1, lineType=cv2.LINE_AA)
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    with open(path, 'r', encoding="utf8") as myfile:
        return myfile.read()
    return None

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image(image_fullPath):
    print(image_fullPath)
    image = cv2.imread(image_fullPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def run_on_image(image):
    """
    Args:
        image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            This is the format used by OpenCV.

    Returns:
        predictions (dict): the output of the model.
    """
    start_time = time.time()
    predictions = predictor(image)
    print(
        "{}: {} in {:.2f}s".format(
            "Format + Predict Image",
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )
    print(predictions['instances'])
    return predictions


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml")
    cfg.merge_from_list(["MODEL.WEIGHTS", "./output/model_final.pth"])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg


coco_categories_file = "panoptic_coco_categories.json"
coco_id_label_categories, coco_id_label_super_categories, LABEL_COLORS = load_master_coco_categories("streamlit/", coco_categories_file)
cfg = setup_cfg()
predictor = DefaultPredictor(cfg)

""" 
  streamlit run streamlit_app.py --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x_md.yaml --opts MODEL.WEIGHTS ./output/model_final.pth
""" 
if __name__ == "__main__":
    main()
