# Analysis of Machine Learning Models for Trash Detection
This project evaluates the performance of four deep learning-based models—GroundingDINO, YOLO, DETR, and ResNet—for object detection, trained on a subset of the TACO (Trash Annotations in Context) dataset. Each model was assessed based on Precision, Recall, and mean Average Precision (mAP) to determine the most effective solution. Additionally, the models were tested in real-time to evaluate their practical applicability. A comprehensive evaluation of this project is available on our [webpage](https://isaac-berlin.github.io/CV_Garbage_Detection/) and detailed in our [paper](docs/computer_vision_project.pdf). This project was the final for our class CSCI 5561 Intro to Computer Vision and we would like to extend a special thank you to our professor Volkan Isler.

## Data

The [TACO](http://tacodataset.org/TACO) dataset (Trash Annotations in Context) is a comprehensive collection of images depicting trash and recycling objects in real-world settings. This dataset, used for training and evaluating our models, includes images containing between 0 and 40 objects spanning 19 classes, such as bottles, cans, and plastic bags. Additionally, it features a "catch-all" class, labeled as unlabeled litter, for unidentified items. The dataset comprises 4,000 images, which were a.ugmented to expand the total to 6,000 images, all resized to 416x416 pixels. The annotations were created using Roboflow and are available [here](https://universe.roboflow.com/divya-lzcld/taco-mqclx).

## Models

I was personally in charge of the DETR model. A DETR (DEtection TRansformer) is a transformer-based object detection model that eliminates the need for traditional hand-crafted components like anchor generation and NMS, using an end-to-end approach with a bipartite matching loss to predict objects directly from input images. 

This model was first introduced in a paper by Facebook AI in 2020. Since then, numerous similarly constructed models have been developed, each offering unique benefits and improvements tailored to specific applications. Below is a list of the models we utilized for this project.

- DETR
- RT-DETR
- Conditional DETR
- Deformable DETR

## Results

Due to isues with compute power and time, we were unable to get the DETR and RT-DETR to converge in a reasonable amount of time. We trained the Conditional DETR and Deformable DETR at varrying epochs to achieve the results detailed below.

| model | Number of Epochs | mAP 50-90 | mAP 50 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Conditional DETR | 50 | 0.083 | 0.117 | 0.682 | 0.109 |
| Conditional DETR | 200 | 0.234 | 0.301 | 0.534 | 0.238 |
| Deformable DETR | 50 | 0.006 | 0.008 | 0.642 | 0.022 |

## Code

To support this project, I implemented a pipeline to convert COCO object annotations into the required Hugging Face Dataset format. Using this pipeline, I fine-tuned pretrained DETR models, initially trained on the COCO dataset, with our TACO dataset. The notebook detailing the fine-tuning process is available at ```src/finetune_DeTr.ipynb```. Additionally, an evaluation notebook can be found at ```src/eval.ipynb```, and a script for loading and running the model in real-time is provided at ```src/real_time_eval.py```.