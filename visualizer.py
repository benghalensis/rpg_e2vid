import torch
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
from utils.event_readers import VisualizationIterator
from utils.inference_utils import event_visualizer
import tqdm
import os
import cv2

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('--images_path', required=True, type=str)
    parser.add_argument('--events_path', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--width', required=True, type=int)
    parser.add_argument('--height', required=True, type=int)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()

    # Create output dir
    os.makedirs(args.output_path, exist_ok = True)

    # Read sensor size from the first first line of the event file
    path_to_events = args.events_path
    path_to_images = args.images_path

    width, height = (args.width, args.height)
    print('Sensor size: {} x {}'.format(width, height))

    visualization_iterator = VisualizationIterator(path_to_events, path_to_images)

    base_image = None
    for i, (events, image) in tqdm.tqdm(enumerate(visualization_iterator)):
        if (i % 100==0):
            base_image = image
        
        event_preview = event_visualizer(events, base_image)
        cv2.imwrite(os.path.join(args.output_path,f"{i:06d}.png"), event_preview)