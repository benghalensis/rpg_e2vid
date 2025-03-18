import torch
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader, FixedSizeNPZFileIterator, FixedDurationNPZFileIterator, FixedDurationH5FileIterator
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
import time
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
import tqdm
import os
import shutil

'''
Running this script:
python3 run_reconstruction.py --path_to_model pretrained/E2VID_lightweight.pth.tar 
-i /ocean/projects/cis220039p/shared/tartanair_v2_event/IndustrialHangarAutoExposure/Data_easy/P001/events/events_output/ 
--width 640 --height 640 --auto_hdr --npz_file_iterator_fd --window_duration 50 
--output_folder /ocean/projects/cis220039p/shared/tartanair_v2_event/IndustrialHangarAutoExposure/Data_easy/P001/events/events_output/
'''
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('--width', required=True, type=int)
    parser.add_argument('--height', required=True, type=int)
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.add_argument('--npz_file_iterator', dest='npz_file_iterator', action='store_true')
    parser.add_argument('--npz_file_iterator_fd', dest='npz_file_iterator_fd', action='store_true')
    parser.add_argument('--h5_file_iterator', dest='h5_file_iterator', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()

    # Parse the input path
    if args.input_file.endswith('/'):
        args.input_file = args.input_file[:-1]
    path_to_events = args.input_file

    width, height = (args.width, args.height)
    print('Sensor size: {} x {}'.format(width, height))

    # Load model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()
    
    # Remove the reconstruction output directory
    if args.output_folder is None:
        args.output_folder = os.path.dirname(args.input_file)
    event_previews_folder = os.path.join(args.output_folder, args.dataset_name)
    if os.path.exists(event_previews_folder):
        shutil.rmtree(event_previews_folder)

    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    """ Read chunks of events using Pandas """

    # Loop through the events and reconstruct images
    N = args.window_size
    if not args.fixed_duration:
        if N is None:
            N = int(width * height * args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                N, args.num_events_per_pixel))
        else:
            print('Will use {} events per tensor (user-specified)'.format(N))
            mean_num_events_per_pixel = float(N) / float(width * height)
            if mean_num_events_per_pixel < 0.1:
                print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
            elif mean_num_events_per_pixel > 1.5:
                print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    if args.fixed_duration:
        event_window_iterator = FixedDurationEventReader(path_to_events,
                                                         duration_ms=args.window_duration,
                                                         start_index=start_index)
    elif args.npz_file_iterator:
        event_window_iterator = FixedSizeNPZFileIterator(path_to_events, num_events=N)
    
    elif args.h5_file_iterator:
        event_window_iterator = FixedDurationH5FileIterator(path_to_events, duration_ms=args.window_duration)

    elif args.npz_file_iterator_fd:
        event_window_iterator = FixedDurationNPZFileIterator(path_to_events, duration_ms=args.window_duration)

    else:
        event_window_iterator = FixedSizeEventReader(path_to_events, num_events=N, start_index=start_index)

    with Timer('Processing entire dataset'):
        for event_window in tqdm.tqdm(event_window_iterator):

            last_timestamp = event_window[-1, 0]

            with Timer('Building event tensor'):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_window,
                                                        num_bins=model.num_bins,
                                                        width=width,
                                                        height=height)
                    event_tensor = torch.from_numpy(event_tensor)
                else:
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=model.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)

            num_events_in_window = event_window.shape[0]
            reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

            start_index += num_events_in_window
