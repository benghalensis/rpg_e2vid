import pandas as pd
import zipfile
from os.path import splitext
import numpy as np
from .timers import Timer
import os
import cv2

class FixedSizeNPZFileIterator:
    def __init__(self, folder_path, num_events):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        self.file_list = sorted(self.file_list)
        self.index = 0
        self.num_events = num_events

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.file_list):
            raise StopIteration
        return_data = np.zeros(shape=(1,4))
        while (self.index < len(self.file_list)) and (return_data.shape[0] < self.num_events):
            file_path = os.path.join(self.folder_path, self.file_list[self.index])
            data = np.load(file_path)
            return_data = np.concatenate((return_data, np.array([data['t'], data['x'], data['y'], data['p']]).T))
            self.index += 1

        # Convert time to seconds
        return_data[:,0] /= 10**9

        return return_data[1:]

class VisualizationIterator:
    def __init__(self, event_folder_path, image_folder_path):
        self.image_folder_path = image_folder_path
        self.image_file_list = [f for f in os.listdir(image_folder_path) if f.endswith('.png')]
        
        self.event_folder_path = event_folder_path
        self.event_file_list = [f for f in os.listdir(event_folder_path) if f.endswith('.npz')]

        self.image_file_list = sorted(self.image_file_list)
        self.event_file_list = sorted(self.event_file_list)
        self.event_index = 0

        self.image_freq = 1000
        self.event_freq = 1000

    def __iter__(self):
        return self

    def __next__(self):
        if self.event_index >= len(self.event_file_list):
            raise StopIteration

        event_data = np.zeros(shape=(1,4))
        for i in range(round(self.event_freq/self.image_freq)):
            data = np.load(os.path.join(self.event_folder_path, self.event_file_list[self.event_index]))
            event_data = np.concatenate((event_data, np.array([data['t'], data['x'], data['y'], data['p']]).T))
            self.event_index += 1
        
        # Convert time to seconds
        event_data[:,0] /= 10**9
        image_data = cv2.imread(os.path.join(self.image_folder_path, self.image_file_list[self.event_index]), cv2.IMREAD_COLOR)

        return event_data[1:], image_data

class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.iterator = pd.read_csv(path_to_event_file, delim_whitespace=True, header=None,
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=start_index + 1, chunksize=num_events, nrows=None, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            event_window = self.iterator.__next__().values
        return event_window


class FixedDurationEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt', '.zip'])
        self.is_zip_file = (file_extension == '.zip')

        if self.is_zip_file:  # '.zip'
            self.zip_file = zipfile.ZipFile(path_to_event_file)
            files_in_archive = self.zip_file.namelist()
            assert(len(files_in_archive) == 1)  # make sure there is only one text file in the archive
            self.event_file = self.zip_file.open(files_in_archive[0], 'r')
        else:
            self.event_file = open(path_to_event_file, 'r')

        # ignore header + the first start_index lines
        for i in range(1 + start_index):
            self.event_file.readline()

        self.last_stamp = None
        self.duration_s = duration_ms / 1000.0

    def __iter__(self):
        return self

    def __del__(self):
        if self.is_zip_file:
            self.zip_file.close()

        self.event_file.close()

    def __next__(self):
        with Timer('Reading event window from file'):
            event_list = []
            for line in self.event_file:
                if self.is_zip_file:
                    line = line.decode("utf-8")
                t, x, y, pol = line.split(' ')
                t, x, y, pol = float(t), int(x), int(y), int(pol)
                event_list.append([t, x, y, pol])
                if self.last_stamp is None:
                    self.last_stamp = t
                if t > self.last_stamp + self.duration_s:
                    self.last_stamp = t
                    event_window = np.array(event_list)
                    return event_window

        raise StopIteration
