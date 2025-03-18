import zipfile
from os.path import splitext
import numpy as np
from .timers import Timer
import os
import cv2
from tqdm import tqdm
import h5py

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

class FixedDurationNPZFileIterator:
    def __init__(self, folder_path, duration_ms, zero_start_time=True):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        self.file_list = sorted(self.file_list)
        self.index = 0

        # duration_ms is the duration of the event window in milliseconds
        self.duration_s = duration_ms * 10**-3
        self.events = np.zeros(shape=(1,4))

        if zero_start_time:
            self.current_time = 0
        else:
            # Load the first file and get the start time and assign it as the current time
            data = np.load(os.path.join(self.folder_path, self.file_list[self.index]))['event_data'].astype(np.float64)
            self.current_time = data[0,0]/10**9

        # Add the tqdm progress bar
        self.pbar = tqdm(total=len(self.file_list), desc='NPZ fixed duration iterator progress')

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.file_list) and self.events.shape[0] == 0:
            self.pbar.close()
            raise StopIteration
        
        # Get the end time of the event window and add data to the event window until the end time is reached
        self.current_time = self.current_time + self.duration_s
        self.add_data(self.current_time)

        # Find the event idx where the time is closest to the new_current_time and return it
        idx = np.searchsorted(self.events[:,0], self.current_time, side="left")
        
        event_window = self.events[:idx]
        self.events = self.events[idx:]
        return event_window
    
    def add_data(self, time_s):
        while (self.index < len(self.file_list)) and (self.events[-1][0] < time_s):
            file_path = os.path.join(self.folder_path, self.file_list[self.index])
            data = np.load(file_path)['event_data'].astype(np.float64)
            data[:,0] = data[:,0] / 10**9
            self.events = np.concatenate((self.events, data))
            self.index += 1
            self.pbar.update(1)
        
        # If the first event data is zeros, remove it
        if np.all(self.events[0] == np.zeros(shape=(1,4))):
            self.events = self.events[1:]

class FixedDurationH5FileIterator:
    def __init__(self, event_file_path, duration_ms, zero_start_time=True):
        self.folder_path = event_file_path
        self.file = h5py.File(event_file_path, 'r')
        self.ms_to_idx = self.file['ms_to_idx'][:]
        self.zero_start_time = zero_start_time
        self.fps = 10
        step_ms = int(1e3 / self.fps)
        self.index = 0

        self.metadata_list = []

        for idx in range(0, 9):
            if idx*step_ms >= len(self.ms_to_idx):
                continue

            self.metadata_list.append({
                'event_path': event_file_path,
                'start_ms': int(step_ms*idx),
                'end_ms': int(step_ms*(idx+1))
            })
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.metadata_list):
            raise StopIteration

        metadata = self.metadata_list[self.index]
        event_path = metadata['event_path']
        time_window = [metadata['start_ms'], metadata['end_ms']]
        time_interval = metadata['end_ms'] - metadata['start_ms']

        sidx = self.ms_to_idx[metadata['start_ms']]
        eidx = self.ms_to_idx[metadata['end_ms']]

        t = self.file['events/t'][sidx:eidx].astype(np.float64)
        y = self.file['events/y'][sidx:eidx].astype(np.float64)
        x = self.file['events/x'][sidx:eidx].astype(np.float64)
        p = self.file['events/p'][sidx:eidx].astype(np.float64)

        self.index += 1
        return np.array([t, x, y, p]).T

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
        import pandas as pd
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
