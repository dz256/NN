#Dana Zemel zemel.dana@gmail.com, based on a script by Hua-an Tseng, huaantseng@gmail.com


import collections
import imgaug
from imgaug import augmenters as iaa
import keras
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import PIL
from scipy import ndimage
import scipy.io as sio
import scipy.signal as signal
import scipy.stats as stats
from skimage import feature
from skimage import morphology
import tifffile
import time
from tqdm import tqdm



class test_data_generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_folder_list, model, frequency_idx=None, std_normalization=True, one_normalization=True, channels_order='channels_first', remove_spike=False, apply_interpolation=True):
        'Initialization'
        self.dataset_folder_list = dataset_folder_list     
        self.model = model
        self.frequency_idx = frequency_idx
        self.std_normalization = std_normalization
        self.one_normalization = one_normalization
        self.channels_order = channels_order
        self.remove_spike = remove_spike
        self.apply_interpolation= apply_interpolation

        self.category_number = model.output_shape[1]

        if self.channels_order=='channels_last':          
            self.channel_number = model.input_shape[3]
            self.data_length = int((model.input_shape[2]-1)/2)
            self.channel_axis = 3
        else:
            self.channel_number = model.input_shape[1]
            self.data_length = int((model.input_shape[3]-1)/2)
            self.channel_axis = 1

        self.data_info = []

        self.data_file_list = []
        self.dataset = {}
        self.predict_output = {}
        self.predict_event = {}

        # Go through all folders
        for dataset_folder in dataset_folder_list:
            dataset_folder = os.path.normpath(dataset_folder)

            # Get image file names from folder
            data_filename_list = next(os.walk(dataset_folder))[2]

            # Add data
            for current_data_filename in tqdm(data_filename_list, desc='Files'):
                current_data_path = os.path.join(dataset_folder, current_data_filename)
                self.data_file_list.append(current_data_path)
                [current_data_filename_base, current_data_filename_ext] = os.path.splitext(current_data_filename)

                trace,_,trace_cfs,trace_cfs_freq,_ = self.load_mat_data(current_data_path,remove_spike=self.remove_spike,apply_interpolation=self.apply_interpolation)

                #self.trace_data[current_data_path] = np.squeeze(i_trace)
                #self.trace_time_data[current_data_path] = np.squeeze(i_trace_time)
                #self.trace_cfs_data[current_data_path] = np.squeeze(i_trace_cfs)
                #self.trace_cfs_freq_data[current_data_path] = np.squeeze(i_trace_cfs_freq)
                self.predict_output[current_data_path] = np.zeros((trace.shape[0],2))

                # put everything into data_info

                idx_list = range(self.data_length,trace_cfs.shape[1]-self.data_length)

                for current_idx in idx_list:
                    data_info = {
                        "id": current_data_filename_base+"_"+str(current_idx),
                        "path": current_data_path,
                        "idx": current_idx,
                    }
                    self.data_info.append(data_info)

        freq_idx_max = trace_cfs.shape[0] 
        freq_idx_min = 0
        print("Frequency range: {:1.2f}-{:1.2f} Hz (max idx: {:1.2f} ).".format(trace_cfs_freq[freq_idx_max-1],trace_cfs_freq[freq_idx_min],freq_idx_max))
        try:                   
            if len(self.frequency_idx)==2:
                freq_idx_min = np.amax([freq_idx_min,self.frequency_idx[0]])
                freq_idx_max = np.amin([freq_idx_max,self.frequency_idx[1]+1])
            elif len(self.frequency_idx)==1:
                freq_idx_min = np.amax([freq_idx_min,freq_idx_max-self.frequency_idx[0]])
        except:
            pass
        self.frequency_idx = [freq_idx_min,freq_idx_max]
        self.freuqency_idx_length = self.frequency_idx[1]-self.frequency_idx[0]
        print("Selected frequency range: {:1.2f}-{:1.2f} Hz ({:1.2f} data points).".format(trace_cfs_freq[self.frequency_idx[1]-1],trace_cfs_freq[self.frequency_idx[0]],self.freuqency_idx_length))
        
        print("Total {} samples.".format(len(self.data_info)))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.data_info)

    def __getitem__(self):
        'Generate one batch of data'
        # Generate indexes of the batch

        if self.channel_axis == 1:
            current_batch_data = np.zeros((len(self), self.channel_number, self.freuqency_idx_length, 2*self.data_length+1))
        else:
            current_batch_data = np.zeros((len(self), self.freuqency_idx_length, 2*self.data_length+1, self.channel_number))

        for data_id in tqdm(range(len(self)),desc='Preparing data'):
            current_data_path = self.data_info[data_id]['path']
            current_idx = self.data_info[data_id]['idx']

            _,_,current_data,_,_ = self.load_mat_data(current_data_path,remove_spike=self.remove_spike,apply_interpolation=self.apply_interpolation)
            
            if current_data.ndim==2:
                current_data = np.expand_dims(current_data, axis=0)
            current_data = current_data[:, self.frequency_idx[0]:self.frequency_idx[1],current_idx-self.data_length:current_idx+self.data_length+1]
            current_data = current_data.astype('float')

            if self.std_normalization:
                for channel_idx in range(current_data.shape[0]):
                    current_mean = np.mean(current_data[channel_idx,:,:])
                    current_std = np.std(current_data[channel_idx,:,:])
                    current_data[channel_idx,:,:] = (current_data[channel_idx,:,:]-current_mean)/current_std
            if self.one_normalization:  
                for channel_idx in range(current_data.shape[0]):
                    current_data[channel_idx,:,:] = (current_data[channel_idx,:,:]-np.amin(current_data[channel_idx,:,:]))/(np.amax(current_data[channel_idx,:,:])-np.amin(current_data[channel_idx,:,:]))
            
            if self.channel_axis == 3:
                current_data = np.rollaxis(current_data, 0, 3)  # convert to NHWC
            
            current_batch_data[data_id, :, :, :] = current_data
                        
        return current_batch_data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def load_mat_data(self,filename,remove_spike=False,apply_interpolation=False):
        try:
            current_mat_data = self.dataset[filename]
        except:
            current_mat_data = sio.loadmat(filename, struct_as_record=False)
            self.dataset[filename] = current_mat_data
        
        if remove_spike and apply_interpolation:
            trace = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_rs_trace'))
            trace_time = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_rs_trace_time'))
            trace_cfs = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_rs_trace_cfs'))
            trace_cfs_freq = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_rs_trace_cfs_freq'))
            try:
                event = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_rs_event'))
            except:
                event = np.array([])
        elif remove_spike and not(apply_interpolation):
            trace = np.squeeze(getattr(current_mat_data['roi'][0][0], 'rs_trace'))
            trace_time = np.squeeze(getattr(current_mat_data['roi'][0][0], 'rs_trace_time'))
            trace_cfs = np.squeeze(getattr(current_mat_data['roi'][0][0], 'rs_trace_cfs'))
            trace_cfs_freq = np.squeeze(getattr(current_mat_data['roi'][0][0], 'rs_trace_cfs_freq'))
            try:
                event = np.squeeze(getattr(current_mat_data['roi'][0][0], 'rs_event'))
            except:
                event = np.array([])
        elif not(remove_spike) and apply_interpolation:
            trace = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_trace'))
            trace_time = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_trace_time'))
            trace_cfs = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_trace_cfs'))
            trace_cfs_freq = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_trace_cfs_freq'))
            try:
                event = np.squeeze(getattr(current_mat_data['roi'][0][0], 'i_event'))
            except:
                event = np.array([])
        else:
            trace = np.squeeze(getattr(current_mat_data['roi'][0][0], 'trace'))
            trace_time = np.squeeze(getattr(current_mat_data['roi'][0][0], 'trace_time'))
            trace_cfs = np.squeeze(getattr(current_mat_data['roi'][0][0], 'trace_cfs'))
            trace_cfs_freq = np.squeeze(getattr(current_mat_data['roi'][0][0], 'trace_cfs_freq'))
            try:
                event = np.squeeze(getattr(current_mat_data['roi'][0][0], 'event'))
            except:
                event = np.array([])

        return trace,trace_time,trace_cfs,trace_cfs_freq,event

    def predict(self):
        
        predict_output = self.model.predict(self.__getitem__())

        for data_id, single_predict_output in tqdm(enumerate(predict_output),desc='Processing prediction'):
            self.data_info[data_id]["predict_output"] = single_predict_output

            current_data_path = self.data_info[data_id]["path"]
            current_idx = self.data_info[data_id]["idx"]

            if self.category_number==1:
                single_predict_output = np.array([(np.ones(single_predict_output.shape)-single_predict_output),single_predict_output]).T

            single_predict_output = single_predict_output+self.predict_output[current_data_path][current_idx,:]
            self.predict_output[current_data_path][current_idx,:] = single_predict_output
    
    def create_event(self, event_threshold=0.5):

        for key in tqdm(self.predict_output.keys(),desc='Creating event'):
            predict_output = self.predict_output[key].copy()
            predict_output_sum = np.sum(predict_output, axis=-1)
            predict_output = predict_output[:,1]/(np.finfo(float).eps+predict_output_sum)
            binary_output = np.zeros(predict_output.shape)
            binary_output[predict_output>= event_threshold] = 1

            self.predict_event[key] = binary_output

    def save_predict_roi(self, roi_folder="roi", roi_tag="_roi"):
        for key in tqdm(self.predict_output.keys(),desc='Saving ROI'):
            [current_filefolder,current_filename] = os.path.split(key)
            [current_image_filename_base, current_image_filename_ext] = os.path.splitext(current_filename)
            save_filename = os.path.join(current_filefolder,roi_folder,current_image_filename_base+roi_tag+".tif")
            if os.path.isdir(os.path.join(current_filefolder,roi_folder)) is False:
                os.makedirs(os.path.join(current_filefolder,roi_folder))
            tifffile.imwrite(save_filename, self.predict_roi_masks[key].astype(int))

    def show_predict_event(self,file_idx_list=None,time_window=None, **kwargs):
        if file_idx_list is None:
            file_idx_list = range(len(self.data_file_list))

        for idx,file_idx in enumerate(file_idx_list):
            current_data_path = self.data_file_list[file_idx]
            current_trace,current_trace_time,_,_,ground_truth = self.load_mat_data(current_data_path,remove_spike=self.remove_spike,apply_interpolation=self.apply_interpolation)

            current_predict_output = self.predict_output[current_data_path]
            current_predict_event_idx = np.where(self.predict_event[current_data_path]==1)

            fig = plt.figure(figsize=(2 * 10, 2 * 10))
            gs = gridspec.GridSpec(len(file_idx_list), 3, width_ratios=[1, 1, 1], wspace=0.1, hspace=0.1)
            ax = fig.add_subplot(gs[idx, 0])
            ax.plot(current_trace_time,current_trace,color='black',linewidth=0.5,alpha=0.7)
            ax.scatter(current_trace_time[current_predict_event_idx],current_trace[current_predict_event_idx],color='red')
            try:
                current_event_idx = np.where(ground_truth==1)
                ax.scatter(current_trace_time[current_event_idx],current_trace[current_event_idx],color='blue',marker='x')
            except:
                pass
            if time_window is not None:
                plt.xlim(time_window)

            ax = fig.add_subplot(gs[idx, 1])
            ax.plot(current_trace_time,current_predict_output[:,0],color='black',linewidth=0.5,alpha=0.7)
            ax.scatter(current_trace_time[current_predict_event_idx],current_predict_output[current_predict_event_idx,0],color='red')
            try:
                #current_event_idx = np.where(ground_truth==1)
                ax.scatter(current_trace_time[current_event_idx],current_predict_output[current_event_idx,0],color='blue',marker='x')
            except:
                pass
            if time_window is not None:
                plt.xlim(time_window)

            ax = fig.add_subplot(gs[idx, 2])
            ax.plot(current_trace_time,current_predict_output[:,1],color='black',linewidth=0.5,alpha=0.7)
            ax.scatter(current_trace_time[current_predict_event_idx],current_predict_output[current_predict_event_idx,1],color='red')
            try:
                #current_event_idx = np.where(ground_truth==1)
                ax.scatter(current_trace_time[current_event_idx],current_predict_output[current_event_idx,1],color='blue',marker='x')
            except:
                pass
            if time_window is not None:
                plt.xlim(time_window)

            plt.show()
        


class train_data_generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_folder_list, data_length=1000, frequency_idx=None, category_number=5, channel_number=1, batch_size=64, batch_number=None, shuffle=True, event_ratio=0.5, window_label=False, std_normalization=True, one_normalization=True, channels_order='channels_last', image_augmentation=None):#,  apply_interpolation=True):
        'Initialization'
        self.dataset_folder_list = dataset_folder_list
        self.data_length = data_length
        self.frequency_idx = frequency_idx
        self.category_number = category_number
        self.channel_number = channel_number
        self.batch_size = batch_size
        self.batch_number = batch_number
        self.shuffle = shuffle
        self.event_ratio = event_ratio
        self.window_label = window_label
        self.std_normalization = std_normalization
        self.one_normalization = one_normalization
        self.channels_order = channels_order
        if self.channels_order=='channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3
        self.image_augmentation = image_augmentation
        #self.apply_interpolation= apply_interpolation

        self.data_info = []
        self.dataset = {}

        if self.image_augmentation=='keras':
            print("use Keras for image augmentation")
            self.image_preprocessing_args = dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-06,
                rotation_range=360,
                width_shift_range=0,
                height_shift_range=0,
                # brightness_range=[0.5, 1.5],
                shear_range=0,
                zoom_range=0,
                fill_mode='reflect',  # "constant", "nearest", "reflect" or "wrap"
                # cval=0.0,
                horizontal_flip=True,
                vertical_flip=True,
                # rescale=None,
                data_format=self.channels_order
            )

            self.label_preprocessing_args = self.image_preprocessing_args
            self.label_preprocessing_args['featurewise_center'] = False
            self.label_preprocessing_args['samplewise_center'] = False
            self.label_preprocessing_args['featurewise_std_normalization'] = False
            self.label_preprocessing_args['samplewise_std_normalization'] = False
            self.label_preprocessing_args['zca_whitening'] = False
        elif self.image_augmentation=='imgaug':
            print("use imgaug for image augmentation")
            self.image_preprocessing_args = [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                #iaa.CropAndPad(percent=(-0.25, 0.25),pad_mode=["edge"]),
                #iaa.Multiply((0.1, 1.1)),
                #iaa.MultiplyElementwise((0.5, 1.5)),
                iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0.0, 1.0))),
                #iaa.AdditiveGaussianNoise(scale=(0, 0.001*2**16)),
                #iaa.ContrastNormalization((0.5, 1.5)),
                #iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                iaa.Dropout(p=(0, 0.1)),
                iaa.Affine(scale=1,rotate=(-90, 90), mode=['symmetric','reflect','wrap']),
                #iaa.Affine(shear=(-45, 45), mode=['symmetric','reflect','wrap']),
            ]
        else:
            print("No image augmentation")

        # Go through all folders
        for dataset_folder in dataset_folder_list:
            dataset_folder = os.path.normpath(dataset_folder)

            # Get image file names from folder
            data_filename_list = next(os.walk(dataset_folder))[2]

            # Add data
            for current_data_filename in tqdm(data_filename_list, desc='Files'):
                try:
                    current_data_path = os.path.join(dataset_folder, current_data_filename)
                    [current_data_filename_base, current_data_filename_ext] = os.path.splitext(current_data_filename)
                    _,_,trace_cfs,trace_cfs_freq,event = self.load_mat_data(current_data_path,remove_spike=self.remove_spike,apply_interpolation=self.apply_interpolation)

                    # put everything into data_info
                    idx_list = range(self.data_length,trace_cfs.shape[1]-self.data_length)
                    
                    for current_idx in idx_list:
                        if self.window_label:
                            if np.sum(event[current_idx-self.data_length:current_idx+self.data_length+1])>0:
                                current_event = 1
                            else:
                                current_event = 0

                        else:
                            if event[current_idx]==1:
                                current_event = 1
                            else:
                                current_event = 0
                        data_info = {
                            "id": current_data_filename_base+"_"+str(current_idx),
                            "path": current_data_path,
                            "idx": current_idx,
                            "event": current_event
                        }
                        self.data_info.append(data_info)
                except:
                    pass

        freq_idx_max = trace_cfs.shape[0] 
        freq_idx_min = 0
        print("Frequency range: {:1.2f}-{:1.2f} Hz (max idx: {:1.2f} ).".format(trace_cfs_freq[freq_idx_max-1],trace_cfs_freq[freq_idx_min],freq_idx_max))
        try:                   
            if len(self.frequency_idx)==2:
                freq_idx_min = np.amax([freq_idx_min,self.frequency_idx[0]])
                freq_idx_max = np.amin([freq_idx_max,self.frequency_idx[1]+1])
            elif len(self.frequency_idx)==1:
                freq_idx_min = np.amax([freq_idx_min,freq_idx_max-self.frequency_idx[0]])
        except:
            pass
        self.frequency_idx = [freq_idx_min,freq_idx_max]
        self.frequency_idx_length = self.frequency_idx[1]-self.frequency_idx[0]
        print("Selected frequency range: {:1.2f}-{:1.2f} Hz ({:1.2f} data points).".format(trace_cfs_freq[self.frequency_idx[1]-1],trace_cfs_freq[self.frequency_idx[0]],self.frequency_idx_length))
        print("Total {} samples.".format(len(self.data_info)))

        # calculate sample with events

        event_data_id_list = []
        non_event_data_id_list = []
        for data_id in tqdm(range(len(self.data_info)),desc="Calculating sample with events"):
            if self.data_info[data_id]['event']==1:
                event_data_id_list.append(data_id)
            elif self.data_info[data_id]['event']==0:
                non_event_data_id_list.append(data_id)
            else:
                pass
            
        print("Sample with event: {} ({}%)".format(len(event_data_id_list),100*len(event_data_id_list)/len(self.data_info)))
        print("Sample without event: {} ({}%)".format(len(non_event_data_id_list),100*len(non_event_data_id_list)/len(self.data_info)))

        self.event_data_id_list = event_data_id_list
        self.non_event_data_id_list = non_event_data_id_list

        total_id_count = np.ceil(np.amax((len(self.non_event_data_id_list)/(1-self.event_ratio),len(self.event_data_id_list)/self.event_ratio)))
        event_data_id_list = np.repeat(self.event_data_id_list,int(np.ceil((self.event_ratio*total_id_count)/len(self.event_data_id_list))))
        non_event_data_id_list = np.repeat(self.non_event_data_id_list,int(np.ceil(((1-self.event_ratio)*total_id_count)/len(self.non_event_data_id_list))))
        np.random.shuffle(event_data_id_list)
        np.random.shuffle(non_event_data_id_list)
        event_data_id_list = event_data_id_list[:int(self.event_ratio*total_id_count)]
        non_event_data_id_list = non_event_data_id_list[:int((1-self.event_ratio)*total_id_count)]

        print("Adjusted sample with event: {} ({}%)".format(len(event_data_id_list),100*len(event_data_id_list)/(len(event_data_id_list)+len(non_event_data_id_list))))
        print("Adjusted sample without event: {} ({}%)".format(len(non_event_data_id_list),100*len(non_event_data_id_list)/(len(event_data_id_list)+len(non_event_data_id_list))))

        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.data_info)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        data_id = index

        current_path = self.data_info[data_id]["path"]
        current_frame_idx = self.data_info[data_id]["frame_idx"]
        current_row_idx = self.data_info[data_id]["row_idx"]
        current_col_idx = self.data_info[data_id]["col_idx"]
        current_h_flip = self.data_info[data_id]["h_flip"]
        current_w_flip = self.data_info[data_id]["w_flip"]
        current_rotate = self.data_info[data_id]["rotate"]

        if self.current_data["current_path"]!=current_path or self.current_data["current_frame_idx"]!=current_frame_idx:
            current_data = tifffile.imread(current_path,key=range(current_frame_idx,current_frame_idx+self.frame_number))
            if current_data.ndim==2:
                current_data = np.expand_dims(current_data, axis=0)
            if current_path not in self.project_image:
                self.project_image[current_path] = np.amax(current_data,axis=0)
            self.current_data["data"] = current_data

        
        current_data = self.current_data["data"][:, current_row_idx:current_row_idx + self.patch_size[0],current_col_idx:current_col_idx + self.patch_size[1]]

        if self.channel_axis == 3:
            current_data = np.rollaxis(current_data, 0, 3)  # convert to NHWC

        if self.std_normalization:
            current_data = current_data.astype('float')
            if self.channel_axis==3:
                for channel_idx in range(current_data.shape[2]):
                    current_mean = np.mean(current_data[:,:,channel_idx])
                    current_std = np.std(current_data[:,:,channel_idx])
                    current_data[:,:,channel_idx] = (current_data[:,:,channel_idx]-current_mean)/current_std
            else:
                for channel_idx in range(current_data.shape[0]):
                    current_mean = np.mean(current_data[channel_idx,:,:])
                    current_std = np.std(current_data[channel_idx,:,:])
                    current_data[channel_idx,:,:] = (current_data[channel_idx,:,:]-current_mean)/current_std

        current_data = np.expand_dims(current_data, axis=0)

        if current_h_flip:
            current_data = np.flip(current_data, axis=self.h_axis)
        if current_w_flip:
            current_data = np.flip(current_data, axis=self.w_axis)
        current_data = np.rot90(current_data, current_rotate, axes=(self.h_axis, self.w_axis))
   
        return current_data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def predict(self):
        for data_id in tqdm(range(len(self)),desc='Predicting'):
            predict_output = self.model.predict(self.__getitem__(data_id))
            predict_output = predict_output[0]
            if self.channel_axis == 1:
                predict_output = np.rollaxis(predict_output, 0, 3)  # make sure category at last channel
            self.data_info[data_id]["predict_output"] = predict_output

            current_image_path = self.data_info[data_id]["path"]
            current_row_idx = self.data_info[data_id]["row_idx"]
            current_col_idx = self.data_info[data_id]["col_idx"]
            current_h_flip = self.data_info[data_id]["h_flip"]
            current_w_flip = self.data_info[data_id]["w_flip"]
            current_rotate = self.data_info[data_id]["rotate"]

            if current_h_flip:
                predict_output = np.flip(predict_output, axis=0)
            if current_w_flip:
                predict_output = np.flip(predict_output, axis=1)
            predict_output = np.rot90(predict_output, -1*current_rotate, axes=(0, 1))

            predict_output = predict_output+self.predict_output[current_image_path][current_row_idx:current_row_idx+self.patch_size[0],current_col_idx:current_col_idx+self.patch_size[1],:]
            self.predict_output[current_image_path][current_row_idx:current_row_idx+self.patch_size[0],current_col_idx:current_col_idx+self.patch_size[1],:] = predict_output
    
    def create_roi(self, roi_threshold=0.5, roi_size_limit=[0,float('inf')], erosion_iter=None, watershed=False):
        def apply_watershed(binary_mask):
            # http://www.scipy-lectures.org/advanced/image_processing/auto_examples/plot_watershed_segmentation.html#sphx-glr-advanced-image-processing-auto-examples-plot-watershed-segmentation-py
            distance = ndimage.distance_transform_edt(binary_mask)
            local_max_idx = feature.peak_local_max(distance, indices=False, footprint=np.ones((20, 20)), labels=binary_mask)
            markers = ndimage.label(local_max_idx)[0]
            roi_mask = morphology.watershed(-distance, markers, mask=binary_mask)
            return roi_mask

        for key in tqdm(self.predict_output.keys(),desc='Creating ROI'):
            predict_output = self.predict_output[key].copy()
            predict_output_sum = np.sum(predict_output, axis=-1)
            predict_output = predict_output/(np.finfo(float).eps+np.repeat(predict_output_sum[:,:,np.newaxis],self.catagory_number,axis=2))
            binary_output = np.zeros(predict_output.shape)
            binary_output[predict_output>= roi_threshold] = 1

            current_binary_mask = np.squeeze(binary_output[:,:,1])
            if erosion_iter is not None and erosion_iter != 0:
                current_binary_mask = ndimage.binary_erosion(current_binary_mask, iterations=erosion_iter).astype(current_binary_mask.dtype)
            # remove small areas
            current_binary_mask = ndimage.binary_opening(current_binary_mask)
            # remove small holes
            current_binary_mask = ndimage.binary_closing(current_binary_mask)
            # convert back to int
            current_binary_mask = current_binary_mask.astype('int')
            if watershed:
                current_roi_mask = apply_watershed(current_binary_mask)
            else:
                [current_roi_mask, _] = ndimage.label(current_binary_mask)
            current_roi_sizes = ndimage.sum(current_binary_mask, current_roi_mask, range(current_roi_mask.max()+1))
            try:
                remove_roi_idx = np.logical_or(current_roi_sizes<roi_size_limit[0], current_roi_sizes>roi_size_limit[1])
            except:
                remove_roi_idx = current_roi_sizes<roi_size_limit
            remove_roi_mask = remove_roi_idx[current_roi_mask]
            current_roi_mask[remove_roi_mask] = 0
            unique_roi_idx = np.unique(current_roi_mask)
            current_roi_mask = np.searchsorted(unique_roi_idx, current_roi_mask)
            self.predict_roi_masks[key] = current_roi_mask

    def save_predict_roi(self, roi_folder="roi", roi_tag="_roi"):
        for key in tqdm(self.predict_output.keys(),desc='Saving ROI'):
            [current_filefolder,current_filename] = os.path.split(key)
            [current_image_filename_base, current_image_filename_ext] = os.path.splitext(current_filename)
            save_filename = os.path.join(current_filefolder,roi_folder,current_image_filename_base+roi_tag+".tif")
            if os.path.isdir(os.path.join(current_filefolder,roi_folder)) is False:
                os.makedirs(os.path.join(current_filefolder,roi_folder))
            tifffile.imwrite(save_filename, self.predict_roi_masks[key].astype(int))

    def show_predict_roi(self,file_idx=0,area_idx=None, **kwargs):
        # colors='red',linewidths=0.1

        current_image_path = self.image_file_list[file_idx]
        current_image = self.project_image[current_image_path]
        current_predict_roi_mask = self.predict_roi_masks[current_image_path]
        try:
            current_know_mask = self.known_masks[self.roi_file_list[current_image_path]]
        except:
            current_know_mask = []
        

        if area_idx==None:
            row_idx = slice(0, current_image.shape[0])
            col_idx = slice(0, current_image.shape[1])
        else:
            row_idx = slice(area_idx[0][0], area_idx[0][1])
            col_idx = slice(area_idx[1][0], area_idx[1][1])

        fig = plt.figure(figsize=(2 * 10, 1 * 10))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.1, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(current_image[row_idx, col_idx]) 
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(current_image[row_idx, col_idx])
        ax.contour(current_predict_roi_mask[row_idx, col_idx], **kwargs)
        try:
            ax = fig.add_subplot(gs[0, 2])
            ax.imshow(current_image[row_idx, col_idx])
            ax.contour(current_know_mask[row_idx, col_idx], **kwargs)
        except:
            pass
        plt.show()

    def show_predict_output(self,file_idx=0,area_idx=None, **kwargs):
        # colors='red',linewidths=0.1

        current_image_path = self.image_file_list[file_idx]
        current_image = self.project_image[current_image_path]
        current_predict_output = self.predict_output[current_image_path]
        current_predict_roi_mask = self.predict_roi_masks[current_image_path]
        try:
            current_know_mask = self.known_masks[self.roi_file_list[current_image_path]]
        except:
            current_know_mask = []
        

        if area_idx==None:
            row_idx = slice(0, current_image.shape[0])
            col_idx = slice(0, current_image.shape[1])
        else:
            row_idx = slice(area_idx[0][0], area_idx[0][1])
            col_idx = slice(area_idx[1][0], area_idx[1][1])

        fig = plt.figure(figsize=(2 * 10, 1 * 10))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.1, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(current_image[row_idx, col_idx])
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(current_predict_output[row_idx, col_idx,0])
        ax.contour(current_predict_roi_mask[row_idx, col_idx], **kwargs)
        try:        
            ax.contour(current_know_mask[row_idx, col_idx],linestyles='dashed',**kwargs)
        except:
            pass
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(current_predict_output[row_idx, col_idx,1])
        ax.contour(current_predict_roi_mask[row_idx, col_idx], **kwargs)
        try:        
            ax.contour(current_know_mask[row_idx, col_idx],linestyles='dashed',**kwargs)
        except:
            pass
        plt.show()

class roi_unet_test:
    def __init__(self, filenames, frame_number, catagory_number=2, std_normalization=True, channels_order='channels_first'):

        if not isinstance(filenames[0],list):
            filenames = [filenames]

        self.original_filename = filenames
        self.frame_number = frame_number
        self.catagory_number = catagory_number
        self.label_type_number = 2
        self.known_masks = [''] * len(filenames[0])
        self.predict_masks_all = [''] * len(filenames[0])
        self.predict_masks = [''] * len(filenames[0])
        self.binary_masks = [''] * len(filenames[0])
        self.roi_masks = [''] * len(filenames[0])
        self.max_image = [''] * len(filenames[0])
        self.batch_images = [''] * len(filenames[0])
        self.std_normalization = std_normalization
        self.channels_order = channels_order
        if self.channels_order=='channels_first':
            self.channel_axis = 1
            self.h_axis = 2
            self.w_axis = 3
        else:
            self.h_axis = 1
            self.w_axis = 2
            self.channel_axis = 3

        self.image_preprocessing_args = dict(
            # featurewise_center=True,
            # samplewise_center=True,
            # featurewise_std_normalization=True,
            # samplewise_std_normalization=True,
            # zca_whitening=False,
            # zca_epsilon=1e-06
            # rotation_range=90,
            # width_shift_range=0.4,
            # height_shift_range=0.4,
            # brightness_range=[0.5, 1.5],
            # shear_range=0.4,
            # zoom_range=0.4,
            # fill_mode='reflect',  # "constant", "nearest", "reflect" or "wrap"
            # cval=0.0,
            # horizontal_flip=True,
            # vertical_flip=True,
            # rescale=None,
            data_format=self.channels_order
        )


        print("Total %s datasets." % len(filenames[0]))

    def add_batch_image(self, dataset_idx):
        # construct the image from batch data, and save it to self.batch_image
        current_batch_image = []
        for batch_idx in range(len(self.batch_data)):
            current_batch_data = self.batch_data[batch_idx]
            current_batch_idx_list = self.batch_idx_list[batch_idx]
            current_image = self.patch_to_image(self.data_size, current_batch_idx_list, current_batch_data)
            current_batch_image.append(current_image)
        self.batch_images[dataset_idx] = current_batch_image

    def add_predict_result(self, dataset_idx, batch_predict):

        print("Start add predict result....")
        add_predict_result_start_time = time.time()

        if self.channel_axis == 3:
            current_predict_mask = np.zeros((self.data_size[0],self.data_size[1],self.catagory_number))
        else:
            current_predict_mask = np.zeros((self.catagory_number,self.data_size[0], self.data_size[1]))

        for batch_idx in range(len(batch_predict)):
            current_batch_predict = batch_predict[batch_idx]
            current_batch_idx_list = self.batch_idx_list[batch_idx]

            current_row_idx = current_batch_idx_list[0]
            current_col_idx = current_batch_idx_list[1]

            current_batch_predict = np.rot90(current_batch_predict, current_batch_idx_list[2], axes=(self.h_axis-1, self.w_axis-1))

            if current_batch_idx_list[3]==1:
                current_batch_predict = np.flip(current_batch_predict, axis=self.h_axis-1)

            if current_batch_idx_list[4]==1:
                current_batch_predict = np.flip(current_batch_predict, axis=self.w_axis-1)

            current_batch_predict = np.argmax(current_batch_predict, axis=self.channel_axis-1)
            current_batch_predict = keras.utils.to_categorical(current_batch_predict, num_classes=self.catagory_number)

            if self.channel_axis == 3:
                current_batch_predict = current_batch_predict+current_predict_mask[current_row_idx:current_row_idx + current_batch_predict.shape[0],current_col_idx:current_col_idx + current_batch_predict.shape[1], :]
                current_predict_mask[current_row_idx:current_row_idx+current_batch_predict.shape[0],current_col_idx:current_col_idx+current_batch_predict.shape[1],:] = current_batch_predict
            else:
                current_batch_predict = np.rollaxis(current_batch_predict, 2, 0)
                current_batch_predict = current_batch_predict+current_predict_mask[:,current_row_idx:current_row_idx+current_batch_predict.shape[1],current_col_idx:current_col_idx +current_batch_predict.shape[2]]
                current_predict_mask[:,current_row_idx:current_row_idx + current_batch_predict.shape[1],current_col_idx:current_col_idx + current_batch_predict.shape[2]] = current_batch_predict


        self.predict_masks_all[dataset_idx] = current_predict_mask
        self.predict_masks[dataset_idx] = np.argmax(current_predict_mask, axis=self.channel_axis-1)

        print("\tAdding time: %s seconds." % round((time.time() - add_predict_result_start_time), 2))

    def batch_highpass_flt(self, sigma=50):

        print("Start highpass filtering batch....")
        normalization_start_time = time.time()

        for batch_idx in tqdm(range(len(self.batch_data))):
            batch_data = self.batch_data[batch_idx]

            batch_data_idx_list = range(0, batch_data.shape[0])
            for batch_data_idx in batch_data_idx_list:
                data = batch_data[batch_data_idx]
                lowpass_data = ndimage.gaussian_filter(data, sigma)
                batch_data[batch_data_idx] = data-lowpass_data
            self.batch_data[batch_idx] = batch_data
        print("\tHighpass filtering time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def batch_hist_eq(self):

        print("Start histogram equalizing batch....")
        normalization_start_time = time.time()

        for batch_idx in tqdm(range(len(self.batch_data))):
            batch_data = self.batch_data[batch_idx]

            batch_data_idx_list = range(0, batch_data.shape[0])
            for batch_data_idx in batch_data_idx_list:
                data = batch_data[batch_data_idx]

                dist, bins = np.histogram(data.flatten(), 2 ** 16, density=True)
                cum_dist = dist.cumsum()
                cum_dist = 2 ** 16 * cum_dist / [cum_dist[-1]]
                new_data = np.interp(data.flatten(), bins[:-1], cum_dist)
                batch_data[batch_data_idx] = new_data.reshape(data.shape)
            self.batch_data[batch_idx] = batch_data
        print("\tHistogram equalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def batch_std_eq(self):
        print("Start normalizing batch")
        normalization_start_time = time.time()

        for batch_idx in tqdm(range(len(self.batch_data))):
            batch_data = self.batch_data[batch_idx]

            batch_data_idx_list = range(0, batch_data.shape[0])
            for batch_data_idx in batch_data_idx_list:
                data = batch_data[batch_data_idx]
                data_mean = np.mean(data, axis=None)
                data_std = np.sqrt(np.var(data, axis=None))
                batch_data[batch_data_idx] = (data - data_mean) / data_std

            self.batch_data[batch_idx] = batch_data
        print("\tNormalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def create_binary_mask(self, dataset_idx_list=None, mask_threshold=0.5):
        if dataset_idx_list is None:
            dataset_idx_list = range(len(self.binary_masks))

        for dataset_idx in tqdm(dataset_idx_list):
            try:
                predict_mask = self.predict_masks_all[dataset_idx]
                predict_mask_sum = np.sum(predict_mask, axis=-1)
                predict_mask = predict_mask/np.repeat(predict_mask_sum[:,:,np.newaxis],self.catagory_number,axis=2)
                binary_masks = np.zeros(predict_mask.shape[:2])
                for category_idx in range(1,self.catagory_number):
                    binary_masks[predict_mask[:,:,category_idx] >= mask_threshold] = 1
                self.binary_masks[dataset_idx] = binary_masks
            except:
                print("No predicted mask for dataset %s." % dataset_idx)

    def create_roi(self, dataset_idx_list=None, roi_size_limit=[0,float('inf')], erosion_iter=None, watershed=False):
        def apply_watershed(binary_mask):
            # http://www.scipy-lectures.org/advanced/image_processing/auto_examples/plot_watershed_segmentation.html#sphx-glr-advanced-image-processing-auto-examples-plot-watershed-segmentation-py
            distance = ndimage.distance_transform_edt(binary_mask)
            local_max_idx = feature.peak_local_max(distance, indices=False, footprint=np.ones((20, 20)), labels=binary_mask)
            markers = ndimage.label(local_max_idx)[0]
            roi_mask = morphology.watershed(-distance, markers, mask=binary_mask)
            return roi_mask

        roi_size_limit = np.array(roi_size_limit)
        if dataset_idx_list is None:
            dataset_idx_list = range(len(self.binary_masks))

        for dataset_idx in tqdm(dataset_idx_list):
            try:
                current_binary_mask = self.binary_masks[dataset_idx]
                if erosion_iter is not None and erosion_iter != 0:
                    current_binary_mask = ndimage.binary_erosion(current_binary_mask, iterations=erosion_iter).astype(current_binary_mask.dtype)
                # remove small areas
                current_binary_mask = ndimage.binary_opening(current_binary_mask)
                # remove small holes
                current_binary_mask = ndimage.binary_closing(current_binary_mask)
                # convert back to int
                current_binary_mask = current_binary_mask.astype('int')
                if watershed:
                    current_roi_mask = apply_watershed(current_binary_mask)
                else:
                    [current_roi_mask, _] = ndimage.label(current_binary_mask)
                current_roi_sizes = ndimage.sum(current_binary_mask, current_roi_mask, range(current_roi_mask.max()+1))
                try:
                    remove_roi_idx = np.logical_or(current_roi_sizes<roi_size_limit[0], current_roi_sizes>roi_size_limit[1])
                except:
                    remove_roi_idx = current_roi_sizes<roi_size_limit
                remove_roi_mask = remove_roi_idx[current_roi_mask]
                current_roi_mask[remove_roi_mask] = 0
                unique_roi_idx = np.unique(current_roi_mask)
                current_roi_mask = np.searchsorted(unique_roi_idx, current_roi_mask)
                self.roi_masks[dataset_idx] = current_roi_mask
            except:
                print("No binary mask for dataset %s." % dataset_idx)

    def data_0_1(self):

        print("Start normalizing data to 0-1 (range: %g-%g)" % (
        np.amin(self.data, axis=None), np.amax(self.data, axis=None)))
        normalization_start_time = time.time()
        data = self.data
        data_range = np.amax(data, axis=None) - np.amin(data, axis=None)
        self.data = (data - np.amin(data, axis=None)) / data_range
        print("\tNormalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def data_add_corr(self):

        print("Start calculating correlation coefficient....")
        corr_start_time = time.time()
        data = self.data
        corr_image = np.zeros([data.shape[0], data.shape[1], 1])
        for row_idx in range(self.extend_pixel_number, self.extend_pixel_number + self.data_size[0] - 1):
            for col_idx in range(self.extend_pixel_number, self.extend_pixel_number + self.data_size[1] - 1):
                current_trace = data[row_idx][col_idx]
                target_trace = data[row_idx][col_idx + 1]

                m_current_trace = current_trace - current_trace.mean()
                m_target_trace = target_trace - target_trace.mean()

                corr_image[row_idx][col_idx][0] = np.dot(m_current_trace, m_target_trace.T) / np.sqrt(
                    np.dot((m_current_trace ** 2).sum(), (m_target_trace ** 2).sum()))

        self.corr[:, :, 0] = corr_image[:, :, 0]

        print("\tCalculating time: %s seconds." % round((time.time() - corr_start_time), 2))

    def data_filter(self, frequency_cutoff):

        print("Start low-pass filtering data at %g Hz...." % frequency_cutoff)
        filter_start_time = time.time()
        filtered_data = self.data

        frequency_cutoff = float(frequency_cutoff) / 20
        filter_b, filter_a = signal.butter(8, frequency_cutoff)  # frame rate is 20 Hz
        self.data = signal.filtfilt(filter_b, filter_a, filtered_data, axis=(filtered_data.ndim - 1))
        print("\tFilteringing time: %s seconds." % round((time.time() - filter_start_time), 2))

    def data_frame_hist_eq(self):

        print("Start histogram equalizing frame....")
        normalization_start_time = time.time()

        frame_data_idx_list = range(0, self.data.shape[2])
        for frame_data_idx in frame_data_idx_list:
            data = self.data[:, :, frame_data_idx]

            dist, bins = np.histogram(data.flatten(), 2 ** 16, density=True)
            cum_dist = dist.cumsum()
            cum_dist = 2 ** 16 * cum_dist / [cum_dist[-1]]
            new_data = np.interp(data.flatten(), bins[:-1], cum_dist)
            self.data[:, :, frame_data_idx] = new_data.reshape(data.shape)
        print("\tHistogram equalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def data_std_eq(self):

        normalization_start_time = time.time()
        data = self.data
        data_mean = np.mean(data, axis=None)
        data_std = np.sqrt(np.var(data, axis=None))
        print("Start normalizing data to mean=0 / std=1(mean:%g / std:%g)" % (data_mean, data_std))
        self.data = (data - data_mean) / data_std
        print("\tNormalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def get_batch(self, patch_size, stride_length, rotation=True, flip=True):

        print("Start preparing batch....")
        batch_start_time = time.time()

        self.patch_size = patch_size
        self.stride_length = stride_length
        self.rotation = rotation
        self.flip = flip

        self.batch_idx_list = []
        self.batch_data = []

        current_data = self.data
        batch_idx, batch_data = self.image_to_patch(current_data, patch_size, [0, 0])

        self.batch_idx_list.append(batch_idx)
        self.batch_data.append(batch_data)

        if patch_size[0] / stride_length > 1:
            for stride_step in range(1, patch_size[0] // stride_length):
                #current_data = self.data[:,int(stride_length * stride_step):, :]
                batch_idx, batch_data = self.image_to_patch(self.data, patch_size,
                                                           [int(stride_length * stride_step), 0])
                self.batch_idx_list.append(batch_idx)
                self.batch_data.append(batch_data)

        if patch_size[1] / stride_length > 1:
            for stride_step in range(1, patch_size[1] // stride_length):
                #current_data = self.data[:, int(stride_length * stride_step):]
                batch_idx, batch_dataa = self.image_to_patch(self.data, patch_size,
                                                            [0, int(stride_length * stride_step)])
                self.batch_idx_list.append(batch_idx)
                self.batch_data.append(batch_data)

        if patch_size[0] / stride_length > 1 and patch_size[1] / stride_length > 1:
            for row_stride_step in range(1, patch_size[0] // stride_length):
                for col_stride_step in range(1, patch_size[1] // stride_length):
                    #current_data = self.data[:,int(stride_length * row_stride_step):,int(stride_length * col_stride_step):]
                    batch_idx, batch_data = self.image_to_patch(self.data, patch_size,
                                                               [int(stride_length * row_stride_step),
                                                                int(stride_length * col_stride_step)])
                    self.batch_idx_list.append(batch_idx)
                    self.batch_data.append(batch_data)

        self.batch_idx_list = np.concatenate(self.batch_idx_list)
        self.batch_data = np.concatenate(self.batch_data)
        self.batch_data = self.batch_data.astype(float)

        current_seed = np.random.randint(2 ** 16)
        data_datagen = ImageDataGenerator(**self.image_preprocessing_args)
        # data_datagen.fit(current_batch_data, augment=True, seed=current_seed)
        # label_datagen.fit(current_batch_label, augment=True, seed=current_seed)
        current_data_datagen = data_datagen.flow(self.batch_data, batch_size=self.batch_data.shape[0],
                                                 shuffle=False, seed=current_seed)
        self.batch_data = current_data_datagen.next()

        print("\tTotal %s samples." % self.batch_idx_list.shape[0])

        print("\tPreparing batch time: %s seconds." % round((time.time() - batch_start_time), 2))

    def image_to_patch(self, image, patch_size, start_idx):
        new_idx = []
        new_data = []
        for row_step in range((image.shape[self.h_axis-1]-start_idx[0]) // patch_size[0]):
            for col_step in range((image.shape[self.h_axis-1]-start_idx[1]) // patch_size[1]):

                current_row_idx = row_step * patch_size[0] + start_idx[0]
                current_col_idx = col_step * patch_size[1] + start_idx[1]
                if self.channel_axis == 3:
                    current_image = image[current_row_idx:current_row_idx+patch_size[0],current_col_idx:current_col_idx+patch_size[1],:]
                else:
                    current_image = image[:,current_row_idx:current_row_idx+patch_size[0],current_col_idx:current_col_idx+patch_size[1]]

                new_data.append(current_image)
                new_idx.append(np.asarray([current_row_idx, current_col_idx,0,0,0])) # row, col, rotation, ud, lr

        new_data = np.asarray(new_data)
        new_idx = np.asarray(new_idx)

        if self.std_normalization:
            if self.channel_axis==3:
                for channel_idx in range(new_data.shape[2]):
                    current_mean = np.mean(new_data[:,:,channel_idx])
                    current_std = np.std(new_data[:,:,channel_idx])
                    new_data[:,:,channel_idx] = (new_data[:,:,channel_idx]-current_mean)/current_std
            else:
                for channel_idx in range(new_data.shape[0]):
                    current_mean = np.mean(new_data[channel_idx,:,:])
                    current_std = np.std(new_data[channel_idx,:,:])
                    new_data[channel_idx,:,:] = (new_data[channel_idx,:,:]-current_mean)/current_std

        if self.rotation:
            #temp_new_data = np.copy(new_data)
            #temp_new_idx = np.copy(new_idx)
            for rot_idx in range(1,4):
                temp_new_data = np.rot90(new_data, rot_idx, axes=(self.h_axis, self.w_axis))
                new_data = np.concatenate((new_data, temp_new_data))
                temp_new_idx = np.copy(new_idx)
                temp_new_idx[:,2] = -1*rot_idx
                new_idx = np.concatenate((new_idx,temp_new_idx))

        if self.flip:
            new_data_h = np.flip(new_data, axis=self.h_axis)
            new_idx_h = np.copy(new_idx)
            new_idx_h[:, 3] = 1

            new_data_w = np.flip(new_data, axis=self.w_axis)
            new_idx_w = np.copy(new_idx)
            new_idx_w[:, 4] = 1

            new_data_hw = np.flip(new_data_h, axis=self.w_axis)
            new_idx_hw = np.copy(new_idx_h)
            new_idx_hw[:, 4] = 1

            new_idx = np.concatenate((new_idx, new_idx_h, new_idx_w, new_idx_hw))
            new_data = np.concatenate((new_data, new_data_h, new_data_w, new_data_hw))

        return new_idx, new_data # new_idx: row, col, rot, ud, lr | new_data: nchw or nhwc

    def load_data(self, dataset_idx, starting_frame):

        print("Start loading dataset %s...." % dataset_idx)
        load_start_time = time.time()

        # images
        print("\tImages: %s" % self.original_filename[0][dataset_idx])
        starting_frame = starting_frame - 1

        current_data = tifffile.imread(self.original_filename[0][dataset_idx],key=range(starting_frame , starting_frame  + self.frame_number))

        if current_data.ndim == 2:
            current_data = np.expand_dims(current_data, axis=0)

        if self.channel_axis == 3:
            current_data = np.rollaxis(current_data, 0, 3)  # convert to NHWC

        self.data = current_data.astype(float)

        if self.channel_axis == 3:
            self.data_size = current_data.shape[:2]
            self.max_image[dataset_idx] = np.amax(current_data, axis=-1)
        else:
            self.data_size = current_data.shape[1:]
            self.max_image[dataset_idx] = np.amax(current_data, axis=0)

        # roi
        try:
            roi = sio.loadmat(self.original_filename[1][dataset_idx], struct_as_record=False)

            roi_structure_name_list = ['CellList', 'r_out', 'ROIlist', 'R']

            for roi_structure_name in roi_structure_name_list:
                try:
                    roi = roi[roi_structure_name]
                except:
                    pass

            temp_masks = np.zeros(self.data_size)
            for roi_idx in range(0, roi.shape[1]):
                try:
                    temp_masks[np.unravel_index(roi[0, roi_idx].PixelIdxList - 1, self.data_size, order='F')] = 1
                except:
                    temp_masks[np.unravel_index(roi[0, roi_idx].pixel_idx - 1, self.data_size, order='F')] = 1
            self.known_masks[dataset_idx] = temp_masks
        except:
            pass

        self.predict_masks_all[dataset_idx] = []
        self.predict_masks[dataset_idx] = np.full(self.data_size, np.nan)

        print("\tLoading time: %s seconds." % round((time.time() - load_start_time), 2))

    def patch_to_image(self, image_size, idx_list, patch_data):
        # put patch data into an image
        current_image = np.full(image_size, np.nan)
        for current_idx in range(len(idx_list)):
            current_patch_data = patch_data[current_idx]
            current_row_idx = int(idx_list[current_idx][0])
            current_col_idx = int(idx_list[current_idx][1])
            current_image[current_row_idx:current_row_idx+current_patch_data.shape[1],current_col_idx:current_col_idx+current_patch_data.shape[2]] = np.squeeze(current_patch_data)
        return current_image

    def plot_data_batch_bmask(self, dataset_idx=0, area_idx=None, batch_idx=0, erosion_iter=None, ground_truth=1, save_filename=None):
        # area_idx = [[row_start, row_end],[col_start, col_end]]

        try:
            if area_idx==None:
                row_idx = slice(0,self.max_image[dataset_idx].shape[0])
                col_idx = slice(0, self.max_image[dataset_idx].shape[1])
            else:
                row_idx = slice(area_idx[0][0], area_idx[0][1])
                col_idx = slice(area_idx[1][0], area_idx[1][1])

            current_mask = np.zeros((self.binary_masks[dataset_idx].shape[0], self.binary_masks[dataset_idx].shape[1], 3),dtype='float')
            current_binary_mask = self.binary_masks[dataset_idx]
            if erosion_iter is not None and erosion_iter != 0:
                current_binary_mask = ndimage.binary_erosion(current_binary_mask, iterations=erosion_iter).astype(current_binary_mask.dtype)
            current_mask[:, :, 0] = current_binary_mask
            if ground_truth==1:
                try:
                    current_mask[:, :, 2] = self.known_masks[dataset_idx]
                except:
                    print("No ground truth.")

            fig = plt.figure(figsize=(18, 18))

            plt.title("%s" % self.original_filename[0][dataset_idx])

            plt.subplot(131)
            plt.imshow(self.max_image[dataset_idx][row_idx,col_idx])
            plt.subplot(132)
            plt.imshow(self.batch_images[dataset_idx][batch_idx][row_idx,col_idx])
            plt.subplot(133)
            plt.imshow(current_mask[row_idx,col_idx])
            if save_filename is not None:
                fig.savefig(save_filename)
                print("Saved the figure as %s" % save_filename)
        except AttributeError:
            print("No binary mask for dataset %s." % dataset_idx)

    def plot_data_batch_pmask(self, dataset_idx=0, area_idx=None, batch_idx=0, ground_truth=1, save_filename=None):
        # area_idx = [[row_start, row_end],[col_start, col_end]]

        try:
            if area_idx==None:
                row_idx = slice(0,self.max_image[dataset_idx].shape[0])
                col_idx = slice(0, self.max_image[dataset_idx].shape[1])
            else:
                row_idx = slice(area_idx[0][0], area_idx[0][1])
                col_idx = slice(area_idx[1][0], area_idx[1][1])

            current_mask = np.zeros((self.predict_masks[dataset_idx].shape[0], self.predict_masks[dataset_idx].shape[1], 3),dtype='float')
            current_mask[:, :, 0] = self.predict_masks[dataset_idx]
            if ground_truth==1:
                try:
                    current_mask[:, :, 2] = self.known_masks[dataset_idx]
                except:
                    print("No ground truth.")

            fig = plt.figure(figsize=(18, 18))

            plt.title("%s" % self.original_filename[0][dataset_idx])

            plt.subplot(131)
            plt.imshow(self.max_image[dataset_idx][row_idx,col_idx])
            plt.subplot(132)
            plt.imshow(self.batch_images[dataset_idx][batch_idx][row_idx,col_idx])
            plt.subplot(133)
            plt.imshow(current_mask[row_idx,col_idx])
            if save_filename is not None:
                fig.savefig(save_filename)
                print("Saved the figure as %s" % save_filename)
        except AttributeError:
            print("No predicted mask for dataset %s." % dataset_idx)

    def plot_data_bmask_outline(self, dataset_idx=0, area_idx=None, erosion_iter=None, ground_truth=1, save_filename=None, **kwargs):
        # area_idx = [[row_start, row_end],[col_start, col_end]]
        # **kwargs: linewidths=2, colors='r'

        try:
            if area_idx==None:
                row_idx = slice(0,self.max_image[dataset_idx].shape[0])
                col_idx = slice(0, self.max_image[dataset_idx].shape[1])
            else:
                row_idx = slice(area_idx[0][0], area_idx[0][1])
                col_idx = slice(area_idx[1][0], area_idx[1][1])

            current_mask = self.binary_masks[dataset_idx]
            if erosion_iter is not None and erosion_iter != 0:
                current_mask = ndimage.binary_erosion(current_mask, iterations=erosion_iter).astype(current_mask.dtype)

            fig = plt.figure(figsize=(18, 18))

            plt.title("%s" % self.original_filename[0][dataset_idx])

            plt.subplot(131)
            plt.imshow(self.max_image[dataset_idx][row_idx, col_idx])
            plt.subplot(132)
            plt.imshow(self.max_image[dataset_idx][row_idx, col_idx])
            plt.contour(current_mask[row_idx, col_idx], **kwargs)
            if ground_truth==1:
                try:
                    plt.subplot(133)
                    plt.imshow(self.max_image[dataset_idx][row_idx, col_idx])
                    plt.contour(self.known_masks[dataset_idx][row_idx, col_idx], **kwargs)
                except:
                    print("No ground truth.")
            if save_filename is not None:
                fig.savefig(save_filename)
                print("Saved the figure as %s" % save_filename)
        except AttributeError:
            print("No binary mask for dataset %s." % dataset_idx)

    def plot_data_predict_masks(self, dataset_idx=0, area_idx=None, ground_truth=1, save_filename=None):
        # area_idx = [[row_start, row_end],[col_start, col_end]]
        # **kwargs: linewidths=2, colors='r'

        try:
            if area_idx==None:
                row_idx = slice(0,self.max_image[dataset_idx].shape[0])
                col_idx = slice(0, self.max_image[dataset_idx].shape[1])
            else:
                row_idx = slice(area_idx[0][0], area_idx[0][1])
                col_idx = slice(area_idx[1][0], area_idx[1][1])

            current_predict_result = self.predict_masks[dataset_idx]

            fig = plt.figure(figsize=(2 * 10, 1 * 10))
            gs = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.0)
            # fig.suptitle("%s" % self.original_filename[0][dataset_idx])

            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(self.max_image[dataset_idx][row_idx, col_idx])
            if ground_truth == 1:
                try:
                    ax.contour(self.known_masks[dataset_idx][row_idx, col_idx], linewidths=0.1, colors='r')
                except:
                    print("No ground truth.")

            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(current_predict_result[row_idx, col_idx])
            if ground_truth == 1:
                try:
                    ax.contour(self.known_masks[dataset_idx][row_idx, col_idx], linewidths=0.1, colors='r')
                except:
                    print("No ground truth.")
            plt.show()

            if save_filename is not None:
                fig.savefig(save_filename)
                print("Saved the figure as %s" % save_filename)
        except AttributeError:
            print("No binary mask for dataset %s." % dataset_idx)

    def plot_data_predict_masks_raw(self, dataset_idx=0, area_idx=None, ground_truth=1, save_filename=None):
        # area_idx = [[row_start, row_end],[col_start, col_end]]
        # **kwargs: linewidths=2, colors='r'

        #try:
            if area_idx==None:
                row_idx = slice(0,self.max_image[dataset_idx].shape[0])
                col_idx = slice(0, self.max_image[dataset_idx].shape[1])
            else:
                row_idx = slice(area_idx[0][0], area_idx[0][1])
                col_idx = slice(area_idx[1][0], area_idx[1][1])

            current_predict_result = self.predict_masks_all[dataset_idx]

            fig = plt.figure(figsize=(self.catagory_number * 10, 1 * 10))
            gs = gridspec.GridSpec(1, self.catagory_number+1, wspace=0.1, hspace=0.0)
            #fig.suptitle("%s" % self.original_filename[0][dataset_idx])

            ax = fig.add_subplot(gs[0, 0])
            img = ax.imshow(self.max_image[dataset_idx][row_idx, col_idx])
            divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            cax = divider.append_axes("bottom", size="5%", pad=0.1)
            fig.colorbar(img,cax=cax,orientation='horizontal')
            vmax = np.max(self.predict_masks_all[dataset_idx])
            if ground_truth == 1:
                try:
                    ax.contour(self.known_masks[dataset_idx][row_idx, col_idx], linewidths=0.1, colors='r')
                except:
                    print("No ground truth.")
            for catagory_idx in range(current_predict_result.shape[2]):
                ax = fig.add_subplot(gs[0, 1+catagory_idx])
                img = ax.imshow(self.predict_masks_all[dataset_idx][row_idx, col_idx,catagory_idx]/vmax)
                img.set_clim(vmin=0, vmax=1)
                divider = make_axes_locatable(ax)
                #cax = divider.append_axes("right", size="5%", pad=0.05)
                cax = divider.append_axes("bottom", size="5%", pad=0.1)
                fig.colorbar(img,cax=cax,orientation='horizontal')
                if ground_truth == 1:
                    try:
                        ax.contour(self.known_masks[dataset_idx][row_idx, col_idx], linewidths=0.1, colors='r')
                    except:
                        print("No ground truth.")
                ax.set_title('Category %s' % catagory_idx)
            plt.show()

            if save_filename is not None:
                fig.savefig(save_filename)
                print("Saved the figure as %s" % save_filename)

        #except AttributeError:
            #print("No binary mask for dataset %s." % dataset_idx)

    def plot_data_roi_outline(self, dataset_idx=0, area_idx=None, ground_truth=1, save_filename=None, **kwargs):
        # area_idx = [[row_start, row_end],[col_start, col_end]]
        # **kwargs: linewidths=2, colors='r'

        try:
            if area_idx==None:
                row_idx = slice(0,self.max_image[dataset_idx].shape[0])
                col_idx = slice(0, self.max_image[dataset_idx].shape[1])
            else:
                row_idx = slice(area_idx[0][0], area_idx[0][1])
                col_idx = slice(area_idx[1][0], area_idx[1][1])

            current_roi_mask = self.roi_masks[dataset_idx]

            fig = plt.figure(figsize=(3 * 10, 1 * 10))
            gs = gridspec.GridSpec(1, 3, wspace=0.1, hspace=0.0)
            # fig.suptitle("%s" % self.original_filename[0][dataset_idx])

            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(self.max_image[dataset_idx][row_idx, col_idx])

            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(self.max_image[dataset_idx][row_idx, col_idx])
            ax.contour(current_roi_mask[row_idx, col_idx], list(range(1, current_roi_mask.max() + 1)), **kwargs)

            if ground_truth == 1:
                try:
                    ax = fig.add_subplot(gs[0, 2])
                    ax.imshow(self.max_image[dataset_idx][row_idx, col_idx])
                    plt.contour(self.known_masks[dataset_idx][row_idx, col_idx], **kwargs)
                except:
                    print("No ground truth.")
            plt.show()

            if save_filename is not None:
                fig.savefig(save_filename)
                print("Saved the figure as %s" % save_filename)
        except AttributeError:
            print("No binary mask for dataset %s." % dataset_idx)

    def pre_process(self):
        # self.batch_highpass_flt()
        self.batch_std_eq()

class roi_unet_train:
    def __init__(self, filename_pairs, frame_number, extend_pixel_number):

        if len(filename_pairs[0]) != len(filename_pairs[1]):
            raise Exception("The numbers of tiff files and ROI files do not match.")
        else:
            self.original_filename = filename_pairs
            self.frame_number = frame_number
            self.extend_pixel_number = extend_pixel_number
            self.label_type_number = 2
            self.known_masks = {}
            self.file_frame_number = {}
            print("Total %s datasets." % len(self.original_filename))

    def batch(self, label_type_number, random_number, current_batch_with_roi_min_ratio, retain_ratio):

        print("Start preparing batch....")
        batch_start_time = time.time()

        if hasattr(self, 'batch_data'):
            new_label_type_number = np.array(label_type_number) * (1 - retain_ratio)
            old_batch_number = int(np.sum(np.array(label_type_number)) - np.sum(new_label_type_number))

            old_batch_idx = list(range(self.batch_data.shape[0]))
            np.random.shuffle(old_batch_idx)
            old_batch_idx = old_batch_idx[:old_batch_number]

            old_batch_data = self.batch_data[old_batch_idx, :]
            old_batch_label = self.batch_label[old_batch_idx, :]

            [new_batch_data, new_batch_label] = self.new_batch(new_label_type_number, random_number)

            batch_data = np.concatenate((old_batch_data, new_batch_data), axis=0)
            batch_label = np.concatenate((old_batch_label, new_batch_label), axis=0)

            batch_data_idx = list(range(batch_data.shape[0]))
            np.random.shuffle(batch_data_idx)
            batch_data = batch_data[batch_data_idx, :]
            batch_label = batch_label[batch_data_idx, :]

            print("\tKeep %d data." % (old_batch_data.shape[0]))
            print("\tGenerated %d data." % (new_batch_data.shape[0]))

        else:
            label_type_number = np.array(label_type_number)
            [batch_data, batch_label] = self.new_batch(label_type_number, random_number)
            print("\tGenerated %d data." % (batch_data.shape[0]))

        # plt.figure(figsize=(12,12))
        # plt.imshow(self.batch_map)

        self.batch_data = np.asfarray(batch_data)
        self.batch_label = batch_label

        print("\tPreparing batch time: %s seconds." % round((time.time() - batch_start_time), 2))

    def batch_0_1(self):
        batch_data_idx_list = range(0, self.batch_data.shape[0])
        for batch_data_idx in tqdm(batch_data_idx_list):
            data = np.self.batch_data[batch_data_idx]
            data_range = np.amax(data, axis=None) - np.amin(data, axis=None)
            self.batch_data[batch_data_idx] = (data - np.amin(data, axis=None)) / data_range

    def batch_add_gradient(self):
        gradient_start_time = time.time()
        batch_data_idx_list = range(0, self.batch_data.shape[0])
        batch_gradient_x = []
        batch_gradient_y = []
        for batch_data_idx in tqdm(batch_data_idx_list):
            data = self.batch_data[batch_data_idx]
            gradient_x = np.empty(data.shape)
            gradient_y = np.empty(data.shape)
            for frame_idx in range(0, data.shape[-1]):
                current_frame = data[:, :, frame_idx]
                gradient_x[:, :, frame_idx], gradient_y[:, :, frame_idx] = np.gradient(current_frame)
            batch_gradient_x.append([gradient_x])
            batch_gradient_y.append([gradient_y])

        self.batch_gradient_x = np.asfarray(np.concatenate(batch_gradient_x, axis=0))
        self.batch_gradient_y = np.asfarray(np.concatenate(batch_gradient_y, axis=0))
        print("\tCalculating gradient time: %s seconds." % round((time.time() - gradient_start_time), 2))

    def batch_all(self, label_type_number, current_batch_with_roi_min_ratio, erosion_iter=None, force_separate=True, is_validation=False):

        if force_separate is True:
            print("force_separate is ON.")

        if erosion_iter is not None and erosion_iter != 0:
            print("erosion_iter is ON for %s times." % erosion_iter)

        preprocessing_args = dict(
            # featurewise_center=True,
            # samplewise_center=True,
            # featurewise_std_normalization=True,
            # samplewise_std_normalization=True,
            # zca_whitening=False,
            # zca_epsilon=1e-06
            rotation_range=90,
            # width_shift_range=0.1,
            # height_shift_range=0.1,
            # brightness_range=None,
            # shear_range=0.0,
            zoom_range=0.2,
            fill_mode='reflect', # "constant", "nearest", "reflect" or "wrap"
            # cval=0.0,
            horizontal_flip=True,
            vertical_flip=True,
            # rescale=None,
        )

        print("Start preparing batch....")
        batch_start_time = time.time()

        extend_pixel_number = self.extend_pixel_number

        self.all_data_idx = []
        self.all_pixel_idx = []

        label_type_number = np.sum(label_type_number)

        current_batch_data = np.empty((label_type_number, extend_pixel_number, extend_pixel_number, self.frame_number))
        current_batch_label = np.empty((label_type_number, extend_pixel_number, extend_pixel_number, 2))

        current_batch_with_roi_min_number = current_batch_with_roi_min_ratio * np.sum(label_type_number)

        for batch_data_count in tqdm(range(np.sum(label_type_number))):

            try:
                current_data_idx = self.all_data_idx.pop()
            except IndexError:
                all_data_idx = list(range(0, len(self.original_filename[0])))
                np.random.shuffle(all_data_idx)
                self.all_data_idx = collections.deque(all_data_idx)
                current_data_idx = self.all_data_idx.pop()

            # images
            temp_images = []
            image_file = PIL.Image.open(self.original_filename[0][current_data_idx])
            try:
                image_frame_number = self.file_frame_number[self.original_filename[0][current_data_idx]]
            except KeyError:
                self.file_frame_number[self.original_filename[0][current_data_idx]] = image_file.n_frames
                image_frame_number = self.file_frame_number[self.original_filename[0][current_data_idx]]

            current_frame_idx = np.random.randint(0, np.amin([1023, image_frame_number]))

            image_file.seek(current_frame_idx)
            temp_images.append([np.array(image_file)])
            temp_images = np.concatenate(temp_images, axis=0)
            temp_images = np.rollaxis(temp_images, 0, 3)
            self.data_size = temp_images.shape[:2]
            self.data = temp_images

            # print('%s: %s' % (batch_data_count, self.original_filename[0][current_data_idx]))

            # roi
            try:
                self.mask = self.known_masks[self.original_filename[1][current_data_idx]]
            except KeyError:
                roi = sio.loadmat(self.original_filename[1][current_data_idx], struct_as_record=False)
                try:
                    roi = roi['CellList']
                except:
                    roi = roi['r_out']

                temp_masks = np.zeros(self.data_size)
                for roi_idx in range(0, roi.shape[1]):
                    single_roi_mask = np.zeros(self.data_size)
                    try:
                        single_roi_mask[np.unravel_index(roi[0, roi_idx].PixelIdxList - 1, self.data_size, order='F')] = 1
                    except:
                        single_roi_mask[np.unravel_index(roi[0, roi_idx].pixel_idx - 1, self.data_size, order='F')] = 1

                    if erosion_iter is not None and erosion_iter != 0:
                        single_roi_mask = ndimage.binary_erosion(single_roi_mask,iterations=erosion_iter).astype(single_roi_mask.dtype)
                    if force_separate is True:
                        dilated_single_roi_mask = ndimage.binary_dilation(single_roi_mask)
                        dilated_temp_masks = ndimage.binary_dilation(temp_masks)
                        remove_area = np.logical_and(dilated_single_roi_mask, dilated_temp_masks)
                        temp_masks[single_roi_mask == 1] = 1
                        temp_masks[remove_area] = 0
                    else:
                        temp_masks[single_roi_mask==1] = 1
                self.known_masks[self.original_filename[1][current_data_idx]] = temp_masks
                self.mask = self.known_masks[self.original_filename[1][current_data_idx]]

            while True:
                try:
                    current_pixel_idx = self.all_pixel_idx.pop()
                except IndexError:
                    all_pixel_idx = list(
                        range(0, (self.data_size[0] - extend_pixel_number) * (self.data_size[1] - extend_pixel_number)))
                    np.random.shuffle(all_pixel_idx)
                    self.all_pixel_idx = collections.deque(all_pixel_idx)
                    current_pixel_idx = self.all_pixel_idx.pop()

                current_pixel_idx_rc = np.unravel_index(current_pixel_idx, (
                self.data_size[0] - extend_pixel_number, self.data_size[1] - extend_pixel_number), order='F')
                row_idx = current_pixel_idx_rc[0]
                col_idx = current_pixel_idx_rc[1]

                current_mask = self.mask[row_idx:row_idx + extend_pixel_number, col_idx:col_idx + extend_pixel_number]
                # print('%s-%s: %g' % (row_idx,col_idx,np.sum(current_mask)))
                if batch_data_count <= current_batch_with_roi_min_number:
                    if np.sum(current_mask) > 0:
                        break
                else:
                    break

            frame_idx = 0

            current_image = self.data[row_idx:row_idx + extend_pixel_number, col_idx:col_idx + extend_pixel_number,
                            frame_idx:frame_idx + self.frame_number]
            current_mask = self.mask[row_idx:row_idx + extend_pixel_number, col_idx:col_idx + extend_pixel_number]

            current_null_mask = 1 - current_mask
            current_label = np.dstack([current_null_mask, current_mask])

            current_batch_data[batch_data_count, :, :, :] = current_image
            current_batch_label[batch_data_count, :, :, :] = current_label

        if is_validation:
            self.batch_data = current_batch_data
            self.batch_label = current_batch_label
        else:
            print("Start preprocessing....")
            preprocessing_start_time = time.time()
            seed = np.random.randint(label_type_number)
            data_datagen = ImageDataGenerator(**preprocessing_args)
            label_datagen = ImageDataGenerator(**preprocessing_args)
            data_datagen.fit(current_batch_data, augment=True, seed=seed)
            label_datagen.fit(current_batch_label, augment=True, seed=seed)
            current_data_datagen = data_datagen.flow(current_batch_data, batch_size=label_type_number, shuffle=False, seed=seed)
            current_label_datagen = label_datagen.flow(current_batch_label, batch_size=label_type_number, shuffle=False, seed=seed)
            self.batch_data = current_data_datagen.next()
            self.batch_label = current_label_datagen.next()
            print("\tPreprocessing time: %s seconds." % round((time.time() - preprocessing_start_time), 2))

        print("\tPreparing batch time: %s seconds." % round((time.time() - batch_start_time), 2))

    def batch_filter(self, frequency_cutoff):

        data = self.batch_data
        frequency_cutoff = float(frequency_cutoff) / 20
        filter_b, filter_a = signal.butter(8, frequency_cutoff)  # frame rate is 20 Hz
        self.batch_data = signal.filtfilt(filter_b, filter_a, data, axis=(data.ndim - 1))

    def batch_gradient_std(self):

        normalization_start_time = time.time()
        batch_data_idx_list = range(0, self.batch_gradient_x.shape[0])
        for batch_data_idx in tqdm(batch_data_idx_list):
            data = self.batch_gradient_x[batch_data_idx]
            data_mean = np.mean(data, axis=None)
            data_std = np.sqrt(np.var(data, axis=None))
            self.batch_gradient_x[batch_data_idx] = (data - data_mean) / data_std

        batch_data_idx_list = range(0, self.batch_gradient_y.shape[0])
        for batch_data_idx in tqdm(batch_data_idx_list):
            data = self.batch_gradient_y[batch_data_idx]
            data_mean = np.mean(data, axis=None)
            data_std = np.sqrt(np.var(data, axis=None))
            self.batch_gradient_y[batch_data_idx] = (data - data_mean) / data_std
        print("\tNormalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def batch_highpass_flt(self, sigma=50):

        print("Start highpass filtering batch....")
        normalization_start_time = time.time()

        batch_data_idx_list = range(0, self.batch_data.shape[0])
        for batch_data_idx in tqdm(batch_data_idx_list):
            data = self.batch_data[batch_data_idx]
            lowpass_data = ndimage.gaussian_filter(data, sigma)
            self.batch_data[batch_data_idx] = data-lowpass_data
        print("\tHighpass filtering time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def batch_hist_eq(self):

        print("Start histogram equalizing batch....")
        normalization_start_time = time.time()

        batch_data_idx_list = range(0, self.batch_data.shape[0])
        for batch_data_idx in tqdm(batch_data_idx_list):
            data = self.batch_data[batch_data_idx]

            dist, bins = np.histogram(data.flatten(), 2 ** 16, density=True)
            cum_dist = dist.cumsum()
            cum_dist = 2 ** 16 * cum_dist / [cum_dist[-1]]
            new_data = np.interp(data.flatten(), bins[:-1], cum_dist)
            self.batch_data[batch_data_idx] = new_data.reshape(data.shape)
        print("\tHistogram equalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def batch_label_ratio(self):
        batch_label_ratio = np.mean(self.batch_label)
        print("Batch Label Ratio: %s." % round(batch_label_ratio, 4))

    def batch_noise(self, noise_factor=0.01):
        print("Start adding noise batch....")
        noise_start_time = time.time()

        batch_data_idx_list = range(0, self.batch_data.shape[0])
        for batch_data_idx in tqdm(batch_data_idx_list):
            data = self.batch_data[batch_data_idx]
            data = data+noise_factor*data.std()*(np.random.random(data.shape)*2-1)
            data[data<0] = 0
            self.batch_data[batch_data_idx] = data
        print("\tAdding noise time: %s seconds." % round((time.time() - noise_start_time), 2))

    def batch_std_eq(self):

        print("Start normalizing batch....")
        normalization_start_time = time.time()

        # pool = mp.Pool(processes=6)
        # old_batch_data = self.batch_data
        # old_batch_data.tolist()
        # new_batch_data = pool.map(self.para_batch_std, old_batch_data)
        # self.batch_data = np.asarray(new_batch_data)

        batch_data_idx_list = range(0, self.batch_data.shape[0])
        for batch_data_idx in tqdm(batch_data_idx_list):
            data = self.batch_data[batch_data_idx]
            data_mean = np.mean(data, axis=None)
            data_std = np.sqrt(np.var(data, axis=None))
            self.batch_data[batch_data_idx] = (data - data_mean) / data_std
        print("\tNormalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def batch_zscore(self):
        batch_data_idx_list = range(0, self.batch_data.shape[0])
        for batch_data_idx in tqdm(batch_data_idx_list):
            data = self.batch_data[batch_data_idx]
            self.batch_data[batch_data_idx] = stats.zscore(data, axis=None)

    def data_0_1(self):

        print("Start normalizing data to 0-1 (range: %g-%g)" % (
        np.amin(self.data, axis=None), np.amax(self.data, axis=None)))
        normalization_start_time = time.time()
        data = self.data
        data_range = np.amax(data, axis=None) - np.amin(data, axis=None)
        self.data = (data - np.amin(data, axis=None)) / data_range
        print("\tNormalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def data_add_corr(self):

        print("Start calculating correlation coefficient....")
        corr_start_time = time.time()
        data = self.data
        corr_image = np.zeros([data.shape[0], data.shape[1], 1])
        for row_idx in range(self.extend_pixel_number, self.extend_pixel_number + self.data_size[0] - 1):
            for col_idx in range(self.extend_pixel_number, self.extend_pixel_number + self.data_size[1] - 1):
                current_trace = data[row_idx][col_idx]
                target_trace = data[row_idx][col_idx + 1]

                m_current_trace = current_trace - current_trace.mean()
                m_target_trace = target_trace - target_trace.mean()

                corr_image[row_idx][col_idx][0] = np.dot(m_current_trace, m_target_trace.T) / np.sqrt(
                    np.dot((m_current_trace ** 2).sum(), (m_target_trace ** 2).sum()))

        # self.data = np.concatenate((corr_image,data),axis=2)
        self.data[:, :, 0] = corr_image[:, :, 0]

        print("\tCalculating time: %s seconds." % round((time.time() - corr_start_time), 2))

    def data_filter(self, frequency_cutoff):

        print("Start low-pass filtering data at %g Hz...." % frequency_cutoff)
        filter_start_time = time.time()
        filtered_data = self.data

        frequency_cutoff = float(frequency_cutoff) / 20
        filter_b, filter_a = signal.butter(8, frequency_cutoff)  # frame rate is 20 Hz
        self.data = signal.filtfilt(filter_b, filter_a, filtered_data, axis=(filtered_data.ndim - 1))
        print("\tFilteringing time: %s seconds." % round((time.time() - filter_start_time), 2))

    def data_frame_hist_eq(self):

        print("Start histogram equalizing frame....")
        normalization_start_time = time.time()

        frame_data_idx_list = range(0, self.data.shape[2])
        for frame_data_idx in frame_data_idx_list:
            data = self.data[:, :, frame_data_idx]

            dist, bins = np.histogram(data.flatten(), 2 ** 16, density=True)
            cum_dist = dist.cumsum()
            cum_dist = 2 ** 16 * cum_dist / [cum_dist[-1]]
            new_data = np.interp(data.flatten(), bins[:-1], cum_dist)
            self.data[:, :, frame_data_idx] = new_data.reshape(data.shape)
        print("\tHistogram equalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def data_std_eq(self):

        normalization_start_time = time.time()
        data = self.data
        data_mean = np.mean(data, axis=None)
        data_std = np.sqrt(np.var(data, axis=None))
        print("Start normalizing data to mean=0 / std=1(mean:%g / std:%g)" % (data_mean, data_std))
        self.data = (data - data_mean) / data_std
        print("\tNormalizing time: %s seconds." % round((time.time() - normalization_start_time), 2))

    def load_data(self):
        extend_pixel_number = self.extend_pixel_number

        print("Start loading data....")
        load_start_time = time.time()

        self.all_data_idx = []

        try:
            current_data_idx = self.all_data_idx.pop()
        except IndexError:
            all_data_idx = list(range(0, len(self.original_filename[0])))
            np.random.shuffle(all_data_idx)
            self.all_data_idx = collections.deque(all_data_idx)
            current_data_idx = self.all_data_idx.pop()

        print("\tImages: %s" % self.original_filename[0][current_data_idx])
        print("\tROIS: %s" % self.original_filename[1][current_data_idx])

        # images
        temp_images = []
        image_file = PIL.Image.open(self.original_filename[0][current_data_idx])
        image_frame_number = image_file.n_frames

        for current_frame_idx in tqdm(range(0, np.amin([1023, image_frame_number - 1]))):
            image_file.seek(current_frame_idx)
            temp_images.append([np.array(image_file)])

        temp_images = np.concatenate(temp_images, axis=0)
        temp_images = np.rollaxis(temp_images, 0, 3)
        self.data_size = temp_images.shape[:2]
        self.data = temp_images

        # roi
        roi = sio.loadmat(self.original_filename[1][current_data_idx], struct_as_record=False)
        roi = roi['CellList']

        # self.data_size = roi[0,0].perimeter.shape
        temp_masks = np.zeros(self.data_size)
        for roi_idx in range(0, roi.shape[1]):
            temp_masks[np.unravel_index(roi[0, roi_idx].PixelIdxList - 1, self.data_size, order='F')] = 1
        self.mask = temp_masks

        print("\tLoading time: %s seconds." % round((time.time() - load_start_time), 2))

    def load_data_frame(self):
        extend_pixel_number = self.extend_pixel_number

        print("Start loading data....")
        load_start_time = time.time()

        try:
            current_frame_idx = self.all_frame_idx.pop(0)
        except:
            try:
                current_data_idx = self.all_data_idx.pop()
            except:
                all_data_idx = list(range(0, len(self.original_filename[0])))
                np.random.shuffle(all_data_idx)
                self.all_data_idx = collections.deque(all_data_idx)
                current_data_idx = self.all_data_idx.pop()

                image_file = PIL.Image.open(self.original_filename[0][current_data_idx])
                image_frame_number = image_file.n_frames
                image_frame_number = np.amin([image_frame_number, 1023])

                all_frame_idx = list(range(0, image_frame_number))
                np.random.shuffle(all_frame_idx)
                self.all_frame_idx = all_frame_idx
                current_frame_idx = self.all_frame_idx.pop(0)

        print("\tImages: %s" % self.original_filename[0][current_data_idx])
        print("\tFrame: %s" % current_frame_idx)
        print("\tROIS: %s" % self.original_filename[1][current_data_idx])

        # images
        temp_images = []
        image_file = PIL.Image.open(self.original_filename[0][current_data_idx])
        image_frame_number = image_file.n_frames

        image_file.seek(current_frame_idx)
        temp_images.append([np.array(image_file)])

        temp_images = np.concatenate(temp_images, axis=0)
        temp_images = np.rollaxis(temp_images, 0, 3)
        self.data_size = temp_images.shape[:2]
        self.data = temp_images

        # roi
        roi = sio.loadmat(self.original_filename[1][current_data_idx], struct_as_record=False)
        roi = roi['CellList']

        # self.data_size = roi[0,0].perimeter.shape
        temp_masks = np.zeros(self.data_size)
        for roi_idx in range(0, roi.shape[1]):
            temp_masks[np.unravel_index(roi[0, roi_idx].PixelIdxList - 1, self.data_size, order='F')] = 1
        self.mask = temp_masks

        print("\tLoading time: %s seconds." % round((time.time() - load_start_time), 2))

    def new_batch(self, label_type_number, random_number, current_batch_with_roi_min_ratio):

        # print "Start preparing batch...."
        batch_start_time = time.time()

        extend_pixel_number = self.extend_pixel_number
        self.batch_map = np.zeros(self.data_size)

        if random_number > 0:
            for label_type_idx in range(0, len(label_type_number)):
                label_type_number[label_type_idx] = label_type_number[label_type_idx] + np.random.randint(0,
                                                                                                          random_number)

        new_data = []
        new_label = []

        current_batch_with_roi = 0
        current_batch_with_roi_min_number = current_batch_with_roi_min_ratio * np.sum(label_type_number)

        self.all_pixel_idx = []

        for _ in tqdm(range(np.sum(label_type_number))):

            while True:
                try:
                    current_pixel_idx = self.all_pixel_idx.pop()
                except IndexError:
                    all_pixel_idx = list(
                        range(0, (self.data_size[0] - extend_pixel_number) * (self.data_size[1] - extend_pixel_number)))
                    np.random.shuffle(all_pixel_idx)
                    self.all_pixel_idx = collections.deque(all_pixel_idx)
                    current_pixel_idx = self.all_pixel_idx.pop()

                current_pixel_idx_rc = np.unravel_index(current_pixel_idx, (
                self.data_size[0] - extend_pixel_number, self.data_size[1] - extend_pixel_number), order='F')
                row_idx = current_pixel_idx_rc[0]
                col_idx = current_pixel_idx_rc[1]

                current_mask = self.mask[row_idx:row_idx + extend_pixel_number, col_idx:col_idx + extend_pixel_number]
                # print('%s-%s: %g' % (row_idx,col_idx,np.sum(current_mask)))
                if current_batch_with_roi <= current_batch_with_roi_min_number:
                    if np.sum(current_mask) > 0:
                        current_batch_with_roi = current_batch_with_roi + 1
                        break
                else:
                    break

            if self.data.shape[2] == self.frame_number:
                frame_idx = 0
            else:
                frame_idx = np.random.randint(0, self.data.shape[2] - self.frame_number)

            current_image = self.data[row_idx:row_idx + extend_pixel_number, col_idx:col_idx + extend_pixel_number,
                            frame_idx:frame_idx + self.frame_number]
            current_mask = self.mask[row_idx:row_idx + extend_pixel_number, col_idx:col_idx + extend_pixel_number]

            self.batch_map[row_idx:row_idx + extend_pixel_number, col_idx:col_idx + extend_pixel_number] = np.amax(
                current_image, axis=2)

            rot_idx = np.random.randint(1, 5)
            current_image = np.rot90(current_image, rot_idx)
            current_mask = np.rot90(current_mask, rot_idx)

            if np.random.randint(0, 2) == 1:
                current_image = np.fliplr(current_image)
                current_mask = np.fliplr(current_mask)

            if np.random.randint(0, 2) == 1:
                current_image = np.flipud(current_image)
                current_mask = np.flipud(current_mask)

            current_null_mask = 1 - current_mask
            # current_label = np.concatenate([current_null_mask,current_mask], axis=2)
            current_label = np.dstack([current_null_mask, current_mask])
            # current_label = current_mask

            new_data.append([current_image])
            new_label.append([current_label])

        new_batch_data = np.concatenate(new_data, axis=0)
        new_batch_label = np.concatenate(new_label, axis=0)

        return [new_batch_data, new_batch_label]

    def plot_batch_data(self, batch_data_idx=0):
        current_batch_data = np.squeeze(self.batch_data[batch_data_idx])
        current_batch_label = np.argmax(np.squeeze(self.batch_label[batch_data_idx]), axis=-1)

        plt.figure(figsize=(18, 18))
        plt.subplot(1, 2, 1)
        plt.imshow(current_batch_data)
        plt.subplot(1, 2, 2)
        plt.imshow(current_batch_label)
        plt.show()

    def pre_process(self):
        # self.batch_noise()
        # self.batch_highpass_flt()
        self.batch_std_eq()



class train_data_generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_folder_list, patch_size, frame_number, catagory_number=2, batch_size=64, batch_number=None, erosion_iter=False, force_separate=True, shuffle=True, lh_threshold=0, lh_ratio=1, non_overlap=True, std_normalization=True, roi_folder="", roi_tag="_roi", channels_order='channels_first', image_augmentation=None):
        'Initialization'
        self.dataset_folder_list = dataset_folder_list
        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size]
        else:
            self.patch_size = patch_size
        self.frame_number = frame_number
        self.catagory_number = catagory_number
        self.known_masks = {}
        self.known_masks_ratio = {}
        #self.file_frame_number = {}
        #self.file_image_size = {}
        self.batch_size = batch_size
        self.batch_number = batch_number
        self.shuffle = shuffle
        self.lh_threshold = lh_threshold
        self.lh_ratio = lh_ratio
        self.non_overlap = non_overlap
        self.std_normalization = std_normalization
        self.channels_order = channels_order
        if self.channels_order=='channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3
        self.image_augmentation = image_augmentation
        self.data_info = []

        if self.image_augmentation=='keras':
            print("use Keras for image augmentation")
            self.image_preprocessing_args = dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-06,
                rotation_range=360,
                width_shift_range=0.2,
                height_shift_range=0.2,
                # brightness_range=[0.5, 1.5],
                shear_range=0.1,
                zoom_range=0,
                fill_mode='reflect',  # "constant", "nearest", "reflect" or "wrap"
                # cval=0.0,
                horizontal_flip=True,
                vertical_flip=True,
                # rescale=None,
                data_format=self.channels_order
            )

            self.label_preprocessing_args = self.image_preprocessing_args
            self.label_preprocessing_args['featurewise_center'] = False
            self.label_preprocessing_args['samplewise_center'] = False
            self.label_preprocessing_args['featurewise_std_normalization'] = False
            self.label_preprocessing_args['samplewise_std_normalization'] = False
            self.label_preprocessing_args['zca_whitening'] = False
        elif self.image_augmentation=='imgaug':
            print("use imgaug for image augmentation")
            self.image_preprocessing_args = [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                #iaa.CropAndPad(percent=(-0.25, 0.25),pad_mode=["edge"]),
                iaa.Multiply((0.1, 1.1)),
                #iaa.MultiplyElementwise((0.5, 1.5)),
                iaa.GaussianBlur(sigma=(0.0, 5.0)),
                #iaa.AdditiveGaussianNoise(scale=(0, 0.001*2**16)),
                #iaa.ContrastNormalization((0.5, 1.5)),
                #iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                iaa.Dropout(p=(0, 0.1)),
                iaa.Affine(rotate=(-90, 90), mode=['symmetric','reflect','wrap']),
                #iaa.Affine(shear=(-45, 45), mode=['symmetric','reflect','wrap']),
            ]
        else:
            print("No image augmentation")

        # Go through all folders
        for dataset_folder in dataset_folder_list:
            dataset_folder = os.path.normpath(dataset_folder)

            # Get image file names from folder
            image_filename_list = next(os.walk(dataset_folder))[2]

            # Add images
            for current_image_filename in tqdm(image_filename_list, desc='Files'):
                [current_image_filename_base, current_image_filename_ext] = os.path.splitext(current_image_filename)
                if current_image_filename_ext in [".tif"]:
                    current_image_path = os.path.join(dataset_folder, current_image_filename)
                    image_stack = tifffile.TiffFile(current_image_path)

                    # get image file info
                    if image_stack.series[0].ndim==3:
                        image_frame_number, image_heigh, image_width = image_stack.series[0].shape
                    elif image_stack.series[0].ndim==2:
                        image_frame_number = 1
                        image_heigh, image_width = image_stack.series[0].shape
                    else:
                        print("Skip {}".format(current_image_filename))
                        break

                    # Read roi files 
                    roi_path = os.path.join(dataset_folder, roi_folder, current_image_filename_base+roi_tag+".mat")
                    roi_list = sio.loadmat(roi_path, struct_as_record=False)
                    roi_structure_fieldname_list = ['CellList','r_out','ROIlist', 'R']

                    for roi_structure_fieldname in roi_structure_fieldname_list:
                        try:
                            roi_list = roi_list[roi_structure_fieldname]
                            break
                        except:
                            pass

                    current_roi_mask = np.zeros((image_heigh, image_width),dtype='uint8')
                    for roi_idx in range(0, roi_list.shape[1]):
                        single_roi_mask = np.zeros((image_heigh, image_width))
                        roi_idx_fieldname_list = ['PixelIdxList','pixel_idx']
                        for roi_idx_fieldname in roi_idx_fieldname_list:
                            try:
                                current_roi_idx = getattr(roi_list[0, roi_idx], roi_idx_fieldname)
                                break
                            except:
                                pass
                        single_roi_mask[np.unravel_index(current_roi_idx-1, (image_heigh,image_width), order='F')] = 1

                        if erosion_iter is not None and erosion_iter != 0:
                            single_roi_mask = ndimage.binary_erosion(single_roi_mask, iterations=erosion_iter).astype(single_roi_mask.dtype)
                        if force_separate is True:
                            dilated_single_roi_mask = ndimage.binary_dilation(single_roi_mask)
                            dilated_current_roi_mask = ndimage.binary_dilation(current_roi_mask)
                            remove_area = np.logical_and(dilated_single_roi_mask, dilated_current_roi_mask)
                            current_roi_mask[single_roi_mask == 1] = 1
                            current_roi_mask[remove_area] = 0
                        else:
                            current_roi_mask[single_roi_mask == 1] = 1

                    self.known_masks[roi_path] = current_roi_mask

                    # calculate the roi/non-roi ratio of whole mask
                    conv_kernel = np.ones((self.patch_size[0],self.patch_size[1]))
                    conv_kernel = conv_kernel/np.sum(conv_kernel)
                    self.known_masks_ratio[roi_path] = signal.fftconvolve(current_roi_mask, conv_kernel, mode='valid')

                    # put everything into data_info
                    for frame_idx in range(image_frame_number-self.frame_number+1):
                        if self.non_overlap:
                            row_idx_list = range(0,image_heigh-self.patch_size[0],self.patch_size[0])
                            col_idx_list = range(0,image_width-self.patch_size[1],self.patch_size[1])
                        else:
                            row_idx_list = range(0,image_heigh-self.patch_size[0])
                            col_idx_list = range(0,image_width-self.patch_size[1])
                        for row_idx in row_idx_list:
                            for col_idx in col_idx_list:
                                data_info = {
                                    "id": current_image_filename_base+"_"+str(frame_idx),
                                    "path": current_image_path,
                                    "roi_path": roi_path,
                                    "row_idx": row_idx,
                                    "col_idx": col_idx,
                                    "frame_idx": frame_idx,
                                    "image_heigh": image_heigh,
                                    "image_width": image_width
                                }
                                self.data_info.append(data_info)

        print("Total {} samples.".format(len(self.data_info)))
        
        # calculate roi/non-roi ratio
        if self.lh_threshold>0:

            low_cover_data_id_list = []
            high_cover_data_id_list = []
            for data_id in tqdm(range(len(self.data_info)),desc="Calculating roi/non-roi ratio"):
                current_row_idx = self.data_info[data_id]['row_idx']
                current_col_idx = self.data_info[data_id]['col_idx']
                current_mask_ratio = self.known_masks_ratio[self.data_info[data_id]['roi_path']]
                self.data_info[data_id]['mask_ratio'] = current_mask_ratio[current_row_idx][current_col_idx]
                #current_mask = current_mask[current_row_idx:current_row_idx + self.patch_size,current_col_idx:current_col_idx + self.patch_size]

                if self.data_info[data_id]['mask_ratio']<self.lh_threshold:
                    low_cover_data_id_list.append(data_id)
                else:
                    high_cover_data_id_list.append(data_id)

            print("Cover threshold: {}".format(self.lh_threshold))
            print("High threshold: {} ({}%)".format(len(high_cover_data_id_list),100*len(high_cover_data_id_list)/len(self.data_info)))
            print("Low threshold: {} ({}%)".format(len(low_cover_data_id_list),100*len(low_cover_data_id_list)/len(self.data_info)))
            #print("%s/%s (%s%%) sample removed at threhold %s." % (len(removed_idx),pre_removal_idx_count,100*round((len(removed_idx)/pre_removal_idx_count), 4),self.non_zero))

        self.low_cover_data_id_list = np.asarray(low_cover_data_id_list)
        self.high_cover_data_id_list = np.asarray(high_cover_data_id_list)
        self.on_epoch_end()

        

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.batch_number==None:
            current_batch_number = int(np.ceil(self.data_id_list.shape[0]) / self.batch_size)
        else:
            current_batch_number = int(self.batch_number)

        return current_batch_number

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        batch_data_id_list = self.data_id_list[index*self.batch_size:(index+1)*self.batch_size]

        if self.channel_axis == 1:
            current_batch_data = np.zeros((batch_data_id_list.shape[0], self.frame_number, self.patch_size[0], self.patch_size[1]),dtype='uint16')
            current_batch_label = np.zeros((batch_data_id_list.shape[0], self.catagory_number, self.patch_size[0], self.patch_size[1]),dtype='uint8')
        else:
            current_batch_data = np.zeros((batch_data_id_list.shape[0], self.patch_size[0], self.patch_size[1], self.frame_number),dtype='uint16')
            current_batch_label = np.zeros((batch_data_id_list.shape[0], self.patch_size[0], self.patch_size[1], self.catagory_number),dtype='uint8')

        for idx,data_id in enumerate(batch_data_id_list):
            current_path = self.data_info[data_id]['path']
            current_frame_idx = self.data_info[data_id]['frame_idx']
            current_row_idx = self.data_info[data_id]['row_idx']
            current_col_idx = self.data_info[data_id]['col_idx']

            current_data = tifffile.imread(current_path,key=range(current_frame_idx,current_frame_idx+self.frame_number))

            if current_data.ndim==2:
                current_data = np.expand_dims(current_data, axis=0)
            current_data = current_data[:, current_row_idx:current_row_idx + self.patch_size[0],
                           current_col_idx:current_col_idx + self.patch_size[1]]

            current_mask = self.known_masks[self.data_info[data_id]['roi_path']]
            current_mask = current_mask[current_row_idx:current_row_idx+self.patch_size[0],current_col_idx:current_col_idx+self.patch_size[1]]

            for category_idx in range(self.catagory_number):
                temp_mask = np.zeros(current_mask.shape)
                temp_mask[current_mask == category_idx] = 1
                try:
                    current_label = np.vstack((current_label, [temp_mask]))
                except:
                    current_label = np.asarray([temp_mask])

            if self.channel_axis == 3:
                current_data = np.rollaxis(current_data, 0, 3)  # convert to NHWC
                current_label = np.rollaxis(current_label, 0, 3) # convert to NHWC


            #category_mask_0 = current_mask==0
            #category_mask_1 = current_mask==1
            #category_mask_2 = current_mask==2

            #if self.channel_axis == 3:
                #current_label = np.dstack([category_mask_0, category_mask_1])  # NHWC
            #else:
                #current_label = np.asarray([category_mask_0, category_mask_1])

            current_batch_data[idx, :, :, :] = current_data
            current_batch_label[idx, :, :, :] = current_label


        if self.image_augmentation=='keras':
            data_datagen = ImageDataGenerator(**self.image_preprocessing_args)
            label_datagen = ImageDataGenerator(**self.label_preprocessing_args)
            #data_datagen.fit(current_batch_data, augment=True, seed=current_seed)
            #label_datagen.fit(current_batch_label, augment=True, seed=current_seed)
            for single_data_idx in range(current_batch_data.shape[0]):
                current_seed = np.random.randint(2 ** 16)
                current_data_datagen = data_datagen.flow(current_batch_data[None,single_data_idx,:,:,:], batch_size=1, shuffle=False, seed=current_seed)
                current_label_datagen = label_datagen.flow(current_batch_label[None,single_data_idx,:,:,:], batch_size=1, shuffle=False, seed=current_seed)
                current_batch_data[single_data_idx,:,:,:] = current_data_datagen.next()
                current_batch_label[single_data_idx,:,:,:] = current_label_datagen.next()
        elif self.image_augmentation=='imgaug':
            if self.channel_axis == 1:
                current_batch_data = np.moveaxis(current_batch_data, 1, 3) # convert NCHW to NHWC 
                current_batch_label = np.moveaxis(current_batch_label, 1, 3) # convert NCHW to NHWC 
            augmentation = iaa.Sequential(self.image_preprocessing_args, random_order=True)
            imgaug.seed(np.random.randint(2**16))
            for single_data_idx in range(current_batch_data.shape[0]):
                current_segmap = imgaug.augmentables.segmaps.SegmentationMapOnImage(current_batch_label[single_data_idx].astype('float'), shape=current_batch_data[single_data_idx].shape, nb_classes=self.catagory_number)
                aug_image, aug_label = augmentation(image=current_batch_data[single_data_idx,:,:,:], segmentation_maps=current_segmap)
                aug_label = np.argmax(aug_label.arr,axis=-1)
                aug_label = keras.utils.to_categorical(aug_label, num_classes=self.catagory_number)
                current_batch_data[single_data_idx,:,:,:] = aug_image  
                current_batch_label[single_data_idx,:,:,:] = aug_label
            if self.channel_axis == 1:
                current_batch_data = np.moveaxis(current_batch_data, 3, 1) # convert NHWC to NCHW
                current_batch_label = np.moveaxis(current_batch_label, 3, 1) # convert NHWC to NCHW

        if self.std_normalization:
            current_batch_data = current_batch_data.astype('float')

        for single_data_idx in range(current_batch_data.shape[0]):
            if self.std_normalization:
                current_data = current_batch_data[single_data_idx,:,:,:]
                if self.channel_axis==3:
                    for channel_idx in range(current_data.shape[2]):
                        current_mean = np.mean(current_data[:,:,channel_idx])
                        current_std = np.std(current_data[:,:,channel_idx])
                        current_data[:,:,channel_idx] = (current_data[:,:,channel_idx]-current_mean)/current_std
                else:
                    for channel_idx in range(current_data.shape[0]):
                        current_mean = np.mean(current_data[channel_idx,:,:])
                        current_std = np.std(current_data[channel_idx,:,:])
                        current_data[channel_idx,:,:] = (current_data[channel_idx,:,:]-current_mean)/current_std

                current_batch_data[single_data_idx,:,:,:] = current_data

            current_label = np.argmax(current_batch_label[single_data_idx], axis=self.channel_axis - 1)
            current_label = keras.utils.to_categorical(current_label, num_classes=self.catagory_number)
            current_batch_label[single_data_idx] = current_label

        return current_batch_data, current_batch_label

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        if self.lh_threshold > 0:
            low_cover_id_count = np.int(np.min((self.lh_ratio*self.high_cover_data_id_list.shape[0],self.low_cover_data_id_list.shape[0])))
            np.random.shuffle(self.high_cover_data_id_list)
            np.random.shuffle(self.low_cover_data_id_list)
            self.data_id_list = np.hstack((self.high_cover_data_id_list,self.low_cover_data_id_list[:low_cover_id_count]))
        else:
            self.data_id_list = self.high_cover_data_id_list

        if self.shuffle == True:
            np.random.shuffle(self.data_id_list)

    def show_data(self,n=0,idx=1):
        if n==0:
            n = self.batch_size
        current_data, current_label = self.__getitem__(idx)
        fig = plt.figure(figsize=(2 * 5, n * 5))
        gs = gridspec.GridSpec(n, 2, width_ratios=[1, 1], wspace=0.1, hspace=0.0)

        for current_data_idx in range(n):
            ax = fig.add_subplot(gs[current_data_idx, 0])

            if self.frame_number>1:
                ax.imshow(np.max(current_data[current_data_idx], axis=self.channel_axis - 1)-np.min(current_data[current_data_idx], axis=self.channel_axis - 1))
            else:
                ax.imshow(np.max(current_data[current_data_idx], axis=self.channel_axis - 1))

            ax.contour(np.argmax(current_label[current_data_idx], axis=self.channel_axis - 1))
            ax = fig.add_subplot(gs[current_data_idx, 1])
            ax.imshow(np.argmax(current_label[current_data_idx], axis=self.channel_axis - 1))
        plt.show()