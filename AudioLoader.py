import os

jupyter_backend = 'module://ipykernel.pylab.backend_inline'

import matplotlib.pyplot as plt
################################################
def set_plt_for_spec():
    [old_width, old_height] = plt.rcParams['figure.figsize']
    font =  plt.rcParams['font']
    plt.rc('font', **{

                      'size'   : 10})
    plt.rc('axes', titlesize=8)
    if old_width < 10:
        plt.rc('figure', figsize=[(old_width * 3)/2, old_height/2])
    
    return old_width, old_height, font
###############################################


from tqdm import tqdm 

import torch
import torch.utils.data as data

import torchaudio




AUDIO_EXTENSIONS = ['.mp3', '.wav']


class AudioLoader(data.Dataset):

    def __init__(self, root, sample_rate = None, transform  = None):
        classes, class_to_idx = self.find_classes(root)
        print("classes", classes)
        audios, names = self.make_dataset(root, class_to_idx)
        if len(audios) == 0:
            raise(RuntimeError("Found 0 audios in subfolders of: " + root + "\n"
                               "Supported audio extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.audios = audios
        self.names = names
        self.classes = classes
        self.count_classes = len(classes)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.sample_rate = sample_rate

        self.buffer_spectrograme = [None, None]


    def is_audio_file(self, filename):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in AUDIO_EXTENSIONS)   

    def make_dataset(self, dir, class_to_idx):
        audios = []; names  = []

        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_audio_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        audios.append(item)
                        names.append(fname)

        return audios, names

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx    

    def delete_audio_file(self, index):
        print(index)
        print(os.remove(self.audios[index][0]))
        self.audios.pop(index)
        self.names.pop(index)

    def spectrogram(self, index, show = True, save = False, dpi=30, path = ''):
        if not show: plt.switch_backend('Agg')

        audio, self.sample_rate = torchaudio.load(self.audios[index][0])

        audio = audio.numpy()

        num_channels, _ = audio.shape

        fig, axes = plt.subplots(num_channels, 1)

        if num_channels == 1:
            axes = [axes]
        else: print("Num channels is %d" % num_channels)    

        for c in range(num_channels):
            axes[c].specgram(audio[c], Fs=self.sample_rate)
            if save:
                axes[c].axis('off')
                
                #print(self.audios[index][0][ :-4].replace('wavs', 'spec') +'.png')

                fig.savefig( self.audios[index][0][ :-4].replace('wavs', 'spec') +'.png', bbox_inches='tight', pad_inches=0,  dpi=dpi)
                axes[c].axis('on')
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')

        if show:
            fig.suptitle(self.names[index])
            plt.show()
        

        plt.clf(); plt.cla(); plt.close('all')
        plt.switch_backend(jupyter_backend)


    def plot_audio(self, waveform, name="Waveform", save=False):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / self.sample_rate

        fig, axes = plt.subplots(num_channels, 1)

        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            axes[c].set_xlabel('Time')
            axes[c].set_ylabel('Amplitude')
            fig.suptitle(name) 
            if save:
                fig.savefig('./'+name[:-4]+'.png', dpi=200, bbox_inches='tight', pad_inches=0)

            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')

        plt.show(); plt.clf()
        plt.cla();  plt.close('all')    

    def plot_ind(self, index, save = False):
        audio, self.sample_rate = torchaudio.load(self.audios[index][0])
        self.plot_audio(audio, '('+str(index)+') ' + self.names[index], save)

    def setSampleRate(self, new_sample_rate):
        self.sample_rate = new_sample_rate
    
    def ConcertSampleRate(self, index, new_sample_rate):
        path = self.audios[index][0].split('/')
        path = path[0]+'_'+str(new_sample_rate)+'SR'+'/'+path[1]+'/'+path[2]

        audio, _ = torchaudio.load(self.audios[index][0])
        
        if not os.path.exists(path[0]+'_'+str(new_sample_rate)+'SR'+'/'+path[1]):
            os.makedirs(path[0]+'_'+str(new_sample_rate)+'SR'+'/'+path[1])

        torchaudio.save(path, audio, new_sample_rate)

    def ConcertSampleRateAll(self, new_sample_rate):
        for ind in tqdm(range(len(self.audios))):
            self.ConcertSampleRate(ind, new_sample_rate)


    def get_audio(self, index):
        audio, self.sample_rate = torchaudio.load(self.audios[index][0])
        return audio, self.audios[index][1]


    def __getitem__(self, index):
        audio, target = self.get_audio(index)
        
        if self.transform is not None:
            audio = self.transform(audio)

        return audio, target, self.names[index], index

    def __len__(self):
        return len(self.audios)
        