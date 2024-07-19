"Name       : Roi Halali & Dor Kershberg        "
"Titel      : Random Signals Prossesing project "
"Sub Titel  : Database                          "

#%%
# Libraries:
import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display
import python_speech_features as spf
from pydub import AudioSegment
from pydub.silence import split_on_silence
import shutil
from scipy import signal
import malaya_speech
import IPython.display as ipd
import pyglet
import sounddevice

#%%
# Functions:
Nj=[]  #number frames of each speaker
Njp=[] #number frames per feachers of each speaker
ejp=[] #Njp/Nj

def radar_features(num_speaker,phoneme,start,end):
    #Pre-Process
    
    count_phonemes_dic=['b','d','f','g','eh','k','l','m','n','p','q','r','s','sh','t','th','v','z','ch','zh','iy','ae','eh','uw','aa','ow']
    features=np.zeros((1,13))            
            
    num_frames=np.floor((end-start)/(0.02*16e03))
    
    if (num_speaker+1==len(Nj)):
        Nj[num_speaker]+=num_frames
        Njp[num_speaker]+=eval(phoneme)*num_frames
       
    elif (num_speaker+1>len(Nj)):
          Nj.append(num_frames) 
          Njp.append(eval(phoneme)*num_frames)

def spyder_diagram(features):
    
    features=features.T
    # features=np.trunc(features)
    # features.astype(int)
    
    categories = ['Vocalic','Consonantal','high','Back', 'Low','Anterior','Coronal'
        ,'Round','Tense','Voice','Continuant','Nasal','Strident']

    x = features
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(features))
    
    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    plt.plot(label_loc, x, label='Speaker 0')
    plt.title('Speaker comparison', size=20)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend()
    plt.show()

def spectogram(Signal):
    # spectogram:
    target_sr=16000
    # LPF:
    LPF = signal.butter(6, 5500, 'low', fs=16000, output='sos')
    filtered = signal.sosfilt(LPF, Signal)

    # HPF:
    HPF = signal.butter(3, 400, 'hp', fs=16000, output='sos')
    filtered = signal.sosfilt(HPF, filtered)

    frame_size = int(np.floor(16 * (10 ** -3) / (1 / target_sr)))
    hoplen = int(frame_size / 2)

    n_fft = int(256)
    stft = librosa.stft(filtered, n_fft=n_fft, win_length=frame_size, hop_length=hoplen)
    stft = np.abs(stft)

    # #plot results:
    # plt.title(phoneme_name)
    # plt.title('Spectogram')
    # display.specshow(stft, sr=target_sr, x_axis='time', y_axis='linear', hop_length=hoplen, win_length=frame_size,fmax=int(5500))
    # plt.colorbar(format="%+2.f")
    # plt.show()

    # fmax= 5500
    stft = stft[0:80, :]

    # normalization:
    stft = (stft - np.min(stft)) / (np.max(stft) - np.min(stft))
    return stft

def fix_phoneme_len(audio, begin_phon, end_phone,phoneme_name):

    target_sr = 16000 #note: The sample rate used for audio for TIMIT is 16000.    
    phoneme_fixed_len = int(0.03 * target_sr) #note: (mel window length)*(semple rate)=phonema length
    phon_len = end_phone - begin_phon 
    
    # take missing samples from adjacent samples    
    if phon_len < phoneme_fixed_len*1.3 :
        return None
               
    elif phon_len >= phoneme_fixed_len*1.3:
        #start
        samples_start = audio[begin_phon:begin_phon+phoneme_fixed_len]
        # center
        phon_center = (begin_phon + end_phone) // 2
        if phon_center % 2 != 0:
            phon_center += 1
        samples_center = audio[phon_center - (phoneme_fixed_len // 2):phon_center + (phoneme_fixed_len // 2)]
        #end:
        samples_end = audio[-phoneme_fixed_len:]

    #mfcc:
    # mfcc0= spf.base.mfcc(samples_start,16000,winlen=0.016,winstep=0.008,winfunc=np.hamming,nfft= int(256))
    # mfcc1=spf.base.mfcc(samples_start,16000,winlen=0.016,winstep=0.008,winfunc=np.hamming,nfft= int(256))
    # #plot results:
    # plt.title(phoneme_name)
    # plt.title('Spectogram')
    # display.specshow(stft, sr=target_sr, x_axis='time', y_axis='linear', hop_length=hoplen, win_length=frame_size,fmax=int(5500))
    # plt.colorbar(format="%+2.f")
    # plt.show()

    return spectogram(samples_start), spectogram(samples_center),spectogram(samples_end)

def remove_silents(audio):
    # audio_trim,sr = librosa.effects.trim(audio, top_db=20)

    audio_trim_int = malaya_speech.astype.float_to_int(audio)
    audio_trim_int2 = AudioSegment(
        audio_trim_int.tobytes(),
        frame_rate=16000,
        sample_width=audio_trim_int.dtype.itemsize,
        channels=1
    )
    
    audio_chunks = split_on_silence(
        audio_trim_int2,
        min_silence_len = 200,
        silence_thresh = -30,
        keep_silence = 100,
    )

    y = sum(audio_chunks)
    y = np.array(y.get_array_of_samples())
    y = malaya_speech.astype.int_to_float(y)

    return y

def prepere_new_input(dir_audio,dir_save):
    target_sr = 16000 #note: The sample rate used for audio for TIMIT is 16000.    
    phoneme_fixed_len = int(0.03 * target_sr) #note: (mel window length)*(semple rate)=phonema length
    for phoneme_file in glob.iglob(dir_audio + '**', recursive=True):
        if phoneme_file[-3:] == 'wav':  
            wav_file_name = phoneme_file[:-3] + 'wav'               # changing into wav file
            # audio, fs = librosa.core.load(wav_file_name, sr=None)   #load the file from lobrosa
            audio, sr = malaya_speech.load(wav_file_name)

            #Remove silents:
            audio_trim=remove_silents(audio)
            # audio_trim,sr = librosa.effects.trim(audio, top_db=10)
            #pre-processing audio file:
            audio = librosa.effects.preemphasis(audio_trim)

            audio_list=[]
            for i in range((len(audio)//phoneme_fixed_len)): 
                temp= spectogram( audio[i*(phoneme_fixed_len//2):i*(phoneme_fixed_len//2)+phoneme_fixed_len] )

                np.save(dir_save+str(i), temp, allow_pickle=True) #saving audio file
                
            return audio

def reduce_phonemes(old_phoneme):
    if old_phoneme in ao_v:
        return 'ao'
    elif old_phoneme in ah_v:
        return 'ah'
    elif old_phoneme in k_v:
        return 'k'
    elif old_phoneme in t_v:
        return 't'
    elif old_phoneme in axr_v:
        return 'axr'
    elif old_phoneme in ay_v:
        return 'ay'
    elif old_phoneme in bcl_v:
        return 'bcl'
    elif old_phoneme in t_v:
        return 't'
    elif old_phoneme in dh_v:
        return 'dh'
    elif old_phoneme in m_v:
        return 'm'
    elif old_phoneme in eng_v:
        return 'eng'
    else:
        return old_phoneme
    
#reduce list hebrew
k_v     =['k','kcl']
t_v     =['t','tcl','dcl']
    
def reduce_phonemes_heb(old_phoneme):
    if old_phoneme in k_v:
        return 'k'
    elif old_phoneme in t_v:
        return 't'
    else:
        return old_phoneme
    

#%%
# Dirs & Mode:

train = False
test  = False
val   = False

#data dirs:
train_data_dir      = 'TIMIT/TRAIN/'
test_data_dir       = 'TIMIT/TEST/'
val_data_dir        = 'TIMIT/EVALUATION/'
predict_data_dir    = 'TIMIT/PREDICTION/'


#save dirs phonemes
train_save_dir      = 'data/train/'
test_save_dir       = 'data/test/'
val_save_dir        = 'data/evaluation/'
predict_save_dir    = 'data/prediction/'

#save dirs speekers
speakers_save_dir   = 'data/speakers data train/'


if train==True:
    data_dir = train_data_dir
    save_dir = train_save_dir
         
if test==True:
    data_dir = test_data_dir
    save_dir = test_save_dir
    
if val==True:
    data_dir = val_data_dir
    save_dir = val_save_dir

    

#%%
# SPE definitions:
    
ignore_list = ['h#','1', '2','epi','pau'] #irrelevant phonemes 

#reduce list
ah_v    =['ah','ax','ax-h']
ao_v    =['ao','aw']
k_v     =['k','kcl']
axr_v   =['axr','ih']
ay_v    =['ay','oy']
bcl_v   =['bcl','pcl']
t_v     =['t','tcl','dcl']
dh_v    =['dh','el','th']
m_v     =['m','n']
eng_v   =['eng','ng']

#%%
# #phoneme to featcher vector:
# aa  = np.array([1,0,0,1,1,0,0,0,1,1,1,0,0],dtype=int)
# ae  = np.array([1,0,0,0,1,0,0,0,1,1,1,0,0],dtype=int)
# ah  = np.array([1,0,0,1,0,0,0,0,0,1,1,0,0],dtype=int)
# #ax  = np.array([1,0,0,1,0,0,0,0,0,1,1,0,0],dtype=int)
# #ax_h= np.array([1,0,0,1,0,0,0,0,0,1,1,0,0],dtype=int)
# ao  = np.array([1,0,0,1,1,0,0,1,1,1,1,0,0],dtype=int)
# aw  = np.array([1,0,0,1,1,0,0,1,1,1,1,0,0],dtype=int)
# axr = np.array([1,0,0,0,0,0,0,0,0,1,1,0,0],dtype=int)
# # ih  = np.array([1,0,0,0,0,0,0,0,0,1,1,0,0],dtype=int)
# ay  = np.array([0,0,0,1,1,0,0,0,1,1,1,0,0],dtype=int)
# oy  = np.array([0,0,0,1,1,0,0,0,1,1,1,0,0],dtype=int)
# b   = np.array([0,1,0,0,0,1,0,0,0,1,0,0,0],dtype=int)
# bcl = np.array([0,1,0,0,0,1,0,0,0,0,0,0,0],dtype=int)
# # pcl = np.array([0,1,0,0,0,1,0,0,0,0,0,0,0],dtype=int)
# ch  = np.array([0,1,0,1,0,0,1,0,0,0,0,0,1],dtype=int)
# d   = np.array([0,1,1,0,0,1,1,0,0,1,0,0,0],dtype=int)
# dh  = np.array([0,1,0,0,0,1,1,0,0,1,1,0,0],dtype=int)
# el  = np.array([0,1,0,0,0,1,1,0,0,1,1,0,0],dtype=int)
# th  = np.array([0,1,0,0,0,1,1,0,0,1,1,0,0],dtype=int)
# dx  = np.array([1,1,0,0,0,1,1,0,0,1,0,0,0],dtype=int)
# eh  = np.array([0,0,0,0,0,0,0,0,0,1,1,0,0],dtype=int)
# em  = np.array([0,1,0,0,0,1,1,0,0,1,0,1,0],dtype=int)
# en  = np.array([0,1,0,0,0,1,1,0,0,1,0,1,0],dtype=int)
# eng = np.array([0,1,0,1,0,0,0,0,0,1,0,1,0],dtype=int)
# ng  = np.array([0,1,0,1,0,0,0,0,0,1,0,1,0],dtype=int)
# er  = np.array([1,0,0,0,0,0,0,0,0,1,1,0,0],dtype=int)
# ey  = np.array([0,0,0,0,0,0,0,0,1,1,1,0,0],dtype=int)
# f   = np.array([0,1,0,0,0,1,0,0,0,0,1,0,1],dtype=int)
# g   = np.array([0,1,0,1,0,0,0,0,0,1,1,0,0],dtype=int)
# gcl = np.array([0,1,0,1,0,0,0,0,0,0,0,0,0],dtype=int)
# hh  = np.array([0,1,0,0,1,0,0,0,0,0,1,0,0],dtype=int)
# hv  = np.array([1,1,0,0,1,0,0,0,0,1,1,0,0],dtype=int)

# ix  = np.array([1,0,1,0,0,0,0,0,0,1,1,0,0],dtype=int)
# iy  = np.array([0,0,1,0,0,0,0,0,1,1,1,0,0],dtype=int)
# jh  = np.array([0,1,1,0,0,0,1,0,0,1,0,0,1],dtype=int)
# k   = np.array([0,1,1,1,0,0,0,0,0,0,0,0,0],dtype=int)
# kcl = np.array([0,1,1,1,0,0,0,0,0,0,0,0,0],dtype=int)
# l   = np.array([0,1,1,0,0,1,1,0,0,1,1,0,0],dtype=int)
# m   = np.array([0,1,0,0,0,1,0,0,0,1,0,1,0],dtype=int)
# n   = np.array([0,1,0,0,0,1,0,0,0,1,0,1,0],dtype=int)
# nx  = np.array([1,1,1,0,0,1,1,0,0,1,0,1,0],dtype=int)
# ow  = np.array([1,0,0,1,0,0,0,1,1,1,1,0,0],dtype=int)
# p   = np.array([0,1,1,0,0,1,0,0,0,0,0,0,0],dtype=int)
# q   = np.array([0,1,0,1,1,0,0,0,0,0,0,0,0],dtype=int)
# r   = np.array([1,1,0,0,0,0,1,0,0,1,1,0,0],dtype=int)
# s   = np.array([0,1,0,0,0,1,1,0,0,0,1,0,1],dtype=int)
# sh  = np.array([0,1,1,0,0,0,1,0,0,0,1,0,1],dtype=int)
# t   = np.array([0,1,0,0,0,1,1,0,0,0,0,0,0],dtype=int)
# tcl = np.array([0,1,0,0,0,1,1,0,0,0,0,0,0],dtype=int)
# dcl = np.array([0,1,0,0,0,1,1,0,0,0,0,0,0],dtype=int)
# uh  = np.array([1,0,1,1,0,0,0,0,0,1,1,0,0],dtype=int)
# uw  = np.array([1,0,1,1,0,0,0,1,1,1,1,0,0],dtype=int)
# ux  = np.array([1,0,1,1,0,0,0,1,1,1,1,0,0],dtype=int)
# v   = np.array([0,1,0,0,0,1,0,0,0,1,1,0,1],dtype=int)
# w   = np.array([0,1,1,0,0,0,0,1,0,1,1,0,0],dtype=int)
# y   = np.array([0,1,1,0,0,0,0,0,0,1,1,0,0],dtype=int)
# z   = np.array([0,1,0,0,0,1,1,0,0,1,1,0,1],dtype=int)
# zh  = np.array([0,1,1,0,0,0,1,0,0,1,1,0,1],dtype=int)

#%%

ignore_list_heb = ['ax-h','h#','1', '2','epi','pau','ah','ax','ax_h','ao','aw','axr','ih','ay',
                   'oy','bcl','pcl','dh','el','dx','em','en','eng','ng','er','ey'
                   ,'gcl','hh','hv','ix','jh','nx','uh','ux','w','y'] #irrelevant phonemes 

#Hebrew Phonemes
b   = np.array([0,1,0,0,0,1,0,0,0,1,0,0,0],dtype=int)
d   = np.array([0,1,1,0,0,1,1,0,0,1,0,0,0],dtype=int)
f   = np.array([0,1,0,0,0,1,0,0,0,0,1,0,1],dtype=int)
g   = np.array([0,1,0,1,0,0,0,0,0,1,1,0,0],dtype=int)
k   = np.array([0,1,1,1,0,0,0,0,0,0,0,0,0],dtype=int)
# kcl = np.array([0,1,1,1,0,0,0,0,0,0,0,0,0],dtype=int)
l   = np.array([0,1,1,0,0,1,1,0,0,1,1,0,0],dtype=int)
m   = np.array([0,1,0,0,0,1,0,0,0,1,0,1,0],dtype=int)
n   = np.array([0,1,0,0,0,1,0,0,0,1,0,1,0],dtype=int)
p   = np.array([0,1,1,0,0,1,0,0,0,0,0,0,0],dtype=int)
q   = np.array([0,1,0,1,1,0,0,0,0,0,0,0,0],dtype=int)
r   = np.array([1,1,0,0,0,0,1,0,0,1,1,0,0],dtype=int)
s   = np.array([0,1,0,0,0,1,1,0,0,0,1,0,1],dtype=int)
sh  = np.array([0,1,1,0,0,0,1,0,0,0,1,0,1],dtype=int)
t   = np.array([0,1,0,0,0,1,1,0,0,0,0,0,0],dtype=int)
#tcl = np.array([0,1,0,0,0,1,1,0,0,0,0,0,0],dtype=int)
#dcl = np.array([0,1,0,0,0,1,1,0,0,0,0,0,0],dtype=int)
th  = np.array([0,1,0,0,0,1,1,0,0,1,1,0,0],dtype=int)
v   = np.array([0,1,0,0,0,1,0,0,0,1,1,0,1],dtype=int)
z   = np.array([0,1,0,0,0,1,1,0,0,1,1,0,1],dtype=int)# 'חסר כ' סופית,ח' וע

#special
ch  = np.array([0,1,0,1,0,0,1,0,0,0,0,0,1],dtype=int)
zh  = np.array([0,1,1,0,0,0,1,0,0,1,1,0,1],dtype=int)

#vowels
# iy  = np.array([0,0,1,0,0,0,0,0,1,1,1,0,0],dtype=int)#'אי
# ae  = np.array([1,0,0,0,1,0,0,0,1,1,1,0,0],dtype=int)#'אה
# eh  = np.array([0,0,0,0,0,0,0,0,0,1,1,0,0],dtype=int)#'אה
# uw  = np.array([1,0,1,1,0,0,0,1,1,1,1,0,0],dtype=int)#'אוו
# aa  = np.array([1,0,0,1,1,0,0,0,1,1,1,0,0],dtype=int)#'אאאא
# ow  = np.array([1,0,0,1,0,0,0,1,1,1,1,0,0],dtype=int)#'או

aa  = np.array([1,0,1,0],dtype=int)#'אאאא
eh  = np.array([0,0,0,0],dtype=int)#'אה
iy  = np.array([0,1,0,0],dtype=int)#'אי
uw  = np.array([1,1,1,1],dtype=int)#'אוו
ow  = np.array([1,0,1,1],dtype=int)#'או

#
# #%%
# # phonem wav from PHN:
# flag=False
# if flag==True:
#     shutil.rmtree(speakers_save_dir)
# i=0
# speaker=0
# speaker_sentence=0
#
# sentenses_dir=speakers_save_dir+'/'+'speaker'+str(speaker)+'/'
# os.makedirs(sentenses_dir)
# phoneme_speaker_count=np.zeros((26,1),dtype=int)     #phoneme vector for each sentence
#
# for phoneme_file in glob.iglob(data_dir + '**', recursive=True):
#     if phoneme_file[-3:] == 'PHN':          #check PHN file's from lest three letters
#
#         #set sentence & speeker number:
#         if speaker_sentence%10==0 and speaker_sentence!=0 :          #new speaker after 10 sentenses
#             speaker+=1
#
#             phoneme_speaker_count=np.zeros((26,1),dtype=int)     #phoneme vector for each sentence
#             speaker_sentence=0
#             sentenses_dir=speakers_save_dir+'/'+'speaker'+str(speaker)+'/'
#             os.makedirs(sentenses_dir)
#
#         speaker_sentence+=1                 #sentence number
#
#         #load file:
#         wav_file_name = phoneme_file[:-3] + 'WAV.wav'           # changing into wav file
#         audio, fs = librosa.core.load(wav_file_name, sr=None)   #load the file from librosa
#
#         #pre-processing auidio file:
#         audio = librosa.effects.preemphasis(audio)
#         np.save(sentenses_dir+'audio of sentence '+str(speaker_sentence), audio, allow_pickle=True) #saving audio file
#
#         with open(phoneme_file) as file:
#             lines = file.readlines() # get rid of \n
#         for line in lines:
#             splitted_line = line.split(' ')
#             phoneme = splitted_line[-1][:-1]
#
#             # if not in ignore list, saving the phoneme
#             if phoneme not in ignore_list_heb:
#                 phoneme=reduce_phonemes_heb(phoneme)
#                 #phoneme_speaker_count,counter_spe=count_phoneme(phoneme_speaker_count,phoneme,count_spe)
#                 i+=1
#
#                 #phonema length
#                 end_phone = int(splitted_line[1])
#                 begin_phon = int(splitted_line[0])
#                 radar_features(speaker,phoneme,begin_phon,end_phone)
#                 #saving the phoneme data:
#                 mat = fix_phoneme_len(audio, begin_phon, end_phone,phoneme)
#
#                 if mat is not None:
#                       np.save(save_dir+'/'+phoneme+'/'+str(phoneme)+str(i)+"0", mat[0], allow_pickle=True)
#                       np.save(save_dir+'/'+phoneme+'/'+str(phoneme)+str(i)+"1", mat[1], allow_pickle=True)
#                       np.save(save_dir+'/'+phoneme+'/'+str(phoneme)+str(i)+"2", mat[1], allow_pickle=True)
#
#         np.save(sentenses_dir+'phonemes of sentence '+str(speaker_sentence),phoneme_speaker_count)

# audio = predict(predict_data_dir,predict_save_dir)
# sounddevice.play(audio, 16000)


