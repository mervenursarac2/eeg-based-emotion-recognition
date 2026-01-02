import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import mne.io #ham verileri okumaktan sorumlu olan modül
from mne.preprocessing import ICA

matplotlib.use('Qt5Agg')  # kanal gizleme işaretleme kaydırma gibi etkileşimli grafikler oluşturabilmeyi sağlar

#veri okuma
data_path = 'C:\\Users\\merve\\Desktop\\deap-dataset\\raw\\s01.bdf'
raw_data = mne.io.read_raw_bdf( #bdf formatını okumak için olan fonksiyon
    input_fname = data_path,
    preload = True, # verinin tamamını belleğe yükler ve işlemi hızlandırır
)

print("data loaded")
print(raw_data)


#filtreleme
print("filtering starting")
raw_data.notch_filter(freqs=[50, 100], fir_window='hann') # çentik filtreleme ingiltere 50hz olarak ele aldık
raw_data.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin') #
print("filtering finish")

#downsampling
raw_data.resample(128) # Örnek: 128Hz'e düşür
print("resampled to 128 Hz")

raw_data.set_channel_types({
    'EXG1': 'eog', 'EXG2': 'eog', 'EXG3': 'eog', 'EXG4': 'eog',
    'EXG5': 'eog', 'EXG6': 'eog', 'EXG7': 'eog', 'EXG8': 'eog',
    'GSR1': 'misc', 'GSR2': 'misc',
    'Erg1': 'misc', 'Erg2': 'misc',
    'Resp': 'misc',
    'Plet': 'misc',
    'Temp': 'misc'
})
 
# --------------------------------------------------
# MONTAGE (KANAL KONUMU) EKLE
# --------------------------------------------------
montage = mne.channels.make_standard_montage("standard_1020")
raw_data.set_montage(montage, on_missing='ignore')


picks_eeg = mne.pick_types(
    raw_data.info,
    eeg=True,
    eog=False,
    stim=False,
    exclude='bads'
)


# ICA modeli
ica = ICA(
    n_components=20,
    random_state=97,
    method='fastica',
    max_iter='auto'
)


# ICA fit
ica.fit(raw_data, picks=picks_eeg)

# Otomatik EOG tespiti
eog_indices, eog_scores = ica.find_bads_eog(
    raw_data,
    ch_name=['EXG1', 'EXG2']
)

print("Otomatik bulunan EOG componentleri:", eog_indices)

# Otomatik bulunanlar varsa incele
if len(eog_indices) > 0:
    ica.plot_properties(raw_data, picks=eog_indices)
else:
    print("Otomatik EOG componenti bulunamadı.")

# Tüm componentleri görsel inceleme
ica.plot_components()

# ================
# MANUEL KARAR
# ================
# örnek: [0, 3] göz/kas gibi göründüyse
ica.exclude = eog_indices  # ya da manuel ekle: [0, 3]

# ICA uygula
raw_clean = ica.apply(raw_data.copy())


# finding events (yalnızca event başlangıçları)
events = mne.find_events(raw_data, stim_channel='Status')

# --------------------------------------------------
# 6. GERÇEK 40 TRIAL SEÇİMİ (ZAMAN ARALIĞINA GÖRE)
# --------------------------------------------------
sfreq = raw_data.info['sfreq']
min_trial_gap = int(60 * sfreq)  # trial arası minimum 60 sn

trial_events = [events[0]]
for ev in events[1:]:
    if ev[0] - trial_events[-1][0] > min_trial_gap:
        trial_events.append(ev)

trial_events = np.array(trial_events)

print("trial number:", len(trial_events))  

#yalnızca eeg kanallarını istiyoruz
# raw_data.pick(picks='eeg')


#epochlar ve baseline temizliği-
epochs = mne.Epochs(
    raw_data,
    trial_events,
    tmin=0.0,          # trial başlangıcı
    tmax=63.0,         # 3 sn baseline +  video
    baseline=(0, 3),   # BASELINE REMOVAL
    preload=True
)

print(epochs)

#baseline removal kontrol
sfreq = raw_data.info['sfreq']
data = epochs.get_data()
baseline_samples = int(3 * sfreq)

print(
    "Baseline sonrası ilk 3 saniye ortalaması:",
    data[:, :, :baseline_samples].mean()
)


epochs[0].plot()
plt.show()


