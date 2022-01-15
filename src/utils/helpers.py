"""
Collection of some helpers function
"""

import librosa 

# Convert BGR image format to RGB
def bgr_to_rgb(image):
	image_ = image.copy()
	image[:, :, 0] = image[:, :, -1].copy()
	image[:, :, 2] = image[:, :, 0].copy()
	return image

# Load wave-to-mel
def load_mel_from_file(
    filepath,
    sampling_rate = 16_000,
    n_fft = 1024,
    hop_length = 256,
    win_length = 1024,
    n_mels = 80,
):
    # Load raw waveform
    wave = librosa.load(filepath, sr = sampling_rate)[0]
    # Process wave to mel
    mel = librosa.feature.melspectrogram(
        y = wave,
        sr = sampling_rate,
        n_mels = n_mels,
        n_fft = n_fft,
        hop_length = hop_length,
        win_length = win_length,
    )
    # Take log mel-spectrogram
    mel = librosa.util.normalize(librosa.power_to_db(mel)).T
    
    return mel