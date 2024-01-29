import torch
import numpy as np
from sklearn.metrics import pairwise_distances  # Cosine similarity

esc_50_labels = [
    'Dog', 'Rain', 'Crying baby', 'Door knock', 'Helicopter',
    'Rooster', 'Sea waves', 'Sneezing', 'Mouse click', 'Chainsaw',
    'Pig', 'Crackling fire', 'Clapping', 'Keyboard typing', 'Siren',
    'Cow', 'Crickets', 'Breathing', 'Door, wood creaks', 'Car horn',
    'Frog', 'Chirping birds', 'Coughing', 'Can opening', 'Engine',
    'Cat', 'Water drops', 'Footsteps', 'Washing machine', 'Train',
    'Hen', 'Wind', 'Laughing', 'Vacuum cleaner', 'Church bells',
    'Insects (flying)', 'Pouring water', 'Brushing teeth', 'Clock alarm', 'Airplane',
    'Sheep', 'Toilet flush', 'Snoring', 'Clock tick', 'Fireworks',
    'Crow', 'Thunderstorm', 'Drinking, sipping', 'Glass breaking', 'Hand saw'
]

def classify_this_audio_file(processor, model, audio, text_embeddings, sampling_rate = 48000):
    assert sampling_rate == 48000, "Sampling rate should be 48000."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_file = processor(audios=audio, sampling_rate=sampling_rate,return_tensors="pt", padding=True)
    for key, value in audio_file.items():
        audio_file[key] = value.to(device)
    with torch.inference_mode():
        audioembeddings = model.get_audio_features(**audio_file)

    inputs_text = processor(text=text_embeddings, return_tensors="pt", padding=True)
    for key, value in inputs_text.items():
        inputs_text[key] = value.to(device)

    # Classify
    with torch.inference_mode():
        outputs_text = model.get_text_features(**inputs_text)

    cosine_similarity = 1-pairwise_distances(audioembeddings.cpu(), outputs_text.cpu(), metric="cosine")
    return text_embeddings[np.argmax(cosine_similarity[0,:])]
