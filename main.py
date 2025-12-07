import streamlit as st
import pandas as pd
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from my_functions import Cnn14_emb64_Spec
import glob

# Page configuration
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# Title and description
st.title("🎵 Music Recommendation System")
st.markdown("Upload an audio file to get personalized music recommendations based on deep learning embeddings!")

# Fixed parameters for spectrogram generation
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TARGET_LEN = 1292  # Target length (1292) in frames for 30s audio
TARGET_DURATION = 30.0  # Target duration in seconds

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache decorator for loading model and embeddings (only load once)
def load_sample_songs(music_dir="data/music"):
    """Load available sample songs from the music directory"""
    try:
        audio_files = glob.glob(os.path.join(music_dir, "*.mp3"))
        audio_files.extend(glob.glob(os.path.join(music_dir, "*.wav")))
        audio_files.extend(glob.glob(os.path.join(music_dir, "*.flac")))
        
        sample_songs = {}
        for filepath in sorted(audio_files):
            filename = os.path.basename(filepath)
            song_name = os.path.splitext(filename)[0]
            sample_songs[song_name] = filepath
        
        if sample_songs:
            st.success(f"✓ Found {len(sample_songs)} sample songs")
        return sample_songs
    except Exception as e:
        st.warning(f"Could not load sample songs: {e}")
        return {}

@st.cache_resource
def load_model():
    """Load the pre-trained encoder model"""
    try:
        encoder = Cnn14_emb64_Spec().to(device)
        checkpoint = torch.load('data/best_contrastive_temp002.pth', map_location=device)
        
        if isinstance(encoder, nn.DataParallel):
            encoder.module.load_state_dict(checkpoint['encoder'])
            encoder_single = encoder.module
        else:
            encoder.load_state_dict(checkpoint['encoder'])
            encoder_single = encoder
        
        encoder_single.eval()
        st.success("✓ Model loaded successfully!")
        return encoder_single
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_metadata():
    """Load track metadata"""
    try:
        info_df = pd.read_csv("data/fma_metadata_clean_CNN.csv")
        info_df["track_id"] = info_df['track_id'].astype(str).str.zfill(6)
        st.success(f"✓ Loaded metadata for {len(info_df)} tracks")
        return info_df
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

@st.cache_data
def load_embeddings():
    """Load pre-computed embeddings"""
    try:
        embeddings_dict = np.load('data/embeddings_contrastive002.npy', allow_pickle=True).item()
        st.success(f"✓ Loaded {len(embeddings_dict)} embeddings")
        return embeddings_dict
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

def pad_spectrogram(x, target_len=TARGET_LEN):
    """Pad or truncate spectrogram to target length"""
    cur_len = x.shape[1]
    if cur_len == target_len:
        return x
    elif cur_len < target_len:
        pad_width = target_len - cur_len
        return np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return x[:, :target_len]

def audio_to_melspec(audio_file):
    """Convert audio file to log-scaled mel-spectrogram with duration preprocessing"""
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=SR)
        
        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)
        target_samples = int(TARGET_DURATION * SR)
        
        # Preprocess based on duration
        if duration < TARGET_DURATION * 0.9:  # Less than 27s
            # Too short: repeat audio to reach target duration
            num_repeats = int(np.ceil(target_samples / len(y)))
            y = np.tile(y, num_repeats)[:target_samples]
            st.warning(f"⚠️ Audio is {duration:.1f}s (< 27s). Repeating to {TARGET_DURATION}s for better matching.")
        elif duration < TARGET_DURATION:  # Between 27s-30s
            # Close to target: pad with zeros to avoid unnatural looping
            pad_samples = target_samples - len(y)
            y = np.pad(y, (0, pad_samples), mode='constant', constant_values=0)
            st.info(f"ℹ️ Audio is {duration:.1f}s. Padding to {TARGET_DURATION}s.")
        elif duration > TARGET_DURATION:
            # Too long: take first 30 seconds
            y = y[:target_samples]
            st.info(f"ℹ️ Audio is {duration:.1f}s (> {TARGET_DURATION}s). Using first {TARGET_DURATION}s for analysis.")
        else:
            st.success(f"✓ Audio duration is {duration:.1f}s (optimal)")
        
        # Generate mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def extract_embedding(encoder, spectrogram):
    """Extract embedding from spectrogram using the encoder"""
    try:
        x = pad_spectrogram(spectrogram)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = encoder(x_tensor).cpu().numpy()[0]
            # L2 normalization
            emb = emb / np.linalg.norm(emb)
        
        return emb
    except Exception as e:
        st.error(f"Error extracting embedding: {e}")
        return None

def recommend_songs(query_emb, embeddings_dict, info_df, top_k=10):
    """Recommend songs based on embedding similarity"""
    # Filter valid embeddings
    valid_track_ids = set(info_df['track_id'].astype(str).tolist())
    filtered_embeddings = {tid: emb for tid, emb in embeddings_dict.items() if tid in valid_track_ids}
    
    track_ids = list(filtered_embeddings.keys())
    emb_matrix = np.array([filtered_embeddings[tid] for tid in track_ids])
    
    # Calculate cosine similarity
    query_emb_normalized = query_emb / np.linalg.norm(query_emb)
    similarities = cosine_similarity([query_emb_normalized], emb_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    recommendations = []
    for idx in top_indices:
        tid = track_ids[idx]
        track_match = info_df[info_df['track_id'] == tid]
        
        if len(track_match) > 0:
            track_info = track_match.iloc[0]
            recommendations.append({
                'Track ID': tid,
                'Artist': track_info['artist_name'],
                'Title': track_info['track_title'],
                'Genre': track_info['track_genre_top'],
                'Similarity': f"{similarities[idx]:.4f}"
            })
    
    return recommendations

# Main app logic
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10)

# Load resources
with st.spinner("Loading model and data..."):
    encoder = load_model()
    info_df = load_metadata()
    embeddings_dict = load_embeddings()

# Load sample songs (outside spinner to show its own status)
sample_songs = load_sample_songs()

if encoder is None or info_df is None or embeddings_dict is None:
    st.error("Failed to load required resources. Please check the file paths and try again.")
    st.stop()

# Input method selection
st.subheader("Choose Input Method")
input_method = st.radio(
    "How would you like to provide audio?",
    ["📁 Upload Your Own Audio", "🎵 Select from Sample Songs"],
    horizontal=True
)

selected_audio_path = None
audio_file_display = None

if input_method == "📁 Upload Your Own Audio":
    # File uploader
    audio_file = st.file_uploader(
        "Upload Audio File", 
        type=["wav", "mp3", "flac"],
        help="Audio will be automatically adjusted to 30s for optimal matching"
    )
    
    if audio_file is not None:
        audio_file_display = audio_file
        st.session_state['audio_source'] = 'upload'
        st.session_state['audio_name'] = audio_file.name

else:  # Select from sample songs
    if sample_songs:
        selected_song = st.selectbox(
            "Choose a sample song:",
            options=["-- Select a song --"] + list(sample_songs.keys())
        )
        
        if selected_song != "-- Select a song --":
            selected_audio_path = sample_songs[selected_song]
            st.session_state['audio_source'] = 'sample'
            st.session_state['audio_name'] = selected_song
            st.success(f"Selected: {selected_song}")
    else:
        st.warning("No sample songs available in the 'data/music' directory.")

# Display audio player
if audio_file_display is not None:
    st.audio(audio_file_display, format=f"audio/{audio_file_display.type.split('/')[-1]}")
elif selected_audio_path is not None:
    st.audio(selected_audio_path)

# Process button
process_enabled = audio_file_display is not None or selected_audio_path is not None

if st.button("🔍 Get Recommendations", type="primary", disabled=not process_enabled):
    with st.spinner("Processing audio..."):
        # Determine which audio source to use
        if st.session_state.get('audio_source') == 'upload':
            log_mel_spec = audio_to_melspec(audio_file_display)
        else:  # sample
            log_mel_spec = audio_to_melspec(selected_audio_path)
        
        if log_mel_spec is not None:
            st.info(f"📊 Spectrogram shape: {log_mel_spec.shape} (expected ~1293 time frames for 30s audio)")
            
            # Extract embedding
            with st.spinner("Extracting audio features..."):
                query_embedding = extract_embedding(encoder, log_mel_spec)
            
            if query_embedding is not None:
                st.success(f"✓ Embedding extracted (dimension: {len(query_embedding)})")
                
                # Store embedding in session state
                st.session_state['query_embedding'] = query_embedding
                st.session_state['audio_filename'] = st.session_state.get('audio_name', 'audio')
                
                # Get recommendations
                with st.spinner("Finding similar songs..."):
                    recommendations = recommend_songs(
                        query_embedding, 
                        embeddings_dict, 
                        info_df, 
                        top_k=top_k
                    )
                
                st.session_state['recommendations'] = recommendations
    
# Display results if they exist in session state
if 'recommendations' in st.session_state and st.session_state['recommendations']:
    recommendations = st.session_state['recommendations']
    query_embedding = st.session_state.get('query_embedding')
    audio_filename = st.session_state.get('audio_filename', 'uploaded_audio')
    
    # Embedding visualization and download section
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 View Embedding Vector", use_container_width=True):
            st.session_state['show_embedding'] = not st.session_state.get('show_embedding', False)
    
    with col2:
        # Prepare embedding dict for download
        embedding_dict = {
            'filename': audio_filename,
            'embedding': query_embedding.tolist(),
            'embedding_dim': len(query_embedding),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Convert to JSON string for download
        import json
        json_str = json.dumps(embedding_dict, indent=2)
        
        st.download_button(
            label="⬇️ Download Embedding",
            data=json_str,
            file_name=f"embedding_{audio_filename.split('.')[0]}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Show embedding if button is clicked
    if st.session_state.get('show_embedding', False):
        st.subheader("🔢 Audio Embedding Vector")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Visualization", "📋 Raw Values", "📊 Statistics"])
        
        with tab1:
            # Bar chart visualization
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(range(len(query_embedding)), query_embedding, color='steelblue', alpha=0.7)
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Value')
            ax.set_title(f'Embedding Vector Visualization (64D)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with tab2:
            # Display as dataframe
            emb_df = pd.DataFrame({
                'Dimension': range(len(query_embedding)),
                'Value': query_embedding
            })
            st.dataframe(emb_df, use_container_width=True, height=300)
        
        with tab3:
            # Statistics
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Mean", f"{np.mean(query_embedding):.4f}")
            with col_b:
                st.metric("Std Dev", f"{np.std(query_embedding):.4f}")
            with col_c:
                st.metric("Min", f"{np.min(query_embedding):.4f}")
            with col_d:
                st.metric("Max", f"{np.max(query_embedding):.4f}")
            
            st.write("**L2 Norm:**", f"{np.linalg.norm(query_embedding):.4f}")
    
    st.markdown("---")
    
    # Recommendations section
    if recommendations:
        st.subheader(f"🎼 Top {len(recommendations)} Recommendations")
        
        # Display as dataframe
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(
            rec_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Display detailed cards
        st.subheader("Detailed Results")
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"#{i} - {rec['Title']} by {rec['Artist']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Track ID:** {rec['Track ID']}")
                    st.write(f"**Artist:** {rec['Artist']}")
                with col2:
                    st.write(f"**Genre:** {rec['Genre']}")
                    st.write(f"**Similarity Score:** {rec['Similarity']}")
    else:
        st.warning("No recommendations found.")
else:
    st.info("👆 Please upload an audio file to get started!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This system uses contrastive learning to extract audio embeddings "
    "and recommends similar songs based on cosine similarity."
)


