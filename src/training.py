import os
import json
import base64
import zlib
from collections import Counter
from io import BytesIO

import requests
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# CONFIG
# -----------------------------
RAW_JSON_DIR = "../data/raw_json"
RAW_LEVELS_DIR = "../data/raw_levels"
MODEL_FILE = "../models/gd_level_generator.pt"

EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 16
EPOCHS = 5
MAX_SEQ_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG_MODE = True
MAX_LEVELS = 2000
MAX_FOR_VOCAB = 2000
MAX_BATCHES = 100

# -----------------------------
# SONG CACHE
# -----------------------------
song_cache = {}

def fetch_song(song_id):
    if song_id in song_cache:
        return song_cache[song_id]
    url = f"https://www.newgrounds.com/audio/download/{song_id}"
    r = requests.get(url)
    if r.status_code == 200:
        song_bytes = r.content
        song_cache[song_id] = song_bytes
        return song_bytes
    else:
        print(f"[DEBUG] Failed to fetch song {song_id}, status {r.status_code}")
        return None

def extract_song_features(song_bytes):
    audio, sr = librosa.load(BytesIO(song_bytes), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features = np.concatenate([
        [tempo],
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1)
    ])
    return torch.tensor(features, dtype=torch.float)

# -----------------------------
# LEVEL STRING UTILITIES
# -----------------------------
def decode_level_string(level_string):
    try:
        compressed_bytes = base64.b64decode(level_string + "===")
        decompressed_bytes = zlib.decompress(compressed_bytes)
        return decompressed_bytes.decode("utf-8", errors="ignore")
    except Exception:
        if all(c.isprintable() or c in "\n\r\t" for c in level_string[:200]):
            return level_string
        return ""

def build_vocab(sequences, min_freq=1):
    counter = Counter(char for seq in sequences for char in seq)
    vocab = {c: i+1 for i, (c, cnt) in enumerate(counter.items()) if cnt >= min_freq}
    vocab["<PAD>"] = 0
    vocab["<SOS>"] = len(vocab)
    vocab["<EOS>"] = len(vocab)
    print(f"[DEBUG] Built vocab of size {len(vocab)}")
    return vocab

def encode_sequence(seq, vocab):
    return [vocab["<SOS>"]] + [vocab.get(c, 0) for c in seq] + [vocab["<EOS>"]]

def pad_sequences(sequences, max_len):
    padded = [seq + [0]*(max_len-len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)

# -----------------------------
# DATASET
# -----------------------------
class LevelDataset(Dataset):
    def __init__(self, raw_json_dir, raw_levels_dir, vocab):
        self.sequences = []
        self.song_features = []

        for i, filename in enumerate(os.listdir(raw_json_dir)):
            if not filename.endswith(".json"):
                continue
            if DEBUG_MODE and len(self.sequences) >= MAX_LEVELS:
                print(f"[DEBUG] Reached dataset limit of {MAX_LEVELS} levels")
                break

            level_id = filename.replace(".json", "")
            json_path = os.path.join(raw_json_dir, filename)
            gjl_path = os.path.join(raw_levels_dir, f"{level_id}.gjl")

            if not os.path.exists(gjl_path):
                print(f"[DEBUG] Missing .gjl for {level_id}, skipping")
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            song_id = None
            if isinstance(meta, dict):
                song_id = meta.get("songID") or meta.get("song_id")
            elif isinstance(meta, list) and len(meta) > 0 and isinstance(meta[0], dict):
                song_id = meta[0].get("songID") or meta[0].get("song_id")

            if not song_id:
                print(f"[DEBUG] No songID in {filename}, skipping")
                continue

            level_string = open(gjl_path, "r", encoding="utf-8").read().strip()
            decoded_level = decode_level_string(level_string)
            if not decoded_level:
                print(f"[DEBUG] Failed to decode {gjl_path}, skipping")
                continue

            # Fetch and process song features
            song_bytes = fetch_song(song_id)
            if song_bytes is None:
                continue
            features = extract_song_features(song_bytes)

            self.sequences.append(encode_sequence(decoded_level, vocab))
            self.song_features.append(features)

            if i % 500 == 0:
                print(f"[DEBUG] Processed {i} metadata files...")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.song_features[idx]

# -----------------------------
# MODEL
# -----------------------------
class SongAwareLevelGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, song_feat_dim):
        super().__init__()
        self.level_embedding = nn.Embedding(vocab_size, embed_dim)
        self.song_fc = nn.Linear(song_feat_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, level_seq, song_features):
        batch_size = level_seq.size(0)
        hidden_state = self.song_fc(song_features).unsqueeze(0)  # [1, batch, hidden]
        cell_state = torch.zeros_like(hidden_state)
        x = self.level_embedding(level_seq)
        out, _ = self.lstm(x, (hidden_state, cell_state))
        out = self.fc(out)
        return out

# -----------------------------
# TRAINING UTILITIES
# -----------------------------
def collate_fn(batch):
    seqs, song_feats = zip(*batch)
    seqs = pad_sequences(seqs, MAX_SEQ_LEN)
    song_feats = torch.stack(song_feats)
    return seqs.to(DEVICE), song_feats.to(DEVICE)

# -----------------------------
# TRAINING LOOP
# -----------------------------
def train_and_save():
    # Build vocab from raw levels
    all_decoded = []
    for i, filename in enumerate(os.listdir(RAW_LEVELS_DIR)):
        if not filename.endswith(".gjl"):
            continue
        if DEBUG_MODE and len(all_decoded) >= MAX_FOR_VOCAB:
            print(f"[DEBUG] Reached vocab limit of {MAX_FOR_VOCAB}")
            break
        path = os.path.join(RAW_LEVELS_DIR, filename)
        decoded = decode_level_string(open(path, "r", encoding="utf-8").read().strip())
        if decoded:
            all_decoded.append(decoded)
        if i % 1000 == 0:
            print(f"[DEBUG] Processed {i} .gjl files for vocab")

    vocab = build_vocab(all_decoded)

    dataset = LevelDataset(RAW_JSON_DIR, RAW_LEVELS_DIR, vocab)
    print(f"[DEBUG] Dataset loaded: {len(dataset)} levels")

    if len(dataset) == 0:
        print("[DEBUG] No valid levels found, saving vocab-only checkpoint.")
        torch.save({
            "vocab": vocab,
            "EMBED_DIM": EMBED_DIM,
            "HIDDEN_DIM": HIDDEN_DIM,
            "MAX_SEQ_LEN": MAX_SEQ_LEN
        }, "gd_level_generator_vocab_only.pt")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Determine song feature size
    song_feat_dim = dataset.song_features[0].shape[0]

    model = SongAwareLevelGenerator(len(vocab), EMBED_DIM, HIDDEN_DIM, song_feat_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (level_seq, song_feat) in enumerate(dataloader):
            if DEBUG_MODE and batch_idx >= MAX_BATCHES:
                print(f"[DEBUG] Stopping after {MAX_BATCHES} batches (debug mode)")
                break

            optimizer.zero_grad()
            output = model(level_seq[:, :-1], song_feat)
            loss = loss_fn(output.reshape(-1, output.size(-1)), level_seq[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, batch_idx+1)
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

        # Save checkpoint each epoch
        save_path = f"{MODEL_FILE.replace('.pt','')}_epoch{epoch+1}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab": vocab,
            "EMBED_DIM": EMBED_DIM,
            "HIDDEN_DIM": HIDDEN_DIM,
            "MAX_SEQ_LEN": MAX_SEQ_LEN
        }, save_path)
        print(f"[DEBUG] Saved checkpoint {save_path}")

# -----------------------------
# GENERATION
# -----------------------------
def generate_level(song_id, checkpoint_file, max_len=MAX_SEQ_LEN):
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    vocab = checkpoint["vocab"]
    idx_to_char = {i: c for c, i in vocab.items()}

    song_bytes = fetch_song(song_id)
    if song_bytes is None:
        raise ValueError("Failed to fetch song for generation")
    song_features = extract_song_features(song_bytes).unsqueeze(0).to(DEVICE)

    song_feat_dim = song_features.shape[1]
    model = SongAwareLevelGenerator(len(vocab), checkpoint["EMBED_DIM"], checkpoint["HIDDEN_DIM"], song_feat_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    seq = [vocab["<SOS>"]]
    generated = []

    for step in range(max_len):
        inp = torch.tensor([seq], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = model(inp, song_features)
        next_token = logits.argmax(-1)[0, -1].item()
        if next_token == vocab["<EOS>"]:
            break
        generated.append(next_token)
        seq.append(next_token)

    level_string = "".join(idx_to_char[i] for i in generated if i in idx_to_char)
    return level_string

# -----------------------------
# ENTRY POINT
# -----------------------------
train_and_save()
