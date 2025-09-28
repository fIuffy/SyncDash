import os
import json
import base64
import zlib
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# CONFIG
# -----------------------------
RAW_JSON_DIR = "../data/raw_json"
RAW_LEVELS_DIR = "../data/raw_levels"
MODEL_FILE = "gd_level_generator.pt"

EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 16
EPOCHS = 5        # change for bigger training
MAX_SEQ_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG_MODE = True   # toggle to False for full training
MAX_LEVELS = 2000   # limit dataset size in debug mode
MAX_FOR_VOCAB = 2000
MAX_BATCHES = 100

# -----------------------------
# UTILITIES
# -----------------------------
def decode_level_string(level_string):
    """Try to decode a GD level string (compressed or plain)."""
    try:
        # Try compressed Base64 first
        compressed_bytes = base64.b64decode(level_string + "===")  # padding fix
        decompressed_bytes = zlib.decompress(compressed_bytes)
        return decompressed_bytes.decode("utf-8", errors="ignore")
    except Exception:
        # Fallback: if it looks like text, just return
        if all(c.isprintable() or c in "\n\r\t" for c in level_string[:200]):
            return level_string
        return ""  # skip

def build_vocab(sequences, min_freq=1):
    counter = Counter(char for seq in sequences for char in seq)
    vocab = {c: i+1 for i, (c, cnt) in enumerate(counter.items()) if cnt >= min_freq}  # 0 = PAD
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
        self.song_ids = []
        self.unique_songs = set()

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
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            song_id = None
            if isinstance(meta, dict):
                # Try both common key variants
                song_id = meta.get("songID") or meta.get("song_id")
            elif isinstance(meta, list) and len(meta) > 0 and isinstance(meta[0], dict):
                song_id = meta[0].get("songID") or meta[0].get("song_id")

            if not song_id:
                print(f"[DEBUG] No songID in {filename}, skipping")
                continue

            with open(gjl_path, "r", encoding="utf-8") as f:
                level_string = f.read().strip()

            decoded_level = decode_level_string(level_string)
            if not decoded_level:
                continue

            self.sequences.append(encode_sequence(decoded_level, vocab))
            self.song_ids.append(song_id)
            self.unique_songs.add(song_id)

            if i % 500 == 0:
                print(f"[DEBUG] Processed {i} metadata files...")

        self.unique_songs = sorted(list(self.unique_songs))
        self.song_to_idx = {song: idx for idx, song in enumerate(self.unique_songs)}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.song_to_idx[self.song_ids[idx]]

# -----------------------------
# MODEL
# -----------------------------
class SongToLevel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_songs):
        super().__init__()
        self.song_embedding = nn.Embedding(num_songs, hidden_dim)
        self.level_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, level_seq, song_idx):
        hidden = self.song_embedding(song_idx).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        x = self.level_embedding(level_seq)
        out, _ = self.lstm(x, (hidden, cell))
        out = self.fc(out)
        return out

# -----------------------------
# TRAINING
# -----------------------------
def collate_fn(batch):
    seqs, songs = zip(*batch)
    seqs = pad_sequences(seqs, MAX_SEQ_LEN)
    songs = torch.tensor(songs, dtype=torch.long)
    return seqs.to(DEVICE), songs.to(DEVICE)

def train_and_save():
    # Build vocab
    all_decoded = []
    for i, filename in enumerate(os.listdir(RAW_LEVELS_DIR)):
        if not filename.endswith(".gjl"):
            continue
        if DEBUG_MODE and len(all_decoded) >= MAX_FOR_VOCAB:
            print(f"[DEBUG] Reached vocab limit of {MAX_FOR_VOCAB}")
            break
        path = os.path.join(RAW_LEVELS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            decoded = decode_level_string(f.read().strip())
            if decoded:
                all_decoded.append(decoded)
        if i % 1000 == 0:
            print(f"[DEBUG] Processed {i} .gjl files for vocab")

    vocab = build_vocab(all_decoded)

    dataset = LevelDataset(RAW_JSON_DIR, RAW_LEVELS_DIR, vocab)
    print(f"[DEBUG] Dataset loaded: {len(dataset)} levels, {len(dataset.unique_songs)} unique songs")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = SongToLevel(len(vocab), EMBED_DIM, HIDDEN_DIM, len(dataset.unique_songs)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (level_seq, song_idx) in enumerate(dataloader):
            if DEBUG_MODE and batch_idx >= MAX_BATCHES:
                print(f"[DEBUG] Stopping after {MAX_BATCHES} batches (debug mode)")
                break

            optimizer.zero_grad()
            output = model(level_seq[:, :-1], song_idx)
            loss = loss_fn(output.reshape(-1, output.size(-1)), level_seq[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, batch_idx+1)
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

        # ✅ SAVE CHECKPOINT AFTER EACH EPOCH
        save_path = f"{MODEL_FILE.replace('.pt', '')}_epoch{epoch+1}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab": vocab,
            "song_to_idx": dataset.song_to_idx,
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
    song_to_idx = checkpoint["song_to_idx"]
    idx_to_char = {i: c for c, i in vocab.items()}

    model = SongToLevel(len(vocab), checkpoint["EMBED_DIM"], checkpoint["HIDDEN_DIM"], len(song_to_idx)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if song_id not in song_to_idx:
        raise ValueError("Song ID not in training dataset.")

    song_idx = torch.tensor([song_to_idx[song_id]], dtype=torch.long).to(DEVICE)
    seq = [vocab["<SOS>"]]
    generated = []

    print(f"[DEBUG] Generating level for song {song_id}")

    for step in range(max_len):
        inp = torch.tensor([seq], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = model(inp, song_idx)
        next_token = logits.argmax(-1)[0, -1].item()
        if next_token == vocab["<EOS>"]:
            print(f"[DEBUG] Reached <EOS> at step {step}")
            break
        generated.append(next_token)
        seq.append(next_token)

    level_string = "".join(idx_to_char[i] for i in generated if i in idx_to_char)
    print(f"[DEBUG] Generated sequence length: {len(generated)}")
    return level_string

# -----------------------------
# MAIN
# -----------------------------
train_and_save()

# Example generation using last checkpoint
example_song_id = 63082  # must exist in dataset
checkpoint_file = f"{MODEL_FILE.replace('.pt', '')}_epoch{EPOCHS}.pt"
try:
    new_level = generate_level(example_song_id, checkpoint_file)
    print("Generated Level String:", new_level[:300], "...")
except Exception as e:
    print(f"[DEBUG] Generation failed: {e}")
