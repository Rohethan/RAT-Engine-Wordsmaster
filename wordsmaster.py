#!/usr/bin/env python3

import os
import sys
import sqlite3
import argparse
import tarfile
import shutil
import urllib.request
import wave
import csv
import numpy as np
import librosa
import yaml
from pathlib import Path
from tqdm import tqdm
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Paths
BASE_DIR = Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DATASET_DIR = DATA_DIR / "LJSpeech-1.1"
WAV_DIR = DATASET_DIR / "wavs"
DATASET_ARCHIVE = DATA_DIR / "LJSpeech-1.1.tar.bz2"
METADATA_PATH = DATASET_DIR / "metadata.csv"
DB_PATH = OUTPUT_DIR / "ljspeech_data.db"
TOKENIZER_PATH = OUTPUT_DIR / "universal_tokenizer.json"
CONFIG_PATH = BASE_DIR / "wordsmaster.conf"

# URLs
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RAT-Engine Wordsmaster: Dataset and tokenizer generator")
    parser.add_argument("--cache", action="store_true", help="Use cached LJSpeech dataset if available")
    return parser.parse_args()

def ensure_directories():
    """Ensure necessary directories exist."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

def load_config():
    """Load configuration from wordsmaster.conf file.
    If the file doesn't exist, create it with default values.
    """
    if not CONFIG_PATH.exists():
        # Create default configuration
        default_config = {
            "audio": {
                "original_sample_rate": 22050,
                "target_sample_rate": 7350
            },
            "tokenizer": {
                "vocab_size": 1000,
                "min_frequency": 2,
                "special_tokens": ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
            },
            "output": {
                "database": "ljspeech_data.db",
                "tokenizer": "universal_tokenizer.json"
            }
        }

        # Save default configuration
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        return default_config

    # Load configuration
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    return config

def download_ljspeech(use_cache=False):
    """Download the LJSpeech dataset if it's not already available."""
    if use_cache and DATASET_DIR.exists():
        print(f"Using cached LJSpeech dataset at {DATASET_DIR}")
        return

    if not DATASET_ARCHIVE.exists():
        print(f"Downloading LJSpeech dataset from {LJSPEECH_URL}")

        # Implement a download with progress bar
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        # Download with progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc="LJSpeech dataset") as t:
            urllib.request.urlretrieve(LJSPEECH_URL, DATASET_ARCHIVE, 
                                       reporthook=t.update_to)
        print("Download complete")
    else:
        print(f"Using existing download at {DATASET_ARCHIVE}")

    if not DATASET_DIR.exists():
        print(f"Uncompressing dataset to {DATASET_DIR}")
        # Add extraction progress
        with tarfile.open(DATASET_ARCHIVE, 'r:bz2') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc="Extracting files", unit="files"):
                tar_ref.extract(member, path=DATA_DIR)
        print("Extraction complete")

def create_database():
    """Create the SQLite database with the required schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audio_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recording_id TEXT NOT NULL UNIQUE,
        data_22050 BLOB NOT NULL,
        data_7350 BLOB NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recording_id TEXT NOT NULL,
        sentence TEXT NOT NULL,
        duration REAL
    )
    ''')

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def process_audio_data():
    """Process audio files and store them in the database."""
    print("Processing audio files...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Load configuration
    config = load_config()
    original_sr = config['audio']['original_sample_rate']
    target_sr = config['audio']['target_sample_rate']

    print(f"Using original sample rate: {original_sr}Hz and target sample rate: {target_sr}Hz")

    # Function to convert numpy array to blob
    def numpy_to_blob(arr):
        return sqlite3.Binary(arr.tobytes())

    file_count = 0
    skipped_count = 0

    # Create a progress bar for audio processing
    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    for filename in tqdm(wav_files, desc="Processing audio files", unit="files"):
        file_path = os.path.join(WAV_DIR, filename)
        recording_id = os.path.splitext(filename)[0]

        # Check if this recording_id is already in the database
        cursor.execute("SELECT 1 FROM audio_data WHERE recording_id = ?", (recording_id,))
        if cursor.fetchone():
            skipped_count += 1
            continue

        try:
            # Load audio with original sample rate
            y_original, sr = librosa.load(file_path, sr=original_sr, mono=True)

            # Resample to target sample rate
            y_target = librosa.resample(y_original, orig_sr=original_sr, target_sr=target_sr)

            # Convert the numpy arrays to blobs
            audio_blob_original = numpy_to_blob(y_original)
            audio_blob_target = numpy_to_blob(y_target)

            # Insert the data into the database
            cursor.execute(
                "INSERT INTO audio_data (recording_id, data_22050, data_7350) VALUES (?, ?, ?)",
                (recording_id, audio_blob_original, audio_blob_target)
            )

            file_count += 1

            if file_count % 10 == 0:  # Commit periodically to avoid losing progress
                conn.commit()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Commit final changes
    conn.commit()

    # Print some stats
    cursor.execute("SELECT COUNT(*) FROM audio_data")
    data_entries = cursor.fetchone()[0]

    print(f"\nTotal WAV files processed: {file_count}")
    print(f"Files skipped (already in database): {skipped_count}")
    print(f"Total database entries with dual-rate audio data: {data_entries}")
    print(f"Audio stored at both {original_sr}Hz and {target_sr}Hz sample rates")

    conn.close()

def process_metadata():
    """Process metadata and store text data in the database."""
    print("Processing metadata...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Read and process the metadata file
    with open(METADATA_PATH, 'r', encoding='utf-8') as file:
        # Define the delimiter for the CSV
        csv_reader = csv.reader(file, delimiter='|')

        entry_count = 0
        # Process each line
        for row in csv_reader:
            if len(row) >= 3:
                recording_id = row[0].strip()
                sentence_with_numbers = row[1].strip()
                sentence_with_written_numbers = row[2].strip()

                # Insert both versions into the database
                cursor.execute(
                    "INSERT INTO metadata (recording_id, sentence) VALUES (?, ?)",
                    (recording_id, sentence_with_numbers)
                )
                cursor.execute(
                    "INSERT INTO metadata (recording_id, sentence) VALUES (?, ?)",
                    (recording_id, sentence_with_written_numbers)
                )
                entry_count += 2

                if entry_count % 100 == 0:
                    print(f"Processed {entry_count} metadata entries")

    # Commit the changes
    conn.commit()

    # Print stats
    cursor.execute("SELECT COUNT(*) FROM metadata")
    total_entries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT recording_id) FROM metadata")
    unique_recordings = cursor.fetchone()[0]

    print(f"Total metadata entries: {total_entries}")
    print(f"Unique recordings: {unique_recordings}")
    print(f"Entries per recording: {total_entries / unique_recordings if unique_recordings > 0 else 0}")

    conn.close()

def update_audio_durations():
    """Calculate and update the duration of each audio file in the metadata."""
    print("Updating audio durations...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    file_count = 0
    for filename in os.listdir(WAV_DIR):
        if filename.endswith(".wav"):
            file_path = os.path.join(WAV_DIR, filename)
            recording_id = os.path.splitext(filename)[0]

            try:
                with wave.open(file_path, 'rb') as wf:
                    # Duration = frames / framerate
                    duration = wf.getnframes() / wf.getframerate()

                    # Update the database with the duration
                    cursor.execute(
                        "UPDATE metadata SET duration = ? WHERE recording_id = ?",
                        (duration, recording_id)
                    )
                    file_count += 1

                    if file_count % 100 == 0:
                        print(f"Processed {file_count} duration updates")
            except Exception as e:
                print(f"Error processing duration for {filename}: {e}")

    # Commit the changes
    conn.commit()

    # Print stats
    cursor.execute("SELECT COUNT(*) FROM metadata WHERE duration IS NOT NULL")
    updated_entries = cursor.fetchone()[0]

    print(f"Total files processed for duration: {file_count}")
    print(f"Database entries updated with duration: {updated_entries}")

    conn.close()

def train_tokenizer():
    """Build and train the universal tokenizer."""
    print("Training universal tokenizer...")

    # Load configuration
    config = load_config()
    vocab_size = config['tokenizer']['vocab_size']
    min_frequency = config['tokenizer']['min_frequency']
    special_tokens = config['tokenizer']['special_tokens']

    print(f"Using tokenizer vocabulary size: {vocab_size}, min frequency: {min_frequency}")
    print(f"Special tokens: {', '.join(special_tokens)}")

    # Get all sentences from the database to train the tokenizer
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT sentence FROM metadata")
    sentences = [row[0] for row in cursor.fetchall()]

    conn.close()

    # Create temporary corpus file
    corpus_file = OUTPUT_DIR / "corpus.txt"
    print(f"Creating corpus file with {len(sentences)} sentences")
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Set up the trainer
    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    # Train the tokenizer
    print("Training tokenizer (this may take some time)...")
    tokenizer.train(files=[str(corpus_file)], trainer=trainer)

    # Save the tokenizer
    tokenizer.save(str(TOKENIZER_PATH))

    # Clean up the corpus file
    corpus_file.unlink()

    print(f"Tokenizer trained and saved to {TOKENIZER_PATH}")

def cleanup(use_cache=False):
    """Clean up the dataset files to free space."""
    if not use_cache:
        print("Cleaning up dataset files...")
        if DATASET_ARCHIVE.exists():
            DATASET_ARCHIVE.unlink()
            print(f"Deleted {DATASET_ARCHIVE}")

        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
            print(f"Deleted {DATASET_DIR}")
    else:
        print("Skipping cleanup as --cache flag is set")

def main():
    """Main function to run all the steps."""
    # Parse command-line arguments
    args = parse_args()

    print("\n==== RAT-Engine Wordsmaster ====\n")
    print("Starting the dataset and tokenizer generation process...\n")

    # Step 1: Ensure directories exist
    ensure_directories()

    # Step 2: Load configuration
    config = load_config()
    print(f"Configuration loaded from {CONFIG_PATH}")

    # Step 2: Download and uncompress the LJSpeech dataset
    print("\n-- Step 1: Download and prepare LJSpeech dataset --")
    download_ljspeech(args.cache)

    # Step 3: Create the SQLite database
    print("\n-- Step 2: Create SQLite database --")
    create_database()

    # Step 4: Process dataset's audio and store in DB
    print("\n-- Step 3: Process audio data --")
    process_audio_data()

    # Step 5: Process dataset's text and store in DB
    print("\n-- Step 4: Process metadata --")
    process_metadata()

    # Step 6: Update audio durations
    print("\n-- Step 5: Update audio durations --")
    update_audio_durations()

    # Step 7: Build and train the universal tokenizer
    print("\n-- Step 6: Build and train tokenizer --")
    train_tokenizer()

    # Step 8: Clean up dataset files
    print("\n-- Step 7: Clean up --")
    cleanup(args.cache)

    print("\n==== Process Complete ====\n")
    print(f"Universal tokenizer saved at: {TOKENIZER_PATH}")
    print(f"SQLite database saved at: {DB_PATH}")
    print("\nRAT-Engine Wordsmaster has completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)