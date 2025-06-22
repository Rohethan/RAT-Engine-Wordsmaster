# RAT-Engine-Wordsmaster

This project is part of a bigger project : RAT-Engine (Rohethan AudioText Engine).

## Wordsmaster
Wordsmaster is the project part responsible of generating the dataset db, and the tokenizer. 

Running the main script will do the following :

1. Download and uncompress the LJSpeech dataset
2. Create a SQLite database with the correct fields
3. Process the dataset's audio, text, metadata and store it in the DB.
4. Cleanup LJSpeech dataset file, clean up some space.
5. Build and train the universal tokenizer


Once the Wordsmaster has ran, you'll have in the `output` directory:

- Universal tokenizer (`universal_tokenizer.json`)
- Single SQLite file holding all the data (`ljspeech_data.db`).


### Flags
You have the following flags availible to you :

- `--cache` : Used a cached LJSpeech dataset. Won't download if it's already there, won't delete either.
