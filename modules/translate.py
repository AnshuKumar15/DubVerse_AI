from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json

MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def translate_text(text):
    batch = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    generated = model.generate(**batch, max_length=512)

    output = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return output[0]

def translate_transcript(input_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    translated_segments = []

    for segment in data["segments"]:
        english_text = segment["text"].strip()

        hindi_text = translate_text(english_text)

        translated_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "english": english_text,
            "hindi": hindi_text
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"segments": translated_segments}, f, indent=4, ensure_ascii=False)

    print("Translation Complete.")