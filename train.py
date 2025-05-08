import pandas as pd
import random
from itertools import combinations
import spacy
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# SETTINGS
MAX_TRIPLETS = 5000
EPOCHS       = 3
BATCH_SIZE   = 16
MODEL_OUT    = "fine_tuned_triplet_model"

# 1) Load & clean raw essays
df = pd.read_csv("essays.csv")[['essay','authors']].dropna()
print(f"Loaded {len(df)} essays.")

# 2) spaCy pipeline for masking content words
nlp = spacy.load("en_core_web_sm")
CONTENT_POS = {"NOUN","VERB","PROPN","ADJ","ADV"}
def mask_content(text: str) -> str:
    doc = nlp(text)
    return " ".join(
        f"<{tok.pos_}>" if tok.pos_ in CONTENT_POS else tok.text
        for tok in doc
    )

# 3) Build triplets: (anchor, positive, negative)
by_author = {}
for _, row in df.iterrows():
    by_author.setdefault(row['authors'], []).append(row['essay'])

triplets = []
authors = list(by_author.keys())
for auth, essays in by_author.items():
    if len(essays) < 2:
        continue
    # all same-author combinations
    for a, p in combinations(essays, 2):
        # pick negative from different author
        neg_author = random.choice([x for x in authors if x != auth])
        n = random.choice(by_author[neg_author])
        triplets.append((a, p, n))
        if len(triplets) >= MAX_TRIPLETS:
            break
    if len(triplets) >= MAX_TRIPLETS:
        break

print(f"Built {len(triplets)} triplets.")

# 4) Wrap as InputExample with masked text
train_samples = [
    InputExample(
        texts=[mask_content(a), mask_content(p), mask_content(n)]
    ) for a, p, n in triplets
]

# 5) Load SBERT & TripletLoss
model = SentenceTransformer("all-MiniLM-L6-v2")
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.TripletLoss(model)

# 6) Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=100,
    output_path=MODEL_OUT,
    show_progress_bar=True
)

print(f"Triplet-trained model saved to '{MODEL_OUT}'")

