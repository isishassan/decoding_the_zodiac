# 🔮 Decoding the Zodiac with Machine Learning

**Ironhack Data Science & Machine Learning Bootcamp — Final Project**
**Isis Hassan | March 2026**

Can a machine learning model tell Aries from Pisces just by reading their horoscope? This project investigates that question — and the answer turns out to say more about humans than the stars.

---

## Project Overview

Daily horoscopes are a media staple deliberately written to be generic, so anyone can see themselves in them. This project uses that paradox as its foundation: across three objectives, it attempts to classify, analyse, and generate horoscope text using a full NLP and ML pipeline.

**Objective 1 — Classification:** Can a model predict the zodiac sign from horoscope text alone? (Baseline: 8.3% — random chance across 12 signs)

**Objective 2 — Trends & Themes:** Do patterns emerge across signs, elements, or time? Using sentiment analysis and semantic theme detection.

**Objective 3 — Generation:** Use a local LLM (LLaMA 3.2 via Ollama) to generate new daily horoscopes inspired by the tone and style of real ones.

---

## Key Results

| Task | Approach | Result |
|------|----------|--------|
| Classification | Soft Voting Ensemble (Logistic Regression + Linear SVM) | **14% test accuracy** — beats random chance (8.3%) |
| Sentiment Analysis | `twitter-roberta-base-sentiment-latest` transformer | Horoscopes are more positive during your birthday month; positivity trends decline across the year |
| Theme Detection | Sentence Transformers (`all-MiniLM-L6-v2`) | Theme distributions are near-identical across signs, confirming deliberate genericism |
| Horoscope Generation | LLaMA 3.2 (local, via Ollama) with few-shot prompting | Full year of synthetic horoscopes generated for all 12 signs |

**Takeaway:** Daily horoscopes are engineered to be universal. Any signal we extract tells us more about how humans write than about astrology.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data processing | `pandas`, `numpy` |
| NLP | `scikit-learn` (TF-IDF), `NLTK`, `wordcloud` |
| ML models | Logistic Regression, Linear SVM, Naive Bayes, Random Forest, Soft Voting Ensemble |
| Hyperparameter tuning | `GridSearchCV` |
| Transformers | `transformers` (HuggingFace), `sentence-transformers` |
| Dimensionality reduction | T-SNE |
| LLM / generation | Ollama (LLaMA 3.2), local inference |
| Visualisation | `matplotlib`, `seaborn` |
| Frontend | HTML/CSS (interactive horoscope picker) |

---

## Repository Structure

```
FinalProject/
│
├── data/                        # Not tracked in git (see Data Sources below)
│   ├── data sources/
│   │   └── kaggle_source_1/     # Raw CSVs from Kaggle
│   ├── hindustan_times.csv
│   └── horoscope_com.csv
│
├── data_collection.ipynb        # Data ingestion & standardisation
├── eda.ipynb                    # Exploratory data analysis (TF-IDF, T-SNE, word clouds)
├── classification.ipynb         # ML pipeline, model comparison, ensemble
├── sentiment_analysis.ipynb     # Sentiment and Theme exploration
├── text_generation.ipynb        # Horoscope generation
├── index.html                   # Interactive horoscope viewer (load your CSV)
│
├── requirements.txt
└── README.md
```

---

## Notebooks

**`data_collection.ipynb`** — Ingests and standardises three Kaggle datasets (16,701 horoscopes total). Handles mistyped sign names, date format alignment, column restructuring, and sign anonymisation (replacing sign mentions with `[Sign Name]`).

**`eda.ipynb`** — Exploratory analysis using TF-IDF with bigrams, word clouds, T-SNE clustering, distinctive word heatmaps, transformer-based sentiment analysis, and semantic theme detection across signs and time.

**`classification.ipynb`** — Full ML pipeline: TF-IDF vectorisation → model comparison (Naive Bayes, Logistic Regression, Linear SVM, Random Forest) → GridSearchCV tuning → Soft Voting Ensemble. Includes confusion matrices and iterative debugging of overfitting.

---

## Data Sources

Data is excluded from this repository (see `.gitignore`). All source datasets are publicly available on Kaggle:

- [`adxie12/horoscopes`](https://www.kaggle.com/datasets/adxie12/horoscopes) — Globe horoscopes (scraped + 2025)
- [`prasad22/daily-horoscope-dataset`](https://www.kaggle.com/datasets/prasad22/daily-horoscope-dataset) — Hindustan Times
- [`shahp7575/horoscopes`](https://www.kaggle.com/datasets/shahp7575/horoscopes) — horoscope.com

---

## Setup & Installation

**Prerequisites:** Python 3.9+, virtual environment, [Ollama](https://ollama.com) installed locally (for generation only)

```bash
# Clone the repo
git clone https://github.com/your-username/FinalProject.git
cd FinalProject

# Create and activate virtual environment
python -m venv .env
source .env/bin/activate      # Mac/Linux
.env\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# For horoscope generation only — pull the model via Ollama
ollama pull llama3.2
```

Then download the Kaggle datasets and place them in `data/data sources/kaggle_source_1/`.

Run notebooks in order: `data_collection` → `eda` → `classification` → `sentiment`→ `text_generation`.

---

## What I'd Do Next

1. **Transformer-based classification** — swap TF-IDF for BERT embeddings to capture richer semantic signal
2. **Temporal theme analysis** — explore weekend vs. weekday horoscope patterns
3. **Sign-aware generation** — prompt the LLM with sign personality traits to produce more differentiated output

---

## Presentation

The project slide deck (`Decoding_The_Zodiac.pdf`) is included in this repository.

---

*Project completed as part of the Ironhack Data Science & Machine Learning Bootcamp, March 2026.*
