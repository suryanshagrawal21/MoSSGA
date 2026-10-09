# MoSSGA — Deployment Guide

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy model (one-time)
python -m spacy download en_core_web_sm

# 3. (Optional) Create .env file for GitHub token
copy .env.example .env
# Edit .env and add your GITHUB_TOKEN

# 4. Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Option 1: Streamlit Cloud (Recommended — Free & Easiest)

**Best for:** Quick sharing, demos, public apps.

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) ? Sign in with GitHub
3. Click **New app** ? Select your repo ? Set **Main file path** to `app.py`
4. (Optional) Under **Advanced settings ? Secrets**, add:
   ```
   GITHUB_TOKEN = "your_github_personal_access_token"
   ```
5. Click **Deploy** — done! ?

> **Note:** Streamlit Cloud automatically runs `pip install -r requirements.txt`.
> The spaCy model is downloaded at first startup via the fallback in `preprocessing.py`.

---

## Option 2: Render.com (Free tier, full control)

**Best for:** Production APIs, always-on apps.

1. Go to [render.com](https://render.com) ? New ? **Web Service**
2. Connect your GitHub repository
3. Configure:
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Under **Environment Variables**, add (optional):
   - `GITHUB_TOKEN` = your token
   - `HF_HUB_DISABLE_SYMLINKS_WARNING` = `1`
5. Click **Create Web Service** ?

---

## Option 3: Railway.app

1. Go to [railway.app](https://railway.app) ? New Project ? Deploy from GitHub
2. Add environment variables:
   - `GITHUB_TOKEN` (optional)
3. Railway auto-detects the `Procfile` — done! ?

---

## Option 4: Heroku

1. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Run:
   ```bash
   heroku create mossga-app
   git push heroku main
   ```
3. Add buildpacks if needed:
   ```bash
   heroku buildpacks:add heroku/python
   ```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GITHUB_TOKEN` | Optional | GitHub Personal Access Token — raises API limit from 60 to 5000 req/hr |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | Optional | Set to `1` to suppress HuggingFace symlinks warning on Windows |

---

## Notes

- The **SBERT model** (`all-MiniLM-L6-v2`, ~90MB) is downloaded automatically on first run and cached.
- The **XGBoost ML model** (`data/ml_model.pkl`) is auto-generated at first startup with synthetic training data.
- The `spaCy` model (`en_core_web_sm`) must be explicitly downloaded via `python -m spacy download en_core_web_sm`.
