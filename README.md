# Steel Defect Streamlit Chatbot

This folder contains the Streamlit inference app for the trained NEU defect classifier.

## Files needed

Your GitHub repository should contain:

```text
streamlit_app.py
requirements.txt
neu_defect_compact_cnn.pth
```

The `neu_defect_compact_cnn.pth` file is created by the PyTorch training script.
Do not upload the full training dataset to Streamlit Cloud for inference.

## OpenAI chatbot setup

The PyTorch model performs image classification. The OpenAI API is used only
for natural-language chatbot answers after a prediction is available.

Do not commit your API key to GitHub. Add it in Streamlit Cloud under:

```text
App settings -> Secrets
```

Use this format:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Optional:

```toml
OPENAI_MODEL = "gpt-5-mini"
```

If `OPENAI_API_KEY` is missing, the app still runs, but it falls back to simple
rule-based answers.

## Local test

From this folder, run:

```powershell
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Add `streamlit_app.py`, `requirements.txt`, and `neu_defect_compact_cnn.pth`.
3. Push the files to GitHub.
4. Go to `https://share.streamlit.io`.
5. Click `Create app`.
6. Select your repository, branch, and `streamlit_app.py` as the entrypoint file.
7. Deploy.

If your model file is too large for normal GitHub upload, use Git LFS.
