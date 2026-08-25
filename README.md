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
