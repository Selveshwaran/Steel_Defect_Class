from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


MODEL_PATH = Path("neu_defect_compact_cnn.pth")


class CompactDefectCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 192),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@st.cache_resource
def load_model() -> tuple[nn.Module, list[str], int]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Add your trained .pth file "
            "to the same GitHub folder as streamlit_app.py."
        )

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 128)

    model = CompactDefectCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, image_size


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict_defect(image: Image.Image) -> tuple[str, float, list[tuple[str, float]]]:
    model, class_names, image_size = load_model()
    image_tensor = preprocess_image(image, image_size)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    confidence, prediction_idx = torch.max(probabilities, dim=0)

    ranked = sorted(
        zip(class_names, probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return class_names[prediction_idx.item()], confidence.item(), ranked


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


st.set_page_config(page_title="Steel Defect Chatbot", page_icon="🔍", layout="centered")
st.title("Steel Defect Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Upload a steel surface image and I will classify the defect type."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

try:
    model, class_names, _ = load_model()
    st.caption("Available classes: " + ", ".join(class_names))
except Exception as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload defect image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.chat_message("user"):
        st.image(image, caption="Uploaded image", use_container_width=True)

    predicted_class, confidence, ranked_predictions = predict_defect(image)
    response = (
        f"The predicted defect class is **{predicted_class}** "
        f"with **{confidence * 100:.2f}%** confidence."
    )

    add_message("user", f"Uploaded image: {uploaded_file.name}")
    add_message("assistant", response)

    with st.chat_message("assistant"):
        st.write(response)
        st.write("Top probabilities:")
        for class_name, probability in ranked_predictions:
            st.progress(probability, text=f"{class_name}: {probability * 100:.2f}%")

prompt = st.chat_input("Ask about the prediction or type 'classes'")

if prompt:
    add_message("user", prompt)
    with st.chat_message("user"):
        st.write(prompt)

    lower_prompt = prompt.lower().strip()
    if "class" in lower_prompt:
        answer = "I can classify: " + ", ".join(class_names)
    else:
        answer = (
            "Upload a defect image above. I will return the predicted class, "
            "confidence score, and probabilities for each defect category."
        )

    add_message("assistant", answer)
    with st.chat_message("assistant"):
        st.write(answer)
