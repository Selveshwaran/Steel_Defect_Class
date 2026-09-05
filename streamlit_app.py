from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


MODEL_PATH = Path("neu_defect_compact_cnn.pth")

DEFECT_INFO = {
    "crazing": (
        "Crazing is a network of fine cracks on the steel surface. It often "
        "looks like thin irregular lines caused by surface stress, cooling, "
        "or rolling-related damage."
    ),
    "inclusion": (
        "Inclusion means unwanted non-metallic material is trapped in the "
        "steel. In images, it can appear as small dark spots, streaks, or "
        "embedded particles."
    ),
    "patches": (
        "Patches are irregular surface regions with different texture, tone, "
        "or roughness compared with the surrounding steel."
    ),
    "pitted_surface": (
        "Pitted surface defects are small holes or depressions on the steel "
        "surface. They are commonly linked to corrosion, contamination, or "
        "localized surface damage."
    ),
    "rolled-in_scale": (
        "Rolled-in scale happens when oxide scale is pressed into the steel "
        "surface during rolling. It can appear as rough, flaky, or embedded "
        "dark surface marks."
    ),
    "scratches": (
        "Scratches are long narrow marks or grooves on the steel surface, "
        "usually caused by mechanical contact, handling, or abrasion."
    ),
}

DEFECT_ALIASES = {
    "scratch": "scratches",
    "scratches": "scratches",
    "pit": "pitted_surface",
    "pitted": "pitted_surface",
    "pitted surface": "pitted_surface",
    "pitted_surface": "pitted_surface",
    "rolled in scale": "rolled-in_scale",
    "rolled-in scale": "rolled-in_scale",
    "rolled-in_scale": "rolled-in_scale",
    "scale": "rolled-in_scale",
}


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


def answer_question(prompt: str, class_names: list[str]) -> str:
    lower_prompt = prompt.lower().strip()

    if "class" in lower_prompt or "category" in lower_prompt:
        return "I can classify: " + ", ".join(class_names)

    for alias, class_name in DEFECT_ALIASES.items():
        if alias in lower_prompt and class_name in DEFECT_INFO:
            return f"**{class_name}**: {DEFECT_INFO[class_name]}"

    for class_name in class_names:
        readable_name = class_name.replace("_", " ").replace("-", " ")
        if class_name.lower() in lower_prompt or readable_name.lower() in lower_prompt:
            return f"**{class_name}**: {DEFECT_INFO.get(class_name, 'This is one of the trained defect categories.')}"

    if "confidence" in lower_prompt or "probability" in lower_prompt:
        prediction = st.session_state.get("last_prediction")
        if prediction is None:
            return "Upload an image first, then I can explain the confidence score."
        return (
            f"The latest prediction was **{prediction['class']}** with "
            f"**{prediction['confidence'] * 100:.2f}%** confidence."
        )

    if "what" in lower_prompt or "explain" in lower_prompt or "mean" in lower_prompt:
        return (
            "Ask about one of these defect types: "
            + ", ".join(class_names)
            + ". For example: what is scratch?"
        )

    return (
        "Upload a defect image for classification, or ask me about a defect "
        "type such as scratches, crazing, inclusion, patches, pitted surface, "
        "or rolled-in scale."
    )


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

if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None

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
    upload_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if upload_id != st.session_state.last_upload_id:
        st.session_state.last_upload_id = upload_id

        with st.chat_message("user"):
            st.image(image, caption="Uploaded image", use_container_width=True)

        predicted_class, confidence, ranked_predictions = predict_defect(image)
        response = (
            f"The predicted defect class is **{predicted_class}** "
            f"with **{confidence * 100:.2f}%** confidence."
        )
        st.session_state.last_prediction = {
            "class": predicted_class,
            "confidence": confidence,
        }

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

    answer = answer_question(prompt, class_names)

    add_message("assistant", answer)
    with st.chat_message("assistant"):
        st.write(answer)
