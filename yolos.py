from transformers import YolosConfig, YolosModel, AutoImageProcessor, AutoModelForObjectDetection
import torch
from datasets import load_dataset
from PIL import Image
import requests
from matplotlib import pyplot as plt
import numpy as np

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
model = YolosModel.from_pretrained("hustvl/yolos-small")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjE9HwPyzcyWpvHhyrd_rsu8uKyVcntDzb2jmqm25EhoKfwavVOo9izUXJKeECaILK--g&usqp=CAU"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    0
]

fig, ax = plt.subplots(1)
ax.imshow(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    label_name = model.config.id2label[label.item()]
    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2))
    ax.text(box[0], box[1], f"{label_name} ({round(score.item(), 3)})", fontsize=8, color='white', bbox=dict(facecolor='red', alpha=0.8))
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

plt.show()