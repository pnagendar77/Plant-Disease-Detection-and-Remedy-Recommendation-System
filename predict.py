import torch
from torchvision import transforms
from PIL import Image
from langchain_groq import ChatGroq
import json

class PlantDiseaseClassifier:
    def __init__(self, model_path, class_names, groq_api_key):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.model = torch.hub.load('pytorch/vision:v0.15.1', 'resnet50', pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.class_names = class_names
        self.llm = ChatGroq(
            temperature=0.3,
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_class = self.class_names[predicted.item()]

        return pred_class

    def get_remedy(self, disease_name):
        prompt = f"The detected plant disease is '{disease_name}'. Please provide remedies and treatments for this disease."
        response = self.llm.invoke(prompt)
        return response.content


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python predict.py <model_path> <image_path> <groq_api_key>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    groq_api_key = sys.argv[3]

    # Define your classes in the same order as training
    #classes = ['Apple_leaf', 'Apple_rust_leaf', 'Apple_Scab_Leaf', 'Bell_pepper_leaf', 'Bell_pepper_leaf_spot', 'Blueberry_leaf', 'Cherry_leaf', 'Corn_Gray_leaf_spot', 'Corn_leaf_blight', 'Corn_rust_leaf', 'grape_leaf', 'grape_leaf_black_rot', 'Peach_leaf', 'Potato_leaf_early_blight', 'Potato_leaf_late_blight', 'Raspberry_leaf', 'Soyabeen_leaf', 'Squash_Powdery_mildew_leaf', 'Strawberry_leaf', 'Tomato_Early_blight_leaf', 'Tomato_leaf', 'Tomato_leaf_bacterial_spot', 'Tomato_leaf_late_blight', 'Tomato_leaf_mosaic_virus', 'Tomato_leaf_yellow_virus', 'Tomato_mold_leaf', 'Tomato_Septoria_leaf_spot','Tomato_two_spotted_spider_mites_leaf']

    with open("class_names.json", "r") as f:
        classes = json.load(f)

    classifier = PlantDiseaseClassifier(model_path, classes, groq_api_key)
    disease = classifier.predict(image_path)
    print(f"Predicted disease: {disease}")

    remedy = classifier.get_remedy(disease)
    print(f"Remedy:\n{remedy}")
