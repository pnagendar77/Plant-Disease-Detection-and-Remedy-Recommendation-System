# Plant-Disease-Detection-and-Remedy-Recommendation-System

A deep learning + LLM powered system to detect plant leaf diseases and recommend effective remedies.
This project uses ResNet50 (PyTorch) for image classification and LangChain + ChatGroq (DeepSeek R1 LLaMA-3.3-70B) for generating natural language treatment suggestions. A Streamlit web app makes the system easy to use.

📌 Features
 Disease Detection: Classifies plant leaf images into multiple disease categories using ResNet50.
 Remedy Recommendation: Provides treatment suggestions using LLMs (ChatGroq).
 Interactive Web App: Built with Streamlit for an intuitive UI.
 Custom Dataset Support: Works with PlantDoc dataset (or any plant leaf dataset).
 GPU Acceleration: Supports CUDA-enabled training & inference.

🛠️ Tech Stack

Deep Learning: PyTorch, TorchVision (ResNet50)
LLM Integration: LangChain, ChatGroq (DeepSeek R1 / LLaMA-3.3-70B)
Web App: Streamlit
Dataset: PlantDoc Dataset
Others: Pillow, JSON, Torch Hub

Plant-Disease-Detection-Remedy/
│── Train.py                # Train ResNet50 on PlantDoc dataset
│── predict.py              # Predict disease & get remedy using LLM
│── App.py                  # Streamlit web app
│── class_names.json        # List of disease class labels
│── secrets.toml            # API key storage for Streamlit
│── plantdoc_resnet50.pth   # Trained model (generated after training)
│── requirements.txt        # Dependencies
└── README.md               # Project documentation
