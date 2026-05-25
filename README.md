📌 Project Overview

Garbage Classification for Smart Cities is an AI-powered waste management system designed to automatically detect and classify waste into different categories such as Biodegradable and Non-Biodegradable.
The project helps smart cities improve waste segregation, recycling efficiency, and environmental sustainability using Deep Learning and Computer Vision technologies.

This system can classify waste images in real-time using a trained deep learning model and provides an interactive dashboard for monitoring waste management activities.

🚀 Features

✅ AI-based garbage classification
✅ Real-time image prediction
✅ Live camera waste detection
✅ Smart waste dashboard using Streamlit
✅ Waste location tracking
✅ Collection route monitoring
✅ Analytics & reporting
✅ User authentication system
✅ Bin status monitoring
✅ Eco-friendly smart city solution

🧠 Technologies Used
Python
TensorFlow / Keras
MobileNetV2
Streamlit
OpenCV
NumPy
Pandas
PIL (Python Imaging Library)

📂 Project Structure
Garbage-Classification/
│
├── app.py                     # Streamlit Web Application
├── train_model.py             # Model Training Script
├── waste_classification_model.h5
├── model_config.json
├── dataset-resized/
│
├── smartwaste_users.json
├── smartwaste_state.json
│
└── README.md

📊 Waste Categories

The model classifies waste into:

Biodegradable Waste
Food Waste
Leaves
Organic Waste
Paper/Cardboard
Non-Biodegradable Waste
Plastic
Glass
Metal
Trash

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/Garbage-Classification.git
cd Garbage-Classification

2️⃣ Install Required Libraries
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py
🏋️ Train the Model


Run the training script:
python train_model.py --data_dir "YOUR_DATASET_PATH"

🧪 Model Information
Model Architecture: MobileNetV2
Framework: TensorFlow/Keras
Image Size: 224x224
Training Type: Transfer Learning
📸 Application Modules
🖼️ Image Classification

Upload waste images and get instant predictions.

📹 Live Detection

Detect waste using webcam in real-time.

📍 Waste Locations

Track garbage locations across smart city areas.

🚛 Collection Routes

Optimize waste collection routes.

📊 Analytics & Reports
Monitor waste management performance.

🗑️ Bin Status
Track fill levels of smart bins.

🌍 Smart City Benefits
Improves waste segregation
Reduces manual effort
Supports recycling systems
Enhances cleanliness
Helps municipalities manage waste efficiently
Promotes sustainable urban development

📈 Future Improvements
IoT Smart Bin Integration
GPS-enabled waste tracking
Mobile application support
Cloud deployment
Multi-class waste detection
AI-powered route optimization

👨‍💻 Author
Chanchal

⭐ Conclusion
The Garbage Classification for Smart Cities project demonstrates how Artificial Intelligence can help build cleaner, smarter, and more sustainable cities by automating waste classification and monitoring systems.
