# Handwritten Digit Recognition using SVM

This is my very first Machine Learning project, where I built a **handwritten digit recognition system** using the **Support Vector Machine (SVM)** algorithm. What makes it special is that I created my own dataset by capturing images and labeling them manually.

---

## 📁 Project Structure

📂 assets/ → (Optional assets like images for documentation)<br>
📂 captured_images/ → Raw digit images sorted in folders by label (0–9)<br>
📂 img/ → Temporary folder for live predictions<br>
📂 model/ → Saved model (digit_recognizer.pkl)<br>
📄 dataset.csv → Final dataset (28x28 binary pixel values)<br>
📄 Handwritten digit recognition.ipynb → Core Jupyter notebook<br>

---

##  Highlights

-  **10 Classes** (Digits 0 to 9)
-  **Custom Image Dataset** (50+ images per digit, extendable)
-  **Image Preprocessing**:
  - Grayscale conversion
  - Gaussian blur
  - Thresholding
  - Resizing to 28×28
  - Binarization
- **Model**: Linear **Support Vector Machine (SVM)**
- **Joblib** for model persistence
- **Accuracy Evaluation** using `sklearn.metrics`

---

##  Dataset

Each digit (0-9) has its own folder with 50+ `.png` files. During dataset creation:

- Each image is preprocessed to reduce noise
- Converted to a 28×28 binary image
- Flattened to a 1×784 feature vector
- Stored with its label in `dataset.csv`

---

## Model Training

```
from sklearn.svm import SVC
classifier = SVC(kernel="linear", random_state=6)
classifier.fit(train_x, train_y)
```
Model is saved as:
`joblib.dump(classifier, "model/digit_recognizer")`

## Live Prediction
The system captures your handwritten digit using screen grabbing, preprocesses it the same way, and uses the trained model to predict the digit. You can visualize the prediction using matplotlib.

## Learnings
- Understood fundamentals of image preprocessing
- Built a working ML model with no pre-trained datasets
- Learned how to save and load ML models
- Gained confidence in deploying a basic ML pipeline end-to-end

## Future Improvements
- Increase dataset size for better generalization<br>
- Try CNNs for higher accuracy<br>
- Build a simple web interface using Flask/Streamlit<br>
- Add sound/notification-based feedback system<br>

## 📸 Demo
<div align="centre">
<img src="assets/img (1).png" width="450" />
<img src="assets/img (2).png" width="490" />
<img src="assets/img (4).png" width="650" />
<img src="assets/img (3).png" width="40" />
</div>


## Contributions
This is a personal learning project, but feedback and suggestions are welcome!
