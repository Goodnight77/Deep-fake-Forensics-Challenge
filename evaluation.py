import pandas as pd
import cv2
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm

# Initialize lists to store F1 scores
label_scores = []
labels=[]
predicted_labels=[]
j=0
# Iterate over the test dataset
for k, row in tqdm(test.iterrows()):
    j=j+1
    image_path = row['image']
    try:
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.resize(image, (256, 256))
      image = img_to_array(image)
      image = preprocess_input(image)
      image = np.expand_dims(image, axis=0)

      predicted_mask, predicted_label = model.predict(image)
      label = row['class']
      labels.append(label)

      predicted_label = predicted_label.astype(int)
      predicted_label = predicted_label.flatten()
      predicted_labels.append(predicted_label)

      #calculate F1 score
      label_f1 = f1_score([label], predicted_label, zero_division=1)
      label_scores.append(label_f1)

      # Calculate ROC AUC score
      label_auc = roc_auc_score([label], predicted_label)
      label_scores.append(label_auc)
    except:
        print("error with img")

true_labels_array = np.array(labels)
predicted_labels_array = np.array(predicted_labels)

true_labels_array = true_labels_array.flatten()
predicted_labels_array = predicted_labels_array.flatten()

# Calculate the overall F1 score
overall_f1 = f1_score(true_labels_array, predicted_labels_array, average='micro', zero_division=1)
overall_auc = roc_auc_score(true_labels_array, predicted_labels_array)
print('the f1 score of the labels is :',overall_f1 )
print('the AUC score of the labels is :',overall_auc )
