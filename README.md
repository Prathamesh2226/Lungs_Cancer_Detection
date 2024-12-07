# Lungs_Cancer_Detection

Lung cancer remains a leading cause of cancer-related deaths worldwide, often diagnosed at advanced stages due to a lack of effective early detection methods. This project addresses the critical need for a reliable and automated system to identify lung cancer types from chest X-ray images, leveraging advanced deep learning techniques.

The proposed system utilizes a Convolutional Neural Network (CNN) architecture based on GoogLeNet to achieve accurate classification of lung cancer into multiple categories:
Adenocarcinoma
Squamous Cell Carcinoma (SCC)
Large Cell Carcinoma (LCC)

Objectives

To provide a rapid and reliable diagnostic tool to detect lung cancer types.
To enhance diagnostic precision through the integration of pretrained deep learning models.
To support healthcare professionals in identifying malignancies, especially in resource-constrained settings.
Methodology
The system employs a multi-stage pipeline:

Data Preprocessing:

Input chest X-ray images undergo resizing, normalization, and augmentation to improve model generalization.
Feature Extraction: GoogLeNet’s inception modules extract multi-level spatial features, ensuring robust feature representation.

Classification: The model classifies input images into normal or specific lung cancer categories through fully connected layers and auxiliary classifiers.

Evaluation: Performance is measured using metrics such as accuracy, precision, recall, and F1-score.

Significance
By leveraging GoogLeNet, the system benefits from its ability to extract meaningful features while maintaining computational efficiency. This is particularly important in healthcare scenarios, where timely and accurate diagnosis can significantly impact patient outcomes.

Future Directions
This project sets the foundation for integrating additional diagnostic modalities like CT scans, histopathology, and real-time mobile application deployment to broaden its utility in diverse clinical environments.
