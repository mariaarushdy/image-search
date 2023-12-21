# image-search
image search
# Computer Vision Project: Image Search

## Overview

This project focuses on image search using a combination of traditional image processing techniques and deep learning methods. The primary goal is to utilize various feature extraction methods to enhance image recognition and search capabilities. The DTD (Describable Textures Dataset) is used for experimentation.

## Techniques Used

1. **Color Analysis:**
   - **Color Histogram:** Utilizes the distribution of colors in the image.
   - **Indexed Color Histogram:** An extension of color histogram indexing.

2. **Texture Analysis:**
   - **DCT (Discrete Cosine Transform):** A method to represent the texture features in the frequency domain.

3. **Interest Points:**
   - **HOG (Histogram of Oriented Gradients):** Initially planned but not executed due to computational limitations.
   - **SIFT (Scale-Invariant Feature Transform):** Substituted for HOG due to computational constraints.

4. **Deep Learning Techniques:**
   - **ResNet (Residual Neural Network):** Employed for its powerful feature extraction capabilities.

## Feature Extraction Process

The following steps were taken to extract and combine features for image search:

1. **Color Features:**
   - Color histograms and indexed color histograms were computed for each image.

2. **Texture Features:**
   - DCT was applied to capture texture features in the images.

3. **Interest Points Features:**
   - SIFT was used to detect and describe key interest points in the images.

4. **Deep Learning Features:**
   - ResNet was employed to extract high-level features from the images.

## Dataset

- **DTD (Describable Textures Dataset):** The DTD dataset was used for experimentation in this project.

## Challenges and Limitations

- **Computational Power:** The project faced limitations in terms of computational power, leading to the use of a smaller dataset, which may affect the accuracy of the image search.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/computer-vision-image-search.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the image search script:
   ```bash
   python image_search.py
   ```

## Conclusion

This project explores a variety of image processing techniques and deep learning methods for image search using the DTD dataset. The combination of color, texture, interest points, and deep learning features provides a comprehensive approach to image feature extraction and search. Note that the limitations in the dataset size may impact the overall accuracy of the search results.
