# Apple-Detection 

This project aims to detect good and bad apples using YOLOv3 and OpenCV. The model identifies apples in images and categorizes them as either good or bad based on their visual characteristics. Detected apples are highlighted with bounding boxes: blue boxes for good apples and yellow boxes for bad apples.
## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Results](#results)
- [License](#license)

## Introduction

In this project, we employ a deep learning model based on the YOLOv3 (You Only Look Once) architecture to detect apples in an image. The goal is to distinguish between good apples (those suitable for consumption or sale) and bad apples (those that are blemished or decayed).

The system provides real-time detection and draws bounding boxes around the apples:

Good apples: Blue bounding boxes
Bad apples: Yellow bounding boxes

## Features:

- Real-time detection of apples in images.
- Categorization of apples as good or bad.
- Visual display of the detected apples with colored bounding boxes.
  - **Blue** for good apples.
  - **Yellow** for bad apples.
- High accuracy and speed using the YOLOv3 architecture.

## Technologies Used
- Python
- OpenCV
- YOLOv3
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Apple-Detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Apple-Detection
    ```
3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How It Works
- The YOLOv3 model is used to detect apples in the input images.
- The apples are categorized as good or bad based on their appearance.
- The detected apples are highlighted with colored bounding boxes:
  - **Blue** for good apples
  - **Yellow** for bad apples
 

## Results
- The model achieves high accuracy in detecting and categorizing apples.
- Below are two examples of the detection results showing both good (blue box) and bad (yellow box) apples:

    **Example 1: Detection Result 1**
    
    ![Apple Detection Example 1](https://github.com/Vikas8773/prototype_1_SIH2024/blob/main/Output%20Images/image%201.png)

    **Example 2: Detection Result 2**
    
    ![Apple Detection Example 2](https://github.com/Vikas8773/prototype_1_SIH2024/blob/main/Output%20Images/image%202.png)

## License
This project is licensed under the MIT License.
