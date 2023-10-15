# Food_Vision
* A deep learning-based food image recognition model.
* This project is inspired by the research paper [DeepFood: Deep Learning-Based Food Image Recognition for Computer-Aided Dietary Assessment](https://arxiv.org/abs/1606.05675)

## Dataset: [Food 101](https://www.kaggle.com/datasets/dansbecker/food-101)
This dataset can also be accessed through [Tensorflow Datasets (TFDS)](https://www.tensorflow.org/datasets/overview)
>> This dataset consists of 101 food categories with 101,000 images. For each class, 250 manually reviewed test images are provided, as well as 750 training images. On purpose, the training images were not cleaned and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.

## Model:
* EfficientNet B4 model was used in this project which gave **81%** accuracy upon fine tuning.
* Applied **Mixed precision** training to boost GPU functionality.
  * Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory. By keeping certain parts of the model in the 32-bit types for numeric stability, the model will have a lower step time and train equally as well in terms of the evaluation metrics such as accuracy.
* Used the **Prefetch** feature to cut down on data loading times.
  * Prefetching overlaps the preprocessing and model execution of a training step. While the model is executing training step s, the input pipeline is reading the data for step s+1. Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.
