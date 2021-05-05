# CSC3095 - Evaluating deep learning approaches for perceived age prediction

## Usage

```python
python train_age_classification.py/train_age_regression.py --b 512 --lr 0.001 --layers 8 --type 'VGG16' --mode 'finetune'
```

### Datasets 

These can be placed in the data folder and processed with the relevant scripts in *util/* and *face_detection/*
### Mode

- **pretrain** - IMDb-Wiki dataset
- **finetune** - ChaLearn Percieved Age V2 dataset

### TensorBoard logging 

Can be switched on by adding the callbacks paramater to the `model.fit()` with the callbacks variable

### References 

- Rothe, R., Timofte, R. & Van Gool, L. (2016) Deep Expectation of Real and Apparent Age from a Single Image Without Facial Landmarks. International Journal of Computer Vision. 126 (2-4), 144–157.
- https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
- http://chalearnlap.cvc.uab.es/dataset/19/description/ 
- https://github.com/ageitgey/face_recognition_models
