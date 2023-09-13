# CPT-ANN-Prediction: Predicting Shear Wave Velocity from CPT Parameters

## Description
An ANN-based regression model designed to predict shear wave velocity (Vs) from cone penetration testing (CPT) parameters. This cutting-edge model combines state-of-the-art machine learning techniques with geotechnical mechanics to achieve high prediction accuracy.

## Background
Accurate predictions of soil properties, like shear wave velocity (Vs), are paramount in geotechnical engineering. These predictions influence the design and safety of structures ranging from buildings to bridges. Traditional methods, such as lab testing, can be expensive and time-consuming. This project introduces an efficient machine learning alternative, using CPT parameters to predict Vs values.

## Technical Details

### Data Preprocessing
- The data is cleaned to convert all values to numeric types and remove any missing entries.
- Features are standardized to have a mean of 0 and a standard deviation of 1.

### Neural Network Architecture
- The model is a fully connected feedforward neural network.
- It consists of an input layer, two hidden layers, and an output layer.
- The hidden layers use the hyperbolic tangent (tanh) activation function.
- Batch normalization is applied after each hidden layer.

### Training and Validation
- The model is trained using the Adam optimizer and the backpropagation algorithm.
- Early stopping is implemented to prevent overfitting.
- K-fold cross-validation is used for robust model validation.

## Repository Structure
- `code/`: Contains Python scripts for different stages of the project.
- `data/`: (Not included due to publication constraints) Place your dataset here.
- `results/`: Stores the model's outputs, plots, and trained models (if shared).
- `paper_draft.md`: Draft of the academic paper detailing methodologies and results.

## Dependencies
- Python 3.8+
- Pytorch
- NumPy
- pandas
- scikit-learn

## Instructions for Replication
1. Clone this repository to your local machine.
2. Install the necessary dependencies.
3. Place your dataset in the `data/` directory.
4. Execute scripts in the following order:
   - `data_preprocessing.py`
   - `model_architecture.py`
   - `training.py`
   - `evaluation.py`

## Results (partial)
![image](https://github.com/Amordia/CPT-Vs-Prediction_ANN/assets/78806289/ff8a48fe-c5d9-44de-a481-ade4a16bc3e2)
![image](https://github.com/Amordia/CPT-Vs-Prediction_ANN/assets/78806289/175bc53f-71d0-45f4-bd2f-c49b78cfd392)
![image](https://github.com/Amordia/CPT-Vs-Prediction_ANN/assets/78806289/b8185636-bee5-439b-8d62-fc2c5b03a893)

## Publications
The methodologies and techniques from this project are being compiled for submission to top-tier geotechnical journals, such as "Soil Dynamics and Earthquake Engineering" and "Journal of Earthquake Engineering & Structural Dynamics".

## License
This project is open-sourced under the MIT License. However, the dataset and some results are not shared due to pending publications.

## Contributors
- [Yipeng Xu] - Main researcher and developer
- [Supervisor: Ahmad Mousa] - The idea and some of the dataset
- [Ph.D. student: Abdulhalim SAEED] - Additional contributions

