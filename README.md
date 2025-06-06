# Predicting-Insurance-Charges
This repository contains a script for predicting insurance charges using various Machine Learning techniques and a deep learning approach implemented in Pytorch.

--> Documentation:
This script performs data cleaning, feature engineering, and model training to predict insurance charges using both traditional machine learning models and a deep learning model implemented in PyTorch.
The script also includes timing functionality to measure the execution time of both the machine learning and deep learning models, providing insights into computational efficiency.
The script is structured to allow for easy modification and extension, enabling further experimentation with different models, hyperparameters, and data preprocessing techniques.

--> Conclusion:
It demonstrates the use of various regression models to predict insurance charges, including hyperparameter tuning for each model. It also implements a deep learning model using PyTorch, showcasing the training process and evaluation metrics. The results indicate the effectiveness of both approaches, with the deep learning model providing a robust alternative to traditional regression methods.
We can see that using machine learning models, we can achieve a good prediction of insurance charges. Using multiple models, we can see that Random Forest Regressor with 81% R² yield the best results in terms of R² and other metrics.
The deep learning model, while more complex, can capture non-linear relationships in the data and may outperform traditional models on larger datasets or more intricate patterns. The choice between these approaches depends on the specific dataset characteristics and the complexity of the relationships involved. We have achieved 74% accuracy with the Deep Learning Model.
