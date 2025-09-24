import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('tr_model_svm.sav','rb'))
print("Model loaded successfully!")

# Define prediction function
def prediction(input_data):
    data_tab = np.asarray(input_data)
    data_r = data_tab.reshape(1, -1)
    prediction = model.predict(data_r)
    if prediction[0] == 0:
        return "Patient is not likely to have a stroke"
    else:
        return "Patient is likely to have a stroke"

# Example usage
# Example test input:
# [gender, age, hypertension, heart_disease, avg_glucose, bmi,
#  smoking_Unknown, smoking_Formerly smoked, smoking_Never smoked, smoking_Currently smoking]
sample_input = [1, 65, 0, 1, 150, 28, 0, 0, 1, 0]

result = prediction(sample_input)
print("Prediction Result:", result)
