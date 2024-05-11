from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the saved model
loaded_model = joblib.load('models\decision_treemodel.joblib')
# loaded_model.eval()

def home(request):
    return render(request, 'home.html')

def weather(request):
    return render(request, 'weather_form.html')

def predict_view(request):
    # Extract input data from the request
    precipitation = float(request.GET.get('precipitation'))
    temp_max = float(request.GET.get('temp_max'))
    temp_min = float(request.GET.get('temp_min'))
    wind = float(request.GET.get('wind'))

    # Preprocess the input data
    input_data = np.array([[precipitation, temp_max, temp_min, wind]])

    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_data)

    # Return the prediction as JSON response
    context = {'prediction': prediction[0]}
    return render(request, 'predict.html', context)
# Create your views here.
