from django.shortcuts import render, redirect
from django.urls import reverse
import tensorflow as tf
import pickle
import sklearn
model = tf.keras.models.load_model(
    "H:/Python/Outros/django/previsaoDolar/previsaoPrecoDolar/app_predict_dollar/models/model.h5")

with open("H:/Python/Outros/django/previsaoDolar/previsaoPrecoDolar/app_predict_dollar/models/tokenizer.p", 'rb') as tokenizer:
    token = pickle.load(tokenizer)

with open("H:/Python/Outros/django/previsaoDolar/previsaoPrecoDolar/app_predict_dollar/models/y_scaler.p", 'rb') as scaler1:
    y_scaler = pickle.load(scaler1)

with open("H:/Python/Outros/django/previsaoDolar/previsaoPrecoDolar/app_predict_dollar/models/x_scaler.p", 'rb') as scaler2:
    x_scaler = pickle.load(scaler2)


def predict(request):
    predict_new = [[]]
    if request.method == 'POST':
        date = [request.POST.get("date")]
        token.fit_on_sequences(date)
        token.fit_on_texts(date)
        test_date = token.texts_to_sequences(date)
        test_date_normalized = x_scaler.transform(test_date)
        predict = model.predict(test_date_normalized)
        predict = predict.reshape(1, -1)
        predict_new = y_scaler.inverse_transform(predict)
    print(predict_new)
       
    return render(request, 'predict/home.html', {'prediction': predict_new[0][0] })
