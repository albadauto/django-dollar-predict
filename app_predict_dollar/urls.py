from django.urls import path
from . import views
app_name = "app_predict_dollar"

urlpatterns = [
    path('', views.predict, name="predict"),

]
