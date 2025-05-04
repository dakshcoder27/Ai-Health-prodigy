from django.urls import path, include
from main import views

urlpatterns = [
    path('' , views.front , name = 'index'),
    path('know-your-disease' , views.symptomsDis , name = 'know-your-disease'),

    path('calculate-bmi/' , views.bmiCalc , name = 'bmi-calculator'),

    path('alzheimer/' , views.alzheimerPred , name = 'alzheimer-predict'),

    path('brain-tumor/' , views.brainPred , name = 'brain-tumor-predict'),

    path('liver/' , views.liver , name = 'liver-view'),
    path('liver-prediction/' , views.liverPred , name = 'liver-predict'),

]