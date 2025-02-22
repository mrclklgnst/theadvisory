from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("statementmatcher/", views.statement_matcher, name="statement_matcher"),
    path("analyze_user_input/", views.analyze_user_input, name="analyze_user_input"),

]