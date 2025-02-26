from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("sidebar/", views.sidebar, name="sidebar"),
    path("electionadvisor/", views.electionadvisor, name="electionadvisor"),
    path("analyze_user_input/", views.analyze_user_input, name="analyze_user_input"),

]