from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("sidebar/", views.sidebar, name="sidebar"),
    path("electionadvisor/", views.electionadvisor, name="electionadvisor"),
    path("analyze_user_input/", views.analyze_user_input,
         name="analyze_user_input"),
    path("create_init_prompts/", views.create_init_prompts,
         name="create_init_prompts"),
    path("debug_vector_store/", views.debug_vector_store,
         name="debug_vector_store"),
]
