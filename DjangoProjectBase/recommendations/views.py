from django.shortcuts import render
import os
from dotenv import load_dotenv, find_dotenv
import json
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np


from movie.models import Movie


def create_recommendation(description):

    env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'openAI.env')
    load_dotenv(env_file_path)
    openai.api_key = os.environ['openAI_api_key']

    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'movie_descriptions_embeddings.json')
    with open(json_file_path, 'r') as file:
        file_content = file.read()
        movies = json.loads(file_content)

    req = description
    emb = get_embedding(req, engine='text-embedding-ada-002')

    sim = []

    for i in range(len(movies)):
        sim.append(cosine_similarity(emb,movies[i]['embedding']))
    sim = np.array(sim)
    idx = np.argmax(sim)

    return movies[idx]['title']


def home(request):

    description = request.GET.get('description')
    print(description)

    if description:
        movie_title = create_recommendation(description)
        movie = Movie.objects.filter(title__icontains = movie_title)
        print(movie)
        return render(request, 'recommendations.html', {'description':description, 'movies': movie})
    
    else:
        return render(request, 'recommendations.html', {'movies': None})

