import torch
from openai import OpenAI
import json
import pandas as pd
import os

from openai import OpenAI

# Load API key from config.json
with open('emb_cluster_config.json') as config_file:
    config = json.load(config_file)
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


datasets = ['assist2009', 'assist2012', 'assist2017']

for dataset in datasets:
    print(f'Processing dataset: {dataset}')
    pb_subset_df = pd.read_csv('data_subsets/' + dataset + '/questions.csv')
    
    skill_texts = pb_subset_df['problem_body'].dropna().astype(str).tolist()

    model_dims = [('text-embedding-3-small', 768), ('text-embedding-3-small', 1536), \
    ('text-embedding-3-large', 768), ('text-embedding-3-large', 1536), ('text-embedding-3-large', 3072) \
    , ('text-embedding-ada-002', 1536)]

    emb_filenames = []
    chunk_size = 2046
    skill_texts_stacked_chunks = [skill_texts[i:i + chunk_size] for i in range(0, len(skill_texts), chunk_size)]
    for (model,emb_dim) in model_dims:
        print(model, emb_dim)
        skills_tensor = torch.empty((0, emb_dim))
        for chunk in skill_texts_stacked_chunks:
            if model == 'text-embedding-ada-002':
                response_embeddings = client.embeddings.create(input = chunk, model=model).data
            else:
                response_embeddings = client.embeddings.create(input = chunk, model=model, dimensions=emb_dim).data
            for data in response_embeddings:
                emb = torch.tensor(data.embedding)
                emb = torch.reshape(emb, (1, emb_dim))
                skills_tensor = torch.cat((skills_tensor, emb), dim=0)

        torch.save(skills_tensor, 'embeddings/' + dataset + '/openai_'+ model + '_' +str(emb_dim) +  '.pt')
        emb_filenames.append('embeddings/' + dataset + '/openai_'+ model + '_' +str(emb_dim) +  '.pt')
    print(emb_filenames)
    torch.save(emb_filenames, 'embeddings/' + dataset + '/filenames.pt')
    print('Saved embeddings for dataset:', dataset)

