import pandas as pd
import json
import numpy as np
from bs4 import BeautifulSoup
import math
import os
# import pyarrow
# from transformers import AutoModel
# from datasets import Dataset, load_dataset, DatasetDict




def remove_tags(html):
    soup = BeautifulSoup(html, "html.parser")

    # Find and replace <img> tags with their 'alt' attribute
    for img in soup.find_all('img'):
        alt_text = img.get('alt', '')  # default to empty string if 'alt' is None
        if alt_text:
            # Create a new text node
            new_text = soup.new_string(" " + alt_text + " ")
            img.replace_with(new_text)
        else:
            # Remove the image if no alt text
            img.decompose()

    # Remove all script and style elements
    for tag in soup(['script', 'style']):
        tag.decompose()

    # Extract the text, cleaning up any excessive whitespace
    clean_text = ' '.join(soup.stripped_strings)
    return clean_text

# Example usage with a DataFrame column
# subset_df['problem_body'] = subset_df['problem_body'].apply(remove_tags)


dataset_paths = ['assist2009/skill_builder_data_corrected_collapsed.csv', 
 'assist2012/2012-2013-data-with-predictions-4-final.csv', 
 'assist2015/2015_100_skill_builders_main_problems.csv', 
 'assist2017/anonymized_full_release_competition_dataset.csv']
pb_df = pd.read_csv('./data_subsets/ProblemBodies_23.csv', low_memory=False)


datasets = ['assist2009', 'assist2012', 'assist2015', 'assist2017']

for dataset, dataset_path in zip(datasets, dataset_paths):
    assist_df = pd.read_csv('../data/assist2009/' + dataset_path, encoding = "ISO-8859-1", low_memory=False)
    with open('../data/'+ dataset + '/keyid2idx.json', 'r') as f:
        keyid2idx_pb_subset = json.load(f)
    problem_ids = list(keyid2idx_pb_subset['questions'].keys())
    problem_ids_int = [int(x) for x in problem_ids]
    subset_df_html_selected = pb_df[pb_df['problem_id'].isin(problem_ids_int)]
    # subset_df_html_selected goes to openai and clustering ....

    assist_subset = assist_df[assist_df['problem_id'].isin(problem_ids_int)]
    assist_subset['problem_body'] = assist_subset['problem_body'].apply(remove_tags)
    assist_subset.to_csv('../data_subsets/' + dataset + dataset_path.split('/', 1)[1] , index=False)
    
