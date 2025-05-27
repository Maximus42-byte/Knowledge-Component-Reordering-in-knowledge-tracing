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


# dataset_paths = ['assist2009/skill_builder_data_corrected_collapsed.csv', 
#  'assist2012/2012-2013-data-with-predictions-4-final.csv', 
#  'assist2015/2015_100_skill_builders_main_problems.csv', 
#  'assist2017/anonymized_full_release_competition_dataset.csv']
# datasets = ['assist2009', 'assist2012', 'assist2015', 'assist2017']

dataset_paths = ['assist2009/skill_builder_data_corrected_collapsed.csv', 
 'assist2012/2012-2013-data-with-predictions-4-final.csv', 
 'assist2017/anonymized_full_release_competition_dataset.csv']
datasets = ['assist2009', 'assist2012', 'assist2017']

pb_df = pd.read_csv('./data_subsets/ProblemBodies_23.csv', low_memory=False)




for dataset, dataset_path in zip(datasets, dataset_paths):
    print(f'Processing dataset: {dataset}')

    assist_df = pd.read_csv('raw_data/' + dataset_path, encoding = "ISO-8859-1", low_memory=False)
    problem_id_column_name = 'problem_id'
    if dataset == 'assist2017':
        problem_id_column_name = 'problemId'
        
    questions = assist_df[problem_id_column_name].unique()
    questions_w_text = pb_df[pb_df['problem_id'].isin(questions)]
    print(f'Number of total questions in {dataset} : {len(questions)}')
    print(f'Number of questions in {dataset} with text: {len(questions_w_text)}')
    assist_df_temp = assist_df[assist_df[problem_id_column_name].isin(questions_w_text['problem_id'])]
    assist_df_temp.to_csv('../data/' + dataset_path)

    print(f'Pre Processing dataset: {dataset}')
    os.system(f'python ../examples/data_preprocess.py --dataset_name={dataset}')
    print(f'Pre Processing dataset finished : {dataset}')

    with open('../data/'+ dataset + '/keyid2idx.json', 'r') as f:
        keyid2idx_pb_subset = json.load(f)
    print(f'Loaded keyid2idx_pb_subset for {dataset}')
    problem_ids = list(keyid2idx_pb_subset['questions'].keys())
    problem_ids_int = [int(x) for x in problem_ids]
    print(f'Number of problems in {dataset}: {len(problem_ids_int)}')
    subset_df_html_selected = pb_df[pb_df['problem_id'].isin(problem_ids_int)]


    assist_subset = assist_df[assist_df[problem_id_column_name].isin(problem_ids_int)]
    assist_subset.to_csv('data_subsets/' + dataset_path , index=False)
    subset_df_html_selected['problem_body'] = subset_df_html_selected['problem_body'].apply(remove_tags)
    print('Removed HTML tags from problem bodies')
    subset_df_questions = subset_df_html_selected[['problem_id','problem_body']]
    subset_df_questions.rename(columns={'problem_id': problem_id_column_name}, inplace=True)
    subset_df_questions[problem_id_column_name].fillna("Middle School generic math question", inplace=True)
    subset_df_questions.to_csv('data_subsets/' + dataset + '/questions.csv' , index=False)
    print(f'Saved problem bodies and problem ids for {dataset} to data_subsets/{dataset}/questions.csv')
    print(f'Preprocessing completed for {dataset}')

print('Preprocessing completed for all datasets.')
print('You can now proceed with the clustering and dimensionality reduction steps.')