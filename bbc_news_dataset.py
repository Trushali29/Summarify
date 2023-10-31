import os
import pandas as pd
import csv
news_path = 'D:\\Research Papers\\TextSummary\\BBC News Summary\\News Articles'
summary_news_path = 'D:\\Research Papers\\TextSummary\\BBC News Summary\\Summaries'
dataset = []
for category in os.listdir(news_path):
    category_dir = os.path.join(news_path,category)
    for filename in os.listdir(category_dir):
        if filename.endswith(".txt") and category == "business":
            try:
                with open(os.path.join(category_dir, filename), 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    article = ' '.join(lines[1:]).strip()
                    title = lines[0].strip()
                    data_dict = {'category' : category,'title' : title,'article' : article }
                    dataset.append(data_dict)
            except UnicodeDecodeError:
                with open(os.path.join(category_dir, filename), 'r', encoding='unicode_escape') as file:
                    lines = file.readlines()
                    article = ' '.join(lines[1:]).strip()
                    title = lines[0].strip()
                    data_dict = {'category' : category,'title' : title,'article' : article }
                    dataset.append(data_dict)
    
            

summary_dataset = []        
for category in os.listdir(summary_news_path):
    category_dir = os.path.join(summary_news_path,category)
    for filename in os.listdir(category_dir):
        if filename.endswith(".txt") and category == "business":
            try:
                with open(os.path.join(category_dir, filename), 'r', encoding='utf-8') as file:
                    lines = ' '.join(file.readlines()).strip()
                    data_dict = {'summary':lines}
                    summary_dataset.append(data_dict)
            except UnicodeDecodeError:
                with open(os.path.join(category_dir, filename), 'r', encoding='unicode_escape') as file:
                    lines = ' '.join(file.readlines()).strip()
                    data_dict = {'summary':lines}
                    summary_dataset.append(data_dict)              
                
news_dataset = pd.DataFrame(dataset)
print(news_dataset.loc[0])
news_dataset['summary'] = pd.DataFrame(summary_dataset)
print(news_dataset.head())
news_dataset.to_csv('business_bbc_news.csv')
