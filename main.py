# writing a function to load the json file

import re
import json
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

data_file = 'data/arxiv-metadata-oai-snapshot.json'

""" Using `yield` to load the JSON file in a loop to prevent Python memory issues if JSON is loaded directly"""


def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line


# we will consider below 3 categories for training
paper_categories = ["cs.AI",  # Artificial Intelligence
                    "cs.CV",  # Computer Vision and Pattern Recognition
                    "cs.LG"]  # Machine Learning


def build_dataset(categories=paper_categories):
    titles = []
    abstracts = []
    metadata = get_metadata()
    res = {}
    for paper in tqdm(metadata):
        paper_dict = json.loads(paper)
        category = paper_dict.get('categories')
        if category in res:
            res[category] +=1
        else:
            res[category] =1
            # try:
            #     year = int(paper_dict.get('journal-ref')[-4:])
            #     titles.append(paper_dict.get('title'))
            #     abstracts.append(paper_dict.get('abstract').replace("\n", ""))
            # except:
            #     pass

    # papers = pd.DataFrame({'title': titles, 'abstract': abstracts})
    # papers = papers.dropna()
    # papers["title"] = papers["title"].apply(lambda x: re.sub('\s+', ' ', x))
    # papers["abstract"] = papers["abstract"].apply(lambda x: re.sub('\s+', ' ', x))
    #
    # del titles, abstracts
    return res


def make_dataset(n:int = 100):
    val_size = int(n/20)
    output_file = 'out_train.csv'
    output_file_val = 'out_val.csv'
    titles = []
    abstracts = []
    metadata = get_metadata()
    res = {}
    i=0
    with open(output_file_val, 'a') as f:
        f.write("text,summary\n")
    with open(output_file, 'a') as f:
        f.write("text,summary\n")
        for paper in tqdm(metadata):
            if i < n:
                paper_dict = json.loads(paper)
                f.write(str(paper_dict.get('text')).replace("\n", " ") +"," + str(paper_dict.get('abstract').replace("\n", " ")) + "\n")
            else:
                with open(output_file_val, 'a') as f:
                    if i < n+val_size:
                        paper_dict = json.loads(paper)
                        f.write(str(paper_dict.get('text')).replace("\n", " ") +"," + str(paper_dict.get('abstract')).replace("\n", " ") + "\n")
                    else:
                        return
    return


"""
def generate_csv(n:int=100):
    val_size = int(n/20)
    output_file = 'out_train.csv'
    output_file_val = 'out_val.csv'
    builder = tfds.builder("scientific_papers", data_dir="./")
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train")
    ds = ds.take(n)
    with open(output_file, 'a') as f:
        f.write("text,summary\n")
        for example in ds:  # example is {'image': tf.Tensor, 'label': tf.Tensor}
            abstract = example["abstract"]
            article = example["article"]
            f.write(article.numpy().decode("utf-8") + "," + abstract.numpy().decode("utf-8") + "\n")
    ds = builder.as_dataset(split="validation")
    ds = ds.take(val_size)
    with open(output_file_val, 'a') as f:
        f.write("text,summary\n")
        for example in ds:  # example is {'image': tf.Tensor, 'label': tf.Tensor}
            abstract = example["abstract"]
            article = example["article"]
            f.write(article.numpy().decode("utf-8") + "," + abstract.numpy().decode("utf-8") + "\n")
"""


def generate_csv(n:int=10000):
    val_size = int(n/20)
    output_file = 'out_train.csv'
    output_file_val = 'out_val.csv'
    builder = tfds.builder("scientific_papers", data_dir="./")
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train")
    ds = ds.take(n)
    print(ds)
    with open(output_file, 'w') as f:
        f.write("text,summary\n")
        for example in ds:  # example is {'image': tf.Tensor, 'label': tf.Tensor}
            abstract = example["abstract"]
            article = example["article"]
            f.write('"'+bytes.decode(article.numpy()).replace('"', '\'').replace('\n', ' ') + '" ,'+'"' + bytes.decode(abstract.numpy()).replace('\n', ' ').replace('"', '\'')+'"'+ "\n")
    ds = builder.as_dataset(split="validation")
    ds = ds.take(val_size)
    with open(output_file_val, 'w') as f:
        f.write("text,summary\n")
        for example in ds:  # example is {'image': tf.Tensor, 'label': tf.Tensor}
            abstract = example["abstract"]
            article = example["article"]
            f.write('"'+bytes.decode(article.numpy()).replace('"', '\'').replace('\n', ' ') + '" ,'+'"' + bytes.decode(abstract.numpy()).replace('\n', ' ').replace('"', '\'')+'"'+ "\n")
# Press the green button in the gutter to run the script.

def main():
    #papers = build_dataset()
    #print(papers)
    #print(len(papers))
    generate_csv()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
