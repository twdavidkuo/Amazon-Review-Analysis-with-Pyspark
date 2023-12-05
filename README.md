# Amazon-Review-Analysis-with-Pyspark

The aim of this project is to perform sentiment analysis on the amazon review dataset. You can find the entire dataset at the link:
https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset

The total size of the dataset is around 50 gb after decompression, and the data are saved in the form of tsv.

Most of the data analysis, preprocessing, and model development are completed in PySpark, and the `review_preprocessing.py` file contains some basic functions to transform the data type, tokenize, and vectorize the textual data.

The following picture summarize the pipeline of tokenization and vectorization:

<img width="848" alt="image" src="https://github.com/twdavidkuo/Amazon-Review-Analysis-with-Pyspark/assets/52212633/9be85f00-227c-4e6e-a6d6-ab22b286dd18">

Once the vectorization is complteted, we trained the logistic regression as shown in the `final_data_processing.ipynb`, and the final training result is summarized as below:

<img width="282" alt="image" src="https://github.com/twdavidkuo/Amazon-Review-Analysis-with-Pyspark/assets/52212633/3a2d29ca-60f2-44d7-937f-4ec3bc161cda">
