#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, from_unixtime
from pyspark.sql.types import DateType
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import *
from operator import add
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml import Pipeline
from sparknlp.annotator import Tokenizer
from sparknlp.base import DocumentAssembler, Finisher
from pyspark.ml.clustering import KMeans
import warnings
warnings.filterwarnings('ignore')
import pandas as pd


# In[2]:


from IPython.core.display import HTML
display(HTML("<style>pre { white-space: pre !important; }</style>"))


# In[ ]:


def split_text_column(df, colname):
    # Apply regex to remove punctuation, trim spaces, and convert to lowercase
    df = df.withColumn(colname + "_cleaned", lower(regexp_replace(col(colname), "[^a-zA-Z\\s]", " ")))
    df = df.withColumn(colname + "_cleaned", trim(regexp_replace(col(colname + "_cleaned"), "\\s+", " ")))

    # Spark NLP Tokenization
    document_assembler = DocumentAssembler().setInputCol(colname + "_cleaned").setOutputCol("document")
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("tokenized")
    nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer])

    print("Fitting NLP pipeline for column:", colname)
    nlp_model = nlp_pipeline.fit(df)
    print("Transforming DataFrame for column:", colname)
    df = nlp_model.transform(df)

    # Use Spark NLP Finisher to extract tokens as strings
    finisher = Finisher().setInputCols(["tokenized"]).setCleanAnnotations(True)
    df = finisher.transform(df)
    df = df.withColumnRenamed("finished_tokenized", colname + "_split")

    return df


# In[ ]:


def read_trim_tsv(path, sample=True, sample_size=500):
    num_nodes = 10  # Set the desired number of nodes dynamically

    spark = SparkSession.builder.appName('read')\
            .config('spark.executor.instances', str(num_nodes))\
            .getOrCreate()
     # Read DataFrame
    df = spark.read.option("delimiter", "\t").csv(path, header=True)
    df = df.na.drop()
    # Sample DataFrame
    if sample:
        subset_size = sample_size  # in megabytes
        # Calculate the number of partitions based on subset size
        num_partitions = int(df.rdd.map(lambda x: len(str(x))).sum() / (subset_size * 1024 * 1024)) + 1
        # Randomly split the DataFrame into subsets
        subsets = df.randomSplit([1.0] * num_partitions, seed=42)
        # Keep the first subset for prototyping
        df = subsets[0]

    # Data cleaning and type casting
    df = df.withColumn('review_date', df['review_date'].cast(DateType()))
    df = df.withColumn("star_rating", df["star_rating"].cast("int"))
    df = df.withColumn("helpful_votes", df["helpful_votes"].cast("int"))
    df = df.withColumn("total_votes", df["total_votes"].cast("int"))
    df = df.withColumn("verified_purchase", when(col("verified_purchase") == "Y", True).otherwise(False))
    df = df.withColumn("vine", when(col("vine") == "Y", True).otherwise(False))

    df = df.withColumn("headline_length", length(col("review_headline")))
    df = df.withColumn("review_length", length(col("review_body")))
    
    #split and tokenize each of the string columns
    df = split_text_column(df, "review_body");
    df = split_text_column(df, "review_headline");
    df = split_text_column(df, "product_title")
    # Calculate the number of words in each review
    df = df.withColumn("review_num_words", size(col("review_body_split")))

    # Filter reviews with more than 3 words
    df = df.filter(col('review_num_words') > 3)

    # N-gram extraction with Spark MLlib
    n = 3
    ngram = NGram(n=n, inputCol="review_body_split", outputCol="review_3grams")
    df = ngram.transform(df)
    n = 2
    ngram = NGram(n=n, inputCol="review_body_split", outputCol="review_2grams")
    df = ngram.transform(df)

    return df


# In[ ]:


def repartition_df(df):
    sc = spark.sparkContext
    num_worker_nodes = len(sc.statusTracker().getExecutorInfos())
    current_partitions = df.rdd.getNumPartitions()
    print("number of worker nodes:" + str(num_worker_nodes))
    print("current_partitions:" + str(current_partitions))
    target_partitions = num_worker_nodes * current_partitions
    repartitioned_df = df.repartition(target_partitions)
    return repartitioned_df


# In[ ]:


# use merge = True when you do not want the count vector for each column
# if merge = True, the function will merge all the string columns and return 1 single count vector
def count_vectorize(df, colname, merge = False):
    if merge:
        tokenized_columns = ['review_body_split', 'review_headline_split', 'product_title_split'];
        df = df.withColumn('all_tokens', col('review_body_split').cast('array<string>')
                   .concat(col('review_headline_split')).cast('array<string>')
                   .concat(col('product_title_split')).cast('array<string>'))
        cv = CountVectorizer(inputCol='all_tokens', outputCol='vectorized_features', minDF=100.0, maxDF = 1000000)
        cv_model = cv.fit(df)
        return cv_model.transform(df);
    
    else: 
        cv = CountVectorizer(inputCol = colname, 
                         outputCol = 'count_vectorized_' + colname, 
                         minDF = 100.0, maxDF = 1000000, vocabSize = 5000)
        pca = PCA(k=40, inputCol='count_vectorized_' + colname, outputCol='count_vectorized_' + colname + '_reduced')
        pipeline = Pipeline(stages=[cv, pca])
        model = pipeline.fit(df)
        return model.transform(df)
    


# In[ ]:


# use merge = True when you do not want the tf-idf vector for each column
# if merge = True, the function will merge all the string columns and return 1 single tf-idf vector
def tf_idf(df, colname, merge = False, features = 20):
    if merge:
        tokenized_columns = ['review_body_split', 'review_headline_split', 'product_title_split']
        df = df.withColumn('all_tokens', col('review_body_split').cast('array<string>')
                   .concat(col('review_headline_split')).cast('array<string>')
                   .concat(col('product_title_split')).cast('array<string>'))
        hashingTF = HashingTF(inputCol='all_tokens', outputCol='_raw_features', numFeatures=features)
        featurized_data = hashingTF.transform(df)
        idf = IDF(inputCol='raw_features', outputCol='tf_idf_features')
        idf_model = idf.fit(featurized_data)
        return idf_model.transform(featurized_data)
    else:
        hashingTF = HashingTF(inputCol=colname, outputCol= 'tf_idf_'+colname+'_raw_features' , numFeatures=features)
        featurized_data = hashingTF.transform(df)
        idf = IDF(inputCol='tf_idf_'+colname+'_raw_features', outputCol=colname+'_tf_idf_features')
        idf_model = idf.fit(featurized_data)
        return idf_model.transform(featurized_data)
        


# In[ ]:


def top_k_ngram(df, k=5, n=2):
    review_ngrams_rdd = df.select("review_" + str(n) + "grams").rdd.flatMap(lambda x: x)

# Flatten the list of review_2grams
    flat_review_ngrams = review_ngrams_rdd.flatMap(lambda x: x)

# Count the occurrences of each 2gram
    count_ngrams = flat_review_ngrams.map(lambda x: (x, 1)).reduceByKey(add)

# Get the top 5 most common 2grams
    top_k_ngrams = count_ngrams.takeOrdered(k, key=lambda x: -x[1])  # Use negative key for descending order

# Display the result
    print("k = " + str(k) + ", n = " + str(n));
    print("Top k most common review_ngrams:")
    for gram, count in top_k_ngrams:
        print(gram, ":", count)
        
    return top_k_ngrams

def vectorize_df(df, merge=False, features = 20):
    num_nodes = 10
    spark = spark = SparkSession.builder.appName('vectorize')\
            .config('spark.executor.instances', str(num_nodes))\
            .getOrCreate()
    tokenized_columns = ['review_body_split', 'review_headline_split', 'product_title_split']
    for i in tokenized_columns:
        df = tf_idf(df, colname=i, merge=merge, features=features)
        df = count_vectorize(df, colname=i, merge = merge)
    return df;
        