import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str,silver_dir,gold_dir,spark):
    """Create the main feature store table by joining all silver tables"""
    
    # print(f"Creating gold feature store for {snapshot_date_str}")
    
    # # Load all silver tables
    # loans_df = spark.read.parquet(f"{silver_dir}/silver_loan_performance")\
    #     .filter(col("snapshot_date") == snapshot_date_str)
    # financials_df = spark.read.parquet(f"{silver_dir}/silver_customer_financials")\
    #     .filter(col("snapshot_date") == snapshot_date_str)
    # attributes_df = spark.read.parquet(f"{silver_dir}/silver_customer_attributes")\
    #     .filter(col("snapshot_date") == snapshot_date_str)
    # clickstream_df = spark.read.parquet(f"{silver_dir}/silver_clickstream_features")\
    #     .filter(col("snapshot_date") == snapshot_date_str)
    
    # print(f"Loaded silver tables - Loans: {loans_df.count()}, Financials: {financials_df.count()}, Attributes: {attributes_df.count()}, Clickstream: {clickstream_df.count()}")

       # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # names=["attributes", "clickstream", "financials", "loan_daily"]
    # for name in names:
    # # connect to bronze table
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_dir + "attributes/" + partition_name
    attributes_df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', attributes_df.count())

    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_dir + "clickstream/" + partition_name
    clickstream_df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', clickstream_df.count())

    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_dir + "financials/" + partition_name
    financials_df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', financials_df.count())    

    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_dir + "loan_daily/" + partition_name
    loans_df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', loans_df.count())


    
    # Aggregate loan performance by customer
    loan_agg = loans_df.groupBy("Customer_ID").agg(
        F.sum("loan_amt").alias("Total_Loan_Amount"),
        F.sum("balance").alias("Current_Outstanding_Balance"), 
        F.count("loan_id").alias("Active_Loans_Count"),
        F.avg("dpd").alias("Avg_DPD"),
        F.max("dpd").alias("Max_DPD"),
        F.sum("overdue_amt").alias("Total_Overdue_Amount"),
        F.max("overdue_amt").alias("Max_Overdue_Amount"),
        F.avg("payment_ratio").alias("Avg_Payment_Ratio"),
        F.sum("is_overdue").alias("Overdue_Count"),
        F.count("*").alias("Total_Installments"),
        F.avg("overdue_severity").alias("Avg_Overdue_Severity")
    ).withColumn("Overdue_Rate", col("Overdue_Count") / col("Total_Installments"))
    
    print(f"Loan aggregations created: {loan_agg.count()} customers")
    
    # Join all customer data (start with customers who have all data types)
    feature_store = clickstream_df.join(attributes_df, ["Customer_ID"], "inner")\
        .join(financials_df, ["Customer_ID"], "inner")\
        .join(loan_agg, ["Customer_ID"], "inner")
    
    print(f"Joined feature store: {feature_store.count()} customers")
    
    # Select final feature set for ML
    feature_columns = (
        # Identifiers
        ["Customer_ID"] +
        
        # Clickstream features (behavioral)
        [f"fe_{i}" for i in range(1, 21)] +
        ["positive_behavior_score", "negative_behavior_score", "total_engagement_score"] +
        
        # Demographics  
        ["Age", "Occupation_Category", "Age_Group", "Age_Risk_Score"] +
        
        # Financial features (cleaned)
        ["Annual_Income_Clean", "Monthly_Inhand_Salary", "Num_Bank_Accounts", 
         "Num_Credit_Card", "Interest_Rate", "Credit_Utilization_Ratio",
         "Credit_History_Age_Months", "debt_to_income_ratio", "savings_rate",
         "credit_card_intensity", "Payment_Behaviour_Category"] +
        
        # Loan performance features (aggregated)
        ["Total_Loan_Amount", "Current_Outstanding_Balance", "Active_Loans_Count",
         "Avg_DPD", "Max_DPD", "Total_Overdue_Amount", "Max_Overdue_Amount",
         "Overdue_Rate", "Avg_Payment_Ratio", "Avg_Overdue_Severity"]
    )
    
    feature_store = feature_store.select(*feature_columns)
    
    # Add metadata
    feature_store = feature_store.withColumn("feature_creation_timestamp", F.current_timestamp())\
        .withColumn("feature_store_version", F.lit("v1.0"))
    
   # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_dir + partition_name
    feature_store.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return feature_store

