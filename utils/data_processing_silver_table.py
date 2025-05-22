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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    
    df = df.withColumn("is_current", F.when(col("overdue_amt") == 0, 1).otherwise(0))
    df = df.withColumn("is_overdue", F.when(col("overdue_amt") > 0, 1).otherwise(0))
    df = df.withColumn("payment_ratio", col("paid_amt") / (col("due_amt") + 0.01))
    df = df.withColumn("balance_ratio", col("balance") / col("loan_amt"))
    df = df.withColumn("overdue_severity", 
                      F.when(col("overdue_amt") > 0, col("overdue_amt") / col("due_amt")).otherwise(0))
    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_financials_table(snapshot_date_str, bronze_dir, silver_financials_directory, spark):
     # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean data quality issues (fix the "_" suffixes you discovered)
    df = df.withColumn("Annual_Income_Clean", 
                      F.when(col("Annual_Income").rlike(".*_$"), 
                           F.regexp_replace(col("Annual_Income"), "_", "").cast(FloatType()))
                       .otherwise(col("Annual_Income").cast(FloatType())))
    
    df = df.withColumn("Num_of_Loan_Clean",
                      F.when(col("Num_of_Loan").rlike(".*_$"),
                           F.regexp_replace(col("Num_of_Loan"), "_", "").cast(IntegerType()))
                       .otherwise(col("Num_of_Loan").cast(IntegerType())))
    
    # Parse credit history age from "X Years and Y Months" format
    df = df.withColumn("Credit_History_Age_Months",
                      F.when(col("Credit_History_Age").rlike("(\\d+) Years and (\\d+) Months"),
                           F.regexp_extract(col("Credit_History_Age"), "(\\d+) Years and (\\d+) Months", 1).cast(IntegerType()) * 12 +
                           F.regexp_extract(col("Credit_History_Age"), "(\\d+) Years and (\\d+) Months", 2).cast(IntegerType()))
                       .otherwise(0))
    
    # Financial health indicators for ML
    df = df.withColumn("debt_to_income_ratio", 
                      col("Total_EMI_per_month") / (col("Monthly_Inhand_Salary") + 0.01))
    df = df.withColumn("savings_rate", 
                      col("Amount_invested_monthly") / (col("Monthly_Inhand_Salary") + 0.01))
    df = df.withColumn("credit_card_intensity", 
                      col("Num_Credit_Card") / F.greatest(col("Num_Bank_Accounts"), F.lit(1)))
    
    # Categorize payment behavior
    df = df.withColumn("Payment_Behaviour_Category",
                      F.when(col("Payment_Behaviour").like("%Low_spent%"), "Low_Spender")
                       .when(col("Payment_Behaviour").like("%High_spent%"), "High_Spender")
                       .otherwise("Medium_Spender"))
    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
def process_silver_attributes_table(snapshot_date_str, bronze_dir, silver_attributes_dir, spark):
    """Process customer attributes with categorical encoding"""
    
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # Categorize occupation for better ML performance
    df = df.withColumn("Occupation_Category",
                    F.when(col("Occupation").isin(["Engineer", "Doctor", "Lawyer", "Scientist", "Teacher"]), "Professional")
                    .when(col("Occupation").isin(["Manager", "Executive"]), "Management")
                    .when(col("Occupation").isin(["Developer", "Architect"]), "Technology")
                    .when(col("Occupation").isin(["Entrepreneur", "Business"]), "Business")
                    .when(col("Occupation").isin(["Mechanic", "Technician"]), "Skilled_Trade")
                    .otherwise("Other"))
    
    # Age groups for better model interpretability
    df = df.withColumn("Age_Group",
                    F.when(col("Age") < 25, "Young_Adult")
                    .when(col("Age") < 35, "Young_Professional") 
                    .when(col("Age") < 50, "Mid_Career")
                    .when(col("Age") < 65, "Senior_Professional")
                    .otherwise("Retirement_Age"))
    
    # Age-based risk indicators
    df = df.withColumn("Age_Risk_Score",
                    F.when(col("Age") < 22, 3)  # Very young, higher risk
                    .when(col("Age") < 30, 2)  # Young, moderate risk
                    .when(col("Age") < 50, 1)  # Prime age, lower risk
                    .when(col("Age") < 65, 2)  # Pre-retirement, moderate risk
                    .otherwise(3))  # Senior, higher risk
    
     # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_dir + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

def process_silver_clickstream_table(snapshot_date_str, bronze_dir, silver_clickstream_dir, spark):
    """Process customer attributes with categorical encoding"""
    
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # Calculate behavioral indicators
    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    
    # Positive vs negative behavioral signals
    positive_features = [f"fe_{i}" for i in range(1, 11)]  # Assume first 10 are positive behaviors
    negative_features = [f"fe_{i}" for i in range(11, 21)]  # Assume last 10 are negative behaviors
    
    df = df.withColumn("positive_behavior_score", 
                      sum([F.when(col(f) > 0, col(f)).otherwise(0) for f in positive_features]))
    df = df.withColumn("negative_behavior_score",
                      sum([F.when(col(f) < 0, F.abs(col(f))).otherwise(0) for f in negative_features]))
    
    # Overall engagement score
    df = df.withColumn("total_engagement_score",
                      sum([F.abs(col(f)) for f in feature_cols]))
    
    # Behavioral consistency (variance of features)
    feature_array = F.array(*[col(f) for f in feature_cols])
    df = df.withColumn("behavioral_variance", 
                      F.aggregate(feature_array, F.lit(0), lambda acc, x: acc + (x * x)) / len(feature_cols))
    
     # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_dir + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df