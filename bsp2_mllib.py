import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, when
from pyspark.ml.feature import VectorAssembler
#from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Small MLlib exercise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bsp2_mllib.py <input.csv>", file=sys.stderr)
        sys.exit(-1)

    start = time.perf_counter()

    # 1. Get session

    spark = SparkSession.builder.master('local').appName('Bsp2 MLlib').getOrCreate()
    end = time.perf_counter()
    print(f'1. Created a Spark session in {end-start}s')
    start = end

    # 2. Load dataset

    df = (spark.read
            .format('csv')
            .option('header', True)
            .option('inferschema', True)
            .load(sys.argv[1])
            )
    end = time.perf_counter()
    print(f'2. Loaded dataset in {end-start}s')
    start = end

    print('========\n*** Initial dataset')
    df.show(n=5)
    df.printSchema()
    record_count = df.count()
    print(f'We have {record_count} records')

    # 3. Create a vector from independent variables

    vector_assembler = VectorAssembler(inputCols = [
        'calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars',
        'potass', 'vitamins', 'shelf', 'weight', 'cups'
        ], outputCol = 'features')

    transformed_df = (vector_assembler
            .transform(df)
            .select('features', 'rating')
            )
    transformed_df.show(n=5)
    transformed_df.printSchema()

    # 3. Split data set into roughly 80% training set and 20% test set

    (training_df, test_df) = transformed_df.randomSplit([0.8, 0.2])

    print(f'Training set: {training_df.count()} entries')
    print(f'Test set: {test_df.count()} entries')

    # 4. Fit a linear regression model on the data

    lr = LinearRegression(featuresCol = 'features', labelCol='rating',
            maxIter=100)

    lr_model = lr.fit(training_df)
    print(f'RMSE on training set: {lr_model.summary.rootMeanSquaredError}')
    print(f'r2 on training set: {lr_model.summary.r2}')

    # 5. Predict rating values in test set (and evaluate the prediction)

    lr_predictions = lr_model.transform(test_df)

    test_result = lr_model.evaluate(test_df)
    print(f'RMSE on test set: {test_result.rootMeanSquaredError}')

    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                             labelCol="rating",metricName="r2")
    print(f'r2 on test set: {lr_evaluator.evaluate(lr_predictions)}')

    lr_predictions.select("prediction","rating","features").show(n=100)

    # Shut down the Spark session
    spark.stop()
