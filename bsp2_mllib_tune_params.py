import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import itertools

# Small MLlib exercise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bsp2_mllib.py <input.csv>", file=sys.stderr)
        sys.exit(-1)

    start = time.perf_counter()

    # 1. Get session

    spark = SparkSession.builder.master('local').appName('Bsp2 MLlib Linear Regression Parameter Tuning').getOrCreate()
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

    iters = [1, 10, 100, 1000, 10000]
    losses = ['squaredError', 'huber']
    regParams = [0.0, 0.5, 1.0] # Penalty parameter lambda
    elasticNetParams = [0.0, 0.5, 1.0] # Elastic net penalty alpha/lambda2

    rmse_min = float('inf') # Smallest observed RMSE; we start at infinity
    best_config = None

    for iterations, loss, regParam, elasticNetParam in itertools.product(
            *[iters, losses, regParams, elasticNetParams]
            ):

        print(f'Trying maxIter = {iterations} loss = {loss} regParam = {regParam} elasticNet = {elasticNetParam}')

        try:
            lr = LinearRegression(
                    featuresCol = 'features', labelCol='rating',
                    maxIter = iterations,
                    loss = loss,
                    regParam = regParam,
                    elasticNetParam = elasticNetParam)

            lr_model = lr.fit(training_df)
        except Exception as e:
            print(f'An error occurred, skipping this configuration: {str(e)}')
            continue

        print(f'RMSE on training set: {lr_model.summary.rootMeanSquaredError}')

        # 5. Predict rating values in test set (and evaluate the prediction)

        lr_predictions = lr_model.transform(test_df)

        test_result = lr_model.evaluate(test_df)
        print(f'RMSE on test set: {test_result.rootMeanSquaredError}')

        rmse_test = test_result.rootMeanSquaredError

        if rmse_test < rmse_min:
            best_config = [iterations, loss, regParam, elasticNetParam]
            rmse_min = rmse_test

    # Shut down the Spark session
    spark.stop()

    print(f'Best configuration (RMSE ~ {rmse_min}):')
    maxIter, loss, regParam, elasticNetParam = best_config
    print(f'maxIter = {maxIter}')
    print(f'loss = {loss}')
    print(f'regParam = {regParam}')
    print(f'elasticNetParam = {elasticNetParam}')
