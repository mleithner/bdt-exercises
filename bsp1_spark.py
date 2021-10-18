import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, when

# Small Spark exercise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bsp1_spark.py <input.csv>", file=sys.stderr)
        sys.exit(-1)

    start = time.perf_counter()

    # 1. Get session

    spark = SparkSession.builder.master('local').appName('Bsp1 Apache Spark').getOrCreate()
    end = time.perf_counter()
    print(f'1. Created a Spark session in {end-start}s')
    start = end

    # 2. Load dataset

    df = spark.read.format('csv').option('header', True).load(sys.argv[1])
    end = time.perf_counter()
    print(f'2. Loaded dataset in {end-start}s')
    start = end

    print('========\n*** Initial dataset')
    df.show(n=20)
    df.printSchema()
    record_count = df.count()
    print(f'We have {record_count} records')

    # Rename GEBURTSJAHR_HUND to GJ

    df = df.withColumnRenamed('GEBURTSJAHR_HUND', 'GJ')
    end = time.perf_counter()
    print(f'3. Column GEBURTSJAHR_HUND renamed to GJ in {end-start}s')
    start = end

    # Add synthetic ANZAHL column

    # There are tons of ways to do this, we could reuse the tuplify/reduceByKey
    # approach from class, but this is marginally more elegant.

    # Create a dataframe with two columns: HALTER_ID and ANZAHL
    group_df = (df
            .groupBy('HALTER_ID')
            .count()
            .withColumnRenamed('count', 'ANZAHL')
            )
    # Join it on the original dataframe
    df = df.join(group_df, 'HALTER_ID')

    end = time.perf_counter()
    print(f'4. Added synthetic ANZAHL column in {end-start}s')
    start = end

    # Collect the result

    print('========\n*** Transformed dataset')
    df.show(n=20)
    df.printSchema()

    print('Sanity check via ANZAHL column statistics - mean should be >= 1')
    df.select('ANZAHL').summary().show()

    result = df.collect()
    end = time.perf_counter()
    print(f'5. Collected result in {end-start}s')
    start = end

    # Count ratio of labrador retrievers and chihuahuas

    labrador_retriever_count = df.filter(
            ((df.RASSE1 == 'Labrador Retriever') & df.RASSE1_MISCHLING.isNull() & df.RASSE2.isNull()) |\
                    ((df.RASSE2 == 'Labrador Retriever') & df.RASSE2_MISCHLING.isNull() & df.RASSE1.isNull())
            ).count()
    print(f'Labrador retrievers: {labrador_retriever_count}/{record_count} = {100*labrador_retriever_count/record_count}%')

    chihuahua_count = df.filter(
            ((df.RASSE1 == 'Chihuahua') & df.RASSE1_MISCHLING.isNull() & df.RASSE2.isNull()) |\
                    ((df.RASSE2 == 'Chihuahua') & df.RASSE2_MISCHLING.isNull() & df.RASSE1.isNull())
            ).count()
    print(f'Chihuahuas: {chihuahua_count}/{record_count} = {100*chihuahua_count/record_count}%')

    # Shut down the Spark session
    spark.stop()
