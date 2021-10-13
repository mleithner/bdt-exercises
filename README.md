# bdt-exercises

A few exercises for Spark, MLlib and TensorFlow

## Dependencies

You need PySpark and TensorFlow. You can either set up a virtualenv or install it system wide; the former is usually recommended.

### virtualenv

Run the following *in this folder*:

```
virtualenv -p python3 .
source bin/activate
pip install -r requirements.txt
```

This sets up a virtualenv, activates it and installs pyspark (and any other dependencies listed in requirements.txt).

**You must use `source bin/activate` whenever you want to use this virtualenv, otherwise pyspark will not be available.**

### Plain/system wide

```
pip install pyspark tensorflow
```


## Example usage

```
source bin/activate
python bsp1_spark.py data/20160307hundehalter.csv
```
