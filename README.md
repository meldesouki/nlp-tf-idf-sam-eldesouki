# nlp_tf_idf_hadoop

NLP analysis of Term Frequency:

Inverse Document Frequency using Hadoop

Sam_Eldesouki: Ethan Sam and Merna Eldesouki

## Requirements

- Apache Spark
  - have `pyspark` on path
- Python 3
  - Note 3.8 and above do not work well with pyspark
- Import these Python Packages in environment:
  - math
  - re
  - sys


  To get pyspark:
  1 - pip3 install pyspark
  (If on windows, add SPARK_HOME environment variable)

  To get hadoop
  1 -  

##Note：
- F-strings require at least python3.6 to run correctly
- Python3.8 and above tends to cause issues for pyspark
- Python3.7 tends to work best

To run this project:

-Using Ubuntu bash

```bash
python3.7 spark-submit app.py <file_to_parse> <query_term>
cat output
```

- Abstraction

```bash
make FILE=<file_to_parse> QUERY=<query_term>
```

## Notes
- Output will be written to output.txt
