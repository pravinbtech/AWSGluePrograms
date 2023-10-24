# Databricks notebook source

dbutils.fs.ls("s3://raja-engineering-spark-training/Week_Training/")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col,sum,when,count,expr
spark=SparkSession.builder.getOrCreate()
data=[
    (10,20,58),
    (20,10,12),
    (10,30,20),
    (30,40,100),
    (30,40,200),
    (30,40,200),
    (30,40,500)
]
df=spark.createDataFrame(data,["from_id","to_id","duration"])

col=df.columns

swap_df=(df.withColumn("PERSON1", when(df.from_id>df.to_id,df.to_id).otherwise (df.from_id))
    .withColumn("PERSON2", when(df.from_id>df.to_id,df.from_id).otherwise (df.to_id))
    .select("PERSON1","PERSON2","DURATION"))
     

res_df=(swap_df.groupBy("PERSON1","PERSON2")
       .agg(sum("DURATION").alias("total_duration"),count("DURATION").alias("COUNT"))).show()


# COMMAND ----------

df.createOrReplaceTempView("calls")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT LEAST(from_id,to_id) as person1,
# MAGIC GREATEST(from_id,to_id ) as person2,
# MAGIC COUNT(*) as call_count,
# MAGIC SUM(duration) as total_duration
# MAGIC FROM CALLS
# MAGIC GROUP BY 1,2

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE CALLS;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE employees
# MAGIC    (name STRING, dept STRING, salary INT, age INT);

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO employees
# MAGIC    VALUES ('Lisa', 'Sales', 10000, 35),
# MAGIC           ('Evan', 'Sales', 32000, 38),
# MAGIC           ('Fred', 'Engineering', 21000, 28),
# MAGIC           ('Alex', 'Sales', 30000, 33),
# MAGIC           ('Tom', 'Engineering', 23000, 33),
# MAGIC           ('Jane', 'Marketing', 29000, 28),
# MAGIC           ('Jeff', 'Marketing', 35000, 38),
# MAGIC           ('Paul', 'Engineering', 29000, 23),
# MAGIC           ('Chloe', 'Engineering', 23000, 25);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT name,
# MAGIC          salary,
# MAGIC          dept,
# MAGIC          LAG(salary) OVER (PARTITION BY dept ORDER BY salary) AS lag,
# MAGIC          LEAD(salary, 3, 0) OVER (PARTITION BY dept ORDER BY salary) AS lead
# MAGIC     FROM employees;
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import lead,lag,when,col,coalesce
from pyspark.sql.window import Window
from pyspark.sql.types import StructType,StructField,StringType,IntegerType
data=[(1,'Alice'),(2,'Bob'),(3,'Charlie'),(4,'David'),(5,'Eric')]
schema=StructType([
    StructField("id",IntegerType(),True),
    StructField("Name",StringType(),True)
])
df=spark.createDataFrame(data,schema)
new_df=(df.withColumn("Lead_Student",lead('Name',3).over(Window.orderBy('id')))
        .withColumn("Lag_Student",lag('Name',3).over(Window.orderBy('id'))))

new_df=new_df.withColumn("Exchanged_Seating",
                                               when (new_df['id']%2==1,coalesce(new_df['Lead_Student'],new_df['Name']))
                                              .when(new_df['id']%2==0,coalesce(new_df['Lag_Student'],new_df['Name']))
                                              .otherwise (new_df['Name'] )
                                              ) 

new_df=new_df.drop(new_df['Lead_Student'],new_df['Lag_Student'])
new_df.show()

# COMMAND ----------

from pyspark.sql.functions import lead,lag,when,first,last
from pyspark.sql.window import Window
data=[("2020-06-01","Won"),
      ("2020-06-02","Won"),
      ("2020-06-03","Won"),
      ("2020-06-04","Lost"),
      ("2020-06-05","Lost"),
      ("2020-06-06","Lost"),
      ("2020-06-07","Won")]

df=spark.createDataFrame(data,["Event_Date","Event_Status"]);
df.show()
df=df.withColumn("Lag",lag(df['Event_Status']).over(Window.orderBy('Event_Date')))
df=df.withColumn("Group_ID",when(df['Event_Status']!=df['Lag'],1).otherwise(0))
df=df.withColumn("Running_Total",sum(df['Group_ID']).over(Window.orderBy('Event_Date')))
df=df.drop("Lag","Group_ID")
res_df=df.groupBy("Event_Status","Running_Total")\
         .agg(first("Event_Date").alias("Start_Date"),last("Event_Date").alias("End_Date")).drop("Running_Total").orderBy(col("Start_Date").desc())
        
res_df.show()

# COMMAND ----------

aws_bucket_name = "raja-engineering-spark-training"
mount_name = "SparkTraining-Raja"
dbutils.fs.mount(f"s3a://{aws_bucket_name}", f"/mnt/{mount_name}")
display(dbutils.fs.ls(f"/mnt/{mount_name}"))


# COMMAND ----------

display(dbutils.fs.ls(f"/mnt"))

# COMMAND ----------


full_df=(spark.read.format("csv").option("inferSchema",True).option("header",True).option("sep",",").load("dbfs:/mnt/SparkTraining-Raja/Week_Training/counties.csv"))
#full_df.show()
start_df=(spark.read.format("csv").option("inferSchema",True).option("header",True).option("sep",",").option("skipRows",10).load("dbfs:/mnt/SparkTraining-Raja/Week_Training/counties.csv"))
end_df=(spark.read.format("csv").option("inferSchema",True).option("header",True).option("sep",",").option("skipRows",20).load("dbfs:/mnt/SparkTraining-Raja/Week_Training/counties.csv"))
#start_df.show()
my_df=full_df.subtract(start_df);
my_df=my_df.union(end_df);
my_df.orderBy('county_number').show()

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.types import ArrayType
df=spark.sql("show databases")
df1=spark.sql("show tables").show()
df2=spark.sql("select * from employees")

new_df2=df2.withColumn("First",row_number().over(Window.partitionBy("dept").orderBy("name")))
new_df2.show()




# COMMAND ----------

from pyspark.sql.functions import explode_outer
def flatten(df):
   # compute Complex Fields (Lists and Structs) in Schema   
   complex_fields = dict([(field.name, field.dataType)
                             for field in df.schema.fields
                             if type(field.dataType) == ArrayType or  type(field.dataType) == StructType])
   print(complex_fields)
   while len(complex_fields)!=0:
      col_name=list(complex_fields.keys())[0]
      print ("Processing :"+col_name+" Type : "+str(type(complex_fields[col_name])))
    
      # if StructType then convert all sub element to columns.
      # i.e. flatten structs
      if (type(complex_fields[col_name]) == StructType):
         expanded = [col(col_name+'.'+k).alias(col_name+'_'+k) for k in [ n.name for n in  complex_fields[col_name]]]
         df=df.select("*", *expanded).drop(col_name)
    
      # if ArrayType then add the Array Elements as Rows using the explode function
      # i.e. explode Arrays
      elif (type(complex_fields[col_name]) == ArrayType):    
         df=df.withColumn(col_name,explode_outer(col_name))
    
      # recompute remaining Complex Fields in Schema       
      complex_fields = dict([(field.name, field.dataType)
                             for field in df.schema.fields
                             if type(field.dataType) == ArrayType or  type(field.dataType) == StructType])
   return df

# COMMAND ----------

from pyspark.sql.types import ArrayType,StructType
from pyspark.sql.functions import explode,coalesce,posexplode,explode_outer

df=spark.read.option("multiline","true").option("inferSchema",True).json("s3://raja-engineering-spark-training/Week_Training/sample_json.json")

#df.printSchema()


df=df.withColumn('Actors',explode_outer('Actors'))
display(df)
complex_fields = dict([(field.name, field.dataType)
                             for field in df.schema.fields
                             if type(field.dataType) == ArrayType or  type(field.dataType) == StructType])
print(complex_fields)
    
    
#display(flat_df)


#df2.printSchema()
df3=df.select("Actors.*")
#df3=df2.posexplode(col)
df3.show()
df3.coalesce(1).write.json("s3://raja-engineering-spark-training/Week_Training/split_json")


# COMMAND ----------

my_df=spark.read.option("multiline","true").option("inferSchema",True).json("s3://raja-engineering-spark-training/Week_Training/split_json/*.json")
my_df.show()

# COMMAND ----------

from pyspark.sql.functions import sum,count

players_data=[(1,'Nadal'),(2,'Federer'),(3,'Novak')]
players_columns=['player_id','player_name']
players_df=spark.createDataFrame(players_data,players_columns)
players_df.show()

championships_data=[(2017,2,1,1,2),(2018,3,1,3,2),(2019,3,1,1,3)]
championship_columns=['year','wimbledon','french_open','us_open','au_open']
championship_df=spark.createDataFrame(championships_data,championship_columns)
championship_df.show()

df1=championship_df.select("year","wimbledon")
df2=championship_df.select("year","french_open")
df3=championship_df.select("year","us_open")
df4=championship_df.select("year","au_open")

df5=df1.unionAll(df2).unionAll(df3).unionAll(df4)
df6=df5.groupBy('wimbledon').agg(count("*").alias("grandSlamsCount"))
result_df=df6.join(players_df,df6.wimbledon==players_df.player_id,"inner").select("player_id","player_name","grandSlamsCount")
display(result_df)


# COMMAND ----------



# COMMAND ----------

df1 = spark.createDataFrame([[1, 2, 3]], ["col0", "col1", "col2"])
df2 = spark.createDataFrame([[4, 5, 6]], ["col1", "col2", "col3"])
df1.unionByName(df2, allowMissingColumns=True).show()


# COMMAND ----------

from pyspark.sql.types import IntegerType,Row,StringType,StructField,StructType
df1_marks=[70,80,90]

df2_names=["a","b","c"]
names=StructType([StructField("name", StringType(), True)])
marks=StructType([StructField("mark", IntegerType(), True)])
#df3=spark.createDataFrame(zip(df1_marks,df2_names),["marks","names"])

ref_marks=map(lambda x: [x],df1_marks)
ref_names=map(lambda x: [x],df2_names)
#df3.show()
df5=spark.createDataFrame(ref_marks,"marks:int")
df6=spark.createDataFrame(ref_names,"names:string")
df5.show()
df6.show()

df3_marks=df3.select('marks').rdd.map(lambda x:x[0]).collect()
df3_name=df3.select('name').rdd.map(lambda x:x[0]).collect()
print(df3_marks)
df4=spark.createDataFrame(zip(df3_marks,df3_name),["marks","name"])
#df4.show()


# COMMAND ----------

dbutils.fs.ls("s3://raja-engineering-spark-training/Week_Training/")

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/product.json")

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import *
df=spark.read.option("multiline","true").json("dbfs:/FileStore/tables/product.json")
#display(df)
w=Window().orderBy(lit('A'))
new_df=df.select(df['*'],explode(df.products).alias("JSONEXAMPLES"))
display(new_df)
my_Str_df=new_df.select(to_json("JSONEXAMPLES").alias("MYJSON")).select(json_tuple("MYJSON","brand","description"))
display(my_Str_df)
#display(new_df.select(json_tuple("JSONEXAMPLES","brand","category")))
#df_with_num=new_df.select(col("JSONEXAMPLES.*"))
#df_with_row_num=df_with_num.withColumn("RowId",row_number().over(w))


# COMMAND ----------

import plotly.graph_objects as go
import plotly.io as io
fig = go.Figure(
    data=[go.Bar(y=[2, 1, 3])],
    layout_title_text="A Figure Displaying Itself"
)

fig.write_image(file="my_figure.png", format="png")


# COMMAND ----------

dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------



# COMMAND ----------

# MAGIC %sh
# MAGIC pip install chart_studio

# COMMAND ----------

smtp_server = 'smtp.gmail.com' # for e.g. 'smtp.sendgrid.net'
smtp_port = 465
smtp_user = 'pravinbtech@gmail.com'
smtp_password = 'rvze bxbv mmmc jvux'
email_sender = 'pravinbtech@gmail.com'
email_receiver = 'priya.aaru1603@gmail.com'

# COMMAND ----------

def build_html_email_body(dataframes_list, pre_html_body = '', post_html_body = '', custom_css='', custom_css_class=None, max_rows_per_df=10):

    '''
    Author : Omkar Konnur
    License : MIT

    This function helps to compose an HTML email from your dataframes. Offers a few necessary customizations. Can be extended per your requirement.

    dataframes_list     : list of dataframes to be sent in the email. for e.g [df1,df2,...]
    pre_html_body       : Any html to be appended before the dataframe is displayed in the email. For e.g. '<p>Hi,</p>'
    post_html_body      : Any html to be appended in the end of the email. For e.g. Signatures, closing comments, etc.
    custom_css          : To format the table. Simply, this is the content of your CSS file. Note that the next parameter should pass the class defined in this CSS file.
    custom_css_class    : Single class used to modify the table CSS. This can be done as shown in the doc above
    max_rows_per_df     : Number of records in the dataframe sent in the email. Defaults to 10

    Please note that not all CSS works in an email. To check the latest supported CSS please refer - https://developers.google.com/gmail/design/reference/supported_css
    '''

    html_sep = '<br>'
    html_body = '<html><head><style>' + custom_css +'</style></head><body>' + "Hello welcome to Email from Pravinbtech@gmail.com your current usage charge is $100"
    #html_body+=myhtml
   

    for df in dataframes_list:
        df_count = df.count()
        html_body += f'''
                        <div class = 'table_header'>
                            <h3>Dataframe Total Count : {df_count}</h3>
                            <p> SHOWING MAX {max_rows_per_df} RECORDS FROM THE DATAFRAME </p>
                        </div>
                     '''
        html_body += f'''
                        <div class='table_wrapper'>
                            {df.limit(max_rows_per_df).toPandas().to_html(classes=custom_css_class)}
                        </div>
                     ''' + html_sep

    #html_body+=post_html_body+'</body></html>'
    print(myhtml)
    html_body=myhtml

    return html_body


def sendmail_html(smtp_server, smtp_port, smtp_user, smtp_password, sender_email, receiver_email, email_subject, email_body):
    import smtplib, ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from datetime import datetime

    '''
    Author : Omkar Konnur
    License : MIT

    This function sends email from your python environment. Accepts message type as HTML.

    Usually the SMTP server details will be shared by your organization.
    For testing, you can use your gmail account or use free email services like SendGrid. (https://sendgrid.com)

    smtp_server        : SMTP Server (for e.g smtp.sendgrid.net)
    smtp_port          : SMTP Port
    smtp_user          : SMTP User
    smtp_password      : SMTP User

    sender_email       : Sender's email. Please verify the domains allowed by your SMTP server
    receiver_email     : Receiver's email. In case of multiple recipients, provide a semi-colon seperated string with different emails
    email_subject      : Subject Line. This function has been configured to pre-pend your Subject line with Timestamp and Cluster Name
    email_body         : HTML string
    '''

    email_subject = f"{email_subject}"

    email_message = MIMEMultipart()
    email_message['From'] = sender_email
    email_message['To'] = receiver_email
    email_message['Subject'] = email_subject

    email_message.attach(MIMEText(email_body, "html"))
    email_string = email_message.as_string()

    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=ssl.create_default_context()) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(sender_email, receiver_email, email_string)

# COMMAND ----------

email_body = build_html_email_body([],custom_css=custom_table_style, custom_css_class='custom_table',max_rows_per_df=20)

sendmail_html(smtp_server, smtp_port, smtp_user, smtp_password,
            email_sender, email_receiver, 'My Awesome Dataframe with Custom Styling!',
            email_body
           )

# COMMAND ----------

custom_table_style = '''
*{
    box-sizing: border-box;
    -webkit-box-sizing: border-box;
    -moz-box-sizing: border-box;
}
body{
    font-family: Helvetica;
    -webkit-font-smoothing: antialiased;
}

.table_header {
    text-align: center;
    background-color: #efefef;
    padding: 10px;
    margin: 0px 70px 0px 70px;
    border-top-left-radius: 15px;
    border-top-right-radius: 15px;
}

.table_header p {
    color: #9a8c98;
    font-weight: light;
}

.table_wrapper {
    margin: 0px 70px 10px 70px;
}

.custom_table {
    border-radius: 5px;
    font-size: 12px;
    font-weight: normal;
    border: thin solid #f2e9e4;
    border-collapse: collapse;
    width: 100%;
    max-width: 100%;
    white-space: nowrap;
    background-color: #f2e9e4;
    word-break: break-all;
    word-wrap: break-word;
}

.custom_table td, .custom_table th {
    text-align: center;
    padding: 8px;
}

.custom_table td {
    font-size: 12px;
}

.custom_table thead th {
    color: #edede9;
    background: #22223b;
}
'''
