---
title: Learn Pandas
updated: 2019-12-05 11:52
---
<style type="text/css">
	code{
		color: #000;
		background: #fff;
	}
</style>

1. What it is?
2. Different ways to create a dataframe(List, Dictionary, from files)
3. Different ways to access pandas column, row, and element
4. Oprations on rows and columns
5. Preprocessing and Data Cleaning
6. Useful pandas functions
7. Case studies 
8. Iris dataset example
9. Documentation

## What is it?
As per the documentation: Pandas DataFrameTwo-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. Here axis-0 represents rows and axis-1 represents column.
Simple put DataFrame can be seen as a list of columns. Let's see how to create a DataFrame.

## Different ways to create a dataframe
```py
import pandas as pd
# Create an empty Pandas DataFrame
df = pd.DataFrame()
print(df)
```
![](images/post1/1.png)

Signature of DataFrame() function looks like this : pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False).
Here data can be ndarray (structured or homogeneous), Iterable, dict, or DataFrame Dict can contain Series, arrays, constants, or list-like objects.
Now let's pass a list as data. How do you think it will interpret list as a column or row?
```py
# Create a Pandas DataFrame
data = [1,2,3,4,5] #
df = pd.DataFrame(data)
df
```
![](images/post1/2.png)

In previous example we saw that it considers list as a list of rows, so it breaks in into 5 rows.
What if we want to pass it as single row?
```py
# Create a Pandas DataFrame with single row and list of lists.
data = [[1,2,3,4,5]] 
df = pd.DataFrame(data)
df
```
![](images/post1/3.png)

Now it has only one element(a list) inside the list, hence it has only one row. Below example will make it clear.


```py
# Create a Pandas DataFrame with list of lists.
data = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]
df = pd.DataFrame(data)
df
```
![](images/post1/4.png)

```py
#With single column.and no index
cols = ['col1']
data = [1,2,3,4,5]
df = pd.DataFrame(data, columns=cols)
df
```
![](images/post1/5.PNG)


```py
#With single column.and no index
rows = ['row1', 'row2','row3','row4','row5']
cols = ['col1']
data = [1,2,3,4,5]
df = pd.DataFrame(data, columns=cols, index=rows)
df
```
![](images/post1/6.PNG)


```py
#How to create a dataframe with two different rows? First answer the following how many rows you want? say 5 then 
#you need 5 elements in the list that's it. 
#With single column.and no index
cols = ['col1', 'col2']
data = [[1,'a'],[2,'b'],[3,'c'],[4,'d'],[5,'e']] # 5 rows basically.
df = pd.DataFrame(data, columns=cols)
df
```
![](images/post1/7.PNG)


```py
#Only single datatypes allowed.
cols = ['col1', 'col2', 'col3']
data = [[1,11, 111],[2,22, 222],[3,33, 333]] # 3 rows basically.
df = pd.DataFrame(data, columns=cols, dtype=int)
df
```
![](images/post1/8.PNG)

```py
#Single column update
df['col1'] = df['col1'].astype(float)
df
```
![](images/post1/9.PNG)

```py
#By default it will update all columns
dtype_dict = {'col1': float, 'col2': str}
df = df.astype(dtype_dict)
df
```
![](images/post1/10.PNG)

```py
#None example
cols = ['col1', 'col2']
data = [[1,'a'],[2],[3,'c'],[4,'d'],[5]] # 5 rows basically.
df = pd.DataFrame(data, columns=cols)
df
```
![](images/post1/11.PNG)

```py
#Uses of zip
col1 = [1,2,3,4,5]
col2 = ['A','B','C','D','E']
col3 = ['a','b','c','d','e']
df = pd.DataFrame([col1, col2, col3])
df
```
![](images/post1/12.PNG)

```py
col1 = [1,2,3,4,5]
col2 = ['A','B','C','D','E']
data = list(zip(col1, col2))
df = pd.DataFrame(data)
df
```
![](images/post1/13.PNG)

## Dictionary to dataframe
```py
data = {'name':'Gitesh', 'Gender':'Male','Address':'Hyderabad'}
# df = pd.DataFrame(data)
# df
data = [{'name':'Gitesh', 'Gender':'Male','Address':'Hyderabad'}]
df = pd.DataFrame(data)
df
```
![](images/post1/14.PNG)

```py
data = [{'name':'Gitesh', 'Gender':'Male','Address':'Hyderabad'}, {'name':'Suresh', 'Gender':'Male','Address':'Hyderabad'}]
df = pd.DataFrame(data)
df
```
![](images/post1/15.PNG)

```py
data = {'name':['Gitesh','Suresh'], 'Gender':['Male','Male'],'Address':['Hyderabad', 'Hyd']}
df = pd.DataFrame(data)
df
```
![](images/post1/16.PNG)

## datatypes
```py
# All columns can be of different type as it's a list of columns and we know list can contains different kind of objects
df.dtypes
```
![](images/post1/17.PNG)

```py
df.col1.dtype
```
<code>
dtype('int64')
</code>
```py
df.shape
```
<code>
(62, 10)
</code>
```py
df.info()
```
![](images/post1/18.PNG)

```py
df.describe()
```
![](images/post1/19.PNG)

## Creating DataFrame from files like csv
```py
#Reading without header
filename = "Data/prospects3.csv"
df = pd.read_csv(filename)
df.head()
```
![](images/post1/20.PNG)

```py
#Passing header
filename = "Data/prospects3.csv"
colnames = ['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS', 'OWNS_CAR', 'ANNUAL_SALARY']
df = pd.read_csv(filename ,names=colnames, header=None)
df.head()
```
![](images/post1/21.PNG)

```py
filename = "Data/mpg.csv"
df = pd.read_csv('Data/mpg.csv')
df.head()
```
![](images/post1/22.PNG)

```py
df.dtypes
```
![](images/post1/23.PNG)

```py
col = ['num', 'capital', 'small']
col1 = [1,2,3,4,5,6,7,8,9,10]
col2 = ['A','B','C','D','E','F','G','H','I','J']
col3 = ['a','b','c','d','e','f','g','h','i','j']
data = list(zip(col1, col2, col3))
print(list(data))
df = pd.DataFrame(data, columns = col)
df
```
```
[(1, 'A', 'a'), (2, 'B', 'b'), (3, 'C', 'c'), (4, 'D', 'd'), (5, 'E', 'e'), (6, 'F', 'f'), (7, 'G', 'g'), (8, 'H', 'h'), (9, 'I', 'i'), (10, 'J', 'j')]
```
![](images/post1/24.PNG)

## slicing(head, tail, in place, out place)
```py
df.head()
```
![](images/post1/25.PNG)

```py
#Accessing columns using name
df['A']
type(df['A'])
```
```py
df['small'].values
```
<code>
array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], dtype=object)
</code>
```py
df['small'].tolist()
```
<code>
['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
</code>
```py
df[['capital','small']]
```
![](images/post1/26.PNG)

```py
df[['capital','small']].values.tolist()
```
![](images/post1/27.PNG)

```py
df
```
![](images/post1/28.PNG)

```py
#With rows
df[1:10:3].values.tolist()
```
<code>
[[2, 'B', 'b'], [5, 'E', 'e'], [8, 'H', 'h']]
</code>
## Column and Row addition, deletion
```py
df['newColumn'] = [0,0,0,0,0,0,0,0,0,0]
df
```
![](images/post1/29.PNG)

```py
df['AddedColum'] = df['capital'] + df['small']
df
```
![](images/post1/30.PNG)

```py
#With single column.and no index
cols = ['col1']
data = [1,2,3,4,5]
df = pd.DataFrame(data, columns=cols)
df
```
![](images/post1/31.PNG)

```py
# add Rows(overwriting)
df.iloc[0] = [7]
df
```
![](images/post1/32.PNG)

```py
#appending
new_row = pd.DataFrame({'col1':[8]}, index=[5])
new_row2 = pd.DataFrame({'col1':[8]})
new_row
```
![](images/post1/33.PNG)

```py
df.append(new_row)
```
![](images/post1/34.PNG)

```py
pd.concat([df, new_row2, new_row])
```
![](images/post1/35.PNG)

```py
df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df
```
![](images/post1/36.PNG)

```py
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df = df.append(df2)
df
```
![](images/post1/37.PNG)

## Deletion
```py
# Drop columns
df.drop(['A'],axis=1)
```
![](images/post1/38.PNG)

```py
# Drop both columns
df.drop(['A','B'],axis=1)
```
![](images/post1/39.PNG)

```py
df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df
```
![](images/post1/40.PNG)

```py
# Different ways
# df.drop(df.columns[[0,1]],axis=1)
# df.drop(columns=['B', 'C'])
# drop rows by index
df.drop([1, 1]) # drop row at index 1 to 1
```
![](images/post1/41.PNG)

## Data Cleaning
```py
filename = 'C:\\Users\\GJ250005\\Downloads\\Freelencing\\Chegg\\Solutions\\roux\\undergradSurvey.csv'
df = pd.read_csv(filename)
# df = df[['st_id', 'gender', 'age', 'class_st', 'major', 'grad intention', 'gpa', 'employment', 'salary', 'satisfaction']]
df.head()
```
![](images/post1/42.PNG)

```py
df.tail()
```
![](images/post1/43.PNG)
## Remove NA
```py
df.isnull().any()
```
![](images/post1/44.PNG)
```py
df.count() # If values are not same it means it has Nan
```
![](images/post1/45.PNG)
```py
df = df.dropna(axis=0,thresh = 2)# thresh means how many non-nan you are looking for.
#df = df.dropna(axis=0, subset=['age','age'])
#df = df.dropna(axis=1) # everything will be deleted
#df = df.dropna(axis=1, how = 'all') none of them will be deleted
#df = df.dropna(axis=1, thresh = 63)
#df = df.dropna(axis=1, thresh = 63)
df.tail()
```
![](images/post1/46.PNG)
## Dealing with categorical values
```py
x1 = df
d = {'Male':0, 'Female':1}
x1['gender'] = x1['gender'].apply(lambda x1:d[x1])
x1.head()
```
![](images/post1/47.PNG)
## Average out the missing values
```py
df['age']=df['age'].fillna(df['age'].mean())
df.tail()
```
![](images/post1/48.PNG)
## Case studies
```py
#Wine data
# cols = ['fixed acidity', 'volatile acidity','citric acid','residual sugar', 'chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
data = pd.read_csv("Data/winequality-white.csv", sep = ';')
df = pd.DataFrame(data)
print(df.shape)
print(df.columns)
df.head()
```
![](images/post1/49.PNG)
```py
# Data cleaning
filename = "Data/prospects3.csv"
colnames = ['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS', 'OWNS_CAR', 'ANNUAL_SALARY']
df = pd.read_csv(filename ,names=colnames, header=None)
df.head()
```
![](images/post1/50.PNG)
```py
# Clean it.
#check clean.py
#Signature:- lambda arg1,arg2..., argn : operation
def myfunc(OWNS_CAR):
    if(OWNS_CAR =='yes' or OWNS_CAR == 'y'):
        return 'y'
    else:
        return 'n'
```
```py
# df['OWNS_CAR'] = df.apply(lambda x: myfunc(x.OWNS_CAR), axis=1)
df['OWNS_CAR'] = df['OWNS_CAR'].apply(lambda x: myfunc(x))
# df['OWNS_CAR'] = df.apply(lambda x: myfunc(x['OWNS_CAR']),axis=1)
df.head()
```
![](images/post1/51.PNG)
## Iris
```py
iris = pd.read_csv('Data/iris.csv')
iris.head()
```
![](images/post1/52.PNG)

```py
#Get important informations
print("shape: \n", iris.shape)
print("columns: \n", iris.columns)
print("species counts : \n", iris['species'].value_counts())
```
![](images/post1/53.PNG)

```py
cols = ['sepal_length','sepal_width','petal_length','petal_width']
x = iris[cols]
x.head() 
```
![](images/post1/54.PNG)

```py
y = iris['species']
y.head()
```
![](images/post1/55.PNG)

```py
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
y_test.head()
```
![](images/post1/56.PNG)

```py
y_test.value_counts()
```
![](images/post1/57.PNG)


## miscellaneous
```py
df.T
```
![](images/post1/58.PNG)


```py
df['gender'].value_counts()
```

![](images/post1/59.PNG)


```py
df = pd.DataFrame({'A': range(4), 'B': [2*i for i in range(4)]})
df
```
![](images/post1/60.PNG)


```py
df['A'].corr(df['B'])
```
<code>
	1.0
</code>

```py
df.corr()
```
![](images/post1/61.PNG)


```py

filename = 'C:\\Users\\GJ250005\\Downloads\\Freelencing\\Chegg\\Solutions\\roux\\undergradSurvey.csv'
df = pd.read_csv(filename)
df = df[['st_id', 'gender', 'age', 'class_st', 'major', 'grad intention', 'gpa', 'employment', 'salary', 'satisfaction']]
df = df[df['st_id'].notnull()]
df.head()

```
![](images/post1/62.PNG)

