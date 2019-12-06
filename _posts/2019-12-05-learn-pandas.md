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

Signature of DataFrame() function looks like this : ```pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)```.
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
col1 = [1,2,3,4,5]
col2 = ['A','B','C','D','E']
col3 = ['a','b','c','d','e']
data = list(zip(col1, col2, col3))
print(data)
df = pd.DataFrame(data)
df
```
![](images/post1/5.png)

Now Let's pass other arguments like row index and colum names as per the signature of DataFrame(data=None, index=None, columns=None, dtype=None, copy=False).
```py
# Create a Pandas DataFrame with row index or row names.
rows = ['row1', 'row2','row3','row4','row5']
data = [1,2,3,4,5]
df = pd.DataFrame(data, index=rows)
df
```
![](images/post1/6.png)

```py
# Create a Pandas DataFrame with column names.
cols = ['col1']
data = ['a','b','c','d','d']
df = pd.DataFrame(data, columns=cols)
df
```
![](images/post1/7.png)

```py
# Create a Pandas DataFrame with rows and column names.
rows = ['row1', 'row2','row3','row4','row5']
cols = ['col1', 'col2']
data = [[1,'a'],[2,'b'],[3,'c'],[4,'d'],[5,'e']] # 5 rows basically.
df = pd.DataFrame(data, columns=cols, index = rows)
df
```
![](images/post1/8.png)

We can pass datatype for the columns, but only one datatype can be passed so make sure all the columns are compatible with the datatype you are passing.

```py
# Create a Pandas DataFrame with datatypes.
cols = ['col1', 'col2', 'col3']
data = [[1,11, 111],[2,22, 222],[3,33, 333]] # 3 rows.
df = pd.DataFrame(data, columns=cols, dtype=float)
df
```
![](images/post1/9.png)

Pandas modify the datatype of each columns.

```py
#By default it will update all columns even if you pass it form only one column to the right.
dtype_dict = {'col2': int, 'col3': int}
df = df.astype(dtype_dict)
df
```
![](images/post1/10.png)

Pandas None values.

```py
#None example
cols = ['col1', 'col2']
data = [[1,'a'],[2],[3,'c'],[4,'d'],[5]] # 5 rows basically.
df = pd.DataFrame(data, columns=cols)
df
```
![](images/post1/11.png)

## create pandas DataFrame from dictionary object.

```py
data = [{'name':'Gitesh', 'Gender':'Male','Address':'Hyderabad'}]
df = pd.DataFrame(data)
df
```
![](images/post1/12.png)

```py
data = [{'name':'Gitesh', 'Gender':'Male','Address':'Hyderabad'}, {'name':'Suresh', 'Gender':'Male','Address':'Hyderabad'}]
df = pd.DataFrame(data)
df
```
![](images/post1/13.png)


```py
data = {'name':['Gitesh','Suresh'], 'Gender':['Male','Male'],'Address':['Hyderabad', 'Hyd']}
df = pd.DataFrame(data)
df
```
![](images/post1/14.png)

```py
data = {'name':'Gitesh', 'Gender':'Male','Address':'Hyderabad'}
df = pd.DataFrame(data, index=[1,2,3])
df
```
![](images/post1/15.png)

## Creating DataFrame from files like csv.
```py
#Reading without header
filename = "Data/prospects3.csv"
df = pd.read_csv(filename)
df.head()
```
![](images/post1/16.png)

```py
#Passing header
filename = "Data/prospects3.csv"
colnames = ['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS', 'OWNS_CAR', 'ANNUAL_SALARY']
df = pd.read_csv(filename ,names=colnames, header=None)
df.head()
```
![](images/post1/17.png)

```py
filename = "Data/mpg.csv"
df = pd.read_csv('Data/mpg.csv') # This file already have headers.
df.head()
```
![](images/post1/18.png)

## Different ways to access pandas column, row, and element.

```py
data = [[1,2,3],[10,20,30],[100,200,300], [111, 222, 333]]
cols = ['col1', 'col2', 'col3']
df = pd.DataFrame(data, columns = cols)
df
```
![](images/post1/19.png)

Pandas DataFrame access the column using its name.

```py
df['col1']
```
![](images/post1/20.png)

Aceesing the values of columns in Pandas DataFrame.

```py
df['col1'].values
```
![](images/post1/21.png)

Get the pandas column value in list.

```py
df['col1'].tolist()
```
![](images/post1/22.png)

Pandas DataFrame access mutliple column using its name.

```py
df[['col1','col2']]
```
![](images/post1/23.png)

```py
df[['col1','col2']].values.tolist()
```
![](images/post1/24.png)

Pandas DataFrame access all columns using its name and store it inside a list.

```py
ll = []
for col in df.columns:
    l = df[col].values.tolist()
    ll.append(l)
ll
```
![](images/post1/25.png)

```py
df
```
![](images/post1/26.png)

## Accessing pandas DataFrame rows and slicing.
Here the syntax to access pandas row is ```df[i:j:k]``` where i is the start index of row, j is last index(excluded) and k is the step. For example ```df[0:6:2]``` - This will return first row(0th index start) then skip k-1 rows so here (2-1)=1 rows will be skiped after the current rows so next rows it will return is row with index 2 then 4. It will not include row number 6 because it is not inclsive.
If we don't specify k by default it is 1.

```py
df[1:4]
```
![](images/post1/27.png)

```py
df[1:4:2]
```
![](images/post1/28.png)

```py
df[1:4:2].values.tolist()
```
![](images/post1/29.png)

## Oprations on rows and columns.
Pandas DataFrame create or add or apend new column. Make sure new column length matches with the existing column length.

```py
df['newColumn'] = [0,0,0,0]
df
```
![](images/post1/30.png)

Pandas DataFrame create new column by adding two existing columns.

```py
df['AddedColum'] = df['col1'] + df['col2']
df
```
![](images/post1/31.png)

```py
data = [[1,2,3],[10,20,30],[100,200,300], [111, 222, 333]]
cols = ['col1', 'col2', 'col3']
df = pd.DataFrame(data, columns = cols)
df
```
![](images/post1/32.png)

Pandas DataFrame overwrite the existing row.

```py
df.iloc[0] = [7]
df
```
![](images/post1/33.png)

```py
df.iloc[0] = [7,8,9]
df
```
![](images/post1/34.png)

Pandas DataFrame appending new row. we can append only Series and DataFrame objs.

```py
new_row = pd.DataFrame({'col1':[5],'col2':[6],'col3':[7]})
df.append(new_row)
```
![](images/post1/35.png)

Pandas DataFrame appending new row with index.

```py
new_row = pd.DataFrame({'col1':[5],'col2':[6],'col3':[7]}, index = [4])
df.append(new_row)
```
![](images/post1/36.png)

Concatination of multiple pandas DataFrame.

```py
row1 = pd.DataFrame({'col1':[5],'col2':[6],'col3':[7]}, index = [4])
row2 = pd.DataFrame({'col1':[8],'col2':[9],'col3':[10]}, index = [5])
pd.concat([df, row1, row2])
```
![](images/post1/37.png)

Pandas DataFrame deleting a column. We can pass inplace = True. 

```py
df.drop(['col1'],axis=1)
```
![](images/post1/38.png)

Pandas DataFrame deleting multiple columns

```py
df.drop(['col1','col2'],axis=1)
```
![](images/post1/39.png)

```py
df.drop(df.columns[[0,1]],axis=1)
```
![](images/post1/40.png)

```py
df
```
![](images/post1/41.png)

Pandas DataFrame deleting rows.

```py
df.drop([0,1,3]) # deleting 0th, 1st and 3rd row by index.
```
![](images/post1/42.png)

## Preprocessing and Data Cleaning.

```py
filename = 'Data\\undergradSurvey.csv'
df = pd.read_csv(filename)
df.head()
```
![](images/post1/43.png)

We can observe there are many NaN values inside some colums.

```py
df.tail()
```
![](images/post1/44.png)

Pandas DataFrame check how many columns have NaN values.

```py
df.isnull().any() # True means it contains NaN vlaues. So here all the columns have at least one NaN.
```
![](images/post1/45.png)

Pandas DataFrame check count of NaN values inside each column.

```py
df.isnull().sum()
```
![](images/post1/46.png)

Now we know everything about the presence of NaN values in our data, let's remove it.Pandas DataFrame remove the rows having NaN values in at least on column.

```py
df.dropna(axis=0).tail()# This remove all the rows having NaN vlaue in any of the column.
```
![](images/post1/47.png)

Pandas DataFrame remove the rows having NaN values in selected columns.
```py
df.dropna(axis=0, subset=['age']).tail() # it will remove the row only when NaN exist in age column, we specify multiple columns.
```
![](images/post1/48.png)

```py
df.dropna(axis=0, subset=['age', 'gender']).tail() # it will remove the row if any of the given column(age, gender) contains NaN.
```
![](images/post1/49.png)

Pandas DataFrame remove the rows having NaN values in selected columns.

```py
df.dropna(axis=0,thresh = 1).tail() # Here it will delete the row only if it have less then 1 non-NaN values.
```
![](images/post1/50.png)

Pandas DataFrame remove the column based on the number of NaN values in the column.It will search for only one NaN inside a column if it exists it will delete that column.

```py
df.dropna(axis=1).head() # It has deleted all the columns.
```
![](images/post1/51.png)

Pandas DataFrame remove the column based on the number of NaN values in the column.It will check if a column contains all the values as NaN, that column will be dropped. We can thresh here too.

```py
df.dropna(axis=1, how = 'all').head()
```
![](images/post1/52.png)

## Dealing with categorical values.

```py
x1 = df.dropna(axis=0) # creating the dataframe into x1 after removing all the NaN rows. 
d = {'Male':0, 'Female':1}
x1['gender'] = x1['gender'].apply(lambda x:d[x])# This will assign Male to 0 and Female to 1
x1.head()
```
![](images/post1/53.png)

## Average out the missing values.
```py
df['age'].tail()
```
![](images/post1/54.png)

```py
df['age'].fillna(df['age'].mean()).tail()
```
![](images/post1/55.png)

## Useful pandas functions.

```py
print("each columns datatypes: \n",df.dtypes)
print("age column's datatype: \n",df.age.dtypes)
print("shape of the DataFrame(row*col) is : ",df.shape)
print("Distribution of different age in the given data:\n",df['age'].value_counts())
```
![](images/post1/56.png)

```py
df.info()
```
![](images/post1/57.png)

```py
df.describe()
```
![](images/post1/58.png)
