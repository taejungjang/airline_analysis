## 목적 : 비행기 승객의 만족도 분석

데이터 : kaggle의 비행기 승객 만족도 설문조사 데이터

라이브러리 및 데이터 불러오기


```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
```


```python
path = 'C:/Users/TS/Downloads/archive/train.csv'
df_train = pd.read_csv(path,index_col = 0)
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>...</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70172</td>
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>13</td>
      <td>Personal Travel</td>
      <td>Eco Plus</td>
      <td>460</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5047</td>
      <td>Male</td>
      <td>disloyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>235</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>110028</td>
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>26</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>1142</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24026</td>
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>562</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>9.0</td>
      <td>neutral or dissatisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>119299</td>
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>61</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>214</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>satisfied</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



## 탐색적 데이터 분석(EDA)
데이터 특성을 파악하여 정확한 분석을 하기 위해 수행


```python
print("shape : ",df_train.shape)
print("colunms : ", df_train.columns)
```

    shape :  (103904, 24)
    colunms :  Index(['id', 'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
           'Flight Distance', 'Inflight wifi service',
           'Departure/Arrival time convenient', 'Ease of Online booking',
           'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
           'Inflight entertainment', 'On-board service', 'Leg room service',
           'Baggage handling', 'Checkin service', 'Inflight service',
           'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
           'satisfaction'],
          dtype='object')
    


```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 103904 entries, 0 to 103903
    Data columns (total 24 columns):
     #   Column                             Non-Null Count   Dtype  
    ---  ------                             --------------   -----  
     0   id                                 103904 non-null  int64  
     1   Gender                             103904 non-null  object 
     2   Customer Type                      103904 non-null  object 
     3   Age                                103904 non-null  int64  
     4   Type of Travel                     103904 non-null  object 
     5   Class                              103904 non-null  object 
     6   Flight Distance                    103904 non-null  int64  
     7   Inflight wifi service              103904 non-null  int64  
     8   Departure/Arrival time convenient  103904 non-null  int64  
     9   Ease of Online booking             103904 non-null  int64  
     10  Gate location                      103904 non-null  int64  
     11  Food and drink                     103904 non-null  int64  
     12  Online boarding                    103904 non-null  int64  
     13  Seat comfort                       103904 non-null  int64  
     14  Inflight entertainment             103904 non-null  int64  
     15  On-board service                   103904 non-null  int64  
     16  Leg room service                   103904 non-null  int64  
     17  Baggage handling                   103904 non-null  int64  
     18  Checkin service                    103904 non-null  int64  
     19  Inflight service                   103904 non-null  int64  
     20  Cleanliness                        103904 non-null  int64  
     21  Departure Delay in Minutes         103904 non-null  int64  
     22  Arrival Delay in Minutes           103594 non-null  float64
     23  satisfaction                       103904 non-null  object 
    dtypes: float64(1), int64(18), object(5)
    memory usage: 19.8+ MB
    


```python
#기술분석
df_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Age</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>On-board service</th>
      <th>Leg room service</th>
      <th>Baggage handling</th>
      <th>Checkin service</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103904.000000</td>
      <td>103594.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>64924.210502</td>
      <td>39.379706</td>
      <td>1189.448375</td>
      <td>2.729683</td>
      <td>3.060296</td>
      <td>2.756901</td>
      <td>2.976883</td>
      <td>3.202129</td>
      <td>3.250375</td>
      <td>3.439396</td>
      <td>3.358158</td>
      <td>3.382363</td>
      <td>3.351055</td>
      <td>3.631833</td>
      <td>3.304290</td>
      <td>3.640428</td>
      <td>3.286351</td>
      <td>14.815618</td>
      <td>15.178678</td>
    </tr>
    <tr>
      <th>std</th>
      <td>37463.812252</td>
      <td>15.114964</td>
      <td>997.147281</td>
      <td>1.327829</td>
      <td>1.525075</td>
      <td>1.398929</td>
      <td>1.277621</td>
      <td>1.329533</td>
      <td>1.349509</td>
      <td>1.319088</td>
      <td>1.332991</td>
      <td>1.288354</td>
      <td>1.315605</td>
      <td>1.180903</td>
      <td>1.265396</td>
      <td>1.175663</td>
      <td>1.312273</td>
      <td>38.230901</td>
      <td>38.698682</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>31.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32533.750000</td>
      <td>27.000000</td>
      <td>414.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>64856.500000</td>
      <td>40.000000</td>
      <td>843.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>97368.250000</td>
      <td>51.000000</td>
      <td>1743.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>12.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>129880.000000</td>
      <td>85.000000</td>
      <td>4983.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1592.000000</td>
      <td>1584.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#결측치 비율
df_train.isnull().sum()/len(df_train)  *100
```




    id                                   0.000000
    Gender                               0.000000
    Customer Type                        0.000000
    Age                                  0.000000
    Type of Travel                       0.000000
    Class                                0.000000
    Flight Distance                      0.000000
    Inflight wifi service                0.000000
    Departure/Arrival time convenient    0.000000
    Ease of Online booking               0.000000
    Gate location                        0.000000
    Food and drink                       0.000000
    Online boarding                      0.000000
    Seat comfort                         0.000000
    Inflight entertainment               0.000000
    On-board service                     0.000000
    Leg room service                     0.000000
    Baggage handling                     0.000000
    Checkin service                      0.000000
    Inflight service                     0.000000
    Cleanliness                          0.000000
    Departure Delay in Minutes           0.000000
    Arrival Delay in Minutes             0.298352
    satisfaction                         0.000000
    dtype: float64



### id컬럼 제거
단순 식별자로써 만족도와 관계없음


```python
df_train.drop(['id'],axis = 1, inplace = True)
```

### 결측치처리
출발지연시간과 도착지연시간은 0.9의 높은 상관관계이므로 적은 결측치이지만 제거하지 않고 "회귀대치"를 수행


```python
df_train[['Arrival Delay in Minutes','Departure Delay in Minutes']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Arrival Delay in Minutes</th>
      <th>Departure Delay in Minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Arrival Delay in Minutes</th>
      <td>1.000000</td>
      <td>0.965481</td>
    </tr>
    <tr>
      <th>Departure Delay in Minutes</th>
      <td>0.965481</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train_no_na = df_train.dropna(subset=['Arrival Delay in Minutes'])
x = df_train_no_na[['Departure Delay in Minutes']]
x = sm.add_constant(x)
y = df_train_no_na[['Arrival Delay in Minutes']]

model = sm.OLS(y,x).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>Arrival Delay in Minutes</td> <th>  R-squared:         </th>  <td>   0.932</td>  
</tr>
<tr>
  <th>Model:</th>                       <td>OLS</td>           <th>  Adj. R-squared:    </th>  <td>   0.932</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>Least Squares</td>      <th>  F-statistic:       </th>  <td>1.423e+06</td> 
</tr>
<tr>
  <th>Date:</th>                 <td>Mon, 23 Feb 2026</td>     <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                     <td>17:50:36</td>         <th>  Log-Likelihood:    </th> <td>-3.8635e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>          <td>103594</td>          <th>  AIC:               </th>  <td>7.727e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>              <td>103592</td>          <th>  BIC:               </th>  <td>7.727e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>                  <td>     1</td>          <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>          <td>nonrobust</td>        <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                      <td>    0.7224</td> <td>    0.034</td> <td>   21.513</td> <td> 0.000</td> <td>    0.657</td> <td>    0.788</td>
</tr>
<tr>
  <th>Departure Delay in Minutes</th> <td>    0.9802</td> <td>    0.001</td> <td> 1193.006</td> <td> 0.000</td> <td>    0.979</td> <td>    0.982</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>79305.928</td> <th>  Durbin-Watson:     </th>  <td>   1.999</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>3993601.780</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.238</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>32.720</td>   <th>  Cond. No.          </th>  <td>    43.8</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
#출발지연이 null인 데이터
df_na = df_train[df_train['Arrival Delay in Minutes'].isna()]

#출발지연이 null인 데이터의 도착지연값
X_na = df_na[['Departure Delay in Minutes']]

#상수항 추가 
X_na = sm.add_constant(X_na)

#모델 예측값
predicted_arrival_delay = model.predict(X_na)

#기존 데이터결측치를 모델 예측값으로 메꾸기
df_train.loc[df_train['Arrival Delay in Minutes'].isna(), 'Arrival Delay in Minutes'] = predicted_arrival_delay
```


```python
#결측치 확인
df_train.isnull().sum()/len(df_train)  *100
```




    Gender                               0.0
    Customer Type                        0.0
    Age                                  0.0
    Type of Travel                       0.0
    Class                                0.0
    Flight Distance                      0.0
    Inflight wifi service                0.0
    Departure/Arrival time convenient    0.0
    Ease of Online booking               0.0
    Gate location                        0.0
    Food and drink                       0.0
    Online boarding                      0.0
    Seat comfort                         0.0
    Inflight entertainment               0.0
    On-board service                     0.0
    Leg room service                     0.0
    Baggage handling                     0.0
    Checkin service                      0.0
    Inflight service                     0.0
    Cleanliness                          0.0
    Departure Delay in Minutes           0.0
    Arrival Delay in Minutes             0.0
    satisfaction                         0.0
    dtype: float64



### 시각화

성별과 충성고객에 따른 만족도는 차이가 없어보이고,


비즈니스클래스가 이코노미클래스에 비해 만족하는 경우가 많다.

비즈니스 여행이 개인여행에 비해 만족하는 경우가 많다.


```python
hue_order = ["neutral or dissatisfied", "satisfied"]

palette = {
    "neutral or dissatisfied": "#4C72B0",
    "satisfied": "#DD8452"
}

cat_cols = ["Gender","Customer Type", "Class", "Type of Travel"]
fig, axes = plt.subplots(2,2, figsize=(14, 10))
axes = axes.flatten()

for i,(ax, col) in enumerate(zip(axes,cat_cols)):
    show_legend = True if i == 1 else False
    sns.countplot(ax = ax, data = df_train, x = col, hue = "satisfaction",
                  palette = palette, hue_order = hue_order,legend=show_legend)
                

plt.tight_layout()
plt.show()
```


    
![png](airline_files/airline_19_0.png)
    


나이와 게이트위치, 지연시각은 만족도여부에 큰 상관 없어보임

지연시각의 경우 이상치 다수 존재


```python
palette = {
    "neutral or dissatisfied": "#4C72B0",
    "satisfied": "#DD8452"
}

columns = [
    "Age", "Flight Distance", "Inflight wifi service", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness", "Departure Delay in Minutes","Arrival Delay in Minutes"
]
fig, axes = plt.subplots(5, 4, figsize=(25, 16))
axes = axes.flatten()  

for i,(ax, col) in enumerate(zip(axes,columns)):
    show_legend = True if i == 3 else False
    sns.boxplot(ax = ax, data = df_train, x = "satisfaction", y = col, hue ="satisfaction",
                palette = palette,legend=show_legend)
    
for ax in axes[len(columns):]:
    ax.remove()


plt.tight_layout()
plt.show()
```


    
![png](airline_files/airline_22_0.png)
    


대부분의 변수가 4점을 기준으로 차이를 보임


```python
hue_order = ["neutral or dissatisfied", "satisfied"]


palette = {
    "neutral or dissatisfied": "#4C72B0",
    "satisfied": "#DD8452"
}

columns = [
    "Inflight wifi service", "Ease of Online booking", "Gate location", "Food and drink","Departure/Arrival time convenient",
    "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
    "Leg room service", "Baggage handling", "Checkin service", "Inflight service",
    "Cleanliness"
]

# 4x4 subplot 생성 (남는 subplot은 제거)
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()

# 각 열에 대해 countplot 그리기
for i, (ax, col) in enumerate(zip(axes, columns)):
    # 3 번째 subplot만 legend 켜기, 나머지는 숨기기
    show_legend = True if i == 3 else False
    
    sns.countplot(
        ax=ax,
        data=df_train,
        x=col,
        hue="satisfaction",
        palette=palette,
        hue_order=hue_order,
        legend=show_legend
    )

    
# 남는 subplot 제거
for ax in axes[len(columns):]:
    ax.remove()



plt.tight_layout()
plt.show()
```


    
![png](airline_files/airline_24_0.png)
    


### 이상치 처리
분산을 과도하게 증가시켜 분석이나 모델링 정확도를 감소킴 -> 상한및 하한값으로 대치


```python
# 1. 사분위수 계산
Q1 = df_train["Flight Distance"].quantile(0.25)
Q3 = df_train["Flight Distance"].quantile(0.75)
IQR = Q3 - Q1

# 2. 하한값, 상한값 계산
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 3. 이상치를 하한/상한값으로 대체
df_train["Flight Distance"] = df_train["Flight Distance"].clip(lower=lower_bound, upper=upper_bound)
```


```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 103904 entries, 0 to 103903
    Data columns (total 23 columns):
     #   Column                             Non-Null Count   Dtype  
    ---  ------                             --------------   -----  
     0   Gender                             103904 non-null  object 
     1   Customer Type                      103904 non-null  object 
     2   Age                                103904 non-null  int64  
     3   Type of Travel                     103904 non-null  object 
     4   Class                              103904 non-null  object 
     5   Flight Distance                    103904 non-null  float64
     6   Inflight wifi service              103904 non-null  int64  
     7   Departure/Arrival time convenient  103904 non-null  int64  
     8   Ease of Online booking             103904 non-null  int64  
     9   Gate location                      103904 non-null  int64  
     10  Food and drink                     103904 non-null  int64  
     11  Online boarding                    103904 non-null  int64  
     12  Seat comfort                       103904 non-null  int64  
     13  Inflight entertainment             103904 non-null  int64  
     14  On-board service                   103904 non-null  int64  
     15  Leg room service                   103904 non-null  int64  
     16  Baggage handling                   103904 non-null  int64  
     17  Checkin service                    103904 non-null  int64  
     18  Inflight service                   103904 non-null  int64  
     19  Cleanliness                        103904 non-null  int64  
     20  Departure Delay in Minutes         103904 non-null  int64  
     21  Arrival Delay in Minutes           103904 non-null  float64
     22  satisfaction                       103904 non-null  object 
    dtypes: float64(2), int64(16), object(5)
    memory usage: 19.0+ MB
    

## 모델 학습

변수의 해석력을 중요시하기에 로지스틱회귀분석 선택


```python
#로지스틱 회귀분석을 위해 범주형변수 연속형으로 변환시키기
df_train = pd.get_dummies(df_train,drop_first=True)
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
      <th>Food and drink</th>
      <th>Online boarding</th>
      <th>Seat comfort</th>
      <th>Inflight entertainment</th>
      <th>...</th>
      <th>Inflight service</th>
      <th>Cleanliness</th>
      <th>Departure Delay in Minutes</th>
      <th>Arrival Delay in Minutes</th>
      <th>Gender_Male</th>
      <th>Customer Type_disloyal Customer</th>
      <th>Type of Travel_Personal Travel</th>
      <th>Class_Eco</th>
      <th>Class_Eco Plus</th>
      <th>satisfaction_satisfied</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13</td>
      <td>460.0</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>25</td>
      <td>18.0</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>235.0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>6.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>1142.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>562.0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>9.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61</td>
      <td>214.0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
X = df_train.drop(['satisfaction_satisfied'],axis = 1)
y = df_train['satisfaction_satisfied']
```

음식에 대한 만족도가 1점 상승할 때 불만족대비 만족확률이 2.7% 감소한다는 결과가 나옴. 전체적으로 데이터의 특성을 잘 설명하지 못하고 있다.


```python
X = X.astype(float)
X = sm.add_constant(X)

model = sm.Logit(y, X).fit()
print(model.summary())
```

    Optimization terminated successfully.
             Current function value: 0.333878
             Iterations 7
                                 Logit Regression Results                             
    ==================================================================================
    Dep. Variable:     satisfaction_satisfied   No. Observations:               103904
    Model:                              Logit   Df Residuals:                   103880
    Method:                               MLE   Df Model:                           23
    Date:                    Mon, 23 Feb 2026   Pseudo R-squ.:                  0.5120
    Time:                            17:50:53   Log-Likelihood:                -34691.
    converged:                           True   LL-Null:                       -71094.
    Covariance Type:                nonrobust   LLR p-value:                     0.000
    =====================================================================================================
                                            coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------------------------
    const                                -5.8276      0.075    -77.675      0.000      -5.975      -5.681
    Age                                  -0.0083      0.001    -11.686      0.000      -0.010      -0.007
    Flight Distance                    -1.86e-05   1.14e-05     -1.630      0.103    -4.1e-05    3.77e-06
    Inflight wifi service                 0.3945      0.011     34.437      0.000       0.372       0.417
    Departure/Arrival time convenient    -0.1248      0.008    -15.203      0.000      -0.141      -0.109
    Ease of Online booking               -0.1429      0.011    -12.613      0.000      -0.165      -0.121
    Gate location                         0.0293      0.009      3.194      0.001       0.011       0.047
    Food and drink                       -0.0277      0.011     -2.592      0.010      -0.049      -0.007
    Online boarding                       0.6124      0.010     59.854      0.000       0.592       0.632
    Seat comfort                          0.0666      0.011      5.969      0.000       0.045       0.089
    Inflight entertainment                0.0641      0.014      4.502      0.000       0.036       0.092
    On-board service                      0.3015      0.010     29.628      0.000       0.282       0.321
    Leg room service                      0.2531      0.009     29.689      0.000       0.236       0.270
    Baggage handling                      0.1344      0.011     11.763      0.000       0.112       0.157
    Checkin service                       0.3239      0.009     37.861      0.000       0.307       0.341
    Inflight service                      0.1199      0.012      9.967      0.000       0.096       0.143
    Cleanliness                           0.2232      0.012     18.473      0.000       0.200       0.247
    Departure Delay in Minutes            0.0048      0.001      4.842      0.000       0.003       0.007
    Arrival Delay in Minutes             -0.0094      0.001     -9.661      0.000      -0.011      -0.008
    Gender_Male                           0.0408      0.019      2.098      0.036       0.003       0.079
    Customer Type_disloyal Customer      -2.0371      0.030    -68.107      0.000      -2.096      -1.978
    Type of Travel_Personal Travel       -2.7210      0.031    -86.620      0.000      -2.783      -2.659
    Class_Eco                            -0.7377      0.026    -28.775      0.000      -0.788      -0.687
    Class_Eco Plus                       -0.8531      0.041    -20.570      0.000      -0.934      -0.772
    =====================================================================================================
    

회귀모델은 독립변수들간 독립성을 전제로한다.

다중공선성 확인결과 출발과 도착지연시간이 공선성을지님 (상관계수 0.9)

따라서, 출발지연시간을 삭제한다.


```python
### 다중공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF 계산을 위한 함수 정의
def check_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    # 각 변수별로 VIF 계산
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data.sort_values(by="VIF", ascending=False)

# X에 상수항(const)이 포함된 상태에서 실행
print(check_vif(X))
```

                                  feature        VIF
    0                               const  55.009243
    18           Arrival Delay in Minutes  14.886209
    17         Departure Delay in Minutes  14.873682
    10             Inflight entertainment   3.887935
    16                        Cleanliness   2.872970
    5              Ease of Online booking   2.706871
    3               Inflight wifi service   2.461553
    9                        Seat comfort   2.387512
    7                      Food and drink   2.172716
    22                          Class_Eco   2.141621
    21     Type of Travel_Personal Travel   2.089672
    15                   Inflight service   2.078111
    8                     Online boarding   2.003245
    13                   Baggage handling   1.906975
    11                   On-board service   1.772354
    4   Departure/Arrival time convenient   1.668291
    20    Customer Type_disloyal Customer   1.578427
    6                       Gate location   1.501737
    2                     Flight Distance   1.368663
    12                   Leg room service   1.315016
    23                     Class_Eco Plus   1.303893
    14                    Checkin service   1.226869
    1                                 Age   1.160690
    19                        Gender_Male   1.011272
    


```python
df_train.drop(columns=["Arrival Delay in Minutes"],inplace = True)
```

### 모델 정확성 높이기
변수 구간화를 통해 데이터의 특성을 더 살려준다


```python
#0~1, 2~3, 4~5점으로 3구간 나누는경우
def categorize_score(x):
    if x <= 1:
        return "bad"
    elif x <= 3:
        return "normal"
    else:  # 4~5
        return "good"
    
#4점을 기점으로 변하는경우    
def categorize_score2(x):
    if x <= 3:
        return "bad or normal"
    else:  # 4~5
        return "good"


df_train["Food and drink"] = df_train["Food and drink"].apply(categorize_score)
df_train["Inflight wifi service"] = df_train["Inflight wifi service"].apply(categorize_score2)
df_train["Ease of Online booking"] = df_train["Ease of Online booking"].apply(categorize_score2)
df_train["Online boarding"] = df_train["Online boarding"].apply(categorize_score2)
df_train["Seat comfort"] = df_train["Seat comfort"].apply(categorize_score2)
df_train["Inflight entertainment"] = df_train["Inflight entertainment"].apply(categorize_score2)
df_train["On-board service"] = df_train["On-board service"].apply(categorize_score2)
df_train["Leg room service"] = df_train["Leg room service"].apply(categorize_score)
df_train["Baggage handling"] = df_train["Baggage handling"].apply(categorize_score2)
df_train["Checkin service"] = df_train["Checkin service"].apply(categorize_score)
df_train["Inflight service"] = df_train["Inflight service"].apply(categorize_score2)
df_train["Cleanliness"] = df_train["Cleanliness"].apply(categorize_score)
df_train["Departure/Arrival time convenient"] = df_train["Departure/Arrival time convenient"].apply(categorize_score2)



# 더미변수 생성
df_train = pd.get_dummies(df_train, drop_first=True)
```


```python
X = df_train.drop(['satisfaction_satisfied'],axis = 1)
y = df_train[['satisfaction_satisfied']]
```

변수 구간화를 통해 모델설명력 0.5120 -> 0.5690 증가


```python
X = X.astype(float)
X = sm.add_constant(X)

model2 = sm.Logit(y, X).fit()
print(model2.summary())
```

    Optimization terminated successfully.
             Current function value: 0.295507
             Iterations 7
                                 Logit Regression Results                             
    ==================================================================================
    Dep. Variable:     satisfaction_satisfied   No. Observations:               103904
    Model:                              Logit   Df Residuals:                   103877
    Method:                               MLE   Df Model:                           26
    Date:                    Mon, 23 Feb 2026   Pseudo R-squ.:                  0.5681
    Time:                            17:50:56   Log-Likelihood:                -30704.
    converged:                           True   LL-Null:                       -71094.
    Covariance Type:                nonrobust   LLR p-value:                     0.000
    ==========================================================================================================
                                                 coef    std err          z      P>|z|      [0.025      0.975]
    ----------------------------------------------------------------------------------------------------------
    const                                     -2.0943      0.072    -29.223      0.000      -2.235      -1.954
    Age                                       -0.0073      0.001     -9.669      0.000      -0.009      -0.006
    Flight Distance                         8.511e-06   1.21e-05      0.703      0.482   -1.52e-05    3.22e-05
    Gate location                             -0.0781      0.009     -8.350      0.000      -0.096      -0.060
    Departure Delay in Minutes                -0.0039      0.000    -14.053      0.000      -0.004      -0.003
    Gender_Male                                0.0224      0.021      1.072      0.284      -0.019       0.063
    Customer Type_disloyal Customer           -2.0897      0.032    -65.059      0.000      -2.153      -2.027
    Type of Travel_Personal Travel            -2.6786      0.034    -79.809      0.000      -2.744      -2.613
    Class_Eco                                 -0.8759      0.028    -31.315      0.000      -0.931      -0.821
    Class_Eco Plus                            -1.0159      0.045    -22.380      0.000      -1.105      -0.927
    Inflight wifi service_good                 1.7260      0.032     53.889      0.000       1.663       1.789
    Departure/Arrival time convenient_good    -0.4477      0.028    -16.136      0.000      -0.502      -0.393
    Ease of Online booking_good               -0.0170      0.033     -0.511      0.610      -0.082       0.048
    Food and drink_good                        0.0635      0.046      1.391      0.164      -0.026       0.153
    Food and drink_normal                      0.1972      0.043      4.585      0.000       0.113       0.281
    Online boarding_good                       1.6140      0.026     61.683      0.000       1.563       1.665
    Seat comfort_good                          0.5137      0.029     17.567      0.000       0.456       0.571
    Inflight entertainment_good               -0.0160      0.038     -0.425      0.671      -0.090       0.058
    On-board service_good                      0.5483      0.026     20.743      0.000       0.497       0.600
    Leg room service_good                      0.7782      0.038     20.631      0.000       0.704       0.852
    Leg room service_normal                    0.1043      0.037      2.832      0.005       0.032       0.176
    Baggage handling_good                      0.4737      0.029     16.611      0.000       0.418       0.530
    Checkin service_good                       0.9042      0.035     25.765      0.000       0.835       0.973
    Checkin service_normal                     0.4765      0.035     13.809      0.000       0.409       0.544
    Inflight service_good                      0.4784      0.030     15.930      0.000       0.420       0.537
    Cleanliness_good                           0.4759      0.046     10.277      0.000       0.385       0.567
    Cleanliness_normal                         0.2469      0.043      5.790      0.000       0.163       0.330
    ==========================================================================================================
    


```python
# 계수
coef = model2.params

# 오즈비
odds_ratio = np.exp(coef)

# 95% 신뢰구간
conf = model2.conf_int()
conf.columns = ["2.5%", "97.5%"]

or_table = pd.DataFrame({
    "coef": coef,
    "odds_ratio": odds_ratio,
    "OR_lower_95%": np.exp(conf["2.5%"]),
    "OR_upper_95%": np.exp(conf["97.5%"]),
    "p_value": model2.pvalues
})
```


```python
importance_table = (
    or_table
    .drop(index="const")  # 상수항 제거
    .assign(importance=lambda x: np.abs(x["coef"]))
    .sort_values("importance", ascending=False)
)

importance_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coef</th>
      <th>odds_ratio</th>
      <th>OR_lower_95%</th>
      <th>OR_upper_95%</th>
      <th>p_value</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Type of Travel_Personal Travel</th>
      <td>-2.678616</td>
      <td>0.068658</td>
      <td>0.064287</td>
      <td>0.073326</td>
      <td>0.000000e+00</td>
      <td>2.678616</td>
    </tr>
    <tr>
      <th>Customer Type_disloyal Customer</th>
      <td>-2.089681</td>
      <td>0.123727</td>
      <td>0.116178</td>
      <td>0.131766</td>
      <td>0.000000e+00</td>
      <td>2.089681</td>
    </tr>
    <tr>
      <th>Inflight wifi service_good</th>
      <td>1.725973</td>
      <td>5.617983</td>
      <td>5.276159</td>
      <td>5.981952</td>
      <td>0.000000e+00</td>
      <td>1.725973</td>
    </tr>
    <tr>
      <th>Online boarding_good</th>
      <td>1.614017</td>
      <td>5.022946</td>
      <td>4.771839</td>
      <td>5.287267</td>
      <td>0.000000e+00</td>
      <td>1.614017</td>
    </tr>
    <tr>
      <th>Class_Eco Plus</th>
      <td>-1.015864</td>
      <td>0.362089</td>
      <td>0.331267</td>
      <td>0.395780</td>
      <td>6.226325e-111</td>
      <td>1.015864</td>
    </tr>
    <tr>
      <th>Checkin service_good</th>
      <td>0.904238</td>
      <td>2.470048</td>
      <td>2.305854</td>
      <td>2.645935</td>
      <td>2.203151e-146</td>
      <td>0.904238</td>
    </tr>
    <tr>
      <th>Class_Eco</th>
      <td>-0.875876</td>
      <td>0.416497</td>
      <td>0.394279</td>
      <td>0.439967</td>
      <td>2.954020e-215</td>
      <td>0.875876</td>
    </tr>
    <tr>
      <th>Leg room service_good</th>
      <td>0.778201</td>
      <td>2.177551</td>
      <td>2.022371</td>
      <td>2.344638</td>
      <td>1.448535e-94</td>
      <td>0.778201</td>
    </tr>
    <tr>
      <th>On-board service_good</th>
      <td>0.548346</td>
      <td>1.730389</td>
      <td>1.643018</td>
      <td>1.822405</td>
      <td>1.406257e-95</td>
      <td>0.548346</td>
    </tr>
    <tr>
      <th>Seat comfort_good</th>
      <td>0.513721</td>
      <td>1.671499</td>
      <td>1.578391</td>
      <td>1.770099</td>
      <td>4.376950e-69</td>
      <td>0.513721</td>
    </tr>
    <tr>
      <th>Inflight service_good</th>
      <td>0.478374</td>
      <td>1.613448</td>
      <td>1.521228</td>
      <td>1.711259</td>
      <td>3.898091e-57</td>
      <td>0.478374</td>
    </tr>
    <tr>
      <th>Checkin service_normal</th>
      <td>0.476517</td>
      <td>1.610455</td>
      <td>1.505139</td>
      <td>1.723140</td>
      <td>2.234795e-43</td>
      <td>0.476517</td>
    </tr>
    <tr>
      <th>Cleanliness_good</th>
      <td>0.475945</td>
      <td>1.609535</td>
      <td>1.469877</td>
      <td>1.762461</td>
      <td>8.914668e-25</td>
      <td>0.475945</td>
    </tr>
    <tr>
      <th>Baggage handling_good</th>
      <td>0.473726</td>
      <td>1.605967</td>
      <td>1.518664</td>
      <td>1.698289</td>
      <td>5.781382e-62</td>
      <td>0.473726</td>
    </tr>
    <tr>
      <th>Departure/Arrival time convenient_good</th>
      <td>-0.447659</td>
      <td>0.639123</td>
      <td>0.605299</td>
      <td>0.674837</td>
      <td>1.418780e-58</td>
      <td>0.447659</td>
    </tr>
    <tr>
      <th>Cleanliness_normal</th>
      <td>0.246855</td>
      <td>1.279994</td>
      <td>1.177381</td>
      <td>1.391551</td>
      <td>7.040745e-09</td>
      <td>0.246855</td>
    </tr>
    <tr>
      <th>Food and drink_normal</th>
      <td>0.197161</td>
      <td>1.217941</td>
      <td>1.119498</td>
      <td>1.325040</td>
      <td>4.539713e-06</td>
      <td>0.197161</td>
    </tr>
    <tr>
      <th>Leg room service_normal</th>
      <td>0.104268</td>
      <td>1.109898</td>
      <td>1.032628</td>
      <td>1.192950</td>
      <td>4.625345e-03</td>
      <td>0.104268</td>
    </tr>
    <tr>
      <th>Gate location</th>
      <td>-0.078108</td>
      <td>0.924865</td>
      <td>0.908063</td>
      <td>0.941978</td>
      <td>6.827525e-17</td>
      <td>0.078108</td>
    </tr>
    <tr>
      <th>Food and drink_good</th>
      <td>0.063542</td>
      <td>1.065604</td>
      <td>0.974344</td>
      <td>1.165411</td>
      <td>1.642262e-01</td>
      <td>0.063542</td>
    </tr>
    <tr>
      <th>Gender_Male</th>
      <td>0.022408</td>
      <td>1.022661</td>
      <td>0.981605</td>
      <td>1.065434</td>
      <td>2.837772e-01</td>
      <td>0.022408</td>
    </tr>
    <tr>
      <th>Ease of Online booking_good</th>
      <td>-0.017046</td>
      <td>0.983098</td>
      <td>0.920832</td>
      <td>1.049574</td>
      <td>6.096168e-01</td>
      <td>0.017046</td>
    </tr>
    <tr>
      <th>Inflight entertainment_good</th>
      <td>-0.016020</td>
      <td>0.984107</td>
      <td>0.914050</td>
      <td>1.059535</td>
      <td>6.707120e-01</td>
      <td>0.016020</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.007342</td>
      <td>0.992685</td>
      <td>0.991209</td>
      <td>0.994164</td>
      <td>4.088621e-22</td>
      <td>0.007342</td>
    </tr>
    <tr>
      <th>Departure Delay in Minutes</th>
      <td>-0.003928</td>
      <td>0.996079</td>
      <td>0.995534</td>
      <td>0.996625</td>
      <td>7.417075e-45</td>
      <td>0.003928</td>
    </tr>
    <tr>
      <th>Flight Distance</th>
      <td>0.000009</td>
      <td>1.000009</td>
      <td>0.999985</td>
      <td>1.000032</td>
      <td>4.821593e-01</td>
      <td>0.000009</td>
    </tr>
  </tbody>
</table>
</div>



# 결과 해석

## <항공사의 개선 우선순위 >

### 효과 확실한 것 :
1. 와이파이 : 4점이상 개선시 약 5.08배 불만족대비 만족확률증가

2. 온라인 탑승확인 시스템 : 4점이상으로 개선시 약 5배 증가

3. 공항 내 체크인 시스템 : 4점이상 개선시 약 2.46배 증가 

### 그 후 개선하면 긍정적인 것 : 
1. 확실한 다리공간 넓이 확보, : 약 2.23배 증가

2. 좌석 편암함 : 약 1.65배 증가

3. 수하물 처리 : 약 1.61배 증가


### 못바꾸지만 큰 영향 미치는 것: 
1. 여행 목적 : 비즈니스가 14.6배 증가
2. 고객 충성도 : 충성고객이 8.46배 증가
3. 좌석 등급 : 비즈니스가 이코노비에 비해 2.41배 증가


# <세부사항>
## 1. 여행목적 

비즈니스 여행객은 개인 여행객보다 만족할 확률 14.6배 높다.
    
## 2. 고객 충성도

 충성 고객은 비충성 고객보다 약 8.46배 만족확률 높다.

## 3. 기내 와이파이 서비스

기내 와이파이가 4점이상인 경우가 그렇지 않은 경우에 비해 약 5.08배 증가한다.

-> 와이파이 개선하려면 확실히 좋게 해야 한다.

## 4. 온라인 탑승확인

온라인 탑승확인이 4점이상인 경우가 그렇지 않은 경우에 비해 약 5배 증가한다.

--> 어플로 온라인 탑승확인을 확실히 좋게 해야 한다.


## 5. 비즈니스와 이코노미플러스, 이코노미 티켓

비즈니스 클래스 승객은 이코노미 plus 승객보다  2.77배 만족확률 높다.

비즈니스 클래스 승객은 이코노미 승객보다 약 2.41배 만족확률 높다.

## 6. 공항 체크인 서비스

bad → good : 2.46배 증가

bad → normal : 1.61배 증가

->체크인서비스는 확실히 좋아질 수록 만족도에 영향을 미친다.

## 7. 래그 룸 서비스 (leg room service)

레그 룸 서비스 점수가 4\~5점인 집단은 0\~1점인 집단 대비
만족할 오즈가 약 2.23배 높다.

레그 룸 서비스 를 2\~3점으로 평가한 집단은  0\~1점인 집단 대비
만족할 오즈가 약 1.13배 높다.

--> 애매하게 다리공간을 늘리는 것보다 확실히 늘리는 것이 중요하다.

## 8. On-board service

on-board 서비스가 4\~5점인 경우
0\~3점인 경우보다 만족 가능성이 약 1.7배 높다.

## 9. 좌석편안함(seat comfort)

좌석 편안함을 4\~5점으로 평가한 집단은 0\~3점으로 평가한 집단 대비
만족할 오즈가 약 1.65배 높다.

-> 애매하게 늘리는것 보다 확실히 늘려야 1.66배정도 더 만족한다.


## 10 기내 서비스(inflight service)

기내 서비스를  4\~5점으로 평가한 집단은 0\~3점으로 평가한 집단 대비
만족할 오즈가 약 1.61배 높다.

## 11. 수하물 처리 (baggage handling)

수하물 처리를  4\~5점으로 평가한 집단은 0\~3점으로 평가한 집단 대비
만족할 오즈가 약 1.61배 높다.

## 12. 기내 청결도 (Cleanliness)

bad → good : 1.60배 

bad → normal : 1.28배 
-> 기본적으로 깨끗하면 크게 늘어나진 않음

## 13. 기내 식음료 (Food and drink)

bad → good : 1.07배  그러나 pvalue > 0.05이므로 통계적으로 유의하지않음

bad → normal : 1.22배 

--> 식음료는 기내 만족도에 큰 영향을 주지 못함

## 14. 출발 및 도착시간의 편리성(Departure/Arrival time convenient)
이전 대비 약 0.85배 수준으로 감소한다.

## 15. 온라인 예약 (Ease of Online booking)

온라인 예약을 4~5점으로 평가한 경우
만족 오즈는 0.93배가 된다.

## 16. Gate location
큰 영향을 주지 못한다.

## 17. 성별
남성은 여성 대비
만족 오즈가 1.02배이다.

통계적으로 유의하지 않다.

## 18. 나이
나이 1세 증가할 때마다
만족 확률이 만족하지 않을 확률에 비해 0.993배가 된다.
사실상 1배에 매우 가까워 영향은 매우 작다.

## 19. Inflight entertainment

OR = 0.9948

4~5점 평가 시
만족 오즈는 0.995배이다.

1배와 거의 동일하며 유의하지 않다.

## 20. 도착지연시간 (Departure Delay in Minutes)

지연 1분 증가할 때마다
만족 오즈는 0.996배가 된다.

1배에 매우 가까워 영향은 작다.

## 21. 비행거리 (Flight Distance)

OR = 1.000013

비행거리 1단위 증가 시
만족 오즈는 1.000013배이다.

사실상 1배로, 영향이 없다고 볼 수 있다.


