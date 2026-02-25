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
  <th>Date:</th>                 <td>Wed, 25 Feb 2026</td>     <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                     <td>14:04:32</td>         <th>  Log-Likelihood:    </th> <td>-3.8635e+05</td>
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


비즈니스 클래스가 이코노미 클래스에 비해 만족하는 경우가 많다.

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


    
![png](output_19_0.png)
    


나이와 입구 위치, 지연시각은 만족도 여부에 큰 상관 없어보임

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


    
![png](output_22_0.png)
    


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


    
![png](output_24_0.png)
    


### 이상치 처리
분산을 과도하게 증가시켜 분석 및 모델링 정확도를 감소킴 -> 상한 및 하한 값으로 대치


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

## 모델 학습

변수의 해석력을 중요시 하기에 로지스틱 회귀분석 선택


```python
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
      <th>Gender</th>
      <th>Customer Type</th>
      <th>Age</th>
      <th>Type of Travel</th>
      <th>Class</th>
      <th>Flight Distance</th>
      <th>Inflight wifi service</th>
      <th>Departure/Arrival time convenient</th>
      <th>Ease of Online booking</th>
      <th>Gate location</th>
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
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>13</td>
      <td>Personal Travel</td>
      <td>Eco Plus</td>
      <td>460.0</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
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
      <td>Male</td>
      <td>disloyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>235.0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
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
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>26</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>1142.0</td>
      <td>2</td>
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
      <td>Female</td>
      <td>Loyal Customer</td>
      <td>25</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>562.0</td>
      <td>2</td>
      <td>5</td>
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
      <td>Male</td>
      <td>Loyal Customer</td>
      <td>61</td>
      <td>Business travel</td>
      <td>Business</td>
      <td>214.0</td>
      <td>3</td>
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
<p>5 rows × 23 columns</p>
</div>




```python
X_raw = df_train.copy()

from sklearn.model_selection import train_test_split

# 타겟 분리
X = df_train.drop('satisfaction', axis=1)
y = df_train['satisfaction']
y = pd.get_dummies(y, drop_first=True)

# 먼저 나누기
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

<베이스라인 모델>

정확도 87%

 AUC: 0.9281358163791855


```python
from sklearn.metrics import accuracy_score, roc_auc_score

# 더미화
X_train_base = pd.get_dummies(X_train_raw, drop_first=True)
X_val_base = pd.get_dummies(X_val_raw, drop_first=True)

# 컬럼 맞추기 (필수)
X_train_base, X_val_base = X_train_base.align(X_val_base, 
                                              join='left', axis=1, fill_value=0)

# 상수항
X_train_base = sm.add_constant(X_train_base)
X_val_base = sm.add_constant(X_val_base)
X_train_base = X_train_base.astype(float)
X_val_base = X_val_base.astype(float)


model_base = sm.Logit(y_train, X_train_base).fit()
pred_base = model_base.predict(X_val_base)
# 0.5 기준으로 클래스 예측
pred_class_base = (pred_base >= 0.5).astype(int)

print("=== Baseline ===")
print("Accuracy:", accuracy_score(y_val, pred_class_base))
print("Baseline AUC:", roc_auc_score(y_val, pred_base))

```

    Optimization terminated successfully.
             Current function value: 0.334595
             Iterations 7
    === Baseline ===
    Accuracy: 0.8765218228189211
    Baseline AUC: 0.9281358163791855
    

## <영향력 측면>

wifi service, online boarding, checkin service, class가 만족도에 영향을 크게 주는 것으로 나옴

반면 age, food and drink, gate location은  만족도에 영향이 없어보임

## <해석적 측면의 문제점> 

음식에 대한 만족도가 1점 상승할 때 불만족 대비 만족확률이 2.3% 감소한다는 결과가 나옴

영향력은 어느정도 설명되지만, 그 크기에 대해선 데이터의 특성을 잘 설명하지 못하고 있음


```python
model_base.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>satisfied</td>    <th>  No. Observations:  </th>  <td> 83123</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 83099</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    23</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 25 Feb 2026</td> <th>  Pseudo R-squ.:     </th>  <td>0.5110</td> 
</tr>
<tr>
  <th>Time:</th>                <td>14:05:18</td>     <th>  Log-Likelihood:    </th> <td> -27813.</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -56875.</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
                  <td></td>                     <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                             <td>   -5.8419</td> <td>    0.084</td> <td>  -69.719</td> <td> 0.000</td> <td>   -6.006</td> <td>   -5.678</td>
</tr>
<tr>
  <th>Age</th>                               <td>   -0.0082</td> <td>    0.001</td> <td>  -10.425</td> <td> 0.000</td> <td>   -0.010</td> <td>   -0.007</td>
</tr>
<tr>
  <th>Flight Distance</th>                   <td>-2.234e-05</td> <td> 1.28e-05</td> <td>   -1.749</td> <td> 0.080</td> <td>-4.74e-05</td> <td>  2.7e-06</td>
</tr>
<tr>
  <th>Inflight wifi service</th>             <td>    0.3988</td> <td>    0.013</td> <td>   31.064</td> <td> 0.000</td> <td>    0.374</td> <td>    0.424</td>
</tr>
<tr>
  <th>Departure/Arrival time convenient</th> <td>   -0.1195</td> <td>    0.009</td> <td>  -13.022</td> <td> 0.000</td> <td>   -0.137</td> <td>   -0.102</td>
</tr>
<tr>
  <th>Ease of Online booking</th>            <td>   -0.1347</td> <td>    0.013</td> <td>  -10.590</td> <td> 0.000</td> <td>   -0.160</td> <td>   -0.110</td>
</tr>
<tr>
  <th>Gate location</th>                     <td>    0.0172</td> <td>    0.010</td> <td>    1.680</td> <td> 0.093</td> <td>   -0.003</td> <td>    0.037</td>
</tr>
<tr>
  <th>Food and drink</th>                    <td>   -0.0237</td> <td>    0.012</td> <td>   -1.981</td> <td> 0.048</td> <td>   -0.047</td> <td>   -0.000</td>
</tr>
<tr>
  <th>Online boarding</th>                   <td>    0.6074</td> <td>    0.011</td> <td>   53.007</td> <td> 0.000</td> <td>    0.585</td> <td>    0.630</td>
</tr>
<tr>
  <th>Seat comfort</th>                      <td>    0.0709</td> <td>    0.012</td> <td>    5.682</td> <td> 0.000</td> <td>    0.046</td> <td>    0.095</td>
</tr>
<tr>
  <th>Inflight entertainment</th>            <td>    0.0571</td> <td>    0.016</td> <td>    3.590</td> <td> 0.000</td> <td>    0.026</td> <td>    0.088</td>
</tr>
<tr>
  <th>On-board service</th>                  <td>    0.2946</td> <td>    0.011</td> <td>   25.942</td> <td> 0.000</td> <td>    0.272</td> <td>    0.317</td>
</tr>
<tr>
  <th>Leg room service</th>                  <td>    0.2555</td> <td>    0.009</td> <td>   26.922</td> <td> 0.000</td> <td>    0.237</td> <td>    0.274</td>
</tr>
<tr>
  <th>Baggage handling</th>                  <td>    0.1429</td> <td>    0.013</td> <td>   11.208</td> <td> 0.000</td> <td>    0.118</td> <td>    0.168</td>
</tr>
<tr>
  <th>Checkin service</th>                   <td>    0.3258</td> <td>    0.010</td> <td>   34.063</td> <td> 0.000</td> <td>    0.307</td> <td>    0.345</td>
</tr>
<tr>
  <th>Inflight service</th>                  <td>    0.1200</td> <td>    0.013</td> <td>    8.965</td> <td> 0.000</td> <td>    0.094</td> <td>    0.146</td>
</tr>
<tr>
  <th>Cleanliness</th>                       <td>    0.2227</td> <td>    0.014</td> <td>   16.466</td> <td> 0.000</td> <td>    0.196</td> <td>    0.249</td>
</tr>
<tr>
  <th>Departure Delay in Minutes</th>        <td>    0.0052</td> <td>    0.001</td> <td>    4.691</td> <td> 0.000</td> <td>    0.003</td> <td>    0.007</td>
</tr>
<tr>
  <th>Arrival Delay in Minutes</th>          <td>   -0.0096</td> <td>    0.001</td> <td>   -8.804</td> <td> 0.000</td> <td>   -0.012</td> <td>   -0.007</td>
</tr>
<tr>
  <th>Gender_Male</th>                       <td>    0.0325</td> <td>    0.022</td> <td>    1.495</td> <td> 0.135</td> <td>   -0.010</td> <td>    0.075</td>
</tr>
<tr>
  <th>Customer Type_disloyal Customer</th>   <td>   -2.0494</td> <td>    0.033</td> <td>  -61.313</td> <td> 0.000</td> <td>   -2.115</td> <td>   -1.984</td>
</tr>
<tr>
  <th>Type of Travel_Personal Travel</th>    <td>   -2.7072</td> <td>    0.035</td> <td>  -77.254</td> <td> 0.000</td> <td>   -2.776</td> <td>   -2.638</td>
</tr>
<tr>
  <th>Class_Eco</th>                         <td>   -0.7452</td> <td>    0.029</td> <td>  -26.044</td> <td> 0.000</td> <td>   -0.801</td> <td>   -0.689</td>
</tr>
<tr>
  <th>Class_Eco Plus</th>                    <td>   -0.8747</td> <td>    0.046</td> <td>  -18.858</td> <td> 0.000</td> <td>   -0.966</td> <td>   -0.784</td>
</tr>
</table>



## 모델 성능 향상

## 다중공선성 확인 
다중공선성 확인결과 출발,도착 지연시간이 공선성을 지님 (VIF>10) -> 변수 해석 어려움을 미칠 수 있음
> 따라서, 출발지연시간을 삭제


```python
# 다중공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF 계산을 위한 함수 정의
def check_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_train_base.columns
    # 각 변수별로 VIF 계산
    vif_data["VIF"] = [variance_inflation_factor(X_train_base.values, i) for i in range(len(X_train_base.columns))]
    return vif_data.sort_values(by="VIF", ascending=False)

# X에 상수항(const)이 포함된 상태에서 실행
print(check_vif(X_train_base))

X_train_raw.drop(columns=["Arrival Delay in Minutes"],inplace = True)
```

                                  feature        VIF
    0                               const  54.871795
    18           Arrival Delay in Minutes  14.943120
    17         Departure Delay in Minutes  14.930710
    10             Inflight entertainment   3.894109
    16                        Cleanliness   2.885057
    5              Ease of Online booking   2.728364
    3               Inflight wifi service   2.467927
    9                        Seat comfort   2.387650
    7                      Food and drink   2.181222
    22                          Class_Eco   2.134596
    21     Type of Travel_Personal Travel   2.088841
    15                   Inflight service   2.070383
    8                     Online boarding   2.012522
    13                   Baggage handling   1.897093
    11                   On-board service   1.773759
    4   Departure/Arrival time convenient   1.673494
    20    Customer Type_disloyal Customer   1.578276
    6                       Gate location   1.504651
    2                     Flight Distance   1.367377
    12                   Leg room service   1.314694
    23                     Class_Eco Plus   1.302284
    14                    Checkin service   1.225002
    1                                 Age   1.162346
    19                        Gender_Male   1.011337
    

## 변수 구간화
- > 특정 구간 전 후로 만족도 차이가 많이 난다. 이를 모델에 반영하기 위해 특정 구간을 중심으로 파생변수 생성


```python
# 복사 (raw 데이터에서)
X_train_fe = X_train_raw.copy()
X_val_fe = X_val_raw.copy()


# 함수 정의
def categorize_score(x):
    if x <= 1:
        return "bad"
    elif x <= 3:
        return "normal"
    else:
        return "good"

def categorize_score2(x):
    if x <= 3:
        return "bad_or_normal"
    else:
        return "good"


# 적용할 컬럼 목록
cols_score = [
    "Food and drink",
    "Leg room service",
    "Checkin service",
    "Cleanliness"
]

cols_score2 = [
    "Inflight wifi service",
    "Ease of Online booking",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Baggage handling",
    "Inflight service",
    "Departure/Arrival time convenient"
]


# train / val 모두 동일하게 적용
for col in cols_score:
    X_train_fe[col] = X_train_fe[col].apply(categorize_score)
    X_val_fe[col] = X_val_fe[col].apply(categorize_score)

for col in cols_score2:
    X_train_fe[col] = X_train_fe[col].apply(categorize_score2)
    X_val_fe[col] = X_val_fe[col].apply(categorize_score2)
```


```python
# 더미화
X_train_fe = pd.get_dummies(X_train_fe, drop_first=True)
X_val_fe = pd.get_dummies(X_val_fe, drop_first=True)

# 컬럼 맞추기
X_train_fe, X_val_fe = X_train_fe.align(
    X_val_fe, join='left', axis=1, fill_value=0
)

# 상수항
X_train_fe = sm.add_constant(X_train_fe)
X_val_fe = sm.add_constant(X_val_fe)

# float 변환
X_train_fe = X_train_fe.astype(float)
X_val_fe = X_val_fe.astype(float)
```

## 모델 학습

## 개선된 점
정확도 87 % -> 89 % 증가

AUC: 0.92 -> 0.94 증가



```python
#  모델 학습
model_fe = sm.Logit(y_train, X_train_fe).fit(disp=0)

#  확률 예측
pred_prob_fe = model_fe.predict(X_val_fe)

#  0.5 기준 분류
pred_class_fe = (pred_prob_fe >= 0.5).astype(int)

#  성능 출력
print("=== Feature Engineering Model ===")
print("Accuracy:", accuracy_score(y_val, pred_class_fe))
print("AUC:", roc_auc_score(y_val, pred_prob_fe))
```

    === Feature Engineering Model ===
    Accuracy: 0.892786680140513
    AUC: 0.9432929018902542
    


```python
print("\n=== Improvement ===")
print("Accuracy 차이:",
      accuracy_score(y_val, pred_class_fe)
      - accuracy_score(y_val, pred_class_base)) 

print("AUC 차이:",
      roc_auc_score(y_val, pred_prob_fe)
      - roc_auc_score(y_val, pred_base))
```

    
    === Improvement ===
    Accuracy 차이: 0.016264857321591886
    AUC 차이: 0.015157085511068646
    

## 모델 해석


```python
# 계수
coef = model_fe.params

# 오즈비
odds_ratio = np.exp(coef)

# 95% 신뢰구간 (계수 기준)
conf = model_fe.conf_int()
conf.columns = ["2.5%", "97.5%"]

# OR 테이블 만들기
or_table = pd.DataFrame({
    "coef": coef,
    "odds_ratio": odds_ratio,
    "OR_lower_95%": np.exp(conf["2.5%"]),
    "OR_upper_95%": np.exp(conf["97.5%"]),
    "p_value": model_fe.pvalues
})

# 중요도 테이블 (절대계수 기준)
importance_table = (
    or_table
    .drop(index="const")   # 상수항 제거
    .assign(importance=lambda x: np.abs(x["coef"]))
    .sort_values("importance", ascending=False)
)

importance_table.head(10)
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
      <td>-2.670391</td>
      <td>0.069225</td>
      <td>0.064324</td>
      <td>0.074499</td>
      <td>0.000000e+00</td>
      <td>2.670391</td>
    </tr>
    <tr>
      <th>Customer Type_disloyal Customer</th>
      <td>-2.116258</td>
      <td>0.120482</td>
      <td>0.112287</td>
      <td>0.129274</td>
      <td>0.000000e+00</td>
      <td>2.116258</td>
    </tr>
    <tr>
      <th>Inflight wifi service_good</th>
      <td>1.744948</td>
      <td>5.725607</td>
      <td>5.335843</td>
      <td>6.143841</td>
      <td>0.000000e+00</td>
      <td>1.744948</td>
    </tr>
    <tr>
      <th>Online boarding_good</th>
      <td>1.584200</td>
      <td>4.875389</td>
      <td>4.603579</td>
      <td>5.163247</td>
      <td>0.000000e+00</td>
      <td>1.584200</td>
    </tr>
    <tr>
      <th>Class_Eco Plus</th>
      <td>-1.030069</td>
      <td>0.356982</td>
      <td>0.323167</td>
      <td>0.394335</td>
      <td>1.666624e-91</td>
      <td>1.030069</td>
    </tr>
    <tr>
      <th>Checkin service_good</th>
      <td>0.935191</td>
      <td>2.547700</td>
      <td>2.358223</td>
      <td>2.752400</td>
      <td>2.383980e-124</td>
      <td>0.935191</td>
    </tr>
    <tr>
      <th>Class_Eco</th>
      <td>-0.875801</td>
      <td>0.416528</td>
      <td>0.391792</td>
      <td>0.442826</td>
      <td>5.648960e-173</td>
      <td>0.875801</td>
    </tr>
    <tr>
      <th>Leg room service_good</th>
      <td>0.797434</td>
      <td>2.219838</td>
      <td>2.044262</td>
      <td>2.410494</td>
      <td>3.115998e-80</td>
      <td>0.797434</td>
    </tr>
    <tr>
      <th>On-board service_good</th>
      <td>0.541065</td>
      <td>1.717835</td>
      <td>1.621121</td>
      <td>1.820319</td>
      <td>8.176641e-75</td>
      <td>0.541065</td>
    </tr>
    <tr>
      <th>Seat comfort_good</th>
      <td>0.530965</td>
      <td>1.700573</td>
      <td>1.594951</td>
      <td>1.813190</td>
      <td>3.122664e-59</td>
      <td>0.530965</td>
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
4. 확실한 다리공간 넓이 확보, : 약 2.23배 증가

5. 좌석 편암함 : 약 1.65배 증가

6. 수하물 처리 : 약 1.61배 증가


### 못바꾸지만 큰 영향 미치는 것: 
1. 여행 목적 : 비즈니스가 14.6배 증가
2. 고객 충성도 : 충성고객이 8.46배 증가
3. 좌석 등급 : 비즈니스가 이코노비에 비해 2.41배 증가


# <세부사항>
## 1. 여행목적 

비즈니스 여행객은 개인 여행객보다 만족할 확률 14.5배 높다.
    
## 2. 고객 충성도

 충성 고객은 비충성 고객보다 약 8.34배 만족확률 높다.

## 3. 기내 와이파이 서비스

기내 와이파이가 4점이상인 경우가 그렇지 않은 경우에 비해 약 5.72배 증가한다.

→ 와이파이는 확실한 개선이 필요

## 4. 온라인 탑승확인

온라인 탑승확인이 4점이상인 경우가 그렇지 않은 경우에 비해 약 4.87배 증가한다.

--> 어플로 온라인 탑승확인을 확실히 좋게 해야 한다.


## 5. 비즈니스와 이코노미플러스, 이코노미 티켓

비즈니스 클래스 승객은 이코노미 plus 승객보다  2.79배 만족확률 높다.

비즈니스 클래스 승객은 이코노미 승객보다 약 2.4배 만족확률 높다.

## 6. 공항 체크인 서비스

bad → good : 2.56배 증가

bad → normal : 1.68배 증가

->체크인서비스는 확실히 좋아질 수록 만족도에 영향을 미친다.

## 7. 래그 룸 서비스 (leg room service)

레그 룸 서비스 점수가 4\~5점인 집단은 0\~1점인 집단 대비
만족할 오즈가 약 2.22배 높다.

레그 룸 서비스 를 2\~3점으로 평가한 집단은  0\~1점인 집단 대비
만족할 오즈가 약 1.13배 높다.

--> 애매하게 다리공간을 늘리는 것보다 확실히 늘리는 것이 중요하다.

## 8. On-board service

on-board 서비스가 4\~5점인 경우
0\~3점인 경우보다 만족 가능성이 약 1.72배 높다.

## 9. 좌석편안함(seat comfort)

좌석 편안함을 4\~5점으로 평가한 집단은 0\~3점으로 평가한 집단 대비
만족할 오즈가 약 1.70배 높다.

-> 애매하게 늘리는것 보다 확실히 늘려야 1.70배정도 더 만족한다.

## 10. 수하물 처리 (baggage handling)

수하물 처리를  4\~5점으로 평가한 집단은 0\~3점으로 평가한 집단 대비
만족할 오즈가 약 1.63배 높다.


## 11. 기내 서비스(inflight service)

기내 서비스를  4\~5점으로 평가한 집단은 0\~3점으로 평가한 집단 대비
만족할 오즈가 약 1.62배 높다.


## 12. 기내 청결도 (Cleanliness)

bad → good : 1.65배 

bad → normal : 1.30배 
→ 기본적 청결 확보는 중요하지만 폭발적 영향은 아님

## 13. 기내 식음료 (Food and drink)

bad → good : 1.05배 (pvalue > 0.05이므로 통계적으로 유의하지않음)

bad → normal : 1.19배 

--> 식음료는 기내 만족도에 큰 영향을 주지 못함

## 14. 출발 및 도착시간의 편리성(Departure/Arrival time convenient)
이전 대비 약 0.64배 수준으로 감소한다.

->여행 목적, 좌석 등급, 서비스 품질 등 강한 변수들을 통제한 이후에는 시간 편리성의 순수 효과가 상대적으로 약해지거나 방향이 조정된 결과로 해석할 수 있다.
## 15. 온라인 예약 (Ease of Online booking)

온라인 예약을 4~5점으로 평가한 경우
만족 오즈는 1.01배가 된다.
통계적으로 유의하지 않으며 영향은 거의 없다.

## 16. Gate location
만족 오즈가 약 0.92배로 소폭 감소한다.
통계적으로는 유의하지만, 영향력은 크지 않다.

## 17. 성별
남성은 여성 대비
만족 오즈가 1.02배이다.

통계적으로 유의하지 않다.

## 18. 나이
나이 1세 증가할 때마다 만족 오즈는 0.993배가 된다.
약 0.7% 감소 효과로, 실질적인 영향은 매우 작다.

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


