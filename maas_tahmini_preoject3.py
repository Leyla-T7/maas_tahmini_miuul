
# ************** İŞ PROBLEMİ ***********

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
# oyuncularının maaş tahminleri için bir makine öğrenmesi modeli geliştiriniz.

# ---- VERİ SETİ HİKAYESİ ------

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan
# StatLib kütüphanesinden alınmıştır. Veri seti 1988 ASA Grafik Bölümü
# Poster Oturumu'nda kullanılan verilerin bir parçasıdır. Maaş verileri
# orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır. 1986 ve
# kariyer istatistikleri, Collier Books, Macmillan Publishing Company,
# New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi
# Güncellemesinden elde edilmiştir.


# GÖREV
#Veri ön işleme,
# Özellik mühendisliği
# işlemleri gerçekleştirerek maaş tahmin modeli geliştiriniz.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import warnings

warnings.simplefilter(action="ignore")
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv('datasets/hitters.csv')
df.head()

### CHECK DATA GENERAL ####
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df.columns = [col.upper()for col in df.columns]
df.columns


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "object"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "object"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "object"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"] and "Salary" not in col]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols


#### NUMERİK VE KATEGORİK DEĞİŞKENLER ANALİZİ

def cat_summary(dataframe, col_name, plot= False):
    print({col_name: dataframe[col_name].value_counts(),
           "Ratio": 100*dataframe[col_name].value_counts() / len(dataframe)})
    print("***********************************")
    if plot:
        sns.countplot(x=dataframe[col_name], data= dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot= False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("***********************************")

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
    num_summary(df, col, True)


df_low_salary = df[df["SALARY"]<200]
df_low_salary["ERRORS"]

sns.scatterplot(x="SALARY", y="ERRORS", data=df_low_salary)


# TARGET ANALİZİ
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "SALARY", col)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "SALARY", col)


## KOLERASYON ANALİZİ
df.corr()
cor_matrix=df.corr().abs() # hepsını mutlsk degere cevısrık
upper_triangle_matrix= cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

def high_correlated_cols(dataframe,plot=False,corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 90) ]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize":(15,15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

## OUTLİERS ANALİZİ##
df.isnull().sum()

def outlier_tresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)



### EKSIK GOZLEM ANALIZI
df.isnull().any()

def missing_values(dataframe, nan_name=False):
    N_columns= [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    missing = dataframe[N_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[N_columns].isnull().sum() / dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df = pd.concat([missing, np.round(ratio , 2)], axis=1, keys=["missing", "ratio"])
    print(missing_df, end="\n")
    if  nan_name:
        return N_columns
N_columns = missing_values(df, True)

imputer = KNNImputer(n_neighbors= 5)
df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols)
df.head()
### BASE MODEL

dff = df.copy()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["SALARY"]
X = dff.drop(["SALARY"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

lin_model = LinearRegression().fit(X_train, y_train)
y_pred = lin_model.predict(X_test)

lin_model.score(X_test, y_test)


## Train rmse
y_pred = lin_model.predict(X_train)
np.sqrt((mean_squared_error(y_train, y_pred)))

## test rmse
y_pred = lin_model.predict(X_test)
np.sqrt((mean_squared_error(y_test, y_pred)))

lin_model.score(X_train, y_train)
lin_model.score(X_test, y_test)


### ENCODİNG
def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col]= label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes=="object" and df[col].nunique()==2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)
cat_cols


scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()


### MODELLEME
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

reg_model.score(X_test, y_test)

reg_model.intercept_ ## (b sabiti)
reg_model.coef_ # (w katsayi)


## train rmse
y_pred = reg_model.predict(X_train)
np.sqrt((mean_squared_error(y_train, y_pred)))


# test rmse
y_pred = reg_model.predict(X_test)
np.sqrt((mean_squared_error(y_test, y_pred)))

## train rkare
reg_model.score(X_train,y_train)
## test rkare
reg_model.score(X_test,y_test)

### CROSS VALİDATİON
np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv=10,scoring="neg_mean_absolute_error")))

