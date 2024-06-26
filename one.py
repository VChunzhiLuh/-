import pandas as pd
import numpy as np

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

# 合并 train 和 test 数据集
df = pd.concat([train, test], ignore_index=True)

print(df.describe().T)

# 1 查看统计量
print(df.describe().T)
# 2 duration分箱展示
import matplotlib.pyplot as plt
import seaborn as sns

# 3.查看数据分布
# 分离数值变量与分类变量
Nu_feature = list(df.select_dtypes(exclude=['object']).columns)
Ca_feature = list(df.select_dtypes(include=['object']).columns)
Ca_feature.remove('subscribe')
col1 = Ca_feature
plt.figure(figsize=(20, 10))
j = 1
for col in col1:
    ax = plt.subplot(4, 5, j)
    ax = plt.scatter(x=range(len(df)), y=df[col], color='red')
    plt.title(col)
    j += 1
k = 11
for col in col1:
    ax = plt.subplot(4, 5, k)
    ax = plt.scatter(x=range(len(test)), y=test[col], color='cyan')
    plt.title(col)
    k += 1
plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.show()

# # 4.数据相关图
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
cols = Ca_feature
for m in cols:
    df[m] = lb.fit_transform(df[m])
    test[m] = lb.fit_transform(test[m])
#
df['subscribe'] = df['subscribe'].replace(['no', 'yes'], [0, 1])

correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, vmax=0.9, linewidths=0.05, cmap="RdGy")
plt.show()

# 检查 'unknown' 值
train_unknown_mean = train.isin(['unknown']).mean() * 100
test_unknown_mean = test.isin(['unknown']).mean() * 100

# 打印 'unknown' 值的百分比
print("Train set 'unknown' values percentage: ", train_unknown_mean)
print("Test set 'unknown' values percentage: ", test_unknown_mean)

'''
#数据没有NA值但是有unknow值
train_set.isin(['unknown']).mean()*100
test_set.isin(['unknown']).mean()*100
# 工作，教育和沟通方式用众数填充
'''
train['default'].replace(['unknown'], test['default'].mode(), inplace=True)
train['job'].replace(['unknown'], train['job'].mode(), inplace=True)
train['education'].replace(['unknown'], train['education'].mode(), inplace=True)
train['marital'].replace(['unknown'], train['marital'].mode(), inplace=True)
train['housing'].replace(['unknown'], train['housing'].mode(), inplace=True)
train['loan'].replace(['unknown'], train['loan'].mode(), inplace=True)

# test.drop(['default'], inplace=True, axis=1)
test['default'].replace(['unknown'], test['default'].mode(), inplace=True)
test['job'].replace(['unknown'], test['job'].mode(), inplace=True)
test['education'].replace(['unknown'], test['education'].mode(), inplace=True)
test['marital'].replace(['unknown'], test['marital'].mode(), inplace=True)
test['housing'].replace(['unknown'], test['housing'].mode(), inplace=True)
test['loan'].replace(['unknown'], test['loan'].mode(), inplace=True)
print(train["job"].value_counts())

# #统计图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
train['subscribe'] = train['subscribe'].replace(['no', 'yes'], [0,1])
plt.figure(figsize = [15,10])#画板大小
sns.barplot(x = "job", y ="subscribe" , data = train)
x_1=["管理者","蓝领","技术员","服务员","经营者","退役人员","企业家","个体经营者","女佣","失业人员","学生"]
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname='C:\Windows\Fonts\STHUPO.TTF',size=20)
# plt.xticks(range(len(x_1)),x_1,fontproperties = my_font)
plt.xticks(range(len(x_1)),x_1,fontsize=20,rotation=45)
plt.yticks(fontsize=15)
my= font_manager.FontProperties(size=20)
plt.xlabel("“客户身份”",fontproperties = my)
plt.ylabel("产品购买数量指数",fontproperties = my)
plt.title("客户购买银行产品意向图",fontdict={"size": 25})
plt.tight_layout()
plt.show()

import seaborn as sns

object_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                  'poutcome']
# 连续变量列名
num_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', "cons_conf_index", 'emp_var_rate',
               "cons_price_index", "lending_rate3m", "nr_employed"]
# # 统计图

# #统计图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.figure(figsize=[10, 10])  # 画板大小
sns.barplot(x="marital", y="subscribe", data=train)
x_1 = ["结婚", " 已婚", "单身"]
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname='C:\Windows\Fonts\STHUPO.TTF', size=20)
# plt.xticks(range(len(x_1)),x_1,fontproperties = my_font)
plt.xticks(range(len(x_1)), x_1, fontsize=20)
plt.yticks(fontsize=15)
my = font_manager.FontProperties(size=20)
plt.xlabel("客户婚姻状态", fontproperties=my)
plt.ylabel("产品购买数量指数", fontproperties=my)
plt.title("不同婚姻状态的客户购买银行产品意向图", fontdict={"size": 25})
plt.tight_layout()
plt.show()

# #统计图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.figure(figsize = [10,8])#画板大小
sns.barplot(x = "education", y ="subscribe" , data = train)
x_1=["大学学历"," 高中","基本9年","教授","基本4年","基本6年","文盲"]
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname='C:\Windows\Fonts\STHUPO.TTF',size=20)
# plt.xticks(range(len(x_1)),x_1,fontproperties = my_font)
plt.xticks(range(len(x_1)),x_1,fontsize=25)
plt.yticks(fontsize=15)
my= font_manager.FontProperties(size=20)
plt.xlabel("客户教育程度",fontproperties = my)
plt.ylabel("产品购买数量指数",fontproperties = my)
plt.title("不同教育程度的客户购买银行产品意向图",fontdict={"size": 25})
plt.tight_layout()
plt.show()
print(train["education"].value_counts())

object_columns = ['job', 'marital', 'education', 'default', 'housing','loan', 'contact','month','day_of_week','poutcome']
# #统计图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.figure(figsize = [8,8])#画板大小
sns.barplot(x = "month", y ="subscribe" , data = train)
# x_1=["大学学历"," 高中","基本9年","教授","基本4年","基本6年","文盲"]
from matplotlib import font_manager
# my_font = font_manager.FontProperties(fname='C:\Windows\Fonts\STHUPO.TTF',size=20)
# plt.xticks(range(len(x_1)),x_1,fontproperties = my_font)
plt.xticks(fontsize=25)
plt.yticks(fontsize=15)
my= font_manager.FontProperties(size=20)
plt.xlabel("月份",fontproperties = my)
plt.ylabel("产品购买数量指数",fontproperties = my)
plt.title("不同月份最后联系客户购买银行产品意向图",fontdict={"size": 25})
plt.tight_layout()
plt.show()

# print(trian["marital"].value_counts())
marital_colum=["married" ,"single" ,"divorced"]
# # 选取某列含有特定“marital”的行
trian1 = train[train['marital'].isin([marital_colum[0]])]

# 修改后的代码，只设置了 thresh 参数
trian1 = train[train['marital'].isin([marital_colum[0]])].dropna(axis=0, thresh=0.5)

'''
trian1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
'''
print(trian1["marital"].value_counts())
plt.figure(figsize=[10, 10])
sns.barplot(x="default", y="subscribe", hue="education", data=trian1, palette="muted")
x_1=["yes"," no"]
from matplotlib import font_manager
plt.xticks(range(len(x_1)),x_1,fontsize=25)
plt.yticks(fontsize=15)
my= font_manager.FontProperties(size=20)
plt.xlabel("有无违约记录",fontproperties = my)
plt.ylabel("产品购买数量指数",fontproperties = my)
plt.title("已婚",fontdict={"size": 25})
plt.legend(prop = {'size':18})
plt.tight_layout()
plt.show()

# print(trian["marital"].value_counts())
marital_colum=["married" ,"single" ,"divorced"]
# # 选取某列含有特定“marital”的行
trian1 = train[train['marital'].isin([marital_colum[1]])]

# 修改后的代码，只设置了 thresh 参数
trian1 = train[train['marital'].isin([marital_colum[0]])].dropna(axis=0, thresh=0.5)

'''
trian1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
'''
print(trian1["marital"].value_counts())
plt.figure(figsize=[10, 10])
sns.barplot(x="default", y="subscribe", hue="education", data=trian1, palette="muted")
x_1=["no"," yes"]
from matplotlib import font_manager
plt.xticks(range(len(x_1)),x_1,fontsize=25)
plt.yticks(fontsize=15)
my= font_manager.FontProperties(size=20)
plt.xlabel("有无违约记录",fontproperties = my)
plt.ylabel("产品购买数量指数",fontproperties = my)
plt.title("单身",fontdict={"size": 25})
plt.legend(prop = {'size':18})
plt.tight_layout()
plt.show()

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
z = 0
while (z <= 9):
    trian1 = train.loc[:, [num_columns[z], 'subscribe']]
    # ax = plt.subplot(3, 3, z + 1)
    f = pd.melt(trian1, value_vars=num_columns[z], id_vars='subscribe')
    g = sns.FacetGrid(f, col='variable', hue='subscribe')
    z = z + 1
    g = g.map(sns.distplot, "value", bins=20)
    plt.show()

from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, auc, roc_auc_score

X = df.drop(columns=['id', 'subscribe'])
Y = df['subscribe']
testA = test.drop(columns='id')
# 划分训练及测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

model = xgb.XGBClassifier()
# 交叉验证
result1 = []
mean_score1 = 0
n_folds = 10
import time

start = time.time()
kf = KFold(n_splits=n_folds, shuffle=True, random_state=2022)
for train_index, test_index in kf.split(X):
    x_train = X.iloc[train_index]
    y_train = Y.iloc[train_index]
    x_test = X.iloc[test_index]
    y_test = Y.iloc[test_index]
    model.fit(x_train, y_train)
    y_pred1 = model.predict_proba((x_test))[:, 1]
    print('验证集AUC:{}'.format(roc_auc_score(y_test, y_pred1)))
    mean_score1 += roc_auc_score(y_test, y_pred1) / n_folds
    y_pred_final1 = model.predict_proba((testA))[:, 1]
    y_pred_test1 = y_pred_final1
    result1.append(y_pred_test1)
end = time.time()
print('程序运行时间为: %s Seconds' % (end - start))

cat_pre1 = sum(result1) / n_folds
ret1 = pd.DataFrame(cat_pre1, columns=['subscribe'])
ret1['subscribe'] = np.where(ret1['subscribe'] > 0.5, 'yes', 'no').astype('str')
ret1.to_csv('./XGB预测.csv', index=False)