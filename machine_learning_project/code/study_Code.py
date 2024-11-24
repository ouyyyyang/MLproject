import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# 导入数据
file_path = '../data/loan_data.csv'
loan_data = pd.read_csv(file_path)
# 加载并检查数据
loan_data.info(),loan_data.head()
# 显示数据的基本属性
print(loan_data.describe(include='object'))
# 统计缺失值
print(loan_data.isnull().sum())
# 处理缺失值，为保留特征内部统计特性，使用均值或中位数等填充
loan_data['loan_percent_income'] = loan_data['loan_percent_income'].fillna(loan_data['loan_percent_income'].mean())
loan_data['credit_score'] = loan_data['credit_score'].fillna(loan_data['credit_score'].median())
print(loan_data.describe(include='object'), loan_data.isnull().sum())


# 进行探索性数据分析
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['text.usetex'] = False  # 禁用数学公式以显示下划线
# 2.1 单变量图，取贷款百分比收入作为示例展示
# 首先查看结果:创造subplots去查看loan status的情况
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
# 对其中的值进行统计
loan_status_counts = loan_data['loan_status'].value_counts()
# 用柱状图作为其中一个子图，显示loan status的分布情况
sns.barplot(x=loan_status_counts.index, y=loan_status_counts, ax=axes[0], hue=loan_status_counts.index, palette='tab10')
axes[0].set_title('Loan Status的分布情况')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Loan Status (0 = 拒绝, 1 = 同意)')
# 设置标注在条柱上方显示数量
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height()}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 10), textcoords='offset points')
sns.despine(left=True, bottom=True)
# 饼状图显示loan status的分布情况
loan_status_percentage = loan_status_counts / loan_status_counts.sum() * 100
axes[1].pie(loan_status_percentage, labels=loan_status_percentage.index, autopct='%1.1f%%',
            colors=sns.color_palette('tab10'))
axes[1].set_title('Loan Status的分布情况')
plt.legend(['拒绝 (0)', '同意 (1)'])
plt.tight_layout()  # 自调整子图间距，避免重叠
plt.show()
# 上图表明，与已批准的贷款相比，被拒绝的贷款数量更高，这表明在构建预测模型时应解决类别不平衡问题
# 构建数值列的单变量图进行单变量分析，用univariate_analysis函数实现
def univariate_analysis(data, columns):
    plt.figure(figsize=(10, 12))
    for i, column in enumerate(columns, 1):  # 内置函数，用于遍历可迭代对象
        plt.subplot(4, 2, i)
        sns.histplot(data[column], kde=True, bins=30, color='dodgerblue')
        plt.title(f'{column}  的KDE分布')  # KDE(核密度估计曲线)，描述数据的分布情况,KDE的存在有助于平滑分布，从而更容易可视化
        plt.xlabel(column)
        plt.ylabel('频率')
    plt.tight_layout()
    plt.show()

# 数值特征列
columns_to_analyze = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
                      'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
# 调用画图函数
univariate_analysis(loan_data, columns_to_analyze)

# Person age:年龄分布略微右偏，数据集中的大多数人年龄在 20 到 40 岁之间
# Person Income：收入分配高度右偏，很大一部分收入集中在较低的值。存在一些非常高的收入值，如果不加以解决，可能会影响模型性能的潜在异常值
# Person Employment Experience:大多数人的经验少于 10 年，随着年龄的增长，频率会迅速下降，但是一些实例显示非常高的值，可能是异常值，如果不解决，可能会使分析出现偏差
# Loan Amount:贷款金额集中在较低的价值，这表明大多数申请人要求的贷款额较小，分布逐渐减少，少数申请人要求高额贷款
# Loan Interest Rate：利率主要集中在 10% 至 15% 左右，在 5% 到 10% 之间也有显著的密度，这可能表明申请人的风险较低
# Loan Percent Income：分布表明，大多数贷款金额只占申请人收入的一小部分，通常不到20%。少数的百分比较高，表明相对于贷款金额，申请人的风险更高或收入更低
# Credit History Length：信用记录长度在 3 到 5 年左右达到峰值，拥有 10 年信用记录的人较少，反映年轻人群或刚接触信用系统的个人
# Credit Score:信用评分通常分布在中间范围 （600-700） 左右，分布在 850 附近逐渐下降


# 去除异常值：使用箱线图或基于统计量（如1.5倍IQR）的规则识别并去除异常值
def univariate_analysis(data, column, title):
    print(f'\n{title}的整体状态为:\n', data[column].describe())
    plt.figure(figsize=(10, 2))
    sns.boxplot(x=data[column], color='sandybrown')
    plt.title(f'{title} 箱线图')
    plt.tight_layout()
    plt.show()

# 构建数值列的箱线图进行异常值分析
for column in columns_to_analyze:
    univariate_analysis(loan_data, column, column.replace('_', ' '))

# Person age:年龄从20岁到144岁不等，中位年龄为26岁。较高的最大值表示一些异常值(144岁还来贷款，明显不合理)，IQR相当窄，大多数值在24到30之间
# Person Income：收入分配范围很广，从8,000到720多万不等，中位数约为67,048,较高的最大值表示极端异常值,数值相差太大会使模型出现偏差
# Person Employment Experience:大多数值低于10年，中位数为4年，大多集中在1年到8年内，最长125年不合理，表示异常值或数据异常
# Loan Amount:贷款金额从500到35,000不等，中位数为8,000，这个整体倒是合理一些
# Loan Interest Rate：利率从5.42%到20%不等，中位数为11.01%，大多数利率集中在8.59%至12.99%之间，符合正常利率
# Loan Percent Income：范围从0到0.66，中位数为0.12，表明大多数贷款不到借款人收入的20%，0.66 附近的高值代表申请人的风险更高或收入更低
# Credit History Length：信用记录从2年到30年不等，中位数为4年，大多数申请人的信用记录较短，反映贷款人群以年轻人为主
# Credit Score:信用评分从390到850不等，中位数为640，分布似乎在632的平均值附近相当对称，大多数值都位于列内

# 去除年龄异常值,将高于特定阈值（100）的年龄替换为数据集的年龄中位数，以保持真实的分布
# 将极值替换为中位数有助于消除不切实际的值，而无需删除任何行，从而保持数据集的完整性
# median_age = loan_data['person_age'].median()
# loan_data['person_age'] = loan_data['person_age'].apply(lambda x: median_age if x > 80 else x)
# print('person age的整体状况为：', loan_data['person_age'].describe())
# # 去除工作年龄异常值，以保持真实的分布
# med_age = loan_data['person_emp_exp'].median()
# loan_data['person_emp_exp'] = loan_data['person_emp_exp'].apply(lambda x: med_age if x > 60 else x)
# print('Person Employment Experience的整体状况为：', loan_data['person_emp_exp'].describe())
for column in columns_to_analyze:
    # 计算Q1、Q3和IQR
    Q1 = loan_data[column].quantile(0.25)
    Q3 = loan_data[column].quantile(0.75)
    IQR = Q3 - Q1
    # 设定异常值的上下限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 计算中位数
    median = loan_data[column].median()
    # 替代异常值为中位数
    loan_data[column] = loan_data[column].where((loan_data[column] >= lower_bound) & (loan_data[column] <= upper_bound), median)
print("用中位数替代异常值后的数据框：")
print(loan_data)
# 可视化loan_data中非数值列的分布情况，通过条形图和饼状图展现类别型数据的分布特点，获取非数值列的基本信息
def plot_categorical_distribution(column_name, data=loan_data):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    # countplot：用于自动计算每个类别的频数，并在图表上显示出来。hue：可以进一步按照另一分类变量为每个条形分配不同的颜色
    sns.countplot(y=column_name, data=loan_data, palette='muted',hue=column_name,legend=False)
    plt.title(f'{column_name}的分布情况')
    ax = plt.gca()  # 获取当前的坐标轴对象
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='center', va='center', xytext=(10, 0), textcoords='offset points')
    sns.despine(left=True, bottom=True)  # 快速去除上面和右边的边框
    plt.subplot(1, 2, 2)
    # autopct:设置饼图每个扇区的百分比格式,保留 1 位小数并显示百分比;muted:使用 seaborn 提供的 muted 色板来设置饼图的颜色;
    # startangle:设置饼图的起始角度为 90°，让第一个扇区从顶部开始;explode:用于控制哪些扇区稍微突出;
    loan_data[column_name].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('muted'), startangle=90,
                                            explode=[0.05] * loan_data[column_name].nunique())
    plt.title(f'{column_name}的百分比分布饼状图')
    plt.ylabel('')

    plt.tight_layout()
    plt.show()
plot_categorical_distribution('person_gender')
plot_categorical_distribution('person_education')
plot_categorical_distribution('person_home_ownership')
plot_categorical_distribution('loan_intent')
plot_categorical_distribution('previous_loan_defaults_on_file')

# 对于非数值列分布：
# Person Gender:在性别方面相对平衡，略微偏向于男性
# Person Education：大多数申请者拥有高中、学士或硕士学位，拥有副学士学位或博士学位的申请者较少
# Person Home Ownership:大多数申请人要么租房，要么拥有房屋，少数人拥有抵押贷款或被归类为“其他”；具有不同房屋所有权状况的申请人具有不同的财务稳定性，从而影响借贷风险
# Loan Intent：贷款用途多种多样，有个人使用、债务合并、医疗费用和教育等，目的不同风险不同，影响贷款审批标准
# Previous Loan Defaults on File：大多数申请人以前没有贷款违约记录，但仍有不少人违约。该列强烈影响贷款决策，因为过去的违约表明风险更高

#--------------------------------双变量分析-----------------------------------------------------

# 可视化 数值类特征 vs Loan Status
numerical_columns_1 = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt']
numerical_columns_2 = ['loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

fig, axes = plt.subplots(2, 2, figsize=(16, 20))
fig.suptitle('数值类特征 vs Loan Status (密度图)', fontsize=16)
for i, col in enumerate(numerical_columns_1):
    # 绘制核密度估计曲线
    sns.kdeplot(data=loan_data, x=col, hue='loan_status', ax=axes[i//2, i % 2], fill=True, common_norm=False, palette='muted')
    axes[i//2, i%2].set_title(f'{col} vs Loan Status')
    axes[i//2, i%2].set_ylabel('密度')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16, 20))
fig.suptitle('数值类特征 vs Loan Status (密度图)', fontsize=16)
for i, col in enumerate(numerical_columns_2):
    # 绘制核密度估计曲线
    sns.kdeplot(data=loan_data, x=col, hue='loan_status', ax=axes[i // 2, i % 2], fill=True, common_norm=False,
                palette='muted')
    axes[i // 2, i % 2].set_title(f'{col} vs Loan Status')
    axes[i // 2, i % 2].set_ylabel('密度')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

fig, axes = plt.subplots(len(numerical_columns_1), 1, figsize=(10,20 ))
fig.suptitle('Loan Status与数值类特征的箱线图', fontsize=16)
for i, feature in enumerate(numerical_columns_1):
    sns.boxplot(data=loan_data, x='loan_status', y=feature, ax=axes[i], hue='loan_status', palette='muted', legend=False)
    axes[i].set_title(f'{feature} vs Loan Status')
    axes[i].set_xlabel('Loan Status')
    axes[i].set_ylabel(feature)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

fig, axes = plt.subplots(len(numerical_columns_2), 1, figsize=(10,20 ))
fig.suptitle('Loan Status与数值类特征的箱线图', fontsize=16)
for i, feature in enumerate(numerical_columns_2):
    sns.boxplot(data=loan_data, x='loan_status', y=feature, ax=axes[i], hue='loan_status', palette='muted', legend=False)
    axes[i].set_title(f'{feature} vs Loan Status')
    axes[i].set_xlabel('Loan Status')
    axes[i].set_ylabel(feature)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# 对数值类特征与目标之间的相关性分析：
# person age：获批贷款的年龄中位数略小，但分布差异很小；被拒绝的贷款的分布范围更广，异常值位于上限，可能表明较高的年龄是一个次要风险因素
# person income：获批贷款通常对应于收入较高的申请人，已批准贷款的收入中位数明显更高，并且已批准的申请存在许多高收入异常值，这表明收入对批准有积极影响
# Person Employment Experience：对于拒绝和获批贷款的人而言，整体上的密度分布范围基本相同，表明从业经验可能对获批贷款并没有太大的影响
# Loan Amount：批准和拒绝的贷款金额相对相似，但在被拒绝的贷款中观察到的中位数略低，这可能表明较大的贷款更容易被同意，但差异并不大，并不明显
# Loan Interest Rate:与被拒绝的贷款相比，批准的贷款的平均利率往往略低
# Loan Percent Income：贷款获批的申请人的贷款收入比通常较低，这表明占收入比例较小的贷款更有可能获得批准；被拒绝贷款的高贷款收入比表明，当贷款金额占收入的很大一部分时，拒绝的概率更大
# Credit History Length：对于已批准的贷款，具有较长信用记录的人占比较大，这表明具有既定信用记录的申请人获得批准的可能性更高；反映了贷方偏爱具有更多信贷经验的借款人
# credit score：批准的贷款与更高的信用评分相关，信用更高的人同意贷款的概率更大，这种显著差异凸显了信用评分是贷款批准的有力预测指标，分数越高反映风险越低

# 构建双变量图探查以往贷款数额、贷款利率与贷款状态之间的关系
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.violinplot(x='loan_status', y='loan_amnt', data=loan_data)  # 显示数据的集中趋势，又能展示数据的分布形态
plt.title('Loan Amount 在 Loan Status 中的分布情况')
plt.subplot(1, 2, 2)
sns.violinplot(x='loan_status', y='loan_int_rate', data=loan_data)
plt.title('Loan Interest Rate 在 Loan Status 中的分布情况')
plt.tight_layout()
plt.show()
# 在loan amount中，被拒贷款里数额低的贷款占比更大，而中高数额贷款更容易被同意
# 在loan Interest Rate中，被拒贷款的利率大多较低，而中高利率的贷款更容易被同意

#各列的特征与目标之间的相关性
target_col = 'loan_status'

#数值类特征
numeric_cols = numerical_columns_1 + numerical_columns_2

# 计算点双列相关系数，分析连续变量与目标二分类变量之间的相关性，逐列计算然后将结果放置到集合中，绘制到同一张图上
# 存储分析结果
correlation_results = {}
# 计算相关性和P值
for col in numeric_cols:
    correlation, p_value = pointbiserialr(loan_data[col], loan_data[target_col])
    correlation_results[col] = {
        '相关性': correlation,
        'P-值': p_value
    }
correlation_df = pd.DataFrame(correlation_results).T
print(correlation_df)

# 点双列相关系数绘图
def plot_point_biserial(correlation_df):
    # 按相关性降序排序
    correlation_df = correlation_df.sort_values(by='相关性', ascending=False)
    plt.figure(figsize=(10, 6))
    # 绘制横向条形图
    plt.barh(correlation_df.index, correlation_df['相关性'], color='skyblue', edgecolor='black')
    plt.xlabel('点双列相关系数 (Point Biserial Correlation)', fontsize=12)
    plt.ylabel('连续变量', fontsize=12)
    plt.title('连续变量与目标变量的相关性', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
# 调用函数
plot_point_biserial(correlation_df)

# 非数值特征
# 探索特征与目标之间的相关性
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Loan Status与各种分类特征之间的关系", fontsize=18)

# 可视化 person_gender 与 loan_status
sns.countplot(data=loan_data, x='person_gender', hue='loan_status', ax=axes[0, 0], palette='muted')
axes[0, 0].set_title("Loan Status 与 Gender")
axes[0, 0].set_xlabel("Gender")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend(title='Loan Status', labels=['0 = 拒绝', '1 = 接受'])

# 可视化 person_education 与 loan_status
sns.countplot(data=loan_data, x='person_education', hue='loan_status', ax=axes[0, 1], palette='muted')
axes[0, 1].set_title("Loan Status 与 Education Level")
axes[0, 1].set_xlabel("Education Level")
axes[0, 1].set_ylabel("Count")
axes[0, 1].legend(title='Loan Status', labels=['0 = 拒绝', '1 = 接受'])
axes[0, 1].tick_params(axis='x', rotation=45)

# 可视化 person_home_ownership 与 loan_status
sns.countplot(data=loan_data, x='person_home_ownership', hue='loan_status', ax=axes[0, 2], palette='muted')
axes[0, 2].set_title("Loan Status 与 Home Ownership")
axes[0, 2].set_xlabel("Home Ownership")
axes[0, 2].set_ylabel("Count")
axes[0, 2].legend(title='Loan Status', labels=['0 = 拒绝', '1 = 接受'])

# 可视化 loan_intent 与 loan_status
sns.countplot(data=loan_data, x='loan_intent', hue='loan_status', ax=axes[1, 0], palette='muted')
axes[1, 0].set_title("Loan Status 与 Loan Intent")
axes[1, 0].set_xlabel("Loan Intent")
axes[1, 0].set_ylabel("Count")
axes[1, 0].legend(title='Loan Status', labels=['0 = 拒绝', '1 = 接受'])

# 可视化 previous_loan_defaults_on_file 与 loan_status
sns.countplot(data=loan_data, x='previous_loan_defaults_on_file', hue='loan_status', ax=axes[1, 1], palette='muted')
axes[1, 1].set_title("Loan Status 与 Previous Loan Defaults")
axes[1, 1].set_xlabel("Previous Loan Defaults")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend(title='Loan Status', labels=['0 = 拒绝', '1 = 接受'])
# 隐藏最后一个子图，因为第六个图用不上
fig.delaxes(axes[1][2])

plt.tight_layout(rect=[0, 0.03, 1, 0.99])
plt.show()

# Person Gender:贷款批准和拒绝在性别之间相当平衡，表明性别可能不是贷款审批结果的重要决定因素

# Person Education：与教育水平较低的申请人相比，受教育程度较高的申请人的贷款批准次数更高
# 男性和女性申请人在不同教育阶段的贷款批准率和拒绝率方面表现基本相同，小部分略有不同
# 只有高中教育的申请人被拒绝的次数似乎多于批准次数，这表明教育水平可能会影响贷款结果
# 受过更多教育的申请人（例如，拥有硕士和学士学位的申请人）更有可能获得批准，这表明教育水平是贷款批准的积极指标，可能反映了更稳定的财务状况
# 受教育程度较低的申请人的拒绝率更高，这可能表明贷方认为风险更高，这些模式表明，教育水平是贷款审批决策的一个影响因素，并且在性别间的交互作用相似

# Person Home Ownership:与有抵押贷款或拥有房屋的人相比，租房的人似乎有更高的贷款拒绝率，表明房屋所有权状况被认为是一个风险因素，因为租房者的财务稳定性可能不如房主
# 两种性别的趋势基本相同，但租房基数更高，贷款成功率更高，可能是因为年轻人购买需求大(住房、车)等，还款潜力大

# Loan Intent：某些贷款目的，如债务合并和个人贷款，贷款拒绝比批准多；企业和教育的贷款似乎具有相对平衡的批准率和拒绝率，可能是被认为具有创收的潜力
# 具图分析，债务合并(debtconsol dation)和医疗意向的同意率更高，相比之下，用于教育、投资和个人贷款的同意率更低

# Previous Loan Defaults on File：与没有违约史的申请人相比，有过贷款违约历史的申请人的拒绝率要高得多，几乎为100%，此特征可能对贷款状况有很大影响
# 以前有过贷款违约史的申请人，贷款拒绝率为100%,这表明贷款违约历史是贷款审批决定的一个很大的负面因素
# 而以前没有贷款违约的人明显有更多的批准，这凸显了干净的信用记录与更高的批准率相关，表明以往贷款是否违约是贷款批准的关键因素

#卡方检验
# 分类变量，分析非连续量与目标变量之间的相关性，逐列计算然后将结果放置到集合中，绘制到同一张图上
categorical_cols = [
    'person_gender', 'previous_loan_defaults_on_file', 'person_education',
    'person_home_ownership', 'loan_intent']
chi_square_results = {}

# 定义计算 Cramér's V 的函数
def cramers_v(chi2, n, k, r):
    return np.sqrt(chi2 / (n * min(k - 1, r - 1)))

for col in categorical_cols:
    # 创建列联表
    contingency_table = pd.crosstab(loan_data[col], loan_data[target_col])
    # 卡方检验
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    # 计算样本总数
    n = contingency_table.sum().sum()
    # Cramér's V 来衡量相关性的强弱
    k = contingency_table.shape[1]  # 目标变量类别数
    r = contingency_table.shape[0]  # 当前变量类别数
    cramers_v_value = cramers_v(chi2, n, k, r)
    # 存储结果
    chi_square_results[col] = {
        'Chi-Square': chi2,
        'P-值': p,
        'Cramer\'s V': cramers_v_value
    }
chi_square_df = pd.DataFrame(chi_square_results).T
print(chi_square_df)

# 卡方检验 (Cramér's V) 绘图
def plot_cramers_v(chi_square_df):
    # 按相关性(Cramér's V)排序
    chi_square_df = chi_square_df.sort_values(by='Cramer\'s V', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(chi_square_df.index, chi_square_df['Cramer\'s V'], color='lightgreen', edgecolor='black')
    plt.xlabel('Cramer\'s V', fontsize=12)
    plt.ylabel('分类变量', fontsize=12)
    plt.title('分类变量与目标变量的相关性 (Cramer\'s V)', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

plot_cramers_v(chi_square_df)

# 2.5.1 绘制 Pairs Plot，展示数值列的状态
#################################################################################################################
numerical_columns_with_target = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
sns.pairplot(loan_data[numerical_columns_with_target + ['loan_status']],
             hue='loan_status',
             palette='muted'
            )
plt.show()
# 绘制数据集中的多个数值特征之间的散点图矩阵,pairplot用于可视化多维数据分布和变量之间的关系，特别适合探索数据的潜在模式



# 对机器学习数据集进行特征选择
# 数据预处理
# 将非数值型数据列进行编码：二进制编码(Binary Encoding):适用于二值化类型，映射为0或1
# 顺序编码(Ordinal Encoding)：将类别值映射为整数值，并保留其顺序特性.较低的数字代表较低的类别，较高的数字代表较高的类别
# 独热编码(One-Hot Encoding):将类别变量转换为数值变量的编码方式，适用于没有顺序关系的类别数据,每个类别视为独立的特征，并为每个类别创建一个二进制的虚拟变量
#  二进制编码
loan_data['person_gender'] = loan_data['person_gender'].map({'female': 0, 'male': 1})
loan_data['previous_loan_defaults_on_file'] = loan_data['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
# 顺序编码
education_order = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}
loan_data['person_education'] = loan_data['person_education'].map(education_order)
# 独热编码
loan_data = pd.get_dummies(loan_data, columns=['person_home_ownership', 'loan_intent'], drop_first=True)
# drop_first=True这个参数代表会删除每个类别特征中的第一个类别列，之所以删除第一个类别列，
# 是为了防止多重共线性。如果不删除第一个类别列，模型可能会因为缺少基准类别（即截距项）而出现不必要的共线性。
# 删除的类别列充当了基准类别，其他类别则相对于该基准类别进行表示，即基准化
print(loan_data.isna().sum())
print(loan_data.head())  # 查看编码完成后的情况

# 绘制相关性热力图，展示数据集中各个特征之间的相关关系，了解哪些特征之间具有较强的正相关或负相关
# 如果两个特征之间的相关性非常高(大于0.9)，可能会导致多重共线性问题。在这种情况下可以选择删除其中一个特征，减少冗余信息
# 查看哪些特征与标签有较强的相关性，选择对预测有较大影响的特征(特征选择)。
corr_matrix = loan_data.corr(method='spearman')

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('相关性热力图')
plt.show()
target_variable = 'loan_status'  # 目标变量
target_corr = corr_matrix[[target_variable]].sort_values(by=target_variable, ascending=False)
plt.figure(figsize=(4, 6))
sns.heatmap(target_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title(f'{target_variable}的相关性')
plt.show()
# 分析：与 loan_status 正相关特征
# loan_percent_income（0.38）：这是与loan_status最强的正相关，表明贷款金额相对于其收入较高的申请人获得批准的可能性可能更高
# loan_int_rate（0.33）：较高的利率与审批状态呈正相关，这可能表明风险较高的申请人或利率较高的申请人仍经常获得批准
# person_home_ownership_RENT（0.26）：租房状态呈正相关，表明租房者的批准率可能高于其他房屋所有权状态
# loan_amnt（0.11）：贷款金额具有较弱的正相关，表明贷款金额较高的人获批倾向较小
# medical（0.07）和HomeImproveRent（0.03）等贷款意向也显示出较弱的正相关，表明有些特定的贷款用途可能会略微影响审批结果
# 与loan_status负相关的特征：
# previous_loan_defaults_on_file（-0.54）：这是最显著的负相关特征，表明以前的贷款违约历史是降低批准几率的重要因素
# person_income（-0.14）：较高的收入与贷款审批呈弱负相关，这表明收入较高的申请人可能会申请高风险贷款......
# person_home_ownership_OWN（-0.09）：自有住房状况略有负相关，这表明拥有住房的人获批几率可能会低一些
# venture（-0.09）和education（-0.06）等贷款意向与审批呈负相关，这可能是因为这些贷款用途具有较高的风险
# 综上：贷款批准的最强预测因素是贷款百分比收入、贷款利率和过往贷款违约记录这三个特征

# 删除互相关程度高的特征
loan_data.drop(columns=['person_emp_exp', 'cb_person_cred_hist_length'], inplace=True)

# 3.3 划分训练集和测试集
# 从数据集中分离特征和目标变量
X = loan_data.drop(['loan_status'],axis=1)
y = loan_data['loan_status']
print(X.head())  # 查看是否分离成功
print(y.head())

# 划分训练集和测试集，指定测试集的大小为20%，训练集大小为80%，设置随机种子，确保每次拆分数据时结果一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# 将训练集和测试集存入excel表格中保存
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
with pd.ExcelWriter('../data/train_test_data.xlsx') as writer:
    train_data.to_excel(writer, sheet_name='Train Data', index=False)
    test_data.to_excel(writer, sheet_name='Test Data', index=False)

# 4：应用机器学习构建模型，并在测试上评估最佳模型
# 使用StandardScaler来对训练集和测试集的数据进行标准化:将每个特征缩放到均值为0,标准差为1的分布上
scaler = StandardScaler()
X_train_standard = scaler.fit_transform(X_train)
X_test_standard = scaler.transform(X_test)
# 使用MinMaxScaler来对训练集和测试集的数据进行归一:将每个特征缩放到均值为0和1之间
minmax_scaler = MinMaxScaler()
X_train_scaled = minmax_scaler.fit_transform(X_train_standard)
X_test_scaled = minmax_scaler.transform(X_test_standard)

# 构建模型
# 选用逻辑回归模型、极端梯度提升模型(XGBoost)、基于梯度提升决策树(CatBoost)、轻量级梯度提升机(LightGBM)、随机森林(Random Forest)、支持向量机(SVM)
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    # verbose=0：无输出，不显示任何日志信息。verbose=1：显示最基本的输出，如训练开始和结束时的消息。verbose=2：显示更详细的输出信息
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
    'XGBoost': XGBClassifier(random_state=42),
    'Classification SVM': SVC(kernel='rbf', C=1.0, gamma='scale')   # 'scale':自动设置的常见选择
    # kernel:指定核函数，包括'linear'(线性核)、'poly'(多项式核)、'rbf'(径向基核)C: 正则化参数，控制误差的惩罚程度 gamma:核函数的参数，控制样本的影响范围
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)  # 采用fit方法训练模型，喂数据

param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],  # 正则化强度，值越小，正则化效果越强
        'tol': [1e-1, 1e-2, 1e-3, 1e-4],  # 优化过程中的容忍度，收敛的阈值
        'solver': ['lbfgs', 'saga', 'liblinear'],  # 优化算法选择，接受不同的输入数据类型和规模
        'max_iter': [200, 500, 1000, 3000]  # 收敛之前的最大迭代次数
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],  # 森林中树的数量，更多的树通常会提高性能
        'max_depth': [None, 10, 20],  # 单棵树的最大深度，以限制过拟合
        'min_samples_split': [2, 5, 10],  # 拆分内部节点所需的最小样本数
        'min_samples_leaf': [1, 2, 4]  # 在叶子节点上所需的最小样本数，防止过拟合
    },
    'LightGBM': {
        'num_leaves': [20, 30, 40],  # 平衡要使用的叶子数，过多的叶子可能导致过拟合
        'max_depth': [4, 6, 8],  # 树的最大深度，以限制模型复杂度
        'learning_rate': [0.01, 0.05, 0.1],  # 学习率，控制模型的更新步伐
        'n_estimators': [100, 200, 300],  # 弱学习器的数量
        'feature_fraction': [0.6, 0.8, 1.0],  # 每棵树使用的特征比例，防止过拟合
        'bagging_fraction': [0.6, 0.8, 1.0]  # 每棵树使用的数据比例，用于随机采样
    },
    'CatBoost': {
        'iterations': [100, 200, 300],  # 迭代次数，构建的树的数量
        'depth': [4, 6, 8],  # 树的深度，控制复杂度
        'learning_rate': [0.01, 0.05, 0.1],  # 学习率，影响模型的收敛速度
        'l2_leaf_reg': [1, 3, 5]  # L2 正则化参数，控制过拟合
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],  # 森林中树的数量，更多的树通常会提高性能
        'max_depth': [4, 6, 8],  # 树的最大深度，控制模型的复杂度
        'learning_rate': [0.01, 0.05, 0.1],  # 学习率，控制模型的更新步伐
        'colsample_bytree': [0.6, 0.8, 1.0],  # 每棵树所使用的特征列子采样比
        'subsample': [0.6, 0.8, 1.0]  # 每棵树随机采样的训练数据比例，防止过拟合
    },
    'Classification SVM': {
        'C': [0.01, 0.1, 1, 10, 100],  # 正则化参数，控制分类平面的位置和复杂度
        'gamma': ['scale', 'auto', 0.01, 0.1, 0.5],  # 核函数的参数，定义支持向量的影响范围
        'kernel': ['rbf', 'linear', 'poly'],  # 核函数类型，影响决策边界的形式
        'degree': [3, 4, 5],  # 多项式核的次数，仅适用于 poly 核
        'tol': [1e-1, 1e-2, 1e-3],  # 优化过程中使用的停止条件
        'max_iter': [5000, 10000]  # 最大迭代次数，以避免过长训练时间
    }
}

tune_models = {
    'Logistic Regression': LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(verbose=False, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'Classification SVM': SVC(random_state=42)
}

best_models = {}
model_f1_scores = {}  #存储每个模型的f1分数值
for model_name in tune_models.keys():
    print(f"对 {model_name} 调优")
    # tune_models：需要调优的模型 param_grids：模型对应的超参数搜索空间 n_iter：随机搜索的迭代次数 scoring：以准确率作为模型性能的评估指标
    # cv: 交叉验证的折数,进行5折交叉验证 random_state：设置随机数生成器的种子 n_jobs：表示使用所有可用的CPU核心来并行化计算，减少运行时间
    random_search = RandomizedSearchCV(tune_models[model_name], param_grids[model_name], n_iter=10, scoring='f1', cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_train_scaled, y_train)
    best_models[model_name] = random_search.best_estimator_  # 选择交叉验证评估后，表现最好的模型存入
    model_f1_scores[model_name] = random_search.best_score_
    print(f"最优化参数: {random_search.best_params_}")
    print(f"最佳f1分数: {random_search.best_score_}")

    # 使用最佳模型在测试集上进行预测
    y_pred = random_search.best_estimator_.predict(X_test_scaled)  # 使用训练好的模型对测试集X_test_scaled进行预测

    # 计算并打印测试集上的准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} 测试集准确率: {accuracy}")

    # 计算并生成分类报告，包含模型的precision（精确率）、recall（召回率）、f1-score等指标
    classification_rep = classification_report(y_test, y_pred)
    print(f"{model_name} 分类报告:\n{classification_rep}")

    # 计算并生成混淆矩阵，展示模型在每个类别的预测结果
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"{model_name} 混淆矩阵:\n{conf_matrix}")

    print("-" * 80)

# 作图可视化每个模型的F1分数
models = list(model_f1_scores.keys())
f1_scores = list(model_f1_scores.values())

# 绘制条形图
plt.figure(figsize=(10, 6))
bars = plt.bar(models, f1_scores, color='lightcoral')

# 添加标题和标签
plt.title('每个模型的 f1 分数比较图 ')
plt.xlabel('模型')
plt.ylabel('F1 分数')

# 给每个条状添加F1值标签
for bar, f1 in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{f1:.5f}', ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)
plt.tight_layout()  # 调整布局避免标签重叠
plt.show()

# 得出结论：以上六个模型的表现都非常良好，都获得了很高的准确率，其中XBoost的模型表现最好，拥有0.845的f1分数，测试集测试拥有93.47%的最高准确率
