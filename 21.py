import pickle
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier


class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        self._show_result()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A,
                                                                                                    auc_A) * 1 / len(
            V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B,
                                                                                                    auc_B) * 1 / len(
            V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,
                                                                                                       auc_A,
                                                                                                       auc_B) * 1 / len(
            V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = stats.norm.sf(abs(z)) * 2

        return z, p

    def _show_result(self):
        z, p = self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.8f};")
        if p < self.threshold:
            print("There is a significant difference")
        else:
            print("There is NO significant difference")

file_path = "./sepsis.txt"

# If the file is in a format like CSV, you can use pandas to read it.
# You might need to adjust the separator depending on the file format.
# try:
#     data = pd.read_csv(file_path, sep='\t')  # Adjust separator if needed
#     data_second_hospital = pd.read_csv("./newdata.txt", sep='\t')  # Adjust separator if needed
#     print("File read successfully. Here's a preview of the data:")
#     print(data.head())
# except Exception as e:
#     print("An error occurred:", e)

# Reload the data using the correct tab delimiter
data = pd.read_csv(file_path, delimiter='\t')
data_second_hospital = pd.read_csv("./newdata.txt", sep='\t')  # Adjust separator if needed
data.to_csv('./spesis.csv', index=False)
df3 = pd.concat([data, data_second_hospital]).reset_index(drop=True)
# df3.to_csv('./spesis_new.csv', index=False)

# Display the first few rows of the dataframe again to confirm the correct structure
data = pd.read_csv("./sepsis_new.csv")

import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# 定义二分类变量和分类变量列表
binary_vars = ['mortality', 'gender', 'pre_shock', 'Inhalation_Damage', 'new_onset_shock',
               'new_onset_infection', 'Wound_culture', 'MDR', 'open_decompression',
               'tracheotomy', 'Diabetes', 'hypertension']
categorical_vars = ['type_of_burn']

# 将数据按sepsis的值（0和1）分成两组
sepsis_0 = data[data['sepsis'] == 0]
sepsis_1 = data[data['sepsis'] == 1]

# 定义要存储的结果
results = []

# 对于每个属性，计算相应的统计信息
for column in data.columns:
    if column == 'sepsis':
        continue
    if column in binary_vars or column in categorical_vars:
        # 计算每个分类的数量和占比
        count_0 = sepsis_0[column].value_counts()
        count_1 = sepsis_1[column].value_counts()
        prop_0 = round(count_0 / len(sepsis_0), 2)
        prop_1 = round(count_1 / len(sepsis_1), 2)

        # 构造列联表并进行卡方检验来计算p值
        contingency_table = pd.crosstab(data['sepsis'], data[column])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        p_value = round(p_value, 2)

        # 将结果存入列表
        results.append({
            'Attribute': column,
            'Count_Sepsis_0': count_0.to_dict(),
            'Prop_Sepsis_0': prop_0.to_dict(),
            'Count_Sepsis_1': count_1.to_dict(),
            'Prop_Sepsis_1': prop_1.to_dict(),
            'p_value': p_value
        })
    else:
        # 计算非二分类和非分类变量的均值和四分位数
        mean_0 = round(sepsis_0[column].mean(), 2)
        mean_1 = round(sepsis_1[column].mean(), 2)
        q25_0 = round(sepsis_0[column].quantile(0.25), 2)
        q75_0 = round(sepsis_0[column].quantile(0.75), 2)
        q25_1 = round(sepsis_1[column].quantile(0.25), 2)
        q75_1 = round(sepsis_1[column].quantile(0.75), 2)

        # 进行t检验，计算p值，并保留3位小数
        stat, p_value = ttest_ind(sepsis_0[column], sepsis_1[column], nan_policy='omit')
        p_value = round(p_value, 2)

        # 将结果存入列表
        results.append({
            'Attribute': column,
            'Mean_Sepsis_0': mean_0,
            'Mean_Sepsis_1': mean_1,
            'Q25_Sepsis_0': q25_0,
            'Q75_Sepsis_0': q75_0,
            'Q25_Sepsis_1': q25_1,
            'Q75_Sepsis_1': q75_1,
            'p_value': p_value
        })

# 转换为DataFrame方便查看
results_df = pd.DataFrame(results)
# results_df.to_csv("123.csv")
ans = pd.DataFrame(columns=["Variables", "Group", "non-Sepsis", "Sepsis", "p-value"])
for i in range(1, len(results_df)):
    name = results_df.loc[i, 'Attribute']
    if name in binary_vars:
        name = name + ' (%)'
        set1 = results_df.loc[i, 'Count_Sepsis_0']
        set2 = results_df.loc[i, 'Prop_Sepsis_0']
        nonSep_1 = str(set1[1])+' ('+str(set2[1])+')'
        nonSep_0 = str(set1[0])+' ('+str(set2[0])+')'

        set1 = results_df.loc[i, 'Count_Sepsis_1']
        set2 = results_df.loc[i, 'Prop_Sepsis_1']
        Sep_1 = str(set1[1]) + ' (' + str(set2[1]) + ')'
        Sep_0 = str(set1[0]) + ' (' + str(set2[0]) + ')'

        data0 = pd.DataFrame([[name, 0, nonSep_0, Sep_0, results_df.loc[i, 'p_value']]], columns=["Variables", "Group", "non-Sepsis", "Sepsis", "p-value"])
        data1 = pd.DataFrame([[name, 1, nonSep_1, Sep_1, results_df.loc[i, 'p_value']]], columns=["Variables", "Group", "non-Sepsis", "Sepsis", "p-value"])
        ans = pd.concat([ans, data0]).reset_index(drop=True)
        ans = pd.concat([ans, data1]).reset_index(drop=True)

    elif name in categorical_vars:
        name = name + ' (%)'
        set1 = results_df.loc[i, 'Count_Sepsis_0']
        set2 = results_df.loc[i, 'Prop_Sepsis_0']
        set3 = results_df.loc[i, 'Count_Sepsis_1']
        set4 = results_df.loc[i, 'Prop_Sepsis_1']
        for j in set(set1.keys()).union(set(set3.keys())):
            if j in set1.keys():
                nonSep = str(set1[j]) + ' (' + str(set2[j]) + ')'
            else:
                nonSep = '0 (0)'
            if j in set3.keys():
                Sep = str(set3[j]) + ' (' + str(set4[j]) + ')'
            else:
                Sep = '0 (0)'
            data0 = pd.DataFrame([[name, j, nonSep, Sep, results_df.loc[i, 'p_value']]], columns=["Variables", "Group", "non-Sepsis", "Sepsis", "p-value"])
            ans = pd.concat([ans, data0]).reset_index(drop=True)
    else:
        name = name + ' (Median[IQR])'
        set1 = results_df.loc[i, 'Mean_Sepsis_0']
        set2 = results_df.loc[i, 'Mean_Sepsis_1']

        nonSep = str(results_df.loc[i, 'Mean_Sepsis_0']) + ' [' + str(results_df.loc[i, 'Q25_Sepsis_0']) + ", " + str(results_df.loc[i, 'Q75_Sepsis_0']) + ']'
        Sep = str(results_df.loc[i, 'Mean_Sepsis_1']) + ' [' + str(results_df.loc[i, 'Q25_Sepsis_1']) + ", " + str(
            results_df.loc[i, 'Q75_Sepsis_1']) + ']'
        data0 = pd.DataFrame([[name, "", nonSep, Sep, results_df.loc[i, 'p_value']]], columns=["Variables", "Group", "non-Sepsis", "Sepsis", "p-value"])
        ans = pd.concat([ans, data0]).reset_index(drop=True)
ans.to_csv('1234.csv')

data = pd.read_csv("./sepsis_new.csv")
data = data.sample(frac=1, random_state=42)

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
# 准备特征和目标变量
X = data.drop('sepsis', axis=1)
y = data['sepsis']

# 可能需要对数据进行预处理（例如，缩放）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用 SMOTE
smote = SMOTE(random_state=40)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# 将平衡后的数据转换回 DataFrame
columns = data.columns.drop('sepsis')
X_balanced_df = pd.DataFrame(X_balanced, columns=columns)
y_balanced_df = pd.DataFrame(y_balanced, columns=['sepsis'])

# 合并特征和目标变量到一个新的 DataFrame
balanced_data = pd.concat([X_balanced_df, y_balanced_df], axis=1)

# 保存平衡后的数据到新文件
balanced_data.to_csv('./balanced_data_sepsis.csv', index=False)

summary = balanced_data.describe()

import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix



import statsmodels.api as sm

independent_vars = balanced_data.drop(columns=["ID", "sepsis"])
independent_vars = sm.add_constant(independent_vars)  # Add a constant (intercept) term

# Define the dependent variable
dependent_var = balanced_data["sepsis"]

# Fit the univariate logistic regression models and collect results
significant_results = []
for column in independent_vars.columns:
    if column != 'const':
        model = sm.Logit(dependent_var, independent_vars[[column, 'const']]).fit(disp=0)
        p_value = model.pvalues[column]
        if p_value < 0.001:
            coef = model.params[column]
            or_value = sm.families.links.logit().inverse(coef)
            ci_lower, ci_upper = model.conf_int().loc[column]
            significant_results.append({
                'Variable': column,
                'Coef': coef,
                'OR': or_value,
                'CI Lower': sm.families.links.logit().inverse(ci_lower),
                'CI Upper': sm.families.links.logit().inverse(ci_upper),
                'P Value': p_value
            })

# Convert results to a DataFrame for better display
significant_df = pd.DataFrame(significant_results)

# Remove 'AST' from the significant results if present
significant_df = significant_df[significant_df['Variable'] != 'Wound_culture']
significant_df = significant_df[significant_df['Variable'] != 'WBC']
significant_df = significant_df[significant_df['Variable'] != 'RBC']
significant_df = significant_df[significant_df['Variable'] != 'TB']
significant_df = significant_df[significant_df['Variable'] != 'BUN']
significant_df = significant_df[significant_df['Variable'] != 'Na']
significant_df_filtered = significant_df[significant_df['Variable'] != 'new_onset_infection']

# Display the filtered dataframe to the user
print(significant_df_filtered)
significant_df_filtered.to_csv('table2_sepsis.csv')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# Models to be evaluated
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=41),
    "Random Forest": RandomForestClassifier(random_state=41),
    "XGBoost":  XGBClassifier(random_state=41, booster="gbtree", max_depth=3),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM":  SVC(probability=True, random_state=41),
    "Logistic Regression": LogisticRegression(solver="saga", random_state=41),
    "GBT": GradientBoostingClassifier(random_state=41),
    "Adaboost": AdaBoostClassifier(random_state=41),
    "MLP": MLPClassifier(solver='adam', random_state=41)
}
X_final = balanced_data[significant_df_filtered['Variable'].tolist()]
y = balanced_data['sepsis']
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y, test_size=0.3, random_state=42)


# Results dictionary
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 定义五折交叉验证

# Evaluate each model
for name, model in models.items():
    # Fit the model
    # model.fit(X_train_final, y_train_final)
    # Predict on the test set
    # y_pred = model.predict(X_test_final)

    # Calculate metrics
    accuracy = np.mean(cross_val_score(model, X_final, y, cv=kf, scoring='accuracy'))
    recall = np.mean(cross_val_score(model, X_final, y, cv=kf, scoring='recall'))
    precision = np.mean(cross_val_score(model, X_final, y, cv=kf, scoring='precision'))
    f1 = np.mean(cross_val_score(model, X_final, y, cv=kf, scoring='f1'))
    # y_pred_proba = cross_val_score(model, X_final, y, cv=kf, scoring='accuracy')
    # Calculate AUC
    auc = np.mean(cross_val_score(model, X_final, y, cv=kf, scoring='roc_auc'))

    # Store the metrics
    results[name] = {"Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1 Score": f1, "AUC": auc}

    # Plot confusion matrix
    # conf_matrix = confusion_matrix(y_test_final, y_pred)
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Mortal', 'Mortal'],
    #             yticklabels=['Not Mortal', 'Mortal'])
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.title(f'Confusion Matrix - {name}')
    # plt.show()

results_df = pd.DataFrame(results).T
results_df.to_csv("21.csv")

from sklearn.metrics import roc_curve, auc

# Initialize plot
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 8))

# Calculate ROC curve and ROC area for each model
plt.legend(loc="lower right", fontsize=12)
for name, model in models.items():
    # Predict probabilities
    model.fit(X_train_final, y_train_final)
    y_pred_proba = model.predict_proba(X_test_final)[:, 1]
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    # Plot
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.3f})')

plt.rcParams.update({'font.size': 16})
# Plot Base Rate ROC
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)

# Plot details
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
shap_summary_rf_dot_pdf = "./result/roc_total_sepsis.pdf"
plt.savefig(shap_summary_rf_dot_pdf, bbox_inches='tight')
plt.show()

saved_model = RandomForestClassifier(random_state=41, n_estimators=100)
X = X[significant_df_filtered['Variable'].tolist()]
y = data['sepsis']
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
saved_model.fit(X,y)
joblib.dump(saved_model, 'rf_model.joblib')
joblib.dump(scaler, 'rf_scaler.joblib')
aq1


# Calculate precision, recall, and F1-score specifically for the 'Sepsis' class (class '1')
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize a dictionary to store the evaluation metrics for each model including AUC
evaluation_metrics = {}

# Calculate the evaluation metrics for each model
for name, model in models.items():
    y_pred = model.predict(X_test_final)
    y_pred_proba = model.predict_proba(X_test_final)[:, 1]

    # Metrics calculation
    accuracy = accuracy_score(y_test_final, y_pred)
    precision = precision_score(y_test_final, y_pred)
    recall = recall_score(y_test_final, y_pred)
    f1 = f1_score(y_test_final, y_pred)
    aucc = roc_auc_score(y_test_final, y_pred_proba)

    # Store the metrics
    evaluation_metrics[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall,
                                "F1-Score": f1, "AUC": aucc}


# Convert dictionary to DataFrame and sort by AUC
evaluation_metrics_df = pd.DataFrame(evaluation_metrics).T
evaluation_metrics_df_sorted = evaluation_metrics_df.sort_values(by="AUC", ascending=False)

transposed_evaluation_metrics_df_sorted = np.transpose(evaluation_metrics_df_sorted)

# Subdued color palette for scientific visualization
subdued_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


evaluation_metrics_chart_pdf = "./result/SepsisEvaluation_Metrics_Comparison_Chart.pdf"
plt.figure(figsize=(14, 8))
ax = transposed_evaluation_metrics_df_sorted.plot(kind='bar', color=subdued_colors)
ax.set_xlabel('Evaluation Metric')
ax.set_ylabel('Score')
ax.set_title('Comparison Across Models for Different Metrics')
ax.set_ylim(0, 1.2)
plt.xticks(rotation=45)
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(evaluation_metrics_chart_pdf, bbox_inches='tight')
plt.close()

from sklearn.inspection import permutation_importance

model = RandomForestClassifier(random_state=41)
model.fit(X_final, y)
X_final.to_csv("x_final_sepsis.csv")
s = pickle.dumps(model)
with open('rf_model.pkl','wb+') as f:#注意此处mode是'wb+'，表示二进制写入
    f.write(s)

# rf
xgb_model = models["Random Forest"]
# joblib.dump(xgb_model, 'rf_model.joblib')
result = permutation_importance(model, X_train_final, y_train_final, n_repeats=10, random_state=42)
svm_pred = xgb_model.predict(X_test_final)
# svm_feature_importances = svm_model.feature_importances_
# Creating a DataFrame for the feature names and their corresponding importances
svm_features_df = pd.DataFrame({'Feature': X_train_final.columns, 'Importance': result.importances_mean})

# Sorting the features by importance in descending order
svm_features_df_sorted = svm_features_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importance
Feature_Importance_svm_pdf = "./result/Feature_Importance_RF_21.pdf"
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=svm_features_df_sorted, palette='viridis')
plt.title('Feature Importance in RF Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig(Feature_Importance_svm_pdf, bbox_inches='tight')
plt.close()

import shap

# # Initialize the SHAP explainer for the Random Forest model
# XGB_model = XGBClassifier(random_state=4, booster="gbtree")
# XGB_model.fit(X_train_final, y_train_final)
# explainer_rf = shap.TreeExplainer(XGB_model)
# shap_values_rf = explainer_rf.shap_values(X_test_final)
# print(shap_values_rf)
# shap.summary_plot(shap_values_rf, X_test_final, show=False, max_display=7)
# plt.title('SHAP Summary Plot (Dot) for XGBoost')
# shap_summary_xgb_dot_pdf = "./result/SHAP_Summary_XGBoost_21.pdf"
# plt.savefig(shap_summary_xgb_dot_pdf, bbox_inches='tight')
# plt.close()

# lr_model = LogisticRegression()
# lr_model.fit(X_train_final, y_train_final)
# explainer_lr = shap.LinearExplainer(lr_model, X_train_final)
# shap_values_lr = explainer_lr.shap_values(X_test_final)
# shap.summary_plot(shap_values_lr, X_test_final, show=False)
# plt.title('SHAP Summary Plot (Dot) for LR')
# shap_summary_lr_dot_pdf = "./result/SHAP_Summary_LR_21.pdf"
# plt.savefig(shap_summary_lr_dot_pdf, bbox_inches='tight')
# plt.close()

lr_model = models["Random Forest"]
explainer_rf = shap.TreeExplainer(lr_model)
shap_values_rf = explainer_rf.shap_values(X_test_final)
shap_values_rf1 = shap_values_rf[:, :, 0]
shap.summary_plot(shap_values_rf1, X_test_final, show=False, max_display=11)
plt.title('SHAP Summary Plot (Dot) for RF')
shap_summary_lr_dot_pdf = "./result/SHAP_Summary_RF_21.pdf"
plt.savefig(shap_summary_lr_dot_pdf, bbox_inches='tight')
plt.close()

base_value = explainer_rf.expected_value
plt.figure()
shap.initjs()
shap_values_rf = shap_values_rf.transpose(0, 2, 1)
html = shap.force_plot(base_value[0], shap_values_rf[0][0], X_test_final.iloc[0], matplotlib=False, show=False)

# 显示图形
shap.save_html('./result/SHAP_force_plot_sample_0_sepsis.html', html)


# explainer = shap.KernelExplainer(svm_model.predict_proba, X_test_final)
# shap_values = explainer.shap_values(X_test_final)
# # 绘制SHAP特征重要性
# shap.summary_plot(shap_values, X_test_final, show=False, max_display=7)
# plt.title('SHAP Summary Plot (Bar) for SVM')
# shap_summary_svm_pdf = "./result/SHAP_Beeswarm_SVM_21.pdf"
# plt.savefig(shap_summary_svm_pdf, bbox_inches='tight')
# plt.close()

y_probs_model_x = lr_model.predict_proba(X_test_final)[:, 1]

# ROC curve and AUC for model_x
fpr_x, tpr_x, _ = roc_curve(y_test_final, y_probs_model_x)
roc_auc_x = auc(fpr_x, tpr_x)
roc_auc_rf = roc_auc_score(y_test_final, y_probs_model_x)

X_sofa_train = X_train_final["SOFA"].values.reshape(-1, 1)
X_sofa_test = X_test_final["SOFA"].values.reshape(-1, 1)
model_lr_sofa = LogisticRegression()
model_lr_sofa.fit(X_sofa_train, y_train_final)
y_probs_sofa = model_lr_sofa.predict(X_sofa_test)
fpr_sofa, tpr_sofa, _ = roc_curve(y_test_final, y_probs_sofa)
roc_auc_sofa = auc(fpr_sofa, tpr_sofa)

# DeLong检验（你可以使用scikit-posthocs或自定义实现）
DelongTest(y_probs_model_x,y_probs_sofa,y_test_final)


# Plot the ROC curves for both models
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 6))
plt.plot(fpr_x, tpr_x, color='blue', lw=2, label=f'RF AUC = {roc_auc_x:.3f}')
plt.plot(fpr_sofa, tpr_sofa, color='red', lw=2, label=f'SOFA Logistic Regression AUC = {roc_auc_sofa:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: RF vs SOFA Logistic Regression')
plt.legend(loc='lower right')
shap_summary_rf_dot_pdf = "./result/roc_vs_sofa_sepsis.pdf"
plt.savefig(shap_summary_rf_dot_pdf, bbox_inches='tight')
plt.show()