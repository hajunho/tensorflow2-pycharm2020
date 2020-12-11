import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def main():
    # https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    # german_credit_data.csv를 이용해 Risk를 예측하는 모델을 만든후
    # finance_test_feature.csv의 Risk를 예측해보세요.
    df_origin = pd.read_csv("german.data.csv")
    df_clean = df_origin

    df_clean['Saving accounts'] = df_origin['Saving accounts'].fillna('Others')
    df_clean['Checking account'] = df_origin['Checking account'].fillna('Others')

    df_clean = df_clean.replace(['good', 'bad'], [0, 1])
    cat_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    num_features = ['Age', 'Job', 'Credit amount', 'Duration', 'Risk']

    for variable in cat_features:
        dummies = pd.get_dummies(df_clean[cat_features])
        df1 = pd.concat([df_clean[num_features], dummies], axis=1)

    x = df1.drop(columns=['Risk']).to_numpy()
    y = df_clean['Risk']
    y = y.to_numpy().ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    models = []
    # models.append(('LR', LogisticRegression(max_iter =5000)))
    models.append(('LDA', LinearDiscriminantAnalysis()))  # LDA 모델
    models.append(('KNN', KNeighborsClassifier()))  # KNN 모델
    models.append(('CART', DecisionTreeClassifier()))  # 의사결정트리 모델
    models.append(('NB', GaussianNB()))  # 가우시안 나이브 베이즈 모델
    models.append(('RF', RandomForestClassifier()))  # 랜덤포레스트 모델
    models.append(('SVM', SVC(gamma='auto')))  # SVM 모델
    models.append(('XGB', XGBClassifier()))  # XGB 모델

    for name, model in models:
        model.fit(x_train, y_train)
        msg = "%s - train_score : %f, test score : %f" % (
        name, model.score(x_train, y_train), model.score(x_test, y_test))
        print(msg)

    parameters = {'criterion': ['gini', 'entropy'],
                  'max_depth': [3, 4, 5, 6],
                  'max_features': ['auto', 'sqrt', 'log2']}
    cv = KFold(n_splits=5)
    DT = DecisionTreeClassifier()
    DT_CV = GridSearchCV(DT, parameters, scoring='accuracy', cv=cv, n_jobs=-1)
    DT_CV.fit(x_train, y_train)
    print(DT_CV.score(x_train, y_train))
    print(DT_CV.score(x_test, y_test))
    best_DT_CV = DT_CV.best_estimator_
    print(best_DT_CV.score(x_test, y_test))

    model_predition = models[-1][1].predict(x_test)
    cm = confusion_matrix(y_test, model_predition)

    plt.rcParams['figure.figsize'] = (5, 5)
    sns.set(style='dark', font_scale=1.4)
    ax = sns.heatmap(cm, annot=True)
    plt.xlabel('Real Data')
    plt.ylabel('Prediction')
    plt.show()
    cm

    print("Recall score: {}".format(recall_score(y_test, model_predition)))
    print("Precision score: {}".format(precision_score(y_test, model_predition)))

    dataframe = pd.read_csv("german.data2.csv")

    for i in range(10):
        prediction = models[-1][1].predict(x_test[i].reshape(1, -1))
        print("{} 번째 테스트 데이터의 예측 결과: {}, 실제 데이터: {}".format(i, prediction[0], y_test[i]))

    # print(df1)

    # finance_test_feature.csv의 순서대로 [index, Risk]를 가지는 dataframe을
    # return 하도록 합니다.


if __name__ == "__main__":
    main()