# IMBK_Bank_Customer_Churn_ML


<br>


## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | 고객 이탈 분류 ML 및 인사이트 분석 |
| **기간** | 2026년 4월 9일 |
| **목표** | 은행 고객 데이터를 기반으로 이탈 여부(churn)를 예측하고, SHAP을 통해 이탈 원인 인사이트 도출 |
| **평가지표** | F1-Score |


<br>


## 🛠 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 데이터 처리 | `pandas` `numpy` |
| 전처리 | `scikit-learn` (LabelEncoder, StandardScaler, train_test_split) |
| AutoML | `PyCaret` |
| 하이퍼파라미터 튜닝 | `Optuna` |
| 모델링 | `scikit-learn` (RandomForest, AdaBoost, GradientBoosting, LogisticRegression, StackingClassifier), `LightGBM` |
| 사후 분석 | `SHAP` |
| 시각화 | `matplotlib` |


<br>


## 📂 데이터 출처

- **출처**: [Kaggle — Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)
- **규모**: 10,000 rows × 12 columns
- **타겟**: `churn` (이탈 여부, 0/1 이진 분류)

| 컬럼명 | 설명 |
|--------|------|
| customer_id | 고객 식별자 (학습에서 제거) |
| country | 국가 (범주형) |
| gender | 성별 (범주형) |
| age | 나이 |
| tenure | 거래 기간 |
| balance | 계좌 잔액 |
| products_number | 보유 상품 수 |
| credit_card | 신용카드 보유 여부 |
| active_member | 활성 회원 여부 |
| estimated_salary | 추정 연봉 |
| credit_score | 신용 점수 |
| churn | **이탈 여부 (Target)** |


<br>


## ⚙️ 데이터 전처리

**1. 불필요한 컬럼 제거**
- `customer_id`는 단순 고객 식별자로 이탈 여부와 인과관계가 없는 변수이므로 제거
- 학습에 포함 시 모델이 의미 없는 패턴을 학습할 위험 존재

**2. 범주형 변수 인코딩**
- `country`, `gender`는 문자열 범주형 변수 → 머신러닝 모델은 수치형 입력만 처리 가능
- `LabelEncoder`를 적용해 수치형으로 변환

**3. Train / Validation 분리 및 StandardScaler 적용**
- 8:2 비율 분리 시 `stratify=y` 적용 → 클래스 불균형으로 인한 평가 왜곡 방지
- `fit_transform`은 학습셋에만, `transform`은 검증셋에 적용 → **데이터 누수(Data Leakage) 차단**

```python
scaler = StandardScaler()
train_df[X] = scaler.fit_transform(train_df[X])   # train 기준으로 기준 저장
valid_df[X] = scaler.transform(valid_df[X])        # 동일 기준 적용
```


<br>


## 📊 EDA 및 해석

**성별(gender)에 따른 이탈률(churn) 분석**

<img width="539" height="387" alt="image" src="https://github.com/user-attachments/assets/8fa7bd19-f7c8-40a6-b855-009c4683e1d7" />



- 여성(0) 이탈률 **약 25.1%** vs 남성(1) 이탈률 **약 16.5%** → 여성이 약 **8.6%p 더 높음**
- 성별이 고객 이탈 여부를 예측하는 데 **유의미한 변수**임을 시사
- 여성 고객이 이탈에 더 민감하게 반응하는 원인으로는 금융 상품 니즈 차이, 서비스 만족도 차이, 경쟁 은행으로의 이동 가능성 등을 고려할 수 있음
- 따라서 모델 학습 시 `gender` 변수를 중요한 피처로 포함하는 것이 타당


<br>


## 🤖 AutoML → Hyperparameter Tuning → Stacking Pipeline

### 1️⃣ AutoML — PyCaret으로 모델 비교

- `PyCaret`의 `compare_models(sort="F1")`으로 14개 모델을 F1 기준으로 한 번에 비교
- **F1 Score 상위 4개 모델 선정**: Ada Boost, Gradient Boosting, LightGBM, Random Forest

> 이탈 예측은 클래스 불균형이 존재하는 이진 분류 문제이므로 Accuracy보다 Precision과 Recall을 균형 있게 반영하는 **F1이 더 적합한 기준**

```python
clf = setup(data=train_df, target="churn", session_id=42)
best_model = compare_models(sort="F1")
```


### 2️⃣ Hyperparameter Tuning — Optuna

각 모델에 대해 `Optuna`로 F1-Score를 최대화하는 하이퍼파라미터를 탐색 (n_trials=10)

```python
def objective_ada(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0)
    }
    model = AdaBoostClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    return f1_score(y_valid, model.predict(X_valid))

study_ada = optuna.create_study(direction='maximize')
study_ada.optimize(objective_ada, n_trials=10)
```

| 모델 | 튜닝 방법 | 탐색 파라미터 예시 |
|------|----------|--------------------|
| AdaBoost | Optuna | n_estimators, learning_rate |
| GradientBoosting | Optuna | n_estimators, learning_rate, max_depth |
| LightGBM | Optuna | num_leaves, learning_rate, n_estimators |
| RandomForest | Optuna | n_estimators, max_depth, min_samples_split |


### 3️⃣ Stacking Pipeline

튜닝된 4개 모델을 **전방 모델(Base Learner)**로, `LogisticRegression`을 **후방 모델(Meta Learner)**로 사용

```python
stack = StackingClassifier(
    estimators=[('ada', ada), ('gbc', gbc), ('lgbm', lgbm), ('rf', rf)],
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
```

---
**최종 Validation 성능**

| 지표 | 개별 최고 모델 (AdaBoost) | Stacking 앙상블 |
|------|--------------------------|-----------------|
| F1-Score | 0.5748 | **0.5918** ⬆️ |
| Accuracy | — | **0.8655** |

> 개별 모델의 예측 결과를 결합함으로써 단일 모델의 한계를 보완하는 앙상블의 효과가 유효하게 작용
---

<br>


## 🔍 SHAP Value 사후 분석

> Random Forest 모델 기준 SHAP Summary Plot 분석

```python
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_valid)
shap.summary_plot(shap_values, X_valid)
```
<img width="708" height="499" alt="image" src="https://github.com/user-attachments/assets/a02995eb-6499-400d-ab13-167cd3be9a2d" />


**고객 이탈에 영향을 미치는 상위 변수**

| 순위 | 변수 | 해석 |
|------|------|------|
| 1 | `age` | 고령일수록 이탈 가능성 높음. SHAP 분포 범위가 가장 넓어 예측에 가장 강한 영향력 |
| 2 | `products_number` | 보유 상품 수가 너무 적거나 너무 많은 경우 모두 이탈 방향으로 작용하는 비선형 패턴 |
| 3 | `active_member` | 비활성 고객(파란색)이 양의 SHAP 값 → 활동하지 않는 고객일수록 이탈 가능성 높음 |


<br>


## 💡 인사이트 제안

SHAP 분석 결과, **고령 + 낮은 상품 보유 + 비활성** 고객이 이탈 위험군의 핵심 프로파일로 나타났다.

**제안 1 — 고령 고객 대상 리텐션 프로그램**
- 나이가 많을수록 이탈 가능성이 높으므로, 50대 이상 고객을 대상으로 전담 상담사 배정 또는 맞춤형 금융 상품 제안 프로그램 운영

**제안 2 — 적정 상품 크로스셀링 전략**
- 보유 상품이 1개 이하인 고객에게는 적극적인 추가 상품 가입 유도
- 단, 3개 이상 보유 고객에서도 이탈이 발생하므로 상품 수 증가보다 **상품 만족도 관리**가 중요

**제안 3 — 비활성 고객 Early Warning 시스템**
- 일정 기간 거래가 없는 비활성 고객을 자동 탐지하여 이탈 전 선제적 리텐션 마케팅(쿠폰, 혜택 알림 등) 적용


<br>


## 📚 Reference

- Dataset: [Kaggle — Bank Customer Churn Dataset by Gaurav Topre](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)
- [PyCaret Documentation](https://pycaret.org/)
- [Optuna Documentation](https://optuna.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Scikit-learn StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)

