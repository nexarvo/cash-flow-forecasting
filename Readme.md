# 💸 Cash Flow Forecasting & Transaction Categorization using ML

This project demonstrates how machine learning can be used to:

1. Forecast future cash flow using historical transaction data
2. Automatically categorize raw bank transactions using natural language processing (NLP)

It’s designed to reflect a real-world fintech scenario — helping businesses or individuals understand their financial behavior, optimize spending, and gain insights from noisy bank data.

---

## 🚀 Features

### 🔮 Cash Flow Forecasting

- Daily net cash flow aggregated from raw transaction logs
- Forecasting using **Facebook Prophet** to handle trends and seasonality
- Time-series cross-validation and error metrics (MAE, RMSE)

### 🧠 Transaction Categorization

- Cleaned, tokenized transaction descriptions
- Fine-tuned **DistilBERT** model to classify transactions into categories like:
  - Groceries
  - Restaurant
  - Entertainment
  - Health
  - Electronics
  - Clothing

---

## 📁 Dataset

Used a public dataset from Kaggle:  
**[Bank Transaction Fraud Detection](https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection)**

This dataset includes:

- Transaction descriptions
- Transaction types and amounts
- Account balance and device metadata
- Merchant categories (used as labels)

---

## 🧱 Tech Stack

| Task                          | Library/Tool                                 |
| ----------------------------- | -------------------------------------------- |
| Forecasting                   | `Facebook Prophet`                           |
| NLP Classification            | `Transformers`, `DistilBERT`, `scikit-learn` |
| Data Manipulation             | `pandas`, `numpy`                            |
| Model Evaluation              | `sklearn`, `matplotlib`, `seaborn`           |
| Environment & Version Control | `Python`, `Git`, `Jupyter` or `Colab`        |

---

## 🧪 How to Run

1. **Clone the repo**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/cash-flow-forecasting.git
   cd cash-flow-forecasting
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run forecasting notebook**:

   - cash_flow_forecasting.ipynb

4. **Run transaction classifier training**:
   - transaction_classifier_distilbert.ipynb

## 📈 Results

- Forecasting MAE (1-6 days): 3M–11M range
- Transaction classification accuracy: ~85% with DistilBERT
