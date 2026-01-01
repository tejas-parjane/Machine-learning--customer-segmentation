# Machine Learning-Driven Customer Segmentation for Personalized Marketing Strategy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

## 📋 Project Overview

This repository contains a comprehensive **machine learning-driven customer segmentation framework** for e-commerce businesses, designed to enable personalized marketing strategies through data-driven insights. The project demonstrates how unsupervised learning techniques can identify distinct customer groups and translate analytical findings into actionable business strategies.

### Key Features
- ✅ **Advanced Customer Segmentation** using K-Means clustering
- ✅ **18 Engineered Features** capturing demographics, behavior, and engagement
- ✅ **Statistical Validation** of education's role as behavioral modifier
- ✅ **Personalized Marketing Strategies** for each identified segment
- ✅ **A/B Testing Framework** for strategy evaluation
- ✅ **Production-Ready Deployment** module with real-time prediction capability

---

##  Research Objectives

1. Develop machine learning models for customer segmentation based on behavioral patterns
2. Analyze the impact of educational attainment on customer behavior and marketing responsiveness
3. Design personalized marketing strategies tailored to each customer segment
4. Implement A/B testing framework to evaluate strategy effectiveness
5. Create production-ready deployment system for real-time segmentation

---

## 📊 Dataset

**Source:** [Kaggle - Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

**Dataset Characteristics:**
- **Records:** 2,240 customers
- **Features:** 29 original variables
- **Time Period:** 2012-2014
- **Data Types:** Demographics, transactional data, behavioral metrics, campaign responses

**Key Variables:**
- Demographics: Age, Education, Marital Status, Income, Family Composition
- Spending: 6 product categories (Wines, Fruits, Meat, Fish, Sweets, Gold)
- Channels: Web, Catalog, Store, Deals purchases
- Engagement: Campaign acceptance rates, website visits

---

## 🔬 Methodology

### 1. Data Preprocessing
- Missing value imputation (education-stratified median for income)
- Outlier detection and retention of legitimate high-value customers
- Feature scaling using StandardScaler

### 2. Feature Engineering
Created 18 engineered features including:
- **Demographic:** Age, Customer_Tenure, Family_Size
- **Behavioral:** Total_Spend, Spend_per_Purchase, Digital_Engagement
- **Engagement:** Total_Campaign_Accepted, Campaign_Response_Rate
- **Channel:** Store_vs_Web ratio, Total_Purchases

### 3. Clustering Model Development
- **Algorithm:** K-Means clustering
- **Optimal Clusters:** k=4 (determined via multi-metric evaluation)
- **Evaluation Metrics:**
  - Silhouette Score: 0.1497
  - Calinski-Harabasz Index: 1,502.34
  - Davies-Bouldin Index: 1.38

### 4. Statistical Validation
- Kruskal-Wallis H-tests for education's impact on spending and campaign response
- Mann-Whitney U post-hoc pairwise comparisons
- Spearman correlation analysis for effect sizes

---

##  Identified Customer Segments

### Cluster 0: Budget-Conscious Educated (30.7%)
- **Income:** $42,920 | **Spend:** $125
- **Strategy:** Value-focused email campaigns, 15-20% discounts

### Cluster 1: Digital Mid-Value (35.0%)
- **Income:** $59,878 | **Spend:** $833
- **Strategy:** Omnichannel engagement, cross-sell bundles, 10-15% discounts

### Cluster 2: Premium High-Value (20.0%)
- **Income:** $78,185 | **Spend:** $1,388
- **Strategy:** VIP experiences, exclusive access, premium catalogs, 5-10% discounts

### Cluster 3: Traditional Price-Sensitive (14.4%)
- **Income:** $29,877 | **Spend:** $111
- **Strategy:** In-store promotions, SMS alerts, 20-25% discounts

---

##  Key Findings

### Statistical Results
- ✅ **Education significantly impacts spending** (H=71.02, p<0.001)
- ✅ **Education affects campaign response** (H=16.41, p=0.0025)
- ✅ **Clusters differ dramatically** in value (12-fold spending difference)
- ✅ **Education correlates more with income (ρ=0.41)** than spending (ρ=0.28)

### Business Insights
- **Premium segment (Cluster 2) generates 583% ROI** despite 20% population share
- **Wine category dominates** spending (35-59% across segments)
- **Omnichannel behavior** varies significantly by segment
- **15% response rate lift** projected from personalized strategies

---

##  Technical Stack

### Core Libraries
```python
pandas==2.0.3          # Data manipulation
numpy==1.24.3          # Numerical computing
scikit-learn==1.3.0    # Machine learning
scipy==1.11.1          # Statistical testing
```

### Visualization
```python
matplotlib==3.7.2      # Plotting
seaborn==0.12.2        # Statistical graphics
```

### Data Acquisition
```python
kagglehub==0.1.0       # Kaggle dataset download
```

### Model Deployment
```python
joblib==1.3.1          # Model serialization
```

---

##  Repository Structure

```
├── notebooks/
│   └── Machine_learning_driven_customer_segmentation.ipynb
├── data/
│   └── marketing_campaign.csv (downloaded via Kaggle API)
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── docs/
│   ├── Complete_Project_Report.pdf
│   ├── Literature_Review.pdf
│   └── Methodology.pdf
├── results/
│   ├── cluster_profiles.csv
│   ├── customer_segments_with_strategies.csv
│   └── ab_test_results.csv
├── requirements.txt
├── README.md
└── LICENSE
```

---

##  Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Kaggle API credentials

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/customer-segmentation-ml.git
cd customer-segmentation-ml
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup Kaggle API**
```bash
# Place your kaggle.json in ~/.kaggle/ directory
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

5. **Run the Jupyter Notebook**
```bash
jupyter notebook notebooks/Machine_learning_driven_customer_segmentation.ipynb
```

---

##  Usage Examples

### Load Pre-trained Model
```python
import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Predict segment for new customer
new_customer = pd.DataFrame({...})  # Your customer data
X_scaled = scaler.transform(new_customer)
cluster = model.predict(X_scaled)
print(f"Customer belongs to Cluster: {cluster[0]}")
```

### Generate Strategy Recommendations
```python
from strategy_engine import MarketingStrategyEngine

engine = MarketingStrategyEngine()
strategy = engine.get_strategy(cluster_id=2)
print(f"Recommended channels: {strategy['channels']}")
print(f"Offer type: {strategy['offer_type']}")
```

---


##  Academic Context

This project was completed as part of the **Master of Computer Applications (MCA)** program at **Amity University Online**, representing a comprehensive application of machine learning to real-world business problems.

### Research Contributions
1. Empirical evidence of education's role as behavioral modifier in customer segmentation
2. Production-ready ML segmentation framework with deployment architecture
3. Integration of statistical validation with business strategy development
4. Comprehensive A/B testing methodology for personalization evaluation

---

##  Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | 0.1497 | Moderate separation (typical for customer data) |
| Calinski-Harabasz | 1,502.34 | Strong between/within cluster variance ratio |
| Davies-Bouldin | 1.38 | Good cluster separation (< 2.0 threshold) |
| Training Time | 0.42s | Highly efficient for production use |
| Prediction Latency | <1s | Suitable for real-time applications |

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Real-time dashboard for segment monitoring
- [ ] Automated retraining pipeline with drift detection
- [ ] Integration with marketing automation platforms
- [ ] Deep learning-based segmentation (autoencoders)
- [ ] Recommendation system for product personalization
- [ ] Customer lifetime value prediction models

### Research Extensions
- [ ] Longitudinal analysis tracking segment migration
- [ ] Multi-market segmentation comparison
- [ ] Integration of social media behavioral data
- [ ] Causal inference methods for strategy validation

---

##  Acknowledgments

- **Kaggle** for providing the Customer Personality Analysis dataset
- **Akash Patel** for compiling and publishing the dataset
- **Scikit-learn community** for excellent machine learning tools
- **Amity University Online** for academic support and guidance
- **Research community** for foundational work in customer segmentation

---

*Made with ❤️ for the data science and marketing analytics community*
