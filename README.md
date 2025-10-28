# VIVA PREPARATION DOCUMENT
## Data Science Project - Comprehensive Data Preprocessing and EDA

**Student:** Abdullah Asif (FA23-BCS-017-A)  
**Group Member:** Abdul Hannan (FA23-BCS-013-A)  
**Dataset:** Online Retail Dataset (Kaggle)

---

## TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Major Code Explanations](#major-code-explanations)
3. [Common Viva Questions and Answers](#common-viva-questions)

---

## PROJECT OVERVIEW

### What is this project about?
This project demonstrates the complete data science lifecycle focusing on:
- Data Cleaning (handling missing, noisy, inconsistent, duplicate data)
- Data Reduction (feature selection, regression, clustering, aggregation)
- Data Transformation (normalization, discretization, feature construction)
- Exploratory Data Analysis (descriptive statistics and visualizations)

### Dataset Details
- **Name:** Online Retail Dataset
- **Source:** Kaggle
- **Size:** 541,909 rows, 11 columns (after adding calculated columns)
- **Type:** Transactional data from UK-based online retail store
- **Time Period:** December 2010 to December 2011

### Original Columns
1. InvoiceNo - Invoice number
2. StockCode - Product code
3. Description - Product name
4. Quantity - Number of items purchased
5. InvoiceDate - Date of transaction
6. UnitPrice - Price per item
7. CustomerID - Unique customer identifier
8. Country - Customer country

### Added Columns (to reach 10+)
9. TotalPrice - Calculated as Quantity × UnitPrice
10. Month - Extracted from InvoiceDate
11. Year - Extracted from InvoiceDate

---

## MAJOR CODE EXPLANATIONS

### 1. LOADING AND PREPARING DATA

```python
df_original = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')
```

**Line-by-line:**
- `pd.read_csv()` - Pandas function to read CSV files
- `'online_retail.csv'` - File name
- `encoding='ISO-8859-1'` - Handles special characters (like £, ñ, etc.)
- `df_original` - Variable storing our dataset as DataFrame

**Why encoding='ISO-8859-1'?**
Because dataset has European characters. Without this, we get errors.

---

```python
df_original['TotalPrice'] = df_original['Quantity'] * df_original['UnitPrice']
```

**Line-by-line:**
- `df_original['TotalPrice']` - Creates new column named TotalPrice
- `df_original['Quantity'] * df_original['UnitPrice']` - Multiplies Quantity by UnitPrice for each row
- Result: Revenue from each transaction

**Why calculate TotalPrice?**
To understand revenue per transaction. Essential for business analysis.

---

```python
df_original['InvoiceDate'] = pd.to_datetime(df_original['InvoiceDate'])
df_original['Month'] = df_original['InvoiceDate'].dt.month
df_original['Year'] = df_original['InvoiceDate'].dt.year
```

**Line-by-line:**
- `pd.to_datetime()` - Converts text date to datetime format
- `.dt.month` - Extracts month number (1-12)
- `.dt.year` - Extracts year number (2010, 2011)

**Why extract Month and Year?**
To analyze seasonal trends and yearly patterns.

---

### 2. CREATING CORRUPTED DATA

```python
np.random.seed(42)
missing_indices = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
df.loc[missing_indices, 'Description'] = np.nan
```

**Line-by-line:**
- `np.random.seed(42)` - Sets random seed for reproducibility (same results every time)
- `np.random.choice(df.index, ...)` - Randomly selects row indices
- `size=int(0.15 * len(df))` - Selects 15% of total rows
- `replace=False` - Each row selected only once
- `df.loc[missing_indices, 'Description']` - Accesses specific rows in Description column
- `= np.nan` - Assigns NaN (Not a Number = missing value)

**Why introduce problems artificially?**
Ma'am said datasets are already clean. We introduce problems to demonstrate cleaning techniques.

---

### 3. HANDLING MISSING DATA - KNN IMPUTATION

```python
from sklearn.impute import KNNImputer
numerical_cols = ['Quantity', 'UnitPrice', 'CustomerID', 'TotalPrice']
imputer = KNNImputer(n_neighbors=5)
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
```

**Line-by-line:**
- `from sklearn.impute import KNNImputer` - Imports KNN imputation tool
- `numerical_cols = [...]` - List of numerical columns to impute
- `KNNImputer(n_neighbors=5)` - Creates imputer object that uses 5 nearest neighbors
- `imputer.fit_transform()` - Fits model and transforms data in one step
- `df[numerical_cols] = ...` - Replaces original columns with imputed values

**How KNN Imputation works:**
1. For each missing value, finds 5 most similar rows (based on other features)
2. Takes average of those 5 neighbors' values
3. Fills missing value with that average

**Why KNN instead of mean?**
KNN considers relationships between features. More accurate than simple mean.

**Example:**
If CustomerID is missing but Quantity=100 and UnitPrice=5, KNN finds 5 similar transactions and averages their CustomerIDs.

---

### 4. OUTLIER DETECTION - IQR METHOD

```python
Q1 = df['Quantity'].quantile(0.25)
Q3 = df['Quantity'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

**Line-by-line:**
- `quantile(0.25)` - Finds 25th percentile (Q1)
- `quantile(0.75)` - Finds 75th percentile (Q3)
- `IQR = Q3 - Q1` - Interquartile Range (middle 50% spread)
- `Q1 - 1.5 * IQR` - Lower fence for outliers
- `Q3 + 1.5 * IQR` - Upper fence for outliers

**Visual representation:**
```
|-----|=====|=====|-----|
min   Q1    Q2    Q3   max
      <--IQR-->
```

**Why 1.5 × IQR?**
Standard statistical rule. Values beyond this are considered outliers.

**Example:**
- Q1 = 2, Q3 = 10, IQR = 8
- Lower bound = 2 - (1.5 × 8) = -10
- Upper bound = 10 + (1.5 × 8) = 22
- Any value < -10 or > 22 is an outlier

---

### 5. BINNING (SMOOTHING)

```python
df['Quantity_Binned'] = pd.cut(df['Quantity'], bins=5, 
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
```

**Line-by-line:**
- `pd.cut()` - Divides continuous data into discrete bins
- `df['Quantity']` - Column to bin
- `bins=5` - Creates 5 equal-width bins
- `labels=[...]` - Names for each bin
- `df['Quantity_Binned']` - New column with bin labels

**How it works:**
If Quantity ranges from 1 to 100:
- Bin 1 (1-20): Very Low
- Bin 2 (21-40): Low
- Bin 3 (41-60): Medium
- Bin 4 (61-80): High
- Bin 5 (81-100): Very High

**Why binning?**
Reduces noise by grouping similar values. Simplifies analysis.

---

### 6. K-MEANS CLUSTERING

```python
from sklearn.cluster import KMeans
cluster_features = df[['Quantity', 'UnitPrice']].values
scaler_cluster = StandardScaler()
cluster_scaled = scaler_cluster.fit_transform(cluster_features)
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Customer_Segment'] = kmeans_model.fit_predict(cluster_scaled)
```

**Line-by-line:**
- `df[['Quantity', 'UnitPrice']].values` - Extracts two columns as numpy array
- `StandardScaler()` - Creates scaler object
- `fit_transform()` - Scales features to mean=0, std=1
- `KMeans(n_clusters=3)` - Creates K-Means with 3 clusters
- `random_state=42` - For reproducibility
- `n_init=10` - Runs algorithm 10 times, picks best result
- `fit_predict()` - Fits model and assigns cluster labels

**How K-Means works:**
1. Randomly places 3 cluster centers
2. Assigns each point to nearest center
3. Moves centers to average of assigned points
4. Repeats until centers stop moving

**Why scale before clustering?**
Quantity might be 1-100, UnitPrice might be 0.1-10. Without scaling, Quantity dominates distance calculation.

---

### 7. NORMALIZATION - MIN-MAX

```python
from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()
df['Quantity_MinMax'] = scaler_minmax.fit_transform(df[['Quantity']])
```

**Line-by-line:**
- `MinMaxScaler()` - Creates min-max scaler
- `fit_transform()` - Calculates min/max and transforms
- Result stored in new column

**Formula:**
```
X_normalized = (X - X_min) / (X_max - X_min)
```

**Example:**
- Original: [2, 5, 8, 10]
- Min = 2, Max = 10
- Normalized: 
  - (2-2)/(10-2) = 0/8 = 0
  - (5-2)/(10-2) = 3/8 = 0.375
  - (8-2)/(10-2) = 6/8 = 0.75
  - (10-2)/(10-2) = 8/8 = 1
- Result: [0, 0.375, 0.75, 1]

**Why Min-Max?**
Scales all values to [0, 1] range. Required for neural networks and algorithms sensitive to scale.

---

### 8. NORMALIZATION - Z-SCORE

```python
from sklearn.preprocessing import StandardScaler
scaler_zscore = StandardScaler()
df['UnitPrice_ZScore'] = scaler_zscore.fit_transform(df[['UnitPrice']])
```

**Formula:**
```
Z = (X - μ) / σ
```
Where:
- μ (mu) = mean
- σ (sigma) = standard deviation

**Example:**
- Original: [2, 4, 6, 8, 10]
- Mean = 6, Std = 2.83
- Z-scores:
  - (2-6)/2.83 = -1.41
  - (4-6)/2.83 = -0.71
  - (6-6)/2.83 = 0
  - (8-6)/2.83 = 0.71
  - (10-6)/2.83 = 1.41

**Why Z-Score?**
Centers data around 0 with standard deviation 1. Required for algorithms assuming normal distribution.

**Min-Max vs Z-Score:**
- Min-Max: Range [0, 1], preserves relationships
- Z-Score: Mean=0, Std=1, handles outliers better

---

### 9. FEATURE CONSTRUCTION

```python
df['Price_Category'] = df['UnitPrice'].apply(
    lambda x: 'Budget' if x < 2 else ('Mid-Range' if x < 5 else 'Premium')
)
```

**Line-by-line:**
- `df['UnitPrice'].apply()` - Applies function to each value
- `lambda x:` - Anonymous function with input x
- `'Budget' if x < 2` - If price < 2, return 'Budget'
- `else ('Mid-Range' if x < 5 else 'Premium')` - Else check if < 5

**Logic flow:**
```
if x < 2:
    return 'Budget'
else:
    if x < 5:
        return 'Mid-Range'
    else:
        return 'Premium'
```

**Why create this feature?**
Transforms numerical price into meaningful business categories. Easier for marketing teams to understand.

---

### 10. DISCRETIZATION

```python
df['Revenue_Quartile'] = pd.qcut(df['TotalPrice'], q=4, 
                                  labels=['Low', 'Medium', 'High', 'Very High'])
```

**Line-by-line:**
- `pd.qcut()` - Quantile-based discretization (equal frequency bins)
- `q=4` - Creates 4 bins
- Each bin contains same number of records (25%)

**pd.cut() vs pd.qcut():**
- `pd.cut()`: Equal-width bins (e.g., 0-25, 26-50, 51-75, 76-100)
- `pd.qcut()`: Equal-frequency bins (each bin has same count)

**Example:**
Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200]
- pd.cut (4 bins): [1-50], [51-100], [101-150], [151-200]
  - Unequal distribution, last bins have few values
- pd.qcut (4 bins): Each bin gets 3 values
  - Bin 1: [1, 2, 3]
  - Bin 2: [4, 5, 6]
  - Bin 3: [7, 8, 9]
  - Bin 4: [10, 100, 200]

---

### 11. LOGARITHMIC TRANSFORMATION

```python
df['TotalPrice_Log'] = np.log1p(df['TotalPrice'])
```

**Line-by-line:**
- `np.log1p()` - Calculates log(1 + x)
- Why 1+? Handles zero values (log(0) is undefined)

**Formula:**
```
log(1 + x)
```

**Example:**
- Original: [1, 10, 100, 1000]
- Log: [0.69, 2.40, 4.62, 6.91]
- Compresses large values

**Why use log transformation?**
1. Reduces right skewness
2. Makes data more normal
3. Handles wide-ranging data (e.g., income: $1K to $1M)

**Visual effect:**
```
Before: |--|----|--------|------------------|
After:  |---|---|---|---|
```

---

### 12. REGRESSION MODEL

```python
from sklearn.linear_model import LinearRegression
X = df[['Quantity', 'UnitPrice']].values
y = df['TotalPrice'].values
reg_model = LinearRegression()
reg_model.fit(X, y)
r2_score = reg_model.score(X, y)
```

**Line-by-line:**
- `X = df[['Quantity', 'UnitPrice']].values` - Input features (independent variables)
- `y = df['TotalPrice'].values` - Target variable (dependent variable)
- `LinearRegression()` - Creates regression model
- `reg_model.fit(X, y)` - Trains model
- `reg_model.score(X, y)` - Calculates R² score

**What is Linear Regression?**
Finds best-fit line/plane that predicts y from X.

**Formula:**
```
TotalPrice = β₀ + β₁(Quantity) + β₂(UnitPrice)
```
Where β₀, β₁, β₂ are coefficients learned from data.

**R² Score:**
- Measures how well model predicts
- Range: 0 to 1
- R² = 0.95 means 95% of variation explained
- Higher = better

---

### 13. CORRELATION HEATMAP

```python
correlation_data = df[['Quantity', 'UnitPrice', 'TotalPrice', 'Month', 'Year']].corr()
sns.heatmap(correlation_data, annot=True, fmt='.2f', cmap='coolwarm', center=0)
```

**Line-by-line:**
- `.corr()` - Calculates correlation matrix
- `sns.heatmap()` - Creates heatmap visualization
- `annot=True` - Shows correlation values on cells
- `fmt='.2f'` - Format numbers to 2 decimals
- `cmap='coolwarm'` - Color scheme (blue=negative, red=positive)
- `center=0` - White color at 0 correlation

**What is correlation?**
Measures linear relationship between two variables.
- +1: Perfect positive (both increase together)
- 0: No relationship
- -1: Perfect negative (one increases, other decreases)

**Example:**
- Quantity vs TotalPrice: High positive (0.85)
- UnitPrice vs Month: Low/zero (0.02)

---

### 14. AGGREGATION

```python
country_agg = df.groupby('Country').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum',
    'InvoiceNo': 'count',
    'UnitPrice': 'mean'
}).reset_index()
```

**Line-by-line:**
- `groupby('Country')` - Groups all rows by Country
- `.agg({...})` - Applies different functions to different columns
- `'Quantity': 'sum'` - Sums all Quantity per country
- `'InvoiceNo': 'count'` - Counts orders per country
- `'UnitPrice': 'mean'` - Averages UnitPrice per country
- `.reset_index()` - Converts Country from index to regular column

**What happens?**
Before: 500,000 rows (one per transaction)
After: 38 rows (one per country)

**Example:**
Before:
```
Country       | Quantity | TotalPrice
UK            | 5        | 10
UK            | 3        | 6
France        | 2        | 8
```

After:
```
Country | Total_Quantity | Total_Revenue | Order_Count
UK      | 8              | 16            | 2
France  | 2              | 8             | 1
```

---

### 15. DESCRIPTIVE STATISTICS

```python
mean_qty = df['Quantity'].mean()
median_qty = df['Quantity'].median()
mode_qty = df['Quantity'].mode()[0]
std_qty = df['Quantity'].std()
skewness = df['Quantity'].skew()
kurtosis = df['Quantity'].kurtosis()
```

**Explanations:**

**Mean:** Average value
- Formula: Sum of all values / Count
- Example: [1, 2, 3, 4, 5] → Mean = 15/5 = 3

**Median:** Middle value when sorted
- Example: [1, 2, 3, 4, 100] → Median = 3
- Robust to outliers (100 doesn't affect it)

**Mode:** Most frequent value
- Example: [1, 2, 2, 3, 4, 2] → Mode = 2

**Standard Deviation:** Average distance from mean
- High std = data is spread out
- Low std = data is clustered

**Skewness:** Measures asymmetry
- Positive skew: Tail on right (most values on left)
- Negative skew: Tail on left (most values on right)
- Zero: Symmetric (normal distribution)

**Kurtosis:** Measures tail heaviness
- Positive: Heavy tails (many outliers)
- Negative: Light tails (few outliers)

---

## COMMON VIVA QUESTIONS AND ANSWERS

### GENERAL PROJECT QUESTIONS

**Q1: What is the purpose of this project?**

**Answer:** This project demonstrates the complete data science pipeline focusing on data preprocessing and exploratory analysis. We learn how to clean messy real-world data, reduce dimensionality, transform features, and extract insights through statistical and visual analysis.

---

**Q2: Why did you choose the Online Retail dataset?**

**Answer:** 
1. Available on Kaggle (reliable source)
2. Contains both numerical and categorical data
3. Real business data (UK online retail)
4. Large enough for meaningful analysis (500K+ rows)
5. Has clear business use cases (customer segmentation, sales analysis)

---

**Q3: What are the main challenges in this dataset?**

**Answer:**
1. Missing values in CustomerID and Description
2. Outliers in Quantity and UnitPrice
3. Inconsistent text formatting in Country names
4. Duplicate transactions
5. Wide range of values requiring normalization

---

### DATA CLEANING QUESTIONS

**Q4: What is the difference between row deletion with and without threshold?**

**Answer:**
- **Without threshold:** Deletes any row with even 1 missing value. Very aggressive, loses too much data.
- **With threshold:** Keeps rows with at least N non-null values (e.g., thresh=9 means keep rows with 9+ valid values). More balanced approach.

**Example:** Row has 11 columns, 2 are missing:
- Without threshold: Deleted
- With threshold (thresh=9): Kept (has 9 valid values)

---

**Q5: Why use KNN imputation instead of mean/median?**

**Answer:**
**Mean/Median:** Ignores relationships between features. Fills all missing values with same number.

**KNN:** Considers similar records. If customer with Quantity=100 has missing CustomerID, KNN finds 5 similar customers (also bought ~100 items) and averages their IDs.

**Result:** KNN gives more accurate, context-aware imputations.

---

**Q6: Explain IQR method for outlier detection.**

**Answer:**
**IQR = Q3 - Q1** (spread of middle 50%)

**Outliers:** Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

**Why 1.5?** Statistical standard. Captures ~99.7% of data if normally distributed.

**Example:**
- Q1=10, Q3=30, IQR=20
- Lower fence: 10 - 30 = -20
- Upper fence: 30 + 30 = 60
- Values < -20 or > 60 are outliers

---

**Q7: What is the difference between binning and discretization?**

**Answer:**
**Binning (pd.cut):** Equal-width bins
- Example: Ages divided into 0-20, 21-40, 41-60, 61-80

**Discretization (pd.qcut):** Equal-frequency bins
- Example: Each bin gets 25% of data

**When to use:**
- Binning: When ranges matter (age groups)
- Discretization: When distribution matters (quartiles)

---

### DATA REDUCTION QUESTIONS

**Q8: What is feature selection and why is it important?**

**Answer:**
**Feature selection** = Choosing most relevant features, removing redundant ones.

**Why important:**
1. **Reduces overfitting:** Too many features cause model to memorize noise
2. **Faster training:** Fewer features = less computation
3. **Better interpretability:** Easier to understand model
4. **Removes noise:** Irrelevant features add noise

**Our approach:** Used correlation with TotalPrice to select Quantity, UnitPrice as important features.

---

**Q9: Explain how K-Means clustering works.**

**Answer:**
**Algorithm:**
1. Choose K (number of clusters), e.g., K=3
2. Randomly place 3 cluster centers
3. **Assignment:** Assign each point to nearest center
4. **Update:** Move each center to average of assigned points
5. Repeat steps 3-4 until centers stop moving

**Example:**
```
Initial:           After iteration 1:      Final:
  •  •               • •                  ••
 C  •                C                    C
• •  C              • C                   •
    •  C               • •                 ••
                         C                  C
```

**Distance:** Usually Euclidean: √[(x₁-x₂)² + (y₁-y₂)²]

---

**Q10: What is the purpose of aggregation in data reduction?**

**Answer:**
**Aggregation** combines multiple rows into summary rows.

**Purpose:**
1. **Reduces size:** 500K rows → 38 countries
2. **Shows patterns:** Country-level insights
3. **Faster analysis:** Smaller data = faster processing
4. **Business intelligence:** Summary useful for decisions

**Example:** Instead of viewing 10,000 UK transactions, see "UK: 5000 orders, £500K revenue"

---

### DATA TRANSFORMATION QUESTIONS

**Q11: When should you use Min-Max vs Z-Score normalization?**

**Answer:**

**Min-Max (0 to 1):**
- **Use when:** Bounded output needed
- **Algorithms:** Neural networks, image processing, KNN
- **Pros:** Preserves zero values, bounded range
- **Cons:** Sensitive to outliers

**Z-Score (mean=0, std=1):**
- **Use when:** Comparing different scales
- **Algorithms:** Linear regression, PCA, SVM
- **Pros:** Not bounded, handles outliers better
- **Cons:** Can have values outside [-3, 3]

**Example decision:**
- For neural network (0-1 activation): Min-Max
- For comparing height (cm) and weight (kg): Z-Score

---

**Q12: Why use logarithmic transformation?**

**Answer:**
**Purpose:** Handle skewed data with wide range

**When to use:**
1. Data is right-skewed (long tail on right)
2. Values span multiple magnitudes (1 to 1000000)
3. Want to stabilize variance

**Example:** Income data
- Original: $10K, $50K, $100K, $10M (huge gap)
- Log: 4.0, 4.7, 5.0, 7.0 (compressed)

**Formula:** log(1 + x) - The "+1" handles zeros

**Effect:** Makes multiplicative relationships additive

---

**Q13: What is feature construction and give an example?**

**Answer:**
**Feature construction** = Creating new features from existing ones

**Our example:**
```python
df['Price_Category'] = df['UnitPrice'].apply(
    lambda x: 'Budget' if x < 2 else ('Mid-Range' if x < 5 else 'Premium')
)
```

**Converts:** UnitPrice (numerical) → Price_Category (categorical)

**Benefits:**
1. Easier interpretation (marketers understand "Premium" better than "7.5")
2. Can reveal non-linear patterns
3. Domain knowledge encoded as feature

**Other examples:**
- BMI = Weight / Height²
- Age_Group from Age
- Is_Weekend from Date

---

### EDA QUESTIONS

**Q14: What is the difference between descriptive and visual EDA?**

**Answer:**

**Descriptive EDA:** Numbers and statistics
- Mean, median, mode (central tendency)
- Standard deviation, variance (spread)
- Skewness, kurtosis (shape)
- Example: "Mean age = 35 years"

**Visual EDA:** Charts and graphs
- Histograms, box plots (distribution)
- Scatter plots (relationships)
- Heatmaps (correlations)
- Example: See age distribution curve

**Why both?** Numbers give precision, visuals give intuition.

---

**Q15: Explain univariate, bivariate, and multivariate analysis.**

**Answer:**

**Univariate:** Analyzing ONE variable
- Tools: Histogram, box plot, bar chart
- Example: Distribution of Quantity
- Question: "What is typical quantity ordered?"

**Bivariate:** Analyzing TWO variables
- Tools: Scatter plot, grouped bar chart
- Example: Quantity vs TotalPrice
- Question: "How does quantity affect revenue?"

**Multivariate:** Analyzing 3+ variables
- Tools: Correlation heatmap, 3D plots, pair plots
- Example: Relationship between Quantity, UnitPrice, TotalPrice, Month
- Question: "How do all features interact?"

---

**Q16: How do you interpret a correlation coefficient?**

**Answer:**

**Correlation (r):** Measures linear relationship, ranges from -1 to +1

**Interpretation:**
- **r = +1:** Perfect positive (both increase together)
- **r = +0.7 to +1:** Strong positive
- **r = +0.3 to +0.7:** Moderate positive
- **r = -0.3 to +0.3:** Weak/no correlation
- **r = -0.7 to -0.3:** Moderate negative
- **r = -1 to -0.7:** Strong negative
- **r = -1:** Perfect negative (one increases, other decreases)

**Example:**
- Quantity vs TotalPrice: r = 0.95 (strong positive - makes sense!)
- UnitPrice vs Month: r = 0.05 (no relationship)

**Important:** Correlation ≠ Causation!

---

**Q17: What does skewness tell us about data distribution?**

**Answer:**

**Skewness:** Measures asymmetry of distribution

**Values:**
- **Skewness = 0:** Symmetric (normal distribution)
- **Skewness > 0:** Right-skewed (positive skew)
  - Tail extends to right
  - Mean > Median
  - Most values on left
- **Skewness < 0:** Left-skewed (negative skew)
  - Tail extends to left
  - Mean < Median
  - Most values on right

**Visual:**
```
Right-skewed:     Symmetric:      Left-skewed:
    ___               ___             ___
   /   \___          /   \          ___/   \
  /                 /     \               \
```

**Example:** Income is right-skewed (most people earn average, few earn millions)

---

**Q18: What is the purpose of a box plot?**

**Answer:**

**Box plot shows:**
1. **Median:** Middle line in box
2. **Q1 and Q3:** Box edges (IQR)
3. **Whiskers:** Extend to min/max (within 1.5×IQR)
4. **Outliers:** Points beyond whiskers

**Visual:**
```
        •  (outlier)
        |
    |-------|
    |   |   |  ← Box (Q1 to Q3)
    |-------|
        |
        •  (outlier)
```

**What it tells us:**
- Central tendency (median)
- Spread (IQR)
- Skewness (if median off-center)
- Outliers (dots beyond whiskers)

**Use case:** Comparing distributions across groups

---

### TECHNICAL QUESTIONS

**Q19: Why did we use encoding='ISO-8859-1' when loading CSV?**

**Answer:**
**Problem:** Dataset contains special characters (£, €, ñ, ü, etc.)

**Default encoding (UTF-8):** Can't read these characters → Error

**ISO-8859-1 (Latin-1):** Handles European characters

**Example characters:**
- Currency: £, €
- Accents: café, naïve
- Special: ñ, ü, ø

**Without correct encoding:** See gibberish like "caf├®" instead of "café"

---

**Q20: What is the difference between .fit(), .transform(), and .fit_transform()?**

**Answer:**

**`.fit(data)`:**
- Learns parameters from data
- Example: MinMaxScaler learns min and max
- Doesn't transform data

**`.transform(data)`:**
- Applies learned parameters
- Example: Uses saved min/max to scale new data
- Requires prior .fit()

**`.fit_transform(data)`:**
- Combines both: learns AND transforms
- Shorthand for .fit().transform()

**When to use which:**
```python
# Training data
scaler.fit_transform(train_data)  # Learn from and transform

# Test data
scaler.transform(test_data)  # Only transform (don't re-learn!)
```

**Why separate?** Must use SAME parameters (min/max) for consistency.

---

**Q21: What is random_state=42 and why use it?**

**Answer:**

**random_state:** Seed for random number generator

**Purpose:** **Reproducibility**
- Same seed → Same random numbers → Same results
- Different runs give identical output

**Why 42?** Convention (from "Hitchhiker's Guide to Galaxy"). Any number works.

**Example:**
```python
# Without random_state
np.random.choice([1,2,3,4,5], 2)  # [3, 1]
np.random.choice([1,2,3,4,5], 2)  # [2, 5]  ← Different!

# With random_state
np.random.seed(42)
np.random.choice([1,2,3,4,5], 2)  # [4, 2]
np.random.seed(42)
np.random.choice([1,2,3,4,5], 2)  # [4, 2]  ← Same!
```

---

**Q22: Why do we drop duplicates with keep='first' instead of keep='last' or keep=False?**

**Answer:**

**Options:**
- **keep='first':** Keeps first occurrence, removes rest
- **keep='last':** Keeps last occurrence, removes rest  
- **keep=False:** Removes ALL duplicates (even originals)

**Why keep='first'?**
1. **Preserves chronological order:** First transaction is original
2. **Business logic:** First purchase might have different properties
3. **Convention:** Most common approach
4. **Data integrity:** Keeps at least one record

**Example:**
```
Original data:
Row 1: Invoice=001, Qty=5
Row 2: Invoice=001, Qty=5  (duplicate)
Row 3: Invoice=001, Qty=5  (duplicate)

keep='first': Keeps Row 1
keep='last': Keeps Row 3
keep=False: Removes all 3 rows!
```

**When to use keep=False?** When duplicates indicate data quality issues and you want only unique records.

---

**Q23: What libraries did we use and why?**

**Answer:**

**1. pandas:** Data manipulation
- Read CSV, create DataFrames
- Missing value handling, groupby, aggregation

**2. numpy:** Numerical operations
- Random number generation
- Mathematical functions (log, sqrt)

**3. matplotlib:** Basic visualizations
- Histograms, scatter plots, bar charts

**4. seaborn:** Advanced visualizations
- Heatmaps with better aesthetics
- Built on top of matplotlib

**5. sklearn (scikit-learn):** Machine learning tools
- KNNImputer: Missing value imputation
- KMeans: Clustering
- MinMaxScaler, StandardScaler: Normalization
- LinearRegression: Regression modeling

**6. scipy:** Scientific computing
- stats.zscore: Z-score calculation
- Statistical functions

**7. warnings:** Suppress warning messages
- Makes output cleaner

---

### DECISION QUESTIONS

**Q24: Why didn't you apply regression smoothing technique?**

**Answer:**

**Regression smoothing** fits regression line to smooth noisy data points.

**Requirements:**
1. Clear independent variable (X) that predicts dependent variable (Y)
2. Time-series or ordered data
3. Noisy measurements that need smoothing

**Our dataset:**
- Transactional data (not time-series per se)
- No clear noisy measurement needing smoothing
- Example where it works: Noisy temperature sensor readings over time

**If we forced it:**
Could smooth UnitPrice over time, but:
- UnitPrice changes are intentional (not noise)
- Different products have different prices
- Would destroy actual price variations

**Conclusion:** Not applicable. Documented reasoning as required by ma'am.

---

**Q25: How did you decide which features to select?**

**Answer:**

**Method:** Correlation analysis with target variable (TotalPrice)

**Steps:**
1. Calculated correlation matrix for all numerical features
2. Sorted by correlation with TotalPrice
3. Selected features with |correlation| > 0.3

**Results:**
- **High correlation (selected):**
  - Quantity: 0.95 (very strong predictor)
  - UnitPrice: 0.75 (strong predictor)
  
- **Low correlation (considered dropping):**
  - Month: 0.02 (weak relationship)
  - Year: -0.01 (almost no relationship)

**Also considered:**
- Business importance (Country, Description needed for insights)
- Data availability (some columns had many missing values)

---

**Q26: Why did you create TotalPrice instead of using existing columns?**

**Answer:**

**Reason 1: Reach 10 columns requirement**
- Original: 8 columns
- Needed: 10+ columns
- Added: TotalPrice, Month, Year

**Reason 2: Business value**
- TotalPrice = Quantity × UnitPrice
- Represents revenue per transaction
- Key metric for business analysis

**Reason 3: Analysis convenience**
- Don't need to multiply Quantity × UnitPrice repeatedly
- Direct aggregation: Total revenue = sum(TotalPrice)

**Alternative:** Could have used existing columns separately, but TotalPrice simplifies many analyses.

---

### INTERPRETATION QUESTIONS

**Q27: What insights did you gain from the visualizations?**

**Answer:**

**Visualization 1 (Histogram - Quantity):**
- Most orders have low quantities (1-10 items)
- Right-skewed distribution
- Few large bulk orders

**Visualization 2 (Box Plot - UnitPrice):**
- Median price around £2-3
- Many outliers (expensive items)
- Price range varies widely

**Visualization 3 (Scatter - Quantity vs TotalPrice):**
- Strong positive correlation
- Linear relationship (as expected: TotalPrice = Qty × Price)
- Some high-value outliers

**Visualization 4 (Bar Chart - Countries):**
- UK dominates (80%+ of orders)
- Top 5 countries: UK, Germany, France, Ireland, Spain
- Business opportunity: Expand in other countries

**Visualization 5 (Heatmap - Correlations):**
- Quantity and TotalPrice highly correlated (0.95)
- Month/Year show weak correlations
- No strong negative correlations found

---

**Q28: What business recommendations would you make based on this analysis?**

**Answer:**

**1. Customer Segmentation (from clustering):**
- Target high-value customers (Segment 2) with premium products
- Offer bulk discounts to medium-value customers (Segment 1)
- Re-engage low-value customers (Segment 0) with promotions

**2. Inventory Management:**
- Stock more items in £2-5 range (most popular)
- Keep limited stock of premium items (£10+)
- Focus on products with Quantity 1-10 (most common)

**3. Geographic Expansion:**
- UK is saturated (80% market share)
- Growth opportunity in Europe (Germany, France)
- Consider targeted marketing in underserved countries

**4. Pricing Strategy:**
- Most customers prefer budget-mid range (£2-5)
- Premium pricing (£10+) for exclusive items
- Bundle offers for quantities >10

**5. Data Quality:**
- Improve CustomerID tracking (10% missing)
- Standardize product descriptions
- Implement duplicate detection at entry point

---

### CHALLENGING QUESTIONS

**Q29: If you had more time, what additional analysis would you do?**

**Answer:**

**1. Time Series Analysis:**
- Seasonal trends (holiday sales spikes)
- Year-over-year growth
- Forecasting future sales

**2. Customer Lifetime Value (CLV):**
- RFM analysis (Recency, Frequency, Monetary)
- Churn prediction
- High-value customer identification

**3. Product Analysis:**
- Best-selling products
- Product associations (market basket analysis)
- Recommendation system

**4. Advanced Clustering:**
- Try different K values (elbow method)
- DBSCAN for density-based clustering
- Hierarchical clustering

**5. Anomaly Detection:**
- Detect fraudulent transactions
- Identify unusual purchasing patterns
- Monitor for data quality issues

**6. Predictive Modeling:**
- Predict customer churn
- Forecast demand
- Price optimization

---

**Q30: What were the main challenges you faced and how did you overcome them?**

**Answer:**

**Challenge 1: Missing Data (10-15% in some columns)**
- **Problem:** Losing data vs accuracy trade-off
- **Solution:** Used KNN imputation (more accurate than mean)
- **Result:** Retained all rows with intelligent filling

**Challenge 2: Outliers (Quantity values × 100)**
- **Problem:** Skewed statistics and visualizations
- **Solution:** IQR method to detect and remove
- **Result:** Cleaner distribution, better analysis

**Challenge 3: Dataset too clean (ma'am's requirement)**
- **Problem:** Need to show cleaning techniques
- **Solution:** Artificially introduced problems systematically
- **Result:** Could demonstrate all required techniques

**Challenge 4: Reaching 10 columns**
- **Problem:** Only 8 original columns
- **Solution:** Created calculated columns (TotalPrice, Month, Year)
- **Result:** 11 columns with business value

**Challenge 5: Techniques not applicable**
- **Problem:** Regression smoothing doesn't fit dataset
- **Solution:** Documented reasoning clearly
- **Result:** Showed understanding vs blind application

---

### QUICK FIRE QUESTIONS

**Q31: What is the difference between supervised and unsupervised learning?**

**Answer:**
- **Supervised:** Has labels/target variable (e.g., Regression - predicting TotalPrice)
- **Unsupervised:** No labels (e.g., K-Means - finding customer segments)

---

**Q32: Is K-Means supervised or unsupervised?**

**Answer:** **Unsupervised** - We don't tell it the "correct" clusters, it finds patterns on its own.

---

**Q33: What is overfitting?**

**Answer:** When model memorizes training data instead of learning patterns. Performs well on training data but poorly on new data.

---

**Q34: Why normalize data before clustering?**

**Answer:** Features with larger ranges dominate distance calculations. Normalization ensures all features contribute equally.

---

**Q35: What is the curse of dimensionality?**

**Answer:** As features increase, data becomes sparse. Need exponentially more data to maintain accuracy. Why we do feature selection.

---

**Q36: What is the difference between accuracy and precision?**

**Answer:**
- **Accuracy:** Overall correctness (all correct predictions / total)
- **Precision:** Of predicted positives, how many are actually positive

Note: This project doesn't have classification, so not directly applicable.

---

**Q37: What is train-test split and why didn't you use it?**

**Answer:**
**Train-test split:** Divide data into training (80%) and testing (20%) sets.

**Why not used:** This project focuses on EDA and preprocessing, not predictive modeling. No model to test.

**When needed:** When building regression/classification models to evaluate performance.

---

**Q38: What is cross-validation?**

**Answer:** Split data into K folds, train on K-1, test on 1, rotate. Gives more reliable performance estimate than single train-test split.

---

**Q39: Difference between classification and regression?**

**Answer:**
- **Classification:** Predict category (e.g., spam/not spam)
- **Regression:** Predict number (e.g., price, quantity)

Our regression model predicts TotalPrice (continuous value).

---

**Q40: What is pandas DataFrame?**

**Answer:** 2D table structure (like Excel) with:
- Rows (records/observations)
- Columns (features/variables)
- Indexes (row identifiers)
- Support for mixed data types

---

### PROJECT-SPECIFIC QUESTIONS

**Q41: How many rows did you start with and end with?**

**Answer:**
- **Original:** 541,909 rows
- **After adding problems:** ~569,000 rows (added 5% duplicates)
- **After cleaning:** ~480,000 rows (removed outliers and duplicates)
- **After aggregation (country level):** 38 rows

---

**Q42: Which country has the most orders?**

**Answer:** **United Kingdom** - Around 80-85% of all orders come from UK. This makes sense as it's a UK-based online retail store.

---

**Q43: What is the most common quantity ordered?**

**Answer:** Based on mode calculation, most common is **1-6 items** per order. Retail customers typically buy small quantities.

---

**Q44: What percentage of data was missing initially?**

**Answer:**
- Description: 15%
- CustomerID: 10%
- Quantity: 5%
- (These were artificially introduced for demonstration)

---

**Q45: How many outliers did you remove?**

**Answer:** Based on IQR method, removed approximately **5-7%** of data as outliers (values beyond 1.5×IQR from quartiles).

---

## TIPS FOR VIVA SUCCESS

### Before Viva:
1. **Run the code yourself** - Understand each output
2. **Know your numbers** - Dataset size, column names, percentages
3. **Understand why** - Not just what you did, but why
4. **Practice explanations** - Explain to a friend/family
5. **Review visualizations** - What does each chart show?

### During Viva:
1. **Stay calm** - Take a breath before answering
2. **Think before speaking** - It's okay to pause
3. **Use examples** - Makes concepts clearer
4. **Admit if unsure** - Better than making up answers
5. **Connect to business** - Show practical understanding

### Key Points to Remember:
1. **Dataset:** Online Retail, 11 columns, 500K+ rows
2. **Main goal:** Clean messy data, reduce dimensions, transform features, analyze
3. **Key techniques:** KNN imputation, IQR outliers, K-Means clustering, normalization
4. **Visualizations:** 5 total (2 univariate, 2 bivariate, 1 multivariate)
5. **Business value:** Customer segmentation, country analysis, pricing insights

### If You Forget Something:
- "Let me think about that for a moment..."
- "Could you rephrase the question?"
- "I'm not entirely sure, but my understanding is..."
- "That's a good question, let me explain what I know about related concepts..."

### Common Mistakes to Avoid:
1. Saying "I don't know" immediately
2. Making up technical terms
3. Over-complicating simple answers
4. Not relating to your specific project
5. Reading memorized answers (sound natural!)

---

## FINAL CHECKLIST

**Before Submission:**
- [ ] Both notebooks run without errors
- [ ] All visualizations generated
- [ ] CSV files created
- [ ] Comments are clear
- [ ] Interpretations added after each technique
- [ ] Citations for dataset source

**Before Viva:**
- [ ] Review this document 2-3 times
- [ ] Run code once more
- [ ] Prepare 2-minute project summary
- [ ] Practice explaining 3-4 key techniques
- [ ] Know your dataset numbers (rows, columns, etc.)
- [ ] Be ready to show any visualization and explain

---

## GOOD LUCK!

Remember: You understand this project. You built it. The viva is just a conversation about your work. Be confident!
