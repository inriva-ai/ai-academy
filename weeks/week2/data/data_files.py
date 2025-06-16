"""
Week 2 Data File Generator
==========================

This script generates sample datasets for Week 2 exercises.
Creates realistic datasets with intentional data quality issues
for practicing advanced preprocessing techniques.

Usage:
    python generate_week2_data.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_week2_data_directory():
    """Create data directory structure for Week 2"""
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def generate_titanic_dataset(n_samples=891):
    """Generate Titanic-like dataset with realistic patterns and missing values"""
    np.random.seed(42)
    
    # Basic demographics
    ages = np.random.normal(30, 15, n_samples)
    ages = np.clip(ages, 0, 80)
    
    # Introduce realistic missing patterns for Age
    # Missing more likely for certain groups
    missing_prob = np.where(ages > 60, 0.3, 0.15)  # Elderly records more likely to be incomplete
    missing_age_mask = np.random.random(n_samples) < missing_prob
    ages[missing_age_mask] = np.nan
    
    # Gender with realistic distribution
    sexes = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])
    
    # Class with social stratification
    pclasses = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
    
    # Embarkation ports
    embarked = np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
    # Some missing embarkation data
    missing_embarked_mask = np.random.random(n_samples) < 0.002
    embarked[missing_embarked_mask] = None
    
    # Family relationships
    sibsp = np.random.poisson(0.5, n_samples)
    parch = np.random.poisson(0.4, n_samples)
    
    # Fare correlated with class and family size
    fare_base = {1: 80, 2: 20, 3: 10}
    fares = []
    for i, pc in enumerate(pclasses):
        base_fare = fare_base[pc]
        family_size = sibsp[i] + parch[i] + 1
        # More people = higher total fare, but with some randomness
        fare = np.random.lognormal(np.log(base_fare * family_size * 0.8), 0.4)
        fares.append(fare)
    
    # Some missing fare values
    missing_fare_mask = np.random.random(n_samples) < 0.001
    fares = np.array(fares)
    fares[missing_fare_mask] = np.nan
    
    # Names with extractable titles
    titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.', 'Rev.', 'Col.', 'Major.']
    title_probs = [0.5, 0.15, 0.15, 0.08, 0.04, 0.03, 0.03, 0.02]
    
    # Title distribution influenced by class and gender
    names = []
    for i in range(n_samples):
        if sexes[i] == 'male':
            if ages[i] < 18:
                title = 'Master.'
            else:
                title = np.random.choice(['Mr.', 'Dr.', 'Rev.', 'Col.', 'Major.'], 
                                       p=[0.85, 0.06, 0.04, 0.03, 0.02])
        else:
            if np.random.random() < 0.3:  # 30% married
                title = 'Mrs.'
            else:
                title = np.random.choice(['Miss.', 'Dr.'], p=[0.92, 0.08])
        
        # Generate full name
        first_names = ['John', 'Mary', 'James', 'Patricia', 'Robert', 'Jennifer', 
                      'Michael', 'Linda', 'William', 'Elizabeth']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
                     'Miller', 'Davis', 'Rodriguez', 'Martinez']
        
        first_name = np.random.choice(first_names)
        last_name = np.random.choice(last_names)
        full_name = f"{last_name}, {title} {first_name}"
        names.append(full_name)
    
    # Ticket numbers (for demonstration of text features)
    tickets = []
    for i in range(n_samples):
        if np.random.random() < 0.7:  # 70% have alphanumeric tickets
            ticket = f"{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 2))}{np.random.randint(1000, 99999)}"
        else:  # 30% have numeric tickets
            ticket = str(np.random.randint(10000, 999999))
        tickets.append(ticket)
    
    # Cabin information (mostly missing, as was typical)
    cabins = []
    for i in range(n_samples):
        if pclasses[i] == 1 and np.random.random() < 0.6:  # First class more likely to have cabin
            deck = np.random.choice(['A', 'B', 'C'])
            cabin_num = np.random.randint(1, 200)
            cabins.append(f"{deck}{cabin_num}")
        elif pclasses[i] == 2 and np.random.random() < 0.2:  # Second class sometimes
            deck = np.random.choice(['D', 'E'])
            cabin_num = np.random.randint(1, 200)
            cabins.append(f"{deck}{cabin_num}")
        else:
            cabins.append(None)  # Most don't have cabin info
    
    # Survival with realistic patterns
    survival_prob = 0.32  # Overall historical survival rate
    
    # Apply survival biases
    survival_adjustments = np.zeros(n_samples)
    
    # Women and children first
    survival_adjustments += (sexes == 'female') * 0.4
    survival_adjustments += (ages < 16) * 0.3
    
    # Class privilege
    survival_adjustments += (pclasses == 1) * 0.35
    survival_adjustments += (pclasses == 2) * 0.15
    survival_adjustments -= (pclasses == 3) * 0.1
    
    # Family effects (medium families had better survival)
    family_size = sibsp + parch + 1
    survival_adjustments += ((family_size >= 2) & (family_size <= 4)) * 0.1
    survival_adjustments -= (family_size > 6) * 0.2  # Large families struggled
    
    # Apply adjustments
    final_survival_prob = np.clip(survival_prob + survival_adjustments, 0.05, 0.95)
    survived = np.random.binomial(1, final_survival_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'PassengerId': range(1, n_samples + 1),
        'Survived': survived,
        'Pclass': pclasses,
        'Name': names,
        'Sex': sexes,
        'Age': ages,
        'SibSp': sibsp,
        'Parch': parch,
        'Ticket': tickets,
        'Fare': fares,
        'Cabin': cabins,
        'Embarked': embarked
    })
    
    return df

def generate_customer_reviews_dataset(n_samples=500):
    """Generate customer reviews dataset for text processing exercises"""
    np.random.seed(42)
    
    # Product categories
    categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports']
    product_categories = np.random.choice(categories, n_samples)
    
    # Ratings (1-5 stars)
    ratings = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.1, 0.2, 0.35, 0.25])
    
    # Generate review text based on rating
    review_templates = {
        1: [
            "Terrible product. Complete waste of money. Would not recommend.",
            "Poor quality and terrible customer service. Very disappointed.",
            "Worst purchase ever. Product broke immediately. Avoid at all costs.",
            "Cheaply made and overpriced. Save your money and buy elsewhere."
        ],
        2: [
            "Below average product. Had some issues with quality.",
            "Not impressed. Product didn't meet expectations.",
            "Mediocre quality for the price. Would look elsewhere next time.",
            "Some problems with the product. Customer service was unhelpful."
        ],
        3: [
            "Average product. Nothing special but does what it's supposed to.",
            "Decent quality for the price. Not amazing but acceptable.",
            "It's okay. Gets the job done but could be better.",
            "Fair product. Some good points, some bad points."
        ],
        4: [
            "Good product with minor issues. Would recommend with reservations.",
            "Pretty good quality and value. Happy with the purchase overall.",
            "Solid product. A few small problems but generally satisfied.",
            "Good value for money. Minor flaws but mostly positive experience."
        ],
        5: [
            "Excellent product! Exceeded expectations. Highly recommend!",
            "Outstanding quality and fantastic customer service. Perfect!",
            "Amazing product. Best purchase I've made in years. Five stars!",
            "Perfect product with incredible value. Will definitely buy again!"
        ]
    }
    
    reviews = []
    for rating in ratings:
        template = np.random.choice(review_templates[rating])
        # Add some variation
        if np.random.random() < 0.3:  # 30% chance of longer review
            additional_text = " " + np.random.choice([
                "Shipping was fast and packaging was good.",
                "Customer service was helpful when I had questions.",
                "Product arrived as described and on time.",
                "Would consider buying from this seller again.",
                "Overall experience was smooth and hassle-free."
            ])
            template += additional_text
        reviews.append(template)
    
    # Customer information
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_samples + 1)]
    
    # Purchase amounts correlated with ratings
    base_amounts = np.random.lognormal(np.log(50), 0.8, n_samples)
    # Higher ratings tend to be for more expensive items (satisfaction bias)
    rating_multipliers = {1: 0.7, 2: 0.8, 3: 1.0, 4: 1.2, 5: 1.3}
    purchase_amounts = []
    for i, rating in enumerate(ratings):
        amount = base_amounts[i] * rating_multipliers[rating]
        purchase_amounts.append(round(amount, 2))
    
    # Review dates (last 6 months)
    start_date = pd.Timestamp('2024-07-01')
    end_date = pd.Timestamp('2024-12-31')
    date_range = pd.date_range(start_date, end_date, freq='D')
    review_dates = np.random.choice(date_range, n_samples)
    
    # Some missing values in review text (incomplete reviews)
    missing_review_mask = np.random.random(n_samples) < 0.02
    reviews = np.array(reviews, dtype=object)
    reviews[missing_review_mask] = None
    
    df = pd.DataFrame({
        'ReviewID': range(1, n_samples + 1),
        'CustomerID': customer_ids,
        'ProductCategory': product_categories,
        'Rating': ratings,
        'ReviewText': reviews,
        'PurchaseAmount': purchase_amounts,
        'ReviewDate': review_dates
    })
    
    return df

def generate_financial_dataset(n_samples=1000):
    """Generate financial dataset with complex relationships for advanced exercises"""
    np.random.seed(42)
    
    # Customer demographics
    ages = np.random.normal(45, 18, n_samples)
    ages = np.clip(ages, 18, 85)
    
    # Education levels
    education_levels = ['High School', 'Some College', 'Bachelor', 'Master', 'PhD']
    education_probs = [0.3, 0.2, 0.3, 0.15, 0.05]
    education = np.random.choice(education_levels, n_samples, p=education_probs)
    
    # Employment status
    employment_status = np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], 
                                       n_samples, p=[0.65, 0.15, 0.08, 0.12])
    
    # Income correlated with education and employment
    education_income_map = {
        'High School': 35000, 'Some College': 40000, 'Bachelor': 55000, 
        'Master': 75000, 'PhD': 90000
    }
    employment_multiplier = {
        'Employed': 1.0, 'Self-Employed': 1.2, 'Unemployed': 0.1, 'Retired': 0.4
    }
    
    incomes = []
    for i in range(n_samples):
        base_income = education_income_map[education[i]]
        emp_mult = employment_multiplier[employment_status[i]]
        age_factor = 1 + (ages[i] - 25) * 0.01  # Income increases with age
        
        income = base_income * emp_mult * age_factor * np.random.lognormal(0, 0.3)
        incomes.append(max(0, income))
    
    # Credit score correlated with income and age
    base_credit_scores = 650 + (np.array(incomes) / 1000) * 0.3 + (ages - 18) * 2
    credit_scores = np.clip(base_credit_scores + np.random.normal(0, 50, n_samples), 300, 850)
    
    # Loan amounts requested
    loan_amounts = np.random.lognormal(np.log(25000), 1.2, n_samples)
    loan_amounts = np.clip(loan_amounts, 1000, 500000)
    
    # Debt-to-income ratio
    monthly_debt = np.random.gamma(2, 500, n_samples)  # Existing monthly debt
    monthly_income = np.array(incomes) / 12
    debt_to_income = monthly_debt / (monthly_income + 1)  # Avoid division by zero
    
    # Employment length (years)
    employment_length = np.random.exponential(5, n_samples)
    employment_length = np.clip(employment_length, 0, 40)
    
    # Number of open credit lines
    num_credit_lines = np.random.poisson(3, n_samples)
    num_credit_lines = np.clip(num_credit_lines, 0, 15)
    
    # Home ownership
    home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.4, 0.25, 0.35])
    
    # Loan purpose
    loan_purposes = ['debt_consolidation', 'home_improvement', 'major_purchase', 
                    'medical', 'vacation', 'wedding', 'car', 'other']
    loan_purpose = np.random.choice(loan_purposes, n_samples, 
                                   p=[0.3, 0.15, 0.12, 0.1, 0.08, 0.05, 0.1, 0.1])
    
    # Interest rate (determined by credit score and other factors)
    base_interest_rate = 25 - (credit_scores - 300) / 550 * 20  # Higher credit = lower rate
    interest_rate = np.clip(base_interest_rate + np.random.normal(0, 2, n_samples), 3, 25)
    
    # Loan approval (target variable)
    # Complex approval logic
    approval_prob = 0.5  # Base probability
    
    # Credit score effect
    approval_prob += (credit_scores - 650) / 200 * 0.3
    
    # Income effect  
    approval_prob += (np.log(np.array(incomes) + 1) - np.log(50000)) / 5 * 0.2
    
    # Debt-to-income effect
    approval_prob -= debt_to_income * 0.5
    
    # Employment length effect
    approval_prob += np.minimum(employment_length / 10, 1) * 0.1
    
    # Loan amount effect (larger loans harder to get)
    approval_prob -= (loan_amounts - 25000) / 100000 * 0.2
    
    # Clip probabilities
    approval_prob = np.clip(approval_prob, 0.05, 0.95)
    
    # Generate approvals
    loan_approved = np.random.binomial(1, approval_prob)
    
    # Add some missing values
    missing_employment_mask = np.random.random(n_samples) < 0.05
    employment_length[missing_employment_mask] = np.nan
    
    missing_income_mask = np.random.random(n_samples) < 0.02
    incomes = np.array(incomes)
    incomes[missing_income_mask] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'LoanID': [f"LOAN_{i:06d}" for i in range(1, n_samples + 1)],
        'Age': ages.astype(int),
        'Education': education,
        'EmploymentStatus': employment_status,
        'EmploymentLength': employment_length,
        'AnnualIncome': incomes,
        'CreditScore': credit_scores.astype(int),
        'LoanAmount': loan_amounts,
        'InterestRate': interest_rate,
        'DebtToIncome': debt_to_income,
        'NumCreditLines': num_credit_lines,
        'HomeOwnership': home_ownership,
        'LoanPurpose': loan_purpose,
        'LoanApproved': loan_approved
    })
    
    return df

def create_data_metadata():
    """Create metadata describing the datasets"""
    metadata = {
        "titanic.csv": {
            "description": "Titanic passenger dataset with realistic missing values and survival patterns",
            "target_variable": "Survived",
            "features": {
                "PassengerId": "Unique passenger identifier",
                "Pclass": "Ticket class (1=1st, 2=2nd, 3=3rd)",
                "Name": "Passenger name with extractable title",
                "Sex": "Gender",
                "Age": "Age in years (has missing values)",
                "SibSp": "Number of siblings/spouses aboard",
                "Parch": "Number of parents/children aboard", 
                "Ticket": "Ticket number",
                "Fare": "Passenger fare (has some missing values)",
                "Cabin": "Cabin number (mostly missing)",
                "Embarked": "Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)"
            },
            "missing_values": ["Age (~20%)", "Fare (~0.1%)", "Cabin (~77%)", "Embarked (~0.2%)"],
            "use_cases": ["Classification", "Missing value handling", "Feature engineering"]
        },
        "customer_reviews.csv": {
            "description": "Customer product reviews with ratings and text for NLP exercises",
            "target_variable": "Rating",
            "features": {
                "ReviewID": "Unique review identifier",
                "CustomerID": "Customer identifier", 
                "ProductCategory": "Product category",
                "Rating": "1-5 star rating",
                "ReviewText": "Customer review text (some missing)",
                "PurchaseAmount": "Purchase amount in dollars",
                "ReviewDate": "Date of review"
            },
            "missing_values": ["ReviewText (~2%)"],
            "use_cases": ["Text processing", "Sentiment analysis", "LangChain integration"]
        },
        "financial_data.csv": {
            "description": "Loan application dataset with complex feature relationships",
            "target_variable": "LoanApproved", 
            "features": {
                "LoanID": "Unique loan identifier",
                "Age": "Applicant age",
                "Education": "Education level",
                "EmploymentStatus": "Employment status",
                "EmploymentLength": "Years employed (has missing values)",
                "AnnualIncome": "Annual income (has missing values)",
                "CreditScore": "Credit score (300-850)",
                "LoanAmount": "Requested loan amount",
                "InterestRate": "Offered interest rate",
                "DebtToIncome": "Debt-to-income ratio",
                "NumCreditLines": "Number of open credit lines",
                "HomeOwnership": "Home ownership status",
                "LoanPurpose": "Purpose of loan"
            },
            "missing_values": ["EmploymentLength (~5%)", "AnnualIncome (~2%)"],
            "use_cases": ["Binary classification", "Feature interactions", "Complex preprocessing"]
        }
    }
    return metadata

def main():
    """Generate all Week 2 datasets"""
    print("🔄 Generating Week 2 datasets...")
    
    # Create data directory
    data_dir = create_week2_data_directory()
    print(f"📁 Data directory: {data_dir}")
    
    # Generate datasets
    print("\n📊 Generating datasets...")
    
    # 1. Titanic dataset
    print("   🚢 Creating Titanic dataset...")
    titanic_df = generate_titanic_dataset(891)
    titanic_path = data_dir / "titanic.csv"
    titanic_df.to_csv(titanic_path, index=False)
    print(f"      ✅ Saved: {titanic_path} ({titanic_df.shape})")
    
    # 2. Customer reviews dataset
    print("   📝 Creating customer reviews dataset...")
    reviews_df = generate_customer_reviews_dataset(500)
    reviews_path = data_dir / "customer_reviews.csv"
    reviews_df.to_csv(reviews_path, index=False)
    print(f"      ✅ Saved: {reviews_path} ({reviews_df.shape})")
    
    # 3. Financial dataset
    print("   💰 Creating financial dataset...")
    financial_df = generate_financial_dataset(1000)
    financial_path = data_dir / "financial_data.csv"
    financial_df.to_csv(financial_path, index=False)
    print(f"      ✅ Saved: {financial_path} ({financial_df.shape})")
    
    # 4. Create metadata file
    print("   📋 Creating metadata...")
    metadata = create_data_metadata()
    metadata_path = data_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"      ✅ Saved: {metadata_path}")
    
    # 5. Create README
    print("   📚 Creating data README...")
    readme_content = """# Week 2 Data Files

This directory contains sample datasets for Week 2 exercises and workshops.

## Datasets

### titanic.csv
- **Rows**: 891 passengers
- **Purpose**: Classification, missing value handling, feature engineering
- **Target**: Survived (0/1)
- **Missing Values**: Age (~20%), Cabin (~77%), Fare (~0.1%), Embarked (~0.2%)

### customer_reviews.csv  
- **Rows**: 500 reviews
- **Purpose**: Text processing, sentiment analysis, LangChain integration
- **Target**: Rating (1-5 stars)
- **Missing Values**: ReviewText (~2%)

### financial_data.csv
- **Rows**: 1000 loan applications  
- **Purpose**: Binary classification, feature interactions, complex preprocessing
- **Target**: LoanApproved (0/1)
- **Missing Values**: EmploymentLength (~5%), AnnualIncome (~2%)

## Data Quality Issues

Each dataset includes realistic data quality issues for practicing advanced preprocessing:

- **Missing Values**: Strategic patterns (not random)
- **Outliers**: Extreme values in numerical columns
- **Invalid Data**: Some negative values where inappropriate
- **Inconsistent Formats**: Varied text formats and categorical encodings
- **Correlated Features**: Highly correlated variables for feature selection practice

## Usage

Load datasets with pandas:
```python
import pandas as pd

# Load dataset
df = pd.read_csv('titanic.csv')

# Check for missing values
print(df.isnull().sum())

# Basic info
print(df.info())
```

See `dataset_metadata.json` for detailed feature descriptions.
"""
    
    readme_path = data_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"      ✅ Saved: {readme_path}")
    
    print(f"\n🎉 Week 2 data generation complete!")
    print(f"📁 Files created in: {data_dir}")
    print(f"   - titanic.csv ({titanic_df.shape[0]} rows)")
    print(f"   - customer_reviews.csv ({reviews_df.shape[0]} rows)")
    print(f"   - financial_data.csv ({financial_df.shape[0]} rows)")
    print(f"   - dataset_metadata.json")
    print(f"   - README.md")
    
    print(f"\n🎯 Ready for Week 2 exercises!")

if __name__ == "__main__":
    main()