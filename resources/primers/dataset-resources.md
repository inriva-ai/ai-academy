# Open Source Datasets for AI/ML Internship Program

## Table of Contents
1. [Week 1-2: Foundational Datasets](#week-1-2-foundational-datasets)
2. [Week 3-4: Supervised Learning Datasets](#week-3-4-supervised-learning-datasets)
3. [Week 5-6: Advanced ML & Text Datasets](#week-5-6-advanced-ml--text-datasets)
4. [Week 7-8: Capstone Project Datasets](#week-7-8-capstone-project-datasets)
5. [Specialized Collections](#specialized-collections)
6. [Real-Time & API Datasets](#real-time--api-datasets)
7. [Image & Computer Vision](#image--computer-vision)
8. [Time Series & Financial Data](#time-series--financial-data)
9. [Government & Research Data](#government--research-data)
10. [Synthetic & Generated Datasets](#synthetic--generated-datasets)

---

## Week 1-2: Foundational Datasets
*Small, clean datasets perfect for learning pandas, visualization, and basic statistics*

### Classic Learning Datasets

**Iris Dataset**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- **Size**: 150 samples, 4 features
- **Use Case**: Classification, basic statistics, visualization
- **Skills**: Pandas basics, matplotlib, seaborn
- **Download**: Built into scikit-learn: `from sklearn.datasets import load_iris`

**Boston Housing**
- **Source**: [Kaggle](https://www.kaggle.com/c/boston-housing)
- **Size**: 506 samples, 13 features
- **Use Case**: Regression, feature analysis
- **Skills**: Data preprocessing, correlation analysis
- **Note**: Good for understanding feature relationships

**Wine Quality**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Size**: 6,497 samples, 11 features
- **Use Case**: Classification/regression, quality prediction
- **Skills**: EDA, feature engineering
- **Direct Link**: [Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)

**Tips Dataset**
- **Source**: [Seaborn Built-in](https://github.com/mwaskom/seaborn-data)
- **Size**: 244 samples, 7 features
- **Use Case**: Basic analysis, group comparisons
- **Skills**: Groupby operations, categorical analysis
- **Access**: `import seaborn as sns; tips = sns.load_dataset('tips')`

### Simple Business Datasets

**Superstore Sales**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)
- **Size**: 9,994 sales records
- **Use Case**: Sales analysis, time series basics
- **Skills**: Date handling, business metrics
- **Topics**: Revenue analysis, regional comparisons

**Employee Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set)
- **Size**: 311 employees, 36 features
- **Use Case**: HR analytics, employee satisfaction
- **Skills**: Categorical data, correlation analysis
- **Topics**: Salary analysis, performance metrics

---

## Week 3-4: Supervised Learning Datasets
*Medium-sized datasets for classification and regression practice*

### Classification Datasets

**Titanic Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/c/titanic)
- **Size**: 891 training samples, 12 features
- **Use Case**: Binary classification, survival prediction
- **Skills**: Missing value handling, feature engineering
- **Challenge**: Famous beginner ML competition
- **Topics**: Data cleaning, categorical encoding

**Heart Disease Dataset**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Size**: 303 samples, 14 features
- **Use Case**: Medical diagnosis prediction
- **Skills**: Binary classification, medical data
- **Alternative**: [Kaggle Version](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)

**Mushroom Dataset**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/mushroom)
- **Size**: 8,124 samples, 22 features
- **Use Case**: Binary classification (edible/poisonous)
- **Skills**: Categorical features, safety-critical ML
- **Topic**: Decision trees work particularly well

**Bank Marketing Dataset**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Size**: 45,211 samples, 17 features
- **Use Case**: Marketing campaign success prediction
- **Skills**: Imbalanced classification, business metrics
- **Topics**: Customer segmentation, campaign optimization

### Regression Datasets

**California Housing**
- **Source**: [Scikit-learn Built-in](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- **Size**: 20,640 samples, 8 features
- **Use Case**: House price prediction
- **Skills**: Regression, geographic data
- **Access**: `from sklearn.datasets import fetch_california_housing`

**Bike Sharing Dataset**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- **Size**: 17,379 samples, 16 features
- **Use Case**: Demand prediction, time series regression
- **Skills**: Seasonal patterns, weather impact
- **Alternative**: [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)

**Energy Efficiency Dataset**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)
- **Size**: 768 samples, 10 features
- **Use Case**: Building energy prediction
- **Skills**: Multi-output regression, engineering applications
- **Topics**: Green technology, optimization

### Multi-class Classification

**Digits Dataset**
- **Source**: [Scikit-learn Built-in](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- **Size**: 1,797 samples, 64 features (8x8 images)
- **Use Case**: Handwritten digit recognition
- **Skills**: Image classification, digit recognition
- **Access**: `from sklearn.datasets import load_digits`

**Wine Dataset**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine)
- **Size**: 178 samples, 13 features
- **Use Case**: Wine type classification
- **Skills**: Multi-class classification, chemical analysis
- **Access**: `from sklearn.datasets import load_wine`

---

## Week 5-6: Advanced ML & Text Datasets
*Larger, more complex datasets for advanced techniques and NLP*

### Text and NLP Datasets

**IMDB Movie Reviews**
- **Source**: [Stanford AI](http://ai.stanford.edu/~amaas/data/sentiment/)
- **Size**: 50,000 movie reviews
- **Use Case**: Sentiment analysis, text classification
- **Skills**: Text preprocessing, TF-IDF, word embeddings
- **Alternative**: [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

**News Category Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **Size**: 210,000+ news articles
- **Use Case**: Text classification, topic modeling
- **Skills**: Large text processing, category prediction
- **Topics**: News analysis, content classification

**Twitter Sentiment Analysis**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size**: 1.6 million tweets
- **Use Case**: Social media sentiment analysis
- **Skills**: Short text processing, emoji handling
- **Challenge**: Informal language, abbreviations

**Amazon Product Reviews**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Size**: 568,454 reviews
- **Use Case**: Review analysis, rating prediction
- **Skills**: Text mining, customer analytics
- **Topics**: E-commerce insights, customer satisfaction

**BBC News Articles**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification)
- **Size**: 2,225 articles, 5 categories
- **Use Case**: Document classification, topic analysis
- **Skills**: Long text processing, journalism analysis
- **Good for**: RAG systems, document Q&A

### Large-Scale ML Datasets

**Credit Card Fraud Detection**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Use Case**: Anomaly detection, fraud prevention
- **Skills**: Imbalanced data, ensemble methods
- **Challenge**: High class imbalance (0.17% fraud)

**Adult Census Income**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Size**: 48,842 samples, 14 features
- **Use Case**: Income prediction, demographic analysis
- **Skills**: Categorical encoding, bias detection
- **Topics**: Fairness in ML, social impact
- **Alternative**: [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income)

**Online Retail Dataset**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Size**: 541,909 transactions
- **Use Case**: Market basket analysis, customer segmentation
- **Skills**: Association rules, clustering
- **Topics**: E-commerce analytics, recommendation systems

---

## Week 7-8: Capstone Project Datasets
*Complex, real-world datasets suitable for comprehensive projects*

### Comprehensive Business Datasets

**E-commerce Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **Size**: 541,909 transactions
- **Use Case**: Complete e-commerce analysis
- **Skills**: Customer analytics, sales forecasting, recommendation
- **Project Ideas**: Customer lifetime value, churn prediction, recommendation system

**Airbnb Dataset**
- **Source**: [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
- **Size**: Varies by city (NYC: ~50k listings)
- **Use Case**: Price prediction, location analysis
- **Skills**: Geographic data, price modeling, feature engineering
- **Project Ideas**: Dynamic pricing, market analysis, demand forecasting

**Spotify Dataset**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
- **Size**: 586,672 tracks with audio features
- **Use Case**: Music recommendation, audio analysis
- **Skills**: Clustering, recommendation systems, audio features
- **Project Ideas**: Playlist generation, mood detection, music trends

**Healthcare Dataset - Diabetes**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- **Size**: 768 samples, 8 features
- **Use Case**: Medical diagnosis prediction
- **Skills**: Medical ML, feature importance, interpretability
- **Ethics**: Bias in healthcare, model explainability

### Multi-Modal Datasets

**Fashion MNIST**
- **Source**: [GitHub](https://github.com/zalandoresearch/fashion-mnist)
- **Size**: 70,000 fashion images
- **Use Case**: Image classification, computer vision
- **Skills**: CNN, image processing, fashion AI
- **Access**: `tf.keras.datasets.fashion_mnist.load_data()`

**CIFAR-10**
- **Source**: [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Size**: 60,000 32x32 color images, 10 classes
- **Use Case**: Object recognition, computer vision
- **Skills**: Deep learning, image classification
- **Access**: `tf.keras.datasets.cifar10.load_data()`

**Common Voice Dataset**
- **Source**: [Mozilla](https://commonvoice.mozilla.org/en/datasets)
- **Size**: 19,000+ hours of speech in 100+ languages
- **Use Case**: Speech recognition, voice AI
- **Skills**: Audio processing, sequence modeling
- **Note**: Large download, consider samples

---

## Specialized Collections

### Kaggle Datasets
*Curated collection with competitions and community datasets*

**Kaggle Datasets Homepage**
- **Source**: [Kaggle Datasets](https://www.kaggle.com/datasets)
- **Categories**: Business, health, science, technology, sports
- **Features**: Notebooks, discussions, competitions
- **Skills**: Community learning, competitive ML

**Popular Kaggle Collections**:
- [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) - Advanced regression
- [Pokemon Dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon) - Fun analysis project
- [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) - Gaming industry analysis
- [World Happiness Report](https://www.kaggle.com/datasets/unsdsn/world-happiness) - Social data analysis

### Google Dataset Search
- **Source**: [Google Dataset Search](https://datasetsearch.research.google.com/)
- **Coverage**: Academic, government, organization datasets
- **Use Case**: Research-quality datasets
- **Skills**: Data discovery, academic standards

### Papers With Code Datasets
- **Source**: [Papers With Code](https://paperswithcode.com/datasets)
- **Focus**: State-of-the-art research datasets
- **Use Case**: Cutting-edge ML techniques
- **Skills**: Research replication, advanced methods

---

## Real-Time & API Datasets

### Financial APIs
*Real-time data for dynamic projects*

**Alpha Vantage**
- **Source**: [Alpha Vantage](https://www.alphavantage.co/)
- **Data**: Stock prices, forex, crypto, economic indicators
- **API**: Free tier with 5 calls/minute
- **Skills**: API integration, financial analysis
- **Project Ideas**: Portfolio optimization, trend prediction

**Yahoo Finance (via yfinance)**
- **Source**: [yfinance Python Package](https://pypi.org/project/yfinance/)
- **Data**: Stock data, historical prices, company info
- **Access**: `pip install yfinance`
- **Skills**: Time series analysis, financial modeling
- **Example**: `import yfinance as yf; data = yf.download("AAPL")`

**CoinGecko API**
- **Source**: [CoinGecko](https://www.coingecko.com/en/api)
- **Data**: Cryptocurrency prices and market data
- **API**: Free tier available
- **Skills**: Crypto analysis, volatility modeling

### Social Media APIs

**Twitter API v2**
- **Source**: [Twitter Developer](https://developer.twitter.com/en/docs/twitter-api)
- **Data**: Tweets, user data, trends
- **Access**: Academic Research Track for students
- **Skills**: Social media analysis, NLP, sentiment
- **Note**: Requires application approval

**Reddit API**
- **Source**: [Reddit API](https://www.reddit.com/dev/api/)
- **Data**: Posts, comments, subreddit data
- **Access**: Free with rate limits
- **Skills**: Social network analysis, community detection
- **Package**: `praw` Python package

### Weather and Environment

**OpenWeatherMap**
- **Source**: [OpenWeatherMap](https://openweathermap.org/api)
- **Data**: Current weather, forecasts, historical data
- **API**: Free tier with 1000 calls/day
- **Skills**: Environmental data, forecasting
- **Use Case**: Weather impact on business/behavior

**NASA Open Data**
- **Source**: [NASA Open Data](https://data.nasa.gov/)
- **Data**: Climate, space, earth observation
- **Format**: Various (CSV, JSON, APIs)
- **Skills**: Scientific data analysis, environmental modeling

---

## Image & Computer Vision

### Beginner-Friendly Image Datasets

**MNIST**
- **Source**: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Size**: 70,000 handwritten digits
- **Use Case**: Basic image classification
- **Access**: `tf.keras.datasets.mnist.load_data()`

**Cats vs Dogs**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- **Size**: 25,000 images
- **Use Case**: Binary image classification
- **Skills**: CNN, transfer learning, data augmentation

### Advanced Computer Vision

**COCO Dataset**
- **Source**: [COCO](https://cocodataset.org/)
- **Size**: 330k images, 80 object categories
- **Use Case**: Object detection, segmentation
- **Skills**: Advanced computer vision, multi-task learning
- **Note**: Large dataset, consider subsets

**ImageNet (Sample)**
- **Source**: [ImageNet](https://www.image-net.org/)
- **Size**: 14+ million images, 20k+ categories
- **Use Case**: Large-scale image classification
- **Skills**: Transfer learning, deep networks
- **Access**: Subset available through torchvision

**Flickr8k**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Size**: 8,000 images with captions
- **Use Case**: Image captioning, multimodal learning
- **Skills**: Computer vision + NLP, attention mechanisms

---

## Time Series & Financial Data

### Stock Market Data

**S&P 500 Historical Data**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500)
- **Size**: Historical prices for S&P 500 companies
- **Use Case**: Portfolio analysis, risk modeling
- **Skills**: Time series, financial metrics

**Cryptocurrency Data**
- **Source**: [CoinMarketCap](https://coinmarketcap.com/api/) or [Kaggle](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory)
- **Size**: Historical crypto prices
- **Use Case**: Volatility analysis, price prediction
- **Skills**: High-frequency time series, volatility modeling

### Energy and Utilities

**Household Electric Power Consumption**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
- **Size**: 2M measurements over 4 years
- **Use Case**: Energy consumption forecasting
- **Skills**: Time series forecasting, anomaly detection

**Solar Power Generation**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- **Size**: Solar plant generation data
- **Use Case**: Renewable energy prediction
- **Skills**: Weather impact modeling, forecasting

---

## Government & Research Data

### U.S. Government Data

**Data.gov**
- **Source**: [Data.gov](https://www.data.gov/)
- **Coverage**: Federal, state, local government data
- **Categories**: Health, education, climate, economy
- **Format**: CSV, JSON, APIs
- **Skills**: Public policy analysis, civic data science

**Census Data**
- **Source**: [U.S. Census Bureau](https://www.census.gov/data.html)
- **Data**: Demographics, economics, housing
- **API**: Census API available
- **Skills**: Demographic analysis, social research

**Bureau of Labor Statistics**
- **Source**: [BLS Data](https://www.bls.gov/data/)
- **Data**: Employment, wages, inflation
- **Skills**: Economic analysis, labor market trends

### International Data

**World Bank Open Data**
- **Source**: [World Bank](https://data.worldbank.org/)
- **Data**: Global development indicators
- **Coverage**: 217 countries, 1,600+ indicators
- **Skills**: International development, economic analysis

**Our World in Data**
- **Source**: [Our World in Data](https://ourworldindata.org/)
- **Data**: Global problems, long-term trends
- **Topics**: Health, education, environment, economics
- **Skills**: Global analysis, data visualization

**UN Data**
- **Source**: [UN Data](http://data.un.org/)
- **Data**: International statistics
- **Coverage**: Population, trade, environment
- **Skills**: International relations, global trends

---

## Synthetic & Generated Datasets

### Tool-Generated Datasets

**Scikit-learn Synthetic Datasets**
- **Source**: Built into scikit-learn
- **Types**: Classification, regression, clustering
- **Use Case**: Algorithm testing, controlled experiments
- **Examples**:
  - `make_classification()` - Classification problems
  - `make_regression()` - Regression problems
  - `make_blobs()` - Clustering problems

**Faker Library**
- **Source**: [Faker Documentation](https://faker.readthedocs.io/)
- **Use Case**: Generate realistic fake data
- **Skills**: Data generation, testing pipelines
- **Example**: Customer data, transactions, profiles

### Simulation Datasets

**Retail Simulation**
- **Source**: Create using business rules
- **Use Case**: E-commerce simulation
- **Skills**: Business modeling, synthetic data generation
- **Topics**: Customer journey, inventory management

---

## Dataset Selection Guide by Week

### Week 1-2: Start Here
- **Iris Dataset** - Perfect first dataset
- **Tips Dataset** - Great for visualization
- **Wine Quality** - Good size and features

### Week 3: Classification Focus
- **Titanic** - Classic ML competition
- **Heart Disease** - Medical application
- **Mushroom** - Safety-critical application

### Week 4: Regression & Advanced
- **Boston/California Housing** - Regression practice
- **Bike Sharing** - Time series elements
- **Credit Card Fraud** - Imbalanced data

### Week 5: Text & NLP
- **IMDB Reviews** - Sentiment analysis
- **News Categories** - Text classification
- **Twitter Sentiment** - Social media data

### Week 6: Advanced ML
- **Online Retail** - Customer analytics
- **Adult Census** - Bias and fairness
- **Energy Efficiency** - Engineering applications

### Week 7-8: Capstone Projects
- **E-commerce Dataset** - Business intelligence
- **Airbnb Data** - Geographic and pricing
- **Spotify Tracks** - Recommendation systems
- **Healthcare Data** - Ethical AI considerations

---

## Best Practices for Dataset Usage

### Data Ethics and Legal Considerations
1. **Check Licenses**: Always verify dataset licensing
2. **Privacy**: Be aware of personal information
3. **Bias**: Consider representation and fairness
4. **Attribution**: Properly cite data sources

### Technical Best Practices
1. **Start Small**: Use samples for development
2. **Version Control**: Track dataset versions
3. **Documentation**: Document preprocessing steps
4. **Validation**: Always have test/validation splits

### Project Development
1. **Progressive Complexity**: Start simple, add features
2. **Iterative Approach**: Build minimum viable products first
3. **Real-World Application**: Connect to business problems
4. **Reproducibility**: Ensure others can replicate results

---

## Quick Access Commands

### Loading Common Datasets in Python

```python
# Scikit-learn datasets
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.datasets import fetch_california_housing, make_classification

# Seaborn datasets
import seaborn as sns
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

# TensorFlow/Keras datasets
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Yahoo Finance
import yfinance as yf
stock_data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Kaggle (requires kaggle API setup)
import kaggle
kaggle.api.dataset_download_files('dataset-name', path='data/', unzip=True)
```

This comprehensive list provides datasets suitable for every stage of your internship program, from basic pandas operations to advanced ML projects and ethical AI considerations. Each dataset includes clear use cases, required skills, and practical applications that align with your learning objectives.