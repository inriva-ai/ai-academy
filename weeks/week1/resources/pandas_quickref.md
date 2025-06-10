# Pandas Quick Reference

## Data Loading and Basic Info

```python
import pandas as pd

# Load data
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')

# Basic info
df.info()                    # Data types and memory usage
df.describe()                # Statistical summary
df.shape                     # (rows, columns)
df.head(n=5)                # First n rows
df.tail(n=5)                # Last n rows
```

## Data Exploration

```python
# Missing values
df.isnull().sum()            # Count nulls per column
df.isnull().sum().sum()      # Total nulls

# Unique values
df['column'].nunique()       # Number of unique values
df['column'].value_counts()  # Count of each value

# Data types
df.dtypes                    # Column data types
df.select_dtypes(include=[np.number])  # Numeric columns only
```

## Data Filtering and Selection

```python
# Select columns
df['column']                 # Single column
df[['col1', 'col2']]        # Multiple columns

# Filter rows
df[df['column'] > 5]         # Boolean filtering
df[df['col'].isin(['A', 'B'])]  # Filter by list
df.query('column > 5 and other_col == "A"')  # Query syntax

# Locate data
df.loc[row_indexer, column_indexer]  # Label-based
df.iloc[row_indexer, column_indexer] # Position-based
```

## Data Transformation

```python
# Apply functions
df['new_col'] = df['col'].apply(lambda x: x * 2)
df.apply(function, axis=0)   # Apply to columns
df.apply(function, axis=1)   # Apply to rows

# String operations
df['col'].str.upper()        # Uppercase
df['col'].str.contains('pattern')  # Contains pattern
df['col'].str.split('delimiter')   # Split strings

# Date operations
df['date'] = pd.to_datetime(df['date_col'])
df['year'] = df['date'].dt.year
```

## Grouping and Aggregation

```python
# Group by
df.groupby('column').mean()          # Group and aggregate
df.groupby(['col1', 'col2']).sum()   # Multiple columns
df.groupby('col').agg({'col2': 'mean', 'col3': 'sum'})  # Custom agg

# Pivot tables
pd.pivot_table(df, values='value', index='row', columns='col')
```

## Data Cleaning

```python
# Handle missing values
df.dropna()                  # Drop rows with any NaN
df.fillna(value)            # Fill NaN with value
df.fillna(df.mean())        # Fill with column means

# Remove duplicates
df.drop_duplicates()         # Remove duplicate rows
df.drop_duplicates(subset=['col'])  # Based on specific columns

# Data type conversion
df['col'] = df['col'].astype('int64')
pd.to_numeric(df['col'], errors='coerce')  # Convert to numeric
```

## Merging and Joining

```python
# Concatenate
pd.concat([df1, df2])        # Stack vertically
pd.concat([df1, df2], axis=1)  # Stack horizontally

# Merge
pd.merge(df1, df2, on='key')           # Inner join
pd.merge(df1, df2, how='left')         # Left join
pd.merge(df1, df2, left_on='a', right_on='b')  # Different key names
```

## Quick Statistics

```python
# Correlation
df.corr()                    # Correlation matrix
df['col1'].corr(df['col2'])  # Correlation between two columns

# Statistical functions
df.mean()                    # Column means
df.std()                     # Standard deviation
df.quantile([0.25, 0.5, 0.75])  # Quartiles
```

## Performance Tips

1. **Use vectorized operations instead of loops**
2. **Use `.loc` and `.iloc` for indexing**
3. **Specify dtypes when reading files**
4. **Use `pd.cut()` for binning**
5. **Use `pd.factorize()` for categorical encoding**