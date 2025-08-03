import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("online_retail.csv")  
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Set style
sns.set(style="whitegrid")

# 1. Transaction Volume by Country
country_sales = df.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
country_sales.plot(kind='bar', color='skyblue')
plt.title('Transaction Volume by Country')
plt.ylabel('Number of Transactions')
plt.xlabel('Country')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("eda_country_volume.png")
plt.show()

# 2. Top-Selling Products

top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 10 Selling Products')
plt.xlabel('Quantity Sold')
plt.ylabel('Product')
plt.tight_layout()
plt.savefig("eda_top_products.png")
plt.show()

# 3. Purchase Trends Over Time
df['Date'] = df['InvoiceDate'].dt.date
daily_orders = df.groupby('Date')['InvoiceNo'].nunique()

plt.figure(figsize=(14, 6))
daily_orders.plot(color='purple')
plt.title('Daily Transactions Over Time')
plt.ylabel('Number of Transactions')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()
plt.savefig("eda_purchase_trend.png")
plt.show()

# 4. Transaction Value Distribution
transaction_value = df.groupby(['InvoiceNo'])['TotalPrice'].sum()

plt.figure(figsize=(10, 5))
sns.histplot(transaction_value, bins=50, kde=True, color='coral')
plt.title('Distribution of Transaction Values')
plt.xlabel('Transaction Value')
plt.tight_layout()
plt.savefig("eda_transaction_value.png")
plt.show()

# 5. Customer Monetary Value
customer_value = df.groupby(['CustomerID'])['TotalPrice'].sum()

plt.figure(figsize=(10, 5))
sns.histplot(customer_value, bins=50, kde=True, color='green')
plt.title('Distribution of Customer Lifetime Value')
plt.xlabel('Total Spend per Customer')
plt.tight_layout()
plt.savefig("eda_customer_value.png")
plt.show()

# 6. RFM Histograms
try:
    rfm = pd.read_csv("rfm.csv")

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(rfm['Recency'], kde=True, bins=30, color='skyblue')
    plt.title('Recency Distribution')

    plt.subplot(1, 3, 2)
    sns.histplot(rfm['Frequency'], kde=True, bins=30, color='orange')
    plt.title('Frequency Distribution')

    plt.subplot(1, 3, 3)
    sns.histplot(rfm['Monetary'], kde=True, bins=30, color='green')
    plt.title('Monetary Distribution')

    plt.tight_layout()
    plt.savefig("eda_rfm_histograms.png")
    plt.show()
except Exception as e:
    print("RFM file not found or incomplete:", e)
