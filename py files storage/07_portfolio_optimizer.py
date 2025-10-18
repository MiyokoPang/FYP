# ========================================
# CELL 1: Portfolio Optimizer - Select Top 20 Stocks
# ========================================
import pandas as pd
import numpy as np
import mysql.connector
import pickle
import os

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'trading_system'
}

def get_stock_performance():
    """Get average directional accuracy per stock from model_performance table"""
    conn = mysql.connector.connect(**db_config)
    
    query = """
    SELECT 
        symbol,
        AVG(directional_accuracy) as avg_dir_acc,
        AVG(rmse) as avg_rmse,
        COUNT(*) as model_count
    FROM model_performance
    WHERE model_type LIKE '%Enhanced%'
    GROUP BY symbol
    ORDER BY avg_dir_acc DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

print("Analyzing stock performance from enhanced models...")
print("="*70)

performance_df = get_stock_performance()

print("\nAll Stocks Ranked by Directional Accuracy:")
print("-"*70)
print(f"{'Rank':<6} {'Symbol':<8} {'Avg Dir Acc':<15} {'Avg RMSE':<12} {'Models'}")
print("-"*70)

for idx, row in performance_df.iterrows():
    rank = idx + 1
    symbol = row['symbol']
    dir_acc = row['avg_dir_acc']
    rmse = row['avg_rmse']
    models = int(row['model_count'])
    
    # Color coding
    status = "🟢" if dir_acc > 55 else "🟡" if dir_acc > 50 else "🔴"
    
    print(f"{rank:<6} {symbol:<8} {dir_acc:>6.2f}%  {status:<6} {rmse:>6.2f}%  {models:>6}")

# Select top 20 and bottom 4
top_20_stocks = performance_df.head(20)['symbol'].tolist()
bottom_4_stocks = performance_df.tail(4)['symbol'].tolist()

print("\n" + "="*70)
print("PORTFOLIO SELECTION:")
print("="*70)
print(f"\n✅ TOP 20 STOCKS (Keeping):")
print(", ".join(top_20_stocks))

print(f"\n❌ BOTTOM 4 STOCKS (Removing):")
print(", ".join(bottom_4_stocks))

# Statistics
top_20_avg_acc = performance_df.head(20)['avg_dir_acc'].mean()
bottom_4_avg_acc = performance_df.tail(4)['avg_dir_acc'].mean()

print(f"\nTop 20 Average Directional Accuracy: {top_20_avg_acc:.2f}%")
print(f"Bottom 4 Average Directional Accuracy: {bottom_4_avg_acc:.2f}%")
print(f"Improvement by removing bottom 4: +{top_20_avg_acc - performance_df['avg_dir_acc'].mean():.2f}%")


# ========================================
# CELL 2: Save Portfolio Configuration
# ========================================
import json
from datetime import datetime

portfolio_config = {
    'selected_stocks': top_20_stocks,
    'excluded_stocks': bottom_4_stocks,
    'selection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'selection_criteria': 'Top 20 by average directional accuracy from enhanced models',
    'avg_dir_acc': float(top_20_avg_acc),
    'total_stocks_evaluated': len(performance_df)
}

# Save to file
os.makedirs('data', exist_ok=True)

with open('data/portfolio_config.json', 'w') as f:
    json.dump(portfolio_config, f, indent=4)

# Also save as pickle for easy loading
with open('data/portfolio_stocks.pkl', 'wb') as f:
    pickle.dump(top_20_stocks, f)

print("\n✓ Portfolio configuration saved to:")
print("  - data/portfolio_config.json")
print("  - data/portfolio_stocks.pkl")


# ========================================
# CELL 3: Update Stock List in Database
# ========================================
def mark_excluded_stocks():
    """Add a flag to stocks table for excluded stocks"""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    # Add column if it doesn't exist
    try:
        cursor.execute("""
            ALTER TABLE stocks 
            ADD COLUMN is_active TINYINT(1) DEFAULT 1
        """)
        conn.commit()
    except:
        pass  # Column already exists
    
    # Mark bottom 4 as inactive
    for symbol in bottom_4_stocks:
        cursor.execute("""
            UPDATE stocks 
            SET is_active = 0 
            WHERE symbol = %s
        """, (symbol,))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n✓ Database updated - bottom 4 stocks marked as inactive")

mark_excluded_stocks()


# ========================================
# CELL 4: Filter Enhanced Stock Data
# ========================================
print("\nFiltering enhanced_stock_data to top 20 stocks...")

# Load existing enhanced data
try:
    with open('data/enhanced_stock_data.pkl', 'rb') as f:
        full_enhanced_data = pickle.load(f)
    
    # Filter to top 20
    filtered_enhanced_data = {
        symbol: data 
        for symbol, data in full_enhanced_data.items() 
        if symbol in top_20_stocks
    }
    
    # Save filtered version
    with open('data/enhanced_stock_data_top20.pkl', 'wb') as f:
        pickle.dump(filtered_enhanced_data, f)
    
    print(f"✓ Filtered from {len(full_enhanced_data)} to {len(filtered_enhanced_data)} stocks")
    print(f"✓ Saved to: data/enhanced_stock_data_top20.pkl")
    
    # Show what we have
    print("\nFiltered Stock Data Contents:")
    for symbol in filtered_enhanced_data.keys():
        data = filtered_enhanced_data[symbol]
        print(f"  {symbol}: {len(data['X_train'])} train samples, "
              f"{len(data['X_test'])} test samples, "
              f"{'with sentiment' if data.get('has_sentiment') else 'no sentiment'}")
    
except FileNotFoundError:
    print("⚠️  enhanced_stock_data.pkl not found in data/ folder")
    print("   Make sure to move it from root directory first")


# ========================================
# CELL 5: Generate Updated Stock List for Collection
# ========================================
print("\n" + "="*70)
print("UPDATED STOCK LIST FOR DATA COLLECTION")
print("="*70)

# Get full stock details from database
conn = mysql.connector.connect(**db_config)
query = """
SELECT symbol, company_name, sector 
FROM stocks 
WHERE symbol IN ({})
ORDER BY symbol
""".format(','.join(['%s']*len(top_20_stocks)))

top_20_details = pd.read_sql(query, conn, params=top_20_stocks)
conn.close()

print("\nCopy this into your 01_data_collection.ipynb:")
print("-"*70)
print("stocks_to_track = [")
for _, row in top_20_details.iterrows():
    print(f"    ('{row['symbol']}', '{row['company_name']}', '{row['sector']}'),")
print("]")

print("\n✓ Portfolio optimization complete!")
print(f"✓ Training will now use only these {len(top_20_stocks)} high-performing stocks")