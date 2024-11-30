import json
import pandas as pd
from datetime import datetime, timedelta

def create_data(dump_dir="data"):
    # Generate 50 products with more details
    categories = ["Gaming Peripherals", "Monitors", "Laptops", "Components", "Storage", "Audio", "Networking"]
    brands = ["Razer", "Logitech", "Corsair", "ASUS", "Dell", "Samsung", "HyperX", "MSI"]

    products = []
    for i in range(50):
        base_price = round(float(50 + (i * 43.75)), 2)  # Varied base prices
        discount_percent = round(float(5 + (i % 20)), 2)  # Discounts between 5-25%
        discounted_price = round(base_price * (1 - discount_percent/100), 2)
        
        product = {
            "id": 100001 + i,
            "product_name": f"Product-{100001 + i}",
            "brand": brands[i % len(brands)],
            "price": base_price,
            "discount_percent": discount_percent,
            "discounted_price": discounted_price,
            "in_stock": bool(i % 3),  # Mix of True/False
            "stock_quantity": i % 3 * 25,  # 0, 25, or 50 items
            "category": categories[i % len(categories)],
            "sub_category": f"Sub-{categories[i % len(categories)]}",
            "description": f"High-quality {categories[i % len(categories)]} product with premium features",
            "specifications": {
                "weight": f"{round(0.5 + (i/10), 2)}kg",
                "dimensions": f"{20+i}x{15+i}x{5+i}cm",
                "color": ["Black", "White", "Gray"][i % 3]
            },
            "ratings": round(3.5 + (i % 15) / 10, 1),  # Ratings between 3.5-5.0
            "reviews_count": (i + 1) * 13,
            "warranty_months": [12, 24, 36][i % 3],
            "added_date": (datetime.now() - timedelta(days=i*5)).strftime("%Y-%m-%d"),
            "tags": [
                categories[i % len(categories)],
                brands[i % len(brands)],
                ["Premium", "Budget", "Mid-range"][i % 3]
            ]
        }
        products.append(product)

    ecommerce_data = {"products": products}

    # Save JSON to file
    with open(f'{dump_dir}/ecommerce_data.json', 'w') as f:
        json.dump(ecommerce_data, f, indent=4)

def save_into_excel(data, dump_dir="data") -> None: 
    df = pd.DataFrame(data)
    # Convert specifications from dict to string for Excel compatibility
    df['specifications'] = df['specifications'].apply(str)
    df['tags'] = df['tags'].apply(str)
    df.to_excel(f'{dump_dir}/ecommerce_data.xlsx', index=False)
    print("Excel file created successfully!")

def read_file_content(path):
# Read JSON file
    with open('ecommerce_data.json', 'r') as f:
        return json.load(f)
