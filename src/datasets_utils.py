import pandas as pd

def get_luxury_data(with_prints=False):
    data = pd.read_csv('./datasets/luxury_apparel_data/Luxury_Products_Apparel_Data.csv')
    data = data[['Category', 'SubCategory', 'Description']]
    if with_prints:
        print(f"Num. of obs. where either cat or subcar is None: {data[(data['Category'].isna()) | (data['Category'].isna())].size}")
    data = data[(~data['Category'].isna()) & (~data['Category'].isna())]
    data['category'] = data['Category'] + 'DELIMITER' + data['SubCategory']
    data['description'] = data['Description']
    data = data.drop(['Category', 'SubCategory', 'Description'], axis=1)
    data['category'] = data['category'].str.split('DELIMITER')
    data = data.dropna()
    data['categories_count'] = data['category'].str.len()
    data = data[data['categories_count'] != 1]
    # data = data[(data['description'].str.split().str.len() > 10) & (data['description'].str.split().str.len() <= 130)]
    data = data.reset_index()
    data = data.drop(['index'], axis=1)
    return data[['description', 'category']]

def get_tech_data(with_prints=False):
    def prepare_category(x):
        result = x.split('\n')
        result.pop(0)
        return ''.join(result).split(', ')

    data = pd.read_csv('./datasets/electronics_1/final_data.csv')
    data = data[['Category', 'Description']]
    data = data.rename(columns={ 'Category': 'category', 'Description': 'description' })
    if with_prints:
        print(f"Num. of obs. where cat is None: {data[(data['category'].isna())].size}")
        print(f"Num. of obs. where desc is None: {data[(data['description'].isna())].size}")
    data = data.dropna()
    if with_prints:
        print(f"Num. of obs. where cat is None: {data[(data['category'].isna())].size}")
        print(f"Num. of obs. where desc is None: {data[(data['description'].isna())].size}")
    data['category'] = data['category'].apply(prepare_category)
    data = data.dropna()
    data['categories_count'] = data['category'].str.len()
    data = data[data['categories_count'] != 1]
    # data = data[(data['description'].str.split().str.len() > 10) & (data['description'].str.split().str.len() <= 300)]
    data = data.reset_index()
    data = data.drop(['index'], axis=1)
    return data[['description', 'category']]

def get_retail_data():
    data = pd.read_csv('./datasets/retail_products_classification/train.csv')
    data = data[['categories', 'description']]
    data['categories'] = data['categories'].str.split(',')
    data['categories'] = data['categories'].apply(lambda x: [i.split('&') for i in x])
    data['categories'] = data['categories'].apply(lambda x: [j for i in x for j in i])
    data['categories'] = data['categories'].apply(lambda x: [c.strip() for c in x])
    data = data.rename(columns={ 'categories': 'category' })
    data = data.dropna()
    data['categories_count'] = data['category'].str.len()
    data = data[data['categories_count'] != 1]
    # data = data[(data['description'].str.split().str.len() > 10) & (data['description'].str.split().str.len() <= 300)]
    data = data.reset_index()
    data = data.drop(['index'], axis=1)
    return data[['description', 'category']]

def get_big_basket_data():
    data = pd.read_csv('./datasets/bigbasket_products/BigBasket Products.csv')
    data = data[['category', 'sub_category', 'description']]
    data['category'] = data['category'] + 'DELIMITER' + data['sub_category']
    data = data.drop(['sub_category'], axis=1)
    data['category'] = data['category'].str.split('DELIMITER')
    data['category'] = data['category'].apply(lambda x: [*x[0].split(','), *x[1:]])
    data['category'] = data['category'].apply(lambda x: [c.strip() for c in x])
    data = data.dropna()
    data['categories_count'] = data['category'].str.len()
    data = data[data['categories_count'] != 1]
    # data = data[(data['description'].str.split().str.len() > 10) & (data['description'].str.split().str.len() <= 300)]
    data = data.reset_index()
    data = data.drop(['index'], axis=1)
    return data[['description', 'category']]