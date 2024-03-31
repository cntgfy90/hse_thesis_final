import pandas as pd

def get_child_categories_count_by_parent_categories(data):
    parent_categories = list(set([x[0] for x in data['category']]))

    parent_categories_with_child_categories_count = dict({ x: [] for x in parent_categories })
    parent_unique_categories_hash = dict({ x: x for x in parent_categories })

    for categories in data['category']:
        if parent_unique_categories_hash[categories[0]]:
            parent_categories_with_child_categories_count[categories[0]] += categories[1:]

    parent_categories_with_counts_df = pd.DataFrame({
        'parent_category': parent_categories_with_child_categories_count.keys(),
        'count': parent_categories_with_child_categories_count.values()
    })

    parent_categories_with_counts_df['count'] = parent_categories_with_counts_df['count'].apply(lambda x: len(set(x)))

    return parent_categories_with_counts_df