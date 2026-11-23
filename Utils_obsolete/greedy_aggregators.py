import random

def min_surplus_items_to_balance_with_details(ellipsis_items, surplus_items):
    total_ellipsis = -sum(item['value'] for item in ellipsis_items)  # Total ellipsis to be balanced
    sorted_surplus_items = sorted(surplus_items, key=lambda x: x['value'], reverse=True)  # Sort surplus in descending order

    ellipsis_queue = sorted(ellipsis_items, key=lambda x: abs(x['value']), reverse=True)  # Sort ellipses by magnitude
    surplus_used = []
    ellipsis_associations = {}

    cumulative_surplus = 0
    num_surplus_items = 0

    for surplus in sorted_surplus_items:
        if not ellipsis_queue:
            break
        
        surplus_id = surplus['id']
        surplus_value = surplus['value']
        ellipsis_associations[surplus_id] = []
        
        while ellipsis_queue and surplus_value > 0:
            current_ellipsis = ellipsis_queue.pop(0)
            current_ellipsis_value = abs(current_ellipsis['value'])
            if surplus_value >= current_ellipsis_value:
                surplus_value -= current_ellipsis_value
                cumulative_surplus += current_ellipsis_value
                ellipsis_associations[surplus_id].append(current_ellipsis['id'])
            else:
                ellipsis_queue.insert(0, current_ellipsis)
                break
        
        if ellipsis_associations[surplus_id]:
            surplus_used.append(surplus_id)
            num_surplus_items += 1

        if cumulative_surplus >= total_ellipsis:
            break

    if cumulative_surplus < total_ellipsis:
        return -1, [], {}

    return num_surplus_items, surplus_used, ellipsis_associations

def main(items):
    ellipsis_items = [{'id': i+1, 'value': item} for i, item in enumerate(items) if item < 0]
    surplus_items = [{'id': i+1, 'value': item} for i, item in enumerate(items) if item > 0]

    if not ellipsis_items:
        return 0, [], {}  # No ellipses to balance

    if not surplus_items:
        return -1, [], {}  # No surplus items to balance the ellipses

    return min_surplus_items_to_balance_with_details(ellipsis_items, surplus_items)

# Generate 100 random items with values ranging from -100 to 100
random.seed(42)  # For reproducibility
items = [random.randint(-30, 100) for _ in range(100)]

# Print the list of generated items and their corresponding values
print("Generated items and their values:")
for i, value in enumerate(items):
    print(f"Item {i+1}: {value}")

# Process the items to find the minimum number of surplus items needed
result, surplus_used, associations = main(items)

# Print the results
print(f"\nMinimum number of surplus items needed: {result}")
print(f"Surplus items used (by ID): {surplus_used}")
print("Ellipsis associations (surplus ID: ellipsis IDs):")
for surplus_id, ellipsis_ids in associations.items():
    print(f"  Item {surplus_id}: {', '.join(map(str, ellipsis_ids))}")
