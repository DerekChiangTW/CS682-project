import json
import csv
from collections import defaultdict
import random



def parse_restaurant_ids(num_of_restaurants=2500):
    """
    This method parses the restaurants ids from yelp dataset
    :param num_of_restaurants: number of restaurants
    :return: list of restaurants'  ids
    """
    restaurants = []
    with open('./dataset/business.json', 'r') as file:
        counter = 0
        for line in file:
            if counter >= num_of_restaurants:
                break
            temp = json.loads(line)
            b_id = temp['business_id']
            if 'RestaurantsTakeOut' in temp['attributes']:
                restaurants.append(b_id)
                counter += 1
    print("***Finished Getting Restaurant***")
    file.close()
    return restaurants


def parse_review_big(restaurant_ids, total_reviews_limit=50000):
    """
    Retrieve reviews  from the dataset  that correspond to the restaurant ids.
    :param restaurant_ids:  array of restauratn ids
    :param total_reviews_limit:   total number of data
    :return:  reviews that matches the star rating distribution
    """
    with open('./dataset/review.json', 'r', encoding='utf-8') as file:
        with open('./dataset/parsed_review.csv', 'w', encoding='utf-8', newline='') as csvfile:
            fieldnames = ['id', 'stars', 'text']
            star_stat = [0] * 5
            star_limit = [total_reviews_limit * x for x in [0.1, 0.25, 0.3, 0.25, 0.1]]
            print(star_limit)

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            all_review = []
            for line in file:  # store the reviews and shuffle
                all_review.append(json.loads(line))
            print("***Finished Reading Reviews***")

            random.shuffle(all_review)
            for review in all_review:  # start writing reviews to csv while tracking the star stat
                if sum(star_stat) >= total_reviews_limit:
                    break
                if review['business_id'] in restaurant_ids and star_stat[int(review['stars']) - 1] < (
                        star_limit[int(review['stars']) - 1]):
                    writer.writerow(
                        {'id': review['business_id'], 'stars': review['stars'], 'text': review['text']})
                    star_stat[int(review['stars']) - 1] += 1
                    print("Reviews Obtained: ", sum(star_stat))
            print(star_stat)


"""  Don't need anymore
def parse_review(restaurants):
    with open('./dataset/review.json', 'r') as f:
        with open('./dataset/review.csv', 'w') as csvfile:
            counter = 0
            store_counter = 0
            fieldnames = ['id', 'stars', 'text', 'labels']
            b_dict = defaultdict(float)
            star_dict = defaultdict(float)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for l in f:
                if counter >= 450:
                    break
                temp = json.loads(l)
                b_id = temp['business_id']
                if b_id in restaurants:
                    if store_counter < 50:
                        if star_dict[temp['stars']] < 90:
                            writer.writerow(
                                {'id': b_id, 'stars': temp['stars'], 'text': temp['text'], 'labels': ''})
                            counter += 1
                            store_counter += 1
                            star_dict[temp['stars']] += 1
                    else:
                        store_counter = 0
                    if b_dict[b_id] > 50:
                        restaurants.remove(b_id)
                        store_counter = 0
"""

if __name__ == '__main__':
    restaurants = parse_restaurant_ids()
    # parse_review(set(restaurants))   # don't need anymore
    parse_review_big(restaurants)
