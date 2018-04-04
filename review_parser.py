import json
import csv
from collections import defaultdict
import random


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


def parse_review_big(restaurants):
    with open('./dataset/review.json', 'r', encoding='utf-8') as file:
        with open('./dataset/review_big.csv', 'w', encoding='utf-8', newline='') as csvfile:
            total_data = 50000
            fieldnames = ['id', 'stars', 'text']
            star_stat = [0] * 5

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            all_review = []

            for line in file:  # store the reviews and shuffle
                all_review.append(json.loads(line))
            print("***Finished Reading Reviews***")
            random.shuffle(all_review)

            for review in all_review:  # start writing reviews to csv while tracking the star stat
                if sum(star_stat) >= total_data:
                    break
                if review['business_id'] in restaurants and star_stat[int(review['stars'])-1] < (total_data/5.0):
                    writer.writerow(
                        {'id': review['business_id'], 'stars': review['stars'], 'text': review['text']})
                    star_stat[int(review['stars'])-1] += 1
                    print("Reviews Obtained: ", sum(star_stat))
            print(star_stat)


def parse_restaurants():
    restaurants = []
    with open('./dataset/business.json', 'r') as f:
        counter = 0
        for l in f:
            if counter >= 2000:
                break
            temp = json.loads(l)
            b_id = temp['business_id']
            if 'RestaurantsTakeOut' in temp['attributes']:
                restaurants.append(b_id)
                counter += 1
    print("***Finished Getting Restaurant***")
    return restaurants


if __name__ == '__main__':
    restaurants = parse_restaurants()
    # parse_review(set(restaurants))
    parse_review_big(restaurants)
