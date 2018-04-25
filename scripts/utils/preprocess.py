#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json


def extract_restaurant(num_of_restaurants=3000):
    """ Extract the top k restaurant ids from the dataset.

    @Parameters
    -----------
    num_of_restaurants : int
        Number of extracted restaurants.

    @Returns
    --------
    ids : list
        The ids of the extracted restaurants.

    """
    ids = set()
    with open("./../../data/yelp/business.json", "r") as f:
        for line in f:
            if len(ids) == num_of_restaurants:
                break
            temp = json.loads(line)
            if "Restaurants" in temp["categories"]:
                ids.add(temp['business_id'])
    return list(ids)


def extract_review(restaurant_ids, total_reviews_limit=6500):
    """ Retrieve reviews from the dataset that corresponds to the restaurant ids.

    @Parameters
    -----------
    restaurant_ids : list
        The ids of the extracted restaurants.

    total_reviews_limit : int
        The total number of reviews used as our dataset.

    """
    dir_path = './../../data/yelp/'
    with open(dir_path + 'review.json', 'r', encoding='utf-8') as f:
        # Set the number of reviews we want to extract for each number of star
        dist = [0.1, 0.25, 0.3, 0.25, 0.1]
        star_limit = {i + 1: int(total_reviews_limit * d) for i, d in enumerate(dist)}

        # Keep track of the number of extracted reviews for each number of star
        num_review = {i + 1: 0 for i in range(5)}
        extracted_review = {i + 1: [] for i in range(5)}

        for line in f:
            review = json.loads(line)
            b_id, stars, text = review['business_id'], int(review['stars']), review['text']
            if sum(num_review.values()) == total_reviews_limit:
                break
            if b_id in restaurant_ids and num_review[stars] < star_limit[stars]:
                content = {'business_id': b_id, 'stars': stars, 'text': text}
                extracted_review[stars].append(content)
                num_review[stars] += 1
        # Partition the dataset into train, validate and test set
        partition_dist = [0.6, 0.2, 0.2]

        train_review, validate_review, test_review = [], [], []
        for i in range(1, 6):
            train_end = int(partition_dist[0] * star_limit[i])
            validate_end = train_end + int(partition_dist[1] * star_limit[i])
            train_review += extracted_review[i][:train_end]
            validate_review += extracted_review[i][train_end:validate_end]
            test_review += extracted_review[i][validate_end:]

        fields = ['business_id', 'stars', 'text']
        # Save the training set
        with open(dir_path + 'small_train_review.csv', 'w', encoding='utf-8') as trainfile:
            writer = csv.DictWriter(trainfile, fieldnames=fields)
            writer.writeheader()
            for rev in train_review:
                writer.writerow(rev)
        trainfile.close()
        print("Successfully saved the training set.")

        # Save the validation set
        with open(dir_path + 'small_validate_review.csv', 'w', encoding='utf-8') as validatefile:
            writer = csv.DictWriter(validatefile, fieldnames=fields)
            writer.writeheader()
            for rev in validate_review:
                writer.writerow(rev)
        validatefile.close()
        print("Successfully saved the validation set.")

        # Save the test set
        with open(dir_path + 'small_test_review.csv', 'w', encoding='utf-8') as testfile:
            writer = csv.DictWriter(testfile, fieldnames=fields)
            writer.writeheader()
            for rev in test_review:
                writer.writerow(rev)
        testfile.close()
        print("Successfully saved the test set.")
    f.close()


if __name__ == "__main__":
    ids = extract_restaurant()
    extract_review(ids)

    # Check the dataset
    # with open('./../../data/yelp/test_review.csv', 'r', encoding='utf-8') as f:
    #     reader = csv.DictReader(f, fieldnames=['business_id', 'stars', 'text'])
    #     next(reader, None)
    #     stats = [0 for i in range(5)]
    #     for row in reader:
    #         stats[int(row['stars']) - 1] += 1
    #     print(stats)
