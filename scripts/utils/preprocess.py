#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json


def extract_restaurant(num_of_restaurants=2500):
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
    with open("./../../datasets/yelp/business.json", "r") as f:
        for line in f:
            if len(ids) == num_of_restaurants:
                break
            temp = json.loads(line)
            if "Restaurants" in temp["categories"]:
                ids.add(temp['business_id'])
    return list(ids)


def extract_review(restaurant_ids, total_reviews_limit=50000):
    """ Retrieve reviews from the dataset that corresponds to the restaurant ids.

    @Parameters
    -----------
    restaurant_ids : list
        The ids of the extracted restaurants.

    total_reviews_limit : int
        The total number of reviews used as our dataset.

    """
    dir_path = './../../datasets/yelp/'
    with open(dir_path + 'review.json', 'r', encoding='utf-8') as f:
        with open(dir_path + 'parsed_review.json', 'w', encoding='utf-8') as outfile:
            # Set the number of reviews we want to extract for each number of star
            dist = [0.1, 0.25, 0.3, 0.25, 0.1]
            star_limit = {i + 1: int(total_reviews_limit * d) for i, d in enumerate(dist)}

            # Keep track of the number of extracted reviews for each number of star
            extracted_reviews = {}
            num_review = {i + 1: 0 for i in range(5)}

            for line in f:
                review = json.loads(line)
                b_id, star, text = review['business_id'], int(review['stars']), review['text']
                if sum(num_review.values()) == total_reviews_limit:
                    break
                if review['business_id'] in restaurant_ids and num_review[star] < star_limit[star]:
                    content = {'business_id': b_id, 'star': star, 'text': text}
                    extracted_reviews[review['review_id']] = content
                    num_review[star] += 1
            json.dump(extracted_reviews, outfile, indent=4, sort_keys=True, separators=(',', ':'))
            print("Successfully saved the json file.")
        outfile.close()
    f.close()


if __name__ == "__main__":
    ids = extract_restaurant()
    # extract_review(ids)
    review_dict = json.load(open('./../../datasets/yelp/parsed_review.json', 'r'))
    # print(len(set([r['business_id'] for r in review_dict.values()])))
