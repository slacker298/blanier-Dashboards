#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
Description:
    1. Scrape all comments from a given reddit thread
    2. Extract top level comments
    3. Save to a csv file

Author:
    Copyright (c) Ian Hussey 2016 (ian.hussey@ugent.be) 
    Released under the GPLv3+ license.

Known issues:
    None. 

Notes:
    1. Although the script only uses publiclly available information, 
    PRAW's call to the reddit API requires a reddit login (see line 47).
    2. Reddit API limits number of calls (1 per second IIRC). 
    For a large thread (e.g., 1000s of comments) script execution time may therefore be c.1 hour.
    3. Because of this bottleneck, the entire data object is written to a pickle before anything is discarded. 
    This speeds up testing etc.
    4. Does not extract comment creation date (or other properties), which might be useful. 
"""

# Dependencies
import praw
import csv
import os
import sys
import pickle
import Scraper_config as cfg

# Set encoding to utf-8 rather than ascii, as is default for python 2.
# This avoids ascii errors on csv write.
reload(sys)
sys.setdefaultencoding('utf-8') 

# Change directory to that of the current script
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

# Acquire comments via reddit API
r = praw.Reddit('Comment Scraper 1.0 by u/_Daimon_ see '
    'https://praw.readthedocs.org/en/latest/'
    'pages/comment_parsing.html')
r.login(cfg.username, cfg.password, disable_warning=True)

# override this in config to decide which attributes to save from a comment object
def default_comment_to_list(comment):
    return [comment.body]

if hasattr(cfg, "comment_to_list"):
    comment_to_list = cfg.comment_to_list
else:
    comment_to_list = default_comment_to_list

def get_submission_comments(uniq_id):
    submission = r.get_submission(submission_id=uniq_id)  # UNIQUE ID FOR THE THREAD GOES HERE - GET FROM THE URL
    submission.replace_more_comments(limit=None, threshold=0)  # all comments, not just first page

    # Save object to pickle
    output = open(cfg.output_file, 'wb')
    pickle.dump(submission, output, -1)
    output.close()

    ## Load object from pickle
    # pkl_file = open(cfg.output_file, 'rb')
    # submission = pickle.load(pkl_file)
    ##pprint.pprint(submission)
    # pkl_file.close()

    # Extract first level comments only
    forest_comments = submission.comments  # get comments tree
    already_done = set()
    top_level_comments = []
    for comment in forest_comments:
        if not hasattr(comment, 'body'):  # only comments with body text
            continue
        if comment.is_root:  # only first level comments
            if comment.id not in already_done:
                already_done.add(comment.id)  # add it to the list of checked comments
                top_level_comments.append(comment_to_list(comment))  # append to list for saving
                # print(comment.body)
    return top_level_comments

def get_subreddit_comments(uniq_id):
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    comStream = praw.helpers.comment_stream(r, uniq_id[3:], limit=limit) # Get the comment string
    comments = map(lambda _: [next(comStream).__str__()], range(limit)) # Get the raw string of each comment obj
    return list(comments) # Convert to list if running on Python3

uniq_id = cfg.uniq_id
if len(sys.argv) > 1:
    uniq_id = sys.argv[1]

if uniq_id[:3] == '/r/':
    top_level_comments = get_subreddit_comments(uniq_id)
else:
    top_level_comments = get_submission_comments(uniq_id)

# Save comments to disk
with open(cfg.output_csv_file, "wb") as output:
    writer = csv.writer(output)
    writer.writerows(top_level_comments)

