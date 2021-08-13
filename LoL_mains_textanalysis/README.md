# League Of Legends subreddit text analytics 

A personal for fun project involving the video game, League of Legends, and their respective champions. 

## Method
Each champion in the game has their own dedicated subreddit, without exception

We gather all the comments from the top 50 posts of each subreddit, and do some basic NLP, clustering, and dimensionality reduction

## Results 

The results are shockingly good.  For data that is composed of large amounts of raw text, we see clustering results that largely make sense.  Champions are placed in clusters with similiar roles, and even between clusters the distance between similiar champions makes much sense to a veteran of the game.  Example - ChoGath, Kassadin, and Akali end up very close in the final plot, which are all melee mages. 

We also create a Network of champion mentions across subreddits (Subreddit_Network.ipynb) and see which champions are talking about eachother the most.  While I will not go into detail here, I can say that similiar roles and character talk about eachother, and the stronger connections seen once again make a large amount of sense to a veteran of the game (in this case, Riven and Yasuo mains love to talk about eachother).