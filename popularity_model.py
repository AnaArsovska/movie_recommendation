from ratings_data import *

#makes popularity model using training data
popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

#check recommended movies for a sample 5 users
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)

#check whether the recommended movies are the highest rated ones
ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)
