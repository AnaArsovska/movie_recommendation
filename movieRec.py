import pandas as pd 
import graphlab

#user data
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zipcode']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

#film rating data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

#movies data
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

#training and test data
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

#converting data to sframe objects
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)

#makes popularity model using training data
popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

#check recommended movies for a sample 5 users
#popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
#popularity_recomm.print_rows(num_rows=25)

#check whether the recommended movies are the highest rated ones
ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)
