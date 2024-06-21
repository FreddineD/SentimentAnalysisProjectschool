import tweepy
import pandas as pd
from time import sleep

# Votre jeton d'accès (bearer token)
bearer_token = '******************************************************************************************************************'

# Initialisation du client Tweepy pour l'API v2
client = tweepy.Client(bearer_token=bearer_token)

# Définir la requête de recherche
query = '"restaurant delivery, food, uber eats, takeaway, delivrero, fast food" lang:en'
max_results = 100
result = []

def get_tweets(query, max_results, num_tweets):
    tweets_stored = 0
    next_token = None
    
    while tweets_stored < num_tweets:
       
        response = client.search_all_tweets(
            query=query,
            tweet_fields=['author_id', 'created_at', 'text'],
            user_fields=['username', 'location'],
            expansions=['author_id'],
            max_results=max_results,
            next_token=next_token
        )

        if response.meta['result_count'] == 0:
            break

        # Traiter les utilisateurs
        for user in response.includes['users']:
            user_dict[user['id']] = {
                'username': user['username'],
                'location': user.location if user.location else 'N/A'
            }

        # Traiter les tweets
        for tweet in response.data:
            author_info = user_dict[tweet.author_id]
            result.append({
                'User_ID': tweet.author_id,
                'User_Name': author_info['username'],
                'User_Location': author_info['location'],
                'Tweet_Content': tweet.text,
                'Tweet_URL': f"https://twitter.com/{author_info['username']}/status/{tweet.id}"
            })

            tweets_stored += 1

        try:
            next_token = response.meta['next_token']
        except KeyError:
            print("No More Entries")
            break

        
        sleep(5)

    return result

# Initialiser le dictionnaire des utilisateurs
user_dict = {}

# Récupérer les tweets
get_tweets(query, max_results, 670)

# Convertir les résultats en DataFrame pandas
df = pd.DataFrame(result)

# Exporter les résultats en fichier CSV
df.to_csv("Output.csv", index=False)
# SentimentAnalysisProjectschool
