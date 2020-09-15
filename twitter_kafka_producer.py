import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import argparse
import json
from datetime import datetime
from kafka import KafkaProducer
from secret import *
from utilis import * 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocessor.api import clean

analyzer = SentimentIntensityAnalyzer()
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])


def initTwitterAPI(api_key, api_secret_key, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    return tweepy.API(auth)

class TweetListener(StreamListener):

    # parse json tweet object stream to get desired data
    def on_data(self, data):
        try:
            json_data = json.loads(data)
            send_data = '{}'
            json_send_data = json.loads(send_data)			

            # make checks for retweet and extended tweet-->done for truncated text
            if "retweeted_status" in json_data:
                try:
                    json_send_data['text'] = (json_data['retweeted_status']['extended_tweet']['full_text'])
                except:
                    json_send_data['text'] = (json_data['retweeted_status']['text'])
            else:
                try:
                    json_send_data['text'] = (json_data['extended_tweet']['full_text'])  
                except:
                    json_send_data['text'] = (json_data['text'])
                    
            json_send_data['creation_datetime'] = json_data['created_at']
            json_send_data['username'] = json_data['user']['name']
            json_send_data['location'] = json_data['user']['location'] 
            json_send_data['userDescr'] = json_data['user']['description']            
            json_send_data['followers'] = json_data['user']['followers_count']
            json_send_data['retweets'] = json_data['retweet_count']            
            json_send_data['favorites'] = json_data['favorite_count']
            json_send_data['sentiment'] = sentimentprediction(json_data['text'])
            if len(json_data["entities"]["hashtags"])>0:
                hashtags=json_data["entities"]["hashtags"][0]["text"].title()
            else:
                hashtags="None"
            json_send_data['hashtag'] = hashtags
            #json_send_data['sentiment'] = sentiments(json_data['text'])[0]
            #json_send_data['negative'] = sentiments(json_data['text'])[1]
            #json_send_data['neutral'] = sentiments(json_data['text'])[2]
            #json_send_data['positive'] = sentiments(json_data['text'])[3]
            

            print(json_send_data)

            # push data to producer
            producer.send("twitter", json.dumps(json_send_data).encode())
            return True
        except KeyError:
            return True
        
    def on_error(self, status):
        print(status)
        return True

words = ['كورونا']
debug = True
topic = "twitter"

if __name__ == "__main__":
	
    # twitter api credentials
	consumer_key = api_key
	consumer_secret = api_secret_key
	access_token = access_token
	access_secret = access_token_secret

	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_secret)
	
	# create AFINN object for sentiment analysis
	#afinn = Afinn(emoticons=True)

    # perform activities on stream
	twitter_stream = Stream(auth, TweetListener(), tweet_mode='extended')
	twitter_stream.filter(track=words)