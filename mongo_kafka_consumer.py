from kafka import KafkaConsumer
from pymongo import MongoClient
import json

# connect to database
try:
   client = MongoClient(host="localhost",
                     port=27017, 
                     username="root", 
                     password="example",
                    authSource="admin")
   print(client.list_database_names())
   db = client.twitter
   print("Connected successfully !")
except:  
   print("Could not connect to MongoDB")
   
# create kafka consumer
consumer = KafkaConsumer('twitter', bootstrap_servers=['localhost:9092'])

# parse json twitter objects  
for msg in consumer:
    #print(msg)
    record = json.loads(msg.value)
    creation_datetime = record['creation_datetime']
    username = record['username']
    location = record['location']
    userDescr = record['userDescr']
    followers = record['followers']
    retweets = record['retweets']          
    favorites = record['favorites']
    sentiment = record['sentiment'] 

    # create and ingest JSON data into mongo
    try:
       twitter_rec = {'creation_datetime':creation_datetime,'username' :username,'location':location,
                      'userDescr':userDescr,'followers':followers,'retweets':retweets,'favorites':favorites,
                      'sentiment':sentiment}
       print(twitter_rec)
       rec = db.tweets.insert_one(twitter_rec)
       print("Data inserted successfully")
    except:
       print("Data Could not be inserted")