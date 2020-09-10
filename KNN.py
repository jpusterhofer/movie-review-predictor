import numpy as np

#Read in raw data, get number of users and movies
data = np.genfromtxt("./u1-base.base", dtype=int, usecols=(0, 1, 2))
num_users = data[-1, 0]
num_movies = max(data[:, 1])

#Convert data into a table
#Rows are users, columns are movies, values are ratings
users = np.zeros((num_users, num_movies))
for row in data:
    users[row[0] - 1, row[1] - 1] = row[2]

#Give unrated movies rating of 3.5
#Track which movies weren't rated originally
has_rating = np.ones((num_users, num_movies), dtype=bool)
for user in range(num_users):
    for movie in range(num_movies):
        if users[user, movie] == 0:
            users[user, movie] = 3.5
            has_rating[user, movie] = False

#User-based collaborative filtering
#Calculate cos similarity of every user with every other user
sim = np.zeros((num_users, num_users))
magnitudes = [np.linalg.norm(row) for row in users]
for i in range(num_users):
    for j in range(num_users):
        if i != j:
            sim[i, j] = np.dot(users[i], users[j]) / (magnitudes[i] * magnitudes[j]) 

#For every user, find k users that are most similar
k = 3
best_k = []
for user in range(num_users):
    best_k.append(np.argpartition(sim[user], -k)[-k:])

#Predict ratings for movies the user hasn't rated
predict_users = users.copy()
for user in range(num_users):
    for movie in range(num_movies):
        if not has_rating[user, movie]:
            predict_users[user, movie] = sum(sim[user, best_k[user]] * users[best_k[user], movie]) / sum(sim[user, best_k[user]])

#Calculate mean squared error from test data
test_data = np.genfromtxt("./u1-test.test", dtype=int, usecols=(0, 1, 2))
error = 0
for row in test_data:
    error += (row[2] - predict_users[row[0] - 1, row[1] - 1])**2
error /= len(test_data)
print("User-based CF error: " + str(error))

#Leave one out cross validation
print("Leave one out CF:")
sim = np.zeros(num_users)
for k in range(1, 6):
    error = 0
    
    #Test on one movie's rating, train on rest
    for user in range(num_users):
        for movie in range(num_movies):
            if has_rating[user, movie]:

                #Replace rating with "unrated" score of 3.5
                rating = users[user, movie]
                users[user, movie] = 3.5

                #Calculate cos similarity
                for i in range(num_users):
                    if i != user:
                        sim[i] = np.dot(users[user], users[i]) / (np.linalg.norm(users[user]) * magnitudes[i])
                sim[user] = 0

                best_k = np.argpartition(sim, -k)[-k:] #Find k nearest neighbors
                prediction = sum(sim[best_k] * users[best_k, movie]) / sum(sim[best_k]) #Make prediction
                error += (rating - prediction)**2 #Add to error
                users[user, movie] = rating #Restore original rating

    #Print error
    error /= len(data)
    print("k=" + str(k) + ", Error=" + str(error))

#Item based collaborative filtering
#Read in genres
file = open("u-item.item", "r", encoding='latin1')
lines = file.readlines()
resultArray = []
for i in lines:
    i = i.rstrip("\n")
    i = i.split("|")
    newArray = i[5:]
    resultArray.append(newArray)
genres = np.array(resultArray, dtype=int)

#Calculate cos similarity of every movie with every other movie
sim = np.zeros((num_movies, num_movies))
magnitudes = [np.linalg.norm(row) for row in genres]
for i in range(num_movies):
    for j in range(num_movies):
        if i != j:
            sim[i, j] = np.dot(genres[i], genres[j]) / (magnitudes[i] * magnitudes[j]) 

#For every movie, find k movies that are most similar
best_k = []
for movie in range(num_movies):
    best_k.append(np.argpartition(sim[movie], -k)[-k:])

#Predict ratings for movies the user hasn't rated
predict_users = users.copy()
for user in range(num_users):
    for movie in range(num_movies):
        if not has_rating[user, movie]:
            predict_users[user, movie] = sum(sim[movie, best_k[movie]] * users[user, best_k[movie]]) / sum(sim[movie, best_k[movie]])

#Calculate mean squared error
test_data = np.genfromtxt("./u1-test.test", dtype=int, usecols=(0, 1, 2))
error = 0
for row in test_data:
    error += (row[2] - predict_users[row[0] - 1, row[1] - 1])**2
error /= len(test_data)
print("Item-based CF error: " + str(error))
