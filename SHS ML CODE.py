#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from tabulate import tabulate

# Read data from the Excel file into a DataFrame
df = pd.read_excel("SHS DATASET.xls")

# Create a binary label indicating if 'karthik' is present in the 'doctor name' column
df['label'] = df["doctor name"].str.contains('karthik', case=False, na=False).astype(int)

# Extract features and labels from the DataFrame
x = df["hospital name"]
y = df[["doctor name", "location", "time", "review"]]  # Added 'availability' column

# Convert hospital names into a bag-of-words representation using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(x)

# Convert the 'time' column to string to avoid TypeError
y['time'] = y['time'].astype(str)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Multinomial Naive Bayes classifier wrapped in MultiOutputClassifier
model = MultiOutputClassifier(MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Take user input for the hospital name
user_input = input("Enter hospital name: ")
user_location = input("Enter your location:")

# Transform user input using the same CountVectorizer
user_data = cv.transform([user_input])

# Make predictions for the user input
output = model.predict(user_data)

# Display predictions
predicted_doctor, predicted_location, predicted_time, predicted_availability = output[0]
print(f"Predicted Doctor: {predicted_doctor}")
print(f"Predicted Location: {predicted_location}")
print(f"Predicted Time: {predicted_time}")


# Display previous reviews about the doctor by the patients
doctor_name = predicted_doctor
previous_reviews = df[df["doctor name"] == doctor_name]

# Format reviews into a table with patient names
reviews_table = []
for index, row in previous_reviews.iterrows():
    reviews_table.append([row['patient name'], row['review']])

# Print the reviews table
print("\nPrevious Reviews about the Doctor:")
print(tabulate(reviews_table, headers=['Patient Name', 'Review'], tablefmt='grid'))



# In[ ]:




