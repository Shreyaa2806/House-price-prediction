#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# In[34]:


data = pd.read_csv("E:\\project\\lectures\\house_prices.csv")
print("First 5 rows:")


# In[ ]:





# In[36]:


data.head()


# In[43]:


# 3. Choose features (X) and target (y)
# Example columns: Area, Rooms, Location, Price
# If your dataset has different names, change here
X = data[["Area_sqft", "Rooms", "Bathrooms", "Age"]]
y = data["Price"]


# In[45]:


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[47]:


# 5. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)


# In[49]:


# 6. Predict
y_pred = model.predict(X_test)


# In[51]:


# 7. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[53]:


print("\nMODEL PERFORMANCE:")
print("MAE :", mae)
print("MSE :", mse)
print("R²  :", r2)


# In[55]:


# 8. Predict new house price
new_house = [[1800, 3, 2, 10]]
prediction = model.predict(new_house)
print("\nPredicted Price for New House:", prediction[0])


# In[ ]:




