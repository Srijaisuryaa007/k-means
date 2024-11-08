#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
df= pd.read_csv("Sales.csv")
df.head(5)








# In[7]:


dup=df.drop_duplicates()
print(dup)


# In[8]:


dup.head(5)


# In[9]:


dup["Customer_Age"].max()


# In[10]:


dup.pivot_table(values="Order_Quantity",index="Date",columns="Product",fill_value=0)


# In[11]:


dup["Revenue"].mean()


# In[16]:


c=dup.value_counts("Sub_Category")
print(c)


# In[12]:


lable =["Tires and Tubes","Bottles and Cages","Road Bikes","Helmets",         
"Mountain Bikes",    
"Jersey",          
"Caps",                
"Fenders",               
"Touring Bikes",
"Gloves",               
"Cleaners",             
"Shorts",            
"Hydration Packs",      
"Socks",               
"Vests",                  
"Bike Racks ",            
"Bike Stands " ]


# In[57]:


plt.pie(c, labels= lable)

plt.title("product vs quantity sold")
plt.show()


# In[ ]:


data=(())


# In[18]:


dup.describe()


# In[1]:


def op_k_means(dup, max_k):
    means = []
    inertias = []
    
    for k in range(1, max_k + 1): 
        km = KMeans(n_clusters=k, random_state=42) 
        km.fit(dup)
        means.append(k)
        inertias.append(km.inertia_)
        
    plt.figure(figsize=(10, 5))
    plt.plot(means, inertias, "o-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.xticks(means)  
    plt.show()




# In[2]:


country=["usa","aus","canada","uk","germany","france"]


# In[134]:


op_k_means(dup[["Cost","Profit"]],20)




kmeans= KMeans(n_clusters=3)
kmeans.fit(dup[["Cost","Profit"]])




dup.loc[:,"kmeans_3"]=kmeans.labels_




plt.scatter(dup["Unit_Price"], dup["Revenue"], c=dup["kmeans_3"], cmap='viridis')

plt.xlim(-0.1,100)
plt.ylim(-0.1,100)
plt.show()



# In[ ]:





# In[79]:


df.value_counts("Country")


# In[133]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X = df[['Customer_Age', 'Revenue']].values 


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')


plt.title('K-Means Clustering')
plt.xlabel('age')
plt.ylabel('Revenue')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True)


plt.show()


# In[ ]:




