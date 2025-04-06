import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\lenovo\Documents\python\years_experience_salary.csv")

x=df[['YearsExperience']]
y=df['Salary']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("mean:",mean_squared_error(y_test,y_pred))
print("r2:",r2_score(y_test,y_pred))




# Scatter plot: actual data
plt.scatter(x_test, y_test, color='blue', label='Actual')

# Line plot: predicted regression line
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Prediction')

# Labels and title
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction vs Actual Data")
plt.legend()
plt.grid(True)
plt.show()
