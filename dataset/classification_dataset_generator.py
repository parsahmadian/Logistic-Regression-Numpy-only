import numpy as np
import pandas as pd

# ---------- settings ----------
np.random.seed(42)
num_samples = 1000
file_name = f"buying_house_possibility_dataset_{num_samples}.csv"

# ---------- features ----------
income = np.random.normal(loc=15, scale=5, size=num_samples)
income = np.clip(income, 5, 40)  
income = income.astype(int)

age = np.random.normal(loc=35, scale=10, size=num_samples)
age = np.clip(age, 20, 60)
age = age.astype(int)

family_size = np.random.randint(1, 6, size=num_samples)

credit_score = np.random.normal(loc=700, scale=50, size=num_samples)
credit_score = np.clip(credit_score, 500, 850)

house_price = np.random.normal(loc=300, scale=50, size=num_samples)
house_price = np.clip(house_price, 150, 500)
house_price = house_price.astype(int)

# ---------- labels ----------
prob_buy = (
    0.3 * (income / income.max()) +
    0.3 * (credit_score / credit_score.max()) +
    0.2 * (1 - (family_size - 1)/5) +
    0.2 * (1 - (house_price - 150)/350)
)

# add noise
prob_buy += np.random.normal(0, 0.05, size=num_samples)
prob_buy = np.clip(prob_buy, 0, 1)

# 0 | 1 
buy_house = np.random.binomial(1, prob_buy)

# ---------- dataframe ----------
df = pd.DataFrame({
    "income": income,
    "age": age,
    "family_size": family_size,
    "credit_score": credit_score,
    "house_price": house_price,
    "buy_house": buy_house
})

# ---------- save file ----------
df.to_csv(file_name, index=False)
print(f"{file_name} created! ✅")
print(df.head())
