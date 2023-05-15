# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: PVproject
#     language: python
#     name: python3
# ---

# %%
#hide
import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd

# %matplotlib inline

# %%
np.random.seed(42)
x = np.random.uniform(1, 10, 100)
y = np.random.normal(5, 0.2, size=x.shape)

## ask for what is the intetion of this line in Nicolas's code
plt.scatter(x,y, label='y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend();

# %%
# model prediction
pred = 5
reg = 0 # regularization

# g stands for gradient and h stands for hessian
# g = 2*(y-x)*x from the mean square error
g = np.array([2*(i-pred)*i for i in y])
h = np.array([-2*i for i in y])

# The optimal weight for the model is the negative sum of the gradient divided by the sum of the hessian
# We can see this in the XG Boost paper
w = -g.sum()/(h.sum()+ reg)

print(f'the optimal weight is {w:.4} for the model with a constant prediction of y={pred}')

# %%
# model prediction
pred = 1

# g stands for gradient and h stands for hessian
# g = 2*(y-x)*x from the mean square error
g = np.array([2*(i-pred)*i for i in y])
h = np.array([-2*i for i in y])

# The optimal weight for the model is the negative sum of the gradient divided by the sum of the hessian
# We can see this in the XG Boost paper
w = -g.sum()/(h.sum()+ reg)

print(f'the optimal weight is {w:.4} for the model with a constant prediction of y={pred}')

# %%

y = np.where(x < 5, x,5) + np.random.normal(0, 0.3, size=x.shape)
#x = x.reshape(-1,1)

df = pd.DataFrame({"X":x,"y":y})

plt.scatter(df["X"],df["y"], label='y')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated data')
plt.legend();

# %%
df

# %%
f_0 = np.mean(y) # initial prediction using the mean of the target

#Compute gradient and hessian of each sample
df["g"] = 2*(df["y"]-f_0)/len(df["y"])  # is this correct?
df["h"] = -2/len(df["y"])               # is this correct?


df = df.sort_values(by=['X'])
df.reset_index(inplace = True, drop = True)
#df.reset_index(inplace = True, drop = True) # We are not using the index anymore
#Keep track of the loss

l = []

#Test the different threshold, this is to find the optimal threshold to split the data
# Each threshold will be tested on the whole dataset
for i in range(len(df)):
    
    # select the threshold to split the data
    t = df["X"][i]
    df_L =  df[df["X"]<t]
    df_R =  df[df["X"]>=t]
    
    
    G, H =np.sum(df["g"]), np.sum(df["h"])
    
    G_L = np.sum(df_L["g"])
    G_R = np.sum(df_R["g"])
    
    H_L = np.sum(df_L["h"])
    H_R = np.sum(df_R["h"])
    
    if (H==0) or (H_R==0) or (H_L ==0):
        continue
        
    split = 1/2*((G_L**2)/H_L + (G_R**2)/H_R - (G**2)/H)
    
    l.append(split)
    
    opt_split = min(opt_split, split)
    
    if split == opt_split:
        x_opt = df["X"][i]

print(f'The optimal threshold value of {x_opt:.4}')
plt.plot(df["X"][1:],l)
plt.title("Split impact on loss for different threshold values")
plt.xlabel("Threshold value")
plt.ylabel("Split impact")


# %%

#Compute the new leaf weights
def split(df, x_opt):
    # split the data into two parts
    df_left  = df[df["X"]<x_opt]
    df_right = df[df["X"]>x_opt]
    # compute the gradient and hessian for each part
    G_Left = np.sum(df_left["g"])
    H_Left = np.sum(df_left["h"])
    
    G_Right = np.sum(df_right["g"])
    H_Right = np.sum(df_right["h"])

    w_left = -G_Left/H_Left
    w_right = -G_Right/H_Right
    
    return w_left, w_right

w_left, w_right = split(df, x_opt)

print(f'The optimal weight for the left leaf is {w_left:.4}')
print(f'The optimal weight for the right leaf is {w_right:.4}')

# %%
f_1 = []

for i in df["X"]: 
    if i<x_opt:
        f_1.append(f_0 + w_left)
    else :
        f_1.append(f_0 + w_right)

plt.figure()
plt.scatter(df["X"],df["y"])
plt.plot(df["X"],f_1, color = "red")

# %%
fig = plt.figure()
plt.bar(df["X"],df["g"], edgecolor="black", width = 0.2,zorder = 0, label = "$\mathregular{G_b}$")
plt.scatter(df["X"], [0]*len(df["X"]), marker = '+',zorder = 10, label = "Bin representent")
plt.legend()

# %%
