from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X = glass_identification.data.features 
y = glass_identification.data.targets 

# metadata 
print(glass_identification.metadata) 
  
# # variable information 
print(glass_identification.variables) 
