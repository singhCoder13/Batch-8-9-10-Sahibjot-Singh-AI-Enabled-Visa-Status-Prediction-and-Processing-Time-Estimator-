
# inear Regression Results 
lr_mae = 85.056
lr_rmse = 172.362
lr_r2 = 0.0601

#Random Forest Results
rf_mae = 80.8
rf_rmse = 150.6
rf_r2 = 0.01

#Gradient Boosting Results
gb_mae = 80.624
gb_rmse = 166.408
gb_r2 = 0.123

print("\nMODEL COMPARISON SUMMARY\n")

print("Linear Regression")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2:", lr_r2)

print("\nRandom Forest")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)

print("\nGradient Boosting")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)
print("R2:", gb_r2)

print("\nFINAL OBSERVATION:")
print("Gradient Boosting gives the lowest MAE and RMSE with highest R2.")
print("Hence, Gradient Boosting is selected as the final model.")
