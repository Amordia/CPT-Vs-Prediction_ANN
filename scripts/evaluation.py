def average_bias(y_true, y_pred):
    return np.mean(y_pred/y_true)

def willmott_index_of_agreement(y_true, y_pred):
    MSE = np.mean((y_pred - y_true)**2)
    PE = np.mean((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2)
    return 1 - MSE / PE

import seaborn as sns
sns.set(style="whitegrid")
y_pred = final_net_clay(X_tensor).detach().numpy().flatten()
MAE = mean_absolute_error(y, y_pred)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y, y_pred)
bias_final = average_bias(y, y_pred)
WIA_final = willmott_index_of_agreement(y, y_pred)
n_data_points = len(y)

plt.figure(figsize=(10, 6),dpi=150)
sns.scatterplot(x=y, y=y_pred, edgecolor='k', s=50, alpha=0.8)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r', linewidth=2)
plt.xlabel("True Values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.title("Comparison between True and Predicted Values for the Final Model (Clay)", fontsize=16)
print(f"Final Model Average Bias: {bias_final}")
print(f"Final Model Willmott Index of Agreement: {WIA_final}")

text = "MAE: {:.2f}\nMSE: {:.2f}\nRMSE: {:.2f}\nR2: {:.2f}\nAverage Bias: {:.2f}\nIndex_of_Agreement: {:.2f}\nData Points: {}".format(MAE, MSE, RMSE, R2, bias_final, WIA_final, n_data_points)
plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.savefig('improved_model_comparison_plot(clay).png', dpi=150, bbox_inches='tight')
plt.show()

sns.set(style="white")
g = sns.JointGrid(data=pd.DataFrame({"True Values": y, "Predicted Values": y_pred}), x="True Values", y="Predicted Values", height=8, space=0)

cmap = sns.cubehelix_palette(dark=0.3, light=0.8, as_cmap=True)
g = g.plot_joint(sns.histplot, bins="auto", cmap=cmap, cbar=True, cbar_kws={"label": "Density"})
g = g.plot_joint(sns.regplot, scatter=False, color="r", truncate=False, ci=None)

# plot_envelope(y, y_pred, g.ax_joint)

g = g.plot_marginals(sns.histplot, kde=False, bins=15, color="black", alpha=0.6)

g.ax_joint.set_xlim(min(y), max(y))
g.ax_joint.set_ylim(min(y), max(y))

text = "MAE: {:.2f}\nMSE: {:.2f}\nRMSE: {:.2f}\nR2: {:.2f}\nData Points: {}".format(MAE, MSE, RMSE, R2, n_data_points)
g.ax_joint.text(0.05, 0.95, text, transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

g.savefig('advanced_final_model_comparison_plot(clay).png', dpi=300, bbox_inches='tight')
plt.show()