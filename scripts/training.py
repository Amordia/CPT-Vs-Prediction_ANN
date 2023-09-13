# Training function
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, n_epochs, min_delta, patience):
    best_val_loss = np.inf
    patience_counter = 0
    training_loss = []
    validation_loss = []

    for epoch in range(n_epochs):
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            outputs = model(X_train_batch)
            loss = criterion(outputs, y_train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_val_batch, y_val_batch in val_loader:
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()

        val_loss /= len(val_loader)  # Take the average
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        training_loss.append(loss.item())
        validation_loss.append(val_loss)

    return model, training_loss, validation_loss, best_val_loss

#Train and validate using K-fold cross-validation
val_scores1 = []
val_scores2 = []
val_scores3 = []
# Convert your data to DataLoader
for train_index, val_index in kf.split(X_scaled):
    X_train_tensor = torch.tensor(X_scaled[train_index], dtype=torch.float32)
    y_train_tensor = torch.tensor(y[train_index], dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_scaled[val_index], dtype=torch.float32)
    y_val_tensor = torch.tensor(y[val_index], dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    Clay_net = Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(Clay_net.parameters(), lr=0.05, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    Clay_net, training_loss, validation_loss, best_val_loss = train_model(
        Clay_net, 
        criterion, 
        optimizer, 
        scheduler, 
        train_loader, 
        val_loader, 
        n_epochs, 
        min_delta, 
        patience
    )

    y_val_pred = Clay_net(X_val_tensor).detach().numpy().flatten()
    R2_val = r2_score(y_val_tensor.numpy().flatten(), y_val_pred)
    bias_val = average_bias(y_val_tensor.numpy().flatten(), y_val_pred)
    WIA_val = willmott_index_of_agreement(y_val_tensor.numpy().flatten(), y_val_pred)
    val_scores1.append(R2_val)
    val_scores2.append(bias_val)
    val_scores3.append(WIA_val)
    print(f"Validation R2 score: {R2_val}")
    print(f"Validation Average Bias: {bias_val}")
    print(f"Validation Willmott Index of Agreement: {WIA_val}")


    # New code for plotting validation results
    plt.figure(figsize=(10, 6), dpi=150)
    y_val_tensor_np = y_val_tensor.cpu().numpy().flatten()
    sns.scatterplot(x=np.array(y_val_tensor.numpy()).flatten(), y=y_val_pred, edgecolor='k', s=50, alpha=0.8)
    plt.plot([min(y_val_tensor.numpy()), max(y_val_tensor.numpy())], [min(y_val_pred), max(y_val_pred)], 'r', linewidth=2)
    plt.xlabel("True Values", fontsize=14)
    plt.ylabel("Predicted Values", fontsize=14)
    #plt.title(f"Comparison between True and Predicted Values for Validation Fold {train_index+1}", fontsize=16)
    plt.show()
    
avg_val_score1 = np.mean(val_scores1)
avg_val_score2 = np.mean(val_scores2)
avg_val_score3 = np.mean(val_scores3)
print(f"Average validation R2 score across {n_splits} folds: {avg_val_score1:.4f}")
print(f"Average validation Average Bias across {n_splits} folds: {avg_val_score2:.4f}")
print(f"Average validation Index of Agreement across {n_splits} folds: {avg_val_score3:.4f}")

plt.plot(training_loss, label="Training Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over the Epochs")
plt.legend()
plt.show()

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

final_net_clay = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_net_clay.parameters(), lr=0.01, weight_decay=0.001)

training_loss = []
best_loss = np.inf
patience_counter = 0

for epoch in range(1000):
    outputs = final_net_clay(X_tensor)
    loss = criterion(outputs, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if loss.item() < best_loss - min_delta:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    training_loss.append(loss.item())

plt.plot(training_loss, label="Training Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over the Epochs")
plt.legend()
plt.show()

# Save model after training
torch.save(final_net_clay.state_dict(), 'final_net_clay.pth')

# Create a Pandas Excel writer using XlsxWriter as the engine
excel_writer = pd.ExcelWriter('output_results.xlsx', engine='openpyxl')

for i, (train_index, val_index) in enumerate(kf.split(X_scaled)):
    # Convert to tensors and predict
    X_val_tensor = torch.tensor(X_scaled[val_index], dtype=torch.float32)
    y_val_tensor = torch.tensor(y[val_index], dtype=torch.float32).view(-1, 1)
    
    # Get predictions on validation dataset
    Clay_net.eval()
    with torch.no_grad():
        y_pred_tensor = Clay_net(X_val_tensor)
    
    y_pred_flatten = y_pred_tensor.numpy().flatten()

    # Create DataFrame to store the output results with original feature values
    results_df = pd.DataFrame(scaler.inverse_transform(X_scaled[val_index]), columns=['Depth', 'qt', 'us'])
    results_df['Actual Vs'] = y[val_index]
    results_df['Predicted Vs'] = y_pred_flatten
    results_df['Bias'] = bias_final
    results_df['Willmott Index of Agreement'] = WIA_final
    # Calculate error between predicted and actual values
    results_df['Error'] = abs(results_df['Predicted Vs'] - results_df['Actual Vs'])

    # Write each DataFrame to a different worksheet
    results_df.to_excel(excel_writer, sheet_name=f'Fold_{i+1}', index=False)

# Save the results to the Excel file
excel_writer.close()

# Load your model (make sure it's already trained)
model = Net()
model.load_state_dict(torch.load('final_net_clay.pth'))
model.eval()

# Wrap your model in a function that allows it to predict from numpy input
def pred(X):
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).numpy()

# Convert your tensor data back to numpy
X_np = X_tensor.numpy()
y_np = y_tensor.numpy().ravel()

num_repeats = 100
feature_names = ['Depth', 'qt', 'u2']
feature_importances = np.zeros((num_repeats, len(feature_names)))
original_preds = pred(X_np)

# Repeat the process for a more robust estimate
for r in range(num_repeats):
    # Calculate importance for each feature
    for i, feature_name in enumerate(feature_names):
        # Copy array
        X_np_copy = X_np.copy()

        # Permute column
        np.random.shuffle(X_np_copy[:, i])

        # Make predictions
        shuff_preds = pred(X_np_copy)

        # Get metric
        original_metric = r2_score(y_np, original_preds)
        permuted_metric = r2_score(y_np, shuff_preds)

        # The importance of a feature is determined by the decrease in the performance metric
        importance = original_metric - permuted_metric

        feature_importances[r, i] = importance

# Calculate mean importance for each feature
mean_importances = feature_importances.mean(axis=0)

# Print and plot the mean importance of each feature
for i, imp in enumerate(mean_importances):
    print(f'{feature_names[i]} has a mean permutation importance of {imp:.4f}')

# Plotting the average importance as we take more measurements
plt.figure(figsize=(10, 6))
for i, feature_name in enumerate(feature_names):
    plt.plot(np.arange(1, num_repeats + 1), feature_importances[:num_repeats, i].cumsum() / np.arange(1, num_repeats + 1), label=feature_name)
plt.xlabel('Number of measurements')
plt.ylabel('Average Permutation Importance')
plt.legend()
plt.title('Change of the Average Permutation Importance with Number of Measurements')
plt.show()

# Plotting the final mean importances
plt.figure(figsize=(10, 6))
plt.bar(feature_names, mean_importances)
plt.xlabel('Features')
plt.ylabel('Mean Permutation Importance')
plt.title('Final Mean Permutation Importance for Each Feature')
plt.show()