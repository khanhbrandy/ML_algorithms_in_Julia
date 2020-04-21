using DataFrames
using CSV
using StatsBase
using Statistics

# Define func: train_test_split
function train_test_split(df)
    train_size = 0.8
    n = 1:size(df,1)
    train_idx = sample(n, Int64(floor(size(df,1)*train_size)), replace = false)
    test_idx = setdiff(n, train_idx)
    train_data = df[train_idx, :]
    test_data = df[test_idx, :]
    return train_data , test_data
    println("Done train/test splitting!")
end

# Define func: get theta matrix using Normal equation
function fit(X_train, y_train)
    println("Start fitting training set...")
    N = length(y_train)
    X = hcat(ones(N),X_train)
    theta = inv(X'*X)*X'*y_train
    println("Done fitting training set!")
    return theta
end

# Define func: get predictions
function predict(X_test, theta)
    println("Start predicting...")
    N = size(X_test,1)
    X = hcat(ones(N),X_test)
    preds = X * theta
    println("Done predicting!")
    return preds
end    

# Define func: squared error
function squared_error(preds, y_test)
    err = sum((y_test - preds).^2)
    println("Total squared error = $(err)")
    return err
end

# Main
df = DataFrame(CSV.File("forbes.txt", header = 0))   # Getting raw data
data = transpose(convert(Array, df))
train_data , test_data = train_test_split(data)
X_train = train_data[:,1:end-1]
y_train = train_data[:,end]
X_test = test_data[:,1:end-1]
y_test = test_data[:, end]
theta = fit(X_train, y_train)
preds = predict(X_test, theta)
err = squared_error(preds, y_test)
