#############################################
# Created on 2020-01-19
# Creator: khanh.brandy
# Linear Regression for simple classification
#############################################

using DataFrames
using CSV
using StatsBase
using Statistics
using Plots

# Define func: train_test_split
function train_test_split(df)
    train_size = 0.8
    n = 1:size(df,1)
    train_idx = sample(n, Int64(floor(size(df,1)*train_size)), replace = false)
    test_idx = setdiff(n, train_idx)
    train_set = df[train_idx, :]
    test_set = df[test_idx, :]
    return train_set , test_set
    println("Done train/test splitting!")
end

# Transform data
function transform(data)
    N = size(data,1)
    df = hcat(ones(N),data)
    return df
end

# Define func: get theta matrix using Normal equation
function fit(X_train::Array{Float64}, y_train::Array{Float64})::Array{Float64}
    println("Start fitting training set...")
    theta = inv(X_train'*X_train)*X_train'*y_train
    println("Done fitting training set!")
    return theta
end

# Define func: get predictions
function predict(X::Array{Float64}, theta::Array{Float64})
    println("Start predicting...")
    preds = X * theta
    println("Done predicting!")
    return preds
end    

# Define func: squared error
function squared_error(preds::Array{Float64}, y_test::Array{Float64})::Float64
    err = sum((y_test - preds).^2)
    println("Total squared error = $(err)")
    return err
end

# Define classification func 
function lbl_predict(x::Array{Float64,1}, theta::Array{Float64}, e::Float64 = 0.5)::Int
    if x' * theta > e
        return 1
    else
        return 0
    end    
end

function classify(X_test::Array{Float64,2}, theta::Array{Float64}, e::Float64 = 0.5)::Array{Int}
    N = size(X_test,1)
    map(i -> lbl_predict(X_test[i,:], theta, e), collect(1:N))
end

# Define func: model evaluation
function model_eval(X::Array{Float64,2}, y::Array{Float64}, theta::Array{Float64}, e::Float64 = 0.5)::Float64
    prediction = classify(X, theta,e)
    sum(prediction .== y)/length(y)
end

# Define func: find optimal threshold for e
function optimal_e(X_test, y_test, theta)
    es = collect(0.1:0.01:0.8)
    as = map(i -> model_eval(X_test, y_test, theta, es[i]), 1:length(es))
    plot(es, as, xlabel="threshold", ylabel="accuracy", label="Best threshold = $(es[argmax(as)])")
#     es[argmax(as)]
end

# Main
df = DataFrame(CSV.File("wdbc.txt", header = 0))   # Getting raw data
data = convert(Array, df)
train_set , test_set = train_test_split(data) #train_test_split
train_data = transform(train_set)
test_data = transform(test_set)
X_train = train_data[:,4:13]
y_train = train_data[:,3]
X_test = test_data[:,4:13]
y_test = test_data[:, 3]
theta = fit(X_train, y_train)
optimal_e(X_test, y_test, theta)

