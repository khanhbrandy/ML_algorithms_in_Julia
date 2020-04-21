using DataFrames
using Printf
using CSV
using StatsBase
using Statistics

# Get data
df = DataFrame(CSV.File("iris.data", header=[
                        "sepal_length",
                        "sepal_width",
                        "petal_length",
                        "petal_width",
                        "class"]))
names = Dict(zip(unique(df.class), [1,2,3]))
df.class = map(akey -> names[akey], df.class)

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

# Define func: calculate parameters
function params_calculate(X_train, y_train)
    N, D = size(X_train)
    K = length(unique(y_train))
    theta = zeros(K)
    mu = zeros(K, D)
    sigma = zeros(K, D)
    for k=1:K
        k_idx = y_train .== k
        theta[k] = sum(k_idx)/N
        X_k = X_train[k_idx, :]
        mu[k,:] = mean(X_k, dims = 1)
        sigma[k,:] = std(X_k, dims = 1)
    end
    return theta, mu, sigma
end

# Define func: calculate posterior probabilities
function prob_calculate(x, theta, mu, sigma)::Int
    K = length(theta)
    probs = zeros(K)
    for k = 1:K
        a = -log.((sqrt(2*pi)*sigma[k,:]))
        b = (x-mu[k,:]).^2
        c = 2*(sigma[k,:].^2)
        prob = log(theta[k]) + sum(a - b./c)
        probs[k] = prob
    end
    return argmax(probs)
end

# Define func: predict
function predict(X, theta, mu, sigma)::Array{Int}
    N = size(X,1)
    preds = zeros(N)
    for i = 1:N
        pred = prob_calculate(X[i,:], theta, mu, sigma)
        preds[i] = pred
    end
    return preds
end

# Define func: model evaluation
function model_eval(X::Array{Float64,2}, y::Array{Int}, theta, mu, sigma)::Float64
    prediction = predict(X, theta, mu, sigma)
    sum(prediction .== y)/length(y)
end

# Main
train_data , test_data = train_test_split(df)
X_train = convert(Array, train_data[:,1:end-1])
y_train = Int.(train_data[:,end])

X_test = convert(Array, test_data[:,1:end-1])
y_test = Int.(test_data[:, end])

theta, mu, sigma = params_calculate(X_train, y_train)
preds = predict(X_test, theta, mu, sigma)
train_accuracy = model_eval(X_train, y_train, theta, mu, sigma)
test_accuracy = model_eval(X_test, y_test, theta, mu, sigma)
println("Model accuracy on training set is: ",  convert(Float64, train_accuracy*100), "%")
println("Model accuracy on testing set is: ",  convert(Float64, test_accuracy*100), "%")
