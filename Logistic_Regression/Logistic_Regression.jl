#############################################
# Created on 2020-02-21
# Creator: khanh.brandy
# Logistic Regression using Batch Gradient Descent
#############################################

using DelimitedFiles
using StatsBase
using Statistics
using Plots

# Transform data
function transform(data)
    N = size(data,1)
    df = hcat(ones(N),data)
    return df
end

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

# Define logit func
logit(z) = 1 ./ (1 .+ exp.(-z))

# Define loss func
function loss(theta::Array{Float64}, X, y)::Float64
    h = logit(X*theta)
    J = y' * log.(h) + (1 .- y)'* log.(1 .- h)
    -J/length(y) # Find min(-J) instead of max(Likelihood function)
end

# Define gradient func
function gradient(theta::Array{Float64}, X, y)
    h = logit(X*theta)
    N = size(X,1)
    X'*(h - y)/N
end

# Define batch gradient algorithm
function bgd(X, y, alpha= 0.001, T = 256)::Array{Tuple{Array{Float64},Float64}}
    result =[]
    N, D = size(X)
    theta_0 = zeros(D)
    theta = theta_0
    for t = 1:T
        theta = theta - alpha*gradient(theta, X, y)
        value = loss(theta, X, y)
        push!(result, (theta,value))
    end
    result
end

# Define label mapping func 
function lbl_predict(pred, e::Float64 = 0.5)::Int
    if pred > e
        return 1
    else
        return 0
    end    
end

# Define func: get predictions
function predict(X::Array{Float64,2}, theta::Array{Float64})
    println("Start predicting...")
    z = X * theta
    probs = logit(z)
    preds = map(i-> lbl_predict(probs[i]), collect(1:length(probs)))
    println("Done predicting!")
    return preds
end   

# Define func: model evaluation
function model_eval(preds::Array{Int64}, y::Array{Int64})::Float64
    sum(preds .== y)/length(y)
end

# Main
data = readdlm("admission.txt", ',')
df = transform(data)
train_set , test_set = train_test_split(df)
X_train = train_set[:,1:3]
y_train = Int.(train_set[:,4])
X_test = test_set[:,1:3]
y_test = Int.(test_set[:,4])
alpha = 0.001
n_iters = 1000
results = bgd(X_train, y_train, alpha, n_iters) 
theta_best = results[n_iters][1]
preds = predict(X_train, theta_best)
acc = model_eval(preds,y_train)
# Plot loss curve
values = map(t -> results[t][2], 1:100)
plot(1:100, values, label = "Loss")

# Print results
println("Model accuracy = $(acc)")
println("Done model development!")