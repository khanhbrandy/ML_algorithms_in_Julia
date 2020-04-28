#############################################
# Created on 2020-02-21
# Creator: khanh.brandy
# Logistic Multinomial Regression using Batch Gradient Descent
#############################################

using DelimitedFiles
using StatsBase
using Statistics
using Plots

# Transform data
function transform(data)
    n = size(data,1)
    df = hcat(ones(n),data)
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

# Define Log-Sum-Exp func
function LSE(L)
    maxL = findmax(L)[1]
    new_L = map(i -> exp(i-maxL), L)
    maxL + log(sum(new_L))
end

# Define Loss func
function loss(X, y, theta)
    n = length(y)
    J_1 = sum(X.*theta[y,:])
    M = X*theta'
    N = map(i-> LSE(M[i,:]), collect(1:n))
    J_2 = sum(N)
    J = J_1 - J_2
    return 1/n * J
end

# Define sample_likelihood
function sample_likelihood(X, y)
    n,d = size(X)
    c = length(unique(y))
    H = zeros(c,d)
    for i=1:c
        K = (y .== i)
        H[i,:] = (1/n) .* sum(X[K,:], dims = 1) 
    end
    return H
end

########## In progress #########

# Define model likelihood 
function model_likelihood(X, y, theta)
    n,d = size(X)
    c = length(unique(y))
    M = sum(exp.(X*theta'))
    H = zeros(c)
    for i=1:c
        K = (y .== i)
        H[i] = sum(exp.(X[K,:]*theta'))/M  
    end
    return H
end


# Main
data = readdlm("Wine Origin/wine.data", ',')
train_set , test_set = train_test_split(data)
X_train = transform(train_set[:,2:end])
y_train = Int.(train_set[:,1])
X_test = transform(test_set[:,2:end])
y_test = Int.(test_set[:,1])

# Print results
println("Done model development!")
