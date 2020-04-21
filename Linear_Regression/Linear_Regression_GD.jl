#############################################
# Created on 2020-01-19
# Creator: khanh.brandy
# Linear Regression: Using Batch Gradient Descent
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

# Define loss func
function loss(X,y,theta)
    u = X*theta-y
    u'*u/length(y)
end

# Define batch gradient algorithm
function bgd(X,y, alpha = 0.01, n_iters = 1000)::Array{Float64,2}
    N,D = size(X)
    theta = zeros(D, n_iters)
    for t=1:n_iters-1
        nabla = X'*(X*theta[:,t] - y)/N
        theta[:,t+1] = theta[:,t] - alpha*nabla
    end
    return theta
end

# Define func: get predictions
function predict(X, theta)
    println("Start predicting...")
    preds = X * theta
    println("Done predicting!")
    return preds
end    

# Define func: squared error
function squared_error(preds, y_test)
    err = sum((y_test - preds).^2)
    return err
end

# Main
data = readdlm("profits.txt", ',')
df = transform(data)
train_set , test_set = train_test_split(df)
X_train = train_set[:,1:2]
y_train = train_set[:,3]
X_test = test_set[:,1:2]
y_test = test_set[:,3]
alpha = 0.001
n_iters = 1000
theta_s = bgd(X_train, y_train)
theta_best = theta_s[:,n_iters]
preds = predict(X_test, theta_best)
err = squared_error(preds, y_test)
# Plot loss curve
J = map(j -> loss(X_train,y_train,theta_s[:,j]), collect(1:n_iters))
iters = collect(1:n_iters)a
plot(iters, J, label="The best theta = $(theta_s[:,n_iters])", 
    xlabel = "Number of iterations", 
    ylabel = "Loss values", 
    linewidth = 2, color =:red)

# Print results
println("Total squared error = $(err)")
println("Done model development!")