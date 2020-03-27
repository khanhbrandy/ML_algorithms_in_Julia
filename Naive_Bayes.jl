using DataFrames
using Printf

data = DataFrame(
    _message = String[],
    _secret = Int[],
    _offer = Int[],
    _low = Int[],
    _price = Int[],
    _valued = Int[],
    _customer = Int[],
    _today = Int[],
    _dollar = Int[],
    _million = Int[],
    _sports = Int[],
    _is = Int[],
    _for = Int[],
    _play = Int[],
    _healthy = Int[],
    _pizza = Int[],
    class = Int[]
)
vocabs = ("secret",
    "offer",
    "low",
    "price",
    "valued",
    "customer",
    "today",
    "dollar",
    "million",
    "sports",
    "is",
    "for",
    "play",
    "healthy",
    "pizza",
)
messages = Dict("million dollar offer" => 1, 
    "secret offer today" => 1,
    "secret is secret" => 1,
    "low price for valued customer" => 0,
    "play secret sports today" => 0,
    "sports is healthy" => 0,
    "low price pizza" => 0,
)

# Generate observations func
function generate_obs(vocabs, messages)
    println("Start generating observations!")
    result = Any[]
    for m in keys(messages)
        r = Any[m]
        for v in vocabs
            if occursin(v,m)
                push!(r,1)
            else
                push!(r,0)
            end
        end
    push!(r,messages[m])
    push!(result,r)
    end
    println("Done generating observations!")
    return result
end

# Add observations func
function add_obs(data, observations)
    for obs in observations
        push!(data, obs)
    end
    return data
end

# Calculate Likelihood
function calculateLikelihood(df, class)
    #println("Start calculating Likelihood for class ", class)
    data = df[df.class .== class, :][!, 1:15]
    n= size(data, 1)
    result = Any[]
    for col in names(data)
        theta_1 = sum(data[!, col])/n
        push!(result,theta_1)
    end
    #println("Done calculating Likelihood for class ", class)
    return result
end

#Generate Likelihood matrix
function generateLiMatrix(df)
    result = Dict()
    for cls in unique(df.class)
        println("Start generating Likelihood matrix for class ", cls)
        theta = calculateLikelihood(df, cls)
        result[cls] = theta
        println("Done generating Likelihood matrix for class ", cls)
    end
    return result
end

# Naive Bayes 
function calPosteriorDis(sample, likelihood_dis)
    result = Dict()
    range = 1:1:length(likelihood_dis)
    for cls in keys(likelihood_dis)
        r = Any[]
        for i in range
            if sample[i] .== 1
                push!(r, likelihood_dis[cls][i])
            else
                push!(r, 1-likelihood_dis[cls][i])
            end
        end
        posterior_prob = prod(r)*prior_dis
        result[cls] = posterior_prob
    end
    return result
end

# Main
sample = [0,0,1,1,0,0,1,0,1,0,0,0,1,0,0]
observations = generate_obs(vocabs, messages)
df = add_obs(data, observations)[!, 2:17]
prior_dis = sum(df.class)/size(df, 1)
likelihood_dis = generateLiMatrix(df)
result = calPosteriorDis(sample, likelihood_dis)
println("\nThe predicted probabilities are as follows:")
@printf "Spam (1): %0.3f \n" float(result[1])
@printf "Non-Spam (0): %0.3f" float(result[0])