using Ladder
using Losses
using DataFrames
using Gadfly

function boostingattack(n,k)

    y = rand(0:1.0,n)

    l = ladder(y)

    U = rand(0:1.0,n,k)
    ks = Float64[]
    ls = Float64[]
    goodis = Int[]

    for i in 1:k
        # kaggle score with 5 bits of precision
        kaggle_score = round(empirical_loss(y,U[:,i],loss01),5)
        push!(ks,kaggle_score)
        push!(ls,score!(l,Submission(U[:,i]),loss01))
    end

    # select all variables that lowered the score in the ladder mechanism
    if ls[1]< 0.5
        push!(goodis,1)
    end
    for i in 2:k
        if ls[i] < ls[i-1]
            push!(goodis,i)
        end
    end

    # select all positions with loss < 0.5
    bk = float(ks .<= 0.5)
    # select all positions that lowered the score
    bl = zeros(k)
    bl[goodis] = 1.0

    # compute boosted vectors
    ku = float(U * bk .>= countnz(bk)/2)
    lu = float(U * bl  .>= countnz(bl)/2)

    return empirical_loss(y,ku,loss01), empirical_loss(y,lu,loss01)

end

function plotboostingattack(n,k,r)

    x = [i for i in 10:10:k]
    kaggleavg = zeros(int(k/10))
    ladderavg = zeros(int(k/10))
    testavg = zeros(int(k/10))

    for _ in 1:r
        for i in 10:10:k
            ks,ls = boostingattack(n,i)
            kaggleavg[div(i,10)] += ks/r
            ladderavg[div(i,10)] += ls/r
            testavg[div(i,10)] += empirical_loss(rand(0:1.0,2*n),rand(0:1.0,2*n),loss01)/r
        end
    end

    df1 = DataFrame(x=x,y=kaggleavg,label="Kaggle")
    df2 = DataFrame(x=x,y=ladderavg,label="Ladder")
    df3 = DataFrame(x=x,y=testavg,label="Test")
    df = vcat(df1,df2,df3)

    kaggleplot = plot(df, x="x", y="y",color="label",Geom.point,Geom.line,
                       Scale.color_discrete_manual("red","blue","green"),
                       Coord.Cartesian(ymin=0.42,ymax=0.51),
                       Guide.yticks(ticks=[0.42,0.44,0.46,0.48,0.5]),
                       Guide.xlabel("Number of queries"), Guide.ylabel("Reported loss"),
                       Guide.title("Ladder vs Kaggle (normal precision)"),
                       Theme(minor_label_font_size=15pt,major_label_font_size=15pt,
                             key_label_font_size=14pt,key_title_font_size=15pt,key_position = :none))

    draw(PDF(string("kaggleboosting-",n,"-",k,"-",r,".pdf"), 6inch, 4.5inch),kaggleplot)
    writetable(string("kaggleboosting-",n,"-",k,"-",r,".csv"),df)

    return df
end
