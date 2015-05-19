#
# load and analyze Kaggle photo quality prediction data
#
using Ladder
using Losses
using Gadfly
using DataFrames
using Distributions
using DataFrames

# Required files:
# location of directory containing all submissions in .csv format
all_submissions_dir = "../photo/ALLSUBS"
# location of csv file containing lines of form: "team","submission name" for all submissions
team_file = "../photo/teams.csv"
# location of csv file containing solution labels
solution_file = "../photo/PhotoSolution.csv"

function sub2labels(sub)
    # Parse labels in file sub
    f = open(sub)
    lines = readlines(f)
    close(f)
    # dealing with missing line breaks
    if length(lines) == 1
        lines = split(lines[1],"\r")
    end
    # removing header line if present
    if lines[1][1:2] == "id" || length(lines) == 12001
        lines = lines[2:end]
    end
    if length(lines) != 12000
        # fixes some submissions that include training labels
        lines = lines[1:12000]
    end
    labels = Float64[]
    for line in lines
        # fixes two common formatting errors
        line = replace(line,"\"","")
        line = replace(line,";",",")
        tokens = split(line,",")
        #handles submissions with only one column
        if length(tokens)==1
            # extra string() conversion fixed julia bug in float
            push!(labels,float(string(tokens[1],"")))
        else
            push!(labels,float(string(tokens[2],"")))
        end
    end
    return labels
end

function privpart(solution)
    # parse which labels are private
    f = open(solution)
    lines = readlines(f)
    close(f)
    lines = lines[2:end]
    private = Int[]
    for line in lines
        if split(line,",")[3][1:2]=="Pr"
            push!(private,1)
        else
            push!(private,0)
        end
    end
    return private
end

function build_teams()
    # Assumes team.csv contains lines of form: team,submission name
    # for all submissions
    team2id = (String => Array{String,1})[]
    id2team = (String => String)[]
    teams = readcsv(team_file)
    t = size(teams)[1]
    for i in 1:t
        team2id[teams[i,1]] = String[]
    end
    for i in 1:t
        push!(team2id[teams[i,1]],idfromname(teams[i,2]))
        id2team[idfromname(teams[i,2])] = teams[i,1]
    end
    return team2id,id2team
end

idfromname(name) = split(name,' ')[1]

function build_allsubs()
    allsubs = Submission[]
    badsubs = Submission[]
    csvfiles = readdir(all_submissions_dir)
    for sub in csvfiles
        id = idfromname(sub)
        try
            predictions = sub2labels(string(all_submissions_dir,"/",sub))
            # test if submission is valid by scoring it
            empirical_loss(solution[private],predictions[private],cliplogloss)
            empirical_loss(solution[public],predictions[public],cliplogloss)
            push!(allsubs,Submission(id,id2team[id],predictions))
        catch
            push!(badsubs,Submission(id,id2team[id],Float64[]))
        end
    end
    return allsubs,badsubs
end

function lb(privs=private,pubs=public,p=kaggle_precision,lossfct=cliplogloss)
    # Kaggle leaderboard per team is equivalent to running fixed step size ladder
    # for each team with Kaggle accuracy
    prl = (String => LadderState)[]
    pbl = (String => LadderState)[]
    for team in keys(team2id)
        prl[team] = ladder(solution[privs],10.0^(-p))
        pbl[team] = ladder(solution[pubs],10.0^(-p))
    end
    for sub in allsubs
        score!(prl[sub.team],Submission(sub.id,sub.team,sub.labels[privs]),lossfct)
        score!(pbl[sub.team],Submission(sub.id,sub.team,sub.labels[pubs]),lossfct)
    end
    return prl,pbl
end

function lbpf(privs=private,pubs=public,lossfct=cliplogloss)
    #
    # Evaluate parameter free ladder, using one instance per team
    #
    prl = (String => LadderState)[]
    pbl = (String => LadderState)[]
    for team in keys(team2id)
        prl[team] = ladder(solution[privs])
        pbl[team] = ladder(solution[pubs])
    end
    for sub in allsubs
        score!(prl[sub.team],Submission(sub.id,sub.team,sub.labels[privs]),lossfct)
        score!(pbl[sub.team],Submission(sub.id,sub.team,sub.labels[pubs]),lossfct)
    end
    return prl,pbl
end

function lbdiff(prl::Dict{String,LadderState},pbl::Dict{String,LadderState})
    prm = (Float64,String)[]
    pbm = (Float64,String)[]
    for team in keys(team2id)
        push!(prm,(prl[team].current,team))
        push!(pbm,(pbl[team].current,team))
    end
    return prm,pbm
end

#
# Experiments
#

function standard_splits()
    println("Plot: Score distribution")
    # compute kaggle leaderboards for each team
    prl, pbl = lb()
    # compute best sub for each team
    prm, pbm = lbdiff(prl,pbl)
    # compute leaderboards according to parameter-free ladder
    prlpf,pblpf = lbpf()
    prmpf, pbmpf = lbdiff(prlpf,pblpf)

    function plotscores(df,title="Deviation of scores")
        return Gadfly.plot(df, x="x", y="y", Geom.point,
                           Coord.Cartesian(ymin=0.182,ymax=0.202,xmin=0.182,xmax=0.202),
                           Guide.yticks(ticks=[0.19,0.20]),
                           Guide.xticks(ticks=[0.18,0.19,0.20]),
                           Guide.xlabel("Public score"), Guide.ylabel("Private score"),
                           Guide.title(title),
                           Theme(minor_label_font_size=15pt,major_label_font_size=15pt,
                                 key_label_font_size=14pt,key_title_font_size=15pt),
    layer(x=[0.182,0.202],y=[0.182,0.202],Geom.line,Theme(default_color=color("red"))))
    end

    top50 = sort(prm)[1:50]
    top50priv = [x for (x,_) in top50]
    top50pub = [pbl[t].current for (_,t) in top50]
    top50pf = sort(prmpf)[1:50]
    top50pfpriv = [x for (x,_) in top50pf]
    top50pfpub = [pblpf[t].current for (_,t) in top50pf]
    df1 = DataFrame(x=top50pub,y=top50priv)
    df2 = DataFrame(x=top50pfpub,y=top50pfpriv)
    top50kaggle = plotscores(df1,"Top 50 Kaggle scores on private data")
    top50ladder = plotscores(df2,"Top 50 Ladder scores on private data")
    draw(PDF(string("top50kaggle-private.pdf"), 5inch, 4inch),top50kaggle)
    draw(PDF(string("top50ladder-private.pdf"), 5inch, 4inch),top50ladder)
    writetable(string("top50kaggle-private.csv"),df1)
    writetable(string("top50ladder-private.csv"),df2)

    println("Plot: Top 50 score comparison")
    dftop50test = DataFrame(x=[1:50],y=top50priv,label="Test")
    dftop50kaggle = DataFrame(x=[1:50],y=top50pub,label="Kaggle")
    dftop50ladder = DataFrame(x=[1:50],y=top50pfpub,label="Ladder")
    dftop50 = vcat(dftop50test,dftop50kaggle,dftop50ladder)
    top50plot = Gadfly.plot(dftop50, x="x", y="y", Geom.point,
                           Coord.Cartesian(ymin=0.182,ymax=0.202),
                            Geom.line,color="label",Scale.color_discrete_manual("green","red","blue"),
                           Guide.yticks(ticks=[0.19,0.20]),
                           Guide.xticks(ticks=[1,10,20,30,40,50]),
                           Guide.xlabel("Reported score"), Guide.ylabel("Rank by test score"),
                           Guide.title("Top 50 submissions by test score"),
                           Theme(minor_label_font_size=15pt,major_label_font_size=15pt,
                                 key_label_font_size=14pt,key_title_font_size=15pt))
    draw(PDF(string("publicscore-top50.pdf"), 5inch, 4inch),top50plot)
    writetable(string("publicscore-top50.csv"),dftop50)

    println("Plot: Significance analysis")
    #function ttest(m1,s1,m2,s2)
    function ttest(n,m,s)
        # one-sided Welch's t-test
        t = m/(s/sqrt(n))
        T = TDist(n-1)
        return 2.0* min(cdf(T,t),1-cdf(T,t))
    end

    function pvalues()
        ranking = sort(pbm)
        n = 8400
        k = 50
        losses = Array{Float64,1}[]
        pvals = Float64[]
        for i in 1:k
            labels = subs[pbl[ranking[i][2]].leader.id].labels
            push!(losses,map(cliplogloss,solution[private],labels[private]))
        end
        for i in 1:k
            push!(pvals,ttest(n,mean(losses[1]-losses[i]),std(losses[1]-losses[i])))
        end
        return pvals
    end

    function plotvals(pvals)
        df = DataFrame(x=[1:length(pvals)],y=pvals)
        pvalplot = Gadfly.plot(df, x="x", y="y", Geom.point,
                           Guide.yticks(ticks=[0.05,0.2,0.4,0.6,0.8]),
                           Guide.xticks(ticks=[2:length(pvals)]),
                           Coord.Cartesian(ymin=0.0,ymax=1.0),
                           Guide.xlabel("rank"), Guide.ylabel("p value"),
                           Theme(minor_label_font_size=15pt,major_label_font_size=15pt,
                                 key_label_font_size=14pt,key_title_font_size=15pt),
                                layer(x=[2,length(pvals)],y=[0.05,0.05],Geom.line,Theme(default_color=color("red"))))
        return df,pvalplot
    end

    pvals = pvalues()
    pvaldf,pvalplot = plotvals(pvals[1:10])
    draw(PDF(string("top10pvals.pdf"), 5inch, 4inch),pvalplot)
    writetable(string("top10pvals.csv"),pvaldf)
    # bonferroni correction for 9 comparisons
    pvalsbon = pvals[1:10]*9
    pvaldfbon,pvalplotbon = plotvals(pvalsbon)
    draw(PDF(string("top10pvalsbon.pdf"), 5inch, 4inch),pvalplotbon)
    writetable(string("top10pvalsbon.csv"),pvaldfbon)
end

function fresh_splits()
    #
    # Plot 3: Score deviations on new splits
    #
    println("Plot: Score distribution on fresh splits")

    function newsplit()
        rand(1:n,n)
        randperm = shuffle([1:n][private])
        split1 = bool(zeros(n))
        split2 = bool(zeros(n))
        split1[randperm[1:4200]]=true
        split2[randperm[4201:end]]=true

        # compute kaggle leaderboards for each team
        prl, pbl = lb(split1,split2)
        # compute best sub for each team
        prm, pbm = lbdiff(prl,pbl)
        # compute leaderboards according to parameter-free ladder
        prlpf,pblpf = lbpf(split1,split2)
        prmpf, pbmpf = lbdiff(prlpf,pblpf)

        top50 = sort(prm)[1:50]
        top50priv = [x for (x,_) in top50]
        top50pub = [pbl[t].current for (_,t) in top50]

        top50pf = sort(prmpf)[1:50]
        top50pfpriv = [x for (x,_) in top50pf]
        top50pfpub = [pblpf[t].current for (_,t) in top50pf]

        df1 = DataFrame(x=top50pub,y=top50priv)
        df2 = DataFrame(x=top50pfpub,y=top50pfpriv)

        return df1,df2
    end

    function plotscoresbars(means,stds,title="Top 50 scores on fresh split")
        return plot(x=means, y=means, ymin=means-stds,ymax=means+stds,Geom.point,Geom.errorbar,
                           Coord.Cartesian(ymin=0.182,ymax=0.202,xmin=0.182,xmax=0.202),
                           Guide.yticks(ticks=[0.19,0.20]),
                           Guide.xticks(ticks=[0.18,0.19,0.20]),
                           Guide.xlabel("Public score"), Guide.ylabel("Private score"),
                           Guide.title(title),
                           Theme(minor_label_font_size=15pt,major_label_font_size=15pt,
                                 key_label_font_size=13pt,key_title_font_size=15pt),
    layer(x=[0.182,0.202],y=[0.182,0.202],Geom.line,Theme(default_color=color("red"))))
    end

    # compute statistics across 20 independent iterations
    dfs = [newsplit() for _ in 1:20]
    dfs1 = [ x[1] for x in dfs]
    dfs2 = [ x[2] for x in dfs]
    scores1 = [ float([dfs1[i][j,1] for i in 1:20]) for j in 1:50]
    means1 = float([mean(scores1[i]) for i in 1:50])
    stds1 = float([std(scores1[i]) for i in 1:50])
    scores2 = [ float([dfs2[i][j,1] for i in 1:20]) for j in 1:50]
    means2 = float([mean(scores2[i]) for i in 1:50])
    stds2 = float([std(scores2[i]) for i in 1:50])
    plotbars1 = plotscoresbars(means1,stds1,"Kaggle top 50 scores on fresh splits")
    plotbars2 = plotscoresbars(means2,stds2,"Ladder top 50 scores on fresh splits")
    draw(PDF(string("top50kaggle-fresh.pdf"), 6inch, 4.5inch),plotbars1)
    draw(PDF(string("top50ladder-fresh.pdf"), 6inch, 4.5inch),plotbars2)
end

# process all teams and submission
solution = sub2labels(solution_file)
n = length(solution)
private = bool(privpart(solution_file))
# Kaggle used 5 bits of precision
kaggle_precision = 5
public = ~private

println("Building teams.")
team2id, id2team = build_teams()
teams = [ team for team in keys(team2id)]
println("Building submissions.")
allsubs, badsubs = build_allsubs()
subs = (String => Submission)[]
for sub in allsubs
    subs[sub.id] = sub
end
println("Done.")
