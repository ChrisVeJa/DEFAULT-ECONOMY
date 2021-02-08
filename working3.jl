###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################
using Random, Distributions, Statistics, LinearAlgebra, StatsBase
using Parameters, Flux, ColorSchemes, Gadfly, JLD
using Cairo, Fontconfig, Tables, DataFrames, Compose
include("supcodes.jl");

############################################################
# [1]  FUNCTIONS TO BE USED
############################################################
    #...................................................
    # Updating policy function based on estimations
    update_solve(vr, vd, settings, params, uf,hf) = begin
        @unpack P, b, y = settings;
        @unpack r,β, θ, σrisk, nx, ne, fhat = params;
        p0 = findmin(abs.(0 .- b))[2]
        udef = uf(hf(y, fhat), σrisk);
        udef = repeat(udef', ne, 1);
        yb = b .+ y';
        # ----------------------------------------
        vf = max.(vr,vd);
        D  = 1 * (vd .> vr)
        bb   = repeat(b, 1, nx)
        vf1, vr1, vd1, D1, bp, q, eδD =
            value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf);
        return (vf = vf1, vr = vr1, vd = vd1, D = D1,
                bb = bb[bp], q = q, eδD = eδD)
    end

    #...................................................
    # Chebyshev polynomials
    cheby(x, d) = begin
        mat1 = Array{Float64,2}(undef, size(x, 1), d + 1)
        mat1[:, 1:2] = [ones(size(x, 1)) x]
        for i = 3:d+1
            mat1[:, i] = 2 .* x .* mat1[:, i-1] - mat1[:, i-1]
        end
        return mat1
    end

    #...................................................
    # Expansion of degree d
    myexpansion(vars::Tuple, d) = begin
        nvar = size(vars, 1)
        auxi = vars[1]
        numi = convert(Array, 0:size(auxi, 2)-1)'
        for i = 2:nvar
            n2v = size(auxi, 2)
            auxi2 = hcat([auxi[:, j] .* vars[i] for j = 1:n2v]...)
            numi2 = hcat([
                numi[:, j] .+ convert(Array, 0:size(vars[i], 2)-1)' for j = 1:n2v
            ]...)
            auxi = auxi2
            numi = numi2
        end
        xbasis = vcat(numi, auxi)
        xbasis = xbasis[:, xbasis[1, :].<=d]
        xbasis = xbasis[2:end, :]
        return xbasis
    end

    #...................................................
    # Estimation of neural networks
    NeuralEsti(NN, data, x, y1,y2) = begin
        lossf(x, y) = Flux.mse(NN(x), y)
        pstrain = Flux.params(NN)
        Flux.@epochs 10 Flux.Optimise.train!(lossf, pstrain, data, Descent())
        hatvrNN = ((1 / 2 * (NN(x')' .+ 1)) * (maximum(y1) - minimum(y1)) .+ minimum(y1))
        resNN = y2 - hatvrNN
        return hatvrNN, resNN
    end
    # ...................................................
    # Functions for plotting

    _myplot(ModelData,x, titlex) = Gadfly.plot(ModelData, x = "debt", y = x,
        color = "output", Geom.line, Guide.ylabel(""),
        Theme(background_color = "white",key_position = :right,
        key_title_font_size = 6pt, key_label_font_size = 6pt),
        Guide.xlabel("Debt (t)"), Guide.title(titlex));

    _myheatD(ModelData,yticks) = Gadfly.plot(ModelData, x = "debt",
        y = "output", color = "D",Geom.rectbin, Theme(background_color = "white",
        key_position=:bottom, plot_padding=[1mm],
        key_title_font_size =8pt, key_label_font_size = 8pt),
        Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"),
        Guide.yticks(ticks = yticks), Scale.color_discrete_manual("red","green"),
        Guide.colorkey(title = "", labels = ["Default","No Default"]),
        Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]));

    _myheatDS(ModelData,yticks) = Gadfly.plot(ModelData, x = "debt",
            y = "output", color = "D",Geom.rectbin, Theme(background_color = "white",
            key_position=:bottom, plot_padding=[1mm],
            key_title_font_size = 6pt, key_label_font_size = 6pt),
            Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"),
            Guide.yticks(ticks = yticks), Scale.color_discrete_manual("green","red","white"),
            Guide.colorkey(title = "", labels = ["Default","No Default",""]),
            Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]));

    _myheatPD(ModelData,yticks) = Gadfly.plot(ModelData, x = "debt",
            y = "output", color = "pd",Geom.rectbin, Theme(background_color = "white",
            key_position=:right, plot_padding=[1mm],
            key_title_font_size = 8pt, key_label_font_size = 8pt),
            Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"),
            Guide.yticks(ticks = yticks),
            Scale.color_continuous(colormap=Scale.lab_gradient("midnightblue",
            "white", "yellow1")),
            Guide.colorkey(title = "P(D=1)"),
            Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]));

    myplot(xs,set,par; simulation=false) = begin
        if ~simulation
            MoDel = hcat([vec(i) for i in xs]...);
            MoDel = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1)) MoDel];
            heads = [:debt, :output, :vf, :vr, :vd, :D, :b, :q, :pd];
            ModelData = DataFrame(Tables.table(MoDel, header = heads));
        else
            mataux = unique(xs.sim[:,2:end],dims=1); # only uniques
            heads = [:debt, :output, :yj, :b, :bj,:D,:vf, :vr, :vd, :q, :pd,:θ];
            MoDel = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1))]
            MatAux= [MoDel fill(NaN,size(MoDel,1),length(heads)-2)]; #whole grid
            MatAux[:,end-1].=0.5 ; # just for plotting
            for i in 1:size(mataux,1)
                l1 = findfirst(x -> x == mataux[i, 1], set.b)
                l2 = findfirst(x -> x == mataux[i, 3], set.y)
                MatAux[(l2-1)*par.ne+l1,3:end] = mataux[i,3:end]
            end
            ModelData = DataFrame(Tables.table(MatAux, header = heads));
            sort!(ModelData, :D);
        end
        yticks = round.(set.y, digits = 2);
        yticks = [yticks[1], yticks[6], yticks[11], yticks[16], yticks[end]];
        vars = ["vf" "vr" "vd" "b"]
        tvars =
            ["Value function" "Value of repayment" "Value of default" "PF for debt"]
        ppplot = Array{Any}(undef,6)
        for i = 1:length(vars)
            ppplot[i] = _myplot(ModelData,vars[i], tvars[i])
        end
        if ~simulation
            ppplot[5] = _myheatD(ModelData,yticks)
        else
            ppplot[5] = _myheatDS(ModelData,yticks)
        end
        ppplot[6] = _myheatPD(ModelData,yticks)
        return ppplot;
    end

    #...................................................
    # Residual analysis
    f1(x) = sqrt(mean(x .^ 2))                     # Square root of Mean Square Error
    f2(x) = maximum(abs.(x))                       # Maximum Absolute Deviation
    f3(x, y) = sqrt(mean((x ./ y) .^ 2)) * 100     # Square root of Mean Relative Square Error
    f4(x, y) = maximum(abs.(x ./ y)) * 100         # Maximum Relative deviation

############################################################
# [2] SETTING THE MODEL
############################################################
#...................................................
# Parameters
par = ( r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
        θ = 0.282,nx = 21, m = 3, μ = 0.0,fhat = 0.969,
        ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 1001);
#...................................................
# Utility function
uf(x, σrisk) = x .^ (1 - σrisk) / (1 - σrisk);

#...................................................
# Penalty function for defaulting
hf(y, fhat) = min.(y, fhat * mean(y));

############################################################
# [3] THE MODEL
############################################################

#...................................................
# Solving the model
polfun, set = Solver(par, hf, uf);
plotM1 = myplot(polfun,set,par);

#...................................................
# Simulating data from  the model
Nsim = 100000;
sim0 = ModelSim(par, polfun, set, hf, nsim = Nsim);
plotS1 = myplot(sim0,set,par, simulation=true);

#...................................................
# With different initial points
    sim1a = ModelSim(par, polfun, set, hf, nsim = Nsim, burn = 0, ini_st = [1 0 1]);
    sim1b = ModelSim(par, polfun, set, hf, nsim = Nsim, burn = 0, ini_st = [1 0 par.ne]);
    sim1c = ModelSim(par, polfun, set, hf, nsim = Nsim, burn = 0, ini_st = [par.nx 0 1]);
    sim1d = ModelSim(par, polfun, set, hf, nsim = Nsim, burn = 0, ini_st = [par.nx 0 par.ne]);
    plotS1a = myplot(sim1a,set,par, simulation=true);
    plotS1b = myplot(sim1b,set,par, simulation=true);
    plotS1c = myplot(sim1c,set,par, simulation=true);
    plotS1d = myplot(sim1d,set,par, simulation=true);
    #= It gives us the first problems:
        □ The number of unique observations are small
        □ Some yellow whenm they shoul dbe black
     =#

# ...................................................
# [Note] To be sure that the updating code is well,
#       I input the actual value functions and verify the
#       deviations in policy functions
# ...................................................
    hat_vr = polfun.vr
    hat_vd = polfun.vd
    trial1 = update_solve(hat_vr, hat_vd, set, par, uf,hf)
    difPolFun = max(maximum(abs.(trial1.bb - polfun.bb)), maximum(abs.(trial1.D - polfun.D)))
    display("After updating the difference in Policy functions is : $difPolFun")

# ##########################################################################################
#  [4] ESTIMATING VALUE OF REPAYMENT WITH FULL GRID
# ##########################################################################################
#...................................................
#  PRELIMINARIES
vr = vec(polfun.vr);
x = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1)) vr];
lx = minimum(x, dims = 1);
ux = maximum(x, dims = 1);
xst = 2 * (x .- lx) ./ (ux - lx) .- 1;
sst = xst[:,1:2];
vrt = xst[:,3];
d = 4;
result = Array{Any,2}(undef, 7, 2); # [fit, residual]

#...................................................
# Power Basis
mat = (sst[:, 1] .^ convert(Array, 0:d)', sst[:, 2] .^ convert(Array, 0:d)');
xs2 = myexpansion(mat, d);
β2 = (xs2' * xs2) \ (xs2' * vrt);
result[1, 1] = ((1 / 2 * ((xs2 * β2) .+ 1)) * (ux[3] - lx[3]) .+ lx[3]);
result[1, 2] = vr - result[1, 1];
# ...................................................
# Chebyshev Polynomials
mat = (cheby(sst[:, 1], d), cheby(sst[:, 2], d));
xs3 = myexpansion(mat, d) # remember that it start at 0
β3 = (xs3' * xs3) \ (xs3' * vrt);
result[2, 1] = ((1 / 2 * ((xs3 * β3) .+ 1)) * (ux[3] - lx[3]) .+ lx[3]);
result[2, 2] = vr - result[2, 1];

# ...................................................
# Neural Networks

    # ------------------------------------------------------------
    #  Interpolation
    np = 3;
    nbatch = 10;
    Lx = reshape(xst, par.ne, par.nx,3);
    gridx = (Lx[2:end,:,:] - Lx[1:end-1,:,:])/(np+1);
    Dx = [vec([kron(Lx[1:end-1,:,i], ones(np+1)) +  kron(gridx[:,:,i], 0:np);
            Lx[end,:,i]']) for i in 1:3];
    Dx = repeat(hcat(Dx...),nbatch,1);
    traindata = Flux.Data.DataLoader((Dx[:, 1:2]', Dx[:, 3]'));

    # ------------------------------------------------------------
    #  Models
    d = 16
    NN1 = Chain(Dense(2, d, softplus), Dense(d, 1));
    NN2 = Chain(Dense(2, d, tanh), Dense(d, 1));
    NN3 = Chain(Dense(2, d, elu), Dense(d, 1));
    NN4 = Chain(Dense(2, d, sigmoid), Dense(d, 1));
    NN5 = Chain(Dense(2, d, swish), Dense(d, 1));

    # ------------------------------------------------------------
    #  Training
    result[3, 1], result[3, 2] = NeuralEsti(NN1, traindata, sst, vr,vr);
    result[4, 1], result[4, 2] = NeuralEsti(NN2, traindata, sst, vr,vr);
    result[5, 1], result[5, 2] = NeuralEsti(NN3, traindata, sst, vr,vr);
    result[6, 1], result[6, 2] = NeuralEsti(NN4, traindata, sst, vr,vr);
    result[7, 1], result[7, 2] = NeuralEsti(NN5, traindata, sst, vr,vr);

# ...................................................
# Summarizing results
sumR = Array{Float32,2}(undef, 7, 4)
for i = 1:size(sumR, 1)
    sumR[i, :] =
        [f1(result[i, 2]) f2(result[i, 2]) f3(result[i, 2], vr) f4(result[i, 2], vr)]
end
#   Then the position ij of the matrix sumR is the f_j(residual_mdoel_i)

# ...................................................
# Plotting approximations

heads = [:debt,:output,:VR1,:VR2,:VR3,:VR4,:VR5,:VR6,:VR7,
        :Res1,:Res2,:Res3,:Res4,:Res5,:Res6,:Res7];
rr = vec(result);
modls = DataFrame(Tables.table([x[:,1:2] hcat(rr...)], header = heads))
models = ["Power series" "Chebyshev" "Softplus" "Tanh" "Elu" "Sigmoid" "Swish"]
plotE1 = Array{Any}(undef, 14); # [̂vr ϵ] by model
for i in eachindex(plotE1)
    plotE1[i] = _myplot(modls,heads[2+i], models[div(i-1,2)+1]);
end

# .............................................................
# [4.1] UPDATING POLICY FUNCTIONS BASED ON PREVIOUS ESTIMATIONS
resultU = Array{Any}(undef,7,4)
for i in 1:size(resultU)[1]
    hat_vrfit = reshape(modls[:, 2+i], par.ne, par.nx);
    polfunfit = update_solve(hat_vrfit, polfun.vd, set, par, uf,hf);
    resultU[i,1] = polfunfit;
    resultU[i,2] = myplot(polfunfit,set,par);
    simfit    = ModelSim(par, polfunfit, set, hf, nsim = Nsim);
    resultU[i,3] = simfit.sim;
    resultU[i,4] = myplot(simfit,set,par, simulation=true);
    pdef1 = round(100 * sum(simfit.sim[:, 7]) / Nsim; digits = 2);
    display("The model $i has $pdef1 percent of default");
end


################################################################
# [5] ESTIMATING VALUE OF REPAYMENT WITH SIMULATED DATA
#################################################################
# ...........................................................
# Preliminaries
    sim1 = ModelSim(par, polfun, set, hf, nsim = Nsim);
    x  = sim1.sim[:,[2,4,9]];
    vr = x[:,3];
    lx = minimum(x, dims = 1);
    ux = maximum(x, dims = 1);
    xst = 2 * (x .- lx) ./ (ux - lx) .- 1;
    sst = xst[:,1:2];
    vrt = xst[:,3];
    sswho = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1))];
    sswhole =  2 * (sswho .- lx[1:2]') ./ (ux[1:2]' - lx[1:2]') .- 1;
    vrwhole = vec(polfun.vr);
    resultA = Array{Any,2}(undef, 7, 2); # [fit, residual]
# ...........................................................
# Power Basis
    # Estimation
        d = 4;
        mat = (sst[:, 1] .^ convert(Array, 0:d)', sst[:, 2] .^ convert(Array, 0:d)');
        xs2 = myexpansion(mat, d);
        β2 = (xs2' * xs2) \ (xs2' * vrt);
    # Projection
        mat = (sswhole[:, 1] .^ convert(Array, 0:d)', sswhole[:, 2] .^ convert(Array, 0:d)');
        xs2 = myexpansion(mat, d);
        resultA[1, 1] = ((1 / 2 * ((xs2 * β2) .+ 1)) * (ux[3] - lx[3]) .+ lx[3])
        resultA[1, 2] = vrwhole - resultA[1, 1];
# ...........................................................
# Chebyshev Polynomials
    # Estimation
        mat = (cheby(sst[:, 1], d), cheby(sst[:, 2], d));
        xs3 = myexpansion(mat, d) # remember that it start at 0
        β3 = (xs3' * xs3) \ (xs3' * vrt);
    # Projection
        mat = (cheby(sswhole[:, 1], d), cheby(sswhole[:, 2], d));
        xs3 = myexpansion(mat, d)
        resultA[2, 1] = ((1 / 2 * ((xs3 * β3) .+ 1)) * (ux[3] - lx[3]) .+ lx[3]);
        resultA[2, 2] = vrwhole - resultA[2, 1];

# ...........................................................
# Neural Networks

    # ------------------------------------------------------------
    #  Models
    d = 16
    traindata = Flux.Data.DataLoader((sst', vrt'));
    NN1A = Chain(Dense(2, d, softplus), Dense(d, 1));
    NN2A = Chain(Dense(2, d, tanh), Dense(d, 1));
    NN3A = Chain(Dense(2, d, elu), Dense(d, 1));
    NN4A = Chain(Dense(2, d, sigmoid), Dense(d, 1));
    NN5A = Chain(Dense(2, d, swish), Dense(d, 1));

    # ------------------------------------------------------------
    #  Training + Projection
    resultA[3, 1], resultA[3, 2] = NeuralEsti(NN1A, traindata, sswhole, vr,vrwhole)
    resultA[4, 1], resultA[4, 2] = NeuralEsti(NN2A, traindata, sswhole, vr,vrwhole)
    resultA[5, 1], resultA[5, 2] = NeuralEsti(NN3A, traindata, sswhole, vr,vrwhole)
    resultA[6, 1], resultA[6, 2] = NeuralEsti(NN4A, traindata, sswhole, vr,vrwhole)
    resultA[7, 1], resultA[7, 2] = NeuralEsti(NN5A, traindata, sswhole, vr,vrwhole)

# ...........................................................
# Summarizing results
sumRA = Array{Float32,2}(undef, 7, 4)
for i = 1:size(sumRA, 1)
    sumRA[i, :] =
                [f1(resultA[i, 2]),
                f2(resultA[i, 2]),
                f3(resultA[i, 2], vrwhole),
                f4(resultA[i, 2], vrwhole)]
end
# ...........................................................
# Plotting approximations
heads = [:debt,:output,:VR1,:VR2,:VR3,:VR4,:VR5,:VR6,:VR7,
        :Res1,:Res2,:Res3,:Res4,:Res5,:Res6,:Res7];
rr    = vec(resultA);
modlsA = DataFrame(Tables.table([sswho[:,1:2] hcat(rr...)], header = heads))
models = ["Power series" "Chebyshev" "Softplus" "Tanh" "Elu" "Sigmoid" "Swish"]
plotE1A = Array{Any}(undef, 14); # [̂vr ϵ] by model
for i in eachindex(plotE1A)
    plotE1A[i] = _myplot(modlsA,heads[2+i], models[div(i-1,2)+1]);
end

# ...........................................................
# [5.1] UPDATING POLICY FUNCTIONS BASED ON PREVIOUS ESTIMATIONS
resultUA = Array{Any}(undef,7,4)
for i in 1:size(resultUA)[1]
    hat_vrfit = reshape(modlsA[:, 2+i], par.ne, par.nx);
    polfunfit = update_solve(hat_vrfit, polfun.vd, set, par, uf,hf);
    resultUA[i,1] = polfunfit;
    resultUA[i,2] = myplot(polfunfit,set,par);
    simfit    = ModelSim(par, polfunfit, set, hf, nsim = Nsim);
    resultUA[i,3] = simfit.sim;
    resultUA[i,4] = myplot(simfit,set,par, simulation=true);
    pdef1 = round(100 * sum(simfit.sim[:, 7]) / Nsim; digits = 2);
    display("The model $i has $pdef1 percent of default");
end


# =========================================================
#  [] Reporting results
# =========================================================

# setplot1 : Plots of policy functions
setplot1 = (plotM1,resultU[1,2],resultU[2,2],resultU[3,2],
            resultU[4,2], resultU[5,2],resultU[6,2],resultU[7,2],
            resultUA[1,2],resultUA[2,2],resultUA[3,2],resultUA[4,2],
            resultUA[5,2],resultUA[6,2],resultUA[7,2]);

setplot2 = (plotS1,resultU[1,2],resultU[2,4],resultU[3,4],
            resultU[4,4], resultU[5,4],resultU[6,4],resultU[7,4],
            resultUA[1,4],resultUA[2,4],resultUA[3,4],resultUA[4,4],
            resultUA[5,4],resultUA[6,4],resultUA[7,4]);

j = 1;
for i in 1:length(setplot1)
    xx = setplot1[i];
    titl = "./Plots/figurePOLFUN_" * string(i) *".png";
    draw(PNG(titl,24cm, 14cm),gridstack([xx[1] xx[2] xx[3];xx[4] xx[5] xx[6]]));
    xx = setplot2[i];
    titl = "./Plots/figureSIM_" * string(i) *".png";
    draw(PNG(titl,24cm, 14cm),gridstack([xx[1] xx[2] xx[3];xx[4] xx[5] xx[6]]));
    j +=1;
end

tit = "./Plots/figureResPOLFUN.png"
draw(PNG(tit,22cm, 22cm),gridstack([plotM1[2] plotE1[1] plotE1[2];
plotE1[3] plotE1[4] plotE1[5]; plotE1[6] plotE1[7] plotM1[2]]));
tit = "./Plots/figureResSIM.png"
draw(PNG(tit,22cm, 22cm),gridstack([plotM1[2] plotE1[8] plotE1[9];
plotE1[10] plotE1[11] plotE1[12]; plotE1[13] plotE1[14] plotM1[2]]));

@save("work0.jld")
