###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################
using Random, Distributions, Statistics, LinearAlgebra,
        StatsBase, Parameters, Flux, ColorSchemes, Gadfly, Caito, Fontconfig,
        Tables, DataFrames, Compose
include("supcodes.jl");

############################################################
# [1]  FUNCTIONS TO BE USED
############################################################
# ----------------------------------------------------------
# [1.a] Choosing unique states from the simulated data
# ----------------------------------------------------------
    myunique(data) = begin
        dataTu = [Tuple(data[i,:])  for i in 1:size(data)[1]]
        dataTu = unique(dataTu)
        dataTu = [[dataTu[i]...]' for i in 1:length(dataTu)]
        data   = [vcat(dataTu...)][1]
        return data
    end
# ----------------------------------------------------------
# [1.b]  Normalization:
#     ̃x = (x - 1/2(xₘₐₓ + xₘᵢₙ)) /(1/2(xₘₐₓ - xₘᵢₙ))
# ----------------------------------------------------------
    mynorm(x) = begin
        ux = maximum(x, dims=1)
        lx = minimum(x, dims=1)
        nx = (x .- 0.5*(ux+lx)) ./ (0.5*(ux-lx))
        return nx, ux, lx
    end
# ----------------------------------------------------------
# [1.c]  Training of neural network
# ----------------------------------------------------------
    mytrain(NN,data) = begin
        lossf(x,y) = Flux.mse(NN(x),y);
        traindata  = Flux.Data.DataLoader(data)
        pstrain = Flux.params(NN)
        Flux.@epochs 10 Flux.Optimise.train!(lossf, pstrain, traindata, Descent())
    end
# ----------------------------------------------------------
# [1.d]  Policy function conditional on expected values
# ----------------------------------------------------------
    update_solve(hat_vr, hat_vd, settings,params,uf) = begin
        # Starting the psolution of the model
        @unpack P, b,y = settings
        @unpack r, β, ne, nx, σrisk,θ = params
        p0 = findmin(abs.(0 .- b))[2]
        udef = repeat(settings.udef', ne, 1)
        hat_vf = max.(hat_vr,hat_vd)
        hat_D  = 1 * (hat_vd .> hat_vr)
        evf1 = hat_vf * P'
        evd1 = hat_vd * P'
        eδD1 = hat_D  * P'
        q1   = (1 / (1 + r)) * (1 .- eδD1) # price
        qb1  = q1 .* b
        βevf1= β*evf1
        vrnew = Array{Float64,2}(undef,ne,nx)
        cc1    = Array{Float64,2}(undef,ne,nx)
        bpnew = Array{CartesianIndex{2},2}(undef, ne, nx)
        yb    = b .+ y'
        @inbounds for i = 1:ne
            cc1 = yb[i, :]' .- qb1
            cc1 = max.(cc1,0)
            aux_u = uf.(cc1, σrisk) + βevf1
            vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
        end
        bb    = repeat(b, 1, nx)
        bb    = bb[bpnew]
        evaux = θ * evf1[p0, :]' .+  (1 - θ) * evd1
        vdnew = udef + β*evaux
        vfnew = max.(vrnew, vdnew)
        Dnew  = 1 * (vdnew .> vrnew)
        eδD   = Dnew  * P'
        qnew  = (1 / (1 + r)) * (1 .- eδD)
        return (vf = vfnew, vr = vrnew, vd = vdnew, D = Dnew, bb =  bb, q = qnew, bp = bpnew)
    end

############################################################
# [2] SETTING
############################################################
params = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
        θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969,
        ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 251);
uf(x, σrisk)= x.^(1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# [3] THE MODEL
############################################################
# ----------------------------------------------------------
# [3.a] Solving the model
# ----------------------------------------------------------
polfun, settings = Solver(params, hf, uf);
MoDel = [vec(polfun[i]) for i in 1:6]
MoDel = [repeat(settings.b,params.nx) repeat(settings.y,inner= (params.ne,1))  hcat(MoDel...)]
heads = [:debt, :output, :vf, :vr, :vd, :D, :b, :q]
ModelData = DataFrame(Tables.table(MoDel, header =heads))
# ----------------------------------------------------------
# [3.b] Plotting results from the Model
# ----------------------------------------------------------
p0 = Gadfly.plot(ModelData, x = "debt", y = "vf", color = "output", Geom.line,
        Theme(background_color = "white", key_position = :right ,key_title_font_size = 6pt,key_label_font_size = 6pt))
p1 = Gadfly.plot(ModelData, x = "debt", y = "vr", color = "output", Geom.line,
        Theme(background_color = "white", key_position = :right ,key_title_font_size = 6pt,key_label_font_size = 6pt))
p2 = Gadfly.plot(ModelData, x = "debt", y = "vd", color = "output", Geom.line,Theme(background_color = "white", key_position = :none))
p3 = Gadfly.plot(ModelData, x = "debt", y = "b", color = "output", Geom.line,Theme(background_color = "white" ,key_position = :none))
p4 = Gadfly.plot(ModelData, x =  "debt", y = "output", color = "D",Geom.rectbin,
     Scale.color_discrete_manual("yellow", "black"),Theme(background_color = "white"))
Gadfly.gridstack([p1 p2; p3 p4])
#savefig("./Figures/heatD0.png")

# ----------------------------------------------------------
# [3.c] Simulating data from  the model
# ----------------------------------------------------------
econsim0 = ModelSim(params,polfun, settings,hf, nsim=100000);
data0 = myunique(econsim0.sim)
DDsimulated= fill(NaN,params.ne*params.nx, 3)
DDsimulated[:,1:2] = [repeat(settings.b,params.nx) repeat(settings.y,inner= (params.ne,1))]
for i in 1: size(data0,1)
    posb = findfirst(x -> x == data0[i,2], settings.b)
    posy = findfirst(x -> x == data0[i,8], settings.y)
    DDsimulated[(posy-1)*params.ne+posb ,3] = data0[i,5]
end
heads = [:debt, :output, :D]
DDsimulated = DataFrame(Tables.table(DDsimulated, header =heads))
p5 = Gadfly.plot(DDsimulated, x =  "debt", y = "output", color = "D",Geom.rectbin,
     Scale.color_discrete_manual("white","black","yellow"),Theme(background_color = "white"))
pdef = round(100 * sum(econsim0.sim[:, 5])/ 100000; digits = 2);
display("Simulation finished, with frequency of $pdef default events");
#savefig("./Figures/heatmap_D.png");


#= It gives us the first problems:
    □ The number of unique observations are small
    □ Some yellow whenm they shoul dbe black
 =#
set_default_plot_size(8cm, 12cm)
heat1 = Gadfly.vstack(p4, p5)



# **********************************************************
# [Note] To be sure that the updating code is well,
#       I input the actual value functions and verify the
#       deviations in policy functions
# **********************************************************
hat_vr = polfun.vr
hat_vd = polfun.vd
trial1 = update_solve(hat_vr, hat_vd, settings,params,uf)
difPolFun = max(maximum(abs.(trial1.bb - polfun.bb)),maximum(abs.(trial1.D - polfun.D)))
display("After updating the difference in Policy functions is : $difPolFun")

# ##########################################################
# [4] NEURAL NETWORKS WITH FULL INFORMATION
# ##########################################################

# ***************************************
# [4.a] Value of repayment
# ***************************************

# Normalizing
    ss  = [repeat(settings.b,params.nx) repeat(settings.y,inner= (params.ne,1))]   # bₜ, yₜ
    vr  = vec(polfun.vr)   # vᵣ
    vrtilde, uvr, lvr = mynorm(vr);
    sstilde, uss, lss = mynorm(ss);
    data = (Array{Float32}(sstilde'), Array{Float32}(vrtilde'));

NNR1 = Chain(Dense(2, 16, softplus), Dense(16, 1));
NNR2 = Chain(Dense(2, 16, tanh), Dense(16, 1));
NNR3 = Chain(Dense(2, 32, relu), Dense(32, 16,softplus), Dense(16,1));
NNR4 = Chain(Dense(2, 32, relu), Dense(32, 16,tanh), Dense(16,1));

mytrain(NNR1,data);
mytrain(NNR2,data);
mytrain(NNR3,data);
mytrain(NNR4,data);

vrtildehat = [NNR1(sstilde')' NNR2(sstilde')' NNR3(sstilde')' NNR4(sstilde')']
VRhat = vrtildehat.*(0.5*(uvr-lvr)) .+ 0.5*(uvr+lvr)
NNresults = [ss vec(polfun[2]) VRhat]
heads = [:debt, :output, :vr,:NN1, :NN2, :NN3, :NN4]
NNhat = DataFrame(Tables.table(NNresults, header =heads))



set_default_plot_size(12cm, 12cm)
p1 = Gadfly.plot(NNhat, x = "debt", y = "vr", color = "output", Geom.line,
        Geom.line,Theme(background_color = "white"))
p2 = Gadfly.plot(NNhat, x = "debt", y = "NN1", color = "output", Geom.line,
        Geom.line,Theme(background_color = "white", key_position = :none))
p3 = Gadfly.plot(NNhat, x = "debt", y = "NN2", color = "output", Geom.line,
        Geom.line,Theme(background_color = "white", key_position = :none))
p4 = Gadfly.plot(NNhat, x = "debt", y = "NN3", color = "output", Geom.line,
        Geom.line,Theme(background_color = "white", key_position = :none))
p5 = Gadfly.plot(NNhat, x = "debt", y = "NN4", color = "output", Geom.line,
        Geom.line,Theme(background_color = "white", key_position = :none))
h2 = Gadfly.gridstack([p2 p3; p4 p5])



resid = vr .-  VRhat;
rest = DataFrame(Tables.table([ss resid], header =[:debt,:y,:error1, :error2, :error3, :error4]))
p1 = plot(rest, x = "debt", y = "error1",color="y", Geom.line,
        Theme(background_color = "white"))
p2 = plot(rest, x = "debt", y = "error2",color="y", Geom.line,
        Theme(background_color = "white", key_position = :none))
p3 = plot(rest, x = "debt", y = "error3",color="y", Geom.line,
        Theme(background_color = "white", key_position = :none))
p4 = plot(rest, x = "debt", y = "error4",color="y", Geom.line,
        Theme(background_color = "white", key_position = :none))
h3 = gridstack([p1 p2; p3 p4])
