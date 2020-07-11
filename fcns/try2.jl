function try2(PolFun,NetWorkNDold, NetWorkDold, Params, ExtFea, uf, tsim, tburn)
    mytup = DefEcon._unpack(Params, ExtFea, uf)
    ψnDold, x1x = Flux.destructure(NetWorkNDold)
    ψDold, x2x = Flux.destructure(NetWorkDold)
    NetWorkND = nothing
    NetWorkD = nothing
    PolFun1 = nothing
    difΨ = 1
    difΨ1 = 1
    difΨ2 = 1
    value = 1
    repli = 1
    EconSim1 = nothing
    while difΨ > 1e-8 && repli < 400
        # ======================================================================
        # Taking the structure from the initial neural network:
        #   It is defining neural network as copies of the original ones
        # ======================================================================
        @label newsimul
        NetWorkND = NetWorkNDold
        ΨnD = Flux.params(NetWorkND)
        Lnd(x, y) = Flux.mse(NetWorkND(x), y)
        NetWorkD = NetWorkDold;
        ΨD = Flux.params(NetWorkD)
        Ld(x, y) = Flux.mse(NetWorkD(x), y)
        # ======================================================================
        # New Simulation
        #   We know that this Policy Function gives us a stable simulation
        # ======================================================================
        @label dosim
        EconSim1 =
            DefEcon.ModelSim(Params, PolFun, ExtFea, nsim = tsim, burn = tburn)


        # ======================================================================
        # Taking the simulation to train our model again
        #   We are training the parameters and model define at @label newsimul
        # ======================================================================
        vf = EconSim1.Sim[:, 6]
        st = EconSim1.Sim[:, 2:3]
        dfs = EconSim1.Sim[:, 5]
        NDef = sum(dfs)
        Porc = round(100 * NDef/ tsim, digits=2)
        display("% of default events: $Porc")
        if Porc == 0.0
            @goto dosim
        end



        vnd = vf[dfs.==0]
        snd = st[dfs.==0, :]
        Ynd = DefEcon.mynorm(vnd)
        Snd = DefEcon.mynorm(snd)
        vd = vf[dfs.==1]
        sd = st[dfs.==1, :]
        Yd = DefEcon.mynorm(vd)
        Sd = DefEcon.mynorm(sd)
        dataND = Flux.Data.DataLoader(Snd', Ynd')
        dataD = Flux.Data.DataLoader(Sd', Yd')

        # [] Training No Default Neural Network
        #    Now ΨnD, NetWorkND is updated
        Flux.Optimise.train!(Lnd, ΨnD, dataND, Descent())
        # [] Training Default Neural Network
        #    Now ΨD, NetWorkD is updated
        Flux.Optimise.train!(Ld, ΨD, dataD, Descent()) # Now Ψ is updated

        # ======================================================================
        # Updating Parameters and defining the Neural Networks
        # We make this because our updating have a persistence parameter
        # ======================================================================

        # [] non Default Neural Network
        ψnD, modnD = Flux.destructure(NetWorkND)
        ψnD = 0.1 * ψnD + 0.9 * ψnDold
        difΨ1 = maximum(abs.(ψnD - ψnDold))
        NetWorkND = modnD(ψnD)
        ΨnD = Flux.params(NetWorkND)
        Lnd(x, y) = Flux.mse(NetWorkND(x), y)
        DataND = (vnd, snd)

        # [] Default Neural Network
        ψD, modD = Flux.destructure(NetWorkD)
        ψD = 0.1 * ψD + 0.9 * ψDold
        difΨ2 = maximum(abs.(ψD - ψDold))
        NetWorkD = modD(ψD)
        ΨD = Flux.params(NetWorkD)
        Ld(x, y) = Flux.mse(NetWorkD(x), y)
        DataD = (vd, sd)

        # ======================================================================
        # New solution >> veryfing that the simulation -- training  leads
        #                 an new PolFun witpuh absorbent states
        #              >> Otherwise, we return to make everythin before update
        # ======================================================================
        PolFun1 = DefEcon.SolverCon(DataND, NetWorkND, DataD, NetWorkD, mytup)
        value =
            sum(minimum(PolFun1.BP, dims = 2) .== maximum(PolFun1.BP, dims = 2))
        display(value)
        if value != 0
            @goto newsimul
        end

        # ======================================================================
        # Updating for next round
        # ======================================================================
        PolFun = PolFun1;
        ψnDold = ψnD
        ψDold = ψD
        NetWorkNDold = NetWorkND
        NetWorkDold = NetWorkD
        difΨ = max(difΨ1, difΨ2)
        display("Iteration $repli: with a difference of $difΨ")
        display("with a dif. $difΨ1 for No-Default, $difΨ2 for Default")
        repli += 1
    end
    if difΨ > 1e-8
        display("Convergence not achieved")
    end
    return PolFun1,NetWorkND, NetWorkD;
end
