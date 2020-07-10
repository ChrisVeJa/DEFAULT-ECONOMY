###############################################################################
# [-] CODES FOR CONVERGENCE MODEL - NN - MODEL
###############################################################################
function try1()
   ############################################################
   #[1] Setting >> Solving
   ############################################################
   Params, hdef, uf = DefEcon.ModelSettings();
   Params = (
       r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
       θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969, ne = 251,
       ub = 0.0, lb = -0.4, tol = 1e-8, maxite = 1e3,
   );
   EconDef= DefEcon.SolveR(Params, hdef, uf);
   PolFun = EconDef.PolFun;
   ExtFea = EconDef.Ext;
   tsim = 30000;  tburn = 0.05;

   ############################################################
   #[2] Simulating
   ############################################################
   @label simul_start
   EconSim = DefEcon.ModelSim(Params, PF, Ext, nsim = tsim, burn = tburn);
   NDef = sum(EconSim.Sim[:,5]);
   PDef = 100*NDef/ tsim;
   display("Simulation finished, number of default events is $NDef with a frequency of $PDef")

   ############################################################
   #[3]	Neural Network
   ############################################################
   ϕf(x)= log1p(exp(x));
   opt  = Descent();
   Q    = 16;
   ns   = 2;
   norm = DefEcon.mynorm;
   vf   = EconSim.Sim[:, 6];
   st   = EconSim.Sim[:, 2:3];
   defstatus = EconSim.Sim[:, 5];

   vnd   = vf[defstatus .== 0]
   snd   = st[defstatus .== 0, :]
   NetWorkNDold = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
   Lnd(x, y) = Flux.mse(NetWorkNDold(x), y);
   NseTnD= (mhat = NetWorkNDold, loss = Lnd, opt = opt);
   VNDhat= DefEcon.NeuTra(vnd, snd, NseTnD, norm, Nepoch = 10);


   vd    = vf[defstatus .== 1];
   sd    = st[defstatus .== 1, :];
   NetWorkDold  = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
   Ld(x, y) = Flux.mse(NetWorkDold(x), y);
   NseTD = (mhat = NetWorkDold, loss = Ld, opt = opt);
   VDhat = DefEcon.NeuTra(vd, sd, NseTD, norm, Nepoch = 10);

   # ===========================================================================
   # (3) Iteration for convergence
   mytup = _unpack(Params, Ext, uf);
   DataND = VNDhat.Data;
   DataD  = VDhat.Data;
   PolFun1= SolverCon(DataND, NetWorkNDold, DataD, NetWorkDold, mytup);

   # [3.1] Verifying if the solution has no absorbent states
   value = sum(minimum(PolFun1.BP, dims=2) .== maximum(PolFun1.BP, dims=2));
   if value != 0
       @goto simul_start
   end
   ψnDold,ax1 = Flux.destructure(NetWorkNDold);
   ψDold, ax2 = Flux.destructure(NetWorkDold);
   repli= 1;
   EconSim1 = nothing;
   difΨ = 1;
   difΨ1= 1;
   difΨ2= 1;
   value= 1;
   NetWorkND = NetWorkNDold;
   NetWorkND = NetWorkDold;
   while difΨ > 1e-8 && repli < 400
        # ===========================================================================
        # (2) Defining Neural Networks structure
        # (2.1) No Default events
        @label newsimul
        ΨnD       = Flux.params(NetWorkNDold);
        Lnd(x, y) = Flux.mse(NetWorkND(x), y);
        # (2.1) Default events
        ΨD       = Flux.params(NetWorkDold);
        Ld(x, y) = Flux.mse(NetWorkD(x), y);
        # ========================================================================
        # [] New Simulation >> could have reduce grid
        EconSim1=
            DefEcon.ModelSim(Params, PolFun1, Ext, nsim = tsim, burn = tburn);
        # ========================================================================
        # [] Updating of the Neural Network
        # [] Data
        vf = EconSim1.Sim[:, 6];
        st = EconSim1.Sim[:, 2:3];
        defstatus =  EconSim1.Sim[:, 5];
        vnd = vf[defstatus .== 0];
        snd = st[defstatus .== 0, :];
        Ynd = DefEcon.mynorm(vnd);
        Snd = DefEcon.mynorm(snd);
        vd = vf[defstatus .== 1]
        sd = st[defstatus .== 1, :]
        Yd = DefEcon.mynorm(vd);
        Sd = DefEcon.mynorm(sd);
        dataND = Flux.Data.DataLoader(Snd', Ynd');
        dataD = Flux.Data.DataLoader(Sd', Yd');
        # [] Training No Default Neural Network
        Flux.Optimise.train!(Lnd, ΨnD, dataND, Descent()); # Now Ψ is updated
        # [] Training Default Neural Network
        Flux.Optimise.train!(Ld, ΨD, dataD, Descent()); # Now Ψ is updated

        # ========================================================================
        # [] Updating for new round

        # [] No Default Neural Network
        ψnD, modnD= Flux.destructure(NetWorkNDold);
        ψnD       = 0.9 * ψnD + 0.1 * ψnDold;
        difΨ1     = maximum(abs.(ψnD - ψnDold));
        NetWorkND = modnD(ψnD);
        ΨnD       = Flux.params(NetWorkND);
        Lnd(x, y) = Flux.mse(NetWorkND(x), y);
        DataND    = (vnd, snd);

        # [] Default Neural Network

        ψD, modD = Flux.destructure(NetWorkDold);
        ψD       = 0.9 * ψD + 0.1 * ψDold;
        difΨ2    = maximum(abs.(ψD - ψDold));
        NetWorkD = modD(ψD);
        ΨD       = Flux.params(NetWorkD);
        Ld(x, y) = Flux.mse(NetWorkD(x), y);
        DataD    = (vd, sd);

        # ========================================================================
        # [] New Solution for the model, see function
        PolFun1 = SolverCon(DataND, NetWorkND, DataD, NetWorkD, mytup);
        value= sum(minimum(PolFun1.BP, dims=2) .== maximum(PolFun1.BP, dims=2));
        if value != 0
            @goto newsimul
        end
        # ========================================================================
        ψnDold= ψnD;
        ψDold = ψD;
        NetWorkNDold = NetWorkND;
        NetWorkDold = NetWorkND;
        difΨ  = max(difΨ1,difΨ2);
        NDefaults = sum(defstatus);
        PorcDef = 100*NDefaults / tsim;
        display("Iteration $repli: with a difference of $difΨ")
        display("with a dif. $difΨ1 for No-Default, $difΨ2 for Default")
        display("Number of default events: $NDefaults");
        repli += 1;
   end
   if difΨ>1e-8
      display("Convergence not achieved")
   end
   return NetWorkND, NetWorkD;
end
