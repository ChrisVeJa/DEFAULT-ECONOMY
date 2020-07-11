###############################################################################
# [-] CODES FOR CONVERGENCE MODEL - NN - MODEL
###############################################################################
function (Params, EconDef; tsim = 10000, tburn = 0.5)

   PolFun = EconDef.PolFun;
   ExtFea = EconDef.Ext;
   ############################################################
   #[] Simulating a new economy conditional on the
   #    solution
   ############################################################
   @label simul_start
   EconSim = DefEcon.ModelSim(Params, PolFun, ExtFea, nsim = tsim, burn = tburn);
   NDef = sum(EconSim.Sim[:,5]);
   PDef = round(100*NDef/ tsim; digits = 2);
   display("Simulation finished, with $NDef defaults event and a frequency of $PDef")

   ############################################################
   #[] Training a neural network conditional on the
   #    simulation
   ############################################################
   ϕf(x)= log1p(exp(x));
   opt = Descent();
   Q = 16;
   ns = 2;
   norm = DefEcon.mynorm;
   vf = EconSim.Sim[:, 6];
   st = EconSim.Sim[:, 2:3];
   defstatus = EconSim.Sim[:, 5];

   vnd = vf[defstatus .== 0]
   snd = st[defstatus .== 0, :]
   NetWorkND = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
   Lnd(x, y) = Flux.mse(NetWorkND(x), y);
   NseTnD = (mhat = NetWorkND, loss = Lnd, opt = opt);
   VNDhat = DefEcon.NeuTra(vnd, snd, NseTnD, norm, Nepoch = 10);


   vd = vf[defstatus .== 1];
   sd = st[defstatus .== 1, :];
   NetWorkD = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
   Ld(x, y) = Flux.mse(NetWorkD(x), y);
   NseTD = (mhat = NetWorkD, loss = Ld, opt = opt);
   VDhat = DefEcon.NeuTra(vd, sd, NseTD, norm, Nepoch = 10);

   ############################################################
   #[] Verifying if the neural network gives as new solution
   #   without absorbent states
   ############################################################
   mytup = DefEcon._unpack(Params, ExtFea, uf);
   DataND= VNDhat.Data;
   DataD = VDhat.Data;
   PolFun1= DefEcon.SolverCon(DataND, NetWorkNDold, DataD, NetWorkDold, mytup);
   value = sum(minimum(PolFun1.BP, dims=2) .== maximum(PolFun1.BP, dims=2));
   if value != 0
       @goto simul_start
   end
   return PolFun1,NetWorkND, NetWorkD;
end
