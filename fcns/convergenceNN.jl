function convergenceNN(EconSol,VFhat, totalburn)
   # ------------------------------------------
   # [0. No dependencies on Θ]
   # ------------------------------------------
   @unpack r,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = EconSol.Set.Params;
   _σ   = EconSol.Set.Params.σ;
   ϕfun(x)  = log1p(exp(x));
   Tsim = length(VFhat.Yhat);
   bgrid= EconSol.Sup.Bgrid;
   ygrid= EconSol.Sup.Ygrid;
   nx,ne= (length(ygrid), length(bgrid));
   pix  = EconSol.Sup.MarkMat;
   ydef = EconSol.Sup.Ydef;
   udef = EconSol.Set.UtiFun.(ydef,_σ);
   yb   = bgrid .+ ygrid';
   BB   = repeat(bgrid,1,nx);
   posb0= findmin(abs.(0 .- bgrid))[2];
   state= [repeat(bgrid,nx,1) repeat(ygrid,inner = (ne,1))];
   opt  = VFhat.Sett.Opti;
   # ------------------------------------------
   # [1. Setting the Loop]
   # ------------------------------------------

   # [1.1] Neural Network of the model
   DataOri = VFhat.DataOri;
   ψ, mod  = Flux.destructure(VFhat.Mhat);
   ψold    = ψ;
   NetWork = mod(ψ);
   Ψ       = Flux.params(NetWork);
   loss(x,y) = Flux.mse(NetWork(x),y);
   difΨ = 1;
   repli =1;

      # [1.2] Getting the extreme values
      maxV = maximum(DataOri[1]);
      minV = minimum(DataOri[1]);
      maxs = maximum(DataOri[2], dims=1);
      mins = minimum(DataOri[2], dims=1);

      # [1.3] Normalization of states
      staN  = (state .- 0.5* (maxs+mins))./(0.5*(maxs-mins));
      staDN = ([zeros(nx) ydef]  .- 0.5* (maxs+mins))./(0.5*(maxs-mins));

      # [1.4] Expected Value of Continuation:
      #        hat >> inverse normalization >> matrix form >> E[]
      vcpre = NetWork(staN');
      vcpre = (0.5*(maxV-minV) * vcpre) .+ 0.5*(maxV+minV);
      VC    = reshape(vcpre,ne,nx);
      EVC   = VC*pix';

      # [1.5] Expected Value of Default:
      #        hat >> inverse normalization >> matrix form >> E[]
      vdpre = convert(Array,NetWork(staDN'));
      vdpre = (0.5*(maxV-minV) * vdpre) .+ 0.5*(maxV+minV);
      VD    = repeat(vdpre,ne,1);
      EVD   = VD*pix';

      # [1.6] Income by issuing bonds
      #q  = EconSol.Sol.BPrice;
      D  = 1*(VD.>VC);
      q  = (1/(1+r))*(1 .-(D*pix'));
      qB = q.*bgrid;

      # [1.7] Policy function for bonds under continuation
      βEV = β*EVC;
      VC1 = Array{Float64,2}(undef,ne,nx)
      Bindex= Array{CartesianIndex{2},2}(undef,ne,nx)
      @inbounds for i in 1:ne
         cc         = yb[i,:]' .- qB;
         cc[cc.<0] .= 0;
         aux_u      = EconSol.Set.UtiFun.(cc,_σ) + βEV;
         VC1[i,:],Bindex[i,:] = findmax(aux_u,dims=1);
      end
      B1 = BB[Bindex];

      # [1.8] Value function of default
      βθEVC0 = β*θ*EVC[posb0,:];
      VD1    = βθEVC0'.+ (udef'.+ β*(1-θ)*EVD);

      # --------------------------------------------------------------
      # [1.9]. New Continuation Value, Default choice, price
      VF1= max.(VC1,VD1);
      D1 = 1*(VD1.>VC1);
      q1 = (1/(1+r))*(1 .-(D1*pix'));
      PolFun1  = DefaultEconomy.PolicyFunction(VF1,VC1,VD1,D1,B1,q1);

      # --------------------------------------------------------------
      # [1.10]. New Solution for the model
      EconSol1 = DefaultEconomy.ModelSolve(EconSol.Set,PolFun1,EconSol.Sup);
      #std(EconSol1.Sol.DebtPF[end,:])
      #maximum(EconSol1.Sol.DebtPF[end,:])
      #EconSol1.Sup.Bgrid[end]
      # --------------------------------------------------------------
      # [1.11]. New Simulation
      EconSim1 = DefaultEconomy.ModelSimulate(EconSol1,nsim=Tsim,burn=totalburn);

      #= [1.12] New data for training
      Y = DefaultEconomy.mynorm(EconSim1.Sim[:,6]);
      S = DefaultEconomy.mynorm(EconSim1.Sim[:,2:3]);
      data = Flux.Data.DataLoader(S',Y');

      # [1.13] New Parameters
      Flux.Optimise.train!(loss, Ψ, data, Descent()); # Now Ψ is updated

      ψ, mod  = Flux.destructure(NetWork);
      ψ       = 0.8*ψ + 0.2*ψold;
      NetWork = mod(ψ);
      Ψ       = Flux.params(NetWork);
      loss(x,y) = Flux.mse(NetWork(x),y);

      EconSol = EconSol1;
      DataOri = (EconSim1.Sim[:,6],EconSim1.Sim[:,2:3]);
      difΨ = maximum(abs.(ψ-ψold));
      display(difΨ);
      repli+=1;
=#
   return EconSol1, EconSim1, (vc₀= VC, vd₀= VD);
end
