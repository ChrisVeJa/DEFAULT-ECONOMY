function convergenceNN(EconSol,VFhat, tburn)
   # --------------------------------------------------------------------------
   # [0. No dependencies on Θ]
   # --------------------------------------------------------------------------
   ϕfun(x) = log1p(exp(x));
   opt   = VFhat.Sett.Opti;
   mytup = _unpack(EconSol);
   # --------------------------------------------------------------------------
   # [1. Setting the Loop]
   # --------------------------------------------------------------------------

   # [1.1] Neural Network of the model
   DataOri = VFhat.DataOri;
   ψ, mod  = Flux.destructure(VFhat.Mhat);
   ψold    = ψ;
   NetWork = mod(ψ);
   Ψ       = Flux.params(NetWork);
   loss(x,y) = Flux.mse(NetWork(x),y);
   repli = 1

   # --------------------------------------------------------------------------
   # [2. Displaying the Loop]
   # --------------------------------------------------------------------------
   while repli < 400
      # ----------------------------------------------------------------------
      # [2.1]. New Solution for the model
      PolFun1, indexes  = _solver(DataOri,mytup, NetWork);
      Support  = DefaultEconomy.Supporting(EconSol.Sup.Bgrid[indexes],
                  EconSol.Sup.Ygrid,EconSol.Sup.Ydef,EconSol.Sup.MarkMat);
      EconSol1 = DefaultEconomy.ModelSolve(EconSol.Set,PolFun1,Support);

      # ----------------------------------------------------------------------
      # [2.2]. New Simulation >> could have reduce grid
      EconSim1 = DefaultEconomy.ModelSimulate(EconSol1,nsim=tsim,burn=tburn);

      # ----------------------------------------------------------------------
      # [2.3] Updating of the Neural Network

      # [2.3.1] New Data for Training
      Y = DefaultEconomy.mynorm(EconSim1.Sim[:,6]);
      S = DefaultEconomy.mynorm(EconSim1.Sim[:,2:3]);
      data = Flux.Data.DataLoader(S',Y');

      # [2.3.2] New Parameters
      Flux.Optimise.train!(loss, Ψ, data, Descent()); # Now Ψ is updated

      # ----------------------------------------------------------------------
      # [2.4] Updating for new round
      ψ, mod  = Flux.destructure(NetWork);
      ψ       = 0.8*ψ + 0.2*ψold;
      NetWork = mod(ψ);
      Ψ       = Flux.params(NetWork);
      loss(x,y) = Flux.mse(NetWork(x),y);
      DataOri = (EconSim1.Sim[:,6],EconSim1.Sim[:,2:3]);
      difΨ = maximum(abs.(ψ-ψold));
      display(difΨ);
      repli+=1;
   end
   #return EconSol1, EconSim1, (vc₀= VC, vd₀= VD);
end

function _unpack(EconSol)
   r    = EconSol.Set.Params.r;
   β    = EconSol.Set.Params.β;
   θ    = EconSol.Set.Params.θ;
   σrisk= EconSol.Set.Params.σrisk;
   tsim = length(VFhat.Yhat);
   bgrid= EconSol.Sup.Bgrid;
   ygrid= EconSol.Sup.Ygrid;
   nx,ne= (length(ygrid), length(bgrid));
   pix  = EconSol.Sup.MarkMat;
   ydef = EconSol.Sup.Ydef;
   udef = EconSol.Set.UtiFun.(ydef,σrisk);
   yb   = bgrid .+ ygrid';
   BB   = repeat(bgrid,1,nx);
   posb0= findmin(abs.(0 .- bgrid))[2];
   state= [repeat(bgrid,nx,1) repeat(ygrid,inner = (ne,1))];

   mytup = (
      r=r, β= β,θ= θ, σrisk= σrisk,  tsim= tsim, bgrid= bgrid, ygrid= ygrid,
      nx= nx, ne= ne, pix= pix, ydef= ydef, udef= udef, yb= yb, BB= BB,
      posb0= posb0, state= state, utf = EconSol.Set.UtiFun,
   )
   return mytup;
end

function _solver(DataOri,mytup, NetWork)
   @unpack σrisk, tsim,bgrid, ygrid, nx, ne, pix, ydef, udef, yb, BB, posb0,
      state, utf, β,r , θ= mytup;

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
      aux_u      = utf.(cc,σrisk) + βEV;
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

   index  = findall(x -> x > 1e-10,std(B1, dims=2));
   indexes = [index[i][1] for i in index]
   VF2= VF1[indexes,:];
   VC2= VC1[indexes,:];
   VD2= VD1[indexes,:];
   D2 = D1[indexes,:];
   B2 = B1[indexes,:];
   q2 = q1[indexes,:];
   PolFun1  = DefaultEconomy.PolicyFunction(VF2,VC2,VD2,D2,B2,q2);
   return PolFun1, indexes;
end
