function convergence(VFhat,NseT, Params, Ext, uf, tburn)
   # --------------------------------------------------------------------------
   # [0. No dependencies on Θ]
   # --------------------------------------------------------------------------
   # Settings
   mytup = _unpack(Params, Ext, uf);
   # Simulation
   tsim = length(VFhat.Hat);
   # Neural Network
   Data = VFhat.Data;
   ϕf(x)= log1p(exp(x));
   opt  = Descent();
   ψ, mod  = Flux.destructure(VFhat.Mhat);
   ψold    = ψ;
   NetWork = mod(ψ);
   Ψ       = Flux.params(NetWork);
   loss(x,y) = Flux.mse(NetWork(x),y);
   repli = 1
   PolFun1=  nothing;
   Ext1    = Ext;
   # --------------------------------------------------------------------------
   # [Displaying the Loop]
   # --------------------------------------------------------------------------
   while repli<401
      # ----------------------------------------------------------------------
      # [2.1]. New Solution for the model
      PolFun1 = _solver(Data,mytup, NetWork);
      val1    = minimum(PolFun1.BP[end,:]);
      val2    = maximum(PolFun1.BP[end,:]);
      mytup1  = mytup;

      while val1 == val2
         # We need to change the grid for B both
         # in value as in points of grid
         Ext1 = (ygrid = Ext1.ygrid, bgrid= Ext1.bgrid[1:end-1], ydef= Ext1.ydef, P= Ext1.P);
         mytup1 = _unpack(Params, Ext1, uf);
         PolFun1 = _solver(Data,mytup1, NetWork);
         val1    = minimum(PolFun1.BP[end,:]);
         val2    = maximum(PolFun1.BP[end,:]);
      end
      if length(Ext1.bgrid) <= 1
         return display("Please, try with another initial simulation");
      end
      # ----------------------------------------------------------------------
      # [2.2]. New Simulation >> could have reduce grid
      EconSim1 = DefEcon.ModelSim(Params,PolFun1,Ext1,nsim=tsim,burn=tburn);

      # ----------------------------------------------------------------------
      # [2.3] Updating of the Neural Network

      # [2.3.1] New Data for Training
      Y = DefEcon.mynorm(EconSim1.Sim[:,6]);
      S = DefEcon.mynorm(EconSim1.Sim[:,2:3]);
      data = Flux.Data.DataLoader(S',Y');

      # [2.3.2] New Parameters
      Flux.Optimise.train!(loss, Ψ, data, Descent()); # Now Ψ is updated

      # ----------------------------------------------------------------------
      # [2.4] Updating for new round
      ψ, mod  = Flux.destructure(NetWork);
      ψ       = 0.9*ψ + 0.1*ψold;
      NetWork = mod(ψ);
      Ψ       = Flux.params(NetWork);
      loss(x,y) = Flux.mse(NetWork(x),y);
      Data = (EconSim1.Sim[:,6],EconSim1.Sim[:,2:3]);
      difΨ = maximum(abs.(ψ-ψold));
      display(difΨ);
      repli+=1;
   end
   return PolFun1, EconSim1;
end

function _unpack(Params, Ext, uf)
   @unpack r, β, θ, σrisk = Params;
   @unpack bgrid, ygrid, ydef, P = Ext;
   nx = length(ygrid);
   ne = length(bgrid);
   yb = bgrid .+ ygrid';
   BB = repeat(bgrid,1,nx);
   p0 = findmin(abs.(0 .- bgrid))[2];
   state= [repeat(bgrid,nx,1) repeat(ygrid,inner = (ne,1))];
   udef = uf.(ydef,σrisk);
   mytup = (
      r=r, β= β,θ= θ, σrisk= σrisk, bgrid= bgrid, ygrid= ygrid,
      nx= nx, ne= ne, P= P, ydef= ydef, udef= udef, yb= yb, BB= BB,
      p0= p0, state= state, utf = uf,
   )
   return mytup;
end

function _solver(Data,mytup, NetWork)
   @unpack σrisk,bgrid, ygrid, nx, ne, P, ydef, udef, yb, BB, p0,
      state, utf, β,r , θ= mytup;

   maxV = maximum(Data[1]);
   minV = minimum(Data[1]);
   maxs = maximum(Data[2], dims=1);
   mins = minimum(Data[2], dims=1);

   # [1.3] Normalization of states
   staN  = (state .- 0.5* (maxs+mins))./(0.5*(maxs-mins));
   staDN = ([zeros(nx) ydef]  .- 0.5* (maxs+mins))./(0.5*(maxs-mins));

   # [1.4] Expected Value of Continuation:
   #        hat >> inverse normalization >> matrix form >> E[]
   vcpre = NetWork(staN');
   vcpre = (0.5*(maxV-minV) * vcpre) .+ 0.5*(maxV+minV);
   VC    = reshape(vcpre,ne,nx);
   EVC   = VC*P';

   # [1.5] Expected Value of Default:
   #        hat >> inverse normalization >> matrix form >> E[]
   vdpre = convert(Array,NetWork(staDN'));
   vdpre = (0.5*(maxV-minV) * vdpre) .+ 0.5*(maxV+minV);
   VD    = repeat(vdpre,ne,1);
   EVD   = VD*P';

   # [1.6] Income by issuing bonds
   #q  = EconSol.Sol.BPrice;
   D  = 1*(VD.>VC);
   q  = (1/(1+r))*(1 .-(D*P'));
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
   βθEVC0 = β*θ*EVC[p0,:];
   VD1    = βθEVC0'.+ (udef'.+ β*(1-θ)*EVD);

   # --------------------------------------------------------------
   # [1.9]. New Continuation Value, Default choice, price
   VF1= max.(VC1,VD1);
   D1 = 1*(VD1.>VC1);
   q1 = (1/(1+r))*(1 .-(D1*P'));
   PolFun1  = (VF = VF1, VC =VC1, VD =VD1, D= D1, BP= B1, Price = q1);
   return PolFun1;
end
