function convergNN(EconSol,VFhat, totalburn)
   # This part does not depend on Θ
   ϕfun(x)  = log1p(exp(x));
   totalsim = length(VFhat.Yhat);
   bgrid= EconSol.Sup.Bgrid;
   ygrid= EconSol.Sup.Ygrid;
   nx, ne = (length(ygrid), length(bgrid));
   pix  = EconSol.Sup.MarkMat;
   ydef = EconSol.Sup.Ydef;
   yb   = bgrid .+ ygrid';
   BB   = repeat(bgrid,1,nx);
   posb0= findmin(abs.(0 .- bgrid))[2];
   _σ   = EconSol.Set.Params.σ;
   udef = EconSol.Set.UtiFun.(ydef,_σ);
   q    = EconSol.Sol.BPrice;
   sta  = [repeat(bgrid,nx,1) repeat(ygrid,inner = (ne,1))];
   staN = DefaultEconomy.mynorm(sta);
   opt  = VFhat.Sett.Opti;

   # This part depends on Θ
   VFhatAux    = VFhat;
   EconSolAux  = EconSol;
   # Initial guess conditional to the OLD NN
   maxV, minV  = (maximum(VFhatAux.DataOri[1]), minimum(VFhatAux.DataOri[1]));
   vfpre = VFhatAux.Mhat(staN');
   vfpre = DefaultEconomy.mynorminv(vfpre,maxV,minV);
   VC    = reshape(vfpre,ne,nx);
   vdpre = convert(Array,VFhatAux.Mhat([zeros(nx) ydef]')');
   vdpre = DefaultEconomy.mynorminv(vdpre,maxV,minV);
   VD    = repeat(vdpre',ne,1);
   VO    = max.(VC,VD);
   D     = 1*(VD.>VC);

   # Solving the Model conditional the Guess for VF
   VO1,VC1,VD1,D1,Bprime1, q1 =
      DefaultEconomy.MaxBellman(EconSolAux.Set,VO,VC,VD,D,q,pix,posb0,udef,yb,bgrid,BB,
   );
   PolFun     = DefaultEconomy.PolicyFunction(VO1,VC1,VD1,D1,Bprime1,q1);
   EconSolAux = DefaultEconomy.ModelSolve(EconSolAux.Set,PolFun,EconSol.Sup);

   # New Simulation
   EconSimAux = DefaultEconomy.ModelSimulate(EconSolAux,nsim=totalsim,burn=totalburn);

   # Updating parameters of the Neural Network
   ps1, aux = Flux.destructure(VFhatAux.Mhat); ja = ps1;
   mhat = aux(ps1);
   ps = Flux.Params(mhat);
   ## New simulated data
   Y = DefaultEconomy.mynorm(EconSimAux.Sim[:,6]);
   S = DefaultEconomy.mynorm(EconSimAux.Sim[:,2:3]);
   data  = Flux.Data.DataLoader(S',Y');
   lossf(x,y) = Flux.mse(mhat(x),y);
   for d in data
      display(d)
      gs = gradient(ps) do
               lossf(d...)
            end
      Flux.update!(opt,ps,gs)
   end
   mhat2  = aux(ps)
   ps2, aux = Flux.destructure(mhat);
end
dd = Flux.Params(ps);
ps2, aux = Flux.destructure(mhat);
