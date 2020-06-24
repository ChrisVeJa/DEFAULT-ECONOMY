function (EconSol,VFNeuF,VFhat,qhat;Nsim=100000, Burn=0.05,ns=2,Q=16)
	EconSolAux = DefaultEconomy.UpdateModel(EconSol,VFNeuF,VFhat,qhat);
	EconSimAux = DefaultEconomy.ModelSimulate(EconSolAux,nsim=Nsim,burn=Burn);
	VFNeuFAux  = (vf= EconSimAux.Simulation[:,6], q = EconSimAux.Simulation[:,7],states= EconSimAux.Simulation[:,2:3]);
	mhat_q     = Chain(Dense(ns,Q,ϕfun),Dense(Q,Q,ϕfun), Dense(Q,1));
	loss(x,y)  = Flux.mse(mhat_q(x),y);
	opt        = RADAM();
	NQChar     = DefaultEconomy.NeuralSettings(mhat_q,loss,opt);
	VFhatAux   = DefaultEconomy.neuralAprrox(VFNeuF[1],VFNeuF[3]);
	qhatAux    = DefaultEconomy.neuralAprrox(VFNeuF[2],VFNeuF[3],neuSettings=NQChar);
	return VFhatAux, qhatAux;
end
