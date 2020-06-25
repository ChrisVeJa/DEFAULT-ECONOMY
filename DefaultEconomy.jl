#= #############################################################################
 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
The following module has all the functions used in the master.jl
Written by: Christian Velasquez (velasqcb@bc.edu)
Dont hesitate in send any comment
=# #############################################################################

module DefaultEconomy
using   Random, Distributions, Statistics, LinearAlgebra, Plots, StatsBase,
		Parameters, Flux;
## #############################################################################
#  DEFAULT STRUCTURES
#  #############################################################################

#= -----------------------------------------------------------------------------
(1.1) Required structure to solve the model:
----------------------------------------------------------------------------- =#

# [1.1.1] Setting of the model
mutable struct ModelSettings
	Params::NamedTuple;    # (r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol,maxite)
	UtiFun::Function;      # Utility Function
	DefFun::Function;      # Default Loss function
end
# [1.1.2] Policy functions
struct  PolicyFunction
	ValFun::Array{Float64};  # Value Function
	ValNoD::Array{Float64};  # Value Function under No Default
	ValDef::Array{Float64};  # Value Function of Default
	DefCho::Array{Float64};	 # Default Choice (1:= Default)
	DebtPF::Array{Float64};	 # Issued Debt for t+1
	BPrice::Array{Float64};	 # Price of new issued debt
end
# [1.1.3] Additional feautures of the model
struct Supporting
	Bgrid;	# Grid for debt
	Ygrid;	# Grid for y
	Ydef;	# Output under default
	MarkMat;# Transition matrix
end

# [1.1.4] Solution of the model
struct ModelSolve
	Set::ModelSettings;		# Settings of the Model
	Sol::PolicyFunction;	# Solution of the Model
	Sup::Supporting;		# Some support charact
end

#= -----------------------------------------------------------------------------
(1.2) Required structures for simulation:
----------------------------------------------------------------------------- =#
# [1.2.1] Model Simulation Results
struct ModelSim
	Set::NamedTuple;		# Settings of the simulation
	Mod::ModelSolve;		# Model in which the simulation is based
	Sim::Array{Float64};	# Simulation matrix
	Nam::String;			# Head names of the simulation columns
end

#= -----------------------------------------------------------------------------
(1.3) Structures used in Neural Network :
----------------------------------------------------------------------------- =#
# [1.3.1] Settings for the neural network
struct NeuralSettings
	Mhat;	# Neural Network
	Loss;	# Loss function
	Opti;	# Optimizer type
end

# [1.3.2] Neural Network results
mutable struct NeuralApprox
	Data::Tuple;			# Data inputs
	Yhat::Array;			# Fit after training
	Mhat::Any;				# Neural Network
	Sett::NeuralSettings; 	# Initial settings
end

## #############################################################################
#  (2) FUNCTIONS TO FIND THE SOLUTION OF THE MODEL
################################################################################

# [2.1] Default settings
function ModelSettings()
	Params = (r = 0.017,σ = 2.0,ρ=0.945,η=0.025,β=0.953,θ=0.282,nx=21,m=3,μ=0.0,fhat=0.969,ne=251,ub=0.4,lb=-0.4,tol=1e-8,maxite=1e3);
	UtiFun = ((x,σ)    -> (x^(1-σ))/(1-σ));
	DefFun = ((y,fhat) -> min.(y,fhat*mean(y)));
	return modelbase = ModelSettings(Params,UtiFun, DefFun);
end

# [2.2] Solver of the Default Econmy
function SolveDefEcon(Model::ModelSettings)
	# ----------------------------------------
	# 0. Unpacking Parameters
	# ----------------------------------------
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = Model.Params;
	# ----------------------------------------
	# 1. Tauchen discretization of log-output
	# ----------------------------------------
	ly, pix = mytauch(μ,ρ,η,nx,m);
	y       = exp.(ly);
	# ----------------------------------------
	# 2. Output in case of default
	# ----------------------------------------
	ydef    = Model.DefFun(y,fhat);
	udef    = Model.UtiFun.(ydef,σ);
	# ----------------------------------------
	# 3. To calculate the intervals of debt I will consider
	grid    = (ub-lb)/(ne-1);
	b       = [lb+(i-1)*grid for i in 1:ne];
	posb0   = findmin(abs.(0 .- b))[2];
	b[posb0]= 0;
	# ----------------------------------------
	# 3. Solving the fixed point problem
	Va,VCa,VDa,Da,BPa,qa = FixedPoint(b,y,udef,pix,posb0,Model);
	PolFun  = PolicyFunction(Va,VCa,VDa,Da,BPa,qa);
	display("Model solved, please see your results");
	return ModelSolve(Model,PolFun,Supporting(b,y,ydef,pix));
end

# [2.3] Fixed Point Problem
function  FixedPoint(b,y,udef,pix,posb0,model::ModelSettings)
	# ----------------------------------------
	# 1. Some initial parameters
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol,maxite = model.Params;
	dif  = 1
	rep  = 0;
	yb   = b .+ y';
	# ----------------------------------------
	# 2. Educated Guess
	utf  = model.UtiFun;
	VC   = 1/(1-β)*utf.((r/(1+r))*b.+ y',σ);
	udef = repeat(udef',ne,1);
	VD   = 1/(1-β)*udef;
	VO   = max.(VC,VD);
	D    = 1*(VD.>VC);
	BB   = repeat(b,1,nx);
	Bprime = Array{CartesianIndex{2},2}(undef,ne,nx);
	q    = Array{Float64,2}(undef,ne,nx);
	# ----------------------------------------
	# 3. Fixed Problem
	while dif>tol  && rep<maxite
		VO,VC, VD,D, Bprime,q,dif = value_functions!(VO,VC,VD,D,Bprime,q,dif,b,
									pix,posb0,yb,udef,β,θ,utf,r,σ);
		rep+=1;
	end
	if rep==maxite
		display("The maximization has not achieved convergence!!!!!!!!")
	else
		print("Convergence achieve after $rep replications \n")
		Bprime = BB[Bprime];
	end
	return VO,VC,VD,D,Bprime,q;
end

# [2.4] Updating the value functions
function value_functions!(VO,VC,VD,D,Bprime,q,dif,b,pix,posb0,yb,udef,β,θ,utf,r,σ)
	# ----------------------------------------
	# 1. Saving old information
	VOold = VO;
	ne,nx = size(VO);
	# ----------------------------------------
	# 2. Expected future Value Function
	EVC = VC * pix';
	EδD = D  * pix';
	EVD = VD * pix';
	βEV = β  * EVC;
	qold   = (1/(1+r))*(1 .-EδD);
	qB  = qold.*b;
	# --------------------------------------------------------------
	# 3. Value function of continuation
	VC, Bprime =updateBellman!(VC, Bprime,yb,qB,βEV,utf,σ,ne,nx);
	# --------------------------------------------------------------
	# 4. Value function of default
	βθEVC0= β*θ*EVC[posb0,:];
	VD    = βθEVC0'.+ (udef+β*(1-θ)*EVD);
	# --------------------------------------------------------------
	# 5. Continuation Value and Default choice
	VO,D  = (max.(VC,VD), 1*(VD.>VC));
	q     = (1/(1+r))*(1 .-(D*pix'));
	# --------------------------------------------------------------
	# 6.  Divergence respect the initial point
	dif   = maximum(abs.(VO-VOold));
	return VO,VC, VD,D, Bprime,q,dif;
end

# [2.5] Solving the Bellman operator under No Default
function updateBellman!(VC,Bprime,yb,qB,βEV,utf::Function,σ::Float64,ne::Int64,nx::Int64)
	@inbounds for i in 1:ne
		cc         = yb[i,:]' .- qB;
		cc[cc.<0] .= 0;
		aux_u      = utf.(cc,σ) + βEV;
		VC[i,:],Bprime[i,:] = findmax(aux_u,dims=1);
	end
	return VC, Bprime;
end
## #############################################################################
#  (3) FUNCTIONS TO SIMULATE THE ECONOMY
################################################################################

# [3.1] Simulating the model
function ModelSimulate(modsol::ModelSolve; nsim=100000, burn=0.05)
	# -------------------------------------------------------------------------
	# 0. Settings
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = modsol.Set.Params;
	EconBase = (Va = modsol.Sol.ValFun, VCa = modsol.Sol.ValNoD,VDa = modsol.Sol.ValDef,
			    Da = modsol.Sol.DefCho, BPa = modsol.Sol.DebtPF, qa = modsol.Sol.BPrice);
	pix   = modsol.Sup.MarkMat;
	b     = modsol.Sup.Bgrid
	y     = modsol.Sup.Ygrid;
	ydef  = modsol.Sup.Ydef;
	posb0 = findmin(abs.(0 .- b))[2];
	nsim2 = Int(floor(nsim*(1+burn)));

	# -------------------------------------------------------------------------
	# 1. State simulation
	choices     = 1:1:nx;   			 # Possible states
	simul_state = Int((nx+1)/2)*ones(Int64,nsim2);
	Random.seed!(41461);				 # To obtain always the same solution

	for i in 2:nsim2
		simul_state[i] = sample(view(choices,:,:),Weights(view(pix,simul_state[i-1] ,:)));
	end

	# -------------------------------------------------------------------------
	# 2. Simulation of the Economy
	distϕ          = Bernoulli(θ);				      # Probability of re-enter
	EconSim        = Array{Float64,2}(undef,nsim2,7); # Container matrix = [Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]
	EconSim[1,1:2] = [0 0];							  # Initial point
	defchoice      = EconBase.Da[posb0,simul_state[1]];
	Random.seed!(7319);
	EconSim = simulation!(EconSim,simul_state,EconBase,y, ydef, b, distϕ, nsim2,posb0)

	# -------------------------------------------------------------------------
	# 3. Burning and storaging
	# -------------------------------------------------------------------------
	orderName ="[Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]";
	EconSim = EconSim[end-nsim:end-1,:];
	modelsim = ModelSim((replications = nsim, burning= burn),modsol,EconSim,orderName);
	display("Simulation finished");
	return modelsim;
end

# [3.2] Loop for simulation
function simulation!(EconSim,simul_state,EconBase,y,ydef,b, distϕ, nsim2,posb0)
	for i in 1:nsim2-1
		bi = findfirst(x -> x==EconSim[i,2],b);
		j  = simul_state[i];
		# Choice if there is not previous default
		if EconSim[i,1] == 0
			defchoice        = EconBase.Da[bi,j];
			ysim             = (1-defchoice)*y[j]+ defchoice*ydef[j];
			bsim             = (1-defchoice)*EconBase.BPa[bi,j];
			EconSim[i,3:7]   = [ysim bsim defchoice EconBase.Va[bi,j] EconBase.qa[bi,j]];
			EconSim[i+1,1:2] = [defchoice bsim];
		else
		# Under previous default, I simulate if the economy could reenter to the market
			defstat = rand(distϕ);
			if defstat ==1 # They are in the market
				defchoice        = EconBase.Da[posb0,j]; 							# default again?
				ysim             = (1-defchoice)*y[j]+ defchoice*ydef[j];	# output | choice
				bsim             = (1-defchoice)*EconBase.BPa[posb0,j];
				EconSim[i,3:7]   = [ysim bsim defchoice EconBase.Va[posb0,j] EconBase.qa[posb0,j]];
				EconSim[i+1,1:2] = [defchoice bsim];
			else # They are out the market
				EconSim[i,3:7]   = [ydef[j] 0 1 EconBase.Va[posb0,j] EconBase.qa[posb0,j]];
				EconSim[i+1,1:2] = [1 0];
			end
		end
	end
	return EconSim;
end

## #############################################################################
#  (4) NEURAL NETWORK APPROXIMATION
################################################################################

# [4.1] Default Settings
function NeuralSettings(s)
	Q          = 16;
	n, ns      = size(s);
	ϕfun(x)    = log(1+exp(x));
	mhat       = Chain(Dense(ns,Q,ϕfun), Dense(Q,1));
	lossf(x,y) = Flux.mse(mhat(x),y);
	opt        = RADAM();
	return  NeuralSettings(mhat,lossf,opt);
end

# [4.1] Training of the NN
function NeuralTraining(y,s; neuSettings=nothing ,fnorm::Function = mynorm, Nepoch = 1)
	# -------------------------------------------------------------------------
	# 0. Settings
	n, ns = size(s);
	if isa(neuSettings, Nothing)
		neuSettings = NeuralSettings(s);
	end
	mhat  = neuSettings.Mhat;
	loss  = neuSettings.Loss;
	opt   = neuSettings.Opti;
	# -------------------------------------------------------------------------
	# 1. Data and construction of the NN
	Y     = fnorm(y);
	S     = fnorm(s);
	data  = Flux.Data.DataLoader(S',Y');
	ps    = Flux.params(mhat);
	# -------------------------------------------------------------------------
	# 2. Training
	Flux.@epochs Nepoch  begin
		Flux.Optimise.train!(loss, ps, data, opt) ;
		@show loss(S',Y');
	end;
	# -------------------------------------------------------------------------
	# 3. Predicting
	aux   = mhat(S')';
	hat   = convert(Array{Float64},aux);
	return  NeuralApprox((Y,S), hat, mhat,neuSettings);
end

## ============================================================================
#  (7)			Extensions
#  ==========================================================================
function  MaxBellman(model::ModelSettings,VO::Array,VC::Array,VD::Array,D::Array,q::Array,pix::Array,posb0::Int64,udef,yb,b,BB)
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol,maxite = model.Params;
	utf     = model.UtilFun;
	udef    = repeat(udef',ne,1);
	Bprime  = Array{CartesianIndex{2},2}(undef,ne,nx);
	VO1,VC1, VD1,D1, Bprime,q1,dif = value_functions!(VO,VC,VD,D,Bprime,q,1,b,pix,posb0,yb,udef,β,θ,utf,r,σ);
	Bprime1 = BB[Bprime];
	return VO1,VC1,VD1,D1,Bprime1,q1;
end

function UpdateModel(EconSolve::ModelSolve,VFNeuF::NamedTuple,VFhat::NeuralApprox,qhat::NeuralApprox; NormFun::Function = mynorm , NormInv::Function = mynorminv)
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol,maxite = EconSolve.Settings.Params;
	bgrid    = EconSolve.Support.bgrid;
	ygrid    = EconSolve.Support.ygrid;
	pix      = EconSolve.Support.pix;
	ydef     = EconSolve.Support.ydef;
	# ----------------------------------------
	# 2. Output in case of default
	# ----------------------------------------
	udef    = EconSolve.Settings.UtilFun.(ydef,σ);
	states   = [repeat(bgrid,length(ygrid),1) repeat(ygrid,inner = (length(bgrid),1))];
	sta_norm = NormFun(states);
	vfpre    = VFhat.mhat(sta_norm');
	qpre     = qhat.mhat(sta_norm');
	vfpre    = NormInv(vfpre,maximum(VFNeuF.vf),minimum(VFNeuF.vf));
	qpre     = NormInv(qpre,maximum(VFNeuF.q),minimum(VFNeuF.q));
	qpre     = max.(qpre,0);
	VC       = reshape(vfpre,length(bgrid),length(ygrid));
	q        = reshape(qpre,length(bgrid),length(ygrid));
	vdpre    = convert(Array,VFhat.mhat([zeros(nx) ydef]')');
	vdpre    = NormInv(vdpre,maximum(VFNeuF.vf),minimum(VFNeuF.vf));
	VD       = repeat(vdpre',ne,1);
	VO       = max.(VC,VD);
	D        = 1*(VD.>VC);
	yb       = bgrid .+ ygrid';
	BB       = repeat(bgrid,1,nx);
	posb0    = findmin(abs.(0 .- bgrid))[2];
	VO1,VC1,VD1,D1,Bprime1,q1 = DefaultEconomy.MaxBellman(EconSolve.Settings,VO,VC,VD,D,q,pix,posb0,udef,yb,bgrid,BB);
	return ModelSolve(EconSolve.Settings,PolicyFunction(VO1,VC1,VD1,D1,Bprime1,q1),EconSolve.Support)
end


function lalaland(EconSol,VFNeuF,VFhat,qhat)
  Γ1old, re11 = Flux.destructure(VFhat.mhat);
  Γ2old, re21 = Flux.destructure(qhat.mhat);
  VFNeuFAux, VFhatAux, qhatAux,Γ1new, re12,difΓ = (nothing,nothing,nothing,nothing,nothing,nothing);
  rep =1
  while rep < 100
    VFNeuFAux, VFhatAux, qhatAux = UpdateNN(EconSol,VFNeuF,VFhat,qhat)
    Γ1new, re12= Flux.destructure(VFhatAux.mhat);
    Γ2new, re22= Flux.destructure(qhatAux.mhat);
    difΓ       = max(maximum(abs.(Γ1new - Γ1old)), maximum(abs.(Γ2new-Γ2old)));
    Γ1old      = 0.75*Γ1old .+ 0.25*Γ1new;
    Γ2old      = 0.75*Γ2old .+ 0.25*Γ2new;
    VFNeuF     = VFNeuFAux;
    VFhat      = VFhatAux;    VFhat.mhat = re12(Γ1old);
    qhat       = qhatAux;     qhat.mhat  = re22(Γ2old);
    display("Iteration $rep: -> The maximum difference is $difΓ");
    rep+=1;
  end
end

function UpdateNN(EconSol,VFNeuF,VFhat,qhat;fnorm::Function = mynorm, Nsim=100000, Burn=0.05, NEpoch=1)
	# ****************************************
	# Updating the optimal policies
	# ****************************************
	EconSolAux = DefaultEconomy.UpdateModel(EconSol,VFNeuF,VFhat,qhat);

	# ****************************************
	# Simulating the new Economy
	# ****************************************
	EconSimAux = DefaultEconomy.ModelSimulate(EconSolAux,nsim=Nsim,burn=Burn);

	# ****************************************
	# Data normalization for NN
	# ****************************************
	VFNeuFAux  = (vf= EconSimAux.Simulation[:,6], q = EconSimAux.Simulation[:,7],states= EconSimAux.Simulation[:,2:3]);
	Yvf        = fnorm(VFNeuFAux.vf);
	Yqf        = fnorm(VFNeuFAux.q);
	S          = fnorm(VFNeuFAux.states);

	# ****************************************
	# NN for Value Function
	# ****************************************
	mhat_vf    = VFhat.mhat;
	loss(x,y)  = Flux.mse(mhat_vf(x),y);
	data       = Flux.Data.DataLoader(S',Yvf');
	ps         = Flux.params(mhat_vf);
	opt        = RADAM();
	if NEpoch > 1
		Flux.@epochs Nepoch Flux.Optimise.train!(loss, ps, data, opt) ;
	else
		Flux.Optimise.train!(loss, ps, data, opt);
		@show loss(S',Yvf');
	end
	aux        = mhat_vf(S')';
	hatvf      = convert(Array{Float64},aux);
	VFhatAux   = NeuralApprox((Yvf,S), hatvf, mhat_vf);

	# ****************************************
	# NN for Bond Price
	# ****************************************
	mhat_qf    = qhat.mhat;
	lossq(x,y) = Flux.mse(mhat_qf(x),y);
	dataq      = Flux.Data.DataLoader(S',Yqf');
	psq        = Flux.params(mhat_qf);
	opt        = RADAM();
	if NEpoch > 1
		Flux.@epochs Nepoch Flux.Optimise.train!(lossq, psq, dataq, opt) ;
	else
		Flux.Optimise.train!(lossq, psq, dataq, opt);
		@show loss(S',Yqf')
	end
	auxq       = mhat_qf(S')';
	hatqf      = convert(Array{Float64},auxq);
	qhatAux    = NeuralApprox((Yqf,S), hatqf, mhat_qf);
	return VFNeuFAux,VFhatAux, qhatAux;
end
## #############################################################################
#  (6) CODE FOR GRAPHICS
###############################################################################

# [6.1]	Graphics for Defaulting modelbase
function graph_solve(modsol::ModelSolve)
	# --------------------------------------------------------------
	# 0. Unpacking
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = modsol.Settings.Params;
	EconBase = (Va = modsol.Sol.ValFun, VCa = modsol.Sol.ValNoD,VDa = modsol.Sol.ValDef,
				Da = modsol.Sol.DefCho, BPa = modsol.Sol.DebtPF, qa = modsol.Sol.BPrice);
	b     = modsol.Sup.Bgrid
	y     = modsol.Sup.Ygrid;
	ydef  = modsol.Sup.Ydef;
	posb0 = findmin(abs.(0 .- b))[2];

	# --------------------------------------------------------------
	# 1. Data for figures

	# 1.1 Bond Prices
	b_in = findall(x-> x>=-0.35 && x<=0,b);
	y_li = findmin(abs.(y .- 0.95*mean(y)))[2];
	y_hi = findmin(abs.(y .- 1.05*mean(y)))[2];
	q_gra = [EconBase.qa[b_in,y_li] EconBase.qa[b_in,y_hi]];

	# 1.2 Savings Policies
	b_in2 = findall(x-> x>=-0.3 && x<=0.2,b);
	B_p = [EconBase.BPa[b_in2,y_li] EconBase.BPa[b_in2,y_hi]];

	# 1.3 Value function
	V_p = [EconBase.Va[b_in2,y_li] EconBase.Va[b_in2,y_hi]];

	# --------------------------------------------------------------
	# 2. Figures
	if ~isdir(".\\Figures")
		mkdir(".\\Figures")
	end
	# 2.1 Bond Price
	plot(b[b_in],q_gra,xlabel="B'",lw=[1.15],title="Bond price schedule q(B',y)",
		titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["y-low." "y-high"],
		foreground_color_legend = nothing, background_color_legend=nothing, legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\BondPrice.png");

	# 2.2 Savings Policies
	plot(b[b_in2],B_p,xlabel="B",lw=[1.15],title="Savings function B'(B,y)",
		titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["y-low." "y-high"],
		foreground_color_legend = nothing,background_color_legend=nothing, legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\Savings.png");

	# 2.3 Value function
	plot(b[b_in2],V_p,xlabel="B",lw=[1.15],title="Value function vo(B,y)",
		titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["y-low." "y-high"],
		foreground_color_legend = nothing,background_color_legend=nothing, legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\ValFun.png");
end

# [6.2]	Graphics for Simulation
function graph_simul(EconSimul::ModelSim; smpl=1:250)
	Sim    = EconSimul.Sim;
	Dstate = Sim[smpl,5];
	p1 = plot(Sim[smpl,3], label="", title="Output (t)", lw=2);
	p1 = bar!(twinx(),Dstate, fc = :grey,lc =:grey,label="",alpha=0.15; yticks=nothing,framestyle = :none);
	p2 = plot(Sim[smpl,4], label="", title="Issued Debt (t+1)", lw=2);
	p2 = bar!(twinx(),Dstate, fc = :grey,lc =:grey,label="",alpha=0.15; yticks=nothing,framestyle = :none);
	p3 = plot(Sim[smpl,4] ./ Sim[smpl,3], label="", title="Debt to output ratio (t)", lw=2)
	p3 = bar!(twinx(),Dstate, fc = :grey,lc =:grey,label="",alpha=0.15; yticks=nothing,framestyle = :none);
	p4 = plot(Sim[smpl,6], label="", title="Value function (t)", lw=2);
	p4 = bar!(twinx(),Dstate, fc = :grey,lc =:grey,label="",alpha=0.15; yticks=nothing,framestyle = :none);
	p5 = plot(Sim[smpl,7], label="", title="Bond price q(t)", lw=2);
	p5 = bar!(twinx(),Dstate, fc = :grey,lc =:grey,label="",alpha=0.15; yticks=nothing,framestyle = :none);
	if ~isdir(".\\Figures")
		mkdir(".\\Figures")
	end
	plot(p1,p2,layout=(2,1));
	savefig(".\\Figures\\FigSim1.png");
	plot(p4,p5,layout=(2,1));
	savefig(".\\Figures\\FigSim2.png");
	plot(p3);
	savefig(".\\Figures\\FigSim3.png");
	display("See your graphics")
end

# [6.3] Graphics for Neural Network Approximation
function graph_neural(approx::NeuralApprox, namevar::String, titles; smpl=1:250)
	# --------------------------------------------------------------
	# Checking directory
	# --------------------------------------------------------------
	if ~isdir(".\\Figures")
		mkdir(".\\Figures")
	end

	# --------------------------------------------------------------
	# Data
	# --------------------------------------------------------------
	datplot = [approx.Data[1] approx.Yhat];
	nsim    =  size(datplot)[1];
	if smpl[end] > nsim
		display("Sample size is longer than whole data");
	end
	# --------------------------------------------------------------
	# Figures
	# --------------------------------------------------------------
	theme(:sand);
	title = titles[1];
	plot(1:nsim,datplot,lw=[1.15],
		title="Approximation of $namevar", titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["actual" "approximated"],
		foreground_color_legend = nothing, background_color_legend=nothing,
		legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\$title");

	title = titles[2];
	plot(smpl,datplot[smpl,:],lw=[1.15],
		title="Approximation of $namevar", titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["actual" "approximated"],
		foreground_color_legend = nothing, background_color_legend=nothing,
		legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\$title");
end
## #############################################################################
# (7) ADDITIONAL CODES
###############################################################################

# [7.1] Tauchen Approximation
function mytauch(μ::Float64,ρ::Float64,σ::Float64,N::Int64,m::Int64)
	if (N%2!=1)
		return "N should be an odd number"
	end
	grid  = (2*m)/(N-1);
	sig_z = σ/sqrt(1-ρ^2);
	Z     = -m:grid:m;
	Z     = μ.+ Z.*sig_z;
	d5    = 0.5*(Z[2] - Z[1]);
	pix   = Array{Float64,2}(undef,N,N);
	for i in 1:N
		s        = -(1-ρ)*μ - ρ*Z[i];
		pix[i,1] =  cdf(Normal(),(Z[1] +d5+s)/σ);
		for j    in 2:N-1
			pix[i,j] = cdf(Normal(),(Z[j] + d5+s)/σ) - cdf(Normal(),(Z[j] - d5+s)/σ) ;
		end
		pix[i,N] =  1 - cdf(Normal(),(Z[N] - d5+s)/σ);
	end
	pix = pix ./ sum(pix,dims=2); # Normalization to achieve 1
	return Z, pix;
end

# [7.2] Normalization to [-1:1] interval
function mynorm(x)
	fmax   = maximum(x,dims=1);
    fmin   = minimum(x,dims=1);
    frange = 0.5*(fmax-fmin);
    normx  = (x .- 0.5*(fmin+fmax)) ./ frange;
    return normx;
end
# [7.2] Inverse of the Normalization
function mynorminv(x,xmax,xmin)
	 xorig = (x * 0.5*(xmax-xmin)) .+ 0.5*(xmax+xmin);
	 return xorig;
end
end
