#= #############################################################################
 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
The following module has all the functions used in the master.jl
Written by: Christian Velasquez (velasqcb@bc.edu)
Dont hesitate in send any comment
=# #############################################################################

module DefaultEconomy
using   Random, Distributions, Statistics, LinearAlgebra, Plots, StatsBase,
		Parameters, Flux;

###############################################################################
# <> DEFAULT STRUCTURES <>
###############################################################################

# [1.1] Setting of the model
mutable struct ModelSettings
	Params::NamedTuple;    # (r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol,maxite)
	UtiFun::Function;      # Utility Function
	DefFun::Function;      # Default Loss function
end
# [1.2] Policy functions
struct  PolicyFunction
	ValFun::Array{Float64};  # Value Function
	ValNoD::Array{Float64};  # Value Function under No Default
	ValDef::Array{Float64};  # Value Function of Default
	DefCho::Array{Float64};	 # Default Choice (1:= Default)
	DebtPF::Array{Float64};	 # Issued Debt for t+1
	BPrice::Array{Float64};	 # Price of new issued debt
end
# [1.3] Additional feautures of the model
struct Supporting
	Bgrid;	# Grid for debt
	Ygrid;	# Grid for y
	Ydef;	# Output under default
	MarkMat;# Transition matrix
end

# [1.4] Solution of the model
struct ModelSolve
	Set::ModelSettings;		# Settings of the Model
	Sol::PolicyFunction;	# Solution of the Model
	Sup::Supporting;		# Some support charact
end

# [1.5] Model Simulation Results
struct ModelSim
	Set::NamedTuple;		# Settings of the simulation
	Mod::ModelSolve;		# Model in which the simulation is based
	Sim::Array{Float64};	# Simulation matrix
	Nam::String;			# Head names of the simulation columns
end

# [1.6] Settings for the neural network
struct NeuralSettings
	Mhat;	# Neural Network
	Loss;	# Loss function
	Opti;	# Optimizer type
end

# [1.7] Neural Network results
mutable struct NeuralApprox
	Data::Tuple;			# Data inputs
	DataOri::Tuple;
	Yhat::Array;			# Fit after training
	Mhat::Any;				# Neural Network
	Sett::NeuralSettings; 	# Initial settings
end

################################################################################
#  <> SOLUTION OF THE MODEL <>
################################################################################
function SolveDefEcon(Model::ModelSettings)
	# --------------------------------------------------------------
	# 0. Unpacking Parameters
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = Model.Params;
	# --------------------------------------------------------------
	# 1. Tauchen discretization of log-output
	ly, pix = mytauch(μ,ρ,η,nx,m);
	y       = exp.(ly);
	# --------------------------------------------------------------
	# 2. Output in case of default
	ydef    = Model.DefFun(y,fhat);
	udef    = Model.UtiFun.(ydef,σ);
	# --------------------------------------------------------------
	# 3. To calculate the intervals of debt I will consider
	grid    = (ub-lb)/(ne-1);
	b       = [lb+(i-1)*grid for i in 1:ne];
	posb0   = findmin(abs.(0 .- b))[2];
	b[posb0]= 0;
	# --------------------------------------------------------------
	# 4. Solving the fixed point problem
	Va,VCa,VDa,Da,BPa,qa = FixedPoint(b,y,udef,pix,posb0,Model);
	PolFun  = PolicyFunction(Va,VCa,VDa,Da,BPa,qa);
	display("Model solved, please see your results");
	return ModelSolve(Model,PolFun,Supporting(b,y,ydef,pix));
end

################################################################################
# <> SIMULATING THE ECONOMY <>
################################################################################

# [3.1] Simulating the model
function ModelSimulate(modsol::ModelSolve; nsim=100000, burn=0.05, nseed = 0)
	# -------------------------------------------------------------------------
	# 0. Settings
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = modsol.Set.Params;
	EconBase = (Va = modsol.Sol.ValFun,
		VCa  = modsol.Sol.ValNoD,
		VDa  = modsol.Sol.ValDef,
		Da   = modsol.Sol.DefCho,
		BPa  = modsol.Sol.DebtPF,
		qa   = modsol.Sol.BPrice,
	);

	pix   = modsol.Sup.MarkMat;
	b     = modsol.Sup.Bgrid
	y     = modsol.Sup.Ygrid;
	ydef  = modsol.Sup.Ydef;
	posb0 = findmin(abs.(0 .- b))[2];
	nsim2 = Int(floor(nsim*(1+burn)));
	# -------------------------------------------------------------------------
	# 1. State simulation
	choices     = 1:nx;   			 # Possible states
	simul_state = Int((nx+1)/2)*ones(Int64,nsim2);
	if nseed != 0
		Random.seed!(nseed[1]);			 # To obtain always the same solution
	end
	for i in 2:nsim2
		simul_state[i] =
		 sample(view(choices,:,:),Weights(view(pix,simul_state[i-1] ,:)));
	end
	# -------------------------------------------------------------------------
	# 2. Simulation of the Economy
	orderName ="[Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]";
	distϕ     = Bernoulli(θ);
	EconSim   = Array{Float64,2}(undef,nsim2,7); # [Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]
	EconSim[1,1:2] = [0 0];							  # Initial point
	defchoice = EconBase.Da[posb0,simul_state[1]];
	if nseed != 0
		Random.seed!(nseed[2]);			 # To obtain always the same solution
	end
	EconSim = simulation!(EconSim,simul_state,EconBase,y, ydef, b, distϕ, nsim2,posb0)
	# -------------------------------------------------------------------------
	# 3. Burning and storaging
	EconSim  = EconSim[end-nsim:end-1,:];
	modelsim = ModelSim((replications = nsim, burning= burn),modsol,EconSim,orderName);
	#display("Simulation finished");
	return modelsim;
end

################################################################################
# <> NEURAL NETWORK APPROXIMATION <>
################################################################################

# [] Training of the NN
function NeuralTraining(y,s; neuSettings=nothing ,fnorm = mynorm, Nepoch = 1)
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
	return  NeuralApprox((Y,S),(y,s), hat, mhat,neuSettings);
end
###############################################################################
# [] Main supporting codes
###############################################################################
# [] Default settings
function ModelSettings()
	Params = (r = 0.017,
		σ = 2.0,
		ρ=0.945,
		η=0.025,
		β=0.953,
		θ=0.282,
		nx=21,
		m=3,
		μ=0.0,
		fhat=0.969,
		ne=251,
		ub=0.4,
		lb=-0.4,
		tol=1e-8,
		maxite=1e3,
	);
	UtiFun = ((x,σ)    -> (x^(1-σ))/(1-σ));
	DefFun = ((y,fhat) -> min.(y,fhat*mean(y)));
	return modelbase = ModelSettings(Params,UtiFun, DefFun);
end

# [∘] Fixed Point Problem
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

# [∘] Updating the value functions
function value_functions!(VO,VC,VD,D,Bprime,q,dif,b,pix,posb0,yb,udef,β,θ,utf,r,σ)
	# ----------------------------------------
	# 1. Saving old information
	VOold = VO;
	ne,nx = size(VO);
	# ----------------------------------------
	# 2. Expected future Value Function
	EVC = VC*pix';
	EδD = D*pix';
	EVD = VD*pix';
	βEV = β*EVC;
	qold= (1/(1+r))*(1 .-EδD);
	qB  = qold.*b;
	# --------------------------------------------------------------
	# 3. Value function of continuation
	VC, Bprime = updateBellman!(VC, Bprime,yb,qB,βEV,utf,σ,ne,nx);
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

# [∘] Solving the Bellman operator under No Default
function updateBellman!(VC,Bprime,yb,qB,βEV,utf::Function,σ::Float64,ne::Int64,nx::Int64)
	@inbounds for i in 1:ne
		cc         = yb[i,:]' .- qB;
		cc[cc.<0] .= 0;
		aux_u      = utf.(cc,σ) + βEV;
		VC[i,:],Bprime[i,:] = findmax(aux_u,dims=1);
	end
	return VC, Bprime;
end

# [∘] Loop for simulation
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
				defchoice        = EconBase.Da[posb0,j]; 					# default again?
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

# [∘] Default Settings
function NeuralSettings(s)
	Q          = 16;
	n, ns      = size(s);
	ϕfun(x)    = log1p(exp(x));
	mhat       = Chain(Dense(ns,Q,ϕfun), Dense(Q,1));
	lossf(x,y) = Flux.mse(mhat(x),y);
	opt        = Descent();
	return  NeuralSettings(mhat,lossf,opt);
end

# [∘] Belman Operator -> New Policy Functions
function  MaxBellman(model,VO,VC,VD,D,q,pix,posb0,udef,yb,b,BB)
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol,maxite = model.Params;
	utf     = model.UtiFun;
	udef    = repeat(udef',ne,1);
	Bprime  = Array{CartesianIndex{2},2}(undef,ne,nx);
	VO1,VC1, VD1,D1, Bprime,q1,dif = value_functions!(VO,VC,VD,D,Bprime,q,1,b,pix,posb0,yb,udef,β,θ,utf,r,σ);
	Bprime1 = BB[Bprime];
	return VO1,VC1,VD1,D1,Bprime1,q1;
end

###############################################################################
# [] ADDITIONAL SUPPORTING CODES
###############################################################################

# [] Tauchen Approximation
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

# [] Normalization to [-1:1] interval
function mynorm(x)
	fmax   = maximum(x,dims=1);
    fmin   = minimum(x,dims=1);
    frange = 0.5*(fmax-fmin);
    normx  = (x .- 0.5*(fmin+fmax)) ./ frange;
    return normx;
end
# [] Inverse of the Normalization
function mynorminv(x,xmax,xmin)
	 xorig = (x * 0.5*(xmax-xmin)) .+ 0.5*(xmax+xmin);
	 return xorig;
end
end
