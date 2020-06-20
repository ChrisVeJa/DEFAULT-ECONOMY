module DefaultEconomy
using Random, Distributions,Statistics, LinearAlgebra, Plots,StatsBase,Parameters, Flux;
export ModelSolve,ModelSim;

###############################################################################
#(1)                           CONTAINERS:
###############################################################################
#= ------------------------------------------------------------------------
(1.1) Settings of the Model:
--------------------------------------------------------------------------- =#
mutable struct ModelSettings
	Params::NamedTuple;    # Parameters
	UtilFun::Function;     # Utility Function
	yDef::Function;        # Default Loss
end

#= ------------------------------------------------------------------------
(1.2) Policy Functions:
--------------------------------------------------------------------------- =#
struct  PolicyFunction
	valuefunction::Array{Float64};  # Value Function
	valueNoDefault::Array{Float64};  # Value Function of no defaulting
	valueDefault::Array{Float64};  # Value Function of default
	defaulchoice::Array{Float64};	# Default Choice (1:= default)
	bondpolfun::Array{Float64};		# Issued Debt for t+1
	bondprice::Array{Float64};		# Price of the debt
end

#= ------------------------------------------------------------------------
(1.3) Model Solution:
--------------------------------------------------------------------------- =#
struct ModelSolve
	Settings::ModelSettings;		# Settings of the Model
	Solution::PolicyFunction;		# Solution of the Model
end

#= ------------------------------------------------------------------------
(1.4) Simulated Economy :
--------------------------------------------------------------------------- =#
struct ModelSim
	SimSettings::NamedTuple;
	ModelBase::ModelSolve;
	Simulation::Array{Float64};
	HeadNames::String;
end

#= ------------------------------------------------------------------------
(1.5) Neural Network :
--------------------------------------------------------------------------- =#
struct NeuralApprox
	Data::Tuple;
	yhat;
	Θ;
end
#= ------------------------------------------------------------------------
(1.6) Settings for Neural network approximation :
--------------------------------------------------------------------------- =#
struct NeuralSettings
	mhat;
	loss;
	opt;
end

###############################################################################
#(2) 					SOLVING THE ECONOMY
###############################################################################

# ----------------------------------------------------------------------
# (2.1) Predetermined settings (if you want you can change this)
# ----------------------------------------------------------------------
function ModelSettings()
	Params  = (r = 0.017,σ = 2.0,ρ=0.945,η=0.025,β=0.953,θ=0.282,nx=21,m=3,μ=0.0,fhat=0.969,ne=251,ub=0.4,lb=-0.4,tol=1e-8,maxite=1e3);
	UtilFun = ((x,σ)    -> (x^(1-σ))/(1-σ));
	yDef    = ((y,fhat) -> min.(y,fhat*mean(y)));
	return modelbase = ModelSettings(Params,UtilFun, yDef);
end

function neuralSettings(s)
	Q          = 16;
	n, ns      = size(s);
	ϕfun(x)    = log(1+exp(x));
	mhat       = Chain(Dense(ns,Q,ϕfun), Dense(Q,1));
	lossf(x,y) = Flux.mse(mhat(x),y);
	opt        = RADAM();
	return  NeuralSettings(mhat,lossf,opt);
end
# ----------------------------------------------------------------------
# (2.2) Solution of the Economy
# ----------------------------------------------------------------------
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
	ydef    = Model.yDef(y,fhat);
	udef    = Model.UtilFun.(ydef,σ);
	# ----------------------------------------
	# 3. To calculate the intervals of debt I will consider
	grid    = (ub-lb)/(ne-1);
	b       = [lb+(i-1)*grid for i in 1:ne];
	posb0   = findmin(abs.(0 .- b))[2];
	b[posb0]= 0;
	Va,VCa,VDa,Da,BPa,qa = solver(b,y,udef,pix,posb0,Model);
	PolFun  = PolicyFunction(Va,VCa,VDa,Da,BPa,qa);
	EconSol = ModelSolve(Model,PolFun);
	display("See your results");
	return EconSol;
end

# ----------------------------------------------------------------------
# (2.3) Fixed Point Problem
# ----------------------------------------------------------------------
function  solver(b,y,udef,pix,posb0,model::ModelSettings)
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol,maxite = model.Params;
	yb      = b .+ y';
	utf     = model.UtilFun;
	dif,rep = (1.0, 0);
	VC   = 1/(1-β)*utf.((r/(1+r))*b.+ y',σ);
	udef = repeat(udef',ne,1);
	VD   = 1/(1-β)*udef;
	VO   = max.(VC,VD);
	D    = 1*(VD.>VC);
	BB   = repeat(b,1,nx);
	Bprime = Array{CartesianIndex{2},2}(undef,ne,nx);
	q       = Array{Float64,2}(undef,ne,nx);
	while dif>tol  && rep<maxite
		VO,VC, VD,D, Bprime,q,dif = value_functions!(VO,VC,VD,D,Bprime,q,dif,b,pix,posb0,yb,udef,β,θ,utf,r,σ);
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

###############################################################################
#(3) SIMULATION OF THE ECONOMY
###############################################################################

function ModelSimulate(modsol::ModelSolve; nsim=10000, burn=0.5)
	# -------------------------------------------------------------------------
	# 0. Unpacking Parameters
	# -------------------------------------------------------------------------
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = modsol.Settings.Params;
	Va,VCa,VDa,Da,BPa,qa  = (modsol.Solution.valuefunction,modsol.Solution.valueNoDefault ,
							modsol.Solution.valueDefault, modsol.Solution.defaulchoice,
							modsol.Solution.bondpolfun, modsol.Solution.bondprice);
	ly, pix = mytauch(μ,ρ,η,nx,m); # Markov matrix
	grid    = (ub-lb)/(ne-1);
	b       = [lb+(i-1)*grid for i in 1:ne];
	posb0   = findmin(abs.(0 .- b))[2];
	y       = exp.(ly);
	ydef    = modsol.Settings.yDef(y,fhat);
	nsim2   = Int(floor(nsim*(1+burn)));
	# -------------------------------------------------------------------------
	# 1. State simulation
	# -------------------------------------------------------------------------
	choices       = 1:1:nx;   			 # Possible states
	simul_state   = Int((nx+1)/2)*ones(Int64,nsim2);
	Random.seed!(41461);

	for i in 2:nsim2
		simul_state[i] = sample(view(choices,:,:),Weights(view(pix,simul_state[i-1] ,:)));
	end

	# -------------------------------------------------------------------------
	# 2. Simulation of the Economy
	# -------------------------------------------------------------------------
	distϕ          = Bernoulli(θ);				      # Probability of re-enter
	EconSim        = Array{Float64,2}(undef,nsim2,7); # Container matrix = [Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]
	EconSim[1,1:2] = [0 0];							  # Initial point
	defchoice      = Da[posb0,simul_state[1]];
	Random.seed!(7319);

	# Simulation
	orderName ="[Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]";
	for i in 1:nsim2-1
		bi = findfirst(x -> x==EconSim[i,2],b);
		j  = simul_state[i];
		# Choice if there is not previous default
		if EconSim[i,1] == 0
			defchoice        = Da[bi,j];
			ysim             = (1-defchoice)*y[j]+ defchoice*ydef[j];
			bsim             = (1-defchoice)*BPa[bi,j];
			EconSim[i,3:7]   = [ysim bsim defchoice Va[bi,j] qa[bi,j]];
			EconSim[i+1,1:2] = [defchoice bsim];
		else
		# Under previous default, I simulate if the economy could reenter to the market
			defstat = rand(distϕ);
			if defstat ==1 # They are in the market
				defchoice        = Da[posb0,j]; 							# default again?
				ysim             = (1-defchoice)*y[j]+ defchoice*ydef[j];	# output | choice
				bsim             = (1-defchoice)*BPa[posb0,j];
				EconSim[i,3:7]   = [ysim bsim defchoice Va[posb0,j] qa[posb0,j]];
				EconSim[i+1,1:2] = [defchoice bsim];
			else # They are out the market
				EconSim[i,3:7]   = [ydef[j] 0 1 Va[posb0,j] qa[posb0,j]];
				EconSim[i+1,1:2] = [1 0];
			end
		end
	end
	# -------------------------------------------------------------------------
	# 3. Burning and storaging
	# -------------------------------------------------------------------------
	EconSim = EconSim[end-nsim:end-1,:];
	modelsim = ModelSim((replications = nsim, burning= burn),modsol,EconSim,orderName);
	display("See your results");
	return modelsim;
end

###############################################################################
#(4) NEURAL NETWORK APPROXIMATION
###############################################################################
function neuralAprrox(y,s; neuSettings=nothing ,fnorm::Function = mynorm)
	n, ns = size(s);
	if isa(neuSettings, Nothing)
		neuSettings = neuralSettings(s);
	end
	mhat  = neuSettings.mhat;
	loss  = neuSettings.loss;
	opt   = neuSettings.opt;
	Y     = fnorm(y);
	S     = fnorm(s);
	#data  = [(x,y) = (S[i,:], Y[i])  for i in 1:n];
	data  = Flux.Data.DataLoader(S',Y');
	#data  = [(S',Y')];
	ps    = Flux.params(mhat);
	Flux.@epochs 4 begin Flux.Optimise.train!(loss, ps, data, opt) ; @show loss(S',Y') end;
	aux   = mhat(S')';
	hat   = convert(Array{Float64},aux);
	return  NeuralApprox((Y,S), hat, Flux.params(mhat));
end

###############################################################################
#(5) SUPPORTING CODES
###############################################################################

#= ============================================================================
(5.1) VALUE FUNCTIONS:
===============================================================================
	VF. OF CONTINUE
	-----------------
	The value of continuation is vᶜ(B,y) = u(y+B-qB) + βE[vᵒ(B',y')|y]

	VF. OF DEFAULT
	---------------
	The value of defaul is vᵈ(y) = u(yᵈ) + βθ E[vᵒ(0,y')|y] + β(1-θ)E[vᵈ(y')|y]
		E[vᵒ(0,yᵢ)|yⱼ]  =  ∑₁ⁿ Vᵒ(0,yᵢ)*p(yᵢ|yⱼ)
		E[vᵈ(yᵢ)|yⱼ]    =  ∑₁ⁿ vᵈ(yᵢ)*p(yᵢ|yⱼ)
=============================================================================#
function value_functions!(VO,VC,VD,D,Bprime,q,dif,b,pix,posb0,yb,udef,β,θ,utf,r,σ)
	ne,nx = size(VO);
	EVC = VC * pix';
	EδD = D  * pix';
	EVD = VD * pix';
	βEV = β  * EVC;
	VOold= VO;
	qold   = (1/(1+r))*(1 .-EδD);
	qB  = qold.*b;
	# --------------------------------------------------------------
	# Value function of continuation
	# --------------------------------------------------------------
	VC, Bprime =updateBellman!(VC, Bprime,yb,qB,βEV,utf,σ,ne,nx);
	# --------------------------------------------------------------
	# Value function of default
	# --------------------------------------------------------------
	βθEVC0= β*θ*EVC[posb0,:];
	VD    = βθEVC0'.+ (udef+β*(1-θ)*EVD);
	VO,D  = (max.(VC,VD), 1*(VD.>VC));
	q     = (1/(1+r))*(1 .-(D*pix'));
	dif   = maximum(abs.(VO-VOold));
	return VO,VC, VD,D, Bprime,q,dif;
end

#= ============================================================================
(5.2) Bellman Operator:
============================================================================== =#
function updateBellman!(VC,Bprime,yb,qB,βEV,utf::Function,σ::Float64,ne::Int64,nx::Int64)
	@inbounds for i in 1:ne
		cc         = yb[i,:]' .- qB;
		cc[cc.<0] .= 0;
		aux_u      = utf.(cc,σ) + βEV;
		VC[i,:],Bprime[i,:] = findmax(aux_u,dims=1);
	end
	return VC, Bprime;
end

#= ============================================================================
(5.3)	S.1 Tauchen discretization:
-------------------------------------------------------------------------------
		This function implements the tauchen discretization and have as arguments
			⋅) μ: Unconditional mean of the process
			⋅) ρ: AR(1) coefficient
			⋅) σ: Standard deviation of the error
			⋅) N: Number of points
			⋅) m: Coverage (# of std)
=# # ==========================================================================
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

#= ============================================================================
(5.4) Mormalization [-1:1]
---------------------------------------------------------------------------- =#
function mynorm(x)
	fmax   = maximum(x,dims=1);
    fmin   = minimum(x,dims=1);
    frange = 0.5*(fmax-fmin);
    normx  = (x .- 0.5*(fmin+fmax)) ./ frange;
    return normx;
end

###############################################################################
#(6) CODE FOR GRAPHICS
###############################################################################

#= ============================================================================
(6.1)	Graphics for Defaulting modelbase
=# # ==========================================================================
function graph_solve(modsol::ModelSolve)
	# -------------------------------------------------------------------------
	# 0. Unpacking
	# -------------------------------------------------------------------------
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = modsol.Settings.Params;
	Va,VCa,VDa,Da,BPa,qa  = (modsol.Solution.valuefunction,modsol.Solution.valueNoDefault ,
							modsol.Solution.valueDefault, modsol.Solution.defaulchoice,
							modsol.Solution.bondpolfun, modsol.Solution.bondprice);
	ly, pix = mytauch(μ,ρ,η,nx,m); # Markov matrix
	grid    = (ub-lb)/(ne-1);
	b       = [lb+(i-1)*grid for i in 1:ne];
	y       = exp.(ly);

	# --------------------------------------------------------------
	# Data for figures
	# --------------------------------------------------------------
	# Bond Prices
	b_in = findall(x-> x>=-0.35 && x<=0,b);
	y_li = findmin(abs.(y .- 0.95*mean(y)))[2];
	y_hi = findmin(abs.(y .- 1.05*mean(y)))[2];
	q_gra = [qa[b_in,y_li] qa[b_in,y_hi]];

	# Savings Policies
	b_in2 = findall(x-> x>=-0.3 && x<=0.2,b);
	B_p = [BPa[b_in2,y_li] BPa[b_in2,y_hi]];

	# Value function
	V_p = [Va[b_in2,y_li] Va[b_in2,y_hi]];

	# --------------------------------------------------------------
	# Figures
	# --------------------------------------------------------------
	if ~isdir(".\\Figures")
		mkdir(".\\Figures")
	end
	plot(b[b_in],q_gra,xlabel="B'",lw=[1.15],title="Bond price schedule q(B',y)",
		titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["y-low." "y-high"],
		foreground_color_legend = nothing, background_color_legend=nothing, legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\BondPrice.png");
	# Savings Policies
	plot(b[b_in2],B_p,xlabel="B",lw=[1.15],title="Savings function B'(B,y)",
		titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["y-low." "y-high"],
		foreground_color_legend = nothing,background_color_legend=nothing, legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\Savings.png");
	# Value function
	plot(b[b_in2],V_p,xlabel="B",lw=[1.15],title="Value function vo(B,y)",
		titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["y-low." "y-high"],
		foreground_color_legend = nothing,background_color_legend=nothing, legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\ValFun.png");
end
#= ============================================================================
(6.2)	Graphics for Simulation
=# # ==========================================================================
function graph_simul(EconSimul::ModelSim; smpl=1:250)
	Sim    = EconSimul.Simulation;
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
	savefig(".\\Figures\\FigSim1.pdf");
	plot(p4,p5,layout=(2,1));
	savefig(".\\Figures\\FigSim2.pdf");
	plot(p3);
	savefig(".\\Figures\\FigSim3.pdf");
	display("See your graphics")
end

#= ============================================================================
(6.2)	Graphics for Neural Network Approximation
=# # ==========================================================================
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
	datplot = [approx.Data[1] approx.yhat];
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
end
