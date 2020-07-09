#= #############################################################################
 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
The following module has all the functions used in the master.jl
Written by: Christian Velasquez (velasqcb@bc.edu)
Dont hesitate in send any comment
=# #############################################################################
module DefEcon
	using   Random, Distributions, Statistics, LinearAlgebra, Plots, StatsBase,
		Parameters, Flux;
	include("supcodes.jl");
	include("mainsup.jl");
	include("graphs.jl");
	################################################################################
	# [1] SOLUTION OF THE MODEL
	################################################################################
	function SolveR(Params,DefFun,UtiFun)
		# --------------------------------------------------------------
		# 0. Unpacking Parameters
		@unpack r,σrisk,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = Params;
		# --------------------------------------------------------------
		# 1. Tauchen discretization of log-output
		ly, P = mytauch(μ,ρ,η,nx,m);
		y       = exp.(ly);
		# --------------------------------------------------------------
		# 2. Output in case of default
		ydef    = DefFun(y,fhat);
		udef    = UtiFun.(ydef,σrisk);
		# --------------------------------------------------------------
		# 3. To calculate the intervals of debt I will consider
		grid = (ub-lb)/(ne-1);
		b    = [lb+(i-1)*grid for i in 1:ne];
		p0   = findmin(abs.(0 .- b))[2];
		b[p0]= 0;
		# --------------------------------------------------------------
		# 4. Solving the fixed point problem
		V,VC,VD,D,BP,q = FixedPoint(b,y,udef,P,p0,Params, UtiFun);
		PolFun  = (VF = V, VC = VC, VD = VD,D = D, BP = BP, Price =q);
		ModelSolve = (Mod = (Θ = Params, h = DefFun, uf = UtiFun) ,
						PolFun = PolFun,
						Ext = (bgrid= b, ygrid = y, ydef=ydef, P = P),
					)
		display("Model solved, please see your results");
		return ModelSolve;
	end

	################################################################################
	# <> SIMULATING THE ECONOMY <>
	################################################################################

	# [3.1] Simulating the model
	function ModelSim(Params, PF, Ext; nsim=100000, burn=0.05, nseed = 0)
		# -------------------------------------------------------------------------
		# 0. Settings
		@unpack r,σrisk,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = Params;
		EconBase = (VF  = PF.VF, VC  = PF.VC, VD  = PF.VD,
					D   = PF.D, BP  = PF.BP, q  = PF.Price,
		);

		P    = Ext.P;
		b    = Ext.bgrid;
		nbp  = length(b);
		y    = Ext.ygrid;
		ydef = Ext.ydef;
		p0= findmin(abs.(0 .- b))[2];
		nsim2= Int(floor(nsim*(1+burn)));
		# -------------------------------------------------------------------------
		# 1. State simulation
		choices     = 1:nx;   			 # Possible states
		simul_state = Int((nx+1)/2)*ones(Int64,nsim2);
		if nseed != 0
			Random.seed!(nseed[1]);			 # To obtain always the same solution
		end
		for i in 2:nsim2
			simul_state[i] =
			 sample(view(choices,:,:),Weights(view(P,simul_state[i-1] ,:)));
		end
		# -------------------------------------------------------------------------
		# 2. Simulation of the Economy
		orderName ="[Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]";
		distϕ     = Bernoulli(θ);
		EconSim   = Array{Float64,2}(undef,nsim2,7);      # [Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]
		if nbp == ne
			EconSim[1,1:2] = [0 0];							  # Initial point
		else
			auxpos = rand(1:nbp);
			EconSim[1,1:2] = [0 b[auxpos]];
		end
		defchoice = EconBase.D[p0,simul_state[1]];
		if nseed != 0
			Random.seed!(nseed[2]);			 # To obtain always the same solution
		end

		EconSim = simulation!(EconSim,simul_state,EconBase,y, ydef, b, distϕ, nsim2,p0)
		# -------------------------------------------------------------------------
		# 3. Burning and storaging
		EconSim  = EconSim[end-nsim:end-1,:];
		modelsim = (Sim = EconSim, order = orderName);
		return modelsim;
	end

	################################################################################
	# <> NEURAL NETWORK APPROXIMATION <>
	################################################################################

	# [] Training of the NN
	function NeuTra(y,s, neuSettings ,fnorm; Nepoch = 1)
		# -------------------------------------------------------------------------
		# 0. Settings
		n, ns = size(s);
		mhat  = neuSettings.mhat;
		loss  = neuSettings.loss;
		opt   = neuSettings.opt;
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
		return  NeuralApprox = (DataNorm = (Y,S), Data = (y,s), Hat = hat, Mhat = mhat);
	end
end
