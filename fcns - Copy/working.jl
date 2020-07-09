###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random, Distributions,Statistics, LinearAlgebra,
	Plots,StatsBase,Parameters, Flux;
include("DefaultEconomy.jl");
include("convergenceNN.jl");
include("graphs.jl");

############################################################
#[1] Setting >> Solving
#	As first step, I will set the model
#   in its defaults features
############################################################
EconDef = DefaultEconomy.ModelSettings();
EconSol = DefaultEconomy.SolveDefEcon(EconDef);
graph_solve(EconSol);

############################################################
#	[Simulation]
############################################################
tsim    = 100000; tburn   = 0.05;
EconSim = DefaultEconomy.ModelSimulate(EconSol,nsim=tsim,burn=tburn);
graph_simul(EconSim, smpl=1:500);

############################################################
#	[Neural Network]
############################################################
VFNeuF  = (vf= EconSim.Sim[:,6], q = EconSim.Sim[:,7],states= EconSim.Sim[:,2:3]);
VFhat   = DefaultEconomy.NeuralTraining(VFNeuF[1],VFNeuF[3], Nepoch = 10);
graph_neural(VFhat,"Value Function", ["VFneural.png" "VFSmpl.png"],smpl=1:500);

############################################################
# [2] Solving - simulating - training
############################################################
convergenceNN(EconSol,VFhat, tburn);
EconSol1, EconSim1, mytup = convergenceNN(EconSol,VFhat, tburn);
graph_solve(EconSol1,titles=["BondPrice1.png" "Savings1.png" "ValFun1.png"]);

MatVal1 = cat(EconSol.Sol.ValNoD, mytup.vc₀, EconSol1.Sol.ValNoD, dims=3);
MatVal2 = cat(EconSol.Sol.ValDef, mytup.vd₀, EconSol1.Sol.ValDef, dims=3);

############################################################
#[Graphics]
#	[∘] some graphics less standard
############################################################
_auxgraph(MatVal1,1:9,".//Figures//CompVC1");
_auxgraph(MatVal1,10:18,".//Figures//CompVC2");
_auxgraph(MatVal1,19:21,".//Figures//CompVC3");
_auxgraph(MatVal2,1:9,".//Figures//CompVD1");
_auxgraph(MatVal2,10:18,".//Figures//CompVD2");
_auxgraph(MatVal2,19:21,".//Figures//CompVD3");

function _auxgraph(Mat,list,name)
	mycol = [:blue :red :purple];
	theme(:default)
	n = length(list);
	r = Int(floor(sqrt(n)));
	p = plot(layout=(r,r), tickfontsize = 4);
	count=1;
	for i in list
		if count !=3
			p = plot!(subplot=count,Mat[:,i,:], label="",c = mycol, w = [0.75 1 1.5],style =[:solid :dash :dot]);
		else
			p = plot!(subplot=count,Mat[:,i,:], legend= :bottomright,legendfontsize =[5 5 5], label=["actual" "hat"  "post"],c = mycol, w = [0.75 1 1.5],style =[:solid :dash :dot]);
		end
		count+=1;
	end
	savefig(name)
end
