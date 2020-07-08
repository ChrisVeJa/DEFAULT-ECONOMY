###############################################################################
# [] CODE FOR GRAPHICS
###############################################################################
function graph_solve(modsol; titles=["BondPrice.png" "Savings.png" "ValFun.png"])
	# --------------------------------------------------------------
	# 0. Unpacking
	@unpack r,σrisk,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol = modsol.Set.Params;
	EconBase = (Va = modsol.Sol.ValFun,
		VCa = modsol.Sol.ValNoD,
		VDa = modsol.Sol.ValDef,
		Da = modsol.Sol.DefCho,
		BPa = modsol.Sol.DebtPF,
		qa = modsol.Sol.BPrice,
	);
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
	q_gra= [EconBase.qa[b_in,y_li] EconBase.qa[b_in,y_hi]];
	# 1.2 Savings Policies
	b_in2= findall(x-> x>=-0.3 && x<=0.2,b);
	B_p  = [EconBase.BPa[b_in2,y_li] EconBase.BPa[b_in2,y_hi]];
	# 1.3 Value function
	V_p  = [EconBase.Va[b_in2,y_li] EconBase.Va[b_in2,y_hi]];
	# --------------------------------------------------------------
	# 2. Figures
	if ~isdir(".\\Figures")
		mkdir(".\\Figures")
	end
	# 2.1 Bond Price
	title = titles[1];
	plot(b[b_in],q_gra,xlabel="B'",lw=[1.15],title="Bond price schedule q(B',y)",
		titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["y-low." "y-high"],
		foreground_color_legend = nothing, background_color_legend=nothing, legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\$title");
	# 2.2 Savings Policies
	title = titles[2];
	plot(b[b_in2],B_p,xlabel="B",lw=[1.15],title="Savings function B'(B,y)",
		titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["y-low." "y-high"],
		foreground_color_legend = nothing,background_color_legend=nothing, legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\$title");
	# 2.3 Value function
	title = titles[3];
	plot(b[b_in2],V_p,xlabel="B",lw=[1.15],title="Value function vo(B,y)",
		titlefont = font(12),linestyle=[:solid :dash],
		linecolor=[:red :blue] ,label=["y-low." "y-high"],
		foreground_color_legend = nothing,background_color_legend=nothing, legendfontsize=8,legend=:topleft)
	savefig(".\\Figures\\$title");
end

# [2]	Graphics for Simulation
function graph_simul(EconSimul; smpl=1:250, titles = ["FigSim1.png" "FigSim2.png" "FigSim3.png"])
	Sim    = EconSimul.Sim;
	Dstate = Sim[smpl,5];

	p1 = plot(Sim[smpl,3], label="", title="Output (t)", lw=2);
	p1 = bar!(twinx(), Dstate, fc = :grey,
			lc =:grey, label="", alpha=0.15;
			yticks=nothing, framestyle = :none,
		);
	p2 = plot(Sim[smpl,4], label="", title="Issued Debt (t+1)", lw=2);
	p2 = bar!(twinx(), Dstate, fc = :grey,
			lc =:grey, label="", alpha=0.15;
			yticks=nothing, framestyle = :none,
		);
	p3 = plot(Sim[smpl,4] ./ Sim[smpl,3], label="", title="Debt to output ratio (t)", lw=2)
	p3 = bar!(twinx(), Dstate, fc = :grey,
			lc =:grey, label="", alpha=0.15;
			yticks=nothing, framestyle = :none,
		);
	p4 = plot(Sim[smpl,6], label="", title="Value function (t)", lw=2);
	p4 = bar!(twinx(), Dstate, fc = :grey,
			lc =:grey, label="", alpha=0.15;
			yticks=nothing, framestyle = :none,
		);
	p5 = plot(Sim[smpl,7], label="", title="Bond price q(t)", lw=2);
	p5 = bar!(twinx(), Dstate, fc = :grey,
			lc =:grey, label="", alpha=0.15;
			yticks=nothing, framestyle = :none,
			);

	if ~isdir(".\\Figures")
		mkdir(".\\Figures")
	end
	title = titles[1]; plot(p1,p2,layout=(2,1)); savefig(".\\Figures\\$title");
	title = titles[2]; plot(p4,p5,layout=(2,1)); savefig(".\\Figures\\$title");
	title = titles[3]; plot(p3);	  savefig(".\\Figures\\$title");
	display("See your graphics")
end

# [3] Graphics for Neural Network Approximation
function graph_neural(approx, namevar::String, titles; smpl=1:250)
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
	plot(1:nsim,
		datplot,
		lw=[1.15],
		title="Approximation of $namevar",
		titlefont = font(12),
		linestyle=[:solid :dash],
		linecolor=[:red :blue] ,
		label=["actual" "approximated"],
		foreground_color_legend = nothing,
		background_color_legend=nothing,
		legendfontsize=8,
		legend=:topleft,
	)
	savefig(".\\Figures\\$title");

	title = titles[2];
	plot(smpl,
		datplot[smpl,:],
		lw=[1.15],
		title="Approximation of $namevar",
		titlefont = font(12),
		linestyle=[:solid :dash],
		linecolor=[:red :blue] ,
		label=["actual" "approximated"],
		foreground_color_legend = nothing,
		background_color_legend=nothing,
		legendfontsize=8,
		legend=:topleft,
	)
	savefig(".\\Figures\\$title");
end
