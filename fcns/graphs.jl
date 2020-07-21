###############################################################################
# [] CODE FOR GRAPHICS
###############################################################################
function graph_solve(
    PolFun,
    ext;
    titles = ["BondPrice.png" "Savings.png" "ValFun.png"],
    )
    # --------------------------------------------------------------
    # 0. Unpacking
    @unpack vf, vr, vd, D, bp, q= PolFun;
    b = ext.bgrid
    y = ext.ygrid
    ydef = ext.ydef
    p0 = findmin(abs.(0 .- b))[2]
    # --------------------------------------------------------------
    # 1. Data for figures
    # 1.1 Bond Prices
    b_in = findall(x -> x >= -0.30 && x <= 0, b)
    y_li = findmin(abs.(y .- 0.95 * mean(y)))[2]
    y_hi = findmin(abs.(y .- 1.05 * mean(y)))[2]
    q_gra = [q[b_in, y_li] q[b_in, y_hi]]
    # 1.2 Savings Policies
    b_in2 = findall(x -> x >= -0.3 && x <= 0.2, b)
    b_p = [bp[b_in2, y_li] bp[b_in2, y_hi]]
    # 1.3 Value function
    vf_p = [vf[b_in2, y_li] vf[b_in2, y_hi]]
    # --------------------------------------------------------------
    # 2. Figures
    if ~isdir(".\\Figures")
        mkdir(".\\Figures")
    end
    # 2.1 Bond Price
    title = titles[1]
    plot(
        b[b_in],
        q_gra,
        xlabel = "B'",
        lw = [1.15],
        title = "Bond price schedule q(B',y)",
        titlefont = font(12),
        linestyle = [:solid :dash],
        linecolor = [:red :blue],
        label = ["y-low." "y-high"],
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        legendfontsize = 8,
        legend = :topleft,
    )
    savefig(".\\Figures\\$title")
    # 2.2 Savings Policies
    title = titles[2]
    plot(
        b[b_in2],
        b_p,
        xlabel = "B",
        lw = [1.15],
        title = "Savings function B'(B,y)",
        titlefont = font(12),
        linestyle = [:solid :dash],
        linecolor = [:red :blue],
        label = ["y-low." "y-high"],
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        legendfontsize = 8,
        legend = :topleft,
    )
    savefig(".\\Figures\\$title")
    # 2.3 Value function
    title = titles[3]
    plot(
        b[b_in2],
        vf_p,
        xlabel = "B",
        lw = [1.15],
        title = "Value function vo(B,y)",
        titlefont = font(12),
        linestyle = [:solid :dash],
        linecolor = [:red :blue],
        label = ["y-low." "y-high"],
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        legendfontsize = 8,
        legend = :topleft,
    )
    savefig(".\\Figures\\$title")
end

# [2]	Graphics for Simulation
function graph_simul(
    Sim;
    smpl = 1:500,
    titles = ["FigSim1.png" "FigSim2.png" "FigSim3.png"],
    )
    default = Sim[smpl, 5]

    p1 = plot(Sim[smpl, 3], label = "", title = "Output (t)", lw = 2)
    p1 = bar!(twinx(),default, fc = :grey, lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    p2 = plot(Sim[smpl, 4], label = "", title = "Issued Debt (t+1)", lw = 2)
    p2 = bar!(twinx(),default, fc = :grey,lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    p3 = plot(Sim[smpl, 4] ./ Sim[smpl, 3], label = "", lw = 2,
        title = "Debt to output ratio (t)",
    )
    p3 = bar!(twinx(), default,fc = :grey, lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    p4 = plot(Sim[smpl, 6], label = "", title = "Value function (t)", lw = 2)
    p4 = bar!(twinx(), default, fc = :grey, lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    p5 = plot(Sim[smpl, 7], label = "", title = "Bond price q(t)", lw = 2)
    p5 = bar!(twinx(), default, fc = :grey, lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    if ~isdir(".\\Figures")
        mkdir(".\\Figures")
    end
    title = titles[1]
    plot(p1, p2, layout = (2, 1))
    savefig(".\\Figures\\$title")
    title = titles[2]
    plot(p4, p5, layout = (2, 1))
    savefig(".\\Figures\\$title")
    title = titles[3]
    plot(p3)
    savefig(".\\Figures\\$title")
    display("See your graphics")
end
