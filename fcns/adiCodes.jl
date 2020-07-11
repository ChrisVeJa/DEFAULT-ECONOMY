###############################################################################
# [-] CODES FOR CONVERGENCE MODEL - NN - MODEL
###############################################################################
function SolverCon(DataND, NetWorkND, DataD, NetWorkD, mytup)
   @unpack σrisk, bgrid, ygrid, nx, ne, P, ydef,
      udef, yb, BB, p0, stateND, stateD, utf, β, r, θ = mytup
   #=
   Note: To calculate the the expected value we will follow next steps:
      1. Let Θₕ be the vector of parameters estimated in the Neural Network
         for h = {No Default, Default};
      2. As in every case, I previously normalized them with the formula
         ̃x = [x - 1/2 (max(x) + min(x))]/(1/2 (max(x) - min(x))),
         I get the maximum value from each one: states (sₕ), values (vₕ).
         Let x₀ₕ represents the variable x in for h-choice in the initial
         normalization
      3. Then, I apply the same normalization with the states from the grid
         ̃x₁ₕ =  [x₁ - 1/2 (max(x₀ₕ) + min(x₀ₕ))]/(1/2 (max(x₀ₕ) - min(x₀ₕ)))
         this allow me to have the grid values in the same scale than the
         inputs for the neural network
      4. Later I predict Vₕ in each case conditional on the grid,
      5. Convert the new prediction ̂y₁ₕ using the extremums in for y₀ₕ
         and calculate the expected using the markov matrix
   =#

   # ===========================================================================
   # [1] Expected Value of Continuation:
   #        hat >> inverse normalization >> matrix form >> E[]
   maxVND, minVND = (maximum(DataND[1]),minimum(DataND[1]));
   maxsND, minsND = (maximum(DataND[2], dims = 1),minimum(DataND[2], dims = 1));
   staN_nD = (stateND .- 0.5 * (maxsND + minsND)) ./ (0.5 * (maxsND - minsND));
   vc = NetWorkND(staN_nD');
   vc = (0.5 * (maxVND - minVND) * vc) .+ 0.5 * (maxVND + minVND);
   VC = reshape(vc, ne, nx);
   EVC = VC * P';  # Expected value of no defaulting

   # ===========================================================================
   # [2] Expected Value of Default:
   #        hat >> inverse normalization >> matrix form >> E[]
   maxVD, minVD = (maximum(DataD[1]),minimum(DataD[1]));
   maxsD, minsD = (maximum(DataD[2], dims = 1),minimum(DataD[2], dims = 1));
   staN_D  = (stateD .- 0.5 * (maxsD + minsD)) ./ (0.5 * (maxsD - minsD));
   vd = NetWorkD(staN_D');
   vd = (0.5 * (maxVD - minVD) * vd) .+ 0.5 * (maxVD + minVD);
   VD = reshape(vd, ne, nx);
   EVD = VD * P'; # expected value of being in default in the next period

   # [1.6] Income by issuing bonds
   #q  = EconSol.Sol.BPrice;
   D = 1 * (VD .> VC);
   q = (1 / (1 + r)) * (1 .- (D * P'));
   qB = q .* bgrid;

   # [1.7] Policy function for bonds under continuation
   βEV = β * EVC;
   VC1 = Array{Float64,2}(undef, ne, nx);
   Bindex = Array{CartesianIndex{2},2}(undef, ne, nx);
   @inbounds for i = 1:ne
      cc = yb[i, :]' .- qB;
      cc[cc.<0] .= 0;
      aux_u = utf.(cc, σrisk) + βEV;
      VC1[i, :], Bindex[i, :] = findmax(aux_u, dims = 1);
   end
   B1 = BB[Bindex];

   # [1.8] Value function of default
   βθEVC0 = β * θ * EVC[p0, :];
   VD1 = βθEVC0' .+ (udef' .+ β * (1 - θ) * EVD);

   # --------------------------------------------------------------
   # [1.9]. New Continuation Value, Default choice, price
   VF1 = max.(VC1, VD1);
   D1 = 1 * (VD1 .> VC1);
   q1 = (1 / (1 + r)) * (1 .- (D1 * P'));
   PolFun1 = (VF = VF1, VC = VC1, VD = VD1, D = D1, BP = B1, Price = q1);
   return PolFun1;
end

function _unpack(Params, Ext, uf)
   @unpack r, β, θ, σrisk = Params
   @unpack bgrid, ygrid, ydef, P = Ext
   nx = length(ygrid)
   ne = length(bgrid)
   yb = bgrid .+ ygrid'
   BB = repeat(bgrid, 1, nx)
   p0 = findmin(abs.(0 .- bgrid))[2]
   stateND= [repeat(bgrid, nx, 1) repeat(ygrid, inner = (ne, 1))]
   stateD = [repeat(bgrid, nx, 1) repeat(ydef, inner = (ne, 1))]
   #stateD = [zeros(ne*nx) repeat(ydef, inner = (ne, 1))]
   udef = uf.(ydef, σrisk)
   mytup = (
      r = r, β = β, θ = θ, σrisk = σrisk, bgrid = bgrid,
      ygrid = ygrid, nx = nx,  ne = ne,  P = P, ydef = ydef,
      udef = udef,  yb = yb, BB = BB, p0 = p0, stateND = stateND,
      stateD = stateD, utf = uf,
   )
   return mytup;
end
