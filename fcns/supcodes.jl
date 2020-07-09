###############################################################################
# [] ADDITIONAL SUPPORTING CODES
###############################################################################

# [] Tauchen Approximation
function mytauch(μ::Float64,ρ::Float64,σrisk::Float64,N::Int64,m::Int64)
	if (N%2!=1)
		return "N should be an odd number"
	end
	grid  = (2*m)/(N-1);
	sig_z = σrisk/sqrt(1-ρ^2);
	Z     = -m:grid:m;
	Z     = μ.+ Z.*sig_z;
	d5    = 0.5*(Z[2] - Z[1]);
	pix   = Array{Float64,2}(undef,N,N);
	for i in 1:N
		s        = -(1-ρ)*μ - ρ*Z[i];
		pix[i,1] =  cdf(Normal(),(Z[1] +d5+s)/σrisk);
		for j    in 2:N-1
			pix[i,j] = cdf(Normal(),(Z[j] + d5+s)/σrisk) - cdf(Normal(),(Z[j] - d5+s)/σrisk) ;
		end
		pix[i,N] =  1 - cdf(Normal(),(Z[N] - d5+s)/σrisk);
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
