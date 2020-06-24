NormFun(x)           = (x .- 0.5*(maximum(x,dims=1)+minimum(x,dims=1))) ./ (0.5*(maximum(x,dims=1)-minimum(x,dims=1)));
NormInv(x,xmax,xmin) = (x * 0.5*(xmax-xmin)) .+ 0.5*(xmax+xmin);

function myfun1(EconSol,VFhat,qhat; normfun::Function = NormFun , norminv::Function = NormInv)
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol,maxite = EconSol.Settings.Params;
	bgrid    = EconSol.Support.bgrid;
	ygrid    = EconSol.Support.ygrid;
	pix      = EconSol.Support.pix;
	ydef     = EconSol.Support.ydef;
	# ----------------------------------------
	# 2. Output in case of default
	# ----------------------------------------
	udef    = EconSol.Settings.UtilFun.(ydef,σ);


	states   = [repeat(bgrid,length(ygrid),1) repeat(ygrid,inner = (length(bgrid),1))];
	sta_norm = normfun(states);
	vfpre    = VFhat.mhat(sta_norm');
	qpre     = qhat.mhat(sta_norm');
	vfpre    = norminv(vfpre,maximum(VFNeuF.vf),minimum(VFNeuF.vf));
	qpre     = norminv(qpre,maximum(VFNeuF.q),minimum(VFNeuF.q));
	qpre     = max.(qpre,0);

	VC       = reshape(vfpre,length(bgrid),length(ygrid));
	q        = reshape(qpre,length(bgrid),length(ygrid));
	VO       = max.(VC,VD);
	D        = 1*(VD.>VC);
	yb       = bgrid .+ ygrid';
	BB       = repeat(bgrid,1,nx);

	MaxBellman(EconSol.Settings,V0::Array,VC,VD::Array,D::Array,q,pix::Array,posb0::Int64,udef,yb,BB)
end









#### This should be in the module
function  MaxBellman(model::ModelSettings,V0::Array,VC::Array,VD::Array,D::Array,q::Array,pix::Array,posb0::Int64,udef,yb,BB)
	@unpack r,σ,ρ,η,β,θ,nx,m,μ,fhat,ne,ub,lb,tol,maxite = model.Params;
	utf     = model.UtilFun;
	udef    = repeat(udef',ne,1);
	Bprime  = Array{CartesianIndex{2},2}(undef,ne,nx);
	VO1,VC1, VD1,D1, Bprime,q1,dif = value_functions!(VO,VC,VD,D,Bprime,q,1,b,pix,posb0,yb,udef,β,θ,utf,r,σ);
	Bprime1 = BB[Bprime];
	return VO1,VC1,VD1,D1,Bprime1,q1;
end
