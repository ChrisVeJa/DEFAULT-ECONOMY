##  APPROXIMATION OF DEFAULT ECONOMIES WITH NEURAL NETWORKS
### Setting the model:

This document is based on the basic model from Arellano (2008), in this economy the country is populated by a representative agent who maximizes the expected value of her consumption path which can be smoothed by issued sovereign debt traded in a competitive foreign market compounded by atomistic risk neutral investors.

This debt is defaultable but if the country chooses to do it, he faces an instantaneous output cost and a exclusion from financial markets, In addition, there is a probability θ of re-entering next period. Hence, the houshehold problem is summarized as:

\begin{equation*} 	 c = 	 \begin{cases}	 y + B - q(B',y)B' \quad \text{if repay}\\	 y^{def}	 \text{\hspace{2.3cm} if they default}	 \end{cases}\end{equation*}
