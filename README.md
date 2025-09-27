# Fisher-LoRA 

Do LoRA updates in the *Fisher-whitened* space of each linear map instead of raw parameter space.
In whitened coordinates, plain gradient descent = **natural gradient** (K-FAC) in the original weights. That means updates respect KL geometry, are more invariant to rescaling, and are rank-efficient.

**Key objects per layer**

* $A=\mathbb{E}[x x^\top]$ (input second moment), $B=\mathbb{E}[g g^\top]$ (output-grad second moment).
* Whiteners: $A^{-1/2}, B^{-1/2}$ (with damping).

**Core play**

1. Define whitened weights $\tilde{W} := B^{1/2} W A^{1/2}$.
2. Do a rank-$r$ LoRA update **in whitened space**: $\Delta \tilde{W} = U S V^\top$ (or just $U V^\top$).
3. Map back:
   $$\boxed{\Delta W = B^{-1/2} \Delta \tilde{W} A^{-1/2}}$$

**Runtime form (drop-in LoRA)**

* Precompute $L_0:=B^{-1/2}U$; $R_0:=A^{-1/2}V$.
* Forward: $y = W x + L_0 S (R_0^\top x)$ (or $S$ omitted). Same cost as LoRA.

**Why this beats Euclidean LoRA**

* Rank-$r$ step aligns with **top singular modes of the whitened gradient** ⇒ best rank-$r$ *natural* step.
* Approx. invariant to feature/output rescalings; usually allows larger, stabler steps.

**Minimal training loop notes**

* Keep EMAs of $A,B$; periodically refresh $A^{-1/2},B^{-1/2}$ and re-bake $L_0,R_0$.
* Base weights stay frozen; only $U,V$ (and optionally $S$) are trained.
