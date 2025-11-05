Introduction
============

Motivation
----------

Wave phenomena appear in physics, engineering, and geoscience — from vibrating strings to seismic and electromagnetic propagation.
In realistic settings, material parameters, boundary data, or external forces are rarely known exactly and may fluctuate in time or space.
To capture this uncertainty, classical deterministic models are extended to **stochastic partial differential equations (SPDEs)**, which describe not one trajectory but the *statistical behavior* of an ensemble of possible wave fields.

This project implements a **Continuous Time Galerkin (CTG)** finite element method for the **stochastic wave equation**, providing a stable and high-order space–time solver suitable for uncertainty quantification and parametric studies.


The Stochastic Wave Equation
----------------------------

The deterministic wave equation on a spatial domain :math:`D ⊂ ℝ^d` with time interval :math:`(0,T)` reads

.. math::

   ∂_{tt} u(x,t) - Δu(x,t) = f(x,t),
   \quad u|_{t=0} = u_0, \; ∂_t u|_{t=0} = v_0.

To include random effects such as noisy forcing or uncertain media, one introduces a stochastic term driven by a Wiener process :math:`W(t)` (see [Dalang2009]_):

.. math::

   ∂_{tt} U(x,t,ω) - ΔU(x,t,ω) = f(x,t) + σ(U) \, \dot{W}(t),
   \quad (x,t) ∈ D × (0,T), \; ω ∈ Ω.

This **stochastic wave equation (SWE)** models, for example, vibrations under random forcing or wave propagation in random media.
The simplest case, with :math:`σ(U)=U`, is the **Hyperbolic Anderson Model**.

Because stochastic integrals involve the irregular signal :math:`\dot{W}(t)`, numerical treatment often starts by reformulating the equation into a deterministic **parametric PDE** using the *Doss–Sussmann transform*.


Doss–Sussmann Transform
-----------------------

The Doss–Sussmann approach removes the stochastic differential by transforming the unknown :math:`U` into a random but deterministic function :math:`u`:

.. math::

   u = e^{-W(t)σ} U.

This yields an equivalent **random-coefficient wave equation**

.. math::

   ∂_t u = A u + f + \text{terms depending on } W(t),

where :math:`A` denotes the spatial operator.
By expanding :math:`W(t)` in a finite stochastic basis (e.g. Lévy–Ciesielski), one obtains a *parametric wave equation* that can be solved deterministically for many parameter samples.
This forms the link between the stochastic PDE and numerical solvers like CTG.


Continuous Time Galerkin (CTG) Method
-------------------------------------

The **Continuous Time Galerkin (CTG)** method ([FrenchPeterson1996]_ and [Gomez2025]_) is a **space-time finite element formulation** that treats time similarly to space.
It seeks an approximate solution :math:`u_h(t,x)` continuous in time and piecewise polynomial over time slabs :math:`I_n = [t_n, t_{n+1}]`.

The variational formulation reads:

For each space–time slab :math:`I_n × D`, find :math:`u_h,v_h ∈ S_h^p ⊗ P_τ^q` such that

.. math::

   (∂_t u_h - v_h, ϕ)_{I_n × D} = 0, \\
   (∂_t v_h, ψ)_{I_n × D} + (∇u_h, ∇ψ)_{I_n × D} = (f, ψ)_{I_n × D},

for all test functions :math:`ϕ, ψ ∈ S_h^p ⊗ P_τ^{q-1}`.

The scheme has the following advantageous properties:

* **Continuous in time** — no jump between time slabs.
* **Unconditionally stable** — no CFL constraint on the time step.
* **Energy-conserving** — discrete energy remains constant for :math:`f=0`.
* **High-order accurate** — supports arbitrary polynomial degrees in space and time.

For stochastic or parametric wave equations, CTG solves each realization independently, providing robust, high-accuracy trajectories that can be averaged or post-processed statistically.


Project Scope
-------------

This repository provides a Python implementation of the Continuous Time Galerkin (CTG) method for the stochastic wave equation, featuring:

* Unified **space–time formulation** with DOLFINx.
* Support for **deterministic and stochastic** variants.
* Tools for sampling, convergence analysis, and energy tracking.
* Reproducible configurations and visualization utilities.

It serves as both a **research tool** for exploring stochastic hyperbolic problems and a **template** for building advanced UQ or machine-learning surrogates.


References
----------

.. [FrenchPeterson1996] D.A. French and T.E. Peterson, *A Continuous Space-Time Finite Element Method for the Wave Equation*, Math. Comp. 65(214):491-506, 1996.

.. [Dalang2009] R.C. Dalang and C. Mueller, *Intermittency Properties in a Hyperbolic Anderson Problem*, Ann. Inst. H. Poincaré Probab. Stat. 45(4):1150-1164, 2009.

.. [Gomez2025] S. Gómez, *A Variational Approach to the Analysis of the Continuous Space-Time FEM for the Wave Equation*, 2025.
