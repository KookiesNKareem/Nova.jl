module QuasarReactantExt

using Quasar
using Quasar.AD: ReactantBackend, _gradient, _hessian, _jacobian, _value_and_gradient
using Quasar.Core: ADBackend

using Reactant
using Enzyme

# ============================================================================
# Reactant Backend Implementation
#
# Reactant compiles Julia functions to XLA via MLIR. For autodiff, we use
# Enzyme inside compiled functions - Reactant's EnzymeMLIR handles the
# compilation of Enzyme operations to efficient XLA code.
#
# Gradient and value_and_gradient are fully Reactant-accelerated.
# Hessian and Jacobian use Enzyme directly (Reactant nested compilation is complex).
#
# KNOWN LIMITATIONS:
# - Complex number AD not supported: Functions using complex arithmetic
#   (e.g., Heston characteristic function) will crash during MLIR compilation.
#   Error: "unsupported eltype: <<NULL TYPE>> of type tensor<complex<f64>>"
# - Scalar indexing disabled: Use sum(params .* mask) pattern instead of params[i]
# - Compilation overhead significant for small problems (< 1000 parameters)
#
# TODO: Monitor Reactant releases for complex number AD support
# TODO: Consider implementing real-valued Heston (Carr-Madan cosine) for GPU
# TODO: Add function caching to avoid recompilation
# ============================================================================

function Quasar.AD._gradient(::ReactantBackend, f, x)
    x_react = x isa Reactant.ConcreteRArray ? x : Reactant.ConcreteRArray(x)

    # Define gradient function using Enzyme
    function grad_fn(x_in)
        dx = zero(x_in)
        Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Duplicated(x_in, dx))
        return dx
    end

    # Compile to XLA
    compiled = Reactant.@compile grad_fn(x_react)
    result = compiled(x_react)

    return Array(result)
end

function Quasar.AD._value_and_gradient(::ReactantBackend, f, x)
    x_react = x isa Reactant.ConcreteRArray ? x : Reactant.ConcreteRArray(x)

    # Define function that returns both value and gradient
    function val_grad_fn(x_in)
        dx = zero(x_in)
        _, val = Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(x_in, dx))
        return val, dx
    end

    # Compile to XLA
    compiled = Reactant.@compile val_grad_fn(x_react)
    val_result, grad_result = compiled(x_react)

    # Extract scalar value
    val_out = Reactant.@allowscalar Float64(val_result[])
    return (val_out, Array(grad_result))
end

function Quasar.AD._hessian(::ReactantBackend, f, x)
    # Use Enzyme directly for hessian (nested Reactant compilation is complex)
    n = length(x)
    H = zeros(eltype(x), n, n)

    for i in 1:n
        function grad_i(y)
            dy = zero(y)
            Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Duplicated(y, dy))
            return dy[i]
        end
        dH = zero(x)
        Enzyme.autodiff(Enzyme.Reverse, grad_i, Enzyme.Active, Enzyme.Duplicated(x, dH))
        H[i, :] = dH
    end

    return H
end

function Quasar.AD._jacobian(::ReactantBackend, f, x)
    # Use Enzyme directly for jacobian
    y0 = f(x)
    m = length(y0)
    n = length(x)
    J = zeros(eltype(x), m, n)

    for j in 1:n
        dx = zeros(n)
        dx[j] = 1.0
        dy = Enzyme.autodiff(Enzyme.Forward, f, Enzyme.Duplicated(x, dx))[1]
        J[:, j] = dy
    end

    return J
end

# ============================================================================
# BatchPricing Pre-compiled Calibration
# ============================================================================

function _reactant_compile_gpu!(cal::Quasar.BatchPricing.PrecompiledSABRCalibrator)
    F, T, β = cal.F, cal.T, cal.β
    strikes, market_vols, n = cal.strikes, cal.market_vols, cal.n
    MASK3_1, MASK3_2, MASK3_3 = Quasar.BatchPricing.MASK3_1, Quasar.BatchPricing.MASK3_2, Quasar.BatchPricing.MASK3_3

    function loss(params)
        α = abs(sum(params .* MASK3_1))
        ρ = tanh(sum(params .* MASK3_2))
        ν = exp(sum(params .* MASK3_3))
        err = zero(eltype(params))
        for i in 1:n
            err += (Quasar.BatchPricing._sabr_vol_gpu(F, strikes[i], T, α, β, ρ, ν) - market_vols[i])^2
        end
        err / n
    end

    x_react = Reactant.ConcreteRArray([0.2, 0.0, log(0.3)])
    grad_fn(x) = begin
        dx = zero(x)
        Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(x, dx))
        dx
    end
    cal.compiled_grad = Reactant.@compile grad_fn(x_react)
    cal.backend = Quasar.AD.ReactantBackend()
    cal
end

function _reactant_call_compiled_grad(cal::Quasar.BatchPricing.PrecompiledSABRCalibrator, x)
    Array(cal.compiled_grad(Reactant.ConcreteRArray(x)))
end

# ============================================================================
# General Calibrator (compile once, run on any data of same shape)
# Uses one-hot masks to avoid scalar indexing on traced arrays
# ============================================================================

function _reactant_compile_general!(cal::Quasar.BatchPricing.GeneralSABRCalibrator)
    F, T, β, n = cal.F, cal.T, cal.β, cal.n
    MASK3_1 = Quasar.BatchPricing.MASK3_1
    MASK3_2 = Quasar.BatchPricing.MASK3_2
    MASK3_3 = Quasar.BatchPricing.MASK3_3

    # Pre-compute one-hot masks for each strike index
    strike_masks = [Float64[i == j ? 1.0 : 0.0 for j in 1:n] for i in 1:n]

    # Vectorized SABR vol that works on traced values
    function sabr_vol_traced(F, K, T, α, β, ρ, ν)
        logFK = log(F / K + 1e-12)
        FK_mid = (F * K)^((1 - β) / 2)
        z = (ν / α) * FK_mid * logFK
        z_sq = z^2
        sqrt_term = sqrt(1 - 2*ρ*z + z_sq + 1e-12)
        log_arg = max((sqrt_term + z - ρ) / (1 - ρ + 1e-12), 1e-12)
        x_z_full = z / (log(log_arg) + 1e-12)
        w = z_sq / (z_sq + 1e-6)
        x_z = (1 - w) * (1 + ρ * z / 2) + w * x_z_full
        denom = 1 + (1-β)^2/24 * logFK^2 + (1-β)^4/1920 * logFK^4
        A = α / (FK_mid * denom + 1e-12)
        C1 = ((1-β)^2 / 24) * (α^2 / (FK_mid^2 + 1e-12))
        C2 = (ρ * β * ν * α) / (4 * FK_mid + 1e-12)
        C3 = (2 - 3*ρ^2) * ν^2 / 24
        A * x_z * (1 + (C1 + C2 + C3) * T)
    end

    # Loss using mask-based extraction (no scalar indexing)
    function loss(params, K_vec, mv_vec)
        α = abs(sum(params .* MASK3_1))
        ρ = tanh(sum(params .* MASK3_2))
        ν = exp(sum(params .* MASK3_3))
        err = zero(eltype(params))
        for i in 1:n
            K_i = sum(K_vec .* strike_masks[i])
            mv_i = sum(mv_vec .* strike_masks[i])
            model_vol = sabr_vol_traced(F, K_i, T, α, β, ρ, ν)
            err += (model_vol - mv_i)^2
        end
        err / n
    end

    # Gradient with respect to params only
    function grad_fn(params, K_vec, mv_vec)
        dp = zero(params)
        Enzyme.autodiff(Enzyme.Reverse,
            (p, k, m) -> loss(p, k, m),
            Enzyme.Active,
            Enzyme.Duplicated(params, dp),
            Enzyme.Const(K_vec),
            Enzyme.Const(mv_vec))
        dp
    end

    # Compile with dummy inputs of correct shape
    x_react = Reactant.ConcreteRArray([0.2, 0.0, log(0.3)])
    K_react = Reactant.ConcreteRArray(collect(range(80.0, 120.0, length=n)))
    mv_react = Reactant.ConcreteRArray(fill(0.2, n))

    cal.compiled_grad = Reactant.@compile grad_fn(x_react, K_react, mv_react)
    cal
end

function _reactant_call_general_grad(cal::Quasar.BatchPricing.GeneralSABRCalibrator,
                                     x, strikes, market_vols)
    x_r = Reactant.ConcreteRArray(x)
    K_r = Reactant.ConcreteRArray(strikes)
    mv_r = Reactant.ConcreteRArray(market_vols)
    Array(cal.compiled_grad(x_r, K_r, mv_r))
end

# ============================================================================
# Batch Calibrator - ALL smiles in ONE GPU operation
# ============================================================================

function _reactant_compile_batch!(cal::Quasar.BatchPricing.BatchSABRCalibrator)
    F, T, β = cal.F, cal.T, cal.β
    n_strikes, n_smiles = cal.n_strikes, cal.n_smiles
    MASK3_1 = Quasar.BatchPricing.MASK3_1
    MASK3_2 = Quasar.BatchPricing.MASK3_2
    MASK3_3 = Quasar.BatchPricing.MASK3_3

    # One-hot masks for strikes
    strike_masks = [Float64[i == j ? 1.0 : 0.0 for j in 1:n_strikes] for i in 1:n_strikes]

    function sabr_vol_traced(F, K, T, α, β, ρ, ν)
        logFK = log(F / K + 1e-12)
        FK_mid = (F * K)^((1 - β) / 2)
        z = (ν / α) * FK_mid * logFK
        z_sq = z^2
        sqrt_term = sqrt(1 - 2*ρ*z + z_sq + 1e-12)
        log_arg = max((sqrt_term + z - ρ) / (1 - ρ + 1e-12), 1e-12)
        x_z_full = z / (log(log_arg) + 1e-12)
        w = z_sq / (z_sq + 1e-6)
        x_z = (1 - w) * (1 + ρ * z / 2) + w * x_z_full
        denom = 1 + (1-β)^2/24 * logFK^2 + (1-β)^4/1920 * logFK^4
        A = α / (FK_mid * denom + 1e-12)
        C1 = ((1-β)^2 / 24) * (α^2 / (FK_mid^2 + 1e-12))
        C2 = (ρ * β * ν * α) / (4 * FK_mid + 1e-12)
        C3 = (2 - 3*ρ^2) * ν^2 / 24
        A * x_z * (1 + (C1 + C2 + C3) * T)
    end

    # One-hot masks for smile index
    smile_masks = [Float64[i == j ? 1.0 : 0.0 for j in 1:n_smiles] for i in 1:n_smiles]

    # Total loss = sum of all individual losses
    function total_loss(params, K_mat, mv_mat)
        # params: [3 × n_smiles], K_mat: [n_strikes × n_smiles], mv_mat: [n_strikes × n_smiles]
        total_err = zero(eltype(params))
        for s in 1:n_smiles
            # Extract params for this smile using masks
            p1 = sum(params[1, :] .* smile_masks[s])
            p2 = sum(params[2, :] .* smile_masks[s])
            p3 = sum(params[3, :] .* smile_masks[s])
            α = abs(p1)
            ρ = tanh(p2)
            ν = exp(p3)

            for i in 1:n_strikes
                K_i = sum(K_mat[:, s] .* strike_masks[i])
                mv_i = sum(mv_mat[:, s] .* strike_masks[i])
                model_vol = sabr_vol_traced(F, K_i, T, α, β, ρ, ν)
                total_err += (model_vol - mv_i)^2
            end
        end
        total_err / (n_strikes * n_smiles)
    end

    function grad_fn(params, K_mat, mv_mat)
        dp = zero(params)
        Enzyme.autodiff(Enzyme.Reverse,
            (p, k, m) -> total_loss(p, k, m),
            Enzyme.Active,
            Enzyme.Duplicated(params, dp),
            Enzyme.Const(K_mat),
            Enzyme.Const(mv_mat))
        dp
    end

    # Compile with dummy inputs
    x_react = Reactant.ConcreteRArray(fill(0.2, 3, n_smiles))
    K_react = Reactant.ConcreteRArray(fill(100.0, n_strikes, n_smiles))
    mv_react = Reactant.ConcreteRArray(fill(0.2, n_strikes, n_smiles))

    cal.compiled_grad = Reactant.@compile grad_fn(x_react, K_react, mv_react)
    cal
end

function _reactant_call_batch_grad(cal::Quasar.BatchPricing.BatchSABRCalibrator,
                                   x, strikes, market_vols)
    x_r = Reactant.ConcreteRArray(x)
    K_r = Reactant.ConcreteRArray(strikes)
    mv_r = Reactant.ConcreteRArray(market_vols)
    Array(cal.compiled_grad(x_r, K_r, mv_r))
end

# Register callbacks
function __init__()
    Quasar.BatchPricing._REACTANT_COMPILE_GPU[] = _reactant_compile_gpu!
    Quasar.BatchPricing._REACTANT_CALL_GRAD[] = _reactant_call_compiled_grad
    Quasar.BatchPricing._REACTANT_COMPILE_GENERAL[] = _reactant_compile_general!
    Quasar.BatchPricing._REACTANT_CALL_GENERAL_GRAD[] = _reactant_call_general_grad
    Quasar.BatchPricing._REACTANT_COMPILE_BATCH[] = _reactant_compile_batch!
    Quasar.BatchPricing._REACTANT_CALL_BATCH_GRAD[] = _reactant_call_batch_grad
end

end # module
