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

# Register callbacks
function __init__()
    Quasar.BatchPricing._REACTANT_COMPILE_GPU[] = _reactant_compile_gpu!
    Quasar.BatchPricing._REACTANT_CALL_GRAD[] = _reactant_call_compiled_grad
end

end # module
