using Test
using Quasar

@testset "Instruments" begin
    @testset "Stock" begin
        stock = Stock("AAPL")

        @test stock.symbol == "AAPL"
        @test stock isa AbstractEquity
        @test stock isa AbstractInstrument

        # Traits
        @test ispriceable(stock)
        @test isdifferentiable(stock)

        # Pricing
        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )
        @test price(stock, state) == 150.0
    end

    @testset "EuropeanOption" begin
        # Call option
        call = EuropeanOption("AAPL", 150.0, 1.0, :call)

        @test call.underlying == "AAPL"
        @test call.strike == 150.0
        @test call.expiry == 1.0
        @test call.optiontype == :call
        @test call isa AbstractOption

        # Traits
        @test ispriceable(call)
        @test isdifferentiable(call)
        @test hasgreeks(call)

        # Black-Scholes pricing (ATM call, 1 year, 20% vol, 5% rate)
        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # ATM call should be worth roughly S * 0.4 * σ * √T for small σ
        # More precisely, use known BS value
        p = price(call, state)
        @test p > 0.0
        @test p < 150.0  # Can't be worth more than underlying

        # Put option
        put = EuropeanOption("AAPL", 150.0, 1.0, :put)
        p_put = price(put, state)

        # Put-call parity: C - P = S - K*exp(-rT)
        S = 150.0
        K = 150.0
        r = 0.05
        T = 1.0
        @test p - p_put ≈ S - K * exp(-r * T) atol=1e-10
    end

    @testset "Greeks (Analytical Black-Scholes)" begin
        call = EuropeanOption("AAPL", 150.0, 1.0, :call)

        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # Get all Greeks (uses analytical formulas for EuropeanOption)
        greeks = compute_greeks(call, state)

        # Validate against known Black-Scholes values
        # S=150, K=150, T=1, r=0.05, σ=0.2 (ATM call)
        # d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T) = (0 + 0.07) / 0.2 = 0.35
        # d2 = d1 - σ√T = 0.35 - 0.2 = 0.15

        # Delta = N(d1)
        @test greeks.delta ≈ 0.6368306511756191 atol=1e-10

        # Gamma = n(d1) / (S * σ * √T)
        @test greeks.gamma ≈ 0.012508011563897931 atol=1e-10

        # Vega = S * n(d1) * √T * 0.01
        @test greeks.vega ≈ 0.5628605203754069 atol=1e-10

        # Theta = -S*n(d1)*σ/(2√T) - r*K*exp(-rT)*N(d2)
        @test greeks.theta ≈ -9.621041319657294 atol=1e-8

        # Rho = K*T*exp(-rT)*N(d2) * 0.01
        @test greeks.rho ≈ 0.798487223180645 atol=1e-10

        # Sanity checks
        @test 0.0 < greeks.delta < 1.0  # Call delta bounded
        @test greeks.gamma > 0.0         # Gamma always positive
        @test greeks.vega > 0.0          # Long options have positive vega
        @test greeks.theta < 0.0         # Time decay for long options
        @test greeks.rho > 0.0           # Call rho positive

        # Put Greeks sanity check
        put = EuropeanOption("AAPL", 150.0, 1.0, :put)
        put_greeks = compute_greeks(put, state)

        @test -1.0 < put_greeks.delta < 0.0  # Put delta negative
        @test put_greeks.gamma > 0.0          # Same gamma as call
        @test put_greeks.gamma ≈ greeks.gamma atol=1e-12  # Gamma same for call/put
        @test put_greeks.vega ≈ greeks.vega atol=1e-12    # Vega same for call/put
        @test put_greeks.rho < 0.0            # Put rho negative
    end
end
