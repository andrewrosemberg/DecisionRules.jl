# scaling layers for proxy models


struct InputScaler{T}
    l::T
    u::T
end

function (s::InputScaler)(x)
    x .- s.l ./ (s.u - s.l)
end

Flux.@layer InputScaler trainable=()
Base.show(io::IO, ::InputScaler) = print(io, "InputScaler(l, u)")



struct OutputScaler{T}
    l::T
    u::T
end

function (s::OutputScaler)(x)
    # sigmoid(x) .* (s.u - s.l) .+ s.l
    softplus(x .- s.l) .- softplus(x .- s.u) .+ s.l
end

Flux.@layer OutputScaler trainable=()
Base.show(io::IO, ::OutputScaler) = print(io, "OutputScaler(l, u)")
