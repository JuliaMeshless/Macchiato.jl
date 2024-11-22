abstract type Time <:AbstractModel end

struct Steady{T} <: Time
    max_time::T
end

struct Unsteady{T} <: Time
    max_time::T
end
