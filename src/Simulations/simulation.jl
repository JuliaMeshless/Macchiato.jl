struct Simulation{D, S}
    domains::D
    stopping_criteria::S
    function Simulation(domain)
        return new{D, S}(domain)
    end
end
