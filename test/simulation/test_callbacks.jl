using Test
using Macchiato

@testset "Callbacks" begin
    @testset "Schedule constructors" begin
        @testset "IterationInterval" begin
            s = IterationInterval(10)
            @test s.N == 10
            @test_throws ArgumentError IterationInterval(0)
            @test_throws ArgumentError IterationInterval(-1)
        end

        @testset "TimeInterval" begin
            s = TimeInterval(0.5)
            @test s.Δt == 0.5
            s2 = TimeInterval(1)
            @test s2.Δt == 1.0
            @test_throws ArgumentError TimeInterval(0.0)
            @test_throws ArgumentError TimeInterval(-0.1)
        end

        @testset "WallTimeInterval" begin
            s = WallTimeInterval(60.0)
            @test s.seconds == 60.0
            s2 = WallTimeInterval(30)
            @test s2.seconds == 30.0
            @test_throws ArgumentError WallTimeInterval(0.0)
            @test_throws ArgumentError WallTimeInterval(-1.0)
        end

        @testset "SpecifiedTimes" begin
            s = SpecifiedTimes([0.0, 0.5, 1.0])
            @test s.times == [0.0, 0.5, 1.0]
            s2 = SpecifiedTimes(0.0, 1.0, 2.0)
            @test s2.times == [0.0, 1.0, 2.0]
            @test_throws ArgumentError SpecifiedTimes(Float64[])
            @test_throws ArgumentError SpecifiedTimes([1.0, 0.5, 2.0])
        end
    end

    @testset "should_execute" begin
        @testset "IterationInterval" begin
            schedule = IterationInterval(5)
            state = Macchiato.ScheduleState()

            @test !Macchiato.should_execute(schedule, state, 1, 0.0)
            @test !Macchiato.should_execute(schedule, state, 4, 0.0)
            @test Macchiato.should_execute(schedule, state, 5, 0.0)
            @test state.last_iteration == 5

            @test !Macchiato.should_execute(schedule, state, 6, 0.0)
            @test !Macchiato.should_execute(schedule, state, 9, 0.0)
            @test Macchiato.should_execute(schedule, state, 10, 0.0)
            @test state.last_iteration == 10
        end

        @testset "TimeInterval" begin
            schedule = TimeInterval(0.1)
            state = Macchiato.ScheduleState()

            @test !Macchiato.should_execute(schedule, state, 1, 0.05)
            @test Macchiato.should_execute(schedule, state, 2, 0.1)
            @test state.last_time == 0.1

            @test !Macchiato.should_execute(schedule, state, 3, 0.15)
            @test Macchiato.should_execute(schedule, state, 4, 0.2)
            @test state.last_time == 0.2
        end

        @testset "SpecifiedTimes" begin
            schedule = SpecifiedTimes([0.0, 0.5, 1.0])
            state = Macchiato.ScheduleState()

            @test Macchiato.should_execute(schedule, state, 0, 0.0)
            @test state.next_specified_idx == 2

            @test !Macchiato.should_execute(schedule, state, 1, 0.3)
            @test Macchiato.should_execute(schedule, state, 2, 0.5)
            @test state.next_specified_idx == 3

            @test Macchiato.should_execute(schedule, state, 3, 1.0)
            @test state.next_specified_idx == 4

            @test !Macchiato.should_execute(schedule, state, 4, 1.5)
        end
    end

    @testset "Callback construction and execution" begin
        call_count = Ref(0)
        function test_func(sim)
            call_count[] += 1
        end

        cb = Callback(test_func, IterationInterval(2))
        @test cb.func === test_func
        @test cb.schedule isa IterationInterval
        @test cb.parameters === nothing

        cb_with_params = Callback(test_func, IterationInterval(2); parameters=(a=1, b=2))
        @test cb_with_params.parameters == (a=1, b=2)

        Macchiato.reset!(cb)
        @test cb._state.last_iteration == 0
        @test cb._state.last_time == 0.0
        @test cb._state.next_specified_idx == 1
    end

    @testset "show methods" begin
        @test occursin("IterationInterval(10)", string(IterationInterval(10)))
        @test occursin("TimeInterval(0.5)", string(TimeInterval(0.5)))
        @test occursin("WallTimeInterval", string(WallTimeInterval(60.0)))
        @test occursin("SpecifiedTimes", string(SpecifiedTimes([0.0, 1.0])))

        cb = Callback(identity, IterationInterval(10))
        @test occursin("Callback", string(cb))
    end
end
