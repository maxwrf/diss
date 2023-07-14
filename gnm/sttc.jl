using Statistics

function run_T(n::Int, dt::Float64, start::Float64, stop::Float64, spike_times_1::Vector{Float64})
    time_A = 2 * n * dt

    if n == 1
        if (spike_times_1[1] - start) < dt
            time_A -= start - spike_times_1[1] + dt
        elseif (spike_times_1[1] + dt) > stop
            time_A -= spike_times_1[1] + dt - stop
        end
    else
        i = 1
        while i < n
            diff = spike_times_1[i+1] - spike_times_1[i]
            if diff < 2 * dt
                time_A -= 2 * dt - diff
            end
            i += 1
        end

        if (spike_times_1[1] - start) < dt
            time_A -= start - spike_times_1[1] + dt
        end
        if (stop - spike_times_1[n]) < dt
            time_A -= spike_times_1[n] + dt - stop
        end
    end

    return time_A
end


function run_P(na::Int, nb::Int, dt::Float64, sta_data::Vector{Float64}, stb_data::Vector{Float64})
    N12 = 0
    j = 1
    for i = 1:na
        while j <= nb
            if abs(sta_data[i] - stb_data[j]) <= dt
                N12 += 1
                break
            elseif stb_data[j] > sta_data[i]
                break
            else
                j += 1
            end
        end
    end
    return N12
end


function sttc(
    spike_times_1::Array{Float64,1},
    spike_times_2::Array{Float64,1},
    n1::Int,
    n2::Int,
    dt::Float64,
    rec_time::Array{Float64,1}
)::Float64
    """
    Warning: Spike times 1 & 2 need to be sorted smallest to largest
    """
    T = rec_time[2] - rec_time[1]
    TA = run_T(n1, dt, rec_time[1], rec_time[2], spike_times_1) / T
    TB = run_T(n2, dt, rec_time[1], rec_time[2], spike_times_2) / T
    PA = run_P(n1, n2, dt, spike_times_1, spike_times_2) / n1
    PB = run_P(n2, n1, dt, spike_times_2, spike_times_1) / n2
    return 0.5 * (PA - TB) / (1 - TB * PA) + 0.5 * (PB - TA) / (1 - TA * PB)
end

function sttc_tiling(
    dt::Float64,
    rec_time::Vector{Float64},
    spikes::Vector{Float64},
    spike_counts::Vector{Int32}
)::Matrix{Float64}
    """
    Computes pairwise sttc for all spikes in the Vector
    """
    n_electrodes = length(spike_counts)
    st_cumsum = [0; cumsum(spike_counts)]
    result = zeros(n_electrodes, n_electrodes)

    for i in 1:n_electrodes
        n1 = st_cumsum[i+1] - st_cumsum[i]
        st1 = spikes[(st_cumsum[i]+1):(st_cumsum[i+1])]
        for j in i:n_electrodes
            n2 = st_cumsum[j+1] - st_cumsum[j]
            st2 = spikes[(st_cumsum[j]+1):(st_cumsum[j+1])]
            result[i, j] = result[j, i] = sttc(st1, st2, n1, n2, dt, rec_time)
        end
    end

    return result
end

