using ImageFiltering
using TestImages
using ImageView
using Base.Cartesian
using ProgressMeter
using FileIO

# Parameters
gamma = 1
scales = exp.(linspace(0, log(50), 40))

# Load the image
img = float.(Colors.Gray.(testimage("mandrill")))

# Define derivative convolution matrices
Dy = Array(parent(Kernel.ando5()[1]))
Dx = Array(parent(Kernel.ando5()[2]))

Dx /= sum(Dx .* (Dx .> 0))
Dy /= sum(Dy .* (Dy .> 0))

function combine_kernels(kers...)
    return reduce(conv2, kers)
end

function convolve_image(I, kers...)
    kernel = combine_kernels(kers...)
    return imfilter(I, centered(kernel))
end

function convolve_scale_space(L, kers...)
    return mapslices(
        scale_slice -> convolve_image(scale_slice, kers...),
        L,
        (1,2)
    )
end

function convolve_gaussian(img, sigma)
    # The dimension of the convolution matrix
    length = 8*ceil(Int, sigma) + 1
    return imfilter(img, reflect(Kernel.gaussian((sigma, sigma), (length, length))))
end

L = cat(3, (convolve_gaussian(img, sigma) for sigma in scales)...)

Lx = convolve_scale_space(L, Dx)
Ly = convolve_scale_space(L, Dy)

Lxx = convolve_scale_space(Lx, Dx)
Lxy = convolve_scale_space(Lx, Dy)
Lyy = convolve_scale_space(Ly, Dy)

Lxxx = convolve_scale_space(Lxx, Dx)
Lxxy = convolve_scale_space(Lxx, Dy)
Lxyy = convolve_scale_space(Lxy, Dy)
Lyyy = convolve_scale_space(Lyy, Dy)

# Lxxxx = convolve_scale_space(Lxxx, Dx)
# Lxxxy = convolve_scale_space(Lxxx, Dy)
# Lxxyy = convolve_scale_space(Lxxy, Dy)
# Lxyyy = convolve_scale_space(Lxyy, Dy)
# Lyyyy = convolve_scale_space(Lyyy, Dy)

Lxxxxx = convolve_scale_space(Lxxx, Dx, Dx)
Lxxxxy = convolve_scale_space(Lxxx, Dx, Dy)
Lxxxyy = convolve_scale_space(Lxxx, Dy, Dy)
Lxxyyy = convolve_scale_space(Lxxy, Dy, Dy)
Lxyyyy = convolve_scale_space(Lxyy, Dy, Dy)
Lyyyyy = convolve_scale_space(Lyyy, Dy, Dy)

const Lvv = Lx.^2.*Lxx + 2Lx.*Ly.*Lxy + Ly.^2.*Lyy
const Lvvv = (Lx.^3.*Lxxx + 3Lx.^2.*Ly.*Lxxy + 3Lx.*Ly.^2.*Lxyy + Ly.^3.*Lyyy) .< 0

scales3 = reshape(scales, 1, 1, length(scales))

const GLt = gamma.*scales3.^(gamma-1) .* (Lx.^2 + Ly.^2) + scales3.^gamma.*(Lx.*(Lxxx + Lxyy) + Ly.*(Lxxy + Lyyy))

const GLtt = (gamma.*(gamma - 1).*scales3.^(gamma - 2).*(Lx.^2 + Ly.^2) + 2gamma.*scales3.^(gamma-1).*(Lx.*(Lxxx + Lxyy) + Ly.*(Lxxy + Lyyy)) + scales3.^gamma/2.*((Lxxx + Lxyy).^2 + (Lxxy + Lyyy).^2 + Lx.*(Lxxxxx + 2Lxxxyy + Lxyyyy) + Ly.*(Lxxxxy + 2Lxxyyy + Lyyyyy))) .< 0

const GL = scales3.^(gamma).*(Lx.^2+Ly.^2)

Z12 = Lvvv .& GLtt

function linear_interpolate(p1, p2, v1, v2)
    return (abs(v1)*collect(p1) + abs(v2)*collect(p2))/(abs(v1) + abs(v2))
end

function segment_intersect(p1, p2, p3, p4, e)
    p13, p43, p21 = p1 - p3, p4 - p3, p2 - p1

    # If the line segments have zero length
    if norm(p43) < e || norm(p21) < e
        return Nullable()
    end

    d1343 = dot(p13, p43)
    d4321 = dot(p43, p21)
    d1321 = dot(p13, p21)
    d4343 = dot(p43, p43)
    d2121 = dot(p21, p21)

    numer = d1343 * d4321 - d1321 * d4343;
    denom = d2121 * d4343 - d4321 * d4321;

    if abs(denom) < e
        return Nullable()
    end

    mua = numer/denom
    mub = (d1343 + d4321 * mua) / d4343
    pa = p1 + mua * p21
    pb = p3 + mub * p43

    return Nullable((norm(pa - pb), (pa + pb)/2))
end

const cube_edges = [
    # bottom edges
    ((1,1,1), (1,2,1)),
    ((1,2,1), (2,2,1)),
    ((2,2,1), (2,1,1)),
    ((2,1,1), (1,1,1)),

    # side edges
    ((1,1,1), (1,1,2)),
    ((1,2,1), (1,2,2)),
    ((2,2,1), (2,2,2)),
    ((2,1,1), (2,1,2)),

    # top edges
    ((1,1,2), (1,2,2)),
    ((1,2,2), (2,2,2)),
    ((2,2,2), (2,1,2)),
    ((2,1,2), (1,1,2))
]

const cube_faces =  [
    ([ 0, 0,-1], ((1,1,1), (2,1,1), (1,2,1), (2,2,1))),
    ([ 0,-1, 0], ((1,1,1), (2,1,1), (1,1,2), (2,1,2))),
    ([ 1, 0, 0], ((2,1,1), (2,1,2), (2,2,2), (2,2,1))),
    ([-1, 0, 0], ((1,1,1), (1,1,2), (1,2,1), (1,2,2))),
    ([ 0, 1, 0], ((1,2,1), (2,2,1), (1,2,2), (2,2,2))),
    ([ 0, 0, 1], ((1,1,2), (1,2,2), (2,2,2), (2,1,2)))
]

function marching_cubes(x, y, t, visited)
    if visited[x, y, t]
        return Set()
    end

    visited[x, y, t] = true
    const corners = (x:x+1, y:y+1, t:t+1)

    # Note: Maybe they don't need to be in the same corner
    if !(any(view(GLtt, corners...)) && any(view(Lvvv, corners...)))
        return Set()
    end

    @views Z1, Z2 = Lvv[corners...], GLt[corners...]
    Z1_crossings = Array{Tuple{NTuple{3,Int}, NTuple{3,Int}, Array{Float64, 1}}, 1}()
    Z2_crossings = Array{Tuple{NTuple{3,Int}, NTuple{3,Int}, Array{Float64, 1}}, 1}()

    # Find all sign crossings w/ linear interpolation
    for (a, b) in cube_edges
        if signbit(Z1[a...]) != signbit(Z1[b...])
            push!(Z1_crossings, (a, b, linear_interpolate(a, b, Z1[a...], Z1[b...])))
        end

        if signbit(Z2[a...]) != signbit(Z2[b...])
            push!(Z2_crossings, (a, b, linear_interpolate(a, b, Z2[a...], Z2[b...])))
        end
    end

    const epsilon = 10 * eps()

    face_intersections = []
    result = Set()
    for (normal, face) in cube_faces
        Z1_zeros, Z2_zeros = [], []
        for (a, b, mid) in Z1_crossings
            if a in face && b in face
                push!(Z1_zeros, mid)
            end
        end

        for (a, b, mid) in Z2_crossings
            if a in face && b in face
                push!(Z2_zeros, mid)
            end
        end

        # Reject if there are more than two crossings for either invariant
        if !(length(Z1_zeros) == length(Z2_zeros) == 2)
            continue
        end

        # Check the intersection of the segments defined by the two lines
        intersect = segment_intersect(Z1_zeros..., Z2_zeros..., epsilon)
        if isnull(intersect)
            continue
        end

        # Check that the intersection lies on a face
        distance, midpoint = get(intersect)
        if distance > epsilon || !all(1 - epsilon .<= midpoint .<= 2 + epsilon)
            continue
        end

        push!(face_intersections, normal)
    end

    if length(face_intersections) == 2
        for normal in face_intersections
            next_voxel = [x, y, t] + normal
            if all(1 .<= next_voxel .<= size(visited))
                union!(result, marching_cubes(next_voxel..., visited))
            end
        end
        push!(result, (x, y, t))
    end

    return result
end

function find_edges()
    voxel_visited = falses((x->x-1).(size(L)))
    edges = []
    p = Progress(prod(size(voxel_visited)), 1)
    @nloops 3 i voxel_visited begin
        edge = marching_cubes((@ntuple 3 i)..., voxel_visited)
        if length(edge) > 0
            push!(edges, edge)
        end
        next!(p)
    end
    return edges
end

function flatten_edges(edges)
    edge_flat = reduce(union, edges)
    edge_map = falses((x->x-1).(size(L)))
    for (x, y, t) in union(edge_flat)
        edge_map[x,y,t] = true
    end
    return edge_map
end

function planar_zeros(Lp)
    Lp_pos = signbit.(Lp)
    Lp_zeros = falses(Lp)
    for (i, scale) in enumerate(scales)
        for x in 2:(size(Lp)[1]-1)
            for y in 2:(size(Lp)[2]-1)
                @views neighbors = Lp_pos[x-1:x+1, y-1:y+1, i]
                if (Lp_pos[x,y,i] && !all(neighbors)) ||
                    (!Lp_pos[x,y,i] && any(neighbors))
                    Lp_zeros[x,y,i] = true
                end
            end
        end
    end
    return Lp_zeros
end

function scale_zeros(Lp)
    Lp_pos = signbit.(Lp)
    Lp_zeros = falses(Lp)
    for i in 2:length(scales)-1
        Lp_zeros[:,:,i] = (Lp_pos[:,:,i-1] .!= Lp_pos[:,:,i]) .| (Lp_pos[:,:,i] .!= Lp_pos[:,:,i+1])
    end
    return Lp_zeros
end

function scale_maxima(Lp)
    Lp_pos = signbit.(Lp)
    Lp_zeros = falses(Lp)
    for i in 2:length(scales)
        Lp_zeros[:,:,i] = (Lp_pos[:,:,i-1] .& .!Lp_pos[:,:,i])
    end
    return Lp_zeros
end

function edge_importance(edge)
    total = 0
    for (x, y, t) in edge
        total += sqrt(GL[x,y,t])
    end
    return total
end

function n_strongest_edges(edges, n)
    return sort(edges, by=edge_importance, rev=true)[1:n]
end

function flatten_scale(Lp)
    return mapslices(any, Lp, 3)
end

function main()
    edges = find_edges()
    n_strongest = n_strongest_edges(edges, 500)
    edge_flat = flatten_edges(n_strongest)
    save("output.png", flatten_scale(edge_flat))
end
