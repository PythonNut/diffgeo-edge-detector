using ImageFiltering
using TestImages
using ImageView
using Base.Cartesian
using ProgressMeter
using FileIO
using SpecialFunctions
using FastConv # Pkg.clone("https://github.com/aamini/FastConv.jl")

function combine_kernels(kers...)
    return reduce(fastconv, kers)
end

function convolve_image(I, kers...)
    kernel = combine_kernels(kers...)
    return imfilter(I, centered(kernel))
end

function gaussian_kernel(t, length)
    G = hcat(besselix.(-length:length, t))
    return combine_kernels(G, G')
end

function convolve_gaussian(img, t)
    # The dimension of the convolution matrix
    length = 4*ceil(Int, sqrt(t))
    kernel = gaussian_kernel(t, length)
    return convolve_image(img, kernel)
end

# Load the image
img = float.(ColorTypes.Gray.(load("Images/block.jpg")))

scales = exp.(linspace(log(0.1), log(256), 40))
L = cat(3, (convolve_gaussian(img, t) for t in scales)...)

# Define derivative convolution matrices
Dy = Array(parent(Kernel.ando5()[1]))
Dx = Array(parent(Kernel.ando5()[2]))

Dx /= sum(Dx .* (Dx .> 0))
Dy /= sum(Dy .* (Dy .> 0))

function convolve_scale_space(L, kers...)
    return mapslices(
        scale_slice -> convolve_image(scale_slice, kers...),
        L,
        (1,2)
    )
end

function spatial_derivative(L, x, y)
    return convolve_scale_space(L, fill(Dx, x)..., fill(Dy, y)...)
end

Lx = spatial_derivative(L, 1, 0)
Ly = spatial_derivative(L, 0, 1)

Lxx = spatial_derivative(L, 2, 0)
Lxy = spatial_derivative(L, 1, 1)
Lyy = spatial_derivative(L, 0, 2)

Lxxx = spatial_derivative(L, 3, 0)
Lxxy = spatial_derivative(L, 2, 1)
Lxyy = spatial_derivative(L, 1, 2)
Lyyy = spatial_derivative(L, 0, 3)

Lxxxxx = spatial_derivative(L, 5, 0)
Lxxxxy = spatial_derivative(L, 4, 1)
Lxxxyy = spatial_derivative(L, 3, 2)
Lxxyyy = spatial_derivative(L, 2, 3)
Lxyyyy = spatial_derivative(L, 1, 4)
Lyyyyy = spatial_derivative(L, 0, 5)

const Lvv = @. Lx^2*Lxx + 2Lx*Ly*Lxy + Ly^2*Lyy
const Lvvv = @. (Lx^3*Lxxx + 3Lx^2*Ly*Lxxy + 3Lx*Ly^2*Lxyy + Ly^3*Lyyy) < 0

# Shape the scales vector to be a vector with depth
scales3 = reshape(scales, 1, 1, length(scales))

# Gamma defines the scale bias, larger numbers = more bias towards diffuse edges.
# The value 1 represents no bias.
gamma = 1

# Definition of the gradient edge strength (magnitude)
const GL = scales3.^(gamma).*(Lx.^2+Ly.^2)

# Derivative of edge strength gradinet with respect to scale
const GLt = @. gamma*scales3^(gamma-1)*(Lx^2 + Ly^2) + scales3^gamma*(Lx*(Lxxx + Lxyy) + Ly*(Lxxy + Lyyy))

# Second derivative of edge strength gradinet with respect to scale
const GLtt = @. (gamma*(gamma - 1)*scales3^(gamma - 2)*(Lx^2 + Ly^2) + 2gamma*scales3^(gamma-1)*(Lx*(Lxxx + Lxyy) + Ly*(Lxxy + Lyyy)) + scales3^gamma/2*((Lxxx + Lxyy)^2 + (Lxxy + Lyyy)^2 + Lx*(Lxxxxx + 2Lxxxyy + Lxyyyy) + Ly*(Lxxxxy + 2Lxxyyy + Lyyyyy))) < 0

function linear_interpolate(p1, p2, v1, v2)
    return (abs(v1)*collect(p1) + abs(v2)*collect(p2))/(abs(v1) + abs(v2))
end

function line_distance(p1, p2, p3, p4, e)
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
        distance_check = line_distance(Z1_zeros..., Z2_zeros..., epsilon)
        if isnull(distance_check)
            continue
        end

        # Check that the intersection lies on a face
        const epsilon = 10 * eps()
        distance, midpoint = get(distance_check)
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

function flatten_edges(edges, L)
    edge_map = falses((x->x-1).(size(L)))
    for edge in edges
        for (x, y, t) in edge
            edge_map[x,y,t] = true
        end
    end
    return edge_map
end

function spatial_zeros(Lp)
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

function flatten_scale(Lp, func=any)
    return mapslices(func, Lp, 3)
end

function main(fname="output.png")
    edges = find_edges()
    n_strongest = n_strongest_edges(edges, 500)
    edge_flat = flatten_edges(n_strongest, L)
    save(fname, flatten_scale(edge_flat))
end

clamp_signed(L) = clamp_signed(L, L)
function clamp_signed(slice, L)
    min_val, max_val = minimum(L), maximum(L)
    return @. (slice - min_val)/(max_val - min_val)
end

clamp_slices(L) = mapslices(clamp_signed, L, (1,2))

function export_at_scales(L, scale_list, fname)
    for s in scale_list
        save((@eval @sprintf($fname, $(scales[s]))), L[:,:,s])
    end
end
