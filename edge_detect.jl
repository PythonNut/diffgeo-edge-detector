using ImageFiltering
using TestImages
using ImageView
using SymPy
using Base.Cartesian

# Parameters
gamma = 1/2
scales = exp.(linspace(0,log(50),40))

# Load the image
img = float.(Colors.Gray.(testimage("mandrill")))

# Define derivative convolution matrices
Dy = Array(parent(Kernel.ando5()[1]))
Dx = Array(parent(Kernel.ando5()[2]))

function combine_kernels(kers...)
    return reduce(conv2, kers)
end

function convolve_image(I, kers...)
    kernel = combine_kernels(kers...)
    return imfilter(I, reflect(centered(kernel)))
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

Lxxxx = convolve_scale_space(Lxxx, Dx)
# Lxxxy = convolve_scale_space(Lxxx, Dy)
Lxxyy = convolve_scale_space(Lxxy, Dy)
# Lxyyy = convolve_scale_space(Lxyy, Dy)
Lyyyy = convolve_scale_space(Lyyy, Dy)

t = Sym("t")
E = t^gamma
Et, Ett = diff(E, t), diff(E, t, t)

Lvv = Lx.^2.*Lxx + 2Lx.*Ly.*Lxy + Ly.^2.*Lyy
Lvvv = Lx.^3.*Lxxx + 3Lx.^2.*Ly.*Lxxy + 3Lx.*Ly.^2.*Lxyy + Ly.^3.*Lyyy

ELt, ELtt = zeros(L), zeros(L)
for (i, scale) in enumerate(scales)
    ELt[:,:,i] = float(Et(scale)) * L[:,:,i] + (Lxx[:,:,i] + Lyy[:,:,i])/2 * float(E(scale))
    ELtt[:,:,i] = float(Ett(scale)) * L[:,:,i] + float(Et(scale))*(Lxx[:,:,i] + Lyy[:,:,i]) + (Lxxxx[:,:,i] + 2Lxxyy[:,:,i] + Lyyyy[:,:,i])/4 * float(E(scale))
end

function linear_interpolate(p1, p2, v1, v2)
    return (abs(v1)*v1 + abs(v2)*v2)/(abs(v1) + abs(v2))
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

cube_edges = [
    # bottom edges
    ([1,1,1], [1,2,1]),
    ([1,2,1], [2,2,1]),
    ([2,2,1], [2,1,1]),
    ([2,1,1], [1,1,1]),

    # side edges
    ([1,1,1], [1,1,2]),
    ([1,2,1], [1,2,2]),
    ([2,2,1], [2,2,2]),
    ([2,1,1], [2,1,2]),

    # top edges
    ([1,1,2], [1,2,2]),
    ([1,2,2], [2,2,2]),
    ([2,2,2], [2,1,2]),
    ([2,1,2], [1,1,2])
]

cube_faces =  [
    ([ 0, 0,-1], ([1,1,1], [2,1,1], [1,2,1], [2,2,1])),
    ([ 0,-1, 0], ([1,1,1], [2,1,1], [1,1,2], [2,1,2])),
    ([ 1, 0, 0], ([2,1,1], [2,1,2], [2,2,2], [2,2,1])),
    ([-1, 0, 0], ([1,1,1], [1,1,2], [1,2,1], [1,2,2])),
    ([ 0, 1, 0], ([1,2,1], [2,2,1], [1,2,2], [2,2,2])),
    ([ 0, 0, 1], ([1,1,2], [1,2,2], [2,2,2], [2,1,2]))
]

function marching_cubes(x, y, t, visited)
    if visited[x, y, t]
        return Set()
    end
    visited[x, y, t] = true
    const corners = (x:x+1, y:y+1, t:t+1)

    # Note: Maybe they don't need to be in the same corner
    if !any((view(Lvvv, corners...) .< 0) .& (view(ELtt, corners...) .< 0))
        return Set()
    end

    Z1 = view(Lvv, corners...)
    Z2 = view(ELt, corners...)
    Z1_crossings = []
    Z2_crossings = []

    # Find all sign crossings w/ linear interpolation
    for (a, b) in cube_edges
        if sign(Z1[a...]) != sign(Z1[b...])
            push!(Z1_crossings, (a, b, linear_interpolate(a, b, Z1[a...], Z1[b...])))
        end

        if sign(Z2[a...]) != sign(Z2[b...])
            push!(Z2_crossings, (a, b, linear_interpolate(a, b, Z2[a...], Z2[b...])))
        end
    end

    const epsilon = 0.01

    result = Set()
    for (normal, face) in cube_faces
        Z1_zeros = []
        Z2_zeros = []
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
        intersect = segment_intersect(Z1_crossings..., Z2_crossings..., epsilon)
        if isnull(intersect)
            continue
        end

        # Check that the intersection lies on a face
        distance, midpoint = intersect

        # if !all(-eps < x < eps || 1-eps < x < 1+eps for x in midpoint)
        #     return
        # end

        if distance > epsilon
            continue
        end

        xprime, yprime, tprime = [x, y, t] + normal
        if !(1 <= xprime <= size(visited)[1] &&
             1 <= yprime <= size(visited)[2] &&
             1 <= tprime <= size(visited)[3])
            continue
        end

        result = union(result, marching_cubes(xprime, yprime, tprime, visited))
    end

    if length(result) > 0
        push!(result, (x, y, t))
    end

    return result
end

S12 = (Lvvv .< 0) .& (ELtt .< 0)

function marching_cubes2(x, y, t, visited)
    # Faster but dumber version?
    # Still too slow...
    if !(1 <= x <= size(visited)[1] &&
         1 <= y <= size(visited)[2] &&
         1 <= t <= size(visited)[3])
        return Set()
    end

    if visited[x, y, t]
        return Set()
    end

    visited[x, y, t] = true
    const corners = (x:x+1, y:y+1, t:t+1)

    # Note: Maybe they don't need to be in the same corner
    if !any(S12[corners...] .< 0)
        return Set()
    end

    Z1 = Lvv[corners...]
    Z2 = ELt[corners...]

    if all(Z1 .< 0) || all(Z1 .> 0) || all(Z2 .< 0) || all(Z2 .> 0)
        return Set()
    end

    result = Set([(x, y, t)])
    result = union(result, marching_cubes2(x-1, y, t, visited))
    result = union(result, marching_cubes2(x+1, y, t, visited))
    result = union(result, marching_cubes2(x, y-1, t, visited))
    result = union(result, marching_cubes2(x, y+1, t, visited))
    result = union(result, marching_cubes2(x, y, t-1, visited))
    result = union(result, marching_cubes2(x, y, t+1, visited))
    return result
end


function find_edges()
    voxel_visited = falses((x->x-1).(size(L)))
    edges = []
    @nloops 3 i voxel_visited begin
        edge = marching_cubes((@ntuple 3 i)..., voxel_visited)
        if length(edge) > 0
            push!(edges, edge)
        end
        if i_1 == 1
            println(@ntuple 3 i)
        end
    end
    return edges
end
