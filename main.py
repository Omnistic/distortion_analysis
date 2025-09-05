import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from skimage import filters, morphology, measure
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from bioio import BioImage

filepath = 'test_sample.czi'

lines_per_inch = 200
magnification = 5
pixel_size_in_um = 6.5

min_area = 4100
max_area = 4999

spacing_in_um = 1 / lines_per_inch * 25400
spacing_in_pixels = spacing_in_um / (pixel_size_in_um / magnification)

data = BioImage(filepath)
raw = data.get_image_data('YX')

threshold = filters.threshold_otsu(raw)
binary = raw > threshold

closed = morphology.binary_closing(binary, morphology.disk(10))

labeled = measure.label(closed)
props = measure.regionprops(labeled)
areas = [prop.area for prop in props]

filtered_binary = np.zeros_like(binary, dtype=bool)
centroids = []
for prop in props:
    if min_area <= prop.area <= max_area:
        filtered_binary[labeled == prop.label] = True
        centroids.append(prop.centroid)
centroids = np.array(centroids)

sample_center_guess = np.mean(centroids, axis=0)
dist_centroids_to_center = np.linalg.norm(centroids - sample_center_guess, axis=1)
max_dist = np.max(dist_centroids_to_center)

def create_grid(center, spacing, max_dist, angle):
    grid = []
    angle_rad = np.deg2rad(angle)
    rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad),  np.cos(angle_rad)]])
    span_x = np.arange(0, 2*max_dist+spacing, spacing) + center[1] - np.amax(np.arange(0, 2*max_dist+spacing, spacing)) / 2
    span_y = np.arange(0, 2*max_dist+spacing, spacing) + center[0] - np.amax(np.arange(0, 2*max_dist+spacing, spacing)) / 2
    if len(span_x) % 2 != 0:
        span_x = span_x - spacing / 2
        span_x = np.append(span_x, span_x[-1] + spacing)
    if len(span_y) % 2 != 0:
        span_y = span_y - spacing / 2
        span_y = np.append(span_y, span_y[-1] + spacing)
    for xx in span_x:
        for yy in span_y:
            if np.linalg.norm(np.array([yy, xx]) - center) <= max_dist:
                xx_rot, yy_rot = rot_mat @ (np.array([xx, yy]) - center) + center
                grid.append((yy_rot, xx_rot))
    grid = np.array(grid)

    dists_to_center = np.linalg.norm(grid - center, axis=1)
    center_indices = np.argsort(dists_to_center)[:4]
    grid = np.delete(grid, center_indices, axis=0)

    return grid

def objective_function(params, centroids, initial_max_dist):
    center_y, center_x, spacing, angle = params
    center = np.array([center_y, center_x])
    grid = create_grid(center, spacing, initial_max_dist, angle)
    if len(grid) == 0:
        return 1e10
    
    distances = cdist(centroids, grid)
    min_distances = np.min(distances, axis=1)
    
    dist_from_center = np.linalg.norm(centroids - center, axis=1)
    weights = np.exp(-dist_from_center / (initial_max_dist * 0.5))

    return np.sum(weights * min_distances**2)

initial_center = sample_center_guess
initial_spacing = spacing_in_pixels
initial_angle = 0

x0 = [initial_center[0], initial_center[1], initial_spacing, initial_angle]

bounds = [
   (initial_center[0] - 50, initial_center[0] + 50),
   (initial_center[1] - 50, initial_center[1] + 50),
   (initial_spacing - 10, initial_spacing + 10),
   (-45, 45)
]

result = minimize(
   objective_function,
   x0,
   args=(centroids, max_dist),
   method='L-BFGS-B',
   bounds=bounds,
   options={'maxiter': 1000}
)

opt_center_y, opt_center_x, opt_spacing, opt_angle = result.x
opt_center = np.array([opt_center_y, opt_center_x])

optimized_grid = create_grid(opt_center, opt_spacing, max_dist, opt_angle)

# fig = make_subplots(rows=3, cols=3)

# fig.add_trace(
#     go.Heatmap(z=raw, colorscale='gray'),
#     row=1, col=1
# )

# fig.add_trace(
#     go.Heatmap(z=binary.astype(int), colorscale='gray'),
#     row=1, col=2
# )

# fig.add_trace(
#     go.Heatmap(z=closed.astype(int), colorscale='gray'),
#     row=1, col=3
# )

# fig.add_trace(
#     go.Histogram(x=areas, nbinsx=100, marker_color='gray'),
#     row=2, col=1
# )

# fig.add_trace(
#     go.Heatmap(z=filtered_binary.astype(int), colorscale='gray'),
#     row=2, col=2
# )

# fig.add_trace(
#     go.Scatter(
#         x=centroids[:, 1],
#         y=centroids[:, 0],
#         mode='markers',
#         marker=dict(
#             color='gray'
#         )
#     ),
#     row=2, col=3
# )
# fig.add_trace(
#     go.Scatter(
#         x=[sample_center_guess[1]],
#         y=[sample_center_guess[0]],
#         mode='markers',
#         marker=dict(
#             symbol='x',
#             color='black'
#         )
#     ),
#     row=2, col=3
# )
# fig.add_trace(
#     go.Scatter(
#         x=optimized_grid[:, 1],
#         y=optimized_grid[:, 0],
#         mode='markers',
#         marker=dict(
#             symbol='circle-open',
#             color='#D55E00'
#         )
#     ),
#     row=2, col=3
# )

# fig.update_traces(
#     colorscale='gray',
#     showscale=False,
#     selector=dict(type='heatmap')
# )

# fig.update_layout(
#     showlegend=False,
#     xaxis=dict(
#         scaleanchor="y",
#         scaleratio=1
#     )
# )

# fig.show()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=centroids[:, 1],
        y=centroids[:, 0],
        mode='markers',
        marker=dict(
            color='rgba(0, 158, 115, 1.0)',
            size=10
        ),
        name='Grid From Image'
    )
)
fig.add_trace(
    go.Scatter(
        x=optimized_grid[:, 1],
        y=optimized_grid[:, 0],
        mode='markers',
        marker=dict(
            symbol='circle-open',
            color='rgba(213, 94, 0, 0.5)',
            size=15,
            line=dict(width=3)
        ),
        name='Distortion-Free Grid'
    )
)
fig.add_trace(
    go.Heatmap(
        z=raw,
        colorscale='gray',
        showscale=False,
        opacity=0.5,
        hoverinfo='skip'
    )
)
fig.update_layout(
    showlegend=True,
    xaxis=dict(
        scaleanchor="y",
        scaleratio=1,
        visible=False
    ),
    yaxis=dict(visible=False)
)
fig.show()