import cv2
import open3d as o3d
import numpy as np

# Load 3D car model
mesh = o3d.io.read_triangle_mesh("car_model.obj")
mesh.compute_vertex_normals()

# Load video
video_path = "your_video.mov"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = cv2.VideoWriter('rendered_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Dummy camera intrinsics (adjust as needed)
camera_matrix = np.array([[800, 0, width/2],
                          [0, 800, height/2],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

# Placeholder 3D keypoints on the car model (choose corners or wheels, etc.)
object_points = np.array([
    [-1, -1,  0],
    [ 1, -1,  0],
    [ 1,  1,  0],
    [-1,  1,  0]
], dtype=np.float32)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # OPTIONAL: manually define 2D points in the frame or use detection
    image_points = np.array([
        [width*0.3, height*0.6],
        [width*0.7, height*0.6],
        [width*0.7, height*0.8],
        [width*0.3, height*0.8]
    ], dtype=np.float32)

    # Estimate pose
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if success:
        R, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = tvec.flatten()
        transformed_mesh = mesh.transform(transform.copy())

        # Render the transformed mesh to an image (Open3D rendering)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        vis.add_geometry(transformed_mesh)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        rendered_img = (np.asarray(image) * 255).astype(np.uint8)

        # Blend with original frame (or replace)
        blended = cv2.addWeighted(frame, 0.5, rendered_img, 0.5, 0)

        output.write(blended)
    else:
        output.write(frame)  # fallback to original

    frame_count += 1

cap.release()
output.release()
print("Rendering complete. Output saved as rendered_output.mp4")
