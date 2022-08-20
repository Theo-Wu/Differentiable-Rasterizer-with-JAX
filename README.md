# Differentiable-rendering-with-JAX

## Optimizing vertex positions (blurriness scale = $10^{-1.5}$)

| Original and target image |  Optimization |  Loss history |
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text](fig/triangle_mesh/triangle_mesh.png)  |  ![alt-text](fig/triangle_mesh/triangle-gaussian-mesh.gif)  |  ![alt-text](fig/triangle_mesh/triangle_mesh_loss.png)
![alt-text](fig/cube_mesh/cube_mesh.png)  |  ![alt-text](fig/cube_mesh/cube-gaussian-mesh.gif)  |  ![alt-text](fig/cube_mesh/cube_mesh_loss.png)

## Optimizing camera position (blurriness scale = $10^{-3}$)

| Original and target image |  Optimization |  Loss history |
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text](fig/cube_camera/cube_camera.png)  |  ![alt-text](fig/cube_camera/cube-gaussian-1e-3.gif)  |  ![alt-text](fig/cube_camera/cube_camera_loss.png)