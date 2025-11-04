# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import socket
import struct
import json
import time
from estimater import *
from datareader import *
import argparse

def recv_exact(s,n):
    b=b''
    while len(b)<n:
        ch=s.recv(n-len(b))
        if not ch: raise ConnectionError
        b+=ch
    return b

def recv_packet(conn):
    hlen = struct.unpack("!I", recv_exact(conn,4))[0]
    hdr  = json.loads(recv_exact(conn, hlen).decode("utf-8"))
    csz, dsz = hdr["color"]["size"], hdr["depth"]["size"]
    color_raw = recv_exact(conn, csz)
    depth_raw = recv_exact(conn, dsz)
    return hdr, color_raw, depth_raw

def send_pose(sock, pose_4x4, seq):
    pose64 = np.asarray(pose_4x4, dtype=np.float64).reshape(16)
    pkt = struct.pack('!I16d', int(seq), *pose64)
    sock.sendall(pkt)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/mesh_files/box/textured.obj') # change to your mesh file path
  parser.add_argument('--est_refine_iter', type=int, default=20)
  parser.add_argument('--track_refine_iter', type=int, default=16)
  parser.add_argument('--debug', type=int, default=2)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  # Box scaling
  c_x, c_y, c_z = 47/60, 155/158, 190/210  # From paper
#   c_x, c_y, c_z = 47/70.22891, 155/167.03523, 190/237.82831 # From mesh
  S = np.diag([c_x, c_y, c_z, 1.0]) 
  mesh.apply_transform(S)
  # Cylinder scaling
#   c_x, c_y, c_z = 68/75, 68/75, 205/250  # From paper
#   c_x, c_y, c_z = 68/75.28108, 68/77.85001, 205/245.5281624548645 # From mesh
#   S = np.diag([c_x, c_y, c_z, 1.0]) 
#   mesh.apply_transform(S)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')


  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")
  to_origin, extents = trimesh.bounds.oriented_bounds(est.mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  pointcloud_init = np.asarray(trimesh.sample.sample_surface(est.mesh, 512)[0])

  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
  server.bind(('0.0.0.0', 819))
  server.listen(1)
  print('Waiting for connection...')
  conn, addr = server.accept()
  print(f'Connected to {addr}')

  # Camera matrix
  K = np.array([[609.7949829101562, 0.0, 320.93017578125],
                [0.0, 609.4755859375, 8.19635009765625],
                [0.0, 0.0, 1.0]], dtype=np.float64)
  # Transformation from robot base to camera frame (예시 값, 실제 값으로 교체 필요)
  T_robot2cam = np.array([[ 0.07788345,  0.55504573, -0.82816569,  1.08868188],
       [ 0.99696153, -0.0444955 ,  0.06393618, -0.08193968],
       [-0.00136216, -0.8306289 , -0.55682471,  0.87013487],
       [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float64)
  # Pointcloud 저장 경로
  pointcloud_save_dir = "pointclouds"
  os.makedirs(pointcloud_save_dir, exist_ok=True)

  initialized = False
  logging.disable(logging.INFO)   # 디버깅을 위해 로깅 전부 비활성화 (WARNING 이상만 출력)

  while True:
    hdr, color_raw, depth_raw = recv_packet(conn)
    W, H = hdr["color"]["w"], hdr["color"]["h"]
    color = np.frombuffer(color_raw, dtype=np.uint8).reshape(H, W, 3).copy()          # RGB uint8
    depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(H, W).copy().astype(np.float64)       # meters float64

    invalid = (np.isnan(depth)) | (~np.isfinite(depth))
    depth[invalid] = 0.0
    if initialized:
        print("Depth stats: min %.4f max %.4f mean %.4f median %.4f valid_ratio %.4f" %
          (depth[~invalid].min(), depth[~invalid].max(), depth[~invalid].mean(), np.median(depth[~invalid]), np.sum(~invalid)/(H*W)))
        print(f'frame {hdr["seq"]} received: color {color.shape} {color.dtype}, depth {depth.shape} {depth.dtype}, time elapsed {time.time()-prev_time:.4f} sec')
    prev_time = time.time()
    # === 초기화 vs 추적 ===
    if not initialized:
        ob_mask = np.zeros_like(depth, dtype=bool)
        ob_mask[H//4:H*3//4, W//4:W*3//4] = True  # 초기 프레임에서는 중앙 영역을 객체 영역으로 사용
        try:
            pose = est.register(K=K, rgb=color, depth=depth,
                                ob_mask=ob_mask, iteration=args.est_refine_iter)
            initialized = True
            print("register() done and tracking starts")
        except Exception as e:
            print(f"register failed: {e}")
            continue  # 다음 프레임으로
    else:
        pose = est.track_one(rgb=color, depth=depth, K=K,
                                iteration=args.track_refine_iter)
        T_robot_2ob = T_robot2cam @ pose
        send_pose(conn, T_robot_2ob, hdr["seq"])
        pointcloud = pointcloud_init @ T_robot_2ob[:3, :3].T + T_robot_2ob[:3, 3]
        np.save(os.path.join(pointcloud_save_dir, "pointcloud_latest.npy"), pointcloud)
        np.save(os.path.join(pointcloud_save_dir, "pose_latest.npy"), T_robot_2ob)

        if debug>=1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.waitKey(1)
        if debug>=2:
            # Save pose visualization images
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{hdr["seq"]}.png', vis)
            # Save depth images
            os.makedirs(f'{debug_dir}/depth', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/depth/{hdr["seq"]}.png', (depth/depth.max()*255).astype(np.uint8))
            # Save object point cloud in robot frame
            os.makedirs(f'{debug_dir}/point', exist_ok=True)
            plt.figure(figsize=(8, 6))
            ax = plt.axes(projection='3d')
            ax.scatter3D(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], s=1, c='b', alpha=0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Point Cloud Visualization')
            plt.show()
            plt.savefig(f'{debug_dir}/point/{hdr["seq"]}.png')


