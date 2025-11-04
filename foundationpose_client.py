#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import rospy, socket, json, struct, time
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters

server_ip = '192.168.0.2'
server_port = 819
FRAME_BYTES = 4 + 16*8  # 132

class FoundationPose_ClientNode(object):
    def __init__(self):
        self.color_topic = "/rgb/image_raw"
        self.depth_topic = "/depth_to_rgb/image_raw"
        self.sync_slop = 0.03  # 30ms

        print("Connecting to server at {}:{} ...".format(server_ip, server_port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((server_ip, server_port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Connected to server.")

        self.bridge = CvBridge()
        c_sub = message_filters.Subscriber(self.color_topic, Image, queue_size=5)
        d_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=5)
        ats = message_filters.ApproximateTimeSynchronizer([c_sub, d_sub], queue_size=5, slop=self.sync_slop)
        ats.registerCallback(self.callback)

        self.seq = 0
        rospy.on_shutdown(self.cleanup)

    def cleanup(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass

    def callback(self, color_msg, depth_msg):
        # ---- color: RGB8 raw ----
        cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")  # H×W×3, uint8
        Hc, Wc = cv_color.shape[:2]


        # depth_msg.encoding = "32FC1"
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
        # ---- depth: float32 meters raw ----
        Hd, Wd = depth.shape[:2]

        assert (Hc, Wc) == (Hd, Wd), "Color-Depth image dimension mismatch"

        # crop
        cv_color = cv_color[Hc//2:, Wc//2 - Wc//4:Wc//2 + Wc//4]
        depth = depth[Hd//2:, Wd//2 - Wd//4:Wc//2 + Wd//4]
        H, W = cv_color.shape[:2]
        color_bytes = cv_color.tobytes(order="C")
        depth_bytes = depth.tobytes(order="C")  # float32 raw
        

        # ---- header ----
        self.seq += 1
        header = {
            "seq": self.seq,
            "t": color_msg.header.stamp.to_sec() if color_msg.header.stamp else time.time(),
            "frame_id": color_msg.header.frame_id,
            "color": {"w": W, "h": H, "enc": "rgb8",  "dtype": "uint8",   "size": len(color_bytes)},
            "depth": {"w": W, "h": H, "enc": "f32m",  "dtype": "float32", "size": len(depth_bytes)}
        }
        hbin = json.dumps(header, separators=(',', ':')).encode("utf-8")
        packet = struct.pack("!I", len(hbin)) + hbin + color_bytes + depth_bytes

        # ---- send ----
        try:
            self.sock.sendall(packet)
        except Exception as e:
            rospy.logerr("socket send error: %s", str(e))
            rospy.signal_shutdown("socket error")
            return

        # ---- pose 수신(있을 때만) ----
        pkt = recv_exact_with_timeout(self.sock, FRAME_BYTES, timeout_s=0.002)
        if pkt is not None:
            seq = struct.unpack('!I', pkt[:4])[0]
            pose = np.frombuffer(pkt[4:], dtype='>f8').reshape(4,4).astype(np.float64)
            rospy.loginfo("pose seq %d\n%s", seq, pose)


def recv_exact_with_timeout(sock, n, timeout_s=0.002):
    prev_to = sock.gettimeout()
    sock.settimeout(timeout_s)
    try:
        buf = b''
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return buf
    except socket.timeout:
        return None
    finally:
        sock.settimeout(prev_to)


if __name__ == '__main__':
    print("Starting FoundationPose Client Node...")
    rospy.init_node('foundation_pose_client')
    node = FoundationPose_ClientNode()
    rospy.spin()